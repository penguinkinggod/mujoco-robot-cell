"""diag_feed_trajectory.py -- 从 TCP 轨迹复算「哪里一顿一顿」。

为什么单独一个脚本
------------------
`diag_feed_chuck.py` 的 `phase_stats` 只记接触和力，没有速度；它自带的
平滑度打印用 8 mm/s 判「静止」，和 0.02 m/s 不是同一门槛，而且只打印不落盘。
所以「停点几个 / 归零几次 / 最大反向多少度 / 最短起步片段多少毫秒」在旧
产物里根本复算不出来。

本脚本只读结果 JSON（需要 `FEED_TRACE_TCP=1` 导出的 `tcp_trace`），按调用
时给的阈值重新算，并把每一处超标的位置、时间、阶段标签、前后方向列出来。
它不参与 PASS/FAIL，不改 `feed-criteria-v1`，也不改任何控制量。

用法
----
    D:\\robot-env\\Scripts\\python.exe src\\diag_feed_trajectory.py ^
        _build\\p5.8.23-diagnose\\before.json ^
        --json-out _build\\p5.8.23-diagnose\\before.traj.json

    # 多个文件并排比较（配对/消融）
    ... src\\diag_feed_trajectory.py before.json afterA.json afterAB.json

    # 改门槛
    ... --stop-threshold 0.005
"""
import argparse
import json
import os
import sys

import numpy as np

DEFAULT_STOP_THRESHOLD = 0.02   # m/s
DEFAULT_MIN_STOP_MS = 20.0      # 短于此的「低于门槛」视为穿越，不单列为一个区间
DEFAULT_MIN_SEGMENT_MS = 100.0  # 方向统计只采信这么长的运动片段
DEFAULT_MIN_SEGMENT_MM = 0.5    # 且行程至少这么长，否则方向是噪声
REVERSE_ANGLE_DEG = 90.0

# 豁免规则（用户 2026-11-12 定）：行程短于这个长度的段，不要求「高速」
# （速度 > 0.02 m/s），只要求「不停」（速度从不为 0）。理由：一段只有
# 2.76 mm 的夹紧修正，即使瞬间加速到 20 mm/s 也只能维持 138 ms，要求它
# 「全程高速」在数学上不可能，除非删掉动作本身。门槛取 20 mm：它等于
# 「0.02 m/s 跑满 1 秒」的行程，也就是低于这个长度时，任何满足
# 0.02 m/s 的运动都撑不过 1 秒，指标没有物理意义。
EXEMPT_SHORT_TRAVEL_MM = 20.0
# 「不停」的判定：单帧位移换算速度低于这个值就算速度为 0。用 1 µm/s
# 而不是浮点 0，因为 MuJoCo 的 qpos 收敛后会停在 1e-12 量级的抖动上。
STILL_EPSILON_M_S = 1e-6


def _load_trace(path):
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    trace = result.get("tcp_trace")
    if not trace or not trace.get("tcp_position_m"):
        raise SystemExit(
            "%s 没有 tcp_trace。先带 FEED_TRACE_TCP=1 重跑该 case。" % path)
    pos = np.asarray(trace["tcp_position_m"], dtype=float)
    time_s = np.asarray(trace["time_s"], dtype=float)
    labels = [str(v) for v in trace["labels"]]
    dt = float(trace.get("sample_interval_s") or 0.002)
    return result, pos, time_s, labels, dt


def _speed_series(pos, dt):
    """逐帧 TCP 速度：v[i] 用 i-1 -> i 的后向差分，首帧记 0。"""
    if pos.shape[0] < 2:
        return np.zeros(pos.shape[0])
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    return np.concatenate([[0.0], step])


def _runs(mask):
    runs = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _label_runs(labels):
    runs = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    if labels:
        runs.append((start, len(labels), labels[start]))
    return runs


def _stage_table(labels, time_s, speed, threshold, dt):
    """时间顺序的阶段表：每一段标签里，速度到底是多少。

    这是「哪里有问题」的主表。某段标签如果峰值速度都低于门槛，说明机械臂
    本来就不该动（夹爪斜坡、三爪拧紧、原地松爪、解锁静置），单独标出来，
    免得把夹爪动作误算成轨迹顿挫。
    """
    rows = []
    for start, end, name in _label_runs(labels):
        chunk = speed[start:end]
        below = chunk <= threshold
        longest = max((e - s for s, e in _runs(below)), default=0)
        rows.append({
            "label": name,
            "start_time_s": float(time_s[start]),
            "end_time_s": float(time_s[end - 1]),
            "samples": int(end - start),
            "duration_s": float((end - start) * dt),
            "peak_speed_m_s": float(chunk.max()) if chunk.size else 0.0,
            "median_speed_m_s": float(np.median(chunk)) if chunk.size else 0.0,
            "p95_speed_m_s": float(np.percentile(chunk, 95)) if chunk.size else 0.0,
            "below_threshold_samples": int(below.sum()),
            "below_threshold_fraction": float(below.mean()) if chunk.size else 0.0,
            "below_threshold_time_s": float(below.sum() * dt),
            "longest_below_run_ms": float(longest * dt * 1000.0),
            "arm_motion_expected": bool(chunk.size and chunk.max() > threshold),
        })
    return rows


def _exemption(travel_mm, exempt_mm):
    """按用户口径给每段判定它适用哪套要求。

    行程 < exempt_mm 的段：只要求「不停」（速度从不为 0），不要求高速。
    其余段：要求「不停」+「速度 > 阈值」。
    """
    return {
        "travel_mm": float(travel_mm),
        "exempt_from_speed": bool(travel_mm < exempt_mm),
        "requirement": ("不停（速度永不为 0）" if travel_mm < exempt_mm
                        else "不停 + 高速（速度 > 阈值）"),
    }


def _still_runs(speed, epsilon):
    """速度为 0 的区间：用极小 epsilon 而不是 0.02 门槛。"""
    return _runs(np.abs(speed) <= epsilon)


def _analyze_scope(name, idx, pos, time_s, labels, speed, dt,
                   threshold, min_stop_ms, min_segment_ms, min_segment_mm,
                   exempt_mm, epsilon):
    if idx.size < 3:
        return {"scope": name, "samples": int(idx.size),
                "insufficient_samples": True}
    scoped = speed[idx]

    stops = []
    for start, end in _runs(scoped <= threshold):
        duration_ms = (end - start) * dt * 1000.0
        if duration_ms < min_stop_ms:
            continue
        seg = idx[start:end]
        stops.append({
            "start_time_s": float(time_s[seg[0]]),
            "end_time_s": float(time_s[seg[-1]]),
            "duration_ms": duration_ms,
            "labels": [n for _s, _e, n in _label_runs(
                [labels[i] for i in seg])],
            "tcp_mean_m": [float(v) for v in pos[seg].mean(axis=0)],
            "tcp_travel_mm": float(np.linalg.norm(
                np.diff(pos[seg], axis=0), axis=1).sum() * 1000.0)
            if seg.size > 1 else 0.0,
        })

    angles = []
    for a, b in zip(idx[:-1], idx[1:]):
        if speed[a] <= threshold or speed[b] <= threshold:
            continue
        ua = pos[a] - pos[a - 1]
        ub = pos[b] - pos[b - 1]
        na, nb = float(np.linalg.norm(ua)), float(np.linalg.norm(ub))
        if na < 1e-12 or nb < 1e-12:
            continue
        cos = float(np.dot(ua, ub) / (na * nb))
        angles.append((float(np.degrees(np.arccos(np.clip(cos, -1, 1)))),
                       int(b), float(speed[a]), float(speed[b])))
    max_angle = max(angles, key=lambda kv: kv[0]) if angles else None

    segments = []
    for start, end in _runs(scoped > threshold):
        seg = idx[start:end]
        if seg.size < 2:
            continue
        path_len = float(np.linalg.norm(
            np.diff(pos[seg], axis=0), axis=1).sum())
        duration_ms = (seg.size - 1) * dt * 1000.0
        direction = pos[seg[-1]] - pos[seg[0]]
        norm = float(np.linalg.norm(direction))
        segments.append({
            "start_time_s": float(time_s[seg[0]]),
            "end_time_s": float(time_s[seg[-1]]),
            "duration_ms": duration_ms,
            "labels": [n for _s, _e, n in _label_runs(
                [labels[i] for i in seg])],
            "path_length_mm": path_len * 1000.0,
            "peak_speed_m_s": float(scoped[start:end].max()),
            "mean_speed_m_s": float(scoped[start:end].mean()),
            "direction_unit": ([float(v) for v in direction / norm]
                               if norm > 1e-12 else None),
            "usable_for_direction": bool(
                duration_ms >= min_segment_ms
                and path_len * 1000.0 >= min_segment_mm),
        })

    # 短行程段豁免：这些段只要「不停」就算合格，不要求高速。
    # 单独把它们的「是否真的停过」算出来，作为豁免后的实际判据。
    exempt_reports = []
    for start, end in _runs((scoped > epsilon) & (scoped <= threshold)):
        seg = idx[start:end]
        if seg.size < 2:
            continue
        travel = float(np.linalg.norm(
            np.diff(pos[seg], axis=0), axis=1).sum() * 1000.0)
        exempt_reports.append({
            "start_time_s": float(time_s[seg[0]]),
            "end_time_s": float(time_s[seg[-1]]),
            "duration_ms": (seg.size - 1) * dt * 1000.0,
            "travel_mm": travel,
            "labels": [n for _s, _e, n in _label_runs(
                [labels[i] for i in seg])],
            **_exemption(travel, exempt_mm),
        })

    # 「永不为 0」的硬检查：用 epsilon 判 0，不受 0.02 门槛影响。
    still_runs = []
    for start, end in _still_runs(scoped, epsilon):
        duration_ms = (end - start) * dt * 1000.0
        if duration_ms < min_stop_ms:
            continue
        seg = idx[start:end]
        travel = float(np.linalg.norm(
            np.diff(pos[seg], axis=0), axis=1).sum() * 1000.0) \
            if seg.size > 1 else 0.0
        still_runs.append({
            "start_time_s": float(time_s[seg[0]]),
            "end_time_s": float(time_s[seg[-1]]),
            "duration_ms": duration_ms,
            "travel_mm": travel,
            "labels": [n for _s, _e, n in _label_runs(
                [labels[i] for i in seg])],
            **_exemption(travel, exempt_mm),
        })

    usable = [s for s in segments if s["usable_for_direction"]]
    reversals = []
    for a, b in zip(usable[:-1], usable[1:]):
        ua = np.asarray(a["direction_unit"])
        ub = np.asarray(b["direction_unit"])
        angle = float(np.degrees(np.arccos(
            float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))
        reversals.append({
            "between": "%s -> %s" % ("/".join(a["labels"]),
                                     "/".join(b["labels"])),
            "angle_deg": angle,
            "at_time_s": b["start_time_s"],
            "prev_duration_ms": a["duration_ms"],
            "next_duration_ms": b["duration_ms"],
        })
    reversals.sort(key=lambda r: -r["angle_deg"])

    shortest = min(usable, key=lambda s: s["duration_ms"]) if usable else None
    return {
        "scope": name,
        "samples": int(idx.size),
        "duration_s": float(idx.size * dt),
        "stop_threshold_m_s": threshold,
        "stop_count": len(stops),
        "stop_time_s": float(sum(s["duration_ms"] for s in stops) / 1000.0),
        "stops": stops,
        "max_turn_angle_deg": (max_angle[0] if max_angle else None),
        "max_turn_at_time_s": (time_s[max_angle[1]] if max_angle else None),
        "max_turn_at_label": (labels[max_angle[1]] if max_angle else None),
        "max_turn_position_m": ([float(v) for v in pos[max_angle[1]]]
                                if max_angle else None),
        "turn_frames_analyzed": len(angles),
        "turn_above_90_count": int(
            sum(1 for value, _i, _a, _b in angles
                if value > REVERSE_ANGLE_DEG)),
        "motion_segment_count": len(segments),
        "usable_segment_count": len(usable),
        "motion_segments": segments,
        "shortest_segment": shortest,
        "segment_reversals": reversals,
        "max_segment_reversal_deg": (
            reversals[0]["angle_deg"] if reversals else None),
        "exempt_threshold_mm": exempt_mm,
        "still_epsilon_m_s": epsilon,
        "still_runs": still_runs,
        "still_run_count": len(still_runs),
        "still_time_s": float(sum(r["duration_ms"] for r in still_runs) / 1000.0),
        "exempt_low_speed_runs": exempt_reports,
        "peak_speed_m_s": float(scoped.max()),
        "median_speed_m_s": float(np.median(scoped)),
        "acceleration_rms_m_s2": float(np.sqrt(np.mean(
            (np.diff(scoped) / dt) ** 2))) if idx.size > 2 else None,
    }


def _stage_intervals(result, time_s):
    """用检查点/阶段记录把时间轴切成互不重叠的动作区间。

    为什么不直接用 tcp_trace 里的 label：`settle()` 和若干 `move3()` 沿用
    上一个阶段残留的 `state["label"]`，同一个 label 会横跨退出、静置和拉拔
    测试，导致「阶段」既不对应动作也不对应时间。检查点带真实 sim_time，
    用它分段才和「第几步在干什么」对得上。
    """
    marks = []
    for record in (result.get("checkpoints") or []) + (
            result.get("stages") or []):
        t = record.get("sim_time_s")
        name = record.get("stage")
        if t is None or not name:
            continue
        marks.append((float(t), str(name)))
    if not marks:
        return []
    marks.sort(key=lambda item: item[0])
    deduped = []
    for t, name in marks:
        if deduped and abs(t - deduped[-1][0]) < 1e-9:
            deduped[-1] = (t, deduped[-1][1] + " | " + name)
            continue
        deduped.append((t, name))
    t0, t1 = float(time_s[0]), float(time_s[-1])
    bounds = [(t0, deduped[0][1])]
    for t, name in deduped:
        if t <= t0:
            continue
        bounds.append((t, name))
    intervals = []
    for i, (start, name) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else t1 + 1e-9
        if end <= start:
            continue
        next_name = bounds[i + 1][1] if i + 1 < len(bounds) else "END"
        # 名字写成「起点标记 → 下一个标记」，因为动作发生在两个标记之间；
        # 只写起点名会让人以为这段是起点那个动作本身。
        intervals.append((start, end, "%s → %s" % (name, next_name)))
    return intervals


def _analyze_intervals(result, pos, time_s, labels, speed, dt, threshold,
                       min_stop_ms, exempt_mm, epsilon):
    """每个检查点区间的动作质量：这段到底是在动还是在停。

    每段都带上豁免判定：行程 < exempt_mm 时只要求「不停」，否则要求
    「不停 + 高速」。这样「哪里有问题」直接按用户口径读，不用二次换算。
    """
    rows = []
    for start, end, name in _stage_intervals(result, time_s):
        idx = np.flatnonzero((time_s >= start) & (time_s < end))
        if idx.size < 3:
            continue
        chunk = speed[idx]
        below = chunk <= threshold
        runs = [(e - s, s, e) for s, e in _runs(below)]
        longest = max(runs, default=(0, 0, 0))
        still = _still_runs(chunk, epsilon)
        still_ms = sum((e - s) for s, e in still) * dt * 1000.0
        displacement = pos[idx[-1]] - pos[idx[0]]
        path_len = float(np.linalg.norm(
            np.diff(pos[idx], axis=0), axis=1).sum() * 1000.0)
        rule = _exemption(path_len, exempt_mm)
        rows.append({
            "interval": name,
            "start_time_s": float(time_s[idx[0]]),
            "end_time_s": float(time_s[idx[-1]]),
            "samples": int(idx.size),
            "duration_s": float(idx.size * dt),
            "peak_speed_m_s": float(chunk.max()),
            "median_speed_m_s": float(np.median(chunk)),
            "below_threshold_time_s": float(below.sum() * dt),
            "below_threshold_fraction": float(below.mean()),
            "longest_stop_ms": float(longest[0] * dt * 1000.0),
            "longest_stop_start_s": (
                float(time_s[idx[longest[1]]]) if longest[0] else None),
            "stop_count": int(sum(1 for d, _s, _e in runs
                                  if d * dt * 1000.0 >= min_stop_ms)),
            "still_time_ms": float(still_ms),
            "still_run_count": len(still),
            "net_displacement_mm": float(np.linalg.norm(displacement) * 1000.0),
            "path_length_mm": path_len,
            "arm_motion_expected": bool(chunk.max() > threshold),
            **rule,
            "trace_labels": [n for _s, _e, n in _label_runs(
                [labels[i] for i in idx])],
        })
    return rows


ALLOWED_STOP_ROLES = (
    ("抓取点", "⑤ 夹紧｜真实接触抓取"),
    ("夹紧点", "three_jaw_clamped"),
    ("交接点(松爪)", "gripper_released_in_place"),
)


def _classify_stops(result, intervals, min_stop_ms):
    """按用户 2026-11-12 口径给停点分类：松爪算独立停点。

    三个允许停点：抓取点（V 爪闭合）、夹紧点（三爪拧紧）、交接点（松爪）。
    其余任何持续 >= min_stop_ms 的静止都记为「多余停点」。
    """
    allowed = []
    for role, marker in ALLOWED_STOP_ROLES:
        match = next((row for row in intervals
                      if marker in row["interval"]), None)
        allowed.append({
            "role": role,
            "marker": marker,
            "found": match is not None,
            "interval": match["interval"] if match else None,
            "still_time_ms": match["still_time_ms"] if match else None,
            "duration_s": match["duration_s"] if match else None,
        })
    extra = [row for row in intervals
             if row["arm_motion_expected"]
             and row["longest_stop_ms"] >= min_stop_ms]
    return {
        "allowed_stop_count": len(ALLOWED_STOP_ROLES),
        "allowed_stops": allowed,
        "extra_stop_interval_count": len(extra),
        "extra_stop_intervals": [
            {"interval": row["interval"],
             "longest_stop_ms": row["longest_stop_ms"],
             "stop_count": row["stop_count"]} for row in extra],
        "note": ("允许停点固定 3 个（抓取/夹紧/松爪）；"
                 "额外区间另计，不计入停点总数"),
    }


def analyze_run(path, threshold, min_stop_ms, min_segment_ms,
                min_segment_mm, exempt_mm=EXEMPT_SHORT_TRAVEL_MM,
                epsilon=STILL_EPSILON_M_S):
    result, pos, time_s, labels, dt = _load_trace(path)
    speed = _speed_series(pos, dt)
    stages = _stage_table(labels, time_s, speed, threshold, dt)
    static_labels = sorted({r["label"] for r in stages
                            if not r["arm_motion_expected"]})
    static_set = set(static_labels)
    in_scope = np.array([name not in static_set for name in labels],
                        dtype=bool)
    common = dict(pos=pos, time_s=time_s, labels=labels, speed=speed, dt=dt,
                  threshold=threshold, min_stop_ms=min_stop_ms,
                  min_segment_ms=min_segment_ms,
                  min_segment_mm=min_segment_mm,
                  exempt_mm=exempt_mm, epsilon=epsilon)
    intervals = _analyze_intervals(
        result, pos, time_s, labels, speed, dt, threshold, min_stop_ms,
        exempt_mm, epsilon)
    return {
        "source_json": os.path.abspath(path),
        "case_id": result.get("case_id"),
        "outcome_class": result.get("outcome_class"),
        "overall_pass": result.get("overall_pass"),
        "trace_samples": int(pos.shape[0]),
        "sample_interval_s": dt,
        "trace_span_s": float(time_s[-1] - time_s[0]) if time_s.size else 0.0,
        "static_labels": static_labels,
        "exempt_threshold_mm": exempt_mm,
        "still_epsilon_m_s": epsilon,
        "stage_table": stages,
        "checkpoint_intervals": intervals,
        "stop_classification": _classify_stops(result, intervals, min_stop_ms),
        "all_frames": _analyze_scope(
            "all_frames", np.arange(len(labels)), **common),
        "arm_motion_only": _analyze_scope(
            "arm_motion_only", np.flatnonzero(in_scope), **common),
        "feed_metrics_unchanged": {
            key: result.get("metrics", {}).get(key)
            for key in ("coax_mm", "tilt_deg", "contacts", "slip_mm",
                        "pull_mm", "clamp_swing_mm", "stock_ecc_mm")
        },
        "criterion": "diagnostic_only_no_pass_fail_threshold",
    }


def _print_report(report, threshold):
    print("=" * 92)
    print("case %s   outcome=%s   trace=%d 帧 / %.3f s"
          % (report["case_id"], report["outcome_class"],
             report["trace_samples"], report["trace_span_s"]))
    print("  机械臂本来就不动的阶段（数据判定，峰值速度 <= %.3f m/s）：%s"
          % (threshold, ", ".join(
              "`%s`" % s if s else "`<空标签>`"
              for s in report["static_labels"]) or "无"))
    print("-" * 92)
    print("阶段表（时间顺序；v 单位 m/s）")
    print("  %-34s %8s %7s %7s %7s %8s %9s"
          % ("label", "时长s", "峰值", "中位", "p95", "低于门槛", "最长连续"))
    for row in report["stage_table"]:
        print("  %-34s %8.2f %7.3f %7.3f %7.3f %7.0f%% %8.0fms"
              % (row["label"] if row["label"] else "<空标签>",
                 row["duration_s"], row["peak_speed_m_s"],
                 row["median_speed_m_s"], row["p95_speed_m_s"],
                 row["below_threshold_fraction"] * 100.0,
                 row["longest_below_run_ms"]))
    print("-" * 92)
    print("检查点区间表（时间顺序；用检查点 sim_time 切段，避免 label 跨段污染）")
    print("  豁免规则：行程 < %.0f mm 的段只要求「不停」，其余要求「不停 + 高速」"
          % report["exempt_threshold_mm"])
    print("  %-34s %7s %7s %7s %7s %8s %8s %6s"
          % ("interval", "时长s", "行程mm", "峰值", "中位", "停占比",
             "真静止", "要求"))
    for row in report["checkpoint_intervals"]:
        req = "不停" if row["exempt_from_speed"] else "不停+高速"
        flag = "" if row["arm_motion_expected"] else "  (夹爪/卡盘动作)"
        print("  %-34s %7.2f %7.1f %7.3f %7.3f %7.0f%% %7.0fms %6s%s"
              % (row["interval"][:34], row["duration_s"],
                 row["path_length_mm"], row["peak_speed_m_s"],
                 row["median_speed_m_s"],
                 row["below_threshold_fraction"] * 100.0,
                 row["still_time_ms"], req, flag))
    print("-" * 92)
    print("真静止区间（速度 <= %.0e m/s，即「停了」；豁免段也不许出现）"
          % report["still_epsilon_m_s"])
    still = report["arm_motion_only"]["still_runs"]
    if not still:
        print("  无")
    for run in still:
        print("  %7.3f-%7.3f s %7.0f ms  行程%5.2fmm  %-10s %s"
              % (run["start_time_s"], run["end_time_s"], run["duration_ms"],
                 run["travel_mm"], "/".join(run["labels"]),
                 "豁免段(只需不停)→不合格" if run["exempt_from_speed"]
                 else "非豁免段→不合格"))
    print("-" * 92)
    for scope in ("all_frames", "arm_motion_only"):
        row = report[scope]
        tag = "全程" if scope == "all_frames" else "仅机械臂运动段"
        print("[%s]  %d 帧 / %.2f s" % (tag, row["samples"], row["duration_s"]))
        print("  停点区间（v<=%.3f，>=%.0f ms）: %d 个，合计 %.2f s"
              % (threshold, report["min_stop_ms"], row["stop_count"],
                 row["stop_time_s"]))
        if row["max_turn_angle_deg"] is None:
            print("  最大相邻帧方向夹角: 无法定义（可用运动帧不足）")
        else:
            print("  最大相邻帧方向夹角: %.1f°  @%.3f s  阶段=%s"
                  % (row["max_turn_angle_deg"], row["max_turn_at_time_s"],
                     row["max_turn_at_label"]))
        print("  >90° 帧数: %d / %d 个可定义帧"
              % (row["turn_above_90_count"], row["turn_frames_analyzed"]))
        print("  运动片段: %d 个（其中 %d 个够长够远，可判方向）"
              % (row["motion_segment_count"], row["usable_segment_count"]))
        short = row["shortest_segment"]
        if short:
            print("  最短可用片段: %.0f ms  %.3f-%.3f s  阶段=%s  行程=%.1f mm"
                  % (short["duration_ms"], short["start_time_s"],
                     short["end_time_s"], "/".join(short["labels"]),
                     short["path_length_mm"]))
        if row["segment_reversals"]:
            print("  片段级方向突变 Top3:")
            for item in row["segment_reversals"][:3]:
                print("     %6.1f°  %s  @%.3f s"
                      % (item["angle_deg"], item["between"],
                         item["at_time_s"]))
        print("  峰值 %.3f  中位 %.4f  加速度RMS %.2f m/s^2"
              % (row["peak_speed_m_s"], row["median_speed_m_s"],
                 row["acceleration_rms_m_s2"] or 0.0))
    print("-" * 92)
    print("停顿明细（仅机械臂运动段）：")
    stops = report["arm_motion_only"]["stops"]
    if not stops:
        print("  无")
    for stop in stops:
        print("  %7.3f-%7.3f s %7.0f ms  行程%5.2fmm  %s"
              % (stop["start_time_s"], stop["end_time_s"],
                 stop["duration_ms"], stop["tcp_travel_mm"],
                 "/".join(stop["labels"])))
    print("=" * 92)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="从 tcp_trace 复算停点/归零/方向夹角/最短片段")
    parser.add_argument("results", nargs="+",
                        help="结果 JSON（需 FEED_TRACE_TCP=1）")
    parser.add_argument("--stop-threshold", type=float,
                        default=DEFAULT_STOP_THRESHOLD,
                        help="静止门槛，默认 0.02 m/s")
    parser.add_argument("--min-stop-ms", type=float,
                        default=DEFAULT_MIN_STOP_MS,
                        help="低于门槛持续多久才算一个区间，默认 20 ms")
    parser.add_argument("--min-segment-ms", type=float,
                        default=DEFAULT_MIN_SEGMENT_MS,
                        help="判方向的最短片段，默认 100 ms")
    parser.add_argument("--min-segment-mm", type=float,
                        default=DEFAULT_MIN_SEGMENT_MM,
                        help="判方向的最短行程，默认 0.5 mm")
    parser.add_argument("--json-out", metavar="PATH")
    args = parser.parse_args(argv)

    reports = []
    for path in args.results:
        report = analyze_run(path, args.stop_threshold, args.min_stop_ms,
                             args.min_segment_ms, args.min_segment_mm)
        report["min_stop_ms"] = args.min_stop_ms
        report["min_segment_ms"] = args.min_segment_ms
        report["min_segment_mm"] = args.min_segment_mm
        _print_report(report, args.stop_threshold)
        reports.append(report)

    if len(reports) > 1:
        print()
        print("配对比较（仅机械臂运动段）：")
        print("  %-22s %7s %8s %10s %12s %12s"
              % ("case", "停点", "停点s", "片段", "最大夹角", "最短片段"))
        for report in reports:
            row = report["arm_motion_only"]
            short = row["shortest_segment"] or {}
            print("  %-22s %7d %8.2f %10d %11.1f° %10.0f ms"
                  % (report["case_id"], row["stop_count"],
                     row["stop_time_s"], row["motion_segment_count"],
                     row["max_turn_angle_deg"] or float("nan"),
                     short.get("duration_ms", float("nan"))))

    payload = {
        "stop_threshold_m_s": args.stop_threshold,
        "min_stop_ms": args.min_stop_ms,
        "min_segment_ms": args.min_segment_ms,
        "min_segment_mm": args.min_segment_mm,
        "reverse_angle_deg": REVERSE_ANGLE_DEG,
        "criterion": "diagnostic_only_no_pass_fail_threshold",
        "runs": reports,
    }
    if args.json_out:
        path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        print("TRAJ_DIAG_JSON %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
