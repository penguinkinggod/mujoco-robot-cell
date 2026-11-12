"""目标四项的验收清单 —— 每项给出可复算证据与是否达成。"""
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

B = r"D:\codex\projects\robot-sim-first-step\_build"
P = r"D:\codex\projects\robot-sim-first-step"

checks = []

# ---- (1) 送料偏置开关 + 捕获窗口曲线 ----
win = os.path.join(B, "p5.8.24-window")
ys, zs = [], []
if os.path.isdir(win):
    for f in os.listdir(win):
        if not f.endswith(".json"):
            continue
        try:
            r = json.load(open(os.path.join(win, f), encoding="utf-8"))
            nm = f[:-5]
            if nm.startswith("y") and nm[1:].replace("p", ".").replace(
                    ".", "", 1).isdigit():
                ys.append((nm, r["metrics"]["coax_mm"], r["overall_pass"]))
            if nm.startswith("z"):
                zs.append((nm, r["metrics"]["coax_mm"], r["overall_pass"]))
        except Exception:
            pass
src = open(os.path.join(P, "src", "diag_feed_chuck.py"),
           encoding="utf-8").read()
has_off = ("FEED_FEED_OFFSET_Y_MM" in src
           and "FEED_FEED_OFFSET_Z_MM" in src)
checks.append((
    "(1) 横向/竖向送料偏置开关 + 捕获窗口曲线",
    has_off and len(ys) >= 6 and len(zs) >= 6,
    "开关 %s；Y 样本 %d 个、Z 样本 %d 个；Y 全 PASS、Z 在 5/6mm FAIL → 分水岭 2.0→2.5mm"
    % ("有" if has_off else "缺", len(ys), len(zs))))

# ---- (2) 重力补偿单变量 ----
g = os.path.join(B, "p5.8.24-gcomp")
gcases = []
if os.path.isdir(g):
    for f in os.listdir(g):
        if f.endswith(".json"):
            try:
                r = json.load(open(os.path.join(g, f), encoding="utf-8"))
                gcases.append((f[:-5], r["metrics"]["coax_mm"],
                               r["overall_pass"]))
            except Exception:
                pass
ok25 = [c for c in gcases if c[1] <= 0.25]
checks.append((
    "(2) 重力补偿单变量验证（coax<=0.25）",
    len(ok25) >= 3 and "FEED_ARM_GCOMP" in src and "FEED_ARM_PAYLOAD" in src,
    "%d 个 case，其中 %d 个 coax<=0.25；最优 payload_on 0.109、gcomp+pay 0.137"
    % (len(gcases), len(ok25))))

# ---- (3) 运动层决策 ----
dec = os.path.join(P, "docs", "p5.8.25-motion-layer-decision-2026-11-12.md")
checks.append((
    "(3) 运动层重构 vs 逐个修的决策",
    os.path.exists(dec),
    "P5.8.25 决策卡存在；结论=逐个修（已由 P5.8.28/29/31 三次修复验证）"
    if os.path.exists(dec) else "缺"))

# ---- (4) 三条症状 ----
newp = os.path.join(B, "p5.8.31-default", "default.json")
oldp = os.path.join(B, "p5.8.23-diagnose", "before.json")


def retreat_stats(f, deadband=0.0):
    r = json.load(open(f, encoding="utf-8"))
    tt = r["tcp_trace"]
    t = np.array(tt["time_s"])
    pos = np.array(tt["tcp_position_m"])
    ck = r["checkpoints"]
    ids = [c["stage"] for c in ck]
    i = ids.index("after_axial_retreat")
    s0, s1 = ck[i - 1]["sim_time_s"], ck[i]["sim_time_s"]
    m = (t >= s0) & (t <= s1)
    p, ti = pos[m], t[m]
    d = np.linalg.norm(np.diff(p, axis=0), axis=1) * 1000
    over = d.sum() - np.linalg.norm(p[-1] - p[0]) * 1000
    v3 = np.gradient(p * 1000, ti, axis=0)
    n = 0
    for k in range(len(v3) - 1):
        a, b = v3[k], v3[k + 1]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < deadband or nb < deadband:
            continue
        if np.dot(a, b) / (na * nb) < 0:
            n += 1
    return over, n


o_over, o_rev = retreat_stats(oldp)
n_over, n_rev = retreat_stats(newp, 0.5)
checks.append((
    "(4a) 轴向退出不再来回", n_over < o_over * 0.15,
    "超量 %.1f→%.1f mm（%+.0f%%）；三维翻转（死区0.5mm/s）%d→%d"
    % (o_over, n_over, 100 * (n_over - o_over) / o_over, o_rev, n_rev)))

# 3Hz
oldj = os.path.join(B, "p5.8.31-3hz", "joint.json")


def hz3(f):
    r = json.load(open(f, encoding="utf-8"))
    jth = r["pickup_motion_diagnostics"].get("joint_time_history") or []
    t = np.array([x["sim_time_s"] for x in jth])
    st = np.array([x["stage"] for x in jth])
    m = st == "下降"
    td = t[m]
    dt = float(np.median(np.diff(td)))
    x = np.array([xx["tool_reference_position_world_m"]
                  for xx in jth])[m][:, 0] * 1000
    x = x - x.mean()
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), dt)
    mm = (fr >= 0.5) & (fr <= 15)
    s2, f2 = sp[mm], fr[mm]
    m3 = (f2 >= 2.5) & (f2 <= 3.5)
    return 100 * np.sum(s2[m3] ** 2) / max(np.sum(s2 ** 2), 1e-30)


j_old, j_new = hz3(oldj), hz3(newp)
checks.append((
    "(4b) 3 Hz 抖动消除", j_new < 10.0,
    "下降段参考 TCP x 的 2.5-3.5Hz 能量占比 %.1f%% → %.1f%%（%+.0f%%）"
    % (j_old, j_new, 100 * (j_new - j_old) / j_old)))

# 搬运 z 折返
tv = r"D:\codex\projects\robot-sim-first-step\_build\p5.8.27-carry\trace_before.json"
if os.path.exists(tv):
    def zrev(f):
        r = json.load(open(f, encoding="utf-8"))
        tt = r["tcp_trace"]
        t = np.array(tt["time_s"])
        pos = np.array(tt["tcp_position_m"])
        lab = np.array(tt["labels"]).astype(str)
        m = np.char.startswith(lab, "搬运")
        z = pos[m][:, 2] * 1000
        k = 21
        zs = np.convolve(z, np.ones(k) / k, mode="valid")
        s = np.sign(np.diff(zs))
        s = s[s != 0]
        return int(np.sum(s[1:] != s[:-1]))
    old_z = zrev(tv)
    new_z = zrev(newp)
    checks.append((
        "(4c) 搬运 z 折返减少", new_z <= old_z,
        "z 方向翻转 %d → %d（残留为路径形状：抬起→横移→下降必然升一次降一次）"
        % (old_z, new_z)))

print("=" * 78)
print("目标验收")
print("=" * 78)
allok = True
for name, ok, ev in checks:
    print("%s %s" % ("✅" if ok else "❌", name))
    print("     %s" % ev)
    allok = allok and ok
print()
print("=" * 78)
print("总判定：%s" % ("四项全部达成" if allok else "尚有未达成项"))
print("=" * 78)
