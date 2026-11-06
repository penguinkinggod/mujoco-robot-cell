"""diag_jaw_wrap.py -- 三爪包角/段数扫描：能不能「三块 120deg 拼满一圈」？

问题（用户提出）：一个 Phi20 的孔直接砍成三份 120deg，每份当一个爪，
闭合时正好拼回完整的圆，为什么不做？

三个可量化的东西决定答案，这里一次测完：

    wrap    每爪包角（deg）、每爪段数
    force   三爪法向力、力比、接触点数
    jam     爪-爪端面接触对数（包角 120 时端面在角度上重合成共面，
            丝杠过盈 5 mm 会先被端面顶住，力全耗在爪自身）
    hold    20 N 轴向拉拔位移、夹紧后下沉

Run:
    D:\\robot-env\\Scripts\\python.exe diag_jaw_wrap.py
"""
import sys

import numpy as np
import mujoco

import scene_machining as S

SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])
PULL_N = 20.0
SETTLE = 1500
RAMP = 500


def _prefix_k(model, gid):
    """geom 名字 -> 爪号字符串；非卡爪返回 None。"""
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    if n.startswith("jaw_geom_"):
        return n.split("_")[2]
    if n.startswith("jaw_back_"):
        return n.split("_")[2]
    return None


def run(half_deg, segs, clamp_ctrl=None, force_limit=None, pull_n=PULL_N):
    S.JAW_ARC_HALF_DEG = float(half_deg)
    S.JAW_ARC_SEGS = int(segs)
    if force_limit is not None:
        S.CHUCK_FORCE_LIMIT = float(force_limit)
    if clamp_ctrl is None:
        clamp_ctrl = S.CHUCK_CLAMP_CTRL

    model, data, ids = S.make_scene()
    sq = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    ja = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_slide_%d" % k)] for k in range(3)]
    acts = ids["jaw_acts"]
    shaft_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
    shaft_body = int(model.geom_bodyid[shaft_geom])

    mujoco.mj_resetData(model, data)
    data.qpos[sq:sq + 3] = [S.LATHE_X - 0.050, S.LATHE_Y, S.LATHE_Z]
    data.qpos[sq + 3:sq + 7] = SHAFT_Q0
    data.qvel[:] = 0.0
    for a in ja:
        data.qpos[a] = S.JAW_ARC_Q0
    mujoco.mj_forward(model, data)
    p0 = data.qpos[sq:sq + 3].copy()

    for i in range(RAMP):
        t = (i + 1) / RAMP
        s = t * t * (3.0 - 2.0 * t)
        for a in acts:
            data.ctrl[a] = S.JAW_ARC_Q0 + s * (clamp_ctrl - S.JAW_ARC_Q0)
        mujoco.mj_step(model, data)
    for _ in range(SETTLE):
        for a in acts:
            data.ctrl[a] = clamp_ctrl
        mujoco.mj_step(model, data)

    jaw_q = np.array([data.qpos[a] * 1000.0 for a in ja])
    spread = float(jaw_q.max() - jaw_q.min())
    # 执行器实际出力 vs 力上限：判断夹紧力是被 forcerange 封顶，
    # 还是被 kp x 位移误差（位置伺服）决定。这两者对应的调法完全不同。
    trn_j = int(model.actuator_trnid[acts[0]][0])
    act_force = float(data.actuator_force[acts[0]])
    act_err = float(clamp_ctrl - data.qpos[model.jnt_qposadr[trn_j]]) * 1000.0
    p = data.qpos[sq:sq + 3]
    ecc = float(np.hypot((p[1] - S.LATHE_Y) * 1000, (p[2] - S.LATHE_Z) * 1000))
    sink = float((p0[2] - p[2]) * 1000)
    rod_ax = data.xmat[shaft_body].reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(rod_ax[0]))))))

    forces = [0.0, 0.0, 0.0]
    contacts = 0
    jam = 0
    on_chuck = 0
    for i in range(data.ncon):
        c = data.contact[i]
        ks = (_prefix_k(model, c.geom1), _prefix_k(model, c.geom2))
        if ks[0] is not None and ks[1] is not None and ks[0] != ks[1]:
            jam += 1
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if (ks[0] is not None and n2.startswith("chuck")) or \
           (ks[1] is not None and n1.startswith("chuck")):
            on_chuck += 1
        if shaft_geom not in (c.geom1, c.geom2):
            continue
        for k in range(3):
            if ks[0] == str(k) or ks[1] == str(k):
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f)
                contacts += 1
                forces[k] += abs(f[0])
    ratio = (max(forces) / max(min(forces), 1e-6)
             if contacts >= 6 else float("inf"))

    # 轴向拉拔（沿 -x，把料从孔里往外拽）
    data.xfrc_applied[:] = 0.0
    data.xfrc_applied[shaft_body, :3] = [-pull_n, 0.0, 0.0]
    for _ in range(500):
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0.0
    pull = float(abs(data.qpos[sq] - p[0]) * 1000)
    p_after = data.qpos[sq:sq + 3].copy()
    pull_ecc = float(np.hypot((p_after[1] - S.LATHE_Y) * 1000,
                              (p_after[2] - S.LATHE_Z) * 1000))

    return dict(wrap=2 * half_deg, segs=segs,
                step=2 * half_deg / segs,
                chord=S.SHAFT_D / 2 * 1000 * (1 - np.cos(np.deg2rad(half_deg / segs))),
                jaw_q=jaw_q, spread=spread, forces=forces, ratio=ratio,
                contacts=contacts, jam=jam, on_chuck=on_chuck,
                ecc=ecc, sink=sink, tilt=tilt,
                pull=pull, pull_ecc=pull_ecc, pull_n=pull_n,
                force_limit=S.CHUCK_FORCE_LIMIT,
                act_force=act_force, act_err=act_err)


def sweep_wrap():
    print("=" * 92)
    print("卡爪包角 / 段数扫描（一个孔砍成三份？）")
    print("=" * 92)
    print("  判据：位移差 <= 0.2 mm  力比 <= 2.0  接触 >= 6  下沉 <= 0.5 mm")
    print()

    cases = [
        (S.JAW_ARC_HALF_DEG, 5,
         "当前 %.0fdeg / 5 段" % (2 * S.JAW_ARC_HALF_DEG)),
        (60.0, 6, "120deg / 6 段  20deg 一段"),
        (60.0, 3, "120deg / 3 段  40deg 一段"),
        (58.0, 6, "116deg / 6 段  留 4deg 缝"),
        (60.0, 6, "120deg / 6 段  丝杠不过盈"),
    ]
    clamp_sets = [None, None, None, None, S.JAW_ARC_Q0]

    rows = []
    for (h, n, label), cc in zip(cases, clamp_sets):
        r = run(h, n, cc)
        r["label"] = label
        rows.append(r)
        print("  %-28s 包角 %5.1f  段 %d x %4.1f deg  弦高 %5.2f mm"
              % (label, r["wrap"], r["segs"], r["step"], r["chord"]))
        print("      三爪位移 %s mm  差 %.3f"
              % (np.round(r["jaw_q"], 3), r["spread"]))
        print("      法向力   %s N   力比 %.2f   接触 %d   爪-爪端面 %d"
              % (np.round(r["forces"], 1), r["ratio"], r["contacts"], r["jam"]))
        print("      下沉 %.3f mm   偏心 %.2f mm   倾斜 %.2f deg"
              % (r["sink"], r["ecc"], r["tilt"]))
        print("      20 N 拉拔 %.3f mm   拔后偏心 %.2f mm"
              % (r["pull"], r["pull_ecc"]))
        print()

    print("-" * 92)
    print("  汇总（下沉 / 拉拔 / 力比 / 爪碰爪）")
    for r in rows:
        ok = ("PASS" if (r["spread"] <= 0.2 and r["ratio"] <= 2.0
                         and r["sink"] <= 0.5 and r["pull"] <= 0.5
                         and r["jam"] == 0) else "FAIL")
        print("  %-28s 下沉 %6.3f  拉拔 %6.3f  力比 %5.2f  端面 %2d  %s"
              % (r["label"], r["sink"], r["pull"], r["ratio"], r["jam"], ok))


def sweep_force():
    print("=" * 92)
    print("夹紧力上限扫描（包角取当前默认 %.0fdeg / 5 段，拉拔 200 N）"
          % (2 * S.JAW_ARC_HALF_DEG))
    print("=" * 92)
    print("  判据：下沉 <= 0.5 mm   200 N 拉拔 <= 0.5 mm   爪-爪端面 0   爪-卡盘 0")
    print()
    for fl in (150.0, 300.0, 500.0, 800.0, 1200.0):
        r = run(S.JAW_ARC_HALF_DEG, 5, None,
                force_limit=fl, pull_n=200.0)
        r["label"] = "上限 %4.0f N" % fl
        print("  %s   执行器出力 %6.1f N   位移误差 %5.2f mm（%s）"
              % (r["label"], r["act_force"], r["act_err"],
                 "已封顶" if abs(r["act_force"]) >= 0.95 * fl else "未封顶"))
        print("      三爪力 %s N  合计 %5.1f N  力比 %5.2f"
              % (np.round(r["forces"], 1), sum(r["forces"]), r["ratio"]))
        print("      接触 %2d   爪-爪端面 %d   爪-卡盘 %d"
              % (r["contacts"], r["jam"], r["on_chuck"]))
        print("      下沉 %6.3f mm  偏心 %5.2f mm  倾斜 %4.2f deg"
              % (r["sink"], r["ecc"], r["tilt"]))
        print("      200 N 拉拔 %7.3f mm   拔后偏心 %5.2f mm"
              % (r["pull"], r["pull_ecc"]))
        print()


def sweep_deep():
    """丝杠模型正解：目标位置给到远超行程的深度，力由 forcerange 封顶。

    现状 ctrl=-0.020 只比实际停位 (-0.0158) 深 4.2 mm，kp*0.0042 = 84 N，
    这时 forcerange 是个永不生效的上限。把目标给到 -0.030 以后，
    位移误差 14 mm -> kp*0.014 = 280 N，被封顶到 forcerange，才真正
    变成「扭矩上限决定夹紧力」。
    """
    print("=" * 92)
    print("深目标 + 力限封顶（包角当前默认 %.0fdeg / 5 段，丝杠目标 -0.030）"
          % (2 * S.JAW_ARC_HALF_DEG))
    print("=" * 92)
    print("  判据：下沉 <= 0.5 mm   200 N 拉拔 <= 0.5 mm   爪-爪端面 0   爪-卡盘 0")
    print()
    S.CHUCK_CTRL_RANGE = (-0.032, 0.012)
    S.CHUCK_CLAMP_CTRL = -0.030
    for fl in (150.0, 300.0, 500.0, 800.0):
        r = run(S.JAW_ARC_HALF_DEG, 5, -0.030,
                force_limit=fl, pull_n=200.0)
        blown = "  <== 拉拔后料被拽飞，数据不可信" if r["pull_ecc"] > 5.0 else ""
        print("  力限 %4.0f N   执行器出力 %6.1f N   位移误差 %5.2f mm（%s）"
              % (fl, r["act_force"], r["act_err"],
                 "已封顶" if abs(r["act_force"]) >= 0.95 * fl else "未封顶"))
        print("      三爪力 %s N  合计 %5.1f N  力比 %5.2f"
              % (np.round(r["forces"], 1), sum(r["forces"]), r["ratio"]))
        print("      接触 %2d   爪-爪端面 %d   爪-卡盘 %d   滑座 %s mm"
              % (r["contacts"], r["jam"], r["on_chuck"],
                 np.round(r["jaw_q"], 2)))
        print("      下沉 %6.3f mm  偏心 %5.2f mm  倾斜 %4.2f deg"
              % (r["sink"], r["ecc"], r["tilt"]))
        print("      200 N 拉拔 %7.3f mm   拔后偏心 %5.2f mm%s"
              % (r["pull"], r["pull_ecc"], blown))
        print()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "wrap"
    if mode == "force":
        sweep_force()
    elif mode == "deep":
        sweep_deep()
    else:
        sweep_wrap()


if __name__ == "__main__":
    main()
