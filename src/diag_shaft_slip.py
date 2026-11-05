"""diag_shaft_slip.py —— 查工件为什么在下降入座时从夹爪里滑出来。

现象（demo_08 日志）
    ⑤ 横移到钳口上方  pad 中心 [0.159, 0.075, 0.187]  工件 [0.159, 0.075, 0.187]  ✓ 对齐
    ⑥ 下降到钳口      pad 中心 [0.160, 0.075, 0.022]  工件 [0.196, 0.043, 0.061]  ✗ 跑了 60 mm

工件从 y=0.075 跑到 y=0.043，正好贴着 jaw_R（y∈[0.047, 0.061]）的外侧。
说明下降过程中工件撞到了钳口的【顶面】或者【侧面】，被顶飞了。

这个脚本逐帧记录：
    - pad 中心
    - 工件中心
    - 两者距离（正常应该 < 5 mm，变大叫"滑脱"）
    - 接触对（哪个 geom 和工件碰）

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_shaft_slip.py
"""
import numpy as np
import mujoco

import scene_full as S
from demo_08_pick_and_load import Arm, WRIST_ROLL_GRASP, GRIP_CLOSE, GRIP_OPEN


def geom_name(model, i):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    return n or "geom%d" % i


def main():
    model, data = S.make_scene()
    arm = Arm(model, data)

    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]

    print("=" * 78)
    print("钳口几何（判断工件能不能落进缝里）")
    print("=" * 78)
    print()
    jawL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "vise_jaw_L")
    jawR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "vise_jaw_R")
    for nm, gi in (("jawL", jawL), ("jawR", jawR)):
        p, s = model.geom_pos[gi], model.geom_size[gi]
        print("  %s: 中心 y=%.4f  半厚 %.4f  -> y ∈ [%.4f, %.4f]"
              % (nm, p[1], s[1], p[1] - s[1], p[1] + s[1]))
        print("        x 宽 ±%.4f -> [%.4f, %.4f]" % (s[0], p[0] - s[0], p[0] + s[0]))
        print("        z 高 ±%.4f -> [%.4f, %.4f]" % (s[2], p[2] - s[2], p[2] + s[2]))
    print()
    print("  净空: y ∈ [%.4f, %.4f]  宽 %.1f mm"
          % (model.geom_pos[jawR][1] + model.geom_size[jawR][1],
             model.geom_pos[jawL][1] - model.geom_size[jawL][1],
             (model.geom_pos[jawL][1] - model.geom_size[jawL][1]
              - model.geom_pos[jawR][1] - model.geom_size[jawR][1]) * 1000))
    print("  工件 Φ%.0f mm" % (S.SHAFT_D * 1000))
    print("  钳口顶面 z = %.4f" % (model.geom_pos[jawL][2] + model.geom_size[jawL][2]))
    print()

    # 摆到抓取姿态
    arm.d.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    arm.d.ctrl[4] = WRIST_ROLL_GRASP
    arm.d.ctrl[arm.grip_act] = GRIP_OPEN
    for _ in range(600):
        for k in range(5):
            arm.d.ctrl[k] = arm.d.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)

    shaft_home = data.qpos[sq:sq + 3].copy()
    R = S.SHAFT_D / 2
    grasp = np.array([shaft_home[0], shaft_home[1], shaft_home[2]])

    jaw_c = S.jaw_center()
    JAW_X, JAW_Y, JAW_Z = jaw_c

    # 先抓起来
    for tgt, grip in [
        ([grasp[0], grasp[1], grasp[2] + R + 0.075], GRIP_OPEN),
        ([grasp[0], grasp[1], grasp[2] + 0.002], GRIP_OPEN),
        ([grasp[0], grasp[1], grasp[2] + 0.002], GRIP_CLOSE),
        ([grasp[0], grasp[1], grasp[2] + 0.13], GRIP_CLOSE),
        ([JAW_X, JAW_Y, JAW_Z + 0.12], GRIP_CLOSE),
    ]:
        for _ in range(3):
            arm.move_to(np.array(tgt), n_steps=200, grip=grip,
                        wrist_roll=WRIST_ROLL_GRASP)

    print("=" * 78)
    print("⑤ 到位后开始下降，逐帧监控滑脱")
    print("=" * 78)
    print()
    print("  %-6s %-22s %-22s %-9s %s"
          % ("step", "pad 中心", "工件中心", "偏差mm", "接触"))
    print("  " + "-" * 84)

    # 下降
    target = np.array([JAW_X, JAW_Y, JAW_Z - 0.002])
    off = arm.pad_offset()
    q_goal, _ = arm.ik(target - off, lock_wrist_roll=WRIST_ROLL_GRASP)
    q_start = np.array([data.qpos[a] for a in arm.qadr])

    worst = 0.0
    first_contact = None
    for i in range(2000):
        frac = (i + 1) / 1400.0
        frac = min(frac, 1.0)
        s = frac * frac * (3 - 2 * frac)
        q = q_start + s * (q_goal - q_start)
        for a, v in zip(arm.qadr, q):
            data.ctrl[a] = v
        data.ctrl[arm.grip_act] = GRIP_CLOSE
        mujoco.mj_step(model, data)

        pad = arm.pad_center()
        sh = data.qpos[sq:sq + 3]
        d = np.linalg.norm(pad - sh)
        worst = max(worst, d)

        cts = []
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = con.geom1, con.geom2
            if g1 == model.geom("shaft_geom").id or g2 == model.geom("shaft_geom").id:
                other = g2 if g1 == model.geom("shaft_geom").id else g1
                cts.append(geom_name(model, other))
        cts = sorted(set(cts))

        if cts and first_contact is None and i > 5:
            first_contact = (i, sh.copy(), cts)

        if i % 100 == 0:
            print("  %-6d %-22s %-22s %-9.1f %s"
                  % (i, np.round(pad, 3), np.round(sh, 3), d * 1000,
                     ",".join(cts)[:40]))

    print()
    print("  最大偏差 %.1f mm" % (worst * 1000))
    if first_contact:
        print("  首次接触 step=%d  工件=%s  碰到的: %s"
              % (first_contact[0], np.round(first_contact[1], 3),
                 first_contact[2]))
    print()
    print("  最终 pad 中心 %s" % np.round(arm.pad_center(), 4))
    print("  最终 工件     %s" % np.round(data.qpos[sq:sq + 3], 4))
    print()
    print("OK")


if __name__ == "__main__":
    main()
