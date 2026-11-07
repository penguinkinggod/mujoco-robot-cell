"""diag_chuck_seat.py -- 卡盘自己能不能把一根歪料扶正？

2026-11-07 的疑问：送料末端棒料已有约 1 deg 倾斜、只插进卡爪 25 mm，
夹紧后反而更歪（2.1 deg），棒料中心偏心 1.47 mm。机械臂松爪对照
（FEED_CLAMP_GRIP=0/0.2/0.5）没有解决，所以把机械臂完全拿掉，只问卡盘：

    给定插入深度 + 初始倾斜，三爪闭合后棒料是否回到主轴轴线？

和 diag_chuck_grip 的区别：那个脚本永远把料完美摆正在卡盘轴线上，
只能证明"放正了能夹住"，证明不了"放歪了能扶正"。

Run:
    D:\\robot-env\\Scripts\\python.exe src/diag_chuck_seat.py
"""
import numpy as np
import mujoco

import scene_machining as S

SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])
JAW_FRONT = S.LATHE_X - S.JAW_LEN - 0.002


def quat_tilt_y(deg):
    """在基准姿态上叠加绕世界 y 的小角度转动（棒料在 x-z 面内歪）。"""
    a = np.deg2rad(deg) / 2.0
    q = np.array([np.cos(a), 0.0, np.sin(a), 0.0])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q, SHAFT_Q0)
    return out


def run(insert_mm, tilt_deg, clamp_ctrl=None):
    model, data, ids = S.make_scene()
    sq = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    ja = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_slide_%d" % k)] for k in range(3)]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
    bid = ids["shaft"]
    clamp_ctrl = S.CHUCK_CLAMP_CTRL if clamp_ctrl is None else clamp_ctrl

    tip_x = JAW_FRONT + insert_mm / 1000.0
    mujoco.mj_resetData(model, data)
    data.qpos[sq:sq + 3] = [tip_x - S.SHAFT_L / 2, S.LATHE_Y, S.LATHE_Z]
    data.qpos[sq + 3:sq + 7] = quat_tilt_y(tilt_deg)
    data.qvel[:] = 0.0
    # 张开位内径 44 mm，棒料悬空会在丝杠走完之前先掉到台面上（第一版
    # 六组全部 sink 50 mm、接触 0，就是这个摆法错误，不是卡盘夹不住）。
    # 真实送料是机械臂托着料进卡爪，所以夹紧阶段先关重力等效"托住"，
    # 三爪咬上之后再恢复重力，才量得到"扶正 + 夹住"两件事。
    g0 = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    mujoco.mj_forward(model, data)

    for i in range(500):
        t = (i + 1) / 500.0
        s = t * t * (3.0 - 2.0 * t)
        for a in ids["jaw_acts"]:
            data.ctrl[a] = S.CHUCK_OPEN_CTRL + s * (clamp_ctrl - S.CHUCK_OPEN_CTRL)
        mujoco.mj_step(model, data)
    model.opt.gravity[:] = g0
    for _ in range(1500):
        for a in ids["jaw_acts"]:
            data.ctrl[a] = clamp_ctrl
        mujoco.mj_step(model, data)

    p = data.qpos[sq:sq + 3].copy()
    R = data.xmat[bid].reshape(3, 3)
    rod = R[:, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(rod[0]))))))
    # 棒料头端（卡爪段内）的位置比棒料中心更能说明"爪里夹正没有"：
    # 中心悬在爪外 55 mm，倾斜 1 deg 就会贡献约 1 mm 假偏心。
    tip = p + rod * (S.SHAFT_L / 2)
    centre_ecc = float(np.hypot(p[1] - S.LATHE_Y, p[2] - S.LATHE_Z) * 1000)
    tip_ecc = float(np.hypot(tip[1] - S.LATHE_Y, tip[2] - S.LATHE_Z) * 1000)
    sink = float((S.LATHE_Z - p[2]) * 1000)

    forces = [0.0, 0.0, 0.0]
    contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if gid not in (c.geom1, c.geom2):
            continue
        for k in range(3):
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     for g in (c.geom1, c.geom2)]
            if any(n.startswith("jaw_geom_%d" % k) for n in names):
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f)
                forces[k] += abs(f[0])
                contacts += 1
    jaw_q = np.array([data.qpos[a] * 1000 for a in ja])
    return dict(insert_mm=insert_mm, tilt0_deg=tilt_deg, tilt_deg=tilt,
                centre_ecc_mm=centre_ecc, tip_ecc_mm=tip_ecc, sink_mm=sink,
                contacts=contacts, forces=forces,
                jaw_q=jaw_q, spread=float(jaw_q.max() - jaw_q.min()))


def main():
    print("=" * 88)
    print("卡盘扶正能力：只摆料、不动机械臂")
    print("=" * 88)
    print("  卡爪前端 x=%.3f  爪长 %.0f mm  软爪半径间隙 %.2f mm"
          % (JAW_FRONT, S.JAW_LEN * 1000, S.JAW_ARC_CLEAR * 1000))
    print("  %-8s %-8s %9s %11s %11s %8s %6s  %s"
          % ("插入mm", "初始倾°", "最终倾°", "中心偏mm", "头端偏mm", "下沉mm",
             "接触", "三爪力N"))
    for insert in (25, 47):
        for tilt in (0.0, 1.0, 2.0):
            r = run(insert, tilt)
            print("  %-8.0f %-8.1f %9.2f %11.2f %11.2f %8.3f %6d  %s"
                  % (r["insert_mm"], r["tilt0_deg"], r["tilt_deg"],
                     r["centre_ecc_mm"], r["tip_ecc_mm"], r["sink_mm"],
                     r["contacts"], np.round(r["forces"], 1)))
    print()
    print("  读法：头端偏 = 爪里那段有没有夹正；中心偏 = 悬在外面那段被倾斜放大多少。")
    print("OK")


if __name__ == "__main__":
    main()
