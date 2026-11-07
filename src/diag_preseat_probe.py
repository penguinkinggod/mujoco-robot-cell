"""diag_preseat_probe.py -- 预收托圈的孔径到底是不是 1.5 mm？

2026-11-07 现象：送料末端把软爪预收到"料半径 + 1.5 mm"，棒料还没被夹，
同轴度却从 0.08 mm 跳到 3.60 mm 且出现 3 个接触。若几何正确，1.5 mm
的环形缝隙里不可能有接触。这个脚本把机械臂拿掉、把料精确摆在主轴轴线
上，只预收软爪，量三爪内表面到轴线的实际半径。

Run:
    D:\\robot-env\\Scripts\\python.exe src/diag_preseat_probe.py
"""
import numpy as np
import mujoco

import scene_machining as S

SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])
JAW_FRONT = S.LATHE_X - S.JAW_LEN - 0.002


def jaw_inner_radius(model, data, k):
    """第 k 爪所有弧片内表面到主轴轴线的最小距离（解析量，不靠接触）。"""
    ax = np.array([S.LATHE_Y, S.LATHE_Z])
    best = 1e9
    for s in range(S.JAW_ARC_SEGS):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                              "jaw_geom_%d_%02d" % (k, s))
        if g < 0:
            continue
        gid_b = model.geom_bodyid[g]
        R = data.xmat[gid_b].reshape(3, 3)
        p = data.xpos[gid_b] + R @ model.geom_pos[g]
        half = model.geom_size[g]
        # 弧片是 box，径向半厚 = half[1]；用 box 中心到轴线的距离减去它
        r = np.hypot(p[1] - ax[0], p[2] - ax[1]) - half[1]
        best = min(best, r)
    return best


def run(preseat_clear_mm, hold=True):
    model, data, ids = S.make_scene()
    sq = model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    ja = [model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_slide_%d" % k)] for k in range(3)]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
    bid = ids["shaft"]

    tip_x = JAW_FRONT + 0.025
    mujoco.mj_resetData(model, data)
    data.qpos[sq:sq + 3] = [tip_x - S.SHAFT_L / 2, S.LATHE_Y, S.LATHE_Z]
    data.qpos[sq + 3:sq + 7] = SHAFT_Q0
    data.qvel[:] = 0.0
    g0 = model.opt.gravity.copy()
    if hold:
        model.opt.gravity[:] = 0.0
    mujoco.mj_forward(model, data)

    q = S.JAW_ARC_Q0 + (preseat_clear_mm / 1000.0 - S.JAW_ARC_CLEAR)
    for i in range(500):
        t = (i + 1) / 500.0
        s = t * t * (3.0 - 2.0 * t)
        for a in ids["jaw_acts"]:
            data.ctrl[a] = S.CHUCK_OPEN_CTRL + s * (q - S.CHUCK_OPEN_CTRL)
        mujoco.mj_step(model, data)
    for _ in range(600):
        for a in ids["jaw_acts"]:
            data.ctrl[a] = q
        mujoco.mj_step(model, data)

    p = data.qpos[sq:sq + 3].copy()
    ecc = float(np.hypot(p[1] - S.LATHE_Y, p[2] - S.LATHE_Z) * 1000)
    ncon = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if gid in (c.geom1, c.geom2):
            ncon += 1
    inner = [jaw_inner_radius(model, data, k) * 1000 for k in range(3)]
    jaw_q = np.array([data.qpos[a] * 1000 for a in ja])
    print("  留缝%4.1f mm  实际内半径 %s mm（料半径 %.1f）  偏心 %5.2f mm  "
          "接触 %d  爪位移 %s mm 同步差 %.2f mm"
          % (preseat_clear_mm, np.round(inner, 2), S.SHAFT_D / 2 * 1000,
             ecc, ncon, np.round(jaw_q, 2), jaw_q.max() - jaw_q.min()))


def main():
    print("=" * 88)
    print("预收托圈孔径实测（料精确摆在轴线上，重力关掉=机械臂托着）")
    print("=" * 88)
    for clear in (1.5, 3.0, 0.2, 0.0):
        run(clear)
    print("OK")


if __name__ == "__main__":
    main()
