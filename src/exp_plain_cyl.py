"""exp_plain_cyl.py -- 纯圆柱工件（无铣扁、无凹槽）+ 不同夹紧量的抓取对照。

用户要求：删虎钳、删浅平台、工件换纯圆柱，机械臂把它抓起来送进卡盘。
这个脚本只回答一个问题：**现状夹爪能不能夹住纯圆柱，卡在哪一步。**

对每个 GRIP_CLOSE（夹爪控制量，越大开口越大）：
    1. 复位后在台面上静置 600 步  -> 纯圆柱会自己滚走吗？
    2. 接近 -> 下降到轴心
    3. 夹紧
    4. 抬升 80 mm（真实物理，不加任何运动学锁）
    5. 量：抬升高度、轴倾角、pad 中心到轴心距离

跑法：
    D:\\robot-env\\Scripts\\python.exe exp_plain_cyl.py
"""
import numpy as np
import mujoco

import diag_feed_chuck as D
from demo_08_pick_and_load import (
    Arm, WRIST_ROLL_GRASP, GRIP_OPEN,
)

GRIP_VALUES = (0.15, 0.17, 0.19, 0.21, 0.24)


def one_run(model, data, grip_close, wrist=WRIST_ROLL_GRASP):
    arm = Arm(model, data)
    sj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
    sq = model.jnt_qposadr[sj]
    axis = np.array([D.LATHE_ORIGIN[0] - 0.016,
                     D.LATHE_ORIGIN[1], D.LATHE_ORIGIN[2]])

    def shaft():
        return data.qpos[sq:sq + 3].copy()

    def tilt():
        return D.tilt_deg(data.qpos[sq + 3:sq + 7])

    def coax():
        p = shaft()
        return float(np.hypot(p[1] - axis[1], p[2] - axis[2]) * 1000)

    # ---- reset / home ----
    mujoco.mj_resetData(model, data)
    data.qpos[arm.qadr[4]] = wrist
    data.ctrl[4] = wrist
    data.ctrl[arm.grip_act] = GRIP_OPEN
    for _ in range(600):
        for k in range(5):
            data.ctrl[k] = data.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)

    rest = shaft()
    roll = float(np.linalg.norm(rest - np.array(D.F.SHAFT_POS)) * 1000)

    # ---- grasp ----
    pick = [
        ("approach", [rest[0], rest[1], rest[2] + 0.085], GRIP_OPEN, 450),
        ("descend",  [rest[0], rest[1], rest[2] + 0.002], GRIP_OPEN, 380),
        ("close",    [rest[0], rest[1], rest[2] + 0.002], grip_close, 320),
    ]
    for _name, tgt, grip, ns in pick:
        arm.move_to(np.array(tgt), n_steps=ns, grip=grip,
                    wrist_roll=wrist)

    at_close = shaft()
    tilt_close = tilt()
    dpad = float(np.linalg.norm(arm.pad_center() - at_close) * 1000)

    # ---- lift, real physics ----
    lift_tgt = np.array([at_close[0], at_close[1], at_close[2] + 0.08])
    arm.move_to(lift_tgt, n_steps=420, grip=grip_close)

    after = shaft()
    risen = (after[2] - at_close[2]) * 1000
    grab = "GRAB" if risen > 40 else "DROP"
    return dict(grip=grip_close, roll=roll, tilt_close=tilt_close,
                tilt_lift=tilt(), dpad=dpad, risen=risen,
                coax=coax(), verdict=grab)


def main():
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else "none"
    fix = None if raw in ("none", "") else raw
    dia = float(sys.argv[2]) if len(sys.argv) > 2 else None
    if dia:
        D.F.SHAFT_D = dia / 1000.0
        D.F.SHAFT_POS[2] = D.F.TABLE_TOP_Z + D.F.SHAFT_D / 2
        D.F.SHAFT_MASS = 0.197 * (dia / 20.0) ** 2
    model = D.build_model(grip_fix=None if fix == "wrist" else fix)
    data = mujoco.MjData(model)
    print("=" * 78)
    print("纯圆柱 Φ%d（无铣扁 / 无浅槽平台）抓取对照  grip_fix=%s"
          % (D.F.SHAFT_D * 1000, fix or "none"))
    print("=" * 78)
    print("  %-6s %-9s %-11s %-9s %-10s %-9s %s"
          % ("GRIP", "台面滚走", "夹紧时倾角", "pad-轴", "抬升", "同轴", "判定"))
    print("  " + "-" * 74)
    if fix == "wrist":
        # 纯圆柱下重扫手腕角度：dpad 越小、抬起后越平越好
        print("  %-8s %-9s %-11s %-9s %-10s %s"
              % ("WRIST", "台面滚走", "夹紧时倾角", "pad-轴", "抬升", "判定"))
        print("  " + "-" * 70)
        best = None
        for wr in (0.0, 0.4, 0.8, 1.2, 1.55, 1.9, 2.3, 2.7, 3.14):
            r = one_run(model, data, 0.17, wrist=wr)
            print("  %-8.2f %-9s %-11s %-9s %-10s %s"
                  % (wr, "%.1f mm" % r["roll"], "%.1f deg" % r["tilt_close"],
                     "%.1f mm" % r["dpad"], "%+.1f mm" % r["risen"],
                     r["verdict"]))
            if r["verdict"] == "GRAB" and r["risen"] > 40:
                score = r["tilt_lift"] + max(0.0, 20.0 - r["risen"])
                if best is None or score < best[0]:
                    best = (score, wr, r)
        print()
        if best:
            print("BEST_WRIST=%.2f risen=%.1f mm tilt_lift=%.1f deg"
                  % (best[1], best[2]["risen"], best[2]["tilt_lift"]))
        else:
            print("NO_GRAB_ANY_WRIST")
        print("DONE")
        return 0

    rows = []
    for gc in GRIP_VALUES:
        r = one_run(model, data, gc)
        rows.append(r)
        print("  %-6.2f %-9s %-11s %-9s %-10s %-9s %s"
              % (gc, "%.1f mm" % r["roll"], "%.1f deg" % r["tilt_close"],
                 "%.1f mm" % r["dpad"], "%+.1f mm" % r["risen"],
                 "%.1f mm" % r["coax"], r["verdict"]))
    ok = [r for r in rows if r["verdict"] == "GRAB"]
    print()
    if ok:
        best = max(ok, key=lambda r: r["risen"])
        print("BEST_GRIP=%.2f risen=%.1f mm tilt=%.1f deg"
              % (best["grip"], best["risen"], best["tilt_lift"]))
    else:
        print("NO_GRAB: 现状夹爪在所有夹紧量下都抓不住纯圆柱")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
