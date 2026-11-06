"""diag_diameter_sweep.py -- 一套软爪能不能通吃 Φ15~Φ25。

范围只到卡盘这一站：机械臂、V 槽、抓取姿态、进给轨迹一律不动。
控制变量，出来的数才说得出是哪一处引起的。

两组对照：

    fit    每个直径按该直径就地镗孔（弧爪半径跟着料走） = 设计上限
    phi20  固定 Φ20 弧爪去夹别的直径（不换爪）          = 现实情况

判据和主线一致：同轴取工件中心在 yz 面到卡盘轴线的距离，
倾斜取工件轴线与世界 x 的夹角（不是四元数总转角，圆棒在 V 槽里滚一下
会让后者报出几十度的假数）。

跑法：
    D:\\robot-env\\Scripts\\python.exe src/diag_diameter_sweep.py

输出：_build/diameter_sweep.csv、_build/diameter_sweep.png
"""
import csv
import math
import os
import sys

import numpy as np
import mujoco

import scene_machining as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "_build")
os.makedirs(OUT, exist_ok=True)

DIAMETERS = [0.015, 0.018, 0.020, 0.022, 0.025]
MODES = ("fit", "phi20")
_SHAFT_XY = S.SHAFT_POS[:2].copy()
Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])
# 工件露在卡盘外面的长度。主线送料是插入 25 mm、外留 55 mm（和
# diag_feed_chuck.FEED_INSERT_MM 对得上），扫描必须用同一个摆位，
# 否则量到的是"料被塞进卡盘孔里"的工况，不是夹持。
INSET = float(os.environ.get("SWEEP_INSET_MM", "55")) / 1000.0
# import 时抓一次默认值。configure() 每次显式写回，避免扫描改过全局后串扰。
_DEF_HALF = S.JAW_ARC_HALF_DEG
_DEF_OVERLAP = S.JAW_ARC_OVERLAP
_DEF_THICK = S.JAW_ARC_THICK


def configure(D, mode, half_deg=None, overlap=None, thick=None):
    """改模块全局再重建场景。make_scene() 每次从头 build_scene()，读的就是这些量。

    half_deg / overlap / thick 给包角扫描用，None 表示回到出厂值。
    """
    S.SHAFT_D = D
    S.SHAFT_MASS = math.pi * (D / 2) ** 2 * S.SHAFT_L * S.STEEL_RHO
    S.SHAFT_POS = np.array([_SHAFT_XY[0], _SHAFT_XY[1],
                            S.TABLE_TOP_Z + D / 2])
    S.JAW_ARC_BORE_R = None if mode == "fit" else 0.010
    S.JAW_ARC_HALF_DEG = _DEF_HALF if half_deg is None else half_deg
    S.JAW_ARC_OVERLAP = _DEF_OVERLAP if overlap is None else overlap
    S.JAW_ARC_THICK = _DEF_THICK if thick is None else thick


def run_case(D, mode, half_deg=None, overlap=None, thick=None):
    configure(D, mode, half_deg, overlap, thick)
    model, data, ids = S.make_scene()

    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    bid = ids["shaft"]
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
    acts = ids["jaw_acts"]
    axis = np.array([S.LATHE_X, S.LATHE_Y, S.LATHE_Z])

    def place(inset=None):
        inset = INSET if inset is None else inset
        mujoco.mj_resetData(model, data)
        data.qpos[sq:sq + 3] = [
            S.LATHE_X + S.SHAFT_L / 2 - inset, S.LATHE_Y, S.LATHE_Z]
        data.qpos[sq + 3:sq + 7] = Q0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    def drive(target, steps, pull=0.0):
        for _ in range(steps):
            for a in acts:
                data.ctrl[a] = target
            data.xfrc_applied[bid, 0] = pull
            mujoco.mj_step(model, data)
        data.xfrc_applied[bid, 0] = 0.0

    def contacts():
        n, force, inter, jj_force, body_force, back_force = 0, 0.0, 0, 0.0, 0.0, 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            fmag = abs(f[0])
            is_jaw1, is_jaw2 = n1.startswith("jaw_geom"), n2.startswith("jaw_geom")
            is_back1, is_back2 = n1.startswith("jaw_back"), n2.startswith("jaw_back")
            if gid in (c.geom1, c.geom2):
                if is_jaw1 or is_jaw2:
                    n += 1
                    force += fmag
                elif any(k in n1 + n2 for k in
                         ("chuck_body", "spindle", "lathe_headstock")):
                    # 盘体现在有 Φ30 通孔，正常夹持时这里应该是 0。
                    # 只要冒出 kN 级，就说明工件又穿进实体里了。
                    body_force += fmag
            # 爪与爪互顶：名字第三段是爪号，不同号才算。
            # 只数 > 1 N 的：Φ20 主线有 2 对 0.4 N 的擦边接触，不是互顶。
            if is_jaw1 and is_jaw2 and n1.split("_")[2] != n2.split("_")[2]:
                jj_force += fmag
                if fmag > 1.0:
                    inter += 1
            # 背板也是刚体，缩弧片之后它可能变成新的互顶源，单独记一路。
            s1 = n1.split("_")[2] if (is_jaw1 or is_back1) else None
            s2 = n2.split("_")[2] if (is_jaw2 or is_back2) else None
            if (is_back1 or is_back2) and s1 and s2 and s1 != s2:
                back_force += fmag
        return n, force, inter, jj_force, body_force, back_force

    # ---- 张开 -> 夹紧，量几何和力 ----
    # 真机上张开、送进、夹紧这整段都是机械臂托着料，三爪抱死之后才松手。
    # 扫描里没有机械臂，用关重力等效：工件外留 55 mm、重心悬在爪外侧，
    # 只要夹紧没完成就放重力，它会被力矩掀翻（实测倾斜 60°），量到的
    # 是掉下去的角度，不是夹持精度。夹紧完再恢复，接着测下沉才有意义。
    place()
    g0 = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    drive(S.CHUCK_OPEN_CTRL, 200)
    drive(S.CHUCK_CLAMP_CTRL, 1500)
    model.opt.gravity[:] = g0
    n, force, inter, jj_force, body_force, back_force = contacts()

    p = data.qpos[sq:sq + 3]
    coax = float(np.hypot(p[1] - axis[1], p[2] - axis[2]) * 1000)
    ax = data.xmat[bid].reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(ax[0]))))))

    # ---- 重力下沉 ----
    z0 = data.qpos[sq + 2]
    drive(S.CHUCK_CLAMP_CTRL, 2000)
    sink = float((z0 - data.qpos[sq + 2]) * 1000)

    # ---- 20 N 轴向拉拔（重新夹紧一遍，避免和上面的历史耦合）----
    # 同样先托住再夹紧，拉拔本身在恢复重力之后做。
    place()
    model.opt.gravity[:] = 0.0
    drive(S.CHUCK_CLAMP_CTRL, 1200)
    model.opt.gravity[:] = g0
    x0 = data.qpos[sq]
    drive(S.CHUCK_CLAMP_CTRL, 800, pull=20.0)
    pull = float((data.qpos[sq] - x0) * 1000)

    weight = float(model.body_mass[bid] * 9.81)
    ok = (coax < 2.0 and tilt < 2.0 and n >= 6
          and abs(sink) < 0.5 and abs(pull) < 0.3 and inter == 0)

    return dict(mode=mode, D_mm=D * 1000, coax_mm=coax, tilt_deg=tilt,
                contacts=n, force_N=force, sink_mm=sink,
                pull20_mm=pull, jaw_jam_pairs=inter, jaw_jam_N=jj_force,
                back_jam_N=back_force, chuck_body_N=body_force,
                weight_N=weight, ok=ok)


def main():
    rows = []
    print("=" * 92)
    print("变径扫描：只测卡盘这一站，机械臂 / V 槽 / 轨迹不动")
    print("=" * 92)
    print("  %-6s %-6s %8s %7s %6s %8s %8s %9s %8s %9s %-5s"
          % ("组", "直径", "同轴mm", "倾斜°", "接触", "爪力N",
             "下沉mm", "20N拉拔mm", "互顶对", "互顶力N", "判定"))
    print("  " + "-" * 98)
    for mode in MODES:
        for D in DIAMETERS:
            try:
                r = run_case(D, mode)
            except Exception as e:
                print("  %-6s %-6s  仿真异常: %s" % (mode, "Φ%.0f" % (D * 1000), e))
                continue
            rows.append(r)
            print("  %-6s Φ%-5.0f %8.2f %7.2f %6d %8.1f %8.2f %9.2f %8d %9.0f %-5s"
                  % (mode, r["D_mm"], r["coax_mm"], r["tilt_deg"],
                     r["contacts"], r["force_N"], r["sink_mm"],
                     r["pull20_mm"], r["jaw_jam_pairs"], r["jaw_jam_N"],
                     "PASS" if r["ok"] else "FAIL"))
    print()
    print("  注：卡盘盘体和主轴箱现在都有 Φ30 通孔，工件只由三爪夹持，外留 %.0f mm。"
          % (INSET * 1000))
    print("      chuck_body_N 应为 0（重力擦到孔壁算几牛以内）；若是 kN 级，说明又穿模了。")
    print("      爪力 N 是第一手实测接触力，可以引用。互顶对只数 > 1 N 的接触。")

    csv_path = os.path.join(OUT, "diameter_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot(rows, os.path.join(OUT, "diameter_sweep.png"))
    print("已写出 %s" % csv_path)
    print("       %s" % os.path.join(OUT, "diameter_sweep.png"))
    return 0


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    series = (("coax_mm", "coaxiality (mm)"),
              ("force_N", "jaw-shaft normal force (N)"),
              ("jaw_jam_N", "jaw-jaw interference force (N)"))
    for a, (key, title) in zip(axes, series):
        for mode, label, color, marker in (("fit", "bore per diameter", "#c0392b", "o"),
                                           ("phi20", "fixed D20 jaws", "#2c3e50", "s")):
            sub = [r for r in rows if r["mode"] == mode]
            a.plot([r["D_mm"] for r in sub], [r[key] for r in sub],
                   marker + "-", color=color, label=label, linewidth=1.6)
        if key == "coax_mm":
            a.axhline(2.0, color="#999999", linestyle="--", linewidth=1)
        a.set_title(title, fontsize=10)
        a.set_xlabel("workpiece diameter (mm)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
