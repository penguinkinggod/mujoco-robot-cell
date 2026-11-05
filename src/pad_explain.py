"""pad_explain.py -- 把「指垫」是什么、为什么架不住，画成一张图。

四个画面，同一次仿真：
  (1) 整体机位：机械臂伸到工件上方
  (2) 夹爪特写：两片指垫涂成橙色，这就是「指垫」
  (3) 夹紧瞬间：轴已经被挤歪 41 deg
  (4) 抬起后：轴吊在两片平板下面，歪到 56 deg

跑法：
    D:\\robot-env\\Scripts\\python.exe pad_explain.py
产物：pad_explain.png
"""
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import diag_feed_chuck as D
from cnfont import setup_cn_font
from demo_08_pick_and_load import Arm, GRIP_OPEN, WRIST_ROLL_GRASP

setup_cn_font()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
OUT = os.path.join(OUT, "pad_explain.png")

GRIP = 0.17          # exp_plain_cyl.py 扫出来的稳定夹紧量
PAD_RGBA = [0.85, 0.08, 0.12, 1.0]      # 鲜红 = 指垫（SO-101 本体是黄色，红才分得开）


def make_cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = lookat
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c


def main():
    model = D.build_model()
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, height=560, width=760)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True

    pads = [g for g in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            .endswith("_pad")]

    arm = Arm(model, data)
    sj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
    sq = model.jnt_qposadr[sj]

    mujoco.mj_resetData(model, data)
    data.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    data.ctrl[4] = WRIST_ROLL_GRASP
    data.ctrl[arm.grip_act] = GRIP_OPEN
    for _ in range(600):
        for k in range(5):
            data.ctrl[k] = data.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)

    rest = data.qpos[sq:sq + 3].copy()
    for tgt, grip, ns in (
            ([rest[0], rest[1], rest[2] + 0.085], GRIP_OPEN, 450),
            ([rest[0], rest[1], rest[2] + 0.002], GRIP_OPEN, 380),
            ([rest[0], rest[1], rest[2] + 0.002], GRIP, 320)):
        arm.move_to(np.array(tgt), n_steps=ns, grip=grip,
                    wrist_roll=WRIST_ROLL_GRASP)

    at_close = data.qpos[sq:sq + 3].copy()
    pc = arm.pad_center()
    tilt_close = D.tilt_deg(data.qpos[sq + 3:sq + 7])

    wide = make_cam(pc, 0.34, -50, -28)
    near = make_cam(pc, 0.135, -70, -14)
    side = make_cam(pc, 0.155, 25, -8)
    plan = make_cam(pc, 0.30, 0, -86)        # 俯视：看轴歪了多少最直观

    shots = {}
    for k in pads:
        model.geom_rgba[k] = PAD_RGBA
    rend.update_scene(data, camera=near)
    shots["pads"] = rend.render().copy()
    rend.update_scene(data, camera=wide)
    shots["wide"] = rend.render().copy()
    rend.update_scene(data, camera=plan)
    shots["close"] = rend.render().copy()      # 夹紧瞬间，抬升之前

    arm.move_to(np.array([at_close[0], at_close[1], at_close[2] + 0.08]),
                n_steps=420, grip=GRIP)
    tilt_lift = D.tilt_deg(data.qpos[sq + 3:sq + 7])
    side_lift = make_cam(arm.pad_center(), 0.30, 25, -10)
    rend.update_scene(data, camera=side_lift)
    shots["lift"] = rend.render().copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.6))
    for ax, (key, title) in zip(
            (axes[0][0], axes[0][1], axes[1][0], axes[1][1]),
            (("pads", "① 指垫特写｜红色两块 = 指垫，尺寸 10.4 × 16 mm"),
             ("close", "② 俯视：一夹紧就歪 %.0f°，已经偏离卡盘轴线"
              % tilt_close),
             ("lift", "③ 抬起来歪 %.0f°｜轴吊在平板下方，靠摩擦悬着" % tilt_lift),
             (None, "④ 为什么：线接触管不住转动"))):
        if key is not None:
            ax.imshow(shots[key])
            ax.set_title(title, fontsize=11.5, pad=6)
        ax.axis("off")
    draw_schematic(axes[1][1])

    fig.suptitle("「指垫」是夹爪指尖那两块平板 —— 它夹不住圆柱，不是因为滑，是因为线接触",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT, dpi=115, facecolor="white")
    plt.close(fig)
    print("PAD_PNG %s" % OUT)
    print("TILT close=%.1f lift=%.1f" % (tilt_close, tilt_lift))
    return 0


def draw_schematic(ax):
    """侧视示意：两片平板 + 圆柱 + 两条接触线 + 转动力矩。"""
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.15)
    ax.set_aspect("equal")

    # 圆柱
    circ = plt.Circle((0, 0), 0.62, facecolor="#eef2f7",
                      edgecolor="#4a5568", linewidth=2, zorder=2)
    ax.add_patch(circ)

    # 两片平板
    for sx in (-1, 1):
        x0 = sx * 0.92 if sx < 0 else 0.62
        ax.add_patch(plt.Rectangle((x0, -0.55), 0.30, 1.10,
                                   facecolor="#c9302c", edgecolor="#7a1a17",
                                   linewidth=1.5, zorder=3,
                                   alpha=0.9))
        ax.annotate("", xy=(sx * 0.77, -0.28), xytext=(sx * 0.77, -0.88),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2,
                                    color="#7a1a17"), zorder=4)

    # 两条接触线（点）
    for sx in (-1, 1):
        ax.plot([sx * 0.62], [0], marker="o", markersize=9,
                color="#ffd400", markeredgecolor="#7a1a17",
                markeredgewidth=1.4, zorder=6)
    ax.annotate("接触只有\n一条线", xy=(0.62, 0.0), xytext=(0.95, 0.62),
                fontsize=10.5, ha="center",
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#7a1a17",
                                connectionstyle="arc3,rad=0.25"))

    # 转动力矩：轴想绕接触线转下去
    tt = np.linspace(-0.35, 0.85, 60)
    ax.plot(0.30 + 0.55 * np.cos(tt), 0.0 - 0.02 + 0.42 * np.sin(tt),
            color="#1f6fb2", lw=2.2, zorder=5)
    ax.annotate("", xy=(0.72, -0.18), xytext=(0.86, 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1f6fb2"),
                zorder=6)
    ax.text(-0.02, -1.10,
            "平板只能压住法向：压力再大，也挡不住轴绕接触线转下去\n"
            "要稳住它，得让夹具「包住」圆柱（V 口 / 圆弧爪 / 三点托）",
            fontsize=10.2, ha="center", va="bottom", color="#2d3748")
    ax.set_axis_off()


if __name__ == "__main__":
    raise SystemExit(main())
