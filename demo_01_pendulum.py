"""第 1 步：单摆 —— 看"物理自己会动"和"控制器让它听话"的区别。

运行：
    .venv\\Scripts\\python.exe demo_01_pendulum.py
产出：
    demo_01_pendulum.png  （左：不加控制，右：加 PD 控制）
    demo_01_data.csv      （时间 / 角度，可以用 Excel 打开）

这一段是机器人控制里最重要的一个概念：
    没有控制器 -> 系统按物理规律自己演化，通常不会停在你要的位置。
    有控制器   -> 你每一帧读状态、算误差、给力矩，它就能稳。
后面所有机械臂、人形、四足，控制的骨架都是这三行：
    误差 = 目标 - 当前
    力矩 = Kp * 误差 - Kd * 速度
    交给仿真器执行
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. 造一个单摆 ----------
# 一个只能绕 y 轴转的铰链（hinge），杆子长 0.5m，末端一个小球
XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <light pos="0 1 1"/>
    <camera name="side" pos="0 -1.6 0.3" xyaxes="1 0 0 0 0.3 1"/>
    <geom type="plane" size="1 1 .1" rgba=".9 .9 .9 1"/>
    <body name="link" pos="0 0 0.6">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba=".2 .4 .8 1"/>
      <body name="tip" pos="0 0 -0.5">
        <geom type="sphere" size="0.06" rgba=".9 .3 .2 1" mass="1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

DT = model.opt.timestep
N_STEPS = int(4.0 / DT)          # 仿真 4 秒
TARGET_ANGLE = np.pi / 2         # 目标：杆子抬到水平

# 重力对铰链的力矩，你不需要自己推公式：MuJoCo 已经算好了。
# data.qfrc_bias 就是"为了抵消重力+哥氏力，各关节需要多大力矩"。
# 这就是"逆动力学"，后面做机械臂控制会反复用到它。

def run(mode: str):
    """mode: 'none' 不加控制 / 'pd' 只有 PD / 'pd_ff' PD + 重力前馈"""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = np.radians(-60.0)   # 从"斜着下垂"开始，让它有东西可摆
    renderer = mujoco.Renderer(model, height=360, width=480)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0, 0, 0.35]
    cam.distance = 1.6
    cam.elevation = -10
    cam.azimuth = 90

    ts, angs = [], []
    KP, KD = 25.0, 3.0           # 两个可调参数，改它们看效果变化

    for i in range(N_STEPS):
        if mode == "none":
            data.ctrl[0] = 0.0
        else:
            err = TARGET_ANGLE - data.qpos[0]
            vel = data.qvel[0]
            u = KP * err - KD * vel
            if mode == "pd_ff":
                # 前馈：先问 MuJoCo "托住这个姿势要多少力"，再叠 PD 修正误差
                u += float(data.qfrc_bias[0])
            data.ctrl[0] = u
        mujoco.mj_step(model, data)
        if i % 20 == 0:
            ts.append(data.time)
            angs.append(float(data.qpos[0]))

    renderer.update_scene(data, camera=cam)
    return np.array(ts), np.array(angs), np.degrees(angs), renderer.render()

t_free, a_free, a_free_deg, img_free = run("none")
t_ctl,  a_ctl,  a_ctl_deg,  img_ctl  = run("pd")
t_ff,   a_ff,   a_ff_deg,   img_ff   = run("pd_ff")

print("不加控制：4 秒后角度 = %.1f 度，末端一直在摆" % a_free_deg[-1])
print("只用 PD ：4 秒后角度 = %.1f 度（目标 %.1f，差 %.1f 度 —— 这就是静差）"
      % (a_ctl_deg[-1], np.degrees(TARGET_ANGLE), np.degrees(TARGET_ANGLE) - a_ctl_deg[-1]))
print("PD+前馈 ：4 秒后角度 = %.1f 度（差 %.2f 度）"
      % (a_ff_deg[-1], abs(np.degrees(TARGET_ANGLE) - a_ff_deg[-1])))

# ---------- 2. 存数据 ----------
csv_path = os.path.join(HERE, "demo_01_data.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("t,angle_no_control_deg,angle_pd_deg,angle_pd_feedforward_deg\n")
    n = min(len(t_free), len(t_ctl), len(t_ff))
    for i in range(n):
        f.write("%.4f,%.2f,%.2f,%.2f\n" % (t_free[i], a_free_deg[i], a_ctl_deg[i], a_ff_deg[i]))
print("已保存", csv_path)

# ---------- 3. 画一张对比图 ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    axes[0].plot(t_free, a_free_deg, label="no control", color="tab:red")
    axes[0].plot(t_ctl, a_ctl_deg, label="PD only", color="tab:blue")
    axes[0].plot(t_ff, a_ff_deg, label="PD + gravity feedforward", color="tab:green")
    axes[0].axhline(np.degrees(TARGET_ANGLE), ls="--", c="k", lw=1, label="target")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("angle (deg)")
    axes[0].set_title("same pendulum, three control laws")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)

    for ax, img, t in zip(axes[1:], [img_free, img_ctl, img_ff],
                          ["no control", "PD only", "PD + feedforward"]):
        ax.imshow(img); ax.set_title(t + " (t=4s)"); ax.axis("off")

    fig.tight_layout()
    out = os.path.join(HERE, "demo_01_pendulum.png")
    fig.savefig(out, dpi=110)
    print("已保存", out)
except ImportError:
    print("没装 matplotlib，跳过画图（uv pip install --python .venv\\Scripts\\python.exe matplotlib）")

print("OK")
