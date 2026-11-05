"""第 2 步：二连杆机械臂 —— 让末端"够到"一个目标点。

运行：
    .venv\\Scripts\\python.exe demo_02_arm.py
产出：
    demo_02_arm.png    （三帧：起始 / 中途 / 到达）
    demo_02_arm.mp4    （整段动画）

这里第一次出现真·机器人问题：
    电机转多少度，末端才到那个位置？  -> 这就是"运动学"
    我用的是最笨但最有效的方法：雅可比伪逆（Jacobian pseudo-inverse）。
    你不需要现在推导公式，只要看懂这段在做什么：
        1) 现在末端在哪（正运动学）
        2) 末端离目标差多少（误差）
        3) 每个关节动一点，末端会往哪走（雅可比矩阵）
        4) 反解出"关节该动多少"，动一小步
        5) 重复 300 次
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <worldbody>
    <light pos="0 -1 2"/>
    <camera name="front" pos="0 -2.2 0.8" xyaxes="1 0 0 0 0.35 1"/>
    <geom type="plane" size="2 2 .1" rgba=".92 .92 .92 1"/>
    <body name="base" pos="0 0 0.05">
      <geom type="cylinder" size="0.07 0.05" rgba=".3 .3 .35 1"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 1 0" range="-170 170"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.03" rgba=".2 .45 .85 1"/>
        <body name="link2" pos="0.4 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-170 170"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.025" rgba=".25 .6 .45 1"/>
          <body name="ee" pos="0.3 0 0">
            <geom type="sphere" size="0.04" rgba=".95 .75 .1 1"/>
            <site name="ee_site" pos="0 0 0" size="0.01"/>
          </body>
        </body>
      </body>
    </body>
    <body name="goal" pos="0.35 0 0.42">
      <geom type="sphere" size="0.035" rgba=".9 .2 .2 .85" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-60 60"/>
    <motor joint="j2" ctrlrange="-40 40"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
data.qpos[:] = [np.radians(-20), np.radians(-70)]   # 起始姿态
mujoco.mj_forward(model, data)

SITE_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
GOAL = model.body("goal").pos.copy()

renderer = mujoco.Renderer(model, height=400, width=520)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.2, 0, 0.35]
cam.distance = 1.9
cam.elevation = -12
cam.azimuth = 90

jacp = np.zeros((3, model.nv))
frames, snapshots = [], []

ITERS = 400
STEP_GAIN = 0.01          # 每次迭代允许的最大关节增量（rad）
print("目标点:", np.round(GOAL, 3))

for it in range(ITERS):
    # 1) 正运动学：末端现在在哪
    ee = data.site_xpos[SITE_ID].copy()
    err = GOAL - ee

    # 2) 雅可比：关节速度 -> 末端速度
    mujoco.mj_jacSite(model, data, jacp, None, SITE_ID)
    J = jacp[:, :2]        # 只取两个转动关节对应的两列

    # 3) 伪逆反解：末端要走 err，关节该走多少
    dq = np.linalg.pinv(J) @ err

    # 4) 限幅，动一小步
    n = np.linalg.norm(dq)
    if n > STEP_GAIN:
        dq = dq / n * STEP_GAIN

    # 5) 直接写位置（这里偷懒跳过动力学，是"运动学仿真"；换成 motor 才是真动力学）
    data.qpos[:2] += dq
    mujoco.mj_forward(model, data)

    if it % 8 == 0:
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    if it in (0, ITERS // 2, ITERS - 1):
        snapshots.append(renderer.render())

ee_final = data.site_xpos[SITE_ID].copy()
dist = np.linalg.norm(GOAL - ee_final)
print("末端最终位置:", np.round(ee_final, 3))
print("离目标距离: %.4f m" % dist)
print("末端在 goal 点上 =", dist < 0.01)

# ---------- 存图 ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = ["start", "mid", "reached (err=%.3fm)" % dist]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, img, t in zip(axes, snapshots, titles):
        ax.imshow(img); ax.set_title(t); ax.axis("off")
    fig.tight_layout()
    out = os.path.join(HERE, "demo_02_arm.png")
    fig.savefig(out, dpi=110)
    print("已保存", out)
except ImportError:
    print("没装 matplotlib，跳过拼图")

# ---------- 存 mp4 ----------
try:
    import imageio.v2 as imageio
    mp4 = os.path.join(HERE, "demo_02_arm.mp4")
    imageio.mimsave(mp4, frames, fps=25, macro_block_size=1)
    print("已保存", mp4, "共", len(frames), "帧")
except Exception as e:
    print("mp4 没存成：", type(e).__name__, e)

print("OK")
