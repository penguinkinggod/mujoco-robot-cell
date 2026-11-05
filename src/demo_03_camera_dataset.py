"""第 3 步：机器人 + 腕部相机 + 采集数据集。

这一步是"具身智能"的真正起点：机器人不靠人告诉坐标，
而是靠相机看到画面，把"画面 + 关节角 + 动作"一起存下来——
数据集的形状就是 (图像, 关节角, 目标, 动作)。

运行：
    .venv\\Scripts\\python.exe demo_03_camera_dataset.py
产出：
    demo_03_views.png       （8 个采样时刻的腕部相机画面）
    demo_03_dataset.npz     （image / qpos / action / goal）
    demo_03_dataset.csv     （用 Excel 看数字）

这份 npz 后面可以直接喂给 LeRobot 那种框架做模仿学习。
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
rng = np.random.default_rng(0)

XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.01"/>
  <visual>
    <global offwidth="320" offheight="240"/>
  </visual>
  <worldbody>
    <light pos="0 -0.6 1.6" directional="true"/>
    <!-- 一台固定在工作台前的相机，就像真实产线上装的那台 -->
    <camera name="side" pos="0.18 -1.15 0.55" xyaxes="1 0 0 0 0.5 1" fovy="45"/>
    <geom name="table" type="plane" size="1 1 .1" rgba=".85 .85 .88 1"/>

    <body name="base" pos="0 0 0.04">
      <geom type="cylinder" size="0.06 0.04" rgba=".25 .25 .3 1"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="j1" type="hinge" axis="0 1 0" range="-100 100" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.35 0 0" size="0.025" rgba=".2 .45 .85 1"/>
        <body name="link2" pos="0.35 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-150 150" limited="true"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" rgba=".25 .6 .45 1"/>
          <body name="ee" pos="0.25 0 0">
            <geom type="sphere" size="0.03" rgba=".95 .75 .1 1"/>
            <site name="ee_site" pos="0 0 0" size="0.005"/>
          </body>
        </body>
      </body>
    </body>

    <body name="target_marker" pos="0 0 0.005" mocap="true">
      <geom type="cylinder" size="0.03 0.004" rgba=".9 .15 .15 .6"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
SITE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
MARKER = model.body("target_marker").id
MOCAP_ID = model.body_mocapid[MARKER]

renderer = mujoco.Renderer(model, height=240, width=320)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")

# 每个关节的下界 / 上界（形状都是 (2,)）
LO = model.jnt_range[:, 0].copy()
HI = model.jnt_range[:, 1].copy()


def fk(q):
    """给定两个关节角，返回末端位置。"""
    d = mujoco.MjData(model)
    d.qpos[:2] = q
    mujoco.mj_forward(model, d)
    return d.site_xpos[SITE].copy()


N_TRAJ = 12          # 采 12 条轨迹
N_STEP = 100         # 每条 100 步
images, qposs, actions, goals = [], [], [], []

print("开始采集：%d 条轨迹 x %d 步" % (N_TRAJ, N_STEP))
print("关节范围 j1 = [%.0f, %.0f] 度, j2 = [%.0f, %.0f] 度"
      % (np.degrees(LO[0]), np.degrees(HI[0]), np.degrees(LO[1]), np.degrees(HI[1])))

for traj in range(N_TRAJ):
    # 1) 先随机一个"目标关节角"，用它算出的末端位置当目标点（保证一定够得到）
    q_goal = rng.uniform(LO + 0.25 * (HI - LO), HI - 0.25 * (HI - LO))
    goal = fk(q_goal)

    # 2) 机器人从随机姿态出发
    mujoco.mj_resetData(model, data)
    data.qpos[:2] = rng.uniform(LO + 0.2 * (HI - LO), HI - 0.2 * (HI - LO))
    mujoco.mj_forward(model, data)

    # 3) 把红色圆盘挪到目标点，机器人"看得见"
    data.mocap_pos[MOCAP_ID] = [goal[0], 0.0, 0.006]
    mujoco.mj_forward(model, data)

    goals.append(goal)

    for step in range(N_STEP):
        # 看到目标 -> 算误差 -> 关节动一点（雅可比伪逆，和第 2 步同一套）
        ee = data.site_xpos[SITE].copy()
        err = goal - ee
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, SITE)
        dq = np.linalg.pinv(jacp[:, :2]) @ err * 0.35      # 0.35 = 平滑系数
        dq = np.clip(dq, -0.06, 0.06)

        data.qpos[:2] = np.clip(data.qpos[:2] + dq, LO, HI)
        mujoco.mj_forward(model, data)

        if step % 5 == 0:
            renderer.update_scene(data, camera=cam)
            images.append(renderer.render().copy())
            qposs.append(data.qpos[:2].copy())
            actions.append(dq.copy())

    # 4) 检查这条轨迹最后够到没有
    final_err = np.linalg.norm(goal - data.site_xpos[SITE])
    print("  轨迹 %2d  目标 (%+.2f, %+.2f)  末误差 %.4f m  %s"
          % (traj + 1, goal[0], goal[2], final_err,
             "OK" if final_err < 0.03 else "偏了"))

images = np.asarray(images, dtype=np.uint8)
qposs = np.asarray(qposs, dtype=np.float32)
actions = np.asarray(actions, dtype=np.float32)
goals = np.asarray(goals, dtype=np.float32)

print("\n数据集形状：")
print("  image  :", images.shape, images.dtype, "  <- (样本, 高, 宽, RGB)")
print("  qpos   :", qposs.shape, "       <- 两个关节角 (rad)")
print("  action :", actions.shape, "       <- 每步关节增量")
print("  goal   :", goals.shape, "        <- 每条轨迹的目标点 (x,y,z)")

npz = os.path.join(OUT, "demo_03_dataset.npz")
np.savez_compressed(npz, image=images, qpos=qposs, action=actions, goal=goals)
print("\n已保存 %s  (%.1f MB)" % (npz, os.path.getsize(npz) / 1e6))

csv = os.path.join(OUT, "demo_03_dataset.csv")
with open(csv, "w", encoding="utf-8") as f:
    f.write("sample,q1_rad,q2_rad,action_q1,action_q2\n")
    for i in range(len(qposs)):
        f.write("%d,%.4f,%.4f,%.5f,%.5f\n"
                % (i, qposs[i, 0], qposs[i, 1], actions[i, 0], actions[i, 1]))
print("已保存", csv)

# 拼一张 8 格图，看清"相机看到了什么"
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(12, 5.0))
    idx = np.linspace(0, len(images) - 1, 8).astype(int)
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(images[i])
        ax.set_title("sample %d  q=(%.2f, %.2f)" % (i, qposs[i, 0], qposs[i, 1]),
                     fontsize=8)
        ax.axis("off")
    fig.suptitle("wrist camera — this is what the robot 'sees'", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT, "demo_03_views.png")
    fig.savefig(out, dpi=110)
    print("已保存", out)
except ImportError:
    print("没装 matplotlib，跳过拼图")

print("OK")
