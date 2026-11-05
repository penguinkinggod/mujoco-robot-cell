"""第 4 步：真·Franka Panda 七轴机械臂 —— 完整抓取流程。

前三步的机器人是我自己用 XML 捏的玩具。这一步换成真家伙：
MuJoCo Menagerie 官方的 Franka Emika Panda，7 个关节 + 平行夹爪 + 完整碰撞体。

和 demo_02 的关键区别（"做动画"和"做仿真"的分界线）：
    demo_02: 我直接改 qpos   -> 纯运动学，可以穿模，物体不受力
    demo_04: 我只给 ctrl（目标关节角 + 夹爪开合），剩下交给伺服和刚体动力学

整个流程是真实产线上抓取的标准五段式：
    1. move    移动到方块正上方（避障高度）
    2. down    竖直下降，让夹爪夹住方块
    3. close   夹爪闭合
    4. lift    抬起来
    5. hold    保持，检查有没有滑落

运行：
    D:\\robot-env\\Scripts\\python.exe demo_04_panda_grasp.py
产出：
    demo_04_panda.png   五段关键帧
    demo_04_panda.mp4   完整动画
    demo_04_panda.gif   （README 首屏用这个）

画面配置在 looks.py 里（纹理/材质/三点布光/天空盒/阴影）。
那部分只影响"看起来怎样"，不影响任何物理结果。
"""
import os
import numpy as np
import mujoco

import looks

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
PANDA = os.path.join(ROOT, "models", "panda", "panda.xml")

# ============ 1. 在官方模型上"加料"：工作台 + 方块 + 灯 + 相机 ============
spec = mujoco.MjSpec.from_file(PANDA)

# 先把"摄影棚"配好：纹理、材质、三点布光、天空盒、渲染质量
# 全部在 looks.py 里，这里一行调用。它不影响物理。
spec = looks.make_pretty(spec)

table = spec.worldbody.add_geom()
table.name = "table"
table.type = mujoco.mjtGeom.mjGEOM_PLANE
table.size = [1.4, 1.4, 0.1]
table.rgba = [0.30, 0.32, 0.36, 1.0]
table.material = "floor_mat"          # 棋盘格纹理

CUBE_XY = (0.55, 0.05)
CUBE_HALF = 0.0215
CUBE_Z = CUBE_HALF                      # 底面贴桌面

cube = spec.worldbody.add_body()
cube.name = "cube"
cube.pos = [CUBE_XY[0], CUBE_XY[1], CUBE_Z]
cg = cube.add_geom()
cg.name = "cube_geom"
cg.type = mujoco.mjtGeom.mjGEOM_BOX
cg.size = [CUBE_HALF] * 3
cg.rgba = [0.70, 0.15, 0.12, 1.0]
cg.material = "red_metal"             # 金属红材质
cg.mass = 0.04
cg.friction = [1.5, 0.08, 0.002]        # 摩擦给足，否则夹住也会滑
cube.add_freejoint()

model = spec.compile()
data = mujoco.MjData(model)

BID = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
       for n in ("hand", "left_finger", "right_finger", "cube")}
NCAM = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "side")
JNT_RANGE = model.jnt_range[:7].copy()

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

# 夹爪"抓取中心"相对 hand 原点的偏移，沿 hand 自身 z 轴。
#
# 这个值不是猜的，是量出来的（见 diag_gripper_geom.py）：
#     hand 原点              z = 0.6245
#     指根 (finger body)     z = 0.5661   -> 偏移 0.0584
#     指垫最低点             z = 0.5105   -> 偏移 0.1140
#     指垫长度                               0.0556 m (55.6 mm)
#
# 之前用 0.0584（指根）当"抓取中心"是错的：
# 指垫还往下伸 55.6 mm，所以真实接触中心在 0.0862 处。
# 用错这个值会导致"以为夹爪在方块中心，实际指垫已经插进桌面 24 mm"。
FINGER_OFFSET = 0.0862        # 指垫中心

print("模型加载：nq=%d  nu=%d  nbody=%d" % (model.nq, model.nu, model.nbody))
print("夹爪抓取中心 = hand 位置 + %.4f * hand 的 z 轴（指垫中心）" % FINGER_OFFSET)
print("指垫长度 %.1f mm，方块高度 %.1f mm"
      % ((0.1140 - 0.0584) * 1000, CUBE_HALF * 2000))


# ============ 2. 六维 IK：位置 + 姿态一起解 ============
# 只解位置不够——夹爪会歪着过去，夹不住方块。必须连姿态一起约束。
R_DES = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])   # hand 的 z 轴朝下 = 从上往下抓


def grasp_point(d):
    """当前夹爪抓取中心的世界坐标。"""
    R = d.xmat[BID["hand"]].reshape(3, 3)
    return d.xpos[BID["hand"]] + FINGER_OFFSET * R[:, 2]


def rot_error(R_cur, R_des):
    """两个旋转矩阵之间的误差，转成 3 维角速度向量。"""
    e = np.zeros(3)
    for i in range(3):
        e += np.cross(R_cur[:, i], R_des[:, i])
    return 0.5 * e


def ik(target_pos, q_init, iters=500, gain=0.6, lam=0.05):
    """阻尼最小二乘 IK，6 维误差：[位置 3, 姿态 3]。"""
    q = q_init.copy()
    d = mujoco.MjData(model)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.ones(6)
    for _ in range(iters):
        d.qpos[:7] = q
        mujoco.mj_forward(model, d)
        e_pos = target_pos - grasp_point(d)
        e_rot = rot_error(d.xmat[BID["hand"]].reshape(3, 3), R_DES)
        err = np.concatenate([e_pos, 0.35 * e_rot])
        if np.linalg.norm(err) < 1e-3:
            break
        mujoco.mj_jacBody(model, d, jacp, jacr, BID["hand"])
        J = np.vstack([jacp[:, :7], 0.35 * jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        q = np.clip(q + gain * dq, JNT_RANGE[:, 0], JNT_RANGE[:, 1])
    return q, float(np.linalg.norm(err))


Q_HOME = data.qpos[:7].copy()

# 抓取高度必须满足几何约束：
#   指垫中心 = hand + 0.0862，指垫下端还要再往下 27.8 mm
#   所以指垫中心至少要在桌面上方 27.8 mm + 安全间隙
PAD_HALF_LEN = (0.1140 - 0.0584) / 2      # 27.8 mm
SAFE_CLEARANCE = 0.004                     # 4 mm 安全间隙
GRASP_Z = PAD_HALF_LEN + SAFE_CLEARANCE    # 指垫中心的最低安全高度
print("指垫半长 %.1f mm + 间隙 %.1f mm => 抓取中心最低 z = %.4f"
      % (PAD_HALF_LEN * 1000, SAFE_CLEARANCE * 1000, GRASP_Z))

PLAN = [
    ("move",  np.array([CUBE_XY[0], CUBE_XY[1], GRASP_Z + 0.22])),
    ("down",  np.array([CUBE_XY[0], CUBE_XY[1], GRASP_Z])),
    ("close", None),
    ("lift",  np.array([CUBE_XY[0], CUBE_XY[1], GRASP_Z + 0.20])),
    ("hold",  None),
]

q_prev = Q_HOME.copy()
SEQ = []
for name, target in PLAN:
    if target is None:
        SEQ.append((name, q_prev.copy(), 0.0))
        continue
    q_sol, e = ik(target, q_prev)
    SEQ.append((name, q_sol, e))
    print("IK %-5s 目标 %-24s 残差 %.5f  m   关节 %s"
          % (name, np.round(target, 2), e, np.round(q_sol, 2)))
    q_prev = q_sol


# ============ 2.5 执行前先检查可达性 ============
# 这一步很重要：IK 解得出关节角，不代表机器人走得到。
# 会撞桌面、关节超限、力矩饱和、伺服误差过大 —— 都在这里拦住。
print()
print("=" * 70)
print("可达性检查")
print("=" * 70)

from reachability import check_grasp_pose

all_ok = True
for name, target in PLAN:
    if target is None:
        continue
    q_for_check = dict(SEQ and [(n, q) for n, q, _ in SEQ])[name]
    rep = check_grasp_pose(model, q_for_check, target,
                           hand_name="hand",
                           finger_offset=FINGER_OFFSET,
                           table_z=0.0,
                           expected_contacts=("cube_geom",))
    status = "OK" if rep.ok else "拦截"
    print("  %-6s %-8s 夹爪最低z=%+.4f  间隙%+.1fmm  力矩%.0f%%"
          % (name, status,
             rep.metrics.get("夹爪最低点z(m)", float("nan")),
             rep.metrics.get("离桌面间隙(m)", float("nan")) * 1000,
             rep.metrics.get("最大力矩占用比", 0) * 100))
    if not rep.ok:
        all_ok = False
        for kind, msg in rep.problems:
            print("           [%s] %s" % (kind, msg))

print()
if not all_ok:
    print("有路段不可达，但继续执行（看看会怎样）。实际工程里这里应该报错退出。")
else:
    print("全部路段可达。")


# ============ 3. 按段执行，只喂 ctrl ============
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
SEG_STEPS = {"move": 800, "down": 500, "close": 300, "lift": 600, "hold": 400}

renderer = mujoco.Renderer(model, height=640, width=960)
looks.enable_scene_flags(renderer)          # 开阴影/天空盒/雾/反射

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.42, 0.02, 0.22]   # 看向机器人腰部和方块之间
cam.distance = 1.45
cam.azimuth = 145
cam.elevation = -16

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
data.ctrl[:7] = Q_HOME
data.ctrl[7] = GRIP_OPEN
data.qpos[7:9] = [0.04, 0.04]        # 夹爪初始张开

frames, snaps, labels = [], [], []
q_from, grip_from = Q_HOME.copy(), GRIP_OPEN

for name, q_to, _ in SEQ:
    n = SEG_STEPS[name]
    grip_to = GRIP_CLOSED if name in ("close", "lift", "hold") else GRIP_OPEN
    for i in range(n):
        a = (i + 1) / n
        a = 0.5 - 0.5 * np.cos(np.pi * a)          # 余弦缓动，起停无冲击
        data.ctrl[:7] = (1 - a) * q_from + a * q_to
        data.ctrl[7] = (1 - a) * grip_from + a * grip_to
        mujoco.mj_step(model, data)
        if i % 12 == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    renderer.update_scene(data, camera=cam)
    snaps.append(renderer.render())
    labels.append(name)

    width = float(np.linalg.norm(data.xpos[BID["left_finger"]] - data.xpos[BID["right_finger"]]))
    print("%-5s 结束：抓取中心 z=%.3f  夹爪开口=%.3f m  方块 z=%.3f m  接触数=%d"
          % (name, grasp_point(data)[2], width, data.xpos[BID["cube"]][2], data.ncon))

    q_from, grip_from = q_to.copy(), grip_to


# ============ 4. 结果判定 ============
cube_lift = float(data.xpos[BID["cube"]][2] - CUBE_Z)
width = float(np.linalg.norm(data.xpos[BID["left_finger"]] - data.xpos[BID["right_finger"]]))
ok = cube_lift > 0.05

print("\n=== 结果 ===")
print("方块被抬高   : %.3f m" % cube_lift)
print("夹爪最终开口 : %.3f m（方块宽 %.3f m）" % (width, 2 * CUBE_HALF))
print("判定         : %s" % ("抓起来了 OK" if ok else "没夹住 FAIL"))


# ============ 5. 存图 / 存视频 ============
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(snaps), figsize=(3.0 * len(snaps), 3.0))
    for ax, img, lab in zip(axes, snaps, labels):
        ax.imshow(img); ax.set_title(lab, fontsize=10); ax.axis("off")
    fig.suptitle("Franka Panda grasp — real dynamics, ctrl setpoints only", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT, "demo_04_panda.png")
    fig.savefig(out, dpi=105)
    print("已保存", out)
except ImportError:
    print("没装 matplotlib")

try:
    import imageio.v2 as imageio
    mp4 = os.path.join(OUT, "demo_04_panda.mp4")
    imageio.mimsave(mp4, frames, fps=30, macro_block_size=1)
    print("已保存 %s  (%d 帧, %.1f 秒)" % (mp4, len(frames), len(frames) / 30))

    gif = os.path.join(OUT, "demo_04_panda.gif")
    step = 2                                     # 隔帧取样，压体积
    small = frames[::step]
    imageio.mimsave(gif, small, fps=15, loop=0)
    print("已保存 %s  (%.1f MB)" % (gif, os.path.getsize(gif) / 1e6))
except Exception as e:
    print("视频没存成：", type(e).__name__, e)

print("OK")
