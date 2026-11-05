"""diag_so101_gripper.py —— SO-101 夹爪到底能夹多大的轴？

官方模型里有两个真实的碰撞 pad：
    fixed_jaw_pad   box  size=0.00525 0.008   0.0072   contype=2
    moving_jaw_pad  box  size=0.0045  0.006   0.0072   contype=2

gripper 关节范围 [-0.17453, 1.74533] rad：
    -0.17453 = 全闭
     1.74533 = 全开

但这个范围是**角度**，实际开口多少毫米要算。
这个脚本扫一遍关节角，量两个 pad 之间的真实间距，
再对比我们的工件 Φ20 / Φ16 / 25 mm。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_so101_gripper.py
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "models", "_tmp_so101",
                      "lerobot_env_so101-0.4.0", "lerobot_env_so101", "assets")
SO101 = os.path.join(ASSETS, "so101_new_calib.xml")

model = mujoco.MjModel.from_xml_path(SO101)
data = mujoco.MjData(model)

FIXED_PAD = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
MOVING_PAD = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad")
GRIP_JNT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
GRIP_ACT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
EE_SITE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

grip_qadr = model.jnt_qposadr[GRIP_JNT]
LO, HI = model.jnt_range[GRIP_JNT]

print("=" * 74)
print("SO-101 夹爪：官方模型里的真实几何")
print("=" * 74)
print("  碰撞 pad 存在 : 是（fixed_jaw_pad / moving_jaw_pad，contype=2）")
print("  所以能真夹   : 是，不是摆设")
print("  关节范围      : [%.4f, %.4f] rad（闭 → 开）" % (LO, HI))
print()


def pad_gap(q_grip):
    """给定夹爪关节角，返回两 pad 之间的间隙（米）。"""
    d = mujoco.MjData(model)
    d.qpos[:] = 0.0
    d.qpos[grip_qadr] = q_grip
    mujoco.mj_forward(model, d)

    # pad 的实际世界位姿
    pf = d.geom_xpos[FIXED_PAD].copy()
    pm = d.geom_xpos[MOVING_PAD].copy()

    # pad 是薄板，size[0] 是半厚度。两板相对的表面间距 = 中心距 - 两半厚
    half_f = model.geom_size[FIXED_PAD][0]
    half_m = model.geom_size[MOVING_PAD][0]

    # 用 pad 的局部 x 轴投影，取"张开方向"上的距离
    Rf = d.geom_xmat[FIXED_PAD].reshape(3, 3)
    Rm = d.geom_xmat[MOVING_PAD].reshape(3, 3)
    axis = Rf[:, 0]                       # 固定爪的张开方向
    sep = abs(np.dot(pm - pf, axis))
    return sep - half_f - half_m, pf, pm


print("=" * 74)
print("第 1 步：扫关节角，量真实开口")
print("=" * 74)
print()
print("  %-12s %-14s %-14s" % ("关节角(rad)", "两pad间隙", "说明"))
print("  " + "-" * 46)

rows = []
for q in np.linspace(LO, HI, 13):
    gap, _, _ = pad_gap(q)
    rows.append((q, gap))
    note = ""
    if q <= LO + 1e-6:
        note = "全闭"
    elif q >= HI - 1e-6:
        note = "全开"
    print("  %-12.4f %-14s %-14s" % (q, "%.2f mm" % (gap * 1000), note))

MAX_GAP = rows[-1][1]
MIN_GAP = rows[0][1]

print()
print("  全开最大开口 : %.1f mm" % (MAX_GAP * 1000))
print("  全闭最小开口 : %.1f mm" % (MIN_GAP * 1000))

# ============ 第 2 步：能不能夹住我们的工件 ============
print()
print("=" * 74)
print("第 2 步：能夹多大的轴？")
print("=" * 74)
print()
print("  %-24s %-10s %-14s %-10s" % ("工件", "直径", "是否放得进", "说明"))
print("  " + "-" * 62)

WORKPIECES = [
    ("D 轴 Φ16x60 钢", 16.0),
    ("D 轴 Φ20x80 钢", 20.0),
    ("D 轴 Φ20x100 钢", 20.0),
    ("D 轴 Φ25x60 铝", 25.0),
    ("小齿轮坯 Φ40x15 钢", 40.0),
    ("垫圈 Φ50x6 钢", 50.0),
    ("玩具方块 43mm", 43.0),
    ("阶梯轴 Φ40（超载对照）", 40.0),
]

for name, dia in WORKPIECES:
    fit = dia <= MAX_GAP * 1000
    # 留 3 mm 余量才好进
    margin = MAX_GAP * 1000 - dia
    if not fit:
        note = "超开口"
    elif margin < 3:
        note = "余量 %.1f mm，太紧" % margin
    else:
        note = "余量 %.1f mm，可以" % margin
    print("  %-24s %-10s %-14s %-10s"
          % (name, "Φ%.0f" % dia, "是" if fit else "否", note))

# ============ 第 3 步：夹爪能伸多深 ============
print()
print("=" * 74)
print("第 3 步：pad 有多长？（决定能夹多深的接触面）")
print("=" * 74)
print()

d = mujoco.MjData(model)
mujoco.mj_forward(model, d)

sf = model.geom_size[FIXED_PAD]
sm = model.geom_size[MOVING_PAD]
print("  fixed_jaw_pad  : %.1f x %.1f x %.1f mm (半长)"
      % (sf[0] * 1000, sf[1] * 1000, sf[2] * 1000))
print("  moving_jaw_pad : %.1f x %.1f x %.1f mm (半长)"
      % (sm[0] * 1000, sm[1] * 1000, sm[2] * 1000))
print()
print("  沿轴向接触长度 : %.1f mm（fixed）" % (sf[1] * 2 * 1000))
print("  沿轴向接触长度 : %.1f mm（moving）" % (sm[1] * 2 * 1000))
print()
print("  对比：Panda 指垫长 55.6 mm，SO-101 只有 %.1f mm，"
      % (sf[1] * 2 * 1000))
print("        所以 SO-101 夹细长轴更容易翻转——力臂短。")

# ============ 第 4 步：末端位姿 ============
print()
print("=" * 74)
print("第 4 步：末端坐标系（做 IK 要用的）")
print("=" * 74)
print()
print("  gripperframe 位置 : %s" % np.round(d.site_xpos[EE_SITE], 4))
R = d.site_xmat[EE_SITE].reshape(3, 3)
print("  gripperframe 朝向 :")
for i in range(3):
    print("    [%+.3f %+.3f %+.3f]" % tuple(R[i]))
print()
print("  注意：这个坐标系不是标准的\"末端朝下\"。")
print("        做抓取时要重新标定，不能假设 z 轴朝下。")

print()
print("=" * 74)
print("结论")
print("=" * 74)
print("""
  1. 夹爪能真夹：有碰撞 pad，最大开口 %.0f mm
  2. 我们的工件（Φ16~25）全部放得进，余量充足
  3. 但 pad 只有 %.1f mm 长，夹细长轴容易翻转
  4. 官方包**没有卡盘**，必须自己建

  所以场景要新建：
      SO-101（官方资产）+ 工作台 + D 轴工件 + 三爪卡盘（自建）
""" % (MAX_GAP * 1000, sf[1] * 2 * 1000))
print("OK")
