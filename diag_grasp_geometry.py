"""diag_grasp_geometry.py —— 夹爪的抓取几何：朝向、开口轴、接近方向。

为什么必须先测
-------------
SO-101 的 gripperframe site 在 home 位置的坐标轴【不是】末端朝下方向。
上一轮（Panda 时代）就是因为直接假设"末端朝下"导致所有抓取失败。

要抓一根沿 x 轴横放、Φ20 的圆柱，需要知道：
    1. 两个 pad 之间的连线方向（夹爪闭合轴）—— 必须垂直于工件轴线（x）
    2. 夹爪"伸出去"的方向（接近方向）—— 应该朝下（-z）才能从上方压下去
    3. 夹爪关节往哪个方向动是"闭合"（ctrl 增大还是减小）
    4. pad 的实际中心位置和尺寸

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_grasp_geometry.py
"""
import numpy as np
import mujoco

import scene_full as S

model, data = S.make_scene()

print("=" * 74)
print("夹爪抓取几何诊断")
print("=" * 74)
print()

SITE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
GRIP_BODY = model.site_bodyid[SITE]

mujoco.mj_forward(model, data)

print("gripperframe site：")
print("  世界位置 = %s" % np.round(data.site_xpos[SITE], 4))
R = data.site_xmat[SITE].reshape(3, 3)
print("  世界轴向（列向量 = site 局部 x/y/z 在世界里的方向）：")
for i, ax in enumerate("xyz"):
    print("    site %s -> %s" % (ax, np.round(R[:, i], 3)))
print()

# ---- 找 pad ----
pads = {}
for i in range(model.ngeom):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if n and "pad" in n.lower():
        pads[n] = i

print("pad 几何：")
for n, i in pads.items():
    print("  %-18s 世界位置 %s" % (n, np.round(data.geom_xpos[i], 4)))
    print("  %-18s 半尺寸 %s  contype=%d conaffinity=%d"
          % ("", np.round(model.geom_size[i], 4),
             model.geom_contype[i], model.geom_conaffinity[i]))
print()

if len(pads) >= 2:
    names = list(pads.keys())
    p0 = data.geom_xpos[pads[names[0]]]
    p1 = data.geom_xpos[pads[names[1]]]
    axis = p1 - p0
    dist = np.linalg.norm(axis)
    axis_n = axis / dist
    print("  %s -> %s 距离 = %.1f mm" % (names[0], names[1], dist * 1000))
    print("  闭合轴方向 = %s" % np.round(axis_n, 3))
    print()
    print("  这个方向必须【垂直于】工件轴线（工件沿 x）")
    print("  当前与 x 轴夹角 = %.1f deg"
          % (np.degrees(np.arccos(np.clip(abs(axis_n[0]), -1, 1)))))
    print("  -> 90° 才是理想（垂直）")

print()
print("=" * 74)
print("夹爪关节与执行器")
print("=" * 74)
print()
gj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
print("  gripper 关节 qposadr=%d range=[%.4f, %.4f]"
      % (model.jnt_qposadr[gj], *model.jnt_range[gj]))
for i in range(model.nu):
    aid_j = model.actuator_trnid[i][0]
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, aid_j)
    an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print("  act[%d] %-14s -> joint %-12s ctrl=[%.4f, %.4f]"
          % (i, an, n, *model.actuator_ctrlrange[i]))

print()
print("=" * 74)
print("闭合方向测试（实测，不猜）")
print("=" * 74)
print()
print("  %-10s %-14s %s" % ("ctrl", "qpos", "pad 间距(mm)"))
print("  " + "-" * 44)

gi = model.jnt_qposadr[gj]
for c in (-0.17, -0.05, 0.0, 0.5, 1.0, 1.74):
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qpos[gi] = c
    mujoco.mj_forward(model, d)
    if len(pads) >= 2:
        nm = list(pads.keys())
        dd = np.linalg.norm(d.geom_xpos[pads[nm[1]]] - d.geom_xpos[pads[nm[0]]])
        print("  %-10.3f %-14.4f %.1f" % (c, d.qpos[gi], dd * 1000))

print()
print("=" * 74)
print("结论：怎么抓这根轴")
print("=" * 74)
print("""
  需要同时满足：
    1. 闭合轴（两个 pad 的连线）垂直于工件轴线 x
       -> 闭合轴应该落在 y 或 z 方向
    2. 接近方向（从 gripper body 指向 pad 的方向）朝下
       -> 往下压才能从上方夹住躺着的轴
    3. 手腕要绕自身轴转一个角度来对齐

  下面的抓取姿态由 demo_08 用 IK + 姿态优化自动求，不手算。
""")
print("OK")
