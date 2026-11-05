"""用可达性检查扫一遍抓取高度，找出安全区间。

跑法：
    D:\\robot-env\\Scripts\\python.exe check_reach.py
"""
import os
import numpy as np
import mujoco

import reachability

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
PANDA = os.path.join(ROOT, "models", "panda", "panda.xml")

spec = mujoco.MjSpec.from_file(PANDA)
t = spec.worldbody.add_geom()
t.name = "table"
t.type = mujoco.mjtGeom.mjGEOM_PLANE
t.size = [1.4, 1.4, 0.1]

cube = spec.worldbody.add_body()
cube.name = "cube"
cube.pos = [0.55, 0.05, 0.0215]
cg = cube.add_geom()
cg.name = "cube_geom"
cg.type = mujoco.mjtGeom.mjGEOM_BOX
cg.size = [0.0215] * 3
cg.mass = 0.04
cube.add_freejoint()

model = spec.compile()
data = mujoco.MjData(model)

HAND = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
JNT_RANGE = model.jnt_range[:7].copy()
FINGER_OFFSET = 0.0584

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
Q_HOME = data.qpos[:7].copy()

R_DES = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])


def grasp_point(d):
    R = d.xmat[HAND].reshape(3, 3)
    return d.xpos[HAND] + FINGER_OFFSET * R[:, 2]


def rot_error(R_cur, R_des):
    e = np.zeros(3)
    for i in range(3):
        e += np.cross(R_cur[:, i], R_des[:, i])
    return 0.5 * e


def ik(target_pos, q_init, iters=500, gain=0.6, lam=0.05):
    q = q_init.copy()
    d = mujoco.MjData(model)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    err = np.ones(6)
    for _ in range(iters):
        d.qpos[:7] = q
        mujoco.mj_forward(model, d)
        e_pos = target_pos - grasp_point(d)
        e_rot = rot_error(d.xmat[HAND].reshape(3, 3), R_DES)
        err = np.concatenate([e_pos, 0.35 * e_rot])
        if np.linalg.norm(err) < 1e-3:
            break
        mujoco.mj_jacBody(model, d, jacp, jacr, HAND)
        J = np.vstack([jacp[:, :7], 0.35 * jacr[:, :7]])
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), err)
        q = np.clip(q + gain * dq, JNT_RANGE[:, 0], JNT_RANGE[:, 1])
    return q, float(np.linalg.norm(err))


XY = (0.55, 0.05)

print("=" * 74)
print("扫描抓取高度：从 z=0.005 到 z=0.20，每 5 mm 一个点")
print("=" * 74)
print("  %-8s %-10s %-12s %-12s %-10s %-8s"
      % ("目标z", "IK残差", "夹爪最低点", "离桌面", "力矩占用", "判定"))
print("  " + "-" * 66)

first_ok = None
for z_mm in range(5, 205, 5):
    z = z_mm / 1000.0
    tgt = np.array([XY[0], XY[1], z])
    q, ik_err = ik(tgt, Q_HOME)
    rep = reachability.check_grasp_pose(model, q, tgt,
                                        hand_name="hand",
                                        finger_offset=FINGER_OFFSET,
                                        table_z=0.0)

    lowest = rep.metrics.get("夹爪最低点z(m)", float("nan"))
    clearance = rep.metrics.get("离桌面间隙(m)", float("nan"))
    torque = rep.metrics.get("最大力矩占用比", float("nan"))
    verdict = "OK" if rep.ok else rep.problems[0][0]
    if rep.ok and first_ok is None:
        first_ok = z_mm

    print("  %-8.3f %-10.5f %-12.4f %-12.1f %-10.2f %-8s"
          % (z, ik_err, lowest, clearance * 1000, torque, verdict))

print()
print("=" * 74)
print("二分查找最低安全高度")
print("=" * 74)
z_safe, q_safe, rep_safe = reachability.find_safe_grasp_height(
    model, XY, ik, z_lo=0.005, z_hi=0.20,
    hand_name="hand", finger_offset=FINGER_OFFSET, table_z=0.0,
    q_init=Q_HOME)

if z_safe is None:
    print("  高处也不可达，需要换放置位置")
else:
    print("  最低安全抓取高度 : %.4f m (%.1f mm)" % (z_safe, z_safe * 1000))
    print("  扫描粗测          : %.1f mm" % first_ok)
    print()
    print("  该高度下的详细指标：")
    print(rep_safe.table())
    print()
    print("  %s" % rep_safe.summary())

print()
print("=" * 74)
print("对照：现在 demo_04 用的高度 z=0.03")
print("=" * 74)
tgt_bad = np.array([XY[0], XY[1], 0.03])
q_bad, e_bad = ik(tgt_bad, Q_HOME)
rep_bad = reachability.check_grasp_pose(model, q_bad, tgt_bad,
                                        hand_name="hand",
                                        finger_offset=FINGER_OFFSET,
                                        table_z=0.0)
print(rep_bad.table())
print()
print(rep_bad.summary())

print()
print("=" * 74)
print("对照：撞桌面那个姿势，实际执行时会差多少")
print("=" * 74)
# 真的跑一遍，看实际停在哪
d = mujoco.MjData(model)
d.qpos[:7] = Q_HOME
d.ctrl[:7] = Q_HOME
d.ctrl[7] = 255.0
d.qpos[7:9] = [0.04, 0.04]
mujoco.mj_forward(model, d)
ramp = 1500
for i in range(ramp):
    a = (i + 1) / ramp
    a = 0.5 - 0.5 * np.cos(np.pi * a)
    d.ctrl[:7] = (1 - a) * Q_HOME + a * q_bad
    mujoco.mj_step(model, d)
for i in range(3000):
    d.ctrl[:7] = q_bad
    mujoco.mj_step(model, d)

actual = grasp_point(d)
print("  目标    : %s" % np.round(tgt_bad, 4))
print("  实际到达: %s" % np.round(actual, 4))
print("  差      : %.1f mm" % (np.linalg.norm(tgt_bad - actual) * 1000))
print()
print("  => 这就是为什么需要检查：IK 说 0.5 mm，实际差 30+ mm。")

print("OK")
