"""diag_scene_full_reach.py —— 检查完整场景里每个关键点机器人够不够得着。

不检查就摆位，等于闭眼开车。上一版把台钳放到 x=0.30,y=0.16，
结果截图里机器人歪着够不到。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_scene_full_reach.py
"""
import numpy as np
import mujoco

import scene_full as S

model, data = S.make_scene()

# ---- 找关节和 site ----
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll"]
ARM_QADR = [model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]
ARM_DADR = [model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]

# 用夹爪 frame site 做末端参考点
SITE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
if SITE < 0:
    for i in range(model.nsite):
        print("site[%d] %s" % (i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)))
    raise SystemExit("找不到 gripperframe")

GRIP_BODY = model.site_bodyid[SITE]

print("=" * 74)
print("完整场景可达性检查")
print("=" * 74)
print()
print("  夹爪 site   : %s" % mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, SITE))
print("  所属 body   : %s"
      % mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, GRIP_BODY))
print()


def fk(q_arm):
    """给 5 个手臂关节角，返回夹爪 site 的世界位置。"""
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    for adr, q in zip(ARM_QADR, q_arm):
        d.qpos[adr] = q
    mujoco.mj_kinematics(model, d)
    return d.site_xpos[SITE].copy()


def ik_to(target, q_init=None, iters=800, lam=0.05, step=0.6):
    """阻尼最小二乘 IK，只控位置。

    踩过的坑：mj_kinematics 只算位置，不算雅可比需要的中间量，
    直接 mj_jacSite 会拿到全 0 的雅可比（我测过，J 范数 = 0）。
    必须调 mj_comPos（算质心/复合量）或者干脆用 mj_forward。
    这里用 mj_forward，虽然慢一点但绝对正确。

    第二个坑：SO-101 没有 wrist_yaw，只有 5 个手臂关节控 3 维位置，
    冗余度够，但初始解不能全 0（全 0 在小腿伸直位，雅可比退化）。
    """
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    if q_init is None:
        # 给一个"肘部弯曲"的初始姿态，避开奇异位形
        q_init = np.array([0.0, -0.4, 0.8, -0.4, 0.0])
    for adr, q in zip(ARM_QADR, q_init):
        d.qpos[adr] = q
    mujoco.mj_forward(model, d)

    jacp = np.zeros((3, model.nv))
    lo = np.array([model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][0]
        for n in ARM_JOINTS])
    hi = np.array([model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)][1]
        for n in ARM_JOINTS])

    for it in range(iters):
        err = target - d.site_xpos[SITE]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_forward(model, d)
        mujoco.mj_jacSite(model, d, jacp, None, SITE)
        J = jacp[:, ARM_DADR]
        dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), err)
        q = np.array([d.qpos[a] for a in ARM_QADR]) + step * dq
        q = np.clip(q, lo, hi)
        for adr, v in zip(ARM_QADR, q):
            d.qpos[adr] = v

    mujoco.mj_forward(model, d)
    pos = d.site_xpos[SITE].copy()
    return pos, np.linalg.norm(target - pos), np.array(
        [d.qpos[a] for a in ARM_QADR])


# 关键点
shaft_home = data.qpos[-7:-4]
POINTS = [
    ("工件中心（抓取点）", shaft_home),
    ("工件正上方 10cm", shaft_home + np.array([0, 0, 0.10])),
    ("钳口中心", np.array([S.VISE_POS[0] + 0.01, 0.085, 0.044])),
    ("钳口正上方 8cm", np.array([S.VISE_POS[0] + 0.01, 0.085, 0.124])),
    ("台面中心", np.array([0.28, 0.02, 0.0])),
]

print("  %-22s %-26s %-10s %s" % ("目标点", "位置", "残差", "结论"))
print("  " + "-" * 76)

q = np.zeros(5)
results = []
for label, target in POINTS:
    pos, err, qsol = ik_to(target, q_init=q)
    ok = err < 0.005
    print("  %-22s [%.3f, %.3f, %.3f]%s %-10s %s"
          % (label, target[0], target[1], target[2],
             " " * 2, "%.2f mm" % (err * 1000),
             "可达" if ok else ("偏差大" if err < 0.03 else "够不着")))
    results.append((label, err, ok))
    if ok:
        q = qsol

print()
nok = sum(1 for _, _, o in results if o)
print("  可达 %d / %d" % (nok, len(results)))

print()
print("=" * 74)
print("布局参数（如果不可达，改这些）")
print("=" * 74)
print("  VISE_POS  = %s" % (S.VISE_POS,))
print("  SHAFT_POS = %s" % (S.SHAFT_POS,))
print("  钳口中心  = [%.3f, 0.085, 0.044]" % (S.VISE_POS[0] + 0.01))
print()
print("OK")
