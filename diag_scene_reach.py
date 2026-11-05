"""diag_scene_reach.py —— 场景里的关键点，SO-101 到底够不够得着？

在动手调场景之前先量三件事：
    1. 工件抓取点，离底座多远？够得到吗？
    2. 卡盘中心，够得到吗？
    3. 各个相机的实际朝向对不对？

SO-101 的臂展约 300 mm（从 shoulder 到 gripper），
但底座在原点，shoulder 在 z=0.0624，
所以工作空间是个离底座 0.1~0.35 m 的球壳。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_scene_reach.py
"""
import os
import numpy as np
import mujoco
import scene_machining as S

model, data, ids = S.make_scene()

BASE = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")]
SHOULDER = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                       "shoulder")]
EE = data.site_xpos[ids["ee"]]

print("=" * 74)
print("关键点位置")
print("=" * 74)
print("  底座      : %s" % np.round(BASE, 4))
print("  shoulder  : %s" % np.round(SHOULDER, 4))
print("  末端(home): %s" % np.round(EE, 4))
print()

# ============ 1. 工作空间边界 ============
print("=" * 74)
print("第 1 步：扫描工作空间（用 IK 试，比算理论球壳准）")
print("=" * 74)

# SO-101 的末端 site
EE_SITE = ids["ee"]
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll"]
ARM_QADR = [model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]
ARM_DADR = [model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS]


def ik_to(target, q_init=None, iters=300):
    """5 轴 IK（位置 only）。SO-101 只有 5 个臂关节，姿态没法完全控制。"""
    q = np.zeros(model.nq) if q_init is None else q_init.copy()
    d = mujoco.MjData(model)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    for _ in range(iters):
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        err = target - d.site_xpos[EE_SITE]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(model, d, jp, jr, EE_SITE)
        # 只取臂关节
        J = jp[:, ARM_DADR]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
        for k, qa in enumerate(ARM_QADR):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                    ARM_JOINTS[k])
            lo, hi = model.jnt_range[jid]
            q[qa] = np.clip(q[qa] + 0.6 * dq[k], lo, hi)
        # 夹爪保持张开
        q[model.jnt_qposadr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]] = 1.0
    d.qpos[:] = q
    mujoco.mj_forward(model, d)
    return q, np.linalg.norm(target - d.site_xpos[EE_SITE])


print()
print("  %-30s %-14s %-12s" % ("目标点", "离底座距离", "IK 能否到达"))
print("  " + "-" * 58)

POINTS = {
    "工件中心（抓取点）": S.SHAFT_POS,
    "工件上方 8cm（接近点）": S.SHAFT_POS + np.array([0, 0, 0.08]),
    "卡盘中心": np.array([S.LATHE_X, S.LATHE_Y, S.LATHE_Z]),
    "卡盘前方 10cm": np.array([S.LATHE_X - 0.10, S.LATHE_Y, S.LATHE_Z]),
    "台面中心": np.array([0.25, 0.0, 0.0]),
}

REACH = {}
for name, p in POINTS.items():
    dist = np.linalg.norm(p - BASE)
    q, err = ik_to(np.array(p, dtype=float))
    ok = err < 0.01
    REACH[name] = (dist, ok, err)
    print("  %-30s %-14s %-12s"
          % (name, "%.3f m" % dist,
             "可以" if ok else "不行(差%.0fmm)" % (err * 1000)))

# ============ 2. 结论 ============
print()
print("=" * 74)
print("第 2 步：结论")
print("=" * 74)
bad = [k for k, v in REACH.items() if not v[1]]
if bad:
    print("  够不到的点：")
    for k in bad:
        print("    %-30s 距离 %.3f m" % (k, REACH[k][0]))
else:
    print("  所有关键点都够得到")

# ============ 3. 相机检查 ============
print()
print("=" * 74)
print("第 3 步：相机朝向")
print("=" * 74)
for cname in ("overview", "gripper_view"):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cname)
    pos = model.cam_pos[cid]
    R = model.cam_mat0[cid].reshape(3, 3)
    fwd = -R[:, 2]     # 相机看向 -z（相机坐标系）
    print("  %-14s pos=%s  看向=%s"
          % (cname, np.round(pos, 3), np.round(fwd, 3)))

print()
print("  说明：MuJoCo 相机看向自身坐标系的 -z 方向。")
print("        如果不指向场景中心，画面就是黑的。")
print("OK")
