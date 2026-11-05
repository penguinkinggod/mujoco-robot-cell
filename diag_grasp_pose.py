"""diag_grasp_pose.py —— 找正确的抓取姿态：手腕要转多少度。

问题
----
实测发现 SO-101 在 home 姿态下，两个 pad 的连线方向是
    [0.165, -0.041, 0.985]
也就是几乎沿 +z（竖直）。这表示夹爪是"上下开口"的，
只能夹竖直放着的东西。

但我们要夹的是一根【沿 x 轴横放】的圆柱。
要让夹爪从上方夹住它，pad 连线必须变成沿 y（水平，垂直于工件轴）。

要转的是 wrist_roll（绕夹爪自身轴），不是 wrist_flex（那是俯仰）。

这个脚本扫 wrist_roll，输出每个角度下：
    - pad 连线的方向
    - pad 连线与世界 x 轴的夹角（90° 最好 = 垂直于工件）
    - pad 连线与水平面的夹角（0° 最好 = 水平张开）

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_grasp_pose.py
"""
import numpy as np
import mujoco

import scene_full as S

model, data = S.make_scene()

SITE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
       "wrist_roll"]
QADR = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ARM]

pads = []
for i in range(model.ngeom):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    if n and "pad" in n.lower():
        pads.append((n, i))
pads.sort()

print("=" * 74)
print("扫描 wrist_roll，找让 pad 连线垂直于工件的角度")
print("=" * 74)
print()
print("  pads: %s" % [n for n, _ in pads])
print()
print("  %-10s %-26s %-12s %-12s %s"
      % ("wrist_roll", "pad 连线方向", "与x夹角", "与水平夹角", "评价"))
print("  " + "-" * 76)

best = None
for wr in np.linspace(-2.7, 2.8, 23):
    d = mujoco.MjData(model)
    d.qpos[:] = data.qpos
    d.qpos[QADR[4]] = wr
    d.qpos[model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]] = 0.6
    mujoco.mj_forward(model, d)

    p0 = d.geom_xpos[pads[0][1]]
    p1 = d.geom_xpos[pads[1][1]]
    axis = p1 - p0
    if np.linalg.norm(axis) < 1e-6:
        continue
    axis = axis / np.linalg.norm(axis)

    # 与 x 轴夹角（希望 90°，即垂直）
    ang_x = np.degrees(np.arccos(np.clip(abs(axis[0]), -1, 1)))
    # 与水平面夹角（希望 0°，即两个 pad 一样高，水平张开）
    ang_h = np.degrees(np.arcsin(np.clip(abs(axis[2]), -1, 1)))

    # 评分：越接近 (90, 0) 越好
    score = abs(ang_x - 90) + ang_h

    verdict = ""
    if score < 12:
        verdict = "★ 很适合"
    elif score < 30:
        verdict = "可用"
    else:
        verdict = ""

    print("  %-10.2f %-26s %-12.1f %-12.1f %s"
          % (wr, np.round(axis, 3), ang_x, ang_h, verdict))

    if best is None or score < best[0]:
        best = (score, wr, axis, ang_x, ang_h)

print()
print("=" * 74)
print("最佳 wrist_roll = %.3f rad (%.1f deg)" % (best[1], np.degrees(best[1])))
print("  pad 连线方向 = %s" % np.round(best[2], 3))
print("  与 x 轴夹角  = %.1f deg（理想 90）" % best[3])
print("  与水平夹角   = %.1f deg（理想 0）" % best[4])
print("=" * 74)

# ---- 再查：这个姿态下，pad 中心相对 site 的偏移 ----
wr = best[1]
d = mujoco.MjData(model)
d.qpos[:] = data.qpos
d.qpos[QADR[4]] = wr
d.qpos[model.jnt_qposadr[mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]] = 0.6
mujoco.mj_forward(model, d)

site_p = d.site_xpos[SITE]
pad_mid = (d.geom_xpos[pads[0][1]] + d.geom_xpos[pads[1][1]]) / 2
print()
print("pad 中心相对 gripperframe 的偏移：")
off = pad_mid - site_p
print("  世界偏移 = %s  (|%.1f| mm)"
      % (np.round(off, 4), np.linalg.norm(off) * 1000))
print("  这意味着：要让 pad 中心落在工件上，")
print("            site 位置 = 工件位置 - 偏移 = %s"
      % np.round(-off, 4))
print()
print("OK")
