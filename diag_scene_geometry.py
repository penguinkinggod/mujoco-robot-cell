"""diag_scene_geometry.py —— 场景里每个东西到底在哪？

渲染看起来不对，先量坐标，不靠眼睛猜。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_scene_geometry.py
"""
import os
import numpy as np
import mujoco
import scene_machining as S

model, data, ids = S.make_scene()
mujoco.mj_forward(model, data)

print("=" * 78)
print("所有几何体的实际世界位置")
print("=" * 78)
print("  %-20s %-8s %-26s %-24s" % ("名称", "类型", "位置(x,y,z)", "尺寸"))
print("  " + "-" * 78)

TYPES = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid",
         5: "cylinder", 6: "box", 7: "mesh"}

for gid in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "(无名)"
    if "jaw" in name or "shaft" not in name:
        pass
    t = TYPES.get(model.geom_type[gid], "?")
    pos = data.geom_xpos[gid]
    size = model.geom_size[gid]
    print("  %-20s %-8s %-26s %s"
          % (name, t,
             "(%.3f, %.3f, %.3f)" % tuple(pos),
             np.round(size, 4)))

print()
print("=" * 78)
print("关键点核对")
print("=" * 78)

SHAFT_BODY = ids["shaft"]
shaft_pos = data.xpos[SHAFT_BODY]
print("  工件 body 位置     : (%.3f, %.3f, %.3f)" % tuple(shaft_pos))
print("  工件底面 z         : %.4f" % (shaft_pos[2] - S.SHAFT_D / 2))
print("  工作台顶面 z       : %.4f" % S.TABLE_TOP_Z)
print("  -> 工件是否贴台面  : %s"
      % ("是" if abs(shaft_pos[2] - S.SHAFT_D / 2 - S.TABLE_TOP_Z) < 1e-4
         else "否！悬空 %.1f mm" % ((shaft_pos[2] - S.SHAFT_D / 2
                                    - S.TABLE_TOP_Z) * 1000)))

print()
print("  V 形块位置：")
for gid in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
    if "vblock" in name:
        print("    %-14s (%.3f, %.3f, %.3f)  size=%s"
              % (name, *data.geom_xpos[gid], np.round(model.geom_size[gid], 4)))

print()
print("  卡盘端面 x         : %.4f" % S.LATHE_X)
print("  卡盘中心           : (%.3f, %.3f, %.3f)"
      % (S.LATHE_X + S.CHUCK_L / 2, S.LATHE_Y, S.LATHE_Z))
print("  主轴箱中心         : (%.3f, %.3f, %.3f)"
      % (S.LATHE_X + 0.090, S.LATHE_Y, S.TABLE_TOP_Z + 0.065))

print()
print("  卡爪位置（张开时）：")
for k in range(3):
    bid = ids["jaws"][k]
    print("    jaw_%d  (%.4f, %.4f, %.4f)" % (k, *data.xpos[bid]))
    print("           离轴线距离 %.1f mm"
          % (np.linalg.norm(data.xpos[bid][1:] - np.array([S.LATHE_Y, S.LATHE_Z])) * 1000))

print()
print("=" * 78)
print("问题诊断")
print("=" * 78)
problems = []

if abs(shaft_pos[2] - S.SHAFT_D / 2 - S.TABLE_TOP_Z) > 1e-4:
    problems.append("工件不贴台面（V 形块把它顶起来了）")

# 主轴箱是否挡住卡盘
head_front = S.LATHE_X + 0.090 - 0.045   # 主轴箱前面
print("  主轴箱前表面 x = %.4f，卡盘端面 x = %.4f" % (head_front, S.LATHE_X))
if head_front < S.LATHE_X + S.CHUCK_L:
    problems.append("主轴箱前表面(%.3f) 比卡盘端面(%.3f) 更靠前 -> 挡住卡爪"
                    % (head_front, S.LATHE_X))

for p in problems:
    print("  [!] %s" % p)
if not problems:
    print("  没发现问题")

print("OK")
