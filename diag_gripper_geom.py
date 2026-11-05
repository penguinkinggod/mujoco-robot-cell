"""把夹爪的真实几何量出来，不再靠猜。

背景
----
之前代码里写 FINGER_OFFSET = 0.0584，注释说这是"夹爪抓取中心"。
实测发现：0.0584 只是**指根**（finger body 原点）的位置，
真正的指垫还往下伸了 50 mm。

这导致两个问题：
    1. IK 以为把指根放到物体中心就算抓住了，其实指垫中心还在下面
    2. 可达性检查报"穿桌面"，因为指尖确实伸到了桌面以下

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_gripper_geom.py
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.join(HERE, "models", "panda", "panda.xml")

spec = mujoco.MjSpec.from_file(PANDA)
model = spec.compile()
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

HAND = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
LF = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
RF = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")


def finger_geoms(body_id):
    out = []
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        p = b
        while p != 0:
            if p == body_id:
                out.append(g)
                break
            p = model.body_parentid[p]
    return out


print("=" * 72)
print("1. 各个关键点的世界坐标（home 位姿）")
print("=" * 72)
print("  hand           z = %+.5f" % data.xpos[HAND][2])
print("  left_finger    z = %+.5f  (指根)" % data.xpos[LF][2])
print("  right_finger   z = %+.5f  (指根)" % data.xpos[RF][2])

lf_g = finger_geoms(LF)
rf_g = finger_geoms(RF)
all_f = lf_g + rf_g

# 每个 geom 的实际最低点：geom_xpos 是 geom 中心，还要减去 size
def geom_lowest_z(g):
    """用 geom 中心 + 尺寸估算最低点。box 用 size[2]，capsule 用 size[0]。"""
    z = data.geom_xpos[g][2]
    gtype = model.geom_type[g]
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return z - model.geom_size[g][2]
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return z - model.geom_size[g][0]
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z - model.geom_size[g][0]
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
        return z - model.geom_size[g][2]   # mesh 的 size 是半边长
    return z


lows = [(g, geom_lowest_z(g)) for g in all_f]
lows.sort(key=lambda x: x[1])
print()
print("  指垫各 geom 的最低点（从低到高，前 6 个）：")
for g, z in lows[:6]:
    owner = "left" if g in lf_g else "right"
    print("    geom%-3d (%s) 最低 z = %+.5f  type=%d"
          % (g, owner, z, model.geom_type[g]))

pad_bottom_z = lows[0][1]
print()
print("  => 指垫最低点 z = %+.5f" % pad_bottom_z)

print()
print("=" * 72)
print("2. 换算成相对 hand 的偏移（沿 hand 的 z 轴）")
print("=" * 72)
hand_z = data.xpos[HAND][2]
R = data.xmat[HAND].reshape(3, 3)
print("  hand 的 z 轴指向     : %s" % np.round(R[:, 2], 3))
print("  hand 的 z 轴朝下吗   : %s" % ("是" if R[2, 2] < 0 else "否"))

off_finger_origin = hand_z - data.xpos[LF][2]
off_pad_bottom = hand_z - pad_bottom_z
off_pad_center = (off_finger_origin + off_pad_bottom) / 2

print("  指根相对 hand 的偏移   : %.5f m   (代码里写的 0.0584)" % off_finger_origin)
print("  指垫最低点相对 hand     : %.5f m" % off_pad_bottom)
print("  指垫中心相对 hand       : %.5f m   <-- 这才是该用的值" % off_pad_center)
print("  指垫长度               : %.5f m (%.1f mm)"
      % (off_pad_bottom - off_finger_origin,
         (off_pad_bottom - off_finger_origin) * 1000))

print()
print("=" * 72)
print("3. 夹爪开口范围")
print("=" * 72)
for q in (0.0, 0.02, 0.04):
    d = mujoco.MjData(model)
    d.qpos[7:9] = [q, q]
    mujoco.mj_forward(model, d)
    w = float(np.linalg.norm(d.xpos[LF] - d.xpos[RF]))
    print("  finger_joint = %.3f rad -> 开口 %.4f m (%.1f mm)" % (q, w, w * 1000))

print()
print("=" * 72)
print("4. 结论：抓 43mm 高的方块，指垫必然碰到桌面吗？")
print("=" * 72)
CUBE_HALF = 0.0215
cube_h = CUBE_HALF * 2
pad_len = off_pad_bottom - off_finger_origin
print("  方块高度           : %.1f mm" % (cube_h * 1000))
print("  指垫长度           : %.1f mm" % (pad_len * 1000))
print()
if pad_len > cube_h:
    print("  指垫比方块长 %.1f mm。" % ((pad_len - cube_h) * 1000))
    print("  如果让指垫中心对齐方块中心，指垫下端会低于方块底面 %.1f mm。"
          % ((pad_len - cube_h) / 2 * 1000))
    print("  方块底面贴桌面(z=0)，所以指垫必然穿进桌面。")
    print()
    print("  这就是为什么 demo_04 里夹爪会撞桌面 —— 不是参数写错，是几何决定。")
    print("  正确做法有两种：")
    print("    A. 抬高抓取：让指垫下端刚好在桌面上方，方块只被夹住上部")
    print("    B. 把方块垫高：放在一个小台阶上，让指垫有空间伸下去")
else:
    print("  指垫比方块短，可以完整包住方块，不会撞桌面。")

print()
print("=" * 72)
print("5. 建议的 FINGER_OFFSET")
print("=" * 72)
print("  如果你想让'夹爪中心'= 指垫中心（推荐）：")
print("      FINGER_OFFSET = %.4f" % off_pad_center)
print()
print("  注意：改成这个值之后，同样的目标位置会让手臂整体抬高 %.1f mm，"
      % ((off_pad_center - 0.0584) * 1000))
print("  抓取高度参数要跟着重算。")

print("OK")
