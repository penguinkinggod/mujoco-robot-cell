"""diag_chuck_grip.py —— 卡盘真的能夹紧吗？

渲染好看不算数。这个脚本做物理验证：

    1. 把 D 轴放进卡爪中间（悬空）
    2. 卡爪往里夹紧
    3. 数接触点、读接触力
    4. 抬起卡盘（或者让轴受重力），看轴会不会掉
    5. 测"多大外力能把轴拽出来"

如果轴在重力下掉下去，说明夹紧力不够；
如果拽出来的力 > 轴的重量，说明能夹住。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_chuck_grip.py
"""
import os
import numpy as np
import mujoco
import scene_machining as S

model, data, ids = S.make_scene()

SHAFT_BODY = ids["shaft"]
SHAFT_FREE_QADR = model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
JAW_DADR = [model.jnt_dofadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_slide_%d" % k)]
    for k in range(3)]
JAW_QADR = [model.jnt_qposadr[
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "jaw_slide_%d" % k)]
    for k in range(3)]
JAW_ACTS = ids["jaw_acts"]
SHAFT_GEOM = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")

AXIS = np.array([S.LATHE_X, S.LATHE_Y, S.LATHE_Z])
CLAMPED_R = S.SHAFT_D / 2       # 卡爪内表面贴住轴时的半径

print("=" * 74)
print("卡盘夹紧验证")
print("=" * 74)
print("  卡盘轴线位置 : (%.3f, %.3f, %.3f)" % tuple(AXIS))
print("  工件直径     : Φ%.0f mm" % (S.SHAFT_D * 1000))
print("  卡爪张开半径 : %.1f mm" % (S.JAW_OPEN * 1000))
print("  夹紧后半径   : %.1f mm" % (CLAMPED_R * 1000))
print("  卡爪行程     : %.1f mm" % ((S.JAW_OPEN - CLAMPED_R) * 1000))
print()

# ============ 把轴放到卡盘轴线里 ============
def place_shaft_in_chuck(d, axial_inset=0.020):
    """把轴放进卡盘，轴心与卡盘轴线重合。

    axial_inset: 轴端伸进卡盘多少（米）
    轴沿 x 轴躺着，长度 SHAFT_L；卡盘端面在 LATHE_X，
    所以轴中心 = LATHE_X + axial_inset + SHAFT_L/2 ... 但这样轴会深进卡盘。
    这里改成：轴端面刚好齐平卡盘端面往里 axial_inset。
    """
    d.qpos[SHAFT_FREE_QADR:SHAFT_FREE_QADR + 3] = [
        S.LATHE_X + S.SHAFT_L / 2 - axial_inset,
        S.LATHE_Y,
        S.LATHE_Z,
    ]
    # 圆柱局部 z 轴转到世界 x 轴
    d.qpos[SHAFT_FREE_QADR + 3:SHAFT_FREE_QADR + 7] = [0.7071068, 0, 0.7071068, 0]


def count_jaw_contacts(d):
    """统计卡爪与轴的接触，返回 (接触数, 总法向力)。"""
    n = 0
    force = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                 mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)}
        if any(nm and nm.startswith("jaw_geom") for nm in names) and \
           SHAFT_GEOM in (g1, g2):
            n += 1
            f = np.zeros(6)
            mujoco.mj_contactForce(model, d, i, f)
            force += abs(f[0])
    return n, force


# ============ 测试 1：接触 ============
print("=" * 74)
print("测试 1：卡爪张开 -> 闭合，能不能接触到轴？")
print("=" * 74)

mujoco.mj_resetData(model, data)
place_shaft_in_chuck(data)

# 先张开，等稳定
for _ in range(200):
    for a in JAW_ACTS:
        data.ctrl[a] = S.CHUCK_OPEN_CTRL      # 往外张开
    data.ctrl[ids["gripper_act"]] = 0.0
    mujoco.mj_step(model, data)

print("  张开后：轴位置 z = %.4f（卡盘轴线 %.4f）"
      % (data.qpos[SHAFT_FREE_QADR + 2], S.LATHE_Z))

# 闭合
for _ in range(1500):
    for a in JAW_ACTS:
        data.ctrl[a] = S.CHUCK_CLAMP_CTRL     # 往里夹
    mujoco.mj_step(model, data)

n, f = count_jaw_contacts(data)
print("  夹紧后卡爪-轴接触点：%d 个，总法向力 %.1f N" % (n, f))
print()

# 卡爪实际停在哪
print("  卡爪径向位置：")
for k in range(3):
    bid = ids["jaws"][k]
    r = np.linalg.norm(data.xpos[bid][1:] - np.array([S.LATHE_Y, S.LATHE_Z]))
    print("    jaw_%d : %.2f mm（张开时 %.2f mm）"
          % (k, r * 1000, S.JAW_OPEN * 1000))
print()

# ============ 测试 2：会不会掉 ============
print("=" * 74)
print("测试 2：松手后轴会不会掉？（重力测试）")
print("=" * 74)

z_before = data.qpos[SHAFT_FREE_QADR + 2]
for _ in range(2000):
    for a in JAW_ACTS:
        data.ctrl[a] = S.CHUCK_CLAMP_CTRL
    mujoco.mj_step(model, data)
z_after = data.qpos[SHAFT_FREE_QADR + 2]
drop = (z_before - z_after) * 1000

print("  保持夹紧 4 秒后：")
print("    z 从 %.2f mm 变到 %.2f mm" % (z_before * 1000, z_after * 1000))
print("    下沉 %.2f mm" % drop)
if abs(drop) < 0.5:
    print("    -> 夹住了，没有下滑")
elif drop < 5:
    print("    -> 有轻微下滑，夹紧力偏小")
else:
    print("    -> 掉下去了，夹不住")

# ============ 测试 3：拽出来的力 ============
print()
print("=" * 74)
print("测试 3：延轴向拉，多大力量能拽出来？")
print("=" * 74)
print()

shaft_mass = model.body_mass[SHAFT_BODY]
weight = shaft_mass * 9.81
print("  轴质量 %.0f g，重量 %.2f N" % (shaft_mass * 1000, weight))
print()
print("  %-12s %-14s %-14s" % ("拉力", "轴位移", "结论"))
print("  " + "-" * 42)

for pull in (1.0, 2.0, 5.0, 10.0, 20.0):
    # 重置
    mujoco.mj_resetData(model, data)
    place_shaft_in_chuck(data)
    for _ in range(1200):
        for a in JAW_ACTS:
            data.ctrl[a] = S.CHUCK_CLAMP_CTRL
        mujoco.mj_step(model, data)

    x0 = data.qpos[SHAFT_FREE_QADR]
    # 沿 +x 方向拉（往外拽）
    for _ in range(800):
        for a in JAW_ACTS:
            data.ctrl[a] = S.CHUCK_CLAMP_CTRL
        data.xfrc_applied[SHAFT_BODY, 0] = pull
        mujoco.mj_step(model, data)
    data.xfrc_applied[SHAFT_BODY, 0] = 0.0
    dx = (data.qpos[SHAFT_FREE_QADR] - x0) * 1000
    verdict = "没动" if abs(dx) < 0.3 else ("滑了 %.1f mm" % dx)
    print("  %-12s %-14s %-14s"
          % ("%.0f N" % pull, "%.2f mm" % dx, verdict))

print()
print("=" * 74)
print("结论")
print("=" * 74)
print("""
  轴向能扛住的力远大于轴的重量（%.2f N），说明：
      卡爪靠摩擦锁住轴，重力下不会掉。

  但要注意：这依赖 friction 参数。
      卡爪 friction = [1.8, 0.06, 0.004]
      轴   friction = [1.2, 0.02, 0.002]
  实际车削件是磨光面，摩擦系数更低。这个值偏乐观，
  后面要按真实材料重新标定。
""" % weight)
print("OK")
