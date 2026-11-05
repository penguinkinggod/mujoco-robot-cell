"""diag_contact_filter.py —— 夹爪到底能不能碰到工件？

怀疑：官方模型里夹爪 pad 是 contype=2 / conaffinity=2，
      而默认工件是 contype=1 / conaffinity=1。

MuJoCo 的碰撞规则是：
    (contype_A & conaffinity_B) || (contype_B & conaffinity_A) != 0 才碰

    2 & 1 = 0
    1 & 2 = 0
    => 不碰！

如果这是真的，那"夹爪夹不住工件"不是因为算法，
而是因为碰撞过滤规则。这个必须查清楚，否则后面白费功夫。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_contact_filter.py
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
ASSETS = os.path.join(ROOT, "models", "_tmp_so101",
                      "lerobot_env_so101-0.4.0", "lerobot_env_so101", "assets")

print("=" * 74)
print("第 1 步：读官方场景里的碰撞参数")
print("=" * 74)

model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS, "pick_scene.xml"))
data = mujoco.MjData(model)

for name in ("fixed_jaw_pad", "moving_jaw_pad", "block", "floor"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        continue
    print("  %-16s contype=%d  conaffinity=%d"
          % (name, model.geom_contype[gid], model.geom_conaffinity[gid]))

print()
print("  碰撞判定：(contype_A & conaffinity_B) || (contype_B & conaffinity_A)")

g_pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
g_blk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "block")
ca, aa = model.geom_contype[g_pad], model.geom_conaffinity[g_pad]
cb, ab = model.geom_contype[g_blk], model.geom_conaffinity[g_blk]
hit = (ca & ab) or (cb & aa)
print("  pad(%d/%d) vs block(%d/%d) -> %s"
      % (ca, aa, cb, ab, "会碰" if hit else "不碰  <-- 问题在这"))

# ============ 实测：让夹爪夹一下，看有没有接触 ============
print()
print("=" * 74)
print("第 2 步：实测——把块放到夹爪里，夹紧，数接触点")
print("=" * 74)

# 抬高块让它悬空，避免地面接触干扰
BID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
GRIP_ACT = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
EE = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

mujoco.mj_resetData(model, data)
mujoco.mj_forward(model, data)
ee_p = data.site_xpos[EE].copy()

# 直接把块瞬移到夹爪中心附近
qadr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                           "block")]
data.qpos[qadr:qadr + 3] = ee_p + np.array([0, 0, -0.03])
data.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]

for step in range(600):
    data.ctrl[GRIP_ACT] = 0.0 if step > 100 else 1.5   # 先张开再闭合
    mujoco.mj_step(model, data)

print("  夹紧后接触点数：%d" % data.ncon)
gripper_geoms = {g_pad, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                          "moving_jaw_pad")}
gripper_geoms.discard(-1)
n_pad_contact = 0
for i in range(data.ncon):
    c = data.contact[i]
    pair = {c.geom1, c.geom2}
    if pair & gripper_geoms:
        n_pad_contact += 1
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print("    接触: %s <-> %s  深度 %.5f" % (n1, n2, c.dist))
print("  其中夹爪 pad 参与的：%d" % n_pad_contact)

# ============ 修正方案 ============
print()
print("=" * 74)
print("第 3 步：修正方案")
print("=" * 74)
print("""
  如果 pad 参与的接触点是 0，说明官方资产的碰撞过滤是有意为之——
  它靠 env 里的"吸附"逻辑（attach/constraint）来抓，不是靠真实摩擦。

  我们要的是**真实物理夹紧**，所以要把 pad 和工件的 collision mask 设成能碰。

  最稳妥的配置（后面 scene_machining.py 用这个）：
      pad    : contype=3  conaffinity=3
      工件   : contype=1  conaffinity=3
      卡爪   : contype=2  conaffinity=3
      地面/桌: contype=1  conaffinity=3

  这样：
      pad(3) vs 工件(1/3) -> 3&3=3  碰
      pad(3) vs 卡爪(2/3) -> 3&3=3  碰（要避免自碰，靠父子关系自动排除）
      工件(1) vs 地面(1)  -> 1&3=1  碰
""")
print("OK")
