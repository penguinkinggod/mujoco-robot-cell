"""diag_so101_payload.py —— SO-101 到底能搬多重的工件？

为什么先查这个
--------------
用户想做"SO-101 抓起工件放进二轴数控车的卡盘"。

但 SO-101 是 3D 打印的桌面级机械臂，额定负载约 500 g。
而 demo_07 里的车削件：
    阶梯轴 Φ40x200  = 1292 g   <- 超载 2.6 倍
    套筒 Φ60x80     = 1776 g   <- 超载 3.6 倍
    法兰盘 Φ120     = 2315 g   <- 超载 4.6 倍

所以第一件事不是写抓取算法，是**搞清楚真实负载边界**，
再按边界选工件。这就是用户说的"先用符合负载的工件"。

这个脚本做的事
--------------
在最不利姿态下，逐步加末端负载，看哪个关节先饱和。
输出"安全负载上限"，作为后面所有 demo 的硬约束。

方法
----
用 mj_applyFT 在末端(body)施加竖直向下的力，读 qfrc_bias（重力+负载的广义力），
对比 actuator_forcerange。这是静力学，不跑动力学，快且准。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_so101_payload.py
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
SO101 = os.path.join(ROOT, "models", "_tmp_so101",
                     "lerobot_env_so101-0.4.0", "lerobot_env_so101",
                     "assets", "so101_new_calib.xml")
G = 9.81

model = mujoco.MjModel.from_xml_path(SO101)
data = mujoco.MjData(model)

JOINT_NAMES = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
               for i in range(model.njnt)]
LIMITS = model.actuator_forcerange[:, 1].copy()   # 每个执行器的力矩上限
GRIP_BODY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

print("=" * 76)
print("SO-101 真实能力（从官方 MuJoCo 模型读出来的）")
print("=" * 76)
print("  关节数     : %d" % model.njnt)
print("  总质量     : %.3f kg" % model.body_mass.sum())
print("  执行器力矩 : ±%.2f N·m（STS3215，六个关节相同）" % LIMITS[0])
print("  末端可达距 : 约 300 mm（从 base 到 gripper）")
print()

# ============ 姿态扫描：哪个姿态最"吃力" ============
def worst_joint_ratio(q, load_kg):
    """在姿态 q、末端挂 load_kg 时，各关节力矩占用比的最大值。

    静力学：静止时 M*qacc=0，所以
        执行器力矩 = qfrc_bias - qfrc_applied
    qfrc_bias 是重力/科氏等偏置力，qfrc_applied 是外加负载的广义力。
    两个都要，只看 qfrc_bias 会漏掉负载（第一版就是这么错的）。
    """
    d = mujoco.MjData(model)
    d.qpos[:] = q
    mujoco.mj_forward(model, d)

    if load_kg > 0:
        force = np.array([0.0, 0.0, -load_kg * G])
        torque = np.zeros(3)
        point = d.xpos[GRIP_BODY].copy()
        # 注意 qfrc_target 必须传 body id，不是 data 数组
        mujoco.mj_applyFT(model, d, force, torque, point, GRIP_BODY,
                          d.qfrc_applied)
        applied = d.qfrc_applied.copy()
    else:
        applied = np.zeros(model.nv)

    need = np.abs(d.qfrc_bias - applied)
    return need / LIMITS, need


print("=" * 76)
print("第 1 步：找最不利姿态（臂伸最平，力臂最大）")
print("=" * 76)

POSES = {
    "全零（直立）": np.array([0, 0, 0, 0, 0, 1.0]),
    "前伸水平": np.array([0, 0.9, -0.5, 0, 0, 1.0]),
    "最低前伸": np.array([0, 1.2, -1.0, -0.2, 0, 1.0]),
    "高举": np.array([0, -1.0, 0.8, 0.4, 0, 1.0]),
    "侧伸": np.array([1.4, 0.9, -0.5, 0, 0, 1.0]),
}

print("  %-16s %-12s %-12s %-10s" % ("姿态", "空载最大占用", "挂 300g", "关节"))
print("  " + "-" * 60)
for name, q in POSES.items():
    r0, _ = worst_joint_ratio(q, 0.0)
    r3, n3 = worst_joint_ratio(q, 0.3)
    wi = int(np.argmax(r3))
    print("  %-16s %-12s %-12s %-10s"
          % (name, "%.0f%%" % (r0.max() * 100),
             "%.0f%%" % (r3.max() * 100), JOINT_NAMES[wi]))

# 选占用比最高的姿态作为最不利姿态
worst_name, worst_q = max(
    POSES.items(), key=lambda kv: worst_joint_ratio(kv[1], 0.3)[0].max())
print()
print("  最不利姿态：%s" % worst_name)

# ============ 负载扫描 ============
print()
print("=" * 76)
print("第 2 步：在最不利姿态下，逐步加负载")
print("=" * 76)
print()
print("  %-12s %-14s %-14s %-10s %-8s"
      % ("负载", "最大关节力矩", "上限", "占用比", "结论"))
print("  " + "-" * 66)

SAFE = None
for kg in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    r, need = worst_joint_ratio(worst_q, kg)
    wi = int(np.argmax(r))
    ok = r.max() < 0.8                       # 留 20% 给动态和摩擦
    if ok:
        SAFE = kg
    print("  %-12s %-14s %-14s %-10s %-8s"
          % ("%.0f g" % (kg * 1000), "%.3f N·m" % need[wi],
             "%.2f N·m" % LIMITS[wi], "%.0f%%" % (r.max() * 100),
             "OK" if ok else "饱和（会掉）"))

print()
print("  安全负载上限（占用 < 80%%）：%.0f g" % (SAFE * 1000 if SAFE else 0))

# ============ 推荐工件 ============
print()
print("=" * 76)
print("第 3 步：按负载上限，重新选工件")
print("=" * 76)
print()
print("  %-26s %-10s %-12s %-10s"
      % ("工件", "质量", "能否搬动", "说明"))
print("  " + "-" * 62)

# 钢/铝密度
RHO = {"钢": 7850.0, "铝": 2700.0}


def solid_mass(d_mm, l_mm, mat="钢"):
    """实心圆柱质量，返回**克**。d_mm/l_mm 是毫米。"""
    r_m = d_mm / 2000.0          # mm -> m
    l_m = l_mm / 1000.0
    return np.pi * r_m ** 2 * l_m * RHO[mat] * 1000.0     # kg -> g


CANDIDATES = [
    # 前三个用 demo_07 里实际算出来的质量（阶梯轴是三段，不是实心 Φ40x200）
    ("阶梯轴 Φ40x200（demo_07）", 1292.0, "钢"),
    ("套筒 Φ60x80（demo_07）", 1776.0, "钢"),
    ("法兰盘 Φ120（demo_07）", 2315.0, "钢"),
    ("D 轴 Φ16x60 钢", solid_mass(16, 60, "钢"), "钢"),
    ("D 轴 Φ20x80 钢", solid_mass(20, 80, "钢"), "钢"),
    ("D 轴 Φ20x100 钢", solid_mass(20, 100, "钢"), "钢"),
    ("D 轴 Φ25x60 铝", solid_mass(25, 60, "铝"), "铝"),
    ("小齿轮坯 Φ40x15 钢", solid_mass(40, 15, "钢"), "钢"),
    ("垫圈 Φ50x6 钢", solid_mass(50, 6, "钢"), "钢"),
    ("玩具方块 43mm", 168.6, "铝"),
]

FIT = []
for name, mass_g, mat in CANDIDATES:
    mass_kg = mass_g / 1000.0
    r, need = worst_joint_ratio(worst_q, mass_kg)
    ok = r.max() < 0.8
    util = r.max() * 100
    if ok:
        FIT.append((name, mass_g, util))
    print("  %-26s %-10s %-12s %-10s"
          % (name, "%.0f g" % mass_g,
             "可以" if ok else "不行",
             "力矩占用 %.0f%%" % util))

print()
if FIT:
    heaviest = max(FIT, key=lambda x: x[1])
    print("  能用的工件里，最重的：%s（%.0f g，力矩占用 %.0f%%）"
          % (heaviest[0], heaviest[1], heaviest[2]))
    print("  能用的工件共 %d / %d 种" % (len(FIT), len(CANDIDATES)))

# ============ 结论 ============
print()
print("=" * 76)
print("结论：为什么必须先换工件")
print("=" * 76)
print("""
  SO-101 的安全负载约 %d g。这不是"保守估计"，是舵机力矩算出来的硬边界。

  demo_07 里的工业件（1292 g / 1776 g / 2315 g）它一件也搬不动。

  所以正确的顺序是：
      1. 先按负载边界选工件（就这一步，你已经想到了）
      2. 在能力范围内把"抓取 + 放置到卡盘"的算法打通
      3. 工件换成更大的，只需要改质量参数，算法不用重写

  下一步的工件建议：Φ20x80 钢轴（约 %.0f g），
  重得足以检验算法，又在 SO-101 能力范围内。
""" % (SAFE * 1000 if SAFE else 0, solid_mass(20, 80, "钢")))
print("OK")
