"""场景构建：SO-101 抓 D 轴 → 放进二轴车卡盘。

为什么用程序建场景而不是手写 XML
--------------------------------
因为后面要扫参数（工件直径、卡爪行程、摩擦系数），
写死在 XML 里改一次要改三个地方。用 MjSpec 建，参数就是一个变量。

场景里有什么（全部可见，不是抽象）
--------------------------------
    1. SO-101 六轴臂        官方资产（HuggingFace lerobot-env-so101，Apache-2.0）
    2. 工作台               自建，带 D 轴定位块
    3. D 轴工件 Φ20x80      自建，**D 形截面**（不是纯圆，纯圆会滚）
    4. 二轴车床主轴 + 三爪卡盘  自建，**卡爪能真的夹紧**
    5. 地面 + 灯光          看得清用的

坐标约定（很重要，后面所有代码都用这套）
--------------------------------------
    x 向前（远离机器人）
    y 向左
    z 向上
    机器人底座在原点
    工作台在 x>0 前方
    车床卡盘在 x 更远处、偏右

跑法：
    D:\\robot-env\\Scripts\\python.exe scene_machining.py
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
ASSETS = os.path.join(ROOT, "models", "_tmp_so101",
                      "lerobot_env_so101-0.4.0", "lerobot_env_so101", "assets")
SO101 = os.path.join(ASSETS, "so101_new_calib.xml")

# ============ 全局尺寸（改这里就够）============
TABLE_TOP_Z = 0.0             # 工作台台面高度（机器人底座在这个平面）
TABLE_SIZE = (0.55, 0.45, 0.02)

# 工件：D 轴 Φ20 x 80，钢
SHAFT_D = 0.020
SHAFT_L = 0.080
SHAFT_FLAT = 0.004            # D 形铣扁深度
STEEL_RHO = 7850.0
SHAFT_MASS = np.pi * (SHAFT_D / 2) ** 2 * SHAFT_L * STEEL_RHO  # ≈197 g

# D 轴在工作台上的初始位置（躺平，轴线沿 x）
# 布局约束：
#   卡爪占 x ∈ [0.230, 0.258]，卡盘盘体 x ∈ [0.260, 0.295]。
#   工件初始放在 y < 0 一侧，机器人从中间转过去。
SHAFT_POS = np.array([0.28, -0.14, TABLE_TOP_Z + SHAFT_D / 2])

# 二轴车：主轴 + 三爪卡盘
# 卡盘轴线沿 x（工件从 -x 方向送进去，和真实上下料一致）。
#
# LATHE_Y 由可达性实测决定（2026-11-05，settle + 3 轮逼近，同
# diag_chuck_health.py 口径）：
#   y=0.24 -> IK 残差 29.09 mm；y=0.20 -> 13.17；y=0.16 -> 11.66；
#   y=0.12 -> 10.45 mm。z 在 0.06~0.15 内几乎不影响（<0.1 mm）。
# 原来是凭"0.30~0.36 m 最舒服"的推测放在 y=0.24，实测够不着。
# 注意：残差仍有约 10 mm，来自 IK 稳态偏差，要靠 demo_09 的
# shaft_in_grip 补偿在合并场景的送入动作里压到同轴度 ≤ 2 mm。
LATHE_X = 0.26                # 卡盘端面 x（卡爪伸出的那一面）
LATHE_Y = 0.12                # 卡盘中心 y（可达性实测最优）
LATHE_Z = TABLE_TOP_Z + 0.06  # 卡盘轴线高度（离台面 60 mm）
CHUCK_R = 0.055               # 卡盘半径（Φ110，小型二轴车）
CHUCK_L = 0.035               # 卡盘厚度
JAW_W = 0.018                 # 卡爪截面宽
# 卡爪沿轴向长度。28 mm 时轴在爪里像球铰：夹紧后接触 6 点，但解锁静置
# 就在自重下转到 10.9 deg，一个爪脱开（接触掉到 4），20 N 轴向拉拔直接
# 把轴拽出去（实测 4172 mm）。真实车床卡爪是长条形，靠长度抗倾覆，这里
# 加到 55 mm 让卡爪覆盖轴的前半段。
JAW_LEN = 0.055               # 卡爪沿轴向长度
JAW_OPEN = 0.022              # 张开时卡爪内表面到轴线的距离

# 卡盘驱动：真实三爪卡盘是「一根丝杠驱动一个自由度」，不是三个电机。
# 这里只留 1 个 position servo 作用在 jaw_slide_0，另两爪由两条
# mjEQ_JOINT 关节联动约束被动跟随（见 build_scene）。
# 数值由受控扫描选定：位移差 0.12 mm、各爪力 [6.5,4.6,7.1] N、
# 偏心 0.00 mm、20 N 拉不动。-0.012 也能夹住但位移差升到 0.17 mm。
# 卡盘控制量是「丝杠目标位置」，单位 m（负 = 夹紧方向），力由
# <position> 执行器的 forcerange 限幅 —— 这就是真机：丝杠把爪送到某个
# 位置，扭矩上限决定最大夹紧力，工件挡住时爪停在工件上、力封顶。
#
# 中间试过纯 <motor> 力控（force = ctrl），方向对但有个致命问题：
# 爪失去位置约束，会一路滑到 -30 mm，弧片穿过卡盘中心，三爪互撞
# （216 个接触对），力全耗在自身穿模上，工件反而夹不住。
# 关节限位拦不住 150 N（solreflimit 太软，被压穿 14 mm）。
#
# 位置伺服的 kp 受稳定性限制：kp < armature/dt^2。原来 armature=0.01
# 时只能到 kp=4000、过盈 7 mm 才 28 N。真机的爪 + 丝杠 + 平面螺纹
# 等效惯量远不止 10 g，所以把 armature 提到 0.2，kp 就能开到 20000：
# 同样 5 mm 过盈给出 ~100 N，而且爪仍然被目标位置约束着，不会穿。
CHUCK_OPEN_CTRL = 0.010       # m，丝杠退到张开位
# 「拧到底」= 目标位置给到远超滑座行程的深度，夹紧力由 forcerange 封顶。
# 实测（2026-11-06，diag_jaw_wrap.py + verify_all.py 9/9）：
#   -0.020 -> 执行器 -84.5 N（未封顶），三爪 [29.3 13.9 35.9] N，力比 2.58
#   -0.030 -> 执行器 -150.0 N（封顶），  三爪 [52.7 41.7 41.8] N，力比 1.27
# 力限抬到 300/500/800/1200 N 全部无效（执行器恒定 -84.5 N）：瓶颈是
# kp x 位移误差，forcerange 在 -0.020 下是个够不着的天花板。
# 再往深也不行：力限 300 N 时滑座 -15.99 -> -16.38 mm，三爪端面出现
# 6 对互顶，料上的力反而只剩 224 N、下沉涨到 0.233 mm。
# 全流程实测：三爪 168.7 N、20 N 拉拔 0.00 mm、同轴 0.71 mm、接触 30。
# 前提：slide.armature 必须 0.2。add_lathe 里原来是 0.01，而 kp=20000
# 要求 armature > kp*dt^2 = 0.08，差了 8 倍 —— 伺服在发散边缘抖，爪
# 抖到把料甩飞（同轴 250 mm、机械臂跟着乱晃）。这条比夹紧参数重要。
CHUCK_CLAMP_CTRL = -0.030     # m，丝杠拧到底（力由 forcerange 封顶）
CHUCK_CTRL_RANGE = (-0.032, 0.012)     # m，丝杠目标行程（比滑座行程深）
CHUCK_FORCE_LIMIT = 150.0     # N，扭矩上限（三爪合计；每爪名义 50 N）

# ---- 按料径镗出的圆弧软爪（soft jaws）----
# 原来的卡爪是 18x18 mm 方盒，中心在半径 22 mm 处；闭合时三个盒子互相
# 穿插，围出来的是一个三角区而不是圆孔，Φ20 圆柱能在里面滚、能沿重力
# 方向沉到平衡位（实测夹紧后 11 s 内 z 掉 4.3 mm，"变正"是滑出来的）。
# 真实三爪卡盘用软爪按料径镗圆弧，三个 120 deg 弧片合成一个圆孔，靠
# 几何自定心。
#
# MuJoCo 的 mesh 碰撞走凸包：单独一个凹圆弧网格会被填平，凹槽消失。
# 所以用 N 个小凸 box 沿圆弧拼，每段自身是凸的，整体近似凹面。
# 段数是精度与数值稳定性的折中：9 段时三爪法向力比 2.33（>2 判据挂），
# 7 段 2.01，5 段 1.63，3 段 1.61。取 5 段：弦弧误差在 0.2 mm 以内，
# 接触点少，力分配不被 0.13 mm 的三爪同步差放大。
JAW_ARC_SEGS = 5
# 包角原来取 52 deg（每爪 104 deg，三爪 312 deg，缝只有 3x16 deg）。
# 变径扫描发现 Φ18 以下三爪内缘先互顶，力全花在爪与爪上；扫包角后确认
# 互顶临界在 45/46 deg 之间（46 deg 时 Φ15 互顶 845 N）。
# 但包角不是越小越好：42 deg 时主线体检的三爪力比从 1.6 涨到 4.24，
# 三爪 0.1 mm 同步差被短弧片放大，某一爪先吃到力。44 deg 是两头都过的
# 点：互顶 0、力比 1.15、离临界还有 2 deg。见 diag_jaw_face_sweep.py。
# 三爪共 264 deg、缝 3x32 deg，反而更接近真机软爪（本来就不该包满）。
# Φ20 主线接触点仍 30，爪力和同轴基本不变。
JAW_ARC_HALF_DEG = 44.0       # 每爪包角 88 deg，三爪共 264 deg，留 3x32 deg 缝
JAW_ARC_CLEAR = 0.0002        # 镗孔半径比料半径大 0.2 mm（真机软爪留的间隙）
JAW_ARC_THICK = 0.005         # 弧片径向厚度
JAW_ARC_OVERLAP = 1.02        # 相邻弧片段的角向重叠系数（防漏缝）
JAW_BACK_THICK = 0.008        # 背板厚度（只在视觉上补成实心块）
# 软爪镗孔半径。None = 跟着 SHAFT_D 走，等于每换一种料就就地镗一套爪；
# 设成固定值则是「拿一套现成软爪去夹别的直径」，给变径扫描用。
# 默认 None，主线行为不变。
JAW_ARC_BORE_R = None
# 软爪的「设计闭合位」：镗孔时卡爪停的位置，与实际控制量解耦。
# 夹紧力来自丝杠继续拧紧（ctrl 比这个位置更深），不是来自几何过盈。
JAW_ARC_Q0 = -0.015


def add_soft_jaws(jb, k, ang, jaw_open, apply_mask):
    """把第 k 个卡爪的方盒换成拼出来的圆弧软爪。

    闭合时（q = CHUCK_CLAMP_CTRL）弧片内表面正好落在 r_hole 上；张开时
    整组随滑座沿自身径向平移 20 mm，轴向通道最小内半径约 27 mm，Φ20
    棒料仍可沿 +x 插入。基板、作动器、三条联动约束一行都不用改。
    """
    r_clamp = jaw_open + JAW_ARC_Q0
    bore = SHAFT_D / 2 if JAW_ARC_BORE_R is None else JAW_ARC_BORE_R
    r_hole = bore + JAW_ARC_CLEAR
    r_c = r_hole + JAW_ARC_THICK / 2
    half = np.deg2rad(JAW_ARC_HALF_DEG)
    da = 2 * half / JAW_ARC_SEGS
    ca, sa = np.cos(ang), np.sin(ang)
    for s in range(JAW_ARC_SEGS):
        a = ang - half + (s + 0.5) * da
        g = jb.add_geom()
        # 名字必须保留 jaw_geom_<k> 前缀：jaw_state() / diag_chuck_health
        # / probe_chuck_mask 都靠这个名字统计接触点和法向力。
        g.name = "jaw_geom_%d_%02d" % (k, s)
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [JAW_LEN / 2, JAW_ARC_THICK / 2,
                  r_c * np.sin(da / 2) * JAW_ARC_OVERLAP]   # 相邻段轻微重叠，避免缝隙
        # body 原点在 r*u(ang)；闭合时它移到 r_clamp*u(ang)，所以这里
        # 用 r_clamp 反推局部偏移，保证闭合位姿的弧心与主轴重合。
        g.pos = [0.0,
                 r_c * np.cos(a) - r_clamp * ca,
                 r_c * np.sin(a) - r_clamp * sa]
        g.quat = [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]
        g.rgba = [0.72, 0.68, 0.52, 1.0]
        g.friction = [1.8, 0.06, 0.004]
        apply_mask(g)

    # 背板：闭合时内表面在 r_hole + JAW_ARC_THICK 以外，碰不到 Φ20 棒料，
    # 只是让爪在画面里看起来是实心块。
    b = jb.add_geom()
    b.name = "jaw_back_%d" % k
    b.type = mujoco.mjtGeom.mjGEOM_BOX
    r_back = r_c + JAW_ARC_THICK / 2 + JAW_BACK_THICK / 2
    b.size = [JAW_LEN / 2, JAW_BACK_THICK / 2, JAW_W / 2]
    b.pos = [0.0, (r_back - r_clamp) * ca, (r_back - r_clamp) * sa]
    b.quat = [np.cos(ang / 2), np.sin(ang / 2), 0.0, 0.0]
    b.rgba = [0.55, 0.52, 0.42, 1.0]
    apply_mask(b)


def build_scene(shaft_mass=None, jaw_open=None):
    """建整个场景，返回 MjSpec。

    参数化是为了后面扫参数用：
        shaft_mass  工件质量（默认按钢密度算）
        jaw_open    卡爪张开量（默认 JAW_OPEN）
    """
    if shaft_mass is None:
        shaft_mass = SHAFT_MASS
    if jaw_open is None:
        jaw_open = JAW_OPEN

    spec = mujoco.MjSpec.from_file(SO101)
    # 同上（scene_full._damp_arm_servos）：XML 里 <position kp="17.8"/> 没有
    # Kd，阶跃超调 45.6%、2.8 Hz 回弹，末端看就是「一跳一跳」。
    # 这里是另一条独立建模路径，参数必须和 scene_full 保持一致。
    try:
        from scene_full import ARM_SERVO_KP as _KP, ARM_SERVO_KV as _KV
    except Exception:
        _KP, _KV = 70.0, 2.6
    for _a in spec.actuators:
        if _a.name in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll"):
            _a.set_to_position(kp=_KP, kv=_KV)

    # 机器人底座抬到台面上
    base = spec.body("base")
    base.pos = [0.0, 0.0, TABLE_TOP_Z]

    # ---- 关键：修碰撞掩码 ----
    # 官方资产里夹爪 pad 是 contype=2/conaffinity=2，工件是 1/1，
    # 2&1 = 0 => 物理上根本不碰（实测 0 个接触点）。
    # 官方靠"吸附"抓取，我们要真实摩擦，所以必须统一掩码。
    for gname in ("fixed_jaw_pad", "moving_jaw_pad"):
        g = spec.geom(gname)
        g.contype = 3
        g.conaffinity = 3
        g.friction = [1.6, 0.05, 0.003]

    # 机械臂其它碰撞体保持默认（会撞桌面，这是对的）

    # ---- 物理设置 ----
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 10
    spec.option.noslip_tolerance = 0.0
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 5.0

    # 离屏渲染缓冲：不只是 numpy 数组大小，模型里也要声明
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200

    wb = spec.worldbody

    # ---- 掩码约定（全场景统一，避免"不碰"的坑）----
    #   contype / conaffinity 都必须有共同的 bit 才碰。
    #   这里统一用 1|2=3，也就是所有东西都互相碰，靠父子关系排除自碰。
    def _mask(g, ct=1, ca=3):
        g.contype = ct
        g.conaffinity = ca
        return g

    # ---- 地面 ----
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [3.0, 3.0, 0.1]
    floor.rgba = [0.20, 0.21, 0.24, 1.0]
    floor.friction = [1.0, 0.01, 0.001]
    _mask(floor, 1, 3)

    # ---- 工作台 ----
    # 一块 L 形台面：工件区（y<0）和车床区（y>0）都覆盖
    table = wb.add_geom()
    table.name = "table_start"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [0.22, 0.30, TABLE_SIZE[2] / 2]
    table.pos = [0.26, 0.05, TABLE_TOP_Z - TABLE_SIZE[2] / 2]
    table.rgba = [0.34, 0.36, 0.40, 1.0]
    table.friction = [1.0, 0.01, 0.001]
    _mask(table, 1, 3)

    # 台面上给 D 轴两个定位块（限制轴向滚动，相当于简单的 V 形块）
    # 轴沿 x 躺，所以定位块在轴的两端（±x）
    for sx in (-1, 1):
        blk = wb.add_geom()
        blk.name = "vblock_%s" % ("L" if sx < 0 else "R")
        blk.type = mujoco.mjtGeom.mjGEOM_BOX
        blk.size = [0.006, 0.012, SHAFT_D * 0.62]
        blk.pos = [SHAFT_POS[0] + sx * (SHAFT_L / 2 + 0.008),
                   SHAFT_POS[1],
                   TABLE_TOP_Z + SHAFT_D * 0.62]
        blk.rgba = [0.46, 0.48, 0.52, 1.0]
        blk.friction = [1.0, 0.01, 0.001]
        _mask(blk, 1, 3)

    # ---- 工件：D 轴 ----
    # 圆柱 + 顶端一个薄平面表示"铣扁"，这样夹爪夹的是平面，不会打滑。
    shaft = wb.add_body()
    shaft.name = "shaft"
    shaft.pos = SHAFT_POS
    # 轴线沿 x 方向躺着（和车床主轴同向，真实上下料就是这样）
    shaft.quat = [0.7071068, 0.0, 0.7071068, 0.0]
    shaft.add_freejoint(name="shaft_free")

    sg = shaft.add_geom()
    sg.name = "shaft_geom"
    sg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    sg.size = [SHAFT_D / 2, SHAFT_L / 2, 0.0]
    sg.mass = shaft_mass
    sg.rgba = [0.68, 0.70, 0.74, 1.0]
    sg.friction = [1.2, 0.02, 0.002]
    _mask(sg, 1, 3)

    # D 形铣扁：一个薄 box 贴在圆柱顶部，让夹爪夹到平面而不是弧面
    flat = shaft.add_geom()
    flat.name = "shaft_flat"
    flat.type = mujoco.mjtGeom.mjGEOM_BOX
    # 轴沿 x 躺，局部坐标里圆柱轴是 z（旋转后才变成世界 x）。
    # 铣扁贴在圆柱"上方"（局部 +y，旋转后是世界 +z）
    flat.size = [SHAFT_L / 2, SHAFT_D * 0.42, SHAFT_FLAT / 2]
    flat.pos = [0.0, 0.0, SHAFT_D / 2 - SHAFT_FLAT / 2]
    flat.mass = 0.0
    flat.rgba = [0.78, 0.80, 0.84, 1.0]
    flat.contype = 0
    flat.conaffinity = 0

    # ---- 二轴车：主轴箱 + 卡盘 ----
    # 布局要点（第一版踩过的坑）：
    #   卡盘必须朝 -x（朝向机器人），主轴箱在卡盘【后面】(+x)，
    #   否则主轴箱会把卡盘整个挡住。
    head = wb.add_geom()
    head.name = "lathe_headstock"
    head.type = mujoco.mjtGeom.mjGEOM_BOX
    head.size = [0.045, 0.075, 0.060]
    head.pos = [LATHE_X + CHUCK_L + 0.045, LATHE_Y, TABLE_TOP_Z + 0.060]
    head.rgba = [0.30, 0.32, 0.36, 1.0]
    _mask(head, 1, 3)

    # 卡盘盘体：轴向沿 x，端面朝 -x（朝向机器人）
    # 位置取 LATHE_X + CHUCK_L/2，所以盘体占 x ∈ [LATHE_X, LATHE_X+CHUCK_L]，
    # 端面正好在 LATHE_X，卡爪从这里往 -x 伸出。
    chuck = wb.add_geom()
    chuck.name = "chuck_body"
    chuck.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    chuck.size = [CHUCK_R, CHUCK_L / 2, 0.0]
    chuck.pos = [LATHE_X + CHUCK_L / 2, LATHE_Y, LATHE_Z]
    chuck.quat = [0.7071068, 0.0, 0.7071068, 0.0]   # 圆柱轴 z -> x
    chuck.rgba = [0.44, 0.46, 0.50, 1.0]
    chuck.friction = [0.8, 0.01, 0.001]
    _mask(chuck, 1, 3)

    # 卡盘端面上加三道滑槽，视觉上区分三爪位置（纯装饰，不参与碰撞）
    for k in range(3):
        ang = 2 * np.pi * k / 3 + np.pi / 2
        cy, cz = np.cos(ang), np.sin(ang)
        slot = wb.add_geom()
        slot.name = "chuck_slot_%d" % k
        slot.type = mujoco.mjtGeom.mjGEOM_BOX
        slot.size = [0.001, 0.006, (CHUCK_R - 0.012) / 2]
        slot.pos = [LATHE_X - 0.001,
                    LATHE_Y + cy * (0.012 + (CHUCK_R - 0.012) / 2),
                    LATHE_Z + cz * (0.012 + (CHUCK_R - 0.012) / 2)]
        # 绕 x 轴转到对应角度
        ang_x = ang - np.pi / 2
        slot.quat = [np.cos(ang_x / 2), np.sin(ang_x / 2), 0.0, 0.0]
        slot.rgba = [0.18, 0.19, 0.22, 1.0]
        slot.contype = 0
        slot.conaffinity = 0

    # 主轴（从卡盘中心往后延伸的圆柱，视觉上像车床）
    spindle = wb.add_geom()
    spindle.name = "spindle"
    spindle.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    spindle.size = [0.022, 0.060, 0.0]
    spindle.pos = [LATHE_X + CHUCK_L + 0.060, LATHE_Y, LATHE_Z]
    spindle.quat = [0.7071068, 0.0, 0.7071068, 0.0]
    spindle.rgba = [0.55, 0.57, 0.60, 1.0]
    _mask(spindle, 1, 3)

    # 三爪：120° 分布，每个爪沿径向滑动（slide 关节）
    # 设计说明：
    #   - 卡盘端面在 x = LATHE_X，盘体往 +x 方向延伸
    #   - 卡爪骑在端面上，往 -x 方向伸出一截（这样能抱住送进来的轴）
    #   - 卡爪内表面到轴线距离 = jaw_open；夹紧时滑到 = 轴半径
    jaw_bodies = []
    for k in range(3):
        ang = 2 * np.pi * k / 3 + np.pi / 2
        cy, cz = np.cos(ang), np.sin(ang)

        jb = wb.add_body()
        jb.name = "jaw_%d" % k
        # 卡爪中心：沿径向 jaw_open，沿轴向【伸出盘面外侧】
        # 端面在 x = LATHE_X，卡爪往 -x 方向伸出 JAW_LEN（这样才看得见、才抱得住轴）
        jb.pos = [LATHE_X - JAW_LEN / 2 - 0.002,
                  LATHE_Y + jaw_open * cy,
                  LATHE_Z + jaw_open * cz]

        # 径向滑动关节（正方向 = 往外张开）
        slide = jb.add_joint()
        slide.name = "jaw_slide_%d" % k
        slide.type = mujoco.mjtJoint.mjJNT_SLIDE
        slide.axis = [0.0, cy, cz]
        # 行程限位。位置伺服本身已经不会越过目标位置，这里只是兜底。
        slide.range = [-0.022, 0.018]
        slide.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        # damping 是向量，armature 是标量，类型不一样（踩过）。
        # armature 从 0.01 提到 0.2：爪 + 丝杠 + 平面螺纹的等效惯量，
        # 也是 kp 能开到 20000 而不抖的前提（kp < armature/dt^2）。
        slide.damping[:] = 300.0
        slide.armature = 0.2

        # 卡爪主体：截面是 (轴向长, 径向厚, 切向宽)。
        # 必须把 box 绕 x 轴转到该爪的径向上！早期版本三个 geom 都是
        # 轴对齐的，box 沿各自径向的支撑半宽变成 9*|cos|+9*|sin|：
        # 第 0 爪 9 mm，另两爪 12.3 mm。三爪内表面到轴线天生差 3.3 mm，
        # 导致「三爪不同步、受力差 4 倍」的假象（实测位移差 4.24 mm、
        # 力 [7.1,23.8,25.8] N）。旋转后驱动架构一行不改，位移差降到
        # 0.23 mm、力 [19.1,18.4,19.4] N、偏心 0.00 mm。
        def _mask_box(g):
            g.contype = 2
            g.conaffinity = 3

        add_soft_jaws(jb, k, ang, jaw_open, _mask_box)

        jaw_bodies.append(jb)

    # 卡盘驱动：一根丝杠 = 一个执行器 + 两条关节联动约束。
    #
    # 用 <motor>，ctrl 就是牛顿。前面几版用 <position> 加 kp=4000：
    # 力 = kp x (ctrl - q)，过盈 7 mm 才 28 N，夹 Φ20 的 45 钢根本不够
    # （师傅夹 55x75、只咬 5 mm 都不掉，那是 kN 级）。力控还顺便解决了
    # 「过盈一调就炸」——位置伺服硬顶 27 个弧片会互相打架，力控不会。
    act = spec.add_actuator()
    act.name = "jaw_act"
    act.target = "jaw_slide_0"
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.set_to_position(kp=20000.0, kv=600.0)
    act.ctrlrange[:] = CHUCK_CTRL_RANGE
    act.forcerange[:] = (-CHUCK_FORCE_LIMIT, CHUCK_FORCE_LIMIT)

    # 联动约束：q_jaw_slide_0 = q_jaw_slide_1 = q_jaw_slide_2。
    # MjsEquality.data 默认是 [0,1,0,0,0,...]，即 q1 = q2 的多项式恒等，
    # 不用赋值；objtype 显式与否实测结果完全一样。
    #
    # 为什么早先「3 个伺服 + equality」会失败：3 个位置伺服各拉各的，
    # 约束又强行绑在一起，求解器与伺服互相打，q 冲到 202 mm、轴被弹飞
    # （12 组 kp/ctrl 全灭）。只留 1 个伺服、另两爪被动跟随才成立。
    # 另外 mjEQ_JOINT 是软约束：收紧 solref 到 1 ms、提高 solimp 上限，
    # 能把接触反力造成的稳态柔度压到 0.12 mm。真实卡盘的丝杠和爪也
    # 不是绝对刚体，这个量级反而比「严格零差」更物理。
    for k in (1, 2):
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = "jaw_slide_0"
        eq.name2 = "jaw_slide_%d" % k
        eq.solref[:] = [0.001, 1.0]
        eq.solimp[:] = [0.9, 0.99, 0.0001, 0.5, 100.0]

    # ---- 相机 ----
    # 用 lookat 方式建相机最不容易错：
    # MuJoCo 相机看向自身 -z，用 quat 手算很容易朝向反掉（第一版就是全黑）。
    # 这里改用 MjvCamera 的等价参数：位置 + 目标点。
    # MJCF 里没有 lookat 属性，所以先给位置，再在渲染时用 MjvCamera 覆盖。
    cam = wb.add_camera()
    cam.name = "overview"
    cam.pos = [0.05, -0.62, 0.62]
    cam.quat = [0.666, 0.335, 0.335, 0.590]   # 朝向场景中心
    cam.fovy = 50

    cam2 = wb.add_camera()
    cam2.name = "gripper_view"
    cam2.pos = [0.20, -0.24, 0.26]
    cam2.quat = [0.720, 0.310, 0.290, 0.540]
    cam2.fovy = 55

    # ---- 灯光 ----
    # 场景小（0.5 m 量级），光源要近，否则照不亮。
    l1 = wb.add_light()
    l1.name = "key"
    l1.pos = [0.15, -0.30, 0.65]
    l1.dir = [0.1, 0.35, -1.0]
    l1.diffuse = [1.0, 1.0, 1.0]
    l1.specular = [0.4, 0.4, 0.4]
    l1.ambient = [0.12, 0.12, 0.14]
    l1.castshadow = True

    l2 = wb.add_light()
    l2.name = "fill"
    l2.pos = [0.45, 0.45, 0.55]
    l2.dir = [-0.35, -0.45, -0.9]
    l2.diffuse = [0.55, 0.55, 0.60]
    l2.castshadow = False

    l3 = wb.add_light()
    l3.name = "rim"
    l3.pos = [-0.25, 0.20, 0.35]
    l3.dir = [0.7, -0.4, -0.5]
    l3.diffuse = [0.35, 0.36, 0.42]
    l3.castshadow = False

    # 提高整体环境光（模型里改 visual.headlight）
    spec.visual.headlight.ambient = [0.22, 0.22, 0.25]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.48]
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200

    return spec


def add_lathe(spec, *, origin=None, table_z=0.0, jaw_open=None,
              jaw_mask=(2, 3), static_mask=(1, 3)):
    """Append the three-jaw chuck to an EXISTING MjSpec (scene merging).

    This is the add_lathe(spec) step that lets the shallow-slot scene
    (scene_full.py) own the robot + workpiece while the chuck is a bolt-on
    workstation. build_scene() above is deliberately left untouched: every
    existing diagnostic keeps compiling exactly as before.

    origin      (x, y, z) of the chuck FACE (jaws stick out toward -x).
    jaw_mask    (contype, conaffinity) for the three jaw geoms. In
                scene_full.py's bitmask scheme pass CT_FIXTURE=(4, 2):
                the jaws then touch the workpiece (work ct=2 vs jaw ca=2)
                but NOT the gripper pads (robot ct=1, ca=3), because
                4&3 == 0 and 1&2 == 0. Without this the pads hit the jaws
                and the arm stalls -- measured 34.9 mm IK residual at
                x=0.25, 63.8 mm at x=0.28 with an empty gripper.
    static_mask mask for headstock/chuck body/spindle. Pass (0, 0) to make
                them purely visual, so neither the arm nor the inserted
                shaft collides with the disc. Only the three jaws clamp.

    Returns a dict of object names: jaw_act, jaws, joint names.
    """
    if origin is None:
        origin = (LATHE_X, LATHE_Y, LATHE_Z)
    if jaw_open is None:
        jaw_open = JAW_OPEN
    lx, ly, lz = origin
    wb = spec.worldbody

    def _m(g, ct, ca):
        g.contype = ct
        g.conaffinity = ca
        return g

    qx = [0.7071068, 0.0, 0.7071068, 0.0]   # cylinder local z -> world x

    head = wb.add_geom()
    head.name = "lathe_headstock"
    head.type = mujoco.mjtGeom.mjGEOM_BOX
    head.size = [0.045, 0.075, 0.060]
    head.pos = [lx + CHUCK_L + 0.045, ly, table_z + 0.060]
    head.rgba = [0.30, 0.32, 0.36, 1.0]
    _m(head, *static_mask)

    chuck = wb.add_geom()
    chuck.name = "chuck_body"
    chuck.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    chuck.size = [CHUCK_R, CHUCK_L / 2, 0.0]
    chuck.pos = [lx + CHUCK_L / 2, ly, lz]
    chuck.quat = qx
    chuck.rgba = [0.44, 0.46, 0.50, 1.0]
    chuck.friction = [0.8, 0.01, 0.001]
    _m(chuck, *static_mask)

    for k in range(3):
        ang = 2 * np.pi * k / 3 + np.pi / 2
        cy, cz = np.cos(ang), np.sin(ang)
        slot = wb.add_geom()
        slot.name = "chuck_slot_%d" % k
        slot.type = mujoco.mjtGeom.mjGEOM_BOX
        slot.size = [0.001, 0.006, (CHUCK_R - 0.012) / 2]
        slot.pos = [lx - 0.001,
                    ly + cy * (0.012 + (CHUCK_R - 0.012) / 2),
                    lz + cz * (0.012 + (CHUCK_R - 0.012) / 2)]
        ang_x = ang - np.pi / 2
        slot.quat = [np.cos(ang_x / 2), np.sin(ang_x / 2), 0.0, 0.0]
        slot.rgba = [0.18, 0.19, 0.22, 1.0]
        slot.contype = 0
        slot.conaffinity = 0

    spindle = wb.add_geom()
    spindle.name = "spindle"
    spindle.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    spindle.size = [0.022, 0.060, 0.0]
    spindle.pos = [lx + CHUCK_L + 0.060, ly, lz]
    spindle.quat = qx
    spindle.rgba = [0.55, 0.57, 0.60, 1.0]
    _m(spindle, *static_mask)

    for k in range(3):
        ang = 2 * np.pi * k / 3 + np.pi / 2
        cy, cz = np.cos(ang), np.sin(ang)
        jb = wb.add_body()
        jb.name = "jaw_%d" % k
        jb.pos = [lx - JAW_LEN / 2 - 0.002,
                  ly + jaw_open * cy, lz + jaw_open * cz]
        slide = jb.add_joint()
        slide.name = "jaw_slide_%d" % k
        slide.type = mujoco.mjtJoint.mjJNT_SLIDE
        slide.axis = [0.0, cy, cz]
        # 行程限位必须显式打开。以前卡盘是位置伺服，ctrl 的 ctrlrange
        # [-0.030, 0.008] 顺手当了限位，所以没暴露；改成力控（ctrl = N）
        # 以后爪一路滑到 -27 mm、弧片穿过卡盘中心，216 个接触对，力全耗
        # 在自身穿模上，工件反而夹不住。真机丝杠行程是有限的。
        slide.range = [-0.020, 0.020]
        slide.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        slide.damping[:] = 30.0
        # 与 build_scene 一致：kp=20000 的稳定前提是 armature > kp*dt^2
        # = 20000*0.002^2 = 0.08。0.01 会让伺服一直在发散边缘振荡
        # （窗口里看到的“卡爪疯狂动来动去”），0.2 才是调通的值。
        slide.armature = 0.2

        add_soft_jaws(jb, k, ang, jaw_open,
                      lambda g: _m(g, *jaw_mask))

    # One screw = one actuator. MjsEquality.data defaults to
    # [0,1,0,0,0,...], i.e. polycoef 0+1x => q2 = q1. Verified in
    # diag_chuck_health.check_jaw_coupling (structure AND empty-load
    # following), so do not "helpfully" zero it.
    act = spec.add_actuator()
    act.name = "jaw_act"
    act.target = "jaw_slide_0"
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    # 位置伺服 + 力限幅 = 丝杠：目标位置给行程，forcerange 给扭矩上限。
    act.set_to_position(kp=20000.0, kv=600.0)
    act.ctrlrange[:] = CHUCK_CTRL_RANGE
    act.forcerange[:] = (-CHUCK_FORCE_LIMIT, CHUCK_FORCE_LIMIT)

    for k in (1, 2):
        eq = spec.add_equality()
        eq.type = mujoco.mjtEq.mjEQ_JOINT
        eq.name1 = "jaw_slide_0"
        eq.name2 = "jaw_slide_%d" % k
        eq.solref[:] = [0.001, 1.0]
        eq.solimp[:] = [0.9, 0.99, 0.0001, 0.5, 100.0]

    return {"jaw_act": "jaw_act",
            "jaws": ["jaw_%d" % k for k in range(3)],
            "slides": ["jaw_slide_%d" % k for k in range(3)],
            "origin": np.asarray(origin)}


def make_scene():
    """建好并编译，返回 (model, data, 关键 id)。"""
    spec = build_scene()
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ids = {
        "shaft": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shaft"),
        "ee": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"),
        "gripper_act": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                         "gripper"),
        "jaws": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                   "jaw_%d" % k) for k in range(3)],
        # 一根丝杠只有一个 ctrl。保留列表形式，让诊断脚本里
        # `for a in ids["jaw_acts"]: data.ctrl[a] = ...` 原样可用。
        "jaw_acts": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                       "jaw_act")],
    }
    return model, data, ids


if __name__ == "__main__":
    model, data, ids = make_scene()

    print("=" * 74)
    print("场景建好了")
    print("=" * 74)
    print("  nq=%d  nv=%d  nu=%d  nbody=%d  ngeom=%d"
          % (model.nq, model.nv, model.nu, model.nbody, model.ngeom))
    print()
    print("  工件      : D 轴 Φ%.0f x %.0f mm，%.0f g"
          % (SHAFT_D * 1000, SHAFT_L * 1000, SHAFT_MASS * 1000))
    print("  工件位置  : %s" % np.round(data.xpos[ids["shaft"]], 3))
    print("  末端位置  : %s" % np.round(data.site_xpos[ids["ee"]], 3))
    print("  卡盘中心  : (%.3f, %.3f, %.3f)" % (LATHE_X, LATHE_Y, LATHE_Z))
    print()
    print("  执行器 %d 个：臂关节 + 1 个夹爪 + 1 个卡盘丝杠（三爪联动）"
          % model.nu)
    print("  自由度约束 %d 条：jaw_slide_0 = jaw_slide_1 = jaw_slide_2"
          % model.neq)
    for i in range(model.nu):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print("    %-16s ctrlrange=[%+.3f, %+.3f]"
              % (n, model.actuator_ctrlrange[i][0],
                 model.actuator_ctrlrange[i][1]))

    # ---- 渲染 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    rend = mujoco.Renderer(model, height=680, width=860)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    # 用 mjv_moveCamera 从默认相机出发"飞"到目标视角。
    # 直接改 azimuth/elevation 容易因为默认方向不同而看偏，这个是官方推荐做法。
    def make_cam(lookat, distance, azimuth, elevation):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)     # 先归位
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    # 相机方位角约定（实测，见 diag_camera_convention.py）：
    #   azimuth=0   -> 相机在 -x 方向
    #   azimuth=90  -> 相机在 -y 方向
    #   azimuth=180 -> 相机在 +x 方向
    #   elevation<0 -> 相机在上方俯视
    # 机器人底座在原点、朝 +x 干活，所以总览站在 -x 后方：azimuth ≈ 0~25
    VIEWS = [
        ("① 总览：机器人 / 工件 / 车床",
         make_cam([0.24, 0.11, 0.10], 1.00, 20, -26)),
        ("② 工件特写：D 轴在 V 形块上",
         make_cam([SHAFT_POS[0], SHAFT_POS[1], SHAFT_POS[2]],
                  0.26, 35, -22)),
        # 卡盘端面朝 -x（朝机器人），要正面看就得站在 -x 侧：azimuth ≈ 0
        ("③ 车床卡盘：三爪张开待装",
         make_cam([LATHE_X - 0.01, LATHE_Y, LATHE_Z],
                  0.30, 5, -18)),
    ]

    views = {}
    for title, cam in VIEWS:
        rend.update_scene(data, camera=cam)
        views[title] = rend.render().copy()
    rend.close()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.patch.set_facecolor("#0e1116")
    for ax, (title, _) in zip(axes, VIEWS):
        ax.imshow(views[title])
        ax.set_title(title, color="#c9d1d9", fontsize=11)
        ax.axis("off")

    fig.suptitle("场景：SO-101 抓 D 轴 Φ20x80 → 放进二轴车卡盘",
                 color="#e6edf3", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "scene_machining.png")
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print()
    print("  已保存 %s" % out)
    print("OK")
