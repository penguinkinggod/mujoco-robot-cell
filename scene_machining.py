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
ASSETS = os.path.join(HERE, "models", "_tmp_so101",
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
#   卡爪占 x ∈ [0.230, 0.258]，卡盘盘体 x ∈ [0.260, 0.295]，都在 y > 0.15 一侧。
#   工件放到 y < 0 一侧，x 与卡盘同区也没关系，机器人从中间转过去。
SHAFT_POS = np.array([0.28, -0.14, TABLE_TOP_Z + SHAFT_D / 2])

# 二轴车：主轴 + 三爪卡盘
# 位置由可达性决定：实测 SO-101 在 0.30~0.36 m 处最舒服。
# 卡盘轴线沿 x（工件从 -x 方向送进去，和真实上下料一致）。
LATHE_X = 0.26                # 卡盘端面 x（卡爪伸出的那一面）
LATHE_Y = 0.24                # 卡盘中心 y（放左侧，避开工作台）
LATHE_Z = TABLE_TOP_Z + 0.06  # 卡盘轴线高度（离台面 60 mm）
CHUCK_R = 0.055               # 卡盘半径（Φ110，小型二轴车）
CHUCK_L = 0.035               # 卡盘厚度
JAW_W = 0.018                 # 卡爪截面宽
JAW_LEN = 0.028               # 卡爪沿轴向长度
JAW_OPEN = 0.022              # 张开时卡爪内表面到轴线的距离


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
        slide.range = [-0.015, 0.008]
        # damping 是向量，armature 是标量，类型不一样（踩过）
        slide.damping[:] = 30.0
        slide.armature = 0.01

        # 卡爪主体：截面是 (轴向长, 径向厚, 切向宽)
        jgeom = jb.add_geom()
        jgeom.name = "jaw_geom_%d" % k
        jgeom.type = mujoco.mjtGeom.mjGEOM_BOX
        jgeom.size = [JAW_LEN / 2, JAW_W / 2, JAW_W / 2]
        jgeom.rgba = [0.72, 0.68, 0.52, 1.0]
        jgeom.friction = [1.8, 0.06, 0.004]
        jgeom.contype = 2
        jgeom.conaffinity = 3

        jaw_bodies.append(jb)

    # 卡爪执行器：位置伺服，往里夹
    # 关键：用 MuJoCo 的 <position> 简写等价形式。
    # 手写 gainprm/biasprm 很容易把符号搞反（第一版就是这样，
    # 卡爪不动、还把轴顶飞了）。正确的关系是：
    #     force = gainprm[0]*(ctrl - qpos) - biasprm[2]*qvel
    # 也就是 biasprm[1] 必须 = -gainprm[0]，biasprm[2] = -阻尼。
    # 用 set_to_position 让 MuJoCo 自己填，不手写。
    for k in range(3):
        act = spec.add_actuator()
        act.name = "jaw_act_%d" % k
        act.target = "jaw_slide_%d" % k
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.set_to_position(kp=4000.0, kv=120.0)
        act.ctrlrange = [-0.015, 0.008]
        act.forcerange = [-500.0, 500.0]

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
        "jaw_acts": [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                       "jaw_act_%d" % k) for k in range(3)],
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
    print("  执行器 %d 个：6 个臂关节 + 1 个夹爪 + 3 个卡爪" % model.nu)
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
    out = os.path.join(HERE, "scene_machining.png")
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print()
    print("  已保存 %s" % out)
    print("OK")
