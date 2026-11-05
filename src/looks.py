"""视觉外观配置：让 MuJoCo 渲染出来不像"方块堆"。

这个文件只做一件事：往 MjSpec 里加材质、纹理、灯光、天空盒。
**它不碰物理。** 删掉它，抓取结果一模一样，只是画面难看。

原理：MuJoCo 默认配置是"能看清就行"——
    默认头灯（headlight）打正面光，没有阴影，没有纹理，没有背景。
所以渲染出来就是一片平、一片黑、全是纯色块。

真正让画面"像那么回事"的，是这四件事：
    1. 阴影        (castshadow + scene flag)
    2. 三点布光    (主光 / 补光 / 轮廓光，而不是一盏头灯)
    3. 材质纹理    (粗糙度、金属度、棋盘地面)
    4. 天空盒      (背景不是死黑，有渐变)
"""
import mujoco


def make_pretty(spec):
    """给 MjSpec 加上一套'摄影棚'配置。原地修改，返回 spec。

    参数
    ----
    spec : mujoco.MjSpec
        已经加载好模型的 spec（通常是 MjSpec.from_file(...)）

    返回
    ----
    spec : 同一个 spec，已加上纹理 / 材质 / 灯光 / 天空盒
    """

    # ============ 1. 纹理 ============
    # 棋盘格地面：让空间有尺度感，物体不像飘在空中
    floor_tex = spec.add_texture()
    floor_tex.name = "floor_tex"
    floor_tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    floor_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    floor_tex.rgb1 = [0.26, 0.28, 0.32]      # 深格
    floor_tex.rgb2 = [0.34, 0.36, 0.40]      # 浅格
    floor_tex.width = 512
    floor_tex.height = 512

    # 天空盒渐变：上亮下暗，模拟环境光
    sky_tex = spec.add_texture()
    sky_tex.name = "sky_tex"
    sky_tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky_tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky_tex.rgb1 = [0.32, 0.40, 0.54]        # 天顶：冷灰蓝
    sky_tex.rgb2 = [0.08, 0.09, 0.12]        # 地平线：近黑
    sky_tex.width = 512
    sky_tex.height = 512

    # ============ 2. 材质 ============
    floor_mat = spec.add_material()
    floor_mat.name = "floor_mat"
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    floor_mat.texrepeat = [7, 7]             # 棋盘重复次数
    floor_mat.texuniform = True
    floor_mat.roughness = 0.9                # 粗糙，不反光
    floor_mat.metallic = 0.0
    floor_mat.specular = 0.15

    # 方块：金属红，有高光，看起来像零件而不是橡皮
    red_mat = spec.add_material()
    red_mat.name = "red_metal"
    red_mat.rgba = [0.70, 0.15, 0.12, 1.0]
    red_mat.roughness = 0.35
    red_mat.metallic = 0.65
    red_mat.reflectance = 0.12
    red_mat.specular = 0.6

    # 机械臂本体的补充材质（Franka 原厂是白+黑，这里让白更亮一点）
    arm_mat = spec.add_material()
    arm_mat.name = "arm_white"
    arm_mat.rgba = [0.92, 0.92, 0.93, 1.0]
    arm_mat.roughness = 0.45
    arm_mat.metallic = 0.1

    # ============ 3. 三点布光 ============
    # 关掉默认头灯。这是最关键的一步：
    # 头灯是"跟着相机走的正面光"，打出来完全没有立体感，
    # 而且会盖过你自己加的所有灯。
    spec.visual.headlight.active = False

    # 主光（key light）：右前上方，最亮，投阴影
    key = spec.worldbody.add_light()
    key.name = "key_light"
    key.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    key.pos = [0.50, -0.60, 1.75]
    key.dir = [-0.30, 0.34, -1.0]
    key.diffuse = [0.95, 0.93, 0.88]         # 略暖
    key.ambient = [0.18, 0.19, 0.22]
    key.specular = [0.35, 0.35, 0.35]
    key.castshadow = True

    # 补光（fill light）：左前方，冷色，弱，不投影
    # 作用：把阴影里的死黑拉回来，看清结构
    fill = spec.worldbody.add_light()
    fill.name = "fill_light"
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.pos = [1.15, 0.85, 1.30]
    fill.dir = [-0.55, -0.45, -0.90]
    fill.diffuse = [0.24, 0.30, 0.42]        # 偏冷
    fill.ambient = [0.06, 0.07, 0.10]
    fill.specular = [0.0, 0.0, 0.0]
    fill.castshadow = False

    # 轮廓光（rim light）：从背后打，勾出边缘
    # 作用：让机械臂从背景里"跳"出来，这是产品图的标准做法
    rim = spec.worldbody.add_light()
    rim.name = "rim_light"
    rim.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    rim.pos = [-0.85, 0.75, 1.05]
    rim.dir = [0.62, -0.52, -0.58]
    rim.diffuse = [0.20, 0.23, 0.30]
    rim.ambient = [0.0, 0.0, 0.0]
    rim.castshadow = False

    # ============ 4. 渲染质量 ============
    spec.visual.map.haze = 0.10              # 远景雾化，增加纵深
    spec.visual.map.fogstart = 2.0
    spec.visual.map.fogend = 5.0
    spec.visual.quality.shadowsize = 4096    # 阴影分辨率（默认 1024，糊）
    spec.visual.global_.fovy = 42            # 视野角，小一点透视更舒服

    # 离屏渲染缓冲区。必须 >= 你要输出的分辨率，否则 Renderer 会报错
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960

    return spec


def enable_scene_flags(renderer):
    """打开渲染器的场景特效开关。必须在 update_scene 之前调用。

    MuJoCo 的 Renderer 默认这些是关的，开了之后才有阴影和背景。
    """
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 1
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
    return renderer
