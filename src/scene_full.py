"""scene_full.py -- SO-101 + SimReady bench vise + workpiece, full scene.

Run:
    D:\\robot-env\\Scripts\\python.exe scene_full.py

===============================================================
COLLISION GROUPS (the single most important design in this file)
===============================================================
Measured: the distance from the SO-101 gripper pad center to the bottom of
the wrist geometry is only 8.2 mm. The moment the pads enter a jaw slot the
wrist enters too. The wrist is 65 mm wide along y, far wider than any usable
slot. Result: the wrist hits the fixture top face and the arm stalls --
measured residual jumped 11 mm -> 127 mm.

That is a structural limit of this arm, not a tunable number. Real fixtures
are designed to clear the wrist. Same effect achieved here with bitmasks:

    group         contype  conaffinity   purpose
    ---------------------------------------------------------
    robot            1         3         must touch the workpiece
    workpiece        2         3         grasped, rests on the table
    fixture          4         2         touches workpiece ONLY
    table/floor      4         3         touches robot and workpiece

    rule: (ct_A & ca_B) || (ct_B & ca_A) != 0

    robot(1,3) vs fixture(4,2) = (1&2)|(4&3) = 0|0 = 0  -> no contact
    robot(1,3) vs table(4,3)   = (1&3)|(4&3) = 1|0 = 1  -> contact
    work(2,3)  vs fixture(4,2) = (2&2)|(4&3) = 2|0 = 2  -> contact
    work(2,3)  vs table(4,3)   = (2&3)|(4&3) = 2|0 = 2  -> contact

===============================================================
FIXTURE GEOMETRY EVOLUTION (each row was a measured failure)
===============================================================
  1. jaws split along x (parallel to shaft) -> pushed from both ends
  2. narrow 25 mm slot + square step        -> ejected at 5 mm offset
  3. narrow slot + 32 deg chamfer           -> wedged on the lip at 4 mm
  4. pure V, 45 deg point bottom            -> 9/15, stuck at 10-25 mm
  5. trapezoid slot (V with flat bottom)    -> 2/11, hangs at z=20
  6. box stairs, 10 steps                   -> 4/11, rides a step edge
  7. box stairs, 24 steps, 35 deg           -> 6/11, stuck at z=11-20
  8. twin rollers                           -> 2/13, ejected on hard contact
  9. shallow slot platform (ROUTE B)        -> current

Route B abandons self-centering entirely. The arm's IK residual is under
0.3 mm, so the fixture only has to SUPPORT the shaft, not steer it.
A 24 mm wide / 10 mm deep slot for a D20 shaft leaves 2 mm clearance per
side. No slopes, no rollers, no catch points -- vertical walls cannot
centre anything. Built from three boxes: a base plate plus two walls.
A concave shape must never come from a mesh convex hull, which always
fills the cavity.
"""
import os

import numpy as np
import mujoco

from simready_loader import (
    load_asset, place_asset, register_meshes, total_mass,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
SO101_ASSETS = os.path.join(
    ROOT, "models", "_tmp_so101", "lerobot_env_so101-0.4.0",
    "lerobot_env_so101", "assets")
SO101_XML = os.path.join(SO101_ASSETS, "so101_new_calib.xml")
# The 13 STL meshes actually live one level deeper, in assets/assets/.
# MuJoCo has a single global meshdir, so our generated OBJs go there too.
SO101_MESH_DIR = os.path.join(SO101_ASSETS, "assets")

TABLE_TOP_Z = 0.0
TABLE_SIZE = (0.60, 0.50, 0.024)

CT_ROBOT = (1, 3)
CT_WORK = (2, 3)
CT_FIXTURE = (4, 2)
CT_STATIC = (4, 3)
# 机床实体（卡盘本体 / 床头箱 / 主轴）：只挡机器人，不挡工件。
# 之前这里是 (0,0)，即"完全不参与碰撞"——所以爪子能整块从卡盘里穿过去，
# 画面上看得见穿模，而指标照过（工件只和三根爪接触）。
#   vs 机器人 (1,3):  ca=1 & ct=1 = 1  -> 碰
#   vs 工件   (2,3):  ca=1 & ct=2 = 0, ca=3 & ct=8 = 0  -> 不碰
# 工件仍然不碰盘面，才能沿轴线进给到卡爪里；机器人碰，才不能穿模。
CT_MACHINE = (8, 1)
# Feed-scene robot shell mask: it receives machine/static contacts via bits
# 8/4, but does not receive the workpiece bit 2. The feed scene has no
# separate fixture, and only fingertip pads / added V blades use CT_ROBOT to
# touch the rod. Applied only when with_lathe=True so other demos keep their
# original collision behavior.
CT_ROBOT_SHELL = (8, 12)
ROBOT_BODY_NAMES = {
    "base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
    "moving_jaw_so101_v1",
}

# ---- workpiece: D-shaft 20 x 80 mm with a milled flat so it will not roll
SHAFT_D = 0.020
SHAFT_L = 0.080
SHAFT_FLAT = 0.004
SHAFT_MASS = 0.197
SHAFT_POS = [0.26, -0.13, TABLE_TOP_Z + SHAFT_D / 2]

# ---- fixture: shallow slot platform (route B, no self-centering) ----
SLOT_W = 0.024                # slot width along y, D20 shaft + 2 mm/side
# Wall height must reach the shaft axis. A 3 mm wall stops nothing: at
# 3 mm above the floor the D20 cross-section is only 7.45 mm wide, far
# short of the 12 mm half-slot, so the cylinder rolls right over it.
# A 10 mm wall has its lip at rest_z, where the section is the full
# 10 mm radius -- the shaft is then trapped within +-2 mm of centre.
SLOT_D = 0.010                # slot depth / wall height
PLATFORM_L = 0.130            # length along x
PLATFORM_W = 0.070            # overall width along y, incl. both walls
FIXTURE_X = 0.16
FIXTURE_Y = 0.075

# Keep the workpiece axis at the same height as the old roller V-block
# (22.2 mm) so every carry/place target in demo_08 / demo_09 stays valid.
REST_Z = 0.0222

# Bench vise sits at x=0.20, not 0.30: SO-101 reach is only ~40 cm. At 0.30
# the arm was fully extended during the whole carry and end-effector
# accuracy collapsed (measured 160 mm residual).
VISE_POS = [0.20, 0.17, TABLE_TOP_Z]


def slot_geometry():
    """Solve the slot from the rest height (kept at the old V-block value).

    base plate top = slot floor; two walls rise SLOT_D above it. The inner
    wall faces sit at y = FIXTURE_Y +- SLOT_W/2, giving 2 mm clearance on
    each side of the D20 shaft.
    """
    Rw = SHAFT_D / 2
    slot_bottom = REST_Z - Rw
    platform_top = slot_bottom + SLOT_D
    wall_w = (PLATFORM_W - SLOT_W) / 2
    return {
        "Rw": Rw,
        "slot_w": SLOT_W,
        "slot_d": SLOT_D,
        "rest_z": REST_Z,
        "slot_bottom": slot_bottom,
        "platform_top": platform_top,
        "wall_w": wall_w,
        "wall_y": SLOT_W / 2 + wall_w / 2,
    }


def jaw_center():
    """Axis position of the workpiece seated in the slot."""
    return np.array([FIXTURE_X, FIXTURE_Y, slot_geometry()["rest_z"]])


def jaw_geometry():
    g = slot_geometry()
    return {
        "rest_z": g["rest_z"],
        "slot_w": g["slot_w"],
        "slot_d": g["slot_d"],
        "slot_bottom": g["slot_bottom"],
        "platform_top": g["platform_top"],
        "wall_w": g["wall_w"],
        "wall_y": g["wall_y"],
        "platform_w": PLATFORM_W,
        "platform_l": PLATFORM_L,
        "top_gap": g["slot_w"],
    }


def _cam_quat(x_axis, y_axis):
    """Quat for a camera given its world x/y axes (MuJoCo cam looks down -z)."""
    x = np.asarray(x_axis, float)
    y = np.asarray(y_axis, float)
    x /= np.linalg.norm(x)
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


# Official lerobot/so101-tabletop-mujoco world, copied verbatim from
# so101_tabletop.xml (Apache-2.0). Its table surface is z = 0 -- the same
# datum this project already calibrates every pose to -- so adopting it
# changes NOTHING about IK targets, the lathe origin or the V-jaw geometry.
OFFICIAL_TABLE_SIZE = (0.80, 0.80, 0.03)     # 80 x 80 cm top, 3 cm thick
OFFICIAL_TABLE_POS = (0.20, 0.0, -0.015)
OFFICIAL_FLOOR_Z = -0.75                      # table stands on 75 cm legs
OFFICIAL_CAMS = (
    ("front", (0.5, 0.0, 0.2), (0, 1, 0), (-0.493, 0, 0.87), 60.0),
    ("top", (0.2, 0.0, 0.48), (0, -1, 0), (1, 0, 0), 60.0),
)
# `front` 是官方给"方块进盒子"定的机位：它在 x=+0.5，也就是卡盘**背面**。
# 我们这类车床工位最关键的一段（工件插进三爪）正好被床头箱挡死 —— 实测录
# 出来的 gif 里只能看见手臂把一个圆柱推到卡盘后面去。
# 所以另给一个 `cell` 机位，从 -y 侧斜看工作区，卡盘口朝 -x 正好对着它。
CELL_CAM_POS = (0.02, -0.52, 0.28)
CELL_CAM_TARGET = (0.21, 0.04, 0.075)
CELL_CAM_FOVY = 48.0


def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
    """Camera quat for a camera at `pos` pointing at `target` (looks down -z)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    z = -f                                    # MuJoCo cameras look along -z
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-8:              # looking straight up/down
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q
# Wrist camera: official `camera_mount` body sits at identity under `gripper`,
# so the same offset can be hung straight on the gripper body. Our arm model
# (lerobot_env_so101-0.4.0) does not ship this camera.
WRIST_CAM_POS = (-0.00296, 0.05436, -0.05310)
WRIST_CAM_QUAT = (0.958280, -0.285792, -0.001556, 0.004547)
WRIST_CAM_FOVY = 61.0


ARM_SERVO_KV = 2.6      # Nm·s/rad，位置伺服缺失的微分项（见 _damp_arm_servos）
# 2026-11-07：用户反馈「找轴心吨吨吨几下」= 接触段欠阻尼 hunting
# （IEEE IE Magazine / Loughborough 综述：先加阻尼，别加比例增益）。
# 全送料扫描 2.6/3.5/4/5/6/8：
#   末端抖动 RMS  0.046 -> 0.036/0.033/0.031/0.029/0.027 mm
#   最大抖动      0.878 -> 0.390/0.378/0.377/0.400/0.501 mm
#   进给段起步次数 2/2/1 -> 1/1/1（吨吨吨从 5 次降到 3 次）
#   最终同轴      1.26 -> 1.00/0.92/0.87/0.84/0.70 mm
# 6 以上 grasp_tilt 涨到 1.7 deg、8 直接 27.2 deg；5.0 是最佳点。
# 但【不能全局改】：demo_09「搬运到槽上方」要求 xy<5 mm，阻尼一加大
# 稳态跟踪误差就涨（2.6 -> 4.0 mm，3.5 -> 5.4，4 -> 6.1，5 -> 5.9，
# 6 -> 6.6），回归会掉到 8/9。所以高阻尼只由送料场景经
# build_scene(arm_kv=...) 单独注入，其它场景保持 2.6。
ARM_SERVO_KP = 70.0     # Nm/rad；XML 里是 17.8（下垂 17 mm），官方新版是 998
ARM_ACT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll")


def _damp_arm_servos(spec, kv=ARM_SERVO_KV):
    """给 SO101 的五个位置伺服补上微分项（Kd）。

    so101_new_calib.xml 里 <position kp="17.8"/> 只给了刚度、没给阻尼，
    XML 自己那行注释也写着「Kd 与 Kp 不是 1:1，需要按 lerobot Kp=16 Kd=32
    重新辨识」—— 也就是说这一版模型只做了 Kp 辨识，Kd 是缺的。

    缺 Kd 的后果实测（elbow_flex 阶跃 0.15 rad，静置 0.8 s 后给阶跃）：

        kv=0.0  超调 45.6%   0.36 s 一个来回，末端看就是「一跳一跳」
        kv=1.6  超调 13.9%   同样的阶跃，一次到位不再回弹

    超调 45% 意味着每次加速/减速，末端都会先冲过头再荡回来，频率约
    2.8 Hz —— 窗口里看到的「不是顿，是一跳一跳」就是这个，和轨迹规划
    无关：轨迹再光滑，伺服自己会振。关节自带的 damping=0.6 太小
    （临界阻尼要 2.4 左右）。

    只动五个臂关节，不动 gripper：夹持力是按现在这组参数标定的。
    """
    kv = float(os.environ.get("SO101_KV", kv))
    kp_env = os.environ.get("SO101_KP")
    for act in spec.actuators:
        if act.name in ARM_ACT_NAMES:
            kp = ARM_SERVO_KP if kp_env is None else float(kp_env)
            act.set_to_position(kp=kp, kv=kv)


def build_scene(shaft_mass=None, with_lathe=False, lathe_origin=None,
                with_vise=True, with_fixture=True, with_shaft_flat=True,
                machine_solid=True, official_world=False, arm_kv=None):
    """Build the merged scene.

    The three with_* switches default to True so demo_08 / demo_09 and every
    existing diagnostic keep their exact baseline. The feed scene turns them
    off: no bench vise, no shallow slot platform, and a PLAIN CYLINDER
    workpiece (no milled flat). See diag_feed_chuck.build_model().
    """
    if shaft_mass is None:
        shaft_mass = SHAFT_MASS

    spec = mujoco.MjSpec.from_file(SO101_XML)
    _damp_arm_servos(spec,
                     kv=ARM_SERVO_KV if arm_kv is None else arm_kv)
    mesh_out = SO101_MESH_DIR
    spec.meshdir = SO101_MESH_DIR
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200
    spec.visual.headlight.ambient = [0.26, 0.26, 0.29]
    spec.visual.headlight.diffuse = [0.52, 0.52, 0.55]

    wb = spec.worldbody

    def _mask(g, ct, ca):
        g.contype = ct
        g.conaffinity = ca

    # ---- floor + table ----
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.rgba = [0.27, 0.29, 0.33, 1.0]
    if official_world:
        # Official room: the table stands on 75 cm legs, so the ground plane
        # is far below the work surface. Nothing in the feed flow goes there
        # unless the part is dropped, which is exactly when we want to see it.
        floor.size = [1.0, 1.5, 0.05]
        floor.pos = [0.0, 0.0, OFFICIAL_FLOOR_Z]
    else:
        floor.size = [2.0, 2.0, 0.02]
        floor.pos = [0.0, 0.0, TABLE_TOP_Z - TABLE_SIZE[2]]
    _mask(floor, *CT_STATIC)

    table = wb.add_geom()
    table.name = "table"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    if official_world:
        table.size = [OFFICIAL_TABLE_SIZE[0] / 2, OFFICIAL_TABLE_SIZE[1] / 2,
                      OFFICIAL_TABLE_SIZE[2] / 2]
        table.pos = list(OFFICIAL_TABLE_POS)
    else:
        table.size = [TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_SIZE[2] / 2]
        table.pos = [0.28, 0.02, TABLE_TOP_Z - TABLE_SIZE[2] / 2]
    table.rgba = [0.37, 0.39, 0.43, 1.0]
    _mask(table, *CT_STATIC)

    if official_world:
        # Three 2 m walls, visual only (contype 0): they exist so a rendered
        # frame is recognisable as the official tabletop scene, not so the
        # arm can bump into a room.
        for i, (size, pos) in enumerate(
                (([0.02, 1.5, 1.0], [-1.0, 0.0, 0.25]),
                 ([1.0, 0.02, 1.0], [0.0, 1.5, 0.25]),
                 ([1.0, 0.02, 1.0], [0.0, -1.5, 0.25]))):
            w = wb.add_geom()
            w.name = "room_wall_%d" % i
            w.type = mujoco.mjtGeom.mjGEOM_BOX
            w.size = size
            w.pos = pos
            w.rgba = [0.85, 0.84, 0.80, 1.0]
            _mask(w, 0, 0)

    if official_world:
        # Same two world cameras as so101_tabletop.xml, same poses.
        for name, pos, xa, ya, fovy in OFFICIAL_CAMS:
            c = wb.add_camera()
            c.name = name
            c.pos = list(pos)
            c.quat = _cam_quat(xa, ya)
            c.fovy = fovy
        cc = wb.add_camera()
        cc.name = "cell"
        cc.pos = list(CELL_CAM_POS)
        cc.quat = _look_at(CELL_CAM_POS, CELL_CAM_TARGET)
        cc.fovy = CELL_CAM_FOVY
        # Wrist view: what the official env calls `wrist_cam`, hung on the
        # gripper body at the offset its `camera_mount` body uses (identity
        # pose, so the two are equivalent).
        try:
            gc = spec.body("gripper").add_camera()
            gc.name = "wrist_cam"
            gc.pos = list(WRIST_CAM_POS)
            gc.quat = list(WRIST_CAM_QUAT)
            gc.fovy = WRIST_CAM_FOVY
        except Exception as exc:            # noqa: BLE001 - model may vary
            print("    wrist_cam not attached: %s" % exc)


    # ---- lights ----
    for nm, pos, dir_, dif, sh in [
        ("key", [0.02, -0.50, 0.90], [0.25, 0.45, -1.0], [1.0, 1.0, 1.0], True),
        ("fill", [0.70, 0.55, 0.65], [-0.4, -0.4, -0.85], [0.5, 0.5, 0.55], False),
        ("rim", [-0.30, 0.30, 0.45], [0.7, -0.4, -0.5], [0.32, 0.33, 0.38], False),
    ]:
        L = wb.add_light()
        L.name = nm
        L.pos = pos
        L.dir = dir_
        L.diffuse = dif
        L.castshadow = sh

    # ---- real SimReady bench vise (optional; the feed scene removes it) ----
    vv_name = vc_name = None
    if with_vise:
        print("Loading SimReady asset:")
        vise = load_asset("equipment_bench_vise_a01")
        print("    vise mass %.2f kg (from SimReady physics)" % total_mass(vise))

        # Local frame: disc body Y in [0, 0.027], jaws extend along +Y,
        # leadscrew along X. Rotating +90 deg about Z maps local +Y to
        # world -X, so the jaw opening faces the robot. Picked by
        # diag_vise_orient.py, not by reasoning.
        vise_quat = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]

        # VISUAL ONLY (ct=0). The convex hull fills the jaw slot into a
        # solid block: measured with diag_shaft_slip.py, the workpiece hit
        # vise_sr_col on the way down and was thrown out (pad-to-workpiece
        # error grew 17 -> 52 mm). A hull always fills concavities and that
        # is not tunable, so functional collision comes from the rollers.
        vb, vv_name, vc_name = place_asset(
            spec, wb, "vise_sr", vise,
            pos=VISE_POS, quat=vise_quat,
            rgba=(0.40, 0.44, 0.50, 1.0),
            out_dir=mesh_out, ct=0, ca=0,
        )
    else:
        print("    vise: removed (with_vise=False)")

    # ---- fixture: shallow slot platform (optional; feed scene drops it) --
    if with_fixture:
        g = slot_geometry()
        print("    fixture (shallow slot platform, no self-centering):")
        print("      slot %.0f mm wide x %.0f mm deep, platform %.0f x %.0f mm"
              % (SLOT_W * 1000, SLOT_D * 1000,
                 PLATFORM_L * 1000, PLATFORM_W * 1000))
        print("      slot floor z %.1f mm, platform top z %.1f mm"
              % (g["slot_bottom"] * 1000, g["platform_top"] * 1000))
        print("      workpiece D%.0f mm rests with axis at z %.1f mm"
              % (SHAFT_D * 1000, g["rest_z"] * 1000))

        # base plate: its top face IS the slot floor
        base = wb.add_geom()
        base.name = "slot_base"
        base.type = mujoco.mjtGeom.mjGEOM_BOX
        base.size = [PLATFORM_L / 2, PLATFORM_W / 2, g["slot_bottom"] / 2]
        base.pos = [FIXTURE_X, FIXTURE_Y, TABLE_TOP_Z + g["slot_bottom"] / 2]
        base.rgba = [0.42, 0.45, 0.50, 1.0]
        base.friction = [0.8, 0.01, 0.001]
        _mask(base, *CT_FIXTURE)

        # two walls: inner faces at y = FIXTURE_Y +- SLOT_W/2
        for side, sy in (("L", -1), ("R", +1)):
            w = wb.add_geom()
            w.name = "slot_wall_%s" % side
            w.type = mujoco.mjtGeom.mjGEOM_BOX
            w.size = [PLATFORM_L / 2, g["wall_w"] / 2, SLOT_D / 2]
            w.pos = [FIXTURE_X,
                     FIXTURE_Y + sy * g["wall_y"],
                     TABLE_TOP_Z + g["slot_bottom"] + SLOT_D / 2]
            w.rgba = [0.56, 0.59, 0.65, 1.0]
            w.friction = [0.8, 0.01, 0.001]
            _mask(w, *CT_FIXTURE)
    else:
        print("    fixture: removed (with_fixture=False)")

    if with_vise:
        register_meshes(spec, [vv_name, vc_name])

    # ---- workpiece: D-shaft, free body ----
    shaft = wb.add_body()
    shaft.name = "shaft"
    shaft.pos = SHAFT_POS
    shaft.quat = [0.7071068, 0.0, 0.7071068, 0.0]
    shaft.add_freejoint(name="shaft_free")

    sg = shaft.add_geom()
    sg.name = "shaft_geom"
    sg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    sg.size = [SHAFT_D / 2, SHAFT_L / 2, 0.0]
    sg.mass = shaft_mass
    sg.rgba = [0.72, 0.75, 0.79, 1.0]
    sg.friction = [1.2, 0.02, 0.002]
    _mask(sg, *CT_WORK)

    if with_shaft_flat:
        flat = shaft.add_geom()
        flat.name = "shaft_flat"
        flat.type = mujoco.mjtGeom.mjGEOM_BOX
        # The shaft body is rotated: local z -> world x (shaft axis),
        # local x -> -world z. So the 80 mm half-length belongs in size[2]
        # and the 4 mm thickness in size[0]. With these swapped the "milled
        # flat" becomes an 80 mm TALL vertical plate that punches through
        # the table and holds the shaft up in mid-air -- measured world
        # bbox z[-0.030, 0.050] instead of a thin strip on the cylinder.
        flat.size = [SHAFT_FLAT / 2, SHAFT_D * 0.42, SHAFT_L / 2]
        flat.pos = [-(SHAFT_D / 2 - SHAFT_FLAT / 2), 0.0, 0.0]
        flat.mass = 0.0
        flat.rgba = [0.80, 0.82, 0.86, 1.0]
        # VISUAL ONLY. A box cannot sit flush on a cylinder: its corners
        # stick ~3 mm past the D20 envelope (sqrt(8.4^2 + 10^2) = 13.1 mm).
        # Those corners land first, kick the shaft sideways and make it
        # roll -- measured in diag_slot_alone.py: every drop converged to
        # y = +4 mm with quat rotated, and the shaft never touched a slot
        # wall. The slot walls already provide lateral restraint, so let
        # the cylinder own all contact and keep the flat purely for looks.
        _mask(flat, 0, 0)
    else:
        print("    shaft_flat: omitted -> PLAIN CYLINDER D%.0f"
              % (SHAFT_D * 1000))

    # ---- robot collision masks ----
    # The stock collision meshes default to (1, 1), which also makes the
    # wrist/motor shells collide with the free workpiece (2, 3). Keep those
    # shells solid against the machine, but reserve workpiece contact for the
    # pads and explicitly added grasp geometry.
    npad = 0
    nshell = 0
    for body in spec.bodies:
        for gg in body.geoms:
            if gg.name and "pad" in gg.name:
                gg.contype = CT_ROBOT[0]
                gg.conaffinity = CT_ROBOT[1]
                npad += 1
                continue
            if (with_lathe and body.name in ROBOT_BODY_NAMES
                    and gg.group == 3):
                gg.contype = CT_ROBOT_SHELL[0]
                gg.conaffinity = CT_ROBOT_SHELL[1]
                nshell += 1
    print("    normalized %d pads to %s and %d robot shell geoms to %s"
          % (npad, CT_ROBOT, nshell, CT_ROBOT_SHELL))

    if with_lathe:
        # Bolt the lathe on as a workstation. Default OFF: demo_08 /
        # demo_09 and every existing diagnostic keep their baseline.
        # CT_FIXTURE (4,2) makes the jaws touch the workpiece but NOT the
        # gripper pads, which is the same trick used for the slot fixture
        # (the wrist sits only 8.2 mm above the pads). The shaft must not hit
        # the disc or it can never seat -- but the ARM must, or the gripper
        # walks straight through the chuck. CT_MACHINE gives exactly that:
        # blocks contype=1 (robot), ignores contype=2 (workpiece).
        import scene_machining as SM
        if lathe_origin is None:
            lathe_origin = (0.26, 0.12, TABLE_TOP_Z + 0.060)
        SM.add_lathe(
            spec,
            origin=lathe_origin,
            table_z=TABLE_TOP_Z,
            jaw_mask=CT_FIXTURE,
            # machine_solid=False reproduces the old pass-through scene. It
            # exists ONLY so the legacy flat-pad baseline can be re-run and
            # compared; every new path must keep the machine solid.
            static_mask=CT_MACHINE if machine_solid else (0, 0),
        )

    return spec


def make_scene(shaft_mass=None, with_lathe=False, lathe_origin=None,
               with_vise=True, with_fixture=True, with_shaft_flat=True):
    spec = build_scene(shaft_mass, with_lathe=with_lathe,
                       lathe_origin=lathe_origin, with_vise=with_vise,
                       with_fixture=with_fixture,
                       with_shaft_flat=with_shaft_flat)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def make_camera():
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = [0.22, 0.02, 0.09]
    c.distance = 0.92
    c.azimuth = 22
    c.elevation = -26
    return c


def render(model, data, path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cnfont import setup_cn_font
    setup_cn_font()

    rend = mujoco.Renderer(model, height=660, width=880)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    def cam(lookat, dist, az, el):
        c = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(c)
        c.lookat[:] = lookat
        c.distance = dist
        c.azimuth = az
        c.elevation = el
        return c

    jaw = jaw_center()
    VIEWS = [
        ("scene: SO-101 + SimReady vise + D-shaft",
         cam([0.20, 0.04, 0.10], 1.00, 22, -26)),
        ("fixture: shallow slot platform",
         cam([jaw[0], jaw[1], jaw[2] + 0.03], 0.36, 8, -14)),
        ("workpiece D20 x 80",
         cam(list(SHAFT_POS), 0.22, 35, -22)),
    ]

    imgs = []
    for title, c in VIEWS:
        rend.update_scene(data, camera=c)
        imgs.append((title, rend.render().copy()))
    rend.close()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    fig.suptitle("Full scene: real SimReady assets + SO-101", fontsize=15,
                 fontweight="bold")
    for ax, (t, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = path or os.path.join(OUT, "scene_full.png")
    plt.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
    print("  saved %s" % out)
    return out


if __name__ == "__main__":
    print("=" * 74)
    print("Full scene: SO-101 + real SimReady bench vise")
    print("=" * 74)
    print()

    model, data = make_scene()

    print()
    print("model stats:")
    print("  nq=%d nv=%d nu=%d nbody=%d ngeom=%d nmesh=%d"
          % (model.nq, model.nv, model.nu, model.nbody,
             model.ngeom, model.nmesh))
    print()

    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    print("run 2 s, check the workpiece settles on the table:")
    for i in range(1000):
        mujoco.mj_step(model, data)
        if i % 250 == 0:
            q = data.qpos[sq:sq + 3]
            print("  t=%.2fs  shaft z=%.4f  contacts=%d"
                  % (data.time, q[2], data.ncon))

    print()
    render(model, data)
    print("OK")
