"""scene_simready.py —— 用真实 NVIDIA SimReady 资产搭 MuJoCo 场景。

跟 scene_machining.py 的区别
---------------------------
scene_machining.py 里的台钳、V 形块、工件都是我手搓的 primitive。
这个脚本用真实的 SimReady 资产：
    - equipment_bench_vise_a01  真台钳（35cm，有固定钳口+滑动钳口+丝杠）
    - drivetrain_axle_shaft_d01 真轴（62cm，Φ65mm）

关键工程问题：三角网格不能直接当 MuJoCo 碰撞体
------------------------------------------------
106,162 个三角面做碰撞检测，每步都要遍历，帧率会掉到个位数。
正确做法（这也是 MuJoCo 官方推荐）：
    视觉 geom  -> 用完整 mesh，contype=0 conaffinity=0（只显示不碰撞）
    碰撞 geom  -> 用 convex hull 或者 primitive（box/cylinder），便宜

所以每个资产要拆成"看得见的"和"撞得到的"两套几何。

跑法：
    D:\\robot-env\\Scripts\\python.exe scene_simready.py
"""
import os

import numpy as np
import trimesh
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
OBJ_DIR = os.path.join(ROOT, "models", "simready", "obj")
SO101_DIR = os.path.join(
    ROOT, "models", "_tmp_so101", "lerobot_env_so101-0.4.0",
    "lerobot_env_so101", "assets")


# ============================================================
# 资产加载：读 OBJ，返回 (顶点数组, 面数组, 包围盒)
# ============================================================
def load_obj(path):
    V, F = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                V.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
                for k in range(1, len(idx) - 1):
                    F.append((idx[0], idx[k], idx[k + 1]))
    return np.array(V, dtype=np.float64), np.array(F, dtype=np.int32)


def write_obj(path, V, F, name="part"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 自动生成：%s\n" % name)
        for v in V:
            f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for t in F:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


# ============================================================
# 用 trimesh 的凸包减面：106k 面 -> 十几面
# ============================================================
def make_convex_hull_mesh(V, F, name, simplify=32):
    """把三角网格简化成凸包，作为碰撞体。

    为什么要这样做：
        106,162 面的 mesh 当碰撞体，MuJoCo 每步都要做三角-三角检测，
        帧率会掉到个位数。
        凸包把它压到 1362 面（78 倍），而且是凸的，
        碰撞检测算法可以走快路径。

    注意：凸包会"填平"凹槽。台钳的钳口凹槽会被填上，
          所以真实夹持要用单独的 box 碰撞体（下一步做）。
    """
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    try:
        hull = mesh.convex_hull
        # 可选减面：trimesh 的 fast_simplification 依赖不一定装了，
        # 1362 面本身已经可用，失败就用原凸包。
        if simplify and len(hull.faces) > simplify * 4:
            try:
                hull = hull.simplify_quadric_decimation(simplify)
            except Exception:
                pass
        return np.asarray(hull.vertices), np.asarray(hull.faces)
    except Exception as e:
        print("    凸包失败（%s），改用原网格" % type(e).__name__)
        return V, F


# ============================================================
# 场景构建
# ============================================================
TABLE_TOP_Z = 0.0
TABLE_SIZE = (0.62, 0.52, 0.024)

# 台钳放在工作台左侧
VISE_XY = (0.30, 0.10)
# 轴放在台钳钳口里（沿 x 方向）
SHAFT_XY = (0.30, 0.10)

MESH_DIR = os.path.join(ROOT, "models", "simready", "mjcf_meshes")


def build_scene():
    spec = mujoco.MjSpec()
    spec.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_AUTO

    # 编译时从 MESH_DIR 找 mesh 文件（属性名是 meshdir，不是 assetdir）
    spec.meshdir = MESH_DIR
    spec.modelfiledir = ""

    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200
    spec.visual.headlight.ambient = [0.24, 0.24, 0.27]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.53]

    wb = spec.worldbody

    def _mask(g, ct=1, ca=3):
        g.contype = ct
        g.conaffinity = ca

    # ---------- 地面 ----------
    floor = wb.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [2.0, 2.0, 0.02]
    floor.pos = [0.0, 0.0, TABLE_TOP_Z - TABLE_SIZE[2]]
    floor.rgba = [0.26, 0.28, 0.32, 1.0]
    _mask(floor, 1, 3)

    # ---------- 工作台 ----------
    table = wb.add_geom()
    table.name = "table"
    table.type = mujoco.mjtGeom.mjGEOM_BOX
    table.size = [TABLE_SIZE[0] / 2, TABLE_SIZE[1] / 2, TABLE_SIZE[2] / 2]
    table.pos = [0.30, 0.06, TABLE_TOP_Z - TABLE_SIZE[2] / 2]
    table.rgba = [0.36, 0.38, 0.42, 1.0]
    _mask(table, 1, 3)

    # ---------- 灯光 ----------
    for nm, pos, dir_, dif, shadow in [
        ("key", [0.05, -0.45, 0.85], [0.25, 0.45, -1.0], [1.0, 1.0, 1.0], True),
        ("fill", [0.65, 0.55, 0.60], [-0.4, -0.4, -0.85], [0.5, 0.5, 0.55], False),
    ]:
        L = wb.add_light()
        L.name = nm
        L.pos = pos
        L.dir = dir_
        L.diffuse = dif
        L.castshadow = shadow

    # ---------- 台钳（真实 SimReady 资产）----------
    vise_obj = os.path.join(OBJ_DIR, "equipment_bench_vise_a01.obj")
    Vv, Fv = load_obj(vise_obj)
    lo, hi = Vv.min(axis=0), Vv.max(axis=0)
    size = hi - lo
    # 居中到原点，底面落到 z=0
    Vv = Vv - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    Vv_small, Fv_small = make_convex_hull_mesh(Vv, Fv, "vise")

    mv = os.path.join(MESH_DIR, "vise_visual.obj")
    write_obj(mv, Vv, Fv, "台钳-视觉")
    mc = os.path.join(MESH_DIR, "vise_collision.obj")
    write_obj(mc, Vv_small, Fv_small, "台钳-碰撞")

    vise_body = wb.add_body()
    vise_body.name = "vise"
    vise_body.pos = [VISE_XY[0], VISE_XY[1], TABLE_TOP_Z]

    gv = vise_body.add_geom()
    gv.name = "vise_visual"
    gv.type = mujoco.mjtGeom.mjGEOM_MESH
    gv.meshname = "vise_visual"
    gv.rgba = [0.42, 0.46, 0.52, 1.0]
    gv.contype = 0
    gv.conaffinity = 0

    gc = vise_body.add_geom()
    gc.name = "vise_collision"
    gc.type = mujoco.mjtGeom.mjGEOM_MESH
    gc.meshname = "vise_collision"
    gc.rgba = [0.42, 0.46, 0.52, 0.0]
    gc.friction = [1.2, 0.02, 0.002]
    _mask(gc, 1, 3)

    print("  台钳: %.3f x %.3f x %.3f m，碰撞面 %d"
          % (size[0], size[1], size[2], len(Fv_small)))

    # ---------- 轴（真实 SimReady 资产，自由落体）----------
    shaft_obj = os.path.join(OBJ_DIR, "drivetrain_axle_shaft_d01.obj")
    Vs, Fs = load_obj(shaft_obj)
    lo2, hi2 = Vs.min(axis=0), Vs.max(axis=0)
    size2 = hi2 - lo2
    # 轴自身居中
    Vs = Vs - (lo2 + hi2) / 2

    ms = os.path.join(MESH_DIR, "shaft_visual.obj")
    write_obj(ms, Vs, Fs, "轴-视觉")

    shaft_body = wb.add_body()
    shaft_body.name = "shaft"
    # 轴躺在台钳上方
    shaft_body.pos = [SHAFT_XY[0], SHAFT_XY[1], TABLE_TOP_Z + 0.10]
    shaft_body.add_freejoint(name="shaft_free")

    sg = shaft_body.add_geom()
    sg.name = "shaft_geom"
    sg.type = mujoco.mjtGeom.mjGEOM_MESH
    sg.meshname = "shaft_visual"
    sg.mass = 0.25
    sg.rgba = [0.72, 0.74, 0.78, 1.0]
    sg.friction = [1.2, 0.02, 0.002]
    _mask(sg, 1, 3)

    print("  轴  : %.3f x %.3f x %.3f m，%d 面"
          % (size2[0], size2[1], size2[2], len(Fs)))

    # ---------- 注册 mesh 资产 ----------
    for nm in ("vise_visual", "vise_collision", "shaft_visual"):
        a = spec.add_mesh()
        a.name = nm
        a.file = nm + ".obj"

    return spec


def make_scene():
    spec = build_scene()
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def render_scene(model, data, path=None):
    """渲染三个视角，肉眼确认资产没问题。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cnfont import setup_cn_font
    setup_cn_font()

    rend = mujoco.Renderer(model, height=640, width=820)
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

    vx, vy = VISE_XY
    VIEWS = [
        ("① 全景：真实 SimReady 台钳 + 轴", cam([0.30, 0.08, 0.08], 0.78, 15, -24)),
        ("② 台钳特写（视觉网格 106k 面）", cam([vx, vy, 0.09], 0.42, 25, -20)),
        ("③ 轴特写（Φ65x621mm 真轴）", cam([vx, vy, 0.06], 0.30, 55, -18)),
    ]

    imgs = []
    for title, c in VIEWS:
        rend.update_scene(data, camera=c)
        imgs.append((title, rend.render().copy()))
    rend.close()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    fig.suptitle("NVIDIA SimReady 真实资产 -> MuJoCo 场景", fontsize=15,
                 fontweight="bold")
    for ax, (title, img) in zip(axes, imgs):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = path or os.path.join(OUT, "scene_simready.png")
    plt.savefig(out, dpi=115, bbox_inches="tight", facecolor="white")
    print("  已保存 %s" % out)
    return out


if __name__ == "__main__":
    print("=" * 74)
    print("用真实 NVIDIA SimReady 资产搭 MuJoCo 场景")
    print("=" * 74)
    print()

    model, data = make_scene()

    print()
    print("模型统计：")
    print("  nq=%d nv=%d nbody=%d ngeom=%d nmesh=%d"
          % (model.nq, model.nv, model.nbody, model.ngeom, model.nmesh))
    for i in range(model.nmesh):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        print("    mesh[%d] %-18s verts=%-6d faces=%d"
              % (i, nm, model.mesh_vertnum[i], model.mesh_facenum[i]))

    print()
    print("跑 1 秒物理，看轴会不会掉到台钳上：")
    for i in range(500):
        mujoco.mj_step(model, data)
        if i % 100 == 0:
            q = data.qpos[:3]
            print("  t=%.2fs  轴位置 [%.4f, %.4f, %.4f]"
                  % (data.time, q[0], q[1], q[2]))
    q = data.qpos[:3]
    print("  t=%.2fs  轴位置 [%.4f, %.4f, %.4f]  (最终)"
          % (data.time, q[0], q[1], q[2]))
    print()
    print("接触点数: %d" % data.ncon)

    print()
    render_scene(model, data)
    print("OK")
