"""simready_loader.py —— 把 SimReady 的 OBJ 资产装进 MuJoCo 的可复用工具。

设计要点（都是踩过坑之后定的）
-----------------------------
1. 视觉/碰撞分离
   106k 面的视觉网格直接当碰撞体会让物理掉到个位数帧率。
   视觉用完整网格（contype=0），碰撞用凸包（1362 面，78 倍加速）。

2. 凸包会填平凹槽
   台钳的钳口、料箱的内腔都会被填上。所以"能夹住东西"的部件事后
   必须补独立的 box/cylinder 碰撞体，不能指望凸包。

3. 坐标归一
   SimReady 资产的局部原点千奇百怪（有的在几何中心，有的在一角）。
   统一做法：读进来后按包围盒中心归零、底面落到 z=0，这样摆位时
   只需要给 (x, y) 就能"放在台面上"。

用法：
    from simready_loader import load_asset, place_asset
    mesh = load_asset("equipment_bench_vise_a01")
    place_asset(spec, wb, "vise", mesh, pos=[0.3, 0.1, 0.0])
"""
import os

import numpy as np
import trimesh
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
OBJ_DIR = os.path.join(ROOT, "models", "simready", "obj")
MESH_OUT = os.path.join(ROOT, "models", "simready", "mjcf_meshes")


# ============================================================
# OBJ 读写（不用 trimesh 读，因为 SimReady 的 OBJ 里有大量 group，
# trimesh 的 force='mesh' 会丢 group 信息；我们自己解析更可控）
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
        f.write("# 自动生成：%s  verts=%d faces=%d\n" % (name, len(V), len(F)))
        for v in V:
            f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        for t in F:
            f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


# ============================================================
# 资产加载 + 归一化
# ============================================================
class AssetMesh:
    """一个 SimReady 资产的几何，已经归一化过。"""

    def __init__(self, name, V, F, V_hull, F_hull, size, phys):
        self.name = name
        self.V = V
        self.F = F
        self.V_hull = V_hull
        self.F_hull = F_hull
        self.size = size          # 归一化后的尺寸（米）
        self.phys = phys          # {子部件名: 质量}

    def __repr__(self):
        return ("AssetMesh(%s, %.3f x %.3f x %.3f m, %d 面)"
                % (self.name, *self.size, len(self.F)))


def load_asset(name, center_xy=True, bottom_at_zero=True, verbose=True):
    """读一个 SimReady 资产，归一化后返回 AssetMesh。

    center_xy      : 把几何中心移到 (0,0)
    bottom_at_zero : 把底面移到 z=0
    """
    import json

    obj = os.path.join(OBJ_DIR, name + ".obj")
    if not os.path.exists(obj):
        raise FileNotFoundError("资产不存在：%s\n  先跑 tools/glb2obj 转换" % obj)

    V, F = load_obj(obj)
    lo, hi = V.min(axis=0), V.max(axis=0)
    size = hi - lo

    shift = np.zeros(3)
    if center_xy:
        shift[0] = (lo[0] + hi[0]) / 2
        shift[1] = (lo[1] + hi[1]) / 2
    if bottom_at_zero:
        shift[2] = lo[2]
    V = V - shift

    # 凸包（碰撞用）
    try:
        mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
        hull = mesh.convex_hull
        V_hull = np.asarray(hull.vertices)
        F_hull = np.asarray(hull.faces)
    except Exception as e:
        if verbose:
            print("    凸包失败（%s），碰撞体用原网格" % type(e).__name__)
        V_hull, F_hull = V, F

    # 物理数据
    phys = {}
    pj = os.path.join(OBJ_DIR, name + ".phys.json")
    if os.path.exists(pj):
        with open(pj, "r", encoding="utf-8") as f:
            d = json.load(f)
        for node in d["nodes"]:
            phys[node["name"]] = {
                "mass": node.get("mass"),
                "density": node.get("density"),
            }

    if verbose:
        print("    %s : %.3f x %.3f x %.3f m, 视觉 %d 面 -> 碰撞 %d 面"
              % (name, *size, len(F), len(F_hull)))

    return AssetMesh(name, V, F, V_hull, F_hull, size, phys)


def total_mass(asset):
    """资产的总质量（来自 SimReady 物理数据）。"""
    return sum(v["mass"] or 0.0 for v in asset.phys.values())


# ============================================================
# 放进 MuJoCo 场景
# ============================================================
def place_asset(spec, parent_body, tag, asset, pos=(0, 0, 0),
                quat=None, rgba=(0.45, 0.48, 0.54, 1.0),
                mass=None, free=False, friction=(1.2, 0.02, 0.002),
                ct=1, ca=3, out_dir=None):
    """把一个资产作为 body 放进场景，视觉 + 凸包碰撞各一个 geom。

    out_dir: OBJ 写到哪个目录。MuJoCo 只有一个全局 meshdir，
             所以必须和模型里其他 mesh 放同一个目录。
    """
    d = out_dir or MESH_OUT
    os.makedirs(d, exist_ok=True)
    vis_name = tag + "_vis"
    col_name = tag + "_col"

    write_obj(os.path.join(d, vis_name + ".obj"),
              asset.V, asset.F, "%s 视觉" % tag)
    write_obj(os.path.join(d, col_name + ".obj"),
              asset.V_hull, asset.F_hull, "%s 碰撞" % tag)

    b = parent_body.add_body()
    b.name = tag
    b.pos = list(pos)
    if quat is not None:
        b.quat = list(quat)
    if free:
        b.add_freejoint(name=tag + "_free")

    gv = b.add_geom()
    gv.name = vis_name
    gv.type = mujoco.mjtGeom.mjGEOM_MESH
    gv.meshname = vis_name
    gv.rgba = list(rgba)
    gv.contype = 0
    gv.conaffinity = 0

    gc = b.add_geom()
    gc.name = col_name
    gc.type = mujoco.mjtGeom.mjGEOM_MESH
    gc.meshname = col_name
    gc.rgba = [rgba[0], rgba[1], rgba[2], 0.0]
    gc.friction = list(friction)
    gc.contype = ct
    gc.conaffinity = ca

    if mass is not None:
        gc.mass = mass

    return b, vis_name, col_name


def register_meshes(spec, names):
    for nm in names:
        a = spec.add_mesh()
        a.name = nm
        a.file = nm + ".obj"
