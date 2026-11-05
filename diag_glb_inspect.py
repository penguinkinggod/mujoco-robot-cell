"""diag_glb_inspect.py —— 不装任何库，直接解析 GLB 二进制，看里面有什么。

为什么手写解析
--------------
要知道清 GLB 里到底有什么，才能判断能不能转成 MJCF。
装 trimesh / pygltflib 之前先看原始结构，不走弯路。

GLB 格式（公开规范）：
    [0:4]   magic  "glTF"
    [4:8]   version (uint32) = 2
    [8:12]  total length (uint32)
    [12:...] chunk 0: JSON  (length uint32, type "JSON", data)
    [...:..] chunk 1: BIN   (length uint32, type "BIN\\0", data)

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_glb_inspect.py
"""
import json
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
GLB_DIR = os.path.join(HERE, "models", "simready")

FILES = [
    "equipment_machine_vise_c01.glb",
    "drivetrain_axle_shaft_a01.glb",
    "equipment_bench_vise_heavy_b01.glb",
]


def read_glb(path):
    with open(path, "rb") as f:
        raw = f.read()
    magic, version, total = struct.unpack("<4sII", raw[:12])
    assert magic == b"glTF", "不是 GLB 文件"
    chunks = {}
    off = 12
    while off < total:
        clen, ctype = struct.unpack("<I4s", raw[off:off + 8])
        data = raw[off + 8:off + 8 + clen]
        chunks[ctype.rstrip(b"\x00").decode()] = data
        off += 8 + clen
    gltf = json.loads(chunks["JSON"].decode("utf-8"))
    return gltf, chunks.get("BIN", b"")


for fname in FILES:
    path = os.path.join(GLB_DIR, fname)
    if not os.path.exists(path):
        print("跳过（不存在）：%s" % fname)
        continue

    gltf, binchunk = read_glb(path)
    size_mb = os.path.getsize(path) / 1024 / 1024

    print("=" * 74)
    print("%s  (%.2f MB)" % (fname, size_mb))
    print("=" * 74)

    print("  asset      : %s" % gltf.get("asset", {}).get("generator", "?"))
    print("  extensions : %s" % gltf.get("extensionsUsed", []))
    print()

    for key in ("scenes", "nodes", "meshes", "materials", "textures",
                "accessors", "bufferViews", "animations"):
        print("  %-12s : %d" % (key, len(gltf.get(key, []))))

    # ---- 物理扩展 ----
    ext = gltf.get("extensions", {})
    print()
    print("  顶层 extensions: %s" % list(ext.keys()))

    phys = ext.get("KHR_physics_rigid_bodies", {})
    if phys:
        bodies = phys.get("rigidBodies", [])
        shapes = phys.get("colliders", phys.get("shapes", []))
        joints = phys.get("joints", [])
        print("  物理刚体     : %d 个" % len(bodies))
        print("  碰撞体       : %d 个" % len(shapes))
        print("  关节         : %d 个" % len(joints))
        for i, b in enumerate(bodies[:6]):
            m = b.get("mass")
            print("    body[%d] mass=%.4f  name=%s"
                  % (i, m if m is not None else -1, b.get("name", "")))
        for i, j in enumerate(joints[:6]):
            print("    joint[%d] %s" % (i, json.dumps(j)[:150]))

    # ---- 节点层级的 extras（碰撞凸包常放这里）----
    print()
    print("  节点信息（前 8 个）：")
    for i, n in enumerate(gltf.get("nodes", [])[:8]):
        extras = n.get("extras", {})
        keys = list(extras.keys())
        print("    node[%d] name=%-28s mesh=%s extras=%s"
              % (i, n.get("name", "")[:28], n.get("mesh"), keys))
        if "collisionHulls" in extras:
            ch = extras["collisionHulls"]
            print("         collisionHulls: %d 个, 类型=%s"
                  % (len(ch) if isinstance(ch, list) else 1, type(ch).__name__))

    # ---- 找材质里的物理参数 ----
    print()
    print("  材质物理属性（前 5 个）：")
    for i, mat in enumerate(gltf.get("materials", [])[:5]):
        pbr = mat.get("pbrMetallicRoughness", {})
        mext = mat.get("extensions", {})
        print("    mat[%d] %-24s metallic=%.2f rough=%.2f ext=%s"
              % (i, mat.get("name", "")[:24],
                 pbr.get("metallicFactor", 1.0),
                 pbr.get("roughnessFactor", 1.0),
                 list(mext.keys())))
        for k in ("KHR_materials_physics", "PHYSICS_material"):
            if k in mext:
                print("         %s: %s" % (k, json.dumps(mext[k])[:200]))

    # ---- 网格顶点数 ----
    total_tris = 0
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            idx = prim.get("indices")
            if idx is not None:
                acc = gltf["accessors"][idx]
                total_tris += acc.get("count", 0) // 3
    print()
    print("  总三角面   : %d" % total_tris)
    print()

print("=" * 74)
print("判断")
print("=" * 74)
print("""
  要转成 MJCF，需要三样东西：
      1. 三角网格（mesh）      -> 在 glTF 的 meshes/accessors 里
      2. 质量/惯量             -> 在 KHR_physics_rigid_bodies 里
      3. 碰撞体                -> 在 extras.collisionHulls 里

  如果三样都在，转换是纯数据搬运，不需要重建几何。
""")
print("OK")
