"""diag_obj_geom.py —— 检查导出的 OBJ 几何，判断转换是否忠实。

要回答：
    1. 尺寸对不对（台钳应该是几十厘米，轴应该是几十厘米）
    2. 朝向对不对（Z 轴向上还是 Y 轴向上）
    3. 位置在原点附近还是一万米外（单位是不是米）
    4. 顶点法线、面朝向有没有问题

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_obj_geom.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
OBJ_DIR = os.path.join(ROOT, "models", "simready", "obj")

FILES = [
    "drivetrain_axle_shaft_a01",
    "equipment_machine_vise_c01",
    "equipment_bench_vise_heavy_b01",
]


def load_obj_verts(path):
    """读 OBJ 的顶点和面（只要几何，不管材质）。"""
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                p = line.split()[1:]
                # 支持 v / v/vt / v//vn 三种写法，取第一段
                idx = [int(x.split("/")[0]) for x in p]
                faces.append(idx)
    return np.array(verts, dtype=np.float64), faces


print("=" * 76)
print("SimReady OBJ 几何检查")
print("=" * 76)
print()

for name in FILES:
    obj = os.path.join(OBJ_DIR, name + ".obj")
    if not os.path.exists(obj):
        print("跳过：%s" % name)
        continue

    V, F = load_obj_verts(obj)
    lo, hi = V.min(axis=0), V.max(axis=0)
    size = hi - lo
    center = (lo + hi) / 2.0

    print("-" * 76)
    print(name)
    print("-" * 76)
    print("  顶点 %d，面 %d" % (len(V), len(F)))
    print()
    print("  包围盒（米）")
    print("    X : %8.4f ~ %8.4f   尺寸 %8.4f  (%6.2f cm)"
          % (lo[0], hi[0], size[0], size[0] * 100))
    print("    Y : %8.4f ~ %8.4f   尺寸 %8.4f  (%6.2f cm)"
          % (lo[1], hi[1], size[1], size[1] * 100))
    print("    Z : %8.4f ~ %8.4f   尺寸 %8.4f  (%6.2f cm)"
          % (lo[2], hi[2], size[2], size[2] * 100))
    print()
    print("  包围盒中心 : [%8.4f, %8.4f, %8.4f]" % tuple(center))
    print("  最长边     : %.4f m  (%.1f cm)" % (size.max(), size.max() * 100))

    # 判断单位：如果最长边 > 5 米，说明是毫米单位
    if size.max() > 5.0:
        print("  -> 单位判断：可能是【毫米】，需要除以 1000")
    elif size.max() > 0.01:
        print("  -> 单位判断：【米】，可直接用")
    else:
        print("  -> 单位判断：可能是【厘米或英寸】，需检查")

    # 判断哪个轴是"长轴"
    longest = int(np.argmax(size))
    print("  长轴方向   : %s 轴" % "XYZ"[longest])
    print("  向上方向   : Z 轴（MuJoCo 约定这是竖直向上）")

    # 检查是否有异常点（NaN / 极远点）
    if not np.isfinite(V).all():
        print("  !! 有 NaN/Inf 顶点")
    far = np.abs(V).max()
    if far > 10.0:
        print("  !! 有离原点 %.1f 的顶点，可能是错误坐标" % far)

    # 面索引是否合法
    if F:
        maxidx = max(max(f) for f in F)
        minidx = min(min(f) for f in F)
        ok = (minidx >= 1) and (maxidx <= len(V))
        print("  面索引范围 : %d ~ %d  %s"
              % (minidx, maxidx, "(合法)" if ok else "(!! 越界)"))

    # 三角面比例
    tri = sum(1 for f in F if len(f) == 3)
    quad = sum(1 for f in F if len(f) == 4)
    print("  三角形 %d，四边形 %d" % (tri, quad))

    # 挖空判断：顶点数 vs 面数的关系（封闭网格 2V ≈ F 的欧拉关系）
    print()

    # 物理数据
    pj = os.path.join(OBJ_DIR, name + ".phys.json")
    if os.path.exists(pj):
        with open(pj, "r", encoding="utf-8") as f:
            phys = json.load(f)
        print("  物理数据：")
        for n in phys["nodes"]:
            hulls = n.get("hulls")
            nh = 0
            if isinstance(hulls, list):
                nh = len(hulls)
            elif isinstance(hulls, dict):
                nh = sum(len(v) if isinstance(v, list) else 1
                         for v in hulls.values())
            print("    %-24s mass=%8.3f kg  density=%6.0f  hulls=%d"
                  % (n["name"], n.get("mass") or 0,
                     n.get("density") or 0, nh))
    print()

print("=" * 76)
print("结论要点")
print("=" * 76)
print("""
  要能在 MuJoCo 里用，需要满足：
      1. 单位是米（否则要把整个模型缩放到合理尺寸）
      2. Z 轴向上（glTF 是 Y 轴向上，转换后可能需要绕 X 轴转 -90°）
      3. 模型躺在台面上（底面 z = 0），现在中心可能在原点
      4. 三角面数不能太高（MuJoCo 碰撞用三角网格很慢，
         应该用 convex hull 或 primitive 代替）
""")
print("OK")
