"""render_simready.py —— 渲染 SimReady 转换后的 OBJ，确认转换忠实。

目的：肉眼确认几何没坏、朝向对、比例对。
不确认就往下做，后面全是白费。

跑法：
    D:\\robot-env\\Scripts\\python.exe render_simready.py
"""
import os

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
OBJ_DIR = os.path.join(ROOT, "models", "simready", "obj")

ASSETS = [
    ("drivetrain_axle_shaft_a01", "传动轴（真实 SimReady 资产）"),
    ("equipment_machine_vise_c01", "机床台钳（真实 SimReady 资产）"),
    ("equipment_bench_vise_heavy_b01", "重型台钳（真实 SimReady 资产）"),
]

fig = plt.figure(figsize=(16, 11))
fig.suptitle("NVIDIA SimReady -> OBJ 转换验证", fontsize=17, fontweight="bold")

for row, (name, title) in enumerate(ASSETS):
    path = os.path.join(OBJ_DIR, name + ".obj")
    if not os.path.exists(path):
        continue

    mesh = trimesh.load(path, force="mesh", process=False)
    V = np.asarray(mesh.vertices)
    lo, hi = V.min(axis=0), V.max(axis=0)
    size = hi - lo

    # 两个视角：等轴测 + 侧视
    views = [
        (25, 35, "等轴测"),
        (0, 0, "侧视 XZ"),
    ]
    for col, (elev, azim, vname) in enumerate(views):
        ax = fig.add_subplot(3, 2, row * 2 + col + 1, projection="3d")
        step = max(1, len(mesh.faces) // 60000)
        tris = mesh.faces[::step]
        poly = V[tris]
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        pc = Poly3DCollection(poly, alpha=0.92)
        pc.set_facecolor("#b8c0cc")
        pc.set_edgecolor("none")
        ax.add_collection3d(pc)

        # 等比例显示，否则会拉变形
        c = (lo + hi) / 2
        r = size.max() / 2 * 1.05
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.tick_params(labelsize=7)

        if col == 0:
            ax.set_title(
                "%s\n%.3f x %.3f x %.3f m（%d 面）"
                % (title, size[0], size[1], size[2], len(mesh.faces)),
                fontsize=11, fontweight="bold")
        else:
            ax.set_title(vname, fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(OUT, "simready_check.png")
plt.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
print("已保存 %s" % out)

# 数值摘要
print()
print("=" * 70)
print("几何摘要")
print("=" * 70)
for name, title in ASSETS:
    path = os.path.join(OBJ_DIR, name + ".obj")
    if not os.path.exists(path):
        continue
    mesh = trimesh.load(path, force="mesh", process=False)
    V = np.asarray(mesh.vertices)
    lo, hi = V.min(axis=0), V.max(axis=0)
    print("  %-32s  尺寸 %.3f x %.3f x %.3f m"
          % (name, *(hi - lo)))
    print("  %-32s  中心 [%.3f, %.3f, %.3f]"
          % ("", *((lo + hi) / 2)))
print()
print("OK")
