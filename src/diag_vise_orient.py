"""diag_vise_orient.py —— 台钳该转多少度？四个候选一次全渲染出来对比。

为什么这样做
-----------
台钳的局部坐标系是 CAD 导出的：
    盘体：Y ∈ [0, 0.027]，在 XZ 平面是圆的（直径 233mm）
    钳口：Y ∈ [0.017, 0.18]，沿 +Y 伸出
    丝杠：沿 X，Y ∈ [0.07, 0.096]
    -> 背面朝 -Y，钳口朝 +Y，丝杠沿 X

所以要让"钳口朝向机器人（-x 方向）"，需要把局部 +Y 转到世界 -x。
理论上绕 Z 转 +90° 就够。但别再靠推理，直接渲染四个候选对比。

跑法：
    D:\\robot-env\\Scripts\\python.exe diag_vise_orient.py
"""
import os

import numpy as np
import mujoco

from simready_loader import load_asset, place_asset, register_meshes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked
SO101_ASSETS = os.path.join(
    ROOT, "models", "_tmp_so101", "lerobot_env_so101-0.4.0",
    "lerobot_env_so101", "assets")
SO101_XML = os.path.join(SO101_ASSETS, "so101_new_calib.xml")
SO101_MESH_DIR = os.path.join(SO101_ASSETS, "assets")

VISE_POS = [0.30, 0.16, 0.0]

CANDIDATES = [
    ("0° 不转", 0.0, [1, 0, 0, 0]),
    ("绕Z +90°", np.pi / 2, None),
    ("绕Z -90°", -np.pi / 2, None),
    ("绕Z 180°", np.pi, None),
]


def z_quat(angle):
    return [np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]


def build_one(angle, quat):
    spec = mujoco.MjSpec.from_file(SO101_XML)
    spec.meshdir = SO101_MESH_DIR
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200
    wb = spec.worldbody

    fl = wb.add_geom()
    fl.name = "floor"
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [2.0, 2.0, 0.02]
    fl.rgba = [0.27, 0.29, 0.33, 1.0]

    tb = wb.add_geom()
    tb.name = "table"
    tb.type = mujoco.mjtGeom.mjGEOM_BOX
    tb.size = [0.30, 0.25, 0.012]
    tb.pos = [0.28, 0.02, -0.012]
    tb.rgba = [0.37, 0.39, 0.43, 1.0]

    L = wb.add_light()
    L.name = "key"
    L.pos = [0.02, -0.50, 0.90]
    L.dir = [0.25, 0.45, -1.0]
    L.diffuse = [1.0, 1.0, 1.0]
    L.castshadow = True

    L2 = wb.add_light()
    L2.name = "fill"
    L2.pos = [0.70, 0.55, 0.65]
    L2.dir = [-0.4, -0.4, -0.85]
    L2.diffuse = [0.55, 0.55, 0.6]

    vise = load_asset("equipment_bench_vise_a01", verbose=False)
    vb, vn, cn = place_asset(
        spec, wb, "vise", vise, pos=VISE_POS,
        quat=quat, rgba=(0.42, 0.46, 0.52, 1.0),
        out_dir=SO101_MESH_DIR)
    register_meshes(spec, [vn, cn])

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def render_all():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cnfont import setup_cn_font
    setup_cn_font()

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.4))
    fig.suptitle("台钳朝向对比：哪个角度让钳口朝向机器人？",
                 fontsize=15, fontweight="bold")

    for i, (label, angle, quat) in enumerate(CANDIDATES):
        q = quat if quat is not None else z_quat(angle)
        model, data = build_one(angle, q)
        rend = mujoco.Renderer(model, height=560, width=660)
        rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False

        views = [
            (30, -22, "俯视斜前"),
            (100, -12, "侧视（看钳口方向）"),
        ]
        for col, (az, el, vname) in enumerate(views):
            c = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(c)
            c.lookat[:] = [0.30, 0.10, 0.08]
            c.distance = 0.62
            c.azimuth = az
            c.elevation = el
            rend.update_scene(data, camera=c)
            img = rend.render().copy()
            ax = axes[col][i]
            ax.imshow(img)
            ax.set_title("%s / %s" % (label, vname), fontsize=10)
            ax.axis("off")
        rend.close()
        print("  渲染完成: %s" % label)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT, "vise_orient.png")
    plt.savefig(out, dpi=105, bbox_inches="tight", facecolor="white")
    print("已保存 %s" % out)
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("台钳朝向 4 候对比")
    print("=" * 70)
    render_all()
    print("OK")
