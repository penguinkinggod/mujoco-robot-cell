"""diag_fixture_profile.py -- plot the fixture cross-section to eyeball it.

Rendering the whole scene hides geometry bugs. This plots the y-z
cross-section of every fixture geom plus the workpiece circle, so a
flipped wedge is obvious at a glance.

Run:
    D:\\robot-env\\Scripts\\python.exe diag_fixture_profile.py
"""
import numpy as np
import mujoco

import scene_full as S


def main():
    model = S.build_scene().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from cnfont import setup_cn_font
    setup_cn_font()

    fig, ax = plt.subplots(figsize=(9, 7))
    jaw = S.jaw_center()
    geo = S.jaw_geometry()

    print("=" * 70)
    print("FIXTURE CROSS-SECTION (y-z plane at x = jaw center)")
    print("=" * 70)
    print()

    colors = {"slot_wall_L": "tab:blue", "slot_wall_R": "tab:orange",
              "slot_base": "tab:gray"}

    for gi in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if nm not in colors:
            if "vise_sr" in nm:
                continue
            continue

        gtype = model.geom_type[gi]
        pos = data.geom_xpos[gi]
        mat = data.geom_xmat[gi].reshape(3, 3)

        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mi = model.geom_dataid[gi]
            va, vn = model.mesh_vertadr[mi], model.mesh_vertnum[mi]
            V = model.mesh_vert[va:va + vn].copy()
            # mesh 顶点是局部坐标，转到世界
            Vw = V @ mat.T + pos
            print("  %-10s mesh  y [%.4f, %.4f]  z [%.4f, %.4f]"
                  % (nm, Vw[:, 1].min(), Vw[:, 1].max(),
                     Vw[:, 2].min(), Vw[:, 2].max()))
            # 画 y-z 投影
            ax.fill(Vw[:, 1], Vw[:, 2], alpha=0.35,
                    color=colors[nm], edgecolor="k", lw=1.2)
            # 画三角形轮廓点
            ax.plot(Vw[:, 1], Vw[:, 2], ".", ms=3, color=colors[nm])

        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            sx, sy, sz = model.geom_size[gi]
            corners = np.array([
                [-sx, -sy, -sz], [-sx, sy, -sz], [-sx, sy, sz], [-sx, -sy, sz],
                [sx, -sy, -sz], [sx, sy, -sz], [sx, sy, sz], [sx, -sy, sz],
            ])
            Cw = corners @ mat.T + pos
            print("  %-10s box   y [%.4f, %.4f]  z [%.4f, %.4f]"
                  % (nm, Cw[:, 1].min(), Cw[:, 1].max(),
                     Cw[:, 2].min(), Cw[:, 2].max()))
            # 只画 y-z 投影的外框
            ys = [Cw[:, 1].min(), Cw[:, 1].max(),
                  Cw[:, 1].max(), Cw[:, 1].min()]
            zs = [Cw[:, 2].min(), Cw[:, 2].min(),
                  Cw[:, 2].max(), Cw[:, 2].max()]
            ax.add_patch(Polygon(list(zip(ys, zs)), closed=True,
                                 alpha=0.3, color=colors[nm],
                                 edgecolor="k", lw=1.2))

    # 工件圆（在静止高度）
    R = S.SHAFT_D / 2
    ax.add_patch(Circle((jaw[1], jaw[2]), R, fill=False,
                        edgecolor="crimson", lw=2.5))
    ax.plot([jaw[1]], [jaw[2]], "r+", ms=12)

    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.axhline(S.TABLE_TOP_Z, color="k", ls=":", lw=1)
    ax.axvline(jaw[1], color="r", ls=":", lw=0.8)
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Fixture cross-section (y-z) with workpiece at rest height\n"
                 "slot %.0f mm wide x %.0f mm deep, rest z = %.1f mm"
                 % (geo["slot_w"] * 1000, geo["slot_d"] * 1000,
                    geo["rest_z"] * 1000),
                 fontsize=12)
    ax.set_xlim(jaw[1] - 0.10, jaw[1] + 0.10)
    ax.set_ylim(-0.03, 0.10)

    plt.tight_layout()
    out = "fixture_profile.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    print()
    print("saved %s" % out)
    print()
    print("预期：灰色底板 + 两侧直壁围成平底槽，红色圆（工件）底部")
    print("      正好落在槽底，壁顶与轴心齐平。直壁不做自定心，")
    print("      只负责挡住圆柱滚动；看到斜面或托辊就是旧版残留。")
    print()
    print("OK")


if __name__ == "__main__":
    main()
