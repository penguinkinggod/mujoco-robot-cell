"""exp_pad_friction.py -- 「架不住」是不是因为太光滑？

只回答一个问题：把夹爪指垫的摩擦系数从 0.1 一路拉到 3.0（比橡胶还糙几倍），
纯圆柱抓起来之后的倾角会不会变好。

如果倾角基本不动 -> 不是打滑、不是摩擦不够，是几何（平板对圆柱只有线接触）。
如果倾角随摩擦明显下降 -> 才是光滑问题，加摩擦皮/换材质就能解决。

跑法：
    D:\\robot-env\\Scripts\\python.exe exp_pad_friction.py
"""
import numpy as np
import mujoco

import diag_feed_chuck as D
from exp_plain_cyl import one_run

MUS = (0.1, 0.3, 0.8, 1.5, 3.0)
GRIP = 0.17          # exp_plain_cyl.py 扫出来的稳定夹紧量


def pad_geoms(model):
    out = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and "pad" in name:
            out.append((gid, name))
    return out


def contact_report(model, data):
    """pad 与轴之间的接触点数 + 法向力总和。"""
    n, force = 0, 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        pair = (("pad" in n1 and "shaft" in n2) or
                ("pad" in n2 and "shaft" in n1))
        if not pair:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        n += 1
        force += abs(float(f[0]))
    return n, force


def main():
    model = D.build_model()
    data = mujoco.MjData(model)
    pads = pad_geoms(model)
    shaft = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_cyl")

    print("=" * 78)
    print("纯圆柱 Φ%d 抓取：只变指垫摩擦系数" % (D.F.SHAFT_D * 1000))
    print("=" * 78)
    print("指垫 geom：")
    for gid, name in pads:
        print("  %-16s size=%-22s friction=%s"
              % (name, np.round(model.geom_size[gid], 4),
                 np.round(model.geom_friction[gid], 3)))

    print()
    print("  %-6s %-11s %-11s %-10s %-9s %s"
          % ("mu", "夹紧时倾角", "抬起后倾角", "抬升", "pad-轴", "判定"))
    print("  " + "-" * 74)
    rows = []
    for mu in MUS:
        for gid, _n in pads:
            model.geom_friction[gid, 0] = mu
        model.geom_friction[shaft, 0] = mu
        r = one_run(model, data, GRIP)
        rows.append((mu, r))
        print("  %-6.2f %-11s %-11s %-10s %-9s %s"
              % (mu, "%.1f deg" % r["tilt_close"], "%.1f deg" % r["tilt_lift"],
                 "%+.1f mm" % r["risen"], "%.1f mm" % r["dpad"],
                 r["verdict"]))

    print()
    lift_ok = [(mu, r) for mu, r in rows if r["verdict"] == "GRAB"]
    lo = min(lift_ok, key=lambda t: t[0])       # 最滑（冰面）
    hi = max(lift_ok, key=lambda t: t[0])       # 能抬起来的最糙
    delta = lo[1]["tilt_lift"] - hi[1]["tilt_lift"]
    print("TILT_DROP=%.1f deg  (mu %.1f -> %.1f，摩擦 %.0f 倍，仍能抬起的范围)"
          % (delta, lo[0], hi[0], hi[0] / lo[0]))
    print("BEST_MU_WITH_LIFT=%.2f tilt=%.1f deg"
          % (hi[0], hi[1]["tilt_lift"]))
    dropped = [(mu, r) for mu, r in rows if r["verdict"] != "GRAB"]
    for mu, r in dropped:
        print("MU=%.2f 连抬都抬不起来（抬升 %+.1f mm）" % (mu, r["risen"]))
    # 判定只看「能抬起来」的样本：摩擦从 0.1 拉到 1.5，倾角该降多少才算摩擦主导？
    if delta < 10.0 and hi[1]["tilt_lift"] > D.TILT_LIMIT:
        print("VERDICT=NOT_FRICTION  摩擦 15 倍只换来 %.1f deg 改善，"
              "仍 %.0f deg（阈值 %.0f）-> 不是打滑，是线接触管不住转动"
              % (delta, hi[1]["tilt_lift"], D.TILT_LIMIT))
    elif delta >= 10.0:
        print("VERDICT=FRICTION_DOMINATED  倾角随摩擦明显变化，值得换材质")

    # 抓起来那一刻的接触：几片 pad 在碰、法向力多大
    model.geom_friction[[g for g, _ in pads], 0] = 0.8
    one_run(model, data, GRIP)
    n, force = contact_report(model, data)
    print("CONTACT_AT_LIFT pads=%d normal_sum=%.1f N" % (n, force))
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
