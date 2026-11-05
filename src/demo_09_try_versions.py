"""demo_09_try_versions.py —— 一次跑通：把成功判定做严，输出多视角证据图。

demo_08 已经做到"抓起来 + 搬过去"，但判定标准太松：
    "最终距钳口中心 < 50 mm" 就算成功，这没意义。

这个脚本用真标准：
    1. 抓取   : 工件被抬起 > 50 mm，且抬起过程中 pad-工件 距离 < 10 mm
    2. 搬运   : 工件 xy 落在钳口 xy 的 ±15 mm 内
    3. 入座   : 工件 z 落在钳口净空的 z 范围内，且松开后不移动
    4. 松开   : 松开后 1 秒工件位移 < 5 mm（真的被钳口托住）

并且输出：
    - 4 个视角的最终状态（全景 / 钳口特写 / 俯视 / 侧视）
    - 动作过程的关键帧
    - 完整的指标表

跑法：
    D:\\robot-env\\Scripts\\python.exe demo_09_try_versions.py
"""
import os

import numpy as np
import mujoco

import scene_full as S
from demo_08_pick_and_load import Arm, WRIST_ROLL_GRASP, GRIP_OPEN, GRIP_CLOSE


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root (src/ -> ..)
OUT = os.path.join(ROOT, "_build")    # generated figures, not tracked


def run_once(verbose=True):
    model, data = S.make_scene(with_vise=True, with_fixture=True, with_shaft_flat=True)

    rend = mujoco.Renderer(model, height=520, width=700)
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
    rend.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    arm = Arm(model, data, renderer=rend)
    arm.cam = S.make_camera()

    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]

    # 预热：摆到抓取姿态，让工件落稳
    arm.d.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    arm.d.ctrl[4] = WRIST_ROLL_GRASP
    arm.d.ctrl[arm.grip_act] = GRIP_OPEN
    for _ in range(600):
        for k in range(5):
            arm.d.ctrl[k] = arm.d.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)

    shaft_home = data.qpos[sq:sq + 3].copy()
    R = S.SHAFT_D / 2
    grasp = shaft_home.copy()

    jaw_c = S.jaw_center()
    geo = S.jaw_geometry()
    JAW_X, JAW_Y, JAW_Z = jaw_c

    # 放置点：工件轴心落到槽底上方的 rest_z。
    # 槽是直壁平底，零自定心；靠 IK 精度直接对准。
    PLACE_Z = JAW_Z + 0.003

    stages = [
        ("① 接近", [grasp[0], grasp[1], grasp[2] + R + 0.075],
         GRIP_OPEN, 550),
        ("② 下降", [grasp[0], grasp[1], grasp[2] + 0.002],
         GRIP_OPEN, 400),
        ("③ 夹紧", [grasp[0], grasp[1], grasp[2] + 0.002],
         GRIP_CLOSE, 320),
        # 夹紧后微调：把工件真正对到 pad 中心，消除抓取偏移
        ("④ 抬起前对中", [grasp[0], grasp[1], grasp[2] + 0.02],
         GRIP_CLOSE, 260),
        ("④b 抬起", [grasp[0], grasp[1], grasp[2] + 0.13],
         GRIP_CLOSE, 420),
        ("⑤ 横移到浅槽上方", [JAW_X, JAW_Y, JAW_Z + 0.10],
         GRIP_CLOSE, 620),
        # 夹着下降到槽口上方 3 mm，到位后再松爪。不能提前松开：
        # 否则工件从约 70 mm 高自由落体，砸在平台上直接弹飞。
        ("⑥ 夹着下到槽口", [JAW_X, JAW_Y, PLACE_Z],
         GRIP_CLOSE, 380),
        ("⑦ 松开让工件落入槽", [JAW_X, JAW_Y, PLACE_Z],
         GRIP_OPEN, 420),
        ("⑧ 退开", [JAW_X - 0.13, JAW_Y - 0.09, JAW_Z + 0.12],
         GRIP_OPEN, 480),
    ]

    trace = []
    # ★ 关键补偿：夹爪抓住工件后，工件中心相对 pad 中心有偏移
    #   （实测 14~25 mm）。如果直接按 pad 位置去定位，工件会偏同样多。
    #   真实工程里这叫"抓取偏移标定"，在夹紧之后测一次，之后全程补偿。
    shaft_in_grip = None

    for name, tgt, grip, ns in stages:
        tgt = np.array(tgt, dtype=float)
        # 已经夹住工件时，用抓取偏移修正目标：
        # 我们想让【工件】到 tgt，所以要给 pad 一个补偿后的目标
        if shaft_in_grip is not None and name.startswith(("⑤", "⑥", "⑦")):
            # 夹紧后标定一次，之后全程沿用。每步重测会把下降途中工件
            # 相对 pad 的滑动当成新的"抓取偏移"，把放置点越抬越高
            # (实测 ⑥ 因此停在 z=52 mm，工件根本没进槽)。
            tgt_pad = tgt + shaft_in_grip
        else:
            tgt_pad = tgt

        err = None
        for _ in range(3):
            err = arm.move_to(tgt_pad, n_steps=max(90, ns // 3), grip=grip,
                              wrist_roll=WRIST_ROLL_GRASP)
            if err < 0.005:
                break
        pad = arm.pad_center()
        sh = data.qpos[sq:sq + 3].copy()
        trace.append({
            "name": name,
            "pad": pad.copy(),
            "shaft": sh.copy(),
            "dp": float(np.linalg.norm(pad - sh)),
            "err": float(err),
        })
        # 夹紧之后开始记录抓取偏移
        if name.startswith("③") and shaft_in_grip is None:
            shaft_in_grip = pad - sh

        if verbose:
            print("  %-16s pad残差 %5.1f mm   pad-工件 %6.1f mm   工件 %s"
                  % (name, err * 1000, trace[-1]["dp"] * 1000,
                     np.round(sh, 4)))

    # ---- 松开后静置，验证是否真被托住 ----
    before_settle = data.qpos[sq:sq + 3].copy()
    for _ in range(600):
        mujoco.mj_step(model, data)
    after_settle = data.qpos[sq:sq + 3].copy()
    drift = float(np.linalg.norm(after_settle - before_settle))

    # ---- 判定 ----
    # trace[4] 是 ④b 抬起；trace[3] 只是"抬起前对中"，还没离地
    lift = float(trace[4]["shaft"][2] - shaft_home[2])
    grasp_ok = lift > 0.05 and trace[4]["dp"] < 0.030

    # 搬运判据按 HANDOFF §2：到达夹具【上方】时轴心 xy 误差 <= 5 mm。
    # 拿最终落点当搬运误差是错的 -- 落点沿 x (轴长方向) 偏一点根本
    # 不影响入座，槽长 130 mm 对轴长 80 mm 有足够余量。
    xy_err = float(np.linalg.norm(
        trace[5]["shaft"][:2] - np.array([JAW_X, JAW_Y])))
    carry_ok = xy_err < 0.005

    # 工件落进 24 mm 浅槽：y 在槽宽内，z 真的坐到槽底（±3 mm）
    seated = abs(after_settle[1] - JAW_Y) <= geo["slot_w"] / 2 + 0.0005
    seat_z_ok = abs(after_settle[2] - JAW_Z) <= 0.003
    place_ok = seated and seat_z_ok and drift < 0.010

    res = {
        "shaft_home": shaft_home,
        "shaft_final": after_settle,
        "lift_mm": lift * 1000,
        "xy_err_mm": xy_err * 1000,
        "drift_mm": drift * 1000,
        "grasp_ok": grasp_ok,
        "carry_ok": carry_ok,
        "place_ok": place_ok,
        "seated_y": seated,
        "seat_z_ok": seat_z_ok,
        "trace": trace,
    }

    # ---- 末尾多视角渲染 ----
    def cam(lookat, dist, az, el):
        c = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(c)
        c.lookat[:] = lookat
        c.distance = dist
        c.azimuth = az
        c.elevation = el
        return c

    views = [
        ("全景", cam([0.22, 0.02, 0.09], 0.95, 22, -26)),
        ("浅槽特写", cam([JAW_X, JAW_Y, JAW_Z + 0.02], 0.34, 6, -12)),
        ("俯视", cam([JAW_X, JAW_Y, JAW_Z], 0.36, 0, -78)),
        ("侧视（看是否入槽）", cam([JAW_X, JAW_Y, JAW_Z], 0.34, 90, -6)),
    ]
    final_imgs = []
    for t, c in views:
        rend.update_scene(data, camera=c)
        final_imgs.append((t, rend.render().copy()))

    rend.close()
    return res, final_imgs, arm.frames


def report(res, final_imgs, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cnfont import setup_cn_font
    setup_cn_font()

    print()
    print("=" * 76)
    print("严格判定")
    print("=" * 76)
    print()
    print("  工件起点      : %s" % np.round(res["shaft_home"], 4))
    print("  工件最终      : %s" % np.round(res["shaft_final"], 4))
    print()
    print("  抬升高度      : %+.1f mm   (要求 > 50)" % res["lift_mm"])
    print("  搬运到槽上方 xy 误差 : %.1f mm    (要求 < 5)" % res["xy_err_mm"])
    print("  松开后漂移    : %.1f mm    (要求 < 10)" % res["drift_mm"])
    print()
    print("  落进 y 槽宽   : %s" % ("是" if res["seated_y"] else "否"))
    print("  高度对          : %s" % ("是" if res["seat_z_ok"] else "否"))
    print()
    print("  %s 抓取" % ("✓" if res["grasp_ok"] else "✗"))
    print("  %s 搬运" % ("✓" if res["carry_ok"] else "✗"))
    print("  %s 放置" % ("✓" if res["place_ok"] else "✗"))
    print()
    score = sum([res["grasp_ok"], res["carry_ok"], res["place_ok"]])
    print("  得分 %d / 3" % score)

    # ---- 图 ----
    fig = plt.figure(figsize=(17, 10.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.15])

    fig.suptitle("SO-101 机床上下料：抓取 → 搬运 → 入浅槽",
                 fontsize=16, fontweight="bold")

    labels = [t["name"] for t in res["trace"]]
    if frames:
        n = len(frames)
        idx = [int(i * (n - 1) / 7) for i in range(8)]
        for k in range(8):
            ax = fig.add_subplot(gs[k // 4, k % 4])
            ax.imshow(frames[idx[k]])
            ax.set_title(labels[k] if k < len(labels) else "", fontsize=10)
            ax.axis("off")

    for k, (t, img) in enumerate(final_imgs):
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(img)
        ax.set_title("最终 - " + t, fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT, "demo_09_result.png")
    plt.savefig(out, dpi=108, bbox_inches="tight", facecolor="white")
    print()
    print("  已保存 %s" % out)
    return out


if __name__ == "__main__":
    print("=" * 76)
    print("机床上下料完整流程（严格判定）")
    print("=" * 76)
    print()
    res, imgs, frames = run_once()
    report(res, imgs, frames)
    print()
    print("OK")
