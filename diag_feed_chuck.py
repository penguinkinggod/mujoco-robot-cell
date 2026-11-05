"""diag_feed_chuck.py -- can the ROBOT feed the shaft into the chuck?

KNOWN FAIL as of 2026-11-05. This script is an honest, reproducible
baseline, NOT a passing check. It is deliberately NOT wired into
verify_all.py's CHECKS: doing so would turn the 6/6 green baseline red and
hide regressions. Wire it in once the grip-hold defect below is fixed.

What it does
------------
Builds the MERGED scene (scene_full.py with the bolt-on lathe), then runs
the same 5-stage pick sequence demo_09 proved, carries the shaft to the
chuck axis, clamps the three jaws, releases and retreats the robot, and
measures gravity slip plus a 20 N pull-out. Every stage prints the shaft
state so the exact failure point is visible.

Measured root cause (7 controlled experiment rounds)
---------------------------------------------------
The shaft starts rotating the moment the pads touch it, and has turned
~98 degrees BY the end of the lift stage -- before any carry or chuck
interaction:

    after descend    23.6 deg
    after close      43.8 deg
    after centring   57.7 deg
    after lift       98.2 deg   (required <= 10 deg)
    pad - shaft      11 mm at lift, 19 mm after traverse

The real collision shape is a SMOOTH CYLINDER (the D-flat is visual-only,
contype=0), and at WRIST_ROLL_GRASP=1.55 the line between the two pads has
a 0.068 z component, so they are not level. Contact asymmetry rolls the
cylinder from the first touch instead of holding its orientation. Once
lifted it also hangs ~11 mm below the pad centre on friction alone, which
adds a pendulum swing. Everything downstream (coaxiality, jaw contacts,
pull-out) is then measuring a rotated stick, not an axial shaft.

demo_09's 3/3 never caught this: its grasp_ok / carry_ok / place_ok only
check lift height, xy error and whether the part settles in the slot. A
vertical shaft dropped into a 24 mm slot still "passes". Feeding a chuck
requires a specific orientation, which is why this hidden defect surfaces
here.

Things already RULED OUT by controlled scans (do not re-try blindly)
--------------------------------------------------------------------
- higher pad friction [2.5, 0.1, 0.005] -> shaft sticks and is flung,
  pad-shaft distance at lift explodes from 11 mm to 255 mm
- tighter GRIP_CLOSE 0.12 -> still flips, pull-out fails
- static closed-loop pose correction -> move_to's 260 "pin" steps make
  incremental targets kick the shaft; coaxiality worsens 2.18 -> 19 mm
- moving the chuck x 0.20/0.22/0.24/0.26 and y -0.02/+0.12,
  lift height 0.07/0.13 -> lift tilt stays 93-98 deg in every case
- clamping order (pre-clamp -> release -> final clamp) -> the shaft is
  already vertical by then; it drops at the release stage

The fix is therefore in the GRASP, not in the feed path: the pads must
hold the shaft at its centre line (or cradle it from below) so it cannot
rotate. Candidates: make the D-flat a real collision geom so the cylinder
cannot roll, level the two pads, change wrist_roll so the pads cradle from
below, or keep a support under the shaft until the chuck has gripped it.

Pass/fail thresholds (written down BEFORE any of the scans)
-----------------------------------------------------------
    coaxiality      <= 2.0 mm   shaft axis vs chuck axis
    shaft tilt      <= 10.0 deg vs the original axial orientation
    jaw contacts    >= 6
    gravity slip    <= 2.0 mm
    20 N pull-out   <= 2.0 mm

Run:
    D:\\robot-env\\Scripts\\python.exe diag_feed_chuck.py
"""
import numpy as np
import mujoco

import scene_full as F
import scene_machining as SM
from demo_08_pick_and_load import (
    Arm, WRIST_ROLL_GRASP, GRIP_OPEN, GRIP_CLOSE,
)

COAX_LIMIT = 2.0       # mm
TILT_LIMIT = 10.0      # deg
CONTACT_MIN = 6
SLIP_LIMIT = 2.0       # mm
PULL_LIMIT = 2.0       # mm

# Closest-to-success configuration from the position scan. x=0.20 keeps
# the carry short (the shaft slides ~2 mm in the pads instead of 18 mm).
LATHE_ORIGIN = (0.20, 0.12, 0.060)
SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])


def tilt_deg(quat):
    return float(np.degrees(
        2 * np.arccos(min(1.0, abs(float(np.dot(SHAFT_Q0, quat)))))))


def main():
    model, data = F.make_scene(with_lathe=True,
                               lathe_origin=LATHE_ORIGIN)
    jaw_act = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
    shaft_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    shaft_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    axis = np.array([LATHE_ORIGIN[0] - 0.016,
                     LATHE_ORIGIN[1], LATHE_ORIGIN[2]])

    def shaft():
        return data.qpos[sq:sq + 3].copy()

    def coax():
        p = shaft()
        return float(np.hypot(p[1] - axis[1], p[2] - axis[2]) * 1000)

    def jaw_contacts():
        n = 0
        for i in range(data.ncon):
            c = data.contact[i]
            if shaft_geom not in (c.geom1, c.geom2):
                continue
            g1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if "jaw_geom" in (g1 + "|" + g2):
                n += 1
        return n

    arm = Arm(model, data)
    mujoco.mj_resetData(model, data)
    data.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    data.ctrl[4] = WRIST_ROLL_GRASP
    data.ctrl[arm.grip_act] = GRIP_OPEN
    data.ctrl[jaw_act] = SM.CHUCK_OPEN_CTRL
    for _ in range(600):
        for k in range(5):
            data.ctrl[k] = data.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)

    home = shaft()
    R = F.SHAFT_D / 2
    g = home
    print("=" * 74)
    print("机械臂送料到卡盘（合并场景；当前为已知失败基线）")
    print("=" * 74)
    print("  工件起点      : %s" % np.round(home, 4))
    print("  卡盘轴线      : %s" % np.round(axis, 4))
    print()

    pick = [
        ("接近", [g[0], g[1], g[2] + R + 0.075], GRIP_OPEN, 550),
        ("下降", [g[0], g[1], g[2] + 0.002], GRIP_OPEN, 400),
        ("夹紧", [g[0], g[1], g[2] + 0.002], GRIP_CLOSE, 320),
        ("对中", [g[0], g[1], g[2] + 0.020], GRIP_CLOSE, 260),
        ("抬起", [g[0], g[1], g[2] + 0.130], GRIP_CLOSE, 420),
    ]
    offset = None
    lift_tilt = None
    for idx, (name, tgt, grip, ns) in enumerate(pick):
        for _ in range(3):
            err = arm.move_to(np.array(tgt), n_steps=max(90, ns // 3),
                              grip=grip, wrist_roll=WRIST_ROLL_GRASP)
            if err < 0.005:
                break
        if idx == 2:
            offset = arm.pad_center() - shaft()
        if idx == 4:
            lift_tilt = tilt_deg(data.qpos[sq + 3:sq + 7])
        print("  %-6s 轴=%s 倾斜=%5.1f deg  pad-轴=%5.1f mm"
              % (name, np.round(shaft(), 4),
                 tilt_deg(data.qpos[sq + 3:sq + 7]),
                 np.linalg.norm(arm.pad_center() - shaft()) * 1000))

    def move3(target, ns, grip):
        for _ in range(3):
            e = arm.move_to(np.asarray(target),
                            n_steps=max(90, ns // 3), grip=grip,
                            wrist_roll=WRIST_ROLL_GRASP)
            if e < 0.005:
                break
        return e

    print("  shaft_in_grip = %s mm（夹紧后只标定一次）"
          % np.round(offset * 1000, 2))
    move3(axis + np.array([0.0, 0.0, 0.10]) + offset, 900, GRIP_CLOSE)
    print("  高位横移  轴=%s 倾斜=%5.1f deg"
          % (np.round(shaft(), 4), tilt_deg(data.qpos[sq + 3:sq + 7])))
    move3(axis + offset, 700, GRIP_CLOSE)
    print("  下到轴线  轴=%s 同轴=%5.2f mm 倾斜=%5.1f deg"
          % (np.round(shaft(), 4), coax(),
             tilt_deg(data.qpos[sq + 3:sq + 7])))

    # Clamp while the robot still holds, then release and retreat.
    data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
    data.ctrl[arm.grip_act] = GRIP_CLOSE
    for _ in range(700):
        mujoco.mj_step(model, data)
    print("  卡爪夹紧  轴=%s 同轴=%5.2f mm 接触=%d"
          % (np.round(shaft(), 4), coax(), jaw_contacts()))
    move3(np.array([axis[0] - 0.08, axis[1] - 0.08, 0.15]), 400,
          GRIP_OPEN)
    for _ in range(800):
        data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
        mujoco.mj_step(model, data)
    final_coax = coax()
    final_tilt = tilt_deg(data.qpos[sq + 3:sq + 7])
    contacts = jaw_contacts()
    print("  松开退出  轴=%s 同轴=%5.2f mm 倾斜=%5.1f deg 接触=%d"
          % (np.round(shaft(), 4), final_coax, final_tilt, contacts))

    z0 = shaft()[2]
    for _ in range(800):
        data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
        mujoco.mj_step(model, data)
    slip = abs(z0 - shaft()[2]) * 1000
    x0 = shaft()[0]
    for _ in range(700):
        data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
        data.xfrc_applied[shaft_body, 0] = -20.0
        mujoco.mj_step(model, data)
    pull = abs(shaft()[0] - x0) * 1000
    print("  重力下沉  %.2f mm      20 N 拉  %.2f mm" % (slip, pull))
    print()

    ok = (final_coax <= COAX_LIMIT and final_tilt <= TILT_LIMIT
          and contacts >= CONTACT_MIN and slip <= SLIP_LIMIT
          and pull <= PULL_LIMIT)
    print("=" * 74)
    print("FEED_METRIC lift_tilt_deg=%.1f coax_mm=%.2f tilt_deg=%.1f "
          "contacts=%d slip_mm=%.2f pull_mm=%.2f"
          % (lift_tilt, final_coax, final_tilt, contacts, slip, pull))
    print("FEED_METRIC %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        print("  已知阻塞：轴在抬起时翻转 %.1f deg，见文件头说明。"
              % lift_tilt)
        print("  修复点在抓取姿态，不在送料路径；本脚本故意不挂 verify_all。")
    print("=" * 74)
    print("OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
