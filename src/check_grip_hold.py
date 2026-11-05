"""Unit check for the grasp, independent of the 12-stage feed flow.

Why this exists
---------------
Tuning jaw geometry by running diag_feed_chuck.main() costs ~25 s per trial
and only reports the jaw state as a side effect of 12 other stages. A "pass"
in the full flow says nothing about whether the part was ever held -- the
verified 7/7 uses a kinematic lock for the carry.

This check asks one question, in ~3 s:

    with the arm parked and friction nearly zero, does the workpiece stay
    in the jaws under a downward load, and does it follow a vertical lift?

That is the definition of form closure. If the jaws only work at mu=1.2,
they are a friction grip and no amount of carry tuning will make them real.

2026-11-06 correction
---------------------
The old version counted contacts by "does either name contain 'pad'". That
never checked the OTHER side, so a jaw pressing on the static cradle looked
like a good grip. probe_grasp_contact.py showed the V jaw had only 2 real
jaw-on-shaft contacts, 4 cradle contacts, and 0 contacts after a lift. The
filter here now requires the other geom to be the shaft.

Usage
-----
    python src/check_grip_hold.py            # bored soft jaws (default)
    python src/check_grip_hold.py --v       # V slots, for contrast
    python src/check_grip_hold.py --flat    # stock flat pads, for contrast

The full flow is the LAST place to validate a jaw, not the first.
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_feed_chuck as D
from demo_08_pick_and_load import Arm, WRIST_ROLL_GRASP, GRIP_OPEN


def is_jaw(name):
    return ("pad" in name or "arc" in name or
            "v1" in name or "v2" in name)


def jaw_contacts(m, d):
    """(count, total normal force) for REAL jaw-on-shaft contacts only."""
    f6 = np.zeros(6)
    n, total = 0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        a = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        has_shaft = "shaft" in a or "shaft" in b
        has_jaw = is_jaw(a) or is_jaw(b)
        if has_shaft and has_jaw:
            mujoco.mj_contactForce(m, d, i, f6)
            n += 1
            total += abs(f6[0])
    return n, total


def cradle_contacts(m, d):
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        a = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if "cradle" in a or "cradle" in b:
            n += 1
    return n


def axis_tilt_deg(d, sh):
    """Angle between the rod AXIS and world x -- NOT the quaternion angle.

    tilt_deg() measures total rotation against SHAFT_Q0, so a cylinder that
    merely rolls about its own axis reports tens of degrees of "tilt" even
    though its axis never moved. For a round part that is harmless; only the
    axis direction matters for loading it into a chuck.
    """
    ax = d.xmat[sh].reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(min(1.0, abs(float(ax[0]))))))


def run(kind="arc", grip_close=None, mu=0.05, pull_N=5.0, steps=1200,
        disable_base_contact=False):
    v_jaw = kind == "v"
    arc_jaw = kind == "arc"
    m = D.build_model(v_jaw=v_jaw, arc_jaw=arc_jaw, lift=True)
    if disable_base_contact:
        gripper_body = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if (m.geom_bodyid[g] == gripper_body and "pad" not in name
                    and "_v" not in name):
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
    # MuJoCo takes max(friction of the two surfaces), so lower every geom.
    m.geom_friction[:, 0] = mu
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    if grip_close is None:
        if arc_jaw:
            import arc_jaws
            grip_close = arc_jaws.ARC_CLOSE
        else:
            grip_close = -0.010
    if arc_jaw:
        import arc_jaws
        wrist = arc_jaws.ARC_WRIST
    else:
        wrist = D.V_WRIST if v_jaw else WRIST_ROLL_GRASP

    sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    sq = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                          "shaft_free")]
    c0 = d.xpos[sh].copy()
    arm = Arm(m, d)

    arm.move_to(c0 + np.array([0, 0, 0.060]), 300, GRIP_OPEN, wrist)
    arm.move_to(c0 + np.array([0, 0, 0.002]), 400, GRIP_OPEN, wrist)
    for _ in range(600):
        d.ctrl[arm.grip_act] = grip_close
        mujoco.mj_step(m, d)

    n, fn = jaw_contacts(m, d)
    nc = cradle_contacts(m, d)
    held = d.xpos[sh].copy()
    tilt = D.tilt_deg(d.qpos[sq + 3:sq + 7])
    ax0 = axis_tilt_deg(d, sh)
    print("  闭合   夹爪-工件 %d 点 / %.1f N   四元数倾角 %.1f deg  "
          "轴线偏离 %.1f deg  托架 %d 点"
          % (n, fn, tilt, ax0, nc))

    # 1) downward load, arm parked, almost no friction
    for _ in range(steps):
        d.ctrl[arm.grip_act] = grip_close
        d.xfrc_applied[sh, 2] = -pull_N
        mujoco.mj_step(m, d)
    d.xfrc_applied[sh, :] = 0.0
    pull_move = np.linalg.norm(d.xpos[sh] - held) * 1000
    print("  %.0f N 下拉 / mu=%.2f   位移 %.1f mm" % (pull_N, mu, pull_move))

    # 2) lift the arm 60 mm; does the part follow?
    pz0 = arm.pad_center()[2]
    z_before = d.xpos[sh][2]
    arm.move_to(c0 + np.array([0, 0, 0.062]), 500, grip_close, wrist)
    z_after = d.xpos[sh][2]
    dp = (arm.pad_center()[2] - pz0) * 1000
    n_after, _ = jaw_contacts(m, d)
    follow = (z_after - z_before) * 1000
    print("  抬起   pad %+.1f mm | 工件 z %.1f -> %.1f mm (跟走 %+.1f) "
          "| 夹爪-工件 %d 点"
          % (dp, z_before * 1000, z_after * 1000, follow, n_after))

    ax1 = axis_tilt_deg(d, sh)
    # follow is measured RELATIVE to how far the pads actually rose. The
    # position servos droop under load (a 60 mm command gave 33 mm here), so
    # an absolute follow threshold would fail a perfectly rigid grip. What
    # matters is that the rod tracks the pad 1:1 and stays in contact.
    tracking = follow / dp if dp > 1.0 else 0.0
    print("  跟踪比 %.2f   抬起后轴线偏离 %.1f deg" % (tracking, ax1))
    ok = (n >= 4 and ax0 <= 10.0 and pull_move < 2.0
          and dp > 20.0 and tracking > 0.85 and n_after >= 2)
    print("  => %s" % ("形封闭成立" if ok else "仍未夹稳"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", action="store_true", help="stock flat pads")
    ap.add_argument("--v", action="store_true", help="V slots")
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--grip", type=float, default=None)
    ap.add_argument("--no-base", action="store_true",
                    help="disable gripper base contacts for diagnosis")
    a = ap.parse_args()
    kind = "flat" if a.flat else ("v" if a.v else "arc")
    label = {"arc": "圆弧软爪", "v": "V 槽爪", "flat": "平板爪"}[kind]
    print("\n=== %s  (mu=%.2f) ===" % (label, a.mu))
    run(kind=kind, grip_close=a.grip, mu=a.mu,
        disable_base_contact=a.no_base)
