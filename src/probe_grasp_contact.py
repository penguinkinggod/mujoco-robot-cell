"""probe_grasp_contact.py -- print EVERY contact when the jaws close.

Why this exists
---------------
check_grip_hold.py counted contacts by asking "does either geom name contain
'pad'". After the V-jaw swap the blades are named fixed_jaw_pad_v1 .. v2, so
the filter still fires -- but it never asked what is on the OTHER side. A jaw
blade pressing on the static cradle, or on the other jaw, produces a big
normal force and looks exactly like a good grip. That is how "2 contacts /
96.5 N / only 2.3 mm of slip" got reported as an improvement while the arm
lifted 60 mm and the workpiece did not follow.

This probe names both sides, measures the pad centre and the shaft together,
and sweeps the closing command. No pass/fail, just facts.

Usage
-----
    python src/probe_grasp_contact.py            # V jaws, lifted rod
    python src/probe_grasp_contact.py --flat    # stock flat pads
    python src/probe_grasp_contact.py --sweep   # several closing commands
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_feed_chuck as D
from demo_08_pick_and_load import Arm, WRIST_ROLL_GRASP, GRIP_OPEN


def gname(m, gid):
    """Unnamed geoms are common in the SO-101 asset; show body + id instead."""
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if n:
        return n
    b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid])
    return "%s#%d" % (b or "?", gid)


def all_contacts(m, d):
    """[(geom1, geom2, normal force, world position), ...]"""
    f6 = np.zeros(6)
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        a = gname(m, c.geom1)
        b = gname(m, c.geom2)
        mujoco.mj_contactForce(m, d, i, f6)
        out.append((a, b, abs(f6[0]), np.array(c.pos)))
    return out


def brief(row):
    a, b, f, p = row
    return "%-26s %-26s %6.1f N  z=%+.1fmm" % (a, b, f, p[2] * 1000)


def run(v_jaw, close, mu, lift=True, verbose=True,
        arc_jaw=False, trace=False):
    m = D.build_model(v_jaw=v_jaw, arc_jaw=arc_jaw, lift=lift)
    m.geom_friction[:, 0] = mu
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    sq = m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    if arc_jaw:
        import arc_jaws
        wrist = arc_jaws.ARC_WRIST
        if close is None:
            close = arc_jaws.ARC_CLOSE
    else:
        wrist = D.V_WRIST if v_jaw else WRIST_ROLL_GRASP
    arm = Arm(m, d)
    c0 = d.xpos[sh].copy()

    arm.move_to(c0 + np.array([0, 0, 0.060]), 300, GRIP_OPEN, wrist)
    print("  after APPROACH  shaft=%s tilt=%.1f" %
          (np.round(d.xpos[sh], 4), D.tilt_deg(d.qpos[sq + 3:sq + 7])))
    arm.move_to(c0 + np.array([0, 0, 0.002]), 400, GRIP_OPEN, wrist)
    axv = d.xmat[sh].reshape(3, 3)[:, 2]
    print("  after DESCEND   shaft=%s tilt=%.1f  axis=%s  pads=%s" %
          (np.round(d.xpos[sh], 4), D.tilt_deg(d.qpos[sq + 3:sq + 7]),
           np.round(axv, 3), np.round(arm.pad_center(), 4)))
    if trace:
        for r in all_contacts(m, d)[:10]:
            print("     " + brief(r))
    for i in range(600):
        d.ctrl[arm.grip_act] = close
        mujoco.mj_step(m, d)
        if trace and i % 100 == 0:
            rows = all_contacts(m, d)
            ax = d.xmat[sh].reshape(3, 3)[:, 2]
            print("     t=%3d shaft=%s axis=%s ncon=%d | %s" %
                  (i, np.round(d.xpos[sh], 4), np.round(ax, 3), len(rows),
                   "; ".join("%s/%s %.0fN" % (r[0], r[1], r[2])
                             for r in rows[:4])))

    rows = all_contacts(m, d)
    shaft_rows = [r for r in rows if "shaft" in r[0] or "shaft" in r[1]]
    jaw_rows = [r for r in rows if "pad" in r[0] or "pad" in r[1] or
                "v1" in r[0] or "v1" in r[1] or
                "v2" in r[0] or "v2" in r[1]]
    jaw_on_shaft = [r for r in jaw_rows
                    if "shaft" in r[0] or "shaft" in r[1]]
    cradle_rows = [r for r in rows if "cradle" in r[0] or "cradle" in r[1]]

    print("\n--- close=%.3f  %s  mu=%.2f ---" %
          (close, "V jaw" if v_jaw else "flat", mu))
    print("  shaft  pos=%s  tilt=%.1f deg" %
          (np.round(d.xpos[sh], 4),
           D.tilt_deg(d.qpos[sq + 3:sq + 7])))
    print("  pads   centre=%s  gap-to-shaft=%.1f mm" %
          (np.round(arm.pad_center(), 4),
           np.linalg.norm(arm.pad_center() - d.xpos[sh]) * 1000))
    print("  contacts total=%d | shaft=%d | jaw-on-shaft=%d | cradle=%d" %
          (len(rows), len(shaft_rows), len(jaw_on_shaft), len(cradle_rows)))
    if verbose:
        for r in rows[:16]:
            print("     " + brief(r))
        if len(rows) > 16:
            print("     ... %d more" % (len(rows) - 16))

    # lift: measure BOTH the pad and the shaft, so a servo droop cannot be
    # mistaken for the part being dropped.
    pz0 = arm.pad_center()[2]
    sz0 = d.xpos[sh][2]
    arm.move_to(c0 + np.array([0, 0, 0.062]), 500, close, wrist)
    dp = (arm.pad_center()[2] - pz0) * 1000
    ds = (d.xpos[sh][2] - sz0) * 1000
    after = all_contacts(m, d)
    jaw_after = [r for r in after if
                 ("shaft" in r[0] or "shaft" in r[1]) and
                 ("pad" in r[0] or "pad" in r[1] or
                  "v1" in r[0] or "v1" in r[1] or
                  "v2" in r[0] or "v2" in r[1])]
    print("  LIFT   pad %+.1f mm | shaft %+.1f mm | jaw-on-shaft after = %d"
          % (dp, ds, len(jaw_after)))
    return len(jaw_on_shaft), ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--arc", action="store_true", help="bored soft jaws")
    ap.add_argument("--trace", action="store_true",
                    help="print shaft state while the jaws close")
    ap.add_argument("--sweep", action="store_true", help="sweep closing cmd")
    ap.add_argument("--wsweep", action="store_true",
                    help="sweep wrist_roll with a fixed closing cmd")
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--close", type=float, default=-0.010)
    a = ap.parse_args()
    if a.wsweep:
        print("  wrist | jaw-on-shaft | lift follow | mouth axis")
        print("  " + "-" * 62)
        for w in (-1.60, -1.30, -1.00, -0.60, 0.0, 0.60, 1.00, 1.55):
            D.V_WRIST = w
            try:
                n, ds = run(v_jaw=True, close=a.close, mu=a.mu,
                            verbose=False)
                print("  %+5.2f |      %2d       |  %+7.1f mm | %s"
                      % (w, n, ds, "see build log above"))
            except Exception as e:                     # noqa: BLE001
                print("  %+5.2f | FAILED: %s" % (w, e))
    else:
        base = None if a.arc else a.close
        closes = ([0.15, 0.10, 0.05, 0.00, -0.010]
                  if a.sweep else [base])
        for c in closes:
            run(v_jaw=not a.flat and not a.arc, close=c, mu=a.mu,
                arc_jaw=a.arc, trace=a.trace)
