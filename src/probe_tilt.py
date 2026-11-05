"""probe_tilt.py -- who tips the rod? Contact-by-contact torque in the rod frame.

The V jaws hold the rod with 4 contacts at 105 N and it follows the arm 1:1,
yet the rod sits 20 deg off axis. Force alone cannot explain that: a balanced
4-point V grip produces a net moment of zero. So the tilt must come from one
asymmetric contact -- the gripper base box pressing on top, a blade striking
one side only, or the two V mouths being staggered along the rod.

This prints, for every contact that touches the shaft:
  s        axial position along the rod (mm)
  lat      perpendicular offset from the rod centre line (mm)
  Fn       normal force (N)
  dM       moment about the rod centre (N.mm), split into the two bending
           axes and the spin axis

A contact set that is symmetric about s=0 with balanced dM is fine. Anything
lopsided is the tilt source. Read-only facts, no pass/fail.

Usage
-----
    python src/probe_tilt.py            # V jaws
    python src/probe_tilt.py --flat     # stock flat pads
    python src/probe_tilt.py --stage descend|closed
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_feed_chuck as D
from demo_08_pick_and_load import Arm, WRIST_ROLL_GRASP, GRIP_OPEN
from probe_grasp_contact import gname


def contact_force(m, d, i):
    """World-frame contact force (on geom2) and its position."""
    c = d.contact[i]
    f6 = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f6)
    frame = np.array(c.frame).reshape(3, 3)   # rows: normal, tangent, tangent
    return frame.T @ f6[:3], np.array(c.pos)


def report(m, d, sh, tag):
    c = d.xpos[sh]
    R = d.xmat[sh].reshape(3, 3)
    v = R[:, 2] / np.linalg.norm(R[:, 2])
    u1, u2 = R[:, 0], R[:, 1]
    print("\n  [%s] rod centre %s" % (tag, np.round(c, 4)))
    print("  [%s] rod axis   %s   deviation %.1f deg"
          % (tag, np.round(v, 3),
             np.degrees(np.arccos(min(1.0, abs(v @ np.array([1.0, 0, 0])))))))

    tot = np.zeros(3)
    rows = []
    for i in range(d.ncon):
        a = gname(m, d.contact[i].geom1)
        b = gname(m, d.contact[i].geom2)
        if not ("shaft" in a or "shaft" in b):
            continue
        F, p = contact_force(m, d, i)
        r = p - c
        s = float(r @ v)
        lat = r - s * v
        M = np.cross(r, F)
        tot += M
        rows.append((a, b, s, lat, float(np.linalg.norm(F)),
                     float(M @ u1), float(M @ u2), float(M @ v)))

    print("  %-24s %-24s %7s %6s %6s %8s %8s %8s"
          % ("geom1", "geom2", "s/mm", "lat1", "lat2", "dM1", "dM2", "spin"))
    for a, b, s, lat, fn, m1, m2, msp in sorted(rows, key=lambda r: -r[4]):
        print("  %-24s %-24s %+7.2f %+6.2f %+6.2f %+8.2f %+8.2f %+8.2f"
              % (a[:24], b[:24], s * 1000, lat @ u1 * 1000, lat @ u2 * 1000,
                 m1 * 1000, m2 * 1000, msp * 1000))
    print("  net moment  bend1 %+.2f  bend2 %+.2f  spin %+.2f  (N.mm)"
          % (tot @ u1 * 1000, tot @ u2 * 1000, tot @ v * 1000))
    if not rows:
        print("  no shaft contacts")


def main(v_jaw=True, close=-0.010, mu=0.05, stage="closed"):
    m = D.build_model(v_jaw=v_jaw, lift=True)
    m.geom_friction[:, 0] = mu
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    sq = m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    wrist = D.V_WRIST if v_jaw else WRIST_ROLL_GRASP
    arm = Arm(m, d)
    c0 = d.xpos[sh].copy()

    report(m, d, sh, "rest")
    arm.move_to(c0 + np.array([0, 0, 0.060]), 300, GRIP_OPEN, wrist)
    arm.move_to(c0 + np.array([0, 0, 0.002]), 400, GRIP_OPEN, wrist)
    if stage == "descend":
        report(m, d, sh, "descend")
        return
    report(m, d, sh, "descend")

    for i in range(600):
        d.ctrl[arm.grip_act] = close
        mujoco.mj_step(m, d)
        if i in (60, 120, 300):
            report(m, d, sh, "closing t=%d" % i)
    report(m, d, sh, "closed")
    if stage == "lift":
        arm.move_to(c0 + np.array([0, 0, 0.062]), 500, close, wrist)
        report(m, d, sh, "lifted")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", action="store_true")
    ap.add_argument("--close", type=float, default=-0.010)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--stage", default="closed",
                    choices=["descend", "closed"])
    a = ap.parse_args()
    main(v_jaw=not a.flat, close=a.close, mu=a.mu, stage=a.stage)
