"""probe_grasp_pose.py -- find a wrist roll where the jaws can actually grip.

Why this exists
---------------
Every V-jaw attempt so far was cut at wrist_roll = 1.55 and every one failed.
The reason is kinematic, not frictional: at 1.55 the MOVING finger travels
along [15.5 16.3 -1.4] mm while the fixed finger never moves, so the closing
direction is 46 deg away from perpendicular to the rod. Two jaws closing
diagonally along a cylinder cannot straddle it -- they push it lengthwise.

Worse, `move_to()` aims the MIDPOINT of the two pads. With one fixed finger
and one moving finger that midpoint drifts by half the finger travel every
time the gripper opens or closes, so the arm aims at a point that is not
where the jaws will meet.

This probe measures, for a range of wrist rolls, at the real grasp pose:
  u.v   |cos| between the closing direction and the rod axis   (want ~0)
  u.z   how far the closing direction leans out of horizontal  (want ~0)
  mid   where the pads' midpoint is relative to the rod        (want ~0)
  span  pad-centre distance at the closed command
  reach IK residual for the approach pose (is the pose reachable at all)

Read-only. No pass/fail, just the numbers needed to pick a pose.

Usage
-----
    python src/probe_grasp_pose.py
    python src/probe_grasp_pose.py --qclose 0.10 --qopen 0.50
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diag_feed_chuck as D
from demo_08_pick_and_load import Arm, GRIP_OPEN


def measure(wrist, qclose, qopen, rod=None):
    m = D.build_model(lift=True)          # stock flat pads, no V blocks
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    g1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
    g2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad")
    qa = m.jnt_qposadr[mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]

    arm = Arm(m, d)
    c0 = d.xpos[sh].copy()
    try:
        arm.move_to(c0 + np.array([0, 0, 0.060]), 300, GRIP_OPEN, wrist)
        arm.move_to(c0 + np.array([0, 0, 0.002]), 400, GRIP_OPEN, wrist)
    except Exception as e:                                   # noqa: BLE001
        return None, str(e)

    c = d.xpos[sh].copy()
    R = d.xmat[sh].reshape(3, 3)
    v = R[:, 2] / np.linalg.norm(R[:, 2])
    if rod is not None:
        v = rod

    d.qpos[qa] = qclose
    mujoco.mj_forward(m, d)
    p1c, p2c = d.geom_xpos[g1] - c, d.geom_xpos[g2] - c
    d.qpos[qa] = qopen
    mujoco.mj_forward(m, d)
    p2o = d.geom_xpos[g2] - c
    d.qpos[qa] = qclose
    mujoco.mj_forward(m, d)

    u_raw = p2o - p2c                   # how the moving finger travels
    u = u_raw - (u_raw @ v) * v         # strip the along-rod part
    if np.linalg.norm(u) < 1e-9:
        return None, "no travel"
    # IMPORTANT: report the RAW axial share. Stripping it first (as an earlier
    # version did) makes |u.v| identically zero and hides the real defect: at
    # wrist 1.55 the finger travels 16 mm ALONG the rod while closing, which
    # drags the part lengthwise instead of straddling it.
    uv_raw = float(abs(u_raw @ v) / np.linalg.norm(u_raw))
    u /= np.linalg.norm(u)
    mid = 0.5 * (p1c + p2c)
    span = float(np.linalg.norm(p2c - p1c))
    # slop: how much the midpoint moves between open and closed
    d.qpos[qa] = qopen
    mujoco.mj_forward(m, d)
    mid_o = 0.5 * ((d.geom_xpos[g1] - c) + (d.geom_xpos[g2] - c))
    d.qpos[qa] = qclose
    mujoco.mj_forward(m, d)

    return dict(
        u=u, uv=uv_raw, uz=float(abs(u[2])),
        mid=mid, mid_sp=float(np.linalg.norm(mid_o - mid)),
        span=span, reach=float(np.linalg.norm(mid)),
        p1=p1c, p2=p2c), None


def main(qclose=0.10, qopen=0.50):
    print("qclose=%.2f qopen=%.2f" % (qclose, qopen))
    print("%6s %8s %6s %7s %8s %8s %8s  %s"
          % ("wrist", "along%", "u.z", "span/mm", "midX", "midY", "midZ",
             "verdict"))
    print("-" * 78)
    best = None
    for w in np.arange(-3.0, 3.01, 0.2):
        r, err = measure(w, qclose, qopen)
        if r is None:
            print("%+6.2f  --  %s" % (w, err))
            continue
        mid = r["mid"] * 1000
        ok = r["uv"] < 0.20 and r["uz"] < 0.20
        print("%+6.2f %6.3f %6.3f %7.1f %+8.1f %+8.1f %+8.1f  %s"
              % (w, r["uv"], r["uz"], r["span"] * 1000,
                 mid[0], mid[1], mid[2], "OK" if ok else ""))
        score = r["uv"] * 3 + r["uz"] + r["reach"] / 0.030
        if best is None or score < best[0]:
            best = (score, w, r)
    if best:
        _, w, r = best
        print("\nbest wrist %.2f: |u.v|=%.3f u.z=%.3f mid=%s mm span=%.1f mm"
              % (w, r["uv"], r["uz"], np.round(r["mid"] * 1000, 1),
                 r["span"] * 1000))
        print("  closing direction u = %s" % np.round(r["u"], 3))
        print("  fixed pad at %s mm" % np.round(r["p1"] * 1000, 1))
        print("  moving pad at %s mm" % np.round(r["p2"] * 1000, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qclose", type=float, default=0.10)
    ap.add_argument("--qopen", type=float, default=0.50)
    a = ap.parse_args()
    main(a.qclose, a.qopen)
