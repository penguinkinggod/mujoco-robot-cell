"""arc_jaws.py -- [ABANDONED EXPERIMENT, kept for reference] bored soft jaws.

STATUS: never got a stable grasp. The 90-degree V route in
diag_feed_chuck.py won (8/8, carry=physics), so this is parked. Do not
mistake it for the working gripper.

arc_jaws.py -- bored soft jaws: two half-round liners that CAGE the rod.

Why this replaces the V groove
------------------------------
Measured with probe_grasp_contact.py (2026-11-06), V slots never worked:

  - Every wrist_roll from -1.60 to +1.55 gave at most 2 jaw-on-shaft
    contacts, and 0 after a 60 mm lift. The part never followed the arm.
  - Each V contributed only ONE blade (v1). Two contact points at
    z=40.2 / 44.2 mm are a single line again, so the rod pivots 16.6 deg.
  - The "only 2.3 mm slip under 5 N" was the static cradle holding the rod,
    not the jaws (4 cradle contacts at clamp time).

Geometry, not tuning: a 90 deg V needs H = R/sin(45) = 14.14 mm to reach the
tangent points of a D20 rod. The groove was 8 mm, so the blades' END FACES
pushed on the rod instead of the flanks. Deepening it would drive the pads
through the table. Tuning wrist angle cannot change that.

What real machine-tool loading uses is a bored soft jaw: each finger carries
a half-round pocket cut to the part diameter. Closed, the two halves form a
round hole that surrounds the rod over its full circumference. That is
CAGING / form closure -- the rod cannot rotate about a contact line because
there is no single contact line, and it cannot fall out because the two
halves meet at top and bottom. Gravity is carried by the lower half of the
hole; friction only has to stop axial sliding.

Design (all values in metres)
-----------------------------
    ARC_R_IN   10.3 mm  pocket radius (D20 rod + 0.3 mm diametral clearance)
    ARC_L      32 mm    axial length -- resist the tipping couple
    ARC_N      9        flat facets per half circle (MuJoCo has no concave
                        primitive; boxes are the reliable approximation)
    ARC_TH     1.8 mm   liner wall thickness

The pocket centre is pinned to the WORKPIECE CENTRE in world space, and the
closing command is auto-calibrated so the two half pockets sit naturally on
their pads. Both halves share one centre, so at that command they form one
complete circle with no gap at top or bottom.

Usage
-----
    python src/arc_jaws.py                 # self-test: print the calibration
    from arc_jaws import install, ARC_CLOSE, ARC_WRIST
    install(spec)                          # then spec.compile()
"""
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scene_full as F
from demo_08_pick_and_load import Arm, GRIP_OPEN

ARC_WRIST = -1.60         # u = jaw axis is horizontal & square to the rod
ARC_R_IN = 0.0103         # pocket radius, D20 rod + 0.3 mm clearance
ARC_L = 0.032             # axial length of the pocket
ARC_N = 9                 # facets per half circle
ARC_TH = 0.0018           # liner wall
ARC_SPAN = np.pi / 2      # half circle: -90 .. +90 deg
ARC_CLOSE = None          # auto-calibrated by install(), read it afterwards

PADS = ("fixed_jaw_pad", "moving_jaw_pad")


def _body_of_geom(spec, m, geom_id):
    bid = m.geom_bodyid[geom_id]
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
    return next(b for b in spec.bodies if b.name == name)


def _pad_faces(td, m, g1, g2):
    """World-space inner faces of the two pads and their separation."""
    p1 = td.geom_xpos[g1].copy()
    p2 = td.geom_xpos[g2].copy()
    u = p2 - p1
    u /= np.linalg.norm(u)
    faces = []
    for g, p, other in ((g1, p1, p2), (g2, p2, p1)):
        x = td.geom_xmat[g].reshape(3, 3)[:, 0].copy()
        if np.dot(x, other - p) < 0:
            x = -x
        faces.append(p + m.geom_size[g][0] * x)
    gap = float(np.dot(faces[1] - faces[0], u))
    return faces, u, gap


def _calibrate_close(td, m, g1, g2, qadr):
    """Pick the gripper qpos whose pad faces are 2*ARC_R_IN apart."""
    best = None
    q0 = td.qpos[qadr]
    for q in np.linspace(0.0, 0.35, 71):
        td.qpos[qadr] = q
        mujoco.mj_forward(m, td)
        _, _, gap = _pad_faces(td, m, g1, g2)
        err = abs(gap - 2 * ARC_R_IN)
        if best is None or err < best[0]:
            best = (err, q, gap)
    td.qpos[qadr] = q0
    mujoco.mj_forward(m, td)
    return best[1], best[2]


def _facet(O, u, v, w, side, theta, radius, width):
    """One box of the half-round liner.

    side=+1 -> the moving-jaw half (covers +u), -1 -> fixed-jaw half.
    Returns (centre, rotation columns, half sizes).
    """
    d = side * np.cos(theta) * u + np.sin(theta) * w
    c = O + radius * d
    x = -d                                   # box normal points at the centre
    y = v                                    # axial
    z = np.cross(x, y)                       # tangential
    z /= np.linalg.norm(z)
    R = np.column_stack([x, y, z])
    return c, R, np.array([ARC_TH / 2, ARC_L / 2, width / 2])


def install(spec, verbose=True):
    """Replace both flat pads with bored soft jaws. Returns ARC_CLOSE."""
    global ARC_CLOSE

    tmp = spec.compile()
    td = mujoco.MjData(tmp)
    gid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_GEOM, n)
    jid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_JOINT, n)
    bid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_BODY, n)
    qadr = tmp.jnt_qposadr[jid("gripper")]

    # Solve the grasp IK exactly as the real flow does, then lock the wrist.
    # Cutting the pockets at the HOME pose puts them 44 deg off the rod axis.
    #
    # NEVER do td.qpos[:] = 0 here. "shaft" has a FREE joint, whose qpos IS
    # its world pose; zeroing it teleports the workpiece to the origin and
    # zeroes its quaternion. The old V-jaw builder did exactly that, so every
    # slot it cut was positioned against a phantom rod at (0,0,0) -- measured
    # pocket centre [-0.071 -0.008 -0.014] instead of [0.28 -0.14 0.04].
    # That single line explains why no wrist angle or slot depth ever worked.
    td.qpos[tmp.jnt_qposadr[jid("wrist_roll")]] = ARC_WRIST
    mujoco.mj_forward(tmp, td)
    arm = Arm(tmp, td)
    sh = bid("shaft")
    c0 = td.xpos[sh].copy()
    arm.move_to(c0 + np.array([0, 0, 0.060]), 300, GRIP_OPEN, ARC_WRIST)
    arm.move_to(c0 + np.array([0, 0, 0.002]), 400, GRIP_OPEN, ARC_WRIST)

    g1, g2 = gid(PADS[0]), gid(PADS[1])
    ARC_CLOSE, gap = _calibrate_close(td, tmp, g1, g2, qadr)
    td.qpos[qadr] = ARC_CLOSE
    mujoco.mj_forward(tmp, td)

    # The pocket centre is the WORKPIECE centre, not an average of block
    # centres. Pinning it in world space is what makes both halves share one
    # circle instead of two staggered ones.
    O = td.xpos[sh].copy()
    v = td.xmat[sh].reshape(3, 3)[:, 2].copy()
    v /= np.linalg.norm(v)
    faces, u, gap = _pad_faces(td, tmp, g1, g2)
    u -= np.dot(u, v) * v
    u /= np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    w = z - np.dot(z, u) * u - np.dot(z, v) * v
    w /= np.linalg.norm(w)

    if verbose:
        print("    arc_jaw: wrist %.2f  close=%.4f  pad-face gap %.2f mm "
              "(target %.2f)" % (ARC_WRIST, ARC_CLOSE, gap * 1000,
                                 2 * ARC_R_IN * 1000))
        print("    arc_jaw: pocket centre %s  axis %s  u %s  w %s"
              % (np.round(O, 4), np.round(v, 2), np.round(u, 2),
                 np.round(w, 2)))
        print("    arc_jaw: pad bodies %s / %s" %
              (mujoco.mj_id2name(tmp, mujoco.mjtObj.mjOBJ_BODY,
                                 tmp.geom_bodyid[g1]),
               mujoco.mj_id2name(tmp, mujoco.mjtObj.mjOBJ_BODY,
                                 tmp.geom_bodyid[g2])))

    radius = ARC_R_IN + ARC_TH / 2
    step = 2 * ARC_SPAN / ARC_N
    width = 2 * radius * np.sin(step / 2) * 1.12     # slight overlap
    bodies = {PADS[0]: _body_of_geom(spec, tmp, g1),
              PADS[1]: _body_of_geom(spec, tmp, g2)}
    body_ids = {PADS[0]: tmp.geom_bodyid[g1], PADS[1]: tmp.geom_bodyid[g2]}

    for name, side in ((PADS[0], -1.0), (PADS[1], +1.0)):
        b_id = body_ids[name]
        p_b = td.xpos[b_id].copy()
        R_b = td.xmat[b_id].reshape(3, 3)
        body = bodies[name]

        for k in range(ARC_N):
            theta = -ARC_SPAN + (k + 0.5) * step
            c, R, hs = _facet(O, u, v, w, side, theta, radius, width)
            ng = body.add_geom()
            ng.name = "%s_arc%d" % (name.split("_")[0], k + 1)
            ng.type = mujoco.mjtGeom.mjGEOM_BOX
            ng.size = hs
            ng.pos = R_b.T @ (c - p_b)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(
                q, np.ascontiguousarray(R_b.T @ R).reshape(-1))
            ng.quat = q
            ng.rgba = [0.90, 0.58, 0.16, 1.0]
            ng.contype, ng.conaffinity = F.CT_ROBOT
            ng.friction = [1.2, 0.02, 0.002]

    # Inert dots at the SHARED pocket centre. BOTH go on the base body.
    #
    # The first version put one ref on each pad body. The moving pad slides
    # ~11 mm when the fingers open, so the average of the two refs drifted
    # by half that -- Arm.pad_center() then aimed IK at a point 11 mm away
    # from the hole, and the liner hit the rod during the DESCEND, before
    # any closing happened (measured: tilt 41.9 deg right after descent,
    # shaft_geom/moving_arc1 contact). Anchoring both refs to the body that
    # does NOT slide keeps pad_center() exactly on the hole centre at any
    # aperture.
    base_id = body_ids[PADS[0]]
    p_base = td.xpos[base_id].copy()
    R_base = td.xmat[base_id].reshape(3, 3)
    base_body = bodies[PADS[0]]
    for k in range(2):
        ref = base_body.add_geom()
        ref.name = "grasp_ref_%d" % k
        ref.type = mujoco.mjtGeom.mjGEOM_SPHERE
        ref.size = [0.0008, 0.0, 0.0]
        ref.pos = R_base.T @ (O - p_base)
        ref.rgba = [0.1, 0.1, 0.1, 0.0]
        ref.contype, ref.conaffinity = 0, 0

    for name in PADS:
        g = spec.geom(name)
        if g is not None:
            spec.delete(g)
    if verbose:
        print("    arc_jaw: 2 x %d facets -> D%.1f mm pocket, L%.0f mm"
              % (ARC_N, 2 * ARC_R_IN * 1000, ARC_L * 1000))
    return ARC_CLOSE


if __name__ == "__main__":
    import diag_feed_chuck as D
    D.build_model(arc_jaw=True, lift=True)
    print("ARC_CLOSE =", ARC_CLOSE)
