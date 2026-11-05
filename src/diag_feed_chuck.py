"""diag_feed_chuck.py -- the ROBOT feeds a shaft into the three-jaw chuck.

STATUS 2026-11-05: PASS. Wired into verify_all.py CHECKS.

SCOPE: ONE EXPLICIT SIMPLIFICATION -- do NOT read this as full physics
---------------------------------------------------------------------
The CARRY is KINEMATIC. From the centring stage until the chuck has
clamped, every step writes the shaft qpos/quat back to
(world pin / site, SHAFT_Q0) and zeroes qvel. This ASSUMES the gripper
has perfect form closure. It does NOT prove the real gripper can hold the
shaft -- two flat pads on a D20 rod touch along one line, so the rod
pivots about it. The lock is released BEFORE any load test, and
FEED_METRIC reports that fact so nobody can quietly extend it through the
pull-out and fake a pass.

The workpiece is a PLAIN CYLINDER (no milled flat, no slot platform).
That is a deliberate choice, not an omission -- see HANDOFF section 16 for
why the flat and the fixture were deleted and what it cost.

What this DOES prove: given a shaft delivered on-axis and correctly
oriented, the chuck station can centre it, clamp it, hold it against
gravity and resist a 20 N pull-out AFTER the robot has fully retreated.

Evidence chain (controlled rounds)
----------------------------------
Grasp orientation:
- With the plain cylinder the real grasp used to come up 43.8 deg off
  axis, because the arm stopped 25-37 mm short under load and never
  reached the descend target. move3() now integrates the measured error
  back into the commanded target (re-sending the same absolute point just
  lands on the same droop): residual 24.8 -> 1.2 mm, grasp tilt 43.8 ->
  2.5 deg.

Hand-over to the chuck (this is where the flow actually broke):
- GRIP_CLOSE while the jaws close: robot and chuck fight for one shaft,
  the losing servo drags the pads 6.46 mm off axis.
- GRIP_OPEN with the lock anchored on the gripperframe site: the jaws
  push the shaft into the still-open pads, the arm gives way 12.07 mm and
  carries the shaft with it.
- Working order: pin the shaft in WORLD space, back the arm 55 mm clear,
  then let the jaws close on a shaft that nothing else touches.
  coax 1.05 mm, tilt 0.0 deg, contacts 6 (2+2+2), pull-out 0.00 mm.

Chuck jaws:
- JAW_LEN 28 -> 55 mm. A 28 mm jaw lets the 80 mm rod pivot like a ball
  joint: 6 contacts at clamp, one jaw lets go on release (4), axis
  10.9 deg off, 20 N pull-out ejects the shaft 4172 mm. 55 mm holds it.
- CHUCK_CLAMP_CTRL -0.009 -> -0.010. Less (0.003-0.007) leaves the rod
  loose in the jaws (tilt 15-18 deg); more (-0.014) shoves it 7.34 mm
  off-centre and one jaw leaves the surface entirely.
- Longer jaws reach 27 mm further out, so diag_chuck_health's "gripper
  stops outside the chuck" probe moved from -70 to -90 mm (13.78 -> 7.91).

Earlier rounds, kept because they still bind:
- scene_full.py sets NO spec.option, while scene_machining.py sets
  noslip_iterations=10 / noslip_tolerance=0 / cone=ELLIPTIC / impratio=5.
  Applying those from the start breaks the grasp, whose constants were
  tuned under the defaults. Patching only the friction-relevant fields
  into model.opt AT THE CLAMP STAGE raises total jaw normal force
  8.4 -> 18.7 N and drops 20 N pull-out from 1522 mm to 0.62 mm.
- Retreat bug: the move helper used to write CHUCK_OPEN_CTRL on every
  step, so retreating OPENED the chuck and dropped the shaft (coax 50.85
  mm). Passing CHUCK_CLAMP_CTRL for the retreat fixes it.

Still open, do not close: the carry is a kinematic lock. Two flat pads on
a cylinder are a LINE contact and cannot resist the tipping couple, which
is why the rod still hangs rather than being gripped. demo_09's 3/3 also
never checks orientation, so a vertical stick dropped in the slot still
"passes". A real fix is a jaw that closes around the diameter (V block or
bored soft jaws), not more tuning of the arm.

Pass/fail thresholds (written down BEFORE any of the scans)
-----------------------------------------------------------
    coaxiality      <= 2.0 mm   shaft axis vs chuck axis
    shaft tilt      <= 10.0 deg vs original axial orientation
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

LATHE_ORIGIN = (0.20, 0.12, 0.060)   # closest-to-success from x scan
SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])


def tilt_deg(quat):
    return float(np.degrees(
        2 * np.arccos(min(1.0, abs(float(np.dot(SHAFT_Q0, quat)))))))


CRADLE_DROP = 0.011      # 托条中心比两 pad 中点低多少（R10 + 板半厚 1 mm）
CRADLE_LEN = 0.060       # 托条沿工件轴向的长度


def add_grip_fix(spec, kind="cradle"):
    """Optional gripper patches, selected for measurement, not by taste.

    Why a plain cylinder still fails after deleting the flat and the slot
    (measured this session, exp_pad_align.py + exp_pad_friction.py):

    1. NOT friction. Sweeping the pad mu 0.1/0.3/0.8 gives literally
       identical tilt (41.2 / 56.3 deg) -> the contact is not sliding.
    2. NOT the pad's own 26.4 deg twist. Counter-rotating the moving pad
       to be parallel at the working aperture buys 11 deg (56.3 -> 44.9).
    3. It is the CONTACT LENGTH ALONG THE SHAFT. A 14.4 mm patch pressed
       on a Phi20 x 160 mm rod resists a tipping couple of only
       N * 7 mm ~ 0.08 N.m, while the rod's own weight hung 10 mm below
       the contact line needs ~0.02-0.04 N.m. So the rod swings like a
       pendulum until it is nearly vertical (measured 87-118 deg).
    "long" fixes the cause: extend both pads 40 mm along the shaft, which
    raises the tipping couple ~3x and spreads the squeeze over the rod.
    """
    if kind in ("v", "both"):
        _add_v_jaw(spec)
    if kind in ("cradle", "both"):
        _add_cradle(spec)
    # "long" (stretch the pads along the shaft) was tried and REJECTED:
    # it dropped closing tilt 41 -> 20 deg but the shaft then flipped to
    # 107 deg and no aperture lifted it. Adding plates is the wrong lever;
    # the real fix was the hand-over order plus longer CHUCK jaws. See
    # HANDOFF section 16.
    if kind not in ("v", "cradle", "both"):
        raise ValueError("unknown grip_fix: %r" % (kind,))


def _add_v_jaw(spec, half_angle_deg=45.0, arm=0.006):
    """V-groove on the FIXED pad only: two 45 deg blades opening toward the
    moving jaw. Fixed pad has no local rotation, so its +x is the inward
    face -- measured with _probe_pads.py, not guessed.

    V (two contact points) + the flat moving pad (one point) gives three
    points of contact, which stops the shaft rotating about a single line.
    """
    body = next(b for b in spec.bodies if b.name == "gripper")
    pad = next(g for g in body.geoms if g.name == "fixed_jaw_pad")
    vx = pad.pos[0] + pad.size[0]          # inner face of the pad
    vy = pad.pos[1]
    vz = pad.pos[2]
    a = np.radians(half_angle_deg)
    for sign in (+1, -1):
        g = body.add_geom()
        g.name = "vjaw_%s" % ("up" if sign > 0 else "dn")
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [0.0012, 0.008, arm]
        g.pos = [vx + arm * np.sin(a) * 0.5,
                 vy,
                 vz + sign * arm * np.cos(a) * 0.5]
        g.quat = [np.cos(a / 2), 0.0, sign * np.sin(a / 2), 0.0]
        g.rgba = [0.90, 0.58, 0.16, 1.0]
        g.contype, g.conaffinity = F.CT_ROBOT
    print("    grip_fix: V jaw at inner face x=%.4f (arm %.0f mm)"
          % (vx, arm * 1000))


def _add_cradle(spec):

    tmp = spec.compile()
    td = mujoco.MjData(tmp)
    jid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_JOINT, n)
    gid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_BODY, n)

    td.qpos[:] = 0.0
    td.qpos[tmp.jnt_qposadr[jid("gripper")]] = 0.15
    td.qpos[tmp.jnt_qposadr[jid("wrist_roll")]] = 1.55
    mujoco.mj_forward(tmp, td)

    Rg = td.xmat[bid("gripper")].reshape(3, 3)      # gripper local -> world
    pg = td.xpos[bid("gripper")].copy()
    mid = (td.geom_xpos[gid("fixed_jaw_pad")] +
           td.geom_xpos[gid("moving_jaw_pad")]) / 2.0

    # bar centre: straight below the grasp point, in WORLD z
    p_local = Rg.T @ (mid - np.array([0.0, 0.0, CRADLE_DROP]) - pg)
    # bar axes in world: local x -> -z (thickness), y -> y, z -> +x (length)
    Rt = np.array([[0.0, 0.0, 1.0],
                   [0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0]])
    q_local = np.zeros(4)
    mujoco.mju_mat2Quat(q_local, np.ascontiguousarray(Rg.T @ Rt).reshape(-1))

    body = next(b for b in spec.bodies if b.name == "gripper")
    g = body.add_geom()
    g.name = "grip_cradle"
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.001, 0.006, CRADLE_LEN / 2]
    g.pos = p_local
    g.quat = q_local
    g.rgba = [0.90, 0.58, 0.16, 1.0]
    g.contype, g.conaffinity = F.CT_ROBOT
    print("    grip_fix: cradle at local %s (len %.0f mm)"
          % (np.round(p_local, 4), CRADLE_LEN * 1000))


# ---------------- V-slot jaws (both pads) ----------------
# Geometry, not friction: a Phi20 rod sits in a 90 deg V at 14.14 mm from the
# apex, so the slot has to be that deep. The pads hang only 3 mm above the
# table at the grasp point (measured: pad lowest corner z = +2.3 / +3.0 mm),
# so a downward-opening slot would go through the table. That is why the
# workpiece has to be lifted onto two cradles first -- the cradles hold it at
# its ENDS, leaving the middle 80 mm free for the jaws to reach underneath.
# US9808936 ("Two-phase gripper to reorient and grasp") gives the jaw-cavity
# constraint H <= R/sin(theta) so the FINGERS do not touch before the part is
# held. For a 90 deg V and Phi20 that is H <= 14.14 mm -- an UPPER bound, not
# a minimum. I had this backwards and set 16 mm; the sweep below showed the
# shallower the better: 8 mm -> 16.6 deg / 2.3 mm, 16 mm -> 53.6 deg / 3.9 mm.
V_DEPTH = 0.008          # slot depth (m); swept 8..20 mm, 8 is best so far
V_BAR = 0.004            # blade thickness
V_LEN = 0.024            # slot length along the workpiece axis
V_QREF = 0.10            # gripper qpos the slot frame is measured at
# The V slot has to be cut along the rod axis, so the jaw closing axis must
# be perpendicular to it.
#
# 2026-11-06 RE-MEASURED after fixing the free-joint bug (the old builder
# zeroed qpos and teleported the rod to the origin, so the earlier scan that
# picked -1.00 was measuring a phantom). probe_grasp_contact.py --wsweep,
# jaw-on-shaft contacts at clamp / shaft follow on a 60 mm lift:
#   -1.60 -> 0 / -16.1 mm | -1.30 -> 0 / -21.9 | -1.00 -> 0 / -23.9
#   -0.60 -> 1 / +39.3 mm |  0.00 -> 2 /   0.0 | +0.60 -> 0 /   0.0
#   +1.00 -> 0 /   0.0    | +1.55 -> 4 / +31.6 mm  <-- only real grip
# +1.55 is the stock WRIST_ROLL_GRASP from demo_08: it was right all along.
V_WRIST = -0.95
# A 90 deg V seat locates the rod by geometry, but CARRYING it is still
# friction-limited: 41 N of clamp x mu=0.05 is 2.05 N of friction against a
# 1.93 N part, so the rod rotated away 77 deg on the first fast traverse.
# Closing harder is what an operator does; -0.10 measured 22.9 N on the chuck
# jaws and 0.07 mm of 20 N pull-out.
V_GRIP_CLOSE = -0.10
LIFT_DZ = 0.030          # workpiece raised off the table
CRADLE_HALF_X = 0.005    # cradle half-width along the workpiece axis
CRADLE_BAR = 0.005
# Cradle depth is INDEPENDENT of jaw depth: the cradle only has to locate the
# rod, whereas the jaws have to swallow it. Sharing V_DEPTH dropped the rod
# 7.5 mm when the jaws got deeper (39.8 -> 32.3 mm) and invalidated the run.
CRADLE_DEPTH = 0.010


def _v_plates(O, u_ax, w_ax, v_ax, depth=None, bar=None, length=None):
    """Two 45 deg blades forming a V. u_ax = mouth normal, v_ax = slot axis.

    Defaults are resolved at CALL time, not at definition time, so a sweep
    can override V_DEPTH / V_LEN without reloading the module.
    """
    depth = V_DEPTH if depth is None else depth
    bar = V_BAR if bar is None else bar
    length = V_LEN if length is None else length
    out = []
    off = depth / 2 + bar / (2 * np.sqrt(2))
    for sgn in (+1, -1):
        c = O + (-off) * u_ax + (sgn * off) * w_ax
        x = (u_ax + sgn * w_ax) / np.sqrt(2)
        z = np.cross(x, v_ax)
        R = np.column_stack([x, v_ax, z])
        out.append((c, R, [depth * np.sqrt(2) / 2, length / 2, bar / 2]))
    return out


def _install_v_jaws(spec):
    """Replace both flat pads with V blocks, opening toward each other.

    Each pad gets a 90 deg groove whose axis runs along the workpiece, so the
    slot frame must be built from three mutually perpendicular directions:
      v = rod axis, u = jaw closing axis, w = the in-plane "up" direction.
    Hardcoding u = world +/-Z was wrong by 86.3 deg at the real grasp pose --
    the two grooves ended up staggered instead of facing each other.
    """
    tmp = spec.compile()
    td = mujoco.MjData(tmp)
    jid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_JOINT, n)
    gid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_GEOM, n)
    bid = lambda n: mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_BODY, n)

    # The slot has to face WORLD +/-Z AT THE GRASP POSE, not at home. The arm
    # swings tens of degrees between the two, so defining the blades at home
    # put both jaws above the rod and the slot pointed off into space
    # (measured: gripper blades at z=+53 mm while the rod centre was z=+40).
    # So solve the grasp IK first with the stock pads, then cut the slots in
    # that pose.
    from demo_08_pick_and_load import Arm
    # NEVER zero the whole qpos here: "shaft" has a FREE joint, so its qpos
    # IS its world pose. Zeroing it teleports the workpiece to the origin and
    # zeroes its quaternion, so every slot below was cut against a phantom rod
    # at (0,0,0). All earlier wrist/depth conclusions came from that bug.
    td.qpos[tmp.jnt_qposadr[jid("wrist_roll")]] = V_WRIST
    mujoco.mj_forward(tmp, td)
    arm = Arm(tmp, td)
    sh = mujoco.mj_name2id(tmp, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    c0 = td.xpos[sh].copy()
    # Same approach the real flow uses, stepped through physics. A bare ik()
    # call returns a different joint solution for the same tool pose (the arm
    # is redundant), and the blades ended up rotated 44 deg away from the
    # slot axis -- enough that the rod never entered the groove.
    arm.move_to(c0 + np.array([0, 0, 0.060]), 300, 0.50, V_WRIST)
    arm.move_to(c0 + np.array([0, 0, 0.002]), 400, 0.50, V_WRIST)
    td.qpos[tmp.jnt_qposadr[jid("gripper")]] = V_QREF
    mujoco.mj_forward(tmp, td)

    # v = rod axis; u = closing axis with its along-rod part stripped;
    # w = gravity projected into the plane left over. All three orthogonal.
    v_ax = td.xmat[sh].reshape(3, 3)[:, 2].copy()
    v_ax /= np.linalg.norm(v_ax)
    p1 = td.geom_xpos[gid("fixed_jaw_pad")].copy()
    p2 = td.geom_xpos[gid("moving_jaw_pad")].copy()
    u = p2 - p1
    u -= np.dot(u, v_ax) * v_ax
    u /= np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])
    w_ax = z - np.dot(z, u) * u - np.dot(z, v_ax) * v_ax
    w_ax /= np.linalg.norm(w_ax)
    print("    v_jaw: wrist %.2f, mouth axis [%+.2f %+.2f %+.2f] (u.v=%.3f)"
          % (V_WRIST, *u, np.dot(u, v_ax)))

    # The stock SO-101 pads are NOT axially aligned: the fixed pad's face sits
    # 14.8 mm behind the moving pad's. Cutting each V at its own pad therefore
    # staggers the two mouths 15.4 mm apart along the rod; the jaws then WEDGE
    # the rod instead of cradling it and it ends up 20 deg off axis (measured,
    # with 4 contacts and 105 N -- force alone never showed this). So build
    # both V blocks on ONE common axial station, halfway between the pads.
    pads = []
    for pad_name, usgn in (("fixed_jaw_pad", +1.0), ("moving_jaw_pad", -1.0)):
        g = gid(pad_name)
        p = td.geom_xpos[g].copy()
        half_th = tmp.geom_size[g][0]
        O = p + u * half_th * usgn
        pads.append((pad_name, usgn, g, p, half_th, O,
                     float((O - c0) @ v_ax)))
    s_mid = 0.5 * (pads[0][6] + pads[1][6])
    print("    v_jaw: pad mouths were %.1f mm apart along the rod -> "
          "aligned to one station" % ((pads[1][6] - pads[0][6]) * 1000))

    mouths = []                            # (usgn, mouth centre) per finger
    for pad_name, usgn, g, p, half_th, O, s in pads:
        b = tmp.geom_bodyid[g]
        O = O + (s_mid - s) * v_ax         # common axial station
        mouths.append((usgn, O))
        u_ax = u * usgn                    # mouth normal = closing axis
        print("    v_jaw: %s mouth s=%+.1f mm, lateral %+.1f mm"
              % (pad_name, (O - c0) @ v_ax * 1000, (O - c0) @ w_ax * 1000))

        p_b = td.xpos[b].copy()
        R_b = td.xmat[b].reshape(3, 3)
        body = next(bb for bb in spec.bodies
                    if bb.name == mujoco.mj_id2name(
                        tmp, mujoco.mjtObj.mjOBJ_BODY, b))

        for k, (c, R, hs) in enumerate(_v_plates(O, u_ax, w_ax, v_ax)):
            ng = body.add_geom()
            ng.name = "%s_v%d" % (pad_name, k + 1)
            ng.type = mujoco.mjtGeom.mjGEOM_BOX
            ng.size = hs
            ng.pos = R_b.T @ (c - p_b)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.ascontiguousarray(R_b.T @ R).reshape(-1))
            ng.quat = q
            ng.rgba = [0.90, 0.58, 0.16, 1.0]
            ng.contype, ng.conaffinity = F.CT_ROBOT

    # ONE stable TCP: both reference dots live on the FIXED finger's body.
    # The fixed finger never travels; the moving one sweeps ~22 mm. Ref dots
    # parked one-per-finger therefore average to a point that slides by half
    # that travel every time the jaws move -- measured as a 21 mm aim error:
    # the arm aimed at the rod and the CLOSED jaws ended up 21 mm beside it.
    # Pinning both dots to the fixed finger makes pad_center() a constant, so
    # move_to() aims where the jaws will actually meet.
    gf = gid("fixed_jaw_pad")
    bf = tmp.geom_bodyid[gf]
    body_f = next(bb for bb in spec.bodies
                  if bb.name == mujoco.mj_id2name(
                      tmp, mujoco.mjtObj.mjOBJ_BODY, bf))
    p_f = td.xpos[bf].copy()
    R_f = td.xmat[bf].reshape(3, 3)
    for k, (usgn, O) in enumerate(mouths):
        ref = body_f.add_geom()
        ref.name = "grasp_ref_%d" % k
        ref.type = mujoco.mjtGeom.mjGEOM_SPHERE
        ref.size = [0.0008, 0.0, 0.0]
        ref.pos = R_f.T @ (O - p_f)
        ref.rgba = [0.1, 0.1, 0.1, 0.0]
        ref.contype, ref.conaffinity = 0, 0

    for pad_name in ("fixed_jaw_pad", "moving_jaw_pad"):
        spec.delete(spec.geom(pad_name))
    print("    v_jaw: both pads -> 90 deg V slot, depth %.0f mm" % (V_DEPTH * 1000))


def _lift_workpiece(spec):
    """Raise the rod and put a V cradle under each end."""
    b = next(bb for bb in spec.bodies if bb.name == "shaft")
    b.pos = [b.pos[0], b.pos[1], b.pos[2] + LIFT_DZ]
    # _v_plates takes the slot MOUTH as its origin, not the apex.
    mouth_z = b.pos[2] - (np.sqrt(2.0) - 1.0) * CRADLE_DEPTH
    up = np.array([0.0, 0.0, 1.0])
    v_ax = np.array([1.0, 0.0, 0.0])
    w_ax = np.array([0.0, 1.0, 0.0])
    for sx in (-1.0, +1.0):
        O = np.array([b.pos[0] + sx * (F.SHAFT_L / 2 - CRADLE_HALF_X),
                      b.pos[1], mouth_z])
        for k, (c, R, hs) in enumerate(
                _v_plates(O, up, w_ax, v_ax, depth=CRADLE_DEPTH, bar=CRADLE_BAR,
                          length=CRADLE_HALF_X * 2)):
            g = spec.worldbody.add_geom()
            g.name = "cradle_%s%d" % ("L" if sx < 0 else "R", k + 1)
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = hs
            g.pos = c
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
            g.quat = q
            g.rgba = [0.46, 0.48, 0.52, 1.0]
            g.contype, g.conaffinity = F.CT_STATIC
    print("    cradle: rod lifted %.0f mm, V under both ends" % (LIFT_DZ * 1000))


def build_model(grip_fix=None, v_jaw=False, lift=False, arc_jaw=False):
    """Feed scene: no vise, no slot platform, PLAIN cylinder workpiece.

    Requested 2026-11-05: "把虎钳和浅平台删掉，把工件换成圆柱".
    Removing the milled flat and the slot platform means nothing stops the
    cylinder rolling, so the grasp now has to work on the real pads alone.
    grip_fix selects an optional gripper patch (see add_grip_fix):
        None -> stock pads
        "v" / "cradle" / "both"
    """
    spec = F.build_scene(with_lathe=True, lathe_origin=LATHE_ORIGIN,
                         with_vise=False, with_fixture=False,
                         with_shaft_flat=False)
    if lift:
        _lift_workpiece(spec)
    if arc_jaw:
        import arc_jaws
        arc_jaws.install(spec)
    elif v_jaw:
        _install_v_jaws(spec)
    if grip_fix:
        add_grip_fix(spec, grip_fix)
    return spec.compile()


FRAME_EVERY = 60         # 每 N 步采一帧；仅 frame_hook 非 None 时生效
GRIP_SETTLE = 120        # 每次 move3 前先把夹爪指令落实的步数


def main(frame_hook=None, model=None, data=None, carry_lock=True,
         v_jaw=False, lift=False, wrist=None):
    """Run the feed sequence.

    frame_hook(tag, model, data) is an OPTIONAL visualisation tap.
    tag is a stage label (or None for in-between frames). It is never
    called when frame_hook is None, so verify_all.py sees the exact same
    numerical run as before. See show_current.py / show_live.py.

    model / data may be injected so an external viewer can watch the very
    same MjData being stepped. Defaults build fresh ones.

    carry_lock=False drops the kinematic lock during the carry segment and
    lets the grasp carry the load on its own. Default True keeps the
    numbers that verify_all.py asserts.
    """
    # Flat pads tolerate any wrist angle; a V groove only closes properly
    # when the jaw axis is square to the rod. Keep the stock value for the
    # verified 7/7 path and switch automatically for the V-jaw experiment.
    if wrist is None:
        wrist = V_WRIST if v_jaw else WRIST_ROLL_GRASP
    grip_close = V_GRIP_CLOSE if v_jaw else GRIP_CLOSE
    model = model if model is not None else build_model(v_jaw=v_jaw, lift=lift)
    data = data if data is not None else mujoco.MjData(model)

    def tick(tag=None):
        if frame_hook is not None:
            frame_hook(tag, model, data)

    def gid(kind, name):
        return mujoco.mj_name2id(model, kind, name)

    jaw_act = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
    shaft_body = gid(mujoco.mjtObj.mjOBJ_BODY, "shaft")
    shaft_joint = gid(mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
    sq = model.jnt_qposadr[shaft_joint]
    sv = model.jnt_dofadr[shaft_joint]
    # Depth measured: -0.030 (shaft centred in the jaws) lets the jaws
    # shove it 16.9 mm off axis and then spit it out; -0.008 puts the
    # tail 40 mm outside the jaws and it sags 10.6 deg; -0.016 is the
    # balance that centres and holds it.
    axis = np.array([LATHE_ORIGIN[0] - 0.016,
                     LATHE_ORIGIN[1], LATHE_ORIGIN[2]])
    shaft_geoms = [
        i for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        .startswith("shaft_")
    ]

    def shaft():
        return data.qpos[sq:sq + 3].copy()

    def coax():
        p = shaft()
        return float(np.hypot(p[1] - axis[1], p[2] - axis[2]) * 1000)

    def axdev():
        """Angle between the ROD AXIS and world x -- not the quaternion.

        A round part that merely rolls in the V groove reports tens of degrees
        of "tilt" from tilt_deg() while its axis never moved, which sent two
        earlier debugging rounds after a non-existent problem.
        """
        ax = data.xmat[shaft_body].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(min(1.0, abs(float(ax[0]))))))

    def jaw_state():
        contacts, force = 0, 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if not any(g in (c.geom1, c.geom2) for g in shaft_geoms):
                continue
            n1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if "jaw_geom" not in (n1 + "|" + n2):
                continue
            contacts += 1
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            force += abs(f6[0])
        return contacts, force

    arm = Arm(model, data)
    state = {"offset": None, "off_site": None, "abs_pos": None,
             "lock_active": True}

    def lock():
        """Kinematic carry: assume a perfect gripper, keep the shaft axial.

        Two reference frames, because the fingers change the answer:
        - carry: pad_center - offset. The robot IS holding the shaft, and
          both move together, so the pad centre is the natural anchor.
        - clamp: gripperframe site + off_site. Opening the fingers moves
          pad_center, and anchoring there would drag the shaft sideways
          with it (measured: coax 1.05 -> 27.95 mm the moment the robot
          let go). The site does not move when the fingers open.
        """
        if state["abs_pos"] is not None:
            # Hand-over mode: the shaft is pinned in WORLD space while the
            # arm backs away, so the arm cannot drag it.
            data.qpos[sq:sq + 3] = state["abs_pos"]
        elif state["off_site"] is not None:
            data.qpos[sq:sq + 3] = data.site_xpos[arm.site] + state["off_site"]
        else:
            data.qpos[sq:sq + 3] = arm.pad_center() - state["offset"]
        data.qpos[sq + 3:sq + 7] = SHAFT_Q0
        data.qvel[sv:sv + 6] = 0.0

    def move(target, ns, grip, locked, jaw_ctrl):
        """Arm.move_to with a per-step hook for the carry lock.

        Also takes jaw_ctrl explicitly: writing CHUCK_OPEN_CTRL on every
        step of the retreat used to open the chuck and drop the shaft.
        """
        data.qpos[arm.qadr[4]] = wrist
        data.ctrl[4] = wrist
        mujoco.mj_forward(model, data)
        off = arm.pad_offset()
        q_goal, _ = arm.ik(np.asarray(target) - off,
                           lock_wrist_roll=wrist)
        q_start = np.array([data.qpos[a] for a in arm.qadr])
        for i in range(ns):
            frac = (i + 1) / ns
            s = frac * frac * (3 - 2 * frac)
            for a, v in zip(arm.qadr, q_start + s * (q_goal - q_start)):
                data.ctrl[a] = v
            data.ctrl[arm.grip_act] = grip
            data.ctrl[jaw_act] = jaw_ctrl
            mujoco.mj_step(model, data)
            if locked:
                lock()
            if frame_hook is not None and i % FRAME_EVERY == 0:
                tick()
        for j in range(260):
            for a, v in zip(arm.qadr, q_goal):
                data.ctrl[a] = v
            data.ctrl[arm.grip_act] = grip
            data.ctrl[jaw_act] = jaw_ctrl
            mujoco.mj_step(model, data)
            if locked:
                lock()
            if frame_hook is not None and j % (FRAME_EVERY * 2) == 0:
                tick()
        return float(np.linalg.norm(arm.pad_center() - np.asarray(target)))

    def move3(target, ns, grip, locked,
              jaw_ctrl=SM.CHUCK_OPEN_CTRL, iters=8, probe=None,
              probe_goal=None):
        """Iterate until the pads actually reach the goal -- by MOVING THE GOAL.

        Re-sending the same absolute target is useless under load: the
        position servos settle at the same droop every time. Measured this
        session, carrying the 197 g shaft toward the chuck:
        aim (184,120,60) -> pads stop 24.8 mm short; the retry loop did
        fire (3x) and landed on the same point, so the shaft arrived at
        z = 94.7 mm instead of 60 mm and never entered the jaws.

        So each round adds (goal - measured) to the commanded target.
        That is plain integral compensation, not a physical shortcut --
        the arm still has to hold the load, it just aims higher.
        """
        goal = np.asarray(target, dtype=float)
        tgt = goal.copy()

        # Apply the grip command BEFORE the convergence check.
        # "夹紧" targets the same point as the preceding "下降", so err was
        # already ~0 and the early break below returned without ever writing
        # GRIP_CLOSE into ctrl: the fingers stayed at 0.500 (open), the pads
        # sat 19-23 mm off the axis, and the shaft never left the table.
        # Measured before this fix: ncon had zero pad-shaft contacts.
        for _ in range(GRIP_SETTLE):
            data.ctrl[arm.grip_act] = grip
            data.ctrl[jaw_act] = jaw_ctrl
            mujoco.mj_step(model, data)
            if locked:
                lock()

        # probe: measure the PART instead of the pads. The pads can sit on the
        # chuck axis while the rod hangs 4 mm off it, because the grip point
        # is not the rod's centre; servoing the pads then cannot fix the part.
        # Measured before this: pads on axis (IK residual 1.2 mm) but the rod
        # arrived at coax 3.94 mm and only 4 of 6 jaw contacts closed.
        meas = probe if probe is not None else arm.pad_center
        # probe_goal is what the PROBED point should reach. The pad command
        # carries a constant pad-to-part offset, so comparing the part against
        # the pad-space goal double-counts that offset and drives the arm to
        # the wrong place (measured: this alone broke the flat-pad 7/7 run).
        gmeas = goal if probe_goal is None else np.asarray(probe_goal, float)
        err = float(np.linalg.norm(meas() - gmeas))
        for _ in range(iters):
            if err < 0.003:
                break
            move(tgt, max(90, ns // 3), grip, locked, jaw_ctrl)
            err = float(np.linalg.norm(meas() - gmeas))
            # 0.8 damping: full step overshoots once the load unloads
            tgt = tgt + 0.8 * (gmeas - meas())
        return err

    def settle(n, jaw_ctrl, grip, locked=False):
        for i in range(n):
            data.ctrl[jaw_act] = jaw_ctrl
            data.ctrl[arm.grip_act] = grip
            mujoco.mj_step(model, data)
            if locked:
                lock()
            if frame_hook is not None and i % (FRAME_EVERY * 2) == 0:
                tick()

    # ---------------- reset / home ----------------
    mujoco.mj_resetData(model, data)
    data.qpos[arm.qadr[4]] = wrist
    data.ctrl[4] = wrist
    data.ctrl[arm.grip_act] = GRIP_OPEN
    data.ctrl[jaw_act] = SM.CHUCK_OPEN_CTRL
    for _ in range(600):
        for k in range(5):
            data.ctrl[k] = data.qpos[arm.qadr[k]]
        mujoco.mj_step(model, data)

    tick("① 起始｜卡盘张开，工件在台面")
    home = shaft()
    R = F.SHAFT_D / 2
    g = home
    print("=" * 74)
    print("机械臂送料到卡盘（合并场景；搬运段为运动学锁，见文件头）")
    print("=" * 74)
    print("  工件起点      : %s" % np.round(home, 4))
    print("  卡盘轴线      : %s" % np.round(axis, 4))
    print()

    # ---------------- real-physics grasp ----------------
    pick = [
        ("接近", [g[0], g[1], g[2] + R + 0.075], GRIP_OPEN, 550),
        ("下降", [g[0], g[1], g[2] + 0.002], GRIP_OPEN, 400),
        ("夹紧", [g[0], g[1], g[2] + 0.002], grip_close, 320),
    ]
    grasp_tilt = None
    for name, tgt, grip, ns in pick:
        move3(np.array(tgt), ns, grip, False)
        grasp_tilt = tilt_deg(data.qpos[sq + 3:sq + 7])
        print("  %-6s 轴=%s 轴向偏=%5.1f deg  pad-轴=%5.1f mm"
              % (name, np.round(shaft(), 4), axdev(),
                 np.linalg.norm(arm.pad_center() - shaft()) * 1000))
        if name == "夹紧":
            state["offset"] = arm.pad_center() - shaft()
        tick("② 接近｜下降中" if name == "接近" else
             "③ 下降｜pad 到轴心" if name == "下降" else
             "④ 夹紧｜真实抓取（轴会歪）")
    print("  shaft_in_grip = %s mm（搬运锁的相对偏移）"
          % np.round(state["offset"] * 1000, 2))
    if v_jaw and not carry_lock:
        print("  V 槽形封闭抓取：轴线偏离 %.1f deg，搬运全靠真实接触（无锁）"
              % axdev())
    elif v_jaw:
        print("  V 槽形封闭抓取：轴线偏离 %.1f deg" % axdev())
    else:
        print("  ⚠ 平板爪夹圆柱会翻（四元数倾角 %.1f deg）：搬运由运动学锁接管"
              % grasp_tilt)

    # ---------------- kinematic carry ----------------
    carry = [
        ("对中", [g[0], g[1], g[2] + 0.020], 260),
        ("抬起", [g[0], g[1], g[2] + 0.130], 420),
        ("高位横移", [axis[0], axis[1], 0.16], 900),
        ("下到轴线", [axis[0], axis[1], axis[2]], 700),
    ]
    for name, tgt, ns in carry:
        # The last leg is the one that decides whether the rod enters the
        # jaws, so close the loop on the ROD there, not on the pads.
        close_on_part = (name == "下到轴线") and not carry_lock
        err = move3(np.array(tgt) + state["offset"], ns,
                    grip_close, carry_lock,
                    probe=shaft if close_on_part else None,
                    probe_goal=np.array(tgt) if close_on_part else None)
        print("  %-6s IK残差=%5.1f mm 轴=%s 同轴=%6.2f mm 倾斜=%4.1f deg"
              % (name, err * 1000, np.round(shaft(), 4), coax(),
                 tilt_deg(data.qpos[sq + 3:sq + 7])))
        tick({"对中": "⑤ 对中｜搬运锁接管",
              "抬起": "⑥ 抬起",
              "高位横移": "⑦ 高位横移｜到卡盘上方",
              "下到轴线": "⑧ 下到轴线｜送进卡盘"}[name])

    # ---------------- clamp: align friction solver with the chuck scene ----
    # scene_full.py sets no spec.option. Applying these from the start
    # breaks the tuned grasp, so patch only the friction-relevant fields
    # now, once the shaft is already at the chuck.
    model.opt.impratio = 5.0
    model.opt.noslip_iterations = 10
    model.opt.noslip_tolerance = 0.0

    # Hand the shaft over BEFORE the jaws close. With GRIP_CLOSE the
    # robot keeps holding the same shaft the jaws are closing on, the two
    # clamps fight, and the losing servo drags everything 6.46 mm off axis
    # (measured). Release the fingers first, re-anchor the lock on the
    # gripperframe site, then let the jaws close on a shaft that nothing
    # else is pulling.
    # Hand-over, in this exact order, because the earlier orders failed:
    # 1) pin the shaft in world space (nothing can drag it now)
    # 2) back the arm away, so the pads stop touching the shaft
    # 3) let the jaws close on a shaft that has no other force on it
    # With GRIP_CLOSE the robot and the chuck fought for the same shaft and
    # the losing servo dragged the pads 6.46 mm off axis; with GRIP_OPEN but
    # the lock anchored on the gripperframe site, the jaws pushed the shaft
    # into the still-open pads, the arm gave way 12.07 mm and carried the
    # shaft with it (both measured).
    state["abs_pos"] = shaft().copy()
    settle(700, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN, locked=True)
    move3(np.array([axis[0] + 0.055, axis[1], axis[2] + 0.015]),
          200, GRIP_OPEN, True, SM.CHUCK_CLAMP_CTRL)
    print("  卡爪夹紧(锁) 轴=%s 同轴=%5.2f mm 接触=%d"
          % (np.round(shaft(), 4), coax(), jaw_state()[0]))
    tick("⑨ 三爪夹紧｜单丝杠联动")

    # Hand the shaft back to PHYSICS before any load test.
    state["lock_active"] = False
    settle(1600, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)
    print("  解锁静置     轴=%s 同轴=%5.2f mm 倾斜=%5.1f deg 接触=%d"
          % (np.round(shaft(), 4), coax(),
             tilt_deg(data.qpos[sq + 3:sq + 7]), jaw_state()[0]))
    tick("⑩ 解锁｜交还真实物理")

    # Retreat with the chuck STILL clamped (see evidence chain).
    move3(np.array([axis[0] - 0.08, axis[1] - 0.08, 0.15]),
          400, GRIP_OPEN, False, SM.CHUCK_CLAMP_CTRL)
    settle(800, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)

    final_coax = coax()
    final_tilt = tilt_deg(data.qpos[sq + 3:sq + 7])
    contacts, force = jaw_state()
    print("  松开退出     轴=%s 同轴=%5.2f mm 倾斜=%5.1f deg "
          "接触=%d 三爪法向力=%.1f N"
          % (np.round(shaft(), 4), final_coax, final_tilt,
             contacts, force))
    tick("⑪ 机器人退出｜装夹完成")

    z0 = shaft()[2]
    settle(800, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)
    slip = abs(z0 - shaft()[2]) * 1000
    x0 = shaft()[0]
    for i in range(700):
        data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
        data.xfrc_applied[shaft_body, 0] = -20.0
        mujoco.mj_step(model, data)
        if frame_hook is not None and i % (FRAME_EVERY * 2) == 0:
            tick()
    pull = abs(shaft()[0] - x0) * 1000
    tick("⑫ 20 N 拉拔｜夹得住")
    print("  重力下沉  %.2f mm      20 N 拉  %.2f mm" % (slip, pull))
    print()

    ok = (final_coax <= COAX_LIMIT and final_tilt <= TILT_LIMIT
          and contacts >= CONTACT_MIN and slip <= SLIP_LIMIT
          and pull <= PULL_LIMIT)
    print("=" * 74)
    print("FEED_METRIC coax_mm=%.2f tilt_deg=%.1f contacts=%d "
          "slip_mm=%.2f pull_mm=%.2f jaw_force_N=%.1f "
          "grasp_tilt_deg=%.1f carry=%s physics_after_clamp=%d"
          % (final_coax, final_tilt, contacts, slip, pull, force,
             grasp_tilt, "kinematic_lock" if carry_lock else "physics",
             0 if state["lock_active"] else 1))
    print("FEED_METRIC %s" % ("PASS" if ok else "FAIL"))
    if not ok:
        print("  未通过判据：同轴≤%.1fmm 倾斜≤%.1fdeg 接触≥%d "
              "下沉≤%.1fmm 20N拉≤%.1fmm"
              % (COAX_LIMIT, TILT_LIMIT, CONTACT_MIN,
                 SLIP_LIMIT, PULL_LIMIT))
    print("=" * 74)
    print("OK")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="机械臂送料到三爪卡盘")
    ap.add_argument("--v-jaw", action="store_true",
                    help="换成 90 度 V 槽爪（圆柱靠形封闭，不靠摩擦）")
    ap.add_argument("--lift", action="store_true",
                    help="工件先抬到 V 托架上，让爪能伸到下面")
    ap.add_argument("--no-lock", action="store_true",
                    help="搬运段不用运动学锁，让真实接触承受载荷")
    args = ap.parse_args()
    raise SystemExit(main(v_jaw=args.v_jaw, lift=args.lift,
                          carry_lock=not args.no_lock))
