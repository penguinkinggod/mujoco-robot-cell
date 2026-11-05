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


def build_model(grip_fix=None):
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
    if grip_fix:
        add_grip_fix(spec, grip_fix)
    return spec.compile()


FRAME_EVERY = 60         # 每 N 步采一帧；仅 frame_hook 非 None 时生效


def main(frame_hook=None, model=None, data=None):
    """Run the feed sequence.

    frame_hook(tag, model, data) is an OPTIONAL visualisation tap.
    tag is a stage label (or None for in-between frames). It is never
    called when frame_hook is None, so verify_all.py sees the exact same
    numerical run as before. See show_current.py / show_live.py.

    model / data may be injected so an external viewer can watch the very
    same MjData being stepped. Defaults build fresh ones.
    """
    model = model if model is not None else build_model()
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
        data.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
        data.ctrl[4] = WRIST_ROLL_GRASP
        mujoco.mj_forward(model, data)
        off = arm.pad_offset()
        q_goal, _ = arm.ik(np.asarray(target) - off,
                           lock_wrist_roll=WRIST_ROLL_GRASP)
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
              jaw_ctrl=SM.CHUCK_OPEN_CTRL, iters=8):
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
        err = float(np.linalg.norm(arm.pad_center() - goal))
        for _ in range(iters):
            if err < 0.003:
                break
            move(tgt, max(90, ns // 3), grip, locked, jaw_ctrl)
            err = float(np.linalg.norm(arm.pad_center() - goal))
            # 0.8 damping: full step overshoots once the load unloads
            tgt = tgt + 0.8 * (goal - arm.pad_center())
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
    data.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    data.ctrl[4] = WRIST_ROLL_GRASP
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
        ("夹紧", [g[0], g[1], g[2] + 0.002], GRIP_CLOSE, 320),
    ]
    grasp_tilt = None
    for name, tgt, grip, ns in pick:
        move3(np.array(tgt), ns, grip, False)
        grasp_tilt = tilt_deg(data.qpos[sq + 3:sq + 7])
        print("  %-6s 轴=%s 倾斜=%5.1f deg  pad-轴=%5.1f mm"
              % (name, np.round(shaft(), 4), grasp_tilt,
                 np.linalg.norm(arm.pad_center() - shaft()) * 1000))
        if name == "夹紧":
            state["offset"] = arm.pad_center() - shaft()
        tick("② 接近｜下降中" if name == "接近" else
             "③ 下降｜pad 到轴心" if name == "下降" else
             "④ 夹紧｜真实抓取（轴会歪）")
    print("  shaft_in_grip = %s mm（搬运锁的相对偏移）"
          % np.round(state["offset"] * 1000, 2))
    print("  ⚠ 真实抓取倾斜 %.1f deg：夹爪形封闭尚未解决，此处由搬运锁接管"
          % grasp_tilt)

    # ---------------- kinematic carry ----------------
    carry = [
        ("对中", [g[0], g[1], g[2] + 0.020], 260),
        ("抬起", [g[0], g[1], g[2] + 0.130], 420),
        ("高位横移", [axis[0], axis[1], 0.16], 900),
        ("下到轴线", [axis[0], axis[1], axis[2]], 700),
    ]
    for name, tgt, ns in carry:
        err = move3(np.array(tgt) + state["offset"], ns,
                    GRIP_CLOSE, True)
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
          "grasp_tilt_deg=%.1f carry=kinematic_lock physics_after_clamp=%d"
          % (final_coax, final_tilt, contacts, slip, pull, force,
             grasp_tilt, 0 if state["lock_active"] else 1))
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
    raise SystemExit(main())
