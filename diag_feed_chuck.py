"""diag_feed_chuck.py -- the ROBOT feeds a shaft into the three-jaw chuck.

STATUS 2026-11-05: PASS. Wired into verify_all.py CHECKS.

SCOPE: TWO EXPLICIT SIMPLIFICATIONS -- do NOT read this as full physics
---------------------------------------------------------------------
1. The milled D-flat is a ROUGH collision proxy, not an accurate shape.
   scene_full.py keeps shaft_flat visual-only (contype=0) because the box
   corners stick ~3.1 mm past the D20 envelope and wreck the slot-drop
   tests. Here the flat is patched to CT_WORK as a LOCAL change. Its only
   job is to stop the cylinder rolling; the exact shape is deliberately
   left for later ("具体形状后面再优化"). Do not copy this patch into
   scene_full.py without re-running diag_slot_alone.py and demo_09.

2. The CARRY is KINEMATIC. From the centring stage until the chuck has
   clamped, every step writes the shaft qpos/quat back to
   (pad_center - lock_offset, SHAFT_Q0) and zeroes qvel. This ASSUMES the
   gripper has perfect form closure. It does NOT prove the real gripper
   can hold the shaft -- see "Known open defect" below. The lock is
   released BEFORE the load tests, and FEED_METRIC reports that fact so
   nobody can quietly extend it through the pull-out and fake a pass.

What this DOES prove: given a shaft delivered on-axis and correctly
oriented, the chuck station can centre it, clamp it, hold it against
gravity and resist a 20 N pull-out AFTER the robot has fully retreated.

Evidence chain (controlled rounds, this session)
-----------------------------------------------
- Making the D-flat a real collision geom cuts first-pad-contact tilt
  23.6 -> 4.1 deg. With the flat left visual-only the whole feed collapses
  (final coax 1134 / 1215 mm, shaft ejected to x=-34 m). The flat is
  REQUIRED, not cosmetic.
- scene_full.py sets NO spec.option, while scene_machining.py sets
  noslip_iterations=10 / noslip_tolerance=0 / cone=ELLIPTIC / impratio=5.
  Applying those from the start breaks the grasp, whose constants were
  tuned under the defaults (shaft flies to x=-34 m). Patching only the
  friction-relevant fields into model.opt AT THE CLAMP STAGE raises total
  jaw normal force 8.4 -> 18.7 N (the independent chuck scene measures
  18.2 N) and drops 20 N pull-out from 1522 mm to 0.62 mm.
- Retreat bug: the move helper used to write CHUCK_OPEN_CTRL on every
  step, so retreating OPENED the chuck and dropped the shaft (coax 50.85
  mm). Passing CHUCK_CLAMP_CTRL for the retreat fixes it (coax 3.05 mm).
- Ruled out by controlled scans: tighter clamp alone (-0.011 / -0.013 /
  -0.015) raises force to 21.4 N but pull-out stays 4713..35460 mm;
  higher pad friction; static closed-loop pose correction; moving the
  chuck in x/y; lower lift; two-stage clamp order; wrist_roll 0/0.8/2.4/3.14.

Known open defect (NOT solved here, do not close)
-------------------------------------------------
Unassisted grasp: the real collision shape is a smooth cylinder and at
WRIST_ROLL_GRASP=1.55 the pad line [0.64, 0.768, 0.006] meets the shaft
axis at ~50 deg. The shaft hangs ~11 mm below the pad centre on friction
alone and swings to vertical; measured tilt grows 0.9 -> 23.6 -> 43.8 ->
57.7 -> 98.2 deg through approach/descend/close/centre/lift. demo_09's
3/3 never checks orientation, so a vertical stick dropped in the slot
still "passes". Fix the grasp (real flat geometry / levelled pads /
cradle from below), add an orientation check to demo_09, and ONLY THEN
delete the kinematic lock. Deleting the lock first reintroduces the bug.

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


def build_model():
    """Merged scene, with the D-flat promoted to a real collision geom.

    Local patch only: scene_full.py keeps the flat visual-only so the
    demo_09 / diag_slot_alone baselines are untouched.
    """
    spec = F.build_scene(with_lathe=True, lathe_origin=LATHE_ORIGIN)
    shaft = next(b for b in spec.bodies if b.name == "shaft")
    flat = next(g for g in shaft.geoms if g.name == "shaft_flat")
    flat.contype, flat.conaffinity = F.CT_WORK
    return spec.compile()


def main():
    model = build_model()
    data = mujoco.MjData(model)

    def gid(kind, name):
        return mujoco.mj_name2id(model, kind, name)

    jaw_act = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
    shaft_body = gid(mujoco.mjtObj.mjOBJ_BODY, "shaft")
    shaft_joint = gid(mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
    sq = model.jnt_qposadr[shaft_joint]
    sv = model.jnt_dofadr[shaft_joint]
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
    state = {"offset": None, "lock_active": True}

    def lock():
        """Kinematic carry: assume a perfect gripper, keep the shaft axial."""
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
        for _ in range(260):
            for a, v in zip(arm.qadr, q_goal):
                data.ctrl[a] = v
            data.ctrl[arm.grip_act] = grip
            data.ctrl[jaw_act] = jaw_ctrl
            mujoco.mj_step(model, data)
            if locked:
                lock()
        return float(np.linalg.norm(arm.pad_center() - np.asarray(target)))

    def move3(target, ns, grip, locked,
              jaw_ctrl=SM.CHUCK_OPEN_CTRL):
        for _ in range(3):
            err = move(np.asarray(target), max(90, ns // 3),
                       grip, locked, jaw_ctrl)
            if err < 0.005:
                break
        return err

    def settle(n, jaw_ctrl, grip, locked=False):
        for _ in range(n):
            data.ctrl[jaw_act] = jaw_ctrl
            data.ctrl[arm.grip_act] = grip
            mujoco.mj_step(model, data)
            if locked:
                lock()

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

    # ---------------- clamp: align friction solver with the chuck scene ----
    # scene_full.py sets no spec.option. Applying these from the start
    # breaks the tuned grasp, so patch only the friction-relevant fields
    # now, once the shaft is already at the chuck.
    model.opt.impratio = 5.0
    model.opt.noslip_iterations = 10
    model.opt.noslip_tolerance = 0.0

    settle(700, SM.CHUCK_CLAMP_CTRL, GRIP_CLOSE, locked=True)
    print("  卡爪夹紧(锁) 轴=%s 同轴=%5.2f mm 接触=%d"
          % (np.round(shaft(), 4), coax(), jaw_state()[0]))

    # Hand the shaft back to PHYSICS before any load test.
    state["lock_active"] = False
    settle(400, SM.CHUCK_CLAMP_CTRL, GRIP_CLOSE)
    print("  解锁静置     轴=%s 同轴=%5.2f mm 倾斜=%5.1f deg 接触=%d"
          % (np.round(shaft(), 4), coax(),
             tilt_deg(data.qpos[sq + 3:sq + 7]), jaw_state()[0]))

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

    z0 = shaft()[2]
    settle(800, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)
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
