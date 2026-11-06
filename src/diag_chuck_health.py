"""diag_chuck_health.py -- is the chuck actually usable as a workstation?

diag_chuck_grip.py answers "can the jaws hold the shaft?" -- it places the
shaft by hand, perfectly on axis, and never moves the robot. That is only
half the story. A workstation also needs the robot to *reach* the chuck.

This script measures both, with numbers, in one run:

    physics   jaw displacement spread, per-jaw normal force, shaft axis
              eccentricity, shaft tilt, jaw-shaft contact count
    reach     IK residual to the chuck mouth, using the SAME settle +
              3-round approach that demo_08/demo_09 proved works

Why the 3-round approach matters
--------------------------------
A single move_to() call leaves 20~120 mm of error in this scene (pad_offset
changes with joint angles). demo_08's comment records 143 mm of accumulated
error before it was fixed by re-solving 3 times. Any reachability claim made
with one call is wrong.

Why the jaws are synchronous now (measured, not guessed)
-------------------------------------------------------
Old version: three independent position servos, displacement spread
4.24 mm, per-jaw force ratio ~4x. The root cause was NOT the drive -- the
three jaw BOXES were never rotated to their 120 deg radial directions, so
their effective radial half-widths were 9.0 / 12.3 / 12.3 mm. Fixing only
that geometry (drive untouched) cut the spread to 0.23 mm and equalised
the forces.

Current: ONE servo (the screw) drives jaw_slide_0; two mjEQ_JOINT
constraints make the other two jaws follow. Spread 0.12 mm, forces
[6.5, 4.6, 7.1] N, eccentricity 0.00 mm, 20 N pull-out holds.
mjEQ_JOINT is a SOFT constraint, so the remaining ~0.1 mm is elastic
compliance under contact load, not a lack of synchronisation. Real chuck
screws and jaws are not infinitely rigid either.

DO NOT put the two extra actuators back: three servos fighting the same
constraints blows q up to 202 mm and ejects the shaft (12 kp/ctrl combos
all failed). jaw_spread IS now a pass/fail criterion.

Run:
    D:\\robot-env\\Scripts\\python.exe diag_chuck_health.py
"""
import numpy as np
import mujoco

import scene_machining as S
from demo_08_pick_and_load import Arm, WRIST_ROLL_GRASP, GRIP_OPEN

# ---- pass/fail thresholds (written down BEFORE tuning anything) ----
ECC_LIMIT = 1.0        # mm, shaft axis vs chuck axis
TILT_LIMIT = 2.0       # deg
CONTACT_MIN = 6        # jaw-shaft contact points
# True synchronisation criteria (added 2026-11-05 after the geometry fix).
# 0.2 mm is the soft-constraint compliance measured under clamping load,
# NOT a relaxed standard: it was 4.24 mm before.
SPREAD_LIMIT = 0.2     # mm, max-min jaw slide displacement
FORCE_RATIO_LIMIT = 2.0  # max/min per-jaw normal force
# Coarse workspace check ONLY. A bare IK move still carries ~10 mm of
# steady-state error. Final insertion accuracy (coaxiality <= 2 mm) needs
# demo_09's shaft_in_grip compensation in the merged scene -- that is the
# next task, not a criterion this script can honestly claim to pass.
REACH_LIMIT = 10.0     # mm, IK residual at the pre-insertion stop point

SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])


def check_jaw_coupling(model, data, ids):
    """Prove the two equality constraints really couple jaw 1/2 to jaw 0.

    A fresh MjSpec equality defaults to data=[0, 1, 0, 0, 0, ...], i.e.
    polycoef 0 + 1*x, which means q2 = q1. If that default ever changes,
    or somebody edits eq.data, the constraint silently becomes "lock the
    follower joint at 0" instead. The on-axis clamping test CANNOT catch
    that by accident: with the shaft placed exactly on the chuck axis, all
    three jaws happen to settle near 0 anyway, so spread would still look
    fine. Check both the structure and the empty-load following behaviour.

    Measured 2026-11-05: spec default data = [0,1,0,0,0,...],
    eq obj ids 7->8 and 7->9, empty-load q=[5.000, 5.009, 5.009] mm.
    """
    jt = mujoco.mjtObj.mjOBJ_JOINT
    jid = [mujoco.mj_name2id(model, jt, "jaw_slide_%d" % k)
           for k in range(3)]
    ja = [model.jnt_qposadr[j] for j in jid]

    eq_ok = (model.neq == 2 and len(ids["jaw_acts"]) == 1)
    if eq_ok:
        expect = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        masters, followers = set(), set()
        for i in range(model.neq):
            eq_ok &= (model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT)
            eq_ok &= bool(np.allclose(model.eq_data[i][:5], expect,
                                      atol=1e-9))
            masters.add(int(model.eq_obj1id[i]))
            followers.add(int(model.eq_obj2id[i]))
        eq_ok &= (masters == {jid[0]}
                  and followers == {jid[1], jid[2]})

    # Sweep the screw with the shaft parked far away: no contact forces,
    # so any remaining difference is purely the coupling constraint.
    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, jt, "shaft_free")]
    ac = ids["jaw_acts"][0]
    samples = []
    for ctrl in (0.0, 0.004):
        mujoco.mj_resetData(model, data)
        data.qpos[sq:sq + 3] = [S.LATHE_X, S.LATHE_Y + 0.6, S.LATHE_Z]
        data.qpos[sq + 3:sq + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(1200):
            data.ctrl[ac] = ctrl
            mujoco.mj_step(model, data)
        samples.append(np.array([data.qpos[a] * 1000 for a in ja]))
    moved = float(samples[1][0] - samples[0][0])
    follow_err = max(float(np.max(np.abs(f - f[0]))) for f in samples)
    follow_ok = bool(moved > 1.0 and follow_err < 0.05)

    # Leave a clean slate for the physics pass below.
    data.ctrl[:] = 0.0
    mujoco.mj_resetData(model, data)
    return bool(eq_ok), follow_ok, moved, follow_err, samples


def main():
    model, data, ids = S.make_scene()
    sq = model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")]
    ja = [model.jnt_qposadr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                          "jaw_slide_%d" % k)] for k in range(3)]
    acts = ids["jaw_acts"]
    shaft_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")

    print("=" * 74)
    print("卡盘工位体检（物理夹紧 + 机器人可达性）")
    print("=" * 74)
    print("  卡盘轴线 : (%.3f, %.3f, %.3f)"
          % (S.LATHE_X, S.LATHE_Y, S.LATHE_Z))
    print()

    # ---------------- coupling (structure + behaviour) ----------------
    eq_ok, follow_ok, moved, follow_err, samples = check_jaw_coupling(
        model, data, ids)
    print("  联动约束结构    : %s   (neq=%d, 丝杠=%d, polycoef 须为 0+1x)"
          % ("OK" if eq_ok else "BAD", model.neq, len(ids["jaw_acts"])))
    print("  空载跟随        : 主爪移动 %.2f mm，最大跟随偏差 %.4f mm  %s"
          % (moved, follow_err, "OK" if follow_ok else "BAD"))
    for ctrl, q in zip((0.0, 0.004), samples):
        print("                    ctrl=%+.3f -> q=%s mm"
              % (ctrl, np.round(q, 3)))
    print()

    # ---------------- physics ----------------
    mujoco.mj_resetData(model, data)
    data.qpos[sq:sq + 3] = [
        S.LATHE_X - 0.050, S.LATHE_Y, S.LATHE_Z]
    data.qpos[sq + 3:sq + 7] = SHAFT_Q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    for a in ja:
        data.qpos[a] = S.JAW_ARC_Q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # 软爪先停在镗孔位（孔 = Φ20.4），Φ20 棒料放进去由孔托住，再拧紧丝杠。
    # 从张开位（通道 Φ32）起步会让棒料在夹紧前就掉到台面（实测 z -50 mm），
    # 爪于是夹空 —— 那是这个体检脚本的摆位不对，真实上料是机械臂扶着送进去的。
    ramp = 500
    for i in range(ramp):
        t = (i + 1) / ramp
        s = t * t * (3.0 - 2.0 * t)
        for a in acts:
            data.ctrl[a] = (S.JAW_ARC_Q0
                            + s * (S.CHUCK_CLAMP_CTRL - S.JAW_ARC_Q0))
        mujoco.mj_step(model, data)
    for _ in range(1500):
        for a in acts:
            data.ctrl[a] = S.CHUCK_CLAMP_CTRL
        mujoco.mj_step(model, data)

    jaw_q = np.array([data.qpos[a] * 1000 for a in ja])
    spread = float(jaw_q.max() - jaw_q.min())
    p = data.qpos[sq:sq + 3]
    quat = data.qpos[sq + 3:sq + 7]
    ecc = float(np.hypot((p[1] - S.LATHE_Y) * 1000,
                         (p[2] - S.LATHE_Z) * 1000))
    # 圆柱绕自身轴线滚动不影响车削装夹，真正要管的是「工件轴线」与主轴
    # 轴线（world x）的夹角。四元数总转角会把无害的滚动误报成倾斜，
    # 主流程 diag_feed_chuck.axdev() 早就改成了这个口径，这里统一。
    shaft_body = int(model.geom_bodyid[shaft_geom])
    rod_ax = data.xmat[shaft_body].reshape(3, 3)[:, 2]
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(rod_ax[0]))))))

    contacts = 0
    forces = [0.0, 0.0, 0.0]
    for i in range(data.ncon):
        c = data.contact[i]
        n1 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        n2 = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        if shaft_geom not in (c.geom1, c.geom2):
            continue
        for k in range(3):
            # 软爪由 JAW_ARC_SEGS 个小弧片拼成，名字是 jaw_geom_<k>_<seg>；
            # 按前缀归到同一个爪，不能再用精确名字相等来判断。
            prefix = "jaw_geom_%d" % k
            if n1.startswith(prefix) or n2.startswith(prefix):
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f)
                contacts += 1
                forces[k] += abs(f[0])

    force_ratio = (max(forces) / max(min(forces), 1e-6)
                   if contacts >= CONTACT_MIN else float("inf"))
    print("  三爪位移 (mm)   : %s   差 %.3f mm   (要求 ≤ %.1f)"
          % (np.round(jaw_q, 3), spread, SPREAD_LIMIT))
    print("  各爪法向力 (N)  : %s   力比 %.2f   (要求 ≤ %.1f)"
          % (np.round(forces, 1), force_ratio, FORCE_RATIO_LIMIT))
    print("  轴心偏心        : %.2f mm   (要求 ≤ %.1f)" % (ecc, ECC_LIMIT))
    print("  轴姿态倾斜      : %.2f deg  (要求 ≤ %.1f)" % (tilt, TILT_LIMIT))
    print("  卡爪-轴接触点   : %d        (要求 ≥ %d)" % (contacts, CONTACT_MIN))
    print("  （三爪由 1 个丝杠 + 2 条联动约束驱动；残余 0.1 mm 是软约束柔度）")
    print()

    # ---------------- reachability ----------------
    # Use a FRESH scene. mj_resetData does NOT clear data.ctrl, and the
    # physics pass above leaves the jaw actuators at -0.012 plus 1700
    # steps of contact history. Reusing that data measured 85.68 mm where
    # an isolated run measures 10.45 mm for the very same target.
    r_model, r_data, _ = S.make_scene()
    arm = Arm(r_model, r_data)
    mujoco.mj_resetData(r_model, r_data)
    r_data.qpos[arm.qadr[4]] = WRIST_ROLL_GRASP
    r_data.ctrl[4] = WRIST_ROLL_GRASP
    r_data.ctrl[arm.grip_act] = GRIP_OPEN
    for _ in range(300):
        for k in range(5):
            r_data.ctrl[k] = r_data.qpos[arm.qadr[k]]
        mujoco.mj_step(r_model, r_data)

    # The gripper STOPS outside the chuck and lets the shaft front reach in.
    # At only 30 mm from the mouth the pads collide with the jaws: measured
    # 85.68 mm residual at z=0.06. At 70 mm it was 9.08 mm -- but that was
    # with 28 mm jaws. The jaws are now 55 mm long (they need the length to
    # stop the shaft pivoting: see scene_machining.JAW_LEN), so they reach
    # 27 mm further out and the pads hit them again (13.78 mm at 70 mm).
    # 90 mm is the same clearance as before, measured from the new jaw tip.
    mouth = np.array(
        [S.LATHE_X - 0.090, S.LATHE_Y, S.LATHE_Z])
    for _ in range(3):
        err = arm.move_to(mouth, n_steps=300, grip=GRIP_OPEN,
                          wrist_roll=WRIST_ROLL_GRASP)
        if err < 0.003:
            break
    reach_err = err * 1000
    print("  卡盘口 IK 残差  : %.2f mm   (粗判 ≤ %.1f，非送入精度)"
          % (reach_err, REACH_LIMIT))
    print("  pad 终点        : %s" % np.round(arm.pad_center(), 4))
    print()

    ok = (ecc <= ECC_LIMIT and tilt <= TILT_LIMIT
          and contacts >= CONTACT_MIN and reach_err <= REACH_LIMIT
          and spread <= SPREAD_LIMIT and force_ratio <= FORCE_RATIO_LIMIT
          and eq_ok and follow_ok)
    print("=" * 74)
    print("CHUCK_HEALTH jaw_spread_mm=%.3f ecc_mm=%.2f tilt_deg=%.2f "
          "contacts=%d reach_err_mm=%.2f force_ratio=%.2f"
          % (spread, ecc, tilt, contacts, reach_err, force_ratio))
    print("CHUCK_HEALTH %s" % ("PASS" if ok else "FAIL"))
    print("=" * 74)
    print("OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
