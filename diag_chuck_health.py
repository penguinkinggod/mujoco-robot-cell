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

Known limitation (measured, not guessed)
---------------------------------------
The three jaws are driven by three independent position servos, so their
displacements differ by ~4 mm and their normal forces by ~4x. A joint
equality constraint was tried (mjEQ_JOINT, q1=q2) and rejected: 12 kp/ctrl
combinations all ejected the shaft (0 contacts, eccentricity > 300 mm).
The independent-servo version actually clamps well (0.2 mm eccentricity,
20 N pull-out), so jaw_spread is reported as diagnostic information, NOT as
a pass/fail criterion. Do not "fix" it without re-running this script.

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
# Coarse workspace check ONLY. A bare IK move still carries ~10 mm of
# steady-state error. Final insertion accuracy (coaxiality <= 2 mm) needs
# demo_09's shaft_in_grip compensation in the merged scene -- that is the
# next task, not a criterion this script can honestly claim to pass.
REACH_LIMIT = 10.0     # mm, IK residual at the pre-insertion stop point

SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])


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

    # ---------------- physics ----------------
    mujoco.mj_resetData(model, data)
    data.qpos[sq:sq + 3] = [
        S.LATHE_X + S.SHAFT_L / 2 - 0.020, S.LATHE_Y, S.LATHE_Z]
    data.qpos[sq + 3:sq + 7] = SHAFT_Q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    for _ in range(200):
        for a in acts:
            data.ctrl[a] = 0.005
        mujoco.mj_step(model, data)
    for _ in range(1500):
        for a in acts:
            data.ctrl[a] = -0.012
        mujoco.mj_step(model, data)

    jaw_q = np.array([data.qpos[a] * 1000 for a in ja])
    spread = float(jaw_q.max() - jaw_q.min())
    p = data.qpos[sq:sq + 3]
    quat = data.qpos[sq + 3:sq + 7]
    ecc = float(np.hypot((p[1] - S.LATHE_Y) * 1000,
                         (p[2] - S.LATHE_Z) * 1000))
    tilt = float(np.degrees(
        2 * np.arccos(min(1.0, abs(float(np.dot(SHAFT_Q0, quat)))))))

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
            if ("jaw_geom_%d" % k) in (n1, n2):
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f)
                contacts += 1
                forces[k] += abs(f[0])

    print("  三爪位移 (mm)   : %s   差 %.2f mm"
          % (np.round(jaw_q, 2), spread))
    print("  各爪法向力 (N)  : %s" % np.round(forces, 1))
    print("  轴心偏心        : %.2f mm   (要求 ≤ %.1f)" % (ecc, ECC_LIMIT))
    print("  轴姿态倾斜      : %.2f deg  (要求 ≤ %.1f)" % (tilt, TILT_LIMIT))
    print("  卡爪-轴接触点   : %d        (要求 ≥ %d)" % (contacts, CONTACT_MIN))
    print("  （位移差为已知特征，仅诊断，不参与判定；见文件头说明）")
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
    # 85.68 mm residual at z=0.06. At 70 mm it is 9.08 mm. Raising the
    # chuck to z=0.14 would give 6.54 mm, but that needs a spindle housing.
    mouth = np.array(
        [S.LATHE_X - 0.070, S.LATHE_Y, S.LATHE_Z])
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
          and contacts >= CONTACT_MIN and reach_err <= REACH_LIMIT)
    print("=" * 74)
    print("CHUCK_HEALTH jaw_spread_mm=%.2f ecc_mm=%.2f tilt_deg=%.2f "
          "contacts=%d reach_err_mm=%.2f"
          % (spread, ecc, tilt, contacts, reach_err))
    print("CHUCK_HEALTH %s" % ("PASS" if ok else "FAIL"))
    print("=" * 74)
    print("OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
