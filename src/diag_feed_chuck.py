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
import json
import os
import platform
import subprocess
import time
import uuid

import numpy as np
import mujoco
import scipy
from scipy.interpolate import CubicSpline

import scene_full as F
import scene_machining as SM
from demo_08_pick_and_load import (
    Arm, WRIST_ROLL_GRASP, GRIP_OPEN, GRIP_CLOSE,
)


RESULT_SCHEMA_VERSION = 13
METRIC_SCHEMA_VERSION = 1
CRITERION_SET_ID = "feed-criteria-v1"
PLANNING_MODES = ("M0", "M1", "M2")
DEFAULT_INITIAL_SETTLE_STEPS = 600
INITIAL_STABILITY_WINDOW_STEPS = 100
INITIAL_STABILITY_ENDPOINT_EXCURSION_LIMIT_MM = 0.2
WORLD_X_AXIS = np.array([1.0, 0.0, 0.0])
# Frozen from the P4.2 zero-disturbance case after the same 600-step fixture
# settle. This is calibration data, not a per-case measurement consumed by M0.
NOMINAL_SETTLED_POSITION_M = np.array([
    0.260000000, -0.130000000, 0.039812104])
FEED_CHECKPOINTS = (
    ("feed_complete", "进给完成"),
    ("chuck_clamped", "三爪夹紧完成"),
    ("gripper_released_in_place", "原地松爪完成"),
    ("axial_retreat_complete", "轴向退出完成"),
    ("lateral_retreat_complete", "侧向退出完成"),
    ("unlock_settled", "解锁静置完成"),
)


def _resolve_initial_yaw_deg(legacy_roll_deg, yaw_world_z_deg):
    legacy_value = 0.0 if legacy_roll_deg is None else float(legacy_roll_deg)
    if yaw_world_z_deg is None:
        return legacy_value
    if legacy_value != 0.0:
        raise ValueError(
            "use either initial_yaw_deg or legacy initial_roll_deg")
    return float(yaw_world_z_deg)


def _apply_world_yaw_to_quaternion(quaternion_wxyz, yaw_world_z_deg):
    half_angle = np.radians(float(yaw_world_z_deg)) / 2.0
    yaw_quaternion = np.array([
        np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
    rotated = np.zeros(4)
    mujoco.mju_mulQuat(
        rotated, yaw_quaternion, np.asarray(quaternion_wxyz, dtype=float))
    return rotated


def _normalise_axis(axis):
    axis = np.asarray(axis, dtype=float)
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _pose_in_tool_frame(tool_rotation, object_position, tool_position,
                        object_axis):
    rotation = np.asarray(tool_rotation, dtype=float).reshape(3, 3)
    offset_tool = rotation.T @ (
        np.asarray(object_position, dtype=float)
        - np.asarray(tool_position, dtype=float))
    axis_tool = _normalise_axis(
        rotation.T @ np.asarray(object_axis, dtype=float))
    return offset_tool, axis_tool


def _relative_pose_change(reference_offset_tool, reference_axis_tool,
                          current_offset_tool, current_axis_tool):
    translation_delta = (
        np.asarray(current_offset_tool, dtype=float)
        - np.asarray(reference_offset_tool, dtype=float))
    axis_dot = abs(float(np.dot(
        _normalise_axis(reference_axis_tool),
        _normalise_axis(current_axis_tool))))
    axis_change_deg = float(np.degrees(np.arccos(
        np.clip(axis_dot, -1.0, 1.0))))
    return translation_delta, axis_change_deg


def _pose_axis_from_quaternion(quaternion_wxyz):
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion_wxyz, dtype=float))
    return _normalise_axis(rotation.reshape(3, 3)[:, 2])


def _resolve_planning_pose(planning_mode, true_pose, nominal_position_m,
                           nominal_quaternion_wxyz, m2_error_mm=(0.0, 0.0,
                                                                  0.0),
                           m2_error_yaw_deg=0.0):
    """Choose the pose the planner may consume for one physical case.

    The simulator always keeps ``true_pose`` for evidence.  This function is
    the information boundary: M0 never consumes it, M1 consumes it directly,
    and M2 consumes only a deterministic noisy copy supplied by the caller.
    """
    mode = str(planning_mode).upper()
    if mode not in PLANNING_MODES:
        raise ValueError(
            "planning_mode must be one of %s, got %r"
            % ("/".join(PLANNING_MODES), planning_mode))
    error_mm = np.asarray(m2_error_mm, dtype=float)
    if error_mm.shape != (3,) or not np.all(np.isfinite(error_mm)):
        raise ValueError("m2_error_mm must contain three finite values")
    if not np.isfinite(float(m2_error_yaw_deg)):
        raise ValueError("m2_error_yaw_deg must be finite")

    if mode == "M0":
        position = np.asarray(nominal_position_m, dtype=float).copy()
        quaternion = np.asarray(
            nominal_quaternion_wxyz, dtype=float).copy()
        position_source = "fixed_nominal_settled_workpiece_center"
        orientation_source = "fixed_nominal_workpiece_pose"
        error_used = np.zeros(3)
        yaw_error_used = 0.0
    elif mode == "M1":
        true_position = np.asarray(true_pose["position_m"], dtype=float)
        true_quaternion = np.asarray(
            true_pose["quaternion_wxyz"], dtype=float)
        position = true_position.copy()
        quaternion = true_quaternion.copy()
        position_source = "settled_true_workpiece_center"
        orientation_source = "settled_true_workpiece_pose"
        error_used = np.zeros(3)
        yaw_error_used = 0.0
    else:
        true_position = np.asarray(true_pose["position_m"], dtype=float)
        true_quaternion = np.asarray(
            true_pose["quaternion_wxyz"], dtype=float)
        position = true_position + error_mm / 1000.0
        quaternion = _apply_world_yaw_to_quaternion(
            true_quaternion, m2_error_yaw_deg)
        position_source = "estimated_workpiece_center"
        orientation_source = "estimated_workpiece_pose"
        error_used = error_mm
        yaw_error_used = float(m2_error_yaw_deg)
    return {
        "mode": mode,
        "position_m": position,
        "quaternion_wxyz": quaternion,
        "shaft_axis_world_unit": _pose_axis_from_quaternion(quaternion),
        "position_source": position_source,
        "orientation_source": orientation_source,
        "m2_error_used_mm": error_used,
        "m2_error_used_yaw_deg": yaw_error_used,
    }


_REPRODUCIBLE_ENV_KEYS = {
    "FEED_INSERT_MM", "FEED_Z_COMP_MM", "FEED_GRIP_STATION_MM",
    "FEED_RETREAT_AXIAL", "FEED_LEVEL_ROD", "FEED_AXIS_LEAD_DEG",
    "FEED_SEAT_TAP_MM", "FEED_BUSHING", "FEED_BUSHING_GAP",
    "FEED_BUSHING_FRONT", "FEED_BUSHING_LEN", "FEED_BUSHING_N",
    "FEED_BUSHING_MODE", "FEED_BUSHING_MU", "FEED_BUSHING_OPEN",
    "FEED_BUSHING_STIFF", "FEED_BUSHING_ALWAYS",
    "FEED_BUSHING_RELEASE", "GRIP_RAMP", "FEED_ITERS", "FEED_GAIN",
    "FEED_STEPS", "FEED_ARM_KV", "FEED_CLAMP_SETTLE",
    "FEED_CLAMP_FLOAT", "FEED_ARM_GCOMP", "FEED_ARM_GCOMP_GAIN",
    "FEED_ARM_PAYLOAD", "FEED_ARM_KP", "FEED_GRIP_FORCE_NM",
    "FEED_GRIP_KP", "FEED_CLAMP_ANCHOR", "FEED_CLAMP_LIFT",
    "FEED_HANDOVER_FIRST", "FEED_SEAT_K", "FEED_SEAT_C",
    "FEED_SEAT_DWELL", "FEED_SEAT_RADIAL_K", "FEED_SEAT_RADIAL_C",
    "FEED_SEAT_FADE", "FEED_PRECLAMP", "FEED_PRECLAMP_CTRL",
    "FEED_PRECLAMP_STEPS", "FEED_PRECLAMP_DWELL",
    "FEED_PRECLAMP_SEAT_K", "FEED_PRECLAMP_SEAT_C",
    "FEED_ARM_LEAD", "FEED_ARM_DAMP", "FEED_FLOAT_DAMP",
    "FEED_CONVERGE", "FEED_CONVERGE_TOL", "FEED_BORE_PROBE",
    "FEED_GRIP_CLOSE", "FEED_DIAG_NO_RETREAT", "SMOOTH",
    "FEED_TRACE_RELATIVE_MOTION",
    "SO101_KP", "SO101_KV", "SM_JAW_BORE_Q", "FEED_DEBUG",
    "FEED_CLAMP_TRACE", "FEED_SEAT_TRACE",
    "FEED_V_CONTACT_SOLREF_TIME",
}
_SOURCE_IDENTITY = None


def _capture_environment_overrides(environ=None):
    """Capture physics/debug overrides, excluding case/output bookkeeping."""
    environ = os.environ if environ is None else environ
    return {
        key: str(environ[key])
        for key in sorted(_REPRODUCIBLE_ENV_KEYS)
        if key in environ
    }


def _source_identity():
    global _SOURCE_IDENTITY
    if _SOURCE_IDENTITY is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, capture_output=True, text=True,
            check=False).stdout.strip()
        _SOURCE_IDENTITY = {
            "git_revision": revision or "unknown",
            "git_worktree_dirty": bool(dirty),
        }
    return _SOURCE_IDENTITY


def _criteria_snapshot():
    return {
        "hard": {
            "ik_result_required": True,
            "v_jaw_grasp_contacts_min": 1,
            "chuck_contacts_min": CONTACT_MIN,
            "pull_out_mm_max": PULL_LIMIT,
            "robot_machine_contacts_max": 0,
            "rod_external_force_N_max_when_tuned": 1e-6,
        },
        "quality": {
            "coax_mm_max": COAX_LIMIT,
            "tilt_deg_max": TILT_LIMIT,
            "gravity_slip_mm_max": SLIP_LIMIT,
            "handover_swing_mm_max": CLAMP_SWING_LIMIT,
            "stock_eccentricity_mm_max": STOCK_ALLOW_MM,
        },
    }


def _model_options_snapshot(model):
    opt = model.opt
    return {
        "timestep_s": float(opt.timestep),
        "gravity_m_s2": [float(x) for x in opt.gravity],
        "integrator": int(opt.integrator),
        "solver": int(opt.solver),
        "iterations": int(opt.iterations),
        "tolerance": float(opt.tolerance),
        "cone": int(opt.cone),
        "impratio": float(opt.impratio),
        "noslip_iterations": int(opt.noslip_iterations),
        "noslip_tolerance": float(opt.noslip_tolerance),
    }


def _versions_snapshot():
    identity = _source_identity()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mujoco": getattr(mujoco, "__version__", "unknown"),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        **identity,
    }


def _case_metadata(state, model):
    versions = state["versions"]
    versions = dict(versions)
    versions.update(_source_identity())
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "criterion_set_id": CRITERION_SET_ID,
        "criteria": _criteria_snapshot(),
        "inputs": state["inputs"],
        "actual_initial_pose": state["actual_initial_pose"],
        "control_mode": state["control_mode"],
        "effective_parameters": state["effective_parameters"],
        "environment_overrides": state["environment_overrides"],
        "model_options": {
            "initial": state["model_options_initial"],
            "at_result": _model_options_snapshot(model),
        },
        "versions": versions,
        # Keep the original top-level version fields for existing readers.
        **versions,
    }


def _feed_checkpoint_statuses(axial_retreat, no_retreat_diagnostic):
    axial_done = axial_retreat and not no_retreat_diagnostic
    lateral_done = not no_retreat_diagnostic
    return {
        "axial_retreat_complete": "completed" if axial_done else "skipped",
        "lateral_retreat_complete": (
            "completed" if lateral_done else "skipped"),
    }


def _finite_sim_state(qpos, qvel, ctrl, qacc, qfrc_applied,
                      xfrc_applied):
    return all(np.all(np.isfinite(values)) for values in (
        qpos, qvel, ctrl, qacc, qfrc_applied, xfrc_applied))


def _case_classification(execution_status, evaluation_scope, hard_pass,
                         quality_pass, overall_pass):
    """Separate trial usability and evaluation scope from measured outcome."""
    trial_data_valid = execution_status in ("completed", "aborted")
    primary_trial = evaluation_scope == "primary_physics_feed"
    if execution_status == "error":
        outcome_class = "runtime_error"
    elif execution_status == "aborted":
        outcome_class = "aborted"
    elif not primary_trial:
        prefix = ("reference" if evaluation_scope
                  == "kinematic_lock_reference" else "diagnostic")
        outcome_class = "%s_%s" % (
            prefix, "pass" if overall_pass else "fail")
    elif not hard_pass:
        outcome_class = "hard_fail"
    elif not quality_pass:
        outcome_class = "quality_fail"
    elif overall_pass:
        outcome_class = "physical_success"
    else:
        outcome_class = "criteria_fail"
    return {
        "evaluation_scope": evaluation_scope,
        "trial_data_valid": trial_data_valid,
        "include_in_primary_trial_stats": (
            primary_trial and trial_data_valid),
        "physical_task_success": (
            primary_trial and execution_status == "completed"
            and bool(overall_pass)),
        "outcome_class": outcome_class,
    }


def _robot_machine_geom_ids(model):
    """Return semantic robot and machine geom sets for contact detection."""
    base_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "base")
    robot_bodies = set()
    if base_id >= 0:
        for body_id in range(model.nbody):
            parent_id = body_id
            while parent_id > 0 and parent_id != base_id:
                parent_id = int(model.body_parentid[parent_id])
            if parent_id == base_id:
                robot_bodies.add(body_id)

    robot_geom_ids = {
        geom_id for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in robot_bodies
    }
    machine_mask = F.CT_MACHINE
    machine_geom_ids = {
        geom_id for geom_id in range(model.ngeom)
        if (int(model.geom_contype[geom_id]),
            int(model.geom_conaffinity[geom_id])) == machine_mask
    }
    return robot_geom_ids, machine_geom_ids


def _is_robot_machine_contact(geom1, geom2, robot_geom_ids,
                              machine_geom_ids):
    return ((geom1 in robot_geom_ids and geom2 in machine_geom_ids)
            or (geom2 in robot_geom_ids and geom1 in machine_geom_ids))


_CONTACT_CATEGORIES = (
    "robot_workpiece",
    "robot_machine",
    "workpiece_machine",
    "workpiece_fixture",
    "robot_fixture",
    "other",
)


def _semantic_contact_geom_sets(model, robot_geom_ids=None):
    """Classify collision geometry by model identity, not mask bits."""
    if robot_geom_ids is None:
        robot_geom_ids, _ = _robot_machine_geom_ids(model)
    shaft_geom_ids = set()
    machine_geom_ids = set()
    fixture_geom_ids = set()
    for geom_id in range(model.ngeom):
        name = (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        if name.startswith("shaft_"):
            shaft_geom_ids.add(geom_id)
        elif (name.startswith(("lathe_", "chuck_", "jaw_geom_",
                               "jaw_back_"))
              or name == "spindle"):
            machine_geom_ids.add(geom_id)
        elif name in ("floor", "table") or name.startswith(
                ("bushing_", "cradle_", "fixture_", "vise_")):
            fixture_geom_ids.add(geom_id)
    return {
        "robot": set(robot_geom_ids),
        "workpiece": shaft_geom_ids,
        "machine": machine_geom_ids,
        "fixture": fixture_geom_ids,
    }


def _contact_category(geom1, geom2, geom_sets):
    """Return a semantic class for one MuJoCo contact geometry pair."""
    first, second = int(geom1), int(geom2)
    robot = geom_sets["robot"]
    workpiece = geom_sets["workpiece"]
    machine = geom_sets["machine"]
    fixture = geom_sets["fixture"]
    if ((first in robot and second in workpiece)
            or (second in robot and first in workpiece)):
        return "robot_workpiece"
    if ((first in robot and second in machine)
            or (second in robot and first in machine)):
        return "robot_machine"
    if ((first in workpiece and second in machine)
            or (second in workpiece and first in machine)):
        return "workpiece_machine"
    if ((first in workpiece and second in fixture)
            or (second in workpiece and first in fixture)):
        return "workpiece_fixture"
    if ((first in robot and second in fixture)
            or (second in robot and first in fixture)):
        return "robot_fixture"
    return "other"


def _contact_frame_snapshot(model, data, geom_sets):
    """Describe all contacts in the current physics frame."""
    counts = {category: 0 for category in _CONTACT_CATEGORIES}
    force_sums = {category: 0.0 for category in _CONTACT_CATEGORIES}
    pairs = {}
    force6 = np.zeros(6)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        category = _contact_category(
            geom_ids[0], geom_ids[1], geom_sets)
        names = []
        bodies = []
        for geom_id in geom_ids:
            names.append(mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "?")
            body_id = int(model.geom_bodyid[geom_id])
            bodies.append(mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "?")
        geom_body_pairs = sorted(zip(names, bodies))
        pair_geoms = [row[0] for row in geom_body_pairs]
        pair_bodies = [row[1] for row in geom_body_pairs]
        key = "%s|%s|%s" % (category, pair_geoms[0], pair_geoms[1])
        mujoco.mj_contactForce(model, data, contact_id, force6)
        normal_force = abs(float(force6[0]))
        counts[category] += 1
        force_sums[category] += normal_force
        pair = pairs.setdefault(key, {
            "category": category,
            "geoms": pair_geoms,
            "bodies": pair_bodies,
            "contact_count": 0,
            "normal_force_sum_N": 0.0,
            "normal_force_peak_N": 0.0,
        })
        pair["contact_count"] += 1
        pair["normal_force_sum_N"] += normal_force
        pair["normal_force_peak_N"] = max(
            pair["normal_force_peak_N"], normal_force)
    return {
        "total_contact_count": int(data.ncon),
        "counts_by_category": counts,
        "normal_force_sum_by_category_N": force_sums,
        "pairs": [pairs[key] for key in sorted(pairs)],
    }


def _applied_load_snapshot(model, data):
    """Read externally applied generalized and body-space loads."""
    qfrc = []
    for dof_id, value in enumerate(data.qfrc_applied):
        value = float(value)
        if value == 0.0:
            continue
        joint_id = int(model.dof_jntid[dof_id])
        joint_type = int(model.jnt_type[joint_id])
        local_dof = dof_id - int(model.jnt_dofadr[joint_id])
        force_dof = (
            joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE)
            or (joint_type == int(mujoco.mjtJoint.mjJNT_FREE)
                and local_dof < 3))
        body_id = int(model.jnt_bodyid[joint_id])
        qfrc.append({
            "dof_id": int(dof_id),
            "joint": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "?",
            "body": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "?",
            "local_dof": int(local_dof),
            "value": value,
            "unit": "N" if force_dof else "N*m",
        })

    xfrc = []
    for body_id in range(1, model.nbody):
        wrench = np.asarray(data.xfrc_applied[body_id], dtype=float)
        if not np.any(wrench != 0.0):
            continue
        xfrc.append({
            "body_id": int(body_id),
            "body": mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "?",
            "force_N": [float(value) for value in wrench[:3]],
            "force_magnitude_N": float(np.linalg.norm(wrench[:3])),
            "torque_Nm": [float(value) for value in wrench[3:]],
            "torque_magnitude_Nm": float(np.linalg.norm(wrench[3:])),
        })
    return {"qfrc_applied": qfrc, "xfrc_applied": xfrc}


def _new_diagnostic_rollup():
    return {
        "samples": 0,
        "sim_time_start_s": None,
        "sim_time_end_s": None,
        "categories": {
            category: {
                "samples_with_contacts": 0,
                "peak_contacts_per_step": 0,
                "peak_normal_force_sum_N": 0.0,
            }
            for category in _CONTACT_CATEGORIES
        },
        "pairs": {},
        "qfrc_by_dof": {},
        "xfrc_by_body": {},
    }


def _new_load_rollup():
    return {
        "samples": 0,
        "sim_time_start_s": None,
        "sim_time_end_s": None,
        "qfrc_by_dof": {},
        "xfrc_by_body": {},
    }


def _accumulate_contact_rollup(rollup, snapshot, sim_time_s):
    if rollup["sim_time_start_s"] is None:
        rollup["sim_time_start_s"] = float(sim_time_s)
    rollup["samples"] += 1
    rollup["sim_time_end_s"] = float(sim_time_s)
    for category in _CONTACT_CATEGORIES:
        count = snapshot["counts_by_category"][category]
        summary = rollup["categories"][category]
        if count:
            summary["samples_with_contacts"] += 1
        summary["peak_contacts_per_step"] = max(
            summary["peak_contacts_per_step"], count)
        summary["peak_normal_force_sum_N"] = max(
            summary["peak_normal_force_sum_N"],
            snapshot["normal_force_sum_by_category_N"][category])
    for pair in snapshot["pairs"]:
        key = "%s|%s" % (pair["category"], "|".join(pair["geoms"]))
        summary = rollup["pairs"].setdefault(key, {
            "category": pair["category"],
            "geoms": pair["geoms"],
            "bodies": pair["bodies"],
            "contact_samples": 0,
            "peak_contacts_per_step": 0,
            "peak_normal_force_per_contact_N": 0.0,
            "peak_normal_force_sum_N": 0.0,
        })
        summary["contact_samples"] += 1
        summary["peak_contacts_per_step"] = max(
            summary["peak_contacts_per_step"], pair["contact_count"])
        summary["peak_normal_force_per_contact_N"] = max(
            summary["peak_normal_force_per_contact_N"],
            pair["normal_force_peak_N"])
        summary["peak_normal_force_sum_N"] = max(
            summary["peak_normal_force_sum_N"],
            pair["normal_force_sum_N"])


def _accumulate_load_rollup(rollup, snapshot, sim_time_s):
    if rollup["sim_time_start_s"] is None:
        rollup["sim_time_start_s"] = float(sim_time_s)
    rollup["samples"] += 1
    rollup["sim_time_end_s"] = float(sim_time_s)
    for load in snapshot["qfrc_applied"]:
        key = "%s:dof%d" % (load["joint"], load["local_dof"])
        summary = rollup["qfrc_by_dof"].setdefault(key, {
            "joint": load["joint"],
            "body": load["body"],
            "local_dof": load["local_dof"],
            "unit": load["unit"],
            "nonzero_samples": 0,
            "peak_abs": 0.0,
            "value_at_peak": 0.0,
        })
        summary["nonzero_samples"] += 1
        if abs(load["value"]) > summary["peak_abs"]:
            summary["peak_abs"] = abs(load["value"])
            summary["value_at_peak"] = load["value"]
    for load in snapshot["xfrc_applied"]:
        summary = rollup["xfrc_by_body"].setdefault(load["body"], {
            "body_id": load["body_id"],
            "nonzero_samples": 0,
            "force_peak_N": 0.0,
            "force_vector_at_peak_N": [0.0, 0.0, 0.0],
            "torque_peak_Nm": 0.0,
            "torque_vector_at_peak_Nm": [0.0, 0.0, 0.0],
        })
        summary["nonzero_samples"] += 1
        if load["force_magnitude_N"] > summary["force_peak_N"]:
            summary["force_peak_N"] = load["force_magnitude_N"]
            summary["force_vector_at_peak_N"] = load["force_N"]
        if load["torque_magnitude_Nm"] > summary["torque_peak_Nm"]:
            summary["torque_peak_Nm"] = load["torque_magnitude_Nm"]
            summary["torque_vector_at_peak_Nm"] = load["torque_Nm"]


def _update_diagnostic_rollups(state, scope_name, phase, sim_time_s,
                              contact_snapshot, load_snapshot):
    contact_scope = state["contact_diagnostics"]["scopes"].setdefault(
        scope_name, {"overall": _new_diagnostic_rollup(), "phases": {}})
    contact_phase = contact_scope["phases"].setdefault(
        phase, _new_diagnostic_rollup())
    _accumulate_contact_rollup(
        contact_scope["overall"], contact_snapshot, sim_time_s)
    _accumulate_contact_rollup(contact_phase, contact_snapshot, sim_time_s)

    load_scope = state["applied_load_diagnostics"]["scopes"].setdefault(
        scope_name, {"overall": _new_load_rollup(), "phases": {}})
    load_phase = load_scope["phases"].setdefault(
        phase, _new_load_rollup())
    _accumulate_load_rollup(load_scope["overall"], load_snapshot, sim_time_s)
    _accumulate_load_rollup(load_phase, load_snapshot, sim_time_s)


def _diagnostic_result_fields(state):
    return {
        "contact_diagnostics": state.get(
            "contact_diagnostics", {"scopes": {}}),
        "applied_load_diagnostics": state.get(
            "applied_load_diagnostics", {"scopes": {}}),
        "penetration_diagnostics": state.get(
            "penetration_diagnostics", {}),
        "pickup_motion_diagnostics": state.get(
            "pickup_motion_diagnostics", {}),
        "gripper_relative_workpiece_motion": state.get(
            "gripper_relative_workpiece_motion", {}),
        "initial_stability": state.get("initial_stability"),
        "diagnostic_scope_definitions": {
            "feed": "送料、夹紧、松爪、退出和解锁静置；不含 20 N 拉拔测试",
            "pull_test": "机器人退出后单独施加的 20 N 轴向拉拔测试",
        },
    }


def _summarize_initial_stability(
        samples, timestep_s, shaft_length_m, support_geom_names_by_side,
        gravity, workpiece_gravcomp):
    """Summarize the pre-motion settling window without changing feed criteria."""
    samples = list(samples)
    sample_count = len(samples)
    window_complete = sample_count == INITIAL_STABILITY_WINDOW_STEPS
    summary = {
        "status": "evaluated" if window_complete else "insufficient_samples",
        "sample_count": sample_count,
        "required_sample_count": INITIAL_STABILITY_WINDOW_STEPS,
        "sample_interval_s": float(timestep_s),
        "observed_span_s": (
            float(samples[-1]["sim_time_s"] - samples[0]["sim_time_s"])
            if sample_count > 1 else 0.0),
        "support_geom_names_by_side": {
            side: sorted(names)
            for side, names in support_geom_names_by_side.items()
        },
        "endpoint_excursion_limit_mm": (
            INITIAL_STABILITY_ENDPOINT_EXCURSION_LIMIT_MM),
        "endpoint_excursion_limit_basis": (
            "project provisional budget: 10% of the existing 2.0 mm "
            "stock allowance; not an industry standard"),
        "workpiece_gravcomp": float(workpiece_gravcomp),
        "gravity_m_s2": [float(value) for value in gravity],
        "samples": samples,
        "stability_pass": None,
    }
    if not samples:
        return summary

    reference_position = np.asarray(samples[0]["position_m"], dtype=float)
    reference_axis = np.asarray(
        samples[0]["shaft_axis_world_unit"], dtype=float)
    reference_axis /= max(float(np.linalg.norm(reference_axis)), 1e-12)
    half_length = float(shaft_length_m) / 2.0
    max_center_excursion_m = 0.0
    max_axis_change_deg = 0.0
    max_endpoint_excursion_m = 0.0
    max_linear_speed_m_s = 0.0
    max_angular_speed_rad_s = 0.0
    support_contact_samples = {"L": 0, "R": 0}
    max_qfrc_force_N = 0.0
    max_qfrc_torque_Nm = 0.0
    max_xfrc_force_N = 0.0
    max_xfrc_torque_Nm = 0.0

    for sample in samples:
        position = np.asarray(sample["position_m"], dtype=float)
        axis = np.asarray(
            sample["shaft_axis_world_unit"], dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        max_center_excursion_m = max(
            max_center_excursion_m,
            float(np.linalg.norm(position - reference_position)))
        axis_dot = float(np.clip(np.dot(reference_axis, axis), -1.0, 1.0))
        max_axis_change_deg = max(
            max_axis_change_deg,
            float(np.degrees(np.arccos(abs(axis_dot)))))
        if axis_dot < 0.0:
            axis = -axis
        for sign in (-1.0, 1.0):
            reference_end = (
                reference_position + sign * half_length * reference_axis)
            current_end = position + sign * half_length * axis
            max_endpoint_excursion_m = max(
                max_endpoint_excursion_m,
                float(np.linalg.norm(current_end - reference_end)))

        linear_velocity = np.asarray(
            sample["linear_velocity_m_s"], dtype=float)
        angular_velocity = np.asarray(
            sample["angular_velocity_rad_s"], dtype=float)
        max_linear_speed_m_s = max(
            max_linear_speed_m_s, float(np.linalg.norm(linear_velocity)))
        max_angular_speed_rad_s = max(
            max_angular_speed_rad_s, float(np.linalg.norm(angular_velocity)))
        for side in ("L", "R"):
            if sample["support_contact_count_by_side"].get(side, 0) > 0:
                support_contact_samples[side] += 1

        qfrc_force = np.asarray(
            sample["qfrc_applied_force_N"], dtype=float)
        qfrc_torque = np.asarray(
            sample["qfrc_applied_torque_Nm"], dtype=float)
        xfrc_force = np.asarray(
            sample["xfrc_applied_force_N"], dtype=float)
        xfrc_torque = np.asarray(
            sample["xfrc_applied_torque_Nm"], dtype=float)
        max_qfrc_force_N = max(
            max_qfrc_force_N, float(np.linalg.norm(qfrc_force)))
        max_qfrc_torque_Nm = max(
            max_qfrc_torque_Nm, float(np.linalg.norm(qfrc_torque)))
        max_xfrc_force_N = max(
            max_xfrc_force_N, float(np.linalg.norm(xfrc_force)))
        max_xfrc_torque_Nm = max(
            max_xfrc_torque_Nm, float(np.linalg.norm(xfrc_torque)))

    endpoint_excursion_mm = max_endpoint_excursion_m * 1000.0
    support_contact_continuous = {
        side: (
            bool(support_geom_names_by_side.get(side))
            and support_contact_samples[side] == sample_count
        )
        for side in ("L", "R")
    }
    gravity_enabled = bool(np.allclose(
        np.asarray(gravity, dtype=float),
        np.array([0.0, 0.0, -9.81]),
        rtol=0.0,
        atol=1e-12,
    ))
    no_external_applied_loads = all(value == 0.0 for value in (
        max_qfrc_force_N, max_qfrc_torque_Nm,
        max_xfrc_force_N, max_xfrc_torque_Nm,
    ))
    checks = {
        "endpoint_excursion_within_limit": (
            endpoint_excursion_mm
            <= INITIAL_STABILITY_ENDPOINT_EXCURSION_LIMIT_MM),
        "left_support_contact_continuous": (
            support_contact_continuous["L"]),
        "right_support_contact_continuous": (
            support_contact_continuous["R"]),
        "gravity_enabled": gravity_enabled,
        "workpiece_gravcomp_disabled": float(workpiece_gravcomp) == 0.0,
        "no_external_applied_loads_on_workpiece": (
            no_external_applied_loads),
    }
    summary.update({
        "reference_pose": {
            "position_m": [float(value) for value in reference_position],
            "shaft_axis_world_unit": [
                float(value) for value in reference_axis],
        },
        "max_center_excursion_mm": max_center_excursion_m * 1000.0,
        "max_axis_change_deg": max_axis_change_deg,
        "max_endpoint_excursion_mm": endpoint_excursion_mm,
        "max_linear_speed_m_s": max_linear_speed_m_s,
        "max_angular_speed_rad_s": max_angular_speed_rad_s,
        "support_contact_samples_by_side": support_contact_samples,
        "support_contact_continuous_by_side": (
            support_contact_continuous),
        "max_qfrc_applied_force_N": max_qfrc_force_N,
        "max_qfrc_applied_torque_Nm": max_qfrc_torque_Nm,
        "max_xfrc_applied_force_N": max_xfrc_force_N,
        "max_xfrc_applied_torque_Nm": max_xfrc_torque_Nm,
        "no_external_applied_loads_on_workpiece": (
            no_external_applied_loads),
        "checks": checks if window_complete else None,
        "stability_pass": (
            bool(all(checks.values())) if window_complete else None),
    })
    return summary


COAX_LIMIT = 2.0       # mm
TILT_LIMIT = 10.0      # deg
CONTACT_MIN = 6
SLIP_LIMIT = 2.0       # mm
PULL_LIMIT = 2.0       # mm
# 交接全程（夹紧+松爪+退出）棒料 z 的峰谷。slip_mm 只测解锁后静置，
# 覆盖不到人眼看到的那一下位移。
# 2026-11-07 用户确认：真机三爪本来就是这个分布，夹紧瞬态的可见位移
# 会被第一刀切掉，不影响成品，不作为工艺质量目标。这里只保留一个宽松
# 护栏，防止回归时真的把料甩飞；不再为了「画面看不出动」去调参数。
CLAMP_SWING_LIMIT = 12.0   # mm（甩料护栏，非精度指标）

# 加工验收判据：整根棒料相对主轴轴线的最大径向偏心必须小于单边余量。
# 一次装夹内车出的所有表面天然同轴于主轴，来料偏心只决定第一刀能否
# 见光（会不会留黑皮），不决定成品精度。默认按 Φ20 棒料粗车到 Φ16、
# 单边 2.0 mm 余量；实际工件尺寸变了要改这里。
STOCK_ALLOW_MM = 2.0      # mm

# 0.20 was picked by an old x-scan that only checked POSITION. With the rod
# axis held horizontal, a 5-DoF SO-101 cannot reach that part of the table:
# measured IK residuals at the final site were 6-11 mm / 18-24 deg, with
# shoulder_lift and wrist_roll against their limits. A workspace scan showed
# x >= 0.15-0.20 m is reachable at y=0.12. Moving the lathe to x=0.28 puts
# every feed waypoint inside that region; the table spans x=-0.02..0.58 m.
LATHE_ORIGIN = (0.28, 0.12, 0.060)
SHAFT_Q0 = np.array([0.7071068, 0.0, 0.7071068, 0.0])

# ---------------- 装夹深度：工艺参数，不是调出来的 ----------------
# 工件「头端」越过卡爪前端面的距离。真实车床上定的是悬伸（stick-out），
# D20 x 80 的棒料夹 55 伸 25 是常规值，对应进给 80-55=25 mm。
#
# 旧值 axis = LATHE_ORIGIN - 0.016（工件中心 x=0.184）把整根 80 mm 的料
# 推到只剩 1 mm 露在卡爪外面 —— 看起来像被卡盘「吸」进去了。算一遍：
#   卡爪前端面 x = 0.200 - 0.055 = 0.145
#   工件占 [0.184-0.040, 0.184+0.040] = [0.144, 0.224]
#   露头 = 0.145 - 0.144 = 0.001 m
# 现在改成按「进多少」给，默认 25 mm（扫描出来的最小稳定值）：
# 卡爪咬住自由端前段，夹爪夹着尾段 —— 两者轴向不重叠，交接不打架。
# Sweep 20/22/23/24/25/26/27/30/35 mm: 20-23 mm loses two jaw contacts
# once the robot lets go; 24-26 mm holds all six. 25 mm is the measured
# minimum stable depth (coax 2.07 mm, axis 1.0 deg, pull-out 0.02 mm).
INSERT_MM = float(os.environ.get("FEED_INSERT_MM", "25")) / 1000.0
APPROACH_MM = 0.030        # 对轴时工件头端离卡爪前端面的间隙
# 进给高度补偿。它原本是为了抵消「夹紧力不够、工件松手后沉 2.4 mm」，
# 属于拿工艺参数糊指标，用户一眼就看出来了（"工件慢慢往下滑"）。
# 卡盘改成力控（-150 N，法向 200 N）以后，实测下沉只剩 0.04 mm，
# 补偿自然归零：0 / 0.5 / 1 mm 扫描对应同轴 0.14 / 0.52 / 1.02 mm，
# 越补偿越偏。留这个开关只为量测，默认必须是 0。
FEED_Z_COMP = float(os.environ.get("FEED_Z_COMP_MM", "0")) / 1000.0
# 抓取点沿棒料轴线相对工件中心的位置：负 x = 靠机器人这一侧的尾段。
# 让自由端（+x）伸出去插卡盘，夹爪始终留在卡盘口外。
# 2026-11-07 重定：原来是 -15 mm（贴尾端）。实测 -15 时料的重心离夹爪
# 只有 15 mm，三爪闭合的推力全作用在「手臂-料」这个长杠杆上，夹爪被顶
# 得抬高 3.36 mm，交接位移 3.75 mm。把槽挪到 -5 mm（更靠料重心）后，
# 咬合同轴 3.06->0.37 mm、交接位移 3.75->1.43 mm、松爪 dz -1.42->0.02。
# 扫描 0 / -5 差别 <0.02 mm，取 -5 留 20 mm 让夹爪本体不贴卡爪口。
GRIP_STATION = -float(os.environ.get("FEED_GRIP_STATION_MM", "5")) / 1000.0
# 松爪退出分两段：先沿轴线退到料尾之外，再侧向撤离。斜着一把撤时张开
# 的刀片会横刮还没脱离的料尾，把刚定好心的料带歪。
RETREAT_AXIAL = os.environ.get("FEED_RETREAT_AXIAL", "1") == "1"
# 进给到位后把料轴调平（见 level_rod()）。默认开：不调平时料"尾端翘起"
# 约 1.2°，80 mm 料两端高差 1.7 mm，肉眼一眼看得出来。
LEVEL_ROD = os.environ.get("FEED_LEVEL_ROD", "0") == "1"
# 进给轴目标的前倾补偿（度）。舵机软，目标是水平时实际会低约 1°，80 mm
# 料两端差 1.7 mm，肉眼就是"尾端翘着"。把 IK 目标轴预先抬高同样的角度，
# 末端实际才落平。只改机器人目标姿态，不给料加外力。
AXIS_LEAD_DEG = float(os.environ.get("FEED_AXIS_LEAD_DEG", "1.0"))
_AXIS_TGT = np.array([np.cos(np.radians(AXIS_LEAD_DEG)), 0.0,
                      np.sin(np.radians(AXIS_LEAD_DEG))])
# 落座：抓起后先抬离台面 SEAT_TAP，静置让料在 V 槽里坐到底。
# 【已否掉，默认 0，留作对照】实测只吃掉 0.33 mm（相对爪 -0.22 -> -0.55），
# 交接位移 1.56 -> 1.54。原因见下：料不是"滑落"，是夹爪被料重撑开，
# 抓多久都一样要回到同一个重力平衡位。
SEAT_TAP = float(os.environ.get("FEED_SEAT_TAP_MM", "0")) / 1000.0


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
    # Keep the V groove cut at the PROVEN centre-grasp pose. Moving the IK
    # reference 15 mm along the rod changed the wrist pitch by ~1 deg, which
    # dropped one blade onto the D20 rod and pushed it 4.9 mm through the
    # table (measured). The groove axis is already parallel to the rod, so
    # the tail-section pickup only needs a translation at RUN time.
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
    # Move the GROOVE, not the arm. Sliding the V blocks 15 mm toward the
    # rod tail leaves the proven wrist pose and descent height untouched,
    # but puts the actual contact patch on the rear section so the free end
    # can enter the chuck without the gripper following it in.
    grip_s = s_mid + GRIP_STATION
    print("    v_jaw: groove station %+.0f mm from rod centre"
          % (GRIP_STATION * 1000))

    mouths = []                            # (usgn, mouth centre) per finger
    for pad_name, usgn, g, p, half_th, O, s in pads:
        b = tmp.geom_bodyid[g]
        O = O + (grip_s - s) * v_ax        # common TAIL-section station
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


def _apply_v_contact_solref_time(model, time_constant_s):
    """Set the experimental solref time on the shaft and four V-jaw geoms."""
    time_constant_s = float(time_constant_s)
    if not np.isfinite(time_constant_s) or time_constant_s <= 0.0:
        raise ValueError(
            "FEED_V_CONTACT_SOLREF_TIME must be a positive finite number")
    target_names = {
        "shaft_geom",
        "fixed_jaw_pad_v1", "fixed_jaw_pad_v2",
        "moving_jaw_pad_v1", "moving_jaw_pad_v2",
    }
    found = set()
    for geom_id in range(model.ngeom):
        name = (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
        if name in target_names:
            model.geom_solref[geom_id, 0] = time_constant_s
            found.add(name)
    missing = target_names - found
    if missing:
        raise ValueError(
            "contact sensitivity model is missing geoms: %s"
            % ", ".join(sorted(missing)))
    return tuple(sorted(found))


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


# ---- 实体托料导套（2026-11-07，CURRENT.md「下一步」唯一一条）----
# 真机长棒料送料会在卡爪口外装导套/中心架，55 mm 悬空段才有【不可撤销】
# 的径向约束。之前 SEAT_RADIAL_K 只是「力等效」支撑，夹紧后必须撤销：
# 一步撤销弹飞 10.27 mm / 13.3 deg，quintic 渐变卸载也只压到 2.38 mm，
# 撤完仍跷 2.2 deg。这里改成常驻实体几何，从对轴一路带到拉拔测试。
# 2026-11-07 第二轮定案：单独加实体导套是【阴性】——软导套被三爪
# 挤穿（顶回 9~17 N），硬导套又与三爪抢料（7.24 mm / 8.6 deg FAIL）。
# 只有配合「先松夹爪 + 径向托力渐变卸载」才成立：导套在夹爪松开到
# 三爪咬合之间顶住料，咬合后自己卸载（解锁静置导套=0/0.0N，拉拔
# 不是假通过）。实测交接位移 4.77->1.05 mm、首次接触推挤 4.20->0.53。
# 2026-11-07 用户否掉这组定案：拆开对照证实「把料摆回主轴轴线」的主要是
# SEAT_RADIAL_K 那根以【理论轴线坐标】为目标的数学弹簧，等于替机械臂完成
# 最后对中，不能算机械臂送料成功。故全部默认回退为关，主路径输出的是
# 【无作弊机械臂基线】（交接位移 4.77 mm，过程姿态未达标，待重做）。
# 开关全部保留，只用于对照；复现旧定案须同时开下面六个，见 CURRENT.md。
# 2026-11-07 阶段 B 定案：下半环托架，间隙 1.6 mm，进给到位才合上。
# 间隙 <1.44 mm 时结果对 0.02 mm 敏感（接触离散化跳变）；1.46~2.0 mm
# 连续 PASS，取中值。托架只在下垂超过间隙时才接触，不给料任何"轴线
# 目标"，交接期棒料 qfrc 峰值为 0（无隐藏力场）。
BUSHING = os.environ.get("FEED_BUSHING", "1") == "1"
BUSHING_GAP = float(os.environ.get("FEED_BUSHING_GAP", "1.6")) / 1000.0
BUSHING_FRONT = float(os.environ.get("FEED_BUSHING_FRONT", "6")) / 1000.0
BUSHING_LEN = float(os.environ.get("FEED_BUSHING_LEN", "14")) / 1000.0
BUSHING_N = int(os.environ.get("FEED_BUSHING_N", "12"))
BUSHING_MODE = os.environ.get("FEED_BUSHING_MODE", "lower")  # full / lower
BUSHING_MU = float(os.environ.get("FEED_BUSHING_MU", "0.05"))
BUSHING_WALL = 0.004
# 中心架初始张开量（径向）。搬运路径横穿卡爪口外，闭合的常驻环会被
# 棒料撞上；真机中心架送料时本来就是张开的，到位才合上。
BUSHING_OPEN = float(os.environ.get("FEED_BUSHING_OPEN", "40")) / 1000.0
# 1 = 硬接触（短 solref + 陡 solimp）。软接触只顶回 9~17 N、被三爪挤穿，
# 料照样下垂；硬接触才托得住。默认开。
BUSHING_STIFF = os.environ.get("FEED_BUSHING_STIFF", "1") == "1"
# 1 = 常驻（构建期就在闭合位，不张开也不闭合）。配合 MODE=lower 就是
# 真机的下半圆托料架：上方全敞开，机械臂把料从上方放进来，托架从第一
# 步就托住悬空段，料不会先在夹爪里低头 1.5 mm 才落到托架上。
# 与「张开/合上」方案的区别：没有瞬移时序、没有驱动，全程同一位姿。
BUSHING_ALWAYS = os.environ.get("FEED_BUSHING_ALWAYS", "0") == "1"
# 1 = 三爪夹住后把托架退回张开位。卡爪已经承担全部载荷，托架继续托着
# 会形成过约束：实测常驻小间隙托架前段很好（咬合 0.87 mm、交接 2.74），
# 但最终劣化到 1.91 mm / 3.9 deg。真机中心架在卡盘夹紧后也常打开。
BUSHING_RELEASE = os.environ.get("FEED_BUSHING_RELEASE", "0") == "1"
BUSHING_CLOSED = {}   # geom name -> 闭合位 pos，供 close_bushing() 用
BUSHING_OPENED = {}   # geom name -> 张开位 pos，供 release_bushing() 用


def _install_bushing(spec):
    """卡爪口外装一段常驻导套，给悬空段真实径向约束。

    MuJoCo 没有空心圆柱 primitive，用 N 块切向 box 拼多边形环，内切圆
    半径 = 棒料半径 + 单边间隙。
    掩码刻意用 (contype=0, conaffinity=2) 而不是卡爪的 CT_FIXTURE(4,2)：
    工件是 (2,3)，(2 & 2) 相碰；机器人本体 (8,12)、指垫 (1,3)、
    机床 (8,1) 的 contype 都不含 bit1，一概不碰。第一次试 CT_FIXTURE
    时 4 & 12 = 4，机械臂搬运途中撞上导套，IK 残差 27.7 mm、料直接
    掉到桌面（夹紧前同轴 55.8 mm / 倾斜 56 deg）。

    但即使换成 (0,2)，常驻环仍然横在搬运路径上：棒料抬升/横移途中撞到
    导套外壁，夹紧前同轴 51.4 mm、倾斜 74.7 deg；扫 5 档轴向位置全 FAIL。
    零碰撞对照能逐位复现基线，确证是碰撞而非几何本身。真机中心架也是
    送料时张开、棒料到位后合上，所以 12 块几何构建期就带 (0,2) 掩码，
    初始沿径向张开 40 mm（不碰棒料），由 close_bushing() 在进给到位时
    把 geom_pos 移回闭合位，之后一路不撤销。
    坑：掩码不能运行时改 —— 编译期 (0,0) 的 geom 即使后来把
    model.geom_conaffinity 写成 2，mj_step 也不产生接触（探针 ncon=0）；
    而构建期就是 (0,2) 时同样摆位立刻 ncon=12。位置是每步从
    model.geom_pos 重算的，所以改用移动几何来「合上」。
    """
    BUSHING_CLOSED.clear()
    BUSHING_OPENED.clear()
    lx, ly, lz = LATHE_ORIGIN
    r_in = F.SHAFT_D / 2 + BUSHING_GAP
    r_out = r_in + BUSHING_WALL
    r_mid = (r_in + r_out) / 2
    x_front = lx - SM.JAW_LEN - BUSHING_FRONT
    xc = x_front - BUSHING_LEN / 2
    # 切向半宽按内切多边形算，+2% 让相邻块在角上重叠、不留缝。
    half_tan = r_in * np.tan(np.pi / BUSHING_N) * 1.02
    n = 0
    for k in range(BUSHING_N):
        ang = 2.0 * np.pi * (k + 0.5) / BUSHING_N
        cy, cz = np.cos(ang), np.sin(ang)
        if BUSHING_MODE == "lower" and cz > 0.0:
            continue
        # local x -> world x(轴向), local z -> 径向, local y -> 切向
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.sin(ang), cy],
                      [0.0, -cy, cz]])
        g = spec.worldbody.add_geom()
        g.name = "bushing_%02d" % k
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [BUSHING_LEN / 2, half_tan, BUSHING_WALL / 2]
        closed = np.array([xc, ly + r_mid * cy, lz + r_mid * cz])
        BUSHING_CLOSED[g.name] = closed
        opened = closed + BUSHING_OPEN * np.array([0.0, cy, cz])
        BUSHING_OPENED[g.name] = opened
        g.pos = closed if BUSHING_ALWAYS else opened
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
        g.quat = q
        g.rgba = [0.30, 0.42, 0.52, 0.95]
        g.friction = [BUSHING_MU, BUSHING_MU * 0.1, 0.0001]
        # BUSHING_STIFF=1 时把接触做硬（默认 MuJoCo 接触对 0.197 kg
        # 的料偏软，只顶回 9~17 N）。但硬顶会与三爪抢料：实测导套
        # 顶到 60 N、峰值 250 N，反而把料别到 7.24 mm / 8.6 deg。
        if BUSHING_STIFF:
            g.solref = [0.001, 1.0]
            g.solimp = [0.95, 0.999, 0.0001, 0.5, 100.0]
        g.contype, g.conaffinity = (0, 2)   # 只接工件；张开靠位置
        n += 1
    print("    bushing: %s R_in=%.2f gap=%.2f mm "
          "x=[%.0f,%.0f] mm from jaw_front  %d segs"
          % (BUSHING_MODE, r_in * 1000, BUSHING_GAP * 1000,
             -(BUSHING_FRONT + BUSHING_LEN) * 1000,
             -BUSHING_FRONT * 1000, n))


def build_model(grip_fix=None, v_jaw=False, lift=False, arc_jaw=False,
                machine_solid=True, official_world=False):
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
                         with_shaft_flat=False,
                         machine_solid=machine_solid,
                         official_world=official_world,
                         arm_kv=FEED_ARM_KV)
    if lift:
        _lift_workpiece(spec)
    # 官方 env 场景是另一套世界（自有 timestep / 相机 / 几何），导套按
    # 主场景 LATHE_ORIGIN 标定，放进去会把料别歪（实测 3.20 mm / 7.2 deg、
    # 余量 8.17 FAIL）。它保持旧交接顺序，作为独立对照。
    if BUSHING and not official_world:
        _install_bushing(spec)
    if arc_jaw:
        import arc_jaws
        arc_jaws.install(spec)
    elif v_jaw:
        _install_v_jaws(spec)
    if grip_fix:
        add_grip_fix(spec, grip_fix)
    model = spec.compile()
    contact_solref_time = os.environ.get("FEED_V_CONTACT_SOLREF_TIME")
    if contact_solref_time is not None:
        if not v_jaw:
            raise ValueError(
                "FEED_V_CONTACT_SOLREF_TIME requires --v-jaw")
        changed = _apply_v_contact_solref_time(
            model, contact_solref_time)
        print(
            "  接触敏感性探索  shaft + 4 V 爪 solref 时间常数 = %.4f s"
            " (%s)"
            % (float(contact_solref_time), ", ".join(changed)))
    return model


FRAME_EVERY = 60         # 每 N 步采一帧；仅 frame_hook 非 None 时生效
# FEED_DEBUG=1 打印 move3 每一轮的误差序列。用来判断"手臂一晃一晃"
# 到底是伺服跟不上，还是闭环在来回摆（误差反号就是摆）。
DEBUG_MOVE = os.environ.get("FEED_DEBUG") == "1"
GRIP_SETTLE = 120        # 每次 move3 前先把夹爪指令落实的步数
GRIP_RAMP = int(os.environ.get("GRIP_RAMP", 400))   # 夹爪闭合斜坡步数（400 = 0.80 s）
FEED_ITERS = int(os.environ.get("FEED_ITERS", 3))   # 进给闭环修正轮数
FEED_GAIN = float(os.environ.get("FEED_GAIN", 0.8))  # 闭环修正增益
FEED_STEPS = int(os.environ.get("FEED_STEPS", 600))  # 进给每轮步数（1.2 s，慢到不激起工件摆动）
MIN_FIX_STEPS = 150      # 闭环修正轮的最短时长（0.30 s）：再短伺服落不了地

# ---- 送料场景专用伺服阻尼（文献 B：hunting 是欠阻尼，先加阻尼）----
# 全局 ARM_SERVO_KV 必须保持 2.6，否则 demo_09 搬运到槽上方 xy 误差会从
# 4.0 涨到 5.4~6.6 mm（要求 <5），回归掉到 8/9。高阻尼只在本场景注入。
# 实测 5.0：末端抖动 RMS 0.046->0.031 mm、最大 0.878->0.377 mm，
# 进给段重新起步 5->3 次，最终同轴 1.26->0.87 mm、余量 1.55->1.05 mm。
FEED_ARM_KV = float(os.environ.get("FEED_ARM_KV", "5.0"))

# ---- 交接参数（2026-11-07 实测定案，见 CURRENT.md）----
# 逐帧跟踪发现：进给到位时料在轴线上（同轴 0.08 mm），但夹紧前停等
# 100 步期间机械臂位置伺服还在补 2.6 mm 的 IK 残差，会把料慢慢带到
# 2.55 mm。停等取消，进给结束立即交接。
CLAMP_SETTLE = int(os.environ.get("FEED_CLAMP_SETTLE", "0"))
# 夹紧期手臂柔顺：把 5 个关节的位置目标逐步拉向实际角，手臂不再刚性
# 顶着料，三爪能带着料找正。纯浮动（gain=1）会让料掉 42 mm；0.01 是
# 实测「让得动、又撑得住」的点。夹紧完成后恢复刚性，否则松爪/退出段
# 手臂继续软塌，会把刚定心好的料又拖歪（实测退出后劣化到 5 deg）。
# 用可变容器存，让嵌套的 float_arm() 能读到运行时改动。
CLAMP_FLOAT_GAIN = float(os.environ.get("FEED_CLAMP_FLOAT", "0.01"))
_FLOAT = [CLAMP_FLOAT_GAIN]
# ---- 交接期「机械臂本体」零力柔顺（2026-11-07 阶段 B 新方案）----
# FANUC soft float / 重力补偿零力控制：对机器人 5 个关节施加
#   qfrc_applied[dof] = -qfrc_bias[dof]
# 抵消机器人自身的重力与科氏项，位置伺服目标同时跟随实际角，手臂对外
# 表现为「自由 + 阻尼」。三点与已否掉的对中弹簧有本质区别：
#   1) 力加在【机器人关节】上，不作用在棒料上；
#   2) 目标取机器人自身的动力学偏差，不含任何主轴轴线信息；
#   3) 夹爪保持夹持，棒料由真实接触托住，不需要给棒料加 mg 托力。
# 这是机器人本体的控制模式（真机伺服的标准功能），不是替它对中的力场。
ARM_GCOMP = os.environ.get("FEED_ARM_GCOMP", "0") == "1"
ARM_GCOMP_GAIN = float(os.environ.get("FEED_ARM_GCOMP_GAIN", "1.0"))
# 负载重力补偿：机器人关节按 J^T·(0,0,+mg) 主动承担棒料重量，力最终
# 经夹爪【接触】传给棒料，棒料 body 上没有任何 qfrc_applied。真机伺服
# 的 payload gravity compensation / 负载辨识就是这个，属于机械臂自身
# 控制，不是给工件加的隐藏力场。0 = 不补，1 = 全额。
ARM_PAYLOAD = float(os.environ.get("FEED_ARM_PAYLOAD", "0"))
# 交接期位置伺服刚度倍数：SO-101 出厂 kp=17.8，挂 0.2 kg 悬臂料会明显
# 变形（实测夹爪带着料漂 3~5 mm）。真机可切高刚度模式，这属于伺服参数，
# 不施加任何外力。>1 时同步保持 kv 不变（阻尼比会降，注意 hunting）。
ARM_KP_SCALE = float(os.environ.get("FEED_ARM_KP", "1"))
# 夹紧期逐帧跟踪，用来分清「手臂硬顶」还是「夹爪带着料跑」。默认关。
CLAMP_TRACE = os.environ.get("FEED_CLAMP_TRACE", "0") == "1"
# 落座诊断：打印 V 槽两指间距，看「料在槽里滑」还是「夹爪被撑开」。
SEAT_TRACE = os.environ.get("FEED_SEAT_TRACE", "0") == "1"
GRIP_FORCE_NM = float(os.environ.get("FEED_GRIP_FORCE_NM", "0"))
GRIP_KP_SCALE = float(os.environ.get("FEED_GRIP_KP", "1"))
# 进给到位后把位置伺服目标一次性同步到实际角。逐帧跟踪发现：夹紧斜坡
# 前 300 步卡爪还没碰到料，夹爪却带着料漂了 3.3 mm —— 根源是伺服还在
# 追进给终点 2.5 mm 的 IK 残差。锚定让手臂停在【实际到位姿态】，
# 不再追一个它到不了的旧目标。纯机械臂侧动作，不施加任何外力。
CLAMP_ANCHOR = os.environ.get("FEED_CLAMP_ANCHOR", "1") == "1"
# 交接期托住棒料：抵消自重，等效真机长棒料送料的导套/托料架。料只有
# 25 mm 插在爪里、55 mm 悬空，夹爪一让位就会自由落体（实测掉 6~12 mm，
# 正是肉眼看到的「下沉又回来一点」）。解锁前必须撤销，否则下沉指标假通过。
# 竖直抗重力托力比例：1 = 全额抵消自重（旧默认），0 = 不托。
# 2026-11-07 定案改为 0：逐帧跟踪证实这根 +z 托力会沿斜 V 槽把料顶滑
# 3.3 mm（料与夹爪同步横移），是「往左上怼」的主要来源之一。改由真实
# 下半环托架承担防坠，棒料 qfrc 峰值从 1.933 N 降到 0。
# 只在真实交接路径（tuned）生效，历史基线/官方场景仍按旧行为托住。
CLAMP_LIFT = float(os.environ.get("FEED_CLAMP_LIFT", "0"))

# ---- 2026-11-07 文献/论坛对照：交接顺序（第二轮已设为默认）----
# 机械社区论坛原话：「机器人送入，然后卡盘夹紧，然后机器人再松开。造成
# 两者交替时，有一个两者都夹紧的状态，较劲」。FANUC 软浮动专利
# DE69837741T2 与 CNC 上下料手册都要求：卡爪闭合【之前】先让夹爪松开，
# 但松开的料必须靠端面挡块/弹簧推料器顶住，否则会掉。
# 1 = 先松夹爪（手臂原地、soft floating）-> 端面轴向定位 -> 卡爪闭合。
# 单独用会 FAIL（撤支撑后跷 2.2 deg）；配实体硬导套 + 径向弹簧后 PASS，
# 但那组结果已被用户否掉（作弊弹簧替机械臂对中），故默认回退为关。
# 注意 tuned 依赖「有导套几何」：FEED_BUSHING=0 会把整组一起短路。
HANDOVER_FIRST = os.environ.get("FEED_HANDOVER_FIRST", "0") == "1"
# 轴向弹性定位（等效端面挡块 + 弹簧推料器），只在交接期生效，径向完全
# 放开让三爪自由找正。棒料约 0.197 kg，k=300 N/m 时临界阻尼约 15 N·s/m。
SEAT_K = float(os.environ.get("FEED_SEAT_K", "300"))
SEAT_C = float(os.environ.get("FEED_SEAT_C", "12"))
SEAT_DWELL = int(os.environ.get("FEED_SEAT_DWELL", "200"))
# 【已否掉】径向约束：qfrc_applied 数学弹簧，目标直接取理论主轴轴线
# axis[1]/axis[2]。拆开对照：关掉它、保留导套，导套反以 483 N 把料别到
# 14.2 deg FAIL，证明是这根弹簧在替机械臂对中，而不是实体几何。真机没有
# 「知道轴线在哪」的力场，故默认 0。保留开关仅供对照，勿重新设为默认。
SEAT_RADIAL_K = float(os.environ.get("FEED_SEAT_RADIAL_K", "0"))
# 0 = 按临界阻尼自动取 1.4*sqrt(k*m)，避免为每档刚度再调阻尼。
SEAT_RADIAL_C = float(os.environ.get("FEED_SEAT_RADIAL_C", "0"))
# 夹紧完成后径向支撑的渐变卸载步数。0 = 一步清零（旧行为）。
# 上一轮实测：k=5000 时夹紧瞬态只有 z −0.42 mm（默认 +2.81），但一步
# 撤销会把弹簧储能瞬间释放，把料弹飞到 10.27 mm / 13.3 deg。按 quintic
# 把刚度和阻尼一起降到 0，让三爪逐步接手全部径向载荷。
# 旧定案用 400 步；随径向弹簧一起回退为 0。fade 段日志里的「卸载托力」
# 指撤销这层【力等效托力】，不是撤掉常驻的实体导套几何（那个从不撤销）。
SEAT_FADE_STEPS = int(os.environ.get("FEED_SEAT_FADE", "0"))

# ---- 文献 C：低压预夹 -> 端面顶正 -> 高压终夹（默认关闭）----
# 与已否掉的 CLAMP_TWOSTAGE 不同：那是把一段位移斜坡拆慢；这里中间真的
# 插入一次「料还能微动时的端面顶正」，且预夹段用较浅的丝杠目标给出低压。
# 对应 Practical Machinist 的「刀塔弹簧推料器在循环中把料重新顶正 + M
# 代码重新夹紧」。0 = 单段夹紧（当前默认）。
PRECLAMP = os.environ.get("FEED_PRECLAMP", "0") == "1"
PRECLAMP_CTRL = float(os.environ.get("FEED_PRECLAMP_CTRL", "-0.017"))
PRECLAMP_STEPS = int(os.environ.get("FEED_PRECLAMP_STEPS", "300"))
PRECLAMP_DWELL = int(os.environ.get("FEED_PRECLAMP_DWELL", "200"))
PRECLAMP_SEAT_K = float(os.environ.get("FEED_PRECLAMP_SEAT_K", "300"))
PRECLAMP_SEAT_C = float(os.environ.get("FEED_PRECLAMP_SEAT_C", "12"))

# ---- 文献 B：「吨吨吨」= 接触段欠阻尼 hunting（默认关闭）----
# move()/move3() 原来只写纯位置指令，没有 move_path() 已有的速度前馈。
# 速度前馈 = kv/kp 抵消匀速段跟踪滞后；速度误差阻尼抑制接触时反复回弹。
# 文献明确：振荡是欠阻尼，先加阻尼，别加比例增益。
ARM_MOVE_LEAD = float(os.environ.get("FEED_ARM_LEAD", "0"))
ARM_MOVE_DAMP = float(os.environ.get("FEED_ARM_DAMP", "0"))
# 夹紧期柔顺跟随原是纯比例（g*(qpos-ctrl)），只有跟随没有阻尼，接触时
# 容易 hunting。给跟随目标叠一个与关节速度反向的阻尼项。0 = 保持原样。
FLOAT_DAMP = float(os.environ.get("FEED_FLOAT_DAMP", "0"))
# 夹紧前让伺服在【不拖拽工件】的前提下收敛：目标跟随实际角，只等关节
# 速度降下来，不空等固定步数（固定停等 100 步会把料从 0.08 带到 2.55 mm）。
CONVERGE_STEPS = int(os.environ.get("FEED_CONVERGE", "0"))
CONVERGE_TOL = float(os.environ.get("FEED_CONVERGE_TOL", "0.02"))

# 打印夹紧稳定时三个滑座的实际 qpos，用于标定受载自镗位置。
BORE_PROBE = os.environ.get("FEED_BORE_PROBE", "0") == "1"

# SMOOTH=1 走「一条连续轨迹」：抬升 -> 横移 -> 下降 -> 进给 合并成一次
# Ruckig 规划，中途不减速、不停车。0 = 老的分段 rest-to-rest 路径。
SMOOTH_PATH = os.environ.get("SMOOTH", "1") == "1"
# 位置伺服的速度前馈系数 = kv/kp（秒）。纯 PD 在匀速段的跟踪误差是
# (kv/kp)*qdot，把参考速度按这个比例提前量加进指令即可补偿。
# 注意：这里刻意用【全局】ARM_SERVO_KV(2.6)，不是本场景注入的 5.0。
# 2026-11-07 的 5.0 阻尼扫描正是在「伺服 kv=5 + 前馈 2.6/70」这个
# 组合下测出 RMS 0.031 mm 的；把前馈也改成 5/70 是未验证的变化，
# 不要为了公式好看顺手统一。
ARM_LEAD = F.ARM_SERVO_KV / F.ARM_SERVO_KP


def main(frame_hook=None, model=None, data=None, carry_lock=True,
         v_jaw=False, lift=False, wrist=None, machine_solid=True,
         initial_offset_mm=(0.0, 0.0, 0.0), initial_roll_deg=None,
         random_seed=None, initial_yaw_deg=None, planning_mode="M1",
         m2_error_mm=(0.0, 0.0, 0.0), m2_error_yaw_deg=0.0,
         initial_settle_steps=DEFAULT_INITIAL_SETTLE_STEPS):
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
    if type(initial_settle_steps) is not int or initial_settle_steps < 0:
        raise ValueError("initial_settle_steps must be a nonnegative integer")
    # Flat pads tolerate any wrist angle; a V groove only closes properly
    # when the jaw axis is square to the rod. Keep the stock value for the
    # verified 7/7 path and switch automatically for the V-jaw experiment.
    if wrist is None:
        wrist = V_WRIST if v_jaw else WRIST_ROLL_GRASP
    grip_close = V_GRIP_CLOSE if v_jaw else GRIP_CLOSE
    # 交接期夹持力可调：V_GRIP_CLOSE 是抓取段标定值，交接期料要悬空
    # 55 mm，夹持不够会让料在 V 槽里缓慢下滑（实测无托力时下沉 6.9 mm）。
    _gc = os.environ.get("FEED_GRIP_CLOSE")
    if _gc is not None:
        grip_close = float(_gc)
    model = model if model is not None else build_model(
        v_jaw=v_jaw, lift=lift, machine_solid=machine_solid)
    data = data if data is not None else mujoco.MjData(model)

    def tick(tag=None):
        if tag is not None and "stage_records" in state:
            record_stage(tag)
        if frame_hook is not None:
            frame_hook(tag, model, data)

    def gid(kind, name):
        return mujoco.mj_name2id(model, kind, name)

    jaw_act = gid(mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
    shaft_body = gid(mujoco.mjtObj.mjOBJ_BODY, "shaft")
    shaft_joint = gid(mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
    sq = model.jnt_qposadr[shaft_joint]
    sv = model.jnt_dofadr[shaft_joint]
    # 卡爪前端面：工件从 -x 侧进来，先碰到这个面。
    jaw_front = LATHE_ORIGIN[0] - SM.JAW_LEN
    # 工件「中心」的目标位置由「进多少」反推，不再写死 -0.016。
    #   头端 x = jaw_front + INSERT_MM
    #   中心 x = 头端 - SHAFT_L/2
    axis = np.array([jaw_front + INSERT_MM - F.SHAFT_L / 2,
                     LATHE_ORIGIN[1], LATHE_ORIGIN[2]])
    # 对轴工位：工件整体停在卡爪外面，头端离卡爪前端面 APPROACH_MM。
    stage_x = jaw_front - F.SHAFT_L / 2 - APPROACH_MM
    shaft_geoms = [
        i for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        .startswith("shaft_")
    ]
    initial_support_geom_ids_by_side = {
        side: {
            i for i in range(model.ngeom)
            if (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            .startswith("cradle_%s" % side)
        }
        for side in ("L", "R")
    }
    robot_geom_ids, machine_geom_ids = _robot_machine_geom_ids(model)
    contact_geom_sets = _semantic_contact_geom_sets(model, robot_geom_ids)
    bushing_ids = [
        i for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        .startswith("bushing_")
    ]
    # 定案交接（实体导套 + 先松爪 + 渐变卸载）只给「真实交接的全物理
    # V 槽路径」。历史基线 / 官方 env 场景是独立对照，保持旧行为。
    tuned = bool(bushing_ids) and v_jaw and lift and not carry_lock
    # 新交接行为（去托力、锚定、托架）只给真实交接路径；历史基线与官方
    # env 场景保持旧行为，作为独立对照。
    _lift = CLAMP_LIFT if tuned else 1.0
    no_retreat_diagnostic = (
        os.environ.get("FEED_DIAG_NO_RETREAT") == "1")
    diagnostic_mode = (
        "stationary_after_release" if no_retreat_diagnostic
        else "no_axial_retreat" if not RETREAT_AXIAL else "standard")
    primary_protocol = (
        tuned and machine_solid and diagnostic_mode == "standard")
    evaluation_scope = (
        "primary_physics_feed" if primary_protocol
        else "kinematic_lock_reference" if carry_lock
        else "diagnostic_configuration")

    def close_bushing():
        """中心架在棒料送到位后闭合，之后一路不再打开。

        与「夹紧后要撤销的力等效支撑」不同：合上就是实体接触，交接、
        松爪、退出、解锁静置、拉拔全程都在，只是不参与搬运路径。
        model.geom_pos 每步被 mj_kinematics 重算，所以移动几何有效；
        掩码则不行（见 _install_bushing 的注释）。
        """
        if not bushing_ids or not BUSHING_CLOSED:
            return
        if BUSHING_ALWAYS:
            print("  导套常驻    %d 块 间隙 %.2f mm 同轴=%.2f mm 接触=%d"
                  % (len(bushing_ids), BUSHING_GAP * 1000, coax(),
                     sleeve_state()[0]))
            return
        for i in bushing_ids:
            name = (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            if name in BUSHING_CLOSED:
                model.geom_pos[i][:] = BUSHING_CLOSED[name]
        mujoco.mj_forward(model, data)
        print("  导套闭合    %d 块 间隙 %.2f mm 同轴=%.2f mm 接触=%d"
              % (len(bushing_ids), BUSHING_GAP * 1000, coax(),
                 sleeve_state()[0]))

    def release_bushing():
        """三爪夹住后把托架退回张开位，避免与卡爪过约束。"""
        if not BUSHING_RELEASE or not bushing_ids or not BUSHING_OPENED:
            return
        for i in bushing_ids:
            name = (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
            if name in BUSHING_OPENED:
                model.geom_pos[i][:] = BUSHING_OPENED[name]
        mujoco.mj_forward(model, data)
        print("  托架退回    间隙 %.2f mm 同轴=%.2f mm 接触=%d"
              % (BUSHING_GAP * 1000, coax(), sleeve_state()[0]))

    # Negative contact distance indicates overlap between the collision shapes.
    PEN = {"min": 0.0, "label": "", "pair": ""}
    penetration_sample_times = {}

    _cf6 = np.zeros(6)

    def contact_geom_parameters(geom_id):
        return {
            "name": (
                mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "?"),
            "type_id": int(model.geom_type[geom_id]),
            "size_m": [
                float(value) for value in model.geom_size[geom_id]],
            "solref": [
                float(value) for value in model.geom_solref[geom_id]],
            "solimp": [
                float(value) for value in model.geom_solimp[geom_id]],
            "margin_m": float(model.geom_margin[geom_id]),
            "gap_m": float(model.geom_gap[geom_id]),
            "friction": [
                float(value) for value in model.geom_friction[geom_id]],
        }

    def track_pen(label):
        ensure_finite_state(label)
        _fmax = 0.0
        machine_contacts = []
        shaft_contacts = 0
        shaft_pair_samples = {}
        for i in range(data.ncon):
            c = data.contact[i]
            n1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
            n2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
            if _is_robot_machine_contact(
                    c.geom1, c.geom2, robot_geom_ids, machine_geom_ids):
                mujoco.mj_contactForce(model, data, i, _cf6)
                machine_contacts.append({
                    "geoms": [n1, n2], "normal_force_N": abs(float(_cf6[0]))})
            if any(g_ in (c.geom1, c.geom2) for g_ in shaft_geoms):
                shaft_contacts += 1
            if not any(g_ in (c.geom1, c.geom2) for g_ in shaft_geoms):
                continue
            mujoco.mj_contactForce(model, data, i, _cf6)
            _fmax = max(_fmax, abs(float(_cf6[0])))
            pair_key = "%s|%s" % (n1, n2)
            pair_sample = shaft_pair_samples.setdefault(pair_key, {
                "geoms": [n1, n2],
                "geom_ids": [int(c.geom1), int(c.geom2)],
                "contact_count": 0,
                "min_distance_m": float(c.dist),
            })
            pair_sample["contact_count"] += 1
            pair_sample["min_distance_m"] = min(
                pair_sample["min_distance_m"], float(c.dist))
            if c.dist < PEN["min"]:
                PEN["min"] = float(c.dist)
                PEN["label"] = label
                PEN["pair"] = pair_key
                shaft_geom = (
                    c.geom1 if c.geom1 in shaft_geoms else c.geom2)
                other_geom = (
                    c.geom2 if c.geom1 in shaft_geoms else c.geom1)
                other_body = int(model.geom_bodyid[other_geom])
                other_rotation = data.geom_xmat[other_geom].reshape(3, 3)
                shaft_rotation = data.geom_xmat[shaft_geom].reshape(3, 3)
                other_quaternion = np.zeros(4, dtype=float)
                mujoco.mju_mat2Quat(
                    other_quaternion,
                    np.ascontiguousarray(other_rotation).reshape(-1))
                state["penetration_diagnostics"]["minimum_contact"] = {
                    "distance_m": float(c.dist),
                    "time_s": float(data.time),
                    "phase": label,
                    "geoms": [n1, n2],
                    "contact_geometry_parameters": [
                        contact_geom_parameters(int(c.geom1)),
                        contact_geom_parameters(int(c.geom2)),
                    ],
                    "contact_position_world_m": [
                        float(value) for value in c.pos],
                    "contact_normal_world_unit": [
                        float(value) for value in c.frame[:3]],
                    "normal_force_N": abs(float(_cf6[0])),
                    "shaft_position_world_m": [
                        float(value) for value in data.xpos[shaft_body]],
                    "shaft_quaternion_wxyz": [
                        float(value) for value in data.xquat[shaft_body]],
                    "contact_position_shaft_geom_local_m": [
                        float(value) for value in
                        shaft_rotation.T
                        @ (c.pos - data.geom_xpos[shaft_geom])],
                    "other_geom": {
                        "type_id": int(model.geom_type[other_geom]),
                        "size_m": [
                            float(value)
                            for value in model.geom_size[other_geom]],
                        "position_world_m": [
                            float(value)
                            for value in data.geom_xpos[other_geom]],
                        "contact_position_local_m": [
                            float(value) for value in
                            other_rotation.T
                            @ (c.pos - data.geom_xpos[other_geom])],
                        "quaternion_wxyz": [
                            float(value) for value in other_quaternion],
                        "body": (
                            mujoco.mj_id2name(
                                model, mujoco.mjtObj.mjOBJ_BODY, other_body)
                            or "?"),
                    },
                }
        phase_pairs = state["penetration_diagnostics"][
            "by_phase"].setdefault(label, {})
        for pair_key, pair_sample in shaft_pair_samples.items():
            pair_summary = phase_pairs.setdefault(pair_key, {
                "geoms": pair_sample["geoms"],
                "contact_geometry_parameters": [
                    contact_geom_parameters(geom_id)
                    for geom_id in pair_sample["geom_ids"]],
                "contact_steps": 0,
                "overlap_steps": 0,
                "overlap_gt_0_2mm_steps": 0,
                "overlap_gt_1mm_steps": 0,
                "max_contacts_per_step": 0,
                "max_penetration_mm": 0.0,
                "minimum_contact_distance_m": 0.0,
                "first_contact_time_s": None,
                "last_contact_time_s": None,
                "observed_contact_span_s": 0.0,
                "sampled_contact_duration_s": 0.0,
                "time_of_minimum_s": None,
            })
            pair_summary["max_contacts_per_step"] = max(
                pair_summary["max_contacts_per_step"],
                pair_sample["contact_count"])
            if (pair_sample["min_distance_m"]
                    < pair_summary["minimum_contact_distance_m"]):
                pair_summary["minimum_contact_distance_m"] = (
                    pair_sample["min_distance_m"])
                pair_summary["time_of_minimum_s"] = float(data.time)
            if pair_sample["min_distance_m"] < 0.0:
                depth_mm = -pair_sample["min_distance_m"] * 1000.0
                pair_summary["max_penetration_mm"] = max(
                    pair_summary["max_penetration_mm"], depth_mm)
            sample_key = (label, pair_key)
            last_sample_time = penetration_sample_times.get(sample_key)
            if (last_sample_time is None
                    or abs(last_sample_time - float(data.time)) > 1e-12):
                pair_summary["contact_steps"] += 1
                if pair_summary["first_contact_time_s"] is None:
                    pair_summary["first_contact_time_s"] = float(data.time)
                pair_summary["last_contact_time_s"] = float(data.time)
                pair_summary["observed_contact_span_s"] = (
                    pair_summary["last_contact_time_s"]
                    - pair_summary["first_contact_time_s"])
                pair_summary["sampled_contact_duration_s"] = (
                    pair_summary["contact_steps"]
                    * float(model.opt.timestep))
                if pair_sample["min_distance_m"] < 0.0:
                    pair_summary["overlap_steps"] += 1
                if pair_sample["min_distance_m"] < -0.0002:
                    pair_summary["overlap_gt_0_2mm_steps"] += 1
                if pair_sample["min_distance_m"] < -0.001:
                    pair_summary["overlap_gt_1mm_steps"] += 1
                penetration_sample_times[sample_key] = float(data.time)
        state["robot_machine_contact_max"] = max(
            state["robot_machine_contact_max"], len(machine_contacts))
        state["shaft_contact_max"] = max(state["shaft_contact_max"],
                                           shaft_contacts)
        phase = state.setdefault("phase_stats", {}).setdefault(label, {
            "samples": 0, "robot_machine_contact_max": 0,
            "shaft_contact_max": 0, "rod_external_force_max_N": 0.0,
            "robot_machine_pairs": {},
        })
        phase["samples"] += 1
        phase["robot_machine_contact_max"] = max(
            phase["robot_machine_contact_max"], len(machine_contacts))
        phase["shaft_contact_max"] = max(phase["shaft_contact_max"],
                                           shaft_contacts)
        _external_force = float(np.max(np.abs(data.qfrc_applied[sv:sv + 3])))
        phase["rod_external_force_max_N"] = max(
            phase["rod_external_force_max_N"], _external_force)
        state["rod_force_max"] = max(state["rod_force_max"], _external_force)
        for pair in machine_contacts:
            key = "|".join(pair["geoms"])
            phase["robot_machine_pairs"][key] = max(
                phase["robot_machine_pairs"].get(key, 0.0),
                pair["normal_force_N"])
        state.setdefault("rod_p", []).append(data.qpos[sq:sq + 3].copy())
        state.setdefault("rod_f", []).append(_fmax)
        state.setdefault("rod_lab", []).append(label)

    def shaft():
        return data.qpos[sq:sq + 3].copy()

    def shaft_pose_snapshot():
        shaft_axis = data.xmat[shaft_body].reshape(3, 3)[:, 2]
        shaft_axis = shaft_axis / max(
            float(np.linalg.norm(shaft_axis)), 1e-12)
        return {
            "position_m": [float(x) for x in shaft()],
            "quaternion_wxyz": [
                float(x) for x in data.qpos[sq + 3:sq + 7]],
            "shaft_axis_world_unit": [
                float(x) for x in shaft_axis],
        }

    def coax():
        p = shaft()
        return float(np.hypot(p[1] - axis[1], p[2] - axis[2]) * 1000)

    def stock_ecc():
        """整根棒料相对主轴轴线的最大径向偏心（mm），取两端点较坏值。

        同轴度只量中心，倾斜的那一段会漏掉；车外圆时真正决定「第一刀
        能不能见光」的是最外端的偏心。夹紧后料与主轴刚性固连，车出来
        的表面自动同轴，所以这里只判断余量够不够，不是成品精度。
        """
        p = shaft()
        a = data.xmat[shaft_body].reshape(3, 3)[:, 2]
        e = 0.0
        for s in (-1.0, 1.0):
            q = p + s * (F.SHAFT_L / 2) * a
            e = max(e, float(np.hypot(q[1] - axis[1], q[2] - axis[2])))
        return e * 1000

    def axdev():
        """Angle between the ROD AXIS and world x -- not the quaternion.

        A round part that merely rolls in the V groove reports tens of degrees
        of "tilt" from tilt_deg() while its axis never moved, which sent two
        earlier debugging rounds after a non-existent problem.
        """
        ax = data.xmat[shaft_body].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(min(1.0, abs(float(ax[0]))))))

    def rod_pitch():
        """带符号的轴线仰角：正 = 头端（+x，插卡盘那头）比尾端高。

        axdev() 只给绝对值，看不出"翘向哪头"。验收看姿态时必须带符号，
        否则料是头端上翘还是尾端上翘分不清。
        """
        a = data.xmat[shaft_body].reshape(3, 3)[:, 2].astype(float)
        if a[0] < 0:
            a = -a
        return float(np.degrees(np.arctan2(a[2], max(1e-9, abs(a[0])))))

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

    def sleeve_state():
        """实体导套与棒料的接触数/法向力：径向约束是否真的在受力。"""
        n, f = 0, 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if not any(g in (c.geom1, c.geom2) for g in shaft_geoms):
                continue
            n1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if "bushing_" not in (n1 + "|" + n2):
                continue
            n += 1
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            f += abs(f6[0])
        return n, f

    def grip_state():
        """V-blade contacts on the rod and their total normal force."""
        count, force = 0, 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if not any(g in (c.geom1, c.geom2) for g in shaft_geoms):
                continue
            n1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            n2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if "_v1" not in n1 + n2 and "_v2" not in n1 + n2:
                continue
            count += 1
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f6)
            force += abs(f6[0])
        return count, force

    arm = Arm(model, data)
    started_at = time.perf_counter()
    case_id = (os.environ.get("FEED_CASE_ID") or uuid.uuid4().hex[:12])
    if random_seed is None:
        seed_env = os.environ.get("FEED_RANDOM_SEED")
        random_seed = None if seed_env in (None, "") else int(seed_env)
    offset_mm = np.asarray(initial_offset_mm, dtype=float)
    if offset_mm.shape != (3,):
        raise ValueError("initial_offset_mm must have three values: dx, dy, dz")
    legacy_roll_deg = (
        0.0 if initial_roll_deg is None else float(initial_roll_deg))
    yaw_world_z_deg = _resolve_initial_yaw_deg(
        legacy_roll_deg, initial_yaw_deg)
    if (not np.all(np.isfinite(offset_mm))
            or not np.isfinite(yaw_world_z_deg)):
        raise ValueError("initial pose values must be finite")
    planning_mode = str(planning_mode).upper()
    if planning_mode not in PLANNING_MODES:
        raise ValueError(
            "planning_mode must be one of %s, got %r"
            % ("/".join(PLANNING_MODES), planning_mode))
    m2_error_mm = np.asarray(m2_error_mm, dtype=float)
    if m2_error_mm.shape != (3,) or not np.all(np.isfinite(m2_error_mm)):
        raise ValueError("m2_error_mm must contain three finite values")
    if not np.isfinite(float(m2_error_yaw_deg)):
        raise ValueError("m2_error_yaw_deg must be finite")
    _KP0 = [float(model.actuator_gainprm[a, 0]) for a in arm.qadr]
    # 夹爪执行器力矩上限（N·m）。SO-101 出厂 ±3.35，挂 80 mm 料时料重经
    # V 槽楔形放大，实测把两指撑开 3.8 mm、料跟着下沉 1.4 mm。这里只是
    # 把伺服力限调大做「如果夹爪更硬」的对照，不改任何几何或外力。
    if GRIP_FORCE_NM > 0:
        model.actuator_forcerange[arm.grip_act] = [-GRIP_FORCE_NM,
                                                  GRIP_FORCE_NM]
        print("  夹爪力限      %.2f -> %.2f N·m（诊断用）"
              % (3.35, GRIP_FORCE_NM))
    if GRIP_KP_SCALE != 1.0:
        _gk = float(model.actuator_gainprm[arm.grip_act, 0])
        model.actuator_gainprm[arm.grip_act, 0] = _gk * GRIP_KP_SCALE
        model.actuator_biasprm[arm.grip_act, 1] = -_gk * GRIP_KP_SCALE
        print("  夹爪刚度      kp %.1f -> %.1f（诊断用）"
              % (_gk, _gk * GRIP_KP_SCALE))

    def set_arm_stiff(scale):
        """交接期改手臂位置伺服刚度。只动 servo 参数，不加任何外力。"""
        for k, a in enumerate(arm.qadr):
            model.actuator_gainprm[a, 0] = _KP0[k] * scale
            model.actuator_biasprm[a, 1] = -_KP0[k] * scale

    state = {"offset": None, "off_site": None, "abs_pos": None,
             "lock_active": True, "axis_hold": False,
             "stage_records": [], "checkpoints": [], "case_id": case_id,
             "random_seed": random_seed,
             "rod_force_max": 0.0, "robot_machine_contact_max": 0,
             "shaft_contact_max": 0, "phase_stats": {}, "ik_solver": [],
             "diagnostic_scope": "feed",
             "contact_diagnostics": {
                 "sampled_every_physics_step": True,
                 "sample_interval_s": float(model.opt.timestep),
                 "scopes": {},
             },
             "applied_load_diagnostics": {
                 "sampled_every_physics_step": True,
                 "sample_interval_s": float(model.opt.timestep),
                 "scopes": {},
             },
             "penetration_diagnostics": {
                 "sampling_method": (
                     "shaft contacts inspected by track_pen after each "
                     "simulated action step"),
                 "sample_interval_s": float(model.opt.timestep),
                 "by_phase": {},
                 "minimum_contact": None,
             },
             "pickup_motion_diagnostics": {
                 "status": "awaiting_initial_pose",
                 "reference_pose": None,
                 "measurement_scope": [
                     "接近", "侧向进入", "下降", "夹紧"],
                 "overall": {
                     "samples": 0,
                     "robot_workpiece_contact_samples": 0,
                     "peak_robot_workpiece_contacts_per_step": 0,
                     "peak_robot_workpiece_normal_force_sum_N": 0.0,
                     "max_center_displacement_mm": 0.0,
                     "center_displacement_vector_at_max_mm": [0.0, 0.0, 0.0],
                     "max_shaft_axis_change_deg": 0.0,
                 },
                 "phases": {},
                 "criterion": "diagnostic_only_no_pass_fail_threshold",
             },
             "gripper_relative_workpiece_motion": {
                 "status": "not_captured",
                 "reference_pose": None,
                 "active": False,
                 "time_history_enabled": (
                     os.environ.get("FEED_TRACE_RELATIVE_MOTION") == "1"),
                 "time_history_sample_interval_s": (
                     float(model.opt.timestep)),
                 "time_history": [],
                 "observation_start_stage": None,
                 "observation_end_stage": None,
                 "overall": {
                     "samples": 0,
                     "max_translation_slip_mm": 0.0,
                     "translation_slip_vector_tool_mm_at_max": [
                         0.0, 0.0, 0.0],
                     "max_relative_axis_change_deg": 0.0,
                 },
                 "phases": {},
                 "criterion": "diagnostic_only_no_pass_fail_threshold",
             },
             "initial_stability": None,
             "ik_residuals_mm": {}, "versions": _versions_snapshot(),
             "actual_initial_pose": None,
             "model_options_initial": _model_options_snapshot(model),
             "inputs": {
                 "workpiece": {
                     "shape": "cylinder",
                     "material": "steel",
                     "diameter_mm": float(SM.SHAFT_D * 1000),
                      "length_mm": float(SM.SHAFT_L * 1000),
                      "mass_kg": float(SM.SHAFT_MASS),
                      "density_kg_m3": float(SM.STEEL_RHO),
                      "nominal_start_position_m": None,
                      "nominal_start_position_source": (
                          "reset_model_pose_before_requested_perturbation"),
                  },
                 "requested_initial_pose": {
                     "offset_mm": [float(x) for x in offset_mm],
                     "offset_world_xyz_mm": [
                         float(x) for x in offset_mm],
                     "yaw_world_z_deg": float(yaw_world_z_deg),
                     "roll_deg": float(legacy_roll_deg),
                     "legacy_roll_argument_deg": (
                         None if initial_roll_deg is None
                         else float(initial_roll_deg)),
                 },
                 "pose_before_perturbation": None,
                 "pose_after_perturbation_before_settle": None,
                 "grasp_target_pose": None,
                 "insert_mm": float(INSERT_MM * 1000),
                 "random_seed": random_seed,
                 "planning_mode": planning_mode,
                 "m2_error_mm": [float(x) for x in m2_error_mm],
                 "m2_error_yaw_deg": float(m2_error_yaw_deg),
                 "initial_settle_steps": initial_settle_steps,
             },
             "control_mode": {
                 "planning_mode": planning_mode,
                 "planning_pose_scope": "pre_grasp_target_only",
                 "carry_feedback_policy": (
                     "planned_pose_and_arm_site_transform_only"),
                 "v_jaw": bool(v_jaw),
                 "lift": bool(lift),
                 "carry": ("kinematic_lock" if carry_lock
                           else "unlocked_physics"),
                 "machine_solid": bool(machine_solid),
                 "evaluation_scope": evaluation_scope,
                 "diagnostic_mode": diagnostic_mode,
                 "wrist_command_rad": float(wrist),
                 "grip_close_command": float(grip_close),
             },
             "effective_parameters": {
                 "feed": {
                     "insert_mm": float(INSERT_MM * 1000),
                     "approach_mm": float(APPROACH_MM * 1000),
                     "z_comp_mm": float(FEED_Z_COMP * 1000),
                     "grip_station_mm": float(GRIP_STATION * 1000),
                     "retreat_axial": bool(RETREAT_AXIAL),
                     "diagnostic_no_retreat": bool(no_retreat_diagnostic),
                     "level_rod": bool(LEVEL_ROD),
                     "axis_lead_deg": float(AXIS_LEAD_DEG),
                     "seat_tap_mm": float(SEAT_TAP * 1000),
                     "smooth_path": bool(SMOOTH_PATH),
                 },
                 "gripper": {
                     "v_angle_deg": 90.0,
                     "v_depth_mm": float(V_DEPTH * 1000),
                     "v_length_mm": float(V_LEN * 1000),
                     "v_bar_mm": float(V_BAR * 1000),
                     "v_reference_position": float(V_QREF),
                     "cradle_depth_mm": float(CRADLE_DEPTH * 1000),
                     "cradle_length_mm": float(CRADLE_LEN * 1000),
                     "cradle_half_width_mm": float(
                         CRADLE_HALF_X * 1000),
                     "cradle_bar_mm": float(CRADLE_BAR * 1000),
                     "cradle_drop_mm": float(CRADLE_DROP * 1000),
                     "workpiece_lift_mm": float(LIFT_DZ * 1000),
                     "v_wrist_rad": float(V_WRIST),
                     "v_grip_close": float(V_GRIP_CLOSE),
                     "effective_wrist_rad": float(wrist),
                     "effective_grip_close": float(grip_close),
                     "grip_ramp_steps": int(GRIP_RAMP),
                     "grip_settle_steps": int(GRIP_SETTLE),
                     "grip_force_limit_Nm": float(GRIP_FORCE_NM),
                     "grip_kp_scale": float(GRIP_KP_SCALE),
                 },
                 "bushing": {
                     "enabled": bool(bushing_ids),
                     "gap_mm": float(BUSHING_GAP * 1000),
                     "front_mm": float(BUSHING_FRONT * 1000),
                     "length_mm": float(BUSHING_LEN * 1000),
                     "segments": int(BUSHING_N),
                     "mode": BUSHING_MODE,
                     "friction": float(BUSHING_MU),
                     "wall_mm": float(BUSHING_WALL * 1000),
                     "open_mm": float(BUSHING_OPEN * 1000),
                     "stiff": bool(BUSHING_STIFF),
                     "always": bool(BUSHING_ALWAYS),
                     "release": bool(BUSHING_RELEASE),
                 },
                 "controller": {
                     "feed_iterations": int(FEED_ITERS),
                     "feed_gain": float(FEED_GAIN),
                     "feed_steps_per_iteration": int(FEED_STEPS),
                     "minimum_fix_steps": int(MIN_FIX_STEPS),
                     "feed_arm_kv": float(FEED_ARM_KV),
                     "arm_gravity_compensation": bool(ARM_GCOMP),
                     "arm_gravity_compensation_gain": float(
                         ARM_GCOMP_GAIN),
                     "arm_payload_compensation": float(ARM_PAYLOAD),
                     "arm_kp_scale": float(ARM_KP_SCALE),
                     "arm_move_lead": float(ARM_MOVE_LEAD),
                     "arm_move_damping": float(ARM_MOVE_DAMP),
                     "float_damping": float(FLOAT_DAMP),
                     "converge_steps": int(CONVERGE_STEPS),
                     "converge_tolerance": float(CONVERGE_TOL),
                 },
                 "handover": {
                     "clamp_settle_steps": int(CLAMP_SETTLE),
                     "clamp_float_gain": float(CLAMP_FLOAT_GAIN),
                     "clamp_anchor": bool(CLAMP_ANCHOR),
                     "clamp_lift": float(CLAMP_LIFT),
                     "handover_first": bool(HANDOVER_FIRST),
                     "seat_stiffness": float(SEAT_K),
                     "seat_damping": float(SEAT_C),
                     "seat_dwell_steps": int(SEAT_DWELL),
                     "seat_radial_stiffness": float(SEAT_RADIAL_K),
                     "seat_radial_damping": float(SEAT_RADIAL_C),
                     "seat_fade_steps": int(SEAT_FADE_STEPS),
                     "preclamp": bool(PRECLAMP),
                     "preclamp_ctrl": float(PRECLAMP_CTRL),
                     "preclamp_steps": int(PRECLAMP_STEPS),
                     "preclamp_dwell_steps": int(PRECLAMP_DWELL),
                     "preclamp_seat_stiffness": float(PRECLAMP_SEAT_K),
                     "preclamp_seat_damping": float(PRECLAMP_SEAT_C),
                 },
                 "chuck": {
                     "radius_mm": float(SM.CHUCK_R * 1000),
                     "length_mm": float(SM.CHUCK_L * 1000),
                     "bore_radius_mm": float(SM.BORE_R * 1000),
                     "bore_segments": int(SM.BORE_SEGS),
                     "jaw_length_mm": float(SM.JAW_LEN * 1000),
                     "open_ctrl": float(SM.CHUCK_OPEN_CTRL),
                     "clamp_ctrl": float(SM.CHUCK_CLAMP_CTRL),
                     "force_limit_N": float(SM.CHUCK_FORCE_LIMIT),
                     "lathe_origin_m": [float(x) for x in LATHE_ORIGIN],
                 },
                 "arm_servo": {
                     "base_kp_Nm_rad": float(F.ARM_SERVO_KP),
                     "effective_kv_Nms_rad": float(
                         os.environ.get("SO101_KV", FEED_ARM_KV)),
                     "effective_kp_Nm_rad": float(
                         os.environ.get("SO101_KP", F.ARM_SERVO_KP)),
                 },
             },
             "environment_overrides": _capture_environment_overrides(),
             "config": {
                 "insert_mm": float(INSERT_MM * 1000),
                 "grip_station_mm": float(GRIP_STATION * 1000),
                 "v_depth_mm": float(V_DEPTH * 1000),
                 "v_length_mm": float(V_LEN * 1000),
                 "v_angle_deg": 90.0,
                 "initial_offset_mm": [float(x) for x in offset_mm],
                 "initial_roll_deg": float(legacy_roll_deg),
                 "initial_yaw_world_z_deg": float(yaw_world_z_deg),
                 "planning_mode": planning_mode,
                 "m2_error_mm": [float(x) for x in m2_error_mm],
                 "m2_error_yaw_deg": float(m2_error_yaw_deg),
                 "v_jaw": bool(v_jaw), "lift": bool(lift),
                 "carry_lock": bool(carry_lock),
                 "machine_solid": bool(machine_solid),
                 "evaluation_scope": evaluation_scope,
                 "retreat_axial": bool(
                     RETREAT_AXIAL and not no_retreat_diagnostic),
                 "diagnostic_mode": diagnostic_mode,
             }}

    robot_machine_peaks = {}

    def record_diagnostic_frame(phase=None):
        """Capture one post-step contact/load frame into compact rollups."""
        phase = phase or state.get("label") or "simulation"
        contact_snapshot = _contact_frame_snapshot(
            model, data, contact_geom_sets)
        load_snapshot = _applied_load_snapshot(model, data)
        _update_diagnostic_rollups(
            state,
            state.get("diagnostic_scope", "feed"),
            phase,
            float(data.time),
            contact_snapshot,
            load_snapshot,
        )

        pickup = state["pickup_motion_diagnostics"]
        pickup_group = next(
            (name for name in pickup["measurement_scope"]
             if phase == name or phase.startswith(name + "#")),
            None,
        )
        if pickup_group is not None and state["actual_initial_pose"]:
            current_position = shaft()
            reference_position = np.asarray(
                pickup["reference_pose"]["position_m"], dtype=float)
            current_axis = rod_axis()
            reference_axis = np.asarray(
                pickup["reference_pose"]["shaft_axis_world_unit"],
                dtype=float)
            center_delta = current_position - reference_position
            center_displacement_mm = float(
                np.linalg.norm(center_delta) * 1000.0)
            axis_change_deg = float(np.degrees(np.arccos(np.clip(
                abs(float(np.dot(current_axis, reference_axis))),
                -1.0, 1.0))))
            robot_workpiece = contact_snapshot["counts_by_category"][
                "robot_workpiece"]
            robot_workpiece_force = (
                contact_snapshot["normal_force_sum_by_category_N"][
                    "robot_workpiece"])
            phase_row = pickup["phases"].setdefault(pickup_group, {
                "samples": 0,
                "robot_workpiece_contact_samples": 0,
                "peak_robot_workpiece_contacts_per_step": 0,
                "peak_robot_workpiece_normal_force_sum_N": 0.0,
                "max_center_displacement_mm": 0.0,
                "center_displacement_vector_at_max_mm": [
                    0.0, 0.0, 0.0],
                "max_shaft_axis_change_deg": 0.0,
            })
            for summary in (pickup["overall"], phase_row):
                summary["samples"] += 1
                summary["robot_workpiece_contact_samples"] += int(
                    robot_workpiece > 0)
                summary["peak_robot_workpiece_contacts_per_step"] = max(
                    summary["peak_robot_workpiece_contacts_per_step"],
                    robot_workpiece)
                summary["peak_robot_workpiece_normal_force_sum_N"] = max(
                    summary["peak_robot_workpiece_normal_force_sum_N"],
                    robot_workpiece_force)
                if (center_displacement_mm
                        > summary["max_center_displacement_mm"]):
                    summary["max_center_displacement_mm"] = (
                        center_displacement_mm)
                    summary["center_displacement_vector_at_max_mm"] = [
                        float(value * 1000.0) for value in center_delta]
                summary["max_shaft_axis_change_deg"] = max(
                    summary["max_shaft_axis_change_deg"], axis_change_deg)
            pickup["status"] = "observed"

        relative_motion = state["gripper_relative_workpiece_motion"]
        if relative_motion["active"]:
            site_rotation = data.site_xmat[arm.site].reshape(3, 3)
            site_position = data.site_xpos[arm.site].copy()
            pad_position = arm.pad_center()
            shaft_position = shaft()
            shaft_axis = rod_axis()
            current_offset_local, current_axis_local = _pose_in_tool_frame(
                site_rotation, shaft_position, pad_position, shaft_axis)
            reference_offset_local = np.asarray(
                relative_motion["reference_pose"][
                    "center_offset_tool_m"],
                dtype=float)
            reference_axis_local = np.asarray(
                relative_motion["reference_pose"]["axis_tool_unit"],
                dtype=float)
            translation_delta, axis_change_deg = _relative_pose_change(
                reference_offset_local, reference_axis_local,
                current_offset_local, current_axis_local)
            translation_mm = float(
                np.linalg.norm(translation_delta) * 1000.0)
            phase_group = (
                "进给" if phase == "进给" or phase.startswith("进给#")
                else "搬运")
            phase_row = relative_motion["phases"].setdefault(
                phase_group, {
                    "samples": 0,
                    "max_translation_slip_mm": 0.0,
                    "translation_slip_vector_tool_mm_at_max": [
                        0.0, 0.0, 0.0],
                    "max_relative_axis_change_deg": 0.0,
                    "time_of_max_translation_slip_s": None,
                })
            previous_translation = relative_motion.get(
                "_previous_translation_delta_tool_m")
            step_delta_m = (
                np.zeros(3, dtype=float) if previous_translation is None
                else translation_delta - np.asarray(
                    previous_translation, dtype=float))
            for summary in (relative_motion["overall"], phase_row):
                summary["samples"] += 1
                if translation_mm > summary["max_translation_slip_mm"]:
                    summary["max_translation_slip_mm"] = translation_mm
                    summary["translation_slip_vector_tool_mm_at_max"] = [
                        float(value * 1000.0)
                        for value in translation_delta]
                    summary["time_of_max_translation_slip_s"] = float(
                        data.time)
                summary["max_relative_axis_change_deg"] = max(
                    summary["max_relative_axis_change_deg"],
                    axis_change_deg)
            relative_motion["_previous_translation_delta_tool_m"] = [
                float(value) for value in translation_delta]

            if relative_motion["time_history_enabled"]:
                shaft_contact_pairs = {}
                for contact_id in range(data.ncon):
                    contact = data.contact[contact_id]
                    if (contact.geom1 not in contact_geom_sets["workpiece"]
                            and contact.geom2 not in
                            contact_geom_sets["workpiece"]):
                        continue
                    geom_names = sorted(
                        mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "?"
                        for geom_id in (contact.geom1, contact.geom2))
                    category = _contact_category(
                        contact.geom1, contact.geom2, contact_geom_sets)
                    pair_key = "%s|%s|%s" % (
                        category, geom_names[0], geom_names[1])
                    pair = shaft_contact_pairs.setdefault(pair_key, {
                        "category": category,
                        "geoms": geom_names,
                        "contact_count": 0,
                        "normal_force_sum_N": 0.0,
                        "minimum_distance_mm": float("inf"),
                    })
                    mujoco.mj_contactForce(
                        model, data, contact_id, _cf6)
                    pair["contact_count"] += 1
                    pair["normal_force_sum_N"] += abs(float(_cf6[0]))
                    pair["minimum_distance_mm"] = min(
                        pair["minimum_distance_mm"],
                        float(contact.dist) * 1000.0)

                site_quaternion = np.zeros(4, dtype=float)
                mujoco.mju_mat2Quat(
                    site_quaternion,
                    np.ascontiguousarray(site_rotation).reshape(-1))
                relative_motion["time_history"].append({
                    "sim_time_s": float(data.time),
                    "phase": phase,
                    "phase_group": phase_group,
                    "shaft_center_world_m": [
                        float(value) for value in shaft_position],
                    "shaft_axis_world_unit": [
                        float(value) for value in shaft_axis],
                    "pad_center_world_m": [
                        float(value) for value in pad_position],
                    "gripper_site_position_world_m": [
                        float(value) for value in site_position],
                    "gripper_site_quaternion_wxyz": [
                        float(value) for value in site_quaternion],
                    "shaft_center_offset_tool_mm": [
                        float(value * 1000.0)
                        for value in current_offset_local],
                    "translation_change_from_carry_start_tool_mm": [
                        float(value * 1000.0)
                        for value in translation_delta],
                    "translation_change_from_carry_start_mm": (
                        translation_mm),
                    "translation_change_since_previous_step_tool_mm": [
                        float(value * 1000.0) for value in step_delta_m],
                    "translation_change_since_previous_step_mm": float(
                        np.linalg.norm(step_delta_m) * 1000.0),
                    "relative_axis_change_from_carry_start_deg": (
                        axis_change_deg),
                    "gripper_command": float(data.ctrl[arm.grip_act]),
                    "shaft_contact_pairs": [
                        {
                            **pair,
                            "minimum_distance_mm": float(
                                pair["minimum_distance_mm"]),
                        }
                        for _key, pair in sorted(
                            shaft_contact_pairs.items())
                    ],
                })
        return contact_snapshot, load_snapshot

    def write_result_json(result):
        json_path = os.environ.get("FEED_RESULT_JSON")
        if not json_path:
            return
        relative_motion = result.get(
            "gripper_relative_workpiece_motion", {})
        relative_motion.pop("_previous_translation_delta_tool_m", None)
        result.update(_case_metadata(state, model))
        path = os.path.abspath(json_path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
        print("FEED_RESULT_JSON %s" % path)

    def ensure_finite_state(label):
        if (np.isfinite(data.time)
                and _finite_sim_state(
                    data.qpos, data.qvel, data.ctrl, data.qacc,
                    data.qfrc_applied, data.xfrc_applied)):
            return
        print("FEED_ABORT stage=%s reason=non_finite_state" % label)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "case_id": state["case_id"],
            "random_seed": state["random_seed"],
            "created_at_unix": time.time(),
            "duration_s": time.perf_counter() - started_at,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "config": state["config"],
            "execution_status": "error",
            "abort_stage": label,
            "abort_reason": "non_finite_state",
            "metrics": {},
            "hard_pass": False,
            "quality_pass": None,
            "overall_pass": False,
            "hard_failures": ["non_finite_state"],
            "quality_failures": [],
            "phase_stats": state.get("phase_stats", {}),
            **_diagnostic_result_fields(state),
            "ik_solver": state.get("ik_solver", []),
            "ik_residuals_mm": state.get("ik_residuals_mm", {}),
            "stages": state.get("stage_records", []),
            "checkpoints": state.get("checkpoints", []),
        }
        result.update(_case_classification(
            result["execution_status"], evaluation_scope,
            result["hard_pass"], result["quality_pass"],
            result["overall_pass"]))
        write_result_json(result)
        raise FloatingPointError(
            "non-finite simulation state at stage %s" % label)

    def step_sim(phase=None):
        label = phase or state.get("label") or "simulation"
        ensure_finite_state("%s:before_step" % label)
        mujoco.mj_step(model, data)
        ensure_finite_state("%s:after_step" % label)
        record_diagnostic_frame(label)

    def record_stage(name, checkpoint_id=None,
                     checkpoint_status="completed"):
        ensure_finite_state(name)
        mujoco.mj_forward(model, data)
        ensure_finite_state("%s:after_forward" % name)
        contacts, jaw_force = jaw_state()
        axis_now = rod_axis()
        p = shaft()
        machine_pairs = {}
        shaft_contacts = 0
        for i in range(data.ncon):
            c = data.contact[i]
            n1 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "?"
            n2 = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or "?"
            if _is_robot_machine_contact(
                    c.geom1, c.geom2, robot_geom_ids, machine_geom_ids):
                mujoco.mj_contactForce(model, data, i, _cf6)
                key = "%s|%s" % (n1, n2)
                machine_pairs[key] = max(
                    machine_pairs.get(key, 0.0), abs(float(_cf6[0])))
            if any(g in (c.geom1, c.geom2) for g in shaft_geoms):
                shaft_contacts += 1
        for key, value in machine_pairs.items():
            robot_machine_peaks[key] = max(
                robot_machine_peaks.get(key, 0.0), value)
        row = {
            "stage": name,
            "sim_time_s": float(data.time),
            "shaft_position_m": [float(x) for x in p],
            "shaft_axis": [float(x) for x in axis_now],
            "shaft_axis_error_deg": axdev(),
            "coax_mm": coax(),
            "stock_ecc_mm": stock_ecc(),
            "gripper_site_position_m": [
                float(x) for x in data.site_xpos[arm.site]],
            "jaw_contact_count": contacts,
            "jaw_normal_force_N": jaw_force,
            "shaft_contact_count": shaft_contacts,
            "robot_machine_contact_count": len(machine_pairs),
            "robot_machine_pairs_force_N": machine_pairs,
            "shaft_qfrc_applied_N": [
                float(x) for x in data.qfrc_applied[sv:sv + 3]],
            "shaft_external_force_max_N": float(
                np.max(np.abs(data.qfrc_applied[sv:sv + 3]))),
            "penetration_min_m": float(PEN["min"]),
            "penetration_pair": PEN["pair"],
            "penetration_phase": PEN["label"],
            "diagnostic_scope": state.get("diagnostic_scope", "feed"),
            "contact_diagnostics": _contact_frame_snapshot(
                model, data, contact_geom_sets),
            "applied_loads": _applied_load_snapshot(model, data),
        }
        if checkpoint_id is not None:
            checkpoint_order = next(
                (index for index, (key, _label)
                 in enumerate(FEED_CHECKPOINTS, start=1)
                 if key == checkpoint_id),
                None,
            )
            if checkpoint_order is None:
                raise ValueError("unknown feed checkpoint: %s" % checkpoint_id)
            if checkpoint_status not in ("completed", "skipped"):
                raise ValueError(
                    "invalid feed checkpoint status: %s"
                    % checkpoint_status)
            row.update({
                "checkpoint_id": checkpoint_id,
                "checkpoint_order": checkpoint_order,
                "checkpoint_label": FEED_CHECKPOINTS[
                    checkpoint_order - 1][1],
                "checkpoint_status": checkpoint_status,
            })
            state["checkpoints"].append(dict(row))
        state["stage_records"].append(row)
        return row

    def stage_phase_snapshot(name):
        phases = state.get("phase_stats", {})
        if name == "robot_machine_pairs_peak":
            return dict(robot_machine_peaks)
        if name == "phase_stats":
            return phases
        return None

    def part_offset():
        """Transform the grasp-time estimated part offset with the tool."""
        if carry_lock and state["offset"] is not None:
            return state["offset"]
        if state.get("grasp_pad_offset_local") is not None:
            mujoco.mj_forward(model, data)
            R_site = data.site_xmat[arm.site].reshape(3, 3)
            return R_site @ state["grasp_pad_offset_local"]
        return arm.pad_center() - shaft()

    def pad_target(tgt):
        return np.asarray(tgt, dtype=float) + part_offset()

    def rod_axis():
        a = data.xmat[shaft_body].reshape(3, 3)[:, 2].copy()
        return a / max(float(np.linalg.norm(a)), 1e-12)

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

    def move(target, ns, grip, locked, jaw_ctrl, axis_hold=None,
             grip_end=None, s0=0.0, s1=1.0, settle_steps=None):
        """Move the grip point, holding the rod axis during the feed.

        Position-only IK lets wrist_flex drift while the arm translates; that
        is why the rod used to arrive 80 deg from the chuck axis.  In feed
        mode we solve position + rod-axis with all five joints, so the last
        wrist joint actively counter-rotates.  Full 6-D orientation is NOT
        demanded: a cylindrical rod has a free roll about its own axis, and
        forcing it made the five-joint arm chase an unreachable pose.

        grip_end optionally ramps the finger command along the same smooth
        time law, so release and retreat happen together rather than the
        gripper kicking the clamped rod while still parked against it.
        """
        if axis_hold is None:
            axis_hold = state["axis_hold"]
        grasp_axis_active = state.get("grasp_axis_active", False)
        use_axis_ik = axis_hold or grasp_axis_active
        if not use_axis_ik:
            # 只写 ctrl，绝对不要写 qpos。
            # 直接给 qpos 会在每段第一帧把腕关节一步拽到 wrist，末端
            # 瞬跳 5~10 mm（实测 夹紧段 10.3 mm / 5.13 m/s、对中段
            # 4.8 mm / 2.39 m/s，全部落在 步=0）。那就是窗口里的
            # 「抖一抖」—— 不是伺服抖，是每段起步被硬拽一下。
            data.ctrl[4] = wrist
        mujoco.mj_forward(model, data)
        off = arm.pad_offset()
        site_target = np.asarray(target) - off
        q_start = np.array([data.qpos[a] for a in arm.qadr])
        q_path = None
        grid = None
        if use_axis_ik:
            # Solve IK at several Cartesian waypoints instead of only at the
            # segment end. A straight joint-space interpolation between two
            # IK solutions lets the tool axis wander 20+ deg in between.
            R_site = data.site_xmat[arm.site].reshape(3, 3)
            if axis_hold:
                a_local = state["a_local"]
                axis_target = _AXIS_TGT
            else:
                a_local = state["grasp_axis_local"]
                axis_target = state["grasp_axis_target"]
            site_start = data.site_xpos[arm.site].copy()
            nwp = 8
            grid = np.linspace(0.0, 1.0, nwp + 1)
            pts = [q_start]
            q_prev = q_start.copy()
            last_ik = (0.0, 0.0)
            for frac in grid[1:]:
                p = site_start + frac * (site_target - site_start)
                q_prev, pe, ae = arm.ik_axis(
                    p, axis_target, a_local,
                    q_init=q_prev, iters=240)
                last_ik = (pe, ae)
                state.setdefault("ik_solver", []).append({
                    "stage": state.get("label", "?"),
                    "position_residual_mm": float(pe * 1000),
                    "axis_residual_deg": float(np.degrees(ae)),
                })
                pts.append(q_prev.copy())
            q_path = np.asarray(pts)
            # 关节空间样条：C1/C2 连续，配合 quintic 时间律才不折角。
            _spl = [CubicSpline(grid, q_path[:, j], bc_type="natural")
                    for j in range(q_path.shape[1])]
            q_goal = q_path[-1].copy()
            if DEBUG_MOVE:
                print("    ik_axis[%s] 终点 %.1f mm / %.1f deg"
                      % (state.get("label", ""), last_ik[0] * 1000,
                         np.degrees(last_ik[1])))
        else:
            q_goal, solver_residual = arm.ik(
                site_target, lock_wrist_roll=wrist)
            state.setdefault("ik_solver", []).append({
                "stage": state.get("label", "?"),
                "position_residual_mm": float(solver_residual * 1000),
                "axis_residual_deg": None,
            })

        # Quintic (minimum-jerk) time law on a GLOBAL time axis [s0, s1].
        # move3 splits one stage into several closed-loop passes; giving each
        # pass its own 0->1 ramp made the arm stop dead between passes (the
        # "鸡啄米" look). Sharing one smooth time law keeps the velocity
        # continuous across passes, so the arm never parks mid-stage.
        def _smooth(t):
            return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))

        S0, S1 = _smooth(s0), _smooth(s1)
        span = max(S1 - S0, 1e-12)
        for i in range(ns):
            t = s0 + (s1 - s0) * (i + 1) / ns
            s = (_smooth(t) - S0) / span
            if q_path is None:
                q = q_start + s * (q_goal - q_start)
            else:
                # 样条而不是 np.interp：线性插值让关节速度在 8 个路点处
                # 折一下（加速度阶跃），远控窗口里就是一格一格地蹭。
                # 三次样条让 v 和 a 都连续，时间律本来就是 quintic，
                # 两者乘起来才是一条真正光滑的轨迹。
                q = np.array([_spl[j](s) for j in range(len(arm.qadr))])
            if ARM_MOVE_LEAD or ARM_MOVE_DAMP:
                # 参考速度由下一时刻的同一条样条求出，和 move_path() 的
                # 前馈一致。速度前馈抵掉纯 PD 匀速段 (kv/kp)*qdot 的滞后；
                # 速度误差阻尼把「实际跟不上参考」时的超调拉回来。
                _dt = float(model.opt.timestep)
                t2 = min(s1, s0 + (s1 - s0) * (i + 2) / ns)
                s2 = min(1.0, max(0.0, (_smooth(t2) - S0) / span))
                if q_path is None:
                    q2 = q_start + s2 * (q_goal - q_start)
                else:
                    q2 = np.array([_spl[j](s2)
                                   for j in range(len(arm.qadr))])
                qd = (q2 - q) / _dt
                for a, v, w in zip(arm.qadr, q, qd):
                    data.ctrl[a] = (v + ARM_MOVE_LEAD * w
                                    - ARM_MOVE_DAMP
                                    * (float(data.qvel[a]) - w))
            else:
                for a, v in zip(arm.qadr, q):
                    data.ctrl[a] = v
            gcmd = grip if grip_end is None else grip + s * (grip_end - grip)
            data.ctrl[arm.grip_act] = grip_cmd(gcmd)
            data.ctrl[jaw_act] = jaw_ctrl
            step_sim()
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_t", []).append(float(data.time))
            state.setdefault("traj_label", []).append(state.get("label", "?"))
            if (len(state["traj"]) > 1 and state.get("njump", 0) < 8):
                _sm = float(np.linalg.norm(
                    state["traj"][-1] - state["traj"][-2])) * 1000.0
                if _sm > 4.0:
                    state["njump"] = state.get("njump", 0) + 1
                    _v = _sm / model.opt.timestep / 1000.0
                    print("    JUMP %5.1f mm (%.2f m/s) 段=%s 步=%d q=%s"
                          % (_sm, _v, state.get("label", "?"), i,
                             np.round([data.qpos[a] for a in arm.qadr], 3)))
            if locked:
                lock()
            track_pen(state.get("label", "move"))
            if frame_hook is not None and i % FRAME_EVERY == 0:
                tick()
        # Only the LAST pass of a stage gets a servo settle. Holding 180
        # steps after every pass re-created the stop-motion effect that the
        # shared time law just removed.
        # 2026-11-06: 180/260 步 = 0.36/0.52 s 原地不动，8 段路点加起来
        # 4 秒多静止，远控窗口里就是「走一下停一下」。位置伺服收敛只要
        # 0.1 s 量级，剩下的时间纯属空等。
        if settle_steps is None:
            settle_steps = 60 if axis_hold else 80
        final_grip = grip if grip_end is None else grip_end
        for j in range(settle_steps):
            for a, v in zip(arm.qadr, q_goal):
                data.ctrl[a] = v
            data.ctrl[arm.grip_act] = grip_cmd(final_grip)
            data.ctrl[jaw_act] = jaw_ctrl
            step_sim()
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_t", []).append(float(data.time))
            state.setdefault("traj_label", []).append(state.get("label", "?"))
            if locked:
                lock()
            if frame_hook is not None and j % (FRAME_EVERY * 2) == 0:
                tick()
        return float(np.linalg.norm(arm.pad_center() - np.asarray(target)))

    def move3(target, ns, grip, locked,
              jaw_ctrl=SM.CHUCK_OPEN_CTRL, iters=8, probe=None,
              probe_goal=None, label=""):
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
        state["label"] = label
        # After grasp, use ONE continuous minimum-jerk trajectory. The old
        # loop chopped ns into thirds and repeated 2-8 point-to-point moves;
        # at dt=0.002 s that moved the wrist 130 mm in 0.28 s, which the
        # position servos could not follow (measured 45 deg lag). Long single
        # segments are both smoother and more accurate.
        hold_done = (state["axis_hold"] and state["offset"] is not None)
        move_steps = ns if hold_done else max(90, ns // 3)
        if hold_done and iters == 8:
            # Full-length minimum-jerk corrections, not chopped point-to-point
            # retries. Measured error halves each pass (51 -> 17 -> 8 -> 2 mm).
            # The final feed gets one extra pass because its position decides
            # whether all six chuck contacts survive the gripper release.
            iters = 4 if label == "进给" else 3

        # Apply the grip command BEFORE the convergence check.
        # "夹紧" targets the same point as the preceding "下降", so err was
        # already ~0 and the early break below returned without ever writing
        # GRIP_CLOSE into ctrl: the fingers stayed at 0.500 (open), the pads
        # sat 19-23 mm off the axis, and the shaft never left the table.
        # Measured before this fix: ncon had zero pad-shaft contacts.
        for _ in range(30 if hold_done else GRIP_SETTLE):
            data.ctrl[arm.grip_act] = grip_cmd(grip)
            data.ctrl[jaw_act] = jaw_ctrl
            step_sim()
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_t", []).append(float(data.time))
            state.setdefault("traj_label", []).append(state.get("label", "?"))
            track_pen(label)
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
        trace = [err]
        tol = 0.001 if label == "进给" else 0.003
        # 第一轮的行程就是「整段距离」，后面每一轮只差几毫米。
        # 2026-11-06 实测（FEED_DEBUG=1）：进给 55.2 -> 12.4 -> 5.7 -> 2.0 -> 0.7 mm，
        # 四轮各给 500 步。第一轮 2.53 m/s 冲过去，后三轮拿同样的时间搬
        # 12/6/2 mm，末端速度不到 0.01 m/s —— 窗口里就是「跳一下，然后
        # 长时间几乎不动，再跳一下」。按剩余距离缩时，修正轮变成一次
        # 短促的小动作，整段看起来是一条连续轨迹。
        dist0 = max(err, 1e-4)
        for k in range(iters):
            if err < tol:
                break
            # 时间律共享之后，每轮的行程必须配同样的时长，否则「缩短的
            # 修正轮」会用更短的时间走同样的 s 区间，末端速度反而更高。
            # 收敛轮的位移小、速度自然低，看起来是减速收车，不再是一顿
            # 一顿。
            steps_k = move_steps
            # 多轮闭环共享一条时间律，轮间不停车。原来只在 axis_hold
            # 时这么做，其余段每轮各跑一条 0->1 的五次曲线，速度每轮
            # 归零一次 —— 进给 3 轮就是 3 次急停急起。五次曲线的导数
            # 只在 t=0/1 处为零，按 [k/iters,(k+1)/iters] 切分后轮间
            # 速度是连续的。
            kw = {"s0": k / iters, "s1": (k + 1) / iters}
            state["label"] = "%s#%d" % (label, k + 1)
            kw["settle_steps"] = 90 if k == iters - 1 else 60
            move(tgt, steps_k, grip, locked, jaw_ctrl, **kw)
            err = float(np.linalg.norm(meas() - gmeas))
            trace.append(err)
            # 0.8 damping: full step overshoots once the load unloads
            tgt = tgt + FEED_GAIN * (gmeas - meas())
        state.setdefault("ik_residuals_mm", {})[label] = float(err * 1000)
        if DEBUG_MOVE:
            # 正负号变化 = 闭环在目标两侧来回；单调下降 = 只是慢。
            sign_flips = sum(
                1 for a, b, c in zip(trace, trace[1:], trace[2:])
                if (b - a) * (c - b) < 0)
            print("    err[%-4s] %s mm  轮数=%d 过冲=%d"
                  % (label, " ".join("%.1f" % (e * 1000) for e in trace),
                     len(trace) - 1, sign_flips))
        return err

    def move_path(targets, grip, locked, jaw_ctrl=SM.CHUCK_OPEN_CTRL,
                  label="搬运", samples=None, v_cruise=0.55, a_max=2.0,
                  j_max=25.0, lead=None):
        """一条连续轨迹走完整串路点：中途不减速、不停车。

        老写法每个路点一次 move3（rest -> rest），抬升/横移/下降/进给
        四次就是四次完整起停，速度每个路点归零一次。Ruckig 的做法是
        路点只约束位置，速度由全局时间律安排：jerk 受限的梯形速度曲线，
        整条作业只有一次起步、一次收车。

        路径按弧长均匀重采样后逐点链式解 IK（保持工件轴线），再用
        PCHIP 连接。用 PCHIP 而不是三次样条：自然样条在路点密的地方
        会过冲，关节速度因此出现负向小摆动，窗口里就是「一顿一顿」。

        播放时叠加速度前馈 ctrl = q_ref + (kv/kp)*qdot_ref。纯 PD 在
        匀速段的滞后是 (kv/kp)*qdot，这一项把它抵掉。
        """
        from scipy.interpolate import PchipInterpolator
        from ruckig import Ruckig, InputParameter, Trajectory

        if lead is None:
            lead = ARM_LEAD
        state["label"] = label
        mujoco.mj_forward(model, data)
        off = arm.pad_offset()
        site_start = data.site_xpos[arm.site].copy()
        pts = [site_start] + [np.asarray(t, float) - off for t in targets]
        uniq = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - uniq[-1]) > 1e-6:
                uniq.append(p)
        seg = np.array([np.linalg.norm(b - a) for a, b in zip(uniq, uniq[1:])])
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        if total < 1e-6:
            return 0.0
        if samples is None:
            samples = int(np.clip(total / 0.012, 24, 80))

        s_grid = np.linspace(0.0, 1.0, samples + 1)
        dense = []
        for s in s_grid:
            d = s * total
            i = int(np.clip(np.searchsorted(cum, d) - 1, 0, len(seg) - 1))
            f = 0.0 if seg[i] < 1e-9 else (d - cum[i]) / seg[i]
            dense.append(uniq[i] + np.clip(f, 0.0, 1.0) * (uniq[i + 1] - uniq[i]))

        # 逐点链式 IK：每一点的初值取上一点的解，避免跳分支。
        q_start = np.array([data.qpos[a] for a in arm.qadr])
        R_site = data.site_xmat[arm.site].reshape(3, 3)
        a_local = state["a_local"]
        qs = [q_start]
        q_prev = q_start.copy()
        worst = 0.0
        for p in dense[1:]:
            # 每个点都从同一个初值解，不要链式传递。4 个自由关节解 3 维
            # 位置本身就是冗余的（有 1 维零空间），逐点以上一点为初值，
            # 47 个点走下来会沿零空间一路漂，实测末端在 x 方向少走 100 mm、
            # 机械臂自己折起来（同轴 32 mm、接触 0）。
            if state["axis_hold"]:
                q_sol, pe, _ae = arm.ik_axis(
                    p, _AXIS_TGT, a_local,
                    q_init=q_start, iters=300)
            else:
                q_sol, pe = arm.ik(p, lock_wrist_roll=wrist)
            worst = max(worst, float(pe))
            qs.append(np.asarray(q_sol, float).copy())
        Q = np.asarray(qs)
        state.setdefault("ik_solver", []).append({
            "stage": label, "position_residual_mm": float(worst * 1000),
            "axis_residual_deg": None,
        })

        # 归一化弧长上做 jerk 受限的时间参数化。
        dt = float(model.opt.timestep)
        otg = Ruckig(1, dt)
        inp = InputParameter(1)
        inp.current_position = [0.0]
        inp.current_velocity = [0.0]
        inp.current_acceleration = [0.0]
        inp.target_position = [1.0]
        inp.target_velocity = [0.0]
        inp.target_acceleration = [0.0]
        inp.max_velocity = [v_cruise / total]
        inp.max_acceleration = [a_max / total]
        inp.max_jerk = [j_max / total]
        traj = Trajectory(1)
        otg.calculate(inp, traj)
        n = max(2, int(np.ceil(traj.duration / dt)))

        prof = [traj.at_time(min(i * dt, traj.duration)) for i in range(n + 1)]
        pchip = PchipInterpolator(s_grid, Q, axis=0)

        for i in range(n):
            s = float(np.clip(prof[i][0][0], 0.0, 1.0))
            sd = float(prof[i][1][0])
            s2 = float(np.clip(prof[min(i + 1, n)][0][0], 0.0, 1.0))
            q = pchip(s)
            qd = (pchip(s2) - q) / dt
            for a, v, w in zip(arm.qadr, q, qd):
                data.ctrl[a] = v + lead * w
            data.ctrl[arm.grip_act] = grip_cmd(grip)
            data.ctrl[jaw_act] = jaw_ctrl
            step_sim()
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_t", []).append(float(data.time))
            state.setdefault("traj_label", []).append(state.get("label", "?"))
            if locked:
                lock()
            track_pen(label)
            if DEBUG_MOVE and i % 100 == 0:
                print("      i=%4d s=%.3f cmd=%s qpos=%s pad=%s ncon=%d"
                      % (i, s, np.round(q, 2).tolist(),
                         np.round([data.qpos[a] for a in arm.qadr], 2).tolist(),
                         np.round(arm.pad_center(), 3).tolist(), data.ncon))
            if frame_hook is not None and i % FRAME_EVERY == 0:
                tick()

        if DEBUG_MOVE:
            print("    path[%s] 弧长 %.0f mm / %d 点 / 规划 %.2f s / IK 最大残差 %.1f mm"
                  % (label, total * 1000, samples + 1, traj.duration,
                     worst * 1000))
        return float(np.linalg.norm(arm.pad_center() - np.asarray(targets[-1])))

    def support_rod():
        """交接期托住棒料：抵消自重，等效真机的导套/托料架。"""
        if _lift > 0:
            data.qfrc_applied[sv + 2] = _lift * float(
                model.body_mass[shaft_body] * abs(model.opt.gravity[2]))

    def seat_rod(x0):
        """交接期轴向弹性定位：等效真机端面挡块 + 弹簧推料器。

        只约束轴向 (x)，径向 y/z 完全放开，让三爪能自由找正。
        qfrc_applied 是持久量，交接结束必须显式清零。
        """
        if not tuned or not HANDOVER_FIRST:
            return
        p = shaft()
        # 轴向：端面挡块 + 弹簧推料器。
        data.qfrc_applied[sv] = (SEAT_K * (x0 - float(p[0]))
                                 - SEAT_C * float(data.qvel[sv]))
        # 径向：等效导套/中心架。竖直项叠加自重补偿，和 support_rod()
        # 保持一致；kr=0 时退化为原来的纯托力。
        mg = float(model.body_mass[shaft_body]
                   * abs(model.opt.gravity[2]))
        cr = (SEAT_RADIAL_C if SEAT_RADIAL_C > 0 else
              1.4 * np.sqrt(max(SEAT_RADIAL_K, 1.0)
                            * model.body_mass[shaft_body]))
        data.qfrc_applied[sv + 1] = (
            SEAT_RADIAL_K * (axis[1] - float(p[1]))
            - cr * float(data.qvel[sv + 1]))
        data.qfrc_applied[sv + 2] = (
            mg + SEAT_RADIAL_K * (axis[2] - float(p[2]))
            - cr * float(data.qvel[sv + 2]))

    def pusher_rod(x0):
        """端面弹簧推料器：只在低压预夹期生效，轴向顶正，径向完全放开。

        Practical Machinist：磨损卡盘会向外让、把料推离挡块，车间做法是
        刀塔上装弹簧推料器在夹紧循环中把料重新顶正。qfrc_applied 是持久
        量，夹紧完成必须显式清零，否则会一路带到拉拔测试里。
        """
        if not PRECLAMP:
            return
        data.qfrc_applied[sv] = (
            PRECLAMP_SEAT_K * (x0 - float(shaft()[0]))
            - PRECLAMP_SEAT_C * float(data.qvel[sv]))

    def float_arm():
        """夹紧期把 5 个手臂关节的位置目标拉向当前实际角，让手臂变软。

        位置伺服原本死死盯着进给终点的关节角；三爪想找正料时，夹爪还夹着
        尾段，这个硬目标会把料顶偏。把目标跟着实际位置走，伺服就不再跟
        卡盘较劲，只剩阻尼。gain=1 纯浮动，小 gain 保留部分支撑刚度。
        夹爪自身 (arm.grip_act) 不参与浮动。
        """
        # 逐帧记录作用在【棒料】上的外力峰值：0 说明交接期没有隐藏对中
        # 力场，棒料只受重力 / 接触力 / 夹爪力。
        _f = float(np.max(np.abs(data.qfrc_applied[sv:sv + 3])))
        if _f > state.get("rod_force_max", 0.0):
            state["rod_force_max"] = _f
        g = _FLOAT[0]
        _dt = float(model.opt.timestep)
        if ARM_GCOMP or ARM_PAYLOAD > 0:
            # qfrc_bias 含重力 + 科氏 + 被动关节力；补偿掉它，手臂在
            # 无外力时既不塌也不自己动。用的是上一步的 bias，准静态够。
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, None, arm.site)
            _mg = (ARM_PAYLOAD * float(model.body_mass[shaft_body])
                   * abs(float(model.opt.gravity[2])))
            f_up = np.array([0.0, 0.0, _mg])
            tau_load = jacp[:, arm.dadr].T @ f_up
            for _k, d in enumerate(arm.dadr):
                data.qfrc_applied[d] = (
                    -ARM_GCOMP_GAIN * float(data.qfrc_bias[d]) + tau_load[_k])
        if g <= 0:
            return
        for a in arm.qadr:
            data.ctrl[a] += (g * (data.qpos[a] - data.ctrl[a])
                             - FLOAT_DAMP * float(data.qvel[a]) * _dt)

    def grip_direct(value):
        """Write an explicitly planned per-step gripper trajectory and sync its cache."""
        value = float(value)
        data.ctrl[arm.grip_act] = value
        state["grip_now"] = value
        state["grip_tgt"] = value
        state["grip_from"] = value
        state["grip_k"] = 0
        return value

    def grip_cmd(target):
        """Move the gripper command toward a fixed target without a step jump."""
        tgt = float(target)
        cur = float(data.ctrl[arm.grip_act])
        if abs(tgt - state.get("grip_tgt", 1e9)) > 1e-9:
            state["grip_tgt"] = tgt
            state["grip_from"] = cur
            state["grip_k"] = 0
        state["grip_k"] = state.get("grip_k", 0) + 1
        k = 1.0 if GRIP_RAMP <= 0 else min(1.0, state["grip_k"] / float(GRIP_RAMP))
        v = state["grip_from"] + (tgt - state["grip_from"]) * k
        state["grip_now"] = v
        return v

    def settle(n, jaw_ctrl, grip, locked=False, floating=False):
        for i in range(n):
            data.ctrl[jaw_act] = jaw_ctrl
            data.ctrl[arm.grip_act] = grip_cmd(grip)
            if floating:
                float_arm()
            step_sim()
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_t", []).append(float(data.time))
            state.setdefault("traj_label", []).append(state.get("label", "?"))
            if locked:
                lock()
            track_pen(state.get("label", "settle"))
            if frame_hook is not None and i % (FRAME_EVERY * 2) == 0:
                tick()

    # A late "整姿" stage is deliberately gone.  It used to let the rod tip
    # to 75-80 deg and then ask a 5-DoF arm for a full 6-D orientation; the
    # measured IK residual was 19-36 mm / 37-61 deg, i.e. unreachable.
    # Axis holding now begins at grasp, so there is nothing to undo.

    # ---------------- reset / home ----------------
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    reset_pose = shaft_pose_snapshot()
    state["inputs"]["workpiece"]["nominal_start_position_m"] = (
        reset_pose["position_m"])
    state["inputs"]["pose_before_perturbation"] = reset_pose
    data.qpos[sq:sq + 3] += offset_mm / 1000.0
    if yaw_world_z_deg:
        data.qpos[sq + 3:sq + 7] = _apply_world_yaw_to_quaternion(
            data.qpos[sq + 3:sq + 7].copy(), yaw_world_z_deg)
    mujoco.mj_forward(model, data)
    state["inputs"]["pose_after_perturbation_before_settle"] = (
        shaft_pose_snapshot())
    data.qpos[arm.qadr[4]] = wrist
    data.ctrl[4] = wrist
    grip_direct(GRIP_OPEN)
    data.ctrl[jaw_act] = SM.CHUCK_OPEN_CTRL
    initial_stability_samples = []
    for settle_step in range(initial_settle_steps):
        for k in range(5):
            data.ctrl[k] = data.qpos[arm.qadr[k]]
        step_sim("initial_settle")
        if (settle_step >=
                initial_settle_steps - INITIAL_STABILITY_WINDOW_STEPS):
            velocity = np.zeros(6, dtype=float)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY,
                shaft_body, velocity, 0)
            support_contact_count_by_side = {"L": 0, "R": 0}
            for contact_id in range(data.ncon):
                contact = data.contact[contact_id]
                first, second = int(contact.geom1), int(contact.geom2)
                if first in shaft_geoms:
                    other = second
                elif second in shaft_geoms:
                    other = first
                else:
                    continue
                for side, support_geom_ids in (
                        initial_support_geom_ids_by_side.items()):
                    if other in support_geom_ids:
                        support_contact_count_by_side[side] += 1
            shaft_qfrc_applied = np.asarray(
                data.qfrc_applied[sv:sv + 6], dtype=float)
            shaft_xfrc_applied = np.asarray(
                data.xfrc_applied[shaft_body], dtype=float)
            pose = shaft_pose_snapshot()
            initial_stability_samples.append({
                "sim_time_s": float(data.time),
                **pose,
                "angular_velocity_rad_s": [
                    float(value) for value in velocity[:3]],
                "linear_velocity_m_s": [
                    float(value) for value in velocity[3:]],
                "support_contact_count_by_side": (
                    support_contact_count_by_side),
                "qfrc_applied_force_N": [
                    float(value) for value in shaft_qfrc_applied[:3]],
                "qfrc_applied_torque_Nm": [
                    float(value) for value in shaft_qfrc_applied[3:]],
                "xfrc_applied_force_N": [
                    float(value) for value in shaft_xfrc_applied[:3]],
                "xfrc_applied_torque_Nm": [
                    float(value) for value in shaft_xfrc_applied[3:]],
            })
    support_geom_names_by_side = {
        side: [
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "?"
            for geom_id in sorted(geom_ids)
        ]
        for side, geom_ids in initial_support_geom_ids_by_side.items()
    }
    state["initial_stability"] = _summarize_initial_stability(
        initial_stability_samples,
        model.opt.timestep,
        F.SHAFT_L,
        support_geom_names_by_side,
        model.opt.gravity,
        model.body_gravcomp[shaft_body],
    )
    stability = state["initial_stability"]
    stability_result = (
        "PASS" if stability["stability_pass"] is True else
        "FAIL" if stability["stability_pass"] is False else
        "NOT_EVALUATED")
    print(
        "FEED_STABILITY %s samples=%d/%d endpoint_excursion_mm=%.6f "
        "limit_mm=%.3f support_L=%d/%d support_R=%d/%d "
        "max_linear_speed_m_s=%.9g max_angular_speed_rad_s=%.9g "
        "gravcomp=%.1f external_loads=%s"
        % (
            stability_result,
            stability["sample_count"],
            stability["required_sample_count"],
            stability.get("max_endpoint_excursion_mm", 0.0),
            stability["endpoint_excursion_limit_mm"],
            stability.get("support_contact_samples_by_side", {}).get("L", 0),
            stability["sample_count"],
            stability.get("support_contact_samples_by_side", {}).get("R", 0),
            stability["sample_count"],
            stability.get("max_linear_speed_m_s", 0.0),
            stability.get("max_angular_speed_rad_s", 0.0),
            stability["workpiece_gravcomp"],
            "none" if stability.get(
                "no_external_applied_loads_on_workpiece") is True
            else "present_or_unmeasured",
        ))
    # Pick uses the calibrated position-only IK. Axis holding starts only
    # AFTER the V jaws close; applying it during approach changes the
    # calibrated grasp pose and the blades shove the rod into the table.

    tick("① 起始｜卡盘张开，工件在台面")
    home = shaft()
    state["actual_initial_pose"] = shaft_pose_snapshot()
    state["pickup_motion_diagnostics"]["reference_pose"] = (
        state["actual_initial_pose"])
    state["pickup_motion_diagnostics"]["status"] = "observing"
    if lift:
        nominal_position = NOMINAL_SETTLED_POSITION_M.copy()
    else:
        nominal_position = np.asarray(SM.SHAFT_POS, dtype=float).copy()
    nominal_pose = _resolve_planning_pose(
        "M0",
        state["actual_initial_pose"],
        nominal_position,
        SHAFT_Q0,
    )
    planning_pose = _resolve_planning_pose(
        planning_mode,
        state["actual_initial_pose"],
        nominal_position,
        SHAFT_Q0,
        m2_error_mm=m2_error_mm,
        m2_error_yaw_deg=m2_error_yaw_deg,
    )
    pick_at = planning_pose["position_m"].copy()
    planning_axis = planning_pose["shaft_axis_world_unit"].copy()
    # Calibrate the tool-local rod axis at the fixed nominal grasp target.
    # M0/M2 carry this planned axis forward instead of re-reading the shaft.
    mujoco.mj_forward(model, data)
    nominal_offset = arm.pad_offset()
    nominal_q, _nominal_residual = arm.ik(
        nominal_position - nominal_offset, lock_wrist_roll=wrist)
    calibration_data = mujoco.MjData(model)
    calibration_data.qpos[:] = data.qpos
    for qadr, value in zip(arm.qadr, nominal_q):
        calibration_data.qpos[qadr] = value
    mujoco.mj_forward(model, calibration_data)
    calibrated_site_rotation = calibration_data.site_xmat[
        arm.site].reshape(3, 3)
    state["grasp_axis_local"] = (
        calibrated_site_rotation.T @ WORLD_X_AXIS)
    state["grasp_axis_target"] = planning_axis
    state["grasp_axis_active"] = (
        np.degrees(np.arccos(np.clip(
            float(np.dot(planning_axis, WORLD_X_AXIS)), -1.0, 1.0)))
        > 0.25)
    state["inputs"]["grasp_target_pose"] = {
        "mode": planning_mode,
        "workpiece_center_m": [float(x) for x in pick_at],
        "quaternion_wxyz": [
            float(x) for x in planning_pose["quaternion_wxyz"]],
        "shaft_axis_world_unit": [
            float(x) for x in planning_axis],
        "descent_position_m": [
            float(pick_at[0]), float(pick_at[1]),
            float(pick_at[2] + 0.002)],
        "position_source": planning_pose["position_source"],
        "orientation_source": planning_pose["orientation_source"],
        "carry_offset_source": "planned_grasp_target_pose",
        "m2_error_used_mm": [
            float(x) for x in planning_pose["m2_error_used_mm"]],
        "m2_error_used_yaw_deg": float(
            planning_pose["m2_error_used_yaw_deg"]),
        "nominal_workpiece_center_m": [
            float(x) for x in nominal_pose["position_m"]],
        "true_workpiece_center_m": [
            float(x) for x in state["actual_initial_pose"]["position_m"]],
    }
    R = F.SHAFT_D / 2
    # Carry targets remain in the planner's pose estimate. The physical pose
    # is retained separately for scoring and diagnosis.
    g = pick_at
    # The ARM still approaches the calibrated rod centre. The V groove is
    # physically shifted to GRIP_STATION inside the gripper, so the contact
    # patch is on the tail while the approach pose stays unchanged.
    print("=" * 74)
    carry_mode = "运动学锁参考" if carry_lock else "物理接触无锁"
    print("机械臂送料到卡盘（合并场景；搬运模式：%s）" % carry_mode)
    print("=" * 74)
    print("  工件起点      : %s" % np.round(home, 4))
    print("  卡盘轴线      : %s" % np.round(axis, 4))
    print("  抓取站位      : V 槽位于工件中心 %+.0f mm（尾段夹持，自由端朝 +x）"
          % (GRIP_STATION * 1000))
    print()

    # ---------------- real-physics grasp ----------------
    # A vertical drop makes the first V-blade edge hit the rod before the
    # groove is underneath it.  That contact pushed the free rod 6.97 mm.
    # Enter through the actual mouth direction instead: the direction is
    # measured from the two V-jaw mouth centres at the approach pose, with
    # the rod-axis component removed.  This keeps the change local to pickup
    # and avoids hard-coding a world Y/Z direction for a calibrated wrist.
    pick = [
        ("接近", [pick_at[0], pick_at[1], pick_at[2] + R + 0.075],
         GRIP_OPEN, 550),
    ]
    if v_jaw:
        pick.append(("侧向进入", None, GRIP_OPEN, 400))
    pick.extend([
        ("下降", [pick_at[0], pick_at[1], pick_at[2] + 0.002],
         GRIP_OPEN, 400),
        ("夹紧", [pick_at[0], pick_at[1], pick_at[2] + 0.002],
         grip_close, 320),
    ])
    grasp_tilt = None
    for name, tgt, grip, ns in pick:
        if name == "侧向进入":
            fixed = [mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM,
                "fixed_jaw_pad_v%d" % k) for k in (1, 2)]
            moving = [mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM,
                "moving_jaw_pad_v%d" % k) for k in (1, 2)]
            entry_axis = (np.mean(data.geom_xpos[moving], axis=0)
                          - np.mean(data.geom_xpos[fixed], axis=0))
            rod = planning_axis
            entry_axis -= np.dot(entry_axis, rod) * rod
            entry_axis /= max(float(np.linalg.norm(entry_axis)), 1e-12)
            tgt = pick_at - 0.020 * entry_axis
            print("  抓取接近轴  fixed->moving=%s  进入目标=%s"
                  % (np.round(entry_axis, 4), np.round(tgt, 4)))
        move3(np.array(tgt), ns, grip, False, label=name)
        grasp_tilt = tilt_deg(data.qpos[sq + 3:sq + 7])
        print("  %-6s 轴=%s 轴向偏=%5.1f deg  pad-轴=%5.1f mm"
              % (name, np.round(shaft(), 4), axdev(),
                 np.linalg.norm(arm.pad_center() - shaft()) * 1000))
        if name == "夹紧":
            # Keep the same information boundary in every mode: the carry
            # target is derived from the planned grasp pose, never a fresh
            # measurement of the physical shaft at clamp time.
            state["offset"] = arm.pad_center() - pick_at
            R_site = data.site_xmat[arm.site].reshape(3, 3)
            state["grasp_pad_offset_local"] = (
                R_site.T @ state["offset"])
            state["a_local"] = R_site.T @ planning_axis
        tick({"接近": "② 接近｜到槽口外侧",
              "侧向进入": "③ 侧向进入｜沿 V 槽开口",
              "下降": "④ 下降｜pad 到轴心",
              "夹紧": "⑤ 夹紧｜真实接触抓取"}[name])
    # 夹爪是斜坡闭合（GRIP_RAMP 步），"夹紧"那段 320 步走完时它还没合到
    # 位。一抬臂，工件就从 V 槽里滑出去：实测全物理路径同轴 250 mm、倾斜
    # 133 deg、接触 0，就是这里漏了一段保持。先让爪子合完再搬。
    if GRIP_RAMP > 320:
        settle(GRIP_RAMP - 320 + 80, SM.CHUCK_OPEN_CTRL, grip_close)
    state["grasp_contact_count"] = grip_state()[0]
    if v_jaw and not carry_lock and state["grasp_contact_count"] == 0:
        state["gripper_relative_workpiece_motion"]["status"] = (
            "aborted_before_carry")
        record_stage("grasp_abort")
        print("FEED_ABORT stage=grasp reason=no_v_jaw_contact; "
              "搬运和卡盘闭合已取消")
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "case_id": state["case_id"],
            "random_seed": state["random_seed"],
            "created_at_unix": time.time(),
            "duration_s": time.perf_counter() - started_at,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "git_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))),
                capture_output=True, text=True, check=False).stdout.strip(),
            "config": state["config"],
            "execution_status": "aborted",
            "abort_stage": "grasp",
            "abort_reason": "no_v_jaw_contact",
            "metrics": {
                "coax_mm": None,
                "tilt_deg": None,
                "contacts": None,
                "slip_mm": None,
                "pull_mm": None,
                "jaw_force_N": None,
                "grasp_tilt_deg": grasp_tilt,
                "grasp_contact_count": 0,
                "clamp_swing_mm": None,
                "stock_ecc_mm": None,
                "rod_external_force_max_N": state.get(
                    "rod_force_max", 0.0),
                "robot_machine_contact_max": state.get(
                    "robot_machine_contact_max", 0),
                "shaft_contact_max": state.get("shaft_contact_max", 0),
                "penetration_min_m": float(PEN["min"]),
                "penetration_pair": PEN["pair"],
                "penetration_phase": PEN["label"],
            },
            "hard_pass": False,
            "quality_pass": None,
            "overall_pass": False,
            "hard_failures": ["grasp_empty"],
            "quality_failures": [],
            "phase_stats": state.get("phase_stats", {}),
            **_diagnostic_result_fields(state),
            "ik_solver": state.get("ik_solver", []),
            "ik_residuals_mm": state.get("ik_residuals_mm", {}),
            "stages": state.get("stage_records", []),
            "checkpoints": state.get("checkpoints", []),
        }
        result.update(_case_classification(
            result["execution_status"], evaluation_scope,
            result["hard_pass"], result["quality_pass"],
            result["overall_pass"]))
        write_result_json(result)
        return 1
    record_stage("grasp_complete")
    state["grasp_axis_target"] = None
    state["grasp_axis_active"] = False
    print("  shaft_in_grip = %s mm（按抓取规划位姿计算的夹爪-工件相对偏移）"
          % np.round(state["offset"] * 1000, 2))
    # 落座：抓起后先把料抬离台面几毫米再静置，让它在 V 槽里坐到底。
    # 【已否掉，实测无效】料"落座"的真相不是沿斜面滑，而是 V 槽被料重撑开
    # （两指间距 16.99 -> 20.77 mm，全程三爪接触=0），料跟着落到重力平衡位。
    # 抓取必须过闭合才有摩擦保持，过闭合量就是落座量，提前提一下只能吃掉
    # 0.33 / 1.43 mm。夹松一点落座变小、手臂下沉同步变大，峰谷守恒。
    # 留在代码里只为复现这张对照表。
    if SEAT_TAP > 0 and v_jaw and not carry_lock:
        _rel0 = (shaft()[2] - arm.pad_center()[2]) * 1000.0
        move3(pad_target(pick_at + np.array([0.0, 0.0, SEAT_TAP])), 260,
              grip_close, False, label="落座")
        settle(140, SM.CHUCK_OPEN_CTRL, grip_close)
        _rel1 = (shaft()[2] - arm.pad_center()[2]) * 1000.0
        state["offset"] = arm.pad_center() - shaft()
        print("  落座     提 %.0f mm 后静置：料相对爪 z %+.2f -> %+.2f mm"
              "（本步吃掉 %.2f mm）"
              % (SEAT_TAP * 1000, _rel0, _rel1, _rel0 - _rel1))
    if v_jaw and not carry_lock:
        print("  V 槽形封闭抓取：轴线偏离 %.1f deg，搬运全靠真实接触（无锁）"
              % axdev())
    elif v_jaw:
        print("  V 槽形封闭抓取：轴线偏离 %.1f deg" % axdev())
    else:
        print("  ⚠ 平板爪夹圆柱会翻（四元数倾角 %.1f deg）：搬运由运动学锁接管"
              % grasp_tilt)
    # Axis holding begins once the part is actually in the V jaws. The
    # calibrated approach is over; from here the wrist must counter-rotate
    # so the rod never tips 80 deg during the long traverse.
    state["axis_hold"] = not carry_lock
    relative_motion = state["gripper_relative_workpiece_motion"]
    if v_jaw and not carry_lock:
        site_rotation = data.site_xmat[arm.site].reshape(3, 3)
        center_offset_tool, axis_tool = _pose_in_tool_frame(
            site_rotation, shaft(), arm.pad_center(), rod_axis())
        relative_motion["reference_pose"] = {
            "stage": "carry_start_after_grasp_settle",
            "sim_time_s": float(data.time),
            "center_offset_tool_m": [
                float(value) for value in center_offset_tool],
            "axis_tool_unit": [float(value) for value in axis_tool],
        }
        relative_motion["active"] = True
        relative_motion["status"] = "observing"
        relative_motion["observation_start_stage"] = (
            "carry_start_after_grasp_settle")
    else:
        relative_motion["status"] = "not_applicable_for_carry_mode"

    # ---------------- carry: 先对轴，再直线进给 ----------------
    # 工艺顺序（真实车床上下料就是这么走的）：
    #   抬到卡盘上方 -> 横移到卡盘【外侧】-> 降到轴线高度 -> 把工件轴线
    #   转正 -> 沿 +x 直线推进到设定深度 -> 卡爪夹紧 -> 夹爪松开 -> 退出
    # 老版本是「横移时 x 已经在卡盘里，然后垂直往下捅」：工件是斜着插进
    # 卡爪的，而且一步就到 axis（整根 80 mm 只剩 1 mm 露头）。
    carry = [
        ("对中", [g[0], g[1], g[2] + 0.020], 260),
        ("抬起", [g[0], g[1], g[2] + 0.130], 420),
        ("高位横移", [stage_x, axis[1], 0.16], 900),
        ("下到轴线", [stage_x, axis[1], axis[2]], 700),
    ]
    # 直线进给：只动 x，y/z 和工件轴线由 ik_axis 全程保持。
    feed_goal = axis.copy()
    if not carry_lock:
        feed_goal[2] += FEED_Z_COMP

    if SMOOTH_PATH:
        # 抬升 / 横移 / 下降 / 进给 合成一条连续轨迹。老写法是四个
        # move3，每个都是一次 rest -> rest，等于四次完整起停。
        err = move_path([pad_target(t) for _n, t, _ns in carry]
                        + [pad_target(feed_goal)],
                        grip_close, carry_lock, label="搬运")
        print("  搬运     四段路点 -> 一条连续轨迹  IK残差=%5.1f mm "
              "同轴=%6.2f mm 轴向偏=%4.1f deg 倾斜=%4.1f deg"
              % (err * 1000, coax(), axdev(),
                 tilt_deg(data.qpos[sq + 3:sq + 7])))
        tick("⑤–⑧ 连续搬运｜对中·抬升·横移·下降")
        # 连续轨迹是开环 + 前馈，末端还剩十几毫米的稳态误差（kp=70 的
        # 下垂）。进给是唯一必须精确到位的地方，所以只在这里补一次闭环，
        # 而不是把整条轨迹切回逐段 rest-to-rest。
        err = move3(pad_target(feed_goal), FEED_STEPS, grip_close, carry_lock,
                    label="进给", iters=FEED_ITERS)
    else:
        for name, tgt, ns in carry:
            # All modes close the carry loop on the planned pad target. The
            # settled shaft pose selects the grasp target once; no mode gets
            # extra live shaft-state feedback during transport.
            err = move3(pad_target(tgt), ns,
                        grip_close, carry_lock,
                        label=name)
            ngc, fgrip = grip_state()
            a_site = (data.site_xmat[arm.site].reshape(3, 3)
                      @ state["a_local"])
            site_dev = float(np.degrees(np.arccos(
                min(1.0, abs(float(a_site[0]))))))
            print("  %-6s IK残差=%5.1f mm 轴=%s 同轴=%6.2f mm 倾斜=%4.1f deg "
                  "轴向偏=%4.1f deg 工具轴偏=%4.1f deg V槽=%d/%.0fN"
                  % (name, err * 1000, np.round(shaft(), 4), coax(),
                     tilt_deg(data.qpos[sq + 3:sq + 7]), axdev(), site_dev,
                     ngc, fgrip))
            tick({"对中": "⑤ 对中｜搬运锁接管",
                  "抬起": "⑥ 抬起",
                  "高位横移": "⑦ 高位横移｜到卡盘外侧",
                  "下到轴线": "⑧ 下到轴线｜停在卡盘外"}[name])
            if name == "高位横移" and not carry_lock:
                print("  轴线保持    %4.1f deg（搬运中持续用 5 轴位置+轴线 IK）"
                      % axdev())

        # 直线进给：只动 x，y/z 和工件轴线由 ik_axis 全程保持。
        print("  进给     目标 x=%.3f（自由端进卡爪 %.0f mm，工件留在外 %.0f mm，"
              "高度补偿 %.1f mm）"
              % (axis[0], INSERT_MM * 1000,
                 (F.SHAFT_L - INSERT_MM) * 1000, FEED_Z_COMP * 1000))
        err = move3(pad_target(feed_goal), 500, grip_close,
                    carry_lock,
                    label="进给")
    # 夹爪站位按 pad 中心量；真实碰撞再查机器人与机床的接触对。
    station_x = arm.pad_center()[0]
    machine_hits = 0
    for ci in range(data.ncon):
        c = data.contact[ci]
        if _is_robot_machine_contact(
                c.geom1, c.geom2, robot_geom_ids, machine_geom_ids):
            machine_hits += 1
    print("  进给     IK残差=%5.1f mm 轴=%s 同轴=%6.2f mm 轴向偏=%4.1f deg"
          % (err * 1000, np.round(shaft(), 4), coax(), axdev()))
    print("  夹爪站位  x=%.3f，距卡爪前端 %.0f mm，机器人-机床接触=%d"
          % (station_x, (jaw_front - station_x) * 1000, machine_hits))
    state["ik_residuals_mm"]["feed"] = float(err * 1000)
    record_stage("feed_complete", checkpoint_id="feed_complete")
    relative_motion["active"] = False
    relative_motion["observation_end_stage"] = "feed_complete"
    if relative_motion["status"] == "observing":
        relative_motion["status"] = (
            "observed" if relative_motion["overall"]["samples"] > 0
            else "insufficient_samples")
    tick("⑨ 直线进给｜沿 +x 插入")

    # ---------------- clamp: align friction solver with the chuck scene ----
    # scene_full.py sets no spec.option. Applying these from the start
    # breaks the tuned grasp, so patch only the friction-relevant fields
    # now, once the shaft is already at the chuck.
    model.opt.impratio = 5.0
    model.opt.noslip_iterations = 10
    model.opt.noslip_tolerance = 0.0
    # Reachable retreat point used by the combined release+retreat move.
    retreat = np.array([axis[0] + 0.02, axis[1] - 0.12, 0.16])

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
    if carry_lock:
        state["abs_pos"] = shaft().copy()
        # 力控一样要斜坡：这一步原来直接写 -150 N，QACC 直接溢出，
        # 「历史基线·穿模」那一项就是这么挂掉的（同轴 1230 mm）。
        for i in range(400):
            t = (i + 1) / 400.0
            s = t * t * (3.0 - 2.0 * t)
            data.ctrl[jaw_act] = (SM.CHUCK_OPEN_CTRL
                                  + s * (SM.CHUCK_CLAMP_CTRL
                                         - SM.CHUCK_OPEN_CTRL))
            data.ctrl[arm.grip_act] = grip_cmd(GRIP_OPEN)
            step_sim("locked_chuck_clamp")
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_label", []).append(state.get("label", "?"))
            state.setdefault("traj_t", []).append(float(data.time))
            lock()          # lock() 自己看 state["lock_active"]，无需外部判断
        settle(300, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN, locked=True)
        move3(np.array([axis[0] + 0.055, axis[1], axis[2] + 0.015]),
              200, GRIP_OPEN, True, SM.CHUCK_CLAMP_CTRL)
        print("  卡爪夹紧(锁) 轴=%s 同轴=%5.2f mm 接触=%d"
              % (np.round(shaft(), 4), coax(), jaw_state()[0]))
    else:
        # 真实交接，全程不写 qpos。顺序不能反：
        #   1) 夹爪先别松 —— 工件还要靠它托着
        #   2) 三爪咬上去（自由端前 INSERT_MM 在爪里，夹爪夹着尾段，
        #      轴向不重叠，两者不会抢同一段料）
        #   3) 再松夹爪 —— 此时卡爪已经咬住，工件掉不了
        # 反面教材：先松夹爪再夹紧，工件在 44 mm 内径里悬空，直接掉。
        # 交接（2026-11-07 实测定案）：
        #   1) 进给到位立即夹紧 —— 停等 100 步会让 IK 残差把料带偏 2.55 mm
        #   2) support_rod() 托住悬空段，等效真机导套/托料架
        #   3) 三爪闭合期手臂柔顺，让卡盘能把料找正
        #   4) 夹完恢复刚性，再松爪退出
        # 已否掉、别再重试：预收托圈、换卡爪相位、抬夹紧力、两段闭合、
        # 夹爪半开。详见 CURRENT.md。
        if tuned:
            # 料"尾端翘起"不是 geometry 问题，是舵机太软：同样的目标姿态，
            # 挂上 80 mm 料后实际会低约 1°（两端高差 1.7 mm）。这里按实测
            # 仰角做积分补偿——把 IK 目标过冲同样的角度，让实际落到水平。
            # 只改机器人关节目标，不给料加任何外力（棒料 qfrc 仍为 0）。
            def level_rod(iters=4, ns=160):
                trail, comp = [], 0.0
                for _ in range(iters):
                    pitch = rod_pitch()
                    trail.append(pitch)
                    if abs(pitch) < 0.12:
                        break
                    comp += -pitch
                    th = np.radians(comp)
                    tgt = np.array([np.cos(th), 0.0, np.sin(th)])
                    R_site = data.site_xmat[arm.site].reshape(3, 3)
                    a_local = R_site.T @ rod_axis()
                    q_now = np.array([data.qpos[a] for a in arm.qadr])
                    q_goal, _pe, _ae = arm.ik_axis(
                        data.site_xpos[arm.site].copy(), tgt, a_local,
                        q_init=q_now, iters=400)
                    for i in range(ns):
                        f = (i + 1) / float(ns)
                        s = f * f * (3.0 - 2.0 * f)
                        for k, a in enumerate(arm.qadr):
                            data.ctrl[a] = (q_now[k]
                                            + s * (q_goal[k] - q_now[k]))
                        data.ctrl[arm.grip_act] = grip_cmd(grip_close)
                        data.ctrl[jaw_act] = SM.CHUCK_OPEN_CTRL
                        step_sim("axis_leveling")
                        state.setdefault("traj", []).append(
                            data.site_xpos[arm.site].copy())
                        state.setdefault("traj_label", []).append("料轴调平")
                        state.setdefault("traj_t", []).append(float(data.time))
                return trail

            if LEVEL_ROD:
                _trail = level_rod()
                print("  料轴调平     %s -> %+.2f deg（%d 轮）"
                      % (" ".join("%+.2f" % v for v in _trail),
                         rod_pitch(), len(_trail)))
            close_bushing()
        if ARM_KP_SCALE != 1.0:
            set_arm_stiff(ARM_KP_SCALE)
            print("  伺服刚度    kp x%.1f（%.1f -> %.1f N·m/rad）"
                  % (ARM_KP_SCALE, _KP0[0], _KP0[0] * ARM_KP_SCALE))
        print("  夹紧前      轴=%s 同轴=%5.2f mm 轴向偏=%4.1f deg 翘向=%+.2f deg"
              % (np.round(shaft(), 4), coax(), axdev(), rod_pitch()))
        # 交接期下坠归因：把「料」和「爪」分开记。爪子下沉 = 手臂伺服在
        # 负载下让位；料相对爪下沉 = 料在 V 槽里滑。两者必须分开量，
        # 否则会把手臂软当成夹持打滑（上一轮就是这么误判的）。
        state["clamp_arm_z0"] = float(arm.pad_center()[2])
        state["clamp_rod_z0"] = float(shaft()[2])
        state["clamp_rel_z0"] = (state["clamp_rod_z0"]
                                 - state["clamp_arm_z0"])
        state["arm_drop_min"] = 0.0
        state["arm_drop_max"] = 0.0
        state["rel_drop_min"] = 0.0
        state["rel_drop_max"] = 0.0
        state["rod_drop_min"] = 0.0

        def clamp_stat():
            _pc = arm.pad_center()
            _p = shaft()
            _a = (float(_pc[2]) - state["clamp_arm_z0"]) * 1000.0
            _r = (float(_p[2]) - state["clamp_rod_z0"]) * 1000.0
            _rel = (_r - _a)
            state["arm_drop_min"] = min(state["arm_drop_min"], _a)
            state["arm_drop_max"] = max(state["arm_drop_max"], _a)
            state["rel_drop_min"] = min(state["rel_drop_min"], _rel)
            state["rel_drop_max"] = max(state["rel_drop_max"], _rel)
            state["rod_drop_min"] = min(state["rod_drop_min"], _r)
            return _a, _r, _rel
        # qfrc_applied 是持久量，设一次持续到解锁前撤销。
        support_rod()
        settle(CLAMP_SETTLE, SM.CHUCK_OPEN_CTRL, grip_close, floating=True)
        if CONVERGE_STEPS > 0:
            # 文献 B：夹紧前让伺服收敛。不能用固定停等 —— 实测停 100 步
            # 会把料从同轴 0.08 带到 2.55 mm。这里目标跟随实际角（不拖拽
            # 工件），只等关节速度降到阈值就立刻交接。
            _qvel, _used = 9.9, 0
            for _j in range(CONVERGE_STEPS):
                data.ctrl[jaw_act] = SM.CHUCK_OPEN_CTRL
                data.ctrl[arm.grip_act] = grip_cmd(grip_close)
                float_arm()
                step_sim("servo_convergence")
                _used = _j + 1
                _qvel = float(np.linalg.norm(
                    [data.qvel[a] for a in arm.qadr]))
                state.setdefault("traj", []).append(
                    data.site_xpos[arm.site].copy())
                state.setdefault("traj_t", []).append(float(data.time))
                state.setdefault("traj_label", []).append("伺服收敛")
                if _qvel < CONVERGE_TOL and _j >= 20:
                    break
            print("  伺服收敛    等待%3d 步 关节速度=%.4f rad/s 同轴=%5.2f mm"
                  % (_used, _qvel, coax()))
        # 交接起点在进给到位时取，两种顺序共用同一基准，夹紧瞬态才可比。
        clamp_p0 = shaft().copy()
        if CLAMP_ANCHOR and tuned:
            for _a in arm.qadr:
                data.ctrl[_a] = float(data.qpos[_a])
            print("  终点锚定    伺服目标同步到实际角（不再追 IK 残差）")
        seat_x0 = float(clamp_p0[0])
        clamp_z_min = clamp_p0[2]
        clamp_z_max = clamp_p0[2]
        jaw_from = SM.CHUCK_OPEN_CTRL
        clamp_grip = grip_close
        if tuned and HANDOVER_FIRST:
            # 文献顺序：卡爪闭合【之前】先松开夹爪，手臂原地不动并保持
            # soft floating；料由竖直托力 + 轴向端面弹簧定位托住，不会掉。
            # 三爪闭合时不存在「两边都夹紧」的较劲期。
            print("  交接方式    先松夹爪 -> 端面定位 -> 三爪闭合"
                  "（k=%.0f c=%.0f 停留%d步）"
                  % (SEAT_K, SEAT_C, SEAT_DWELL))
            for i in range(360):
                t = (i + 1) / 360.0
                s = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
                data.ctrl[jaw_act] = SM.CHUCK_OPEN_CTRL
                grip_direct(grip_close + s * (GRIP_OPEN - grip_close))
                seat_rod(seat_x0)
                float_arm()
                step_sim("先松夹爪")
                _z = float(shaft()[2])
                clamp_z_min = min(clamp_z_min, _z)
                clamp_z_max = max(clamp_z_max, _z)
                state.setdefault("traj", []).append(
                    data.site_xpos[arm.site].copy())
                state.setdefault("traj_label", []).append("先松夹爪")
                state.setdefault("traj_t", []).append(float(data.time))
                track_pen("先松夹爪")
                if frame_hook is not None and i % FRAME_EVERY == 0:
                    tick()
            # 力控就位手册要求夹紧前留足 dwell 让料坐实：
            # clamping before the part has slipped to full depth locks a
            # proud part into the jaws。
            for _ in range(SEAT_DWELL):
                data.ctrl[jaw_act] = SM.CHUCK_OPEN_CTRL
                data.ctrl[arm.grip_act] = grip_cmd(GRIP_OPEN)
                seat_rod(seat_x0)
                float_arm()
                step_sim("端面就位")
                _z = float(shaft()[2])
                clamp_z_min = min(clamp_z_min, _z)
                clamp_z_max = max(clamp_z_max, _z)
                state.setdefault("traj", []).append(
                    data.site_xpos[arm.site].copy())
                state.setdefault("traj_label", []).append("端面就位")
                state.setdefault("traj_t", []).append(float(data.time))
            print("  夹爪先松    轴=%s 同轴=%5.2f mm 接触=%d"
                  % (np.round(shaft(), 4), coax(), jaw_state()[0]))
            clamp_grip = GRIP_OPEN
        else:
            print("  交接方式    夹爪保持，三爪闭合后原地松爪（当前默认）")
        if PRECLAMP:
            # 文献 C：低压贴料 -> 端面推料器把料顶正 -> 高压终夹。
            # 关键不是「把位移斜坡拆慢」（CLAMP_TWOSTAGE 已否掉），而是
            # 中间留一段「三爪轻贴、料还能微动」的窗口给推料器。
            print("  低压预夹    丝杠目标=%+.4f m -> 端面顶正 %d 步 -> 高压终夹"
                  % (PRECLAMP_CTRL, PRECLAMP_DWELL))
            for i in range(PRECLAMP_STEPS):
                t = (i + 1) / PRECLAMP_STEPS
                s = t * t * (3.0 - 2.0 * t)
                data.ctrl[jaw_act] = (jaw_from
                                      + s * (PRECLAMP_CTRL - jaw_from))
                data.ctrl[arm.grip_act] = grip_cmd(clamp_grip)
                pusher_rod(seat_x0)
                seat_rod(seat_x0)
                float_arm()
                step_sim("低压预夹")
                clamp_p = shaft()
                clamp_z_min = min(clamp_z_min, float(clamp_p[2]))
                clamp_z_max = max(clamp_z_max, float(clamp_p[2]))
                state.setdefault("traj", []).append(
                    data.site_xpos[arm.site].copy())
                state.setdefault("traj_label", []).append("低压预夹")
                state.setdefault("traj_t", []).append(float(data.time))
                track_pen("低压预夹")
            print("  预夹贴料    同轴=%5.2f mm 接触=%d"
                  % (coax(), jaw_state()[0]))
            for _ in range(PRECLAMP_DWELL):
                data.ctrl[jaw_act] = PRECLAMP_CTRL
                data.ctrl[arm.grip_act] = grip_cmd(clamp_grip)
                pusher_rod(seat_x0)
                seat_rod(seat_x0)
                float_arm()
                step_sim("端面顶正")
                clamp_p = shaft()
                clamp_z_min = min(clamp_z_min, float(clamp_p[2]))
                clamp_z_max = max(clamp_z_max, float(clamp_p[2]))
                state.setdefault("traj", []).append(
                    data.site_xpos[arm.site].copy())
                state.setdefault("traj_label", []).append("端面顶正")
                state.setdefault("traj_t", []).append(float(data.time))
                track_pen("端面顶正")
            print("  端面顶正    同轴=%5.2f mm 接触=%d"
                  % (coax(), jaw_state()[0]))
            # 高压终夹前撤掉推料器：卡爪接手后不再顶，避免残余轴向力
            # 一路带到拉拔测试。qfrc_applied 是持久量，必须显式清零。
            data.qfrc_applied[sv] = 0.0
            jaw_from = PRECLAMP_CTRL
        # 单段连续斜坡：丝杠从张开位（或预夹位）拧到夹紧力上限。
        state["label"] = "三爪拧紧"
        _seat_pads = None
        if SEAT_TRACE:
            _seat_pads = ([gid(mujoco.mjtObj.mjOBJ_GEOM,
                               "fixed_jaw_pad_v%d" % _k) for _k in (1, 2)],
                          [gid(mujoco.mjtObj.mjOBJ_GEOM,
                               "moving_jaw_pad_v%d" % _k) for _k in (1, 2)])
            print("    SEAT 起始 指距=%.3f mm"
                  % (np.linalg.norm(
                      np.mean(data.geom_xpos[_seat_pads[1]], axis=0)
                      - np.mean(data.geom_xpos[_seat_pads[0]], axis=0)) * 1000))
        for i in range(500):
            t = (i + 1) / 500.0
            s = t * t * (3.0 - 2.0 * t)
            data.ctrl[jaw_act] = (jaw_from
                                  + s * (SM.CHUCK_CLAMP_CTRL - jaw_from))
            data.ctrl[arm.grip_act] = grip_cmd(clamp_grip)
            seat_rod(seat_x0)
            float_arm()
            step_sim("三爪拧紧")
            clamp_p = shaft()
            clamp_z_min = min(clamp_z_min, float(clamp_p[2]))
            clamp_z_max = max(clamp_z_max, float(clamp_p[2]))
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_label", []).append("三爪拧紧")
            state.setdefault("traj_t", []).append(float(data.time))
            track_pen("三爪拧紧")
            _a, _r, _rel = clamp_stat()
            if _seat_pads is not None and i % 10 == 0 and i <= 200:
                _gap = (np.mean(data.geom_xpos[_seat_pads[1]], axis=0)
                        - np.mean(data.geom_xpos[_seat_pads[0]], axis=0))
                print("    SEAT %3d 指距=%.3f mm 爪沉=%+.2f 相对=%+.2f 接触=%d"
                      % (i, float(np.linalg.norm(_gap)) * 1000,
                         _a, _rel, jaw_state()[0]))
            if CLAMP_TRACE and i % 25 == 0:
                _p, _pc = shaft(), arm.pad_center()
                print("    TRACE %3d 丝杠=%+.4f 料(y,z)=(%+.2f,%+.2f) "
                      "爪(y,z)=(%+.2f,%+.2f) 卡爪接触=%d "
                      "| 爪沉%+.2f 料沉%+.2f 相对%+.2f"
                      % (i, float(data.ctrl[jaw_act]),
                         (_p[1] - axis[1]) * 1000, (_p[2] - axis[2]) * 1000,
                         (_pc[1] - axis[1]) * 1000,
                         (_pc[2] - axis[2]) * 1000, jaw_state()[0],
                         _a, _r, _rel))
            if frame_hook is not None and i % FRAME_EVERY == 0:
                tick()
        # Keep the same 300-step post-clamp hold as settle(), but sample the
        # workpiece each step so a brief drop-and-rebound cannot be hidden by
        # the later gravity-slip measurement.
        for _ in range(300):
            data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
            data.ctrl[arm.grip_act] = grip_cmd(clamp_grip)
            seat_rod(seat_x0)
            float_arm()
            step_sim("三爪稳定")
            clamp_p = shaft()
            clamp_z_min = min(clamp_z_min, float(clamp_p[2]))
            clamp_z_max = max(clamp_z_max, float(clamp_p[2]))
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_label", []).append("三爪稳定")
            state.setdefault("traj_t", []).append(float(data.time))
            track_pen("三爪稳定")
            _a, _r, _rel = clamp_stat()
        clamp_p = shaft()
        nc, fN = jaw_state()
        bn, bf = sleeve_state()
        print("  交接外力    棒料 qfrc 峰值=%.3f N  关节重力补偿=%s"
              % (state.get("rod_force_max", 0.0),
                 "开" if ARM_GCOMP else "关"))
        print("  三爪咬合     轴=%s 同轴=%5.2f mm 接触=%d 力=%.1f N "
              "导套=%d/%.1fN 翘向=%+.2f deg"
              % (np.round(shaft(), 4), coax(), nc, fN, bn, bf, rod_pitch()))
        if BORE_PROBE:
            _jq = [float(data.qpos[model.jnt_qposadr[
                gid(mujoco.mjtObj.mjOBJ_JOINT, "jaw_slide_%d" % _k)]])
                for _k in range(3)]
            _qb = (SM.JAW_ARC_Q0 if SM.JAW_ARC_BORE_Q is None
                   else SM.JAW_ARC_BORE_Q)
            print("  受载爪位    q=[%+.5f %+.5f %+.5f] m 镗爪位=%+.5f "
                  "几何过盈=%+.2f mm 同步差=%.3f mm"
                  % (_jq[0], _jq[1], _jq[2], _qb,
                     (_qb - _jq[0]) * 1000,
                     (max(_jq) - min(_jq)) * 1000))
        print("  夹紧瞬态     z变化=%+.2f mm  峰值范围=[%+.2f, %+.2f] mm  "
              "y/z终点偏移=(%+.2f,%+.2f) mm"
              % ((clamp_p[2] - clamp_p0[2]) * 1000,
                 (clamp_z_min - clamp_p0[2]) * 1000,
                 (clamp_z_max - clamp_p0[2]) * 1000,
                 (clamp_p[1] - clamp_p0[1]) * 1000,
                 (clamp_p[2] - clamp_p0[2]) * 1000))
        record_stage("three_jaw_clamped", checkpoint_id="chuck_clamped")
        # 交接全程的 z 峰谷 = 人眼看到的「下沉—回升」。slip_mm 只在解锁
        # 静置后测，完全覆盖不到这一段，必须单列并纳入判据，否则又会
        # 出现「指标 0.02 mm 但画面明显在动」。采样延伸到松爪/退出段。
        # 夹紧完成：手臂恢复刚性，目标就钉在当前实际角（柔顺期间
        # ctrl 已跟随 qpos，不会跳变）。松爪/退出不能再让位。
        _FLOAT[0] = 0.0
        if ARM_GCOMP:
            # 恢复刚性前撤掉机器人关节的重力补偿，不能带进松爪/退出/拉拔。
            for d in arm.dadr:
                data.qfrc_applied[d] = 0.0
        release_bushing()
        # 卡爪已提供轴向/径向约束，撤掉端面定位与导套弹簧。qfrc_applied
        # 是持久量，不撤销会一路带到拉拔测试里；竖直恢复成纯托力。
        data.qfrc_applied[sv] = 0.0
        data.qfrc_applied[sv + 1] = 0.0
        data.qfrc_applied[sv + 2] = (
            _lift * float(model.body_mass[shaft_body]
                               * abs(model.opt.gravity[2])))
        if tuned and HANDOVER_FIRST and SEAT_RADIAL_K > 0 and SEAT_FADE_STEPS > 0:
            # 径向支撑不一步清零：按 quintic 把刚度+阻尼同步降到 0，让
            # 三爪在几百步里逐步接手全部径向载荷，弹簧储能不瞬间释放。
            mg2 = float(model.body_mass[shaft_body]
                        * abs(model.opt.gravity[2]))
            cr2 = (SEAT_RADIAL_C if SEAT_RADIAL_C > 0 else
                   1.4 * np.sqrt(SEAT_RADIAL_K
                                 * model.body_mass[shaft_body]))
            for j in range(SEAT_FADE_STEPS):
                t = (j + 1) / SEAT_FADE_STEPS
                s = 1.0 - t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
                p = shaft()
                data.qfrc_applied[sv + 1] = s * (
                    SEAT_RADIAL_K * (axis[1] - float(p[1]))
                    - cr2 * float(data.qvel[sv + 1]))
                data.qfrc_applied[sv + 2] = (
                    _lift * mg2
                    + s * (SEAT_RADIAL_K * (axis[2] - float(p[2]))
                           - cr2 * float(data.qvel[sv + 2])))
                data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
                data.ctrl[arm.grip_act] = grip_cmd(clamp_grip)
                step_sim("卸载托力")
                _z = float(shaft()[2])
                clamp_z_min = min(clamp_z_min, _z)
                clamp_z_max = max(clamp_z_max, _z)
                state.setdefault("traj", []).append(
                    data.site_xpos[arm.site].copy())
                state.setdefault("traj_label", []).append("卸载托力")
                state.setdefault("traj_t", []).append(float(data.time))
            print("  卸载托力    %d 步渐变 同轴=%5.2f mm 接触=%d"
                  % (SEAT_FADE_STEPS, coax(), jaw_state()[0]))
        # 三爪夹住之后，机械臂先【原地】松开夹爪，等 V 槽刀片完全脱离，
        # 再移走。边松边退会让还没脱离的刀片横向刮着棒料，把工件从卡爪
        # 里带出去（实测同轴 1.6 mm -> 283 mm，工件直接被带飞）。
        for i in range(360):
            t = (i + 1) / 360
            s = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
            data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
            grip_direct(clamp_grip + s * (GRIP_OPEN - clamp_grip))
            step_sim("原地松爪")
            _z = float(shaft()[2])
            clamp_z_min = min(clamp_z_min, _z)
            clamp_z_max = max(clamp_z_max, _z)
            state.setdefault("traj", []).append(
                data.site_xpos[arm.site].copy())
            state.setdefault("traj_label", []).append("原地松爪")
            state.setdefault("traj_t", []).append(float(data.time))
            track_pen("原地松爪")
            if frame_hook is not None and i % FRAME_EVERY == 0:
                tick()
        settle(120, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)
        release_p = shaft()
        _z = float(release_p[2])
        clamp_z_min = min(clamp_z_min, _z)
        clamp_z_max = max(clamp_z_max, _z)
        print("  原地松爪     同轴=%5.2f mm 轴向偏=%4.1f deg 接触=%d "
              "相对夹紧终点 dz=%+.2f mm 翘向=%+.2f deg"
              % (coax(), axdev(), jaw_state()[0],
                 (release_p[2] - clamp_p[2]) * 1000, rod_pitch()))
        record_stage(
            "gripper_released_in_place",
            checkpoint_id="gripper_released_in_place",
        )
        # 退出必须分两段：先沿轴线退到料尾之外，再侧向撤离。一次斜着
        # 撤退时，张开的刀片会横刮还没脱离的料尾，把卡盘里刚定好心的
        # 料带歪（-5 mm 站点实测：同轴 0.44 -> 1.20 mm、倾斜 1.2 -> 2.0
        # deg，余量偏心因此 2.45 mm 判 FAIL）。分两段后侧移段不再碰料。
        # 退出段不能再套 part_offset()：料已经归卡盘，夹爪和料分开了，
        # 实时量出来的「pad − 料」是个 40 mm 的大偏置，套上去等于让手臂
        # 去追一个歪掉的目标（原实现实测残差 41.6 mm）。这里直接给世界点。
        if RETREAT_AXIAL and os.environ.get("FEED_DIAG_NO_RETREAT") != "1":
            pull = np.array([axis[0] - 0.070, axis[1], axis[2] + 0.010])
            move3(pull, 260, GRIP_OPEN, False, SM.CHUCK_CLAMP_CTRL)
            print("  轴向退出     沿轴线退 %.0f mm，刀片已离开料尾"
                  % (70.0))
        elif os.environ.get("FEED_DIAG_NO_RETREAT") == "1":
            print("  诊断对照     松爪后不移动机械臂（仅定位用）")
        checkpoint_statuses = _feed_checkpoint_statuses(
            RETREAT_AXIAL,
            os.environ.get("FEED_DIAG_NO_RETREAT") == "1",
        )
        record_stage(
            "after_axial_retreat",
            checkpoint_id="axial_retreat_complete",
            checkpoint_status=checkpoint_statuses[
                "axial_retreat_complete"],
        )
        # 侧向撤离从「已在料尾外侧」起步，行程比原来长 70 mm。
        if os.environ.get("FEED_DIAG_NO_RETREAT") == "1":
            rerr = 0.0
        else:
            rerr = move(retreat, 1100, GRIP_OPEN, False,
                        SM.CHUCK_CLAMP_CTRL)
        record_stage(
            "after_lateral_retreat",
            checkpoint_id="lateral_retreat_complete",
            checkpoint_status=checkpoint_statuses[
                "lateral_retreat_complete"],
        )
        retreat_p = shaft()
        _z = float(retreat_p[2])
        clamp_z_min = min(clamp_z_min, _z)
        clamp_z_max = max(clamp_z_max, _z)
        state["clamp_swing_mm"] = (clamp_z_max - clamp_z_min) * 1000
        print("  交接位移     z峰谷=%.2f mm" % state["clamp_swing_mm"])
        print("  位移归因     爪沉[%+.2f,%+.2f] 料相对爪[%+.2f,%+.2f] "
              "= 手臂伺服 %.0f%% / 槽内滑移 %.0f%%"
              % (state["arm_drop_min"], state["arm_drop_max"],
                 state["rel_drop_min"], state["rel_drop_max"],
                 100.0 * abs(state["arm_drop_min"]) / max(
                     1e-9, abs(state["arm_drop_min"])
                     + abs(state["rel_drop_min"])),
                 100.0 * abs(state["rel_drop_min"]) / max(
                     1e-9, abs(state["arm_drop_min"])
                     + abs(state["rel_drop_min"]))))
        print("  松手退出     IK残差=%5.1f mm 同轴=%5.2f mm "
              "轴向偏=%4.1f deg 接触=%d 相对原地松爪 dz=%+.2f mm"
              % (rerr * 1000, coax(), axdev(), jaw_state()[0],
                 (retreat_p[2] - release_p[2]) * 1000))
        if ARM_KP_SCALE != 1.0:
            set_arm_stiff(1.0)
    tick("⑩ 三爪夹紧｜单丝杠联动")

    # Hand the shaft back to PHYSICS before any load test.
    # 撤掉交接期的托力/轴向定位，否则重力下沉、拉拔指标会被托成假通过。
    data.qfrc_applied[sv] = 0.0
    data.qfrc_applied[sv + 1] = 0.0
    data.qfrc_applied[sv + 2] = 0.0
    state["lock_active"] = False
    record_stage("before_unlock_settle")
    settle(1600, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)
    bn, bf = sleeve_state()
    print("  解锁静置     轴=%s 同轴=%5.2f mm 轴向偏=%5.1f deg 接触=%d "
          "导套=%d/%.1fN"
          % (np.round(shaft(), 4), coax(), axdev(), jaw_state()[0], bn, bf))
    tick("⑪ 撤销临时支撑｜自由物理静置")
    record_stage("after_unlock_settle", checkpoint_id="unlock_settled")

    # The full-physics path has already released and retreated in one move.
    # Only the legacy kinematic-lock path still needs its separate retreat.
    if carry_lock:
        move3(retreat, 400, GRIP_OPEN, False,
              SM.CHUCK_CLAMP_CTRL)
    settle(800, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)

    final_coax = coax()
    stock = stock_ecc()
    state["stock_ecc_mm"] = stock
    # A cylinder may ROLL about its own axis freely; tilt_deg() reports that
    # harmless roll as >100 deg. What matters for a chuck is the angle
    # between the rod AXIS and the spindle axis, which axdev() measures.
    final_tilt = axdev()
    contacts, force = jaw_state()
    print("  松开退出     轴=%s 同轴=%5.2f mm 轴向偏=%5.1f deg "
          "接触=%d 三爪法向力=%.1f N 翘向=%+.2f deg"
          % (np.round(shaft(), 4), final_coax, final_tilt,
             contacts, force, rod_pitch()))
    record_stage("robot_clear")
    tick("⑫ 机器人退出｜装夹完成")

    z0 = shaft()[2]
    settle(800, SM.CHUCK_CLAMP_CTRL, GRIP_OPEN)
    slip = abs(z0 - shaft()[2]) * 1000
    x0 = shaft()[0]
    state["diagnostic_scope"] = "pull_test"
    for i in range(700):
        data.ctrl[jaw_act] = SM.CHUCK_CLAMP_CTRL
        data.xfrc_applied[shaft_body, 0] = -20.0
        step_sim("20N_axial_pull_test")
        if frame_hook is not None and i % (FRAME_EVERY * 2) == 0:
            tick()
    pull = abs(shaft()[0] - x0) * 1000
    record_stage("after_20N_pull_test")
    tick("⑬ 20 N 拉拔｜位移 %.2f mm" % pull)
    print("  重力下沉  %.2f mm      20 N 拉  %.2f mm" % (slip, pull))
    print()

    # ---- 末端轨迹平滑度（远控画质看不出抖，只能看数）----
    # 「抖一抖」在数据里的样子：速度在相邻几步里反复加速/减速，加速度
    # 尖峰多、且与中位数差几个量级。平滑轨迹的加速度应该是一个量级。
    tr = np.asarray(state.get("traj", []))
    if len(tr) > 40:
        dt = model.opt.timestep
        v = np.linalg.norm(np.diff(tr, axis=0), axis=1) / dt
        a = np.diff(v) / dt
        amed = float(np.median(np.abs(a))) + 1e-9
        spikes = int(np.sum(np.abs(a) > 8.0 * amed))
        dv = np.diff(v)
        flips = int(np.sum((dv[:-1] > 1e-4) & (dv[1:] < -1e-4)))
        print("  末端速度 最大 %.3f m/s  中位 %.3f m/s  加速度 RMS %.1f m/s^2"
              % (v.max(), float(np.median(v)), float(np.sqrt((a ** 2).mean()))))
        print("  加速度尖峰 %d 次 / %d 步（>8x 中位 %.1f）  加速-减速切换 %d 次"
              % (spikes, len(v), amed, flips))
        labs = state.get("traj_label", [])[:-1]
        worst = []
        for lab in dict.fromkeys(labs):
            mask = np.array([l == lab for l in labs])
            if mask.sum() > 5:
                worst.append((float(v[mask].max()), lab))
        # 「抖」的可视化等价物：末端轨迹相对其自身滑动平均的残差。
        # 眼睛看到的抖动幅度就是这个数，和加速度尖峰无关。
        _k = 11
        _sm = np.vstack([np.convolve(tr[:, c], np.ones(_k) / _k, "valid")
                         for c in range(3)]).T
        _res = tr[_k // 2:-(_k // 2)] - _sm
        _wob = np.sqrt((_res ** 2).sum(axis=1))
        _mov = v > 0.03
        print("  末端抖动幅度 全段 RMS %.3f mm / 最大 %.3f mm"
              % (np.sqrt((_wob ** 2).mean()) * 1000, _wob.max() * 1000))
        if _mov[:len(_wob)].sum() > 10:
            print("                运动段 RMS %.3f mm / 最大 %.3f mm"
                  % (np.sqrt((_wob[_mov[:len(_wob)]] ** 2).mean()) * 1000,
                     _wob[_mov[:len(_wob)]].max() * 1000))
        labs_w = labs[_k // 2: _k // 2 + len(_wob)]
        per = {}
        for _lab, _w in zip(labs_w, _wob):
            per.setdefault(_lab, []).append(_w)
        rows = sorted(((max(ws) * 1000, float(np.sqrt(np.mean(np.square(ws)))) * 1000, _lab)
                       for _lab, ws in per.items() if len(ws) > 20), reverse=True)
        _rp = np.asarray(state.get("rod_p", []))
        if len(_rp) > 20:
            _rlab = state.get("rod_lab", [])
            _rsm = np.vstack([np.convolve(_rp[:, c], np.ones(11) / 11, "valid")
                              for c in range(3)]).T
            _rres = _rp[5:5 + len(_rsm)] - _rsm
            # 只算横向（y/z）：轴向的位移是进给本身，混进来会把
            # 「插进去」误判成「抖」。
            _rw = np.sqrt((_rres[:, 1:3] ** 2).sum(axis=1)) * 1000
            _rl = _rlab[5:5 + len(_rw)]
            _agg = {}
            for _l, _w, _f in zip(_rl, _rw, state["rod_f"][5:5 + len(_rw)]):
                a = _agg.setdefault(_l, [0.0, 0.0])
                a[0] = max(a[0], _w)
                a[1] = max(a[1], _f)
            print("  工件抖动/接触力：%s"
                  % "  ".join("%s %.2fmm/%.0fN" % (l, v[0], v[1])
                              for l, v in sorted(_agg.items(),
                                                 key=lambda kv: -kv[1][0])[:6]))
        print("  分段抖动 最大/RMS（mm）：%s"
              % "  ".join("%s %.2f/%.2f" % (l, mx, rm) for mx, rm, l in rows[:6]))
        worst.sort(reverse=True)
        print("  分段最大末端速度：%s"
              % "  ".join("%s %.2f" % (l, s) for s, l in worst[:6]))
        # 「一跳一跳」的量化：速度从近零重新涨起来 = 一次新的起步。
        # 一条连续轨迹每段只能起步 1 次；数值 = 该段闭环轮数（每轮
        # move() 都从自相位重新加速）或路点之间的空等。
        rest = []
        for lab in dict.fromkeys(labs):
            mask = np.array([l == lab for l in labs])
            vv = v[mask]
            if len(vv) < 8:
                continue
            # 绝对阈值 8 mm/s：低于这个速度人眼看就是停住了。
            # 用「百分比的峰值」会把慢段的正常运动也算成静止。
            thr = 0.008
            mv = vv > thr
            n_start = int(np.sum((~mv[:-1]) & mv[1:]))
            n_rest = int((~mv).sum())
            rest.append("  %-4s 起步%d次 静止%.2fs/%.2fs 峰值%.2f"
                        % (lab, n_start, n_rest * dt, len(vv) * dt, vv.max()))
        if rest:
            print("  分段起步/静止：")
            for line in rest:
                print(line)
        print()

    # 全物理交接才有 clamp_swing_mm；运动学锁基线不测交接/余量判据。
    swing = state.get("clamp_swing_mm")
    stock = state.get("stock_ecc_mm")
    handover_ok = (swing is None or
                   (swing <= CLAMP_SWING_LIMIT
                    and (stock is None or stock <= STOCK_ALLOW_MM)))
    hard_failures = []
    quality_failures = []
    if not state.get("ik_residuals_mm"):
        hard_failures.append("no_ik_result")
    if state.get("grasp_contact_count", 0) <= 0:
        if v_jaw and not carry_lock:
            hard_failures.append("grasp_empty")
    if contacts < CONTACT_MIN:
        hard_failures.append("chuck_contact_insufficient")
    if pull > PULL_LIMIT:
        hard_failures.append("pull_out")
    if state.get("robot_machine_contact_max", 0) > 0:
        hard_failures.append("robot_machine_collision")
    # The tuned V-jaw handoff is the no-hidden-force experiment. The
    # official compatibility scene still uses its historical gravity support
    # because it has no physical bushing geometry; keep that scene separate
    # instead of calling its legacy support a tuned-path pass.
    if tuned and state.get("rod_force_max", 0.0) > 1e-6:
        hard_failures.append("rod_external_force")
    if final_coax > COAX_LIMIT:
        quality_failures.append("coaxiality")
    if final_tilt > TILT_LIMIT:
        quality_failures.append("axis_tilt")
    if slip > SLIP_LIMIT:
        quality_failures.append("gravity_slip")
    if swing is not None and swing > CLAMP_SWING_LIMIT:
        quality_failures.append("handover_swing")
    if stock is not None and stock > STOCK_ALLOW_MM:
        quality_failures.append("stock_eccentricity")
    hard_pass = not hard_failures
    quality_pass = not quality_failures
    ok = hard_pass and quality_pass
    print("=" * 74)
    print("FEED_METRIC coax_mm=%.2f tilt_deg=%.1f contacts=%d "
          "slip_mm=%.2f pull_mm=%.2f jaw_force_N=%.1f "
          "grasp_tilt_deg=%.1f carry=%s physics_after_clamp=%d"
          % (final_coax, final_tilt, contacts, slip, pull, force,
             grasp_tilt, "kinematic_lock" if carry_lock else "physics",
             0 if state["lock_active"] else 1))
    if swing is not None:
        print("FEED_METRIC_HANDOVER clamp_swing_mm=%.2f guard=%.1f"
              % (swing, CLAMP_SWING_LIMIT))
    if stock is not None:
        print("FEED_METRIC_MACHINING stock_ecc_mm=%.2f allow_mm=%.1f"
              % (stock, STOCK_ALLOW_MM))
    print("FEED_METRIC %s" % ("PASS" if ok else "FAIL"))
    pickup_motion = state["pickup_motion_diagnostics"]
    if pickup_motion["status"] == "observed":
        print("阶段诊断：抓取接近期间，工件中心最多移动 %.3f mm，"
              "轴线最多变化 %.3f°（仅观察，不单独判 PASS/FAIL）。"
              % (pickup_motion["overall"]["max_center_displacement_mm"],
                 pickup_motion["overall"]["max_shaft_axis_change_deg"]))
    relative_motion = state["gripper_relative_workpiece_motion"]
    if relative_motion["status"] == "observed":
        print("阶段诊断：夹爪搬运/进给期间，工件相对夹爪中心最多滑移 "
              "%.3f mm，轴线最多变化 %.3f°（仅观察，尚无独立门槛）。"
              % (relative_motion["overall"]["max_translation_slip_mm"],
                 relative_motion["overall"][
                     "max_relative_axis_change_deg"]))
    print("结果翻译：")
    print("  coax_mm：最终棒料中心离卡盘主轴轴线 %.2f mm（越小越居中）。"
          % final_coax)
    print("  tilt_deg：最终棒料轴线与卡盘轴线相差 %.1f°。"
          % final_tilt)
    print("  contacts：棒料与卡盘爪之间统计到 %d 个接触点，不是爪的数量。"
          % contacts)
    print("  slip_mm：机器人退出后，棒料在卡盘内静置时的滑移量 %.2f mm。"
          % slip)
    print("  pull_mm：单独施加 20 N 轴向拉力时，棒料移动 %.2f mm。"
          % pull)
    print("  jaw_force_N：三爪接触法向力汇总 %.1f N，仅作观测，不单独决定通过。"
          % force)
    print("  grasp_tilt_deg：夹住棒料那一刻的角度差 %.1f°，不是最终倾角。"
          % grasp_tilt)
    carry_cn = "物理接触搬运（未使用运动学锁）" if not carry_lock else \
        "运动学锁搬运（参考模式）"
    print("  carry：%s。" % carry_cn)
    print("  physics_after_clamp：%s。"
          % ("卡盘夹紧后继续物理仿真" if not state["lock_active"]
             else "卡盘夹紧后仍处于运动学锁定"))
    if swing is not None:
        print("  clamp_swing_mm：夹紧交接期间棒料 z 方向峰谷位移 %.2f mm；"
              "guard=%.1f mm 是项目护栏，不是精度目标。"
              % (swing, CLAMP_SWING_LIMIT))
    if stock is not None:
        print("  stock_ecc_mm：棒料相对主轴轴线的最大径向偏心 %.2f mm；"
              "allow=%.1f mm 是本项目假设的单边余量，只做几何检查。"
              % (stock, STOCK_ALLOW_MM))
    print("  PASS/FAIL：只说明本次输入在当前仿真判据下的结果，"
          "不代表所有工况或真实机械臂表现。")
    if not ok:
        print("  未通过判据：同轴≤%.1fmm 倾斜≤%.1fdeg 接触≥%d "
              "下沉≤%.1fmm 20N拉≤%.1fmm 交接甩料≤%.1fmm 余量偏心≤%.1fmm"
              % (COAX_LIMIT, TILT_LIMIT, CONTACT_MIN,
                 SLIP_LIMIT, PULL_LIMIT, CLAMP_SWING_LIMIT, STOCK_ALLOW_MM))
    print("=" * 74)
    print("OK")

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "case_id": state["case_id"],
        "random_seed": state["random_seed"],
        "created_at_unix": time.time(),
        "duration_s": time.perf_counter() - started_at,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mujoco": getattr(mujoco, "__version__", "unknown"),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))),
            capture_output=True, text=True, check=False).stdout.strip(),
        "config": state["config"],
        "execution_status": "completed",
        "metrics": {
            "coax_mm": final_coax, "tilt_deg": final_tilt,
            "contacts": contacts, "slip_mm": slip, "pull_mm": pull,
            "jaw_force_N": force, "grasp_tilt_deg": grasp_tilt,
            "grasp_contact_count": state.get("grasp_contact_count", 0),
            "clamp_swing_mm": swing, "stock_ecc_mm": stock,
            "rod_external_force_max_N": state.get("rod_force_max", 0.0),
            "robot_machine_contact_max": state.get(
                "robot_machine_contact_max", 0),
            "shaft_contact_max": state.get("shaft_contact_max", 0),
            "penetration_min_m": float(PEN["min"]),
            "penetration_pair": PEN["pair"],
            "penetration_phase": PEN["label"],
        },
        "hard_pass": hard_pass,
        "quality_pass": quality_pass,
        "overall_pass": ok,
        "hard_failures": hard_failures,
        "quality_failures": quality_failures,
        "phase_stats": state.get("phase_stats", {}),
        **_diagnostic_result_fields(state),
        "ik_solver": state.get("ik_solver", []),
        "ik_residuals_mm": state.get("ik_residuals_mm", {}),
        "stages": state.get("stage_records", []),
        "checkpoints": state.get("checkpoints", []),
    }
    result.update(_case_classification(
        result["execution_status"], evaluation_scope,
        result["hard_pass"], result["quality_pass"],
        result["overall_pass"]))
    write_result_json(result)
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
    ap.add_argument("--passthrough", action="store_true",
                    help="历史基线专用：卡盘外壳不参与碰撞（机器人可穿模）。"
                         "只用于和旧数字对比，新流程一律不要加")
    ap.add_argument("--json-out", metavar="PATH",
                    help="write structured case results to this JSON file")
    ap.add_argument("--initial-dy-mm", type=float, default=0.0)
    ap.add_argument("--initial-dz-mm", type=float, default=0.0)
    ap.add_argument(
        "--initial-settle-steps",
        type=int,
        default=DEFAULT_INITIAL_SETTLE_STEPS,
        help=("physics steps to let the incoming workpiece settle before "
              "robot approach (default: 600)"),
    )
    ap.add_argument(
        "--initial-yaw-deg",
        type=float,
        help="rotate the workpiece about world +Z (positive X axis toward +Y)",
    )
    ap.add_argument(
        "--initial-roll-deg",
        type=float,
        help="deprecated alias for --initial-yaw-deg; it was never axial roll",
    )
    ap.add_argument(
        "--control-mode",
        choices=PLANNING_MODES,
        default="M1",
        help="pose information available to grasp planning (default: M1)",
    )
    ap.add_argument("--m2-error-dx-mm", type=float, default=0.0)
    ap.add_argument("--m2-error-dy-mm", type=float, default=0.0)
    ap.add_argument("--m2-error-dz-mm", type=float, default=0.0)
    ap.add_argument("--m2-error-yaw-deg", type=float, default=0.0)
    ap.add_argument("--random-seed", type=int)
    ap.add_argument("--case-id")
    args = ap.parse_args()
    if (args.initial_yaw_deg is not None
            and args.initial_roll_deg is not None):
        ap.error("choose either --initial-yaw-deg or legacy --initial-roll-deg")
    if args.initial_settle_steps < 0:
        ap.error("--initial-settle-steps must be nonnegative")
    if args.json_out:
        os.environ["FEED_RESULT_JSON"] = args.json_out
    if args.case_id:
        os.environ["FEED_CASE_ID"] = args.case_id
    if args.random_seed is not None:
        os.environ["FEED_RANDOM_SEED"] = str(args.random_seed)
    if args.passthrough:
        print("  ⚠ 穿模模式：卡盘外壳不参与碰撞，仅用于复现历史基线数字")
    raise SystemExit(main(v_jaw=args.v_jaw, lift=args.lift,
                          carry_lock=not args.no_lock,
                          machine_solid=not args.passthrough,
                          initial_offset_mm=(0.0, args.initial_dy_mm,
                                             args.initial_dz_mm),
                          initial_roll_deg=args.initial_roll_deg,
                          initial_yaw_deg=args.initial_yaw_deg,
                          planning_mode=args.control_mode,
                          m2_error_mm=(args.m2_error_dx_mm,
                                       args.m2_error_dy_mm,
                                       args.m2_error_dz_mm),
                          m2_error_yaw_deg=args.m2_error_yaw_deg,
                          random_seed=args.random_seed,
                          initial_settle_steps=args.initial_settle_steps))
