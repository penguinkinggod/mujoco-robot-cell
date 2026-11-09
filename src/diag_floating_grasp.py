"""Probe a single 6-D workpiece pose with a gravity-compensated free body.

This is a grasp/contact diagnostic, not the lathe-feed protocol. The shaft
keeps its mass and inertia, has no supporting fixture, and remains free to
move when contacted. Its own gravity is compensated so a pose can be sampled
without the part falling before the robot reaches it.
"""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import scene_full as F
import diag_feed_chuck as feed
from demo_08_pick_and_load import (
    Arm, GRIP_OPEN,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "_build" / "floating-grasp" / "case.json"
NOMINAL_AXIS = np.array([1.0, 0.0, 0.0])
IK_POSITION_LIMIT_M = 0.010
IK_AXIS_LIMIT_RAD = math.radians(10.0)
GRASP_MIN_PAD_CONTACTS = 2
GRASP_RELATIVE_POSITION_LIMIT_M = 0.002
GRASP_RELATIVE_AXIS_LIMIT_RAD = math.radians(5.0)
GRASP_LIFT_DISTANCE_M = 0.030
GRASP_HOLD_STEPS = 150


def pose_quaternion_world_xyz(rpy_deg):
    """Return MuJoCo wxyz quaternion after world-frame XYZ rotations.

    The unrotated shaft axis is world +X. The requested RPY rotation is
    applied in the world frame to that nominal shaft pose.
    """
    base = Rotation.from_quat([0.0, 0.7071068, 0.0, 0.7071068])
    delta = Rotation.from_euler("xyz", rpy_deg, degrees=True)
    xyzw = (delta * base).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def build_floating_scene():
    """Build the robot and unsupported workpiece, with gravity compensation."""
    spec = F.build_scene(
        with_lathe=False,
        with_vise=False,
        with_fixture=False,
        with_shaft_flat=False,
    )
    shaft_spec = next(
        (body for body in spec.bodies if body.name == "shaft"), None)
    if shaft_spec is None:
        raise RuntimeError("floating scene is missing the shaft body")
    # MuJoCo compiles a model-level fast-path flag for gravity compensation.
    # Set this before compile rather than mutating the compiled model array.
    shaft_spec.gravcomp = 1.0
    feed._install_v_jaws(spec)
    for body in spec.bodies:
        if body.name not in F.ROBOT_BODY_NAMES:
            continue
        for geom in body.geoms:
            if geom.group == 3 and "pad" not in (geom.name or ""):
                geom.contype = F.CT_ROBOT_SHELL[0]
                geom.conaffinity = F.CT_ROBOT_SHELL[1]
    model = spec.compile()
    data = mujoco.MjData(model)
    shaft_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    shaft_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
    if shaft_body < 0 or shaft_joint < 0:
        raise RuntimeError("floating scene is missing the shaft free body")
    if not model.flg_gravcomp:
        raise RuntimeError("MuJoCo did not enable gravity compensation")
    return model, data, shaft_body, shaft_joint


def set_workpiece_pose(model, data, shaft_body, shaft_joint,
                       position_m, rpy_deg):
    """Place the free body once; physics owns its pose after this call."""
    position = np.asarray(position_m, dtype=float)
    rpy = np.asarray(rpy_deg, dtype=float)
    if position.shape != (3,) or rpy.shape != (3,):
        raise ValueError("position_m and rpy_deg must each have 3 values")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rpy)):
        raise ValueError("pose values must be finite")

    quat = pose_quaternion_world_xyz(rpy)
    qadr = model.jnt_qposadr[shaft_joint]
    dadr = model.jnt_dofadr[shaft_joint]
    data.qpos[qadr:qadr + 3] = position
    data.qpos[qadr + 3:qadr + 7] = quat
    data.qvel[dadr:dadr + 6] = 0.0
    mujoco.mj_forward(model, data)

    axis = data.xmat[shaft_body].reshape(3, 3)[:, 2].copy()
    half_height = (
        F.SHAFT_L / 2 * float(np.linalg.norm(axis[:2]))
        + F.SHAFT_D / 2 * abs(float(axis[2]))
    )
    if position[2] - half_height <= F.TABLE_TOP_Z + 0.002:
        raise ValueError(
            "pose intersects the table; raise the shaft before testing")
    return axis


def _pose_snapshot(model, data, shaft_body, shaft_joint):
    qadr = model.jnt_qposadr[shaft_joint]
    position = data.qpos[qadr:qadr + 3].copy()
    quat = data.qpos[qadr + 3:qadr + 7].copy()
    axis = data.xmat[shaft_body].reshape(3, 3)[:, 2].copy()
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return {
        "position_m": [float(value) for value in position],
        "quaternion_wxyz": [float(value) for value in quat],
        "shaft_axis_world_unit": [float(value) for value in axis],
    }


def _gripper_relative_pose(model, data, arm, shaft_body):
    """Workpiece centre and axis expressed in the gripper site frame."""
    site_rotation = data.site_xmat[arm.site].reshape(3, 3)
    shaft_position = data.xpos[shaft_body].copy()
    shaft_axis = data.xmat[shaft_body].reshape(3, 3)[:, 2].copy()
    shaft_axis /= max(float(np.linalg.norm(shaft_axis)), 1e-12)
    return {
        "center_local_m": [
            float(x) for x in
            site_rotation.T @ (shaft_position - arm.pad_center(data))],
        "axis_local_unit": [
            float(x) for x in site_rotation.T @ shaft_axis],
    }


def _relative_pose_change(reference, current):
    reference_center = np.asarray(reference["center_local_m"], dtype=float)
    current_center = np.asarray(current["center_local_m"], dtype=float)
    reference_axis = np.asarray(reference["axis_local_unit"], dtype=float)
    current_axis = np.asarray(current["axis_local_unit"], dtype=float)
    axis_dot = float(np.clip(
        abs(np.dot(reference_axis, current_axis)), 0.0, 1.0))
    return {
        "center_drift_m": float(np.linalg.norm(
            current_center - reference_center)),
        "axis_drift_rad": math.acos(axis_dot),
    }


def _contacted_pad_sides(pairs):
    pad_names = {pair[1] for pair in pairs}
    sides = set()
    if any(name.startswith("fixed_jaw_pad_v") for name in pad_names):
        sides.add("fixed")
    if any(name.startswith("moving_jaw_pad_v") for name in pad_names):
        sides.add("moving")
    return sides


def _evaluate_grasp(close_pad_contact_count, close_pad_sides,
                    hold_pad_contact_count, hold_pad_sides,
                    hold_opposing_contact_steps, hold_sample_count,
                    workpiece_lift_m, max_center_drift_m,
                    max_axis_drift_rad, lift_executed):
    """Evaluate grasp retention without treating pre-contact motion as failure."""
    checks = {
        "opposing_jaw_contact": (
            close_pad_contact_count >= GRASP_MIN_PAD_CONTACTS
            and close_pad_sides == {"fixed", "moving"}),
        "lift_executed": bool(lift_executed),
        "contact_retained_during_hold": (
            hold_pad_contact_count >= GRASP_MIN_PAD_CONTACTS
            and hold_pad_sides == {"fixed", "moving"}
            and hold_sample_count > 0
            and hold_opposing_contact_steps == hold_sample_count),
        "minimum_workpiece_lift": (
            workpiece_lift_m >= GRASP_LIFT_DISTANCE_M * 0.8),
        "relative_center_stable": (
            max_center_drift_m <= GRASP_RELATIVE_POSITION_LIMIT_M),
        "relative_axis_stable": (
            max_axis_drift_rad <= GRASP_RELATIVE_AXIS_LIMIT_RAD),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    if not checks["opposing_jaw_contact"]:
        outcome = "no_opposing_jaw_contact"
    elif not checks["lift_executed"]:
        outcome = "lift_not_executed"
    elif not checks["contact_retained_during_hold"]:
        outcome = "contact_lost_during_lift"
    elif not checks["minimum_workpiece_lift"]:
        outcome = "insufficient_lift"
    elif not (checks["relative_center_stable"]
              and checks["relative_axis_stable"]):
        outcome = "workpiece_slipped_relative_to_gripper"
    else:
        outcome = "grasp_hold_pass_gravity_compensated"
    return {
        "checks": checks,
        "failure_reasons": reasons,
        "passed": all(checks.values()),
        "outcome": outcome,
    }


def _shaft_geom_ids(model):
    return {
        geom_id for geom_id in range(model.ngeom)
        if (mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
                "shaft_")
    }


def _workpiece_contact_snapshot(model, data, shaft_ids, pad_ids):
    """Describe all current workpiece contacts with named geometry pairs."""
    contacts = []
    pad_count = 0
    pad_pairs = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.geom1 in shaft_ids:
            shaft_geom, other_geom = contact.geom1, contact.geom2
        elif contact.geom2 in shaft_ids:
            shaft_geom, other_geom = contact.geom2, contact.geom1
        else:
            continue

        shaft_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, shaft_geom)
        other_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
        other_body_id = int(model.geom_bodyid[other_geom])
        other_body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, other_body_id)
        is_pad = other_geom in pad_ids
        contacts.append({
            "workpiece_geom": shaft_name,
            "other_geom": other_name,
            "other_body": other_body_name,
            "is_v_jaw_pad": is_pad,
            "distance_m": float(contact.dist),
            "position_world_m": [
                float(value) for value in contact.pos],
        })
        if is_pad:
            pad_count += 1
            pad_pairs.add((shaft_name, other_name))
    contacts.sort(key=lambda item: (
        item["other_body"] or "", item["other_geom"] or "",
        item["workpiece_geom"] or ""))
    return pad_count, sorted(pad_pairs), contacts


def _shaft_contact_counts(model, data, arm):
    shaft_ids = _shaft_geom_ids(model)
    pad_ids = set(arm.pads)
    count, pairs, _ = _workpiece_contact_snapshot(
        model, data, shaft_ids, pad_ids)
    return count, pairs


def _calibrate_tool_frame(model, data, arm):
    """Calibrate the V-jaw axis and its stable reference-point offset."""
    data.qpos[arm.qadr[4]] = feed.V_WRIST
    data.ctrl[4] = feed.V_WRIST
    data.ctrl[arm.grip_act] = GRIP_OPEN
    mujoco.mj_forward(model, data)
    nominal_center = np.array([0.26, -0.13, 0.15])
    nominal_site = nominal_center - arm.pad_offset()
    q_nominal, _ = arm.ik(
        nominal_site, lock_wrist_roll=feed.V_WRIST)

    calibration = mujoco.MjData(model)
    calibration.qpos[:] = data.qpos
    calibration.ctrl[:] = data.ctrl
    for qadr, value in zip(arm.qadr, q_nominal):
        calibration.qpos[qadr] = value
        calibration.ctrl[qadr] = value
    mujoco.mj_forward(model, calibration)

    rotation = calibration.site_xmat[arm.site].reshape(3, 3)
    axis_local = rotation.T @ NOMINAL_AXIS
    pad_offset_local = rotation.T @ (
        arm.pad_center(calibration) - calibration.site_xpos[arm.site])
    return axis_local, pad_offset_local


def _solve_pad_center(model, data, arm, target_center, target_axis,
                      axis_local, closed_pad_offset_local, q_seed):
    """Solve pad-centre position IK with the calibrated wrist orientation."""
    q = np.asarray(q_seed, dtype=float).copy()
    position_error = float("inf")
    axis_error = 0.0
    trial = mujoco.MjData(model)
    for _ in range(6):
        trial.qpos[:] = data.qpos
        for qadr, value in zip(arm.qadr, q):
            trial.qpos[qadr] = value
        mujoco.mj_forward(model, trial)
        rotation = trial.site_xmat[arm.site].reshape(3, 3)
        site_target = target_center - rotation @ closed_pad_offset_local
        q, position_error = arm.ik(
            site_target, q_init=q, lock_wrist_roll=feed.V_WRIST)
    trial.qpos[:] = data.qpos
    for qadr, value in zip(arm.qadr, q):
        trial.qpos[qadr] = value
    mujoco.mj_forward(model, trial)
    actual_axis = (
        trial.site_xmat[arm.site].reshape(3, 3) @ axis_local)
    actual_axis /= max(float(np.linalg.norm(actual_axis)), 1e-12)
    target_axis = np.asarray(target_axis, dtype=float)
    target_axis /= max(float(np.linalg.norm(target_axis)), 1e-12)
    axis_error = math.acos(float(np.clip(
        np.dot(actual_axis, target_axis), -1.0, 1.0)))
    return q, float(position_error), float(axis_error)


def _move_joints(model, data, arm, q_goal, grip, steps,
                 relative_pose_reference=None, shaft_body=None):
    q_start = np.array([data.qpos[qadr] for qadr in arm.qadr])
    shaft_ids = _shaft_geom_ids(model)
    pad_ids = set(arm.pads)
    pad_contact_max = 0
    pad_contact_min = None
    opposing_contact_steps = 0
    first_workpiece_contact = None
    first_pad_contact = None
    max_relative_center_drift_m = 0.0
    max_relative_axis_drift_rad = 0.0
    for index in range(steps):
        fraction = (index + 1) / float(steps)
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        for qadr, value in zip(
                arm.qadr, q_start + blend * (q_goal - q_start)):
            data.ctrl[qadr] = value
        data.ctrl[arm.grip_act] = grip
        mujoco.mj_step(model, data)
        pad_count, pad_pairs, contacts = _workpiece_contact_snapshot(
            model, data, shaft_ids, pad_ids)
        pad_contact_max = max(pad_contact_max, pad_count)
        pad_contact_min = (
            pad_count if pad_contact_min is None
            else min(pad_contact_min, pad_count))
        if _contacted_pad_sides(pad_pairs) == {"fixed", "moving"}:
            opposing_contact_steps += 1
        if contacts and first_workpiece_contact is None:
            first_workpiece_contact = {
                "step_in_phase": index + 1,
                "sim_time_s": float(data.time),
                "contacts": contacts,
            }
        pad_contacts = [
            contact for contact in contacts if contact["is_v_jaw_pad"]]
        if pad_contacts and first_pad_contact is None:
            first_pad_contact = {
                "step_in_phase": index + 1,
                "sim_time_s": float(data.time),
                "contacts": pad_contacts,
            }
        if relative_pose_reference is not None:
            relative_pose = _gripper_relative_pose(
                model, data, arm, shaft_body)
            change = _relative_pose_change(
                relative_pose_reference, relative_pose)
            max_relative_center_drift_m = max(
                max_relative_center_drift_m, change["center_drift_m"])
            max_relative_axis_drift_rad = max(
                max_relative_axis_drift_rad, change["axis_drift_rad"])
    return {
        "pad_contact_max": pad_contact_max,
        "pad_contact_min": (
            0 if pad_contact_min is None else pad_contact_min),
        "opposing_contact_steps": opposing_contact_steps,
        "sample_count": steps,
        "first_workpiece_contact": first_workpiece_contact,
        "first_pad_contact": first_pad_contact,
        "max_relative_center_drift_m": max_relative_center_drift_m,
        "max_relative_axis_drift_rad": max_relative_axis_drift_rad,
    }


def _v_jaw_entry_axis(model, data, shaft_axis):
    fixed_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM,
            "fixed_jaw_pad_v%d" % index)
        for index in (1, 2)
    ]
    moving_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM,
            "moving_jaw_pad_v%d" % index)
        for index in (1, 2)
    ]
    if min(fixed_ids + moving_ids) < 0:
        raise RuntimeError("floating scene is missing V-jaw geometry")
    entry = (np.mean(data.geom_xpos[moving_ids], axis=0)
             - np.mean(data.geom_xpos[fixed_ids], axis=0))
    entry -= float(np.dot(entry, shaft_axis)) * shaft_axis
    entry /= max(float(np.linalg.norm(entry)), 1e-12)
    return entry


def run_case(position_mm=(260.0, -130.0, 150.0),
             rpy_deg=(0.0, 0.0, 0.0),
             approach_mm=20.0, move_steps=600):
    if not math.isfinite(approach_mm) or approach_mm <= 0:
        raise ValueError("approach_mm must be positive and finite")
    if type(move_steps) is not int or move_steps <= 0:
        raise ValueError("move_steps must be a positive integer")

    model, data, shaft_body, shaft_joint = build_floating_scene()
    mujoco.mj_resetData(model, data)
    position_m = np.asarray(position_mm, dtype=float) / 1000.0
    shaft_axis = set_workpiece_pose(
        model, data, shaft_body, shaft_joint, position_m, rpy_deg)
    arm = Arm(model, data)
    for qadr in arm.qadr:
        data.ctrl[qadr] = data.qpos[qadr]
    data.ctrl[arm.grip_act] = GRIP_OPEN
    axis_local, closed_pad_offset_local = _calibrate_tool_frame(
        model, data, arm)
    mujoco.mj_forward(model, data)
    if any(
            data.contact[index].geom1 in {
                geom_id for geom_id in range(model.ngeom)
                if (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
                .startswith("shaft_")
            }
            or data.contact[index].geom2 in {
                geom_id for geom_id in range(model.ngeom)
                if (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
                .startswith("shaft_")
            }
            for index in range(data.ncon)):
        raise ValueError(
            "starting pose already touches another object; choose a clear pose")

    initial_pose = _pose_snapshot(model, data, shaft_body, shaft_joint)
    q_start = np.array([data.qpos[qadr] for qadr in arm.qadr])
    grasp_center = position_m + np.array([0.0, 0.0, 0.002])
    top_center = position_m + np.array(
        [0.0, 0.0, F.SHAFT_D / 2 + 0.075])
    top_q, top_pos_error, top_axis_error = _solve_pad_center(
        model, data, arm, top_center, shaft_axis,
        axis_local, closed_pad_offset_local, q_start)
    result = {
        "schema_version": 1,
        "experiment": "floating_grasp_contact_diagnostic",
        "scope": (
            "single-pose contact and lift probe; not the lathe-feed task"),
        "physics": {
            "workpiece_mass_kg": float(model.body_mass[shaft_body]),
            "workpiece_gravcomp": float(model.body_gravcomp[shaft_body]),
            "gravity_m_s2": [float(x) for x in model.opt.gravity],
            "fixture_present": False,
            "workpiece_is_free_body": True,
        },
        "acceptance_criteria": {
            "preclose_workpiece_motion_is_failure": False,
            "minimum_pad_contacts": GRASP_MIN_PAD_CONTACTS,
            "required_jaw_sides": ["fixed", "moving"],
            "target_lift_mm": GRASP_LIFT_DISTANCE_M * 1000.0,
            "minimum_lift_mm": GRASP_LIFT_DISTANCE_M * 1000.0 * 0.8,
            "maximum_relative_center_drift_mm": (
                GRASP_RELATIVE_POSITION_LIMIT_M * 1000.0),
            "maximum_relative_axis_drift_deg": math.degrees(
                GRASP_RELATIVE_AXIS_LIMIT_RAD),
            "hold_steps": GRASP_HOLD_STEPS,
            "required_opposing_contact_fraction_during_hold": 1.0,
            "hold_duration_s": (
                GRASP_HOLD_STEPS * float(model.opt.timestep)),
            "threshold_status": "provisional_project_test_limits",
        },
        "inputs": {
            "position_mm": [float(x) for x in position_mm],
            "world_rpy_xyz_deg": [float(x) for x in rpy_deg],
            "approach_mm": float(approach_mm),
            "top_approach_mm": (F.SHAFT_D / 2 + 0.075) * 1000.0,
            "move_steps": move_steps,
            "approach_direction_world": None,
            "jaw_entry_axis_world": None,
            "gripper": "existing calibrated 90-degree V jaws",
        },
        "initial_workpiece_pose": initial_pose,
        "ik": {
            "top_approach_position_residual_mm": top_pos_error * 1000.0,
            "top_approach_axis_residual_deg": math.degrees(
                top_axis_error),
            "approach_position_residual_mm": None,
            "approach_axis_residual_deg": None,
            "grasp_position_residual_mm": None,
            "grasp_axis_residual_deg": None,
            "provisional_gate_position_mm": IK_POSITION_LIMIT_M * 1000.0,
            "provisional_gate_axis_deg": math.degrees(IK_AXIS_LIMIT_RAD),
        },
        "stages": [],
    }

    def record_stage(name, movement=None):
        pose = _pose_snapshot(model, data, shaft_body, shaft_joint)
        contact_count, pairs = _shaft_contact_counts(model, data, arm)
        stage = {
            "stage": name,
            "sim_time_s": float(data.time),
            "workpiece_pose": pose,
            "pad_workpiece_contact_count": contact_count,
            "pad_workpiece_contact_pairs": pairs,
        }
        if movement is not None:
            stage["first_workpiece_contact_in_stage"] = (
                movement["first_workpiece_contact"])
            stage["first_v_jaw_contact_in_stage"] = (
                movement["first_pad_contact"])
        result["stages"].append(stage)

    first_workpiece_contact = None
    first_pad_contact = None

    def record_movement_contact(movement):
        nonlocal first_workpiece_contact, first_pad_contact
        if (first_workpiece_contact is None
                and movement["first_workpiece_contact"] is not None):
            first_workpiece_contact = movement["first_workpiece_contact"]
        if (first_pad_contact is None
                and movement["first_pad_contact"] is not None):
            first_pad_contact = movement["first_pad_contact"]

    if top_pos_error > IK_POSITION_LIMIT_M:
        result["execution_status"] = "ik_gate_rejected"
        result["interpretation"] = (
            "The top-approach position did not pass the provisional IK "
            "gate; no robot motion was run.")
        return result

    top_movement = _move_joints(
        model, data, arm, top_q, GRIP_OPEN, move_steps)
    record_movement_contact(top_movement)
    record_stage("top_approach", top_movement)

    # The calibrated opening direction changes slightly with arm posture.
    # Measure it at the actual top-approach pose, matching the proven pickup.
    entry_axis = _v_jaw_entry_axis(model, data, shaft_axis)
    approach = -entry_axis
    approach_center = position_m + approach * (approach_mm / 1000.0)
    approach_q, approach_pos_error, approach_axis_error = _solve_pad_center(
        model, data, arm, approach_center, shaft_axis,
        axis_local, closed_pad_offset_local,
        np.array([data.qpos[qadr] for qadr in arm.qadr]))
    result["inputs"]["approach_direction_world"] = [
        float(x) for x in approach]
    result["inputs"]["jaw_entry_axis_world"] = [
        float(x) for x in entry_axis]
    result["ik"]["approach_position_residual_mm"] = (
        approach_pos_error * 1000.0)
    result["ik"]["approach_axis_residual_deg"] = math.degrees(
        approach_axis_error)
    if (approach_pos_error > IK_POSITION_LIMIT_M
            or approach_axis_error > IK_AXIS_LIMIT_RAD):
        result["execution_status"] = "ik_gate_rejected"
        result["interpretation"] = (
            "The measured side-entry target did not pass the provisional "
            "IK gate; no side-entry motion was run.")
        return result

    approach_movement = _move_joints(
        model, data, arm, approach_q, GRIP_OPEN, move_steps)
    record_movement_contact(approach_movement)
    record_stage("side_entry", approach_movement)

    grasp_q, grasp_pos_error, grasp_axis_error = _solve_pad_center(
        model, data, arm, grasp_center, shaft_axis,
        axis_local, closed_pad_offset_local,
        np.array([data.qpos[qadr] for qadr in arm.qadr]))
    result["ik"]["grasp_position_residual_mm"] = (
        grasp_pos_error * 1000.0)
    result["ik"]["grasp_axis_residual_deg"] = math.degrees(
        grasp_axis_error)
    if (grasp_pos_error > IK_POSITION_LIMIT_M
            or grasp_axis_error > IK_AXIS_LIMIT_RAD):
        result["execution_status"] = "ik_gate_rejected"
        result["interpretation"] = (
            "The descent target did not pass the provisional IK gate; "
            "no descent motion was run.")
        return result

    descend_movement = _move_joints(
        model, data, arm, grasp_q, GRIP_OPEN, move_steps)
    record_movement_contact(descend_movement)
    record_stage("descend", descend_movement)
    before_close = _pose_snapshot(model, data, shaft_body, shaft_joint)
    close_movement = _move_joints(
        model, data, arm, grasp_q, feed.V_GRIP_CLOSE, move_steps)
    record_movement_contact(close_movement)
    record_stage("close", close_movement)
    close_pose = _pose_snapshot(model, data, shaft_body, shaft_joint)
    close_position = np.asarray(close_pose["position_m"])
    initial_axis = np.asarray(initial_pose["shaft_axis_world_unit"])
    before_close_axis = np.asarray(
        before_close["shaft_axis_world_unit"])
    axis_dot = float(np.clip(
        abs(np.dot(initial_axis, before_close_axis)), -1.0, 1.0))
    axis_change_before_close_deg = math.degrees(math.acos(axis_dot))
    close_contacts, close_pairs = _shaft_contact_counts(model, data, arm)
    close_pad_sides = _contacted_pad_sides(close_pairs)
    close_relative_pose = _gripper_relative_pose(
        model, data, arm, shaft_body)
    any_pad_contact = max(
        approach_movement["pad_contact_max"],
        descend_movement["pad_contact_max"],
        close_movement["pad_contact_max"])

    lift_position_error = None
    lift_axis_error = None
    lift_executed = False
    lift_movement = None
    hold_movement = None
    hold_contacts = 0
    hold_pairs = []
    hold_opposing_contact_steps = 0
    hold_sample_count = 0
    max_relative_center_drift_m = 0.0
    max_relative_axis_drift_rad = 0.0
    gripper_lift_projection_m = 0.0
    if close_pad_sides == {"fixed", "moving"}:
        lift_direction = np.array([0.0, 0.0, 1.0])
        close_pad_center = arm.pad_center(data).copy()
        lift_center = close_pad_center + lift_direction * GRASP_LIFT_DISTANCE_M
        close_axis = np.asarray(
            close_pose["shaft_axis_world_unit"], dtype=float)
        lift_q, lift_pos_error, lift_axis_error = _solve_pad_center(
            model, data, arm, lift_center, close_axis,
            axis_local, closed_pad_offset_local,
            np.array([data.qpos[qadr] for qadr in arm.qadr]))
        lift_position_error = lift_pos_error
        if (lift_pos_error <= IK_POSITION_LIMIT_M):
            lift_executed = True
            lift_movement = _move_joints(
                model, data, arm, lift_q, feed.V_GRIP_CLOSE, move_steps,
                relative_pose_reference=close_relative_pose,
                shaft_body=shaft_body)
            record_movement_contact(lift_movement)
            record_stage("lift_probe", lift_movement)
            lift_pad_center = arm.pad_center(data).copy()
            gripper_lift_projection_m = float(
                (lift_pad_center - close_pad_center)[2])
            result["ik"]["lift_position_residual_mm"] = (
                lift_pos_error * 1000.0)
            result["ik"]["lift_axis_residual_deg"] = math.degrees(
                lift_axis_error)
            hold_q = np.array([data.qpos[qadr] for qadr in arm.qadr])
            hold_movement = _move_joints(
                model, data, arm, hold_q, feed.V_GRIP_CLOSE,
                GRASP_HOLD_STEPS,
                relative_pose_reference=close_relative_pose,
                shaft_body=shaft_body)
            record_movement_contact(hold_movement)
            record_stage("hold", hold_movement)
            hold_contacts, hold_pairs = _shaft_contact_counts(
                model, data, arm)
            hold_opposing_contact_steps = (
                hold_movement["opposing_contact_steps"])
            hold_sample_count = hold_movement["sample_count"]
            max_relative_center_drift_m = max(
                lift_movement["max_relative_center_drift_m"],
                hold_movement["max_relative_center_drift_m"])
            max_relative_axis_drift_rad = max(
                lift_movement["max_relative_axis_drift_rad"],
                hold_movement["max_relative_axis_drift_rad"])

    final_pose = _pose_snapshot(model, data, shaft_body, shaft_joint)
    final_position = np.asarray(final_pose["position_m"])
    lift_delta = final_position - close_position
    lift_projection_mm = float(lift_delta[2] * 1000.0)
    final_contacts, _ = _shaft_contact_counts(model, data, arm)
    final_pad_sides = _contacted_pad_sides(
        _shaft_contact_counts(model, data, arm)[1])
    grasp_evaluation = _evaluate_grasp(
        close_contacts, close_pad_sides,
        hold_contacts, _contacted_pad_sides(hold_pairs),
        hold_opposing_contact_steps, hold_sample_count,
        lift_projection_mm / 1000.0,
        max_relative_center_drift_m, max_relative_axis_drift_rad,
        lift_executed)
    result["execution_status"] = "completed"
    before_close_position = np.asarray(before_close["position_m"])
    result["observations"] = {
        "first_workpiece_contact": first_workpiece_contact,
        "first_v_jaw_contact": first_pad_contact,
        "pad_contact_after_close": close_contacts,
        "pad_sides_after_close": sorted(close_pad_sides),
        "pad_contact_after_hold": hold_contacts,
        "pad_sides_after_hold": sorted(_contacted_pad_sides(hold_pairs)),
        "pad_sides_at_end": sorted(final_pad_sides),
        "pad_contact_max_during_top_approach": (
            top_movement["pad_contact_max"]),
        "pad_contact_max_during_approach": (
            approach_movement["pad_contact_max"]),
        "pad_contact_max_during_descend": (
            descend_movement["pad_contact_max"]),
        "pad_contact_max_during_close": (
            close_movement["pad_contact_max"]),
        "pad_contact_max_during_lift": (
            0 if lift_movement is None
            else lift_movement["pad_contact_max"]),
        "pad_contact_max_during_hold": (
            0 if hold_movement is None
            else hold_movement["pad_contact_max"]),
        "pad_contact_min_during_hold": (
            0 if hold_movement is None
            else hold_movement["pad_contact_min"]),
        "opposing_pad_contact_steps_during_hold": (
            hold_opposing_contact_steps),
        "hold_sample_count": hold_sample_count,
        "opposing_pad_contact_fraction_during_hold": (
            0.0 if hold_sample_count == 0
            else hold_opposing_contact_steps / hold_sample_count),
        "pad_contact_at_end": final_contacts,
        "workpiece_displacement_before_close_mm": float(
            np.linalg.norm(before_close_position - position_m) * 1000.0),
        "workpiece_axis_change_before_close_deg": (
            axis_change_before_close_deg),
        "workpiece_displacement_during_close_mm": float(
            np.linalg.norm(close_position - before_close_position) * 1000.0),
        "workpiece_lift_during_probe_mm": lift_projection_mm,
        "gripper_lift_during_probe_mm": (
            gripper_lift_projection_m * 1000.0),
        "maximum_workpiece_center_drift_relative_to_gripper_during_lift_hold_mm": (
            max_relative_center_drift_m * 1000.0),
        "maximum_workpiece_axis_drift_relative_to_gripper_during_lift_hold_deg": (
            math.degrees(max_relative_axis_drift_rad)),
        "pad_contact_detected": any_pad_contact > 0,
        "lift_probe_executed": lift_executed,
        "lift_ik_position_residual_mm": (
            None if lift_position_error is None
            else lift_position_error * 1000.0),
        "lift_ik_axis_residual_deg": (
            None if lift_axis_error is None
            else math.degrees(lift_axis_error)),
        "grasp_passed": grasp_evaluation["passed"],
        "grasp_checks": grasp_evaluation["checks"],
        "grasp_failure_reasons": grasp_evaluation["failure_reasons"],
        "grasp_outcome": grasp_evaluation["outcome"],
        "transfer_readiness": "not_evaluated_no_chuck_in_scene",
        "preclose_motion_interpretation": (
            "recorded_but_not_a_grasp_failure_by_itself"),
        "interpretation": (
            "Grasp hold is evaluated in a gravity-compensated diagnostic "
            "scene. Passing does not prove a gravity-loaded hold or "
            "readiness for chuck transfer.")
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position-mm", nargs=3, type=float,
        default=(260.0, -130.0, 150.0), metavar=("X", "Y", "Z"),
        help="workpiece centre in world mm (default: 260 -130 150)")
    parser.add_argument(
        "--rpy-deg", nargs=3, type=float,
        default=(0.0, 0.0, 0.0), metavar=("ROLL", "PITCH", "YAW"),
        help="world-frame XYZ rotation from the nominal +X shaft pose")
    parser.add_argument(
        "--approach-mm", type=float, default=20.0,
        help="distance to the V-jaw entry point (default: 20 mm)")
    parser.add_argument(
        "--move-steps", type=int, default=600,
        help="simulation steps for each Cartesian phase")
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_OUTPUT,
        help="result JSON path")
    args = parser.parse_args(argv)

    result = run_case(
        position_mm=args.position_mm,
        rpy_deg=args.rpy_deg,
        approach_mm=args.approach_mm,
        move_steps=args.move_steps,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("status=%s" % result["execution_status"])
    print("result=%s" % args.json_out)
    if "observations" in result:
        print(json.dumps(result["observations"], ensure_ascii=False))
    else:
        print(json.dumps(result["ik"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
