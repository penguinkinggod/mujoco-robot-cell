"""Test grasp retention using the exact pickup path from the feed flow.

The feed flow owns initialization, settling, lateral V-jaw entry, descent,
closure ramp, and the grasp-complete cut point. This diagnostic then removes
the end cradles and measures free-gravity hold plus a short vertical lift.
It never runs the chuck handover and never applies an external workpiece load.
"""
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diag_feed_chuck as feed
import scene_machining as machining
from demo_08_pick_and_load import Arm


HOLD_STEPS = 400
LIFT_MM = 30.0
LIFT_STEPS = 500
POST_LIFT_HOLD_STEPS = 260


def _geom_name(model, geom_id):
    return (mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")


def _jaw_shaft_contacts(model, data):
    count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = _geom_name(model, int(contact.geom1))
        second = _geom_name(model, int(contact.geom2))
        is_pair = (
            ("shaft" in first and "jaw_pad_v" in second)
            or ("shaft" in second and "jaw_pad_v" in first)
        )
        count += int(is_pair)
    return count


def _shaft_contact_pairs(model, data):
    pairs = {}
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(contact.geom1)
        second = int(contact.geom2)
        first_name = _geom_name(model, first)
        second_name = _geom_name(model, second)
        if not (first_name.startswith("shaft_")
                or second_name.startswith("shaft_")):
            continue
        if first_name.startswith("shaft_"):
            partner_geom = second
            shaft_name, partner_name = first_name, second_name
        else:
            partner_geom = first
            shaft_name, partner_name = second_name, first_name
        partner_body = int(model.geom_bodyid[partner_geom])
        partner_body_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, partner_body) or ""
        key = "%s|%s" % (shaft_name, partner_name)
        pair = pairs.setdefault(key, {
            "shaft_geom": shaft_name,
            "partner_geom": partner_name,
            "partner_body": partner_body_name,
            "contacts": 0,
            "minimum_distance_m": float("inf"),
        })
        pair["contacts"] += 1
        pair["minimum_distance_m"] = min(
            pair["minimum_distance_m"], float(contact.dist))
    return pairs


def _accumulate_contact_pairs(summary, frame_pairs):
    for key, pair in frame_pairs.items():
        record = summary.setdefault(key, {
            "shaft_geom": pair["shaft_geom"],
            "partner_geom": pair["partner_geom"],
            "partner_body": pair["partner_body"],
            "frames_with_contact": 0,
            "contact_records": 0,
            "minimum_distance_m": float("inf"),
            "maximum_contacts_in_frame": 0,
        })
        record["frames_with_contact"] += 1
        record["contact_records"] += pair["contacts"]
        record["minimum_distance_m"] = min(
            record["minimum_distance_m"], pair["minimum_distance_m"])
        record["maximum_contacts_in_frame"] = max(
            record["maximum_contacts_in_frame"], pair["contacts"])


def _finalize_contact_pairs(summary):
    return {
        key: {
            **pair,
            "minimum_distance_m": float(pair["minimum_distance_m"]),
        }
        for key, pair in sorted(summary.items())
    }


def _shaft_position(data, shaft_body):
    return data.xpos[shaft_body].copy()


def _shaft_axis(data, shaft_body):
    axis = data.xmat[shaft_body].reshape(3, 3)[:, 2].copy()
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _axis_delta_deg(first, second):
    dot = abs(float(np.dot(first, second)))
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


def _relative_pose(model, data, arm, shaft_body):
    rotation = data.site_xmat[arm.site].reshape(3, 3)
    offset_tool = rotation.T @ (
        _shaft_position(data, shaft_body) - data.site_xpos[arm.site])
    axis_tool = rotation.T @ _shaft_axis(data, shaft_body)
    return offset_tool, axis_tool


def _disable_end_cradles(model):
    names = []
    for geom_id in range(model.ngeom):
        name = _geom_name(model, geom_id)
        if name.startswith("cradle_"):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            names.append(name)
    if len(names) != 4:
        raise RuntimeError(
            "expected four end-cradle geoms, found %d: %s"
            % (len(names), names))
    return sorted(names)


def _move_with_axis_hold(model, data, arm, shaft_body, pad_target,
                         target_axis, axis_local, jaw_act, grip_target,
                         steps):
    site_target = np.asarray(pad_target, dtype=float) - arm.pad_offset()
    q_start = np.asarray(
        [data.qpos[address] for address in arm.qadr], dtype=float)
    q_goal, ik_position_error, ik_axis_error = arm.ik_axis(
        site_target, target_axis, axis_local, q_init=q_start, iters=500)
    pad_start = arm.pad_center().copy()
    shaft_start = _shaft_position(data, shaft_body)
    axis_start = _shaft_axis(data, shaft_body)
    offset_start, axis_tool_start = _relative_pose(
        model, data, arm, shaft_body)
    max_relative_translation_mm = 0.0
    max_relative_axis_change_deg = 0.0
    contact_samples = []
    contact_pairs = {}

    def step_with_target(q_target):
        for index, value in enumerate(q_target):
            data.ctrl[index] = value
        data.ctrl[arm.grip_act] = grip_target
        data.ctrl[jaw_act] = machining.CHUCK_OPEN_CTRL
        mujoco.mj_step(model, data)

    for step_index in range(steps):
        fraction = (step_index + 1) / float(steps)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        q_target = q_start + smooth * (q_goal - q_start)
        step_with_target(q_target)
        offset_now, axis_tool_now = _relative_pose(
            model, data, arm, shaft_body)
        max_relative_translation_mm = max(
            max_relative_translation_mm,
            float(np.linalg.norm(offset_now - offset_start) * 1000.0))
        max_relative_axis_change_deg = max(
            max_relative_axis_change_deg,
            _axis_delta_deg(axis_tool_start, axis_tool_now))
        contact_samples.append(_jaw_shaft_contacts(model, data))
        _accumulate_contact_pairs(
            contact_pairs, _shaft_contact_pairs(model, data))

    for _ in range(POST_LIFT_HOLD_STEPS):
        step_with_target(q_goal)
        offset_now, axis_tool_now = _relative_pose(
            model, data, arm, shaft_body)
        max_relative_translation_mm = max(
            max_relative_translation_mm,
            float(np.linalg.norm(offset_now - offset_start) * 1000.0))
        max_relative_axis_change_deg = max(
            max_relative_axis_change_deg,
            _axis_delta_deg(axis_tool_start, axis_tool_now))
        contact_samples.append(_jaw_shaft_contacts(model, data))
        _accumulate_contact_pairs(
            contact_pairs, _shaft_contact_pairs(model, data))

    pad_end = arm.pad_center().copy()
    shaft_end = _shaft_position(data, shaft_body)
    pad_delta = pad_end - pad_start
    shaft_delta = shaft_end - shaft_start
    pad_delta_sq = float(np.dot(pad_delta, pad_delta))
    follow_ratio = (
        float(np.dot(shaft_delta, pad_delta) / pad_delta_sq)
        if pad_delta_sq > 1e-12 else None)
    axis_end = _shaft_axis(data, shaft_body)
    return {
        "requested_pad_lift_mm": float(LIFT_MM),
        "ik_position_residual_mm": float(ik_position_error * 1000.0),
        "ik_axis_residual_deg": float(math.degrees(ik_axis_error)),
        "actual_pad_displacement_mm": [
            float(value * 1000.0) for value in pad_delta],
        "actual_shaft_displacement_mm": [
            float(value * 1000.0) for value in shaft_delta],
        "shaft_follow_projection_ratio": follow_ratio,
        "shaft_axis_change_deg": _axis_delta_deg(axis_start, axis_end),
        "relative_translation_from_lift_start_mm": (
            max_relative_translation_mm),
        "relative_axis_change_from_lift_start_deg": (
            max_relative_axis_change_deg),
        "jaw_shaft_contact_samples": len(contact_samples),
        "jaw_shaft_contact_samples_with_contact": sum(
            count > 0 for count in contact_samples),
        "jaw_shaft_contact_min": (
            min(contact_samples) if contact_samples else 0),
        "jaw_shaft_contact_end": _jaw_shaft_contacts(model, data),
        "shaft_position_start_m": [float(value) for value in shaft_start],
        "shaft_position_end_m": [float(value) for value in shaft_end],
        "pad_position_start_m": [float(value) for value in pad_start],
        "pad_position_end_m": [float(value) for value in pad_end],
        "shaft_contact_pairs": _finalize_contact_pairs(contact_pairs),
    }


def run():
    # The diagnostic must own its output path; never inherit an external
    # result-file override that might overwrite a user's prior experiment.
    os.environ.pop("FEED_RESULT_JSON", None)
    model = feed.build_model(v_jaw=True, lift=True)
    data = mujoco.MjData(model)
    arm = Arm(model, data)

    grasp = feed.main(
        model=model,
        data=data,
        carry_lock=False,
        v_jaw=True,
        lift=True,
        stop_after_grasp=True,
    )

    shaft_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "shaft")
    jaw_act = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
    if not isinstance(grasp, dict):
        result = {
            "experiment": "matched_main_path_grasp_retention",
            "execution_status": "aborted_before_grasp_complete",
            "feed_return_code": grasp,
            "initial_v_jaw_shaft_contact_records": _jaw_shaft_contacts(
                model, data),
            "interpretation": (
                "The exact main-flow pickup did not reach grasp_complete; "
                "this is a valid grasp-path failure observation."),
        }
        return result

    if grasp["carry_lock"]:
        raise RuntimeError("matched grasp diagnostic must use no carry lock")

    initial_position = _shaft_position(data, shaft_body)
    initial_axis = _shaft_axis(data, shaft_body)
    initial_relative_offset, initial_relative_axis = _relative_pose(
        model, data, arm, shaft_body)
    grasp_arm_targets = np.asarray(data.ctrl[:5], dtype=float).copy()
    grasp_grip_target = float(data.ctrl[arm.grip_act])
    initial_contact_records = _jaw_shaft_contacts(model, data)
    initial_contact_pairs = _shaft_contact_pairs(model, data)
    removed_cradles = _disable_end_cradles(model)
    mujoco.mj_forward(model, data)

    hold_positions = []
    hold_contacts = []
    hold_axis_changes = []
    hold_offset_changes = []
    hold_relative_axis_changes = []
    hold_contact_pairs = {}
    hold_pad_start = arm.pad_center().copy()
    hold_pad_positions = []
    for _ in range(HOLD_STEPS):
        data.ctrl[:5] = grasp_arm_targets
        data.ctrl[arm.grip_act] = grasp_grip_target
        data.ctrl[jaw_act] = machining.CHUCK_OPEN_CTRL
        mujoco.mj_step(model, data)
        hold_positions.append(_shaft_position(data, shaft_body))
        hold_contacts.append(_jaw_shaft_contacts(model, data))
        hold_pad_positions.append(arm.pad_center().copy())
        _accumulate_contact_pairs(
            hold_contact_pairs, _shaft_contact_pairs(model, data))
        hold_axis_changes.append(_axis_delta_deg(
            initial_axis, _shaft_axis(data, shaft_body)))
        offset_now, axis_now = _relative_pose(
            model, data, arm, shaft_body)
        hold_offset_changes.append(float(
            np.linalg.norm(offset_now - initial_relative_offset) * 1000.0))
        hold_relative_axis_changes.append(_axis_delta_deg(
            initial_relative_axis, axis_now))

    position_after_hold = _shaft_position(data, shaft_body)
    axis_after_hold = _shaft_axis(data, shaft_body)
    pad_after_hold = arm.pad_center().copy()
    jaw_contacts_after_hold = _jaw_shaft_contacts(model, data)
    relative_offset_after_hold, relative_axis_after_hold = _relative_pose(
        model, data, arm, shaft_body)
    relative_translation_after_hold_mm = float(
        np.linalg.norm(
            relative_offset_after_hold - initial_relative_offset) * 1000.0)
    relative_axis_change_after_hold_deg = _axis_delta_deg(
        initial_relative_axis, relative_axis_after_hold)
    site_rotation = data.site_xmat[arm.site].reshape(3, 3)
    axis_local = site_rotation.T @ axis_after_hold
    pad_target = arm.pad_center().copy() + np.array(
        [0.0, 0.0, LIFT_MM / 1000.0])
    jaw_act = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
    lift = _move_with_axis_hold(
        model, data, arm, shaft_body, pad_target,
        axis_after_hold, axis_local, jaw_act, grasp_grip_target,
        LIFT_STEPS)
    lift["shaft_center_displacement_after_hold_mm"] = float(
        np.linalg.norm(position_after_hold - initial_position) * 1000.0)
    lift["shaft_axis_change_after_hold_deg"] = _axis_delta_deg(
        initial_axis, axis_after_hold)

    result = {
        "experiment": "matched_main_path_grasp_retention",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "completed_observation",
        "protocol": {
            "source": "diag_feed_chuck.main(stop_after_grasp=True)",
            "v_jaw": True,
            "lifted_supported_incoming_stock": True,
            "initial_settle_steps": feed.DEFAULT_INITIAL_SETTLE_STEPS,
            "carry_lock": False,
            "gripper_target_after_main_grasp": grasp_grip_target,
            "end_cradles_disabled_after_grasp": removed_cradles,
            "hold_steps": HOLD_STEPS,
            "hold_duration_s": (
                HOLD_STEPS * float(model.opt.timestep)),
            "lift_requested_mm": LIFT_MM,
            "lift_steps": LIFT_STEPS,
            "post_lift_hold_steps": POST_LIFT_HOLD_STEPS,
            "workpiece_external_force_N": 0.0,
            "pass_fail_thresholds": None,
        },
        "grasp_cutpoint": grasp,
        "initial_v_jaw_shaft_contact_records": int(
            initial_contact_records),
        "initial_shaft_contact_pairs": initial_contact_pairs,
        "hold_after_support_removal": {
            "samples": len(hold_positions),
            "shaft_center_max_displacement_mm": float(max(
                np.linalg.norm(position - initial_position) * 1000.0
                for position in hold_positions)),
            "shaft_center_displacement_at_end_mm": float(
                np.linalg.norm(position_after_hold - initial_position)
                * 1000.0),
            "pad_center_max_displacement_mm": float(max(
                np.linalg.norm(position - hold_pad_start) * 1000.0
                for position in hold_pad_positions)),
            "pad_center_displacement_at_end_mm": float(
                np.linalg.norm(pad_after_hold - hold_pad_start) * 1000.0),
            "pad_center_displacement_vector_at_end_mm": [
                float(value * 1000.0)
                for value in (pad_after_hold - hold_pad_start)],
            "shaft_axis_max_change_deg": float(max(hold_axis_changes)),
            "relative_to_gripper_max_translation_mm": float(
                max(hold_offset_changes)),
            "relative_to_gripper_translation_at_end_mm": (
                relative_translation_after_hold_mm),
            "relative_to_gripper_max_axis_change_deg": float(max(
                hold_relative_axis_changes)),
            "relative_to_gripper_axis_change_at_end_deg": (
                relative_axis_change_after_hold_deg),
            "shaft_contact_pairs": _finalize_contact_pairs(
                hold_contact_pairs),
            "jaw_shaft_contact_samples_with_contact": int(sum(
                count > 0 for count in hold_contacts)),
            "jaw_shaft_contact_min": int(min(hold_contacts)),
            "jaw_shaft_contact_end": int(
                jaw_contacts_after_hold),
            "shaft_position_start_m": [
                float(value) for value in initial_position],
            "shaft_position_end_m": [
                float(value) for value in position_after_hold],
            "shaft_axis_change_at_end_deg": _axis_delta_deg(
                initial_axis, axis_after_hold),
        },
        "lift": lift,
        "interpretation": (
            "Observations only. This matched-path test has no independent "
            "grasp-retention PASS/FAIL threshold; compare contact persistence, "
            "relative motion, and lift following before choosing a criterion."),
    }
    return result


def main():
    result = run()
    output_dir = ROOT / "_build" / "p5.8.2-grasp-retention"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        "run-%s-%s.json"
        % (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
           uuid.uuid4().hex[:8]))
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("GRASP_HOLD_RESULT_JSON %s" % output_path)


if __name__ == "__main__":
    main()
