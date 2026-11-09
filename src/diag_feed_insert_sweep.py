"""Run the complete robot-to-chuck feed check at several insertion depths.

This holds the current gripper, workpiece, approach and handoff settings fixed.
Only FEED_INSERT_MM changes. Results are written to _build/feed_insert_sweep.csv.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEED_SCRIPT = ROOT / "src" / "diag_feed_chuck.py"
DEFAULT_OUTPUT = ROOT / "_build" / "feed_insert_sweep.csv"
DEFAULT_INSERTS_MM = (5, 6, 7, 8, 9, 10, 15, 20, 25, 30)
EXPECTED_SCHEMA_VERSION = 10
EXPECTED_INITIAL_SETTLE_STEPS = 600
EXPECTED_INITIAL_STABILITY_WINDOW_STEPS = 100
CASE_TIMEOUT_SECONDS = 600
COMPLETED_METRICS = (
    "coax_mm",
    "tilt_deg",
    "contacts",
    "slip_mm",
    "pull_mm",
    "jaw_force_N",
    "grasp_tilt_deg",
    "clamp_swing_mm",
    "stock_ecc_mm",
    "rod_external_force_max_N",
    "robot_machine_contact_max",
)


def _new_row(insert_mm, json_path, case_id, dy_mm, dz_mm, yaw_deg,
             random_seed, planning_mode="M1", sample_id="",
             m2_error_dy_mm=0.0, m2_error_dz_mm=0.0,
             m2_error_yaw_deg=0.0):
    return {
        "case_id": case_id,
        "sample_id": sample_id,
        "random_seed": "" if random_seed is None else random_seed,
        "insert_mm": insert_mm,
        "dy_mm": dy_mm,
        "dz_mm": dz_mm,
        "yaw_deg": yaw_deg,
        "planning_mode": planning_mode,
        "m2_error_dy_mm": m2_error_dy_mm,
        "m2_error_dz_mm": m2_error_dz_mm,
        "m2_error_yaw_deg": m2_error_yaw_deg,
        "status": "ERROR",
        "execution_status": "",
        "evaluation_scope": "",
        "trial_data_valid": "",
        "include_in_primary_trial_stats": "",
        "physical_task_success": "",
        "outcome_class": "runtime_error",
        "diagnostic_mode": "",
        "abort_stage": "",
        "hard_pass": "",
        "quality_pass": "",
        "hard_failures": "",
        "quality_failures": "",
        "exit_code": "",
        "schema_version": "",
        "coax_mm": "",
        "tilt_deg": "",
        "contacts": "",
        "grasp_contact_count": "",
        "slip_mm": "",
        "pull_mm": "",
        "jaw_force_N": "",
        "grasp_tilt_deg": "",
        "carry": "",
        "physics_after_clamp": "",
        "clamp_swing_mm": "",
        "stock_ecc_mm": "",
        "rod_external_force_max_N": "",
        "robot_machine_contact_max": "",
        "feed_residual_mm": "",
        "json_path": str(json_path),
        "error": "",
    }


def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _output_tail(output):
    return _text(output).strip()[-1200:]


def _set_error(row, execution_status, message, output=""):
    row.update({
        "status": "ERROR",
        "execution_status": execution_status,
        "trial_data_valid": False,
        "include_in_primary_trial_stats": False,
        "physical_task_success": False,
        "outcome_class": "runtime_error",
    })
    details = [part for part in (message, _output_tail(output)) if part]
    row["error"] = "\n".join(details)
    return row


def _is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _result_error(result, case_id, expected_mode=None,
                  expected_m2_error=None):
    """Return why a case JSON is unusable, or None for a supported schema."""
    if not isinstance(result, dict):
        return "result JSON root must be an object"
    schema_version = result.get("schema_version")
    if (type(schema_version) is not int
            or schema_version != EXPECTED_SCHEMA_VERSION):
        return (
            "unsupported schema_version: %r (expected %d)"
            % (schema_version, EXPECTED_SCHEMA_VERSION)
        )
    if result.get("case_id") != case_id:
        return "case_id mismatch: expected %s, got %r" % (
            case_id, result.get("case_id"))

    status = result.get("execution_status")
    if status not in ("completed", "aborted", "error"):
        return "invalid execution_status: %r" % status

    required = (
        "config",
        "control_mode",
        "inputs",
        "initial_stability",
        "evaluation_scope",
        "trial_data_valid",
        "include_in_primary_trial_stats",
        "physical_task_success",
        "outcome_class",
        "metrics",
        "hard_pass",
        "quality_pass",
        "overall_pass",
        "hard_failures",
        "quality_failures",
    )
    missing = [key for key in required if key not in result]
    if missing:
        return "missing required fields: %s" % ", ".join(missing)

    if not isinstance(result["config"], dict):
        return "config must be an object"
    if not isinstance(result["control_mode"], dict):
        return "control_mode must be an object"
    if not isinstance(result["inputs"], dict):
        return "inputs must be an object"
    stability = result["initial_stability"]
    if status != "error":
        if not isinstance(stability, dict):
            return "completed/aborted result must include initial_stability"
        if (stability.get("status") != "evaluated"
                or stability.get("sample_count")
                != EXPECTED_INITIAL_STABILITY_WINDOW_STEPS):
            return "initial_stability window was not fully evaluated"
        if stability.get("stability_pass") is not True:
            return "initial_stability did not pass"
    initial_settle_steps = result["inputs"].get("initial_settle_steps")
    if initial_settle_steps != EXPECTED_INITIAL_SETTLE_STEPS:
        return (
            "initial_settle_steps mismatch: expected %d, got %r"
            % (EXPECTED_INITIAL_SETTLE_STEPS, initial_settle_steps))
    mode = result["control_mode"].get("planning_mode")
    if mode not in ("M0", "M1", "M2"):
        return "control_mode.planning_mode must be M0, M1, or M2"
    if (result["control_mode"].get("planning_pose_scope")
            != "pre_grasp_target_only"):
        return "unsupported planning pose scope"
    if (result["control_mode"].get("carry_feedback_policy")
            != "planned_pose_and_arm_site_transform_only"):
        return "unsupported carry feedback policy"
    if expected_mode is not None and mode != expected_mode:
        return "planning mode mismatch: expected %s, got %s" % (
            expected_mode, mode)
    target = result["inputs"].get("grasp_target_pose")
    if status != "error" and not isinstance(target, dict):
        return "completed/aborted result must include grasp_target_pose"
    if target is not None:
        if not isinstance(target, dict) or target.get("mode") != mode:
            return "grasp_target_pose mode does not match control_mode"
        if (target.get("carry_offset_source")
                != "planned_grasp_target_pose"):
            return "unsupported grasp-time carry offset source"
        if expected_m2_error is not None:
            actual_error = target.get("m2_error_used_mm")
            actual_yaw = target.get("m2_error_used_yaw_deg")
            if (not isinstance(actual_error, list)
                    or len(actual_error) != 3
                    or any(not _is_finite_number(value)
                           for value in actual_error)
                    or not _is_finite_number(actual_yaw)):
                return "grasp_target_pose has invalid M2 error audit fields"
            expected_position_error, expected_yaw_error = expected_m2_error
            if (any(abs(float(actual_error[index]) - expected_position_error[index])
                    > 1e-9 for index in range(3))
                    or abs(float(actual_yaw) - expected_yaw_error) > 1e-9):
                return "grasp_target_pose M2 errors differ from requested values"
    if not isinstance(result["evaluation_scope"], str):
        return "evaluation_scope must be a string"
    for key in (
        "trial_data_valid",
        "include_in_primary_trial_stats",
        "physical_task_success",
        "hard_pass",
        "overall_pass",
    ):
        if not isinstance(result[key], bool):
            return "%s must be a boolean" % key
    if result["quality_pass"] is not None and not isinstance(
            result["quality_pass"], bool):
        return "quality_pass must be a boolean or null"
    if not isinstance(result["outcome_class"], str):
        return "outcome_class must be a string"
    for key in ("hard_failures", "quality_failures"):
        if (not isinstance(result[key], list)
                or any(not isinstance(item, str) for item in result[key])):
            return "%s must be a list of strings" % key
    if not isinstance(result["metrics"], dict):
        return "metrics must be an object"

    if status == "completed":
        if not isinstance(result["quality_pass"], bool):
            return "completed result quality_pass must be a boolean"
        if result["overall_pass"] != (
                result["hard_pass"] and result["quality_pass"]):
            return "overall_pass disagrees with hard_pass/quality_pass"
        metrics = result["metrics"]
        missing_metrics = [
            key for key in COMPLETED_METRICS if key not in metrics
        ]
        if missing_metrics:
            return "missing metrics: %s" % ", ".join(missing_metrics)
        invalid_metrics = [
            key for key in COMPLETED_METRICS
            if not _is_finite_number(metrics[key])
        ]
        if invalid_metrics:
            return "non-numeric or non-finite metrics: %s" % ", ".join(
                invalid_metrics)
        residuals = result.get("ik_residuals_mm")
        if not isinstance(residuals, dict):
            return "ik_residuals_mm must be an object"
        if not _is_finite_number(residuals.get("feed")):
            return "ik_residuals_mm.feed must be a finite number"
    elif status == "error":
        if not isinstance(result.get("abort_reason"), str):
            return "error result must include abort_reason"
    elif status == "aborted":
        if (not isinstance(result.get("abort_stage"), str)
                or not isinstance(result.get("abort_reason"), str)):
            return "aborted result must include abort_stage and abort_reason"

    return None


def _copy_classification(row, result):
    row.update({
        "execution_status": result["execution_status"],
        "evaluation_scope": result["evaluation_scope"],
        "trial_data_valid": result["trial_data_valid"],
        "include_in_primary_trial_stats": result[
            "include_in_primary_trial_stats"],
        "physical_task_success": result["physical_task_success"],
        "outcome_class": result["outcome_class"],
        "diagnostic_mode": result["config"].get("diagnostic_mode", ""),
        "planning_mode": result["control_mode"].get("planning_mode", ""),
        "abort_stage": result.get("abort_stage", ""),
        "hard_pass": result["hard_pass"],
        "quality_pass": result["quality_pass"],
        "hard_failures": ";".join(result["hard_failures"]),
        "quality_failures": ";".join(result["quality_failures"]),
    })


def run_case(insert_mm, json_path, case_id=None, dy_mm=0.0, dz_mm=0.0,
             yaw_deg=0.0, random_seed=None, planning_mode="M1",
             sample_id="", m2_error_dy_mm=0.0, m2_error_dz_mm=0.0,
             m2_error_yaw_deg=0.0):
    case_id = case_id or uuid.uuid4().hex[:12]
    json_path = Path(json_path)
    row = _new_row(
        insert_mm, json_path, case_id, dy_mm, dz_mm, yaw_deg, random_seed,
        planning_mode, sample_id, m2_error_dy_mm, m2_error_dz_mm,
        m2_error_yaw_deg)
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.unlink(missing_ok=True)
    except OSError as error:
        return _set_error(row, "result_path_error", str(error))

    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("FEED_"):
            env.pop(key)
    env["FEED_INSERT_MM"] = format(insert_mm, "g")
    env["FEED_CASE_ID"] = case_id
    env["FEED_RESULT_JSON"] = str(json_path.resolve())

    command = [
        sys.executable, str(FEED_SCRIPT), "--v-jaw", "--lift", "--no-lock",
        "--initial-dy-mm", str(dy_mm), "--initial-dz-mm", str(dz_mm),
        "--initial-yaw-deg", str(yaw_deg), "--case-id", case_id,
        "--control-mode", planning_mode,
        "--m2-error-dy-mm", str(m2_error_dy_mm),
        "--m2-error-dz-mm", str(m2_error_dz_mm),
        "--m2-error-yaw-deg", str(m2_error_yaw_deg),
    ]
    if random_seed is not None:
        command.extend(["--random-seed", str(random_seed)])

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=CASE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        output = _text(error.stdout) + "\n" + _text(error.stderr)
        row["exit_code"] = -1
        return _set_error(
            row, "timeout",
            "child process timed out after %d seconds"
            % CASE_TIMEOUT_SECONDS, output)
    except (OSError, subprocess.SubprocessError) as error:
        return _set_error(row, "launch_error", str(error))

    output = completed.stdout + completed.stderr
    row["exit_code"] = completed.returncode
    try:
        with json_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
    except FileNotFoundError:
        return _set_error(row, "missing_result", "result JSON was not created",
                          output)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return _set_error(row, "invalid_result", str(error), output)

    expected_m2_error = (
        (0.0, m2_error_dy_mm, m2_error_dz_mm),
        m2_error_yaw_deg,
    )
    invalid_reason = _result_error(
        result, case_id, expected_mode=planning_mode,
        expected_m2_error=expected_m2_error)
    if isinstance(result, dict):
        row["schema_version"] = result.get("schema_version", "")
    if invalid_reason:
        schema_version = (
            result.get("schema_version")
            if isinstance(result, dict) else None
        )
        status = (
            "unsupported_schema"
            if (type(schema_version) is not int
                    or schema_version != EXPECTED_SCHEMA_VERSION)
            else "invalid_result"
        )
        return _set_error(row, status, invalid_reason, output)

    execution_status = result["execution_status"]
    if execution_status == "error":
        _copy_classification(row, result)
        return _set_error(
            row, "error",
            result.get("abort_reason", "simulation_error"), output)

    expected_exit_code = (
        0 if execution_status == "completed" and result["overall_pass"]
        else 1
    )
    if completed.returncode != expected_exit_code:
        _copy_classification(row, result)
        return _set_error(
            row, "exit_code_mismatch",
            "exit code %s conflicts with %s result (expected %s)"
            % (completed.returncode, execution_status, expected_exit_code),
            output)

    _copy_classification(row, result)
    if execution_status == "aborted":
        row.update({
            "status": "ABORT",
            "error": result["abort_reason"],
        })
        return row

    metrics = result["metrics"]
    row.update({
        "status": "PASS" if result["overall_pass"] else "FAIL",
        "coax_mm": metrics["coax_mm"],
        "tilt_deg": metrics["tilt_deg"],
        "contacts": metrics["contacts"],
        "grasp_contact_count": metrics.get("grasp_contact_count", ""),
        "slip_mm": metrics["slip_mm"],
        "pull_mm": metrics["pull_mm"],
        "jaw_force_N": metrics["jaw_force_N"],
        "grasp_tilt_deg": metrics["grasp_tilt_deg"],
        "carry": "physics",
        "physics_after_clamp": 1,
        "clamp_swing_mm": metrics["clamp_swing_mm"],
        "stock_ecc_mm": metrics["stock_ecc_mm"],
        "rod_external_force_max_N": metrics["rod_external_force_max_N"],
        "robot_machine_contact_max": metrics["robot_machine_contact_max"],
        "feed_residual_mm": result["ik_residuals_mm"]["feed"],
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inserts",
        nargs="+",
        type=float,
        default=DEFAULT_INSERTS_MM,
        metavar="MM",
        help="insertion depths to test (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV output path (default: %(default)s)",
    )
    args = parser.parse_args()

    if any(value <= 0 for value in args.inserts):
        parser.error("insertion depths must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output.parent / (args.output.stem + "_cases")
    cases_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for insert_mm in args.inserts:
        case_id = "d%g-%s" % (insert_mm, uuid.uuid4().hex[:8])
        row = run_case(insert_mm, cases_dir / (case_id + ".json"),
                       case_id=case_id)
        rows.append(row)
        print(
            "D=%g mm  %-5s hard=%s quality=%s coax=%s mm tilt=%s deg "
            "swing=%s mm stock_ecc=%s mm"
            % (insert_mm, row["status"], row["hard_pass"],
               row["quality_pass"], row["coax_mm"], row["tilt_deg"],
               row["clamp_swing_mm"], row["stock_ecc_mm"])
        )

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    passed = [row["insert_mm"] for row in rows if row["status"] == "PASS"]
    if passed:
        print("Shallowest PASS among tested depths: %g mm" % min(passed))
    else:
        print("No tested depth passed.")
    print("CSV: %s" % args.output)
    if any(row["status"] == "ERROR" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
