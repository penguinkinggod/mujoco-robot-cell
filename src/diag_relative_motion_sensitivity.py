"""Small one-factor-at-a-time P5.7 study of gripper-relative motion."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FEED_SCRIPT = ROOT / "src" / "diag_feed_chuck.py"
CASE_TIMEOUT_SECONDS = 600
NON_FEED_OVERRIDES_TO_CLEAR = {
    "GRIP_RAMP", "SMOOTH", "SO101_KP", "SO101_KV", "SM_JAW_BORE_Q",
}
CASES = (
    ("baseline", "none", {}),
    ("slow_carry", "carry_speed", {"FEED_CARRY_V_CRUISE": "0.275"}),
    ("long_v_jaw", "v_jaw_length",
     {"FEED_V_JAW_LENGTH_MM": "36"}),
    ("stiffer_contact", "contact_solref",
     {"FEED_V_CONTACT_SOLREF_TIME": "0.002"}),
    ("slow_carry_repeat", "carry_speed",
     {"FEED_CARRY_V_CRUISE": "0.275"}),
    ("stiffer_contact_repeat", "contact_solref",
     {"FEED_V_CONTACT_SOLREF_TIME": "0.002"}),
)
CSV_FIELDS = (
    "case_id", "changed_factor", "changed_value", "execution_status",
    "overall_status", "outcome_class", "hard_failures", "quality_failures",
    "carry_peak_mm", "carry_peak_time_s",
    "carry_peak_vector_tool_mm", "carry_axis_peak_deg", "carry_duration_s",
    "carry_max_step_mm", "peak_vjaw_contacts", "peak_vjaw_min_distance_mm",
    "peak_vjaw_normal_force_sum_N", "carry_vjaw_max_penetration_mm",
    "feed_peak_mm", "feed_peak_time_s", "final_coax_mm", "final_tilt_deg",
    "final_stock_ecc_mm", "chuck_contacts", "result_json", "log_file",
)


def _finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def summarize_result(result, case_id, changed_factor, changed_value,
                     result_path, log_path):
    if (not isinstance(result, dict)
            or result.get("schema_version") not in (14, 15, 16)):
        raise ValueError("expected a schema v14, v15, or v16 result JSON")
    if result.get("execution_status") != "completed":
        raise ValueError(
            "case did not complete: %s" % result.get("execution_status"))
    if result.get("control_mode", {}).get("carry") != "unlocked_physics":
        raise ValueError("case did not use unlocked physical carry")
    if result.get("control_mode", {}).get("planning_mode") != "M1":
        raise ValueError("case did not use M1 planning")
    if result.get("inputs", {}).get("initial_settle_steps") != 600:
        raise ValueError("case did not use the frozen 600-step settle")
    config = result.get("config", {})
    if (config.get("v_jaw") is not True
            or config.get("lift") is not True
            or config.get("carry_lock") is not False
            or config.get("initial_offset_mm") != [0.0, 0.0, 0.0]
            or config.get("initial_yaw_world_z_deg") != 0.0
            or config.get("planning_mode") != "M1"):
        raise ValueError("case does not match the frozen nominal inputs")

    trace = result.get("gripper_relative_workpiece_motion", {})
    history = trace.get("time_history")
    if (trace.get("status") != "observed"
            or trace.get("time_history_enabled") is not True
            or not isinstance(history, list) or not history):
        raise ValueError("relative-motion time history is missing")
    carry = [row for row in history if row.get("phase_group") == "搬运"]
    feed = [row for row in history if row.get("phase_group") == "进给"]
    if not carry or not feed:
        raise ValueError("trace must include carry and feed samples")
    carry_peak = max(
        carry, key=lambda row: row["translation_change_from_carry_start_mm"])
    feed_peak = max(
        feed, key=lambda row: row["translation_change_from_carry_start_mm"])
    carry_peak_value = carry_peak["translation_change_from_carry_start_mm"]
    feed_peak_value = feed_peak["translation_change_from_carry_start_mm"]
    if not _finite_number(carry_peak_value) or not _finite_number(
            feed_peak_value):
        raise ValueError("trace contains a non-finite motion metric")

    carry_times = [row["sim_time_s"] for row in carry]
    carry_max_step = max(
        row["translation_change_since_previous_step_mm"] for row in carry)
    vjaw_contacts = []
    vjaw_distances = []
    vjaw_normal_force_sums = []
    for pair in carry_peak.get("shaft_contact_pairs", []):
        if ("robot_workpiece" in pair.get("category", "")
                and any("jaw_pad_v" in geom
                        for geom in pair.get("geoms", []))):
            vjaw_contacts.append(pair.get("contact_count", 0))
            distance = pair.get("minimum_distance_mm")
            if _finite_number(distance):
                vjaw_distances.append(distance)
            force_sum = pair.get("normal_force_sum_N")
            if _finite_number(force_sum):
                vjaw_normal_force_sums.append(force_sum)

    metrics = result.get("metrics", {})
    carry_vjaw_penetration = (
        result.get("penetration_diagnostics", {})
        .get("by_phase", {})
        .get("搬运", {})
        .get("shaft_geom|moving_jaw_pad_v2", {})
        .get("max_penetration_mm"))
    parameters = result.get("effective_parameters", {})
    controller = parameters.get("controller", {})
    gripper = parameters.get("gripper", {})
    expected_speed = (
        float(changed_value) if changed_factor == "carry_speed" else 0.55)
    expected_length = (
        float(changed_value) if changed_factor == "v_jaw_length" else 24.0)
    if not math.isclose(
            controller.get("carry_v_cruise_m_s", float("nan")),
            expected_speed, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("recorded carry speed does not match case")
    if not math.isclose(
            gripper.get("v_length_mm", float("nan")),
            expected_length, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("recorded V-jaw length does not match case")
    overrides = result.get("environment_overrides", {})
    for key, factor in (
            ("FEED_CARRY_V_CRUISE", "carry_speed"),
            ("FEED_V_JAW_LENGTH_MM", "v_jaw_length")):
        expected = float(changed_value) if changed_factor == factor else None
        recorded = overrides.get(key)
        if expected is None:
            if recorded is not None:
                raise ValueError("unexpected %s override" % key)
        else:
            try:
                recorded = float(recorded)
            except (TypeError, ValueError) as error:
                raise ValueError("%s override is missing" % key) from error
            if not math.isclose(
                    recorded, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("%s override does not match case" % key)
    expected_solref = (
        float(changed_value) if changed_factor == "contact_solref" else None)
    recorded_solref = overrides.get("FEED_V_CONTACT_SOLREF_TIME")
    if expected_solref is None:
        if recorded_solref is not None:
            raise ValueError("unexpected contact solref override")
    else:
        try:
            recorded_solref = float(recorded_solref)
        except (TypeError, ValueError) as error:
            raise ValueError("contact solref override is missing") from error
        if not math.isclose(
                recorded_solref, expected_solref,
                rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("recorded contact solref does not match case")

    return {
        "case_id": case_id,
        "changed_factor": changed_factor,
        "changed_value": changed_value,
        "execution_status": result.get("execution_status"),
        "overall_status": "PASS" if result.get("overall_pass") else "FAIL",
        "outcome_class": result.get("outcome_class"),
        "hard_failures": ";".join(result.get("hard_failures", [])),
        "quality_failures": ";".join(result.get("quality_failures", [])),
        "carry_peak_mm": carry_peak_value,
        "carry_peak_time_s": carry_peak.get("sim_time_s"),
        "carry_peak_vector_tool_mm": json.dumps(
            carry_peak.get("translation_change_from_carry_start_tool_mm"),
            separators=(",", ":")),
        "carry_axis_peak_deg": parameters.get(
            "relative_motion_peak_axis_change_deg",
            trace.get("phases", {}).get("搬运", {}).get(
                "max_relative_axis_change_deg")),
        "carry_duration_s": max(carry_times) - min(carry_times),
        "carry_max_step_mm": carry_max_step,
        "peak_vjaw_contacts": sum(vjaw_contacts),
        "peak_vjaw_min_distance_mm": (
            min(vjaw_distances) if vjaw_distances else ""),
        "peak_vjaw_normal_force_sum_N": (
            sum(vjaw_normal_force_sums)
            if vjaw_normal_force_sums else ""),
        "carry_vjaw_max_penetration_mm": carry_vjaw_penetration,
        "feed_peak_mm": feed_peak_value,
        "feed_peak_time_s": feed_peak.get("sim_time_s"),
        "final_coax_mm": metrics.get("coax_mm"),
        "final_tilt_deg": metrics.get("tilt_deg"),
        "final_stock_ecc_mm": metrics.get("stock_ecc_mm"),
        "chuck_contacts": metrics.get("contacts"),
        "result_json": str(result_path),
        "log_file": str(log_path),
    }


def _case_value(changes):
    if not changes:
        return ""
    return next(iter(changes.values()))


def run_case(case_id, changed_factor, changes, output_dir):
    json_path = output_dir / (case_id + ".json")
    log_path = output_dir / (case_id + ".log")
    env = os.environ.copy()
    for key in tuple(env):
        if (key.startswith("FEED_")
                or key in NON_FEED_OVERRIDES_TO_CLEAR):
            env.pop(key)
    env.update(changes)
    env.update({
        "FEED_CASE_ID": case_id,
        "FEED_RESULT_JSON": str(json_path.resolve()),
        "FEED_TRACE_RELATIVE_MOTION": "1",
        "MUJOCO_GL": "glfw",
        "PYTHONUTF8": "1",
    })
    command = [
        sys.executable, "-X", "utf8", str(FEED_SCRIPT),
        "--v-jaw", "--lift", "--no-lock",
        "--initial-dy-mm", "0",
        "--initial-dz-mm", "0",
        "--initial-yaw-deg", "0",
        "--initial-settle-steps", "600",
        "--control-mode", "M1",
        "--case-id", case_id,
    ]
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
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr
                            if completed.stderr else ""),
        encoding="utf-8",
    )
    with json_path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    row = summarize_result(
        result, case_id, changed_factor, _case_value(changes),
        json_path, log_path)
    expected_rc = 0 if result.get("overall_pass") else 1
    if completed.returncode != expected_rc:
        raise ValueError(
            "%s exit code %d conflicts with overall_pass=%r"
            % (case_id, completed.returncode, result.get("overall_pass")))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        help="new output directory (default: timestamped under _build)")
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (
        ROOT / "_build" / ("p5.7-relative-motion-%s"
                           % datetime.now().strftime("%Y%m%d-%H%M%S")))
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    for case_id, factor, changes in CASES:
        row = run_case(case_id, factor, changes, output_dir)
        rows.append(row)
        print(
            "%-17s %-14s carry峰值=%7.3f mm @ %6.3f s "
            "进给峰值=%7.3f mm 送料=%s"
            % (case_id, factor, row["carry_peak_mm"],
               row["carry_peak_time_s"], row["feed_peak_mm"],
               row["overall_status"]),
            flush=True,
        )

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"schema_version": 1, "cases": rows},
            ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("CSV: %s" % csv_path)
    print("JSON: %s" % summary_path)
    print("Raw cases and logs: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
