"""Run the complete robot-to-chuck feed check at several insertion depths.

This holds the current gripper, workpiece, approach and handoff settings fixed.
Only FEED_INSERT_MM changes. Results are written to _build/feed_insert_sweep.csv.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEED_SCRIPT = ROOT / "src" / "diag_feed_chuck.py"
DEFAULT_OUTPUT = ROOT / "_build" / "feed_insert_sweep.csv"
DEFAULT_INSERTS_MM = (5, 6, 7, 8, 9, 10, 15, 20, 25, 30)


def run_case(insert_mm, json_path, case_id=None, dy_mm=0.0, dz_mm=0.0,
             roll_deg=0.0, random_seed=None):
    case_id = case_id or uuid.uuid4().hex[:12]
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.unlink(missing_ok=True)
    env = os.environ.copy()
    for key in tuple(env):
        if key.startswith("FEED_"):
            env.pop(key)
    env["FEED_INSERT_MM"] = format(insert_mm, "g")
    env["FEED_CASE_ID"] = case_id
    env["FEED_RESULT_JSON"] = str(json_path.resolve())

    completed = subprocess.run(
        [sys.executable, str(FEED_SCRIPT), "--v-jaw", "--lift", "--no-lock",
         "--initial-dy-mm", str(dy_mm), "--initial-dz-mm", str(dz_mm),
         "--initial-roll-deg", str(roll_deg), "--case-id", case_id]
        + ([] if random_seed is None else
           ["--random-seed", str(random_seed)]),
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    row = {
        "case_id": case_id,
        "random_seed": "" if random_seed is None else random_seed,
        "insert_mm": insert_mm,
        "dy_mm": dy_mm,
        "dz_mm": dz_mm,
        "roll_deg": roll_deg,
        "status": "ERROR",
        "execution_status": "",
        "evaluation_scope": "",
        "trial_data_valid": "",
        "include_in_primary_trial_stats": "",
        "physical_task_success": "",
        "outcome_class": "missing_result",
        "diagnostic_mode": "",
        "abort_stage": "",
        "hard_pass": "",
        "quality_pass": "",
        "hard_failures": "",
        "quality_failures": "",
        "exit_code": completed.returncode,
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
    try:
        with Path(json_path).open(encoding="utf-8") as stream:
            result = json.load(stream)
    except (OSError, json.JSONDecodeError):
        result = None
    if result:
        execution_status = result.get("execution_status", "completed")
        if execution_status == "error":
            row.update({
                "execution_status": execution_status,
                "evaluation_scope": result.get("evaluation_scope", ""),
                "trial_data_valid": result.get("trial_data_valid", False),
                "include_in_primary_trial_stats": result.get(
                    "include_in_primary_trial_stats", False),
                "physical_task_success": result.get(
                    "physical_task_success", False),
                "outcome_class": result.get("outcome_class", "runtime_error"),
                "diagnostic_mode": result.get("config", {}).get(
                    "diagnostic_mode", ""),
                "abort_stage": result.get("abort_stage", ""),
                "error": result.get("abort_reason", "simulation_error"),
            })
            return row
        metrics = result["metrics"]
        row.update({
            "status": (
                "ABORT" if execution_status == "aborted"
                else "PASS" if result["overall_pass"] else "FAIL"),
            "execution_status": execution_status,
            "evaluation_scope": result.get("evaluation_scope", ""),
            "trial_data_valid": result.get("trial_data_valid", ""),
            "include_in_primary_trial_stats": result.get(
                "include_in_primary_trial_stats", ""),
            "physical_task_success": result.get(
                "physical_task_success", ""),
            "outcome_class": result.get("outcome_class", ""),
            "diagnostic_mode": result.get("config", {}).get(
                "diagnostic_mode", ""),
            "abort_stage": result.get("abort_stage", ""),
            "hard_pass": result["hard_pass"],
            "quality_pass": result["quality_pass"],
            "hard_failures": ";".join(result["hard_failures"]),
            "quality_failures": ";".join(result["quality_failures"]),
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
            "rod_external_force_max_N": metrics[
                "rod_external_force_max_N"],
            "robot_machine_contact_max": metrics[
                "robot_machine_contact_max"],
            "feed_residual_mm": result["ik_residuals_mm"].get("feed", ""),
        })
    else:
        row["error"] = output[-1200:].strip()
        row.update({
            "execution_status": "missing",
            "trial_data_valid": False,
            "include_in_primary_trial_stats": False,
            "physical_task_success": False,
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
