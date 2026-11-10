"""Repeatable contact-solref sensitivity experiment for the P5.1 nominal case.

Only the shaft and four SO-101 V-jaw collision geoms change. The shaft uses
that solref for every contact partner, so chuck contact changes are measured
as an outcome rather than treated as an independent control variable.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEED_SCRIPT = ROOT / "src" / "diag_feed_chuck.py"
DEFAULT_OUTPUT = ROOT / "_build" / "p5.3-contact-sensitivity.csv"
DEFAULT_SOLREF_TIMES_S = (0.020, 0.010, 0.005, 0.002)
CASE_TIMEOUT_SECONDS = 600
NON_FEED_OVERRIDES_TO_CLEAR = {
    "GRIP_RAMP", "SMOOTH", "SO101_KP", "SO101_KV", "SM_JAW_BORE_Q",
}

CSV_FIELDS = (
    "case_id",
    "solref_time_s",
    "status",
    "execution_status",
    "hard_pass",
    "quality_pass",
    "overall_pass",
    "coax_mm",
    "tilt_deg",
    "chuck_contacts",
    "slip_mm",
    "pull_mm",
    "jaw_force_N",
    "clamp_swing_mm",
    "stock_ecc_mm",
    "v_jaw_pair",
    "v_jaw_max_penetration_mm",
    "v_jaw_overlap_steps",
    "v_jaw_contact_steps",
    "v_jaw_peak_normal_force_N",
    "chuck_pair",
    "chuck_phase",
    "chuck_max_penetration_mm",
    "chuck_overlap_steps",
    "chuck_contact_steps",
    "chuck_peak_normal_force_N",
    "result_json",
    "error",
)


def _number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _penetration_rows(result):
    by_phase = result.get("penetration_diagnostics", {}).get("by_phase", {})
    rows = []
    for phase, pairs in by_phase.items():
        if not isinstance(pairs, dict):
            continue
        for pair, metrics in pairs.items():
            rows.append((phase, pair, metrics))
    return rows


def _peak_force(result, pair_fragment, category):
    feed = (
        result.get("contact_diagnostics", {})
        .get("scopes", {})
        .get("feed", {})
        .get("overall", {})
    )
    values = [
        metrics.get("peak_normal_force_per_contact_N")
        for key, metrics in feed.get("pairs", {}).items()
        if pair_fragment in key and metrics.get("category") == category
    ]
    values = [value for value in values if _number(value)]
    return max(values) if values else None


def summarize_result(result, solref_time_s, result_path):
    """Validate one result and flatten its relevant contact metrics."""
    if (not isinstance(result, dict)
            or result.get("schema_version") not in (14, 15)):
        raise ValueError("expected a schema v14 or v15 result JSON")
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
    override = result.get("environment_overrides", {}).get(
        "FEED_V_CONTACT_SOLREF_TIME")
    try:
        recorded_solref_time = float(override)
    except (TypeError, ValueError):
        recorded_solref_time = float("nan")
    if not _number(recorded_solref_time) or not math.isclose(
            recorded_solref_time, solref_time_s,
            rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("requested solref value is missing from result metadata")

    rows = _penetration_rows(result)
    carry_vjaw = [
        row for row in rows
        if row[0] == "搬运" and "_jaw_pad_v" in row[1]
    ]
    if not carry_vjaw:
        raise ValueError("no V-jaw contact metrics in the carry phase")
    for _, _, metrics in carry_vjaw:
        params = metrics.get("contact_geometry_parameters", [])
        actual = {
            item.get("name"): item.get("solref", [None])[0]
            for item in params
        }
        shaft_solref = actual.get("shaft_geom")
        if (not _number(shaft_solref)
                or not math.isclose(
                    float(shaft_solref), solref_time_s,
                    rel_tol=0.0, abs_tol=1e-12)
                or not any(
                    name.endswith("_jaw_pad_v1")
                    or name.endswith("_jaw_pad_v2")
                    for name, value in actual.items()
                    if _number(value)
                    and math.isclose(
                        float(value), solref_time_s,
                        rel_tol=0.0, abs_tol=1e-12))):
            raise ValueError("V-jaw contact pair does not contain requested solref")
    vjaw_phase, vjaw_pair, vjaw = max(
        carry_vjaw, key=lambda row: row[2].get("max_penetration_mm", -1.0))

    chuck_rows = [
        row for row in rows if "jaw_geom_" in row[1]
    ]
    if chuck_rows:
        chuck_phase, chuck_pair, chuck = max(
            chuck_rows,
            key=lambda row: row[2].get("max_penetration_mm", -1.0))
        chuck_penetration = chuck.get("max_penetration_mm")
        chuck_overlap_steps = chuck.get("overlap_steps")
        chuck_contact_steps = chuck.get("contact_steps")
    else:
        chuck_phase = chuck_pair = ""
        chuck_penetration = chuck_overlap_steps = chuck_contact_steps = None

    metrics = result.get("metrics", {})
    return {
        "case_id": result.get("case_id", ""),
        "solref_time_s": solref_time_s,
        "status": "PASS" if result.get("overall_pass") else "FAIL",
        "execution_status": result.get("execution_status", ""),
        "hard_pass": result.get("hard_pass"),
        "quality_pass": result.get("quality_pass"),
        "overall_pass": result.get("overall_pass"),
        "coax_mm": metrics.get("coax_mm"),
        "tilt_deg": metrics.get("tilt_deg"),
        "chuck_contacts": metrics.get("contacts"),
        "slip_mm": metrics.get("slip_mm"),
        "pull_mm": metrics.get("pull_mm"),
        "jaw_force_N": metrics.get("jaw_force_N"),
        "clamp_swing_mm": metrics.get("clamp_swing_mm"),
        "stock_ecc_mm": metrics.get("stock_ecc_mm"),
        "v_jaw_pair": vjaw_pair,
        "v_jaw_max_penetration_mm": vjaw.get("max_penetration_mm"),
        "v_jaw_overlap_steps": vjaw.get("overlap_steps"),
        "v_jaw_contact_steps": vjaw.get("contact_steps"),
        "v_jaw_peak_normal_force_N": _peak_force(
            result, "jaw_pad_v", "robot_workpiece"),
        "chuck_pair": chuck_pair,
        "chuck_phase": chuck_phase,
        "chuck_max_penetration_mm": chuck_penetration,
        "chuck_overlap_steps": chuck_overlap_steps,
        "chuck_contact_steps": chuck_contact_steps,
        "chuck_peak_normal_force_N": _peak_force(
            result, "jaw_geom_", "workpiece_machine"),
        "result_json": str(result_path),
        "error": "",
    }


def _failed_row(solref_time_s, result_path, error):
    return {
        **{field: "" for field in CSV_FIELDS},
        "solref_time_s": solref_time_s,
        "status": "ERROR",
        "result_json": str(result_path),
        "error": str(error),
    }


def _solref_label(solref_time_s):
    return format(solref_time_s, ".12g").replace("-", "m").replace(".", "p")


def run_case(solref_time_s, json_path):
    label = _solref_label(solref_time_s)
    case_id = "p53-solref-%s" % label
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.unlink(missing_ok=True)

    env = os.environ.copy()
    for key in tuple(env):
        if (key.startswith("FEED_")
                or key in NON_FEED_OVERRIDES_TO_CLEAR):
            env.pop(key)
    env.update({
        "FEED_CASE_ID": case_id,
        "FEED_RESULT_JSON": str(json_path.resolve()),
        "FEED_V_CONTACT_SOLREF_TIME": format(solref_time_s, ".12g"),
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
        with json_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        row = summarize_result(result, solref_time_s, json_path)
        expected_rc = 0 if result.get("overall_pass") else 1
        if completed.returncode != expected_rc:
            raise ValueError(
                "exit code %d conflicts with JSON overall_pass=%r"
                % (completed.returncode, result.get("overall_pass")))
        return row
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError,
            json.JSONDecodeError, ValueError) as error:
        return _failed_row(solref_time_s, json_path, error)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solref-times",
        nargs="+",
        type=float,
        default=DEFAULT_SOLREF_TIMES_S,
        metavar="SECONDS",
        help="positive solref time constants (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="CSV output path (default: %(default)s)",
    )
    parser.add_argument(
        "--json-dir", type=Path,
        help="raw per-case JSON directory (default: alongside CSV)",
    )
    args = parser.parse_args(argv)
    if (not args.solref_times
            or any(not math.isfinite(value) or value <= 0.0
                   for value in args.solref_times)):
        parser.error("solref times must be positive finite values")
    if len(set(args.solref_times)) != len(args.solref_times):
        parser.error("solref times must be unique")

    json_dir = args.json_dir or args.output.with_name(
        args.output.stem + "_cases")
    json_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for solref_time_s in args.solref_times:
        json_path = json_dir / ("solref_%s_s.json"
                                % _solref_label(solref_time_s))
        row = run_case(solref_time_s, json_path)
        rows.append(row)
        print(
            "solref=%.3f s %-5s V爪重叠=%s mm 卡盘重叠=%s mm PASS=%s"
            % (solref_time_s, row["status"],
               row.get("v_jaw_max_penetration_mm", ""),
               row.get("chuck_max_penetration_mm", ""),
               row.get("overall_pass", "")),
            flush=True,
        )

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print("CSV: %s" % args.output.resolve())
    print("JSON: %s" % json_dir.resolve())
    return 1 if any(row["status"] == "ERROR" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
