"""Run the frozen P4.3 pose x insertion-depth x planning-mode matrix."""
import argparse
import csv
import json
import os
from pathlib import Path

import diag_feed_insert_sweep as feed_sweep
import diag_feed_perturbation_manifest as manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "_build" / "p44_perturbation_sweep.csv"


def _expected_cases():
    frozen = json.loads(manifest.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    cases = []
    for sample in frozen["physical_pose_design"]["samples"]:
        requested = sample["requested_pose_before_settle"]
        error = sample["m2_estimation_error"]
        for insert_mm in frozen["paired_comparison"]["insert_depth_mm"]:
            for mode in manifest.MODES:
                case_id = "%s-%s-d%g-%s" % (
                    "p44", sample["sample_id"], insert_mm, mode)
                cases.append({
                    "case_id": case_id,
                    "sample_id": sample["sample_id"],
                    "insert_mm": float(insert_mm),
                    "dy_mm": float(requested["dy_mm"]),
                    "dz_mm": float(requested["dz_mm"]),
                    "yaw_deg": float(requested["yaw_world_z_deg"]),
                    "planning_mode": mode,
                    "m2_error_dy_mm": (
                        float(error["dy_mm"]) if mode == "M2" else 0.0),
                    "m2_error_dz_mm": (
                        float(error["dz_mm"]) if mode == "M2" else 0.0),
                    "m2_error_yaw_deg": (
                        float(error["yaw_world_z_deg"])
                        if mode == "M2" else 0.0),
                })
    return cases


def _fieldnames():
    row = feed_sweep._new_row(
        0, "", "template", 0, 0, 0, None,
        planning_mode="M0", sample_id="template")
    return list(row)


def _load_existing(path, fieldnames, expected_ids):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fieldnames:
            raise ValueError("existing CSV has a different schema")
        rows = list(reader)
    ids = [row.get("case_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("existing CSV contains duplicate case IDs")
    unknown = sorted(set(ids) - expected_ids)
    if unknown:
        raise ValueError("existing CSV contains unknown case IDs: %s"
                         % ", ".join(unknown[:5]))
    return set(ids), rows


def _write_summary(rows, output_path):
    counts = {}
    for row in rows:
        key = (row["planning_mode"], float(row["insert_mm"]), row["status"])
        counts[key] = counts.get(key, 0) + 1
    print("P4.4 status counts by mode/depth:")
    for mode in manifest.MODES:
        for depth in manifest.INSERT_DEPTHS_MM:
            parts = [
                "%s=%d" % (status, counts.get((mode, depth, status), 0))
                for status in ("PASS", "FAIL", "ABORT", "ERROR")
            ]
            print("  %s %g mm: %s" % (mode, depth, " ".join(parts)))
    print("CSV: %s" % output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume", action="store_true",
        help="append cases missing from an existing P4.4 CSV")
    parser.add_argument(
        "--max-cases", type=int,
        help="run only the first N missing cases; intended for preflight")
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")

    expected_manifest = manifest._canonical_json(manifest.build_manifest())
    try:
        actual_manifest = manifest.DEFAULT_MANIFEST.read_text(encoding="utf-8")
    except OSError as error:
        parser.error("cannot read frozen P4.3 manifest: %s" % error)
    if actual_manifest != expected_manifest:
        parser.error("frozen P4.3 manifest does not match its generator")

    cases = _expected_cases()
    fieldnames = _fieldnames()
    expected_ids = {case["case_id"] for case in cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases_dir = args.output.parent / (args.output.stem + "_cases")
    cases_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = set()
    old_rows = []
    if args.output.exists():
        if not args.resume:
            parser.error(
                "output already exists; choose a new --output or use --resume")
        try:
            existing_ids, old_rows = _load_existing(
                args.output, fieldnames, expected_ids)
        except (OSError, ValueError) as error:
            parser.error("cannot resume: %s" % error)
    elif args.resume:
        parser.error("--resume requires an existing CSV")

    pending = [case for case in cases
               if case["case_id"] not in existing_ids]
    if args.max_cases is not None:
        pending = pending[:args.max_cases]
    if not pending:
        _write_summary(old_rows, args.output)
        print("No missing cases.")
        return 1 if any(row["status"] == "ERROR" for row in old_rows) else 0

    mode = "a" if args.output.exists() else "w"
    all_rows = list(old_rows)
    with args.output.open(
            mode, newline="", encoding="utf-8", buffering=1) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
            stream.flush()
            os.fsync(stream.fileno())
        for index, case in enumerate(pending, start=1):
            json_path = cases_dir / (case["case_id"] + ".json")
            row = feed_sweep.run_case(
                case["insert_mm"], json_path,
                case_id=case["case_id"],
                dy_mm=case["dy_mm"],
                dz_mm=case["dz_mm"],
                yaw_deg=case["yaw_deg"],
                planning_mode=case["planning_mode"],
                sample_id=case["sample_id"],
                m2_error_dy_mm=case["m2_error_dy_mm"],
                m2_error_dz_mm=case["m2_error_dz_mm"],
                m2_error_yaw_deg=case["m2_error_yaw_deg"],
            )
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
            all_rows.append(row)
            print(
                "[%d/%d] %-36s %-5s status=%s hard=%s quality=%s"
                % (index, len(pending), case["case_id"], case["planning_mode"],
                   row["status"], row["hard_pass"], row["quality_pass"]),
                flush=True,
            )

    _write_summary(all_rows, args.output)
    if len(all_rows) < len(cases):
        print("Partial run: %d/%d cases recorded; continue with --resume."
              % (len(all_rows), len(cases)))
        return 0
    return 1 if any(row["status"] == "ERROR" for row in all_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
