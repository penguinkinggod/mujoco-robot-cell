"""Create or verify the frozen P4.3 shared perturbation manifest."""
import argparse
import json
import random
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "p4.3-shared-samples-v1.json"

DY_LEVELS_MM = (-3.0, 0.0, 3.0)
DZ_LEVELS_MM = (-2.0, 0.0, 2.0)
YAW_LEVELS_DEG = (-5.0, 0.0, 5.0)
INSERT_DEPTHS_MM = (8.0, 10.0, 25.0)
MODES = ("M0", "M1", "M2")

M2_GRID_ERROR_SEED = 20261109
TRAIN_SAMPLE_SEED = 20261110
HOLDOUT_SAMPLE_SEED = 20261111
RANDOM_SAMPLES_PER_SPLIT = 20


def _sample_id(dy_mm, dz_mm, yaw_deg):
    def token(axis, value):
        if value == 0:
            sign = "z"
        else:
            sign = "m" if value < 0 else "p"
        return "%s_%s%s" % (axis, sign, int(abs(value)))

    return "p43_%s_%s_%s" % (
        token("dy", dy_mm), token("dz", dz_mm), token("yaw", yaw_deg))


def _make_m2_error_samples():
    rng = random.Random(M2_GRID_ERROR_SEED)
    rows = []
    for _ in range(
            len(DY_LEVELS_MM) * len(DZ_LEVELS_MM) * len(YAW_LEVELS_DEG)):
        rows.append({
            "dy_mm": round(rng.uniform(-1.0, 1.0), 6),
            "dz_mm": round(rng.uniform(-1.0, 1.0), 6),
            "yaw_world_z_deg": round(rng.uniform(-2.0, 2.0), 6),
        })
    return rows


def build_manifest():
    """Return the canonical, machine-readable P4.3 sample definition."""
    m2_errors = iter(_make_m2_error_samples())
    samples = []
    for dy_mm, dz_mm, yaw_deg in product(
            DY_LEVELS_MM, DZ_LEVELS_MM, YAW_LEVELS_DEG):
        samples.append({
            "sample_id": _sample_id(dy_mm, dz_mm, yaw_deg),
            "requested_pose_before_settle": {
                "dx_mm": 0.0,
                "dy_mm": dy_mm,
                "dz_mm": dz_mm,
                "yaw_world_z_deg": yaw_deg,
            },
            "m2_estimation_error": next(m2_errors),
        })

    run_count = (
        len(samples) * len(INSERT_DEPTHS_MM) * len(MODES))
    return {
        "manifest_id": "p4.3-shared-perturbation-v1",
        "date": "2026-11-08",
        "status": "frozen",
        "seed_policy": {
            "grid_generation": (
                "canonical cartesian product; deterministic and needs no seed"),
            "m2_grid_error_seed": M2_GRID_ERROR_SEED,
            "training_sample_seed": TRAIN_SAMPLE_SEED,
            "holdout_sample_seed": HOLDOUT_SAMPLE_SEED,
        },
        "scope": {
            "workpiece": "cylinder_d20_l80",
            "fixture": "current_90deg_v_jaw_and_bushing",
            "settle_steps": 600,
            "settle_steps_source": "diag_feed_chuck.py initial_settle",
            "requested_pose_stage": (
                "after mj_resetData and before initial_settle"),
            "measured_pose_stage": (
                "after initial_settle and before robot approach"),
            "interpretation": (
                "incoming placement plus fixture self-centering; "
                "not robot-only disturbance tolerance"),
            "criterion_set_id": "feed-criteria-v1",
        },
        "physical_pose_design": {
            "design": "full_factorial",
            "dx_mm": [0.0],
            "dy_mm": list(DY_LEVELS_MM),
            "dz_mm": list(DZ_LEVELS_MM),
            "yaw_world_z_deg": list(YAW_LEVELS_DEG),
            "sample_count": len(samples),
            "order": "dy_then_dz_then_yaw_ascending",
            "samples": samples,
        },
        "paired_comparison": {
            "insert_depth_mm": list(INSERT_DEPTHS_MM),
            "modes": {
                "M0": (
                    "Use the fixed nominal grasp target; do not use "
                    "the per-case measured workpiece pose for planning."),
                "M1": (
                    "Use the settled true workpiece pose as an ideal "
                    "oracle and transform the calibrated grasp pose."),
                "M2": (
                    "Use settled true pose plus the sample's synthetic "
                    "estimation error; never plan directly from truth."),
            },
            "m2_error_definition": (
                "estimated_pose = settled_true_pose + error"),
            "m2_error_range": {
                "dy_mm": [-1.0, 1.0],
                "dz_mm": [-1.0, 1.0],
                "yaw_world_z_deg": [-2.0, 2.0],
            },
            "m2_error_basis": (
                "Exploratory simulation assumption only; no camera or "
                "sensor measurement supports these bounds."),
            "m2_grid_error_seed": M2_GRID_ERROR_SEED,
            "same_physical_sample_across_modes_and_depths": True,
            "expected_initial_grid_runs": run_count,
            "run_count_formula": "27 physical samples x 3 depths x 3 modes",
        },
        "future_random_splits": {
            "samples_per_split": RANDOM_SAMPLES_PER_SPLIT,
            "physical_pose_distribution": {
                "dy_mm": "uniform[-3.0, 3.0]",
                "dz_mm": "uniform[-2.0, 2.0]",
                "yaw_world_z_deg": "uniform[-5.0, 5.0]",
            },
            "m2_error_distribution": {
                "dy_mm": "uniform[-1.0, 1.0]",
                "dz_mm": "uniform[-1.0, 1.0]",
                "yaw_world_z_deg": "uniform[-2.0, 2.0]",
            },
            "seeds": {
                "training": TRAIN_SAMPLE_SEED,
                "holdout": HOLDOUT_SAMPLE_SEED,
            },
            "holdout_must_not_be_used_for_tuning": True,
            "exact_rows_generated_and_saved_in": "P4.5",
        },
    }


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    expected = _canonical_json(build_manifest())

    if args.write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected, encoding="utf-8", newline="\n")
        print("Wrote frozen P4.3 manifest: %s" % args.output)
        return 0

    try:
        actual = args.output.read_text(encoding="utf-8")
    except OSError as error:
        parser.error("cannot read manifest: %s" % error)
    if actual != expected:
        print("Manifest differs from deterministic generator: %s" % args.output)
        return 1
    print("P4.3 manifest matches deterministic generator: %s" % args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
