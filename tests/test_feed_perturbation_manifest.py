import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diag_feed_perturbation_manifest as manifest


class FeedPerturbationManifestTests(unittest.TestCase):
    def test_grid_has_27_unique_shared_physical_samples(self):
        result = manifest.build_manifest()
        design = result["physical_pose_design"]
        samples = design["samples"]

        self.assertEqual(27, design["sample_count"])
        self.assertEqual(27, len(samples))
        self.assertEqual(
            27, len({sample["sample_id"] for sample in samples}))
        self.assertEqual(
            27,
            len({
                tuple(sample["requested_pose_before_settle"].values())
                for sample in samples
            }),
        )
        self.assertEqual(
            {
                (dy, dz, yaw)
                for dy in manifest.DY_LEVELS_MM
                for dz in manifest.DZ_LEVELS_MM
                for yaw in manifest.YAW_LEVELS_DEG
            },
            {
                (
                    sample["requested_pose_before_settle"]["dy_mm"],
                    sample["requested_pose_before_settle"]["dz_mm"],
                    sample["requested_pose_before_settle"][
                        "yaw_world_z_deg"],
                )
                for sample in samples
            },
        )

    def test_manifest_is_reproducible_and_matches_tracked_file(self):
        expected = manifest._canonical_json(manifest.build_manifest())
        tracked = manifest.DEFAULT_MANIFEST.read_text(encoding="utf-8")
        self.assertEqual(expected, tracked)
        self.assertEqual(expected, manifest._canonical_json(
            manifest.build_manifest()))

    def test_modes_are_paired_without_claiming_m2_sensor_accuracy(self):
        result = manifest.build_manifest()
        paired = result["paired_comparison"]
        seeds = result["seed_policy"]

        self.assertTrue(
            paired["same_physical_sample_across_modes_and_depths"])
        self.assertEqual(243, paired["expected_initial_grid_runs"])
        self.assertEqual(
            "27 physical samples x 3 depths x 3 modes",
            paired["run_count_formula"])
        self.assertIn("Exploratory", paired["m2_error_basis"])
        self.assertEqual(
            manifest.M2_GRID_ERROR_SEED,
            paired["m2_grid_error_seed"])
        self.assertEqual(
            "canonical cartesian product; deterministic and needs no seed",
            seeds["grid_generation"])
        self.assertEqual(
            {"M0", "M1", "M2"},
            set(paired["modes"]))

    def test_future_random_splits_have_fixed_distinct_seeds(self):
        result = manifest.build_manifest()["future_random_splits"]
        seeds = result["seeds"]

        self.assertEqual(20, result["samples_per_split"])
        self.assertNotEqual(seeds["training"], seeds["holdout"])
        self.assertTrue(result["holdout_must_not_be_used_for_tuning"])
        self.assertEqual(
            "P4.5", result["exact_rows_generated_and_saved_in"])

    def test_each_sample_has_deterministic_m2_error_within_envelope(self):
        result = manifest.build_manifest()
        envelope = result["paired_comparison"]["m2_error_range"]
        for sample in result["physical_pose_design"]["samples"]:
            error = sample["m2_estimation_error"]
            self.assertGreaterEqual(error["dy_mm"], envelope["dy_mm"][0])
            self.assertLessEqual(error["dy_mm"], envelope["dy_mm"][1])
            self.assertGreaterEqual(error["dz_mm"], envelope["dz_mm"][0])
            self.assertLessEqual(error["dz_mm"], envelope["dz_mm"][1])
            self.assertGreaterEqual(
                error["yaw_world_z_deg"],
                envelope["yaw_world_z_deg"][0])
            self.assertLessEqual(
                error["yaw_world_z_deg"],
                envelope["yaw_world_z_deg"][1])


if __name__ == "__main__":
    unittest.main()
