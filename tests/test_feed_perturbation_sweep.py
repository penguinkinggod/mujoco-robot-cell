import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diag_feed_perturbation_sweep as sweep
import diag_feed_perturbation_manifest as manifest


class FeedPerturbationSweepTests(unittest.TestCase):
    def test_matrix_is_243_unique_paired_cases(self):
        cases = sweep._expected_cases()

        self.assertEqual(243, len(cases))
        self.assertEqual(243, len({case["case_id"] for case in cases}))
        self.assertEqual(
            27 * 3,
            len({(case["sample_id"], case["insert_mm"])
                 for case in cases}))
        self.assertEqual(
            {"M0", "M1", "M2"},
            {case["planning_mode"] for case in cases})

    def test_each_m2_case_uses_frozen_error_and_others_use_zero_error(self):
        cases = sweep._expected_cases()
        samples = {
            sample["sample_id"]: sample
            for sample in manifest.build_manifest()[
                "physical_pose_design"]["samples"]
        }
        for case in cases:
            if case["planning_mode"] == "M2":
                error = samples[case["sample_id"]]["m2_estimation_error"]
                self.assertEqual(error["dy_mm"], case["m2_error_dy_mm"])
                self.assertEqual(error["dz_mm"], case["m2_error_dz_mm"])
                self.assertEqual(
                    error["yaw_world_z_deg"], case["m2_error_yaw_deg"])
            else:
                self.assertEqual(0.0, case["m2_error_dy_mm"])
                self.assertEqual(0.0, case["m2_error_dz_mm"])
                self.assertEqual(0.0, case["m2_error_yaw_deg"])

    def test_resume_reader_rejects_duplicates_and_unknown_ids(self):
        fieldnames = sweep._fieldnames()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.csv"
            cases = sweep._expected_cases()
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    key: "" for key in fieldnames
                } | {"case_id": cases[0]["case_id"]})
            ids, rows = sweep._load_existing(
                path, fieldnames,
                {case["case_id"] for case in cases})
            self.assertEqual({cases[0]["case_id"]}, ids)
            self.assertEqual(1, len(rows))

            with path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writerow({
                    key: "" for key in fieldnames
                } | {"case_id": "not-in-the-manifest"})
            with self.assertRaises(ValueError):
                sweep._load_existing(
                    path, fieldnames,
                    {case["case_id"] for case in cases})

    def test_summary_counts_csv_depths_loaded_as_strings(self):
        output = io.StringIO()
        with redirect_stdout(output):
            sweep._write_summary([
                {"planning_mode": "M0", "insert_mm": "8.0",
                 "status": "PASS"}
            ], "saved.csv")

        self.assertIn("M0 8 mm: PASS=1", output.getvalue())


if __name__ == "__main__":
    unittest.main()
