import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diag_feed_insert_sweep as sweep


def make_result(case_id, status="completed", passed=True):
    hard_pass = passed
    quality_pass = passed if status == "completed" else None
    result = {
        "schema_version": 6,
        "case_id": case_id,
        "execution_status": status,
        "evaluation_scope": "primary_physics_feed",
        "trial_data_valid": status in ("completed", "aborted"),
        "include_in_primary_trial_stats": status in ("completed", "aborted"),
        "physical_task_success": status == "completed" and passed,
        "outcome_class": (
            "physical_success" if passed else
            "quality_fail" if status == "completed" else
            "aborted" if status == "aborted" else "runtime_error"
        ),
        "config": {"diagnostic_mode": "standard"},
        "metrics": {},
        "hard_pass": hard_pass,
        "quality_pass": quality_pass,
        "overall_pass": passed if status == "completed" else False,
        "hard_failures": [],
        "quality_failures": [] if passed else ["stock_eccentricity"],
        "ik_residuals_mm": {"feed": 1.25},
    }
    if status == "completed":
        result["metrics"] = {
            "coax_mm": 0.64,
            "tilt_deg": 0.2,
            "contacts": 30,
            "slip_mm": 0.0,
            "pull_mm": 0.0,
            "jaw_force_N": 164.0,
            "grasp_tilt_deg": 0.0,
            "grasp_contact_count": 8,
            "clamp_swing_mm": 1.58,
            "stock_ecc_mm": 0.7 if passed else 2.2,
            "rod_external_force_max_N": 0.0,
            "robot_machine_contact_max": 0,
        }
    elif status == "aborted":
        result.update({
            "abort_stage": "grasp",
            "abort_reason": "no_v_jaw_contact",
        })
    elif status == "error":
        result.update({"abort_reason": "non_finite_state"})
    return result


class FeedInsertSweepTests(unittest.TestCase):
    def _run_case(self, result=None, returncode=0, side_effect=None,
                  stale_result=None):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        case_id = "sweep-test-case"
        json_path = Path(temp_dir.name) / (case_id + ".json")
        if stale_result is not None:
            json_path.write_text(json.dumps(stale_result), encoding="utf-8")

        def fake_run(command, **kwargs):
            if side_effect is not None:
                return side_effect(command, kwargs, json_path)
            if result is not None:
                json_path.write_text(
                    json.dumps(result), encoding="utf-8")
            return subprocess.CompletedProcess(
                command, returncode, stdout="child stdout", stderr="")

        with mock.patch.object(sweep.subprocess, "run", side_effect=fake_run):
            row = sweep.run_case(
                10, json_path, case_id=case_id, random_seed=7)
        return row, json_path

    def test_completed_pass_is_parsed_and_keeps_trace_fields(self):
        row, json_path = self._run_case(make_result("sweep-test-case"))

        self.assertEqual("PASS", row["status"])
        self.assertEqual(0, row["exit_code"])
        self.assertEqual(6, row["schema_version"])
        self.assertEqual("sweep-test-case", row["case_id"])
        self.assertEqual(str(json_path), row["json_path"])
        self.assertEqual(0.64, row["coax_mm"])
        self.assertEqual(1.25, row["feed_residual_mm"])
        self.assertEqual("standard", row["diagnostic_mode"])

    def test_expected_simulation_fail_and_abort_are_not_runtime_errors(self):
        failed = make_result(
            "sweep-test-case", status="completed", passed=False)
        fail_row, _ = self._run_case(failed, returncode=1)
        self.assertEqual("FAIL", fail_row["status"])
        self.assertEqual("quality_fail", fail_row["outcome_class"])

        aborted = make_result("sweep-test-case", status="aborted")
        abort_row, _ = self._run_case(aborted, returncode=1)
        self.assertEqual("ABORT", abort_row["status"])
        self.assertEqual("no_v_jaw_contact", abort_row["error"])

    def test_missing_result_is_error(self):
        row, _ = self._run_case()

        self.assertEqual("ERROR", row["status"])
        self.assertEqual("missing_result", row["execution_status"])
        self.assertIn("was not created", row["error"])
        self.assertFalse(row["trial_data_valid"])

    def test_preexisting_json_is_not_reused_when_child_writes_nothing(self):
        stale = make_result("sweep-test-case")
        row, _ = self._run_case(stale_result=stale)

        self.assertEqual("ERROR", row["status"])
        self.assertEqual("missing_result", row["execution_status"])
        self.assertEqual(0, row["exit_code"])

    def test_corrupt_json_is_error(self):
        def corrupt(_command, _kwargs, json_path):
            json_path.write_text("{broken", encoding="utf-8")
            return subprocess.CompletedProcess([], 0, "out", "err")

        row, _ = self._run_case(side_effect=corrupt)

        self.assertEqual("ERROR", row["status"])
        self.assertEqual("invalid_result", row["execution_status"])

    def test_old_schema_and_incomplete_v6_are_errors(self):
        old = make_result("sweep-test-case")
        old["schema_version"] = 5
        row, _ = self._run_case(old)
        self.assertEqual("ERROR", row["status"])
        self.assertEqual("unsupported_schema", row["execution_status"])

        non_integer_version = make_result("sweep-test-case")
        non_integer_version["schema_version"] = 6.0
        row, _ = self._run_case(non_integer_version)
        self.assertEqual("ERROR", row["status"])
        self.assertEqual("unsupported_schema", row["execution_status"])

        incomplete = make_result("sweep-test-case")
        del incomplete["metrics"]["stock_ecc_mm"]
        row, _ = self._run_case(incomplete)
        self.assertEqual("ERROR", row["status"])
        self.assertIn("missing metrics", row["error"])

    def test_non_finite_metric_is_error(self):
        result = make_result("sweep-test-case")
        result["metrics"]["coax_mm"] = float("nan")
        row, _ = self._run_case(result)

        self.assertEqual("ERROR", row["status"])
        self.assertIn("non-numeric or non-finite metrics", row["error"])

    def test_launch_failure_and_timeout_are_errors(self):
        def launch_failure(_command, _kwargs, _json_path):
            raise FileNotFoundError("python executable missing")

        row, _ = self._run_case(side_effect=launch_failure)
        self.assertEqual("ERROR", row["status"])
        self.assertEqual("launch_error", row["execution_status"])
        self.assertIn("python executable missing", row["error"])

        def timeout(_command, kwargs, _json_path):
            self.assertEqual(sweep.CASE_TIMEOUT_SECONDS, kwargs["timeout"])
            raise subprocess.TimeoutExpired(
                "feed", kwargs["timeout"], output=b"started",
                stderr=b"still running")

        row, _ = self._run_case(side_effect=timeout)
        self.assertEqual("ERROR", row["status"])
        self.assertEqual("timeout", row["execution_status"])
        self.assertEqual(-1, row["exit_code"])
        self.assertIn("timed out", row["error"])
        self.assertIn("still running", row["error"])

    def test_crash_exit_code_cannot_be_overridden_by_pass_json(self):
        row, _ = self._run_case(
            make_result("sweep-test-case"), returncode=139)

        self.assertEqual("ERROR", row["status"])
        self.assertEqual("exit_code_mismatch", row["execution_status"])
        self.assertIn("conflicts with completed result", row["error"])

    def test_runtime_error_json_is_reported_as_error(self):
        row, _ = self._run_case(
            make_result("sweep-test-case", status="error"), returncode=1)

        self.assertEqual("ERROR", row["status"])
        self.assertEqual("error", row["execution_status"])
        self.assertEqual("non_finite_state", row["error"].splitlines()[0])


if __name__ == "__main__":
    unittest.main()
