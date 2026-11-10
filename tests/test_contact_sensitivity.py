import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diag_feed_chuck as feed
from src.diag_contact_sensitivity import summarize_result
from src.diag_relative_motion_sensitivity import (
    summarize_result as summarize_relative_motion_result,
)


def _pair(solref_time_s, depth_mm, overlap_steps, contact_steps):
    return {
        "max_penetration_mm": depth_mm,
        "overlap_steps": overlap_steps,
        "contact_steps": contact_steps,
        "contact_geometry_parameters": [
            {"name": "shaft_geom", "solref": [solref_time_s, 1.0]},
            {"name": "moving_jaw_pad_v2", "solref": [solref_time_s, 1.0]},
        ],
    }


class ContactSensitivityTests(unittest.TestCase):
    def _result(self, solref_time_s=0.005):
        return {
            "schema_version": 14,
            "case_id": "test-solref",
            "execution_status": "completed",
            "control_mode": {
                "carry": "unlocked_physics",
                "planning_mode": "M1",
            },
            "inputs": {"initial_settle_steps": 600},
            "config": {
                "v_jaw": True,
                "lift": True,
                "carry_lock": False,
                "initial_offset_mm": [0.0, 0.0, 0.0],
                "initial_yaw_world_z_deg": 0.0,
                "planning_mode": "M1",
            },
            "environment_overrides": {
                "FEED_V_CONTACT_SOLREF_TIME": str(solref_time_s),
            },
            "penetration_diagnostics": {
                "by_phase": {
                    "搬运": {
                        "shaft_geom|moving_jaw_pad_v2": _pair(
                            solref_time_s, 0.42, 12, 20),
                    },
                    "三爪稳定": {
                        "shaft_geom|jaw_geom_1_02": {
                            "max_penetration_mm": 0.81,
                            "overlap_steps": 50,
                            "contact_steps": 60,
                        },
                    },
                },
            },
            "contact_diagnostics": {
                "scopes": {
                    "feed": {
                        "overall": {
                            "pairs": {
                                "robot_workpiece|moving_jaw_pad_v2|shaft_geom": {
                                    "category": "robot_workpiece",
                                    "peak_normal_force_per_contact_N": 18.5,
                                },
                                "workpiece_machine|jaw_geom_1_02|shaft_geom": {
                                    "category": "workpiece_machine",
                                    "peak_normal_force_per_contact_N": 14.2,
                                },
                            },
                        },
                    },
                },
            },
            "metrics": {
                "coax_mm": 0.55,
                "tilt_deg": 1.5,
                "contacts": 30,
                "slip_mm": 0.0,
                "pull_mm": 0.0,
                "jaw_force_N": 166.4,
                "clamp_swing_mm": 2.95,
                "stock_ecc_mm": 1.42,
            },
            "hard_pass": True,
            "quality_pass": True,
            "overall_pass": True,
        }

    def test_summarizes_vjaw_and_chuck_separately(self):
        row = summarize_result(
            self._result(), 0.005, "case.json")

        self.assertEqual("PASS", row["status"])
        self.assertEqual(
            "shaft_geom|moving_jaw_pad_v2", row["v_jaw_pair"])
        self.assertEqual(0.42, row["v_jaw_max_penetration_mm"])
        self.assertEqual(12, row["v_jaw_overlap_steps"])
        self.assertEqual(18.5, row["v_jaw_peak_normal_force_N"])
        self.assertEqual("shaft_geom|jaw_geom_1_02", row["chuck_pair"])
        self.assertEqual("三爪稳定", row["chuck_phase"])
        self.assertEqual(0.81, row["chuck_max_penetration_mm"])
        self.assertEqual(14.2, row["chuck_peak_normal_force_N"])

    def test_rejects_mismatched_contact_parameter(self):
        result = self._result()
        result["penetration_diagnostics"]["by_phase"]["搬运"][
            "shaft_geom|moving_jaw_pad_v2"] = _pair(
                0.020, 2.9, 680, 680)

        with self.assertRaisesRegex(ValueError, "requested solref"):
            summarize_result(result, 0.005, "case.json")

    def test_rejects_non_nominal_run(self):
        result = self._result()
        result["config"]["carry_lock"] = True

        with self.assertRaisesRegex(ValueError, "frozen nominal"):
            summarize_result(result, 0.005, "case.json")

    def test_contact_parameter_override_only_changes_five_target_geoms(self):
        names = [
            "shaft_geom",
            "fixed_jaw_pad_v1",
            "fixed_jaw_pad_v2",
            "moving_jaw_pad_v1",
            "moving_jaw_pad_v2",
            "jaw_geom_0_00",
        ]
        model = SimpleNamespace(
            ngeom=len(names),
            geom_solref=np.tile([0.020, 1.0], (len(names), 1)),
        )
        with mock.patch.object(
                feed.mujoco, "mj_id2name",
                side_effect=lambda _model, _kind, geom_id: names[geom_id]):
            changed = feed._apply_v_contact_solref_time(model, 0.005)

        self.assertEqual(tuple(sorted(names[:5])), changed)
        np.testing.assert_allclose(model.geom_solref[:5, 0], 0.005)
        self.assertEqual(0.020, model.geom_solref[5, 0])

    def test_contact_parameter_override_rejects_nonpositive_value(self):
        model = SimpleNamespace(ngeom=0, geom_solref=np.zeros((0, 2)))

        with self.assertRaisesRegex(ValueError, "positive finite"):
            feed._apply_v_contact_solref_time(model, 0.0)

    def test_positive_finite_environment_float(self):
        with mock.patch.dict("os.environ", {"TEST_FEED_VALUE": "0.275"}):
            self.assertEqual(
                0.275, feed._positive_finite_env_float(
                    "TEST_FEED_VALUE", 0.55))
        with mock.patch.dict("os.environ", {"TEST_FEED_VALUE": "nan"}):
            with self.assertRaisesRegex(ValueError, "positive finite"):
                feed._positive_finite_env_float(
                    "TEST_FEED_VALUE", 0.55)

    def test_relative_motion_summary_checks_requested_factor(self):
        result = {
            "schema_version": 14,
            "execution_status": "completed",
            "control_mode": {
                "carry": "unlocked_physics",
                "planning_mode": "M1",
            },
            "inputs": {"initial_settle_steps": 600},
            "config": {
                "v_jaw": True,
                "lift": True,
                "carry_lock": False,
                "initial_offset_mm": [0.0, 0.0, 0.0],
                "initial_yaw_world_z_deg": 0.0,
                "planning_mode": "M1",
            },
            "environment_overrides": {
                "FEED_CARRY_V_CRUISE": "0.275",
            },
            "effective_parameters": {
                "controller": {"carry_v_cruise_m_s": 0.275},
                "gripper": {"v_length_mm": 24.0},
            },
            "gripper_relative_workpiece_motion": {
                "status": "observed",
                "time_history_enabled": True,
                "phases": {
                    "搬运": {"max_relative_axis_change_deg": 0.8},
                },
                "time_history": [
                    {
                        "sim_time_s": 1.0,
                        "phase_group": "搬运",
                        "translation_change_from_carry_start_mm": 1.0,
                        "translation_change_from_carry_start_tool_mm": [
                            1.0, 0.0, 0.0],
                        "translation_change_since_previous_step_mm": 0.1,
                        "shaft_contact_pairs": [],
                    },
                    {
                        "sim_time_s": 1.2,
                        "phase_group": "搬运",
                        "translation_change_from_carry_start_mm": 2.0,
                        "translation_change_from_carry_start_tool_mm": [
                            2.0, 0.0, 0.0],
                        "translation_change_since_previous_step_mm": 0.2,
                        "shaft_contact_pairs": [
                            {
                                "category": "robot_workpiece",
                                "geoms": [
                                    "moving_jaw_pad_v2", "shaft_geom"],
                                "contact_count": 2,
                                "normal_force_sum_N": 18.0,
                                "minimum_distance_mm": -0.4,
                            },
                        ],
                    },
                    {
                        "sim_time_s": 2.0,
                        "phase_group": "进给",
                        "translation_change_from_carry_start_mm": 1.5,
                        "translation_change_from_carry_start_tool_mm": [
                            1.5, 0.0, 0.0],
                        "translation_change_since_previous_step_mm": 0.1,
                        "shaft_contact_pairs": [],
                    },
                ],
            },
            "metrics": {
                "coax_mm": 0.55,
                "tilt_deg": 1.5,
                "stock_ecc_mm": 1.42,
                "contacts": 30,
            },
            "overall_pass": True,
        }
        row = summarize_relative_motion_result(
            result, "slow_carry", "carry_speed", 0.275,
            "case.json", "case.log")
        self.assertEqual(2.0, row["carry_peak_mm"])
        self.assertEqual(1.2, row["carry_peak_time_s"])
        self.assertAlmostEqual(0.2, row["carry_duration_s"])
        self.assertEqual(2, row["peak_vjaw_contacts"])
        self.assertEqual(-0.4, row["peak_vjaw_min_distance_mm"])
        self.assertEqual(18.0, row["peak_vjaw_normal_force_sum_N"])
        self.assertEqual(1.5, row["feed_peak_mm"])

        result["effective_parameters"]["controller"][
            "carry_v_cruise_m_s"] = 0.55
        with self.assertRaisesRegex(ValueError, "carry speed"):
            summarize_relative_motion_result(
                result, "slow_carry", "carry_speed", 0.275,
                "case.json", "case.log")


if __name__ == "__main__":
    unittest.main()
