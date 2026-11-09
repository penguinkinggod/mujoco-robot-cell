import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diag_floating_grasp as floating


class FloatingGraspPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.model, self.data, self.body, self.joint = (
            floating.build_floating_scene())
        mujoco.mj_resetData(self.model, self.data)

    def test_gravity_compensated_part_keeps_pose_without_support(self):
        position = np.array([0.26, -0.13, 0.15])
        floating.set_workpiece_pose(
            self.model, self.data, self.body, self.joint,
            position, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(
            1.0, float(self.model.body_gravcomp[self.body]))
        self.assertTrue(self.model.flg_gravcomp)
        self.assertGreater(self.model.ngravcomp, 0)
        self.assertAlmostEqual(
            floating.F.SHAFT_MASS,
            float(self.model.body_mass[self.body]), places=5)

        qadr = self.model.jnt_qposadr[self.joint]
        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

        np.testing.assert_allclose(
            self.data.qpos[qadr:qadr + 3], position, atol=1e-8)

    def test_world_rpy_changes_the_shaft_axis(self):
        axis = floating.set_workpiece_pose(
            self.model, self.data, self.body, self.joint,
            (0.26, -0.13, 0.15), (0.0, 30.0, 0.0))

        np.testing.assert_allclose(
            [np.cos(np.radians(30.0)), 0.0,
             -np.sin(np.radians(30.0))],
            axis,
            atol=1e-6,
        )

    def test_free_body_moves_when_given_an_impulse(self):
        floating.set_workpiece_pose(
            self.model, self.data, self.body, self.joint,
            (0.26, -0.13, 0.15), (0.0, 0.0, 0.0))
        qadr = self.model.jnt_qposadr[self.joint]
        dadr = self.model.jnt_dofadr[self.joint]
        self.data.qvel[dadr + 1] = 0.1
        start_y = float(self.data.qpos[qadr + 1])
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

        self.assertGreater(
            abs(float(self.data.qpos[qadr + 1]) - start_y), 1e-4)

    def test_table_intersection_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "intersects the table"):
            floating.set_workpiece_pose(
                self.model, self.data, self.body, self.joint,
                (0.26, -0.13, 0.01), (0.0, 0.0, 0.0))

    def test_contact_snapshot_names_table_contact(self):
        qadr = self.model.jnt_qposadr[self.joint]
        self.data.qpos[qadr:qadr + 3] = (
            0.26, -0.13, floating.F.TABLE_TOP_Z + 0.009)
        self.data.qpos[qadr + 3:qadr + 7] = (
            floating.pose_quaternion_world_xyz((0.0, 0.0, 0.0)))
        mujoco.mj_forward(self.model, self.data)

        pad_count, pad_pairs, contacts = (
            floating._workpiece_contact_snapshot(
                self.model, self.data, floating._shaft_geom_ids(self.model),
                set()))

        self.assertEqual(0, pad_count)
        self.assertEqual([], pad_pairs)
        self.assertTrue(any(
            contact["other_geom"] == "table"
            and contact["other_body"] == "world"
            for contact in contacts))
        self.assertTrue(all(
            len(contact["position_world_m"]) == 3
            for contact in contacts))


class FloatingGraspEvaluationTests(unittest.TestCase):
    def evaluate(self, **overrides):
        values = {
            "close_pad_contact_count": 2,
            "close_pad_sides": {"fixed", "moving"},
            "hold_pad_contact_count": 2,
            "hold_pad_sides": {"fixed", "moving"},
            "hold_opposing_contact_steps": floating.GRASP_HOLD_STEPS,
            "hold_sample_count": floating.GRASP_HOLD_STEPS,
            "workpiece_lift_m": floating.GRASP_LIFT_DISTANCE_M * 0.8,
            "max_center_drift_m": (
                floating.GRASP_RELATIVE_POSITION_LIMIT_M),
            "max_axis_drift_rad": (
                floating.GRASP_RELATIVE_AXIS_LIMIT_RAD),
            "lift_executed": True,
        }
        values.update(overrides)
        return floating._evaluate_grasp(**values)

    def test_grasp_passes_at_inclusive_provisional_limits(self):
        result = self.evaluate()

        self.assertTrue(result["passed"])
        self.assertEqual([], result["failure_reasons"])
        self.assertEqual(
            "grasp_hold_pass_gravity_compensated", result["outcome"])

    def test_one_sided_contact_does_not_count_as_a_grasp(self):
        result = self.evaluate(
            close_pad_contact_count=3,
            close_pad_sides={"moving"})

        self.assertFalse(result["checks"]["opposing_jaw_contact"])
        self.assertEqual("no_opposing_jaw_contact", result["outcome"])

    def test_opposing_contact_must_persist_for_the_entire_hold(self):
        result = self.evaluate(hold_opposing_contact_steps=(
            floating.GRASP_HOLD_STEPS - 1))

        self.assertFalse(
            result["checks"]["contact_retained_during_hold"])
        self.assertEqual("contact_lost_during_lift", result["outcome"])

    def test_lift_below_provisional_minimum_fails(self):
        result = self.evaluate(workpiece_lift_m=0.0239)

        self.assertFalse(result["checks"]["minimum_workpiece_lift"])
        self.assertEqual("insufficient_lift", result["outcome"])

    def test_relative_center_drift_above_limit_fails(self):
        result = self.evaluate(max_center_drift_m=0.0021)

        self.assertFalse(result["checks"]["relative_center_stable"])
        self.assertEqual(
            "workpiece_slipped_relative_to_gripper", result["outcome"])

    def test_unexecuted_lift_is_not_a_pass(self):
        result = self.evaluate(lift_executed=False)

        self.assertFalse(result["checks"]["lift_executed"])
        self.assertEqual("lift_not_executed", result["outcome"])


if __name__ == "__main__":
    unittest.main()
