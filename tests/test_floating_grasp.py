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


if __name__ == "__main__":
    unittest.main()
