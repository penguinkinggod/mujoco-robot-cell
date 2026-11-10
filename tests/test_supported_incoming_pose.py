import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import diag_feed_chuck as feed


class SupportedIncomingPoseTests(unittest.TestCase):
    def test_cradles_rigidly_follow_requested_translation_and_yaw(self):
        model = feed.build_model(v_jaw=True, lift=True)
        cradle_ids = [
            geom_id for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(
                    "cradle_")
        ]
        shaft_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
        shaft_qadr = model.jnt_qposadr[shaft_joint]
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        center = data.qpos[shaft_qadr:shaft_qadr + 3].copy()
        original_positions = model.geom_pos[cradle_ids].copy()
        original_quaternions = model.geom_quat[cradle_ids].copy()

        count = feed._move_cradles_with_initial_pose(
            model, center, (0.0, 3.0, 2.0), 5.0)

        angle = np.radians(5.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        expected_positions = np.array([
            center + np.array([0.0, 0.003, 0.002])
            + rotation @ (position - center)
            for position in original_positions
        ])
        np.testing.assert_allclose(
            model.geom_pos[cradle_ids], expected_positions, atol=1e-12)
        expected_quaternions = np.array([
            feed._apply_world_yaw_to_quaternion(quaternion, 5.0)
            for quaternion in original_quaternions
        ])
        np.testing.assert_allclose(
            np.abs(np.sum(
                model.geom_quat[cradle_ids] * expected_quaternions,
                axis=1)),
            np.ones(len(cradle_ids)),
            atol=1e-12,
        )
        first_positions = model.geom_pos[cradle_ids].copy()
        first_quaternions = model.geom_quat[cradle_ids].copy()
        feed._move_cradles_with_initial_pose(
            model, center, (0.0, 3.0, 2.0), 5.0)
        np.testing.assert_allclose(
            model.geom_pos[cradle_ids], first_positions, atol=1e-12)
        np.testing.assert_allclose(
            np.abs(np.sum(
                model.geom_quat[cradle_ids] * first_quaternions,
                axis=1)),
            np.ones(len(cradle_ids)),
            atol=1e-12,
        )
        self.assertEqual(len(cradle_ids), count)

    def test_lateral_offset_remains_after_gravity_settle_with_both_cradles(self):
        model = feed.build_model(v_jaw=True, lift=True)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        shaft_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
        shaft_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "shaft")
        shaft_qadr = model.jnt_qposadr[shaft_joint]
        shaft_dadr = model.jnt_dofadr[shaft_joint]
        nominal_center = data.qpos[shaft_qadr:shaft_qadr + 3].copy()
        cradle_ids_by_side = {
            side: {
                geom_id for geom_id in range(model.ngeom)
                if (mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
                .startswith("cradle_%s" % side)
            }
            for side in ("L", "R")
        }
        shaft_geom_ids = {
            geom_id for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "")
            .startswith("shaft_")
        }

        feed._move_cradles_with_initial_pose(
            model, nominal_center, (0.0, 3.0, 0.0), 0.0)
        data.qpos[shaft_qadr:shaft_qadr + 3] += [0.0, 0.003, 0.0]
        mujoco.mj_forward(model, data)
        arm = feed.Arm(model, data)
        jaw_act = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "jaw_act")
        data.ctrl[jaw_act] = feed.SM.CHUCK_OPEN_CTRL
        data.ctrl[arm.grip_act] = feed.GRIP_OPEN

        positions = []
        support_samples = {"L": 0, "R": 0}
        for step in range(600):
            for joint_index in range(5):
                data.ctrl[joint_index] = data.qpos[arm.qadr[joint_index]]
            mujoco.mj_step(model, data)
            if step < 500:
                continue
            positions.append(data.qpos[shaft_qadr:shaft_qadr + 3].copy())
            contacts_by_side = {"L": False, "R": False}
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                first, second = int(contact.geom1), int(contact.geom2)
                if first in shaft_geom_ids:
                    other = second
                elif second in shaft_geom_ids:
                    other = first
                else:
                    continue
                for side, geom_ids in cradle_ids_by_side.items():
                    contacts_by_side[side] |= other in geom_ids
            for side in ("L", "R"):
                support_samples[side] += int(contacts_by_side[side])

        actual_center = data.qpos[shaft_qadr:shaft_qadr + 3].copy()
        displacement_mm = float(
            np.linalg.norm(actual_center - nominal_center) * 1000.0)
        max_center_excursion_mm = float(
            np.max(np.linalg.norm(
                np.asarray(positions) - positions[0], axis=1)) * 1000.0)

        self.assertAlmostEqual(3.0, displacement_mm, delta=0.05)
        self.assertLessEqual(
            max_center_excursion_mm,
            feed.INITIAL_STABILITY_ENDPOINT_EXCURSION_LIMIT_MM)
        self.assertEqual(
            {"L": 100, "R": 100}, support_samples)
        self.assertEqual(0.0, model.body_gravcomp[shaft_body])
        np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -9.81])
        np.testing.assert_allclose(
            data.qfrc_applied[shaft_dadr:shaft_dadr + 6],
            np.zeros(6),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            data.xfrc_applied[shaft_body], np.zeros(6), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
