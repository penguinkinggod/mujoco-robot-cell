import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for key in tuple(os.environ):
    if key.startswith("FEED_") or key in ("GRIP_RAMP", "SMOOTH"):
        os.environ.pop(key, None)

import diag_feed_chuck as feed


class WorldYawConventionTests(unittest.TestCase):
    def test_world_yaw_rotates_shaft_axis_about_world_z(self):
        quat = feed._apply_world_yaw_to_quaternion(feed.SHAFT_Q0, 30.0)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quat)
        axis = rotation.reshape(3, 3)[:, 2]

        np.testing.assert_allclose(
            [np.cos(np.radians(30.0)),
             np.sin(np.radians(30.0)), 0.0],
            axis,
            atol=1e-6,
        )

    def test_legacy_roll_argument_is_yaw_alias(self):
        self.assertEqual(
            5.0, feed._resolve_initial_yaw_deg(5.0, None))
        self.assertEqual(
            5.0, feed._resolve_initial_yaw_deg(None, 5.0))
        with self.assertRaises(ValueError):
            feed._resolve_initial_yaw_deg(5.0, -5.0)


class ToolFrameMotionTests(unittest.TestCase):
    def test_rigid_tool_rotation_is_not_reported_as_slip(self):
        angle = np.radians(73.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        offset_tool = np.array([0.012, -0.021, 0.034])
        axis_tool = feed._normalise_axis([1.0, 0.2, -0.1])
        tool_start = np.array([0.2, -0.1, 0.05])
        tool_end = np.array([-0.3, 0.4, 0.8])

        reference_offset, reference_axis = feed._pose_in_tool_frame(
            np.eye(3), tool_start + offset_tool, tool_start, axis_tool)
        current_offset, current_axis = feed._pose_in_tool_frame(
            rotation,
            tool_end + rotation @ offset_tool,
            tool_end,
            rotation @ axis_tool,
        )
        translation_delta, axis_change_deg = feed._relative_pose_change(
            reference_offset, reference_axis, current_offset, current_axis)

        np.testing.assert_allclose(offset_tool, reference_offset, atol=1e-12)
        np.testing.assert_allclose(offset_tool, current_offset, atol=1e-12)
        np.testing.assert_allclose(
            np.zeros(3), translation_delta, atol=1e-12)
        self.assertAlmostEqual(0.0, axis_change_deg, places=5)

    def test_local_translation_change_is_measured_in_tool_coordinates(self):
        angle = np.radians(41.0)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        tool_origin = np.array([0.2, 0.1, -0.4])
        offset_tool = np.array([0.01, 0.02, -0.03])
        expected_slip_tool = np.array([0.0, 0.004, 0.0])
        axis_tool = np.array([1.0, 0.0, 0.0])
        reference_offset, reference_axis = feed._pose_in_tool_frame(
            np.eye(3), offset_tool, np.zeros(3), axis_tool)
        current_offset, current_axis = feed._pose_in_tool_frame(
            rotation,
            tool_origin + rotation @ (offset_tool + expected_slip_tool),
            tool_origin,
            rotation @ axis_tool,
        )
        translation_delta, axis_change_deg = feed._relative_pose_change(
            reference_offset, reference_axis, current_offset, current_axis)

        np.testing.assert_allclose(
            expected_slip_tool, translation_delta, atol=1e-12)
        self.assertAlmostEqual(4.0, np.linalg.norm(translation_delta) * 1000)
        self.assertAlmostEqual(0.0, axis_change_deg, places=5)


class PlanningModeIsolationTests(unittest.TestCase):
    def setUp(self):
        self.true_pose = {
            "position_m": [0.26, -0.13, 0.04],
            "quaternion_wxyz": [
                0.7071067812, 0.0, 0.7071067812, 0.0],
        }
        self.nominal_position = [0.26, -0.13, 0.039812104]

    def test_m0_does_not_consume_per_case_truth(self):
        first = feed._resolve_planning_pose(
            "M0", self.true_pose, self.nominal_position, feed.SHAFT_Q0)
        second = feed._resolve_planning_pose(
            "M0", {"invalid": "truth is not an input"},
            self.nominal_position, feed.SHAFT_Q0)

        np.testing.assert_allclose(first["position_m"], second["position_m"])
        np.testing.assert_allclose(
            first["shaft_axis_world_unit"],
            second["shaft_axis_world_unit"])
        self.assertEqual(
            "fixed_nominal_settled_workpiece_center",
            first["position_source"])

    def test_m1_uses_the_settled_true_pose(self):
        result = feed._resolve_planning_pose(
            "M1", self.true_pose, self.nominal_position, feed.SHAFT_Q0)

        np.testing.assert_allclose(
            self.true_pose["position_m"], result["position_m"])
        self.assertEqual(
            "settled_true_workpiece_center", result["position_source"])

    def test_m2_uses_true_pose_plus_declared_estimation_error(self):
        result = feed._resolve_planning_pose(
            "M2", self.true_pose, self.nominal_position, feed.SHAFT_Q0,
            m2_error_mm=(0.0, 0.5, -0.25), m2_error_yaw_deg=2.0)

        np.testing.assert_allclose(
            [0.26, -0.1295, 0.03975], result["position_m"])
        self.assertAlmostEqual(2.0, result["m2_error_used_yaw_deg"])
        self.assertEqual(
            "estimated_workpiece_center", result["position_source"])

    def test_m2_rejects_nonfinite_error(self):
        with self.assertRaises(ValueError):
            feed._resolve_planning_pose(
                "M2", self.true_pose, self.nominal_position,
                feed.SHAFT_Q0, m2_error_mm=(0.0, float("nan"), 0.0))


def assert_reproducibility_fields(test_case, result):
    test_case.assertEqual(12, result["schema_version"])
    test_case.assertIn("pickup_motion_diagnostics", result)
    test_case.assertIn("gripper_relative_workpiece_motion", result)
    test_case.assertEqual(1, result["metric_schema_version"])
    test_case.assertEqual("feed-criteria-v1", result["criterion_set_id"])
    test_case.assertTrue(result["case_id"])
    test_case.assertIn(result["execution_status"],
                       ("completed", "aborted", "error"))
    test_case.assertIn("stages", result)
    test_case.assertIn("checkpoints", result)
    test_case.assertIn("contact_diagnostics", result)
    test_case.assertIn("applied_load_diagnostics", result)
    test_case.assertIn("penetration_diagnostics", result)
    test_case.assertIn("initial_stability", result)
    test_case.assertTrue(
        result["contact_diagnostics"]["sampled_every_physics_step"])
    test_case.assertTrue(
        result["applied_load_diagnostics"]["sampled_every_physics_step"])
    test_case.assertIn("hard_failures", result)
    test_case.assertIn("quality_failures", result)
    test_case.assertEqual(result["git_revision"],
                          result["versions"]["git_revision"])
    test_case.assertIsInstance(result["git_worktree_dirty"], bool)
    test_case.assertIn("workpiece", result["inputs"])
    test_case.assertEqual(20.0, result["inputs"]["workpiece"]["diameter_mm"])
    test_case.assertEqual(80.0, result["inputs"]["workpiece"]["length_mm"])
    test_case.assertIn("requested_initial_pose", result["inputs"])
    test_case.assertIn("initial_settle_steps", result["inputs"])
    test_case.assertIn(
        "offset_world_xyz_mm",
        result["inputs"]["requested_initial_pose"])
    test_case.assertIn(
        "yaw_world_z_deg",
        result["inputs"]["requested_initial_pose"])
    test_case.assertIn("pose_before_perturbation", result["inputs"])
    test_case.assertIn(
        "pose_after_perturbation_before_settle", result["inputs"])
    test_case.assertIn("grasp_target_pose", result["inputs"])
    test_case.assertIn("v_jaw", result["control_mode"])
    test_case.assertIn("feed", result["effective_parameters"])
    test_case.assertIn("bushing", result["effective_parameters"])
    test_case.assertIn("timestep_s", result["model_options"]["initial"])
    test_case.assertIn("timestep_s", result["model_options"]["at_result"])
    test_case.assertIn("hard", result["criteria"])
    test_case.assertIn("quality", result["criteria"])
    if result["actual_initial_pose"] is not None:
        test_case.assertEqual(
            3, len(result["actual_initial_pose"]["position_m"]))
        test_case.assertEqual(
            4, len(result["actual_initial_pose"]["quaternion_wxyz"]))
        test_case.assertEqual(
            3, len(result["actual_initial_pose"]["shaft_axis_world_unit"]))


def assert_feed_checkpoints(test_case, result, expected_status=None):
    checkpoints = result["checkpoints"]
    expected_ids = (
        "feed_complete",
        "chuck_clamped",
        "gripper_released_in_place",
        "axial_retreat_complete",
        "lateral_retreat_complete",
        "unlock_settled",
    )
    test_case.assertEqual(expected_ids, tuple(
        row["checkpoint_id"] for row in checkpoints))
    for order, row in enumerate(checkpoints, start=1):
        test_case.assertEqual(order, row["checkpoint_order"])
        test_case.assertTrue(row["checkpoint_label"])
        test_case.assertIn(
            row["checkpoint_status"], ("completed", "skipped"))
        test_case.assertIsInstance(row["sim_time_s"], float)
        test_case.assertEqual(3, len(row["shaft_position_m"]))
        test_case.assertEqual(3, len(row["shaft_axis"]))
        test_case.assertIn("coax_mm", row)
        test_case.assertIn("stock_ecc_mm", row)
        test_case.assertIn("jaw_contact_count", row)
        test_case.assertIn("contact_diagnostics", row)
        test_case.assertIn("applied_loads", row)
        test_case.assertEqual("feed", row["diagnostic_scope"])
    if expected_status is not None:
        test_case.assertEqual(
            expected_status,
            tuple(row["checkpoint_status"] for row in checkpoints))


class ExperimentEnvironmentCaptureTests(unittest.TestCase):
    def test_captures_physics_overrides_not_run_bookkeeping(self):
        captured = feed._capture_environment_overrides({
            "FEED_INSERT_MM": "25",
            "FEED_BUSHING": "0",
            "FEED_CASE_ID": "case-1",
            "FEED_RANDOM_SEED": "19",
            "FEED_RESULT_JSON": "_build/case-1.json",
            "SO101_KP": "70",
            "SMOOTH": "0",
            "GRIP_RAMP": "400",
            "UNRELATED_SETTING": "ignored",
        })
        self.assertEqual({
            "FEED_INSERT_MM": "25",
            "FEED_BUSHING": "0",
            "SO101_KP": "70",
            "SMOOTH": "0",
            "GRIP_RAMP": "400",
        }, captured)

    def test_retreat_diagnostics_distinguish_skipped_actions(self):
        self.assertEqual({
            "axial_retreat_complete": "completed",
            "lateral_retreat_complete": "completed",
        }, feed._feed_checkpoint_statuses(True, False))
        self.assertEqual({
            "axial_retreat_complete": "skipped",
            "lateral_retreat_complete": "completed",
        }, feed._feed_checkpoint_statuses(False, False))
        self.assertEqual({
            "axial_retreat_complete": "skipped",
            "lateral_retreat_complete": "skipped",
        }, feed._feed_checkpoint_statuses(True, True))


class RobotMachineContactTests(unittest.TestCase):
    def _build_model(self):
        model = feed.build_model(v_jaw=True, lift=True, machine_solid=True)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        return model, data

    def test_normal_initial_pose_has_no_robot_machine_contact(self):
        model, data = self._build_model()
        robot_geoms, machine_geoms = feed._robot_machine_geom_ids(model)

        hits = [
            (int(data.contact[i].geom1), int(data.contact[i].geom2))
            for i in range(data.ncon)
            if feed._is_robot_machine_contact(
                int(data.contact[i].geom1), int(data.contact[i].geom2),
                robot_geoms, machine_geoms)
        ]
        self.assertEqual([], hits)

    def test_semantic_categories_do_not_depend_on_collision_masks(self):
        model, _data = self._build_model()
        robot_geoms, _machine_geoms = feed._robot_machine_geom_ids(model)
        shaft_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
        jaw_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "jaw_geom_0_00")
        table_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table")
        bushing_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "bushing_06")
        robot_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_v1")

        # Deliberately make mask values uninformative: role classification
        # must continue to come from model identity and robot ancestry.
        model.geom_contype[[shaft_geom, jaw_geom, table_geom, bushing_geom]] = 0
        sets = feed._semantic_contact_geom_sets(model, robot_geoms)

        self.assertEqual(
            "robot_workpiece",
            feed._contact_category(robot_geom, shaft_geom, sets))
        self.assertEqual(
            "workpiece_machine",
            feed._contact_category(shaft_geom, jaw_geom, sets))
        self.assertEqual(
            "workpiece_fixture",
            feed._contact_category(shaft_geom, bushing_geom, sets))
        self.assertEqual(
            "robot_fixture",
            feed._contact_category(robot_geom, table_geom, sets))

    def test_applied_load_snapshot_reports_force_and_torque_units(self):
        model, data = self._build_model()
        shaft_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "shaft_free")
        shaft_dof = int(model.jnt_dofadr[shaft_joint])
        data.qfrc_applied[shaft_dof] = -2.5
        data.qfrc_applied[shaft_dof + 3] = 0.75
        shaft_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "shaft")
        data.xfrc_applied[shaft_body, :3] = [-20.0, 0.0, 0.0]

        snapshot = feed._applied_load_snapshot(model, data)

        self.assertEqual(
            ["N", "N*m"],
            [load["unit"] for load in snapshot["qfrc_applied"]])
        self.assertEqual([-2.5, 0.75], [
            load["value"] for load in snapshot["qfrc_applied"]])
        self.assertEqual(20.0, snapshot["xfrc_applied"][0][
            "force_magnitude_N"])
        rollup = feed._new_load_rollup()
        feed._accumulate_load_rollup(rollup, snapshot, 0.002)
        self.assertEqual(2.5, rollup["qfrc_by_dof"]["shaft_free:dof0"][
            "peak_abs"])

    def test_collision_between_robot_shell_and_machine_is_detected(self):
        model, data = self._build_model()
        robot_geoms, machine_geoms = feed._robot_machine_geom_ids(model)
        machine_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "lathe_headstock_2")
        shoulder_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "shoulder")
        shoulder_shells = {
            geom_id for geom_id in robot_geoms
            if int(model.geom_bodyid[geom_id]) == shoulder_id
            and int(model.geom_contype[geom_id]) == feed.F.CT_ROBOT_SHELL[0]
        }
        self.assertTrue(shoulder_shells)
        robot_shell = min(shoulder_shells)

        model.geom_pos[machine_geom] = data.geom_xpos[robot_shell].copy()
        model.geom_size[machine_geom] = [0.12, 0.12, 0.12]
        mujoco.mj_setConst(model, data)
        mujoco.mj_forward(model, data)

        pairs = [
            (int(data.contact[i].geom1), int(data.contact[i].geom2))
            for i in range(data.ncon)
            if feed._is_robot_machine_contact(
                int(data.contact[i].geom1), int(data.contact[i].geom2),
                robot_geoms, machine_geoms)
        ]
        self.assertTrue(pairs)
        self.assertTrue(any(machine_geom in pair
                            and bool(shoulder_shells.intersection(pair))
                            for pair in pairs))
        self.assertEqual(8, int(model.geom_contype[robot_shell]))
        self.assertEqual(8, int(model.geom_contype[machine_geom]))

        shaft_geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "shaft_geom")
        self.assertFalse(feed._is_robot_machine_contact(
            shaft_geom, machine_geom, robot_geoms, machine_geoms))

    def test_full_feed_reports_injected_collision_as_hard_failure(self):
        model, data = self._build_model()
        stage_count = [0]
        injected = [False]

        def inject_collision(tag, current_model, current_data):
            if tag is None or injected[0]:
                return
            stage_count[0] += 1
            if stage_count[0] != 9:
                return

            machine_geom = mujoco.mj_name2id(
                current_model, mujoco.mjtObj.mjOBJ_GEOM,
                "lathe_headstock_2")
            robot_geoms, _ = feed._robot_machine_geom_ids(current_model)
            robot_shell = min(
                geom_id for geom_id in robot_geoms
                if int(current_model.geom_contype[geom_id])
                == feed.F.CT_ROBOT_SHELL[0])
            current_model.geom_pos[machine_geom] = (
                current_data.geom_xpos[robot_shell].copy())
            current_model.geom_size[machine_geom] = [0.01, 0.01, 0.01]
            mujoco.mj_setConst(current_model, current_data)
            injected[0] = True

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "collision.json"
            previous_path = os.environ.get("FEED_RESULT_JSON")
            os.environ["FEED_RESULT_JSON"] = str(result_path)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = feed.main(
                        frame_hook=inject_collision,
                        model=model,
                        data=data,
                        v_jaw=True,
                        lift=True,
                        carry_lock=False,
                    )
            finally:
                if previous_path is None:
                    os.environ.pop("FEED_RESULT_JSON", None)
                else:
                    os.environ["FEED_RESULT_JSON"] = previous_path

            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertTrue(injected[0])
        self.assertEqual(1, exit_code)
        assert_reproducibility_fields(self, result)
        assert_feed_checkpoints(
            self, result, ("completed",) * 6)
        self.assertEqual("completed", result["execution_status"])
        self.assertGreater(
            result["metrics"]["robot_machine_contact_max"], 0)
        self.assertIn("robot_machine_collision", result["hard_failures"])
        self.assertEqual("hard_fail", result["outcome_class"])
        feed_contacts = result["contact_diagnostics"]["scopes"]["feed"]
        self.assertGreater(feed_contacts["overall"]["samples"], 10000)
        self.assertGreater(
            feed_contacts["overall"]["categories"]["robot_machine"][
                "peak_contacts_per_step"], 0)
        load_scopes = result["applied_load_diagnostics"]["scopes"]
        self.assertNotIn("shaft", load_scopes["feed"]["overall"]["xfrc_by_body"])
        pull_load = load_scopes["pull_test"]["overall"]["xfrc_by_body"]["shaft"]
        self.assertAlmostEqual(20.0, pull_load["force_peak_N"], places=6)
        self.assertEqual(700, pull_load["nonzero_samples"])
        self.assertGreater(
            feed_contacts["overall"]["categories"]["robot_workpiece"][
                "peak_contacts_per_step"], 0)
        self.assertGreater(
            feed_contacts["overall"]["categories"]["workpiece_machine"][
                "peak_contacts_per_step"], 0)
        self.assertTrue(feed_contacts["overall"]["pairs"])
        pull_stage = next(
            row for row in result["stages"]
            if row["stage"] == "after_20N_pull_test")
        self.assertEqual("pull_test", pull_stage["diagnostic_scope"])
        self.assertEqual(
            20.0,
            pull_stage["applied_loads"]["xfrc_applied"][0][
                "force_magnitude_N"])

    def test_full_physics_records_contacts_and_separates_pull_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "contact_loads.json"
            previous_path = os.environ.get("FEED_RESULT_JSON")
            previous_case_id = os.environ.get("FEED_CASE_ID")
            os.environ["FEED_RESULT_JSON"] = str(result_path)
            os.environ["FEED_CASE_ID"] = "test-contact-load-records"
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = feed.main(
                        v_jaw=True, lift=True, carry_lock=False)
            finally:
                if previous_path is None:
                    os.environ.pop("FEED_RESULT_JSON", None)
                else:
                    os.environ["FEED_RESULT_JSON"] = previous_path
                if previous_case_id is None:
                    os.environ.pop("FEED_CASE_ID", None)
                else:
                    os.environ["FEED_CASE_ID"] = previous_case_id
            result = json.loads(result_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertTrue(result["overall_pass"])
        self.assertEqual("physical_success", result["outcome_class"])
        self.assertEqual([], result["hard_failures"])
        self.assertEqual([], result["quality_failures"])
        assert_reproducibility_fields(self, result)
        pickup_motion = result["pickup_motion_diagnostics"]
        self.assertEqual("observed", pickup_motion["status"])
        self.assertGreater(pickup_motion["overall"]["samples"], 0)
        self.assertGreaterEqual(
            pickup_motion["overall"]["max_center_displacement_mm"], 0.0)
        self.assertEqual(
            "diagnostic_only_no_pass_fail_threshold",
            pickup_motion["criterion"])
        relative_motion = result["gripper_relative_workpiece_motion"]
        self.assertEqual("observed", relative_motion["status"])
        self.assertFalse(relative_motion["active"])
        self.assertEqual(
            "feed_complete", relative_motion["observation_end_stage"])
        self.assertGreater(relative_motion["overall"]["samples"], 0)
        self.assertGreaterEqual(
            relative_motion["overall"]["max_translation_slip_mm"], 0.0)
        self.assertIn("搬运", relative_motion["phases"])
        self.assertIn("进给", relative_motion["phases"])
        self.assertEqual(
            "diagnostic_only_no_pass_fail_threshold",
            relative_motion["criterion"])
        self.assertEqual(
            0.0,
            result["inputs"]["requested_initial_pose"]["yaw_world_z_deg"])
        self.assertEqual(
            "settled_true_workpiece_center",
            result["inputs"]["grasp_target_pose"]["position_source"])
        self.assertEqual(
            "settled_true_workpiece_pose",
            result["inputs"]["grasp_target_pose"]["orientation_source"])
        workpiece = result["inputs"]["workpiece"]
        self.assertEqual(
            "reset_model_pose_before_requested_perturbation",
            workpiece["nominal_start_position_source"])
        np.testing.assert_allclose(
            result["inputs"]["pose_before_perturbation"]["position_m"],
            workpiece["nominal_start_position_m"],
            atol=1e-12,
        )
        self.assertNotEqual(
            list(feed.SM.SHAFT_POS),
            workpiece["nominal_start_position_m"])
        stability = result["initial_stability"]
        self.assertEqual("evaluated", stability["status"])
        self.assertEqual(100, stability["sample_count"])
        self.assertEqual(100, len(stability["samples"]))
        self.assertTrue(stability["stability_pass"])
        self.assertLessEqual(
            stability["max_endpoint_excursion_mm"],
            stability["endpoint_excursion_limit_mm"])
        self.assertEqual(
            {"L": True, "R": True},
            stability["support_contact_continuous_by_side"])
        self.assertEqual(
            100, stability["support_contact_samples_by_side"]["L"])
        self.assertEqual(
            100, stability["support_contact_samples_by_side"]["R"])
        self.assertEqual(0.0, stability["workpiece_gravcomp"])
        self.assertTrue(stability["no_external_applied_loads_on_workpiece"])
        self.assertEqual([0.0, 0.0, -9.81], stability["gravity_m_s2"])
        for sample in stability["samples"]:
            self.assertEqual(
                [0.0, 0.0, 0.0], sample["qfrc_applied_force_N"])
            self.assertEqual(
                [0.0, 0.0, 0.0], sample["qfrc_applied_torque_Nm"])
            self.assertEqual(
                [0.0, 0.0, 0.0], sample["xfrc_applied_force_N"])
            self.assertEqual(
                [0.0, 0.0, 0.0], sample["xfrc_applied_torque_Nm"])
        penetration = result["penetration_diagnostics"]
        minimum_contact = penetration["minimum_contact"]
        self.assertIsNotNone(minimum_contact)
        self.assertLess(minimum_contact["distance_m"], 0.0)
        self.assertTrue(minimum_contact["time_s"] > 0.0)
        self.assertTrue(minimum_contact["contact_position_world_m"])
        self.assertIn("normal_force_N", minimum_contact)
        self.assertEqual(
            2,
            len(minimum_contact["contact_geometry_parameters"]))
        self.assertIn(
            "solref",
            minimum_contact["contact_geometry_parameters"][0])
        self.assertIn(
            "contact_position_local_m", minimum_contact["other_geom"])
        self.assertIn("other_geom", minimum_contact)
        carry_penetration = penetration["by_phase"]["搬运"][
            "shaft_geom|moving_jaw_pad_v2"]
        self.assertGreater(carry_penetration["contact_steps"], 0)
        self.assertGreater(carry_penetration["overlap_steps"], 0)
        self.assertGreater(
            carry_penetration["max_penetration_mm"], 0.0)
        self.assertEqual(
            [0.02, 1.0],
            carry_penetration["contact_geometry_parameters"][0]["solref"])
        self.assertLess(
            carry_penetration["first_contact_time_s"],
            carry_penetration["last_contact_time_s"])
        self.assertAlmostEqual(
            carry_penetration["contact_steps"]
            * penetration["sample_interval_s"],
            carry_penetration["sampled_contact_duration_s"],
        )
        contact_scopes = result["contact_diagnostics"]["scopes"]
        feed_contacts = contact_scopes["feed"]["overall"]
        self.assertGreater(feed_contacts["samples"], 10000)
        self.assertGreater(
            feed_contacts["categories"]["robot_workpiece"][
                "peak_contacts_per_step"], 0)
        self.assertGreater(
            feed_contacts["categories"]["workpiece_machine"][
                "peak_contacts_per_step"], 0)
        self.assertTrue(any(
            pair["peak_normal_force_per_contact_N"] > 0.0
            for pair in feed_contacts["pairs"].values()))

        load_scopes = result["applied_load_diagnostics"]["scopes"]
        self.assertNotIn(
            "shaft", load_scopes["feed"]["overall"]["xfrc_by_body"])
        pull_load = load_scopes["pull_test"]["overall"]["xfrc_by_body"]["shaft"]
        self.assertAlmostEqual(20.0, pull_load["force_peak_N"], places=6)
        self.assertEqual(700, pull_load["nonzero_samples"])
        pull_stage = next(
            row for row in result["stages"]
            if row["stage"] == "after_20N_pull_test")
        self.assertEqual("pull_test", pull_stage["diagnostic_scope"])


class FeedFailureStatusTests(unittest.TestCase):
    def _run_feed(self, case_id, **kwargs):
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / (case_id + ".json")
            previous_path = os.environ.get("FEED_RESULT_JSON")
            previous_case_id = os.environ.get("FEED_CASE_ID")
            os.environ["FEED_RESULT_JSON"] = str(result_path)
            os.environ["FEED_CASE_ID"] = case_id
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = feed.main(**kwargs)
            finally:
                if previous_path is None:
                    os.environ.pop("FEED_RESULT_JSON", None)
                else:
                    os.environ["FEED_RESULT_JSON"] = previous_path
                if previous_case_id is None:
                    os.environ.pop("FEED_CASE_ID", None)
                else:
                    os.environ["FEED_CASE_ID"] = previous_case_id
            result = json.loads(result_path.read_text(encoding="utf-8"))
        return exit_code, result

    def test_empty_grasp_aborts_and_is_counted_as_attempt(self):
        exit_code, result = self._run_feed(
            "test-empty-grasp",
            v_jaw=True,
            lift=True,
            carry_lock=False,
            planning_mode="M2",
            m2_error_mm=(0.0, 30.0, 0.0),
        )
        self.assertEqual(1, exit_code)
        assert_reproducibility_fields(self, result)
        self.assertEqual("aborted", result["execution_status"])
        self.assertEqual("grasp", result["abort_stage"])
        self.assertEqual("no_v_jaw_contact", result["abort_reason"])
        self.assertEqual(["grasp_empty"], result["hard_failures"])
        self.assertEqual(0, result["metrics"]["grasp_contact_count"])
        self.assertTrue(result["initial_stability"]["stability_pass"])
        self.assertEqual(
            [0.0, 30.0, 0.0],
            result["inputs"]["grasp_target_pose"]["m2_error_used_mm"])
        self.assertEqual(
            "aborted_before_carry",
            result["gripper_relative_workpiece_motion"]["status"])
        self.assertEqual(
            0,
            result["gripper_relative_workpiece_motion"]["overall"]["samples"])
        self.assertNotIn(
            "feed_complete", [row["stage"] for row in result["stages"]])
        self.assertNotIn("chuck_clamped", [
            row["checkpoint_id"] for row in result["checkpoints"]])
        self.assertTrue(result["trial_data_valid"])
        self.assertTrue(result["include_in_primary_trial_stats"])
        self.assertEqual("aborted", result["outcome_class"])

    def test_controlled_quality_failure_is_not_mislabeled_as_abort(self):
        previous_solref = os.environ.get("FEED_V_CONTACT_SOLREF_TIME")
        os.environ["FEED_V_CONTACT_SOLREF_TIME"] = "0.005"
        try:
            exit_code, result = self._run_feed(
                "test-quality-fail-solref-005",
                v_jaw=True,
                lift=True,
                carry_lock=False,
            )
        finally:
            if previous_solref is None:
                os.environ.pop("FEED_V_CONTACT_SOLREF_TIME", None)
            else:
                os.environ["FEED_V_CONTACT_SOLREF_TIME"] = previous_solref

        self.assertEqual(1, exit_code)
        assert_reproducibility_fields(self, result)
        self.assertEqual("completed", result["execution_status"])
        self.assertTrue(result["hard_pass"])
        self.assertFalse(result["quality_pass"])
        self.assertFalse(result["overall_pass"])
        self.assertEqual(["stock_eccentricity"], result["quality_failures"])
        self.assertEqual([], result["hard_failures"])
        self.assertEqual("quality_fail", result["outcome_class"])
        self.assertGreater(result["metrics"]["stock_ecc_mm"], 2.0)
        assert_feed_checkpoints(
            self, result, ("completed",) * 6)

    def test_zero_initial_settle_records_the_perturbed_grasp_start_pose(self):
        _exit_code, result = self._run_feed(
            "test-zero-initial-settle",
            v_jaw=True,
            lift=True,
            carry_lock=False,
            initial_offset_mm=(0.0, 3.0, 2.0),
            initial_yaw_deg=5.0,
            initial_settle_steps=0,
        )

        self.assertEqual(0, result["inputs"]["initial_settle_steps"])
        self.assertEqual(
            "insufficient_samples", result["initial_stability"]["status"])
        self.assertIsNone(result["initial_stability"]["stability_pass"])
        requested = result["inputs"]["pose_after_perturbation_before_settle"]
        actual = result["actual_initial_pose"]
        np.testing.assert_allclose(
            requested["position_m"], actual["position_m"], atol=1e-9)
        np.testing.assert_allclose(
            requested["shaft_axis_world_unit"],
            actual["shaft_axis_world_unit"],
            atol=1e-9,
        )

    def test_initial_settle_steps_must_be_a_nonnegative_integer(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    feed.main(initial_settle_steps=value)

    def test_non_finite_state_is_runtime_error_not_task_failure(self):
        stage_calls = [0]

        def inject_nan(tag, model, data):
            if tag is not None:
                stage_calls[0] += 1
                if stage_calls[0] == 2:
                    data.qpos[0] = np.nan

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "runtime_error.json"
            previous_path = os.environ.get("FEED_RESULT_JSON")
            os.environ["FEED_RESULT_JSON"] = str(result_path)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(FloatingPointError):
                        feed.main(
                            frame_hook=inject_nan,
                            v_jaw=True,
                            lift=True,
                            carry_lock=False,
                        )
            finally:
                if previous_path is None:
                    os.environ.pop("FEED_RESULT_JSON", None)
                else:
                    os.environ["FEED_RESULT_JSON"] = previous_path
            result = json.loads(result_path.read_text(encoding="utf-8"))

        assert_reproducibility_fields(self, result)
        self.assertEqual("error", result["execution_status"])
        self.assertEqual("non_finite_state", result["abort_reason"])
        self.assertEqual(["non_finite_state"], result["hard_failures"])
        self.assertFalse(result["trial_data_valid"])
        self.assertFalse(result["include_in_primary_trial_stats"])
        self.assertEqual("runtime_error", result["outcome_class"])


if __name__ == "__main__":
    unittest.main()
