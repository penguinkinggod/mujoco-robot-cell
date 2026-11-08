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


def assert_reproducibility_fields(test_case, result):
    test_case.assertEqual(7, result["schema_version"])
    test_case.assertEqual(1, result["metric_schema_version"])
    test_case.assertEqual("feed-criteria-v1", result["criterion_set_id"])
    test_case.assertTrue(result["case_id"])
    test_case.assertIn(result["execution_status"],
                       ("completed", "aborted", "error"))
    test_case.assertIn("stages", result)
    test_case.assertIn("checkpoints", result)
    test_case.assertIn("contact_diagnostics", result)
    test_case.assertIn("applied_load_diagnostics", result)
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
        assert_reproducibility_fields(self, result)
        self.assertEqual(
            0.0,
            result["inputs"]["requested_initial_pose"]["yaw_world_z_deg"])
        self.assertEqual(
            "settled_true_workpiece_center",
            result["inputs"]["grasp_target_pose"]["position_source"])
        self.assertEqual(
            "fixed_calibrated_grasp_pose",
            result["inputs"]["grasp_target_pose"]["orientation_source"])
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
            initial_offset_mm=(0.0, 50.0, 0.0),
        )
        self.assertEqual(1, exit_code)
        assert_reproducibility_fields(self, result)
        self.assertEqual("aborted", result["execution_status"])
        self.assertEqual("grasp", result["abort_stage"])
        self.assertEqual("no_v_jaw_contact", result["abort_reason"])
        self.assertEqual(["grasp_empty"], result["hard_failures"])
        self.assertEqual(0, result["metrics"]["grasp_contact_count"])
        self.assertTrue(result["trial_data_valid"])
        self.assertTrue(result["include_in_primary_trial_stats"])
        self.assertEqual("aborted", result["outcome_class"])

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
