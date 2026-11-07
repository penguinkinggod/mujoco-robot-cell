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


def assert_reproducibility_fields(test_case, result):
    test_case.assertEqual(4, result["schema_version"])
    test_case.assertEqual(1, result["metric_schema_version"])
    test_case.assertEqual("feed-criteria-v1", result["criterion_set_id"])
    test_case.assertTrue(result["case_id"])
    test_case.assertIn(result["execution_status"],
                       ("completed", "aborted", "error"))
    test_case.assertIn("stages", result)
    test_case.assertIn("hard_failures", result)
    test_case.assertIn("quality_failures", result)
    test_case.assertEqual(result["git_revision"],
                          result["versions"]["git_revision"])
    test_case.assertIsInstance(result["git_worktree_dirty"], bool)
    test_case.assertIn("workpiece", result["inputs"])
    test_case.assertEqual(20.0, result["inputs"]["workpiece"]["diameter_mm"])
    test_case.assertEqual(80.0, result["inputs"]["workpiece"]["length_mm"])
    test_case.assertIn("requested_initial_pose", result["inputs"])
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
        self.assertEqual("completed", result["execution_status"])
        self.assertGreater(
            result["metrics"]["robot_machine_contact_max"], 0)
        self.assertIn("robot_machine_collision", result["hard_failures"])
        self.assertEqual("hard_fail", result["outcome_class"])


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
