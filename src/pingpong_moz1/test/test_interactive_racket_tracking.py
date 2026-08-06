#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

PROBE_PATH = (
    PACKAGE_ROOT / "pingpong_controller" / "tools"
    / "interactive_policy_safety_probe.py"
)
SPEC = importlib.util.spec_from_file_location(
    "interactive_racket_tracking_probe", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


class InteractiveRacketTrackingIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        args = PROBE.build_arg_parser().parse_args([
            "--disable-ball-state-estimator",
            "--disable-done-monitoring",
        ])
        cls.probe = PROBE.PolicySafetyProbe(PROBE.config_from_args(args))

    @classmethod
    def tearDownClass(cls):
        cls.probe.shutdown()

    def setUp(self):
        self.probe.reset()
        # Reset recovery is a real-robot parking window.  Unit integration
        # starts after that window so it can exercise the selected controller.
        self.probe.reset_workspace_bypass_until_wall = 0.0
        self.probe.reset_workspace_bypass_target_q = None

    def tracking_payload(self, *, use_mpc=False):
        ball = self.probe.initial_ball_pos()
        return {
            "ball": list(ball),
            "velocity": [0.0, 0.0, 0.0],
            "valid": True,
            "controlMode": "racket_tracking",
            "racketTrackingHeightOffsetM": 0.25,
            "racketTrackingUseMpc": bool(use_mpc),
            "useBallEstimator": False,
            "enableCaptureGate": True,
            # Deliberately leave the RL release gate enabled.  Hand-held-ball
            # tracking must bypass it instead of waiting for a falling ball.
            "enableReleaseGate": True,
            "enableDoneMonitoring": False,
            "workspaceSafetyMode": "off",
            "dt": 0.02,
        }

    def test_tracking_bypasses_release_gate_and_reports_height_target(self):
        snapshot = self.probe.step(self.tracking_payload(use_mpc=False))

        self.assertEqual(snapshot["mode"], "racket_tracking")
        self.assertTrue(snapshot["tracking"]["active"])
        self.assertEqual(snapshot["tracking"]["solver"], "dls")
        self.assertAlmostEqual(snapshot["tracking"]["heightOffsetM"], 0.25)
        self.assertAlmostEqual(
            snapshot["tracking"]["target"][2],
            snapshot["observedBall"][2] - 0.25,
            places=6,
        )
        self.assertTrue(snapshot["releaseGate"]["bypassedForTracking"])
        self.assertEqual(snapshot["runtime"]["controlMode"], "racket_tracking")

    def test_tracking_mpc_is_distinct_from_actuator_compensation(self):
        payload = self.tracking_payload(use_mpc=True)
        payload["enableCompensation"] = False
        snapshot = self.probe.step(payload)

        self.assertEqual(snapshot["mode"], "racket_tracking")
        self.assertTrue(snapshot["tracking"]["mpcRequested"])
        self.assertTrue(snapshot["tracking"]["mpcActive"])
        self.assertEqual(snapshot["tracking"]["solver"], "linear_mpc")
        self.assertFalse(snapshot["compensation"]["enabled"])

    def test_web_switch_changes_ball_estimator_backend_at_runtime(self):
        payload = self.tracking_payload(use_mpc=False)
        payload.update({
            "useBallEstimator": True,
            "useHybridBallTracker": True,
            "estimatorPrediction": True,
        })

        hybrid = self.probe.step(payload)
        self.assertEqual(hybrid["estimator"]["backend"], "hybrid_ekf9")
        self.assertEqual(
            hybrid["runtime"]["estimatorBackend"], "hybrid_ekf9")

        payload["useHybridBallTracker"] = False
        legacy = self.probe.step(payload)
        self.assertEqual(legacy["estimator"]["backend"], "legacy_kf6")
        self.assertEqual(
            legacy["runtime"]["estimatorBackend"], "legacy_kf6")

    def test_real_fk_tracking_reduces_error_over_multiple_ticks(self):
        payload = self.tracking_payload(use_mpc=False)
        first = self.probe.step(payload)
        last = first
        for _ in range(99):
            last = self.probe.step(payload)

        self.assertLess(
            last["tracking"]["errorNormM"],
            first["tracking"]["errorNormM"],
        )
        self.assertTrue(np.all(np.isfinite(
            last["tracking"]["commandPosition"])))

    def test_stale_ball_state_fails_closed_to_tracking_hold(self):
        payload = self.tracking_payload()
        options = replace(
            self.probe._runtime_options_from_payload(payload),
            enable_ball_capture_gate=False,
        )
        ball = np.asarray(payload["ball"], dtype=np.float32)

        mode, reason, _conditions, _age, executed = (
            self.probe._run_one_control_step(
                ball,
                np.zeros(3, dtype=np.float32),
                True,
                (
                    self.probe.cfg.racket_tracking_max_ball_age_s + 0.001),
                options,
            )
        )

        self.assertEqual(mode, "racket_tracking_hold")
        self.assertIn("stale", reason)
        self.assertFalse(executed)
        self.assertFalse(self.probe.last_tracking_info["active"])

    def test_long_ball_loss_holds_instead_of_rl_parking(self):
        payload = self.tracking_payload()
        options = replace(
            self.probe._runtime_options_from_payload(payload),
            enable_ball_capture_gate=False,
        )
        command_before = self.probe.controller.current_arm_cmd_q().copy()

        mode, reason, _conditions, _age, executed = (
            self.probe._run_one_control_step(
                np.asarray(payload["ball"], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                self.probe.cfg.ball_lost_timeout_s,
                options,
            )
        )

        self.assertEqual(mode, "racket_tracking_hold")
        self.assertIn("measured hold", reason)
        self.assertFalse(executed)
        np.testing.assert_allclose(
            self.probe.controller.current_arm_cmd_q(), command_before)

    def test_uncalibrated_camera_frame_fails_closed(self):
        class UncalibratedCameraBridge:
            @staticmethod
            def camera_control_frame_status():
                return False, "camera"

        payload = self.tracking_payload()
        options = replace(
            self.probe._runtime_options_from_payload(payload),
            ball_source="camera",
        )
        original_bridge = self.probe.ros_bridge
        self.probe.ros_bridge = UncalibratedCameraBridge()
        try:
            mode, reason, _conditions, _age, executed = (
                self.probe._run_one_control_step(
                    np.asarray(payload["ball"], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    True,
                    0.0,
                    options,
                )
            )
        finally:
            self.probe.ros_bridge = original_bridge

        self.assertEqual(mode, "racket_tracking_hold")
        self.assertIn("camera calibration required", reason)
        self.assertFalse(executed)

    def test_camera_timestamp_age_is_kept_without_estimator(self):
        class ClockBridge:
            @staticmethod
            def now_sec():
                return 10.0

        payload = self.tracking_payload()
        options = replace(
            self.probe._runtime_options_from_payload(payload),
            ball_source="camera",
            use_ball_estimator=False,
        )
        original_bridge = self.probe.ros_bridge
        self.probe.ros_bridge = ClockBridge()
        try:
            _pos, _vel, valid, age_s, _info = (
                self.probe._resolve_control_ball_state(
                    np.asarray(payload["ball"], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    True,
                    options,
                    measurement_stamp_s=9.8,
                )
            )
        finally:
            self.probe.ros_bridge = original_bridge

        self.assertTrue(valid)
        self.assertAlmostEqual(age_s, 0.2)

    def test_missing_camera_timestamp_fails_closed_without_estimator(self):
        class ClockBridge:
            @staticmethod
            def now_sec():
                return 10.0

        payload = self.tracking_payload()
        options = replace(
            self.probe._runtime_options_from_payload(payload),
            ball_source="camera",
            use_ball_estimator=False,
        )
        original_bridge = self.probe.ros_bridge
        self.probe.ros_bridge = ClockBridge()
        try:
            _pos, _vel, valid, age_s, _info = (
                self.probe._resolve_control_ball_state(
                    np.asarray(payload["ball"], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    True,
                    options,
                    measurement_stamp_s=None,
                )
            )
        finally:
            self.probe.ros_bridge = original_bridge

        self.assertTrue(valid)
        self.assertGreater(
            age_s, self.probe.cfg.racket_tracking_max_ball_age_s)

    def test_publishing_requires_fresh_joint_feedback(self):
        probe = self.probe

        class StaleFeedbackBridge:
            @staticmethod
            def get_joint_feedback_with_age():
                return (
                    probe.controller.current_arm_cmd_q(),
                    np.zeros(7, dtype=np.float32),
                    0.051,
                    "joint_states",
                )

        payload = self.tracking_payload()
        options = replace(
            self.probe._runtime_options_from_payload(payload),
            use_ros_state=True,
            publish_commands=True,
        )
        original_bridge = self.probe.ros_bridge
        self.probe.ros_bridge = StaleFeedbackBridge()
        try:
            mode, reason, _conditions, _age, executed = (
                self.probe._run_one_control_step(
                    np.asarray(payload["ball"], dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    True,
                    0.0,
                    options,
                )
            )
        finally:
            self.probe.ros_bridge = original_bridge

        self.assertEqual(mode, "racket_tracking_hold")
        self.assertIn("fresh joint feedback required", reason)
        self.assertFalse(executed)


if __name__ == "__main__":
    unittest.main()
