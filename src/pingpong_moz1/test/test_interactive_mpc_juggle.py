#!/usr/bin/env python3

"""Integration coverage for the browser-facing MPC juggling controller."""

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
    "interactive_mpc_juggle_probe", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


EXPECTED_INITIAL_Q_DEG = np.array(
    [5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214],
    dtype=np.float64,
)


class InteractiveMpcJuggleIntegrationTest(unittest.TestCase):

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
        # The reset parking window is a real-robot recovery feature.  These
        # integration tests begin after it, as the existing tracking tests do.
        current_reset_q = self.probe.reset_initial_pose_q.copy()
        self.probe.slow_reset_trajectory.start(
            current_reset_q, current_reset_q)
        self.probe._sync_policy_to_safe_pose(current_reset_q)
        self.probe.reset_workspace_bypass_until_wall = 0.0
        self.probe.reset_workspace_bypass_target_q = None

    def juggle_payload(self, ball, *, run=False, nonce=0, **updates):
        payload = {
            "ball": np.asarray(ball, dtype=np.float64).tolist(),
            "velocity": [0.0, 0.0, 0.0],
            "valid": True,
            "controlMode": "mpc_juggle",
            "juggleRunRequested": bool(run),
            "juggleStartNonce": int(nonce),
            "useBallEstimator": False,
            # These are deliberately left enabled: neither RL lifecycle gate
            # may prevent placement, countdown, launch, or flight control.
            "enableCaptureGate": True,
            "enableReleaseGate": True,
            "enableDoneMonitoring": False,
            "workspaceSafetyMode": "off",
            "dt": self.probe.cfg.control_dt,
        }
        payload.update(updates)
        return payload

    def tracking_payload(self, ball, **updates):
        payload = {
            "ball": np.asarray(ball, dtype=np.float64).tolist(),
            "velocity": [0.03, -0.01, 0.0],
            "valid": True,
            "controlMode": "racket_tracking",
            "racketTrackingHeightOffsetM": 0.25,
            "racketTrackingUseMpc": True,
            "useBallEstimator": False,
            "enableCaptureGate": True,
            "enableReleaseGate": True,
            "enableDoneMonitoring": False,
            "workspaceSafetyMode": "off",
            "dt": self.probe.cfg.control_dt,
        }
        payload.update(updates)
        return payload

    def seed_initial_pose(self, *, ball_radius_m=0.020):
        """Place command feedback and the ball on the requested racket pose."""

        q = np.asarray(PROBE.JUGGLE_INITIAL_Q_RAD, dtype=np.float32)
        zeros = np.zeros(q.size, dtype=np.float32)
        self.probe._sync_controller_to_tracking_command(
            q, zeros, reset_acceleration=True)
        # The deployment runtime intentionally exposes no safety-limiter
        # methods.  Seed only the independent workspace command-history state.
        self.probe.safety_limiter.last_cmd = q.astype(np.float64).copy()
        self.probe.safety_limiter.last_vel = zeros.astype(np.float64).copy()
        position, quaternion, _j_pos, _j_rot, ok = (
            self.probe.controller._compute_racket_pose_jacobians_base(q)
        )
        self.assertTrue(ok)
        normal = PROBE.racket_normal_from_quaternion_xyzw(quaternion)
        self.assertGreater(normal[2], 0.95)
        ball = (
            np.asarray(position, dtype=np.float64)
            + float(ball_radius_m) * np.asarray(normal, dtype=np.float64)
        )
        return q, np.asarray(position, dtype=np.float64), normal, ball

    @staticmethod
    def shared_mpc_payload():
        return {
            "racketMpcHorizonSteps": 7,
            "racketMpcPositionWeight": 13.0,
            "racketMpcVelocityWeight": 4.5,
            "racketMpcEffortWeight": 0.003,
            "racketMpcSmoothWeight": 0.004,
            "racketMpcPostureWeight": 0.005,
            "racketMpcMaxCartesianSpeedMps": 1.25,
            "racketMpcMaxCartesianAccelerationMps2": 20.0,
            "racketMpcJointVelocityScale": 0.80,
            "racketMpcJointAccelerationScale": 1.0,
        }

    def zero_delay_start_payload(self):
        return {
            "juggleCountdownS": 0.0,
            "juggleReadySettleS": 0.0,
            "juggleSeedApexHeightM": 0.002,
            "juggleLaunchPeakAccelerationMps2": 12.0,
            **self.shared_mpc_payload(),
        }

    def advance_to_launch(self, *, nonce=1):
        _q, _racket, _normal, ball = self.seed_initial_pose()
        common = self.zero_delay_start_payload()

        ready = self.probe.step(self.juggle_payload(
            ball, run=False, nonce=0, **common))
        self.assertEqual(ready["juggle"]["state"], "ready_wait")
        self.assertTrue(ready["juggle"]["ready"])

        countdown = self.probe.step(self.juggle_payload(
            ball, run=True, nonce=nonce, **common))
        self.assertEqual(countdown["juggle"]["state"], "countdown")
        self.assertEqual(countdown["juggle"]["countdownRemainingS"], 0.0)

        launch = self.probe.step(self.juggle_payload(
            ball, run=True, nonce=nonce, **common))
        self.assertEqual(launch["mode"], "mpc_juggle")
        self.assertEqual(launch["juggle"]["state"], "launch")
        self.assertEqual(launch["juggle"]["controlKind"], "cartesian")
        return ball, common, launch

    def test_browser_config_disables_mode_but_retains_diagnostic_defaults(self):
        config = self.probe.browser_config()

        self.assertNotIn("mpc_juggle", config["control"]["modes"])
        self.assertIn("racketMpc", config)
        self.assertIn("mpcJuggle", config)
        self.assertFalse(config["mpcJuggle"]["available"])
        self.assertEqual(
            config["mpcJuggle"]["disabledReason"],
            "disabled_by_operator_request",
        )
        self.assertFalse(config["mpcJuggle"]["runRequested"])
        self.assertEqual(config["mpcJuggle"]["startNonce"], 0)
        self.assertEqual(
            config["mpcJuggle"]["contactGraceS"],
            self.probe.cfg.juggle_contact_grace_s,
        )
        np.testing.assert_allclose(
            config["mpcJuggle"]["initialJointPositionDeg"],
            EXPECTED_INITIAL_Q_DEG,
            rtol=0.0,
            # The web payload is produced from the float32 command pose.
            atol=1.0e-5,
        )
        self.assertEqual(
            config["racketMpc"]["horizonSteps"],
            self.probe.cfg.racket_tracking_mpc_horizon_steps,
        )
        for key in (
            "positionWeight",
            "velocityWeight",
            "effortWeight",
            "smoothWeight",
            "postureWeight",
            "maxCartesianSpeedMps",
            "maxCartesianAccelerationMps2",
            "jointVelocityScale",
            "jointAccelerationScale",
        ):
            self.assertIn(key, config["racketMpc"])

    def test_disarmed_mode_bypasses_rl_gates_and_positions_then_reports_ready(self):
        snapshot = self.probe.step(self.juggle_payload(
            self.probe.initial_ball_pos(), run=False))

        self.assertEqual(snapshot["mode"], "mpc_juggle")
        self.assertEqual(snapshot["juggle"]["state"], "positioning")
        self.assertEqual(snapshot["juggle"]["controlKind"], "joint")
        self.assertFalse(snapshot["runtime"]["juggleRunRequested"])
        self.assertTrue(snapshot["releaseGate"]["bypassedForTracking"])
        self.assertFalse(self.probe.has_released)

        _q, _racket, _normal, ball = self.seed_initial_pose()
        ready = self.probe.step(self.juggle_payload(
            ball,
            run=False,
            juggleReadySettleS=0.0,
        ))
        self.assertEqual(ready["mode"], "mpc_juggle")
        self.assertEqual(ready["juggle"]["state"], "ready_wait")
        self.assertTrue(ready["juggle"]["ready"])
        self.assertAlmostEqual(
            ready["juggle"]["placement"]["heightErrorM"], 0.0,
            places=7,
        )

    def test_shared_runtime_mpc_tuning_reaches_tracking_and_juggle_low_level(self):
        custom = self.shared_mpc_payload()
        tracking = self.probe.step(self.tracking_payload(
            self.probe.initial_ball_pos(), **custom))
        expected = tracking["runtime"]["racketMpc"]

        self.assertEqual(expected["horizonSteps"], 7)
        self.assertEqual(expected["positionWeight"], 13.0)
        self.assertEqual(expected["velocityWeight"], 4.5)
        self.assertEqual(expected["effortWeight"], 0.003)
        self.assertEqual(tracking["tracking"]["solver"], "linear_mpc")
        self.assertEqual(tracking["tracking"]["mpcTuning"], expected)

        self.probe.reset()
        current_reset_q = self.probe.reset_initial_pose_q.copy()
        self.probe.slow_reset_trajectory.start(
            current_reset_q, current_reset_q)
        self.probe._sync_policy_to_safe_pose(current_reset_q)
        self.probe.reset_workspace_bypass_until_wall = 0.0
        self.probe.reset_workspace_bypass_target_q = None
        _ball, _common, launch = self.advance_to_launch(nonce=19)
        self.assertEqual(launch["runtime"]["racketMpc"], expected)
        self.assertEqual(
            launch["juggle"]["lowLevel"]["mpcTuning"], expected)
        self.assertEqual(
            launch["juggle"]["lowLevel"]["solver"], "linear_mpc")

    def test_ball_on_real_fk_surface_start_nonce_launches_and_stop_holds(self):
        ball, common, launch = self.advance_to_launch(nonce=41)

        self.assertTrue(launch["runtime"]["juggleRunRequested"])
        self.assertEqual(launch["runtime"]["juggleStartNonce"], 41)
        self.assertEqual(launch["juggle"]["lowLevel"]["solver"], "linear_mpc")
        self.assertAlmostEqual(
            launch["juggle"]["placement"]["normalGapM"],
            launch["runtime"]["jugglePhysics"]["ballRadiusM"],
            places=7,
        )

        stop = self.probe.step(self.juggle_payload(
            ball, run=False, nonce=41, **common))
        self.assertEqual(stop["mode"], "mpc_juggle_hold")
        self.assertEqual(stop["juggle"]["state"], "stopped_hold")
        self.assertEqual(stop["juggle"]["controlKind"], "hold")
        self.assertFalse(stop["runtime"]["juggleRunRequested"])
        self.assertFalse(stop["juggle"]["active"])

    def test_stale_start_payload_cannot_restart_after_mode_reentry(self):
        ball, common, _launch = self.advance_to_launch(nonce=73)
        self.probe.step(self.tracking_payload(ball))
        _q, _racket, _normal, ball = self.seed_initial_pose()

        stale_first = self.probe.step(self.juggle_payload(
            ball, run=True, nonce=73, **common))
        stale_second = self.probe.step(self.juggle_payload(
            ball, run=True, nonce=73, **common))

        self.assertNotIn(
            stale_first["juggle"]["state"], {"countdown", "launch"})
        self.assertNotIn(
            stale_second["juggle"]["state"], {"countdown", "launch"})
        self.assertFalse(stale_second["juggle"]["startEdgeArmed"])

    def test_active_stale_ball_fails_closed_to_hold(self):
        ball, common, _launch = self.advance_to_launch(nonce=51)
        payload = self.juggle_payload(
            ball, run=True, nonce=51, **common)
        options = self.probe._runtime_options_from_payload(payload)

        mode, reason, _conditions, _age, executed = (
            self.probe._run_one_control_step(
                np.asarray(ball, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                True,
                options.juggle_physics.max_ball_age_s + 0.001,
                options,
            )
        )

        self.assertEqual(mode, "mpc_juggle_hold")
        self.assertIn("stale", reason)
        self.assertFalse(executed)
        self.assertEqual(self.probe.last_juggle_info["state"], "fault_hold")
        self.assertTrue(self.probe.last_juggle_info["requiresRearm"])

    def test_commanding_requires_fresh_feedback_but_accepts_fresh_sample(self):
        q, _racket, _normal, ball = self.seed_initial_pose()
        payload = self.juggle_payload(
            ball,
            run=False,
            juggleReadySettleS=0.0,
        )
        options = replace(
            self.probe._runtime_options_from_payload(payload),
            use_ros_state=True,
            publish_commands=True,
        )

        class FeedbackBridge:
            def __init__(self, age_s):
                self.age_s = age_s

            def get_joint_feedback_with_age(self):
                return (
                    q.copy(),
                    np.zeros(7, dtype=np.float32),
                    self.age_s,
                    "joint_states",
                )

            @staticmethod
            def get_cart_feedback():
                return None, None, None

        original_bridge = self.probe.ros_bridge
        try:
            self.probe.ros_bridge = FeedbackBridge(0.0)
            fresh_mode, fresh_reason, _conditions, _age, _executed = (
                self.probe._run_one_control_step(
                    np.asarray(ball, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    True,
                    0.0,
                    options,
                )
            )
            self.assertEqual(fresh_mode, "mpc_juggle")
            self.assertEqual(fresh_reason, "")

            # Re-enter, arm, and reach an active countdown before making the
            # feedback stale.  This proves the guard is not just an idle check.
            self.probe.juggle_controller.enter_mode()
            self.probe.last_control_mode = "mpc_juggle"
            self.probe.juggle_controller.state = "countdown"
            self.probe.juggle_controller.run_requested = True
            self.probe.ros_bridge = FeedbackBridge(0.051)
            stale_options = replace(
                options, juggle_run_requested=True, juggle_start_nonce=9)
            stale_mode, stale_reason, _conditions, _age, executed = (
                self.probe._run_one_control_step(
                    np.asarray(ball, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    True,
                    0.0,
                    stale_options,
                )
            )
        finally:
            self.probe.ros_bridge = original_bridge

        self.assertEqual(stale_mode, "mpc_juggle_hold")
        self.assertIn("fresh joint feedback required", stale_reason)
        self.assertFalse(executed)
        self.assertEqual(self.probe.last_juggle_info["state"], "fault_hold")

    def test_juggle_planner_uses_fixed_step_control_clock(self):
        """Host scheduling jitter must not change the MPC event grid."""

        _q, _racket, _normal, ball = self.seed_initial_pose()
        payload = self.juggle_payload(
            ball,
            run=False,
            juggleReadySettleS=0.0,
        )
        options = self.probe._runtime_options_from_payload(payload)
        self.probe.now_s = 12.345
        captured_now_s = []
        original_step = self.probe.juggle_controller.step

        def capture_control_time(**kwargs):
            captured_now_s.append(float(kwargs["wall_now_s"]))
            return original_step(**kwargs)

        try:
            self.probe.juggle_controller.step = capture_control_time
            self.probe._run_one_control_step(
                np.asarray(ball, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                True,
                0.0,
                options,
            )
        finally:
            self.probe.juggle_controller.step = original_step

        self.assertEqual(len(captured_now_s), 1)
        self.assertAlmostEqual(captured_now_s[0], 12.350, places=12)
        self.assertAlmostEqual(self.probe.now_s, 12.350, places=12)


if __name__ == "__main__":
    unittest.main()
