#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.racket_ball_tracker import (  # noqa: E402
    RacketBallTracker,
    RacketBallTrackerConfig,
    RacketMpcTuning,
    racket_target_below_ball,
)


class LinearKinematics:
    """Identity XYZ kinematics in the first three of seven joints."""

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(7)
        position = q[:3].copy()
        jacobian = np.zeros((3, 7), dtype=np.float64)
        jacobian[:, :3] = np.eye(3)
        return (
            position,
            np.array([0.0, 0.0, 0.0, 1.0]),
            jacobian,
            np.zeros((3, 7), dtype=np.float64),
            True,
        )


class RedundantCancellationKinematics:
    """Two joints whose equal velocities cancel in task space."""

    def __call__(self, q):
        np.asarray(q, dtype=np.float64).reshape(2)
        jacobian = np.array([
            [1.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        return (
            np.zeros(3),
            np.array([0.0, 0.0, 0.0, 1.0]),
            jacobian,
            np.zeros((3, 2), dtype=np.float64),
            True,
        )


def make_tracker(**overrides):
    config_values = {
        "dt": 0.01,
        "position_gain_s": 4.0,
        "dls_damping": 0.01,
        "max_cartesian_speed_m_s": 1.0,
        "joint_velocity_scale": 1.0,
        "joint_acceleration_scale": 1.0,
        "posture_gain_s": 0.0,
        "mpc_horizon_steps": 10,
        "mpc_velocity_weight": 1.0e-6,
        "mpc_smooth_weight": 1.0e-6,
        "mpc_posture_weight": 0.0,
        "target_boundary_margin_m": 0.0,
    }
    config_values.update(overrides)
    return RacketBallTracker(
        kinematics=LinearKinematics(),
        nominal_q_rad=np.zeros(7),
        joint_lower_rad=np.full(7, -2.0),
        joint_upper_rad=np.full(7, 2.0),
        joint_velocity_limit_rad_s=np.ones(7),
        joint_acceleration_limit_rad_s2=np.full(7, 100.0),
        config=RacketBallTrackerConfig(**config_values),
    )


class TargetGeometryTest(unittest.TestCase):

    def test_target_is_directly_below_ball(self):
        target = racket_target_below_ball(
            np.array([0.2, -0.4, 1.3]), 0.25)

        np.testing.assert_allclose(target, [0.2, -0.4, 1.05])

    def test_negative_or_non_finite_offset_is_rejected(self):
        with self.assertRaises(ValueError):
            racket_target_below_ball(np.zeros(3), -0.1)
        with self.assertRaises(ValueError):
            racket_target_below_ball(np.zeros(3), float("nan"))

    def test_outer_planner_can_reuse_exact_low_level_inset_limits(self):
        tracker = make_tracker(target_boundary_margin_m=0.01)

        limits = tracker.inset_position_limits(
            ((-0.5, 0.5), None, (0.7, 1.4)))

        self.assertEqual(limits, ((-0.49, 0.49), None, (0.71, 1.39)))


class TrackerSolverTest(unittest.TestCase):

    def test_joint_position_move_reuses_runtime_rate_limits(self):
        tracker = make_tracker()
        tuning = RacketMpcTuning(
            max_cartesian_speed_m_s=1.0,
            max_cartesian_acceleration_m_s2=2.0,
            joint_velocity_scale=0.2,
            joint_acceleration_scale=0.05,
        )

        command, diagnostics = tracker.move_to_joint_position(
            feedback_q_rad=np.zeros(7),
            feedback_dq_rad_s=np.zeros(7),
            target_q_rad=np.ones(7),
            mpc_tuning=tuning,
        )

        max_first_velocity = 100.0 * 0.05 * 0.01
        self.assertGreater(command[0], 0.0)
        self.assertLessEqual(
            max(abs(v) for v in diagnostics["commandVelocity"]),
            max_first_velocity + 1e-12,
        )
        self.assertEqual(diagnostics["solver"], "joint_position_ramp")
        self.assertEqual(diagnostics["mpcTuning"], tuning.diagnostics())
        self.assertTrue(diagnostics["accelerationLimited"])

    def test_dls_step_reduces_cartesian_error(self):
        tracker = make_tracker()
        q = np.zeros(7)
        tracker.reset(q)
        target = np.array([0.2, -0.1, 0.3])
        initial_error = np.linalg.norm(target - q[:3])

        command, diagnostics = tracker.step(
            feedback_q_rad=q,
            feedback_dq_rad_s=np.zeros(7),
            ball_position_m=np.array([0.2, -0.1, 0.5]),
            ball_velocity_m_s=np.zeros(3),
            height_offset_m=0.2,
            use_mpc=False,
        )

        self.assertEqual(diagnostics["solver"], "dls")
        self.assertLess(np.linalg.norm(target - command[:3]), initial_error)
        self.assertAlmostEqual(diagnostics["heightOffsetM"], 0.2)

    def test_linear_mpc_path_is_finite_and_moves_toward_target(self):
        tracker = make_tracker()
        q = np.zeros(7)
        tracker.reset(q)

        command, diagnostics = tracker.step(
            feedback_q_rad=q,
            feedback_dq_rad_s=np.zeros(7),
            ball_position_m=np.array([0.15, 0.1, 0.4]),
            ball_velocity_m_s=np.array([0.05, 0.0, 0.0]),
            height_offset_m=0.2,
            use_mpc=True,
        )

        self.assertTrue(np.all(np.isfinite(command)))
        self.assertEqual(diagnostics["solver"], "linear_mpc")
        self.assertTrue(diagnostics["mpcActive"])
        self.assertGreater(command[0], 0.0)
        self.assertGreater(command[2], 0.0)

    def test_runtime_tuning_is_applied_and_echoed(self):
        tracker = make_tracker(
            max_cartesian_speed_m_s=1.0,
            joint_velocity_scale=1.0,
            joint_acceleration_scale=1.0,
        )
        tracker.joint_acceleration_limit[:] = 1.0e6
        tuning = RacketMpcTuning(
            horizon_steps=3,
            position_weight=2.0,
            velocity_weight=0.25,
            effort_weight=0.5,
            smooth_weight=0.125,
            posture_weight=0.0625,
            max_cartesian_speed_m_s=0.05,
            max_cartesian_acceleration_m_s2=3.0,
            joint_velocity_scale=0.4,
            joint_acceleration_scale=0.3,
        )

        _command, diagnostics = tracker.step(
            feedback_q_rad=np.zeros(7),
            feedback_dq_rad_s=np.zeros(7),
            ball_position_m=np.array([1.0, 0.0, 0.0]),
            ball_velocity_m_s=np.zeros(3),
            height_offset_m=0.0,
            use_mpc=True,
            mpc_tuning=tuning,
        )

        self.assertLessEqual(diagnostics["cartesianSpeedMps"], 0.05 + 1e-9)
        self.assertEqual(diagnostics["mpcTuning"], tuning.diagnostics())
        self.assertEqual(diagnostics["mpcTuning"]["horizonSteps"], 3)
        self.assertAlmostEqual(
            diagnostics["mpcTuning"]["jointVelocityScale"], 0.4)

    def test_runtime_joint_acceleration_scale_limits_this_tick(self):
        tracker = make_tracker(
            max_cartesian_speed_m_s=2.0,
            joint_velocity_scale=1.0,
            joint_acceleration_scale=1.0,
        )
        tracker.reset(np.zeros(7), np.zeros(7))
        tuning = RacketMpcTuning(
            horizon_steps=2,
            max_cartesian_speed_m_s=2.0,
            max_cartesian_acceleration_m_s2=2.0,
            joint_velocity_scale=1.0,
            joint_acceleration_scale=0.01,
        )

        _command, diagnostics = tracker.step(
            feedback_q_rad=np.zeros(7),
            feedback_dq_rad_s=np.zeros(7),
            ball_position_m=np.array([1.0, 0.0, 0.0]),
            ball_velocity_m_s=np.zeros(3),
            height_offset_m=0.0,
            use_mpc=True,
            mpc_tuning=tuning,
        )

        expected_max_dq = 100.0 * 0.01 * 0.01
        self.assertLessEqual(
            max(abs(v) for v in diagnostics["commandVelocity"]),
            expected_max_dq + 1e-12,
        )
        self.assertTrue(diagnostics["accelerationLimited"])

    def test_shared_cartesian_acceleration_limit_applies_to_both_solvers(self):
        for use_mpc in (False, True):
            tracker = make_tracker(max_cartesian_speed_m_s=2.0)
            tracker.joint_acceleration_limit[:] = 1.0e6
            tracker.reset(np.zeros(7), np.zeros(7))
            tuning = RacketMpcTuning(
                max_cartesian_speed_m_s=2.0,
                max_cartesian_acceleration_m_s2=0.20,
                joint_velocity_scale=1.0,
                joint_acceleration_scale=1.0,
            )

            _command, diagnostics = tracker.step(
                feedback_q_rad=np.zeros(7),
                feedback_dq_rad_s=np.zeros(7),
                ball_position_m=np.array([1.0, 0.0, 0.0]),
                ball_velocity_m_s=np.zeros(3),
                height_offset_m=0.0,
                use_mpc=use_mpc,
                mpc_tuning=tuning,
            )

            self.assertLessEqual(
                diagnostics["cartesianAccelerationMps2"],
                tuning.max_cartesian_acceleration_m_s2 + 1e-9,
            )
            self.assertTrue(diagnostics["cartesianAccelerationLimited"])

    def test_linear_mpc_has_explicit_target_velocity_cost(self):
        def command_velocity(velocity_weight):
            tracker = make_tracker()
            tracker.joint_acceleration_limit[:] = 1.0e6
            tuning = RacketMpcTuning(
                horizon_steps=1,
                position_weight=1.0e-9,
                velocity_weight=velocity_weight,
                effort_weight=1.0,
                smooth_weight=0.0,
                posture_weight=0.0,
                max_cartesian_speed_m_s=1.0,
                max_cartesian_acceleration_m_s2=100.0,
                joint_velocity_scale=1.0,
                joint_acceleration_scale=1.0,
            )
            _command, diagnostics = tracker.step(
                feedback_q_rad=np.zeros(7),
                feedback_dq_rad_s=np.zeros(7),
                ball_position_m=np.zeros(3),
                ball_velocity_m_s=np.array([0.4, 0.0, 0.0]),
                height_offset_m=0.0,
                use_mpc=True,
                mpc_tuning=tuning,
            )
            return diagnostics["commandVelocity"][0]

        without_velocity_cost = command_velocity(0.0)
        with_velocity_cost = command_velocity(100.0)

        self.assertLess(abs(without_velocity_cost), 1.0e-6)
        self.assertGreater(with_velocity_cost, 0.35)

    def test_target_is_clamped_inside_cartesian_limits(self):
        tracker = make_tracker()
        q = np.zeros(7)
        tracker.reset(q)

        _command, diagnostics = tracker.step(
            feedback_q_rad=q,
            feedback_dq_rad_s=np.zeros(7),
            ball_position_m=np.array([2.0, 0.0, 0.5]),
            ball_velocity_m_s=np.array([1.0, 0.0, 0.0]),
            height_offset_m=0.2,
            use_mpc=True,
            position_limits_m=((-0.5, 0.5), None, (0.0, 1.0)),
        )

        self.assertTrue(diagnostics["targetClipped"])
        self.assertAlmostEqual(diagnostics["target"][0], 0.5)
        self.assertTrue(diagnostics["saturated"])

    def test_velocity_and_acceleration_limits_are_shared_by_both_solvers(self):
        for use_mpc in (False, True):
            tracker = make_tracker(
                joint_velocity_scale=0.2,
                joint_acceleration_scale=0.1,
            )
            q = np.zeros(7)
            tracker.reset(q)
            command, diagnostics = tracker.step(
                feedback_q_rad=q,
                feedback_dq_rad_s=np.zeros(7),
                ball_position_m=np.array([2.0, 2.0, 2.2]),
                ball_velocity_m_s=np.zeros(3),
                height_offset_m=0.2,
                use_mpc=use_mpc,
            )
            max_delta = 100.0 * 0.1 * 0.01 * 0.01
            self.assertLessEqual(float(np.max(np.abs(command - q))), max_delta + 1e-8)
            self.assertTrue(diagnostics["accelerationLimited"])

    def test_unavailable_fk_is_a_safe_error(self):
        tracker = make_tracker()
        tracker.kinematics = lambda _q: (
            np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]),
            np.zeros((3, 7)), np.zeros((3, 7)), False)
        tracker.reset(np.zeros(7))

        with self.assertRaisesRegex(RuntimeError, "FK/Jacobian"):
            tracker.step(
                feedback_q_rad=np.zeros(7),
                feedback_dq_rad_s=np.zeros(7),
                ball_position_m=np.array([0.0, 0.0, 0.5]),
                ball_velocity_m_s=np.zeros(3),
                height_offset_m=0.2,
                use_mpc=False,
            )

    def test_both_solvers_share_cartesian_speed_limit(self):
        for use_mpc in (False, True):
            tracker = make_tracker(
                max_cartesian_speed_m_s=0.1,
                joint_velocity_scale=1.0,
                joint_acceleration_scale=1.0,
            )
            tracker.joint_acceleration_limit[:] = 1.0e6
            q = np.zeros(7)
            tracker.reset(q)
            _command, diagnostics = tracker.step(
                feedback_q_rad=q,
                feedback_dq_rad_s=np.zeros(7),
                ball_position_m=np.array([1.0, 0.0, 0.2]),
                ball_velocity_m_s=np.zeros(3),
                height_offset_m=0.0,
                use_mpc=use_mpc,
            )
            self.assertLessEqual(
                diagnostics["cartesianSpeedMps"], 0.1 + 1e-9)

    def test_joint_clipping_cannot_break_cartesian_speed_limit(self):
        tracker = RacketBallTracker(
            kinematics=RedundantCancellationKinematics(),
            nominal_q_rad=np.array([2.0, 2.0]),
            joint_lower_rad=np.full(2, -3.0),
            joint_upper_rad=np.full(2, 3.0),
            joint_velocity_limit_rad_s=np.array([0.1, 1.0]),
            joint_acceleration_limit_rad_s2=np.full(2, 1.0e6),
            config=RacketBallTrackerConfig(
                dt=0.01,
                position_gain_s=1.0,
                dls_damping=0.01,
                max_cartesian_speed_m_s=0.2,
                max_cartesian_acceleration_m_s2=1.0e6,
                joint_velocity_scale=1.0,
                joint_acceleration_scale=1.0,
                posture_gain_s=10.0,
                target_boundary_margin_m=0.0,
            ),
        )
        tracker.reset(np.zeros(2), np.zeros(2))

        command, diagnostics = tracker.step(
            feedback_q_rad=np.zeros(2),
            feedback_dq_rad_s=np.zeros(2),
            ball_position_m=np.zeros(3),
            ball_velocity_m_s=np.zeros(3),
            height_offset_m=0.0,
            use_mpc=False,
        )

        self.assertLessEqual(
            diagnostics["cartesianSpeedMps"], 0.2 + 1e-9)
        self.assertTrue(diagnostics["velocityLimited"])
        self.assertTrue(diagnostics["finalCartesianSpeedLimited"])
        self.assertLessEqual(abs(float(command[0])), 0.001 + 1e-12)

    def test_hard_position_and_cartesian_limits_override_infeasible_accel(self):
        tracker = RacketBallTracker(
            kinematics=RedundantCancellationKinematics(),
            nominal_q_rad=np.zeros(2),
            joint_lower_rad=np.full(2, -3.0),
            joint_upper_rad=np.full(2, 3.0),
            joint_velocity_limit_rad_s=np.ones(2),
            joint_acceleration_limit_rad_s2=np.ones(2),
            config=RacketBallTrackerConfig(
                dt=0.01,
                position_gain_s=1.0,
                dls_damping=0.01,
                max_cartesian_speed_m_s=0.05,
                joint_velocity_scale=1.0,
                joint_acceleration_scale=1.0,
                posture_gain_s=0.0,
                target_boundary_margin_m=0.0,
            ),
        )
        q = np.array([3.0, 0.0])
        previous_dq = np.array([0.1, 0.1])
        tracker.reset(q, previous_dq)
        tracker._dls_velocity = lambda *_args: np.array([0.1, 1.0])

        command, diagnostics = tracker.step(
            feedback_q_rad=q,
            feedback_dq_rad_s=previous_dq,
            ball_position_m=np.zeros(3),
            ball_velocity_m_s=np.zeros(3),
            height_offset_m=0.0,
            use_mpc=False,
        )

        self.assertLessEqual(command[0], 3.0)
        self.assertLessEqual(
            diagnostics["cartesianSpeedMps"], 0.05 + 1e-9)
        self.assertTrue(diagnostics["positionLimited"])
        self.assertTrue(diagnostics["accelerationConstraintOverridden"])

    def test_outward_feedforward_is_removed_at_workspace_boundary(self):
        tracker = make_tracker(max_cartesian_speed_m_s=2.0)
        q = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
        tracker.reset(q)

        command, diagnostics = tracker.step(
            feedback_q_rad=q,
            feedback_dq_rad_s=np.zeros(7),
            ball_position_m=np.array([0.5, 0.0, 0.5]),
            ball_velocity_m_s=np.array([1.0, 0.0, 0.0]),
            height_offset_m=0.2,
            use_mpc=False,
            position_limits_m=((-0.5, 0.5), None, None),
        )

        self.assertLessEqual(command[0], 0.5 + 1e-9)
        self.assertTrue(diagnostics["targetClipped"])

    def test_fixed_feedback_cannot_wind_command_to_joint_limit(self):
        tracker = make_tracker(max_command_feedback_error_rad=0.05)
        feedback = np.zeros(7)
        tracker.reset(feedback)

        diagnostics = None
        for _ in range(300):
            _command, diagnostics = tracker.step(
                feedback_q_rad=feedback,
                feedback_dq_rad_s=np.zeros(7),
                ball_position_m=np.array([0.2, 0.0, 0.2]),
                ball_velocity_m_s=np.zeros(3),
                height_offset_m=0.0,
                use_mpc=False,
            )

        self.assertLessEqual(abs(float(tracker.command_q[0])), 0.06)
        self.assertTrue(diagnostics["commandReanchored"] or abs(
            float(tracker.command_q[0])) <= 0.05)

    def test_nominal_anti_windup_reanchors_tracker_state(self):
        tracker = make_tracker()
        tracker.reset(np.zeros(7))
        tracker.command_q[0] = 0.2
        corrected = np.zeros(7)

        tracker.observe_nominal_applied(corrected, np.zeros(7))

        np.testing.assert_allclose(tracker.command_q, corrected)
        self.assertTrue(tracker.last_diagnostics["antiWindupApplied"])


class TrackerConfigValidationTest(unittest.TestCase):

    def test_legacy_config_maps_to_shared_runtime_tuning(self):
        config = RacketBallTrackerConfig(
            max_cartesian_speed_m_s=0.3,
            max_cartesian_acceleration_m_s2=4.0,
            joint_velocity_scale=0.4,
            joint_acceleration_scale=0.25,
            mpc_horizon_steps=7,
            mpc_position_weight=2.0,
            mpc_target_velocity_weight=3.0,
            mpc_velocity_weight=0.4,
            mpc_smooth_weight=0.5,
            mpc_posture_weight=0.6,
        )

        tuning = RacketMpcTuning.from_tracker_config(config)

        self.assertEqual(tuning.horizon_steps, 7)
        self.assertAlmostEqual(tuning.position_weight, 2.0)
        self.assertAlmostEqual(tuning.velocity_weight, 3.0)
        self.assertAlmostEqual(tuning.effort_weight, 0.4)
        self.assertAlmostEqual(tuning.smooth_weight, 0.5)
        self.assertAlmostEqual(tuning.posture_weight, 0.6)
        self.assertAlmostEqual(tuning.max_cartesian_speed_m_s, 0.3)
        self.assertAlmostEqual(tuning.max_cartesian_acceleration_m_s2, 4.0)
        self.assertAlmostEqual(tuning.joint_velocity_scale, 0.4)
        self.assertAlmostEqual(tuning.joint_acceleration_scale, 0.25)

    def test_runtime_tuning_rejects_invalid_values(self):
        invalid_overrides = (
            {"horizon_steps": True},
            {"horizon_steps": 3.0},
            {"horizon_steps": 0},
            {"horizon_steps": 129},
            {"position_weight": 0.0},
            {"velocity_weight": -1.0},
            {"effort_weight": float("inf")},
            {"smooth_weight": float("nan")},
            {"posture_weight": -1.0},
            {"max_cartesian_speed_m_s": 0.0},
            {"max_cartesian_acceleration_m_s2": float("inf")},
            {"joint_velocity_scale": -0.01},
            {"joint_velocity_scale": 1.01},
            {"joint_acceleration_scale": 0.0},
            {"joint_acceleration_scale": 1.01},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                RacketMpcTuning(**overrides)

    def test_runtime_tuning_is_immutable(self):
        tuning = RacketMpcTuning()
        with self.assertRaises(AttributeError):
            tuning.horizon_steps = 4

    def test_joint_limit_scales_cannot_exceed_hardware_limits(self):
        with self.assertRaises(ValueError):
            RacketBallTrackerConfig(joint_velocity_scale=1.01)
        with self.assertRaises(ValueError):
            RacketBallTrackerConfig(joint_acceleration_scale=0.0)
        with self.assertRaises(ValueError):
            RacketBallTrackerConfig(mpc_horizon_steps=129)


if __name__ == "__main__":
    unittest.main()
