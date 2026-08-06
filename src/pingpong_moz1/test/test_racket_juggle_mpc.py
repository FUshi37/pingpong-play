#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.racket_ball_tracker import RacketMpcTuning  # noqa: E402
from pingpong_controller.racket_juggle_mpc import (  # noqa: E402
    GRAVITY_M_S2,
    JUGGLE_INITIAL_Q_DEG,
    JUGGLE_INITIAL_Q_RAD,
    JugglePhysicsConfig,
    RacketJuggleMpc,
    launch_profile_for_seed_apex,
    predict_descending_impact,
    required_paddle_velocity,
    select_feasible_ball_intercept,
    solve_cartesian_first_exit_mpc,
)


def make_tuning(**overrides):
    values = {
        "horizon_steps": 12,
        "position_weight": 10.0,
        "velocity_weight": 5.0,
        "effort_weight": 1.0e-3,
        "smooth_weight": 1.0e-3,
        "posture_weight": 0.0,
        "max_cartesian_speed_m_s": 2.0,
        "max_cartesian_acceleration_m_s2": 20.0,
        "joint_velocity_scale": 0.2,
        "joint_acceleration_scale": 0.1,
    }
    values.update(overrides)
    return RacketMpcTuning(**values)


def make_physics(**overrides):
    values = {
        "countdown_s": 0.20,
        "target_apex_height_m": 0.10,
        "seed_apex_height_m": 0.01,
        "apex_ramp_per_bounce_m": 0.02,
        "launch_peak_acceleration_m_s2": 16.0,
        "drag_coefficient_per_m": 0.0,
        "ready_settle_s": 0.10,
    }
    values.update(overrides)
    return JugglePhysicsConfig(**values)


def step_juggler(
    juggler,
    now_s,
    *,
    run_requested=False,
    start_nonce=0,
    physics=None,
    tuning=None,
    feedback_q_rad=None,
    racket_position_m=None,
    racket_velocity_m_s=None,
    ball_position_m=None,
    ball_velocity_m_s=None,
    ball_valid=True,
    ball_age_s=0.0,
    available_vertical_acceleration_m_s2=None,
):
    racket = np.array(
        [0.10, -0.30, 0.80] if racket_position_m is None
        else racket_position_m,
        dtype=np.float64,
    )
    ball = np.array(
        racket + np.array([0.0, 0.0, 0.020])
        if ball_position_m is None else ball_position_m,
        dtype=np.float64,
    )
    return juggler.step(
        wall_now_s=now_s,
        run_requested=run_requested,
        start_nonce=start_nonce,
        feedback_q_rad=(
            JUGGLE_INITIAL_Q_RAD if feedback_q_rad is None
            else feedback_q_rad
        ),
        feedback_dq_rad_s=np.zeros(7),
        racket_position_m=racket,
        racket_velocity_m_s=(
            np.zeros(3) if racket_velocity_m_s is None
            else racket_velocity_m_s
        ),
        racket_normal=np.array([0.0, 0.0, 1.0]),
        ball_position_m=ball,
        ball_velocity_m_s=(
            np.zeros(3) if ball_velocity_m_s is None
            else ball_velocity_m_s
        ),
        ball_valid=ball_valid,
        ball_age_s=ball_age_s,
        tuning=make_tuning() if tuning is None else tuning,
        physics=make_physics() if physics is None else physics,
        available_vertical_acceleration_m_s2=(
            available_vertical_acceleration_m_s2
        ),
    )


def advance_to_launch(juggler, *, physics=None, tuning=None, nonce=1):
    physics = make_physics() if physics is None else physics
    tuning = make_tuning() if tuning is None else tuning
    step_juggler(
        juggler, 0.0, physics=physics, tuning=tuning,
        run_requested=False,
    )
    countdown = step_juggler(
        juggler,
        physics.ready_settle_s + 0.01,
        physics=physics,
        tuning=tuning,
        run_requested=True,
        start_nonce=nonce,
    )
    deadline = countdown.diagnostics["countdownRemainingS"]
    if deadline is None:
        raise AssertionError("test fixture did not reach countdown")
    launch_time = (
        physics.ready_settle_s + 0.01 + deadline + 1.0e-6
    )
    launch = step_juggler(
        juggler,
        launch_time,
        physics=physics,
        tuning=tuning,
        run_requested=True,
        start_nonce=nonce,
    )
    return launch_time, launch


class InitialConfigurationTest(unittest.TestCase):

    def test_initial_joint_angles_match_requested_right_arm_pose(self):
        expected_degrees = np.array(
            [5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214]
        )

        np.testing.assert_array_equal(JUGGLE_INITIAL_Q_DEG, expected_degrees)
        np.testing.assert_allclose(
            JUGGLE_INITIAL_Q_RAD,
            np.deg2rad(expected_degrees),
            rtol=0.0,
            atol=1.0e-14,
        )


class LaunchProfileTest(unittest.TestCase):

    def test_target_apex_cannot_be_below_launch_seed_apex(self):
        with self.assertRaisesRegex(ValueError, "target_apex_height_m"):
            make_physics(
                seed_apex_height_m=0.05,
                target_apex_height_m=0.04,
            )

    def test_separation_occurs_at_minus_gravity_and_seeds_requested_apex(self):
        seed_apex = 0.035
        profile = launch_profile_for_seed_apex(
            seed_apex,
            peak_acceleration_m_s2=16.0,
        )

        _displacement, separation_speed, separation_acceleration = (
            profile.sample(profile.separation_time_s)
        )

        self.assertAlmostEqual(
            separation_acceleration, -GRAVITY_M_S2, places=11)
        self.assertAlmostEqual(
            separation_speed ** 2 / (2.0 * GRAVITY_M_S2),
            seed_apex,
            places=11,
        )
        self.assertAlmostEqual(
            profile.predicted_apex_above_separation_m,
            seed_apex,
            places=14,
        )
        self.assertLess(profile.separation_time_s, profile.half_cycle_s)

    def test_launch_rejects_parameters_that_cannot_open_contact(self):
        for peak_acceleration in (
            -1.0,
            0.0,
            GRAVITY_M_S2,
            GRAVITY_M_S2 - 1.0e-6,
        ):
            with self.subTest(peak_acceleration=peak_acceleration):
                with self.assertRaisesRegex(ValueError, "exceed gravity"):
                    launch_profile_for_seed_apex(
                        0.01,
                        peak_acceleration_m_s2=peak_acceleration,
                    )
        with self.assertRaisesRegex(ValueError, "seed_apex_height_m"):
            launch_profile_for_seed_apex(
                0.0,
                peak_acceleration_m_s2=16.0,
            )

    def test_state_machine_faults_when_launch_exceeds_shared_acceleration(self):
        physics = make_physics(
            countdown_s=0.0,
            launch_peak_acceleration_m_s2=16.0,
        )
        tuning = make_tuning(max_cartesian_acceleration_m_s2=10.0)
        juggler = RacketJuggleMpc(control_dt_s=0.005)

        step_juggler(
            juggler, 0.0, physics=physics, tuning=tuning,
            run_requested=False,
        )
        countdown = step_juggler(
            juggler,
            physics.ready_settle_s + 0.01,
            physics=physics,
            tuning=tuning,
            run_requested=True,
            start_nonce=3,
        )
        self.assertEqual(countdown.diagnostics["state"], "countdown")
        plan = step_juggler(
            juggler,
            physics.ready_settle_s + 0.011,
            physics=physics,
            tuning=tuning,
            run_requested=True,
            start_nonce=3,
        )

        self.assertEqual(plan.control_kind, "hold")
        self.assertEqual(plan.diagnostics["state"], "fault_hold")
        self.assertTrue(plan.diagnostics["requiresRearm"])
        self.assertIn("peak acceleration", plan.diagnostics["fault"])


class BallPredictionTest(unittest.TestCase):

    def test_drag_free_prediction_matches_closed_form_descending_crossing(self):
        position = np.array([0.10, -0.20, 1.00])
        velocity = np.array([0.30, -0.10, 2.00])
        contact_z = 0.50
        discriminant = velocity[2] ** 2 + (
            2.0 * GRAVITY_M_S2 * (position[2] - contact_z)
        )
        expected_time = (
            velocity[2] + np.sqrt(discriminant)
        ) / GRAVITY_M_S2
        expected_position = position + velocity * expected_time
        expected_position[2] = contact_z
        expected_velocity = velocity + np.array(
            [0.0, 0.0, -GRAVITY_M_S2 * expected_time]
        )

        prediction = predict_descending_impact(
            position,
            velocity,
            contact_z,
            drag_coefficient_per_m=0.0,
            max_time_s=2.0,
            integration_dt_s=0.0002,
        )

        self.assertIsNotNone(prediction)
        self.assertAlmostEqual(prediction.time_s, expected_time, places=7)
        np.testing.assert_allclose(
            prediction.position_m, expected_position, atol=1.0e-7)
        np.testing.assert_allclose(
            prediction.velocity_m_s, expected_velocity, atol=1.0e-7)
        self.assertLess(prediction.velocity_m_s[2], 0.0)

    def test_ball_already_below_plane_and_falling_has_no_future_contact(self):
        prediction = predict_descending_impact(
            np.array([0.0, 0.0, 0.49]),
            np.array([0.0, 0.0, -0.1]),
            0.50,
        )

        self.assertIsNone(prediction)


class ImpactModelTest(unittest.TestCase):

    def test_required_paddle_velocity_inverts_contact_model(self):
        incoming = np.array([0.35, -0.22, -1.80])
        desired_outgoing = np.array([-0.12, 0.18, 1.25])
        normal = np.array([0.08, -0.04, 0.996])
        normal = normal / np.linalg.norm(normal)
        restitution = 0.61
        tangential_transfer = 0.28

        paddle = required_paddle_velocity(
            incoming,
            desired_outgoing,
            normal,
            restitution=restitution,
            tangential_transfer=tangential_transfer,
        )

        incoming_normal = float(normal @ incoming)
        paddle_normal = float(normal @ paddle)
        reconstructed_normal = (
            -restitution * incoming_normal
            + (1.0 + restitution) * paddle_normal
        )
        incoming_tangent = incoming - incoming_normal * normal
        paddle_tangent = paddle - paddle_normal * normal
        reconstructed_tangent = (
            (1.0 - tangential_transfer) * incoming_tangent
            + tangential_transfer * paddle_tangent
        )
        reconstructed = (
            reconstructed_normal * normal + reconstructed_tangent
        )

        np.testing.assert_allclose(
            reconstructed, desired_outgoing, atol=1.0e-12)


class InterceptSelectionTest(unittest.TestCase):

    def test_launch_recovery_finds_a_limit_feasible_moving_contact(self):
        physics = JugglePhysicsConfig()
        tuning = make_tuning(
            horizon_steps=24,
            max_cartesian_speed_m_s=0.65,
            max_cartesian_acceleration_m_s2=16.0,
        )
        profile = launch_profile_for_seed_apex(
            physics.seed_apex_height_m,
            physics.launch_peak_acceleration_m_s2,
        )
        transition_time = 1.20 * profile.half_cycle_s
        separation_displacement, separation_speed, _ = profile.sample(
            profile.separation_time_s)
        flight_time = transition_time - profile.separation_time_s
        launch_origin = np.array([0.0, 0.0, 0.80])
        # FK of the user-provided seven-joint placement pose.  Its roughly
        # two-degree tilt is essential: an ideal [0, 0, 1] test hides the
        # first-bounce tangential feasibility problem.
        racket_normal = np.array(
            [-0.005836, 0.035625, 0.999348], dtype=np.float64)
        racket_normal /= np.linalg.norm(racket_normal)
        racket_displacement, racket_speed, racket_acceleration = (
            profile.sample(transition_time))
        racket_position = (
            launch_origin + racket_displacement * racket_normal)
        racket_velocity = racket_speed * racket_normal
        separation_position = (
            launch_origin
            + (separation_displacement + physics.ball_radius_m)
            * racket_normal)
        ball_position = (
            separation_position
            + separation_speed * flight_time * racket_normal
            + np.array([
                0.0, 0.0,
                -0.5 * GRAVITY_M_S2 * flight_time ** 2,
            ]))
        ball_velocity = (
            separation_speed * racket_normal
            + np.array([0.0, 0.0, -GRAVITY_M_S2 * flight_time]))

        candidate = select_feasible_ball_intercept(
            ball_position,
            ball_velocity,
            racket_position,
            racket_velocity,
            racket_normal,
            launch_origin,
            physics.seed_apex_height_m,
            physics=physics,
            tuning=tuning,
            minimum_time_s=physics.min_impact_lead_s,
            lateral_correction_fraction=0.0,
            position_limits_m=((-0.5, 0.5), (-0.5, 0.5), (0.69, 1.46)),
        )

        self.assertIsNotNone(candidate)
        self.assertLess(candidate.racket_position_m[2], racket_position[2])
        self.assertLess(
            candidate.cubic_peak_acceleration_m_s2,
            tuning.max_cartesian_acceleration_m_s2,
        )
        incoming_tangent = (
            candidate.ball_velocity_m_s
            - float(racket_normal @ candidate.ball_velocity_m_s)
            * racket_normal)
        outgoing_tangent = (
            candidate.outgoing_ball_velocity_m_s
            - float(racket_normal @ candidate.outgoing_ball_velocity_m_s)
            * racket_normal)
        np.testing.assert_allclose(
            outgoing_tangent, incoming_tangent, atol=1.0e-12)
        solution = solve_cartesian_first_exit_mpc(
            racket_position,
            racket_velocity,
            candidate.racket_position_m,
            candidate.racket_velocity_m_s,
            candidate.time_s,
            0.005,
            tuning,
            previous_acceleration_m_s2=(
                racket_acceleration * racket_normal),
        )
        self.assertFalse(solution.saturated)
        self.assertLess(solution.terminal_position_error_m, 1.0e-9)
        self.assertLess(solution.terminal_velocity_error_m_s, 1.0e-9)

        # The requested +25 mm second-bounce ramp is slightly too aggressive
        # under the same 16 m/s² bound.  The controller's feasibility ramp
        # must find a lower event and the bounded MPC must keep its hard
        # terminal equalities instead of clipping an unconstrained solution.
        requested_second_apex = (
            physics.seed_apex_height_m
            + physics.apex_ramp_per_bounce_m)
        requested_lateral_fraction = (
            (requested_second_apex - physics.seed_apex_height_m)
            / (physics.target_apex_height_m
               - physics.seed_apex_height_m))
        trial_scales = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
        rebound_profiles = tuple(
            (
                physics.seed_apex_height_m
                + scale * (
                    requested_second_apex
                    - physics.seed_apex_height_m),
                scale * requested_lateral_fraction,
            )
            for scale in trial_scales)
        second_candidate = select_feasible_ball_intercept(
            candidate.ball_position_m,
            candidate.outgoing_ball_velocity_m_s,
            candidate.racket_position_m,
            candidate.racket_velocity_m_s,
            racket_normal,
            launch_origin,
            rebound_profiles[0][0],
            physics=physics,
            tuning=tuning,
            minimum_time_s=physics.min_impact_lead_s,
            lateral_correction_fraction=rebound_profiles[0][1],
            position_limits_m=(
                (-0.5, 0.5), (-0.5, 0.5), (0.69, 1.46)),
            rebound_profiles=rebound_profiles,
        )
        self.assertIsNotNone(second_candidate)
        selected_scale = trial_scales[
            second_candidate.rebound_profile_index]
        applied_second_apex = rebound_profiles[
            second_candidate.rebound_profile_index][0]
        self.assertAlmostEqual(selected_scale, 0.6)
        self.assertLess(applied_second_apex, requested_second_apex)
        second_solution = solve_cartesian_first_exit_mpc(
            candidate.racket_position_m,
            candidate.racket_velocity_m_s,
            second_candidate.racket_position_m,
            second_candidate.racket_velocity_m_s,
            second_candidate.time_s,
            0.005,
            tuning,
        )
        self.assertTrue(second_solution.constraint_active)
        self.assertFalse(second_solution.saturated)
        self.assertLess(second_solution.terminal_position_error_m, 1.0e-9)
        self.assertLess(second_solution.terminal_velocity_error_m_s, 1.0e-9)


class CartesianMpcTest(unittest.TestCase):

    def test_full_trajectory_workspace_violation_is_fail_closed(self):
        tuning = make_tuning(
            horizon_steps=12,
            max_cartesian_speed_m_s=0.65,
            max_cartesian_acceleration_m_s2=16.0,
        )
        position = np.array([0.0, 0.0, 1.455])

        solution = solve_cartesian_first_exit_mpc(
            position,
            np.array([0.0, 0.0, 0.6]),
            position,
            np.array([0.0, 0.0, -0.6]),
            0.2,
            0.005,
            tuning,
            position_limits_m=(None, None, (0.692, 1.462)),
        )

        self.assertTrue(solution.saturated)

    def test_roundoff_at_active_constraint_is_not_reported_as_saturation(self):
        tuning = RacketMpcTuning(
            horizon_steps=24,
            position_weight=1.0,
            velocity_weight=1.0,
            effort_weight=1.0e-4,
            smooth_weight=2.0e-4,
            posture_weight=1.0e-4,
            max_cartesian_speed_m_s=0.65,
            max_cartesian_acceleration_m_s2=16.0,
            joint_velocity_scale=0.25,
            joint_acceleration_scale=0.35,
        )
        solution = solve_cartesian_first_exit_mpc(
            np.array([
                -0.02109262559431819,
                0.020370402118252487,
                0.9838813838104594,
            ]),
            np.array([
                -0.12331058581197532,
                0.009924307767715171,
                0.07839769488445429,
            ]),
            np.array([
                0.009725574857214118,
                0.01105942020428435,
                1.0003035238415214,
            ]),
            np.array([
                -0.09491121671183879,
                -0.10665339798253956,
                0.10877704769601736,
            ]),
            0.14000000000000004,
            0.005,
            tuning,
        )

        self.assertTrue(solution.constraint_active)
        self.assertFalse(solution.saturated)
        self.assertLess(solution.terminal_position_error_m, 1.0e-9)
        self.assertLess(solution.terminal_velocity_error_m_s, 1.0e-9)

    def test_shifted_feasible_tail_can_anchor_a_locked_event(self):
        position = np.array([
            -0.0073171919238631256,
            0.08088489485440695,
            -0.05072885989442895,
        ])
        velocity = np.array([
            0.04373320529391517,
            -0.03812625527332747,
            0.07888526262807324,
        ])
        target_position = np.array([
            0.007710476594998366,
            0.06172850257703265,
            -0.03641510825973204,
        ])
        target_velocity = np.array([
            -0.13949758654768799,
            -0.16489217072818216,
            0.08752307369625653,
        ])
        feasible_tail = np.array([
            [2.6978413234452354, 0.37842926371446195,
             0.30597954499657426],
            [-0.39378205323689686, -2.4882362157228766,
             -0.31016698989999997],
            [-2.961240961414498, 0.49638730576738704,
             -0.9693031222386532],
            [-3.0074341456259033, -0.9218986628560663,
             1.1462467885057446],
        ])
        tuning = RacketMpcTuning(
            horizon_steps=4,
            position_weight=1.876230505180286,
            velocity_weight=0.2720186262013206,
            effort_weight=0.0007340682927809703,
            smooth_weight=165.15264228807368,
            posture_weight=0.0,
            max_cartesian_speed_m_s=0.8,
            max_cartesian_acceleration_m_s2=4.0,
            joint_velocity_scale=0.2,
            joint_acceleration_scale=0.2,
        )

        solution = solve_cartesian_first_exit_mpc(
            position,
            velocity,
            target_position,
            target_velocity,
            0.20,
            0.05,
            tuning,
            feasible_acceleration_hint_m_s2=feasible_tail,
            position_limits_m=((-1.0, 1.0),) * 3,
        )

        self.assertTrue(solution.tail_anchor_used)
        self.assertTrue(solution.constraint_active)
        self.assertFalse(solution.saturated)
        self.assertLess(solution.terminal_position_error_m, 1.0e-9)
        self.assertLess(solution.terminal_velocity_error_m_s, 1.0e-9)

    def test_solution_reduces_terminal_position_and_velocity_error(self):
        current_position = np.zeros(3)
        current_velocity = np.zeros(3)
        target_position = np.array([0.20, -0.10, 0.15])
        target_velocity = np.array([0.10, 0.00, 0.20])
        duration = 0.50
        tuning = make_tuning(
            max_cartesian_speed_m_s=10.0,
            max_cartesian_acceleration_m_s2=100.0,
        )

        solution = solve_cartesian_first_exit_mpc(
            current_position,
            current_velocity,
            target_position,
            target_velocity,
            duration,
            0.01,
            tuning,
        )

        uncontrolled_terminal_position = (
            current_position + duration * current_velocity
        )
        self.assertLess(
            np.linalg.norm(solution.terminal_position_m - target_position),
            np.linalg.norm(
                uncontrolled_terminal_position - target_position),
        )
        self.assertLess(
            np.linalg.norm(solution.terminal_velocity_m_s - target_velocity),
            np.linalg.norm(current_velocity - target_velocity),
        )
        self.assertLess(solution.terminal_position_error_m, 1.0e-10)
        self.assertLess(solution.terminal_velocity_error_m_s, 1.0e-10)
        self.assertLessEqual(
            np.linalg.norm(solution.first_acceleration_m_s2),
            tuning.max_cartesian_acceleration_m_s2 + 1.0e-12,
        )

    def test_acceleration_is_projected_to_shared_cartesian_limit(self):
        tuning = make_tuning(
            horizon_steps=8,
            position_weight=100.0,
            velocity_weight=100.0,
            effort_weight=1.0e-9,
            smooth_weight=0.0,
            max_cartesian_speed_m_s=5.0,
            max_cartesian_acceleration_m_s2=0.50,
        )

        solution = solve_cartesian_first_exit_mpc(
            np.zeros(3),
            np.zeros(3),
            np.array([2.0, -2.0, 1.0]),
            np.array([1.0, -1.0, 0.5]),
            0.20,
            0.01,
            tuning,
        )

        self.assertTrue(solution.saturated)
        self.assertAlmostEqual(
            np.linalg.norm(solution.first_acceleration_m_s2),
            tuning.max_cartesian_acceleration_m_s2,
            places=12,
        )


class JuggleStateMachineTest(unittest.TestCase):

    def test_default_state_positions_arm_without_starting_juggle(self):
        juggler = RacketJuggleMpc(control_dt_s=0.005)

        plan = step_juggler(juggler, 0.0, run_requested=False)

        self.assertEqual(plan.control_kind, "joint")
        np.testing.assert_allclose(
            plan.joint_target_q_rad, JUGGLE_INITIAL_Q_RAD)
        self.assertEqual(plan.diagnostics["state"], "ready_wait")
        self.assertFalse(plan.diagnostics["active"])
        self.assertFalse(plan.diagnostics["runRequested"])
        self.assertIsNone(plan.diagnostics["countdownRemainingS"])

    def test_stable_placement_starts_countdown_and_repeated_nonce_does_not_restart(self):
        physics = make_physics(countdown_s=0.50, ready_settle_s=0.10)
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        step_juggler(
            juggler, 0.0, physics=physics, run_requested=False)

        first = step_juggler(
            juggler,
            0.11,
            physics=physics,
            run_requested=True,
            start_nonce=17,
        )
        repeated = step_juggler(
            juggler,
            0.21,
            physics=physics,
            run_requested=True,
            start_nonce=17,
        )

        self.assertEqual(first.diagnostics["state"], "countdown")
        self.assertTrue(first.diagnostics["ready"])
        self.assertAlmostEqual(
            first.diagnostics["countdownRemainingS"], 0.50)
        self.assertEqual(repeated.diagnostics["state"], "countdown")
        self.assertAlmostEqual(
            repeated.diagnostics["countdownRemainingS"], 0.40)
        self.assertAlmostEqual(juggler.countdown_deadline_s, 0.61)

    def test_countdown_enters_launch_and_verified_separation_enters_flight(self):
        physics = make_physics()
        tuning = make_tuning()
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        launch_time, launch = advance_to_launch(
            juggler, physics=physics, tuning=tuning, nonce=4)

        self.assertEqual(launch.control_kind, "cartesian")
        self.assertEqual(launch.diagnostics["state"], "launch")
        self.assertFalse(launch.diagnostics["separationDetected"])
        self.assertAlmostEqual(
            launch.diagnostics["launchProfile"]["predictedSeedApexM"],
            physics.seed_apex_height_m,
        )

        profile = juggler.launch_profile
        self.assertIsNotNone(profile)
        still_recovering = step_juggler(
            juggler,
            launch_time + profile.half_cycle_s + 1.0e-4,
            physics=physics,
            tuning=tuning,
            run_requested=True,
            start_nonce=4,
            ball_position_m=np.array([0.10, -0.30, 0.85]),
            ball_velocity_m_s=np.array([0.0, 0.0, 0.30]),
        )
        separated = step_juggler(
            juggler,
            launch_time + 1.20 * profile.half_cycle_s + 1.0e-4,
            physics=physics,
            tuning=tuning,
            run_requested=True,
            start_nonce=4,
            ball_position_m=np.array([0.10, -0.30, 0.85]),
            ball_velocity_m_s=np.array([0.0, 0.0, 0.20]),
        )

        self.assertEqual(still_recovering.diagnostics["state"], "launch")
        self.assertLess(still_recovering.target_velocity_m_s[2], 0.0)
        self.assertEqual(separated.diagnostics["state"], "flight")
        self.assertTrue(separated.diagnostics["separationDetected"])
        self.assertTrue(separated.diagnostics["active"])

    def test_stop_while_armed_wait_cannot_start_a_later_countdown(self):
        physics = make_physics(ready_settle_s=0.05)
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        step_juggler(
            juggler, 0.0, physics=physics, run_requested=False,
            ball_valid=False)
        armed = step_juggler(
            juggler, 0.01, physics=physics, run_requested=True,
            start_nonce=21, ball_valid=False)
        stopped = step_juggler(
            juggler, 0.02, physics=physics, run_requested=False,
            start_nonce=21, ball_valid=False)
        later = step_juggler(
            juggler, 0.20, physics=physics, run_requested=False,
            start_nonce=21)

        self.assertEqual(armed.diagnostics["state"], "armed_wait")
        self.assertEqual(stopped.diagnostics["state"], "stopped_hold")
        self.assertEqual(later.diagnostics["state"], "stopped_hold")
        self.assertIsNone(later.diagnostics["countdownRemainingS"])

    def test_mode_reentry_requires_off_and_a_new_start_nonce(self):
        physics = make_physics(ready_settle_s=0.05)
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        step_juggler(juggler, 0.0, physics=physics, run_requested=False)
        started = step_juggler(
            juggler, 0.06, physics=physics, run_requested=True,
            start_nonce=7)
        self.assertEqual(started.diagnostics["state"], "countdown")

        juggler.deactivate("test")
        juggler.enter_mode()
        stale_first = step_juggler(
            juggler, 1.0, physics=physics, run_requested=True,
            start_nonce=7)
        stale_later = step_juggler(
            juggler, 1.2, physics=physics, run_requested=True,
            start_nonce=7)
        self.assertIn(
            stale_first.diagnostics["state"], {"positioning", "ready_wait"})
        self.assertEqual(stale_later.diagnostics["state"], "ready_wait")
        self.assertFalse(stale_later.diagnostics["startEdgeArmed"])

        step_juggler(
            juggler, 1.21, physics=physics, run_requested=False,
            start_nonce=7)
        restarted = step_juggler(
            juggler, 1.22, physics=physics, run_requested=True,
            start_nonce=8)
        self.assertEqual(restarted.diagnostics["state"], "countdown")

    def test_contact_window_tolerates_delayed_descending_observation(self):
        physics = make_physics(contact_grace_s=0.06)
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        juggler.state = "flight"
        juggler.run_requested = True
        juggler.launch_origin_m = np.array([0.10, -0.30, 0.80])
        juggler.launch_normal = np.array([0.0, 0.0, 1.0])
        juggler.expected_impact_s = 2.01
        juggler.contact_target_position_m = np.array([0.10, -0.30, 0.80])
        juggler.contact_target_velocity_m_s = np.zeros(3)
        juggler.contact_ball_position_m = np.array([0.10, -0.30, 0.82])
        juggler.previous_ball_velocity_m_s = np.array([0.0, 0.0, -0.50])

        entering = step_juggler(
            juggler, 2.0, physics=physics, run_requested=True,
            start_nonce=1,
            ball_position_m=np.array([0.10, -0.30, 0.821]),
            ball_velocity_m_s=np.array([0.0, 0.0, -0.45]))
        delayed = step_juggler(
            juggler, 2.03, physics=physics, run_requested=True,
            start_nonce=1,
            ball_position_m=np.array([0.10, -0.30, 0.815]),
            ball_velocity_m_s=np.array([0.0, 0.0, -0.20]))

        self.assertEqual(entering.diagnostics["state"], "contact_wait")
        self.assertEqual(delayed.control_kind, "cartesian")
        self.assertEqual(delayed.diagnostics["state"], "contact_wait")
        self.assertTrue(delayed.diagnostics["contactGraceActive"])
        self.assertIsNone(delayed.diagnostics["fault"])

    def test_new_planned_impact_is_not_hidden_by_previous_bounce_refractory(self):
        physics = make_physics(contact_grace_s=0.06)
        tuning = make_tuning()
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        juggler.state = "contact_wait"
        juggler.run_requested = True
        juggler.launch_origin_m = np.array([0.10, -0.30, 0.80])
        juggler.launch_normal = np.array([0.0, 0.0, 1.0])
        juggler.expected_impact_s = 0.110
        juggler.contact_target_position_m = np.array([0.10, -0.30, 0.80])
        juggler.contact_target_velocity_m_s = np.array([0.0, 0.0, 0.10])
        juggler.contact_ball_position_m = np.array([0.10, -0.30, 0.82])
        juggler.contact_outgoing_ball_velocity_m_s = np.array(
            [0.0, 0.0, 0.45])
        juggler.contact_apex_height_m = 0.03
        juggler.contact_lateral_correction_fraction = 0.2
        juggler.flight_profile_scale_cap = 0.6
        juggler.previous_ball_velocity_m_s = np.array([0.0, 0.0, -0.45])
        juggler.last_impact_s = 0.040
        juggler.bounce_count = 1

        plan = step_juggler(
            juggler,
            0.110,
            run_requested=True,
            start_nonce=1,
            physics=physics,
            tuning=tuning,
            racket_position_m=np.array([0.10, -0.30, 0.80]),
            racket_velocity_m_s=np.array([0.0, 0.0, 0.10]),
            ball_position_m=np.array([0.10, -0.30, 0.82]),
            ball_velocity_m_s=np.array([0.0, 0.0, 0.45]),
        )

        self.assertEqual(plan.diagnostics["bounceCount"], 2)
        self.assertEqual(plan.diagnostics["state"], "flight")
        self.assertIsNone(plan.diagnostics["fault"])
        self.assertAlmostEqual(juggler.last_impact_s, 0.110)

    def test_replanning_cannot_upgrade_profile_before_the_same_impact(self):
        physics = make_physics()
        tuning = make_tuning()
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        juggler.state = "flight"
        juggler.run_requested = True
        juggler.launch_origin_m = np.array([0.10, -0.30, 0.80])
        juggler.launch_normal = np.array([0.0, 0.0, 1.0])
        juggler.bounce_count = 1
        juggler.last_impact_s = 0.0
        juggler.flight_profile_scale_cap = 0.6
        juggler.expected_impact_s = 0.50
        juggler.contact_target_position_m = np.array([0.10, -0.30, 0.80])
        juggler.contact_target_velocity_m_s = np.zeros(3)
        juggler.contact_ball_position_m = np.array([0.10, -0.30, 0.82])
        juggler.contact_outgoing_ball_velocity_m_s = np.array(
            [0.0, 0.0, 0.45])
        juggler.contact_apex_height_m = 0.022
        juggler.contact_lateral_correction_fraction = 0.10
        juggler.previous_ball_velocity_m_s = np.array([0.0, 0.0, 0.44])

        plan = step_juggler(
            juggler,
            0.010,
            run_requested=True,
            start_nonce=1,
            physics=physics,
            tuning=tuning,
            racket_position_m=np.array([0.10, -0.30, 0.80]),
            racket_velocity_m_s=np.zeros(3),
            ball_position_m=np.array([0.10, -0.30, 0.82]),
            ball_velocity_m_s=np.array([0.0, 0.0, 0.45]),
        )

        self.assertEqual(plan.diagnostics["state"], "flight")
        self.assertIsNone(plan.diagnostics["fault"])
        self.assertLessEqual(
            plan.diagnostics["apexFeasibilityScale"], 0.6)
        self.assertLessEqual(juggler.flight_profile_scale_cap, 0.6)

    def test_two_distinct_planned_impacts_can_be_closer_than_refractory(self):
        physics = JugglePhysicsConfig(impact_refractory_s=0.20)
        tuning = make_tuning(
            horizon_steps=24,
            position_weight=1.0,
            velocity_weight=1.0,
            effort_weight=1.0e-4,
            smooth_weight=2.0e-4,
            posture_weight=1.0e-4,
            max_cartesian_speed_m_s=0.65,
            max_cartesian_acceleration_m_s2=16.0,
            joint_velocity_scale=0.25,
            joint_acceleration_scale=0.35,
        )
        dt = 0.005
        normal = np.array(
            [-0.005836, 0.035625, 0.999348], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        origin = np.array([-0.114136, -0.466601, 1.059307])
        limits = ((-0.298, 0.163), (-0.502, -0.290), (0.692, 1.462))
        profile = launch_profile_for_seed_apex(
            physics.seed_apex_height_m,
            physics.launch_peak_acceleration_m_s2)
        transition = 1.20 * profile.half_cycle_s
        separation_d, separation_v, _ = profile.sample(
            profile.separation_time_s)
        flight_s = transition - profile.separation_time_s
        racket_d, racket_vn, racket_a = profile.sample(transition)
        racket = origin + racket_d * normal
        racket_v = racket_vn * normal
        ball = (
            origin
            + (separation_d + physics.ball_radius_m) * normal
            + separation_v * flight_s * normal
            + np.array([
                0.0,
                0.0,
                -0.5 * GRAVITY_M_S2 * flight_s ** 2,
            ]))
        ball_v = (
            separation_v * normal
            + np.array([0.0, 0.0, -GRAVITY_M_S2 * flight_s]))

        juggler = RacketJuggleMpc(control_dt_s=dt)
        juggler.state = "flight"
        juggler.run_requested = True
        juggler.launch_origin_m = origin.copy()
        juggler.launch_normal = normal.copy()
        juggler.separation_detected = True
        juggler.previous_ball_position_m = ball.copy()
        juggler.previous_ball_velocity_m_s = ball_v.copy()
        juggler.previous_acceleration_m_s2 = racket_a * normal
        now = 0.0
        impact_times = []

        def advance_ball(position, velocity, physical_dt_s):
            gravity = np.array([0.0, 0.0, -GRAVITY_M_S2])
            speed = float(np.linalg.norm(velocity))
            a0 = (
                gravity
                - physics.drag_coefficient_per_m * speed * velocity)
            next_position = (
                position + physical_dt_s * velocity
                + 0.5 * physical_dt_s ** 2 * a0)
            provisional = velocity + physical_dt_s * a0
            next_speed = float(np.linalg.norm(provisional))
            a1 = (
                gravity
                - physics.drag_coefficient_per_m
                * next_speed * provisional)
            return (
                next_position,
                velocity + 0.5 * physical_dt_s * (a0 + a1),
            )

        target_bounces = 20
        physical_step_pattern_s = (0.004, 0.006)
        for tick in range(2000):
            before = juggler.bounce_count
            plan = juggler.step(
                wall_now_s=now,
                run_requested=True,
                start_nonce=1,
                feedback_q_rad=JUGGLE_INITIAL_Q_RAD,
                feedback_dq_rad_s=np.zeros(7),
                racket_position_m=racket,
                racket_velocity_m_s=racket_v,
                racket_normal=normal,
                ball_position_m=ball,
                ball_velocity_m_s=ball_v,
                ball_valid=True,
                ball_age_s=0.0,
                tuning=tuning,
                physics=physics,
                position_limits_m=limits,
            )
            self.assertIsNone(plan.diagnostics.get("fault"))
            if juggler.bounce_count > before:
                impact_times.append(now)
            if juggler.bounce_count >= target_bounces:
                break
            if plan.target_position_m is not None:
                racket = plan.target_position_m.copy()
                racket_v = plan.target_velocity_m_s.copy()
            # The planner advances on the backend's 5 ms logical clock while
            # the measured ball state reflects ordinary host/ROS jitter.
            # Alternating +/-1 ms used to make the locked event falsely
            # unreachable when wall monotonic time drove the event grid.
            ball, ball_v = advance_ball(
                ball,
                ball_v,
                physical_step_pattern_s[tick % len(
                    physical_step_pattern_s)],
            )
            now += dt
            if (juggler.state == "contact_wait"
                    and juggler.expected_impact_s is not None
                    and now + 1.0e-12 >= juggler.expected_impact_s):
                # Perfect camera/impact fixture: apply the exact outgoing event
                # saved by the planner at its expected boundary.
                ball = juggler.contact_ball_position_m.copy()
                ball_v = juggler.contact_outgoing_ball_velocity_m_s.copy()
                racket = juggler.contact_target_position_m.copy()
                racket_v = juggler.contact_target_velocity_m_s.copy()

        self.assertEqual(juggler.bounce_count, target_bounces)
        self.assertEqual(len(impact_times), target_bounces)
        self.assertLess(
            impact_times[1] - impact_times[0],
            physics.impact_refractory_s)
        self.assertNotEqual(juggler.state, "fault_hold")

    def test_stop_during_active_state_holds_and_clears_request(self):
        physics = make_physics()
        tuning = make_tuning()
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        _launch_time, launch = advance_to_launch(
            juggler, physics=physics, tuning=tuning, nonce=8)
        self.assertEqual(launch.diagnostics["state"], "launch")

        stopped = step_juggler(
            juggler,
            0.50,
            physics=physics,
            tuning=tuning,
            run_requested=False,
            start_nonce=8,
        )

        self.assertEqual(stopped.control_kind, "hold")
        self.assertEqual(stopped.diagnostics["state"], "stopped_hold")
        self.assertFalse(stopped.diagnostics["active"])
        self.assertFalse(stopped.diagnostics["runRequested"])

    def test_stale_ball_during_launch_faults_and_requires_off_on_rearm(self):
        physics = make_physics()
        tuning = make_tuning()
        juggler = RacketJuggleMpc(control_dt_s=0.005)
        launch_time, launch = advance_to_launch(
            juggler, physics=physics, tuning=tuning, nonce=12)
        self.assertEqual(launch.diagnostics["state"], "launch")

        faulted = step_juggler(
            juggler,
            launch_time + 0.01,
            physics=physics,
            tuning=tuning,
            run_requested=True,
            start_nonce=12,
            ball_age_s=physics.max_ball_age_s + 0.001,
        )
        still_faulted = step_juggler(
            juggler,
            launch_time + 0.02,
            physics=physics,
            tuning=tuning,
            run_requested=True,
            start_nonce=12,
        )

        self.assertEqual(faulted.control_kind, "hold")
        self.assertEqual(faulted.diagnostics["state"], "fault_hold")
        self.assertTrue(faulted.diagnostics["requiresRearm"])
        self.assertIn("stale", faulted.diagnostics["fault"])
        self.assertEqual(still_faulted.diagnostics["state"], "fault_hold")
        self.assertTrue(still_faulted.diagnostics["requiresRearm"])

        disarmed = step_juggler(
            juggler,
            launch_time + 0.03,
            physics=physics,
            tuning=tuning,
            run_requested=False,
            start_nonce=12,
        )
        rearmed = step_juggler(
            juggler,
            launch_time + 0.04,
            physics=physics,
            tuning=tuning,
            run_requested=True,
            start_nonce=13,
        )

        self.assertEqual(disarmed.diagnostics["state"], "stopped_hold")
        self.assertFalse(disarmed.diagnostics["requiresRearm"])
        self.assertNotEqual(rearmed.diagnostics["state"], "fault_hold")
        self.assertTrue(rearmed.diagnostics["runRequested"])


if __name__ == "__main__":
    unittest.main()
