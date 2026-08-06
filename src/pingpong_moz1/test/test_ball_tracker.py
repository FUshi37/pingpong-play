#!/usr/bin/env python3

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.ball_tracker import (  # noqa: E402
    MODE_CONTACT_CANDIDATE,
    MODE_POST_IMPACT,
    BallEKFTracker,
    BallObservationConfig,
    BallStateEstimator,
    BallTrackerConfig,
    ball_dynamics,
    build_rl_observation,
    jacobian,
    predict_contact,
)
from pingpong_controller.ball_tracker.experiment_logger import (  # noqa: E402
    BallExperimentLogger,
    RLPerformanceLogger,
)


class BallModelTest(unittest.TestCase):

    def test_dynamics_and_jacobian(self):
        state = np.array([
            0.2, -0.1, 1.0,
            1.0, 2.0, 3.0,
            0.1, -0.2, -9.81,
        ])
        dt = 0.02
        expected_position = (
            state[:3] + state[3:6] * dt + 0.5 * state[6:9] * dt ** 2)
        expected_velocity = state[3:6] + state[6:9] * dt
        propagated = ball_dynamics(state, dt)
        np.testing.assert_allclose(propagated[:3], expected_position)
        np.testing.assert_allclose(propagated[3:6], expected_velocity)
        np.testing.assert_array_equal(propagated[6:9], state[6:9])

        analytic = jacobian(state, dt)
        numeric = np.zeros((9, 9))
        epsilon = 1e-6
        for column in range(9):
            perturbation = np.zeros(9)
            perturbation[column] = epsilon
            numeric[:, column] = (
                ball_dynamics(state + perturbation, dt)
                - ball_dynamics(state - perturbation, dt)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numeric, atol=1e-9)


class ImpactPredictorTest(unittest.TestCase):

    def test_predicts_position_velocity_and_uncertainty_on_plane(self):
        state = np.array([
            0.0, 0.0, 1.0,
            1.0, -0.5, -1.0,
            0.0, 0.0, -9.81,
        ])
        covariance = np.eye(9) * 1e-4
        result = predict_contact(
            state, {'z_racket': 0.5}, covariance=covariance)

        self.assertTrue(result['valid'])
        contact_time = result['time_to_contact']
        self.assertGreater(contact_time, 0.0)
        self.assertLess(contact_time, 0.6)
        self.assertAlmostEqual(result['impact_position'][2], 0.5, places=9)
        self.assertAlmostEqual(
            result['impact_velocity'][2],
            state[5] + state[8] * contact_time,
            places=9,
        )
        self.assertGreater(result['time_std_s'], 0.0)
        impact_covariance = np.asarray(
            result['impact_position_covariance'])
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(impact_covariance))), -1e-12)

    def test_rejects_ball_moving_away_from_playable_side(self):
        state = np.array([
            0.0, 0.0, 0.4,
            0.0, 0.0, -1.0,
            0.0, 0.0, -9.81,
        ])
        result = predict_contact(state, {'z': 0.5})
        self.assertFalse(result['valid'])


class BallEKFTrackerTest(unittest.TestCase):

    def test_release_gate_switches_between_held_and_ballistic_dynamics(self):
        tracker = BallEKFTracker(BallTrackerConfig(
            innovation_gate_chi2=0.0))
        tracker.initialize([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])

        tracker.set_ballistic_dynamics(False)
        held = tracker.project(0.1)
        self.assertTrue(held['valid'])
        self.assertAlmostEqual(held['state'][2], 1.0, places=9)
        self.assertAlmostEqual(held['state'][5], 0.0, places=9)
        np.testing.assert_allclose(held['state'][6:9], 0.0)

        tracker.set_ballistic_dynamics(True)
        flight = tracker.project(0.1)
        self.assertTrue(flight['valid'])
        self.assertLess(flight['state'][2], held['state'][2])
        self.assertLess(flight['state'][5], -0.9)
        self.assertTrue(
            tracker.diagnostics()['ballistic_dynamics_enabled'])

    def test_filter_estimates_velocity_acceleration_and_psd_covariance(self):
        rng = np.random.default_rng(2)
        config = BallTrackerConfig(innovation_gate_chi2=0.0)
        estimator = BallStateEstimator(config)
        truth = np.array([
            0.1, -0.2, 1.2,
            0.5, 0.1, 2.0,
            0.0, 0.0, -9.81,
        ])
        estimate = None
        for frame in range(61):
            timestamp = frame / 60.0
            if frame:
                truth = ball_dynamics(truth, 1.0 / 60.0)
            measurement = truth[:3] + rng.normal(
                0.0, config.measurement_std_m)
            estimate = estimator.update(measurement, timestamp)

        self.assertIsNotNone(estimate)
        self.assertTrue(estimate.valid)
        self.assertLess(np.linalg.norm(estimate.position - truth[:3]), 0.01)
        self.assertLess(np.linalg.norm(estimate.velocity - truth[3:6]), 0.10)
        self.assertLess(
            np.linalg.norm(estimate.acceleration - truth[6:9]), 0.70)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(estimate.state_covariance))),
            -1e-12,
        )

    def test_missing_observation_predicts_then_resets_after_200ms(self):
        estimator = BallStateEstimator(BallTrackerConfig(
            max_prediction_duration_s=0.2,
            innovation_gate_chi2=0.0,
        ))
        estimator.update([0.0, 0.0, 1.0], 1.0)
        estimator.update([0.0, 0.0, 1.01], 1.02)
        initial = estimator.predict_to(1.02)
        predicted = estimator.predict_to(1.219)
        timed_out = estimator.predict_to(1.221)

        self.assertTrue(predicted.valid)
        self.assertAlmostEqual(
            predicted.time_since_measurement_s, 0.199, places=9)
        self.assertGreater(
            np.trace(predicted.state_covariance),
            np.trace(initial.state_covariance),
        )
        self.assertFalse(timed_out.valid)
        self.assertEqual(
            timed_out.metadata['reject_reason'], 'prediction_timeout_reset')
        self.assertFalse(estimator.diagnostics()['initialized'])

    def test_control_prediction_does_not_block_delayed_image_update(self):
        estimator = BallStateEstimator(BallTrackerConfig(
            innovation_gate_chi2=0.0))
        estimator.update([0.0, 0.0, 1.0], 10.0)
        prediction = estimator.predict_to(10.030)
        delayed_frame = estimator.update([0.01, 0.0, 1.01], 10.016)

        self.assertTrue(prediction.valid)
        self.assertTrue(delayed_frame.measurement_accepted)
        self.assertEqual(
            estimator.diagnostics()['out_of_sequence_measurements'], 0)

    def test_contact_candidate_and_post_impact_velocity_reset(self):
        tracker = BallEKFTracker(BallTrackerConfig(
            innovation_gate_chi2=0.0,
            contact_distance_threshold_m=0.06,
        ))
        racket = {'z': 0.65}
        tracker.initialize([0.0, 0.0, 0.70], [0.0, 0.0, -3.0])
        tracker.predict(0.01, racket)
        self.assertEqual(tracker.mode, MODE_CONTACT_CANDIDATE)
        tracker.update([0.0, 0.0, 0.66], racket_pose=racket)
        tracker.predict(0.01, racket)
        tracker.update([0.0, 0.0, 0.67], racket_pose=racket)

        self.assertEqual(tracker.mode, MODE_POST_IMPACT)
        self.assertGreater(tracker.get_state()[5], 0.0)
        self.assertEqual(tracker.diagnostics()['impacts_detected'], 1)

        # The following purely post-impact interval refines the coarse
        # across-contact finite-difference velocity without counting twice.
        tracker.predict(0.01, racket)
        tracker.update([0.0, 0.0, 0.70], racket_pose=racket)
        self.assertGreater(tracker.get_state()[5], 2.5)
        self.assertEqual(tracker.diagnostics()['impacts_detected'], 1)


class EstimatorIntegrationTest(unittest.TestCase):

    def test_yaml_and_rl_observation_contract(self):
        config_path = (
            PACKAGE_ROOT / 'pingpong_controller' / 'ball_tracker' /
            'config.yaml')
        config = BallTrackerConfig.from_yaml(config_path)
        self.assertAlmostEqual(config.max_prediction_duration_s, 0.2)

        estimator = BallStateEstimator(config)
        estimate = estimator.update(
            [0.0, 0.0, 1.0], 0.0, racket_pose={'z': 0.5})
        q = np.array([0.1, 0.2])
        dq = np.array([0.3, 0.4])
        observation = build_rl_observation(q, dq, estimate)
        self.assertEqual(observation.shape, (14,))
        np.testing.assert_allclose(observation[:4], [0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(observation[4:7], estimate.position)
        np.testing.assert_allclose(observation[7:10], estimate.velocity)

        extended = build_rl_observation(
            q,
            dq,
            estimate,
            BallObservationConfig(
                include_acceleration=True,
                include_uncertainty=True,
                include_validity=True,
            ),
        )
        self.assertEqual(extended.shape, (25,))
        self.assertTrue(np.all(np.isfinite(extended)))

    def test_experiment_logger_writes_csv_and_summary(self):
        estimator = BallStateEstimator()
        estimate = estimator.update([0.0, 0.0, 1.0], 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.csv'
            logger = BallExperimentLogger(path, flush_every=1)
            logger.log(
                estimate,
                measurement=[0.0, 0.0, 1.0],
                ground_truth_state=[
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, -9.81,
                ],
            )
            summary_path = logger.write_summary({'position_rmse_m': 0.0})
            logger.close()

            with path.open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['estimate_valid'], '1')
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            self.assertEqual(summary['rows'], 1)

    def test_rl_performance_logger_aggregates_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy_comparison.csv'
            logger = RLPerformanceLogger(path)
            logger.log_episode(
                'raw', 0, hits=3, continuous_juggling_time_s=1.2,
                impact_position_error_m=0.04, failure_case='fov_loss')
            logger.log_episode(
                'raw', 1, hits=5, continuous_juggling_time_s=1.8,
                impact_position_error_m=0.02)
            logger.log_episode(
                'ekf_impact', 0, hits=8, continuous_juggling_time_s=3.0,
                impact_position_error_m=0.01)
            summary_path = logger.write_summary()
            logger.close()

            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            self.assertEqual(summary['raw']['average_hits'], 4.0)
            self.assertEqual(
                summary['raw']['maximum_continuous_juggling_time_s'], 1.8)
            self.assertEqual(summary['raw']['failure_cases']['fov_loss'], 1)


if __name__ == '__main__':
    unittest.main()
