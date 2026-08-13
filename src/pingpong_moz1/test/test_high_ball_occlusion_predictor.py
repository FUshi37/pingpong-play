#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.high_ball_occlusion_predictor import (  # noqa: E402
    HighBallOcclusionConfig,
    HighBallOcclusionPredictor,
    propagate_paper_model,
)


class HighBallOcclusionPredictorTest(unittest.TestCase):

    @staticmethod
    def _config(**overrides):
        values = {
            "history_frames": 10,
            "activation_z_m": 1.35,
            "min_upward_velocity_m_s": 0.2,
            "max_prediction_s": 0.45,
            "stop_prediction_z_m": 1.20,
            "drag_coefficient_per_m": 0.0,
            "integration_step_s": 0.001,
            "max_history_gap_s": 0.03,
            "min_history_span_s": 0.08,
            "activation_max_gap_s": 0.04,
            "fit_max_rmse_m": 0.005,
        }
        values.update(overrides)
        return HighBallOcclusionConfig(**values)

    @staticmethod
    def _ballistic_position(t):
        return np.array([
            -0.1 + 0.30 * t,
            -0.45 - 0.20 * t,
            1.20 + 3.0 * t - 0.5 * 9.8 * t * t,
        ])

    @staticmethod
    def _ballistic_velocity(t):
        return np.array([0.30, -0.20, 3.0 - 9.8 * t])

    def _seed_rising_track(self, predictor):
        result = None
        for index in range(10):
            timestamp = 0.01 * index
            result = predictor.step(
                self._ballistic_position(timestamp),
                self._ballistic_velocity(timestamp),
                True,
                timestamp,
                timestamp,
            )
        return result

    def test_visible_measurements_are_passed_through_unchanged(self):
        predictor = HighBallOcclusionPredictor(self._config())
        position = np.array([0.1, -0.2, 1.1])
        velocity = np.array([0.2, 0.3, 0.4])

        result = predictor.step(position, velocity, True, 1.0, 1.01)

        np.testing.assert_allclose(result.position_m, position)
        np.testing.assert_allclose(result.velocity_m_s, velocity)
        self.assertTrue(result.valid)
        self.assertFalse(result.predicted)

    def test_sub_microsecond_timestamp_variants_do_not_duplicate_history(self):
        predictor = HighBallOcclusionPredictor(self._config())

        result = None
        for index in range(10):
            timestamp = 0.01 * index
            position = self._ballistic_position(timestamp)
            velocity = self._ballistic_velocity(timestamp)
            result = predictor.step(
                position, velocity, True, timestamp, timestamp)
            # Reproduce one physical frame arriving through raw and formerly
            # rounded telemetry timestamp paths.
            result = predictor.step(
                position, velocity, True,
                timestamp + 2.0e-7,
                timestamp + 2.0e-7,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.diagnostics["historySamples"], 10)
        self.assertAlmostEqual(
            float(result.diagnostics["historySpanS"]), 0.09, places=6)

        predicted = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        self.assertTrue(predicted.predicted)
        np.testing.assert_allclose(
            predicted.diagnostics["initialVelocityMps"],
            self._ballistic_velocity(0.09),
            atol=2.0e-3,
        )

    def test_high_rising_loss_uses_paper_fit_and_ballistic_prediction(self):
        predictor = HighBallOcclusionPredictor(self._config())
        self._seed_rising_track(predictor)

        result = predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)

        self.assertTrue(result.valid)
        self.assertTrue(result.predicted)
        self.assertTrue(predictor.active)
        np.testing.assert_allclose(
            result.position_m, self._ballistic_position(0.10), atol=6.0e-4)
        np.testing.assert_allclose(
            result.velocity_m_s, self._ballistic_velocity(0.10), atol=2.0e-3)
        self.assertEqual(
            result.diagnostics["model"],
            "iros2249_ls_gravity_quadratic_drag",
        )

    def test_low_or_falling_loss_is_not_predicted(self):
        low_predictor = HighBallOcclusionPredictor(
            self._config(activation_z_m=1.60))
        self._seed_rising_track(low_predictor)
        low = low_predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        self.assertFalse(low.valid)
        self.assertEqual(
            low.diagnostics["reason"], "last_measurement_below_activation_z")

        falling_predictor = HighBallOcclusionPredictor(
            self._config(min_upward_velocity_m_s=3.0))
        self._seed_rising_track(falling_predictor)
        falling = falling_predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        self.assertFalse(falling.valid)
        self.assertEqual(
            falling.diagnostics["reason"], "ball_not_rising_at_loss")

    def test_sensor_depth_anomaly_forces_prediction_below_high_fov_gates(self):
        predictor = HighBallOcclusionPredictor(self._config(
            activation_z_m=1.60,
            min_upward_velocity_m_s=3.0,
            stop_prediction_z_m=1.20,
        ))
        for index in range(10):
            timestamp = 0.01 * index
            predictor.step(
                [0.1 + 0.2 * timestamp, -0.4, 1.0 - 0.1 * timestamp],
                [0.2, 0.0, -0.1],
                True,
                timestamp,
                timestamp,
            )

        predicted = predictor.step(
            np.zeros(3),
            np.zeros(3),
            False,
            0.10,
            0.10,
            force_prediction_reason="sensor_depth_anomaly",
            allow_high_fov_prediction=False,
        )

        self.assertTrue(predicted.valid)
        self.assertTrue(predicted.predicted)
        self.assertEqual(
            predicted.diagnostics["reason"],
            "predicting_sensor_depth_anomaly",
        )
        self.assertEqual(
            predicted.diagnostics["activeTrigger"],
            "sensor_depth_anomaly",
        )
        self.assertLess(float(predicted.position_m[2]), 1.20)

    def test_ir_top_edge_exit_bypasses_world_height_but_still_requires_rising(self):
        predictor = HighBallOcclusionPredictor(self._config(
            activation_z_m=1.60,
            stop_prediction_z_m=1.24,
        ))
        for index in range(10):
            timestamp = 0.01 * index
            predictor.step(
                [0.1, -0.4, 1.0 + 1.5 * timestamp
                 - 4.9 * timestamp * timestamp],
                [0.0, 0.0, 1.5 - 9.8 * timestamp],
                True,
                timestamp,
                timestamp,
            )

        predicted = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10,
            force_prediction_reason="high_fov_edge_exit")

        self.assertTrue(predicted.valid)
        self.assertTrue(predicted.predicted)
        self.assertEqual(
            predicted.diagnostics["activeTrigger"], "high_fov_edge_exit")
        self.assertEqual(
            predicted.diagnostics["reason"], "predicting_high_fov_edge_exit")

        # An explicit IR edge exit remains bounded by time, but must not be
        # terminated early by a world-Z threshold that depends on camera tilt.
        later = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.40, 0.40)
        self.assertTrue(later.valid)
        self.assertTrue(later.predicted)
        self.assertLess(float(later.position_m[2]), 1.24)

        falling = HighBallOcclusionPredictor(self._config(
            activation_z_m=1.60,
            stop_prediction_z_m=0.1,
        ))
        for index in range(10):
            timestamp = 0.01 * index
            falling.step(
                [0.1, -0.4, 1.2 - 0.5 * timestamp],
                [0.0, 0.0, -0.5],
                True,
                timestamp,
                timestamp,
            )
        rejected = falling.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10,
            force_prediction_reason="high_fov_edge_exit")
        self.assertFalse(rejected.valid)
        self.assertEqual(
            rejected.diagnostics["reason"], "ball_not_rising_at_loss")

    def test_sensor_depth_recovery_rejects_unconfirmed_position_jump(self):
        predictor = HighBallOcclusionPredictor(self._config(
            activation_z_m=1.60,
            min_upward_velocity_m_s=3.0,
            stop_prediction_z_m=0.1,
            reacquisition_gate_m=0.01,
        ))
        for index in range(10):
            timestamp = 0.01 * index
            predictor.step(
                [0.1 + 0.2 * timestamp, -0.4, 1.0],
                [0.2, 0.0, 0.0],
                True,
                timestamp,
                timestamp,
            )
        predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10,
            force_prediction_reason="sensor_depth_anomaly",
        )

        recovered_position = np.array([0.8, -0.2, 0.9])
        recovered_velocity = np.array([0.1, 0.2, -0.3])
        recovered = predictor.step(
            recovered_position, recovered_velocity, True, 0.11, 0.11)

        self.assertTrue(recovered.valid)
        self.assertTrue(recovered.predicted)
        self.assertTrue(predictor.active)
        self.assertEqual(
            recovered.diagnostics["reason"], "reacquisition_outside_gate")
        self.assertFalse(np.allclose(
            recovered.position_m, recovered_position))

    def test_detector_confirmed_recovery_can_override_inaccurate_model(self):
        predictor = HighBallOcclusionPredictor(self._config(
            activation_z_m=1.60,
            min_upward_velocity_m_s=3.0,
            stop_prediction_z_m=0.1,
            reacquisition_gate_m=0.01,
            enable_reacquisition_velocity_handover=True,
        ))
        for index in range(10):
            timestamp = 0.01 * index
            predictor.step(
                [0.1 + 0.2 * timestamp, -0.4, 1.0],
                [0.2, 0.0, 0.0], True, timestamp, timestamp)
        predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10,
            force_prediction_reason="sensor_depth_anomaly")

        recovered = predictor.step(
            [0.8, -0.2, 0.9], np.zeros(3), True, 0.11, 0.11,
            measurement_velocity_valid=False,
            measurement_reacquisition_confirmed=True)

        self.assertTrue(recovered.valid)
        self.assertTrue(recovered.predicted)
        self.assertEqual(
            recovered.diagnostics["reason"],
            "reacquisition_position_only")
        self.assertTrue(
            recovered.diagnostics["reacquisitionMeasurementConfirmed"])
        np.testing.assert_allclose(
            recovered.position_m, [0.8, -0.2, 0.9])

    def test_position_only_recovery_uses_model_velocity_until_fresh_fit(self):
        predictor = HighBallOcclusionPredictor(self._config(
            enable_reacquisition_velocity_handover=False,
            reacquisition_velocity_frames=3,
            reacquisition_velocity_blend_s=0.04,
        ))
        self._seed_rising_track(predictor)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)

        recovered = predictor.step(
            self._ballistic_position(0.11), np.zeros(3), True, 0.11, 0.11,
            measurement_velocity_valid=False)

        self.assertTrue(recovered.predicted)
        self.assertEqual(
            recovered.diagnostics["state"], "reacquisition_handover")
        self.assertTrue(
            recovered.diagnostics["reacquisitionPositionOnlySafetyHandover"])
        np.testing.assert_allclose(
            recovered.velocity_m_s,
            self._ballistic_velocity(0.11), atol=3e-3)

    def test_position_only_recovery_uses_enabled_full_state_smoothing(self):
        predictor = HighBallOcclusionPredictor(self._config(
            enable_reacquisition_smoothing=True,
            reacquisition_smoothing_frames=2,
            reacquisition_smoothing_blend_s=0.025,
            reacquisition_gate_m=0.01,
        ))
        self._seed_rising_track(predictor)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        reacquired = np.array([0.7, -0.2, 1.4])

        first = predictor.step(
            reacquired, np.zeros(3), True, 0.11, 0.11,
            measurement_velocity_valid=False,
            measurement_reacquisition_confirmed=True)

        self.assertTrue(first.valid)
        self.assertTrue(first.predicted)
        self.assertEqual(
            first.diagnostics["state"], "reacquisition_smoothing")
        self.assertEqual(
            first.diagnostics["reason"],
            "reacquisition_position_only_state_smoothing")
        self.assertFalse(np.allclose(first.position_m, reacquired))

    def test_sensor_depth_prediction_keeps_the_shared_safety_timeout(self):
        predictor = HighBallOcclusionPredictor(self._config(
            max_prediction_s=0.12,
            stop_prediction_z_m=1.20,
        ))
        position = np.array([0.1, -0.4, 1.0])
        velocity = np.array([0.2, 0.0, -0.1])
        predictor.step(position, velocity, True, 0.0, 0.0)

        first = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.01, 0.01,
            force_prediction_reason="sensor_depth_anomaly",
            allow_high_fov_prediction=False,
        )
        beyond_high_fov_timeout = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.50, 0.50,
            force_prediction_reason="sensor_depth_anomaly",
            allow_high_fov_prediction=False,
        )

        self.assertTrue(first.predicted)
        self.assertFalse(beyond_high_fov_timeout.valid)
        self.assertFalse(beyond_high_fov_timeout.predicted)
        self.assertEqual(
            beyond_high_fov_timeout.diagnostics["reason"],
            "prediction_timeout",
        )

    def test_reacquisition_immediately_returns_measurement_and_records_error(self):
        predictor = HighBallOcclusionPredictor(self._config())
        self._seed_rising_track(predictor)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.20, 0.20)
        measured_position = self._ballistic_position(0.21)
        measured_velocity = self._ballistic_velocity(0.21)

        result = predictor.step(
            measured_position, measured_velocity, True, 0.21, 0.215)

        self.assertTrue(result.valid)
        self.assertFalse(result.predicted)
        self.assertFalse(predictor.active)
        self.assertEqual(result.diagnostics["reason"], "reacquired")
        self.assertEqual(result.diagnostics["reacquisitionCount"], 1)
        np.testing.assert_allclose(result.position_m, measured_position)
        self.assertLess(
            float(result.diagnostics["lastReacquisitionErrorNormMm"]), 2.0)

    def test_prediction_is_bounded_by_timeout(self):
        predictor = HighBallOcclusionPredictor(
            self._config(max_prediction_s=0.12, stop_prediction_z_m=0.1))
        self._seed_rising_track(predictor)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)

        expired = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.25, 0.25)

        self.assertFalse(expired.valid)
        self.assertFalse(expired.predicted)
        self.assertFalse(predictor.active)
        self.assertEqual(expired.diagnostics["reason"], "prediction_timeout")

    def test_implausible_single_frame_reacquisition_is_rejected(self):
        predictor = HighBallOcclusionPredictor(
            self._config(reacquisition_gate_m=0.10))
        self._seed_rising_track(predictor)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)

        outlier = predictor.step(
            [0.8, 0.8, 0.8], [0.0, 0.0, 0.0], True, 0.11, 0.11)

        self.assertTrue(outlier.valid)
        self.assertTrue(outlier.predicted)
        self.assertTrue(predictor.active)
        self.assertEqual(
            outlier.diagnostics["reason"], "reacquisition_outside_gate")
        self.assertEqual(outlier.diagnostics["reacquisitionRejectedCount"], 1)

    def test_reacquisition_velocity_handover_rejects_endpoint_zero_velocity(self):
        predictor = HighBallOcclusionPredictor(self._config(
            enable_reacquisition_velocity_handover=True,
            reacquisition_velocity_frames=3,
            reacquisition_velocity_blend_s=0.04,
        ))
        self._seed_rising_track(predictor)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.20, 0.20)

        first = predictor.step(
            self._ballistic_position(0.21),
            np.zeros(3),
            True,
            0.21,
            0.21,
        )

        np.testing.assert_allclose(
            first.position_m, self._ballistic_position(0.21), atol=1e-7)
        np.testing.assert_allclose(
            first.velocity_m_s, self._ballistic_velocity(0.21), atol=3e-3)
        self.assertTrue(first.predicted)
        self.assertEqual(first.diagnostics["state"], "reacquisition_handover")
        self.assertFalse(first.diagnostics["positionPredicted"])
        self.assertTrue(first.diagnostics["velocityPredicted"])
        self.assertTrue(
            first.diagnostics["reacquisitionVelocityHandoverActive"])

    def test_reacquisition_velocity_handover_blends_to_ready_detector_velocity(self):
        predictor = HighBallOcclusionPredictor(self._config(
            enable_reacquisition_velocity_handover=True,
            reacquisition_velocity_frames=3,
            reacquisition_velocity_blend_s=0.04,
        ))
        self._seed_rising_track(predictor)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        predictor.step(np.zeros(3), np.zeros(3), False, 0.20, 0.20)
        for timestamp in (0.21, 0.22, 0.23):
            collecting = predictor.step(
                self._ballistic_position(timestamp),
                self._ballistic_velocity(timestamp),
                True,
                timestamp,
                timestamp,
            )
        self.assertEqual(
            collecting.diagnostics["reason"],
            "reacquisition_velocity_blending",
        )
        self.assertEqual(
            collecting.diagnostics["reacquisitionVelocitySamples"], 3)

        repeated_control_tick = predictor.step(
            self._ballistic_position(0.23),
            self._ballistic_velocity(0.23),
            True,
            0.23,
            0.25,
        )
        self.assertEqual(
            repeated_control_tick.diagnostics[
                "reacquisitionVelocityHandoverAlpha"],
            0.0,
        )

        halfway = predictor.step(
            self._ballistic_position(0.25),
            self._ballistic_velocity(0.25),
            True,
            0.25,
            0.25,
        )
        self.assertAlmostEqual(
            halfway.diagnostics["reacquisitionVelocityHandoverAlpha"],
            0.5,
            places=5,
        )

        completed = predictor.step(
            self._ballistic_position(0.27),
            self._ballistic_velocity(0.27),
            True,
            0.27,
            0.27,
        )
        self.assertFalse(completed.predicted)
        self.assertFalse(predictor.active)
        self.assertEqual(
            completed.diagnostics["reason"],
            "reacquisition_velocity_handover_complete",
        )
        np.testing.assert_allclose(
            completed.velocity_m_s,
            self._ballistic_velocity(0.27),
            atol=2e-3,
        )

    def test_quadratic_drag_reduces_speed(self):
        no_drag_position, no_drag_velocity = propagate_paper_model(
            [0.0, 0.0, 1.4], [2.0, 0.0, 2.0], 0.20,
            drag_coefficient_per_m=0.0,
        )
        drag_position, drag_velocity = propagate_paper_model(
            [0.0, 0.0, 1.4], [2.0, 0.0, 2.0], 0.20,
            drag_coefficient_per_m=0.13545,
        )

        self.assertLess(drag_position[0], no_drag_position[0])
        self.assertLess(drag_velocity[0], no_drag_velocity[0])
        self.assertTrue(np.all(np.isfinite(drag_position)))
        self.assertTrue(np.all(np.isfinite(drag_velocity)))

    def test_opt_in_control_time_compensation_bridges_visible_frame_hold(self):
        predictor = HighBallOcclusionPredictor(self._config(
            enable_control_time_compensation=True,
            control_time_compensation_max_s=0.025,
        ))
        position = np.array([0.0, 0.0, 1.5])
        velocity = np.array([1.0, -2.0, 3.0])

        result = predictor.step(position, velocity, True, 1.0, 1.01)

        expected_position, expected_velocity = propagate_paper_model(
            position, velocity, 0.01,
            drag_coefficient_per_m=0.0,
            gravity_m_s2=9.8,
            integration_step_s=0.001,
        )
        np.testing.assert_allclose(result.position_m, expected_position, atol=1e-5)
        np.testing.assert_allclose(result.velocity_m_s, expected_velocity, atol=1e-5)
        self.assertFalse(result.predicted)
        self.assertEqual(
            result.diagnostics["reason"],
            "measured_control_time_compensated")
        self.assertAlmostEqual(
            result.diagnostics["controlTimeCompensationAppliedS"],
            0.01, places=6)

    def test_opt_in_prediction_initialization_keeps_paper_velocity_neutral(self):
        predictor = HighBallOcclusionPredictor(self._config(
            history_frames=10,
            enable_optimized_initial_state=True,
        ))
        self._seed_rising_track(predictor)

        result = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10)

        expected_raw = self._ballistic_velocity(0.09)
        self.assertTrue(result.predicted)
        self.assertTrue(result.diagnostics["optimizedInitialStateEnabled"])
        self.assertEqual(result.diagnostics["historyFrames"], 9)
        self.assertAlmostEqual(result.diagnostics["activationZMinM"], 1.40)
        self.assertAlmostEqual(result.diagnostics["maxPredictionS"], 0.50)
        self.assertAlmostEqual(result.diagnostics["reacquisitionGateM"], 0.12)
        np.testing.assert_allclose(
            result.diagnostics["initialVelocityMps"], expected_raw, atol=2e-3)

    def test_opt_in_depth_fault_initialization_uses_neutral_calibration(self):
        predictor = HighBallOcclusionPredictor(self._config(
            history_frames=10,
            enable_optimized_initial_state=True,
        ))
        self._seed_rising_track(predictor)

        result = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10,
            allow_high_fov_prediction=False,
            force_prediction_reason="sensor_depth_anomaly",
        )

        self.assertTrue(result.predicted)
        self.assertEqual(
            result.diagnostics["activeTrigger"], "sensor_depth_anomaly")
        np.testing.assert_allclose(
            result.diagnostics["initialVelocityMps"],
            self._ballistic_velocity(0.09),
            atol=2e-3,
        )

    def test_depth_fault_fit_discards_recorded_direction_reversing_tail(self):
        predictor = HighBallOcclusionPredictor(self._config(
            history_frames=10,
            fit_max_rmse_m=0.030,
            enable_optimized_initial_state=True,
        ))
        recorded = (
            (0.000000, [0.044312, -0.395288, 1.226254]),
            (0.011185, [0.039911, -0.393551, 1.245965]),
            (0.022532, [0.035599, -0.391313, 1.264482]),
            (0.033538, [0.031889, -0.389141, 1.281533]),
            (0.044531, [0.028310, -0.387192, 1.297365]),
            (0.056200, [0.024954, -0.384573, 1.312752]),
            (0.066779, [0.021166, -0.382564, 1.325698]),
            (0.078229, [0.017324, -0.379331, 1.337997]),
            (0.089075, [0.014121, -0.379143, 1.347370]),
            (0.100332, [0.010372, -0.382863, 1.354144]),
            (0.111825, [0.005702, -0.388174, 1.358575]),
            (0.123320, [0.000947, -0.393598, 1.361619]),
            (0.134557, [-0.003705, -0.396974, 1.364388]),
        )
        for timestamp, position in recorded:
            predictor.step(
                position, [0.0, 0.0, 0.0], True,
                timestamp, timestamp)

        result = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.145, 0.145,
            allow_high_fov_prediction=False,
            force_prediction_reason="sensor_depth_anomaly",
        )

        self.assertTrue(result.predicted)
        self.assertEqual(
            result.diagnostics["depthFaultTailTrimmedSamples"], 3)
        self.assertGreater(
            float(result.diagnostics["initialVelocityMps"][1]), 0.10)

    def test_opt_in_reacquisition_smoothing_confirms_and_blends_full_state(self):
        predictor = HighBallOcclusionPredictor(self._config(
            enable_reacquisition_smoothing=True,
            reacquisition_smoothing_frames=2,
            reacquisition_smoothing_blend_s=0.025,
        ))
        self._seed_rising_track(predictor)
        predicted = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10)
        self.assertTrue(predicted.predicted)

        first = predictor.step(
            self._ballistic_position(0.11),
            self._ballistic_velocity(0.11),
            True, 0.11, 0.11)
        self.assertTrue(first.predicted)
        self.assertEqual(
            first.diagnostics["state"], "reacquisition_smoothing")

        completed = None
        for timestamp in (0.12, 0.13, 0.14, 0.15, 0.16):
            completed = predictor.step(
                self._ballistic_position(timestamp),
                self._ballistic_velocity(timestamp),
                True, timestamp, timestamp)
        self.assertIsNotNone(completed)
        self.assertFalse(completed.predicted)
        self.assertFalse(predictor.active)
        self.assertEqual(
            completed.diagnostics["reason"],
            "reacquisition_state_smoothing_complete")
        np.testing.assert_allclose(
            completed.position_m, self._ballistic_position(0.16), atol=1e-6)

    def test_recorded_000234_pattern_waits_for_ready_velocity_without_spike(self):
        predictor = HighBallOcclusionPredictor(self._config(
            enable_reacquisition_smoothing=True,
            reacquisition_smoothing_frames=2,
            reacquisition_smoothing_blend_s=0.025,
            reacquisition_velocity_frames=3,
        ))
        self._seed_rising_track(predictor)
        predictor.step(
            np.zeros(3), np.zeros(3), False, 0.10, 0.10,
            force_prediction_reason="high_fov_edge_exit")

        outputs = []
        # Reproduce the relevant contract from recording 20260812_160950:
        # four trusted positions arrive with an unready zero placeholder, one
        # of them has a 10 mm lateral perturbation, then the detector's robust
        # five-frame velocity becomes ready.
        for index, timestamp in enumerate((0.41, 0.42, 0.43, 0.44)):
            position = self._ballistic_position(timestamp)
            if index == 2:
                position = position + np.array([0.0, 0.010, 0.0])
            outputs.append(predictor.step(
                position, np.zeros(3), True, timestamp, timestamp,
                measurement_velocity_valid=False,
                measurement_reacquisition_confirmed=(index == 0)))

        detector_velocities = (
            np.array([0.10, 0.18, -1.54]),
            np.array([0.09, 0.12, -1.66]),
            np.array([0.055, -0.04, -1.80]),
            np.array([0.037, -0.11, -1.91]),
            np.array([0.025, -0.14, -2.02]),
            np.array([0.020, -0.15, -2.12]),
        )
        for timestamp, detector_velocity in zip(
                (0.45, 0.46, 0.47, 0.48, 0.49, 0.50),
                detector_velocities):
            outputs.append(predictor.step(
                self._ballistic_position(timestamp), detector_velocity,
                True, timestamp, timestamp,
                measurement_velocity_valid=True))

        velocities = np.asarray([result.velocity_m_s for result in outputs])
        self.assertTrue(all(result.valid for result in outputs))
        self.assertTrue(np.all(np.linalg.norm(velocities, axis=1) > 0.1))
        # The 10 mm position perturbation must not become a short-window vy
        # estimate such as the recorded +0.884 m/s value.
        self.assertLess(abs(float(velocities[2, 1])), 0.4)
        # The first ready detector frame starts, rather than completes, the
        # handover; it cannot create a zero -> detector impulse.
        np.testing.assert_allclose(
            velocities[4], self._ballistic_velocity(0.45), atol=3e-3)
        velocity_steps = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
        self.assertLess(float(np.max(velocity_steps)), 0.5)
        self.assertFalse(outputs[-1].predicted)

    def test_ordinary_reacquisition_bridges_velocity_from_last_trusted_state(self):
        predictor = HighBallOcclusionPredictor(self._config(
            activation_z_m=2.0,
            enable_reacquisition_smoothing=True,
        ))
        for timestamp in (0.00, 0.01, 0.02, 0.03, 0.04):
            predictor.step(
                [0.1 + 0.2 * timestamp, -0.4, 1.0 - 0.3 * timestamp],
                [0.2, 0.0, -0.3 - 9.8 * timestamp],
                True, timestamp, timestamp)
        lost = predictor.step(
            np.zeros(3), np.zeros(3), False, 0.05, 0.05)
        self.assertFalse(lost.valid)

        warming = predictor.step(
            [0.112, -0.4, 0.98], np.zeros(3), True, 0.06, 0.06,
            measurement_velocity_valid=False)

        self.assertTrue(warming.valid)
        self.assertTrue(warming.predicted)
        self.assertEqual(
            warming.diagnostics["reason"], "velocity_warmup_bridge_started")
        self.assertEqual(warming.diagnostics["velocityWarmupBridgeCount"], 1)
        self.assertGreater(float(np.linalg.norm(warming.velocity_m_s)), 0.1)

    def test_ordinary_warmup_does_not_bridge_stale_state_across_impact(self):
        predictor = HighBallOcclusionPredictor(self._config(
            activation_z_m=2.0,
            enable_reacquisition_smoothing=True,
        ))
        for timestamp in (0.00, 0.01, 0.02, 0.03, 0.04):
            predictor.step(
                [0.1 + 0.2 * timestamp, -0.4,
                 1.0 - 0.3 * timestamp],
                [0.2, 0.0, -0.3 - 9.8 * timestamp],
                True, timestamp, timestamp)
        predictor.step(
            np.zeros(3), np.zeros(3), False, 0.05, 0.05)

        # A post-impact point is incompatible with propagation of the last
        # pre-impact downward state.  Publishing that stale velocity is worse
        # than keeping the 6D observation invalid during the short detector
        # velocity warm-up.
        warming = predictor.step(
            [0.112, -0.4, 1.30], np.zeros(3), True, 0.06, 0.06,
            measurement_velocity_valid=False)

        self.assertFalse(warming.valid)
        self.assertFalse(warming.predicted)
        self.assertEqual(
            warming.diagnostics["reason"], "velocity_warmup_outside_gate")
        self.assertEqual(
            warming.diagnostics["velocityWarmupBridgeRejectedCount"], 1)
        self.assertGreater(
            warming.diagnostics["velocityWarmupRejectionErrorNormMm"],
            120.0)


if __name__ == "__main__":
    unittest.main()
