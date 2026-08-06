import numpy as np

from pingpong_controller.ir_depth_kalman_filter import (
    IRDepthKalmanConfig,
    IRDepthKalmanFilter,
)


def test_tracks_linear_motion_with_variable_frame_intervals():
    estimator = IRDepthKalmanFilter(IRDepthKalmanConfig(
        innovation_gate_sigma=0.0,
    ))
    velocity_mm_s = np.array([250.0, -120.0, 900.0])
    times = np.array([0.0, 0.011, 0.023, 0.034, 0.046, 0.057, 0.069])

    for timestamp in times:
        position = np.array([100.0, -50.0, 600.0]) + velocity_mm_s * timestamp
        _, velocity = estimator.update(position, timestamp)

    np.testing.assert_allclose(velocity, velocity_mm_s, atol=80.0)


def test_rejects_single_large_depth_outlier_without_position_jump():
    estimator = IRDepthKalmanFilter(IRDepthKalmanConfig(
        innovation_gate_sigma=5.0,
        reset_after_outliers=3,
    ))
    estimator.update([0.0, 0.0, 500.0], 0.0)
    before, _ = estimator.update([1.0, 0.0, 500.0], 0.01)

    after, _ = estimator.update([2.0, 0.0, 900.0], 0.02)

    assert estimator.outliers_rejected == 1
    assert abs(after[2] - before[2]) < 20.0


def test_reinitializes_after_consistent_gated_measurements():
    estimator = IRDepthKalmanFilter(IRDepthKalmanConfig(
        innovation_gate_sigma=3.0,
        reset_after_outliers=2,
    ))
    estimator.update([0.0, 0.0, 500.0], 0.0)
    estimator.update([0.0, 0.0, 800.0], 0.01)
    position, velocity = estimator.update([0.0, 0.0, 805.0], 0.02)

    assert estimator.reinitializations == 1
    np.testing.assert_allclose(position, [0.0, 0.0, 805.0])
    np.testing.assert_allclose(velocity, [0.0, 0.0, 500.0])


def test_rotates_depth_measurement_uncertainty_into_output_frame():
    rotation = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    estimator = IRDepthKalmanFilter(IRDepthKalmanConfig(
        lateral_measurement_std_mm=2.0,
        depth_measurement_std_mm=10.0,
    ), frame_rotation=rotation)

    diagonal = np.diag(estimator._measurement_covariance)

    np.testing.assert_allclose(diagonal, [100.0, 4.0, 4.0])
