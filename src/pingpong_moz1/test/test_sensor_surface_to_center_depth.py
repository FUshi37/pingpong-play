from collections import deque
import sys
import types

import pytest

# This unit only exercises scalar depth reconstruction.  Avoid importing the
# locally CUDA-linked YOLO/torch stack during test collection.
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

from pingpong_controller.hybrid_ball_detector import HybridBallDetector


class _ResetCounter:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def _correction_only_detector(enabled=True):
    detector = HybridBallDetector.__new__(HybridBallDetector)
    detector.sensor_surface_to_center_depth_enabled = bool(enabled)
    return detector


def test_surface_to_center_correction_is_sensor_only():
    detector = _correction_only_detector(enabled=True)

    center, diagnostics = detector._sensor_surface_to_center_depth(
        500.0, 'sensor')
    assert center == pytest.approx(514.96392834)
    assert diagnostics['sensor_surface_depth_mm'] == pytest.approx(500.0)
    assert diagnostics['sensor_center_depth_mm'] == pytest.approx(center)
    assert diagnostics['sensor_center_depth_correction_mm'] == pytest.approx(
        14.96392834)
    assert diagnostics['sensor_surface_to_center_depth_applied'] is True

    geometry, diagnostics = detector._sensor_surface_to_center_depth(
        500.0, 'geometry')
    assert geometry == pytest.approx(500.0)
    assert diagnostics['sensor_surface_to_center_depth_applied'] is False
    assert diagnostics['sensor_surface_depth_mm'] is None


def test_surface_to_center_correction_clamps_only_model_input():
    detector = _correction_only_detector(enabled=True)

    near_center, near = detector._sensor_surface_to_center_depth(
        250.0, 'adaptive_sensor')
    near_correction = (
        HybridBallDetector.SENSOR_CENTER_DEPTH_INTERCEPT_MM
        + HybridBallDetector.SENSOR_CENTER_DEPTH_SLOPE_PER_100_MM
        * (HybridBallDetector.SENSOR_CENTER_DEPTH_VALID_MIN_MM - 500.0)
        / 100.0)
    assert near_center == pytest.approx(250.0 + near_correction)
    assert near['sensor_center_depth_calibration_clamped'] is True

    far_center, far = detector._sensor_surface_to_center_depth(
        750.0, 'sensor')
    far_correction = (
        HybridBallDetector.SENSOR_CENTER_DEPTH_INTERCEPT_MM
        + HybridBallDetector.SENSOR_CENTER_DEPTH_SLOPE_PER_100_MM
        * (HybridBallDetector.SENSOR_CENTER_DEPTH_VALID_MAX_MM - 500.0)
        / 100.0)
    assert far_center == pytest.approx(750.0 + far_correction)
    assert far['sensor_center_depth_calibration_clamped'] is True


def test_disabled_correction_is_identity():
    detector = _correction_only_detector(enabled=False)
    depth, diagnostics = detector._sensor_surface_to_center_depth(
        500.0, 'sensor')
    assert depth == pytest.approx(500.0)
    assert diagnostics['sensor_surface_to_center_depth_applied'] is False
    assert diagnostics['sensor_center_depth_mm'] is None


def test_live_toggle_clears_3d_and_velocity_state_but_not_2d_state():
    detector = _correction_only_detector(enabled=False)
    detector.depth_history = deque([490.0, 500.0], maxlen=3)
    detector.ir_depth_kf = _ResetCounter()
    detector.position_history = deque([(1.0, [1.0, 2.0, 3.0])], maxlen=3)
    detector.velocity_window_frames = 3
    detector.physics_constrained_velocity_enabled = False
    detector.pixel_kf = object()
    original_pixel_kf = detector.pixel_kf
    detector.motion_last_trusted_time = 1.0
    detector.motion_last_trusted_position = [1.0, 2.0, 3.0]
    detector.motion_last_trusted_velocity = [4.0, 5.0, 6.0]
    detector.motion_last_innovation_mm = [0.1, 0.2, 0.3]
    detector.motion_last_gate_reason = 'tracking'
    detector.observation_gap_active = False
    detector.observation_buffers_cleared_for_gap = False
    detector.motion_reacquisition_candidates = deque([(1.0, [1, 2, 3])])
    detector.motion_reacquisition_seed = (1.0, [1, 2, 3])

    detector.configure_sensor_surface_to_center_depth(enabled=True)

    assert detector.sensor_surface_to_center_depth_enabled is True
    assert list(detector.depth_history) == []
    assert list(detector.position_history) == []
    assert detector.ir_depth_kf.reset_count == 1
    assert detector.motion_last_trusted_time is None
    assert detector.motion_reacquisition_seed is None
    assert detector.observation_gap_active is True
    assert detector.observation_buffers_cleared_for_gap is False
    assert detector.pixel_kf is original_pixel_kf
