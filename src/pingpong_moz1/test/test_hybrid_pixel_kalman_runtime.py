import cv2
import numpy as np
import sys
import types

# The tests exercise the IR post-processing path without constructing YOLO.
# Avoid importing the local CUDA-linked torch build during test collection.
ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

from pingpong_controller.hybrid_ball_detector import HybridBallDetector
from pingpong_controller.pixel_kalman_filter import PixelKalmanFilter


def _detector_for_ir_unit_test(enabled: bool) -> HybridBallDetector:
    detector = HybridBallDetector.__new__(HybridBallDetector)
    detector.fps = 60
    detector.pixel_kalman_enabled = enabled
    detector.pixel_kalman_process_noise_pos = 0.3
    detector.pixel_kalman_process_noise_vel = 3.0
    detector.pixel_kalman_measurement_noise = 1.2
    detector.pixel_kalman_innovation_gate_px = 15.0
    detector.pixel_kalman_reset_after_outliers = 3
    detector.ir_depth_kalman_enabled = False
    detector.synchronized_depth_enabled = False
    detector.depth_temporal_window_frames = 5
    detector.velocity_least_squares_enabled = True
    detector.velocity_window_frames = 3
    detector.impact_velocity_reset_enabled = True
    detector.impact_min_downward_velocity_mm_s = 250.0
    detector.impact_min_upward_velocity_mm_s = 250.0
    detector.impact_min_velocity_jump_mm_s = 800.0
    detector.impact_refractory_s = 0.08
    detector.last_velocity_impact_time = None
    detector.last_velocity_impact_detected = False
    detector.candidate_filter_enabled = False
    detector.candidate_min_brightness = 23000.0
    detector.candidate_max_depth_disagreement_mm = 180.0
    detector.candidate_motion_scale_px = 130.0
    detector.depth_scale = 0.001
    detector.depth_neighborhood_px = 2
    detector.ball_diameter_mm = 40.0
    detector.ir_fx = 430.0
    detector.ir_fy = 430.0
    detector.image_width = 120
    detector.image_height = 100
    detector.roi_margin = 20
    detector.depth_correction_model_path = "/nonexistent/test-depth-model.pkl"
    detector.brightness_percentile = 25.0
    detector.background_intensity = None
    detector.ir_params = {
        "threshold": 100,
        "use_adaptive": False,
        "dilate_iterations": 0,
        "close_kernel_size": 0,
        "denoise_kernel_size": 0,
        "min_radius": 5,
        "max_radius": 30,
        "circularity_min": 0.5,
    }
    detector._init_runtime_state()
    return detector


def _ball_image(cx: int, cy: int = 50) -> np.ndarray:
    image = np.zeros((100, 120), dtype=np.uint8)
    cv2.circle(image, (cx, cy), 10, 255, -1)
    return image


def test_matching_rgb_resolution_is_not_resized_at_high_stereo_rate():
    detector = HybridBallDetector.__new__(HybridBallDetector)
    detector.image_width = 848
    detector.image_height = 480
    rgb = np.zeros((480, 848, 3), dtype=np.uint8)

    normalized = detector._rgb_on_ir_canvas(rgb)

    assert normalized is rgb


def test_invalid_stereo_color_frame_polls_independent_rgb_pipeline():
    class InvalidFrame:
        def __bool__(self):
            return False

    class ColorFrame:
        def __bool__(self):
            return True

        def get_data(self):
            return np.full((4, 6, 3), 73, dtype=np.uint8)

        def get_timestamp(self):
            return 1000.0

    class Frames:
        def __init__(self, frame):
            self.frame = frame

        def __bool__(self):
            return True

        def get_color_frame(self):
            return self.frame

    class ColorPipeline:
        def __init__(self):
            self.poll_count = 0

        def poll_for_frames(self):
            self.poll_count += 1
            return Frames(ColorFrame())

    detector = HybridBallDetector.__new__(HybridBallDetector)
    detector.image_width = 6
    detector.image_height = 4
    detector.latest_color_image = np.zeros((4, 6, 3), dtype=np.uint8)
    detector.color_pipeline = ColorPipeline()
    detector.color_frame_count = 0
    detector.color_last_timestamp_ms = None
    detector.color_fps_queue = []

    image = detector.color_image_for_frames(Frames(InvalidFrame()))

    assert detector.color_pipeline.poll_count == 1
    assert detector.color_frame_count == 1
    assert np.all(image == 73)


def test_disabled_hybrid_pixel_kalman_returns_raw_ir_center():
    detector = _detector_for_ir_unit_test(enabled=False)

    detection = detector._detect_in_ir(_ball_image(75))

    assert detection is not None
    assert detection["cx"] == detection["cx_raw"]
    assert detection["cy"] == detection["cy_raw"]


def test_ir_roi_is_cropped_and_coordinates_stay_full_frame():
    detector = _detector_for_ir_unit_test(enabled=False)

    detection = detector._detect_in_ir(
        _ball_image(75), roi=(55, 30, 95, 70))

    assert detection is not None
    assert abs(detection["cx"] - 75.0) < 1.0
    assert abs(detection["cy"] - 50.0) < 1.0
    assert detection["mask"].shape[0] < 40
    assert detection["mask"].shape[1] < 40
    assert detection["mask_origin"][0] > 55


def test_ir_candidate_clipped_by_tracking_roi_is_rejected_for_full_frame_retry():
    detector = _detector_for_ir_unit_test(enabled=False)

    detection = detector._detect_in_ir(
        _ball_image(60), roi=(60, 30, 95, 70))

    assert detection is None
    assert detector.ir_roi_edge_rejected_count == 1


def test_ir_contour_touching_top_edge_is_not_a_valid_ball_observation():
    detector = _detector_for_ir_unit_test(enabled=False)

    assert detector._ir_contour_is_edge_clipped(
        575.324087, 28.199232, 28.317993, (480, 848))
    assert not detector._ir_contour_is_edge_clipped(
        569.917506, 49.659020, 26.934210, (480, 848))

    detection = detector._detect_in_ir(_ball_image(60, cy=9))
    assert detection is None
    assert detector.ir_edge_rejected_count == 1


def test_local_candidate_mask_reads_matching_depth_crop():
    detector = _detector_for_ir_unit_test(enabled=False)
    depth = np.zeros((100, 120), dtype=np.uint16)
    depth[42:59, 67:84] = 620
    mask = np.full((17, 17), 255, dtype=np.uint8)

    measured = detector._get_depth_from_depthmap(
        75.0, 50.0, depth, mask, (67, 42))

    assert measured == 620.0


def test_adaptive_depth_sampling_keeps_sparse_foreground_depth():
    detector = _detector_for_ir_unit_test(enabled=False)
    depth = np.full((10, 10), 1200, dtype=np.uint16)
    depth[:3, :] = 320
    mask = np.full((10, 10), 255, dtype=np.uint8)

    legacy = detector._get_depth_from_depthmap(
        5.0, 5.0, depth, mask, (0, 0))
    adaptive = detector._get_depth_from_depthmap(
        5.0, 5.0, depth, mask, (0, 0), prefer_near_surface=True)

    assert legacy == 1200.0
    assert adaptive == 320.0


def test_existing_depth_modes_keep_their_legacy_source_priority():
    detector = _detector_for_ir_unit_test(enabled=False)

    detector.set_depth_mode("sensor")
    assert detector._select_depth(1550.0, 293.0) == (1550.0, "sensor")

    detector.set_depth_mode("geometry")
    assert detector._select_depth(1550.0, 293.0) == (293.0, "geometry")


def test_adaptive_depth_keeps_valid_near_sensor_measurement():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")

    depth, method = detector._select_depth(
        350.0, 310.0, 0.00, observed_radius_px=25.0)

    assert depth == 350.0
    assert method == "adaptive_sensor"
    assert detector.last_depth_selection["sensor_rejected"] is False
    assert detector.last_depth_selection["sensor_weight"] == 1.0


def test_deferred_adaptive_depth_commits_only_after_complete_3d_acceptance():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")

    detector._select_depth(
        350.0, 340.0, 0.00, observed_radius_px=25.0,
        defer_commit=True)

    assert len(detector.adaptive_sensor_history) == 0
    detector._finalize_adaptive_depth_selection(True)
    assert list(detector.adaptive_sensor_history) == [(0.0, 350.0)]


def test_rejected_3d_candidate_rolls_back_pending_adaptive_state():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")
    detector._select_depth(350.0, 340.0, 0.00)
    last_raw = detector.adaptive_last_raw_sensor_mm

    detector._select_depth(
        360.0, 350.0, 0.01, defer_commit=True)
    detector._finalize_adaptive_depth_selection(False)

    assert len(detector.adaptive_sensor_history) == 1
    assert detector.adaptive_last_raw_sensor_mm == last_raw


def test_piecewise_geometry_depth_correction_is_applied():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.depth_correction_model = {
        "type": "piecewise_linear",
        "params": {
            "breakpoint": 450.0,
            "near": {"scale": 0.75, "offset": 40.0},
            "far": {"scale": 0.70, "offset": 60.0},
        },
    }

    assert detector._apply_depth_correction(400.0) == 340.0
    assert detector._apply_depth_correction(500.0) == 410.0


def test_depth_preview_does_not_write_temporal_history():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.depth_history.extend([400.0, 410.0])

    preview = detector._depth_for_reconstruction(420.0, commit=False)

    assert preview == 410.0
    assert list(detector.depth_history) == [400.0, 410.0]


def test_rejected_3d_frame_rolls_back_pixel_and_candidate_history():
    detector = _detector_for_ir_unit_test(enabled=True)
    detector.candidate_position_history.append((0.0, 20.0, 30.0))
    detector.pixel_kf.update([20.0, 30.0])
    snapshot = detector._snapshot_ir_tracking_state()

    detector.candidate_position_history.append((0.01, 90.0, 80.0))
    detector.pixel_kf.update([90.0, 80.0])
    detector._restore_ir_tracking_state(snapshot)

    assert list(detector.candidate_position_history) == [
        (0.0, 20.0, 30.0)]
    np.testing.assert_allclose(detector.pixel_kf.state[:2], [20.0, 30.0])


def test_adaptive_depth_rejects_sensor_jump_by_itself():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")
    detector._select_depth(400.0, 400.0, 0.00)
    detector._select_depth(410.0, 410.0, 0.01)

    depth, method = detector._select_depth(650.0, 600.0, 0.02)

    assert method == "adaptive_prediction_geometry"
    assert depth == 450.0
    assert detector.last_depth_selection["sensor_rejected"] is True
    assert detector.last_depth_selection["rejection_reason"] == "sensor_jump"


def test_adaptive_depth_rejects_sensor_over_1000mm_by_itself():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")

    depth, method = detector._select_depth(1200.0, 950.0, 0.00)

    assert (depth, method) == (950.0, "adaptive_geometry_fallback")
    assert detector.last_depth_selection["rejection_reason"] == (
        "sensor_over_1000mm")


def test_adaptive_depth_rejects_geometry_gap_by_itself():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")

    depth, method = detector._select_depth(800.0, 450.0, 0.00)

    assert (depth, method) == (450.0, "adaptive_geometry_fallback")
    assert detector.last_depth_selection["rejection_reason"] == (
        "geometry_disagreement")


def test_adaptive_depth_rejects_ir_radius_disagreement_by_itself():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")

    depth, method = detector._select_depth(
        386.0, 296.293375, 0.00, observed_radius_px=29.076077)

    assert (depth, method) == (296.293375, "adaptive_geometry_fallback")
    assert detector.last_depth_selection["sensor_rejected"] is True
    assert detector.last_depth_selection["rejection_reason"] == (
        "radius_disagreement")
    assert detector.last_depth_selection["radius_mismatch_ratio"] > 0.23


def test_recorded_386mm_polluted_frame_triggers_radius_guard():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")
    detector._select_depth(
        330.0, 327.377043, 0.000000, observed_radius_px=26.315372)
    detector._select_depth(
        391.0, 330.067712, 0.011167, observed_radius_px=26.100853)

    _depth, method = detector._select_depth(
        386.0, 296.293375, 0.022456, observed_radius_px=29.076077)

    assert method == "adaptive_prediction_geometry"
    assert detector.last_depth_selection["sensor_rejected"] is True
    assert detector.last_depth_selection["rejection_reason"] == (
        "radius_disagreement")
    assert len(detector.adaptive_sensor_history) == 2


def test_recorded_near_field_plateau_is_rejected_before_large_depth_jump():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")
    recorded = (
        # sensor, geometry, radius, cy, timestamp
        (327.0, 307.083312, 28.054435, 166.114052, 0.000000),
        (337.0, 315.881136, 27.273072, 156.512405, 0.011257),
        (330.0, 290.712188, 29.634288, 148.434562, 0.022751),
    )

    outputs = [
        detector._select_depth(
            sensor, geometry, timestamp,
            observed_radius_px=radius,
            observed_cy_px=cy)
        for sensor, geometry, radius, cy, timestamp in recorded
    ]

    assert outputs[0][1] == "adaptive_sensor"
    assert outputs[1][1] == "adaptive_sensor"
    assert outputs[2][1] == "adaptive_prediction_geometry"
    assert detector.last_depth_selection["sensor_rejected"] is True
    assert detector.last_depth_selection["near_field_scale_stall"] is True
    assert detector.last_depth_selection["rejection_reason"] == (
        "near_field_scale_stall")
    assert detector.last_depth_selection[
        "near_field_window_samples"] == 3
    assert detector.last_depth_selection[
        "near_field_window_span_s"] < 0.025


def test_near_field_guard_does_not_reject_real_slow_image_apex():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")
    recorded_apex = (
        (399.0, 380.853230, 22.620391, 87.402548, 0.000),
        (400.0, 337.897293, 25.496058, 82.242578, 0.011),
        (399.0, 334.559658, 25.750411, 79.808084, 0.025),
    )

    outputs = [
        detector._select_depth(
            sensor, geometry, timestamp,
            observed_radius_px=radius,
            observed_cy_px=cy)
        for sensor, geometry, radius, cy, timestamp in recorded_apex
    ]

    assert all(method == "adaptive_sensor" for _depth, method in outputs)
    assert detector.last_depth_selection["near_field_scale_stall"] is False
    assert detector.last_depth_selection[
        "near_field_upward_pixel_speed_px_s"] < 500.0


def test_adaptive_depth_uses_sensor_when_all_four_checks_pass():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")

    depth, method = detector._select_depth(
        900.0, 700.0, 0.00, observed_radius_px=9.5)

    assert (depth, method) == (900.0, "adaptive_sensor")


def test_adaptive_depth_recovers_after_two_normal_sensor_frames():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")
    detector._select_depth(400.0, 400.0, 0.00)
    detector._select_depth(410.0, 410.0, 0.01)
    detector._select_depth(650.0, 600.0, 0.02)

    outputs = [
        detector._select_depth(sensor, sensor, timestamp)
        for sensor, timestamp in (
            (500.0, 0.03), (510.0, 0.04))
    ]

    assert outputs[0][1] == "adaptive_prediction_geometry"
    assert outputs[1] == (510.0, "adaptive_sensor")
    assert detector.last_depth_selection["sensor_rejected"] is False
    assert detector.last_depth_selection["sensor_recovery_count"] == 0


def test_disabling_yolo_clears_bbox_and_uses_ir_tracking_roi():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.last_yolo_bbox = np.array([1, 2, 3, 4])
    detector.yolo_miss_count = 2
    detector.candidate_position_history.extend([
        (0.98, 60.0, 50.0),
        (1.00, 65.0, 50.0),
    ])

    detector.configure_yolo(False)
    roi = detector._predicted_ir_roi(1.02)

    assert detector.last_yolo_bbox is None
    assert detector.yolo_miss_count == 0
    assert roi is not None
    assert roi[0] < 70 < roi[2]


def test_runtime_reconfigure_changes_gate_and_resets_filter_state():
    detector = _detector_for_ir_unit_test(enabled=True)
    first = detector._detect_in_ir(_ball_image(25))
    lagged = detector._detect_in_ir(_ball_image(75))
    assert first is not None and lagged is not None
    assert abs(lagged["cx"] - lagged["cx_raw"]) > 20.0

    detector.configure_pixel_kalman(
        enabled=True,
        process_noise_pos=0.4,
        process_noise_vel=4.0,
        measurement_noise=1.0,
        innovation_gate_px=100.0,
        reset_after_outliers=1,
    )
    reacquired = detector._detect_in_ir(_ball_image(75))

    assert reacquired is not None
    assert reacquired["cx"] == reacquired["cx_raw"]
    assert detector.pixel_kf.innovation_gate == 100.0
    assert detector.pixel_kf.reset_after_outliers == 1


def test_synchronized_depth_and_joint_kalman_are_independent_switches():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_ir_depth_processing(
        kalman_enabled=True,
        synchronized_depth_enabled=False,
        depth_temporal_window_frames=5,
    )
    assert detector.ir_depth_kalman_enabled is True
    assert detector.synchronized_depth_enabled is False
    assert detector._depth_for_reconstruction(500.0) == 500.0
    assert detector._depth_for_reconstruction(510.0) == 510.0
    assert detector._depth_for_reconstruction(700.0) == 510.0

    detector.configure_ir_depth_processing(
        kalman_enabled=False,
        synchronized_depth_enabled=True,
        depth_temporal_window_frames=7,
    )
    assert detector.ir_depth_kalman_enabled is False
    assert detector.synchronized_depth_enabled is True
    assert detector.depth_temporal_window_frames == 7
    assert detector.depth_history.maxlen == 7
    assert detector._depth_for_reconstruction(700.0) == 700.0


def test_ir_depth_reconfiguration_invalidates_velocity_readiness():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.last_velocity_ready = True

    detector.configure_ir_depth_processing(
        kalman_enabled=True,
        synchronized_depth_enabled=False,
        depth_temporal_window_frames=5,
    )

    assert detector.last_velocity_ready is False
    assert len(detector.depth_history) == 0

    detector.configure_ir_depth_processing(
        kalman_enabled=False,
        synchronized_depth_enabled=False,
        depth_temporal_window_frames=1,
    )
    assert detector._depth_for_reconstruction(500.0) == 500.0
    assert detector._depth_for_reconstruction(700.0) == 700.0


def test_velocity_least_squares_uses_configured_three_frame_window():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_velocity_estimation(
        least_squares_enabled=True,
        window_frames=3,
        impact_reset_enabled=False,
        impact_min_downward_velocity_m_s=0.25,
        impact_min_upward_velocity_m_s=0.25,
        impact_min_velocity_jump_m_s=0.80,
        impact_refractory_s=0.08,
    )

    detector._append_velocity_sample(0.00, [0.0, 0.0, 1000.0])
    detector._append_velocity_sample(0.01, [1.0, -2.0, 990.0])
    velocity = detector._append_velocity_sample(
        0.02, [2.0, -4.0, 980.0])

    np.testing.assert_allclose(velocity, [100.0, -200.0, -1000.0])
    assert detector.position_history.maxlen == 3


def test_impact_reset_confirms_reversal_and_does_not_mix_flight_segments():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_velocity_estimation(
        least_squares_enabled=True,
        window_frames=3,
        impact_reset_enabled=True,
        impact_min_downward_velocity_m_s=0.25,
        impact_min_upward_velocity_m_s=0.25,
        impact_min_velocity_jump_m_s=0.80,
        impact_refractory_s=0.08,
    )
    detector._append_velocity_sample(0.00, [0.0, 0.0, 1000.0])
    detector._append_velocity_sample(0.01, [0.0, 0.0, 990.0])

    candidate_velocity = detector._append_velocity_sample(
        0.02, [0.0, 0.0, 1000.0])

    assert detector.last_velocity_impact_detected is False
    assert detector.pending_impact is not None
    assert len(detector.position_history) == 2
    np.testing.assert_allclose(candidate_velocity, [0.0, 0.0, -1000.0])

    post_velocity = detector._append_velocity_sample(
        0.03, [1.0, -1.0, 1010.0])
    assert detector.last_velocity_impact_detected is True
    assert detector.pending_impact is None
    assert len(detector.position_history) == 2
    np.testing.assert_allclose(post_velocity, [100.0, -100.0, 1000.0])


def test_opt_in_loss_reset_clears_all_temporal_observation_buffers_once():
    detector = _detector_for_ir_unit_test(enabled=True)
    detector.configure_observation_quality(
        clear_buffers_on_detection_loss_enabled=True,
        motion_consistency_gate_enabled=False,
        physics_constrained_velocity_enabled=False,
    )
    detector.depth_history.extend([400.0, 410.0])
    detector.position_history.append((1.0, [1.0, 2.0, 3.0]))
    detector.pixel_history.append((10.0, 20.0, 5.0))
    detector.candidate_position_history.append((1.0, 10.0, 20.0))
    detector.pixel_kf.update([10.0, 20.0])
    detector.ir_depth_kf.update([1.0, 2.0, 3.0], 1.0)

    detector._handle_observation_loss("unit_test_loss")
    detector._handle_observation_loss("same_gap_second_frame")

    assert len(detector.depth_history) == 0
    assert len(detector.position_history) == 0
    assert len(detector.pixel_history) == 0
    assert len(detector.candidate_position_history) == 0
    assert detector.pixel_kf.initialized is False
    assert detector.ir_depth_kf.initialized is False
    assert detector.observation_buffer_clear_count == 1


def test_single_90hz_ir_flicker_preserves_continuous_velocity_history():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_observation_quality(
        clear_buffers_on_detection_loss_enabled=True,
        motion_consistency_gate_enabled=True,
        physics_constrained_velocity_enabled=True,
    )
    detector.position_history.append((1.0, [1.0, 2.0, 3.0]))
    detector.ir_tracking_miss_count = 1

    detector._handle_observation_loss("ir_detection_lost")

    assert len(detector.position_history) == 1
    assert detector.observation_buffer_clear_count == 0

    detector.ir_tracking_miss_count = 2
    detector._handle_observation_loss("ir_detection_lost")

    assert len(detector.position_history) == 0
    assert detector.observation_buffer_clear_count == 1


def test_opt_in_3d_motion_gate_rejects_non_ballistic_jump():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_observation_quality(
        clear_buffers_on_detection_loss_enabled=False,
        motion_consistency_gate_enabled=True,
        physics_constrained_velocity_enabled=False,
    )
    detector._remember_motion_state(
        1.0, [0.0, 0.0, 1000.0], [100.0, 200.0, 1000.0])

    expected = np.array([1.0, 2.0, 1009.51])
    assert detector._motion_measurement_consistent(1.01, expected)
    assert not detector._motion_measurement_consistent(
        1.01, expected + np.array([90.0, 0.0, 0.0]))
    assert detector.motion_last_gate_reason == "xy_innovation"
    assert detector.motion_gate_rejected_count == 1


def test_motion_gate_does_not_unconditionally_accept_after_80ms_gap():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_observation_quality(
        clear_buffers_on_detection_loss_enabled=False,
        motion_consistency_gate_enabled=True,
        physics_constrained_velocity_enabled=False,
    )
    detector._remember_motion_state(
        1.0, [0.0, 0.0, 1000.0], [0.0, 0.0, 0.0])
    detector.observation_gap_active = True

    assert not detector._motion_measurement_consistent(
        1.20, [500.0, 0.0, 800.0])
    assert detector.motion_last_gate_reason == "reacquisition_confirming"
    assert detector._motion_measurement_consistent(
        1.21, [510.0, 0.0, 790.0])
    assert detector.motion_last_gate_reason == "reacquisition_confirmed"


def test_adaptive_depth_fault_does_not_pollute_any_3d_history():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.set_depth_mode("adaptive")
    detector.yolo_enabled = False
    detector.frame_count = 1
    detector.output_frame = "camera"
    detector.ir_cx = 60.0
    detector.ir_cy = 50.0
    detector.T_ir_to_rgb = np.eye(4)
    detector.depth_history.append(330.0)
    detector.position_history.append((0.0, [0.0, 0.0, 330.0]))
    detector._detect_in_ir = lambda *_args, **_kwargs: {
        "cx": 60.0,
        "cy": 50.0,
        "cx_raw": 60.0,
        "cy_raw": 50.0,
        "radius": 29.076077,
        "geometry_radius_px": 29.076077,
        "geometry_quality_valid": True,
        "aspect_ratio": 1.0,
        "brightness": 50000.0,
        "mask": np.ones((5, 5), dtype=np.uint8) * 255,
        "mask_origin": (58, 48),
        "selection_sensor_depth_mm": 386.0,
        "selection_geom_depth_raw_mm": 296.293375,
        "selection_geom_depth_corrected_mm": 296.293375,
    }

    result = detector._detect_core(
        np.zeros((100, 120, 3), dtype=np.uint8),
        np.zeros((100, 120), dtype=np.uint8),
        np.full((100, 120), 386, dtype=np.uint16),
        0.01,
    )

    assert result["detected"] is False
    assert result["raw_measurement"]["observation_state"] == "depth_fault"
    assert result["raw_measurement"]["measurement_trusted_3d"] is False
    assert list(detector.depth_history) == [330.0]
    assert list(detector.position_history) == [
        (0.0, [0.0, 0.0, 330.0])]


def test_opt_in_physics_velocity_fits_current_ballistic_velocity():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_velocity_estimation(
        least_squares_enabled=False,
        window_frames=3,
        impact_reset_enabled=False,
        impact_min_downward_velocity_m_s=0.25,
        impact_min_upward_velocity_m_s=0.25,
        impact_min_velocity_jump_m_s=0.80,
        impact_refractory_s=0.08,
    )
    detector.configure_observation_quality(
        clear_buffers_on_detection_loss_enabled=False,
        motion_consistency_gate_enabled=False,
        physics_constrained_velocity_enabled=True,
    )
    velocities = []
    for timestamp in np.linspace(0.0, 0.04, 5):
        position = [
            100.0 * timestamp,
            -200.0 * timestamp,
            1000.0 + 2000.0 * timestamp - 4900.0 * timestamp ** 2,
        ]
        velocities.append(
            detector._append_velocity_sample(timestamp, position))

    for velocity in velocities[:-1]:
        np.testing.assert_array_equal(velocity, [0.0, 0.0, 0.0])
    assert detector.last_velocity_ready is True
    np.testing.assert_allclose(
        velocities[-1], [100.0, -200.0, 1608.0], atol=1e-6)
    assert detector.position_history.maxlen == 5


def test_physics_velocity_restart_bridges_first_ready_fit_after_loss():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    _seed_falling_flight(detector)
    reference_time = float(detector.last_velocity_output_time)
    reference_velocity = detector.last_velocity_output.copy()

    detector._clear_observation_buffers()
    assert detector.last_velocity_ready is False
    assert detector.velocity_restart_reference_time == reference_time

    samples = []
    for index, timestamp in enumerate(np.linspace(0.05, 0.09, 5)):
        lateral_noise = (-2.5, -2.5, 7.2, 7.2, 7.3)[index]
        position = [
            500.0 * timestamp,
            lateral_noise,
            1000.0 - 500.0 * timestamp - 4900.0 * timestamp ** 2,
        ]
        samples.append((timestamp, position))
        velocity = detector._append_velocity_sample(timestamp, position)

    _, raw_first_fit = detector._fit_trajectory_state(
        list(detector.position_history), gravity=True)
    predicted = reference_velocity.copy()
    predicted[2] -= 9800.0 * (samples[-1][0] - reference_time)

    assert detector.last_velocity_ready is True
    assert detector.last_velocity_restart_bridge_used is True
    assert detector.velocity_restart_bridge_count == 1
    assert detector.velocity_restart_reference_time is None
    assert abs(velocity[1]) < abs(raw_first_fit[1])
    assert np.linalg.norm(np.asarray(velocity) - predicted) <= (
        detector.FLIGHT_VELOCITY_MAX_CORRECTION_MM_S + 1e-9)


def _configure_stable_physics_velocity(detector, *, smoothing=True):
    detector.configure_velocity_estimation(
        least_squares_enabled=False,
        window_frames=3,
        impact_reset_enabled=True,
        impact_min_downward_velocity_m_s=0.25,
        impact_min_upward_velocity_m_s=0.25,
        impact_min_velocity_jump_m_s=0.80,
        impact_refractory_s=0.08,
    )
    detector.configure_observation_quality(
        clear_buffers_on_detection_loss_enabled=False,
        motion_consistency_gate_enabled=False,
        physics_constrained_velocity_enabled=True,
        trajectory_position_smoothing_enabled=smoothing,
    )


def _seed_falling_flight(detector):
    for timestamp in np.linspace(0.0, 0.04, 5):
        detector._append_velocity_sample(timestamp, [
            500.0 * timestamp,
            0.0,
            1000.0 - 500.0 * timestamp - 4900.0 * timestamp ** 2,
        ])
    assert detector.velocity_motion_state == "flight"
    assert detector.last_velocity_output[2] < -250.0


def test_physics_velocity_recognizes_supported_ball_without_fake_gravity():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)

    for timestamp in np.linspace(0.0, 0.06, 7):
        detector.set_racket_state([0.0, 0.0, 1.0], timestamp)
        velocity = detector._append_velocity_sample(
            timestamp, [0.0, 0.0, 1000.0])

    assert detector.velocity_motion_state == "supported"
    assert detector.velocity_model == "racket_supported"
    np.testing.assert_allclose(velocity, [0.0, 0.0, 0.0], atol=1e-9)


def test_physics_impact_requires_two_rising_samples_and_keeps_post_only_history():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    _seed_falling_flight(detector)

    detector.set_racket_state([0.025, 0.0, 0.982], 0.05)
    candidate_velocity = detector._append_velocity_sample(
        0.05, [25.0, 0.0, 982.0])

    assert detector.last_velocity_impact_detected is False
    assert detector.pending_impact is not None
    assert detector.velocity_motion_state == "impact_confirming"
    assert len(detector.position_history) == 5
    assert candidate_velocity[2] < 0.0

    confirmed_velocity = detector._append_velocity_sample(
        0.06, [30.0, 0.0, 993.5])

    assert detector.last_velocity_impact_detected is True
    assert detector.pending_impact is None
    assert detector.velocity_impact_confirmed_count == 1
    assert detector.last_impact_racket_gate_used is True
    assert len(detector.position_history) == 2
    assert confirmed_velocity[2] > 0.0
    assert detector.velocity_motion_state == "post_impact_fast"


def test_motion_gate_allows_pending_impact_confirmation_sample():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    detector.motion_consistency_gate_enabled = True
    _seed_falling_flight(detector)

    detector.set_racket_state([0.025, 0.0, 0.982], 0.05)
    candidate_position = [25.0, 0.0, 982.0]
    candidate_velocity = detector._append_velocity_sample(
        0.05, candidate_position)
    detector._remember_motion_state(
        0.05, candidate_position, candidate_velocity,
        velocity_ready=True)
    assert detector.pending_impact is not None

    # This is a valid second rising point, but it is deliberately more than
    # the ordinary 50 mm Z innovation allowance away from the pre-impact
    # falling model.  Contact confirmation, rather than that old model, owns
    # the discontinuity.
    confirmation_position = [30.0, 0.0, 1040.0]
    assert detector._motion_measurement_consistent(
        0.06, confirmation_position)
    assert detector.motion_last_gate_reason == "impact_confirmation_bypass"
    assert detector.motion_impact_confirmation_bypass_count == 1

    confirmed_velocity = detector._append_velocity_sample(
        0.06, confirmation_position)
    assert detector.last_velocity_impact_detected is True
    assert confirmed_velocity[2] > 0.0


def test_motion_gate_allows_strong_first_impact_candidate():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    detector.motion_consistency_gate_enabled = True
    _seed_falling_flight(detector)
    detector._remember_motion_state(
        0.04, detector.last_position_estimate,
        detector.last_velocity_output, velocity_ready=True)

    detector.set_racket_state([0.030, 0.0, 1.015], 0.06)
    candidate_position = [30.0, 0.0, 1040.0]
    assert detector._motion_measurement_consistent(
        0.06, candidate_position)
    assert detector.motion_last_gate_reason == "impact_candidate_bypass"
    assert detector.motion_impact_candidate_bypass_count == 1

    candidate_velocity = detector._append_velocity_sample(
        0.06, candidate_position)
    assert detector.pending_impact is not None
    assert candidate_velocity[2] < 0.0

    detector.set_racket_state([0.035, 0.0, 1.025], 0.07)
    confirmation_position = [35.0, 0.0, 1070.0]
    assert detector._motion_measurement_consistent(
        0.07, confirmation_position)
    confirmed_velocity = detector._append_velocity_sample(
        0.07, confirmation_position)
    assert detector.last_velocity_impact_detected is True
    assert confirmed_velocity[2] > 0.0


def test_physics_impact_rejects_single_frame_reversal_without_resetting_track():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    _seed_falling_flight(detector)

    detector._append_velocity_sample(0.05, [25.0, 0.0, 982.0])
    assert detector.pending_impact is not None
    detector._append_velocity_sample(0.06, [30.0, 0.0, 978.0])

    assert detector.last_velocity_impact_detected is False
    assert detector.pending_impact is None
    assert detector.velocity_impact_confirmed_count == 0
    assert detector.velocity_impact_rejected_count == 1
    assert len(detector.position_history) == 5


def test_fresh_far_racket_pose_blocks_false_impact_candidate():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    _seed_falling_flight(detector)

    detector.set_racket_state([1.0, 1.0, 1.0], 0.05)
    detector._append_velocity_sample(0.05, [25.0, 0.0, 982.0])

    assert detector.pending_impact is None
    assert detector.velocity_impact_candidate_count == 0
    assert detector.last_velocity_impact_detected is False


def test_required_racket_gate_fails_closed_when_fk_is_unavailable():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    detector.impact_racket_gate_required = True
    _seed_falling_flight(detector)

    detector._append_velocity_sample(0.05, [25.0, 0.0, 982.0])

    assert detector.pending_impact is None
    assert detector.velocity_impact_candidate_count == 0
    assert detector.last_velocity_impact_detected is False


def test_required_impact_gate_accepts_recent_racket_state_during_fk_jitter():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    detector.impact_racket_gate_required = True
    _seed_falling_flight(detector)
    # Older than the 80 ms support/geometry freshness window, but still recent
    # enough to validate a two-frame trajectory reversal conservatively.
    detector.set_racket_state([0.025, 0.0, 0.982], -0.10)

    detector._append_velocity_sample(0.05, [25.0, 0.0, 982.0])

    assert detector.pending_impact is not None
    assert detector.last_impact_racket_gate_used is True
    assert np.isclose(detector.last_impact_racket_state_age_s, 0.15)


def test_recorded_152353_reversal_resets_segment_and_rejects_late_bad_point():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    detector.motion_consistency_gate_enabled = True
    detector.impact_racket_gate_required = True
    rows = [
        (39.4612312, [-81.009, -444.696, 1182.889]),
        (39.4723442, [-79.117, -444.166, 1165.908]),
        (39.4833230, [-78.948, -445.483, 1147.108]),
        (39.4943726, [-78.799, -445.874, 1127.360]),
        (39.5056580, [-79.064, -451.617, 1117.477]),
        (39.5165827, [-82.463, -466.667, 1133.185]),
        (39.5278847, [-81.214, -468.479, 1165.810]),
        (39.5388803, [-79.586, -468.886, 1196.982]),
        (39.5500230, [-77.274, -463.681, 1234.086]),
        (39.5611744, [-72.880, -456.946, 1265.660]),
    ]
    origin = rows[0][0]
    impact_seen = False
    for index, (absolute_time, position) in enumerate(rows):
        stamp = absolute_time - origin
        if index >= 5:
            detector.set_racket_state([
                position[0] / 1000.0,
                position[1] / 1000.0,
                (position[2] - 25.0) / 1000.0,
            ], stamp)
        assert detector._motion_measurement_consistent(stamp, position)
        velocity = detector._append_velocity_sample(stamp, position)
        detector._remember_motion_state(
            stamp, detector.last_position_estimate, velocity,
            velocity_ready=detector.last_velocity_ready)
        impact_seen = impact_seen or detector.last_velocity_impact_detected

    assert impact_seen is True
    assert detector.velocity_impact_confirmed_count == 1
    assert detector.last_velocity_output[2] > 2000.0

    late_stamp = 39.6101046 - origin
    late_point = [-70.579, -455.792, 1302.997]
    assert not detector._motion_measurement_consistent(
        late_stamp, late_point)
    assert detector.motion_last_gate_reason == "z_innovation"


def test_trajectory_endpoint_smoothing_reduces_latest_position_outlier():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    for index, timestamp in enumerate(np.linspace(0.0, 0.04, 5)):
        clean_x = 100.0 * timestamp
        raw_x = clean_x + (8.0 if index == 4 else 0.0)
        detector._append_velocity_sample(timestamp, [
            raw_x,
            -200.0 * timestamp,
            1000.0 + 1500.0 * timestamp - 4900.0 * timestamp ** 2,
        ])

    clean_last_x = 4.0
    raw_last_x = 12.0
    assert abs(detector.last_position_estimate[0] - clean_last_x) < abs(
        raw_last_x - clean_last_x)
    assert 0.0 < detector.last_position_filter_residual_mm <= 12.0


def test_trajectory_endpoint_smoothing_caps_published_position_step_speed():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    detector.last_position_estimate = np.asarray(
        [0.0, 0.0, 1000.0], dtype=np.float64)
    detector.last_velocity_output_time = 1.0

    detector._commit_velocity_output(
        1.01,
        [100.0, 0.0, 1000.0],
        [112.0, 0.0, 1000.0],
        [0.0, 0.0, 0.0],
        ready=True,
        motion_state="flight",
        velocity_model="test",
    )

    assert np.isclose(detector.last_position_estimate[0], 60.0)


def test_adaptive_flight_velocity_gain_smooths_small_residual_only():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector)
    detector.FLIGHT_VELOCITY_ADAPTIVE_GAIN_ENABLED = True
    detector.last_velocity_ready = True
    detector.velocity_motion_state = "flight"
    detector.last_velocity_output_time = 1.0
    detector.last_velocity_output = np.asarray(
        [0.0, 0.0, 0.0], dtype=np.float64)

    small, _ = detector._stabilize_flight_velocity(
        1.01, np.asarray([100.0, 0.0, -98.0]))
    # A fixed 0.65 update would expose 65 mm/s in X.  Adaptive gain keeps the
    # small noisy residual substantially below that without freezing it.
    assert 35.0 < small[0] < 50.0

    detector.last_velocity_output_time = 2.0
    detector.last_velocity_output = np.zeros(3, dtype=np.float64)
    large, _ = detector._stabilize_flight_velocity(
        2.01, np.asarray([1000.0, 0.0, -98.0]))
    # Large mismatch reaches the established correction cap immediately.
    assert np.isclose(large[0], 450.0, atol=1.0)


def test_live_position_smoothing_switch_clears_old_estimator_segment():
    detector = _detector_for_ir_unit_test(enabled=False)
    _configure_stable_physics_velocity(detector, smoothing=False)
    _seed_falling_flight(detector)
    assert detector.last_velocity_ready is True

    detector.configure_observation_quality(
        clear_buffers_on_detection_loss_enabled=False,
        motion_consistency_gate_enabled=False,
        physics_constrained_velocity_enabled=True,
        trajectory_position_smoothing_enabled=True,
    )

    assert len(detector.position_history) == 0
    assert detector.last_velocity_ready is False
    velocity = detector._append_velocity_sample(
        0.05, [25.0, 0.0, 970.0])
    np.testing.assert_allclose(velocity, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        detector.last_position_estimate, [25.0, 0.0, 970.0])


def test_pixel_kalman_reset_after_outliers_is_configurable():
    pixel_filter = PixelKalmanFilter(
        innovation_gate=5.0,
        reset_after_outliers=1,
    )
    pixel_filter.update([0.0, 0.0])

    updated = pixel_filter.update([50.0, 0.0])

    np.testing.assert_allclose(updated, [50.0, 0.0])


def test_candidate_filter_rejects_robot_blob_and_keeps_ball():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_candidate_filter(
        enabled=True,
        min_brightness=23000.0,
        max_depth_disagreement_mm=180.0,
        motion_scale_px=130.0,
    )
    depth = np.zeros((100, 120), dtype=np.uint16)
    true_mask = np.zeros_like(depth, dtype=np.uint8)
    false_mask = np.zeros_like(depth, dtype=np.uint8)
    cv2.circle(true_mask, (30, 50), 17, 255, -1)
    cv2.circle(false_mask, (90, 50), 12, 255, -1)
    depth[true_mask > 0] = 520
    depth[false_mask > 0] = 430
    candidates = [
        {
            "cx": 30.0, "cy": 50.0, "radius": 17.0,
            "brightness": 50000.0, "mask": true_mask,
        },
        {
            "cx": 90.0, "cy": 50.0, "radius": 12.0,
            "brightness": 12222.0, "mask": false_mask,
        },
    ]

    selected = detector._select_ir_candidate(candidates, depth, 1.0)

    assert selected is not None
    assert selected["cx"] == 30.0
    assert detector.candidate_filter_stats["rejected_brightness"] == 1


def test_candidate_scoring_prefers_motion_consistent_candidate_over_brightest():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_candidate_filter(
        enabled=True,
        min_brightness=10000.0,
        max_depth_disagreement_mm=180.0,
        motion_scale_px=20.0,
    )
    detector.candidate_position_history.extend([
        (0.98, 25.0, 50.0),
        (1.00, 30.0, 50.0),
    ])
    depth = np.full((100, 120), 520, dtype=np.uint16)
    mask = np.full_like(depth, 255, dtype=np.uint8)
    candidates = [
        {"cx": 35.0, "cy": 50.0, "radius": 17.0,
         "brightness": 25000.0, "mask": mask},
        {"cx": 100.0, "cy": 50.0, "radius": 17.0,
         "brightness": 60000.0, "mask": mask},
    ]

    selected = detector._select_ir_candidate(candidates, depth, 1.02)

    assert selected is not None
    assert selected["cx"] == 35.0


def test_candidate_filter_rejects_depth_inconsistent_bright_blob():
    detector = _detector_for_ir_unit_test(enabled=False)
    detector.configure_candidate_filter(
        enabled=True,
        min_brightness=23000.0,
        max_depth_disagreement_mm=180.0,
        motion_scale_px=130.0,
    )
    depth = np.full((100, 120), 430, dtype=np.uint16)
    mask = np.full_like(depth, 255, dtype=np.uint8)
    robot_blob = {
        "cx": 60.0, "cy": 50.0, "radius": 12.0,
        "brightness": 50000.0, "mask": mask,
    }

    selected = detector._select_ir_candidate([robot_blob], depth, 1.0)

    assert selected is None
    assert detector.candidate_filter_stats["rejected_depth"] == 1
