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


def test_local_candidate_mask_reads_matching_depth_crop():
    detector = _detector_for_ir_unit_test(enabled=False)
    depth = np.zeros((100, 120), dtype=np.uint16)
    depth[42:59, 67:84] = 620
    mask = np.full((17, 17), 255, dtype=np.uint8)

    measured = detector._get_depth_from_depthmap(
        75.0, 50.0, depth, mask, (67, 42))

    assert measured == 620.0


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
    assert len(detector.depth_history) == 0

    detector.configure_ir_depth_processing(
        kalman_enabled=False,
        synchronized_depth_enabled=False,
        depth_temporal_window_frames=1,
    )
    assert detector._depth_for_reconstruction(500.0) == 500.0
    assert detector._depth_for_reconstruction(700.0) == 700.0


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
