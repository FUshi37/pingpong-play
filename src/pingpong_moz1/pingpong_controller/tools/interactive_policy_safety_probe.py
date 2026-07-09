#!/usr/bin/env python3
"""Interactive browser probe for ping-pong policy safety behavior.

The Python side runs the same MJX policy adapter used by deployment, plus a
small simulation of the high-level safety gates from pingpong_node:
initial capture gate, ball-lost failsafe, and done-condition monitoring.
The browser side lets the operator drag or hide the ball and watch the
policy/racket response in real time.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
    from mc_core_interface.msg import (
        MechUnitCmd,
        MechUnitCmdArray,
        MechUnitStateArray,
    )
    _ROS_AVAILABLE = True
    _ROS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - ROS bridge is optional.
    rclpy = None
    MultiThreadedExecutor = None
    Node = object
    CameraInfo = None
    CompressedImage = None
    Image = None
    JointState = None
    MechUnitCmd = None
    MechUnitCmdArray = None
    MechUnitStateArray = None
    _ROS_AVAILABLE = False
    _ROS_IMPORT_ERROR = exc


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.ball_state_estimator import (
    BallStateEstimator,
    BallStateEstimatorConfig,
    gravity_for_base_link,
)
from pingpong_controller.done_condition_monitor import DoneConditionMonitor
from pingpong_controller.mjx_rl_policy import MjxRLPolicyController
from pingpong_controller.rl_policy import RIGHT_ARM_JOINTS, target_right_arm_q_rad

try:
    import cv2
    from pingpong_controller.ball_detector import PingPongBallDetector
    from pingpong_controller.calibration_loader import load_calibration
    from pingpong_controller.hybrid_ball_detector_adapter import HybridBallDetectorAdapter
    from pingpong_controller.viz_render import (
        draw_sim_done_overlay,
        render_detection_overlay,
    )
    _CAMERA_BALL_AVAILABLE = True
    _CAMERA_BALL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - camera ball source is optional.
    cv2 = None
    PingPongBallDetector = None
    HybridBallDetectorAdapter = None
    load_calibration = None
    draw_sim_done_overlay = None
    render_detection_overlay = None
    _CAMERA_BALL_AVAILABLE = False
    _CAMERA_BALL_IMPORT_ERROR = exc


DEFAULT_MODEL_PATH = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "outputs" / "rl_sim" / "inverse_compensation"
    / "18_stage4g_strong_contact_dr_inverse_mpc.pkl"
)
DEFAULT_ROBOT_XML = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "models" / "moz1_pd.xml"
)
DEFAULT_VISION_MODEL = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "models" / "best.pt"
)
DEFAULT_CALIBRATION_JSON = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "outputs" / "vision_calib" / "eye_hand_result_848_undistorted.json"
)
DEFAULT_IR_PARAMS_JSON = REPO_ROOT / "ir_params.json"
DEFAULT_DEPTH_CORRECTION_MODEL = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "depth_correction_model.pkl"
)
SOURCE_MODULE_DIR = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
)


def finite_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"expected finite float, got {value!r}")
    return out


def parse_limits(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned or cleaned == "[]":
        return None
    cleaned = cleaned.strip("[]")
    if not cleaned:
        return None
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected 'min,max' limits, got {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise argparse.ArgumentTypeError(
            f"invalid limits {text!r}; expected finite min < max")
    return (lo, hi)


def parse_int_triplet(text: str) -> tuple[int, int, int]:
    cleaned = str(text).strip().strip("[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 'a,b,c' integer triplet, got {text!r}")
    values = tuple(int(p) for p in parts)
    for value in values:
        if value < 0 or value > 255:
            raise argparse.ArgumentTypeError(
                f"HSV values must be in [0, 255], got {text!r}")
    return values


def vec3_from_payload(payload: dict[str, Any], key: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = payload.get(key, default)
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(3)
    except Exception:
        arr = np.asarray(default, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        arr = np.asarray(default, dtype=np.float32)
    return arr


def box_payload(
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    zlim: tuple[float, float] | None,
) -> dict[str, float] | None:
    if not (xlim and ylim and zlim):
        return None
    return {
        "x0": float(xlim[0]), "x1": float(xlim[1]),
        "y0": float(ylim[0]), "y1": float(ylim[1]),
        "z0": float(zlim[0]), "z1": float(zlim[1]),
    }


@dataclass
class ProbeConfig:
    host: str
    port: int
    model_path: Path
    robot_xml_path: Path
    policy_obs_dim: int
    control_rate_hz: float
    ball_lost_timeout_s: float
    enable_ball_capture_gate: bool
    enable_ball_release_gate: bool
    enable_ball_lost_failsafe: bool
    enable_done_monitoring: bool
    ball_release_min_downward_velocity_m_s: float
    ball_release_min_z_drop_m: float
    ball_release_window_s: float
    ball_release_required_consecutive_ticks: int
    ball_release_max_obs_age_s: float
    done_z_min_m: float
    done_z_max_m: float
    done_x_range_m: float
    done_y_range_m: float
    done_racket_anchor_distance_m: float
    done_racket_z_high_offset_m: float
    done_racket_z_low_offset_m: float
    ee_x_limits_m: tuple[float, float] | None
    ee_y_limits_m: tuple[float, float] | None
    ee_z_limits_m: tuple[float, float] | None
    ball_noise_std_m: float
    ball_observation_rate_hz: float
    browser_step_rate_hz: float
    display_rate_hz: float
    enable_ball_state_estimator: bool
    ball_estimator_enable_prediction: bool
    ball_estimator_enable_aero_drag: bool
    ball_estimator_max_extrapolation_s: float
    ball_estimator_trust_loss_s: float
    ball_estimator_degrade_loss_s: float
    ball_estimator_invalid_after_s: float
    ball_estimator_prediction_horizon_s: float
    ros_bridge: bool
    ros_publish_commands: bool
    command_topic: str
    joint_state_topic: str
    cart_state_topic: str
    right_arm_mu_idx: int
    enable_camera_ball: bool
    camera_source_mode: str
    use_hybrid_detector: bool
    use_depth_sensor: bool
    image_width: int
    image_height: int
    realsense_fps: int
    camera_image_topic: str
    camera_raw_image_topic: str
    camera_image_transport: str
    camera_info_topic: str
    calibration_json_path: str
    yolo_model_path: Path
    target_class: str
    conf_threshold: float
    ir_params_json: str
    yolo_interval: int
    ir_threshold: int
    use_adaptive_ir: bool
    ir_exposure: int | None
    ir_gain: int | None
    ir_laser_power: int
    ir_min_radius: int
    ir_max_radius: int
    ir_circularity_min: float
    ir_brightness_percentile: int
    hybrid_roi_margin: int
    depth_correction_model_path: str
    depth_neighborhood_px: int
    infer_imgsz: int
    detector_device: str
    detector_half: bool
    use_kalman: bool
    camera_tilt_deg: float
    use_undistort: bool
    enable_ellipse_fit: bool
    enable_temporal_tracker: bool
    tracker_max_miss_bridge: int
    tracker_gate_px: float
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    color_match_threshold: float
    color_circularity_min: float
    enable_camera_detection: bool
    enable_camera_visualization: bool
    camera_visualization_rate_hz: float
    camera_web_max_width: int
    camera_web_jpeg_quality: int
    enable_camera_color_filter: bool
    sim_cartesian_x_offset_m: float
    sim_cartesian_y_offset_m: float
    sim_base_height_m: float

    @property
    def control_dt(self) -> float:
        return 1.0 / max(self.control_rate_hz, 1e-6)


@dataclass(frozen=True)
class CameraRuntimeOptions:
    detect: bool
    visualize: bool
    use_depth_sensor: bool
    use_undistort: bool
    use_kalman: bool
    enable_ellipse_fit: bool
    enable_temporal_tracker: bool
    enable_color_filter: bool

    def key(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        return (
            bool(self.use_depth_sensor),
            bool(self.use_undistort),
            bool(self.use_kalman),
            bool(self.enable_ellipse_fit),
            bool(self.enable_temporal_tracker),
            bool(self.enable_color_filter),
        )

    def payload(self) -> dict[str, bool]:
        return {
            "detect": bool(self.detect),
            "visualize": bool(self.visualize),
            "sensorDepth": bool(self.use_depth_sensor),
            "undistort": bool(self.use_undistort),
            "kalman": bool(self.use_kalman),
            "ellipseFit": bool(self.enable_ellipse_fit),
            "temporalTracker": bool(self.enable_temporal_tracker),
            "colorFilter": bool(self.enable_color_filter),
        }


@dataclass(frozen=True)
class RuntimeOptions:
    use_ros_state: bool
    publish_commands: bool
    enable_ball_capture_gate: bool
    enable_ball_release_gate: bool
    enable_ball_lost_failsafe: bool
    enable_done_monitoring: bool
    use_ball_estimator: bool
    estimator_prediction: bool
    estimator_aero_drag: bool
    ball_observation_rate_hz: float
    ball_source: str
    camera: CameraRuntimeOptions


def _expand_path_text(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(path)))


def _stamp_to_sec(stamp: Any) -> float:
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return time.time()


class RosBridgeNode(Node):
    def __init__(self, cfg: ProbeConfig):
        super().__init__("interactive_policy_safety_probe")
        self.cfg = cfg
        self.lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.camera_rebuild_lock = threading.Lock()
        self.latest_joint_q: np.ndarray | None = None
        self.latest_joint_dq: np.ndarray | None = None
        self.latest_joint_stamp_wall: float | None = None
        self.latest_cart_pos: np.ndarray | None = None
        self.latest_cart_quat_xyzw: np.ndarray | None = None
        self.latest_cart_psi: float | None = None
        self.latest_cart_stamp_wall: float | None = None
        self.publish_count = 0
        self.cmd_pub = None

        self.camera_detector = None
        self.camera_detector_key: tuple[Any, ...] | None = None
        self.camera_direct_running = False
        self.camera_direct_thread: threading.Thread | None = None
        self.camera_info_payload: dict[str, Any] | None = None
        self.camera_info_received = False
        self.camera_output_frame: str | None = None
        self.camera_error: str | None = None
        self.camera_options = CameraRuntimeOptions(
            detect=bool(cfg.enable_camera_detection),
            visualize=bool(cfg.enable_camera_visualization),
            use_depth_sensor=bool(cfg.use_depth_sensor),
            use_undistort=bool(cfg.use_undistort),
            use_kalman=bool(cfg.use_kalman),
            enable_ellipse_fit=bool(cfg.enable_ellipse_fit),
            enable_temporal_tracker=bool(cfg.enable_temporal_tracker),
            enable_color_filter=bool(cfg.enable_camera_color_filter),
        )
        self.camera_frame_count = 0
        self.camera_detection_count = 0
        self.camera_callback_times: deque[float] = deque(maxlen=30)
        self.camera_last_callback_wall: float | None = None
        self.camera_active_transport: str | None = None
        self.camera_last_image_wall_by_transport: dict[str, float] = {}
        self.latest_camera_ball_pos_m: np.ndarray | None = None
        self.latest_camera_ball_vel_m_s: np.ndarray | None = None
        self.latest_camera_detector_pos_m: np.ndarray | None = None
        self.latest_camera_detector_vel_m_s: np.ndarray | None = None
        self.latest_camera_detector_frame: str | None = None
        self.latest_camera_valid = False
        self.latest_camera_stamp_sec: float | None = None
        self.latest_camera_wall: float | None = None
        self.latest_camera_raw_measurement: Any = None
        self.latest_viz_jpeg: bytes | None = None
        self.latest_viz_seq = 0
        self.latest_viz_wall: float | None = None
        self.latest_policy_viz_state: dict[str, Any] = {}
        self.camera_viz_slot_lock = threading.Lock()
        self.camera_viz_event = threading.Event()
        self.camera_viz_pending: tuple[np.ndarray, np.ndarray | None, dict[str, Any], bool] | None = None
        self.camera_viz_thread: threading.Thread | None = None
        self.camera_viz_running = False
        self.camera_viz_last_enqueue_wall = 0.0
        self.camera_viz_min_period_s = (
            1.0 / max(float(cfg.camera_visualization_rate_hz), 1e-6))

        if cfg.ros_bridge:
            self.cmd_pub = self.create_publisher(
                MechUnitCmdArray, cfg.command_topic, 10)
            self.create_subscription(
                JointState, cfg.joint_state_topic, self._joint_state_callback, 10)
            self.create_subscription(
                MechUnitStateArray, cfg.cart_state_topic, self._cart_state_callback, 10)
            self.get_logger().info(
                f"ROS feedback bridge active: publish {cfg.command_topic}, "
                f"subscribe {cfg.joint_state_topic}, {cfg.cart_state_topic}")

        if cfg.enable_camera_ball:
            self._init_camera_ball_source()

    def _joint_state_callback(self, msg) -> None:
        name_to_pos: dict[str, float] = {}
        name_to_vel: dict[str, float] = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                value = float(msg.position[i])
                if np.isfinite(value):
                    name_to_pos[str(name)] = value
            if i < len(msg.velocity):
                value = float(msg.velocity[i])
                if np.isfinite(value):
                    name_to_vel[str(name)] = value
        try:
            q = np.asarray([name_to_pos[n] for n in RIGHT_ARM_JOINTS], dtype=np.float32)
        except KeyError:
            return
        dq_vals = [name_to_vel.get(n, np.nan) for n in RIGHT_ARM_JOINTS]
        dq = np.asarray(dq_vals, dtype=np.float32)
        if not np.all(np.isfinite(dq)):
            dq = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        with self.lock:
            self.latest_joint_q = q
            self.latest_joint_dq = dq
            self.latest_joint_stamp_wall = time.time()

    def _cart_state_callback(self, msg) -> None:
        for state in msg.states:
            if int(state.mu_idx) != int(self.cfg.right_arm_mu_idx):
                continue
            pose = state.end_pose
            pos = np.asarray([
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ], dtype=np.float32)
            quat = np.asarray([
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ], dtype=np.float32)
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
                return
            psi = float(state.psi)
            with self.lock:
                self.latest_cart_pos = pos
                self.latest_cart_quat_xyzw = quat
                self.latest_cart_psi = psi if np.isfinite(psi) else None
                self.latest_cart_stamp_wall = time.time()
            return

    def _init_camera_ball_source(self) -> None:
        if not _CAMERA_BALL_AVAILABLE:
            raise RuntimeError(
                f"camera ball source unavailable: {_CAMERA_BALL_IMPORT_ERROR}")
        source_mode = str(self.cfg.camera_source_mode).strip().lower()
        if source_mode == "direct_realsense":
            self._init_direct_realsense_camera_source()
            return
        if source_mode != "ros_topic":
            raise RuntimeError(f"unsupported camera source mode: {source_mode}")
        if CameraInfo is None or CompressedImage is None or Image is None:
            raise RuntimeError("camera ball source requires sensor_msgs CameraInfo/Image/CompressedImage")
        self.create_subscription(
            CameraInfo, self.cfg.camera_info_topic,
            self._camera_info_callback, 10)
        transport = str(self.cfg.camera_image_transport).strip().lower()
        if transport in ("compressed", "auto"):
            self.create_subscription(
                CompressedImage, self.cfg.camera_image_topic,
                self._compressed_image_callback, 10)
        if transport in ("raw", "auto"):
            self.create_subscription(
                Image, self.cfg.camera_raw_image_topic,
                self._raw_image_callback, 10)
        self.get_logger().info(
            "Camera ball source active: "
            f"transport={transport}, compressed={self.cfg.camera_image_topic}, "
            f"raw={self.cfg.camera_raw_image_topic}, info={self.cfg.camera_info_topic}")

    def _load_direct_realsense_calibration(self):
        if load_calibration is None:
            return None
        calibration_path = self._resolve_calibration_json_path(
            self.cfg.calibration_json_path,
            int(self.cfg.image_width),
            int(self.cfg.image_height))
        if not os.path.exists(calibration_path):
            self.get_logger().warn(
                f"No calibration JSON found for direct RealSense: "
                f"{calibration_path}; output will stay in camera frame")
            return None
        try:
            calibration = load_calibration(calibration_path)
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to load direct RealSense calibration {calibration_path}: "
                f"{exc}; output will stay in camera frame")
            return None
        self.get_logger().info(
            f"Loaded T_base_camera for direct RealSense from: {calibration_path}")
        return calibration

    def _init_direct_realsense_camera_source(self) -> None:
        if not self.cfg.use_hybrid_detector:
            raise RuntimeError(
                "direct_realsense camera source currently requires "
                "--use-hybrid-detector-compatible defaults; use "
                "--camera-source-mode ros_topic for the RGB detector path")
        if HybridBallDetectorAdapter is None:
            raise RuntimeError(
                f"HybridBallDetectorAdapter unavailable: {_CAMERA_BALL_IMPORT_ERROR}")

        calibration = self._load_direct_realsense_calibration()
        if calibration is not None:
            width = int(calibration.image_width)
            height = int(calibration.image_height)
            output_frame = "base_link"
        else:
            width = int(self.cfg.image_width)
            height = int(self.cfg.image_height)
            output_frame = "camera"

        detector = HybridBallDetectorAdapter(
            yolo_model_path=str(self.cfg.yolo_model_path),
            ball_diameter_mm=40.0,
            width=width,
            height=height,
            fps=int(self.cfg.realsense_fps),
            yolo_interval=int(self.cfg.yolo_interval),
            yolo_conf=float(self.cfg.conf_threshold),
            target_class=str(self.cfg.target_class),
            ir_params_json=_expand_path_text(self.cfg.ir_params_json),
            ir_threshold=int(self.cfg.ir_threshold),
            use_adaptive=bool(self.cfg.use_adaptive_ir),
            min_radius=int(self.cfg.ir_min_radius),
            max_radius=int(self.cfg.ir_max_radius),
            circularity_min=float(self.cfg.ir_circularity_min),
            laser_power=int(self.cfg.ir_laser_power),
            exposure=self.cfg.ir_exposure,
            gain=self.cfg.ir_gain,
            brightness_percentile=int(self.cfg.ir_brightness_percentile),
            roi_margin=int(self.cfg.hybrid_roi_margin),
            calibration_data=calibration,
            output_frame=output_frame,
            print_every=999999,
            use_undistort=True,
            depth_scale=0.001,
            depth_correction_model_path=_expand_path_text(
                self.cfg.depth_correction_model_path),
            depth_neighborhood_px=int(self.cfg.depth_neighborhood_px),
            color_match_threshold=0.0,
            color_circularity_min=0.0,
            use_depth_sensor=bool(self.cfg.use_depth_sensor),
        )
        with self.camera_lock:
            self.camera_detector = detector
            self.camera_detector_key = (
                "direct_realsense", "hybrid", width, height,
                int(self.cfg.realsense_fps), output_frame)
            self.camera_info_payload = {
                "width": width,
                "height": height,
                "source": "direct_realsense",
            }
            self.camera_info_received = True
            self.camera_output_frame = output_frame
            self.camera_active_transport = "direct_realsense/hybrid"
            self.camera_error = None

        self.camera_direct_running = True
        self.camera_direct_thread = threading.Thread(
            target=self._direct_hybrid_reader_loop,
            name="policy_probe_direct_realsense",
            daemon=True,
        )
        self.camera_direct_thread.start()
        self.get_logger().info(
            "Camera ball source active: direct_realsense/hybrid "
            f"{width}x{height}@{int(self.cfg.realsense_fps)}fps, "
            f"depth={bool(self.cfg.use_depth_sensor)}, output={output_frame}")

    def _resolve_calibration_json_path(
        self,
        requested_path: str,
        image_width: int,
        image_height: int,
    ) -> str:
        if requested_path and str(requested_path).lower() not in ("auto", ""):
            return _expand_path_text(requested_path)

        search_dirs: list[str] = []
        for base_dir in (SOURCE_MODULE_DIR, SCRIPT_PATH.parents[1]):
            calib_dir = os.path.join(str(base_dir), "outputs", "vision_calib")
            if calib_dir not in search_dirs:
                search_dirs.append(calib_dir)

        candidates: list[str] = []
        for calib_dir in search_dirs:
            candidates.extend(glob.glob(os.path.join(
                calib_dir, "eye_hand_result_*_undistorted.json")))
        candidates = sorted(set(candidates))
        for candidate in candidates:
            try:
                calibration = load_calibration(candidate)
            except Exception:
                continue
            if (int(calibration.image_width) == int(image_width)
                    and int(calibration.image_height) == int(image_height)):
                self.get_logger().info(
                    f"Auto-selected calibration for "
                    f"{image_width}x{image_height}: {candidate}")
                return candidate

        fallback = os.path.join(
            search_dirs[0],
            f"eye_hand_result_{int(image_width)}_undistorted.json")
        self.get_logger().warn(
            f"No calibration JSON matched {image_width}x{image_height}; "
            f"trying {fallback}")
        return fallback

    def _load_extrinsics_for_resolution(self, image_width: int, image_height: int):
        calibration_path = self._resolve_calibration_json_path(
            self.cfg.calibration_json_path,
            int(image_width),
            int(image_height))

        if not os.path.exists(calibration_path):
            self.get_logger().warn(
                f"No calibration JSON found for {image_width}x{image_height}: "
                f"{calibration_path}; output will stay in camera frame")
            return None
        try:
            calibration = load_calibration(calibration_path)
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to load extrinsics calibration {calibration_path}: {exc}; "
                "output will stay in camera frame")
            return None
        if (int(calibration.image_width) != int(image_width)
                or int(calibration.image_height) != int(image_height)):
            self.get_logger().warn(
                f"Calibration JSON resolution {calibration.image_width}x"
                f"{calibration.image_height} does not match CameraInfo "
                f"{image_width}x{image_height}; output will stay in camera frame")
            return None
        self.get_logger().info(
            f"Loaded T_base_camera for camera ball source from: {calibration_path}")
        return calibration

    def _camera_detector_key(
        self,
        info: dict[str, Any],
        options: CameraRuntimeOptions,
    ) -> tuple[Any, ...]:
        return (
            int(info["width"]),
            int(info["height"]),
            round(float(info["fx"]), 6),
            round(float(info["fy"]), 6),
            round(float(info["cx"]), 6),
            round(float(info["cy"]), 6),
            str(self.cfg.calibration_json_path),
            str(self.cfg.yolo_model_path),
            str(self.cfg.target_class),
            round(float(self.cfg.conf_threshold), 6),
            int(self.cfg.infer_imgsz),
            str(self.cfg.detector_device),
            bool(self.cfg.detector_half),
            round(float(self.cfg.camera_tilt_deg), 6),
            int(self.cfg.tracker_max_miss_bridge),
            round(float(self.cfg.tracker_gate_px), 6),
            tuple(self.cfg.hsv_lower),
            tuple(self.cfg.hsv_upper),
            round(float(self.cfg.color_match_threshold), 6),
            round(float(self.cfg.color_circularity_min), 6),
            options.key(),
        )

    def set_camera_runtime_options(self, options: CameraRuntimeOptions) -> None:
        if not self.cfg.enable_camera_ball:
            return
        with self.camera_lock:
            old_key = self.camera_options.key()
            self.camera_options = options
            new_key = options.key()
        if str(self.cfg.camera_source_mode).strip().lower() == "direct_realsense":
            return
        if old_key != new_key:
            self._rebuild_camera_detector("runtime option change")

    def set_policy_viz_state(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        valid: bool,
        racket_anchor_m: np.ndarray,
        racket_pos_m: np.ndarray,
    ) -> None:
        with self.camera_lock:
            self.latest_policy_viz_state = {
                "ball_pos_m": np.asarray(ball_pos_m, dtype=np.float32).reshape(3).copy(),
                "ball_vel_m_s": np.asarray(ball_vel_m_s, dtype=np.float32).reshape(3).copy(),
                "valid": bool(valid),
                "racket_anchor_m": np.asarray(racket_anchor_m, dtype=np.float32).reshape(3).copy(),
                "racket_pos_m": np.asarray(racket_pos_m, dtype=np.float32).reshape(3).copy(),
                "stamp_wall": time.time(),
            }

    def _camera_info_callback(self, msg) -> None:
        if PingPongBallDetector is None:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        dist_coeffs = list(msg.d) if len(msg.d) > 0 else None
        width = int(msg.width)
        height = int(msg.height)
        self.get_logger().info(
            "CameraInfo received for interactive probe: "
            f"{width}x{height}, fx={fx:.2f}, fy={fy:.2f}, "
            f"cx={cx:.2f}, cy={cy:.2f}")
        with self.camera_lock:
            if self.camera_info_received:
                return
            self.camera_info_payload = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "dist_coeffs": dist_coeffs,
                "width": width,
                "height": height,
                "distortion_model": str(getattr(msg, "distortion_model", "")),
            }
            self.camera_info_received = True
        self._rebuild_camera_detector("CameraInfo")

    def _rebuild_camera_detector(self, reason: str) -> None:
        if not self.cfg.enable_camera_ball or PingPongBallDetector is None:
            return
        with self.camera_rebuild_lock:
            with self.camera_lock:
                info = None if self.camera_info_payload is None else dict(self.camera_info_payload)
                options = self.camera_options
                current_key = self.camera_detector_key
            if info is None:
                return
            new_key = self._camera_detector_key(info, options)
            if current_key == new_key and self.camera_detector is not None:
                return
            try:
                detector, output_frame = self._build_camera_detector(info, options)
            except Exception as exc:
                with self.camera_lock:
                    self.camera_detector = None
                    self.camera_detector_key = None
                    self.camera_output_frame = None
                    self.camera_error = str(exc)
                self.get_logger().error(f"Failed to initialize camera detector: {exc}")
                raise
            with self.camera_lock:
                self.camera_detector = detector
                self.camera_detector_key = new_key
                self.camera_output_frame = output_frame
                self.camera_error = None
            self.get_logger().info(
                f"Camera ball detector initialized ({reason}, output: {output_frame}, "
                f"options={options.payload()})")

    def _build_camera_detector(
        self,
        info: dict[str, Any],
        options: CameraRuntimeOptions,
    ):
        width = int(info["width"])
        height = int(info["height"])
        extrinsics_calibration = self._load_extrinsics_for_resolution(width, height)
        output_frame = "base_link" if extrinsics_calibration is not None else "camera"
        color_match_threshold = (
            self.cfg.color_match_threshold
            if options.enable_color_filter else 0.0)
        color_circularity_min = (
            self.cfg.color_circularity_min
            if options.enable_color_filter else 0.0)
        detector = PingPongBallDetector(
            yolo_model_path=str(self.cfg.yolo_model_path),
            ball_diameter_mm=40.0,
            width=width,
            height=height,
            fps=60,
            conf_threshold=self.cfg.conf_threshold,
            target_class=self.cfg.target_class,
            camera_tilt_deg=self.cfg.camera_tilt_deg,
            use_kalman=options.use_kalman,
            infer_imgsz=self.cfg.infer_imgsz,
            device=self.cfg.detector_device,
            half=self.cfg.detector_half,
            use_undistort=options.use_undistort,
            print_every=999999,
            profile_every=0,
            use_camera=False,
            camera_fx=float(info["fx"]),
            camera_fy=float(info["fy"]),
            camera_cx=float(info["cx"]),
            camera_cy=float(info["cy"]),
            camera_dist_coeffs=info.get("dist_coeffs"),
            output_frame=output_frame,
            enable_ellipse_fit=options.enable_ellipse_fit,
            enable_temporal_tracker=options.enable_temporal_tracker,
            tracker_max_miss_bridge=self.cfg.tracker_max_miss_bridge,
            tracker_gate_px=self.cfg.tracker_gate_px,
            hsv_lower=list(self.cfg.hsv_lower),
            hsv_upper=list(self.cfg.hsv_upper),
            color_match_threshold=color_match_threshold,
            color_circularity_min=color_circularity_min,
        )
        if extrinsics_calibration is not None:
            detector.calibration_data = extrinsics_calibration
            detector.T_base_camera = extrinsics_calibration.T_base_camera
            detector.R_base_camera = detector.T_base_camera[:3, :3]
            detector.output_frame = "base_link"
            output_frame = "base_link"
        return detector, output_frame

    def _should_process_transport(self, transport: str) -> bool:
        now = time.time()
        with self.camera_lock:
            self.camera_last_image_wall_by_transport[transport] = now
            active = self.camera_active_transport
            if active is None:
                self.camera_active_transport = transport
                return True
            if active == transport:
                return True
            active_age = now - self.camera_last_image_wall_by_transport.get(active, 0.0)
            if active_age > 1.0:
                self.camera_active_transport = transport
                return True
            return False

    def _compressed_image_callback(self, msg) -> None:
        if not self._should_process_transport("compressed"):
            return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warn(
                    "Failed to decode compressed camera image",
                    throttle_duration_sec=1.0)
                return
            stamp_sec = _stamp_to_sec(msg.header.stamp)
            self._handle_cv_image(image, stamp_sec)
        except Exception as exc:
            with self.camera_lock:
                self.camera_error = str(exc)
            self.get_logger().error(
                f"Error in compressed camera image callback: {exc}",
                throttle_duration_sec=1.0)

    def _raw_image_callback(self, msg) -> None:
        if not self._should_process_transport("raw"):
            return
        try:
            image = self._decode_raw_image_msg(msg)
            stamp_sec = _stamp_to_sec(msg.header.stamp)
            self._handle_cv_image(image, stamp_sec)
        except Exception as exc:
            with self.camera_lock:
                self.camera_error = str(exc)
            self.get_logger().error(
                f"Error in raw camera image callback: {exc}",
                throttle_duration_sec=1.0)

    def _decode_raw_image_msg(self, msg) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid raw image size {width}x{height}")
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
            "8uc1": 1,
            "8uc3": 3,
            "8uc4": 4,
        }
        channels = channels_by_encoding.get(encoding)
        if channels is None:
            raise ValueError(f"unsupported raw image encoding: {msg.encoding}")
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        step = int(msg.step)
        min_step = width * channels
        if step < min_step:
            raise ValueError(
                f"invalid raw image step {step} for {width}x{height} {encoding}")
        rows = raw.reshape((height, step))[:, :min_step]
        if channels == 1:
            image = rows.reshape((height, width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = rows.reshape((height, width, channels))
        if encoding in ("rgb8", "8uc3"):
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _direct_hybrid_reader_loop(self) -> None:
        self.get_logger().info("Direct RealSense hybrid reader loop: entering")
        with self.camera_lock:
            detector = self.camera_detector
        pipeline = getattr(getattr(detector, "detector", None), "pipeline", None)
        if pipeline is None:
            with self.camera_lock:
                self.camera_error = "direct RealSense hybrid pipeline is unavailable"
            return

        while self.camera_direct_running:
            try:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as exc:
                    with self.camera_lock:
                        self.camera_error = f"wait_for_frames timeout: {exc}"
                    self.get_logger().warn(
                        f"wait_for_frames timeout: {exc}",
                        throttle_duration_sec=1.0)
                    continue

                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame(1)
                if not color_frame:
                    continue
                rgb_image = np.asanyarray(color_frame.get_data())
                if rgb_image is None or rgb_image.size == 0:
                    continue
                ir_image = (
                    np.asanyarray(ir_frame.get_data())
                    if ir_frame is not None else None)
                stamp = self.get_clock().now().to_msg()
                stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9

                with self.camera_lock:
                    options = self.camera_options
                    detector = self.camera_detector
                    self.camera_last_image_wall_by_transport[
                        "direct_realsense"] = time.time()
                    self.camera_active_transport = "direct_realsense/hybrid"

                if not options.detect:
                    observation = {
                        "detected": False,
                        "position": [0, 0, 0],
                        "velocity": [0, 0, 0],
                        "raw_measurement": {"detection_disabled": True},
                    }
                elif detector is None:
                    observation = {
                        "detected": False,
                        "position": [0, 0, 0],
                        "velocity": [0, 0, 0],
                        "raw_measurement": {"no_detector": True},
                    }
                elif not options.use_depth_sensor:
                    if ir_image is None:
                        observation = {
                            "detected": False,
                            "position": [0, 0, 0],
                            "velocity": [0, 0, 0],
                            "raw_measurement": {"missing_ir_frame": True},
                        }
                    else:
                        observation = detector.detector.get_observation_offline(
                            rgb_image, ir_image, None, stamp_sec)
                else:
                    observation = detector.detector.get_observation(
                        observation_time=stamp_sec,
                        frames=frames,
                    )
                    if observation is None:
                        observation = {
                            "detected": False,
                            "position": [0, 0, 0],
                            "velocity": [0, 0, 0],
                            "raw_measurement": {"no_observation": True},
                        }

                self._handle_direct_hybrid_observation(
                    rgb_image, ir_image, observation, stamp_sec)
            except Exception as exc:  # noqa: BLE001 - keep reader alive.
                with self.camera_lock:
                    self.camera_error = str(exc)
                self.get_logger().error(
                    f"Direct RealSense hybrid reader error: {exc}",
                    throttle_duration_sec=1.0)
                time.sleep(0.1)
        self.get_logger().info("Direct RealSense hybrid reader loop: exited")

    def _handle_direct_hybrid_observation(
        self,
        image: np.ndarray,
        ir_image: np.ndarray | None,
        observation: dict[str, Any],
        stamp_sec: float,
    ) -> None:
        now_wall = time.time()
        with self.camera_lock:
            if self.camera_last_callback_wall is not None:
                dt = now_wall - self.camera_last_callback_wall
                if dt > 0.0:
                    self.camera_callback_times.append(dt)
            self.camera_last_callback_wall = now_wall
            self.camera_frame_count += 1
            if bool(observation.get("detected", False)) and not bool(
                    observation.get("predicted", False)):
                self.camera_detection_count += 1
            options = self.camera_options
            detector = self.camera_detector

        self._store_camera_observation(observation, stamp_sec)
        if not options.visualize:
            return
        if ir_image is not None and detector is not None:
            try:
                viz_image = detector.detector._render_visualization(
                    image, ir_image, observation)
                if viz_image is not None:
                    hybrid_obs_for_viz = dict(observation)
                    hybrid_obs_for_viz["_hybrid_viz_rendered"] = True
                    self._enqueue_camera_visualization(
                        viz_image, hybrid_obs_for_viz)
                    return
            except Exception as exc:  # noqa: BLE001 - fallback below.
                with self.camera_lock:
                    self.camera_error = f"hybrid visualization failed: {exc}"
        self._enqueue_camera_visualization(image, observation)

    def _handle_cv_image(self, image: np.ndarray, stamp_sec: float) -> None:
        now_wall = time.time()
        with self.camera_lock:
            if self.camera_last_callback_wall is not None:
                dt = now_wall - self.camera_last_callback_wall
                if dt > 0.0:
                    self.camera_callback_times.append(dt)
            self.camera_last_callback_wall = now_wall
            options = self.camera_options
            detector = self.camera_detector
            detector_ready = bool(self.camera_info_received and detector is not None)

        if not options.detect:
            observation = {
                "detected": False,
                "raw_measurement": {"detection_disabled": True},
            }
        elif not detector_ready:
            observation = {
                "detected": False,
                "raw_measurement": {"waiting_for_camera_info": not self.camera_info_received},
            }
        else:
            observation = detector.get_observation(
                image, observation_time=stamp_sec)

        detected = bool(observation.get("detected", False))
        is_predicted = bool(observation.get("predicted", False))
        with self.camera_lock:
            self.camera_frame_count += 1
            if detected and not is_predicted:
                self.camera_detection_count += 1

        self._store_camera_observation(observation, stamp_sec)
        if options.visualize:
            self._enqueue_camera_visualization(image, observation)

    def _transform_ball_pos_to_sim_frame(self, ball_pos_m: np.ndarray) -> np.ndarray:
        transformed = np.asarray(ball_pos_m, dtype=np.float32).reshape(3).copy()
        transformed[0] += float(self.cfg.sim_cartesian_x_offset_m)
        transformed[1] += float(self.cfg.sim_cartesian_y_offset_m)
        transformed[2] += float(self.cfg.sim_base_height_m)
        return transformed

    def _store_camera_observation(self, observation: dict[str, Any], stamp_sec: float) -> None:
        now_wall = time.time()
        detected = bool(observation.get("detected", False))
        pos_m = None
        vel_m_s = None
        detector_pos_m = None
        detector_vel_m_s = None
        with self.camera_lock:
            detector_frame = self.camera_output_frame or "camera"
        if detected:
            try:
                detector_pos_m = np.asarray(
                    observation.get("position", [0.0, 0.0, 0.0]),
                    dtype=np.float32).reshape(3) * 1e-3
                detector_vel_m_s = np.asarray(
                    observation.get("velocity", [0.0, 0.0, 0.0]),
                    dtype=np.float32).reshape(3) * 1e-3
                if detector_frame == "sim_frame":
                    pos_m = detector_pos_m.astype(np.float32)
                else:
                    pos_m = self._transform_ball_pos_to_sim_frame(detector_pos_m)
                vel_m_s = detector_vel_m_s.astype(np.float32)
                if not (np.all(np.isfinite(detector_pos_m))
                        and np.all(np.isfinite(detector_vel_m_s))
                        and np.all(np.isfinite(pos_m))
                        and np.all(np.isfinite(vel_m_s))):
                    detected = False
                    pos_m = None
                    vel_m_s = None
                    detector_pos_m = None
                    detector_vel_m_s = None
            except Exception:
                detected = False
                pos_m = None
                vel_m_s = None
                detector_pos_m = None
                detector_vel_m_s = None
        with self.camera_lock:
            self.latest_camera_ball_pos_m = pos_m
            self.latest_camera_ball_vel_m_s = vel_m_s
            self.latest_camera_detector_pos_m = detector_pos_m
            self.latest_camera_detector_vel_m_s = detector_vel_m_s
            self.latest_camera_detector_frame = detector_frame
            self.latest_camera_valid = bool(detected)
            self.latest_camera_stamp_sec = float(stamp_sec)
            self.latest_camera_wall = now_wall
            self.latest_camera_raw_measurement = observation.get("raw_measurement")
            self.camera_error = None

    def _viz_done_thresholds(self) -> dict[str, float]:
        return {
            "ball_z_floor": float(self.cfg.done_z_min_m),
            "ball_z_ceiling": float(self.cfg.done_z_max_m),
            "ball_x_range": float(self.cfg.done_x_range_m),
            "ball_y_range": float(self.cfg.done_y_range_m),
            "racket_anchor_distance": float(self.cfg.done_racket_anchor_distance_m),
            "racket_z_high_offset": float(self.cfg.done_racket_z_high_offset_m),
            "racket_z_low_offset": float(self.cfg.done_racket_z_low_offset_m),
        }

    def _compute_viz_sim_ball_state(self, observation: dict[str, Any]):
        with self.camera_lock:
            state = dict(self.latest_policy_viz_state)
        if state:
            return (
                np.asarray(state["ball_pos_m"], dtype=np.float64).reshape(3),
                np.asarray(state["ball_vel_m_s"], dtype=np.float64).reshape(3),
                bool(state.get("valid", False)),
                "[SIM control state]",
            )
        if not bool(observation.get("detected", False)):
            return None, None, False, "[SIM raw frame]"
        pos = observation.get("position")
        if pos is None:
            return None, None, False, "[SIM raw frame]"
        vel = observation.get("velocity")
        if vel is None:
            vel = [0.0, 0.0, 0.0]
        try:
            pos_m = np.asarray(pos, dtype=np.float64).reshape(3) * 1e-3
            vel_m_s = np.asarray(vel, dtype=np.float64).reshape(3) * 1e-3
        except (TypeError, ValueError):
            return None, None, False, "[SIM raw frame]"
        return self._transform_ball_pos_to_sim_frame(pos_m), vel_m_s, True, "[SIM raw frame]"

    def _render_visualization_frame(
        self,
        frame: np.ndarray,
        observation: dict[str, Any],
    ) -> np.ndarray:
        sim_ball_pos_m, sim_ball_vel_m_s, sim_detected, sim_label = (
            self._compute_viz_sim_ball_state(observation))
        with self.camera_lock:
            frame_count = int(self.camera_frame_count)
            detection_count = int(self.camera_detection_count)
            callback_times = list(self.camera_callback_times)
            output_frame = self.camera_output_frame or "camera"
            state = dict(self.latest_policy_viz_state)
        if callback_times:
            avg_dt = sum(callback_times) / len(callback_times)
            callback_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            callback_hz = 0.0
        racket_anchor_m = state.get("racket_anchor_m")
        racket_pos_m = state.get("racket_pos_m")
        if observation.get("_hybrid_viz_rendered", False):
            if (draw_sim_done_overlay is not None
                    and sim_ball_pos_m is not None
                    and sim_ball_vel_m_s is not None):
                draw_sim_done_overlay(
                    frame,
                    sim_ball_pos_m,
                    sim_ball_vel_m_s,
                    racket_anchor_m,
                    racket_pos_m=racket_pos_m,
                    thresholds=self._viz_done_thresholds(),
                    detected=sim_detected,
                    label=sim_label,
                )
            return frame
        if render_detection_overlay is None:
            return frame
        return render_detection_overlay(
            frame,
            observation,
            frame_count=frame_count,
            detection_count=detection_count,
            callback_hz=callback_hz,
            output_frame=output_frame,
            sim_ball_pos_m=sim_ball_pos_m,
            sim_ball_vel_m_s=sim_ball_vel_m_s,
            racket_anchor_m=racket_anchor_m,
            racket_pos_m=racket_pos_m,
            done_thresholds=self._viz_done_thresholds(),
            sim_detected=sim_detected,
            sim_label=sim_label,
        )

    def _start_camera_visualization_worker(self) -> None:
        if self.camera_viz_thread is not None and self.camera_viz_thread.is_alive():
            return
        self.camera_viz_running = True
        self.camera_viz_thread = threading.Thread(
            target=self._camera_visualization_worker_loop,
            name="policy_probe_camera_viz",
            daemon=True,
        )
        self.camera_viz_thread.start()

    def _enqueue_camera_visualization(
        self,
        image: np.ndarray,
        observation: dict[str, Any],
        ir_image: np.ndarray | None = None,
        hybrid: bool = False,
    ) -> None:
        if cv2 is None or image is None or image.size == 0:
            return
        self._start_camera_visualization_worker()
        now = time.time()
        with self.camera_viz_slot_lock:
            if self.camera_viz_pending is not None:
                return
            min_period = max(0.0, self.camera_viz_min_period_s - 0.001)
            if now - self.camera_viz_last_enqueue_wall < min_period:
                return
        frame_copy = image.copy()
        ir_copy = None if ir_image is None else ir_image.copy()
        obs_copy = dict(observation)
        with self.camera_viz_slot_lock:
            if self.camera_viz_pending is not None:
                return
            self.camera_viz_pending = (frame_copy, ir_copy, obs_copy, bool(hybrid))
            self.camera_viz_last_enqueue_wall = now
        self.camera_viz_event.set()

    def _camera_visualization_worker_loop(self) -> None:
        while self.camera_viz_running:
            if not self.camera_viz_event.wait(timeout=0.5):
                continue
            self.camera_viz_event.clear()
            with self.camera_viz_slot_lock:
                pending = self.camera_viz_pending
                self.camera_viz_pending = None
            if pending is None:
                continue
            image, ir_image, observation, hybrid = pending
            with self.camera_lock:
                options = self.camera_options
                detector = self.camera_detector
            if not options.visualize:
                continue

            if hybrid and ir_image is not None and detector is not None:
                try:
                    viz_image = detector.detector._render_visualization(
                        image, ir_image, observation)
                    if viz_image is not None:
                        hybrid_obs_for_viz = dict(observation)
                        hybrid_obs_for_viz["_hybrid_viz_rendered"] = True
                        self._store_camera_visualization(
                            viz_image, hybrid_obs_for_viz)
                        continue
                except Exception as exc:  # noqa: BLE001 - fallback below.
                    with self.camera_lock:
                        self.camera_error = f"hybrid visualization failed: {exc}"

            self._store_camera_visualization(image, observation)

    def _store_camera_visualization(
        self,
        image: np.ndarray,
        observation: dict[str, Any],
    ) -> None:
        if cv2 is None:
            return
        try:
            overlay = self._render_visualization_frame(image, observation)
            max_width = int(self.cfg.camera_web_max_width)
            if max_width > 0 and overlay.shape[1] > max_width:
                scale = float(max_width) / float(overlay.shape[1])
                overlay = cv2.resize(
                    overlay,
                    (max_width, max(1, int(round(overlay.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            quality = int(min(95, max(25, self.cfg.camera_web_jpeg_quality)))
            ok, encoded = cv2.imencode(
                ".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return
            with self.camera_lock:
                self.latest_viz_jpeg = encoded.tobytes()
                self.latest_viz_seq += 1
                self.latest_viz_wall = time.time()
        except Exception as exc:  # noqa: BLE001 - viz must not affect control.
            with self.camera_lock:
                self.camera_error = f"visualization failed: {exc}"

    def latest_camera_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        now = time.time()
        with self.camera_lock:
            age = (
                None if self.latest_viz_wall is None
                else max(0.0, now - self.latest_viz_wall))
            info = {
                "seq": int(self.latest_viz_seq),
                "ageS": None if age is None else round(float(age), 6),
                "enabled": bool(self.camera_options.visualize),
            }
            frame = self.latest_viz_jpeg
        return frame, info

    def camera_ball_state(self) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        now = time.time()
        with self.camera_lock:
            pos = (
                None if self.latest_camera_ball_pos_m is None
                else self.latest_camera_ball_pos_m.copy())
            vel = (
                None if self.latest_camera_ball_vel_m_s is None
                else self.latest_camera_ball_vel_m_s.copy())
            detector_pos = (
                None if self.latest_camera_detector_pos_m is None
                else self.latest_camera_detector_pos_m.copy())
            detector_vel = (
                None if self.latest_camera_detector_vel_m_s is None
                else self.latest_camera_detector_vel_m_s.copy())
            age = (
                None if self.latest_camera_wall is None
                else max(0.0, now - self.latest_camera_wall))
            viz_age = (
                None if self.latest_viz_wall is None
                else max(0.0, now - self.latest_viz_wall))
            image_ages = {
                key: round(float(max(0.0, now - wall)), 6)
                for key, wall in self.camera_last_image_wall_by_transport.items()
            }
            feedback = {
                "enabled": bool(self.cfg.enable_camera_ball),
                "available": bool(_CAMERA_BALL_AVAILABLE),
                "ready": bool(self.camera_info_received and self.camera_detector is not None),
                "sourceMode": self.cfg.camera_source_mode,
                "useHybridDetector": bool(self.cfg.use_hybrid_detector),
                "useDepthSensor": bool(self.cfg.use_depth_sensor),
                "imageWidth": int(self.cfg.image_width),
                "imageHeight": int(self.cfg.image_height),
                "realsenseFps": int(self.cfg.realsense_fps),
                "cameraInfoReceived": bool(self.camera_info_received),
                "detectorReady": bool(self.camera_detector is not None),
                "detected": bool(self.latest_camera_valid),
                "ageS": None if age is None else round(float(age), 6),
                "frame": self.camera_output_frame,
                "detectorFrame": self.latest_camera_detector_frame,
                "controlFrame": "sim_frame",
                "detectorBall": (
                    None if detector_pos is None
                    else [round(float(v), 6) for v in detector_pos]),
                "detectorVelocity": (
                    None if detector_vel is None
                    else [round(float(v), 6) for v in detector_vel]),
                "controlBall": (
                    None if pos is None
                    else [round(float(v), 6) for v in pos]),
                "controlVelocity": (
                    None if vel is None
                    else [round(float(v), 6) for v in vel]),
                "imageTopic": self.cfg.camera_image_topic,
                "rawImageTopic": self.cfg.camera_raw_image_topic,
                "imageTransport": self.cfg.camera_image_transport,
                "activeTransport": self.camera_active_transport,
                "imageAgesS": image_ages,
                "cameraInfoTopic": self.cfg.camera_info_topic,
                "frameCount": int(self.camera_frame_count),
                "detectionCount": int(self.camera_detection_count),
                "callbackHz": (
                    round(float(1.0 / (sum(self.camera_callback_times) / len(self.camera_callback_times))), 3)
                    if len(self.camera_callback_times) > 0
                    and sum(self.camera_callback_times) > 0.0 else 0.0
                ),
                "visualizationRateHz": round(
                    float(self.cfg.camera_visualization_rate_hz), 3),
                "webMaxWidth": int(self.cfg.camera_web_max_width),
                "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
                "visualizationSeq": int(self.latest_viz_seq),
                "visualizationAgeS": None if viz_age is None else round(float(viz_age), 6),
                "runtime": self.camera_options.payload(),
            "error": self.camera_error,
            }
            valid = bool(self.latest_camera_valid and pos is not None and vel is not None)
        if not valid:
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                feedback,
            )
        return pos, vel, True, feedback

    def get_joint_feedback(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self.lock:
            q = None if self.latest_joint_q is None else self.latest_joint_q.copy()
            dq = None if self.latest_joint_dq is None else self.latest_joint_dq.copy()
        return q, dq

    def feedback_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            q = None if self.latest_joint_q is None else self.latest_joint_q.copy()
            dq = None if self.latest_joint_dq is None else self.latest_joint_dq.copy()
            q_age = (
                None if self.latest_joint_stamp_wall is None
                else max(0.0, now - self.latest_joint_stamp_wall)
            )
            cart = None if self.latest_cart_pos is None else self.latest_cart_pos.copy()
            quat = (
                None if self.latest_cart_quat_xyzw is None
                else self.latest_cart_quat_xyzw.copy()
            )
            cart_age = (
                None if self.latest_cart_stamp_wall is None
                else max(0.0, now - self.latest_cart_stamp_wall)
            )
            psi = self.latest_cart_psi
            publish_count = int(self.publish_count)
        return {
            "jointPosition": None if q is None else [round(float(v), 6) for v in q],
            "jointVelocity": None if dq is None else [round(float(v), 6) for v in dq],
            "jointAgeS": None if q_age is None else round(float(q_age), 6),
            "cartPosition": None if cart is None else [round(float(v), 6) for v in cart],
            "cartQuatXyzw": None if quat is None else [round(float(v), 6) for v in quat],
            "cartPsi": None if psi is None else round(float(psi), 6),
            "cartAgeS": None if cart_age is None else round(float(cart_age), 6),
            "publishCount": publish_count,
        }

    def publish_right_arm_command(self, q: np.ndarray) -> None:
        if self.cmd_pub is None:
            return
        msg = MechUnitCmdArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""
        cmd = MechUnitCmd()
        cmd.mu_idx = int(self.cfg.right_arm_mu_idx)
        cmd.jnt_pos = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)]
        cmd.use_jnt = True
        cmd.psi = 0.0
        msg.cmds = [cmd]
        self.cmd_pub.publish(msg)
        with self.lock:
            self.publish_count += 1

    def shutdown_camera_source(self) -> None:
        self.camera_direct_running = False
        thread = self.camera_direct_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self.camera_viz_running = False
        self.camera_viz_event.set()
        viz_thread = self.camera_viz_thread
        if viz_thread is not None and viz_thread.is_alive():
            viz_thread.join(timeout=1.0)
        with self.camera_lock:
            detector = self.camera_detector
            self.camera_direct_thread = None
            self.camera_viz_thread = None
        if detector is not None and hasattr(detector, "close"):
            try:
                detector.close()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Error closing camera detector: {exc}")


class RosBridge:
    def __init__(self, cfg: ProbeConfig):
        if not _ROS_AVAILABLE:
            raise RuntimeError(f"ROS2 Python bridge unavailable: {_ROS_IMPORT_ERROR}")
        if rclpy is None or MultiThreadedExecutor is None:
            raise RuntimeError("ROS2 Python bridge unavailable")
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = RosBridgeNode(cfg)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(
            target=self._spin,
            name="policy_probe_ros_spin",
            daemon=True,
        )
        self.thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except Exception:
            pass

    def get_joint_feedback(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.node.get_joint_feedback()

    def feedback_snapshot(self) -> dict[str, Any]:
        return self.node.feedback_snapshot()

    def camera_ball_state(self) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        return self.node.camera_ball_state()

    def set_camera_runtime_options(self, options: CameraRuntimeOptions) -> None:
        self.node.set_camera_runtime_options(options)

    def set_policy_viz_state(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        valid: bool,
        racket_anchor_m: np.ndarray,
        racket_pos_m: np.ndarray,
    ) -> None:
        self.node.set_policy_viz_state(
            ball_pos_m, ball_vel_m_s, valid, racket_anchor_m, racket_pos_m)

    def latest_camera_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        return self.node.latest_camera_visualization()

    def publish_right_arm_command(self, q: np.ndarray) -> None:
        self.node.publish_right_arm_command(q)

    def shutdown(self) -> None:
        try:
            self.node.shutdown_camera_source()
            self.executor.shutdown()
            self.node.destroy_node()
            self.thread.join(timeout=1.0)
        except Exception:
            pass


class PolicySafetyProbe:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.ros_bridge: RosBridge | None = None
        self.ball_estimator: BallStateEstimator | None = None
        self.ball_estimator_key: tuple[bool, bool] | None = None
        self.safe_q = target_right_arm_q_rad().astype(np.float32)
        self.controller = MjxRLPolicyController(
            str(cfg.model_path),
            dt=cfg.control_dt,
            robot_xml_path=str(cfg.robot_xml_path),
            enable_delay_conditioning=(cfg.policy_obs_dim == 67),
        )
        self.done_monitor = DoneConditionMonitor(
            ball_z_floor=cfg.done_z_min_m,
            ball_z_ceiling=cfg.done_z_max_m,
            ball_x_range=cfg.done_x_range_m,
            ball_y_range=cfg.done_y_range_m,
            racket_anchor_distance_limit=cfg.done_racket_anchor_distance_m,
            racket_z_high_offset=cfg.done_racket_z_high_offset_m,
            racket_z_low_offset=cfg.done_racket_z_low_offset_m,
            ee_x_limits=cfg.ee_x_limits_m,
            ee_y_limits=cfg.ee_y_limits_m,
            ee_z_limits=cfg.ee_z_limits_m,
        )
        if cfg.ros_bridge or cfg.enable_camera_ball:
            self.ros_bridge = RosBridge(cfg)
        self.reset()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.controller.reset()
            self.done_monitor.reset_counters()
            self.now_s = 0.0
            self.has_captured = False
            self.has_released = False
            self.release_candidate_ticks = 0
            self.release_samples: list[tuple[float, float, float]] = []
            self.last_valid_time_s: float | None = None
            self.failsafe_active = False
            self.done_failsafe_active = False
            self.last_ball_pos = np.asarray(self.initial_ball_pos(), dtype=np.float32)
            self.last_ball_vel = np.zeros(3, dtype=np.float32)
            self.last_observed_ball_pos = self.last_ball_pos.copy()
            self.last_observed_ball_vel = np.zeros(3, dtype=np.float32)
            self.last_measured_ball_pos = self.last_ball_pos.copy()
            self.last_measured_ball_vel = np.zeros(3, dtype=np.float32)
            self.ball_estimator = None
            self.ball_estimator_key = None
            self.next_ball_observation_s = 0.0
            self.last_estimator_info = self._estimator_off_payload()
            self.last_release_gate_info = self._release_gate_payload(
                valid=False,
                age_s=None,
                vz_m_s=None,
                z_drop_m=0.0,
                candidate=False,
            )
            self._sync_policy_to_safe_pose()
            return self.snapshot(
                "reset", "", {}, self.last_ball_pos,
                self.last_observed_ball_pos, self.last_observed_ball_vel,
                False, 0.0, 0.0,
                measured_ball_pos=self.last_measured_ball_pos,
                measured_ball_vel=self.last_measured_ball_vel,
                estimator_info=self.last_estimator_info,
                ball_source="drag",
                camera_feedback=self.camera_feedback())

    def initial_ball_pos(self) -> tuple[float, float, float]:
        anchor = self.anchor()
        return (
            float(anchor[0]),
            float(anchor[1]),
            min(self.cfg.done_z_max_m - 0.12, max(self.cfg.done_z_min_m + 0.12, float(anchor[2]) + 0.30)),
        )

    def anchor(self) -> np.ndarray:
        anchor = getattr(self.controller, "_racket_anchor_pos_base", None)
        if anchor is None or not np.all(np.isfinite(anchor)):
            return np.zeros(3, dtype=np.float32)
        return np.asarray(anchor, dtype=np.float32).reshape(3)

    def _sync_policy_to_safe_pose(self) -> None:
        self.controller.arm_cmd_q = self.safe_q.copy()
        self.controller.arm_cmd_qvel = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.controller.last_cmd_q = self.safe_q.copy()
        self.controller.last_cmd_qvel = self.controller.arm_cmd_qvel.copy()
        self.controller.last_cmd_qacc = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.controller.last_action = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)

    def _racket_pos(self) -> tuple[np.ndarray, bool]:
        pos, ok = self.controller._compute_racket_pose_base(self.controller.arm_cmd_q)
        return np.asarray(pos, dtype=np.float32).reshape(3), bool(ok)

    def _ros_joint_racket_pos(self) -> tuple[np.ndarray | None, bool]:
        if self.ros_bridge is None:
            return None, False
        q_fb, _dq_fb = self.ros_bridge.get_joint_feedback()
        if q_fb is None:
            return None, False
        pos, ok = self.controller._compute_racket_pose_base(q_fb)
        if not ok:
            return None, False
        return np.asarray(pos, dtype=np.float32).reshape(3), True

    def _estimator_off_payload(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "prediction": False,
            "aeroDrag": False,
            "valid": False,
            "state": "off",
            "predicted": False,
            "ageS": None,
            "lossS": None,
            "extrapolationS": None,
            "clamped": False,
            "updatesAccepted": 0,
            "updatesRejected": 0,
            "reinits": 0,
            "bouncesDetected": 0,
            "lastEvent": None,
        }

    def _make_ball_estimator(
        self,
        enable_prediction: bool,
        enable_aero_drag: bool,
    ) -> BallStateEstimator:
        cfg = BallStateEstimatorConfig(
            gravity_mm_s2=gravity_for_base_link(),
            max_extrapolation_s=self.cfg.ball_estimator_max_extrapolation_s,
            trust_loss_s=self.cfg.ball_estimator_trust_loss_s,
            degrade_loss_s=self.cfg.ball_estimator_degrade_loss_s,
            invalid_after_s=self.cfg.ball_estimator_invalid_after_s,
            prediction_horizon_s=self.cfg.ball_estimator_prediction_horizon_s,
            enable_prediction=bool(enable_prediction),
            enable_aero_drag=bool(enable_aero_drag),
        )
        return BallStateEstimator(cfg)

    def _ensure_ball_estimator(self, options: RuntimeOptions) -> BallStateEstimator:
        key = (bool(options.estimator_prediction), bool(options.estimator_aero_drag))
        if self.ball_estimator is None or self.ball_estimator_key != key:
            self.ball_estimator = self._make_ball_estimator(*key)
            self.ball_estimator_key = key
            self.next_ball_observation_s = self.now_s
        return self.ball_estimator

    def _estimator_payload(
        self,
        options: RuntimeOptions,
        out: dict[str, Any],
    ) -> dict[str, Any]:
        diag = (
            self.ball_estimator.diagnostics()
            if self.ball_estimator is not None else {}
        )
        return {
            "enabled": True,
            "prediction": bool(options.estimator_prediction),
            "aeroDrag": bool(options.estimator_aero_drag),
            "valid": bool(out.get("valid", False)),
            "state": str(out.get("state", "")),
            "predicted": bool(out.get("predicted", False)),
            "ageS": (
                None if out.get("age_s") is None
                else round(float(out.get("age_s", 0.0)), 6)
            ),
            "lossS": (
                None if out.get("loss_s") is None
                else round(float(out.get("loss_s", 0.0)), 6)
            ),
            "extrapolationS": (
                None if out.get("extrapolation_s") is None
                else round(float(out.get("extrapolation_s", 0.0)), 6)
            ),
            "clamped": bool(out.get("clamped", False)),
            "updatesAccepted": int(diag.get("updates_accepted", 0)),
            "updatesRejected": int(diag.get("updates_rejected", 0)),
            "reinits": int(diag.get("reinits", 0)),
            "bouncesDetected": int(diag.get("bounces_detected", 0)),
            "lastEvent": diag.get("last_event"),
        }

    def _resolve_control_ball_state(
        self,
        measured_pos: np.ndarray,
        measured_vel: np.ndarray,
        measured_valid: bool,
        options: RuntimeOptions,
    ) -> tuple[np.ndarray, np.ndarray, bool, float | None, dict[str, Any]]:
        if not options.use_ball_estimator:
            self.ball_estimator = None
            self.ball_estimator_key = None
            return (
                measured_pos.copy(), measured_vel.copy(), bool(measured_valid),
                None, self._estimator_off_payload())

        est = self._ensure_ball_estimator(options)
        t_control = self.now_s + self.cfg.control_dt
        obs_rate = max(float(options.ball_observation_rate_hz), 1e-6)
        obs_period = 1.0 / obs_rate

        if self.next_ball_observation_s <= 0.0:
            self.next_ball_observation_s = t_control
        if measured_valid and t_control + 1e-9 >= self.next_ball_observation_s:
            est.update((measured_pos * 1000.0).astype(np.float64), t_control)
            while self.next_ball_observation_s <= t_control + 1e-9:
                self.next_ball_observation_s += obs_period

        out = est.predict_to(t_control)
        info = self._estimator_payload(options, out)
        if not bool(out.get("valid", False)):
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                None,
                info,
            )
        pos = np.asarray(out.get("position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3) * 1e-3
        vel = np.asarray(out.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3) * 1e-3
        age_s = float(out.get("age_s", 0.0))
        if not math.isfinite(age_s):
            age_s = 0.0
        return pos, vel, True, max(0.0, age_s), info

    def _run_one_control_step(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid: bool,
        age_override_s: float | None,
        options: RuntimeOptions,
    ) -> tuple[str, str, dict[str, bool], float]:
        self.now_s += self.cfg.control_dt
        if valid:
            self.last_valid_time_s = self.now_s
            if not self.has_captured:
                self.has_captured = True
            self.failsafe_active = False

        if options.enable_ball_capture_gate and not self.has_captured:
            self._sync_policy_to_safe_pose()
            return "capture_gate", "", {}, self.cfg.ball_lost_timeout_s

        if options.enable_ball_release_gate and not self.has_released:
            release_age_s = (
                max(0.0, float(age_override_s))
                if age_override_s is not None
                else (0.0 if valid else self.cfg.ball_lost_timeout_s)
            )
            released = self._release_gate_observes_release(
                ball_pos, ball_vel, valid, age_s=release_age_s)
            if not released:
                self._sync_policy_to_safe_pose()
                return "release_gate", self._release_gate_reason(), {}, release_age_s
            self.has_released = True
            self._sync_policy_to_safe_pose()

        if age_override_s is not None:
            age_s = max(0.0, float(age_override_s))
        elif self.last_valid_time_s is None:
            age_s = self.cfg.ball_lost_timeout_s
        else:
            age_s = max(0.0, self.now_s - self.last_valid_time_s)

        if (options.enable_ball_lost_failsafe and not valid
                and self.last_valid_time_s is not None
                and age_s >= self.cfg.ball_lost_timeout_s):
            self.failsafe_active = True
            self._sync_policy_to_safe_pose()
            return "ball_lost_failsafe", "", {}, age_s

        arm_q = None
        arm_dq = None
        if options.use_ros_state and self.ros_bridge is not None:
            arm_q, arm_dq = self.ros_bridge.get_joint_feedback()
        self.controller.predict(
            ball_pos,
            ball_vel,
            ball_valid=valid,
            ball_obs_age_s=age_s,
            dt=self.cfg.control_dt,
            arm_q=arm_q,
            arm_dq=arm_dq,
        )
        racket_pos, fk_ok = self._racket_pos()
        if options.enable_done_monitoring and valid and fk_ok:
            is_done, reason, conditions = self.done_monitor.check_done(
                ball_pos_m=ball_pos,
                ball_vel_m_s=ball_vel,
                racket_pos_m=racket_pos,
                racket_anchor_m=self.anchor(),
                base_pos_m=None,
            )
            if is_done:
                self.done_failsafe_active = True
                self._sync_policy_to_safe_pose()
                return "done_failsafe", reason, conditions, age_s
        self.done_failsafe_active = False
        return "policy", "", {}, age_s

    def _release_gate_observes_release(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid: bool,
        age_s: float,
    ) -> bool:
        if not valid:
            self.release_samples.clear()
            self.release_candidate_ticks = 0
            self.last_release_gate_info = self._release_gate_payload(
                valid=False,
                age_s=age_s,
                vz_m_s=None,
                z_drop_m=0.0,
                candidate=False,
            )
            return False
        if age_s > self.cfg.ball_release_max_obs_age_s:
            self.release_samples.clear()
            self.release_candidate_ticks = 0
            vz = float(ball_vel[2]) if np.isfinite(float(ball_vel[2])) else None
            self.last_release_gate_info = self._release_gate_payload(
                valid=True,
                age_s=age_s,
                vz_m_s=vz,
                z_drop_m=0.0,
                candidate=False,
            )
            return False
        z = float(ball_pos[2])
        vz = float(ball_vel[2])
        self.release_samples.append((self.now_s, z, vz))
        while (self.release_samples
               and self.now_s - self.release_samples[0][0]
               > self.cfg.ball_release_window_s):
            self.release_samples.pop(0)
        max_z = max(sample[1] for sample in self.release_samples)
        z_drop = max(0.0, max_z - z)
        candidate = (
            vz <= -abs(self.cfg.ball_release_min_downward_velocity_m_s)
            and z_drop >= self.cfg.ball_release_min_z_drop_m)
        if candidate:
            self.release_candidate_ticks += 1
        else:
            self.release_candidate_ticks = 0
        self.last_release_gate_info = self._release_gate_payload(
            valid=True,
            age_s=age_s,
            vz_m_s=vz,
            z_drop_m=z_drop,
            candidate=candidate,
        )
        return (
            self.release_candidate_ticks
            >= self.cfg.ball_release_required_consecutive_ticks)

    def _release_gate_payload(
        self,
        valid: bool,
        age_s: float | None,
        vz_m_s: float | None,
        z_drop_m: float,
        candidate: bool,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(self.cfg.enable_ball_release_gate),
            "released": bool(getattr(self, "has_released", False)),
            "valid": bool(valid),
            "ageS": (
                None if age_s is None or not math.isfinite(float(age_s))
                else round(float(age_s), 6)),
            "fresh": (
                bool(age_s is not None and math.isfinite(float(age_s))
                     and float(age_s) <= self.cfg.ball_release_max_obs_age_s)),
            "vzMS": (
                None if vz_m_s is None or not math.isfinite(float(vz_m_s))
                else round(float(vz_m_s), 6)),
            "zDropM": round(float(max(0.0, z_drop_m)), 6),
            "candidate": bool(candidate),
            "candidateTicks": int(getattr(self, "release_candidate_ticks", 0)),
            "requiredTicks": int(self.cfg.ball_release_required_consecutive_ticks),
            "minDownwardVelocityMS": float(
                self.cfg.ball_release_min_downward_velocity_m_s),
            "minZDropM": float(self.cfg.ball_release_min_z_drop_m),
            "maxObsAgeS": float(self.cfg.ball_release_max_obs_age_s),
        }

    def _release_gate_reason(self) -> str:
        info = getattr(self, "last_release_gate_info", {})
        vz = info.get("vzMS")
        drop = info.get("zDropM")
        ticks = info.get("candidateTicks", 0)
        required = info.get(
            "requiredTicks", self.cfg.ball_release_required_consecutive_ticks)
        if not info.get("valid", False):
            return "waiting for valid ball before release gate clears"
        if not info.get("fresh", False):
            return "waiting for fresh ball observation before release gate clears"
        return (
            "waiting for downward release "
            f"(vz={vz if vz is not None else 'nan'} m/s, "
            f"z_drop={drop} m, ticks={ticks}/{required})")

    def _done_preview_payload(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        valid: bool,
        racket_pos_m: np.ndarray | None,
        fk_ok: bool,
    ) -> dict[str, Any]:
        conds = {
            "ball_too_low": False,
            "ball_too_high": False,
            "ball_x_out_of_bounds": False,
            "ball_y_out_of_bounds": False,
            "base_x_out_of_bounds": False,
            "base_y_out_of_bounds": False,
            "racket_too_far_from_anchor": False,
            "racket_too_high": False,
            "racket_too_low": False,
        }
        if self.cfg.ee_x_limits_m is not None:
            conds["ee_x_out_of_workspace"] = False
        if self.cfg.ee_y_limits_m is not None:
            conds["ee_y_out_of_workspace"] = False
        if self.cfg.ee_z_limits_m is not None:
            conds["ee_z_out_of_workspace"] = False

        if valid:
            ball = np.asarray(ball_pos_m, dtype=np.float64).reshape(3)
            anchor = np.asarray(self.anchor(), dtype=np.float64).reshape(3)
            conds["ball_too_low"] = bool(ball[2] < self.cfg.done_z_min_m)
            conds["ball_too_high"] = bool(ball[2] > self.cfg.done_z_max_m)
            conds["ball_x_out_of_bounds"] = bool(
                abs(ball[0] - anchor[0]) > self.cfg.done_x_range_m)
            conds["ball_y_out_of_bounds"] = bool(
                abs(ball[1] - anchor[1]) > self.cfg.done_y_range_m)
            if fk_ok and racket_pos_m is not None:
                racket = np.asarray(racket_pos_m, dtype=np.float64).reshape(3)
                dist = float(np.linalg.norm(racket - anchor))
                dz = float(racket[2] - anchor[2])
                conds["racket_too_far_from_anchor"] = bool(
                    dist > self.cfg.done_racket_anchor_distance_m)
                conds["racket_too_high"] = bool(
                    dz > self.cfg.done_racket_z_high_offset_m)
                conds["racket_too_low"] = bool(
                    dz < self.cfg.done_racket_z_low_offset_m)
                if self.cfg.ee_x_limits_m is not None:
                    lo, hi = self.cfg.ee_x_limits_m
                    conds["ee_x_out_of_workspace"] = not bool(lo <= racket[0] <= hi)
                if self.cfg.ee_y_limits_m is not None:
                    lo, hi = self.cfg.ee_y_limits_m
                    conds["ee_y_out_of_workspace"] = not bool(lo <= racket[1] <= hi)
                if self.cfg.ee_z_limits_m is not None:
                    lo, hi = self.cfg.ee_z_limits_m
                    conds["ee_z_out_of_workspace"] = not bool(lo <= racket[2] <= hi)

        active = [name for name, on in conds.items() if on]
        return {
            "done": bool(active),
            "reason": active[0] if active else "",
            "conditions": conds,
            "valid": bool(valid),
            "fkOk": bool(fk_ok),
            "ball": [round(float(x), 6) for x in np.asarray(
                ball_pos_m, dtype=np.float64).reshape(3)],
            "velocity": [round(float(x), 6) for x in np.asarray(
                ball_vel_m_s, dtype=np.float64).reshape(3)],
            "anchor": [round(float(x), 6) for x in self.anchor()],
        }

    def _apply_ball_noise(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid: bool,
        noise_std_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not valid or noise_std_m <= 0.0:
            return ball_pos.copy(), ball_vel.copy()
        pos_noise = np.random.normal(0.0, noise_std_m, 3).astype(np.float32)
        # Training-time observation noise commonly used ~3 mm position and
        # ~0.03 m/s velocity, so use a 10 s^-1 ratio for interactive probing.
        vel_noise = np.random.normal(0.0, noise_std_m * 10.0, 3).astype(np.float32)
        return (ball_pos + pos_noise).astype(np.float32), (ball_vel + vel_noise).astype(np.float32)

    def _runtime_options_from_payload(self, payload: dict[str, Any]) -> RuntimeOptions:
        use_ros_state = bool(payload.get("useRosState", self.cfg.ros_bridge))
        use_ros_state = bool(
            use_ros_state and self.cfg.ros_bridge and self.ros_bridge is not None)
        publish_commands = bool(payload.get("publishCommands", self.cfg.ros_publish_commands))
        publish_commands = bool(
            publish_commands
            and self.ros_bridge is not None
            and self.cfg.ros_publish_commands
        )
        if publish_commands:
            use_ros_state = True
        ball_source = str(payload.get("ballSource", "drag")).strip().lower()
        if ball_source not in ("drag", "camera"):
            ball_source = "drag"
        if ball_source == "camera" and not self.cfg.enable_camera_ball:
            ball_source = "drag"
        obs_rate = float(payload.get(
            "ballObservationRateHz", self.cfg.ball_observation_rate_hz))
        if not math.isfinite(obs_rate) or obs_rate <= 0.0:
            obs_rate = self.cfg.ball_observation_rate_hz
        obs_rate = min(max(obs_rate, 1.0), 240.0)
        camera_options = CameraRuntimeOptions(
            detect=bool(payload.get(
                "cameraDetect", self.cfg.enable_camera_detection)),
            visualize=bool(payload.get(
                "cameraVisualize", self.cfg.enable_camera_visualization)),
            use_depth_sensor=bool(payload.get(
                "cameraSensorDepth", self.cfg.use_depth_sensor)),
            use_undistort=bool(payload.get(
                "cameraUndistort", self.cfg.use_undistort)),
            use_kalman=bool(payload.get(
                "cameraKalman", self.cfg.use_kalman)),
            enable_ellipse_fit=bool(payload.get(
                "cameraEllipseFit", self.cfg.enable_ellipse_fit)),
            enable_temporal_tracker=bool(payload.get(
                "cameraTemporalTracker", self.cfg.enable_temporal_tracker)),
            enable_color_filter=bool(payload.get(
                "cameraColorFilter", self.cfg.enable_camera_color_filter)),
        )
        return RuntimeOptions(
            use_ros_state=use_ros_state,
            publish_commands=publish_commands,
            enable_ball_capture_gate=bool(payload.get(
                "enableCaptureGate", self.cfg.enable_ball_capture_gate)),
            enable_ball_release_gate=bool(payload.get(
                "enableReleaseGate", self.cfg.enable_ball_release_gate)),
            enable_ball_lost_failsafe=bool(payload.get(
                "enableLostFailsafe", self.cfg.enable_ball_lost_failsafe)),
            enable_done_monitoring=bool(payload.get(
                "enableDoneMonitoring", self.cfg.enable_done_monitoring)),
            use_ball_estimator=bool(payload.get(
                "useBallEstimator", self.cfg.enable_ball_state_estimator)),
            estimator_prediction=bool(payload.get(
                "estimatorPrediction", self.cfg.ball_estimator_enable_prediction)),
            estimator_aero_drag=bool(payload.get(
                "estimatorAeroDrag", self.cfg.ball_estimator_enable_aero_drag)),
            ball_observation_rate_hz=obs_rate,
            ball_source=ball_source,
            camera=camera_options,
        )

    def camera_feedback(self) -> dict[str, Any]:
        if self.ros_bridge is not None and self.cfg.enable_camera_ball:
            try:
                _pos, _vel, _valid, feedback = self.ros_bridge.camera_ball_state()
                return feedback
            except Exception as exc:
                return {
                    "enabled": bool(self.cfg.enable_camera_ball),
                    "available": bool(_CAMERA_BALL_AVAILABLE),
                    "ready": False,
                    "sourceMode": self.cfg.camera_source_mode,
                    "useHybridDetector": bool(self.cfg.use_hybrid_detector),
                    "useDepthSensor": bool(self.cfg.use_depth_sensor),
                    "imageWidth": int(self.cfg.image_width),
                    "imageHeight": int(self.cfg.image_height),
                    "realsenseFps": int(self.cfg.realsense_fps),
                    "cameraInfoReceived": False,
                    "detectorReady": False,
                    "detected": False,
                    "ageS": None,
                    "frame": None,
                    "detectorFrame": None,
                    "controlFrame": "sim_frame",
                    "detectorBall": None,
                    "detectorVelocity": None,
                    "controlBall": None,
                    "controlVelocity": None,
                    "imageTopic": self.cfg.camera_image_topic,
                    "rawImageTopic": self.cfg.camera_raw_image_topic,
                    "imageTransport": self.cfg.camera_image_transport,
                    "activeTransport": None,
                    "imageAgesS": {},
                    "cameraInfoTopic": self.cfg.camera_info_topic,
                    "frameCount": 0,
                    "detectionCount": 0,
                    "callbackHz": 0.0,
                    "visualizationRateHz": round(
                        float(self.cfg.camera_visualization_rate_hz), 3),
                    "webMaxWidth": int(self.cfg.camera_web_max_width),
                    "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
                    "visualizationSeq": 0,
                    "visualizationAgeS": None,
                    "runtime": {
                        "detect": bool(self.cfg.enable_camera_detection),
                        "visualize": bool(self.cfg.enable_camera_visualization),
                        "sensorDepth": bool(self.cfg.use_depth_sensor),
                        "undistort": bool(self.cfg.use_undistort),
                        "kalman": bool(self.cfg.use_kalman),
                        "ellipseFit": bool(self.cfg.enable_ellipse_fit),
                        "temporalTracker": bool(self.cfg.enable_temporal_tracker),
                        "colorFilter": bool(self.cfg.enable_camera_color_filter),
                    },
                    "error": str(exc),
                }
        return {
            "enabled": bool(self.cfg.enable_camera_ball),
            "available": bool(_CAMERA_BALL_AVAILABLE),
            "ready": False,
            "sourceMode": self.cfg.camera_source_mode,
            "useHybridDetector": bool(self.cfg.use_hybrid_detector),
            "useDepthSensor": bool(self.cfg.use_depth_sensor),
            "imageWidth": int(self.cfg.image_width),
            "imageHeight": int(self.cfg.image_height),
            "realsenseFps": int(self.cfg.realsense_fps),
            "cameraInfoReceived": False,
            "detectorReady": False,
            "detected": False,
            "ageS": None,
            "frame": None,
            "detectorFrame": None,
            "controlFrame": "sim_frame",
            "detectorBall": None,
            "detectorVelocity": None,
            "controlBall": None,
            "controlVelocity": None,
            "imageTopic": self.cfg.camera_image_topic,
            "rawImageTopic": self.cfg.camera_raw_image_topic,
            "imageTransport": self.cfg.camera_image_transport,
            "activeTransport": None,
            "imageAgesS": {},
            "cameraInfoTopic": self.cfg.camera_info_topic,
            "frameCount": 0,
            "detectionCount": 0,
            "callbackHz": 0.0,
            "visualizationRateHz": round(
                float(self.cfg.camera_visualization_rate_hz), 3),
            "webMaxWidth": int(self.cfg.camera_web_max_width),
            "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
            "visualizationSeq": 0,
            "visualizationAgeS": None,
            "runtime": {
                "detect": bool(self.cfg.enable_camera_detection),
                "visualize": bool(self.cfg.enable_camera_visualization),
                "sensorDepth": bool(self.cfg.use_depth_sensor),
                "undistort": bool(self.cfg.use_undistort),
                "kalman": bool(self.cfg.use_kalman),
                "ellipseFit": bool(self.cfg.enable_ellipse_fit),
                "temporalTracker": bool(self.cfg.enable_temporal_tracker),
                "colorFilter": bool(self.cfg.enable_camera_color_filter),
            },
            "error": None if _CAMERA_BALL_AVAILABLE else str(_CAMERA_BALL_IMPORT_ERROR),
        }

    def latest_camera_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        if self.ros_bridge is not None and self.cfg.enable_camera_ball:
            return self.ros_bridge.latest_camera_visualization()
        return None, {"seq": 0, "ageS": None, "enabled": False}

    def _select_ball_source(
        self,
        drag_pos: np.ndarray,
        drag_vel: np.ndarray,
        drag_valid: bool,
        options: RuntimeOptions,
    ) -> tuple[np.ndarray, np.ndarray, bool, str, dict[str, Any]]:
        camera_feedback = self.camera_feedback()
        if options.ball_source != "camera":
            return drag_pos, drag_vel, bool(drag_valid), "drag", camera_feedback
        if self.ros_bridge is None:
            camera_feedback["error"] = (
                camera_feedback.get("error") or "ROS node is not active")
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                "camera",
                camera_feedback,
            )
        cam_pos, cam_vel, cam_valid, camera_feedback = self.ros_bridge.camera_ball_state()
        if not cam_valid:
            cam_pos = self.last_ball_pos.copy()
            cam_vel = self.last_ball_vel.copy()
        if not drag_valid:
            cam_valid = False
        return cam_pos, cam_vel, cam_valid, "camera", camera_feedback

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("reset"):
            return self.reset()
        drag_ball_pos = vec3_from_payload(payload, "ball", self.initial_ball_pos())
        drag_ball_vel = vec3_from_payload(payload, "velocity", (0.0, 0.0, 0.0))
        drag_valid = bool(payload.get("valid", True))
        noise_std_m = float(payload.get("noiseStdM", self.cfg.ball_noise_std_m))
        if not math.isfinite(noise_std_m) or noise_std_m < 0.0:
            noise_std_m = self.cfg.ball_noise_std_m
        noise_std_m = min(max(noise_std_m, 0.0), 0.25)
        options = self._runtime_options_from_payload(payload)
        dt = float(payload.get("dt", self.cfg.control_dt))
        if not math.isfinite(dt) or dt <= 0.0:
            dt = self.cfg.control_dt
        dt = min(dt, 0.25)
        if self.ros_bridge is not None and self.cfg.enable_camera_ball:
            self.ros_bridge.set_camera_runtime_options(options.camera)

        with self.lock:
            ball_pos, ball_vel, valid, ball_source, camera_feedback = (
                self._select_ball_source(
                    drag_ball_pos, drag_ball_vel, drag_valid, options))
            start_pos = self.last_ball_pos.copy()
            start_vel = self.last_ball_vel.copy()
            n_steps = max(1, min(100, int(round(dt / self.cfg.control_dt))))
            mode = "policy"
            reason = ""
            conditions: dict[str, bool] = {}
            age_s = 0.0
            control_valid = bool(valid)
            obs_p = self.last_observed_ball_pos.copy()
            obs_v = self.last_observed_ball_vel.copy()
            measured_p = self.last_measured_ball_pos.copy()
            measured_v = self.last_measured_ball_vel.copy()
            estimator_info = self.last_estimator_info
            for i in range(n_steps):
                alpha = float(i + 1) / float(n_steps)
                p = start_pos * (1.0 - alpha) + ball_pos * alpha
                v = start_vel * (1.0 - alpha) + ball_vel * alpha
                measured_p, measured_v = self._apply_ball_noise(p, v, valid, noise_std_m)
                obs_p, obs_v, control_valid, age_override_s, estimator_info = (
                    self._resolve_control_ball_state(
                        measured_p, measured_v, valid, options))
                mode, reason, conditions, age_s = self._run_one_control_step(
                    obs_p, obs_v, control_valid, age_override_s, options)
            if self.ros_bridge is not None and self.cfg.enable_camera_ball:
                racket_pos, _fk_ok = self._racket_pos()
                if options.publish_commands:
                    ros_racket_pos, ros_fk_ok = self._ros_joint_racket_pos()
                    if ros_fk_ok and ros_racket_pos is not None:
                        racket_pos = ros_racket_pos
                self.ros_bridge.set_policy_viz_state(
                    obs_p, obs_v, control_valid, self.anchor(), racket_pos)
            self.last_ball_pos = ball_pos.copy()
            self.last_ball_vel = ball_vel.copy()
            self.last_observed_ball_pos = obs_p.copy()
            self.last_observed_ball_vel = obs_v.copy()
            self.last_measured_ball_pos = measured_p.copy()
            self.last_measured_ball_vel = measured_v.copy()
            self.last_estimator_info = estimator_info
            command_published = False
            if options.publish_commands:
                self.ros_bridge.publish_right_arm_command(
                    self.controller.arm_cmd_q.copy())
                command_published = True
            return self.snapshot(
                mode, reason, conditions, ball_pos, obs_p, obs_v,
                control_valid, age_s, noise_std_m,
                measured_ball_pos=measured_p,
                measured_ball_vel=measured_v,
                estimator_info=estimator_info,
                raw_valid=valid,
                use_ros_state=options.use_ros_state,
                publish_commands=options.publish_commands,
                command_published=command_published,
                runtime_options=options,
                ball_source=ball_source,
                camera_feedback=camera_feedback)

    def snapshot(
        self,
        mode: str,
        reason: str,
        conditions: dict[str, bool],
        raw_ball_pos: np.ndarray,
        observed_ball_pos: np.ndarray,
        observed_ball_vel: np.ndarray,
        valid: bool,
        age_s: float,
        noise_std_m: float,
        measured_ball_pos: np.ndarray | None = None,
        measured_ball_vel: np.ndarray | None = None,
        estimator_info: dict[str, Any] | None = None,
        raw_valid: bool | None = None,
        use_ros_state: bool | None = None,
        publish_commands: bool | None = None,
        command_published: bool = False,
        runtime_options: RuntimeOptions | None = None,
        ball_source: str = "drag",
        camera_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        racket_pos, fk_ok = self._racket_pos()
        cmd_racket_pos = racket_pos.copy()
        cmd_fk_ok = bool(fk_ok)
        racket_source = "command"
        cmd = self.controller.latest_command_state()
        obs_action = self.controller.latest_obs_and_action()
        measured_ball_pos = (
            observed_ball_pos if measured_ball_pos is None else measured_ball_pos)
        measured_ball_vel = (
            observed_ball_vel if measured_ball_vel is None else measured_ball_vel)
        estimator_info = (
            self._estimator_off_payload()
            if estimator_info is None else estimator_info)
        raw_valid = bool(valid) if raw_valid is None else bool(raw_valid)
        if use_ros_state is None:
            use_ros_state = bool(self.cfg.ros_bridge and self.ros_bridge is not None)
        if publish_commands is None:
            publish_commands = bool(
                self.cfg.ros_publish_commands and self.ros_bridge is not None)
        if runtime_options is None:
            runtime_payload = {
                "captureGate": bool(self.cfg.enable_ball_capture_gate),
                "releaseGate": bool(self.cfg.enable_ball_release_gate),
                "lostFailsafe": bool(self.cfg.enable_ball_lost_failsafe),
                "doneMonitoring": bool(self.cfg.enable_done_monitoring),
                "ballObservationRateHz": round(float(self.cfg.ball_observation_rate_hz), 6),
            }
        else:
            runtime_payload = {
                "captureGate": bool(runtime_options.enable_ball_capture_gate),
                "releaseGate": bool(runtime_options.enable_ball_release_gate),
                "lostFailsafe": bool(runtime_options.enable_ball_lost_failsafe),
                "doneMonitoring": bool(runtime_options.enable_done_monitoring),
                "ballObservationRateHz": round(float(runtime_options.ball_observation_rate_hz), 6),
            }
        release_gate_info = dict(getattr(self, "last_release_gate_info", {}))
        release_gate_info["enabled"] = bool(
            runtime_options.enable_ball_release_gate
            if runtime_options is not None else self.cfg.enable_ball_release_gate)
        feedback_racket_pos: np.ndarray | None = None
        feedback_fk_ok = False
        ros_feedback = {
            "enabled": bool(self.ros_bridge is not None),
            "useState": bool(use_ros_state),
            "useStateAllowed": bool(self.cfg.ros_bridge),
            "publishAllowed": bool(self.cfg.ros_publish_commands),
            "publishCommands": bool(publish_commands),
            "commandPublished": bool(command_published),
            "commandTopic": self.cfg.command_topic,
            "jointStateTopic": self.cfg.joint_state_topic,
            "cartStateTopic": self.cfg.cart_state_topic,
            "jointPosition": None,
            "jointVelocity": None,
            "jointAgeS": None,
            "cartPosition": None,
            "cartQuatXyzw": None,
            "cartPsi": None,
            "cartAgeS": None,
            "publishCount": 0,
            "jointEe": None,
            "jointEeFkOk": False,
            "jointCommandErrorAbsMax": None,
        }
        if self.ros_bridge is not None:
            ros_feedback.update(self.ros_bridge.feedback_snapshot())
            if ros_feedback.get("jointPosition") is not None:
                q_fb = np.asarray(ros_feedback["jointPosition"], dtype=np.float32)
                joint_ee, joint_fk_ok = self.controller._compute_racket_pose_base(q_fb)
                feedback_fk_ok = bool(joint_fk_ok)
                if feedback_fk_ok:
                    feedback_racket_pos = np.asarray(
                        joint_ee, dtype=np.float32).reshape(3)
                ros_feedback["jointEe"] = (
                    [round(float(x), 6) for x in joint_ee]
                    if joint_fk_ok else None
                )
                ros_feedback["jointEeFkOk"] = bool(joint_fk_ok)
                try:
                    err = np.asarray(cmd["position"], dtype=np.float32) - q_fb
                    ros_feedback["jointCommandErrorAbsMax"] = round(
                        float(np.max(np.abs(err))), 6)
                except Exception:
                    ros_feedback["jointCommandErrorAbsMax"] = None
        if publish_commands and feedback_fk_ok and feedback_racket_pos is not None:
            racket_pos = feedback_racket_pos
            fk_ok = True
            racket_source = "joint_state_fk"
        done_preview = self._done_preview_payload(
            observed_ball_pos, observed_ball_vel, bool(valid), racket_pos, fk_ok)
        camera_feedback = (
            self.camera_feedback()
            if camera_feedback is None else camera_feedback)
        return {
            "time": round(float(self.now_s), 6),
            "mode": mode,
            "reason": reason,
            "conditions": {k: bool(v) for k, v in conditions.items()},
            "ball": [round(float(x), 6) for x in observed_ball_pos],
            "rawBall": [round(float(x), 6) for x in raw_ball_pos],
            "measuredBall": [round(float(x), 6) for x in measured_ball_pos],
            "observedBall": [round(float(x), 6) for x in observed_ball_pos],
            "measuredVelocity": [round(float(x), 6) for x in measured_ball_vel],
            "velocity": [round(float(x), 6) for x in observed_ball_vel],
            "rawValid": raw_valid,
            "valid": bool(valid),
            "age": round(float(age_s), 6),
            "noiseStdM": round(float(noise_std_m), 6),
            "ballSource": str(ball_source),
            "runtime": runtime_payload,
            "estimator": estimator_info,
            "releaseGate": release_gate_info,
            "wouldDone": done_preview,
            "racket": [round(float(x), 6) for x in racket_pos],
            "racketSource": racket_source,
            "commandRacket": [round(float(x), 6) for x in cmd_racket_pos],
            "commandFkOk": cmd_fk_ok,
            "fkOk": fk_ok,
            "anchor": [round(float(x), 6) for x in self.anchor()],
            "jointPosition": [round(float(x), 6) for x in cmd["position"]],
            "jointVelocity": [round(float(x), 6) for x in cmd["velocity"]],
            "action": [round(float(x), 6) for x in obs_action["action"]],
            "actionAbsMax": round(float(np.max(np.abs(obs_action["action"]))), 6),
            "doneStats": self.done_monitor.diagnostics(),
            "ros": ros_feedback,
            "camera": camera_feedback,
        }

    def browser_config(self) -> dict[str, Any]:
        anchor = self.anchor()
        ball_box = box_payload(
            (float(anchor[0] - self.cfg.done_x_range_m), float(anchor[0] + self.cfg.done_x_range_m)),
            (float(anchor[1] - self.cfg.done_y_range_m), float(anchor[1] + self.cfg.done_y_range_m)),
            (self.cfg.done_z_min_m, self.cfg.done_z_max_m),
        )
        ee_box = box_payload(
            self.cfg.ee_x_limits_m,
            self.cfg.ee_y_limits_m,
            self.cfg.ee_z_limits_m,
        )
        xs = [ball_box["x0"], ball_box["x1"]]
        ys = [ball_box["y0"], ball_box["y1"]]
        zs = [ball_box["z0"], ball_box["z1"]]
        if ee_box:
            xs += [ee_box["x0"], ee_box["x1"]]
            ys += [ee_box["y0"], ee_box["y1"]]
            zs += [ee_box["z0"], ee_box["z1"]]
        margin = 0.18
        return {
            "controlRateHz": self.cfg.control_rate_hz,
            "browserStepRateHz": self.cfg.browser_step_rate_hz,
            "displayRateHz": self.cfg.display_rate_hz,
            "lostTimeoutS": self.cfg.ball_lost_timeout_s,
            "releaseGate": {
                "enabled": bool(self.cfg.enable_ball_release_gate),
                "minDownwardVelocityMS": self.cfg.ball_release_min_downward_velocity_m_s,
                "minZDropM": self.cfg.ball_release_min_z_drop_m,
                "windowS": self.cfg.ball_release_window_s,
                "requiredTicks": self.cfg.ball_release_required_consecutive_ticks,
                "maxObsAgeS": self.cfg.ball_release_max_obs_age_s,
            },
            "defaultNoiseStdM": self.cfg.ball_noise_std_m,
            "defaultBallObservationRateHz": self.cfg.ball_observation_rate_hz,
            "flowDefaults": {
                "captureGate": bool(self.cfg.enable_ball_capture_gate),
                "releaseGate": bool(self.cfg.enable_ball_release_gate),
                "lostFailsafe": bool(self.cfg.enable_ball_lost_failsafe),
                "doneMonitoring": bool(self.cfg.enable_done_monitoring),
            },
            "ballEstimator": {
                "enabled": bool(self.cfg.enable_ball_state_estimator),
                "prediction": bool(self.cfg.ball_estimator_enable_prediction),
                "aeroDrag": bool(self.cfg.ball_estimator_enable_aero_drag),
                "maxExtrapolationS": self.cfg.ball_estimator_max_extrapolation_s,
                "trustLossS": self.cfg.ball_estimator_trust_loss_s,
                "degradeLossS": self.cfg.ball_estimator_degrade_loss_s,
                "invalidAfterS": self.cfg.ball_estimator_invalid_after_s,
                "predictionHorizonS": self.cfg.ball_estimator_prediction_horizon_s,
            },
            "rosBridge": self.cfg.ros_bridge,
            "rosPublishCommands": self.cfg.ros_publish_commands,
            "rosDefaultUseState": self.cfg.ros_bridge,
            "rosDefaultPublishCommands": False,
            "rosPublishAllowed": self.cfg.ros_publish_commands,
            "commandTopic": self.cfg.command_topic,
            "jointStateTopic": self.cfg.joint_state_topic,
            "cartStateTopic": self.cfg.cart_state_topic,
            "cameraBall": {
                "enabled": bool(self.cfg.enable_camera_ball),
                "available": bool(_CAMERA_BALL_AVAILABLE),
                "sourceMode": self.cfg.camera_source_mode,
                "useHybridDetector": bool(self.cfg.use_hybrid_detector),
                "useDepthSensor": bool(self.cfg.use_depth_sensor),
                "imageWidth": int(self.cfg.image_width),
                "imageHeight": int(self.cfg.image_height),
                "realsenseFps": int(self.cfg.realsense_fps),
                "visualizationRateHz": round(
                    float(self.cfg.camera_visualization_rate_hz), 3),
                "webMaxWidth": int(self.cfg.camera_web_max_width),
                "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
                "imageTopic": self.cfg.camera_image_topic,
                "rawImageTopic": self.cfg.camera_raw_image_topic,
                "imageTransport": self.cfg.camera_image_transport,
                "cameraInfoTopic": self.cfg.camera_info_topic,
                "calibrationJsonPath": self.cfg.calibration_json_path,
                "yoloModelPath": str(self.cfg.yolo_model_path),
                "defaults": {
                    "detect": bool(self.cfg.enable_camera_detection),
                    "visualize": bool(self.cfg.enable_camera_visualization),
                    "sensorDepth": bool(self.cfg.use_depth_sensor),
                    "undistort": bool(self.cfg.use_undistort),
                    "kalman": bool(self.cfg.use_kalman),
                    "ellipseFit": bool(self.cfg.enable_ellipse_fit),
                    "temporalTracker": bool(self.cfg.enable_temporal_tracker),
                    "colorFilter": bool(self.cfg.enable_camera_color_filter),
                },
                "importError": (
                    None if _CAMERA_BALL_AVAILABLE
                    else str(_CAMERA_BALL_IMPORT_ERROR)),
            },
            "anchor": [round(float(x), 6) for x in anchor],
            "initialBall": [round(float(x), 6) for x in self.initial_ball_pos()],
            "ballBox": ball_box,
            "eeBox": ee_box,
            "limits": {
                "x0": min(xs) - margin, "x1": max(xs) + margin,
                "y0": min(ys) - margin, "y1": max(ys) + margin,
                "z0": max(0.0, min(zs) - margin), "z1": max(zs) + margin,
            },
        }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy Safety Probe</title>
<style>
  :root {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #1f2933;
    background: #f5f7fa;
  }
  * { box-sizing: border-box; }
  html, body { width: 100%; height: 100%; }
  body { margin: 0; min-height: 100vh; overflow: hidden; background: #f5f7fa; }
  .app {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    width: 100vw;
    height: 100vh;
    min-height: 0;
    overflow: hidden;
  }
  header {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 16px;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #d7dee8;
    background: #ffffff;
  }
  h1 { margin: 0; font-size: 18px; font-weight: 650; }
  .status {
    display: grid;
    grid-auto-flow: column;
    gap: 10px;
    align-items: center;
    font-size: 13px;
    color: #44546a;
    font-variant-numeric: tabular-nums;
  }
  .badge {
    min-width: 132px;
    border: 1px solid #c6d0dd;
    border-radius: 6px;
    padding: 6px 9px;
    background: #ffffff;
    color: #243244;
  }
  .badge.warn { border-color: #f59e0b; color: #92400e; background: #fff8eb; }
  .badge.stop { border-color: #ef4444; color: #991b1b; background: #fff1f2; }
  main {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 300px;
    gap: 0;
    min-height: 0;
    overflow: hidden;
  }
  .views {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.85fr);
    grid-template-rows: 1fr 1fr 0.78fr;
    min-height: 0;
    height: 100%;
  }
  canvas {
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
    background: linear-gradient(#ffffff, #edf2f7);
    border-right: 1px solid #d7dee8;
    border-bottom: 1px solid #d7dee8;
    touch-action: none;
    cursor: crosshair;
  }
  .view3d {
    grid-row: 1 / span 3;
    cursor: grab;
  }
  .view3d.dragging { cursor: grabbing; }
  .plane { min-height: 0; }
  .vision-panel {
    position: relative;
    min-height: 0;
    overflow: hidden;
    background: #0f172a;
    border-right: 1px solid #d7dee8;
    border-bottom: 1px solid #d7dee8;
  }
  .vision-panel img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #0f172a;
  }
  .vision-empty {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 12px;
    color: #cbd5e1;
    font-size: 13px;
    text-align: center;
  }
  aside {
    padding: 14px;
    border-left: 1px solid #d7dee8;
    background: #ffffff;
    overflow: auto;
    min-height: 0;
  }
  .row { display: grid; grid-template-columns: 96px 1fr; align-items: center; gap: 8px; margin-bottom: 10px; }
  label { font-size: 13px; color: #526173; }
  input[type="number"], select {
    width: 100%;
    border: 1px solid #bdc8d6;
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
  button {
    border: 1px solid #b7c2d0;
    border-radius: 6px;
    background: #ffffff;
    padding: 7px 10px;
    font-size: 14px;
    cursor: pointer;
  }
  button:hover { background: #f0f4f8; }
  .toggle { display: grid; grid-template-columns: 1fr auto; align-items: center; margin: 12px 0; }
  .section-title {
    margin: 16px 0 8px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
    font-size: 12px;
    font-weight: 650;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .checks {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px 10px;
    margin-bottom: 8px;
  }
  .checkrow {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 7px;
    align-items: center;
    min-height: 22px;
    font-size: 13px;
    color: #526173;
  }
  input:disabled + span,
  .checkrow.disabled span,
  .toggle.disabled label {
    color: #94a3b8;
  }
  .readout {
    margin-top: 14px;
    font-size: 13px;
    line-height: 1.7;
    color: #334155;
    font-variant-numeric: tabular-nums;
  }
  .plot {
    width: 100%;
    height: 150px;
    min-height: 150px;
    margin-top: 12px;
    border: 1px solid #d7dee8;
    border-radius: 6px;
    background: #ffffff;
    cursor: default;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 10px 16px;
    border-top: 1px solid #d7dee8;
    background: #ffffff;
    font-size: 13px;
    color: #44546a;
    overflow-x: auto;
  }
  .item::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 3px;
    margin-right: 6px;
    vertical-align: middle;
    background: var(--c);
  }
  @media (max-width: 900px) {
    body { overflow: auto; }
    .app { height: auto; min-height: 100vh; overflow: visible; }
    main { grid-template-columns: 1fr; overflow: visible; }
    aside { border-left: 0; border-top: 1px solid #d7dee8; }
    .views {
      grid-template-columns: 1fr;
      grid-template-rows: minmax(360px, 46vh) minmax(260px, 32vh) minmax(260px, 32vh) minmax(220px, 28vh);
      height: auto;
    }
    .view3d { grid-row: auto; }
    canvas { min-height: 260px; }
    header { grid-template-columns: 1fr; }
    .status { grid-auto-flow: row; }
  }
  @media (max-height: 760px) and (min-width: 901px) {
    header { padding: 8px 12px; }
    aside { padding: 10px; }
    .toggle { margin: 8px 0; }
    .row { margin-bottom: 7px; }
    .section-title { margin: 10px 0 6px; padding-top: 8px; }
    .checks { gap: 5px 8px; }
    .plot { height: 110px; min-height: 110px; }
    .legend { padding: 7px 12px; }
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>Policy Safety Probe</h1>
    <div class="status">
      <div id="mode" class="badge">mode</div>
      <div id="ros" class="badge">ros=off</div>
      <div id="time">t=0.000s</div>
      <div id="age">age=0.000s</div>
    </div>
  </header>
  <main>
    <div class="views">
      <canvas id="view3d" class="view3d"></canvas>
      <canvas id="top" class="plane"></canvas>
      <canvas id="side" class="plane"></canvas>
      <div class="vision-panel">
        <img id="visionFrame" alt="">
        <div id="visionEmpty" class="vision-empty">camera overlay waiting</div>
      </div>
    </div>
    <aside>
      <div class="toggle">
        <label for="visible">Ball visible</label>
        <input id="visible" type="checkbox" checked>
      </div>
      <div class="row">
        <label for="ballSource">ball source</label>
        <select id="ballSource">
          <option value="drag">drag ball</option>
          <option value="camera">camera ball</option>
        </select>
      </div>
      <div class="section-title">Runtime</div>
      <div class="toggle">
        <label for="useRosState">Use ROS state</label>
        <input id="useRosState" type="checkbox">
      </div>
      <div class="toggle">
        <label for="publishCommands">Send command</label>
        <input id="publishCommands" type="checkbox">
      </div>
      <div class="section-title">Node Flow</div>
      <div class="checks">
        <label class="checkrow"><input id="useBallEstimator" type="checkbox"><span>estimator</span></label>
        <label class="checkrow"><input id="estimatorPrediction" type="checkbox"><span>prediction</span></label>
        <label class="checkrow"><input id="estimatorAeroDrag" type="checkbox"><span>aero drag</span></label>
        <label class="checkrow"><input id="enableCaptureGate" type="checkbox"><span>capture gate</span></label>
        <label class="checkrow"><input id="enableReleaseGate" type="checkbox"><span>release gate</span></label>
        <label class="checkrow"><input id="enableLostFailsafe" type="checkbox"><span>lost failsafe</span></label>
        <label class="checkrow"><input id="enableDoneMonitoring" type="checkbox"><span>done monitor</span></label>
      </div>
      <div class="section-title">Camera Flow</div>
      <div class="checks">
        <label class="checkrow"><input id="cameraDetect" type="checkbox"><span>detection</span></label>
        <label class="checkrow"><input id="cameraVisualize" type="checkbox"><span>overlay</span></label>
        <label class="checkrow"><input id="cameraSensorDepth" type="checkbox"><span>sensor depth</span></label>
        <label class="checkrow"><input id="cameraUndistort" type="checkbox"><span>undistort</span></label>
        <label class="checkrow"><input id="cameraKalman" type="checkbox"><span>kalman</span></label>
        <label class="checkrow"><input id="cameraEllipseFit" type="checkbox"><span>ellipse fit</span></label>
        <label class="checkrow"><input id="cameraTemporalTracker" type="checkbox"><span>tracker</span></label>
        <label class="checkrow"><input id="cameraColorFilter" type="checkbox"><span>color filter</span></label>
      </div>
      <div class="row"><label for="x">ball x</label><input id="x" type="number" step="0.005"></div>
      <div class="row"><label for="y">ball y</label><input id="y" type="number" step="0.005"></div>
      <div class="row"><label for="z">ball z</label><input id="z" type="number" step="0.005"></div>
      <div class="row"><label for="noise">noise mm</label><input id="noise" type="number" min="0" step="0.5"></div>
      <div class="row"><label for="obsHz">ball obs hz</label><input id="obsHz" type="number" min="1" max="240" step="1"></div>
      <div class="section-title">Curves</div>
      <div class="checks">
        <label class="checkrow"><input id="showBall" type="checkbox" checked><span>ball</span></label>
        <label class="checkrow"><input id="showObserved" type="checkbox" checked><span>noisy obs</span></label>
        <label class="checkrow"><input id="showRacket" type="checkbox" checked><span>robot EE</span></label>
        <label class="checkrow"><input id="showJoint" type="checkbox" checked><span>joint FK</span></label>
        <label class="checkrow"><input id="showCart" type="checkbox" checked><span>cart</span></label>
        <label class="checkrow"><input id="showBounds" type="checkbox" checked><span>bounds</span></label>
        <label class="checkrow"><input id="showActionPlot" type="checkbox" checked><span>action plot</span></label>
        <label class="checkrow"><input id="showErrPlot" type="checkbox" checked><span>err plot</span></label>
      </div>
      <button id="reset">Reset</button>
      <div class="readout" id="readout"></div>
      <canvas id="plot" class="plot"></canvas>
    </aside>
  </main>
  <div class="legend">
    <span class="item" style="--c:#f28e2b">ball</span>
    <span class="item" style="--c:#e15759">noisy obs</span>
    <span class="item" style="--c:#4e79a7">robot EE</span>
    <span class="item" style="--c:#16a34a">joint_state FK</span>
    <span class="item" style="--c:#9333ea">cart_states</span>
    <span class="item" style="--c:#111827">ball done bounds</span>
    <span class="item" style="--c:#0891b2">EE workspace</span>
    <span class="item" style="--c:#6b7280">hidden ball</span>
    <span class="item" style="--c:#ef4444">failsafe</span>
  </div>
</div>
<script>
const CONFIG = __CONFIG_JSON__;
const STEP_RATE_HZ = Math.max(1, Math.min(120, Number(CONFIG.browserStepRateHz) || 60));
const DISPLAY_RATE_HZ = Math.max(1, Math.min(240, Number(CONFIG.displayRateHz) || 60));
const DISPLAY_INTERVAL_MS = 1000 / DISPLAY_RATE_HZ;
let ball = {x: CONFIG.initialBall[0], y: CONFIG.initialBall[1], z: CONFIG.initialBall[2]};
let observedBall = {...ball};
let lastSentBall = {...ball};
let lastSentAt = performance.now();
let racket = {x: CONFIG.anchor[0], y: CONFIG.anchor[1], z: CONFIG.anchor[2]};
let jointEe = null;
let cartEe = null;
let trailBall = [];
let trailObservedBall = [];
let trailRacket = [];
let trailJointEe = [];
let trailCartEe = [];
let plotHistory = [];
let latest = null;
let dragging = null;
let inFlight = false;
let viewYaw = -0.72;
let viewPitch = 0.46;
let viewZoom = 1.0;
let lastDragX = 0;
let lastDragY = 0;
let lastVisionSeq = 0;
let lastRenderAt = 0;

const view3dCanvas = document.getElementById('view3d');
const topCanvas = document.getElementById('top');
const sideCanvas = document.getElementById('side');
const visionFrameEl = document.getElementById('visionFrame');
const visionEmptyEl = document.getElementById('visionEmpty');
const visibleEl = document.getElementById('visible');
const ballSourceEl = document.getElementById('ballSource');
const useRosStateEl = document.getElementById('useRosState');
const publishCommandsEl = document.getElementById('publishCommands');
const modeEl = document.getElementById('mode');
const rosEl = document.getElementById('ros');
const timeEl = document.getElementById('time');
const ageEl = document.getElementById('age');
const readoutEl = document.getElementById('readout');
const plotCanvas = document.getElementById('plot');
const curveEls = {
  ball: document.getElementById('showBall'),
  observed: document.getElementById('showObserved'),
  racket: document.getElementById('showRacket'),
  joint: document.getElementById('showJoint'),
  cart: document.getElementById('showCart'),
  bounds: document.getElementById('showBounds'),
  actionPlot: document.getElementById('showActionPlot'),
  errPlot: document.getElementById('showErrPlot')
};
const flowEls = {
  useBallEstimator: document.getElementById('useBallEstimator'),
  estimatorPrediction: document.getElementById('estimatorPrediction'),
  estimatorAeroDrag: document.getElementById('estimatorAeroDrag'),
  enableCaptureGate: document.getElementById('enableCaptureGate'),
  enableReleaseGate: document.getElementById('enableReleaseGate'),
  enableLostFailsafe: document.getElementById('enableLostFailsafe'),
  enableDoneMonitoring: document.getElementById('enableDoneMonitoring')
};
const cameraFlowEls = {
  detect: document.getElementById('cameraDetect'),
  visualize: document.getElementById('cameraVisualize'),
  sensorDepth: document.getElementById('cameraSensorDepth'),
  undistort: document.getElementById('cameraUndistort'),
  kalman: document.getElementById('cameraKalman'),
  ellipseFit: document.getElementById('cameraEllipseFit'),
  temporalTracker: document.getElementById('cameraTemporalTracker'),
  colorFilter: document.getElementById('cameraColorFilter')
};
const inputs = {
  x: document.getElementById('x'),
  y: document.getElementById('y'),
  z: document.getElementById('z'),
  noise: document.getElementById('noise'),
  obsHz: document.getElementById('obsHz')
};

function fmt(v, n=3) { return Number.isFinite(v) ? v.toFixed(n) : 'nan'; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function curveOn(name) { return curveEls[name] ? curveEls[name].checked : true; }
function metersPerSecond(a, b, dt) {
  return [(a.x - b.x) / dt, (a.y - b.y) / dt, (a.z - b.z) / dt].map(v => Number.isFinite(v) ? v : 0);
}

function setRuntimeDefaults() {
  useRosStateEl.checked = !!CONFIG.rosDefaultUseState && !!CONFIG.rosBridge;
  useRosStateEl.disabled = !CONFIG.rosBridge;
  publishCommandsEl.checked = !!CONFIG.rosDefaultPublishCommands && !!CONFIG.rosBridge && !!CONFIG.rosPublishAllowed;
  publishCommandsEl.disabled = !CONFIG.rosBridge || !CONFIG.rosPublishAllowed;
  const cameraEnabled = !!(CONFIG.cameraBall && CONFIG.cameraBall.enabled);
  const cameraAvailable = !!(CONFIG.cameraBall && CONFIG.cameraBall.available);
  for (const opt of Array.from(ballSourceEl.options)) {
    if (opt.value === 'camera') opt.disabled = !(cameraEnabled && cameraAvailable);
  }
  if (cameraEnabled && cameraAvailable) {
    ballSourceEl.value = 'camera';
  }
  if (!(cameraEnabled && cameraAvailable) && ballSourceEl.value === 'camera') {
    ballSourceEl.value = 'drag';
  }
  const cameraDefaults = (CONFIG.cameraBall && CONFIG.cameraBall.defaults) || {};
  cameraFlowEls.detect.checked = cameraDefaults.detect !== false;
  cameraFlowEls.visualize.checked = cameraDefaults.visualize !== false;
  cameraFlowEls.sensorDepth.checked = cameraDefaults.sensorDepth !== false;
  cameraFlowEls.undistort.checked = cameraDefaults.undistort !== false;
  cameraFlowEls.kalman.checked = !!cameraDefaults.kalman;
  cameraFlowEls.ellipseFit.checked = cameraDefaults.ellipseFit !== false;
  cameraFlowEls.temporalTracker.checked = !!cameraDefaults.temporalTracker;
  cameraFlowEls.colorFilter.checked = cameraDefaults.colorFilter !== false;
  for (const el of Object.values(cameraFlowEls)) {
    el.disabled = !(cameraEnabled && cameraAvailable);
    el.closest('.checkrow').classList.toggle('disabled', el.disabled);
  }
  useRosStateEl.parentElement.classList.toggle('disabled', useRosStateEl.disabled);
  publishCommandsEl.parentElement.classList.toggle('disabled', publishCommandsEl.disabled);
  flowEls.useBallEstimator.checked = !!(CONFIG.ballEstimator && CONFIG.ballEstimator.enabled);
  flowEls.estimatorPrediction.checked = !!(CONFIG.ballEstimator && CONFIG.ballEstimator.prediction);
  flowEls.estimatorAeroDrag.checked = !!(CONFIG.ballEstimator && CONFIG.ballEstimator.aeroDrag);
  flowEls.enableCaptureGate.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.captureGate);
  flowEls.enableReleaseGate.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.releaseGate);
  flowEls.enableLostFailsafe.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.lostFailsafe);
  flowEls.enableDoneMonitoring.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.doneMonitoring);
}

function setInputsFromBall() {
  inputs.x.value = fmt(ball.x, 4);
  inputs.y.value = fmt(ball.y, 4);
  inputs.z.value = fmt(ball.z, 4);
  if (!inputs.noise.value) inputs.noise.value = fmt((CONFIG.defaultNoiseStdM || 0) * 1000, 1);
  if (!inputs.obsHz.value) inputs.obsHz.value = fmt(CONFIG.defaultBallObservationRateHz || 60, 0);
}

function setBallFromInputs() {
  ball.x = Number(inputs.x.value);
  ball.y = Number(inputs.y.value);
  ball.z = Number(inputs.z.value);
}

for (const el of [inputs.x, inputs.y, inputs.z]) {
  el.addEventListener('change', () => { setBallFromInputs(); drawAll(); });
}
inputs.noise.addEventListener('change', drawAll);
inputs.obsHz.addEventListener('change', drawAll);

function resizeCanvas(canvas) {
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function projectTop(p) {
  const rect = topCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: (p.x - l.x0) / (l.x1 - l.x0) * rect.width,
    y: rect.height - (p.y - l.y0) / (l.y1 - l.y0) * rect.height
  };
}

function unprojectTop(q) {
  const rect = topCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: l.x0 + q.x / rect.width * (l.x1 - l.x0),
    y: l.y0 + (1 - q.y / rect.height) * (l.y1 - l.y0)
  };
}

function projectSide(p) {
  const rect = sideCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: (p.x - l.x0) / (l.x1 - l.x0) * rect.width,
    y: rect.height - (p.z - l.z0) / (l.z1 - l.z0) * rect.height
  };
}

function unprojectSide(q) {
  const rect = sideCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: l.x0 + q.x / rect.width * (l.x1 - l.x0),
    z: l.z0 + (1 - q.y / rect.height) * (l.z1 - l.z0)
  };
}

function sceneCenter() {
  const l = CONFIG.limits;
  return {
    x: (l.x0 + l.x1) * 0.5,
    y: (l.y0 + l.y1) * 0.5,
    z: (l.z0 + l.z1) * 0.5
  };
}

function sceneRadius() {
  const l = CONFIG.limits;
  return Math.max(l.x1 - l.x0, l.y1 - l.y0, l.z1 - l.z0) * 0.58;
}

function rotate3D(p) {
  const c = sceneCenter();
  let x = p.x - c.x, y = p.y - c.y, z = p.z - c.z;
  const cy = Math.cos(viewYaw), sy = Math.sin(viewYaw);
  const cp = Math.cos(viewPitch), sp = Math.sin(viewPitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const y2 = cp * y1 - sp * z;
  const z2 = sp * y1 + cp * z;
  return {x:x1, y:y2, z:z2};
}

function project3D(p) {
  const rect = view3dCanvas.getBoundingClientRect();
  const r = rotate3D(p);
  const scale = Math.min(rect.width, rect.height) * 0.42 * viewZoom / Math.max(sceneRadius(), 1e-6);
  return {
    x: rect.width * 0.52 + r.x * scale,
    y: rect.height * 0.54 - r.y * scale,
    depth: r.z
  };
}

function drawRect(ctx, canvas, box, projection, color, alpha, label) {
  if (!box) return;
  const corners = projection === projectTop
    ? [{x:box.x0,y:box.y0,z:box.z0},{x:box.x1,y:box.y1,z:box.z1}]
    : [{x:box.x0,y:0,z:box.z0},{x:box.x1,y:0,z:box.z1}];
  const a = projection(corners[0]);
  const b = projection(corners[1]);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  ctx.fillText(label, Math.min(a.x, b.x) + 8, Math.min(a.y, b.y) + 18);
  ctx.restore();
}

function drawBox3D(ctx, box, color, alpha, label) {
  if (!box) return;
  const pts = [
    {x:box.x0,y:box.y0,z:box.z0},{x:box.x1,y:box.y0,z:box.z0},{x:box.x1,y:box.y1,z:box.z0},{x:box.x0,y:box.y1,z:box.z0},
    {x:box.x0,y:box.y0,z:box.z1},{x:box.x1,y:box.y0,z:box.z1},{x:box.x1,y:box.y1,z:box.z1},{x:box.x0,y:box.y1,z:box.z1}
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (const [a,b] of edges) {
    const pa = project3D(pts[a]), pb = project3D(pts[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  const q = project3D(pts[4]);
  ctx.fillText(label, q.x + 6, q.y - 6);
  ctx.restore();
}

function drawTrail(ctx, trail, projection, color, alpha) {
  if (trail.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < trail.length; i++) {
    const p = projection(trail[i]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawDot(ctx, p, projection, color, radius, stroke='#ffffff') {
  const q = projection(p);
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(q.x, q.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawPanel(canvas, title, projection, axesText) {
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (curveOn('bounds')) {
    drawRect(ctx, canvas, CONFIG.ballBox, projection, '#111827', 0.24, 'done');
    drawRect(ctx, canvas, CONFIG.eeBox, projection, '#0891b2', 0.48, 'EE');
    drawDot(ctx, {x:CONFIG.anchor[0], y:CONFIG.anchor[1], z:CONFIG.anchor[2]}, projection, '#111827', 5);
  }
  if (curveOn('ball')) drawTrail(ctx, trailBall, projection, visibleEl.checked ? '#f28e2b' : '#6b7280', 0.42);
  if (curveOn('observed')) drawTrail(ctx, trailObservedBall, projection, '#e15759', 0.32);
  if (curveOn('racket')) drawTrail(ctx, trailRacket, projection, '#4e79a7', 0.65);
  if (curveOn('joint')) drawTrail(ctx, trailJointEe, projection, '#16a34a', 0.70);
  if (curveOn('cart')) drawTrail(ctx, trailCartEe, projection, '#9333ea', 0.70);
  if (curveOn('ball')) drawDot(ctx, ball, projection, visibleEl.checked ? '#f28e2b' : '#6b7280', 8);
  if (latest && latest.valid && curveOn('observed')) drawDot(ctx, observedBall, projection, '#e15759', 5, '#ffffff');
  if (curveOn('racket')) drawDot(ctx, racket, projection, latest && latest.mode.includes('failsafe') ? '#ef4444' : '#4e79a7', 8);
  if (jointEe && curveOn('joint')) drawDot(ctx, jointEe, projection, '#16a34a', 6, '#ffffff');
  if (cartEe && curveOn('cart')) drawDot(ctx, cartEe, projection, '#9333ea', 6, '#ffffff');
  ctx.save();
  ctx.fillStyle = '#273445';
  ctx.font = '14px sans-serif';
  ctx.fillText(title, 14, 22);
  ctx.fillStyle = '#64748b';
  ctx.font = '12px sans-serif';
  ctx.fillText(axesText, 14, 40);
  ctx.restore();
}

function draw3D() {
  const canvas = view3dCanvas;
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (curveOn('bounds')) {
    drawBox3D(ctx, CONFIG.ballBox, '#111827', 0.24, 'done');
    drawBox3D(ctx, CONFIG.eeBox, '#0891b2', 0.52, 'EE');
    drawDot(ctx, {x:CONFIG.anchor[0], y:CONFIG.anchor[1], z:CONFIG.anchor[2]}, project3D, '#111827', 5);
  }
  if (curveOn('ball')) drawTrail(ctx, trailBall, project3D, visibleEl.checked ? '#f28e2b' : '#6b7280', 0.38);
  if (curveOn('observed')) drawTrail(ctx, trailObservedBall, project3D, '#e15759', 0.42);
  if (curveOn('racket')) drawTrail(ctx, trailRacket, project3D, '#4e79a7', 0.78);
  if (curveOn('joint')) drawTrail(ctx, trailJointEe, project3D, '#16a34a', 0.86);
  if (curveOn('cart')) drawTrail(ctx, trailCartEe, project3D, '#9333ea', 0.86);
  if (curveOn('ball')) drawDot(ctx, ball, project3D, visibleEl.checked ? '#f28e2b' : '#6b7280', 8);
  if (latest && latest.valid && curveOn('observed')) drawDot(ctx, observedBall, project3D, '#e15759', 6, '#ffffff');
  if (curveOn('racket')) drawDot(ctx, racket, project3D, latest && latest.mode.includes('failsafe') ? '#ef4444' : '#4e79a7', 8);
  if (jointEe && curveOn('joint')) drawDot(ctx, jointEe, project3D, '#16a34a', 6, '#ffffff');
  if (cartEe && curveOn('cart')) drawDot(ctx, cartEe, project3D, '#9333ea', 6, '#ffffff');
  ctx.save();
  ctx.fillStyle = '#273445';
  ctx.font = '14px sans-serif';
  ctx.fillText('3D view', 14, 22);
  ctx.fillStyle = '#64748b';
  ctx.font = '12px sans-serif';
  ctx.fillText('drag to rotate / wheel to zoom', 14, 40);
  ctx.restore();
}

function drawAll() {
  draw3D();
  drawPanel(topCanvas, 'Top view', projectTop, 'x / y');
  drawPanel(sideCanvas, 'Side view', projectSide, 'x / z');
  drawPlot();
}

function renderLoop(timestamp) {
  if (!lastRenderAt || timestamp - lastRenderAt >= DISPLAY_INTERVAL_MS - 1) {
    lastRenderAt = timestamp;
    drawAll();
  }
  requestAnimationFrame(renderLoop);
}

function updateVisionFrame(camera) {
  const seq = camera && Number.isFinite(camera.visualizationSeq)
    ? camera.visualizationSeq : 0;
  const vizOn = !!(camera && camera.runtime && camera.runtime.visualize);
  if (!vizOn) {
    visionEmptyEl.style.display = 'grid';
    visionEmptyEl.textContent = 'camera overlay off';
    return;
  }
  if (seq > 0 && seq !== lastVisionSeq) {
    lastVisionSeq = seq;
    visionFrameEl.src = `/camera_frame.jpg?seq=${seq}&t=${Date.now()}`;
    visionEmptyEl.style.display = 'none';
  } else if (seq <= 0) {
    const frames = camera && Number.isFinite(camera.frameCount) ? camera.frameCount : 0;
    const active = camera && camera.activeTransport ? camera.activeTransport : 'none';
    const info = camera && camera.cameraInfoReceived ? 'info' : 'no-info';
    const detector = camera && camera.detectorReady ? 'detector' : 'no-detector';
    const source = camera && camera.sourceMode ? camera.sourceMode : 'ros_topic';
    const camRt = (camera && camera.runtime) || {};
    const sensorDepthOn = camRt.sensorDepth !== false;
    const sourceText = source === 'direct_realsense'
      ? `direct RealSense ${camera.imageWidth || '?'}x${camera.imageHeight || '?'}@${camera.realsenseFps || '?'} hybrid=${camera.useHybridDetector ? 'on' : 'off'} sensorDepth=${sensorDepthOn ? 'on' : 'off'}`
      : (camera && camera.imageTransport === 'raw'
        ? `topic=${camera.rawImageTopic}`
        : (camera && camera.imageTransport === 'compressed'
          ? `topic=${camera.imageTopic}`
          : `topic=${camera ? camera.imageTopic : ''} | ${camera ? camera.rawImageTopic : ''}`));
    const err = camera && camera.error ? ` | error=${camera.error}` : '';
    visionEmptyEl.style.display = 'grid';
    visionEmptyEl.textContent = camera && camera.enabled
      ? `camera overlay waiting | frames=${frames} | ${info} | ${detector} | active=${active} | source=${sourceText || 'none'}${err}`
      : 'camera source off';
  }
}

function drawPlot() {
  const ctx = plotCanvas.getContext('2d');
  const rect = plotCanvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.save();
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = rect.height * i / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }
  ctx.fillStyle = '#334155';
  ctx.font = '12px sans-serif';
  const showAction = curveOn('actionPlot');
  const showErr = curveOn('errPlot');
  const title = showAction && showErr
    ? 'action max / cmd-joint err(rad)'
    : (showAction ? 'action max' : (showErr ? 'cmd-joint err(rad)' : 'plot hidden'));
  ctx.fillText(title, 8, 16);

  function drawSeries(key, color, scale) {
    if (plotHistory.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < plotHistory.length; i++) {
      const v = Math.min(1, Math.max(0, (plotHistory[i][key] || 0) / scale));
      const x = rect.width * i / Math.max(1, plotHistory.length - 1);
      const y = rect.height - 18 - v * (rect.height - 34);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  if (showAction) drawSeries('action', '#4e79a7', 1.0);
  if (showErr) drawSeries('err', '#16a34a', 0.35);
  if (showAction) {
    ctx.fillStyle = '#4e79a7';
    ctx.fillText('action', 8, rect.height - 8);
  }
  if (showErr) {
    ctx.fillStyle = '#16a34a';
    ctx.fillText('err', showAction ? 70 : 8, rect.height - 8);
  }
  ctx.restore();
}

function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {x: event.clientX - rect.left, y: event.clientY - rect.top};
}

function beginDrag(event, kind) {
  dragging = kind;
  event.currentTarget.setPointerCapture(event.pointerId);
  updateDrag(event);
}

function updateDrag(event) {
  if (!dragging) return;
  if (dragging === 'top') {
    const p = unprojectTop(canvasPoint(event, topCanvas));
    ball.x = clamp(p.x, CONFIG.limits.x0, CONFIG.limits.x1);
    ball.y = clamp(p.y, CONFIG.limits.y0, CONFIG.limits.y1);
  } else if (dragging === 'side') {
    const p = unprojectSide(canvasPoint(event, sideCanvas));
    ball.x = clamp(p.x, CONFIG.limits.x0, CONFIG.limits.x1);
    ball.z = clamp(p.z, CONFIG.limits.z0, CONFIG.limits.z1);
  } else if (dragging === '3d') {
    const dx = event.clientX - lastDragX;
    const dy = event.clientY - lastDragY;
    lastDragX = event.clientX;
    lastDragY = event.clientY;
    viewYaw += dx * 0.008;
    viewPitch = clamp(viewPitch + dy * 0.008, -1.35, 1.35);
  }
  setInputsFromBall();
  drawAll();
}

view3dCanvas.addEventListener('pointerdown', e => {
  dragging = '3d';
  lastDragX = e.clientX;
  lastDragY = e.clientY;
  view3dCanvas.classList.add('dragging');
  view3dCanvas.setPointerCapture(e.pointerId);
});
topCanvas.addEventListener('pointerdown', e => beginDrag(e, 'top'));
sideCanvas.addEventListener('pointerdown', e => beginDrag(e, 'side'));
for (const canvas of [view3dCanvas, topCanvas, sideCanvas]) {
  canvas.addEventListener('pointermove', updateDrag);
  canvas.addEventListener('pointerup', e => {
    dragging = null;
    view3dCanvas.classList.remove('dragging');
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
  });
  canvas.addEventListener('pointerleave', () => {
    dragging = null;
    view3dCanvas.classList.remove('dragging');
  });
}
view3dCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  viewZoom *= Math.exp(-e.deltaY * 0.001);
  viewZoom = clamp(viewZoom, 0.25, 8);
  drawAll();
}, {passive:false});

async function postJson(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function step() {
  if (inFlight) return;
  inFlight = true;
  const now = performance.now();
  const dt = Math.max(0.001, Math.min(0.25, (now - lastSentAt) / 1000));
  const velocity = metersPerSecond(ball, lastSentBall, dt);
  lastSentAt = now;
  lastSentBall = {...ball};
  try {
    latest = await postJson('/step', {
      ball: [ball.x, ball.y, ball.z],
      velocity,
      valid: visibleEl.checked,
      noiseStdM: Math.max(0, Number(inputs.noise.value) || 0) * 0.001,
      ballObservationRateHz: Math.max(1, Math.min(240, Number(inputs.obsHz.value) || 60)),
      ballSource: ballSourceEl.value,
      useRosState: (useRosStateEl.checked || (publishCommandsEl.checked && !publishCommandsEl.disabled)) && !useRosStateEl.disabled,
      publishCommands: publishCommandsEl.checked && !publishCommandsEl.disabled,
      useBallEstimator: flowEls.useBallEstimator.checked,
      estimatorPrediction: flowEls.estimatorPrediction.checked,
      estimatorAeroDrag: flowEls.estimatorAeroDrag.checked,
      enableCaptureGate: flowEls.enableCaptureGate.checked,
      enableReleaseGate: flowEls.enableReleaseGate.checked,
      enableLostFailsafe: flowEls.enableLostFailsafe.checked,
      enableDoneMonitoring: flowEls.enableDoneMonitoring.checked,
      cameraDetect: cameraFlowEls.detect.checked && !cameraFlowEls.detect.disabled,
      cameraVisualize: cameraFlowEls.visualize.checked && !cameraFlowEls.visualize.disabled,
      cameraSensorDepth: cameraFlowEls.sensorDepth.checked && !cameraFlowEls.sensorDepth.disabled,
      cameraUndistort: cameraFlowEls.undistort.checked && !cameraFlowEls.undistort.disabled,
      cameraKalman: cameraFlowEls.kalman.checked && !cameraFlowEls.kalman.disabled,
      cameraEllipseFit: cameraFlowEls.ellipseFit.checked && !cameraFlowEls.ellipseFit.disabled,
      cameraTemporalTracker: cameraFlowEls.temporalTracker.checked && !cameraFlowEls.temporalTracker.disabled,
      cameraColorFilter: cameraFlowEls.colorFilter.checked && !cameraFlowEls.colorFilter.disabled,
      dt
    });
    if (latest.observedBall) {
      observedBall = {x: latest.observedBall[0], y: latest.observedBall[1], z: latest.observedBall[2]};
    } else {
      observedBall = {...ball};
    }
    if (latest.ballSource === 'camera' && latest.rawBall) {
      ball = {x: latest.rawBall[0], y: latest.rawBall[1], z: latest.rawBall[2]};
      setInputsFromBall();
    }
    racket = {x: latest.racket[0], y: latest.racket[1], z: latest.racket[2]};
    const ros = latest.ros || {};
    const camera = latest.camera || {};
    updateVisionFrame(camera);
    jointEe = ros.jointEe ? {x: ros.jointEe[0], y: ros.jointEe[1], z: ros.jointEe[2]} : jointEe;
    cartEe = ros.cartPosition ? {x: ros.cartPosition[0], y: ros.cartPosition[1], z: ros.cartPosition[2]} : cartEe;
    trailBall.push({...ball});
    trailObservedBall.push({...observedBall});
    trailRacket.push({...racket});
    if (jointEe) trailJointEe.push({...jointEe});
    if (cartEe) trailCartEe.push({...cartEe});
    while (trailBall.length > 320) trailBall.shift();
    while (trailObservedBall.length > 320) trailObservedBall.shift();
    while (trailRacket.length > 320) trailRacket.shift();
    while (trailJointEe.length > 320) trailJointEe.shift();
    while (trailCartEe.length > 320) trailCartEe.shift();
    const cmdErr = Number.isFinite(ros.jointCommandErrorAbsMax) ? ros.jointCommandErrorAbsMax : 0;
    plotHistory.push({action: latest.actionAbsMax || 0, err: cmdErr});
    while (plotHistory.length > 180) plotHistory.shift();
    modeEl.textContent = latest.reason ? `${latest.mode}: ${latest.reason}` : latest.mode;
    modeEl.className = latest.mode.includes('failsafe') ? 'badge stop' :
      ((latest.mode === 'capture_gate' || latest.mode === 'release_gate') ? 'badge warn' : 'badge');
    rosEl.textContent = ros.enabled
      ? (ros.useStateAllowed
        ? `ros ${ros.useState ? 'state' : 'local'}/${ros.publishCommands ? 'pub' : 'dry'} #${ros.publishCount || 0}`
        : (camera.enabled ? `ros camera ${camera.detected ? 'det' : 'wait'}` : 'ros node'))
      : 'ros=off';
    rosEl.className = ros.enabled ? (ros.publishCommands ? 'badge warn' : 'badge') : 'badge warn';
    timeEl.textContent = `t=${fmt(latest.time)}s`;
    ageEl.textContent = `age=${fmt(latest.age)}s`;
    const conds = Object.entries(latest.conditions || {}).filter(([, v]) => v).map(([k]) => k).join(', ') || 'none';
    const jointAge = Number.isFinite(ros.jointAgeS) ? `${fmt(ros.jointAgeS)}s` : 'none';
    const cartAge = Number.isFinite(ros.cartAgeS) ? `${fmt(ros.cartAgeS)}s` : 'none';
    const est = latest.estimator || {};
    const camRt = camera.runtime || {};
    const imageAges = camera.imageAgesS || {};
    const imageAgeBits = Object.entries(imageAges)
      .map(([k, v]) => `${k}=${Number.isFinite(v) ? fmt(v) : 'none'}s`)
      .join(' ');
    const cameraSource = camera.sourceMode || 'ros_topic';
    const sensorDepthOn = camRt.sensorDepth !== false;
    const cameraPath = cameraSource === 'direct_realsense'
      ? `direct_realsense ${camera.imageWidth || '?'}x${camera.imageHeight || '?'}@${camera.realsenseFps || '?'} hybrid=${camera.useHybridDetector ? 'on' : 'off'} sensorDepth=${sensorDepthOn ? 'on' : 'off'} active=${camera.activeTransport || 'none'} ${imageAgeBits}`
      : `topics transport=${camera.imageTransport || 'none'} active=${camera.activeTransport || 'none'} compressed=${camera.imageTopic || 'none'} raw=${camera.rawImageTopic || 'none'} ${imageAgeBits}`;
    const measured = latest.measuredBall || [NaN, NaN, NaN];
    const commandRacket = latest.commandRacket || null;
    const racketSource = latest.racketSource || 'command';
    const camDetectorBall = camera.detectorBall || null;
    const camControlBall = camera.controlBall || null;
    const releaseGate = latest.releaseGate || {};
    const wouldDone = latest.wouldDone || {};
    const wouldDoneActive = Object.entries(wouldDone.conditions || {})
      .filter(([, v]) => v).map(([k]) => k).join(', ') || 'none';
    const estBits = est.enabled
      ? `${est.state || 'unknown'} ${est.predicted ? 'pred' : 'meas'} loss=${Number.isFinite(est.lossS) ? fmt(est.lossS) : 'none'}s extrap=${Number.isFinite(est.extrapolationS) ? fmt(est.extrapolationS) : 'none'}s acc=${est.updatesAccepted || 0} rej=${est.updatesRejected || 0}`
      : 'off';
    const releaseBits = releaseGate.enabled
      ? `${releaseGate.released ? 'cleared' : 'holding'} valid=${releaseGate.valid ? 'true' : 'false'} fresh=${releaseGate.fresh ? 'true' : 'false'} vz=${Number.isFinite(releaseGate.vzMS) ? fmt(releaseGate.vzMS) : 'none'} z_drop=${Number.isFinite(releaseGate.zDropM) ? fmt(releaseGate.zDropM) : 'none'} ticks=${releaseGate.candidateTicks || 0}/${releaseGate.requiredTicks || 0}`
      : 'off';
    readoutEl.innerHTML = [
      `runtime: ${ros.enabled ? `${ros.useState ? 'ros-state' : 'local-state'} / ${ros.publishCommands ? 'publish' : 'dry-run'}` : 'local only'} ui=${fmt(STEP_RATE_HZ, 0)}Hz display=${fmt(DISPLAY_RATE_HZ, 0)}Hz`,
      `ball source: ${latest.ballSource || ballSourceEl.value}`,
      `camera: ${camera.enabled ? `${cameraSource} / ${camera.cameraInfoReceived ? 'info' : 'no-info'} / ${camera.detectorReady ? 'detector' : 'no-detector'} / ${camera.detected ? 'detected' : 'lost'} age=${Number.isFinite(camera.ageS) ? fmt(camera.ageS) : 'none'}s hz=${Number.isFinite(camera.callbackHz) ? fmt(camera.callbackHz, 1) : '0.0'} frame=${camera.frame || 'none'}` : 'off'}`,
      camera.enabled ? `camera path: ${cameraPath}` : null,
      camera.enabled ? `camera flow: detect=${camRt.detect ? 'on' : 'off'} overlay=${camRt.visualize ? 'on' : 'off'} sensorDepth=${sensorDepthOn ? 'on' : 'off'} undistort=${camRt.undistort ? 'on' : 'off'} kalman=${camRt.kalman ? 'on' : 'off'} ellipse=${camRt.ellipseFit ? 'on' : 'off'} tracker=${camRt.temporalTracker ? 'on' : 'off'} color=${camRt.colorFilter ? 'on' : 'off'} frames=${camera.frameCount || 0} det=${camera.detectionCount || 0} vizHz=${Number.isFinite(camera.visualizationRateHz) ? fmt(camera.visualizationRateHz, 0) : '60'} web=${camera.webMaxWidth || 'full'}px/q${camera.webJpegQuality || 62}` : null,
      camDetectorBall ? `camera detector ${camera.detectorFrame || camera.frame || 'camera'}: (${fmt(camDetectorBall[0])}, ${fmt(camDetectorBall[1])}, ${fmt(camDetectorBall[2])})` : null,
      camControlBall ? `camera control sim: (${fmt(camControlBall[0])}, ${fmt(camControlBall[1])}, ${fmt(camControlBall[2])})` : null,
      camera.error ? `camera error: ${camera.error}` : null,
      `raw ball: (${fmt(ball.x)}, ${fmt(ball.y)}, ${fmt(ball.z)})`,
      `measured: (${fmt(measured[0])}, ${fmt(measured[1])}, ${fmt(measured[2])}) raw_valid=${latest.rawValid ? 'true' : 'false'}`,
      `control obs: (${fmt(observedBall.x)}, ${fmt(observedBall.y)}, ${fmt(observedBall.z)}) valid=${latest.valid ? 'true' : 'false'}`,
      `would done now: ${wouldDone.done ? `YES ${wouldDone.reason || ''}` : 'no'} active=${wouldDoneActive}`,
      `release gate: ${releaseBits}`,
      `estimator: ${estBits}`,
      `robot EE [${racketSource}]: (${fmt(racket.x)}, ${fmt(racket.y)}, ${fmt(racket.z)})`,
      commandRacket && racketSource !== 'command' ? `cmd EE: (${fmt(commandRacket[0])}, ${fmt(commandRacket[1])}, ${fmt(commandRacket[2])})` : null,
      jointEe ? `joint FK: (${fmt(jointEe.x)}, ${fmt(jointEe.y)}, ${fmt(jointEe.z)}) age=${jointAge}` : `joint FK: none age=${jointAge}`,
      cartEe ? `cart: (${fmt(cartEe.x)}, ${fmt(cartEe.y)}, ${fmt(cartEe.z)}) age=${cartAge}` : `cart: none age=${cartAge}`,
      `cmd-joint err max: ${Number.isFinite(cmdErr) ? fmt(cmdErr) : 'none'}`,
      `noise: ${fmt((latest.noiseStdM || 0) * 1000, 1)} mm`,
      `action max: ${fmt(latest.actionAbsMax)}`,
      `done hits: ${conds}`,
      `fk: ${latest.fkOk ? 'ok' : 'off'}`
    ].filter(Boolean).join('<br>');
  } catch (err) {
    modeEl.textContent = 'server error';
    modeEl.className = 'badge stop';
  } finally {
    inFlight = false;
  }
}

document.getElementById('reset').addEventListener('click', async () => {
  const data = await postJson('/reset', {});
  latest = data;
  ball = {x: CONFIG.initialBall[0], y: CONFIG.initialBall[1], z: CONFIG.initialBall[2]};
  observedBall = {...ball};
  jointEe = null;
  cartEe = null;
  lastSentBall = {...ball};
  trailBall = [];
  trailObservedBall = [];
  trailRacket = [];
  trailJointEe = [];
  trailCartEe = [];
  plotHistory = [];
  lastVisionSeq = 0;
  visibleEl.checked = true;
  setInputsFromBall();
  drawAll();
});

visibleEl.addEventListener('change', drawAll);
ballSourceEl.addEventListener('change', drawAll);
useRosStateEl.addEventListener('change', () => {
  if (publishCommandsEl.checked && !publishCommandsEl.disabled) {
    useRosStateEl.checked = true;
  }
  drawAll();
});
publishCommandsEl.addEventListener('change', () => {
  if (publishCommandsEl.checked && !publishCommandsEl.disabled) {
    useRosStateEl.checked = true;
  }
  drawAll();
});
for (const el of Object.values(curveEls)) {
  el.addEventListener('change', drawAll);
}
for (const el of Object.values(flowEls)) {
  el.addEventListener('change', drawAll);
}
for (const el of Object.values(cameraFlowEls)) {
  el.addEventListener('change', drawAll);
}
window.addEventListener('resize', () => {
  resizeCanvas(view3dCanvas);
  resizeCanvas(topCanvas);
  resizeCanvas(sideCanvas);
  resizeCanvas(plotCanvas);
  drawAll();
});

resizeCanvas(view3dCanvas);
resizeCanvas(topCanvas);
resizeCanvas(sideCanvas);
resizeCanvas(plotCanvas);
setRuntimeDefaults();
setInputsFromBall();
drawAll();
requestAnimationFrame(renderLoop);
setInterval(step, 1000 / STEP_RATE_HZ);
</script>
</body>
</html>
"""


class ProbeRequestHandler(BaseHTTPRequestHandler):
    probe: PolicySafetyProbe

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[probe] " + (fmt % args) + "\n")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            data = {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = HTML_TEMPLATE.replace(
                "__CONFIG_JSON__",
                json.dumps(self.probe.browser_config(), separators=(",", ":")),
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html)
            return
        if self.path == "/config":
            self._send_json(self.probe.browser_config())
            return
        if self.path.startswith("/camera_frame.jpg"):
            frame, info = self.probe.latest_camera_visualization()
            if frame is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Camera-Frame-Seq", str(info.get("seq", 0)))
            self.end_headers()
            self.wfile.write(frame)
            return
        if self.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if self.path == "/step":
                self._send_json(self.probe.step(self._read_json()))
                return
            if self.path == "/reset":
                self._send_json(self.probe.reset())
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch an interactive browser policy safety probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--robot-xml-path", type=Path, default=DEFAULT_ROBOT_XML)
    p.add_argument("--policy-obs-dim", type=int, choices=(50, 67), default=67)
    p.add_argument("--control-rate-hz", type=finite_float, default=200.0)
    p.add_argument("--ball-lost-failsafe-timeout-s", type=finite_float, default=0.6)
    p.add_argument("--disable-ball-capture-gate", action="store_true")
    p.add_argument("--disable-ball-release-gate", action="store_true")
    p.add_argument("--disable-ball-lost-failsafe", action="store_true")
    p.add_argument("--disable-done-monitoring", action="store_true")
    p.add_argument("--ball-release-min-downward-velocity-m-s", type=finite_float, default=0.15)
    p.add_argument("--ball-release-min-z-drop-m", type=finite_float, default=0.015)
    p.add_argument("--ball-release-window-s", type=finite_float, default=0.12)
    p.add_argument("--ball-release-required-consecutive-ticks", type=int, default=3)
    p.add_argument("--ball-release-max-obs-age-s", type=finite_float, default=0.08)
    p.add_argument("--done-z-min-m", type=finite_float, default=0.3)
    p.add_argument("--done-z-max-m", type=finite_float, default=1.9)
    p.add_argument("--done-x-range-m", type=finite_float, default=0.5)
    p.add_argument("--done-y-range-m", type=finite_float, default=0.5)
    p.add_argument("--done-racket-anchor-distance-m", type=finite_float, default=1.1)
    p.add_argument("--done-racket-z-high-offset-m", type=finite_float, default=0.24)
    p.add_argument("--done-racket-z-low-offset-m", type=finite_float, default=-0.12)
    p.add_argument("--ee-x-limits-m", default="-0.298,0.163")
    p.add_argument("--ee-y-limits-m", default="-0.502,-0.290")
    p.add_argument("--ee-z-limits-m", default="0.692,1.462")
    p.add_argument(
        "--ball-noise-std-m",
        type=finite_float,
        default=0.0,
        help="Default Gaussian position noise std for ball observations, in meters.",
    )
    p.add_argument("--ball-observation-rate-hz", type=finite_float, default=60.0)
    p.add_argument(
        "--browser-step-rate-hz",
        type=finite_float,
        default=60.0,
        help="Browser -> backend policy probe update cadence.",
    )
    p.add_argument(
        "--display-rate-hz",
        type=finite_float,
        default=60.0,
        help="Browser canvas redraw cadence.",
    )
    p.add_argument(
        "--disable-ball-state-estimator",
        action="store_true",
        help="Disable the pingpong_node-style 60 Hz update / 200 Hz predict ball estimator.",
    )
    p.add_argument(
        "--disable-ball-estimator-prediction",
        action="store_true",
        help="Keep estimator smoothing/loss aging but disable ballistic predict_to extrapolation.",
    )
    p.add_argument(
        "--disable-ball-estimator-aero-drag",
        action="store_true",
        help="Disable quadratic aerodynamic drag in estimator prediction.",
    )
    p.add_argument("--ball-estimator-max-extrapolation-s", type=finite_float, default=0.45)
    p.add_argument("--ball-estimator-trust-loss-s", type=finite_float, default=0.10)
    p.add_argument("--ball-estimator-degrade-loss-s", type=finite_float, default=0.40)
    p.add_argument("--ball-estimator-invalid-after-s", type=finite_float, default=0.60)
    p.add_argument("--ball-estimator-prediction-horizon-s", type=finite_float, default=0.0)
    p.add_argument(
        "--ros-bridge",
        action="store_true",
        help="Enable ROS2 bridge: publish /mx_mix_command and subscribe feedback topics.",
    )
    p.add_argument(
        "--disable-command-publish",
        action="store_true",
        help="With --ros-bridge, subscribe feedback only and do not publish commands.",
    )
    p.add_argument("--command-topic", default="/mx_mix_command")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--cart-state-topic", default="/cart_states")
    p.add_argument("--right-arm-mu-idx", type=int, default=3)
    p.add_argument(
        "--camera-ball",
        action="store_true",
        help="Enable pingpong_node-style camera ball source.",
    )
    p.add_argument(
        "--camera-source-mode",
        choices=("direct_realsense", "ros_topic"),
        default="direct_realsense",
        help="Camera source path. direct_realsense matches use_direct_realsense:=true.",
    )
    p.add_argument(
        "--disable-hybrid-detector",
        action="store_true",
        help="Disable the hybrid detector path. direct_realsense currently expects hybrid.",
    )
    p.add_argument(
        "--disable-depth-sensor",
        action="store_true",
        help="Disable hardware depth for the hybrid/direct RealSense path.",
    )
    p.add_argument("--image-width", type=int, default=848)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--realsense-fps", type=int, default=60)
    p.add_argument("--camera-image-topic", default="/camera/camera/color/image_raw/compressed")
    p.add_argument("--camera-raw-image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--camera-image-transport", choices=("compressed", "raw", "auto"), default="auto")
    p.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    p.add_argument("--calibration-json-path", default=str(DEFAULT_CALIBRATION_JSON))
    p.add_argument("--yolo-model-path", type=Path, default=DEFAULT_VISION_MODEL)
    p.add_argument("--target-class", default="Class 0")
    p.add_argument("--conf-threshold", type=finite_float, default=0.25)
    p.add_argument("--ir-params-json", default=str(DEFAULT_IR_PARAMS_JSON))
    p.add_argument("--yolo-interval", type=int, default=1)
    p.add_argument("--ir-threshold", type=int, default=170)
    p.add_argument("--disable-adaptive-ir", action="store_true")
    p.add_argument("--ir-exposure", type=int, default=1000)
    p.add_argument("--ir-gain", type=int, default=16)
    p.add_argument("--ir-laser-power", type=int, default=360)
    p.add_argument("--ir-min-radius", type=int, default=15)
    p.add_argument("--ir-max-radius", type=int, default=60)
    p.add_argument("--ir-circularity-min", type=finite_float, default=0.7)
    p.add_argument("--ir-brightness-percentile", type=int, default=25)
    p.add_argument("--hybrid-roi-margin", type=int, default=100)
    p.add_argument(
        "--depth-correction-model-path",
        default=str(DEFAULT_DEPTH_CORRECTION_MODEL),
    )
    p.add_argument("--depth-neighborhood-px", type=int, default=2)
    p.add_argument("--infer-imgsz", type=int, default=640)
    p.add_argument("--detector-device", default="0")
    p.add_argument("--detector-half", action="store_true")
    p.add_argument("--use-kalman", action="store_true")
    p.add_argument("--camera-tilt-deg", type=finite_float, default=50.33)
    p.add_argument("--disable-undistort", action="store_true")
    p.add_argument("--disable-ellipse-fit", action="store_true")
    p.add_argument("--enable-temporal-tracker", action="store_true")
    p.add_argument("--tracker-max-miss-bridge", type=int, default=2)
    p.add_argument("--tracker-gate-px", type=finite_float, default=120.0)
    p.add_argument("--hsv-lower", type=parse_int_triplet, default=(5, 50, 50))
    p.add_argument("--hsv-upper", type=parse_int_triplet, default=(30, 255, 255))
    p.add_argument("--color-match-threshold", type=finite_float, default=0.3)
    p.add_argument("--color-circularity-min", type=finite_float, default=0.4)
    p.add_argument("--disable-camera-detection", action="store_true")
    p.add_argument("--disable-camera-visualization", action="store_true")
    p.add_argument(
        "--camera-visualization-rate-hz",
        type=finite_float,
        default=60.0,
        help="Max rate for JPEG camera overlay rendering in the backend.",
    )
    p.add_argument(
        "--camera-web-max-width",
        type=int,
        default=848,
        help="Resize HTML camera JPEG frames to this max width; <=0 keeps full size.",
    )
    p.add_argument(
        "--camera-web-jpeg-quality",
        type=int,
        default=62,
        help="JPEG quality for the HTML camera stream.",
    )
    p.add_argument("--disable-camera-color-filter", action="store_true")
    p.add_argument("--sim-cartesian-x-offset-m", type=finite_float, default=0.0)
    p.add_argument("--sim-cartesian-y-offset-m", type=finite_float, default=0.023583)
    p.add_argument("--sim-base-height-m", type=finite_float, default=0.947396)
    p.add_argument("--open-browser", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> ProbeConfig:
    model_path = args.model_path.expanduser().resolve()
    robot_xml = args.robot_xml_path.expanduser().resolve()
    yolo_model_path = args.yolo_model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"robot XML not found: {robot_xml}")
    if bool(args.camera_ball) and not yolo_model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
    return ProbeConfig(
        host=args.host,
        port=int(args.port),
        model_path=model_path,
        robot_xml_path=robot_xml,
        policy_obs_dim=int(args.policy_obs_dim),
        control_rate_hz=float(args.control_rate_hz),
        ball_lost_timeout_s=float(args.ball_lost_failsafe_timeout_s),
        enable_ball_capture_gate=not bool(args.disable_ball_capture_gate),
        enable_ball_release_gate=not bool(args.disable_ball_release_gate),
        enable_ball_lost_failsafe=not bool(args.disable_ball_lost_failsafe),
        enable_done_monitoring=not bool(args.disable_done_monitoring),
        ball_release_min_downward_velocity_m_s=max(
            0.0, float(args.ball_release_min_downward_velocity_m_s)),
        ball_release_min_z_drop_m=max(
            0.0, float(args.ball_release_min_z_drop_m)),
        ball_release_window_s=max(0.02, float(args.ball_release_window_s)),
        ball_release_required_consecutive_ticks=max(
            1, int(args.ball_release_required_consecutive_ticks)),
        ball_release_max_obs_age_s=max(
            0.0, float(args.ball_release_max_obs_age_s)),
        done_z_min_m=float(args.done_z_min_m),
        done_z_max_m=float(args.done_z_max_m),
        done_x_range_m=float(args.done_x_range_m),
        done_y_range_m=float(args.done_y_range_m),
        done_racket_anchor_distance_m=float(args.done_racket_anchor_distance_m),
        done_racket_z_high_offset_m=float(args.done_racket_z_high_offset_m),
        done_racket_z_low_offset_m=float(args.done_racket_z_low_offset_m),
        ee_x_limits_m=parse_limits(args.ee_x_limits_m),
        ee_y_limits_m=parse_limits(args.ee_y_limits_m),
        ee_z_limits_m=parse_limits(args.ee_z_limits_m),
        ball_noise_std_m=max(0.0, float(args.ball_noise_std_m)),
        ball_observation_rate_hz=min(
            240.0, max(1.0, float(args.ball_observation_rate_hz))),
        browser_step_rate_hz=min(
            120.0, max(1.0, float(args.browser_step_rate_hz))),
        display_rate_hz=min(
            240.0, max(1.0, float(args.display_rate_hz))),
        enable_ball_state_estimator=not bool(args.disable_ball_state_estimator),
        ball_estimator_enable_prediction=not bool(
            args.disable_ball_estimator_prediction),
        ball_estimator_enable_aero_drag=not bool(
            args.disable_ball_estimator_aero_drag),
        ball_estimator_max_extrapolation_s=max(
            0.0, float(args.ball_estimator_max_extrapolation_s)),
        ball_estimator_trust_loss_s=max(
            0.0, float(args.ball_estimator_trust_loss_s)),
        ball_estimator_degrade_loss_s=max(
            0.0, float(args.ball_estimator_degrade_loss_s)),
        ball_estimator_invalid_after_s=max(
            0.0, float(args.ball_estimator_invalid_after_s)),
        ball_estimator_prediction_horizon_s=max(
            0.0, float(args.ball_estimator_prediction_horizon_s)),
        ros_bridge=bool(args.ros_bridge),
        ros_publish_commands=bool(args.ros_bridge and not args.disable_command_publish),
        command_topic=str(args.command_topic),
        joint_state_topic=str(args.joint_state_topic),
        cart_state_topic=str(args.cart_state_topic),
        right_arm_mu_idx=int(args.right_arm_mu_idx),
        enable_camera_ball=bool(args.camera_ball),
        camera_source_mode=str(args.camera_source_mode),
        use_hybrid_detector=not bool(args.disable_hybrid_detector),
        use_depth_sensor=not bool(args.disable_depth_sensor),
        image_width=max(1, int(args.image_width)),
        image_height=max(1, int(args.image_height)),
        realsense_fps=max(1, int(args.realsense_fps)),
        camera_image_topic=str(args.camera_image_topic),
        camera_raw_image_topic=str(args.camera_raw_image_topic),
        camera_image_transport=str(args.camera_image_transport),
        camera_info_topic=str(args.camera_info_topic),
        calibration_json_path=str(args.calibration_json_path),
        yolo_model_path=yolo_model_path,
        target_class=str(args.target_class),
        conf_threshold=float(args.conf_threshold),
        ir_params_json=str(args.ir_params_json),
        yolo_interval=max(1, int(args.yolo_interval)),
        ir_threshold=max(0, int(args.ir_threshold)),
        use_adaptive_ir=not bool(args.disable_adaptive_ir),
        ir_exposure=(
            int(args.ir_exposure) if int(args.ir_exposure) > 0 else None),
        ir_gain=(
            int(args.ir_gain) if int(args.ir_gain) > 0 else None),
        ir_laser_power=max(0, int(args.ir_laser_power)),
        ir_min_radius=max(0, int(args.ir_min_radius)),
        ir_max_radius=max(0, int(args.ir_max_radius)),
        ir_circularity_min=max(0.0, float(args.ir_circularity_min)),
        ir_brightness_percentile=max(0, int(args.ir_brightness_percentile)),
        hybrid_roi_margin=max(0, int(args.hybrid_roi_margin)),
        depth_correction_model_path=str(args.depth_correction_model_path),
        depth_neighborhood_px=max(0, int(args.depth_neighborhood_px)),
        infer_imgsz=max(1, int(args.infer_imgsz)),
        detector_device=str(args.detector_device),
        detector_half=bool(args.detector_half),
        use_kalman=bool(args.use_kalman),
        camera_tilt_deg=float(args.camera_tilt_deg),
        use_undistort=not bool(args.disable_undistort),
        enable_ellipse_fit=not bool(args.disable_ellipse_fit),
        enable_temporal_tracker=bool(args.enable_temporal_tracker),
        tracker_max_miss_bridge=max(0, int(args.tracker_max_miss_bridge)),
        tracker_gate_px=max(0.0, float(args.tracker_gate_px)),
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
        color_match_threshold=max(0.0, float(args.color_match_threshold)),
        color_circularity_min=max(0.0, float(args.color_circularity_min)),
        enable_camera_detection=not bool(args.disable_camera_detection),
        enable_camera_visualization=not bool(args.disable_camera_visualization),
        camera_visualization_rate_hz=min(
            120.0, max(1.0, float(args.camera_visualization_rate_hz))),
        camera_web_max_width=max(0, int(args.camera_web_max_width)),
        camera_web_jpeg_quality=min(
            95, max(25, int(args.camera_web_jpeg_quality))),
        enable_camera_color_filter=not bool(args.disable_camera_color_filter),
        sim_cartesian_x_offset_m=float(args.sim_cartesian_x_offset_m),
        sim_cartesian_y_offset_m=float(args.sim_cartesian_y_offset_m),
        sim_base_height_m=float(args.sim_base_height_m),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    probe = PolicySafetyProbe(cfg)
    if args.smoke_test:
        try:
            ball = np.asarray(probe.initial_ball_pos(), dtype=np.float32)
            first = probe.step({"ball": ball.tolist(), "velocity": [0, 0, 0], "valid": True, "dt": 0.02})
            released_ball = ball.copy()
            released_ball[2] -= max(0.03, cfg.ball_release_min_z_drop_m * 2.0)
            released = probe.step({
                "ball": released_ball.tolist(),
                "velocity": [0, 0, -0.6],
                "valid": True,
                "dt": max(0.04, cfg.ball_release_window_s),
            })
            probe.step({"ball": ball.tolist(), "velocity": [0, 0, 0], "valid": False, "dt": 0.25})
            hidden = probe.step({"ball": ball.tolist(), "velocity": [0, 0, 0], "valid": False, "dt": 0.25})
            noisy = probe.step({
                "ball": ball.tolist(),
                "velocity": [0, 0, 0],
                "valid": True,
                "noiseStdM": max(0.003, cfg.ball_noise_std_m),
                "dt": 0.02,
            })
            print(json.dumps({
                "first_mode": first["mode"],
                "released_mode": released["mode"],
                "hidden_mode": hidden["mode"],
                "noisy_observed_ball": noisy["observedBall"],
                "ros_enabled": first["ros"]["enabled"],
                "ros_publish_commands": first["ros"]["publishCommands"],
                "anchor": first["anchor"],
                "racket": first["racket"],
                "actionAbsMax": first["actionAbsMax"],
            }, indent=2))
        finally:
            if probe.ros_bridge is not None:
                probe.ros_bridge.shutdown()
        return 0

    ProbeRequestHandler.probe = probe
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), ProbeRequestHandler)
    url = f"http://{cfg.host}:{cfg.port}/"
    print("========================================================================")
    print("Interactive policy safety probe")
    print("========================================================================")
    print(f"URL: {url}")
    print(f"Model: {cfg.model_path}")
    print(f"Robot XML: {cfg.robot_xml_path}")
    if cfg.enable_camera_ball:
        if cfg.camera_source_mode == "direct_realsense":
            print(
                "Camera: "
                f"source=direct_realsense, hybrid={cfg.use_hybrid_detector}, "
                f"depth={cfg.use_depth_sensor}, "
                f"{cfg.image_width}x{cfg.image_height}@{cfg.realsense_fps}, "
                f"ir={cfg.ir_params_json}")
        else:
            print(
                "Camera: "
                f"source=ros_topic, transport={cfg.camera_image_transport}, "
                f"compressed={cfg.camera_image_topic}, "
                f"raw={cfg.camera_raw_image_topic}, "
                f"info={cfg.camera_info_topic}")
    print("Press Ctrl-C to stop.")
    print("")
    if args.open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n[probe] stopped")
    finally:
        httpd.server_close()
        if probe.ros_bridge is not None:
            probe.ros_bridge.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
