#!/home/jacky/miniconda3/envs/gmr/bin/python
"""
PingPong Controller Node - ROS2 node for ping-pong ball juggling control.

Supports two image input modes:
  - direct_realsense (default): reads BGR frames from RealSense via pyrealsense2
  - ros_topic: subscribes to ROS CompressedImage + CameraInfo topics

RL control will be added later.
"""

import numpy as np
import cv2
import csv
from dataclasses import dataclass
from typing import Optional
import time
import os
import threading
import glob

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import CompressedImage, CameraInfo, JointState
from builtin_interfaces.msg import Time
from mc_core_interface.msg import MechUnitCmd
from nav_msgs.msg import Odometry

from pingpong_controller.ball_detector import PingPongBallDetector
from pingpong_controller.calibration_loader import load_calibration
from pingpong_controller.rl_policy import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEG_WAIST_JOINTS,
    RIGHT_ARM_JOINTS,
    RLPolicyController,
    TARGET_DEGREES,
    target_head_q_rad,
    target_leg_waist_q_rad,
    target_left_arm_q_rad,
    target_right_arm_q_rad,
)
from pingpong_controller.mjx_policy import MJXPolicyController
from pingpong_controller.safety_limiter import RightArmCommandSafetyLimiter


@dataclass
class BallState:
    """Represents the detected state of the ping-pong ball.

    Units: position and velocity are in mm and mm/s (from ball_detector output).
    """
    position: np.ndarray  # 3D position [x, y, z] in mm
    velocity: np.ndarray  # 3D velocity [vx, vy, vz] in mm/s
    stamp: Time           # ROS timestamp
    valid: bool           # Whether detection is valid
    raw_measurement: dict = None  # Raw detector output


class PingPongControllerNode(Node):
    """ROS2 node for ping-pong ball juggling control."""

    @staticmethod
    def _resolve_source_module_dir(module_dir):
        """Prefer the workspace src package directory for generated artifacts."""
        install_marker = os.sep + 'install' + os.sep
        if install_marker in module_dir:
            workspace_root = module_dir.split(install_marker, 1)[0]
            source_module_dir = os.path.join(
                workspace_root, 'src', 'pingpong_controller',
                'pingpong_controller')
            if os.path.isdir(source_module_dir):
                return source_module_dir
        return module_dir

    def __init__(self):
        super().__init__('pingpong_controller')

        # Declare ROS parameters
        self.declare_parameter(
            'image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter(
            'camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('command_topic', '/mx_mix_command')
        self.declare_parameter('joint_state_topic', '/joint_states')
        self.declare_parameter('joint_names', list(RIGHT_ARM_JOINTS))
        default_right_arm_rad = [
            float(x) for x in target_right_arm_q_rad().tolist()]
        self.declare_parameter(
            'default_joint_positions', default_right_arm_rad)
        self.declare_parameter('publish_invalid_detection', False)
        self.declare_parameter('use_direct_realsense', False)
        self.declare_parameter('save_trajectory_plot', True)

        # Debug/monitoring topics under /pingpong namespace.
        self.declare_parameter('ball_state_topic', '/pingpong/ball_state')
        self.declare_parameter('publish_ball_state_debug', True)
        self.declare_parameter(
            'rl_joint_cmd_state_topic', '/pingpong/rl_joint_cmd_state')
        self.declare_parameter('publish_rl_joint_cmd_debug', True)
        self.declare_parameter('record_rl_trace', False)
        self.declare_parameter('rl_trace_csv_path', 'auto')
        self.declare_parameter('rl_trace_sample_every', 1)
        self.declare_parameter('rl_trace_flush_every', 100)

        # RL policy parameters
        default_rl_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models', 'best_model.zip')
        default_robot_xml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models', 'moz1_pd.xml')
        self.declare_parameter('rl_model_path', default_rl_model_path)
        self.declare_parameter('enable_rl_policy', True)
        self.declare_parameter('rl_policy_backend', 'sb3')
        # 200 Hz control loop -> dt = 0.005 s.
        self.declare_parameter('rl_policy_dt', 0.005)
        self.declare_parameter('control_rate_hz', 200.0)
        self.declare_parameter('rl_policy_device', 'cpu')
        self.declare_parameter('rl_ball_obs_age_clip', 0.20)
        self.declare_parameter('rl_action_gain', 1.0)
        self.declare_parameter('rl_action_scale_mult', 1.0)
        self.declare_parameter('rl_require_fk', False)
        self.declare_parameter('robot_xml_path', default_robot_xml)

        # Whole-body init pose publishing.
        self.declare_parameter('enable_init_pose', True)
        self.declare_parameter('init_pose_duration_s', 3.0)
        self.declare_parameter('init_pose_tolerance_rad', 0.05)
        # Mechanical unit indices. All four MechUnitCmd streams share
        # command_topic (default /mx_mix_command). TODO(jacky): set these to
        # real unit ids before deployment.
        self.declare_parameter('right_arm_mu_idx', 0)
        self.declare_parameter('left_arm_mu_idx', 1)
        self.declare_parameter('leg_waist_mu_idx', 2)
        self.declare_parameter('head_mu_idx', 3)

        # Ball detector parameters
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.module_dir = module_dir
        self.source_module_dir = self._resolve_source_module_dir(module_dir)
        default_model_path = os.path.join(module_dir, 'models', 'best.pt')
        default_trajectory_plot_path = os.path.join(
            self.source_module_dir, 'outputs', 'pingpong_base_trajectory.png')

        self.declare_parameter('yolo_model_path', default_model_path)
        self.declare_parameter(
            'trajectory_plot_path', default_trajectory_plot_path)
        self.declare_parameter('calibration_json_path', 'auto')
        self.declare_parameter('target_class', 'pingpong ball')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('infer_imgsz', 640)
        self.declare_parameter('device', '0')
        self.declare_parameter('half', False)
        self.declare_parameter('use_kalman', False)
        self.declare_parameter('camera_tilt_deg', 40.0)
        self.declare_parameter('use_undistort', True)
        self.declare_parameter('print_every', 1)
        self.declare_parameter('realsense_fps', 60)

        # Camera intrinsics (optional override, default from CameraInfo)
        self.declare_parameter('camera_fx', 0.0)
        self.declare_parameter('camera_fy', 0.0)
        self.declare_parameter('camera_cx', 0.0)
        self.declare_parameter('camera_cy', 0.0)
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)

        # Read parameters
        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        command_topic = self.get_parameter('command_topic').value
        joint_state_topic = self.get_parameter('joint_state_topic').value
        self.joint_names = self.get_parameter('joint_names').value
        self.default_joint_positions = self.get_parameter(
            'default_joint_positions').value
        self.publish_invalid_detection = self.get_parameter(
            'publish_invalid_detection').value
        self.use_direct_realsense = self.get_parameter(
            'use_direct_realsense').value
        self.save_trajectory_plot_enabled = self.get_parameter(
            'save_trajectory_plot').value
        self.trajectory_plot_path = self.get_parameter(
            'trajectory_plot_path').value

        # Debug/monitoring topic parameters.
        self.ball_state_topic = self.get_parameter('ball_state_topic').value
        self.publish_ball_state_debug = bool(
            self.get_parameter('publish_ball_state_debug').value)
        self.rl_joint_cmd_state_topic = self.get_parameter(
            'rl_joint_cmd_state_topic').value
        self.publish_rl_joint_cmd_debug = bool(
            self.get_parameter('publish_rl_joint_cmd_debug').value)
        self.record_rl_trace = bool(
            self.get_parameter('record_rl_trace').value)
        self.rl_trace_csv_path_param = self.get_parameter(
            'rl_trace_csv_path').value
        self.rl_trace_sample_every = max(
            1, int(self.get_parameter('rl_trace_sample_every').value))
        self.rl_trace_flush_every = max(
            1, int(self.get_parameter('rl_trace_flush_every').value))

        self.yolo_model_path = self.get_parameter('yolo_model_path').value
        self.calibration_json_path = self.get_parameter(
            'calibration_json_path').value
        self.target_class = self.get_parameter('target_class').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.infer_imgsz = self.get_parameter('infer_imgsz').value
        self.device = self.get_parameter('device').value
        self.half = self.get_parameter('half').value
        self.use_kalman = self.get_parameter('use_kalman').value
        self.camera_tilt_deg = self.get_parameter('camera_tilt_deg').value
        self.use_undistort = self.get_parameter('use_undistort').value
        self.print_every = self.get_parameter('print_every').value
        self.realsense_fps = self.get_parameter('realsense_fps').value

        self.override_fx = self.get_parameter('camera_fx').value
        self.override_fy = self.get_parameter('camera_fy').value
        self.override_cx = self.get_parameter('camera_cx').value
        self.override_cy = self.get_parameter('camera_cy').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value

        if len(self.joint_names) != len(self.default_joint_positions):
            raise ValueError(
                'joint_names and default_joint_positions must have the same length')

        # RL policy setup
        self.rl_model_path = self.get_parameter('rl_model_path').value
        self.enable_rl_policy = bool(
            self.get_parameter('enable_rl_policy').value)
        self.rl_policy_backend = str(
            self.get_parameter('rl_policy_backend').value).strip().lower()
        if self.rl_policy_backend not in ('sb3', 'mjx'):
            self.get_logger().warn(
                f'Unknown rl_policy_backend="{self.rl_policy_backend}"; '
                'falling back to "sb3"')
            self.rl_policy_backend = 'sb3'
        self.rl_policy_dt = float(self.get_parameter('rl_policy_dt').value)
        self.control_rate_hz = float(
            self.get_parameter('control_rate_hz').value)
        if not np.isfinite(self.control_rate_hz) or self.control_rate_hz <= 0.0:
            self.get_logger().warn(
                f'Invalid control_rate_hz={self.control_rate_hz}; '
                'falling back to 200.0 Hz')
            self.control_rate_hz = 200.0
        self.control_dt = 1.0 / self.control_rate_hz
        self.rl_policy_device = self.get_parameter('rl_policy_device').value
        self.rl_ball_obs_age_clip = float(
            self.get_parameter('rl_ball_obs_age_clip').value)
        self.rl_action_gain = float(
            self.get_parameter('rl_action_gain').value)
        self.rl_action_scale_mult = float(
            self.get_parameter('rl_action_scale_mult').value)
        self.rl_require_fk = bool(
            self.get_parameter('rl_require_fk').value)

        # Robot XML for FK (default to models/moz1_pd.xml in same directory).
        self.robot_xml_path = self.get_parameter('robot_xml_path').value

        self.rl_controller = None
        self._last_valid_ball_wall_time = None
        self._last_rl_ball_age_s = self.rl_ball_obs_age_clip
        # Post-safety-limiter command (what was last actually published).
        self._last_safe_joint_positions = np.array(
            self.default_joint_positions, dtype=np.float64)
        # Pre-safety-limiter RL output (useful for debugging/replay).
        self._last_raw_joint_positions = self._last_safe_joint_positions.copy()
        self._rl_expected_joint_count = len(RIGHT_ARM_JOINTS)

        # Warn early if the user sets mismatched RL dt vs control dt — the
        # RL integrator and the timer would drift otherwise.
        if abs(self.rl_policy_dt - self.control_dt) > 1e-6:
            self.get_logger().warn(
                f'rl_policy_dt={self.rl_policy_dt:.6f} does not match '
                f'control_dt={self.control_dt:.6f}; RL predict() will be '
                'called with dt=control_dt to stay in sync with the timer')

        if self.enable_rl_policy:
            if len(self.joint_names) != self._rl_expected_joint_count:
                self.get_logger().error(
                    f'RL policy expects {self._rl_expected_joint_count} '
                    f'joints but joint_names has {len(self.joint_names)}; '
                    'disabling RL policy')
                self.enable_rl_policy = False
            else:
                try:
                    if self.rl_policy_backend == 'mjx':
                        self.rl_controller = MJXPolicyController(
                            checkpoint_path=self.rl_model_path,
                            dt=self.control_dt,
                            action_gain=self.rl_action_gain,
                            action_scale_mult=self.rl_action_scale_mult,
                            ball_obs_age_clip_s=self.rl_ball_obs_age_clip,
                            robot_xml_path=self.robot_xml_path,
                            require_fk=self.rl_require_fk,
                            logger=self.get_logger(),
                        )
                        summary = self.rl_controller.checkpoint_summary()
                        self.get_logger().info(
                            'MJX policy summary: '
                            f"step={summary.get('step')}, "
                            f"fk_enabled={summary.get('fk_enabled')}, "
                            f"action_gain={summary.get('action_gain')}, "
                            f"action_scale_mult={summary.get('action_scale_mult')}, "
                            f"arm_action_limiter="
                            f"{summary.get('arm_action_limiter')}")
                    else:
                        self.rl_controller = RLPolicyController(
                            model_path=self.rl_model_path,
                            dt=self.control_dt,
                            ball_obs_age_clip_s=self.rl_ball_obs_age_clip,
                            device=self.rl_policy_device,
                            robot_xml_path=self.robot_xml_path,
                            logger=self.get_logger(),
                        )
                    self._last_safe_joint_positions = \
                        self.rl_controller.current_arm_cmd_q().astype(
                            np.float64)
                    self._last_raw_joint_positions = \
                        self._last_safe_joint_positions.copy()
                    self.get_logger().info(
                        f'RL controller ready '
                        f'(backend={self.rl_policy_backend}, '
                        f'model={self.rl_model_path})')
                except Exception as exc:
                    self.get_logger().error(
                        f'Failed to initialize RL controller '
                        f'(backend={self.rl_policy_backend}): {exc}; '
                        'falling back to default joint positions')
                    self.rl_controller = None
                    self.enable_rl_policy = False
        else:
            self.get_logger().info(
                'RL policy disabled (enable_rl_policy=False); '
                'publishing default joint positions')

        # Safety limiter: hardcoded RightArm position/vel/acc limits, applied
        # right before publishing MechUnitCmd. Must NOT be skipped even if the
        # RL policy or detector is disabled.
        initial_safe_cmd = (
            self.rl_controller.current_arm_cmd_q().astype(np.float64)
            if self.rl_controller is not None
            else np.asarray(target_right_arm_q_rad(), dtype=np.float64)
        )
        self.safety_limiter = RightArmCommandSafetyLimiter(
            initial_safe_cmd, dt=self.control_dt)
        self.get_logger().info(
            f'Safety limiter active: dt={self.control_dt:.4f}s '
            f'(control_rate_hz={self.control_rate_hz:.1f})')
        self._safety_log_period_s = 1.0
        self._safety_log_last_wall = time.time()

        # Init pose state (whole-body drive to TARGET_DEGREES before RL runs).
        self.enable_init_pose = bool(
            self.get_parameter('enable_init_pose').value)
        self.init_pose_duration_s = float(
            self.get_parameter('init_pose_duration_s').value)
        self.right_arm_mu_idx = int(
            self.get_parameter('right_arm_mu_idx').value)
        self.left_arm_mu_idx = int(
            self.get_parameter('left_arm_mu_idx').value)
        self.leg_waist_mu_idx = int(
            self.get_parameter('leg_waist_mu_idx').value)
        self.head_mu_idx = int(self.get_parameter('head_mu_idx').value)
        self.get_logger().info(
            f'MechUnitCmd -> "{command_topic}" '
            f'(right={self.right_arm_mu_idx}, left={self.left_arm_mu_idx}, '
            f'leg_waist={self.leg_waist_mu_idx}, head={self.head_mu_idx}); '
            'set mu_idx params to real unit ids before deployment')
        self._init_pose_active = bool(
            self.enable_init_pose and self.init_pose_duration_s > 0.0)
        self._init_pose_start_wall: Optional[float] = None
        if self._init_pose_active:
            self.get_logger().info(
                f'Init-pose phase enabled: driving full body to TARGET for '
                f'{self.init_pose_duration_s:.2f}s before starting RL policy')
        else:
            self.get_logger().info(
                'Init-pose phase disabled (enable_init_pose=False or '
                'init_pose_duration_s<=0); RL policy starts immediately')

        # Latest BallState cached by image callbacks; control timer consumes it.
        self._latest_ball_state = None
        self._latest_ball_lock = threading.Lock()

        # Latest real joint feedback from /joint_states; control timer + init
        # pose logic consume it. Values are indexed by joint name.
        self._joint_state_lock = threading.Lock()
        self._latest_joint_positions: dict = {}
        self._latest_joint_velocities: dict = {}
        self._latest_joint_state_stamp_sec: Optional[float] = None
        self._joint_state_received = False
        self._safety_synced_to_joint_state = False
        self._joint_state_log_warned_missing = False
        # Right-arm dq estimate from position differentiation: used when
        # JointState.velocity is empty or stuck at zero (seen on some
        # hardware bringups).
        self._prev_right_arm_q_for_dq: Optional[np.ndarray] = None
        self._prev_right_arm_stamp_for_dq: Optional[float] = None
        self._estimated_right_arm_dq = np.zeros(7, dtype=np.float32)
        self._prev_right_arm_dq_for_ddq: Optional[np.ndarray] = None
        self._prev_right_arm_stamp_for_ddq: Optional[float] = None
        self._estimated_right_arm_ddq = np.zeros(7, dtype=np.float32)
        # Hard cap for the finite-difference dq so a single noisy JointState
        # frame cannot feed unbounded dq into the RL obs.
        # Limits match RightArmCommandSafetyLimiter.VEL_LIMIT_DEG_S.
        self._right_arm_dq_clip_rad_s = np.deg2rad(np.array(
            [210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0],
            dtype=np.float32))
        self._right_arm_ddq_clip_rad_s2 = np.deg2rad(np.array(
            [1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0],
            dtype=np.float32))
        self._control_tick_index = 0

        # Optional CSV trace for real-robot policy experiments.
        self._rl_trace_file = None
        self._rl_trace_writer = None
        self._rl_trace_rows_since_flush = 0
        self._rl_trace_path = None
        self._init_rl_trace_logger()

        # Callback groups: isolate the 200 Hz control timer from the heavy
        # vision callback so YOLO does not block control publishing. Vision
        # uses MutuallyExclusiveCallbackGroup so image_callback and
        # camera_info_callback cannot re-enter the detector concurrently
        # (detector/YOLO/Kalman state is not guaranteed thread-safe).
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.vision_callback_group = MutuallyExclusiveCallbackGroup()
        # Dedicated group for /joint_states so state updates cannot block
        # the control timer either.
        self.state_callback_group = MutuallyExclusiveCallbackGroup()
        self.joint_state_sub = self.create_subscription(
            JointState, joint_state_topic,
            self._joint_state_callback, 10,
            callback_group=self.state_callback_group)
        self.get_logger().info(
            f'Subscribed to joint states: "{joint_state_topic}"')

        # --- Load full calibration JSON only for direct RealSense mode.
        # ROS topic mode waits for CameraInfo, then loads the matching JSON
        # only for T_base_camera.
        self.calibration_data = None
        if self.use_direct_realsense:
            self.calibration_json_path = self._resolve_calibration_json_path(
                self.calibration_json_path, module_dir)
        else:
            self.get_logger().info(
                'ROS topic mode: CameraInfo will provide intrinsics; '
                'matching calibration JSON will provide T_base_camera')

        if self.use_direct_realsense and os.path.exists(self.calibration_json_path):
            try:
                self.calibration_data = load_calibration(
                    self.calibration_json_path)
                self.get_logger().info(
                    f'Loaded calibration from: {self.calibration_json_path}')
                self.get_logger().info(f'  {self.calibration_data}')
            except Exception as e:
                self.get_logger().warn(
                    f'Failed to load calibration: {e}')
                self.get_logger().warn('Will use CameraInfo / RealSense intrinsics instead')
        else:
            if self.use_direct_realsense:
                self.get_logger().warn(
                    f'Calibration file not found: {self.calibration_json_path}')

        # --- Initialize detector from calibration only in direct RealSense mode.
        # ROS topic mode intentionally uses CameraInfo intrinsics instead.
        self.detector = None
        self.camera_info_received = False
        if self.use_direct_realsense and self.calibration_data is not None:
            self._init_detector_from_calibration()

        # --- Statistics ---
        self.frame_count = 0
        self.detection_count = 0
        self.last_callback_time = None
        self.callback_times = []
        self.trajectory_log = []
        self.trajectory_lock = threading.Lock()
        self._warned_non_base_trajectory = False

        # --- Publisher (created before any thread starts) ---
        # Single MechUnitCmd publisher on command_topic (default
        # /mx_mix_command). Used by both the init-pose phase (all four
        # mechanical units) and the RL phase (right arm only).
        self.cmd_pub = self.create_publisher(
            MechUnitCmd, command_topic, 10)

        # Debug/monitoring publishers under /pingpong namespace.
        self.ball_state_pub = self.create_publisher(
            Odometry, self.ball_state_topic, 10)
        self.rl_joint_cmd_state_pub = self.create_publisher(
            JointState, self.rl_joint_cmd_state_topic, 10)
        if self.publish_ball_state_debug:
            self.get_logger().info(
                f'Ball state debug: Odometry on "{self.ball_state_topic}"')
        if self.publish_rl_joint_cmd_debug:
            self.get_logger().info(
                f'RL joint cmd debug: JointState on '
                f'"{self.rl_joint_cmd_state_topic}"')

        # Fixed-rate control timer drives the publish path, decoupling it from
        # the camera callback cadence. The image callback only updates
        # self._latest_ball_state.
        self.control_timer = self.create_timer(
            self.control_dt, self._control_timer_callback,
            callback_group=self.control_callback_group)
        self.get_logger().info(
            f'Control timer started at {self.control_rate_hz:.1f} Hz '
            f'(dt={self.control_dt:.4f}s)')

        # --- RealSense state (may be set by _init_direct_realsense) ---
        self._rs_pipeline = None
        self._rs_running = False
        self._rs_thread = None

        # --- Branch on input mode ---
        if self.use_direct_realsense:
            self.get_logger().info('Image input mode: direct_realsense')
            self.get_logger().info(
                'Note: realsense2_camera ROS driver should NOT be running '
                'when use_direct_realsense=True')
            self._init_direct_realsense()
        else:
            self.get_logger().info('Image input mode: ros_topic')
            if self.calibration_data is not None:
                self.get_logger().info(
                    'ROS topic mode: using CameraInfo intrinsics; '
                    'calibration JSON will not initialize detector')
            self.camera_info_sub = self.create_subscription(
                CameraInfo, camera_info_topic,
                self._camera_info_callback, 10,
                callback_group=self.vision_callback_group)
            self.image_sub = self.create_subscription(
                CompressedImage, image_topic,
                self._image_callback, 10,
                callback_group=self.vision_callback_group)

        # --- Startup log ---
        self.get_logger().info(f'YOLO model: {self.yolo_model_path}')
        self.get_logger().info(
            f'Target class: {self.target_class}, '
            f'conf_threshold: {self.conf_threshold}')
        actual_undistort = (self.detector.use_undistort
                            if self.detector is not None
                            else self.use_undistort)
        detector_uses_calibration = (
            self.detector is not None
            and getattr(self.detector, 'calibration_data', None) is not None)
        if detector_uses_calibration and actual_undistort:
            undistort_text = 'True (forced by calibration JSON)'
        else:
            undistort_text = str(actual_undistort)
        self.get_logger().info(
            f'Use undistort: {undistort_text}, Use Kalman: {self.use_kalman}')
        self.get_logger().info(f'Print every: {self.print_every} frames')
        if self.detector is not None:
            self.get_logger().info('Detector ready, waiting for images...')
        elif not self.use_direct_realsense:
            self.get_logger().info('Waiting for CameraInfo...')

    # ------------------------------------------------------------------
    # Detector initialization helpers
    # ------------------------------------------------------------------

    def _resolve_calibration_json_path(
            self, requested_path, module_dir, image_width=None, image_height=None):
        """Resolve calibration JSON path from a manual path or image resolution."""
        image_width = self.image_width if image_width is None else int(image_width)
        image_height = self.image_height if image_height is None else int(image_height)

        if requested_path and str(requested_path).lower() not in ('auto', ''):
            return os.path.expandvars(os.path.expanduser(str(requested_path)))

        search_dirs = []
        for base_dir in (self.source_module_dir, module_dir):
            calib_dir = os.path.join(base_dir, 'outputs', 'vision_calib')
            if calib_dir not in search_dirs:
                search_dirs.append(calib_dir)

        candidates = []
        for calib_dir in search_dirs:
            candidates.extend(glob.glob(os.path.join(
                calib_dir, 'eye_hand_result_*_undistorted.json')))
        candidates = sorted(set(candidates))
        for candidate in candidates:
            try:
                calibration = load_calibration(candidate)
            except Exception:
                continue
            if (calibration.image_width == image_width
                    and calibration.image_height == image_height):
                self.get_logger().info(
                    f'Auto-selected calibration for '
                    f'{image_width}x{image_height}: {candidate}')
                return candidate

        fallback = os.path.join(
            search_dirs[0], f'eye_hand_result_{image_width}_undistorted.json')
        self.get_logger().warn(
            f'No calibration JSON matched {image_width}x'
            f'{image_height}; trying {fallback}')
        return fallback

    def _load_extrinsics_for_resolution(self, image_width, image_height):
        """Load matching calibration JSON for T_base_camera in ROS topic mode."""
        calibration_path = self._resolve_calibration_json_path(
            self.calibration_json_path,
            self.source_module_dir,
            image_width=image_width,
            image_height=image_height)

        if not os.path.exists(calibration_path):
            self.get_logger().warn(
                f'No calibration JSON found for {image_width}x{image_height}: '
                f'{calibration_path}; output will stay in camera frame')
            return None

        try:
            calibration = load_calibration(calibration_path)
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to load extrinsics calibration {calibration_path}: {exc}; '
                'output will stay in camera frame')
            return None

        if (calibration.image_width != image_width
                or calibration.image_height != image_height):
            self.get_logger().warn(
                f'Calibration JSON resolution {calibration.image_width}x'
                f'{calibration.image_height} does not match CameraInfo '
                f'{image_width}x{image_height}; output will stay in camera frame')
            return None

        self.get_logger().info(
            f'Loaded T_base_camera for ROS topic mode from: {calibration_path}')
        return calibration

    def _init_detector_from_calibration(self):
        """Initialize detector using calibration JSON data."""
        self.get_logger().info(
            'Initializing PingPongBallDetector with CalibrationData...')
        try:
            self.detector = PingPongBallDetector(
                yolo_model_path=self.yolo_model_path,
                ball_diameter_mm=40.0,
                width=self.calibration_data.image_width,
                height=self.calibration_data.image_height,
                fps=60,
                conf_threshold=self.conf_threshold,
                target_class=self.target_class,
                camera_tilt_deg=self.camera_tilt_deg,
                use_kalman=self.use_kalman,
                infer_imgsz=self.infer_imgsz,
                device=self.device,
                half=self.half,
                use_undistort=True,
                print_every=999999,
                profile_every=0,
                use_camera=False,
                calibration_data=self.calibration_data,
                output_frame='base_link'
            )
            self.camera_info_received = True
            self.get_logger().info(
                'PingPongBallDetector initialized (output: base_link)')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize detector: {e}')
            raise

    def _init_detector_from_realsense_intrinsics(self, intrinsics):
        """Initialize detector using intrinsics from RealSense profile."""
        self.get_logger().info(
            'Initializing PingPongBallDetector with RealSense intrinsics...')
        dist_coeffs = list(intrinsics.coeffs)
        try:
            self.detector = PingPongBallDetector(
                yolo_model_path=self.yolo_model_path,
                ball_diameter_mm=40.0,
                width=intrinsics.width,
                height=intrinsics.height,
                fps=self.realsense_fps,
                conf_threshold=self.conf_threshold,
                target_class=self.target_class,
                camera_tilt_deg=self.camera_tilt_deg,
                use_kalman=self.use_kalman,
                infer_imgsz=self.infer_imgsz,
                device=self.device,
                half=self.half,
                use_undistort=self.use_undistort,
                print_every=999999,
                profile_every=0,
                use_camera=False,
                camera_fx=intrinsics.fx,
                camera_fy=intrinsics.fy,
                camera_cx=intrinsics.ppx,
                camera_cy=intrinsics.ppy,
                camera_dist_coeffs=dist_coeffs,
                output_frame='camera'
            )
            self.camera_info_received = True
            self.get_logger().info(
                'PingPongBallDetector initialized (output: camera)')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize detector: {e}')
            raise

    # ------------------------------------------------------------------
    # Direct RealSense mode
    # ------------------------------------------------------------------

    def _start_color_stream(self, width, height, fps,
                            allow_resolution_fallback=False):
        """Start RealSense color stream with fallback modes.

        Args:
            allow_resolution_fallback: If False, only fps fallback is allowed
                (resolution stays at width x height). If True, 640x480 is
                tried as a last resort.
        """
        candidate_modes = [
            (width, height, fps),
            (width, height, 30),
        ]
        if allow_resolution_fallback:
            candidate_modes.append((640, 480, 30))

        tried = set()
        last_error = None
        for mode in candidate_modes:
            if mode in tried:
                continue
            tried.add(mode)
            w, h, f = mode
            config = rs.config()
            config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
            try:
                profile = self._rs_pipeline.start(config)
                if mode != (width, height, fps):
                    self.get_logger().warn(
                        f'Requested {width}x{height}@{fps} unavailable, '
                        f'fell back to {w}x{h}@{f}')
                return profile
            except RuntimeError as exc:
                last_error = exc
                self.get_logger().warn(f'Failed to start {w}x{h}@{f}: {exc}')
        raise RuntimeError(f'Cannot start RealSense color stream: {last_error}')

    def _init_direct_realsense(self):
        """Initialize RealSense pipeline, warmup, and start reader thread."""
        self._rs_pipeline = rs.pipeline()

        if self.calibration_data is not None:
            target_w = self.calibration_data.image_width
            target_h = self.calibration_data.image_height
            allow_fallback = False
        else:
            target_w = self.image_width
            target_h = self.image_height
            allow_fallback = True

        self.get_logger().info(
            f'Direct RealSense target resolution: '
            f'{target_w}x{target_h}@{self.realsense_fps}')
        if not allow_fallback:
            self.get_logger().info(
                'Resolution fallback disabled because calibration JSON is active')

        profile = self._start_color_stream(
            target_w, target_h, self.realsense_fps,
            allow_resolution_fallback=allow_fallback)

        color_stream = profile.get_stream(rs.stream.color)
        video_profile = color_stream.as_video_stream_profile()
        rs_intrinsics = video_profile.get_intrinsics()
        self.get_logger().info(
            f'RealSense stream: {rs_intrinsics.width}x{rs_intrinsics.height}'
            f'@{video_profile.fps()}fps')

        if self.calibration_data is not None:
            if (rs_intrinsics.width != self.calibration_data.image_width
                    or rs_intrinsics.height != self.calibration_data.image_height):
                raise RuntimeError(
                    f'RealSense stream resolution '
                    f'{rs_intrinsics.width}x{rs_intrinsics.height} does not '
                    f'match calibration JSON '
                    f'{self.calibration_data.image_width}x'
                    f'{self.calibration_data.image_height}')

        if self.detector is None:
            self._init_detector_from_realsense_intrinsics(rs_intrinsics)

        self.get_logger().info('Camera warmup (30 frames)...')
        for i in range(30):
            try:
                frames = self._rs_pipeline.wait_for_frames(timeout_ms=1000)
                cf = frames.get_color_frame()
                if cf and i == 0:
                    img = np.asanyarray(cf.get_data())
                    self.get_logger().info(
                        f'First frame: shape={img.shape}, dtype={img.dtype}')
            except Exception as e:
                self.get_logger().warn(f'Warmup frame {i} failed: {e}')
        self.get_logger().info('Camera warmup complete')

        self._rs_running = True
        self._rs_thread = threading.Thread(
            target=self._realsense_reader_loop, daemon=True)
        self._rs_thread.start()
        self.get_logger().info('RealSense reader thread started')

    def _realsense_reader_loop(self):
        """Background thread: read RealSense frames and run detection."""
        while self._rs_running:
            try:
                frames = self._rs_pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                if self._rs_running:
                    self.get_logger().warn(
                        f'wait_for_frames failed: {e}',
                        throttle_duration_sec=1.0)
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            if color_image is None or color_image.size == 0:
                continue
            if len(color_image.shape) != 3:
                continue

            stamp = self.get_clock().now().to_msg()
            self._handle_cv_image(color_image, stamp)

    def _shutdown_realsense(self):
        """Stop RealSense pipeline and reader thread."""
        if not self._rs_running:
            return
        self._rs_running = False
        if self._rs_thread is not None and self._rs_thread.is_alive():
            self._rs_thread.join(timeout=6.0)
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
                self.get_logger().info('RealSense pipeline stopped')
            except Exception as e:
                self.get_logger().warn(f'Error stopping pipeline: {e}')

    # ------------------------------------------------------------------
    # ROS topic mode callbacks
    # ------------------------------------------------------------------

    def _camera_info_callback(self, msg: CameraInfo):
        """Handle CameraInfo message and initialize detector in ROS topic mode."""
        if self.camera_info_received:
            return

        fx = msg.k[0] if self.override_fx == 0.0 else self.override_fx
        fy = msg.k[4] if self.override_fy == 0.0 else self.override_fy
        cx = msg.k[2] if self.override_cx == 0.0 else self.override_cx
        cy = msg.k[5] if self.override_cy == 0.0 else self.override_cy
        dist_coeffs = list(msg.d) if len(msg.d) > 0 else None

        self.get_logger().info('=' * 60)
        self.get_logger().info('CameraInfo received (ROS topic mode):')
        self.get_logger().info(
            f'  Intrinsics: fx={fx:.2f}, fy={fy:.2f}, '
            f'cx={cx:.2f}, cy={cy:.2f}')
        self.get_logger().info(
            f'  Resolution: {msg.width}x{msg.height}')
        self.get_logger().info(
            f'  Distortion model: {msg.distortion_model}')
        self.get_logger().info('=' * 60)

        self.get_logger().info(
            'Initializing PingPongBallDetector with CameraInfo...')
        extrinsics_calibration = self._load_extrinsics_for_resolution(
            msg.width, msg.height)
        output_frame = 'base_link' if extrinsics_calibration is not None else 'camera'

        try:
            self.detector = PingPongBallDetector(
                yolo_model_path=self.yolo_model_path,
                ball_diameter_mm=40.0,
                width=msg.width,
                height=msg.height,
                fps=60,
                conf_threshold=self.conf_threshold,
                target_class=self.target_class,
                camera_tilt_deg=self.camera_tilt_deg,
                use_kalman=self.use_kalman,
                infer_imgsz=self.infer_imgsz,
                device=self.device,
                half=self.half,
                use_undistort=self.use_undistort,
                print_every=999999,
                profile_every=0,
                use_camera=False,
                camera_fx=fx, camera_fy=fy,
                camera_cx=cx, camera_cy=cy,
                camera_dist_coeffs=dist_coeffs,
                output_frame=output_frame
            )
            if extrinsics_calibration is not None:
                # Keep CameraInfo intrinsics, but use matching JSON extrinsics
                # for camera -> base_link output.
                self.detector.calibration_data = extrinsics_calibration
                self.detector.T_base_camera = extrinsics_calibration.T_base_camera
                self.detector.R_base_camera = self.detector.T_base_camera[:3, :3]
                self.detector.output_frame = 'base_link'
            self.camera_info_received = True
            self.get_logger().info(
                f'PingPongBallDetector initialized (output: {output_frame})')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize detector: {e}')
            raise

    def _image_callback(self, msg: CompressedImage):
        """Handle compressed image messages (ROS topic mode)."""
        if not self.camera_info_received:
            if self.frame_count % 30 == 0:
                self.get_logger().info(
                    'Waiting for camera_info...', throttle_duration_sec=1.0)
            self.frame_count += 1
            return

        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warn(
                    'Failed to decode compressed image',
                    throttle_duration_sec=1.0)
                return
            self._handle_cv_image(image, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(
                f'Error in image callback: {e}',
                throttle_duration_sec=1.0)

    # ------------------------------------------------------------------
    # Shared detection logic
    # ------------------------------------------------------------------

    def _handle_cv_image(self, image: np.ndarray, stamp: Time):
        """Process a BGR image through the detector and update stats/logs."""
        now_wall = time.time()
        if self.last_callback_time is not None:
            dt = now_wall - self.last_callback_time
            self.callback_times.append(dt)
            if len(self.callback_times) > 30:
                self.callback_times.pop(0)
        self.last_callback_time = now_wall

        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        observation = self.detector.get_observation(
            image, observation_time=stamp_sec)

        self.frame_count += 1

        if observation['detected']:
            self.detection_count += 1
            ball_state = BallState(
                position=np.array(observation['position']),
                velocity=np.array(observation['velocity']),
                stamp=stamp, valid=True,
                raw_measurement=observation.get('raw_measurement'))
            self._record_base_trajectory(stamp_sec, ball_state)
        else:
            ball_state = BallState(
                position=np.zeros(3), velocity=np.zeros(3),
                stamp=stamp, valid=False,
                raw_measurement=observation.get('raw_measurement'))

        if self.frame_count % self.print_every == 0:
            self._log_detection_results(ball_state)

        # Publish ball state debug topic (reflects vision detection cadence).
        self._publish_ball_state_debug(ball_state)

        # Only cache the latest BallState here. The 200 Hz control timer
        # consumes it and drives publishing — publish cadence is decoupled
        # from the image callback.
        with self._latest_ball_lock:
            self._latest_ball_state = ball_state

    # ------------------------------------------------------------------
    # Logging / control helpers
    # ------------------------------------------------------------------

    def _record_base_trajectory(self, stamp_sec, ball_state: BallState):
        """Record valid ball states only when detector output is base_link."""
        output_frame = (
            getattr(self.detector, 'output_frame', 'camera')
            if self.detector is not None else 'camera')
        if output_frame != 'base_link':
            if not self._warned_non_base_trajectory:
                self.get_logger().warn(
                    'Trajectory plot only records base_link output; '
                    f'current output is {output_frame}')
                self._warned_non_base_trajectory = True
            return

        with self.trajectory_lock:
            self.trajectory_log.append({
                'time': float(stamp_sec),
                'position': ball_state.position.astype(float).tolist(),
                'velocity': ball_state.velocity.astype(float).tolist(),
                'frame': output_frame,
            })

    def save_trajectory_plot(self):
        """Save base_link position and velocity curves collected during runtime."""
        if not self.save_trajectory_plot_enabled:
            return

        with self.trajectory_lock:
            trajectory = list(self.trajectory_log)

        if not trajectory:
            self.get_logger().info(
                'No base_link trajectory samples recorded; skipping trajectory plot')
            return

        times = np.array([sample['time'] for sample in trajectory], dtype=float)
        times = times - times[0]
        positions = np.array(
            [sample['position'] for sample in trajectory], dtype=float)
        velocities = np.array(
            [sample['velocity'] for sample in trajectory], dtype=float)

        output_dir = os.path.dirname(self.trajectory_plot_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
        fig.suptitle('Ball State in base_link Frame')

        labels = ('x', 'y', 'z')
        for axis, label in enumerate(labels):
            axes[axis, 0].plot(
                times, positions[:, axis], label=f'{label}_base')
            axes[axis, 0].set_ylabel(f'{label}_base (mm)')
            axes[axis, 0].grid(True, alpha=0.3)
            axes[axis, 0].legend(loc='best')

            axes[axis, 1].plot(
                times, velocities[:, axis], label=f'v{label}_base')
            axes[axis, 1].set_ylabel(f'v{label}_base (mm/s)')
            axes[axis, 1].grid(True, alpha=0.3)
            axes[axis, 1].legend(loc='best')

        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[0, 0].set_title('Position')
        axes[0, 1].set_title('Velocity')

        fig.tight_layout()
        fig.savefig(self.trajectory_plot_path, dpi=150)
        plt.close(fig)
        self.get_logger().info(
            f'Saved base_link trajectory plot to: {self.trajectory_plot_path}')

    def _resolve_rl_trace_csv_path(self):
        requested = str(self.rl_trace_csv_path_param or '').strip()
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if requested.lower() in ('', 'auto'):
            return os.path.join(
                self.source_module_dir, 'outputs', 'rl_2real',
                f'rl_trace_{timestamp}.csv')
        requested = requested.replace('{timestamp}', timestamp)
        return os.path.expandvars(os.path.expanduser(requested))

    @staticmethod
    def _joint_key(joint_name: str) -> str:
        return joint_name.replace('/', '_')

    def _rl_trace_fieldnames(self):
        fields = [
            'control_tick',
            'wall_time_sec',
            'ros_time_sec',
            'backend',
            'ball_valid',
            'ball_age_s',
            'ball_x_m',
            'ball_y_m',
            'ball_z_m',
            'ball_vx_m_s',
            'ball_vy_m_s',
            'ball_vz_m_s',
            'joint_state_available',
        ]
        prefixes = (
            'policy_action',
            'policy_qdd_rad_s2',
            'policy_qdd_deg_s2',
            'policy_qvel_rad_s',
            'policy_qvel_deg_s',
            'policy_q_rad',
            'policy_q_deg',
            'raw_cmd_q_rad',
            'raw_cmd_q_deg',
            'safe_q_rad',
            'safe_q_deg',
            'safe_qvel_rad_s',
            'safe_qvel_deg_s',
            'safe_qdd_rad_s2',
            'safe_qdd_deg_s2',
            'real_q_rad',
            'real_q_deg',
            'real_qvel_rad_s',
            'real_qvel_deg_s',
            'real_qdd_rad_s2',
            'real_qdd_deg_s2',
        )
        for prefix in prefixes:
            for joint in RIGHT_ARM_JOINTS:
                fields.append(f'{prefix}/{self._joint_key(joint)}')
        return fields

    def _init_rl_trace_logger(self) -> None:
        if not self.record_rl_trace:
            return
        path = self._resolve_rl_trace_csv_path()
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            self._rl_trace_file = open(path, 'w', newline='')
            self._rl_trace_writer = csv.DictWriter(
                self._rl_trace_file, fieldnames=self._rl_trace_fieldnames())
            self._rl_trace_writer.writeheader()
            self._rl_trace_path = path
            self.get_logger().info(
                f'RL trace recording enabled: {path} '
                f'(sample_every={self.rl_trace_sample_every}, '
                f'flush_every={self.rl_trace_flush_every})')
        except Exception as exc:
            self.get_logger().error(
                f'Failed to open RL trace CSV "{path}": {exc}; '
                'disabling record_rl_trace')
            self.record_rl_trace = False
            self._rl_trace_file = None
            self._rl_trace_writer = None

    def close_rl_trace(self) -> None:
        if self._rl_trace_file is None:
            return
        try:
            self._rl_trace_file.flush()
            self._rl_trace_file.close()
            self.get_logger().info(f'Closed RL trace CSV: {self._rl_trace_path}')
        except Exception as exc:
            self.get_logger().warn(f'Failed to close RL trace CSV: {exc}')
        finally:
            self._rl_trace_file = None
            self._rl_trace_writer = None

    @staticmethod
    def _finite_vec_or_nan(value, n: int = 7) -> np.ndarray:
        if value is None:
            return np.full(n, np.nan, dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape != (n,) or not np.all(np.isfinite(arr)):
            return np.full(n, np.nan, dtype=np.float64)
        return arr

    def _add_trace_vec(
            self, row: dict, prefix: str, values, *,
            deg_prefix: Optional[str] = None,
            deg_factor: float = 1.0) -> None:
        arr = self._finite_vec_or_nan(values, len(RIGHT_ARM_JOINTS))
        for i, joint in enumerate(RIGHT_ARM_JOINTS):
            key = self._joint_key(joint)
            row[f'{prefix}/{key}'] = float(arr[i])
            if deg_prefix is not None:
                row[f'{deg_prefix}/{key}'] = float(np.rad2deg(arr[i]) * deg_factor)

    def _record_rl_trace(self, ball_state: BallState, raw_cmd, safe_cmd) -> None:
        if not self.record_rl_trace or self._rl_trace_writer is None:
            return
        if (self._control_tick_index % self.rl_trace_sample_every) != 0:
            return
        try:
            policy_state = (
                self.rl_controller.latest_command_state()
                if self.rl_controller is not None else {}
            )
            real_q, real_dq = self._get_right_arm_feedback()
            with self._joint_state_lock:
                real_ddq = (
                    self._estimated_right_arm_ddq.copy()
                    if self._joint_state_received else None
                )
            safe_q = self._finite_vec_or_nan(
                safe_cmd if safe_cmd is not None else self._last_safe_joint_positions)
            ball_pos_m = np.asarray(ball_state.position, dtype=np.float64) * 1e-3
            ball_vel_m_s = np.asarray(ball_state.velocity, dtype=np.float64) * 1e-3
            now_msg = self.get_clock().now().to_msg()
            ros_time_sec = float(now_msg.sec) + float(now_msg.nanosec) * 1e-9

            row = {
                'control_tick': int(self._control_tick_index),
                'wall_time_sec': float(time.time()),
                'ros_time_sec': ros_time_sec,
                'backend': self.rl_policy_backend,
                'ball_valid': int(bool(ball_state.valid)),
                'ball_age_s': float(self._last_rl_ball_age_s),
                'ball_x_m': float(ball_pos_m[0]),
                'ball_y_m': float(ball_pos_m[1]),
                'ball_z_m': float(ball_pos_m[2]),
                'ball_vx_m_s': float(ball_vel_m_s[0]),
                'ball_vy_m_s': float(ball_vel_m_s[1]),
                'ball_vz_m_s': float(ball_vel_m_s[2]),
                'joint_state_available': int(real_q is not None),
            }

            self._add_trace_vec(row, 'policy_action', policy_state.get('action'))
            self._add_trace_vec(
                row, 'policy_qdd_rad_s2', policy_state.get('acceleration'),
                deg_prefix='policy_qdd_deg_s2')
            self._add_trace_vec(
                row, 'policy_qvel_rad_s', policy_state.get('velocity'),
                deg_prefix='policy_qvel_deg_s')
            self._add_trace_vec(
                row, 'policy_q_rad', policy_state.get('position'),
                deg_prefix='policy_q_deg')
            self._add_trace_vec(
                row, 'raw_cmd_q_rad', raw_cmd, deg_prefix='raw_cmd_q_deg')
            self._add_trace_vec(
                row, 'safe_q_rad', safe_q, deg_prefix='safe_q_deg')
            self._add_trace_vec(
                row, 'safe_qvel_rad_s',
                getattr(self.safety_limiter, 'last_vel', None),
                deg_prefix='safe_qvel_deg_s')
            self._add_trace_vec(
                row, 'safe_qdd_rad_s2',
                getattr(self.safety_limiter, 'last_acc', None),
                deg_prefix='safe_qdd_deg_s2')
            self._add_trace_vec(
                row, 'real_q_rad', real_q, deg_prefix='real_q_deg')
            self._add_trace_vec(
                row, 'real_qvel_rad_s', real_dq,
                deg_prefix='real_qvel_deg_s')
            self._add_trace_vec(
                row, 'real_qdd_rad_s2', real_ddq,
                deg_prefix='real_qdd_deg_s2')

            self._rl_trace_writer.writerow(row)
            self._rl_trace_rows_since_flush += 1
            if self._rl_trace_rows_since_flush >= self.rl_trace_flush_every:
                self._rl_trace_file.flush()
                self._rl_trace_rows_since_flush = 0
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to write RL trace row: {exc}',
                throttle_duration_sec=1.0)

    def _log_detection_results(self, ball_state: BallState):
        """Log detection results with statistics."""
        if len(self.callback_times) > 0:
            avg_dt = sum(self.callback_times) / len(self.callback_times)
            callback_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            callback_hz = 0.0

        detection_rate = (100.0 * self.detection_count / self.frame_count
                          if self.frame_count > 0 else 0.0)

        if ball_state.valid:
            pos = ball_state.position
            vel = ball_state.velocity
            raw = ball_state.raw_measurement or {}
            output_frame = (
                getattr(self.detector, 'output_frame', 'camera')
                if self.detector is not None else 'camera')
            camera_pos_text = ''
            if output_frame != 'camera' and all(
                    key in raw for key in ('X_mm', 'Y_mm', 'Z_mm')):
                camera_pos_text = (
                    f'Pos[camera]: '
                    f'({raw["X_mm"]:.1f}, {raw["Y_mm"]:.1f}, '
                    f'{raw["Z_mm"]:.1f}) mm | ')
            self.get_logger().info(
                f'Frame {self.frame_count} | '
                f'Callback: {callback_hz:.1f} Hz | '
                f'Detection: {detection_rate:.1f}% | '
                f'Pixel: ({raw.get("image_x", 0.0):.1f}, '
                f'{raw.get("image_y", 0.0):.1f}) | '
                f'Radius: {raw.get("radius_px", 0.0):.1f} px | '
                f'Depth: {raw.get("depth_mm", 0.0):.1f} mm | '
                f'Method: {raw.get("detection_method", "?")} | '
                f'{camera_pos_text}'
                f'Pos[{output_frame}]: '
                f'({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm | '
                f'Vel[{output_frame}]: '
                f'({vel[0]:.0f}, {vel[1]:.0f}, {vel[2]:.0f}) mm/s')
        else:
            self.get_logger().info(
                f'Frame {self.frame_count} | '
                f'Callback: {callback_hz:.1f} Hz | '
                f'Detection: NO ({detection_rate:.1f}%)')

    def _control_timer_callback(self):
        """Fixed-rate (control_rate_hz) control loop.

        Phase 1 (init pose, if enabled): publish whole-body TARGET_DEGREES as
        individual MechUnitCmd messages on the shared command_topic. The
        right-arm entry passes through the safety limiter; the other units
        (left arm, leg+waist, head) are sent directly.

        Phase 2 (RL control): every tick we call compute_action() regardless
        of detection validity so the RL policy's internal hold-last / dropout
        logic matches training. Raw RL output goes through the safety limiter
        before being published.
        """
        now_wall = time.time()
        self._control_tick_index += 1
        with self._latest_ball_lock:
            ball_state = self._latest_ball_state

        if ball_state is None:
            ball_state = BallState(
                position=np.zeros(3), velocity=np.zeros(3),
                stamp=self.get_clock().now().to_msg(), valid=False)

        if self._init_pose_active:
            if self._init_pose_start_wall is None:
                self._init_pose_start_wall = now_wall
            elapsed = now_wall - self._init_pose_start_wall
            # Once real /joint_states arrives, re-anchor the safety limiter
            # to the current right-arm position so the first post-init
            # command does not jump. Called at most once.
            self._maybe_sync_safety_limiter_to_joint_state()
            # During init, right-arm target is the RL initial posture (the
            # same thing the limiter was seeded with). This keeps the arm
            # still while the other units reach TARGET.
            raw_cmd = (
                self.rl_controller.current_arm_cmd_q().astype(np.float64)
                if self.rl_controller is not None
                else np.asarray(target_right_arm_q_rad(), dtype=np.float64)
            )
            self._last_raw_joint_positions = raw_cmd.copy()
            # Publish all four units (right arm goes through safety limiter).
            self._publish_whole_body_init_pose(raw_cmd, ball_state.stamp)
            time_up = elapsed >= self.init_pose_duration_s
            pose_reached = self._init_pose_reached()
            if time_up or pose_reached:
                self._init_pose_active = False
                reason = 'pose_reached' if pose_reached else 'duration_elapsed'
                self.get_logger().info(
                    f'Init-pose phase done after {elapsed:.2f}s '
                    f'({reason}); switching to RL control')
            return

        # Phase 2: always run RL, even on invalid detections. The RL
        # policy's hold-last/dropout logic (see RLPolicyController.predict)
        # reproduces the training-time short-dropout behavior.
        # If init pose was disabled entirely, still try a one-shot sync of
        # the safety limiter to the first real /joint_states so the first
        # RL tick does not jump.
        self._maybe_sync_safety_limiter_to_joint_state()
        raw_cmd = self.compute_action(ball_state)
        self._last_raw_joint_positions = np.asarray(
            raw_cmd, dtype=np.float64).reshape(-1).copy()
        # Publish RL joint command debug (reflects 200 Hz control cadence).
        self._publish_rl_joint_cmd_debug()
        safe_cmd = self._publish_right_arm_command(
            self._last_raw_joint_positions, ball_state.stamp)
        self._record_rl_trace(ball_state, self._last_raw_joint_positions, safe_cmd)

    def _joint_state_callback(self, msg: JointState):
        """Cache the latest JointState; values are looked up by name.

        The topic carries Base-*, LegWaist-*, LeftArm-*, RightArm-* (and maybe
        Head-*) in a single frame. We store everything by joint name and let
        consumers pick out the subset they care about (right-arm feedback,
        init-pose tolerance check, etc.). NaN/Inf values are dropped here so
        downstream code can trust whatever is in the dicts. We also maintain
        a finite-difference estimate of the right-arm dq because on this
        hardware `msg.velocity` is often empty or stuck at zero.
        """
        stamp_sec = (float(msg.header.stamp.sec)
                     + float(msg.header.stamp.nanosec) * 1e-9)
        positions = msg.position
        velocities = msg.velocity
        with self._joint_state_lock:
            for i, name in enumerate(msg.name):
                if i < len(positions):
                    val = float(positions[i])
                    if np.isfinite(val):
                        self._latest_joint_positions[name] = val
                if i < len(velocities):
                    val = float(velocities[i])
                    if np.isfinite(val):
                        self._latest_joint_velocities[name] = val
            self._latest_joint_state_stamp_sec = stamp_sec
            self._joint_state_received = True

            # Finite-difference dq for the right arm, clipped to hardware
            # velocity limits so a single-frame jump cannot blow up the obs.
            try:
                cur_q = np.array(
                    [self._latest_joint_positions[n]
                     for n in RIGHT_ARM_JOINTS],
                    dtype=np.float32)
            except KeyError:
                cur_q = None

            if cur_q is not None and np.all(np.isfinite(cur_q)):
                if (self._prev_right_arm_q_for_dq is not None
                        and self._prev_right_arm_stamp_for_dq is not None):
                    dt = stamp_sec - self._prev_right_arm_stamp_for_dq
                    if dt > 1e-4 and np.isfinite(dt):
                        dq_est = (
                            (cur_q - self._prev_right_arm_q_for_dq)
                            / max(dt, 1e-6)
                        ).astype(np.float32)
                        dq_est = np.clip(
                            dq_est,
                            -self._right_arm_dq_clip_rad_s,
                            self._right_arm_dq_clip_rad_s)
                        self._estimated_right_arm_dq = dq_est
                        if (self._prev_right_arm_dq_for_ddq is not None
                                and self._prev_right_arm_stamp_for_ddq is not None):
                            ddq_dt = stamp_sec - self._prev_right_arm_stamp_for_ddq
                            if ddq_dt > 1e-4 and np.isfinite(ddq_dt):
                                ddq_est = (
                                    (dq_est - self._prev_right_arm_dq_for_ddq)
                                    / max(ddq_dt, 1e-6)
                                ).astype(np.float32)
                                ddq_est = np.clip(
                                    ddq_est,
                                    -self._right_arm_ddq_clip_rad_s2,
                                    self._right_arm_ddq_clip_rad_s2)
                                self._estimated_right_arm_ddq = ddq_est
                        self._prev_right_arm_dq_for_ddq = dq_est.copy()
                        self._prev_right_arm_stamp_for_ddq = stamp_sec
                self._prev_right_arm_q_for_dq = cur_q.copy()
                self._prev_right_arm_stamp_for_dq = stamp_sec

    def _get_right_arm_feedback(self):
        """Return (q, dq) as float32 arrays of shape (7,) or (None, None).

        dq prefers `msg.velocity` when it looks usable (non-empty and not
        all-zero / near-zero). Otherwise it falls back to the
        finite-difference estimate computed from position history.
        """
        with self._joint_state_lock:
            if not self._joint_state_received:
                return None, None
            try:
                q = np.array(
                    [self._latest_joint_positions[n] for n in RIGHT_ARM_JOINTS],
                    dtype=np.float32)
            except KeyError:
                return None, None
            msg_dq_vals = [
                self._latest_joint_velocities.get(n, np.nan)
                for n in RIGHT_ARM_JOINTS
            ]
            dq_from_msg = np.array(msg_dq_vals, dtype=np.float32)
            dq_est = self._estimated_right_arm_dq.copy()

        if q.shape != (7,) or not np.all(np.isfinite(q)):
            return None, None

        # Decide which dq to trust.
        velocity_is_useful = (
            dq_from_msg.shape == (7,)
            and np.all(np.isfinite(dq_from_msg))
            and float(np.linalg.norm(dq_from_msg)) > 1e-4
        )
        if velocity_is_useful:
            dq = dq_from_msg
        elif np.all(np.isfinite(dq_est)):
            dq = dq_est
        else:
            dq = None
        return q, dq

    def _init_pose_reached(self) -> bool:
        """True when every non-base joint is within tolerance of TARGET.

        We compare JointState positions against TARGET_DEGREES for every
        joint we have a target for — left arm, right arm, leg+waist, head.
        Base-0..Base-3 are ignored. Missing joints fall back to returning
        False so we keep publishing init pose until the full body catches up.
        """
        tol = float(
            self.get_parameter('init_pose_tolerance_rad').value)
        with self._joint_state_lock:
            if not self._joint_state_received:
                return False
            snapshot = dict(self._latest_joint_positions)
        for name, deg in TARGET_DEGREES.items():
            if name not in snapshot:
                return False
            target_rad = float(np.deg2rad(deg))
            if abs(snapshot[name] - target_rad) > tol:
                return False
        return True

    def _maybe_sync_safety_limiter_to_joint_state(self) -> None:
        """Re-anchor the safety limiter once the real right-arm state arrives.

        Call this while still in init-pose phase so the first post-init
        command does not jump from the seeded initial posture to a real
        joint position that may differ by several degrees.
        """
        if self._safety_synced_to_joint_state:
            return
        q, _ = self._get_right_arm_feedback()
        if q is None:
            return
        try:
            self.safety_limiter.reset(q.astype(np.float64))
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to sync safety limiter to /joint_states: {exc}')
            return
        self._last_safe_joint_positions = \
            self.safety_limiter.last_cmd.copy()
        self._last_raw_joint_positions = \
            self._last_safe_joint_positions.copy()
        self._safety_synced_to_joint_state = True
        self.get_logger().info(
            'Safety limiter anchored to real right-arm /joint_states '
            f'(deg={np.rad2deg(q).round(2).tolist()})')

    def compute_action(self, ball_state: BallState) -> np.ndarray:
        """Compute right-arm joint targets (rad) using the RL policy.

        Always invoked from the control timer, including when the detection
        is invalid — RLPolicyController.predict() holds the last valid ball
        observation internally and reports the age, matching the training
        env's dropout/hold-last behavior. The safety limiter, not this
        function, is responsible for clamping the output before publish.
        """
        if not self.enable_rl_policy or self.rl_controller is None:
            return self._last_raw_joint_positions.astype(np.float64)

        now_wall = time.time()
        if ball_state.valid:
            age_s = 0.0
            self._last_valid_ball_wall_time = now_wall
        elif self._last_valid_ball_wall_time is not None:
            age_s = max(0.0, now_wall - self._last_valid_ball_wall_time)
        else:
            age_s = self.rl_ball_obs_age_clip
        self._last_rl_ball_age_s = float(age_s)

        # BallState carries mm / mm/s; the RL policy expects m / m/s.
        ball_pos_m = np.asarray(
            ball_state.position, dtype=np.float32) * 1e-3
        ball_vel_m_s = np.asarray(
            ball_state.velocity, dtype=np.float32) * 1e-3

        # Real right-arm feedback for the RL observation. If it's missing,
        # predict() falls back to arm_cmd_q/arm_cmd_qvel internally, which
        # matches the pre-feedback behavior.
        right_arm_q, right_arm_dq = self._get_right_arm_feedback()
        if right_arm_q is None:
            if not self._joint_state_log_warned_missing:
                self.get_logger().warn(
                    'No right-arm /joint_states yet; RL obs uses command '
                    'estimate for q/dq until feedback arrives',
                    throttle_duration_sec=5.0)
                self._joint_state_log_warned_missing = True

        try:
            joint_cmd = self.rl_controller.predict(
                ball_pos_m=ball_pos_m,
                ball_vel_m_s=ball_vel_m_s,
                ball_valid=bool(ball_state.valid),
                ball_obs_age_s=float(age_s),
                dt=self.control_dt,
                arm_q=right_arm_q,
                arm_dq=right_arm_dq,
            )
        except Exception as exc:
            self.get_logger().error(
                f'RL policy predict failed: {exc}; '
                'returning last raw command',
                throttle_duration_sec=1.0)
            return self._last_raw_joint_positions.astype(np.float64)

        joint_cmd = np.asarray(joint_cmd, dtype=np.float64).reshape(-1)
        if (joint_cmd.shape[0] != self._rl_expected_joint_count
                or not np.all(np.isfinite(joint_cmd))):
            self.get_logger().error(
                'RL policy returned invalid joint command; '
                'returning last raw command',
                throttle_duration_sec=1.0)
            return self._last_raw_joint_positions.astype(np.float64)

        return joint_cmd

    def _publish_right_arm_command(
            self, joint_positions: np.ndarray, stamp: Time):
        """Safety-limit + publish a right-arm MechUnitCmd on command_topic.

        This is the ONLY path that may send right-arm joint commands: every
        raw right-arm target (RL or init-pose) must come through here so it
        passes `RightArmCommandSafetyLimiter` first.
        """
        del stamp  # MechUnitCmd has no header; kept for caller compatibility.
        if len(self.joint_names) != self.safety_limiter.N_JOINTS:
            self.get_logger().error(
                'joint_names length does not match safety limiter '
                f'({len(self.joint_names)} vs '
                f'{self.safety_limiter.N_JOINTS})',
                throttle_duration_sec=1.0)
            return None

        raw = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        safe_cmd = self.safety_limiter.filter(raw)
        self._last_safe_joint_positions = safe_cmd.copy()
        self._maybe_log_safety_counts()

        msg = MechUnitCmd()
        msg.mu_idx = self.right_arm_mu_idx
        msg.jnt_pos = safe_cmd.tolist()
        msg.use_jnt = True
        msg.psi = 0.0
        # msg.end_pose left at the default identity Pose; joint control only.
        self.cmd_pub.publish(msg)
        return safe_cmd.copy()

    # Backward-compatible alias: the old internal name is kept in case
    # external code/tests still reference it.
    _publish_joint_command = _publish_right_arm_command

    def _publish_unit_cmd(self, mu_idx: int, joint_positions_rad) -> None:
        """Publish a non-right-arm MechUnitCmd (no safety limiter needed).

        Used during init pose for left arm, leg+waist, and head. All four
        units share command_topic (default /mx_mix_command).
        """
        msg = MechUnitCmd()
        msg.mu_idx = int(mu_idx)
        msg.jnt_pos = [float(x) for x in joint_positions_rad]
        msg.use_jnt = True
        msg.psi = 0.0
        # msg.end_pose left at the default identity Pose; joint control only.
        self.cmd_pub.publish(msg)

    def _publish_whole_body_init_pose(self, right_arm_raw_q: np.ndarray,
                                      stamp: Time) -> None:
        """Drive every mechanical unit toward TARGET_DEGREES during init.

        Right arm goes through `_publish_right_arm_command` (so the safety
        limiter runs). Left arm, leg+waist, and head are sent their
        TARGET_DEGREES setpoints directly — they do not share the right-arm
        limits, and publishing the same setpoint every tick is fine because
        the downstream controller tracks at its own cadence.
        """
        self._publish_right_arm_command(right_arm_raw_q, stamp)
        self._publish_unit_cmd(self.left_arm_mu_idx, target_left_arm_q_rad())
        self._publish_unit_cmd(
            self.leg_waist_mu_idx, target_leg_waist_q_rad())
        self._publish_unit_cmd(self.head_mu_idx, target_head_q_rad())

    def _publish_ball_state_debug(self, ball_state: BallState) -> None:
        """Publish ball detection result as Odometry on /pingpong/ball_state.

        Position and velocity are in base_link frame (m, m/s). When detection
        is invalid, we still publish a frame with the last-known or zero
        values so downstream nodes can track the detection cadence.
        """
        if not self.publish_ball_state_debug:
            return
        msg = Odometry()
        msg.header.stamp = ball_state.stamp
        msg.header.frame_id = 'base_link'
        msg.child_frame_id = 'pingpong_ball'
        # BallState carries mm / mm/s; convert to m / m/s for Odometry.
        pos_m = np.asarray(ball_state.position, dtype=float) * 1e-3
        vel_m_s = np.asarray(ball_state.velocity, dtype=float) * 1e-3
        msg.pose.pose.position.x = float(pos_m[0])
        msg.pose.pose.position.y = float(pos_m[1])
        msg.pose.pose.position.z = float(pos_m[2])
        msg.twist.twist.linear.x = float(vel_m_s[0])
        msg.twist.twist.linear.y = float(vel_m_s[1])
        msg.twist.twist.linear.z = float(vel_m_s[2])
        # Optional: mark invalid detections with large covariance, but for
        # simplicity we leave covariance at default (zeros) and rely on
        # downstream to check the detection rate or compare against a
        # known-good range.
        self.ball_state_pub.publish(msg)

    def _publish_rl_joint_cmd_debug(self) -> None:
        """Publish RL policy command trajectory as JointState.

        Published on /pingpong/rl_joint_cmd_state. Fields:
        - position: commanded joint position (rad)
        - velocity: commanded joint velocity (rad/s)
        - effort: commanded joint acceleration (rad/s²) — note that 'effort'
          semantically means torque, but we repurpose it here for debug to
          avoid defining a custom message type.

        Only publishes when RL policy is active. During init pose or when RL
        is disabled, this topic is silent.
        """
        if not self.publish_rl_joint_cmd_debug:
            return
        if self.rl_controller is None:
            return
        state = self.rl_controller.latest_command_state()
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(RIGHT_ARM_JOINTS)
        msg.position = state["position"].astype(float).tolist()
        msg.velocity = state["velocity"].astype(float).tolist()
        # effort field is commanded acceleration debug, rad/s².
        msg.effort = state["acceleration"].astype(float).tolist()
        self.rl_joint_cmd_state_pub.publish(msg)

    def _maybe_log_safety_counts(self):
        """Throttled log of safety limiter activity (once per second)."""
        now_wall = time.time()
        if (now_wall - self._safety_log_last_wall) < self._safety_log_period_s:
            return
        counts = self.safety_limiter.consume_clip_counts()
        self._safety_log_last_wall = now_wall
        total = counts['invalid'] + counts['pos'] + counts['vel'] + counts['acc']
        if total == 0:
            return
        self.get_logger().warn(
            f'Safety limiter clipped last 1s: '
            f"invalid={counts['invalid']} pos={counts['pos']} "
            f"vel={counts['vel']} acc={counts['acc']}")


def main(args=None):
    """Run the ping-pong controller node."""
    rclpy.init(args=args)
    node = None
    executor = None

    try:
        node = PingPongControllerNode()
        # Multi-threaded executor so the 200 Hz control timer is not blocked
        # by the heavy YOLO image callback. Control, vision, and joint-state
        # callbacks are on separate callback groups, so at least one thread
        # is always free for control.
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'Unhandled exception: {e}')
        raise
    finally:
        if node and node.frame_count > 0:
            detection_rate = 100.0 * node.detection_count / node.frame_count
            node.get_logger().info('=' * 60)
            node.get_logger().info('Detection Statistics')
            node.get_logger().info('=' * 60)
            node.get_logger().info(f'Total frames: {node.frame_count}')
            node.get_logger().info(f'Detected frames: {node.detection_count}')
            node.get_logger().info(f'Detection rate: {detection_rate:.2f}%')
            node.get_logger().info('=' * 60)

        if node:
            node._shutdown_realsense()
            node.save_trajectory_plot()
            node.close_rl_trace()
            if executor is not None:
                executor.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
