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
from dataclasses import dataclass
from typing import Optional
import time
import os
import threading
from queue import Queue, Full, Empty
import glob
import multiprocessing as mp
import json
import csv

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
from mc_core_interface.msg import MechUnitCmd, MechUnitCmdArray
from nav_msgs.msg import Odometry

from pingpong_controller.ball_detector import PingPongBallDetector
from pingpong_controller.calibration_loader import load_calibration
from pingpong_controller.viz_render import render_detection_overlay
from pingpong_controller.fov_loss_compensator import (
    FovLossCompensator,
    FovLossCompensatorConfig,
    gravity_for_base_link,
    gravity_for_camera_frame,
)
from pingpong_controller.mjx_rl_policy import MjxRLPolicyController
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
                workspace_root, 'src', 'pingpong_moz1',
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
        self.declare_parameter('save_trajectory_csv', True)
        self.declare_parameter('save_trajectory_json', True)

        # Live visualization window (off by default; turn on for debugging
        # with `enable_visualization:=true`). A dedicated worker thread does
        # the drawing and ships frames to a GUI subprocess, fed by a
        # single-slot drop-old handoff from the vision callback, so it never
        # steals time from the 200 Hz control loop or the YOLO image callback.
        self.declare_parameter('enable_visualization', False)
        self.declare_parameter('visualization_window_name', 'PingPong Detection')

        # Raw video recording: save the unprocessed RealSense RGB stream at its
        # native rate (e.g. 848x480@60Hz) on a dedicated thread, completely
        # independent of the visualization pipeline. No overlay, no undistort,
        # no detection — just the raw frames straight off the camera. Run the
        # offline tool tools/annotate_raw_video.py afterwards to produce the
        # detection-annotated video. raw_video_path blank => timestamped file
        # under outputs/videos/. raw_video_fps<=0 => use realsense_fps.
        self.declare_parameter('save_raw_video', False)
        self.declare_parameter('raw_video_path', '')
        self.declare_parameter('raw_video_fps', 0.0)

        # Debug/monitoring topics under /pingpong namespace.
        self.declare_parameter('ball_state_topic', '/pingpong/ball_state')
        self.declare_parameter('publish_ball_state_debug', True)
        self.declare_parameter(
            'rl_joint_cmd_state_topic', '/pingpong/rl_joint_cmd_state')
        self.declare_parameter('publish_rl_joint_cmd_debug', True)

        # RL policy parameters
        default_rl_model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models', 'best_model.zip')
        self.declare_parameter('rl_model_path', default_rl_model_path)
        self.declare_parameter('enable_rl_policy', True)
        # Policy backend: 'sb3' loads a stable-baselines3 PPO .zip via
        # RLPolicyController; 'mjx' loads an MJX/JAX curriculum .pkl via
        # MjxRLPolicyController. The MJX adapter runs the 2-layer tanh MLP
        # in pure NumPy (no JAX at runtime) and auto-reads env_cfg from the
        # checkpoint to align action_acc_scale, ball_obs_age_clip, arm
        # limits, and arm_action_limiter with training.
        self.declare_parameter('policy_backend', 'sb3')
        # 200 Hz control loop -> dt = 0.005 s.
        self.declare_parameter('rl_policy_dt', 0.005)
        self.declare_parameter('control_rate_hz', 200.0)
        self.declare_parameter('rl_policy_device', 'cpu')
        self.declare_parameter('rl_ball_obs_age_clip', 0.20)

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
        default_trajectory_csv_path = os.path.join(
            self.source_module_dir, 'outputs', 'pingpong_base_trajectory.csv')
        default_trajectory_json_path = os.path.join(
            self.source_module_dir, 'outputs', 'pingpong_base_trajectory.json')
        default_safe_init_pose_csv = os.path.join(
            self.source_module_dir, 'outputs', 'rl_sim',
            'stage4g_action_trace.csv')

        # Ball-lost failsafe: after the ball has gone undetected for longer
        # than the timeout, stop running the RL policy and smoothly drive the
        # right arm back to a fixed safe init posture (the right-arm angles in
        # frame 0 of the recorded action trace below). The safety limiter
        # ramps toward that posture under the normal vel/acc limits, so the
        # move stays smooth. RL resumes as soon as the ball is reacquired.
        self.declare_parameter('enable_ball_lost_failsafe', True)
        self.declare_parameter('ball_lost_failsafe_timeout_s', 1.0)
        self.declare_parameter(
            'safe_init_pose_csv_path', default_safe_init_pose_csv)

        self.declare_parameter('yolo_model_path', default_model_path)
        self.declare_parameter(
            'trajectory_plot_path', default_trajectory_plot_path)
        self.declare_parameter(
            'trajectory_csv_path', default_trajectory_csv_path)
        self.declare_parameter(
            'trajectory_json_path', default_trajectory_json_path)
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

        # FOV-loss compensator: when the ball flies above the camera's
        # field of view during a juggle, propagate a ballistic prediction
        # forward instead of letting detection drop to invalid. Default
        # OFF — controllers fall back to RL hold-last just like before.
        self.declare_parameter('enable_fov_loss_compensation', False)
        self.declare_parameter('fov_loss_max_predict_duration_s', 0.8)
        self.declare_parameter('fov_loss_miss_frames_to_predict', 2)
        self.declare_parameter('fov_loss_min_upward_velocity_mm_s', 200.0)
        self.declare_parameter('fov_loss_history_window_s', 0.15)
        self.declare_parameter('fov_loss_drag_coeff_per_s', 0.0)
        self.declare_parameter('fov_loss_gate_max_mahalanobis', 4.0)
        self.declare_parameter('fov_loss_log_period_s', 1.0)

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
        self.save_trajectory_csv_enabled = self.get_parameter(
            'save_trajectory_csv').value
        self.trajectory_csv_path = self.get_parameter(
            'trajectory_csv_path').value
        self.save_trajectory_json_enabled = self.get_parameter(
            'save_trajectory_json').value
        self.trajectory_json_path = self.get_parameter(
            'trajectory_json_path').value

        # Visualization toggle + window name (read-only after init; restart
        # the node to flip them).
        self.enable_visualization = bool(
            self.get_parameter('enable_visualization').value)
        self.visualization_window_name = str(
            self.get_parameter('visualization_window_name').value)

        # Raw video recording: independent of visualization. Saves the
        # unprocessed RealSense RGB frames straight off the camera.
        self.save_raw_video = bool(
            self.get_parameter('save_raw_video').value)
        self.raw_video_path = str(
            self.get_parameter('raw_video_path').value).strip()
        self.raw_video_fps = float(
            self.get_parameter('raw_video_fps').value)

        # Debug/monitoring topic parameters.
        self.ball_state_topic = self.get_parameter('ball_state_topic').value
        self.publish_ball_state_debug = bool(
            self.get_parameter('publish_ball_state_debug').value)
        self.rl_joint_cmd_state_topic = self.get_parameter(
            'rl_joint_cmd_state_topic').value
        self.publish_rl_joint_cmd_debug = bool(
            self.get_parameter('publish_rl_joint_cmd_debug').value)

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

        # FOV-loss compensator parameters (read once; the compensator
        # itself is built lazily on first detector frame so it can match
        # the detector's actual output_frame).
        self.enable_fov_loss_compensation = bool(
            self.get_parameter('enable_fov_loss_compensation').value)
        self.fov_loss_max_predict_duration_s = float(
            self.get_parameter('fov_loss_max_predict_duration_s').value)
        self.fov_loss_miss_frames_to_predict = int(
            self.get_parameter('fov_loss_miss_frames_to_predict').value)
        self.fov_loss_min_upward_velocity_mm_s = float(
            self.get_parameter('fov_loss_min_upward_velocity_mm_s').value)
        self.fov_loss_history_window_s = float(
            self.get_parameter('fov_loss_history_window_s').value)
        self.fov_loss_drag_coeff_per_s = float(
            self.get_parameter('fov_loss_drag_coeff_per_s').value)
        self.fov_loss_gate_max_mahalanobis = float(
            self.get_parameter('fov_loss_gate_max_mahalanobis').value)
        self.fov_loss_log_period_s = float(
            self.get_parameter('fov_loss_log_period_s').value)
        self._fov_loss_compensator: Optional[FovLossCompensator] = None
        self._fov_loss_log_last_wall = 0.0
        self._fov_loss_last_predicts_emitted = 0
        self._fov_loss_last_state: Optional[str] = None

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
        self.policy_backend = str(
            self.get_parameter('policy_backend').value).lower()
        if self.policy_backend not in ('sb3', 'mjx'):
            self.get_logger().error(
                f'Invalid policy_backend={self.policy_backend}; must be '
                '"sb3" or "mjx". Falling back to sb3.')
            self.policy_backend = 'sb3'
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

        # Ball-lost failsafe parameters.
        self.enable_ball_lost_failsafe = bool(
            self.get_parameter('enable_ball_lost_failsafe').value)
        self.ball_lost_failsafe_timeout_s = float(
            self.get_parameter('ball_lost_failsafe_timeout_s').value)
        self.safe_init_pose_csv_path = self.get_parameter(
            'safe_init_pose_csv_path').value
        # Resolved length-7 safe init posture (rad); built after RL setup so we
        # can fall back to the policy's training initial posture if the CSV is
        # unreadable. _failsafe_active tracks whether we are currently parked.
        self.safe_init_pose_q = None
        self._failsafe_active = False

        # Robot XML for FK (default to models/moz1_pd.xml in same directory).
        default_robot_xml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models', 'moz1_pd.xml')
        self.declare_parameter('robot_xml_path', default_robot_xml)
        self.robot_xml_path = self.get_parameter('robot_xml_path').value

        self.rl_controller = None
        self._last_valid_ball_wall_time = None
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
                    if self.policy_backend == 'mjx':
                        self.get_logger().info(
                            f'Loading MJX policy from: {self.rl_model_path}')
                        self.rl_controller = MjxRLPolicyController(
                            model_path=self.rl_model_path,
                            dt=self.control_dt,
                            ball_obs_age_clip_s=self.rl_ball_obs_age_clip,
                            device=self.rl_policy_device,
                            robot_xml_path=self.robot_xml_path,
                            logger=self.get_logger(),
                            align_with_checkpoint=True,
                        )
                    else:  # sb3
                        self.get_logger().info(
                            f'Loading SB3 policy from: {self.rl_model_path}')
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
                        f'RLPolicyController ready '
                        f'(backend={self.policy_backend}, '
                        f'model={self.rl_model_path})')
                except Exception as exc:
                    self.get_logger().error(
                        f'Failed to initialize RLPolicyController: {exc}; '
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

        # Resolve the ball-lost failsafe target posture now that the RL
        # initial posture (used as fallback) is available.
        self.safe_init_pose_q = self._load_safe_init_pose()
        if self.enable_ball_lost_failsafe:
            self.get_logger().info(
                'Ball-lost failsafe enabled: after '
                f'{self.ball_lost_failsafe_timeout_s:.2f}s without a valid '
                'detection, RL is paused and the right arm ramps to safe pose '
                f'(deg={np.rad2deg(self.safe_init_pose_q).round(2).tolist()})')
        else:
            self.get_logger().info(
                'Ball-lost failsafe disabled (enable_ball_lost_failsafe=False)')

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
            f'MechUnitCmdArray -> "{command_topic}" '
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
        # Hard cap for the finite-difference dq so a single noisy JointState
        # frame cannot feed unbounded dq into the RL obs.
        # Limits match RightArmCommandSafetyLimiter.VEL_LIMIT_DEG_S.
        self._right_arm_dq_clip_rad_s = np.deg2rad(np.array(
            [210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0],
            dtype=np.float32))

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
        # MechUnitCmdArray publisher on command_topic (default
        # /mx_mix_command). The real downstream (MCControlNode) subscribes to
        # MechUnitCmdArray — a header plus a list of per-unit MechUnitCmd — so
        # a bare MechUnitCmd is never received. We wrap every command (RL phase:
        # right arm only; init phase: all four mechanical units) in an array.
        # This mirrors the on-hardware-verified pingpong_replay publisher.
        self.cmd_pub = self.create_publisher(
            MechUnitCmdArray, command_topic, 10)

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

        # --- Visualization state ---
        # OpenCV's Qt backend insists on running its event loop on the
        # process main thread, but our main thread sits inside
        # executor.spin() at the C level, so any cv2 GUI call from the
        # vision callback would deadlock. Workaround: spawn a tiny
        # subprocess whose only job is to imshow frames sent over a
        # multiprocessing pipe. The subprocess has its own main thread and
        # owns Qt cleanly. When the toggle is off we never spawn it, so
        # headless / no-DISPLAY runs stay free of GUI overhead.
        self._viz_proc = None
        self._viz_send = None  # multiprocessing.Connection (parent end)
        self._viz_lock = threading.Lock()
        # Disables the entire viz pipeline (worker thread + producer handoff).
        self._viz_disabled_runtime = False
        # Disables ONLY the GUI subprocess display while keeping the worker
        # thread alive — so video recording can continue even after the GUI
        # window dies or is unavailable (headless / no DISPLAY).
        self._viz_gui_disabled = False
        # Single-slot drop-old handoff between the vision callback (producer)
        # and the viz worker (consumer): the worker always renders the LATEST
        # frame and drops any it missed, so the window stays real-time even
        # when rendering/display can't keep up with the camera. Recording is
        # NOT done here anymore — the raw RealSense stream is recorded on its
        # own thread (see _enqueue_raw_video_frame), so visualization never
        # needs the lossless queue that previously caused display lag.
        self._viz_slot_lock = threading.Lock()
        self._viz_pending = None  # (raw_bgr, observation) or None
        self._viz_queue_mode = False  # always single-slot (real-time, drop old)
        self._viz_frame_queue = None
        self._viz_event = threading.Event()
        self._viz_thread = None
        self._viz_thread_running = False

        # --- Raw video recording (independent pipeline; unprocessed RGB) ---
        # Fed directly from the RealSense reader thread, bypassing detection
        # and visualization entirely, so it captures the true native-rate
        # stream regardless of how slow rendering or YOLO are.
        self._raw_video_writer = None
        self._raw_video_lock = threading.Lock()
        self._raw_video_path_resolved = None
        self._raw_video_frame_size = None
        self._raw_video_frames_written = 0
        self._raw_video_disabled_runtime = False
        self._raw_video_queue = None
        self._raw_video_writer_thread = None
        self._raw_video_writer_running = False
        self._raw_video_queue_max_size = 360  # ~6 seconds at 60fps buffer

        if self.enable_visualization:
            self.get_logger().info(
                f'Visualization enabled: window "{self.visualization_window_name}" '
                '(press q or ESC in the window to close it)')
        else:
            self.get_logger().info(
                'Visualization disabled (enable_visualization=False)')

        # Raw recording is fed from the direct-RealSense reader loop. In ROS
        # topic mode there is no such loop, so disable it with a warning.
        if self.save_raw_video and not self.use_direct_realsense:
            self.get_logger().warn(
                'save_raw_video=true but use_direct_realsense=false; raw '
                'recording is only supported in direct_realsense mode. '
                'Disabling raw video recording.')
            self.save_raw_video = False
        if self.save_raw_video:
            raw_fps_text = (
                f'{self.raw_video_fps:.1f}'
                if self.raw_video_fps > 0.0 else 'auto (realsense_fps)')
            self.get_logger().info(
                'Raw video recording enabled: saving unprocessed RealSense '
                f'RGB stream (fps={raw_fps_text}); file resolved on first '
                'frame. Post-process with tools/annotate_raw_video.py.')

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

    def _load_safe_init_pose(self) -> np.ndarray:
        """Resolve the ball-lost failsafe right-arm posture (rad, length 7).

        Reads the right-arm ``arm_cmd_q/RightArm-{0..6}`` columns from frame 0
        (first data row) of the action-trace CSV. Falls back to the RL policy's
        training initial posture if the CSV is missing, malformed, or lacks the
        expected columns, so the failsafe always has a usable target.
        """
        fallback = np.asarray(
            target_right_arm_q_rad(), dtype=np.float64).reshape(-1)
        path = self.safe_init_pose_csv_path
        if not path or not os.path.exists(path):
            self.get_logger().warn(
                f'Safe init pose CSV not found ({path}); '
                'using RL training initial posture as failsafe target')
            return fallback
        try:
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                first_row = next(reader)
            cols = [f'arm_cmd_q/{name}' for name in RIGHT_ARM_JOINTS]
            pose = np.array(
                [float(first_row[col]) for col in cols], dtype=np.float64)
        except (StopIteration, KeyError, ValueError, OSError) as exc:
            self.get_logger().warn(
                f'Failed to read safe init pose from {path}: {exc}; '
                'using RL training initial posture as failsafe target')
            return fallback
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            self.get_logger().warn(
                f'Safe init pose from {path} is invalid; '
                'using RL training initial posture as failsafe target')
            return fallback
        self.get_logger().info(
            f'Loaded safe init pose from frame 0 of: {path}')
        return pose

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

            # Record the raw frame BEFORE detection/visualization touch it, so
            # the saved video is the true native-rate RealSense stream. This
            # only enqueues (a non-blocking O(1) handoff to the raw writer
            # thread); the actual encode happens off this thread.
            if self.save_raw_video and not self._raw_video_disabled_runtime:
                self._enqueue_raw_video_frame(color_image)

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

        # Optional FOV-loss compensation. Switch is default OFF; when on,
        # this swaps the detector's invalid output for a ballistic
        # prediction during the upward phase of a juggle so the controller
        # keeps a usable ball state through the apex. The function lives
        # in fov_loss_compensator.py and is treated as a black-box wrap.
        if self.enable_fov_loss_compensation:
            observation = self._apply_fov_loss_compensation(
                observation, stamp_sec)

        self.frame_count += 1

        if observation['detected']:
            # Predicted frames are usable for control but should not be
            # counted as YOLO detections or saved into the recorded
            # trajectory — keep those stats honest. raw_measurement still
            # carries `fov_predicted=True` for downstream debugging.
            is_predicted = bool(observation.get('predicted', False))
            if not is_predicted:
                self.detection_count += 1
            ball_state = BallState(
                position=np.array(observation['position']),
                velocity=np.array(observation['velocity']),
                stamp=stamp, valid=True,
                raw_measurement=observation.get('raw_measurement'))
            if not is_predicted:
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

        # Hand the latest frame + observation to the viz worker thread (if
        # enabled). This is a cheap drop-old enqueue: it overwrites the single
        # pending slot and returns immediately, so the worker (which does the
        # remap/render/blocking pipe-send) never back-pressures this detection
        # callback. Intermediate frames are dropped, not queued.
        if self.enable_visualization:
            self._publish_visualization_frame(image, observation)

        # Only cache the latest BallState here. The 200 Hz control timer
        # consumes it and drives publishing — publish cadence is decoupled
        # from the image callback.
        with self._latest_ball_lock:
            self._latest_ball_state = ball_state

    # ------------------------------------------------------------------
    # Logging / control helpers
    # ------------------------------------------------------------------

    def _ensure_fov_loss_compensator(self) -> Optional[FovLossCompensator]:
        """Lazily build the FOV-loss compensator using detector outputs.

        We wait until the detector exists and has settled on an
        ``output_frame`` so the gravity vector matches whatever space
        position/velocity arrive in. Returns the compensator (cached), or
        ``None`` if the detector is not ready yet.
        """
        if self._fov_loss_compensator is not None:
            return self._fov_loss_compensator
        if self.detector is None:
            return None

        output_frame = getattr(self.detector, 'output_frame', 'camera')
        if output_frame == 'base_link':
            gravity = gravity_for_base_link()
        else:
            gravity = gravity_for_camera_frame(self.camera_tilt_deg)

        config = FovLossCompensatorConfig(
            gravity_mm_s2=gravity,
            miss_frames_to_predict=self.fov_loss_miss_frames_to_predict,
            max_predict_duration_s=self.fov_loss_max_predict_duration_s,
            min_upward_velocity_mm_s=self.fov_loss_min_upward_velocity_mm_s,
            history_window_s=self.fov_loss_history_window_s,
            drag_coeff_per_s=self.fov_loss_drag_coeff_per_s,
            gate_max_mahalanobis=self.fov_loss_gate_max_mahalanobis,
        )
        self._fov_loss_compensator = FovLossCompensator(config)
        self.get_logger().info(
            'FOV-loss compensation enabled: '
            f'frame={output_frame}, gravity_mm_s2={gravity}, '
            f'max_predict={self.fov_loss_max_predict_duration_s:.2f}s, '
            f'min_upward_v={self.fov_loss_min_upward_velocity_mm_s:.0f}mm/s')
        return self._fov_loss_compensator

    def _apply_fov_loss_compensation(self, observation: dict,
                                     stamp_sec: float) -> dict:
        """Run the compensator on a detector observation and log activity.

        The compensator is a thin pure-Python wrapper, so any exception
        here is a bug — we catch it once and disable the path for the
        rest of the run rather than killing the detection callback.
        """
        comp = self._ensure_fov_loss_compensator()
        if comp is None:
            return observation
        try:
            wrapped = comp.update(observation, observation_time=stamp_sec)
        except Exception as exc:  # noqa: BLE001 - keep callback alive
            self.get_logger().error(
                f'FOV-loss compensator failed: {exc}; disabling for run')
            self.enable_fov_loss_compensation = False
            self._fov_loss_compensator = None
            return observation
        self._maybe_log_fov_loss_status(comp)
        return wrapped

    def _maybe_log_fov_loss_status(self, comp: FovLossCompensator) -> None:
        """Throttled visibility into the compensator state machine."""
        now_wall = time.time()
        state_changed = comp.state != self._fov_loss_last_state
        period_elapsed = (
            now_wall - self._fov_loss_log_last_wall
            >= max(self.fov_loss_log_period_s, 0.1))
        if not (state_changed or period_elapsed):
            return
        delta_predicts = comp.predicts_emitted - self._fov_loss_last_predicts_emitted
        if delta_predicts > 0 or state_changed:
            self.get_logger().info(
                f'FOV-loss: state={comp.state} '
                f'last_event={comp.last_event} '
                f'predicts(+{delta_predicts}, total={comp.predicts_emitted}) '
                f'reacq_ok={comp.reacquire_accepted} '
                f'reacq_rej={comp.reacquire_rejected}')
        self._fov_loss_log_last_wall = now_wall
        self._fov_loss_last_predicts_emitted = comp.predicts_emitted
        self._fov_loss_last_state = comp.state

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
        with self.trajectory_lock:
            trajectory = list(self.trajectory_log)

        if not trajectory:
            self.get_logger().info(
                'No base_link trajectory samples recorded; skipping trajectory save')
            return

        # Save CSV
        if self.save_trajectory_csv_enabled:
            try:
                output_dir = os.path.dirname(self.trajectory_csv_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                with open(self.trajectory_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(['timestamp', 'x_mm', 'y_mm', 'z_mm',
                                     'vx_mm_s', 'vy_mm_s', 'vz_mm_s', 'frame'])
                    # Write data
                    for sample in trajectory:
                        pos = sample['position']
                        vel = sample['velocity']
                        writer.writerow([
                            sample['time'],
                            pos[0], pos[1], pos[2],
                            vel[0], vel[1], vel[2],
                            sample['frame']
                        ])
                self.get_logger().info(
                    f'Saved trajectory CSV to: {self.trajectory_csv_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to save CSV: {e}')

        # Save JSON
        if self.save_trajectory_json_enabled:
            try:
                output_dir = os.path.dirname(self.trajectory_json_path)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)

                with open(self.trajectory_json_path, 'w') as jsonfile:
                    json.dump(trajectory, jsonfile, indent=2)
                self.get_logger().info(
                    f'Saved trajectory JSON to: {self.trajectory_json_path}')
            except Exception as e:
                self.get_logger().error(f'Failed to save JSON: {e}')

        # Save plot
        if not self.save_trajectory_plot_enabled:
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

        Phase 1 (init pose, if enabled): publish right-arm TARGET_DEGREES as
        a single-element MechUnitCmdArray (mu_idx=0 only) on command_topic,
        matching the hardware setup where only the right arm is powered
        during juggling experiments. The command passes through the safety
        limiter.

        Phase 2 (RL control): every tick we call compute_action() regardless
        of detection validity so the RL policy's internal hold-last / dropout
        logic matches training. Raw RL output goes through the safety limiter
        before being published.
        """
        now_wall = time.time()
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
            # Publish only right arm (matching hardware setup where only the
            # right arm is powered during juggling experiments).
            self._publish_right_arm_init_pose(raw_cmd, ball_state.stamp)
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

        # Ball-lost failsafe: if the ball has been undetected for longer than
        # the timeout, stop running the RL policy and command the fixed safe
        # init posture instead. The safety limiter ramps the right arm toward
        # it under the normal vel/acc limits. RL resumes on reacquisition.
        if self._should_engage_ball_lost_failsafe(ball_state, now_wall):
            self._run_ball_lost_failsafe(ball_state.stamp)
            return

        raw_cmd = self.compute_action(ball_state)
        self._last_raw_joint_positions = np.asarray(
            raw_cmd, dtype=np.float64).reshape(-1).copy()
        # Publish RL joint command debug (reflects 200 Hz control cadence).
        self._publish_rl_joint_cmd_debug()
        self._publish_right_arm_command(
            self._last_raw_joint_positions, ball_state.stamp)

    def _should_engage_ball_lost_failsafe(
            self, ball_state: BallState, now_wall: float) -> bool:
        """True when the ball has been lost long enough to park the arm.

        A valid detection clears the failsafe immediately so RL resumes on the
        next tick. With no detection, we compare the age since the last valid
        ball against the timeout. Before any ball has ever been seen we do NOT
        engage — the arm holds its RL initial posture until the first juggle.
        """
        if not self.enable_ball_lost_failsafe:
            return False
        if ball_state.valid:
            if self._failsafe_active:
                self.get_logger().info(
                    'Ball reacquired; exiting failsafe, resuming RL control')
            self._failsafe_active = False
            return False
        if self._last_valid_ball_wall_time is None:
            return False
        lost_for = now_wall - self._last_valid_ball_wall_time
        return lost_for >= self.ball_lost_failsafe_timeout_s

    def _run_ball_lost_failsafe(self, stamp: Time) -> None:
        """Pause RL and drive the right arm toward the safe init posture.

        We feed the fixed target straight to the safety limiter every tick;
        the limiter's velocity/acceleration clamps turn it into a smooth ramp.
        We then pin the RL controller's internal command integrator to the
        actual post-limiter command (zero velocity) so that when RL resumes it
        integrates from where the arm really is, not from a stale pre-loss
        command nor from the not-yet-reached final target.
        """
        if not self._failsafe_active:
            self._failsafe_active = True
            self.get_logger().warn(
                'Ball lost beyond '
                f'{self.ball_lost_failsafe_timeout_s:.2f}s; pausing RL and '
                'moving right arm to safe init posture')
        target = self.safe_init_pose_q
        self._last_raw_joint_positions = target.copy()
        self._publish_right_arm_command(target, stamp)
        if self.rl_controller is not None:
            # Sync the policy integrator to what was actually commanded this
            # tick (the ramp position) with zero velocity, so a resume does
            # not snap from a stale state.
            self.rl_controller.arm_cmd_q = \
                self._last_safe_joint_positions.astype(np.float32).copy()
            self.rl_controller.arm_cmd_qvel = np.zeros(
                self._rl_expected_joint_count, dtype=np.float32)

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

    def _make_unit_cmd(self, mu_idx: int, joint_positions_rad) -> MechUnitCmd:
        """Build a single per-unit MechUnitCmd (joint control only)."""
        cmd = MechUnitCmd()
        cmd.mu_idx = int(mu_idx)
        cmd.jnt_pos = [float(x) for x in joint_positions_rad]
        cmd.use_jnt = True
        cmd.psi = 0.0
        # cmd.end_pose left at the default identity Pose; joint control only.
        return cmd

    def _publish_cmd_array(self, cmds, stamp: Time) -> None:
        """Wrap one or more MechUnitCmd in a MechUnitCmdArray and publish.

        The downstream MCControlNode subscribes to MechUnitCmdArray (header +
        a list of per-unit commands), matching the on-hardware-verified
        pingpong_replay publisher. All publish paths funnel through here.
        """
        msg = MechUnitCmdArray()
        msg.header.stamp = stamp if stamp is not None else \
            self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.cmds = list(cmds)
        self.cmd_pub.publish(msg)

    def _publish_right_arm_command(
            self, joint_positions: np.ndarray, stamp: Time):
        """Safety-limit + publish a right-arm command on command_topic.

        This is the ONLY path that may send right-arm joint commands: every
        raw right-arm target (RL or init-pose) must come through here so it
        passes `RightArmCommandSafetyLimiter` first. The safe command is
        wrapped in a single-element MechUnitCmdArray.
        """
        if len(self.joint_names) != self.safety_limiter.N_JOINTS:
            self.get_logger().error(
                'joint_names length does not match safety limiter '
                f'({len(self.joint_names)} vs '
                f'{self.safety_limiter.N_JOINTS})',
                throttle_duration_sec=1.0)
            return

        raw = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        safe_cmd = self.safety_limiter.filter(raw)
        self._last_safe_joint_positions = safe_cmd.copy()
        self._maybe_log_safety_counts()

        cmd = self._make_unit_cmd(self.right_arm_mu_idx, safe_cmd.tolist())
        self._publish_cmd_array([cmd], stamp)

    # Backward-compatible alias: the old internal name is kept in case
    # external code/tests still reference it.
    _publish_joint_command = _publish_right_arm_command

    def _publish_right_arm_init_pose(self, right_arm_raw_q: np.ndarray,
                                     stamp: Time) -> None:
        """Drive only the right arm toward its TARGET_DEGREES during init.

        This is the default init-pose behavior: only the right arm is
        controlled by this juggling controller (matching the
        on-hardware-verified pingpong_replay). Other mechanical units (left
        arm, leg+waist, head) are maintained by their own controllers or
        teleop. The right-arm command is safety-limited and wrapped in a
        single-element MechUnitCmdArray, keeping the init phase at the same
        topic rate and message structure as the RL phase (one right-arm cmd
        per tick, mu_idx=0 only).

        Use this when only the right arm is powered during experiments.
        """
        # Exactly the same path as _publish_right_arm_command — init and RL
        # phases now publish identically (one right-arm cmd per tick).
        self._publish_right_arm_command(right_arm_raw_q, stamp)

    def _publish_whole_body_init_pose(self, right_arm_raw_q: np.ndarray,
                                      stamp: Time) -> None:
        """Drive every mechanical unit toward TARGET_DEGREES during init.

        The right arm is safety-limited; the other three units (left arm,
        leg+waist, head) are sent their TARGET_DEGREES setpoints directly.
        All four are packed into ONE MechUnitCmdArray and published in a
        single message, so the init phase publishes at the same topic rate
        as the RL phase (one array per control tick) rather than four
        separate messages.
        """
        if len(self.joint_names) != self.safety_limiter.N_JOINTS:
            self.get_logger().error(
                'joint_names length does not match safety limiter '
                f'({len(self.joint_names)} vs '
                f'{self.safety_limiter.N_JOINTS})',
                throttle_duration_sec=1.0)
            return

        raw = np.asarray(right_arm_raw_q, dtype=np.float64).reshape(-1)
        safe_cmd = self.safety_limiter.filter(raw)
        self._last_safe_joint_positions = safe_cmd.copy()
        self._maybe_log_safety_counts()

        cmds = [
            self._make_unit_cmd(self.right_arm_mu_idx, safe_cmd.tolist()),
            self._make_unit_cmd(self.left_arm_mu_idx, target_left_arm_q_rad()),
            self._make_unit_cmd(
                self.leg_waist_mu_idx, target_leg_waist_q_rad()),
            self._make_unit_cmd(self.head_mu_idx, target_head_q_rad()),
        ]
        self._publish_cmd_array(cmds, stamp)

    def _publish_visualization_frame(self, raw_bgr: np.ndarray,
                                     observation: dict) -> None:
        """Hand the latest frame to the viz worker thread (drop-old, O(1)).

        Called from the vision detection callback. This MUST stay cheap: it
        only copies the raw frame into a single slot (overwriting any pending
        frame the worker has not picked up yet) and signals the worker. All
        the expensive work — undistort remap, overlay rendering, and the
        potentially-blocking pipe send to the GUI subprocess — happens in
        `_visualization_worker_loop` on its own thread, so a slow or stalled
        GUI can never back-pressure detection. Intermediate frames are simply
        dropped, which is fine for a debug view.
        """
        if self._viz_disabled_runtime:
            return
        if raw_bgr is None or raw_bgr.size == 0:
            return

        # Lazily start the worker thread on the first frame (it owns spawning
        # the GUI subprocess) so we don't start it before the detector runs.
        if self._viz_thread is None:
            self._start_visualization_worker()
        if self._viz_disabled_runtime:
            return

        # Copy the frame: the source buffer (RealSense frame / decoded ROS
        # image) may be recycled once this callback returns, and the worker
        # reads it asynchronously. The ~1.2 MB memcpy is microseconds — far
        # cheaper than the remap/render/send we just moved off this thread.
        frame_data = (raw_bgr.copy(), observation)

        if self._viz_queue_mode:
            # Queue mode: enqueue all frames for video recording
            try:
                self._viz_frame_queue.put(frame_data, block=False)
                self._viz_event.set()
            except Full:
                # Queue full - log warning and drop this frame
                self.get_logger().warn(
                    'Visualization frame queue full; dropping frame '
                    '(worker thread too slow for 60fps)',
                    throttle_duration_sec=5.0)
        else:
            # Single-slot mode: overwrite with latest (for GUI display only)
            with self._viz_slot_lock:
                self._viz_pending = frame_data
            self._viz_event.set()

    def _start_visualization_worker(self) -> bool:
        """Start the dedicated viz worker thread. True on success."""
        if self._viz_disabled_runtime:
            return False
        if self._viz_thread is not None and self._viz_thread.is_alive():
            return True
        self._viz_thread_running = True
        self._viz_thread = threading.Thread(
            target=self._visualization_worker_loop,
            name='pingpong_viz_worker', daemon=True)
        self._viz_thread.start()
        return True

    def _visualization_worker_loop(self) -> None:
        """Consume the latest frame slot, record it, and drive the GUI.

        Owns the expensive path (undistort remap, overlay render, optional
        video write, GUI pipe send) so the vision callback stays at full
        detection rate. Always processes the most recent frame in the slot,
        dropping any it missed.

        Recording and the GUI display are decoupled: if the GUI subprocess
        dies or its pipe breaks, only the display stops — the worker keeps
        rendering and writing frames to the video file. The worker exits only
        when there is nothing left to do (GUI gone AND not recording) or on a
        render failure.
        """
        while self._viz_thread_running:
            # Wait for a new frame (with a timeout so shutdown is responsive).
            if not self._viz_event.wait(timeout=0.5):
                continue
            self._viz_event.clear()

            # Get frame from queue or single slot
            if self._viz_queue_mode:
                try:
                    pending = self._viz_frame_queue.get(timeout=0.1)
                except Empty:
                    continue
            else:
                with self._viz_slot_lock:
                    pending = self._viz_pending
                    self._viz_pending = None
                if pending is None:
                    continue

            raw_bgr, observation = pending

            # Use the same image space the detector worked in so the
            # circles/bboxes land on the right pixels.
            if (self.detector is not None
                    and self.detector.use_undistort
                    and getattr(self.detector, 'map1', None) is not None
                    and getattr(self.detector, 'map2', None) is not None):
                try:
                    viz_bgr = cv2.remap(
                        raw_bgr, self.detector.map1, self.detector.map2,
                        cv2.INTER_LINEAR)
                except cv2.error:
                    viz_bgr = raw_bgr
            else:
                viz_bgr = raw_bgr

            try:
                annotated = self._render_visualization_frame(
                    viz_bgr, observation)
            except Exception as exc:  # noqa: BLE001 - keep thread alive path
                self.get_logger().warn(
                    f'Visualization render failed: {exc}; '
                    'disabling visualization',
                    throttle_duration_sec=2.0)
                self._viz_disabled_runtime = True
                self._cleanup_visualization_subprocess()
                return

            # Make the array contiguous so pickle-over-pipe send is fast.
            if not annotated.flags['C_CONTIGUOUS']:
                annotated = np.ascontiguousarray(annotated)

            # Best-effort GUI display.
            if not self._viz_gui_disabled:
                self._send_frame_to_gui(annotated)

            # Nothing left to do? Exit the worker.
            if self._viz_gui_disabled:
                self._viz_disabled_runtime = True
                return

    def _send_frame_to_gui(self, annotated: np.ndarray) -> None:
        """Ship one rendered frame to the GUI subprocess (best-effort).

        Spawns the subprocess lazily on the first frame. Any failure (spawn
        error, dead process, broken pipe) disables only the GUI display via
        ``_viz_gui_disabled`` — recording, if enabled, keeps running.
        """
        # Lazily spawn the GUI subprocess on the first real frame.
        if self._viz_proc is None and not self._start_visualization_subprocess():
            self._viz_gui_disabled = True
            return
        if self._viz_proc is not None and not self._viz_proc.is_alive():
            self.get_logger().info(
                'Visualization subprocess exited; '
                'continuing without display')
            self._viz_gui_disabled = True
            self._cleanup_visualization_subprocess()
            return

        with self._viz_lock:
            send = self._viz_send
        if send is None:
            self._viz_gui_disabled = True
            return
        # This send may block if the GUI is slow — but it blocks THIS worker
        # thread, not the detection callback.
        try:
            send.send(annotated)
        except (BrokenPipeError, OSError, EOFError) as exc:
            self.get_logger().info(
                f'Visualization pipe closed ({exc}); '
                'continuing without display')
            self._viz_gui_disabled = True
            self._cleanup_visualization_subprocess()

    def _start_visualization_subprocess(self) -> bool:
        """Spawn the GUI subprocess; True on success, False on failure.

        A failure here disables only the GUI display (``_viz_gui_disabled``),
        never the whole pipeline — video recording, if enabled, keeps running.
        """
        if self._viz_disabled_runtime or self._viz_gui_disabled:
            return False
        try:
            # `spawn` instead of `fork` so the child gets a clean Python
            # interpreter without our ROS / CUDA / Qt globals. Import the
            # worker from a minimal module so the child only loads cv2/numpy,
            # not torch/ultralytics/pyrealsense2/rclpy.
            from pingpong_controller.viz_subprocess import viz_main
            ctx = mp.get_context('spawn')
            # Pipe(duplex=False) returns (reader, writer). Parent sends
            # frames -> writer; child reads from reader.
            child_conn, parent_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=viz_main,
                args=(child_conn, self.visualization_window_name),
                name='pingpong_viz', daemon=True)
            proc.start()
            # The child holds its own end; close ours so EOF propagates
            # cleanly when the child exits.
            child_conn.close()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'Failed to start visualization subprocess: {exc}; '
                'disabling GUI display')
            self._viz_gui_disabled = True
            return False

        self._viz_proc = proc
        self._viz_send = parent_conn
        self.get_logger().info(
            f'Visualization subprocess started (pid={proc.pid}); '
            f'window "{self.visualization_window_name}"')
        return True

    def _cleanup_visualization_subprocess(self) -> None:
        """Close the pipe and reap the GUI subprocess."""
        with self._viz_lock:
            send = self._viz_send
            self._viz_send = None
        if send is not None:
            try:
                send.send(None)  # graceful shutdown signal
            except (BrokenPipeError, OSError, EOFError):
                pass
            try:
                send.close()
            except OSError:
                pass
        if self._viz_proc is not None:
            try:
                self._viz_proc.join(timeout=2.0)
            except Exception:  # noqa: BLE001
                pass
            if self._viz_proc.is_alive():
                try:
                    self._viz_proc.terminate()
                    self._viz_proc.join(timeout=1.0)
                except Exception:  # noqa: BLE001
                    pass
            self._viz_proc = None

    def _shutdown_visualization(self) -> None:
        """Tear down the viz worker thread + subprocess. Safe to call twice."""
        self._viz_disabled_runtime = True
        # Stop the worker thread first so it can't touch the pipe or the video
        # writer mid-teardown.
        self._viz_thread_running = False
        self._viz_event.set()
        if self._viz_thread is not None and self._viz_thread.is_alive():
            self._viz_thread.join(timeout=2.0)
        self._viz_thread = None

        self._cleanup_visualization_subprocess()

    # ------------------------------------------------------------------
    # Raw video recording helpers (independent of visualization). Fed from
    # the RealSense reader thread via _enqueue_raw_video_frame; a dedicated
    # writer thread does the encoding so the reader never blocks on disk.
    # ------------------------------------------------------------------

    def _enqueue_raw_video_frame(self, raw_bgr: np.ndarray) -> None:
        """Hand one raw RealSense frame to the raw-video writer thread.

        Called from the RealSense reader loop. O(1) non-blocking: copies the
        frame (the RealSense buffer is recycled after the callback) and pushes
        it onto the writer queue, dropping the frame only if the queue is full
        (disk/encoder can't keep up).
        """
        if self._raw_video_writer_thread is None:
            self._start_raw_video_writer_thread()
        if self._raw_video_queue is None:
            return
        try:
            self._raw_video_queue.put(raw_bgr.copy(), block=False)
        except Full:
            self.get_logger().warn(
                f'Raw video queue full ({self._raw_video_queue_max_size} '
                'frames); dropping frame (disk/encoder too slow)',
                throttle_duration_sec=5.0)

    def _start_raw_video_writer_thread(self) -> None:
        """Start the dedicated raw-video writer thread (idempotent)."""
        if self._raw_video_writer_thread is not None:
            return
        self._raw_video_queue = Queue(maxsize=self._raw_video_queue_max_size)
        self._raw_video_writer_running = True
        self._raw_video_writer_thread = threading.Thread(
            target=self._raw_video_writer_loop,
            name='raw_video_writer', daemon=True)
        self._raw_video_writer_thread.start()
        self.get_logger().info('Raw video writer thread started')

    def _raw_video_writer_loop(self) -> None:
        """Dedicated thread: drain the queue and encode raw frames."""
        while self._raw_video_writer_running:
            try:
                frame = self._raw_video_queue.get(timeout=0.1)
                self._write_raw_video_frame(frame)
            except Empty:
                continue
            except Exception as exc:  # noqa: BLE001 - keep node alive
                self.get_logger().error(
                    f'Raw video writer thread error: {exc}; stopping '
                    'raw recording', throttle_duration_sec=2.0)
                self._raw_video_disabled_runtime = True
                break

    def _write_raw_video_frame(self, raw_bgr: np.ndarray) -> None:
        """Open the raw writer lazily and append one BGR frame.

        Runs on the raw-video writer thread. The first frame fixes resolution
        and fps for the whole file. Any failure disables raw recording for the
        rest of the run without killing the thread.
        """
        if raw_bgr is None or raw_bgr.size == 0:
            return
        height, width = raw_bgr.shape[:2]
        with self._raw_video_lock:
            if self._raw_video_writer is None:
                if not self._open_raw_video_writer(width, height):
                    self._raw_video_disabled_runtime = True
                    return
            frame = raw_bgr
            if (self._raw_video_frame_size is not None
                    and (width, height) != self._raw_video_frame_size):
                try:
                    frame = cv2.resize(raw_bgr, self._raw_video_frame_size)
                except cv2.error:
                    return
            try:
                self._raw_video_writer.write(frame)
                self._raw_video_frames_written += 1
            except Exception as exc:  # noqa: BLE001 - keep thread alive
                self.get_logger().warn(
                    f'Raw video write failed: {exc}; disabling raw recording',
                    throttle_duration_sec=2.0)
                self._raw_video_disabled_runtime = True
                self._release_raw_video_writer_locked()

    def _resolve_raw_video_path(self) -> str:
        """Return the raw video output path, auto-generating one if unset."""
        path = self.raw_video_path
        if path:
            return os.path.expandvars(os.path.expanduser(path))
        ts = time.strftime('%Y%m%d_%H%M%S')
        return os.path.join(
            self.source_module_dir, 'outputs', 'videos',
            f'pingpong_raw_{ts}.mp4')

    def _resolve_raw_video_fps(self) -> float:
        """Pick the raw recording fps: explicit param, else realsense fps."""
        if self.raw_video_fps > 0.0:
            return float(self.raw_video_fps)
        try:
            rs_fps = float(self.realsense_fps)
            if rs_fps > 0.0:
                return rs_fps
        except (TypeError, ValueError):
            pass
        return 60.0

    def _open_raw_video_writer(self, width: int, height: int) -> bool:
        """Create the raw cv2.VideoWriter. Caller must hold _raw_video_lock."""
        path = self._resolve_raw_video_path()
        out_dir = os.path.dirname(path)
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as exc:
                self.get_logger().error(
                    f'Cannot create raw video output dir {out_dir}: {exc}; '
                    'raw recording disabled')
                return False

        fps = self._resolve_raw_video_fps()
        # mp4v is the most broadly available codec; raw frames are large but
        # this keeps the offline post-process step (re-decode + detect) simple.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not writer.isOpened():
            self.get_logger().error(
                f'Failed to open raw VideoWriter for {path} '
                f'({width}x{height}@{fps:.1f}); raw recording disabled')
            return False

        self._raw_video_writer = writer
        self._raw_video_path_resolved = path
        self._raw_video_frame_size = (width, height)
        self._raw_video_frames_written = 0
        self.get_logger().info(
            f'Recording raw video -> {path} '
            f'({width}x{height}@{fps:.1f} fps)')
        return True

    def _release_raw_video_writer_locked(self) -> None:
        """Release the raw writer and log a summary. Caller holds the lock."""
        if self._raw_video_writer is None:
            return
        try:
            self._raw_video_writer.release()
        except Exception:  # noqa: BLE001
            pass
        self._raw_video_writer = None
        if self._raw_video_frames_written > 0 and self._raw_video_path_resolved:
            self.get_logger().info(
                f'Saved raw video: {self._raw_video_path_resolved} '
                f'({self._raw_video_frames_written} frames)')
        elif self._raw_video_path_resolved:
            self.get_logger().warn(
                'Raw video recording produced no frames; file may be empty: '
                f'{self._raw_video_path_resolved}')

    def _stop_raw_video_recording(self) -> None:
        """Stop the raw writer thread, flush the queue, finalize the file."""
        if self._raw_video_writer_thread is not None:
            self._raw_video_writer_running = False
            if self._raw_video_writer_thread.is_alive():
                self._raw_video_writer_thread.join(timeout=5.0)
            # Flush whatever is still queued so we don't lose the tail.
            if self._raw_video_queue is not None:
                remaining = self._raw_video_queue.qsize()
                if remaining > 0:
                    self.get_logger().info(
                        f'Flushing {remaining} remaining raw video frames...')
                while True:
                    try:
                        frame = self._raw_video_queue.get_nowait()
                    except Empty:
                        break
                    self._write_raw_video_frame(frame)
            self._raw_video_writer_thread = None
            self._raw_video_queue = None
        with self._raw_video_lock:
            self._release_raw_video_writer_locked()

    def _render_visualization_frame(self, frame: np.ndarray,
                                    observation: dict) -> np.ndarray:
        """Overlay detection results on a BGR frame.

        Thin wrapper around the shared ``render_detection_overlay`` so the
        live visualization window and the offline raw-video annotation tool
        draw identical overlays. All node-specific state (frame counters,
        callback rate, output frame) is gathered here and passed in.
        """
        if len(self.callback_times) > 0:
            avg_dt = sum(self.callback_times) / len(self.callback_times)
            callback_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            callback_hz = 0.0
        output_frame = (
            getattr(self.detector, 'output_frame', 'camera')
            if self.detector is not None else 'camera')
        return render_detection_overlay(
            frame, observation,
            frame_count=self.frame_count,
            detection_count=self.detection_count,
            callback_hz=callback_hz,
            output_frame=output_frame)

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
        # is always free for control. The visualization window (when enabled)
        # is driven by a dedicated viz worker thread fed via a single-slot
        # drop-old handoff from the vision callback, and the actual cv2.imshow
        # runs in a separate GUI subprocess — so neither rendering nor a
        # stalled GUI can back-pressure detection or control.
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
            node._stop_raw_video_recording()
            node._shutdown_visualization()
            node.save_trajectory_plot()
            if executor is not None:
                executor.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
