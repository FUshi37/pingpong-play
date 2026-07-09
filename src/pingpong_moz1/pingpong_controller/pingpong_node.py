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
import sys
import threading
from collections import deque
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
from std_msgs.msg import Empty as EmptyMsg
from builtin_interfaces.msg import Time
from mc_core_interface.msg import MechUnitCmd, MechUnitCmdArray
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor

from pingpong_controller.ball_detector import PingPongBallDetector
from pingpong_controller.hybrid_ball_detector_adapter import HybridBallDetectorAdapter
from pingpong_controller.calibration_loader import load_calibration
from pingpong_controller.viz_render import (
    render_detection_overlay,
    draw_sim_done_overlay,
)
from pingpong_controller.fov_loss_compensator import (
    FovLossCompensator,
    FovLossCompensatorConfig,
    gravity_for_base_link,
    gravity_for_camera_frame,
)
from pingpong_controller.ball_state_estimator import (
    BallStateEstimator,
    BallStateEstimatorConfig,
    gravity_for_base_link as estimator_gravity_for_base_link,
    gravity_for_camera_frame as estimator_gravity_for_camera_frame,
)
from pingpong_controller.mjx_rl_policy import (
    MjxRLPolicyController,
    inspect_mjx_checkpoint_metadata,
)
from pingpong_controller.rl_policy import (
    EXPECTED_OBS_DIM,
    EXPECTED_OBS_DIM_WITH_DELAY,
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
from pingpong_controller.done_condition_monitor import DoneConditionMonitor

# Optional MuJoCo FK for the racket anchor (used by the viz done-condition
# overlay independent of the RL policy, so it works with enable_rl_policy:=false).
try:
    import mujoco as _mj
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False
    _mj = None

# Import compensation from pingpong_replay
try:
    from pingpong_replay.compensation import CommandCompensator
    _COMPENSATION_AVAILABLE = True
except ImportError:
    _COMPENSATION_AVAILABLE = False
    CommandCompensator = None


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

    def _resolve_rl_model_path(self, requested_path: str) -> str:
        """Resolve an RL model file; MJX paths may point at a log directory."""
        path = os.path.expandvars(os.path.expanduser(str(requested_path)))
        if self.policy_backend != 'mjx' or not os.path.isdir(path):
            return path

        candidates = sorted(glob.glob(os.path.join(path, '*.pkl')))
        if not candidates:
            self.get_logger().warn(
                f'MJX rl_model_path is a directory but contains no .pkl: {path}')
            return path

        preferred = [
            p for p in candidates
            if 'curriculum_last' not in os.path.basename(p)
            and 'safety_stop' not in os.path.basename(p)
        ]
        pool = preferred or candidates
        selected = max(pool, key=lambda p: os.path.getmtime(p))
        self.get_logger().info(
            f'MJX rl_model_path directory resolved to checkpoint: {selected}')
        return selected

    def _configure_policy_observation_path(self) -> None:
        """Select 50D or 67D policy observations before loading the controller."""
        requested_dim = int(self.get_parameter('policy_obs_dim').value)
        if requested_dim in (EXPECTED_OBS_DIM, EXPECTED_OBS_DIM_WITH_DELAY):
            self.enable_delay_conditioning = (
                requested_dim == EXPECTED_OBS_DIM_WITH_DELAY)
            self.policy_obs_dim = requested_dim
            self.get_logger().info(
                f'Policy observation path forced by policy_obs_dim='
                f'{requested_dim}: '
                f'delay_conditioning={self.enable_delay_conditioning}')
            return
        if requested_dim not in (0,):
            self.get_logger().warn(
                f'Invalid policy_obs_dim={requested_dim}; expected 0, '
                f'{EXPECTED_OBS_DIM}, or {EXPECTED_OBS_DIM_WITH_DELAY}. '
                'Falling back to enable_delay_conditioning.')

        self.policy_obs_dim = (
            EXPECTED_OBS_DIM_WITH_DELAY if self.enable_delay_conditioning
            else EXPECTED_OBS_DIM)
        if self.policy_backend != 'mjx':
            return

        try:
            meta = inspect_mjx_checkpoint_metadata(self.rl_model_path)
            checkpoint_dim = int(meta.get('obs_dim', 0) or 0)
        except Exception as exc:  # noqa: BLE001 - keep old path usable
            self.get_logger().warn(
                f'Could not inspect MJX checkpoint obs_dim from '
                f'{self.rl_model_path}: {exc}; using '
                f'enable_delay_conditioning={self.enable_delay_conditioning}')
            return

        if checkpoint_dim not in (EXPECTED_OBS_DIM, EXPECTED_OBS_DIM_WITH_DELAY):
            self.get_logger().warn(
                f'MJX checkpoint has unexpected obs_dim={checkpoint_dim}; '
                f'using enable_delay_conditioning={self.enable_delay_conditioning}')
            return

        self.enable_delay_conditioning = (
            checkpoint_dim == EXPECTED_OBS_DIM_WITH_DELAY)
        self.policy_obs_dim = checkpoint_dim
        mode = (
            '67D delay-conditioned'
            if self.enable_delay_conditioning else '50D legacy')
        self.get_logger().info(
            f'MJX checkpoint obs_dim={checkpoint_dim}; using {mode} '
            'policy observation path')

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

        # Aligned-depth recording for OFFLINE hardware-depth replay validation.
        # When on (requires save_raw_video + use_direct_realsense +
        # use_depth_sensor), the reader loop records the aligned 16-bit depth
        # frame alongside each RGB frame, plus a per-frame timestamp sidecar so
        # the offline tool can pair depth<->RGB by frame index and replay with
        # the TRUE capture times instead of a constant-fps assumption. RGB,
        # depth, and timestamps are written in lockstep by one writer thread, so
        # a dropped frame drops the whole bundle and the indices never desync.
        # Depth is stored as per-frame .npy (16-bit z16 can't go in mp4v).
        self.declare_parameter('save_depth_frames', False)

        # When on (requires save_raw_video + use_hybrid_detector), the hybrid
        # reader loop also records the IR (infrared-1) frame alongside each RGB
        # frame, plus a one-shot camera_params JSON (IR/RGB intrinsics + IR->RGB
        # extrinsics + depth_scale). The hybrid detector localizes on the IR
        # stream, so this is what tools/annotate_raw_video_hybrid.py needs to
        # reproduce detection offline. IR is stored as per-frame .npy (uint8).
        self.declare_parameter('save_ir_frames', False)

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
        # Enable delay conditioning features in observation (67-dim vs 50-dim).
        # Set to True for checkpoints trained with delay conditioning (e.g.,
        # 18_stage4g_strong_contact_dr_inverse_mpc.pkl); set to False for
        # legacy 50-dim checkpoints. Default True for new deployments.
        self.declare_parameter('enable_delay_conditioning', True)
        # Explicit policy observation dimension. 0 means auto for MJX
        # checkpoints (read obs_dim from the .pkl); 50 forces the legacy
        # no-delay path; 67 forces delay conditioning.
        self.declare_parameter('policy_obs_dim', 0)

        # Replicate simulation bug for validation: the training environment has a bug
        # where racket_vel ≈ 0 due to using already-updated prev_racket_pos. Set True
        # to replicate and verify policy with perfectly matched obs distribution.
        self.declare_parameter('force_obs_to_training_dist', True)

        # Actuator delay compensation parameters
        # These must match the training environment's compensation settings.
        # Default values are from the shipped inverse_mpc checkpoint.
        self.declare_parameter('enable_compensation', True)
        self.declare_parameter('compensation_mode', 'inverse_mpc')
        self.declare_parameter('comp_delay_ms', 72.0)
        self.declare_parameter('comp_actuator_tau_s', 0.074)
        self.declare_parameter('comp_actuator_gain', 1.0)
        self.declare_parameter('comp_buffer_extra_steps', 4)
        self.declare_parameter('comp_delay_conditioning', True)
        # inverse-MPC parameters (from checkpoint env_cfg)
        self.declare_parameter('comp_mpc_beta', 1.2)
        self.declare_parameter('comp_mpc_delay_scale', 1.05)
        self.declare_parameter('comp_mpc_tau_scale', 0.75)
        self.declare_parameter('comp_mpc_horizon_steps', 6)
        self.declare_parameter('comp_mpc_tracking_weight', 1.0)
        self.declare_parameter('comp_mpc_nominal_weight', 0.25)
        self.declare_parameter('comp_mpc_delta_weight', 0.05)
        self.declare_parameter('comp_mpc_max_delta_rad', 0.5235987755982988)  # 30 deg

        # Replay mode: use pre-recorded ball state CSV instead of live detection
        self.declare_parameter('enable_replay_mode', False)
        self.declare_parameter('replay_csv_path', '')
        self.declare_parameter('replay_loop', False)
        self.declare_parameter('replay_data_is_sim_frame', False)  # True if replay CSV is already in sim coordinates
        self.declare_parameter('replay_auto_exit', True)

        # Simulation ball state input: subscribe to ball state from simulation
        # instead of using vision detection. Useful for testing policy with
        # perfect ball state input from simulation.
        self.declare_parameter('enable_sim_ball_input', False)
        self.declare_parameter('sim_ball_topic', '/sim/ball_state')

        # Motion capture ball state input: receive ball state from mocap system
        # via UDP. Requires prior calibration using mocap_calibration.py.
        self.declare_parameter('enable_mocap_input', False)
        self.declare_parameter('mocap_bind_host', '0.0.0.0')
        self.declare_parameter('mocap_udp_port', 9870)
        self.declare_parameter('mocap_calibration_path', '')
        self.declare_parameter('mocap_timeout_s', 0.5)
        self.declare_parameter('mocap_marker_id', 1)
        self.declare_parameter('mocap_position_unit', 'auto')
        self.declare_parameter('mocap_timestamp_unit', 's')
        self.declare_parameter('mocap_velocity_alpha', 0.35)
        self.declare_parameter('mocap_max_velocity_m_s', 20.0)

        # Safety limiter: can be disabled for simulation/testing
        self.declare_parameter('enable_safety_limiter', True)

        # Done condition monitoring: replicate sim termination conditions
        self.declare_parameter('enable_done_monitoring', True)
        self.declare_parameter('done_ball_z_floor_m', 0.8)
        self.declare_parameter('done_ball_z_ceiling_m', 1.9)
        self.declare_parameter('done_ball_x_range_m', 0.5)
        self.declare_parameter('done_ball_y_range_m', 0.5)
        self.declare_parameter('done_racket_anchor_distance_m', 1.1)
        self.declare_parameter('done_racket_z_high_offset_m', 0.24)
        self.declare_parameter('done_racket_z_low_offset_m', -0.12)
        # End-effector workspace limits (optional hard constraints, base frame)
        # Set to empty list to disable; otherwise [min, max] in meters
        dynamic_array_param = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter(
            'done_ee_x_limits_m', [], dynamic_array_param)
        self.declare_parameter(
            'done_ee_y_limits_m', [], dynamic_array_param)
        self.declare_parameter(
            'done_ee_z_limits_m', [], dynamic_array_param)
        self.declare_parameter('done_log_period_s', 2.0)

        # Whole-body init pose publishing.
        self.declare_parameter('enable_init_pose', True)
        self.declare_parameter('init_pose_duration_s', 6.0)
        self.declare_parameter('init_pose_tolerance_rad', 0.05)
        # After reaching the init pose, hold and wait for the operator to press
        # Enter on stdin or publish Empty to start_confirm_topic before
        # starting RL (mirrors pingpong_replay, with a hardware-safe ROS topic
        # fallback for GUI/headless runs).
        self.declare_parameter('require_enter_confirm', False)
        self.declare_parameter('start_confirm_topic', '/pingpong/start_control')
        # For hardware safety, stdin EOF/non-TTY does not confirm start unless
        # this is explicitly enabled. Use start_confirm_topic for headless runs.
        self.declare_parameter('auto_continue_on_stdin_eof', False)
        self.declare_parameter('enter_prompt_log_period_s', 1.0)
        # After Enter is confirmed, keep holding the init pose for this many
        # seconds before the policy is allowed to run. This gives the operator
        # time to step clear / prepare the first toss on hardware.
        self.declare_parameter('post_enter_start_delay_s', 0.0)
        # Formal run duration after init + Enter + post-enter delay. <=0 means
        # unlimited. When it expires, park at the safe init pose and keep
        # publishing that pose instead of abruptly stopping command output.
        self.declare_parameter('run_duration_s', 0.0)
        # Optional auto-exit for scripted real experiments. Keep disabled by
        # default so manual hardware sessions can continue parking until the
        # operator stops the node.
        self.declare_parameter('shutdown_after_run_duration', False)
        self.declare_parameter('shutdown_after_run_duration_delay_s', 1.0)
        # Mechanical unit indices. All four MechUnitCmd streams share
        # command_topic (default /mx_mix_command). TODO(jacky): set these to
        # real unit ids before deployment.
        self.declare_parameter('right_arm_mu_idx', 0)
        self.declare_parameter('left_arm_mu_idx', 1)
        self.declare_parameter('leg_waist_mu_idx', 2)
        self.declare_parameter('head_mu_idx', 3)
        self.declare_parameter('warn_no_command_subscriber', True)
        self.declare_parameter('command_subscriber_warn_period_s', 2.0)

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
        #
        # Default tightened 1.0 -> 0.4 s: 1.0 s is far too long for a fast
        # ball. When the ball-state estimator is enabled it is the single loss
        # authority — it keeps the ball valid through graded extrapolation up
        # to ball_estimator_invalid_after_s (~0.60 s, covering a full toss),
        # which refreshes _last_valid_ball_wall_time every tick; only once the
        # estimator reports invalid does this timeout start counting and fire.
        # So the arm holds the toss out and only parks if the ball is truly
        # gone past the estimator's window.
        self.declare_parameter('enable_ball_lost_failsafe', True)
        self.declare_parameter('enable_ball_capture_gate', True)
        # Release gate: after the operator confirms start and a valid ball is
        # visible, keep parking until the held ball is actually released. A
        # hand-held static ball is a valid observation, but it should not start
        # RL. Release is detected from fresh downward motion over a short
        # window, then RL/compensation are re-anchored before the first policy
        # tick.
        self.declare_parameter('enable_ball_release_gate', True)
        self.declare_parameter('ball_release_min_downward_velocity_m_s', 0.15)
        self.declare_parameter('ball_release_min_z_drop_m', 0.015)
        self.declare_parameter('ball_release_window_s', 0.12)
        self.declare_parameter('ball_release_required_consecutive_ticks', 3)
        self.declare_parameter('ball_release_max_obs_age_s', 0.08)
        self.declare_parameter('ball_lost_failsafe_timeout_s', 0.4)
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
        # 方案1：椭圆拟合短轴作半径（对运动拖影鲁棒，降低深度噪声）。默认开。
        self.declare_parameter('enable_ellipse_fit', True)
        # 方案4：多帧时序跟踪层（运动一致性门控 + 短暂漏检桥接 + ROI 引导）。
        # 默认关，向后兼容；开后对快球/短暂遮挡更鲁棒。
        self.declare_parameter('enable_temporal_tracker', False)
        self.declare_parameter('tracker_max_miss_bridge', 2)
        self.declare_parameter('tracker_gate_px', 120.0)

        # 使用HybridBallDetector (RGB YOLO + IR检测) 代替传统检测器
        # 适用于灰色反光球，不依赖颜色筛选
        self.declare_parameter('use_hybrid_detector', False)
        self.declare_parameter('ir_params_json', '')  # IR参数JSON文件路径，优先级高于单独参数
        self.declare_parameter('depth_correction_model_path', 'depth_correction_model.pkl')  # 深度校正模型路径
        self.declare_parameter('yolo_interval', 1)  # YOLO每N帧运行一次，默认1（每帧）
        self.declare_parameter('ir_threshold', 170)
        self.declare_parameter('use_adaptive_ir', True)
        self.declare_parameter('ir_exposure', 1000)  # μs，1ms
        self.declare_parameter('ir_gain', 16)
        self.declare_parameter('ir_laser_power', 360)
        self.declare_parameter('ir_min_radius', 15)
        self.declare_parameter('ir_max_radius', 60)
        self.declare_parameter('ir_circularity_min', 0.7)
        self.declare_parameter('ir_brightness_percentile', 25)
        self.declare_parameter('hybrid_roi_margin', 100)

        # 颜色判据（用 tools/vision_calib/hsv_color_tuner.py 调好后填入）。
        # 收紧 HSV 范围/提高阈值可减少把别的橙色物体误判成球。
        # 灰色反光球时设置color_match_threshold=0禁用颜色筛选
        self.declare_parameter('hsv_lower', [5, 50, 50])
        self.declare_parameter('hsv_upper', [30, 255, 255])
        self.declare_parameter('color_match_threshold', 0.3)
        self.declare_parameter('color_circularity_min', 0.4)
        self.declare_parameter('camera_tilt_deg', 50.33)
        self.declare_parameter('use_undistort', True)
        # Suppress non-critical vision status logs for hardware runs. Errors
        # and throttled warnings still print.
        self.declare_parameter('quiet_vision_logs', False)
        # <=0 disables per-frame detection result logging.
        self.declare_parameter('print_every', 1)
        self.declare_parameter('realsense_fps', 60)
        # Use the RealSense hardware frame timestamp (exposure time) instead of
        # the reader-loop clock for stamping detections. Captures the true
        # image-capture instant and includes exposure+USB latency in the
        # perception-latency de-aging; the cost is clock-domain alignment —
        # RS and ROS clocks drift over long runs. Default OFF; when ON, a
        # one-shot offset is calibrated at the first frame. Only effective in
        # direct_realsense mode.
        self.declare_parameter('use_realsense_hw_timestamp', False)
        # Hardware depth (D455): use the aligned depth frame for the ball's
        # depth instead of monocular circle-radius geometry. Default ON — the
        # D455's wide-baseline active stereo gives ~10-20 mm depth noise vs the
        # ~80 mm monocular floor that dominates the prediction error, and
        # get_depth_from_depthmap guards small-ball background contamination
        # (ball-mask sampling + 25th-percentile + monocular-prior gate),
        # falling back to monocular per-frame when the gate rejects. Only
        # effective in direct_realsense mode (the depth stream is opened
        # there); in ROS-topic mode it is inert and detection stays monocular.
        self.declare_parameter('use_depth_sensor', True)
        self.declare_parameter('depth_neighborhood_px', 2)

        # FOV-loss compensator: when the ball flies above the camera's
        # field of view during a juggle, propagate a ballistic prediction
        # forward instead of letting detection drop to invalid. Default
        # OFF — controllers fall back to RL hold-last just like before.
        self.declare_parameter('enable_fov_loss_compensation', False)
        # 0.8 s of ballistic extrapolation is unreliable for a spinning
        # pingpong ball (Magnus error dominates after ~100-200 ms), so the
        # default is tightened to 0.20 s. Raise it explicitly if you trust
        # longer predicts.
        self.declare_parameter('fov_loss_max_predict_duration_s', 0.2)
        self.declare_parameter('fov_loss_miss_frames_to_predict', 2)
        self.declare_parameter('fov_loss_min_upward_velocity_mm_s', 200.0)
        self.declare_parameter('fov_loss_history_window_s', 0.15)
        self.declare_parameter('fov_loss_drag_coeff_per_s', 0.0)
        self.declare_parameter('fov_loss_gate_max_mahalanobis', 4.0)
        self.declare_parameter('fov_loss_log_period_s', 1.0)

        # Ball-state estimator: a single Kalman filter that bridges the ~60 Hz
        # vision feed to the 200 Hz control loop AND compensates the perception
        # latency. update() fuses each real detection stamped at its image time
        # (t_img); predict_to() rolls the committed state forward to the
        # control tick, so the elapsed (t_now - t_img) de-ages the measurement
        # and absorbs the ~21-40 ms sensing latency with no hand-tuned
        # constant. Execution latency is NOT compensated here (it is modeled by
        # the RL actuator model), so prediction_horizon_s stays 0.
        #
        # Default ON: it is the single state source that delivers all three of
        # 200 Hz output, perception-latency de-aging, and toss-loss robustness.
        # It SUPERSEDES the FOV-loss compensator (above) — do not enable both,
        # or the toss arc gets propagated twice. When this is off the control
        # loop consumes the detector observation verbatim. See
        # ball_state_estimator.py.
        self.declare_parameter('enable_ball_state_estimator', True)
        # Extrapolation / loss timeline sized for ball-juggling tosses. A toss
        # above the camera FOV is pure free flight (no contact), so the gravity
        # +drag ballistic model is accurate across the whole out-of-view arc;
        # the limit is depth-axis velocity noise, not the dynamics. A 0.5 m toss
        # leaves the FOV for ~0.3-0.4 s, so the extrapolation cap and the
        # invalid timeout are set above that with margin (was 0.20 / 0.45 —
        # tuned for short frame drops, too tight for a full toss).
        self.declare_parameter('ball_estimator_max_extrapolation_s', 0.45)
        self.declare_parameter('ball_estimator_trust_loss_s', 0.10)
        self.declare_parameter('ball_estimator_degrade_loss_s', 0.40)
        self.declare_parameter('ball_estimator_invalid_after_s', 0.60)
        self.declare_parameter('ball_estimator_prediction_horizon_s', 0.0)
        # Master switch for the ballistic extrapolation in predict_to(). Default
        # True keeps the full behavior (60->200 Hz bridge + perception-latency
        # de-aging). Set False for "no prediction, just hold the latest fused
        # ball state and publish it at 200 Hz": the Kalman filter still fuses and
        # smooths each real detection at the camera rate, but the control loop
        # sees the held measurement between frames instead of a forward-propagated
        # one. Mirrors a sim whose ball state updates at the camera rate (e.g.
        # 50/60 Hz) and is merely re-issued at 200 Hz. See ball_state_estimator.py.
        self.declare_parameter('ball_estimator_enable_prediction', True)
        self.declare_parameter('ball_estimator_r_lateral_mm', 20.0)
        self.declare_parameter('ball_estimator_r_depth_mm', 60.0)
        self.declare_parameter('ball_estimator_q_depth_pos_mult', 0.50)
        self.declare_parameter('ball_estimator_q_depth_vel_mult', 0.20)
        # Aerodynamic drag in the ballistic propagation. A 2.7 g / 40 mm ball
        # has a high drag-to-mass ratio; a no-drag model overshoots a fast ball
        # by ~80 mm over a 150 ms detector dropout. Default ON. The coefficient
        # derives from the ITTF ball + standard air (Cd=0.40); see
        # ball_state_estimator.BallStateEstimatorConfig.drag_k_per_mm.
        self.declare_parameter('ball_estimator_enable_aero_drag', True)
        self.declare_parameter('ball_estimator_drag_k_per_mm', 1.117e-4)
        # Magnus (spin) lift. OFF by default and inert at zero spin: this rig
        # has no spin sensor and spin is not observable from position-only
        # arcs. Hook for experiments with an externally KNOWN serve spin only.
        self.declare_parameter('ball_estimator_enable_magnus', False)
        self.declare_parameter('ball_estimator_magnus_k', 0.0)
        # Table plane height (mm) along the up axis in the estimator frame,
        # used to arm bounce detection before the ball reaches the table. NaN
        # disables the height test (bounce flagged from velocity sign flip +
        # speed jump only). Measure this on hardware to enable height-armed
        # bounce guarding.
        self.declare_parameter('ball_estimator_table_z_mm', float('nan'))
        self.declare_parameter('ball_estimator_log_period_s', 1.0)
        # Record the estimator's full 200 Hz predict_to() output to CSV for
        # offline debugging (one row per control tick). Captures the bridged /
        # de-aged ball state alongside the latest raw 60 Hz detection so you
        # can compare them. Written on shutdown. Default OFF; blank path =>
        # timestamped file under outputs/rl_sim/.
        self.declare_parameter('save_estimator_log', False)
        self.declare_parameter('estimator_log_path', '')

        # RL policy obs/action logging for debugging and analysis
        self.declare_parameter('save_policy_log', False)
        self.declare_parameter('policy_log_path', '')

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
        self.save_depth_frames = bool(
            self.get_parameter('save_depth_frames').value)
        self.save_ir_frames = bool(
            self.get_parameter('save_ir_frames').value)

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
        self.enable_ellipse_fit = bool(
            self.get_parameter('enable_ellipse_fit').value)
        self.enable_temporal_tracker = bool(
            self.get_parameter('enable_temporal_tracker').value)
        self.tracker_max_miss_bridge = int(
            self.get_parameter('tracker_max_miss_bridge').value)
        self.tracker_gate_px = float(
            self.get_parameter('tracker_gate_px').value)

        # Hybrid detector parameters
        self.use_hybrid_detector = bool(
            self.get_parameter('use_hybrid_detector').value)
        self.ir_params_json = str(
            self.get_parameter('ir_params_json').value).strip()
        self.depth_correction_model_path = str(
            self.get_parameter('depth_correction_model_path').value).strip()
        self.yolo_interval = int(
            self.get_parameter('yolo_interval').value)
        self.ir_threshold = int(
            self.get_parameter('ir_threshold').value)
        self.use_adaptive_ir = bool(
            self.get_parameter('use_adaptive_ir').value)
        self.ir_exposure = int(
            self.get_parameter('ir_exposure').value)
        self.ir_gain = int(
            self.get_parameter('ir_gain').value)
        self.ir_laser_power = int(
            self.get_parameter('ir_laser_power').value)
        self.ir_min_radius = int(
            self.get_parameter('ir_min_radius').value)
        self.ir_max_radius = int(
            self.get_parameter('ir_max_radius').value)
        self.ir_circularity_min = float(
            self.get_parameter('ir_circularity_min').value)
        self.ir_brightness_percentile = int(
            self.get_parameter('ir_brightness_percentile').value)
        self.hybrid_roi_margin = int(
            self.get_parameter('hybrid_roi_margin').value)

        self.hsv_lower = [int(v) for v in
                          self.get_parameter('hsv_lower').value]
        self.hsv_upper = [int(v) for v in
                          self.get_parameter('hsv_upper').value]
        self.color_match_threshold = float(
            self.get_parameter('color_match_threshold').value)
        self.color_circularity_min = float(
            self.get_parameter('color_circularity_min').value)
        self.camera_tilt_deg = self.get_parameter('camera_tilt_deg').value
        self.use_undistort = self.get_parameter('use_undistort').value
        self.quiet_vision_logs = bool(
            self.get_parameter('quiet_vision_logs').value)
        requested_print_every = int(self.get_parameter('print_every').value)
        self.print_every = 0 if self.quiet_vision_logs else max(
            0, requested_print_every)
        self.realsense_fps = self.get_parameter('realsense_fps').value
        self.use_realsense_hw_timestamp = bool(
            self.get_parameter('use_realsense_hw_timestamp').value)
        self.use_depth_sensor = bool(
            self.get_parameter('use_depth_sensor').value)
        self.depth_neighborhood_px = int(
            self.get_parameter('depth_neighborhood_px').value)
        # Set on the RealSense profile once the depth stream is opened; used to
        # convert z16 counts -> mm in the detector.
        self._depth_scale = 0.001
        # Clock-domain offset (ns) calibrated on the first frame when
        # use_realsense_hw_timestamp is on: offset_ns = ros_t_ns - rs_t_ms*1e6.
        # Then per-frame: ros_t_ns = rs_t_ms*1e6 + offset_ns. None => uncalibrated.
        self._rs_time_offset_ns = None
        # rs.align(color) instance + per-frame numpy depth handoff.
        self._rs_align = None

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

        # Ball-state estimator parameters (read once; the estimator is built
        # lazily on the first detector frame so it can match the detector's
        # output_frame and grab R_base_camera for anisotropic depth noise).
        self.enable_ball_state_estimator = bool(
            self.get_parameter('enable_ball_state_estimator').value)
        # The estimator and the FOV-loss compensator both fill in the ball
        # state across a toss; running both would propagate the arc twice
        # (once at 60 Hz in the vision callback, once at 200 Hz in the control
        # tick). The estimator is the more complete of the two, so it wins:
        # disable the FOV compensator whenever the estimator is on.
        if self.enable_ball_state_estimator and self.enable_fov_loss_compensation:
            self.get_logger().warn(
                'Both enable_ball_state_estimator and '
                'enable_fov_loss_compensation are true; these overlap and '
                'would double-propagate the toss. Disabling the FOV-loss '
                'compensator in favor of the ball-state estimator.')
            self.enable_fov_loss_compensation = False
        self.ball_estimator_max_extrapolation_s = float(
            self.get_parameter('ball_estimator_max_extrapolation_s').value)
        self.ball_estimator_trust_loss_s = float(
            self.get_parameter('ball_estimator_trust_loss_s').value)
        self.ball_estimator_degrade_loss_s = float(
            self.get_parameter('ball_estimator_degrade_loss_s').value)
        self.ball_estimator_invalid_after_s = float(
            self.get_parameter('ball_estimator_invalid_after_s').value)
        self.ball_estimator_prediction_horizon_s = float(
            self.get_parameter('ball_estimator_prediction_horizon_s').value)
        self.ball_estimator_enable_prediction = bool(
            self.get_parameter('ball_estimator_enable_prediction').value)
        self.ball_estimator_r_lateral_mm = float(
            self.get_parameter('ball_estimator_r_lateral_mm').value)
        self.ball_estimator_r_depth_mm = float(
            self.get_parameter('ball_estimator_r_depth_mm').value)
        self.ball_estimator_q_depth_pos_mult = float(
            self.get_parameter('ball_estimator_q_depth_pos_mult').value)
        self.ball_estimator_q_depth_vel_mult = float(
            self.get_parameter('ball_estimator_q_depth_vel_mult').value)
        self.ball_estimator_enable_aero_drag = bool(
            self.get_parameter('ball_estimator_enable_aero_drag').value)
        self.ball_estimator_drag_k_per_mm = float(
            self.get_parameter('ball_estimator_drag_k_per_mm').value)
        self.ball_estimator_enable_magnus = bool(
            self.get_parameter('ball_estimator_enable_magnus').value)
        self.ball_estimator_magnus_k = float(
            self.get_parameter('ball_estimator_magnus_k').value)
        self.ball_estimator_table_z_mm = float(
            self.get_parameter('ball_estimator_table_z_mm').value)
        self.ball_estimator_log_period_s = float(
            self.get_parameter('ball_estimator_log_period_s').value)
        self._ball_estimator = None
        self._ball_estimator_lock = threading.Lock()
        self._ball_estimator_log_last_wall = 0.0
        # 200 Hz estimator debug recorder.
        self.save_estimator_log = bool(
            self.get_parameter('save_estimator_log').value)
        self.estimator_log_path = str(
            self.get_parameter('estimator_log_path').value).strip()
        self._estimator_log_rows: list = []
        self._estimator_log_lock = threading.Lock()

        # 200 Hz RL policy obs/action debug recorder.
        self.save_policy_log = bool(
            self.get_parameter('save_policy_log').value)
        self.policy_log_path = str(
            self.get_parameter('policy_log_path').value).strip()
        self._policy_log_rows: list = []
        self._policy_log_lock = threading.Lock()
        self._policy_step_counter = 0

        # Most recent raw 60 Hz detection (position mm in detector output_frame).
        # Cached unconditionally so estimator logs and done-condition warnings
        # can compare the control/estimator state against the latest detector
        # measurement.
        self._latest_raw_detection = None

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
        self.rl_model_path = self._resolve_rl_model_path(self.rl_model_path)
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
        self.enable_delay_conditioning = bool(
            self.get_parameter('enable_delay_conditioning').value)
        self.policy_obs_dim = (
            EXPECTED_OBS_DIM_WITH_DELAY if self.enable_delay_conditioning
            else EXPECTED_OBS_DIM)
        self._configure_policy_observation_path()

        # Safety limiter enable flag
        self.enable_safety_limiter = bool(
            self.get_parameter('enable_safety_limiter').value)

        # Done condition monitoring parameters
        self.enable_done_monitoring = bool(
            self.get_parameter('enable_done_monitoring').value)
        self.done_ball_z_floor_m = float(
            self.get_parameter('done_ball_z_floor_m').value)
        self.done_ball_z_ceiling_m = float(
            self.get_parameter('done_ball_z_ceiling_m').value)
        self.done_ball_x_range_m = float(
            self.get_parameter('done_ball_x_range_m').value)
        self.done_ball_y_range_m = float(
            self.get_parameter('done_ball_y_range_m').value)
        self.done_racket_anchor_distance_m = float(
            self.get_parameter('done_racket_anchor_distance_m').value)
        self.done_racket_z_high_offset_m = float(
            self.get_parameter('done_racket_z_high_offset_m').value)
        self.done_racket_z_low_offset_m = float(
            self.get_parameter('done_racket_z_low_offset_m').value)
        # Parse workspace limits (empty list => None)
        ee_x_raw = self.get_parameter('done_ee_x_limits_m').value
        ee_y_raw = self.get_parameter('done_ee_y_limits_m').value
        ee_z_raw = self.get_parameter('done_ee_z_limits_m').value
        self.done_ee_x_limits_m = tuple(ee_x_raw) if len(ee_x_raw) == 2 else None
        self.done_ee_y_limits_m = tuple(ee_y_raw) if len(ee_y_raw) == 2 else None
        self.done_ee_z_limits_m = tuple(ee_z_raw) if len(ee_z_raw) == 2 else None
        self.done_log_period_s = float(
            self.get_parameter('done_log_period_s').value)

        # Replay mode parameters
        self.enable_replay_mode = bool(
            self.get_parameter('enable_replay_mode').value)
        self.replay_csv_path = self.get_parameter('replay_csv_path').value
        self.replay_loop = bool(self.get_parameter('replay_loop').value)
        self.replay_data_is_sim_frame = bool(
            self.get_parameter('replay_data_is_sim_frame').value)
        self.replay_auto_exit = bool(self.get_parameter('replay_auto_exit').value)
        self._replay_data = None
        self._replay_idx = 0
        self._replay_start_time = None
        self._last_replay_estimator_idx = None

        # Simulation ball state input parameters
        self.enable_sim_ball_input = bool(
            self.get_parameter('enable_sim_ball_input').value)
        self.sim_ball_topic = self.get_parameter('sim_ball_topic').value
        self._sim_ball_state = None
        self._sim_ball_lock = threading.Lock()

        # Motion capture ball state input parameters
        self.enable_mocap_input = bool(
            self.get_parameter('enable_mocap_input').value)
        self.mocap_bind_host = str(
            self.get_parameter('mocap_bind_host').value)
        self.mocap_udp_port = int(
            self.get_parameter('mocap_udp_port').value)
        self.mocap_calibration_path = self.get_parameter(
            'mocap_calibration_path').value
        self.mocap_timeout_s = float(
            self.get_parameter('mocap_timeout_s').value)
        self.mocap_marker_id = int(
            self.get_parameter('mocap_marker_id').value)
        self.mocap_position_unit = str(
            self.get_parameter('mocap_position_unit').value)
        self.mocap_timestamp_unit = str(
            self.get_parameter('mocap_timestamp_unit').value)
        self.mocap_velocity_alpha = float(
            self.get_parameter('mocap_velocity_alpha').value)
        self.mocap_max_velocity_m_s = float(
            self.get_parameter('mocap_max_velocity_m_s').value)
        self._mocap_detector = None
        self._mocap_ball_state = None
        self._mocap_ball_lock = threading.Lock()
        self._last_mocap_estimator_packet_count = 0
        self._last_mocap_recorded_packet_count = 0

        # Ball-lost failsafe parameters.
        self.enable_ball_lost_failsafe = bool(
            self.get_parameter('enable_ball_lost_failsafe').value)
        self.enable_ball_capture_gate = bool(
            self.get_parameter('enable_ball_capture_gate').value)
        self.enable_ball_release_gate = bool(
            self.get_parameter('enable_ball_release_gate').value)
        self.ball_release_min_downward_velocity_m_s = float(
            self.get_parameter(
                'ball_release_min_downward_velocity_m_s').value)
        self.ball_release_min_z_drop_m = float(
            self.get_parameter('ball_release_min_z_drop_m').value)
        self.ball_release_window_s = max(
            0.02, float(self.get_parameter('ball_release_window_s').value))
        self.ball_release_required_consecutive_ticks = max(
            1, int(self.get_parameter(
                'ball_release_required_consecutive_ticks').value))
        self.ball_release_max_obs_age_s = max(
            0.0, float(self.get_parameter(
                'ball_release_max_obs_age_s').value))
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
        self._has_captured_valid_ball = False
        self._ball_capture_gate_active = False
        self._has_released_ball = False
        self._ball_release_gate_active = False
        self._ball_release_candidate_ticks = 0
        self._ball_release_samples = deque()
        self._ball_release_gate_log_last_wall = 0.0
        self._ball_release_last_debug = {}
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
                            enable_delay_conditioning=self.enable_delay_conditioning,
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
                            enable_delay_conditioning=self.enable_delay_conditioning,
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

                    # Initialize actuator delay compensator if enabled
                    self._init_compensator()

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
        if self.enable_safety_limiter:
            self.get_logger().info(
                f'Safety limiter ENABLED: dt={self.control_dt:.4f}s '
                f'(control_rate_hz={self.control_rate_hz:.1f})')
        else:
            self.get_logger().warn(
                f'Safety limiter DISABLED: raw RL commands will be published directly! '
                f'Only use in simulation.')
        self._safety_log_period_s = 1.0
        self._safety_log_last_wall = time.time()

        # Coordinate frame transform: real Cartesian frame -> simulation frame.
        #
        # Transform formula (from code analysis):
        #   Simulation (mjx_juggle_env._point_to_base):
        #     Returns [x_rel, y_rel, z_world] - z is world absolute height
        #   Real system (via T_base_camera):
        #     Returns [x_rel, y_rel, z_rel] relative to the robot Cartesian
        #     frame; on this robot that frame corresponds to moz1_pd.xml's
        #     waist02 body.
        #   Therefore: p_sim = p_waist02 + waist02_origin_in_sim_base
        #
        # Cartesian-frame origin parameters:
        #   waist02 origin in the simulation base-yaw frame when the robot is in
        #   TARGET_DEGREES posture [0, 60, -90, 30, 0, 0].
        #
        #   Computed via forward kinematics from moz1_pd.xml + TARGET_DEGREES:
        #   - The real Cartesian frame is the waist02 body.
        #   - In TARGET_DEGREES posture:
        #       waist02 relative to sim base = [0.0, 23.583, 847.396] mm
        #       waist02 world height       = 0.947396 m (base z is 0.1 m)
        #
        #   Coordinate transformation:
        #     x_sim = x_real + 0.0 m
        #     y_sim = y_real + 0.023583 m
        #     z_sim = z_real + 0.947396 m
        #
        #   IMPORTANT: This value is posture-dependent. If robot posture changes
        #   from TARGET_DEGREES, this parameter must be recalibrated.
        self.declare_parameter('sim_cartesian_x_offset_m', 0.0)
        self.declare_parameter('sim_cartesian_y_offset_m', 0.023583)
        self.declare_parameter('sim_base_height_m', 0.947396)
        self._sim_cartesian_x_offset_m = float(
            self.get_parameter('sim_cartesian_x_offset_m').value)
        self._sim_cartesian_y_offset_m = float(
            self.get_parameter('sim_cartesian_y_offset_m').value)
        self._sim_base_height_m = float(
            self.get_parameter('sim_base_height_m').value)

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
        if self.enable_ball_capture_gate:
            self.get_logger().info(
                'Ball-capture gate enabled: RL stays parked at safe pose until '
                'the first valid ball state is available')
        if self.enable_ball_release_gate:
            self.get_logger().info(
                'Ball-release gate enabled: after capture, RL stays parked '
                'until fresh downward ball motion is observed '
                f'(vz<-{self.ball_release_min_downward_velocity_m_s:.2f}m/s, '
                f'z_drop>{self.ball_release_min_z_drop_m:.3f}m over '
                f'{self.ball_release_window_s:.2f}s, '
                f'{self.ball_release_required_consecutive_ticks} ticks, '
                f'age<={self.ball_release_max_obs_age_s:.3f}s)')
        else:
            self.get_logger().info(
                'Ball-release gate disabled (enable_ball_release_gate=False)')

        # Done condition monitor: replicate sim termination conditions
        self.done_monitor = None
        self._done_failsafe_active = False
        self._done_log_last_wall = time.time()
        if self.enable_done_monitoring:
            self.done_monitor = DoneConditionMonitor(
                ball_z_floor=self.done_ball_z_floor_m,
                ball_z_ceiling=self.done_ball_z_ceiling_m,
                ball_x_range=self.done_ball_x_range_m,
                ball_y_range=self.done_ball_y_range_m,
                racket_anchor_distance_limit=self.done_racket_anchor_distance_m,
                racket_z_high_offset=self.done_racket_z_high_offset_m,
                racket_z_low_offset=self.done_racket_z_low_offset_m,
                ee_x_limits=self.done_ee_x_limits_m,
                ee_y_limits=self.done_ee_y_limits_m,
                ee_z_limits=self.done_ee_z_limits_m,
            )
            ee_limits_msg = []
            if self.done_ee_x_limits_m:
                ee_limits_msg.append(f'x={self.done_ee_x_limits_m}m')
            if self.done_ee_y_limits_m:
                ee_limits_msg.append(f'y={self.done_ee_y_limits_m}m')
            if self.done_ee_z_limits_m:
                ee_limits_msg.append(f'z={self.done_ee_z_limits_m}m')
            ee_limits_str = ', '.join(ee_limits_msg) if ee_limits_msg else 'none'
            self.get_logger().info(
                'Done-condition monitoring ENABLED: '
                f'ball_z=[{self.done_ball_z_floor_m}, {self.done_ball_z_ceiling_m}]m, '
                f'ball_xy_range={self.done_ball_x_range_m}m, '
                f'racket_anchor_dist<{self.done_racket_anchor_distance_m}m, '
                f'EE workspace limits: {ee_limits_str}')
        else:
            self.get_logger().info(
                'Done-condition monitoring DISABLED (enable_done_monitoring=False)')

        # Standalone racket-anchor FK for the viz done-condition overlay, so the
        # sim-frame ball state + done status show even with enable_rl_policy:=false.
        self._init_viz_fk()

        # Init pose state (whole-body drive to TARGET_DEGREES before RL runs).
        self.enable_init_pose = bool(
            self.get_parameter('enable_init_pose').value)
        self.init_pose_duration_s = float(
            self.get_parameter('init_pose_duration_s').value)
        self.require_enter_confirm = bool(
            self.get_parameter('require_enter_confirm').value)
        self.start_confirm_topic = str(
            self.get_parameter('start_confirm_topic').value).strip()
        if not self.start_confirm_topic:
            self.start_confirm_topic = '/pingpong/start_control'
        self.auto_continue_on_stdin_eof = bool(
            self.get_parameter('auto_continue_on_stdin_eof').value)
        self.enter_prompt_log_period_s = max(0.1, float(
            self.get_parameter('enter_prompt_log_period_s').value))
        self.post_enter_start_delay_s = max(0.0, float(
            self.get_parameter('post_enter_start_delay_s').value))
        self.run_duration_s = max(0.0, float(
            self.get_parameter('run_duration_s').value))
        self.shutdown_after_run_duration = bool(
            self.get_parameter('shutdown_after_run_duration').value)
        self.shutdown_after_run_duration_delay_s = max(0.0, float(
            self.get_parameter('shutdown_after_run_duration_delay_s').value))
        self.right_arm_mu_idx = int(
            self.get_parameter('right_arm_mu_idx').value)
        self.left_arm_mu_idx = int(
            self.get_parameter('left_arm_mu_idx').value)
        self.leg_waist_mu_idx = int(
            self.get_parameter('leg_waist_mu_idx').value)
        self.head_mu_idx = int(self.get_parameter('head_mu_idx').value)
        self.warn_no_command_subscriber = bool(
            self.get_parameter('warn_no_command_subscriber').value)
        self.command_subscriber_warn_period_s = max(
            0.5, float(
                self.get_parameter('command_subscriber_warn_period_s').value))
        self.get_logger().info(
            f'MechUnitCmdArray -> "{command_topic}" '
            f'(right={self.right_arm_mu_idx}, left={self.left_arm_mu_idx}, '
            f'leg_waist={self.leg_waist_mu_idx}, head={self.head_mu_idx}); '
            'set mu_idx params to real unit ids before deployment')
        self._init_pose_active = bool(
            self.enable_init_pose and self.init_pose_duration_s > 0.0)
        self._init_pose_start_wall: Optional[float] = None
        self._init_pose_start_q: Optional[np.ndarray] = None
        self._init_pose_start_source = ''
        # After the init-pose ramp completes, optionally hold at the start pose
        # and wait for the operator to press Enter before starting RL (mirrors
        # pingpong_replay's require_enter_confirm gate). A daemon thread reads
        # stdin and sets an Event; the timer polls it non-blockingly.
        self._await_enter = bool(
            self.enable_init_pose and self.require_enter_confirm)
        self._enter_event = threading.Event()
        self._enter_event_source = ''
        self._enter_thread_started = False
        self._enter_prompted = False
        self._enter_prompt_last_wall = 0.0
        self._stdin_is_tty = bool(getattr(sys.stdin, 'isatty', lambda: False)())
        self._start_confirm_ignored_last_wall = 0.0
        self._post_enter_delay_until_wall: Optional[float] = None
        self._formal_run_start_wall: Optional[float] = None
        self._run_duration_elapsed = False
        self._run_duration_shutdown_started = False
        self._last_done_debug = {}
        if self._init_pose_active:
            self.get_logger().info(
                f'Init-pose phase enabled: smoothly driving right arm to TARGET for '
                f'{self.init_pose_duration_s:.2f}s before starting RL policy')
        else:
            self.get_logger().info(
                'Init-pose phase disabled (enable_init_pose=False or '
                'init_pose_duration_s<=0); RL policy starts immediately')
        if self._await_enter and self.post_enter_start_delay_s > 0.0:
            self.get_logger().info(
                f'After Enter confirmation, holding init pose for '
                f'{self.post_enter_start_delay_s:.2f}s before formal start')
        if self._await_enter:
            self.get_logger().info(
                f'Start confirmation enabled: press Enter in this terminal '
                f'or publish std_msgs/Empty to "{self.start_confirm_topic}" '
                'after the init pose is reached')
        if self.run_duration_s > 0.0:
            self.get_logger().info(
                f'Run duration limit enabled: formal control will run for '
                f'{self.run_duration_s:.2f}s, then park at safe init pose')

        # Latest BallState cached by image callbacks; control timer consumes it.
        self._latest_ball_state = None
        self._latest_ball_lock = threading.Lock()

        # Replay mode: load CSV data if enabled
        if self.enable_replay_mode:
            self._init_replay_mode()

        # Simulation ball state input: subscribe to ball state from simulation
        if self.enable_sim_ball_input:
            self._init_sim_ball_input()

        # Motion capture ball state input: start mocap receiver
        if self.enable_mocap_input:
            self._init_mocap_input()

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
        # Initialize to zeros; may be overridden by replay mode initialization
        if not hasattr(self, '_estimated_right_arm_dq'):
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
        self.start_confirm_sub = None
        if self._await_enter:
            self.start_confirm_sub = self.create_subscription(
                EmptyMsg, self.start_confirm_topic,
                self._start_confirm_callback, 10,
                callback_group=self.state_callback_group)
            self.get_logger().info(
                f'Subscribed to start confirmation: "{self.start_confirm_topic}"')
            if not self._stdin_is_tty and not self.auto_continue_on_stdin_eof:
                self.get_logger().warn(
                    'stdin is not an interactive TTY; Enter confirmation will '
                    f'not work here. Start with: ros2 topic pub --once '
                    f'{self.start_confirm_topic} std_msgs/msg/Empty "{{}}"')

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

        # --- Sim-frame ball state (for visualization done-condition overlay) ---
        # Stashed by _compute_rl_command each control tick: the ball state in
        # meters, sim base frame (MJX _point_to_base convention: base yaw + world z),
        # that was passed into rl_controller.predict(). The viz HUD reads these to
        # display sim-frame pos/vel and evaluate the sim done-conditions so the user
        # can see when the ball would terminate a training episode.
        self._last_sim_ball_pos_m = None
        self._last_sim_ball_vel_m_s = None
        self._last_sim_ball_valid = False

        # --- Publisher (created before any thread starts) ---
        # MechUnitCmdArray publisher on command_topic (default
        # /mx_mix_command). The real downstream (MCControlNode) subscribes to
        # MechUnitCmdArray — a header plus a list of per-unit MechUnitCmd — so
        # a bare MechUnitCmd is never received. We wrap every command (RL phase:
        # right arm only; init phase: all four mechanical units) in an array.
        # This mirrors the on-hardware-verified pingpong_replay publisher.
        self.cmd_pub = self.create_publisher(
            MechUnitCmdArray, command_topic, 10)
        self._last_cmd_pub_no_sub_warn_wall = 0.0

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
        # Aligned-depth recording sidecars (see save_depth_frames). The depth
        # frames go into a sibling directory of the raw mp4 and per-frame
        # timestamps into a CSV; both keyed by the same frame index the RGB
        # video uses so the offline tool pairs them positionally.
        self._depth_frames_dir = None
        self._depth_frames_written = 0
        self._timestamps_csv_path = None
        self._timestamps_file = None
        self._recording_index = 0  # shared RGB/depth/timestamp frame counter
        self._depth_disabled_runtime = False
        # IR-frame recording sidecar (hybrid mode; see save_ir_frames). Mirrors
        # the depth sidecar: one ir_<idx>.npy per RGB frame + a one-shot
        # camera_params JSON, keyed by the same _recording_index.
        self._ir_frames_dir = None
        self._ir_frames_written = 0
        self._ir_disabled_runtime = False
        self._camera_params_written = False

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
        # Depth recording rides on the raw-video path and needs hardware depth.
        if self.save_depth_frames and not (
                self.save_raw_video and self.use_depth_sensor):
            self.get_logger().warn(
                'save_depth_frames=true requires save_raw_video=true and '
                'use_depth_sensor=true (direct_realsense mode). Disabling '
                'depth-frame recording.')
            self.save_depth_frames = False
        # IR recording rides on the raw-video path and needs the hybrid detector
        # (that is what owns the IR stream and the intrinsics/extrinsics).
        if self.save_ir_frames and not (
                self.save_raw_video and self.use_hybrid_detector):
            self.get_logger().warn(
                'save_ir_frames=true requires save_raw_video=true and '
                'use_hybrid_detector=true (direct_realsense mode). Disabling '
                'IR-frame recording.')
            self.save_ir_frames = False
        if self.save_raw_video:
            raw_fps_text = (
                f'{self.raw_video_fps:.1f}'
                if self.raw_video_fps > 0.0 else 'auto (realsense_fps)')
            self.get_logger().info(
                'Raw video recording enabled: saving unprocessed RealSense '
                f'RGB stream (fps={raw_fps_text}); file resolved on first '
                'frame. Post-process with tools/annotate_raw_video.py.')
        if self.save_depth_frames:
            self.get_logger().info(
                'Depth-frame recording enabled: saving aligned 16-bit depth '
                'frames (.npy) + per-frame timestamp CSV alongside the raw '
                'video, for offline hardware-depth replay validation.')
        if self.save_ir_frames:
            self.get_logger().info(
                'IR-frame recording enabled: saving IR (infrared-1) frames '
                '(.npy) + a camera_params JSON alongside the raw video, for '
                'offline hybrid detection with '
                'tools/annotate_raw_video_hybrid.py.')

        # --- Branch on input mode ---
        if self.use_direct_realsense:
            self.get_logger().info('Image input mode: direct_realsense')
            self.get_logger().info(
                'Note: realsense2_camera ROS driver should NOT be running '
                'when use_direct_realsense=True')
            if self.use_hybrid_detector:
                self.get_logger().info(
                    'Using HybridBallDetector: camera managed by detector, '
                    'starting hybrid detector reader loop')
                self._init_hybrid_detector_loop()
            else:
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
        self.get_logger().info(
            f'Ellipse-fit radius: {self.enable_ellipse_fit}, '
            f'Temporal tracker: {self.enable_temporal_tracker} '
            f'(bridge<={self.tracker_max_miss_bridge}, '
            f'gate={self.tracker_gate_px:.0f}px)')
        self.get_logger().info(
            f'Color gate: HSV {self.hsv_lower}-{self.hsv_upper}, '
            f'match>={self.color_match_threshold:.2f}, '
            f'circularity>={self.color_circularity_min:.2f}')
        if self.print_every > 0:
            self.get_logger().info(
                f'Print every: {self.print_every} frames')
        else:
            self.get_logger().info(
                'Per-frame vision logs disabled '
                '(quiet_vision_logs=true or print_every<=0)')
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
        detector_type = "HybridBallDetector" if self.use_hybrid_detector else "PingPongBallDetector"
        self.get_logger().info(
            f'Initializing {detector_type} with CalibrationData...')
        try:
            if self.use_hybrid_detector:
                # 使用Hybrid检测器（灰色反光球）
                self.detector = HybridBallDetectorAdapter(
                    yolo_model_path=self.yolo_model_path,
                    ball_diameter_mm=40.0,
                    width=self.calibration_data.image_width,
                    height=self.calibration_data.image_height,
                    fps=60,
                    yolo_interval=self.yolo_interval,
                    yolo_conf=self.conf_threshold,
                    target_class=self.target_class,
                    ir_params_json=self.ir_params_json,  # 优先从JSON加载
                    ir_threshold=self.ir_threshold,
                    use_adaptive=self.use_adaptive_ir,
                    min_radius=self.ir_min_radius,
                    max_radius=self.ir_max_radius,
                    circularity_min=self.ir_circularity_min,
                    laser_power=self.ir_laser_power,
                    exposure=self.ir_exposure,
                    gain=self.ir_gain,
                    brightness_percentile=self.ir_brightness_percentile,
                    roi_margin=self.hybrid_roi_margin,
                    calibration_data=self.calibration_data,
                    output_frame='base_link',
                    print_every=999999,
                    use_undistort=True,
                    depth_scale=self._depth_scale,
                    depth_correction_model_path=self.depth_correction_model_path,  # 新增
                    depth_neighborhood_px=self.depth_neighborhood_px,
                    # 灰色球禁用颜色筛选
                    color_match_threshold=0.0,
                    color_circularity_min=0.0,
                )
            else:
                # 使用传统检测器（橙色球）
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
                    output_frame='base_link',
                    enable_ellipse_fit=self.enable_ellipse_fit,
                    enable_temporal_tracker=self.enable_temporal_tracker,
                    tracker_max_miss_bridge=self.tracker_max_miss_bridge,
                    tracker_gate_px=self.tracker_gate_px,
                    hsv_lower=self.hsv_lower,
                    hsv_upper=self.hsv_upper,
                    color_match_threshold=self.color_match_threshold,
                    color_circularity_min=self.color_circularity_min,
                    use_depth_sensor=self.use_depth_sensor,
                    depth_scale=self._depth_scale,
                    depth_neighborhood_px=self.depth_neighborhood_px,
                )
            self.camera_info_received = True
            self.get_logger().info(
                f'{detector_type} initialized (output: base_link)')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize detector: {e}')
            raise

    def _init_detector_from_realsense_intrinsics(self, intrinsics):
        """Initialize detector using intrinsics from RealSense profile."""
        detector_type = "HybridBallDetector" if self.use_hybrid_detector else "PingPongBallDetector"
        self.get_logger().info(
            f'Initializing {detector_type} with RealSense intrinsics...')
        dist_coeffs = list(intrinsics.coeffs)
        try:
            if self.use_hybrid_detector:
                # Hybrid检测器（灰色球）
                # 注意：hybrid_detector内部会直接从RealSense读取IR和depth
                # 这里不需要传camera_fx等参数，它会自己获取
                self.detector = HybridBallDetectorAdapter(
                    yolo_model_path=self.yolo_model_path,
                    ball_diameter_mm=40.0,
                    width=intrinsics.width,
                    height=intrinsics.height,
                    fps=self.realsense_fps,
                    yolo_interval=self.yolo_interval,
                    yolo_conf=self.conf_threshold,
                    target_class=self.target_class,
                    ir_params_json=self.ir_params_json,  # 优先从JSON加载
                    ir_threshold=self.ir_threshold,
                    use_adaptive=self.use_adaptive_ir,
                    min_radius=self.ir_min_radius,
                    max_radius=self.ir_max_radius,
                    circularity_min=self.ir_circularity_min,
                    laser_power=self.ir_laser_power,
                    exposure=self.ir_exposure,
                    gain=self.ir_gain,
                    brightness_percentile=self.ir_brightness_percentile,
                    roi_margin=self.hybrid_roi_margin,
                    calibration_data=None,  # camera模式不用标定
                    output_frame='camera',
                    print_every=999999,
                    use_undistort=self.use_undistort,
                    depth_scale=self._depth_scale,
                    depth_correction_model_path=self.depth_correction_model_path,
                    depth_neighborhood_px=self.depth_neighborhood_px,
                    # 灰色球禁用颜色筛选
                    color_match_threshold=0.0,
                    color_circularity_min=0.0,
                )
            else:
                # 传统检测器（橙色球）
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
                    output_frame='camera',
                    enable_ellipse_fit=self.enable_ellipse_fit,
                    enable_temporal_tracker=self.enable_temporal_tracker,
                    tracker_max_miss_bridge=self.tracker_max_miss_bridge,
                    tracker_gate_px=self.tracker_gate_px,
                    hsv_lower=self.hsv_lower,
                    hsv_upper=self.hsv_upper,
                    color_match_threshold=self.color_match_threshold,
                    color_circularity_min=self.color_circularity_min,
                    use_depth_sensor=self.use_depth_sensor,
                    depth_scale=self._depth_scale,
                    depth_neighborhood_px=self.depth_neighborhood_px,
                )
            self.camera_info_received = True
            self.get_logger().info(
                f'{detector_type} initialized (output: camera)')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize detector: {e}')
            raise

    # ------------------------------------------------------------------
    # Direct RealSense mode
    # ------------------------------------------------------------------

    def _start_color_stream(self, width, height, fps,
                            allow_resolution_fallback=False,
                            enable_depth=False):
        """Start RealSense color stream with fallback modes.

        Args:
            allow_resolution_fallback: If False, only fps fallback is allowed
                (resolution stays at width x height). If True, 640x480 is
                tried as a last resort.
            enable_depth: Also enable the depth stream at the same resolution
                (for use_depth_sensor). The depth stream is added to each
                candidate config so it shares the color stream's fallback.
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
            if enable_depth:
                # Depth at the same WxH so align(color) is a clean 1:1 map.
                config.enable_stream(rs.stream.depth, w, h, rs.format.z16, f)
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
            allow_resolution_fallback=allow_fallback,
            enable_depth=self.use_depth_sensor)

        # Set up depth->color alignment and read the depth scale (z16 -> m).
        if self.use_depth_sensor:
            try:
                self._rs_align = rs.align(rs.stream.color)
                depth_sensor = profile.get_device().first_depth_sensor()
                self._depth_scale = float(depth_sensor.get_depth_scale())
                # The calibration-mode detector is built in __init__ (before
                # this runs) with the default scale; push the real device scale
                # and enable flag onto it now so they can't drift apart.
                if self.detector is not None:
                    self.detector.use_depth_sensor = True
                    self.detector.depth_scale = self._depth_scale
                    self.detector.depth_neighborhood_px = \
                        self.depth_neighborhood_px
                self.get_logger().info(
                    f'Hardware depth ENABLED: depth_scale={self._depth_scale} '
                    f'm/count, aligned to color, neighborhood='
                    f'{self.depth_neighborhood_px}px')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f'Failed to set up depth alignment ({exc}); '
                    'disabling hardware depth, falling back to monocular')
                self.use_depth_sensor = False
                self._rs_align = None
                if self.detector is not None:
                    self.detector.use_depth_sensor = False

        color_stream = profile.get_stream(rs.stream.color)
        video_profile = color_stream.as_video_stream_profile()
        rs_intrinsics = video_profile.get_intrinsics()
        self.get_logger().info(
            f'RealSense stream: {rs_intrinsics.width}x{rs_intrinsics.height}'
            f'@{video_profile.fps()}fps')
        if self.use_realsense_hw_timestamp:
            self.get_logger().info(
                'Timestamp mode: RealSense hardware exposure time (offset '
                'calibrated on first frame; includes exposure+USB latency in '
                'perception de-aging).')
        else:
            self.get_logger().info(
                'Timestamp mode: ROS reader-loop clock (exposure+USB latency '
                'NOT included in perception de-aging).')

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

            # Align depth to color and extract the depth image when hardware
            # depth is enabled. align.process gives a frameset whose depth is
            # registered to the color pixels (same WxH).
            depth_image = None
            if self.use_depth_sensor and self._rs_align is not None:
                try:
                    aligned = self._rs_align.process(frames)
                    depth_frame = aligned.get_depth_frame()
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data())
                except RuntimeError:
                    depth_image = None  # transient; fall back to monocular

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            if color_image is None or color_image.size == 0:
                continue
            if len(color_image.shape) != 3:
                continue

            # One capture timestamp for this frame, shared by the recorder
            # sidecar and the detection path so the offline replay times line
            # up with what the live node actually saw. When use_realsense_hw_
            # timestamp is on, use the color frame's hardware timestamp
            # (exposure ms) and map into ROS clock domain via the first-frame
            # calibrated offset; this captures the true image-capture instant
            # and includes exposure+USB latency in latency de-aging. When off
            # or the HW time is unavailable, fall back to the reader-loop clock.
            stamp = self._stamp_frame(color_frame)

            # Record the raw frame BEFORE detection/visualization touch it, so
            # the saved video is the true native-rate RealSense stream. This
            # only enqueues (a non-blocking O(1) handoff to the raw writer
            # thread); the actual encode happens off this thread. When depth
            # recording is on, RGB+depth+timestamp travel as one bundle so a
            # dropped frame drops the whole bundle and indices never desync.
            if self.save_raw_video and not self._raw_video_disabled_runtime:
                rec_depth = (depth_image if self.save_depth_frames
                             and not self._depth_disabled_runtime else None)
                t_capture = stamp.sec + stamp.nanosec * 1e-9
                self._enqueue_raw_video_frame(
                    color_image, depth_image=rec_depth, t_capture=t_capture)

            self._handle_cv_image(color_image, stamp, depth_image=depth_image)

    def _stamp_frame(self, color_frame) -> Time:
        """Return the frame timestamp: hardware exposure time or reader-loop clock.

        When ``use_realsense_hw_timestamp`` is on, reads the frame's hardware
        timestamp (ms), maps it to ROS clock (ns) via the offset calibrated on
        the first frame, and returns a Time message. The offset is calibrated
        by taking the first frame's hardware timestamp and the node clock at
        that instant, computing offset = ros_t_ns - (rs_t_ms * 1e6). Every
        subsequent frame is then ros_t_ns = rs_t_ms*1e6 + offset. This bakes in
        drift over the entire run, but the alternative (per-frame clock reads
        for offset) defeats the purpose. If the hardware timestamp is
        unavailable or the feature is off, falls back to the reader-loop clock.
        """
        if self.use_realsense_hw_timestamp:
            try:
                hw_ms = color_frame.get_timestamp()
                if self._rs_time_offset_ns is None:
                    # First-frame calibration: offset = ROS now - HW now.
                    ros_now_ns = self.get_clock().now().nanoseconds
                    self._rs_time_offset_ns = ros_now_ns - int(hw_ms * 1e6)
                    self.get_logger().info(
                        f'RealSense HW timestamp enabled: first-frame '
                        f'calibration offset = {self._rs_time_offset_ns} ns. '
                        f'Hardware exposure time is now used for detection '
                        f'timestamping (includes exposure+USB latency in '
                        f'perception de-aging).')
                ros_ns = int(hw_ms * 1e6) + self._rs_time_offset_ns
                return Time(nanoseconds=ros_ns)
            except Exception:  # noqa: BLE001 - fallback on any RS API failure
                pass
        # Fallback: the reader-loop clock (as before).
        return self.get_clock().now().to_msg()

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

    def _init_hybrid_detector_loop(self):
        """Initialize reader loop for HybridBallDetector (camera managed by detector)."""
        # HybridBallDetector已经启动了RealSense，我们只需要启动reader loop
        # 从detector的get_observation()读取帧和检测结果

        self.get_logger().info('Starting HybridBallDetector reader loop...')

        # 启动reader线程
        self._rs_running = True
        self._rs_thread = threading.Thread(
            target=self._hybrid_reader_loop, daemon=True)
        self._rs_thread.start()

        self.get_logger().info('HybridBallDetector reader loop started')

    def _hybrid_reader_loop(self):
        """Reader loop for HybridBallDetector: get observations and feed to vision callback."""
        self.get_logger().info('Hybrid reader loop: entering main loop')

        # 直接访问HybridBallDetector的pipeline
        pipeline = self.detector.detector.pipeline

        loop_count = 0
        while self._rs_running:
            try:
                loop_count += 1

                # 等待新帧（直接从pipeline，不通过HybridBallDetector）
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as e:
                    self.get_logger().warn(f'wait_for_frames timeout: {e}', throttle_duration_sec=1.0)
                    continue

                # 获取RGB和IR图像
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame(1)

                if not color_frame:
                    continue

                rgb_image = np.asanyarray(color_frame.get_data())
                if rgb_image is None or rgb_image.size == 0:
                    continue

                ir_image = None
                if ir_frame:
                    ir_image = np.asanyarray(ir_frame.get_data())

                # 创建时间戳
                stamp = self.get_clock().now().to_msg()
                # One capture time (s) shared by the recorder sidecar (t_capture
                # -> timestamps.csv) and the detector. Passing it into
                # get_observation makes the detector compute velocity from the
                # SAME capture-time series that the offline tool replays, so the
                # record->annotate and live-juggle paths derive velocity
                # identically instead of the live path using wall-clock
                # (time.time()) at processing time.
                stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9

                # 保存原始视频（hybrid 模式额外存 IR/depth，供离线混合检测）
                if self.save_raw_video and not self._raw_video_disabled_runtime:
                    t_capture = stamp_sec
                    rec_depth = None
                    if self.save_depth_frames:
                        depth_frame = frames.get_depth_frame()
                        if depth_frame:
                            rec_depth = np.asanyarray(depth_frame.get_data())
                    rec_ir = ir_image if self.save_ir_frames else None
                    self._enqueue_raw_video_frame(
                        rgb_image, depth_image=rec_depth, ir_image=rec_ir,
                        t_capture=t_capture)

                # 调用HybridBallDetector的get_observation处理这一帧
                observation = self.detector.detector.get_observation(
                    observation_time=stamp_sec, frames=frames)

                if observation is None:
                    observation = {'detected': False, 'position': [0, 0, 0], 'velocity': [0, 0, 0]}

                # Hybrid可视化
                if self.enable_visualization and ir_image is not None:
                    try:
                        viz_image = self.detector.detector._render_visualization(rgb_image, ir_image, observation)
                        if viz_image is not None:
                            # 直接发送渲染好的图像到可视化worker
                            # 创建一个假的observation，标记为hybrid_viz_rendered
                            hybrid_obs_for_viz = observation.copy()
                            hybrid_obs_for_viz['_hybrid_viz_rendered'] = True
                            self._publish_visualization_frame(viz_image, hybrid_obs_for_viz)
                    except Exception as e:
                        self.get_logger().error(f'Visualization error: {e}', throttle_duration_sec=5.0)

                # 传递给后处理流程
                self._handle_hybrid_cv_image(rgb_image, stamp, observation)

                if loop_count == 1 and not self.quiet_vision_logs:
                    self.get_logger().info('Hybrid reader loop: first frame processed!')

                if loop_count % 100 == 0 and not self.quiet_vision_logs:
                    self.get_logger().info(f'Hybrid reader loop: {loop_count} frames processed')

            except Exception as e:
                self.get_logger().error(f'Hybrid reader loop error: {e}', throttle_duration_sec=1.0)
                import traceback
                self.get_logger().error(f'Traceback: {traceback.format_exc()}', throttle_duration_sec=5.0)
                time.sleep(0.1)
                continue

        self.get_logger().info('Hybrid reader loop: exited')

    def _handle_hybrid_cv_image(self, image, stamp, hybrid_observation):
        """处理HybridBallDetector的结果，复用原有的后处理流程。

        与_handle_cv_image的区别：
        - 不调用detector.get_observation()（已经有结果了）
        - 直接使用hybrid_observation
        - 其他流程（estimator、FOV补偿、可视化、记录）完全相同
        """
        now_wall = time.time()
        if self.last_callback_time is not None:
            dt = now_wall - self.last_callback_time
            self.callback_times.append(dt)
            if len(self.callback_times) > 30:
                self.callback_times.pop(0)
        self.last_callback_time = now_wall

        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        observation = hybrid_observation  # 直接使用hybrid检测结果

        # Optional ball-state estimator
        if self.enable_ball_state_estimator:
            self._feed_ball_state_estimator(observation, stamp_sec)

        # Optional FOV-loss compensation
        if self.enable_fov_loss_compensation:
            observation = self._apply_fov_loss_compensation(
                observation, stamp_sec)

        self.frame_count += 1

        if observation['detected']:
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

        if self.print_every > 0 and self.frame_count % self.print_every == 0:
            self._log_detection_results(ball_state)

        # Publish ball state debug topic
        self._publish_ball_state_debug(ball_state)

        # 可视化已经在_hybrid_reader_loop中处理了，这里跳过
        # （hybrid检测器有自己的渲染逻辑，显示RGB+IR双图）

        # Cache latest BallState
        with self._latest_ball_lock:
            self._latest_ball_state = ball_state

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
                output_frame=output_frame,
                enable_ellipse_fit=self.enable_ellipse_fit,
                enable_temporal_tracker=self.enable_temporal_tracker,
                tracker_max_miss_bridge=self.tracker_max_miss_bridge,
                tracker_gate_px=self.tracker_gate_px,
                hsv_lower=self.hsv_lower,
                hsv_upper=self.hsv_upper,
                color_match_threshold=self.color_match_threshold,
                color_circularity_min=self.color_circularity_min,
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

    def _handle_cv_image(self, image: np.ndarray, stamp: Time,
                         depth_image: np.ndarray = None):
        """Process a BGR image through the detector and update stats/logs.

        depth_image: optional aligned depth frame (direct_realsense +
        use_depth_sensor). None in ROS-topic mode or when hardware depth is
        off; the detector then uses monocular geometry.
        """
        now_wall = time.time()
        if self.last_callback_time is not None:
            dt = now_wall - self.last_callback_time
            self.callback_times.append(dt)
            if len(self.callback_times) > 30:
                self.callback_times.pop(0)
        self.last_callback_time = now_wall

        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        observation = self.detector.get_observation(
            image, observation_time=stamp_sec, depth_image=depth_image)

        # Optional ball-state estimator: fuse this REAL detection, stamped at
        # its image time (stamp_sec == t_img). Fed before FOV compensation so
        # the filter only ever ingests genuine measurements, never the FOV
        # compensator's own ballistic predictions. The 200 Hz control loop
        # reads predict_to() separately; here we only push the measurement in.
        # Default OFF.
        if self.enable_ball_state_estimator:
            self._feed_ball_state_estimator(observation, stamp_sec)

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

        if self.print_every > 0 and self.frame_count % self.print_every == 0:
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

    def _ensure_ball_state_estimator(self) -> Optional[BallStateEstimator]:
        """Lazily build the ball-state estimator from detector outputs.

        Waits until the detector exists and has settled its ``output_frame``
        so the gravity vector and the anisotropic depth-noise rotation match
        the space position/velocity arrive in. Returns the cached estimator,
        or ``None`` if the detector is not ready yet.
        """
        if self._ball_estimator is not None:
            return self._ball_estimator
        if (self.detector is None
                and not self.enable_mocap_input
                and not self.enable_replay_mode):
            return None

        if self.enable_replay_mode:
            output_frame = (
                'sim_frame' if self.replay_data_is_sim_frame else 'base_link')
        else:
            output_frame = (
                'base_link'
                if self.enable_mocap_input
                else getattr(self.detector, 'output_frame', 'camera'))
        if output_frame in ('base_link', 'sim_frame'):
            gravity = estimator_gravity_for_base_link()
            # R_base_camera rotates camera-frame measurement noise into base.
            frame_rotation = (
                getattr(self.detector, 'R_base_camera', None)
                if (self.detector is not None
                    and not self.enable_mocap_input
                    and not self.enable_replay_mode)
                else None)
        else:
            gravity = estimator_gravity_for_camera_frame(self.camera_tilt_deg)
            # Already in camera frame: no rotation needed.
            frame_rotation = None

        table_z = self.ball_estimator_table_z_mm
        table_z = None if (table_z is None or not np.isfinite(table_z)) else float(
            table_z)
        q_depth_pos_mult = self.ball_estimator_q_depth_pos_mult
        q_depth_vel_mult = self.ball_estimator_q_depth_vel_mult
        if output_frame in ('base_link', 'sim_frame') and frame_rotation is None:
            q_depth_pos_mult = 1.0
            q_depth_vel_mult = 1.0

        config = BallStateEstimatorConfig(
            gravity_mm_s2=gravity,
            r_lateral_mm=self.ball_estimator_r_lateral_mm,
            r_depth_mm=self.ball_estimator_r_depth_mm,
            q_depth_pos_mult=q_depth_pos_mult,
            q_depth_vel_mult=q_depth_vel_mult,
            max_extrapolation_s=self.ball_estimator_max_extrapolation_s,
            trust_loss_s=self.ball_estimator_trust_loss_s,
            degrade_loss_s=self.ball_estimator_degrade_loss_s,
            invalid_after_s=self.ball_estimator_invalid_after_s,
            prediction_horizon_s=self.ball_estimator_prediction_horizon_s,
            enable_prediction=self.ball_estimator_enable_prediction,
            enable_aero_drag=self.ball_estimator_enable_aero_drag,
            drag_k_per_mm=self.ball_estimator_drag_k_per_mm,
            enable_magnus=self.ball_estimator_enable_magnus,
            magnus_k=self.ball_estimator_magnus_k,
            table_z_mm=table_z,
        )
        self._ball_estimator = BallStateEstimator(
            config, frame_rotation=frame_rotation)
        self.get_logger().info(
            'Ball-state estimator enabled: '
            f'frame={output_frame}, gravity_mm_s2={gravity}, '
            f'max_extrap={self.ball_estimator_max_extrapolation_s:.3f}s, '
            f'loss(trust/degrade/invalid)='
            f'{self.ball_estimator_trust_loss_s:.2f}/'
            f'{self.ball_estimator_degrade_loss_s:.2f}/'
            f'{self.ball_estimator_invalid_after_s:.2f}s, '
            f'pred_horizon={self.ball_estimator_prediction_horizon_s:.3f}s '
            '(0 => execution latency handled by RL actuator model), '
            f'prediction={"on" if self.ball_estimator_enable_prediction else "off (hold latest fused state @200Hz, no extrapolation)"}, '
            f'aero_drag={"on" if self.ball_estimator_enable_aero_drag else "off"}'
            f'(k={self.ball_estimator_drag_k_per_mm:.3e}/mm), '
            f'magnus={"on" if self.ball_estimator_enable_magnus else "off"}, '
            f'table_z_mm={table_z}, '
            f'q_depth_mult(pos/vel)={q_depth_pos_mult:.2f}/'
            f'{q_depth_vel_mult:.2f}, '
            f'R_base_camera={"set" if frame_rotation is not None else "none"}')
        return self._ball_estimator

    def _feed_ball_state_estimator(self, observation: dict,
                                   stamp_sec: float) -> None:
        """Push one REAL detection into the estimator, stamped at t_img.

        Only genuine, non-rejected detections are fused. The detector tags a
        gated/rejected observation with raw_measurement['filter_rejected'];
        we skip those so the estimator's own innovation gate sees a clean
        stream (it has its own outlier handling, but feeding it the detector's
        own rejects would be redundant noise). A miss simply isn't fed — the
        estimator ages it out via predict_to() on the control side.
        """
        comp = self._ensure_ball_state_estimator()
        if comp is None:
            return
        if not bool(observation.get('detected', False)):
            return
        # Skip detector-flagged outliers (innovation gate rejects); the
        # estimator should fuse only trustworthy measurements.
        raw = observation.get('raw_measurement') or {}
        if raw.get('filter_rejected', False):
            return
        position = observation.get('position')
        # Cache the raw measurement (position + its image time) so the 200 Hz
        # estimator recorder and done warning can put each prediction next to
        # the detection it came from. Done regardless of the fuse below so the
        # freshest raw measurement is available even on the tick it arrives.
        try:
            raw_radius = raw.get('radius_px', raw.get('radius', float('nan')))
            raw_depth = raw.get('depth_mm', float('nan'))
            output_frame = (
                getattr(self.detector, 'output_frame', 'camera')
                if self.detector is not None else 'camera')
            with self._estimator_log_lock:
                self._latest_raw_detection = (
                    float(stamp_sec),
                    [float(position[0]), float(position[1]),
                     float(position[2])],
                    float(raw_radius) if raw_radius is not None
                    else float('nan'),
                    float(raw_depth) if raw_depth is not None
                    else float('nan'),
                    str(output_frame))
        except (TypeError, IndexError, ValueError):
            pass
        try:
            with self._ball_estimator_lock:
                comp.update(position, stamp_sec)
        except Exception as exc:  # noqa: BLE001 - keep callback alive
            self.get_logger().error(
                f'Ball-state estimator update failed: {exc}; '
                'disabling estimator for run')
            self.enable_ball_state_estimator = False
            self._ball_estimator = None

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
            'base_link' if self.enable_mocap_input else
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

    def _stdin_enter_thread(self):
        """Daemon thread: wait for stdin Enter and set the event."""
        if not self._stdin_is_tty:
            if self.auto_continue_on_stdin_eof:
                self.get_logger().warn(
                    'stdin is not an interactive TTY; auto-continuing because '
                    'auto_continue_on_stdin_eof=true')
                self._confirm_start('stdin EOF auto-continue')
            else:
                self.get_logger().warn(
                    'stdin is not an interactive TTY; waiting for ROS topic '
                    f'confirmation on "{self.start_confirm_topic}"')
            return
        try:
            input()  # blocks until Enter
            self._confirm_start('stdin Enter')
        except (EOFError, OSError) as exc:
            if self.auto_continue_on_stdin_eof:
                self.get_logger().warn(
                    f'stdin unavailable ({exc}); auto-continuing because '
                    'auto_continue_on_stdin_eof=true')
                self._confirm_start('stdin EOF auto-continue')
            else:
                self.get_logger().warn(
                    f'stdin unavailable ({exc}); waiting for ROS topic '
                    f'confirmation on "{self.start_confirm_topic}"')

    def _confirm_start(self, source: str):
        """Mark the operator start confirmation as received."""
        if self._enter_event.is_set():
            return
        self._enter_event_source = source
        self._enter_event.set()

    def _start_confirm_callback(self, _msg: EmptyMsg):
        """Accept a ROS topic start signal once the init pose has completed."""
        if not self._await_enter:
            return
        now_wall = time.time()
        if self._init_pose_active:
            if now_wall - self._start_confirm_ignored_last_wall >= 1.0:
                self.get_logger().warn(
                    'Start confirmation ignored because init-pose phase is '
                    'still active; send it after the init pose is reached')
                self._start_confirm_ignored_last_wall = now_wall
            return
        self.get_logger().info(
            f'Start confirmation received on "{self.start_confirm_topic}"')
        self._confirm_start(f'ROS topic {self.start_confirm_topic}')

    def _maybe_log_enter_prompt(self, now_wall: float):
        """Periodically remind the operator how to release the start gate."""
        if (self._enter_prompted and
                now_wall - self._enter_prompt_last_wall <
                self.enter_prompt_log_period_s):
            return
        topic_cmd = (
            f'ros2 topic pub --once {self.start_confirm_topic} '
            'std_msgs/msg/Empty "{}"')
        self.get_logger().warn(
            'WAITING FOR START: press Enter in this terminal or run: '
            f'{topic_cmd}')
        if not self._enter_prompted and self._stdin_is_tty:
            print("\n=== Press Enter to start RL control ===", flush=True)
        self._enter_prompted = True
        self._enter_prompt_last_wall = now_wall

    def _control_timer_callback(self):
        """Fixed-rate (control_rate_hz) control loop.

        Phase 1 (init pose, if enabled): smoothly ramp the right arm from the
        measured joint state to TARGET_DEGREES, then publish it as a
        single-element MechUnitCmdArray on command_topic. This keeps startup
        gentle even when the runtime safety limiter is disabled for RL tests.

        Phase 2 (RL control): every tick we call compute_action() regardless
        of detection validity so the RL policy's internal hold-last / dropout
        logic matches training. Raw RL output goes through the safety limiter
        before being published.
        """
        now_wall = time.time()

        # Phase 0: init-pose ramp (if enabled). Move smoothly to TARGET_DEGREES
        # until the time/tolerance condition is met. Do NOT start the replay
        # clock yet.
        if self._init_pose_active:
            if self._init_pose_start_wall is None:
                self._init_pose_start_wall = now_wall
            elapsed = now_wall - self._init_pose_start_wall
            self._maybe_sync_safety_limiter_to_joint_state()
            # Right-arm holds the RL initial posture
            raw_cmd = (
                self.rl_controller.current_arm_cmd_q().astype(np.float64)
                if self.rl_controller is not None
                else np.asarray(target_right_arm_q_rad(), dtype=np.float64)
            )
            init_cmd = self._compute_init_pose_ramp_command(raw_cmd, elapsed)
            self._last_raw_joint_positions = init_cmd.copy()
            self._publish_right_arm_init_pose(
                init_cmd, self.get_clock().now().to_msg())
            time_up = elapsed >= self.init_pose_duration_s
            pose_reached = self._init_pose_reached()
            if time_up or pose_reached:
                self._init_pose_active = False
                reason = 'pose_reached' if pose_reached else 'duration_elapsed'
                self.get_logger().info(
                    f'Init-pose phase done after {elapsed:.2f}s ({reason})')
                # Transition to Enter-wait if configured
                if self._await_enter:
                    self.get_logger().info(
                        'Waiting for operator confirmation before starting RL...')
            return

        # Phase 0.5: Enter-wait (if require_enter_confirm is on). Keep holding
        # the init pose. Do NOT start the replay clock yet. A daemon thread
        # listens on stdin and sets _enter_event when Enter is pressed.
        if self._await_enter:
            if not self._enter_event.is_set():
                # Start the input thread once
                if not self._enter_thread_started:
                    threading.Thread(target=self._stdin_enter_thread, daemon=True).start()
                    self._enter_thread_started = True
                self._maybe_log_enter_prompt(now_wall)
                # Keep holding the init pose
                raw_cmd = (
                    self.rl_controller.current_arm_cmd_q().astype(np.float64)
                    if self.rl_controller is not None
                    else np.asarray(target_right_arm_q_rad(), dtype=np.float64)
                )
                self._last_raw_joint_positions = raw_cmd.copy()
                self._publish_right_arm_init_pose(raw_cmd, self.get_clock().now().to_msg())
                return
            else:
                # Enter was pressed; transition to RL
                source = self._enter_event_source or 'operator confirmation'
                self.get_logger().info(f'Start confirmed by {source}')
                self._await_enter = False  # only log once

                if self.post_enter_start_delay_s > 0.0:
                    self._post_enter_delay_until_wall = (
                        now_wall + self.post_enter_start_delay_s)
                    self.get_logger().info(
                        f'Holding init pose for '
                        f'{self.post_enter_start_delay_s:.2f}s before formal '
                        'RL start')
                    raw_cmd = (
                        self.rl_controller.current_arm_cmd_q().astype(np.float64)
                        if self.rl_controller is not None
                        else np.asarray(target_right_arm_q_rad(), dtype=np.float64)
                    )
                    self._last_raw_joint_positions = raw_cmd.copy()
                    self._publish_right_arm_init_pose(
                        raw_cmd, self.get_clock().now().to_msg())
                    return
                self._reset_compensator_to_current_arm()

        # Phase 0.75: post-Enter countdown. Keep holding the init pose until the
        # configured delay elapses, then reset the compensator right before the
        # formal run starts so it is anchored to the current hardware feedback.
        if self._post_enter_delay_until_wall is not None:
            remaining = self._post_enter_delay_until_wall - now_wall
            if remaining > 0.0:
                raw_cmd = (
                    self.rl_controller.current_arm_cmd_q().astype(np.float64)
                    if self.rl_controller is not None
                    else np.asarray(target_right_arm_q_rad(), dtype=np.float64)
                )
                self._last_raw_joint_positions = raw_cmd.copy()
                self._publish_right_arm_init_pose(
                    raw_cmd, self.get_clock().now().to_msg())
                return
            self._post_enter_delay_until_wall = None
            self.get_logger().info('Post-Enter delay complete; starting formal RL control')
            self._reset_compensator_to_current_arm()

        # NOW start fetching the ball state (so the replay clock only advances
        # once we're past init + Enter). Formal run timing starts later, after
        # capture/release gates have actually handed control to the policy.
        # Get ball state from different sources based on mode
        if self.enable_replay_mode:
            # Replay mode: get ball state from CSV
            ball_state = self._get_replay_ball_state()
            if ball_state is None:
                # Replay finished
                self.get_logger().info('Replay finished, stopping control timer')
                self.control_timer.cancel()
                if self.replay_auto_exit:
                    self.get_logger().info('Auto-exit enabled, shutting down node')
                    threading.Thread(target=self._shutdown_after_replay, daemon=True).start()
                return
        elif self.enable_sim_ball_input:
            # Simulation ball input mode: get ball state from simulation topic
            with self._sim_ball_lock:
                ball_state = self._sim_ball_state
        elif self.enable_mocap_input:
            # Motion capture input mode: get ball state from mocap detector
            ball_state = self._get_mocap_ball_state()
        else:
            # Normal mode: get from vision
            with self._latest_ball_lock:
                ball_state = self._latest_ball_state

        if ball_state is None:
            ball_state = BallState(
                position=np.zeros(3), velocity=np.zeros(3),
                stamp=self.get_clock().now().to_msg(), valid=False)

        # Phase 2: always run RL, even on invalid detections. The RL
        # policy's hold-last/dropout logic (see RLPolicyController.predict)
        # reproduces the training-time short-dropout behavior.
        # If init pose was disabled entirely, still try a one-shot sync of
        # the safety limiter to the first real /joint_states so the first
        # RL tick does not jump.
        self._maybe_sync_safety_limiter_to_joint_state()

        # When the ball-state estimator is on, replace the (possibly stale,
        # 60 Hz) cached detection with a fresh state extrapolated to *now*.
        # This bridges 60->200 Hz and de-ages the measurement past the
        # perception latency in one step. The graded age it reports overrides
        # the wall-clock age compute_action would otherwise derive. When off,
        # ball_state passes through unchanged and age_override stays None.
        ball_state, ball_age_override = self._resolve_control_ball_state(
            ball_state)

        if (self.enable_replay_mode
                or self.enable_sim_ball_input
                or self.enable_mocap_input):
            self._publish_ball_state_debug(ball_state)

        # Initial ball-capture gate: before the first trustworthy ball state,
        # do not feed the policy a synthetic zero/held ball. Stay parked at the
        # safe init posture until a real or estimator-valid ball is available.
        if self._should_hold_for_initial_ball_capture(ball_state):
            self._run_ball_capture_gate(ball_state)
            return

        # Release gate: the first valid ball may still be in the operator's
        # hand. Keep parking until the ball is freshly observed moving
        # downward over a short window, then start the formal policy run.
        if self._should_hold_for_ball_release(ball_state, ball_age_override):
            self._run_ball_release_gate(ball_state, ball_age_override)
            return

        # Ball-lost failsafe: if the ball has been undetected for longer than
        # the timeout, stop running the RL policy and command the fixed safe
        # init posture instead. The safety limiter ramps the right arm toward
        # it under the normal vel/acc limits. RL resumes on reacquisition.
        if self._should_engage_ball_lost_failsafe(ball_state, now_wall):
            self._run_ball_lost_failsafe(ball_state.stamp)
            return

        # Formal run timing starts only once all operator and ball gates have
        # cleared and the policy is about to be allowed to produce commands.
        if self._formal_run_start_wall is None:
            self._formal_run_start_wall = now_wall
            if self.run_duration_s > 0.0:
                self.get_logger().info(
                    f'Formal run started; duration limit '
                    f'{self.run_duration_s:.2f}s')
        if self.run_duration_s > 0.0:
            run_elapsed = now_wall - self._formal_run_start_wall
            if run_elapsed >= self.run_duration_s:
                self._run_duration_failsafe(self.get_clock().now().to_msg())
                return

        raw_cmd = self.compute_action(ball_state, age_override=ball_age_override)
        self._last_raw_joint_positions = np.asarray(
            raw_cmd, dtype=np.float64).reshape(-1).copy()

        # Done-condition failsafe: check if RL is operating outside trained
        # boundaries (ball/racket out of bounds, etc.). If any condition fires,
        # stop RL and ramp to safe pose, just like ball-lost failsafe.
        is_done, done_reason, done_conditions = self._should_engage_done_failsafe(
            ball_state, self._last_sim_ball_pos_m, self._last_sim_ball_vel_m_s)
        if is_done:
            self._run_done_failsafe(ball_state.stamp, done_reason)
            # Log diagnostics periodically
            self._maybe_log_done_status(done_conditions)
            return
        else:
            # Clear done failsafe if conditions are OK
            if self._done_failsafe_active:
                self.get_logger().info(
                    'Done conditions cleared; resuming RL control')
                self._done_failsafe_active = False
            # Log diagnostics periodically
            self._maybe_log_done_status(done_conditions)

        # Publish RL joint command debug (reflects 200 Hz control cadence).
        self._publish_rl_joint_cmd_debug()
        self._publish_right_arm_command(
            self._last_raw_joint_positions, ball_state.stamp)

    def _resolve_control_ball_state(
            self, ball_state: BallState) -> 'tuple[BallState, Optional[float]]':
        """Return the ball state the control loop should act on this tick.

        With the estimator off (default), this is a no-op: the cached detector
        BallState passes through and the age override is None, so behavior is
        byte-for-byte the prior path. With the estimator on, we extrapolate the
        committed filter state to *now* (same clock domain the image stamps
        live in, so the elapsed time absorbs the perception latency) and return
        a synthesized BallState plus the estimator's graded age for the policy.
        """
        if not self.enable_ball_state_estimator:
            return ball_state, None
        est = self._ensure_ball_state_estimator()
        if est is None:
            return ball_state, None

        now_time = self.get_clock().now()
        t_now_sec = now_time.nanoseconds * 1e-9
        try:
            with self._ball_estimator_lock:
                out = est.predict_to(t_now_sec)
        except Exception as exc:  # noqa: BLE001 - keep control loop alive
            self.get_logger().error(
                f'Ball-state estimator predict failed: {exc}; '
                'disabling estimator for run')
            self.enable_ball_state_estimator = False
            self._ball_estimator = None
            return ball_state, None

        self._maybe_log_ball_estimator_status(est, out)
        if self.save_estimator_log:
            self._record_estimator_log(t_now_sec, out)

        if not bool(out.get('valid', False)):
            # Estimator gave up (loss past invalid_after_s) or never tracked:
            # report an invalid BallState so the failsafe path takes over.
            invalid = BallState(
                position=np.zeros(3), velocity=np.zeros(3),
                stamp=now_time.to_msg(), valid=False,
                raw_measurement={'ball_estimator': out})
            return invalid, None

        est_state = BallState(
            position=np.array(out['position'], dtype=float),
            velocity=np.array(out['velocity'], dtype=float),
            stamp=now_time.to_msg(), valid=True,
            raw_measurement={'ball_estimator': out})
        return est_state, float(out.get('age_s', 0.0))

    def _maybe_log_ball_estimator_status(self, est: BallStateEstimator,
                                         out: dict) -> None:
        """Throttled visibility into the estimator state and counters."""
        now_wall = time.time()
        if (now_wall - self._ball_estimator_log_last_wall
                < max(self.ball_estimator_log_period_s, 0.1)):
            return
        self._ball_estimator_log_last_wall = now_wall
        diag = est.diagnostics()
        self.get_logger().info(
            f'BallEstimator: state={out.get("state")} '
            f'valid={out.get("valid")} loss={out.get("loss_s", 0.0) * 1e3:.0f}ms '
            f'extrap={out.get("extrapolation_s", 0.0) * 1e3:.0f}ms '
            f'age={out.get("age_s", 0.0) * 1e3:.0f}ms '
            f'acc={diag["updates_accepted"]} rej={diag["updates_rejected"]} '
            f'reinit={diag["reinits"]} bounce={diag["bounces_detected"]}')

    def _record_estimator_log(self, t_now_sec: float, out: dict) -> None:
        """Append one 200 Hz estimator row for offline debug (in-memory).

        Each row pairs the estimator's extrapolated/de-aged state at this
        control tick with the most recent raw 60 Hz detection, so the offline
        comparison can show how far the bridge moved the state and whether the
        prediction lined up with the next measurement. Rows are flushed to CSV
        on shutdown by ``save_estimator_log_file``.
        """
        with self._estimator_log_lock:
            raw = self._latest_raw_detection
        raw_t = raw[0] if raw is not None else float('nan')
        raw_p = raw[1] if raw is not None else [float('nan')] * 3
        # radius_px / depth_mm were appended to the cached tuple so the offline
        # analysis can separate monocular-depth error from pixel error and
        # compare against the D455 hardware depth. Tolerate the old 2-field
        # tuple shape (no radius/depth) so stale caches don't crash.
        raw_radius = raw[2] if (raw is not None and len(raw) > 2) else float('nan')
        raw_depth = raw[3] if (raw is not None and len(raw) > 3) else float('nan')
        pos = out.get('position', [float('nan')] * 3)
        vel = out.get('velocity', [float('nan')] * 3)
        row = {
            't_now': float(t_now_sec),
            'valid': int(bool(out.get('valid', False))),
            'predicted': int(bool(out.get('predicted', False))),
            'state': str(out.get('state', '')),
            'x_mm': float(pos[0]), 'y_mm': float(pos[1]), 'z_mm': float(pos[2]),
            'vx_mm_s': float(vel[0]), 'vy_mm_s': float(vel[1]),
            'vz_mm_s': float(vel[2]),
            'age_s': float(out.get('age_s', 0.0)),
            'loss_s': float(out.get('loss_s', 0.0)),
            'extrapolation_s': float(out.get('extrapolation_s', 0.0)),
            'clamped': int(bool(out.get('clamped', False))),
            # Latest raw 60 Hz detection (image-stamped) for side-by-side cmp.
            'raw_t_img': float(raw_t),
            'raw_x_mm': float(raw_p[0]), 'raw_y_mm': float(raw_p[1]),
            'raw_z_mm': float(raw_p[2]),
            # Monocular depth cue from the detection: circle radius -> depth.
            'raw_radius_px': float(raw_radius),
            'raw_depth_mm': float(raw_depth),
        }
        with self._estimator_log_lock:
            self._estimator_log_rows.append(row)

    def save_estimator_log_file(self) -> None:
        """Write the buffered 200 Hz estimator rows to CSV (called on shutdown)."""
        if not self.save_estimator_log:
            return
        with self._estimator_log_lock:
            rows = list(self._estimator_log_rows)
        if not rows:
            self.get_logger().info(
                'Estimator log empty; skipping save (was the estimator '
                'enabled and did the ball get detected?)')
            return
        path = self.estimator_log_path
        if not path:
            ts = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(
                self.source_module_dir, 'outputs', 'rl_sim',
                f'estimator_log_{ts}.csv')
        path = os.path.expandvars(os.path.expanduser(path))
        try:
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            self.get_logger().info(
                f'Saved {len(rows)} estimator rows (200 Hz) to: {path}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to save estimator log: {exc}')

    def _record_policy_log(
            self, ball_pos_m: np.ndarray, ball_vel_m_s: np.ndarray,
            ball_valid: bool, age_s: float,
            arm_q: Optional[np.ndarray], arm_dq: Optional[np.ndarray],
            control_mode: str = 'policy') -> None:
        """Append one 200 Hz policy step (obs + action) for offline analysis.

        Records the policy input (observation vector) and output (action vector)
        at each control tick for comparison against reference traces from
        simulation validation runs.
        """
        if self.rl_controller is None:
            return

        # Get the latest obs and action from the RL controller
        try:
            obs_action = self.rl_controller.latest_obs_and_action()
            obs = obs_action['obs']
            action = obs_action['action']
        except Exception:  # noqa: BLE001
            return

        # Build the row with metadata + obs + action
        row = {
            'step': int(self._policy_step_counter),
            'time_sec': float(self._policy_step_counter * self.control_dt),
            'control_mode': str(control_mode),
            'ball_valid': int(ball_valid),
            'ball_age_s': float(age_s),
            'ball_pos_x_m': float(ball_pos_m[0]),
            'ball_pos_y_m': float(ball_pos_m[1]),
            'ball_pos_z_m': float(ball_pos_m[2]),
            'ball_vel_x_m_s': float(ball_vel_m_s[0]),
            'ball_vel_y_m_s': float(ball_vel_m_s[1]),
            'ball_vel_z_m_s': float(ball_vel_m_s[2]),
        }

        # Add arm feedback if available
        if arm_q is not None:
            for i, joint_name in enumerate(
                    ['RightArm-0', 'RightArm-1', 'RightArm-2',
                     'RightArm-3', 'RightArm-4', 'RightArm-5', 'RightArm-6']):
                row[f'arm_q/{joint_name}'] = float(arm_q[i])
        else:
            for joint_name in ['RightArm-0', 'RightArm-1', 'RightArm-2',
                               'RightArm-3', 'RightArm-4', 'RightArm-5', 'RightArm-6']:
                row[f'arm_q/{joint_name}'] = float('nan')

        if arm_dq is not None:
            for i, joint_name in enumerate(
                    ['RightArm-0', 'RightArm-1', 'RightArm-2',
                     'RightArm-3', 'RightArm-4', 'RightArm-5', 'RightArm-6']):
                row[f'arm_dq/{joint_name}'] = float(arm_dq[i])
        else:
            for joint_name in ['RightArm-0', 'RightArm-1', 'RightArm-2',
                               'RightArm-3', 'RightArm-4', 'RightArm-5', 'RightArm-6']:
                row[f'arm_dq/{joint_name}'] = float('nan')

        # Add observation vector (obs/000, obs/001, ...)
        for i in range(len(obs)):
            row[f'obs/{i:03d}'] = float(obs[i])

        # Add action vector
        for i, joint_name in enumerate(
                ['RightArm-0', 'RightArm-1', 'RightArm-2',
                 'RightArm-3', 'RightArm-4', 'RightArm-5', 'RightArm-6']):
            row[f'action/{joint_name}'] = float(action[i])

        with self._policy_log_lock:
            self._policy_log_rows.append(row)
        self._policy_step_counter += 1

    def save_policy_log_file(self) -> None:
        """Write the buffered 200 Hz policy rows to CSV (called on shutdown)."""
        if not self.save_policy_log:
            return
        with self._policy_log_lock:
            rows = list(self._policy_log_rows)
        if not rows:
            self.get_logger().info(
                'Policy log empty; skipping save (was the RL policy enabled?)')
            return
        path = self.policy_log_path
        if not path:
            ts = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(
                self.source_module_dir, 'outputs', 'rl_sim',
                f'policy_obs_trace_{ts}.csv')
        path = os.path.expandvars(os.path.expanduser(path))
        try:
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            self.get_logger().info(
                f'Saved {len(rows)} policy rows (200 Hz) to: {path}')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Failed to save policy log: {exc}')

    def _shutdown_after_replay(self) -> None:
        """Shutdown node gracefully after replay completes (called in a thread)."""
        import time
        # Give the control loop a moment to finish cleanly
        time.sleep(0.5)
        # Save logs before shutting down
        self.save_estimator_log_file()
        self.save_policy_log_file()
        # Request shutdown
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    def _shutdown_after_run_duration(self) -> None:
        """Shutdown after a timed real run has parked at the safe pose."""
        import time
        time.sleep(self.shutdown_after_run_duration_delay_s)
        self.save_estimator_log_file()
        self.save_policy_log_file()
        try:
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass


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

    def _should_hold_for_initial_ball_capture(
            self, ball_state: BallState) -> bool:
        """True while no valid ball has ever reached the control loop."""
        if not self.enable_ball_capture_gate:
            return False
        if ball_state.valid:
            if not self._has_captured_valid_ball:
                self.get_logger().info(
                    'First valid ball state captured; releasing RL control')
            if self._ball_capture_gate_active:
                self.get_logger().info(
                    'Ball-capture gate cleared; RL resumes from safe pose')
            self._has_captured_valid_ball = True
            self._ball_capture_gate_active = False
            return False
        return not self._has_captured_valid_ball

    def _release_gate_age_s(
            self, ball_state: BallState,
            age_override: Optional[float]) -> float:
        if age_override is not None and np.isfinite(age_override):
            return max(0.0, float(age_override))
        if ball_state.valid:
            return 0.0
        return self.rl_ball_obs_age_clip

    def _should_hold_for_ball_release(
            self, ball_state: BallState,
            age_override: Optional[float]) -> bool:
        """True until the initially captured ball is actually released."""
        if not self.enable_ball_release_gate:
            return False
        if self._has_released_ball:
            return False

        now_wall = time.time()
        age_s = self._release_gate_age_s(ball_state, age_override)
        debug = {
            'valid': bool(ball_state.valid),
            'age_s': float(age_s),
            'fresh': False,
            'vz_m_s': float('nan'),
            'z_drop_m': 0.0,
            'candidate': False,
            'candidate_ticks': int(self._ball_release_candidate_ticks),
        }

        if not ball_state.valid:
            self._ball_release_samples.clear()
            self._ball_release_candidate_ticks = 0
            debug['candidate_ticks'] = 0
            self._ball_release_last_debug = debug
            return True

        pos = np.asarray(ball_state.position, dtype=np.float64).reshape(-1)
        vel = np.asarray(ball_state.velocity, dtype=np.float64).reshape(-1)
        if (pos.shape[0] != 3 or vel.shape[0] != 3
                or not np.all(np.isfinite(pos))
                or not np.all(np.isfinite(vel))):
            self._ball_release_samples.clear()
            self._ball_release_candidate_ticks = 0
            debug['candidate_ticks'] = 0
            self._ball_release_last_debug = debug
            return True

        fresh = age_s <= self.ball_release_max_obs_age_s
        vz_m_s = float(vel[2] * 1e-3)
        z_mm = float(pos[2])
        if fresh:
            self._ball_release_samples.append((now_wall, z_mm, vz_m_s))
            while (self._ball_release_samples
                   and now_wall - self._ball_release_samples[0][0]
                   > self.ball_release_window_s):
                self._ball_release_samples.popleft()
        else:
            self._ball_release_samples.clear()

        z_drop_m = 0.0
        if self._ball_release_samples:
            max_z_mm = max(sample[1] for sample in self._ball_release_samples)
            z_drop_m = max(0.0, (max_z_mm - z_mm) * 1e-3)

        velocity_ok = (
            vz_m_s <= -abs(self.ball_release_min_downward_velocity_m_s))
        drop_ok = z_drop_m >= self.ball_release_min_z_drop_m
        candidate = bool(fresh and velocity_ok and drop_ok)
        if candidate:
            self._ball_release_candidate_ticks += 1
        else:
            self._ball_release_candidate_ticks = 0

        debug.update({
            'fresh': bool(fresh),
            'vz_m_s': float(vz_m_s),
            'z_drop_m': float(z_drop_m),
            'candidate': bool(candidate),
            'candidate_ticks': int(self._ball_release_candidate_ticks),
        })
        self._ball_release_last_debug = debug

        if (self._ball_release_candidate_ticks
                >= self.ball_release_required_consecutive_ticks):
            self._has_released_ball = True
            self._ball_release_gate_active = False
            self._last_valid_ball_wall_time = now_wall
            self._ball_release_samples.clear()
            self._ball_release_candidate_ticks = 0
            self.get_logger().info(
                'Ball-release gate cleared; starting RL from released ball '
                f'(vz={vz_m_s:.3f}m/s, z_drop={z_drop_m:.3f}m, '
                f'age={age_s * 1e3:.0f}ms)')
            self._sync_rl_integrator_to_last_safe_command()
            self._reset_compensator_to_current_arm()
            return False

        return True

    def _sync_rl_integrator_to_last_safe_command(self) -> None:
        """Pin the RL command integrator to the last published safe command."""
        if self.rl_controller is None:
            return
        self.rl_controller.arm_cmd_q = \
            self._last_safe_joint_positions.astype(np.float32).copy()
        self.rl_controller.arm_cmd_qvel = np.zeros(
            self._rl_expected_joint_count, dtype=np.float32)

    def _reset_compensator_to_current_arm(self) -> None:
        """Reset actuator compensation from the latest right-arm feedback."""
        if self.compensator is None:
            return
        try:
            current_q, _ = self._get_right_arm_feedback()
            if current_q is None and self.rl_controller is not None:
                current_q = self.rl_controller.current_arm_cmd_q()
            if current_q is not None:
                self.compensator.reset(current_q.astype(np.float32))
                self.get_logger().info(
                    'Compensator reset with current arm position')
        except Exception as e:
            self.get_logger().warn(f'Failed to reset compensator: {e}')

    def _record_policy_hold_log(
            self, ball_state: BallState,
            age_s: float,
            arm_q: Optional[np.ndarray],
            arm_dq: Optional[np.ndarray],
            control_mode: str = 'hold') -> None:
        """Record a 200 Hz row while RL inference is deliberately parked."""
        if not self.save_policy_log or self.rl_controller is None:
            return
        ball_pos_m, ball_vel_m_s = self._ball_state_to_policy_sim_frame(
            ball_state)
        try:
            if hasattr(self.rl_controller, '_build_obs'):
                obs = self.rl_controller._build_obs(
                    self.rl_controller._last_valid_ball_pos_m,
                    self.rl_controller._last_valid_ball_vel_m_s,
                    float(age_s), self.control_dt,
                    arm_q=arm_q, arm_dq=arm_dq)
                self.rl_controller.last_obs = obs.astype(np.float32).copy()
                self.rl_controller.last_action = np.zeros(
                    self._rl_expected_joint_count, dtype=np.float32)
            self._record_policy_log(
                ball_pos_m, ball_vel_m_s, bool(ball_state.valid),
                float(age_s), arm_q, arm_dq, control_mode=control_mode)
        except Exception:  # noqa: BLE001 - logging must not affect safety path
            pass

    def _run_ball_capture_gate(self, ball_state: BallState) -> None:
        """Hold safe init pose until the first valid ball state is captured."""
        if not self._ball_capture_gate_active:
            self._ball_capture_gate_active = True
            self.get_logger().warn(
                'No valid ball captured yet; holding safe init pose and '
                'skipping RL inference')
        target = self.safe_init_pose_q
        self._last_raw_joint_positions = target.copy()
        self._publish_right_arm_command(target, ball_state.stamp)
        self._sync_rl_integrator_to_last_safe_command()
        right_arm_q, right_arm_dq = self._get_right_arm_feedback()
        self._record_policy_hold_log(
            ball_state, self.rl_ball_obs_age_clip, right_arm_q, right_arm_dq,
            control_mode='capture_gate')

    def _run_ball_release_gate(
            self, ball_state: BallState,
            age_override: Optional[float]) -> None:
        """Hold safe init pose until the operator-released ball is moving."""
        now_wall = time.time()
        if not self._ball_release_gate_active:
            self._ball_release_gate_active = True
            self._ball_release_gate_log_last_wall = now_wall
            self.get_logger().warn(
                'Waiting for ball release; holding safe init pose and '
                'skipping RL inference')
        elif now_wall - self._ball_release_gate_log_last_wall >= 1.0:
            dbg = getattr(self, '_ball_release_last_debug', {}) or {}
            self.get_logger().info(
                'Ball-release gate waiting: '
                f"valid={dbg.get('valid')} "
                f"fresh={dbg.get('fresh')} "
                f"age={float(dbg.get('age_s', 0.0)) * 1e3:.0f}ms "
                f"vz={float(dbg.get('vz_m_s', float('nan'))):.3f}m/s "
                f"z_drop={float(dbg.get('z_drop_m', 0.0)):.3f}m "
                f"ticks={int(dbg.get('candidate_ticks', 0))}/"
                f'{self.ball_release_required_consecutive_ticks}')
            self._ball_release_gate_log_last_wall = now_wall

        target = self.safe_init_pose_q
        self._last_raw_joint_positions = target.copy()
        self._publish_right_arm_command(target, ball_state.stamp)
        self._sync_rl_integrator_to_last_safe_command()
        right_arm_q, right_arm_dq = self._get_right_arm_feedback()
        self._record_policy_hold_log(
            ball_state,
            self._release_gate_age_s(ball_state, age_override),
            right_arm_q, right_arm_dq,
            control_mode='release_gate')

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
        # Sync the policy integrator to what was actually commanded this tick
        # (the ramp position) with zero velocity, so a resume does not snap.
        self._sync_rl_integrator_to_last_safe_command()

    def _should_engage_done_failsafe(
            self, ball_state: BallState, ball_pos_m: np.ndarray,
            ball_vel_m_s: np.ndarray) -> tuple[bool, str, dict]:
        """Check done conditions and return (should_engage, reason, conditions).

        Returns (False, "", {}) if done monitoring is disabled or if the policy
        FK is unavailable. Otherwise evaluates all conditions and returns True
        if ANY fires.
        """
        if not self.enable_done_monitoring or self.done_monitor is None:
            return False, "", {}
        if not ball_state.valid:
            # No ball => cannot evaluate ball-state conditions, skip check.
            return False, "", {}
        if self.rl_controller is None:
            return False, "", {}

        # Get racket position from real feedback when available; replay/headless
        # paths fall back to the policy's command trajectory.
        right_arm_q, _ = self._get_right_arm_feedback()
        racket_q = (
            right_arm_q if right_arm_q is not None
            else self.rl_controller.arm_cmd_q)
        racket_pos_m, fk_ok = self.rl_controller._compute_racket_pose_base(
            racket_q)
        if not fk_ok:
            # FK unavailable => cannot check racket conditions
            racket_pos_m = None
        racket_anchor_m = getattr(
            self.rl_controller, '_racket_anchor_pos_base', None)
        if racket_anchor_m is None:
            racket_anchor_m = self._viz_racket_anchor_sim
        if racket_anchor_m is None:
            return False, "", {}

        # RLPolicyController FK already uses the MJX base-yaw convention:
        # x/y are base-relative and z is world/sim height. Do not apply
        # _transform_ball_pos_to_sim_frame(), which is only for real ball
        # measurements whose z is base_link-relative.
        if racket_pos_m is not None:
            racket_pos_sim = np.asarray(racket_pos_m, dtype=np.float64).reshape(3)
        else:
            racket_pos_sim = None
        racket_anchor_sim = np.asarray(racket_anchor_m, dtype=np.float64).reshape(3)

        # Check done conditions
        is_done, reason, conditions = self.done_monitor.check_done(
            ball_pos_m=ball_pos_m,
            ball_vel_m_s=ball_vel_m_s,
            racket_pos_m=racket_pos_sim,
            racket_anchor_m=racket_anchor_sim,
            base_pos_m=None,  # Stationary base => [0, 0, z]
        )
        if is_done:
            raw_sim_pos = None
            raw_age_s = None
            raw_frame = None
            try:
                with self._estimator_log_lock:
                    raw = self._latest_raw_detection
                if raw is not None and len(raw) >= 2:
                    raw_t = float(raw[0])
                    raw_pos_m = (
                        np.asarray(raw[1], dtype=np.float64).reshape(3) * 1e-3)
                    raw_frame = str(raw[4]) if len(raw) > 4 else 'base_link'
                    if raw_frame == 'sim_frame':
                        raw_sim_pos = raw_pos_m
                    elif raw_frame == 'base_link':
                        raw_sim_pos = self._transform_ball_pos_to_sim_frame(raw_pos_m)
                    raw_age_s = max(
                        0.0,
                        self.get_clock().now().nanoseconds * 1e-9 - raw_t)
            except Exception:  # noqa: BLE001 - diagnostics must not affect safety
                raw_sim_pos = None
                raw_age_s = None
            self._last_done_debug = {
                'ball_pos_m': np.asarray(ball_pos_m, dtype=np.float64).reshape(3),
                'ball_vel_m_s': np.asarray(ball_vel_m_s, dtype=np.float64).reshape(3),
                'raw_ball_pos_m': raw_sim_pos,
                'raw_ball_age_s': raw_age_s,
                'raw_ball_frame': raw_frame,
                'racket_pos_m': (
                    None if racket_pos_sim is None
                    else np.asarray(racket_pos_sim, dtype=np.float64).reshape(3)),
                'racket_anchor_m': racket_anchor_sim,
                'conditions': dict(conditions),
            }

        return is_done, reason, conditions

    def _run_done_failsafe(self, stamp: Time, reason: str) -> None:
        """Pause RL and drive to safe pose when done condition fires.

        Same behavior as ball-lost failsafe: smooth ramp to safe_init_pose_q
        via the safety limiter, pinning the RL integrator so it can resume
        cleanly if the violation clears.
        """
        if not self._done_failsafe_active:
            self._done_failsafe_active = True
            debug = getattr(self, '_last_done_debug', {}) or {}
            ball = debug.get('ball_pos_m')
            raw_ball = debug.get('raw_ball_pos_m')
            raw_age_s = debug.get('raw_ball_age_s')
            racket = debug.get('racket_pos_m')
            anchor = debug.get('racket_anchor_m')
            detail = ''
            if ball is not None:
                detail += f' control_ball={np.round(ball, 3).tolist()}m'
            if raw_ball is not None:
                detail += f' raw_ball={np.round(raw_ball, 3).tolist()}m'
                if ball is not None:
                    dz = float(np.asarray(ball)[2] - np.asarray(raw_ball)[2])
                    detail += f' dz_control_minus_raw={dz:+.3f}m'
                if raw_age_s is not None:
                    detail += f' raw_age={raw_age_s * 1e3:.0f}ms'
            if racket is not None:
                detail += f' racket={np.round(racket, 3).tolist()}m'
            if anchor is not None:
                detail += f' anchor={np.round(anchor, 3).tolist()}m'
            self.get_logger().warn(
                f'Done condition triggered: {reason} - '
                f'Pausing RL and moving to safe pose.{detail}')
        target = self.safe_init_pose_q
        self._last_raw_joint_positions = target.copy()
        self._publish_right_arm_command(target, stamp)
        self._sync_rl_integrator_to_last_safe_command()

    def _run_duration_failsafe(self, stamp: Time) -> None:
        """Park at safe pose after the configured formal run duration expires."""
        if not self._run_duration_elapsed:
            self._run_duration_elapsed = True
            elapsed = 0.0
            if self._formal_run_start_wall is not None:
                elapsed = max(0.0, time.time() - self._formal_run_start_wall)
            self.get_logger().warn(
                f'Run duration reached ({elapsed:.2f}s >= '
                f'{self.run_duration_s:.2f}s); parking at safe init pose')
            if (self.shutdown_after_run_duration
                    and not self._run_duration_shutdown_started):
                self._run_duration_shutdown_started = True
                self.get_logger().info(
                    'shutdown_after_run_duration=true; shutting down after '
                    f'{self.shutdown_after_run_duration_delay_s:.2f}s hold')
                threading.Thread(
                    target=self._shutdown_after_run_duration,
                    daemon=True).start()
        target = self.safe_init_pose_q
        self._last_raw_joint_positions = target.copy()
        self._publish_right_arm_command(target, stamp)
        self._sync_rl_integrator_to_last_safe_command()

    def _maybe_log_done_status(self, conditions: dict) -> None:
        """Throttled log of done-condition diagnostics."""
        now_wall = time.time()
        if (now_wall - self._done_log_last_wall) < self.done_log_period_s:
            return
        self._done_log_last_wall = now_wall
        if self.done_monitor is None:
            return
        diag = self.done_monitor.diagnostics()
        active = [k for k, v in conditions.items() if v]
        self.get_logger().info(
            f'DoneMonitor: checks={diag["total_checks"]} '
            f'violations={diag["total_violations"]} '
            f'rate={diag["violation_rate"]*100:.1f}% '
            f'active={active if active else "none"}')

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

    # ------------------------------------------------------------------
    # Replay mode helpers
    # ------------------------------------------------------------------

    def _init_compensator(self):
        """Initialize actuator delay compensator.

        The compensator pre-distorts joint commands to cancel the real actuator's
        transport delay + first-order lag. This must match the compensation used
        during training (inverse_mpc with 72ms delay, tau=0.074s).
        """
        self.compensator = None

        enable_comp = bool(self.get_parameter('enable_compensation').value)

        if not enable_comp:
            self.get_logger().info('Actuator compensation DISABLED')
            return

        if not _COMPENSATION_AVAILABLE or CommandCompensator is None:
            self.get_logger().error(
                'CommandCompensator not available (pingpong_replay not installed). '
                'Actuator compensation DISABLED.')
            return

        if self.rl_controller is None:
            self.get_logger().warn(
                'Cannot initialize compensator: RL controller not available')
            return

        try:
            # Get compensation parameters from ROS params
            comp_mode = str(self.get_parameter('compensation_mode').value)

            # Get joint limits from RL controller
            q_low = self.rl_controller.arm_q_low
            q_high = self.rl_controller.arm_q_high

            self.compensator = CommandCompensator(
                n_joints=7,
                dt=self.control_dt,
                mode=comp_mode,
                delay_ms=float(self.get_parameter('comp_delay_ms').value),
                actuator_tau_s=float(self.get_parameter('comp_actuator_tau_s').value),
                actuator_gain=float(self.get_parameter('comp_actuator_gain').value),
                command_buffer_extra_steps=int(
                    self.get_parameter('comp_buffer_extra_steps').value),
                delay_conditioning=bool(
                    self.get_parameter('comp_delay_conditioning').value),
                # inverse-MPC parameters
                mpc_beta=float(self.get_parameter('comp_mpc_beta').value),
                mpc_delay_scale=float(self.get_parameter('comp_mpc_delay_scale').value),
                mpc_tau_scale=float(self.get_parameter('comp_mpc_tau_scale').value),
                mpc_horizon_steps=int(self.get_parameter('comp_mpc_horizon_steps').value),
                mpc_tracking_weight=float(
                    self.get_parameter('comp_mpc_tracking_weight').value),
                mpc_nominal_weight=float(
                    self.get_parameter('comp_mpc_nominal_weight').value),
                mpc_delta_weight=float(self.get_parameter('comp_mpc_delta_weight').value),
                mpc_max_delta_rad=float(
                    self.get_parameter('comp_mpc_max_delta_rad').value),
                q_low=q_low,
                q_high=q_high,
            )

            self.get_logger().info(
                f'✅ Actuator compensation initialized: mode={comp_mode}, '
                f'delay={self.compensator.delay_ms:.1f}ms, '
                f'tau={self.compensator.actuator_tau_s:.3f}s, '
                f'buffer_len={self.compensator.command_buffer_len}')

        except Exception as e:
            self.get_logger().error(
                f'Failed to initialize compensator: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.compensator = None

    def _init_replay_mode(self):
        """Initialize replay mode: load CSV data."""
        import pandas as pd

        if not self.replay_csv_path:
            self.get_logger().error('Replay mode enabled but replay_csv_path is empty!')
            self.enable_replay_mode = False
            return

        try:
            self._replay_data = pd.read_csv(self.replay_csv_path)
            self._replay_idx = 0
            self._replay_start_time = None

            self.get_logger().info(
                f'✅ Replay mode enabled: loaded {len(self._replay_data)} frames from {self.replay_csv_path}')
            self.get_logger().info(
                f'   Duration: {self._replay_data["t_rel"].max():.2f}s, '
                f'Loop: {self.replay_loop}')

            # FIX: Initialize RL controller state from replay data to match MJX
            # validation baseline. This ensures the first observation uses actual
            # values instead of zeros, matching the MJX environment's reset().
            if self.enable_rl_policy and self.rl_controller is not None:
                self._init_rl_controller_from_replay_data()

        except Exception as e:
            self.get_logger().error(f'Failed to load replay CSV: {e}')
            self.enable_replay_mode = False

    def _init_rl_controller_from_replay_data(self):
        """Initialize RL controller state from the first frame of replay data.

        This matches how MJX environment initializes at reset(), ensuring the
        first predict() call produces observations identical to the validation
        baseline rather than using default zero-initialized states.
        """
        self.get_logger().info('🔧 Attempting to initialize RL controller from MJX reference data...')

        if self._replay_data is None or len(self._replay_data) == 0:
            self.get_logger().warn('Replay data is None or empty, skipping RL controller initialization')
            return

        # CRITICAL: The replay CSV (mjx_ball_replay.csv) only contains ball state,
        # NOT joint velocities. We need to load from the reference observation CSV
        # to get the initial joint velocities for proper initialization.

        # Load reference observation trace that contains full state
        # Try multiple possible paths (source vs install)
        module_dir = os.path.dirname(__file__)
        source_module_dir = self._resolve_source_module_dir(module_dir)

        reference_obs_path = os.path.join(
            source_module_dir,
            'data/rl_sim/validate_obs_trace/validate_obs_trace.csv'
        )

        if not os.path.isfile(reference_obs_path):
            self.get_logger().warn(
                f'Reference observation trace not found: {reference_obs_path}. '
                'Cannot initialize joint velocities from MJX baseline.')
            return

        try:
            import pandas as pd
            reference_df = pd.read_csv(reference_obs_path)
            first_row = reference_df.iloc[0]

            # Initialize joint velocity estimate from MJX validation data
            # This avoids the 5-7 step finite-difference warmup delay
            arm_dq_cols = [f'obs_arm_dq_rad_s/RightArm-{i}' for i in range(7)]
            if all(col in reference_df.columns for col in arm_dq_cols):
                initial_dq = np.array([
                    first_row[col] for col in arm_dq_cols
                ], dtype=np.float32)

                # Directly set the attribute (lock not yet created during __init__)
                self._estimated_right_arm_dq = initial_dq
                self.get_logger().info(
                    f'✅ Initialized arm velocities from MJX reference (rad/s): '
                    f'{np.round(initial_dq, 4).tolist()}')
            else:
                self.get_logger().warn(
                    'Reference CSV missing obs_arm_dq_rad_s columns. '
                    'Joint velocity initialization skipped.')

            # Initialize joint position estimate as well for consistency
            arm_q_cols = [f'obs_arm_q_rad/RightArm-{i}' for i in range(7)]
            if all(col in reference_df.columns for col in arm_q_cols):
                initial_q = np.array([
                    first_row[col] for col in arm_q_cols
                ], dtype=np.float32)

                # Directly set the attributes (lock not yet created during __init__)
                self._prev_right_arm_q_for_dq = initial_q.copy()
                self._prev_right_arm_stamp_for_dq = time.time()

                self.get_logger().info(
                    f'✅ Initialized arm positions from MJX reference (rad): '
                    f'{np.round(initial_q, 4).tolist()}')

            self.get_logger().info(
                '✅ RL controller state initialized from MJX validation baseline')

        except Exception as e:
            self.get_logger().error(
                f'Failed to initialize from reference observation trace: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _init_sim_ball_input(self):
        """Initialize simulation ball state input: subscribe to sim ball topic."""
        from nav_msgs.msg import Odometry

        self.create_subscription(
            Odometry,
            self.sim_ball_topic,
            self._sim_ball_callback,
            10
        )
        self.get_logger().info(
            f'✅ Simulation ball input enabled: subscribing to "{self.sim_ball_topic}"')

    def _init_mocap_input(self):
        """Initialize motion capture ball state input: start UDP receiver."""
        from pingpong_controller.mocap_ball_detector import MocapBallDetector

        try:
            self._mocap_detector = MocapBallDetector(
                udp_host=self.mocap_bind_host,
                udp_port=self.mocap_udp_port,
                calibration_path=self.mocap_calibration_path if self.mocap_calibration_path else None,
                timeout_s=self.mocap_timeout_s,
                marker_id=self.mocap_marker_id,
                position_unit=self.mocap_position_unit,
                timestamp_unit=self.mocap_timestamp_unit,
                velocity_alpha=self.mocap_velocity_alpha,
                max_velocity_m_s=self.mocap_max_velocity_m_s
            )
            self._mocap_detector.start()
            self.get_logger().info(
                f'✅ Motion capture input enabled: '
                f'{self.mocap_bind_host}:{self.mocap_udp_port}, '
                f'marker {self.mocap_marker_id}, '
                f'unit={self.mocap_position_unit}, '
                f'timestamp_unit={self.mocap_timestamp_unit}, '
                f'timeout {self.mocap_timeout_s}s')
            if self.mocap_calibration_path:
                self.get_logger().info(
                    f'   Calibration: {self.mocap_calibration_path}')
            else:
                self.get_logger().warn(
                    '   No calibration file specified - using identity transform!')
        except Exception as e:
            self.get_logger().error(
                f'Failed to initialize mocap input: {e}')
            self.enable_mocap_input = False

    def _sim_ball_callback(self, msg) -> None:
        """Callback for simulation ball state (Odometry message)."""
        # Extract position (in meters, convert to mm for consistency)
        pos_mm = np.array([
            msg.pose.pose.position.x * 1000.0,
            msg.pose.pose.position.y * 1000.0,
            msg.pose.pose.position.z * 1000.0
        ], dtype=np.float32)

        # Extract velocity (in m/s, convert to mm/s)
        vel_mm_s = np.array([
            msg.twist.twist.linear.x * 1000.0,
            msg.twist.twist.linear.y * 1000.0,
            msg.twist.twist.linear.z * 1000.0
        ], dtype=np.float32)

        # Create BallState
        ball_state = BallState(
            position=pos_mm,
            velocity=vel_mm_s,
            stamp=msg.header.stamp,
            valid=True
        )

        with self._sim_ball_lock:
            self._sim_ball_state = ball_state

    def _get_mocap_ball_state(self) -> Optional[BallState]:
        """Generate BallState from motion capture detector.

        Returns:
            BallState with position/velocity in mm/mm_s (robot base_link frame),
            or None if mocap detector is not available
        """
        if self._mocap_detector is None:
            return None

        # Get ball state from mocap detector
        pos_m, vel_m_s, valid = self._mocap_detector.get_ball_state()
        diagnostics = self._mocap_detector.get_diagnostics()

        # Convert from meters to mm for consistency with vision pipeline
        pos_mm = pos_m * 1000.0
        vel_mm_s = vel_m_s * 1000.0

        # Create BallState
        now_time = self.get_clock().now()
        raw_measurement = {
            'source': 'mocap',
            'detection_method': 'mocap_udp',
            'mocap': diagnostics,
        }
        ball_state = BallState(
            position=pos_mm.astype(np.float64),
            velocity=vel_mm_s.astype(np.float64),
            stamp=now_time.to_msg(),
            valid=valid,
            raw_measurement=raw_measurement
        )

        packet_count = int(diagnostics.get('packet_count', 0) or 0)
        is_new_packet = packet_count > self._last_mocap_estimator_packet_count
        if valid and is_new_packet:
            stamp_sec = diagnostics.get('arrival_time_s')
            if stamp_sec is None or not np.isfinite(float(stamp_sec)):
                stamp_sec = now_time.nanoseconds * 1e-9
            else:
                stamp_sec = float(stamp_sec)

            if self.enable_ball_state_estimator:
                observation = {
                    'detected': True,
                    'position': ball_state.position.tolist(),
                    'velocity': ball_state.velocity.tolist(),
                    'raw_measurement': raw_measurement,
                }
                self._feed_ball_state_estimator(observation, stamp_sec)
            self._last_mocap_estimator_packet_count = packet_count

            if packet_count > self._last_mocap_recorded_packet_count:
                self._record_base_trajectory(stamp_sec, ball_state)
                self._last_mocap_recorded_packet_count = packet_count

        return ball_state

    def _get_replay_ball_state(self) -> Optional[BallState]:
        """Generate BallState from replay CSV data."""
        if self._replay_data is None or len(self._replay_data) == 0:
            return None

        # Initialize start time on first call
        if self._replay_start_time is None:
            self._replay_start_time = time.time()

        # Calculate current replay time
        elapsed = time.time() - self._replay_start_time

        # Handle loop mode
        duration = self._replay_data['t_rel'].max()
        if elapsed > duration:
            if self.replay_loop:
                elapsed = elapsed % duration
                self._replay_idx = 0
            else:
                # Replay finished
                return None

        # Find the closest frame
        df = self._replay_data
        idx = (df['t_rel'] - elapsed).abs().argmin()
        row = df.iloc[idx]

        # Create BallState with current ROS time
        now_time = self.get_clock().now()
        state = BallState(
            valid=bool(row['valid']),
            position=np.array([row['x_mm'], row['y_mm'], row['z_mm']], dtype=np.float64),
            velocity=np.array([row['vx_mm_s'], row['vy_mm_s'], row['vz_mm_s']], dtype=np.float64),
            stamp=now_time.to_msg(),
        )

        if (self.enable_ball_state_estimator
                and state.valid
                and idx != self._last_replay_estimator_idx):
            raw_measurement = {
                'source': 'replay',
                'replay_idx': int(idx),
                'replay_t_rel': float(row['t_rel']),
                'frame': 'sim_frame' if self.replay_data_is_sim_frame else 'base_link',
            }
            observation = {
                'detected': True,
                'position': state.position.tolist(),
                'velocity': state.velocity.tolist(),
                'raw_measurement': raw_measurement,
            }
            stamp_sec = now_time.nanoseconds * 1e-9
            self._feed_ball_state_estimator(observation, stamp_sec)
            self._last_replay_estimator_idx = idx

        self._replay_idx = idx
        return state

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

    def _compute_init_pose_ramp_command(
            self, target_q: np.ndarray, elapsed_s: float) -> np.ndarray:
        """Return a smooth right-arm init-pose command for the current tick."""
        target = np.asarray(target_q, dtype=np.float64).reshape(-1)
        if self._init_pose_start_q is None:
            q, _ = self._get_right_arm_feedback()
            if q is not None:
                self._init_pose_start_q = q.astype(np.float64)
                self._init_pose_start_source = '/joint_states'
            else:
                self._init_pose_start_q = np.asarray(
                    self._last_safe_joint_positions,
                    dtype=np.float64).reshape(-1).copy()
                self._init_pose_start_source = 'last command'
                self.get_logger().warn(
                    'No right-arm /joint_states yet for init-pose ramp; '
                    'using last command as ramp start',
                    throttle_duration_sec=2.0)

            self.get_logger().info(
                'Init-pose ramp start '
                f'({self._init_pose_start_source}, deg='
                f'{np.rad2deg(self._init_pose_start_q).round(2).tolist()})')

        duration = max(float(self.init_pose_duration_s), 1e-6)
        alpha = float(np.clip(elapsed_s / duration, 0.0, 1.0))
        # Smoothstep gives zero slope at both ends, which avoids a sharp
        # command jump when safety limiting is intentionally disabled.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return self._init_pose_start_q + alpha * (target - self._init_pose_start_q)

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

    def _ball_state_to_policy_sim_frame(
            self, ball_state: BallState) -> tuple[np.ndarray, np.ndarray]:
        """Return ball position/velocity in the sim-frame convention for RL."""
        # BallState carries mm / mm/s; the RL policy expects m / m/s.
        ball_pos_m = np.asarray(
            ball_state.position, dtype=np.float32) * 1e-3
        ball_vel_m_s = np.asarray(
            ball_state.velocity, dtype=np.float32) * 1e-3

        # CRITICAL: Transform ball coordinates from the real Cartesian frame
        # (waist02) to the simulation frame to match training observations.
        #
        # Simulation coordinate system (MJX):
        #   - Uses _point_to_base() which returns [x_rel, y_rel, z_world]
        #   - x, y: relative to base, rotated by yaw
        #   - z: ABSOLUTE world frame z (NOT relative to base!)
        #
        # Real system coordinate system:
        #   - Ball position is in the robot Cartesian frame via T_base_camera
        #   - That Cartesian frame corresponds to moz1_pd.xml body waist02
        #   - x, y, z: ALL relative to waist02 origin
        #
        # Transform: p_sim = p_waist02 + waist02_origin_in_sim_base.
        #
        # IMPORTANT: Skip transformation if replay data is already in sim frame
        # (e.g., mjx_ball_replay.csv from simulation).
        if not (self.enable_replay_mode and self.replay_data_is_sim_frame):
            ball_pos_m = self._transform_ball_pos_to_sim_frame(ball_pos_m)
        return ball_pos_m, ball_vel_m_s

    def compute_action(self, ball_state: BallState,
                       age_override: Optional[float] = None) -> np.ndarray:
        """Compute right-arm joint targets (rad) using the RL policy.

        Always invoked from the control timer, including when the detection
        is invalid — RLPolicyController.predict() holds the last valid ball
        observation internally and reports the age, matching the training
        env's dropout/hold-last behavior. The safety limiter, not this
        function, is responsible for clamping the output before publish.

        ``age_override`` (seconds), when provided, is the ball-observation age
        the policy should see. The ball-state estimator passes its graded age
        here: a freshly de-aged extrapolation reports ~0, and a stale one
        reports a growing age so the policy discounts it. When None (estimator
        off) the age is derived from the wall-clock gap exactly as before.
        """
        if not self.enable_rl_policy or self.rl_controller is None:
            return self._last_raw_joint_positions.astype(np.float64)

        now_wall = time.time()
        if ball_state.valid:
            # A valid ball (real detection or trusted estimator extrapolation)
            # refreshes the loss clock the failsafe reads.
            self._last_valid_ball_wall_time = now_wall
            age_s = age_override if age_override is not None else 0.0
        elif age_override is not None:
            age_s = age_override
        elif self._last_valid_ball_wall_time is not None:
            age_s = max(0.0, now_wall - self._last_valid_ball_wall_time)
        else:
            age_s = self.rl_ball_obs_age_clip
        age_s = float(max(0.0, age_s))

        ball_pos_m, ball_vel_m_s = self._ball_state_to_policy_sim_frame(
            ball_state)

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

        # Stash the sim-frame ball state for the viz done-condition overlay.
        self._last_sim_ball_pos_m = ball_pos_m.copy()
        self._last_sim_ball_vel_m_s = ball_vel_m_s.copy()
        self._last_sim_ball_valid = bool(ball_state.valid)

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

            # Record policy obs and action for debugging
            if self.save_policy_log:
                self._record_policy_log(
                    ball_pos_m, ball_vel_m_s, ball_state.valid,
                    age_s, right_arm_q, right_arm_dq)

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

        # Apply actuator delay compensation if enabled.
        # This pre-distorts the nominal command to cancel the real actuator's
        # transport delay + first-order lag, matching what the training env did.
        if self.compensator is not None and self.compensator.enabled:
            try:
                # Get nominal command derivatives from RL controller
                cmd_state = self.rl_controller.latest_command_state()
                nominal_qvel = cmd_state['velocity']
                nominal_qdd = cmd_state['acceleration']

                # Get actual feedback (or use nominal if not available)
                actual_q, _ = self._get_right_arm_feedback()
                applied_q = (actual_q if actual_q is not None
                            else joint_cmd)

                # Apply compensation
                joint_cmd = self.compensator.step(
                    q_cmd=joint_cmd.astype(np.float32),
                    qdot_cmd=nominal_qvel,
                    qdd_cmd=nominal_qdd,
                    applied_q=applied_q.astype(np.float32),
                ).astype(np.float64)

            except Exception as e:
                self.get_logger().error(
                    f'Compensation failed: {e}; using uncompensated command',
                    throttle_duration_sec=1.0)

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
        if self.warn_no_command_subscriber:
            try:
                sub_count = self.cmd_pub.get_subscription_count()
            except Exception:  # noqa: BLE001 - diagnostic only
                sub_count = 1
            if sub_count <= 0:
                now_wall = time.time()
                if (now_wall - self._last_cmd_pub_no_sub_warn_wall
                        >= self.command_subscriber_warn_period_s):
                    self._last_cmd_pub_no_sub_warn_wall = now_wall
                    self.get_logger().warn(
                        'No subscribers on command_topic; MechUnitCmdArray is '
                        'being published but no controller is receiving it')
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

        # Apply safety limiter if enabled
        if self.enable_safety_limiter:
            safe_cmd = self.safety_limiter.filter(raw)
            self._maybe_log_safety_counts()
        else:
            # Bypass safety limiter - use raw command directly
            safe_cmd = raw.copy()

        self._last_safe_joint_positions = safe_cmd.copy()

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

        # Apply safety limiter if enabled
        if self.enable_safety_limiter:
            safe_cmd = self.safety_limiter.filter(raw)
            self._maybe_log_safety_counts()
        else:
            # Bypass safety limiter - use raw command directly
            safe_cmd = raw.copy()

        self._last_safe_joint_positions = safe_cmd.copy()

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

    def _enqueue_raw_video_frame(self, raw_bgr: np.ndarray,
                                 depth_image: np.ndarray = None,
                                 ir_image: np.ndarray = None,
                                 t_capture: float = None) -> None:
        """Hand one raw RealSense frame bundle to the writer thread.

        Called from the RealSense/hybrid reader loop. O(1) non-blocking: copies
        the frame(s) (the RealSense buffer is recycled after the callback) and
        pushes them onto the writer queue as one bundle, dropping the whole
        bundle only if the queue is full (disk/encoder can't keep up). Bundling
        RGB+depth+IR+timestamp keeps their frame indices locked together.
        """
        if self._raw_video_writer_thread is None:
            self._start_raw_video_writer_thread()
        if self._raw_video_queue is None:
            return
        item = {
            'bgr': raw_bgr.copy(),
            'depth': depth_image.copy() if depth_image is not None else None,
            'ir': ir_image.copy() if ir_image is not None else None,
            't_capture': t_capture,
        }
        try:
            self._raw_video_queue.put(item, block=False)
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
        """Dedicated thread: drain the queue and encode raw frame bundles."""
        while self._raw_video_writer_running:
            try:
                item = self._raw_video_queue.get(timeout=0.1)
                self._write_raw_video_frame(item)
            except Empty:
                continue
            except Exception as exc:  # noqa: BLE001 - keep node alive
                self.get_logger().error(
                    f'Raw video writer thread error: {exc}; stopping '
                    'raw recording', throttle_duration_sec=2.0)
                self._raw_video_disabled_runtime = True
                break

    def _write_raw_video_frame(self, item) -> None:
        """Open the writers lazily and append one RGB(+depth+timestamp) bundle.

        Runs on the raw-video writer thread. The first frame fixes resolution
        and fps for the whole file. Any failure disables raw recording for the
        rest of the run without killing the thread. ``item`` is the bundle dict
        produced by ``_enqueue_raw_video_frame`` (or a bare ndarray for
        backward-compat). RGB, depth, and timestamp share one frame index
        (``_recording_index``) so the offline tool pairs them positionally.
        """
        if isinstance(item, dict):
            raw_bgr = item.get('bgr')
            depth_image = item.get('depth')
            ir_image = item.get('ir')
            t_capture = item.get('t_capture')
        else:  # backward-compat: a bare BGR frame
            raw_bgr, depth_image, ir_image, t_capture = item, None, None, None
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
                idx = self._recording_index
                # Depth + IR + timestamp sidecars, keyed by this RGB frame index.
                if self.save_depth_frames and not self._depth_disabled_runtime:
                    self._write_depth_sidecar(idx, depth_image)
                if self.save_ir_frames and not self._ir_disabled_runtime:
                    self._write_ir_sidecar(idx, ir_image)
                if t_capture is not None:
                    self._write_timestamp_sidecar(idx, t_capture)
                self._raw_video_frames_written += 1
                self._recording_index += 1
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
        if not self._open_recording_sidecars(path):
            # Sidecar failure must not kill RGB recording; just disable depth.
            self.save_depth_frames = False
            self._depth_disabled_runtime = True
        return True

    def _open_recording_sidecars(self, video_path: str) -> bool:
        """Create the timestamp CSV and depth-frame dir next to the raw video.

        The depth directory and the timestamp CSV are named from the raw video
        stem so a recording's three artifacts (``*.mp4``, ``*_timestamps.csv``,
        ``*_depth/``) stay grouped. Caller holds ``_raw_video_lock``. The
        timestamp CSV is always opened (it costs nothing and helps the offline
        replay even for RGB-only runs); the depth dir only when recording depth.
        Returns False only on a depth-dir failure the caller should act on.
        """
        stem = os.path.splitext(video_path)[0]
        # Per-frame timestamp sidecar (always; one row per written RGB frame).
        try:
            self._timestamps_csv_path = stem + '_timestamps.csv'
            self._timestamps_file = open(
                self._timestamps_csv_path, 'w', buffering=1)
            self._timestamps_file.write('frame_idx,t_capture_sec\n')
        except OSError as exc:
            self.get_logger().warn(
                f'Cannot open timestamp sidecar {self._timestamps_csv_path}: '
                f'{exc}; offline replay will fall back to constant fps')
            self._timestamps_file = None
            self._timestamps_csv_path = None
        # Depth frame directory (only when recording hardware depth).
        if self.save_depth_frames:
            self._depth_frames_dir = stem + '_depth'
            try:
                os.makedirs(self._depth_frames_dir, exist_ok=True)
                self._depth_frames_written = 0
                self.get_logger().info(
                    f'Recording aligned depth frames -> '
                    f'{self._depth_frames_dir}/ (16-bit .npy per frame)')
            except OSError as exc:
                self.get_logger().error(
                    f'Cannot create depth frame dir {self._depth_frames_dir}: '
                    f'{exc}; depth recording disabled')
                self._depth_frames_dir = None
                return False
        # IR frame directory + camera_params JSON (only in hybrid IR recording).
        if self.save_ir_frames:
            self._ir_frames_dir = stem + '_ir'
            try:
                os.makedirs(self._ir_frames_dir, exist_ok=True)
                self._ir_frames_written = 0
                self.get_logger().info(
                    f'Recording IR frames -> '
                    f'{self._ir_frames_dir}/ (uint8 .npy per frame)')
            except OSError as exc:
                self.get_logger().error(
                    f'Cannot create IR frame dir {self._ir_frames_dir}: '
                    f'{exc}; IR recording disabled')
                self._ir_frames_dir = None
                self._ir_disabled_runtime = True
            self._write_camera_params_sidecar(stem)
        return True

    def _write_camera_params_sidecar(self, stem: str) -> None:
        """Dump the hybrid detector's intrinsics/extrinsics to <stem>_camera_params.json.

        Written once. The offline hybrid tool rebuilds rs.intrinsics /
        rs.extrinsics from this so its projection math matches the live run.
        Requires the hybrid detector (which owns the IR/RGB intrinsics and the
        IR->RGB extrinsics read from the live RealSense profile).
        """
        if self._camera_params_written:
            return
        inner = getattr(self.detector, 'detector', None)
        if inner is None or not hasattr(inner, 'export_camera_params'):
            self.get_logger().warn(
                'save_ir_frames set but detector has no export_camera_params; '
                'offline hybrid replay will be missing camera_params JSON.')
            return
        try:
            params = inner.export_camera_params()
            path = stem + '_camera_params.json'
            with open(path, 'w') as fh:
                json.dump(params, fh, indent=2)
            self._camera_params_written = True
            self.get_logger().info(f'Wrote camera params -> {path}')
        except Exception as exc:  # noqa: BLE001 - non-fatal for recording
            self.get_logger().warn(
                f'Failed to write camera_params JSON: {exc}; offline hybrid '
                'replay will be missing it.')

    def _write_ir_sidecar(self, idx: int, ir_image) -> None:
        """Save one IR frame as ``ir_<idx>.npy``. Holds the lock.

        A missing IR frame is skipped (no file); the offline tool treats an
        absent ir_<idx>.npy as 'no detection this frame' so indices stay aligned
        with the RGB video.
        """
        if self._ir_frames_dir is None or ir_image is None:
            return
        try:
            np.save(
                os.path.join(self._ir_frames_dir, f'ir_{idx:06d}.npy'),
                np.ascontiguousarray(ir_image))
            self._ir_frames_written += 1
        except Exception as exc:  # noqa: BLE001 - keep recording alive
            self.get_logger().warn(
                f'IR frame write failed at idx {idx}: {exc}; '
                'disabling IR recording',
                throttle_duration_sec=2.0)
            self._ir_disabled_runtime = True

    def _write_depth_sidecar(self, idx: int, depth_image) -> None:
        """Save one aligned depth frame as ``depth_<idx>.npy``. Holds the lock.

        A missing depth frame (transient align failure) is written as an empty
        marker so the frame index stays aligned with the RGB video; the offline
        tool treats a zero-size / absent depth frame as 'no hardware depth this
        frame' and falls back to monocular.
        """
        if self._depth_frames_dir is None or depth_image is None:
            return
        try:
            np.save(
                os.path.join(self._depth_frames_dir, f'depth_{idx:06d}.npy'),
                np.ascontiguousarray(depth_image))
            self._depth_frames_written += 1
        except Exception as exc:  # noqa: BLE001 - keep recording alive
            self.get_logger().warn(
                f'Depth frame write failed at idx {idx}: {exc}; '
                'disabling depth recording',
                throttle_duration_sec=2.0)
            self._depth_disabled_runtime = True

    def _write_timestamp_sidecar(self, idx: int, t_capture: float) -> None:
        """Append one ``frame_idx,t_capture_sec`` row. Caller holds the lock."""
        if self._timestamps_file is None:
            return
        try:
            self._timestamps_file.write(f'{idx},{t_capture:.6f}\n')
        except Exception:  # noqa: BLE001 - non-fatal; stop timestamp logging
            self._timestamps_file = None

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
        # Close the depth + timestamp sidecars in lockstep with the video.
        if self._timestamps_file is not None:
            try:
                self._timestamps_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._timestamps_file = None
        if self._depth_frames_dir is not None:
            self.get_logger().info(
                f'Saved aligned depth frames: {self._depth_frames_dir}/ '
                f'({self._depth_frames_written} frames)')
        if self._ir_frames_dir is not None:
            self.get_logger().info(
                f'Saved IR frames: {self._ir_frames_dir}/ '
                f'({self._ir_frames_written} frames)')

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
                        item = self._raw_video_queue.get_nowait()
                    except Empty:
                        break
                    self._write_raw_video_frame(item)
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

        The sim-frame ball state + done-condition block is drawn for BOTH the
        hybrid detector's own composite render and the non-hybrid overlay.
        When RL is active, it displays the control-loop sim state that is fed to
        the policy and done monitor; otherwise it falls back to the raw frame.
        """
        # Sim-frame ball state + done overlay pieces.
        (sim_ball_pos_m, sim_ball_vel_m_s, sim_detected,
         sim_label) = self._compute_viz_sim_ball_state(observation)
        racket_anchor_m = self._get_racket_anchor_sim()
        racket_pos_m = self._get_racket_pos_sim()
        detected = bool(observation.get('detected', False))

        # Hybrid detector已经完成渲染（RGB|IR 双图）：把 sim/done 叠加画到成品图后返回。
        if observation.get('_hybrid_viz_rendered', False):
            if sim_ball_pos_m is not None and sim_ball_vel_m_s is not None:
                draw_sim_done_overlay(
                    frame, sim_ball_pos_m, sim_ball_vel_m_s, racket_anchor_m,
                    racket_pos_m=racket_pos_m,
                    thresholds=self._viz_done_thresholds(),
                    detected=sim_detected,
                    label=sim_label)
            return frame

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
            output_frame=output_frame,
            sim_ball_pos_m=sim_ball_pos_m,
            sim_ball_vel_m_s=sim_ball_vel_m_s,
            racket_anchor_m=racket_anchor_m,
            racket_pos_m=racket_pos_m,
            done_thresholds=self._viz_done_thresholds(),
            sim_detected=sim_detected,
            sim_label=sim_label)

    def _viz_done_thresholds(self) -> dict:
        """Done-condition thresholds for the viz overlay (from node params)."""
        return {
            'ball_z_floor': self.done_ball_z_floor_m,
            'ball_z_ceiling': self.done_ball_z_ceiling_m,
            'ball_x_range': self.done_ball_x_range_m,
            'ball_y_range': self.done_ball_y_range_m,
            'racket_anchor_distance': self.done_racket_anchor_distance_m,
            'racket_z_high_offset': self.done_racket_z_high_offset_m,
            'racket_z_low_offset': self.done_racket_z_low_offset_m,
        }

    def _compute_viz_sim_ball_state(self, observation: dict):
        """Ball pos (m, sim frame) + vel (m/s) for the viz done overlay.

        Prefer the control-loop sim state when available, because that is the
        value fed to RL and done monitoring. Fall back to the per-frame detector
        observation before the first control tick, or when RL is disabled.
        """
        if (self.enable_rl_policy and self.rl_controller is not None
                and self._last_sim_ball_pos_m is not None
                and self._last_sim_ball_vel_m_s is not None):
            return (
                np.asarray(self._last_sim_ball_pos_m, dtype=np.float64).reshape(3),
                np.asarray(self._last_sim_ball_vel_m_s, dtype=np.float64).reshape(3),
                bool(self._last_sim_ball_valid),
                "[SIM control state]",
            )

        if not bool(observation.get('detected', False)):
            return None, None, False, "[SIM raw frame]"
        pos = observation.get('position')
        if pos is None:
            return None, None, False, "[SIM raw frame]"
        vel = observation.get('velocity')
        if vel is None:
            vel = [0.0, 0.0, 0.0]
        try:
            pos_m = np.asarray(pos, dtype=np.float64).reshape(3) * 1e-3
            vel_m_s = np.asarray(vel, dtype=np.float64).reshape(3) * 1e-3
        except (TypeError, ValueError):
            return None, None, False, "[SIM raw frame]"
        # Real Cartesian/waist02 frame -> sim frame, matching the transform used
        # to feed the policy.
        pos_sim = self._transform_ball_pos_to_sim_frame(pos_m)
        return pos_sim, vel_m_s, True, "[SIM raw frame]"

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

    def _transform_ball_pos_to_sim_frame(self, ball_pos_m: np.ndarray) -> np.ndarray:
        """Transform ball position from real Cartesian/waist02 frame to sim frame.

        Coordinate system difference (from code analysis):

        Simulation (mjx_juggle_env._point_to_base):
            def _point_to_base(self, point, base_q):
                ...
                return jnp.stack([x_rot, y_rot, point[:, 2]], axis=-1)
                                                  ^^^^^^^^^^
            Key: z directly returns point[:, 2] WITHOUT subtracting base_q's z!
            Result: [x_rel, y_rel, z_world_absolute]

        Real system (via T_base_camera transformation):
            All coordinates relative to the robot Cartesian frame, which
            corresponds to moz1_pd.xml's waist02 body.
            Result: [x_waist02, y_waist02, z_waist02]

        Transform:
            p_sim = p_waist02 + waist02_origin_in_sim_base

        where waist02_origin_in_sim_base is computed from moz1_pd.xml at the
        TARGET_DEGREES posture. Default [0.0, 0.023583, 0.947396] m.

        Args:
            ball_pos_m: Ball position in waist02 frame [x, y, z] in meters

        Returns:
            Ball position in simulation frame [x, y, z_world] in meters
        """
        transformed = ball_pos_m.copy()
        transformed[0] = ball_pos_m[0] + self._sim_cartesian_x_offset_m
        transformed[1] = ball_pos_m[1] + self._sim_cartesian_y_offset_m
        transformed[2] = ball_pos_m[2] + self._sim_base_height_m
        return transformed

    # ------------------------------------------------------------------
    # Racket-anchor FK for the viz done-condition overlay (RL-independent)
    # ------------------------------------------------------------------

    def _init_viz_fk(self) -> None:
        """Load a standalone MuJoCo model for the racket anchor / racket pose.

        The done-condition overlay needs the racket anchor (right_ee_site at the
        training initial posture) and, when joint feedback is available, the
        current racket pose. The RL policy already does this FK, but it is absent
        when enable_rl_policy:=false, so we build our own lightweight model here.
        Mirrors RLPolicyController's FK: non-right-arm joints seeded to
        TARGET_DEGREES, world->base yaw transform. Sets self._viz_racket_anchor_sim
        (constant, sim frame) and keeps the model for per-frame racket pose FK.
        """
        self._viz_mj_model = None
        self._viz_mj_data = None
        self._viz_racket_site_id = -1
        self._viz_arm_jids = []
        self._viz_base_body_id = -1
        self._viz_racket_anchor_sim = None
        if not _MUJOCO_AVAILABLE:
            self.get_logger().info(
                'Viz FK: mujoco not available; sim done-condition overlay will '
                'show ball pos/vel and z done-conditions only.')
            return
        xml = self.robot_xml_path
        if not xml or not os.path.exists(xml):
            self.get_logger().warn(
                f'Viz FK: robot_xml_path not found ({xml}); racket anchor '
                'unavailable; done overlay will show ball pos/vel and z only.')
            return
        try:
            model = _mj.MjModel.from_xml_path(str(xml))
            data = _mj.MjData(model)
            site_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_SITE, 'right_ee_site')
            arm_jids = [
                _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, n)
                for n in RIGHT_ARM_JOINTS]
            if site_id < 0 or any(j < 0 for j in arm_jids):
                self.get_logger().warn(
                    'Viz FK: model missing right_ee_site or right-arm joints; '
                    'done overlay will show ball pos/vel and z only.')
                return
            base_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, 'base_link')
            if base_bid < 0:
                base_bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, 'base')
            self._viz_mj_model = model
            self._viz_mj_data = data
            self._viz_racket_site_id = int(site_id)
            self._viz_arm_jids = arm_jids
            self._viz_base_body_id = int(base_bid)
            # Seed non-right-arm joints to TARGET_DEGREES (matches training FK).
            right_set = set(RIGHT_ARM_JOINTS)
            for name, deg in TARGET_DEGREES.items():
                if name in right_set:
                    continue
                jid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    continue
                qadr = int(model.jnt_qposadr[jid])
                data.qpos[qadr] = float(np.deg2rad(deg))
            # Anchor = right_ee_site at the training initial right-arm posture.
            anchor_base = self._viz_racket_pose_base(
                np.asarray(target_right_arm_q_rad(), dtype=np.float64))
            if anchor_base is not None:
                # _viz_racket_pose_base follows the MJX base-yaw convention:
                # x/y are base-relative, z is already world/sim height.
                self._viz_racket_anchor_sim = anchor_base
                self.get_logger().info(
                    'Viz FK: racket anchor (sim frame) = '
                    f'{np.round(self._viz_racket_anchor_sim, 3).tolist()} m')
        except Exception as exc:  # noqa: BLE001 - non-fatal for viz
            self.get_logger().warn(
                f'Viz FK init failed: {exc}; done overlay will show ball '
                'pos/vel and z only.')
            self._viz_mj_model = None
            self._viz_mj_data = None

    def _viz_racket_pose_base(self, arm_q: np.ndarray):
        """FK the right arm to `arm_q`; return racket pos in base yaw frame (m).

        Non-right-arm joints stay seeded to TARGET_DEGREES. Returns a length-3
        numpy array or None if FK is unavailable.
        """
        if self._viz_mj_model is None or self._viz_mj_data is None:
            return None
        model, data = self._viz_mj_model, self._viz_mj_data
        q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        for i, jid in enumerate(self._viz_arm_jids):
            qadr = int(model.jnt_qposadr[jid])
            data.qpos[qadr] = float(q[i])
        _mj.mj_forward(model, data)
        rpos_world = np.asarray(
            data.site_xpos[self._viz_racket_site_id], dtype=np.float64)
        # world -> base yaw frame (matches RLPolicyController._world_point_to_base).
        if self._viz_base_body_id < 0:
            return rpos_world
        base_pos = np.asarray(
            data.xpos[self._viz_base_body_id], dtype=np.float64)
        base_R = np.asarray(
            data.xmat[self._viz_base_body_id], dtype=np.float64).reshape(3, 3)
        yaw = np.arctan2(base_R[1, 0], base_R[0, 0])
        dx = rpos_world[0] - base_pos[0]
        dy = rpos_world[1] - base_pos[1]
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy, rpos_world[2]],
                        dtype=np.float64)

    def _get_racket_anchor_sim(self):
        """Return the racket anchor in sim frame (m), or None.

        Prefers the live RL policy anchor when the policy is running (kept in
        sync with its FK); otherwise the standalone init-time constant.
        """
        if self.rl_controller is not None:
            anchor = getattr(self.rl_controller, '_racket_anchor_pos_base', None)
            if anchor is not None:
                # RLPolicyController stores the racket anchor in the same MJX
                # base-yaw convention: z is already world/sim height.
                return np.asarray(anchor, dtype=np.float64).reshape(3)
        return self._viz_racket_anchor_sim

    def _get_racket_pos_sim(self):
        """Return the CURRENT racket pose in sim frame (m) from joint feedback.

        FK the live right-arm q from /joint_states. Returns None when feedback
        or FK is unavailable (then the overlay omits the racket-z/-far checks).
        """
        q, _ = self._get_right_arm_feedback()
        if q is None:
            return None
        rpos_base = self._viz_racket_pose_base(q)
        if rpos_base is None:
            return None
        # _viz_racket_pose_base already returns sim-frame z (world height).
        return rpos_base

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
            # Stop mocap detector if running
            if hasattr(node, '_mocap_detector') and node._mocap_detector is not None:
                node._mocap_detector.stop()
            node.save_trajectory_plot()
            node.save_estimator_log_file()
            node.save_policy_log_file()
            if executor is not None:
                executor.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
