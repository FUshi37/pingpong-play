#!/usr/bin/env python3
"""Replay a recorded RL right-arm trajectory on the real robot.

The trajectory is the one produced by the simulation policy and logged to
``stage4g_action_trace.csv`` (right arm performing the ping-pong juggle task).
This node reads that CSV, reconstructs the per-step right-arm joint *position*
command, and publishes it as a ``MechUnitCmd`` on ``/mx_mix_command`` at a
fixed control rate -- mirroring how ``pingpong_node`` drives the right arm.

Why everything ends up as a position command
---------------------------------------------
``MechUnitCmd`` only carries joint angles (``jnt_pos`` in rad, ``use_jnt``).
The original controller (see ``pingpong_controller/rl_policy.py``) turns the
policy *action* into a desired joint acceleration, integrates it to a velocity
and then to a position command every control tick, and publishes only that
position.  This node supports four *input* representations of the same
recorded motion and converts each one back to the position stream:

  * ``action``  -- raw policy action in [-1, 1]; multiplied by the per-joint
                   acceleration scale, then double-integrated (qdd -> qvel -> q).
  * ``qdd``     -- desired angular acceleration (rad/s^2); double-integrated.
  * ``qvel``    -- commanded angular velocity (rad/s); single-integrated.
  * ``angle``   -- commanded joint angle (rad); published directly.

All four were verified against the recorded ``arm_cmd_q`` column: reconstruct
error <= 6e-6 rad over the full 1200-step trace, so any mode reproduces the
same motion.  ``angle`` is the most faithful (no integration drift) and is the
recommended default for a first hardware check; the other modes exist so you
can validate the integration path that the live policy actually uses.

Safety
------
A hardware arm is involved, so by default the node:
  * refuses to start streaming until it has read the real arm pose from
    ``/joint_states`` and smoothly interpolates from there to the trajectory
    start over ``move_to_start_duration`` seconds (default 5 s; set
    ``move_to_start:=false`` to skip),
  * holds at the trajectory start pose and waits for the operator to press
    Enter on stdin before beginning the replay (set
    ``require_enter_confirm:=false`` for headless runs), and
  * clips every published angle to ``[q_low, q_high]``.
The replay then plays exactly one loop and stops (set ``loop:=true`` to
restart automatically).  Review the limits and the CSV before running on
real hardware.
"""

import csv
import json
import os
import sys
import threading
from typing import List, Optional, Dict, Any

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import JointState
from mc_core_interface.msg import MechUnitCmd, MechUnitCmdArray


# Right-arm joint names, in MechUnitCmd / CSV column order. Mirrors
# pingpong_controller.rl_policy.RIGHT_ARM_JOINTS.
RIGHT_ARM_JOINTS: List[str] = [
    "RightArm-0", "RightArm-1", "RightArm-2",
    "RightArm-3", "RightArm-4", "RightArm-5", "RightArm-6",
]
N_JOINTS = len(RIGHT_ARM_JOINTS)

# --- Sim action->joint mapping constants (authoritative source:
#     pingpong_controller/tools/rl_sim/mjx_juggle_env.py: MjxJuggleEnv.step
#     and validate_juggle_mjx_ppo.py: make_eval_step). The recorded trace was
#     produced by exactly this pipeline, so we reproduce it here. ---
#
# Per-joint acceleration limit (deg/s^2), MjxJuggleConfig.arm_acc_limit_deg_s2.
DEFAULT_ARM_ACC_LIMIT_DEG_S2: List[float] = [
    1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0,
]
# Per-joint velocity limit (deg/s), MjxJuggleConfig.arm_vel_limit_deg_s. Only
# used when arm_action_limiter is True (it is False in the recorded run).
DEFAULT_ARM_VEL_LIMIT_DEG_S: List[float] = [
    210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0,
]
# Scalar action->acc scale. In the env this is
#   action_acc_scale (config) * action_scale_mult (=1.0 without domain
#   randomization, as in validation). Recovered from the trace by
#   least-squares: 0.939124 (std ~4e-8), constant across all joints/steps.
# So per-joint scale (rad/s^2 per unit action) = deg2rad(acc_limit)*this.
DEFAULT_ACTION_ACC_SCALE: float = 0.939124

# Position clip = robot XML jnt_range for the right-arm joints (moz1_pd.xml).
# The env clips arm_cmd_q to these MuJoCo joint ranges every step.
DEFAULT_Q_LOW_RAD: List[float] = [
    -2.0944, -2.96706, -3.05433, -0.174533, -3.05433, -1.65806, -1.5708,
]
DEFAULT_Q_HIGH_RAD: List[float] = [
    3.14159, 0.15708, 3.05433, 2.25147, 3.05433, 1.65806, 1.5708,
]

VALID_MODES = ("action", "qdd", "qvel", "angle")

# CSV column prefixes per mode. Values are read as "<prefix>/<joint_name>".
MODE_COLUMN_PREFIX = {
    "action": "effective_action",   # in [-1, 1]
    "qdd": "desired_qdd_raw",        # rad/s^2
    "qvel": "arm_cmd_qvel",          # rad/s
    "angle": "arm_cmd_q",            # rad
}


class ReplayNode(Node):
    """Stream a recorded right-arm trajectory to /mx_mix_command."""

    def __init__(self) -> None:
        super().__init__('pingpong_replay_node')

        # --- Parameters ---
        self.declare_parameter('csv_path', '')
        self.declare_parameter('command_topic', '/mx_mix_command')
        self.declare_parameter('joint_state_topic', '/joint_states')
        # One of: action | qdd | qvel | angle
        self.declare_parameter('input_mode', 'angle')
        self.declare_parameter('right_arm_mu_idx', 0)

        # The trace was recorded at 200 Hz (dt = 0.005 s). control_rate_hz sets
        # the publish cadence; integration uses integration_dt.
        self.declare_parameter('control_rate_hz', 200.0)
        self.declare_parameter('integration_dt', 0.005)

        # Replay range / repetition.
        self.declare_parameter('start_index', 0)
        self.declare_parameter('end_index', -1)   # -1 -> last row
        self.declare_parameter('loop', False)

        # Data recording for analysis.
        self.declare_parameter('record_data', True)
        self.declare_parameter('output_dir', '')
        # Use velocity/acceleration from JointState feedback instead of numerical
        # differentiation. JointState.velocity is standard; JointState.effort is
        # repurposed as acceleration (rad/s²) in this system (see pingpong_node.py).
        self.declare_parameter('use_feedback_velocity', True)
        self.declare_parameter('use_feedback_acceleration', True)

        # Move-to-start ramp from the measured arm pose to trajectory[0].
        self.declare_parameter('move_to_start', True)
        self.declare_parameter('move_to_start_duration', 5.0)
        self.declare_parameter('joint_state_wait_timeout', 5.0)
        # After reaching the start pose, hold there and wait for the operator
        # to press Enter on stdin before replaying. Set false for headless
        # launches (CI, launch files without a TTY).
        self.declare_parameter('require_enter_confirm', True)

        # Per-joint conversion / clipping (mirror the sim env exactly).
        self.declare_parameter(
            'arm_acc_limit_deg_s2', DEFAULT_ARM_ACC_LIMIT_DEG_S2)
        self.declare_parameter(
            'arm_vel_limit_deg_s', DEFAULT_ARM_VEL_LIMIT_DEG_S)
        self.declare_parameter('action_acc_scale', DEFAULT_ACTION_ACC_SCALE)
        # arm_action_limiter was False in the recorded run; when True, qdd and
        # qvel are clipped to the accel/vel limits, matching the sim option.
        self.declare_parameter('arm_action_limiter', False)
        self.declare_parameter('q_low', DEFAULT_Q_LOW_RAD)
        self.declare_parameter('q_high', DEFAULT_Q_HIGH_RAD)
        self.declare_parameter('clip_to_limits', True)

        # --- Resolve parameters ---
        self.command_topic = self.get_parameter('command_topic').value
        self.joint_state_topic = self.get_parameter('joint_state_topic').value
        self.input_mode = str(self.get_parameter('input_mode').value).lower()
        if self.input_mode not in VALID_MODES:
            raise ValueError(
                f"input_mode must be one of {VALID_MODES}, "
                f"got '{self.input_mode}'")
        self.right_arm_mu_idx = int(
            self.get_parameter('right_arm_mu_idx').value)

        self.control_rate_hz = float(
            self.get_parameter('control_rate_hz').value)
        if not np.isfinite(self.control_rate_hz) or self.control_rate_hz <= 0:
            self.get_logger().warn(
                f'Invalid control_rate_hz={self.control_rate_hz}; '
                'using 200.0 Hz')
            self.control_rate_hz = 200.0
        self.control_dt = 1.0 / self.control_rate_hz
        self.integration_dt = float(
            self.get_parameter('integration_dt').value)

        self.loop = bool(self.get_parameter('loop').value)
        self.move_to_start = bool(self.get_parameter('move_to_start').value)
        self.move_to_start_duration = float(
            self.get_parameter('move_to_start_duration').value)
        self.joint_state_wait_timeout = float(
            self.get_parameter('joint_state_wait_timeout').value)
        self.require_enter_confirm = bool(
            self.get_parameter('require_enter_confirm').value)

        self.record_data = bool(self.get_parameter('record_data').value)
        self.use_feedback_velocity = bool(
            self.get_parameter('use_feedback_velocity').value)
        self.use_feedback_acceleration = bool(
            self.get_parameter('use_feedback_acceleration').value)

        # Action->joint mapping (mirrors mjx_juggle_env.MjxJuggleEnv.step):
        #   desired_qdd_raw = action * arm_acc_limit_rad_s2 * action_acc_scale
        # arm_acc_limit_rad_s2 and arm_vel_limit_rad_s come from deg configs.
        self.arm_acc_limit_rad_s2 = np.deg2rad(np.asarray(
            self.get_parameter('arm_acc_limit_deg_s2').value,
            dtype=np.float64))
        self.arm_vel_limit_rad_s = np.deg2rad(np.asarray(
            self.get_parameter('arm_vel_limit_deg_s').value,
            dtype=np.float64))
        self.action_acc_scale = float(
            self.get_parameter('action_acc_scale').value)
        self.arm_action_limiter = bool(
            self.get_parameter('arm_action_limiter').value)
        # Effective per-joint action scale (rad/s^2 per unit action).
        self.action_scale_per_joint = (
            self.arm_acc_limit_rad_s2 * self.action_acc_scale)
        self.q_low = np.asarray(
            self.get_parameter('q_low').value, dtype=np.float64)
        self.q_high = np.asarray(
            self.get_parameter('q_high').value, dtype=np.float64)
        self.clip_to_limits = bool(self.get_parameter('clip_to_limits').value)
        for name, arr in (('arm_acc_limit_deg_s2', self.arm_acc_limit_rad_s2),
                          ('arm_vel_limit_deg_s', self.arm_vel_limit_rad_s),
                          ('q_low', self.q_low), ('q_high', self.q_high)):
            if arr.shape != (N_JOINTS,):
                raise ValueError(
                    f"{name} must have length {N_JOINTS}, got {arr.shape}")

        csv_path = self._resolve_csv_path(
            str(self.get_parameter('csv_path').value))

        # Resolve output directory after csv_path is known
        self.output_dir = str(self.get_parameter('output_dir').value)
        if not self.output_dir:
            # Default to the same directory as the CSV
            self.output_dir = os.path.dirname(csv_path) if csv_path else os.getcwd()

        # --- Build the position trajectory for the chosen mode ---
        start_index = int(self.get_parameter('start_index').value)
        end_index = int(self.get_parameter('end_index').value)

        # Store the original CSV path for loading action data later
        self._csv_path = csv_path
        self._start_index = start_index
        self._end_index = end_index

        self.trajectory = self._build_trajectory(
            csv_path, self.input_mode, start_index, end_index)
        self.n_steps = self.trajectory.shape[0]
        if self.n_steps == 0:
            raise ValueError('Trajectory is empty after applying start/end '
                             'index; nothing to replay')

        # Load the action trajectory for theoretical calculations
        self.action_trajectory = None
        self.theoretical_qdd = None
        self.theoretical_qvel = None
        self.theoretical_q = None
        if self.record_data:
            self._load_theoretical_trajectory(csv_path, start_index, end_index)

        # --- Latest joint feedback (for move-to-start) ---
        self._latest_q: Optional[np.ndarray] = None
        self._latest_qvel: Optional[np.ndarray] = None
        self._latest_qdd: Optional[np.ndarray] = None
        self._joint_state_lock = threading.Lock()

        # --- Data recording ---
        self.recorded_data: List[Dict[str, Any]] = []
        self._recording_active = False

        # --- ROS interfaces ---
        # The real /mx_mix_command carries MechUnitCmdArray (header + a list
        # of per-unit MechUnitCmd), NOT a bare MechUnitCmd. We publish a single
        # right-arm command wrapped in the array.
        self.cmd_pub = self.create_publisher(
            MechUnitCmdArray, self.command_topic, 10)
        self.js_cb_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            JointState, self.joint_state_topic,
            self._joint_state_callback, 10,
            callback_group=self.js_cb_group)

        # --- Replay state machine ---
        # phase: 'init' -> 'move_to_start' -> 'wait_confirm' -> 'replay' -> 'done'
        self._ramp_traj: Optional[np.ndarray] = None
        self._ramp_idx = 0
        self._replay_idx = 0
        self._phase = 'init'
        self._js_wait_start = self.get_clock().now()
        self._confirm_event = threading.Event()
        self._confirm_thread: Optional[threading.Thread] = None
        self._confirm_prompted = False

        # Pose the arm must reach before replay starts: the first pose of the
        # (sliced) trajectory. With the default start_index=0 this is exactly
        # the CSV row-0 right-arm pose (arm_cmd_q at t=0).
        self.replay_start_pose = self.trajectory[0].copy()

        self.get_logger().info(
            f'pingpong_replay: mode="{self.input_mode}" '
            f'steps={self.n_steps} rate={self.control_rate_hz:.1f}Hz '
            f'mu_idx={self.right_arm_mu_idx} topic="{self.command_topic}" '
            f'loop={self.loop} move_to_start={self.move_to_start} '
            f'clip={self.clip_to_limits}')
        if self.input_mode in ('action', 'qdd'):
            self.get_logger().info(
                f'  mapping: action_acc_scale={self.action_acc_scale} '
                f'arm_action_limiter={self.arm_action_limiter} '
                f'integration_dt={self.integration_dt}')
        self.get_logger().info(
            f'  start pose (CSV t=0, deg) = '
            f'{np.round(np.degrees(self.replay_start_pose), 2).tolist()}')

        self.timer = self.create_timer(self.control_dt, self._on_timer)

    # ------------------------------------------------------------------
    # CSV loading / trajectory reconstruction
    # ------------------------------------------------------------------
    def _resolve_csv_path(self, csv_path: str) -> str:
        if csv_path:
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(f'csv_path not found: {csv_path}')
            return csv_path
        # Fallback to the known recorded trace, searched relative to a few
        # likely roots so the node runs both from source and an install tree.
        rel = ('pingpong_controller/outputs/rl_sim/'
               'stage4g_action_trace.csv')
        candidates = [
            os.path.join(os.getcwd(), 'src', 'pingpong_moz1', rel),
            os.path.join(os.getcwd(), rel),
        ]
        for c in candidates:
            if os.path.isfile(c):
                self.get_logger().info(f'Using default csv_path: {c}')
                return c
        raise FileNotFoundError(
            'csv_path not set and default stage4g_action_trace.csv not found. '
            'Pass -p csv_path:=/abs/path/to/stage4g_action_trace.csv')

    def _read_columns(self, csv_path: str, prefix: str) -> np.ndarray:
        """Read columns "<prefix>/<joint>" for every right-arm joint.

        Returns array of shape (n_rows, N_JOINTS).
        """
        cols = [f'{prefix}/{j}' for j in RIGHT_ARM_JOINTS]
        rows: List[List[float]] = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            missing = [c for c in cols if c not in reader.fieldnames]
            if missing:
                raise KeyError(
                    f'CSV is missing expected columns for mode: {missing}')
            for r in reader:
                rows.append([float(r[c]) for c in cols])
        data = np.asarray(rows, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] != N_JOINTS:
            raise ValueError(
                f'Expected {N_JOINTS} columns from prefix "{prefix}", '
                f'got shape {data.shape}')
        return data

    def _build_trajectory(self, csv_path: str, mode: str,
                          start_index: int, end_index: int) -> np.ndarray:
        """Reconstruct the right-arm position command sequence (rad).

        For action/qdd/qvel the recorded initial command state (arm_cmd_q,
        arm_cmd_qvel of the first row) seeds the integration so the
        reconstruction matches the logged arm_cmd_q. ``angle`` reads positions
        directly. Slicing by start/end_index happens after reconstruction so
        integrated modes always start from the true initial state.
        """
        # Initial command state is always available; used to seed integration
        # and as the natural trajectory origin.
        q0 = self._read_columns(csv_path, 'arm_cmd_q')[0].copy()
        qvel0 = self._read_columns(csv_path, 'arm_cmd_qvel')[0].copy()
        dt = self.integration_dt

        if mode == 'angle':
            traj = self._read_columns(csv_path, MODE_COLUMN_PREFIX['angle'])
            if self.clip_to_limits:
                traj = np.clip(traj, self.q_low, self.q_high)
        else:
            src = self._read_columns(csv_path, MODE_COLUMN_PREFIX[mode])
            n = src.shape[0]
            traj = np.empty((n, N_JOINTS), dtype=np.float64)
            # Row 0 is the command state *before* any action is applied, so
            # traj[0] is the recorded initial pose. For k >= 1, integrating
            # src[k] forward from row 0's state reproduces arm_cmd_q[k]
            # exactly (verified: max error <= 6.5e-6 rad over the full trace).
            #
            # This loop mirrors MjxJuggleEnv.step() one-to-one:
            #   desired_qdd_raw = action * arm_acc_limit_rad_s2 * action_acc_scale
            #   desired_qdd     = clip(.., +/-acc) if arm_action_limiter else raw
            #   cmd_qvel        = clip(qvel + qdd*dt, +/-vel) if limiter else raw
            #   arm_cmd_q       = clip(q + cmd_qvel*dt, q_low, q_high)
            # For qdd mode, src is desired_qdd_raw directly; for qvel mode, src
            # is the already-integrated arm_cmd_qvel so we assign rather than
            # accumulate.
            traj[0] = q0
            q = q0.copy()
            qvel = qvel0.copy()
            for k in range(1, n):
                if mode == 'action':
                    qdd = src[k] * self.action_scale_per_joint
                    if self.arm_action_limiter:
                        qdd = np.clip(qdd, -self.arm_acc_limit_rad_s2,
                                      self.arm_acc_limit_rad_s2)
                    qvel = qvel + qdd * dt
                    if self.arm_action_limiter:
                        qvel = np.clip(qvel, -self.arm_vel_limit_rad_s,
                                       self.arm_vel_limit_rad_s)
                    q = q + qvel * dt
                elif mode == 'qdd':
                    qdd = src[k]
                    if self.arm_action_limiter:
                        qdd = np.clip(qdd, -self.arm_acc_limit_rad_s2,
                                      self.arm_acc_limit_rad_s2)
                    qvel = qvel + qdd * dt
                    if self.arm_action_limiter:
                        qvel = np.clip(qvel, -self.arm_vel_limit_rad_s,
                                       self.arm_vel_limit_rad_s)
                    q = q + qvel * dt
                else:  # qvel: arm_cmd_qvel[k] is the post-integration velocity
                    qvel = src[k]
                    q = q + qvel * dt
                if self.clip_to_limits:
                    q = np.clip(q, self.q_low, self.q_high)
                traj[k] = q

        # Apply start/end slicing.
        n = traj.shape[0]
        s = max(0, start_index)
        e = n if end_index < 0 else min(end_index, n)
        return traj[s:e].copy()

    def _load_theoretical_trajectory(self, csv_path: str,
                                     start_index: int, end_index: int) -> None:
        """Load action trajectory and compute theoretical qdd, qvel, q."""
        # Load action data
        action_data = self._read_columns(csv_path, 'effective_action')

        # Load initial states
        q0 = self._read_columns(csv_path, 'arm_cmd_q')[0].copy()
        qvel0 = self._read_columns(csv_path, 'arm_cmd_qvel')[0].copy()

        n = action_data.shape[0]
        dt = self.integration_dt

        # Initialize arrays for theoretical values
        self.theoretical_qdd = np.empty((n, N_JOINTS), dtype=np.float64)
        self.theoretical_qvel = np.empty((n, N_JOINTS), dtype=np.float64)
        self.theoretical_q = np.empty((n, N_JOINTS), dtype=np.float64)

        # Row 0 initial state
        self.theoretical_q[0] = q0
        self.theoretical_qvel[0] = qvel0
        self.theoretical_qdd[0] = np.zeros(N_JOINTS)  # No action at t=0

        q = q0.copy()
        qvel = qvel0.copy()

        # Integrate action to get qdd, qvel, q
        for k in range(1, n):
            # Convert action to acceleration
            qdd = action_data[k] * self.action_scale_per_joint
            if self.arm_action_limiter:
                qdd = np.clip(qdd, -self.arm_acc_limit_rad_s2,
                             self.arm_acc_limit_rad_s2)

            # Integrate to velocity
            qvel = qvel + qdd * dt
            if self.arm_action_limiter:
                qvel = np.clip(qvel, -self.arm_vel_limit_rad_s,
                              self.arm_vel_limit_rad_s)

            # Integrate to position
            q = q + qvel * dt
            if self.clip_to_limits:
                q = np.clip(q, self.q_low, self.q_high)

            self.theoretical_qdd[k] = qdd
            self.theoretical_qvel[k] = qvel
            self.theoretical_q[k] = q

        # Apply slicing
        s = max(0, start_index)
        e = n if end_index < 0 else min(end_index, n)
        self.action_trajectory = action_data[s:e].copy()
        self.theoretical_qdd = self.theoretical_qdd[s:e].copy()
        self.theoretical_qvel = self.theoretical_qvel[s:e].copy()
        self.theoretical_q = self.theoretical_q[s:e].copy()

        self.get_logger().info(
            f'Loaded theoretical trajectory for comparison: {self.theoretical_q.shape[0]} steps')

    # ------------------------------------------------------------------
    # Joint feedback
    # ------------------------------------------------------------------
    def _joint_state_callback(self, msg: JointState) -> None:
        name_to_pos = {}
        name_to_vel = {}
        name_to_acc = {}

        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                v = float(msg.position[i])
                if np.isfinite(v):
                    name_to_pos[name] = v
            if i < len(msg.velocity):
                v = float(msg.velocity[i])
                if np.isfinite(v):
                    name_to_vel[name] = v
            # In this system, JointState.effort carries acceleration (rad/s²)
            if i < len(msg.effort):
                v = float(msg.effort[i])
                if np.isfinite(v):
                    name_to_acc[name] = v

        try:
            q = np.array([name_to_pos[n] for n in RIGHT_ARM_JOINTS],
                         dtype=np.float64)
        except KeyError:
            return  # frame did not contain all right-arm joints

        # Velocity and acceleration are optional
        qvel = None
        if len(name_to_vel) >= len(RIGHT_ARM_JOINTS):
            try:
                qvel = np.array([name_to_vel[n] for n in RIGHT_ARM_JOINTS],
                               dtype=np.float64)
            except KeyError:
                pass

        qdd = None
        if len(name_to_acc) >= len(RIGHT_ARM_JOINTS):
            try:
                qdd = np.array([name_to_acc[n] for n in RIGHT_ARM_JOINTS],
                              dtype=np.float64)
            except KeyError:
                pass

        with self._joint_state_lock:
            self._latest_q = q
            self._latest_qvel = qvel
            self._latest_qdd = qdd

    def _get_latest_q(self) -> Optional[np.ndarray]:
        with self._joint_state_lock:
            return None if self._latest_q is None else self._latest_q.copy()

    def _get_latest_state(self) -> tuple[Optional[np.ndarray],
                                         Optional[np.ndarray],
                                         Optional[np.ndarray]]:
        """Get latest position, velocity, and acceleration."""
        with self._joint_state_lock:
            q = None if self._latest_q is None else self._latest_q.copy()
            qvel = None if self._latest_qvel is None else self._latest_qvel.copy()
            qdd = None if self._latest_qdd is None else self._latest_qdd.copy()
            return q, qvel, qdd

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish_q(self, q: np.ndarray) -> None:
        if self.clip_to_limits:
            q = np.clip(q, self.q_low, self.q_high)
        cmd = MechUnitCmd()
        cmd.mu_idx = self.right_arm_mu_idx
        cmd.jnt_pos = [float(x) for x in q]
        cmd.use_jnt = True
        cmd.psi = 0.0
        # cmd.end_pose left at the default identity Pose (joint control only).
        msg = MechUnitCmdArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.cmds = [cmd]
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Control timer / state machine
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        if self._phase == 'init':
            self._enter_after_init()
            return
        if self._phase == 'move_to_start':
            self._step_move_to_start()
            return
        if self._phase == 'wait_confirm':
            self._step_wait_confirm()
            return
        if self._phase == 'replay':
            self._step_replay()
            return
        # 'done' -> nothing.

    def _enter_after_init(self) -> None:
        target0 = self.replay_start_pose
        if not self.move_to_start:
            self._enter_wait_confirm()
            self.get_logger().info('move_to_start disabled')
            return

        cur = self._get_latest_q()
        if cur is None:
            elapsed = (self.get_clock().now()
                       - self._js_wait_start).nanoseconds * 1e-9
            if elapsed > self.joint_state_wait_timeout:
                self.get_logger().error(
                    f'No /joint_states on "{self.joint_state_topic}" after '
                    f'{self.joint_state_wait_timeout:.1f}s. Refusing to jump '
                    'to the trajectory start without knowing the current arm '
                    'pose. Aborting (set move_to_start:=false to override).')
                self._phase = 'done'
                self.timer.cancel()
            return  # keep waiting for feedback

        n_ramp = max(1, int(round(
            self.move_to_start_duration / self.control_dt)))
        # Linear interpolation cur -> target0 over n_ramp ticks.
        alphas = np.linspace(1.0 / n_ramp, 1.0, n_ramp)[:, None]
        self._ramp_traj = (1.0 - alphas) * cur[None, :] + alphas * target0[None, :]
        self._ramp_idx = 0
        self._phase = 'move_to_start'
        self.get_logger().info(
            f'Moving to trajectory start over {self.move_to_start_duration:.1f}s '
            f'({n_ramp} steps). cur(deg)={np.round(np.degrees(cur),1).tolist()} '
            f'-> start(deg)={np.round(np.degrees(target0),1).tolist()}')

    def _step_move_to_start(self) -> None:
        self._publish_q(self._ramp_traj[self._ramp_idx])
        self._ramp_idx += 1
        if self._ramp_idx >= self._ramp_traj.shape[0]:
            self.get_logger().info('Reached start pose; holding')
            self._enter_wait_confirm()

    def _enter_wait_confirm(self) -> None:
        """Transition to the hold-and-wait-for-Enter phase."""
        self._phase = 'wait_confirm'
        if not self.require_enter_confirm:
            self.get_logger().info(
                'require_enter_confirm=false; beginning replay immediately')
            self._begin_replay()
            return
        if not sys.stdin.isatty():
            self.get_logger().warn(
                'require_enter_confirm=true but stdin is not a TTY; '
                'cannot read Enter. Beginning replay immediately. '
                'Set require_enter_confirm:=false to silence this.')
            self._begin_replay()
            return
        # Spawn a daemon thread that blocks on stdin and signals on Enter.
        if self._confirm_thread is None:
            self._confirm_thread = threading.Thread(
                target=self._stdin_confirm_worker, daemon=True)
            self._confirm_thread.start()

    def _stdin_confirm_worker(self) -> None:
        try:
            sys.stdin.readline()
        except Exception:  # noqa: BLE001 - stdin closed / interrupted
            pass
        self._confirm_event.set()

    def _step_wait_confirm(self) -> None:
        # Hold the start pose every tick so the controller keeps a valid
        # setpoint while we wait for the operator.
        self._publish_q(self.replay_start_pose)
        if not self._confirm_prompted:
            self.get_logger().info(
                'Holding at trajectory start. Press <Enter> to begin replay.')
            self._confirm_prompted = True
        if self._confirm_event.is_set():
            self._begin_replay()

    def _begin_replay(self) -> None:
        self._phase = 'replay'
        self._replay_idx = 0
        if self.record_data:
            self._recording_active = True
            self.recorded_data = []
            self.get_logger().info('Data recording activated')
        self.get_logger().info(
            f'Beginning replay ({self.n_steps} steps, single loop)')

    def _step_replay(self) -> None:
        idx = self._replay_idx
        cmd_q = self.trajectory[idx]

        # Record data if recording is active
        if self._recording_active:
            actual_q, actual_qvel, actual_qdd = self._get_latest_state()
            if actual_q is not None:
                record = {
                    'step': idx,
                    'timestamp': self.get_clock().now().nanoseconds * 1e-9,
                    'cmd_q': cmd_q.copy(),
                    'actual_q': actual_q.copy(),
                }
                # Store feedback velocity/acceleration if available
                if actual_qvel is not None:
                    record['actual_qvel_feedback'] = actual_qvel.copy()
                if actual_qdd is not None:
                    record['actual_qdd_feedback'] = actual_qdd.copy()

                # Add theoretical values if available
                if self.theoretical_q is not None and idx < len(self.theoretical_q):
                    record['theoretical_q'] = self.theoretical_q[idx].copy()
                    record['theoretical_qvel'] = self.theoretical_qvel[idx].copy()
                    record['theoretical_qdd'] = self.theoretical_qdd[idx].copy()
                    if self.action_trajectory is not None:
                        record['action'] = self.action_trajectory[idx].copy()
                self.recorded_data.append(record)

        self._publish_q(cmd_q)
        self._replay_idx += 1
        if self._replay_idx >= self.n_steps:
            if self.loop:
                self._replay_idx = 0
                self.get_logger().info('Replay loop: restarting trajectory')
            else:
                self._phase = 'done'
                self.timer.cancel()
                if self._recording_active:
                    self._save_recorded_data()
                self.get_logger().info(
                    f'Replay complete ({self.n_steps} steps). Holding last '
                    'command. Ctrl-C to exit.')

    # ------------------------------------------------------------------
    # Data recording and analysis
    # ------------------------------------------------------------------
    def _save_recorded_data(self) -> None:
        """Save recorded data and compute actual velocities and accelerations."""
        if not self.recorded_data:
            self.get_logger().warn('No data recorded')
            return

        self.get_logger().info(f'Processing {len(self.recorded_data)} recorded samples...')

        # Convert to numpy arrays for easier processing
        n = len(self.recorded_data)
        times = np.array([r['timestamp'] for r in self.recorded_data])
        actual_q = np.array([r['actual_q'] for r in self.recorded_data])
        cmd_q = np.array([r['cmd_q'] for r in self.recorded_data])

        # Check if we have feedback velocity and acceleration
        has_feedback_qvel = 'actual_qvel_feedback' in self.recorded_data[0]
        has_feedback_qdd = 'actual_qdd_feedback' in self.recorded_data[0]

        # Compute or use actual velocities
        actual_qvel = np.zeros_like(actual_q)
        if self.use_feedback_velocity and has_feedback_qvel:
            actual_qvel = np.array([r['actual_qvel_feedback'] for r in self.recorded_data])
            self.get_logger().info('Using velocity from JointState feedback')
        else:
            # Compute via numerical differentiation
            for i in range(1, n):
                dt = times[i] - times[i-1]
                if dt > 0:
                    actual_qvel[i] = (actual_q[i] - actual_q[i-1]) / dt
            self.get_logger().info('Computed velocity via numerical differentiation')

        # Compute or use actual accelerations
        actual_qdd = np.zeros_like(actual_q)
        if self.use_feedback_acceleration and has_feedback_qdd:
            actual_qdd = np.array([r['actual_qdd_feedback'] for r in self.recorded_data])
            self.get_logger().info('Using acceleration from JointState feedback (effort field)')
        else:
            # Compute via numerical differentiation of velocity
            for i in range(1, n):
                dt = times[i] - times[i-1]
                if dt > 0:
                    actual_qdd[i] = (actual_qvel[i] - actual_qvel[i-1]) / dt
            self.get_logger().info('Computed acceleration via numerical differentiation')

        # Prepare data for CSV export
        csv_rows = []
        for i in range(n):
            row = {
                'step': self.recorded_data[i]['step'],
                'timestamp': times[i],
            }
            # Add actual values
            for j, joint in enumerate(RIGHT_ARM_JOINTS):
                row[f'actual_q/{joint}'] = actual_q[i, j]
                row[f'actual_qvel/{joint}'] = actual_qvel[i, j]
                row[f'actual_qdd/{joint}'] = actual_qdd[i, j]
                row[f'cmd_q/{joint}'] = cmd_q[i, j]

            # Add theoretical values if available
            if 'theoretical_q' in self.recorded_data[i]:
                for j, joint in enumerate(RIGHT_ARM_JOINTS):
                    row[f'theoretical_q/{joint}'] = self.recorded_data[i]['theoretical_q'][j]
                    row[f'theoretical_qvel/{joint}'] = self.recorded_data[i]['theoretical_qvel'][j]
                    row[f'theoretical_qdd/{joint}'] = self.recorded_data[i]['theoretical_qdd'][j]
                if 'action' in self.recorded_data[i]:
                    for j, joint in enumerate(RIGHT_ARM_JOINTS):
                        row[f'action/{joint}'] = self.recorded_data[i]['action'][j]

            csv_rows.append(row)

        # Save to CSV
        timestamp_str = self.get_clock().now().to_msg()
        timestamp_sec = timestamp_str.sec
        output_csv = os.path.join(
            self.output_dir,
            f'replay_analysis_{timestamp_sec}.csv'
        )

        os.makedirs(self.output_dir, exist_ok=True)

        with open(output_csv, 'w', newline='') as f:
            if csv_rows:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

        self.get_logger().info(f'Saved analysis data to: {output_csv}')

        # Also save a summary JSON
        summary = {
            'n_samples': n,
            'duration': float(times[-1] - times[0]),
            'control_rate_hz': self.control_rate_hz,
            'input_mode': self.input_mode,
            'joints': RIGHT_ARM_JOINTS,
            'output_csv': output_csv,
            'use_feedback_velocity': self.use_feedback_velocity and has_feedback_qvel,
            'use_feedback_acceleration': self.use_feedback_acceleration and has_feedback_qdd,
        }

        # Compute tracking errors
        if 'theoretical_q' in self.recorded_data[0]:
            theoretical_q = np.array([r['theoretical_q'] for r in self.recorded_data])
            theoretical_qvel = np.array([r['theoretical_qvel'] for r in self.recorded_data])
            theoretical_qdd = np.array([r['theoretical_qdd'] for r in self.recorded_data])

            q_error = actual_q - theoretical_q
            qvel_error = actual_qvel - theoretical_qvel
            qdd_error = actual_qdd - theoretical_qdd

            summary['tracking_errors'] = {
                'position': {
                    'mean_abs_rad': float(np.mean(np.abs(q_error))),
                    'max_abs_rad': float(np.max(np.abs(q_error))),
                    'rms_rad': float(np.sqrt(np.mean(q_error**2))),
                    'mean_abs_deg': float(np.degrees(np.mean(np.abs(q_error)))),
                    'max_abs_deg': float(np.degrees(np.max(np.abs(q_error)))),
                },
                'velocity': {
                    'mean_abs_rad_s': float(np.mean(np.abs(qvel_error))),
                    'max_abs_rad_s': float(np.max(np.abs(qvel_error))),
                    'rms_rad_s': float(np.sqrt(np.mean(qvel_error**2))),
                },
                'acceleration': {
                    'mean_abs_rad_s2': float(np.mean(np.abs(qdd_error))),
                    'max_abs_rad_s2': float(np.max(np.abs(qdd_error))),
                    'rms_rad_s2': float(np.sqrt(np.mean(qdd_error**2))),
                }
            }

        output_json = os.path.join(
            self.output_dir,
            f'replay_analysis_{timestamp_sec}.json'
        )
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)

        self.get_logger().info(f'Saved summary to: {output_json}')
        if 'tracking_errors' in summary:
            self.get_logger().info(
                f"Position tracking error: "
                f"mean={summary['tracking_errors']['position']['mean_abs_deg']:.3f}° "
                f"max={summary['tracking_errors']['position']['max_abs_deg']:.3f}°"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ReplayNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
