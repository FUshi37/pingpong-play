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

Actuator-delay compensation
----------------------------
The real joints track a position command with a transport delay plus a
first-order lag, so the arm lands behind the commanded trajectory. The sim
policy was trained with an actuator compensator (``compensate_q_ref`` in
``delay_control.py``) that pre-distorts each command to cancel exactly this
delay+lag. This node can run the same compensator online: set
``compensation_mode`` to ``inverse_mpc`` (recommended), ``inverse_smith`` or
``lead`` and it pre-distorts every published command using the recorded nominal
``(q, qvel, qdd)`` and the live ``/joint_states`` feedback. The defaults match
the shipped ``inverse_mpc`` checkpoint's training config. The default
``compensation_mode='none'`` publishes the recorded command unchanged (legacy
behaviour). See ``compensation.py`` for the full parameter list.

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

Cartesian output mode
---------------------
By default the node publishes a joint command (``MechUnitCmd.jnt_pos`` with
``use_jnt=True``). Set ``output_mode:=cartesian`` to instead convert every
joint command to an end-effector pose and publish it as
``MechUnitCmd.end_pose`` (``use_jnt=False``) plus the redundant arm angle
``psi`` -- the same Cartesian command format the teleop driver uses. The pose
is computed by a dependency-free forward kinematics over the right-arm chain
parsed from ``robot_xml_path`` (``moz1_pd.xml``); ``mujoco`` is not required.

Because the FK is rooted at the arm-mount body, the controller's reported pose
relates to ours by ``T_report = B @ fk(q) @ C``: a base-frame offset ``B``
(controller base relative to our root) and a tool offset ``C`` (the
controller's TCP is not our model site). At startup (``cart_calibrate=true``,
default) the move-to-start ramp is published *in joint space* while the node
collects ``(measured q, reported pose, reported psi)`` samples from
``/joint_states`` + ``/cart_states``; when the ramp completes it fits ``B`` and
``C`` (least squares) and the ``psi`` convention (geometric swivel angle, with
the reference direction/sign/offset chosen to match the reported psi), then
switches to cartesian publishing for the hold and replay. A single ramp sweep
(~30 deg of orientation change) fits the pose model to a few mm. ``psi`` is
handled per ``psi_mode``: ``computed`` (fitted swivel angle; default), ``zero``
(let the controller choose), or ``passthrough`` (echo the latest reported psi).
Set ``cart_calibrate:=false`` to publish raw FK poses (``B=C=identity``; only
correct if the controller expects the arm-mount frame and site TCP). The
recorded analysis CSV gains ``cart_cmd_*`` / ``cart_actual_*`` columns so
task-space tracking (and the psi convention) can be validated.

Note on redundancy: streaming a joint-recorded trajectory through cartesian
mode is only faithful if ``psi`` is right -- the controller re-solves IK for
the 7-DOF arm and picks the elbow from ``psi``. If the fitted psi rmse is high
(logged at calibration), the reproduced posture will differ; prefer joint
output mode when exact posture matters.
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
from mc_core_interface.msg import (
    MechUnitCmd, MechUnitCmdArray, MechUnitStateArray)

from pingpong_replay.compensation import (
    CommandCompensator, VALID_COMPENSATION_MODES)
from pingpong_replay.cartesian_fk import RightArmFK

try:
    from scipy.spatial.transform import Rotation as _ScipyRotation
    from scipy.optimize import least_squares as _scipy_least_squares
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


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

# Output representation published on /mx_mix_command. 'joint' is the legacy
# behaviour (MechUnitCmd.jnt_pos, use_jnt=True). 'cartesian' converts each
# joint command to an end-effector pose via FK and publishes it as
# MechUnitCmd.end_pose (use_jnt=False) plus the redundant arm angle psi.
VALID_OUTPUT_MODES = ("joint", "cartesian")

# How the redundant arm angle (psi) is filled in cartesian mode.
#   computed     -- FK swivel-angle estimate, offset to match /cart_states.
#   zero         -- always 0.0 (let the controller pick the arm posture).
#   passthrough  -- echo the latest controller-reported psi from /cart_states.
VALID_PSI_MODES = ("computed", "zero", "passthrough")

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

        # --- Cartesian output mode -------------------------------------
        # output_mode='joint' (default) publishes the joint command unchanged
        # (legacy). output_mode='cartesian' converts each joint command to an
        # end-effector pose via FK (parsed from robot_xml_path) and publishes
        # it as MechUnitCmd.end_pose with use_jnt=False, plus the redundant
        # arm angle psi. The pose is round-tripped against the controller's
        # own /cart_states at startup (cart_calibrate=True), so it lands in the
        # exact frame/TCP the controller expects regardless of how this FK's
        # root frame is defined.
        self.declare_parameter('output_mode', 'joint')   # joint | cartesian
        default_robot_xml = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'moz1_pd.xml')
        self.declare_parameter('robot_xml_path', default_robot_xml)
        self.declare_parameter('cart_state_topic', '/cart_states')
        self.declare_parameter('psi_mode', 'computed')  # computed|zero|passthrough
        self.declare_parameter('cart_calibrate', True)
        self.declare_parameter('cart_state_wait_timeout', 5.0)
        # Minimum number of (q, reported-pose) samples to collect over the
        # move-to-start ramp before fitting the B/C/psi calibration. The ramp
        # must provide enough orientation diversity; ~60 samples (a ~18deg
        # sweep) already fits the pose model to a few mm.
        self.declare_parameter('cart_cal_min_samples', 60)

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

        # --- Actuator-delay compensation (mirrors the sim policy's
        #     compensate_q_ref path). Defaults are the values the shipped
        #     inverse_mpc checkpoint was trained with (extracted from its
        #     env_cfg). compensation_mode='none' (default) keeps the legacy
        #     behaviour of publishing the recorded command unchanged. ---
        self.declare_parameter('compensation_mode', 'none')  # none|lead|inverse_smith|inverse_mpc
        self.declare_parameter('comp_delay_ms', 72.0)
        self.declare_parameter('comp_actuator_tau_s', 0.074)
        self.declare_parameter('comp_actuator_gain', 1.0)
        self.declare_parameter('comp_command_buffer_extra_steps', 4)
        self.declare_parameter('comp_delay_conditioning', True)
        # lead
        self.declare_parameter('comp_lead_beta', 0.0)
        self.declare_parameter('comp_lead_delay_scale', 1.0)
        self.declare_parameter('comp_lead_tau_scale', 1.0)
        self.declare_parameter('comp_lead_max_delta_rad', 0.0)
        # inverse_smith
        self.declare_parameter('comp_inverse_beta', 1.0)
        self.declare_parameter('comp_inverse_delay_scale', 1.0)
        self.declare_parameter('comp_inverse_tau_scale', 1.0)
        self.declare_parameter('comp_inverse_max_delta_rad', 0.0)
        # inverse_mpc
        self.declare_parameter('comp_mpc_beta', 1.2)
        self.declare_parameter('comp_mpc_delay_scale', 1.05)
        self.declare_parameter('comp_mpc_tau_scale', 0.75)
        self.declare_parameter('comp_mpc_horizon_steps', 6)
        self.declare_parameter('comp_mpc_tracking_weight', 1.0)
        self.declare_parameter('comp_mpc_nominal_weight', 0.25)
        self.declare_parameter('comp_mpc_delta_weight', 0.05)
        self.declare_parameter('comp_mpc_max_delta_rad', 0.5235987755982988)  # 30 deg

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

        # --- Resolve cartesian output mode ---
        self.output_mode = str(
            self.get_parameter('output_mode').value).strip().lower()
        if self.output_mode in ('', 'jnt'):
            self.output_mode = 'joint'
        if self.output_mode not in VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {VALID_OUTPUT_MODES}, "
                f"got '{self.output_mode}'")
        self.cart_state_topic = self.get_parameter('cart_state_topic').value
        self.robot_xml_path = str(self.get_parameter('robot_xml_path').value)
        self.psi_mode = str(
            self.get_parameter('psi_mode').value).strip().lower()
        if self.psi_mode not in VALID_PSI_MODES:
            raise ValueError(
                f"psi_mode must be one of {VALID_PSI_MODES}, "
                f"got '{self.psi_mode}'")
        self.cart_calibrate = bool(self.get_parameter('cart_calibrate').value)
        self.cart_state_wait_timeout = float(
            self.get_parameter('cart_state_wait_timeout').value)
        self.cart_cal_min_samples = max(
            12, int(self.get_parameter('cart_cal_min_samples').value))

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

        # --- Resolve actuator-delay compensation parameters ---
        self.compensation_mode = str(
            self.get_parameter('compensation_mode').value).strip().lower()
        if self.compensation_mode in ('', 'off', 'false', '0'):
            self.compensation_mode = 'none'
        if self.compensation_mode not in VALID_COMPENSATION_MODES:
            raise ValueError(
                f"compensation_mode must be one of {VALID_COMPENSATION_MODES}, "
                f"got '{self.compensation_mode}'")
        # Build the compensator. Position clip uses the same q_low/q_high as the
        # publisher so the pre-distorted command never exceeds joint limits.
        self.compensator = CommandCompensator(
            n_joints=N_JOINTS,
            dt=self.integration_dt,
            mode=self.compensation_mode,
            delay_ms=float(self.get_parameter('comp_delay_ms').value),
            actuator_tau_s=float(self.get_parameter('comp_actuator_tau_s').value),
            actuator_gain=float(self.get_parameter('comp_actuator_gain').value),
            command_buffer_extra_steps=int(
                self.get_parameter('comp_command_buffer_extra_steps').value),
            delay_conditioning=bool(
                self.get_parameter('comp_delay_conditioning').value),
            lead_beta=float(self.get_parameter('comp_lead_beta').value),
            lead_delay_scale=float(self.get_parameter('comp_lead_delay_scale').value),
            lead_tau_scale=float(self.get_parameter('comp_lead_tau_scale').value),
            lead_max_delta_rad=float(
                self.get_parameter('comp_lead_max_delta_rad').value),
            inverse_beta=float(self.get_parameter('comp_inverse_beta').value),
            inverse_delay_scale=float(
                self.get_parameter('comp_inverse_delay_scale').value),
            inverse_tau_scale=float(
                self.get_parameter('comp_inverse_tau_scale').value),
            inverse_max_delta_rad=float(
                self.get_parameter('comp_inverse_max_delta_rad').value),
            mpc_beta=float(self.get_parameter('comp_mpc_beta').value),
            mpc_delay_scale=float(self.get_parameter('comp_mpc_delay_scale').value),
            mpc_tau_scale=float(self.get_parameter('comp_mpc_tau_scale').value),
            mpc_horizon_steps=int(self.get_parameter('comp_mpc_horizon_steps').value),
            mpc_tracking_weight=float(
                self.get_parameter('comp_mpc_tracking_weight').value),
            mpc_nominal_weight=float(
                self.get_parameter('comp_mpc_nominal_weight').value),
            mpc_delta_weight=float(self.get_parameter('comp_mpc_delta_weight').value),
            mpc_max_delta_rad=float(self.get_parameter('comp_mpc_max_delta_rad').value),
            q_low=self.q_low if self.clip_to_limits else None,
            q_high=self.q_high if self.clip_to_limits else None,
        )

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

        # Nominal command velocity/acceleration aligned with self.trajectory.
        # The compensator needs (q_cmd, qdot_cmd, qdd_cmd) every tick; these are
        # the recorded arm_cmd_qvel and desired_qdd columns (same sliced range).
        # Only loaded when compensation is active so a plain replay is unaffected.
        self.cmd_qvel_traj: Optional[np.ndarray] = None
        self.cmd_qdd_traj: Optional[np.ndarray] = None
        if self.compensator.enabled:
            self._load_nominal_command_derivatives(
                csv_path, start_index, end_index)

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

        # --- Cartesian output mode: FK + calibration state ---
        # _fk converts a joint command to an end-effector pose. The controller's
        # reported pose relates to ours by T_report = B @ fk(q) @ C, where B is a
        # base-frame offset and C a tool/TCP offset (the controller's TCP is not
        # our model site). Both are fitted from samples collected while the arm
        # sweeps the move-to-start ramp (published in joint space), together with
        # the redundant arm angle psi (a geometric swivel angle whose reference
        # direction and constant bias are chosen to match the reported psi).
        self.fk: Optional[RightArmFK] = None
        self._cart_B = np.eye(4)             # base offset (controller <- our root)
        self._cart_C = np.eye(4)             # tool offset (our site -> controller TCP)
        self._cart_psi_ref = np.array([-1.0, 0.0, 0.0])  # arm-angle reference dir
        self._cart_psi_sign = 1.0
        self._cart_psi_offset = 0.0
        self._cart_calibrated = False        # True once B/C/psi are usable
        self._cart_cal_samples: List[Dict[str, np.ndarray]] = []
        self._cart_fit_pos_rms = float('nan')
        self._cart_fit_pos_max = float('nan')
        self._latest_cart_pose: Optional[np.ndarray] = None  # 4x4, reported
        self._latest_cart_psi: Optional[float] = None
        self._cart_state_lock = threading.Lock()
        self._cart_wait_start = self.get_clock().now()
        if self.output_mode == 'cartesian':
            self._init_cartesian()

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
        # Cartesian feedback (controller-reported end-effector pose + psi).
        # Needed in cartesian mode for the startup calibration / psi passthrough,
        # and in *any* mode when recording, so the actual EE pose is logged for
        # task-space comparison between joint and cartesian replays.
        if self.output_mode == 'cartesian' or self.record_data:
            self.create_subscription(
                MechUnitStateArray, self.cart_state_topic,
                self._cart_state_callback, 10,
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
        self.get_logger().info(f'  {self.compensator.describe()}')

        if self.output_mode == 'cartesian':
            self.get_logger().info(
                f'  output_mode=cartesian psi_mode={self.psi_mode} '
                f'calibrate={self.cart_calibrate} '
                f'xml={os.path.basename(self.robot_xml_path)} '
                f'cart_state_topic="{self.cart_state_topic}"')

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

    def _load_nominal_command_derivatives(self, csv_path: str,
                                          start_index: int,
                                          end_index: int) -> None:
        """Load nominal command velocity/acceleration for the compensator.

        The compensator pre-distorts each step from the nominal command state
        ``(q_cmd, qdot_cmd, qdd_cmd)``. ``q_cmd`` is ``self.trajectory`` (already
        built/sliced); here we load the matching ``arm_cmd_qvel`` (qdot) and
        ``desired_qdd`` (qdd) columns and slice them identically. ``desired_qdd``
        falls back to ``desired_qdd_raw`` if the post-limiter column is absent.
        The arrays line up row-for-row with ``self.trajectory``.
        """
        qvel = self._read_columns(csv_path, 'arm_cmd_qvel')
        try:
            qdd = self._read_columns(csv_path, 'desired_qdd')
        except KeyError:
            qdd = self._read_columns(csv_path, 'desired_qdd_raw')

        n = qvel.shape[0]
        s = max(0, start_index)
        e = n if end_index < 0 else min(end_index, n)
        self.cmd_qvel_traj = qvel[s:e].copy()
        self.cmd_qdd_traj = qdd[s:e].copy()

        if (self.cmd_qvel_traj.shape[0] != self.n_steps
                or self.cmd_qdd_traj.shape[0] != self.n_steps):
            raise ValueError(
                'nominal qvel/qdd length '
                f'({self.cmd_qvel_traj.shape[0]}/{self.cmd_qdd_traj.shape[0]}) '
                f'does not match trajectory length ({self.n_steps})')
        self.get_logger().info(
            f'Loaded nominal command derivatives for compensation: '
            f'{self.cmd_qvel_traj.shape[0]} steps')

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
    # Cartesian output mode: FK, calibration, publishing
    # ------------------------------------------------------------------
    def _init_cartesian(self) -> None:
        """Load the FK model for cartesian output mode (fail fast if broken)."""
        if not _SCIPY_AVAILABLE:
            raise RuntimeError(
                'output_mode=cartesian requires scipy '
                '(scipy.spatial.transform.Rotation + scipy.optimize) for the '
                'pose/psi calibration, but scipy is not importable')
        if self.cart_calibrate and not self.move_to_start:
            raise ValueError(
                'output_mode=cartesian with cart_calibrate=true needs the '
                'move_to_start ramp to collect calibration samples, but '
                'move_to_start=false. Enable the ramp, or set '
                'cart_calibrate:=false to publish raw FK poses (uncalibrated).')
        if not os.path.isfile(self.robot_xml_path):
            resolved = self._resolve_robot_xml(self.robot_xml_path)
            if resolved is None:
                raise FileNotFoundError(
                    f'output_mode=cartesian needs robot_xml_path; '
                    f'"{self.robot_xml_path}" not found and no moz1_pd.xml '
                    'found in the known controller model locations. Pass '
                    '-p robot_xml_path:=/abs/path/to/moz1_pd.xml')
            self.robot_xml_path = resolved
        self.fk = RightArmFK(self.robot_xml_path, RIGHT_ARM_JOINTS)
        self.get_logger().info(
            f'Cartesian FK loaded from {self.robot_xml_path} '
            f'(ee site="{self.fk.EE_SITE}")')

    def _resolve_robot_xml(self, requested: str) -> Optional[str]:
        """Find moz1_pd.xml in the known controller model locations."""
        rel = 'pingpong_controller/models/moz1_pd.xml'
        candidates = [
            os.path.join(os.getcwd(), 'src', 'pingpong_moz1', rel),
            os.path.join(os.getcwd(), rel),
        ]
        for c in candidates:
            if os.path.isfile(c):
                self.get_logger().info(f'Using default robot_xml_path: {c}')
                return c
        return None

    @staticmethod
    def _pose_msg_to_T(pose) -> np.ndarray:
        """geometry_msgs/Pose -> 4x4 homogeneous transform."""
        T = np.eye(4)
        q_xyzw = [pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w]
        T[:3, :3] = _ScipyRotation.from_quat(q_xyzw).as_matrix()
        T[0, 3] = pose.position.x
        T[1, 3] = pose.position.y
        T[2, 3] = pose.position.z
        return T

    def _cart_state_callback(self, msg: MechUnitStateArray) -> None:
        """Cache the controller-reported end-effector pose + psi for our unit."""
        for state in msg.states:
            if int(state.mu_idx) != self.right_arm_mu_idx:
                continue
            try:
                T = self._pose_msg_to_T(state.end_pose)
            except Exception:  # noqa: BLE001 - malformed pose
                return
            if not np.all(np.isfinite(T)):
                return
            psi = float(state.psi) if np.isfinite(float(state.psi)) else None
            with self._cart_state_lock:
                self._latest_cart_pose = T
                self._latest_cart_psi = psi
            return

    def _get_latest_cart(self):
        with self._cart_state_lock:
            T = None if self._latest_cart_pose is None \
                else self._latest_cart_pose.copy()
            psi = self._latest_cart_psi
            return T, psi

    def _collect_cal_sample(self) -> None:
        """Snapshot a (measured q, reported pose, reported psi) calibration tuple.

        Called every tick while the move-to-start ramp drives the arm in joint
        space. Only stored when both a joint feedback and a reported cartesian
        pose are available so the pair is self-consistent.
        """
        q_meas = self._get_latest_q()
        T_report, psi_report = self._get_latest_cart()
        if q_meas is None or T_report is None:
            return
        self._cart_cal_samples.append({
            'q': q_meas, 'T': T_report,
            'psi': np.nan if psi_report is None else float(psi_report),
        })

    @staticmethod
    def _vec_to_T(v: np.ndarray) -> np.ndarray:
        """6-vector (tx,ty,tz, rx,ry,rz rotvec) -> 4x4 homogeneous transform."""
        T = np.eye(4)
        T[:3, :3] = _ScipyRotation.from_rotvec(v[3:]).as_matrix()
        T[:3, 3] = v[:3]
        return T

    def _fit_cartesian_calibration(self) -> bool:
        """Fit T_report = B @ fk(q) @ C and the psi convention from samples.

        Returns True on success. The controller's reported pose differs from our
        FK by a base-frame offset B (controller base <- our arm-mount root) and a
        tool offset C (our site -> controller TCP); a single left-multiply cannot
        absorb C because it does not commute with the arm's changing orientation.
        psi is fitted as sign*arm_angle(q, ref) + offset over a small set of
        candidate reference directions, picking the best match to reported psi.
        """
        samples = self._cart_cal_samples
        if len(samples) < self.cart_cal_min_samples:
            self.get_logger().error(
                f'Cartesian calibration: only {len(samples)} samples collected '
                f'(need >= {self.cart_cal_min_samples}). The move-to-start ramp '
                'did not yield enough /cart_states + /joint_states pairs. '
                'Aborting.')
            return False

        # Guard against a degenerate ramp: if the arm barely moved (e.g. it
        # already started at the trajectory start), the samples are near
        # identical and the B/C fit is unidentifiable -- least squares would
        # report a meaningless "0mm" residual from a rank-deficient problem.
        # Require enough joint-space sweep to pin down both offsets.
        q_all = np.array([s['q'] for s in samples])
        q_span_deg = float(np.degrees(np.max(q_all.max(0) - q_all.min(0))))
        if q_span_deg < 5.0:
            self.get_logger().error(
                f'Cartesian calibration: ramp swept only {q_span_deg:.1f} deg '
                '(max per-joint range); too little motion to identify the '
                'pose offset. The arm likely started at the trajectory start. '
                'Start the arm away from the start pose, increase '
                'move_to_start_duration, or use cart_calibrate:=false. '
                'Aborting.')
            return False

        A = [self.fk.ee_pose(s['q']) for s in samples]
        T_rep = [s['T'] for s in samples]

        def resid(p):
            B = self._vec_to_T(p[:6])
            C = self._vec_to_T(p[6:])
            r = []
            for Ai, Ti in zip(A, T_rep):
                E = B @ Ai @ C @ np.linalg.inv(Ti)
                r.extend(E[:3, 3])
                r.extend(_ScipyRotation.from_matrix(E[:3, :3]).as_rotvec())
            return r

        sol = _scipy_least_squares(
            resid, np.zeros(12), method='lm', max_nfev=20000)
        self._cart_B = self._vec_to_T(sol.x[:6])
        self._cart_C = self._vec_to_T(sol.x[6:])

        # Fit-quality on the calibration samples (position residual).
        pe = []
        for Ai, Ti in zip(A, T_rep):
            Tp = self._cart_B @ Ai @ self._cart_C
            pe.append(float(np.linalg.norm(Tp[:3, 3] - Ti[:3, 3])))
        pe = np.asarray(pe)
        self._cart_fit_pos_rms = float(np.sqrt(np.mean(pe ** 2)))
        self._cart_fit_pos_max = float(pe.max())

        # --- psi convention: choose reference direction + sign + offset ---
        psis = np.array([s['psi'] for s in samples])
        have_psi = np.isfinite(psis)
        if have_psi.sum() >= max(12, self.cart_cal_min_samples // 2):
            self._fit_psi_convention(
                [s['q'] for s in samples], psis, have_psi)
        else:
            self.get_logger().warn(
                'Cartesian calibration: too few reported psi samples; '
                'psi computed with default reference (may not match controller)')

        self._cart_calibrated = True
        self.get_logger().info(
            'Cartesian calibration fitted from '
            f'{len(samples)} samples: pose model T=B@fk(q)@C '
            f'pos rms={self._cart_fit_pos_rms*1e3:.1f}mm '
            f'max={self._cart_fit_pos_max*1e3:.1f}mm | '
            f'B_t={np.round(self._cart_B[:3,3],4).tolist()} '
            f'C_t={np.round(self._cart_C[:3,3],4).tolist()}')
        if self._cart_fit_pos_max > 0.02:
            self.get_logger().warn(
                f'Cartesian pose-fit max residual '
                f'{self._cart_fit_pos_max*1e3:.0f}mm is high; the ramp may lack '
                'orientation diversity or the model is off. Inspect tracking.')
        return True

    def _fit_psi_convention(self, qs, psis, mask) -> None:
        """Pick arm-angle reference dir + sign + circular offset matching psi."""
        candidates = {
            '+X': np.array([1.0, 0.0, 0.0]), '-X': np.array([-1.0, 0.0, 0.0]),
            '+Y': np.array([0.0, 1.0, 0.0]), '-Y': np.array([0.0, -1.0, 0.0]),
            '+Z': np.array([0.0, 0.0, 1.0]), '-Z': np.array([0.0, 0.0, -1.0]),
        }
        idx = np.where(mask)[0]
        best = None  # (rmse, name, ref, sign, offset)
        for name, ref in candidates.items():
            aa = np.array([self.fk.arm_angle(qs[i], ref) for i in idx])
            ok = np.isfinite(aa)
            if ok.sum() < 12:
                continue
            for sign in (1.0, -1.0):
                d = psis[idx][ok] - sign * aa[ok]
                off = float(np.arctan2(np.sin(d).mean(), np.cos(d).mean()))
                err = np.angle(np.exp(1j * (psis[idx][ok] - (sign * aa[ok] + off))))
                rmse = float(np.sqrt(np.mean(err ** 2)))
                if best is None or rmse < best[0]:
                    best = (rmse, name, ref, sign, off)
        if best is None:
            self.get_logger().warn(
                'psi convention fit failed (degenerate samples); using default')
            return
        rmse, name, ref, sign, off = best
        self._cart_psi_ref = ref
        self._cart_psi_sign = sign
        self._cart_psi_offset = off
        self.get_logger().info(
            f'psi convention: ref={name} sign={sign:+.0f} '
            f'offset={np.degrees(off):.1f}deg rmse={np.degrees(rmse):.2f}deg')
        if rmse > np.radians(10.0):
            self.get_logger().warn(
                f'psi fit rmse {np.degrees(rmse):.1f}deg is high; the '
                'controller may use a different redundancy convention. '
                'Consider psi_mode:=zero or passthrough.')

    def _compute_cartesian_cmd(self, q: np.ndarray):
        """Return (T_cmd 4x4, psi) for joint command ``q`` in controller frame."""
        T_cmd = self._cart_B @ self.fk.ee_pose(q) @ self._cart_C
        if self.psi_mode == 'zero':
            psi = 0.0
        elif self.psi_mode == 'passthrough':
            _, psi_report = self._get_latest_cart()
            psi = float(psi_report) if psi_report is not None else 0.0
        else:  # computed
            psi = (self._cart_psi_sign
                   * self.fk.arm_angle(q, self._cart_psi_ref)
                   + self._cart_psi_offset)
        return T_cmd, psi

    def _publish_cartesian(self, q: np.ndarray) -> None:
        """Publish ``q`` as an end-effector pose command (use_jnt=False)."""
        T_cmd, psi = self._compute_cartesian_cmd(q)
        cmd = MechUnitCmd()
        cmd.mu_idx = self.right_arm_mu_idx
        cmd.use_jnt = False
        cmd.psi = float(psi)
        cmd.end_pose.position.x = float(T_cmd[0, 3])
        cmd.end_pose.position.y = float(T_cmd[1, 3])
        cmd.end_pose.position.z = float(T_cmd[2, 3])
        quat_xyzw = _ScipyRotation.from_matrix(T_cmd[:3, :3]).as_quat()
        cmd.end_pose.orientation.x = float(quat_xyzw[0])
        cmd.end_pose.orientation.y = float(quat_xyzw[1])
        cmd.end_pose.orientation.z = float(quat_xyzw[2])
        cmd.end_pose.orientation.w = float(quat_xyzw[3])
        msg = MechUnitCmdArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.cmds = [cmd]
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish_q(self, q: np.ndarray) -> None:
        """Publish one right-arm command for joint position ``q`` (rad).

        Single funnel for every phase (ramp, hold, replay). In cartesian output
        mode the command is published as an end-effector pose (use_jnt=False)
        *once calibration is complete*; while the move-to-start ramp is still
        collecting calibration samples it is published in joint space, so no
        cartesian command is ever sent against an unfitted pose model.
        """
        if self.clip_to_limits:
            q = np.clip(q, self.q_low, self.q_high)
        if self.output_mode == 'cartesian' and self._cart_calibrated:
            self._publish_cartesian(q)
            return
        self._publish_joint(q)

    def _publish_joint(self, q: np.ndarray) -> None:
        """Publish ``q`` as a joint position command (use_jnt=True)."""
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
        # In cartesian mode the ramp is published in joint space and used to
        # collect calibration samples; the fit happens when the ramp completes
        # (see _finish_ramp). With move_to_start disabled, calibration is not
        # possible (guarded in _init_cartesian unless cart_calibrate=false).
        if (self.output_mode == 'cartesian' and not self.cart_calibrate
                and not self._cart_calibrated):
            self.get_logger().warn(
                'cart_calibrate=false: publishing raw FK poses (B=C=identity). '
                'This only matches the robot if the controller expects poses '
                'in our arm-mount frame with TCP at the model site.')
            self._cart_calibrated = True

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
        # Collect a calibration sample each tick while the arm sweeps the ramp
        # in joint space (cartesian mode with calibration enabled only).
        if (self.output_mode == 'cartesian' and self.cart_calibrate
                and not self._cart_calibrated):
            self._collect_cal_sample()
        self._ramp_idx += 1
        if self._ramp_idx >= self._ramp_traj.shape[0]:
            self.get_logger().info('Reached start pose; holding')
            # Fit the cartesian calibration from the ramp samples before any
            # cartesian command is published. Abort if the fit is unusable.
            if (self.output_mode == 'cartesian' and self.cart_calibrate
                    and not self._cart_calibrated):
                if not self._fit_cartesian_calibration():
                    self._phase = 'done'
                    self.timer.cancel()
                    return
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
        # Seed the compensator's command buffer with the pose the arm is
        # actually holding (start pose), so the first pre-distorted commands are
        # computed against a consistent history instead of zeros.
        if self.compensator.enabled:
            seed_q = self._get_latest_q()
            if seed_q is None:
                seed_q = self.replay_start_pose
            self.compensator.reset(seed_q)
        if self.record_data:
            self._recording_active = True
            self.recorded_data = []
            self.get_logger().info('Data recording activated')
        self.get_logger().info(
            f'Beginning replay ({self.n_steps} steps, single loop)')

    def _step_replay(self) -> None:
        idx = self._replay_idx
        nominal_q = self.trajectory[idx]

        # Apply actuator-delay compensation if enabled: pre-distort the nominal
        # command using the recorded (qdot, qdd) and the latest measured pose so
        # the real actuator's delay+lag realizes the nominal motion. When the
        # compensator is in 'none' mode this returns the nominal command (only
        # clipped), so the published command is identical to the legacy path.
        actual_q, actual_qvel, actual_qdd = self._get_latest_state()
        if self.compensator.enabled:
            applied_q = actual_q if actual_q is not None else nominal_q
            cmd_q = self.compensator.step(
                nominal_q,
                self.cmd_qvel_traj[idx],
                self.cmd_qdd_traj[idx],
                applied_q,
            )
        else:
            cmd_q = nominal_q

        # Record data if recording is active
        if self._recording_active:
            if actual_q is not None:
                record = {
                    'step': idx,
                    'timestamp': self.get_clock().now().nanoseconds * 1e-9,
                    'cmd_q': cmd_q.copy(),
                    'nominal_q': nominal_q.copy(),
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

                # Log the controller-reported actual EE pose in EVERY mode (it
                # comes straight from /cart_states), so joint and cartesian
                # replays can be compared in task space. The commanded EE pose
                # (cart_cmd_*) is only available in cartesian mode, where FK is
                # loaded.
                T_report, psi_report = self._get_latest_cart()
                if T_report is not None:
                    record['cart_actual_pos'] = T_report[:3, 3].copy()
                    record['cart_actual_quat'] = _ScipyRotation.from_matrix(
                        T_report[:3, :3]).as_quat()
                if psi_report is not None:
                    record['cart_actual_psi'] = float(psi_report)
                if self.output_mode == 'cartesian':
                    T_cmd, psi_cmd = self._compute_cartesian_cmd(cmd_q)
                    record['cart_cmd_pos'] = T_cmd[:3, 3].copy()
                    record['cart_cmd_quat'] = _ScipyRotation.from_matrix(
                        T_cmd[:3, :3]).as_quat()  # xyzw
                    record['cart_cmd_psi'] = float(psi_cmd)
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

            # Nominal (pre-compensation) command, present when compensation ran.
            # cmd_q is what was published; nominal_q is the recorded trajectory.
            if 'nominal_q' in self.recorded_data[i]:
                for j, joint in enumerate(RIGHT_ARM_JOINTS):
                    row[f'nominal_q/{joint}'] = self.recorded_data[i]['nominal_q'][j]

            # Add theoretical values if available
            if 'theoretical_q' in self.recorded_data[i]:
                for j, joint in enumerate(RIGHT_ARM_JOINTS):
                    row[f'theoretical_q/{joint}'] = self.recorded_data[i]['theoretical_q'][j]
                    row[f'theoretical_qvel/{joint}'] = self.recorded_data[i]['theoretical_qvel'][j]
                    row[f'theoretical_qdd/{joint}'] = self.recorded_data[i]['theoretical_qdd'][j]
                if 'action' in self.recorded_data[i]:
                    for j, joint in enumerate(RIGHT_ARM_JOINTS):
                        row[f'action/{joint}'] = self.recorded_data[i]['action'][j]

            # End-effector pose columns. cart_actual_* (controller-reported) is
            # written in EVERY mode so joint and cartesian replays compare in
            # task space; cart_cmd_* (what we commanded) only exists in
            # cartesian mode. Emit a fixed column set so the CSV header is
            # consistent across rows; missing samples are NaN.
            axes = ('x', 'y', 'z')
            quat_axes = ('x', 'y', 'z', 'w')
            rec = self.recorded_data[i]
            act_pos = rec.get('cart_actual_pos')
            act_quat = rec.get('cart_actual_quat')
            for a in range(3):
                row[f'cart_actual_pos/{axes[a]}'] = (
                    float(act_pos[a]) if act_pos is not None else np.nan)
            for a in range(4):
                row[f'cart_actual_quat/{quat_axes[a]}'] = (
                    float(act_quat[a]) if act_quat is not None else np.nan)
            row['cart_actual_psi'] = float(rec.get('cart_actual_psi', np.nan))
            if self.output_mode == 'cartesian':
                cmd_pos = rec.get('cart_cmd_pos')
                cmd_quat = rec.get('cart_cmd_quat')
                for a in range(3):
                    row[f'cart_cmd_pos/{axes[a]}'] = (
                        float(cmd_pos[a]) if cmd_pos is not None else np.nan)
                for a in range(4):
                    row[f'cart_cmd_quat/{quat_axes[a]}'] = (
                        float(cmd_quat[a]) if cmd_quat is not None else np.nan)
                row['cart_cmd_psi'] = float(rec.get('cart_cmd_psi', np.nan))

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
            'compensation_mode': self.compensation_mode,
            'output_mode': self.output_mode,
        }
        if self.output_mode == 'cartesian':
            summary['cartesian'] = {
                'psi_mode': self.psi_mode,
                'calibrated': self._cart_calibrated,
                'cart_calibrate': self.cart_calibrate,
                'robot_xml_path': self.robot_xml_path,
                'n_cal_samples': len(self._cart_cal_samples),
                'pose_model': 'T_cmd = B @ fk(q) @ C',
                'B': self._cart_B.astype(float).tolist(),
                'C': self._cart_C.astype(float).tolist(),
                'B_translation_m': self._cart_B[:3, 3].astype(float).tolist(),
                'C_translation_m': self._cart_C[:3, 3].astype(float).tolist(),
                'fit_pos_rms_m': self._cart_fit_pos_rms,
                'fit_pos_max_m': self._cart_fit_pos_max,
                'psi_ref_dir': self._cart_psi_ref.astype(float).tolist(),
                'psi_sign': float(self._cart_psi_sign),
                'psi_offset_rad': float(self._cart_psi_offset),
            }
        if self.compensator.enabled:
            summary['compensation'] = {
                'mode': self.compensator.mode,
                'delay_ms': self.compensator.delay_ms,
                'delay_steps': self.compensator.delay_steps,
                'actuator_tau_s': self.compensator.actuator_tau_s,
                'actuator_gain': self.compensator.actuator_gain,
                'command_buffer_len': self.compensator.command_buffer_len,
                'mpc_beta': self.compensator.mpc_beta,
                'mpc_horizon_steps': self.compensator.mpc_horizon_steps,
                'mpc_delay_scale': self.compensator.mpc_delay_scale,
                'mpc_tau_scale': self.compensator.mpc_tau_scale,
                'mpc_max_delta_rad': self.compensator.mpc_max_delta_rad,
            }
            # How far the published command was pushed away from the nominal.
            if 'nominal_q' in self.recorded_data[0]:
                nominal_q = np.array(
                    [r['nominal_q'] for r in self.recorded_data])
                comp_delta = cmd_q - nominal_q
                summary['compensation']['applied_delta'] = {
                    'mean_abs_deg': float(np.degrees(np.mean(np.abs(comp_delta)))),
                    'max_abs_deg': float(np.degrees(np.max(np.abs(comp_delta)))),
                    'rms_deg': float(np.degrees(np.sqrt(np.mean(comp_delta**2)))),
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
