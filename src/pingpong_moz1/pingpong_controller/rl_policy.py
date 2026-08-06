"""RL policy controller for right-arm ping-pong juggling.

Wraps a stable-baselines3 PPO policy that was trained with
`rl_juggle_env_random.JuggleEnv`. Converts incoming ball observations
(position / velocity in base_link frame, meters) into 7 right-arm joint
angles (radians), reproducing the env's action-to-joint-position
integration so the deployed behavior matches training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import mujoco as mj
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False


RIGHT_ARM_JOINTS = [
    "RightArm-0", "RightArm-1", "RightArm-2",
    "RightArm-3", "RightArm-4", "RightArm-5", "RightArm-6",
]

LEFT_ARM_JOINTS = [
    "LeftArm-0", "LeftArm-1", "LeftArm-2",
    "LeftArm-3", "LeftArm-4", "LeftArm-5", "LeftArm-6",
]

LEG_WAIST_JOINTS = [
    "LegWaist-0", "LegWaist-1", "LegWaist-2",
    "LegWaist-3", "LegWaist-4", "LegWaist-5",
]

HEAD_JOINTS = ["Head-0", "Head-1"]

# Full-body TARGET pose (degrees) mirroring rl_juggle_env_random.TARGET_DEGREES.
TARGET_DEGREES = {
    "RightArm-0": 9.0, "RightArm-1": -50.0, "RightArm-2": 20.0,
    "RightArm-3": 90.0, "RightArm-4": 45.0, "RightArm-5": -8.0,
    "RightArm-6": 45.0,
    "LeftArm-0": -9.0, "LeftArm-1": -50.0, "LeftArm-2": -20.0,
    "LeftArm-3": -90.0, "LeftArm-4": -35.0, "LeftArm-5": 8.0,
    "LeftArm-6": -45.0,
    "LegWaist-0": 0.0, "LegWaist-1": 60.0, "LegWaist-2": -90.0,
    "LegWaist-3": 30.0, "LegWaist-4": 0.0, "LegWaist-5": 0.0,
    "Head-0": 0.0, "Head-1": 40.0,
}

# Mirrors JuggleConfig.arm_{vel,acc}_limit_*.
DEFAULT_ARM_VEL_LIMIT_DEG_S = (210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0)
DEFAULT_ARM_ACC_LIMIT_DEG_S2 = (
    1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0)

# Conservative joint limits fallback (training env reads these from MuJoCo XML
# via jnt_range; we do not load the XML at runtime, so use wide safe bounds).
# TODO(jacky): load exact jnt_range from robot XML to match training clip.
DEFAULT_ARM_Q_LOW_RAD = tuple(-np.pi for _ in range(7))
DEFAULT_ARM_Q_HIGH_RAD = tuple(np.pi for _ in range(7))

# JuggleConfig.ball_obs_age_clip.
DEFAULT_BALL_OBS_AGE_CLIP_S = 0.20
# JuggleConfig.action_acc_scale (base value; action_scale_mult is DR-only).
DEFAULT_ACTION_ACC_SCALE = 1.0
# MJX _point_to_base keeps the input point's z unchanged, so FK-derived racket
# positions must keep MuJoCo world z. The real-camera path converts its z into
# this same world-height convention before building policy observations.
DEPLOYMENT_FK_WORLD_Z_CORRECTION_M = 0.0

EXPECTED_OBS_DIM = 50  # Base observation dimension (legacy)
EXPECTED_OBS_DIM_WITH_DELAY = 67  # 50 base + 17 delay conditioning

# Observation contracts are explicit because the deployed legacy checkpoint was
# trained with two observation bugs, while simcode2/ideal-PD67 was trained after
# those bugs were fixed.  The legacy contract remains the default so existing
# launch files and checkpoints keep their current behaviour.
OBSERVATION_PROFILE_LEGACY_BUG_COMPAT = "legacy_bug_compatible"
OBSERVATION_PROFILE_SIMCODE2_IDEALPD67 = "simcode2_idealpd67"
OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67 = "simcode3_actuator67"
OBSERVATION_PROFILE_SIMCODE4_SERVO67 = "simcode4_servo67"
OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67 = "simcode5_pure_actuator67"
OBSERVATION_PROFILES = (
    OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
    OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
    OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
    OBSERVATION_PROFILE_SIMCODE4_SERVO67,
    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
)
CORRECTED_67D_OBSERVATION_PROFILES = (
    OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
    OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
    OBSERVATION_PROFILE_SIMCODE4_SERVO67,
    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
)


def uses_corrected_67d_observation(value: object) -> bool:
    """Whether a profile uses the simulation-aligned corrected 67D contract."""
    return normalize_observation_profile(value) in CORRECTED_67D_OBSERVATION_PROFILES


def normalize_observation_profile(value: object) -> str:
    """Return a canonical deployment observation profile name."""
    profile = str(value or OBSERVATION_PROFILE_LEGACY_BUG_COMPAT).strip().lower()
    profile = profile.replace("-", "_").replace(" ", "_")
    aliases = {
        "legacy": OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
        "legacy_bug": OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
        "bug_compatible": OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
        "simcode2": OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
        "idealpd67": OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
        "ideal_pd67": OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
        "simcode3": OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
        "actuator67": OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
        "actuator_67": OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
        "simcode4": OBSERVATION_PROFILE_SIMCODE4_SERVO67,
        "servo67": OBSERVATION_PROFILE_SIMCODE4_SERVO67,
        "servo_67": OBSERVATION_PROFILE_SIMCODE4_SERVO67,
        "simcode5": OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
        "pure_actuator67": OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
        "pure_actuator_67": OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
    }
    profile = aliases.get(profile, profile)
    if profile not in OBSERVATION_PROFILES:
        raise ValueError(
            f"unknown observation_profile={value!r}; expected one of "
            f"{OBSERVATION_PROFILES}")
    return profile


def target_right_arm_q_rad() -> np.ndarray:
    """Initial right-arm posture (rad) matching the training TARGET_DEGREES."""
    return np.deg2rad(
        np.array([TARGET_DEGREES[n] for n in RIGHT_ARM_JOINTS], dtype=np.float32)
    )


def target_joint_positions_rad(joint_names) -> np.ndarray:
    """Look up TARGET_DEGREES for each joint name and return radians."""
    return np.deg2rad(np.array(
        [TARGET_DEGREES[n] for n in joint_names], dtype=np.float64))


def target_left_arm_q_rad() -> np.ndarray:
    return target_joint_positions_rad(LEFT_ARM_JOINTS)


def target_leg_waist_q_rad() -> np.ndarray:
    return target_joint_positions_rad(LEG_WAIST_JOINTS)


def target_head_q_rad() -> np.ndarray:
    return target_joint_positions_rad(HEAD_JOINTS)


@dataclass
class _LoggerShim:
    """Minimal logger shim so this module can run without a ROS node."""

    def info(self, msg: str) -> None:
        print(f"[RLPolicy][INFO] {msg}")

    def warn(self, msg: str) -> None:
        print(f"[RLPolicy][WARN] {msg}")

    def error(self, msg: str) -> None:
        print(f"[RLPolicy][ERROR] {msg}")


class RLPolicyController:
    """Loads a PPO policy and maps ball state to right-arm joint targets.

    The controller is stateful: every call updates the internal command
    trajectory (arm_cmd_q, arm_cmd_qvel) via action acceleration integration,
    exactly like `JuggleEnv.step` with `arm_action_limiter=False`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        dt: float = 0.005,
        action_acc_scale: float = DEFAULT_ACTION_ACC_SCALE,
        arm_vel_limit_deg_s=DEFAULT_ARM_VEL_LIMIT_DEG_S,
        arm_acc_limit_deg_s2=DEFAULT_ARM_ACC_LIMIT_DEG_S2,
        arm_q_low_rad=DEFAULT_ARM_Q_LOW_RAD,
        arm_q_high_rad=DEFAULT_ARM_Q_HIGH_RAD,
        ball_obs_age_clip_s: float = DEFAULT_BALL_OBS_AGE_CLIP_S,
        deterministic: bool = True,
        device: str = "cpu",
        robot_xml_path: Optional[str] = None,
        logger: Optional[object] = None,
        arm_action_limiter: bool = False,
        enable_delay_conditioning: bool = True,
        use_valid_ball_obs_age: bool = True,
        observation_profile: str = OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
    ) -> None:
        self._logger = logger or _LoggerShim()
        self.dt = float(dt)
        self.action_acc_scale = float(action_acc_scale)
        self.deterministic = bool(deterministic)
        self.ball_obs_age_clip_s = max(1e-6, float(ball_obs_age_clip_s))
        # When True, clip mapped joint acceleration and the integrated joint
        # velocity to the arm limits inside predict(), mirroring the training
        # env's `arm_action_limiter=True` branch. The SB3 reference path was
        # trained with the limiter off, so this defaults to False to keep
        # that path byte-for-byte unchanged.
        self.arm_action_limiter = bool(arm_action_limiter)

        # Delay conditioning: when True, observation includes 17 extra dimensions
        # for delay compensation features (obs_dim=67); when False, uses legacy
        # 50-dim observation without delay features.
        self.enable_delay_conditioning = bool(enable_delay_conditioning)
        self.use_valid_ball_obs_age = bool(use_valid_ball_obs_age)
        self.observation_profile = normalize_observation_profile(
            observation_profile)
        if (uses_corrected_67d_observation(self.observation_profile)
                and not self.enable_delay_conditioning):
            raise ValueError(
                f"{self.observation_profile} requires the 67D delay-conditioned "
                "observation path")
        self._expected_obs_dim = (
            EXPECTED_OBS_DIM_WITH_DELAY if self.enable_delay_conditioning
            else EXPECTED_OBS_DIM
        )

        # Replicate simulation bug for validation purposes.
        # The training environment (mjx_juggle_env.py) has a bug where _make_obs
        # uses already-updated prev_racket_pos, causing racket_vel ≈ 0 in training.
        # Set to True to replicate this bug and verify policy performance when
        # real-robot obs perfectly match training distribution.
        # Set to False to use physically correct racket velocity.
        self.force_obs_to_training_dist = (
            self.observation_profile
            == OBSERVATION_PROFILE_LEGACY_BUG_COMPAT)
        # When True, clip mapped joint acceleration and the integrated joint
        # velocity to the arm limits inside predict(), mirroring the training
        # env's `arm_action_limiter=True` branch. The SB3 reference path was
        # trained with the limiter off, so this defaults to False to keep
        # that path byte-for-byte unchanged.
        self.arm_action_limiter = bool(arm_action_limiter)

        self.n_arm = len(RIGHT_ARM_JOINTS)
        self.arm_vel_limit_rad_s = np.deg2rad(
            np.asarray(arm_vel_limit_deg_s, dtype=np.float32))
        self.arm_acc_limit_rad_s2 = np.deg2rad(
            np.asarray(arm_acc_limit_deg_s2, dtype=np.float32))
        self.arm_q_low = np.asarray(arm_q_low_rad, dtype=np.float32)
        self.arm_q_high = np.asarray(arm_q_high_rad, dtype=np.float32)
        self.initial_arm_q_rad = target_right_arm_q_rad().astype(np.float32)
        self.delay_conditioning_tau_s = 0.072
        self.delay_conditioning_max_s = 0.072
        self.command_buffer_extra_steps = 4
        self.max_contact_time_s = 0.50
        self.contact_height_offset_m = 0.0
        self.contact_lost_ball_timeout_s = (
            0.35
            if uses_corrected_67d_observation(self.observation_profile)
            else 0.15)
        if (len(self.arm_vel_limit_rad_s) != self.n_arm
                or len(self.arm_acc_limit_rad_s2) != self.n_arm
                or len(self.arm_q_low) != self.n_arm
                or len(self.arm_q_high) != self.n_arm):
            raise ValueError(
                "arm limit arrays must have length 7 (RIGHT_ARM_JOINTS)")

        # Load the policy. Subclasses override `_load_model` to support
        # other checkpoint formats (e.g. MJX/JAX .pkl). The base class loads
        # a stable-baselines3 PPO .zip. Failures are re-raised so the caller
        # can fall back to default joint positions.
        self._load_model(model_path, device)
        self._validate_io_shapes()
        self._logger.info(
            f"RL policy loaded | obs_dim={self._expected_obs_dim}, "
            f"act_dim={self.n_arm}, dt={self.dt:.4f}s, device={device}, "
            f"arm_action_limiter={self.arm_action_limiter}, "
            f"delay_conditioning={self.enable_delay_conditioning}, "
            f"use_valid_ball_obs_age={self.use_valid_ball_obs_age}, "
            f"observation_profile={self.observation_profile}")

        # Optional MuJoCo FK for racket position/velocity.
        self.mj_model = None
        self.mj_data = None
        self._racket_site_id = -1
        self._racket_geom_id = -1
        self._arm_jids = []
        self._base_body_id = -1
        self._base_x_qadr = -1
        self._base_y_qadr = -1
        self._base_yaw_qadr = -1
        self._base_x_vadr = -1
        self._base_y_vadr = -1
        self._base_yaw_vadr = -1
        self._base_obs_q = np.zeros(3, dtype=np.float32)
        self._base_obs_dq = np.zeros(3, dtype=np.float32)
        self._prev_racket_pos_base = np.zeros(3, dtype=np.float32)
        if robot_xml_path and _MUJOCO_AVAILABLE:
            try:
                xml_path = Path(robot_xml_path).resolve()
                if not xml_path.exists():
                    self._logger.warn(
                        f"robot_xml_path does not exist: {xml_path}; "
                        "racket FK disabled")
                else:
                    if uses_corrected_67d_observation(
                            self.observation_profile):
                        from pingpong_controller.simcode2_xml import (
                            build_simcode2_training_xml,
                        )
                        xml_path = build_simcode2_training_xml(xml_path)
                    self.mj_model = mj.MjModel.from_xml_path(str(xml_path))
                    self.mj_data = mj.MjData(self.mj_model)
                    self._racket_site_id = mj.mj_name2id(
                        self.mj_model, mj.mjtObj.mjOBJ_SITE, "right_ee_site")
                    self._racket_geom_id = mj.mj_name2id(
                        self.mj_model, mj.mjtObj.mjOBJ_GEOM,
                        "racket_rubber_fore")
                    self._arm_jids = [
                        mj.mj_name2id(
                            self.mj_model, mj.mjtObj.mjOBJ_JOINT, n)
                        for n in RIGHT_ARM_JOINTS
                    ]
                    if self._racket_site_id < 0 or any(
                            j < 0 for j in self._arm_jids):
                        self._logger.warn(
                            "MuJoCo model missing right_ee_site or right-arm "
                            "joints; racket FK disabled")
                        self.mj_model = None
                        self.mj_data = None
                    else:
                        # Override joint limits with the XML's jnt_range so
                        # arm_cmd_q is clipped the same way as in training.
                        xml_lo = np.array([
                            float(self.mj_model.jnt_range[j, 0])
                            for j in self._arm_jids
                        ], dtype=np.float32)
                        xml_hi = np.array([
                            float(self.mj_model.jnt_range[j, 1])
                            for j in self._arm_jids
                        ], dtype=np.float32)
                        self.arm_q_low = xml_lo
                        self.arm_q_high = xml_hi
                        # Base body (for world->base transforms). "base_link"
                        # name matches JuggleEnv; fall back to "base" for the
                        # training-env XML which uses that name.
                        base_bid = mj.mj_name2id(
                            self.mj_model, mj.mjtObj.mjOBJ_BODY, "base_link")
                        if base_bid < 0:
                            base_bid = mj.mj_name2id(
                                self.mj_model, mj.mjtObj.mjOBJ_BODY, "base")
                        self._base_body_id = int(base_bid)
                        # MJX _point_to_base uses the generalized base_x,
                        # base_y and base_yaw coordinates directly.  Extracting
                        # yaw from the tilted base body's xmat is not equivalent
                        # when base roll/pitch are non-zero after warm-up.
                        for joint_name, q_attr, v_attr in (
                                ("base_x", "_base_x_qadr", "_base_x_vadr"),
                                ("base_y", "_base_y_qadr", "_base_y_vadr"),
                                ("base_yaw", "_base_yaw_qadr", "_base_yaw_vadr")):
                            jid = mj.mj_name2id(
                                self.mj_model, mj.mjtObj.mjOBJ_JOINT,
                                joint_name)
                            if jid >= 0:
                                setattr(
                                    self, q_attr,
                                    int(self.mj_model.jnt_qposadr[jid]))
                                setattr(
                                    self, v_attr,
                                    int(self.mj_model.jnt_dofadr[jid]))
                        # Seed non-right-arm joints to the training TARGET_DEGREES
                        # so FK matches the posture the robot will actually hold.
                        self._seed_non_right_arm_joints_to_target()
                        if uses_corrected_67d_observation(
                                self.observation_profile):
                            # simcode2 builds every observation from the
                            # 700-step XML-PD warm state.  Keeping the other
                            # joints at their exact target angles raises the
                            # racket by about 1 cm and is measurably outside
                            # the launch19 training observation distribution.
                            self._warm_fk_model_to_training_posture()
                        self._logger.info(
                            f"MuJoCo FK enabled (xml={xml_path.name}); "
                            "arm joint limits taken from XML; "
                            "non-right-arm joints seeded to TARGET_DEGREES")
            except Exception as exc:
                self._logger.warn(
                    f"Failed to load MuJoCo model from {robot_xml_path}: "
                    f"{exc}; racket FK disabled")
                self.mj_model = None
                self.mj_data = None
        elif robot_xml_path and not _MUJOCO_AVAILABLE:
            self._logger.warn(
                "robot_xml_path provided but mujoco not available; "
                "racket FK disabled")

        self.reset()

    # ------------------------------------------------------------------
    # Model loading / inference (overridable by subclasses)
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, device: str) -> None:
        """Load a stable-baselines3 PPO policy from a .zip checkpoint.

        Subclasses override this to support other formats. After this
        returns, `self.model` must expose `observation_space.shape` and
        `action_space.shape`, and `_policy_action` must be able to run a
        forward pass on a (EXPECTED_OBS_DIM,) observation.
        """
        try:
            from stable_baselines3 import PPO  # type: ignore
        except Exception as exc:  # pragma: no cover - import-time only
            raise RuntimeError(
                f"stable-baselines3 is required to load the RL policy: {exc}"
            ) from exc

        self._logger.info(f"Loading PPO policy from: {model_path}")
        self.model = PPO.load(model_path, device=device)

    def _validate_io_shapes(self) -> None:
        """Check the loaded model's obs/act dims against expectations."""
        obs_space = getattr(self.model, "observation_space", None)
        obs_shape = getattr(obs_space, "shape", None)
        if obs_shape is None or tuple(obs_shape) != (self._expected_obs_dim,):
            raise RuntimeError(
                f"Loaded model observation shape {obs_shape} does not match "
                f"expected ({self._expected_obs_dim},)")
        act_space = getattr(self.model, "action_space", None)
        act_shape = getattr(act_space, "shape", None)
        if act_shape is None or tuple(act_shape) != (self.n_arm,):
            raise RuntimeError(
                f"Loaded model action shape {act_shape} does not match "
                f"expected ({self.n_arm},)")

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        """Map an observation to a raw (pre-clip) action vector.

        Base implementation calls SB3's `model.predict`. Subclasses
        override to run their own forward pass. Returns a float32 array
        of shape (n_arm,); the caller clips it to [-1, 1].
        """
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(self.n_arm)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset command state to the training initial posture."""
        self.arm_cmd_q = self.initial_arm_q_rad.astype(np.float32).copy()
        self.arm_cmd_q = np.clip(
            self.arm_cmd_q, self.arm_q_low, self.arm_q_high)
        self.arm_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.prev_action = np.zeros(self.n_arm, dtype=np.float32)
        self._last_ball_stamp: Optional[float] = None
        # Cached ball observation for dropout-like hold-last behavior.
        self._last_valid_ball_pos_m = np.zeros(3, dtype=np.float32)
        self._last_valid_ball_vel_m_s = np.zeros(3, dtype=np.float32)
        self._has_valid_ball_obs = False
        # Fixed reset racket anchor for done-condition checks. Keep it separate
        # from _prev_racket_pos_base, which is mutated every observation step.
        self._racket_anchor_pos_base: Optional[np.ndarray] = None
        # Initialize prev racket pos (base frame) for velocity estimation.
        if self.mj_model is not None and self.mj_data is not None:
            rpos_base, _ = self._compute_racket_pose_base(self.arm_cmd_q)
            self._racket_anchor_pos_base = rpos_base.astype(np.float32).copy()
            self._prev_racket_pos_base = rpos_base.astype(np.float32)
        else:
            self._prev_racket_pos_base = np.zeros(3, dtype=np.float32)
        # Real joint feedback is sampled independently from the policy timer.
        # Keep a separate source-sample finite-difference state so repeated
        # policy ticks can hold the most recent encoder-derived velocity
        # without changing the simulator's per-policy-tick observation path.
        self._prev_racket_sample_pos_base: Optional[np.ndarray] = None
        self._prev_racket_sample_stamp_s: Optional[float] = None
        self._last_racket_sample_vel_base = np.zeros(3, dtype=np.float32)
        # Debug state: latest command trajectory for external monitoring.
        self.last_action = np.zeros(self.n_arm, dtype=np.float32)
        self.last_cmd_q = self.arm_cmd_q.copy()
        self.last_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.last_cmd_qacc = np.zeros(self.n_arm, dtype=np.float32)
        obs_dim = EXPECTED_OBS_DIM_WITH_DELAY if self.enable_delay_conditioning else EXPECTED_OBS_DIM
        self.last_obs = np.zeros(obs_dim, dtype=np.float32)
        self.last_obs_unscaled = np.zeros(obs_dim, dtype=np.float32)
        self.last_observation_scales = np.ones(obs_dim, dtype=np.float32)
        self.last_observation_scale_indices: tuple[int, ...] = ()

        # Delay conditioning state: command history for active_command_error
        if self.enable_delay_conditioning:
            if uses_corrected_67d_observation(self.observation_profile):
                # Corrected 67D actors use the fixed nominal delay rounded to
                # control ticks, plus the checkpoint's extra history slots.
                # control ticks.  Its buffer is 14 delay + 4 extra + current.
                self._delay_steps = max(
                    0, int(round(self.delay_conditioning_tau_s / self.dt)))
                self._command_buffer_len = max(
                    1,
                    self._delay_steps
                    + int(self.command_buffer_extra_steps)
                    + 1,
                )
                self._cmd_history = [
                    self.arm_cmd_q.copy()
                    for _ in range(self._command_buffer_len)
                ]
            else:
                # Preserve the existing legacy indexing and startup behaviour.
                self._delay_steps = max(
                    1, int(self.delay_conditioning_tau_s / self.dt) + 1)
                self._command_buffer_len = self._delay_steps + 10
                self._cmd_history = []

    # ------------------------------------------------------------------
    # Forward kinematics (MuJoCo, world -> base)
    # ------------------------------------------------------------------

    def _seed_non_right_arm_joints_to_target(self) -> None:
        """Set every named target joint (except right arm) to TARGET_DEGREES.

        This gives FK a posture that matches what the robot is commanded to
        hold, so the racket site position reflects training-like geometry.
        """
        if self.mj_model is None or self.mj_data is None:
            return
        right_arm_set = set(RIGHT_ARM_JOINTS)
        for name, deg in TARGET_DEGREES.items():
            if name in right_arm_set:
                continue
            jid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr] = float(np.deg2rad(deg))

    def _warm_fk_model_to_training_posture(self) -> None:
        """Reproduce simcode2's nominal 700-step XML-PD warm start.

        Only the resulting non-right-arm/base posture is retained for online
        FK.  Every observation call still overwrites the seven right-arm
        joints with current real/sim feedback before ``mj_forward``.
        """
        if self.mj_model is None or self.mj_data is None:
            return
        target_degrees = dict(TARGET_DEGREES)
        for name, q_rad in zip(RIGHT_ARM_JOINTS, self.initial_arm_q_rad):
            target_degrees[name] = float(np.rad2deg(q_rad))
        self.mj_data = mj.MjData(self.mj_model)
        self.mj_data.ctrl[:] = 0.0
        for name, deg in target_degrees.items():
            target = float(np.deg2rad(deg))
            aid = mj.mj_name2id(
                self.mj_model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                self.mj_data.ctrl[aid] = target
        for _ in range(700):
            mj.mj_step(self.mj_model, self.mj_data)
        mj.mj_forward(self.mj_model, self.mj_data)
        q_addresses = (
            self._base_x_qadr,
            self._base_y_qadr,
            self._base_yaw_qadr,
        )
        v_addresses = (
            self._base_x_vadr,
            self._base_y_vadr,
            self._base_yaw_vadr,
        )
        if min(*q_addresses, *v_addresses) >= 0:
            # The actor was trained on the XML's 700-step warm base state,
            # not an all-zero base vector.  Hardware has no equivalent
            # floating-base JointState, so retain this exact frozen reference
            # instead of injecting a distribution-shifting zero constant.
            self._base_obs_q = np.asarray([
                self.mj_data.qpos[address] for address in q_addresses
            ], dtype=np.float32)
            self._base_obs_dq = np.asarray([
                self.mj_data.qvel[address] for address in v_addresses
            ], dtype=np.float32)

    def _world_point_to_base(self, p_world: np.ndarray) -> np.ndarray:
        """Transform a world-frame point into base yaw frame (matching MJX).

        NOTE: This uses the base yaw frame convention from MJX juggle env,
        NOT the full base_link 6-DOF frame. Only yaw rotation is applied;
        x/y are translated, and z remains MuJoCo world height.
        """
        if self._base_body_id < 0:
            # Training XML kept base at origin for juggle; identity fallback.
            return np.asarray(p_world, dtype=np.float32)

        if min(
                self._base_x_qadr,
                self._base_y_qadr,
                self._base_yaw_qadr) >= 0:
            base_x = float(self.mj_data.qpos[self._base_x_qadr])
            base_y = float(self.mj_data.qpos[self._base_y_qadr])
            yaw = float(self.mj_data.qpos[self._base_yaw_qadr])
        else:
            base_pos = np.asarray(
                self.mj_data.xpos[self._base_body_id], dtype=np.float32)
            base_x = float(base_pos[0])
            base_y = float(base_pos[1])
            base_R = np.asarray(
                self.mj_data.xmat[self._base_body_id], dtype=np.float32
            ).reshape(3, 3)
            yaw = float(np.arctan2(base_R[1, 0], base_R[0, 0]))

        # Apply base yaw frame transform (MJX convention)
        dx = p_world[0] - base_x
        dy = p_world[1] - base_y
        c = np.cos(yaw)
        s = np.sin(yaw)

        return np.array([
            c * dx + s * dy,
            -s * dx + c * dy,
            p_world[2] + DEPLOYMENT_FK_WORLD_Z_CORRECTION_M,
        ], dtype=np.float32)

    def _world_rot_to_base_yaw(self, R_world: np.ndarray) -> np.ndarray:
        """Rotate a world-frame orientation into the same base-yaw frame as FK."""
        Rw = np.asarray(R_world, dtype=np.float32).reshape(3, 3)
        if self._base_body_id < 0:
            return Rw
        if self._base_yaw_qadr >= 0:
            yaw = float(self.mj_data.qpos[self._base_yaw_qadr])
        else:
            base_R = np.asarray(
                self.mj_data.xmat[self._base_body_id], dtype=np.float32
            ).reshape(3, 3)
            yaw = float(np.arctan2(base_R[1, 0], base_R[0, 0]))
        c = np.cos(yaw)
        s = np.sin(yaw)
        R_base_from_world_yaw = np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return R_base_from_world_yaw @ Rw

    @staticmethod
    def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to a normalized [x, y, z, w] quaternion."""
        m = np.asarray(R, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        q = np.asarray([x, y, z, w], dtype=np.float32)
        norm = float(np.linalg.norm(q))
        return q / norm if norm > 1e-9 else np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _compute_racket_pose_base(
        self, arm_q: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Run FK with `arm_q` on the right arm and return racket pos in base.

        Non-right-arm joints remain seeded to TARGET_DEGREES. Returns
        (racket_pos_base, ok) where ok is False if FK is unavailable.
        """
        if self.mj_model is None or self.mj_data is None:
            return np.zeros(3, dtype=np.float32), False
        for i, jid in enumerate(self._arm_jids):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr] = float(arm_q[i])
        mj.mj_forward(self.mj_model, self.mj_data)
        rpos_world = np.asarray(
            self.mj_data.site_xpos[self._racket_site_id], dtype=np.float32)
        return self._world_point_to_base(rpos_world), True

    def _compute_racket_pose_quat_base(
        self, arm_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Run FK and return racket position plus [x, y, z, w] orientation."""
        if self.mj_model is None or self.mj_data is None:
            return (
                np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                False,
            )
        for i, jid in enumerate(self._arm_jids):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr] = float(arm_q[i])
        mj.mj_forward(self.mj_model, self.mj_data)
        rpos_world = np.asarray(
            self.mj_data.site_xpos[self._racket_site_id], dtype=np.float32)
        R_world = np.asarray(
            self.mj_data.site_xmat[self._racket_site_id],
            dtype=np.float32).reshape(3, 3)
        R_base = self._world_rot_to_base_yaw(R_world)
        return (
            self._world_point_to_base(rpos_world),
            self._rotmat_to_quat_xyzw(R_base),
            True,
        )

    def _compute_racket_launch_frame_base(
        self, arm_q: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        float, float, bool,
    ]:
        """Return the exact simcode2 ``racket_launch`` geometry in base frame.

        The launch reset is defined from the ``racket_rubber_fore`` geom, not
        from a fixed world-z offset above ``right_ee_site``.  The tuple is
        ``(center, tangent_x, tangent_y, upward_normal, radius,
        half_thickness, ok)``.
        """
        fallback = (
            np.zeros(3, dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            0.08,
            0.002,
            False,
        )
        if (self.mj_model is None or self.mj_data is None
                or self._racket_geom_id < 0):
            return fallback
        q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self._arm_jids) or not np.all(np.isfinite(q)):
            return fallback
        for i, jid in enumerate(self._arm_jids):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr] = float(q[i])
        mj.mj_forward(self.mj_model, self.mj_data)
        geom_id = int(self._racket_geom_id)
        center_world = np.asarray(
            self.mj_data.geom_xpos[geom_id], dtype=np.float32)
        rotation_world = np.asarray(
            self.mj_data.geom_xmat[geom_id], dtype=np.float32,
        ).reshape(3, 3)
        rotation_base = self._world_rot_to_base_yaw(rotation_world)
        tangent_x = np.asarray(rotation_base[:, 0], dtype=np.float32)
        tangent_y = np.asarray(rotation_base[:, 1], dtype=np.float32)
        normal = np.asarray(rotation_base[:, 2], dtype=np.float32)
        if float(normal[2]) < 0.0:
            normal = -normal
        geom_size = np.asarray(
            self.mj_model.geom_size[geom_id], dtype=np.float64).reshape(-1)
        radius = float(geom_size[0])
        half_thickness = float(geom_size[1])
        values = np.concatenate([
            np.asarray(center_world, dtype=np.float64),
            np.asarray(rotation_base, dtype=np.float64).reshape(-1),
            np.asarray([radius, half_thickness], dtype=np.float64),
        ])
        if not np.all(np.isfinite(values)):
            return fallback
        return (
            self._world_point_to_base(center_world),
            tangent_x,
            tangent_y,
            normal,
            radius,
            half_thickness,
            True,
        )

    def _compute_racket_pose_jacobians_base(
        self, arm_q: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool,
    ]:
        """Return racket pose plus exact MuJoCo translational/angular Jacobians.

        Jacobian columns follow ``RIGHT_ARM_JOINTS`` and are expressed in the
        same base-yaw frame as :meth:`_compute_racket_pose_quat_base`.  This is
        substantially cheaper and better conditioned than seven finite-
        difference FK calls in the 200 Hz workspace safety filter.
        """
        if (not _MUJOCO_AVAILABLE or self.mj_model is None
                or self.mj_data is None):
            zeros3 = np.zeros(3, dtype=np.float32)
            zeros_j = np.zeros((3, len(RIGHT_ARM_JOINTS)), dtype=np.float64)
            return (
                zeros3,
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                zeros_j,
                zeros_j.copy(),
                False,
            )
        q = np.asarray(arm_q, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self._arm_jids) or not np.all(np.isfinite(q)):
            zeros3 = np.zeros(3, dtype=np.float32)
            zeros_j = np.zeros((3, len(RIGHT_ARM_JOINTS)), dtype=np.float64)
            return (
                zeros3,
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                zeros_j,
                zeros_j.copy(),
                False,
            )
        for i, jid in enumerate(self._arm_jids):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr] = float(q[i])
        mj.mj_forward(self.mj_model, self.mj_data)

        rpos_world = np.asarray(
            self.mj_data.site_xpos[self._racket_site_id], dtype=np.float32)
        rotation_world = np.asarray(
            self.mj_data.site_xmat[self._racket_site_id],
            dtype=np.float32,
        ).reshape(3, 3)
        if self._base_body_id < 0:
            rotation_base_from_world = np.eye(3, dtype=np.float64)
        else:
            base_rotation = np.asarray(
                self.mj_data.xmat[self._base_body_id],
                dtype=np.float64,
            ).reshape(3, 3)
            yaw = math.atan2(base_rotation[1, 0], base_rotation[0, 0])
            c = math.cos(yaw)
            s = math.sin(yaw)
            rotation_base_from_world = np.array([
                [c, s, 0.0],
                [-s, c, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

        jacobian_position_world = np.zeros(
            (3, int(self.mj_model.nv)), dtype=np.float64)
        jacobian_angular_world = np.zeros_like(jacobian_position_world)
        mj.mj_jacSite(
            self.mj_model,
            self.mj_data,
            jacobian_position_world,
            jacobian_angular_world,
            int(self._racket_site_id),
        )
        dof_addresses = np.asarray([
            int(self.mj_model.jnt_dofadr[jid]) for jid in self._arm_jids
        ], dtype=np.int64)
        jacobian_position_base = (
            rotation_base_from_world
            @ jacobian_position_world[:, dof_addresses])
        jacobian_angular_base = (
            rotation_base_from_world
            @ jacobian_angular_world[:, dof_addresses])
        rotation_base = rotation_base_from_world @ rotation_world
        return (
            self._world_point_to_base(rpos_world),
            self._rotmat_to_quat_xyzw(rotation_base),
            jacobian_position_base,
            jacobian_angular_base,
            True,
        )

    def current_arm_cmd_q(self) -> np.ndarray:
        """Return the latest 7-DOF right-arm command (rad)."""
        return self.arm_cmd_q.copy()

    # ------------------------------------------------------------------
    # Delay conditioning helpers
    # ------------------------------------------------------------------

    def _get_active_delayed_command(self) -> np.ndarray:
        """Return the command issued 72ms ago (simulating execution delay).

        Used for computing active_command_error in delay conditioning mode.
        Returns the current command if history is insufficient (startup phase).
        """
        if not self.enable_delay_conditioning:
            return self.arm_cmd_q.astype(np.float32)

        if uses_corrected_67d_observation(self.observation_profile):
            # External safety/reset code may reseed arm_cmd_q directly.  Fence
            # the nominal history at that pose instead of leaking commands from
            # the previously selected policy profile.
            if (not self._cmd_history
                    or not np.allclose(
                        self._cmd_history[-1], self.arm_cmd_q,
                        rtol=0.0, atol=1e-7)):
                self._cmd_history = [
                    self.arm_cmd_q.astype(np.float32).copy()
                    for _ in range(self._command_buffer_len)
                ]
            index = len(self._cmd_history) - 1 - self._delay_steps
            return self._cmd_history[max(0, index)].astype(np.float32)

        if len(self._cmd_history) < self._delay_steps:
            # Startup phase: not enough history yet
            return self.arm_cmd_q.astype(np.float32)
        else:
            # Return command from delay_steps ago
            return self._cmd_history[-self._delay_steps].astype(np.float32)

    def _estimate_contact_time(
        self, rel_z: float, ball_vz: float, racket_vz: float = 0.0,
        ball_obs_age_s: float = 0.0,
        max_time: float = 0.5, contact_height_offset: float = 0.0,
        gravity: float = 9.81, lost_ball_timeout_ms: float = 150.0
    ) -> float:
        """Estimate time until ball-racket contact (parabolic motion).

        This matches the training env's _estimate_contact_time_from_z_vz
        formula, which solves the parabolic motion equation considering
        gravity and relative velocity between ball and racket.

        Args:
            rel_z: Ball height minus racket height (m), should be positive
            ball_vz: Ball vertical velocity (m/s), negative when falling
            racket_vz: Racket vertical velocity (m/s)
            ball_obs_age_s: Ball observation age in seconds (for lost ball detection)
            max_time: Maximum contact time to return (s)
            contact_height_offset: Contact detection offset (m)
            gravity: Gravitational acceleration (m/s²)
            lost_ball_timeout_ms: Ball lost timeout in milliseconds

        Returns:
            Estimated contact time in seconds, clipped to [0, max_time].
            Returns max_time if no valid solution exists or ball is lost.
        """
        if not self.enable_delay_conditioning:
            return 0.0

        # Check if ball is lost (matching simulation Line 1532-1533)
        lost_timeout_s = max(0.0, float(lost_ball_timeout_ms)) * 1e-3
        if lost_timeout_s > 0.0 and ball_obs_age_s >= lost_timeout_s:
            return max_time

        # Use relative velocity: ball approaching racket
        vz_rel = ball_vz - racket_vz
        g = max(abs(float(gravity)), 1e-6)
        max_t = max(0.0, float(max_time))

        # Adjust height for contact offset
        h = rel_z - float(contact_height_offset)

        # Check validity bounds (matching training env)
        if (not np.isfinite(h) or not np.isfinite(vz_rel) or
            abs(vz_rel) > 50.0 or abs(h) > 10.0):
            return max_t

        # Solve parabolic equation: h = vz_rel*t + 0.5*g*t²
        # Rearranged: 0.5*g*t² + vz_rel*t - h = 0
        # Using quadratic formula with discriminant
        disc = vz_rel * vz_rel + 2.0 * g * h

        if disc < 0.0 or not np.isfinite(disc):
            return max_t

        root = np.sqrt(disc)
        # Two solutions from quadratic formula
        # Formula: t = (vz_rel ± sqrt(disc)) / g
        # NOTE: Sign matches simulation (mjx_juggle_env.py Line 1523-1524)
        t1 = (vz_rel + root) / g
        t2 = (vz_rel - root) / g

        # Select positive, finite solution(s)
        candidates = []
        if t1 >= 0.0 and np.isfinite(t1):
            candidates.append(t1)
        if t2 >= 0.0 and np.isfinite(t2):
            candidates.append(t2)

        if not candidates:
            return max_t

        # Return the smallest positive solution
        t = min(candidates)
        return float(np.clip(t, 0.0, max_t))

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        ball_obs_age_s: float,
        step_dt: float,
        arm_q: Optional[np.ndarray] = None,
        arm_dq: Optional[np.ndarray] = None,
        arm_sample_stamp_s: Optional[float] = None,
    ) -> np.ndarray:
        # Prefer real joint feedback when available; otherwise fall back to
        # the last commanded trajectory. The latter matches the pre-feedback
        # behavior and keeps the obs finite when /joint_states is missing.
        arm_q_valid = (
            arm_q is not None
            and np.asarray(arm_q).shape == (self.n_arm,)
            and np.all(np.isfinite(arm_q))
        )
        if arm_q_valid:
            q = np.asarray(arm_q, dtype=np.float32)
        else:
            q = self.arm_cmd_q.astype(np.float32)

        arm_dq_valid = (
            arm_dq is not None
            and np.asarray(arm_dq).shape == (self.n_arm,)
            and np.all(np.isfinite(arm_dq))
        )
        if arm_dq_valid:
            dq = np.asarray(arm_dq, dtype=np.float32)
        else:
            dq = self.arm_cmd_qvel.astype(np.float32)

        base_q = self._base_obs_q.astype(np.float32).copy()
        base_dq = self._base_obs_dq.astype(np.float32).copy()

        bpos_base = np.asarray(ball_pos_m, dtype=np.float32).reshape(3)
        bvel_base = np.asarray(ball_vel_m_s, dtype=np.float32).reshape(3)

        # Compute racket pose/velocity in the deployment base-yaw/world-z
        # convention. This mirrors JuggleEnv._point_to_base: x/y are transformed
        # by base yaw, while z remains MuJoCo world height. When FK is
        # unavailable we fall back to zeros.
        rpos_base, fk_ok = self._compute_racket_pose_base(q)
        rpos_base = np.asarray(rpos_base, dtype=np.float32)
        fk_ok = bool(
            fk_ok
            and rpos_base.shape == (3,)
            and np.all(np.isfinite(rpos_base))
        )
        sample_stamp = None
        if arm_sample_stamp_s is not None:
            try:
                sample_stamp = float(arm_sample_stamp_s)
            except (TypeError, ValueError):
                sample_stamp = None
            if sample_stamp is not None and not np.isfinite(sample_stamp):
                sample_stamp = None

        if fk_ok:
            use_sample_clock = bool(
                uses_corrected_67d_observation(self.observation_profile)
                and normalize_observation_profile(self.observation_profile)
                != OBSERVATION_PROFILE_SIMCODE4_SERVO67
                and arm_q_valid
                and sample_stamp is not None
            )
            if use_sample_clock:
                previous_stamp = self._prev_racket_sample_stamp_s
                previous_pos = self._prev_racket_sample_pos_base
                if previous_stamp is None or previous_pos is None:
                    # The reset/reference pose is not an encoder sample.  The
                    # first coherent hardware sample establishes the anchor.
                    rvel_base = np.zeros(3, dtype=np.float32)
                    self._last_racket_sample_vel_base = rvel_base.copy()
                    self._prev_racket_sample_pos_base = rpos_base.copy()
                    self._prev_racket_sample_stamp_s = sample_stamp
                elif (sample_stamp == previous_stamp
                      and np.array_equal(rpos_base, previous_pos)):
                    # A 200 Hz policy tick may run more than once for the same
                    # JointState. Hold the last source-sample velocity instead
                    # of emitting zero followed by a fixed-dt spike.
                    rvel_base = self._last_racket_sample_vel_base.copy()
                elif sample_stamp == previous_stamp:
                    # A producer that reuses a timestamp while changing q is
                    # not a coherent repeated sample. Do not carry the older
                    # position into the next derivative (which would create a
                    # one-tick spike); re-anchor and wait for a causal dt.
                    rvel_base = np.zeros(3, dtype=np.float32)
                    self._last_racket_sample_vel_base = rvel_base.copy()
                    self._prev_racket_sample_pos_base = rpos_base.copy()
                else:
                    sample_dt = sample_stamp - previous_stamp
                    if 1e-4 <= sample_dt <= 0.10:
                        candidate = (
                            (rpos_base - previous_pos) / sample_dt
                        ).astype(np.float32)
                        if np.all(np.isfinite(candidate)):
                            rvel_base = candidate
                        else:
                            rvel_base = np.zeros(3, dtype=np.float32)
                    else:
                        # Non-monotonic or stale source time cannot define a
                        # causal derivative. Re-anchor at this sample.
                        rvel_base = np.zeros(3, dtype=np.float32)
                    self._last_racket_sample_vel_base = rvel_base.copy()
                    self._prev_racket_sample_pos_base = rpos_base.copy()
                    self._prev_racket_sample_stamp_s = sample_stamp
                # Keep the policy-clock anchor coherent in case feedback is
                # intentionally disabled on a later tick.
                self._prev_racket_pos_base = rpos_base.copy()
            else:
                # Frozen training contract: successive policy states are
                # differenced using the fixed 5 ms environment step. This is
                # also the exact path used by the ideal-PD sim mirror, which
                # has no external JointState source timestamp.
                rvel_base = (rpos_base - self._prev_racket_pos_base) / max(
                    step_dt, 1e-6)
                self._prev_racket_pos_base = rpos_base.copy()
        else:
            rpos_base = np.zeros(3, dtype=np.float32)
            rvel_base = np.zeros(3, dtype=np.float32)

        # Replicate the simulation bug to align obs distribution with training.
        # Bug in mjx_juggle_env.py: _make_obs uses already-updated prev_racket_pos,
        # causing rvel ≈ 0. To verify policy performance when obs perfectly match
        # training distribution, we replicate this bug by swapping the update order.
        if self.force_obs_to_training_dist:
            # Store current for next step BEFORE computing velocity
            # This causes: rvel = (rpos - rpos) / dt ≈ 0
            self._prev_racket_pos_base = rpos_base.copy()
            # Now recompute with updated prev (bug replication)
            rvel_base = (rpos_base - self._prev_racket_pos_base) / max(
                step_dt, 1e-6)
            # Note: This will be very close to 0, matching simulation bug

        rel_base = bpos_base - rpos_base

        # arm_cmd_error = cmd - q. With real q this is the actual tracking
        # error; without it, cmd equals q and the error is zero.
        arm_cmd_error = (
            self.arm_cmd_q.astype(np.float32) - q
        ).astype(np.float32)

        age_norm = np.float32(
            min(max(0.0, ball_obs_age_s) / self.ball_obs_age_clip_s, 1.0))

        # Base observation (50 dimensions): obs[0-49]
        base_obs = np.concatenate([
            q, dq, base_q, base_dq,
            bpos_base, bvel_base,
            rpos_base, rvel_base, rel_base,
            self.prev_action, arm_cmd_error,
            np.array([age_norm], dtype=np.float32),
        ]).astype(np.float32)

        # Delay conditioning features (17 dimensions): obs[50-66]
        if self.enable_delay_conditioning:
            # obs[50]: reproduce MJX's float32 ``tau_ms * 1e-3`` before
            # dividing by the Python-float delay maximum.  At 72 ms the
            # frozen checkpoint sees 1.000000119..., not an exact 1.0.
            if uses_corrected_67d_observation(self.observation_profile):
                tau_act_episode = (
                    np.float32(self.delay_conditioning_tau_s * 1000.0)
                    * np.float32(1e-3)
                )
                tau_act_norm = np.float32(np.clip(
                    tau_act_episode / self.delay_conditioning_max_s,
                    0.0,
                    1.5,
                ))
            else:
                tau_act_norm = np.float32(1.0)

            # obs[51-57]: arm_cmd_qvel (already maintained by controller)
            arm_cmd_qvel_obs = self.arm_cmd_qvel.astype(np.float32)

            # obs[58-64]: active_command_error (delayed command error)
            active_cmd = self._get_active_delayed_command()
            active_command_error = (active_cmd - q).astype(np.float32)

            # obs[65]: causal estimated time to ball-racket contact.  Keep the
            # legacy rel_x/zero-age bug only for its original checkpoint.  The
            # simcode2 contract uses rel_z and advances the stale ball state by
            # its observation age before solving the same quadratic.
            ball_vz = float(bvel_base[2])
            racket_vz = float(rvel_base[2])
            if uses_corrected_67d_observation(self.observation_profile):
                age_s = max(0.0, float(ball_obs_age_s))
                gravity_z = -9.81
                rel_z_now = (
                    float(rel_base[2])
                    + ball_vz * age_s
                    + 0.5 * gravity_z * age_s * age_s
                )
                ball_vz_now = ball_vz + gravity_z * age_s
                t_contact_est = np.float32(self._estimate_contact_time(
                    rel_z_now,
                    ball_vz_now,
                    racket_vz,
                    ball_obs_age_s=age_s,
                    max_time=self.max_contact_time_s,
                    contact_height_offset=self.contact_height_offset_m,
                    gravity=abs(gravity_z),
                    lost_ball_timeout_ms=(
                        self.contact_lost_ball_timeout_s * 1000.0),
                ))
            else:
                rel_x = float(rel_base[0])
                t_contact_est = np.float32(self._estimate_contact_time(
                    rel_x,
                    ball_vz,
                    racket_vz,
                    ball_obs_age_s=0.0,
                    max_time=0.5,
                    contact_height_offset=0.0,
                    gravity=9.81,
                    lost_ball_timeout_ms=150.0,
                ))

            # obs[66]: t_margin (contact time margin)
            tau_act = self.delay_conditioning_tau_s
            t_margin = np.float32(t_contact_est - tau_act)

            # Concatenate base (50) + delay conditioning (17) = 67
            obs = np.concatenate([
                base_obs,
                np.array([tau_act_norm], dtype=np.float32),
                arm_cmd_qvel_obs,
                active_command_error,
                np.array([t_contact_est, t_margin], dtype=np.float32),
            ]).astype(np.float32)
        else:
            # Legacy mode: only 50-dim base observation
            obs = base_obs

        expected_dim = self._expected_obs_dim
        if obs.shape[0] != expected_dim:
            raise RuntimeError(
                f"Assembled observation has wrong size: {obs.shape[0]} "
                f"vs expected {expected_dim}")
        return obs

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        *,
        ball_valid: bool,
        ball_obs_age_s: float = 0.0,
        dt: Optional[float] = None,
        arm_q: Optional[np.ndarray] = None,
        arm_dq: Optional[np.ndarray] = None,
        arm_sample_stamp_s: Optional[float] = None,
        observation_scales: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Run one policy step and return the 7 right-arm joint targets (rad).

        If `ball_valid` is False the last known ball observation is reused
        (hold-last, matching the training dropout behavior). When no valid
        observation has ever been received, the ball state stays zero.
        `arm_q` / `arm_dq` are optional real joint feedback (rad / rad·s⁻¹);
        when absent or malformed, the internal command trajectory is used.
        """
        step_dt = float(dt) if dt is not None else self.dt
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            step_dt = self.dt

        # FIX: Update ball state BEFORE building observation to ensure
        # the first predict() call uses the actual ball state, not zeros.
        # Previously the update happened after _build_obs(), causing the
        # first observation to use initial zero values.
        obs_age = float(ball_obs_age_s)
        if not np.isfinite(obs_age) or obs_age < 0.0:
            obs_age = 0.0

        if ball_valid:
            self._last_valid_ball_pos_m = np.asarray(
                ball_pos_m, dtype=np.float32).reshape(3).copy()
            self._last_valid_ball_vel_m_s = np.asarray(
                ball_vel_m_s, dtype=np.float32).reshape(3).copy()
            self._has_valid_ball_obs = True
            # A valid estimator prediction can still be aged/stale. Preserve
            # that age so obs[49] matches the estimator/dropout contract.
            age_for_obs = obs_age if self.use_valid_ball_obs_age else 0.0
        else:
            age_for_obs = max(obs_age, step_dt)

        # Now build observation with the updated ball state
        obs = self._build_obs(
            self._last_valid_ball_pos_m,
            self._last_valid_ball_vel_m_s,
            age_for_obs,
            step_dt,
            arm_q=arm_q,
            arm_dq=arm_dq,
            arm_sample_stamp_s=arm_sample_stamp_s,
        )

        # Interactive probe hook: scale only the actor-facing observation.
        # Internal controller state and the physical ball measurement remain
        # untouched, so this is a controlled observation ablation rather than
        # a change to the estimator, safety checks, or command integration.
        unscaled_obs = obs.copy()
        scale_vector = np.ones(obs.shape[0], dtype=np.float32)
        if observation_scales is not None:
            candidate = np.asarray(
                observation_scales, dtype=np.float32).reshape(-1)
            if candidate.shape == scale_vector.shape:
                scale_vector = np.where(
                    np.isfinite(candidate), candidate, 1.0).astype(np.float32)
        scale_indices = tuple(np.flatnonzero(
            scale_vector != np.float32(1.0)).astype(int).tolist())
        if scale_indices:
            obs = obs.copy()
            obs *= scale_vector

        action = self._policy_action(obs)
        if not np.all(np.isfinite(action)):
            self._logger.warn(
                "RL policy returned non-finite action; holding last command")
            return self.arm_cmd_q.copy()
        action = np.clip(action, -1.0, 1.0)

        # Action-to-joint integration. Mirrors JuggleEnv.step:
        #   desired_qdd = action * acc_limit * acc_scale
        #   cmd_qvel   += desired_qdd * dt
        #   cmd_q      += cmd_qvel * dt    (clipped to joint range)
        # When arm_action_limiter is True (set per checkpoint), the mapped
        # acceleration and the integrated velocity are additionally clipped
        # to the arm limits, matching the training env's limiter branch.
        desired_qdd = action * self.arm_acc_limit_rad_s2 * self.action_acc_scale
        if self.arm_action_limiter:
            desired_qdd = np.clip(
                desired_qdd,
                -self.arm_acc_limit_rad_s2, self.arm_acc_limit_rad_s2)
        cmd_qvel = self.arm_cmd_qvel + desired_qdd * step_dt
        if self.arm_action_limiter:
            cmd_qvel = np.clip(
                cmd_qvel, -self.arm_vel_limit_rad_s, self.arm_vel_limit_rad_s)
        self.arm_cmd_q = np.clip(
            self.arm_cmd_q + cmd_qvel * step_dt,
            self.arm_q_low, self.arm_q_high).astype(np.float32)
        self.arm_cmd_qvel = cmd_qvel.astype(np.float32)
        self.prev_action = action.astype(np.float32)

        # Cache command history for delay conditioning (active_command_error)
        if self.enable_delay_conditioning:
            self._cmd_history.append(self.arm_cmd_q.copy())
            # simcode2 uses the exact 19-element nominal buffer.  The legacy
            # profile retains its historical delay_steps + 10 capacity.
            if len(self._cmd_history) > self._command_buffer_len:
                self._cmd_history.pop(0)

        # Cache latest command state for external debug/monitoring.
        self.last_action = action.copy()
        self.last_cmd_qacc = desired_qdd.copy()
        self.last_cmd_qvel = self.arm_cmd_qvel.copy()
        self.last_cmd_q = self.arm_cmd_q.copy()
        self.last_obs = obs.copy()
        self.last_obs_unscaled = unscaled_obs
        self.last_observation_scales = scale_vector
        self.last_observation_scale_indices = scale_indices

        return self.arm_cmd_q.copy()

    def latest_command_state(self) -> dict:
        """Return the most recent policy command trajectory for debug/monitoring.

        Returns a dict with keys: position, velocity, acceleration, action.
        All arrays are float32, shape (7,). Acceleration is the commanded
        joint acceleration (rad/s²) from the last predict() call.
        """
        return {
            "position": self.last_cmd_q.copy(),
            "velocity": self.last_cmd_qvel.copy(),
            "acceleration": self.last_cmd_qacc.copy(),
            "action": self.last_action.copy(),
        }

    def latest_obs_and_action(self) -> dict:
        """Return the most recent observation and action for logging/analysis.

        Returns a dict with keys: obs, action.
        obs is float32, shape (50,) or (67,) depending on delay conditioning.
        action is float32, shape (7,).
        """
        return {
            "obs": self.last_obs.copy(),
            "obs_unscaled": self.last_obs_unscaled.copy(),
            "action": self.last_action.copy(),
            "observation_scale_indices": list(
                self.last_observation_scale_indices),
            "observation_scales": self.last_observation_scales.copy(),
        }
