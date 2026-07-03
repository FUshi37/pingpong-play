"""RL policy controller for right-arm ping-pong juggling.

Wraps a stable-baselines3 PPO policy that was trained with
`rl_juggle_env_random.JuggleEnv`. Converts incoming ball observations
(position / velocity in base_link frame, meters) into 7 right-arm joint
angles (radians), reproducing the env's action-to-joint-position
integration so the deployed behavior matches training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

EXPECTED_OBS_DIM = 50  # Base observation dimension (legacy)
EXPECTED_OBS_DIM_WITH_DELAY = 67  # 50 base + 17 delay conditioning


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
        self.force_obs_to_training_dist = True
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
            f"delay_conditioning={self.enable_delay_conditioning}")

        # Optional MuJoCo FK for racket position/velocity.
        self.mj_model = None
        self.mj_data = None
        self._racket_site_id = -1
        self._arm_jids = []
        self._base_body_id = -1
        self._prev_racket_pos_base = np.zeros(3, dtype=np.float32)
        if robot_xml_path and _MUJOCO_AVAILABLE:
            try:
                xml_path = Path(robot_xml_path).resolve()
                if not xml_path.exists():
                    self._logger.warn(
                        f"robot_xml_path does not exist: {xml_path}; "
                        "racket FK disabled")
                else:
                    self.mj_model = mj.MjModel.from_xml_path(str(xml_path))
                    self.mj_data = mj.MjData(self.mj_model)
                    self._racket_site_id = mj.mj_name2id(
                        self.mj_model, mj.mjtObj.mjOBJ_SITE, "right_ee_site")
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
                        # Seed non-right-arm joints to the training TARGET_DEGREES
                        # so FK matches the posture the robot will actually hold.
                        self._seed_non_right_arm_joints_to_target()
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
        self.arm_cmd_q = target_right_arm_q_rad().astype(np.float32).copy()
        self.arm_cmd_q = np.clip(
            self.arm_cmd_q, self.arm_q_low, self.arm_q_high)
        self.arm_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.prev_action = np.zeros(self.n_arm, dtype=np.float32)
        self._last_ball_stamp: Optional[float] = None
        # Cached ball observation for dropout-like hold-last behavior.
        self._last_valid_ball_pos_m = np.zeros(3, dtype=np.float32)
        self._last_valid_ball_vel_m_s = np.zeros(3, dtype=np.float32)
        self._has_valid_ball_obs = False
        # Initialize prev racket pos (base frame) for velocity estimation.
        if self.mj_model is not None and self.mj_data is not None:
            rpos_base, _ = self._compute_racket_pose_base(self.arm_cmd_q)
            self._prev_racket_pos_base = rpos_base.astype(np.float32)
        else:
            self._prev_racket_pos_base = np.zeros(3, dtype=np.float32)
        # Debug state: latest command trajectory for external monitoring.
        self.last_action = np.zeros(self.n_arm, dtype=np.float32)
        self.last_cmd_q = self.arm_cmd_q.copy()
        self.last_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.last_cmd_qacc = np.zeros(self.n_arm, dtype=np.float32)
        obs_dim = EXPECTED_OBS_DIM_WITH_DELAY if self.enable_delay_conditioning else EXPECTED_OBS_DIM
        self.last_obs = np.zeros(obs_dim, dtype=np.float32)

        # Delay conditioning state: command history for active_command_error
        if self.enable_delay_conditioning:
            self._delay_steps = max(1, int(0.072 / self.dt) + 1)  # 72ms delay
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

    def _world_point_to_base(self, p_world: np.ndarray) -> np.ndarray:
        """Transform a world-frame point into base yaw frame (matching MJX).

        NOTE: This uses the base yaw frame convention from MJX juggle env,
        NOT the full base_link 6-DOF frame. Only yaw rotation is applied;
        x/y are translated, but z remains world height.
        """
        if self._base_body_id < 0:
            # Training XML kept base at origin for juggle; identity fallback.
            return np.asarray(p_world, dtype=np.float32)

        # Get base position (x, y, z in world)
        base_pos = np.asarray(
            self.mj_data.xpos[self._base_body_id], dtype=np.float32)

        # Extract yaw from base rotation matrix (matches MJX _point_to_base)
        base_R = np.asarray(
            self.mj_data.xmat[self._base_body_id], dtype=np.float32
        ).reshape(3, 3)
        # Yaw from rotation matrix: atan2(R[1,0], R[0,0])
        yaw = np.arctan2(base_R[1, 0], base_R[0, 0])

        # Apply base yaw frame transform (MJX convention)
        dx = p_world[0] - base_pos[0]
        dy = p_world[1] - base_pos[1]
        c = np.cos(yaw)
        s = np.sin(yaw)

        return np.array([
            c * dx + s * dy,
            -s * dx + c * dy,
            p_world[2]  # z is world height, not relative to base
        ], dtype=np.float32)

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
    ) -> np.ndarray:
        # Prefer real joint feedback when available; otherwise fall back to
        # the last commanded trajectory. The latter matches the pre-feedback
        # behavior and keeps the obs finite when /joint_states is missing.
        if (arm_q is not None
                and np.asarray(arm_q).shape == (self.n_arm,)
                and np.all(np.isfinite(arm_q))):
            q = np.asarray(arm_q, dtype=np.float32)
        else:
            q = self.arm_cmd_q.astype(np.float32)

        if (arm_dq is not None
                and np.asarray(arm_dq).shape == (self.n_arm,)
                and np.all(np.isfinite(arm_dq))):
            dq = np.asarray(arm_dq, dtype=np.float32)
        else:
            dq = self.arm_cmd_qvel.astype(np.float32)

        base_q = np.zeros(3, dtype=np.float32)
        base_dq = np.zeros(3, dtype=np.float32)

        bpos_base = np.asarray(ball_pos_m, dtype=np.float32).reshape(3)
        bvel_base = np.asarray(ball_vel_m_s, dtype=np.float32).reshape(3)

        # Compute racket pose/velocity in BASE frame via FK (matches
        # JuggleEnv._get_obs() which uses _world_point_to_base on the racket
        # site). When FK is unavailable we fall back to zeros.
        rpos_base, fk_ok = self._compute_racket_pose_base(q)
        if fk_ok:
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
            # obs[50]: tau_act_norm (fixed 1.0 for 72ms delay)
            tau_act_norm = np.float32(1.0)

            # obs[51-57]: arm_cmd_qvel (already maintained by controller)
            arm_cmd_qvel_obs = self.arm_cmd_qvel.astype(np.float32)

            # obs[58-64]: active_command_error (delayed command error)
            active_cmd = self._get_active_delayed_command()
            active_command_error = (active_cmd - q).astype(np.float32)

            # obs[65]: t_contact_est (estimated time to ball-racket contact)
            # IMPORTANT: Simulation has a bug (mjx_juggle_env.py Line 1506):
            # It uses base_obs[:, 32] which is rel_x, not rel_z!
            # base_obs[32:35] = (rel_x, rel_y, rel_z)
            # To match training, we must replicate this bug and use rel_x.
            rel_x = float(bpos_base[0] - rpos_base[0])  # BUG: should be rel_z!
            ball_vz = float(bvel_base[2])
            racket_vz = float(rvel_base[2])
            # Ball is always observed in replay mode (from successful juggling trajectory)
            # so ball_obs_age_s = 0. Validation trace confirms this (obs[49] = 0 throughout).
            t_contact_est = np.float32(self._estimate_contact_time(
                rel_x,  # BUG: using rel_x instead of rel_z to match simulation
                ball_vz, racket_vz,
                ball_obs_age_s=0.0,
                max_time=0.5, contact_height_offset=0.0, gravity=9.81,
                lost_ball_timeout_ms=150.0))

            # obs[66]: t_margin (contact time margin)
            tau_act = 0.072  # 72ms execution delay
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
        if ball_valid:
            self._last_valid_ball_pos_m = np.asarray(
                ball_pos_m, dtype=np.float32).reshape(3).copy()
            self._last_valid_ball_vel_m_s = np.asarray(
                ball_vel_m_s, dtype=np.float32).reshape(3).copy()
            self._has_valid_ball_obs = True
            age_for_obs = 0.0
        else:
            age_for_obs = max(ball_obs_age_s, step_dt)

        # Now build observation with the updated ball state
        obs = self._build_obs(
            self._last_valid_ball_pos_m,
            self._last_valid_ball_vel_m_s,
            age_for_obs,
            step_dt,
            arm_q=arm_q,
            arm_dq=arm_dq,
        )

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
            # Keep a reasonable buffer size (delay_steps + margin)
            if len(self._cmd_history) > self._delay_steps + 10:
                self._cmd_history.pop(0)

        # Cache latest command state for external debug/monitoring.
        self.last_action = action.copy()
        self.last_cmd_qacc = desired_qdd.copy()
        self.last_cmd_qvel = self.arm_cmd_qvel.copy()
        self.last_cmd_q = self.arm_cmd_q.copy()
        self.last_obs = obs.copy()

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
            "action": self.last_action.copy(),
        }
