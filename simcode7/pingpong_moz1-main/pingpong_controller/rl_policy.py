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

EXPECTED_OBS_DIM = 50


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
    ) -> None:
        self._logger = logger or _LoggerShim()
        self.dt = float(dt)
        self.action_acc_scale = float(action_acc_scale)
        self.deterministic = bool(deterministic)
        self.ball_obs_age_clip_s = max(1e-6, float(ball_obs_age_clip_s))

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

        # Load SB3 policy. Failures are re-raised so the caller can fall back.
        try:
            from stable_baselines3 import PPO  # type: ignore
        except Exception as exc:  # pragma: no cover - import-time only
            raise RuntimeError(
                f"stable-baselines3 is required to load the RL policy: {exc}"
            ) from exc

        self._logger.info(f"Loading PPO policy from: {model_path}")
        self.model = PPO.load(model_path, device=device)
        obs_space = getattr(self.model, "observation_space", None)
        obs_shape = getattr(obs_space, "shape", None)
        if obs_shape is None or tuple(obs_shape) != (EXPECTED_OBS_DIM,):
            raise RuntimeError(
                f"Loaded model observation shape {obs_shape} does not match "
                f"expected ({EXPECTED_OBS_DIM},)")
        act_space = getattr(self.model, "action_space", None)
        act_shape = getattr(act_space, "shape", None)
        if act_shape is None or tuple(act_shape) != (self.n_arm,):
            raise RuntimeError(
                f"Loaded model action shape {act_shape} does not match "
                f"expected ({self.n_arm},)")
        self._logger.info(
            f"RL policy loaded | obs_dim={EXPECTED_OBS_DIM}, "
            f"act_dim={self.n_arm}, dt={self.dt:.4f}s, device={device}")

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
        """Transform a world-frame point into base-link frame (FK-consistent)."""
        if self._base_body_id < 0:
            # Training XML kept base at origin for juggle; identity fallback.
            return np.asarray(p_world, dtype=np.float32)
        base_pos = np.asarray(
            self.mj_data.xpos[self._base_body_id], dtype=np.float32)
        base_R = np.asarray(
            self.mj_data.xmat[self._base_body_id], dtype=np.float32
        ).reshape(3, 3)
        return (base_R.T @ (np.asarray(p_world, dtype=np.float32) - base_pos)
                ).astype(np.float32)

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

        rel_base = bpos_base - rpos_base

        # arm_cmd_error = cmd - q. With real q this is the actual tracking
        # error; without it, cmd equals q and the error is zero.
        arm_cmd_error = (
            self.arm_cmd_q.astype(np.float32) - q
        ).astype(np.float32)

        age_norm = np.float32(
            min(max(0.0, ball_obs_age_s) / self.ball_obs_age_clip_s, 1.0))

        obs = np.concatenate([
            q, dq, base_q, base_dq,
            bpos_base, bvel_base,
            rpos_base, rvel_base, rel_base,
            self.prev_action, arm_cmd_error,
            np.array([age_norm], dtype=np.float32),
        ]).astype(np.float32)

        if obs.shape[0] != EXPECTED_OBS_DIM:
            raise RuntimeError(
                f"Assembled observation has wrong size: {obs.shape[0]} "
                f"vs expected {EXPECTED_OBS_DIM}")
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

        if ball_valid:
            self._last_valid_ball_pos_m = np.asarray(
                ball_pos_m, dtype=np.float32).reshape(3).copy()
            self._last_valid_ball_vel_m_s = np.asarray(
                ball_vel_m_s, dtype=np.float32).reshape(3).copy()
            self._has_valid_ball_obs = True
            age_for_obs = 0.0
        else:
            age_for_obs = max(ball_obs_age_s, step_dt)

        obs = self._build_obs(
            self._last_valid_ball_pos_m,
            self._last_valid_ball_vel_m_s,
            age_for_obs,
            step_dt,
            arm_q=arm_q,
            arm_dq=arm_dq,
        )

        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(self.n_arm)
        if not np.all(np.isfinite(action)):
            self._logger.warn(
                "RL policy returned non-finite action; holding last command")
            return self.arm_cmd_q.copy()
        action = np.clip(action, -1.0, 1.0)

        # Action-to-joint integration (mirrors JuggleEnv.step with
        # arm_action_limiter=False, action_scale_mult=1.0).
        desired_qdd = action * self.arm_acc_limit_rad_s2 * self.action_acc_scale
        cmd_qvel = self.arm_cmd_qvel + desired_qdd * step_dt
        self.arm_cmd_q = np.clip(
            self.arm_cmd_q + cmd_qvel * step_dt,
            self.arm_q_low, self.arm_q_high).astype(np.float32)
        self.arm_cmd_qvel = cmd_qvel.astype(np.float32)
        self.prev_action = action.astype(np.float32)

        # Cache latest command state for external debug/monitoring.
        self.last_action = action.copy()
        self.last_cmd_qacc = desired_qdd.copy()
        self.last_cmd_qvel = self.arm_cmd_qvel.copy()
        self.last_cmd_q = self.arm_cmd_q.copy()

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
