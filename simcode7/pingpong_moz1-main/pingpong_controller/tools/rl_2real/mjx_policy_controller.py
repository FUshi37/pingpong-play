#!/usr/bin/env python3
"""NumPy runtime for deploying an MJX/JAX PPO checkpoint on the real robot.

This module intentionally does *not* run MJX, JAX, or batched simulation.  It
only loads the ``.pkl`` checkpoint produced by the MJX training scripts, runs
the actor MLP forward pass with NumPy, and mirrors the real-time command
integration used by the training environment.

The public ``MJXPolicyController.predict(...)`` signature matches
``pingpong_controller.rl_policy.RLPolicyController.predict(...)`` closely so it
can be wired into ``pingpong_node.py`` later with minimal glue.
"""

from __future__ import annotations

import argparse
from collections import namedtuple
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Allow running this file directly from tools/rl_2real without installing the
# package into the active Python environment.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import mujoco as mj
    _MUJOCO_AVAILABLE = True
except ImportError:  # pragma: no cover - import-time only
    mj = None
    _MUJOCO_AVAILABLE = False

from pingpong_controller.rl_policy import (  # noqa: E402
    DEFAULT_ACTION_ACC_SCALE,
    DEFAULT_ARM_ACC_LIMIT_DEG_S2,
    DEFAULT_ARM_Q_HIGH_RAD,
    DEFAULT_ARM_Q_LOW_RAD,
    DEFAULT_ARM_VEL_LIMIT_DEG_S,
    DEFAULT_BALL_OBS_AGE_CLIP_S,
    EXPECTED_OBS_DIM,
    RIGHT_ARM_JOINTS,
    TARGET_DEGREES,
    target_right_arm_q_rad,
)
from pingpong_controller.tools.rl_sim.delay_control import (  # noqa: E402
    compensate_q_ref,
    command_buffer_length,
    delay_bin_id,
    delay_steps_from_tau,
    estimate_contact_time,
    push_command_buffer,
    smooth_action,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "pingpong_controller"
    / "outputs"
    / "rl_sim"
    / "logs_mjx_curriculum_v7"
    / "18_stage4g_strong_contact_dr.pkl"
)
DEFAULT_ROBOT_XML = (
    REPO_ROOT / "pingpong_controller" / "models" / "moz1_pd.xml"
)

_OptimStateStub = namedtuple("_OptimStateStub", ["m", "v", "t"])


class _MJXCheckpointUnpickler(pickle.Unpickler):
    """Unpickle checkpoints without importing the training stack.

    MJX checkpoints include the Adam optimizer state.  That object is a
    ``NamedTuple`` class defined in ``train_juggle_mjx_ppo.py``.  Real-time
    inference does not need it, so mapping it to a local stub avoids importing
    JAX/training modules on the robot host.
    """

    def find_class(self, module: str, name: str):
        if module == "train_juggle_mjx_ppo" and name == "OptimState":
            return _OptimStateStub
        return super().find_class(module, name)


@dataclass
class _LoggerShim:
    """Small logger shim compatible with rclpy logger methods."""

    def info(self, msg: str) -> None:
        print(f"[MJXPolicy][INFO] {msg}")

    def warn(self, msg: str) -> None:
        print(f"[MJXPolicy][WARN] {msg}")

    def error(self, msg: str) -> None:
        print(f"[MJXPolicy][ERROR] {msg}")


def _to_numpy_tree(value):
    """Convert checkpoint arrays to NumPy without requiring JAX at runtime."""
    if isinstance(value, dict):
        return {key: _to_numpy_tree(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy_tree(val) for val in value)
    if hasattr(value, "__array__"):
        return np.asarray(value, dtype=np.float32)
    return value


def load_mjx_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load and minimally validate an MJX PPO checkpoint payload."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"MJX checkpoint not found: {path}")
    with path.open("rb") as f:
        payload = _MJXCheckpointUnpickler(f).load()
    if not isinstance(payload, dict) or "params" not in payload:
        raise RuntimeError(f"Checkpoint does not contain policy params: {path}")
    payload = dict(payload)
    payload["params"] = _to_numpy_tree(payload["params"])
    return payload


class NumpyMJXActor:
    """Actor MLP copied from ``train_juggle_mjx_ppo.py``.

    Network:
        tanh(obs @ l1.w + l1.b) -> tanh(... l2 ...) -> out

    The training code stores both policy and value networks.  The real robot
    only needs the policy mean; stochastic sampling/log_std is deliberately not
    used for deployment.
    """

    def __init__(self, params: dict[str, object]):
        if "pi" not in params:
            raise RuntimeError("MJX params missing actor network key 'pi'")
        self.pi = params["pi"]
        self._validate()

    def _validate(self) -> None:
        for layer in ("l1", "l2", "out"):
            if layer not in self.pi:
                raise RuntimeError(f"MJX actor params missing layer '{layer}'")
            if "w" not in self.pi[layer] or "b" not in self.pi[layer]:
                raise RuntimeError(f"MJX actor layer '{layer}' missing w/b")
        self.obs_dim = int(np.asarray(self.pi["l1"]["w"]).shape[0])
        self.act_dim = int(np.asarray(self.pi["out"]["b"]).shape[0])
        if self.obs_dim < EXPECTED_OBS_DIM:
            raise RuntimeError(
                f"MJX actor obs_dim={self.obs_dim}, expected at least {EXPECTED_OBS_DIM}"
            )
        if self.act_dim != len(RIGHT_ARM_JOINTS):
            raise RuntimeError(
                f"MJX actor act_dim={self.act_dim}, expected "
                f"{len(RIGHT_ARM_JOINTS)}"
            )

    @staticmethod
    def _linear(x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
        return x @ np.asarray(layer["w"], dtype=np.float32) + np.asarray(
            layer["b"], dtype=np.float32
        )

    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs, dtype=np.float32)
        single = obs_arr.ndim == 1
        if single:
            obs_arr = obs_arr[None, :]
        if obs_arr.ndim != 2 or obs_arr.shape[1] != self.obs_dim:
            raise ValueError(
                f"obs must have shape ({self.obs_dim},) or "
                f"(N, {self.obs_dim}), got {obs_arr.shape}"
            )
        x = np.tanh(self._linear(obs_arr, self.pi["l1"]))
        x = np.tanh(self._linear(x, self.pi["l2"]))
        mean = self._linear(x, self.pi["out"]).astype(np.float32)
        return mean[0] if single else mean


class MJXPolicyController:
    """Real-time right-arm command generator for MJX/JAX PPO policies.

    Inputs are in the same units as ``RLPolicyController``:
    - ball position: base_link frame, meters
    - ball velocity: base_link frame, meters/second
    - arm feedback: radians and radians/second

    Output is a 7-DOF right-arm joint target in radians.  This output should
    still pass through ``RightArmCommandSafetyLimiter`` before it reaches the
    robot.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        dt: float = 0.005,
        action_gain: float = 1.0,
        action_acc_scale: Optional[float] = None,
        action_scale_mult: float = 1.0,
        arm_action_limiter: Optional[bool] = None,
        arm_vel_limit_deg_s=None,
        arm_acc_limit_deg_s2=None,
        arm_q_low_rad=DEFAULT_ARM_Q_LOW_RAD,
        arm_q_high_rad=DEFAULT_ARM_Q_HIGH_RAD,
        ball_obs_age_clip_s: Optional[float] = None,
        robot_xml_path: Optional[str | Path] = None,
        require_fk: bool = False,
        action_latency_steps: Optional[float] = None,
        obs_latency_steps: Optional[float] = None,
        actuator_cmd_tau: Optional[float] = None,
        actuator_cmd_gain: Optional[float] = None,
        actuator_compensation_mode: Optional[str] = None,
        actuator_lead_compensation: Optional[bool] = None,
        actuator_lead_beta: Optional[float] = None,
        actuator_lead_delay_scale: Optional[float] = None,
        actuator_lead_tau_scale: Optional[float] = None,
        actuator_lead_max_delta_rad: Optional[float] = None,
        actuator_inverse_beta: Optional[float] = None,
        actuator_inverse_delay_scale: Optional[float] = None,
        actuator_inverse_tau_scale: Optional[float] = None,
        actuator_inverse_max_delta_rad: Optional[float] = None,
        actuator_mpc_beta: Optional[float] = None,
        actuator_mpc_delay_scale: Optional[float] = None,
        actuator_mpc_tau_scale: Optional[float] = None,
        actuator_mpc_horizon_steps: Optional[int] = None,
        actuator_mpc_tracking_weight: Optional[float] = None,
        actuator_mpc_nominal_weight: Optional[float] = None,
        actuator_mpc_delta_weight: Optional[float] = None,
        actuator_mpc_max_delta_rad: Optional[float] = None,
        actuator_mpc_command_dynamics_constraint: Optional[bool] = None,
        actuator_mpc_command_velocity_weight: Optional[float] = None,
        actuator_mpc_command_acceleration_weight: Optional[float] = None,
        actuator_mpc_command_velocity_scale: Optional[float] = None,
        actuator_mpc_command_acceleration_scale: Optional[float] = None,
        tau_act_ms: Optional[float] = None,
        logger: Optional[object] = None,
    ) -> None:
        self._logger = logger or _LoggerShim()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.payload = load_mjx_checkpoint(self.checkpoint_path)
        self.params = self.payload["params"]
        self.actor = NumpyMJXActor(self.params)
        self.env_cfg = dict(self.payload.get("env_cfg") or {})

        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        self.action_gain = float(action_gain)
        self.action_scale_mult = float(action_scale_mult)
        if not np.isfinite(self.action_gain) or self.action_gain < 0.0:
            raise ValueError(f"action_gain must be finite and >= 0, got {action_gain}")
        if not np.isfinite(self.action_scale_mult) or self.action_scale_mult <= 0.0:
            raise ValueError(
                f"action_scale_mult must be finite and > 0, got {action_scale_mult}"
            )

        if action_acc_scale is None:
            action_acc_scale = self.env_cfg.get(
                "action_acc_scale", DEFAULT_ACTION_ACC_SCALE
            )
        self.action_acc_scale = float(action_acc_scale)

        if arm_action_limiter is None:
            arm_action_limiter = bool(self.env_cfg.get("arm_action_limiter", False))
        self.arm_action_limiter = bool(arm_action_limiter)

        # This is a deployment contract, not another software plant model.
        # In simulation the fields configure the acceleration planner inside
        # the modeled drive, after delay/FOPDT and before position PD.  The real
        # drive already owns that layer, so predict() must continue returning a
        # position goal and must not add a second 72/74 ms delay/filter here.
        self.drive_target_tracking_planner = bool(
            self.env_cfg.get("arm_servo_target_tracking_planner", False)
        )
        self.drive_target_velocity_scale = float(
            self.env_cfg.get("arm_servo_target_velocity_scale", 1.0)
        )
        self.drive_target_acceleration_scale = float(
            self.env_cfg.get("arm_servo_target_acceleration_scale", 1.0)
        )
        if not (0.0 < self.drive_target_velocity_scale <= 1.0):
            raise ValueError(
                "arm_servo_target_velocity_scale must be in (0, 1]"
            )
        if not (0.0 < self.drive_target_acceleration_scale <= 1.0):
            raise ValueError(
                "arm_servo_target_acceleration_scale must be in (0, 1]"
            )

        if ball_obs_age_clip_s is None:
            ball_obs_age_clip_s = self.env_cfg.get(
                "ball_obs_age_clip", DEFAULT_BALL_OBS_AGE_CLIP_S
            )
        self.ball_obs_age_clip_s = max(1e-6, float(ball_obs_age_clip_s))

        self.base_obs_dim = EXPECTED_OBS_DIM
        self.delay_conditioning = bool(self.env_cfg.get("enable_delay_conditioning", False))
        inferred_high_latency = self.actor.obs_dim > self.base_obs_dim and not self.delay_conditioning
        self.high_latency_obs = bool(self.env_cfg.get("high_latency_obs", inferred_high_latency))
        self.high_latency_history_frames = (
            max(1, int(self.env_cfg.get("high_latency_history_frames", 3)))
            if self.high_latency_obs
            else 1
        )
        self.high_latency_prev_frames = max(0, self.high_latency_history_frames - 1)
        self.high_latency_prediction_time_clip = max(
            1e-6,
            float(self.env_cfg.get("high_latency_prediction_time_clip", 0.30)),
        )
        expected_actor_obs_dim = self.base_obs_dim
        if self.high_latency_obs:
            expected_actor_obs_dim += 16 + self.high_latency_prev_frames * (self.base_obs_dim + len(RIGHT_ARM_JOINTS))
        self.delay_extra_dim = self._delay_extra_dim()
        expected_actor_obs_dim += self.delay_extra_dim
        if int(self.actor.obs_dim) != int(expected_actor_obs_dim):
            raise RuntimeError(
                f"MJX actor obs_dim={self.actor.obs_dim}, expected {expected_actor_obs_dim} "
                f"from checkpoint high_latency_obs={self.high_latency_obs}, "
                f"delay_conditioning={self.delay_conditioning}"
            )

        def _range_mid(name: str, default):
            values = self.env_cfg.get(name, default)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                return 0.5 * (float(values[0]) + float(values[1]))
            return float(values)

        def _range_high(name: str, default):
            values = self.env_cfg.get(name, default)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                return max(float(values[0]), float(values[1]))
            return float(values)

        self.action_latency_steps = float(
            action_latency_steps
            if action_latency_steps is not None
            else _range_mid("dr_action_latency_steps_range", (0.0, 0.0))
        )
        self.obs_latency_steps = float(
            obs_latency_steps
            if obs_latency_steps is not None
            else _range_mid("dr_obs_latency_steps_range", (0.0, 0.0))
        )
        self.max_action_latency_steps = max(1.0, _range_high("dr_action_latency_steps_range", (1.0, 1.0)))
        self.max_obs_latency_steps = max(1.0, _range_high("dr_obs_latency_steps_range", (1.0, 1.0)))
        self.actuator_cmd_tau = float(
            actuator_cmd_tau
            if actuator_cmd_tau is not None
            else _range_mid("dr_actuator_cmd_tau_range", (0.0, 0.0))
        )
        self.actuator_cmd_gain = float(
            actuator_cmd_gain
            if actuator_cmd_gain is not None
            else _range_mid("dr_actuator_cmd_gain_range", (1.0, 1.0))
        )
        self.actuator_compensation_mode = str(
            actuator_compensation_mode
            if actuator_compensation_mode is not None
            else self.env_cfg.get("actuator_compensation_mode", "none")
        ).strip().lower().replace("-", "_")
        self.actuator_lead_compensation = bool(
            actuator_lead_compensation
            if actuator_lead_compensation is not None
            else self.env_cfg.get("actuator_lead_compensation", False)
        )
        if self.actuator_lead_compensation and self.actuator_compensation_mode in {"none", "off", "false", "0"}:
            self.actuator_compensation_mode = "lead"
        self.actuator_lead_beta = float(
            actuator_lead_beta
            if actuator_lead_beta is not None
            else self.env_cfg.get("actuator_lead_beta", 0.0)
        )
        self.actuator_lead_delay_scale = float(
            actuator_lead_delay_scale
            if actuator_lead_delay_scale is not None
            else self.env_cfg.get("actuator_lead_delay_scale", 1.0)
        )
        self.actuator_lead_tau_scale = float(
            actuator_lead_tau_scale
            if actuator_lead_tau_scale is not None
            else self.env_cfg.get("actuator_lead_tau_scale", 1.0)
        )
        self.actuator_lead_max_delta_rad = float(
            actuator_lead_max_delta_rad
            if actuator_lead_max_delta_rad is not None
            else self.env_cfg.get("actuator_lead_max_delta_rad", 0.0)
        )
        self.actuator_inverse_beta = float(
            actuator_inverse_beta
            if actuator_inverse_beta is not None
            else self.env_cfg.get("actuator_inverse_beta", 1.0)
        )
        self.actuator_inverse_delay_scale = float(
            actuator_inverse_delay_scale
            if actuator_inverse_delay_scale is not None
            else self.env_cfg.get("actuator_inverse_delay_scale", 1.0)
        )
        self.actuator_inverse_tau_scale = float(
            actuator_inverse_tau_scale
            if actuator_inverse_tau_scale is not None
            else self.env_cfg.get("actuator_inverse_tau_scale", 1.0)
        )
        self.actuator_inverse_max_delta_rad = float(
            actuator_inverse_max_delta_rad
            if actuator_inverse_max_delta_rad is not None
            else self.env_cfg.get("actuator_inverse_max_delta_rad", 0.0)
        )
        self.actuator_mpc_beta = float(
            actuator_mpc_beta
            if actuator_mpc_beta is not None
            else self.env_cfg.get("actuator_mpc_beta", 1.0)
        )
        self.actuator_mpc_delay_scale = float(
            actuator_mpc_delay_scale
            if actuator_mpc_delay_scale is not None
            else self.env_cfg.get("actuator_mpc_delay_scale", 1.0)
        )
        self.actuator_mpc_tau_scale = float(
            actuator_mpc_tau_scale
            if actuator_mpc_tau_scale is not None
            else self.env_cfg.get("actuator_mpc_tau_scale", 1.0)
        )
        self.actuator_mpc_horizon_steps = int(
            actuator_mpc_horizon_steps
            if actuator_mpc_horizon_steps is not None
            else self.env_cfg.get("actuator_mpc_horizon_steps", 4)
        )
        self.actuator_mpc_tracking_weight = float(
            actuator_mpc_tracking_weight
            if actuator_mpc_tracking_weight is not None
            else self.env_cfg.get("actuator_mpc_tracking_weight", 1.0)
        )
        self.actuator_mpc_nominal_weight = float(
            actuator_mpc_nominal_weight
            if actuator_mpc_nominal_weight is not None
            else self.env_cfg.get("actuator_mpc_nominal_weight", 0.25)
        )
        self.actuator_mpc_delta_weight = float(
            actuator_mpc_delta_weight
            if actuator_mpc_delta_weight is not None
            else self.env_cfg.get("actuator_mpc_delta_weight", 0.08)
        )
        self.actuator_mpc_max_delta_rad = float(
            actuator_mpc_max_delta_rad
            if actuator_mpc_max_delta_rad is not None
            else self.env_cfg.get("actuator_mpc_max_delta_rad", 0.0)
        )
        self.actuator_mpc_command_dynamics_constraint = bool(
            actuator_mpc_command_dynamics_constraint
            if actuator_mpc_command_dynamics_constraint is not None
            else self.env_cfg.get("actuator_mpc_command_dynamics_constraint", False)
        )
        self.actuator_mpc_command_velocity_weight = float(
            actuator_mpc_command_velocity_weight
            if actuator_mpc_command_velocity_weight is not None
            else self.env_cfg.get("actuator_mpc_command_velocity_weight", 0.0)
        )
        self.actuator_mpc_command_acceleration_weight = float(
            actuator_mpc_command_acceleration_weight
            if actuator_mpc_command_acceleration_weight is not None
            else self.env_cfg.get("actuator_mpc_command_acceleration_weight", 0.0)
        )
        self.actuator_mpc_command_velocity_scale = float(
            actuator_mpc_command_velocity_scale
            if actuator_mpc_command_velocity_scale is not None
            else self.env_cfg.get("actuator_mpc_command_velocity_scale", 1.0)
        )
        self.actuator_mpc_command_acceleration_scale = float(
            actuator_mpc_command_acceleration_scale
            if actuator_mpc_command_acceleration_scale is not None
            else self.env_cfg.get("actuator_mpc_command_acceleration_scale", 1.0)
        )
        if not (0.0 < self.actuator_mpc_command_velocity_scale <= 1.0):
            raise ValueError("actuator_mpc_command_velocity_scale must be in (0, 1]")
        if not (0.0 < self.actuator_mpc_command_acceleration_scale <= 1.0):
            raise ValueError("actuator_mpc_command_acceleration_scale must be in (0, 1]")
        self.delay_max_s = max(1e-6, float(self.env_cfg.get("delay_max_ms", 150.0)) * 1e-3)
        self.delay_min_s = max(0.0, float(self.env_cfg.get("delay_min_ms", 0.0)) * 1e-3)
        self.tau_act_s = (
            max(0.0, float(tau_act_ms) * 1e-3)
            if tau_act_ms is not None
            else self.delay_min_s
        )
        self.command_buffer_len = command_buffer_length(
            float(self.env_cfg.get("delay_max_ms", 150.0)),
            self.dt,
            int(self.env_cfg.get("command_buffer_extra_steps", 4)),
        )
        self._obs_history = np.zeros((self.high_latency_prev_frames, self.base_obs_dim), dtype=np.float32)
        self._action_history = np.zeros((self.high_latency_prev_frames, len(RIGHT_ARM_JOINTS)), dtype=np.float32)
        self._last_built_base_obs = np.zeros(self.base_obs_dim, dtype=np.float32)

        if arm_vel_limit_deg_s is None:
            arm_vel_limit_deg_s = self.env_cfg.get(
                "arm_vel_limit_deg_s", DEFAULT_ARM_VEL_LIMIT_DEG_S
            )
        if arm_acc_limit_deg_s2 is None:
            arm_acc_limit_deg_s2 = self.env_cfg.get(
                "arm_acc_limit_deg_s2", DEFAULT_ARM_ACC_LIMIT_DEG_S2
            )

        self.n_arm = len(RIGHT_ARM_JOINTS)
        self.arm_vel_limit_rad_s = np.deg2rad(
            np.asarray(arm_vel_limit_deg_s, dtype=np.float32)
        )
        self.arm_acc_limit_rad_s2 = np.deg2rad(
            np.asarray(arm_acc_limit_deg_s2, dtype=np.float32)
        )
        self.arm_q_low = np.asarray(arm_q_low_rad, dtype=np.float32)
        self.arm_q_high = np.asarray(arm_q_high_rad, dtype=np.float32)
        self._validate_limit_shapes()

        self.mj_model = None
        self.mj_data = None
        self._racket_site_id = -1
        self._arm_jids: list[int] = []
        self._base_body_id = -1
        self._prev_racket_pos_base = np.zeros(3, dtype=np.float32)
        self._init_mujoco_fk(robot_xml_path, require_fk=require_fk)

        self.reset()
        self._logger.info(
            "MJXPolicyController ready | "
            f"checkpoint={self.checkpoint_path.name}, "
            f"obs_dim={self.actor.obs_dim}, act_dim={self.actor.act_dim}, "
            f"dt={self.dt:.4f}s, action_acc_scale={self.action_acc_scale:.3g}, "
            f"action_gain={self.action_gain:.3g}, "
            f"arm_action_limiter={self.arm_action_limiter}, "
            f"delay_conditioning={self.delay_conditioning}, "
            f"comp_mode={self.actuator_compensation_mode}, "
            f"lead_comp={self.actuator_lead_compensation}, "
            f"lead_beta={self.actuator_lead_beta:.3g}, "
            f"lead_max_delta_deg={np.rad2deg(self.actuator_lead_max_delta_rad):.2f}, "
            f"inverse_beta={self.actuator_inverse_beta:.3g}, "
            f"inverse_max_delta_deg={np.rad2deg(self.actuator_inverse_max_delta_rad):.2f}, "
            f"mpc_beta={self.actuator_mpc_beta:.3g}, "
            f"mpc_horizon={self.actuator_mpc_horizon_steps}, "
            f"mpc_max_delta_deg={np.rad2deg(self.actuator_mpc_max_delta_rad):.2f}, "
            f"drive_target_planner={self.drive_target_tracking_planner}, "
            f"drive_vel_scale={self.drive_target_velocity_scale:.3g}, "
            f"drive_acc_scale={self.drive_target_acceleration_scale:.3g}"
        )

    def _delay_extra_dim(self) -> int:
        if not self.delay_conditioning:
            return 0
        n_arm = len(RIGHT_ARM_JOINTS)
        dim = 0
        dim += 1 if bool(self.env_cfg.get("include_tau_act_norm", False)) else 0
        dim += n_arm if bool(self.env_cfg.get("include_command_state", False)) else 0
        dim += n_arm if bool(self.env_cfg.get("include_active_command_error", False)) else 0
        dim += 2 if bool(self.env_cfg.get("include_phase_features", False)) else 0
        if bool(self.env_cfg.get("use_delay_embedding", False)):
            dim += max(0, int(self.env_cfg.get("delay_embedding_dim", 0)))
        return dim

    def _validate_limit_shapes(self) -> None:
        arrays = (
            self.arm_vel_limit_rad_s,
            self.arm_acc_limit_rad_s2,
            self.arm_q_low,
            self.arm_q_high,
        )
        if any(arr.shape != (self.n_arm,) for arr in arrays):
            raise ValueError("arm limit arrays must have shape (7,)")

    def _init_mujoco_fk(
        self, robot_xml_path: Optional[str | Path], *, require_fk: bool
    ) -> None:
        if robot_xml_path is None:
            if require_fk:
                raise RuntimeError("require_fk=True but robot_xml_path is None")
            self._logger.warn(
                "robot_xml_path not provided; racket FK disabled. "
                "Real deployment should provide the robot XML."
            )
            return
        if not _MUJOCO_AVAILABLE:
            msg = "mujoco Python package is not available; racket FK disabled"
            if require_fk:
                raise RuntimeError(msg)
            self._logger.warn(msg)
            return

        xml_path = Path(robot_xml_path).expanduser().resolve()
        if not xml_path.exists():
            msg = f"robot_xml_path does not exist: {xml_path}"
            if require_fk:
                raise FileNotFoundError(msg)
            self._logger.warn(f"{msg}; racket FK disabled")
            return

        try:
            self.mj_model = mj.MjModel.from_xml_path(str(xml_path))
            self.mj_data = mj.MjData(self.mj_model)
            self._racket_site_id = mj.mj_name2id(
                self.mj_model, mj.mjtObj.mjOBJ_SITE, "right_ee_site"
            )
            self._arm_jids = [
                mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, name)
                for name in RIGHT_ARM_JOINTS
            ]
            if self._racket_site_id < 0 or any(jid < 0 for jid in self._arm_jids):
                raise RuntimeError("XML missing right_ee_site or right-arm joints")

            self.arm_q_low = np.array(
                [self.mj_model.jnt_range[jid, 0] for jid in self._arm_jids],
                dtype=np.float32,
            )
            self.arm_q_high = np.array(
                [self.mj_model.jnt_range[jid, 1] for jid in self._arm_jids],
                dtype=np.float32,
            )

            base_bid = mj.mj_name2id(
                self.mj_model, mj.mjtObj.mjOBJ_BODY, "base_link"
            )
            if base_bid < 0:
                base_bid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_BODY, "base")
            self._base_body_id = int(base_bid)
            self._seed_non_right_arm_joints_to_target()
            self._logger.info(
                f"MuJoCo FK enabled (xml={xml_path.name}); "
                "right-arm joint limits loaded from XML"
            )
        except Exception:
            self.mj_model = None
            self.mj_data = None
            self._racket_site_id = -1
            self._arm_jids = []
            if require_fk:
                raise
            self._logger.warn("Failed to initialize MuJoCo FK; FK disabled")

    def _seed_non_right_arm_joints_to_target(self) -> None:
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
        if self._base_body_id < 0:
            return np.asarray(p_world, dtype=np.float32)
        base_pos = np.asarray(self.mj_data.xpos[self._base_body_id], dtype=np.float32)
        base_R = np.asarray(
            self.mj_data.xmat[self._base_body_id], dtype=np.float32
        ).reshape(3, 3)
        return (base_R.T @ (np.asarray(p_world, dtype=np.float32) - base_pos)).astype(
            np.float32
        )

    def _compute_racket_pose_base(
        self, arm_q: np.ndarray
    ) -> tuple[np.ndarray, bool]:
        if self.mj_model is None or self.mj_data is None:
            return np.zeros(3, dtype=np.float32), False
        for i, jid in enumerate(self._arm_jids):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr] = float(arm_q[i])
        mj.mj_forward(self.mj_model, self.mj_data)
        rpos_world = np.asarray(
            self.mj_data.site_xpos[self._racket_site_id], dtype=np.float32
        )
        return self._world_point_to_base(rpos_world), True

    def reset(self, arm_q: Optional[np.ndarray] = None) -> None:
        """Reset internal command state.

        ``arm_q`` can be the real right-arm feedback after the robot reaches
        the init pose.  If omitted, the controller resets to the training
        target posture.
        """
        if arm_q is None:
            q = target_right_arm_q_rad().astype(np.float32)
        else:
            q = np.asarray(arm_q, dtype=np.float32).reshape(-1)
            if q.shape != (self.n_arm,) or not np.all(np.isfinite(q)):
                raise ValueError("arm_q must be a finite length-7 array")
        self.arm_cmd_q = np.clip(q, self.arm_q_low, self.arm_q_high).astype(np.float32)
        self.arm_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.arm_q_ref_latest = self.arm_cmd_q.copy()
        self.arm_q_ref_active = self.arm_cmd_q.copy()
        self.arm_actuator_q_ref_latest = self.arm_cmd_q.copy()
        self.arm_actuator_q_ref_active = self.arm_cmd_q.copy()
        self.arm_actuator_q_ref_vel_latest = np.zeros(self.n_arm, dtype=np.float32)
        self.command_buffer = np.broadcast_to(
            self.arm_q_ref_latest.astype(np.float32),
            (self.command_buffer_len, self.n_arm),
        ).copy()
        self.actuator_command_buffer = np.broadcast_to(
            self.arm_actuator_q_ref_latest.astype(np.float32),
            (self.command_buffer_len, self.n_arm),
        ).copy()
        self.prev_action = np.zeros(self.n_arm, dtype=np.float32)
        self.anti_windup_scale = 1.0
        self.delay_steps = delay_steps_from_tau(self.tau_act_s, self.dt)
        self._last_valid_ball_pos_m = np.zeros(3, dtype=np.float32)
        self._last_valid_ball_vel_m_s = np.zeros(3, dtype=np.float32)
        self._has_valid_ball_obs = False

        if self.mj_model is not None and self.mj_data is not None:
            rpos_base, _ = self._compute_racket_pose_base(self.arm_cmd_q)
            self._prev_racket_pos_base = rpos_base.astype(np.float32)
        else:
            self._prev_racket_pos_base = np.zeros(3, dtype=np.float32)

        self.last_action = np.zeros(self.n_arm, dtype=np.float32)
        self.last_cmd_q = self.arm_actuator_q_ref_active.copy()
        self.last_q_cmd_nominal = self.arm_cmd_q.copy()
        self.last_q_ref_latest = self.arm_q_ref_latest.copy()
        self.last_q_ref_active = self.arm_q_ref_active.copy()
        self.last_q_actuator_ref_latest = self.arm_actuator_q_ref_latest.copy()
        self.last_q_actuator_ref_active = self.arm_actuator_q_ref_active.copy()
        self.last_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.last_cmd_qacc = np.zeros(self.n_arm, dtype=np.float32)
        self._obs_history = np.zeros((self.high_latency_prev_frames, self.base_obs_dim), dtype=np.float32)
        self._action_history = np.zeros((self.high_latency_prev_frames, self.n_arm), dtype=np.float32)
        self._last_built_base_obs = np.zeros(self.base_obs_dim, dtype=np.float32)
        self.last_obs = np.zeros(self.actor.obs_dim, dtype=np.float32)

    def current_arm_cmd_q(self) -> np.ndarray:
        return self.arm_actuator_q_ref_active.copy()

    def _build_obs(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        ball_obs_age_s: float,
        step_dt: float,
        arm_q: Optional[np.ndarray] = None,
        arm_dq: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if (
            arm_q is not None
            and np.asarray(arm_q).shape == (self.n_arm,)
            and np.all(np.isfinite(arm_q))
        ):
            q = np.asarray(arm_q, dtype=np.float32)
        else:
            q = self.arm_cmd_q.astype(np.float32)

        if (
            arm_dq is not None
            and np.asarray(arm_dq).shape == (self.n_arm,)
            and np.all(np.isfinite(arm_dq))
        ):
            dq = np.asarray(arm_dq, dtype=np.float32)
        else:
            dq = self.arm_cmd_qvel.astype(np.float32)

        base_q = np.zeros(3, dtype=np.float32)
        base_dq = np.zeros(3, dtype=np.float32)
        bpos_base = np.asarray(ball_pos_m, dtype=np.float32).reshape(3)
        bvel_base = np.asarray(ball_vel_m_s, dtype=np.float32).reshape(3)

        rpos_base, fk_ok = self._compute_racket_pose_base(q)
        if fk_ok:
            rvel_base = (rpos_base - self._prev_racket_pos_base) / max(
                step_dt, 1e-6
            )
            self._prev_racket_pos_base = rpos_base.copy()
        else:
            rpos_base = np.zeros(3, dtype=np.float32)
            rvel_base = np.zeros(3, dtype=np.float32)

        rel_base = bpos_base - rpos_base
        arm_cmd_error = (self.arm_cmd_q.astype(np.float32) - q).astype(np.float32)
        age_norm = np.float32(
            min(max(0.0, ball_obs_age_s) / self.ball_obs_age_clip_s, 1.0)
        )

        base_obs = np.concatenate(
            [
                q,
                dq,
                base_q,
                base_dq,
                bpos_base,
                bvel_base,
                rpos_base,
                rvel_base,
                rel_base,
                self.prev_action,
                arm_cmd_error,
                np.array([age_norm], dtype=np.float32),
            ]
        ).astype(np.float32)
        if base_obs.shape[0] != self.base_obs_dim:
            raise RuntimeError(
                f"Assembled base observation has wrong size: {base_obs.shape[0]} "
                f"vs expected {self.base_obs_dim}"
            )
        self._last_built_base_obs = base_obs.copy()
        obs = self._augment_obs(base_obs)
        if obs.shape[0] != self.actor.obs_dim:
            raise RuntimeError(
                f"Assembled observation has wrong size: {obs.shape[0]} "
                f"vs actor obs_dim {self.actor.obs_dim}"
            )
        return obs

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        if self.high_latency_obs:
            bpos = base_obs[20:23]
            bvel = base_obs[23:26]
            rpos = base_obs[26:29]
            age_seconds = float(base_obs[49]) * self.ball_obs_age_clip_s
            action_latency_sec = self.action_latency_steps * self.dt
            obs_latency_sec = self.obs_latency_steps * self.dt
            pred_time = action_latency_sec
            if bool(self.env_cfg.get("high_latency_prediction_include_obs_latency", True)):
                pred_time += obs_latency_sec
            if bool(self.env_cfg.get("high_latency_prediction_include_ball_age", True)):
                pred_time += age_seconds
            if bool(self.env_cfg.get("high_latency_prediction_include_actuator_tau", True)):
                pred_time += self.actuator_cmd_tau
            pred_time = float(np.clip(pred_time, 0.0, self.high_latency_prediction_time_clip))
            gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)
            pred_bpos = bpos + bvel * pred_time + 0.5 * gravity * pred_time**2
            pred_bvel = bvel + gravity * pred_time
            pred_rel = pred_bpos - rpos
            clip = self.high_latency_prediction_time_clip
            latency_features = np.array(
                [
                    self.action_latency_steps / self.max_action_latency_steps,
                    action_latency_sec / clip,
                    self.obs_latency_steps / self.max_obs_latency_steps,
                    obs_latency_sec / clip,
                    age_seconds / clip,
                    self.actuator_cmd_tau / clip,
                    self.actuator_cmd_gain - 1.0,
                ],
                dtype=np.float32,
            )
            obs = np.concatenate(
                [
                    base_obs,
                    pred_bpos.astype(np.float32),
                    pred_bvel.astype(np.float32),
                    pred_rel.astype(np.float32),
                    latency_features,
                    self._obs_history.reshape(-1),
                    self._action_history.reshape(-1),
                ]
            ).astype(np.float32)
        else:
            obs = base_obs.astype(np.float32)
        if self.delay_extra_dim > 0:
            obs = np.concatenate([obs, self._delay_conditioning_features(base_obs)]).astype(np.float32)
        return obs

    def _delay_conditioning_features(self, base_obs: np.ndarray) -> np.ndarray:
        if not self.delay_conditioning or self.delay_extra_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        parts: list[np.ndarray] = []
        tau_norm = np.array([np.clip(self.tau_act_s / self.delay_max_s, 0.0, 1.5)], dtype=np.float32)
        if bool(self.env_cfg.get("include_tau_act_norm", False)):
            parts.append(tau_norm)
        if bool(self.env_cfg.get("include_command_state", False)):
            parts.append(self.arm_cmd_qvel.astype(np.float32))
        if bool(self.env_cfg.get("include_active_command_error", False)):
            parts.append((self.arm_q_ref_active - base_obs[: self.n_arm]).astype(np.float32))
        if bool(self.env_cfg.get("include_phase_features", False)):
            age_seconds = float(base_obs[49]) * self.ball_obs_age_clip_s
            lost_timeout_s = max(0.0, float(self.env_cfg.get("lost_ball_timeout_ms", 150.0))) * 1e-3
            t_contact = estimate_contact_time(
                float(base_obs[34]),
                float(base_obs[25] - base_obs[31]),
                gravity=9.81,
                contact_height_offset=float(self.env_cfg.get("contact_height_offset", 0.0)),
                max_contact_time=float(self.env_cfg.get("max_contact_time", 0.50)),
                ball_lost=lost_timeout_s > 0.0 and age_seconds >= lost_timeout_s,
            )
            parts.append(np.array([t_contact, t_contact - self.tau_act_s], dtype=np.float32))
        if bool(self.env_cfg.get("use_delay_embedding", False)):
            dim = max(0, int(self.env_cfg.get("delay_embedding_dim", 0)))
            if dim > 0:
                idx = np.arange(dim, dtype=np.float32) + 1.0
                angles = tau_norm[0] * idx * np.pi
                emb = np.where((np.arange(dim) % 2) == 0, np.sin(angles), np.cos(angles))
                parts.append(emb.astype(np.float32))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def _push_history(self, base_obs: np.ndarray, action: np.ndarray) -> None:
        if not self.high_latency_obs or self.high_latency_prev_frames <= 0:
            return
        self._obs_history[:-1] = self._obs_history[1:]
        self._obs_history[-1] = np.asarray(base_obs, dtype=np.float32)
        self._action_history[:-1] = self._action_history[1:]
        self._action_history[-1] = np.asarray(action, dtype=np.float32)

    def predict(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        *,
        ball_valid: bool,
        ball_obs_age_s: float = 0.0,
        dt: Optional[float] = None,
        tau_act_s: Optional[float] = None,
        arm_q: Optional[np.ndarray] = None,
        arm_dq: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run one causal deterministic policy step and return 7 position goals.

        The calculation uses current feedback, retained command/history state,
        and forward predictions from the fixed actuator model.  It never reads
        a future measured joint state.  The physical drive performs the
        checkpoint-declared target trajectory planning; duplicating its FOPDT
        or planner here would add unmodeled delay on the robot.
        """
        step_dt = float(dt) if dt is not None else self.dt
        if not np.isfinite(step_dt) or step_dt <= 0.0:
            step_dt = self.dt
        if tau_act_s is not None:
            tau = float(tau_act_s)
            if np.isfinite(tau):
                self.tau_act_s = float(np.clip(tau, self.delay_min_s, self.delay_max_s))
        self.delay_steps = int(np.clip(delay_steps_from_tau(self.tau_act_s, step_dt), 0, self.command_buffer_len - 1))

        if ball_valid:
            self._last_valid_ball_pos_m = np.asarray(
                ball_pos_m, dtype=np.float32
            ).reshape(3).copy()
            self._last_valid_ball_vel_m_s = np.asarray(
                ball_vel_m_s, dtype=np.float32
            ).reshape(3).copy()
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
        self.last_obs = obs.copy()

        action = self.actor.mean_action(obs).astype(np.float32)
        action = action * np.float32(self.action_gain)
        if not np.all(np.isfinite(action)):
            self._logger.warn("MJX policy returned non-finite action; holding last")
            return self.arm_actuator_q_ref_active.copy()
        action = np.clip(action, -1.0, 1.0)

        q_for_aw = (
            np.asarray(arm_q, dtype=np.float32)
            if arm_q is not None and np.asarray(arm_q).shape == (self.n_arm,) and np.all(np.isfinite(arm_q))
            else self.arm_actuator_q_ref_active
        )
        action, self.anti_windup_scale = smooth_action(
            action,
            self.prev_action,
            dt=step_dt,
            action_acc_limit=float(self.env_cfg.get("action_acc_limit", 1.0)),
            action_filter_tau_ms=float(self.env_cfg.get("action_filter_tau_ms", 0.0)),
            action_jerk_limit=float(self.env_cfg.get("action_jerk_limit", 0.0)),
            e_active=self.arm_q_ref_active - q_for_aw,
            enable_anti_windup=bool(self.env_cfg.get("enable_anti_windup", False)),
            anti_windup_error_threshold=float(self.env_cfg.get("anti_windup_error_threshold", 0.5)),
            anti_windup_min_scale=float(self.env_cfg.get("anti_windup_min_scale", 0.2)),
        )

        desired_qdd_raw = (
            action
            * self.arm_acc_limit_rad_s2
            * np.float32(self.action_acc_scale)
            * np.float32(self.action_scale_mult)
        )
        if self.arm_action_limiter:
            desired_qdd = np.clip(
                desired_qdd_raw,
                -self.arm_acc_limit_rad_s2,
                self.arm_acc_limit_rad_s2,
            )
        else:
            desired_qdd = desired_qdd_raw

        raw_cmd_qvel = self.arm_cmd_qvel + desired_qdd * step_dt
        if self.arm_action_limiter:
            cmd_qvel = np.clip(
                raw_cmd_qvel,
                -self.arm_vel_limit_rad_s,
                self.arm_vel_limit_rad_s,
            )
        else:
            cmd_qvel = raw_cmd_qvel

        self.arm_cmd_q = np.clip(
            self.arm_cmd_q + cmd_qvel * step_dt,
            self.arm_q_low,
            self.arm_q_high,
        ).astype(np.float32)
        self.arm_cmd_qvel = cmd_qvel.astype(np.float32)
        self.arm_q_ref_latest = self.arm_cmd_q.copy()
        prev_actuator_q_ref = self.arm_actuator_q_ref_latest.copy()
        prev_actuator_q_ref_vel = self.arm_actuator_q_ref_vel_latest.copy()
        self.arm_actuator_q_ref_latest = compensate_q_ref(
            self.actuator_compensation_mode,
            self.arm_cmd_q,
            self.arm_cmd_qvel,
            desired_qdd,
            dt=step_dt,
            delay_steps=self.delay_steps,
            actuator_tau_s=self.actuator_cmd_tau,
            actuator_gain=self.actuator_cmd_gain,
            lead_beta=self.actuator_lead_beta,
            lead_delay_scale=self.actuator_lead_delay_scale,
            lead_tau_scale=self.actuator_lead_tau_scale,
            lead_max_delta_rad=self.actuator_lead_max_delta_rad,
            inverse_beta=self.actuator_inverse_beta,
            inverse_delay_scale=self.actuator_inverse_delay_scale,
            inverse_tau_scale=self.actuator_inverse_tau_scale,
            inverse_max_delta_rad=self.actuator_inverse_max_delta_rad,
            mpc_beta=self.actuator_mpc_beta,
            mpc_delay_scale=self.actuator_mpc_delay_scale,
            mpc_tau_scale=self.actuator_mpc_tau_scale,
            mpc_horizon_steps=self.actuator_mpc_horizon_steps,
            mpc_tracking_weight=self.actuator_mpc_tracking_weight,
            mpc_nominal_weight=self.actuator_mpc_nominal_weight,
            mpc_delta_weight=self.actuator_mpc_delta_weight,
            mpc_max_delta_rad=self.actuator_mpc_max_delta_rad,
            mpc_command_dynamics_constraint=self.actuator_mpc_command_dynamics_constraint,
            mpc_command_velocity_weight=self.actuator_mpc_command_velocity_weight,
            mpc_command_acceleration_weight=self.actuator_mpc_command_acceleration_weight,
            mpc_command_velocity_limit_rad_s=(
                self.arm_vel_limit_rad_s * np.float32(self.actuator_mpc_command_velocity_scale)
            ),
            mpc_command_acceleration_limit_rad_s2=(
                self.arm_acc_limit_rad_s2 * np.float32(self.actuator_mpc_command_acceleration_scale)
            ),
            mpc_previous_command_q=prev_actuator_q_ref,
            mpc_previous_command_qvel=prev_actuator_q_ref_vel,
            applied_q=q_for_aw,
            command_buffer=self.actuator_command_buffer,
            warm_q=target_right_arm_q_rad().astype(np.float32),
            q_low=self.arm_q_low,
            q_high=self.arm_q_high,
        )
        self.arm_actuator_q_ref_vel_latest = (
            (self.arm_actuator_q_ref_latest - prev_actuator_q_ref) / step_dt
        ).astype(np.float32)
        self.prev_action = action.astype(np.float32)
        if self.delay_conditioning:
            self.command_buffer, self.arm_q_ref_active = push_command_buffer(
                self.command_buffer,
                self.arm_q_ref_latest,
                self.delay_steps,
            )
            self.actuator_command_buffer, self.arm_actuator_q_ref_active = push_command_buffer(
                self.actuator_command_buffer,
                self.arm_actuator_q_ref_latest,
                self.delay_steps,
            )
        else:
            self.arm_q_ref_active = self.arm_q_ref_latest.copy()
            self.arm_actuator_q_ref_active = self.arm_actuator_q_ref_latest.copy()
            self.command_buffer, _ = push_command_buffer(self.command_buffer, self.arm_q_ref_latest, 0)
            self.actuator_command_buffer, _ = push_command_buffer(
                self.actuator_command_buffer,
                self.arm_actuator_q_ref_latest,
                0,
            )
        self._push_history(self._last_built_base_obs, action)

        self.last_action = action.copy()
        self.last_cmd_qacc = desired_qdd.copy()
        self.last_cmd_qvel = self.arm_cmd_qvel.copy()
        self.last_q_cmd_nominal = self.arm_cmd_q.copy()
        self.last_q_ref_latest = self.arm_q_ref_latest.copy()
        self.last_q_ref_active = self.arm_q_ref_active.copy()
        self.last_q_actuator_ref_latest = self.arm_actuator_q_ref_latest.copy()
        self.last_q_actuator_ref_active = self.arm_actuator_q_ref_active.copy()
        # The physical drive already owns the identified command delay and
        # actuator response. Keep delayed history for the checkpoint's
        # observation, but do not apply the simulated delay a second time to
        # a direct (no-compensation) hardware command.
        direct_physical_actuator = self.actuator_compensation_mode in {
            "none",
            "off",
            "false",
            "0",
        }
        deployment_q = (
            self.arm_actuator_q_ref_latest
            if direct_physical_actuator
            else self.arm_actuator_q_ref_active
        )
        self.last_cmd_q = deployment_q.copy()
        return deployment_q.copy()

    def latest_command_state(self) -> dict[str, np.ndarray]:
        return {
            "position": self.last_cmd_q.copy(),
            "velocity": self.last_cmd_qvel.copy(),
            "acceleration": self.last_cmd_qacc.copy(),
            "action": self.last_action.copy(),
            "obs": self.last_obs.copy(),
            "q_cmd_nominal": self.last_q_cmd_nominal.copy(),
            "q_ref_latest": self.last_q_ref_latest.copy(),
            "q_ref_active": self.last_q_ref_active.copy(),
            "q_actuator_ref_latest": self.last_q_actuator_ref_latest.copy(),
            "q_actuator_ref_active": self.last_q_actuator_ref_active.copy(),
            "dq_ref_latest": self.last_cmd_qvel.copy(),
            "actuator_lead_delta": (self.last_q_actuator_ref_latest - self.last_q_cmd_nominal).copy(),
            "tau_act_s": np.array(self.tau_act_s, dtype=np.float32),
            "delay_steps": np.array(self.delay_steps, dtype=np.float32),
            "delay_bin_id": np.array(
                delay_bin_id(
                    self.tau_act_s * 1000.0,
                    self.env_cfg.get("delay_bin_edges_ms", (0, 150)),
                ),
                dtype=np.float32,
            ),
            "anti_windup_scale": np.array(self.anti_windup_scale, dtype=np.float32),
        }

    def checkpoint_summary(self) -> dict[str, object]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "step": int(self.payload.get("step", -1)),
            "obs_dim": int(self.payload.get("obs_dim", self.actor.obs_dim)),
            "act_dim": int(self.payload.get("act_dim", self.actor.act_dim)),
            "xml": self.payload.get("xml"),
            "mjx_xml": self.payload.get("mjx_xml"),
            "action_acc_scale": self.action_acc_scale,
            "action_gain": self.action_gain,
            "action_scale_mult": self.action_scale_mult,
            "arm_action_limiter": self.arm_action_limiter,
            "drive_target_tracking_planner": self.drive_target_tracking_planner,
            "drive_target_velocity_scale": self.drive_target_velocity_scale,
            "drive_target_acceleration_scale": self.drive_target_acceleration_scale,
            "ball_obs_age_clip_s": self.ball_obs_age_clip_s,
            "high_latency_obs": self.high_latency_obs,
            "high_latency_history_frames": self.high_latency_history_frames,
            "delay_conditioning": self.delay_conditioning,
            "delay_extra_dim": self.delay_extra_dim,
            "tau_act_s": self.tau_act_s,
            "delay_steps": self.delay_steps,
            "action_latency_steps": self.action_latency_steps,
            "obs_latency_steps": self.obs_latency_steps,
            "actuator_cmd_tau": self.actuator_cmd_tau,
            "actuator_cmd_gain": self.actuator_cmd_gain,
            "actuator_compensation_mode": self.actuator_compensation_mode,
            "actuator_lead_compensation": self.actuator_lead_compensation,
            "actuator_lead_beta": self.actuator_lead_beta,
            "actuator_lead_delay_scale": self.actuator_lead_delay_scale,
            "actuator_lead_tau_scale": self.actuator_lead_tau_scale,
            "actuator_lead_max_delta_rad": self.actuator_lead_max_delta_rad,
            "actuator_inverse_beta": self.actuator_inverse_beta,
            "actuator_inverse_delay_scale": self.actuator_inverse_delay_scale,
            "actuator_inverse_tau_scale": self.actuator_inverse_tau_scale,
            "actuator_inverse_max_delta_rad": self.actuator_inverse_max_delta_rad,
            "actuator_mpc_beta": self.actuator_mpc_beta,
            "actuator_mpc_delay_scale": self.actuator_mpc_delay_scale,
            "actuator_mpc_tau_scale": self.actuator_mpc_tau_scale,
            "actuator_mpc_horizon_steps": self.actuator_mpc_horizon_steps,
            "actuator_mpc_tracking_weight": self.actuator_mpc_tracking_weight,
            "actuator_mpc_nominal_weight": self.actuator_mpc_nominal_weight,
            "actuator_mpc_delta_weight": self.actuator_mpc_delta_weight,
            "actuator_mpc_max_delta_rad": self.actuator_mpc_max_delta_rad,
            "actuator_mpc_command_dynamics_constraint": self.actuator_mpc_command_dynamics_constraint,
            "actuator_mpc_command_velocity_weight": self.actuator_mpc_command_velocity_weight,
            "actuator_mpc_command_acceleration_weight": self.actuator_mpc_command_acceleration_weight,
            "actuator_mpc_command_velocity_scale": self.actuator_mpc_command_velocity_scale,
            "actuator_mpc_command_acceleration_scale": self.actuator_mpc_command_acceleration_scale,
            "fk_enabled": self.mj_model is not None,
        }


def _parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected comma-separated x,y,z")
    try:
        return np.array([float(p) for p in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_vec7(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated values")
    try:
        return np.array([float(p) for p in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test a real-time NumPy MJX policy controller."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_ROBOT_XML)
    parser.add_argument("--require-fk", action="store_true")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--action-scale-mult", type=float, default=1.0)
    parser.add_argument("--action-latency-steps", type=float, default=None)
    parser.add_argument("--obs-latency-steps", type=float, default=None)
    parser.add_argument("--actuator-cmd-tau", type=float, default=None)
    parser.add_argument("--actuator-cmd-gain", type=float, default=None)
    parser.add_argument("--actuator-compensation-mode", choices=["none", "lead", "inverse_smith", "inverse_mpc"], default=None)
    parser.add_argument("--actuator-lead-compensation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--actuator-lead-beta", type=float, default=None)
    parser.add_argument("--actuator-lead-delay-scale", type=float, default=None)
    parser.add_argument("--actuator-lead-tau-scale", type=float, default=None)
    parser.add_argument("--actuator-lead-max-delta-rad", type=float, default=None)
    parser.add_argument("--actuator-lead-max-delta-deg", type=float, default=None)
    parser.add_argument("--actuator-inverse-beta", type=float, default=None)
    parser.add_argument("--actuator-inverse-delay-scale", type=float, default=None)
    parser.add_argument("--actuator-inverse-tau-scale", type=float, default=None)
    parser.add_argument("--actuator-inverse-max-delta-rad", type=float, default=None)
    parser.add_argument("--actuator-inverse-max-delta-deg", type=float, default=None)
    parser.add_argument("--actuator-mpc-beta", type=float, default=None)
    parser.add_argument("--actuator-mpc-delay-scale", type=float, default=None)
    parser.add_argument("--actuator-mpc-tau-scale", type=float, default=None)
    parser.add_argument("--actuator-mpc-horizon-steps", type=int, default=None)
    parser.add_argument("--actuator-mpc-tracking-weight", type=float, default=None)
    parser.add_argument("--actuator-mpc-nominal-weight", type=float, default=None)
    parser.add_argument("--actuator-mpc-delta-weight", type=float, default=None)
    parser.add_argument("--actuator-mpc-max-delta-rad", type=float, default=None)
    parser.add_argument("--actuator-mpc-max-delta-deg", type=float, default=None)
    parser.add_argument(
        "--actuator-mpc-command-dynamics-constraint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--actuator-mpc-command-velocity-weight", type=float, default=None)
    parser.add_argument("--actuator-mpc-command-acceleration-weight", type=float, default=None)
    parser.add_argument("--actuator-mpc-command-velocity-scale", type=float, default=None)
    parser.add_argument("--actuator-mpc-command-acceleration-scale", type=float, default=None)
    parser.add_argument(
        "--tau-act-ms",
        type=float,
        default=None,
        help="Measured execution delay for delay-conditioned checkpoints.",
    )
    parser.add_argument(
        "--ball-pos",
        type=_parse_vec3,
        default=np.array([0.35, 0.0, 1.05], dtype=np.float32),
        help="Ball position in base_link frame, meters: x,y,z",
    )
    parser.add_argument(
        "--ball-vel",
        type=_parse_vec3,
        default=np.array([0.0, 0.0, -0.3], dtype=np.float32),
        help="Ball velocity in base_link frame, meters/second: vx,vy,vz",
    )
    parser.add_argument("--ball-valid", action="store_true")
    parser.add_argument(
        "--arm-q-deg",
        type=_parse_vec7,
        default=None,
        help="Optional real right-arm feedback in degrees.",
    )
    parser.add_argument(
        "--arm-dq-deg-s",
        type=_parse_vec7,
        default=None,
        help="Optional real right-arm velocity feedback in degrees/second.",
    )
    parser.add_argument("--print-obs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_q = None if args.arm_q_deg is None else np.deg2rad(args.arm_q_deg)
    arm_dq = None if args.arm_dq_deg_s is None else np.deg2rad(args.arm_dq_deg_s)

    controller = MJXPolicyController(
        args.checkpoint,
        dt=args.dt,
        action_gain=args.action_gain,
        action_scale_mult=args.action_scale_mult,
        robot_xml_path=args.robot_xml,
        require_fk=args.require_fk,
        action_latency_steps=args.action_latency_steps,
        obs_latency_steps=args.obs_latency_steps,
        actuator_cmd_tau=args.actuator_cmd_tau,
        actuator_cmd_gain=args.actuator_cmd_gain,
        actuator_compensation_mode=args.actuator_compensation_mode,
        actuator_lead_compensation=args.actuator_lead_compensation,
        actuator_lead_beta=args.actuator_lead_beta,
        actuator_lead_delay_scale=args.actuator_lead_delay_scale,
        actuator_lead_tau_scale=args.actuator_lead_tau_scale,
        actuator_lead_max_delta_rad=(
            None
            if args.actuator_lead_max_delta_rad is None and args.actuator_lead_max_delta_deg is None
            else (
                args.actuator_lead_max_delta_rad
                if args.actuator_lead_max_delta_rad is not None
                else float(np.deg2rad(args.actuator_lead_max_delta_deg))
            )
        ),
        actuator_inverse_beta=args.actuator_inverse_beta,
        actuator_inverse_delay_scale=args.actuator_inverse_delay_scale,
        actuator_inverse_tau_scale=args.actuator_inverse_tau_scale,
        actuator_inverse_max_delta_rad=(
            None
            if args.actuator_inverse_max_delta_rad is None and args.actuator_inverse_max_delta_deg is None
            else (
                args.actuator_inverse_max_delta_rad
                if args.actuator_inverse_max_delta_rad is not None
                else float(np.deg2rad(args.actuator_inverse_max_delta_deg))
            )
        ),
        actuator_mpc_beta=args.actuator_mpc_beta,
        actuator_mpc_delay_scale=args.actuator_mpc_delay_scale,
        actuator_mpc_tau_scale=args.actuator_mpc_tau_scale,
        actuator_mpc_horizon_steps=args.actuator_mpc_horizon_steps,
        actuator_mpc_tracking_weight=args.actuator_mpc_tracking_weight,
        actuator_mpc_nominal_weight=args.actuator_mpc_nominal_weight,
        actuator_mpc_delta_weight=args.actuator_mpc_delta_weight,
        actuator_mpc_max_delta_rad=(
            None
            if args.actuator_mpc_max_delta_rad is None and args.actuator_mpc_max_delta_deg is None
            else (
                args.actuator_mpc_max_delta_rad
                if args.actuator_mpc_max_delta_rad is not None
                else float(np.deg2rad(args.actuator_mpc_max_delta_deg))
            )
        ),
        actuator_mpc_command_dynamics_constraint=args.actuator_mpc_command_dynamics_constraint,
        actuator_mpc_command_velocity_weight=args.actuator_mpc_command_velocity_weight,
        actuator_mpc_command_acceleration_weight=args.actuator_mpc_command_acceleration_weight,
        actuator_mpc_command_velocity_scale=args.actuator_mpc_command_velocity_scale,
        actuator_mpc_command_acceleration_scale=args.actuator_mpc_command_acceleration_scale,
        tau_act_ms=args.tau_act_ms,
    )
    print("[mjx_policy_controller] summary:")
    for key, value in controller.checkpoint_summary().items():
        print(f"  {key}: {value}")

    if arm_q is not None:
        controller.reset(arm_q=arm_q)

    for step in range(1, max(1, int(args.steps)) + 1):
        cmd = controller.predict(
            args.ball_pos,
            args.ball_vel,
            ball_valid=bool(args.ball_valid),
            ball_obs_age_s=0.0 if args.ball_valid else args.dt * step,
            dt=args.dt,
            tau_act_s=None if args.tau_act_ms is None else args.tau_act_ms * 1e-3,
            arm_q=arm_q,
            arm_dq=arm_dq,
        )
        state = controller.latest_command_state()
        print(
            f"[mjx_policy_controller] step={step} "
            f"cmd_deg={np.rad2deg(cmd).round(3).tolist()} "
            f"action={state['action'].round(4).tolist()} "
            f"qdd_deg_s2={np.rad2deg(state['acceleration']).round(1).tolist()}"
        )
        if args.print_obs:
            print(f"  obs={state['obs'].round(5).tolist()}")


if __name__ == "__main__":
    main()
