"""NumPy deployment reference for the GPU0-QVEL/REAL-RMP V85 model.

The adapter ends at the real robot's existing RMP position-target input.  It
must not run the simulator-side recovered RMP, XML PD, actuator dynamics, or
execution delay.  The released checkpoint is authenticated before unpickling,
validated fail-closed, and evaluated as a deterministic actor mean.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


CHECKPOINT_PATH = (
    "pingpong_controller/outputs/rl_sim/"
    "measured_qvel_rmp_vertical_v85_gpu0_seed20261004_20260828_"
    "stage23_failure_time_survival_online1/mjx_curriculum_best.pkl"
)
CHECKPOINT_SHA256 = (
    "0381dc68a8ea02b2b3db171cc7edcc45291ff1b510d944b5ae2cddd89505c93a"
)
PROFILE_NAME = "goal_d455_measured_qvel_rmp_vertical_v85"
CHECKPOINT_STAGE_NAME = "rmp85_complete_nonexecution_full_episode_polish"
CHECKPOINT_STAGE_INDEX = 24
CHECKPOINT_STAGE_UPDATE = 174
CHECKPOINT_GLOBAL_UPDATE = 430
CHECKPOINT_STEP = 2_922_905_600
JOINT_ORDER = tuple(f"RightArm-{index}" for index in range(7))
ACTOR_OBS_DIM = 57
CRITIC_OBS_DIM = 368
ACTION_DIM = 7
POLICY_DT_S = 0.005
BALL_OBS_RATE_HZ = 90.0
BALL_OBS_AGE_CLIP_S = 0.5
LOST_BALL_TIMEOUT_S = 0.350
REFERENCE_ERROR_HORIZON_S = np.asarray(
    [0.018, 0.019, 0.019, 0.018, 0.018, 0.018, 0.018],
    dtype=np.float32,
)
JOINT_VELOCITY_LIMIT_DEG_S = np.asarray(
    [210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0],
    dtype=np.float32,
)
JOINT_ACCELERATION_LIMIT_DEG_S2 = np.asarray(
    [1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0],
    dtype=np.float32,
)
JOINT_POSITION_LOW_RAD = np.asarray(
    [-2.0944, -2.96706, -3.05433, -0.174533, -3.05433, -1.65806, -1.5708],
    dtype=np.float32,
)
JOINT_POSITION_HIGH_RAD = np.asarray(
    [3.14159, 0.15708, 3.05433, 2.25147, 3.05433, 1.65806, 1.5708],
    dtype=np.float32,
)
RESET_JOINT_POSITION_DEG = np.asarray(
    [5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214],
    dtype=np.float32,
)
ZERO_OBSERVATION_ACTOR_MEAN = np.asarray(
    [
        -0.013018085,
        0.011029976,
        0.019584768,
        0.064141512,
        0.115628816,
        -0.122984409,
        0.017757997,
    ],
    dtype=np.float32,
)


class CheckpointContractError(ValueError):
    """Raised when checkpoint metadata does not identify the V85 contract."""


class StaleJointFeedbackError(RuntimeError):
    """Raised instead of integrating from missing or stale encoder feedback."""


class InvalidJointStateError(RuntimeError):
    """Raised before inference when measured joints violate the contract."""


class InvalidBallStateError(RuntimeError):
    """Raised instead of inferring from invalid represented state."""


def _require_equal(
    values: Mapping[str, object], name: str, expected: object
) -> None:
    actual = values.get(name)
    if actual != expected:
        raise CheckpointContractError(
            f"checkpoint {name}={actual!r}, expected {expected!r}"
        )


def _finite_vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-{size} vector")
    return array


def resolve_checkpoint_path(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> Path:
    """Resolve the released repository-relative checkpoint path."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        repository_root = Path(__file__).resolve().parents[3]
        path = repository_root / path
    return path.resolve()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest without loading an untrusted pickle."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_point_to_base(
    point_world_m: object,
    base_q_world: object,
) -> np.ndarray:
    """Match MJX's yaw-only world-to-base point transform.

    ``base_q_world`` is ``[world_x_m, world_y_m, world_yaw_rad]``.  The z
    component remains XML/world z; training does not subtract a base height.
    """

    point = _finite_vector(point_world_m, 3, "point_world_m")
    base_q = _finite_vector(base_q_world, 3, "base_q_world")
    dx = point[0] - base_q[0]
    dy = point[1] - base_q[1]
    cosine = np.float32(np.cos(base_q[2]))
    sine = np.float32(np.sin(base_q[2]))
    return np.asarray(
        [cosine * dx + sine * dy, -sine * dx + cosine * dy, point[2]],
        dtype=np.float32,
    )


def world_velocity_to_base(
    velocity_world_m_s: object,
    point_world_m: object,
    base_q_world: object,
    base_dq_world: object,
) -> np.ndarray:
    """Match MJX's moving-base velocity transform.

    ``base_dq_world`` is ``[world_vx_m_s, world_vy_m_s, yaw_rate_rad_s]``.
    """

    velocity = _finite_vector(
        velocity_world_m_s, 3, "velocity_world_m_s"
    )
    point = _finite_vector(point_world_m, 3, "point_world_m")
    base_q = _finite_vector(base_q_world, 3, "base_q_world")
    base_dq = _finite_vector(base_dq_world, 3, "base_dq_world")
    relative_x = point[0] - base_q[0]
    relative_y = point[1] - base_q[1]
    relative_vx = velocity[0] - base_dq[0] + base_dq[2] * relative_y
    relative_vy = velocity[1] - base_dq[1] - base_dq[2] * relative_x
    cosine = np.float32(np.cos(base_q[2]))
    sine = np.float32(np.sin(base_q[2]))
    return np.asarray(
        [
            cosine * relative_vx + sine * relative_vy,
            -sine * relative_vx + cosine * relative_vy,
            velocity[2],
        ],
        dtype=np.float32,
    )


def fractional_refresh_due(
    step_index: int,
    last_refresh_step: int,
    *,
    observation_rate_hz: float = BALL_OBS_RATE_HZ,
) -> bool:
    """Match the trainer's fractional camera scheduler on a 200 Hz clock."""

    rate_hz = float(observation_rate_hz)
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("observation_rate_hz must be positive and finite")
    previous_tick = math.floor(
        int(last_refresh_step) * rate_hz * POLICY_DT_S
    )
    current_tick = math.floor(int(step_index) * rate_hz * POLICY_DT_S)
    return current_tick > previous_tick


def represented_ball_age_s(
    control_time_s: float,
    represented_state_time_s: float,
) -> float:
    """Return the age of the state values actually supplied to the actor."""

    query_time_s = float(control_time_s)
    state_time_s = float(represented_state_time_s)
    if not np.isfinite(query_time_s) or not np.isfinite(state_time_s):
        raise InvalidBallStateError("ball-state timestamps must be finite")
    age_s = query_time_s - state_time_s
    if age_s < -1e-6:
        raise InvalidBallStateError(
            "represented ball state is from the future"
        )
    age_s = max(0.0, age_s)
    if age_s > LOST_BALL_TIMEOUT_S:
        raise InvalidBallStateError(
            "represented ball state exceeded the 350 ms validity timeout"
        )
    return age_s


@dataclass(frozen=True)
class HeldBallObservation:
    """One accepted 90 Hz ball state held on the 200 Hz actor clock."""

    position_base_m: np.ndarray
    velocity_base_m_s: np.ndarray
    represented_state_time_s: float
    last_refresh_step: int


class FractionalBallObservation90Hz:
    """Exact no-dropout fractional refresh/hold state used by V85."""

    @staticmethod
    def reset(
        step_index: int,
        position_base_m: object,
        velocity_base_m_s: object,
        represented_state_time_s: float,
    ) -> HeldBallObservation:
        position = _finite_vector(
            position_base_m, 3, "ball_position_base_m"
        )
        velocity = _finite_vector(
            velocity_base_m_s, 3, "ball_velocity_base_m_s"
        )
        timestamp = float(represented_state_time_s)
        if not np.isfinite(timestamp):
            raise InvalidBallStateError(
                "represented ball-state timestamp must be finite"
            )
        return HeldBallObservation(
            position_base_m=position.copy(),
            velocity_base_m_s=velocity.copy(),
            represented_state_time_s=timestamp,
            last_refresh_step=int(step_index),
        )

    @staticmethod
    def update(
        state: HeldBallObservation,
        step_index: int,
        position_base_m: object,
        velocity_base_m_s: object,
        represented_state_time_s: float,
    ) -> HeldBallObservation:
        """Accept the candidate only on the trainer's 90 Hz refresh tick."""

        step = int(step_index)
        if step < state.last_refresh_step:
            raise InvalidBallStateError("ball scheduler step moved backwards")
        if not fractional_refresh_due(step, state.last_refresh_step):
            return state
        return FractionalBallObservation90Hz.reset(
            step,
            position_base_m,
            velocity_base_m_s,
            represented_state_time_s,
        )


def _validate_plain_actor(payload: Mapping[str, object]) -> None:
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise CheckpointContractError("checkpoint actor params are missing")
    if params.get("teacher_pi") is not None:
        raise CheckpointContractError("V85 must contain one plain actor")
    actor = params.get("pi")
    if not isinstance(actor, Mapping):
        raise CheckpointContractError("checkpoint pi network is missing")
    expected_input = ACTOR_OBS_DIM
    for layer_name in ("l1", "l2", "out"):
        layer = actor.get(layer_name)
        if not isinstance(layer, Mapping):
            raise CheckpointContractError(
                f"checkpoint actor layer {layer_name!r} is missing"
            )
        weight = np.asarray(layer.get("w"))
        bias = np.asarray(layer.get("b"))
        if (
            weight.ndim != 2
            or bias.ndim != 1
            or weight.shape[0] != expected_input
            or weight.shape[1] != bias.shape[0]
        ):
            raise CheckpointContractError(
                f"checkpoint actor layer {layer_name!r} has invalid shape"
            )
        expected_input = int(bias.shape[0])
    if expected_input != ACTION_DIM:
        raise CheckpointContractError("checkpoint actor output is not 7-D")


def validate_gpu0_v85_checkpoint(payload: Mapping[str, object]) -> None:
    """Fail closed unless ``payload`` is the selected GPU0 V85 model."""

    _require_equal(payload, "obs_dim", ACTOR_OBS_DIM)
    _require_equal(payload, "critic_obs_dim", CRITIC_OBS_DIM)
    _require_equal(payload, "act_dim", ACTION_DIM)
    _require_equal(payload, "stage_name", CHECKPOINT_STAGE_NAME)
    _require_equal(payload, "stage_index", CHECKPOINT_STAGE_INDEX)
    _require_equal(payload, "stage_update", CHECKPOINT_STAGE_UPDATE)
    _require_equal(payload, "global_update", CHECKPOINT_GLOBAL_UPDATE)
    _require_equal(payload, "step", CHECKPOINT_STEP)
    _validate_plain_actor(payload)
    args = payload.get("args")
    if not isinstance(args, Mapping):
        raise CheckpointContractError("checkpoint args are missing")
    _require_equal(args, "curriculum_profile", PROFILE_NAME)
    _require_equal(args, "seed", 20261004)
    env_cfg = payload.get("env_cfg")
    if not isinstance(env_cfg, Mapping):
        raise CheckpointContractError("checkpoint env_cfg is missing")

    required = {
        "action_command_mode": "velocity",
        "action_velocity_scale": 1.0,
        "policy_integration_feedback_source": "measured",
        "recovered_rmp_motion_mode": True,
        "recovered_rmp_direct_qvel_target": True,
        "recovered_rmp_bounded_qvel_reference": True,
        "include_qvel_reference_error_obs": True,
        "enable_delay_conditioning": False,
        "high_latency_obs": False,
        "ball_obs_rate_hz": BALL_OBS_RATE_HZ,
        "ball_obs_fractional_rate": True,
        "ball_obs_age_tracks_stale": True,
        "ball_obs_age_clip": BALL_OBS_AGE_CLIP_S,
        "ball_obs_velocity_observer_mode": "raw",
        "lost_ball_timeout_ms": LOST_BALL_TIMEOUT_S * 1000.0,
        "actor_mask_previous_action": False,
        "actor_previous_action_scale": 1.0,
        "arm_action_limiter": True,
        "actuator_cmd_filter": False,
        "actuator_compensation_mode": "none",
        "arm_servo_target_tracking_planner": False,
        "right_arm_pd_profile": "recovered_rmp_rmpmd_v2",
        "recovered_rmp_period_s": 0.001,
        "frame_skip": 5,
        "action_filter_tau_ms": 0.0,
        "action_jerk_limit": 0.0,
        "enable_anti_windup": False,
        "dr_rmp_target_update_hold_probability": 0.08,
        "dr_rmp_target_update_hold_tail_probability": 0.022,
        "dr_rmp_target_update_hold_tail_steps_range": (2, 9),
    }
    for name, expected in required.items():
        _require_equal(env_cfg, name, expected)

    scalar_horizon = float(
        env_cfg.get("recovered_rmp_qvel_reference_error_horizon_s", np.nan)
    )
    if scalar_horizon != 0.0:
        raise CheckpointContractError(
            "V85 must use only the seven-value reference horizon"
        )
    horizon = np.asarray(
        env_cfg.get(
            "recovered_rmp_qvel_reference_error_horizon_s_per_joint", ()
        ),
        dtype=np.float32,
    )
    if horizon.shape != (ACTION_DIM,) or not np.array_equal(
        horizon, REFERENCE_ERROR_HORIZON_S
    ):
        raise CheckpointContractError(
            "V85 bounded-reference horizon does not match 18/19 ms contract"
        )
    for name, expected in (
        ("arm_vel_limit_deg_s", JOINT_VELOCITY_LIMIT_DEG_S),
        ("arm_acc_limit_deg_s2", JOINT_ACCELERATION_LIMIT_DEG_S2),
    ):
        actual = np.asarray(env_cfg.get(name, ()), dtype=np.float32)
        if actual.shape != (ACTION_DIM,) or not np.array_equal(
            actual, expected
        ):
            raise CheckpointContractError(
                f"checkpoint {name} does not match the released contract"
            )


def load_released_actor(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> Any:
    """Authenticate and load the selected V85 deterministic NumPy actor.

    Only the repository's trusted released file should be unpickled.  The
    returned ``NumpyMJXActor.mean_action`` implements the exact tanh MLP.
    """

    path = resolve_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"released GPU0 checkpoint not found: {path}")
    actual_digest = sha256_file(path)
    if actual_digest != CHECKPOINT_SHA256:
        raise CheckpointContractError(
            "released GPU0 checkpoint SHA-256 mismatch: "
            f"{actual_digest} != {CHECKPOINT_SHA256}"
        )
    from pingpong_controller.tools.rl_2real.mjx_policy_controller import (
        NumpyMJXActor,
        load_mjx_checkpoint,
    )

    payload = load_mjx_checkpoint(path)
    validate_gpu0_v85_checkpoint(payload)
    actor = NumpyMJXActor(payload["params"])
    if (actor.obs_dim, actor.act_dim) != (ACTOR_OBS_DIM, ACTION_DIM):
        raise CheckpointContractError(
            "released GPU0 actor dimensions changed after loading"
        )
    zero_mean = actor.mean_action(
        np.zeros(ACTOR_OBS_DIM, dtype=np.float32)
    )
    if not np.allclose(
        zero_mean,
        ZERO_OBSERVATION_ACTOR_MEAN,
        rtol=0.0,
        atol=2e-7,
    ):
        raise CheckpointContractError(
            "released GPU0 actor failed the zero-observation golden vector"
        )
    return actor


@dataclass(frozen=True)
class GPU0RealRmpState:
    """State that must persist across 200 Hz actor ticks."""

    q_ref_rad: np.ndarray
    previous_measured_dq_rad_s: np.ndarray
    previous_executed_action: np.ndarray


@dataclass(frozen=True)
class GPU0RealRmpStep:
    """One deterministic policy result and the next persistent state."""

    observation: np.ndarray
    executed_action: np.ndarray
    desired_dq_rad_s: np.ndarray
    rmp_input_q_rad: np.ndarray
    state: GPU0RealRmpState


class GPU0QvelRealRmpReference:
    """Exact 57-D actor and bounded-qref boundary used by GPU0 V85.

    ``actor_mean`` receives one 57-D float32 observation and returns one 7-D
    deterministic mean action.  ``rmp_input_q_rad`` is the position target to
    publish as ``MechUnitCmd.jnt_pos`` after the robot's independent safety
    checks.  It is not the output of the simulator-side recovered RMP.
    """

    def __init__(
        self,
        actor_mean: Callable[[np.ndarray], np.ndarray],
        *,
        joint_position_low_rad: np.ndarray | None = None,
        joint_position_high_rad: np.ndarray | None = None,
        joint_velocity_limit_deg_s: object = JOINT_VELOCITY_LIMIT_DEG_S,
        maximum_joint_feedback_age_s: float = 0.010,
    ) -> None:
        self.actor_mean = actor_mean
        low = (
            JOINT_POSITION_LOW_RAD
            if joint_position_low_rad is None
            else joint_position_low_rad
        )
        high = (
            JOINT_POSITION_HIGH_RAD
            if joint_position_high_rad is None
            else joint_position_high_rad
        )
        self.q_low = _finite_vector(
            low, ACTION_DIM, "joint_position_low_rad"
        )
        self.q_high = _finite_vector(
            high, ACTION_DIM, "joint_position_high_rad"
        )
        if np.any(self.q_low >= self.q_high):
            raise ValueError(
                "joint position low limits must be below high limits"
            )
        velocity_deg_s = _finite_vector(
            joint_velocity_limit_deg_s,
            ACTION_DIM,
            "joint_velocity_limit_deg_s",
        )
        if np.any(velocity_deg_s <= 0.0):
            raise ValueError("joint velocity limits must be positive")
        self.velocity_limit_rad_s = np.deg2rad(velocity_deg_s).astype(
            np.float32
        )
        self.maximum_joint_feedback_age_s = float(
            maximum_joint_feedback_age_s
        )
        if (
            not np.isfinite(self.maximum_joint_feedback_age_s)
            or self.maximum_joint_feedback_age_s < 0.0
        ):
            raise ValueError(
                "maximum_joint_feedback_age_s must be non-negative"
            )

    def reset(
        self, measured_q_rad: np.ndarray, measured_dq_rad_s: np.ndarray
    ) -> GPU0RealRmpState:
        """Reset qref and one-tick encoder history from a fresh measurement."""

        q = _finite_vector(measured_q_rad, ACTION_DIM, "measured_q_rad")
        dq = _finite_vector(
            measured_dq_rad_s, ACTION_DIM, "measured_dq_rad_s"
        )
        if np.any(q < self.q_low) or np.any(q > self.q_high):
            raise InvalidJointStateError(
                "measured_q_rad is outside configured joint limits"
            )
        return GPU0RealRmpState(
            q_ref_rad=q.copy(),
            previous_measured_dq_rad_s=dq.copy(),
            previous_executed_action=np.zeros(ACTION_DIM, dtype=np.float32),
        )

    @staticmethod
    def _build_observation(
        state: GPU0RealRmpState,
        *,
        measured_q_rad: np.ndarray,
        measured_dq_rad_s: np.ndarray,
        base_q: np.ndarray,
        base_dq: np.ndarray,
        ball_position_base_m: np.ndarray,
        ball_velocity_base_m_s: np.ndarray,
        racket_position_base_m: np.ndarray,
        racket_velocity_base_m_s: np.ndarray,
        ball_observation_age_s: float,
    ) -> np.ndarray:
        q = _finite_vector(measured_q_rad, ACTION_DIM, "measured_q_rad")
        dq = _finite_vector(
            measured_dq_rad_s, ACTION_DIM, "measured_dq_rad_s"
        )
        base_position = _finite_vector(base_q, 3, "base_q")
        base_velocity = _finite_vector(base_dq, 3, "base_dq")
        ball_position = _finite_vector(
            ball_position_base_m, 3, "ball_position_base_m"
        )
        ball_velocity = _finite_vector(
            ball_velocity_base_m_s, 3, "ball_velocity_base_m_s"
        )
        racket_position = _finite_vector(
            racket_position_base_m, 3, "racket_position_base_m"
        )
        racket_velocity = _finite_vector(
            racket_velocity_base_m_s, 3, "racket_velocity_base_m_s"
        )
        age_s = float(ball_observation_age_s)
        if not np.isfinite(age_s) or age_s < 0.0:
            raise ValueError("ball_observation_age_s must be non-negative")
        age_norm = np.float32(np.clip(age_s / BALL_OBS_AGE_CLIP_S, 0.0, 1.0))

        observation = np.concatenate(
            [
                q,
                dq,
                base_position,
                base_velocity,
                ball_position,
                ball_velocity,
                racket_position,
                racket_velocity,
                ball_position - racket_position,
                _finite_vector(
                    state.previous_executed_action,
                    ACTION_DIM,
                    "previous_executed_action",
                ),
                dq
                - _finite_vector(
                    state.previous_measured_dq_rad_s,
                    ACTION_DIM,
                    "previous_measured_dq_rad_s",
                ),
                np.asarray([age_norm], dtype=np.float32),
                _finite_vector(state.q_ref_rad, ACTION_DIM, "q_ref_rad") - q,
            ]
        ).astype(np.float32)
        if observation.shape != (ACTOR_OBS_DIM,):
            raise RuntimeError("assembled GPU0 V85 observation is not 57-D")
        return observation

    def step(
        self,
        state: GPU0RealRmpState,
        *,
        measured_q_rad: np.ndarray,
        measured_dq_rad_s: np.ndarray,
        joint_feedback_age_s: float,
        base_q: np.ndarray,
        base_dq: np.ndarray,
        ball_position_base_m: np.ndarray,
        ball_velocity_base_m_s: np.ndarray,
        racket_position_base_m: np.ndarray,
        racket_velocity_base_m_s: np.ndarray,
        control_time_s: float,
        ball_state_time_s: float,
    ) -> GPU0RealRmpStep:
        """Run one 5 ms actor tick without duplicating the robot's RMP."""

        feedback_age_s = float(joint_feedback_age_s)
        if (
            not np.isfinite(feedback_age_s)
            or feedback_age_s < 0.0
            or feedback_age_s > self.maximum_joint_feedback_age_s
        ):
            raise StaleJointFeedbackError(
                "fresh finite q/dq is required; hold or enter supervised "
                "safe state"
            )
        q = _finite_vector(measured_q_rad, ACTION_DIM, "measured_q_rad")
        dq = _finite_vector(
            measured_dq_rad_s, ACTION_DIM, "measured_dq_rad_s"
        )
        if np.any(q < self.q_low) or np.any(q > self.q_high):
            raise InvalidJointStateError(
                "measured_q_rad is outside configured joint limits"
            )
        ball_observation_age_s = represented_ball_age_s(
            control_time_s,
            ball_state_time_s,
        )
        observation = self._build_observation(
            state,
            measured_q_rad=q,
            measured_dq_rad_s=dq,
            base_q=base_q,
            base_dq=base_dq,
            ball_position_base_m=ball_position_base_m,
            ball_velocity_base_m_s=ball_velocity_base_m_s,
            racket_position_base_m=racket_position_base_m,
            racket_velocity_base_m_s=racket_velocity_base_m_s,
            ball_observation_age_s=ball_observation_age_s,
        )
        raw_action = _finite_vector(
            self.actor_mean(observation), ACTION_DIM, "actor mean action"
        )
        action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
        desired_dq = (action * self.velocity_limit_rad_s).astype(np.float32)

        integrated_q_ref = (
            _finite_vector(state.q_ref_rad, ACTION_DIM, "q_ref_rad")
            + desired_dq * np.float32(POLICY_DT_S)
        )
        maximum_error = (
            self.velocity_limit_rad_s * REFERENCE_ERROR_HORIZON_S
        )
        lower = np.maximum(self.q_low, q - maximum_error)
        upper = np.minimum(self.q_high, q + maximum_error)
        q_ref = np.clip(integrated_q_ref, lower, upper).astype(np.float32)
        next_state = GPU0RealRmpState(
            q_ref_rad=q_ref.copy(),
            previous_measured_dq_rad_s=dq.copy(),
            previous_executed_action=action.copy(),
        )
        return GPU0RealRmpStep(
            observation=observation,
            executed_action=action,
            desired_dq_rad_s=desired_dq,
            rmp_input_q_rad=q_ref.copy(),
            state=next_state,
        )
