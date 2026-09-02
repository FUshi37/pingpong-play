"""Authenticated robot adapter for the selected GPU1 V153 QACC actor.

The selected model lineage is V153.  The fitted delayed second-order actuator
and ``sport_taskspace_fit_v1`` PD are training-side surrogates.  This adapter
keeps command history only for the 67-D actor observation and publishes the
current q-only strategy target; it does not run either simulator-side plant
component on the robot.  Ball-state age is derived from the represented
state's timestamp, never from raw-detection absence or predictor horizon.
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
    "selected_best_models_and_normal_reset_videos_20260901/gpu1_v153/"
    "gpu1_v153_best_step14841511936.pkl"
)
CHECKPOINT_SHA256 = (
    "a02b0421029e808b84ea0ee5cf615b3e65c6df3bc28c0f4766ad9953c0e85176"
)
PROFILE_NAME = (
    "goal_d455_sport_taskspace_record_new3_sim2real_fixed_base_ball4g_"
    "dual_domain_homotopy_v153"
)
CHECKPOINT_STAGE_NAME = "record_new3_sim2real_v153_b2_b3_energy_p025_60hz"
CHECKPOINT_STAGE_INDEX = 59
CHECKPOINT_STAGE_UPDATE = 441
CHECKPOINT_GLOBAL_UPDATE = 1_092
CHECKPOINT_STEP = 14_841_511_936
CHECKPOINT_SEED = 82_731
JOINT_ORDER = tuple(f"RightArm-{index}" for index in range(7))
ACTOR_OBS_DIM = 67
CRITIC_OBS_DIM = 279
ACTION_DIM = 7
POLICY_DT_S = 0.005
BALL_OBS_RATE_HZ = 60.0
BALL_OBS_AGE_CLIP_S = 0.5
OBSERVED_COMMAND_DELAY_S = 0.045
LOST_BALL_TIMEOUT_S = 0.350
MAX_CONTACT_TIME_S = 0.5
POST_HIT_BALL_LATENCY_STEPS_RANGE = (0, 2)
NOMINAL_ACTUATOR_DELAY_MS = np.asarray(
    [45.0, 50.0, 45.0, 40.0, 35.0, 45.0, 50.0],
    dtype=np.float32,
)
NOMINAL_ACTUATOR_WN_RAD_S = np.asarray(
    [
        21.8911379437,
        22.0895753812,
        22.6504705,
        21.730533,
        20.1549483562,
        22.2451380938,
        23.0546718187,
    ],
    dtype=np.float64,
)
NOMINAL_ACTUATOR_ZETA = np.asarray(
    [
        0.3330028,
        0.3295521,
        0.3111642,
        0.3114,
        0.3131064,
        0.3112326,
        0.28553475,
    ],
    dtype=np.float64,
)
ZERO_OBSERVATION_ACTOR_MEAN = np.asarray(
    [
        0.264987797,
        0.095902964,
        -0.141679987,
        0.130324334,
        -0.013870809,
        -0.059728369,
        -0.005993952,
    ],
    dtype=np.float32,
)


class CheckpointContractError(ValueError):
    """Raised when checkpoint metadata does not identify the V153 contract."""


class StaleJointFeedbackError(RuntimeError):
    """Raised instead of publishing from missing or stale encoder feedback."""


class InvalidBallStateError(RuntimeError):
    """Raised instead of inferring from invalid timestamped ball state."""


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
    """Resolve the repository-relative V153 checkpoint path."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        repository_root = Path(__file__).resolve().parents[3]
        path = repository_root / path
    return path.resolve()


def sha256_file(path: str | Path) -> str:
    """Return a file digest without unpickling the checkpoint."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    confirmed_hit_count: int,
) -> float:
    """Return V153 actor age while retaining its pre-hit zero-age prefix."""

    query_time_s = float(control_time_s)
    state_time_s = float(represented_state_time_s)
    if not np.isfinite(query_time_s) or not np.isfinite(state_time_s):
        raise InvalidBallStateError("ball-state timestamps must be finite")
    actual_age_s = query_time_s - state_time_s
    if actual_age_s < -1e-6:
        raise InvalidBallStateError(
            "represented ball state is from the future"
        )
    actual_age_s = max(0.0, actual_age_s)
    if actual_age_s > LOST_BALL_TIMEOUT_S:
        raise InvalidBallStateError(
            "represented ball state exceeded the 350 ms validity timeout"
        )
    hit_count = int(confirmed_hit_count)
    if hit_count < 0:
        raise ValueError("confirmed_hit_count must be non-negative")
    return actual_age_s if hit_count >= 1 else 0.0


def _validate_plain_actor(payload: Mapping[str, object]) -> None:
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise CheckpointContractError("checkpoint actor params are missing")
    if params.get("teacher_pi") is not None:
        raise CheckpointContractError("V153 must contain one plain actor")
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


def validate_gpu1_v153_checkpoint(payload: Mapping[str, object]) -> None:
    """Fail closed unless ``payload`` is the selected GPU1 V153 model."""

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
    _require_equal(args, "seed", CHECKPOINT_SEED)
    env_cfg = payload.get("env_cfg")
    if not isinstance(env_cfg, Mapping):
        raise CheckpointContractError("checkpoint env_cfg is missing")

    required = {
        "action_command_mode": "acceleration",
        "action_acc_scale": 1.0,
        "policy_integration_feedback_source": "command",
        "recovered_rmp_motion_mode": False,
        "include_qvel_reference_error_obs": False,
        "enable_delay_conditioning": True,
        "high_latency_obs": False,
        "include_tau_act_norm": True,
        "include_command_state": True,
        "include_active_command_error": True,
        "include_phase_features": True,
        "use_delay_embedding": False,
        "ball_obs_rate_hz": BALL_OBS_RATE_HZ,
        "ball_obs_fractional_rate": True,
        "ball_obs_age_tracks_stale": False,
        "ball_obs_age_clip": BALL_OBS_AGE_CLIP_S,
        "ball_obs_velocity_observer_mode": "raw",
        "ball_obs_camera_missing_prob": 0.0,
        "ball_obs_dropout_prob": 0.0,
        "ball_obs_dropout_burst_prob": 0.0,
        "dr_post_hit_ball_obs_latency": True,
        "dr_post_hit_ball_obs_latency_probability": 1.0,
        "dr_post_hit_ball_obs_latency_min_confirmed_hits": 1,
        "dr_post_hit_ball_obs_latency_timestamp_consistent_age": True,
        "arm_action_limiter": True,
        "actuator_cmd_filter": True,
        "actuator_cmd_model": "second_order",
        "actuator_compensation_mode": "none",
        "arm_servo_target_tracking_planner": False,
        "right_arm_pd_profile": "sport_taskspace_fit_v1",
        "action_filter_tau_ms": 0.0,
        "action_jerk_limit": 0.0,
        "enable_anti_windup": False,
        "delay_min_ms": 45.0,
        "delay_max_ms": 45.0,
        "simulation_base_mode": "aligned_fixed",
        "ball_reset_mode": "falling_contact",
        "dr_ball_normalized_inertia_range": (0.4, 0.4),
    }
    for name, expected in required.items():
        _require_equal(env_cfg, name, expected)

    if (
        tuple(env_cfg.get("dr_obs_latency_steps_range", ()))
        != POST_HIT_BALL_LATENCY_STEPS_RANGE
    ):
        raise CheckpointContractError(
            "V153 post-hit ball-latency support must remain 0--2 ticks"
        )

    actuator_delay_ms = np.asarray(
        env_cfg.get("actuator_cmd_delay_ms_per_joint", ()),
        dtype=np.float32,
    )
    if actuator_delay_ms.shape != (ACTION_DIM,) or not np.array_equal(
        actuator_delay_ms, NOMINAL_ACTUATOR_DELAY_MS
    ):
        raise CheckpointContractError(
            "V153 nominal per-joint actuator delay does not match contract"
        )

    for name, expected in (
        ("actuator_cmd_natural_frequency_rad_s", NOMINAL_ACTUATOR_WN_RAD_S),
        ("actuator_cmd_damping_ratio", NOMINAL_ACTUATOR_ZETA),
        ("dr_ball_mass_range", np.asarray([0.0039, 0.0041])),
    ):
        actual = np.asarray(env_cfg.get(name, ()), dtype=np.float64)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=0.0, atol=1e-12
        ):
            raise CheckpointContractError(
                f"V153 checkpoint {name} does not match the selected model"
            )


def load_released_actor(
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> Any:
    """Authenticate, validate, and load the deterministic V153 actor mean."""

    path = resolve_checkpoint_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"released GPU1 V153 checkpoint not found: {path}")
    actual_digest = sha256_file(path)
    if actual_digest != CHECKPOINT_SHA256:
        raise CheckpointContractError(
            "released GPU1 V153 checkpoint SHA-256 mismatch: "
            f"{actual_digest} != {CHECKPOINT_SHA256}"
        )
    from pingpong_controller.tools.rl_2real.mjx_policy_controller import (
        NumpyMJXActor,
        load_mjx_checkpoint,
    )

    payload = load_mjx_checkpoint(path)
    validate_gpu1_v153_checkpoint(payload)
    actor = NumpyMJXActor(payload["params"])
    if (actor.obs_dim, actor.act_dim) != (ACTOR_OBS_DIM, ACTION_DIM):
        raise CheckpointContractError(
            "released GPU1 V153 actor dimensions changed after loading"
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
            "released GPU1 V153 actor failed the golden-vector check"
        )
    return actor


def _estimate_contact_time_s(
    relative_z_m: float,
    ball_vz_m_s: float,
    racket_vz_m_s: float,
    ball_observation_age_s: float,
) -> float:
    age_s = float(ball_observation_age_s)
    if age_s >= LOST_BALL_TIMEOUT_S:
        return MAX_CONTACT_TIME_S
    gravity_z_m_s2 = -9.81
    propagated_z = (
        float(relative_z_m)
        + float(ball_vz_m_s) * age_s
        + 0.5 * gravity_z_m_s2 * age_s * age_s
    )
    propagated_vz = (
        float(ball_vz_m_s)
        + gravity_z_m_s2 * age_s
        - float(racket_vz_m_s)
    )
    gravity = abs(gravity_z_m_s2)
    discriminant = propagated_vz * propagated_vz + 2.0 * gravity * propagated_z
    if (
        discriminant < 0.0
        or not math.isfinite(discriminant)
        or abs(propagated_z) > 10.0
        or abs(propagated_vz) > 50.0
    ):
        return MAX_CONTACT_TIME_S
    root = math.sqrt(discriminant)
    candidates = (
        (propagated_vz + root) / gravity,
        (propagated_vz - root) / gravity,
    )
    positive = [value for value in candidates if value >= 0.0]
    if not positive:
        return MAX_CONTACT_TIME_S
    return float(np.clip(min(positive), 0.0, MAX_CONTACT_TIME_S))


@dataclass(frozen=True)
class GPU1QaccFtState:
    """Command-state integration and observation-only delay history."""

    q_cmd_rad: np.ndarray
    dq_cmd_rad_s: np.ndarray
    q_ref_active_rad: np.ndarray
    command_buffer_rad: np.ndarray
    previous_executed_action: np.ndarray


@dataclass(frozen=True)
class GPU1QaccFtStep:
    """One deterministic policy result and the next persistent state."""

    observation: np.ndarray
    executed_action: np.ndarray
    commanded_qacc_rad_s2: np.ndarray
    commanded_qvel_rad_s: np.ndarray
    drive_input_q_rad: np.ndarray
    state: GPU1QaccFtState


class GPU1QaccFtReference:
    """Exact 67-D command-state boundary used by the GPU1 V153 actor."""

    def __init__(
        self,
        actor_mean: Callable[[np.ndarray], np.ndarray],
        *,
        joint_position_low_rad: np.ndarray,
        joint_position_high_rad: np.ndarray,
        joint_velocity_limit_deg_s: tuple[float, ...] = (
            210.0,
            210.0,
            240.0,
            240.0,
            300.0,
            300.0,
            300.0,
        ),
        joint_acceleration_limit_deg_s2: tuple[float, ...] = (
            1300.0,
            1300.0,
            1800.0,
            3000.0,
            3000.0,
            3000.0,
            3000.0,
        ),
        maximum_joint_feedback_age_s: float = 0.010,
    ) -> None:
        self.actor_mean = actor_mean
        self.q_low = _finite_vector(
            joint_position_low_rad, ACTION_DIM, "joint_position_low_rad"
        )
        self.q_high = _finite_vector(
            joint_position_high_rad, ACTION_DIM, "joint_position_high_rad"
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
        acceleration_deg_s2 = _finite_vector(
            joint_acceleration_limit_deg_s2,
            ACTION_DIM,
            "joint_acceleration_limit_deg_s2",
        )
        if np.any(velocity_deg_s <= 0.0) or np.any(
            acceleration_deg_s2 <= 0.0
        ):
            raise ValueError(
                "joint velocity and acceleration limits must be positive"
            )
        self.velocity_limit_rad_s = np.deg2rad(velocity_deg_s).astype(
            np.float32
        )
        self.acceleration_limit_rad_s2 = np.deg2rad(
            acceleration_deg_s2
        ).astype(np.float32)
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
        self.delay_steps = int(round(OBSERVED_COMMAND_DELAY_S / POLICY_DT_S))
        self.command_buffer_length = self.delay_steps + 1

    def reset(
        self, measured_q_rad: np.ndarray, measured_dq_rad_s: np.ndarray
    ) -> GPU1QaccFtState:
        """Reset hidden q_cmd/dq_cmd and delay history from fresh feedback."""

        q = _finite_vector(measured_q_rad, ACTION_DIM, "measured_q_rad")
        _finite_vector(measured_dq_rad_s, ACTION_DIM, "measured_dq_rad_s")
        if np.any(q < self.q_low) or np.any(q > self.q_high):
            raise ValueError(
                "measured_q_rad is outside configured joint limits"
            )
        command_buffer = np.broadcast_to(
            q, (self.command_buffer_length, ACTION_DIM)
        ).copy()
        return GPU1QaccFtState(
            q_cmd_rad=q.copy(),
            dq_cmd_rad_s=np.zeros(ACTION_DIM, dtype=np.float32),
            q_ref_active_rad=q.copy(),
            command_buffer_rad=command_buffer.astype(np.float32),
            previous_executed_action=np.zeros(ACTION_DIM, dtype=np.float32),
        )

    @staticmethod
    def _build_observation(
        state: GPU1QaccFtState,
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
        q_cmd = _finite_vector(state.q_cmd_rad, ACTION_DIM, "q_cmd_rad")
        dq_cmd = _finite_vector(
            state.dq_cmd_rad_s, ACTION_DIM, "dq_cmd_rad_s"
        )
        active_q = _finite_vector(
            state.q_ref_active_rad, ACTION_DIM, "q_ref_active_rad"
        )
        relative_position = ball_position - racket_position
        contact_time_s = _estimate_contact_time_s(
            float(relative_position[2]),
            float(ball_velocity[2]),
            float(racket_velocity[2]),
            age_s,
        )

        base_observation = np.concatenate(
            [
                q,
                dq,
                base_position,
                base_velocity,
                ball_position,
                ball_velocity,
                racket_position,
                racket_velocity,
                relative_position,
                _finite_vector(
                    state.previous_executed_action,
                    ACTION_DIM,
                    "previous_executed_action",
                ),
                q_cmd - q,
                np.asarray([age_norm], dtype=np.float32),
            ]
        ).astype(np.float32)
        delay_observation = np.concatenate(
            [
                np.asarray([1.0], dtype=np.float32),
                dq_cmd,
                active_q - q,
                np.asarray(
                    [
                        contact_time_s,
                        contact_time_s - OBSERVED_COMMAND_DELAY_S,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        observation = np.concatenate(
            [base_observation, delay_observation]
        ).astype(np.float32)
        if observation.shape != (ACTOR_OBS_DIM,):
            raise RuntimeError("assembled GPU1 V153 observation is not 67-D")
        return observation

    def step(
        self,
        state: GPU1QaccFtState,
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
        confirmed_hit_count: int,
    ) -> GPU1QaccFtStep:
        """Run one 5 ms QACC tick and return the current q-only target."""

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
        ball_observation_age_s = represented_ball_age_s(
            control_time_s,
            ball_state_time_s,
            confirmed_hit_count,
        )
        observation = self._build_observation(
            state,
            measured_q_rad=measured_q_rad,
            measured_dq_rad_s=measured_dq_rad_s,
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
        qacc = np.clip(
            action * self.acceleration_limit_rad_s2,
            -self.acceleration_limit_rad_s2,
            self.acceleration_limit_rad_s2,
        ).astype(np.float32)
        dq_cmd = np.clip(
            _finite_vector(state.dq_cmd_rad_s, ACTION_DIM, "dq_cmd_rad_s")
            + qacc * np.float32(POLICY_DT_S),
            -self.velocity_limit_rad_s,
            self.velocity_limit_rad_s,
        ).astype(np.float32)
        q_cmd = np.clip(
            _finite_vector(state.q_cmd_rad, ACTION_DIM, "q_cmd_rad")
            + dq_cmd * np.float32(POLICY_DT_S),
            self.q_low,
            self.q_high,
        ).astype(np.float32)

        command_buffer = np.asarray(
            state.command_buffer_rad, dtype=np.float32
        )
        expected_shape = (self.command_buffer_length, ACTION_DIM)
        if command_buffer.shape != expected_shape or not np.all(
            np.isfinite(command_buffer)
        ):
            raise ValueError(
                f"command_buffer_rad must have shape {expected_shape}"
            )
        command_buffer = np.concatenate(
            [command_buffer[1:], q_cmd[None, :]], axis=0
        )
        active_index = command_buffer.shape[0] - 1 - self.delay_steps
        active_q = command_buffer[active_index].copy()
        next_state = GPU1QaccFtState(
            q_cmd_rad=q_cmd.copy(),
            dq_cmd_rad_s=dq_cmd.copy(),
            q_ref_active_rad=active_q,
            command_buffer_rad=command_buffer,
            previous_executed_action=action.copy(),
        )
        return GPU1QaccFtStep(
            observation=observation,
            executed_action=action,
            commanded_qacc_rad_s2=qacc,
            commanded_qvel_rad_s=dq_cmd,
            drive_input_q_rad=q_cmd.copy(),
            state=next_state,
        )
