"""Regression tests for the GPU0-QVEL/REAL-RMP robot boundary."""

import numpy as np
import pytest

from pingpong_controller.tools.rl_2real import (
    gpu0_qvel_real_rmp_reference as gpu0_reference,
)


ACTOR_OBS_DIM = gpu0_reference.ACTOR_OBS_DIM
CHECKPOINT_SHA256 = gpu0_reference.CHECKPOINT_SHA256
CheckpointContractError = gpu0_reference.CheckpointContractError
FractionalBallObservation90Hz = (
    gpu0_reference.FractionalBallObservation90Hz
)
GPU0QvelRealRmpReference = gpu0_reference.GPU0QvelRealRmpReference
InvalidBallStateError = gpu0_reference.InvalidBallStateError
InvalidJointStateError = gpu0_reference.InvalidJointStateError
REFERENCE_ERROR_HORIZON_S = gpu0_reference.REFERENCE_ERROR_HORIZON_S
StaleJointFeedbackError = gpu0_reference.StaleJointFeedbackError
validate_gpu0_v85_checkpoint = gpu0_reference.validate_gpu0_v85_checkpoint


def _payload() -> dict[str, object]:
    return {
        "obs_dim": 57,
        "critic_obs_dim": 368,
        "act_dim": 7,
        "stage_name": gpu0_reference.CHECKPOINT_STAGE_NAME,
        "stage_index": gpu0_reference.CHECKPOINT_STAGE_INDEX,
        "stage_update": gpu0_reference.CHECKPOINT_STAGE_UPDATE,
        "global_update": gpu0_reference.CHECKPOINT_GLOBAL_UPDATE,
        "step": gpu0_reference.CHECKPOINT_STEP,
        "args": {
            "curriculum_profile": gpu0_reference.PROFILE_NAME,
            "seed": 20261004,
        },
        "params": {
            "pi": {
                "l1": {"w": np.zeros((57, 2)), "b": np.zeros(2)},
                "l2": {"w": np.zeros((2, 3)), "b": np.zeros(3)},
                "out": {"w": np.zeros((3, 7)), "b": np.zeros(7)},
            }
        },
        "env_cfg": {
            "action_command_mode": "velocity",
            "action_velocity_scale": 1.0,
            "policy_integration_feedback_source": "measured",
            "recovered_rmp_motion_mode": True,
            "recovered_rmp_direct_qvel_target": True,
            "recovered_rmp_bounded_qvel_reference": True,
            "recovered_rmp_qvel_reference_error_horizon_s": 0.0,
            "recovered_rmp_qvel_reference_error_horizon_s_per_joint": (
                0.018,
                0.019,
                0.019,
                0.018,
                0.018,
                0.018,
                0.018,
            ),
            "include_qvel_reference_error_obs": True,
            "enable_delay_conditioning": False,
            "high_latency_obs": False,
            "ball_obs_rate_hz": 90.0,
            "ball_obs_fractional_rate": True,
            "ball_obs_age_tracks_stale": True,
            "ball_obs_age_clip": 0.5,
            "ball_obs_velocity_observer_mode": "raw",
            "lost_ball_timeout_ms": 350.0,
            "actor_mask_previous_action": False,
            "actor_previous_action_scale": 1.0,
            "arm_action_limiter": True,
            "arm_vel_limit_deg_s": (
                210.0,
                210.0,
                240.0,
                240.0,
                300.0,
                300.0,
                300.0,
            ),
            "arm_acc_limit_deg_s2": (
                1300.0,
                1300.0,
                1800.0,
                3000.0,
                3000.0,
                3000.0,
                3000.0,
            ),
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
        },
    }


def _step_kwargs(q: np.ndarray, dq: np.ndarray) -> dict[str, object]:
    return {
        "measured_q_rad": q,
        "measured_dq_rad_s": dq,
        "joint_feedback_age_s": 0.001,
        "base_q": np.zeros(3, dtype=np.float32),
        "base_dq": np.zeros(3, dtype=np.float32),
        "ball_position_base_m": np.asarray([0.1, -0.2, 1.2]),
        "ball_velocity_base_m_s": np.asarray([0.0, 0.0, -0.3]),
        "racket_position_base_m": np.asarray([0.0, -0.2, 1.0]),
        "racket_velocity_base_m_s": np.asarray([0.0, 0.0, 0.1]),
        "control_time_s": 10.010,
        "ball_state_time_s": 10.000,
    }


def test_gpu0_v85_checkpoint_contract_is_fail_closed() -> None:
    payload = _payload()
    validate_gpu0_v85_checkpoint(payload)

    invalid = dict(payload)
    invalid["env_cfg"] = dict(payload["env_cfg"])
    invalid["env_cfg"]["policy_integration_feedback_source"] = "command"
    with pytest.raises(CheckpointContractError, match="feedback_source"):
        validate_gpu0_v85_checkpoint(invalid)

    wrong_model = dict(payload)
    wrong_model["step"] = gpu0_reference.CHECKPOINT_STEP + 1
    with pytest.raises(CheckpointContractError, match="step"):
        validate_gpu0_v85_checkpoint(wrong_model)


def test_released_gpu0_v85_checkpoint_and_actor_golden_vector() -> None:
    path = gpu0_reference.resolve_checkpoint_path()
    assert gpu0_reference.sha256_file(path) == CHECKPOINT_SHA256
    actor = gpu0_reference.load_released_actor(path)
    action = actor.mean_action(np.zeros(ACTOR_OBS_DIM, dtype=np.float32))
    np.testing.assert_allclose(
        action,
        gpu0_reference.ZERO_OBSERVATION_ACTOR_MEAN,
        rtol=0.0,
        atol=2e-7,
    )


def test_gpu0_frame_transforms_match_training_formula() -> None:
    base_q = np.asarray([1.0, 2.0, np.pi / 2.0], dtype=np.float32)
    base_dq = np.asarray([0.2, -0.1, 0.5], dtype=np.float32)
    point = np.asarray([2.0, 2.0, 3.0], dtype=np.float32)
    velocity = np.asarray([0.4, 0.3, -0.2], dtype=np.float32)

    np.testing.assert_allclose(
        gpu0_reference.world_point_to_base(point, base_q),
        np.asarray([0.0, -1.0, 3.0]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        gpu0_reference.world_velocity_to_base(
            velocity, point, base_q, base_dq
        ),
        np.asarray([-0.1, -0.2, -0.2]),
        atol=1e-6,
    )


def test_gpu0_fractional_ball_scheduler_holds_between_refreshes() -> None:
    scheduler = FractionalBallObservation90Hz()
    initial = scheduler.reset(0, [1.0, 2.0, 3.0], [0.0, 0.0, 1.0], 4.0)
    held = scheduler.update(
        initial, 2, [9.0, 9.0, 9.0], [9.0, 9.0, 9.0], 4.01
    )
    assert held is initial
    refreshed = scheduler.update(
        held, 3, [1.1, 2.1, 3.1], [0.1, 0.2, 0.3], 4.015
    )
    np.testing.assert_allclose(refreshed.position_base_m, [1.1, 2.1, 3.1])
    assert refreshed.last_refresh_step == 3


def test_gpu0_real_rmp_observation_and_bounded_reference_are_exact() -> None:
    captured: list[np.ndarray] = []

    def actor(observation: np.ndarray) -> np.ndarray:
        captured.append(observation.copy())
        return np.ones(7, dtype=np.float32)

    controller = GPU0QvelRealRmpReference(
        actor,
        joint_position_low_rad=np.full(7, -2.0),
        joint_position_high_rad=np.full(7, 2.0),
    )
    q = np.zeros(7, dtype=np.float32)
    dq = np.linspace(0.0, 0.6, 7, dtype=np.float32)
    state = controller.reset(q, dq)
    first = controller.step(state, **_step_kwargs(q, dq))

    assert first.observation.shape == (ACTOR_OBS_DIM,)
    np.testing.assert_array_equal(first.observation[42:49], np.zeros(7))
    np.testing.assert_allclose(first.observation[49], 0.02)
    np.testing.assert_array_equal(first.observation[50:57], np.zeros(7))
    np.testing.assert_array_equal(captured[0], first.observation)
    expected_first = controller.velocity_limit_rad_s * np.float32(0.005)
    np.testing.assert_allclose(first.rmp_input_q_rad, expected_first)

    next_dq = dq + np.float32(0.05)
    second = controller.step(
        first.state,
        **_step_kwargs(q, next_dq),
    )
    np.testing.assert_allclose(second.observation[42:49], next_dq - dq)
    np.testing.assert_allclose(
        second.observation[50:57], first.rmp_input_q_rad - q
    )

    current = second
    for _ in range(20):
        current = controller.step(current.state, **_step_kwargs(q, next_dq))
    maximum_error = controller.velocity_limit_rad_s * REFERENCE_ERROR_HORIZON_S
    np.testing.assert_allclose(current.rmp_input_q_rad, maximum_error)


def test_gpu0_real_rmp_stale_feedback_never_calls_actor() -> None:
    called = False

    def actor(_observation: np.ndarray) -> np.ndarray:
        nonlocal called
        called = True
        return np.zeros(7, dtype=np.float32)

    controller = GPU0QvelRealRmpReference(
        actor,
        joint_position_low_rad=np.full(7, -2.0),
        joint_position_high_rad=np.full(7, 2.0),
    )
    q = np.zeros(7, dtype=np.float32)
    state = controller.reset(q, q)
    kwargs = _step_kwargs(q, q)
    kwargs["joint_feedback_age_s"] = 0.011
    with pytest.raises(StaleJointFeedbackError):
        controller.step(state, **kwargs)
    assert not called


def test_gpu0_real_rmp_out_of_range_joint_never_calls_actor() -> None:
    called = False

    def actor(_observation: np.ndarray) -> np.ndarray:
        nonlocal called
        called = True
        return np.zeros(7, dtype=np.float32)

    controller = GPU0QvelRealRmpReference(actor)
    q = np.zeros(7, dtype=np.float32)
    state = controller.reset(q, q)
    invalid_q = q.copy()
    invalid_q[0] = gpu0_reference.JOINT_POSITION_HIGH_RAD[0] + 0.01
    with pytest.raises(InvalidJointStateError):
        controller.step(state, **_step_kwargs(invalid_q, q))
    assert not called


def test_gpu0_real_rmp_ball_timestamp_is_fail_closed() -> None:
    controller = GPU0QvelRealRmpReference(
        lambda _observation: np.zeros(7, dtype=np.float32),
        joint_position_low_rad=np.full(7, -2.0),
        joint_position_high_rad=np.full(7, 2.0),
    )
    q = np.zeros(7, dtype=np.float32)
    state = controller.reset(q, q)
    kwargs = _step_kwargs(q, q)
    kwargs["ball_state_time_s"] = 9.0
    with pytest.raises(InvalidBallStateError, match="350 ms"):
        controller.step(state, **kwargs)

    assert not gpu0_reference.fractional_refresh_due(2, 0)
    assert gpu0_reference.fractional_refresh_due(3, 0)
