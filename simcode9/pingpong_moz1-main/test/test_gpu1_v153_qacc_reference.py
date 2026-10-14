"""Regression tests for the selected GPU1 V153 robot boundary."""

import copy

import numpy as np
import pytest

from pingpong_controller.tools.rl_2real.mjx_policy_controller import (
    load_mjx_checkpoint,
)
from pingpong_controller.tools.rl_2real.gpu1_v153_qacc_reference import (
    ACTOR_OBS_DIM,
    CHECKPOINT_PATH,
    CheckpointContractError,
    GPU1QaccFtReference,
    InvalidBallStateError,
    StaleJointFeedbackError,
    fractional_refresh_due,
    load_released_actor,
    resolve_checkpoint_path,
    validate_gpu1_v153_checkpoint,
)


def _payload() -> dict[str, object]:
    return {
        "obs_dim": 67,
        "critic_obs_dim": 279,
        "act_dim": 7,
        "stage_name": "record_new3_sim2real_v153_b2_b3_energy_p025_60hz",
        "stage_index": 59,
        "stage_update": 441,
        "global_update": 1092,
        "step": 14841511936,
        "args": {
            "curriculum_profile": (
                "goal_d455_sport_taskspace_record_new3_sim2real_fixed_base_"
                "ball4g_dual_domain_homotopy_v153"
            ),
            "seed": 82731,
        },
        "params": {
            "pi": {
                "l1": {"w": np.zeros((67, 2)), "b": np.zeros(2)},
                "l2": {"w": np.zeros((2, 3)), "b": np.zeros(3)},
                "out": {"w": np.zeros((3, 7)), "b": np.zeros(7)},
            }
        },
        "env_cfg": {
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
            "ball_obs_rate_hz": 60.0,
            "ball_obs_fractional_rate": True,
            "ball_obs_age_tracks_stale": False,
            "ball_obs_age_clip": 0.5,
            "ball_obs_velocity_observer_mode": "raw",
            "ball_obs_camera_missing_prob": 0.0,
            "ball_obs_dropout_prob": 0.0,
            "ball_obs_dropout_burst_prob": 0.0,
            "dr_post_hit_ball_obs_latency": True,
            "dr_post_hit_ball_obs_latency_probability": 1.0,
            "dr_post_hit_ball_obs_latency_min_confirmed_hits": 1,
            "dr_post_hit_ball_obs_latency_timestamp_consistent_age": True,
            "dr_obs_latency_steps_range": (0, 2),
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
            "dr_ball_mass_range": (0.0039, 0.0041),
            "dr_ball_normalized_inertia_range": (0.4, 0.4),
            "actuator_cmd_delay_ms_per_joint": (
                45.0,
                50.0,
                45.0,
                40.0,
                35.0,
                45.0,
                50.0,
            ),
            "actuator_cmd_natural_frequency_rad_s": (
                21.8911379437,
                22.0895753812,
                22.6504705,
                21.730533,
                20.1549483562,
                22.2451380938,
                23.0546718187,
            ),
            "actuator_cmd_damping_ratio": (
                0.3330028,
                0.3295521,
                0.3111642,
                0.3114,
                0.3131064,
                0.3112326,
                0.28553475,
            ),
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
        "confirmed_hit_count": 1,
    }


def test_gpu1_v153_checkpoint_contract_is_fail_closed() -> None:
    payload = _payload()
    validate_gpu1_v153_checkpoint(payload)

    invalid = copy.deepcopy(payload)
    invalid["env_cfg"]["actuator_compensation_mode"] = "inverse_mpc"
    with pytest.raises(CheckpointContractError, match="compensation"):
        validate_gpu1_v153_checkpoint(invalid)


def test_gpu1_v153_checkpoint_and_actor_are_authenticated() -> None:
    payload = load_mjx_checkpoint(resolve_checkpoint_path(CHECKPOINT_PATH))
    validate_gpu1_v153_checkpoint(payload)
    actor = load_released_actor()
    assert (actor.obs_dim, actor.act_dim) == (67, 7)
    action = actor.mean_action(np.zeros(67, dtype=np.float32))
    assert action.shape == (7,)
    assert np.all(np.isfinite(action))


def test_gpu1_qacc_ft_integrates_command_state() -> None:
    captured: list[np.ndarray] = []

    def actor(observation: np.ndarray) -> np.ndarray:
        captured.append(observation.copy())
        return np.ones(7, dtype=np.float32)

    controller = GPU1QaccFtReference(
        actor,
        joint_position_low_rad=np.full(7, -2.0),
        joint_position_high_rad=np.full(7, 2.0),
    )
    q = np.zeros(7, dtype=np.float32)
    dq = np.zeros(7, dtype=np.float32)
    state = controller.reset(q, dq)
    first = controller.step(state, **_step_kwargs(q, dq))

    assert first.observation.shape == (ACTOR_OBS_DIM,)
    np.testing.assert_array_equal(first.observation[42:49], np.zeros(7))
    np.testing.assert_allclose(first.observation[49], 0.02)
    np.testing.assert_array_equal(
        first.observation[50:65], np.r_[1.0, np.zeros(14)]
    )
    assert 0.0 <= first.observation[65] <= 0.5
    np.testing.assert_allclose(
        first.observation[66], first.observation[65] - 0.045
    )
    np.testing.assert_array_equal(captured[0], first.observation)
    expected_dq = controller.acceleration_limit_rad_s2 * np.float32(0.005)
    expected_q = expected_dq * np.float32(0.005)
    np.testing.assert_allclose(first.commanded_qvel_rad_s, expected_dq)
    np.testing.assert_allclose(first.drive_input_q_rad, expected_q)
    np.testing.assert_array_equal(first.state.q_ref_active_rad, q)

    moved_measurement = np.full(7, 0.5, dtype=np.float32)
    second = controller.step(
        first.state,
        **_step_kwargs(moved_measurement, dq),
    )
    expected_second_dq = expected_dq * 2.0
    expected_second_q = expected_q + expected_second_dq * np.float32(0.005)
    np.testing.assert_allclose(second.drive_input_q_rad, expected_second_q)
    np.testing.assert_allclose(
        second.observation[42:49], first.drive_input_q_rad - moved_measurement
    )


def test_gpu1_qacc_ft_delay_history_does_not_delay_published_target() -> None:
    controller = GPU1QaccFtReference(
        lambda _observation: np.ones(7, dtype=np.float32),
        joint_position_low_rad=np.full(7, -2.0),
        joint_position_high_rad=np.full(7, 2.0),
    )
    q = np.zeros(7, dtype=np.float32)
    state = controller.reset(q, q)
    first_target = None
    current = None
    for _ in range(controller.delay_steps + 1):
        current = controller.step(state, **_step_kwargs(q, q))
        state = current.state
        if first_target is None:
            first_target = current.drive_input_q_rad.copy()
    assert current is not None
    assert first_target is not None
    np.testing.assert_allclose(current.state.q_ref_active_rad, first_target)
    assert np.linalg.norm(current.drive_input_q_rad - first_target) > 0.0


def test_gpu1_qacc_ft_stale_feedback_never_calls_actor() -> None:
    called = False

    def actor(_observation: np.ndarray) -> np.ndarray:
        nonlocal called
        called = True
        return np.zeros(7, dtype=np.float32)

    controller = GPU1QaccFtReference(
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


def test_gpu1_qacc_ft_age_switches_on_after_first_confirmed_hit() -> None:
    captured: list[np.ndarray] = []
    controller = GPU1QaccFtReference(
        lambda observation: (
            captured.append(observation.copy())
            or np.zeros(7, dtype=np.float32)
        ),
        joint_position_low_rad=np.full(7, -2.0),
        joint_position_high_rad=np.full(7, 2.0),
    )
    q = np.zeros(7, dtype=np.float32)
    state = controller.reset(q, q)
    before = _step_kwargs(q, q)
    before["confirmed_hit_count"] = 0
    first = controller.step(state, **before)
    np.testing.assert_array_equal(first.observation[49], 0.0)

    after = _step_kwargs(q, q)
    second = controller.step(first.state, **after)
    np.testing.assert_allclose(second.observation[49], 0.02)
    np.testing.assert_array_equal(captured[-1], second.observation)


def test_gpu1_qacc_ft_ball_timestamp_is_fail_closed() -> None:
    controller = GPU1QaccFtReference(
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

    assert not fractional_refresh_due(3, 0)
    assert fractional_refresh_due(4, 0)
