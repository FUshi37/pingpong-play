from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RL_SIM_DIR = ROOT / "pingpong_controller" / "tools" / "rl_sim"
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from delay_control import (  # noqa: E402
    compensate_q_ref,
    delay_steps_from_tau,
    estimate_contact_time,
    push_command_buffer,
    smooth_action,
)

from pingpong_controller.safety_limiter import (  # noqa: E402
    RightArmCommandSafetyLimiter,
    project_damped_target_tracking_command_step_with_jerk,
    project_target_tracking_command_step,
)


def test_delay_steps_150ms_at_200hz() -> None:
    assert delay_steps_from_tau(0.150, 0.005) == 30


def test_delay_zero_active_matches_latest() -> None:
    buffer = np.zeros((4, 7), dtype=np.float32)
    q_ref_latest = np.arange(7, dtype=np.float32)
    _new_buffer, q_ref_active = push_command_buffer(buffer, q_ref_latest, delay_steps=0)
    np.testing.assert_allclose(q_ref_active, q_ref_latest)


def test_command_buffer_delay_output() -> None:
    buffer = np.stack([np.full(7, i, dtype=np.float32) for i in range(4)], axis=0)
    new_buffer, q_ref_active = push_command_buffer(buffer, np.full(7, 4.0, dtype=np.float32), delay_steps=2)
    np.testing.assert_allclose(new_buffer[:, 0], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(q_ref_active, np.full(7, 2.0, dtype=np.float32))


def test_action_jerk_limit() -> None:
    action, scale = smooth_action(
        np.ones(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        dt=0.005,
        action_jerk_limit=10.0,
    )
    assert scale == pytest.approx(1.0)
    np.testing.assert_allclose(action, np.full(7, 0.05, dtype=np.float32), atol=1e-6)


def test_anti_windup_scales_large_error() -> None:
    action, scale = smooth_action(
        np.ones(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        dt=0.005,
        e_active=np.ones(7, dtype=np.float32),
        enable_anti_windup=True,
        anti_windup_error_threshold=0.1,
        anti_windup_min_scale=0.25,
    )
    assert scale == pytest.approx(0.25)
    np.testing.assert_allclose(action, np.full(7, 0.25, dtype=np.float32), atol=1e-6)


def _causal_inverse_mpc_drive_rollout(
    q: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
) -> np.ndarray:
    """Roll the deployable inverse-MPC/FOPDT/drive-planner state forward."""
    dt = 0.005
    delay_steps = 14
    tau_s = 0.074
    warm_q = np.deg2rad(
        np.asarray([5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214])
    ).astype(np.float32)
    q_low = RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD.astype(np.float32)
    q_high = RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD.astype(np.float32)
    vel_limit = np.deg2rad(
        RightArmCommandSafetyLimiter.VEL_LIMIT_DEG_S
    )
    acc_limit = 0.80 * np.deg2rad(
        RightArmCommandSafetyLimiter.ACC_LIMIT_DEG_S2
    )
    buffer_len = 40
    command_buffer = np.broadcast_to(warm_q, (buffer_len, 7)).copy()
    delay_buffer = np.broadcast_to(warm_q, (delay_steps + 1, 7)).copy()
    applied_q = warm_q.astype(np.float64).copy()
    applied_vel = np.zeros(7, dtype=np.float64)
    alpha = dt / (tau_s + dt)
    output = []

    for step in range(q.shape[0]):
        raw = compensate_q_ref(
            "inverse_mpc",
            q[step],
            qvel[step],
            qacc[step],
            dt=dt,
            delay_steps=delay_steps,
            actuator_tau_s=tau_s,
            actuator_gain=1.0,
            mpc_beta=1.2,
            mpc_delay_scale=1.05,
            mpc_tau_scale=0.75,
            mpc_horizon_steps=6,
            mpc_tracking_weight=1.0,
            mpc_nominal_weight=0.25,
            mpc_delta_weight=0.05,
            mpc_max_delta_rad=np.deg2rad(30.0),
            applied_q=applied_q.astype(np.float32),
            command_buffer=command_buffer,
            warm_q=warm_q,
            q_low=q_low,
            q_high=q_high,
        )
        command_buffer = np.concatenate([command_buffer[1:], raw[None, :]])
        delay_buffer = np.concatenate([delay_buffer[1:], raw[None, :]])
        active = delay_buffer[0]
        predicted_fopdt_q = applied_q + alpha * (active - applied_q)
        applied_q, applied_vel, _, _ = project_target_tracking_command_step(
            predicted_fopdt_q,
            applied_q,
            applied_vel,
            q_low,
            q_high,
            vel_limit,
            acc_limit,
            dt,
        )
        output.append(applied_q.copy())
    return np.asarray(output)


def test_inverse_mpc_drive_planner_has_no_future_truth_lookahead() -> None:
    """Changing an unseen future suffix must not change the output prefix."""
    rng = np.random.default_rng(20260720)
    steps = 80
    shared_prefix = 47
    q = rng.uniform(-0.6, 0.6, size=(steps, 7)).astype(np.float32)
    qvel = rng.uniform(-1.2, 1.2, size=(steps, 7)).astype(np.float32)
    qacc = rng.uniform(-8.0, 8.0, size=(steps, 7)).astype(np.float32)
    q_changed = q.copy()
    qvel_changed = qvel.copy()
    qacc_changed = qacc.copy()
    q_changed[shared_prefix:] = rng.uniform(
        -2.0, 2.0, size=(steps - shared_prefix, 7)
    )
    qvel_changed[shared_prefix:] = rng.uniform(
        -5.0, 5.0, size=(steps - shared_prefix, 7)
    )
    qacc_changed[shared_prefix:] = rng.uniform(
        -30.0, 30.0, size=(steps - shared_prefix, 7)
    )

    baseline = _causal_inverse_mpc_drive_rollout(q, qvel, qacc)
    changed = _causal_inverse_mpc_drive_rollout(
        q_changed,
        qvel_changed,
        qacc_changed,
    )
    np.testing.assert_array_equal(
        baseline[:shared_prefix],
        changed[:shared_prefix],
    )


def test_target_tracking_drive_governor_enforces_q_dq_ddq_and_jerk() -> None:
    dt = 0.001
    q_low = RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD
    q_high = RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD
    vel_limit = np.deg2rad(RightArmCommandSafetyLimiter.VEL_LIMIT_DEG_S)
    acc_limit = np.deg2rad(RightArmCommandSafetyLimiter.ACC_LIMIT_DEG_S2)
    jerk_limit = np.deg2rad(np.full(7, 175000.0))
    q = 0.5 * (q_low + q_high)
    qvel = np.zeros(7, dtype=np.float64)
    qacc = np.zeros(7, dtype=np.float64)

    for step in range(800):
        direction = 1.0 if (step // 40) % 2 == 0 else -1.0
        target = np.clip(q + direction * 0.35, q_low + 0.05, q_high - 0.05)
        previous_acc = qacc.copy()
        q, qvel, qacc, jerk_feasible = (
            project_damped_target_tracking_command_step_with_jerk(
                target,
                np.zeros(7, dtype=np.float64),
                q,
                qvel,
                qacc,
                q_low,
                q_high,
                vel_limit,
                acc_limit,
                jerk_limit,
                dt,
                natural_frequency_hz=8.0,
                damping_ratio=1.0,
            )
        )
        assert np.all(jerk_feasible)
        assert np.all(q >= q_low - 1e-10)
        assert np.all(q <= q_high + 1e-10)
        assert np.all(np.abs(qvel) <= vel_limit + 1e-10)
        assert np.all(np.abs(qacc) <= acc_limit + 1e-8)
        assert np.all(np.abs(qacc - previous_acc) <= jerk_limit * dt + 1e-8)


def test_contact_time_simple_ballistic_case() -> None:
    t_contact = estimate_contact_time(
        z_rel=0.20,
        vz_rel=-0.50,
        gravity=9.81,
        contact_height_offset=0.0,
        max_contact_time=0.5,
    )
    assert 0.0 < t_contact < 0.5


def test_ppo_actor_anchor_kl_is_zero_at_reference_and_positive_after_drift() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from train_juggle_mjx_ppo import PpoBatch, init_params, normal_logprob, policy_mean, ppo_loss

    reference = init_params(jax.random.PRNGKey(7), obs_dim=3, act_dim=2, hidden_dim=8)
    obs = jnp.asarray(
        [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1], [0.0, 0.2, -0.3], [0.7, 0.1, -0.2]],
        dtype=jnp.float32,
    )
    action = policy_mean(reference, obs)
    batch = PpoBatch(
        obs=obs,
        critic_obs=obs,
        action=action,
        old_logp=normal_logprob(action, action, reference["log_std"]),
        advantages=jnp.zeros((4,), dtype=jnp.float32),
        returns=jnp.zeros((4,), dtype=jnp.float32),
        old_values=jnp.zeros((4,), dtype=jnp.float32),
    )
    _loss, same_aux = ppo_loss(reference, batch, 0.2, 0.5, 0.0, reference, 1.0)
    drifted = dict(reference)
    drifted_pi = dict(reference["pi"])
    drifted_out = dict(reference["pi"]["out"])
    drifted_out["b"] = drifted_out["b"] + 0.2
    drifted_pi["out"] = drifted_out
    drifted["pi"] = drifted_pi
    _loss, drifted_aux = ppo_loss(drifted, batch, 0.2, 0.5, 0.0, reference, 1.0)
    replay_obs = obs * 1.7 + 0.1
    _loss, replay_aux = ppo_loss(
        drifted,
        batch,
        0.2,
        0.5,
        0.0,
        reference,
        1.0,
        replay_obs,
    )
    assert float(same_aux["actor_anchor_kl"]) == pytest.approx(0.0, abs=1e-6)
    assert float(drifted_aux["actor_anchor_kl"]) > 0.0
    assert float(replay_aux["actor_anchor_current_kl"]) > 0.0
    assert float(replay_aux["actor_anchor_replay_kl"]) > 0.0
    assert float(replay_aux["actor_anchor_kl"]) == pytest.approx(
        0.5
        * (
            float(replay_aux["actor_anchor_current_kl"])
            + float(replay_aux["actor_anchor_replay_kl"])
        )
    )


def test_observation_dimensions_for_delay_presets() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv  # noqa: E402

    xml = RL_SIM_DIR / "moz1_pd.xml"
    base = MjxJuggleConfig(domain_randomization=False, arm_action_limiter=True)
    env = MjxJuggleEnv(xml, n_envs=1, cfg=base)
    assert env.obs_dim == 50

    tau_only = replace(base, enable_delay_conditioning=True, include_tau_act_norm=True)
    env_tau = MjxJuggleEnv(xml, n_envs=1, cfg=tau_only)
    assert env_tau.obs_dim == 51

    command_state = replace(
        tau_only,
        include_command_state=True,
        include_active_command_error=True,
    )
    env_command = MjxJuggleEnv(xml, n_envs=1, cfg=command_state)
    assert env_command.obs_dim == 65

    phase = replace(command_state, include_phase_features=True)
    env_phase = MjxJuggleEnv(xml, n_envs=1, cfg=phase)
    assert env_phase.obs_dim == 67


def test_idealpd67_keeps_72ms_features_but_executes_latest_command() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleEnv  # noqa: E402
    from train_juggle_mjx_curriculum import (  # noqa: E402
        GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE,
        build_curriculum,
    )

    cfg = build_curriculum(
        curriculum_profile=GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE,
    )[0].cfg
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=cfg)
    assert (env.obs_dim, env.critic_obs_dim, env.act_dim) == (67, 231, 7)

    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(67), 1))
    warm_q = np.asarray(state.arm_cmd_q[0])
    next_state, obs, _reward, _done, metrics = env.step(
        state,
        jnp.ones((1, env.act_dim), dtype=jnp.float32),
    )

    q_ref_latest = np.asarray(metrics["q_ref_latest"])[0]
    q_ref_active = np.asarray(metrics["q_ref_active"])[0]
    q_servo_ref_active = np.asarray(metrics["q_servo_ref_active"])[0]
    q_servo_target = np.asarray(metrics["q_servo_target"])[0]
    assert float(np.linalg.norm(q_ref_latest - warm_q)) > 1e-5
    np.testing.assert_allclose(q_ref_active, warm_q, atol=1e-7)
    np.testing.assert_allclose(q_servo_ref_active, q_ref_latest, atol=1e-7)
    np.testing.assert_allclose(q_servo_target, q_ref_latest, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(next_state.arm_q_ref_active)[0],
        warm_q,
        atol=1e-7,
    )

    assert float(np.asarray(metrics["tau_act_ms"])[0]) == pytest.approx(72.0)
    assert float(np.asarray(metrics["delay_steps"])[0]) == pytest.approx(14.0)
    assert float(np.asarray(metrics["servo_execution_delay_steps"])[0]) == pytest.approx(0.0)
    assert float(np.asarray(metrics["servo_execution_delay_ms"])[0]) == pytest.approx(0.0)
    assert float(np.asarray(obs)[0, 50]) == pytest.approx(1.0)
    assert float(np.asarray(obs)[0, 65] - np.asarray(obs)[0, 66]) == pytest.approx(
        0.072,
        abs=1e-6,
    )


def test_actual_limiter_reports_bounded_fraction_and_penalizes_preprojection_violation() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv  # noqa: E402

    cfg = MjxJuggleConfig(
        domain_randomization=False,
        arm_action_limiter=True,
        arm_actual_state_limiter=True,
        action_clip_excess_penalty_weight=5.0,
        arm_limiter_penalty_weight=0.05,
    )
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(140), 1))
    # Force a large causal position-setpoint error so the unconstrained XML
    # position servo violates qacc before the bottom safety projection.
    state = state._replace(arm_cmd_q=state.arm_cmd_q + 0.20)
    next_state, _obs, _reward, _done, metrics = env.step(
        state,
        jnp.full((1, env.act_dim), 2.0, dtype=jnp.float32),
    )

    clip_fraction = float(np.asarray(metrics["arm_actual_safety_clip_fraction"])[0])
    assert 0.0 <= clip_fraction <= 1.0
    assert float(np.asarray(metrics["arm_actual_safety_clip_any"])[0]) == pytest.approx(1.0)
    assert float(np.asarray(metrics["arm_actual_safety_intervention_penalty"])[0]) > 0.0
    assert float(np.asarray(metrics["raw_action_clip_fraction"])[0]) == pytest.approx(1.0)
    assert float(np.asarray(metrics["raw_action_clip_excess_norm"])[0]) > 0.0
    assert float(np.asarray(metrics["reward/action_clip_excess_penalty"])[0]) < 0.0
    assert float(np.asarray(metrics["reward/arm_limiter_penalty"])[0]) < 0.0

    qacc = np.asarray(next_state.data.qacc[0, env.arm_vadr])
    limits = np.asarray(env.arm_acc_limit_rad_s2)
    assert np.all(np.abs(qacc) <= limits + 1e-4)


def test_actual_target_governor_enforces_substep_limits_without_jerk_emergency() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv  # noqa: E402

    cfg = MjxJuggleConfig(
        domain_randomization=False,
        arm_action_limiter=True,
        arm_actual_state_limiter=True,
        arm_actual_target_tracking_governor=True,
        arm_actual_governor_natural_frequency_hz=8.0,
        arm_actual_governor_damping_ratio=1.0,
        arm_actual_jerk_limit_deg_s3=(175000.0,) * 7,
    )
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(141), 1))
    state = state._replace(arm_cmd_q=state.arm_cmd_q + 0.20)
    next_state, _obs, _reward, _done, metrics = env.step(
        state,
        jnp.full((1, env.act_dim), 2.0, dtype=jnp.float32),
    )

    qvel = np.asarray(next_state.data.qvel[0, env.arm_vadr])
    qacc = np.asarray(next_state.data.qacc[0, env.arm_vadr])
    assert np.all(np.abs(qvel) <= np.asarray(env.arm_vel_limit_rad_s) + 1e-4)
    assert np.all(np.abs(qacc) <= np.asarray(env.arm_acc_limit_rad_s2) + 1e-4)
    assert float(
        np.asarray(metrics["arm_actual_governor_jerk_emergency_fraction"])[0]
    ) == pytest.approx(0.0)
    assert float(
        np.asarray(metrics["arm_actual_velocity_limit_utilization_max"])[0]
    ) <= 1.0 + 1e-4
    assert float(
        np.asarray(metrics["arm_actual_acceleration_limit_utilization_max"])[0]
    ) <= 1.0 + 1e-4
    assert float(
        np.asarray(metrics["arm_actual_governor_jerk_limit_utilization_max"])[0]
    ) <= 1.0 + 1e-4
    assert float(np.asarray(metrics["arm_actual_safety_feasible"])[0]) == pytest.approx(
        1.0
    )


def test_step_racket_velocity_uses_previous_racket_position() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv  # noqa: E402

    xml = RL_SIM_DIR / "moz1_pd.xml"
    cfg = MjxJuggleConfig(domain_randomization=False)
    env = MjxJuggleEnv(xml, n_envs=1, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))

    rpos = state.data.site_xpos[:, env.racket_site_id]
    history_offset = jnp.asarray([[0.01, 0.0, 0.0]], dtype=jnp.float32)
    state = state._replace(prev_racket_pos=rpos - history_offset)
    _next_state, obs, _reward, _done, _metrics = env.step(
        state,
        jnp.zeros((1, env.act_dim), dtype=jnp.float32),
    )

    racket_vel_xyz = np.asarray(obs[0, 29:32])
    assert float(np.linalg.norm(racket_vel_xyz)) > 0.5


def test_phase_contact_time_uses_relative_z_not_relative_x() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv  # noqa: E402

    xml = RL_SIM_DIR / "moz1_pd.xml"
    cfg = MjxJuggleConfig(
        domain_randomization=False,
        enable_delay_conditioning=True,
        include_tau_act_norm=True,
        include_command_state=True,
        include_active_command_error=True,
        include_phase_features=True,
    )
    env = MjxJuggleEnv(xml, n_envs=1, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))

    base_obs = jnp.zeros((1, env.base_obs_dim), dtype=jnp.float32)
    base_obs = base_obs.at[:, 25].set(-0.5)  # ball vz
    base_obs = base_obs.at[:, 31].set(0.0)  # racket vz
    base_obs = base_obs.at[:, 32].set(5.0)  # rel_x should not affect contact time
    base_obs = base_obs.at[:, 34].set(0.2)  # rel_z is the contact height input

    actual = env._estimate_contact_time_from_obs(state, base_obs)
    expected = env._estimate_contact_time_from_z_vz(
        state,
        base_obs[:, 34],
        base_obs[:, 25] - base_obs[:, 31],
        base_obs[:, 49] * float(cfg.ball_obs_age_clip),
    )
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)
    assert float(np.asarray(actual)[0]) < float(cfg.max_contact_time)


def test_ball_obs_stale_is_not_reported_as_dropout() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv  # noqa: E402

    xml = RL_SIM_DIR / "moz1_pd.xml"
    cfg = MjxJuggleConfig(
        domain_randomization=False,
        ball_obs_rate_hz=50.0,
        ball_obs_age_tracks_stale=True,
        ball_obs_dropout_prob=0.0,
        ball_obs_dropout_burst_prob=0.0,
    )
    env = MjxJuggleEnv(xml, n_envs=1, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(0), 1))
    _state, _obs, _reward, _done, metrics = env.step(
        state,
        jnp.zeros((1, env.act_dim), dtype=jnp.float32),
    )

    assert float(np.asarray(metrics["ball_obs_stale_active"])[0]) == pytest.approx(1.0)
    assert float(np.asarray(metrics["ball_obs_dropout_active"])[0]) == pytest.approx(0.0)
    assert float(np.asarray(metrics["ball_obs_refresh_due"])[0]) == pytest.approx(0.0)


def test_episode_coherent_missing_stays_fixed_until_reset() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv  # noqa: E402

    xml = RL_SIM_DIR / "moz1_pd.xml"
    cfg = MjxJuggleConfig(
        domain_randomization=False,
        ball_obs_rate_hz=200.0,
        ball_obs_fractional_rate=False,
        ball_obs_require_camera_visible=False,
        ball_obs_require_view_bounds=True,
        ball_obs_view_bounds_missing_prob=0.5,
        ball_obs_missing_episode_coherent_prob=1.0,
        ball_obs_age_tracks_stale=True,
        ball_obs_dropout_prob=0.0,
        ball_obs_dropout_burst_prob=0.0,
    )
    env = MjxJuggleEnv(xml, n_envs=32, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(17), 32))
    episode_mask = np.asarray(state.ball_obs_view_bounds_missing_enabled)
    assert np.all(np.asarray(state.ball_obs_missing_episode_coherent_enabled))
    assert episode_mask.any()
    assert (~episode_mask).any()

    true_bpos = state.data.xpos[:, env.ball_body_id].at[:, 2].set(2.0)
    true_bvel = jnp.zeros((32, 3), dtype=jnp.float32)
    state = state._replace(
        step_count=jnp.ones((32,), dtype=jnp.int32),
        last_ball_obs_step=jnp.zeros((32,), dtype=jnp.int32),
    )
    state, _obs, metrics_1 = env._apply_observation_pipeline(
        state, true_bpos, true_bvel
    )
    state = state._replace(step_count=state.step_count + 1)
    state, _obs, metrics_2 = env._apply_observation_pipeline(
        state, true_bpos, true_bvel
    )

    expected_available = (~episode_mask).astype(np.float32)
    np.testing.assert_array_equal(
        np.asarray(metrics_1["ball_obs_sample_available"]), expected_available
    )
    np.testing.assert_array_equal(
        np.asarray(metrics_2["ball_obs_sample_available"]), expected_available
    )
    np.testing.assert_array_equal(
        np.asarray(state.ball_obs_view_bounds_missing_enabled), episode_mask
    )

    for probability, expected in ((0.0, False), (1.0, True)):
        edge_env = MjxJuggleEnv(
            xml,
            n_envs=2,
            cfg=replace(
                cfg,
                ball_obs_camera_missing_prob=probability,
                ball_obs_view_bounds_missing_prob=probability,
                ball_obs_missing_episode_coherent_prob=probability,
            ),
        )
        edge_state, _obs = edge_env.reset(
            jax.random.split(jax.random.PRNGKey(int(probability) + 31), 2)
        )
        assert np.all(
            np.asarray(edge_state.ball_obs_camera_missing_enabled) == expected
        )
        assert np.all(
            np.asarray(edge_state.ball_obs_view_bounds_missing_enabled) == expected
        )
        assert np.all(
            np.asarray(edge_state.ball_obs_missing_episode_coherent_enabled)
            == expected
        )


def test_robust_juggle_v1_profile_contract() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleEnv  # noqa: E402
    from train_juggle_mjx_curriculum import build_curriculum  # noqa: E402

    stages = build_curriculum(curriculum_profile="robust_juggle_v1")
    assert [stage.name for stage in stages] == [
        "01_contact",
        "02_control",
        "03_range",
        "04_fov",
        "05_missing",
        "06_dynamics",
        "07_robust",
        "08_final",
    ]
    assert [stage.gate_mode for stage in stages] == ["balanced"] * 7 + ["strict"]
    assert all(stage.advance_gate_mode == "collapse" for stage in stages)
    camera_missing = [stage.cfg.ball_obs_camera_missing_prob for stage in stages]
    assert [
        stage.cfg.ball_obs_reset_respects_camera_visibility
        for stage in stages
    ] == [False, True, True, True, True, True, True, True]
    assert camera_missing == pytest.approx([0.0, 0.10, 0.25, 0.45, 0.70, 0.90, 1.0, 1.0])
    assert [stage.cfg.racket_z_hard_limit_up for stage in stages] == pytest.approx(
        [0.34, 0.32, 0.30, 0.28, 0.27, 0.26, 0.26, 0.26]
    )
    assert [stage.cfg.camera_visibility_penalty_weight for stage in stages] == pytest.approx(
        [1.0, 2.0, 4.0, 6.0, 8.0, 8.0, 8.0, 8.0]
    )
    assert [stage.cfg.camera_visible_penalty_weight for stage in stages] == pytest.approx(
        [0.0, 0.5, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    )
    assert [stage.cfg.hit_vxy_penalty_weight for stage in stages] == pytest.approx(
        [0.0] * 8
    )
    assert build_curriculum(curriculum_profile="standard")[0].cfg.hit_vxy_penalty_weight == pytest.approx(0.0)
    assert [stage.target_hit12_rate for stage in stages] == [
        None,
        None,
        None,
        None,
        0.05,
        0.20,
        0.52,
        0.82,
    ]
    assert stages[-1].cfg == stages[-2].cfg
    assert stages[-1].target_mean_hits == pytest.approx(13.0)
    assert stages[-1].target_mean_len_frac == pytest.approx(0.95)
    assert stages[-1].target_episode_truncation_rate == pytest.approx(0.90)
    assert stages[-1].target_hit12_rate == pytest.approx(0.82)
    assert stages[-1].target_hit_camera_visible_rate == pytest.approx(0.98)
    assert stages[-1].target_hit_camera_lower_band_rate == pytest.approx(0.90)
    assert stages[-1].target_min_hit_interval_s == pytest.approx(0.38)
    assert stages[-1].target_max_hit_interval_s == pytest.approx(0.50)
    assert stages[-1].cfg.target_height >= 0.20
    assert stages[-1].cfg.hit_height_center >= 0.24
    assert stages[-1].cfg.ball_view_z_ideal_m == pytest.approx((0.80, 1.28))
    assert stages[-1].cfg.rel_height_center == pytest.approx(0.24)
    assert stages[-1].cfg.hit_height_tolerance == pytest.approx(0.055)
    assert stages[-1].cfg.apex_soft_limit_margin == pytest.approx(0.10)
    assert not stages[2].cfg.dr_randomize_contact
    assert stages[3].cfg.dr_randomize_contact
    assert stages[3].cfg.dr_randomize_actuator_cmd_filter
    assert not stages[3].cfg.dr_randomize_actuator
    assert stages[4].cfg.dr_randomize_actuator
    assert stages[4].cfg.dr_randomize_pd
    assert not stages[4].cfg.dr_randomize_racket_mount
    assert not stages[4].cfg.dr_randomize_ball_obs_frame
    assert stages[5].cfg.dr_randomize_racket_mount
    assert stages[5].cfg.dr_randomize_ball_obs_frame
    assert stages[6].cfg.dr_actuator_cmd_tau_range == pytest.approx((0.060, 0.090))

    for stage in stages:
        cfg = stage.cfg
        assert cfg.enable_delay_conditioning
        assert cfg.include_tau_act_norm
        assert cfg.include_command_state
        assert cfg.include_active_command_error
        assert cfg.include_phase_features
        assert cfg.actuator_cmd_filter
        assert cfg.actuator_compensation_mode == "inverse_mpc"
        assert cfg.asymmetric_critic
        assert cfg.critic_command_history_steps >= 12
        assert cfg.ball_high_termination_z_m == pytest.approx(1.80)
        assert cfg.hit_combo_count_cap == 14
        assert cfg.hit_reward_count_cap == 15
        assert cfg.terminate_on_ball_view_bounds
        assert cfg.terminate_on_ball_view_x_bounds
        assert cfg.terminate_on_ball_view_y_bounds
        assert cfg.terminate_on_ball_view_z_low
        assert not cfg.terminate_on_ball_view_z_high
        assert cfg.terminate_on_racket_z_limit

    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=stages[-1].cfg)
    assert env.obs_dim == 67
    assert env.critic_obs_dim == 231
    assert env.max_steps == 1200


def test_robust_juggle_termination_semantics() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleEnv  # noqa: E402
    from train_juggle_mjx_curriculum import build_curriculum  # noqa: E402

    cfg = build_curriculum(curriculum_profile="robust_juggle_v1")[-1].cfg
    cfg = replace(cfg, ball_view_z_bounds_m=(0.62, 1.30))
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(7), 1))
    rpos = state.data.site_xpos[:, env.racket_site_id]
    bpos = state.data.xpos[:, env.ball_body_id]

    _done, terms = env._termination_terms(
        state.data,
        bpos.at[:, 2].set(1.80),
        rpos,
        state.racket_anchor,
    )
    assert not bool(np.asarray(terms["ball_too_high"])[0])

    done, terms = env._termination_terms(
        state.data,
        bpos.at[:, 2].set(1.801),
        rpos,
        state.racket_anchor,
    )
    assert bool(np.asarray(terms["ball_too_high"])[0])
    assert bool(np.asarray(done)[0])

    _done, terms = env._termination_terms(
        state.data,
        bpos.at[:, 2].set(1.40),
        rpos,
        state.racket_anchor,
    )
    assert not bool(np.asarray(terms["ball_view_z_too_high"])[0])
    assert not bool(np.asarray(terms["ball_too_high"])[0])


    base_x = float(np.asarray(state.data.qpos[0, env.base_x_qadr]))
    base_y = float(np.asarray(state.data.qpos[0, env.base_y_qadr]))
    base_yaw = float(np.asarray(state.data.qpos[0, env.base_yaw_qadr]))
    x_base = float(cfg.ball_view_x_bounds_m[1]) + 0.01
    y_base = 0.5 * sum(cfg.ball_view_y_bounds_m)
    world_x = base_x + np.cos(base_yaw) * x_base - np.sin(base_yaw) * y_base
    world_y = base_y + np.sin(base_yaw) * x_base + np.cos(base_yaw) * y_base
    x_high_bpos = (
        bpos.at[:, 0].set(world_x)
        .at[:, 1].set(world_y)
        .at[:, 2].set(1.0)
    )
    done, terms = env._termination_terms(state.data, x_high_bpos, rpos, state.racket_anchor)
    assert bool(np.asarray(terms["ball_view_x_too_high"])[0])
    assert bool(np.asarray(done)[0])
    done, terms = env._termination_terms(
        state.data,
        bpos.at[:, 2].set(0.60),
        rpos,
        state.racket_anchor,
    )
    assert bool(np.asarray(terms["ball_view_z_too_low"])[0])
    assert bool(np.asarray(done)[0])


def test_low_reset_missing_bridge_ramps_physical_camera_missing() -> None:
    from dataclasses import asdict

    pytest.importorskip("jax")
    pytest.importorskip("mujoco")

    from train_juggle_mjx_curriculum import build_curriculum  # noqa: E402

    stages = build_curriculum(
        curriculum_profile="standard_low_reset_robust15_missing_bridge",
    )
    by_name = {stage.name: stage for stage in stages}
    expected = {
        "stage4ad_contact_bridge_view_missing_no_z_high_terminate": 0.20,
        "stage4ae_contact_bridge_random_view_missing_height": 0.40,
        "stage4aef_contact_bridge_missing_475_no_z_high_terminate": 0.475,
        "stage4af_contact_bridge_missing_55_no_z_high_terminate": 0.55,
        "stage4b_contact_dr": 0.55,
        "stage4ag_first_hit_preservation_missing_55": 0.55,
        "stage4ah_episode_coherent_25_missing_55": 0.55,
        "stage4ai_episode_coherent_50_missing_55": 0.55,
        "stage4aia_episode_coherent_625_missing_55": 0.55,
        "stage4aj_episode_coherent_75_missing_55": 0.55,
        "stage4aj0_episode_coherent_8125_missing_55": 0.55,
        "stage4aja_episode_coherent_875_missing_55": 0.55,
        "stage4ajb_episode_coherent_9375_missing_55": 0.55,
        "stage4ak_episode_coherent_100_missing_55": 0.55,
        "stage4ak1_first_hit_intercept_18_missing_55": 0.55,
        "stage4akb_contact_missing_5625_bridge": 0.5625,
        "stage4al_contact_missing_575_bridge": 0.575,
        "stage4ba_contact_missing_60_bridge": 0.60,
        "stage4baa_contact_missing_6125_bridge": 0.6125,
        "stage4bab_contact_missing_625_bridge": 0.625,
        "stage4bac_contact_missing_6375_bridge": 0.6375,
        "stage4bb_contact_missing_65_bridge": 0.65,
        "stage4bc_lite_actuator_dr_frozen_probe": 0.65,
        "stage4c_lite_actuator_dr": 0.65,
        "stage4cb_actuator_missing_75_bridge": 0.75,
        "stage4d_latency_dr": 0.75,
        "stage4db_latency_missing_85_bridge": 0.85,
        "stage4e_racket_mount_dr": 0.85,
        "stage4f_final_dr_camera_dropout": 0.95,
        "stage4fb_final_missing_100_bridge": 1.0,
        "stage4g_strong_contact_dr": 1.0,
    }
    assert not by_name[
        "stage4ac_contact_bridge_mocap_no_z_high_terminate"
    ].cfg.ball_obs_require_camera_visible
    for name, probability in expected.items():
        cfg = by_name[name].cfg
        assert cfg.ball_obs_require_camera_visible
        assert cfg.ball_obs_require_view_bounds
        assert cfg.ball_obs_camera_missing_prob == pytest.approx(probability)
        assert cfg.ball_obs_view_bounds_missing_prob == pytest.approx(probability)
        assert cfg.ball_obs_reset_respects_camera_visibility
        assert cfg.ball_obs_age_tracks_stale
        assert not cfg.terminate_on_ball_view_z_high

    stage4ae_cfg = asdict(
        by_name["stage4ae_contact_bridge_random_view_missing_height"].cfg
    )
    stage4aef_cfg = asdict(
        by_name["stage4aef_contact_bridge_missing_475_no_z_high_terminate"].cfg
    )
    stage4af_cfg = asdict(
        by_name["stage4af_contact_bridge_missing_55_no_z_high_terminate"].cfg
    )
    assert {
        key for key in stage4ae_cfg if stage4ae_cfg[key] != stage4aef_cfg[key]
    } == {
        "ball_obs_camera_missing_prob",
        "ball_obs_view_bounds_missing_prob",
    }
    assert {
        key for key in stage4aef_cfg if stage4aef_cfg[key] != stage4af_cfg[key]
    } == {
        "ball_obs_camera_missing_prob",
        "ball_obs_view_bounds_missing_prob",
    }
    assert not by_name[
        "stage4af_contact_bridge_missing_55_no_z_high_terminate"
    ].cfg.terminate_on_racket_z_limit
    assert by_name["stage4cb_actuator_missing_75_bridge"].target_mean_hits == pytest.approx(5.2)
    assert by_name["stage4d_latency_dr"].target_mean_hits == pytest.approx(5.2)
    assert by_name["stage4db_latency_missing_85_bridge"].target_mean_hits == pytest.approx(5.0)
    assert by_name["stage4db_latency_missing_85_bridge"].min_recent_mean_return is None
    assert by_name["stage4e_racket_mount_dr"].target_mean_hits == pytest.approx(5.0)
    assert by_name["stage4e_racket_mount_dr"].min_recent_mean_return is None
    assert by_name["stage4f_final_dr_camera_dropout"].target_mean_hits == pytest.approx(4.6)
    assert by_name["stage4f_final_dr_camera_dropout"].target_hit3_rate == pytest.approx(0.53)
    assert by_name["stage4f_final_dr_camera_dropout"].target_mean_hits_ge3 == pytest.approx(7.8)
    assert by_name["stage4f_final_dr_camera_dropout"].min_recent_mean_return is None
    assert by_name["stage4fb_final_missing_100_bridge"].target_mean_hits == pytest.approx(4.6)
    assert by_name["stage4fb_final_missing_100_bridge"].target_hit3_rate == pytest.approx(0.53)
    assert by_name["stage4fb_final_missing_100_bridge"].target_mean_hits_ge3 == pytest.approx(7.8)
    assert by_name["stage4fb_final_missing_100_bridge"].min_recent_mean_return is None
    assert by_name["stage4g_strong_contact_dr"].target_mean_hits == pytest.approx(3.2)
    assert by_name["stage4g_strong_contact_dr"].target_mean_len_frac == pytest.approx(0.38)
    assert by_name["stage4g_strong_contact_dr"].target_hit3_rate == pytest.approx(0.38)
    assert by_name["stage4g_strong_contact_dr"].target_mean_hits_ge3 == pytest.approx(6.6)
    assert by_name["stage4g_strong_contact_dr"].min_recent_mean_return is None
    assert by_name["stage5a0_low_reset_height_noise_bridge"].target_mean_hits == pytest.approx(2.2)
    assert by_name["stage5a0_low_reset_height_noise_bridge"].target_mean_len_frac == pytest.approx(0.18)
    assert by_name["stage5a0_low_reset_height_noise_bridge"].target_hit1_rate == pytest.approx(0.60)
    assert by_name["stage5a0_low_reset_height_noise_bridge"].target_hit3_rate == pytest.approx(0.12)
    assert by_name["stage5a0_low_reset_height_noise_bridge"].min_recent_mean_return is None
    assert by_name["stage5a0_low_reset_height_noise_bridge"].cfg.ball_obs_pos_noise_std == pytest.approx(0.010)
    assert by_name["stage5a0_low_reset_height_noise_bridge"].cfg.ball_obs_vel_noise_std == pytest.approx(0.100)
    assert not by_name["stage5a0_low_reset_height_noise_bridge"].cfg.terminate_on_racket_z_limit
    assert by_name["stage5a1_low_reset_full_noise_bridge"].target_mean_hits == pytest.approx(2.0)
    assert by_name["stage5a1_low_reset_full_noise_bridge"].target_hit3_rate == pytest.approx(0.10)
    assert by_name["stage5a1_low_reset_full_noise_bridge"].cfg.ball_obs_pos_noise_std == pytest.approx(0.030)
    assert not by_name["stage5a1_low_reset_full_noise_bridge"].cfg.terminate_on_racket_z_limit
    assert by_name["stage5a2_low_reset_full_height_no_z_bridge"].target_mean_hits == pytest.approx(1.4)
    assert by_name["stage5a2_low_reset_full_height_no_z_bridge"].target_hit3_rate == pytest.approx(0.05)
    assert not by_name["stage5a2_low_reset_full_height_no_z_bridge"].cfg.terminate_on_racket_z_limit
    assert by_name["stage5a2_low_reset_full_height_no_z_bridge"].cfg.racket_z_hard_limit_up == pytest.approx(0.315)
    assert by_name["stage5a3_low_reset_z_soft_bridge"].target_mean_hits == pytest.approx(2.0)
    assert by_name["stage5a3_low_reset_z_soft_bridge"].target_hit3_rate == pytest.approx(0.20)
    assert not by_name["stage5a3_low_reset_z_soft_bridge"].cfg.terminate_on_racket_z_limit
    assert by_name["stage5a3_low_reset_z_soft_bridge"].cfg.racket_z_soft_penalty_weight == pytest.approx(3.0)
    assert by_name["stage5a3_low_reset_z_soft_bridge"].cfg.racket_z_hard_limit_up == pytest.approx(0.315)
    assert by_name["stage5a4_low_reset_racket_z_bridge"].target_mean_hits == pytest.approx(1.2)
    assert by_name["stage5a4_low_reset_racket_z_bridge"].target_hit3_rate == pytest.approx(0.05)
    assert not by_name["stage5a4_low_reset_racket_z_bridge"].cfg.terminate_on_racket_z_limit
    assert by_name["stage5a4_low_reset_racket_z_bridge"].cfg.racket_z_hard_limit_down == pytest.approx(0.12)
    assert by_name["stage5a4_low_reset_racket_z_bridge"].cfg.racket_z_hard_limit_up == pytest.approx(0.315)
    assert by_name["stage5a_low_reset_generalization_entry"].target_mean_hits == pytest.approx(2.15)
    assert by_name["stage5a_low_reset_generalization_entry"].target_mean_len_frac == pytest.approx(0.36)
    assert by_name["stage5a_low_reset_generalization_entry"].target_camera_visible == pytest.approx(0.83)
    assert by_name["stage5a_low_reset_generalization_entry"].target_hit1_rate == pytest.approx(0.70)
    assert by_name["stage5a_low_reset_generalization_entry"].target_hit3_rate == pytest.approx(0.30)
    assert by_name["stage5a_low_reset_generalization_entry"].min_recent_mean_return is None
    assert by_name["stage5a_low_reset_generalization_entry"].cfg.racket_z_hard_limit_down == pytest.approx(0.12)
    assert by_name["stage5a_low_reset_generalization_entry"].cfg.racket_z_hard_limit_up == pytest.approx(0.315)
    assert not by_name["stage5a_low_reset_generalization_entry"].cfg.terminate_on_racket_z_limit
    assert by_name["stage5a_low_reset_generalization_entry"].cfg.racket_z_soft_penalty_weight == pytest.approx(3.0)
    for name, target_hits, target_hit3 in (
        ("stage5ab_low_reset_generalization_bridge", 1.95, 0.25),
        ("stage5b_low_reset_generalization_mid", 1.65, 0.20),
        ("stage5c_low_reset_generalization_outer", 1.30, 0.14),
        ("stage5d_low_reset_generalization_wide", 1.00, 0.08),
    ):
        assert by_name[name].target_mean_hits == pytest.approx(target_hits)
        assert by_name[name].target_hit3_rate == pytest.approx(target_hit3)
        assert by_name[name].min_recent_mean_return is None
        assert not by_name[name].cfg.terminate_on_racket_z_limit
        assert by_name[name].cfg.racket_z_soft_penalty_weight == pytest.approx(3.0)
        assert by_name[name].cfg.racket_z_hard_limit_up == pytest.approx(0.315)
    for name, target_hits, target_len, target_hit1, target_hit3, target_hge3 in (
        ("stage5e_low_reset_wide_survival_len35_soft", 0.58, 0.095, 0.36, 0.035, 4.0),
        ("stage5e1_low_reset_entry_apex_len38_soft", 0.50, 0.095, 0.33, 0.025, 4.0),
        ("stage5e1b_low_reset_first_hit_bridge_len38_soft", 0.51, 0.098, 0.335, 0.030, 4.0),
        ("stage5e2a_low_reset_view_recenter_len45_soft", 0.51, 0.098, 0.335, 0.027, 4.2),
        ("stage5e2b_low_reset_long_soft_len70", 0.70, 0.115, 0.36, 0.040, 4.5),
        ("stage5e3_low_reset_visible_pre_hard_len85_soft", 1.00, 0.16, 0.40, 0.070, 5.0),
    ):
        assert by_name[name].target_mean_hits == pytest.approx(target_hits)
        assert by_name[name].target_mean_len_frac == pytest.approx(target_len)
        assert by_name[name].target_hit1_rate == pytest.approx(target_hit1)
        assert by_name[name].target_hit3_rate == pytest.approx(target_hit3)
        assert by_name[name].target_mean_hits_ge3 == pytest.approx(target_hge3)
    assert by_name["stage5h_low_reset_wide_final_acceptance_len95"].target_mean_hits == pytest.approx(13.0)
    assert by_name["stage5h_low_reset_wide_final_acceptance_len95"].target_mean_len_frac == pytest.approx(0.95)
    stage4b_cfg = asdict(by_name["stage4b_contact_dr"].cfg)
    assert stage4af_cfg == stage4b_cfg
    assert not by_name["stage4b_contact_dr"].cfg.terminate_on_racket_z_limit
    stage4ag_cfg = asdict(by_name["stage4ag_first_hit_preservation_missing_55"].cfg)
    assert {
        key for key in stage4b_cfg if stage4b_cfg[key] != stage4ag_cfg[key]
    } == {
        "pre_hit_intercept_reward_weight",
        "pre_hit_intercept_sigma",
        "pre_hit_intercept_time_max",
        "pre_hit_intercept_penalty_weight",
        "pre_hit_intercept_penalty_sigma",
        "termination_no_hit_miss_early_penalty",
        "first_hit_apex_reward_weight",
        "first_hit_apex_sigma",
    }
    coherent_stages = [
        ("stage4ah_episode_coherent_25_missing_55", 0.25),
        ("stage4ai_episode_coherent_50_missing_55", 0.50),
        ("stage4aia_episode_coherent_625_missing_55", 0.625),
        ("stage4aj_episode_coherent_75_missing_55", 0.75),
        ("stage4aj0_episode_coherent_8125_missing_55", 0.8125),
        ("stage4aja_episode_coherent_875_missing_55", 0.875),
        ("stage4ajb_episode_coherent_9375_missing_55", 0.9375),
        ("stage4ak_episode_coherent_100_missing_55", 1.00),
    ]
    previous_cfg = stage4ag_cfg
    for name, probability in coherent_stages:
        coherent_cfg = asdict(by_name[name].cfg)
        assert {
            key for key in previous_cfg if previous_cfg[key] != coherent_cfg[key]
        } == {"ball_obs_missing_episode_coherent_prob"}
        assert coherent_cfg["ball_obs_missing_episode_coherent_prob"] == pytest.approx(
            probability
        )
        assert by_name[name].target_hit1_rate == pytest.approx(0.80)
        previous_cfg = coherent_cfg
    assert stage4ag_cfg["ball_obs_missing_episode_coherent_prob"] == pytest.approx(0.0)
    q100_cfg = asdict(by_name["stage4ak_episode_coherent_100_missing_55"].cfg)
    recovery_cfg = asdict(by_name["stage4ak1_first_hit_intercept_18_missing_55"].cfg)
    assert {key for key in q100_cfg if q100_cfg[key] != recovery_cfg[key]} == {
        "pre_hit_intercept_reward_weight"
    }
    assert recovery_cfg["pre_hit_intercept_reward_weight"] == pytest.approx(1.80)
    assert by_name["stage4ak1_first_hit_intercept_18_missing_55"].target_hit1_rate == pytest.approx(
        0.80
    )
    assert by_name["stage4akb_contact_missing_5625_bridge"].target_hit1_rate == pytest.approx(
        0.80
    )
    assert by_name["stage4akb_contact_missing_5625_bridge"].target_mean_hits == pytest.approx(
        6.10
    )
    assert by_name["stage4al_contact_missing_575_bridge"].target_mean_hits == pytest.approx(
        6.50
    )
    assert by_name["stage4al_contact_missing_575_bridge"].target_hit1_rate == pytest.approx(
        0.80
    )
    assert not by_name["stage4al_contact_missing_575_bridge"].policy_updates_enabled
    assert by_name["stage4akb_contact_missing_5625_bridge"].policy_updates_enabled
    assert by_name["stage4ba_contact_missing_60_bridge"].policy_updates_enabled
    for name in (
        "stage4baa_contact_missing_6125_bridge",
        "stage4bab_contact_missing_625_bridge",
        "stage4bac_contact_missing_6375_bridge",
        "stage4bb_contact_missing_65_bridge",
    ):
        assert not by_name[name].policy_updates_enabled
    assert by_name["stage4ba_contact_missing_60_bridge"].target_hit1_rate == pytest.approx(
        0.80
    )
    assert by_name["stage4ba_contact_missing_60_bridge"].target_mean_hits == pytest.approx(
        6.30
    )
    assert by_name["stage4bb_contact_missing_65_bridge"].target_hit1_rate == pytest.approx(
        0.84
    )
    for name in (
        "stage4baa_contact_missing_6125_bridge",
        "stage4bab_contact_missing_625_bridge",
        "stage4bac_contact_missing_6375_bridge",
    ):
        assert by_name[name].target_mean_hits == pytest.approx(6.30)
        assert by_name[name].target_hit1_rate == pytest.approx(0.80)
    for name in expected:
        if name in {
            "stage4ad_contact_bridge_view_missing_no_z_high_terminate",
            "stage4ae_contact_bridge_random_view_missing_height",
            "stage4aef_contact_bridge_missing_475_no_z_high_terminate",
            "stage4af_contact_bridge_missing_55_no_z_high_terminate",
            "stage4b_contact_dr",
            "stage4ag_first_hit_preservation_missing_55",
        }:
            assert by_name[name].cfg.ball_obs_missing_episode_coherent_prob == pytest.approx(0.0)
        else:
            expected_coherent = dict(coherent_stages).get(name, 1.0)
            assert by_name[name].cfg.ball_obs_missing_episode_coherent_prob == pytest.approx(
                expected_coherent
            )
    for name in (
        "stage4bb_contact_missing_65_bridge",
        "stage4c_lite_actuator_dr",
        "stage4cb_actuator_missing_75_bridge",
        "stage4d_latency_dr",
        "stage4db_latency_missing_85_bridge",
        "stage4e_racket_mount_dr",
        "stage4f_final_dr_camera_dropout",
        "stage4fb_final_missing_100_bridge",
        "stage4g_strong_contact_dr",
    ):
        assert not by_name[name].cfg.terminate_on_racket_z_limit

    for source_name, bridge_name in (
        ("stage4ak1_first_hit_intercept_18_missing_55", "stage4akb_contact_missing_5625_bridge"),
        ("stage4akb_contact_missing_5625_bridge", "stage4al_contact_missing_575_bridge"),
        ("stage4al_contact_missing_575_bridge", "stage4ba_contact_missing_60_bridge"),
        ("stage4ba_contact_missing_60_bridge", "stage4baa_contact_missing_6125_bridge"),
        ("stage4baa_contact_missing_6125_bridge", "stage4bab_contact_missing_625_bridge"),
        ("stage4bab_contact_missing_625_bridge", "stage4bac_contact_missing_6375_bridge"),
        ("stage4bac_contact_missing_6375_bridge", "stage4bb_contact_missing_65_bridge"),
        ("stage4c_lite_actuator_dr", "stage4cb_actuator_missing_75_bridge"),
        ("stage4d_latency_dr", "stage4db_latency_missing_85_bridge"),
        ("stage4f_final_dr_camera_dropout", "stage4fb_final_missing_100_bridge"),
    ):
        source_cfg = asdict(by_name[source_name].cfg)
        bridge_cfg = asdict(by_name[bridge_name].cfg)
        assert {
            key for key in source_cfg if source_cfg[key] != bridge_cfg[key]
        } == {
            "ball_obs_camera_missing_prob",
            "ball_obs_view_bounds_missing_prob",
        }

    p65_cfg = asdict(by_name["stage4bb_contact_missing_65_bridge"].cfg)
    actuator_probe = by_name["stage4bc_lite_actuator_dr_frozen_probe"]
    actuator_probe_cfg = asdict(actuator_probe.cfg)
    actuator_cfg = asdict(by_name["stage4c_lite_actuator_dr"].cfg)
    assert not actuator_probe.policy_updates_enabled
    assert actuator_probe_cfg == actuator_cfg
    assert p65_cfg["pre_hit_intercept_reward_weight"] == pytest.approx(1.80)
    assert actuator_cfg["pre_hit_intercept_reward_weight"] == pytest.approx(1.80)
    assert {
        key for key in p65_cfg if p65_cfg[key] != actuator_cfg[key]
    } == {
        "dr_randomize_actuator",
        "dr_action_scale_mult_range",
        "dr_armature_mult_range",
        "dr_damping_mult_range",
        "dr_randomize_pd",
        "dr_pd_kp_mult_range",
        "dr_pd_kv_mult_range",
        "actuator_cmd_filter",
        "actuator_cmd_tau",
        "dr_randomize_actuator_cmd_filter",
        "dr_actuator_cmd_tau_range",
        "dr_actuator_cmd_gain_range",
    }

    final_polish = by_name["stage5g_low_reset_wide_polish_len85"]
    final_acceptance = by_name["stage5h_low_reset_wide_final_acceptance_len95"]
    assert asdict(final_polish.cfg) == asdict(final_acceptance.cfg)
    assert final_acceptance.target_mean_hits == pytest.approx(13.0)
    assert final_acceptance.target_mean_len_frac == pytest.approx(0.95)
    assert final_acceptance.target_episode_truncation_rate == pytest.approx(0.90)
    assert final_acceptance.target_hit12_rate == pytest.approx(0.65)


def test_hit_camera_rollout_metrics_are_event_conditioned() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")
    from types import SimpleNamespace

    from train_juggle_mjx_curriculum import mean_rollout_metrics  # noqa: E402

    transitions = SimpleNamespace(
        metrics={
            "truncated": jnp.asarray([[0.0, 1.0], [0.0, 0.0]]),
            "ball_obs_refresh_due": jnp.ones((2, 2)),
            "ball_obs_missing_on_refresh": jnp.zeros((2, 2)),
            "hit_camera_event": jnp.asarray([[1.0, 0.0], [1.0, 1.0]]),
            "hit_camera_visible_event": jnp.asarray([[1.0, 0.0], [1.0, 0.0]]),
            "hit_camera_in_margin_event": jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "hit_camera_lower_band_event": jnp.asarray([[1.0, 0.0], [0.0, 0.0]]),
            "hit_camera_v_frac_sum": jnp.asarray([[0.60, 0.0], [0.70, 0.0]]),
            "hit_vxy_sum": jnp.asarray([[0.30, 0.0], [0.45, 0.60]]),
        },
        done=jnp.asarray([[False, True], [False, False]]),
    )
    metrics = mean_rollout_metrics(transitions)
    assert metrics["hit_camera_visible_rate"] == pytest.approx(2.0 / 3.0)
    assert metrics["hit_camera_in_margin_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["hit_camera_lower_band_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["mean_hit_camera_v_frac"] == pytest.approx(0.65)
    assert metrics["mean_hit_vxy"] == pytest.approx(0.45)


def test_balanced_and_strict_curriculum_gates() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from types import SimpleNamespace

    from train_juggle_mjx_curriculum import (  # noqa: E402
        build_curriculum,
        convergence_status,
    )

    stages = build_curriculum(curriculum_profile="robust_juggle_v1")
    args = SimpleNamespace(
        convergence_min_episodes=1,
        convergence_window=1,
        min_stage_updates=1,
        advance_mode="converged",
    )

    def passing_row(stage):
        return {
            "episodes": 32,
            "mean_hits": stage.target_mean_hits,
            "mean_len": 1200.0 * stage.target_mean_len_frac,
            "mean_return": 1.0,
            "camera_visible": stage.target_camera_visible or 1.0,
            "reward/camera_reward_dense": 0.0,
            "ball_view_in_bounds": stage.target_ball_view_in_bounds or 1.0,
            "ball_view_z_ideal": stage.target_ball_view_z_ideal or 1.0,
            "hit1_rate": stage.target_hit1_rate or 1.0,
            "hit3_rate": stage.target_hit3_rate or 1.0,
            "hit12_rate": stage.target_hit12_rate or 1.0,
            "mean_hits_ge3": stage.target_mean_hits_ge3 or stage.target_mean_hits,
            "mean_hit_interval_s": 0.44,
            "mean_hit_interval_ge3_s": 0.44,
            "episode_truncation_rate": stage.target_episode_truncation_rate or 1.0,
            "ball_obs_missing_refresh_rate": stage.min_ball_obs_missing_refresh_rate or 0.0,
            "ball_obs_lost_active": 0.0,
            "hit_camera_visible_rate": stage.target_hit_camera_visible_rate or 1.0,
            "hit_camera_lower_band_rate": stage.target_hit_camera_lower_band_rate or 1.0,
            "mean_hit_camera_v_frac": 0.67,
        }

    balanced = stages[6]
    row = passing_row(balanced)
    row["camera_visible"] = float(balanced.target_camera_visible) - 0.02
    status = convergence_status([row], balanced, SimpleNamespace(max_steps=1200), args, 100)
    assert status["convergence/stage_converged"] == pytest.approx(1.0)
    row = passing_row(balanced)
    row["ball_view_z_ideal"] = 0.0
    status = convergence_status(
        [row],
        balanced,
        SimpleNamespace(max_steps=1200),
        args,
        100,
    )
    assert status["convergence/stage_converged"] == pytest.approx(0.0)
    assert status["convergence/balanced_floor_ok"] == pytest.approx(0.0)



    row = passing_row(balanced)
    row["mean_hits"] = 0.70 * balanced.target_mean_hits
    status = convergence_status([row], balanced, SimpleNamespace(max_steps=1200), args, 100)
    assert status["convergence/stage_converged"] == pytest.approx(0.0)
    assert status["convergence/balanced_floor_ok"] == pytest.approx(0.0)

    strict = stages[-1]
    row = passing_row(strict)
    row["camera_visible"] = float(strict.target_camera_visible) - 0.01
    status = convergence_status([row], strict, SimpleNamespace(max_steps=1200), args, 100)
    assert status["convergence/stage_converged"] == pytest.approx(0.0)


def test_zero_hit_camera_nans_do_not_trigger_safety_stop() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from types import SimpleNamespace

    from train_juggle_mjx_curriculum import metric_safety_stop_reason  # noqa: E402

    args = SimpleNamespace(
        safe_stop=True,
        max_abs_mean_return=1.0e6,
        max_loss=1.0e6,
        max_grad_norm_alert=1.0e6,
        max_abs_reward_metric=1.0e6,
    )
    zero_hit_row = {
        "episodes": 0,
        "hit_camera_visible_rate": float("nan"),
        "hit_camera_in_margin_rate": float("nan"),
        "hit_camera_lower_band_rate": float("nan"),
        "mean_hit_camera_v_frac": float("nan"),
        "mean_hit_vxy": float("nan"),
    }
    assert metric_safety_stop_reason(zero_hit_row, args) is None

    optimizer_failure = dict(zero_hit_row)
    optimizer_failure["loss"] = float("nan")
    reason = metric_safety_stop_reason(optimizer_failure, args)
    assert reason is not None
    assert "loss" in reason


def test_validator_episode_metrics_are_globally_weighted() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")

    from validate_juggle_mjx_ppo import (  # noqa: E402
        add_terminal_step_metrics,
        aggregate_episode_metrics,
    )

    rows = [
        {
            "length": 10.0,
            "hit_camera_events": 1.0,
            "hit_camera_visible_events": 1.0,
            "hit_camera_in_margin_events": 1.0,
            "hit_camera_lower_band_events": 1.0,
            "hit_camera_v_frac_sum": 0.60,
            "hit_vxy_sum": 0.30,
            "camera_visible_steps": 10.0,
            "ball_view_z_ideal_steps": 10.0,
            "ball_obs_refresh_count": 2.0,
            "ball_obs_missing_refresh_count": 1.0,
            "ball_obs_reacquired_after_missing_events": 1.0,
            "ball_obs_missing_exposure_count": 1.0,
            "ball_obs_reacquired_after_lost_events": 1.0,
            "ball_obs_lost_exposure_count": 1.0,
        },
        {
            "length": 90.0,
            "hit_camera_events": 9.0,
            "hit_camera_visible_events": 0.0,
            "hit_camera_in_margin_events": 0.0,
            "hit_camera_lower_band_events": 0.0,
            "hit_camera_v_frac_sum": 0.0,
            "hit_vxy_sum": 5.40,
            "camera_visible_steps": 0.0,
            "ball_view_z_ideal_steps": 45.0,
            "ball_obs_refresh_count": 8.0,
            "ball_obs_missing_refresh_count": 1.0,
            "ball_obs_reacquired_after_missing_events": 1.0,
            "ball_obs_missing_exposure_count": 3.0,
            "ball_obs_reacquired_after_lost_events": 0.0,
            "ball_obs_lost_exposure_count": 1.0,
        },
    ]
    metrics = aggregate_episode_metrics(rows)
    assert metrics["hit_camera_visible_rate"] == pytest.approx(0.10)
    assert metrics["hit_camera_lower_band_rate"] == pytest.approx(0.10)
    assert metrics["camera_visible_rate"] == pytest.approx(0.10)
    assert metrics["ball_view_z_ideal_rate"] == pytest.approx(0.55)
    assert metrics["ball_obs_missing_refresh_rate"] == pytest.approx(0.20)
    assert metrics["mean_hit_camera_v_frac"] == pytest.approx(0.60)
    assert metrics["mean_hit_vxy"] == pytest.approx(0.57)
    assert metrics["ball_obs_reacquired_after_missing_rate"] == pytest.approx(0.50)
    assert metrics["ball_obs_reacquired_after_lost_rate"] == pytest.approx(0.50)

    terminal_row = {"hit_camera_v_frac_sum": 1.25}
    terminal_host = {
        "hit_camera_v_frac_sum": np.asarray([0.20]),
        "ball_obs_lost_active": np.asarray([1.0]),
    }
    add_terminal_step_metrics(terminal_row, terminal_host, 0)
    assert terminal_row["hit_camera_v_frac_sum"] == pytest.approx(1.25)
    assert terminal_row["last/hit_camera_v_frac_sum"] == pytest.approx(0.20)
    assert terminal_row["ball_obs_lost_active"] == pytest.approx(1.0)
