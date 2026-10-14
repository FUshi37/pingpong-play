from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RL_SIM_DIR = ROOT / "pingpong_controller" / "tools" / "rl_sim"
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))


def test_unknown_ball_reset_mode_is_rejected() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv

    with pytest.raises(ValueError, match="unknown ball_reset_mode"):
        MjxJuggleEnv(
            RL_SIM_DIR / "moz1_pd.xml",
            n_envs=1,
            cfg=MjxJuggleConfig(ball_reset_mode="not_a_reset_mode"),
        )


def test_racket_launch_reset_uses_surface_relative_geometry() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    mujoco = pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv

    cfg = MjxJuggleConfig(
        domain_randomization=False,
        ball_reset_mode="racket_launch",
        racket_launch_surface_gap_range_m=(0.005, 0.010),
        racket_launch_xy_jitter=0.004,
        racket_launch_vxy_max=0.003,
        racket_launch_vnormal_max=0.003,
        racket_launch_edge_margin=0.005,
    )
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=8, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(20260716), 8))

    racket_xmat = state.data.geom_xmat[:, env.racket_geom_id].reshape((-1, 3, 3))
    normal = racket_xmat[:, :, 2]
    normal = normal * jnp.where(normal[:, 2] >= 0.0, 1.0, -1.0)[:, None]
    racket_surface = (
        state.data.geom_xpos[:, env.racket_geom_id]
        + normal * state.model.geom_size[:, env.racket_geom_id, 1][:, None]
    )
    ball_center = state.data.xpos[:, env.ball_body_id]
    ball_radius = state.model.geom_size[:, env.ball_geom_id, 0]
    center_distance = jnp.sum((ball_center - racket_surface) * normal, axis=-1)
    surface_gap = center_distance - ball_radius

    gap_np = np.asarray(surface_gap)
    center_distance_np = np.asarray(center_distance)
    np.testing.assert_allclose(
        gap_np,
        np.asarray(state.reset_ball_surface_gap),
        atol=2e-6,
    )
    assert float(gap_np.min()) >= 0.005 - 2e-6
    assert float(gap_np.max()) <= 0.010 + 2e-6
    assert float(center_distance_np.min()) >= 0.025 - 2e-6
    assert float(center_distance_np.max()) <= 0.030 + 2e-6

    racket_radius = np.asarray(state.model.geom_size[:, env.racket_geom_id, 0])
    ball_radius_np = np.asarray(ball_radius)
    max_supported_center_offset = (
        racket_radius - ball_radius_np - cfg.racket_launch_edge_margin
    )
    assert np.all(
        np.asarray(state.reset_ball_racket_center_offset)
        <= max_supported_center_offset + 2e-6
    )
    assert float(np.linalg.norm(np.asarray(state.reset_ball_vel), axis=-1).max()) <= 0.006
    # Every reset component is normalized by its active sampler limit.  In
    # particular, racket-launch vertical velocity must not be divided by the
    # inactive legacy ball_init_vz_jitter (zero in this configuration).
    disturbance_strength = np.asarray(state.reset_disturbance_strength)
    assert np.all(np.isfinite(disturbance_strength))
    assert float(disturbance_strength.max()) < 4.0

    # The requested positive 5--10 mm gap is not contact at frame zero.  Under
    # gravity, every sampled ball must reach only the racket face within the
    # corresponding 32--46 ms free-fall window.
    for env_index in range(4):
        data = mujoco.MjData(env.mj_model)
        data.qpos[:] = np.asarray(state.data.qpos[env_index])
        data.qvel[:] = np.asarray(state.data.qvel[env_index])
        data.ctrl[:] = np.asarray(env.default_ctrl)
        mujoco.mj_forward(env.mj_model, data)
        first_racket_contact_step = None
        other_ball_contact = False
        for physics_step in range(60):
            mujoco.mj_step(env.mj_model, data)
            for contact in data.contact:
                geom_pair = {int(contact.geom1), int(contact.geom2)}
                if geom_pair == {env.ball_geom_id, env.racket_geom_id}:
                    if first_racket_contact_step is None:
                        first_racket_contact_step = physics_step + 1
                elif env.ball_geom_id in geom_pair:
                    other_ball_contact = True
        assert first_racket_contact_step is not None
        assert first_racket_contact_step <= 60
        assert not other_ball_contact


def test_anchor_drop_reset_keeps_legacy_surface_metrics_inactive() -> None:
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco")

    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv

    cfg = MjxJuggleConfig(
        domain_randomization=False,
        ball_reset_mode="anchor_drop",
        ball_launch_height=0.32,
        ball_spawn_xy_jitter=0.025,
        ball_spawn_z_jitter=0.035,
        ball_init_vxy_max=0.012,
        ball_init_vz=-0.28,
        ball_init_vz_jitter=0.0,
    )
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=4, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(7), 4))

    np.testing.assert_allclose(np.asarray(state.reset_ball_surface_gap), 0.0)
    np.testing.assert_allclose(np.asarray(state.reset_ball_vel[:, 2]), -0.28, atol=1e-6)
    reset_height = np.asarray(state.reset_ball_pos[:, 2] - state.racket_anchor[:, 2])
    assert float(reset_height.min()) >= 0.32 - 0.035 - 2e-6
    assert float(reset_height.max()) <= 0.32 + 0.035 + 2e-6
