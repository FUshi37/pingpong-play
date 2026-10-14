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
    delay_steps_from_tau,
    estimate_contact_time,
    push_command_buffer,
    smooth_action,
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


def test_contact_time_simple_ballistic_case() -> None:
    t_contact = estimate_contact_time(
        z_rel=0.20,
        vz_rel=-0.50,
        gravity=9.81,
        contact_height_offset=0.0,
        max_contact_time=0.5,
    )
    assert 0.0 < t_contact < 0.5


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
