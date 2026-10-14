from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RL_SIM_DIR = ROOT / "pingpong_controller" / "tools" / "rl_sim"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from sim2real_bridger import (  # noqa: E402
    correction_relative_governor_step_jax,
    correction_relative_governor_step_numpy,
    constrained_compensation_step_jax,
    constrained_compensation_step_numpy,
)


def test_correction_relative_governor_bounds_total_command_and_matches_jax() -> None:
    dt = 0.005
    nominal = np.deg2rad(np.asarray([20.1, -20.1, 0.1, 20.1, -0.1, 0.1, -0.1]))
    previous_nominal = np.deg2rad(np.asarray([20.0, -20.0, 0.0, 20.0, 0.0, 0.0, 0.0]))
    previous_nominal_vel = (nominal - previous_nominal) / dt
    previous_sent = previous_nominal.copy()
    previous_sent_vel = previous_nominal_vel.copy()
    raw = nominal + np.deg2rad(np.asarray([12.0, -12.0, 12.0, -12.0, 12.0, -12.0, 12.0]))
    low = np.full(7, -3.0)
    high = np.full(7, 3.0)
    vmax = np.deg2rad(np.full(7, 300.0))
    amax = np.deg2rad(np.full(7, 3000.0))

    numpy_step = correction_relative_governor_step_numpy(
        raw, nominal, previous_nominal, previous_nominal_vel,
        previous_sent, previous_sent_vel, low, high, vmax, amax, dt=dt,
    )
    jax_step = correction_relative_governor_step_jax(
        raw, nominal, previous_nominal, previous_nominal_vel,
        previous_sent, previous_sent_vel, low, high, vmax, amax, dt=dt,
    )
    for numpy_value, jax_value in zip(numpy_step, jax_step):
        np.testing.assert_allclose(numpy_value, np.asarray(jax_value), atol=5e-6)
    assert np.all(np.abs(numpy_step.qvel) <= vmax + 1e-8)
    assert np.all(np.abs(numpy_step.qacc) <= amax + 1e-8)
    assert np.all(numpy_step.q >= low)
    assert np.all(numpy_step.q <= high)

    # A replayed nominal command may itself contain an infeasible discrete
    # jump.  The final q path must still obey the absolute drive limits.
    jumped_nominal = nominal + np.deg2rad(30.0)
    jumped = correction_relative_governor_step_numpy(
        jumped_nominal,
        jumped_nominal,
        nominal,
        previous_nominal_vel,
        numpy_step.q,
        numpy_step.qvel,
        low,
        high,
        vmax,
        amax,
        dt=dt,
    )
    assert np.all(np.abs(jumped.qvel) <= vmax + 1e-8)
    assert np.all(np.abs(jumped.qacc) <= amax + 1e-8)


from sim2real_bridger_preview_mpc import run_receding_preview_compensation  # noqa: E402
from pingpong_controller.safety_limiter import RightArmCommandSafetyLimiter  # noqa: E402


def _limits() -> tuple[np.ndarray, ...]:
    limiter = RightArmCommandSafetyLimiter
    return (
        limiter.POS_LIMIT_LOW_RAD.astype(np.float64),
        limiter.POS_LIMIT_HIGH_RAD.astype(np.float64),
        np.deg2rad(limiter.VEL_LIMIT_DEG_S).astype(np.float64),
        np.deg2rad(limiter.ACC_LIMIT_DEG_S2).astype(np.float64),
        np.full(7, np.deg2rad(175_000.0), dtype=np.float64),
    )


def test_bridger_aggressive_rollout_is_recursively_hard_feasible() -> None:
    low, high, velocity_limit, acceleration_limit, jerk_limit = _limits()
    dt = 0.005
    q = 0.5 * (low + high)
    qvel = np.zeros(7)
    qacc = np.zeros(7)
    q_history = []
    qvel_history = []
    qacc_history = []

    for step_index in range(2_000):
        target = np.where((step_index // 120) % 2 == 0, high - 0.01, low + 0.01)
        step = constrained_compensation_step_numpy(
            target,
            np.zeros(7),
            q,
            qvel,
            qacc,
            low,
            high,
            velocity_limit,
            acceleration_limit,
            jerk_limit,
            dt=dt,
            natural_frequency_hz=12.0,
        )
        assert bool(step.feasible)
        assert np.all(step.jerk_feasible)
        q, qvel, qacc = step.q, step.qvel, step.qacc
        q_history.append(q)
        qvel_history.append(qvel)
        qacc_history.append(qacc)

    q_array = np.asarray(q_history)
    velocity_array = np.asarray(qvel_history)
    acceleration_array = np.asarray(qacc_history)
    jerk_array = np.diff(
        np.vstack([np.zeros((1, 7)), acceleration_array]), axis=0
    ) / dt
    assert np.all(q_array >= low - 1e-10)
    assert np.all(q_array <= high + 1e-10)
    assert np.all(np.abs(velocity_array) <= velocity_limit + 1e-8)
    assert np.all(np.abs(acceleration_array) <= acceleration_limit + 1e-8)
    assert np.all(np.abs(jerk_array) <= jerk_limit + 1e-8)


def test_bridger_numpy_and_batched_jax_are_equivalent() -> None:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    pytest.importorskip("jax")
    rng = np.random.default_rng(20260723)
    low, high, velocity_limit, acceleration_limit, jerk_limit = _limits()
    q = rng.uniform(low * 0.4, high * 0.4, size=(16, 7))
    qvel = rng.uniform(-0.2 * velocity_limit, 0.2 * velocity_limit, size=(16, 7))
    qacc = rng.uniform(-0.1 * acceleration_limit, 0.1 * acceleration_limit, size=(16, 7))
    target = rng.uniform(low * 0.8, high * 0.8, size=(16, 7))
    target_velocity = rng.uniform(-velocity_limit, velocity_limit, size=(16, 7))
    numpy_step = constrained_compensation_step_numpy(
        target,
        target_velocity,
        q,
        qvel,
        qacc,
        low,
        high,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        dt=0.005,
        natural_frequency_hz=12.0,
    )
    jax_step = constrained_compensation_step_jax(
        target,
        target_velocity,
        q,
        qvel,
        qacc,
        low,
        high,
        velocity_limit,
        acceleration_limit,
        jerk_limit,
        dt=0.005,
        natural_frequency_hz=12.0,
    )
    for numpy_value, jax_value in zip(numpy_step[:3], jax_step[:3]):
        np.testing.assert_allclose(numpy_value, np.asarray(jax_value), rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(numpy_step.feasible, np.asarray(jax_step[3]))
    np.testing.assert_array_equal(numpy_step.jerk_feasible, np.asarray(jax_step[4]))


def test_mjx_bridger_rejects_downstream_governor() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv

    cfg = MjxJuggleConfig(
        domain_randomization=False,
        actuator_compensation_mode="sim2real_bridger",
        arm_actual_state_limiter=True,
        arm_actual_target_tracking_governor=True,
    )
    with pytest.raises(ValueError, match="must be disabled"):
        MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=cfg)


def test_finite_preview_mpc_satisfies_all_command_constraints() -> None:
    pytest.importorskip("scipy")
    low, high, velocity_limit, acceleration_limit, jerk_limit = _limits()
    times = np.arange(80) * 0.005
    reference = 0.5 * (low + high) + 0.08 * np.sin(
        2.0 * np.pi * 1.5 * times[:, None] + np.arange(7)[None, :] * 0.2
    )
    result = run_receding_preview_compensation(
        reference,
        preview_steps=30,
        execute_steps=10,
        dt=0.005,
        delay_steps=14,
        actuator_tau_s=0.074,
        pos_low=low,
        pos_high=high,
        velocity_limit_rad_s=velocity_limit,
        acceleration_limit_rad_s2=acceleration_limit,
        jerk_limit_rad_s3=jerk_limit,
    )
    assert np.all(result.command_q >= low - 1e-9)
    assert np.all(result.command_q <= high + 1e-9)
    assert np.all(np.abs(result.command_qvel) <= velocity_limit + 1e-8)
    assert np.all(np.abs(result.command_qacc) <= acceleration_limit + 1e-8)
    assert np.all(np.abs(result.command_jerk) <= jerk_limit + 1e-8)
