from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RL_SIM_DIR = ROOT / "pingpong_controller" / "tools" / "rl_sim"
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))


def _compute(**kwargs):
    jax = pytest.importorskip("jax")
    from train_juggle_mjx_ppo import compute_gae

    result = compute_gae(**kwargs)
    return tuple(np.asarray(jax.device_get(value)) for value in result)


def test_time_limit_bootstraps_terminal_physical_state_but_cuts_trace() -> None:
    jnp = pytest.importorskip("jax.numpy")
    advantages, returns = _compute(
        rewards=jnp.asarray([[0.0], [0.0]]),
        dones=jnp.asarray([[False], [True]]),
        values=jnp.asarray([[1.0], [2.0]]),
        last_value=jnp.asarray([3.0]),
        gamma=0.9,
        gae_lambda=1.0,
        terminated=jnp.asarray([[False], [False]]),
        truncated=jnp.asarray([[False], [True]]),
        timeout_values=jnp.asarray([[0.0], [10.0]]),
        time_limit_bootstrap=True,
    )

    # t=1: 0 + .9*10 - 2 = 7.  The trace stops at done, so no reset episode
    # advantage can leak backward.  t=0 then sees the ordinary next value 2.
    np.testing.assert_allclose(advantages[:, 0], [7.1, 7.0], rtol=1e-6)
    np.testing.assert_allclose(returns[:, 0], [8.1, 9.0], rtol=1e-6)


def test_true_termination_ignores_timeout_value() -> None:
    jnp = pytest.importorskip("jax.numpy")
    advantages, _ = _compute(
        rewards=jnp.asarray([[0.0]]),
        dones=jnp.asarray([[True]]),
        values=jnp.asarray([[2.0]]),
        last_value=jnp.asarray([3.0]),
        gamma=0.9,
        gae_lambda=1.0,
        terminated=jnp.asarray([[True]]),
        truncated=jnp.asarray([[False]]),
        timeout_values=jnp.asarray([[10.0]]),
        time_limit_bootstrap=True,
    )
    np.testing.assert_allclose(advantages[:, 0], [-2.0], rtol=1e-6)


def test_legacy_mode_preserves_done_as_zero_value_terminal() -> None:
    jnp = pytest.importorskip("jax.numpy")
    advantages, _ = _compute(
        rewards=jnp.asarray([[0.0]]),
        dones=jnp.asarray([[True]]),
        values=jnp.asarray([[2.0]]),
        last_value=jnp.asarray([3.0]),
        gamma=0.9,
        gae_lambda=1.0,
        time_limit_bootstrap=False,
    )
    np.testing.assert_allclose(advantages[:, 0], [-2.0], rtol=1e-6)


def test_failure_focus_excludes_truncations_and_unfinished_suffixes() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from train_juggle_mjx_ppo import completed_failure_focus_mask

    # env 0: a truncation at t=1, then an 11-hit failure at t=4.
    # env 1: a 10-hit failure at t=3, followed by an unfinished suffix.
    dones = jnp.asarray(
        [
            [False, False],
            [True, False],
            [False, False],
            [False, True],
            [True, False],
            [False, False],
        ]
    )
    terminated = jnp.asarray(
        [
            [False, False],
            [False, False],
            [False, False],
            [False, True],
            [True, False],
            [False, False],
        ]
    )
    hit_counts = jnp.asarray(
        [
            [9, 2],
            [11, 4],
            [3, 6],
            [7, 10],
            [11, 1],
            [1, 3],
        ]
    )

    mask = completed_failure_focus_mask(
        dones,
        terminated,
        hit_counts,
        hit_threshold=12,
    )
    np.testing.assert_array_equal(
        np.asarray(mask),
        np.asarray(
            [
                [False, True],
                [False, True],
                [True, True],
                [True, True],
                [True, False],
                [False, False],
            ]
        ),
    )


def test_failure_focus_tail_limits_credit_to_causal_window() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from train_juggle_mjx_ppo import completed_failure_focus_mask

    mask = completed_failure_focus_mask(
        dones=jnp.asarray([[False], [False], [False], [True]]),
        terminated=jnp.asarray([[False], [False], [False], [True]]),
        hit_counts=jnp.asarray([[8], [9], [10], [11]]),
        hit_threshold=12,
        tail_steps=2,
    )
    np.testing.assert_array_equal(np.asarray(mask[:, 0]), [False, False, True, True])


def test_min_log_std_is_shared_by_effective_scale_and_checkpoint_projection() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from train_juggle_mjx_ppo import effective_log_std, project_policy_log_std

    raw = jnp.asarray([-5.2, -4.5, -3.8], dtype=jnp.float32)
    expected = np.asarray([-4.5, -4.5, -3.8], dtype=np.float32)

    np.testing.assert_allclose(np.asarray(effective_log_std(raw, -4.5)), expected)
    params = {"log_std": raw, "other": jnp.asarray([1.0], dtype=jnp.float32)}
    projected = project_policy_log_std(params, -4.5)
    np.testing.assert_allclose(np.asarray(projected["log_std"]), expected)
    np.testing.assert_allclose(np.asarray(projected["other"]), [1.0])
    np.testing.assert_allclose(np.asarray(params["log_std"]), np.asarray(raw))


def test_diagonal_gaussian_kl_is_zero_for_same_policy_and_positive_after_move() -> None:
    jnp = pytest.importorskip("jax.numpy")
    from train_juggle_mjx_ppo import diagonal_gaussian_kl

    old_mean = jnp.asarray([[0.0, 1.0], [0.5, -0.5]], dtype=jnp.float32)
    old_log_std = jnp.asarray([-1.0, -0.5], dtype=jnp.float32)
    same = diagonal_gaussian_kl(old_mean, old_log_std, old_mean, old_log_std)
    moved = diagonal_gaussian_kl(
        old_mean,
        old_log_std,
        old_mean + 0.25,
        old_log_std + 0.1,
    )

    np.testing.assert_allclose(np.asarray(same), 0.0, atol=1e-7)
    assert float(moved) > 0.0


def test_ppo_kl_backtracking_reaches_small_adam_warm_start_steps() -> None:
    pytest.importorskip("jax")
    from train_juggle_mjx_ppo import PPO_KL_BACKTRACK_SCALES

    assert PPO_KL_BACKTRACK_SCALES == (
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
    )
    assert all(
        later == pytest.approx(earlier * 0.5)
        for earlier, later in zip(
            PPO_KL_BACKTRACK_SCALES,
            PPO_KL_BACKTRACK_SCALES[1:],
            strict=True,
        )
    )
