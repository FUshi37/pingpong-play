"""V92 signed physical-height credit and position-history actor input."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


# 200 Hz history reaches 480 ms, approximately one desired juggling period.
HISTORY_LAGS_TICKS = (0, 2, 4, 8, 16, 32, 64, 96)
HISTORY_LENGTH = HISTORY_LAGS_TICKS[-1] + 1
ACTOR_OBS_DIM = 137
ACTOR_OBSERVATION_SCHEMA = "fixed_plane_v92_position_history_no_ball_velocity_137d"


class SignedHeightReward(NamedTuple):
    """Per-impact total and auditable components, each shaped (N,)."""

    total: jax.Array
    absolute: jax.Array
    adjacent_penalty: jax.Array
    unresolved_penalty: jax.Array
    eligible: jax.Array
    below_floor: jax.Array
    in_band: jax.Array
    above_ceiling: jax.Array


def signed_steady_height_reward(
    *, resolved_event: jax.Array, unresolved_contact: jax.Array,
    confirmed_hits_before_event: jax.Array, apex_z_m: jax.Array,
    previous_apex_z_m: jax.Array, previous_valid: jax.Array,
) -> SignedHeightReward:
    """Score every fourth-plus physical contact, including weak rebounds.

    Absolute height has a non-saturating slope outside 1.400--1.450 m.
    In-band credit is positive throughout the closed interval and peaks at
    1.425 m. The adjacent term penalizes alternating heights even if the
    current bounce is in band. A contact that never resolves upward receives
    a separate failure penalty. Pre-transition contacts keep their old reward.
    """
    resolved = jnp.asarray(resolved_event, dtype=bool)
    unresolved = jnp.asarray(unresolved_contact, dtype=bool) & ~resolved
    hits = jnp.asarray(confirmed_hits_before_event)
    apex = jnp.asarray(apex_z_m, dtype=jnp.float32)
    previous = jnp.asarray(previous_apex_z_m, dtype=jnp.float32)
    previous_ok = jnp.asarray(previous_valid, dtype=bool)
    active_resolved = resolved & (hits >= 3)
    active_unresolved = unresolved & (hits >= 3)
    in_band = active_resolved & (apex >= 1.400) & (apex <= 1.450)
    below = active_resolved & (apex < 1.400)
    above = active_resolved & (apex > 1.450)
    centered = (apex - 1.425) / 0.025
    band_credit = 1.0 + 8.0 * jnp.maximum(0.0, 1.0 - centered * centered)
    low_cost = -1.0 - 10.0 * jnp.log1p(jnp.maximum(1.400 - apex, 0.0) / 0.025)
    high_cost = -1.0 - 10.0 * jnp.log1p(jnp.maximum(apex - 1.450, 0.0) / 0.025)
    absolute = jnp.where(
        in_band, band_credit,
        jnp.where(below, low_cost, jnp.where(above, high_cost, 0.0)),
    )
    adjacent_penalty = jnp.where(
        active_resolved & previous_ok,
        -2.0 * jnp.log1p(jnp.abs(apex - previous) / 0.025),
        0.0,
    )
    unresolved_penalty = jnp.where(active_unresolved, -10.0, 0.0)
    return SignedHeightReward(
        total=absolute + adjacent_penalty + unresolved_penalty,
        absolute=absolute,
        adjacent_penalty=adjacent_penalty,
        unresolved_penalty=unresolved_penalty,
        eligible=active_resolved | active_unresolved,
        below_floor=below,
        in_band=in_band,
        above_ceiling=above,
    )


def scale_signed_height_reward(
    reward: SignedHeightReward, *, scale: float, adjacent_multiplier: float = 1.0,
    adjacent_linear: bool = False,
) -> SignedHeightReward:
    """Scale reward components without changing physical event telemetry."""
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("signed-height reward scale must be finite and positive")
    if not math.isfinite(float(adjacent_multiplier)) or float(adjacent_multiplier) <= 0.0:
        raise ValueError("adjacent multiplier must be finite and positive")
    if adjacent_linear:
        adjacent = -2.0 * jnp.expm1(-reward.adjacent_penalty / 2.0)
        reward = reward._replace(adjacent_penalty=adjacent,
            total=reward.absolute + adjacent + reward.unresolved_penalty)
    if float(adjacent_multiplier) != 1.0:
        adjacent = reward.adjacent_penalty * float(adjacent_multiplier)
        reward = reward._replace(adjacent_penalty=adjacent,
            total=reward.absolute + adjacent + reward.unresolved_penalty)
    if float(scale) == 1.0:
        return reward
    return reward._replace(
        total=reward.total * float(scale),
        absolute=reward.absolute * float(scale),
        adjacent_penalty=reward.adjacent_penalty * float(scale),
        unresolved_penalty=reward.unresolved_penalty * float(scale),
    )


def rotation_matrix_to_6d(rotation: jax.Array) -> jax.Array:
    """Return the first two racket-frame axes from an (N,3,3) matrix."""
    matrix = jnp.asarray(rotation, dtype=jnp.float32)
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape (N,3,3)")
    return jnp.swapaxes(matrix[:, :, :2], -1, -2).reshape(matrix.shape[0], 6)


def advance_position_history(
    *, ball_positions: jax.Array, racket_poses: jax.Array,
    ball_ages: jax.Array, ball_position_now: jax.Array,
    racket_pose_now: jax.Array, ball_age_now: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Advance one 200 Hz tick with held camera position and current FK pose."""
    return (
        jnp.concatenate(
            (ball_position_now[:, None, :], ball_positions[:, :-1, :]), axis=1
        ),
        jnp.concatenate((racket_pose_now[:, None, :], racket_poses[:, :-1, :]), axis=1),
        jnp.concatenate((ball_age_now[:, None], ball_ages[:, :-1]), axis=1),
    )


def make_history_actor_observation(
    *, q: jax.Array, dq: jax.Array, q_ref_error: jax.Array,
    previous_action: jax.Array, ball_positions: jax.Array,
    racket_poses: jax.Array, ball_ages: jax.Array,
    desired_apex_z_m: jax.Array, target_xy_m: jax.Array,
    hit_count: jax.Array, ball_valid: jax.Array,
    lags_ticks: tuple[int, ...] = HISTORY_LAGS_TICKS,
) -> jax.Array:
    """Build the 137-D actor input from positions, poses and proprioception.

    The arguments deliberately contain no measured, predicted or desired ball
    velocity. The only time-series inputs are causal held camera positions,
    measured racket FK poses, and corresponding camera-sample ages.
    """
    ball = jnp.asarray(ball_positions, dtype=jnp.float32)
    racket = jnp.asarray(racket_poses, dtype=jnp.float32)
    ages = jnp.asarray(ball_ages, dtype=jnp.float32)
    n_envs = jnp.asarray(q).shape[0]
    if (
        len(lags_ticks) != 8
        or lags_ticks[0] != 0
        or tuple(sorted(set(lags_ticks))) != lags_ticks
    ):
        raise ValueError(
            "history lags must be eight increasing ticks starting at zero"
        )
    history_length = lags_ticks[-1] + 1
    if ball.shape != (n_envs, history_length, 3):
        raise ValueError("ball_positions history shape mismatch")
    if racket.shape != (n_envs, history_length, 9):
        raise ValueError("racket_poses history shape mismatch")
    if ages.shape != (n_envs, history_length):
        raise ValueError("ball_ages history shape mismatch")
    lags = jnp.asarray(lags_ticks, dtype=jnp.int32)
    output = jnp.concatenate((
        q, dq, q_ref_error, previous_action,
        ball[:, lags, :].reshape(n_envs, 24),
        racket[:, lags, :].reshape(n_envs, 72),
        jnp.clip(ages[:, lags] / 0.25, 0.0, 1.0),
        desired_apex_z_m[:, None], target_xy_m,
        jnp.asarray(ball_valid, dtype=jnp.float32)[:, None],
        jnp.clip(jnp.asarray(hit_count, dtype=jnp.float32)[:, None] / 4.0, 0.0, 1.0),
    ), axis=-1)
    if output.shape != (n_envs, ACTOR_OBS_DIM):
        raise ValueError("history actor observation dimension mismatch")
    return output
