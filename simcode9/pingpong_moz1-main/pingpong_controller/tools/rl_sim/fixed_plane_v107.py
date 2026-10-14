"""Opt-in first-launch recognition and ball-height progress for V107."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def first_launch_confirmation_limits(
    *,
    hit_count: jax.Array,
    legacy_clearance_m: float,
    legacy_deadline_steps: int,
    first_clearance_m: float,
    first_deadline_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Extend only the unconfirmed first launch; preserve recurrent hits."""
    if first_clearance_m < 0.0 or first_deadline_steps < 0:
        raise ValueError("first-launch confirmation limits must be nonnegative")
    first = jnp.asarray(hit_count) == 0
    clearance = jnp.where(
        first & (first_clearance_m > 0.0),
        first_clearance_m,
        legacy_clearance_m,
    )
    deadline = jnp.where(
        first & (first_deadline_steps > 0),
        first_deadline_steps,
        legacy_deadline_steps,
    )
    return clearance, deadline


def first_launch_stick_min_steps(
    *,
    hit_count: jax.Array,
    episode_step: jax.Array,
    legacy_min_steps: int,
    first_min_steps: int,
    initial_window_steps: int,
) -> jax.Array:
    """Allow brief initial carrying while keeping later stick checks intact."""
    if first_min_steps < 0 or initial_window_steps < 0:
        raise ValueError("first-launch stick grace must be nonnegative")
    first_window = (
        (jnp.asarray(hit_count) == 0)
        & (jnp.asarray(episode_step) <= initial_window_steps)
        & (first_min_steps > 0)
    )
    return jnp.where(first_window, first_min_steps, legacy_min_steps)


def first_launch_height_progress_credit(
    *,
    pending_hit: jax.Array,
    failed_hit: jax.Array,
    confirmed_hits_before_event: jax.Array,
    ball_z_m: jax.Array,
    previous_ball_z_m: jax.Array,
    reset_ball_z_m: jax.Array,
    target_lift_m: float,
    weight: float,
) -> jax.Array:
    """Credit physical ball rise during the first pending launch.

    The potential telescopes through carried and free flight. A failed first
    launch gives back its remaining potential, so lifting without a valid
    release cannot retain the credit.
    """
    if not (math.isfinite(target_lift_m) and target_lift_m > 0.0):
        raise ValueError("first-launch target lift must be finite and positive")
    if not (math.isfinite(weight) and weight >= 0.0):
        raise ValueError("first-launch progress weight must be nonnegative")
    current = jnp.clip(
        (jnp.asarray(ball_z_m) - reset_ball_z_m) / target_lift_m,
        0.0,
        1.0,
    )
    previous = jnp.clip(
        (jnp.asarray(previous_ball_z_m) - reset_ball_z_m) / target_lift_m,
        0.0,
        1.0,
    )
    first = jnp.asarray(confirmed_hits_before_event) == 0
    pending_credit = (current - previous) * (
        jnp.asarray(pending_hit, dtype=jnp.bool_) & first
    )
    failure_reclaim = current * (
        jnp.asarray(failed_hit, dtype=jnp.bool_) & first
    )
    return float(weight) * (pending_credit - failure_reclaim)
