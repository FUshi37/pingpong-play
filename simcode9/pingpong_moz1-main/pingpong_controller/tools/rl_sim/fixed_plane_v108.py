"""Contact-baselined first-launch lift reward with exact failure recovery."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def first_launch_contact_height_credit(
    *,
    pending_before_resolution: jax.Array,
    pending_started: jax.Array,
    failed_hit: jax.Array,
    confirmed_hits_before_event: jax.Array,
    ball_z_m: jax.Array,
    previous_ball_z_m: jax.Array,
    contact_ball_z_m: jax.Array,
    target_lift_m: float,
    weight: float,
) -> jax.Array:
    """Credit rise after physical contact and reclaim exactly that potential on failure.

    The contact step starts with zero potential even when the ball was already
    elevated. The pending flag is sampled before resolving the current step,
    so the last rise is credited on both success and failure. A failed launch
    returns the cumulative credit to zero; a confirmed launch retains it.
    """
    if not (math.isfinite(target_lift_m) and target_lift_m > 0.0):
        raise ValueError("first-launch target lift must be finite and positive")
    if not (math.isfinite(weight) and weight >= 0.0):
        raise ValueError("first-launch progress weight must be nonnegative")
    current = jnp.clip(
        (jnp.asarray(ball_z_m) - contact_ball_z_m) / target_lift_m,
        0.0,
        1.0,
    )
    previous = jnp.clip(
        (jnp.asarray(previous_ball_z_m) - contact_ball_z_m) / target_lift_m,
        0.0,
        1.0,
    )
    first_pending = (
        jnp.asarray(pending_before_resolution, dtype=jnp.bool_)
        & (jnp.asarray(confirmed_hits_before_event) == 0)
    )
    progress = jnp.where(
        first_pending & ~jnp.asarray(pending_started, dtype=jnp.bool_),
        current - previous,
        0.0,
    )
    reclaim = jnp.where(
        first_pending & jnp.asarray(failed_hit, dtype=jnp.bool_),
        current,
        0.0,
    )
    return float(weight) * (progress - reclaim)
