"""V94 outcome-aligned first-launch reward, opt-in after V93."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pending_recontact_miss_equivalent(
    *, pending_recontact: jax.Array, ball_miss: jax.Array,
    confirmed_hits_before_event: jax.Array, miss_penalty: float,
) -> jax.Array:
    """Charge the same miss cost for failed first-three rebound contacts."""
    active = (
        jnp.asarray(pending_recontact, dtype=bool)
        & ~jnp.asarray(ball_miss, dtype=bool)
        & (jnp.asarray(confirmed_hits_before_event) < 3)
    )
    return jnp.where(active, -float(miss_penalty), 0.0)


def first_confirmed_apex_credit(
    *, counted_hit: jax.Array, confirmed_hits_before_event: jax.Array,
    predicted_apex_z_m: jax.Array, target_apex_z_m: jax.Array,
    weight: float, sigma_m: float,
) -> jax.Array:
    """Credit a true first hit most when its ballistic apex meets its goal."""
    if sigma_m <= 0.0:
        raise ValueError("first-hit apex sigma must be positive")
    error = (
        jnp.asarray(predicted_apex_z_m, dtype=jnp.float32)
        - jnp.asarray(target_apex_z_m, dtype=jnp.float32)
    )
    credit = float(weight) * jnp.exp(-0.5 * (error / float(sigma_m)) ** 2)
    active = (
        jnp.asarray(counted_hit, dtype=bool)
        & (jnp.asarray(confirmed_hits_before_event) == 0)
    )
    return jnp.where(active, credit, 0.0)


def first_hit_early_failure_penalty(
    *, failed_termination: jax.Array, hit_count: jax.Array,
    elapsed_time_s: jax.Array, weight: float, window_s: float,
) -> jax.Array:
    """Charge unfinished first-hit acquisition over its 0.5 s window."""
    if window_s <= 0.0:
        raise ValueError("first-hit failure window must be positive")
    remaining = jnp.clip(
        1.0 - jnp.asarray(elapsed_time_s, dtype=jnp.float32) / float(window_s),
        0.0, 1.0,
    )
    active = (
        jnp.asarray(failed_termination, dtype=bool)
        & (jnp.asarray(hit_count) == 0)
    )
    return jnp.where(active, -float(weight) * remaining, 0.0)
