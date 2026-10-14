"""Smooth physical-impact apex credit for the opt-in V90 probe."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def smooth_steady_height_credit(
    *, event: jax.Array, confirmed_hits_before_event: jax.Array,
    apex_z_m: jax.Array, weight: float,
) -> jax.Array:
    """Give every late rebound positive credit with a usable low-height slope.

    The upper-tail subtraction prevents replacing low bounces with >1.45 m
    bounces. Earlier launch contacts have zero credit. All values are bounded.
    """
    apex = jnp.asarray(apex_z_m, dtype=jnp.float32)
    distance = jnp.abs(apex - 1.425)
    smooth_credit = 0.5 + 9.0 * jnp.exp(-distance / 0.060)
    high_cost = 5.0 * jnp.clip((apex - 1.450) / 0.050, 0.0, 1.0)
    event_credit = jnp.clip(smooth_credit - high_cost, 0.25, 9.5)
    active = jnp.asarray(event, dtype=bool) & (
        jnp.asarray(confirmed_hits_before_event) >= 3
    )
    return jnp.where(active, weight * event_credit, 0.0)
