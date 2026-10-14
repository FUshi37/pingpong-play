"""Outcome credit for the first recurrent-to-third-hit transition."""

import math

import jax
import jax.numpy as jnp


def third_confirmed_hit_bridge_bonus(
    *,
    confirmed_hit: jax.Array,
    previous_hit_count: jax.Array,
    weight: float,
) -> jax.Array:
    """Pay once for a real third hit, never for an unconfirmed contact edge."""

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    event = jnp.asarray(confirmed_hit, dtype=jnp.bool_)
    previous = jnp.asarray(previous_hit_count)
    return jnp.where(event & (previous == 2), weight, 0.0)
