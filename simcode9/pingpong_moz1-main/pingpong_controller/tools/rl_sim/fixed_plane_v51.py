"""Late weak-rebound apex deficit scaling without early-hit changes."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def late_weak_floor_penalty(
    *,
    base_penalty: jax.Array,
    confirmed_hits_before_event: jax.Array,
    late_multiplier: float,
) -> jax.Array:
    """Scale the existing weak-apex deficit only from ordinal four onward."""

    if not math.isfinite(float(late_multiplier)) or late_multiplier < 1.0:
        raise ValueError("late_multiplier must be finite and >= 1")
    base = jnp.asarray(base_penalty)
    previous_hits = jnp.asarray(confirmed_hits_before_event)
    if base.shape != previous_hits.shape:
        raise ValueError("base_penalty and hit count shapes must match")
    return jnp.where(previous_hits >= 3, base * late_multiplier, base)
