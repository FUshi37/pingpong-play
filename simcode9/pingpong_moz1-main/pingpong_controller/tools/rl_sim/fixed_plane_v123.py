"""Align low-rebound cost with the active per-hit apex schedule."""

import math

import jax
import jax.numpy as jnp


def scheduled_low_apex_penalty(
    *,
    predicted_apex_z_m: jax.Array,
    scheduled_apex_z_m: jax.Array,
    margin_m: float,
    weight: float,
) -> jax.Array:
    """Return nonnegative loss below the current hit's scheduled apex."""

    if not math.isfinite(float(margin_m)) or margin_m < 0.0:
        raise ValueError("margin_m must be finite and nonnegative")
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    deficit_m = jnp.maximum(
        0.0,
        jnp.asarray(scheduled_apex_z_m) - float(margin_m)
        - jnp.asarray(predicted_apex_z_m),
    )
    return float(weight) * deficit_m * deficit_m
