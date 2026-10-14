"""Opt-in continuous cost for physical weak rebounds below the apex floor."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def weak_launch_apex_floor_penalty(
    *,
    event: jax.Array,
    apex_z_m: jax.Array,
    target_apex_z_m: jax.Array,
    margin_m: float,
    gap_scale_m: float,
    weight: float,
) -> jax.Array:
    """Charge a bounded squared deficit; leave valid-height events at zero."""

    if not (math.isfinite(margin_m) and margin_m >= 0.0):
        raise ValueError("margin_m must be finite and nonnegative")
    if not (math.isfinite(gap_scale_m) and gap_scale_m > 0.0):
        raise ValueError("gap_scale_m must be finite and positive")
    if not (math.isfinite(weight) and weight >= 0.0):
        raise ValueError("weight must be finite and nonnegative")
    floor = jnp.asarray(target_apex_z_m) - margin_m
    gap = jnp.maximum(floor - jnp.asarray(apex_z_m), 0.0)
    cost = -weight * jnp.minimum((gap / gap_scale_m) ** 2, 1.0)
    return jnp.where(jnp.asarray(event, dtype=bool), cost, 0.0)
