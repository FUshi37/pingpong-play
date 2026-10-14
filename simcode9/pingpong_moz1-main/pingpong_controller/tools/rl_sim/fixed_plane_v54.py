"""Per-physical-rebound apex score for steady fixed-plane juggling."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def late_physical_apex_score(
    *,
    event: jax.Array,
    confirmed_hits_before_event: jax.Array,
    apex_z_m: jax.Array,
    target_apex_z_m: jax.Array,
    floor_z_m: float,
    weight: float,
) -> jax.Array:
    """Reward a repeatable target apex; charge every low physical rebound.

    The score crosses zero at the accepted floor. The bounded Gaussian peak
    prevents unlimited height reward, while the linear floor deficit keeps a
    useful contrast over the observed 1.30--1.40 m weak-rebound range.
    """

    if not (math.isfinite(floor_z_m) and math.isfinite(weight)):
        raise ValueError("V54 floor and weight must be finite")
    if weight < 0.0:
        raise ValueError("V54 apex score weight must be nonnegative")
    apex = jnp.asarray(apex_z_m)
    target = jnp.asarray(target_apex_z_m)
    if apex.shape != target.shape:
        raise ValueError("V54 apex and target shapes must match")
    sigma_m = 0.050
    floor_gap_scale_m = 0.050
    quality = jnp.exp(-0.5 * ((apex - target) / sigma_m) ** 2)
    floor_quality = jnp.exp(-0.5 * ((floor_z_m - target) / sigma_m) ** 2)
    floor_gap = jnp.clip((floor_z_m - apex) / floor_gap_scale_m, 0.0, 2.0)
    score = weight * (quality - floor_quality - floor_gap)
    active = jnp.asarray(event, dtype=bool) & (
        jnp.asarray(confirmed_hits_before_event) >= 3
    )
    return jnp.where(active, score, 0.0)
