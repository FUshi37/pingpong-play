"""Optional true second/third rebound apex reward for V95."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def early_confirmed_apex_bridge(
    *, counted_hit: jax.Array, confirmed_hits_before_event: jax.Array,
    predicted_apex_z_m: jax.Array, target_apex_z_m: jax.Array,
    second_weight: float, third_weight: float, sigma_m: float,
    low_margin_m: float, low_penalty_weight: float,
) -> jax.Array:
    """Score scheduled apex on physical second and third confirmed rebounds."""
    if sigma_m <= 0.0 or low_margin_m < 0.0 or low_penalty_weight < 0.0:
        raise ValueError('invalid V95 early-apex reward scale')
    if second_weight < 0.0 or third_weight < 0.0:
        raise ValueError('V95 early-apex weights must be nonnegative')
    prior = jnp.asarray(confirmed_hits_before_event)
    active = jnp.asarray(counted_hit, dtype=bool) & ((prior == 1) | (prior == 2))
    weight = jnp.where(prior == 1, float(second_weight), float(third_weight))
    error_m = jnp.asarray(predicted_apex_z_m) - jnp.asarray(target_apex_z_m)
    positive = weight * jnp.exp(-0.5 * (error_m / float(sigma_m)) ** 2)
    low_shortfall_m = jnp.maximum(-error_m - float(low_margin_m), 0.0)
    low_penalty = float(low_penalty_weight) * jnp.minimum(
        low_shortfall_m / float(sigma_m), 1.0
    )
    return jnp.where(active, positive - low_penalty, 0.0)
