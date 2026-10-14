"""Bounded credit for actual outgoing speed of the first confirmed hit."""
import math

import jax
import jax.numpy as jnp


def first_confirmed_peak_vz_credit(
    *,
    peak_vz_m_s: jax.Array,
    confirmed_hit: jax.Array,
    previous_hit_count: jax.Array,
    weight: float,
    target_vz_m_s: float,
) -> jax.Array:
    """Pay once on first true confirmed hit; failed/repeated contacts pay zero."""

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("V77 first-hit weight must be finite and nonnegative")
    if not math.isfinite(target_vz_m_s) or target_vz_m_s <= 0.0:
        raise ValueError("V77 first-hit target Vz must be finite and positive")
    eligible = (
        jnp.asarray(confirmed_hit, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) == 0)
        & (weight > 0.0)
    )
    score = jnp.clip(
        jnp.maximum(jnp.asarray(peak_vz_m_s), 0.0) / target_vz_m_s,
        0.0,
        1.0,
    )
    return jnp.where(eligible, weight * score, 0.0)
