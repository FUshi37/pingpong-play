"""Bounded motion cost at one resolved physical rebound."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def resolved_impact_motion_penalty(
    *,
    event: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    vxy_limit_m_s: float,
    omega_limit_rad_s: float,
    vxy_scale_m_s: float,
    omega_scale_rad_s: float,
    vxy_weight: float,
    omega_weight: float,
    nonsaturating_tail: bool = False,
) -> jax.Array:
    """Penalize excess achieved racket motion once for each resolved rebound."""

    values = (
        vxy_limit_m_s, omega_limit_rad_s,
        vxy_scale_m_s, omega_scale_rad_s,
        vxy_weight, omega_weight,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("motion limits, scales and weights must be finite")
    if vxy_limit_m_s < 0.0 or omega_limit_rad_s < 0.0:
        raise ValueError("motion limits must be nonnegative")
    if vxy_scale_m_s <= 0.0 or omega_scale_rad_s <= 0.0:
        raise ValueError("motion scales must be positive")
    if vxy_weight < 0.0 or omega_weight < 0.0:
        raise ValueError("motion weight must be nonnegative")
    vxy_excess = jnp.maximum(jnp.asarray(racket_vxy_m_s) - vxy_limit_m_s, 0.0)
    omega_excess = jnp.maximum(
        jnp.asarray(racket_omega_rad_s) - omega_limit_rad_s, 0.0
    )
    vxy_ratio = vxy_excess / vxy_scale_m_s
    omega_ratio = omega_excess / omega_scale_rad_s
    if nonsaturating_tail:
        # Match V46 exactly through one scale, then retain a smooth,
        # positive gradient for the late high-motion impacts V46 missed.
        vxy_cost = jnp.where(
            vxy_ratio <= 1.0, vxy_ratio ** 2,
            1.0 + 2.0 * jnp.log(jnp.maximum(vxy_ratio, 1.0)),
        )
        omega_cost = jnp.where(
            omega_ratio <= 1.0, omega_ratio ** 2,
            1.0 + 2.0 * jnp.log(jnp.maximum(omega_ratio, 1.0)),
        )
    else:
        vxy_cost = jnp.minimum(vxy_ratio ** 2, 1.0)
        omega_cost = jnp.minimum(omega_ratio ** 2, 1.0)
    cost = vxy_weight * vxy_cost + omega_weight * omega_cost
    return jnp.where(jnp.asarray(event, dtype=bool), -cost, 0.0)
