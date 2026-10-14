"""Opt-in reward components for the 4 g fixed-plane continuation."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def noncontact_motion_penalty(
    *,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    pending_hit: jax.Array,
    resolved_impact: jax.Array,
    action_delta_norm: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    racket_vz_m_s: jax.Array,
    lead_s: float,
    control_dt_s: float,
) -> tuple[jax.Array, jax.Array]:
    """Penalize early-flight motion while leaving the impact window free."""

    if not (math.isfinite(lead_s) and lead_s > 0.0):
        raise ValueError("lead_s must be finite and positive")
    if not (math.isfinite(control_dt_s) and control_dt_s > 0.0):
        raise ValueError("control_dt_s must be finite and positive")
    active = (
        (jnp.asarray(hit_count) > 0)
        & jnp.asarray(crossing_valid, dtype=bool)
        & (jnp.asarray(time_to_contact_s) > lead_s)
        & ~jnp.asarray(pending_hit, dtype=bool)
        & ~jnp.asarray(resolved_impact, dtype=bool)
    )
    motion = (
        0.25 * jnp.asarray(action_delta_norm)
        + 2.0 * jnp.abs(jnp.asarray(racket_vxy_m_s))
        + 0.3 * jnp.abs(jnp.asarray(racket_omega_rad_s))
        + 0.5 * jnp.maximum(jnp.asarray(racket_vz_m_s), 0.0)
    )
    return jnp.where(active, -control_dt_s * motion, 0.0), active


def weak_launch_lift_credit(
    *,
    event: jax.Array,
    apex_z_m: jax.Array,
    contact_center_z_m: jax.Array,
    target_apex_z_m: jax.Array,
    weight: float,
) -> jax.Array:
    """Credit a resolved weak rebound continuously without counting a hit."""

    if not (math.isfinite(weight) and weight >= 0.0):
        raise ValueError("weight must be finite and nonnegative")
    floor = jnp.asarray(contact_center_z_m)
    target = jnp.asarray(target_apex_z_m)
    fraction = jnp.clip(
        (jnp.asarray(apex_z_m) - floor) / jnp.maximum(target - floor, 1e-6),
        0.0,
        1.0,
    )
    return jnp.where(jnp.asarray(event, dtype=bool), weight * fraction, 0.0)
