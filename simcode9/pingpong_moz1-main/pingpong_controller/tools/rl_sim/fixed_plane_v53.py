"""One-sided vertical stroke cost for fast late arrivals."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def fast_arrival_vertical_deficit_penalty(
    *,
    racket_vz_m_s: jax.Array,
    incoming_ball_vz_m_s: jax.Array,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    pending_hit: jax.Array,
    reward_weight: float,
    dt_s: float,
    speed_threshold_m_s: float,
    active_window_s: float,
    base_target_vz_m_s: float,
    target_gain: float,
    max_target_vz_m_s: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Charge a bounded actual-racket-Vz deficit only near fast fourth-plus impacts.

    The incoming velocity is the existing policy-visible contact predictor.
    The score is one-sided so an already sufficient upward stroke receives no
    extra credit, and a downward stroke cannot create an unbounded loss.
    """

    values = (
        reward_weight, dt_s, speed_threshold_m_s, active_window_s,
        base_target_vz_m_s, target_gain, max_target_vz_m_s,
    )
    if (not all(math.isfinite(float(value)) for value in values)
            or reward_weight < 0.0 or dt_s <= 0.0
            or speed_threshold_m_s <= 0.0 or active_window_s <= 0.0
            or base_target_vz_m_s <= 0.0 or target_gain < 0.0
            or max_target_vz_m_s < base_target_vz_m_s):
        raise ValueError("invalid V53 fast-arrival vertical reward parameters")
    incoming = jnp.asarray(incoming_ball_vz_m_s, dtype=jnp.float32)
    racket_vz = jnp.asarray(racket_vz_m_s, dtype=jnp.float32)
    speed_excess = jnp.maximum(-incoming - speed_threshold_m_s, 0.0)
    target = jnp.minimum(
        base_target_vz_m_s + target_gain * speed_excess,
        max_target_vz_m_s,
    )
    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    active = (
        (reward_weight > 0.0)
        & (jnp.asarray(hit_count) >= 3)
        & jnp.asarray(crossing_valid, dtype=jnp.bool_)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (incoming <= -speed_threshold_m_s)
        & (ttc > 0.0)
        & (ttc <= active_window_s)
    )
    deficit = jnp.clip((target - racket_vz) / target, 0.0, 1.0)
    penalty = -reward_weight * dt_s * deficit * active.astype(jnp.float32)
    return penalty, target, active
