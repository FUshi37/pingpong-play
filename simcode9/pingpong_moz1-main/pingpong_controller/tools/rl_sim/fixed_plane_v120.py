"""One-time outgoing-ball outcome credit at the first third-contact edge."""

import math

import jax
import jax.numpy as jnp


def third_first_physical_edge_outgoing_vz_credit(
    *,
    raw_physical_edge: jax.Array,
    previous_hit_count: jax.Array,
    previous_pending: jax.Array,
    previous_physical_contact_time_s: jax.Array,
    last_counted_hit_time_s: jax.Array,
    outgoing_ball_vz_m_s: jax.Array,
    target_vz_m_s: float,
    weight: float,
    floor_vz_m_s: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Score actual upward ball speed once per second-to-third return."""

    if not math.isfinite(float(target_vz_m_s)) or target_vz_m_s <= 0.0:
        raise ValueError("target_vz_m_s must be finite and positive")
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    if not math.isfinite(float(floor_vz_m_s)) or floor_vz_m_s >= target_vz_m_s:
        raise ValueError("floor_vz_m_s must be finite and below target")
    last_hit = jnp.asarray(last_counted_hit_time_s, dtype=jnp.float32)
    previous_edge = jnp.asarray(previous_physical_contact_time_s, dtype=jnp.float32)
    active = (
        jnp.asarray(raw_physical_edge, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) == 2)
        & ~jnp.asarray(previous_pending, dtype=jnp.bool_)
        & (last_hit >= 0.0)
        & (previous_edge <= last_hit + 1.0e-6)
    )
    score = jnp.clip(
        (jnp.asarray(outgoing_ball_vz_m_s, dtype=jnp.float32) - float(floor_vz_m_s))
        / (float(target_vz_m_s) - float(floor_vz_m_s)),
        0.0,
        1.0,
    )
    return jnp.where(active, float(weight) * score, 0.0), active, score
