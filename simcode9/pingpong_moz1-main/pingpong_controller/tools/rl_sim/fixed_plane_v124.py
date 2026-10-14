"""One-time third-contact preparation score based on physical racket motion."""

import math

import jax
import jax.numpy as jnp


def first_third_edge_preparation_reward(
    *,
    raw_physical_edge: jax.Array,
    previous_hit_count: jax.Array,
    previous_pending: jax.Array,
    previous_physical_contact_time_s: jax.Array,
    last_counted_hit_time_s: jax.Array,
    racket_z_m: jax.Array,
    racket_contact_vz_m_s: jax.Array,
    lower_target_z_m: float,
    upper_bound_z_m: float,
    z_penalty_weight: float,
    vz_floor_m_s: float,
    vz_target_m_s: float,
    vz_reward_weight: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Score usable upward contact speed with room to complete the stroke."""

    params = (
        lower_target_z_m,
        upper_bound_z_m,
        z_penalty_weight,
        vz_floor_m_s,
        vz_target_m_s,
        vz_reward_weight,
    )
    if not all(math.isfinite(float(value)) for value in params):
        raise ValueError("third-edge preparation parameters must be finite")
    if upper_bound_z_m <= lower_target_z_m:
        raise ValueError("upper bound must exceed lower target")
    if vz_target_m_s <= vz_floor_m_s:
        raise ValueError("VZ target must exceed its score floor")
    if z_penalty_weight < 0.0 or vz_reward_weight < 0.0:
        raise ValueError("reward weights must be nonnegative")

    active = (
        jnp.asarray(raw_physical_edge, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) == 2)
        & ~jnp.asarray(previous_pending, dtype=jnp.bool_)
        & (jnp.asarray(last_counted_hit_time_s) >= 0.0)
        & (
            jnp.asarray(previous_physical_contact_time_s)
            <= jnp.asarray(last_counted_hit_time_s) + 1.0e-6
        )
    )
    z_penalty = -float(z_penalty_weight) * jnp.clip(
        (jnp.asarray(racket_z_m) - float(lower_target_z_m))
        / (float(upper_bound_z_m) - float(lower_target_z_m)),
        0.0,
        1.0,
    )
    vz_credit = float(vz_reward_weight) * jnp.clip(
        (jnp.asarray(racket_contact_vz_m_s) - float(vz_floor_m_s))
        / (float(vz_target_m_s) - float(vz_floor_m_s)),
        0.0,
        1.0,
    )
    return (
        jnp.where(active, z_penalty + vz_credit, 0.0),
        active,
        jnp.where(active, z_penalty, 0.0),
        jnp.where(active, vz_credit, 0.0),
    )
