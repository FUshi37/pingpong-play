"""One-time third-contact reward using completed physical bounce velocity."""

import math

import jax
import jax.numpy as jnp


def first_third_release_vz_credit(
    *,
    release_event: jax.Array,
    previous_hit_count: jax.Array,
    previous_pending_hit: jax.Array,
    previous_contact_time_s: jax.Array,
    last_counted_hit_time_s: jax.Array,
    previous_claimed: jax.Array,
    outgoing_ball_vz_m_s: jax.Array,
    target_vz_m_s: float,
    weight: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Credit completed third bounce once, including weak upward outcomes."""

    if not math.isfinite(float(target_vz_m_s)) or target_vz_m_s <= 0.0:
        raise ValueError("target_vz_m_s must be finite and positive")
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    active = (
        jnp.asarray(release_event, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) == 2)
        & jnp.asarray(previous_pending_hit, dtype=jnp.bool_)
        & ~jnp.asarray(previous_claimed, dtype=jnp.bool_)
        & (jnp.asarray(last_counted_hit_time_s) >= 0.0)
        & (
            jnp.asarray(previous_contact_time_s)
            > jnp.asarray(last_counted_hit_time_s) + 1.0e-6
        )
    )
    score = jnp.clip(
        jnp.asarray(outgoing_ball_vz_m_s, dtype=jnp.float32)
        / float(target_vz_m_s),
        0.0,
        1.0,
    )
    reward = jnp.where(active, float(weight) * score, 0.0)
    return reward, active, score, jnp.asarray(previous_claimed) | active
