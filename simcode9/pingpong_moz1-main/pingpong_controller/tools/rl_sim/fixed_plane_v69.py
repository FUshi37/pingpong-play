"""One-shot first-contact ball-outcome credit for the opt-in V69 course."""

import math

import jax
import jax.numpy as jnp

FIRST_FAILED_LAUNCH_TARGET_VZ_M_S = 0.9


def pending_first_launch_peak_vz(
    *,
    previous_peak_vz_m_s: jax.Array,
    ball_vz_m_s: jax.Array,
    hit_edge: jax.Array,
    pending_hit: jax.Array,
    resolved_hit: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return this contact's peak positive ball Vz and next pending cache."""

    upward_vz = jnp.maximum(jnp.asarray(ball_vz_m_s), 0.0)
    peak_vz = jnp.where(
        jnp.asarray(hit_edge, dtype=jnp.bool_),
        upward_vz,
        jnp.maximum(jnp.asarray(previous_peak_vz_m_s), upward_vz),
    )
    peak_vz = jnp.where(pending_hit, peak_vz, 0.0)
    stored_vz = jnp.where(pending_hit & (~resolved_hit), peak_vz, 0.0)
    return peak_vz, stored_vz


def failed_first_launch_peak_vz_credit(
    *,
    peak_vz_m_s: jax.Array,
    failed_hit: jax.Array,
    previous_hit_count: jax.Array,
    previous_credit_paid: jax.Array,
    weight: float,
    target_vz_m_s: float = FIRST_FAILED_LAUNCH_TARGET_VZ_M_S,
) -> tuple[jax.Array, jax.Array]:
    """Pay once for actual upward ball motion on failed first contact.

    The score is capped at one, and a true confirmed hit receives its
    existing reward. A failed contact cannot collect this credit twice in
    one episode, including after subsequent contact chatter.
    """

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("V69 failed-first-hit credit weight must be nonnegative")
    if not math.isfinite(target_vz_m_s) or target_vz_m_s <= 0.0:
        raise ValueError("V69 failed-first-hit target Vz must be positive")
    eligible = (
        jnp.asarray(failed_hit, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) == 0)
        & (~jnp.asarray(previous_credit_paid, dtype=jnp.bool_))
        & (weight > 0.0)
    )
    score = jnp.clip(
        jnp.maximum(jnp.asarray(peak_vz_m_s), 0.0) / target_vz_m_s,
        0.0,
        1.0,
    )
    credit = jnp.where(eligible, weight * score, 0.0)
    return credit, jnp.asarray(previous_credit_paid, dtype=jnp.bool_) | eligible
