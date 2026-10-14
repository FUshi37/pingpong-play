"""One-shot weak-inclusive physical first-launch height credit."""
import math

import jax
import jax.numpy as jnp


def first_resolved_launch_peak_vz_credit(
    *,
    peak_vz_m_s: jax.Array,
    resolved_upward_launch: jax.Array,
    previous_hit_count: jax.Array,
    previous_credit_paid: jax.Array,
    weight: float,
    target_vz_m_s: float,
) -> tuple[jax.Array, jax.Array]:
    """Credit the first actual upward outcome even below hit-confirm height.

    The returned latch shares the legacy first-failed-outcome state for this
    opt-in profile, preventing repeated weak contacts from collecting credit.
    """

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("V78 first resolved launch weight must be finite and nonnegative")
    if not math.isfinite(target_vz_m_s) or target_vz_m_s <= 0.0:
        raise ValueError("V78 first resolved launch target must be finite and positive")
    paid = jnp.asarray(previous_credit_paid, dtype=jnp.bool_)
    eligible = (
        jnp.asarray(resolved_upward_launch, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) == 0)
        & (~paid)
        & (weight > 0.0)
    )
    score = jnp.clip(
        jnp.maximum(jnp.asarray(peak_vz_m_s), 0.0) / target_vz_m_s,
        0.0,
        1.0,
    )
    return jnp.where(eligible, weight * score, 0.0), paid | eligible
