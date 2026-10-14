"""Terminal accounting for postcontact potential reward."""

import math

import jax
import jax.numpy as jnp


def terminal_pending_clearance_refund(
    *,
    pending_hit: jax.Array,
    failed_hit: jax.Array,
    counted_hit: jax.Array,
    terminal: jax.Array,
    truncated: jax.Array,
    previous_hit_count: jax.Array,
    relative_z_m: jax.Array,
    contact_baseline_relative_z_m: jax.Array,
    confirm_relative_z_m: float,
    weight: float,
) -> tuple[jax.Array, jax.Array]:
    """Reclaim remaining pending potential on failure, including ball-out."""

    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("clearance weight must be finite and nonnegative")
    if not math.isfinite(float(confirm_relative_z_m)) or confirm_relative_z_m <= 0.0:
        raise ValueError("confirmation clearance must be finite and positive")
    baseline = jnp.asarray(contact_baseline_relative_z_m, dtype=jnp.float32)
    span = jnp.maximum(float(confirm_relative_z_m) - baseline, 0.005)
    current = jnp.clip((jnp.asarray(relative_z_m) - baseline) / span, 0.0, 1.0)
    active = (
        jnp.asarray(pending_hit, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) >= 2)
        & (jnp.asarray(terminal, dtype=jnp.bool_) | jnp.asarray(truncated, dtype=jnp.bool_))
        & ~jnp.asarray(failed_hit, dtype=jnp.bool_)
        & ~jnp.asarray(counted_hit, dtype=jnp.bool_)
    )
    return jnp.where(active, -float(weight) * current, 0.0), active
