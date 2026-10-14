"""Dense bounded intercept guidance after two confirmed contacts."""

import math

import jax
import jax.numpy as jnp


def post_second_intercept_error_penalty(
    *,
    target_xy_m: jax.Array,
    racket_xy_m: jax.Array,
    target_valid: jax.Array,
    previous_hit_count: jax.Array,
    pending_hit: jax.Array,
    in_contact: jax.Array,
    time_to_contact_s: jax.Array,
    weight_per_s: float,
    sigma_m: float,
    window_s: float,
    dt_s: float,
) -> tuple[jax.Array, jax.Array]:
    """Charge bounded racket-to-next-contact error during actionable return flight."""

    if not math.isfinite(weight_per_s) or weight_per_s < 0.0:
        raise ValueError("weight_per_s must be finite and nonnegative")
    if not math.isfinite(sigma_m) or sigma_m <= 0.0:
        raise ValueError("sigma_m must be finite and positive")
    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("window_s must be finite and positive")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    ttc_s = jnp.asarray(time_to_contact_s)
    active = (
        jnp.asarray(target_valid, dtype=jnp.bool_)
        & (jnp.asarray(previous_hit_count) >= 2)
        & ~jnp.asarray(pending_hit, dtype=jnp.bool_)
        & ~jnp.asarray(in_contact, dtype=jnp.bool_)
        & (ttc_s > 0.0)
        & (ttc_s <= window_s)
    )
    error_m = jnp.linalg.norm(
        jnp.asarray(racket_xy_m, dtype=jnp.float32)
        - jnp.asarray(target_xy_m, dtype=jnp.float32),
        axis=-1,
    )
    bounded_loss = -weight_per_s * dt_s * (
        1.0 - jnp.exp(-0.5 * (error_m / sigma_m) ** 2)
    )
    return jnp.where(active, bounded_loss, 0.0), active
