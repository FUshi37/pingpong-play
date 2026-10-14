"""Reward-only second-impact viability adjustment for V96."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


# Three physical outcome scales reuse V41's landing, VXY and contact-height
# scales; two achieved racket-motion scales reuse V46's penalty scales.
VIABILITY_SCALES = (0.05, 0.10, 0.015, 0.10, 0.40)


def second_hit_viability_adjustment(
    *,
    counted_hit: jax.Array,
    confirmed_hits_before_event: jax.Array,
    predicted_apex_z_m: jax.Array,
    target_apex_z_m: jax.Array,
    landing_error_m: jax.Array,
    outgoing_vxy_m_s: jax.Array,
    contact_height_error_m: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    weight: float,
    center: float,
    reference_quality: float,
    apex_sigma_m: float,
) -> jax.Array:
    """Favor a repeatable second rebound while retaining V95's height credit.

    The zero-centered adjustment can be negative for poor second-impact
    outcomes or positive for V53-like ones. It is multiplied by the existing
    scheduled-apex Gaussian, so good horizontal outcomes cannot compensate
    for a badly low or high apex. V95's signed low-height cost is unchanged.
    """
    params = (weight, center, reference_quality, apex_sigma_m)
    if not all(math.isfinite(float(value)) for value in params):
        raise ValueError("V96 viability reward parameters must be finite")
    if weight < 0.0 or not 0.0 <= center <= 1.0:
        raise ValueError("V96 viability weight/center out of range")
    if reference_quality <= 0.0 or apex_sigma_m <= 0.0:
        raise ValueError("V96 viability quality/apex scale must be positive")

    errors = jnp.stack(
        (
            landing_error_m,
            outgoing_vxy_m_s,
            contact_height_error_m,
            racket_vxy_m_s,
            racket_omega_rad_s,
        ),
        axis=-1,
    )
    safe_errors = jnp.nan_to_num(
        jnp.maximum(errors, 0.0), nan=1.0e6, posinf=1.0e6
    )
    components = jnp.exp(
        -safe_errors / jnp.asarray(VIABILITY_SCALES, dtype=safe_errors.dtype)
    )
    arithmetic = jnp.mean(components, axis=-1)
    geometric = jnp.exp(
        jnp.mean(jnp.log(jnp.maximum(components, 1.0e-6)), axis=-1)
    )
    quality = 0.5 * (arithmetic + geometric)
    normalized_quality = jnp.minimum(quality / float(reference_quality), 1.0)
    apex_error = jnp.asarray(predicted_apex_z_m) - jnp.asarray(target_apex_z_m)
    apex_score = jnp.exp(-0.5 * (apex_error / float(apex_sigma_m)) ** 2)
    active = jnp.asarray(counted_hit, dtype=bool) & (
        jnp.asarray(confirmed_hits_before_event) == 1
    )
    adjustment = float(weight) * apex_score * (
        normalized_quality - float(center)
    )
    return jnp.where(active, adjustment, 0.0)
