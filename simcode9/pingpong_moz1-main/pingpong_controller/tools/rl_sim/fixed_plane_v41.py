"""Bounded four-component credit for a resolved physical rebound."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def physical_outcome_credit(
    *,
    event: jax.Array,
    landing_error_m: jax.Array,
    outgoing_vxy_m_s: jax.Array,
    apex_error_m: jax.Array,
    contact_height_error_m: jax.Array,
    weight: float,
) -> jax.Array:
    """Pay each physical outcome once; poor components reduce joint credit."""

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    errors = jnp.stack(
        (
            jnp.asarray(landing_error_m) / 0.05,
            jnp.asarray(outgoing_vxy_m_s) / 0.10,
            jnp.asarray(apex_error_m) / 0.025,
            jnp.asarray(contact_height_error_m) / 0.015,
        ),
        axis=-1,
    )
    component = jnp.exp(-jnp.maximum(errors, 0.0))
    # The mean keeps a gradient when one component is poor; the geometric
    # mean discourages trading a large landing error for a higher apex.
    joint = jnp.exp(jnp.mean(jnp.log(jnp.maximum(component, 1e-6)), axis=-1))
    score = 0.5 * jnp.mean(component, axis=-1) + 0.5 * joint
    return jnp.where(jnp.asarray(event, dtype=bool), weight * score, 0.0)
