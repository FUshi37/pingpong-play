"""Default-off, confirmed-impact VXY contrast for the V32 experiment."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def confirmed_contact_vxy_event_bonus(
    *,
    event: jax.Array,
    racket_vxy_m_s: jax.Array,
    ball_outgoing_vxy_m_s: jax.Array,
    weight: float,
    sigma_m_s: float,
    reference_m_s: float,
) -> jax.Array:
    """Return an impact-only smooth contrast centered on reference speed.

    Inputs are scalar or equally shaped JAX arrays in SI units.  The baseline
    is zero when both speeds equal ``reference_m_s``; quiet impacts gain a
    bounded positive bonus and faster impacts lose credit.  No-hit steps
    receive exactly zero, so a stationary miss cannot collect this reward.
    """

    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("event VXY weight must be finite and nonnegative")
    if not math.isfinite(sigma_m_s) or sigma_m_s <= 0.0:
        raise ValueError("event VXY sigma must be positive and finite")
    if not math.isfinite(reference_m_s) or reference_m_s < 0.0:
        raise ValueError("event VXY reference must be finite and nonnegative")

    racket = jnp.asarray(racket_vxy_m_s, dtype=jnp.float32)
    ball = jnp.asarray(ball_outgoing_vxy_m_s, dtype=jnp.float32)
    active = jnp.asarray(event, dtype=jnp.bool_)
    reference_score = math.exp(-0.5 * (reference_m_s / sigma_m_s) ** 2)
    racket_score = jnp.exp(-0.5 * (racket / sigma_m_s) ** 2)
    ball_score = jnp.exp(-0.5 * (ball / sigma_m_s) ** 2)
    bonus = float(weight) * (
        racket_score + ball_score - 2.0 * reference_score
    )
    return jnp.where(active, bonus, 0.0)
