"""Opt-in first-launch progress credit for a directly released ball."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def first_launch_clearance_credit(
    *,
    pending_hit: jax.Array,
    failed_hit: jax.Array,
    confirmed_hits_before_event: jax.Array,
    ball_vz_m_s: jax.Array,
    relative_z_m: jax.Array,
    previous_relative_z_m: jax.Array,
    launch_baseline_relative_z_m: jax.Array,
    confirm_relative_z_m: float,
    weight: float,
) -> jax.Array:
    """Credit actual upward ball separation after first physical contact.

    The score is a bounded change in ball-to-racket clearance. Carrying the
    ball with the racket leaves the score at zero. Lost clearance is charged
    back while the first contact remains pending, and any remaining credit is
    reclaimed if that contact fails. A weak microbounce cannot farm credit.
    """
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("first-launch progress weight must be nonnegative")
    if not math.isfinite(float(confirm_relative_z_m)) or confirm_relative_z_m <= 0.0:
        raise ValueError("launch confirmation height must be positive")
    baseline = jnp.asarray(launch_baseline_relative_z_m, dtype=jnp.float32)
    span = jnp.maximum(float(confirm_relative_z_m) - baseline, 0.005)
    current = jnp.clip((jnp.asarray(relative_z_m) - baseline) / span, 0.0, 1.0)
    previous = jnp.clip(
        (jnp.asarray(previous_relative_z_m) - baseline) / span, 0.0, 1.0
    )
    delta = current - previous
    first_contact = (
        jnp.asarray(pending_hit, dtype=jnp.bool_)
        & (jnp.asarray(confirmed_hits_before_event) == 0)
    )
    actual_upward_ball = jnp.asarray(ball_vz_m_s) > 0.0
    signed_progress = jnp.where(
        delta > 0.0,
        jnp.where(actual_upward_ball, delta, 0.0),
        delta,
    )
    failed_first_contact = (
        jnp.asarray(failed_hit, dtype=jnp.bool_)
        & (jnp.asarray(confirmed_hits_before_event) == 0)
    )
    return float(weight) * (
        signed_progress * first_contact.astype(jnp.float32)
        - current * failed_first_contact.astype(jnp.float32)
    )


def post_hit_quiet_credit(
    *,
    hit_count: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    vxy_sigma_m_s: float,
    omega_sigma_rad_s: float,
    weight: float,
    dt_s: float,
) -> jax.Array:
    """Reward low achieved lateral and angular racket speed after launch."""
    if any(
        not math.isfinite(float(value)) or value <= 0.0
        for value in (vxy_sigma_m_s, omega_sigma_rad_s, dt_s)
    ):
        raise ValueError("quiet-credit scales and dt must be positive")
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("quiet-credit weight must be nonnegative")
    vxy_score = jnp.exp(
        -0.5 * (jnp.asarray(racket_vxy_m_s) / float(vxy_sigma_m_s)) ** 2
    )
    omega_score = jnp.exp(
        -0.5 * (jnp.asarray(racket_omega_rad_s) / float(omega_sigma_rad_s)) ** 2
    )
    active = (jnp.asarray(hit_count) > 0).astype(jnp.float32)
    return 0.5 * float(weight) * float(dt_s) * (vxy_score + omega_score) * active


def first_hit_promptness_credit(
    *,
    counted_hit: jax.Array,
    confirmed_hits_before_event: jax.Array,
    elapsed_time_s: jax.Array,
    weight: float,
    time_scale_s: float,
) -> jax.Array:
    """Pay a bounded one-time bonus for a real first hit, decaying with delay."""
    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("first-hit promptness weight must be nonnegative")
    if not math.isfinite(float(time_scale_s)) or time_scale_s <= 0.0:
        raise ValueError("first-hit promptness time scale must be positive")
    first_hit = (
        jnp.asarray(counted_hit, dtype=jnp.bool_)
        & (jnp.asarray(confirmed_hits_before_event) == 0)
    )
    elapsed = jnp.maximum(jnp.asarray(elapsed_time_s, dtype=jnp.float32), 0.0)
    return (
        float(weight)
        * jnp.exp(-elapsed / float(time_scale_s))
        * first_hit.astype(jnp.float32)
    )
