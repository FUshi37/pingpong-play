"""Outcome-aligned separation rewards for a third pending contact."""

import math

import jax
import jax.numpy as jnp

from fixed_contact_plane_training import fixed_contact_pending_hit_retract_vz_reward


def post_second_pending_clearance_credit(
    *,
    pending_hit: jax.Array,
    failed_hit: jax.Array,
    previous_hit_count: jax.Array,
    ball_vz_m_s: jax.Array,
    relative_z_m: jax.Array,
    previous_relative_z_m: jax.Array,
    contact_baseline_relative_z_m: jax.Array,
    confirm_relative_z_m: float,
    weight: float,
) -> jax.Array:
    """Pay physical upward ball/racket separation, reclaim it on failure."""

    if not math.isfinite(float(weight)) or weight < 0.0:
        raise ValueError("clearance weight must be finite and nonnegative")
    if not math.isfinite(float(confirm_relative_z_m)) or confirm_relative_z_m <= 0.0:
        raise ValueError("confirmation clearance must be finite and positive")
    baseline = jnp.asarray(contact_baseline_relative_z_m, dtype=jnp.float32)
    span = jnp.maximum(float(confirm_relative_z_m) - baseline, 0.005)
    current = jnp.clip((jnp.asarray(relative_z_m) - baseline) / span, 0.0, 1.0)
    previous = jnp.clip(
        (jnp.asarray(previous_relative_z_m) - baseline) / span, 0.0, 1.0
    )
    delta = current - previous
    late = jnp.asarray(previous_hit_count) >= 2
    pending = jnp.asarray(pending_hit, dtype=jnp.bool_) & late
    failed = jnp.asarray(failed_hit, dtype=jnp.bool_) & late
    upward = jnp.asarray(ball_vz_m_s) > 0.0
    signed_progress = jnp.where(delta > 0.0, jnp.where(upward, delta, 0.0), delta)
    return float(weight) * (
        jnp.where(pending, signed_progress, 0.0)
        - jnp.where(failed, current, 0.0)
    )


def post_second_pending_retract_reward(
    *,
    racket_vz_m_s: jax.Array,
    pending_hit: jax.Array,
    pending_hit_steps: jax.Array,
    previous_hit_count: jax.Array,
    target_retract_vz_m_s: float,
    active_window_s: float,
    reward_weight_per_s: float,
    dt_s: float,
) -> tuple[jax.Array, jax.Array]:
    """Reward braking/retraction briefly after third or later physical contact."""

    reward, _, active = fixed_contact_pending_hit_retract_vz_reward(
        racket_vz_m_s=racket_vz_m_s,
        pending_hit=pending_hit,
        pending_hit_steps=pending_hit_steps,
        hit_count=previous_hit_count,
        target_retract_vz_m_s=target_retract_vz_m_s,
        active_window_s=active_window_s,
        reward_weight=reward_weight_per_s,
        dt_s=dt_s,
        all_confirmed_hits=True,
    )
    after_two = jnp.asarray(previous_hit_count) >= 2
    return jnp.where(after_two, reward, 0.0), active & after_two
