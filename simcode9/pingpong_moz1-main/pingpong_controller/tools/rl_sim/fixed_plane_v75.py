"""Causal same-target racket progress for return-flight acquisition."""

import math

import jax
import jax.numpy as jnp


def post_hit_same_target_xy_progress_reward(
    *,
    target_xy_m: jax.Array,
    racket_xy_before_m: jax.Array,
    racket_xy_after_m: jax.Array,
    target_valid: jax.Array,
    hit_count: jax.Array,
    pending_hit: jax.Array,
    in_contact: jax.Array,
    time_to_contact_s: jax.Array,
    lead_s: float,
    weight: float,
    max_step_m: float,
    min_hit_count: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reward measured racket progress to one held observation target.

    Both errors use the *same* target from the step-start policy observation.
    Target publication alone therefore cannot create progress reward. The
    bounded per-step credit prevents rare kinematic jumps dominating PPO.
    """

    if not math.isfinite(lead_s) or lead_s < 0.0:
        raise ValueError("lead_s must be finite and nonnegative")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("weight must be finite and nonnegative")
    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError("max_step_m must be finite and positive")
    if min_hit_count < 1:
        raise ValueError("min_hit_count must be positive")
    target = jnp.asarray(target_xy_m, dtype=jnp.float32)
    before = jnp.linalg.norm(
        jnp.asarray(racket_xy_before_m, dtype=jnp.float32) - target,
        axis=-1,
    )
    after = jnp.linalg.norm(
        jnp.asarray(racket_xy_after_m, dtype=jnp.float32) - target,
        axis=-1,
    )
    active = (
        jnp.asarray(target_valid, dtype=jnp.bool_)
        & (jnp.asarray(hit_count) >= min_hit_count)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (~jnp.asarray(in_contact, dtype=jnp.bool_))
        & (jnp.asarray(time_to_contact_s) > lead_s)
    )
    progress = jnp.clip(before - after, -max_step_m, max_step_m)
    return jnp.where(active, weight * progress, 0.0), active, before, after
