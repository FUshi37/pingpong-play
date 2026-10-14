"""Bounded, weak-inclusive steady-height reward for the 1 ms V53 course."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def steady_height_score(
    *,
    event: jax.Array,
    confirmed_hits_before_event: jax.Array,
    apex_z_m: jax.Array,
    previous_signed_error_m: jax.Array,
    previous_valid: jax.Array,
    weight: float,
    high_start_z_m: float = 1.450,
    high_cost_weight: float = 3.0,
    pair_cost_weight: float = 4.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Score each late rebound and charge adjacent height jumps.

    The absolute floor term keeps repeated low rebounds costly. The adjacent
    term makes a high/low pair worse than two equal-height acceptable rebounds.
    No credit is paid before three confirmed hits or without a resolved launch.
    """

    apex = jnp.asarray(apex_z_m)
    previous = jnp.asarray(previous_signed_error_m)
    active = jnp.asarray(event, dtype=bool) & (
        jnp.asarray(confirmed_hits_before_event) >= 3
    )
    target_z_m = 1.425
    error = apex - target_z_m
    target_credit = 2.0 * jnp.exp(-0.5 * (error / 0.025) ** 2)
    low_cost = 8.0 * jnp.clip((1.400 - apex) / 0.050, 0.0, 2.0)
    high_cost = high_cost_weight * jnp.clip(
        (apex - high_start_z_m) / 0.050, 0.0, 2.0
    )
    absolute_score = target_credit - low_cost - high_cost
    adjacent_jump_m = jnp.abs(error - previous)
    pair_cost = pair_cost_weight * jnp.clip(
        (adjacent_jump_m - 0.025) / 0.050, 0.0, 2.0
    )
    pair_cost = jnp.where(jnp.asarray(previous_valid, dtype=bool), pair_cost, 0.0)
    return (
        jnp.where(active, weight * (absolute_score - pair_cost), 0.0),
        jnp.where(active, weight * absolute_score, 0.0),
        jnp.where(active, adjacent_jump_m, 0.0),
    )
