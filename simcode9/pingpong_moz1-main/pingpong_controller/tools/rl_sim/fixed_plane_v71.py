"""Opt-in vertical separation before rearming a pending racket impact."""

import math

import jax
import jax.numpy as jnp


def fixed_contact_rearm_eligible(
    *,
    in_contact: jax.Array,
    no_contact_steps: jax.Array,
    separation_distance_m: jax.Array,
    vertical_separation_m: jax.Array,
    required_no_contact_steps: int,
    required_separation_distance_m: float,
    min_vertical_separation_m: float,
) -> jax.Array:
    """Require vertical clearance in addition to the legacy 3-D gap."""

    if not math.isfinite(min_vertical_separation_m) or min_vertical_separation_m < 0:
        raise ValueError("minimum vertical rearm clearance must be nonnegative")
    return (
        (~jnp.asarray(in_contact, dtype=jnp.bool_))
        & (jnp.asarray(no_contact_steps) >= required_no_contact_steps)
        & (jnp.asarray(separation_distance_m) >= required_separation_distance_m)
        & (jnp.asarray(vertical_separation_m) >= min_vertical_separation_m)
    )
