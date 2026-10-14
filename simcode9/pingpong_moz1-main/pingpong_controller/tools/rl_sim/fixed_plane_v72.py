"""Opt-in one-contact ownership for fixed-plane hit confirmation."""

import jax
import jax.numpy as jnp


def atomic_contact_restart_events(
    *,
    raw_edge: jax.Array,
    logical_hit_edge: jax.Array,
    previous_pending: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Close an unconfirmed impact when a new penetrating edge arrives.

    The caller must provide edges made with nonpositive MJX contact gap.
    Every later edge opens a new confirmation window, so its outgoing ball
    velocity cannot retroactively confirm the previous impact.
    """

    restart = (
        jnp.asarray(raw_edge, dtype=jnp.bool_)
        & jnp.asarray(previous_pending, dtype=jnp.bool_)
    )
    return jnp.asarray(logical_hit_edge, dtype=jnp.bool_) | restart, restart
