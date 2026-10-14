"""V93 deployment-available predicted contact geometry for the actor."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from fixed_plane_v92 import rotation_matrix_to_6d


ACTOR_OBS_DIM = 156
ACTOR_OBSERVATION_SCHEMA = "fixed_plane_v93_history_contact_geometry_no_ball_velocity_156d"


def make_predicted_contact_context(
    *, predicted_ball_position_m: jax.Array,
    time_to_contact_s: jax.Array, contact_valid: jax.Array,
    racket_target_position_m: jax.Array,
    racket_target_rotation: jax.Array,
    racket_surface_position_m: jax.Array,
    next_crossing_xy_m: jax.Array,
) -> jax.Array:
    """Return 19-D future geometry/time; no ball-velocity or speed-target slots."""
    ball = jnp.asarray(predicted_ball_position_m, dtype=jnp.float32)
    target = jnp.asarray(racket_target_position_m, dtype=jnp.float32)
    surface = jnp.asarray(racket_surface_position_m, dtype=jnp.float32)
    crossing = jnp.asarray(next_crossing_xy_m, dtype=jnp.float32)
    rotation = jnp.asarray(racket_target_rotation, dtype=jnp.float32)
    n_envs = ball.shape[0]
    if (
        ball.shape != (n_envs, 3)
        or target.shape != (n_envs, 3)
        or surface.shape != (n_envs, 3)
        or crossing.shape != (n_envs, 2)
        or rotation.shape != (n_envs, 3, 3)
        or jnp.shape(time_to_contact_s) != (n_envs,)
        or jnp.shape(contact_valid) != (n_envs,)
    ):
        raise ValueError("V93 contact context input shape mismatch")
    return jnp.concatenate((
        ball,
        jnp.clip(jnp.asarray(time_to_contact_s), 0.0, 2.0)[:, None],
        jnp.asarray(contact_valid, dtype=jnp.float32)[:, None],
        target,
        rotation_matrix_to_6d(rotation),
        target - surface,
        crossing,
    ), axis=-1)
