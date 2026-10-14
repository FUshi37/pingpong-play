"""Deployment-available position history without ballistic actor features."""

from __future__ import annotations

import jax
import jax.numpy as jnp

ACTOR_OBS_DIM = 72
ACTOR_OBSERVATION_SCHEMA = "fixed_plane_v149_position_only_contiguous_72d"


def validate_simple_observation_contract(
    *, schema: str, dim: int, history_enabled: bool, contiguous_enabled: bool,
    high_latency_obs: bool, delay_conditioning: bool,
) -> None:
    """Fail closed when another actor layout or forecast augmentation is set."""
    if (schema != ACTOR_OBSERVATION_SCHEMA or dim != ACTOR_OBS_DIM
            or not history_enabled or not contiguous_enabled
            or high_latency_obs or delay_conditioning):
        raise ValueError("V149 simple actor requires its exact position-only 72-D contract")


def make_simple_position_observation(
    *, q: jax.Array, dq: jax.Array, q_ref_error: jax.Array,
    previous_action: jax.Array, ball_positions: jax.Array,
    racket_pose_now: jax.Array, ball_ages: jax.Array,
    target_xy_m: jax.Array, ball_position_valid: jax.Array,
) -> jax.Array:
    """Build (N,72) from seven-joint encoder/control state and position samples.

    History order is current to oldest, with eight consecutive 5 ms policy
    ticks. A 90 Hz camera sample is held between arrivals; ages describe the
    represented samples. Racket pose contains XYZ and the first two rotation
    axes, derived from current encoders. No phase, impact count, ball speed,
    forecast, true ball state, or RMP-internal state is consumed.
    """
    q = jnp.asarray(q, dtype=jnp.float32)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError("q must have shape (N,7)")
    n = q.shape[0]
    inputs = (
        (dq, (n, 7)), (q_ref_error, (n, 7)),
        (previous_action, (n, 7)), (ball_positions, (n, 8, 3)),
        (racket_pose_now, (n, 9)), (ball_ages, (n, 8)),
        (target_xy_m, (n, 2)), (ball_position_valid, (n,)),
    )
    if any(jnp.shape(value) != shape for value, shape in inputs):
        raise ValueError("V149 simple actor input shape mismatch")
    return jnp.concatenate((
        q, jnp.asarray(dq), jnp.asarray(q_ref_error),
        jnp.asarray(previous_action), jnp.asarray(ball_positions).reshape(n, 24),
        jnp.asarray(racket_pose_now),
        jnp.clip(jnp.asarray(ball_ages) / 0.25, 0.0, 1.0),
        jnp.asarray(target_xy_m),
        jnp.asarray(ball_position_valid, dtype=jnp.float32)[:, None],
    ), axis=-1).astype(jnp.float32)
