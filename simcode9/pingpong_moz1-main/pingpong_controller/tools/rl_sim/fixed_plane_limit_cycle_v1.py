"""Causal contracts for the opt-in fixed-plane limit-cycle V1 method.

The module deliberately contains no controller, phase teacher, or simulator-
only state.  It defines one goal-conditioned launch-to-steady apex schedule
and the 16-value suffix appended to the preserved 80-D fixed-contact actor
prefix.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


FIXED_PLANE_LIMIT_CYCLE_ACTOR_OBS_DIM = 96
FIXED_PLANE_LIMIT_CYCLE_ACTOR_OBS_DIM_V63 = 98
FIXED_PLANE_LIMIT_CYCLE_ACTOR_OBSERVATION_SCHEMA_V63 = (
    "fixed_plane_limit_cycle_v63_previous_apex_error_98d"
)
FIXED_PLANE_LIMIT_CYCLE_ACTOR_OBSERVATION_SCHEMA_V1 = (
    "fixed_plane_limit_cycle_v1_goal_conditioned_96d"
)
FIXED_PLANE_LIMIT_CYCLE_ACTOR_OBS_SLICES = {
    "fixed_contact_prefix": slice(0, 80),
    "desired_post_ball_velocity": slice(80, 83),
    "desired_racket_contact_velocity": slice(83, 86),
    "desired_apex_z": slice(86, 87),
    "measured_dq_delta": slice(87, 94),
    "ball_observation_age": slice(94, 95),
    "apex_progress": slice(95, 96),
}


def limit_cycle_apex_progress(
    confirmed_hit_count: jax.Array,
    *,
    transition_hits: int,
) -> jax.Array:
    """Return the observable launch-to-steady interpolation in ``[0, 1]``.

    With the registered two-hit transition, a pre-contact hit count of zero
    requests the launch apex, one requests the midpoint, and two or more
    request the steady apex.  The same scalar conditions geometry, observation
    and reward so launch and steady objectives are never simultaneously active.
    """

    count = int(transition_hits)
    if count <= 0:
        raise ValueError("transition_hits must be positive")
    hits = jnp.asarray(confirmed_hit_count, dtype=jnp.float32)
    return jnp.clip(hits / float(count), 0.0, 1.0)


def limit_cycle_desired_apex_z_m(
    confirmed_hit_count: jax.Array,
    *,
    launch_apex_z_m: float,
    steady_apex_z_m: float,
    transition_hits: int,
) -> jax.Array:
    """Interpolate the single active absolute apex target for each lane."""

    launch = float(launch_apex_z_m)
    steady = float(steady_apex_z_m)
    if not math.isfinite(launch) or not math.isfinite(steady):
        raise ValueError("apex targets must be finite")
    if launch <= 0.0 or steady <= launch:
        raise ValueError("steady apex must be greater than positive launch apex")
    progress = limit_cycle_apex_progress(
        confirmed_hit_count,
        transition_hits=transition_hits,
    )
    return launch + progress * (steady - launch)


def limit_cycle_hit_apex_thresholds_m(
    desired_apex_z_m: jax.Array,
    *,
    contact_ball_center_z_m: jax.Array,
    quality_margin_m: float,
    survival_margin_m: float,
) -> tuple[jax.Array, jax.Array]:
    """Align hit confirmation with the one active launch/steady objective.

    The legacy confirmation threshold is tied to a fixed relative-height
    course and would reject a deliberately low placed-ball launch.  This
    opt-in contract instead defines absolute thresholds below the actor-visible
    apex target while retaining a minimum clearance above the contact plane.
    """

    quality_margin = float(quality_margin_m)
    survival_margin = float(survival_margin_m)
    if not math.isfinite(quality_margin) or quality_margin <= 0.0:
        raise ValueError("quality_margin_m must be finite and positive")
    if not math.isfinite(survival_margin) or survival_margin < quality_margin:
        raise ValueError(
            "survival_margin_m must be finite and >= quality_margin_m"
        )
    desired = jnp.asarray(desired_apex_z_m, dtype=jnp.float32)
    contact_center = jnp.asarray(
        contact_ball_center_z_m, dtype=jnp.float32
    )
    if desired.shape != contact_center.shape:
        raise ValueError("desired apex and contact center must share shape")
    quality = jnp.maximum(desired - quality_margin, contact_center + 0.030)
    survival = jnp.maximum(desired - survival_margin, contact_center + 0.015)
    return jnp.minimum(survival, quality), quality


def restitution_consistent_racket_velocity_m_s(
    incoming_ball_velocity_m_s: jax.Array,
    desired_post_ball_velocity_m_s: jax.Array,
    *,
    effective_restitution: float,
) -> jax.Array:
    """Return a causal nominal contact-velocity intent for actor conditioning.

    This is an observation target, not a controller or collision override.  It
    uses ``v_out = (1 + e) v_racket - e v_in`` componentwise; native MuJoCo
    contact remains solely responsible for the realized outcome.
    """

    restitution = float(effective_restitution)
    if not math.isfinite(restitution) or not 0.0 <= restitution <= 1.0:
        raise ValueError("effective_restitution must be finite and in [0, 1]")
    incoming = jnp.asarray(incoming_ball_velocity_m_s, dtype=jnp.float32)
    desired = jnp.asarray(desired_post_ball_velocity_m_s, dtype=jnp.float32)
    if incoming.shape != desired.shape or incoming.shape[-1] != 3:
        raise ValueError("incoming and desired velocities must share shape (..., 3)")
    return (desired + restitution * incoming) / (1.0 + restitution)


def limit_cycle_dynamic_racket_vz_target_m_s(
    *,
    incoming_ball_vz_m_s: jax.Array,
    desired_apex_z_m: jax.Array,
    contact_ball_center_z_m: float,
    gravity_m_s2: float,
    effective_restitution: float,
    min_racket_vz_m_s: float,
    max_racket_vz_m_s: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return raw and executable per-impact vertical racket-speed targets.

    The desired outgoing ball speed is the ballistic speed needed to reach
    the lane's current absolute apex goal.  The normal restitution inverse is
    then solved from the predicted incoming vertical speed.  Clipping is an
    explicit actuator/workspace feasibility boundary, not an unreported
    change to the apex goal.
    """

    contact_center = float(contact_ball_center_z_m)
    gravity = abs(float(gravity_m_s2))
    restitution = float(effective_restitution)
    minimum = float(min_racket_vz_m_s)
    maximum = float(max_racket_vz_m_s)
    values = (contact_center, gravity, restitution, minimum, maximum)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("dynamic contact target parameters must be finite")
    if gravity <= 0.0:
        raise ValueError("gravity magnitude must be positive")
    if not 0.0 <= restitution <= 1.0:
        raise ValueError("effective restitution must lie in [0, 1]")
    if minimum < 0.0 or maximum <= minimum:
        raise ValueError("dynamic contact racket-vz bounds must be ordered")

    incoming = jnp.asarray(incoming_ball_vz_m_s, dtype=jnp.float32)
    apex = jnp.asarray(desired_apex_z_m, dtype=jnp.float32)
    if incoming.shape != apex.shape:
        raise ValueError("incoming ball vz and desired apex must share shape")
    apex_height = jnp.maximum(apex - contact_center, 0.0)
    desired_outgoing_vz = jnp.sqrt(2.0 * gravity * apex_height)
    raw_target = (
        desired_outgoing_vz + restitution * incoming
    ) / (1.0 + restitution)
    executable_target = jnp.clip(raw_target, minimum, maximum)
    clipped = jnp.abs(executable_target - raw_target) > 1.0e-6
    return raw_target, executable_target, clipped


def limit_cycle_workspace_bounded_stroke_reference(
    *,
    time_to_contact_s: jax.Array,
    target_contact_vz_m_s: jax.Array,
    crossing_valid: jax.Array,
    pending_hit: jax.Array,
    hit_count: jax.Array,
    contact_surface_z_m: float,
    prestroke_depth_m: float,
    prepare_s: float,
    min_hit_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Build a monotone, depth-bounded stroke ending at the contact plane.

    A cubic smoothstep first moves from the contact plane to a bounded
    pre-stroke depth with zero endpoint velocity.  The terminal segment then
    uses constant acceleration from rest, with duration
    ``2 * effective_depth / contact_speed``.  The effective depth shrinks for
    low requested speeds so at least half of ``prepare_s`` remains available
    for the smooth downstroke.  Position and velocity are continuous at both
    phase boundaries, position stays in ``[surface-depth, surface]``, and the
    lane-specific contact velocity is reached exactly at the plane.
    """

    surface = float(contact_surface_z_m)
    depth = float(prestroke_depth_m)
    prepare = float(prepare_s)
    minimum_hits = int(min_hit_count)
    if not all(math.isfinite(value) for value in (surface, depth, prepare)):
        raise ValueError("workspace-bounded stroke parameters must be finite")
    if depth <= 0.0 or prepare <= 0.0:
        raise ValueError("stroke depth and prepare time must be positive")
    if minimum_hits < 0:
        raise ValueError("stroke minimum hit count must be nonnegative")

    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    speed = jnp.asarray(target_contact_vz_m_s, dtype=jnp.float32)
    if ttc.shape != speed.shape:
        raise ValueError("time-to-contact and target speed must share shape")
    nonnegative_speed = jnp.maximum(speed, 0.0)
    effective_depth = jnp.minimum(
        depth,
        0.25 * nonnegative_speed * prepare,
    )
    terminal_duration = jnp.where(
        nonnegative_speed > 1.0e-6,
        2.0 * effective_depth / jnp.maximum(nonnegative_speed, 1.0e-6),
        0.5 * prepare,
    )
    descent_duration = jnp.maximum(prepare - terminal_duration, 1.0e-6)
    descent_progress = jnp.clip(
        (prepare - ttc) / descent_duration,
        0.0,
        1.0,
    )
    descent_smoothstep = (
        3.0 * descent_progress * descent_progress
        - 2.0 * descent_progress * descent_progress * descent_progress
    )
    descent_smoothstep_rate = (
        6.0 * descent_progress
        - 6.0 * descent_progress * descent_progress
    ) / descent_duration
    descent_z = surface - effective_depth * descent_smoothstep
    descent_vz = -effective_depth * descent_smoothstep_rate

    strike_progress = jnp.clip(
        (terminal_duration - ttc) / jnp.maximum(terminal_duration, 1.0e-6),
        0.0,
        1.0,
    )
    strike_z = (
        surface
        - effective_depth
        + effective_depth * strike_progress * strike_progress
    )
    strike_vz = nonnegative_speed * strike_progress
    striking = ttc <= terminal_duration
    target_z = jnp.where(striking, strike_z, descent_z)
    target_vz = jnp.where(striking, strike_vz, descent_vz)
    active = (
        jnp.asarray(crossing_valid, dtype=jnp.bool_)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (jnp.asarray(hit_count) >= minimum_hits)
        & (ttc >= 0.0)
        & (ttc <= prepare)
    )
    target_z = jnp.where(active, target_z, surface)
    target_vz = jnp.where(active, target_vz, 0.0)
    return target_z, target_vz, active


def make_fixed_plane_limit_cycle_observation(
    *,
    fixed_contact_prefix_80d: jax.Array,
    desired_post_ball_velocity_m_s: jax.Array,
    desired_racket_contact_velocity_m_s: jax.Array,
    desired_apex_z_m: jax.Array,
    measured_dq_delta_rad_s: jax.Array,
    ball_observation_age_s: jax.Array,
    apex_progress: jax.Array,
    ball_age_clip_s: float,
) -> jax.Array:
    """Append the deployment-available 16-D limit-cycle context."""

    prefix = jnp.asarray(fixed_contact_prefix_80d, dtype=jnp.float32)
    post = jnp.asarray(desired_post_ball_velocity_m_s, dtype=jnp.float32)
    racket = jnp.asarray(desired_racket_contact_velocity_m_s, dtype=jnp.float32)
    apex = jnp.asarray(desired_apex_z_m, dtype=jnp.float32)
    dq_delta = jnp.asarray(measured_dq_delta_rad_s, dtype=jnp.float32)
    age = jnp.asarray(ball_observation_age_s, dtype=jnp.float32)
    progress = jnp.asarray(apex_progress, dtype=jnp.float32)
    if prefix.ndim != 2 or prefix.shape[-1] != 80:
        raise ValueError("fixed-contact prefix must have shape (N, 80)")
    n_envs = prefix.shape[0]
    if post.shape != (n_envs, 3) or racket.shape != (n_envs, 3):
        raise ValueError("desired velocities must have shape (N, 3)")
    if dq_delta.shape != (n_envs, 7):
        raise ValueError("measured_dq_delta_rad_s must have shape (N, 7)")
    for name, value in (
        ("desired_apex_z_m", apex),
        ("ball_observation_age_s", age),
        ("apex_progress", progress),
    ):
        if value.shape != (n_envs,):
            raise ValueError(f"{name} must have shape (N,)")
    age_clip = float(ball_age_clip_s)
    if not math.isfinite(age_clip) or age_clip <= 0.0:
        raise ValueError("ball_age_clip_s must be finite and positive")
    return jnp.concatenate(
        [
            prefix,
            post,
            racket,
            apex[:, None],
            dq_delta,
            jnp.clip(age / age_clip, 0.0, 1.0)[:, None],
            jnp.clip(progress, 0.0, 1.0)[:, None],
        ],
        axis=-1,
    )
