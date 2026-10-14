"""Plant-aware nominal command and observation contracts for fixed-plane V1.

This module contains only causal, deployment-available quantities.  The
analytic command is the policy's nominal QVEL reference before the existing
bounded measured-state reference and recovered-RMP/PD plant; it is not a
second downstream planner.  A learned policy contributes a small normalized
residual around that reference.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


PLANT_AWARE_FIXED_PLANE_ACTOR_OBS_DIM = 110
PLANT_AWARE_FIXED_PLANE_ACTOR_OBSERVATION_SCHEMA_V1 = (
    "plant_aware_fixed_plane_v1_outcome_phase_history_110d"
)
PLANT_AWARE_FIXED_PLANE_ACTOR_OBS_SLICES = {
    "legacy_fixed_contact_prefix": slice(0, 80),
    "desired_post_ball_velocity": slice(80, 83),
    "desired_racket_contact_velocity": slice(83, 86),
    "ball_observation_age": slice(86, 87),
    "phase_one_hot": slice(87, 93),
    "time_since_contact": slice(93, 94),
    "pending_hit": slice(94, 95),
    "hit_count": slice(95, 96),
    "measured_dq_delta": slice(96, 103),
    "nominal_normalized_qvel": slice(103, 110),
}

PLANT_AWARE_PHASE_INVALID = 0
PLANT_AWARE_PHASE_PREPARE = 1
PLANT_AWARE_PHASE_ACCELERATE = 2
PLANT_AWARE_PHASE_BRAKE = 3
PLANT_AWARE_PHASE_RETRACT = 4
PLANT_AWARE_PHASE_RECOVER = 5
PLANT_AWARE_PHASE_COUNT = 6


class PlantAwarePhaseReference(NamedTuple):
    """Causal phase label and vertical surface reference."""

    phase_id: jax.Array
    phase_one_hot: jax.Array
    z_offset_m: jax.Array
    z_velocity_m_s: jax.Array


def restitution_consistent_racket_velocity(
    incoming_ball_velocity_m_s: jax.Array,
    desired_post_ball_velocity_m_s: jax.Array,
    *,
    effective_restitution: float,
) -> jax.Array:
    """Return the no-slip-free normal-impact racket velocity target.

    The scalar collision relation is
    ``v_out = (1 + e) * v_racket - e * v_in``.  Applying the same bounded
    geometric inverse to all three world components is an explicit nominal
    approximation; the residual policy and native contact dynamics retain
    responsibility for tangential effects.
    """

    restitution = float(effective_restitution)
    if not math.isfinite(restitution) or not 0.0 <= restitution <= 1.0:
        raise ValueError("effective_restitution must be finite and in [0, 1]")
    incoming = jnp.asarray(incoming_ball_velocity_m_s, dtype=jnp.float32)
    desired = jnp.asarray(desired_post_ball_velocity_m_s, dtype=jnp.float32)
    if incoming.shape != desired.shape or incoming.shape[-1] != 3:
        raise ValueError("incoming and desired velocities must share shape (..., 3)")
    return (desired + restitution * incoming) / (1.0 + restitution)


def plant_aware_phase_reference(
    *,
    target_valid: jax.Array,
    time_to_contact_s: jax.Array,
    incoming_ball_vz_m_s: jax.Array,
    pending_hit: jax.Array,
    pending_hit_steps: jax.Array,
    hit_count: jax.Array,
    dt_s: float,
    stroke_lead_s: float,
    brake_window_s: float,
    prestroke_depth_m: float,
    contact_velocity_m_s: jax.Array,
    retract_window_s: float,
    retract_velocity_m_s: float,
) -> PlantAwarePhaseReference:
    """Construct a C1 pulse/brake reference and explicit six-state phase.

    The accelerate segment holds the pre-stroke depth while a Hann-shaped
    upward velocity pulse excites the plant.  The final brake segment uses a
    cubic smoothstep to return from ``-depth`` to the fixed contact plane with
    zero velocity at both boundaries.  This preserves the successful causal
    timing of the frozen no-learning witness without its position step.
    """

    dt = float(dt_s)
    lead = float(stroke_lead_s)
    brake = float(brake_window_s)
    depth = float(prestroke_depth_m)
    retract_window = float(retract_window_s)
    retract_velocity = float(retract_velocity_m_s)
    values = (dt, lead, brake, depth, retract_window)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("phase timing and prestroke depth must be positive finite")
    if brake >= lead:
        raise ValueError("brake_window_s must be smaller than stroke_lead_s")
    if not math.isfinite(retract_velocity) or retract_velocity >= 0.0:
        raise ValueError("retract_velocity_m_s must be finite and negative")

    valid = jnp.asarray(target_valid, dtype=bool)
    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    incoming_vz = jnp.asarray(incoming_ball_vz_m_s, dtype=jnp.float32)
    pending = jnp.asarray(pending_hit, dtype=bool)
    pending_steps_value = jnp.asarray(pending_hit_steps, dtype=jnp.int32)
    hits = jnp.asarray(hit_count, dtype=jnp.int32)
    contact_velocity = jnp.asarray(contact_velocity_m_s, dtype=jnp.float32)

    descending = incoming_vz < 0.0
    retract = pending & (pending_steps_value * dt <= retract_window)
    prepare = valid & descending & (~pending) & (ttc > lead)
    accelerate = (
        valid
        & descending
        & (~pending)
        & (ttc > brake)
        & (ttc <= lead)
    )
    final_brake = (
        valid
        & descending
        & (~pending)
        & (ttc > 0.0)
        & (ttc <= brake)
    )
    recover = (~retract) & (hits > 0) & ~(accelerate | final_brake)

    accelerate_duration = lead - brake
    accelerate_progress = jnp.clip(
        (lead - ttc) / accelerate_duration, 0.0, 1.0
    )
    accelerate_velocity = contact_velocity * 0.5 * (
        1.0 - jnp.cos(2.0 * jnp.pi * accelerate_progress)
    )
    brake_progress = jnp.clip((brake - ttc) / brake, 0.0, 1.0)
    smoothstep = 3.0 * brake_progress**2 - 2.0 * brake_progress**3
    smoothstep_rate = (
        (6.0 * brake_progress - 6.0 * brake_progress**2) / brake
    )
    brake_position = -depth * (1.0 - smoothstep)
    brake_velocity = depth * smoothstep_rate

    z_offset = jnp.full(ttc.shape, -depth, dtype=jnp.float32)
    z_offset = jnp.where(final_brake, brake_position, z_offset)
    z_velocity = jnp.where(accelerate, accelerate_velocity, 0.0)
    z_velocity = jnp.where(final_brake, brake_velocity, z_velocity)
    retract_elapsed = jnp.minimum(
        pending_steps_value.astype(jnp.float32) * dt,
        retract_window,
    )
    z_offset = jnp.where(
        retract,
        jnp.maximum(-depth, retract_velocity * retract_elapsed),
        z_offset,
    )
    z_velocity = jnp.where(retract, retract_velocity, z_velocity)

    phase_id = jnp.full(ttc.shape, PLANT_AWARE_PHASE_INVALID, dtype=jnp.int32)
    phase_id = jnp.where(prepare, PLANT_AWARE_PHASE_PREPARE, phase_id)
    phase_id = jnp.where(accelerate, PLANT_AWARE_PHASE_ACCELERATE, phase_id)
    phase_id = jnp.where(final_brake, PLANT_AWARE_PHASE_BRAKE, phase_id)
    phase_id = jnp.where(retract, PLANT_AWARE_PHASE_RETRACT, phase_id)
    phase_id = jnp.where(recover, PLANT_AWARE_PHASE_RECOVER, phase_id)
    one_hot = jax.nn.one_hot(
        phase_id, PLANT_AWARE_PHASE_COUNT, dtype=jnp.float32
    )
    return PlantAwarePhaseReference(
        phase_id=phase_id,
        phase_one_hot=one_hot,
        z_offset_m=z_offset,
        z_velocity_m_s=z_velocity,
    )


def compose_bounded_residual_action(
    nominal_normalized_action: jax.Array,
    policy_residual_action: jax.Array,
    *,
    residual_scale: float,
) -> jax.Array:
    """Compose an analytic normalized command and bounded learned residual."""

    scale = float(residual_scale)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("residual_scale must be finite and in [0, 1]")
    nominal = jnp.asarray(nominal_normalized_action, dtype=jnp.float32)
    residual = jnp.asarray(policy_residual_action, dtype=jnp.float32)
    if nominal.shape != residual.shape:
        raise ValueError("nominal and residual actions must have matching shapes")
    return jnp.clip(
        nominal + scale * jnp.clip(residual, -1.0, 1.0),
        -1.0,
        1.0,
    )


def apply_post_first_contact_slew(
    unslewed_qvel_rad_s: jax.Array,
    slew_limited_qvel_rad_s: jax.Array,
    hit_count: jax.Array,
) -> jax.Array:
    """Preserve fast acquisition; enable acceleration-derived slew afterward."""

    unslewed = jnp.asarray(unslewed_qvel_rad_s, dtype=jnp.float32)
    limited = jnp.asarray(slew_limited_qvel_rad_s, dtype=jnp.float32)
    if unslewed.shape != limited.shape:
        raise ValueError("slewed and unslewed QVEL arrays must match")
    return jnp.where(
        (jnp.asarray(hit_count, dtype=jnp.int32) > 0)[:, None],
        limited,
        unslewed,
    )


def apply_rmp_tracking_precompensation(
    nominal_qvel_rad_s: jax.Array,
    measured_qvel_rad_s: jax.Array,
    accelerate_phase: jax.Array,
    tracking_quality_window: jax.Array,
    *,
    transient_scale: tuple[float, ...],
    correction_gain: float,
) -> jax.Array:
    """Apply the witness-identified upstream correction for RMP/PD lag."""

    nominal = jnp.asarray(nominal_qvel_rad_s, dtype=jnp.float32)
    measured = jnp.asarray(measured_qvel_rad_s, dtype=jnp.float32)
    if nominal.ndim != 2 or nominal.shape != measured.shape:
        raise ValueError("nominal and measured QVEL must share shape (N, J)")
    scale = tuple(float(value) for value in transient_scale)
    if len(scale) != nominal.shape[-1] or any(
        not math.isfinite(value) or value <= 0.0 for value in scale
    ):
        raise ValueError(
            "transient_scale must contain one positive finite value per joint"
        )
    gain = float(correction_gain)
    if not math.isfinite(gain) or not 0.0 <= gain <= 1.0:
        raise ValueError("correction_gain must be finite and in [0, 1]")
    accelerated = jnp.where(
        jnp.asarray(accelerate_phase, dtype=bool)[:, None],
        nominal * jnp.asarray(scale, dtype=jnp.float32)[None, :],
        nominal,
    )
    return jnp.where(
        jnp.asarray(tracking_quality_window, dtype=bool)[:, None],
        accelerated + gain * (nominal - measured),
        accelerated,
    )


def make_plant_aware_fixed_plane_observation(
    *,
    fixed_contact_prefix_80d: jax.Array,
    desired_post_ball_velocity_m_s: jax.Array,
    desired_racket_contact_velocity_m_s: jax.Array,
    ball_observation_age_s: jax.Array,
    phase_one_hot: jax.Array,
    time_since_contact_s: jax.Array,
    pending_hit: jax.Array,
    hit_count: jax.Array,
    measured_dq_delta_rad_s: jax.Array,
    nominal_normalized_qvel: jax.Array,
    ball_age_clip_s: float = 0.20,
    contact_time_clip_s: float = 0.30,
    hit_count_clip: int = 16,
) -> jax.Array:
    """Append outcome intent, phase, causal history, and nominal command."""

    age_clip = float(ball_age_clip_s)
    contact_clip = float(contact_time_clip_s)
    hit_clip = int(hit_count_clip)
    if age_clip <= 0.0 or contact_clip <= 0.0 or hit_clip <= 0:
        raise ValueError("observation normalization clips must be positive")
    prefix = jnp.asarray(fixed_contact_prefix_80d, dtype=jnp.float32)
    if prefix.ndim != 2 or prefix.shape[-1] != 80:
        raise ValueError("fixed-contact observation prefix must have shape (N, 80)")
    phase = jnp.asarray(phase_one_hot, dtype=jnp.float32)
    if phase.shape != (prefix.shape[0], PLANT_AWARE_PHASE_COUNT):
        raise ValueError("phase_one_hot must have shape (N, 6)")
    observation = jnp.concatenate(
        [
            prefix,
            jnp.asarray(desired_post_ball_velocity_m_s, dtype=jnp.float32),
            jnp.asarray(desired_racket_contact_velocity_m_s, dtype=jnp.float32),
            jnp.clip(
                jnp.asarray(ball_observation_age_s, dtype=jnp.float32)
                / age_clip,
                0.0,
                1.0,
            )[:, None],
            phase,
            jnp.clip(
                jnp.asarray(time_since_contact_s, dtype=jnp.float32)
                / contact_clip,
                0.0,
                1.0,
            )[:, None],
            jnp.asarray(pending_hit, dtype=jnp.float32)[:, None],
            jnp.clip(
                jnp.asarray(hit_count, dtype=jnp.float32) / float(hit_clip),
                0.0,
                1.0,
            )[:, None],
            jnp.asarray(measured_dq_delta_rad_s, dtype=jnp.float32),
            jnp.asarray(nominal_normalized_qvel, dtype=jnp.float32),
        ],
        axis=-1,
    )
    if observation.shape[-1] != PLANT_AWARE_FIXED_PLANE_ACTOR_OBS_DIM:
        raise ValueError("plant-aware observation assembly produced wrong width")
    return observation
