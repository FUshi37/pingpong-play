"""Primitives for the opt-in fixed-contact-plane juggling course.

The functions in this module are JAX compatible and contain only quantities
available to the real policy path.  Simulator-only truth and domain-randomized
parameters deliberately stay out of the 80-D actor schema.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp


FIXED_CONTACT_ACTOR_OBS_DIM = 80
FIXED_CONTACT_ACTOR_OBSERVATION_SCHEMA_V1 = (
    "fixed_contact_plane_v1_separate_current_future_80d"
)
FIXED_CONTACT_ACTOR_OBSERVATION_SCHEMA_V15 = (
    "fixed_contact_plane_v15_target_conditioning_x10_80d"
)
FIXED_CONTACT_ACTOR_OBSERVATION_SCHEMA_V16 = (
    "fixed_contact_plane_v16_target_conditioning_x10_migrated_80d"
)
FIXED_CONTACT_ACTOR_OBS_SLICES = {
    "q": slice(0, 7),
    "dq": slice(7, 14),
    "q_ref_error": slice(14, 21),
    "previous_executed_action": slice(21, 28),
    # The current camera observation and the future ballistic contact state
    # intentionally occupy different slots.  The current fields are never
    # overwritten by the high-ball predictor.
    "observed_ball_position_current": slice(28, 31),
    "observed_ball_velocity_current": slice(31, 34),
    "predicted_ball_position_at_contact": slice(34, 37),
    "predicted_ball_velocity_at_contact": slice(37, 40),
    "prediction_horizon_to_contact": slice(40, 41),
    "prediction_valid": slice(41, 42),
    "prediction_uses_extrapolated_current_state": slice(42, 43),
    "racket_position_current": slice(43, 46),
    "racket_orientation_current_6d": slice(46, 52),
    "racket_linear_velocity": slice(52, 55),
    "racket_angular_velocity": slice(55, 58),
    "contact_target_position": slice(58, 61),
    "contact_target_orientation_6d": slice(61, 67),
    "contact_position_error": slice(67, 70),
    "contact_orientation_error": slice(70, 73),
    "observed_ball_racket_relative_position": slice(73, 76),
    "contact_target_relative_center_xy": slice(76, 78),
    "next_crossing_target_relative_center_xy": slice(78, 80),
}


def factorized_reset_lane_indices(
    *,
    n_envs: int,
    n_x: int,
    n_y: int,
    n_blends: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Assign lanes deterministically over the complete XY-by-blend product."""

    sizes = (int(n_envs), int(n_x), int(n_y), int(n_blends))
    if any(value <= 0 for value in sizes):
        raise ValueError("factorized reset dimensions must all be positive")
    lane_index = jnp.arange(sizes[0], dtype=jnp.int32)
    blend_index = lane_index % sizes[3]
    y_index = (lane_index // sizes[3]) % sizes[2]
    x_index = (lane_index // (sizes[3] * sizes[2])) % sizes[1]
    return x_index, y_index, blend_index


def factorized_reset_lane_indices_with_ttc(
    *,
    n_envs: int,
    n_x: int,
    n_y: int,
    n_blends: int,
    n_ttc: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Cover the complete XY-by-blend-by-raw-TTC validation product."""

    sizes = (int(n_envs), int(n_x), int(n_y), int(n_blends), int(n_ttc))
    if any(value <= 0 for value in sizes):
        raise ValueError("factorized reset dimensions must all be positive")
    lane_index = jnp.arange(sizes[0], dtype=jnp.int32)
    ttc_index = lane_index % sizes[4]
    blend_index = (lane_index // sizes[4]) % sizes[3]
    y_index = (lane_index // (sizes[4] * sizes[3])) % sizes[2]
    x_index = (
        lane_index // (sizes[4] * sizes[3] * sizes[2])
    ) % sizes[1]
    return x_index, y_index, blend_index, ttc_index


def feasibility_conditioned_reset_ttc_s(
    *,
    sampled_ttc_s: jax.Array,
    target_x_m: jax.Array,
    target_y_m: jax.Array,
    target_follow_blend: jax.Array,
    grid_x_m: tuple[float, ...],
    grid_y_m: tuple[float, ...],
    blend080_ttc_floor_grid_s: tuple[float, ...],
    zero_distance_ttc_floor_s: float,
    reference_blend: float = 0.80,
) -> jax.Array:
    """Raise TTC only enough to keep a registered XY/blend reset feasible.

    The table is row-major over ``grid_x_m`` then ``grid_y_m`` and records
    the minimum TTC for the reference reset distance.  A blend of one starts
    at the target, so it retains the independently registered zero-distance
    floor.  Intermediate blends linearly interpolate *the time floor*, not
    Cartesian distance; the table therefore remains the source of truth for
    the anisotropic closed-loop response.
    """

    x_grid = jnp.asarray(grid_x_m, dtype=jnp.float32)
    y_grid = jnp.asarray(grid_y_m, dtype=jnp.float32)
    floors = jnp.asarray(blend080_ttc_floor_grid_s, dtype=jnp.float32)
    expected = len(grid_x_m) * len(grid_y_m)
    if expected <= 0 or len(blend080_ttc_floor_grid_s) != expected:
        raise ValueError(
            "feasibility TTC floor table must match the XY grid product"
        )
    reference_distance = 1.0 - float(reference_blend)
    if not 0.0 < reference_distance <= 1.0:
        raise ValueError("reference blend must be in [0, 1)")
    base_floor = float(zero_distance_ttc_floor_s)
    if not math.isfinite(base_floor) or base_floor <= 0.0:
        raise ValueError("zero-distance TTC floor must be finite and positive")

    x_index = jnp.argmin(
        jnp.abs(jnp.asarray(target_x_m)[:, None] - x_grid[None, :]), axis=1
    )
    y_index = jnp.argmin(
        jnp.abs(jnp.asarray(target_y_m)[:, None] - y_grid[None, :]), axis=1
    )
    reference_floor = floors.reshape((len(grid_x_m), len(grid_y_m)))[
        x_index, y_index
    ]
    distance_fraction = jnp.clip(
        (1.0 - jnp.asarray(target_follow_blend, dtype=jnp.float32))
        / reference_distance,
        0.0,
        1.0,
    )
    conditioned_floor = base_floor + distance_fraction * (
        reference_floor - base_floor
    )
    return jnp.maximum(
        jnp.asarray(sampled_ttc_s, dtype=jnp.float32), conditioned_floor
    )


def seed_permuted_xy_lane_indices(
    key: jax.Array,
    *,
    n_envs: int,
    n_x: int,
    n_y: int,
) -> tuple[jax.Array, jax.Array]:
    """Cover every XY bin evenly without pinning it to fixed env lanes.

    The historical equal-bin evaluator assigned a given coordinate to the
    same lane indices for every seed.  Any lane-index-correlated reset state
    was therefore confounded with that coordinate.  A seed-deterministic
    permutation preserves the exact per-seed bin counts while rotating which
    environment lanes represent each bin across independent seeds.
    """

    sizes = (int(n_envs), int(n_x), int(n_y))
    if any(value <= 0 for value in sizes):
        raise ValueError("seed-permuted XY dimensions must all be positive")
    lane_index = jax.random.permutation(
        key,
        jnp.arange(sizes[0], dtype=jnp.int32),
    )
    x_index = (lane_index // sizes[2]) % sizes[1]
    y_index = lane_index % sizes[2]
    return x_index, y_index


def select_reset_ball_observation_velocity(
    initialized_velocity_m_s: jax.Array,
    observation_missing: jax.Array,
    *,
    use_initialized_velocity: bool,
) -> jax.Array:
    """Initialize reset-time velocity without leaking through missing vision."""

    velocity = jnp.asarray(initialized_velocity_m_s, dtype=jnp.float32)
    missing = jnp.asarray(observation_missing, dtype=jnp.bool_)
    if velocity.ndim != 2 or velocity.shape[-1] != 3:
        raise ValueError("initialized ball velocity must have shape (N, 3)")
    if missing.shape != velocity.shape[:-1]:
        raise ValueError("reset observation missing mask must have shape (N,)")
    if not bool(use_initialized_velocity):
        return jnp.zeros_like(velocity)
    return jnp.where(missing[:, None], jnp.zeros_like(velocity), velocity)


def sample_local_contact_xy_jitter(
    keys: jax.Array,
    *,
    maximum_m: float,
    minimum_radius_m: float = 0.0,
    mode: str = "square_uniform",
) -> jax.Array:
    """Sample a legacy square or a non-trivial local acquisition annulus.

    ``square_uniform`` deliberately preserves the historical PRNG expression
    byte-for-byte.  ``annulus_uniform`` samples area-uniformly so every reset
    requires at least ``minimum_radius_m`` of Cartesian acquisition instead
    of letting near-zero resets satisfy a precision curriculum by waiting.
    """

    maximum_m = float(maximum_m)
    minimum_radius_m = float(minimum_radius_m)
    if not math.isfinite(maximum_m) or maximum_m < 0.0:
        raise ValueError("maximum local contact jitter must be finite and >= 0")
    if (
        not math.isfinite(minimum_radius_m)
        or minimum_radius_m < 0.0
        or minimum_radius_m > maximum_m
    ):
        raise ValueError(
            "minimum local contact radius must be finite and within the maximum"
        )
    if mode == "square_uniform":
        if minimum_radius_m != 0.0:
            raise ValueError("square_uniform requires zero minimum radius")
        return jax.vmap(
            lambda key: jax.random.uniform(
                key,
                (2,),
                minval=-maximum_m,
                maxval=maximum_m,
            )
        )(keys)
    if mode != "annulus_uniform":
        raise ValueError(
            "local contact XY sampling mode must be square_uniform or "
            "annulus_uniform"
        )
    if maximum_m <= 0.0 or minimum_radius_m <= 0.0:
        raise ValueError("annulus_uniform requires positive radius bounds")
    angle = jax.vmap(
        lambda key: jax.random.uniform(
            jax.random.fold_in(key, 3401),
            (),
            minval=0.0,
            maxval=2.0 * math.pi,
        )
    )(keys)
    radius_sq = jax.vmap(
        lambda key: jax.random.uniform(
            jax.random.fold_in(key, 3402),
            (),
            minval=minimum_radius_m**2,
            maxval=maximum_m**2,
        )
    )(keys)
    radius = jnp.sqrt(radius_sq)
    return jnp.stack([radius * jnp.cos(angle), radius * jnp.sin(angle)], axis=-1)


def reflect_local_xy_jitter_into_bounds(
    anchor_xy_m: jax.Array,
    jitter_xy_m: jax.Array,
    *,
    x_bounds_m: tuple[float, float],
    y_bounds_m: tuple[float, float],
) -> jax.Array:
    """Reflect outward local targets inward without shrinking their distance."""

    anchors = jnp.asarray(anchor_xy_m, dtype=jnp.float32)
    jitter = jnp.asarray(jitter_xy_m, dtype=jnp.float32)
    if anchors.shape != jitter.shape or anchors.shape[-1] != 2:
        raise ValueError("anchor and local jitter must have matching (..., 2) shapes")
    lower = jnp.asarray(
        [min(x_bounds_m), min(y_bounds_m)], dtype=jnp.float32
    )
    upper = jnp.asarray(
        [max(x_bounds_m), max(y_bounds_m)], dtype=jnp.float32
    )
    candidate = anchors + jitter
    return jnp.where(
        candidate < lower,
        jnp.abs(jitter),
        jnp.where(candidate > upper, -jnp.abs(jitter), jitter),
    )


@dataclass(frozen=True)
class ContactTargetConfig:
    """Bounded observable-only contact-target generator settings."""

    target_apex_z_m: float = 1.40
    max_post_ball_vxy_m_s: float = 0.20
    max_lateral_displacement_per_bounce_m: float = 0.050
    max_tilt_deg: float = 3.0
    max_time_to_contact_s: float = 1.0
    gravity_m_s2: float = 9.8

    def __post_init__(self) -> None:
        if self.target_apex_z_m <= 0.0:
            raise ValueError("target_apex_z_m must be positive")
        if self.max_post_ball_vxy_m_s < 0.0:
            raise ValueError("max_post_ball_vxy_m_s must be non-negative")
        if self.max_lateral_displacement_per_bounce_m <= 0.0:
            raise ValueError(
                "max_lateral_displacement_per_bounce_m must be positive"
            )
        if not 0.0 <= self.max_tilt_deg < 45.0:
            raise ValueError("max_tilt_deg must be in [0, 45)")
        if self.max_time_to_contact_s <= 0.0:
            raise ValueError("max_time_to_contact_s must be positive")
        if self.gravity_m_s2 <= 0.0:
            raise ValueError("gravity_m_s2 must be positive")


class ContactTarget(NamedTuple):
    """Per-lane racket target plus the explicitly separate future ball state."""

    position_m: jax.Array
    rotation_matrix: jax.Array
    predicted_ball_position_at_contact_m: jax.Array
    time_to_contact_s: jax.Array
    incoming_ball_velocity_m_s: jax.Array
    next_crossing_target_xy_m: jax.Array
    desired_post_ball_velocity_m_s: jax.Array
    valid: jax.Array


class BallisticContactPredictionV1(NamedTuple):
    """Flight-only prediction, intentionally separate from impact intent."""

    valid: jax.Array
    query_time_s: jax.Array
    contact_time_s: jax.Array
    time_to_contact_s: jax.Array
    contact_position_world_m: jax.Array
    incoming_velocity_world_m_s: jax.Array
    source_state_time_s: jax.Array
    prediction_horizon_s: jax.Array


class GeometricOutcomeTargetV1(NamedTuple):
    """Geometric desired impact outcome without claiming an inverse model."""

    valid: jax.Array
    desired_post_velocity_world_m_s: jax.Array
    desired_apex_abs_m: jax.Array
    desired_next_contact_xy_world_m: jax.Array
    racket_contact_position_world_m: jax.Array
    racket_rotation_world: jax.Array
    racket_normal_world: jax.Array
    nominal_racket_linear_velocity_world_m_s: jax.Array
    max_inward_xy_step_m: jax.Array


FIXED_CONTACT_PHASE_INVALID = -1
FIXED_CONTACT_PHASE_APPROACH = 0
FIXED_CONTACT_PHASE_ARRIVAL_GATE = 1
FIXED_CONTACT_PHASE_SETTLE = 2
FIXED_CONTACT_PHASE_STRIKE = 3
FIXED_CONTACT_PHASE_RETRACT = 4
FIXED_CONTACT_PHASE_RECOVER = 5

FIXED_CONTACT_SKILL_MODE_ARRIVAL_ONLY = 0
FIXED_CONTACT_SKILL_MODE_ARRIVED_HIT = 1
FIXED_CONTACT_SKILL_MODE_RECURRENT = 2


class PostHitRecoveryReferenceErrors(NamedTuple):
    """Position/motion errors under the selected recovery reference."""

    xy_error_m: jax.Array
    z_error_m: jax.Array
    vxy_m_s: jax.Array


def fixed_contact_post_hit_recovery_reference_errors(
    *,
    racket_site_position_m: jax.Array,
    racket_surface_position_m: jax.Array,
    racket_site_velocity_m_s: jax.Array,
    racket_surface_velocity_m_s: jax.Array,
    racket_anchor_m: jax.Array,
    policy_target_position_m: jax.Array,
    use_policy_target: bool,
) -> PostHitRecoveryReferenceErrors:
    """Select a legacy hidden anchor or actor-observable policy target.

    The dynamic-target path deliberately uses the racket contact surface and
    its velocity because those are the quantities present in the actor and
    dense S0 reward contracts.  The legacy path remains byte-for-byte
    selectable for historical checkpoints.
    """

    if use_policy_target:
        position = jnp.asarray(racket_surface_position_m)
        velocity = jnp.asarray(racket_surface_velocity_m_s)
        target = jnp.asarray(policy_target_position_m)
    else:
        position = jnp.asarray(racket_site_position_m)
        velocity = jnp.asarray(racket_site_velocity_m_s)
        target = jnp.asarray(racket_anchor_m)
    return PostHitRecoveryReferenceErrors(
        xy_error_m=jnp.linalg.norm(position[:, :2] - target[:, :2], axis=-1),
        z_error_m=position[:, 2] - target[:, 2],
        vxy_m_s=jnp.linalg.norm(velocity[:, :2], axis=-1),
    )


def fixed_contact_ready_deadline_reward(
    *,
    ready: jax.Array,
    worst_normalized_violation: jax.Array,
    time_to_contact_s: jax.Array,
    crossing_valid: jax.Array,
    deadline_s: float,
    control_dt_s: float,
    success_weight: float,
    miss_weight: float,
) -> tuple[jax.Array, jax.Array]:
    """Emit one non-compensable readiness event at the TTC deadline.

    The event is restricted to one control-width bin.  A policy that becomes
    ready at 80 or 60 ms therefore cannot recover credit missed at 100 ms.
    Miss cost is proportional to the weakest normalized constraint violation.
    Zero weights preserve every historical reward path.
    """

    event = jnp.asarray(crossing_valid, dtype=bool) & (
        jnp.abs(jnp.asarray(time_to_contact_s) - float(deadline_s))
        <= 0.5 * float(control_dt_s) + 1.0e-7
    )
    event_reward = jnp.where(
        jnp.asarray(ready, dtype=bool),
        float(success_weight),
        -float(miss_weight)
        * jnp.maximum(jnp.asarray(worst_normalized_violation), 0.0),
    )
    return jnp.where(event, event_reward, 0.0), event


def fixed_contact_post_hit_recovery_success_mask(
    *,
    hit_count: jax.Array,
    in_contact: jax.Array,
    ball_above_racket_surface_m: jax.Array,
    racket_z_error_m: jax.Array,
    racket_pose_error_rad: jax.Array,
    time_since_counted_hit_s: jax.Array,
    min_elapsed_s: float,
    min_surface_gap_m: float,
    max_racket_z_error_m: float,
    max_racket_pose_error_rad: float,
    require_full_readiness: bool = False,
    racket_xy_error_m: jax.Array | None = None,
    racket_vxy_m_s: jax.Array | None = None,
    racket_omega_rad_s: jax.Array | None = None,
    max_racket_xy_error_m: float = 0.020,
    max_racket_vxy_m_s: float = 0.080,
    max_racket_omega_rad_s: float = 0.75,
    require_vertical_stability: bool = False,
    racket_vz_m_s: jax.Array | None = None,
    max_racket_vz_m_s: float = 0.080,
    min_confirmed_hits: int = 1,
    require_reference_readiness: bool = True,
) -> jax.Array:
    """Recognize a counted-hit detach, optionally with reference recovery.

    Historical V150 callers use the original Z/pose-only contract.  New
    opt-in lessons may either require the complete contact-readiness state or
    use only a later counted hit plus clean free-flight separation.  The
    latter binds success to the real next contact without imposing a static
    reference state on a dynamically successful return.
    """

    required_hits = int(min_confirmed_hits)
    if required_hits <= 0:
        raise ValueError("post-hit recovery requires at least one confirmed hit")
    recovered = (
        (jnp.asarray(hit_count) >= required_hits)
        & (~jnp.asarray(in_contact, dtype=bool))
        & (
            jnp.asarray(ball_above_racket_surface_m)
            >= float(min_surface_gap_m)
        )
        & (
            jnp.asarray(time_since_counted_hit_s)
            >= float(min_elapsed_s)
        )
    )
    if require_reference_readiness:
        recovered = (
            recovered
            & (
                jnp.abs(jnp.asarray(racket_z_error_m))
                <= float(max_racket_z_error_m)
            )
            & (
                jnp.asarray(racket_pose_error_rad)
                <= float(max_racket_pose_error_rad)
            )
        )
    elif require_full_readiness or require_vertical_stability:
        raise ValueError(
            "full or vertical recovery readiness requires reference readiness"
        )
    if require_vertical_stability:
        if racket_vz_m_s is None:
            raise ValueError(
                "vertical post-hit recovery requires a racket Vz array"
            )
        max_vz = float(max_racket_vz_m_s)
        if not math.isfinite(max_vz) or max_vz <= 0.0:
            raise ValueError(
                "vertical post-hit recovery Vz limit must be finite and "
                "positive"
            )
        recovered = recovered & (
            jnp.abs(jnp.asarray(racket_vz_m_s)) <= max_vz
        )
    if not require_full_readiness:
        return recovered
    if (
        racket_xy_error_m is None
        or racket_vxy_m_s is None
        or racket_omega_rad_s is None
    ):
        raise ValueError(
            "full post-hit recovery readiness requires XY error, VXY and "
            "angular-speed arrays"
        )
    return (
        recovered
        & (
            jnp.asarray(racket_xy_error_m)
            <= float(max_racket_xy_error_m)
        )
        & (jnp.asarray(racket_vxy_m_s) <= float(max_racket_vxy_m_s))
        & (
            jnp.asarray(racket_omega_rad_s)
            <= float(max_racket_omega_rad_s)
        )
    )


def fixed_contact_post_hit_recovery_timeout_penalty(
    *,
    truncated: jax.Array,
    lesson_enabled: bool,
    recovery_complete: jax.Array,
    weight: float,
) -> jax.Array:
    """Penalize a one-hit recovery lesson that consumes its full horizon.

    The event is failure-only: any full-horizon lesson that has not completed
    recovery pays the same cost, including a zero-hit escape. Successful
    recovery, non-truncated lanes and historical lessons receive exactly zero.
    """

    penalty_weight = float(weight)
    if not math.isfinite(penalty_weight) or penalty_weight < 0.0:
        raise ValueError(
            "post-hit recovery timeout penalty weight must be finite and "
            "nonnegative"
        )
    failure = (
        bool(lesson_enabled)
        & jnp.asarray(truncated, dtype=bool)
        & (~jnp.asarray(recovery_complete, dtype=bool))
    )
    return jnp.where(failure, -penalty_weight, 0.0)


def fixed_contact_post_hit_recovery_failure_penalty(
    *,
    terminated: jax.Array,
    truncated: jax.Array,
    lesson_enabled: bool,
    recovery_complete: jax.Array,
    weight: float,
) -> jax.Array:
    """Penalize every terminal recovery failure, including early exits.

    V160's timeout-only term cannot see workspace, ball-loss, or other early
    terminations.  This opt-in successor term closes that absorbing escape
    while preserving successful recovery and non-terminal transitions.
    """

    penalty_weight = float(weight)
    if not math.isfinite(penalty_weight) or penalty_weight < 0.0:
        raise ValueError(
            "post-hit recovery failure penalty weight must be finite and "
            "nonnegative"
        )
    failure = (
        bool(lesson_enabled)
        & (
            jnp.asarray(terminated, dtype=bool)
            | jnp.asarray(truncated, dtype=bool)
        )
        & (~jnp.asarray(recovery_complete, dtype=bool))
    )
    return jnp.where(failure, -penalty_weight, 0.0)


def fixed_contact_post_hit_recovery_joint_constraint_penalty(
    *,
    racket_z_error_m: jax.Array,
    racket_pose_error_rad: jax.Array,
    racket_xy_error_m: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    racket_vz_m_s: jax.Array,
    max_racket_z_error_m: float,
    max_racket_pose_error_rad: float,
    max_racket_xy_error_m: float,
    max_racket_vxy_m_s: float,
    max_racket_omega_rad_s: float,
    max_racket_vz_m_s: float,
    include_full_readiness: bool,
    include_vertical_stability: bool,
    active: jax.Array,
    weight: float,
    control_dt_s: float,
) -> jax.Array:
    """Return a smooth all-component post-hit recovery cost.

    Unlike a maximum-only bottleneck cost, the mean robust normalized error
    keeps a gradient on every enabled recovery component.  The exact terminal
    tolerances provide the normalization, including Vz for a dynamically
    invariant recovery state.  Zero weight is numerically inert.
    """

    penalty_weight = float(weight)
    dt_s = float(control_dt_s)
    if not math.isfinite(penalty_weight) or penalty_weight < 0.0:
        raise ValueError(
            "post-hit joint constraint penalty weight must be finite and "
            "nonnegative"
        )
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("post-hit joint constraint dt must be positive")

    values = [
        jnp.abs(jnp.asarray(racket_z_error_m, dtype=jnp.float32)),
        jnp.abs(jnp.asarray(racket_pose_error_rad, dtype=jnp.float32)),
    ]
    limits = [
        float(max_racket_z_error_m),
        float(max_racket_pose_error_rad),
    ]
    if include_full_readiness:
        values.extend(
            [
                jnp.abs(jnp.asarray(racket_xy_error_m, dtype=jnp.float32)),
                jnp.abs(jnp.asarray(racket_vxy_m_s, dtype=jnp.float32)),
                jnp.abs(jnp.asarray(racket_omega_rad_s, dtype=jnp.float32)),
            ]
        )
        limits.extend(
            [
                float(max_racket_xy_error_m),
                float(max_racket_vxy_m_s),
                float(max_racket_omega_rad_s),
            ]
        )
    if include_vertical_stability:
        values.append(jnp.abs(jnp.asarray(racket_vz_m_s, dtype=jnp.float32)))
        limits.append(float(max_racket_vz_m_s))
    if any(value <= 0.0 or not math.isfinite(value) for value in limits):
        raise ValueError(
            "post-hit joint constraint limits must be finite and positive"
        )

    normalized = jnp.stack(values, axis=-1) / jnp.asarray(
        limits, dtype=jnp.float32
    )
    robust_cost = jnp.sqrt(1.0 + normalized**2) - 1.0
    mean_cost = jnp.mean(robust_cost, axis=-1)
    return (
        -penalty_weight
        * mean_cost
        * jnp.asarray(active, dtype=jnp.float32)
        * dt_s
    )


def fixed_contact_return_basin_success_mask(
    *,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    in_contact: jax.Array,
    target_ttc_s: float,
    control_dt_s: float,
) -> jax.Array:
    """End a one-hit lesson at the next descending TTC checkpoint."""

    return (
        (jnp.asarray(hit_count) > 0)
        & jnp.asarray(crossing_valid, dtype=bool)
        & (~jnp.asarray(in_contact, dtype=bool))
        & (
            jnp.abs(jnp.asarray(time_to_contact_s) - float(target_ttc_s))
            <= 0.5 * float(control_dt_s) + 1.0e-7
        )
    )


@dataclass(frozen=True)
class HighBallPredictorConfig:
    """Runtime-matched bounded high-FOV predictor settings in SI units."""

    history_frames: int = 9
    activation_z_m: float = 1.40
    min_upward_velocity_m_s: float = 0.20
    max_prediction_s: float = 0.50
    stop_prediction_z_m: float = 1.24
    drag_coefficient_per_m: float = 0.13545
    gravity_m_s2: float = 9.8
    integration_step_s: float = 0.005
    max_history_gap_s: float = 0.050
    min_history_span_s: float = 0.020
    activation_max_gap_s: float = 0.080
    fit_max_rmse_m: float = 0.030
    max_fitted_speed_m_s: float = 15.0
    reacquisition_gate_m: float = 0.12
    enable_optimized_initial_state: bool = True

    def __post_init__(self) -> None:
        if int(self.history_frames) < 3:
            raise ValueError("history_frames must be at least 3")
        if self.max_prediction_s <= 0.0 or self.integration_step_s <= 0.0:
            raise ValueError("prediction and integration times must be positive")
        if self.drag_coefficient_per_m < 0.0 or self.gravity_m_s2 <= 0.0:
            raise ValueError("drag must be non-negative and gravity positive")


class HighBallPredictorState(NamedTuple):
    """Per-lane causal state; history contains accepted fresh frames only."""

    position_history_m: jax.Array
    time_history_s: jax.Array
    history_count: jax.Array
    active: jax.Array
    initial_time_s: jax.Array
    initial_position_m: jax.Array
    initial_velocity_m_s: jax.Array


class HighBallPredictorOutput(NamedTuple):
    position_m: jax.Array
    velocity_m_s: jax.Array
    valid: jax.Array
    observed: jax.Array
    predicted: jax.Array
    prediction_horizon_s: jax.Array
    represented_state_age_s: jax.Array
    reacquisition_rejected: jax.Array
    fit_rmse_m: jax.Array


class ContactQualityScores(NamedTuple):
    components: jax.Array
    mean: jax.Array
    joint: jax.Array
    combined: jax.Array


class ReturnQualityScores(NamedTuple):
    trajectory: jax.Array
    apex: jax.Array
    joint: jax.Array
    combined: jax.Array
    trajectory_error_m: jax.Array
    horizontal_displacement_m: jax.Array
    center_progress_m: jax.Array


def racket_surface_frame(
    geom_rotation_world: jax.Array, *, normal_axis: int,
) -> jax.Array:
    """Return an upward, right-handed surface frame, shape (..., 3, 3).

    Compiled mesh principal axes need not match cylinder axes or site axes.
    The declared thickness axis supplies the normal; jitter uses its tangent
    plane. Columns are tangent X, tangent Y and upward normal.
    """

    if normal_axis not in (0, 1, 2):
        raise ValueError("normal_axis must be 0, 1, or 2")
    rotation = jnp.asarray(geom_rotation_world)
    normal = rotation[..., :, normal_axis]
    normal = normal * jnp.where(normal[..., 2:3] >= 0., 1., -1.)
    tangent = rotation[..., :, (normal_axis + 1) % 3]
    return jnp.stack((tangent, jnp.cross(normal, tangent), normal), axis=-1)


def fixed_contact_apex_error_cost(
    apex_z_m: jax.Array,
    *,
    target_apex_z_m: float | jax.Array,
    scale_m: float,
) -> jax.Array:
    """Return a symmetric pseudo-Huber apex cost with a linear tail.

    The Gaussian apex score used by the return-quality reward is useful near
    the requested trajectory, but its gradient is effectively zero for a
    gross launch error.  A placed-ball policy can therefore remain at a very
    high launch apex while its next plane crossing is outside the predictor's
    validity horizon.  This cost remains finite and supplies a bounded
    derivative at any error magnitude.  Callers own the event weight; a zero
    weight preserves every historical fixed-contact reward exactly.
    """

    scale = float(scale_m)
    target = jnp.asarray(target_apex_z_m, dtype=jnp.float32)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_m must be positive and finite")
    normalized_error = (
        jnp.asarray(apex_z_m, dtype=jnp.float32) - target
    ) / scale
    return jnp.sqrt(1.0 + normalized_error**2) - 1.0


def propagate_ballistic_state_to_age(
    position_m: jax.Array,
    velocity_m_s: jax.Array,
    age_s: jax.Array,
    *,
    gravity_m_s2: float | jax.Array = 9.8,
) -> tuple[jax.Array, jax.Array]:
    """Advance a held camera sample to the current control tick.

    The fixed-contact actor still receives the last real sample in its
    explicit current-ball slots.  This helper is used only to put the dynamic
    ContactTarget on the same 200 Hz time base as the policy action.  It adds
    no observation channel and uses the same no-drag ballistic model as the
    contact-target solver.
    """

    gravity = float(gravity_m_s2)
    if not math.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity_m_s2 must be positive and finite")
    position = jnp.asarray(position_m, dtype=jnp.float32)
    velocity = jnp.asarray(velocity_m_s, dtype=jnp.float32)
    age = jnp.maximum(jnp.asarray(age_s, dtype=jnp.float32), 0.0)
    while age.ndim < position.ndim:
        age = age[..., None]
    gravity_vector = jnp.asarray([0.0, 0.0, -gravity], dtype=jnp.float32)
    return (
        position + velocity * age + 0.5 * gravity_vector * age**2,
        velocity + gravity_vector * age,
    )


def fixed_contact_outcome_constraint_costs(
    *,
    surface_z_error_m: jax.Array,
    pose_error_rad: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    contact_offset_m: jax.Array,
    horizontal_displacement_m: jax.Array,
    center_progress_m: jax.Array,
    max_surface_z_error_m: float,
    max_pose_error_rad: float,
    max_racket_vxy_m_s: float,
    max_racket_omega_rad_s: float,
    max_contact_offset_m: float,
    max_horizontal_displacement_m: float,
    max_outward_progress_m: float,
) -> jax.Array:
    """Return independent hinge costs for all contact/flight constraints.

    A weighted quality average allowed a good height component to compensate
    for excessive horizontal or angular racket speed.  These seven costs are
    zero only inside every declared physical limit, so the event reward and
    the strict gates describe the same behavior.  The final component charges
    motion away from the workspace center while still allowing a two-
    millimetre numerical tolerance by default.
    """

    limits = (
        float(max_surface_z_error_m),
        float(max_pose_error_rad),
        float(max_racket_vxy_m_s),
        float(max_racket_omega_rad_s),
        float(max_contact_offset_m),
        float(max_horizontal_displacement_m),
        float(max_outward_progress_m),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("fixed-contact outcome limits must be positive and finite")
    values = jnp.stack(
        [
            jnp.abs(jnp.asarray(surface_z_error_m, dtype=jnp.float32)),
            jnp.abs(jnp.asarray(pose_error_rad, dtype=jnp.float32)),
            jnp.abs(jnp.asarray(racket_vxy_m_s, dtype=jnp.float32)),
            jnp.abs(jnp.asarray(racket_omega_rad_s, dtype=jnp.float32)),
            jnp.abs(jnp.asarray(contact_offset_m, dtype=jnp.float32)),
            jnp.abs(
                jnp.asarray(horizontal_displacement_m, dtype=jnp.float32)
            ),
            jnp.maximum(
                -jnp.asarray(center_progress_m, dtype=jnp.float32), 0.0
            ),
        ],
        axis=-1,
    )
    normalized_excess = jnp.maximum(
        values / jnp.asarray(limits, dtype=jnp.float32) - 1.0,
        0.0,
    )
    return jnp.sqrt(1.0 + normalized_excess**2) - 1.0


def aggregate_fixed_contact_outcome_constraint_costs(
    costs: jax.Array,
    aggregation: str = "sum",
) -> jax.Array:
    """Aggregate independent constraint costs without changing legacy runs.

    ``sum`` is the historical V22--V24 behavior.  ``mean`` keeps the event
    penalty on a per-constraint scale, so adding independent safety
    constraints cannot silently multiply the reward magnitude by their
    count.  ``max`` targets the worst normalized violation and therefore
    matches an all-components-ready graduation contract without diluting one
    decisive failure by the component count.  Profiles must opt in
    explicitly.
    """

    values = jnp.asarray(costs, dtype=jnp.float32)
    if values.ndim < 1 or values.shape[-1] <= 0:
        raise ValueError("fixed-contact outcome costs require a component axis")
    mode = str(aggregation).strip().lower()
    if mode == "sum":
        return jnp.sum(values, axis=-1)
    if mode == "mean":
        return jnp.mean(values, axis=-1)
    if mode == "max":
        return jnp.max(values, axis=-1)
    raise ValueError(
        "fixed-contact outcome constraint aggregation must be 'sum', 'mean', "
        "or 'max'"
    )


def fixed_contact_outcome_constraint_event_penalty(
    *,
    event: jax.Array,
    hit_count: jax.Array,
    cost: jax.Array,
    min_hit_count: int,
    weight: float,
) -> jax.Array:
    """Apply contact-quality cost from a selected confirmed-hit ordinal."""

    if min_hit_count < 0:
        raise ValueError("min_hit_count must be nonnegative")
    return jnp.where(
        jnp.asarray(event, dtype=bool)
        & (jnp.asarray(hit_count) >= int(min_hit_count)),
        -float(weight) * jnp.asarray(cost),
        0.0,
    )


def early_racket_xy_motion_penalty(
    *,
    hit_count: jax.Array,
    racket_vxy_m_s: jax.Array,
    control_dt_s: float,
    weight: float,
) -> jax.Array:
    """Penalize launch and first-three-hit lateral racket motion smoothly.

    A 0.03 m/s speed deadband avoids punishing small tracking noise.
    The normalized pseudo-Huber cost retains a useful gradient for the
    large initial sweeps seen in V127 replay. The episode target anchor
    can differ from the reset pose, so it must not pull the racket inward.
    """

    speed_violation = jnp.maximum(jnp.asarray(racket_vxy_m_s) - 0.03, 0.0) / 0.07
    cost = jnp.sqrt(1.0 + speed_violation ** 2) - 1.0
    return jnp.where(
        jnp.asarray(hit_count) < 3,
        -float(weight) * float(control_dt_s) * cost,
        0.0,
    )


def motion_penalty_cost(
    scaled_excess: jax.Array, *, linear_tail: bool,
) -> jax.Array:
    """Keep the legacy cap or retain a linear gradient past excess two."""
    excess = jnp.asarray(scaled_excess)
    return jnp.where(
        bool(linear_tail) & (excess > 2.0),
        4.0 + 4.0 * (excess - 2.0),
        jnp.minimum(excess ** 2, 4.0),
    )


def racket_flatness_penalty_cost(
    up_cos: jax.Array, *, use_angle: bool, cos_target: float, cos_sigma: float,
    angle_deadband_rad: float, angle_scale_rad: float,
) -> jax.Array:
    """Preserve the cosine objective or price small tilts in angle units."""
    cosine = jnp.asarray(up_cos)
    if not use_angle:
        return (jnp.maximum(float(cos_target) - cosine, 0.0)
                / max(1e-6, float(cos_sigma))) ** 2
    # Keep acos gradients finite at a perfectly horizontal normal. The
    # minimum represented angle is below the configured 0.25-degree band.
    tilt_rad = jnp.arccos(jnp.clip(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
    excess = jnp.maximum(tilt_rad - float(angle_deadband_rad), 0.0) / float(angle_scale_rad)
    return motion_penalty_cost(excess, linear_tail=True)


def early_ball_xy_motion_penalty(
    *,
    hit_count: jax.Array,
    ball_velocity_m_s: jax.Array,
    control_dt_s: float,
    weight: float,
    deadband_m_s: float = 0.03,
    scale_m_s: float = 0.05,
) -> jax.Array:
    """Price lateral ball drift before hit three while allowing vertical lift."""

    velocity = jnp.asarray(ball_velocity_m_s)
    speed = jnp.linalg.norm(velocity[..., :2], axis=-1)
    deadband, scale = float(deadband_m_s), float(scale_m_s)
    if not math.isfinite(deadband) or deadband < 0.0:
        raise ValueError("early ball XY deadband must be finite and nonnegative")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("early ball XY scale must be finite and positive")
    violation = jnp.maximum(speed - deadband, 0.0) / scale
    cost = jnp.sqrt(1.0 + violation ** 2) - 1.0
    return jnp.where(
        jnp.asarray(hit_count) < 3,
        -float(weight) * float(control_dt_s) * cost,
        0.0,
    )


def ball_racket_anchor_xy_penalty(
    ball_position_m: jax.Array, racket_anchor_m: jax.Array,
    *, control_dt_s: float, weight: float,
) -> jax.Array:
    """Price ball XY displacement from the episode anchor; ignore height."""
    value = float(weight)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("ball anchor reward weight must be finite and nonnegative")
    distance = jnp.linalg.norm(
        jnp.asarray(ball_position_m)[..., :2] - jnp.asarray(racket_anchor_m)[..., :2],
        axis=-1,
    )
    violation = jnp.maximum(distance - .005, 0.) / .020
    return -value * float(control_dt_s) * (jnp.sqrt(1. + violation ** 2) - 1.)


def early_racket_xy_anchor_weight(
    hit_count: jax.Array, *, weight: float, transition_weight: float | None,
) -> jax.Array:
    """Keep the legacy weight or use a separate count-two transition cost."""
    counts = jnp.asarray(hit_count)
    weights = jnp.full(counts.shape, float(weight), dtype=jnp.float32)
    if transition_weight is None:
        return weights
    value = float(transition_weight)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("XY transition reward weight must be finite and nonnegative")
    return jnp.where(counts == 2, value, weights)


def late_racket_xy_anchor_penalty(
    *,
    hit_count: jax.Array,
    racket_xy_error_m: jax.Array,
    control_dt_s: float,
    weight: float,
) -> jax.Array:
    """Keep the racket near the episode XY anchor after the third contact.

    The 5 mm deadband permits small corrections. A 20 mm scale and smooth
    linear tail keep a restoring reward gradient after a large drift.
    This reward has no vertical target and changes no control commands.
    """

    violation = jnp.maximum(jnp.asarray(racket_xy_error_m) - 0.005, 0.0) / 0.020
    cost = jnp.sqrt(1.0 + violation ** 2) - 1.0
    return jnp.where(
        jnp.asarray(hit_count) >= 3,
        -float(weight) * float(control_dt_s) * cost,
        0.0,
    )


def early_action_overflow_multiplier(
    hit_count: jax.Array,
    *,
    multiplier: float,
) -> jax.Array:
    """Concentrate latent action restraint on launch without changing control."""

    return jnp.where(jnp.asarray(hit_count) < 3, float(multiplier), 1.0)


def action_overflow_loss(
    excess: jax.Array,
    hit_count: jax.Array,
    *,
    quadratic_early: bool,
) -> jax.Array:
    """Use a convex launch loss while retaining the legacy late overflow cost."""

    squared = jnp.square(jnp.asarray(excess))
    legacy = jnp.sum(jnp.log1p(squared), axis=-1)
    if not quadratic_early:
        return legacy
    return jnp.where(
        jnp.asarray(hit_count) < 3, jnp.sum(squared, axis=-1), legacy
    )


def fixed_contact_event_joint_quality_gate(
    *,
    contact_joint: jax.Array,
    return_joint: jax.Array,
    mix: float,
) -> jax.Array:
    """Bound positive event credit by the weakest contact/return quality.

    ``mix=0`` is the legacy identity.  Opt-in successors can progressively
    remove quality-independent event credit without making an initially poor
    contact worth zero: the multiplier remains in ``[1-mix, 1]`` and improves
    only when the weaker of contact quality and return quality improves.
    """

    value = float(mix)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("fixed-contact event joint-quality mix must be in [0, 1]")
    contact = jnp.clip(
        jnp.asarray(contact_joint, dtype=jnp.float32), 0.0, 1.0
    )
    returned = jnp.clip(
        jnp.asarray(return_joint, dtype=jnp.float32), 0.0, 1.0
    )
    weakest = jnp.minimum(contact, returned)
    return (1.0 - value) + value * weakest


def contact_center_penalty_event_mask(
    *, legacy_mask: jax.Array, physical_outcome: jax.Array,
    recoverability_active: jax.Array, all_physical: bool,
) -> jax.Array:
    """Price weak physical rebounds without changing legacy hit accounting."""
    if not all_physical:
        return jnp.asarray(legacy_mask, dtype=bool)
    return (jnp.asarray(physical_outcome, dtype=bool)
            & jnp.asarray(recoverability_active, dtype=bool))


def bound_fixed_contact_hit_event_reward(
    raw_event_reward: jax.Array,
    *,
    positive_hit_reference: jax.Array,
    penalty_budget_fraction: float,
    unbudgeted_event_penalty: jax.Array | None = None,
) -> jax.Array:
    """Bound negative event penalties without hiding quality gradients.

    A confirmed hit is a prerequisite for learning its flight quality.  If
    the sum of event-only constraint penalties is allowed to exceed all
    positive hit credit, PPO can improve return by turning every physical
    contact into a failed launch.  A positive opt-in fraction gives negative
    event shaping a strict budget relative to the already computed positive
    hit reference.  Positive bonuses remain exact; negative deviations use a
    smooth ``tanh`` cap and retain a finite gradient before saturation.

    Zero is the legacy identity. Values at or above one are rejected because
    the budgeted portion would lose its positive confirmed-hit credit.
    An optional unbudgeted penalty is removed before bounding and added back
    exactly once; the final reward can be negative when that cost is large.
    """

    fraction = float(penalty_budget_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError(
            "fixed-contact hit-event penalty budget fraction must be finite "
            "and in [0, 1)"
        )
    raw = jnp.asarray(raw_event_reward, dtype=jnp.float32)
    if fraction == 0.0:
        return raw
    outside_penalty = None
    if unbudgeted_event_penalty is not None:
        outside_penalty = jnp.asarray(unbudgeted_event_penalty, dtype=jnp.float32)
        raw = raw - outside_penalty
    reference = jnp.maximum(
        jnp.asarray(positive_hit_reference, dtype=jnp.float32), 0.0
    )
    budget = fraction * reference
    delta = raw - reference
    bounded_negative_delta = -budget * jnp.tanh(
        (-delta) / jnp.maximum(budget, jnp.finfo(jnp.float32).eps)
    )
    bounded = reference + jnp.where(delta < 0.0, bounded_negative_delta, delta)
    return bounded if outside_penalty is None else bounded + outside_penalty


def blend_falling_reset_arm_q(
    warm_arm_q: jax.Array,
    target_arm_q: jax.Array,
    target_fraction: float,
) -> jax.Array:
    """Blend reset joints from the warm pose toward target-following IK.

    ``target_fraction=1`` is the historical target-following reset and
    ``target_fraction=0`` is the historical fixed warm reset.  Intermediate
    values provide a controlled acquisition-distance curriculum without an
    online teacher or any change to policy observations/actions.
    """

    fraction = float(target_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("falling reset target-follow blend must be in [0, 1]")
    warm = jnp.asarray(warm_arm_q, dtype=jnp.float32)
    target = jnp.asarray(target_arm_q, dtype=jnp.float32)
    return warm + fraction * (target - warm)


def cartesian_reset_blend_offsets(
    target_x_offset_m: jax.Array,
    target_y_offset_m: jax.Array,
    target_fraction: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Scale target offsets in Cartesian space before the IK-table lookup.

    Joint-space interpolation does not preserve racket height, orientation, or
    a Cartesian fraction of the requested displacement.  This opt-in helper
    keeps each requested blend on the fixed contact plane; the environment
    subsequently selects the nearest independently solved pose-preserving IK
    table entry.  Historical profiles continue to use joint interpolation.
    """

    fraction = jnp.asarray(target_fraction, dtype=jnp.float32)
    return (
        jnp.asarray(target_x_offset_m, dtype=jnp.float32) * fraction,
        jnp.asarray(target_y_offset_m, dtype=jnp.float32) * fraction,
    )


def cartesian_reset_blend_ik_axis_range(
    target_range_m: tuple[float, float],
    *,
    fixed_fraction: float,
    fraction_range: tuple[float, float] | None,
    fraction_choices: tuple[float, ...] | None,
) -> tuple[float, float]:
    """Expand one IK-table axis to contain every Cartesian blend request.

    A singleton target course can still request a different reset location:
    for example, target ``-0.20`` at blend ``0.8`` requires an IK entry near
    ``-0.16``.  Keeping only the target interval silently snaps that request
    back to ``-0.20`` during nearest-neighbour lookup.  The original target
    interval is always retained so enabling this helper can expand but never
    shrink the configured reset workspace.
    """

    target_values = tuple(float(value) for value in target_range_m)
    fractions = [float(fixed_fraction)]
    if fraction_range is not None:
        fractions.extend(float(value) for value in fraction_range)
    if fraction_choices is not None:
        fractions.extend(float(value) for value in fraction_choices)
    if (
        len(target_values) != 2
        or not all(math.isfinite(value) for value in target_values)
        or not all(math.isfinite(value) for value in fractions)
        or not all(0.0 <= value <= 1.0 for value in fractions)
    ):
        raise ValueError(
            "Cartesian reset IK range requires finite target bounds and "
            "blend fractions in [0, 1]"
        )
    candidates = list(target_values)
    candidates.extend(
        target * fraction
        for target in target_values
        for fraction in fractions
    )
    return min(candidates), max(candidates)


def cartesian_reset_blend_ik_axis_knots(
    target_range_m: tuple[float, float],
    target_knots_m: tuple[float, ...],
    *,
    fixed_fraction: float,
    fraction_range: tuple[float, float] | None,
    fraction_choices: tuple[float, ...] | None,
) -> tuple[float, ...]:
    """Return exact target and target-by-blend knots inside one axis."""

    # Reuse the range helper's finite/[0, 1] validation.
    cartesian_reset_blend_ik_axis_range(
        target_range_m,
        fixed_fraction=fixed_fraction,
        fraction_range=fraction_range,
        fraction_choices=fraction_choices,
    )
    lo_m, hi_m = sorted(float(value) for value in target_range_m)
    targets = tuple(
        float(value)
        for value in target_knots_m
        if lo_m - 1.0e-9 <= float(value) <= hi_m + 1.0e-9
    )
    if not targets:
        targets = (lo_m, hi_m)
    fractions = [float(fixed_fraction)]
    if fraction_range is not None:
        fractions.extend(float(value) for value in fraction_range)
    if fraction_choices is not None:
        fractions.extend(float(value) for value in fraction_choices)
    candidates = set(targets)
    candidates.update(
        target * fraction for target in targets for fraction in fractions
    )
    return tuple(sorted(candidates))


def sample_reset_target_follow_blend(
    keys: jax.Array,
    *,
    fixed_fraction: float,
    fraction_range: tuple[float, float] | None,
    fraction_choices: tuple[float, ...] | None = None,
    fraction_choice_probabilities: tuple[float, ...] | None = None,
) -> jax.Array:
    """Return one target-follow blend per reset lane.

    A missing range preserves the historical scalar reset exactly.  The
    opt-in range lets one rollout rehearse the acquisition-distance continuum.
    Discrete choices additionally give the warm/full-acquisition and
    target/hold endpoints nonzero probability instead of relying on a
    continuous distribution that samples either endpoint with probability
    zero.
    """

    fixed = float(fixed_fraction)
    if not math.isfinite(fixed) or not 0.0 <= fixed <= 1.0:
        raise ValueError("fixed target-follow blend must be finite and in [0, 1]")
    reset_keys = jnp.asarray(keys)
    if fraction_range is not None and fraction_choices is not None:
        raise ValueError(
            "target-follow blend range and choices are mutually exclusive"
        )
    if fraction_choices is not None:
        values = tuple(float(value) for value in fraction_choices)
        if (
            not values
            or not all(math.isfinite(value) for value in values)
            or not all(0.0 <= value <= 1.0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                "target-follow blend choices must be unique finite values "
                "in [0, 1]"
            )
        choice_array = jnp.asarray(values, dtype=jnp.float32)
        if fraction_choice_probabilities is None:
            # Preserve the exact historical uniform sampler.
            indices = jax.vmap(
                lambda key: jax.random.randint(
                    key,
                    (),
                    minval=0,
                    maxval=len(values),
                    dtype=jnp.int32,
                )
            )(reset_keys)
        else:
            probabilities = tuple(
                float(value) for value in fraction_choice_probabilities
            )
            if (
                len(probabilities) != len(values)
                or not all(math.isfinite(value) for value in probabilities)
                or not all(value > 0.0 for value in probabilities)
                or not math.isclose(
                    sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-9
                )
            ):
                raise ValueError(
                    "target-follow blend choice probabilities must be "
                    "positive finite values matching choices and sum to one"
                )
            logits = jnp.log(jnp.asarray(probabilities, dtype=jnp.float32))
            indices = jax.vmap(
                lambda key: jax.random.categorical(key, logits)
            )(reset_keys).astype(jnp.int32)
        return choice_array[indices]
    if fraction_choice_probabilities is not None:
        raise ValueError(
            "target-follow blend choice probabilities require choices"
        )
    if fraction_range is None:
        return jnp.full(reset_keys.shape[:-1], fixed, dtype=jnp.float32)
    values = tuple(float(value) for value in fraction_range)
    if (
        len(values) != 2
        or not all(math.isfinite(value) for value in values)
        or not 0.0 <= values[0] <= values[1] <= 1.0
    ):
        raise ValueError(
            "target-follow blend range must contain two finite ordered "
            "values in [0, 1]"
        )
    lo, hi = values
    if lo == hi:
        return jnp.full(reset_keys.shape[:-1], lo, dtype=jnp.float32)
    return jax.vmap(
        lambda key: jax.random.uniform(
            key,
            (),
            minval=lo,
            maxval=hi,
            dtype=jnp.float32,
        )
    )(reset_keys)


def apply_sampled_reset_target_follow_blend(
    warm_arm_q: jax.Array,
    target_arm_q: jax.Array,
    *,
    sampled_fraction: jax.Array,
    fixed_fraction: float,
    fraction_range: tuple[float, float] | None,
    fraction_choices: tuple[float, ...] | None,
) -> jax.Array:
    """Apply the configured scalar or per-lane target-follow fraction.

    Legacy scalar profiles retain the original scalar arithmetic. Both
    opt-in sampling modes must instead apply the already sampled per-lane
    fraction; treating discrete choices as scalar would make the metric claim
    a randomized reset while the robot always starts at ``fixed_fraction``.
    """

    if fraction_range is not None and fraction_choices is not None:
        raise ValueError(
            "target-follow blend range and choices are mutually exclusive"
        )
    if fraction_range is None and fraction_choices is None:
        return blend_falling_reset_arm_q(
            warm_arm_q,
            target_arm_q,
            fixed_fraction,
        )
    warm = jnp.asarray(warm_arm_q, dtype=jnp.float32)
    target = jnp.asarray(target_arm_q, dtype=jnp.float32)
    fraction = jnp.asarray(sampled_fraction, dtype=jnp.float32)
    if fraction.shape != warm.shape[:-1]:
        raise ValueError(
            "sampled target-follow blend shape must match reset lanes"
        )
    return warm + fraction[..., None] * (target - warm)


def fixed_contact_s0_component_scores(
    *,
    tracking_xy_error_m: jax.Array,
    surface_z_error_m: jax.Array,
    pose_error_rad: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    xy_sigma_m: float,
    z_sigma_m: float,
    pose_sigma_rad: float,
    vxy_sigma_m_s: float,
    omega_sigma_rad_s: float,
) -> jax.Array:
    """Return independent bounded S0 tracking and braking scores.

    The component order is XY position, surface Z, target-relative pose,
    horizontal speed, and full angular speed.  Keeping these scores separate
    prevents one initially poor component from annihilating every other
    policy-gradient signal, as happened with V1's multiplicative brake score.
    """

    sigmas = (
        float(xy_sigma_m),
        float(z_sigma_m),
        float(pose_sigma_rad),
        float(vxy_sigma_m_s),
        float(omega_sigma_rad_s),
    )
    if any(value <= 0.0 for value in sigmas):
        raise ValueError("fixed-contact S0 score sigmas must be positive")
    errors = jnp.stack(
        [
            jnp.asarray(tracking_xy_error_m, dtype=jnp.float32),
            jnp.asarray(surface_z_error_m, dtype=jnp.float32),
            jnp.asarray(pose_error_rad, dtype=jnp.float32),
            jnp.asarray(racket_vxy_m_s, dtype=jnp.float32),
            jnp.asarray(racket_omega_rad_s, dtype=jnp.float32),
        ],
        axis=-1,
    )
    scales = jnp.asarray(sigmas, dtype=jnp.float32)
    return jnp.exp(-0.5 * (errors / scales) ** 2)


def fixed_contact_s0_gate_aligned_costs(
    *,
    tracking_xy_error_m: jax.Array,
    surface_z_error_m: jax.Array,
    pose_error_rad: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    max_xy_error_m: float,
    max_z_error_m: float,
    max_pose_error_rad: float,
    max_vxy_m_s: float,
    max_omega_rad_s: float,
) -> jax.Array:
    """Return non-saturating costs normalized by the physical S0 gates.

    A positive Gaussian becomes almost flat when a policy is several gate
    radii away from readiness.  ``sqrt(1 + x**2) - 1`` is quadratic near the
    goal and linear in the tail, so each failed physical component continues
    to provide a finite learning signal without an unbounded squared loss.
    The component order matches :func:`fixed_contact_s0_component_scores`.
    """

    limits = (
        float(max_xy_error_m),
        float(max_z_error_m),
        float(max_pose_error_rad),
        float(max_vxy_m_s),
        float(max_omega_rad_s),
    )
    if any(value <= 0.0 for value in limits):
        raise ValueError("fixed-contact gate-aligned limits must be positive")
    values = jnp.stack(
        [
            jnp.asarray(tracking_xy_error_m, dtype=jnp.float32),
            jnp.asarray(surface_z_error_m, dtype=jnp.float32),
            jnp.asarray(pose_error_rad, dtype=jnp.float32),
            jnp.asarray(racket_vxy_m_s, dtype=jnp.float32),
            jnp.asarray(racket_omega_rad_s, dtype=jnp.float32),
        ],
        axis=-1,
    )
    normalized = values / jnp.asarray(limits, dtype=jnp.float32)
    return jnp.sqrt(1.0 + normalized**2) - 1.0


def fixed_contact_s0_worst_normalized_violation(
    *,
    tracking_xy_error_m: jax.Array,
    surface_z_error_m: jax.Array,
    pose_error_rad: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    max_xy_error_m: float,
    max_z_error_m: float,
    max_pose_error_rad: float,
    max_vxy_m_s: float,
    max_omega_rad_s: float,
) -> jax.Array:
    """Return the largest positive violation of the conjunctive S0 gate.

    Each component is divided by the exact physical threshold used by
    :func:`fixed_contact_s0_readiness_mask`.  A value at or below its gate
    contributes zero; a value 50 percent above its gate contributes 0.5.
    Taking the maximum makes the scalar objective match the all-components
    graduation contract instead of allowing good components to average away
    the current bottleneck.
    """

    limits = (
        float(max_xy_error_m),
        float(max_z_error_m),
        float(max_pose_error_rad),
        float(max_vxy_m_s),
        float(max_omega_rad_s),
    )
    if any(value <= 0.0 or not math.isfinite(value) for value in limits):
        raise ValueError("fixed-contact readiness limits must be finite and positive")
    values = jnp.stack(
        [
            jnp.asarray(tracking_xy_error_m, dtype=jnp.float32),
            jnp.asarray(surface_z_error_m, dtype=jnp.float32),
            jnp.asarray(pose_error_rad, dtype=jnp.float32),
            jnp.asarray(racket_vxy_m_s, dtype=jnp.float32),
            jnp.asarray(racket_omega_rad_s, dtype=jnp.float32),
        ],
        axis=-1,
    )
    normalized_excess = (
        values / jnp.asarray(limits, dtype=jnp.float32) - 1.0
    )
    return jnp.max(jnp.maximum(normalized_excess, 0.0), axis=-1)


def fixed_contact_gate_qualified_action_penalty(
    *,
    raw_action_penalty: jax.Array,
    component_scores: jax.Array,
    minimum_component_scores: tuple[float, float, float, float, float],
) -> tuple[jax.Array, jax.Array]:
    """Charge settling effort only after every declared pose gate is met.

    A velocity-action cost paid while the racket is still outside its target
    creates a dead band: stopping short can be cheaper than making the final
    correction.  The opt-in gate makes the objective lexicographic.  Position,
    pose, and motion are corrected first; action magnitude is minimized once
    the complete contact-readiness vector is inside the same component bands
    used for curriculum graduation.  Five zero thresholds reproduce the
    historical always-active action cost exactly.
    """

    thresholds = tuple(float(value) for value in minimum_component_scores)
    if len(thresholds) != 5 or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError(
            "fixed-contact action-penalty gate requires five finite scores "
            "in [0, 1]"
        )
    scores = jnp.asarray(component_scores, dtype=jnp.float32)
    if scores.shape[-1] != 5:
        raise ValueError("fixed-contact component scores must end in size 5")
    eligible = jnp.all(
        scores >= jnp.asarray(thresholds, dtype=jnp.float32), axis=-1
    )
    raw = jnp.asarray(raw_action_penalty, dtype=jnp.float32)
    return jnp.where(eligible, raw, jnp.zeros_like(raw)), eligible


def fixed_contact_gate_qualified_low_action_reward(
    *,
    action: jax.Array,
    component_scores: jax.Array,
    minimum_component_scores: tuple[float, float, float, float, float],
    reward_weight: float,
    active_dt: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reward low action inside every pose gate without a boundary penalty.

    The V57 conditional cost makes entering the accepted region add a negative
    reward discontinuity.  This bounded score reverses that discontinuity:
    qualification can never reduce reward, while ``1 - mean(action**2)`` still
    has its unique maximum at the zero-QVEL solution.  The environment action
    is clipped to [-1, 1], and clipping again here makes the [0, 1] bound an
    explicit contract for direct unit use.
    """

    weight = float(reward_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("qualified low-action reward weight must be finite and nonnegative")
    thresholds = tuple(float(value) for value in minimum_component_scores)
    if len(thresholds) != 5 or any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError(
            "qualified low-action reward requires five finite scores in [0, 1]"
        )
    scores = jnp.asarray(component_scores, dtype=jnp.float32)
    if scores.shape[-1] != 5:
        raise ValueError("fixed-contact component scores must end in size 5")
    bounded_action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), -1.0, 1.0)
    low_action_score = jnp.clip(
        1.0 - jnp.mean(bounded_action**2, axis=-1), 0.0, 1.0
    )
    eligible = jnp.all(
        scores >= jnp.asarray(thresholds, dtype=jnp.float32), axis=-1
    )
    reward = (
        weight
        * low_action_score
        * jnp.asarray(active_dt, dtype=jnp.float32)
    )
    return jnp.where(eligible, reward, jnp.zeros_like(reward)), eligible, low_action_score


def fixed_contact_s0_progress_active_mask(
    *,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    final_window_s: float,
) -> jax.Array:
    """Use directional XY progress only before the final braking window."""

    if float(final_window_s) <= 0.0:
        raise ValueError("fixed-contact final window must be positive")
    return jnp.asarray(crossing_valid, dtype=jnp.bool_) & (
        jnp.asarray(time_to_contact_s, dtype=jnp.float32)
        > float(final_window_s)
    )


def fixed_contact_s0_reward_active_mask(
    *,
    crossing_valid: jax.Array,
    hit_count: jax.Array,
    episode_time_s: jax.Array,
    pre_hit_budget_s: float,
    falling_reset_active: jax.Array | None = None,
    falling_pre_hit_budget_s: float = 0.0,
) -> jax.Array:
    """Bound preparation credit before the first physical hit.

    Placed-ball courses begin with a valid contact target even when the ball
    remains resting on the racket.  An infinite budget preserves the legacy
    reward exactly.  A finite budget prevents an indefinitely sticky, zero-hit
    trajectory from collecting the per-step S0 preparation reward; after the
    first counted hit, the same reward remains available for every return.
    """

    budget_s = float(pre_hit_budget_s)
    falling_budget_s = float(falling_pre_hit_budget_s)
    if budget_s < 0.0 or math.isnan(budget_s):
        raise ValueError("fixed-contact pre-hit S0 budget must be nonnegative")
    if falling_budget_s < 0.0 or math.isnan(falling_budget_s):
        raise ValueError(
            "fixed-contact falling pre-hit S0 budget must be nonnegative"
        )
    episode_time = jnp.asarray(episode_time_s, dtype=jnp.float32)
    base_pre_hit = (budget_s > 0.0) & (episode_time <= budget_s)
    falling_pre_hit = jnp.zeros_like(episode_time, dtype=jnp.bool_)
    if falling_reset_active is not None:
        falling_pre_hit = (
            jnp.asarray(falling_reset_active, dtype=jnp.bool_)
            & (falling_budget_s > 0.0)
            & (episode_time <= falling_budget_s)
        )
    return jnp.asarray(crossing_valid, dtype=jnp.bool_) & (
        (jnp.asarray(hit_count) > 0)
        | base_pre_hit
        | falling_pre_hit
    )


def fixed_contact_post_hit_motion_active_dt(
    *,
    reward_active: jax.Array,
    final_window_active: jax.Array,
    hit_count: jax.Array,
    dense_post_hit_recovery_enabled: bool,
    control_dt_s: float,
) -> jax.Array:
    """Return the shared reward/telemetry mask for post-hit racket braking.

    Historical curricula shape VXY, angular speed, readiness and their
    gate-aligned motion cost only in the final TTC window.  A one-hit recovery
    lesson instead terminates before that next-contact window is reached, so
    an opt-in lesson must expose those same motion components immediately
    after the counted hit.  Pre-hit timing stays bit-for-bit legacy.
    """

    dt_s = float(control_dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("control_dt_s must be positive and finite")
    reward_mask = jnp.asarray(reward_active, dtype=jnp.bool_)
    final_mask = jnp.asarray(final_window_active, dtype=jnp.bool_)
    active = reward_mask & final_mask
    if bool(dense_post_hit_recovery_enabled):
        active = active | (reward_mask & (jnp.asarray(hit_count) > 0))
    return active.astype(jnp.float32) * dt_s


def fixed_contact_first_hit_timeout_mask(
    *,
    hit_count: jax.Array,
    pending_hit: jax.Array,
    episode_time_s: jax.Array,
    deadline_s: float,
) -> jax.Array:
    """End a placed-ball episode that never produces its first valid launch.

    A physical contact already awaiting launch confirmation is allowed to
    resolve through the existing confirmation window. A zero deadline keeps
    every historical course numerically unchanged.
    """

    deadline = float(deadline_s)
    if deadline < 0.0 or not math.isfinite(deadline):
        raise ValueError("fixed-contact first-hit deadline must be finite and nonnegative")
    if deadline == 0.0:
        return jnp.zeros_like(jnp.asarray(hit_count), dtype=jnp.bool_)
    return (
        (jnp.asarray(hit_count) <= 0)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (jnp.asarray(episode_time_s, dtype=jnp.float32) >= deadline)
    )


def fixed_contact_recurrent_hit_timeout_mask(
    *,
    hit_count: jax.Array,
    pending_hit: jax.Array,
    new_counted_hit: jax.Array,
    time_since_counted_hit_s: jax.Array,
    deadline_s: float,
) -> jax.Array:
    """End a one-or-more-hit episode whose next contact never arrives.

    The deadline is measured from the most recent counted hit.  A new counted
    hit and a physical contact awaiting confirmation both suppress timeout on
    the current control tick.  Zero preserves all historical courses.
    """

    deadline = float(deadline_s)
    if deadline < 0.0 or not math.isfinite(deadline):
        raise ValueError(
            "fixed-contact recurrent-hit deadline must be finite and nonnegative"
        )
    if deadline == 0.0:
        return jnp.zeros_like(jnp.asarray(hit_count), dtype=jnp.bool_)
    return (
        (jnp.asarray(hit_count) > 0)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (~jnp.asarray(new_counted_hit, dtype=jnp.bool_))
        & (
            jnp.asarray(time_since_counted_hit_s, dtype=jnp.float32)
            >= deadline
        )
    )


def fixed_contact_conversion_events(
    *,
    hit_edge: jax.Array,
    counted_hit: jax.Array,
    failed_hit: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    pending_hit: jax.Array,
    resolved_only: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return conversion numerator, denominator and terminal censor events.

    Historical courses divide counted hits by every physical contact edge.
    That treats a contact whose confirmation window crosses the finite episode
    boundary as a failure.  Versioned successors may instead count a
    denominator event only when the edge has resolved as either a counted or
    explicit failed hit.  Terminal pending contacts remain observable as a
    separate right-censor event and are never reclassified as successes.
    """

    numerator = jnp.asarray(counted_hit, dtype=jnp.bool_)
    denominator = (
        numerator | jnp.asarray(failed_hit, dtype=jnp.bool_)
        if bool(resolved_only)
        else jnp.asarray(hit_edge, dtype=jnp.bool_)
    )
    terminal_pending = (
        (
            jnp.asarray(terminated, dtype=jnp.bool_)
            | jnp.asarray(truncated, dtype=jnp.bool_)
        )
        & jnp.asarray(pending_hit, dtype=jnp.bool_)
    )
    return numerator, denominator, terminal_pending


def fixed_contact_physical_edge_events(
    *,
    in_contact: jax.Array,
    previous_contact: jax.Array,
    hit_armed: jax.Array,
    pending_hit: jax.Array,
    reject_recontact: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Separate raw contacts, new logical hits and pending recontacts.

    Historical courses suppress every physical contact edge while a previous
    edge is waiting for launch confirmation.  That permits a later impact to
    supply the outgoing velocity for the first edge while rewards and safety
    metrics retain the first edge's cached racket state.  Opt-in courses can
    reject a *distinct* recontact: the ball must first satisfy the existing
    no-contact/distance re-arm contract, so single-step contact chatter does
    not become a false failure.

    The returned logical edge is intentionally identical to the historical
    expression.  This helper changes no legacy dynamics unless the caller
    enables ``reject_recontact``.
    """

    raw_physical_edge = (
        jnp.asarray(in_contact, dtype=jnp.bool_)
        & (~jnp.asarray(previous_contact, dtype=jnp.bool_))
    )
    armed = jnp.asarray(hit_armed, dtype=jnp.bool_)
    pending = jnp.asarray(pending_hit, dtype=jnp.bool_)
    logical_hit_edge = raw_physical_edge & armed & (~pending)
    pending_recontact_edge = jnp.where(
        bool(reject_recontact),
        raw_physical_edge & armed & pending,
        jnp.zeros_like(raw_physical_edge),
    )
    return raw_physical_edge, logical_hit_edge, pending_recontact_edge


def fixed_contact_hit_racket_vxy_penalty_mask(
    *,
    base_mask: jax.Array,
    hit_count: jax.Array,
    min_hit_count: int,
) -> jax.Array:
    """Gate lateral racket-speed shaping by counted-hit ordinal.

    A zero threshold preserves the historical mask exactly.  Positive values
    let versioned courses protect the large, task-required lateral correction
    on early return hits while shaping only the recurrent steady-state orbit.
    """

    minimum_hits = int(min_hit_count)
    if minimum_hits < 0:
        raise ValueError("hit-racket-vxy penalty minimum hit count must be >= 0")
    mask = jnp.asarray(base_mask, dtype=jnp.bool_)
    if minimum_hits == 0:
        return mask
    return mask & (jnp.asarray(hit_count) >= minimum_hits)


def fixed_contact_edge_racket_vz_target_reward(
    *,
    racket_vz_m_s: jax.Array,
    hit_edge: jax.Array,
    hit_count: jax.Array,
    target_vz_m_s: float | jax.Array,
    sigma_vz_m_s: float,
    reward_weight: float,
    min_hit_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reward an outcome-aligned vertical speed at physical contact.

    This event term uses the actual racket velocity at the collision edge.  It
    does not prescribe a full TTC-indexed trajectory, and therefore leaves the
    policy free to choose the preceding motion needed for lateral recovery and
    Cartesian feasibility.  Zero weight preserves historical profiles.
    """

    target = jnp.asarray(target_vz_m_s, dtype=jnp.float32)
    sigma = float(sigma_vz_m_s)
    weight = float(reward_weight)
    minimum_hits = int(min_hit_count)
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("contact-edge racket-vz sigma must be finite and positive")
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError("contact-edge racket-vz weight must be finite and nonnegative")
    if minimum_hits < 0:
        raise ValueError("contact-edge racket-vz minimum hit count must be >= 0")
    error = (jnp.asarray(racket_vz_m_s, dtype=jnp.float32) - target) / sigma
    score = jnp.exp(-0.5 * error * error)
    active = (
        (weight > 0.0)
        & jnp.asarray(hit_edge, dtype=jnp.bool_)
        & (jnp.asarray(hit_count) >= minimum_hits)
    )
    reward = weight * score * active.astype(jnp.float32)
    return reward, score, active


def fixed_contact_gate_event_reward_by_quality(
    *,
    event_reward: jax.Array,
    quality_score: jax.Array,
    gate_mix: float,
) -> tuple[jax.Array, jax.Array]:
    """Couple sparse event credit to its same-edge physical quality.

    A zero mix is an exact legacy identity.  A full mix makes positive event
    credit proportional to the bounded quality score, so optimizing the event
    cannot reward a physically mislocated contact that fails graduation.
    """

    mix = float(gate_mix)
    if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise ValueError("contact-edge event quality gate mix must be in [0, 1]")
    score = jnp.clip(jnp.asarray(quality_score, dtype=jnp.float32), 0.0, 1.0)
    gate = (1.0 - mix) + mix * score
    return jnp.asarray(event_reward, dtype=jnp.float32) * gate, gate


def fixed_contact_phase_consistent_stroke_z_cost(
    *,
    racket_surface_z_m: jax.Array,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    pending_hit: jax.Array,
    contact_surface_z_m: float,
    target_vz_m_s: float,
    lead_s: float,
    prepare_s: float | None = None,
    error_scale_m: float,
    min_hit_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Track a Z reference that reaches the contact plane with nonzero speed.

    During the final lead window the reference is
    ``z_contact - target_vz * TTC``.  Following it therefore places the
    racket below the plane before the stroke and crosses the plane at the
    requested upward speed, instead of holding at the plane and then moving
    upward into an inevitably early/high collision.  The pseudo-Huber cost is
    bounded in slope and constrains only Z, leaving lateral interception free.
    """

    contact_z = float(contact_surface_z_m)
    target_vz = float(target_vz_m_s)
    lead = float(lead_s)
    prepare = lead if prepare_s is None else float(prepare_s)
    scale = float(error_scale_m)
    minimum_hits = int(min_hit_count)
    if not math.isfinite(contact_z):
        raise ValueError("phase-stroke contact surface z must be finite")
    if not math.isfinite(target_vz) or target_vz < 0.0:
        raise ValueError("phase-stroke target vz must be finite and nonnegative")
    if not math.isfinite(lead) or lead <= 0.0:
        raise ValueError("phase-stroke lead must be finite and positive")
    if not math.isfinite(prepare) or prepare < lead:
        raise ValueError("phase-stroke prepare time must be finite and >= lead")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("phase-stroke error scale must be finite and positive")
    if minimum_hits < 0:
        raise ValueError("phase-stroke minimum hit count must be nonnegative")
    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    stroke_target_z = contact_z - target_vz * jnp.clip(ttc, 0.0, lead)
    prepare_target_z = contact_z - target_vz * jnp.clip(
        prepare - ttc,
        0.0,
        prepare - lead,
    )
    target_z = jnp.where(ttc > lead, prepare_target_z, stroke_target_z)
    active = (
        jnp.asarray(crossing_valid, dtype=jnp.bool_)
        & (ttc >= 0.0)
        & (ttc <= prepare)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (jnp.asarray(hit_count) >= minimum_hits)
    )
    normalized_error = (
        jnp.asarray(racket_surface_z_m, dtype=jnp.float32) - target_z
    ) / scale
    cost = jnp.where(
        active,
        jnp.sqrt(1.0 + normalized_error * normalized_error) - 1.0,
        0.0,
    )
    return cost, target_z, active


def fixed_contact_phase_dense_z_target(
    *,
    legacy_target_z_m: jax.Array,
    phase_target_z_m: jax.Array,
    phase_active: jax.Array,
    replace_enabled: bool,
) -> jax.Array:
    """Select one dense Z reference without changing legacy profiles."""

    legacy = jnp.asarray(legacy_target_z_m, dtype=jnp.float32)
    phase = jnp.asarray(phase_target_z_m, dtype=jnp.float32)
    if legacy.shape != phase.shape:
        raise ValueError("legacy and phase dense-Z targets must have equal shape")
    replace = bool(replace_enabled) & jnp.asarray(phase_active, dtype=jnp.bool_)
    return jnp.where(replace, phase, legacy)


def fixed_contact_phase_stroke_vz_cost(
    *,
    racket_vz_m_s: jax.Array,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    pending_hit: jax.Array,
    target_speed_m_s: float,
    lead_s: float,
    prepare_s: float,
    error_scale_m_s: float,
    min_hit_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Track the signed velocity of the continuous prepare/stroke path."""

    speed = float(target_speed_m_s)
    lead = float(lead_s)
    prepare = float(prepare_s)
    scale = float(error_scale_m_s)
    minimum_hits = int(min_hit_count)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("phase-stroke speed must be finite and nonnegative")
    if not math.isfinite(lead) or lead <= 0.0:
        raise ValueError("phase-stroke velocity lead must be positive")
    if not math.isfinite(prepare) or prepare < lead:
        raise ValueError("phase-stroke velocity prepare time must be >= lead")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("phase-stroke velocity scale must be positive")
    if minimum_hits < 0:
        raise ValueError("phase-stroke velocity minimum hits must be nonnegative")
    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    target = jnp.where(ttc > lead, -speed, speed)
    active = (
        jnp.asarray(crossing_valid, dtype=jnp.bool_)
        & (ttc > 0.0)
        & (ttc <= prepare)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (jnp.asarray(hit_count) >= minimum_hits)
    )
    target = jnp.where(active, target, 0.0)
    normalized_error = (
        jnp.asarray(racket_vz_m_s, dtype=jnp.float32) - target
    ) / scale
    cost = jnp.where(
        active,
        jnp.sqrt(1.0 + normalized_error * normalized_error) - 1.0,
        0.0,
    )
    return cost, target, active


def fixed_contact_impact_consistent_racket_vz_target(
    *,
    incoming_ball_vz_m_s: jax.Array,
    contact_surface_z_m: float,
    ball_radius_m: float,
    target_apex_z_m: float,
    gravity_m_s2: float,
    effective_restitution: float,
    min_racket_vz_m_s: float,
    max_racket_vz_m_s: float,
) -> jax.Array:
    """Solve the normal impact speed required by the apex boundary condition.

    For an effectively infinite-mass racket, restitution gives
    ``v_ball_out = (1 + e) * v_racket - e * v_ball_in``.  The requested
    outgoing ball speed is derived from the registered contact plane and
    absolute apex, so contact height and impact energy cannot be optimized as
    independent objectives.
    """

    contact_z = float(contact_surface_z_m)
    radius = float(ball_radius_m)
    apex_z = float(target_apex_z_m)
    gravity = abs(float(gravity_m_s2))
    restitution = float(effective_restitution)
    minimum = float(min_racket_vz_m_s)
    maximum = float(max_racket_vz_m_s)
    scalars = (contact_z, radius, apex_z, gravity, restitution, minimum, maximum)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("impact-consistent target parameters must be finite")
    if radius <= 0.0 or gravity <= 0.0:
        raise ValueError("ball radius and gravity magnitude must be positive")
    if not 0.0 < restitution <= 1.0:
        raise ValueError("effective restitution must lie in (0, 1]")
    if minimum < 0.0 or maximum < minimum:
        raise ValueError("impact-consistent racket-vz bounds must be ordered")
    contact_center_z = contact_z + radius
    if apex_z <= contact_center_z:
        raise ValueError("target apex must lie above the contact ball center")
    desired_ball_out_vz = math.sqrt(
        2.0 * gravity * (apex_z - contact_center_z)
    )
    incoming_vz = jnp.asarray(incoming_ball_vz_m_s, dtype=jnp.float32)
    racket_vz = (
        desired_ball_out_vz + restitution * incoming_vz
    ) / (1.0 + restitution)
    return jnp.clip(racket_vz, minimum, maximum)


def fixed_contact_dynamic_phase_stroke_targets(
    *,
    time_to_contact_s: jax.Array,
    target_speed_m_s: jax.Array,
    crossing_valid: jax.Array,
    pending_hit: jax.Array,
    hit_count: jax.Array,
    contact_surface_z_m: float,
    lead_s: float,
    prepare_s: float,
    min_hit_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a continuous prepare/strike pose and signed speed per lane."""

    contact_z = float(contact_surface_z_m)
    lead = float(lead_s)
    prepare = float(prepare_s)
    minimum_hits = int(min_hit_count)
    if not math.isfinite(contact_z):
        raise ValueError("dynamic phase contact surface z must be finite")
    if not math.isfinite(lead) or lead <= 0.0:
        raise ValueError("dynamic phase lead must be finite and positive")
    if not math.isfinite(prepare) or prepare < lead:
        raise ValueError("dynamic phase prepare time must be >= lead")
    if minimum_hits < 0:
        raise ValueError("dynamic phase minimum hits must be nonnegative")
    speed = jnp.asarray(target_speed_m_s, dtype=jnp.float32)
    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    prepare_z = contact_z - speed * jnp.clip(
        prepare - ttc, 0.0, prepare - lead
    )
    strike_z = contact_z - speed * jnp.clip(ttc, 0.0, lead)
    target_z = jnp.where(ttc > lead, prepare_z, strike_z)
    target_vz = jnp.where(ttc > lead, -speed, speed)
    active = (
        jnp.asarray(crossing_valid, dtype=jnp.bool_)
        & (ttc > 0.0)
        & (ttc <= prepare)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (jnp.asarray(hit_count) >= minimum_hits)
    )
    return target_z, jnp.where(active, target_vz, 0.0), active


def fixed_contact_task_twist_cost(
    *,
    command_linear_velocity_m_s: jax.Array,
    command_angular_velocity_rad_s: jax.Array,
    target_linear_velocity_m_s: jax.Array,
    xy_error_scale_m_s: float,
    z_error_scale_m_s: float,
    tilt_omega_error_scale_rad_s: float,
    active: jax.Array,
    aggregation: str = "mean",
) -> tuple[jax.Array, jax.Array]:
    """Score one executable 5-D surface twist after QVEL projection.

    ``mean`` preserves the historical diagnostic. ``max`` is the
    non-compensable contract for a collision boundary whose XY, Z and tilt
    requirements must all hold at the same instant.
    """

    scales = (
        float(xy_error_scale_m_s),
        float(z_error_scale_m_s),
        float(tilt_omega_error_scale_rad_s),
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("task-twist error scales must be finite and positive")
    aggregation_name = str(aggregation).strip().lower()
    if aggregation_name not in {"mean", "max"}:
        raise ValueError("task-twist aggregation must be 'mean' or 'max'")
    linear = jnp.asarray(command_linear_velocity_m_s, dtype=jnp.float32)
    angular = jnp.asarray(command_angular_velocity_rad_s, dtype=jnp.float32)
    target = jnp.asarray(target_linear_velocity_m_s, dtype=jnp.float32)
    if linear.shape != target.shape or linear.shape[-1] != 3:
        raise ValueError("command and target linear velocities must end in 3")
    if angular.shape != linear.shape:
        raise ValueError("angular velocity must match linear velocity shape")
    xy_error = jnp.linalg.norm(linear[..., :2] - target[..., :2], axis=-1)
    z_error = linear[..., 2] - target[..., 2]
    tilt_error = jnp.linalg.norm(angular[..., :2], axis=-1)
    normalized = jnp.stack(
        [
            xy_error / scales[0],
            z_error / scales[1],
            tilt_error / scales[2],
        ],
        axis=-1,
    )
    components = jnp.sqrt(1.0 + normalized * normalized) - 1.0
    cost = (
        jnp.max(components, axis=-1)
        if aggregation_name == "max"
        else jnp.mean(components, axis=-1)
    )
    return jnp.where(jnp.asarray(active, dtype=jnp.bool_), cost, 0.0), components


def fixed_contact_pre_hit_launch_vz_reward(
    *,
    racket_vz_m_s: jax.Array,
    hit_count: jax.Array,
    episode_time_s: jax.Array,
    target_vz_m_s: float,
    active_window_s: float,
    reward_weight: float,
    dt_s: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reward a brief upward racket launch before the first counted hit.

    The score is signed and bounded, so a stationary racket receives no
    credit and downward motion is worse than waiting.  The first-hit and time
    masks make the term unsuitable as a recurrent or indefinitely collectable
    survival reward.  A zero weight and window preserve historical courses.
    """

    target_vz = float(target_vz_m_s)
    window_s = float(active_window_s)
    weight = float(reward_weight)
    dt = float(dt_s)
    if target_vz <= 0.0 or not math.isfinite(target_vz):
        raise ValueError("fixed-contact launch target vz must be finite and positive")
    if window_s < 0.0 or not math.isfinite(window_s):
        raise ValueError("fixed-contact launch window must be finite and nonnegative")
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError("fixed-contact launch reward weight must be finite and nonnegative")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("fixed-contact launch reward dt must be finite and positive")
    score = jnp.clip(
        jnp.asarray(racket_vz_m_s, dtype=jnp.float32) / target_vz,
        -1.0,
        1.0,
    )
    active = (
        (weight > 0.0)
        & (window_s > 0.0)
        & (jnp.asarray(hit_count) <= 0)
        & (jnp.asarray(episode_time_s, dtype=jnp.float32) <= window_s)
    )
    reward = weight * score * active.astype(jnp.float32) * dt
    return reward, score, active


def fixed_contact_pre_hit_launch_pending_mask(
    *,
    launch_reward: jax.Array,
    launch_active: jax.Array,
    pending_hit: jax.Array,
    enabled: bool,
) -> tuple[jax.Array, jax.Array]:
    """Optionally end upward launch credit at the physical contact edge.

    Historical launch lessons use only the counted-hit mask, so their upward
    reward remains active while an unconfirmed physical contact is pending.
    A post-contact retraction lesson must remove that opposing credit.  The
    explicit flag keeps all existing profiles numerically unchanged.
    """

    if not bool(enabled):
        return launch_reward, launch_active
    keep = ~jnp.asarray(pending_hit, dtype=jnp.bool_)
    return (
        jnp.where(keep, launch_reward, jnp.zeros_like(launch_reward)),
        jnp.asarray(launch_active, dtype=jnp.bool_) & keep,
    )


def fixed_contact_recurrent_launch_vz_reward(
    *,
    racket_vz_m_s: jax.Array,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    pending_hit: jax.Array,
    target_vz_m_s: float,
    active_window_s: float,
    reward_weight: float,
    dt_s: float,
    min_hit_count: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reward a causal upward stroke shortly before recurrent contact.

    Unlike the placed-ball launch helper, this term is disabled until a first
    hit has been confirmed.  It then uses only the policy-visible descending
    contact prediction and current racket velocity.  A pending physical hit
    ends credit, preventing the upward term from opposing post-contact
    separation.  Zero weight/window preserves every historical course.
    """

    target_vz = float(target_vz_m_s)
    window_s = float(active_window_s)
    weight = float(reward_weight)
    dt = float(dt_s)
    if target_vz <= 0.0 or not math.isfinite(target_vz):
        raise ValueError("fixed-contact recurrent launch target vz must be finite and positive")
    if window_s < 0.0 or not math.isfinite(window_s):
        raise ValueError("fixed-contact recurrent launch window must be finite and nonnegative")
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError("fixed-contact recurrent launch weight must be finite and nonnegative")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("fixed-contact recurrent launch dt must be finite and positive")
    minimum_hits = int(min_hit_count)
    if minimum_hits < 0:
        raise ValueError("fixed-contact recurrent launch minimum hit count must be >= 0")
    score = jnp.clip(
        jnp.asarray(racket_vz_m_s, dtype=jnp.float32) / target_vz,
        -1.0,
        1.0,
    )
    time_to_contact = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    active = (
        (weight > 0.0)
        & (window_s > 0.0)
        & (jnp.asarray(hit_count) >= minimum_hits)
        & jnp.asarray(crossing_valid, dtype=jnp.bool_)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (time_to_contact > 0.0)
        & (time_to_contact <= window_s)
    )
    reward = weight * score * active.astype(jnp.float32) * dt
    return reward, score, active


def workspace_barrier_height(
    *,
    site_z_m: jax.Array,
    surface_z_m: jax.Array,
    use_site_height: bool,
) -> jax.Array:
    """Select the termination coordinate or the historical surface coordinate."""
    return jnp.asarray(site_z_m if use_site_height else surface_z_m)


def fixed_contact_workspace_z_barrier_cost(
    *,
    racket_surface_z_m: jax.Array,
    lower_z_m: float,
    upper_z_m: float,
    margin_m: float,
    racket_vertical_velocity_m_s: jax.Array | None = None,
    downward_lookahead_s: float = 0.0,
) -> jax.Array:
    """Return a normalized soft cost before the physical Z termination.

    The zero-cost interval is ``[lower + margin, upper - margin]``.  Cost is
    one exactly at either physical boundary and continues to grow outside it.
    This gives PPO a dense recovery signal while the unchanged outer bounds
    remain the hard safety/episode boundary.  Callers keep the weight at zero
    for every historical profile.
    """

    lower = float(lower_z_m)
    upper = float(upper_z_m)
    margin = float(margin_m)
    if not all(math.isfinite(value) for value in (lower, upper, margin)):
        raise ValueError("workspace Z barrier bounds and margin must be finite")
    if upper <= lower:
        raise ValueError("workspace Z barrier upper bound must exceed lower")
    if margin <= 0.0 or 2.0 * margin >= upper - lower:
        raise ValueError(
            "workspace Z barrier margin must be positive and leave an interior"
        )
    z = jnp.asarray(racket_surface_z_m, dtype=jnp.float32)
    lower_intrusion = jax.nn.relu((lower + margin) - z) / margin
    lookahead = float(downward_lookahead_s)
    if not math.isfinite(lookahead) or lookahead < 0.0:
        raise ValueError("downward reward lookahead must be finite and nonnegative")
    if lookahead > 0.0:
        if racket_vertical_velocity_m_s is None:
            raise ValueError("downward reward lookahead requires racket Z velocity")
        # Reward forecast only: never alter state, commands or termination.
        # Rising motion retains its exact historical reward, including lift.
        vz = jnp.asarray(racket_vertical_velocity_m_s, dtype=jnp.float32)
        forecast_z = z + jnp.minimum(vz, 0.0) * lookahead
        forecast_lower = jax.nn.relu((lower + margin) - forecast_z) / margin
        lower_intrusion = jnp.maximum(lower_intrusion, forecast_lower)
    upper_intrusion = jax.nn.relu(z - (upper - margin)) / margin
    intrusion = lower_intrusion + upper_intrusion
    return intrusion * intrusion


def fixed_contact_smooth_stroke_vz_cost(
    *,
    racket_vz_m_s: jax.Array,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    pending_hit: jax.Array,
    peak_vz_m_s: float,
    lead_s: float,
    brake_lead_s: float,
    error_scale_m_s: float,
    min_hit_count: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return a robust cost for a finite pre-contact vertical stroke.

    The target rises smoothly from zero when the stroke window opens to
    ``peak_vz_m_s`` at ``brake_lead_s`` and then holds that permitted vertical
    contact speed until impact.  A physical contact ends the target; a
    separate bounded retraction term handles separation.  Horizontal and
    angular motion remain governed by the unchanged collision-time gates.

    The pseudo-Huber cost is non-saturating and has a finite gradient for all
    finite velocities.  Only policy-visible crossing state is used.
    """

    peak_vz = float(peak_vz_m_s)
    lead = float(lead_s)
    brake_lead = float(brake_lead_s)
    scale = float(error_scale_m_s)
    minimum_hits = int(min_hit_count)
    if peak_vz <= 0.0 or not math.isfinite(peak_vz):
        raise ValueError("smooth stroke peak vz must be finite and positive")
    if lead <= 0.0 or not math.isfinite(lead):
        raise ValueError("smooth stroke lead must be finite and positive")
    if (
        brake_lead < 0.0
        or not math.isfinite(brake_lead)
        or brake_lead >= lead
    ):
        raise ValueError(
            "smooth stroke brake lead must be finite and in [0, lead)"
        )
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("smooth stroke error scale must be finite and positive")
    if minimum_hits < 0:
        raise ValueError("smooth stroke minimum hit count must be >= 0")

    time_to_contact = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    phase = jnp.clip(
        (lead - time_to_contact) / (lead - brake_lead),
        0.0,
        1.0,
    )
    target = peak_vz * 0.5 * (1.0 - jnp.cos(math.pi * phase))
    active = (
        jnp.asarray(crossing_valid, dtype=jnp.bool_)
        & (~jnp.asarray(pending_hit, dtype=jnp.bool_))
        & (jnp.asarray(hit_count) >= minimum_hits)
        & (time_to_contact > 0.0)
        & (time_to_contact <= lead)
    )
    target = jnp.where(active, target, 0.0)
    normalized_error = (
        jnp.asarray(racket_vz_m_s, dtype=jnp.float32) - target
    ) / scale
    cost = (jnp.sqrt(1.0 + normalized_error**2) - 1.0) * active.astype(
        jnp.float32
    )
    return cost, target, active


def fixed_contact_smooth_stroke_vz_reward(
    *,
    racket_vz_m_s: jax.Array,
    hit_count: jax.Array,
    crossing_valid: jax.Array,
    time_to_contact_s: jax.Array,
    pending_hit: jax.Array,
    peak_vz_m_s: float,
    lead_s: float,
    brake_lead_s: float,
    error_scale_m_s: float,
    reward_weight: float,
    dt_s: float,
    min_hit_count: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return bounded positive credit for matching the smooth VZ target.

    Unlike the corresponding cost, this term never makes entering the
    post-hit state worse than avoiding the preceding confirmed hit.  The
    reciprocal pseudo-Huber score stays finite and supplies a nonzero gradient
    for large errors while remaining bounded by ``reward_weight * dt_s``.
    """

    weight = float(reward_weight)
    dt = float(dt_s)
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError(
            "smooth stroke reward weight must be finite and nonnegative"
        )
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("smooth stroke reward dt must be finite and positive")
    cost, target, active = fixed_contact_smooth_stroke_vz_cost(
        racket_vz_m_s=racket_vz_m_s,
        hit_count=hit_count,
        crossing_valid=crossing_valid,
        time_to_contact_s=time_to_contact_s,
        pending_hit=pending_hit,
        peak_vz_m_s=peak_vz_m_s,
        lead_s=lead_s,
        brake_lead_s=brake_lead_s,
        error_scale_m_s=error_scale_m_s,
        min_hit_count=min_hit_count,
    )
    score = jnp.where(active, 1.0 / (1.0 + cost), 0.0)
    reward = weight * score * dt
    return reward, score, active


def fixed_contact_confirmed_hit_handoff_reward(
    *,
    new_hit: jax.Array,
    hit_count: jax.Array,
    target_hit_count: int,
    reward_weight: float,
) -> jax.Array:
    """Credit one confirmed curriculum handoff event at most once per episode."""

    target_count = int(target_hit_count)
    weight = float(reward_weight)
    if target_count <= 0:
        raise ValueError("handoff target hit count must be positive")
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError("handoff reward weight must be finite and nonnegative")
    handoff = jnp.asarray(new_hit, dtype=jnp.bool_) & (
        jnp.asarray(hit_count) == target_count
    )
    return weight * handoff.astype(jnp.float32)


def fixed_contact_pending_hit_retract_vz_reward(
    *,
    racket_vz_m_s: jax.Array,
    pending_hit: jax.Array,
    pending_hit_steps: jax.Array,
    hit_count: jax.Array,
    target_retract_vz_m_s: float,
    active_window_s: float,
    reward_weight: float,
    dt_s: float,
    all_confirmed_hits: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reward a brief downward racket retraction after physical contact.

    A contact can remain unconfirmed when the racket keeps pushing upward
    with the ball.  The signed, bounded score supplies the missing second
    phase of an up/down stroke.  By default it retains the historical
    first-contact-only behavior.  A versioned recurrent course may opt in to
    every pending contact through ``all_confirmed_hits``.  It is always
    bounded to a finite number of post-contact ticks; zero weight/window
    preserves every historical course.
    """

    target_vz = float(target_retract_vz_m_s)
    window_s = float(active_window_s)
    weight = float(reward_weight)
    dt = float(dt_s)
    if target_vz <= 0.0 or not math.isfinite(target_vz):
        raise ValueError(
            "fixed-contact retract target vz must be finite and positive"
        )
    if window_s < 0.0 or not math.isfinite(window_s):
        raise ValueError(
            "fixed-contact retract window must be finite and nonnegative"
        )
    if weight < 0.0 or not math.isfinite(weight):
        raise ValueError(
            "fixed-contact retract reward weight must be finite and nonnegative"
        )
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError(
            "fixed-contact retract reward dt must be finite and positive"
        )
    score = jnp.clip(
        -jnp.asarray(racket_vz_m_s, dtype=jnp.float32) / target_vz,
        -1.0,
        1.0,
    )
    elapsed_s = (
        jnp.asarray(pending_hit_steps, dtype=jnp.float32) * dt
    )
    active = (
        (weight > 0.0)
        & (window_s > 0.0)
        & jnp.asarray(pending_hit, dtype=jnp.bool_)
        & (
            bool(all_confirmed_hits)
            | (jnp.asarray(hit_count) <= 0)
        )
        & (jnp.asarray(pending_hit_steps) > 0)
        & (elapsed_s <= window_s)
    )
    reward = weight * score * active.astype(jnp.float32) * dt
    return reward, score, active


def fixed_contact_reward_sum(
    *,
    hit_reward: jax.Array,
    survival_reward: jax.Array,
    tracking_potential: jax.Array,
    final_brake_reward: jax.Array,
    s0_component_reward: jax.Array,
    action_penalty: jax.Array,
    action_delta_penalty: jax.Array,
    stick_penalty: jax.Array,
    include_stick_penalty: bool,
    task_aligned_auxiliary_reward: jax.Array | None = None,
    include_task_aligned_auxiliary_reward: bool = False,
    recurrent_event_reward: jax.Array | None = None,
    include_recurrent_event_reward: bool = False,
) -> jax.Array:
    """Compose fixed-contact reward with explicitly versioned opt-in terms.

    Historical fixed-contact courses replaced the general task reward with
    this sum.  Several later profiles nevertheless configured and logged
    general recovery/cadence terms, which therefore never affected PPO.  The
    auxiliary input repairs that contract only when a new profile opts in;
    its default preserves every historical checkpoint's reward exactly.
    """

    reward = (
        hit_reward
        + survival_reward
        + tracking_potential
        + final_brake_reward
        + s0_component_reward
        + action_penalty
        + action_delta_penalty
    )
    if include_stick_penalty:
        reward = reward + stick_penalty
    if include_task_aligned_auxiliary_reward:
        if task_aligned_auxiliary_reward is None:
            raise ValueError(
                "task-aligned auxiliary reward is required when enabled"
            )
        reward = reward + task_aligned_auxiliary_reward
    if include_recurrent_event_reward:
        if recurrent_event_reward is None:
            raise ValueError(
                "recurrent event reward is required when enabled"
            )
        reward = reward + recurrent_event_reward
    return reward


def fixed_contact_s0_readiness_mask(
    *,
    tracking_xy_error_m: jax.Array,
    surface_z_error_m: jax.Array,
    pose_error_rad: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    max_xy_error_m: float,
    max_z_error_m: float,
    max_pose_error_rad: float,
    max_vxy_m_s: float,
    max_omega_rad_s: float,
) -> jax.Array:
    """Require simultaneous physical readiness at the planned contact pose."""

    limits = (
        float(max_xy_error_m),
        float(max_z_error_m),
        float(max_pose_error_rad),
        float(max_vxy_m_s),
        float(max_omega_rad_s),
    )
    if any(value <= 0.0 for value in limits):
        raise ValueError("fixed-contact readiness limits must be positive")
    return (
        (jnp.asarray(tracking_xy_error_m) <= limits[0])
        & (jnp.asarray(surface_z_error_m) <= limits[1])
        & (jnp.asarray(pose_error_rad) <= limits[2])
        & (jnp.asarray(racket_vxy_m_s) <= limits[3])
        & (jnp.asarray(racket_omega_rad_s) <= limits[4])
    )


def nominal_same_height_flight_time_s(
    *,
    contact_ball_center_z_m: float,
    apex_z_m: float,
    gravity_m_s2: float = 9.8,
) -> float:
    """Ballistic contact-to-contact time for an equal-height juggling orbit."""

    lift_m = float(apex_z_m) - float(contact_ball_center_z_m)
    if lift_m <= 0.0 or float(gravity_m_s2) <= 0.0:
        raise ValueError("apex must be above contact and gravity must be positive")
    return 2.0 * math.sqrt(2.0 * lift_m / float(gravity_m_s2))


def nominal_episode_hit_count(
    *,
    episode_duration_s: float,
    first_contact_time_s: float,
    same_height_flight_time_s: float,
) -> int:
    """Count contacts fitting in an episode under the nominal periodic orbit."""

    if same_height_flight_time_s <= 0.0:
        raise ValueError("same_height_flight_time_s must be positive")
    if episode_duration_s < 0.0 or first_contact_time_s < 0.0:
        raise ValueError("episode and first-contact times must be non-negative")
    if first_contact_time_s > episode_duration_s:
        return 0
    return 1 + int(
        math.floor(
            (episode_duration_s - first_contact_time_s)
            / same_height_flight_time_s
        )
    )


def resolve_falling_reset_time_to_contact_s(
    sampled_time_to_contact_s: jax.Array,
    *,
    apex_height_m: jax.Array,
    gravity_m_s2: jax.Array,
    min_downward_speed_m_s: float,
    allow_ascending_phase: bool,
) -> jax.Array:
    """Resolve a physically valid reset-to-descending-contact duration.

    Historical falling resets start strictly after the apex, so their maximum
    duration is one descending half-flight (less the requested minimum
    downward speed).  A positioning prerequisite may instead start anywhere
    on the same contact-to-contact orbit, including its rising half.  That
    raises the physically valid maximum to one complete same-height flight;
    it does not change the apex, gravity, or eventual descending contact.
    """

    sampled = jnp.maximum(
        jnp.asarray(sampled_time_to_contact_s, dtype=jnp.float32), 0.0
    )
    apex_height = jnp.maximum(
        jnp.asarray(apex_height_m, dtype=jnp.float32), 1.0e-6
    )
    gravity = jnp.maximum(
        jnp.asarray(gravity_m_s2, dtype=jnp.float32), 1.0e-6
    )
    time_apex_to_contact = jnp.sqrt(2.0 * apex_height / gravity)
    if allow_ascending_phase:
        maximum_time = 2.0 * time_apex_to_contact
    else:
        maximum_time = jnp.maximum(
            0.02,
            time_apex_to_contact
            - max(0.0, float(min_downward_speed_m_s)) / gravity,
        )
    return jnp.minimum(sampled, maximum_time)


def bounded_next_crossing_target_xy(
    contact_xy_m: jax.Array,
    *,
    workspace_center_xy: tuple[float, float],
    max_lateral_displacement_m: float,
) -> jax.Array:
    """Move toward center by at most one declared lateral step per bounce."""

    contact_xy = jnp.asarray(contact_xy_m, dtype=jnp.float32)
    center = jnp.asarray(workspace_center_xy, dtype=jnp.float32)
    correction = center[None, :] - contact_xy
    correction_norm = jnp.linalg.norm(correction, axis=-1, keepdims=True)
    bounded_correction = correction * jnp.minimum(
        1.0,
        float(max_lateral_displacement_m)
        / jnp.maximum(correction_norm, 1.0e-6),
    )
    return contact_xy + bounded_correction


def fixed_contact_next_contact_target_xy(
    *,
    contact_xy_m: jax.Array,
    episode_anchor_xy_m: jax.Array,
    workspace_center_xy: tuple[float, float],
    max_lateral_displacement_m: float,
    mode: str,
) -> jax.Array:
    """Select the next-contact target shared by reward, gates and CMDP.

    The historical objective always returned to the episode reset anchor.
    The opt-in target-aligned mode instead uses the same bounded centerward
    step as :func:`return_quality_scores`, so the policy is not simultaneously
    asked to hit two different points after one impact.
    """

    target_mode = str(mode)
    if target_mode == "episode_anchor":
        return jnp.asarray(episode_anchor_xy_m, dtype=jnp.float32)
    if target_mode == "bounded_workspace_center_step":
        return bounded_next_crossing_target_xy(
            contact_xy_m,
            workspace_center_xy=workspace_center_xy,
            max_lateral_displacement_m=max_lateral_displacement_m,
        )
    raise ValueError(
        "unsupported fixed-contact next-contact target mode: "
        f"{target_mode!r}"
    )


def empty_high_ball_predictor_state(
    n_envs: int, *, history_frames: int = 9
) -> HighBallPredictorState:
    """Create zeroed predictor state for a vectorized reset."""

    return HighBallPredictorState(
        position_history_m=jnp.zeros(
            (int(n_envs), int(history_frames), 3), dtype=jnp.float32
        ),
        time_history_s=jnp.zeros(
            (int(n_envs), int(history_frames)), dtype=jnp.float32
        ),
        history_count=jnp.zeros((int(n_envs),), dtype=jnp.int32),
        active=jnp.zeros((int(n_envs),), dtype=jnp.bool_),
        initial_time_s=jnp.zeros((int(n_envs),), dtype=jnp.float32),
        initial_position_m=jnp.zeros((int(n_envs), 3), dtype=jnp.float32),
        initial_velocity_m_s=jnp.zeros((int(n_envs), 3), dtype=jnp.float32),
    )


def clear_high_ball_predictor_lanes(
    state: HighBallPredictorState, clear_mask: jax.Array
) -> HighBallPredictorState:
    """Clear complete predictor segments on a per-lane discontinuity mask."""

    mask = jnp.asarray(clear_mask, dtype=jnp.bool_)
    empty = empty_high_ball_predictor_state(
        int(mask.shape[0]),
        history_frames=int(state.position_history_m.shape[1]),
    )

    def select(empty_leaf: jax.Array, current_leaf: jax.Array) -> jax.Array:
        mask_shape = (mask.shape[0],) + (1,) * (current_leaf.ndim - 1)
        return jnp.where(mask.reshape(mask_shape), empty_leaf, current_leaf)

    return jax.tree_util.tree_map(select, empty, state)


def _propagate_quadratic_drag(
    position_m: jax.Array,
    velocity_m_s: jax.Array,
    elapsed_s: jax.Array,
    cfg: HighBallPredictorConfig,
) -> tuple[jax.Array, jax.Array]:
    """Forward-Euler propagation matching ``HighBallOcclusionPredictor``."""

    elapsed_s = jnp.maximum(jnp.asarray(elapsed_s, dtype=jnp.float32), 0.0)
    max_steps = max(
        1,
        int(
            math.ceil(
                float(cfg.max_prediction_s)
                / float(cfg.integration_step_s)
            )
        )
        + 1,
    )
    steps = jnp.maximum(
        1,
        jnp.ceil(elapsed_s / float(cfg.integration_step_s)).astype(jnp.int32),
    )
    step_s = elapsed_s / steps.astype(jnp.float32)
    gravity = jnp.asarray([0.0, 0.0, -cfg.gravity_m_s2], dtype=jnp.float32)

    def integrate(index, carry):
        position, velocity = carry
        enabled = index < steps
        acceleration = gravity - (
            float(cfg.drag_coefficient_per_m)
            * jnp.linalg.norm(velocity, axis=-1, keepdims=True)
            * velocity
        )
        next_position = position + velocity * step_s[:, None]
        next_velocity = velocity + acceleration * step_s[:, None]
        return (
            jnp.where(enabled[:, None], next_position, position),
            jnp.where(enabled[:, None], next_velocity, velocity),
        )

    return jax.lax.fori_loop(
        0, max_steps, integrate, (position_m, velocity_m_s)
    )


def _fit_history(
    state: HighBallPredictorState, cfg: HighBallPredictorConfig
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fit runtime-matched robust lines at the most recent trusted frame."""

    count = jnp.minimum(state.history_count, int(cfg.history_frames))
    indices = jnp.arange(int(cfg.history_frames))[None, :]
    mask = indices >= (int(cfg.history_frames) - count[:, None])
    last_time = state.time_history_s[:, -1]
    relative_time = state.time_history_s - last_time[:, None]
    sample_weights = mask.astype(jnp.float32)
    fitted_values = state.position_history_m.at[:, :, 2].add(
        0.5 * float(cfg.gravity_m_s2) * relative_time * relative_time
    )

    def weighted_line_fit(weights: jax.Array) -> tuple[jax.Array, jax.Array]:
        weight_sum = jnp.maximum(jnp.sum(weights, axis=1), 1.0e-9)
        mean_t = jnp.sum(relative_time * weights, axis=1) / weight_sum
        mean_values = jnp.sum(
            fitted_values * weights[:, :, None], axis=1
        ) / weight_sum[:, None]
        centered_t = relative_time - mean_t[:, None]
        denominator = jnp.maximum(
            jnp.sum(weights * centered_t * centered_t, axis=1), 1.0e-9
        )
        slopes = jnp.sum(
            weights[:, :, None]
            * centered_t[:, :, None]
            * (fitted_values - mean_values[:, None, :]),
            axis=1,
        ) / denominator[:, None]
        return slopes, mean_values - slopes * mean_t[:, None]

    slopes, intercept = weighted_line_fit(sample_weights)
    # Match the real optimized predictor's four Huber-IRLS refinements.  The
    # robust scale and weights are per lane and per Cartesian component.
    for _ in range(4):
        residual = fitted_values - (
            intercept[:, None, :] + slopes[:, None, :] * relative_time[:, :, None]
        )
        masked_residual = jnp.where(mask[:, :, None], residual, jnp.nan)
        median = jnp.nan_to_num(
            jnp.nanmedian(masked_residual, axis=1), nan=0.0
        )
        mad = jnp.nan_to_num(
            jnp.nanmedian(
                jnp.abs(masked_residual - median[:, None, :]), axis=1
            ),
            nan=0.0,
        )
        scale = jnp.maximum(1.0e-3, 1.4826 * mad)
        cutoff = 1.5 * scale
        absolute = jnp.abs(residual)
        robust_weights = jnp.where(
            absolute > cutoff[:, None, :],
            cutoff[:, None, :] / jnp.maximum(absolute, 1.0e-12),
            1.0,
        )
        # ``weighted_line_fit`` accepts one weight per component by folding
        # the component dimension into the closed-form computation here.
        weights = sample_weights[:, :, None] * robust_weights
        weight_sum = jnp.maximum(jnp.sum(weights, axis=1), 1.0e-9)
        mean_t = jnp.sum(
            relative_time[:, :, None] * weights, axis=1
        ) / weight_sum
        mean_values = jnp.sum(fitted_values * weights, axis=1) / weight_sum
        centered_t = relative_time[:, :, None] - mean_t[:, None, :]
        denominator = jnp.maximum(
            jnp.sum(weights * centered_t * centered_t, axis=1), 1.0e-9
        )
        slopes = jnp.sum(
            weights
            * centered_t
            * (fitted_values - mean_values[:, None, :]),
            axis=1,
        ) / denominator
        intercept = mean_values - slopes * mean_t

    fitted = intercept[:, None, :] + slopes[:, None, :] * relative_time[:, :, None]
    fitted = fitted.at[:, :, 2].add(
        -0.5 * float(cfg.gravity_m_s2) * relative_time * relative_time
    )
    residual = (state.position_history_m - fitted) * sample_weights[:, :, None]
    weight_sum = jnp.maximum(jnp.sum(sample_weights, axis=1), 1.0)
    rmse = jnp.sqrt(
        jnp.sum(residual * residual, axis=1) / weight_sum[:, None]
    )
    history_span = state.time_history_s[:, -1] - jnp.min(
        jnp.where(mask, state.time_history_s, jnp.inf), axis=1
    )
    return intercept, slopes, rmse, history_span


def update_high_ball_predictor(
    state: HighBallPredictorState,
    measured_position_m: jax.Array,
    measured_velocity_m_s: jax.Array,
    sample_available: jax.Array,
    query_time_s: jax.Array,
    cfg: HighBallPredictorConfig,
    prediction_requested: jax.Array | None = None,
) -> tuple[HighBallPredictorState, HighBallPredictorOutput]:
    """Pass fresh samples through or predict a bounded top-of-view occlusion."""

    position = jnp.asarray(measured_position_m, dtype=jnp.float32)
    velocity = jnp.asarray(measured_velocity_m_s, dtype=jnp.float32)
    available = jnp.asarray(sample_available, dtype=jnp.bool_)
    request_prediction = (
        ~available
        if prediction_requested is None
        else jnp.asarray(prediction_requested, dtype=jnp.bool_)
    )
    query_time = jnp.asarray(query_time_s, dtype=jnp.float32)
    elapsed_active = jnp.maximum(query_time - state.initial_time_s, 0.0)
    predicted_position, predicted_velocity = _propagate_quadratic_drag(
        state.initial_position_m,
        state.initial_velocity_m_s,
        elapsed_active,
        cfg,
    )
    reacquisition_error = jnp.linalg.norm(position - predicted_position, axis=-1)
    reacquisition_rejected = (
        state.active & available & (reacquisition_error > cfg.reacquisition_gate_m)
    )
    measured_accepted = available & (~state.active | ~reacquisition_rejected)

    prior_sample_gap = query_time - state.time_history_s[:, -1]
    duplicate_sample = (
        measured_accepted
        & (state.history_count > 0)
        & (prior_sample_gap <= 1.0e-9)
    )
    measured_accepted = measured_accepted & ~duplicate_sample
    restart_history = measured_accepted & (
        state.active
        | (
            (state.history_count > 0)
            & (prior_sample_gap > cfg.max_history_gap_s + 1.0e-9)
        )
    )
    base_positions = jnp.where(
        restart_history[:, None, None],
        jnp.zeros_like(state.position_history_m),
        state.position_history_m,
    )
    base_times = jnp.where(
        restart_history[:, None],
        jnp.zeros_like(state.time_history_s),
        state.time_history_s,
    )
    base_count = jnp.where(restart_history, 0, state.history_count)

    appended_positions = jnp.concatenate(
        [base_positions[:, 1:, :], position[:, None, :]], axis=1
    )
    appended_times = jnp.concatenate(
        [base_times[:, 1:], query_time[:, None]], axis=1
    )
    history_positions = jnp.where(
        measured_accepted[:, None, None],
        appended_positions,
        base_positions,
    )
    history_times = jnp.where(
        measured_accepted[:, None], appended_times, base_times
    )
    history_count = jnp.where(
        measured_accepted,
        jnp.minimum(base_count + 1, int(cfg.history_frames)),
        base_count,
    )
    history_state = state._replace(
        position_history_m=history_positions,
        time_history_s=history_times,
        history_count=history_count,
    )
    fit_position, fit_velocity, fit_rmse, history_span = _fit_history(
        history_state, cfg
    )
    last_sample_gap = query_time - history_times[:, -1]
    can_activate = (
        request_prediction
        & (~available)
        & (~state.active)
        & (history_count >= int(cfg.history_frames))
        & (history_span >= cfg.min_history_span_s)
        & (last_sample_gap >= -1.0e-6)
        & (last_sample_gap <= cfg.activation_max_gap_s)
        & (fit_position[:, 2] >= cfg.activation_z_m)
        & (fit_velocity[:, 2] >= cfg.min_upward_velocity_m_s)
        & (jnp.max(fit_rmse, axis=-1) <= cfg.fit_max_rmse_m)
        & (jnp.linalg.norm(fit_velocity, axis=-1) <= cfg.max_fitted_speed_m_s)
    )
    active = jnp.where(measured_accepted, False, state.active | can_activate)
    initial_time = jnp.where(can_activate, history_times[:, -1], state.initial_time_s)
    initial_position = jnp.where(
        can_activate[:, None], fit_position, state.initial_position_m
    )
    initial_velocity = jnp.where(
        can_activate[:, None], fit_velocity, state.initial_velocity_m_s
    )
    active_elapsed = jnp.maximum(query_time - initial_time, 0.0)
    active_position, active_velocity = _propagate_quadratic_drag(
        initial_position, initial_velocity, active_elapsed, cfg
    )
    prediction_valid = (
        active
        & (active_elapsed <= cfg.max_prediction_s + 1.0e-6)
        & (active_position[:, 2] >= cfg.stop_prediction_z_m)
    )
    active = active & prediction_valid
    prediction_expired = (
        (state.active | can_activate)
        & (~prediction_valid)
        & (~measured_accepted)
    )
    output_position = jnp.where(
        measured_accepted[:, None], position, active_position
    )
    output_velocity = jnp.where(
        measured_accepted[:, None], velocity, active_velocity
    )
    valid = measured_accepted | prediction_valid
    output_position = jnp.where(valid[:, None], output_position, 0.0)
    output_velocity = jnp.where(valid[:, None], output_velocity, 0.0)
    next_state = history_state._replace(
        position_history_m=jnp.where(
            prediction_expired[:, None, None],
            jnp.zeros_like(history_positions),
            history_positions,
        ),
        time_history_s=jnp.where(
            prediction_expired[:, None],
            jnp.zeros_like(history_times),
            history_times,
        ),
        history_count=jnp.where(prediction_expired, 0, history_count),
        active=active,
        initial_time_s=initial_time,
        initial_position_m=initial_position,
        initial_velocity_m_s=initial_velocity,
    )
    output = HighBallPredictorOutput(
        position_m=output_position,
        velocity_m_s=output_velocity,
        valid=valid,
        observed=measured_accepted,
        predicted=prediction_valid,
        prediction_horizon_s=jnp.where(prediction_valid, active_elapsed, 0.0),
        # Both an accepted measurement and a propagated estimate represent
        # the query time. Prediction horizon is intentionally separate.
        represented_state_age_s=jnp.zeros_like(query_time),
        reacquisition_rejected=reacquisition_rejected,
        fit_rmse_m=fit_rmse,
    )
    return next_state, output


def predict_descending_plane_crossing(
    position_m: jax.Array,
    velocity_m_s: jax.Array,
    center_height_m: float,
    *,
    gravity_m_s2: float = 9.8,
    max_time_s: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the descending ballistic crossing used by the actor target."""

    position = jnp.asarray(position_m, dtype=jnp.float32)
    velocity = jnp.asarray(velocity_m_s, dtype=jnp.float32)
    discriminant = velocity[:, 2] ** 2 + 2.0 * gravity_m_s2 * (
        position[:, 2] - float(center_height_m)
    )
    root = (velocity[:, 2] + jnp.sqrt(jnp.maximum(discriminant, 0.0))) / float(
        gravity_m_s2
    )
    valid = (discriminant >= 0.0) & (root >= 0.0) & (root <= max_time_s)
    crossing_xy = position[:, :2] + velocity[:, :2] * root[:, None]
    return crossing_xy, jnp.where(valid, root, 0.0), valid


def _rotation_from_normal(
    target_normal: jax.Array,
    reference_rotation: jax.Array,
) -> jax.Array:
    """Build a full pose while preserving the reference in-plane heading."""

    normal = jnp.asarray(target_normal, dtype=jnp.float32)
    reference = jnp.asarray(reference_rotation, dtype=jnp.float32)
    if reference.ndim == 2:
        reference = jnp.broadcast_to(reference, (normal.shape[0], 3, 3))
    reference_x = reference[:, :, 0]
    reference_y = reference[:, :, 1]
    projected_x = reference_x - (
        jnp.sum(reference_x * normal, axis=-1, keepdims=True) * normal
    )
    fallback_x = reference_y - (
        jnp.sum(reference_y * normal, axis=-1, keepdims=True) * normal
    )
    projected_norm = jnp.linalg.norm(projected_x, axis=-1, keepdims=True)
    projected_x = jnp.where(
        projected_norm > 1.0e-6,
        projected_x,
        fallback_x,
    )
    target_x = projected_x / jnp.maximum(
        jnp.linalg.norm(projected_x, axis=-1, keepdims=True), 1.0e-6
    )
    target_y = jnp.cross(normal, target_x)
    target_y = target_y / jnp.maximum(
        jnp.linalg.norm(target_y, axis=-1, keepdims=True), 1.0e-6
    )
    target_x = jnp.cross(target_y, normal)
    return jnp.stack([target_x, target_y, normal], axis=-1)


def compute_dynamic_contact_target(
    ball_position_m: jax.Array,
    ball_velocity_m_s: jax.Array,
    reference_rotation: jax.Array,
    *,
    workspace_center_xy: tuple[float, float],
    contact_surface_z_m: float,
    ball_radius_m: float,
    cfg: ContactTargetConfig = ContactTargetConfig(),
    target_apex_z_m: jax.Array | None = None,
) -> ContactTarget:
    """Compute an observable-only pose target for the next racket contact.

    The target position is the represented ball trajectory's descending
    crossing of the fixed contact plane.  Its normal follows the bounded
    impulse direction required to move the next same-height crossing toward
    the registered workspace center by no more than one bounded lateral step.
    This is a geometric nominal target, not a claim of an identified collision
    inverse.
    """

    ball_position = jnp.asarray(ball_position_m, dtype=jnp.float32)
    ball_velocity = jnp.asarray(ball_velocity_m_s, dtype=jnp.float32)
    reference = jnp.asarray(reference_rotation, dtype=jnp.float32)
    if reference.ndim == 2:
        reference = jnp.broadcast_to(
            reference, (ball_position.shape[0], 3, 3)
        )
    ball_contact_z_m = float(contact_surface_z_m) + float(ball_radius_m)
    crossing_xy, time_to_contact, crossing_valid = (
        predict_descending_plane_crossing(
            ball_position,
            ball_velocity,
            ball_contact_z_m,
            gravity_m_s2=float(cfg.gravity_m_s2),
            max_time_s=float(cfg.max_time_to_contact_s),
        )
    )
    target_xy = jnp.where(
        crossing_valid[:, None], crossing_xy, ball_position[:, :2]
    )
    target_position = jnp.concatenate(
        [
            target_xy,
            jnp.full(
                (ball_position.shape[0], 1),
                float(contact_surface_z_m),
                dtype=jnp.float32,
            ),
        ],
        axis=-1,
    )
    predicted_ball_position_at_contact = jnp.concatenate(
        [
            target_xy,
            jnp.full(
                (ball_position.shape[0], 1),
                ball_contact_z_m,
                dtype=jnp.float32,
            ),
        ],
        axis=-1,
    )

    gravity = float(cfg.gravity_m_s2)
    incoming_velocity = ball_velocity.at[:, 2].add(
        -gravity * time_to_contact
    )
    if target_apex_z_m is None:
        target_apex = jnp.full(
            (ball_position.shape[0],),
            float(cfg.target_apex_z_m),
            dtype=jnp.float32,
        )
    else:
        target_apex = jnp.asarray(target_apex_z_m, dtype=jnp.float32)
        if target_apex.shape != (ball_position.shape[0],):
            raise ValueError("target_apex_z_m must have shape (N,)")
    target_apex_lift_m = jnp.maximum(0.0, target_apex - ball_contact_z_m)
    desired_post_vz = jnp.sqrt(
        2.0 * gravity * target_apex_lift_m
    )
    same_height_flight_s = 2.0 * desired_post_vz / gravity
    next_crossing_target_xy = bounded_next_crossing_target_xy(
        target_xy,
        workspace_center_xy=workspace_center_xy,
        max_lateral_displacement_m=(
            cfg.max_lateral_displacement_per_bounce_m
        ),
    )
    desired_post_vxy = (
        next_crossing_target_xy - target_xy
    ) / jnp.maximum(same_height_flight_s[:, None], 1.0e-6)
    desired_post_vxy_norm = jnp.linalg.norm(
        desired_post_vxy, axis=-1, keepdims=True
    )
    desired_post_vxy = desired_post_vxy * jnp.minimum(
        1.0,
        float(cfg.max_post_ball_vxy_m_s)
        / jnp.maximum(desired_post_vxy_norm, 1.0e-6),
    )
    desired_post_velocity = jnp.concatenate(
        [
            desired_post_vxy,
            desired_post_vz[:, None],
        ],
        axis=-1,
    )

    impulse_direction = desired_post_velocity - incoming_velocity
    impulse_normal = impulse_direction / jnp.maximum(
        jnp.linalg.norm(impulse_direction, axis=-1, keepdims=True), 1.0e-6
    )
    impulse_normal = jnp.where(
        impulse_normal[:, 2:3] >= 0.0,
        impulse_normal,
        -impulse_normal,
    )
    tangent_slope = impulse_normal[:, :2] / jnp.maximum(
        impulse_normal[:, 2:3], 1.0e-3
    )
    tangent_limit = jnp.tan(jnp.deg2rad(float(cfg.max_tilt_deg)))
    tangent_norm = jnp.linalg.norm(tangent_slope, axis=-1, keepdims=True)
    tangent_slope = tangent_slope * jnp.minimum(
        1.0,
        tangent_limit / jnp.maximum(tangent_norm, 1.0e-6),
    )
    target_normal = jnp.concatenate(
        [
            tangent_slope,
            jnp.ones((ball_position.shape[0], 1), dtype=jnp.float32),
        ],
        axis=-1,
    )
    target_normal = target_normal / jnp.maximum(
        jnp.linalg.norm(target_normal, axis=-1, keepdims=True), 1.0e-6
    )
    target_rotation = _rotation_from_normal(target_normal, reference)
    finite = (
        jnp.all(jnp.isfinite(ball_position), axis=-1)
        & jnp.all(jnp.isfinite(ball_velocity), axis=-1)
        & jnp.all(jnp.isfinite(target_position), axis=-1)
        & jnp.all(jnp.isfinite(target_rotation), axis=(-2, -1))
    )
    valid = crossing_valid & finite & (target_apex_lift_m > 0.0)
    target_rotation = jnp.where(
        valid[:, None, None], target_rotation, reference
    )
    return ContactTarget(
        position_m=target_position,
        rotation_matrix=target_rotation,
        predicted_ball_position_at_contact_m=(
            predicted_ball_position_at_contact
        ),
        time_to_contact_s=time_to_contact,
        incoming_ball_velocity_m_s=incoming_velocity,
        next_crossing_target_xy_m=next_crossing_target_xy,
        desired_post_ball_velocity_m_s=desired_post_velocity,
        valid=valid,
    )


def build_predictive_contact_contract_v1(
    ball_position_m: jax.Array,
    ball_velocity_m_s: jax.Array,
    reference_rotation: jax.Array,
    *,
    workspace_center_xy: tuple[float, float],
    contact_surface_z_m: float,
    ball_radius_m: float,
    source_state_time_s: jax.Array,
    query_time_s: jax.Array,
    cfg: ContactTargetConfig = ContactTargetConfig(),
) -> tuple[BallisticContactPredictionV1, GeometricOutcomeTargetV1]:
    """Expose the legacy target as flight and outcome semantic structures.

    This adapter deliberately calls :func:`compute_dynamic_contact_target`
    once and only re-labels its values.  Enabling instrumentation therefore
    cannot perturb the historical 80-D actor inputs or contact geometry.
    """

    target = compute_dynamic_contact_target(
        ball_position_m,
        ball_velocity_m_s,
        reference_rotation,
        workspace_center_xy=workspace_center_xy,
        contact_surface_z_m=contact_surface_z_m,
        ball_radius_m=ball_radius_m,
        cfg=cfg,
    )
    source_time = jnp.asarray(source_state_time_s, dtype=jnp.float32)
    query_time = jnp.asarray(query_time_s, dtype=jnp.float32)
    if source_time.shape != target.time_to_contact_s.shape:
        raise ValueError("source_state_time_s must have shape (N,)")
    if query_time.shape != target.time_to_contact_s.shape:
        raise ValueError("query_time_s must have shape (N,)")
    contact_time = query_time + target.time_to_contact_s
    # Horizon is measured from the timestamp of the ball state that produced
    # the prediction, not merely from the current query.  The latter is the
    # observation age and must not be mislabeled as forecast horizon.
    horizon = jnp.maximum(0.0, contact_time - source_time)
    prediction = BallisticContactPredictionV1(
        valid=target.valid,
        query_time_s=query_time,
        contact_time_s=contact_time,
        time_to_contact_s=target.time_to_contact_s,
        contact_position_world_m=target.predicted_ball_position_at_contact_m,
        incoming_velocity_world_m_s=target.incoming_ball_velocity_m_s,
        source_state_time_s=source_time,
        prediction_horizon_s=horizon,
    )
    normal = target.rotation_matrix[:, :, 2]
    nominal_racket_velocity = 0.5 * (
        target.desired_post_ball_velocity_m_s
        + target.incoming_ball_velocity_m_s
    )
    outcome = GeometricOutcomeTargetV1(
        valid=target.valid,
        desired_post_velocity_world_m_s=target.desired_post_ball_velocity_m_s,
        desired_apex_abs_m=jnp.full_like(
            target.time_to_contact_s, float(cfg.target_apex_z_m)
        ),
        desired_next_contact_xy_world_m=target.next_crossing_target_xy_m,
        racket_contact_position_world_m=target.position_m,
        racket_rotation_world=target.rotation_matrix,
        racket_normal_world=normal,
        nominal_racket_linear_velocity_world_m_s=nominal_racket_velocity,
        max_inward_xy_step_m=jnp.full_like(
            target.time_to_contact_s,
            float(cfg.max_lateral_displacement_per_bounce_m),
        ),
    )
    return prediction, outcome


def fixed_contact_ttc_phase(
    time_to_contact_s: jax.Array,
    prediction_valid: jax.Array,
    *,
    contact_elapsed_s: jax.Array,
    arrival_start_s: float = 0.100,
    settle_start_s: float = 0.080,
    strike_start_s: float = 0.060,
    retract_duration_s: float = 0.100,
) -> jax.Array:
    """Classify every lane into one gap-free predictive-contact phase."""

    if not (
        float(arrival_start_s)
        > float(settle_start_s)
        > float(strike_start_s)
        >= 0.0
    ):
        raise ValueError("TTC phase boundaries must be strictly descending")
    if float(retract_duration_s) < 0.0:
        raise ValueError("retract_duration_s must be non-negative")
    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    valid = jnp.asarray(prediction_valid, dtype=jnp.bool_)
    elapsed = jnp.asarray(contact_elapsed_s, dtype=jnp.float32)
    if ttc.shape != valid.shape or ttc.shape != elapsed.shape:
        raise ValueError("TTC phase inputs must have matching shapes")
    pre_contact = elapsed < 0.0
    phase = jnp.where(
        ttc > float(arrival_start_s),
        FIXED_CONTACT_PHASE_APPROACH,
        jnp.where(
            ttc > float(settle_start_s),
            FIXED_CONTACT_PHASE_ARRIVAL_GATE,
            jnp.where(
                ttc > float(strike_start_s),
                FIXED_CONTACT_PHASE_SETTLE,
                FIXED_CONTACT_PHASE_STRIKE,
            ),
        ),
    )
    phase = jnp.where(
        pre_contact,
        phase,
        jnp.where(
            elapsed <= float(retract_duration_s),
            FIXED_CONTACT_PHASE_RETRACT,
            FIXED_CONTACT_PHASE_RECOVER,
        ),
    )
    finite = jnp.isfinite(elapsed) & (jnp.isfinite(ttc) | (~pre_contact))
    phase_valid = valid | (~pre_contact)
    return jnp.where(
        phase_valid & finite,
        phase,
        jnp.asarray(FIXED_CONTACT_PHASE_INVALID, dtype=jnp.int32),
    ).astype(jnp.int32)


def sample_fixed_contact_skill_mode(
    keys: jax.Array,
    *,
    arrival_only_probability: float,
    arrived_hit_probability: float,
    recurrent_probability: float,
) -> jax.Array:
    """Sample one mode per actual reset, independent of episode duration."""

    probabilities = (
        float(arrival_only_probability),
        float(arrived_hit_probability),
        float(recurrent_probability),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
        raise ValueError("skill-mode probabilities must be finite and non-negative")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("skill-mode probabilities must sum to one")
    reset_keys = jnp.asarray(keys)
    if reset_keys.ndim != 2 or reset_keys.shape[-1] != 2:
        raise ValueError("keys must have shape (N, 2)")
    uniform = jax.vmap(lambda key: jax.random.uniform(key, ()))(reset_keys)
    arrival_cut = probabilities[0]
    hit_cut = probabilities[0] + probabilities[1]
    return jnp.where(
        uniform < arrival_cut,
        FIXED_CONTACT_SKILL_MODE_ARRIVAL_ONLY,
        jnp.where(
            uniform < hit_cut,
            FIXED_CONTACT_SKILL_MODE_ARRIVED_HIT,
            FIXED_CONTACT_SKILL_MODE_RECURRENT,
        ),
    ).astype(jnp.int32)


def scripted_airborne_handoff_ball_state(
    *,
    racket_surface_position_m: jax.Array,
    time_to_contact_s: jax.Array,
    target_apex_abs_m: float,
    ball_radius_m: float,
    minimum_surface_gap_m: float,
    gravity_m_s2: float = 9.8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Create an analytic, separated descending state for actor handoff.

    ``target_apex_abs_m`` documents the post-impact task target only. It does
    not alter the inbound handoff: doing so would make the declared TTC and
    analytic crossing inconsistent.
    """

    racket = jnp.asarray(racket_surface_position_m, dtype=jnp.float32)
    tau = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    if racket.ndim != 2 or racket.shape[-1] != 3:
        raise ValueError("racket_surface_position_m must have shape (N, 3)")
    if tau.shape != racket.shape[:-1]:
        raise ValueError("time_to_contact_s must have shape (N,)")
    if (
        float(ball_radius_m) <= 0.0
        or float(minimum_surface_gap_m) < 0.0
        or not math.isfinite(float(target_apex_abs_m))
    ):
        raise ValueError("invalid scripted-airborne physical parameter")
    gravity = jnp.asarray(gravity_m_s2, dtype=jnp.float32)
    gravity = jnp.broadcast_to(gravity, tau.shape)
    gravity = jnp.where(jnp.isfinite(gravity) & (gravity > 0.0), gravity, 9.8)
    tau = jnp.where(jnp.isfinite(tau) & (tau > 0.0), tau, 0.0)
    contact_center_z = racket[:, 2] + float(ball_radius_m)
    # Start strictly on the descending branch.  The small negative handoff
    # velocity also minimizes unnecessary inbound energy for the requested
    # TTC; analytic back-propagation then determines the separated height.
    handoff_vz = -jnp.full_like(tau, 0.05)
    handoff_z = (
        contact_center_z
        - handoff_vz * tau
        + 0.5 * gravity * tau * tau
    )
    minimum_center_z = (
        contact_center_z + float(minimum_surface_gap_m)
    )
    handoff_z = jnp.maximum(handoff_z, minimum_center_z)
    position = racket.at[:, 2].set(handoff_z)
    velocity = jnp.zeros_like(position).at[:, 2].set(handoff_vz)
    actual_apex_z = handoff_z + handoff_vz * handoff_vz / (2.0 * gravity)
    launch_origin = racket.at[:, 2].set(actual_apex_z)
    return position, velocity, launch_origin


def rotation_matrix_to_6d(rotation: jax.Array) -> jax.Array:
    """Return the continuous first-two-column SO(3) representation."""

    matrix = jnp.asarray(rotation, dtype=jnp.float32)
    return jnp.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def rotation_6d_to_matrix(rotation_6d: jax.Array) -> jax.Array:
    """Reconstruct an orthonormal rotation from its continuous 6-D form."""

    value = jnp.asarray(rotation_6d, dtype=jnp.float32)
    first = value[..., 0:3]
    second = value[..., 3:6]
    x_axis = first / jnp.maximum(
        jnp.linalg.norm(first, axis=-1, keepdims=True), 1.0e-6
    )
    second = second - jnp.sum(
        x_axis * second, axis=-1, keepdims=True
    ) * x_axis
    y_axis = second / jnp.maximum(
        jnp.linalg.norm(second, axis=-1, keepdims=True), 1.0e-6
    )
    z_axis = jnp.cross(x_axis, y_axis)
    return jnp.stack([x_axis, y_axis, z_axis], axis=-1)


def so3_log_error(reference_rotation: jax.Array, rotation: jax.Array) -> jax.Array:
    """Compute ``log(R_ref.T @ R)`` as a 3-vector for each lane."""

    relative = jnp.einsum("...ji,...jk->...ik", reference_rotation, rotation)
    cosine = jnp.clip((jnp.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    angle = jnp.arccos(cosine)
    vee = jnp.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        axis=-1,
    )
    regular_scale = jnp.where(
        angle < 1.0e-5,
        0.5,
        angle / jnp.maximum(2.0 * jnp.sin(angle), 1.0e-6),
    )
    regular = vee * regular_scale[..., None]

    # The skew part is identically zero at a 180-degree rotation, so the
    # regular formula above collapses to a zero error exactly where the
    # attitude error is maximal. Recover the half-turn axis from the
    # symmetric/diagonal part instead. Selecting the largest diagonal-derived
    # component keeps every denominator away from zero; the axis sign is
    # immaterial at exactly pi and is chosen consistently by that component.
    axis_abs = jnp.sqrt(
        jnp.maximum(
            0.5
            * (
                jnp.diagonal(relative, axis1=-2, axis2=-1)
                + 1.0
            ),
            0.0,
        )
    )
    axis_index = jnp.argmax(axis_abs, axis=-1)
    safe_axis_abs = jnp.maximum(axis_abs, 1.0e-7)
    axis_x = jnp.stack(
        [
            axis_abs[..., 0],
            (relative[..., 0, 1] + relative[..., 1, 0])
            / (4.0 * safe_axis_abs[..., 0]),
            (relative[..., 0, 2] + relative[..., 2, 0])
            / (4.0 * safe_axis_abs[..., 0]),
        ],
        axis=-1,
    )
    axis_y = jnp.stack(
        [
            (relative[..., 0, 1] + relative[..., 1, 0])
            / (4.0 * safe_axis_abs[..., 1]),
            axis_abs[..., 1],
            (relative[..., 1, 2] + relative[..., 2, 1])
            / (4.0 * safe_axis_abs[..., 1]),
        ],
        axis=-1,
    )
    axis_z = jnp.stack(
        [
            (relative[..., 0, 2] + relative[..., 2, 0])
            / (4.0 * safe_axis_abs[..., 2]),
            (relative[..., 1, 2] + relative[..., 2, 1])
            / (4.0 * safe_axis_abs[..., 2]),
            axis_abs[..., 2],
        ],
        axis=-1,
    )
    near_pi_axis = jnp.where(
        (axis_index == 0)[..., None],
        axis_x,
        jnp.where((axis_index == 1)[..., None], axis_y, axis_z),
    )
    near_pi_axis = near_pi_axis / jnp.maximum(
        jnp.linalg.norm(near_pi_axis, axis=-1, keepdims=True),
        1.0e-7,
    )
    # Below pi the skew term still carries the unique log-map sign. Align the
    # diagonal-derived axis with it; at exact pi the sign is mathematically
    # ambiguous and the zero alignment intentionally keeps the stable choice.
    alignment = jnp.sum(near_pi_axis * vee, axis=-1, keepdims=True)
    near_pi_axis = near_pi_axis * jnp.where(alignment < 0.0, -1.0, 1.0)
    near_pi = near_pi_axis * angle[..., None]
    return jnp.where((jnp.pi - angle < 1.0e-2)[..., None], near_pi, regular)


def make_fixed_contact_actor_observation(
    *,
    q: jax.Array,
    dq: jax.Array,
    q_ref_error: jax.Array,
    previous_executed_action: jax.Array,
    observed_ball_position_current: jax.Array,
    observed_ball_velocity_current: jax.Array,
    prediction_uses_extrapolated_current_state: jax.Array,
    racket_position_current: jax.Array,
    racket_rotation_current: jax.Array,
    racket_linear_velocity: jax.Array,
    racket_angular_velocity: jax.Array,
    contact_target: ContactTarget,
    workspace_center_xy: tuple[float, float],
    target_conditioning_scale: float = 1.0,
) -> jax.Array:
    """Assemble the versioned 80-D current-observation/future-prediction input."""

    target_conditioning_scale = float(target_conditioning_scale)
    if (
        not math.isfinite(target_conditioning_scale)
        or target_conditioning_scale <= 0.0
    ):
        raise ValueError("target_conditioning_scale must be finite and positive")
    center = jnp.asarray(workspace_center_xy, dtype=jnp.float32)
    conditioning_scale = jnp.asarray(
        target_conditioning_scale,
        dtype=jnp.float32,
    )
    return jnp.concatenate(
        [
            q,
            dq,
            q_ref_error,
            previous_executed_action,
            observed_ball_position_current,
            observed_ball_velocity_current,
            contact_target.predicted_ball_position_at_contact_m,
            contact_target.incoming_ball_velocity_m_s,
            contact_target.time_to_contact_s[:, None],
            contact_target.valid.astype(jnp.float32)[:, None],
            prediction_uses_extrapolated_current_state.astype(jnp.float32)[
                :, None
            ],
            racket_position_current,
            rotation_matrix_to_6d(racket_rotation_current),
            racket_linear_velocity,
            racket_angular_velocity,
            contact_target.position_m,
            rotation_matrix_to_6d(contact_target.rotation_matrix),
            (
                contact_target.position_m - racket_position_current
            ) * conditioning_scale,
            so3_log_error(
                contact_target.rotation_matrix, racket_rotation_current
            ),
            observed_ball_position_current - racket_position_current,
            (
                contact_target.position_m[:, :2] - center[None, :]
            ) * conditioning_scale,
            (
                contact_target.next_crossing_target_xy_m - center[None, :]
            ) * conditioning_scale,
        ],
        axis=-1,
    )


def contact_quality_pose_error(
    full_pose_error_rad: jax.Array,
    racket_up_cos: jax.Array,
    *, uses_normal_tilt: bool = False,
) -> jax.Array:
    """Score horizontal normal alignment without imposing an uncommanded yaw."""
    if not uses_normal_tilt:
        return jnp.asarray(full_pose_error_rad)
    return jnp.arccos(jnp.clip(jnp.asarray(racket_up_cos), -1.0, 1.0))


def contact_quality_scores(
    *,
    contact_surface_z_m: jax.Array,
    pose_error_rad: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    contact_offset_m: jax.Array,
    target_surface_z_m: float = 1.075,
    joint_floor_mix: float = 0.0,
    surface_z_sigma_m: float = 0.010,
) -> ContactQualityScores:
    """Compute the preregistered five-component contact quality reward."""

    if not math.isfinite(surface_z_sigma_m) or surface_z_sigma_m <= 0.0:
        raise ValueError("contact-height reward sigma must be positive and finite")
    values = (
        (jnp.asarray(contact_surface_z_m) - float(target_surface_z_m), surface_z_sigma_m),
        (jnp.asarray(pose_error_rad), jnp.deg2rad(2.0)),
        (jnp.asarray(racket_vxy_m_s), 0.050),
        (jnp.asarray(racket_omega_rad_s), 0.50),
        (jnp.asarray(contact_offset_m), 0.020),
    )
    components = jnp.stack(
        [jnp.exp(-0.5 * (value / sigma) ** 2) for value, sigma in values],
        axis=-1,
    )
    weights = jnp.asarray([0.30, 0.25, 0.20, 0.15, 0.10])
    mean = jnp.sum(components * weights, axis=-1)
    joint = jnp.min(components, axis=-1)
    mix = jnp.clip(jnp.asarray(joint_floor_mix, dtype=jnp.float32), 0.0, 1.0)
    return ContactQualityScores(
        components=components,
        mean=mean,
        joint=joint,
        combined=(1.0 - mix) * mean + mix * joint,
    )


def return_quality_scores(
    crossing_xy_m: jax.Array,
    apex_z_m: jax.Array,
    *,
    launch_contact_xy_m: jax.Array,
    workspace_center_xy: tuple[float, float] = (0.0, -0.49),
    target_apex_z_m: float | jax.Array = 1.40,
    apex_sigma_m: float = 0.020,
    max_lateral_displacement_per_bounce_m: float = 0.050,
) -> ReturnQualityScores:
    """Score the bounded lateral recovery trajectory and absolute apex."""

    sigma = float(apex_sigma_m)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("apex_sigma_m must be positive and finite")

    center = jnp.asarray(workspace_center_xy, dtype=jnp.float32)
    launch_xy = jnp.asarray(launch_contact_xy_m, dtype=jnp.float32)
    crossing_xy = jnp.asarray(crossing_xy_m, dtype=jnp.float32)
    desired_crossing_xy = bounded_next_crossing_target_xy(
        launch_xy,
        workspace_center_xy=workspace_center_xy,
        max_lateral_displacement_m=(
            max_lateral_displacement_per_bounce_m
        ),
    )
    trajectory_error = jnp.linalg.norm(
        crossing_xy - desired_crossing_xy, axis=-1
    )
    horizontal_displacement = jnp.linalg.norm(
        crossing_xy - launch_xy, axis=-1
    )
    center_progress = (
        jnp.linalg.norm(launch_xy - center[None, :], axis=-1)
        - jnp.linalg.norm(crossing_xy - center[None, :], axis=-1)
    )
    apex_error = jnp.asarray(apex_z_m) - jnp.asarray(
        target_apex_z_m, dtype=jnp.float32
    )
    trajectory_score = jnp.exp(-0.5 * (trajectory_error / 0.030) ** 2)
    apex_score = jnp.exp(-0.5 * (apex_error / sigma) ** 2)
    mean_score = 0.55 * trajectory_score + 0.45 * apex_score
    joint_score = jnp.minimum(trajectory_score, apex_score)
    return ReturnQualityScores(
        trajectory=trajectory_score,
        apex=apex_score,
        joint=joint_score,
        combined=0.80 * mean_score + 0.20 * joint_score,
        trajectory_error_m=trajectory_error,
        horizontal_displacement_m=horizontal_displacement,
        center_progress_m=center_progress,
    )


def return_quality_score(
    crossing_xy_m: jax.Array,
    apex_z_m: jax.Array,
    *,
    launch_contact_xy_m: jax.Array,
    workspace_center_xy: tuple[float, float] = (0.0, -0.49),
    target_apex_z_m: float = 1.40,
    apex_sigma_m: float = 0.020,
    max_lateral_displacement_per_bounce_m: float = 0.050,
) -> jax.Array:
    """Compatibility scalar for the bounded flight-quality score."""

    return return_quality_scores(
        crossing_xy_m,
        apex_z_m,
        launch_contact_xy_m=launch_contact_xy_m,
        workspace_center_xy=workspace_center_xy,
        target_apex_z_m=target_apex_z_m,
        apex_sigma_m=apex_sigma_m,
        max_lateral_displacement_per_bounce_m=(
            max_lateral_displacement_per_bounce_m
        ),
    ).combined


def policy_action_reward_input(
    raw_policy_action: jax.Array, *, expose_raw_action_for_reward: bool,
) -> jax.Array:
    """Keep raw overflow observable while the environment bounds execution."""
    if expose_raw_action_for_reward:
        return raw_policy_action
    return jnp.clip(raw_policy_action, -1.0, 1.0)
