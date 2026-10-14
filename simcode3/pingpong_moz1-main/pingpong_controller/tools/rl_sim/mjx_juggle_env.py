"""MJX/JAX juggling environment mirroring ``rl_juggle_env_random.JuggleEnv``.

The environment keeps the CPU task's 50-dimensional observation layout and
right-arm acceleration-command action interface while running batched MJX steps.
Each parallel environment carries its own MJX Model pytree so domain
randomization can change model fields such as mass, contact parameters,
damping, armature, gravity, and racket mount geometry per episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from camera_calibration import (
    D455_848_UNDISTORTED_BASE_POS,
    D455_848_UNDISTORTED_BASE_ROT,
    D455_848_UNDISTORTED_SIM_BASE_BODY,
    D455_848_UNDISTORTED_CX,
    D455_848_UNDISTORTED_CY,
    D455_848_UNDISTORTED_FX,
    D455_848_UNDISTORTED_FY,
    D455_848_UNDISTORTED_HEIGHT,
    D455_848_UNDISTORTED_HFOV_DEG,
    D455_848_UNDISTORTED_PIXEL_MARGIN,
    D455_848_UNDISTORTED_VFOV_DEG,
    D455_848_UNDISTORTED_WIDTH,
)
from delay_control import DEFAULT_DELAY_BIN_EDGES_MS
from mjx_smoke import _write_mjx_contact_only_xml
from rl_juggle_env_random import RIGHT_ARM_JOINTS, TARGET_DEGREES, _build_temp_xml_with_ball
from sim2real_bridger import constrained_compensation_step_jax


BASE_ACTS = ("Base-X", "Base-Y", "Base-Yaw")
BALL_OBS_FRAME_PIVOT_MODES = ("legacy_base_origin", "camera_center")

D455_REAL_RIGHT_ARM_RESET_DEGREES = (
    5.736,
    -44.399,
    30.683,
    97.142,
    49.323,
    -12.269,
    14.214,
)
D455_REAL_VIEW_X_BOUNDS_M = (-0.25, 0.25)
D455_REAL_VIEW_Y_BOUNDS_M = (-0.50, -0.25)
# Physical z measurements are reported as XML/world z minus the 0.100m base height,
# while MJX ball z metrics use the XML/world z directly.
D455_REAL_VIEW_Z_BOUNDS_M = (1.00, 1.47)
D455_REAL_VIEW_Z_IDEAL_M = (1.02, 1.42)
D455_REAL_VIEW_Y_TARGET_M = -0.35

LEGACY_STAGE4G_RIGHT_ARM_PD: dict[str, tuple[float, float]] = {
    "RightArm-0": (32000.0, 2000.0),
    "RightArm-1": (32000.0, 1800.0),
    "RightArm-2": (27000.0, 1500.0),
    "RightArm-3": (20000.0, 900.0),
    "RightArm-4": (13000.0, 500.0),
    "RightArm-5": (15000.0, 500.0),
    "RightArm-6": (10000.0, 350.0),
}

# Fitted only on the three bounded comparison inverse-MPC replays.  Kp and the
# five distal-joint Kv values stay identical to moz1_pd.xml; extra damping is
# limited to the two joints that produced the 200 Hz acceleration overshoot.
COMPARISON_SAFE_RIGHT_ARM_PD: dict[str, tuple[float, float]] = {
    "RightArm-0": (80000.0, 2500.0),
    "RightArm-1": (80000.0, 3240.0),
    "RightArm-2": (67500.0, 435.0),
    "RightArm-3": (50000.0, 337.5),
    "RightArm-4": (32500.0, 187.5),
    "RightArm-5": (37500.0, 212.5),
    "RightArm-6": (25000.0, 131.25),
}


def stopping_velocity_limit_jax(
    distance_rad: jax.Array,
    acc_limit_rad_s2: jax.Array,
    dt: float,
) -> jax.Array:
    """Exact discrete speed cap that leaves enough distance to stop.

    This is the JAX equivalent of
    :func:`pingpong_controller.safety_limiter.stopping_velocity_limit`.
    The semi-implicit command update is ``q_next = q + v_next * dt``.
    """

    distance = jnp.maximum(jnp.asarray(distance_rad), 0.0)
    acc_limit = jnp.asarray(acc_limit_rad_s2, dtype=distance.dtype)
    dt_value = jnp.asarray(float(dt), dtype=distance.dtype)
    accel_step = acc_limit * dt_value
    scaled_distance = 8.0 * distance / jnp.maximum(acc_limit * dt_value**2, 1e-12)
    brake_steps = jnp.floor(0.5 * (jnp.sqrt(1.0 + scaled_distance) - 1.0))
    return distance / (dt_value * (brake_steps + 1.0)) + 0.5 * accel_step * brake_steps


def project_safe_command_step_jax(
    target_rad: jax.Array,
    current_cmd_rad: jax.Array,
    current_vel_rad_s: jax.Array,
    pos_low_rad: jax.Array,
    pos_high_rad: jax.Array,
    vel_limit_rad_s: jax.Array,
    acc_limit_rad_s2: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Project a batched command onto a viable position/rate interval.

    Returns ``q_next, v_next, interval_low, interval_high, feasible``.
    ``feasible`` is per environment and should remain true when reset from a
    bounded position with zero velocity.  It is exposed as a metric instead
    of silently claiming safety if an externally modified state is invalid.
    """

    # Keep a sub-milliradian numerical buffer from the exact position limits.
    # Without it, repeated float32 integration can land on a boundary with a
    # tiny outward velocity and make the next discrete viability interval
    # empty even though the mathematical float64 trajectory is feasible.
    position_margin = jnp.asarray(2e-6, dtype=jnp.asarray(current_cmd_rad).dtype)
    safe_pos_low = pos_low_rad + position_margin
    safe_pos_high = pos_high_rad - position_margin
    target = jnp.clip(target_rad, safe_pos_low, safe_pos_high)
    braking_margin = jnp.asarray(2e-6, dtype=jnp.asarray(current_cmd_rad).dtype)
    distance_low = jnp.maximum(
        current_cmd_rad - safe_pos_low - braking_margin,
        0.0,
    )
    distance_high = jnp.maximum(
        safe_pos_high - current_cmd_rad - braking_margin,
        0.0,
    )
    # A 0.02% internal derating absorbs float32 subtraction/accumulation error
    # so finite-difference outputs remain inside the user-facing hard limits.
    effective_acc_limit = acc_limit_rad_s2 * 0.9998
    effective_vel_limit = vel_limit_rad_s * 0.99999
    stop_low = stopping_velocity_limit_jax(distance_low, effective_acc_limit, dt)
    stop_high = stopping_velocity_limit_jax(distance_high, effective_acc_limit, dt)
    accel_step = effective_acc_limit * float(dt)

    interval_low = jnp.maximum(
        jnp.maximum(-effective_vel_limit, current_vel_rad_s - accel_step),
        jnp.maximum((safe_pos_low - current_cmd_rad) / float(dt), -stop_low),
    )
    interval_high = jnp.minimum(
        jnp.minimum(effective_vel_limit, current_vel_rad_s + accel_step),
        jnp.minimum((safe_pos_high - current_cmd_rad) / float(dt), stop_high),
    )
    tolerance = 5e-5
    feasible_joint = interval_low <= interval_high + tolerance
    feasible = jnp.all(feasible_joint, axis=-1)
    # Valid limiter states have a non-empty interval.  Collapsing only a
    # numerically inverted interval keeps JAX execution defined while the
    # metric makes any true invariant failure visible to validation.
    interval_mid = 0.5 * (interval_low + interval_high)
    safe_low = jnp.minimum(interval_low, interval_mid)
    safe_high = jnp.maximum(interval_high, interval_mid)
    desired_vel = (target - current_cmd_rad) / float(dt)
    next_vel = jnp.minimum(jnp.maximum(desired_vel, safe_low), safe_high)
    next_q = current_cmd_rad + next_vel * float(dt)
    return next_q, next_vel, interval_low, interval_high, feasible


def project_target_tracking_command_step_jax(
    target_rad: jax.Array,
    current_cmd_rad: jax.Array,
    current_vel_rad_s: jax.Array,
    pos_low_rad: jax.Array,
    pos_high_rad: jax.Array,
    vel_limit_rad_s: jax.Array,
    acc_limit_rad_s2: jax.Array,
    dt: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Plan one viable trapezoidal-profile step toward a position target.

    The feasible interval enforces joint position, velocity and acceleration
    limits.  The additional target-distance stopping speed makes the planner
    brake for the requested target instead of overshooting and chasing a
    rapidly changing inverse-MPC/FOPDT position command.
    """

    _, _, interval_low, interval_high, feasible = project_safe_command_step_jax(
        target_rad,
        current_cmd_rad,
        current_vel_rad_s,
        pos_low_rad,
        pos_high_rad,
        vel_limit_rad_s,
        acc_limit_rad_s2,
        dt,
    )
    target = jnp.clip(target_rad, pos_low_rad, pos_high_rad)
    position_error = target - current_cmd_rad
    # Match the float32 safety margin used by the viable interval.
    effective_acc_limit = acc_limit_rad_s2 * 0.9998
    stop_speed = stopping_velocity_limit_jax(
        jnp.abs(position_error),
        effective_acc_limit,
        dt,
    )
    desired_vel = jnp.sign(position_error) * jnp.minimum(
        jnp.abs(position_error) / float(dt),
        stop_speed,
    )
    interval_mid = 0.5 * (interval_low + interval_high)
    safe_low = jnp.minimum(interval_low, interval_mid)
    safe_high = jnp.maximum(interval_high, interval_mid)
    next_vel = jnp.minimum(jnp.maximum(desired_vel, safe_low), safe_high)
    next_q = current_cmd_rad + next_vel * float(dt)
    return next_q, next_vel, interval_low, interval_high, feasible


def project_damped_target_tracking_command_step_jax(
    target_rad: jax.Array,
    current_cmd_rad: jax.Array,
    current_vel_rad_s: jax.Array,
    pos_low_rad: jax.Array,
    pos_high_rad: jax.Array,
    vel_limit_rad_s: jax.Array,
    acc_limit_rad_s2: jax.Array,
    dt: float,
    natural_frequency_hz: float,
    damping_ratio: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Plan a bounded critically damped step toward the current target."""

    _, _, interval_low, interval_high, feasible = project_safe_command_step_jax(
        target_rad,
        current_cmd_rad,
        current_vel_rad_s,
        pos_low_rad,
        pos_high_rad,
        vel_limit_rad_s,
        acc_limit_rad_s2,
        dt,
    )
    target = jnp.clip(target_rad, pos_low_rad, pos_high_rad)
    omega = 2.0 * jnp.pi * float(natural_frequency_hz)
    desired_acc = (
        omega * omega * (target - current_cmd_rad)
        - 2.0 * float(damping_ratio) * omega * current_vel_rad_s
    )
    desired_vel = current_vel_rad_s + desired_acc * float(dt)
    interval_mid = 0.5 * (interval_low + interval_high)
    safe_low = jnp.minimum(interval_low, interval_mid)
    safe_high = jnp.maximum(interval_high, interval_mid)
    next_vel = jnp.minimum(jnp.maximum(desired_vel, safe_low), safe_high)
    next_q = current_cmd_rad + next_vel * float(dt)
    return next_q, next_vel, interval_low, interval_high, feasible


def _apply_right_arm_pd_profile(xml_path: Path, profile: str) -> Path:
    if profile in ("", "xml", "current"):
        return xml_path
    profiles = {
        "legacy_stage4g": LEGACY_STAGE4G_RIGHT_ARM_PD,
        "comparison_safe_v1": COMPARISON_SAFE_RIGHT_ARM_PD,
    }
    if profile not in profiles:
        raise ValueError(f"unknown right_arm_pd_profile={profile!r}")
    pd_values = profiles[profile]
    tree = ET.parse(xml_path)
    root = tree.getroot()
    patched = 0
    for elem in root.findall(".//actuator/position"):
        name = elem.attrib.get("name", "")
        if name not in pd_values:
            continue
        kp, kv = pd_values[name]
        elem.set("kp", f"{kp:g}")
        elem.set("kv", f"{kv:g}")
        patched += 1
    if patched != len(pd_values):
        raise ValueError(
            f"right_arm_pd_profile={profile!r} patched {patched} actuators, "
            f"expected {len(pd_values)}"
        )
    tree.write(xml_path, encoding="unicode")
    return xml_path


@dataclass(frozen=True)
class MjxJuggleConfig:
    horizon_sec: float = 6.0
    frame_skip: int = 5
    right_arm_pd_profile: str = "xml"
    action_scale_arm_rad: float = 0.03
    action_scale_base_xy: float = 0.020
    action_scale_base_yaw: float = 0.030
    action_acc_scale: float = 1.5
    ball_launch_height: float = 0.30
    ball_spawn_cube_size: float = 0.10
    ball_spawn_xy_jitter: float = 0.0
    ball_spawn_z_jitter: float = 0.0
    episode_target_x_range_m: tuple[float, float] = (0.0, 0.0)
    episode_target_y_range_m: tuple[float, float] = (0.0, 0.0)
    episode_racket_anchor_z_range_m: tuple[float, float] = (0.0, 0.0)
    right_arm_reset_degrees: tuple[float, ...] = D455_REAL_RIGHT_ARM_RESET_DEGREES
    ball_init_vxy_max: float = 0.0
    ball_init_vz: float = -0.28
    ball_init_vz_jitter: float = 0.0
    ball_reset_mode: str = "anchor_drop"
    falling_reset_time_to_contact_range_s: tuple[float, float] = (0.12, 0.22)
    falling_reset_apex_height_range_m: tuple[float, float] = (0.20, 0.32)
    falling_reset_vxy_max: float = 0.0
    falling_reset_contact_xy_jitter: float = 0.0
    falling_reset_contact_rel_height: float = -1.0
    falling_reset_min_downward_speed: float = 0.12
    racket_launch_surface_gap_range_m: tuple[float, float] = (0.005, 0.010)
    racket_launch_xy_jitter: float = 0.004
    racket_launch_vxy_max: float = 0.003
    racket_launch_vnormal_max: float = 0.003
    racket_launch_edge_margin: float = 0.005
    ball_obs_rate_hz: float = 50.0
    ball_obs_fractional_rate: bool = False
    ball_obs_pos_noise_std: float = 0.003
    ball_obs_vel_noise_std: float = 0.03
    total_training_steps: int = 10_000_000
    ball_obs_noise_warmup_ratio: float = 0.10
    ball_obs_noise_ramp_ratio: float = 0.20
    target_height: float = 0.34
    # Optional camera-calibrated absolute apex target used only by training
    # rewards.  Prediction remains causal (current position/velocity/gravity).
    # None preserves the historical episode-anchor-relative target.
    hit_apex_target_abs_z: float | None = None
    posture_weight: float = 0.02
    base_pose_weight: float = 0.0
    # Reject policies that exploit the uncommanded base z/roll/pitch DOFs.
    terminate_on_base_stability: bool = True
    base_z_deviation_limit_m: float = 0.03
    base_roll_pitch_limit_rad: float = 0.0872664626  # 5 degrees
    torque_penalty_weight: float = 0.00005
    post_hit_survival_reward_weight: float = 1.4
    post_hit_ball_xy_sigma: float = 0.12
    post_hit_ball_vxy_penalty_weight: float = 0.18
    descending_intercept_reward_weight: float = 1.6
    descending_intercept_sigma: float = 0.10
    pre_hit_intercept_reward_weight: float = 0.0
    pre_hit_intercept_sigma: float = 0.08
    pre_hit_intercept_time_max: float = 0.55
    pre_hit_intercept_penalty_weight: float = 0.0
    pre_hit_intercept_penalty_sigma: float = 0.20
    pre_hit_intercept_penalty_radius: float = 0.025
    pre_hit_intercept_penalty_time_max: float = 0.85
    non_racket_ball_contact_penalty_weight: float = 1.5
    failed_hit_penalty_weight: float = 1.0
    sticky_contact_penalty_growth: float = 0.6
    hit_reward_base: float = 2.5
    hit_reward_combo: float = 1.2
    rel_height_center: float = 0.18
    rel_height_sigma: float = 0.06
    rel_height_bonus_weight: float = 0.45
    racket_xy_gauss_sigma: float = 0.041
    racket_xy_gauss_reward_weight: float = 0.50
    racket_xy_gauss_penalty_weight: float = 0.60
    racket_chest_xy_penalty_weight: float = 1.0
    racket_chest_z_penalty_weight: float = 0.8
    ball_anchor_xy_penalty_weight: float = 0.7
    ball_base_x_penalty_weight: float = 0.0
    ball_base_x_soft_limit: float = 0.20
    ball_base_vxy_penalty_weight: float = 0.0
    ball_vxy_penalty_weight: float = 0.40
    apex_soft_limit_margin: float = 0.04
    apex_soft_penalty_weight: float = 5.0
    ball_xy_soft_limit_radius: float = 0.14
    ball_xy_soft_penalty_weight: float = 3.0
    ball_low_termination_z_m: float = D455_REAL_VIEW_Z_BOUNDS_M[0]
    ball_high_termination_z_m: float = 1.90
    terminate_on_ball_view_bounds: bool = False
    terminate_on_ball_view_x_bounds: bool = True
    terminate_on_ball_view_y_bounds: bool = True
    terminate_on_ball_view_z_low: bool = True
    terminate_on_ball_view_z_high: bool = True
    ball_view_x_bounds_m: tuple[float, float] = D455_REAL_VIEW_X_BOUNDS_M
    ball_view_y_bounds_m: tuple[float, float] = D455_REAL_VIEW_Y_BOUNDS_M
    ball_view_z_bounds_m: tuple[float, float] = D455_REAL_VIEW_Z_BOUNDS_M
    ball_view_z_ideal_m: tuple[float, float] = D455_REAL_VIEW_Z_IDEAL_M
    ball_view_x_target_m: float = 0.0
    ball_view_y_target_m: float = D455_REAL_VIEW_Y_TARGET_M
    ball_view_x_sigma_m: float = 0.08
    ball_view_y_sigma_m: float = 0.10
    ball_view_z_sigma_m: float = 0.08
    ball_view_vxy_soft_limit_m_s: float = 1.0
    ball_view_xy_center_penalty_weight: float = 0.0
    ball_view_z_ideal_penalty_weight: float = 0.0
    ball_view_bounds_penalty_weight: float = 0.0
    ball_view_out_of_bounds_penalty_weight: float = 0.0
    ball_view_z_not_ideal_penalty_weight: float = 0.0
    ball_view_vxy_excess_penalty_weight: float = 0.0
    racket_z_band_down: float = 0.00
    racket_z_band_up: float = 0.20
    racket_z_soft_penalty_weight: float = 1.2
    racket_up_drift_penalty_weight: float = 0.3
    racket_up_drift_vel_thresh: float = 0.02
    racket_flatness_penalty_weight: float = 0.0
    racket_flatness_target_cos: float = 0.970
    racket_flatness_sigma: float = 0.060
    racket_z_hard_limit_down: float = 0.12
    racket_z_hard_limit_up: float = 0.24
    terminate_on_racket_z_limit: bool = True
    racket_z_limit_termination_penalty_base: float = 0.0
    racket_z_limit_termination_penalty_per_hit: float = 0.0
    # Keep workspace escape separate from the z-limit penalty so existing
    # profiles retain their historical reward semantics.  A positive base is
    # required by profiles whose plant can otherwise learn to collect one hit
    # and deliberately terminate via ``racket_too_far_from_anchor``.
    racket_anchor_termination_penalty_base: float = 0.0
    racket_anchor_termination_penalty_per_hit: float = 0.0
    action_penalty_weight: float = 0.003
    action_delta_penalty_weight: float = 0.001
    # Penalize only the part of the Gaussian policy sample outside the
    # executable normalized action range. The command is still hard-clipped.
    action_clip_excess_penalty_weight: float = 0.0
    termination_miss_penalty_base: float = 2.5
    termination_miss_penalty_per_hit: float = 0.8
    termination_miss_penalty_requires_hit: bool = True
    termination_no_hit_miss_early_penalty: float = 0.0
    hit_rearm_no_contact_steps: int = 2
    hit_rearm_distance: float = 0.035
    stick_contact_penalty_weight: float = 0.60
    stick_rel_speed_thresh: float = 0.25
    stick_rel_dist_thresh: float = 0.040
    stick_min_contact_steps: int = 4
    hit_confirm_rel_height: float = 0.06
    hit_confirm_abs_height: float = 1.00
    hit_confirm_max_steps: int = 70
    hit_confirm_use_spawn_cube_band: bool = False
    hit_confirm_spawn_band_margin: float = 0.0
    hit_center_local_sigma: float = 0.035
    hit_center_sigma: float = 0.08
    hit_flatness_target_cos: float = 0.96
    hit_flatness_sigma: float = 0.08
    center_flat_hit_reward_weight: float = 1.8
    contact_flatness_penalty_weight: float = 0.45
    hit_height_center: float = 0.52
    hit_height_tolerance: float = 0.06
    hit_height_penalty_weight: float = 10.0
    hit_vxy_soft_limit_m_s: float = 0.35
    hit_vxy_penalty_weight: float = 0.0
    hit_apex_view_center_penalty_weight: float = 0.0
    hit_apex_view_center_sigma_m: float = 0.12
    hit_next_contact_anchor_penalty_weight: float = 0.0
    hit_next_contact_anchor_sigma_m: float = 0.10
    first_hit_apex_reward_weight: float = 0.0
    first_hit_apex_sigma: float = 0.055
    low_hit_apex_margin: float = 0.06
    low_hit_penalty_weight: float = 10.0
    domain_randomization: bool = True
    dr_randomize_ball: bool = True
    dr_randomize_contact: bool = True
    dr_randomize_actuator: bool = True
    dr_randomize_latency: bool = True
    dr_ball_mass_range: tuple[float, float] = (0.0024, 0.0030)
    dr_ball_friction_range: tuple[float, float] = (0.12, 0.35)
    dr_racket_friction_range: tuple[float, float] = (0.25, 0.55)
    dr_ball_solref_time_range: tuple[float, float] = (0.002, 0.006)
    dr_ball_solref_damping_range: tuple[float, float] = (0.70, 0.95)
    dr_gravity_z_range: tuple[float, float] = (-9.90, -9.70)
    dr_action_scale_mult_range: tuple[float, float] = (0.85, 1.15)
    dr_armature_mult_range: tuple[float, float] = (0.80, 1.20)
    dr_damping_mult_range: tuple[float, float] = (0.70, 1.30)
    dr_randomize_pd: bool = False
    dr_pd_kp_mult_range: tuple[float, float] = (1.0, 1.0)
    dr_pd_kv_mult_range: tuple[float, float] = (1.0, 1.0)
    dr_pd_per_joint: bool = True
    dr_obs_latency_steps_range: tuple[int, int] = (0, 2)
    dr_action_latency_steps_range: tuple[int, int] = (0, 2)
    actuator_cmd_filter: bool = False
    actuator_cmd_tau: float = 0.0
    actuator_cmd_gain: float = 1.0
    dr_randomize_actuator_cmd_filter: bool = False
    dr_actuator_cmd_tau_range: tuple[float, float] = (0.0, 0.0)
    dr_actuator_cmd_gain_range: tuple[float, float] = (1.0, 1.0)
    actuator_compensation_mode: str = "none"
    actuator_lead_compensation: bool = False
    actuator_lead_beta: float = 0.0
    actuator_lead_delay_scale: float = 1.0
    actuator_lead_tau_scale: float = 1.0
    actuator_lead_max_delta_rad: float = 0.0
    actuator_inverse_beta: float = 1.0
    actuator_inverse_delay_scale: float = 1.0
    actuator_inverse_tau_scale: float = 1.0
    actuator_inverse_max_delta_rad: float = 0.0
    actuator_mpc_beta: float = 1.0
    actuator_mpc_delay_scale: float = 1.0
    actuator_mpc_tau_scale: float = 1.0
    actuator_mpc_horizon_steps: int = 4
    actuator_mpc_tracking_weight: float = 1.0
    actuator_mpc_nominal_weight: float = 0.25
    actuator_mpc_delta_weight: float = 0.08
    actuator_mpc_max_delta_rad: float = 0.0
    actuator_mpc_command_dynamics_constraint: bool = False
    actuator_mpc_command_velocity_weight: float = 0.0
    actuator_mpc_command_acceleration_weight: float = 0.0
    actuator_mpc_command_velocity_scale: float = 1.0
    actuator_mpc_command_acceleration_scale: float = 1.0
    actuator_mpc_feedback_source: str = "applied"  # "applied" or "actual"
    actuator_bridger_natural_frequency_hz: float = 6.0
    actuator_bridger_damping_ratio: float = 1.0
    actuator_bridger_jerk_limit_deg_s3: tuple[float, ...] = (
        175000.0,
        175000.0,
        175000.0,
        175000.0,
        175000.0,
        175000.0,
        175000.0,
    )
    camera_visibility_mode: str = "off"
    virtual_camera_pose_mode: str = "base_extrinsic"  # "base_extrinsic" or legacy "body_mount"
    virtual_camera_base_body_name: str = D455_848_UNDISTORTED_SIM_BASE_BODY
    virtual_camera_require_base_body: bool = False
    virtual_camera_body_name: str = "head22"
    virtual_camera_mount_pos: tuple[float, float, float] = (0.0, -0.068, 0.062)
    virtual_camera_mount_quat: tuple[float, float, float, float] = (0.707107, 0.0, 0.0, -0.707107)
    virtual_camera_optical_pos: tuple[float, float, float] = (0.048, 0.0, 0.0)
    virtual_camera_base_pos: tuple[float, float, float] = D455_848_UNDISTORTED_BASE_POS
    virtual_camera_base_rot: tuple[float, ...] = D455_848_UNDISTORTED_BASE_ROT
    camera_image_width: int = D455_848_UNDISTORTED_WIDTH
    camera_image_height: int = D455_848_UNDISTORTED_HEIGHT
    camera_fx: float = D455_848_UNDISTORTED_FX
    camera_fy: float = D455_848_UNDISTORTED_FY
    camera_cx: float = D455_848_UNDISTORTED_CX
    camera_cy: float = D455_848_UNDISTORTED_CY
    camera_hfov_deg: float = D455_848_UNDISTORTED_HFOV_DEG
    camera_vfov_deg: float = D455_848_UNDISTORTED_VFOV_DEG
    camera_min_depth: float = 0.15
    camera_max_depth: float = 2.50
    camera_pixel_margin: float = D455_848_UNDISTORTED_PIXEL_MARGIN
    camera_center_weight: float = 0.0
    camera_visibility_penalty_weight: float = 0.0
    camera_depth_penalty_weight: float = 0.0
    camera_box_penalty_weight: float = 0.0
    camera_visible_penalty_weight: float = 0.0
    camera_top_margin_penalty_weight: float = 0.0
    camera_dense_penalty_clip: float = 20.0
    hit_camera_reward_weight: float = 0.0
    hit_camera_out_of_band_penalty_weight: float = 0.0
    hit_camera_target_v_frac: float = 0.65
    hit_camera_v_sigma_frac: float = 0.15
    hit_camera_lower_band_frac: tuple[float, float] = (0.50, 0.82)
    camera_box_half_width: float = 0.35
    camera_box_half_height: float = 0.35
    camera_box_depth_min: float = 0.20
    camera_box_depth_max: float = 1.50
    arm_action_limiter: bool = False
    # Final command projection after inverse-MPC/lead compensation and before
    # the actuator delay/filter.  This mirrors the real publish-time safety
    # layer and keeps old checkpoints bit-compatible unless explicitly enabled.
    arm_post_compensation_limiter: bool = False
    # Optional plant-side trajectory limiter after pure delay/FOPDT.  This
    # models the drive/servo motion-profile limits observed on hardware while
    # leaving the inverse-MPC command and parameters unchanged.
    arm_servo_target_limiter: bool = False
    # Target-aware drive-side trajectory planner used by constrained inverse
    # MPC.  It acts on the delay/FOPDT output and sends position only to the
    # unchanged XML position PD.  Scales reserve planner headroom without
    # changing the physical/user-facing joint limits.
    arm_servo_target_tracking_planner: bool = False
    arm_servo_target_velocity_scale: float = 1.0
    arm_servo_target_acceleration_scale: float = 1.0
    # Drive-level guard on the simulated joint state after every MJX substep.
    # It models a hardware motion-profile envelope while matching the
    # finite-difference qdot/qddot reported by validation/action plots.
    arm_actual_state_limiter: bool = False
    # Target-aware actual-state drive governor.  Unlike the legacy greedy
    # projection, this brakes for the current delay/FOPDT position target and
    # limits acceleration changes (jerk), preventing the XML PD and hard
    # projection from creating a bang-bang chatter loop.  It requires the
    # actual-state limiter and leaves upstream inverse-MPC/FOPDT/Kp/Kv intact.
    arm_actual_target_tracking_governor: bool = False
    arm_actual_governor_natural_frequency_hz: float = 8.0
    arm_actual_governor_damping_ratio: float = 1.0
    arm_actual_jerk_limit_deg_s3: tuple[float, ...] = (
        175000.0,
        175000.0,
        175000.0,
        175000.0,
        175000.0,
        175000.0,
        175000.0,
    )
    arm_vel_limit_deg_s: tuple[float, ...] = (210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0)
    arm_acc_limit_deg_s2: tuple[float, ...] = (1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0)
    arm_vel_limit_penalty_weight: float = 0.0
    arm_acc_limit_penalty_weight: float = 0.002
    arm_limiter_penalty_weight: float = 0.0
    dr_randomize_racket_mount: bool = False
    dr_racket_pos_offset_m: float = 0.0
    dr_racket_rot_offset_rad: float = 0.0
    dr_racket_radius_offset_m: float = 0.0
    hit_cadence_reward_weight: float = 0.0
    hit_cadence_target_interval: float = 0.65
    hit_cadence_sigma: float = 0.18
    hit_min_interval_penalty_weight: float = 0.0
    hit_min_interval: float = 0.40
    hit_min_count_interval: float = 0.0
    fast_hit_penalty_weight: float = 0.0
    hit_reward_cap_mode: str = "off"
    hit_reward_count_cap: int = 0
    hit_combo_count_cap: int = 14
    hit_reward_cap_target_interval: float = 0.65
    ball_obs_dropout_prob: float = 0.0
    ball_obs_dropout_max_steps: int = 1
    ball_obs_dropout_burst_prob: float = 0.0
    ball_obs_dropout_burst_max_steps: int = 1
    ball_obs_age_clip: float = 0.20
    ball_obs_age_tracks_stale: bool = False
    ball_obs_dropout_on_refresh_only: bool = False
    ball_obs_require_camera_visible: bool = False
    ball_obs_camera_missing_prob: float = 1.0
    ball_obs_reset_respects_camera_visibility: bool = False
    ball_obs_require_view_bounds: bool = False
    ball_obs_view_bounds_missing_prob: float = 1.0
    ball_obs_missing_episode_coherent_prob: float = 0.0
    # Independent upper-height visibility boundary.  Unlike
    # ball_obs_require_view_bounds this does not hide observations at the
    # calibrated x/y or lower-z margins.
    ball_obs_require_view_z_high: bool = False
    ball_obs_view_z_high_missing_range_m: tuple[float, float] = (0.0, 0.0)
    ball_obs_nominal_pos_bias_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ball_obs_nominal_vel_bias_base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ball_obs_frame_pivot_mode: str = "legacy_base_origin"
    dr_randomize_ball_obs_frame: bool = False
    dr_ball_obs_pos_bias_base_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dr_ball_obs_rot_bias_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dr_ball_obs_vel_bias_base_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dr_ball_obs_scale_range: tuple[float, float] = (1.0, 1.0)
    high_latency_obs: bool = False
    high_latency_history_frames: int = 3
    high_latency_obs_history_frames: int | None = None
    high_latency_action_history_frames: int | None = None
    high_latency_prediction_time_clip: float = 0.30
    high_latency_prediction_include_obs_latency: bool = True
    high_latency_prediction_include_ball_age: bool = True
    high_latency_prediction_include_actuator_tau: bool = True
    enable_delay_conditioning: bool = False
    delay_min_ms: float = 0.0
    delay_max_ms: float = 150.0
    delay_bin_edges_ms: tuple[float, ...] = DEFAULT_DELAY_BIN_EDGES_MS
    delay_jitter_ms: float = 0.0
    delay_sampling_mode: str = "balanced_bins"
    # Preserve the nominal hardware-delay command history used by the 67D
    # policy features, while applying the latest command to the simulated
    # servo immediately.  This models an upstream real compensator whose
    # closed-loop residual delay is approximately zero without changing the
    # deployed policy's 67D observation contract.
    actuator_delay_observation_only: bool = False
    include_tau_act_norm: bool = False
    include_command_state: bool = False
    include_phase_features: bool = False
    include_active_command_error: bool = False
    action_filter_tau_ms: float = 0.0
    action_jerk_limit: float = 0.0
    action_acc_limit: float = 1.0
    enable_anti_windup: bool = False
    anti_windup_error_threshold: float = 0.5
    anti_windup_min_scale: float = 0.2
    command_tracking_error_penalty_weight: float = 0.0
    delay_action_jerk_penalty_weight: float = 0.0
    command_buffer_extra_steps: int = 4
    use_delay_embedding: bool = False
    delay_embedding_dim: int = 0
    use_delay_bin_value_heads: bool = False  # TODO: PPO critic still uses one value head.
    asymmetric_critic: bool = False
    critic_command_history_steps: int = 4
    contact_height_offset: float = 0.0
    max_contact_time: float = 0.50
    lost_ball_timeout_ms: float = 150.0


class EnvState(NamedTuple):
    model: object
    data: object
    rng: jax.Array
    step_count: jax.Array
    racket_anchor: jax.Array
    chest_target_offset: jax.Array
    reset_ball_pos: jax.Array
    reset_ball_vel: jax.Array
    reset_target_offset: jax.Array
    reset_disturbance_strength: jax.Array
    reset_ball_surface_gap: jax.Array
    reset_ball_racket_center_offset: jax.Array
    arm_cmd_q: jax.Array
    arm_cmd_qvel: jax.Array
    arm_q_ref_latest: jax.Array
    arm_q_ref_active: jax.Array
    arm_actuator_q_ref_latest: jax.Array
    arm_actuator_q_ref_active: jax.Array
    arm_safe_q_ref_latest: jax.Array
    arm_safe_qvel: jax.Array
    arm_safe_qacc: jax.Array
    reset_ball_obs_missing: jax.Array
    ball_obs_missing_episode_coherent_enabled: jax.Array
    ball_obs_camera_missing_enabled: jax.Array
    ball_obs_view_bounds_missing_enabled: jax.Array
    arm_applied_q: jax.Array
    arm_applied_qvel: jax.Array
    prev_action: jax.Array
    prev_arm_qvel: jax.Array
    prev_ball_pos: jax.Array
    prev_racket_pos: jax.Array
    prev_contact: jax.Array
    hit_armed: jax.Array
    no_contact_steps: jax.Array
    contact_hold_steps: jax.Array
    pending_hit: jax.Array
    pending_hit_steps: jax.Array
    hit_count: jax.Array
    action_buffer: jax.Array
    action_latency_steps: jax.Array
    command_buffer: jax.Array
    actuator_command_buffer: jax.Array
    tau_act_episode: jax.Array
    tau_act: jax.Array
    delay_steps: jax.Array
    delay_bin_id: jax.Array
    anti_windup_scale: jax.Array
    obs_buffer: jax.Array
    pending_hit_camera_visible: jax.Array
    pending_hit_camera_in_lower_band: jax.Array
    pending_hit_camera_in_margin: jax.Array
    pending_hit_camera_v_frac: jax.Array
    obs_latency_steps: jax.Array
    obs_history: jax.Array
    action_history: jax.Array
    cached_ball_obs_pos: jax.Array
    cached_ball_obs_vel: jax.Array
    last_ball_obs_step: jax.Array
    ball_obs_valid_pos: jax.Array
    ball_obs_valid_vel: jax.Array
    ball_obs_age_seconds: jax.Array
    ball_obs_missing_since_sample: jax.Array
    ball_obs_dropout_remaining: jax.Array
    ball_obs_dropout_steps_total: jax.Array
    ball_obs_burst_count: jax.Array
    total_env_steps: jax.Array
    action_scale_mult: jax.Array
    actuator_cmd_tau: jax.Array
    actuator_cmd_gain: jax.Array
    dr_gravity_z: jax.Array
    dr_ball_mass: jax.Array
    dr_ball_friction: jax.Array
    dr_racket_friction: jax.Array
    dr_ball_solref_time: jax.Array
    dr_ball_solref_damping: jax.Array
    dr_damping_mult: jax.Array
    dr_armature_mult: jax.Array
    dr_pd_kp_mult: jax.Array
    dr_pd_kv_mult: jax.Array
    last_hit_time: jax.Array
    last_counted_hit_time: jax.Array
    last_count_gate_hit_time: jax.Array
    confirmed_hit_count: jax.Array
    ignored_fast_hit_count: jax.Array
    rewarded_hit_count: jax.Array
    unrewarded_extra_hit_count: jax.Array
    dr_racket_pos_offset: jax.Array
    dr_racket_rot_offset: jax.Array
    dr_racket_radius_offset: jax.Array
    ball_obs_pos_bias_base: jax.Array
    ball_obs_rot_bias_rpy: jax.Array
    ball_obs_vel_bias_base: jax.Array
    ball_obs_scale: jax.Array
    ball_obs_view_z_high_m: jax.Array


def _deg_to_rad_map(deg_map: dict[str, float]) -> dict[str, float]:
    return {k: float(np.deg2rad(v)) for k, v in deg_map.items()}


def _quat_wxyz_to_mat_np(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.asarray(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float32,
    )


def _quat_mul_wxyz_jax(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = jnp.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=jnp.float32,
    )
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-8)


def _euler_xyz_to_quat_wxyz_jax(euler_xyz: jax.Array) -> jax.Array:
    roll, pitch, yaw = euler_xyz
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    q = jnp.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=jnp.float32,
    )
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-8)


def _euler_xyz_to_mat_jax(euler_xyz: jax.Array) -> jax.Array:
    roll, pitch, yaw = euler_xyz
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    return jnp.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=jnp.float32,
    )


def _apply_ball_obs_frame_transform(
    bpos_base: jax.Array,
    bvel_base: jax.Array,
    rot: jax.Array,
    scale: jax.Array,
    pos_bias_base: jax.Array,
    vel_bias_base: jax.Array,
    pivot_base: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Apply one batched observation-frame perturbation.

    ``pivot_base=None`` preserves the historical base-origin transform exactly.
    A supplied pivot rotates/scales positions around that point, while velocity
    remains a free vector and therefore never receives a positional pivot.
    """

    scale_column = scale[:, None]
    if pivot_base is None:
        bpos_obs = (
            scale_column * jnp.einsum("nij,nj->ni", rot, bpos_base)
            + pos_bias_base
        )
    else:
        bpos_obs = (
            pivot_base
            + scale_column
            * jnp.einsum("nij,nj->ni", rot, bpos_base - pivot_base)
            + pos_bias_base
        )
    bvel_obs = (
        scale_column * jnp.einsum("nij,nj->ni", rot, bvel_base)
        + vel_bias_base
    )
    return bpos_obs, bvel_obs


def _batch_tree(tree, n_envs: int):
    def batch_leaf(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jnp.broadcast_to(x, (n_envs,) + tuple(x.shape))
        return x

    return jax.tree_util.tree_map(batch_leaf, tree)


class MjxJuggleEnv:
    base_obs_dim = 50
    obs_dim = 50
    act_dim = 7

    def __init__(self, xml_path: str | Path, n_envs: int, cfg: MjxJuggleConfig = MjxJuggleConfig()) -> None:
        self.xml_path = Path(xml_path).resolve()
        self.n_envs = int(n_envs)
        self.cfg = cfg
        cfg_values = getattr(cfg, "__dict__", {})
        self.vc_pose_mode = str(cfg_values.get("virtual_camera_pose_mode", "body_mount"))
        self.ball_obs_frame_pivot_mode = str(
            cfg_values.get("ball_obs_frame_pivot_mode", "legacy_base_origin")
        )
        if self.ball_obs_frame_pivot_mode not in BALL_OBS_FRAME_PIVOT_MODES:
            raise ValueError(
                "unknown ball_obs_frame_pivot_mode="
                f"{self.ball_obs_frame_pivot_mode!r}; expected one of "
                f"{list(BALL_OBS_FRAME_PIVOT_MODES)}"
            )
        if (
            self.ball_obs_frame_pivot_mode == "camera_center"
            and self.vc_pose_mode not in {"base_extrinsic", "body_mount"}
        ):
            raise ValueError(
                "camera-centered ball observation frame requires "
                "virtual_camera_pose_mode='base_extrinsic' or 'body_mount', got "
                f"{self.vc_pose_mode!r}"
            )
        self.ball_reset_mode = str(cfg.ball_reset_mode)
        valid_ball_reset_modes = {"anchor_drop", "falling_contact", "racket_launch"}
        if self.ball_reset_mode not in valid_ball_reset_modes:
            raise ValueError(
                f"unknown ball_reset_mode={self.ball_reset_mode!r}; "
                f"expected one of {sorted(valid_ball_reset_modes)}"
            )
        if self.ball_reset_mode == "racket_launch":
            gap_lo, gap_hi = [float(v) for v in cfg.racket_launch_surface_gap_range_m]
            if min(gap_lo, gap_hi) < 0.0:
                raise ValueError("racket_launch_surface_gap_range_m must be non-negative")
            if float(cfg.racket_launch_xy_jitter) < 0.0:
                raise ValueError("racket_launch_xy_jitter must be non-negative")
            if float(cfg.racket_launch_vxy_max) < 0.0 or float(cfg.racket_launch_vnormal_max) < 0.0:
                raise ValueError("racket-launch velocity limits must be non-negative")
            if float(cfg.racket_launch_edge_margin) < 0.0:
                raise ValueError("racket_launch_edge_margin must be non-negative")
        if bool(cfg.use_delay_bin_value_heads):
            raise NotImplementedError(
                "use_delay_bin_value_heads is reserved for a future PPO critic "
                "with per-delay-bin value heads; keep it False for now."
            )
        self.high_latency_obs = bool(cfg.high_latency_obs)
        self.delay_conditioning = bool(cfg.enable_delay_conditioning)
        self.actuator_delay_observation_only = bool(
            cfg.actuator_delay_observation_only
        )
        if self.actuator_delay_observation_only and not self.delay_conditioning:
            raise ValueError(
                "actuator_delay_observation_only requires enable_delay_conditioning"
            )
        self.high_latency_history_frames = (
            max(1, int(cfg.high_latency_history_frames)) if self.high_latency_obs else 1
        )
        obs_history_frames = (
            self.high_latency_history_frames
            if cfg.high_latency_obs_history_frames is None
            else max(1, int(cfg.high_latency_obs_history_frames))
        )
        action_history_frames = (
            self.high_latency_history_frames
            if cfg.high_latency_action_history_frames is None
            else max(1, int(cfg.high_latency_action_history_frames))
        )
        self.high_latency_obs_history_frames = obs_history_frames if self.high_latency_obs else 1
        self.high_latency_action_history_frames = action_history_frames if self.high_latency_obs else 1
        self.high_latency_obs_prev_frames = max(0, self.high_latency_obs_history_frames - 1)
        self.high_latency_action_prev_frames = max(0, self.high_latency_action_history_frames - 1)
        self.high_latency_prev_frames = max(self.high_latency_obs_prev_frames, self.high_latency_action_prev_frames)
        self.high_latency_extra_dim = 0
        if self.high_latency_obs:
            # predicted ball pos/vel/relative pos (9) + latency/actuator scalars (7).
            self.high_latency_extra_dim = (
                16
                + self.high_latency_obs_prev_frames * self.base_obs_dim
                + self.high_latency_action_prev_frames * self.act_dim
            )
        self.delay_extra_dim = 0
        if self.delay_conditioning:
            self.delay_extra_dim += 1 if bool(cfg.include_tau_act_norm) else 0
            self.delay_extra_dim += self.act_dim if bool(cfg.include_command_state) else 0
            self.delay_extra_dim += self.act_dim if bool(cfg.include_active_command_error) else 0
            self.delay_extra_dim += 2 if bool(cfg.include_phase_features) else 0
            if bool(cfg.use_delay_embedding):
                self.delay_extra_dim += max(0, int(cfg.delay_embedding_dim))
        self.obs_dim = self.base_obs_dim + self.high_latency_extra_dim + self.delay_extra_dim

        edges = np.asarray(tuple(cfg.delay_bin_edges_ms), dtype=np.float32)
        if edges.size < 2:
            edges = np.asarray([float(cfg.delay_min_ms), float(cfg.delay_max_ms)], dtype=np.float32)
        edges = np.sort(edges)
        if float(edges[-1]) <= float(edges[0]):
            edges = np.asarray([float(cfg.delay_min_ms), float(cfg.delay_max_ms)], dtype=np.float32)
        self.delay_bin_edges_ms = jnp.asarray(edges, dtype=jnp.float32)
        self.delay_num_bins = max(1, int(edges.size - 1))
        self.delay_max_s = max(1e-6, float(cfg.delay_max_ms) * 1e-3)

        patched_xml = _apply_right_arm_pd_profile(
            _build_temp_xml_with_ball(self.xml_path),
            str(cfg.right_arm_pd_profile),
        )
        self.mjx_xml = _write_mjx_contact_only_xml(patched_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.mjx_xml))
        self.model = mjx.put_model(self.mj_model)

        self.timestep = float(self.mj_model.opt.timestep)
        self.dt = float(self.timestep * cfg.frame_skip)
        self.max_steps = max(1, int(cfg.horizon_sec / self.dt))
        self.max_command_delay_steps = max(0, int(round(max(0.0, float(cfg.delay_max_ms)) * 1e-3 / max(self.dt, 1e-9))))
        self.command_buffer_len = max(
            1,
            self.max_command_delay_steps + max(0, int(cfg.command_buffer_extra_steps)) + 1,
        )
        self.asymmetric_critic = bool(cfg.asymmetric_critic)
        self.critic_command_history_steps = (
            min(self.command_buffer_len, max(0, int(cfg.critic_command_history_steps)))
            if self.asymmetric_critic
            else 0
        )
        self.critic_extra_dim = (
            80 + self.critic_command_history_steps * self.act_dim if self.asymmetric_critic else 0
        )
        self.critic_obs_dim = self.obs_dim + self.critic_extra_dim

        self.arm_jids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_ARM_JOINTS]
        self.arm_aids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in RIGHT_ARM_JOINTS]
        self.base_aids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in BASE_ACTS]
        self.arm_qadr = jnp.asarray([int(self.mj_model.jnt_qposadr[j]) for j in self.arm_jids], dtype=jnp.int32)
        self.arm_vadr = jnp.asarray([int(self.mj_model.jnt_dofadr[j]) for j in self.arm_jids], dtype=jnp.int32)
        self.arm_aids_j = jnp.asarray(self.arm_aids, dtype=jnp.int32)
        self.base_aids_j = jnp.asarray(self.base_aids, dtype=jnp.int32)
        self.original_arm_actuator_kp = jnp.asarray(
            [float(self.mj_model.actuator_gainprm[aid, 0]) for aid in self.arm_aids],
            dtype=jnp.float32,
        )
        self.original_arm_actuator_kv = jnp.asarray(
            [float(-self.mj_model.actuator_biasprm[aid, 2]) for aid in self.arm_aids],
            dtype=jnp.float32,
        )
        self.arm_lo = jnp.asarray([self.mj_model.jnt_range[j, 0] for j in self.arm_jids], dtype=jnp.float32)
        self.arm_hi = jnp.asarray([self.mj_model.jnt_range[j, 1] for j in self.arm_jids], dtype=jnp.float32)

        self.ball_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball")
        self.ball_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self.racket_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")
        self.racket_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_racket")
        self.racket_wood_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "racket_wood")
        self.racket_rubber_geom_id = self.racket_geom_id
        self.racket_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")
        self.waist_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "waist03")
        self.base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_body_id < 0:
            self.base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if self.base_body_id < 0 and self.ball_obs_frame_pivot_mode == "camera_center":
            raise ValueError(
                "camera-centered ball observation frame requires a 'base_link' or 'base' body"
            )
        self.virtual_camera_base_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, str(cfg.virtual_camera_base_body_name)
        )
        if (
            self.virtual_camera_base_body_id < 0
            and (
                bool(cfg.virtual_camera_require_base_body)
                or (
                    self.ball_obs_frame_pivot_mode == "camera_center"
                    and self.vc_pose_mode == "base_extrinsic"
                )
            )
        ):
            raise ValueError(
                "required virtual camera base body is missing: "
                f"{cfg.virtual_camera_base_body_name!r}"
            )
        if self.virtual_camera_base_body_id < 0:
            self.virtual_camera_base_body_id = self.base_body_id
        self.virtual_camera_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, str(cfg.virtual_camera_body_name)
        )
        if (
            self.virtual_camera_body_id < 0
            and self.ball_obs_frame_pivot_mode == "camera_center"
            and self.vc_pose_mode == "body_mount"
        ):
            raise ValueError(
                "camera-centered ball observation frame requires virtual camera body "
                f"{cfg.virtual_camera_body_name!r}"
            )
        non_racket_gids = []
        for gid in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name and name.startswith("mjx_ball_contact_"):
                non_racket_gids.append(int(gid))
        self.non_racket_geom_ids = jnp.asarray(non_racket_gids or [-1], dtype=jnp.int32)
        ball_racket_pair_ids = []
        for pid in range(self.mj_model.npair):
            g1 = int(self.mj_model.pair_geom1[pid])
            g2 = int(self.mj_model.pair_geom2[pid])
            if {g1, g2} == {self.ball_geom_id, self.racket_geom_id}:
                ball_racket_pair_ids.append(pid)
        self.has_ball_racket_pair = bool(ball_racket_pair_ids)
        self.ball_racket_pair_ids = jnp.asarray(ball_racket_pair_ids or [-1], dtype=jnp.int32)
        self.ball_qadr = int(self.mj_model.jnt_qposadr[self.ball_joint_id])
        self.ball_vadr = int(self.mj_model.jnt_dofadr[self.ball_joint_id])

        bx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")
        by = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")
        bz = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_z")
        broll = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_roll")
        bpitch = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")
        byaw = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")
        self.base_x_qadr = int(self.mj_model.jnt_qposadr[bx])
        self.base_y_qadr = int(self.mj_model.jnt_qposadr[by])
        self.base_z_qadr = int(self.mj_model.jnt_qposadr[bz])
        self.base_roll_qadr = int(self.mj_model.jnt_qposadr[broll])
        self.base_pitch_qadr = int(self.mj_model.jnt_qposadr[bpitch])
        self.base_yaw_qadr = int(self.mj_model.jnt_qposadr[byaw])
        self.base_x_vadr = int(self.mj_model.jnt_dofadr[bx])
        self.base_y_vadr = int(self.mj_model.jnt_dofadr[by])
        self.base_yaw_vadr = int(self.mj_model.jnt_dofadr[byaw])

        target_degrees = dict(TARGET_DEGREES)
        if len(tuple(cfg.right_arm_reset_degrees)) > 0:
            if len(tuple(cfg.right_arm_reset_degrees)) != len(RIGHT_ARM_JOINTS):
                raise ValueError(
                    "right_arm_reset_degrees must contain "
                    f"{len(RIGHT_ARM_JOINTS)} values, got {len(tuple(cfg.right_arm_reset_degrees))}"
                )
            for joint_name, joint_deg in zip(RIGHT_ARM_JOINTS, tuple(cfg.right_arm_reset_degrees)):
                target_degrees[joint_name] = float(joint_deg)
        target_rad = _deg_to_rad_map(target_degrees)
        posture_names = list(target_rad.keys())
        posture_jids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in posture_names]
        posture_aids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in posture_names]
        self.posture_qadr = jnp.asarray([int(self.mj_model.jnt_qposadr[j]) for j in posture_jids], dtype=jnp.int32)
        self.posture_targets = jnp.asarray([target_rad[n] for n in posture_names], dtype=jnp.float32)

        default_ctrl = np.zeros(self.mj_model.nu, dtype=np.float32)
        for n, aid in zip(posture_names, posture_aids):
            if aid >= 0:
                default_ctrl[aid] = np.float32(target_rad[n])
        for aid in self.base_aids:
            if aid >= 0:
                default_ctrl[aid] = 0.0
        self.default_ctrl = jnp.asarray(default_ctrl, dtype=jnp.float32)

        warm = mujoco.MjData(self.mj_model)
        warm.ctrl[:] = default_ctrl
        for _ in range(700):
            mujoco.mj_step(self.mj_model, warm)
        mujoco.mj_forward(self.mj_model, warm)

        self.warm_qpos = jnp.asarray(warm.qpos, dtype=jnp.float32)
        self.warm_qvel = jnp.asarray(warm.qvel, dtype=jnp.float32)
        self.warm_ctrl = jnp.asarray(default_ctrl, dtype=jnp.float32)
        self.warm_arm_q = self.warm_qpos[self.arm_qadr]
        self.warm_arm_qvel = self.warm_qvel[self.arm_vadr]
        self.racket_anchor = jnp.asarray(warm.site_xpos[self.racket_site_id], dtype=jnp.float32)
        if self.waist_body_id >= 0:
            self.chest_target_offset = jnp.asarray(
                warm.site_xpos[self.racket_site_id] - warm.xpos[self.waist_body_id],
                dtype=jnp.float32,
            )
        else:
            self.chest_target_offset = jnp.zeros((3,), dtype=jnp.float32)
        self.initial_base_pose = jnp.asarray(
            [
                warm.qpos[self.base_x_qadr],
                warm.qpos[self.base_y_qadr],
                warm.qpos[self.base_yaw_qadr],
            ],
            dtype=jnp.float32,
        )
        self.initial_base_z = jnp.asarray(warm.qpos[self.base_z_qadr], dtype=jnp.float32)
        self.initial_base_roll = jnp.asarray(warm.qpos[self.base_roll_qadr], dtype=jnp.float32)
        self.initial_base_pitch = jnp.asarray(warm.qpos[self.base_pitch_qadr], dtype=jnp.float32)

        self.arm_vel_limit_rad_s = jnp.deg2rad(jnp.asarray(cfg.arm_vel_limit_deg_s, dtype=jnp.float32))
        self.arm_acc_limit_rad_s2 = jnp.deg2rad(jnp.asarray(cfg.arm_acc_limit_deg_s2, dtype=jnp.float32))
        self.arm_actual_jerk_limit_rad_s3 = jnp.deg2rad(
            jnp.asarray(cfg.arm_actual_jerk_limit_deg_s3, dtype=jnp.float32)
        )
        self.arm_bridger_jerk_limit_rad_s3 = jnp.deg2rad(
            jnp.asarray(cfg.actuator_bridger_jerk_limit_deg_s3, dtype=jnp.float32)
        )
        compensation_mode = str(cfg.actuator_compensation_mode or "none").strip().lower().replace("-", "_")
        if compensation_mode in {"sim2real_bridger", "constrained_inverse_mpc", "bridger"}:
            if bool(cfg.arm_actual_target_tracking_governor):
                raise ValueError(
                    "Sim2Real Bridger owns q/dq/ddq/jerk feasibility; "
                    "arm_actual_target_tracking_governor must be disabled"
                )
            if bool(cfg.arm_post_compensation_limiter):
                raise ValueError(
                    "Sim2Real Bridger must not be followed by arm_post_compensation_limiter"
                )
            if tuple(self.arm_bridger_jerk_limit_rad_s3.shape) != (self.act_dim,):
                raise ValueError("actuator_bridger_jerk_limit_deg_s3 must contain seven values")
            if not np.isfinite(float(cfg.actuator_bridger_natural_frequency_hz)) or float(
                cfg.actuator_bridger_natural_frequency_hz
            ) <= 0.0:
                raise ValueError("actuator_bridger_natural_frequency_hz must be positive")
            if not np.isfinite(float(cfg.actuator_bridger_damping_ratio)) or float(
                cfg.actuator_bridger_damping_ratio
            ) <= 0.0:
                raise ValueError("actuator_bridger_damping_ratio must be positive")
        if bool(cfg.arm_actual_target_tracking_governor):
            if not bool(cfg.arm_actual_state_limiter):
                raise ValueError(
                    "arm_actual_target_tracking_governor requires "
                    "arm_actual_state_limiter"
                )
            if tuple(self.arm_actual_jerk_limit_rad_s3.shape) != (self.act_dim,):
                raise ValueError(
                    "arm_actual_jerk_limit_deg_s3 must contain seven values"
                )
            if np.any(
                ~np.isfinite(np.asarray(cfg.arm_actual_jerk_limit_deg_s3, dtype=np.float64))
            ) or np.any(
                np.asarray(cfg.arm_actual_jerk_limit_deg_s3, dtype=np.float64) <= 0.0
            ):
                raise ValueError(
                    "arm_actual_jerk_limit_deg_s3 must be positive and finite"
                )
            if (
                not np.isfinite(float(cfg.arm_actual_governor_natural_frequency_hz))
                or float(cfg.arm_actual_governor_natural_frequency_hz) <= 0.0
            ):
                raise ValueError(
                    "arm_actual_governor_natural_frequency_hz must be positive and finite"
                )
            if (
                not np.isfinite(float(cfg.arm_actual_governor_damping_ratio))
                or float(cfg.arm_actual_governor_damping_ratio) <= 0.0
            ):
                raise ValueError(
                    "arm_actual_governor_damping_ratio must be positive and finite"
                )
        self.default_gravity_z = float(self.mj_model.opt.gravity[2])
        self.gravity_mag = float(np.linalg.norm(self.mj_model.opt.gravity))
        self.original_ball_mass = float(self.mj_model.body_mass[self.ball_body_id]) if self.ball_body_id >= 0 else 0.0027
        self.original_ball_inertia = (
            jnp.asarray(self.mj_model.body_inertia[self.ball_body_id], dtype=jnp.float32)
            if self.ball_body_id >= 0
            else jnp.ones((3,), dtype=jnp.float32)
        )
        self.original_ball_friction = float(self.mj_model.geom_friction[self.ball_geom_id, 0]) if self.ball_geom_id >= 0 else 0.20
        self.original_ball_solref_time = float(self.mj_model.geom_solref[self.ball_geom_id, 0]) if self.ball_geom_id >= 0 else 0.003
        self.original_ball_solref_damping = float(self.mj_model.geom_solref[self.ball_geom_id, 1]) if self.ball_geom_id >= 0 else 0.80
        self.original_racket_friction = (
            float(self.mj_model.geom_friction[self.racket_geom_id, 0]) if self.racket_geom_id >= 0 else 0.35
        )
        self.original_dof_damping = jnp.asarray(self.mj_model.dof_damping, dtype=jnp.float32)
        self.original_dof_armature = jnp.asarray(self.mj_model.dof_armature, dtype=jnp.float32)
        self.original_racket_body_pos = (
            jnp.asarray(self.mj_model.body_pos[self.racket_body_id], dtype=jnp.float32)
            if self.racket_body_id >= 0
            else jnp.zeros((3,), dtype=jnp.float32)
        )
        self.original_racket_body_quat = (
            jnp.asarray(self.mj_model.body_quat[self.racket_body_id], dtype=jnp.float32)
            if self.racket_body_id >= 0
            else jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        )
        self.racket_mount_geom_ids = jnp.asarray(
            [gid for gid in (self.racket_wood_geom_id, self.racket_rubber_geom_id) if gid >= 0] or [-1],
            dtype=jnp.int32,
        )
        self.has_racket_mount_geoms = bool([gid for gid in (self.racket_wood_geom_id, self.racket_rubber_geom_id) if gid >= 0])
        self.original_racket_mount_geom_sizes = jnp.asarray(
            [
                self.mj_model.geom_size[gid]
                for gid in (self.racket_wood_geom_id, self.racket_rubber_geom_id)
                if gid >= 0
            ]
            or [np.zeros(3, dtype=np.float32)],
            dtype=jnp.float32,
        )
        self.ball_obs_every = 1
        self.ball_obs_period = 1.0
        if float(cfg.ball_obs_rate_hz) > 0.0:
            self.ball_obs_period = float(cfg.ball_obs_rate_hz) * self.dt
            self.ball_obs_every = max(1, int(round(1.0 / max(1e-9, self.ball_obs_period))))
        else:
            self.ball_obs_period = 1.0
        self.max_obs_latency_steps = max(0, int(cfg.dr_obs_latency_steps_range[1])) if cfg.domain_randomization else 0
        self.max_action_latency_steps = max(0, int(cfg.dr_action_latency_steps_range[1])) if cfg.domain_randomization else 0
        self.hit_reward_count_cap_active = self._get_hit_reward_count_cap()
        self.vc_mount_R = jnp.asarray(_quat_wxyz_to_mat_np(cfg.virtual_camera_mount_quat), dtype=jnp.float32)
        self.vc_mount_pos = jnp.asarray(cfg.virtual_camera_mount_pos, dtype=jnp.float32)
        self.vc_optical_pos = jnp.asarray(cfg.virtual_camera_optical_pos, dtype=jnp.float32)
        self.vc_base_pos = jnp.asarray(
            cfg_values.get("virtual_camera_base_pos", D455_848_UNDISTORTED_BASE_POS),
            dtype=jnp.float32,
        )
        self.vc_base_R = jnp.asarray(
            cfg_values.get("virtual_camera_base_rot", D455_848_UNDISTORTED_BASE_ROT),
            dtype=jnp.float32,
        ).reshape((3, 3))
        self.vc_mount_to_camera_R = jnp.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=jnp.float32,
        )
        self.base_data = mjx.make_data(self.model).replace(
            qpos=self.warm_qpos,
            qvel=self.warm_qvel,
            ctrl=self.warm_ctrl,
        )
        self.base_data = mjx.forward(self.model, self.base_data)
        self.batched_step = jax.vmap(lambda model, data: mjx.step(model, data))
        self.batched_forward = jax.vmap(lambda model, data: mjx.forward(model, data))

    def _get_hit_reward_count_cap(self) -> int:
        if self.cfg.hit_reward_cap_mode == "off":
            return 0
        if self.cfg.hit_reward_cap_mode == "fixed":
            return max(0, int(self.cfg.hit_reward_count_cap))
        if self.cfg.hit_reward_cap_mode == "auto":
            episode_total_time = float(self.max_steps) * self.dt
            cap = int(np.floor(episode_total_time / max(1e-6, float(self.cfg.hit_reward_cap_target_interval))))
            return max(1, cap)
        raise ValueError(
            f"Invalid hit_reward_cap_mode={self.cfg.hit_reward_cap_mode!r}; expected 'off', 'auto', or 'fixed'."
        )

    def _make_batched_model(
        self,
        n_envs: int,
        dr_gravity_z: jax.Array,
        dr_ball_mass: jax.Array,
        dr_ball_friction: jax.Array,
        dr_racket_friction: jax.Array,
        dr_ball_solref_time: jax.Array,
        dr_ball_solref_damping: jax.Array,
        dr_damping_mult: jax.Array,
        dr_armature_mult: jax.Array,
        dr_pd_kp_mult: jax.Array,
        dr_pd_kv_mult: jax.Array,
        dr_racket_pos_offset: jax.Array,
        dr_racket_rot_offset: jax.Array,
        dr_racket_radius_offset: jax.Array,
    ):
        model = _batch_tree(self.model, n_envs)

        opt = model.opt.replace(gravity=model.opt.gravity.at[:, 2].set(dr_gravity_z))
        model = model.replace(opt=opt)

        if self.ball_body_id >= 0:
            mass_mult = dr_ball_mass / max(self.original_ball_mass, 1e-9)
            body_mass = model.body_mass.at[:, self.ball_body_id].set(dr_ball_mass)
            body_inertia = model.body_inertia.at[:, self.ball_body_id, :].set(
                self.original_ball_inertia[None, :] * mass_mult[:, None]
            )
            model = model.replace(body_mass=body_mass, body_inertia=body_inertia)

        dof_damping = self.original_dof_damping[None, :] * dr_damping_mult[:, None]
        dof_armature = self.original_dof_armature[None, :] * dr_armature_mult[:, None]
        model = model.replace(dof_damping=dof_damping, dof_armature=dof_armature)

        kp = self.original_arm_actuator_kp[None, :] * dr_pd_kp_mult
        kv = self.original_arm_actuator_kv[None, :] * dr_pd_kv_mult
        env_ids = jnp.arange(n_envs, dtype=jnp.int32)[:, None]
        arm_aids = self.arm_aids_j[None, :]
        actuator_gainprm = model.actuator_gainprm.at[env_ids, arm_aids, 0].set(kp)
        actuator_biasprm = model.actuator_biasprm.at[env_ids, arm_aids, 1].set(-kp)
        actuator_biasprm = actuator_biasprm.at[env_ids, arm_aids, 2].set(-kv)
        model = model.replace(actuator_gainprm=actuator_gainprm, actuator_biasprm=actuator_biasprm)

        geom_friction = model.geom_friction
        geom_solref = model.geom_solref
        if self.ball_geom_id >= 0:
            geom_friction = geom_friction.at[:, self.ball_geom_id, 0].set(dr_ball_friction)
            geom_solref = geom_solref.at[:, self.ball_geom_id, 0].set(dr_ball_solref_time)
            geom_solref = geom_solref.at[:, self.ball_geom_id, 1].set(dr_ball_solref_damping)
        if self.racket_geom_id >= 0:
            geom_friction = geom_friction.at[:, self.racket_geom_id, 0].set(dr_racket_friction)
        model = model.replace(geom_friction=geom_friction, geom_solref=geom_solref)

        if self.has_ball_racket_pair:
            pair_friction = model.pair_friction.at[:, self.ball_racket_pair_ids, 0].set(
                dr_racket_friction[:, None]
            )
            pair_solref = model.pair_solref.at[:, self.ball_racket_pair_ids, 0].set(
                dr_ball_solref_time[:, None]
            )
            pair_solref = pair_solref.at[:, self.ball_racket_pair_ids, 1].set(
                dr_ball_solref_damping[:, None]
            )
            model = model.replace(pair_friction=pair_friction, pair_solref=pair_solref)

        if self.racket_body_id >= 0:
            rot_quat = jax.vmap(_euler_xyz_to_quat_wxyz_jax)(dr_racket_rot_offset)
            racket_quat = jax.vmap(lambda q: _quat_mul_wxyz_jax(self.original_racket_body_quat, q))(rot_quat)
            body_pos = model.body_pos.at[:, self.racket_body_id, :].set(
                self.original_racket_body_pos[None, :] + dr_racket_pos_offset
            )
            body_quat = model.body_quat.at[:, self.racket_body_id, :].set(racket_quat)
            model = model.replace(body_pos=body_pos, body_quat=body_quat)

        if self.has_racket_mount_geoms:
            new_sizes = jnp.broadcast_to(
                self.original_racket_mount_geom_sizes[None, :, :],
                (n_envs,) + tuple(self.original_racket_mount_geom_sizes.shape),
            )
            radius = jnp.maximum(0.03, new_sizes[:, :, 0] + dr_racket_radius_offset[:, None])
            new_sizes = new_sizes.at[:, :, 0].set(radius)
            geom_size = model.geom_size.at[:, self.racket_mount_geom_ids, :].set(new_sizes)
            model = model.replace(geom_size=geom_size)

        return model

    def reset(self, keys: jax.Array) -> tuple[EnvState, jax.Array]:
        keys = jnp.asarray(keys)
        n_envs = keys.shape[0]
        data = _batch_tree(self.base_data, n_envs)

        split_keys = jax.vmap(lambda k: jax.random.split(k, 34))(keys)
        next_keys = split_keys[:, 0]
        key_xy = split_keys[:, 1]
        key_z = split_keys[:, 2]
        key_vel = split_keys[:, 3]
        key_action_scale = split_keys[:, 4]
        key_gravity = split_keys[:, 5]
        key_ball_mass = split_keys[:, 6]
        key_ball_friction = split_keys[:, 7]
        key_racket_friction = split_keys[:, 8]
        key_solref = split_keys[:, 9]
        key_obs_latency = split_keys[:, 10]
        key_action_latency = split_keys[:, 11]
        key_damping = split_keys[:, 12]
        key_armature = split_keys[:, 13]
        key_racket_pos = split_keys[:, 14]
        key_racket_rot = split_keys[:, 15]
        key_racket_radius = split_keys[:, 16]
        key_ball_obs_pos_bias = split_keys[:, 17]
        key_ball_obs_rot_bias = split_keys[:, 18]
        key_ball_obs_vel_bias = split_keys[:, 19]
        key_ball_obs_scale = split_keys[:, 20]
        key_actuator_tau = split_keys[:, 21]
        key_actuator_gain = split_keys[:, 22]
        key_delay_bin = split_keys[:, 23]
        key_delay_tau = split_keys[:, 24]
        key_pd_kp = split_keys[:, 25]
        key_pd_kv = split_keys[:, 26]
        key_episode_target_x = split_keys[:, 27]
        key_episode_target_y = split_keys[:, 28]
        key_episode_anchor_z = split_keys[:, 29]
        key_ball_init_vz = split_keys[:, 30]
        key_ball_obs_view_z_high = split_keys[:, 31]
        key_falling_tau = split_keys[:, 32]
        key_falling_apex = split_keys[:, 33]
        falling_reset = self.ball_reset_mode == "falling_contact"
        racket_launch_reset = self.ball_reset_mode == "racket_launch"
        xy_jitter_limit = (
            float(self.cfg.falling_reset_contact_xy_jitter)
            if falling_reset
            else (
                float(self.cfg.racket_launch_xy_jitter)
                if racket_launch_reset
                else float(self.cfg.ball_spawn_xy_jitter)
            )
        )
        vxy_limit = (
            float(self.cfg.falling_reset_vxy_max)
            if falling_reset and float(self.cfg.falling_reset_vxy_max) > 0.0
            else (
                float(self.cfg.racket_launch_vxy_max)
                if racket_launch_reset
                else float(self.cfg.ball_init_vxy_max)
            )
        )
        xy_jitter = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (2,),
                minval=-xy_jitter_limit,
                maxval=xy_jitter_limit,
            )
        )(key_xy)
        if racket_launch_reset:
            gap_lo, gap_hi = sorted(float(v) for v in self.cfg.racket_launch_surface_gap_range_m)
            racket_launch_surface_gap = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=gap_lo,
                    maxval=max(gap_lo + 1e-6, gap_hi),
                )
            )(key_z)
            z_jitter = jnp.zeros((n_envs,), dtype=jnp.float32)
        else:
            racket_launch_surface_gap = jnp.zeros((n_envs,), dtype=jnp.float32)
            z_jitter = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=-float(self.cfg.ball_spawn_z_jitter),
                    maxval=float(self.cfg.ball_spawn_z_jitter),
                )
            )(key_z)
        vxy = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (2,),
                minval=-vxy_limit,
                maxval=vxy_limit,
            )
        )(key_vel)
        vz_jitter_mag = (
            float(self.cfg.racket_launch_vnormal_max)
            if racket_launch_reset
            else abs(float(self.cfg.ball_init_vz_jitter))
        )
        vz_jitter = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (),
                minval=-vz_jitter_mag,
                maxval=vz_jitter_mag,
            )
        )(key_ball_init_vz)
        init_vz = vz_jitter if racket_launch_reset else float(self.cfg.ball_init_vz) + vz_jitter
        target_x_offset = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (),
                minval=float(self.cfg.episode_target_x_range_m[0]),
                maxval=float(self.cfg.episode_target_x_range_m[1]),
            )
        )(key_episode_target_x)
        target_y_offset = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (),
                minval=float(self.cfg.episode_target_y_range_m[0]),
                maxval=float(self.cfg.episode_target_y_range_m[1]),
            )
        )(key_episode_target_y)
        target_z_offset = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (),
                minval=float(self.cfg.episode_racket_anchor_z_range_m[0]),
                maxval=float(self.cfg.episode_racket_anchor_z_range_m[1]),
            )
        )(key_episode_anchor_z)
        tau_lo, tau_hi = [float(v) for v in self.cfg.falling_reset_time_to_contact_range_s]
        tau_lo, tau_hi = min(tau_lo, tau_hi), max(tau_lo, tau_hi)
        falling_tau_raw = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (),
                minval=tau_lo,
                maxval=max(tau_lo + 1e-6, tau_hi),
            )
        )(key_falling_tau)
        apex_lo, apex_hi = [float(v) for v in self.cfg.falling_reset_apex_height_range_m]
        if apex_hi <= 0.0 and apex_lo <= 0.0:
            contact_rel_height_for_default = (
                float(self.cfg.hit_confirm_rel_height)
                if float(self.cfg.falling_reset_contact_rel_height) < 0.0
                else float(self.cfg.falling_reset_contact_rel_height)
            )
            default_apex = max(0.08, float(self.cfg.hit_height_center) - contact_rel_height_for_default)
            apex_lo = default_apex
            apex_hi = default_apex
        apex_lo, apex_hi = min(apex_lo, apex_hi), max(apex_lo, apex_hi)
        falling_apex_height = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (),
                minval=max(1e-4, apex_lo),
                maxval=max(max(1e-4, apex_lo) + 1e-6, apex_hi),
            )
        )(key_falling_apex)

        zero_action = jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32)
        zero_ball_vel = jnp.zeros((n_envs, 3), dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_actuator):
            action_scale_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_action_scale_mult_range[0]),
                    maxval=float(self.cfg.dr_action_scale_mult_range[1]),
                )
            )(key_action_scale)
            dr_damping_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_damping_mult_range[0]),
                    maxval=float(self.cfg.dr_damping_mult_range[1]),
                )
            )(key_damping)
            dr_armature_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_armature_mult_range[0]),
                    maxval=float(self.cfg.dr_armature_mult_range[1]),
                )
            )(key_armature)
        else:
            action_scale_mult = jnp.ones((n_envs,), dtype=jnp.float32)
            dr_damping_mult = jnp.ones((n_envs,), dtype=jnp.float32)
            dr_armature_mult = jnp.ones((n_envs,), dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_actuator and self.cfg.dr_randomize_pd):
            kp_low, kp_high = [float(v) for v in self.cfg.dr_pd_kp_mult_range]
            kv_low, kv_high = [float(v) for v in self.cfg.dr_pd_kv_mult_range]
            sample_shape = (self.act_dim,) if bool(self.cfg.dr_pd_per_joint) else ()
            dr_pd_kp_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    sample_shape,
                    minval=min(kp_low, kp_high),
                    maxval=max(kp_low, kp_high),
                    dtype=jnp.float32,
                )
            )(key_pd_kp)
            dr_pd_kv_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    sample_shape,
                    minval=min(kv_low, kv_high),
                    maxval=max(kv_low, kv_high),
                    dtype=jnp.float32,
                )
            )(key_pd_kv)
            if not bool(self.cfg.dr_pd_per_joint):
                dr_pd_kp_mult = jnp.broadcast_to(dr_pd_kp_mult[:, None], (n_envs, self.act_dim))
                dr_pd_kv_mult = jnp.broadcast_to(dr_pd_kv_mult[:, None], (n_envs, self.act_dim))
        else:
            dr_pd_kp_mult = jnp.ones((n_envs, self.act_dim), dtype=jnp.float32)
            dr_pd_kv_mult = jnp.ones((n_envs, self.act_dim), dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_ball):
            dr_gravity_z = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_gravity_z_range[0]),
                    maxval=float(self.cfg.dr_gravity_z_range[1]),
                )
            )(key_gravity)
            dr_ball_mass = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_ball_mass_range[0]),
                    maxval=float(self.cfg.dr_ball_mass_range[1]),
                )
            )(key_ball_mass)
        else:
            dr_gravity_z = jnp.full((n_envs,), self.default_gravity_z, dtype=jnp.float32)
            dr_ball_mass = jnp.full((n_envs,), self.original_ball_mass, dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_contact):
            dr_ball_friction = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_ball_friction_range[0]),
                    maxval=float(self.cfg.dr_ball_friction_range[1]),
                )
            )(key_ball_friction)
            dr_racket_friction = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_racket_friction_range[0]),
                    maxval=float(self.cfg.dr_racket_friction_range[1]),
                )
            )(key_racket_friction)
            solref_samples = jax.vmap(
                lambda k: jnp.asarray(
                    [
                        jax.random.uniform(
                            k,
                            (),
                            minval=float(self.cfg.dr_ball_solref_time_range[0]),
                            maxval=float(self.cfg.dr_ball_solref_time_range[1]),
                        ),
                        jax.random.uniform(
                            jax.random.fold_in(k, 1),
                            (),
                            minval=float(self.cfg.dr_ball_solref_damping_range[0]),
                            maxval=float(self.cfg.dr_ball_solref_damping_range[1]),
                        ),
                    ],
                    dtype=jnp.float32,
                )
            )(key_solref)
            dr_ball_solref_time = solref_samples[:, 0]
            dr_ball_solref_damping = solref_samples[:, 1]
        else:
            dr_ball_friction = jnp.full((n_envs,), self.original_ball_friction, dtype=jnp.float32)
            dr_racket_friction = jnp.full((n_envs,), self.original_racket_friction, dtype=jnp.float32)
            dr_ball_solref_time = jnp.full((n_envs,), self.original_ball_solref_time, dtype=jnp.float32)
            dr_ball_solref_damping = jnp.full((n_envs,), self.original_ball_solref_damping, dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_latency):
            obs_low, obs_high = [int(v) for v in self.cfg.dr_obs_latency_steps_range]
            act_low, act_high = [int(v) for v in self.cfg.dr_action_latency_steps_range]
            obs_latency_steps = jax.vmap(
                lambda k: jax.random.randint(k, (), minval=obs_low, maxval=max(obs_low + 1, obs_high + 1), dtype=jnp.int32)
            )(key_obs_latency)
            action_latency_steps = jax.vmap(
                lambda k: jax.random.randint(k, (), minval=act_low, maxval=max(act_low + 1, act_high + 1), dtype=jnp.int32)
            )(key_action_latency)
        else:
            obs_latency_steps = jnp.zeros((n_envs,), dtype=jnp.int32)
            action_latency_steps = jnp.zeros((n_envs,), dtype=jnp.int32)

        if self.delay_conditioning:
            delay_min_ms = float(self.cfg.delay_min_ms)
            delay_max_ms = float(self.cfg.delay_max_ms)
            lo_ms = min(delay_min_ms, delay_max_ms)
            hi_ms = max(delay_min_ms, delay_max_ms)
            if str(self.cfg.delay_sampling_mode) == "balanced_bins":
                delay_bin_id = jax.vmap(
                    lambda k: jax.random.randint(k, (), minval=0, maxval=self.delay_num_bins, dtype=jnp.int32)
                )(key_delay_bin)
                bin_lo = self.delay_bin_edges_ms[:-1][delay_bin_id]
                bin_hi = self.delay_bin_edges_ms[1:][delay_bin_id]
                bin_lo = jnp.clip(bin_lo, lo_ms, hi_ms)
                bin_hi = jnp.clip(bin_hi, lo_ms, hi_ms)
                bin_hi = jnp.maximum(bin_hi, bin_lo)
                tau_ms = jax.vmap(lambda k, low, high: jax.random.uniform(k, (), minval=low, maxval=high))(
                    key_delay_tau,
                    bin_lo,
                    bin_hi,
                )
            elif str(self.cfg.delay_sampling_mode) == "uniform":
                tau_ms = jax.vmap(
                    lambda k: jax.random.uniform(k, (), minval=lo_ms, maxval=hi_ms)
                )(key_delay_tau)
                delay_bin_id = jnp.sum(
                    (tau_ms[:, None] >= self.delay_bin_edges_ms[None, 1:]).astype(jnp.int32),
                    axis=-1,
                )
                delay_bin_id = jnp.clip(delay_bin_id, 0, self.delay_num_bins - 1)
            else:
                raise ValueError(
                    f"Invalid delay_sampling_mode={self.cfg.delay_sampling_mode!r}; "
                    "expected 'uniform' or 'balanced_bins'."
                )
            tau_act_episode = jnp.clip(tau_ms, lo_ms, hi_ms) * 1e-3
            tau_act = tau_act_episode
            delay_steps = jnp.rint(tau_act / max(self.dt, 1e-9)).astype(jnp.int32)
            delay_steps = jnp.clip(delay_steps, 0, self.command_buffer_len - 1)
        else:
            tau_act_episode = jnp.zeros((n_envs,), dtype=jnp.float32)
            tau_act = jnp.zeros((n_envs,), dtype=jnp.float32)
            delay_steps = jnp.zeros((n_envs,), dtype=jnp.int32)
            delay_bin_id = jnp.zeros((n_envs,), dtype=jnp.int32)

        if bool(self.cfg.actuator_cmd_filter):
            if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_actuator_cmd_filter):
                tau_low, tau_high = [float(v) for v in self.cfg.dr_actuator_cmd_tau_range]
                gain_low, gain_high = [float(v) for v in self.cfg.dr_actuator_cmd_gain_range]
                actuator_cmd_tau = jax.vmap(
                    lambda k: jax.random.uniform(
                        k,
                        (),
                        minval=min(tau_low, tau_high),
                        maxval=max(tau_low, tau_high),
                    )
                )(key_actuator_tau)
                actuator_cmd_gain = jax.vmap(
                    lambda k: jax.random.uniform(
                        k,
                        (),
                        minval=min(gain_low, gain_high),
                        maxval=max(gain_low, gain_high),
                    )
                )(key_actuator_gain)
            else:
                actuator_cmd_tau = jnp.full((n_envs,), float(self.cfg.actuator_cmd_tau), dtype=jnp.float32)
                actuator_cmd_gain = jnp.full((n_envs,), float(self.cfg.actuator_cmd_gain), dtype=jnp.float32)
        else:
            actuator_cmd_tau = jnp.zeros((n_envs,), dtype=jnp.float32)
            actuator_cmd_gain = jnp.ones((n_envs,), dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_racket_mount):
            dr_racket_pos_offset = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (3,),
                    minval=-float(self.cfg.dr_racket_pos_offset_m),
                    maxval=float(self.cfg.dr_racket_pos_offset_m),
                )
            )(key_racket_pos)
            dr_racket_rot_offset = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (3,),
                    minval=-float(self.cfg.dr_racket_rot_offset_rad),
                    maxval=float(self.cfg.dr_racket_rot_offset_rad),
                )
            )(key_racket_rot)
            dr_racket_radius_offset = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=-float(self.cfg.dr_racket_radius_offset_m),
                    maxval=float(self.cfg.dr_racket_radius_offset_m),
                )
            )(key_racket_radius)
        else:
            dr_racket_pos_offset = jnp.zeros((n_envs, 3), dtype=jnp.float32)
            dr_racket_rot_offset = jnp.zeros((n_envs, 3), dtype=jnp.float32)
            dr_racket_radius_offset = jnp.zeros((n_envs,), dtype=jnp.float32)

        nominal_pos_bias = jnp.asarray(self.cfg.ball_obs_nominal_pos_bias_base, dtype=jnp.float32)
        nominal_vel_bias = jnp.asarray(self.cfg.ball_obs_nominal_vel_bias_base, dtype=jnp.float32)
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_ball_obs_frame):
            pos_bias_lim = jnp.asarray(self.cfg.dr_ball_obs_pos_bias_base_m, dtype=jnp.float32)
            rot_bias_lim = jnp.deg2rad(jnp.asarray(self.cfg.dr_ball_obs_rot_bias_deg, dtype=jnp.float32))
            vel_bias_lim = jnp.asarray(self.cfg.dr_ball_obs_vel_bias_base_m_s, dtype=jnp.float32)
            scale_low, scale_high = [float(v) for v in self.cfg.dr_ball_obs_scale_range]
            ball_obs_pos_bias_base = nominal_pos_bias[None, :] + jax.vmap(
                lambda k: jax.random.uniform(k, (3,), minval=-pos_bias_lim, maxval=pos_bias_lim)
            )(key_ball_obs_pos_bias)
            ball_obs_rot_bias_rpy = jax.vmap(
                lambda k: jax.random.uniform(k, (3,), minval=-rot_bias_lim, maxval=rot_bias_lim)
            )(key_ball_obs_rot_bias)
            ball_obs_vel_bias_base = nominal_vel_bias[None, :] + jax.vmap(
                lambda k: jax.random.uniform(k, (3,), minval=-vel_bias_lim, maxval=vel_bias_lim)
            )(key_ball_obs_vel_bias)
            ball_obs_scale = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=min(scale_low, scale_high),
                    maxval=max(scale_low, scale_high),
                )
            )(key_ball_obs_scale)
        else:
            ball_obs_pos_bias_base = jnp.broadcast_to(nominal_pos_bias, (n_envs, 3))
            ball_obs_rot_bias_rpy = jnp.zeros((n_envs, 3), dtype=jnp.float32)
            ball_obs_vel_bias_base = jnp.broadcast_to(nominal_vel_bias, (n_envs, 3))
            ball_obs_scale = jnp.ones((n_envs,), dtype=jnp.float32)

        z_high_low, z_high_high = [float(v) for v in self.cfg.ball_obs_view_z_high_missing_range_m]
        if z_high_low <= 0.0 and z_high_high <= 0.0:
            z_high_low = float(self.cfg.ball_view_z_bounds_m[1])
            z_high_high = z_high_low
        z_high_low, z_high_high = min(z_high_low, z_high_high), max(z_high_low, z_high_high)
        if z_high_high > z_high_low:
            ball_obs_view_z_high_m = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=z_high_low,
                    maxval=z_high_high,
                    dtype=jnp.float32,
                )
            )(key_ball_obs_view_z_high)
        else:
            ball_obs_view_z_high_m = jnp.full((n_envs,), z_high_low, dtype=jnp.float32)

        zero_strength = jnp.zeros((n_envs,), dtype=jnp.float32)

        def squared_norm(value: jax.Array, scale: float) -> jax.Array:
            denom = max(abs(float(scale)), 1e-6)
            normalized = value / denom
            if normalized.ndim <= 1:
                return normalized * normalized
            return jnp.sum(normalized * normalized, axis=-1)

        def squared_range(value: jax.Array, bounds: tuple[float, float]) -> jax.Array:
            lo, hi = [float(v) for v in bounds]
            center = 0.5 * (lo + hi)
            half_width = max(0.5 * abs(hi - lo), 1e-6)
            return ((value - center) / half_width) ** 2

        disturbance_sq = (
            squared_norm(xy_jitter, xy_jitter_limit)
            + squared_norm(z_jitter, float(self.cfg.ball_spawn_z_jitter))
            + squared_norm(vxy, vxy_limit)
            # Racket-launch reset samples vertical velocity from
            # racket_launch_vnormal_max, while the released-ball reset uses
            # ball_init_vz_jitter.  Normalizing every mode by the latter made
            # autonomous-launch disturbance strength explode when the legacy
            # jitter was zero (0.003 / 1e-6 ~= 3000), corrupting reset-bucket
            # diagnostics and CVaR validation.
            + squared_norm(vz_jitter, vz_jitter_mag)
        )
        if falling_reset:
            disturbance_sq = (
                disturbance_sq
                + squared_range(falling_tau_raw, self.cfg.falling_reset_time_to_contact_range_s)
                + squared_range(
                    falling_apex_height,
                    (
                        max(1e-4, float(apex_lo)),
                        max(max(1e-4, float(apex_lo)) + 1e-6, float(apex_hi)),
                    ),
                )
            )
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_actuator):
            disturbance_sq = (
                disturbance_sq
                + squared_range(action_scale_mult, self.cfg.dr_action_scale_mult_range)
                + squared_range(dr_damping_mult, self.cfg.dr_damping_mult_range)
                + squared_range(dr_armature_mult, self.cfg.dr_armature_mult_range)
            )
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_actuator and self.cfg.dr_randomize_pd):
            disturbance_sq = disturbance_sq + jnp.mean((dr_pd_kp_mult - 1.0) ** 2, axis=-1)
            disturbance_sq = disturbance_sq + jnp.mean((dr_pd_kv_mult - 1.0) ** 2, axis=-1)
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_ball):
            disturbance_sq = (
                disturbance_sq
                + squared_range(dr_gravity_z, self.cfg.dr_gravity_z_range)
                + squared_range(dr_ball_mass, self.cfg.dr_ball_mass_range)
            )
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_contact):
            disturbance_sq = (
                disturbance_sq
                + squared_range(dr_ball_friction, self.cfg.dr_ball_friction_range)
                + squared_range(dr_racket_friction, self.cfg.dr_racket_friction_range)
                + squared_range(dr_ball_solref_time, self.cfg.dr_ball_solref_time_range)
                + squared_range(dr_ball_solref_damping, self.cfg.dr_ball_solref_damping_range)
            )
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_latency):
            obs_span = max(1, int(self.cfg.dr_obs_latency_steps_range[1]) - int(self.cfg.dr_obs_latency_steps_range[0]))
            act_span = max(1, int(self.cfg.dr_action_latency_steps_range[1]) - int(self.cfg.dr_action_latency_steps_range[0]))
            disturbance_sq = (
                disturbance_sq
                + squared_norm(obs_latency_steps.astype(jnp.float32), float(obs_span))
                + squared_norm(action_latency_steps.astype(jnp.float32), float(act_span))
            )
        if bool(self.cfg.actuator_cmd_filter and self.cfg.domain_randomization and self.cfg.dr_randomize_actuator_cmd_filter):
            disturbance_sq = (
                disturbance_sq
                + squared_range(actuator_cmd_tau, self.cfg.dr_actuator_cmd_tau_range)
                + squared_range(actuator_cmd_gain, self.cfg.dr_actuator_cmd_gain_range)
            )
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_racket_mount):
            disturbance_sq = (
                disturbance_sq
                + squared_norm(dr_racket_pos_offset, float(self.cfg.dr_racket_pos_offset_m))
                + squared_norm(dr_racket_rot_offset, float(self.cfg.dr_racket_rot_offset_rad))
                + squared_norm(dr_racket_radius_offset, float(self.cfg.dr_racket_radius_offset_m))
            )
        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_ball_obs_frame):
            pos_bias_scale = max(float(np.linalg.norm(np.asarray(self.cfg.dr_ball_obs_pos_bias_base_m))), 1e-6)
            rot_bias_scale = max(float(np.linalg.norm(np.deg2rad(np.asarray(self.cfg.dr_ball_obs_rot_bias_deg)))), 1e-6)
            vel_bias_scale = max(float(np.linalg.norm(np.asarray(self.cfg.dr_ball_obs_vel_bias_base_m_s))), 1e-6)
            disturbance_sq = (
                disturbance_sq
                + squared_norm(ball_obs_pos_bias_base - nominal_pos_bias[None, :], pos_bias_scale)
                + squared_norm(ball_obs_rot_bias_rpy, rot_bias_scale)
                + squared_norm(ball_obs_vel_bias_base - nominal_vel_bias[None, :], vel_bias_scale)
                + squared_range(ball_obs_scale, self.cfg.dr_ball_obs_scale_range)
            )
        reset_disturbance_strength = jnp.sqrt(jnp.maximum(disturbance_sq, zero_strength))

        model = self._make_batched_model(
            n_envs=n_envs,
            dr_gravity_z=dr_gravity_z,
            dr_ball_mass=dr_ball_mass,
            dr_ball_friction=dr_ball_friction,
            dr_racket_friction=dr_racket_friction,
            dr_ball_solref_time=dr_ball_solref_time,
            dr_ball_solref_damping=dr_ball_solref_damping,
            dr_damping_mult=dr_damping_mult,
            dr_armature_mult=dr_armature_mult,
            dr_pd_kp_mult=dr_pd_kp_mult,
            dr_pd_kv_mult=dr_pd_kv_mult,
            dr_racket_pos_offset=dr_racket_pos_offset,
            dr_racket_rot_offset=dr_racket_rot_offset,
            dr_racket_radius_offset=dr_racket_radius_offset,
        )
        data = self.batched_forward(model, data)
        reset_racket_anchor = data.site_xpos[:, self.racket_site_id]
        episode_target_offset = jnp.stack([target_x_offset, target_y_offset, target_z_offset], axis=-1)
        episode_racket_anchor = reset_racket_anchor + episode_target_offset
        if self.waist_body_id >= 0:
            chest_target_offset = episode_racket_anchor - data.xpos[:, self.waist_body_id]
        else:
            chest_target_offset = jnp.zeros((n_envs, 3), dtype=jnp.float32)

        if falling_reset:
            contact_rel_height = (
                float(self.cfg.hit_confirm_rel_height)
                if float(self.cfg.falling_reset_contact_rel_height) < 0.0
                else float(self.cfg.falling_reset_contact_rel_height)
            )
            g_abs = jnp.maximum(jnp.abs(dr_gravity_z), 1e-6)
            min_downward_speed = max(0.0, float(self.cfg.falling_reset_min_downward_speed))
            apex_height = jnp.maximum(falling_apex_height, 1e-4)
            time_apex_to_contact = jnp.sqrt(2.0 * apex_height / g_abs)
            tau_upper = jnp.maximum(0.02, time_apex_to_contact - min_downward_speed / g_abs)
            tau_lower = jnp.minimum(jnp.full_like(tau_upper, max(0.0, tau_lo)), tau_upper)
            tau = jnp.minimum(jnp.maximum(falling_tau_raw, tau_lower), tau_upper)
            contact_xy = episode_racket_anchor[:, :2] + xy_jitter
            contact_z = episode_racket_anchor[:, 2] + contact_rel_height + z_jitter
            contact_vz = -jnp.sqrt(2.0 * g_abs * apex_height)
            init_vz = contact_vz + g_abs * tau
            ball_xy = contact_xy - vxy * tau[:, None]
            ball_z = contact_z - init_vz * tau + 0.5 * g_abs * tau * tau
            ball_init = jnp.concatenate([ball_xy, ball_z[:, None]], axis=-1)
            ball_init_vel = jnp.concatenate([vxy, init_vz[:, None]], axis=-1)
            reset_ball_racket_center_offset = jnp.linalg.norm(xy_jitter, axis=-1)
        elif racket_launch_reset:
            racket_xmat = data.geom_xmat[:, self.racket_geom_id].reshape((-1, 3, 3))
            tangent_x = racket_xmat[:, :, 0]
            tangent_y = racket_xmat[:, :, 1]
            normal_raw = racket_xmat[:, :, 2]
            normal_sign = jnp.where(normal_raw[:, 2] >= 0.0, 1.0, -1.0)
            racket_normal = normal_raw * normal_sign[:, None]
            racket_half_thickness = model.geom_size[:, self.racket_geom_id, 1]
            racket_radius = model.geom_size[:, self.racket_geom_id, 0]
            ball_radius = model.geom_size[:, self.ball_geom_id, 0]
            max_center_offset = jnp.maximum(
                0.0,
                racket_radius - ball_radius - float(self.cfg.racket_launch_edge_margin),
            )
            raw_offset_norm = jnp.linalg.norm(xy_jitter, axis=-1)
            offset_scale = jnp.minimum(
                1.0,
                max_center_offset / jnp.maximum(raw_offset_norm, 1e-8),
            )
            launch_offset_uv = xy_jitter * offset_scale[:, None]
            launch_tangent_offset = (
                tangent_x * launch_offset_uv[:, 0:1]
                + tangent_y * launch_offset_uv[:, 1:2]
            )
            racket_surface_center = (
                data.geom_xpos[:, self.racket_geom_id]
                + racket_normal * racket_half_thickness[:, None]
            )
            ball_init = (
                racket_surface_center
                + racket_normal * (ball_radius + racket_launch_surface_gap)[:, None]
                + launch_tangent_offset
            )
            ball_init_vel = (
                tangent_x * vxy[:, 0:1]
                + tangent_y * vxy[:, 1:2]
                + racket_normal * init_vz[:, None]
            )
            reset_ball_racket_center_offset = jnp.linalg.norm(launch_offset_uv, axis=-1)
        else:
            ball_init = jnp.concatenate(
                [
                    episode_racket_anchor[:, :2] + xy_jitter,
                    (episode_racket_anchor[:, 2] + float(self.cfg.ball_launch_height) + z_jitter)[:, None],
                ],
                axis=-1,
            )
            ball_init_vel = jnp.concatenate([vxy, init_vz[:, None]], axis=-1)
            reset_ball_racket_center_offset = jnp.linalg.norm(xy_jitter, axis=-1)
        qpos = data.qpos
        qvel = data.qvel
        qpos = qpos.at[:, self.ball_qadr : self.ball_qadr + 3].set(ball_init)
        qpos = qpos.at[:, self.ball_qadr + 3 : self.ball_qadr + 7].set(
            jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (n_envs, 4))
        )
        qvel = qvel.at[:, self.ball_vadr : self.ball_vadr + 6].set(0.0)
        qvel = qvel.at[:, self.ball_vadr : self.ball_vadr + 3].set(ball_init_vel)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.broadcast_to(self.default_ctrl, (n_envs, self.mj_model.nu)))
        data = self.batched_forward(model, data)

        bpos = data.xpos[:, self.ball_body_id]
        rpos = data.site_xpos[:, self.racket_site_id]

        camera_missing_u = jax.vmap(
            lambda key: jax.random.uniform(jax.random.fold_in(key, 1908))
        )(next_keys)
        camera_missing_enabled = (
            camera_missing_u < float(self.cfg.ball_obs_camera_missing_prob)
        )
        view_bounds_missing_u = jax.vmap(
            lambda key: jax.random.uniform(jax.random.fold_in(key, 1909))
        )(next_keys)
        view_bounds_missing_enabled = (
            view_bounds_missing_u < float(self.cfg.ball_obs_view_bounds_missing_prob)
        )
        coherent_missing_u = jax.vmap(
            lambda key: jax.random.uniform(jax.random.fold_in(key, 1910))
        )(next_keys)
        coherent_missing_enabled = (
            coherent_missing_u
            < float(self.cfg.ball_obs_missing_episode_coherent_prob)
        )

        reset_ball_obs_missing = jnp.zeros((n_envs,), dtype=bool)
        reset_ball_obs_pos = bpos
        reset_ball_obs_age = jnp.zeros((n_envs,), dtype=jnp.float32)
        reset_last_ball_obs_step = jnp.zeros((n_envs,), dtype=jnp.int32)
        if bool(
            self.cfg.ball_obs_reset_respects_camera_visibility
            and self.cfg.ball_obs_require_camera_visible
            and self.cfg.camera_visibility_mode != "off"
        ):
            reset_camera_terms = self._camera_reward_terms(data, bpos)
            reset_camera_visible = reset_camera_terms["metric/camera_visible"] > 0.5
            reset_camera_u = jax.vmap(
                lambda key: jax.random.uniform(jax.random.fold_in(key, 1907))
            )(next_keys)
            reset_frame_missing = (~reset_camera_visible) & (
                reset_camera_u < float(self.cfg.ball_obs_camera_missing_prob)
            )
            reset_coherent_missing = (
                (~reset_camera_visible) & camera_missing_enabled
            )
            reset_ball_obs_missing = jnp.where(
                coherent_missing_enabled,
                reset_coherent_missing,
                reset_frame_missing,
            )
            reset_missing_age = min(
                float(self.cfg.ball_obs_age_clip),
                max(
                    self.dt,
                    max(0.0, float(self.cfg.lost_ball_timeout_ms)) * 1e-3,
                ),
            )
            reset_ball_obs_pos = jnp.where(reset_ball_obs_missing[:, None], rpos, bpos)
            reset_ball_obs_age = jnp.where(
                reset_ball_obs_missing,
                reset_missing_age,
                0.0,
            )
            reset_last_ball_obs_step = jnp.where(
                reset_ball_obs_missing,
                -int(np.ceil(reset_missing_age / max(self.dt, 1e-6))),
                0,
            ).astype(jnp.int32)
        state = EnvState(
            model=model,
            data=data,
            rng=next_keys,
            step_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            racket_anchor=episode_racket_anchor,
            chest_target_offset=chest_target_offset,
            reset_ball_pos=ball_init,
            reset_ball_vel=ball_init_vel,
            reset_target_offset=episode_target_offset,
            reset_disturbance_strength=reset_disturbance_strength,
            reset_ball_surface_gap=racket_launch_surface_gap,
            reset_ball_racket_center_offset=reset_ball_racket_center_offset,
            reset_ball_obs_missing=reset_ball_obs_missing,
            arm_cmd_q=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_cmd_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            arm_q_ref_latest=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_q_ref_active=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_actuator_q_ref_latest=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_actuator_q_ref_active=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_safe_q_ref_latest=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_safe_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            arm_safe_qacc=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            ball_obs_missing_episode_coherent_enabled=coherent_missing_enabled,
            ball_obs_camera_missing_enabled=camera_missing_enabled,
            ball_obs_view_bounds_missing_enabled=view_bounds_missing_enabled,
            arm_applied_q=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_applied_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            prev_action=zero_action,
            prev_arm_qvel=jnp.broadcast_to(self.warm_arm_qvel, (n_envs, self.act_dim)),
            prev_ball_pos=bpos,
            prev_racket_pos=rpos,
            prev_contact=jnp.zeros((n_envs,), dtype=bool),
            hit_armed=jnp.ones((n_envs,), dtype=bool),
            no_contact_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            contact_hold_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            pending_hit=jnp.zeros((n_envs,), dtype=bool),
            pending_hit_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            pending_hit_camera_visible=jnp.zeros((n_envs,), dtype=bool),
            pending_hit_camera_in_margin=jnp.zeros((n_envs,), dtype=bool),
            pending_hit_camera_in_lower_band=jnp.zeros((n_envs,), dtype=bool),
            pending_hit_camera_v_frac=jnp.zeros((n_envs,), dtype=jnp.float32),
            hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            action_buffer=jnp.zeros((n_envs, self.max_action_latency_steps + 1, self.act_dim), dtype=jnp.float32),
            action_latency_steps=action_latency_steps,
            command_buffer=jnp.broadcast_to(
                self.warm_arm_q,
                (n_envs, self.command_buffer_len, self.act_dim),
            ),
            actuator_command_buffer=jnp.broadcast_to(
                self.warm_arm_q,
                (n_envs, self.command_buffer_len, self.act_dim),
            ),
            tau_act_episode=tau_act_episode,
            tau_act=tau_act,
            delay_steps=delay_steps,
            delay_bin_id=delay_bin_id,
            anti_windup_scale=jnp.ones((n_envs,), dtype=jnp.float32),
            obs_buffer=jnp.zeros((n_envs, self.max_obs_latency_steps + 1, self.base_obs_dim), dtype=jnp.float32),
            obs_latency_steps=obs_latency_steps,
            obs_history=jnp.zeros(
                (n_envs, self.high_latency_obs_prev_frames, self.base_obs_dim),
                dtype=jnp.float32,
            ),
            action_history=jnp.zeros(
                (n_envs, self.high_latency_action_prev_frames, self.act_dim),
                dtype=jnp.float32,
            ),
            cached_ball_obs_pos=reset_ball_obs_pos,
            cached_ball_obs_vel=zero_ball_vel,
            last_ball_obs_step=reset_last_ball_obs_step,
            ball_obs_valid_pos=reset_ball_obs_pos,
            ball_obs_valid_vel=zero_ball_vel,
            ball_obs_age_seconds=reset_ball_obs_age,
            ball_obs_missing_since_sample=reset_ball_obs_missing,
            ball_obs_dropout_remaining=jnp.zeros((n_envs,), dtype=jnp.int32),
            ball_obs_dropout_steps_total=jnp.zeros((n_envs,), dtype=jnp.int32),
            ball_obs_burst_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            total_env_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            action_scale_mult=action_scale_mult,
            actuator_cmd_tau=actuator_cmd_tau,
            actuator_cmd_gain=actuator_cmd_gain,
            dr_gravity_z=dr_gravity_z,
            dr_ball_mass=dr_ball_mass,
            dr_ball_friction=dr_ball_friction,
            dr_racket_friction=dr_racket_friction,
            dr_ball_solref_time=dr_ball_solref_time,
            dr_ball_solref_damping=dr_ball_solref_damping,
            dr_damping_mult=dr_damping_mult,
            dr_armature_mult=dr_armature_mult,
            dr_pd_kp_mult=dr_pd_kp_mult,
            dr_pd_kv_mult=dr_pd_kv_mult,
            last_hit_time=jnp.full((n_envs,), -1.0, dtype=jnp.float32),
            last_counted_hit_time=jnp.full((n_envs,), -1.0, dtype=jnp.float32),
            last_count_gate_hit_time=jnp.full((n_envs,), -1.0, dtype=jnp.float32),
            confirmed_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            ignored_fast_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            rewarded_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            unrewarded_extra_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            dr_racket_pos_offset=dr_racket_pos_offset,
            dr_racket_rot_offset=dr_racket_rot_offset,
            dr_racket_radius_offset=dr_racket_radius_offset,
            ball_obs_pos_bias_base=ball_obs_pos_bias_base,
            ball_obs_rot_bias_rpy=ball_obs_rot_bias_rpy,
            ball_obs_vel_bias_base=ball_obs_vel_bias_base,
            ball_obs_scale=ball_obs_scale,
            ball_obs_view_z_high_m=ball_obs_view_z_high_m,
        )
        base_obs = self._make_obs(
            state,
            state.ball_obs_valid_pos,
            state.ball_obs_valid_vel,
            state.ball_obs_age_seconds,
        )
        if self.high_latency_obs_prev_frames > 0:
            obs_history = jnp.broadcast_to(
                base_obs[:, None, :],
                (n_envs, self.high_latency_obs_prev_frames, self.base_obs_dim),
            )
        else:
            obs_history = jnp.zeros((n_envs, 0, self.base_obs_dim), dtype=jnp.float32)
        if self.high_latency_action_prev_frames > 0:
            action_history = jnp.zeros(
                (n_envs, self.high_latency_action_prev_frames, self.act_dim),
                dtype=jnp.float32,
            )
        else:
            action_history = jnp.zeros((n_envs, 0, self.act_dim), dtype=jnp.float32)
        state = state._replace(
            obs_buffer=jnp.broadcast_to(
                base_obs[:, None, :],
                (n_envs, self.max_obs_latency_steps + 1, self.base_obs_dim),
            ),
            obs_history=obs_history,
            action_history=action_history,
        )
        obs = self._augment_obs(state, base_obs)
        return state, obs

    def observe(self, state: EnvState) -> jax.Array:
        base_obs = self._make_obs(
            state,
            state.ball_obs_valid_pos,
            state.ball_obs_valid_vel,
            state.ball_obs_age_seconds,
        )
        return self._augment_obs(state, base_obs)

    def get_critic_obs(self, state: EnvState, obs: jax.Array) -> jax.Array:
        if not self.asymmetric_critic:
            return obs

        data = state.data
        q = data.qpos[:, self.arm_qadr]
        base_q = jnp.stack(
            [
                data.qpos[:, self.base_x_qadr],
                data.qpos[:, self.base_y_qadr],
                data.qpos[:, self.base_yaw_qadr],
            ],
            axis=-1,
        )
        base_dq = jnp.stack(
            [
                data.qvel[:, self.base_x_vadr],
                data.qvel[:, self.base_y_vadr],
                data.qvel[:, self.base_yaw_vadr],
            ],
            axis=-1,
        )
        true_bpos = data.xpos[:, self.ball_body_id]
        true_bvel = data.qvel[:, self.ball_vadr : self.ball_vadr + 3]
        rpos = data.site_xpos[:, self.racket_site_id]
        rvel = (rpos - state.prev_racket_pos) / max(self.dt, 1e-6)

        true_bpos_base = self._point_to_base(true_bpos, base_q)
        true_bvel_base = self._vel_to_base(true_bvel, true_bpos, base_q, base_dq)
        rpos_base = self._point_to_base(rpos, base_q)
        rvel_base = self._vel_to_base(rvel, rpos, base_q, base_dq)
        rel_base = true_bpos_base - rpos_base

        delay_den = float(max(1, self.command_buffer_len - 1))
        act_lat_den = float(max(1, self.max_action_latency_steps))
        obs_lat_den = float(max(1, self.max_obs_latency_steps))
        ball_mass_den = max(1e-6, float(self.original_ball_mass))
        ball_fric_den = max(1e-6, float(self.original_ball_friction))
        racket_fric_den = max(1e-6, float(self.original_racket_friction))
        solref_time_den = max(1e-6, float(self.original_ball_solref_time))
        solref_damping_den = max(1e-6, float(self.original_ball_solref_damping))
        gravity_den = max(1e-6, abs(float(self.default_gravity_z)))
        tau_den = max(1e-6, self.delay_max_s)

        scalar_features = jnp.concatenate(
            [
                state.anti_windup_scale[:, None],
                state.tau_act[:, None] / tau_den,
                state.delay_steps.astype(jnp.float32)[:, None] / delay_den,
                state.action_latency_steps.astype(jnp.float32)[:, None] / act_lat_den,
                state.obs_latency_steps.astype(jnp.float32)[:, None] / obs_lat_den,
                state.actuator_cmd_tau[:, None] / tau_den,
                state.actuator_cmd_gain[:, None] - 1.0,
                state.action_scale_mult[:, None] - 1.0,
                state.dr_gravity_z[:, None] / gravity_den,
                state.dr_ball_mass[:, None] / ball_mass_den - 1.0,
                state.dr_ball_friction[:, None] / ball_fric_den - 1.0,
                state.dr_racket_friction[:, None] / racket_fric_den - 1.0,
                state.dr_ball_solref_time[:, None] / solref_time_den - 1.0,
                state.dr_ball_solref_damping[:, None] / solref_damping_den - 1.0,
                state.dr_damping_mult[:, None] - 1.0,
                state.dr_armature_mult[:, None] - 1.0,
                state.dr_racket_radius_offset[:, None],
            ],
            axis=-1,
        )
        vector_features = [
            true_bpos_base,
            true_bvel_base,
            rpos_base,
            rvel_base,
            rel_base,
            state.arm_cmd_q - q,
            state.arm_q_ref_active - q,
            state.arm_applied_q - q,
            state.arm_cmd_qvel,
            state.dr_pd_kp_mult - 1.0,
            state.dr_pd_kv_mult - 1.0,
            state.dr_racket_pos_offset,
            state.dr_racket_rot_offset,
            scalar_features,
        ]
        if self.critic_command_history_steps > 0:
            command_hist = state.command_buffer[:, -self.critic_command_history_steps :, :] - q[:, None, :]
            vector_features.append(command_hist.reshape((obs.shape[0], -1)))
        return jnp.concatenate([obs, *vector_features], axis=-1)

    def _make_obs(
        self,
        state: EnvState,
        ball_obs_pos: jax.Array,
        ball_obs_vel: jax.Array,
        ball_obs_age_seconds: jax.Array,
    ) -> jax.Array:
        data = state.data
        q = data.qpos[:, self.arm_qadr]
        dq = data.qvel[:, self.arm_vadr]
        base_q = jnp.stack(
            [
                data.qpos[:, self.base_x_qadr],
                data.qpos[:, self.base_y_qadr],
                data.qpos[:, self.base_yaw_qadr],
            ],
            axis=-1,
        )
        base_dq = jnp.stack(
            [
                data.qvel[:, self.base_x_vadr],
                data.qvel[:, self.base_y_vadr],
                data.qvel[:, self.base_yaw_vadr],
            ],
            axis=-1,
        )
        rpos = data.site_xpos[:, self.racket_site_id]
        rvel = (rpos - state.prev_racket_pos) / max(self.dt, 1e-6)
        bpos_base = self._point_to_base(ball_obs_pos, base_q)
        rpos_base = self._point_to_base(rpos, base_q)
        bvel_base = self._vel_to_base(ball_obs_vel, ball_obs_pos, base_q, base_dq)
        rvel_base = self._vel_to_base(rvel, rpos, base_q, base_dq)
        bpos_base, bvel_base = self._apply_ball_obs_frame_bias(state, bpos_base, bvel_base)
        rel_base = bpos_base - rpos_base
        arm_cmd_error = state.arm_cmd_q - q
        age = jnp.clip(ball_obs_age_seconds / max(1e-6, float(self.cfg.ball_obs_age_clip)), 0.0, 1.0)[:, None]
        return jnp.concatenate(
            [
                q,
                dq,
                base_q,
                base_dq,
                bpos_base,
                bvel_base,
                rpos_base,
                rvel_base,
                rel_base,
                state.prev_action,
                arm_cmd_error,
                age,
            ],
            axis=-1,
        )

    def _augment_obs(self, state: EnvState, base_obs: jax.Array) -> jax.Array:
        obs = self._augment_high_latency_obs(state, base_obs) if self.high_latency_obs else base_obs
        if self.delay_extra_dim <= 0:
            return obs
        return jnp.concatenate([obs, self._delay_conditioning_features(state, base_obs)], axis=-1)

    def _augment_high_latency_obs(self, state: EnvState, base_obs: jax.Array) -> jax.Array:
        bpos_base = base_obs[:, 20:23]
        bvel_base = base_obs[:, 23:26]
        rpos_base = base_obs[:, 26:29]
        age_seconds = base_obs[:, 49:50] * float(self.cfg.ball_obs_age_clip)
        action_latency_sec = state.action_latency_steps.astype(jnp.float32)[:, None] * self.dt
        obs_latency_sec = state.obs_latency_steps.astype(jnp.float32)[:, None] * self.dt
        actuator_tau = state.actuator_cmd_tau[:, None]
        actuator_gain = state.actuator_cmd_gain[:, None]

        pred_time = action_latency_sec
        if bool(self.cfg.high_latency_prediction_include_obs_latency):
            pred_time = pred_time + obs_latency_sec
        if bool(self.cfg.high_latency_prediction_include_ball_age):
            pred_time = pred_time + age_seconds
        if bool(self.cfg.high_latency_prediction_include_actuator_tau):
            pred_time = pred_time + actuator_tau
        pred_time = jnp.clip(pred_time, 0.0, float(self.cfg.high_latency_prediction_time_clip))

        gravity = jnp.zeros_like(bvel_base)
        gravity = gravity.at[:, 2].set(state.dr_gravity_z)
        pred_bpos_base = bpos_base + bvel_base * pred_time + 0.5 * gravity * pred_time**2
        pred_bvel_base = bvel_base + gravity * pred_time
        pred_rel_base = pred_bpos_base - rpos_base

        pred_clip = max(1e-6, float(self.cfg.high_latency_prediction_time_clip))
        action_den = float(max(1, self.max_action_latency_steps))
        obs_den = float(max(1, self.max_obs_latency_steps))
        latency_features = jnp.concatenate(
            [
                state.action_latency_steps.astype(jnp.float32)[:, None] / action_den,
                action_latency_sec / pred_clip,
                state.obs_latency_steps.astype(jnp.float32)[:, None] / obs_den,
                obs_latency_sec / pred_clip,
                age_seconds / pred_clip,
                actuator_tau / pred_clip,
                actuator_gain - 1.0,
            ],
            axis=-1,
        )
        obs_hist = state.obs_history.reshape((base_obs.shape[0], -1))
        action_hist = state.action_history.reshape((base_obs.shape[0], -1))
        return jnp.concatenate(
            [
                base_obs,
                pred_bpos_base,
                pred_bvel_base,
                pred_rel_base,
                latency_features,
                obs_hist,
                action_hist,
            ],
            axis=-1,
        )

    def _delay_conditioning_features(self, state: EnvState, base_obs: jax.Array) -> jax.Array:
        n_envs = base_obs.shape[0]
        if not self.delay_conditioning or self.delay_extra_dim <= 0:
            return jnp.zeros((n_envs, 0), dtype=jnp.float32)

        features = []
        tau_norm = jnp.clip(state.tau_act / self.delay_max_s, 0.0, 1.5)[:, None]
        if bool(self.cfg.include_tau_act_norm):
            features.append(tau_norm)
        if bool(self.cfg.include_command_state):
            features.append(state.arm_cmd_qvel)
        if bool(self.cfg.include_active_command_error):
            q_real = base_obs[:, : self.act_dim]
            features.append(state.arm_q_ref_active - q_real)
        if bool(self.cfg.include_phase_features):
            t_contact_est = self._estimate_contact_time_from_obs(state, base_obs)
            t_margin = t_contact_est - state.tau_act
            features.append(t_contact_est[:, None])
            features.append(t_margin[:, None])
        if bool(self.cfg.use_delay_embedding) and int(self.cfg.delay_embedding_dim) > 0:
            dim = int(self.cfg.delay_embedding_dim)
            idx = jnp.arange(dim, dtype=jnp.float32)[None, :] + 1.0
            angles = tau_norm * idx * jnp.pi
            parity = (jnp.arange(dim)[None, :] % 2) == 0
            emb = jnp.where(parity, jnp.sin(angles), jnp.cos(angles))
            features.append(emb.astype(jnp.float32))
        if not features:
            return jnp.zeros((n_envs, 0), dtype=jnp.float32)
        return jnp.concatenate(features, axis=-1)

    def _estimate_contact_time_from_obs(self, state: EnvState, base_obs: jax.Array) -> jax.Array:
        age_seconds = base_obs[:, 49] * float(self.cfg.ball_obs_age_clip)
        ball_vz_stale = base_obs[:, 25]
        gravity_z = state.dr_gravity_z
        z_rel = (
            base_obs[:, 34]
            + ball_vz_stale * age_seconds
            + 0.5 * gravity_z * age_seconds * age_seconds
        )
        vz_rel = ball_vz_stale + gravity_z * age_seconds - base_obs[:, 31]
        return self._estimate_contact_time_from_z_vz(state, z_rel, vz_rel, age_seconds)

    def _estimate_contact_time_from_z_vz(
        self,
        state: EnvState,
        z_rel: jax.Array,
        vz_rel: jax.Array,
        age_seconds: jax.Array,
    ) -> jax.Array:
        g = jnp.maximum(jnp.abs(state.dr_gravity_z), 1e-6)
        max_t = float(self.cfg.max_contact_time)
        h = z_rel - float(self.cfg.contact_height_offset)
        disc = vz_rel * vz_rel + 2.0 * g * h
        root = jnp.sqrt(jnp.maximum(disc, 0.0))
        t1 = (vz_rel + root) / g
        t2 = (vz_rel - root) / g
        t1_ok = t1 >= 0.0
        t2_ok = t2 >= 0.0
        t = jnp.where(
            t1_ok & t2_ok,
            jnp.minimum(t1, t2),
            jnp.where(t1_ok, t1, jnp.where(t2_ok, t2, max_t)),
        )
        lost_timeout = max(0.0, float(self.cfg.lost_ball_timeout_ms)) * 1e-3
        lost = (lost_timeout > 0.0) & (age_seconds >= lost_timeout)
        invalid = (
            lost
            | (disc < 0.0)
            | (~jnp.isfinite(t))
            | (~jnp.isfinite(z_rel))
            | (~jnp.isfinite(vz_rel))
            | (jnp.abs(vz_rel) > 50.0)
            | (jnp.abs(z_rel) > 10.0)
        )
        t = jnp.where(invalid, max_t, t)
        return jnp.clip(t, 0.0, max_t)

    def _apply_ball_obs_frame_bias(
        self,
        state: EnvState,
        bpos_base: jax.Array,
        bvel_base: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        rot = jax.vmap(_euler_xyz_to_mat_jax)(state.ball_obs_rot_bias_rpy)
        pivot_base = None
        if self.ball_obs_frame_pivot_mode == "camera_center":
            cam_pos, _, camera_available = self._virtual_camera_pose(state.data)
            if not camera_available:
                raise RuntimeError(
                    "camera-centered ball observation frame has no virtual camera pose"
                )
            base_q = jnp.stack(
                [
                    state.data.qpos[:, self.base_x_qadr],
                    state.data.qpos[:, self.base_y_qadr],
                    state.data.qpos[:, self.base_yaw_qadr],
                ],
                axis=-1,
            )
            pivot_base = self._point_to_base(cam_pos, base_q)
        return _apply_ball_obs_frame_transform(
            bpos_base,
            bvel_base,
            rot,
            state.ball_obs_scale,
            state.ball_obs_pos_bias_base,
            state.ball_obs_vel_bias_base,
            pivot_base,
        )

    def step(self, state: EnvState, action: jax.Array) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        raw_policy_action = action
        policy_action = jnp.clip(raw_policy_action, -1.0, 1.0)
        action_clip_excess = jnp.maximum(jnp.abs(raw_policy_action) - 1.0, 0.0)
        action_buffer = jnp.concatenate([state.action_buffer[:, 1:, :], policy_action[:, None, :]], axis=1)
        action_idx = (self.max_action_latency_steps - state.action_latency_steps).astype(jnp.int32)
        delayed_policy_action = action_buffer[jnp.arange(action_buffer.shape[0]), action_idx]
        a_raw = policy_action if self.delay_conditioning else delayed_policy_action

        if self.delay_conditioning:
            split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(state.rng)
            rng_after_delay = split_keys[:, 0]
            key_delay_jitter = split_keys[:, 1]
            jitter_ms = float(self.cfg.delay_jitter_ms)
            if abs(jitter_ms) > 0.0:
                jitter = jax.vmap(
                    lambda k: jax.random.uniform(k, (), minval=-abs(jitter_ms), maxval=abs(jitter_ms))
                )(key_delay_jitter)
            else:
                jitter = jnp.zeros_like(state.tau_act_episode)
            lo_ms = min(float(self.cfg.delay_min_ms), float(self.cfg.delay_max_ms))
            hi_ms = max(float(self.cfg.delay_min_ms), float(self.cfg.delay_max_ms))
            tau_act = jnp.clip(state.tau_act_episode + jitter * 1e-3, lo_ms * 1e-3, hi_ms * 1e-3)
            delay_steps = jnp.rint(tau_act / max(self.dt, 1e-9)).astype(jnp.int32)
            delay_steps = jnp.clip(delay_steps, 0, self.command_buffer_len - 1)
        else:
            rng_after_delay = state.rng
            tau_act = state.tau_act
            delay_steps = state.delay_steps

        action_acc_limit = float(self.cfg.action_acc_limit)
        if action_acc_limit > 0.0:
            a_clip = jnp.clip(a_raw, -action_acc_limit, action_acc_limit)
        else:
            a_clip = a_raw

        filter_tau_s = max(0.0, float(self.cfg.action_filter_tau_ms)) * 1e-3
        alpha = 1.0 if filter_tau_s <= 1e-9 else self.dt / (filter_tau_s + self.dt)
        a_lpf = alpha * a_clip + (1.0 - alpha) * state.prev_action

        jerk_limit = float(self.cfg.action_jerk_limit)
        if jerk_limit > 0.0:
            max_delta = jerk_limit * self.dt
            a_final = state.prev_action + jnp.clip(a_lpf - state.prev_action, -max_delta, max_delta)
        else:
            a_final = a_lpf

        q_real_for_aw = state.data.qpos[:, self.arm_qadr]
        e_active_for_aw = state.arm_q_ref_active - q_real_for_aw
        if bool(self.cfg.enable_anti_windup):
            threshold = max(1e-9, float(self.cfg.anti_windup_error_threshold))
            min_scale = min(1.0, max(0.0, float(self.cfg.anti_windup_min_scale)))
            anti_windup_scale = jnp.clip(
                1.0 - jnp.linalg.norm(e_active_for_aw, axis=-1) / threshold,
                min_scale,
                1.0,
            )
            action = a_final * anti_windup_scale[:, None]
        else:
            anti_windup_scale = jnp.ones((policy_action.shape[0],), dtype=jnp.float32)
            action = a_final
        da = action - state.prev_action

        desired_qdd_raw = action * self.arm_acc_limit_rad_s2 * float(self.cfg.action_acc_scale) * state.action_scale_mult[:, None]
        if bool(self.cfg.arm_action_limiter):
            desired_qdd = jnp.clip(desired_qdd_raw, -self.arm_acc_limit_rad_s2, self.arm_acc_limit_rad_s2)
        else:
            desired_qdd = desired_qdd_raw
        raw_cmd_qvel = state.arm_cmd_qvel + desired_qdd * self.dt
        if bool(self.cfg.arm_action_limiter):
            cmd_qvel = jnp.clip(raw_cmd_qvel, -self.arm_vel_limit_rad_s, self.arm_vel_limit_rad_s)
        else:
            cmd_qvel = raw_cmd_qvel
        arm_cmd_q = jnp.clip(state.arm_cmd_q + cmd_qvel * self.dt, self.arm_lo, self.arm_hi)

        arm_q_ref_latest = arm_cmd_q
        comp_mode = str(self.cfg.actuator_compensation_mode or "none").strip().lower().replace("-", "_")
        if bool(self.cfg.actuator_lead_compensation) and comp_mode in {"none", "off", "false", "0"}:
            comp_mode = "lead"
        bridger_mode = comp_mode in {"sim2real_bridger", "constrained_inverse_mpc", "bridger"}
        if comp_mode in {"inverse_mpc", "regularized_inverse_mpc", "mpc"} or bridger_mode:
            comp_delay_steps = jnp.rint(
                delay_steps.astype(jnp.float32) * max(0.0, float(self.cfg.actuator_mpc_delay_scale))
            ).astype(jnp.int32)
            comp_delay_steps = jnp.clip(comp_delay_steps, 0, self.command_buffer_len - 1)
            tau_est = jnp.maximum(state.actuator_cmd_tau * max(0.0, float(self.cfg.actuator_mpc_tau_scale)), 0.0)
            alpha_est = jnp.where(tau_est <= 1e-6, 1.0, self.dt / (tau_est + self.dt))
            pred_buffer = jnp.concatenate(
                [state.actuator_command_buffer[:, 1:, :], arm_cmd_q[:, None, :]],
                axis=1,
            )
            mpc_feedback_source = str(self.cfg.actuator_mpc_feedback_source or "applied").strip().lower().replace("-", "_")
            if mpc_feedback_source in {"actual", "sim", "joint", "joint_state"}:
                y_pred = state.data.qpos[:, self.arm_qadr]
            elif mpc_feedback_source in {"applied", "actuator", "servo", "servo_target"}:
                y_pred = state.arm_applied_q
            else:
                raise ValueError(
                    "actuator_mpc_feedback_source must be 'applied' or 'actual'"
                )
            for s in range(self.command_buffer_len - 1):
                idx = jnp.clip(self.command_buffer_len - 1 - comp_delay_steps + s, 0, self.command_buffer_len - 1)
                queued = pred_buffer[jnp.arange(pred_buffer.shape[0]), idx]
                filter_target = self.warm_arm_q[None, :] + state.actuator_cmd_gain[:, None] * (
                    queued - self.warm_arm_q[None, :]
                )
                y_next = y_pred + alpha_est[:, None] * (filter_target - y_pred)
                y_pred = jnp.where((s < comp_delay_steps)[:, None], y_next, y_pred)

            mpc_horizon_steps = max(1, int(self.cfg.actuator_mpc_horizon_steps))
            total_horizon = (comp_delay_steps.astype(jnp.float32) + float(mpc_horizon_steps)) * self.dt
            target_future = (
                arm_cmd_q
                + total_horizon[:, None] * cmd_qvel
                + 0.5 * total_horizon[:, None] * total_horizon[:, None] * desired_qdd
            )
            decay = jnp.power(jnp.clip(1.0 - alpha_est, 0.0, 1.0), float(mpc_horizon_steps))
            response = 1.0 - decay
            gain_est = jnp.where(jnp.abs(state.actuator_cmd_gain) <= 1e-6, 1.0, state.actuator_cmd_gain)
            k = response * gain_est
            b = decay[:, None] * y_pred + response[:, None] * (1.0 - gain_est[:, None]) * self.warm_arm_q[None, :]
            last_actuator_cmd = state.actuator_command_buffer[:, -1, :]
            wt = max(0.0, float(self.cfg.actuator_mpc_tracking_weight))
            wn = max(0.0, float(self.cfg.actuator_mpc_nominal_weight))
            wd = max(0.0, float(self.cfg.actuator_mpc_delta_weight))
            denom = wt * k[:, None] * k[:, None] + wn + wd
            mpc_cmd = (wt * k[:, None] * (target_future - b) + wn * arm_cmd_q + wd * last_actuator_cmd) / jnp.maximum(
                denom,
                1e-6,
            )
            mpc_delta = float(self.cfg.actuator_mpc_beta) * (mpc_cmd - arm_cmd_q)
            max_mpc = float(self.cfg.actuator_mpc_max_delta_rad)
            if max_mpc > 0.0:
                mpc_delta = jnp.clip(mpc_delta, -max_mpc, max_mpc)
            arm_actuator_q_ref_latest = jnp.clip(arm_cmd_q + mpc_delta, self.arm_lo, self.arm_hi)
            mpc_command_velocity_scale = float(self.cfg.actuator_mpc_command_velocity_scale)
            mpc_command_acceleration_scale = float(self.cfg.actuator_mpc_command_acceleration_scale)
            if not (0.0 < mpc_command_velocity_scale <= 1.0):
                raise ValueError("actuator_mpc_command_velocity_scale must be in (0, 1]")
            if not (0.0 < mpc_command_acceleration_scale <= 1.0):
                raise ValueError("actuator_mpc_command_acceleration_scale must be in (0, 1]")
            mpc_command_velocity_limit = self.arm_vel_limit_rad_s * mpc_command_velocity_scale
            mpc_command_acceleration_limit = self.arm_acc_limit_rad_s2 * mpc_command_acceleration_scale
            mpc_velocity_weight = max(0.0, float(self.cfg.actuator_mpc_command_velocity_weight))
            mpc_acceleration_weight = max(0.0, float(self.cfg.actuator_mpc_command_acceleration_weight))
            if mpc_velocity_weight > 0.0 or mpc_acceleration_weight > 0.0:
                refined_num = arm_actuator_q_ref_latest
                refined_den = jnp.ones_like(arm_actuator_q_ref_latest)
                if mpc_velocity_weight > 0.0:
                    vel_scale = jnp.maximum(mpc_command_velocity_limit * self.dt, 1e-6)
                    vel_weight = mpc_velocity_weight / (vel_scale * vel_scale)
                    refined_num = refined_num + vel_weight[None, :] * state.arm_safe_q_ref_latest
                    refined_den = refined_den + vel_weight[None, :]
                if mpc_acceleration_weight > 0.0:
                    acc_center = state.arm_safe_q_ref_latest + state.arm_safe_qvel * self.dt
                    acc_scale = jnp.maximum(mpc_command_acceleration_limit * self.dt * self.dt, 1e-6)
                    acc_weight = mpc_acceleration_weight / (acc_scale * acc_scale)
                    refined_num = refined_num + acc_weight[None, :] * acc_center
                    refined_den = refined_den + acc_weight[None, :]
                arm_actuator_q_ref_latest = refined_num / jnp.maximum(refined_den, 1e-9)
            if bool(self.cfg.actuator_mpc_command_dynamics_constraint):
                command_low = self.arm_lo
                command_high = self.arm_hi
                command_low = jnp.maximum(
                    command_low[None, :],
                    state.arm_safe_q_ref_latest - mpc_command_velocity_limit[None, :] * self.dt,
                )
                command_high = jnp.minimum(
                    command_high[None, :],
                    state.arm_safe_q_ref_latest + mpc_command_velocity_limit[None, :] * self.dt,
                )
                command_acc_center = state.arm_safe_q_ref_latest + state.arm_safe_qvel * self.dt
                command_low = jnp.maximum(
                    command_low,
                    command_acc_center - mpc_command_acceleration_limit[None, :] * self.dt * self.dt,
                )
                command_high = jnp.minimum(
                    command_high,
                    command_acc_center + mpc_command_acceleration_limit[None, :] * self.dt * self.dt,
                )
                command_mid = 0.5 * (command_low + command_high)
                command_safe_low = jnp.minimum(command_low, command_mid)
                command_safe_high = jnp.maximum(command_high, command_mid)
                arm_actuator_q_ref_latest = jnp.minimum(
                    jnp.maximum(arm_actuator_q_ref_latest, command_safe_low),
                    command_safe_high,
                )
            arm_actuator_q_ref_latest = jnp.clip(arm_actuator_q_ref_latest, self.arm_lo, self.arm_hi)
            if bridger_mode:
                bridger_target_qvel = cmd_qvel
                (
                    arm_actuator_q_ref_latest,
                    bridger_qvel,
                    bridger_qacc,
                    bridger_feasible,
                    bridger_jerk_feasible,
                    bridger_interval_low,
                    bridger_interval_high,
                ) = constrained_compensation_step_jax(
                    arm_actuator_q_ref_latest,
                    bridger_target_qvel,
                    state.arm_safe_q_ref_latest,
                    state.arm_safe_qvel,
                    state.arm_safe_qacc,
                    self.arm_lo,
                    self.arm_hi,
                    self.arm_vel_limit_rad_s,
                    self.arm_acc_limit_rad_s2,
                    self.arm_bridger_jerk_limit_rad_s3,
                    dt=self.dt,
                    natural_frequency_hz=float(
                        self.cfg.actuator_bridger_natural_frequency_hz
                    ),
                    damping_ratio=float(self.cfg.actuator_bridger_damping_ratio),
                    target_qacc=desired_qdd,
                )
        elif comp_mode in {"inverse_smith", "smith", "inverse"}:
            comp_delay_steps = jnp.rint(
                delay_steps.astype(jnp.float32) * max(0.0, float(self.cfg.actuator_inverse_delay_scale))
            ).astype(jnp.int32)
            comp_delay_steps = jnp.clip(comp_delay_steps, 0, self.command_buffer_len - 1)
            tau_est = jnp.maximum(state.actuator_cmd_tau * max(0.0, float(self.cfg.actuator_inverse_tau_scale)), 0.0)
            alpha_est = jnp.where(tau_est <= 1e-6, 1.0, self.dt / (tau_est + self.dt))
            pred_buffer = jnp.concatenate(
                [state.actuator_command_buffer[:, 1:, :], arm_cmd_q[:, None, :]],
                axis=1,
            )
            y_pred = state.arm_applied_q
            for s in range(self.command_buffer_len - 1):
                idx = jnp.clip(self.command_buffer_len - 1 - comp_delay_steps + s, 0, self.command_buffer_len - 1)
                queued = pred_buffer[jnp.arange(pred_buffer.shape[0]), idx]
                filter_target = self.warm_arm_q[None, :] + state.actuator_cmd_gain[:, None] * (
                    queued - self.warm_arm_q[None, :]
                )
                y_next = y_pred + alpha_est[:, None] * (filter_target - y_pred)
                y_pred = jnp.where((s < comp_delay_steps)[:, None], y_next, y_pred)

            horizon = comp_delay_steps.astype(jnp.float32) * self.dt
            target_future = arm_cmd_q + horizon[:, None] * cmd_qvel + 0.5 * horizon[:, None] * horizon[:, None] * desired_qdd
            inv_filter_target = (target_future - (1.0 - alpha_est[:, None]) * y_pred) / jnp.maximum(
                alpha_est[:, None],
                1e-6,
            )
            gain_est = jnp.where(jnp.abs(state.actuator_cmd_gain) <= 1e-6, 1.0, state.actuator_cmd_gain)
            inv_cmd = self.warm_arm_q[None, :] + (inv_filter_target - self.warm_arm_q[None, :]) / gain_est[:, None]
            inverse_delta = float(self.cfg.actuator_inverse_beta) * (inv_cmd - arm_cmd_q)
            max_inverse = float(self.cfg.actuator_inverse_max_delta_rad)
            if max_inverse > 0.0:
                inverse_delta = jnp.clip(inverse_delta, -max_inverse, max_inverse)
            arm_actuator_q_ref_latest = jnp.clip(
                arm_cmd_q + inverse_delta,
                self.arm_lo,
                self.arm_hi,
            )
        elif comp_mode == "lead":
            delay_s = delay_steps.astype(jnp.float32) * self.dt
            tau_s = jnp.maximum(state.actuator_cmd_tau, 0.0)
            lead_time = (
                max(0.0, float(self.cfg.actuator_lead_delay_scale)) * delay_s
                + max(0.0, float(self.cfg.actuator_lead_tau_scale)) * tau_s
            )
            lead_delta = float(self.cfg.actuator_lead_beta) * (
                lead_time[:, None] * cmd_qvel
                + 0.5 * lead_time[:, None] * lead_time[:, None] * desired_qdd
            )
            max_lead = float(self.cfg.actuator_lead_max_delta_rad)
            if max_lead > 0.0:
                lead_delta = jnp.clip(lead_delta, -max_lead, max_lead)
            arm_actuator_q_ref_latest = jnp.clip(arm_cmd_q + lead_delta, self.arm_lo, self.arm_hi)
        else:
            arm_actuator_q_ref_latest = arm_q_ref_latest

        if bridger_mode:
            arm_safe_q_ref_latest = arm_actuator_q_ref_latest
            arm_safe_qvel = bridger_qvel
            arm_safe_qacc = bridger_qacc
            arm_safe_interval_low = bridger_interval_low
            arm_safe_interval_high = bridger_interval_high
            arm_safe_feasible = bridger_feasible & jnp.all(bridger_jerk_feasible, axis=-1)
        elif bool(self.cfg.arm_post_compensation_limiter):
            (
                arm_safe_q_ref_latest,
                arm_safe_qvel,
                arm_safe_interval_low,
                arm_safe_interval_high,
                arm_safe_feasible,
            ) = project_safe_command_step_jax(
                arm_actuator_q_ref_latest,
                state.arm_safe_q_ref_latest,
                state.arm_safe_qvel,
                self.arm_lo,
                self.arm_hi,
                self.arm_vel_limit_rad_s,
                self.arm_acc_limit_rad_s2,
                self.dt,
            )
        else:
            arm_safe_q_ref_latest = arm_actuator_q_ref_latest
            arm_safe_qvel = (
                arm_safe_q_ref_latest - state.arm_safe_q_ref_latest
            ) / max(self.dt, 1e-6)
            arm_safe_interval_low = -jnp.broadcast_to(
                self.arm_vel_limit_rad_s,
                arm_safe_qvel.shape,
            )
            arm_safe_interval_high = jnp.broadcast_to(
                self.arm_vel_limit_rad_s,
                arm_safe_qvel.shape,
            )
            arm_safe_feasible = jnp.ones(
                (arm_safe_qvel.shape[0],),
                dtype=bool,
            )
        if not bridger_mode:
            arm_safe_qacc = (
                arm_safe_qvel - state.arm_safe_qvel
            ) / max(self.dt, 1e-6)
        arm_safe_clip = jnp.abs(
            arm_actuator_q_ref_latest - arm_safe_q_ref_latest
        ) > 1e-7

        if self.delay_conditioning:
            command_buffer = jnp.concatenate([state.command_buffer[:, 1:, :], arm_q_ref_latest[:, None, :]], axis=1)
            actuator_command_buffer = jnp.concatenate(
                [state.actuator_command_buffer[:, 1:, :], arm_safe_q_ref_latest[:, None, :]],
                axis=1,
            )
            active_idx = (command_buffer.shape[1] - 1 - delay_steps).astype(jnp.int32)
            arm_q_ref_active = command_buffer[jnp.arange(command_buffer.shape[0]), active_idx]
            arm_actuator_q_ref_active = actuator_command_buffer[jnp.arange(actuator_command_buffer.shape[0]), active_idx]
        else:
            command_buffer = state.command_buffer
            actuator_command_buffer = state.actuator_command_buffer
            arm_q_ref_active = arm_q_ref_latest
            arm_actuator_q_ref_active = arm_safe_q_ref_latest

        # The observation-only-delay ablation intentionally keeps
        # arm_*_q_ref_active on the nominal delayed history for the 67D actor
        # and asymmetric critic, but bypasses that delay on the simulated
        # plant.  No future reference is used: the servo receives only the
        # current command that was just produced by the causal action
        # integration path.
        arm_servo_q_ref_active = (
            arm_safe_q_ref_latest
            if self.actuator_delay_observation_only
            else arm_actuator_q_ref_active
        )
        servo_uses_delayed_reference = bool(
            self.delay_conditioning
            and not self.actuator_delay_observation_only
        )

        if bool(self.cfg.actuator_cmd_filter):
            tau = jnp.maximum(state.actuator_cmd_tau, 0.0)
            alpha = jnp.where(tau <= 1e-6, 1.0, self.dt / (tau + self.dt))
            gain_target_q = self.warm_arm_q[None, :] + state.actuator_cmd_gain[:, None] * (
                arm_servo_q_ref_active - self.warm_arm_q[None, :]
            )
            arm_servo_target_unlimited = jnp.clip(
                state.arm_applied_q + alpha[:, None] * (gain_target_q - state.arm_applied_q),
                self.arm_lo,
                self.arm_hi,
            )
        else:
            arm_servo_target_unlimited = arm_servo_q_ref_active
        servo_velocity_scale = float(self.cfg.arm_servo_target_velocity_scale)
        servo_acceleration_scale = float(self.cfg.arm_servo_target_acceleration_scale)
        if not (0.0 < servo_velocity_scale <= 1.0):
            raise ValueError("arm_servo_target_velocity_scale must be in (0, 1]")
        if not (0.0 < servo_acceleration_scale <= 1.0):
            raise ValueError("arm_servo_target_acceleration_scale must be in (0, 1]")
        servo_velocity_limit = self.arm_vel_limit_rad_s * servo_velocity_scale
        servo_acceleration_limit = self.arm_acc_limit_rad_s2 * servo_acceleration_scale
        if bool(self.cfg.arm_servo_target_tracking_planner):
            (
                arm_applied_q,
                arm_applied_qvel,
                arm_servo_interval_low,
                arm_servo_interval_high,
                arm_servo_feasible,
            ) = project_target_tracking_command_step_jax(
                arm_servo_target_unlimited,
                state.arm_applied_q,
                state.arm_applied_qvel,
                self.arm_lo,
                self.arm_hi,
                servo_velocity_limit,
                servo_acceleration_limit,
                self.dt,
            )
        elif bool(self.cfg.arm_servo_target_limiter):
            (
                arm_applied_q,
                arm_applied_qvel,
                arm_servo_interval_low,
                arm_servo_interval_high,
                arm_servo_feasible,
            ) = project_safe_command_step_jax(
                arm_servo_target_unlimited,
                state.arm_applied_q,
                state.arm_applied_qvel,
                self.arm_lo,
                self.arm_hi,
                servo_velocity_limit,
                servo_acceleration_limit,
                self.dt,
            )
        else:
            arm_applied_q = arm_servo_target_unlimited
            arm_applied_qvel = (
                arm_applied_q - state.arm_applied_q
            ) / max(self.dt, 1e-6)
            arm_servo_interval_low = -jnp.broadcast_to(
                self.arm_vel_limit_rad_s,
                arm_applied_qvel.shape,
            )
            arm_servo_interval_high = jnp.broadcast_to(
                self.arm_vel_limit_rad_s,
                arm_applied_qvel.shape,
            )
            arm_servo_feasible = jnp.ones(
                (arm_applied_qvel.shape[0],),
                dtype=bool,
            )
        arm_applied_qacc = (
            arm_applied_qvel - state.arm_applied_qvel
        ) / max(self.dt, 1e-6)
        arm_servo_clip = jnp.abs(
            arm_servo_target_unlimited - arm_applied_q
        ) > 1e-7

        acc_clip_diff = desired_qdd_raw - desired_qdd
        vel_clip_diff = raw_cmd_qvel - cmd_qvel
        arm_limiter_pen = jnp.mean(
            vel_clip_diff**2 / (self.arm_vel_limit_rad_s**2 + 1e-8)
            + acc_clip_diff**2 / (self.arm_acc_limit_rad_s2**2 + 1e-8),
            axis=-1,
        )

        ctrl = jnp.broadcast_to(self.default_ctrl, (action.shape[0], self.mj_model.nu))
        ctrl = ctrl.at[:, self.arm_aids_j].set(arm_applied_q)
        ctrl = ctrl.at[:, self.base_aids_j].set(0.0)
        data = state.data.replace(ctrl=ctrl)

        contact_init = jnp.zeros((action.shape[0],), dtype=bool)
        contact_camera_v_frac_init = jnp.zeros((action.shape[0],), dtype=jnp.float32)
        arm_actual_clip_count_init = jnp.zeros(
            (action.shape[0], self.act_dim),
            dtype=jnp.int32,
        )
        arm_actual_feasible_init = jnp.ones((action.shape[0],), dtype=bool)
        arm_actual_intervention_pen_init = jnp.zeros(
            (action.shape[0],),
            dtype=jnp.float32,
        )
        arm_actual_jerk_emergency_count_init = jnp.zeros(
            (action.shape[0], self.act_dim),
            dtype=jnp.int32,
        )
        arm_actual_velocity_utilization_max_init = jnp.zeros(
            (action.shape[0],), dtype=jnp.float32
        )
        arm_actual_acceleration_utilization_max_init = jnp.zeros(
            (action.shape[0],), dtype=jnp.float32
        )
        arm_actual_jerk_utilization_max_init = jnp.zeros(
            (action.shape[0],), dtype=jnp.float32
        )
        arm_actual_acceleration_saturation_count_init = jnp.zeros(
            (action.shape[0],), dtype=jnp.int32
        )
        arm_actual_high_acceleration_sign_flip_count_init = jnp.zeros(
            (action.shape[0],), dtype=jnp.int32
        )

        def one_substep(_, carry):
            (
                d,
                contact_any,
                other_ball_contact_any,
                first_contact_camera_visible,
                first_contact_camera_in_margin,
                first_contact_camera_v_frac,
                arm_actual_clip_count,
                arm_actual_feasible,
                arm_actual_intervention_pen,
                arm_actual_jerk_emergency_count,
                arm_actual_velocity_utilization_max,
                arm_actual_acceleration_utilization_max,
                arm_actual_jerk_utilization_max,
                arm_actual_acceleration_saturation_count,
                arm_actual_high_acceleration_sign_flip_count,
            ) = carry
            d_before = d
            d = self.batched_step(state.model, d.replace(ctrl=ctrl))
            if bool(self.cfg.arm_actual_state_limiter):
                arm_q_before = d_before.qpos[:, self.arm_qadr]
                arm_qvel_before = d_before.qvel[:, self.arm_vadr]
                arm_qpos_unlimited = d.qpos[:, self.arm_qadr]
                arm_qvel_unlimited = d.qvel[:, self.arm_vadr]
                arm_qacc_before = d_before.qacc[:, self.arm_vadr]
                if bool(self.cfg.arm_actual_target_tracking_governor):
                    (
                        arm_q_guarded,
                        arm_qvel_guarded,
                        arm_actual_interval_low,
                        arm_actual_interval_high,
                        substep_feasible,
                    ) = project_damped_target_tracking_command_step_jax(
                        arm_applied_q,
                        arm_q_before,
                        arm_qvel_before,
                        self.arm_lo,
                        self.arm_hi,
                        self.arm_vel_limit_rad_s,
                        self.arm_acc_limit_rad_s2,
                        self.timestep,
                        self.cfg.arm_actual_governor_natural_frequency_hz,
                        self.cfg.arm_actual_governor_damping_ratio,
                    )
                    jerk_dv = (
                        self.arm_actual_jerk_limit_rad_s3[None, :]
                        * self.timestep
                        * self.timestep
                    )
                    jerk_center_vel = (
                        arm_qvel_before + arm_qacc_before * self.timestep
                    )
                    governed_low = jnp.maximum(
                        arm_actual_interval_low,
                        jerk_center_vel - jerk_dv,
                    )
                    governed_high = jnp.minimum(
                        arm_actual_interval_high,
                        jerk_center_vel + jerk_dv,
                    )
                    jerk_feasible = governed_low <= governed_high
                    active_low = jnp.where(
                        jerk_feasible,
                        governed_low,
                        arm_actual_interval_low,
                    )
                    active_high = jnp.where(
                        jerk_feasible,
                        governed_high,
                        arm_actual_interval_high,
                    )
                    arm_qvel_guarded = jnp.minimum(
                        jnp.maximum(arm_qvel_guarded, active_low),
                        active_high,
                    )
                    arm_q_guarded = (
                        arm_q_before + arm_qvel_guarded * self.timestep
                    )
                    arm_actual_jerk_emergency_count = (
                        arm_actual_jerk_emergency_count
                        + (~jerk_feasible).astype(jnp.int32)
                    )
                else:
                    (
                        arm_q_guarded,
                        arm_qvel_guarded,
                        _arm_actual_interval_low,
                        _arm_actual_interval_high,
                        substep_feasible,
                    ) = project_safe_command_step_jax(
                        arm_qpos_unlimited,
                        arm_q_before,
                        arm_qvel_before,
                        self.arm_lo,
                        self.arm_hi,
                        self.arm_vel_limit_rad_s,
                        self.arm_acc_limit_rad_s2,
                        self.timestep,
                    )
                arm_actual_clip = (
                    (jnp.abs(arm_qvel_unlimited - arm_qvel_guarded) > 1e-7)
                    | (jnp.abs(arm_qpos_unlimited - arm_q_guarded) > 1e-7)
                )
                arm_q_out = jnp.where(
                    arm_actual_clip,
                    arm_q_guarded,
                    arm_qpos_unlimited,
                )
                arm_qvel_out = jnp.where(
                    arm_actual_clip,
                    arm_qvel_guarded,
                    arm_qvel_unlimited,
                )
                arm_qacc_out = (
                    arm_qvel_out - arm_qvel_before
                ) / max(self.timestep, 1e-9)
                arm_qacc_unlimited = (
                    arm_qvel_unlimited - arm_qvel_before
                ) / max(self.timestep, 1e-9)
                arm_velocity_utilization = jnp.max(
                    jnp.abs(arm_qvel_out)
                    / jnp.maximum(self.arm_vel_limit_rad_s[None, :], 1e-6),
                    axis=-1,
                )
                arm_acceleration_ratio = (
                    jnp.abs(arm_qacc_out)
                    / jnp.maximum(self.arm_acc_limit_rad_s2[None, :], 1e-6)
                )
                arm_acceleration_utilization = jnp.max(
                    arm_acceleration_ratio,
                    axis=-1,
                )
                arm_actual_velocity_utilization_max = jnp.maximum(
                    arm_actual_velocity_utilization_max,
                    arm_velocity_utilization,
                )
                arm_actual_acceleration_utilization_max = jnp.maximum(
                    arm_actual_acceleration_utilization_max,
                    arm_acceleration_utilization,
                )
                if bool(self.cfg.arm_actual_target_tracking_governor):
                    arm_jerk_utilization = jnp.max(
                        jnp.abs(arm_qacc_out - arm_qacc_before)
                        / jnp.maximum(
                            self.arm_actual_jerk_limit_rad_s3[None, :]
                            * self.timestep,
                            1e-6,
                        ),
                        axis=-1,
                    )
                    arm_actual_jerk_utilization_max = jnp.maximum(
                        arm_actual_jerk_utilization_max,
                        arm_jerk_utilization,
                    )
                    high_acceleration_sign_flip = (
                        (jnp.abs(arm_qacc_before) >= 0.5 * self.arm_acc_limit_rad_s2[None, :])
                        & (jnp.abs(arm_qacc_out) >= 0.5 * self.arm_acc_limit_rad_s2[None, :])
                        & (jnp.sign(arm_qacc_before) != jnp.sign(arm_qacc_out))
                    )
                    arm_actual_high_acceleration_sign_flip_count = (
                        arm_actual_high_acceleration_sign_flip_count
                        + jnp.sum(
                            high_acceleration_sign_flip.astype(jnp.int32),
                            axis=-1,
                        )
                    )
                arm_actual_acceleration_saturation_count = (
                    arm_actual_acceleration_saturation_count
                    + jnp.sum(
                        (arm_acceleration_ratio >= 0.99).astype(jnp.int32),
                        axis=-1,
                    )
                )
                arm_qacc_excess_ratio = jnp.maximum(
                    jnp.abs(arm_qacc_unlimited)
                    / jnp.maximum(self.arm_acc_limit_rad_s2[None, :], 1e-6)
                    - 1.0,
                    0.0,
                )
                # Measure the pre-projection violation. log1p prevents the
                # unconstrained XML position servo from creating unbounded
                # reward magnitudes while preserving a learning signal.
                arm_actual_intervention_pen = arm_actual_intervention_pen + jnp.mean(
                    jnp.log1p(arm_qacc_excess_ratio * arm_qacc_excess_ratio),
                    axis=-1,
                )
                d = d.replace(
                    qpos=d.qpos.at[:, self.arm_qadr].set(arm_q_out),
                    qvel=d.qvel.at[:, self.arm_vadr].set(arm_qvel_out),
                    qacc=d.qacc.at[:, self.arm_vadr].set(arm_qacc_out),
                )
                if not bool(self.cfg.arm_actual_target_tracking_governor):
                    d = self.batched_forward(state.model, d)
                    d = d.replace(
                        qacc=d.qacc.at[:, self.arm_vadr].set(arm_qacc_out)
                    )
                arm_actual_clip_count = (
                    arm_actual_clip_count + arm_actual_clip.astype(jnp.int32)
                )
                arm_actual_feasible = arm_actual_feasible & substep_feasible
            substep_contact, substep_other_ball_contact = self._ball_contact_flags(d)
            first_contact = substep_contact & (~contact_any)
            substep_bpos = d.xpos[:, self.ball_body_id]
            substep_camera_terms = self._camera_reward_terms(d, substep_bpos)
            substep_camera_visible = substep_camera_terms["metric/camera_visible"] > 0.5
            substep_camera_in_margin = substep_camera_terms["metric/camera_in_margin"] > 0.5
            substep_camera_v_frac = substep_camera_terms["metric/ball_pixel_v"] / max(
                1.0,
                float(self.cfg.camera_image_height),
            )
            return (
                d,
                contact_any | substep_contact,
                other_ball_contact_any | substep_other_ball_contact,
                jnp.where(first_contact, substep_camera_visible, first_contact_camera_visible),
                jnp.where(first_contact, substep_camera_in_margin, first_contact_camera_in_margin),
                jnp.where(first_contact, substep_camera_v_frac, first_contact_camera_v_frac),
                arm_actual_clip_count,
                arm_actual_feasible,
                arm_actual_intervention_pen,
                arm_actual_jerk_emergency_count,
                arm_actual_velocity_utilization_max,
                arm_actual_acceleration_utilization_max,
                arm_actual_jerk_utilization_max,
                arm_actual_acceleration_saturation_count,
                arm_actual_high_acceleration_sign_flip_count,
            )

        (
            data,
            in_contact,
            other_ball_contact,
            contact_camera_visible,
            contact_camera_in_margin,
            contact_camera_v_frac,
            arm_actual_clip_count,
            arm_actual_feasible,
            arm_actual_intervention_pen,
            arm_actual_jerk_emergency_count,
            arm_actual_velocity_utilization_max,
            arm_actual_acceleration_utilization_max,
            arm_actual_jerk_utilization_max,
            arm_actual_acceleration_saturation_count,
            arm_actual_high_acceleration_sign_flip_count,
        ) = jax.lax.fori_loop(
            0,
            int(self.cfg.frame_skip),
            one_substep,
            (
                data,
                contact_init,
                contact_init,
                contact_init,
                contact_init,
                contact_camera_v_frac_init,
                arm_actual_clip_count_init,
                arm_actual_feasible_init,
                arm_actual_intervention_pen_init,
                arm_actual_jerk_emergency_count_init,
                arm_actual_velocity_utilization_max_init,
                arm_actual_acceleration_utilization_max_init,
                arm_actual_jerk_utilization_max_init,
                arm_actual_acceleration_saturation_count_init,
                arm_actual_high_acceleration_sign_flip_count_init,
            ),
        )
        if bool(self.cfg.arm_actual_target_tracking_governor):
            # mjx.step starts with mjx.forward, so forwarding after every
            # projected substep only repeats work.  One final forward keeps
            # reward/observation kinematics synchronized with the governed
            # state while the next substep still begins from that state.
            final_arm_qacc = data.qacc[:, self.arm_vadr]
            data = self.batched_forward(state.model, data)
            data = data.replace(
                qacc=data.qacc.at[:, self.arm_vadr].set(final_arm_qacc)
            )
        arm_actual_intervention_pen = arm_actual_intervention_pen / max(
            1,
            int(self.cfg.frame_skip),
        )
        arm_limiter_pen = arm_limiter_pen + arm_actual_intervention_pen

        step_count = state.step_count + 1
        bpos = data.xpos[:, self.ball_body_id]
        rpos = data.site_xpos[:, self.racket_site_id]
        rmat = data.site_xmat[:, self.racket_site_id].reshape((-1, 3, 3))
        racket_normal = rmat[:, :, 2]
        bvel = (bpos - state.prev_ball_pos) / max(self.dt, 1e-6)
        rvel = (rpos - state.prev_racket_pos) / max(self.dt, 1e-6)
        rel = bpos - rpos
        rel_local = jnp.einsum("nij,nj->ni", jnp.swapaxes(rmat, 1, 2), rel)

        sep_dist = jnp.linalg.norm(rel, axis=-1)
        no_contact_steps = jnp.where(in_contact, 0, state.no_contact_steps + 1)
        contact_hold_steps = jnp.where(in_contact, state.contact_hold_steps + 1, 0)
        hit_armed = jnp.where(
            (~in_contact)
            & (no_contact_steps >= int(self.cfg.hit_rearm_no_contact_steps))
            & (sep_dist >= float(self.cfg.hit_rearm_distance)),
            True,
            state.hit_armed,
        )

        hit_edge = in_contact & (~state.prev_contact) & hit_armed & (~state.pending_hit)
        camera_terms = self._camera_reward_terms(data, bpos)
        contact_camera_in_lower_band = (
            contact_camera_visible
            & contact_camera_in_margin
            & (contact_camera_v_frac >= float(self.cfg.hit_camera_lower_band_frac[0]))
            & (contact_camera_v_frac <= float(self.cfg.hit_camera_lower_band_frac[1]))
        )
        pending_hit_camera_visible = jnp.where(
            hit_edge,
            contact_camera_visible,
            state.pending_hit_camera_visible,
        )
        pending_hit_camera_in_margin = jnp.where(
            hit_edge,
            contact_camera_in_margin,
            state.pending_hit_camera_in_margin,
        )
        pending_hit_camera_in_lower_band = jnp.where(
            hit_edge,
            contact_camera_in_lower_band,
            state.pending_hit_camera_in_lower_band,
        )
        pending_hit_camera_v_frac = jnp.where(hit_edge, contact_camera_v_frac, state.pending_hit_camera_v_frac)
        pending_hit = state.pending_hit | hit_edge
        pending_steps = jnp.where(pending_hit, state.pending_hit_steps + 1, 0)
        hit_armed = jnp.where(hit_edge, False, hit_armed)

        upward_vz = jnp.maximum(0.0, bvel[:, 2])
        gravity_mag = jnp.maximum(jnp.abs(state.dr_gravity_z), 1e-6)
        predicted_apex_z = bpos[:, 2] + (upward_vz * upward_vz) / (2.0 * gravity_mag)
        min_launch_rel_z = max(float(self.cfg.hit_confirm_rel_height), 0.04)
        min_launch_apex_z = state.racket_anchor[:, 2] + max(0.70 * float(self.cfg.target_height), min_launch_rel_z + 0.06)
        launched_upward_raw = (
            pending_hit
            & (~in_contact)
            & (rel[:, 2] >= min_launch_rel_z)
            & (bvel[:, 2] > 0.0)
            & (predicted_apex_z >= min_launch_apex_z)
        )
        current_time = step_count.astype(jnp.float32) * self.dt
        count_gate_interval = current_time - state.last_count_gate_hit_time
        counted_hit = launched_upward_raw & (
            (float(self.cfg.hit_min_count_interval) <= 0.0)
            | (state.last_count_gate_hit_time < 0.0)
            | (count_gate_interval >= float(self.cfg.hit_min_count_interval))
        )
        ignored_fast_hit = launched_upward_raw & (~counted_hit)
        cap = int(self.hit_reward_count_cap_active)
        rewardable_hit = counted_hit & ((cap <= 0) | (state.rewarded_hit_count < cap))
        unrewarded_extra_hit = counted_hit & (~rewardable_hit)
        launched_upward = counted_hit
        failed_hit = pending_hit & (pending_steps >= int(self.cfg.hit_confirm_max_steps)) & (~launched_upward_raw)
        hit_count = state.hit_count + counted_hit.astype(jnp.int32)
        confirmed_hit_count = state.confirmed_hit_count + launched_upward_raw.astype(jnp.int32)
        ignored_fast_hit_count = state.ignored_fast_hit_count + ignored_fast_hit.astype(jnp.int32)
        rewarded_hit_count = state.rewarded_hit_count + rewardable_hit.astype(jnp.int32)
        unrewarded_extra_hit_count = state.unrewarded_extra_hit_count + unrewarded_extra_hit.astype(jnp.int32)
        last_count_gate_hit_time = jnp.where(launched_upward_raw, current_time, state.last_count_gate_hit_time)
        last_counted_hit_time = jnp.where(counted_hit, current_time, state.last_counted_hit_time)
        hit_interval = current_time - state.last_hit_time
        has_prev_hit = state.last_hit_time >= 0.0
        cadence_eligible = launched_upward_raw & counted_hit & rewardable_hit & has_prev_hit
        hit_cadence_reward = jnp.where(
            cadence_eligible & (float(self.cfg.hit_cadence_reward_weight) > 0.0),
            float(self.cfg.hit_cadence_reward_weight)
            * jnp.exp(
                -0.5
                * (
                    (hit_interval - float(self.cfg.hit_cadence_target_interval))
                    / max(1e-6, float(self.cfg.hit_cadence_sigma))
                )
                ** 2
            ),
            0.0,
        )
        hit_min_interval_penalty = jnp.where(
            cadence_eligible
            & (float(self.cfg.hit_min_interval_penalty_weight) > 0.0)
            & (hit_interval < float(self.cfg.hit_min_interval)),
            float(self.cfg.hit_min_interval_penalty_weight)
            * (
                (float(self.cfg.hit_min_interval) - hit_interval)
                / max(1e-6, float(self.cfg.hit_min_interval))
            )
            ** 2,
            0.0,
        )
        fast_hit_penalty = jnp.where(
            ignored_fast_hit
            & (float(self.cfg.fast_hit_penalty_weight) > 0.0)
            & (float(self.cfg.hit_min_count_interval) > 0.0),
            float(self.cfg.fast_hit_penalty_weight)
            * (
                (float(self.cfg.hit_min_count_interval) - count_gate_interval)
                / max(1e-6, float(self.cfg.hit_min_count_interval))
            )
            ** 2,
            0.0,
        )
        last_hit_time = jnp.where(launched_upward_raw, current_time, state.last_hit_time)
        pending_hit = jnp.where(launched_upward_raw | failed_hit, False, pending_hit)
        pending_steps = jnp.where(launched_upward_raw | failed_hit, 0, pending_steps)
        hit_camera_visible = pending_hit_camera_visible
        hit_camera_in_margin = pending_hit_camera_in_margin
        hit_camera_in_lower_band = pending_hit_camera_in_lower_band
        hit_camera_v_frac = pending_hit_camera_v_frac
        hit_resolved = launched_upward_raw | failed_hit
        pending_hit_camera_visible = jnp.where(hit_resolved, False, pending_hit_camera_visible)
        pending_hit_camera_in_margin = jnp.where(hit_resolved, False, pending_hit_camera_in_margin)
        pending_hit_camera_in_lower_band = jnp.where(hit_resolved, False, pending_hit_camera_in_lower_band)
        pending_hit_camera_v_frac = jnp.where(hit_resolved, 0.0, pending_hit_camera_v_frac)

        reward, reward_terms = self._reward(
            data=data,
            camera_terms=camera_terms,
            action=action,
            da=da,
            action_clip_excess=action_clip_excess,
            arm_limiter_pen=arm_limiter_pen,
            bpos=bpos,
            bvel=bvel,
            rpos=rpos,
            rvel=rvel,
            rel=rel,
            rel_local=rel_local,
            racket_normal=racket_normal,
            predicted_apex_z=predicted_apex_z,
            hit_count=hit_count,
            new_hit=launched_upward,
            rewardable_hit=rewardable_hit,
            failed_hit=failed_hit,
            ignored_fast_hit=ignored_fast_hit,
            hit_cadence_reward=hit_cadence_reward,
            hit_min_interval_penalty=hit_min_interval_penalty,
            fast_hit_penalty=fast_hit_penalty,
            hit_camera_visible=hit_camera_visible,
            hit_camera_in_margin=hit_camera_in_margin,
            hit_camera_in_lower_band=hit_camera_in_lower_band,
            hit_camera_v_frac=hit_camera_v_frac,
            other_ball_contact=other_ball_contact,
            in_contact=in_contact,
            contact_hold_steps=contact_hold_steps,
            rel_speed=jnp.linalg.norm(bvel - rvel, axis=-1),
            cmd_qvel=cmd_qvel,
            prev_arm_qvel=state.prev_arm_qvel,
            racket_anchor=state.racket_anchor,
            chest_target_offset=state.chest_target_offset,
        )

        arm_qvel = data.qvel[:, self.arm_vadr]
        terminated, done_terms = self._termination_terms(data, bpos, rpos, state.racket_anchor)
        ball_miss = (
            done_terms["ball_too_low"]
            | done_terms["ball_too_high"]
            | done_terms["ball_x_out_of_bounds"]
            | done_terms["ball_y_out_of_bounds"]
            | done_terms["ball_view_x_too_low"]
            | done_terms["ball_view_x_too_high"]
            | done_terms["ball_view_y_too_low"]
            | done_terms["ball_view_y_too_high"]
            | done_terms["ball_view_z_too_low"]
            | done_terms["ball_view_z_too_high"]
        )
        racket_limit_done = done_terms["racket_too_high"] | done_terms["racket_too_low"]
        ball_miss_penalty_active = ball_miss & (
            (hit_count > 0)
            | (not bool(self.cfg.termination_miss_penalty_requires_hit))
        )
        no_hit_early_termination_penalty = jnp.where(
            hit_count <= 0,
            float(self.cfg.termination_no_hit_miss_early_penalty)
            * jnp.maximum(0.0, 1.0 - step_count.astype(jnp.float32) / float(self.max_steps)),
            0.0,
        )
        ball_miss_penalty = jnp.where(
            ball_miss_penalty_active,
            -(
                float(self.cfg.termination_miss_penalty_base)
                + float(self.cfg.termination_miss_penalty_per_hit) * hit_count.astype(jnp.float32)
                + no_hit_early_termination_penalty
            ),
            0.0,
        )
        racket_limit_penalty = jnp.where(
            racket_limit_done,
            -(
                float(self.cfg.racket_z_limit_termination_penalty_base)
                + float(self.cfg.racket_z_limit_termination_penalty_per_hit) * hit_count.astype(jnp.float32)
                + no_hit_early_termination_penalty
            ),
            0.0,
        )
        racket_anchor_done = done_terms["racket_too_far_from_anchor"]
        racket_anchor_penalty = jnp.where(
            racket_anchor_done,
            -(
                float(self.cfg.racket_anchor_termination_penalty_base)
                + float(self.cfg.racket_anchor_termination_penalty_per_hit)
                * hit_count.astype(jnp.float32)
                + no_hit_early_termination_penalty
            ),
            0.0,
        )
        reward = (
            reward
            + ball_miss_penalty
            + racket_limit_penalty
            + racket_anchor_penalty
        )
        truncated = step_count >= self.max_steps
        done = terminated | truncated

        next_state = EnvState(
            model=state.model,
            data=data,
            rng=rng_after_delay,
            step_count=step_count,
            racket_anchor=state.racket_anchor,
            chest_target_offset=state.chest_target_offset,
            reset_ball_pos=state.reset_ball_pos,
            reset_ball_vel=state.reset_ball_vel,
            reset_target_offset=state.reset_target_offset,
            reset_disturbance_strength=state.reset_disturbance_strength,
            reset_ball_surface_gap=state.reset_ball_surface_gap,
            reset_ball_racket_center_offset=state.reset_ball_racket_center_offset,
            reset_ball_obs_missing=state.reset_ball_obs_missing,
            arm_cmd_q=arm_cmd_q,
            arm_cmd_qvel=cmd_qvel,
            arm_q_ref_latest=arm_q_ref_latest,
            arm_q_ref_active=arm_q_ref_active,
            arm_actuator_q_ref_latest=arm_actuator_q_ref_latest,
            arm_actuator_q_ref_active=arm_actuator_q_ref_active,
            arm_safe_q_ref_latest=arm_safe_q_ref_latest,
            arm_safe_qvel=arm_safe_qvel,
            arm_safe_qacc=arm_safe_qacc,
            ball_obs_missing_episode_coherent_enabled=state.ball_obs_missing_episode_coherent_enabled,
            ball_obs_camera_missing_enabled=state.ball_obs_camera_missing_enabled,
            ball_obs_view_bounds_missing_enabled=state.ball_obs_view_bounds_missing_enabled,
            arm_applied_q=arm_applied_q,
            arm_applied_qvel=arm_applied_qvel,
            prev_action=action,
            prev_arm_qvel=arm_qvel,
            prev_ball_pos=bpos,
            prev_racket_pos=rpos,
            prev_contact=in_contact,
            hit_armed=hit_armed,
            no_contact_steps=no_contact_steps,
            contact_hold_steps=contact_hold_steps,
            pending_hit=pending_hit,
            pending_hit_steps=pending_steps,
            pending_hit_camera_visible=pending_hit_camera_visible,
            pending_hit_camera_in_margin=pending_hit_camera_in_margin,
            pending_hit_camera_in_lower_band=pending_hit_camera_in_lower_band,
            pending_hit_camera_v_frac=pending_hit_camera_v_frac,
            hit_count=hit_count,
            action_buffer=action_buffer,
            action_latency_steps=state.action_latency_steps,
            command_buffer=command_buffer,
            actuator_command_buffer=actuator_command_buffer,
            tau_act_episode=state.tau_act_episode,
            tau_act=tau_act,
            delay_steps=delay_steps,
            delay_bin_id=state.delay_bin_id,
            anti_windup_scale=anti_windup_scale,
            obs_buffer=state.obs_buffer,
            obs_latency_steps=state.obs_latency_steps,
            obs_history=state.obs_history,
            action_history=state.action_history,
            cached_ball_obs_pos=state.cached_ball_obs_pos,
            cached_ball_obs_vel=state.cached_ball_obs_vel,
            last_ball_obs_step=state.last_ball_obs_step,
            ball_obs_valid_pos=state.ball_obs_valid_pos,
            ball_obs_valid_vel=state.ball_obs_valid_vel,
            ball_obs_age_seconds=state.ball_obs_age_seconds,
            ball_obs_missing_since_sample=state.ball_obs_missing_since_sample,
            ball_obs_dropout_remaining=state.ball_obs_dropout_remaining,
            ball_obs_dropout_steps_total=state.ball_obs_dropout_steps_total,
            ball_obs_burst_count=state.ball_obs_burst_count,
            total_env_steps=state.total_env_steps + 1,
            action_scale_mult=state.action_scale_mult,
            actuator_cmd_tau=state.actuator_cmd_tau,
            actuator_cmd_gain=state.actuator_cmd_gain,
            dr_gravity_z=state.dr_gravity_z,
            dr_ball_mass=state.dr_ball_mass,
            dr_ball_friction=state.dr_ball_friction,
            dr_racket_friction=state.dr_racket_friction,
            dr_ball_solref_time=state.dr_ball_solref_time,
            dr_ball_solref_damping=state.dr_ball_solref_damping,
            dr_damping_mult=state.dr_damping_mult,
            dr_armature_mult=state.dr_armature_mult,
            dr_pd_kp_mult=state.dr_pd_kp_mult,
            dr_pd_kv_mult=state.dr_pd_kv_mult,
            last_hit_time=last_hit_time,
            last_counted_hit_time=last_counted_hit_time,
            last_count_gate_hit_time=last_count_gate_hit_time,
            confirmed_hit_count=confirmed_hit_count,
            ignored_fast_hit_count=ignored_fast_hit_count,
            rewarded_hit_count=rewarded_hit_count,
            unrewarded_extra_hit_count=unrewarded_extra_hit_count,
            dr_racket_pos_offset=state.dr_racket_pos_offset,
            dr_racket_rot_offset=state.dr_racket_rot_offset,
            dr_racket_radius_offset=state.dr_racket_radius_offset,
            ball_obs_pos_bias_base=state.ball_obs_pos_bias_base,
            ball_obs_rot_bias_rpy=state.ball_obs_rot_bias_rpy,
            ball_obs_vel_bias_base=state.ball_obs_vel_bias_base,
            ball_obs_scale=state.ball_obs_scale,
            ball_obs_view_z_high_m=state.ball_obs_view_z_high_m,
        )
        obs_state = next_state._replace(prev_racket_pos=state.prev_racket_pos)
        obs_state, obs, obs_metrics = self._apply_observation_pipeline(obs_state, bpos, bvel)
        next_state = obs_state._replace(prev_racket_pos=rpos)
        e_active = arm_q_ref_active - data.qpos[:, self.arm_qadr]
        e_actuator_active = arm_actuator_q_ref_active - data.qpos[:, self.arm_qadr]
        t_contact_est = self._estimate_contact_time_from_z_vz(
            next_state,
            rel[:, 2],
            bvel[:, 2] - rvel[:, 2],
            next_state.ball_obs_age_seconds,
        )
        t_margin = t_contact_est - tau_act
        action_rate_norm = jnp.linalg.norm(da, axis=-1)
        action_jerk_norm = action_rate_norm / max(self.dt, 1e-6)
        command_tracking_error = jnp.linalg.norm(e_active, axis=-1)
        command_tracking_penalty = (
            -float(self.cfg.command_tracking_error_penalty_weight)
            * command_tracking_error
            * command_tracking_error
            * self.dt
        )
        delay_action_jerk_penalty = (
            -float(self.cfg.delay_action_jerk_penalty_weight)
            * action_jerk_norm
            * action_jerk_norm
            * self.dt
        )
        reward = reward + command_tracking_penalty + delay_action_jerk_penalty
        metrics = {
            "hit_count": hit_count.astype(jnp.float32),
            "new_hit": launched_upward.astype(jnp.float32),
            "confirmed_hit": launched_upward_raw.astype(jnp.float32),
            "rewardable_hit": rewardable_hit.astype(jnp.float32),
            "ignored_fast_hit": ignored_fast_hit.astype(jnp.float32),
            "other_ball_contact": other_ball_contact.astype(jnp.float32),
            "ball_z": bpos[:, 2],
            "racket_z": rpos[:, 2],
            "racket_z_rel": rpos[:, 2] - state.racket_anchor[:, 2],
            "in_contact": in_contact.astype(jnp.float32),
            "reset_ball_x": state.reset_ball_pos[:, 0],
            "reset_ball_y": state.reset_ball_pos[:, 1],
            "reset_ball_z": state.reset_ball_pos[:, 2],
            "reset_ball_vxy": jnp.linalg.norm(state.reset_ball_vel[:, :2], axis=-1),
            "reset_ball_vz": state.reset_ball_vel[:, 2],
            "reset_ball_anchor_dx": state.reset_ball_pos[:, 0] - state.racket_anchor[:, 0],
            "reset_ball_anchor_dy": state.reset_ball_pos[:, 1] - state.racket_anchor[:, 1],
            "reset_ball_anchor_dz": state.reset_ball_pos[:, 2] - state.racket_anchor[:, 2],
            "reset_target_x": state.reset_target_offset[:, 0],
            "reset_target_y": state.reset_target_offset[:, 1],
            "reset_target_z": state.reset_target_offset[:, 2],
            "reset_disturbance_strength": state.reset_disturbance_strength,
            "reset_ball_surface_gap": state.reset_ball_surface_gap,
            "reset_ball_racket_center_offset": state.reset_ball_racket_center_offset,
            "action_scale_mult": state.action_scale_mult,
            "dr_gravity_z": state.dr_gravity_z,
            "reset_ball_obs_missing": state.reset_ball_obs_missing.astype(jnp.float32),
            "dr_ball_mass": state.dr_ball_mass,
            "dr_ball_friction": state.dr_ball_friction,
            "dr_racket_friction": state.dr_racket_friction,
            "dr_ball_solref_time": state.dr_ball_solref_time,
            "dr_ball_solref_damping": state.dr_ball_solref_damping,
            "dr_damping_mult": state.dr_damping_mult,
            "dr_armature_mult": state.dr_armature_mult,
            "dr_pd_kp_mult_mean": jnp.mean(state.dr_pd_kp_mult, axis=-1),
            "dr_pd_kp_mult_min": jnp.min(state.dr_pd_kp_mult, axis=-1),
            "dr_pd_kp_mult_max": jnp.max(state.dr_pd_kp_mult, axis=-1),
            "dr_pd_kv_mult_mean": jnp.mean(state.dr_pd_kv_mult, axis=-1),
            "dr_pd_kv_mult_min": jnp.min(state.dr_pd_kv_mult, axis=-1),
            "dr_pd_kv_mult_max": jnp.max(state.dr_pd_kv_mult, axis=-1),
            "dr_actuator_cmd_tau": state.actuator_cmd_tau,
            "dr_actuator_cmd_gain": state.actuator_cmd_gain,
            "arm_applied_cmd_lag_norm": jnp.linalg.norm(arm_actuator_q_ref_latest - arm_applied_q, axis=-1),
            "arm_nominal_cmd_lag_norm": jnp.linalg.norm(arm_cmd_q - arm_applied_q, axis=-1),
            "arm_actuator_cmd_lag_norm": jnp.linalg.norm(arm_actuator_q_ref_latest - arm_applied_q, axis=-1),
            "actuator_lead_delta_norm": jnp.linalg.norm(arm_actuator_q_ref_latest - arm_q_ref_latest, axis=-1),
            "actuator_comp_delta_norm": jnp.linalg.norm(arm_actuator_q_ref_latest - arm_q_ref_latest, axis=-1),
            "arm_post_comp_safety_delta_norm": jnp.linalg.norm(
                arm_actuator_q_ref_latest - arm_safe_q_ref_latest,
                axis=-1,
            ),
            "arm_post_comp_safety_clip_any": jnp.any(arm_safe_clip, axis=-1).astype(jnp.float32),
            "arm_post_comp_safety_clip_fraction": jnp.mean(arm_safe_clip.astype(jnp.float32), axis=-1),
            "arm_post_comp_safety_feasible": arm_safe_feasible.astype(jnp.float32),
            "arm_servo_safety_delta_norm": jnp.linalg.norm(
                arm_servo_target_unlimited - arm_applied_q,
                axis=-1,
            ),
            "arm_servo_safety_clip_any": jnp.any(arm_servo_clip, axis=-1).astype(jnp.float32),
            "arm_servo_safety_clip_fraction": jnp.mean(arm_servo_clip.astype(jnp.float32), axis=-1),
            "arm_servo_safety_feasible": arm_servo_feasible.astype(jnp.float32),
            "arm_actual_safety_clip_any": jnp.any(
                arm_actual_clip_count > 0,
                axis=-1,
            ).astype(jnp.float32),
            "arm_actual_safety_clip_fraction": jnp.mean(
                arm_actual_clip_count.astype(jnp.float32),
                axis=-1,
            ) / max(1, int(self.cfg.frame_skip)),
            "arm_actual_safety_feasible": arm_actual_feasible.astype(jnp.float32),
            "arm_actual_safety_clip_count": arm_actual_clip_count,
            "arm_actual_safety_intervention_penalty": arm_actual_intervention_pen,
            "arm_actual_governor_jerk_emergency_fraction": jnp.mean(
                arm_actual_jerk_emergency_count.astype(jnp.float32),
                axis=-1,
            ) / max(1, int(self.cfg.frame_skip)),
            "arm_actual_governor_jerk_emergency_count": arm_actual_jerk_emergency_count,
            "arm_actual_velocity_limit_utilization_max": (
                arm_actual_velocity_utilization_max
            ),
            "arm_actual_acceleration_limit_utilization_max": (
                arm_actual_acceleration_utilization_max
            ),
            "arm_actual_acceleration_saturation_fraction": (
                arm_actual_acceleration_saturation_count.astype(jnp.float32)
                / max(1, int(self.cfg.frame_skip) * self.act_dim)
            ),
            "arm_actual_high_acceleration_sign_flip_fraction": (
                arm_actual_high_acceleration_sign_flip_count.astype(jnp.float32)
                / max(1, int(self.cfg.frame_skip) * self.act_dim)
            ),
            "arm_actual_governor_jerk_limit_utilization_max": (
                arm_actual_jerk_utilization_max
            ),
            "arm_actual_governor_tracking_error_norm": jnp.linalg.norm(
                arm_applied_q - data.qpos[:, self.arm_qadr],
                axis=-1,
            ),
            "ball_obs_pos_bias_norm": jnp.linalg.norm(state.ball_obs_pos_bias_base, axis=-1),
            "ball_obs_rot_bias_norm": jnp.linalg.norm(state.ball_obs_rot_bias_rpy, axis=-1),
            "ball_obs_vel_bias_norm": jnp.linalg.norm(state.ball_obs_vel_bias_base, axis=-1),
            "ball_obs_scale": state.ball_obs_scale,
            "dr_racket_pos_offset_norm": jnp.linalg.norm(state.dr_racket_pos_offset, axis=-1),
            "dr_racket_rot_offset_norm": jnp.linalg.norm(state.dr_racket_rot_offset, axis=-1),
            "dr_racket_radius_offset": state.dr_racket_radius_offset,
            "dr_obs_latency_steps": state.obs_latency_steps.astype(jnp.float32),
            "dr_action_latency_steps": state.action_latency_steps.astype(jnp.float32),
            "action_norm": jnp.linalg.norm(action, axis=-1),
            "action_rate": action_rate_norm,
            "action_jerk": action_jerk_norm,
            "raw_action_clip_fraction": jnp.mean(
                (action_clip_excess > 0.0).astype(jnp.float32),
                axis=-1,
            ),
            "raw_action_clip_excess_norm": jnp.linalg.norm(
                action_clip_excess,
                axis=-1,
            ),
            "command_tracking_error": command_tracking_error,
            "actuator_command_tracking_error": jnp.linalg.norm(e_actuator_active, axis=-1),
            "tau_act_ms": tau_act * 1000.0,
            "delay_bin_id": state.delay_bin_id.astype(jnp.float32),
            "delay_steps": delay_steps.astype(jnp.float32),
            "anti_windup_scale": anti_windup_scale,
            "t_contact_est": t_contact_est,
            "t_margin": t_margin,
            "q_cmd_nominal": arm_cmd_q,
            "q_ref_latest": arm_q_ref_latest,
            "q_ref_active": arm_q_ref_active,
            "q_actuator_ref_latest": arm_actuator_q_ref_latest,
            "q_actuator_ref_active": arm_actuator_q_ref_active,
            "q_servo_ref_active": arm_servo_q_ref_active,
            "servo_execution_delay_steps": jnp.where(
                servo_uses_delayed_reference,
                delay_steps,
                jnp.zeros_like(delay_steps),
            ).astype(jnp.float32),
            "servo_execution_delay_ms": jnp.where(
                servo_uses_delayed_reference,
                delay_steps.astype(jnp.float32) * float(self.dt) * 1000.0,
                jnp.zeros_like(tau_act),
            ),
            "q_post_comp_safe_latest": arm_safe_q_ref_latest,
            "dq_post_comp_safe_latest": arm_safe_qvel,
            "ddq_post_comp_safe_latest": arm_safe_qacc,
            "dq_post_comp_safe_interval_low": arm_safe_interval_low,
            "dq_post_comp_safe_interval_high": arm_safe_interval_high,
            "q_servo_target_unlimited": arm_servo_target_unlimited,
            "q_servo_target": arm_applied_q,
            "dq_servo_target": arm_applied_qvel,
            "ddq_servo_target": arm_applied_qacc,
            "dq_servo_safe_interval_low": arm_servo_interval_low,
            "dq_servo_safe_interval_high": arm_servo_interval_high,
            "dq_ref_latest": cmd_qvel,
            "ball_obs_age": next_state.ball_obs_age_seconds,
            "ball_obs_view_z_high_m": next_state.ball_obs_view_z_high_m,
            "terminated": terminated.astype(jnp.float32),
            "truncated": truncated.astype(jnp.float32),
            "episode_step": step_count.astype(jnp.float32),
        }
        for name, value in reward_terms.items():
            if name.startswith("metric/"):
                metrics[name[len("metric/") :]] = value
            else:
                metrics[f"reward/{name}"] = value
        metrics["reward/ball_miss_termination_penalty"] = ball_miss_penalty
        metrics["reward/racket_z_limit_termination_penalty"] = racket_limit_penalty
        metrics["reward/racket_anchor_termination_penalty"] = racket_anchor_penalty
        metrics["reward/command_tracking_error_penalty"] = command_tracking_penalty
        metrics["reward/delay_action_jerk_penalty"] = delay_action_jerk_penalty
        metrics["reward/total"] = reward
        metrics.update(obs_metrics)
        metrics.update({f"done/{name}": value.astype(jnp.float32) for name, value in done_terms.items()})
        return next_state, obs, reward, done, metrics

    def reset_done(self, state: EnvState, obs: jax.Array, done: jax.Array, keys: jax.Array) -> tuple[EnvState, jax.Array]:
        old_total_env_steps = state.total_env_steps
        reset_state, reset_obs = self.reset(keys)

        def select(reset_leaf, leaf):
            if not hasattr(leaf, "shape") or leaf.shape[:1] != done.shape[:1]:
                return leaf
            mask_shape = (done.shape[0],) + (1,) * (leaf.ndim - 1)
            return jnp.where(done.reshape(mask_shape), reset_leaf, leaf)

        state = jax.tree_util.tree_map(select, reset_state, state)
        state = state._replace(total_env_steps=old_total_env_steps)
        obs = jnp.where(done[:, None], reset_obs, obs)
        return state, obs

    def _apply_observation_pipeline(
        self,
        state: EnvState,
        true_bpos: jax.Array,
        true_bvel: jax.Array,
    ) -> tuple[EnvState, jax.Array, dict[str, jax.Array]]:
        split_keys = jax.vmap(lambda k: jax.random.split(k, 8))(state.rng)
        next_rng = split_keys[:, 0]
        key_pos_noise = split_keys[:, 1]
        key_vel_noise = split_keys[:, 2]
        key_dropout = split_keys[:, 3]
        key_dropout_duration = split_keys[:, 4]
        key_burst_duration = split_keys[:, 5]
        key_view_bounds_missing = split_keys[:, 6]
        key_camera_missing = split_keys[:, 7]

        total_steps_cfg = max(1, int(self.cfg.total_training_steps))
        warmup = max(0, int(round(total_steps_cfg * float(self.cfg.ball_obs_noise_warmup_ratio))))
        ramp = max(1, int(round(total_steps_cfg * float(self.cfg.ball_obs_noise_ramp_ratio))))
        noise_scale = jnp.clip((state.total_env_steps.astype(jnp.float32) - float(warmup)) / float(ramp), 0.0, 1.0)
        pos_std = float(self.cfg.ball_obs_pos_noise_std) * noise_scale
        vel_std = float(self.cfg.ball_obs_vel_noise_std) * noise_scale
        pos_noise = jax.vmap(lambda k: jax.random.normal(k, (3,), dtype=jnp.float32))(key_pos_noise) * pos_std[:, None]
        vel_noise = jax.vmap(lambda k: jax.random.normal(k, (3,), dtype=jnp.float32))(key_vel_noise) * vel_std[:, None]

        if bool(self.cfg.ball_obs_fractional_rate):
            prev_tick = jnp.floor(state.last_ball_obs_step.astype(jnp.float32) * float(self.ball_obs_period))
            curr_tick = jnp.floor(state.step_count.astype(jnp.float32) * float(self.ball_obs_period))
            refresh = curr_tick > prev_tick
        else:
            refresh = (state.step_count - state.last_ball_obs_step) >= int(self.ball_obs_every)
        camera_visible_for_obs = jnp.ones((true_bpos.shape[0],), dtype=bool)
        if bool(self.cfg.ball_obs_require_camera_visible) and self.cfg.camera_visibility_mode != "off":
            camera_terms = self._camera_reward_terms(state.data, true_bpos)
            camera_visible = camera_terms["metric/camera_visible"] > 0.5
            camera_missing_prob = float(self.cfg.ball_obs_camera_missing_prob)
            if camera_missing_prob >= 1.0:
                camera_visible_for_obs = camera_visible
            elif camera_missing_prob <= 0.0:
                camera_visible_for_obs = jnp.ones_like(camera_visible, dtype=bool)
            else:
                u_camera_missing = jax.vmap(
                    lambda k: jax.random.uniform(k, (), dtype=jnp.float32)
                )(key_camera_missing)
                frame_missing = (
                    (~camera_visible) & (u_camera_missing < camera_missing_prob)
                )
                coherent_missing = (
                    (~camera_visible) & state.ball_obs_camera_missing_enabled
                )
                camera_missing = jnp.where(
                    state.ball_obs_missing_episode_coherent_enabled,
                    coherent_missing,
                    frame_missing,
                )
                camera_visible_for_obs = camera_visible | (~camera_missing)
        if bool(self.cfg.ball_obs_require_view_z_high):
            camera_visible_for_obs = camera_visible_for_obs & (
                true_bpos[:, 2] <= state.ball_obs_view_z_high_m
            )
        view_bounds_visible_for_obs = jnp.ones((true_bpos.shape[0],), dtype=bool)
        if bool(self.cfg.ball_obs_require_view_bounds):
            data = state.data
            base_q = jnp.stack(
                [
                    data.qpos[:, self.base_x_qadr],
                    data.qpos[:, self.base_y_qadr],
                    data.qpos[:, self.base_yaw_qadr],
                ],
                axis=-1,
            )
            bpos_base = self._point_to_base(true_bpos, base_q)
            view_bounds_visible_for_obs = (
                (bpos_base[:, 0] >= float(self.cfg.ball_view_x_bounds_m[0]))
                & (bpos_base[:, 0] <= float(self.cfg.ball_view_x_bounds_m[1]))
                & (bpos_base[:, 1] >= float(self.cfg.ball_view_y_bounds_m[0]))
                & (bpos_base[:, 1] <= float(self.cfg.ball_view_y_bounds_m[1]))
                & (true_bpos[:, 2] >= float(self.cfg.ball_view_z_bounds_m[0]))
                & (true_bpos[:, 2] <= state.ball_obs_view_z_high_m)
            )
            if float(self.cfg.ball_obs_view_bounds_missing_prob) >= 1.0:
                view_bounds_sample_available = view_bounds_visible_for_obs
            elif float(self.cfg.ball_obs_view_bounds_missing_prob) <= 0.0:
                view_bounds_sample_available = jnp.ones_like(view_bounds_visible_for_obs, dtype=bool)
            else:
                u_view_bounds = jax.vmap(lambda k: jax.random.uniform(k, (), dtype=jnp.float32))(
                    key_view_bounds_missing
                )
                frame_view_bounds_missing = (~view_bounds_visible_for_obs) & (
                    u_view_bounds < float(self.cfg.ball_obs_view_bounds_missing_prob)
                )
                coherent_view_bounds_missing = (
                    (~view_bounds_visible_for_obs)
                    & state.ball_obs_view_bounds_missing_enabled
                )
                view_bounds_missing = jnp.where(
                    state.ball_obs_missing_episode_coherent_enabled,
                    coherent_view_bounds_missing,
                    frame_view_bounds_missing,
                )
                view_bounds_sample_available = view_bounds_visible_for_obs | (~view_bounds_missing)
            camera_visible_for_obs = camera_visible_for_obs & view_bounds_sample_available
        sampled_pos = true_bpos + pos_noise
        sampled_vel = true_bvel + vel_noise
        last_ball_obs_step = jnp.where(refresh, state.step_count, state.last_ball_obs_step)

        still_dropout = state.ball_obs_dropout_remaining > 0
        u = jax.vmap(lambda k: jax.random.uniform(k, (), dtype=jnp.float32))(key_dropout)
        if bool(self.cfg.ball_obs_dropout_on_refresh_only):
            dropout_start_allowed = refresh & camera_visible_for_obs
        else:
            dropout_start_allowed = jnp.ones_like(still_dropout, dtype=bool)
        burst_start = dropout_start_allowed & (~still_dropout) & (u < float(self.cfg.ball_obs_dropout_burst_prob))
        single_start = (
            dropout_start_allowed
            & (~still_dropout)
            & (~burst_start)
            & (u < float(self.cfg.ball_obs_dropout_burst_prob) + float(self.cfg.ball_obs_dropout_prob))
        )
        single_duration = jax.vmap(
            lambda k: jax.random.randint(
                k,
                (),
                minval=1,
                maxval=max(2, int(self.cfg.ball_obs_dropout_max_steps) + 1),
                dtype=jnp.int32,
            )
        )(key_dropout_duration)
        burst_duration = jax.vmap(
            lambda k: jax.random.randint(
                k,
                (),
                minval=1,
                maxval=max(2, int(self.cfg.ball_obs_dropout_burst_max_steps) + 1),
                dtype=jnp.int32,
            )
        )(key_burst_duration)
        start_dropout = burst_start | single_start
        new_duration = jnp.where(burst_start, burst_duration, single_duration)
        dropout_remaining = jnp.where(
            still_dropout,
            jnp.maximum(0, state.ball_obs_dropout_remaining - 1),
            jnp.where(start_dropout, jnp.maximum(0, new_duration - 1), 0),
        )
        blocked_by_dropout = still_dropout | start_dropout
        sample_available = refresh & camera_visible_for_obs & (~blocked_by_dropout)
        previous_age_seconds = state.ball_obs_age_seconds
        missing_on_refresh = refresh & (~sample_available)
        missing_streak_started = missing_on_refresh & (~state.ball_obs_missing_since_sample)
        reacquired_after_missing = (
            sample_available & state.ball_obs_missing_since_sample
        )
        lost_timeout_s = max(0.0, float(self.cfg.lost_ball_timeout_ms)) * 1e-3
        reacquired_after_lost = (
            sample_available
            & (lost_timeout_s > 0.0)
            & (previous_age_seconds >= lost_timeout_s)
        )
        missing_since_sample = jnp.where(
            sample_available,
            False,
            state.ball_obs_missing_since_sample | missing_on_refresh,
        )
        valid_pos = jnp.where(sample_available[:, None], sampled_pos, state.ball_obs_valid_pos)
        valid_vel = jnp.where(sample_available[:, None], sampled_vel, state.ball_obs_valid_vel)
        cached_pos = jnp.where(sample_available[:, None], sampled_pos, state.cached_ball_obs_pos)
        cached_vel = jnp.where(sample_available[:, None], sampled_vel, state.cached_ball_obs_vel)
        if bool(self.cfg.ball_obs_age_tracks_stale):
            age_seconds = jnp.where(sample_available, 0.0, state.ball_obs_age_seconds + self.dt)
        else:
            age_seconds = jnp.where(blocked_by_dropout, state.ball_obs_age_seconds + self.dt, 0.0)
        dropout_steps_total = state.ball_obs_dropout_steps_total + blocked_by_dropout.astype(jnp.int32)
        burst_count = state.ball_obs_burst_count + burst_start.astype(jnp.int32)

        state = state._replace(
            rng=next_rng,
            cached_ball_obs_pos=cached_pos,
            cached_ball_obs_vel=cached_vel,
            last_ball_obs_step=last_ball_obs_step,
            ball_obs_valid_pos=valid_pos,
            ball_obs_valid_vel=valid_vel,
            ball_obs_age_seconds=age_seconds,
            ball_obs_missing_since_sample=missing_since_sample,
            ball_obs_dropout_remaining=dropout_remaining,
            ball_obs_dropout_steps_total=dropout_steps_total,
            ball_obs_burst_count=burst_count,
        )
        raw_base_obs = self._make_obs(state, valid_pos, valid_vel, age_seconds)
        obs_buffer = jnp.concatenate([state.obs_buffer[:, 1:, :], raw_base_obs[:, None, :]], axis=1)
        obs_idx = (self.max_obs_latency_steps - state.obs_latency_steps).astype(jnp.int32)
        delayed_base_obs = obs_buffer[jnp.arange(obs_buffer.shape[0]), obs_idx]
        obs = self._augment_obs(state, delayed_base_obs)
        if self.high_latency_obs_prev_frames > 0:
            obs_history = jnp.concatenate(
                [state.obs_history[:, 1:, :], delayed_base_obs[:, None, :]],
                axis=1,
            )
        else:
            obs_history = state.obs_history
        if self.high_latency_action_prev_frames > 0:
            action_history = jnp.concatenate(
                [state.action_history[:, 1:, :], state.action_buffer[:, -1:, :]],
                axis=1,
            )
        else:
            action_history = state.action_history
        state = state._replace(obs_buffer=obs_buffer, obs_history=obs_history, action_history=action_history)
        lost_timeout_s = max(0.0, float(self.cfg.lost_ball_timeout_ms)) * 1e-3
        lost_active = (lost_timeout_s > 0.0) & (age_seconds >= lost_timeout_s)
        lost_entered = lost_active & (previous_age_seconds < lost_timeout_s)
        metrics = {
            "ball_obs_refresh_due": refresh.astype(jnp.float32),
            "ball_obs_sample_available": sample_available.astype(jnp.float32),
            "ball_obs_camera_available": camera_visible_for_obs.astype(jnp.float32),
            "ball_obs_view_available": view_bounds_visible_for_obs.astype(jnp.float32),
            "ball_obs_missing_on_refresh": missing_on_refresh.astype(jnp.float32),
            "ball_obs_missing_streak_started": missing_streak_started.astype(jnp.float32),
            "ball_obs_stale_active": (age_seconds > 0.0).astype(jnp.float32),
            "ball_obs_dropout_active": blocked_by_dropout.astype(jnp.float32),
            "ball_obs_lost_active": lost_active.astype(jnp.float32),
            "ball_obs_lost_entered": lost_entered.astype(jnp.float32),
            "ball_obs_reacquired": reacquired_after_missing.astype(jnp.float32),
            "ball_obs_reacquired_after_missing": reacquired_after_missing.astype(jnp.float32),
            "ball_obs_reacquired_after_lost": reacquired_after_lost.astype(jnp.float32),
        }
        return state, obs, metrics

    def _ball_contact_flags(self, data) -> tuple[jax.Array, jax.Array]:
        geom = data.contact.geom
        dist = data.contact.dist
        if geom.ndim == 2:
            geom = geom[None, ...]
            dist = dist[None, ...]
        g0 = geom[..., 0]
        g1 = geom[..., 1]
        pair = ((g0 == self.ball_geom_id) & (g1 == self.racket_geom_id)) | (
            (g0 == self.racket_geom_id) & (g1 == self.ball_geom_id)
        )
        ball_pair = (g0 == self.ball_geom_id) | (g1 == self.ball_geom_id)
        racket_pair = ((g0 == self.racket_geom_id) | (g1 == self.racket_geom_id))
        proxy0 = jnp.any(g0[..., None] == self.non_racket_geom_ids, axis=-1)
        proxy1 = jnp.any(g1[..., None] == self.non_racket_geom_ids, axis=-1)
        non_racket_pair = ball_pair & (~racket_pair) & (proxy0 | proxy1)
        close = dist <= 0.002
        return jnp.any(pair & close, axis=-1), jnp.any(non_racket_pair & close, axis=-1)

    def _reward(
        self,
        data,
        action: jax.Array,
        camera_terms: dict[str, jax.Array],
        da: jax.Array,
        action_clip_excess: jax.Array,
        arm_limiter_pen: jax.Array,
        bpos: jax.Array,
        bvel: jax.Array,
        rpos: jax.Array,
        rvel: jax.Array,
        rel: jax.Array,
        rel_local: jax.Array,
        racket_normal: jax.Array,
        predicted_apex_z: jax.Array,
        hit_count: jax.Array,
        new_hit: jax.Array,
        rewardable_hit: jax.Array,
        failed_hit: jax.Array,
        ignored_fast_hit: jax.Array,
        hit_cadence_reward: jax.Array,
        hit_min_interval_penalty: jax.Array,
        fast_hit_penalty: jax.Array,
        hit_camera_visible: jax.Array,
        hit_camera_in_lower_band: jax.Array,
        hit_camera_v_frac: jax.Array,
        hit_camera_in_margin: jax.Array,
        other_ball_contact: jax.Array,
        in_contact: jax.Array,
        contact_hold_steps: jax.Array,
        rel_speed: jax.Array,
        cmd_qvel: jax.Array,
        prev_arm_qvel: jax.Array,
        racket_anchor: jax.Array,
        chest_target_offset: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del cmd_qvel
        if self.cfg.hit_apex_target_abs_z is None:
            target_ball_z = racket_anchor[:, 2] + float(self.cfg.target_height)
            target_hit_apex_z = racket_anchor[:, 2] + float(self.cfg.hit_height_center)
        else:
            target_ball_z = jnp.full_like(
                racket_anchor[:, 2], float(self.cfg.hit_apex_target_abs_z)
            )
            target_hit_apex_z = target_ball_z
        upward_vz = jnp.maximum(0.0, bvel[:, 2])
        dz_up = predicted_apex_z - target_ball_z
        dz_down = bpos[:, 2] - target_ball_z
        ball_height_reward = jnp.where(
            upward_vz > 0.0,
            jnp.exp(-7.0 * dz_up * dz_up),
            jnp.where(rel[:, 2] > 0.05, 0.25 * jnp.exp(-10.0 * dz_down * dz_down), 0.0),
        )
        xy_track_pen = jnp.sum(rel[:, :2] ** 2, axis=-1)
        racket_center_pen = jnp.sum((rpos - racket_anchor) ** 2, axis=-1)
        racket_xy_dist = jnp.linalg.norm((rpos - racket_anchor)[:, :2], axis=-1)
        racket_xy_gauss = jnp.exp(-0.5 * (racket_xy_dist / max(1e-6, float(self.cfg.racket_xy_gauss_sigma))) ** 2)
        racket_xy_gauss_pen = 1.0 - racket_xy_gauss
        waist_pos = data.xpos[:, self.waist_body_id] if self.waist_body_id >= 0 else racket_anchor
        chest_target = waist_pos + chest_target_offset
        racket_chest_xy_pen = jnp.sum((rpos[:, :2] - chest_target[:, :2]) ** 2, axis=-1)
        racket_chest_z_pen = (rpos[:, 2] - chest_target[:, 2]) ** 2
        ball_anchor_xy_pen = jnp.sum((bpos[:, :2] - chest_target[:, :2]) ** 2, axis=-1)
        rel_height_bonus = jnp.exp(
            -0.5 * ((rel[:, 2] - float(self.cfg.rel_height_center)) / max(1e-6, float(self.cfg.rel_height_sigma))) ** 2
        )

        posture_q = data.qpos[:, self.posture_qadr]
        posture_pen = jnp.mean((posture_q - self.posture_targets) ** 2, axis=-1)
        base_pose = jnp.stack(
            [
                data.qpos[:, self.base_x_qadr],
                data.qpos[:, self.base_y_qadr],
                data.qpos[:, self.base_yaw_qadr],
            ],
            axis=-1,
        )
        base_pose_err = base_pose - self.initial_base_pose
        base_pose_err = base_pose_err.at[:, 2].set((base_pose_err[:, 2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi)
        base_pose_pen = jnp.sum(base_pose_err**2, axis=-1)
        base_to_ball_world = bpos[:, :2] - base_pose[:, :2]
        base_yaw = base_pose[:, 2]
        c_yaw = jnp.cos(base_yaw)
        s_yaw = jnp.sin(base_yaw)
        ball_base_x = c_yaw * base_to_ball_world[:, 0] + s_yaw * base_to_ball_world[:, 1]
        ball_base_x_excess = jnp.maximum(0.0, jnp.abs(ball_base_x) - float(self.cfg.ball_base_x_soft_limit))
        ball_base_x_pen = ball_base_x_excess * ball_base_x_excess
        ball_base_vx = c_yaw * bvel[:, 0] + s_yaw * bvel[:, 1]
        ball_base_vy = -s_yaw * bvel[:, 0] + c_yaw * bvel[:, 1]
        ball_base_vxy_pen = ball_base_vx * ball_base_vx + ball_base_vy * ball_base_vy
        ball_vxy_pen = jnp.sum(bvel[:, :2] ** 2, axis=-1)
        gravity_abs = max(1e-6, abs(float(self.default_gravity_z)))
        time_to_apex = upward_vz / gravity_abs
        time_to_next_contact = 2.0 * time_to_apex
        predicted_apex_xy = bpos[:, :2] + bvel[:, :2] * time_to_apex[:, None]
        predicted_next_contact_xy = bpos[:, :2] + bvel[:, :2] * time_to_next_contact[:, None]
        base_to_apex_world = predicted_apex_xy - base_pose[:, :2]
        apex_view_x = c_yaw * base_to_apex_world[:, 0] + s_yaw * base_to_apex_world[:, 1]
        apex_view_y = -s_yaw * base_to_apex_world[:, 0] + c_yaw * base_to_apex_world[:, 1]
        hit_apex_view_center_pen = (
            ((apex_view_x - float(self.cfg.ball_view_x_target_m)) / max(1e-6, float(self.cfg.hit_apex_view_center_sigma_m))) ** 2
            + ((apex_view_y - float(self.cfg.ball_view_y_target_m)) / max(1e-6, float(self.cfg.hit_apex_view_center_sigma_m))) ** 2
        )
        hit_next_contact_anchor_pen = (
            jnp.sum((predicted_next_contact_xy - racket_anchor[:, :2]) ** 2, axis=-1)
            / max(1e-6, float(self.cfg.hit_next_contact_anchor_sigma_m)) ** 2
        )
        ball_view_x = ball_base_x
        ball_view_y = -s_yaw * base_to_ball_world[:, 0] + c_yaw * base_to_ball_world[:, 1]
        ball_view_z = bpos[:, 2]
        ball_view_xy_center_pen = (
            ((ball_view_x - float(self.cfg.ball_view_x_target_m)) / max(1e-6, float(self.cfg.ball_view_x_sigma_m))) ** 2
            + ((ball_view_y - float(self.cfg.ball_view_y_target_m)) / max(1e-6, float(self.cfg.ball_view_y_sigma_m))) ** 2
        )
        z_ideal_low = float(self.cfg.ball_view_z_ideal_m[0])
        z_ideal_high = float(self.cfg.ball_view_z_ideal_m[1])
        ball_view_z_ideal_excess = jnp.maximum(0.0, z_ideal_low - ball_view_z) + jnp.maximum(
            0.0, ball_view_z - z_ideal_high
        )
        ball_view_z_ideal_pen = (
            ball_view_z_ideal_excess / max(1e-6, float(self.cfg.ball_view_z_sigma_m))
        ) ** 2
        x_bound_low = float(self.cfg.ball_view_x_bounds_m[0])
        x_bound_high = float(self.cfg.ball_view_x_bounds_m[1])
        y_bound_low = float(self.cfg.ball_view_y_bounds_m[0])
        y_bound_high = float(self.cfg.ball_view_y_bounds_m[1])
        z_bound_low = float(self.cfg.ball_view_z_bounds_m[0])
        z_bound_high = float(self.cfg.ball_view_z_bounds_m[1])
        ball_view_bounds_pen = (
            jnp.maximum(0.0, x_bound_low - ball_view_x) ** 2
            + jnp.maximum(0.0, ball_view_x - x_bound_high) ** 2
            + jnp.maximum(0.0, y_bound_low - ball_view_y) ** 2
            + jnp.maximum(0.0, ball_view_y - y_bound_high) ** 2
            + jnp.maximum(0.0, z_bound_low - ball_view_z) ** 2
            + jnp.maximum(0.0, ball_view_z - z_bound_high) ** 2
        )
        ball_view_in_bounds = (
            (ball_view_x >= x_bound_low)
            & (ball_view_x <= x_bound_high)
            & (ball_view_y >= y_bound_low)
            & (ball_view_y <= y_bound_high)
            & (ball_view_z >= z_bound_low)
            & (ball_view_z <= z_bound_high)
        )
        ball_view_z_ideal = (ball_view_z >= z_ideal_low) & (ball_view_z <= z_ideal_high)
        ball_view_vxy_excess = (
            jnp.maximum(0.0, jnp.abs(ball_base_vx) - float(self.cfg.ball_view_vxy_soft_limit_m_s)) ** 2
            + jnp.maximum(0.0, jnp.abs(ball_base_vy) - float(self.cfg.ball_view_vxy_soft_limit_m_s)) ** 2
        )
        post_hit_ball_xy_dist = jnp.linalg.norm(bpos[:, :2] - chest_target[:, :2], axis=-1)
        apex_soft_excess = jnp.maximum(0.0, predicted_apex_z - (target_ball_z + float(self.cfg.apex_soft_limit_margin)))
        apex_soft_pen = float(self.cfg.apex_soft_penalty_weight) * apex_soft_excess * apex_soft_excess
        ball_xy_soft_excess = jnp.maximum(0.0, post_hit_ball_xy_dist - float(self.cfg.ball_xy_soft_limit_radius))
        ball_xy_soft_pen = jnp.where(
            (hit_count > 0) | (upward_vz > 0.0) | (bpos[:, 2] > racket_anchor[:, 2]),
            float(self.cfg.ball_xy_soft_penalty_weight) * ball_xy_soft_excess * ball_xy_soft_excess,
            0.0,
        )
        post_hit_ball_xy_score = jnp.exp(
            -0.5 * (post_hit_ball_xy_dist / max(1e-6, float(self.cfg.post_hit_ball_xy_sigma))) ** 2
        )
        post_hit_survival_reward = jnp.where(
            (hit_count > 0) & (bpos[:, 2] >= racket_anchor[:, 2] - 0.02),
            post_hit_ball_xy_score - float(self.cfg.post_hit_ball_vxy_penalty_weight) * ball_vxy_pen,
            0.0,
        )
        drop_dist = bpos[:, 2] - rpos[:, 2]
        vz_abs = jnp.maximum(1e-5, -bvel[:, 2])
        time_to_racket = drop_dist / vz_abs
        projected_ball_xy = bpos[:, :2] + bvel[:, :2] * time_to_racket[:, None]
        descending_intercept_xy_err = jnp.linalg.norm(projected_ball_xy - rpos[:, :2], axis=-1)
        descending_intercept_reward = jnp.where(
            (hit_count > 0) & (bvel[:, 2] < -1e-4) & (bpos[:, 2] > rpos[:, 2]),
            jnp.exp(
                -0.5
                * (descending_intercept_xy_err / max(1e-6, float(self.cfg.descending_intercept_sigma))) ** 2
            ),
            0.0,
        )
        pre_hit_intercept_reward = jnp.where(
            (hit_count <= 0)
            & (bvel[:, 2] < -1e-4)
            & (bpos[:, 2] > rpos[:, 2])
            & (time_to_racket >= 0.0)
            & (time_to_racket <= float(self.cfg.pre_hit_intercept_time_max)),
            jnp.exp(
                -0.5
                * (descending_intercept_xy_err / max(1e-6, float(self.cfg.pre_hit_intercept_sigma))) ** 2
            ),
            0.0,
        )
        pre_hit_intercept_penalty_active = (
            (hit_count <= 0)
            & (bvel[:, 2] < -1e-4)
            & (bpos[:, 2] > rpos[:, 2])
            & (time_to_racket >= 0.0)
            & (time_to_racket <= float(self.cfg.pre_hit_intercept_penalty_time_max))
        )
        pre_hit_intercept_excess = jnp.maximum(
            0.0,
            descending_intercept_xy_err - float(self.cfg.pre_hit_intercept_penalty_radius),
        )
        pre_hit_intercept_penalty = jnp.where(
            pre_hit_intercept_penalty_active,
            (pre_hit_intercept_excess / max(1e-6, float(self.cfg.pre_hit_intercept_penalty_sigma))) ** 2,
            0.0,
        )
        torque_pen = jnp.mean(data.actuator_force[:, self.arm_aids_j] ** 2, axis=-1)
        sep_dist = jnp.linalg.norm(rel, axis=-1)
        sticky_contact = (
            in_contact
            & (contact_hold_steps >= int(self.cfg.stick_min_contact_steps))
            & (sep_dist <= float(self.cfg.stick_rel_dist_thresh))
            & (rel_speed <= float(self.cfg.stick_rel_speed_thresh))
        )
        stick_hold_excess = 1.0 + jnp.maximum(0, contact_hold_steps - int(self.cfg.stick_min_contact_steps)).astype(jnp.float32)
        stick_pen = jnp.where(sticky_contact, stick_hold_excess * float(self.cfg.sticky_contact_penalty_growth), 0.0)
        non_racket_contact_pen = jnp.where(
            other_ball_contact,
            float(self.cfg.non_racket_ball_contact_penalty_weight),
            0.0,
        )

        racket_z_rel = rpos[:, 2] - racket_anchor[:, 2]
        z_excess_up = jnp.maximum(0.0, racket_z_rel - float(self.cfg.racket_z_band_up))
        z_excess_down = jnp.maximum(0.0, -float(self.cfg.racket_z_band_down) - racket_z_rel)
        racket_z_band_pen = z_excess_up * z_excess_up + z_excess_down * z_excess_down
        up_drift_pen = jnp.where(
            (racket_z_rel > 0.0) & (rvel[:, 2] > float(self.cfg.racket_up_drift_vel_thresh)),
            racket_z_rel * jnp.maximum(0.0, rvel[:, 2]),
            0.0,
        )

        arm_qvel = data.qvel[:, self.arm_vadr]
        arm_vel_ratio = jnp.abs(arm_qvel) / jnp.maximum(self.arm_vel_limit_rad_s, 1e-6)
        arm_vel_exceed = jnp.maximum(arm_vel_ratio - 1.0, 0.0)
        arm_vel_limit_pen = jnp.mean(arm_vel_exceed**2, axis=-1)
        arm_qacc = (arm_qvel - prev_arm_qvel) / max(self.dt, 1e-6)
        arm_acc_ratio = jnp.abs(arm_qacc) / jnp.maximum(self.arm_acc_limit_rad_s2, 1e-6)
        arm_acc_exceed = jnp.maximum(arm_acc_ratio - 1.0, 0.0)
        arm_acc_limit_pen = jnp.mean(arm_acc_exceed**2, axis=-1)

        term_ball_height = 1.2 * ball_height_reward
        term_rel_height = float(self.cfg.rel_height_bonus_weight) * rel_height_bonus
        term_xy_track_penalty = -1.4 * xy_track_pen
        term_racket_center_penalty = -0.35 * racket_center_pen
        term_posture_penalty = -float(self.cfg.posture_weight) * posture_pen
        term_base_pose_penalty = -float(self.cfg.base_pose_weight) * base_pose_pen
        term_torque_penalty = -float(self.cfg.torque_penalty_weight) * torque_pen
        term_stick_penalty = -float(self.cfg.stick_contact_penalty_weight) * stick_pen
        term_non_racket_contact_penalty = -non_racket_contact_pen
        term_racket_chest_xy_penalty = -float(self.cfg.racket_chest_xy_penalty_weight) * racket_chest_xy_pen
        term_racket_chest_z_penalty = -float(self.cfg.racket_chest_z_penalty_weight) * racket_chest_z_pen
        term_ball_anchor_xy_penalty = -float(self.cfg.ball_anchor_xy_penalty_weight) * ball_anchor_xy_pen
        term_ball_base_x_penalty = -float(self.cfg.ball_base_x_penalty_weight) * ball_base_x_pen
        term_ball_base_vxy_penalty = -float(self.cfg.ball_base_vxy_penalty_weight) * ball_base_vxy_pen
        term_ball_vxy_penalty = -float(self.cfg.ball_vxy_penalty_weight) * ball_vxy_pen
        term_ball_view_xy_center_penalty = -float(self.cfg.ball_view_xy_center_penalty_weight) * ball_view_xy_center_pen
        term_ball_view_z_ideal_penalty = -float(self.cfg.ball_view_z_ideal_penalty_weight) * ball_view_z_ideal_pen
        term_ball_view_bounds_penalty = -float(self.cfg.ball_view_bounds_penalty_weight) * ball_view_bounds_pen
        term_ball_view_out_of_bounds_penalty = (
            -float(self.cfg.ball_view_out_of_bounds_penalty_weight) * (~ball_view_in_bounds).astype(jnp.float32)
        )
        term_ball_view_z_not_ideal_penalty = (
            -float(self.cfg.ball_view_z_not_ideal_penalty_weight) * (~ball_view_z_ideal).astype(jnp.float32)
        )
        term_ball_view_vxy_excess_penalty = (
            -float(self.cfg.ball_view_vxy_excess_penalty_weight) * ball_view_vxy_excess
        )
        term_apex_soft_penalty = -apex_soft_pen
        term_ball_xy_soft_penalty = -ball_xy_soft_pen
        term_post_hit_survival = float(self.cfg.post_hit_survival_reward_weight) * post_hit_survival_reward
        term_descending_intercept = float(self.cfg.descending_intercept_reward_weight) * descending_intercept_reward
        term_pre_hit_intercept = float(self.cfg.pre_hit_intercept_reward_weight) * pre_hit_intercept_reward
        term_pre_hit_intercept_penalty = (
            -float(self.cfg.pre_hit_intercept_penalty_weight) * pre_hit_intercept_penalty
        )
        term_racket_xy_reward = float(self.cfg.racket_xy_gauss_reward_weight) * racket_xy_gauss
        term_racket_xy_penalty = -float(self.cfg.racket_xy_gauss_penalty_weight) * racket_xy_gauss_pen
        term_racket_z_penalty = -float(self.cfg.racket_z_soft_penalty_weight) * racket_z_band_pen
        term_racket_up_drift_penalty = -float(self.cfg.racket_up_drift_penalty_weight) * up_drift_pen
        racket_up_cos = jnp.maximum(0.0, jnp.sum(racket_normal * jnp.asarray([0.0, 0.0, 1.0]), axis=-1))
        racket_flatness_err = jnp.maximum(0.0, float(self.cfg.racket_flatness_target_cos) - racket_up_cos)
        racket_flatness_pen = (
            racket_flatness_err / max(1e-6, float(self.cfg.racket_flatness_sigma))
        ) ** 2
        term_racket_flatness_penalty = (
            -float(self.cfg.racket_flatness_penalty_weight) * racket_flatness_pen
        )
        flatness_err = jnp.maximum(0.0, float(self.cfg.hit_flatness_target_cos) - racket_up_cos)
        flatness_score = jnp.exp(-0.5 * (flatness_err / max(1e-6, float(self.cfg.hit_flatness_sigma))) ** 2)
        flat_contact_pen = jnp.where(
            in_contact,
            float(self.cfg.contact_flatness_penalty_weight) * jnp.maximum(0.0, 1.0 - flatness_score),
            0.0,
        )
        term_contact_flatness_penalty = -flat_contact_pen
        term_action_penalty = -float(self.cfg.action_penalty_weight) * jnp.sum(action**2, axis=-1)
        term_action_delta_penalty = -float(self.cfg.action_delta_penalty_weight) * jnp.sum(da**2, axis=-1)
        term_action_clip_excess_penalty = -float(
            self.cfg.action_clip_excess_penalty_weight
        ) * jnp.sum(jnp.log1p(action_clip_excess * action_clip_excess), axis=-1)
        term_arm_vel_penalty = -float(self.cfg.arm_vel_limit_penalty_weight) * arm_vel_limit_pen
        term_arm_acc_penalty = -float(self.cfg.arm_acc_limit_penalty_weight) * arm_acc_limit_pen
        term_arm_limiter_penalty = -float(self.cfg.arm_limiter_penalty_weight) * arm_limiter_pen

        dense_reward = (
            term_ball_height
            + term_rel_height
            + term_xy_track_penalty
            + term_racket_center_penalty
            + term_posture_penalty
            + term_base_pose_penalty
            + term_torque_penalty
            + term_stick_penalty
            + term_non_racket_contact_penalty
            + term_racket_chest_xy_penalty
            + term_racket_chest_z_penalty
            + term_ball_anchor_xy_penalty
            + term_ball_base_x_penalty
            + term_ball_base_vxy_penalty
            + term_ball_vxy_penalty
            + term_ball_view_xy_center_penalty
            + term_ball_view_z_ideal_penalty
            + term_ball_view_bounds_penalty
            + term_ball_view_out_of_bounds_penalty
            + term_ball_view_z_not_ideal_penalty
            + term_ball_view_vxy_excess_penalty
            + term_apex_soft_penalty
            + term_ball_xy_soft_penalty
            + term_post_hit_survival
            + term_descending_intercept
            + term_pre_hit_intercept
            + term_pre_hit_intercept_penalty
            + term_racket_xy_reward
            + term_racket_xy_penalty
            + term_racket_z_penalty
            + term_racket_up_drift_penalty
            + term_racket_flatness_penalty
            + term_contact_flatness_penalty
            + camera_terms["camera_reward_dense"]
            + term_action_penalty
            + term_action_delta_penalty
            + term_action_clip_excess_penalty
            + term_arm_vel_penalty
            + term_arm_acc_penalty
            + term_arm_limiter_penalty
        )
        reward = dense_reward * self.dt

        contact_center_dist = jnp.linalg.norm(rel_local[:, :2], axis=-1)
        center_gain = jnp.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_sigma))) ** 2)
        local_center_gain = jnp.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_local_sigma))) ** 2)
        hit_bonus = float(self.cfg.hit_reward_base) + float(self.cfg.hit_reward_combo) * jnp.minimum(
            hit_count.astype(jnp.float32),
            float(self.cfg.hit_combo_count_cap),
        )
        hit_bonus = hit_bonus * jnp.maximum(0.2, center_gain * flatness_score)
        hit_height_err = jnp.abs(predicted_apex_z - target_hit_apex_z)
        hit_height_excess = jnp.maximum(0.0, hit_height_err - float(self.cfg.hit_height_tolerance))
        hit_height_pen = float(self.cfg.hit_height_penalty_weight) * hit_height_excess * hit_height_excess
        hit_vxy = jnp.linalg.norm(bvel[:, :2], axis=-1)
        hit_vxy_excess = jnp.maximum(0.0, hit_vxy - float(self.cfg.hit_vxy_soft_limit_m_s))
        hit_vxy_pen = float(self.cfg.hit_vxy_penalty_weight) * hit_vxy_excess * hit_vxy_excess
        low_hit_deficit = jnp.maximum(0.0, (target_ball_z - float(self.cfg.low_hit_apex_margin)) - predicted_apex_z)
        low_hit_pen = float(self.cfg.low_hit_penalty_weight) * low_hit_deficit * low_hit_deficit
        first_hit_apex_err = (predicted_apex_z - target_hit_apex_z) / max(
            1e-6,
            float(self.cfg.first_hit_apex_sigma),
        )
        first_hit_apex_score = jnp.exp(-0.5 * first_hit_apex_err * first_hit_apex_err)
        center_flat = float(self.cfg.center_flat_hit_reward_weight) * local_center_gain * flatness_score
        height_bonus = jnp.where(
            predicted_apex_z >= target_ball_z,
            0.35 * jnp.exp(-10.0 * (predicted_apex_z - target_ball_z) * (predicted_apex_z - target_ball_z)),
            0.0,
        )
        hit_reward_mask = new_hit & rewardable_hit
        first_hit_reward_mask = hit_reward_mask & (hit_count <= 1)
        hit_camera_safe = hit_camera_visible & hit_camera_in_margin
        hit_camera_score = jnp.where(
            hit_camera_safe,
            jnp.exp(
                -0.5
                * (
                    (hit_camera_v_frac - float(self.cfg.hit_camera_target_v_frac))
                    / max(1e-6, float(self.cfg.hit_camera_v_sigma_frac))
                )
                ** 2
            ),
            0.0,
        )
        term_hit_camera = jnp.where(
            hit_reward_mask,
            float(self.cfg.hit_camera_reward_weight) * hit_camera_score
            - float(self.cfg.hit_camera_out_of_band_penalty_weight)
            * (~hit_camera_in_lower_band).astype(jnp.float32),
            0.0,
        )
        term_hit_bonus = jnp.where(hit_reward_mask, hit_bonus, 0.0)
        term_center_flat_hit = jnp.where(hit_reward_mask, center_flat, 0.0)
        term_hit_height_bonus = jnp.where(hit_reward_mask, height_bonus, 0.0)
        term_first_hit_apex = jnp.where(
            first_hit_reward_mask,
            float(self.cfg.first_hit_apex_reward_weight)
            * first_hit_apex_score
            * jnp.maximum(0.25, local_center_gain * flatness_score),
            0.0,
        )
        term_hit_cadence_reward = jnp.where(hit_reward_mask, hit_cadence_reward, 0.0)
        term_hit_min_interval_penalty = jnp.where(hit_reward_mask, -hit_min_interval_penalty, 0.0)
        term_hit_height_penalty = jnp.where(hit_reward_mask, -hit_height_pen, 0.0)
        term_hit_vxy_penalty = jnp.where(hit_reward_mask, -hit_vxy_pen, 0.0)
        term_hit_apex_view_center_penalty = jnp.where(
            hit_reward_mask,
            -float(self.cfg.hit_apex_view_center_penalty_weight) * hit_apex_view_center_pen,
            0.0,
        )
        term_hit_next_contact_anchor_penalty = jnp.where(
            hit_reward_mask,
            -float(self.cfg.hit_next_contact_anchor_penalty_weight) * hit_next_contact_anchor_pen,
            0.0,
        )
        term_low_hit_penalty = jnp.where(hit_reward_mask, -low_hit_pen, 0.0)
        term_failed_hit_penalty = jnp.where(failed_hit, -float(self.cfg.failed_hit_penalty_weight), 0.0)
        term_fast_hit_penalty = jnp.where(ignored_fast_hit, -fast_hit_penalty, 0.0)
        reward = (
            reward
            + term_hit_bonus
            + term_center_flat_hit
            + term_hit_height_bonus
            + term_hit_camera
            + term_first_hit_apex
            + term_hit_cadence_reward
            + term_hit_min_interval_penalty
            + term_hit_height_penalty
            + term_hit_vxy_penalty
            + term_hit_apex_view_center_penalty
            + term_hit_next_contact_anchor_penalty
            + term_low_hit_penalty
            + term_failed_hit_penalty
            + term_fast_hit_penalty
        )
        terms = {
            "total": reward,
            "dense_scaled": dense_reward * self.dt,
            "ball_height": term_ball_height * self.dt,
            "rel_height": term_rel_height * self.dt,
            "xy_track_penalty": term_xy_track_penalty * self.dt,
            "racket_center_penalty": term_racket_center_penalty * self.dt,
            "posture_penalty": term_posture_penalty * self.dt,
            "base_pose_penalty": term_base_pose_penalty * self.dt,
            "torque_penalty": term_torque_penalty * self.dt,
            "stick_penalty": term_stick_penalty * self.dt,
            "non_racket_contact_penalty": term_non_racket_contact_penalty * self.dt,
            "racket_chest_xy_penalty": term_racket_chest_xy_penalty * self.dt,
            "racket_chest_z_penalty": term_racket_chest_z_penalty * self.dt,
            "ball_anchor_xy_penalty": term_ball_anchor_xy_penalty * self.dt,
            "metric/ball_view_z_high_exceeded": (ball_view_z > z_bound_high).astype(jnp.float32),
            "ball_base_x_penalty": term_ball_base_x_penalty * self.dt,
            "ball_base_vxy_penalty": term_ball_base_vxy_penalty * self.dt,
            "ball_vxy_penalty": term_ball_vxy_penalty * self.dt,
            "ball_view_xy_center_penalty": term_ball_view_xy_center_penalty * self.dt,
            "ball_view_z_ideal_penalty": term_ball_view_z_ideal_penalty * self.dt,
            "ball_view_bounds_penalty": term_ball_view_bounds_penalty * self.dt,
            "ball_view_out_of_bounds_penalty": term_ball_view_out_of_bounds_penalty * self.dt,
            "ball_view_z_not_ideal_penalty": term_ball_view_z_not_ideal_penalty * self.dt,
            "ball_view_vxy_excess_penalty": term_ball_view_vxy_excess_penalty * self.dt,
            "metric/ball_view_x": ball_view_x,
            "metric/ball_view_y": ball_view_y,
            "metric/ball_view_z": ball_view_z,
            "metric/ball_view_vx": ball_base_vx,
            "metric/ball_view_vy": ball_base_vy,
            "metric/ball_view_xy_center_pen": ball_view_xy_center_pen,
            "metric/ball_view_z_ideal_pen": ball_view_z_ideal_pen,
            "metric/ball_view_bounds_pen": ball_view_bounds_pen,
            "metric/ball_view_vxy_excess_pen": ball_view_vxy_excess,
            "metric/ball_view_in_bounds": ball_view_in_bounds.astype(jnp.float32),
            "metric/ball_view_z_ideal": ball_view_z_ideal.astype(jnp.float32),
            "apex_soft_penalty": term_apex_soft_penalty * self.dt,
            "ball_xy_soft_penalty": term_ball_xy_soft_penalty * self.dt,
            "post_hit_survival": term_post_hit_survival * self.dt,
            "descending_intercept": term_descending_intercept * self.dt,
            "pre_hit_intercept": term_pre_hit_intercept * self.dt,
            "pre_hit_intercept_penalty": term_pre_hit_intercept_penalty * self.dt,
            "racket_xy_reward": term_racket_xy_reward * self.dt,
            "racket_xy_penalty": term_racket_xy_penalty * self.dt,
            "racket_z_penalty": term_racket_z_penalty * self.dt,
            "racket_up_drift_penalty": term_racket_up_drift_penalty * self.dt,
            "contact_flatness_penalty": term_contact_flatness_penalty * self.dt,
            "racket_flatness_penalty": term_racket_flatness_penalty * self.dt,
            "metric/racket_up_cos": racket_up_cos,
            "metric/racket_flatness_pen": racket_flatness_pen,
            "camera_reward_dense": camera_terms["camera_reward_dense"] * self.dt,
            "camera_pixel_center_penalty": camera_terms["camera_pixel_center_penalty"] * self.dt,
            "camera_visibility_penalty": camera_terms["camera_visibility_penalty"] * self.dt,
            "camera_depth_penalty": camera_terms["camera_depth_penalty"] * self.dt,
            "camera_box_penalty": camera_terms["camera_box_penalty"] * self.dt,
            "camera_visible_penalty": camera_terms["camera_visible_penalty"] * self.dt,
            "camera_top_margin_penalty": camera_terms["camera_top_margin_penalty"] * self.dt,
            "action_penalty": term_action_penalty * self.dt,
            "action_delta_penalty": term_action_delta_penalty * self.dt,
            "action_clip_excess_penalty": term_action_clip_excess_penalty * self.dt,
            "arm_vel_penalty": term_arm_vel_penalty * self.dt,
            "arm_acc_penalty": term_arm_acc_penalty * self.dt,
            "arm_limiter_penalty": term_arm_limiter_penalty * self.dt,
            # These control-rate measurements remain meaningful when every
            # downstream limiter/governor is disabled.  In that configuration
            # the older arm_actual_* limiter diagnostics stay at zero by
            # construction, while these are the exact ratios that drive the
            # reward-only qvel/qacc exceedance terms above.
            "metric/arm_qvel_limit_utilization_max": jnp.max(arm_vel_ratio, axis=-1),
            "metric/arm_qacc_limit_utilization_max": jnp.max(arm_acc_ratio, axis=-1),
            "metric/arm_qvel_limit_exceed_fraction": jnp.mean(
                (arm_vel_ratio > 1.0).astype(jnp.float32), axis=-1
            ),
            "metric/arm_qacc_limit_exceed_fraction": jnp.mean(
                (arm_acc_ratio > 1.0).astype(jnp.float32), axis=-1
            ),
            "hit_bonus": term_hit_bonus,
            "center_flat_hit": term_center_flat_hit,
            "hit_height_bonus": term_hit_height_bonus,
            "hit_camera": term_hit_camera,
            "metric/hit_camera_event": new_hit.astype(jnp.float32),
            "metric/hit_camera_visible_event": (new_hit & hit_camera_visible).astype(jnp.float32),
            "metric/hit_camera_in_margin_event": (new_hit & hit_camera_safe).astype(jnp.float32),
            "metric/hit_camera_lower_band_event": (new_hit & hit_camera_in_lower_band).astype(jnp.float32),
            "metric/hit_camera_v_frac_sum": jnp.where(
                new_hit & hit_camera_visible,
                hit_camera_v_frac,
                0.0,
            ),
            "metric/hit_event_count": new_hit.astype(jnp.float32),
            "metric/hit_vxy_sum": jnp.where(new_hit, hit_vxy, 0.0),
            "metric/hit_contact_center_dist_sum": jnp.where(new_hit, contact_center_dist, 0.0),
            "metric/hit_racket_up_cos_sum": jnp.where(new_hit, racket_up_cos, 0.0),
            "metric/hit_apex_rel_height_sum": jnp.where(
                new_hit,
                predicted_apex_z - racket_anchor[:, 2],
                0.0,
            ),
            "first_hit_apex": term_first_hit_apex,
            "hit_cadence_reward": term_hit_cadence_reward,
            "hit_min_interval_penalty": term_hit_min_interval_penalty,
            "hit_height_penalty": term_hit_height_penalty,
            "hit_vxy_penalty": term_hit_vxy_penalty,
            "hit_apex_view_center_penalty": term_hit_apex_view_center_penalty,
            "hit_next_contact_anchor_penalty": term_hit_next_contact_anchor_penalty,
            "metric/hit_apex_view_x_sum": jnp.where(new_hit, apex_view_x, 0.0),
            "metric/hit_apex_view_y_sum": jnp.where(new_hit, apex_view_y, 0.0),
            "metric/hit_next_contact_anchor_err_sum": jnp.where(
                new_hit,
                jnp.sqrt(jnp.maximum(hit_next_contact_anchor_pen, 0.0))
                * max(1e-6, float(self.cfg.hit_next_contact_anchor_sigma_m)),
                0.0,
            ),
            "low_hit_penalty": term_low_hit_penalty,
            "failed_hit_penalty": term_failed_hit_penalty,
            "fast_hit_penalty": term_fast_hit_penalty,
        }
        terms.update({name: value for name, value in camera_terms.items() if name.startswith("metric/")})
        return reward, terms

    def _virtual_camera_pose(self, data) -> tuple[jax.Array, jax.Array, bool]:
        n = data.xpos.shape[0]
        if self.vc_pose_mode == "base_extrinsic":
            if self.virtual_camera_base_body_id < 0:
                cam_pos = jnp.zeros((n, 3), dtype=jnp.float32)
                cam_R = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n, 3, 3))
                return cam_pos, cam_R, False
            base_pos = data.xpos[:, self.virtual_camera_base_body_id]
            base_R = data.xmat[:, self.virtual_camera_base_body_id].reshape((n, 3, 3))
            cam_pos = base_pos + jnp.einsum("nij,j->ni", base_R, self.vc_base_pos)
            cam_R = jnp.einsum("nij,jk->nik", base_R, self.vc_base_R)
            return cam_pos, cam_R, True

        if self.virtual_camera_body_id < 0:
            cam_pos = jnp.zeros((n, 3), dtype=jnp.float32)
            cam_R = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (n, 3, 3))
            return cam_pos, cam_R, False
        body_pos = data.xpos[:, self.virtual_camera_body_id]
        body_R = data.xmat[:, self.virtual_camera_body_id].reshape((n, 3, 3))
        mount_offset = self.vc_mount_pos + self.vc_mount_R @ self.vc_optical_pos
        cam_pos = body_pos + jnp.einsum("nij,j->ni", body_R, mount_offset)
        cam_R = jnp.einsum("nij,jk->nik", body_R, self.vc_mount_R @ self.vc_mount_to_camera_R)
        return cam_pos, cam_R, True

    def _camera_reward_terms(self, data, bpos: jax.Array) -> dict[str, jax.Array]:
        n = bpos.shape[0]
        zeros = jnp.zeros((n,), dtype=jnp.float32)
        terms = {
            "camera_reward_dense": zeros,
            "camera_pixel_center_penalty": zeros,
            "camera_visibility_penalty": zeros,
            "camera_depth_penalty": zeros,
            "camera_box_penalty": zeros,
            "camera_visible_penalty": zeros,
            "camera_top_margin_penalty": zeros,
            "metric/camera_available": zeros,
            "metric/camera_in_front": zeros,
            "metric/camera_in_depth": zeros,
            "metric/camera_in_frustum": zeros,
            "metric/camera_in_image": zeros,
            "metric/camera_in_margin": zeros,
            "metric/camera_visible": zeros,
            "metric/camera_pixel_center_pen": zeros,
            "metric/camera_pixel_margin_pen": zeros,
            "metric/camera_top_margin_pen": zeros,
            "metric/camera_depth_pen": zeros,
            "metric/ball_cam_x": zeros,
            "metric/ball_cam_y": zeros,
            "metric/ball_cam_z": zeros,
            "metric/ball_pixel_u": zeros,
            "metric/ball_pixel_v": zeros,
        }
        if self.cfg.camera_visibility_mode == "off":
            return terms

        cam_pos, cam_R, camera_available = self._virtual_camera_pose(data)
        if not camera_available:
            return terms
        p_cam = jnp.einsum("nij,nj->ni", jnp.swapaxes(cam_R, 1, 2), bpos - cam_pos)
        x = p_cam[:, 0]
        y = p_cam[:, 1]
        z = p_cam[:, 2]
        has_projection = z > 1e-6
        z_safe = jnp.where(has_projection, z, 1.0)

        width = float(self.cfg.camera_image_width)
        height = float(self.cfg.camera_image_height)
        fx = float(self.cfg.camera_fx)
        fy = float(self.cfg.camera_fy)
        cx = float(self.cfg.camera_cx)
        cy = float(self.cfg.camera_cy)
        margin = float(self.cfg.camera_pixel_margin)
        u = fx * (x / z_safe) + cx
        # D455 hand-eye calibration uses the OpenCV optical frame:
        # +x right, +y down, +z forward. Pixel v therefore uses a plus sign.
        v = cy + fy * (y / z_safe)

        in_front = has_projection
        in_depth = has_projection & (z >= float(self.cfg.camera_min_depth)) & (z <= float(self.cfg.camera_max_depth))
        x_angle = jnp.arctan2(x, z_safe)
        y_angle = jnp.arctan2(y, z_safe)
        h_half = float(np.deg2rad(float(self.cfg.camera_hfov_deg) * 0.5))
        v_half = float(np.deg2rad(float(self.cfg.camera_vfov_deg) * 0.5))
        x_angle_excess = jnp.maximum(0.0, jnp.abs(x_angle) - h_half)
        y_angle_excess = jnp.maximum(0.0, jnp.abs(y_angle) - v_half)
        in_frustum = has_projection & (x_angle_excess <= 0.0) & (y_angle_excess <= 0.0)
        in_image = has_projection & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        in_margin = has_projection & (u >= margin) & (u <= (width - margin)) & (v >= margin) & (v <= (height - margin))
        visible = in_front & in_depth & in_frustum & in_image

        du = (u - cx) / max(0.5 * width, 1e-6)
        dv = (v - cy) / max(0.5 * height, 1e-6)
        center_pen = du * du + dv * dv
        depth_low = jnp.maximum(0.0, float(self.cfg.camera_min_depth) - z)
        depth_high = jnp.maximum(0.0, z - float(self.cfg.camera_max_depth))
        depth_pen = (depth_low / max(float(self.cfg.camera_min_depth), 1e-6)) ** 2 + (
            depth_high / max(float(self.cfg.camera_max_depth), 1e-6)
        ) ** 2
        u_low = jnp.maximum(0.0, margin - u)
        u_high = jnp.maximum(0.0, u - (width - margin))
        v_low = jnp.maximum(0.0, margin - v)
        v_high = jnp.maximum(0.0, v - (height - margin))
        margin_pen = (
            (u_low / max(width, 1e-6)) ** 2
            + (u_high / max(width, 1e-6)) ** 2
            + (v_low / max(height, 1e-6)) ** 2
            + (v_high / max(height, 1e-6)) ** 2
        )
        frustum_pen = (x_angle_excess / max(h_half, 1e-6)) ** 2 + (y_angle_excess / max(v_half, 1e-6)) ** 2
        x_excess = jnp.maximum(0.0, jnp.abs(x) - float(self.cfg.camera_box_half_width))
        y_excess = jnp.maximum(0.0, jnp.abs(y) - float(self.cfg.camera_box_half_height))
        z_low_excess = jnp.maximum(0.0, float(self.cfg.camera_box_depth_min) - z)
        z_high_excess = jnp.maximum(0.0, z - float(self.cfg.camera_box_depth_max))
        box_pen = x_excess * x_excess + y_excess * y_excess + z_low_excess * z_low_excess + z_high_excess * z_high_excess
        top_margin_pen = (jnp.maximum(0.0, margin - v) / max(height, 1e-6)) ** 2

        # Match the CPU env: if the ball is behind/on the optical plane,
        # geometric camera penalties are zero and only the optional visible
        # fixed penalty can apply.
        center_pen = jnp.where(has_projection, center_pen, 0.0)
        depth_pen = jnp.where(has_projection, depth_pen, 0.0)
        margin_pen = jnp.where(has_projection, margin_pen, 0.0)
        frustum_pen = jnp.where(has_projection, frustum_pen, 0.0)
        box_pen = jnp.where(has_projection, box_pen, 0.0)
        top_margin_pen = jnp.where(has_projection, top_margin_pen, 0.0)
        terms.update(
            {
                "metric/camera_available": jnp.ones((n,), dtype=jnp.float32),
                "metric/camera_in_front": in_front.astype(jnp.float32),
                "metric/camera_in_depth": in_depth.astype(jnp.float32),
                "metric/camera_in_frustum": in_frustum.astype(jnp.float32),
                "metric/camera_in_image": in_image.astype(jnp.float32),
                "metric/camera_in_margin": in_margin.astype(jnp.float32),
                "metric/camera_visible": visible.astype(jnp.float32),
                "metric/camera_pixel_center_pen": center_pen,
                "metric/camera_pixel_margin_pen": margin_pen,
                "metric/camera_top_margin_pen": top_margin_pen,
                "metric/camera_depth_pen": depth_pen,
                "metric/ball_cam_x": x,
                "metric/ball_cam_y": y,
                "metric/ball_cam_z": z,
                "metric/ball_pixel_u": jnp.where(has_projection, u, 0.0),
                "metric/ball_pixel_v": jnp.where(has_projection, v, 0.0),
            }
        )

        if self.cfg.camera_visibility_mode == "box":
            box_term = -float(self.cfg.camera_box_penalty_weight) * box_pen
            dense, (box_term,) = self._clip_camera_dense_terms((box_term,))
            terms.update({"camera_reward_dense": dense, "camera_box_penalty": box_term})
        elif self.cfg.camera_visibility_mode == "frustum":
            vis_term = -float(self.cfg.camera_visibility_penalty_weight) * frustum_pen
            depth_term = -float(self.cfg.camera_depth_penalty_weight) * depth_pen
            dense, (vis_term, depth_term) = self._clip_camera_dense_terms((vis_term, depth_term))
            terms.update(
                {
                    "camera_reward_dense": dense,
                    "camera_visibility_penalty": vis_term,
                    "camera_depth_penalty": depth_term,
                }
            )
        elif self.cfg.camera_visibility_mode == "pixel":
            center_term = -float(self.cfg.camera_center_weight) * center_pen
            vis_term = -float(self.cfg.camera_visibility_penalty_weight) * margin_pen
            depth_term = -float(self.cfg.camera_depth_penalty_weight) * depth_pen
            top_term = -float(self.cfg.camera_top_margin_penalty_weight) * top_margin_pen
            visible_term = jnp.where(
                (float(self.cfg.camera_visible_penalty_weight) > 0.0) & (~visible),
                -float(self.cfg.camera_visible_penalty_weight),
                0.0,
            )
            dense, (center_term, vis_term, depth_term, top_term, visible_term) = self._clip_camera_dense_terms(
                (center_term, vis_term, depth_term, top_term, visible_term)
            )
            terms.update(
                {
                    "camera_reward_dense": dense,
                    "camera_pixel_center_penalty": center_term,
                    "camera_visibility_penalty": vis_term,
                    "camera_depth_penalty": depth_term,
                    "camera_visible_penalty": visible_term,
                    "camera_top_margin_penalty": top_term,
                }
            )
        return terms

    def _clip_camera_dense_terms(self, terms: tuple[jax.Array, ...]) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        dense = sum(terms)
        clip = float(self.cfg.camera_dense_penalty_clip)
        if clip <= 0.0:
            return dense, terms
        dense_clipped = jnp.maximum(dense, -clip)
        scale = jnp.where(dense < -clip, dense_clipped / jnp.minimum(dense, -1e-6), 1.0)
        return dense_clipped, tuple(term * scale for term in terms)

    def _termination_terms(
        self,
        data,
        bpos: jax.Array,
        rpos: jax.Array,
        racket_anchor: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        racket_z_rel = rpos[:, 2] - racket_anchor[:, 2]
        racket_too_high = racket_z_rel > float(self.cfg.racket_z_hard_limit_up)
        racket_too_low = racket_z_rel < -float(self.cfg.racket_z_hard_limit_down)
        if not bool(self.cfg.terminate_on_racket_z_limit):
            racket_too_high = jnp.zeros_like(racket_too_high, dtype=bool)
            racket_too_low = jnp.zeros_like(racket_too_low, dtype=bool)
        base_q = jnp.stack(
            [
                data.qpos[:, self.base_x_qadr],
                data.qpos[:, self.base_y_qadr],
                data.qpos[:, self.base_yaw_qadr],
            ],
            axis=-1,
        )
        bpos_base = self._point_to_base(bpos, base_q)
        ball_view_x = bpos_base[:, 0]
        ball_view_y = bpos_base[:, 1]
        ball_view_z = bpos[:, 2]
        ball_view_x_too_low = ball_view_x < float(self.cfg.ball_view_x_bounds_m[0])
        ball_view_x_too_high = ball_view_x > float(self.cfg.ball_view_x_bounds_m[1])
        ball_view_y_too_low = ball_view_y < float(self.cfg.ball_view_y_bounds_m[0])
        ball_view_y_too_high = ball_view_y > float(self.cfg.ball_view_y_bounds_m[1])
        ball_view_z_too_low = ball_view_z < float(self.cfg.ball_view_z_bounds_m[0])
        ball_view_z_too_high = ball_view_z > float(self.cfg.ball_view_z_bounds_m[1])
        if not bool(self.cfg.terminate_on_ball_view_bounds):
            ball_view_x_too_low = jnp.zeros_like(ball_view_x_too_low, dtype=bool)
            ball_view_x_too_high = jnp.zeros_like(ball_view_x_too_high, dtype=bool)
            ball_view_y_too_low = jnp.zeros_like(ball_view_y_too_low, dtype=bool)
            ball_view_y_too_high = jnp.zeros_like(ball_view_y_too_high, dtype=bool)
            ball_view_z_too_low = jnp.zeros_like(ball_view_z_too_low, dtype=bool)
            ball_view_z_too_high = jnp.zeros_like(ball_view_z_too_high, dtype=bool)
        else:
            if not bool(self.cfg.terminate_on_ball_view_x_bounds):
                ball_view_x_too_low = jnp.zeros_like(ball_view_x_too_low, dtype=bool)
                ball_view_x_too_high = jnp.zeros_like(ball_view_x_too_high, dtype=bool)
            if not bool(self.cfg.terminate_on_ball_view_y_bounds):
                ball_view_y_too_low = jnp.zeros_like(ball_view_y_too_low, dtype=bool)
                ball_view_y_too_high = jnp.zeros_like(ball_view_y_too_high, dtype=bool)
            if not bool(self.cfg.terminate_on_ball_view_z_low):
                ball_view_z_too_low = jnp.zeros_like(ball_view_z_too_low, dtype=bool)
            if not bool(self.cfg.terminate_on_ball_view_z_high):
                ball_view_z_too_high = jnp.zeros_like(ball_view_z_too_high, dtype=bool)
        terms = {
            "ball_too_low": bpos[:, 2] < float(self.cfg.ball_low_termination_z_m),
            "ball_too_high": bpos[:, 2] > float(self.cfg.ball_high_termination_z_m),
            "ball_x_out_of_bounds": jnp.abs(bpos[:, 0] - racket_anchor[:, 0]) > 0.5,
            "ball_y_out_of_bounds": jnp.abs(bpos[:, 1] - racket_anchor[:, 1]) > 0.5,
            "ball_view_x_too_low": ball_view_x_too_low,
            "ball_view_x_too_high": ball_view_x_too_high,
            "ball_view_y_too_low": ball_view_y_too_low,
            "ball_view_y_too_high": ball_view_y_too_high,
            "ball_view_z_too_low": ball_view_z_too_low,
            "ball_view_z_too_high": ball_view_z_too_high,
            "base_x_out_of_bounds": jnp.abs(data.qpos[:, self.base_x_qadr]) > 2.6,
            "base_y_out_of_bounds": jnp.abs(data.qpos[:, self.base_y_qadr]) > 2.6,
            "base_z_out_of_bounds": (
                jnp.abs(data.qpos[:, self.base_z_qadr] - self.initial_base_z)
                > float(self.cfg.base_z_deviation_limit_m)
            ),
            "base_roll_out_of_bounds": (
                jnp.abs(data.qpos[:, self.base_roll_qadr] - self.initial_base_roll)
                > float(self.cfg.base_roll_pitch_limit_rad)
            ),
            "base_pitch_out_of_bounds": (
                jnp.abs(data.qpos[:, self.base_pitch_qadr] - self.initial_base_pitch)
                > float(self.cfg.base_roll_pitch_limit_rad)
            ),
            "racket_too_far_from_anchor": jnp.linalg.norm(rpos - racket_anchor, axis=-1) > 1.1,
            "racket_too_high": racket_too_high,
            "racket_too_low": racket_too_low,
        }
        if not bool(self.cfg.terminate_on_base_stability):
            for key in (
                "base_z_out_of_bounds",
                "base_roll_out_of_bounds",
                "base_pitch_out_of_bounds",
            ):
                terms[key] = jnp.zeros_like(terms[key], dtype=bool)
        terminated = jnp.zeros_like(terms["ball_too_low"], dtype=bool)
        for value in terms.values():
            terminated = terminated | value
        return terminated, terms

    def _point_to_base(self, point: jax.Array, base_q: jax.Array) -> jax.Array:
        dx = point[:, 0] - base_q[:, 0]
        dy = point[:, 1] - base_q[:, 1]
        yaw = base_q[:, 2]
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        return jnp.stack([c * dx + s * dy, -s * dx + c * dy, point[:, 2]], axis=-1)

    def _vel_to_base(self, vel: jax.Array, point: jax.Array, base_q: jax.Array, base_dq: jax.Array) -> jax.Array:
        rel_x = point[:, 0] - base_q[:, 0]
        rel_y = point[:, 1] - base_q[:, 1]
        yaw_rate = base_dq[:, 2]
        rel_vx = vel[:, 0] - base_dq[:, 0] + yaw_rate * rel_y
        rel_vy = vel[:, 1] - base_dq[:, 1] - yaw_rate * rel_x
        yaw = base_q[:, 2]
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        return jnp.stack([c * rel_vx + s * rel_vy, -s * rel_vx + c * rel_vy, vel[:, 2]], axis=-1)
