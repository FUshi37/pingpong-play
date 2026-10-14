"""MJX/JAX juggling environment mirroring ``rl_juggle_env_random.JuggleEnv``.

The environment keeps the CPU task's 50-dimensional observation layout and
right-arm acceleration-command action interface while running batched MJX steps.
Each parallel environment carries its own MJX Model pytree so domain
randomization can change model fields such as mass, contact parameters,
damping, armature, gravity, and racket mount geometry per episode.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
import sys
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
from sim2real_bridger import (
    constrained_compensation_step_jax,
    policy_relative_compensation_step_jax,
)

# ``recovered_rmp`` lives at the workspace root while this module is also
# commonly launched directly from tools/rl_sim.  Find that root without
# depending on the caller's current working directory.
_RECOVERED_RMP_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents if (parent / "recovered_rmp").is_dir()),
    None,
)
if _RECOVERED_RMP_ROOT is not None and str(_RECOVERED_RMP_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECOVERED_RMP_ROOT))

from recovered_rmp import RMPParams  # noqa: E402
from recovered_rmp.jax_backend import (  # noqa: E402
    JointRMPStateJax,
    init_joint_rmp_state_jax,
    joint_rmp_step_jax,
)


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

# Effective sport-mode low-level PD identified jointly with the q-only
# delayed second-order command response.  These are not claimed to be the
# controller's private gains; q-only replay identifies the composite plant.
SPORT_TASKSPACE_FIT_RIGHT_ARM_PD: dict[str, tuple[float, float]] = {
    "RightArm-0": (120000.0, 1250.0),
    "RightArm-1": (120000.0, 675.0),
    "RightArm-2": (101250.0, 1087.5),
    "RightArm-3": (75000.0, 843.75),
    "RightArm-4": (48750.0, 468.75),
    "RightArm-5": (56250.0, 531.25),
    "RightArm-6": (37500.0, 328.125),
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


def _apply_right_arm_pd_profile(
    xml_path: Path,
    profile: str,
    *,
    kp_scale: float = 1.0,
    kv_scale: float = 1.0,
) -> Path:
    profiles = {
        "legacy_stage4g": LEGACY_STAGE4G_RIGHT_ARM_PD,
        "comparison_safe_v1": COMPARISON_SAFE_RIGHT_ARM_PD,
        "sport_taskspace_fit_v1": SPORT_TASKSPACE_FIT_RIGHT_ARM_PD,
    }
    use_xml_values = profile in ("", "xml", "current")
    if not use_xml_values and profile not in profiles:
        raise ValueError(f"unknown right_arm_pd_profile={profile!r}")
    if kp_scale <= 0.0 or kv_scale <= 0.0:
        raise ValueError("right-arm PD scales must be positive")
    if use_xml_values and np.isclose(kp_scale, 1.0) and np.isclose(kv_scale, 1.0):
        return xml_path
    pd_values = profiles.get(profile, {})
    tree = ET.parse(xml_path)
    root = tree.getroot()
    patched = 0
    for elem in root.findall(".//actuator/position"):
        name = elem.attrib.get("name", "")
        if name not in RIGHT_ARM_JOINTS:
            continue
        if use_xml_values:
            kp = float(elem.attrib.get("kp", "0"))
            kv = float(elem.attrib.get("kv", "0"))
        else:
            kp, kv = pd_values[name]
        elem.set("kp", f"{kp * kp_scale:g}")
        elem.set("kv", f"{kv * kv_scale:g}")
        patched += 1
    if patched != len(RIGHT_ARM_JOINTS):
        raise ValueError(
            f"right_arm_pd_profile={profile!r} patched {patched} actuators, "
            f"expected {len(RIGHT_ARM_JOINTS)}"
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
    # Right-arm-only posture shaping.  ``posture_weight`` above covers every
    # configured posture joint on the robot, so a single redundant shoulder or
    # elbow excursion is diluted by the stationary left arm, waist, and head.
    # These terms are disabled by default for checkpoint compatibility.
    arm_posture_penalty_weight: float = 0.0
    arm_command_posture_penalty_weight: float = 0.0
    arm_posture_soft_limit_penalty_weight: float = 0.0
    arm_posture_joint_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    arm_posture_soft_limit_deg: tuple[float, ...] = (180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0)
    # Training-only behavior teacher.  The reference is indexed by the ball's
    # vertical flight phase and contains successful q/dq plus racket-z motion,
    # never old-domain actions.  Empty tables preserve historical behavior.
    phase_teacher_q_reference_rad: tuple[tuple[float, ...], ...] = ()
    phase_teacher_dq_reference_rad_s: tuple[tuple[float, ...], ...] = ()
    phase_teacher_racket_z_rel_reference_m: tuple[float, ...] = ()
    phase_teacher_racket_vz_reference_m_s: tuple[float, ...] = ()
    phase_teacher_ball_vz_scale_m_s: float = 1.0
    phase_teacher_strength: float = 0.0
    phase_teacher_q_weight: float = 0.060
    phase_teacher_dq_weight: float = 0.015
    phase_teacher_racket_z_weight: float = 0.020
    phase_teacher_racket_vz_weight: float = 0.005
    phase_teacher_q_sigma_deg: tuple[float, ...] = (20.0, 20.0, 25.0, 25.0, 25.0, 30.0, 30.0)
    phase_teacher_dq_sigma_rad_s: tuple[float, ...] = (2.5, 2.5, 2.5, 2.5, 2.5, 3.0, 3.0)
    phase_teacher_joint_weights: tuple[float, ...] = (1.25, 1.5, 1.0, 0.75, 1.0, 1.25, 0.75)
    phase_teacher_activate_after_hits: int = 1
    # Best-checkpoint guards are profile opt-ins.  They reject visually unsafe
    # policies without changing curriculum graduation or the physical limits.
    best_checkpoint_max_arm_posture_soft_exceed_fraction: float = 1.0
    best_checkpoint_max_arm_command_soft_exceed_fraction: float = 1.0
    best_checkpoint_max_arm_qvel_limit_exceed_fraction: float = 1.0
    best_checkpoint_max_arm_qacc_limit_exceed_fraction: float = 1.0
    base_pose_weight: float = 0.0
    # Reject policies that exploit the uncommanded base z/roll/pitch DOFs.
    terminate_on_base_stability: bool = True
    base_z_deviation_limit_m: float = 0.03
    base_roll_pitch_limit_rad: float = 0.0872664626  # 5 degrees
    torque_penalty_weight: float = 0.00005
    post_hit_survival_reward_weight: float = 1.4
    post_hit_ball_xy_sigma: float = 0.12
    post_hit_ball_vxy_penalty_weight: float = 0.18
    # Activate post-hit and event-level recoverability penalties only after
    # this numbered hit.  A value of 1 preserves the historical behaviour.
    hit_recoverability_min_count: int = 1
    descending_intercept_reward_weight: float = 1.6
    descending_intercept_sigma: float = 0.10
    # Unlike the positive Gaussian reward, this keeps a usable gradient when
    # the descending return is already outside the nominal interception tube.
    # It is disabled by default so existing checkpoints remain unchanged.
    descending_intercept_excess_penalty_weight: float = 0.0
    descending_intercept_excess_radius: float = 0.10
    descending_intercept_excess_sigma: float = 0.10
    descending_intercept_excess_time_max: float = 0.55
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
    # Bounded task-space smoothness cost.  This directly observes the racket
    # center rather than assuming small joint action deltas imply one clean
    # vertical stroke.  It remains reward-only and needs no deployment input.
    racket_vertical_acc_penalty_weight: float = 0.0
    racket_vertical_acc_scale_m_s2: float = 50.0
    # Dense paddle-normal stability.  Penalize only angular velocity that
    # changes the racket normal; free spin around the normal and translational
    # recovery remain unconstrained.  This closes the gap left by the sparse
    # hit-only angular-speed term below.
    racket_tilt_angular_speed_penalty_weight: float = 0.0
    racket_tilt_angular_speed_soft_limit_rad_s: float = 0.75
    racket_tilt_angular_speed_scale_rad_s: float = 1.25
    # Select which realized angular motion is considered undesirable.  The
    # local_xz mode preserves local-y rotation (the useful vertical juggling
    # stroke) while suppressing face roll and spin around the paddle normal.
    racket_stability_angular_speed_mode: str = "full_norm"
    racket_stability_angular_speed_penalty_weight: float = 0.0
    racket_stability_angular_speed_soft_limit_rad_s: float = 0.65
    racket_stability_angular_speed_scale_rad_s: float = 1.0
    racket_up_drift_vel_thresh: float = 0.02
    racket_flatness_penalty_weight: float = 0.0
    racket_flatness_target_cos: float = 0.970
    racket_flatness_sigma: float = 0.060
    # Event-level contact stability.  This uses the realized racket rigid-body
    # angular velocity, so vertical translation remains free while a policy
    # that rotates the face rapidly through impact is penalized.
    hit_racket_angular_speed_penalty_weight: float = 0.0
    hit_racket_angular_speed_soft_limit_rad_s: float = 1.5
    hit_racket_angular_speed_scale_rad_s: float = 1.5
    # Discourage a large active retreat immediately after a counted hit.  The
    # penalty is soft, finite-window, and anchor-relative; it is not a planner
    # or a hard task-space constraint.
    post_hit_racket_retreat_penalty_weight: float = 0.0
    post_hit_racket_retreat_window_s: float = 0.35
    post_hit_racket_retreat_deadband_m: float = 0.035
    post_hit_racket_retreat_scale_m: float = 0.050
    post_hit_racket_downward_speed_soft_limit_m_s: float = 0.15
    post_hit_racket_downward_speed_scale_m_s: float = 0.50
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
    hit_flatness_excess_penalty_weight: float = 0.0
    # Event-local non-vanishing penalty for off-centre contacts.  The Gaussian
    # centre reward above becomes almost flat for the rare large errors that
    # create unrecoverable lateral returns, so keep this disabled by default
    # and enable it only in evidence-backed repair profiles.
    hit_contact_center_excess_penalty_weight: float = 0.0
    hit_contact_center_excess_radius_m: float = 0.020
    hit_contact_center_excess_sigma_m: float = 0.030
    contact_flatness_penalty_weight: float = 0.45
    # Reward-only periodicity guard for the direct actuator policy.  It does
    # not copy a teacher trajectory or constrain the robot's instantaneous
    # joint/velocity range: only excessive hit-to-hit posture drift and a DC
    # action bias over a completed juggling cycle are penalized.
    hit_cycle_q_closure_penalty_weight: float = 0.0
    hit_cycle_q_deadband_deg: tuple[float, ...] = (8.0, 8.0, 10.0, 12.0, 12.0, 15.0, 15.0)
    hit_cycle_q_scale_deg: tuple[float, ...] = (12.0, 12.0, 15.0, 18.0, 18.0, 22.0, 22.0)
    hit_cycle_joint_weights: tuple[float, ...] = (2.0, 2.5, 2.0, 1.0, 1.5, 0.75, 0.5)
    hit_cycle_action_dc_penalty_weight: float = 0.0
    hit_cycle_action_dc_deadband: float = 0.08
    hit_cycle_action_dc_scale: float = 0.30
    hit_cycle_q_excursion_penalty_weight: float = 0.0
    hit_cycle_q_excursion_deadband_deg: tuple[float, ...] = (
        18.0, 16.0, 22.0, 24.0, 20.0, 20.0, 32.0
    )
    hit_cycle_q_excursion_scale_deg: tuple[float, ...] = (
        12.0, 12.0, 15.0, 18.0, 15.0, 15.0, 20.0
    )
    hit_cycle_min_previous_hits: int = 2
    hit_height_center: float = 0.52
    hit_height_tolerance: float = 0.06
    hit_height_penalty_weight: float = 10.0
    hit_vxy_soft_limit_m_s: float = 0.35
    hit_vxy_penalty_weight: float = 0.0
    # With a delayed position actuator, outgoing ball vxy is often caused by
    # lateral racket chase speed at impact.  Penalizing that controllable
    # precursor is disabled by default and shares the recoverability gate.
    hit_racket_vxy_soft_limit_m_s: float = 0.35
    hit_racket_vxy_penalty_weight: float = 0.0
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
    # Optional training-density change that preserves the complete original
    # DR support.  A fraction of environments samples the upper tail of the
    # two empirically difficult lag/compliance variables; the remainder keeps
    # the original uniform distribution.  Zero exactly reproduces legacy DR.
    dr_hard_tail_fraction: float = 0.0
    dr_hard_tail_lower_quantile: float = 2.0 / 3.0
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
    # Hardware-equivalent right-arm command path:
    # position target (policy/step_q_ref) -> recovered 1 kHz RMP -> q/qd/qdd
    # -> XML low-level PD.  The optional actuator dynamics/latency and command
    # compensation blocks below must remain disabled for this mode.
    recovered_rmp_motion_mode: bool = False
    recovered_rmp_period_s: float = 0.001
    # 1.0/1.0 are the recovered RightArm YAML gains (400/40).  A replay-only
    # fit on record_new3/4 favored roughly 0.30/sqrt(0.30); that is an
    # empirical sim calibration, not evidence that the hardware YAML differs.
    recovered_rmp_jnt_kp_scale: float = 1.0
    recovered_rmp_jnt_kd_scale: float = 1.0
    recovered_rmp_jnt_kp_scale_per_joint: tuple[float, ...] = ()
    recovered_rmp_jnt_kd_scale_per_joint: tuple[float, ...] = ()
    recovered_rmp_dof_damping_scale_per_joint: tuple[float, ...] = ()
    recovered_rmp_dof_armature_scale_per_joint: tuple[float, ...] = ()
    recovered_rmp_target_filter_length: int = 10
    recovered_rmp_estimator_joint_process_var: float = 50.0
    recovered_rmp_estimator_joint_measure_var: float = 0.0001
    recovered_rmp_joint_velocity_feedforward: float = 0.5
    recovered_rmp_joint_acceleration_feedforward: float = 0.0
    recovered_rmp_acceleration_weight: float = 0.001
    # Transport from the 200 Hz application command to the 1 kHz PlanTask.
    # The matched 2026-08-11 DataTracer capture identifies a three-tick delay.
    recovered_rmp_target_update_delay_steps: int = 3
    # Pure downstream transport delay between the 1 kHz RMP output and the
    # low-level PD.  This is separate from (and does not enable) the optional
    # actuator filter/model.
    recovered_rmp_output_delay_steps_per_joint: tuple[int, ...] = ()
    recovered_rmp_use_safety_bounds: bool = True
    # This RMP motion path smooths the three outputs with filter_length=35;
    # the YAML jerk limit is not applied to its raw acceleration solution.
    recovered_rmp_use_jerk_limit: bool = False
    recovered_rmp_controller_kp_scale: float = 1.0
    recovered_rmp_controller_kd_scale: float = 1.0
    recovered_rmp_controller_kp_scale_per_joint: tuple[float, ...] = ()
    recovered_rmp_controller_kd_scale_per_joint: tuple[float, ...] = ()
    recovered_rmp_jerk_limit_deg_s3: tuple[float, ...] = (
        35000.0,
        35000.0,
        45000.0,
        60000.0,
        60000.0,
        60000.0,
        60000.0,
    )
    # XML position actuators implement Kp(q_ref-q)-Kv*qdot.  Adding
    # (Kv/Kp)*RMP_qdot to ctrl makes that exactly
    # Kp(q_ref-q)+Kv(RMP_qdot-qdot).  The qdd term is a tunable low-level
    # acceleration feedforward expressed as an equivalent position time^2;
    # zero matched the available hardware logs best and is therefore default.
    recovered_rmp_acceleration_feedforward_s2: tuple[float, ...] = (0.0,) * 7
    actuator_cmd_filter: bool = False
    actuator_cmd_model: str = "first_order"  # first_order, second_order, parallel_second_order
    actuator_cmd_tau: float = 0.0
    actuator_cmd_gain: float = 1.0
    actuator_cmd_natural_frequency_rad_s: tuple[float, ...] = (21.0,) * 7
    actuator_cmd_damping_ratio: tuple[float, ...] = (0.7,) * 7
    actuator_cmd_gain_per_joint: tuple[float, ...] = (1.0,) * 7
    actuator_cmd_secondary_natural_frequency_rad_s: tuple[float, ...] = (21.0,) * 7
    actuator_cmd_secondary_damping_ratio: tuple[float, ...] = (0.7,) * 7
    actuator_cmd_secondary_mix_per_joint: tuple[float, ...] = (0.0,) * 7
    # Empty keeps the environment-level sampled delay.  A seven-element
    # tuple enables a fixed independently identified delay for each joint.
    actuator_cmd_delay_ms_per_joint: tuple[float, ...] = ()
    dr_randomize_actuator_cmd_filter: bool = False
    dr_actuator_cmd_tau_range: tuple[float, float] = (0.0, 0.0)
    dr_actuator_cmd_gain_range: tuple[float, float] = (1.0, 1.0)
    # Hidden episode-level uncertainty for the identified second-order plant.
    # These are deliberately separate from the legacy first-order tau DR.
    dr_randomize_second_order_actuator: bool = False
    dr_second_order_frequency_scale_range: tuple[float, float] = (1.0, 1.0)
    dr_second_order_damping_scale_range: tuple[float, float] = (1.0, 1.0)
    dr_second_order_gain_scale_range: tuple[float, float] = (1.0, 1.0)
    dr_second_order_delay_offset_steps_range: tuple[int, int] = (0, 0)
    actuator_compensation_mode: str = "none"
    # Causal sport-mode model-inverse network.  It consumes only policy-side
    # q history/qvel/qdd plus physical joint q/dq feedback and emits q.
    actuator_model_inverse_mlp_path: str = ""
    actuator_model_inverse_mlp_position_gain: float = 0.15714285714285708
    actuator_model_inverse_mlp_velocity_gain_s: float = 0.025714285714285717
    actuator_model_inverse_mlp_max_delta_rad: float = 0.2617993877991494
    # Causal analytic inverse for the identified sport-mode actuator.  The
    # coefficients multiply policy-side qdot/qdd/filtered jerk, followed by
    # optional measured q/dq feedback.  No future action or hidden actuator
    # state is used.
    actuator_analytic_inverse_qvel_s: tuple[float, ...] = (
        0.09808619, 0.11203451, 0.06097432, 0.05194193,
        0.08234449, 0.06634608, 0.09917843,
    )
    actuator_analytic_inverse_qdd_s2: tuple[float, ...] = (
        0.003242158, 0.003101023, 0.002421574, 0.003487333,
        0.004123453, 0.003504685, 0.004318348,
    )
    actuator_analytic_inverse_jerk_s3: tuple[float, ...] = (
        0.000206090, 0.000243647, 0.000120336, 0.000071719,
        0.000170747, 0.000098504, 0.000188652,
    )
    actuator_analytic_inverse_position_gain: tuple[float, ...] = (
        0.0, 0.0, 0.08963560, 0.0, 0.0, 0.0, 0.0,
    )
    actuator_analytic_inverse_velocity_gain_s: tuple[float, ...] = (
        0.00322155, 0.0, 0.0, 0.0, 0.01340038, 0.0, 0.0,
    )
    actuator_analytic_inverse_jerk_filter_tau_s: float = 0.020
    actuator_analytic_inverse_max_delta_rad: float = 0.5235987755982988
    # Planned-reference analytic inverse.  The strategy supplies a short
    # future q/qdd horizon; the robot still receives only the current q_send.
    actuator_horizon_inverse_lead_s: tuple[float, ...] = (
        0.11511811, 0.09952756, 0.10377953, 0.10094488,
        0.10661417, 0.10236220, 0.10803150,
    )
    actuator_horizon_inverse_accel_s2: tuple[float, ...] = (0.0011,) * 7
    actuator_horizon_inverse_max_delta_rad: float = 0.20943951023931956
    actuator_horizon_inverse_steps: int = 26
    # Causal, bandwidth-limited inverse of the identified delayed
    # second-order actuator.  Unlike ``sport_horizon_inverse`` this mode does
    # not pretend that an instantaneous 200 Hz policy acceleration remains
    # constant for the next 125 ms.  The acceleration derivative is filtered
    # before it is mapped back into a q-only command, preventing the inverse
    # from turning policy qdd chatter into an immediate position channel.
    actuator_regularized_inverse_accel_filter_tau_s: float = 0.0
    actuator_regularized_inverse_max_delta_rad: float = 0.03490658503988659
    # Conservative production base selected on a policy-integrator-reachable
    # trace.  It inverts the no-delay plant dynamics and leaves anticipation
    # to the policy/harmonic stage instead of treating current qdd as known for
    # the full 35--55 ms delay.
    actuator_regularized_inverse_preview_scale: float = 0.0
    actuator_regularized_inverse_blend: float = 0.35
    # Causal analytic residual loop layered on the regularized plant inverse.
    # Every signal is available from the policy/integrator, the published-q
    # history, and measured joint state; no signal is taken after the physical
    # actuator.  The two residual poles intentionally live well below the
    # fitted 3--4 Hz plant bandwidth.
    actuator_filtered_smith_bandwidth_hz: float = 2.0
    # Replay tuning on the final sport task-space plant selected zero gain for
    # the residual loops.  Keep the stages available for later identification,
    # but fail safe to the validated regularized inverse.
    actuator_filtered_smith_gain: float = 0.0
    actuator_dob_bandwidth_hz: float = 0.8
    actuator_dob_gain: float = 0.0
    actuator_harmonic_prediction_frequency_hz: float = 1.8
    actuator_harmonic_prediction_gain: float = 0.0
    actuator_harmonic_prediction_confidence_scale_deg_s3: float = 20000.0
    # Limit only compensation-induced racket-centre vertical acceleration.
    # This fixed warm-pose Jacobian projection is deterministic in NumPy/JAX
    # and requires no task-space measurement on the robot.
    actuator_racket_reference_governor_acc_limit_m_s2: float = 1.0e6
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
    # The q-only sport compensation is allowed a slightly wider acceleration
    # envelope than the nominal policy integrator.  It remains an absolute
    # limit on the final command, not an additional residual allowance.
    actuator_compensation_governor_natural_frequency_hz: float = 12.0
    actuator_compensation_acc_limit_scale: float = 4.0 / 3.0
    actuator_compensation_acc_limit_margin_deg_s2: float = 1000.0
    actuator_compensation_jerk_limit_deg_s3: float = 2_000_000.0
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
    # Experimental model-aware physical stack: inverse MPC -> predictive
    # pre-actuator servo governor -> delay/FOPDT -> XML position PD.  The
    # governor conjugates the legacy output planner through the fitted FOPDT
    # model instead of moving its old formula unchanged.  False preserves
    # legacy checkpoints, whose planner acts after the fitted actuator model.
    arm_servo_planner_before_actuator_model: bool = False
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
    # Penalize use of the available physical envelope without narrowing it.
    # Unlike the *_limit penalties, these are active below the hard limit and
    # therefore teach economical motion instead of imposing a slower robot.
    arm_velocity_usage_penalty_weight: float = 0.0
    arm_acceleration_usage_penalty_weight: float = 0.0
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
    # Preserve later-hit exploration credit when a rare contact is initially
    # off-centre.  Explicit contact-quality penalties can then improve that
    # contact without also suppressing the count signal.  False keeps the
    # historical multiplicative reward exactly unchanged.
    hit_combo_quality_independent: bool = False
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
    # Optional density curriculum for observation-frame calibration.  The
    # configured ranges always remain the full target support; only this
    # fraction of resets contracts deviations toward zero/identity by
    # ``easy_scale``.  Defaults exactly preserve legacy sampling.
    dr_ball_obs_frame_easy_fraction: float = 0.0
    dr_ball_obs_frame_easy_scale: float = 0.5
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
    anti_windup_directional: bool = False
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
    rmp_q: jax.Array
    rmp_qd: jax.Array
    rmp_qdd: jax.Array
    rmp_internal_q: jax.Array
    rmp_internal_qd: jax.Array
    rmp_internal_qdd: jax.Array
    rmp_accepted_target: jax.Array
    rmp_ticks_since_target_update: jax.Array
    rmp_jnt_kp: jax.Array
    rmp_jnt_kd: jax.Array
    rmp_estimator_joint_process_var: jax.Array
    rmp_estimator_joint_measure_var: jax.Array
    rmp_joint_velocity_feedforward: jax.Array
    rmp_joint_acceleration_feedforward: jax.Array
    rmp_acceleration_weight: jax.Array
    rmp_target_filter_length: jax.Array
    rmp_output_delay_steps: jax.Array
    rmp_output_q_delay_buffer: jax.Array
    rmp_output_qd_delay_buffer: jax.Array
    rmp_output_qdd_delay_buffer: jax.Array
    rmp_target_x: jax.Array
    rmp_target_covariance: jax.Array
    rmp_target_q_buffer: jax.Array
    rmp_target_qd_buffer: jax.Array
    rmp_target_qdd_buffer: jax.Array
    rmp_target_buffer_index: jax.Array
    rmp_filter_q_buffer: jax.Array
    rmp_filter_qd_buffer: jax.Array
    rmp_filter_qdd_buffer: jax.Array
    rmp_filter_buffer_index: jax.Array
    arm_q_ref_latest: jax.Array
    arm_q_ref_active: jax.Array
    arm_actuator_q_ref_latest: jax.Array
    arm_actuator_q_ref_active: jax.Array
    arm_safe_q_ref_latest: jax.Array
    arm_safe_qvel: jax.Array
    arm_safe_qacc: jax.Array
    compensation_prev_qdd: jax.Array
    compensation_filtered_qdd: jax.Array
    compensation_filtered_qdd_stage2: jax.Array
    compensation_filtered_jerk: jax.Array
    compensation_smith_residual: jax.Array
    compensation_dob_residual: jax.Array
    arm_servo_command_q: jax.Array
    arm_servo_command_qvel: jax.Array
    reset_ball_obs_missing: jax.Array
    ball_obs_missing_episode_coherent_enabled: jax.Array
    ball_obs_camera_missing_enabled: jax.Array
    ball_obs_view_bounds_missing_enabled: jax.Array
    arm_applied_q: jax.Array
    arm_applied_qvel: jax.Array
    arm_actuator_mode1_q: jax.Array
    arm_actuator_mode1_qvel: jax.Array
    arm_actuator_mode2_q: jax.Array
    arm_actuator_mode2_qvel: jax.Array
    prev_action: jax.Array
    prev_arm_qvel: jax.Array
    prev_ball_pos: jax.Array
    prev_racket_pos: jax.Array
    prev_racket_vel: jax.Array
    prev_contact: jax.Array
    hit_armed: jax.Array
    no_contact_steps: jax.Array
    contact_hold_steps: jax.Array
    pending_hit: jax.Array
    pending_hit_steps: jax.Array
    hit_count: jax.Array
    last_counted_hit_arm_q: jax.Array
    hit_cycle_arm_q_min: jax.Array
    hit_cycle_arm_q_max: jax.Array
    hit_cycle_action_sum: jax.Array
    hit_cycle_action_steps: jax.Array
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
    second_order_frequency_scale: jax.Array
    second_order_damping_scale: jax.Array
    second_order_gain_scale: jax.Array
    second_order_delay_offset_steps: jax.Array
    dr_gravity_z: jax.Array
    dr_ball_mass: jax.Array
    dr_ball_friction: jax.Array
    dr_racket_friction: jax.Array
    dr_ball_solref_time: jax.Array
    dr_ball_solref_damping: jax.Array
    dr_hard_tail_active: jax.Array
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
        self.racket_stability_angular_speed_mode = str(
            cfg.racket_stability_angular_speed_mode
        )
        if self.racket_stability_angular_speed_mode not in {"full_norm", "local_xz"}:
            raise ValueError(
                "racket_stability_angular_speed_mode must be 'full_norm' or "
                f"'local_xz', got {self.racket_stability_angular_speed_mode!r}"
            )
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

        rmp_controller_kp_scale = float(cfg.recovered_rmp_controller_kp_scale)
        rmp_controller_kd_scale = float(cfg.recovered_rmp_controller_kd_scale)
        if cfg.recovered_rmp_controller_kp_scale_per_joint:
            kp_scale_per_joint = np.asarray(
                cfg.recovered_rmp_controller_kp_scale_per_joint, dtype=np.float32
            )
            if kp_scale_per_joint.shape != (len(RIGHT_ARM_JOINTS),):
                raise ValueError(
                    "recovered_rmp_controller_kp_scale_per_joint must contain seven values"
                )
        else:
            kp_scale_per_joint = np.full(
                len(RIGHT_ARM_JOINTS), rmp_controller_kp_scale, dtype=np.float32
            )
        if cfg.recovered_rmp_controller_kd_scale_per_joint:
            kd_scale_per_joint = np.asarray(
                cfg.recovered_rmp_controller_kd_scale_per_joint, dtype=np.float32
            )
            if kd_scale_per_joint.shape != (len(RIGHT_ARM_JOINTS),):
                raise ValueError(
                    "recovered_rmp_controller_kd_scale_per_joint must contain seven values"
                )
        else:
            kd_scale_per_joint = np.full(
                len(RIGHT_ARM_JOINTS), rmp_controller_kd_scale, dtype=np.float32
            )
        if np.any(kp_scale_per_joint <= 0.0) or np.any(kd_scale_per_joint <= 0.0):
            raise ValueError("recovered RMP controller PD scales must be positive")
        patched_xml = _apply_right_arm_pd_profile(
            _build_temp_xml_with_ball(self.xml_path),
            str(cfg.right_arm_pd_profile),
            kp_scale=1.0,
            kv_scale=1.0,
        )
        if not (
            np.allclose(kp_scale_per_joint, 1.0)
            and np.allclose(kd_scale_per_joint, 1.0)
        ):
            pd_tree = ET.parse(patched_xml)
            pd_root = pd_tree.getroot()
            joint_index = {name: index for index, name in enumerate(RIGHT_ARM_JOINTS)}
            patched_count = 0
            for element in pd_root.findall(".//actuator/position"):
                name = element.attrib.get("name", "")
                if name not in joint_index:
                    continue
                index = joint_index[name]
                element.set(
                    "kp",
                    f"{float(element.attrib.get('kp', '0')) * kp_scale_per_joint[index]:g}",
                )
                element.set(
                    "kv",
                    f"{float(element.attrib.get('kv', '0')) * kd_scale_per_joint[index]:g}",
                )
                patched_count += 1
            if patched_count != len(RIGHT_ARM_JOINTS):
                raise ValueError("failed to apply seven per-joint recovered RMP PD scales")
            pd_tree.write(patched_xml, encoding="unicode")
        self.mjx_xml = _write_mjx_contact_only_xml(patched_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.mjx_xml))
        self.model = mjx.put_model(self.mj_model)

        self.timestep = float(self.mj_model.opt.timestep)
        self.dt = float(self.timestep * cfg.frame_skip)
        self.recovered_rmp_motion_mode = bool(cfg.recovered_rmp_motion_mode)
        rmp_period_s = float(cfg.recovered_rmp_period_s)
        if rmp_period_s <= 0.0:
            raise ValueError("recovered_rmp_period_s must be positive")
        rmp_steps_float = self.dt / rmp_period_s
        self.recovered_rmp_servo_steps = max(1, int(round(rmp_steps_float)))
        if self.recovered_rmp_motion_mode and not np.isclose(
            rmp_steps_float, self.recovered_rmp_servo_steps, rtol=0.0, atol=1.0e-8
        ):
            raise ValueError(
                "the MJX control period must be an integer multiple of "
                "recovered_rmp_period_s"
            )
        if int(cfg.recovered_rmp_target_filter_length) <= 0:
            raise ValueError("recovered_rmp_target_filter_length must be positive")
        rmp_kp_scale = float(cfg.recovered_rmp_jnt_kp_scale)
        rmp_kd_scale = float(cfg.recovered_rmp_jnt_kd_scale)
        if rmp_kp_scale <= 0.0 or rmp_kd_scale <= 0.0:
            raise ValueError("recovered RMP joint gain scales must be positive")
        recovered_defaults = RMPParams()
        self.recovered_rmp_params = dataclass_replace(
            recovered_defaults,
            jnt_kp=recovered_defaults.jnt_kp * rmp_kp_scale,
            jnt_kd=recovered_defaults.jnt_kd * rmp_kd_scale,
            target_filter_length=int(cfg.recovered_rmp_target_filter_length),
            estimator_joint_process_var=float(
                cfg.recovered_rmp_estimator_joint_process_var
            ),
            estimator_joint_measure_var=float(
                cfg.recovered_rmp_estimator_joint_measure_var
            ),
            joint_vel_ffd=float(cfg.recovered_rmp_joint_velocity_feedforward),
            joint_acc_ffd=float(cfg.recovered_rmp_joint_acceleration_feedforward),
            acc_weight=float(cfg.recovered_rmp_acceleration_weight),
        )
        rmp_kp_scales = np.full(self.act_dim, rmp_kp_scale, dtype=np.float32)
        rmp_kd_scales = np.full(self.act_dim, rmp_kd_scale, dtype=np.float32)
        if cfg.recovered_rmp_jnt_kp_scale_per_joint:
            rmp_kp_scales = np.asarray(
                cfg.recovered_rmp_jnt_kp_scale_per_joint, dtype=np.float32
            )
        if cfg.recovered_rmp_jnt_kd_scale_per_joint:
            rmp_kd_scales = np.asarray(
                cfg.recovered_rmp_jnt_kd_scale_per_joint, dtype=np.float32
            )
        damping_scales = np.ones(self.act_dim, dtype=np.float32)
        armature_scales = np.ones(self.act_dim, dtype=np.float32)
        if cfg.recovered_rmp_dof_damping_scale_per_joint:
            damping_scales = np.asarray(
                cfg.recovered_rmp_dof_damping_scale_per_joint, dtype=np.float32
            )
        if cfg.recovered_rmp_dof_armature_scale_per_joint:
            armature_scales = np.asarray(
                cfg.recovered_rmp_dof_armature_scale_per_joint, dtype=np.float32
            )
        for name, values in (
            ("RMP Kp", rmp_kp_scales),
            ("RMP Kd", rmp_kd_scales),
            ("dof damping", damping_scales),
            ("dof armature", armature_scales),
        ):
            if values.shape != (self.act_dim,) or np.any(values <= 0.0):
                raise ValueError(f"{name} per-joint scales must contain seven positives")
        self.recovered_rmp_jnt_kp = jnp.asarray(
            recovered_defaults.jnt_kp * rmp_kp_scales
        )
        self.recovered_rmp_jnt_kd = jnp.asarray(
            recovered_defaults.jnt_kd * rmp_kd_scales
        )
        self.recovered_rmp_dof_damping_scale = jnp.asarray(damping_scales)
        self.recovered_rmp_dof_armature_scale = jnp.asarray(armature_scales)
        self.recovered_rmp_period_s = rmp_period_s
        target_update_delay_steps = int(
            cfg.recovered_rmp_target_update_delay_steps
        )
        if not 0 <= target_update_delay_steps < self.recovered_rmp_servo_steps:
            raise ValueError(
                "recovered_rmp_target_update_delay_steps must be in "
                "[0, recovered_rmp_servo_steps)"
            )
        self.recovered_rmp_target_update_delay_steps = (
            target_update_delay_steps
        )
        output_delay_steps = np.zeros(self.act_dim, dtype=np.int32)
        if cfg.recovered_rmp_output_delay_steps_per_joint:
            output_delay_steps = np.asarray(
                cfg.recovered_rmp_output_delay_steps_per_joint, dtype=np.int32
            )
        if output_delay_steps.shape != (self.act_dim,) or np.any(
            output_delay_steps < 0
        ):
            raise ValueError(
                "recovered_rmp_output_delay_steps_per_joint must contain "
                "seven nonnegative integers"
            )
        self.recovered_rmp_output_delay_steps = jnp.asarray(output_delay_steps)
        self.recovered_rmp_output_delay_buffer_length = max(
            1, int(np.max(output_delay_steps))
        )
        if self.recovered_rmp_motion_mode:
            if not np.isclose(
                self.timestep,
                self.recovered_rmp_period_s,
                rtol=0.0,
                atol=1.0e-10,
            ) or self.recovered_rmp_servo_steps != int(cfg.frame_skip):
                raise ValueError(
                    "recovered_rmp_motion_mode requires one RMP update per MJX "
                    "substep (model timestep == recovered_rmp_period_s)"
                )
            comp_mode = (
                str(cfg.actuator_compensation_mode or "none")
                .strip()
                .lower()
                .replace("-", "_")
            )
            conflicts = {
                "actuator_cmd_filter": bool(cfg.actuator_cmd_filter),
                "actuator_compensation_mode": comp_mode not in {"none", "off", "false", "0"},
                "actuator_lead_compensation": bool(cfg.actuator_lead_compensation),
                "arm_post_compensation_limiter": bool(cfg.arm_post_compensation_limiter),
                "arm_servo_target_limiter": bool(cfg.arm_servo_target_limiter),
                "arm_servo_target_tracking_planner": bool(cfg.arm_servo_target_tracking_planner),
                "arm_actual_state_limiter": bool(cfg.arm_actual_state_limiter),
                "arm_actual_target_tracking_governor": bool(cfg.arm_actual_target_tracking_governor),
                "plant_command_delay": bool(
                    cfg.enable_delay_conditioning
                    and not cfg.actuator_delay_observation_only
                ),
            }
            enabled_conflicts = [name for name, enabled in conflicts.items() if enabled]
            if enabled_conflicts:
                raise ValueError(
                    "recovered_rmp_motion_mode is the no-actuator-model path; "
                    "disable conflicting blocks: " + ", ".join(enabled_conflicts)
                )
        actuator_cmd_model = str(cfg.actuator_cmd_model).strip().lower().replace("-", "_")
        if actuator_cmd_model not in {"first_order", "second_order", "parallel_second_order"}:
            raise ValueError(
                "actuator_cmd_model must be 'first_order', 'second_order', or "
                "'parallel_second_order', "
                f"got {cfg.actuator_cmd_model!r}"
            )
        self.actuator_cmd_model = actuator_cmd_model
        second_order_wn = np.asarray(
            tuple(cfg.actuator_cmd_natural_frequency_rad_s), dtype=np.float32
        )
        second_order_zeta = np.asarray(
            tuple(cfg.actuator_cmd_damping_ratio), dtype=np.float32
        )
        second_order_gain = np.asarray(
            tuple(cfg.actuator_cmd_gain_per_joint), dtype=np.float32
        )
        secondary_wn = np.asarray(
            tuple(cfg.actuator_cmd_secondary_natural_frequency_rad_s), dtype=np.float32
        )
        secondary_zeta = np.asarray(
            tuple(cfg.actuator_cmd_secondary_damping_ratio), dtype=np.float32
        )
        secondary_mix = np.asarray(
            tuple(cfg.actuator_cmd_secondary_mix_per_joint), dtype=np.float32
        )
        for name, values in (
            ("actuator_cmd_natural_frequency_rad_s", second_order_wn),
            ("actuator_cmd_damping_ratio", second_order_zeta),
            ("actuator_cmd_gain_per_joint", second_order_gain),
            ("actuator_cmd_secondary_natural_frequency_rad_s", secondary_wn),
            ("actuator_cmd_secondary_damping_ratio", secondary_zeta),
            ("actuator_cmd_secondary_mix_per_joint", secondary_mix),
        ):
            if values.shape != (len(RIGHT_ARM_JOINTS),):
                raise ValueError(
                    f"{name} must contain {len(RIGHT_ARM_JOINTS)} values, got {values.shape}"
                )
        if np.any(second_order_wn <= 0.0):
            raise ValueError("actuator_cmd_natural_frequency_rad_s must be positive")
        if np.any(second_order_zeta <= 0.0) or np.any(second_order_zeta >= 1.0):
            raise ValueError("actuator_cmd_damping_ratio must be in (0, 1)")
        if np.any(secondary_wn <= 0.0):
            raise ValueError("actuator_cmd_secondary_natural_frequency_rad_s must be positive")
        if np.any(secondary_zeta <= 0.0) or np.any(secondary_zeta >= 1.0):
            raise ValueError("actuator_cmd_secondary_damping_ratio must be in (0, 1)")
        if np.any(secondary_mix < 0.0) or np.any(secondary_mix > 1.0):
            raise ValueError("actuator_cmd_secondary_mix_per_joint must be in [0, 1]")
        self.actuator_cmd_second_order_wn = jnp.asarray(second_order_wn)
        self.actuator_cmd_second_order_zeta = jnp.asarray(second_order_zeta)
        self.actuator_cmd_second_order_gain = jnp.asarray(second_order_gain)
        self.actuator_cmd_secondary_wn = jnp.asarray(secondary_wn)
        self.actuator_cmd_secondary_zeta = jnp.asarray(secondary_zeta)
        self.actuator_cmd_secondary_mix = jnp.asarray(secondary_mix)
        per_joint_delay_ms = np.asarray(
            tuple(cfg.actuator_cmd_delay_ms_per_joint), dtype=np.float32
        )
        if per_joint_delay_ms.size not in (0, len(RIGHT_ARM_JOINTS)):
            raise ValueError(
                "actuator_cmd_delay_ms_per_joint must be empty or contain "
                f"{len(RIGHT_ARM_JOINTS)} values, got {per_joint_delay_ms.shape}"
            )
        if np.any(per_joint_delay_ms < 0.0):
            raise ValueError("actuator_cmd_delay_ms_per_joint must be non-negative")
        self.actuator_cmd_delay_steps_per_joint = (
            jnp.asarray(np.rint(per_joint_delay_ms * 1e-3 / self.dt), dtype=jnp.int32)
            if per_joint_delay_ms.size
            else None
        )
        self.max_steps = max(1, int(cfg.horizon_sec / self.dt))
        buffer_delay_ms = max(0.0, float(cfg.delay_max_ms))
        if per_joint_delay_ms.size:
            buffer_delay_ms = max(buffer_delay_ms, float(np.max(per_joint_delay_ms)))
        self.max_command_delay_steps = max(
            0,
            int(round(buffer_delay_ms * 1e-3 / max(self.dt, 1e-9))),
        )
        compensation_mode_early = str(
            cfg.actuator_compensation_mode or "none"
        ).strip().lower().replace("-", "_")
        self.model_inverse_mlp_history = 0
        if compensation_mode_early in {
            "sport_model_inverse_mlp",
            "model_inverse_mlp",
            "causal_model_inverse_mlp",
        }:
            model_path = Path(str(cfg.actuator_model_inverse_mlp_path)).expanduser()
            if not model_path.is_file():
                raise ValueError(
                    "actuator_model_inverse_mlp_path must point to a fitted model.npz, "
                    f"got {model_path}"
                )
            fitted_inverse = np.load(model_path)
            input_dim = int(fitted_inverse["x_mean"].shape[0])
            self.model_inverse_mlp_history = (input_dim - 14) // self.act_dim + 1
            if input_dim != (self.model_inverse_mlp_history - 1) * self.act_dim + 14:
                raise ValueError("invalid causal model-inverse MLP input dimension")
            self.model_inverse_mlp_x_mean = jnp.asarray(fitted_inverse["x_mean"], dtype=jnp.float32)
            self.model_inverse_mlp_x_std = jnp.asarray(fitted_inverse["x_std"], dtype=jnp.float32)
            self.model_inverse_mlp_y_scale = jnp.asarray(fitted_inverse["y_scale"], dtype=jnp.float32)
            self.model_inverse_mlp_weights = tuple(
                jnp.asarray(fitted_inverse[f"layers__{layer}__weight"], dtype=jnp.float32)
                for layer in (0, 2, 4)
            )
            self.model_inverse_mlp_biases = tuple(
                jnp.asarray(fitted_inverse[f"layers__{layer}__bias"], dtype=jnp.float32)
                for layer in (0, 2, 4)
            )
        self.command_buffer_len = max(
            1,
            self.max_command_delay_steps + max(0, int(cfg.command_buffer_extra_steps)) + 1,
            self.model_inverse_mlp_history,
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
        # Retain host copies for MuJoCo/NumPy initialization.  Converting the
        # device index array back to NumPy later can request pinned host memory
        # during concurrent GPU training and fail under host-memory pressure.
        self.arm_qadr_np = np.asarray(
            [int(self.mj_model.jnt_qposadr[j]) for j in self.arm_jids],
            dtype=np.int32,
        )
        self.arm_vadr_np = np.asarray(
            [int(self.mj_model.jnt_dofadr[j]) for j in self.arm_jids],
            dtype=np.int32,
        )
        self.arm_qadr = jnp.asarray(self.arm_qadr_np)
        self.arm_vadr = jnp.asarray(self.arm_vadr_np)
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
        if len(tuple(cfg.arm_posture_joint_weights)) != self.act_dim:
            raise ValueError(
                "arm_posture_joint_weights must contain "
                f"{self.act_dim} values, got {len(tuple(cfg.arm_posture_joint_weights))}"
            )
        if len(tuple(cfg.arm_posture_soft_limit_deg)) != self.act_dim:
            raise ValueError(
                "arm_posture_soft_limit_deg must contain "
                f"{self.act_dim} values, got {len(tuple(cfg.arm_posture_soft_limit_deg))}"
            )
        self.arm_posture_joint_weights = jnp.asarray(
            cfg.arm_posture_joint_weights, dtype=jnp.float32
        )
        if np.any(np.asarray(cfg.arm_posture_joint_weights, dtype=np.float32) < 0.0):
            raise ValueError("arm_posture_joint_weights must be non-negative")
        if not np.any(np.asarray(cfg.arm_posture_joint_weights, dtype=np.float32) > 0.0):
            raise ValueError("arm_posture_joint_weights must contain a positive value")
        self.arm_posture_soft_limit_rad = jnp.deg2rad(
            jnp.asarray(cfg.arm_posture_soft_limit_deg, dtype=jnp.float32)
        )
        cycle_vector_fields = (
            ("hit_cycle_q_deadband_deg", cfg.hit_cycle_q_deadband_deg),
            ("hit_cycle_q_scale_deg", cfg.hit_cycle_q_scale_deg),
            ("hit_cycle_joint_weights", cfg.hit_cycle_joint_weights),
            (
                "hit_cycle_q_excursion_deadband_deg",
                cfg.hit_cycle_q_excursion_deadband_deg,
            ),
            (
                "hit_cycle_q_excursion_scale_deg",
                cfg.hit_cycle_q_excursion_scale_deg,
            ),
        )
        for name, values in cycle_vector_fields:
            if len(tuple(values)) != self.act_dim:
                raise ValueError(
                    f"{name} must contain {self.act_dim} values, got {len(tuple(values))}"
                )
        if np.any(np.asarray(cfg.hit_cycle_q_deadband_deg, dtype=np.float32) < 0.0):
            raise ValueError("hit_cycle_q_deadband_deg must be non-negative")
        if np.any(np.asarray(cfg.hit_cycle_q_scale_deg, dtype=np.float32) <= 0.0):
            raise ValueError("hit_cycle_q_scale_deg must be positive")
        if np.any(
            np.asarray(cfg.hit_cycle_q_excursion_deadband_deg, dtype=np.float32)
            < 0.0
        ):
            raise ValueError("hit_cycle_q_excursion_deadband_deg must be non-negative")
        if np.any(
            np.asarray(cfg.hit_cycle_q_excursion_scale_deg, dtype=np.float32)
            <= 0.0
        ):
            raise ValueError("hit_cycle_q_excursion_scale_deg must be positive")
        if np.any(np.asarray(cfg.hit_cycle_joint_weights, dtype=np.float32) < 0.0):
            raise ValueError("hit_cycle_joint_weights must be non-negative")
        if not np.any(np.asarray(cfg.hit_cycle_joint_weights, dtype=np.float32) > 0.0):
            raise ValueError("hit_cycle_joint_weights must contain a positive value")
        if float(cfg.hit_cycle_action_dc_scale) <= 0.0:
            raise ValueError("hit_cycle_action_dc_scale must be positive")
        self.hit_cycle_q_deadband_rad = jnp.deg2rad(
            jnp.asarray(cfg.hit_cycle_q_deadband_deg, dtype=jnp.float32)
        )
        self.hit_cycle_q_scale_rad = jnp.deg2rad(
            jnp.asarray(cfg.hit_cycle_q_scale_deg, dtype=jnp.float32)
        )
        self.hit_cycle_q_excursion_deadband_rad = jnp.deg2rad(
            jnp.asarray(cfg.hit_cycle_q_excursion_deadband_deg, dtype=jnp.float32)
        )
        self.hit_cycle_q_excursion_scale_rad = jnp.deg2rad(
            jnp.asarray(cfg.hit_cycle_q_excursion_scale_deg, dtype=jnp.float32)
        )
        self.hit_cycle_joint_weights = jnp.asarray(
            cfg.hit_cycle_joint_weights, dtype=jnp.float32
        )
        if np.any(np.asarray(cfg.arm_posture_soft_limit_deg, dtype=np.float32) <= 0.0):
            raise ValueError("arm_posture_soft_limit_deg must be positive")
        phase_q = np.asarray(cfg.phase_teacher_q_reference_rad, dtype=np.float32)
        phase_dq = np.asarray(cfg.phase_teacher_dq_reference_rad_s, dtype=np.float32)
        phase_racket_z = np.asarray(
            cfg.phase_teacher_racket_z_rel_reference_m, dtype=np.float32
        )
        phase_racket_vz = np.asarray(
            cfg.phase_teacher_racket_vz_reference_m_s, dtype=np.float32
        )
        self.phase_teacher_enabled = bool(
            float(cfg.phase_teacher_strength) > 0.0 and phase_q.size > 0
        )
        if self.phase_teacher_enabled:
            if phase_q.ndim != 2 or phase_q.shape[1] != self.act_dim or phase_q.shape[0] < 2:
                raise ValueError("phase teacher q reference must have shape [bins, act_dim]")
            if phase_dq.shape != phase_q.shape:
                raise ValueError("phase teacher dq reference must match q reference")
            if phase_racket_z.shape != (phase_q.shape[0],) or phase_racket_vz.shape != (phase_q.shape[0],):
                raise ValueError("phase teacher racket references must have shape [bins]")
        else:
            phase_q = np.zeros((2, self.act_dim), dtype=np.float32)
            phase_dq = np.zeros((2, self.act_dim), dtype=np.float32)
            phase_racket_z = np.zeros((2,), dtype=np.float32)
            phase_racket_vz = np.zeros((2,), dtype=np.float32)
        for name, values in (
            ("phase_teacher_q_sigma_deg", cfg.phase_teacher_q_sigma_deg),
            ("phase_teacher_dq_sigma_rad_s", cfg.phase_teacher_dq_sigma_rad_s),
            ("phase_teacher_joint_weights", cfg.phase_teacher_joint_weights),
        ):
            if len(tuple(values)) != self.act_dim:
                raise ValueError(f"{name} must contain {self.act_dim} values")
        self.phase_teacher_q = jnp.asarray(phase_q)
        self.phase_teacher_dq = jnp.asarray(phase_dq)
        self.phase_teacher_racket_z = jnp.asarray(phase_racket_z)
        self.phase_teacher_racket_vz = jnp.asarray(phase_racket_vz)
        self.phase_teacher_q_sigma = jnp.deg2rad(
            jnp.asarray(cfg.phase_teacher_q_sigma_deg, dtype=jnp.float32)
        )
        self.phase_teacher_dq_sigma = jnp.asarray(
            cfg.phase_teacher_dq_sigma_rad_s, dtype=jnp.float32
        )
        self.phase_teacher_joint_weights = jnp.asarray(
            cfg.phase_teacher_joint_weights, dtype=jnp.float32
        )
        if np.any(np.asarray(cfg.phase_teacher_q_sigma_deg) <= 0.0):
            raise ValueError("phase teacher q sigma must be positive")
        if np.any(np.asarray(cfg.phase_teacher_dq_sigma_rad_s) <= 0.0):
            raise ValueError("phase teacher dq sigma must be positive")
        if float(cfg.phase_teacher_ball_vz_scale_m_s) <= 0.0:
            raise ValueError("phase teacher ball vz scale must be positive")
        self.racket_anchor = jnp.asarray(warm.site_xpos[self.racket_site_id], dtype=jnp.float32)
        warm_site_jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.mj_model,
            warm,
            warm_site_jacp,
            None,
            self.racket_site_id,
        )
        self.racket_vertical_arm_jacobian = jnp.asarray(
            warm_site_jacp[2, self.arm_vadr_np],
            dtype=jnp.float32,
        )
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
        recovered_rmp_jerk = np.asarray(
            tuple(cfg.recovered_rmp_jerk_limit_deg_s3), dtype=np.float32
        )
        recovered_rmp_acceleration_ff = np.asarray(
            tuple(cfg.recovered_rmp_acceleration_feedforward_s2), dtype=np.float32
        )
        if recovered_rmp_jerk.shape != (self.act_dim,):
            raise ValueError("recovered_rmp_jerk_limit_deg_s3 must contain seven values")
        if recovered_rmp_acceleration_ff.shape != (self.act_dim,):
            raise ValueError(
                "recovered_rmp_acceleration_feedforward_s2 must contain seven values"
            )
        if np.any(recovered_rmp_jerk <= 0.0):
            raise ValueError("recovered RMP jerk limits must be positive")
        if not np.all(np.isfinite(recovered_rmp_acceleration_ff)):
            raise ValueError("recovered RMP acceleration feedforward must be finite")
        self.recovered_rmp_jerk_limit_rad_s3 = jnp.deg2rad(
            jnp.asarray(recovered_rmp_jerk, dtype=jnp.float32)
        )
        self.recovered_rmp_acceleration_feedforward_s2 = jnp.asarray(
            recovered_rmp_acceleration_ff, dtype=jnp.float32
        )
        self.arm_actual_jerk_limit_rad_s3 = jnp.deg2rad(
            jnp.asarray(cfg.arm_actual_jerk_limit_deg_s3, dtype=jnp.float32)
        )
        self.arm_bridger_jerk_limit_rad_s3 = jnp.deg2rad(
            jnp.asarray(cfg.actuator_bridger_jerk_limit_deg_s3, dtype=jnp.float32)
        )
        compensation_mode = str(cfg.actuator_compensation_mode or "none").strip().lower().replace("-", "_")
        if compensation_mode in {
            "sport_bandlimited_horizon_inverse",
            "bandlimited_horizon_inverse",
            "sport_persistent_analytic_inverse",
            "persistent_analytic_inverse",
            "sport_persistent_analytic_smith",
            "sport_persistent_analytic_smith_dob",
            "sport_persistent_analytic_smith_dob_harmonic",
            "sport_persistent_analytic_full",
        }:
            if (
                not np.isfinite(
                    float(cfg.actuator_compensation_governor_natural_frequency_hz)
                )
                or float(cfg.actuator_compensation_governor_natural_frequency_hz)
                <= 0.0
            ):
                raise ValueError(
                    "actuator_compensation_governor_natural_frequency_hz "
                    "must be positive"
                )
            if (
                not np.isfinite(float(cfg.actuator_compensation_acc_limit_scale))
                or float(cfg.actuator_compensation_acc_limit_scale) <= 0.0
            ):
                raise ValueError(
                    "actuator_compensation_acc_limit_scale must be positive"
                )
            if (
                not np.isfinite(float(cfg.actuator_compensation_jerk_limit_deg_s3))
                or float(cfg.actuator_compensation_jerk_limit_deg_s3) <= 0.0
            ):
                raise ValueError(
                    "actuator_compensation_jerk_limit_deg_s3 must be positive"
                )
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
        dof_damping = dof_damping.at[:, self.arm_vadr].multiply(
            self.recovered_rmp_dof_damping_scale[None, :]
        )
        dof_armature = dof_armature.at[:, self.arm_vadr].multiply(
            self.recovered_rmp_dof_armature_scale[None, :]
        )
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
        hard_tail_fraction = float(self.cfg.dr_hard_tail_fraction)
        hard_tail_lower_quantile = float(self.cfg.dr_hard_tail_lower_quantile)
        if not 0.0 <= hard_tail_fraction <= 1.0:
            raise ValueError("dr_hard_tail_fraction must be in [0, 1]")
        if not 0.0 <= hard_tail_lower_quantile < 1.0:
            raise ValueError("dr_hard_tail_lower_quantile must be in [0, 1)")
        hard_tail_active = (
            jax.vmap(
                lambda k: jax.random.bernoulli(
                    jax.random.fold_in(k, 97),
                    p=hard_tail_fraction,
                )
            )(key_solref)
            if bool(self.cfg.domain_randomization and hard_tail_fraction > 0.0)
            else jnp.zeros((n_envs,), dtype=bool)
        )
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
            solref_time_lo, solref_time_hi = sorted(
                float(v) for v in self.cfg.dr_ball_solref_time_range
            )
            hard_solref_time_lo = solref_time_lo + hard_tail_lower_quantile * (
                solref_time_hi - solref_time_lo
            )
            hard_solref_time = jax.vmap(
                lambda k: jax.random.uniform(
                    jax.random.fold_in(k, 98),
                    (),
                    minval=hard_solref_time_lo,
                    maxval=max(hard_solref_time_lo + 1e-9, solref_time_hi),
                )
            )(key_solref)
            dr_ball_solref_time = jnp.where(
                hard_tail_active,
                hard_solref_time,
                dr_ball_solref_time,
            )
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
                hard_tau_lo = min(tau_low, tau_high) + hard_tail_lower_quantile * abs(
                    tau_high - tau_low
                )
                hard_actuator_cmd_tau = jax.vmap(
                    lambda k: jax.random.uniform(
                        jax.random.fold_in(k, 99),
                        (),
                        minval=hard_tau_lo,
                        maxval=max(hard_tau_lo + 1e-9, max(tau_low, tau_high)),
                    )
                )(key_actuator_tau)
                actuator_cmd_tau = jnp.where(
                    hard_tail_active,
                    hard_actuator_cmd_tau,
                    actuator_cmd_tau,
                )
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

        if bool(
            self.cfg.actuator_cmd_filter
            and self.actuator_cmd_model in {"second_order", "parallel_second_order"}
            and self.cfg.domain_randomization
            and self.cfg.dr_randomize_second_order_actuator
        ):
            def _sample_scale(keys, value_range, fold):
                low, high = sorted(float(v) for v in value_range)
                return jax.vmap(
                    lambda k: jax.random.uniform(
                        jax.random.fold_in(k, fold), (), minval=low, maxval=high
                    )
                )(keys)

            second_order_frequency_scale = _sample_scale(
                key_actuator_tau, self.cfg.dr_second_order_frequency_scale_range, 201
            )
            second_order_damping_scale = _sample_scale(
                key_actuator_tau, self.cfg.dr_second_order_damping_scale_range, 202
            )
            second_order_gain_scale = _sample_scale(
                key_actuator_gain, self.cfg.dr_second_order_gain_scale_range, 203
            )
            delay_low, delay_high = sorted(
                int(v) for v in self.cfg.dr_second_order_delay_offset_steps_range
            )
            second_order_delay_offset_steps = jax.vmap(
                lambda k: jax.random.randint(
                    jax.random.fold_in(k, 204),
                    (),
                    minval=delay_low,
                    maxval=max(delay_low + 1, delay_high + 1),
                    dtype=jnp.int32,
                )
            )(key_actuator_tau)
        else:
            second_order_frequency_scale = jnp.ones((n_envs,), dtype=jnp.float32)
            second_order_damping_scale = jnp.ones((n_envs,), dtype=jnp.float32)
            second_order_gain_scale = jnp.ones((n_envs,), dtype=jnp.float32)
            second_order_delay_offset_steps = jnp.zeros((n_envs,), dtype=jnp.int32)

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
            full_pos_bias = jax.vmap(
                lambda k: jax.random.uniform(k, (3,), minval=-pos_bias_lim, maxval=pos_bias_lim)
            )(key_ball_obs_pos_bias)
            full_rot_bias = jax.vmap(
                lambda k: jax.random.uniform(k, (3,), minval=-rot_bias_lim, maxval=rot_bias_lim)
            )(key_ball_obs_rot_bias)
            full_vel_bias = jax.vmap(
                lambda k: jax.random.uniform(k, (3,), minval=-vel_bias_lim, maxval=vel_bias_lim)
            )(key_ball_obs_vel_bias)
            full_scale = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=min(scale_low, scale_high),
                    maxval=max(scale_low, scale_high),
                )
            )(key_ball_obs_scale)
            easy_fraction = float(self.cfg.dr_ball_obs_frame_easy_fraction)
            easy_scale = float(self.cfg.dr_ball_obs_frame_easy_scale)
            if not 0.0 <= easy_fraction <= 1.0:
                raise ValueError("dr_ball_obs_frame_easy_fraction must be in [0, 1]")
            if not 0.0 <= easy_scale <= 1.0:
                raise ValueError("dr_ball_obs_frame_easy_scale must be in [0, 1]")
            if easy_fraction > 0.0:
                easy_active = jax.vmap(
                    lambda k: jax.random.bernoulli(
                        jax.random.fold_in(k, 101),
                        p=easy_fraction,
                    )
                )(key_ball_obs_pos_bias)
                scale3 = jnp.where(easy_active, easy_scale, 1.0)[:, None]
                scale1 = jnp.where(easy_active, easy_scale, 1.0)
                full_pos_bias = full_pos_bias * scale3
                full_rot_bias = full_rot_bias * scale3
                full_vel_bias = full_vel_bias * scale3
                full_scale = 1.0 + (full_scale - 1.0) * scale1
            ball_obs_pos_bias_base = nominal_pos_bias[None, :] + full_pos_bias
            ball_obs_rot_bias_rpy = full_rot_bias
            ball_obs_vel_bias_base = nominal_vel_bias[None, :] + full_vel_bias
            ball_obs_scale = full_scale
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
        initial_arm_q = jnp.broadcast_to(
            self.warm_arm_q, (n_envs, self.act_dim)
        )
        initial_rmp_state = init_joint_rmp_state_jax(
            initial_arm_q,
            velocity=jnp.zeros_like(initial_arm_q),
            acceleration=jnp.zeros_like(initial_arm_q),
            target_position=initial_arm_q,
            target_filter_length=self.recovered_rmp_params.target_filter_length,
            output_filter_length=self.recovered_rmp_params.filter_length,
        )
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
            arm_cmd_q=initial_arm_q,
            arm_cmd_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            rmp_q=initial_rmp_state.command_q,
            rmp_qd=initial_rmp_state.command_qd,
            rmp_qdd=initial_rmp_state.command_qdd,
            rmp_internal_q=initial_rmp_state.q,
            rmp_internal_qd=initial_rmp_state.qd,
            rmp_internal_qdd=initial_rmp_state.qdd,
            rmp_accepted_target=initial_rmp_state.accepted_target,
            rmp_ticks_since_target_update=(
                initial_rmp_state.ticks_since_target_update
            ),
            rmp_jnt_kp=jnp.broadcast_to(
                self.recovered_rmp_jnt_kp, initial_rmp_state.q.shape
            ),
            rmp_jnt_kd=jnp.broadcast_to(
                self.recovered_rmp_jnt_kd, initial_rmp_state.q.shape
            ),
            rmp_estimator_joint_process_var=jnp.full(
                initial_rmp_state.q.shape,
                self.recovered_rmp_params.estimator_joint_process_var,
                dtype=jnp.float32,
            ),
            rmp_estimator_joint_measure_var=jnp.full(
                initial_rmp_state.q.shape,
                self.recovered_rmp_params.estimator_joint_measure_var,
                dtype=jnp.float32,
            ),
            rmp_joint_velocity_feedforward=jnp.full(
                initial_rmp_state.q.shape,
                self.recovered_rmp_params.joint_vel_ffd,
                dtype=jnp.float32,
            ),
            rmp_joint_acceleration_feedforward=jnp.full(
                initial_rmp_state.q.shape,
                self.recovered_rmp_params.joint_acc_ffd,
                dtype=jnp.float32,
            ),
            rmp_acceleration_weight=jnp.full(
                initial_rmp_state.q.shape,
                self.recovered_rmp_params.acc_weight,
                dtype=jnp.float32,
            ),
            rmp_target_filter_length=jnp.full(
                initial_rmp_state.q.shape,
                self.recovered_rmp_params.target_filter_length,
                dtype=jnp.int32,
            ),
            rmp_output_delay_steps=jnp.broadcast_to(
                self.recovered_rmp_output_delay_steps,
                initial_rmp_state.q.shape,
            ),
            rmp_output_q_delay_buffer=jnp.broadcast_to(
                initial_rmp_state.q[:, None, :],
                (
                    n_envs,
                    self.recovered_rmp_output_delay_buffer_length,
                    self.act_dim,
                ),
            ),
            rmp_output_qd_delay_buffer=jnp.broadcast_to(
                initial_rmp_state.qd[:, None, :],
                (
                    n_envs,
                    self.recovered_rmp_output_delay_buffer_length,
                    self.act_dim,
                ),
            ),
            rmp_output_qdd_delay_buffer=jnp.broadcast_to(
                initial_rmp_state.qdd[:, None, :],
                (
                    n_envs,
                    self.recovered_rmp_output_delay_buffer_length,
                    self.act_dim,
                ),
            ),
            rmp_target_x=initial_rmp_state.target_x,
            rmp_target_covariance=initial_rmp_state.target_covariance,
            rmp_target_q_buffer=initial_rmp_state.target_q_buffer,
            rmp_target_qd_buffer=initial_rmp_state.target_qd_buffer,
            rmp_target_qdd_buffer=initial_rmp_state.target_qdd_buffer,
            rmp_target_buffer_index=initial_rmp_state.target_buffer_index,
            rmp_filter_q_buffer=initial_rmp_state.output_q_buffer,
            rmp_filter_qd_buffer=initial_rmp_state.output_qd_buffer,
            rmp_filter_qdd_buffer=initial_rmp_state.output_qdd_buffer,
            rmp_filter_buffer_index=initial_rmp_state.output_buffer_index,
            arm_q_ref_latest=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_q_ref_active=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_actuator_q_ref_latest=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_actuator_q_ref_active=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_safe_q_ref_latest=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_safe_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            arm_safe_qacc=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            compensation_prev_qdd=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            compensation_filtered_qdd=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            compensation_filtered_qdd_stage2=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            compensation_filtered_jerk=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            compensation_smith_residual=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            compensation_dob_residual=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            arm_servo_command_q=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_servo_command_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            ball_obs_missing_episode_coherent_enabled=coherent_missing_enabled,
            ball_obs_camera_missing_enabled=camera_missing_enabled,
            ball_obs_view_bounds_missing_enabled=view_bounds_missing_enabled,
            arm_applied_q=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_applied_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            arm_actuator_mode1_q=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_actuator_mode1_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            arm_actuator_mode2_q=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_actuator_mode2_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            prev_action=zero_action,
            prev_arm_qvel=jnp.broadcast_to(self.warm_arm_qvel, (n_envs, self.act_dim)),
            prev_ball_pos=bpos,
            prev_racket_pos=rpos,
            prev_racket_vel=jnp.zeros_like(rpos),
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
            last_counted_hit_arm_q=jnp.broadcast_to(
                self.warm_arm_q, (n_envs, self.act_dim)
            ),
            hit_cycle_arm_q_min=jnp.broadcast_to(
                self.warm_arm_q, (n_envs, self.act_dim)
            ),
            hit_cycle_arm_q_max=jnp.broadcast_to(
                self.warm_arm_q, (n_envs, self.act_dim)
            ),
            hit_cycle_action_sum=jnp.zeros(
                (n_envs, self.act_dim), dtype=jnp.float32
            ),
            hit_cycle_action_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
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
            second_order_frequency_scale=second_order_frequency_scale,
            second_order_damping_scale=second_order_damping_scale,
            second_order_gain_scale=second_order_gain_scale,
            second_order_delay_offset_steps=second_order_delay_offset_steps,
            dr_gravity_z=dr_gravity_z,
            dr_ball_mass=dr_ball_mass,
            dr_ball_friction=dr_ball_friction,
            dr_racket_friction=dr_racket_friction,
            dr_ball_solref_time=dr_ball_solref_time,
            dr_ball_solref_damping=dr_ball_solref_damping,
            dr_hard_tail_active=hard_tail_active,
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

    def reset_arm_replay_state(
        self,
        state: EnvState,
        measured_q: jax.Array,
        *,
        measured_qd: jax.Array | None = None,
        measured_qdd: jax.Array | None = None,
        position_target: jax.Array | None = None,
    ) -> EnvState:
        """Anchor physical and recovered-RMP arm state to a recorded sample.

        This is intended for ``record_new3/4`` replay.  It initializes the
        planner from measured q/qd/qdd while separately filling its target
        estimator from the position-only command, matching RMP motion mode.
        """

        q = jnp.asarray(measured_q, dtype=jnp.float32)
        expected_shape = (state.arm_cmd_q.shape[0], self.act_dim)
        if q.shape != expected_shape:
            raise ValueError(f"measured_q shape must be {expected_shape}, got {q.shape}")
        qd = jnp.zeros_like(q) if measured_qd is None else jnp.asarray(
            measured_qd, dtype=jnp.float32
        )
        qdd = jnp.zeros_like(q) if measured_qdd is None else jnp.asarray(
            measured_qdd, dtype=jnp.float32
        )
        target = q if position_target is None else jnp.asarray(
            position_target, dtype=jnp.float32
        )
        for name, value in (("measured_qd", qd), ("measured_qdd", qdd), ("position_target", target)):
            if value.shape != expected_shape:
                raise ValueError(f"{name} shape must be {expected_shape}, got {value.shape}")

        rmp_state = init_joint_rmp_state_jax(
            q,
            velocity=qd,
            acceleration=qdd,
            target_position=target,
            target_filter_length=self.recovered_rmp_params.target_filter_length,
            output_filter_length=self.recovered_rmp_params.filter_length,
        )
        qpos = state.data.qpos.at[:, self.arm_qadr].set(q)
        qvel = state.data.qvel.at[:, self.arm_vadr].set(qd)
        controller_kp = self.original_arm_actuator_kp[None, :] * state.dr_pd_kp_mult
        controller_kv = self.original_arm_actuator_kv[None, :] * state.dr_pd_kv_mult
        arm_ctrl = q
        if self.recovered_rmp_motion_mode:
            arm_ctrl = jnp.clip(
                q
                + controller_kv / jnp.maximum(controller_kp, 1.0e-6) * qd
                + self.recovered_rmp_acceleration_feedforward_s2[None, :] * qdd,
                self.arm_lo,
                self.arm_hi,
            )
        ctrl = state.data.ctrl.at[:, self.arm_aids_j].set(arm_ctrl)
        data = self.batched_forward(
            state.model,
            state.data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl),
        )
        racket_position = data.site_xpos[:, self.racket_site_id]
        command_buffer = jnp.broadcast_to(
            q[:, None, :], state.command_buffer.shape
        )
        actuator_command_buffer = jnp.broadcast_to(
            q[:, None, :], state.actuator_command_buffer.shape
        )
        return state._replace(
            data=data,
            arm_cmd_q=target,
            arm_cmd_qvel=jnp.zeros_like(q),
            rmp_q=rmp_state.command_q,
            rmp_qd=rmp_state.command_qd,
            rmp_qdd=rmp_state.command_qdd,
            rmp_internal_q=rmp_state.q,
            rmp_internal_qd=rmp_state.qd,
            rmp_internal_qdd=rmp_state.qdd,
            rmp_accepted_target=rmp_state.accepted_target,
            rmp_ticks_since_target_update=rmp_state.ticks_since_target_update,
            rmp_target_x=rmp_state.target_x,
            rmp_target_covariance=rmp_state.target_covariance,
            rmp_target_q_buffer=rmp_state.target_q_buffer,
            rmp_target_qd_buffer=rmp_state.target_qd_buffer,
            rmp_target_qdd_buffer=rmp_state.target_qdd_buffer,
            rmp_target_buffer_index=rmp_state.target_buffer_index,
            rmp_filter_q_buffer=rmp_state.output_q_buffer,
            rmp_filter_qd_buffer=rmp_state.output_qd_buffer,
            rmp_filter_qdd_buffer=rmp_state.output_qdd_buffer,
            rmp_filter_buffer_index=rmp_state.output_buffer_index,
            rmp_output_q_delay_buffer=jnp.broadcast_to(
                rmp_state.command_q[:, None, :],
                state.rmp_output_q_delay_buffer.shape,
            ),
            rmp_output_qd_delay_buffer=jnp.broadcast_to(
                rmp_state.command_qd[:, None, :],
                state.rmp_output_qd_delay_buffer.shape,
            ),
            rmp_output_qdd_delay_buffer=jnp.broadcast_to(
                rmp_state.command_qdd[:, None, :],
                state.rmp_output_qdd_delay_buffer.shape,
            ),
            arm_q_ref_latest=q,
            arm_q_ref_active=q,
            arm_actuator_q_ref_latest=q,
            arm_actuator_q_ref_active=q,
            arm_safe_q_ref_latest=q,
            arm_safe_qvel=qd,
            arm_safe_qacc=qdd,
            compensation_prev_qdd=qdd,
            arm_servo_command_q=q,
            arm_servo_command_qvel=qd,
            arm_applied_q=q,
            arm_applied_qvel=qd,
            arm_actuator_mode1_q=q,
            arm_actuator_mode1_qvel=qd,
            arm_actuator_mode2_q=q,
            arm_actuator_mode2_qvel=qd,
            prev_arm_qvel=qd,
            prev_racket_pos=racket_position,
            prev_racket_vel=jnp.zeros_like(racket_position),
            command_buffer=command_buffer,
            actuator_command_buffer=actuator_command_buffer,
        )

    def set_arm_pd_scales(
        self,
        state: EnvState,
        kp_scale: jax.Array,
        kd_scale: jax.Array,
    ) -> EnvState:
        """Set per-environment, per-joint XML PD scales for calibration.

        Both the batched MJX model and the qd-reference equivalent-position
        conversion are updated.  This keeps RMP q/qd/qdd replay physically
        consistent while allowing many PD candidates to run in parallel.
        """

        expected_shape = (state.arm_cmd_q.shape[0], self.act_dim)
        kp = jnp.asarray(kp_scale, dtype=jnp.float32)
        kd = jnp.asarray(kd_scale, dtype=jnp.float32)
        if kp.shape != expected_shape or kd.shape != expected_shape:
            raise ValueError(
                f"kp_scale and kd_scale must have shape {expected_shape}, "
                f"got {kp.shape} and {kd.shape}"
            )
        if bool(jnp.any(kp <= 0.0)) or bool(jnp.any(kd <= 0.0)):
            raise ValueError("PD scales must be positive")
        gain_parameters = state.model.actuator_gainprm.at[
            :, self.arm_aids_j, 0
        ].set(self.original_arm_actuator_kp[None, :] * kp)
        bias_parameters = state.model.actuator_biasprm
        bias_parameters = bias_parameters.at[:, self.arm_aids_j, 1].set(
            -self.original_arm_actuator_kp[None, :] * kp
        )
        bias_parameters = bias_parameters.at[:, self.arm_aids_j, 2].set(
            -self.original_arm_actuator_kv[None, :] * kd
        )
        model = state.model.replace(
            actuator_gainprm=gain_parameters,
            actuator_biasprm=bias_parameters,
        )
        return state._replace(
            model=model,
            dr_pd_kp_mult=kp,
            dr_pd_kv_mult=kd,
        )

    def set_rmp_joint_gain_scales(
        self,
        state: EnvState,
        kp_scale: jax.Array,
        kd_scale: jax.Array,
    ) -> EnvState:
        """Set absolute recovered-RMP gain scales for batched calibration."""

        expected_shape = (state.arm_cmd_q.shape[0], self.act_dim)
        kp = jnp.asarray(kp_scale, dtype=jnp.float32)
        kd = jnp.asarray(kd_scale, dtype=jnp.float32)
        if kp.shape != expected_shape or kd.shape != expected_shape:
            raise ValueError(
                f"RMP gain scales must have shape {expected_shape}, "
                f"got {kp.shape} and {kd.shape}"
            )
        if bool(jnp.any(kp <= 0.0)) or bool(jnp.any(kd <= 0.0)):
            raise ValueError("RMP gain scales must be positive")
        defaults = RMPParams()
        return state._replace(
            rmp_jnt_kp=float(defaults.jnt_kp) * kp,
            rmp_jnt_kd=float(defaults.jnt_kd) * kd,
        )

    def set_rmp_internal_parameters(
        self,
        state: EnvState,
        estimator_process_var: jax.Array,
        estimator_measure_var: jax.Array,
        velocity_feedforward: jax.Array,
        acceleration_feedforward: jax.Array,
        acceleration_weight: jax.Array,
        target_filter_length: jax.Array,
    ) -> EnvState:
        """Set per-environment, per-joint RMP internals for identification."""

        expected_shape = (state.arm_cmd_q.shape[0], self.act_dim)
        values = {
            "estimator_process_var": jnp.asarray(
                estimator_process_var, dtype=jnp.float32
            ),
            "estimator_measure_var": jnp.asarray(
                estimator_measure_var, dtype=jnp.float32
            ),
            "velocity_feedforward": jnp.asarray(
                velocity_feedforward, dtype=jnp.float32
            ),
            "acceleration_feedforward": jnp.asarray(
                acceleration_feedforward, dtype=jnp.float32
            ),
            "acceleration_weight": jnp.asarray(
                acceleration_weight, dtype=jnp.float32
            ),
            "target_filter_length": jnp.asarray(
                target_filter_length, dtype=jnp.int32
            ),
        }
        for name, value in values.items():
            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {value.shape}"
                )
        if (
            bool(jnp.any(values["estimator_process_var"] <= 0.0))
            or bool(jnp.any(values["estimator_measure_var"] <= 0.0))
            or bool(jnp.any(values["acceleration_weight"] < 0.0))
            or bool(jnp.any(values["target_filter_length"] <= 0))
            or bool(
                jnp.any(
                    values["target_filter_length"]
                    > state.rmp_target_q_buffer.shape[-2]
                )
            )
        ):
            raise ValueError("RMP variances/filter lengths must be positive and valid")
        return state._replace(
            rmp_estimator_joint_process_var=values["estimator_process_var"],
            rmp_estimator_joint_measure_var=values["estimator_measure_var"],
            rmp_joint_velocity_feedforward=values["velocity_feedforward"],
            rmp_joint_acceleration_feedforward=values["acceleration_feedforward"],
            rmp_acceleration_weight=values["acceleration_weight"],
            rmp_target_filter_length=values["target_filter_length"],
        )

    def set_rmp_output_delay_steps(
        self,
        state: EnvState,
        delay_steps: jax.Array,
    ) -> EnvState:
        """Set pure RMP-output transport delays for batched calibration."""

        expected_shape = (state.arm_cmd_q.shape[0], self.act_dim)
        delays = jnp.asarray(delay_steps, dtype=jnp.int32)
        if delays.shape != expected_shape:
            raise ValueError(
                f"RMP output delays must have shape {expected_shape}, "
                f"got {delays.shape}"
            )
        if bool(jnp.any(delays < 0)) or bool(
            jnp.any(delays > state.rmp_output_q_delay_buffer.shape[1])
        ):
            raise ValueError(
                "RMP output delays must fit the allocated delay buffer"
            )
        return state._replace(rmp_output_delay_steps=delays)

    def set_arm_dynamics_scales(
        self,
        state: EnvState,
        damping_scale: jax.Array,
        armature_scale: jax.Array,
    ) -> EnvState:
        """Set per-environment arm damping and armature for identification."""

        expected_shape = (state.arm_cmd_q.shape[0], self.act_dim)
        damping = jnp.asarray(damping_scale, dtype=jnp.float32)
        armature = jnp.asarray(armature_scale, dtype=jnp.float32)
        if damping.shape != expected_shape or armature.shape != expected_shape:
            raise ValueError(
                f"dynamics scales must have shape {expected_shape}, got "
                f"{damping.shape} and {armature.shape}"
            )
        if bool(jnp.any(damping <= 0.0)) or bool(jnp.any(armature <= 0.0)):
            raise ValueError("dynamics scales must be positive")
        dof_damping = state.model.dof_damping.at[:, self.arm_vadr].set(
            self.original_dof_damping[self.arm_vadr][None, :] * damping
        )
        dof_armature = state.model.dof_armature.at[:, self.arm_vadr].set(
            self.original_dof_armature[self.arm_vadr][None, :] * armature
        )
        return state._replace(
            model=state.model.replace(
                dof_damping=dof_damping,
                dof_armature=dof_armature,
            )
        )

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
        """Advance the normal policy-action control path."""
        return self._step_impl(state, action)

    def step_q_ref(
        self,
        state: EnvState,
        arm_q_ref: jax.Array,
        arm_qvel_ref: jax.Array | None = None,
        arm_qdd_ref: jax.Array | None = None,
        arm_q_horizon_ref: jax.Array | None = None,
        arm_qdd_horizon_ref: jax.Array | None = None,
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        """Advance with an externally supplied right-arm position command.

        This calibration-only entry point mirrors physical trajectory replay.
        In the legacy path the supplied command traverses the configured
        command/actuator blocks.  With ``recovered_rmp_motion_mode=True`` it is
        instead the position-only target presented to the recovered 1 kHz RMP;
        the RMP-generated q/qd/qdd then feed the XML low-level PD directly.
        When supplied, ``arm_qvel_ref`` and ``arm_qdd_ref`` are upstream policy
        integrator diagnostics; RMP motion mode intentionally does not confuse
        them with the final derivatives generated inside RMP.
        """
        q_ref = jnp.asarray(arm_q_ref, dtype=jnp.float32)
        expected_shape = (state.arm_cmd_q.shape[0], self.act_dim)
        if q_ref.shape != expected_shape:
            raise ValueError(
                f"arm_q_ref shape must be {expected_shape}, got {q_ref.shape}"
            )
        if (arm_qvel_ref is None) != (arm_qdd_ref is None):
            raise ValueError(
                "arm_qvel_ref and arm_qdd_ref must either both be supplied or both be omitted"
            )
        qvel_ref = None
        qdd_ref = None
        q_horizon_ref = None
        qdd_horizon_ref = None
        if arm_qvel_ref is not None and arm_qdd_ref is not None:
            qvel_ref = jnp.asarray(arm_qvel_ref, dtype=jnp.float32)
            qdd_ref = jnp.asarray(arm_qdd_ref, dtype=jnp.float32)
            if qvel_ref.shape != expected_shape:
                raise ValueError(
                    f"arm_qvel_ref shape must be {expected_shape}, got {qvel_ref.shape}"
                )
            if qdd_ref.shape != expected_shape:
                raise ValueError(
                    f"arm_qdd_ref shape must be {expected_shape}, got {qdd_ref.shape}"
                )
        if (arm_q_horizon_ref is None) != (arm_qdd_horizon_ref is None):
            raise ValueError(
                "arm_q_horizon_ref and arm_qdd_horizon_ref must either both be supplied or both omitted"
            )
        if arm_q_horizon_ref is not None and arm_qdd_horizon_ref is not None:
            q_horizon_ref = jnp.asarray(arm_q_horizon_ref, dtype=jnp.float32)
            qdd_horizon_ref = jnp.asarray(arm_qdd_horizon_ref, dtype=jnp.float32)
            if q_horizon_ref.ndim != 3 or q_horizon_ref.shape[0] != expected_shape[0] or q_horizon_ref.shape[2] != self.act_dim:
                raise ValueError(
                    "arm_q_horizon_ref shape must be (n_envs, horizon_steps, 7), "
                    f"got {q_horizon_ref.shape}"
                )
            if qdd_horizon_ref.shape != q_horizon_ref.shape:
                raise ValueError(
                    "arm_qdd_horizon_ref shape must match arm_q_horizon_ref, "
                    f"got {qdd_horizon_ref.shape} versus {q_horizon_ref.shape}"
                )
        zero_action = jnp.zeros_like(state.prev_action)
        return self._step_impl(
            state,
            zero_action,
            external_arm_q_ref=q_ref,
            external_arm_qvel_ref=qvel_ref,
            external_arm_qdd_ref=qdd_ref,
            external_arm_q_horizon_ref=q_horizon_ref,
            external_arm_qdd_horizon_ref=qdd_horizon_ref,
        )

    def _step_impl(
        self,
        state: EnvState,
        action: jax.Array,
        external_arm_q_ref: jax.Array | None = None,
        external_arm_qvel_ref: jax.Array | None = None,
        external_arm_qdd_ref: jax.Array | None = None,
        external_arm_q_horizon_ref: jax.Array | None = None,
        external_arm_qdd_horizon_ref: jax.Array | None = None,
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
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
            if bool(self.cfg.anti_windup_directional):
                # Suppress only a joint whose acceleration would push its
                # active command farther ahead of the measured joint.  Braking
                # remains available and one lagging joint cannot throttle all
                # seven axes.
                joint_scale = jnp.clip(
                    1.0 - jnp.abs(e_active_for_aw) / threshold,
                    min_scale,
                    1.0,
                )
                worsening = a_final * e_active_for_aw > 0.0
                applied_scale = jnp.where(worsening, joint_scale, 1.0)
                action = a_final * applied_scale
                anti_windup_scale = jnp.mean(applied_scale, axis=-1)
            else:
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

        if external_arm_q_ref is not None:
            # Derivatives are upstream diagnostic/compensation inputs only.
            # In recovered RMP mode, the recorded position remains exact here
            # and becomes the RMP target below; it is not the final servo q.
            external_q_ref = jnp.clip(
                external_arm_q_ref,
                self.arm_lo[None, :],
                self.arm_hi[None, :],
            )
            if external_arm_qvel_ref is not None and external_arm_qdd_ref is not None:
                # The normal policy path already owns these two integrator
                # states.  Keeping them explicit here makes calibration replay
                # exercise the same compensation inputs without reconstructing
                # and clipping derivatives from sampled q.
                external_qvel = external_arm_qvel_ref
                external_qdd = external_arm_qdd_ref
                if bool(self.cfg.arm_action_limiter):
                    # Calibration must obey the same deployable action
                    # integrator contract as policy execution.  Some legacy
                    # replay derivative columns contain one-sample planner
                    # spikes above 13k deg/s^2; feeding those directly makes
                    # the nominal path itself infeasible and invalidates any
                    # compensation safety comparison.
                    external_qvel = jnp.clip(
                        external_qvel,
                        -self.arm_vel_limit_rad_s[None, :],
                        self.arm_vel_limit_rad_s[None, :],
                    )
                    external_qdd = jnp.clip(
                        external_qdd,
                        -self.arm_acc_limit_rad_s2[None, :],
                        self.arm_acc_limit_rad_s2[None, :],
                    )
            else:
                external_qvel = (
                    external_q_ref - state.arm_cmd_q
                ) / max(self.dt, 1e-6)
                external_qvel = jnp.clip(
                    external_qvel,
                    -self.arm_vel_limit_rad_s[None, :],
                    self.arm_vel_limit_rad_s[None, :],
                )
                external_qdd = (
                    external_qvel - state.arm_cmd_qvel
                ) / max(self.dt, 1e-6)
                external_qdd = jnp.clip(
                    external_qdd,
                    -self.arm_acc_limit_rad_s2[None, :],
                    self.arm_acc_limit_rad_s2[None, :],
                )
            arm_cmd_q = external_q_ref
            cmd_qvel = external_qvel
            raw_cmd_qvel = external_qvel
            desired_qdd = external_qdd
            desired_qdd_raw = external_qdd

        recovered_rmp_state = JointRMPStateJax(
            q=state.rmp_internal_q,
            qd=state.rmp_internal_qd,
            qdd=state.rmp_internal_qdd,
            command_q=state.rmp_q,
            command_qd=state.rmp_qd,
            command_qdd=state.rmp_qdd,
            accepted_target=state.rmp_accepted_target,
            ticks_since_target_update=state.rmp_ticks_since_target_update,
            target_x=state.rmp_target_x,
            target_covariance=state.rmp_target_covariance,
            target_q_buffer=state.rmp_target_q_buffer,
            target_qd_buffer=state.rmp_target_qd_buffer,
            target_qdd_buffer=state.rmp_target_qdd_buffer,
            target_buffer_index=state.rmp_target_buffer_index,
            output_q_buffer=state.rmp_filter_q_buffer,
            output_qd_buffer=state.rmp_filter_qd_buffer,
            output_qdd_buffer=state.rmp_filter_qdd_buffer,
            output_buffer_index=state.rmp_filter_buffer_index,
        )
        if self.recovered_rmp_motion_mode:
            (
                recovered_rmp_state,
                rmp_command_q_history,
                rmp_command_qvel_history,
                rmp_command_qdd_history,
            ) = joint_rmp_step_jax(
                recovered_rmp_state,
                arm_cmd_q,
                self.arm_lo,
                self.arm_hi,
                self.arm_vel_limit_rad_s,
                self.arm_acc_limit_rad_s2,
                self.recovered_rmp_jerk_limit_rad_s3,
                self.recovered_rmp_params,
                self.recovered_rmp_period_s,
                jnt_kp=state.rmp_jnt_kp,
                jnt_kd=state.rmp_jnt_kd,
                estimator_joint_process_var=(
                    state.rmp_estimator_joint_process_var
                ),
                estimator_joint_measure_var=(
                    state.rmp_estimator_joint_measure_var
                ),
                joint_vel_ffd=state.rmp_joint_velocity_feedforward,
                joint_acc_ffd=state.rmp_joint_acceleration_feedforward,
                acc_weight=state.rmp_acceleration_weight,
                target_filter_length=state.rmp_target_filter_length,
                servo_steps=self.recovered_rmp_servo_steps,
                target_update_delay_steps=(
                    self.recovered_rmp_target_update_delay_steps
                ),
                use_safety_bounds=bool(
                    self.cfg.recovered_rmp_use_safety_bounds
                ),
                use_jerk_limit=bool(self.cfg.recovered_rmp_use_jerk_limit),
                return_history=True,
            )
            rmp_command_q = recovered_rmp_state.command_q
            rmp_command_qvel = recovered_rmp_state.command_qd
            rmp_command_qdd = recovered_rmp_state.command_qdd
        else:
            # Preserve the original environment exactly when the recovered
            # mode is disabled.
            rmp_command_q = arm_cmd_q
            rmp_command_qvel = cmd_qvel
            rmp_command_qdd = desired_qdd
            rmp_command_q_history = jnp.broadcast_to(
                rmp_command_q[None, ...],
                (int(self.cfg.frame_skip),) + rmp_command_q.shape,
            )
            rmp_command_qvel_history = jnp.broadcast_to(
                rmp_command_qvel[None, ...],
                (int(self.cfg.frame_skip),) + rmp_command_qvel.shape,
            )
            rmp_command_qdd_history = jnp.broadcast_to(
                rmp_command_qdd[None, ...],
                (int(self.cfg.frame_skip),) + rmp_command_qdd.shape,
            )

        delay_buffer_length = self.recovered_rmp_output_delay_buffer_length

        def delayed_rmp_history(
            delay_buffer: jax.Array, command_history: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            combined = jnp.concatenate(
                (delay_buffer, jnp.swapaxes(command_history, 0, 1)), axis=1
            )
            substep_indices = jnp.arange(
                command_history.shape[0], dtype=jnp.int32
            )
            source_indices = (
                delay_buffer_length
                + substep_indices[None, :, None]
                - state.rmp_output_delay_steps[:, None, :]
            )
            delayed = jnp.take_along_axis(combined, source_indices, axis=1)
            return jnp.swapaxes(delayed, 0, 1), combined[:, -delay_buffer_length:, :]

        low_level_rmp_q_history, next_rmp_q_delay_buffer = delayed_rmp_history(
            state.rmp_output_q_delay_buffer, rmp_command_q_history
        )
        (
            low_level_rmp_qvel_history,
            next_rmp_qd_delay_buffer,
        ) = delayed_rmp_history(
            state.rmp_output_qd_delay_buffer, rmp_command_qvel_history
        )
        (
            low_level_rmp_qdd_history,
            next_rmp_qdd_delay_buffer,
        ) = delayed_rmp_history(
            state.rmp_output_qdd_delay_buffer, rmp_command_qdd_history
        )
        low_level_rmp_q = low_level_rmp_q_history[-1]
        low_level_rmp_qvel = low_level_rmp_qvel_history[-1]
        low_level_rmp_qdd = low_level_rmp_qdd_history[-1]

        arm_q_ref_latest = rmp_command_q
        servo_velocity_scale = float(self.cfg.arm_servo_target_velocity_scale)
        servo_acceleration_scale = float(self.cfg.arm_servo_target_acceleration_scale)
        if not (0.0 < servo_velocity_scale <= 1.0):
            raise ValueError("arm_servo_target_velocity_scale must be in (0, 1]")
        if not (0.0 < servo_acceleration_scale <= 1.0):
            raise ValueError("arm_servo_target_acceleration_scale must be in (0, 1]")
        servo_velocity_limit = self.arm_vel_limit_rad_s * servo_velocity_scale
        servo_acceleration_limit = self.arm_acc_limit_rad_s2 * servo_acceleration_scale
        planner_before_actuator = bool(
            self.cfg.arm_servo_planner_before_actuator_model
            and self.cfg.arm_servo_target_tracking_planner
        )
        comp_target_q = rmp_command_q
        comp_target_qvel = rmp_command_qvel
        comp_target_qdd = rmp_command_qdd
        comp_mode = str(self.cfg.actuator_compensation_mode or "none").strip().lower().replace("-", "_")
        if bool(self.cfg.actuator_lead_compensation) and comp_mode in {"none", "off", "false", "0"}:
            comp_mode = "lead"
        bridger_mode = comp_mode in {"sim2real_bridger", "constrained_inverse_mpc", "bridger"}
        model_inverse_mlp_mode = comp_mode in {
            "sport_model_inverse_mlp",
            "model_inverse_mlp",
            "causal_model_inverse_mlp",
        }
        analytic_inverse_mode = comp_mode in {
            "sport_analytic_inverse",
            "analytic_inverse",
            "sport_jerk_inverse",
        }
        horizon_inverse_mode = comp_mode in {
            "sport_horizon_inverse",
            "horizon_inverse",
            "planned_horizon_inverse",
            "sport_bandlimited_horizon_inverse",
            "bandlimited_horizon_inverse",
        }
        bandlimited_horizon_inverse_mode = comp_mode in {
            "sport_bandlimited_horizon_inverse",
            "bandlimited_horizon_inverse",
        }
        regularized_inverse_mode = comp_mode in {
            "sport_regularized_inverse",
            "regularized_inverse",
            "bandlimited_model_inverse",
            "sport_safe_analytic_inverse",
            "safe_analytic_inverse",
            "sport_persistent_analytic_inverse",
            "persistent_analytic_inverse",
            "sport_persistent_analytic_smith",
            "sport_persistent_analytic_smith_dob",
            "sport_persistent_analytic_smith_dob_harmonic",
            "sport_persistent_analytic_full",
        }
        persistent_analytic_inverse_mode = comp_mode in {
            "sport_persistent_analytic_inverse",
            "persistent_analytic_inverse",
            "sport_persistent_analytic_smith",
            "sport_persistent_analytic_smith_dob",
            "sport_persistent_analytic_smith_dob_harmonic",
            "sport_persistent_analytic_full",
        }
        filtered_smith_mode = comp_mode in {
            "sport_persistent_analytic_smith",
            "sport_persistent_analytic_smith_dob",
            "sport_persistent_analytic_smith_dob_harmonic",
            "sport_persistent_analytic_full",
        }
        low_bandwidth_dob_mode = comp_mode in {
            "sport_persistent_analytic_smith_dob",
            "sport_persistent_analytic_smith_dob_harmonic",
            "sport_persistent_analytic_full",
        }
        harmonic_prediction_mode = comp_mode in {
            "sport_persistent_analytic_smith_dob_harmonic",
            "sport_persistent_analytic_full",
        }
        racket_reference_governor_mode = comp_mode == "sport_persistent_analytic_full"
        accel_filter_tau = max(
            0.0,
            float(self.cfg.actuator_regularized_inverse_accel_filter_tau_s),
        )
        accel_filter_alpha = (
            1.0
            if accel_filter_tau <= 1e-9
            else self.dt / (accel_filter_tau + self.dt)
        )
        compensation_filtered_qdd = (
            state.compensation_filtered_qdd
            + accel_filter_alpha
            * (comp_target_qdd - state.compensation_filtered_qdd)
        )
        # Two identical causal poles give substantially stronger attenuation
        # above the fitted actuator bandwidth than a single pole, while
        # retaining a small, deterministic state that is easy to reproduce in
        # the NumPy robot runtime.  At the observed 5.67 Hz failure mode and
        # tau=50 ms this suppresses the qdd path to roughly 24% amplitude.
        compensation_filtered_qdd_stage2 = (
            state.compensation_filtered_qdd_stage2
            + accel_filter_alpha
            * (
                compensation_filtered_qdd
                - state.compensation_filtered_qdd_stage2
            )
        )
        raw_compensation_jerk = (
            comp_target_qdd - state.compensation_prev_qdd
        ) / max(self.dt, 1e-6)
        jerk_tau = max(
            0.0, float(self.cfg.actuator_analytic_inverse_jerk_filter_tau_s)
        )
        jerk_alpha = 1.0 if jerk_tau <= 1e-9 else self.dt / (jerk_tau + self.dt)
        compensation_filtered_jerk = (
            state.compensation_filtered_jerk
            + jerk_alpha
            * (raw_compensation_jerk - state.compensation_filtered_jerk)
        )
        # Filtered-Smith innovation: the identified command-side plant state
        # is the nominal prediction and joint q is the measured output.  DOB
        # uses a second, deliberately slower pole of the same model residual.
        # The state reuse is mode-local: these arrays are otherwise only used
        # by the legacy jerk inverse path.
        # Smith/DOB innovation compares measured q with the output of the
        # nominal second-order command model.  Using delayed sent-q here would
        # misclassify the plant's normal phase lag as a disturbance and would
        # compensate the same delay twice.
        model_residual = (
            state.arm_applied_q - state.data.qpos[:, self.arm_qadr]
        )
        smith_bw = max(0.0, float(self.cfg.actuator_filtered_smith_bandwidth_hz))
        smith_alpha = (
            1.0
            if smith_bw <= 1e-9
            else 1.0 - np.exp(-2.0 * np.pi * smith_bw * self.dt)
        )
        smith_residual = state.compensation_smith_residual + smith_alpha * (
            model_residual - state.compensation_smith_residual
        )
        dob_bw = max(0.0, float(self.cfg.actuator_dob_bandwidth_hz))
        dob_alpha = (
            1.0
            if dob_bw <= 1e-9
            else 1.0 - np.exp(-2.0 * np.pi * dob_bw * self.dt)
        )
        dob_residual = state.compensation_dob_residual + dob_alpha * (
            model_residual - state.compensation_dob_residual
        )
        if regularized_inverse_mode:
            # For G(s)=g*wn^2/(s^2+2*zeta*wn*s+wn^2)*exp(-d*s), a
            # second-order Taylor approximation of the desired trajectory at
            # t+d gives the causal inverse below.  Crucially, qdd is first
            # low-pass filtered at the actuator bandwidth; using the raw PPO
            # qdd here lets the policy bypass its q/qdot integrator.
            wn = (
                self.actuator_cmd_second_order_wn[None, :]
                * state.second_order_frequency_scale[:, None]
            )
            zeta = jnp.clip(
                self.actuator_cmd_second_order_zeta[None, :]
                * state.second_order_damping_scale[:, None],
                0.03,
                0.99,
            )
            gain = jnp.maximum(
                1e-3,
                self.actuator_cmd_second_order_gain[None, :]
                * state.second_order_gain_scale[:, None],
            )
            if self.actuator_cmd_delay_steps_per_joint is None:
                delay_s = jnp.zeros_like(wn)
            else:
                model_delay_steps = jnp.maximum(
                    0,
                    self.actuator_cmd_delay_steps_per_joint[None, :]
                    + state.second_order_delay_offset_steps[:, None],
                )
                delay_s = model_delay_steps.astype(jnp.float32) * self.dt
            effective_delay_s = delay_s * float(
                self.cfg.actuator_regularized_inverse_preview_scale
            )
            velocity_coefficient = effective_delay_s + 2.0 * zeta / wn
            acceleration_coefficient = (
                0.5 * effective_delay_s * effective_delay_s
                + 2.0 * zeta * effective_delay_s / wn
                + 1.0 / (wn * wn)
            )
            model_inverse = self.warm_arm_q[None, :] + (
                (comp_target_q - self.warm_arm_q[None, :])
                + velocity_coefficient * comp_target_qvel
                + acceleration_coefficient * compensation_filtered_qdd_stage2
            ) / gain
            if harmonic_prediction_mode:
                # Exact finite-horizon displacement for a local harmonic
                # segment q(t)=c+a*sin(wt)+b*cos(wt).  Relative to the Taylor
                # delay term already present above, this bounded correction
                # avoids assuming constant acceleration for the entire delay.
                harmonic_w = 2.0 * np.pi * max(
                    1e-6,
                    float(self.cfg.actuator_harmonic_prediction_frequency_hz),
                )
                harmonic_phase = harmonic_w * delay_s
                harmonic_q = (
                    comp_target_q
                    + jnp.sin(harmonic_phase) / harmonic_w * comp_target_qvel
                    + (1.0 - jnp.cos(harmonic_phase))
                    / (harmonic_w * harmonic_w)
                    * compensation_filtered_qdd_stage2
                )
                harmonic_qvel = (
                    jnp.cos(harmonic_phase) * comp_target_qvel
                    + jnp.sin(harmonic_phase) / harmonic_w
                    * compensation_filtered_qdd_stage2
                )
                harmonic_qdd = (
                    -harmonic_w * jnp.sin(harmonic_phase) * comp_target_qvel
                    + jnp.cos(harmonic_phase) * compensation_filtered_qdd_stage2
                )
                harmonic_inverse = self.warm_arm_q[None, :] + (
                    (harmonic_q - self.warm_arm_q[None, :])
                    + 2.0 * zeta / wn * harmonic_qvel
                    + harmonic_qdd / (wn * wn)
                ) / gain
                harmonic_jerk_mismatch = (
                    compensation_filtered_jerk
                    + harmonic_w * harmonic_w * comp_target_qvel
                )
                confidence_scale = np.deg2rad(
                    max(
                        1e-6,
                        float(
                            self.cfg.actuator_harmonic_prediction_confidence_scale_deg_s3
                        ),
                    )
                )
                harmonic_confidence = 1.0 / (
                    1.0 + jnp.square(harmonic_jerk_mismatch / confidence_scale)
                )
                harmonic_blend = (
                    float(self.cfg.actuator_harmonic_prediction_gain)
                    * harmonic_confidence
                )
                model_inverse = model_inverse + harmonic_blend * (
                    harmonic_inverse - model_inverse
                )
            if filtered_smith_mode:
                model_inverse = model_inverse + float(
                    self.cfg.actuator_filtered_smith_gain
                ) * smith_residual / gain
            if low_bandwidth_dob_mode:
                model_inverse = model_inverse + float(
                    self.cfg.actuator_dob_gain
                ) * dob_residual / gain
            inverse_delta = float(
                self.cfg.actuator_regularized_inverse_blend
            ) * (model_inverse - comp_target_q)
            max_delta = max(
                0.0,
                float(self.cfg.actuator_regularized_inverse_max_delta_rad),
            )
            if max_delta > 0.0:
                inverse_delta = jnp.clip(inverse_delta, -max_delta, max_delta)
            arm_actuator_q_ref_latest = jnp.clip(
                comp_target_q + inverse_delta,
                self.arm_lo,
                self.arm_hi,
            )
        elif horizon_inverse_mode:
            if external_arm_q_horizon_ref is None or external_arm_qdd_horizon_ref is None:
                # Normal PPO/deployment path: the seven policy outputs are the
                # bounded joint accelerations used by the action integrator.
                # Treat that acceleration as the compact parameterization of
                # a receding 125 ms strategy plan.  This avoids a 26*7 action
                # head while keeping every horizon sample available on the
                # real robot from policy/integrator state alone.
                horizon_steps = int(self.cfg.actuator_horizon_inverse_steps)
                if horizon_steps < 2:
                    raise ValueError("actuator_horizon_inverse_steps must be >= 2")

                def plan_step(carry, _):
                    planned_q, planned_qvel = carry
                    next_qvel = planned_qvel + comp_target_qdd * self.dt
                    if bool(self.cfg.arm_action_limiter):
                        next_qvel = jnp.clip(
                            next_qvel,
                            -self.arm_vel_limit_rad_s[None, :],
                            self.arm_vel_limit_rad_s[None, :],
                        )
                    next_q = jnp.clip(
                        planned_q + next_qvel * self.dt,
                        self.arm_lo[None, :],
                        self.arm_hi[None, :],
                    )
                    return (next_q, next_qvel), next_q

                (_, _), planned_q_tail = jax.lax.scan(
                    plan_step,
                    (comp_target_q, comp_target_qvel),
                    xs=None,
                    length=horizon_steps - 1,
                )
                external_arm_q_horizon_ref = jnp.concatenate(
                    [comp_target_q[:, None, :], jnp.swapaxes(planned_q_tail, 0, 1)],
                    axis=1,
                )
                external_arm_qdd_horizon_ref = jnp.broadcast_to(
                    comp_target_qdd[:, None, :],
                    external_arm_q_horizon_ref.shape,
                )
            horizon_lead_s = (
                np.asarray(self.cfg.actuator_horizon_inverse_lead_s, dtype=np.float32)
                - np.float32(0.010)
                if bandlimited_horizon_inverse_mode
                else self.cfg.actuator_horizon_inverse_lead_s
            )
            lead_steps = jnp.asarray(horizon_lead_s, dtype=jnp.float32) / max(
                self.dt, 1e-6
            )
            lead_low = jnp.floor(lead_steps).astype(jnp.int32)
            lead_high = lead_low + 1
            horizon_last = int(external_arm_q_horizon_ref.shape[1]) - 1
            if horizon_last < int(np.ceil(max(horizon_lead_s) / max(self.dt, 1e-6))):
                raise ValueError(
                    "strategy horizon is shorter than actuator_horizon_inverse_lead_s"
                )
            lead_low = jnp.clip(lead_low, 0, horizon_last)
            lead_high = jnp.clip(lead_high, 0, horizon_last)
            joint_index = jnp.arange(self.act_dim)[None, :]
            env_index = jnp.arange(comp_target_q.shape[0])[:, None]
            q_low = external_arm_q_horizon_ref[env_index, lead_low[None, :], joint_index]
            q_high = external_arm_q_horizon_ref[env_index, lead_high[None, :], joint_index]
            qdd_low = external_arm_qdd_horizon_ref[env_index, lead_low[None, :], joint_index]
            qdd_high = external_arm_qdd_horizon_ref[env_index, lead_high[None, :], joint_index]
            fraction = (lead_steps - jnp.floor(lead_steps))[None, :]
            future_q = q_low + fraction * (q_high - q_low)
            future_qdd = qdd_low + fraction * (qdd_high - qdd_low)
            if bandlimited_horizon_inverse_mode:
                bandlimited_alpha = self.dt / (0.020 + self.dt)
                compensation_filtered_qdd = (
                    state.compensation_filtered_qdd
                    + bandlimited_alpha
                    * (future_qdd - state.compensation_filtered_qdd)
                )
                compensation_filtered_qdd_stage2 = (
                    state.compensation_filtered_qdd_stage2
                    + bandlimited_alpha
                    * (
                        compensation_filtered_qdd
                        - state.compensation_filtered_qdd_stage2
                    )
                )
                future_qdd = compensation_filtered_qdd_stage2
                accel_coefficient = jnp.full(
                    (1, self.act_dim), 0.00165, dtype=jnp.float32
                )
            else:
                accel_coefficient = jnp.asarray(
                    self.cfg.actuator_horizon_inverse_accel_s2,
                    dtype=jnp.float32,
                )[None, :]
            # The 12 degree cap belongs to the analytic second-order inverse
            # term only.  Capping the complete preview displacement silently
            # removes the delay inverse on fast trajectories (simcode2/3).
            accel_delta = accel_coefficient * future_qdd
            max_delta = max(
                0.0, float(self.cfg.actuator_horizon_inverse_max_delta_rad)
            )
            if max_delta > 0.0:
                accel_delta = jnp.clip(accel_delta, -max_delta, max_delta)
            inverse_command = future_q + accel_delta
            arm_actuator_q_ref_latest = jnp.clip(
                inverse_command, self.arm_lo, self.arm_hi
            )
        elif analytic_inverse_mode:
            qvel_coefficient = jnp.asarray(
                self.cfg.actuator_analytic_inverse_qvel_s, dtype=jnp.float32
            )[None, :]
            qdd_coefficient = jnp.asarray(
                self.cfg.actuator_analytic_inverse_qdd_s2, dtype=jnp.float32
            )[None, :]
            jerk_coefficient = jnp.asarray(
                self.cfg.actuator_analytic_inverse_jerk_s3, dtype=jnp.float32
            )[None, :]
            position_gain = jnp.asarray(
                self.cfg.actuator_analytic_inverse_position_gain, dtype=jnp.float32
            )[None, :]
            velocity_gain = jnp.asarray(
                self.cfg.actuator_analytic_inverse_velocity_gain_s, dtype=jnp.float32
            )[None, :]
            model_gain = jnp.where(
                jnp.abs(self.actuator_cmd_second_order_gain) <= 1e-6,
                1.0,
                self.actuator_cmd_second_order_gain,
            )[None, :]
            inverse_nominal = (
                comp_target_q
                + qvel_coefficient * comp_target_qvel
                + qdd_coefficient * comp_target_qdd
                + jerk_coefficient * compensation_filtered_jerk
            ) / model_gain
            q_actual = state.data.qpos[:, self.arm_qadr]
            dq_actual = state.data.qvel[:, self.arm_vadr]
            inverse_command = (
                inverse_nominal
                + position_gain * (comp_target_q - q_actual)
                + velocity_gain * (comp_target_qvel - dq_actual)
            )
            inverse_max_delta = max(
                0.0, float(self.cfg.actuator_analytic_inverse_max_delta_rad)
            )
            if inverse_max_delta > 0.0:
                inverse_command = comp_target_q + jnp.clip(
                    inverse_command - comp_target_q,
                    -inverse_max_delta,
                    inverse_max_delta,
                )
            arm_actuator_q_ref_latest = jnp.clip(
                inverse_command, self.arm_lo, self.arm_hi
            )
        elif model_inverse_mlp_mode:
            history_with_current = jnp.concatenate(
                [state.command_buffer, comp_target_q[:, None, :]], axis=1
            )
            recent = history_with_current[:, -self.model_inverse_mlp_history :, :]
            current_first = jnp.flip(recent, axis=1)
            history_differences = current_first[:, 1:, :] - current_first[:, :1, :]
            inverse_features = jnp.concatenate(
                [
                    history_differences.reshape((comp_target_q.shape[0], -1)),
                    comp_target_qvel,
                    comp_target_qdd,
                ],
                axis=1,
            )
            inverse_value = (
                inverse_features - self.model_inverse_mlp_x_mean[None, :]
            ) / self.model_inverse_mlp_x_std[None, :]
            for weight, bias in zip(
                self.model_inverse_mlp_weights[:2],
                self.model_inverse_mlp_biases[:2],
            ):
                inverse_value = inverse_value @ weight.T + bias[None, :]
                inverse_value = jax.nn.silu(inverse_value)
            inverse_delta = (
                inverse_value @ self.model_inverse_mlp_weights[2].T
                + self.model_inverse_mlp_biases[2][None, :]
            ) * self.model_inverse_mlp_y_scale
            inverse_max_delta = max(
                0.0, float(self.cfg.actuator_model_inverse_mlp_max_delta_rad)
            )
            if inverse_max_delta > 0.0:
                inverse_delta = jnp.clip(
                    inverse_delta, -inverse_max_delta, inverse_max_delta
                )
            q_actual = state.data.qpos[:, self.arm_qadr]
            dq_actual = state.data.qvel[:, self.arm_vadr]
            inverse_feedback = (
                float(self.cfg.actuator_model_inverse_mlp_position_gain)
                * (comp_target_q - q_actual)
                + float(self.cfg.actuator_model_inverse_mlp_velocity_gain_s)
                * (comp_target_qvel - dq_actual)
            )
            arm_actuator_q_ref_latest = jnp.clip(
                comp_target_q + inverse_delta + inverse_feedback,
                self.arm_lo,
                self.arm_hi,
            )
        elif comp_mode in {"inverse_mpc", "regularized_inverse_mpc", "mpc"} or bridger_mode:
            second_order_mpc = self.actuator_cmd_model == "second_order"
            if (
                second_order_mpc
                and self.actuator_cmd_delay_steps_per_joint is not None
            ):
                comp_delay_steps = jnp.rint(
                    self.actuator_cmd_delay_steps_per_joint.astype(jnp.float32)
                    * max(0.0, float(self.cfg.actuator_mpc_delay_scale))
                ).astype(jnp.int32)[None, :]
                comp_delay_steps = jnp.broadcast_to(
                    comp_delay_steps,
                    (comp_target_q.shape[0], self.act_dim),
                )
            else:
                comp_delay_steps = jnp.rint(
                    delay_steps.astype(jnp.float32)
                    * max(0.0, float(self.cfg.actuator_mpc_delay_scale))
                ).astype(jnp.int32)
            comp_delay_steps = jnp.clip(comp_delay_steps, 0, self.command_buffer_len - 1)
            tau_est = jnp.maximum(state.actuator_cmd_tau * max(0.0, float(self.cfg.actuator_mpc_tau_scale)), 0.0)
            alpha_est = jnp.where(tau_est <= 1e-6, 1.0, self.dt / (tau_est + self.dt))
            pred_buffer = jnp.concatenate(
                [state.actuator_command_buffer[:, 1:, :], comp_target_q[:, None, :]],
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
            mpc_horizon_steps = max(1, int(self.cfg.actuator_mpc_horizon_steps))
            if second_order_mpc:
                if mpc_feedback_source in {"actual", "sim", "joint", "joint_state"}:
                    v_pred = state.data.qvel[:, self.arm_vadr]
                else:
                    v_pred = state.arm_applied_qvel
                wn_est = self.actuator_cmd_second_order_wn[None, :]
                zeta_est = self.actuator_cmd_second_order_zeta[None, :]
                root_est = jnp.sqrt(jnp.maximum(1.0 - zeta_est * zeta_est, 1e-8))
                wd_est = wn_est * root_est

                def transition_coefficients(duration_s: float):
                    decay_est = jnp.exp(-zeta_est * wn_est * duration_s)
                    phase_est = wd_est * duration_s
                    sin_est = jnp.sin(phase_est)
                    cos_est = jnp.cos(phase_est)
                    a11_est = decay_est * (
                        cos_est + zeta_est / root_est * sin_est
                    )
                    a12_est = decay_est * sin_est / jnp.maximum(wd_est, 1e-8)
                    a21_est = (
                        -decay_est * wn_est * wn_est * sin_est
                        / jnp.maximum(wd_est, 1e-8)
                    )
                    a22_est = decay_est * (
                        cos_est - zeta_est / root_est * sin_est
                    )
                    return a11_est, a12_est, a21_est, a22_est

                a11_step, a12_step, a21_step, a22_step = transition_coefficients(
                    self.dt
                )
                for s in range(self.command_buffer_len - 1):
                    joint_idx = jnp.clip(
                        self.command_buffer_len - 1 - comp_delay_steps + s,
                        0,
                        self.command_buffer_len - 1,
                    )
                    queued = jnp.take_along_axis(
                        pred_buffer,
                        joint_idx[:, None, :],
                        axis=1,
                    )[:, 0, :]
                    filter_target = self.warm_arm_q[None, :] + self.actuator_cmd_second_order_gain[None, :] * (
                        queued - self.warm_arm_q[None, :]
                    )
                    y_next = (
                        a11_step * y_pred
                        + a12_step * v_pred
                        + (1.0 - a11_step) * filter_target
                    )
                    v_next = (
                        a21_step * y_pred
                        + a22_step * v_pred
                        - a21_step * filter_target
                    )
                    queued_active = s < comp_delay_steps
                    y_pred = jnp.where(queued_active, y_next, y_pred)
                    v_pred = jnp.where(queued_active, v_next, v_pred)

                total_horizon = (
                    comp_delay_steps.astype(jnp.float32)
                    + float(mpc_horizon_steps)
                ) * self.dt
                target_future = (
                    comp_target_q
                    + total_horizon * comp_target_qvel
                    + 0.5 * total_horizon * total_horizon * comp_target_qdd
                )
                a11_h, a12_h, _a21_h, _a22_h = transition_coefficients(
                    float(mpc_horizon_steps) * self.dt
                )
                response = 1.0 - a11_h
                gain_est_joint = jnp.where(
                    jnp.abs(self.actuator_cmd_second_order_gain) <= 1e-6,
                    1.0,
                    self.actuator_cmd_second_order_gain,
                )[None, :]
                k = response * gain_est_joint
                b = (
                    a11_h * y_pred
                    + a12_h * v_pred
                    + response
                    * (1.0 - gain_est_joint)
                    * self.warm_arm_q[None, :]
                )
            else:
                for s in range(self.command_buffer_len - 1):
                    idx = jnp.clip(self.command_buffer_len - 1 - comp_delay_steps + s, 0, self.command_buffer_len - 1)
                    queued = pred_buffer[jnp.arange(pred_buffer.shape[0]), idx]
                    filter_target = self.warm_arm_q[None, :] + state.actuator_cmd_gain[:, None] * (
                        queued - self.warm_arm_q[None, :]
                    )
                    y_next = y_pred + alpha_est[:, None] * (filter_target - y_pred)
                    y_pred = jnp.where((s < comp_delay_steps)[:, None], y_next, y_pred)

                total_horizon = (
                    comp_delay_steps.astype(jnp.float32)
                    + float(mpc_horizon_steps)
                ) * self.dt
                target_future = (
                    comp_target_q
                    + total_horizon[:, None] * comp_target_qvel
                    + 0.5
                    * total_horizon[:, None]
                    * total_horizon[:, None]
                    * comp_target_qdd
                )
                decay = jnp.power(
                    jnp.clip(1.0 - alpha_est, 0.0, 1.0),
                    float(mpc_horizon_steps),
                )
                response = 1.0 - decay
                gain_est = jnp.where(
                    jnp.abs(state.actuator_cmd_gain) <= 1e-6,
                    1.0,
                    state.actuator_cmd_gain,
                )
                k = response[:, None] * gain_est[:, None]
                b = (
                    decay[:, None] * y_pred
                    + response[:, None]
                    * (1.0 - gain_est[:, None])
                    * self.warm_arm_q[None, :]
                )
            last_actuator_cmd = state.actuator_command_buffer[:, -1, :]
            wt = max(0.0, float(self.cfg.actuator_mpc_tracking_weight))
            wn = max(0.0, float(self.cfg.actuator_mpc_nominal_weight))
            wd = max(0.0, float(self.cfg.actuator_mpc_delta_weight))
            denom = wt * k * k + wn + wd
            mpc_cmd = (wt * k * (target_future - b) + wn * comp_target_q + wd * last_actuator_cmd) / jnp.maximum(
                denom,
                1e-6,
            )
            mpc_delta = float(self.cfg.actuator_mpc_beta) * (mpc_cmd - comp_target_q)
            max_mpc = float(self.cfg.actuator_mpc_max_delta_rad)
            if max_mpc > 0.0:
                mpc_delta = jnp.clip(mpc_delta, -max_mpc, max_mpc)
            arm_actuator_q_ref_latest = jnp.clip(comp_target_q + mpc_delta, self.arm_lo, self.arm_hi)
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
                [state.actuator_command_buffer[:, 1:, :], comp_target_q[:, None, :]],
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
            target_future = comp_target_q + horizon[:, None] * comp_target_qvel + 0.5 * horizon[:, None] * horizon[:, None] * comp_target_qdd
            inv_filter_target = (target_future - (1.0 - alpha_est[:, None]) * y_pred) / jnp.maximum(
                alpha_est[:, None],
                1e-6,
            )
            gain_est = jnp.where(jnp.abs(state.actuator_cmd_gain) <= 1e-6, 1.0, state.actuator_cmd_gain)
            inv_cmd = self.warm_arm_q[None, :] + (inv_filter_target - self.warm_arm_q[None, :]) / gain_est[:, None]
            inverse_delta = float(self.cfg.actuator_inverse_beta) * (inv_cmd - comp_target_q)
            max_inverse = float(self.cfg.actuator_inverse_max_delta_rad)
            if max_inverse > 0.0:
                inverse_delta = jnp.clip(inverse_delta, -max_inverse, max_inverse)
            arm_actuator_q_ref_latest = jnp.clip(
                comp_target_q + inverse_delta,
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
                lead_time[:, None] * comp_target_qvel
                + 0.5 * lead_time[:, None] * lead_time[:, None] * comp_target_qdd
            )
            max_lead = float(self.cfg.actuator_lead_max_delta_rad)
            if max_lead > 0.0:
                lead_delta = jnp.clip(lead_delta, -max_lead, max_lead)
            arm_actuator_q_ref_latest = jnp.clip(comp_target_q + lead_delta, self.arm_lo, self.arm_hi)
        else:
            arm_actuator_q_ref_latest = comp_target_q

        if bandlimited_horizon_inverse_mode:
            compensation_target_qvel = jnp.clip(
                (
                    arm_actuator_q_ref_latest
                    - state.arm_actuator_q_ref_latest
                )
                / max(self.dt, 1e-6),
                -self.arm_vel_limit_rad_s,
                self.arm_vel_limit_rad_s,
            )
            (
                arm_safe_q_ref_latest,
                arm_safe_qvel,
                arm_safe_qacc,
                arm_safe_feasible,
                arm_safe_jerk_feasible,
                arm_safe_interval_low,
                arm_safe_interval_high,
            ) = constrained_compensation_step_jax(
                arm_actuator_q_ref_latest,
                compensation_target_qvel,
                state.arm_safe_q_ref_latest,
                state.arm_safe_qvel,
                state.arm_safe_qacc,
                self.arm_lo,
                self.arm_hi,
                self.arm_vel_limit_rad_s,
                self.arm_acc_limit_rad_s2
                + np.deg2rad(
                    float(
                        self.cfg.actuator_compensation_acc_limit_margin_deg_s2
                    )
                ),
                jnp.deg2rad(
                    jnp.full_like(
                        self.arm_acc_limit_rad_s2,
                        float(self.cfg.actuator_compensation_jerk_limit_deg_s3),
                    )
                ),
                dt=self.dt,
                natural_frequency_hz=float(
                    self.cfg.actuator_compensation_governor_natural_frequency_hz
                ),
                damping_ratio=1.0,
                target_qacc=jnp.zeros_like(comp_target_qdd),
            )
            if racket_reference_governor_mode:
                # Scale only the acceleration introduced by compensation so
                # the policy's own feasible stroke is preserved.  The fixed
                # warm-pose vertical Jacobian makes this projection cheap,
                # deterministic, and directly portable to the robot runtime.
                compensation_qacc = arm_safe_qacc - comp_target_qdd
                racket_comp_acc = jnp.sum(
                    compensation_qacc
                    * self.racket_vertical_arm_jacobian[None, :],
                    axis=-1,
                )
                racket_acc_limit = max(
                    1e-6,
                    float(
                        self.cfg.actuator_racket_reference_governor_acc_limit_m_s2
                    ),
                )
                racket_scale = jnp.minimum(
                    1.0,
                    racket_acc_limit / jnp.maximum(jnp.abs(racket_comp_acc), 1e-6),
                )
                governed_qacc = comp_target_qdd + racket_scale[:, None] * compensation_qacc
                feasible_qacc_low = (
                    arm_safe_interval_low - state.arm_safe_qvel
                ) / max(self.dt, 1e-6)
                feasible_qacc_high = (
                    arm_safe_interval_high - state.arm_safe_qvel
                ) / max(self.dt, 1e-6)
                arm_safe_qacc = jnp.clip(
                    governed_qacc,
                    feasible_qacc_low,
                    feasible_qacc_high,
                )
                arm_safe_qvel = state.arm_safe_qvel + arm_safe_qacc * self.dt
                arm_safe_q_ref_latest = (
                    state.arm_safe_q_ref_latest + arm_safe_qvel * self.dt
                )
            arm_safe_feasible = arm_safe_feasible & jnp.all(
                arm_safe_jerk_feasible, axis=-1
            )
        elif persistent_analytic_inverse_mode:
            (
                arm_safe_q_ref_latest,
                arm_safe_qvel,
                arm_safe_qacc,
                _compensation_delta_q,
                _compensation_delta_qvel,
            ) = policy_relative_compensation_step_jax(
                arm_actuator_q_ref_latest,
                comp_target_q,
                comp_target_qvel,
                comp_target_qdd,
                state.arm_cmd_q,
                state.arm_cmd_qvel,
                state.arm_safe_q_ref_latest,
                state.arm_safe_qvel,
                self.arm_lo,
                self.arm_hi,
                self.arm_vel_limit_rad_s,
                self.arm_acc_limit_rad_s2
                * float(self.cfg.actuator_compensation_acc_limit_scale),
                dt=self.dt,
                natural_frequency_hz=float(
                    self.cfg.actuator_compensation_governor_natural_frequency_hz
                ),
            )
            if racket_reference_governor_mode:
                compensation_qacc = arm_safe_qacc - comp_target_qdd
                racket_comp_acc = jnp.sum(
                    compensation_qacc
                    * self.racket_vertical_arm_jacobian[None, :],
                    axis=-1,
                )
                racket_acc_limit = max(
                    1e-6,
                    float(
                        self.cfg.actuator_racket_reference_governor_acc_limit_m_s2
                    ),
                )
                racket_scale = jnp.minimum(
                    1.0,
                    racket_acc_limit / jnp.maximum(jnp.abs(racket_comp_acc), 1e-6),
                )
                governed_qacc = (
                    comp_target_qdd + racket_scale[:, None] * compensation_qacc
                )
                governed_acc_limit = (
                    self.arm_acc_limit_rad_s2[None, :]
                    + np.deg2rad(
                        float(
                            self.cfg.actuator_compensation_acc_limit_margin_deg_s2
                        )
                    )
                )
                feasible_qacc_low = jnp.maximum(
                    -governed_acc_limit,
                    jnp.maximum(
                        (
                            -self.arm_vel_limit_rad_s[None, :]
                            - state.arm_safe_qvel
                        )
                        / self.dt,
                        (
                            self.arm_lo[None, :]
                            - state.arm_safe_q_ref_latest
                            - state.arm_safe_qvel * self.dt
                        )
                        / (self.dt * self.dt),
                    ),
                )
                feasible_qacc_high = jnp.minimum(
                    governed_acc_limit,
                    jnp.minimum(
                        (
                            self.arm_vel_limit_rad_s[None, :]
                            - state.arm_safe_qvel
                        )
                        / self.dt,
                        (
                            self.arm_hi[None, :]
                            - state.arm_safe_q_ref_latest
                            - state.arm_safe_qvel * self.dt
                        )
                        / (self.dt * self.dt),
                    ),
                )
                feasible = feasible_qacc_low <= feasible_qacc_high
                emergency_brake = jnp.clip(
                    -state.arm_safe_qvel / self.dt,
                    -governed_acc_limit,
                    governed_acc_limit,
                )
                arm_safe_qacc = jnp.where(
                    feasible,
                    jnp.clip(
                        governed_qacc, feasible_qacc_low, feasible_qacc_high
                    ),
                    emergency_brake,
                )
                arm_safe_qvel = state.arm_safe_qvel + arm_safe_qacc * self.dt
                arm_safe_q_ref_latest = (
                    state.arm_safe_q_ref_latest + arm_safe_qvel * self.dt
                )
            arm_safe_interval_low = -jnp.broadcast_to(
                self.arm_vel_limit_rad_s, arm_safe_qvel.shape
            )
            arm_safe_interval_high = jnp.broadcast_to(
                self.arm_vel_limit_rad_s, arm_safe_qvel.shape
            )
            arm_safe_feasible = jnp.ones((arm_safe_qvel.shape[0],), dtype=bool)
        elif bridger_mode:
            arm_safe_q_ref_latest = arm_actuator_q_ref_latest
            arm_safe_qvel = bridger_qvel
            arm_safe_qacc = bridger_qacc
            arm_safe_interval_low = bridger_interval_low
            arm_safe_interval_high = bridger_interval_high
            arm_safe_feasible = bridger_feasible & jnp.all(bridger_jerk_feasible, axis=-1)
        elif bool(self.cfg.arm_post_compensation_limiter) or regularized_inverse_mode:
            # The safe analytic inverse is a q-only deployment contract.  Its
            # final q must traverse the same publish-time position/velocity/
            # acceleration viability projection as the physical ROS path.
            # Making this mandatory for the mode prevents a future curriculum
            # override from reopening the post-integration high-bandwidth
            # shortcut that caused the double-frequency racket motion.
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
        if (
            not bridger_mode
            and not bandlimited_horizon_inverse_mode
            and not persistent_analytic_inverse_mode
        ):
            arm_safe_qacc = (
                arm_safe_qvel - state.arm_safe_qvel
            ) / max(self.dt, 1e-6)
        arm_safe_clip = jnp.abs(
            arm_actuator_q_ref_latest - arm_safe_q_ref_latest
        ) > 1e-7

        if planner_before_actuator:
            # Roll the fitted actuator state to the instant just before the
            # command selected now becomes active.  Only already queued
            # commands are used, so the prediction is causal.
            if bool(self.cfg.actuator_cmd_filter):
                actuator_tau = jnp.maximum(state.actuator_cmd_tau, 0.0)
                actuator_alpha = jnp.where(
                    actuator_tau <= 1e-6,
                    1.0,
                    self.dt / (actuator_tau + self.dt),
                )
                actuator_gain = state.actuator_cmd_gain
            else:
                actuator_alpha = jnp.ones_like(state.actuator_cmd_tau)
                actuator_gain = jnp.ones_like(state.actuator_cmd_gain)
            predicted_output = state.arm_applied_q
            predicted_velocity = state.arm_applied_qvel
            for s in range(self.command_buffer_len - 1):
                # ``actuator_command_buffer`` is shifted before the active
                # delayed element is selected below.  Therefore the command
                # applied on this step is old_buffer[L - delay_steps], not
                # old_buffer[L - 1 - delay_steps].  Starting one element too
                # early repeats the command from the preceding control step
                # and gives the arrival-state governor a one-step phase lag.
                idx = jnp.clip(
                    self.command_buffer_len - delay_steps + s,
                    0,
                    self.command_buffer_len - 1,
                )
                queued = state.actuator_command_buffer[
                    jnp.arange(state.actuator_command_buffer.shape[0]), idx
                ]
                queued_target = self.warm_arm_q[None, :] + actuator_gain[:, None] * (
                    queued - self.warm_arm_q[None, :]
                )
                predicted_next = predicted_output + actuator_alpha[:, None] * (
                    queued_target - predicted_output
                )
                predicted_next_velocity = (
                    predicted_next - predicted_output
                ) / max(self.dt, 1e-6)
                active_prediction_step = (s < delay_steps)[:, None]
                predicted_velocity = jnp.where(
                    active_prediction_step,
                    predicted_next_velocity,
                    predicted_velocity,
                )
                predicted_output = jnp.where(
                    active_prediction_step,
                    predicted_next,
                    predicted_output,
                )

            raw_target = self.warm_arm_q[None, :] + actuator_gain[:, None] * (
                arm_safe_q_ref_latest - self.warm_arm_q[None, :]
            )
            raw_future_output = predicted_output + actuator_alpha[:, None] * (
                raw_target - predicted_output
            )
            (
                desired_future_output,
                _,
                arm_servo_interval_low,
                arm_servo_interval_high,
                output_feasible,
            ) = project_target_tracking_command_step_jax(
                raw_future_output,
                predicted_output,
                predicted_velocity,
                self.arm_lo,
                self.arm_hi,
                servo_velocity_limit,
                servo_acceleration_limit,
                self.dt,
            )

            output_low = predicted_output + arm_servo_interval_low * self.dt
            output_high = predicted_output + arm_servo_interval_high * self.dt
            alpha_safe = jnp.maximum(actuator_alpha[:, None], 1e-6)
            gain_safe = jnp.where(
                jnp.abs(actuator_gain[:, None]) <= 1e-6,
                1.0,
                actuator_gain[:, None],
            )

            def command_for_output(output_q: jax.Array) -> jax.Array:
                filter_target_q = (
                    output_q
                    - (1.0 - actuator_alpha[:, None]) * predicted_output
                ) / alpha_safe
                return self.warm_arm_q[None, :] + (
                    filter_target_q - self.warm_arm_q[None, :]
                ) / gain_safe

            desired_command = command_for_output(desired_future_output)
            command_bound_a = command_for_output(output_low)
            command_bound_b = command_for_output(output_high)
            command_low = jnp.maximum(
                self.arm_lo[None, :],
                jnp.minimum(command_bound_a, command_bound_b),
            )
            command_high = jnp.minimum(
                self.arm_hi[None, :],
                jnp.maximum(command_bound_a, command_bound_b),
            )
            command_interval_feasible = command_low <= command_high + 5e-5
            command_mid = 0.5 * (command_low + command_high)
            safe_command_low = jnp.minimum(command_low, command_mid)
            safe_command_high = jnp.maximum(command_high, command_mid)
            actuator_input_q = jnp.minimum(
                jnp.maximum(desired_command, safe_command_low),
                safe_command_high,
            )
            achieved_target = self.warm_arm_q[None, :] + actuator_gain[:, None] * (
                actuator_input_q - self.warm_arm_q[None, :]
            )
            arm_servo_command_q = predicted_output + actuator_alpha[:, None] * (
                achieved_target - predicted_output
            )
            arm_servo_command_qvel = (
                arm_servo_command_q - predicted_output
            ) / max(self.dt, 1e-6)
            arm_servo_feasible = output_feasible & jnp.all(
                command_interval_feasible,
                axis=-1,
            )
            arm_servo_clip = jnp.abs(
                raw_future_output - arm_servo_command_q
            ) > 1e-7
        else:
            actuator_input_q = arm_safe_q_ref_latest

        if self.delay_conditioning:
            command_buffer = jnp.concatenate([state.command_buffer[:, 1:, :], arm_q_ref_latest[:, None, :]], axis=1)
            actuator_command_buffer = jnp.concatenate(
                [state.actuator_command_buffer[:, 1:, :], actuator_input_q[:, None, :]],
                axis=1,
            )
            active_idx = (command_buffer.shape[1] - 1 - delay_steps).astype(jnp.int32)
            arm_q_ref_active = command_buffer[jnp.arange(command_buffer.shape[0]), active_idx]
            arm_actuator_q_ref_active = actuator_command_buffer[jnp.arange(actuator_command_buffer.shape[0]), active_idx]
        else:
            command_buffer = state.command_buffer
            actuator_command_buffer = state.actuator_command_buffer
            arm_q_ref_active = arm_q_ref_latest
            arm_actuator_q_ref_active = actuator_input_q

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

        second_order_actuator = bool(
            self.cfg.actuator_cmd_filter
            and self.actuator_cmd_model in {"second_order", "parallel_second_order"}
        )
        parallel_second_order = bool(
            second_order_actuator and self.actuator_cmd_model == "parallel_second_order"
        )
        if (
            second_order_actuator
            and self.actuator_cmd_delay_steps_per_joint is not None
            and self.delay_conditioning
            and not self.actuator_delay_observation_only
        ):
            joint_delay_steps = jnp.clip(
                self.actuator_cmd_delay_steps_per_joint[None, :]
                + state.second_order_delay_offset_steps[:, None],
                0,
                actuator_command_buffer.shape[1] - 1,
            )
            joint_active_idx = (
                actuator_command_buffer.shape[1] - 1 - joint_delay_steps
            )
            arm_servo_q_ref_active = jnp.take_along_axis(
                actuator_command_buffer,
                joint_active_idx[:, None, :],
                axis=1,
            )[:, 0, :]
        second_order_qvel = state.arm_applied_qvel
        if second_order_actuator:
            wn = (
                self.actuator_cmd_second_order_wn[None, :]
                * state.second_order_frequency_scale[:, None]
            )
            zeta = jnp.clip(
                self.actuator_cmd_second_order_zeta[None, :]
                * state.second_order_damping_scale[:, None],
                0.03,
                0.99,
            )
            root = jnp.sqrt(jnp.maximum(1.0 - zeta * zeta, 1e-8))
            wd = wn * root
            decay = jnp.exp(-zeta * wn * self.dt)
            phase = wd * self.dt
            sin_phase = jnp.sin(phase)
            cos_phase = jnp.cos(phase)
            a11 = decay * (cos_phase + zeta / root * sin_phase)
            a12 = decay * sin_phase / jnp.maximum(wd, 1e-8)
            a21 = -decay * wn * wn * sin_phase / jnp.maximum(wd, 1e-8)
            a22 = decay * (cos_phase - zeta / root * sin_phase)
            gain_target_q = self.warm_arm_q[None, :] + (
                self.actuator_cmd_second_order_gain[None, :]
                * state.second_order_gain_scale[:, None]
            ) * (
                arm_servo_q_ref_active - self.warm_arm_q[None, :]
            )
            mode1_q = (
                a11 * state.arm_actuator_mode1_q
                + a12 * state.arm_actuator_mode1_qvel
                + (1.0 - a11) * gain_target_q
            )
            mode1_qvel = (
                a21 * state.arm_actuator_mode1_q
                + a22 * state.arm_actuator_mode1_qvel
                - a21 * gain_target_q
            )
            secondary_wn = (
                self.actuator_cmd_secondary_wn[None, :]
                * state.second_order_frequency_scale[:, None]
            )
            secondary_zeta = jnp.clip(
                self.actuator_cmd_secondary_zeta[None, :]
                * state.second_order_damping_scale[:, None],
                0.03,
                0.99,
            )
            secondary_root = jnp.sqrt(jnp.maximum(1.0 - secondary_zeta**2, 1e-8))
            secondary_wd = secondary_wn * secondary_root
            secondary_decay = jnp.exp(-secondary_zeta * secondary_wn * self.dt)
            secondary_phase = secondary_wd * self.dt
            secondary_sin = jnp.sin(secondary_phase)
            secondary_cos = jnp.cos(secondary_phase)
            b11 = secondary_decay * (
                secondary_cos + secondary_zeta / secondary_root * secondary_sin
            )
            b12 = secondary_decay * secondary_sin / jnp.maximum(secondary_wd, 1e-8)
            b21 = -secondary_decay * secondary_wn**2 * secondary_sin / jnp.maximum(
                secondary_wd, 1e-8
            )
            b22 = secondary_decay * (
                secondary_cos - secondary_zeta / secondary_root * secondary_sin
            )
            mode2_q = (
                b11 * state.arm_actuator_mode2_q
                + b12 * state.arm_actuator_mode2_qvel
                + (1.0 - b11) * gain_target_q
            )
            mode2_qvel = (
                b21 * state.arm_actuator_mode2_q
                + b22 * state.arm_actuator_mode2_qvel
                - b21 * gain_target_q
            )
            mix = jnp.where(
                parallel_second_order,
                self.actuator_cmd_secondary_mix[None, :],
                0.0,
            )
            arm_servo_target_unlimited = (1.0 - mix) * mode1_q + mix * mode2_q
            second_order_qvel = (1.0 - mix) * mode1_qvel + mix * mode2_qvel
            arm_servo_target_unlimited = jnp.clip(
                arm_servo_target_unlimited,
                self.arm_lo,
                self.arm_hi,
            )
        elif bool(self.cfg.actuator_cmd_filter):
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
        if planner_before_actuator:
            # The fitted actuator model is the final pre-PD block.  Do not
            # alter its output with another command planner.
            arm_applied_q = arm_servo_target_unlimited
            arm_applied_qvel = (
                arm_applied_q - state.arm_applied_q
            ) / max(self.dt, 1e-6)
        elif bool(self.cfg.arm_servo_target_tracking_planner):
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
            arm_applied_qvel = jnp.where(
                second_order_actuator,
                second_order_qvel,
                (arm_applied_q - state.arm_applied_q) / max(self.dt, 1e-6),
            )
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
        if self.recovered_rmp_motion_mode:
            # These are the actual final Plan outputs, not derivatives
            # reconstructed from the surrounding 200 Hz samples.
            arm_applied_q = low_level_rmp_q
            arm_safe_qvel = low_level_rmp_qvel
            arm_safe_qacc = low_level_rmp_qdd
            arm_applied_qvel = low_level_rmp_qvel
            arm_applied_qacc = low_level_rmp_qdd
        if not planner_before_actuator:
            arm_servo_command_q = arm_applied_q
            arm_servo_command_qvel = arm_applied_qvel
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
        if self.recovered_rmp_motion_mode:
            # MuJoCo's position actuator produces Kp*(ctrl-q)-Kv*qdot.  This
            # equivalent ctrl therefore consumes the RMP position and velocity
            # as separate low-level references, without inserting an actuator
            # lag/filter model.  The optional qdd coefficient represents only
            # low-level acceleration feedforward and defaults to the replay-
            # validated value of zero.
            controller_kp = (
                self.original_arm_actuator_kp[None, :]
                * state.dr_pd_kp_mult
            )
            controller_kv = (
                self.original_arm_actuator_kv[None, :]
                * state.dr_pd_kv_mult
            )
            velocity_equivalent_position = (
                controller_kv
                / jnp.maximum(controller_kp, 1.0e-6)
                * low_level_rmp_qvel
            )
            acceleration_equivalent_position = (
                self.recovered_rmp_acceleration_feedforward_s2[None, :]
                * low_level_rmp_qdd
            )
            arm_low_level_ctrl_q = jnp.clip(
                arm_applied_q
                + velocity_equivalent_position
                + acceleration_equivalent_position,
                self.arm_lo,
                self.arm_hi,
            )
            arm_low_level_ctrl_q_history = jnp.clip(
                low_level_rmp_q_history
                + (
                    controller_kv
                    / jnp.maximum(controller_kp, 1.0e-6)
                )[None, ...]
                * low_level_rmp_qvel_history
                + self.recovered_rmp_acceleration_feedforward_s2[None, None, :]
                * low_level_rmp_qdd_history,
                self.arm_lo,
                self.arm_hi,
            )
        else:
            arm_low_level_ctrl_q = arm_applied_q
            arm_low_level_ctrl_q_history = jnp.broadcast_to(
                arm_low_level_ctrl_q[None, ...],
                (int(self.cfg.frame_skip),) + arm_low_level_ctrl_q.shape,
            )
        ctrl = ctrl.at[:, self.arm_aids_j].set(arm_low_level_ctrl_q)
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

        def one_substep(substep_index, carry):
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
            if self.recovered_rmp_motion_mode:
                substep_ctrl = ctrl.at[:, self.arm_aids_j].set(
                    arm_low_level_ctrl_q_history[substep_index]
                )
            else:
                substep_ctrl = ctrl
            d = self.batched_step(state.model, d.replace(ctrl=substep_ctrl))
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
        current_time = step_count.astype(jnp.float32) * self.dt
        bpos = data.xpos[:, self.ball_body_id]
        rpos = data.site_xpos[:, self.racket_site_id]
        rmat = data.site_xmat[:, self.racket_site_id].reshape((-1, 3, 3))
        racket_normal = rmat[:, :, 2]
        bvel = (bpos - state.prev_ball_pos) / max(self.dt, 1e-6)
        rvel = (rpos - state.prev_racket_pos) / max(self.dt, 1e-6)
        racket_vertical_acc = (
            rvel[:, 2] - state.prev_racket_vel[:, 2]
        ) / max(self.dt, 1e-6)
        racket_angular_velocity = data.cvel[:, self.racket_body_id, :3]
        racket_angular_velocity_local = jnp.einsum(
            "nij,nj->ni",
            jnp.swapaxes(rmat, 1, 2),
            racket_angular_velocity,
        )
        racket_angular_speed = jnp.linalg.norm(racket_angular_velocity, axis=-1)
        racket_local_xz_angular_speed = jnp.linalg.norm(
            racket_angular_velocity_local[:, (0, 2)], axis=-1
        )
        racket_stability_angular_speed = (
            racket_local_xz_angular_speed
            if self.racket_stability_angular_speed_mode == "local_xz"
            else racket_angular_speed
        )
        racket_tilt_angular_speed = jnp.linalg.norm(
            jnp.cross(racket_angular_velocity, racket_normal), axis=-1
        )
        time_since_counted_hit = jnp.where(
            state.last_counted_hit_time >= 0.0,
            current_time - state.last_counted_hit_time,
            jnp.full_like(current_time, 1.0e6),
        )
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

        # Measure whether a completed juggling cycle closes in joint space.
        # This is self-referential behavior shaping: no teacher trajectory or
        # action target is involved.  The first two hits remain unconstrained
        # so acquisition and lateral recovery can establish a viable cycle.
        arm_q_now = data.qpos[:, self.arm_qadr]
        cycle_action_sum_acc = state.hit_cycle_action_sum + action
        cycle_action_steps_acc = state.hit_cycle_action_steps + 1
        cycle_q_min_acc = jnp.minimum(state.hit_cycle_arm_q_min, arm_q_now)
        cycle_q_max_acc = jnp.maximum(state.hit_cycle_arm_q_max, arm_q_now)
        cycle_eligible = launched_upward & (
            state.hit_count >= int(self.cfg.hit_cycle_min_previous_hits)
        )
        cycle_q_abs_delta = jnp.abs(arm_q_now - state.last_counted_hit_arm_q)
        cycle_q_excess = jnp.maximum(
            cycle_q_abs_delta - self.hit_cycle_q_deadband_rad[None, :],
            0.0,
        ) / self.hit_cycle_q_scale_rad[None, :]
        cycle_joint_weight_sum = jnp.maximum(
            jnp.sum(self.hit_cycle_joint_weights), 1e-6
        )
        hit_cycle_q_pen = jnp.sum(
            self.hit_cycle_joint_weights[None, :]
            * jnp.minimum(cycle_q_excess**2, 4.0),
            axis=-1,
        ) / cycle_joint_weight_sum
        cycle_action_mean = cycle_action_sum_acc / jnp.maximum(
            cycle_action_steps_acc[:, None].astype(jnp.float32), 1.0
        )
        cycle_action_dc_excess = jnp.maximum(
            jnp.abs(cycle_action_mean)
            - float(self.cfg.hit_cycle_action_dc_deadband),
            0.0,
        ) / max(1e-6, float(self.cfg.hit_cycle_action_dc_scale))
        hit_cycle_action_dc_pen = jnp.mean(
            jnp.minimum(cycle_action_dc_excess**2, 4.0), axis=-1
        )
        cycle_q_excursion = cycle_q_max_acc - cycle_q_min_acc
        cycle_q_excursion_excess = jnp.maximum(
            cycle_q_excursion - self.hit_cycle_q_excursion_deadband_rad[None, :],
            0.0,
        ) / self.hit_cycle_q_excursion_scale_rad[None, :]
        hit_cycle_q_excursion_pen = jnp.sum(
            self.hit_cycle_joint_weights[None, :]
            * jnp.minimum(cycle_q_excursion_excess**2, 4.0),
            axis=-1,
        ) / cycle_joint_weight_sum
        last_counted_hit_arm_q = jnp.where(
            launched_upward[:, None], arm_q_now, state.last_counted_hit_arm_q
        )
        hit_cycle_arm_q_min = jnp.where(
            launched_upward[:, None], arm_q_now, cycle_q_min_acc
        )
        hit_cycle_arm_q_max = jnp.where(
            launched_upward[:, None], arm_q_now, cycle_q_max_acc
        )
        hit_cycle_action_sum = jnp.where(
            launched_upward[:, None],
            jnp.zeros_like(cycle_action_sum_acc),
            cycle_action_sum_acc,
        )
        hit_cycle_action_steps = jnp.where(
            launched_upward,
            jnp.zeros_like(cycle_action_steps_acc),
            cycle_action_steps_acc,
        )

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
            racket_vertical_acc=racket_vertical_acc,
            racket_angular_speed=racket_stability_angular_speed,
            racket_full_angular_speed=racket_angular_speed,
            racket_stability_angular_speed=racket_stability_angular_speed,
            racket_tilt_angular_speed=racket_tilt_angular_speed,
            time_since_counted_hit=time_since_counted_hit,
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
            arm_cmd_q=arm_cmd_q,
            cmd_qvel=cmd_qvel,
            prev_arm_qvel=state.prev_arm_qvel,
            racket_anchor=state.racket_anchor,
            chest_target_offset=state.chest_target_offset,
            hit_cycle_eligible=cycle_eligible,
            hit_cycle_q_pen=hit_cycle_q_pen,
            hit_cycle_q_error_max_rad=jnp.max(cycle_q_abs_delta, axis=-1),
            hit_cycle_action_dc_pen=hit_cycle_action_dc_pen,
            hit_cycle_q_excursion_pen=hit_cycle_q_excursion_pen,
            hit_cycle_q_excursion_max_rad=jnp.max(cycle_q_excursion, axis=-1),
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
            rmp_q=recovered_rmp_state.command_q,
            rmp_qd=recovered_rmp_state.command_qd,
            rmp_qdd=recovered_rmp_state.command_qdd,
            rmp_internal_q=recovered_rmp_state.q,
            rmp_internal_qd=recovered_rmp_state.qd,
            rmp_internal_qdd=recovered_rmp_state.qdd,
            rmp_accepted_target=recovered_rmp_state.accepted_target,
            rmp_ticks_since_target_update=(
                recovered_rmp_state.ticks_since_target_update
            ),
            rmp_jnt_kp=state.rmp_jnt_kp,
            rmp_jnt_kd=state.rmp_jnt_kd,
            rmp_estimator_joint_process_var=state.rmp_estimator_joint_process_var,
            rmp_estimator_joint_measure_var=state.rmp_estimator_joint_measure_var,
            rmp_joint_velocity_feedforward=state.rmp_joint_velocity_feedforward,
            rmp_joint_acceleration_feedforward=(
                state.rmp_joint_acceleration_feedforward
            ),
            rmp_acceleration_weight=state.rmp_acceleration_weight,
            rmp_target_filter_length=state.rmp_target_filter_length,
            rmp_output_delay_steps=state.rmp_output_delay_steps,
            rmp_output_q_delay_buffer=next_rmp_q_delay_buffer,
            rmp_output_qd_delay_buffer=next_rmp_qd_delay_buffer,
            rmp_output_qdd_delay_buffer=next_rmp_qdd_delay_buffer,
            rmp_target_x=recovered_rmp_state.target_x,
            rmp_target_covariance=recovered_rmp_state.target_covariance,
            rmp_target_q_buffer=recovered_rmp_state.target_q_buffer,
            rmp_target_qd_buffer=recovered_rmp_state.target_qd_buffer,
            rmp_target_qdd_buffer=recovered_rmp_state.target_qdd_buffer,
            rmp_target_buffer_index=recovered_rmp_state.target_buffer_index,
            rmp_filter_q_buffer=recovered_rmp_state.output_q_buffer,
            rmp_filter_qd_buffer=recovered_rmp_state.output_qd_buffer,
            rmp_filter_qdd_buffer=recovered_rmp_state.output_qdd_buffer,
            rmp_filter_buffer_index=recovered_rmp_state.output_buffer_index,
            arm_q_ref_latest=arm_q_ref_latest,
            arm_q_ref_active=arm_q_ref_active,
            arm_actuator_q_ref_latest=arm_actuator_q_ref_latest,
            arm_actuator_q_ref_active=arm_actuator_q_ref_active,
            arm_safe_q_ref_latest=arm_safe_q_ref_latest,
            arm_safe_qvel=arm_safe_qvel,
            arm_safe_qacc=arm_safe_qacc,
            compensation_prev_qdd=comp_target_qdd,
            compensation_filtered_qdd=compensation_filtered_qdd,
            compensation_filtered_qdd_stage2=compensation_filtered_qdd_stage2,
            compensation_filtered_jerk=compensation_filtered_jerk,
            compensation_smith_residual=smith_residual,
            compensation_dob_residual=dob_residual,
            arm_servo_command_q=arm_servo_command_q,
            arm_servo_command_qvel=arm_servo_command_qvel,
            ball_obs_missing_episode_coherent_enabled=state.ball_obs_missing_episode_coherent_enabled,
            ball_obs_camera_missing_enabled=state.ball_obs_camera_missing_enabled,
            ball_obs_view_bounds_missing_enabled=state.ball_obs_view_bounds_missing_enabled,
            arm_applied_q=arm_applied_q,
            arm_applied_qvel=arm_applied_qvel,
            arm_actuator_mode1_q=(mode1_q if second_order_actuator else arm_applied_q),
            arm_actuator_mode1_qvel=(mode1_qvel if second_order_actuator else arm_applied_qvel),
            arm_actuator_mode2_q=(mode2_q if second_order_actuator else arm_applied_q),
            arm_actuator_mode2_qvel=(mode2_qvel if second_order_actuator else arm_applied_qvel),
            prev_action=action,
            prev_arm_qvel=arm_qvel,
            prev_ball_pos=bpos,
            prev_racket_pos=rpos,
            prev_racket_vel=rvel,
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
            last_counted_hit_arm_q=last_counted_hit_arm_q,
            hit_cycle_arm_q_min=hit_cycle_arm_q_min,
            hit_cycle_arm_q_max=hit_cycle_arm_q_max,
            hit_cycle_action_sum=hit_cycle_action_sum,
            hit_cycle_action_steps=hit_cycle_action_steps,
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
            second_order_frequency_scale=state.second_order_frequency_scale,
            second_order_damping_scale=state.second_order_damping_scale,
            second_order_gain_scale=state.second_order_gain_scale,
            second_order_delay_offset_steps=state.second_order_delay_offset_steps,
            dr_gravity_z=state.dr_gravity_z,
            dr_ball_mass=state.dr_ball_mass,
            dr_ball_friction=state.dr_ball_friction,
            dr_racket_friction=state.dr_racket_friction,
            dr_ball_solref_time=state.dr_ball_solref_time,
            dr_ball_solref_damping=state.dr_ball_solref_damping,
            dr_hard_tail_active=state.dr_hard_tail_active,
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
        # Miss-cause diagnostics.  A terminal reason such as ``ball_too_low``
        # does not say whether the preceding return was physically awkward,
        # the racket arrived late, or the observed trajectory was wrong.  Log
        # the true and observation-derived descending intersections so the
        # evaluator can separate those cases over the complete pre-miss arc.
        descending_to_racket = (bvel[:, 2] < -1e-4) & (bpos[:, 2] > rpos[:, 2])
        raw_intercept_time_true = (
            (bpos[:, 2] - rpos[:, 2]) / jnp.maximum(-bvel[:, 2], 1e-4)
        )
        intercept_time_true = jnp.where(
            descending_to_racket,
            raw_intercept_time_true,
            0.0,
        )
        intercept_actionable = descending_to_racket & (
            raw_intercept_time_true <= float(self.cfg.pre_hit_intercept_time_max)
        )
        projected_intercept_xy = bpos[:, :2] + bvel[:, :2] * intercept_time_true[:, None]
        intercept_racket_xy_err = jnp.linalg.norm(projected_intercept_xy - rpos[:, :2], axis=-1)
        intercept_anchor_xy_dist = jnp.linalg.norm(
            projected_intercept_xy - state.racket_anchor[:, :2], axis=-1
        )
        intercept_required_racket_speed = intercept_racket_xy_err / jnp.maximum(
            intercept_time_true, self.dt
        )
        intercept_direction = (
            projected_intercept_xy - rpos[:, :2]
        ) / jnp.maximum(intercept_racket_xy_err[:, None], 1e-6)
        intercept_racket_closing_speed = jnp.sum(rvel[:, :2] * intercept_direction, axis=-1)

        observed_bpos = next_state.ball_obs_valid_pos
        observed_bvel = next_state.ball_obs_valid_vel
        observed_descending = (observed_bvel[:, 2] < -1e-4) & (
            observed_bpos[:, 2] > rpos[:, 2]
        )
        intercept_time_observed = jnp.where(
            observed_descending,
            (observed_bpos[:, 2] - rpos[:, 2])
            / jnp.maximum(-observed_bvel[:, 2], 1e-4),
            0.0,
        )
        observed_projected_intercept_xy = (
            observed_bpos[:, :2]
            + observed_bvel[:, :2] * intercept_time_observed[:, None]
        )
        observed_intercept_prediction_error = jnp.linalg.norm(
            observed_projected_intercept_xy - projected_intercept_xy, axis=-1
        )
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
            "dr_hard_tail_active": state.dr_hard_tail_active.astype(jnp.float32),
            "dr_damping_mult": state.dr_damping_mult,
            "dr_armature_mult": state.dr_armature_mult,
            "dr_pd_kp_mult_mean": jnp.mean(state.dr_pd_kp_mult, axis=-1),
            "dr_pd_kp_mult_min": jnp.min(state.dr_pd_kp_mult, axis=-1),
            "dr_pd_kp_mult_max": jnp.max(state.dr_pd_kp_mult, axis=-1),
            "dr_pd_kv_mult_mean": jnp.mean(state.dr_pd_kv_mult, axis=-1),
            "dr_pd_kv_mult_min": jnp.min(state.dr_pd_kv_mult, axis=-1),
            "dr_pd_kv_mult_max": jnp.max(state.dr_pd_kv_mult, axis=-1),
            "recovered_rmp_motion_mode": jnp.full(
                (action.shape[0],),
                float(self.recovered_rmp_motion_mode),
                dtype=jnp.float32,
            ),
            "rmp_target_error_norm": jnp.linalg.norm(
                arm_cmd_q - rmp_command_q, axis=-1
            ),
            "rmp_qd_norm": jnp.linalg.norm(rmp_command_qvel, axis=-1),
            "rmp_qdd_norm": jnp.linalg.norm(rmp_command_qdd, axis=-1),
            "rmp_low_level_velocity_offset_norm": jnp.linalg.norm(
                arm_low_level_ctrl_q - arm_applied_q, axis=-1
            ),
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
                (
                    raw_future_output - arm_servo_command_q
                    if planner_before_actuator
                    else arm_servo_target_unlimited - arm_applied_q
                ),
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
            "racket_x": rpos[:, 0],
            "racket_y": rpos[:, 1],
            "racket_vx": rvel[:, 0],
            "racket_vy": rvel[:, 1],
            "racket_angular_velocity_world_x": racket_angular_velocity[:, 0],
            "racket_angular_velocity_world_y": racket_angular_velocity[:, 1],
            "racket_angular_velocity_world_z": racket_angular_velocity[:, 2],
            "racket_angular_velocity_local_x": racket_angular_velocity_local[:, 0],
            "racket_angular_velocity_local_y": racket_angular_velocity_local[:, 1],
            "racket_angular_velocity_local_z": racket_angular_velocity_local[:, 2],
            "racket_tilt_angular_speed_rad_s": racket_tilt_angular_speed,
            "racket_local_xz_angular_speed_rad_s": racket_local_xz_angular_speed,
            "racket_stability_angular_speed_rad_s": racket_stability_angular_speed,
            "racket_spin_angular_speed_rad_s": jnp.abs(
                racket_angular_velocity_local[:, 2]
            ),
            "ball_vx": bvel[:, 0],
            "ball_vy": bvel[:, 1],
            "ball_vz": bvel[:, 2],
            "descending_intercept_active": descending_to_racket.astype(jnp.float32),
            "intercept_actionable_active": intercept_actionable.astype(jnp.float32),
            "intercept_time_true": intercept_time_true,
            "projected_intercept_x": projected_intercept_xy[:, 0],
            "projected_intercept_y": projected_intercept_xy[:, 1],
            "intercept_racket_xy_err": intercept_racket_xy_err,
            "intercept_anchor_xy_dist": intercept_anchor_xy_dist,
            "intercept_required_racket_speed": intercept_required_racket_speed,
            "intercept_racket_closing_speed": intercept_racket_closing_speed,
            "ball_obs_pos_error_norm": jnp.linalg.norm(observed_bpos - bpos, axis=-1),
            "ball_obs_vel_error_norm": jnp.linalg.norm(observed_bvel - bvel, axis=-1),
            "intercept_time_observed": intercept_time_observed,
            "observed_intercept_prediction_error": observed_intercept_prediction_error,
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
        racket_vertical_acc: jax.Array,
        racket_angular_speed: jax.Array,
        racket_full_angular_speed: jax.Array,
        racket_stability_angular_speed: jax.Array,
        racket_tilt_angular_speed: jax.Array,
        time_since_counted_hit: jax.Array,
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
        arm_cmd_q: jax.Array,
        cmd_qvel: jax.Array,
        prev_arm_qvel: jax.Array,
        racket_anchor: jax.Array,
        chest_target_offset: jax.Array,
        hit_cycle_eligible: jax.Array,
        hit_cycle_q_pen: jax.Array,
        hit_cycle_q_error_max_rad: jax.Array,
        hit_cycle_action_dc_pen: jax.Array,
        hit_cycle_q_excursion_pen: jax.Array,
        hit_cycle_q_excursion_max_rad: jax.Array,
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
        arm_q = data.qpos[:, self.arm_qadr]
        arm_posture_error = arm_q - self.warm_arm_q
        arm_command_posture_error = arm_cmd_q - self.warm_arm_q
        arm_posture_weight_sum = jnp.maximum(
            jnp.sum(self.arm_posture_joint_weights), 1e-6
        )
        arm_posture_pen = jnp.sum(
            self.arm_posture_joint_weights * arm_posture_error**2, axis=-1
        ) / arm_posture_weight_sum
        arm_command_posture_pen = jnp.sum(
            self.arm_posture_joint_weights * arm_command_posture_error**2, axis=-1
        ) / arm_posture_weight_sum
        # Phase-conditioned teacher guidance is deliberately based on the
        # student's realized motion.  Old-domain actions are not targets: the
        # delayed/underdamped actuator remains free to learn different commands
        # that realize the same bounded periodic behavior.
        teacher_phase = jnp.clip(
            0.5
            - 0.5
            * bvel[:, 2]
            / max(1e-6, float(self.cfg.phase_teacher_ball_vz_scale_m_s)),
            0.0,
            1.0,
        )
        teacher_table_pos = teacher_phase * float(self.phase_teacher_q.shape[0] - 1)
        teacher_lo = jnp.floor(teacher_table_pos).astype(jnp.int32)
        teacher_hi = jnp.minimum(teacher_lo + 1, self.phase_teacher_q.shape[0] - 1)
        teacher_mix = teacher_table_pos - teacher_lo.astype(jnp.float32)
        teacher_q_target = (
            self.phase_teacher_q[teacher_lo] * (1.0 - teacher_mix[:, None])
            + self.phase_teacher_q[teacher_hi] * teacher_mix[:, None]
        )
        teacher_dq_target = (
            self.phase_teacher_dq[teacher_lo] * (1.0 - teacher_mix[:, None])
            + self.phase_teacher_dq[teacher_hi] * teacher_mix[:, None]
        )
        teacher_racket_z_target = (
            self.phase_teacher_racket_z[teacher_lo] * (1.0 - teacher_mix)
            + self.phase_teacher_racket_z[teacher_hi] * teacher_mix
        )
        teacher_racket_vz_target = (
            self.phase_teacher_racket_vz[teacher_lo] * (1.0 - teacher_mix)
            + self.phase_teacher_racket_vz[teacher_hi] * teacher_mix
        )
        teacher_joint_weight_sum = jnp.maximum(
            jnp.sum(self.phase_teacher_joint_weights), 1e-6
        )
        teacher_q_norm_error = (arm_q - teacher_q_target) / self.phase_teacher_q_sigma
        teacher_dq_norm_error = (
            data.qvel[:, self.arm_vadr] - teacher_dq_target
        ) / self.phase_teacher_dq_sigma
        teacher_q_pen = jnp.sum(
            self.phase_teacher_joint_weights
            * jnp.minimum(teacher_q_norm_error**2, 4.0),
            axis=-1,
        ) / teacher_joint_weight_sum
        teacher_dq_pen = jnp.sum(
            self.phase_teacher_joint_weights
            * jnp.minimum(teacher_dq_norm_error**2, 4.0),
            axis=-1,
        ) / teacher_joint_weight_sum
        teacher_racket_z_pen = jnp.minimum(
            ((rpos[:, 2] - racket_anchor[:, 2] - teacher_racket_z_target) / 0.08) ** 2,
            4.0,
        )
        teacher_racket_vz_pen = jnp.minimum(
            ((rvel[:, 2] - teacher_racket_vz_target) / 1.0) ** 2,
            4.0,
        )
        teacher_active = (
            hit_count >= int(self.cfg.phase_teacher_activate_after_hits)
        ).astype(jnp.float32)
        teacher_strength = (
            float(self.cfg.phase_teacher_strength)
            if self.phase_teacher_enabled
            else 0.0
        )
        term_phase_teacher_q_penalty = (
            -teacher_strength
            * float(self.cfg.phase_teacher_q_weight)
            * teacher_active
            * teacher_q_pen
        )
        term_phase_teacher_dq_penalty = (
            -teacher_strength
            * float(self.cfg.phase_teacher_dq_weight)
            * teacher_active
            * teacher_dq_pen
        )
        term_phase_teacher_racket_z_penalty = (
            -teacher_strength
            * float(self.cfg.phase_teacher_racket_z_weight)
            * teacher_active
            * teacher_racket_z_pen
        )
        term_phase_teacher_racket_vz_penalty = (
            -teacher_strength
            * float(self.cfg.phase_teacher_racket_vz_weight)
            * teacher_active
            * teacher_racket_vz_pen
        )
        arm_posture_soft_excess = jnp.maximum(
            jnp.abs(arm_posture_error) / self.arm_posture_soft_limit_rad - 1.0,
            0.0,
        )
        arm_command_posture_soft_excess = jnp.maximum(
            jnp.abs(arm_command_posture_error) / self.arm_posture_soft_limit_rad - 1.0,
            0.0,
        )
        arm_posture_soft_limit_pen = jnp.sum(
            self.arm_posture_joint_weights * arm_posture_soft_excess**2, axis=-1
        ) / arm_posture_weight_sum
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
        recoverability_active = hit_count >= int(self.cfg.hit_recoverability_min_count)
        post_hit_survival_reward = jnp.where(
            (hit_count > 0) & (bpos[:, 2] >= racket_anchor[:, 2] - 0.02),
            post_hit_ball_xy_score
            - jnp.where(
                recoverability_active,
                float(self.cfg.post_hit_ball_vxy_penalty_weight) * ball_vxy_pen,
                0.0,
            ),
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
        descending_intercept_excess = jnp.maximum(
            0.0,
            descending_intercept_xy_err
            - float(self.cfg.descending_intercept_excess_radius),
        )
        descending_intercept_excess_penalty = jnp.where(
            (hit_count > 0)
            & (bvel[:, 2] < -1e-4)
            & (bpos[:, 2] > rpos[:, 2])
            & (time_to_racket >= 0.0)
            & (
                time_to_racket
                <= float(self.cfg.descending_intercept_excess_time_max)
            ),
            (
                descending_intercept_excess
                / max(
                    1e-6,
                    float(self.cfg.descending_intercept_excess_sigma),
                )
            )
            ** 2,
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
        arm_velocity_usage_pen = jnp.mean(arm_vel_ratio**2, axis=-1)
        arm_acceleration_usage_pen = jnp.mean(arm_acc_ratio**2, axis=-1)

        term_ball_height = 1.2 * ball_height_reward
        term_rel_height = float(self.cfg.rel_height_bonus_weight) * rel_height_bonus
        term_xy_track_penalty = -1.4 * xy_track_pen
        term_racket_center_penalty = -0.35 * racket_center_pen
        term_posture_penalty = -float(self.cfg.posture_weight) * posture_pen
        term_arm_posture_penalty = (
            -float(self.cfg.arm_posture_penalty_weight) * arm_posture_pen
        )
        term_arm_command_posture_penalty = (
            -float(self.cfg.arm_command_posture_penalty_weight)
            * arm_command_posture_pen
        )
        term_arm_posture_soft_limit_penalty = (
            -float(self.cfg.arm_posture_soft_limit_penalty_weight)
            * arm_posture_soft_limit_pen
        )
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
        term_descending_intercept_excess_penalty = (
            -float(self.cfg.descending_intercept_excess_penalty_weight)
            * descending_intercept_excess_penalty
        )
        term_pre_hit_intercept = float(self.cfg.pre_hit_intercept_reward_weight) * pre_hit_intercept_reward
        term_pre_hit_intercept_penalty = (
            -float(self.cfg.pre_hit_intercept_penalty_weight) * pre_hit_intercept_penalty
        )
        term_racket_xy_reward = float(self.cfg.racket_xy_gauss_reward_weight) * racket_xy_gauss
        term_racket_xy_penalty = -float(self.cfg.racket_xy_gauss_penalty_weight) * racket_xy_gauss_pen
        term_racket_z_penalty = -float(self.cfg.racket_z_soft_penalty_weight) * racket_z_band_pen
        term_racket_up_drift_penalty = -float(self.cfg.racket_up_drift_penalty_weight) * up_drift_pen
        racket_vertical_acc_scale = max(
            1e-6, float(self.cfg.racket_vertical_acc_scale_m_s2)
        )
        racket_vertical_acc_pen = jnp.minimum(
            (racket_vertical_acc / racket_vertical_acc_scale) ** 2,
            4.0,
        )
        term_racket_vertical_acc_penalty = (
            -float(self.cfg.racket_vertical_acc_penalty_weight)
            * racket_vertical_acc_pen
        )
        racket_tilt_angular_speed_excess = jnp.maximum(
            0.0,
            racket_tilt_angular_speed
            - float(self.cfg.racket_tilt_angular_speed_soft_limit_rad_s),
        ) / max(
            1e-6,
            float(self.cfg.racket_tilt_angular_speed_scale_rad_s),
        )
        racket_tilt_angular_speed_pen = jnp.minimum(
            racket_tilt_angular_speed_excess**2,
            4.0,
        )
        term_racket_tilt_angular_speed_penalty = (
            -float(self.cfg.racket_tilt_angular_speed_penalty_weight)
            * racket_tilt_angular_speed_pen
        )
        racket_stability_angular_speed_excess = jnp.maximum(
            0.0,
            racket_stability_angular_speed
            - float(self.cfg.racket_stability_angular_speed_soft_limit_rad_s),
        ) / max(
            1e-6,
            float(self.cfg.racket_stability_angular_speed_scale_rad_s),
        )
        racket_stability_angular_speed_pen = jnp.minimum(
            racket_stability_angular_speed_excess**2,
            4.0,
        )
        term_racket_stability_angular_speed_penalty = (
            -float(self.cfg.racket_stability_angular_speed_penalty_weight)
            * racket_stability_angular_speed_pen
        )
        racket_up_cos = jnp.maximum(0.0, jnp.sum(racket_normal * jnp.asarray([0.0, 0.0, 1.0]), axis=-1))
        racket_flatness_err = jnp.maximum(0.0, float(self.cfg.racket_flatness_target_cos) - racket_up_cos)
        racket_flatness_pen = (
            racket_flatness_err / max(1e-6, float(self.cfg.racket_flatness_sigma))
        ) ** 2
        term_racket_flatness_penalty = (
            -float(self.cfg.racket_flatness_penalty_weight) * racket_flatness_pen
        )
        # Keep the sparse impact objective on the same angular axes as the
        # dense stability objective and curriculum gate.  ``full_norm``
        # preserves the historical behavior; ``local_xz`` deliberately leaves
        # the useful local-y juggling stroke out of the impact penalty.
        hit_racket_angular_speed = racket_stability_angular_speed
        angular_speed_excess = jnp.maximum(
            0.0,
            hit_racket_angular_speed
            - float(self.cfg.hit_racket_angular_speed_soft_limit_rad_s),
        ) / max(1e-6, float(self.cfg.hit_racket_angular_speed_scale_rad_s))
        hit_racket_angular_speed_pen = jnp.minimum(
            angular_speed_excess * angular_speed_excess, 4.0
        )
        post_hit_retreat_active = (
            (hit_count > 0)
            & (~new_hit)
            & (time_since_counted_hit >= 0.0)
            & (
                time_since_counted_hit
                <= float(self.cfg.post_hit_racket_retreat_window_s)
            )
        )
        retreat_depth_excess = jnp.maximum(
            0.0,
            -racket_z_rel - float(self.cfg.post_hit_racket_retreat_deadband_m),
        ) / max(1e-6, float(self.cfg.post_hit_racket_retreat_scale_m))
        downward_speed_excess = jnp.maximum(
            0.0,
            -rvel[:, 2]
            - float(self.cfg.post_hit_racket_downward_speed_soft_limit_m_s),
        ) / max(
            1e-6,
            float(self.cfg.post_hit_racket_downward_speed_scale_m_s),
        )
        post_hit_racket_retreat_pen = jnp.where(
            post_hit_retreat_active,
            jnp.minimum(
                retreat_depth_excess**2 + 0.25 * downward_speed_excess**2,
                4.0,
            ),
            0.0,
        )
        term_post_hit_racket_retreat_penalty = (
            -float(self.cfg.post_hit_racket_retreat_penalty_weight)
            * post_hit_racket_retreat_pen
        )
        flatness_err = jnp.maximum(0.0, float(self.cfg.hit_flatness_target_cos) - racket_up_cos)
        flatness_score = jnp.exp(-0.5 * (flatness_err / max(1e-6, float(self.cfg.hit_flatness_sigma))) ** 2)
        hit_flatness_excess_pen = jnp.minimum(
            (
                flatness_err
                / max(1e-6, float(self.cfg.hit_flatness_sigma))
            )
            ** 2,
            4.0,
        )
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
        term_arm_velocity_usage_penalty = (
            -float(self.cfg.arm_velocity_usage_penalty_weight)
            * arm_velocity_usage_pen
        )
        term_arm_acceleration_usage_penalty = (
            -float(self.cfg.arm_acceleration_usage_penalty_weight)
            * arm_acceleration_usage_pen
        )

        dense_reward = (
            term_ball_height
            + term_rel_height
            + term_xy_track_penalty
            + term_racket_center_penalty
            + term_posture_penalty
            + term_arm_posture_penalty
            + term_arm_command_posture_penalty
            + term_arm_posture_soft_limit_penalty
            + term_phase_teacher_q_penalty
            + term_phase_teacher_dq_penalty
            + term_phase_teacher_racket_z_penalty
            + term_phase_teacher_racket_vz_penalty
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
            + term_descending_intercept_excess_penalty
            + term_pre_hit_intercept
            + term_pre_hit_intercept_penalty
            + term_racket_xy_reward
            + term_racket_xy_penalty
            + term_racket_z_penalty
            + term_racket_up_drift_penalty
            + term_racket_vertical_acc_penalty
            + term_racket_tilt_angular_speed_penalty
            + term_racket_stability_angular_speed_penalty
            + term_racket_flatness_penalty
            + term_post_hit_racket_retreat_penalty
            + term_contact_flatness_penalty
            + camera_terms["camera_reward_dense"]
            + term_action_penalty
            + term_action_delta_penalty
            + term_action_clip_excess_penalty
            + term_arm_vel_penalty
            + term_arm_acc_penalty
            + term_arm_limiter_penalty
            + term_arm_velocity_usage_penalty
            + term_arm_acceleration_usage_penalty
        )
        reward = dense_reward * self.dt

        contact_center_dist = jnp.linalg.norm(rel_local[:, :2], axis=-1)
        center_gain = jnp.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_sigma))) ** 2)
        local_center_gain = jnp.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_local_sigma))) ** 2)
        hit_count_credit = float(self.cfg.hit_reward_combo) * jnp.minimum(
            hit_count.astype(jnp.float32),
            float(self.cfg.hit_combo_count_cap),
        )
        hit_quality = jnp.maximum(0.2, center_gain * flatness_score)
        hit_bonus = jnp.where(
            bool(self.cfg.hit_combo_quality_independent),
            float(self.cfg.hit_reward_base) * hit_quality + hit_count_credit,
            (float(self.cfg.hit_reward_base) + hit_count_credit) * hit_quality,
        )
        hit_height_err = jnp.abs(predicted_apex_z - target_hit_apex_z)
        hit_height_excess = jnp.maximum(0.0, hit_height_err - float(self.cfg.hit_height_tolerance))
        hit_height_pen = float(self.cfg.hit_height_penalty_weight) * hit_height_excess * hit_height_excess
        hit_vxy = jnp.linalg.norm(bvel[:, :2], axis=-1)
        hit_vxy_excess = jnp.maximum(0.0, hit_vxy - float(self.cfg.hit_vxy_soft_limit_m_s))
        hit_vxy_pen = float(self.cfg.hit_vxy_penalty_weight) * hit_vxy_excess * hit_vxy_excess
        hit_racket_vxy = jnp.linalg.norm(rvel[:, :2], axis=-1)
        hit_racket_vxy_excess = jnp.maximum(
            0.0,
            hit_racket_vxy - float(self.cfg.hit_racket_vxy_soft_limit_m_s),
        )
        hit_racket_vxy_pen = (
            float(self.cfg.hit_racket_vxy_penalty_weight)
            * hit_racket_vxy_excess
            * hit_racket_vxy_excess
        )
        low_hit_deficit = jnp.maximum(0.0, (target_ball_z - float(self.cfg.low_hit_apex_margin)) - predicted_apex_z)
        low_hit_pen = float(self.cfg.low_hit_penalty_weight) * low_hit_deficit * low_hit_deficit
        first_hit_apex_err = (predicted_apex_z - target_hit_apex_z) / max(
            1e-6,
            float(self.cfg.first_hit_apex_sigma),
        )
        first_hit_apex_score = jnp.exp(-0.5 * first_hit_apex_err * first_hit_apex_err)
        center_flat = float(self.cfg.center_flat_hit_reward_weight) * local_center_gain * flatness_score
        hit_contact_center_excess = jnp.maximum(
            0.0,
            contact_center_dist
            - float(self.cfg.hit_contact_center_excess_radius_m),
        )
        hit_contact_center_excess_pen = (
            float(self.cfg.hit_contact_center_excess_penalty_weight)
            * (
                hit_contact_center_excess
                / max(
                    1e-6,
                    float(self.cfg.hit_contact_center_excess_sigma_m),
                )
            )
            ** 2
        )
        height_bonus = jnp.where(
            predicted_apex_z >= target_ball_z,
            0.35 * jnp.exp(-10.0 * (predicted_apex_z - target_ball_z) * (predicted_apex_z - target_ball_z)),
            0.0,
        )
        hit_reward_mask = new_hit & rewardable_hit
        first_hit_reward_mask = hit_reward_mask & (hit_count <= 1)
        recoverability_hit_reward_mask = hit_reward_mask & recoverability_active
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
        term_hit_flatness_excess_penalty = jnp.where(
            hit_reward_mask,
            -float(self.cfg.hit_flatness_excess_penalty_weight)
            * hit_flatness_excess_pen,
            0.0,
        )
        term_hit_contact_center_excess_penalty = jnp.where(
            recoverability_hit_reward_mask,
            -hit_contact_center_excess_pen,
            0.0,
        )
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
        term_hit_vxy_penalty = jnp.where(recoverability_hit_reward_mask, -hit_vxy_pen, 0.0)
        term_hit_racket_vxy_penalty = jnp.where(
            recoverability_hit_reward_mask,
            -hit_racket_vxy_pen,
            0.0,
        )
        term_hit_racket_angular_speed_penalty = jnp.where(
            hit_reward_mask,
            -float(self.cfg.hit_racket_angular_speed_penalty_weight)
            * hit_racket_angular_speed_pen,
            0.0,
        )
        term_hit_apex_view_center_penalty = jnp.where(
            hit_reward_mask,
            -float(self.cfg.hit_apex_view_center_penalty_weight) * hit_apex_view_center_pen,
            0.0,
        )
        term_hit_next_contact_anchor_penalty = jnp.where(
            recoverability_hit_reward_mask,
            -float(self.cfg.hit_next_contact_anchor_penalty_weight) * hit_next_contact_anchor_pen,
            0.0,
        )
        term_low_hit_penalty = jnp.where(hit_reward_mask, -low_hit_pen, 0.0)
        term_failed_hit_penalty = jnp.where(failed_hit, -float(self.cfg.failed_hit_penalty_weight), 0.0)
        term_fast_hit_penalty = jnp.where(ignored_fast_hit, -fast_hit_penalty, 0.0)
        term_hit_cycle_q_closure_penalty = jnp.where(
            hit_cycle_eligible,
            -float(self.cfg.hit_cycle_q_closure_penalty_weight)
            * hit_cycle_q_pen,
            0.0,
        )
        term_hit_cycle_action_dc_penalty = jnp.where(
            hit_cycle_eligible,
            -float(self.cfg.hit_cycle_action_dc_penalty_weight)
            * hit_cycle_action_dc_pen,
            0.0,
        )
        term_hit_cycle_q_excursion_penalty = jnp.where(
            hit_cycle_eligible,
            -float(self.cfg.hit_cycle_q_excursion_penalty_weight)
            * hit_cycle_q_excursion_pen,
            0.0,
        )
        reward = (
            reward
            + term_hit_bonus
            + term_center_flat_hit
            + term_hit_flatness_excess_penalty
            + term_hit_contact_center_excess_penalty
            + term_hit_height_bonus
            + term_hit_camera
            + term_first_hit_apex
            + term_hit_cadence_reward
            + term_hit_min_interval_penalty
            + term_hit_height_penalty
            + term_hit_vxy_penalty
            + term_hit_racket_vxy_penalty
            + term_hit_racket_angular_speed_penalty
            + term_hit_apex_view_center_penalty
            + term_hit_next_contact_anchor_penalty
            + term_low_hit_penalty
            + term_failed_hit_penalty
            + term_fast_hit_penalty
            + term_hit_cycle_q_closure_penalty
            + term_hit_cycle_action_dc_penalty
            + term_hit_cycle_q_excursion_penalty
        )
        terms = {
            "total": reward,
            "dense_scaled": dense_reward * self.dt,
            "ball_height": term_ball_height * self.dt,
            "rel_height": term_rel_height * self.dt,
            "xy_track_penalty": term_xy_track_penalty * self.dt,
            "racket_center_penalty": term_racket_center_penalty * self.dt,
            "posture_penalty": term_posture_penalty * self.dt,
            "arm_posture_penalty": term_arm_posture_penalty * self.dt,
            "arm_command_posture_penalty": term_arm_command_posture_penalty * self.dt,
            "arm_posture_soft_limit_penalty": term_arm_posture_soft_limit_penalty * self.dt,
            "phase_teacher_q_penalty": term_phase_teacher_q_penalty * self.dt,
            "phase_teacher_dq_penalty": term_phase_teacher_dq_penalty * self.dt,
            "phase_teacher_racket_z_penalty": term_phase_teacher_racket_z_penalty * self.dt,
            "phase_teacher_racket_vz_penalty": term_phase_teacher_racket_vz_penalty * self.dt,
            "metric/phase_teacher_active": teacher_active,
            "metric/phase_teacher_phase": teacher_phase,
            "metric/phase_teacher_q_norm_error": jnp.sqrt(teacher_q_pen),
            "metric/phase_teacher_dq_norm_error": jnp.sqrt(teacher_dq_pen),
            "metric/phase_teacher_racket_z_error_m": jnp.abs(
                rpos[:, 2] - racket_anchor[:, 2] - teacher_racket_z_target
            ),
            "metric/phase_teacher_racket_vz_error_m_s": jnp.abs(
                rvel[:, 2] - teacher_racket_vz_target
            ),
            "metric/arm_posture_error_max_rad": jnp.max(
                jnp.abs(arm_posture_error), axis=-1
            ),
            "metric/arm_command_posture_error_max_rad": jnp.max(
                jnp.abs(arm_command_posture_error), axis=-1
            ),
            "metric/arm_posture_soft_exceed_fraction": jnp.mean(
                (arm_posture_soft_excess > 0.0).astype(jnp.float32), axis=-1
            ),
            "metric/arm_command_posture_soft_exceed_fraction": jnp.mean(
                (arm_command_posture_soft_excess > 0.0).astype(jnp.float32), axis=-1
            ),
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
            "descending_intercept_excess_penalty": (
                term_descending_intercept_excess_penalty * self.dt
            ),
            "pre_hit_intercept": term_pre_hit_intercept * self.dt,
            "pre_hit_intercept_penalty": term_pre_hit_intercept_penalty * self.dt,
            "racket_xy_reward": term_racket_xy_reward * self.dt,
            "racket_xy_penalty": term_racket_xy_penalty * self.dt,
            "racket_z_penalty": term_racket_z_penalty * self.dt,
            "racket_up_drift_penalty": term_racket_up_drift_penalty * self.dt,
            "racket_vertical_acc_penalty": term_racket_vertical_acc_penalty * self.dt,
            "metric/racket_vertical_acc_abs": jnp.abs(racket_vertical_acc),
            "racket_tilt_angular_speed_penalty": (
                term_racket_tilt_angular_speed_penalty * self.dt
            ),
            "metric/racket_tilt_angular_speed_rad_s": (
                racket_tilt_angular_speed
            ),
            "racket_stability_angular_speed_penalty": (
                term_racket_stability_angular_speed_penalty * self.dt
            ),
            "metric/racket_stability_angular_speed_rad_s": (
                racket_stability_angular_speed
            ),
            "contact_flatness_penalty": term_contact_flatness_penalty * self.dt,
            "racket_flatness_penalty": term_racket_flatness_penalty * self.dt,
            "post_hit_racket_retreat_penalty": (
                term_post_hit_racket_retreat_penalty * self.dt
            ),
            "metric/post_hit_racket_retreat_pen": post_hit_racket_retreat_pen,
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
            "arm_velocity_usage_penalty": term_arm_velocity_usage_penalty * self.dt,
            "arm_acceleration_usage_penalty": term_arm_acceleration_usage_penalty * self.dt,
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
            "hit_flatness_excess_penalty": term_hit_flatness_excess_penalty,
            "hit_contact_center_excess_penalty": (
                term_hit_contact_center_excess_penalty
            ),
            "hit_height_bonus": term_hit_height_bonus,
            "hit_cycle_q_closure_penalty": term_hit_cycle_q_closure_penalty,
            "hit_cycle_action_dc_penalty": term_hit_cycle_action_dc_penalty,
            "hit_cycle_q_excursion_penalty": term_hit_cycle_q_excursion_penalty,
            "metric/hit_cycle_eligible": hit_cycle_eligible.astype(jnp.float32),
            "metric/hit_cycle_q_closure_pen": jnp.where(
                hit_cycle_eligible, hit_cycle_q_pen, 0.0
            ),
            "metric/hit_cycle_q_error_max_deg": jnp.where(
                hit_cycle_eligible,
                jnp.rad2deg(hit_cycle_q_error_max_rad),
                0.0,
            ),
            "metric/hit_cycle_action_dc_pen": jnp.where(
                hit_cycle_eligible, hit_cycle_action_dc_pen, 0.0
            ),
            "metric/hit_cycle_q_excursion_pen": jnp.where(
                hit_cycle_eligible, hit_cycle_q_excursion_pen, 0.0
            ),
            "metric/hit_cycle_q_excursion_max_deg": jnp.where(
                hit_cycle_eligible,
                jnp.rad2deg(hit_cycle_q_excursion_max_rad),
                0.0,
            ),
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
            "hit_racket_vxy_penalty": term_hit_racket_vxy_penalty,
            "hit_racket_angular_speed_penalty": (
                term_hit_racket_angular_speed_penalty
            ),
            "metric/hit_racket_angular_speed_rad_s": jnp.where(
                hit_reward_mask, hit_racket_angular_speed, 0.0
            ),
            "metric/hit_racket_full_angular_speed_rad_s": jnp.where(
                hit_reward_mask, racket_full_angular_speed, 0.0
            ),
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
