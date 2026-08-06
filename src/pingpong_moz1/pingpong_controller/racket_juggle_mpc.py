"""Hybrid first-exit MPC planner for autonomous racket juggling.

This module intentionally separates the discontinuous ball/racket event model
from the robot's joint-level controller.  During free flight it predicts only
up to the *next* descending contact (the first exit), computes the paddle
velocity required by a Newton impact model, and solves a finite-horizon
Cartesian double-integrator MPC.  The caller feeds the first position/velocity
reference to the existing Jacobian/joint safety controller and replans on the
next control tick.

The launch from a ball resting on the racket is a different contact mode.  A
smooth cosine lift gives the ball a common upward velocity and then accelerates
the racket downward faster than gravity so the unilateral contact opens.  A
state machine and an explicit feasibility check prevent a saturated trajectory
from being mistaken for a successful launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pingpong_controller.racket_ball_tracker import RacketMpcTuning


GRAVITY_M_S2 = 9.80665
JUGGLE_INITIAL_Q_DEG = np.array(
    [5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214],
    dtype=np.float64,
)
JUGGLE_INITIAL_Q_RAD = np.deg2rad(JUGGLE_INITIAL_Q_DEG)


@dataclass(frozen=True)
class JugglePhysicsConfig:
    """Runtime-adjustable launch, ball and contact parameters."""

    countdown_s: float = 3.0
    target_apex_height_m: float = 0.10
    seed_apex_height_m: float = 0.01
    apex_ramp_per_bounce_m: float = 0.025
    launch_peak_acceleration_m_s2: float = 16.0
    restitution: float = 0.58
    tangential_transfer: float = 0.20
    lateral_correction_gain: float = 1.0
    drag_coefficient_per_m: float = 0.143
    ball_radius_m: float = 0.020
    ready_xy_tolerance_m: float = 0.045
    ready_height_tolerance_m: float = 0.012
    ready_max_ball_speed_m_s: float = 0.12
    ready_max_racket_speed_m_s: float = 0.05
    ready_joint_tolerance_rad: float = np.deg2rad(5.0)
    ready_settle_s: float = 0.30
    max_ball_age_s: float = 0.08
    max_prediction_time_s: float = 1.0
    min_impact_lead_s: float = 0.035
    impact_refractory_s: float = 0.08
    contact_grace_s: float = 0.06
    contact_wait_lead_s: float = 0.0
    contact_wait_live_target_blend: float = 0.0
    min_bounce_speed_m_s: float = 0.12
    max_lateral_rebound_speed_m_s: float = 0.60
    control_normal_world_up_blend: float = 0.0
    allow_infeasible_mpc_fallback: bool = False
    intercept_lateral_score_weight: float = 0.0
    return_center_current_racket_blend: float = 0.0
    mpc_terminal_position_tolerance_m: float = 0.0
    mpc_terminal_velocity_tolerance_m_s: float = 0.05

    def __post_init__(self) -> None:
        strictly_positive = {
            "target_apex_height_m": self.target_apex_height_m,
            "seed_apex_height_m": self.seed_apex_height_m,
            "launch_peak_acceleration_m_s2": (
                self.launch_peak_acceleration_m_s2),
            "ball_radius_m": self.ball_radius_m,
            "ready_xy_tolerance_m": self.ready_xy_tolerance_m,
            "ready_height_tolerance_m": self.ready_height_tolerance_m,
            "ready_max_ball_speed_m_s": self.ready_max_ball_speed_m_s,
            "ready_max_racket_speed_m_s": self.ready_max_racket_speed_m_s,
            "ready_joint_tolerance_rad": self.ready_joint_tolerance_rad,
            "max_ball_age_s": self.max_ball_age_s,
            "max_prediction_time_s": self.max_prediction_time_s,
            "min_impact_lead_s": self.min_impact_lead_s,
            "impact_refractory_s": self.impact_refractory_s,
            "contact_grace_s": self.contact_grace_s,
            "min_bounce_speed_m_s": self.min_bounce_speed_m_s,
            "max_lateral_rebound_speed_m_s": (
                self.max_lateral_rebound_speed_m_s),
            "mpc_terminal_velocity_tolerance_m_s": (
                self.mpc_terminal_velocity_tolerance_m_s),
        }
        for name, value in strictly_positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name, value in {
            "countdown_s": self.countdown_s,
            "apex_ramp_per_bounce_m": self.apex_ramp_per_bounce_m,
            "drag_coefficient_per_m": self.drag_coefficient_per_m,
            "ready_settle_s": self.ready_settle_s,
            "contact_wait_lead_s": self.contact_wait_lead_s,
            "contact_wait_live_target_blend": (
                self.contact_wait_live_target_blend),
            "control_normal_world_up_blend": (
                self.control_normal_world_up_blend),
            "intercept_lateral_score_weight": (
                self.intercept_lateral_score_weight),
            "return_center_current_racket_blend": (
                self.return_center_current_racket_blend),
            "mpc_terminal_position_tolerance_m": (
                self.mpc_terminal_position_tolerance_m),
        }.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if float(self.control_normal_world_up_blend) > 1.0:
            raise ValueError("control_normal_world_up_blend must be in [0, 1]")
        if float(self.contact_wait_live_target_blend) > 1.0:
            raise ValueError("contact_wait_live_target_blend must be in [0, 1]")
        if float(self.return_center_current_racket_blend) > 1.0:
            raise ValueError("return_center_current_racket_blend must be in [0, 1]")
        if not 0.0 < float(self.restitution) <= 1.2:
            raise ValueError("restitution must be in (0, 1.2]")
        if not 0.0 < float(self.tangential_transfer) <= 1.0:
            raise ValueError("tangential_transfer must be in (0, 1]")
        if (not np.isfinite(self.lateral_correction_gain)
                or not 0.0 <= float(self.lateral_correction_gain) <= 2.0):
            raise ValueError("lateral_correction_gain must be in [0, 2]")
        if float(self.target_apex_height_m) < float(
                self.seed_apex_height_m):
            raise ValueError(
                "target_apex_height_m must be >= seed_apex_height_m")


@dataclass(frozen=True)
class BallImpactPrediction:
    time_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray


@dataclass(frozen=True)
class BallInterceptCandidate:
    """A dynamically feasible future ball/racket impact event."""

    time_s: float
    ball_position_m: np.ndarray
    ball_velocity_m_s: np.ndarray
    racket_position_m: np.ndarray
    racket_velocity_m_s: np.ndarray
    outgoing_ball_velocity_m_s: np.ndarray
    cubic_peak_acceleration_m_s2: float
    cubic_peak_speed_m_s: float
    rebound_profile_index: int = 0
    dynamically_feasible: bool = True
    dynamic_violation_ratio: float = 1.0


@dataclass(frozen=True)
class LaunchProfile:
    amplitude_m: float
    omega_rad_s: float
    separation_time_s: float
    half_cycle_s: float
    full_cycle_s: float
    peak_speed_m_s: float
    peak_acceleration_m_s2: float
    predicted_apex_above_separation_m: float

    def sample(self, elapsed_s: float) -> tuple[float, float, float]:
        """Return displacement, velocity and acceleration along the normal."""

        time_s = float(np.clip(elapsed_s, 0.0, self.full_cycle_s))
        phase = self.omega_rad_s * time_s
        displacement = self.amplitude_m * (1.0 - np.cos(phase))
        velocity = (
            self.amplitude_m * self.omega_rad_s * np.sin(phase))
        acceleration = (
            self.amplitude_m * self.omega_rad_s ** 2 * np.cos(phase))
        return float(displacement), float(velocity), float(acceleration)


@dataclass(frozen=True)
class CartesianMpcSolution:
    next_position_m: np.ndarray
    next_velocity_m_s: np.ndarray
    first_acceleration_m_s2: np.ndarray
    acceleration_sequence_m_s2: np.ndarray
    step_dt_s: float
    terminal_position_m: np.ndarray
    terminal_velocity_m_s: np.ndarray
    terminal_position_error_m: float
    terminal_velocity_error_m_s: float
    cost: float
    saturated: bool
    constraint_active: bool
    tail_anchor_used: bool


@dataclass(frozen=True)
class JugglePlan:
    """One control-tick output from :class:`RacketJuggleMpc`."""

    control_kind: str
    joint_target_q_rad: np.ndarray | None
    target_position_m: np.ndarray | None
    target_velocity_m_s: np.ndarray | None
    diagnostics: dict[str, Any]


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return out.copy()


def _unit_upward_normal(value: np.ndarray) -> np.ndarray:
    normal = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError("racket_normal must be a finite non-zero vector")
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    return normal


def _blend_with_world_up_normal(
    racket_normal: np.ndarray,
    world_up_fraction: float,
) -> np.ndarray:
    normal = _unit_upward_normal(racket_normal)
    fraction = float(world_up_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("world_up_fraction must be in [0, 1]")
    if fraction <= 1.0e-12:
        return normal
    blended = (
        (1.0 - fraction) * normal
        + fraction * np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return _unit_upward_normal(blended)


def racket_normal_from_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Rotate the racket site's local +Z axis into the control frame."""

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-9:
        raise ValueError("quaternion norm must be > 0")
    x, y, z, w = quaternion / norm
    normal = np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ], dtype=np.float64)
    return _unit_upward_normal(normal)


def launch_profile_for_seed_apex(
    seed_apex_height_m: float,
    peak_acceleration_m_s2: float,
    *,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> LaunchProfile:
    """Construct a lift-and-release pulse from an apex above separation.

    ``z=A(1-cos(omega*t))`` starts at rest with upward acceleration.  Contact
    opens when its acceleration first reaches ``-g``.  At that instant the
    common ball/racket speed is chosen to yield the requested ballistic apex.
    """

    height = float(seed_apex_height_m)
    peak_accel = float(peak_acceleration_m_s2)
    gravity = float(gravity_m_s2)
    if not np.isfinite(height) or height <= 0.0:
        raise ValueError("seed_apex_height_m must be finite and > 0")
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and > 0")
    if not np.isfinite(peak_accel) or peak_accel <= gravity:
        raise ValueError(
            "launch peak acceleration must exceed gravity for separation")
    separation_speed = np.sqrt(2.0 * gravity * height)
    separation_sine = np.sqrt(1.0 - (gravity / peak_accel) ** 2)
    omega = peak_accel * separation_sine / separation_speed
    amplitude = peak_accel / (omega ** 2)
    separation_time = np.arccos(-gravity / peak_accel) / omega
    peak_speed = amplitude * omega
    return LaunchProfile(
        amplitude_m=float(amplitude),
        omega_rad_s=float(omega),
        separation_time_s=float(separation_time),
        half_cycle_s=float(np.pi / omega),
        full_cycle_s=float(2.0 * np.pi / omega),
        peak_speed_m_s=float(peak_speed),
        peak_acceleration_m_s2=peak_accel,
        predicted_apex_above_separation_m=height,
    )


def predict_descending_impact(
    ball_position_m: np.ndarray,
    ball_velocity_m_s: np.ndarray,
    contact_ball_center_z_m: float,
    *,
    drag_coefficient_per_m: float = 0.0,
    gravity_m_s2: float = GRAVITY_M_S2,
    max_time_s: float = 1.0,
    integration_dt_s: float = 0.002,
) -> BallImpactPrediction | None:
    """Predict the next descending crossing of a horizontal contact plane."""

    position = np.asarray(ball_position_m, dtype=np.float64).reshape(3).copy()
    velocity = np.asarray(ball_velocity_m_s, dtype=np.float64).reshape(3).copy()
    contact_z = float(contact_ball_center_z_m)
    drag = float(drag_coefficient_per_m)
    gravity = float(gravity_m_s2)
    max_time = float(max_time_s)
    integration_dt = float(integration_dt_s)
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
        raise ValueError("ball state must be finite")
    if not np.isfinite(contact_z):
        raise ValueError("contact_ball_center_z_m must be finite")
    if drag < 0.0 or not np.isfinite(drag):
        raise ValueError("drag_coefficient_per_m must be finite and >= 0")
    if gravity <= 0.0 or not np.isfinite(gravity):
        raise ValueError("gravity_m_s2 must be finite and > 0")
    if max_time <= 0.0 or integration_dt <= 0.0:
        raise ValueError("prediction time and integration dt must be > 0")
    if position[2] < contact_z and velocity[2] <= 0.0:
        return None

    def acceleration(v: np.ndarray) -> np.ndarray:
        out = np.array([0.0, 0.0, -gravity], dtype=np.float64)
        speed = float(np.linalg.norm(v))
        if drag > 0.0 and speed > 0.0:
            out -= drag * speed * v
        return out

    elapsed = 0.0
    previous_position = position.copy()
    previous_velocity = velocity.copy()
    while elapsed < max_time - 1.0e-12:
        step = min(integration_dt, max_time - elapsed)
        # Velocity-Verlet is sufficiently accurate for the short first-exit
        # horizon while retaining the quadratic-drag model used by the papers.
        a0 = acceleration(velocity)
        next_position = position + step * velocity + 0.5 * step ** 2 * a0
        provisional_velocity = velocity + step * a0
        a1 = acceleration(provisional_velocity)
        next_velocity = velocity + 0.5 * step * (a0 + a1)
        next_elapsed = elapsed + step
        if (position[2] >= contact_z
                and next_position[2] <= contact_z
                and next_velocity[2] < 0.0):
            denominator = float(position[2] - next_position[2])
            fraction = (
                0.0 if denominator <= 1.0e-12
                else float(np.clip(
                    (position[2] - contact_z) / denominator, 0.0, 1.0)))
            impact_position = position + fraction * (
                next_position - position)
            impact_velocity = velocity + fraction * (
                next_velocity - velocity)
            impact_position[2] = contact_z
            return BallImpactPrediction(
                time_s=float(elapsed + fraction * step),
                position_m=impact_position,
                velocity_m_s=impact_velocity,
            )
        previous_position = position
        previous_velocity = velocity
        position = next_position
        velocity = next_velocity
        elapsed = next_elapsed
    del previous_position, previous_velocity
    return None


def desired_rebound_velocity(
    impact_position_m: np.ndarray,
    return_center_xy_m: np.ndarray,
    apex_height_m: float,
    *,
    max_lateral_speed_m_s: float,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> np.ndarray:
    """Velocity after impact that reaches an apex then returns toward center."""

    impact = np.asarray(impact_position_m, dtype=np.float64).reshape(3)
    center_xy = np.asarray(return_center_xy_m, dtype=np.float64).reshape(2)
    height = float(apex_height_m)
    max_lateral = float(max_lateral_speed_m_s)
    gravity = float(gravity_m_s2)
    if (not np.all(np.isfinite(impact))
            or not np.all(np.isfinite(center_xy))):
        raise ValueError("impact and center must be finite")
    if height <= 0.0 or max_lateral <= 0.0 or gravity <= 0.0:
        raise ValueError("apex, lateral speed and gravity must be > 0")
    vertical = float(np.sqrt(2.0 * gravity * height))
    flight_time = max(2.0 * vertical / gravity, 1.0e-6)
    lateral = (center_xy - impact[:2]) / flight_time
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm > max_lateral:
        lateral *= max_lateral / lateral_norm
    return np.array([lateral[0], lateral[1], vertical], dtype=np.float64)


def required_paddle_velocity(
    incoming_ball_velocity_m_s: np.ndarray,
    desired_outgoing_ball_velocity_m_s: np.ndarray,
    racket_normal: np.ndarray,
    *,
    restitution: float,
    tangential_transfer: float,
) -> np.ndarray:
    """Invert a normal restitution plus tangential-transfer impact model."""

    incoming = np.asarray(
        incoming_ball_velocity_m_s, dtype=np.float64).reshape(3)
    outgoing = np.asarray(
        desired_outgoing_ball_velocity_m_s, dtype=np.float64).reshape(3)
    normal = _unit_upward_normal(racket_normal)
    restitution_value = float(restitution)
    tangential = float(tangential_transfer)
    if not np.all(np.isfinite(incoming)) or not np.all(np.isfinite(outgoing)):
        raise ValueError("impact velocities must be finite")
    if not 0.0 < restitution_value <= 1.2:
        raise ValueError("restitution must be in (0, 1.2]")
    if not 0.0 < tangential <= 1.0:
        raise ValueError("tangential_transfer must be in (0, 1]")
    incoming_normal = float(normal @ incoming)
    outgoing_normal = float(normal @ outgoing)
    paddle_normal = (
        outgoing_normal + restitution_value * incoming_normal
    ) / (1.0 + restitution_value)
    incoming_tangent = incoming - incoming_normal * normal
    outgoing_tangent = outgoing - outgoing_normal * normal
    paddle_tangent = (
        outgoing_tangent - (1.0 - tangential) * incoming_tangent
    ) / tangential
    return paddle_normal * normal + paddle_tangent


def blend_rebound_tangent(
    incoming_ball_velocity_m_s: np.ndarray,
    desired_ball_velocity_m_s: np.ndarray,
    racket_normal: np.ndarray,
    correction_fraction: float,
) -> np.ndarray:
    """Ramp lateral correction without changing the desired normal rebound.

    A ball launched along a slightly tilted racket normal acquires tangential
    velocity under world gravity.  Correcting all of that on the very first,
    low-apex bounce can require an infeasible paddle velocity when tangential
    transfer is weak.  ``correction_fraction=0`` preserves incoming tangent;
    ``1`` applies the complete return-to-centre request.
    """

    incoming = np.asarray(
        incoming_ball_velocity_m_s, dtype=np.float64).reshape(3)
    desired = np.asarray(
        desired_ball_velocity_m_s, dtype=np.float64).reshape(3)
    normal = _unit_upward_normal(racket_normal)
    fraction = float(correction_fraction)
    if (not np.all(np.isfinite(incoming))
            or not np.all(np.isfinite(desired))):
        raise ValueError("rebound velocities must be finite")
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("correction_fraction must be in [0, 1]")
    incoming_normal = float(normal @ incoming)
    desired_normal = float(normal @ desired)
    incoming_tangent = incoming - incoming_normal * normal
    desired_tangent = desired - desired_normal * normal
    outgoing_tangent = (
        (1.0 - fraction) * incoming_tangent
        + fraction * desired_tangent)
    return desired_normal * normal + outgoing_tangent


def _position_within_limits(
    position: np.ndarray,
    limits: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ],
) -> bool:
    if len(limits) != 3:
        return False
    for axis, axis_limits in enumerate(limits):
        if axis_limits is None:
            continue
        if not float(axis_limits[0]) <= float(position[axis]) <= float(
                axis_limits[1]):
            return False
    return True


def _cubic_connection_bounds(
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    target_position_m: np.ndarray,
    target_velocity_m_s: np.ndarray,
    duration_s: float,
) -> tuple[float, float]:
    """Return exact endpoint-acceleration and sampled-speed bounds."""

    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        return float("inf"), float("inf")
    position = np.asarray(current_position_m, dtype=np.float64).reshape(3)
    velocity = np.asarray(current_velocity_m_s, dtype=np.float64).reshape(3)
    target_position = np.asarray(
        target_position_m, dtype=np.float64).reshape(3)
    target_velocity = np.asarray(
        target_velocity_m_s, dtype=np.float64).reshape(3)
    delta = target_position - position
    quadratic = (
        3.0 * delta / duration ** 2
        - (2.0 * velocity + target_velocity) / duration)
    cubic = (
        -2.0 * delta / duration ** 3
        + (velocity + target_velocity) / duration ** 2)
    acceleration_start = 2.0 * quadratic
    acceleration_end = 2.0 * quadratic + 6.0 * cubic * duration
    # Acceleration is affine, and the norm of an affine vector is convex, so
    # its maximum over the interval is attained at an endpoint.
    peak_acceleration = max(
        float(np.linalg.norm(acceleration_start)),
        float(np.linalg.norm(acceleration_end)),
    )
    sample_times = np.linspace(0.0, duration, 33, dtype=np.float64)
    sampled_velocity = (
        velocity[None, :]
        + 2.0 * sample_times[:, None] * quadratic[None, :]
        + 3.0 * sample_times[:, None] ** 2 * cubic[None, :])
    peak_speed = float(np.max(np.linalg.norm(sampled_velocity, axis=1)))
    return peak_acceleration, peak_speed


def select_feasible_ball_intercept(
    ball_position_m: np.ndarray,
    ball_velocity_m_s: np.ndarray,
    racket_position_m: np.ndarray,
    racket_velocity_m_s: np.ndarray,
    racket_normal: np.ndarray,
    nominal_racket_position_m: np.ndarray,
    apex_height_m: float,
    *,
    physics: JugglePhysicsConfig,
    tuning: RacketMpcTuning,
    minimum_time_s: float,
    lateral_correction_fraction: float = 1.0,
    position_limits_m: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ] = (None, None, None),
    integration_dt_s: float = 0.005,
    rebound_profiles: tuple[tuple[float, float], ...] | None = None,
    allow_infeasible_fallback: bool = False,
) -> BallInterceptCandidate | None:
    """Select a feasible descending impact while coupling paddle and ball.

    Unlike a fixed-plane crossing, every descending ballistic sample defines
    a candidate event ``racket = ball - radius * normal``.  A cubic boundary
    connection rejects candidates that cannot meet both event position and
    impact velocity under the shared Cartesian limits.  Of the feasible
    events, the one nearest the nominal racket height is selected.
    """

    ball = np.asarray(ball_position_m, dtype=np.float64).reshape(3).copy()
    ball_velocity = np.asarray(
        ball_velocity_m_s, dtype=np.float64).reshape(3).copy()
    racket = np.asarray(racket_position_m, dtype=np.float64).reshape(3)
    racket_velocity = np.asarray(
        racket_velocity_m_s, dtype=np.float64).reshape(3)
    normal = _unit_upward_normal(racket_normal)
    nominal = np.asarray(
        nominal_racket_position_m, dtype=np.float64).reshape(3)
    if not all(np.all(np.isfinite(value)) for value in (
            ball, ball_velocity, racket, racket_velocity, nominal)):
        raise ValueError("intercept state must be finite")
    minimum_time = max(float(minimum_time_s), 1.0e-6)
    integration_dt = float(integration_dt_s)
    if not np.isfinite(integration_dt) or integration_dt <= 0.0:
        raise ValueError("integration_dt_s must be finite and > 0")

    drag = float(physics.drag_coefficient_per_m)

    def acceleration(velocity: np.ndarray) -> np.ndarray:
        out = np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if drag > 0.0 and speed > 0.0:
            out -= drag * speed * velocity
        return out

    profiles = (
        ((float(apex_height_m), float(lateral_correction_fraction)),)
        if rebound_profiles is None else
        tuple((float(apex), float(fraction))
              for apex, fraction in rebound_profiles))
    if not profiles:
        raise ValueError("rebound_profiles must not be empty")
    for apex_height, correction_fraction in profiles:
        if not np.isfinite(apex_height) or apex_height <= 0.0:
            raise ValueError("apex heights must be finite and > 0")
        if (not np.isfinite(correction_fraction)
                or not 0.0 <= correction_fraction <= 1.0):
            raise ValueError("lateral correction fractions must be in [0, 1]")

    elapsed = 0.0
    candidate_times: list[float] = []
    candidate_ball_positions: list[np.ndarray] = []
    candidate_ball_velocities: list[np.ndarray] = []
    candidate_racket_positions: list[np.ndarray] = []
    acceleration_limit = 0.98 * float(
        tuning.max_cartesian_acceleration_m_s2)
    speed_limit = 0.98 * float(tuning.max_cartesian_speed_m_s)
    max_time = float(physics.max_prediction_time_s)
    while elapsed < max_time - 1.0e-12:
        step = min(integration_dt, max_time - elapsed)
        a0 = acceleration(ball_velocity)
        next_ball = ball + step * ball_velocity + 0.5 * step ** 2 * a0
        provisional_velocity = ball_velocity + step * a0
        a1 = acceleration(provisional_velocity)
        next_velocity = ball_velocity + 0.5 * step * (a0 + a1)
        elapsed += step
        ball = next_ball
        ball_velocity = next_velocity
        if elapsed + 1.0e-12 < minimum_time:
            continue
        if float(normal @ ball_velocity) >= -physics.min_bounce_speed_m_s:
            continue
        target_position = ball - physics.ball_radius_m * normal
        if not _position_within_limits(target_position, position_limits_m):
            z_limits = position_limits_m[2] if len(position_limits_m) == 3 else None
            if (z_limits is not None
                    and target_position[2] < float(z_limits[0])):
                break
            continue
        candidate_times.append(float(elapsed))
        candidate_ball_positions.append(ball.copy())
        candidate_ball_velocities.append(ball_velocity.copy())
        candidate_racket_positions.append(target_position.copy())

    if not candidate_times:
        return None

    # Evaluate all descending event candidates in one batch.  This is the
    # same impact inverse and cubic connection test as the scalar form, but it
    # avoids hundreds of tiny allocations in the real-time planning path.
    times = np.asarray(candidate_times, dtype=np.float64)
    ball_positions = np.stack(candidate_ball_positions)
    ball_velocities = np.stack(candidate_ball_velocities)
    target_positions = np.stack(candidate_racket_positions)
    incoming_normal = ball_velocities @ normal
    incoming_tangent = (
        ball_velocities - incoming_normal[:, None] * normal[None, :])
    restitution = float(physics.restitution)
    tangential_transfer = float(physics.tangential_transfer)
    durations = times[:, None]
    delta_positions = target_positions - racket[None, :]
    sample_fractions = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    sample_times = times[:, None] * sample_fractions[None, :]
    # Prefer the nominal contact height.  A tiny time term breaks ties in
    # favour of the earlier event without causing contact-height drift.
    target_from_nominal = target_positions - nominal[None, :]
    target_normal_offsets = target_from_nominal @ normal
    target_tangent_offsets = (
        target_from_nominal
        - target_normal_offsets[:, None] * normal[None, :])
    scores = (
        np.abs(target_normal_offsets)
        + float(physics.intercept_lateral_score_weight)
        * np.linalg.norm(target_tangent_offsets, axis=1)
        + 1.0e-5 * times)
    max_lateral = float(physics.max_lateral_rebound_speed_m_s)

    # Profiles are ordered from the requested rebound toward the conservative
    # seed rebound.  Reuse the one ballistic rollout, then stop at the first
    # profile with any feasible moving contact.
    for profile_index, (apex_height, correction_fraction) in enumerate(
            profiles):
        vertical_rebound_speed = float(np.sqrt(
            2.0 * GRAVITY_M_S2 * apex_height))
        rebound_flight_time = max(
            2.0 * vertical_rebound_speed / GRAVITY_M_S2,
            1.0e-6,
        )
        desired_lateral = (
            nominal[None, :2] - ball_positions[:, :2]
        ) / rebound_flight_time
        desired_lateral_norm = np.linalg.norm(desired_lateral, axis=1)
        lateral_scale = np.minimum(
            1.0,
            max_lateral / np.maximum(desired_lateral_norm, 1.0e-15),
        )
        desired_lateral *= lateral_scale[:, None]
        nominal_outgoing = np.column_stack((
            desired_lateral,
            np.full(
                times.shape,
                vertical_rebound_speed,
                dtype=np.float64,
            ),
        ))
        desired_normal = nominal_outgoing @ normal
        desired_tangent = (
            nominal_outgoing
            - desired_normal[:, None] * normal[None, :])
        outgoing_tangent = (
            (1.0 - correction_fraction) * incoming_tangent
            + correction_fraction * desired_tangent)
        outgoing = (
            desired_normal[:, None] * normal[None, :]
            + outgoing_tangent)
        paddle_normal = (
            desired_normal + restitution * incoming_normal
        ) / (1.0 + restitution)
        paddle_tangent = (
            outgoing_tangent
            - (1.0 - tangential_transfer) * incoming_tangent
        ) / tangential_transfer
        target_velocities = (
            paddle_normal[:, None] * normal[None, :] + paddle_tangent)

        quadratic = (
            3.0 * delta_positions / durations ** 2
            - (2.0 * racket_velocity[None, :] + target_velocities)
            / durations)
        cubic = (
            -2.0 * delta_positions / durations ** 3
            + (racket_velocity[None, :] + target_velocities)
            / durations ** 2)
        acceleration_start = 2.0 * quadratic
        acceleration_end = (
            acceleration_start + 6.0 * cubic * durations)
        peak_accelerations = np.maximum(
            np.linalg.norm(acceleration_start, axis=1),
            np.linalg.norm(acceleration_end, axis=1),
        )
        sampled_velocities = (
            racket_velocity[None, None, :]
            + 2.0 * sample_times[:, :, None] * quadratic[:, None, :]
            + 3.0 * sample_times[:, :, None] ** 2
            * cubic[:, None, :])
        peak_speeds = np.max(
            np.linalg.norm(sampled_velocities, axis=2), axis=1)
        target_speeds = np.linalg.norm(target_velocities, axis=1)
        target_ratio = target_speeds / max(speed_limit, 1.0e-12)
        acceleration_ratio = peak_accelerations / max(
            acceleration_limit, 1.0e-12)
        peak_speed_ratio = peak_speeds / max(speed_limit, 1.0e-12)
        violation_ratio = np.maximum.reduce((
            target_ratio,
            acceleration_ratio,
            peak_speed_ratio,
        ))
        feasible = (
            (target_speeds <= speed_limit)
            & (peak_accelerations <= acceleration_limit)
            & (peak_speeds <= speed_limit))
        feasible_indices = np.flatnonzero(feasible)
        if feasible_indices.size == 0:
            if not allow_infeasible_fallback:
                continue
            finite_indices = np.flatnonzero(np.isfinite(violation_ratio))
            if finite_indices.size == 0:
                continue
            normalized_score = scores[finite_indices]
            score_span = float(np.ptp(normalized_score))
            if score_span > 1.0e-12:
                normalized_score = (
                    (normalized_score - float(np.min(normalized_score)))
                    / score_span)
            else:
                normalized_score = np.zeros_like(normalized_score)
            selected = int(finite_indices[
                np.argmin(violation_ratio[finite_indices]
                          + 0.05 * normalized_score)])
            dynamically_feasible = False
        else:
            selected = int(feasible_indices[
                np.argmin(scores[feasible_indices])])
            dynamically_feasible = True
        return BallInterceptCandidate(
            time_s=float(times[selected]),
            ball_position_m=ball_positions[selected].copy(),
            ball_velocity_m_s=ball_velocities[selected].copy(),
            racket_position_m=target_positions[selected].copy(),
            racket_velocity_m_s=target_velocities[selected].copy(),
            outgoing_ball_velocity_m_s=outgoing[selected].copy(),
            cubic_peak_acceleration_m_s2=float(
                peak_accelerations[selected]),
            cubic_peak_speed_m_s=float(peak_speeds[selected]),
            rebound_profile_index=profile_index,
            dynamically_feasible=dynamically_feasible,
            dynamic_violation_ratio=float(violation_ratio[selected]),
        )
    return None


def solve_cartesian_first_exit_mpc(
    current_position_m: np.ndarray,
    current_velocity_m_s: np.ndarray,
    target_position_m: np.ndarray,
    target_velocity_m_s: np.ndarray,
    time_to_target_s: float,
    control_dt_s: float,
    tuning: RacketMpcTuning,
    *,
    previous_acceleration_m_s2: np.ndarray | None = None,
    feasible_acceleration_hint_m_s2: np.ndarray | None = None,
    position_limits_m: tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ] = (None, None, None),
) -> CartesianMpcSolution:
    """Solve a condensed N-step Cartesian acceleration MPC.

    The optimisation variable is the complete ``3*N`` acceleration sequence,
    not a single constant velocity.  The unconstrained Hessian is identical
    and independent on x/y/z, so the three axes are solved together as three
    right-hand sides of one ``N``-by-``N`` system.  This is algebraically
    equivalent to solving the dense ``3*N`` system while being fast enough for
    the control loop.  Exact event position/velocity equalities are retained
    while a convex blend with a verified feasible sequence enforces the shared
    Cartesian speed and acceleration balls.  A shifted tail from the previous
    receding-horizon solve may provide that feasible witness near a locked
    event.  Only the first state reference is returned to the robot controller.
    """

    position = np.asarray(current_position_m, dtype=np.float64).reshape(3)
    velocity = np.asarray(current_velocity_m_s, dtype=np.float64).reshape(3)
    target_position = np.asarray(target_position_m, dtype=np.float64).reshape(3)
    target_velocity = np.asarray(target_velocity_m_s, dtype=np.float64).reshape(3)
    if not all(np.all(np.isfinite(v)) for v in (
            position, velocity, target_position, target_velocity)):
        raise ValueError("MPC state and target must be finite")
    horizon = int(tuning.horizon_steps)
    target_time = float(time_to_target_s)
    control_dt = float(control_dt_s)
    if target_time <= 0.0 or control_dt <= 0.0:
        raise ValueError("MPC time values must be > 0")
    if len(position_limits_m) != 3:
        raise ValueError("position_limits_m must contain x/y/z limits")
    checked_position_limits: list[tuple[float, float] | None] = []
    for axis_limits in position_limits_m:
        if axis_limits is None:
            checked_position_limits.append(None)
            continue
        lower, upper = float(axis_limits[0]), float(axis_limits[1])
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "position limits require finite lower < upper")
        checked_position_limits.append((lower, upper))
    # The event time, rather than the UI horizon in seconds, is the first-exit
    # boundary.  N shared knots are distributed over that interval.
    nominal_control_steps = target_time / control_dt
    # Expected event times are repeatedly decremented by the control period;
    # binary floating point can turn an exact 3-step remainder into
    # 3.0000000000000027.  Do not invent a fourth, shorter knot, because that
    # also makes the shifted feasible tail use a different discretisation.
    effective_horizon = min(
        horizon,
        max(1, int(np.ceil(nominal_control_steps - 1.0e-9))),
    )
    step_dt = target_time / float(effective_horizon)
    n = effective_horizon

    # The scalar condensed maps are shared by x/y/z.  Keeping this axis-
    # separable representation avoids a much slower dense 3N-by-3N solve.
    position_map = np.zeros((n, n), dtype=np.float64)
    velocity_map = np.zeros((n, n), dtype=np.float64)
    for row in range(n):
        for column in range(row + 1):
            position_map[row, column] = (
                (float(row - column) + 0.5) * step_dt ** 2)
            velocity_map[row, column] = step_dt
    baseline_positions = np.stack([
        position + velocity * (float(k + 1) * step_dt)
        for k in range(n)
    ])
    baseline_velocities = np.repeat(velocity[None, :], n, axis=0)
    alphas = np.arange(1, n + 1, dtype=np.float64) / float(n)
    position_reference = np.stack([
        (1.0 - alpha) * position + alpha * target_position
        for alpha in alphas
    ])
    velocity_reference = np.stack([
        (1.0 - alpha) * velocity + alpha * target_velocity
        for alpha in alphas
    ])

    position_weight = float(tuning.position_weight)
    velocity_weight = float(tuning.velocity_weight)
    effort_weight = float(tuning.effort_weight)
    smooth_weight = float(tuning.smooth_weight)
    # Later reference points receive more authority; terminal position and
    # impact velocity therefore dominate without a separate hidden parameter.
    stage_weights = np.sqrt(alphas)
    weighted_P = stage_weights[:, None] * position_map
    weighted_position_rhs = stage_weights[:, None] * (
        position_reference - baseline_positions)
    weighted_V = stage_weights[:, None] * velocity_map
    weighted_velocity_rhs = stage_weights[:, None] * (
        velocity_reference - baseline_velocities)
    hessian = np.zeros((n, n), dtype=np.float64)
    gradient = np.zeros((n, 3), dtype=np.float64)
    if position_weight > 0.0:
        hessian += position_weight * (weighted_P.T @ weighted_P)
        gradient += position_weight * (
            weighted_P.T @ weighted_position_rhs)
    if velocity_weight > 0.0:
        hessian += velocity_weight * (weighted_V.T @ weighted_V)
        gradient += velocity_weight * (
            weighted_V.T @ weighted_velocity_rhs)
    if effort_weight > 0.0:
        hessian += effort_weight * np.eye(n)
    if smooth_weight > 0.0:
        difference = np.eye(n, dtype=np.float64)
        for k in range(n):
            if k > 0:
                difference[k, k - 1] = -1.0
        smooth_rhs = np.zeros((n, 3), dtype=np.float64)
        if previous_acceleration_m_s2 is not None:
            previous = np.asarray(
                previous_acceleration_m_s2, dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(previous)):
                raise ValueError("previous acceleration must be finite")
            smooth_rhs[0] = previous
        hessian += smooth_weight * (difference.T @ difference)
        gradient += smooth_weight * (difference.T @ smooth_rhs)
    if n < 2:
        raise ValueError(
            "first-exit MPC requires at least two effective horizon steps")
    hessian += 1.0e-10 * np.eye(n)

    # The next impact is the hybrid event boundary, so terminal position and
    # velocity are equalities rather than another soft stage cost.  This keeps
    # effort/smoothness weights from silently trading away the strike itself.
    terminal_map = np.stack((position_map[-1], velocity_map[-1]), axis=0)
    terminal_rhs = np.stack((
        target_position - baseline_positions[-1],
        target_velocity - baseline_velocities[-1],
    ), axis=0)
    kkt = np.block([
        [hessian, terminal_map.T],
        [terminal_map, np.zeros((2, 2), dtype=np.float64)],
    ])
    kkt_rhs = np.concatenate((gradient, terminal_rhs), axis=0)
    optimal_accelerations = np.linalg.solve(kkt, kkt_rhs)[:n]
    # A minimum-effort sequence satisfying the same event equalities provides
    # a feasible anchor for the Cartesian speed/acceleration constraints.  The
    # feasible set is convex, so a line search from this anchor toward the
    # weighted MPC optimum retains the exact terminal boundary.
    terminal_gram = terminal_map @ terminal_map.T
    safe_accelerations = terminal_map.T @ np.linalg.solve(
        terminal_gram, terminal_rhs)
    projected_hint: np.ndarray | None = None
    if feasible_acceleration_hint_m_s2 is not None:
        raw_hint = np.asarray(
            feasible_acceleration_hint_m_s2, dtype=np.float64)
        if not np.all(np.isfinite(raw_hint)):
            raise ValueError("feasible acceleration hint must be finite")
        if raw_hint.shape == (n, 3):
            # Measured state can differ slightly from last tick's predicted
            # first state.  Apply the minimum-norm correction that restores the
            # new exact event boundary before testing all path constraints.
            hint_residual = terminal_rhs - terminal_map @ raw_hint
            projected_hint = (
                raw_hint
                + terminal_map.T @ np.linalg.solve(
                    terminal_gram, hint_residual))
    acceleration_limit = float(tuning.max_cartesian_acceleration_m_s2)
    speed_limit = float(tuning.max_cartesian_speed_m_s)
    constraint_tolerance = 1.0e-10

    def sequence_feasible(sequence: np.ndarray) -> bool:
        if np.any(np.linalg.norm(sequence, axis=1) > (
                acceleration_limit + constraint_tolerance)):
            return False
        candidate_velocities = (
            velocity[None, :]
            + step_dt * np.cumsum(sequence, axis=0))
        if np.any(np.linalg.norm(candidate_velocities, axis=1) > (
                speed_limit + constraint_tolerance)):
            return False
        candidate_positions = (
            baseline_positions + position_map @ sequence)
        start_positions = np.vstack((
            position[None, :], candidate_positions[:-1]))
        start_velocities = np.vstack((
            velocity[None, :], candidate_velocities[:-1]))
        for axis, axis_limits in enumerate(checked_position_limits):
            if axis_limits is None:
                continue
            lower, upper = axis_limits
            if (position[axis] < lower - constraint_tolerance
                    or position[axis] > upper + constraint_tolerance):
                return False
            endpoints = candidate_positions[:, axis]
            if (np.any(endpoints < lower - constraint_tolerance)
                    or np.any(endpoints > upper + constraint_tolerance)):
                return False
            axis_acceleration = sequence[:, axis]
            turning_time = np.full(n, np.nan, dtype=np.float64)
            nonzero = np.abs(axis_acceleration) > 1.0e-15
            turning_time[nonzero] = (
                -start_velocities[nonzero, axis]
                / axis_acceleration[nonzero])
            interior = (
                (turning_time > 0.0) & (turning_time < step_dt))
            if np.any(interior):
                time = turning_time[interior]
                turning_position = (
                    start_positions[interior, axis]
                    + start_velocities[interior, axis] * time
                    + 0.5 * axis_acceleration[interior] * time ** 2)
                if (np.any(turning_position < lower - constraint_tolerance)
                        or np.any(
                            turning_position
                            > upper + constraint_tolerance)):
                    return False
        return True

    constraint_active = False
    tail_anchor_used = False
    if sequence_feasible(optimal_accelerations):
        accelerations = optimal_accelerations
    else:
        anchor_accelerations: np.ndarray | None = None
        if (projected_hint is not None
                and sequence_feasible(projected_hint)):
            anchor_accelerations = projected_hint
            tail_anchor_used = True
        elif sequence_feasible(safe_accelerations):
            anchor_accelerations = safe_accelerations
        if anchor_accelerations is None:
            # The caller will reject this event after terminal-error/path
            # validation. Projection still ensures no unsafe raw reference is
            # returned as a usable non-saturated plan.
            accelerations = optimal_accelerations.copy()
        else:
            delta_accelerations = (
                optimal_accelerations - anchor_accelerations)
            anchor_velocities = (
                velocity[None, :]
                + step_dt * np.cumsum(anchor_accelerations, axis=0))
            delta_velocities = step_dt * np.cumsum(
                delta_accelerations, axis=0)
            constraint_bases = np.concatenate((
                anchor_accelerations,
                anchor_velocities,
            ), axis=0)
            constraint_deltas = np.concatenate((
                delta_accelerations,
                delta_velocities,
            ), axis=0)
            constraint_limits = np.concatenate((
                np.full(n, acceleration_limit, dtype=np.float64),
                np.full(n, speed_limit, dtype=np.float64),
            ))
            # For every acceleration and accumulated-velocity row, solve the
            # convex quadratic ||base + alpha*delta||² <= limit².  Alpha=0 is
            # the verified feasible anchor, so the positive root is the exact
            # upper end of its feasible interval.  Position constraints are
            # checked below and use the convex bisection fallback when active.
            quadratic_coeff = np.einsum(
                "ij,ij->i", constraint_deltas, constraint_deltas)
            linear_coeff = 2.0 * np.einsum(
                "ij,ij->i", constraint_bases, constraint_deltas)
            constant_coeff = np.minimum(
                np.einsum(
                    "ij,ij->i", constraint_bases, constraint_bases)
                - constraint_limits ** 2,
                0.0,
            )
            upper_roots = np.full(
                constraint_limits.shape, np.inf, dtype=np.float64)
            curved = quadratic_coeff > 1.0e-18
            discriminant = np.maximum(
                linear_coeff[curved] ** 2
                - 4.0 * quadratic_coeff[curved] * constant_coeff[curved],
                0.0,
            )
            upper_roots[curved] = (
                -linear_coeff[curved] + np.sqrt(discriminant)
            ) / (2.0 * quadratic_coeff[curved])
            linear = (~curved) & (linear_coeff > 1.0e-18)
            upper_roots[linear] = (
                -constant_coeff[linear] / linear_coeff[linear])
            blend = float(np.clip(np.min(upper_roots), 0.0, 1.0))
            if blend < 1.0:
                blend = float(np.nextafter(blend, 0.0))
            accelerations = (
                anchor_accelerations + blend * delta_accelerations)
            if not sequence_feasible(accelerations):
                # Position limits or a degenerate numerical root can make the
                # norm-only analytic blend too large.  The full feasible set
                # is convex, so bisection from the verified anchor is safe.
                low = 0.0
                high = blend
                for _ in range(32):
                    midpoint = 0.5 * (low + high)
                    trial = (
                        anchor_accelerations
                        + midpoint * delta_accelerations)
                    if sequence_feasible(trial):
                        low = midpoint
                    else:
                        high = midpoint
                accelerations = (
                    anchor_accelerations + low * delta_accelerations)
            constraint_active = True
    saturated = not sequence_feasible(accelerations)
    for acceleration in accelerations:
        magnitude = float(np.linalg.norm(acceleration))
        if magnitude > acceleration_limit + constraint_tolerance:
            acceleration *= acceleration_limit / magnitude
            saturated = True
        elif magnitude > acceleration_limit:
            acceleration *= acceleration_limit / magnitude

    predicted_position = position.copy()
    predicted_velocity = velocity.copy()
    first_position = None
    first_velocity = None
    cost = 0.0
    previous_acceleration = (
        np.zeros(3, dtype=np.float64)
        if previous_acceleration_m_s2 is None else
        np.asarray(previous_acceleration_m_s2, dtype=np.float64).reshape(3))
    for index, acceleration in enumerate(accelerations):
        predicted_position = (
            predicted_position + step_dt * predicted_velocity
            + 0.5 * step_dt ** 2 * acceleration)
        predicted_velocity = predicted_velocity + step_dt * acceleration
        speed = float(np.linalg.norm(predicted_velocity))
        if speed > speed_limit + constraint_tolerance:
            predicted_velocity *= speed_limit / speed
            saturated = True
        elif speed > speed_limit:
            predicted_velocity *= speed_limit / speed
        alpha = alphas[index]
        position_ref = (
            (1.0 - alpha) * position + alpha * target_position)
        velocity_ref = (
            (1.0 - alpha) * velocity + alpha * target_velocity)
        cost += position_weight * float(np.sum(
            (predicted_position - position_ref) ** 2))
        cost += velocity_weight * float(np.sum(
            (predicted_velocity - velocity_ref) ** 2))
        cost += effort_weight * float(np.sum(acceleration ** 2))
        cost += smooth_weight * float(np.sum(
            (acceleration - previous_acceleration) ** 2))
        previous_acceleration = acceleration
        if index == 0:
            first_position = predicted_position.copy()
            first_velocity = predicted_velocity.copy()
    assert first_position is not None and first_velocity is not None
    if not all(np.all(np.isfinite(v)) for v in (
            first_position, first_velocity,
            predicted_position, predicted_velocity)):
        raise ValueError("Cartesian MPC produced a non-finite trajectory")
    terminal_position_error = float(np.linalg.norm(
        predicted_position - target_position))
    terminal_velocity_error = float(np.linalg.norm(
        predicted_velocity - target_velocity))
    return CartesianMpcSolution(
        next_position_m=first_position,
        next_velocity_m_s=first_velocity,
        first_acceleration_m_s2=accelerations[0].copy(),
        acceleration_sequence_m_s2=accelerations.copy(),
        step_dt_s=float(step_dt),
        terminal_position_m=predicted_position,
        terminal_velocity_m_s=predicted_velocity,
        terminal_position_error_m=terminal_position_error,
        terminal_velocity_error_m_s=terminal_velocity_error,
        cost=float(cost),
        saturated=bool(saturated),
        constraint_active=bool(constraint_active),
        tail_anchor_used=bool(tail_anchor_used),
    )


class RacketJuggleMpc:
    """Backend-owned launch and first-exit MPC state machine."""

    ACTIVE_STATES = frozenset(
        {"countdown", "launch", "flight", "contact_wait"})

    def __init__(
        self,
        *,
        control_dt_s: float,
        initial_q_rad: np.ndarray = JUGGLE_INITIAL_Q_RAD,
    ) -> None:
        self.control_dt_s = float(control_dt_s)
        if not np.isfinite(self.control_dt_s) or self.control_dt_s <= 0.0:
            raise ValueError("control_dt_s must be finite and > 0")
        self.initial_q_rad = _finite_vector(initial_q_rad, "initial_q_rad")
        self.state = "positioning"
        self.run_requested = False
        self.start_edge_armed = False
        self.requires_rearm = False
        self.last_start_nonce = 0
        self.countdown_deadline_s: float | None = None
        self.ready_since_s: float | None = None
        self.launch_started_s: float | None = None
        self.launch_origin_m: np.ndarray | None = None
        self.launch_normal: np.ndarray | None = None
        self.launch_profile: LaunchProfile | None = None
        self.separation_detected = False
        self.bounce_count = 0
        self.last_impact_s: float | None = None
        self.previous_ball_velocity_m_s: np.ndarray | None = None
        self.previous_ball_position_m: np.ndarray | None = None
        self.previous_acceleration_m_s2 = np.zeros(3, dtype=np.float64)
        self.expected_impact_s: float | None = None
        self.contact_target_position_m: np.ndarray | None = None
        self.contact_target_velocity_m_s: np.ndarray | None = None
        self.contact_ball_position_m: np.ndarray | None = None
        self.contact_outgoing_ball_velocity_m_s: np.ndarray | None = None
        self.contact_apex_height_m: float | None = None
        self.contact_lateral_correction_fraction = 0.0
        self.flight_profile_scale_cap: float | None = None
        self.mpc_acceleration_tail_m_s2: np.ndarray | None = None
        self.mpc_tail_expected_impact_s: float | None = None
        self.mpc_tail_target_position_m: np.ndarray | None = None
        self.mpc_tail_target_velocity_m_s: np.ndarray | None = None
        self.contact_wait_started_s: float | None = None
        self.fault_reason: str | None = None
        self.last_diagnostics: dict[str, Any] = {}
        self.enter_mode()

    def _clear_mpc_tail(self) -> None:
        self.mpc_acceleration_tail_m_s2 = None
        self.mpc_tail_expected_impact_s = None
        self.mpc_tail_target_position_m = None
        self.mpc_tail_target_velocity_m_s = None

    def enter_mode(self) -> None:
        """Prepare the known horizontal pose; never inherits an old Start."""

        self.state = "positioning"
        self.run_requested = False
        # A backend mode entry must observe an explicit OFF payload before a
        # nonce-bearing rising edge can arm motion.  This rejects stale queued
        # Start payloads and browser/API reconnects.
        self.start_edge_armed = False
        self.requires_rearm = False
        self.countdown_deadline_s = None
        self.ready_since_s = None
        self.launch_started_s = None
        self.launch_origin_m = None
        self.launch_normal = None
        self.launch_profile = None
        self.separation_detected = False
        self.bounce_count = 0
        self.last_impact_s = None
        self.previous_ball_velocity_m_s = None
        self.previous_ball_position_m = None
        self.previous_acceleration_m_s2[:] = 0.0
        self.expected_impact_s = None
        self.contact_target_position_m = None
        self.contact_target_velocity_m_s = None
        self.contact_ball_position_m = None
        self.contact_outgoing_ball_velocity_m_s = None
        self.contact_apex_height_m = None
        self.contact_lateral_correction_fraction = 0.0
        self.flight_profile_scale_cap = None
        self._clear_mpc_tail()
        self.contact_wait_started_s = None
        self.fault_reason = None
        self.last_diagnostics = {
            "state": "positioning",
            "active": False,
            "runRequested": False,
            "startEdgeArmed": False,
            "requiresRearm": False,
            "ready": False,
            "readyReasons": ["waiting for first control sample"],
            "placement": {},
            "countdownRemainingS": None,
            "launchElapsedS": None,
            "separationDetected": False,
            "bounceCount": 0,
            "fault": None,
            "initialJointPositionDeg": JUGGLE_INITIAL_Q_DEG.tolist(),
        }

    def deactivate(self, reason: str = "mode_inactive") -> None:
        self.state = "off"
        self.run_requested = False
        self.start_edge_armed = False
        self.requires_rearm = False
        self.countdown_deadline_s = None
        self.launch_started_s = None
        self.expected_impact_s = None
        self.contact_target_position_m = None
        self.contact_target_velocity_m_s = None
        self.contact_ball_position_m = None
        self.contact_outgoing_ball_velocity_m_s = None
        self.contact_apex_height_m = None
        self.contact_lateral_correction_fraction = 0.0
        self.flight_profile_scale_cap = None
        self._clear_mpc_tail()
        self.contact_wait_started_s = None
        self.fault_reason = str(reason)
        self.last_diagnostics = {
            "state": "off",
            "active": False,
            "runRequested": False,
            "startEdgeArmed": False,
            "requiresRearm": False,
            "ready": False,
            "readyReasons": [],
            "placement": {},
            "countdownRemainingS": None,
            "launchElapsedS": None,
            "separationDetected": False,
            "bounceCount": int(self.bounce_count),
            "fault": str(reason),
            "initialJointPositionDeg": JUGGLE_INITIAL_Q_DEG.tolist(),
        }

    @property
    def active(self) -> bool:
        return self.state in self.ACTIVE_STATES

    def _fault(self, reason: str) -> None:
        self.state = "fault_hold"
        self.requires_rearm = True
        self.countdown_deadline_s = None
        self.expected_impact_s = None
        self.contact_wait_started_s = None
        self.fault_reason = str(reason)

    def fail(self, reason: str) -> None:
        """Enter the explicit-rearm hold from an external safety check."""

        self._fault(reason)

    def _placement(
        self,
        *,
        feedback_q_rad: np.ndarray,
        racket_position_m: np.ndarray,
        racket_velocity_m_s: np.ndarray,
        racket_normal: np.ndarray,
        ball_position_m: np.ndarray,
        ball_velocity_m_s: np.ndarray,
        ball_valid: bool,
        ball_age_s: float,
        physics: JugglePhysicsConfig,
    ) -> tuple[bool, list[str], dict[str, Any]]:
        q = np.asarray(feedback_q_rad, dtype=np.float64).reshape(
            self.initial_q_rad.size)
        racket = np.asarray(racket_position_m, dtype=np.float64).reshape(3)
        racket_velocity = np.asarray(
            racket_velocity_m_s, dtype=np.float64).reshape(3)
        normal = _blend_with_world_up_normal(
            racket_normal, physics.control_normal_world_up_blend)
        ball = np.asarray(ball_position_m, dtype=np.float64).reshape(3)
        ball_velocity = np.asarray(ball_velocity_m_s, dtype=np.float64).reshape(3)
        delta = ball - racket
        normal_gap = float(normal @ delta)
        tangential_delta = delta - normal_gap * normal
        tangent_distance = float(np.linalg.norm(tangential_delta))
        q_error = float(np.max(np.abs(q - self.initial_q_rad)))
        ball_speed = float(np.linalg.norm(ball_velocity))
        racket_speed = float(np.linalg.norm(racket_velocity))
        reasons: list[str] = []
        if not bool(ball_valid):
            reasons.append("ball state invalid")
        if (not np.isfinite(ball_age_s)
                or float(ball_age_s) > physics.max_ball_age_s):
            reasons.append(
                f"ball state stale ({float(ball_age_s):.3f}s)")
        if q_error > physics.ready_joint_tolerance_rad:
            reasons.append(
                f"initial pose error {q_error:.3f}rad")
        if normal[2] < 0.95:
            reasons.append("racket face is not sufficiently upward")
        if abs(normal_gap - physics.ball_radius_m) > (
                physics.ready_height_tolerance_m):
            reasons.append(
                f"ball height gap {normal_gap:.3f}m")
        if tangent_distance > physics.ready_xy_tolerance_m:
            reasons.append(
                f"ball is {tangent_distance:.3f}m from racket centre")
        if ball_speed > physics.ready_max_ball_speed_m_s:
            reasons.append(f"ball speed {ball_speed:.3f}m/s")
        if racket_speed > physics.ready_max_racket_speed_m_s:
            reasons.append(f"racket speed {racket_speed:.3f}m/s")
        payload = {
            "normalGapM": normal_gap,
            "heightErrorM": normal_gap - physics.ball_radius_m,
            "tangentDistanceM": tangent_distance,
            "ballSpeedMps": ball_speed,
            "racketSpeedMps": racket_speed,
            "initialPoseErrorMaxRad": q_error,
        }
        return not reasons, reasons, payload

    @staticmethod
    def _within_position_limits(
        position: np.ndarray,
        limits: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
        ],
    ) -> bool:
        return _position_within_limits(position, limits)

    def _diagnostics(
        self,
        *,
        now_s: float,
        ready: bool,
        ready_reasons: list[str],
        placement: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        countdown_remaining = None
        if self.state == "countdown" and self.countdown_deadline_s is not None:
            countdown_remaining = max(
                0.0, float(self.countdown_deadline_s - now_s))
        launch_elapsed = None
        if self.launch_started_s is not None and self.state == "launch":
            launch_elapsed = max(0.0, float(now_s - self.launch_started_s))
        out: dict[str, Any] = {
            "state": self.state,
            "active": self.active,
            "runRequested": bool(self.run_requested),
            "startEdgeArmed": bool(self.start_edge_armed),
            "requiresRearm": bool(self.requires_rearm),
            "ready": bool(ready),
            "readyReasons": list(ready_reasons),
            "placement": dict(placement),
            "countdownRemainingS": countdown_remaining,
            "launchElapsedS": launch_elapsed,
            "separationDetected": bool(self.separation_detected),
            "bounceCount": int(self.bounce_count),
            "fault": self.fault_reason,
            "initialJointPositionDeg": JUGGLE_INITIAL_Q_DEG.tolist(),
        }
        if self.expected_impact_s is not None:
            out["expectedImpactRemainingS"] = float(
                self.expected_impact_s - now_s)
        if extra:
            out.update(extra)
        self.last_diagnostics = out
        return dict(out)

    def step(
        self,
        *,
        wall_now_s: float,
        run_requested: bool,
        start_nonce: int,
        feedback_q_rad: np.ndarray,
        feedback_dq_rad_s: np.ndarray,
        racket_position_m: np.ndarray,
        racket_velocity_m_s: np.ndarray,
        racket_normal: np.ndarray,
        ball_position_m: np.ndarray,
        ball_velocity_m_s: np.ndarray,
        ball_valid: bool,
        ball_age_s: float,
        tuning: RacketMpcTuning,
        physics: JugglePhysicsConfig,
        available_vertical_acceleration_m_s2: float | None = None,
        position_limits_m: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
        ] = (None, None, None),
    ) -> JugglePlan:
        """Advance the launch/juggling state machine by one control tick."""

        now = float(wall_now_s)
        if not np.isfinite(now):
            raise ValueError("wall_now_s must be finite")
        q = np.asarray(feedback_q_rad, dtype=np.float64).reshape(
            self.initial_q_rad.size)
        dq = np.asarray(feedback_dq_rad_s, dtype=np.float64).reshape(
            self.initial_q_rad.size)
        racket = np.asarray(racket_position_m, dtype=np.float64).reshape(3)
        racket_velocity = np.asarray(
            racket_velocity_m_s, dtype=np.float64).reshape(3)
        normal = _unit_upward_normal(racket_normal)
        ball = np.asarray(ball_position_m, dtype=np.float64).reshape(3)
        ball_velocity = np.asarray(ball_velocity_m_s, dtype=np.float64).reshape(3)
        if not all(np.all(np.isfinite(v)) for v in (
                q, dq, racket, racket_velocity, normal, ball, ball_velocity)):
            raise ValueError("juggle state inputs must be finite")

        ready_now, ready_reasons, placement = self._placement(
            feedback_q_rad=q,
            racket_position_m=racket,
            racket_velocity_m_s=racket_velocity,
            racket_normal=normal,
            ball_position_m=ball,
            ball_velocity_m_s=ball_velocity,
            ball_valid=ball_valid,
            ball_age_s=ball_age_s,
            physics=physics,
        )
        if ready_now:
            if self.ready_since_s is None:
                self.ready_since_s = now
        else:
            self.ready_since_s = None
        ready = bool(
            ready_now and self.ready_since_s is not None
            and now - self.ready_since_s >= physics.ready_settle_s)

        requested = bool(run_requested)
        nonce = max(0, int(start_nonce))
        was_requested = bool(self.run_requested)
        if not requested:
            if self.state in self.ACTIVE_STATES or self.state == "armed_wait":
                self.state = "stopped_hold"
                self.fault_reason = None
                self.expected_impact_s = None
                self.contact_target_position_m = None
                self.contact_target_velocity_m_s = None
                self.contact_ball_position_m = None
                self.contact_outgoing_ball_velocity_m_s = None
                self.contact_apex_height_m = None
                self.contact_lateral_correction_fraction = 0.0
                self.flight_profile_scale_cap = None
                self._clear_mpc_tail()
                self.contact_wait_started_s = None
                self.previous_acceleration_m_s2[:] = 0.0
            elif self.state == "fault_hold":
                self.state = "stopped_hold"
                self.fault_reason = None
            self.run_requested = False
            self.start_edge_armed = True
            self.requires_rearm = False
        else:
            new_edge = bool(
                self.start_edge_armed
                and not was_requested
                and nonce != self.last_start_nonce)
            self.run_requested = True
            if new_edge:
                self.last_start_nonce = nonce
                self.start_edge_armed = False
                if not self.requires_rearm:
                    self.state = "armed_wait"
                    self.countdown_deadline_s = None
                    self.launch_started_s = None
                    self.launch_origin_m = None
                    self.launch_profile = None
                    self.separation_detected = False
                    self.bounce_count = 0
                    self.expected_impact_s = None
                    self.contact_target_position_m = None
                    self.contact_target_velocity_m_s = None
                    self.contact_ball_position_m = None
                    self.contact_outgoing_ball_velocity_m_s = None
                    self.contact_apex_height_m = None
                    self.contact_lateral_correction_fraction = 0.0
                    self.flight_profile_scale_cap = None
                    self._clear_mpc_tail()
                    self.contact_wait_started_s = None
                    self.fault_reason = None

        if self.state == "off":
            self.state = "positioning"
        if self.state == "positioning":
            if placement["initialPoseErrorMaxRad"] <= (
                    physics.ready_joint_tolerance_rad):
                self.state = "ready_wait"
            diagnostics = self._diagnostics(
                now_s=now, ready=ready, ready_reasons=ready_reasons,
                placement=placement)
            return JugglePlan(
                "joint", self.initial_q_rad.copy(), None, None, diagnostics)
        if self.state == "armed_wait":
            if ready:
                self.state = "countdown"
                self.countdown_deadline_s = now + physics.countdown_s
            diagnostics = self._diagnostics(
                now_s=now, ready=ready, ready_reasons=ready_reasons,
                placement=placement)
            return JugglePlan(
                "joint", self.initial_q_rad.copy(), None, None, diagnostics)

        if self.state == "countdown":
            if not ready_now:
                self.state = "armed_wait"
                self.countdown_deadline_s = None
                diagnostics = self._diagnostics(
                    now_s=now, ready=False, ready_reasons=ready_reasons,
                    placement=placement)
                return JugglePlan(
                    "joint", self.initial_q_rad.copy(), None, None,
                    diagnostics)
            if (self.countdown_deadline_s is not None
                    and now >= self.countdown_deadline_s):
                try:
                    profile = launch_profile_for_seed_apex(
                        physics.seed_apex_height_m,
                        physics.launch_peak_acceleration_m_s2,
                    )
                except ValueError as exc:
                    self._fault(f"launch infeasible: {exc}")
                    profile = None
                if profile is not None:
                    if profile.peak_speed_m_s > (
                            tuning.max_cartesian_speed_m_s + 1.0e-9):
                        self._fault(
                            "launch infeasible: peak speed "
                            f"{profile.peak_speed_m_s:.3f}m/s exceeds shared "
                            f"limit {tuning.max_cartesian_speed_m_s:.3f}m/s")
                    elif profile.peak_acceleration_m_s2 > (
                            tuning.max_cartesian_acceleration_m_s2 + 1.0e-9):
                        self._fault(
                            "launch infeasible: peak acceleration "
                            f"{profile.peak_acceleration_m_s2:.2f}m/s^2 exceeds "
                            "shared limit "
                            f"{tuning.max_cartesian_acceleration_m_s2:.2f}m/s^2")
                    elif (available_vertical_acceleration_m_s2 is not None
                            and np.isfinite(available_vertical_acceleration_m_s2)
                            and profile.peak_acceleration_m_s2 > float(
                                available_vertical_acceleration_m_s2)):
                        self._fault(
                            "launch infeasible: joint-scaled vertical "
                            f"acceleration {float(available_vertical_acceleration_m_s2):.2f}"
                            f"m/s^2 < requested {profile.peak_acceleration_m_s2:.2f}"
                            "m/s^2")
                    else:
                        launch_peak = racket + 2.0 * profile.amplitude_m * normal
                        if not self._within_position_limits(
                                launch_peak, position_limits_m):
                            self._fault(
                                "launch infeasible: stroke leaves workspace")
                        else:
                            self.state = "launch"
                            self.launch_started_s = now
                            self.launch_origin_m = racket.copy()
                            self.launch_normal = normal.copy()
                            self.launch_profile = profile
                            self.separation_detected = False
                            self.previous_ball_velocity_m_s = ball_velocity.copy()
                            self.previous_ball_position_m = ball.copy()
                            self.previous_acceleration_m_s2[:] = 0.0
            if self.state != "launch":
                diagnostics = self._diagnostics(
                    now_s=now, ready=ready, ready_reasons=ready_reasons,
                    placement=placement)
                return JugglePlan("hold", None, None, None, diagnostics)

        if self.state == "launch":
            assert self.launch_profile is not None
            assert self.launch_origin_m is not None
            assert self.launch_normal is not None
            if (not ball_valid
                    or not np.isfinite(ball_age_s)
                    or ball_age_s > physics.max_ball_age_s):
                self._fault("ball state lost or stale during launch")
                diagnostics = self._diagnostics(
                    now_s=now, ready=False, ready_reasons=ready_reasons,
                    placement=placement)
                return JugglePlan("hold", None, None, None, diagnostics)
            elapsed = max(0.0, now - float(self.launch_started_s))
            displacement, launch_speed, launch_acceleration = (
                self.launch_profile.sample(elapsed))
            target_position = (
                self.launch_origin_m + displacement * self.launch_normal)
            target_velocity = launch_speed * self.launch_normal
            relative = ball - racket
            gap = float(self.launch_normal @ relative)
            relative_normal_velocity = float(
                self.launch_normal @ (ball_velocity - racket_velocity))
            if (elapsed >= 0.75 * self.launch_profile.separation_time_s
                    and relative_normal_velocity > 0.04
                    and gap > physics.ball_radius_m
                    + 0.25 * physics.ready_height_tolerance_m):
                self.separation_detected = True
            # Do not hand the MPC a stationary racket at the top of the lift:
            # with a low seed apex the ball is still only millimetres clear and
            # a soft intercept can drive the racket back upward into it.  Keep
            # following the cosine downstroke through 20% of the recovery half
            # cycle, then inherit its negative acceleration in first-exit MPC.
            recovery_switch_s = 1.20 * self.launch_profile.half_cycle_s
            if elapsed >= recovery_switch_s and self.separation_detected:
                self.state = "flight"
                self.previous_ball_velocity_m_s = ball_velocity.copy()
                self.previous_ball_position_m = ball.copy()
                self.previous_acceleration_m_s2 = (
                    launch_acceleration * self.launch_normal)
            elif (elapsed > self.launch_profile.full_cycle_s
                    and not self.separation_detected):
                self._fault("launch did not produce a verified ball separation")
                diagnostics = self._diagnostics(
                    now_s=now, ready=False, ready_reasons=ready_reasons,
                    placement=placement)
                return JugglePlan("hold", None, None, None, diagnostics)
            extra = {
                "targetPosition": target_position.tolist(),
                "targetVelocity": target_velocity.tolist(),
                "launchAccelerationMps2": float(launch_acceleration),
                "launchProfile": {
                    "amplitudeM": self.launch_profile.amplitude_m,
                    "peakSpeedMps": self.launch_profile.peak_speed_m_s,
                    "peakAccelerationMps2": (
                        self.launch_profile.peak_acceleration_m_s2),
                    "separationTimeS": self.launch_profile.separation_time_s,
                    "recoverySwitchTimeS": recovery_switch_s,
                    "predictedSeedApexM": (
                        self.launch_profile.predicted_apex_above_separation_m),
                },
            }
            diagnostics = self._diagnostics(
                now_s=now, ready=False, ready_reasons=ready_reasons,
                placement=placement, extra=extra)
            return JugglePlan(
                "cartesian", None, target_position, target_velocity,
                diagnostics)

        if self.state in {"flight", "contact_wait"}:
            assert self.launch_origin_m is not None
            flight_normal = (
                normal if self.launch_normal is None else self.launch_normal)
            contact_wait_lead_s = max(
                2.1 * self.control_dt_s, 0.010,
                float(physics.contact_wait_lead_s))
            ball_fresh = bool(
                ball_valid
                and np.isfinite(ball_age_s)
                and ball_age_s <= physics.max_ball_age_s)

            gap = float(flight_normal @ (ball - racket))
            tangent = (ball - racket) - gap * flight_normal
            previous_velocity = self.previous_ball_velocity_m_s
            bounce_observed = False
            if ball_fresh and previous_velocity is not None:
                previous_normal_velocity = float(
                    flight_normal @ previous_velocity)
                current_normal_velocity = float(
                    flight_normal @ ball_velocity)
                outside_refractory = bool(
                    self.last_impact_s is None
                    or now - self.last_impact_s >= physics.impact_refractory_s)
                near_current_racket = bool(
                    abs(gap - physics.ball_radius_m)
                    <= 3.0 * physics.ready_height_tolerance_m
                    and float(np.linalg.norm(tangent))
                    <= physics.ready_xy_tolerance_m * 1.5)
                near_planned_contact = False
                if self.contact_ball_position_m is not None:
                    near_planned_contact = bool(
                        np.linalg.norm(ball - self.contact_ball_position_m)
                        <= max(
                            0.04,
                            3.0 * physics.ready_height_tolerance_m))
                # A low seed apex can produce a complete next flight in less
                # time than the generic noise-refractory interval.  Do not let
                # that interval suppress a *different*, explicitly planned
                # event: its own expected-time window and contact position are
                # the de-duplication key.  Early velocity noise remains blocked.
                new_planned_event_due = bool(
                    self.last_impact_s is not None
                    and self.expected_impact_s is not None
                    and self.expected_impact_s
                    > self.last_impact_s + self.control_dt_s
                    and self.expected_impact_s - contact_wait_lead_s
                    <= now
                    <= self.expected_impact_s + physics.contact_grace_s
                    and near_planned_contact)
                bounce_observed = bool(
                    (outside_refractory or new_planned_event_due)
                    and (near_current_racket or near_planned_contact)
                    and previous_normal_velocity
                    < -physics.min_bounce_speed_m_s
                    and current_normal_velocity
                    > physics.min_bounce_speed_m_s)
            if bounce_observed:
                self.bounce_count += 1
                self.last_impact_s = now
                self.state = "flight"
                self.expected_impact_s = None
                self.contact_wait_started_s = None
                self.contact_target_position_m = None
                self.contact_target_velocity_m_s = None
                self.contact_ball_position_m = None
                self.contact_outgoing_ball_velocity_m_s = None
                self.contact_apex_height_m = None
                self.contact_lateral_correction_fraction = 0.0
                self.flight_profile_scale_cap = None
                self._clear_mpc_tail()
                self.previous_acceleration_m_s2[:] = 0.0

            if self.state == "contact_wait":
                assert self.expected_impact_s is not None
                assert self.contact_target_position_m is not None
                assert self.contact_target_velocity_m_s is not None
                deadline = self.expected_impact_s + physics.contact_grace_s
                if now > deadline:
                    if not bool(physics.allow_infeasible_mpc_fallback):
                        self._fault(
                            "no rebound confirmation inside contact grace window")
                        diagnostics = self._diagnostics(
                            now_s=now, ready=False,
                            ready_reasons=ready_reasons,
                            placement=placement)
                        return JugglePlan("hold", None, None, None, diagnostics)
                    self.state = "flight"
                    self.expected_impact_s = None
                    self.contact_wait_started_s = None
                    self.contact_target_position_m = None
                    self.contact_target_velocity_m_s = None
                    self.contact_ball_position_m = None
                    self.contact_outgoing_ball_velocity_m_s = None
                    self.contact_apex_height_m = None
                    self.contact_lateral_correction_fraction = 0.0
                    self.flight_profile_scale_cap = None
                    self._clear_mpc_tail()
                if self.state == "contact_wait":
                    # Continue the planned strike up to the event.  Immediately
                    # afterwards hold the event position with zero feed-forward
                    # velocity while waiting for the delayed camera observation.
                    wait_target_position = (
                        self.contact_target_position_m.copy())
                    live_blend = float(
                        physics.contact_wait_live_target_blend)
                    if live_blend > 0.0 and ball_fresh:
                        live_target_position = (
                            ball - physics.ball_radius_m * flight_normal)
                        if len(position_limits_m) == 3:
                            for axis, axis_limits in enumerate(position_limits_m):
                                if axis_limits is None:
                                    continue
                                live_target_position[axis] = np.clip(
                                    live_target_position[axis],
                                    float(axis_limits[0]),
                                    float(axis_limits[1]),
                                )
                        # Keep the planned contact height/strike timing, but
                        # remove last-centimetre lateral prediction error.
                        wait_target_position[:2] = (
                            (1.0 - live_blend) * wait_target_position[:2]
                            + live_blend * live_target_position[:2])
                    wait_velocity = (
                        self.contact_target_velocity_m_s.copy()
                        if now <= self.expected_impact_s
                        else np.zeros(3, dtype=np.float64))
                    if ball_fresh:
                        self.previous_ball_velocity_m_s = ball_velocity.copy()
                        self.previous_ball_position_m = ball.copy()
                    extra = {
                        "contactGraceActive": True,
                        "contactGraceRemainingS": max(0.0, deadline - now),
                        "targetPosition": (
                            wait_target_position.tolist()),
                        "targetVelocity": wait_velocity.tolist(),
                        "contactWaitLiveTargetBlend": live_blend,
                    }
                    diagnostics = self._diagnostics(
                        now_s=now, ready=False, ready_reasons=ready_reasons,
                        placement=placement, extra=extra)
                    return JugglePlan(
                        "cartesian", None,
                        wait_target_position,
                        wait_velocity,
                        diagnostics)

            if not ball_fresh:
                self._fault("ball state lost or stale during juggling")
                diagnostics = self._diagnostics(
                    now_s=now, ready=False, ready_reasons=ready_reasons,
                    placement=placement)
                return JugglePlan("hold", None, None, None, diagnostics)

            # Once an event is less than the ordinary replanning lead away,
            # lock its boundary instead of chasing a later/lower crossing.
            locked_contact = bool(
                self.expected_impact_s is not None
                and self.contact_target_position_m is not None
                and self.contact_target_velocity_m_s is not None
                and self.expected_impact_s - now
                <= physics.min_impact_lead_s)
            time_to_contact = (
                None if self.expected_impact_s is None
                else float(self.expected_impact_s - now))
            if (locked_contact and time_to_contact is not None
                    and time_to_contact <= contact_wait_lead_s):
                if time_to_contact > 0.0:
                    boundary_acceleration, boundary_speed = (
                        _cubic_connection_bounds(
                            racket,
                            racket_velocity,
                            self.contact_target_position_m,
                            self.contact_target_velocity_m_s,
                            time_to_contact,
                        ))
                    if (boundary_acceleration
                            > tuning.max_cartesian_acceleration_m_s2
                            or boundary_speed
                            > tuning.max_cartesian_speed_m_s):
                        if not bool(physics.allow_infeasible_mpc_fallback):
                            self._fault(
                                "impact boundary became unreachable before "
                                "contact wait")
                            diagnostics = self._diagnostics(
                                now_s=now, ready=False,
                                ready_reasons=ready_reasons,
                                placement=placement)
                            return JugglePlan(
                                "hold", None, None, None, diagnostics)
                self.state = "contact_wait"
                self.contact_wait_started_s = now
                self.previous_ball_velocity_m_s = ball_velocity.copy()
                self.previous_ball_position_m = ball.copy()
                diagnostics = self._diagnostics(
                    now_s=now, ready=False, ready_reasons=ready_reasons,
                    placement=placement,
                    extra={
                        "contactGraceActive": True,
                        "contactGraceRemainingS": (
                            physics.contact_grace_s
                            + max(0.0, time_to_contact)),
                        "targetPosition": (
                            self.contact_target_position_m.tolist()),
                        "targetVelocity": (
                            self.contact_target_velocity_m_s.tolist()),
                    })
                return JugglePlan(
                    "cartesian", None,
                    self.contact_target_position_m.copy(),
                    self.contact_target_velocity_m_s.copy(),
                    diagnostics)

            requested_apex_height = min(
                physics.target_apex_height_m,
                physics.seed_apex_height_m
                + float(self.bounce_count)
                * physics.apex_ramp_per_bounce_m,
            )
            apex_span = max(
                0.0,
                physics.target_apex_height_m
                - physics.seed_apex_height_m)
            height_progress = (
                0.0 if apex_span <= 1.0e-9 else
                float(np.clip(
                    (requested_apex_height - physics.seed_apex_height_m)
                    / apex_span,
                    0.0,
                    1.0,
                )))
            bounce_progress = min(1.0, float(self.bounce_count) / 4.0)
            requested_lateral_correction_fraction = (
                0.0 if self.bounce_count == 0 else
                max(height_progress, bounce_progress))
            requested_lateral_correction_fraction = float(np.clip(
                requested_lateral_correction_fraction
                * float(physics.lateral_correction_gain),
                0.0,
                1.0,
            ))
            apex_height = (
                float(self.contact_apex_height_m)
                if locked_contact and self.contact_apex_height_m is not None
                else requested_apex_height)
            lateral_correction_fraction = (
                float(self.contact_lateral_correction_fraction)
                if locked_contact else
                requested_lateral_correction_fraction)
            apex_feasibility_scale = 1.0
            if (locked_contact
                    and requested_apex_height
                    > physics.seed_apex_height_m + 1.0e-9):
                apex_feasibility_scale = float(np.clip(
                    (apex_height - physics.seed_apex_height_m)
                    / (requested_apex_height
                       - physics.seed_apex_height_m),
                    0.0,
                    1.0,
                ))
            return_center = self.launch_origin_m.copy()
            center_blend = float(physics.return_center_current_racket_blend)
            if center_blend > 0.0:
                return_center[:2] = (
                    (1.0 - center_blend) * return_center[:2]
                    + center_blend * racket[:2])
            candidate: BallInterceptCandidate | None = None
            if not locked_contact:
                trial_scales = (
                    (1.0,)
                    if (abs(
                        requested_apex_height
                        - physics.seed_apex_height_m) <= 1.0e-9
                        and requested_lateral_correction_fraction <= 1.0e-9)
                    else (1.0, 0.8, 0.6, 0.4, 0.2, 0.0))
                if self.flight_profile_scale_cap is not None:
                    trial_scales = tuple(
                        scale for scale in trial_scales
                        if scale
                        <= self.flight_profile_scale_cap + 1.0e-12)
                trial_profiles = tuple(
                    (
                        physics.seed_apex_height_m
                        + trial_scale * (
                            requested_apex_height
                            - physics.seed_apex_height_m),
                        trial_scale
                        * requested_lateral_correction_fraction,
                    )
                    for trial_scale in trial_scales)
                candidate = select_feasible_ball_intercept(
                    ball,
                    ball_velocity,
                    racket,
                    racket_velocity,
                    flight_normal,
                    return_center,
                    trial_profiles[0][0],
                    physics=physics,
                    tuning=tuning,
                    minimum_time_s=physics.min_impact_lead_s,
                    lateral_correction_fraction=trial_profiles[0][1],
                    position_limits_m=position_limits_m,
                    integration_dt_s=max(self.control_dt_s, 0.004),
                    rebound_profiles=trial_profiles,
                    allow_infeasible_fallback=bool(
                        physics.allow_infeasible_mpc_fallback),
                )
                if candidate is not None:
                    selected_profile = candidate.rebound_profile_index
                    apex_height = float(
                        trial_profiles[selected_profile][0])
                    lateral_correction_fraction = float(
                        trial_profiles[selected_profile][1])
                    apex_feasibility_scale = float(
                        trial_scales[selected_profile])
                    self.flight_profile_scale_cap = (
                        apex_feasibility_scale)
                if candidate is None:
                    if not bool(physics.allow_infeasible_mpc_fallback):
                        self._fault(
                            "no dynamically feasible descending contact under "
                            "the shared MPC limits")
                        diagnostics = self._diagnostics(
                            now_s=now, ready=False,
                            ready_reasons=ready_reasons,
                            placement=placement)
                        return JugglePlan("hold", None, None, None, diagnostics)
                    recovery_target = (
                        ball - physics.ball_radius_m * flight_normal)
                    recovery_target = recovery_target.astype(np.float64)
                    if len(position_limits_m) == 3:
                        for axis, axis_limits in enumerate(position_limits_m):
                            if axis_limits is None:
                                continue
                            recovery_target[axis] = np.clip(
                                recovery_target[axis],
                                float(axis_limits[0]),
                                float(axis_limits[1]),
                            )
                    if float(flight_normal @ ball_velocity) < 0.0:
                        desired_outgoing = desired_rebound_velocity(
                            recovery_target,
                            return_center[:2],
                            max(physics.seed_apex_height_m, apex_height),
                            max_lateral_speed_m_s=(
                                physics.max_lateral_rebound_speed_m_s),
                        )
                        desired_outgoing = blend_rebound_tangent(
                            ball_velocity,
                            desired_outgoing,
                            flight_normal,
                            lateral_correction_fraction,
                        )
                        recovery_velocity = required_paddle_velocity(
                            ball_velocity,
                            desired_outgoing,
                            flight_normal,
                            restitution=physics.restitution,
                            tangential_transfer=physics.tangential_transfer,
                        )
                    else:
                        recovery_velocity = ball_velocity.copy()
                        recovery_velocity[2] = 0.0
                    recovery_speed = float(np.linalg.norm(recovery_velocity))
                    if (recovery_speed > tuning.max_cartesian_speed_m_s
                            and recovery_speed > 1.0e-12):
                        recovery_velocity *= (
                            tuning.max_cartesian_speed_m_s / recovery_speed)
                    self.previous_ball_velocity_m_s = ball_velocity.copy()
                    self.previous_ball_position_m = ball.copy()
                    self._clear_mpc_tail()
                    diagnostics = self._diagnostics(
                        now_s=now, ready=False,
                        ready_reasons=ready_reasons,
                        placement=placement,
                        extra={
                            "recoveryMode": "no_candidate_pursuit",
                            "targetPosition": recovery_target.tolist(),
                            "targetVelocity": recovery_velocity.tolist(),
                            "mpcSoftFallbackEnabled": True,
                            "returnCenter": return_center.tolist(),
                        },
                    )
                    return JugglePlan(
                        "cartesian", None,
                        recovery_target,
                        recovery_velocity,
                        diagnostics)
                time_to_contact = candidate.time_s
                self.expected_impact_s = now + candidate.time_s
                self.contact_target_position_m = (
                    candidate.racket_position_m.copy())
                self.contact_target_velocity_m_s = (
                    candidate.racket_velocity_m_s.copy())
                self.contact_ball_position_m = (
                    candidate.ball_position_m.copy())
                self.contact_outgoing_ball_velocity_m_s = (
                    candidate.outgoing_ball_velocity_m_s.copy())
                self.contact_apex_height_m = apex_height
                self.contact_lateral_correction_fraction = (
                    lateral_correction_fraction)

            assert time_to_contact is not None
            assert self.contact_target_position_m is not None
            assert self.contact_target_velocity_m_s is not None
            acceleration_hint = None
            if (locked_contact
                    and self.mpc_acceleration_tail_m_s2 is not None
                    and self.mpc_tail_expected_impact_s is not None
                    and self.expected_impact_s is not None
                    and abs(
                        self.mpc_tail_expected_impact_s
                        - self.expected_impact_s) <= 1.0e-9
                    and self.mpc_tail_target_position_m is not None
                    and self.mpc_tail_target_velocity_m_s is not None
                    and np.allclose(
                        self.mpc_tail_target_position_m,
                        self.contact_target_position_m,
                        rtol=0.0,
                        atol=1.0e-10,
                    )
                    and np.allclose(
                        self.mpc_tail_target_velocity_m_s,
                        self.contact_target_velocity_m_s,
                        rtol=0.0,
                        atol=1.0e-10,
                    )):
                acceleration_hint = (
                    self.mpc_acceleration_tail_m_s2.copy())
            try:
                solution = solve_cartesian_first_exit_mpc(
                    racket,
                    racket_velocity,
                    self.contact_target_position_m,
                    self.contact_target_velocity_m_s,
                    max(time_to_contact, 2.0 * self.control_dt_s),
                    self.control_dt_s,
                    tuning,
                    previous_acceleration_m_s2=(
                        self.previous_acceleration_m_s2),
                    feasible_acceleration_hint_m_s2=acceleration_hint,
                    position_limits_m=position_limits_m,
                )
            except (np.linalg.LinAlgError, ValueError) as exc:
                self._fault(f"first-exit MPC failed: {exc}")
                diagnostics = self._diagnostics(
                    now_s=now, ready=False, ready_reasons=ready_reasons,
                    placement=placement)
                return JugglePlan("hold", None, None, None, diagnostics)

            derived_terminal_position_tolerance_m = min(
                0.005,
                max(0.001, 0.25 * physics.ready_height_tolerance_m),
            )
            terminal_position_tolerance_m = (
                derived_terminal_position_tolerance_m
                if physics.mpc_terminal_position_tolerance_m <= 0.0 else
                float(physics.mpc_terminal_position_tolerance_m))
            terminal_velocity_tolerance_m_s = float(
                physics.mpc_terminal_velocity_tolerance_m_s)
            saturated_terminal_accepted = bool(
                solution.saturated
                and solution.terminal_position_error_m
                <= terminal_position_tolerance_m
                and solution.terminal_velocity_error_m_s
                <= terminal_velocity_tolerance_m_s)
            if ((solution.saturated and not saturated_terminal_accepted)
                    or solution.terminal_position_error_m
                    > terminal_position_tolerance_m
                    or solution.terminal_velocity_error_m_s
                    > terminal_velocity_tolerance_m_s):
                if not bool(physics.allow_infeasible_mpc_fallback):
                    self._fault(
                        "first-exit MPC cannot satisfy the impact boundary "
                        f"within limits (position error "
                        f"{solution.terminal_position_error_m:.4f}m, velocity "
                        f"error {solution.terminal_velocity_error_m_s:.3f}m/s, "
                        f"saturated={solution.saturated})")
                    diagnostics = self._diagnostics(
                        now_s=now, ready=False, ready_reasons=ready_reasons,
                        placement=placement)
                    return JugglePlan("hold", None, None, None, diagnostics)
                saturated_terminal_accepted = True

            if solution.acceleration_sequence_m_s2.shape[0] > 1:
                self.mpc_acceleration_tail_m_s2 = (
                    solution.acceleration_sequence_m_s2[1:].copy())
                self.mpc_tail_expected_impact_s = float(
                    self.expected_impact_s)
                self.mpc_tail_target_position_m = (
                    self.contact_target_position_m.copy())
                self.mpc_tail_target_velocity_m_s = (
                    self.contact_target_velocity_m_s.copy())
            else:
                self._clear_mpc_tail()
            self.previous_acceleration_m_s2 = (
                solution.first_acceleration_m_s2.copy())
            self.previous_ball_velocity_m_s = ball_velocity.copy()
            self.previous_ball_position_m = ball.copy()
            planned_ball_position = (
                None if self.contact_ball_position_m is None
                else self.contact_ball_position_m.tolist())
            planned_ball_velocity = (
                None if candidate is None
                else candidate.ball_velocity_m_s.tolist())
            outgoing = (
                None if self.contact_outgoing_ball_velocity_m_s is None
                else self.contact_outgoing_ball_velocity_m_s.tolist())
            extra = {
                "predictedImpactTimeS": time_to_contact,
                "predictedImpactPosition": planned_ball_position,
                "predictedImpactVelocity": planned_ball_velocity,
                "targetPosition": solution.next_position_m.tolist(),
                "targetVelocity": solution.next_velocity_m_s.tolist(),
                "desiredImpactRacketVelocityMps": (
                    self.contact_target_velocity_m_s.tolist()),
                "desiredOutgoingBallVelocityMps": outgoing,
                "returnCenter": return_center.tolist(),
                "targetApexHeightM": apex_height,
                "requestedApexHeightM": requested_apex_height,
                "apexFeasibilityScale": apex_feasibility_scale,
                "lateralCorrectionFraction": (
                    lateral_correction_fraction),
                "mpcActive": True,
                "mpcSoftFallbackEnabled": bool(
                    physics.allow_infeasible_mpc_fallback),
                "mpcCost": solution.cost,
                "mpcSaturated": solution.saturated,
                "mpcSaturationAccepted": saturated_terminal_accepted,
                "mpcConstraintActive": solution.constraint_active,
                "mpcTailAnchorUsed": solution.tail_anchor_used,
                "mpcTerminalPosition": solution.terminal_position_m.tolist(),
                "mpcTerminalVelocity": solution.terminal_velocity_m_s.tolist(),
                "mpcTerminalPositionErrorM": (
                    solution.terminal_position_error_m),
                "mpcTerminalVelocityErrorMps": (
                    solution.terminal_velocity_error_m_s),
                "eventBoundaryLocked": locked_contact,
            }
            if candidate is not None:
                extra["cubicFeasibilityPeakAccelerationMps2"] = (
                    candidate.cubic_peak_acceleration_m_s2)
                extra["cubicFeasibilityPeakSpeedMps"] = (
                    candidate.cubic_peak_speed_m_s)
                extra["cubicDynamicallyFeasible"] = (
                    candidate.dynamically_feasible)
                extra["cubicDynamicViolationRatio"] = (
                    candidate.dynamic_violation_ratio)
            diagnostics = self._diagnostics(
                now_s=now, ready=False, ready_reasons=ready_reasons,
                placement=placement, extra=extra)
            return JugglePlan(
                "cartesian", None, solution.next_position_m,
                solution.next_velocity_m_s, diagnostics)

        # ready_wait, stopped_hold and fault_hold are finite stationary modes.
        diagnostics = self._diagnostics(
            now_s=now, ready=ready, ready_reasons=ready_reasons,
            placement=placement)
        if self.state == "ready_wait":
            return JugglePlan(
                "joint", self.initial_q_rad.copy(), None, None, diagnostics)
        return JugglePlan("hold", None, None, None, diagnostics)
