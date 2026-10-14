"""Opt-in Cartesian pose projection for joint-QVEL policy experiments.

The policy action and robot command stay in the seven-joint space.  Forward
kinematics and the racket Jacobians are used only to remove joint-velocity
components that would leave a configured racket-position box or orientation
set.  This module is deliberately independent of the normal MJX step so
historical profiles remain unchanged while the mapping is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CartesianPoseConstraintConfig:
    """Position-box and orientation constraints in one Cartesian frame."""

    enabled: bool = True
    position_bounds_frame: str = "center_relative"
    position_lower_offset_m: tuple[float, float, float] = (-0.20, -0.15, -0.08)
    position_upper_offset_m: tuple[float, float, float] = (0.20, 0.15, 0.10)
    orientation_constraint_mode: str = "normal_cone"
    reference_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    max_tilt_rad: float = math.radians(12.0)
    reference_rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    rpy_lower_rad: tuple[float, float, float] = (-math.pi, -math.pi, -math.pi)
    rpy_upper_rad: tuple[float, float, float] = (math.pi, math.pi, math.pi)
    rpy_axis_enabled: tuple[bool, bool, bool] = (True, True, True)
    position_margin_m: float = 0.0
    orientation_margin_rad: float = 0.0
    position_cbf_gain_s: float = 20.0
    orientation_cbf_gain_s: float = 20.0
    projection_tolerance: float = 1.0e-9
    projection_max_sweeps: int = 80
    exact_backstop_iterations: int = 24


@dataclass(frozen=True)
class PoseKinematics:
    """Racket pose and arm Jacobians in a common Cartesian frame."""

    position_m: np.ndarray
    normal: np.ndarray
    translation_jacobian: np.ndarray
    angular_jacobian: np.ndarray
    rotation_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class CartesianPoseProjectionResult:
    """One projected joint-QVEL command and explicit safety diagnostics."""

    qvel: np.ndarray
    q_next: np.ndarray
    feasible: bool
    current_pose_safe: bool
    exact_pose_safe: bool
    intervened_by_pose: bool
    nonlinear_backstop_applied: bool
    projection_fallback_to_hold: bool
    correction_norm_rad_s: float
    retained_fraction: float
    max_linear_violation: float
    projection_sweeps: int
    active_constraint_names: tuple[str, ...]
    failure_reason: str


KinematicsFunction = Callable[[np.ndarray], PoseKinematics]


def _finite_vector(value: object, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-{size} vector")
    return vector


def _validated_config(
    config: CartesianPoseConstraintConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if config.position_bounds_frame not in {"center_relative", "absolute"}:
        raise ValueError(
            "position_bounds_frame must be 'center_relative' or 'absolute'"
        )
    lower = _finite_vector(config.position_lower_offset_m, 3, "position lower")
    upper = _finite_vector(config.position_upper_offset_m, 3, "position upper")
    margin = float(config.position_margin_m)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("position_margin_m must be finite and non-negative")
    if np.any(lower + margin >= upper - margin):
        raise ValueError("position bounds must remain ordered after the margin")
    orientation_margin = float(config.orientation_margin_rad)
    if not math.isfinite(orientation_margin) or orientation_margin < 0.0:
        raise ValueError(
            "orientation_margin_rad must be finite and non-negative"
        )
    if config.orientation_constraint_mode == "normal_cone":
        normal = _finite_vector(
            config.reference_normal, 3, "reference normal"
        )
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1.0e-12:
            raise ValueError("reference normal must be non-zero")
        normal = normal / normal_norm
        max_tilt = float(config.max_tilt_rad)
        if not math.isfinite(max_tilt) or not 0.0 < max_tilt < math.pi:
            raise ValueError("max_tilt_rad must lie strictly within (0, pi)")
        effective_max_tilt = max_tilt - orientation_margin
        if effective_max_tilt <= 0.0:
            raise ValueError("orientation margin must be smaller than max tilt")
        min_normal_cos = math.cos(effective_max_tilt)
    elif config.orientation_constraint_mode == "relative_rpy_box":
        reference_rotation = _validated_rotation_matrix(
            config.reference_rotation_matrix, "reference rotation"
        )
        normal = reference_rotation[:, 2]
        _validated_rpy_bounds(config)
        _validated_rpy_axis_mask(config)
        min_normal_cos = -1.0
    else:
        raise ValueError(
            "orientation_constraint_mode must be 'normal_cone' or "
            "'relative_rpy_box'"
        )
    for name, value in (
        ("position_cbf_gain_s", config.position_cbf_gain_s),
        ("orientation_cbf_gain_s", config.orientation_cbf_gain_s),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if int(config.projection_max_sweeps) <= 0:
        raise ValueError("projection_max_sweeps must be positive")
    if int(config.exact_backstop_iterations) <= 0:
        raise ValueError("exact_backstop_iterations must be positive")
    return (
        lower + margin,
        upper - margin,
        normal,
        min_normal_cos,
    )


def _validated_rotation_matrix(value: object, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    # MJX exposes site rotations as float32, so long GPU rollouts can carry
    # roundoff above the tighter tolerance used by CPU MuJoCo's float64 FK.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4):
        raise ValueError(f"{name} must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-4):
        raise ValueError(f"{name} must be a proper rotation")
    return rotation


def _validated_rpy_bounds(
    config: CartesianPoseConstraintConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = _finite_vector(config.rpy_lower_rad, 3, "RPY lower")
    upper = _finite_vector(config.rpy_upper_rad, 3, "RPY upper")
    margin = float(config.orientation_margin_rad)
    effective_lower = lower + margin
    effective_upper = upper - margin
    if np.any(effective_lower >= effective_upper):
        raise ValueError("RPY bounds must remain ordered after the margin")
    reference = _validated_rotation_matrix(
        config.reference_rotation_matrix, "reference rotation"
    )
    return effective_lower, effective_upper, reference


def _validated_rpy_axis_mask(
    config: CartesianPoseConstraintConfig,
) -> np.ndarray:
    values = tuple(config.rpy_axis_enabled)
    if len(values) != 3 or any(type(value) is not bool for value in values):
        raise ValueError("rpy_axis_enabled must contain exactly three booleans")
    mask = np.asarray(values, dtype=bool)
    if not np.any(mask):
        raise ValueError("relative RPY mode must enable at least one axis")
    return mask


def _matrix_to_euler_xyz(rotation: np.ndarray) -> np.ndarray:
    """Match the tuner's XYZ Euler-angle extraction convention."""

    matrix = _validated_rotation_matrix(rotation, "racket rotation")
    sine_pitch = float(np.clip(matrix[0, 2], -1.0, 1.0))
    pitch = math.asin(sine_pitch)
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1.0e-8:
        roll = math.atan2(-matrix[1, 2], matrix[2, 2])
        yaw = math.atan2(-matrix[0, 1], matrix[0, 0])
    else:
        roll = math.atan2(matrix[2, 1], matrix[1, 1])
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def _relative_rpy(
    pose: PoseKinematics,
    config: CartesianPoseConstraintConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if pose.rotation_matrix is None:
        raise ValueError("relative RPY constraints require a rotation matrix")
    lower, upper, reference = _validated_rpy_bounds(config)
    current = _validated_rotation_matrix(
        pose.rotation_matrix, "racket rotation"
    )
    return _matrix_to_euler_xyz(reference.T @ current), reference


def _position_coordinates(
    position_m: np.ndarray,
    center_position_m: np.ndarray,
    config: CartesianPoseConstraintConfig,
) -> np.ndarray:
    if config.position_bounds_frame == "absolute":
        return position_m
    return position_m - center_position_m


def _relative_rpy_jacobian(
    rpy: np.ndarray,
    reference_rotation: np.ndarray,
    angular_jacobian: np.ndarray,
) -> np.ndarray:
    """Map joint velocity to relative XYZ-Euler rates."""

    roll, pitch, _ = rpy
    spatial_euler_rate = np.asarray(
        [
            [1.0, 0.0, math.sin(pitch)],
            [0.0, math.cos(roll), -math.sin(roll) * math.cos(pitch)],
            [0.0, math.sin(roll), math.cos(roll) * math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    if abs(float(np.linalg.det(spatial_euler_rate))) <= 1.0e-8:
        raise ValueError("relative RPY Jacobian is singular")
    angular_in_reference = reference_rotation.T @ angular_jacobian
    return np.linalg.solve(spatial_euler_rate, angular_in_reference)


def _validated_kinematics(
    pose: PoseKinematics,
    joint_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = _finite_vector(pose.position_m, 3, "racket position")
    normal = _finite_vector(pose.normal, 3, "racket normal")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-12:
        raise ValueError("racket normal must be non-zero")
    normal = normal / normal_norm
    translation = np.asarray(pose.translation_jacobian, dtype=np.float64)
    angular = np.asarray(pose.angular_jacobian, dtype=np.float64)
    expected = (3, joint_count)
    if translation.shape != expected or not np.all(np.isfinite(translation)):
        raise ValueError(f"translation Jacobian must have shape {expected}")
    if angular.shape != expected or not np.all(np.isfinite(angular)):
        raise ValueError(f"angular Jacobian must have shape {expected}")
    return position, normal, translation, angular


def _pose_is_safe(
    pose: PoseKinematics,
    center_position_m: np.ndarray,
    config: CartesianPoseConstraintConfig,
    *,
    tolerance: float = 1.0e-9,
) -> bool:
    lower, upper, reference_normal, min_normal_cos = _validated_config(config)
    position, normal, _, _ = _validated_kinematics(
        pose, np.asarray(pose.translation_jacobian).shape[1]
    )
    coordinates = _position_coordinates(
        position, center_position_m, config
    )
    position_safe = bool(
        np.all(coordinates >= lower - tolerance)
        and np.all(coordinates <= upper + tolerance)
    )
    if config.orientation_constraint_mode == "normal_cone":
        orientation_safe = bool(
            float(np.dot(normal, reference_normal))
            >= min_normal_cos - tolerance
        )
    else:
        rpy, _ = _relative_rpy(pose, config)
        rpy_lower, rpy_upper, _ = _validated_rpy_bounds(config)
        enabled_axes = _validated_rpy_axis_mask(config)
        orientation_safe = bool(
            np.all(rpy[enabled_axes] >= rpy_lower[enabled_axes] - tolerance)
            and np.all(
                rpy[enabled_axes] <= rpy_upper[enabled_axes] + tolerance
            )
        )
    return bool(position_safe and orientation_safe)


def pose_constraint_observation(
    pose: PoseKinematics,
    *,
    center_position_m: np.ndarray,
    config: CartesianPoseConstraintConfig,
    include_parameters: bool,
) -> np.ndarray:
    """Return deployable pose/slack features, optionally with variable bounds.

    Normal-cone mode returns 13 state features and optionally 10 parameters.
    Relative-RPY mode returns position/RPY, six position slacks, and two
    slacks per enabled orientation axis.  With roll/pitch enabled and yaw
    disabled this is 16 state features, or 26 with the fixed bounds included.
    """

    lower, upper, reference_normal, min_normal_cos = _validated_config(config)
    center = _finite_vector(center_position_m, 3, "center position")
    joint_count = np.asarray(pose.translation_jacobian).shape[1]
    position, normal, _, _ = _validated_kinematics(pose, joint_count)
    coordinates = _position_coordinates(position, center, config)
    if config.orientation_constraint_mode == "normal_cone":
        state = np.concatenate(
            [
                coordinates,
                normal,
                coordinates - lower,
                upper - coordinates,
                np.asarray(
                    [
                        float(np.dot(normal, reference_normal))
                        - min_normal_cos
                    ]
                ),
            ]
        )
        if not include_parameters:
            return state.astype(np.float32)
        parameters = np.concatenate(
            [lower, upper, reference_normal, np.asarray([min_normal_cos])]
        )
    else:
        rpy, _ = _relative_rpy(pose, config)
        rpy_lower, rpy_upper, _ = _validated_rpy_bounds(config)
        enabled_axes = _validated_rpy_axis_mask(config)
        state = np.concatenate(
            [
                coordinates,
                rpy,
                coordinates - lower,
                upper - coordinates,
                (rpy - rpy_lower)[enabled_axes],
                (rpy_upper - rpy)[enabled_axes],
            ]
        )
        if not include_parameters:
            return state.astype(np.float32)
        parameters = np.concatenate(
            [
                lower,
                upper,
                rpy_lower[enabled_axes],
                rpy_upper[enabled_axes],
            ]
        )
    return np.concatenate([state, parameters]).astype(np.float32)


def pose_constraint_violations(
    pose: PoseKinematics,
    *,
    center_position_m: np.ndarray,
    config: CartesianPoseConstraintConfig,
) -> tuple[float, float]:
    """Return maximum position and constrained-orientation violations.

    Position is reported in metres.  Orientation is reported in radians and
    excludes every disabled RPY axis, so a free yaw never counts as a
    violation.
    """

    lower, upper, reference_normal, min_normal_cos = _validated_config(config)
    center = _finite_vector(center_position_m, 3, "center position")
    joint_count = np.asarray(pose.translation_jacobian).shape[1]
    position, normal, _, _ = _validated_kinematics(pose, joint_count)
    coordinates = _position_coordinates(position, center, config)
    position_violation = float(
        max(
            0.0,
            np.max(lower - coordinates),
            np.max(coordinates - upper),
        )
    )
    if config.orientation_constraint_mode == "normal_cone":
        tilt = math.acos(
            float(np.clip(np.dot(normal, reference_normal), -1.0, 1.0))
        )
        effective_max_tilt = math.acos(
            float(np.clip(min_normal_cos, -1.0, 1.0))
        )
        orientation_violation = max(0.0, tilt - effective_max_tilt)
    else:
        rpy, _ = _relative_rpy(pose, config)
        rpy_lower, rpy_upper, _ = _validated_rpy_bounds(config)
        enabled_axes = _validated_rpy_axis_mask(config)
        orientation_violation = float(
            max(
                0.0,
                np.max(rpy_lower[enabled_axes] - rpy[enabled_axes]),
                np.max(rpy[enabled_axes] - rpy_upper[enabled_axes]),
            )
        )
    return position_violation, orientation_violation


def relative_rpy_angles(
    pose: PoseKinematics,
    *,
    config: CartesianPoseConstraintConfig,
) -> np.ndarray:
    """Return relative XYZ-Euler angles for RPY-box diagnostics."""

    if config.orientation_constraint_mode != "relative_rpy_box":
        raise ValueError("relative RPY angles require relative_rpy_box mode")
    rpy, _ = _relative_rpy(pose, config)
    return rpy.copy()


def _project_halfspaces_dykstra(
    nominal: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    scale: np.ndarray,
    tolerance: float,
    max_sweeps: int,
) -> tuple[np.ndarray, bool, int, float]:
    """Project onto linear half-spaces and a box with bounded-cost Dykstra."""

    nominal = np.asarray(nominal, dtype=np.float64)
    scale = np.maximum(np.abs(np.asarray(scale, dtype=np.float64)), 1.0e-9)
    joint_count = nominal.size
    constraint_rows: list[np.ndarray] = []
    constraint_bounds: list[float] = []
    for row, bound in zip(rows, bounds, strict=True):
        transformed = np.asarray(row, dtype=np.float64) * scale
        norm = float(np.linalg.norm(transformed))
        if norm <= 1.0e-12:
            if float(bound) < -tolerance:
                return np.zeros_like(nominal), False, 0, -float(bound)
            continue
        constraint_rows.append(transformed / norm)
        constraint_bounds.append(float(bound) / norm)
    for index in range(joint_count):
        low_row = np.zeros(joint_count, dtype=np.float64)
        low_row[index] = -scale[index]
        constraint_rows.append(low_row / scale[index])
        constraint_bounds.append(float(-lower[index] / scale[index]))
        high_row = np.zeros(joint_count, dtype=np.float64)
        high_row[index] = scale[index]
        constraint_rows.append(high_row / scale[index])
        constraint_bounds.append(float(upper[index] / scale[index]))

    matrix = np.asarray(constraint_rows, dtype=np.float64)
    vector = np.asarray(constraint_bounds, dtype=np.float64)
    transformed_nominal = nominal / scale
    selected = transformed_nominal.copy()
    corrections = np.zeros_like(matrix)
    best = selected.copy()
    best_violation = float("inf")
    sweeps = 0
    for sweep in range(max_sweeps):
        previous = selected.copy()
        for index in range(matrix.shape[0]):
            shifted = selected + corrections[index]
            violation = float(np.dot(matrix[index], shifted) - vector[index])
            projected = (
                shifted - violation * matrix[index]
                if violation > 0.0
                else shifted
            )
            corrections[index] = shifted - projected
            selected = projected
        violation = float(max(0.0, np.max(matrix @ selected - vector)))
        if violation < best_violation:
            best = selected.copy()
            best_violation = violation
        sweeps = sweep + 1
        movement = float(np.max(np.abs(selected - previous)))
        if violation <= tolerance and movement <= max(tolerance, 1.0e-12):
            best = selected.copy()
            best_violation = violation
            break

    projected = best * scale
    projected = np.minimum(np.maximum(projected, lower), upper)
    linear_violation = float(
        max(
            0.0,
            np.max(rows @ projected - bounds) if rows.size else 0.0,
            np.max(lower - projected),
            np.max(projected - upper),
        )
    )
    feasible = bool(linear_violation <= max(1.0e-7, tolerance * 10.0))
    return projected, feasible, sweeps, linear_violation


def _bounded_reference_target(
    reference_q: np.ndarray,
    measured_q: np.ndarray,
    nominal_qvel: np.ndarray,
    dt: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    reference_error_limit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower_q = np.maximum(joint_lower, measured_q - reference_error_limit)
    upper_q = np.minimum(joint_upper, measured_q + reference_error_limit)
    if np.any(lower_q > upper_q):
        raise ValueError("joint and bounded-reference intervals are disjoint")
    q_next = np.minimum(
        np.maximum(reference_q + nominal_qvel * dt, lower_q), upper_q
    )
    return q_next, lower_q, upper_q


def _ray_scale_interval_for_box(
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float,
) -> tuple[float, float] | None:
    """Return alpha bounds for lower <= alpha * direction <= upper."""

    alpha_lower = 0.0
    alpha_upper = 1.0
    for value, low, high in zip(direction, lower, upper, strict=True):
        value = float(value)
        low = float(low)
        high = float(high)
        if abs(value) <= 1.0e-12:
            if low > tolerance or high < -tolerance:
                return None
            continue
        first = low / value
        second = high / value
        alpha_lower = max(alpha_lower, min(first, second))
        alpha_upper = min(alpha_upper, max(first, second))
    alpha_lower = max(0.0, alpha_lower)
    alpha_upper = min(1.0, alpha_upper)
    if alpha_lower > alpha_upper + tolerance:
        return None
    return alpha_lower, alpha_upper


def project_joint_qvel_to_cartesian_pose(
    *,
    reference_q: np.ndarray,
    measured_q: np.ndarray,
    nominal_qvel: np.ndarray,
    center_position_m: np.ndarray,
    kinematics: KinematicsFunction,
    config: CartesianPoseConstraintConfig,
    dt: float,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    joint_velocity_limit: np.ndarray,
    reference_error_limit: np.ndarray,
) -> CartesianPoseProjectionResult:
    """Project one joint-QVEL proposal into a Cartesian pose-safe command."""

    reference = np.asarray(reference_q, dtype=np.float64).reshape(-1)
    joint_count = reference.size
    if joint_count == 0 or not np.all(np.isfinite(reference)):
        raise ValueError("reference_q must be a finite non-empty vector")
    measured = _finite_vector(measured_q, joint_count, "measured_q")
    nominal = _finite_vector(nominal_qvel, joint_count, "nominal_qvel")
    joint_lo = _finite_vector(joint_lower, joint_count, "joint_lower")
    joint_hi = _finite_vector(joint_upper, joint_count, "joint_upper")
    velocity_limit = _finite_vector(
        joint_velocity_limit, joint_count, "joint_velocity_limit"
    )
    error_limit = _finite_vector(
        reference_error_limit, joint_count, "reference_error_limit"
    )
    center = _finite_vector(center_position_m, 3, "center position")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be positive and finite")
    dt = float(dt)
    if np.any(joint_lo >= joint_hi):
        raise ValueError("joint lower bounds must be below upper bounds")
    if np.any(velocity_limit <= 0.0) or np.any(error_limit < 0.0):
        raise ValueError("velocity limits must be positive and errors non-negative")

    bounded_q, lower_q, upper_q = _bounded_reference_target(
        reference,
        measured,
        nominal,
        dt,
        joint_lo,
        joint_hi,
        error_limit,
    )
    bounded_qvel = (bounded_q - reference) / dt
    if not config.enabled:
        return CartesianPoseProjectionResult(
            qvel=bounded_qvel,
            q_next=bounded_q,
            feasible=True,
            current_pose_safe=True,
            exact_pose_safe=True,
            intervened_by_pose=False,
            nonlinear_backstop_applied=False,
            projection_fallback_to_hold=False,
            correction_norm_rad_s=0.0,
            retained_fraction=1.0,
            max_linear_violation=0.0,
            projection_sweeps=0,
            active_constraint_names=(),
            failure_reason="",
        )

    lower_offset, upper_offset, reference_normal, min_normal_cos = (
        _validated_config(config)
    )
    current_pose = kinematics(reference)
    position, normal, translation, angular = _validated_kinematics(
        current_pose, joint_count
    )
    current_safe = _pose_is_safe(current_pose, center, config)
    if not current_safe:
        hold_q, _, _ = _bounded_reference_target(
            reference,
            measured,
            np.zeros_like(reference),
            dt,
            joint_lo,
            joint_hi,
            error_limit,
        )
        hold_exact_safe = _pose_is_safe(kinematics(hold_q), center, config)
        return CartesianPoseProjectionResult(
            qvel=np.zeros_like(reference),
            q_next=hold_q,
            feasible=False,
            current_pose_safe=False,
            exact_pose_safe=hold_exact_safe,
            intervened_by_pose=True,
            nonlinear_backstop_applied=False,
            projection_fallback_to_hold=True,
            correction_norm_rad_s=float(np.linalg.norm(bounded_qvel)),
            retained_fraction=0.0,
            max_linear_violation=float("inf"),
            projection_sweeps=0,
            active_constraint_names=(),
            failure_reason="current_pose_outside_hard_set",
        )

    coordinates = _position_coordinates(position, center, config)
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    names: list[str] = []
    position_gain = float(config.position_cbf_gain_s)
    for axis, axis_name in enumerate(("x", "y", "z")):
        rows.extend((translation[axis], -translation[axis]))
        bounds.extend(
            (
                position_gain * (upper_offset[axis] - coordinates[axis]),
                position_gain * (coordinates[axis] - lower_offset[axis]),
            )
        )
        names.extend((f"position_{axis_name}_high", f"position_{axis_name}_low"))
    orientation_gain = float(config.orientation_cbf_gain_s)
    if config.orientation_constraint_mode == "normal_cone":
        tilt_slack = float(np.dot(normal, reference_normal)) - min_normal_cos
        orientation_row = -np.cross(normal, reference_normal) @ angular
        rows.append(np.asarray(orientation_row, dtype=np.float64))
        bounds.append(orientation_gain * tilt_slack)
        names.append("racket_normal_tilt_cone")
    else:
        rpy, reference_rotation = _relative_rpy(current_pose, config)
        rpy_lower, rpy_upper, _ = _validated_rpy_bounds(config)
        enabled_axes = _validated_rpy_axis_mask(config)
        rpy_jacobian = _relative_rpy_jacobian(
            rpy, reference_rotation, angular
        )
        for axis, axis_name in enumerate(("roll", "pitch", "yaw")):
            if not enabled_axes[axis]:
                continue
            rows.extend((rpy_jacobian[axis], -rpy_jacobian[axis]))
            bounds.extend(
                (
                    orientation_gain * (rpy_upper[axis] - rpy[axis]),
                    orientation_gain * (rpy[axis] - rpy_lower[axis]),
                )
            )
            names.extend(
                (
                    f"orientation_{axis_name}_high",
                    f"orientation_{axis_name}_low",
                )
            )
    row_matrix = np.asarray(rows, dtype=np.float64)
    bound_vector = np.asarray(bounds, dtype=np.float64)

    lower_velocity = np.maximum(-velocity_limit, (lower_q - reference) / dt)
    upper_velocity = np.minimum(velocity_limit, (upper_q - reference) / dt)
    if np.any(lower_velocity > upper_velocity + 1.0e-12):
        return CartesianPoseProjectionResult(
            qvel=np.zeros_like(reference),
            q_next=reference.copy(),
            feasible=False,
            current_pose_safe=True,
            exact_pose_safe=True,
            intervened_by_pose=True,
            nonlinear_backstop_applied=False,
            projection_fallback_to_hold=True,
            correction_norm_rad_s=float(np.linalg.norm(bounded_qvel)),
            retained_fraction=0.0,
            max_linear_violation=float("inf"),
            projection_sweeps=0,
            active_constraint_names=(),
            failure_reason="joint_reference_velocity_interval_empty",
        )
    selected, feasible, sweeps, linear_violation = _project_halfspaces_dykstra(
        bounded_qvel,
        row_matrix,
        bound_vector,
        lower_velocity,
        upper_velocity,
        scale=velocity_limit,
        tolerance=float(config.projection_tolerance),
        max_sweeps=int(config.projection_max_sweeps),
    )
    if not feasible:
        zero_feasible = bool(
            np.all(lower_velocity <= 0.0)
            and np.all(upper_velocity >= 0.0)
            and np.all(bound_vector >= -float(config.projection_tolerance))
        )
        selected = np.zeros_like(reference)
        hold_q, _, _ = _bounded_reference_target(
            reference,
            measured,
            selected,
            dt,
            joint_lo,
            joint_hi,
            error_limit,
        )
        hold_exact_safe = _pose_is_safe(kinematics(hold_q), center, config)
        return CartesianPoseProjectionResult(
            qvel=selected,
            q_next=hold_q,
            feasible=bool(zero_feasible and hold_exact_safe),
            current_pose_safe=True,
            exact_pose_safe=hold_exact_safe,
            intervened_by_pose=True,
            nonlinear_backstop_applied=False,
            projection_fallback_to_hold=True,
            correction_norm_rad_s=float(np.linalg.norm(bounded_qvel)),
            retained_fraction=0.0,
            max_linear_violation=linear_violation,
            projection_sweeps=sweeps,
            active_constraint_names=tuple(names),
            failure_reason=(
                "linear_projection_unfinished_safe_hold"
                if zero_feasible
                else "linear_constraint_set_infeasible"
            ),
        )

    q_next = reference + selected * dt
    exact_safe = _pose_is_safe(kinematics(q_next), center, config)
    backstop_applied = False
    if not exact_safe:
        scale_interval = _ray_scale_interval_for_box(
            selected,
            lower_velocity,
            upper_velocity,
            tolerance=float(config.projection_tolerance),
        )
        low = 0.0 if scale_interval is None else scale_interval[0]
        high = 1.0 if scale_interval is None else scale_interval[1]
        low_q = reference + low * selected * dt
        low_safe = bool(
            scale_interval is not None
            and _pose_is_safe(kinematics(low_q), center, config)
        )
        if low_safe:
            for _ in range(int(config.exact_backstop_iterations)):
                alpha = 0.5 * (low + high)
                trial_q = reference + alpha * selected * dt
                if _pose_is_safe(kinematics(trial_q), center, config):
                    low = alpha
                else:
                    high = alpha
            selected = low * selected
        else:
            selected = np.minimum(
                np.maximum(np.zeros_like(reference), lower_velocity),
                upper_velocity,
            )
        q_next = reference + selected * dt
        exact_safe = _pose_is_safe(kinematics(q_next), center, config)
        backstop_applied = True

    correction = selected - bounded_qvel
    nominal_norm = float(np.linalg.norm(bounded_qvel))
    retained_fraction = float(
        np.linalg.norm(selected) / max(nominal_norm, 1.0e-12)
    )
    active = tuple(
        name
        for name, value, bound in zip(
            names, row_matrix @ bounded_qvel, bound_vector, strict=True
        )
        if float(value) > float(bound) + 1.0e-9
    )
    return CartesianPoseProjectionResult(
        qvel=selected,
        q_next=q_next,
        feasible=bool(exact_safe),
        current_pose_safe=True,
        exact_pose_safe=bool(exact_safe),
        intervened_by_pose=bool(np.linalg.norm(correction) > 1.0e-9),
        nonlinear_backstop_applied=backstop_applied,
        projection_fallback_to_hold=False,
        correction_norm_rad_s=float(np.linalg.norm(correction)),
        retained_fraction=retained_fraction,
        max_linear_violation=linear_violation,
        projection_sweeps=sweeps,
        active_constraint_names=active,
        failure_reason="" if exact_safe else "nonlinear_backstop_failed",
    )
