"""NumPy port of simcode9/10 FPLC numerical contracts.

Sources: fixed_plane_v92/v93/v149, fixed_contact_plane_training and
cartesian_pose_action_projection_jax. The simulator's 24-sweep weighted
Dykstra projection and exact-FK hold backstop are kept distinct. No RMP or
actuator dynamics run here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import math
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _dykstra_single_numpy(selected, unit_rows, normalized_bounds, projection_sweeps):
    """The identical ordered sweeps, specialized to one seven-joint lane."""
    selected = selected.copy()
    corrections = np.zeros_like(unit_rows)
    zero = np.zeros(1, dtype=selected.dtype)[0]
    constraint_count = unit_rows.shape[0]
    for index in range(int(projection_sweeps) * constraint_count):
        ci = index % constraint_count
        row = unit_rows[ci]
        shifted = selected + corrections[ci]
        violation = np.sum(row * shifted) - normalized_bounds[ci]
        projected = shifted - max(violation, zero) * row
        corrections[ci] = shifted - projected
        selected = projected
    return selected


# No fastmath or altered sweep count/order. Compile before any command is armed.
_dykstra_single_compiled = njit(cache=False, fastmath=False, nogil=True)(_dykstra_single_numpy) if njit else None


def warm_fplc_projection(*, require_compiled=False):
    if require_compiled and not hasattr(_dykstra_single_compiled, 'nopython_signatures'):
        raise RuntimeError(
            'FPLC 200 Hz requires the compiled Numba projection. Install numba '
            'in the launch Python environment and unset NUMBA_DISABLE_JIT; '
            'the slow Python fallback is not allowed in this controller.')
    if _dykstra_single_compiled is not None:
        for dtype in (np.float32, np.float64):
            _dykstra_single_compiled(np.zeros(7, dtype), np.zeros((24, 7), dtype),
                                     np.zeros(24, dtype), 24)
    return fplc_projection_runtime()


def fplc_projection_runtime():
    compiled = bool(getattr(_dykstra_single_compiled, 'nopython_signatures', ()))
    return dict(backend='numba_nopython' if compiled else 'python_numpy',
                compiled=compiled,
                releasesGil=bool(compiled and _dykstra_single_compiled.targetoptions.get('nogil')))


HISTORY_LAGS_TICKS = tuple(range(8))
V224_HISTORY_LAGS_TICKS = (10, 20, 40)
V224_PREFIX_HISTORY_LENGTH = 41
ACTOR_OBS_DIM = 137
class CartesianPoseProjectionJaxResult(NamedTuple):
    """Projected batch command and per-lane diagnostics."""

    qvel: np.ndarray
    correction_norm_rad_s: np.ndarray
    retained_fraction: np.ndarray
    intervened: np.ndarray
    max_violation: np.ndarray


def matrix_to_euler_xyz_numpy(rotation: np.ndarray) -> np.ndarray:
    """Extract XYZ Euler angles from one or more proper rotation matrices."""

    sine_pitch = np.clip(rotation[..., 0, 2], -1.0, 1.0)
    pitch = np.arcsin(sine_pitch)
    cosine_pitch = np.cos(pitch)
    regular = np.abs(cosine_pitch) > 1.0e-6
    roll_regular = np.arctan2(-rotation[..., 1, 2], rotation[..., 2, 2])
    yaw_regular = np.arctan2(-rotation[..., 0, 1], rotation[..., 0, 0])
    roll_singular = np.arctan2(rotation[..., 2, 1], rotation[..., 1, 1])
    roll = np.where(regular, roll_regular, roll_singular)
    yaw = np.where(regular, yaw_regular, 0.0)
    return np.stack((roll, pitch, yaw), axis=-1)


def relative_xyz_euler_rate_jacobian_numpy(
    relative_rpy_rad: np.ndarray,
    reference_rotation_world: np.ndarray,
    angular_jacobian_world: np.ndarray,
) -> np.ndarray:
    """Map joint velocity to XYZ-Euler rates relative to a reference pose."""

    roll = relative_rpy_rad[..., 0]
    pitch = relative_rpy_rad[..., 1]
    zeros = np.zeros_like(roll)
    ones = np.ones_like(roll)
    spatial_euler_rate = np.stack(
        (
            np.stack((ones, zeros, np.sin(pitch)), axis=-1),
            np.stack(
                (
                    zeros,
                    np.cos(roll),
                    -np.sin(roll) * np.cos(pitch),
                ),
                axis=-1,
            ),
            np.stack(
                (
                    zeros,
                    np.sin(roll),
                    np.cos(roll) * np.cos(pitch),
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    angular_in_reference = np.einsum(
        "...ji,...jk->...ik",
        reference_rotation_world,
        angular_jacobian_world,
    )
    return np.linalg.solve(spatial_euler_rate, angular_in_reference)


def _project_halfspaces_dykstra_numpy(
    nominal_qvel: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    velocity_scale: np.ndarray,
    projection_sweeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a batch onto linear half-spaces plus its velocity box."""

    batch_size, joint_count = nominal_qvel.shape
    dtype = nominal_qvel.dtype
    scale = np.maximum(np.abs(velocity_scale), np.asarray(1.0e-6, dtype))
    if scale.ndim == 1:
        scale = np.broadcast_to(scale[None, :], nominal_qvel.shape)

    identity = np.broadcast_to(
        np.eye(joint_count, dtype=dtype)[None, :, :],
        (batch_size, joint_count, joint_count),
    )
    all_rows = np.concatenate((rows, identity, -identity), axis=1)
    all_bounds = np.concatenate(
        (bounds, velocity_upper, -velocity_lower), axis=1
    )
    transformed_rows = all_rows * scale[:, None, :]
    row_norm = np.linalg.norm(transformed_rows, axis=-1)
    safe_norm = np.maximum(row_norm, np.asarray(1.0e-9, dtype))
    unit_rows = transformed_rows / safe_norm[..., None]
    normalized_bounds = all_bounds / safe_norm

    selected = nominal_qvel / scale
    corrections = np.zeros_like(unit_rows)
    constraint_count = unit_rows.shape[1]

    if (_dykstra_single_compiled is not None and batch_size == 1 and joint_count == 7
            and unit_rows.dtype == normalized_bounds.dtype):
        selected = _dykstra_single_compiled(
            # The first original shifted = selected + zero corrections
            # promotes to the rows' dtype (FK Euler Jacobians are float64).
            selected[0].astype(unit_rows.dtype), unit_rows[0], normalized_bounds[0],
            int(projection_sweeps))[None]
    else:
        for index in range(int(projection_sweeps) * int(constraint_count)):
            constraint_index = index % constraint_count
            row = unit_rows[:, constraint_index, :]
            shifted = selected + corrections[:, constraint_index, :]
            violation = np.sum(row * shifted, axis=-1) - normalized_bounds[:, constraint_index]
            projected = shifted - np.maximum(violation, 0.0)[:, None] * row
            corrections[:, constraint_index, :] = shifted - projected
            selected = projected
    projected_qvel = np.clip(selected * scale, velocity_lower, velocity_upper)
    pose_violation = np.max(
        np.maximum(np.einsum("nkj,nj->nk", rows, projected_qvel) - bounds, 0.0),
        axis=-1,
    )
    box_violation = np.maximum(
        np.max(velocity_lower - projected_qvel, axis=-1),
        np.max(projected_qvel - velocity_upper, axis=-1),
    )
    return projected_qvel, np.maximum(pose_violation, box_violation)


def project_cartesian_pose_qvel_numpy(
    *,
    nominal_qvel: np.ndarray,
    position_m: np.ndarray,
    relative_rpy_rad: np.ndarray,
    translation_jacobian: np.ndarray,
    relative_rpy_jacobian: np.ndarray,
    position_lower_m: np.ndarray,
    position_upper_m: np.ndarray,
    roll_pitch_limit_rad: np.ndarray,
    velocity_lower: np.ndarray,
    velocity_upper: np.ndarray,
    velocity_scale: np.ndarray,
    position_gain_s: float,
    orientation_gain_s: float,
    projection_sweeps: int,
) -> CartesianPoseProjectionJaxResult:
    """Project QVEL into an XYZ and relative-roll/pitch safe tangent set.

    Yaw is intentionally absent from the orientation rows.  The caller must
    still verify the nonlinear FK candidate before publishing the next
    reference; this function guarantees only the stated linearized set.
    """

    position_lower = np.asarray(position_lower_m, dtype=nominal_qvel.dtype)
    position_upper = np.asarray(position_upper_m, dtype=nominal_qvel.dtype)
    rp_limit = np.asarray(
        roll_pitch_limit_rad, dtype=nominal_qvel.dtype
    )
    upper_position_rows = translation_jacobian
    lower_position_rows = -translation_jacobian
    upper_position_bounds = float(position_gain_s) * (
        position_upper[None, :] - position_m
    )
    lower_position_bounds = float(position_gain_s) * (
        position_m - position_lower[None, :]
    )
    rp_jacobian = relative_rpy_jacobian[:, :2, :]
    rp = relative_rpy_rad[:, :2]
    upper_rp_bounds = float(orientation_gain_s) * (rp_limit[None, :] - rp)
    lower_rp_bounds = float(orientation_gain_s) * (rp + rp_limit[None, :])
    rows = np.concatenate(
        (
            upper_position_rows,
            lower_position_rows,
            rp_jacobian,
            -rp_jacobian,
        ),
        axis=1,
    )
    bounds = np.concatenate(
        (
            upper_position_bounds,
            lower_position_bounds,
            upper_rp_bounds,
            lower_rp_bounds,
        ),
        axis=1,
    )
    selected, max_violation = _project_halfspaces_dykstra_numpy(
        nominal_qvel,
        rows,
        bounds,
        velocity_lower,
        velocity_upper,
        velocity_scale,
        projection_sweeps,
    )
    correction_norm = np.linalg.norm(selected - nominal_qvel, axis=-1)
    nominal_norm = np.linalg.norm(nominal_qvel, axis=-1)
    retained_fraction = np.linalg.norm(selected, axis=-1) / np.maximum(
        nominal_norm, 1.0e-6
    )
    return CartesianPoseProjectionJaxResult(
        qvel=selected,
        correction_norm_rad_s=correction_norm,
        retained_fraction=retained_fraction,
        intervened=correction_norm > 1.0e-6,
        max_violation=max_violation,
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

    position_m: np.ndarray
    rotation_matrix: np.ndarray
    predicted_ball_position_at_contact_m: np.ndarray
    time_to_contact_s: np.ndarray
    incoming_ball_velocity_m_s: np.ndarray
    next_crossing_target_xy_m: np.ndarray
    desired_post_ball_velocity_m_s: np.ndarray
    valid: np.ndarray


def propagate_ballistic_state_to_age(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    age_s: np.ndarray,
    *,
    gravity_m_s2: float | np.ndarray = 9.8,
) -> tuple[np.ndarray, np.ndarray]:
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
    position = np.asarray(position_m, dtype=np.float32)
    velocity = np.asarray(velocity_m_s, dtype=np.float32)
    age = np.maximum(np.asarray(age_s, dtype=np.float32), 0.0)
    while age.ndim < position.ndim:
        age = age[..., None]
    gravity_vector = np.asarray([0.0, 0.0, -gravity], dtype=np.float32)
    return (
        position + velocity * age + 0.5 * gravity_vector * age**2,
        velocity + gravity_vector * age,
    )


def bounded_next_crossing_target_xy(
    contact_xy_m: np.ndarray,
    *,
    workspace_center_xy: tuple[float, float],
    max_lateral_displacement_m: float,
) -> np.ndarray:
    """Move toward center by at most one declared lateral step per bounce."""

    contact_xy = np.asarray(contact_xy_m, dtype=np.float32)
    center = np.asarray(workspace_center_xy, dtype=np.float32)
    correction = center[None, :] - contact_xy
    correction_norm = np.linalg.norm(correction, axis=-1, keepdims=True)
    bounded_correction = correction * np.minimum(
        1.0,
        float(max_lateral_displacement_m)
        / np.maximum(correction_norm, 1.0e-6),
    )
    return contact_xy + bounded_correction


def predict_descending_plane_crossing(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    center_height_m: float,
    *,
    gravity_m_s2: float = 9.8,
    max_time_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the descending ballistic crossing used by the actor target."""

    position = np.asarray(position_m, dtype=np.float32)
    velocity = np.asarray(velocity_m_s, dtype=np.float32)
    discriminant = velocity[:, 2] ** 2 + 2.0 * gravity_m_s2 * (
        position[:, 2] - float(center_height_m)
    )
    root = (velocity[:, 2] + np.sqrt(np.maximum(discriminant, 0.0))) / float(
        gravity_m_s2
    )
    valid = (discriminant >= 0.0) & (root >= 0.0) & (root <= max_time_s)
    crossing_xy = position[:, :2] + velocity[:, :2] * root[:, None]
    return crossing_xy, np.where(valid, root, 0.0), valid


def _rotation_from_normal(
    target_normal: np.ndarray,
    reference_rotation: np.ndarray,
) -> np.ndarray:
    """Build a full pose while preserving the reference in-plane heading."""

    normal = np.asarray(target_normal, dtype=np.float32)
    reference = np.asarray(reference_rotation, dtype=np.float32)
    if reference.ndim == 2:
        reference = np.broadcast_to(reference, (normal.shape[0], 3, 3))
    reference_x = reference[:, :, 0]
    reference_y = reference[:, :, 1]
    projected_x = reference_x - (
        np.sum(reference_x * normal, axis=-1, keepdims=True) * normal
    )
    fallback_x = reference_y - (
        np.sum(reference_y * normal, axis=-1, keepdims=True) * normal
    )
    projected_norm = np.linalg.norm(projected_x, axis=-1, keepdims=True)
    projected_x = np.where(
        projected_norm > 1.0e-6,
        projected_x,
        fallback_x,
    )
    target_x = projected_x / np.maximum(
        np.linalg.norm(projected_x, axis=-1, keepdims=True), 1.0e-6
    )
    target_y = np.cross(normal, target_x)
    target_y = target_y / np.maximum(
        np.linalg.norm(target_y, axis=-1, keepdims=True), 1.0e-6
    )
    target_x = np.cross(target_y, normal)
    return np.stack([target_x, target_y, normal], axis=-1)


def compute_dynamic_contact_target(
    ball_position_m: np.ndarray,
    ball_velocity_m_s: np.ndarray,
    reference_rotation: np.ndarray,
    *,
    workspace_center_xy: tuple[float, float],
    contact_surface_z_m: float,
    ball_radius_m: float,
    cfg: ContactTargetConfig = ContactTargetConfig(),
    target_apex_z_m: np.ndarray | None = None,
) -> ContactTarget:
    """Compute an observable-only pose target for the next racket contact.

    The target position is the represented ball trajectory's descending
    crossing of the fixed contact plane.  Its normal follows the bounded
    impulse direction required to move the next same-height crossing toward
    the registered workspace center by no more than one bounded lateral step.
    This is a geometric nominal target, not a claim of an identified collision
    inverse.
    """

    ball_position = np.asarray(ball_position_m, dtype=np.float32)
    ball_velocity = np.asarray(ball_velocity_m_s, dtype=np.float32)
    reference = np.asarray(reference_rotation, dtype=np.float32)
    if reference.ndim == 2:
        reference = np.broadcast_to(
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
    target_xy = np.where(
        crossing_valid[:, None], crossing_xy, ball_position[:, :2]
    )
    target_position = np.concatenate(
        [
            target_xy,
            np.full(
                (ball_position.shape[0], 1),
                float(contact_surface_z_m),
                dtype=np.float32,
            ),
        ],
        axis=-1,
    )
    predicted_ball_position_at_contact = np.concatenate(
        [
            target_xy,
            np.full(
                (ball_position.shape[0], 1),
                ball_contact_z_m,
                dtype=np.float32,
            ),
        ],
        axis=-1,
    )

    gravity = float(cfg.gravity_m_s2)
    incoming_velocity = ball_velocity.copy()
    incoming_velocity[:, 2] += -gravity * time_to_contact
    if target_apex_z_m is None:
        target_apex = np.full(
            (ball_position.shape[0],),
            float(cfg.target_apex_z_m),
            dtype=np.float32,
        )
    else:
        target_apex = np.asarray(target_apex_z_m, dtype=np.float32)
        if target_apex.shape != (ball_position.shape[0],):
            raise ValueError("target_apex_z_m must have shape (N,)")
    target_apex_lift_m = np.maximum(0.0, target_apex - ball_contact_z_m)
    desired_post_vz = np.sqrt(
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
    ) / np.maximum(same_height_flight_s[:, None], 1.0e-6)
    desired_post_vxy_norm = np.linalg.norm(
        desired_post_vxy, axis=-1, keepdims=True
    )
    desired_post_vxy = desired_post_vxy * np.minimum(
        1.0,
        float(cfg.max_post_ball_vxy_m_s)
        / np.maximum(desired_post_vxy_norm, 1.0e-6),
    )
    desired_post_velocity = np.concatenate(
        [
            desired_post_vxy,
            desired_post_vz[:, None],
        ],
        axis=-1,
    )

    impulse_direction = desired_post_velocity - incoming_velocity
    impulse_normal = impulse_direction / np.maximum(
        np.linalg.norm(impulse_direction, axis=-1, keepdims=True), 1.0e-6
    )
    impulse_normal = np.where(
        impulse_normal[:, 2:3] >= 0.0,
        impulse_normal,
        -impulse_normal,
    )
    tangent_slope = impulse_normal[:, :2] / np.maximum(
        impulse_normal[:, 2:3], 1.0e-3
    )
    tangent_limit = np.tan(np.deg2rad(float(cfg.max_tilt_deg)))
    tangent_norm = np.linalg.norm(tangent_slope, axis=-1, keepdims=True)
    tangent_slope = tangent_slope * np.minimum(
        1.0,
        tangent_limit / np.maximum(tangent_norm, 1.0e-6),
    )
    target_normal = np.concatenate(
        [
            tangent_slope,
            np.ones((ball_position.shape[0], 1), dtype=np.float32),
        ],
        axis=-1,
    )
    target_normal = target_normal / np.maximum(
        np.linalg.norm(target_normal, axis=-1, keepdims=True), 1.0e-6
    )
    target_rotation = _rotation_from_normal(target_normal, reference)
    finite = (
        np.all(np.isfinite(ball_position), axis=-1)
        & np.all(np.isfinite(ball_velocity), axis=-1)
        & np.all(np.isfinite(target_position), axis=-1)
        & np.all(np.isfinite(target_rotation), axis=(-2, -1))
    )
    valid = crossing_valid & finite & (target_apex_lift_m > 0.0)
    target_rotation = np.where(
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

def rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    """Return the first two racket-frame axes from an (N,3,3) matrix."""
    matrix = np.asarray(rotation, dtype=np.float32)
    if matrix.ndim != 3 or matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape (N,3,3)")
    return np.swapaxes(matrix[:, :, :2], -1, -2).reshape(matrix.shape[0], 6)


def advance_position_history(
    *, ball_positions: np.ndarray, racket_poses: np.ndarray,
    ball_ages: np.ndarray, ball_position_now: np.ndarray,
    racket_pose_now: np.ndarray, ball_age_now: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one 200 Hz tick with held camera position and current FK pose."""
    return (
        np.concatenate(
            (ball_position_now[:, None, :], ball_positions[:, :-1, :]), axis=1
        ),
        np.concatenate((racket_pose_now[:, None, :], racket_poses[:, :-1, :]), axis=1),
        np.concatenate((ball_age_now[:, None], ball_ages[:, :-1]), axis=1),
    )


def make_history_actor_observation(
    *, q: np.ndarray, dq: np.ndarray, q_ref_error: np.ndarray,
    previous_action: np.ndarray, ball_positions: np.ndarray,
    racket_poses: np.ndarray, ball_ages: np.ndarray,
    desired_apex_z_m: np.ndarray, target_xy_m: np.ndarray,
    hit_count: np.ndarray, ball_valid: np.ndarray,
    lags_ticks: tuple[int, ...] = HISTORY_LAGS_TICKS,
) -> np.ndarray:
    """Build the 137-D actor input from positions, poses and proprioception.

    The arguments deliberately contain no measured, predicted or desired ball
    velocity. The only time-series inputs are causal held camera positions,
    measured racket FK poses, and corresponding camera-sample ages.
    """
    ball = np.asarray(ball_positions, dtype=np.float32)
    racket = np.asarray(racket_poses, dtype=np.float32)
    ages = np.asarray(ball_ages, dtype=np.float32)
    n_envs = np.asarray(q).shape[0]
    if (
        len(lags_ticks) != 8
        or lags_ticks[0] != 0
        or tuple(sorted(set(lags_ticks))) != lags_ticks
    ):
        raise ValueError(
            "history lags must be eight increasing ticks starting at zero"
        )
    history_length = lags_ticks[-1] + 1
    if ball.shape != (n_envs, history_length, 3):
        raise ValueError("ball_positions history shape mismatch")
    if racket.shape != (n_envs, history_length, 9):
        raise ValueError("racket_poses history shape mismatch")
    if ages.shape != (n_envs, history_length):
        raise ValueError("ball_ages history shape mismatch")
    lags = np.asarray(lags_ticks, dtype=np.int32)
    output = np.concatenate((
        q, dq, q_ref_error, previous_action,
        ball[:, lags, :].reshape(n_envs, 24),
        racket[:, lags, :].reshape(n_envs, 72),
        np.clip(ages[:, lags] / 0.25, 0.0, 1.0),
        desired_apex_z_m[:, None], target_xy_m,
        np.asarray(ball_valid, dtype=np.float32)[:, None],
        np.clip(np.asarray(hit_count, dtype=np.float32)[:, None] / 4.0, 0.0, 1.0),
    ), axis=-1)
    if output.shape != (n_envs, ACTOR_OBS_DIM):
        raise ValueError("history actor observation dimension mismatch")
    return output

def make_predicted_contact_context(
    *, predicted_ball_position_m: np.ndarray,
    time_to_contact_s: np.ndarray, contact_valid: np.ndarray,
    racket_target_position_m: np.ndarray,
    racket_target_rotation: np.ndarray,
    racket_surface_position_m: np.ndarray,
    next_crossing_xy_m: np.ndarray,
) -> np.ndarray:
    """Return 19-D future geometry/time; no ball-velocity or speed-target slots."""
    ball = np.asarray(predicted_ball_position_m, dtype=np.float32)
    target = np.asarray(racket_target_position_m, dtype=np.float32)
    surface = np.asarray(racket_surface_position_m, dtype=np.float32)
    crossing = np.asarray(next_crossing_xy_m, dtype=np.float32)
    rotation = np.asarray(racket_target_rotation, dtype=np.float32)
    n_envs = ball.shape[0]
    if (
        ball.shape != (n_envs, 3)
        or target.shape != (n_envs, 3)
        or surface.shape != (n_envs, 3)
        or crossing.shape != (n_envs, 2)
        or rotation.shape != (n_envs, 3, 3)
        or np.shape(time_to_contact_s) != (n_envs,)
        or np.shape(contact_valid) != (n_envs,)
    ):
        raise ValueError("V93 contact context input shape mismatch")
    return np.concatenate((
        ball,
        np.clip(np.asarray(time_to_contact_s), 0.0, 2.0)[:, None],
        np.asarray(contact_valid, dtype=np.float32)[:, None],
        target,
        rotation_matrix_to_6d(rotation),
        target - surface,
        crossing,
    ), axis=-1)

def make_simple_position_observation(
    *, q: np.ndarray, dq: np.ndarray, q_ref_error: np.ndarray,
    previous_action: np.ndarray, ball_positions: np.ndarray,
    racket_pose_now: np.ndarray, ball_ages: np.ndarray,
    target_xy_m: np.ndarray, ball_position_valid: np.ndarray,
) -> np.ndarray:
    """Build (N,72) from seven-joint encoder/control state and position samples.

    History order is current to oldest, with eight consecutive 5 ms policy
    ticks. A 90 Hz camera sample is held between arrivals; ages describe the
    represented samples. Racket pose contains XYZ and the first two rotation
    axes, derived from current encoders. No phase, impact count, ball speed,
    forecast, true ball state, or RMP-internal state is consumed.
    """
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 7:
        raise ValueError("q must have shape (N,7)")
    n = q.shape[0]
    inputs = (
        (dq, (n, 7)), (q_ref_error, (n, 7)),
        (previous_action, (n, 7)), (ball_positions, (n, 8, 3)),
        (racket_pose_now, (n, 9)), (ball_ages, (n, 8)),
        (target_xy_m, (n, 2)), (ball_position_valid, (n,)),
    )
    if any(np.shape(value) != shape for value, shape in inputs):
        raise ValueError("V149 simple actor input shape mismatch")
    return np.concatenate((
        q, np.asarray(dq), np.asarray(q_ref_error),
        np.asarray(previous_action), np.asarray(ball_positions).reshape(n, 24),
        np.asarray(racket_pose_now),
        np.clip(np.asarray(ball_ages) / 0.25, 0.0, 1.0),
        np.asarray(target_xy_m),
        np.asarray(ball_position_valid, dtype=np.float32)[:, None],
    ), axis=-1).astype(np.float32)


def make_v224_history_context(
    prefix_history: np.ndarray, *, step_count: int, masked: bool = False,
) -> np.ndarray:
    """Build 45-D observed local history after the V149 72-D prefix.

    The current prefix is history[:, 0]. At each 50/100/200 ms lag, use
    observed ball displacement, observed normal change, past q_ref error,
    past normalized source age and validity. Both vector differences are
    expressed in the current observed racket frame; neither is a velocity.
    See the simcode10 six-model archive report's V224 history contract.
    """
    history = np.asarray(prefix_history, dtype=np.float32)
    if history.ndim != 3 or history.shape[1:] != (V224_PREFIX_HISTORY_LENGTH, 72):
        raise ValueError("V224 prefix history must have shape (N,41,72)")
    current = history[:, 0]
    first_axis, second_axis = current[:, 55:58], current[:, 58:61]
    normal = np.cross(first_axis, second_axis)
    rotation = np.stack((first_axis, second_axis, normal), axis=-1)
    blocks = []
    for lag in V224_HISTORY_LAGS_TICKS:
        past = history[:, lag]
        valid = (not masked and step_count >= lag) & (current[:, 71] > .5) & (past[:, 71] > .5)
        past_normal = np.cross(past[:, 55:58], past[:, 58:61])
        delta_ball = np.einsum('nji,nj->ni', rotation, current[:, 28:31] - past[:, 28:31])
        delta_normal = np.einsum('nji,nj->ni', rotation, normal - past_normal)
        block = np.concatenate((delta_ball, delta_normal, past[:, 14:21],
                                past[:, 61:62], valid[:, None].astype(np.float32)), axis=-1)
        blocks.append(np.where(valid[:, None], block, np.float32(0)))
    return np.concatenate(blocks, axis=-1).astype(np.float32)
