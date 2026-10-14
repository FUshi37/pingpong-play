"""JAX-native Cartesian pose projection for constrained QVEL training.

The policy and the recovered-RMP boundary remain seven-joint quantities.  The
helpers in this module only project the joint velocity that advances the
persistent policy reference.  They deliberately contain no MuJoCo state so
the environment can obtain exact reference-pose kinematics with ``mjx.forward``
and keep these numerical pieces small and directly testable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CartesianPoseProjectionJaxResult(NamedTuple):
    """Projected batch command and per-lane diagnostics."""

    qvel: jax.Array
    correction_norm_rad_s: jax.Array
    retained_fraction: jax.Array
    intervened: jax.Array
    max_violation: jax.Array


def matrix_to_euler_xyz_jax(rotation: jax.Array) -> jax.Array:
    """Extract XYZ Euler angles from one or more proper rotation matrices."""

    sine_pitch = jnp.clip(rotation[..., 0, 2], -1.0, 1.0)
    pitch = jnp.arcsin(sine_pitch)
    cosine_pitch = jnp.cos(pitch)
    regular = jnp.abs(cosine_pitch) > 1.0e-6
    roll_regular = jnp.arctan2(-rotation[..., 1, 2], rotation[..., 2, 2])
    yaw_regular = jnp.arctan2(-rotation[..., 0, 1], rotation[..., 0, 0])
    roll_singular = jnp.arctan2(rotation[..., 2, 1], rotation[..., 1, 1])
    roll = jnp.where(regular, roll_regular, roll_singular)
    yaw = jnp.where(regular, yaw_regular, 0.0)
    return jnp.stack((roll, pitch, yaw), axis=-1)


def relative_xyz_euler_rate_jacobian_jax(
    relative_rpy_rad: jax.Array,
    reference_rotation_world: jax.Array,
    angular_jacobian_world: jax.Array,
) -> jax.Array:
    """Map joint velocity to XYZ-Euler rates relative to a reference pose."""

    roll = relative_rpy_rad[..., 0]
    pitch = relative_rpy_rad[..., 1]
    zeros = jnp.zeros_like(roll)
    ones = jnp.ones_like(roll)
    spatial_euler_rate = jnp.stack(
        (
            jnp.stack((ones, zeros, jnp.sin(pitch)), axis=-1),
            jnp.stack(
                (
                    zeros,
                    jnp.cos(roll),
                    -jnp.sin(roll) * jnp.cos(pitch),
                ),
                axis=-1,
            ),
            jnp.stack(
                (
                    zeros,
                    jnp.sin(roll),
                    jnp.cos(roll) * jnp.cos(pitch),
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    angular_in_reference = jnp.einsum(
        "...ji,...jk->...ik",
        reference_rotation_world,
        angular_jacobian_world,
    )
    return jnp.linalg.solve(spatial_euler_rate, angular_in_reference)


def _project_halfspaces_dykstra_jax(
    nominal_qvel: jax.Array,
    rows: jax.Array,
    bounds: jax.Array,
    velocity_lower: jax.Array,
    velocity_upper: jax.Array,
    velocity_scale: jax.Array,
    projection_sweeps: int,
) -> tuple[jax.Array, jax.Array]:
    """Project a batch onto linear half-spaces plus its velocity box."""

    batch_size, joint_count = nominal_qvel.shape
    dtype = nominal_qvel.dtype
    scale = jnp.maximum(jnp.abs(velocity_scale), jnp.asarray(1.0e-6, dtype))
    if scale.ndim == 1:
        scale = jnp.broadcast_to(scale[None, :], nominal_qvel.shape)

    identity = jnp.broadcast_to(
        jnp.eye(joint_count, dtype=dtype)[None, :, :],
        (batch_size, joint_count, joint_count),
    )
    all_rows = jnp.concatenate((rows, identity, -identity), axis=1)
    all_bounds = jnp.concatenate(
        (bounds, velocity_upper, -velocity_lower), axis=1
    )
    transformed_rows = all_rows * scale[:, None, :]
    row_norm = jnp.linalg.norm(transformed_rows, axis=-1)
    safe_norm = jnp.maximum(row_norm, jnp.asarray(1.0e-9, dtype))
    unit_rows = transformed_rows / safe_norm[..., None]
    normalized_bounds = all_bounds / safe_norm

    selected = nominal_qvel / scale
    corrections = jnp.zeros_like(unit_rows)
    constraint_count = unit_rows.shape[1]

    def project_one(index, carry):
        current, current_corrections = carry
        constraint_index = index % constraint_count
        row = unit_rows[:, constraint_index, :]
        correction = current_corrections[:, constraint_index, :]
        shifted = current + correction
        violation = jnp.sum(row * shifted, axis=-1) - normalized_bounds[
            :, constraint_index
        ]
        projected = shifted - jnp.maximum(violation, 0.0)[:, None] * row
        current_corrections = current_corrections.at[
            :, constraint_index, :
        ].set(shifted - projected)
        return projected, current_corrections

    selected, _ = jax.lax.fori_loop(
        0,
        int(projection_sweeps) * int(constraint_count),
        project_one,
        (selected, corrections),
    )
    projected_qvel = jnp.clip(selected * scale, velocity_lower, velocity_upper)
    pose_violation = jnp.max(
        jnp.maximum(jnp.einsum("nkj,nj->nk", rows, projected_qvel) - bounds, 0.0),
        axis=-1,
    )
    box_violation = jnp.maximum(
        jnp.max(velocity_lower - projected_qvel, axis=-1),
        jnp.max(projected_qvel - velocity_upper, axis=-1),
    )
    return projected_qvel, jnp.maximum(pose_violation, box_violation)


def project_cartesian_pose_qvel_jax(
    *,
    nominal_qvel: jax.Array,
    position_m: jax.Array,
    relative_rpy_rad: jax.Array,
    translation_jacobian: jax.Array,
    relative_rpy_jacobian: jax.Array,
    position_lower_m: jax.Array,
    position_upper_m: jax.Array,
    roll_pitch_limit_rad: jax.Array,
    velocity_lower: jax.Array,
    velocity_upper: jax.Array,
    velocity_scale: jax.Array,
    position_gain_s: float,
    orientation_gain_s: float,
    projection_sweeps: int,
) -> CartesianPoseProjectionJaxResult:
    """Project QVEL into an XYZ and relative-roll/pitch safe tangent set.

    Yaw is intentionally absent from the orientation rows.  The caller must
    still verify the nonlinear FK candidate before publishing the next
    reference; this function guarantees only the stated linearized set.
    """

    position_lower = jnp.asarray(position_lower_m, dtype=nominal_qvel.dtype)
    position_upper = jnp.asarray(position_upper_m, dtype=nominal_qvel.dtype)
    rp_limit = jnp.asarray(
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
    rows = jnp.concatenate(
        (
            upper_position_rows,
            lower_position_rows,
            rp_jacobian,
            -rp_jacobian,
        ),
        axis=1,
    )
    bounds = jnp.concatenate(
        (
            upper_position_bounds,
            lower_position_bounds,
            upper_rp_bounds,
            lower_rp_bounds,
        ),
        axis=1,
    )
    selected, max_violation = _project_halfspaces_dykstra_jax(
        nominal_qvel,
        rows,
        bounds,
        velocity_lower,
        velocity_upper,
        velocity_scale,
        projection_sweeps,
    )
    correction_norm = jnp.linalg.norm(selected - nominal_qvel, axis=-1)
    nominal_norm = jnp.linalg.norm(nominal_qvel, axis=-1)
    retained_fraction = jnp.linalg.norm(selected, axis=-1) / jnp.maximum(
        nominal_norm, 1.0e-6
    )
    return CartesianPoseProjectionJaxResult(
        qvel=selected,
        correction_norm_rad_s=correction_norm,
        retained_fraction=retained_fraction,
        intervened=correction_norm > 1.0e-6,
        max_violation=max_violation,
    )
