#!/usr/bin/env python3
"""Shared SE(3) utilities for mocap ↔ robot calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np


def fit_rigid_transform(
        points_src: np.ndarray,
        points_dst: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a rigid transform T_dst_src from paired 3D points.

    Solves ``points_dst ≈ R @ points_src + t`` using SVD (Kabsch).

    Args:
        points_src: Source points, shape (N, 3) or (3, N).
        points_dst: Destination points, same shape as points_src.

    Returns:
        (T_dst_src, rmse_m) where T_dst_src is 4x4 and rmse is in meters.
    """
    P_src = np.asarray(points_src, dtype=np.float64)
    P_dst = np.asarray(points_dst, dtype=np.float64)
    if P_src.ndim != 2 or P_dst.ndim != 2:
        raise ValueError('Point arrays must be 2D')
    if P_src.shape != P_dst.shape:
        raise ValueError(
            f'Point arrays must have the same shape, got {P_src.shape} vs '
            f'{P_dst.shape}')
    if P_src.shape[0] == 3 and P_src.shape[1] != 3:
        P_src = P_src.T
        P_dst = P_dst.T
    if P_src.shape[1] != 3:
        raise ValueError(f'Expected (N, 3) or (3, N) points, got {P_src.shape}')
    if P_src.shape[0] < 3:
        raise ValueError('Need at least 3 point pairs')

    centroid_src = np.mean(P_src, axis=0)
    centroid_dst = np.mean(P_dst, axis=0)
    src_centered = P_src - centroid_src
    dst_centered = P_dst - centroid_dst

    H = dst_centered.T @ src_centered
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    t = centroid_dst - R @ centroid_src
    transformed = (R @ P_src.T).T + t
    residual = P_dst - transformed
    rmse = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, rmse


def pose_msg_to_matrix(position: Sequence[float],
                       orientation_xyzw: Sequence[float]) -> np.ndarray:
    """Build a 4x4 pose matrix from position and quaternion (x, y, z, w)."""
    qx, qy, qz, qw = (float(orientation_xyzw[0]), float(orientation_xyzw[1]),
                      float(orientation_xyzw[2]), float(orientation_xyzw[3]))
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),
         2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
         1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def apply_tool_offset(T_parent: np.ndarray,
                      offset_parent: Iterable[float]) -> np.ndarray:
    """Return the 3D position of a tool point expressed in the parent frame."""
    offset_h = np.ones(4, dtype=np.float64)
    offset_h[:3] = np.asarray(offset_parent, dtype=np.float64).reshape(3)
    return (T_parent @ offset_h)[:3]


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to a 3D point."""
    p_h = np.ones(4, dtype=np.float64)
    p_h[:3] = np.asarray(point, dtype=np.float64).reshape(3)
    return (T @ p_h)[:3]


def pose_wxyz_to_matrix(position: Sequence[float],
                        orientation_wxyz: Sequence[float]) -> np.ndarray:
    """Build ``T_parent_child`` from xyz and a (w, x, y, z) quaternion."""
    qw, qx, qy, qz = (
        float(orientation_wxyz[0]), float(orientation_wxyz[1]),
        float(orientation_wxyz[2]), float(orientation_wxyz[3]))
    norm = float(np.linalg.norm([qw, qx, qy, qz]))
    if norm < 1e-12:
        raise ValueError('Quaternion norm must be non-zero')
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),
         2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),
         2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
         1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return T


def _as_transform(T: np.ndarray) -> np.ndarray:
    result = np.asarray(T, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError(f'Expected a 4x4 transform, got {result.shape}')
    if not np.all(np.isfinite(result)):
        raise ValueError('Transform contains non-finite values')
    return result


def _project_to_rotation(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64).reshape(3, 3))
    projected = U @ Vt
    if np.linalg.det(projected) < 0.0:
        U[:, -1] *= -1.0
        projected = U @ Vt
    return projected


def average_rigid_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average translations and project the mean rotation back onto SO(3)."""
    if not transforms:
        raise ValueError('Need at least one transform to average')
    matrices = [_as_transform(T) for T in transforms]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _project_to_rotation(
        np.mean([T[:3, :3] for T in matrices], axis=0))
    result[:3, 3] = np.mean([T[:3, 3] for T in matrices], axis=0)
    return result


def average_quaternions_wxyz(
        quaternions_wxyz: Sequence[Sequence[float]]) -> np.ndarray:
    """Return a sign-invariant mean unit quaternion in ``(w, x, y, z)`` order.

    Unit quaternions ``q`` and ``-q`` encode the same rotation.  A direct
    component-wise mean can therefore collapse toward zero when a mocap stream
    changes quaternion sign.  The Markley outer-product mean used here is
    invariant to that sign ambiguity.
    """
    if not quaternions_wxyz:
        raise ValueError('Need at least one quaternion to average')
    quaternions = []
    for raw in quaternions_wxyz:
        q = np.asarray(raw, dtype=np.float64).reshape(4)
        if not np.all(np.isfinite(q)):
            raise ValueError('Quaternion contains non-finite values')
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            raise ValueError('Quaternion norm must be non-zero')
        quaternions.append(q / norm)
    accumulator = sum(
        np.outer(q, q) for q in quaternions) / float(len(quaternions))
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    mean_q = eigenvectors[:, int(np.argmax(eigenvalues))]
    # Choose a deterministic sign close to the first packet for readable logs.
    if float(np.dot(mean_q, quaternions[0])) < 0.0:
        mean_q = -mean_q
    return mean_q / np.linalg.norm(mean_q)


def quaternion_spread_deg(
        quaternions_wxyz: Sequence[Sequence[float]],
        mean_wxyz: Sequence[float] | None = None) -> Tuple[float, float]:
    """Return RMS/max geodesic angular deviation from the mean quaternion."""
    if not quaternions_wxyz:
        raise ValueError('Need at least one quaternion')
    mean_q = (
        average_quaternions_wxyz(quaternions_wxyz)
        if mean_wxyz is None
        else np.asarray(mean_wxyz, dtype=np.float64).reshape(4))
    mean_norm = float(np.linalg.norm(mean_q))
    if mean_norm < 1e-12:
        raise ValueError('Mean quaternion norm must be non-zero')
    mean_q = mean_q / mean_norm
    angles_deg = []
    for raw in quaternions_wxyz:
        q = np.asarray(raw, dtype=np.float64).reshape(4)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-12:
            raise ValueError('Quaternion norm must be non-zero')
        q = q / q_norm
        # abs(dot) accounts for q/-q equivalence. Quaternion-space angle is
        # half the SO(3) rotation angle.
        angle_rad = 2.0 * np.arccos(
            np.clip(abs(float(np.dot(q, mean_q))), 0.0, 1.0))
        angles_deg.append(float(np.degrees(angle_rad)))
    angles = np.asarray(angles_deg, dtype=np.float64)
    return (
        float(np.sqrt(np.mean(angles ** 2))),
        float(np.max(angles)),
    )


def _rotation_angle_deg(R: np.ndarray) -> float:
    cos_angle = (float(np.trace(R)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def evaluate_eye_to_hand_pairs(
        T_waist03_ee_list: Sequence[np.ndarray],
        T_mocap_rigid_list: Sequence[np.ndarray],
        T_waist03_mocap: np.ndarray,
        T_ee_rigid: np.ndarray) -> dict:
    """Evaluate ``T_mocap_rigid = inv(T_waist03_mocap) @ T_waist03_ee @ T_ee_rigid``."""
    if len(T_waist03_ee_list) != len(T_mocap_rigid_list):
        raise ValueError('Robot and mocap pose lists must have the same length')
    if not T_waist03_ee_list:
        raise ValueError('Need at least one pose pair')

    T_w_m = _as_transform(T_waist03_mocap)
    T_e_k = _as_transform(T_ee_rigid)
    translation_errors_m = []
    rotation_errors_deg = []
    residuals = []
    T_m_w = np.linalg.inv(T_w_m)
    for T_w_e_raw, T_m_k_raw in zip(
            T_waist03_ee_list, T_mocap_rigid_list):
        T_w_e = _as_transform(T_w_e_raw)
        T_m_k = _as_transform(T_m_k_raw)
        predicted = T_m_w @ T_w_e @ T_e_k
        delta = np.linalg.inv(T_m_k) @ predicted
        translation_m = float(np.linalg.norm(delta[:3, 3]))
        rotation_deg = _rotation_angle_deg(delta[:3, :3])
        translation_errors_m.append(translation_m)
        rotation_errors_deg.append(rotation_deg)
        residuals.append({
            'translation_m': translation_m,
            'translation_mm': translation_m * 1000.0,
            'rotation_deg': rotation_deg,
        })

    translation_arr = np.asarray(translation_errors_m, dtype=np.float64)
    rotation_arr = np.asarray(rotation_errors_deg, dtype=np.float64)
    return {
        'rmse_m': float(np.sqrt(np.mean(translation_arr ** 2))),
        'mean_error_m': float(np.mean(translation_arr)),
        'max_error_m': float(np.max(translation_arr)),
        'std_error_m': float(np.std(translation_arr)),
        'per_sample_error_mm': [float(v * 1000.0) for v in translation_arr],
        'rotation_mean_deg': float(np.mean(rotation_arr)),
        'rotation_max_deg': float(np.max(rotation_arr)),
        'rotation_std_deg': float(np.std(rotation_arr)),
        'per_sample_rotation_deg': [float(v) for v in rotation_arr],
        'residuals': residuals,
    }


def fit_eye_to_hand_transform(
        T_waist03_ee_list: Sequence[np.ndarray],
        T_mocap_rigid_list: Sequence[np.ndarray],
        method: str = 'PARK') -> Tuple[np.ndarray, np.ndarray, dict]:
    """Solve fixed mocap and tool extrinsics from complete pose pairs.

    The equation matches the repository's eye-to-hand calibration:

    ``inv(T_waist03_ee[i]) @ T_waist03_mocap @ T_mocap_rigid[i]``
    is the fixed ``T_ee_rigid`` for every sample ``i``.

    Returns ``(T_waist03_mocap, T_ee_rigid, residual_metrics)``.
    """
    if len(T_waist03_ee_list) != len(T_mocap_rigid_list):
        raise ValueError('Robot and mocap pose lists must have the same length')
    if len(T_waist03_ee_list) < 3:
        raise ValueError('Need at least 3 complete pose pairs')
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            'Full SE(3) mocap calibration requires OpenCV (cv2)') from exc

    method_name = str(method).strip().upper()
    method_map = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    if method_name not in method_map:
        raise ValueError(
            f'Unsupported hand-eye method {method!r}; choose one of '
            f'{", ".join(method_map)}')

    robot_poses = [_as_transform(T) for T in T_waist03_ee_list]
    mocap_poses = [_as_transform(T) for T in T_mocap_rigid_list]
    T_ee_waist03_list = [np.linalg.inv(T) for T in robot_poses]
    R_ee_waist03_list = [T[:3, :3] for T in T_ee_waist03_list]
    t_ee_waist03_list = [T[:3, 3].reshape(3, 1) for T in T_ee_waist03_list]
    R_rigid_mocap_list = [T[:3, :3] for T in mocap_poses]
    t_rigid_mocap_list = [T[:3, 3].reshape(3, 1) for T in mocap_poses]

    R_waist03_mocap, opencv_t_waist03_mocap = cv2.calibrateHandEye(
        R_ee_waist03_list,
        t_ee_waist03_list,
        R_rigid_mocap_list,
        t_rigid_mocap_list,
        method=method_map[method_name],
    )
    if (not np.all(np.isfinite(R_waist03_mocap))
            or not np.all(np.isfinite(opencv_t_waist03_mocap))):
        raise ValueError(
            'Hand-eye solve returned an invalid transform. Collect poses with '
            'rotations about multiple axes; pure translation or nearly '
            'parallel rotations are degenerate.')
    # calibrateHandEye determines the fixed mocap-frame rotation from relative
    # motions.  Its translation and a subsequently averaged tool translation
    # do not, in the presence of noise, jointly minimize the absolute closure
    # residual reported by evaluate_eye_to_hand_pairs.  Keep the solved
    # rotation, estimate the fixed tool rotation, then solve both translations
    # in one linear least-squares system:
    #
    #   R_w_m p_m_k(i) + t_w_m
    #       = R_w_e(i) t_e_k + p_w_e(i).
    #
    # This also makes the saved calibration consistent with its RMSE metric.
    R_waist03_mocap = _project_to_rotation(R_waist03_mocap)
    T_ee_rigid_rotation_samples = []
    for T_w_e, T_m_k in zip(robot_poses, mocap_poses):
        T_sample = np.eye(4, dtype=np.float64)
        T_sample[:3, :3] = (
            T_w_e[:3, :3].T @ R_waist03_mocap @ T_m_k[:3, :3])
        T_ee_rigid_rotation_samples.append(T_sample)
    R_ee_rigid = average_rigid_transforms(
        T_ee_rigid_rotation_samples)[:3, :3]

    translation_design = []
    translation_rhs = []
    for T_w_e, T_m_k in zip(robot_poses, mocap_poses):
        translation_design.append(np.hstack((
            np.eye(3, dtype=np.float64), -T_w_e[:3, :3])))
        translation_rhs.append(
            T_w_e[:3, 3] - R_waist03_mocap @ T_m_k[:3, 3])
    translation_solution, _residuals, rank, singular_values = np.linalg.lstsq(
        np.vstack(translation_design),
        np.hstack(translation_rhs),
        rcond=None,
    )
    if rank < 6:
        raise ValueError(
            'Hand-eye translation solve is rank deficient. Collect poses with '
            'rotations about multiple axes.')

    T_waist03_mocap = np.eye(4, dtype=np.float64)
    T_waist03_mocap[:3, :3] = R_waist03_mocap
    T_waist03_mocap[:3, 3] = translation_solution[:3]
    T_ee_rigid = np.eye(4, dtype=np.float64)
    T_ee_rigid[:3, :3] = R_ee_rigid
    T_ee_rigid[:3, 3] = translation_solution[3:]
    metrics = evaluate_eye_to_hand_pairs(
        robot_poses, mocap_poses, T_waist03_mocap, T_ee_rigid)
    metrics['method'] = f'calibrateHandEye eye-to-hand {method_name}'
    metrics['translation_refinement'] = 'joint absolute-pose least squares'
    metrics['translation_design_rank'] = int(rank)
    metrics['translation_design_condition'] = float(
        singular_values[0] / singular_values[-1])
    metrics['opencv_translation_ignored_m'] = np.asarray(
        opencv_t_waist03_mocap, dtype=np.float64).reshape(3).tolist()
    return T_waist03_mocap, T_ee_rigid, metrics


def save_calibration_matrix(path: str | Path,
                          T_robot_mocap: np.ndarray,
                          header: str = 'T_robot_mocap') -> None:
    """Save a 4x4 calibration matrix as whitespace-separated text."""
    output_path = Path(path).resolve()
    np.savetxt(
        output_path,
        T_robot_mocap,
        fmt='%.8f',
        header=header,
    )


def save_calibration_json(path: str | Path, metadata: dict) -> None:
    """Save calibration metadata as JSON."""
    output_path = Path(path).resolve()
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def load_calibration_matrix(path: str | Path) -> np.ndarray:
    """Load a 4x4 calibration matrix from text."""
    T = np.loadtxt(path, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f'Calibration matrix must be 4x4, got {T.shape}')
    return T


def evaluate_point_pairs(
        points_src: np.ndarray,
        points_dst: np.ndarray,
        T_dst_src: np.ndarray) -> dict:
    """Return per-sample and aggregate calibration residuals in meters."""
    P_src = np.asarray(points_src, dtype=np.float64)
    P_dst = np.asarray(points_dst, dtype=np.float64)
    if P_src.ndim != 2 or P_dst.ndim != 2:
        raise ValueError('Point arrays must be 2D')
    if P_src.shape != P_dst.shape:
        raise ValueError('Point arrays must have the same shape')
    if P_src.shape[1] != 3:
        if P_src.shape[0] == 3:
            P_src = P_src.T
            P_dst = P_dst.T
        else:
            raise ValueError(f'Expected (N, 3) points, got {P_src.shape}')

    R = T_dst_src[:3, :3]
    t = T_dst_src[:3, 3]
    predicted = (R @ P_src.T).T + t
    residual = P_dst - predicted
    per_sample_err_m = np.linalg.norm(residual, axis=1)
    rmse_m = float(np.sqrt(np.mean(per_sample_err_m ** 2)))
    return {
        'rmse_m': rmse_m,
        'max_error_m': float(np.max(per_sample_err_m)),
        'mean_error_m': float(np.mean(per_sample_err_m)),
        'std_error_m': float(np.std(per_sample_err_m)),
        'per_sample_error_mm': [
            float(v * 1000.0) for v in per_sample_err_m.tolist()],
        'residuals_m': residual.tolist(),
    }
