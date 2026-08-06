"""Analytic ball/racket-plane impact prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from .ball_model import ACCELERATION, POSITION, STATE_DIM, VELOCITY, jacobian


@dataclass(frozen=True)
class RacketPlane:
    """A planar racket surface with an optional constant linear velocity."""

    position: np.ndarray
    normal: np.ndarray
    velocity: np.ndarray


def _vector3(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must be a finite vector with shape (3,)')
    return vector.copy()


def racket_plane_from_pose(racket_pose: Any) -> RacketPlane:
    """Normalize common racket-pose representations into a plane.

    Accepted inputs are a :class:`RacketPlane`, a mapping containing
    ``position`` and optional ``normal``/``linear_velocity``, a mapping with
    ``z`` or ``z_racket``, a 4x4 transform (local +Z is the normal), or a
    three-element position (horizontal plane).
    """
    if isinstance(racket_pose, RacketPlane):
        position = _vector3(racket_pose.position, 'racket position')
        normal = _vector3(racket_pose.normal, 'racket normal')
        velocity = _vector3(racket_pose.velocity, 'racket velocity')
    elif isinstance(racket_pose, Mapping):
        if 'position' in racket_pose:
            position = _vector3(racket_pose['position'], 'racket position')
        elif 'z_racket' in racket_pose or 'z' in racket_pose:
            height = racket_pose.get('z_racket', racket_pose.get('z'))
            position = np.array([0.0, 0.0, float(height)], dtype=float)
        else:
            raise ValueError(
                'racket pose mapping requires position, z_racket, or z')
        normal = _vector3(
            racket_pose.get('normal', (0.0, 0.0, 1.0)), 'racket normal')
        velocity = _vector3(
            racket_pose.get(
                'linear_velocity',
                racket_pose.get('velocity', (0.0, 0.0, 0.0))),
            'racket velocity',
        )
    else:
        value = np.asarray(racket_pose, dtype=float)
        if value.shape == (4, 4):
            position = value[:3, 3].copy()
            normal = value[:3, 2].copy()
            velocity = np.zeros(3, dtype=float)
        elif value.shape == (3,):
            position = value.copy()
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
            velocity = np.zeros(3, dtype=float)
        else:
            raise ValueError(
                'racket_pose must be a pose mapping, 4x4 transform, '
                'or position')
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError('racket normal must be non-zero')
    return RacketPlane(position, normal / norm, velocity)


def signed_plane_distance(position: Any, racket_pose: Any) -> float:
    """Return signed point-to-racket-plane distance in metres."""
    plane = racket_plane_from_pose(racket_pose)
    point = _vector3(position, 'position')
    return float(np.dot(plane.normal, point - plane.position))


def _invalid_prediction(reason: str) -> dict[str, Any]:
    return {
        'valid': False,
        'reason': reason,
        'time_to_contact': None,
        'contact_time': None,
        'impact_position': None,
        'contact_position': None,
        'impact_velocity': None,
        'contact_velocity': None,
        'time_std_s': None,
        'impact_position_covariance': None,
        'impact_velocity_covariance': None,
    }


def predict_contact(
        state: Any,
        racket_pose: Any,
        *,
        covariance: Optional[np.ndarray] = None,
        max_time_s: float = 0.600,
        min_approach_speed_m_s: float = 0.010,
        plane_tolerance_m: float = 1.0e-5,
        require_approaching: bool = True,
) -> dict[str, Any]:
    """Predict the next intersection of the ball and racket plane.

    The racket plane may translate at constant velocity.  Its normal should
    face the playable side; by default a valid contact must cross that normal
    in the approaching direction.  The returned covariance uses a first-order
    time-of-impact linearization when a 9x9 state covariance is supplied.
    """
    vector = np.asarray(state, dtype=float)
    if vector.shape != (STATE_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError('state must be a finite vector with shape (9,)')
    horizon = float(max_time_s)
    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError('max_time_s must be finite and positive')
    plane = racket_plane_from_pose(racket_pose)

    position = vector[POSITION]
    velocity = vector[VELOCITY]
    acceleration = vector[ACCELERATION]
    relative_position = position - plane.position
    relative_velocity = velocity - plane.velocity
    c0 = float(np.dot(plane.normal, relative_position))
    c1 = float(np.dot(plane.normal, relative_velocity))
    c2 = 0.5 * float(np.dot(plane.normal, acceleration))
    tolerance = max(float(plane_tolerance_m), 0.0)

    candidates: list[float] = []
    if abs(c0) <= tolerance and c1 <= -float(min_approach_speed_m_s):
        candidates.append(0.0)
    if abs(c2) <= 1e-12:
        if abs(c1) > 1e-12:
            candidates.append(-c0 / c1)
    else:
        discriminant = c1 * c1 - 4.0 * c2 * c0
        if discriminant >= 0.0 and np.isfinite(discriminant):
            root = float(np.sqrt(max(discriminant, 0.0)))
            candidates.extend(((-c1 - root) / (2.0 * c2),
                               (-c1 + root) / (2.0 * c2)))

    valid_candidates = []
    for contact_time in candidates:
        if (not np.isfinite(contact_time) or contact_time < -1e-12
                or contact_time > horizon + 1e-12):
            continue
        contact_time = max(0.0, float(contact_time))
        normal_rate = c1 + 2.0 * c2 * contact_time
        if require_approaching:
            if c0 < -tolerance:
                continue
            if normal_rate > -float(min_approach_speed_m_s):
                continue
        valid_candidates.append(contact_time)
    if not valid_candidates:
        return _invalid_prediction('no_approaching_intersection')

    contact_time = min(valid_candidates)
    impact_position = (
        position + velocity * contact_time
        + 0.5 * acceleration * contact_time * contact_time)
    impact_velocity = velocity + acceleration * contact_time

    time_std = None
    position_covariance = None
    velocity_covariance = None
    if covariance is not None:
        matrix = np.asarray(covariance, dtype=float)
        if matrix.shape != (STATE_DIM, STATE_DIM):
            raise ValueError('covariance must have shape (9, 9)')
        normal_rate = float(np.dot(
            plane.normal, impact_velocity - plane.velocity))
        transition = jacobian(vector, contact_time)
        position_jacobian = transition[POSITION, :]
        velocity_jacobian = transition[VELOCITY, :]
        if abs(normal_rate) > 1e-9:
            plane_gradient = np.concatenate([
                plane.normal,
                contact_time * plane.normal,
                0.5 * contact_time * contact_time * plane.normal,
            ])
            time_jacobian = -plane_gradient / normal_rate
            time_variance = float(time_jacobian @ matrix @ time_jacobian.T)
            time_std = float(np.sqrt(max(time_variance, 0.0)))
            position_jacobian = (
                position_jacobian
                + np.outer(impact_velocity, time_jacobian))
            velocity_jacobian = (
                velocity_jacobian
                + np.outer(acceleration, time_jacobian))
        position_covariance = position_jacobian @ matrix @ position_jacobian.T
        velocity_covariance = velocity_jacobian @ matrix @ velocity_jacobian.T
        position_covariance = 0.5 * (
            position_covariance + position_covariance.T)
        velocity_covariance = 0.5 * (
            velocity_covariance + velocity_covariance.T)

    return {
        'valid': True,
        'reason': None,
        'time_to_contact': float(contact_time),
        'contact_time': float(contact_time),
        'impact_position': impact_position.tolist(),
        'contact_position': impact_position.tolist(),
        'impact_velocity': impact_velocity.tolist(),
        'contact_velocity': impact_velocity.tolist(),
        'time_std_s': time_std,
        'impact_position_covariance': (
            None if position_covariance is None
            else position_covariance.tolist()),
        'impact_velocity_covariance': (
            None if velocity_covariance is None
            else velocity_covariance.tolist()),
    }


def predict_impact(
        state: Any, racket_pose: Any, **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias with the API named in the design objective."""
    return predict_contact(state, racket_pose, **kwargs)
