"""Nine-state constant-acceleration ball model used by the EKF."""

from __future__ import annotations

from typing import Iterable

import numpy as np


STATE_DIM = 9
POSITION = slice(0, 3)
VELOCITY = slice(3, 6)
ACCELERATION = slice(6, 9)


def _state_vector(state: Iterable[float]) -> np.ndarray:
    vector = np.asarray(state, dtype=float)
    if vector.shape != (STATE_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError('state must be a finite vector with shape (9,)')
    return vector


def _time_step(dt: float) -> float:
    value = float(dt)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError('dt must be finite and non-negative')
    return value


def ball_dynamics(state: Iterable[float], dt: float) -> np.ndarray:
    """Propagate ``[position, velocity, acceleration]`` by ``dt`` seconds.

    Acceleration is part of the state and follows a random walk; consequently
    the deterministic dynamics keep it constant between filter updates.
    """
    vector = _state_vector(state)
    step = _time_step(dt)
    output = vector.copy()
    output[POSITION] = (
        vector[POSITION]
        + vector[VELOCITY] * step
        + 0.5 * vector[ACCELERATION] * step * step
    )
    output[VELOCITY] = vector[VELOCITY] + vector[ACCELERATION] * step
    return output


def jacobian(state: Iterable[float], dt: float) -> np.ndarray:
    """Return the 9x9 Jacobian of :func:`ball_dynamics`."""
    _state_vector(state)
    step = _time_step(dt)
    transition = np.eye(STATE_DIM, dtype=float)
    transition[POSITION, VELOCITY] = np.eye(3) * step
    transition[POSITION, ACCELERATION] = np.eye(3) * (0.5 * step * step)
    transition[VELOCITY, ACCELERATION] = np.eye(3) * step
    return transition


def process_noise(
        dt: float,
        acceleration_process_std_m_s3: Iterable[float],
) -> np.ndarray:
    """Discretize independent continuous white-jerk noise on each axis.

    The exact per-axis covariance for the integrated state ``[p, v, a]`` is
    used, including its cross-covariance terms.  This is more consistent than
    adding unrelated diagonal noise to position, velocity, and acceleration.
    """
    step = _time_step(dt)
    std = np.asarray(acceleration_process_std_m_s3, dtype=float)
    if std.shape != (3,) or not np.all(np.isfinite(std)) or np.any(std < 0.0):
        raise ValueError(
            'acceleration_process_std_m_s3 must be three non-negative values')
    covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    powers = np.array([
        [step ** 5 / 20.0, step ** 4 / 8.0, step ** 3 / 6.0],
        [step ** 4 / 8.0, step ** 3 / 3.0, step ** 2 / 2.0],
        [step ** 3 / 6.0, step ** 2 / 2.0, step],
    ])
    for axis in range(3):
        indices = [axis, axis + 3, axis + 6]
        covariance[np.ix_(indices, indices)] = powers * (std[axis] ** 2)
    return covariance


def propagate_covariance(
        covariance: np.ndarray,
        state: Iterable[float],
        dt: float,
        acceleration_process_std_m_s3: Iterable[float],
) -> np.ndarray:
    """Propagate a state covariance through the ball model."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (STATE_DIM, STATE_DIM):
        raise ValueError('covariance must have shape (9, 9)')
    transition = jacobian(state, dt)
    predicted = transition @ matrix @ transition.T
    predicted += process_noise(dt, acceleration_process_std_m_s3)
    return 0.5 * (predicted + predicted.T)
