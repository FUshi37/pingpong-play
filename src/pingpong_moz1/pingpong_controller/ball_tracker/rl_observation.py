"""RL observation adapter for backend-neutral ball estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .estimator import BallEstimate


@dataclass(frozen=True)
class BallObservationConfig:
    """Optional fields appended after the required estimator observation."""

    invalid_contact_time_s: float = 0.600
    include_acceleration: bool = False
    include_uncertainty: bool = False
    include_validity: bool = False


def ball_observation_field_names(
        robot_position_count: int,
        robot_velocity_count: int,
        config: BallObservationConfig = BallObservationConfig(),
) -> list[str]:
    """Return names in exactly the order used by ``build_rl_observation``."""
    names = [f'robot_q_{index}' for index in range(robot_position_count)]
    names += [f'robot_dq_{index}' for index in range(robot_velocity_count)]
    names += [
        'ball_px', 'ball_py', 'ball_pz',
        'ball_vx', 'ball_vy', 'ball_vz',
        'time_to_contact',
        'impact_x', 'impact_y', 'impact_z',
    ]
    if config.include_acceleration:
        names += ['ball_ax', 'ball_ay', 'ball_az']
    if config.include_uncertainty:
        names += [
            'ball_px_std', 'ball_py_std', 'ball_pz_std',
            'ball_vx_std', 'ball_vy_std', 'ball_vz_std',
        ]
    if config.include_validity:
        names += ['ball_state_valid', 'impact_prediction_valid']
    return names


def build_rl_observation(
        robot_q: Iterable[float],
        robot_dq: Iterable[float],
        estimate: BallEstimate,
        config: BallObservationConfig = BallObservationConfig(),
) -> np.ndarray:
    """Build the recommended observation without removing current ball state.

    The required ordering is ``robot_q, robot_dq, p, v, t_hit, p_hit``.
    Acceleration, standard deviations, and validity flags may be appended for a
    newly trained policy.  Invalid impact predictions use the configured time
    horizon and the current filtered position, avoiding NaNs in policy input.
    """
    q = np.asarray(robot_q, dtype=float).reshape(-1)
    dq = np.asarray(robot_dq, dtype=float).reshape(-1)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
        raise ValueError('robot_q and robot_dq must contain finite values')

    if estimate.valid:
        position = estimate.position
        velocity = estimate.velocity
        acceleration = estimate.acceleration
        position_std = estimate.position_std
        velocity_std = estimate.velocity_std
    else:
        position = np.zeros(3, dtype=float)
        velocity = np.zeros(3, dtype=float)
        acceleration = np.zeros(3, dtype=float)
        position_std = np.zeros(3, dtype=float)
        velocity_std = np.zeros(3, dtype=float)

    impact = estimate.impact_prediction or {}
    impact_valid = bool(impact.get('valid', False))
    if impact_valid:
        contact_time = float(impact['time_to_contact'])
        impact_position = np.asarray(
            impact['impact_position'], dtype=float).reshape(3)
    else:
        contact_time = float(config.invalid_contact_time_s)
        impact_position = position.copy()

    components = [
        q, dq, position, velocity,
        np.array([contact_time], dtype=float),
        impact_position,
    ]
    if config.include_acceleration:
        components.append(acceleration)
    if config.include_uncertainty:
        components.extend((position_std, velocity_std))
    if config.include_validity:
        components.append(np.array(
            [float(estimate.valid), float(impact_valid)], dtype=float))
    observation = np.concatenate(components).astype(np.float32)
    if not np.all(np.isfinite(observation)):
        raise RuntimeError(
            'Estimator produced a non-finite policy observation')
    return observation


class BallPolicyObservationAdapter:
    """Callable adapter suitable for dependency injection into an RL loop."""

    def __init__(
            self,
            config: BallObservationConfig = BallObservationConfig(),
    ) -> None:
        self.config = config

    def __call__(
            self, robot_q: Iterable[float], robot_dq: Iterable[float],
            estimate: BallEstimate,
    ) -> np.ndarray:
        return build_rl_observation(robot_q, robot_dq, estimate, self.config)
