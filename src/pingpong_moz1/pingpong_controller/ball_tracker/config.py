"""Configuration for the hybrid ball-state estimator.

The estimator uses SI units internally: metres, seconds, metres/second, and
metres/second squared.  Keeping the units explicit here prevents camera-space
millimetres from being mixed with the RL policy's metre-based observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np


def _triple(
        value: Any, name: str, *, positive: bool = False,
) -> Tuple[float, ...]:
    """Return a validated three-vector as an immutable tuple."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain three finite values')
    if positive and np.any(array <= 0.0):
        raise ValueError(f'{name} values must be positive')
    return tuple(float(item) for item in array)


@dataclass
class BallTrackerConfig:
    """Tunable parameters for :class:`BallEKFTracker`.

    All distance-related parameters use metres.  ``acceleration_process_std``
    is the standard deviation of continuous white jerk driving the slowly
    varying acceleration state.
    """

    gravity_m_s2: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    measurement_std_m: Tuple[float, float, float] = (0.003, 0.003, 0.005)
    acceleration_process_std_m_s3: Tuple[float, float, float] = (
        6.0, 6.0, 10.0)

    initial_position_std_m: float = 0.005
    initial_velocity_std_m_s: float = 2.0
    initial_acceleration_std_m_s2: float = 2.0
    innovation_gate_chi2: float = 16.27

    max_prediction_duration_s: float = 0.200
    max_predict_step_s: float = 0.020

    vertical_axis: int = 2
    contact_distance_threshold_m: float = 0.050
    contact_candidate_timeout_s: float = 0.120
    min_descending_speed_m_s: float = 0.050
    impact_delta_velocity_threshold_m_s: float = 2.0
    min_post_impact_upward_speed_m_s: float = 0.150
    post_impact_velocity_std_m_s: float = 0.50
    post_impact_mode_duration_s: float = 0.040

    impact_prediction_max_time_s: float = 0.600
    impact_min_approach_speed_m_s: float = 0.010
    impact_plane_tolerance_m: float = 1.0e-5

    covariance_eigenvalue_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        self.gravity_m_s2 = _triple(self.gravity_m_s2, 'gravity_m_s2')
        self.measurement_std_m = _triple(
            self.measurement_std_m, 'measurement_std_m', positive=True)
        self.acceleration_process_std_m_s3 = _triple(
            self.acceleration_process_std_m_s3,
            'acceleration_process_std_m_s3',
            positive=True,
        )
        if self.vertical_axis not in (0, 1, 2):
            raise ValueError('vertical_axis must be 0, 1, or 2')

        positive_names = (
            'initial_position_std_m',
            'initial_velocity_std_m_s',
            'initial_acceleration_std_m_s2',
            'max_prediction_duration_s',
            'max_predict_step_s',
            'contact_distance_threshold_m',
            'contact_candidate_timeout_s',
            'impact_delta_velocity_threshold_m_s',
            'post_impact_velocity_std_m_s',
            'post_impact_mode_duration_s',
            'impact_prediction_max_time_s',
            'impact_plane_tolerance_m',
            'covariance_eigenvalue_floor',
        )
        for name in positive_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive')
            setattr(self, name, value)

        nonnegative_names = (
            'innovation_gate_chi2',
            'min_descending_speed_m_s',
            'min_post_impact_upward_speed_m_s',
            'impact_min_approach_speed_m_s',
        )
        for name in nonnegative_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f'{name} must be finite and non-negative')
            setattr(self, name, value)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> 'BallTrackerConfig':
        """Build a config from a mapping and reject misspelled keys."""
        data = dict(values)
        if 'ball_tracker' in data:
            nested = data['ball_tracker']
            if not isinstance(nested, Mapping):
                raise ValueError('ball_tracker YAML section must be a mapping')
            data = dict(nested)
        valid = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(
                'Unknown ball tracker configuration key(s): '
                + ', '.join(unknown))
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'BallTrackerConfig':
        """Load a configuration from a YAML file."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            message = 'PyYAML is required to load ball tracker configuration'
            raise RuntimeError(message) from exc
        config_path = Path(path)
        with config_path.open('r', encoding='utf-8') as stream:
            values = yaml.safe_load(stream) or {}
        if not isinstance(values, Mapping):
            raise ValueError('Ball tracker YAML root must be a mapping')
        return cls.from_mapping(values)

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly copy of the configuration."""
        result = asdict(self)
        for key in (
                'gravity_m_s2', 'measurement_std_m',
                'acceleration_process_std_m_s3'):
            result[key] = list(result[key])
        return result
