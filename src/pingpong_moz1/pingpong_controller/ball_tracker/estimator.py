"""Backend-neutral Ball State Estimator API.

Controllers should depend on :class:`BallStateEstimator`, not directly on the
EKF.  A future transformer or learned-residual backend can therefore implement
the same small protocol without changing policy observation construction.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

import numpy as np

from .ball_model import ACCELERATION, POSITION, STATE_DIM, VELOCITY
from .config import BallTrackerConfig
from .ekf_tracker import BallEKFTracker, MODE_UNINITIALIZED
from .impact_predictor import predict_contact


@dataclass(frozen=True)
class BallEstimate:
    """Immutable policy-facing estimator result in SI units."""

    timestamp_s: float
    state_mean: np.ndarray
    state_covariance: np.ndarray
    valid: bool
    mode: str
    time_since_measurement_s: float
    measurement_accepted: Optional[bool] = None
    impact_prediction: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.state_mean[POSITION].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state_mean[VELOCITY].copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self.state_mean[ACCELERATION].copy()

    @property
    def position_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(
            np.diag(self.state_covariance)[POSITION], 0.0))

    @property
    def velocity_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(
            np.diag(self.state_covariance)[VELOCITY], 0.0))

    @property
    def acceleration_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(
            np.diag(self.state_covariance)[ACCELERATION], 0.0))

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly output contract."""
        return {
            'timestamp_s': float(self.timestamp_s),
            'valid': bool(self.valid),
            'mode': self.mode,
            'time_since_measurement_s': float(
                self.time_since_measurement_s),
            'measurement_accepted': self.measurement_accepted,
            'state_mean': self.state_mean.tolist(),
            'state_covariance': self.state_covariance.tolist(),
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'acceleration': self.acceleration.tolist(),
            'position_std': self.position_std.tolist(),
            'velocity_std': self.velocity_std.tolist(),
            'acceleration_std': self.acceleration_std.tolist(),
            'impact_prediction': self.impact_prediction,
            'metadata': dict(self.metadata),
        }


class BallEstimatorBackend(Protocol):
    """Protocol a replacement EKF/transformer backend must implement."""

    def update(
            self, position_measurement: Any, timestamp_s: float,
            *, racket_pose: Any = None,
            measurement_covariance: Optional[np.ndarray] = None,
    ) -> BallEstimate:
        ...

    def predict_to(
            self, timestamp_s: float, *, racket_pose: Any = None,
    ) -> BallEstimate:
        ...

    def reset(self) -> None:
        ...

    def diagnostics(self) -> dict[str, Any]:
        ...


class EKFEstimatorBackend:
    """Timestamp-aware adapter around :class:`BallEKFTracker`.

    Control-time predictions are deliberately non-mutating.  A 60 Hz camera
    measurement may arrive after a 200 Hz control query while carrying an
    earlier image timestamp; keeping the committed state at the last image
    time allows that measurement to be fused in chronological sensor time.
    """

    def __init__(self, config: Optional[BallTrackerConfig] = None) -> None:
        self.config = config or BallTrackerConfig()
        self.tracker = BallEKFTracker(self.config)
        self._filter_timestamp_s: Optional[float] = None
        self._last_measurement_timestamp_s: Optional[float] = None
        self._lock = threading.RLock()
        self._control_predictions = 0
        self._out_of_sequence_measurements = 0

    def update(
            self,
            position_measurement: Any,
            timestamp_s: float,
            *,
            racket_pose: Any = None,
            measurement_covariance: Optional[np.ndarray] = None,
    ) -> BallEstimate:
        timestamp = self._timestamp(timestamp_s)
        with self._lock:
            if (self._filter_timestamp_s is not None
                    and timestamp < self._filter_timestamp_s - 1e-9):
                self._out_of_sequence_measurements += 1
                return self._make_estimate(
                    self.tracker.get_state(),
                    self.tracker.get_covariance(),
                    timestamp,
                    valid=self.tracker.initialized,
                    mode=self.tracker.mode,
                    measurement_accepted=False,
                    racket_pose=racket_pose,
                    metadata={'reject_reason': 'out_of_sequence_measurement'},
                )

            if not self.tracker.initialized:
                accepted = self.tracker.update(
                    position_measurement,
                    measurement_covariance=measurement_covariance,
                    racket_pose=racket_pose,
                )
            else:
                step = timestamp - float(self._filter_timestamp_s)
                still_valid = self.tracker.predict(
                    step, racket_pose=racket_pose)
                if not still_valid:
                    # A real observation after the configured FOV-loss timeout
                    # starts a fresh track immediately.
                    accepted = self.tracker.update(
                        position_measurement,
                        measurement_covariance=measurement_covariance,
                        racket_pose=racket_pose,
                    )
                else:
                    accepted = self.tracker.update(
                        position_measurement,
                        measurement_covariance=measurement_covariance,
                        racket_pose=racket_pose,
                    )

            self._filter_timestamp_s = timestamp
            if accepted:
                self._last_measurement_timestamp_s = timestamp
            return self._make_estimate(
                self.tracker.get_state(),
                self.tracker.get_covariance(),
                timestamp,
                valid=self.tracker.initialized,
                mode=self.tracker.mode,
                measurement_accepted=accepted,
                racket_pose=racket_pose,
            )

    def predict_to(
            self, timestamp_s: float, *, racket_pose: Any = None,
    ) -> BallEstimate:
        timestamp = self._timestamp(timestamp_s)
        with self._lock:
            if (not self.tracker.initialized
                    or self._filter_timestamp_s is None):
                return self._invalid_estimate(
                    timestamp, 'uninitialized', racket_pose=racket_pose)
            step = timestamp - self._filter_timestamp_s
            if step < -1e-9:
                return self._invalid_estimate(
                    timestamp, 'prediction_precedes_filter_state',
                    racket_pose=racket_pose)
            projection = self.tracker.project(max(step, 0.0), racket_pose)
            self._control_predictions += 1
            if not projection['valid']:
                # The objective requires a reset after the bounded prediction
                # window, rather than silently publishing an old state.
                self.tracker.reset()
                self._filter_timestamp_s = None
                self._last_measurement_timestamp_s = None
                return self._invalid_estimate(
                    timestamp, 'prediction_timeout_reset',
                    racket_pose=racket_pose)
            return self._make_estimate(
                projection['state'],
                projection['covariance'],
                timestamp,
                valid=True,
                mode=projection['mode'],
                measurement_accepted=None,
                racket_pose=racket_pose,
                metadata={
                    'predicted': step > 0.0,
                    'contact_candidate': projection['contact_candidate'],
                },
            )

    def current_estimate(self, *, racket_pose: Any = None) -> BallEstimate:
        """Return the committed measurement-time estimate."""
        with self._lock:
            if (not self.tracker.initialized
                    or self._filter_timestamp_s is None):
                return self._invalid_estimate(
                    0.0, 'uninitialized', racket_pose=racket_pose)
            return self._make_estimate(
                self.tracker.get_state(), self.tracker.get_covariance(),
                self._filter_timestamp_s, valid=True, mode=self.tracker.mode,
                racket_pose=racket_pose)

    def reset(self) -> None:
        with self._lock:
            self.tracker.reset()
            self._filter_timestamp_s = None
            self._last_measurement_timestamp_s = None

    def set_ballistic_dynamics(
            self,
            enabled: bool,
            *,
            reinitialize_from_history: bool = False,
    ) -> None:
        """Forward the probe's held/released phase to the EKF tracker."""
        with self._lock:
            self.tracker.set_ballistic_dynamics(
                enabled,
                reinitialize_from_history=reinitialize_from_history,
            )

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            diagnostics = self.tracker.diagnostics()
            diagnostics.update({
                'filter_timestamp_s': self._filter_timestamp_s,
                'last_measurement_timestamp_s': (
                    self._last_measurement_timestamp_s),
                'control_predictions': self._control_predictions,
                'out_of_sequence_measurements': (
                    self._out_of_sequence_measurements),
            })
            return diagnostics

    def _make_estimate(
            self,
            state: np.ndarray,
            covariance: np.ndarray,
            timestamp_s: float,
            *,
            valid: bool,
            mode: str,
            measurement_accepted: Optional[bool] = None,
            racket_pose: Any = None,
            metadata: Optional[dict[str, Any]] = None,
    ) -> BallEstimate:
        if (self._last_measurement_timestamp_s is None
                or not np.isfinite(timestamp_s)):
            age = 0.0
        else:
            age = max(
                0.0, timestamp_s - self._last_measurement_timestamp_s)
        impact = None
        if valid and racket_pose is not None:
            impact = predict_contact(
                state,
                racket_pose,
                covariance=covariance,
                max_time_s=self.config.impact_prediction_max_time_s,
                min_approach_speed_m_s=(
                    self.config.impact_min_approach_speed_m_s),
                plane_tolerance_m=self.config.impact_plane_tolerance_m,
            )
        return BallEstimate(
            timestamp_s=float(timestamp_s),
            state_mean=np.asarray(
                state, dtype=float).reshape(STATE_DIM).copy(),
            state_covariance=np.asarray(
                covariance, dtype=float).reshape(STATE_DIM, STATE_DIM).copy(),
            valid=bool(valid),
            mode=str(mode),
            time_since_measurement_s=float(age),
            measurement_accepted=measurement_accepted,
            impact_prediction=impact,
            metadata={} if metadata is None else dict(metadata),
        )

    def _invalid_estimate(
            self, timestamp_s: float, reason: str, *, racket_pose: Any = None,
    ) -> BallEstimate:
        del racket_pose
        return BallEstimate(
            timestamp_s=float(timestamp_s),
            state_mean=np.zeros(STATE_DIM, dtype=float),
            state_covariance=np.zeros((STATE_DIM, STATE_DIM), dtype=float),
            valid=False,
            mode=MODE_UNINITIALIZED,
            time_since_measurement_s=0.0,
            measurement_accepted=None,
            impact_prediction=None,
            metadata={'reject_reason': reason},
        )

    @staticmethod
    def _timestamp(value: float) -> float:
        timestamp = float(value)
        if not np.isfinite(timestamp):
            raise ValueError('timestamp_s must be finite')
        return timestamp


class BallStateEstimator:
    """Stable facade consumed by simulation, robot bridges, and RL code."""

    def __init__(
            self,
            config: Optional[
                BallTrackerConfig | Mapping[str, Any] | str | Path] = None,
            *,
            backend: Optional[BallEstimatorBackend] = None,
    ) -> None:
        if backend is not None and config is not None:
            raise ValueError('Pass either config or backend, not both')
        if backend is not None:
            self.backend = backend
        else:
            resolved = self._resolve_config(config)
            self.backend = EKFEstimatorBackend(resolved)

    def update(
            self, position_measurement: Any, timestamp_s: float,
            *, racket_pose: Any = None,
            measurement_covariance: Optional[np.ndarray] = None,
    ) -> BallEstimate:
        return self.backend.update(
            position_measurement,
            timestamp_s,
            racket_pose=racket_pose,
            measurement_covariance=measurement_covariance,
        )

    def predict_to(
            self, timestamp_s: float, *, racket_pose: Any = None,
    ) -> BallEstimate:
        return self.backend.predict_to(timestamp_s, racket_pose=racket_pose)

    def step(
            self,
            timestamp_s: float,
            position_measurement: Optional[Any] = None,
            *,
            racket_pose: Any = None,
            measurement_covariance: Optional[np.ndarray] = None,
    ) -> BallEstimate:
        """Fuse a measurement when present, otherwise prediction-only."""
        if position_measurement is None:
            return self.predict_to(timestamp_s, racket_pose=racket_pose)
        return self.update(
            position_measurement,
            timestamp_s,
            racket_pose=racket_pose,
            measurement_covariance=measurement_covariance,
        )

    def reset(self) -> None:
        self.backend.reset()

    def set_ballistic_dynamics(
            self,
            enabled: bool,
            *,
            reinitialize_from_history: bool = False,
    ) -> None:
        """Set held-ball/flight dynamics when supported by the backend."""
        setter = getattr(self.backend, 'set_ballistic_dynamics', None)
        if setter is None:
            raise NotImplementedError(
                'Selected estimator backend cannot switch dynamics')
        setter(
            enabled,
            reinitialize_from_history=reinitialize_from_history,
        )

    def diagnostics(self) -> dict[str, Any]:
        return self.backend.diagnostics()

    @staticmethod
    def _resolve_config(
            config: Optional[
                BallTrackerConfig | Mapping[str, Any] | str | Path],
    ) -> BallTrackerConfig:
        if config is None:
            return BallTrackerConfig()
        if isinstance(config, BallTrackerConfig):
            return config
        if isinstance(config, Mapping):
            return BallTrackerConfig.from_mapping(config)
        return BallTrackerConfig.from_yaml(config)
