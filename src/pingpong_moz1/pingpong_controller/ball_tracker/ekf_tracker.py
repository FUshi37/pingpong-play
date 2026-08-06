"""Hybrid 9-state EKF for ping-pong ball tracking.

The measurement model is position-only.  Velocity and acceleration are
inferred by the filter, while the mode logic handles the discontinuity caused
by a racket impact without pretending that it is ordinary process noise.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .ball_model import (
    ACCELERATION,
    POSITION,
    STATE_DIM,
    VELOCITY,
    ball_dynamics,
    jacobian,
    process_noise,
)
from .config import BallTrackerConfig
from .impact_predictor import signed_plane_distance


MODE_UNINITIALIZED = 'uninitialized'
MODE_FLIGHT = 'flight'
MODE_CONTACT_CANDIDATE = 'contact_candidate'
MODE_POST_IMPACT = 'post_impact'


def _position_vector(value: Any, name: str = 'position') -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name} must be a finite vector with shape (3,)')
    return vector.copy()


class BallEKFTracker:
    """Estimate position, velocity, acceleration, and their covariance.

    ``predict(dt)`` is intended for every control tick (for example 200 Hz),
    while ``update(position)`` is called only when the IR pipeline produces a
    genuine 3-D measurement (for example 60 Hz).  Calling ``predict`` without
    an update remains valid for ``max_prediction_duration_s``; the tracker
    resets itself only after that bounded FOV-loss interval.
    """

    def __init__(self, config: Optional[BallTrackerConfig] = None) -> None:
        self.config = config or BallTrackerConfig()
        self._measurement_matrix = np.zeros((3, STATE_DIM), dtype=float)
        self._measurement_matrix[:, POSITION] = np.eye(3)
        self._default_measurement_covariance = np.diag(
            np.square(self.config.measurement_std_m))
        self.reset()

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def contact_candidate(self) -> bool:
        return self._mode == MODE_CONTACT_CANDIDATE

    @property
    def time_since_measurement_s(self) -> float:
        return float(self._time_since_measurement_s)

    def reset(self) -> None:
        """Invalidate all state and covariance."""
        ballistic_enabled = getattr(
            self, '_ballistic_dynamics_enabled', True)
        self._state = np.zeros(STATE_DIM, dtype=float)
        self._covariance = np.eye(STATE_DIM, dtype=float)
        self._initialized = False
        self._mode = MODE_UNINITIALIZED
        self._time_since_measurement_s = 0.0
        self._contact_candidate_age_s = 0.0
        self._post_impact_age_s = 0.0
        self._post_impact_velocity_refined = False
        self._pre_impact_velocity: Optional[np.ndarray] = None
        self._last_measurement_position: Optional[np.ndarray] = None
        self._consecutive_rejects = 0
        self.last_event = 'reset'
        self.predictions = 0
        self.updates_accepted = 0
        self.updates_rejected = 0
        self.impacts_detected = 0
        self._ballistic_dynamics_enabled = bool(ballistic_enabled)

    def initialize(
            self,
            position: Any,
            velocity: Optional[Any] = None,
            acceleration: Optional[Any] = None,
            covariance: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize state from a position and optional motion priors."""
        state = np.zeros(STATE_DIM, dtype=float)
        state[POSITION] = _position_vector(position)
        state[VELOCITY] = (
            np.zeros(3, dtype=float)
            if velocity is None else _position_vector(velocity, 'velocity'))
        state[ACCELERATION] = (
            (np.asarray(self.config.gravity_m_s2, dtype=float)
             if self._ballistic_dynamics_enabled else np.zeros(3))
            if acceleration is None
            else _position_vector(acceleration, 'acceleration'))

        if covariance is None:
            variances = np.concatenate([
                np.full(3, self.config.initial_position_std_m ** 2),
                np.full(3, self.config.initial_velocity_std_m_s ** 2),
                np.full(3, self.config.initial_acceleration_std_m_s2 ** 2),
            ])
            matrix = np.diag(variances)
        else:
            matrix = np.asarray(covariance, dtype=float)
            if (matrix.shape != (STATE_DIM, STATE_DIM)
                    or not np.all(np.isfinite(matrix))):
                raise ValueError('covariance must be a finite 9x9 matrix')
            matrix = self._stabilize_covariance(matrix)

        self._state = state
        self._covariance = matrix
        self._initialized = True
        self._mode = MODE_FLIGHT
        self._time_since_measurement_s = 0.0
        self._contact_candidate_age_s = 0.0
        self._post_impact_age_s = 0.0
        self._post_impact_velocity_refined = False
        self._pre_impact_velocity = None
        self._last_measurement_position = state[POSITION].copy()
        self._consecutive_rejects = 0
        self.last_event = 'initialized'

    def set_ballistic_dynamics(
            self,
            enabled: bool,
            *,
            reinitialize_from_history: bool = False,
    ) -> None:
        """Switch between held-ball constant velocity and flight dynamics.

        The probe calls this while its measured release gate is closed.  The
        nine-state layout is retained in both modes; acceleration is decoupled
        from position/velocity and held at zero until flight is enabled.
        ``reinitialize_from_history`` is accepted for API compatibility.  The
        EKF's current velocity already reflects recent position measurements.
        """
        del reinitialize_from_history
        requested = bool(enabled)
        if requested == self._ballistic_dynamics_enabled:
            return
        self._ballistic_dynamics_enabled = requested
        if not self._initialized:
            return
        self._state[ACCELERATION] = (
            np.asarray(self.config.gravity_m_s2, dtype=float)
            if requested else np.zeros(3, dtype=float))
        self._covariance[:6, ACCELERATION] = 0.0
        self._covariance[ACCELERATION, :6] = 0.0
        self._covariance[ACCELERATION, ACCELERATION] = np.eye(3) * (
            self.config.initial_acceleration_std_m_s2 ** 2)
        self._covariance = self._stabilize_covariance(self._covariance)
        self.last_event = (
            'ballistic_dynamics_enabled'
            if requested else 'ballistic_dynamics_disabled')

    def predict(self, dt: float, racket_pose: Any = None) -> bool:
        """Advance the committed filter state by ``dt`` seconds.

        Returns ``False`` when the tracker is uninitialized or when the missing
        observation interval exceeded its configured bound and triggered a
        reset.  Otherwise returns ``True``.
        """
        step = self._validate_dt(dt)
        if not self._initialized:
            return False
        if (self._time_since_measurement_s + step
                > self.config.max_prediction_duration_s + 1e-12):
            self.reset()
            self.last_event = 'prediction_timeout_reset'
            return False

        previous_state = self._state.copy()
        self._state, self._covariance = self._project_mean_covariance(
            self._state, self._covariance, step)
        self._time_since_measurement_s += step
        self.predictions += 1
        self._advance_mode(step, racket_pose, previous_state)
        self.last_event = (
            'contact_candidate' if self.contact_candidate else 'predicted')
        return True

    def project(self, dt: float, racket_pose: Any = None) -> dict[str, Any]:
        """Return a non-mutating prediction used by latency-safe facades."""
        step = self._validate_dt(dt)
        if not self._initialized:
            return {
                'valid': False,
                'state': np.zeros(STATE_DIM, dtype=float),
                'covariance': np.eye(STATE_DIM, dtype=float),
                'mode': MODE_UNINITIALIZED,
                'contact_candidate': False,
            }
        if (self._time_since_measurement_s + step
                > self.config.max_prediction_duration_s + 1e-12):
            return {
                'valid': False,
                'state': np.zeros(STATE_DIM, dtype=float),
                'covariance': np.eye(STATE_DIM, dtype=float),
                'mode': MODE_UNINITIALIZED,
                'contact_candidate': False,
            }
        state, covariance = self._project_mean_covariance(
            self._state, self._covariance, step)
        mode = self._mode
        if (mode == MODE_POST_IMPACT
                and self._post_impact_age_s + step
                >= self.config.post_impact_mode_duration_s):
            mode = MODE_FLIGHT
        is_candidate = self._is_contact_candidate_interval(
            self._state, state, racket_pose)
        if mode != MODE_POST_IMPACT and is_candidate:
            mode = MODE_CONTACT_CANDIDATE
        elif (mode == MODE_CONTACT_CANDIDATE
              and self._contact_candidate_age_s + step
              > self.config.contact_candidate_timeout_s):
            mode = MODE_FLIGHT
        return {
            'valid': True,
            'state': state,
            'covariance': covariance,
            'mode': mode,
            'contact_candidate': mode == MODE_CONTACT_CANDIDATE,
        }

    def update(
            self,
            position_measurement: Any,
            measurement_covariance: Optional[np.ndarray] = None,
            racket_pose: Any = None,
    ) -> bool:
        """Fuse a position measurement and return whether it was accepted."""
        measurement = _position_vector(position_measurement, 'measurement')
        if not self._initialized:
            self.initialize(measurement)
            self.updates_accepted += 1
            return True

        covariance = self._resolve_measurement_covariance(
            measurement_covariance)
        interval = self._time_since_measurement_s
        measured_velocity = None
        if self._last_measurement_position is not None and interval > 1e-5:
            measured_velocity = (
                measurement - self._last_measurement_position) / interval

        if (measured_velocity is not None
                and self._is_post_impact_velocity(measured_velocity)):
            self._reset_post_impact(measurement, measured_velocity, covariance)
            return True

        # The first position after an impact may straddle the discontinuity,
        # so its finite-difference velocity is only a coarse seed.  Refine it
        # once from the next purely post-impact measurement interval.
        axis = self.config.vertical_axis
        if (measured_velocity is not None
                and self._mode == MODE_POST_IMPACT
                and not self._post_impact_velocity_refined
                and measured_velocity[axis]
                > self.config.min_post_impact_upward_speed_m_s):
            instantaneous_velocity = measured_velocity + 0.5 * np.asarray(
                self.config.gravity_m_s2, dtype=float) * interval
            self._reset_post_impact(
                measurement,
                instantaneous_velocity,
                covariance,
                count_event=False,
            )
            self._post_impact_velocity_refined = True
            self.last_event = 'post_impact_velocity_refined'
            return True

        innovation = measurement - self._state[POSITION]
        innovation_covariance = (
            self._measurement_matrix
            @ self._covariance
            @ self._measurement_matrix.T
            + covariance)
        try:
            solved = np.linalg.solve(innovation_covariance, innovation)
            normalized_innovation = float(innovation @ solved)
        except np.linalg.LinAlgError:
            self.updates_rejected += 1
            self._consecutive_rejects += 1
            self.last_event = 'singular_innovation_reject'
            return False

        gate = float(self.config.innovation_gate_chi2)
        if gate > 0.0 and normalized_innovation > gate:
            self.updates_rejected += 1
            self._consecutive_rejects += 1
            self.last_event = 'innovation_gate_reject'
            return False

        # Solve S K^T = H P instead of explicitly inverting S.
        gain = np.linalg.solve(
            innovation_covariance,
            self._measurement_matrix @ self._covariance,
        ).T
        self._state = self._state + gain @ innovation
        if not self._ballistic_dynamics_enabled:
            self._state[ACCELERATION] = 0.0

        # Joseph form preserves positive semi-definiteness under roundoff.
        identity = np.eye(STATE_DIM, dtype=float)
        correction = identity - gain @ self._measurement_matrix
        self._covariance = (
            correction @ self._covariance @ correction.T
            + gain @ covariance @ gain.T)
        self._covariance = self._stabilize_covariance(self._covariance)

        self._last_measurement_position = measurement.copy()
        self._time_since_measurement_s = 0.0
        self._consecutive_rejects = 0
        self.updates_accepted += 1
        if self._mode != MODE_POST_IMPACT:
            if self._is_contact_candidate(self._state, racket_pose):
                self._arm_contact_candidate()
            elif (self._mode == MODE_CONTACT_CANDIDATE
                  and self._contact_candidate_age_s
                  > self.config.contact_candidate_timeout_s):
                self._clear_contact_candidate()
        self.last_event = 'updated'
        return True

    def get_state(self) -> np.ndarray:
        """Return a copy of the 9-vector state mean."""
        return self._state.copy()

    def get_covariance(self) -> np.ndarray:
        """Return a copy of the 9x9 state covariance."""
        return self._covariance.copy()

    def get_state_dict(self) -> dict[str, Any]:
        """Return state, uncertainty, mode, and diagnostic counters."""
        diagonal = np.maximum(np.diag(self._covariance), 0.0)
        std = np.sqrt(diagonal)
        return {
            'initialized': bool(self._initialized),
            'mode': self._mode,
            'state_mean': self._state.tolist(),
            'state_covariance': self._covariance.tolist(),
            'position': self._state[POSITION].tolist(),
            'velocity': self._state[VELOCITY].tolist(),
            'acceleration': self._state[ACCELERATION].tolist(),
            'position_std': std[POSITION].tolist(),
            'velocity_std': std[VELOCITY].tolist(),
            'acceleration_std': std[ACCELERATION].tolist(),
            'time_since_measurement_s': float(self._time_since_measurement_s),
            'contact_candidate': bool(self.contact_candidate),
            'last_event': self.last_event,
            'updates_accepted': int(self.updates_accepted),
            'updates_rejected': int(self.updates_rejected),
            'impacts_detected': int(self.impacts_detected),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return lightweight counters suitable for real-time logging."""
        return {
            'initialized': self._initialized,
            'mode': self._mode,
            'last_event': self.last_event,
            'predictions': self.predictions,
            'updates_accepted': self.updates_accepted,
            'updates_rejected': self.updates_rejected,
            'consecutive_rejects': self._consecutive_rejects,
            'impacts_detected': self.impacts_detected,
            'time_since_measurement_s': self._time_since_measurement_s,
            'ballistic_dynamics_enabled': (
                self._ballistic_dynamics_enabled),
        }

    def _project_mean_covariance(
            self,
            state: np.ndarray,
            covariance: np.ndarray,
            dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if dt <= 0.0:
            return state.copy(), covariance.copy()
        # Sub-stepping bounds linearization/numerical error and makes the same
        # code safe if the motion model later gains nonlinear residual terms.
        count = max(1, int(np.ceil(dt / self.config.max_predict_step_s)))
        substep = dt / count
        mean = state.copy()
        matrix = covariance.copy()
        process_std = self.config.acceleration_process_std_m_s3
        for _ in range(count):
            if self._ballistic_dynamics_enabled:
                transition = jacobian(mean, substep)
                noise = process_noise(substep, process_std)
                mean = ball_dynamics(mean, substep)
            else:
                transition = np.eye(STATE_DIM, dtype=float)
                transition[POSITION, VELOCITY] = np.eye(3) * substep
                noise = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
                for axis in range(3):
                    intensity = process_std[axis] ** 2
                    indices = [axis, axis + 3]
                    noise[np.ix_(indices, indices)] = intensity * np.array([
                        [substep ** 3 / 3.0, substep ** 2 / 2.0],
                        [substep ** 2 / 2.0, substep],
                    ])
                mean = mean.copy()
                mean[POSITION] += mean[VELOCITY] * substep
                mean[ACCELERATION] = 0.0
            matrix = transition @ matrix @ transition.T + noise
        return mean, self._stabilize_covariance(matrix)

    def _resolve_measurement_covariance(
            self, value: Optional[np.ndarray]) -> np.ndarray:
        if value is None:
            return self._default_measurement_covariance.copy()
        matrix = np.asarray(value, dtype=float)
        if matrix.shape == (3,):
            if np.any(matrix <= 0.0):
                raise ValueError(
                    'measurement standard deviations must be positive')
            matrix = np.diag(np.square(matrix))
        if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))):
            raise ValueError(
                'measurement_covariance must be a finite 3x3 matrix or 3 stds')
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        if float(np.min(eigenvalues)) <= 0.0:
            raise ValueError(
                'measurement covariance must be positive definite')
        return 0.5 * (matrix + matrix.T)

    def _advance_mode(
            self, dt: float, racket_pose: Any,
            previous_state: np.ndarray) -> None:
        if self._mode == MODE_POST_IMPACT:
            self._post_impact_age_s += dt
            if (self._post_impact_age_s
                    >= self.config.post_impact_mode_duration_s):
                self._mode = MODE_FLIGHT
                self._post_impact_age_s = 0.0
            return

        if self._mode == MODE_CONTACT_CANDIDATE:
            self._contact_candidate_age_s += dt
        if self._is_contact_candidate_interval(
                previous_state, self._state, racket_pose):
            self._arm_contact_candidate()
        elif (self._mode == MODE_CONTACT_CANDIDATE
              and self._contact_candidate_age_s
              > self.config.contact_candidate_timeout_s):
            self._clear_contact_candidate()

    def _is_contact_candidate(
            self, state: np.ndarray, racket_pose: Any) -> bool:
        if racket_pose is None:
            return False
        vertical_velocity = float(state[3 + self.config.vertical_axis])
        if vertical_velocity >= -self.config.min_descending_speed_m_s:
            return False
        distance = abs(signed_plane_distance(state[POSITION], racket_pose))
        return distance <= self.config.contact_distance_threshold_m

    def _is_contact_candidate_interval(
            self,
            start_state: np.ndarray,
            end_state: np.ndarray,
            racket_pose: Any,
    ) -> bool:
        """Detect near-plane motion even when one 60 Hz step crosses it.

        A fast ball moves roughly 5--8 cm per camera frame. Checking only the
        end of a prediction can therefore jump completely over the 5 cm
        contact band, especially after a dropped frame. A sign change in
        signed plane distance is an unambiguous interval crossing.
        """
        if racket_pose is None:
            return False
        axis = self.config.vertical_axis
        if (start_state[3 + axis]
                >= -self.config.min_descending_speed_m_s
                and end_state[3 + axis]
                >= -self.config.min_descending_speed_m_s):
            return False
        start_distance = signed_plane_distance(
            start_state[POSITION], racket_pose)
        end_distance = signed_plane_distance(end_state[POSITION], racket_pose)
        threshold = self.config.contact_distance_threshold_m
        return bool(
            min(abs(start_distance), abs(end_distance)) <= threshold
            or start_distance * end_distance <= 0.0)

    def _arm_contact_candidate(self) -> None:
        if self._mode != MODE_CONTACT_CANDIDATE:
            self._pre_impact_velocity = self._state[VELOCITY].copy()
            self._contact_candidate_age_s = 0.0
        self._mode = MODE_CONTACT_CANDIDATE

    def _clear_contact_candidate(self) -> None:
        self._mode = MODE_FLIGHT
        self._contact_candidate_age_s = 0.0
        self._pre_impact_velocity = None

    def _is_post_impact_velocity(self, measured_velocity: np.ndarray) -> bool:
        if (self._mode != MODE_CONTACT_CANDIDATE
                or self._pre_impact_velocity is None):
            return False
        axis = self.config.vertical_axis
        before = float(self._pre_impact_velocity[axis])
        after = float(measured_velocity[axis])
        return bool(
            before < -self.config.min_descending_speed_m_s
            and after > self.config.min_post_impact_upward_speed_m_s
            and after - before
            > self.config.impact_delta_velocity_threshold_m_s)

    def _reset_post_impact(
            self,
            measurement: np.ndarray,
            measured_velocity: np.ndarray,
            measurement_covariance: np.ndarray,
            *,
            count_event: bool = True,
    ) -> None:
        self._state[POSITION] = measurement
        self._state[VELOCITY] = measured_velocity
        self._state[ACCELERATION] = np.asarray(
            self.config.gravity_m_s2
            if self._ballistic_dynamics_enabled else (0.0, 0.0, 0.0),
            dtype=float,
        )
        self._covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
        self._covariance[POSITION, POSITION] = measurement_covariance
        self._covariance[VELOCITY, VELOCITY] = np.eye(3) * (
            self.config.post_impact_velocity_std_m_s ** 2)
        self._covariance[ACCELERATION, ACCELERATION] = np.eye(3) * (
            self.config.initial_acceleration_std_m_s2 ** 2)
        self._covariance = self._stabilize_covariance(self._covariance)
        self._last_measurement_position = measurement.copy()
        self._time_since_measurement_s = 0.0
        self._mode = MODE_POST_IMPACT
        self._post_impact_age_s = 0.0
        self._post_impact_velocity_refined = not count_event
        self._contact_candidate_age_s = 0.0
        self._pre_impact_velocity = None
        self._consecutive_rejects = 0
        self.updates_accepted += 1
        if count_event:
            self.impacts_detected += 1
            self.last_event = 'post_impact_velocity_reset'

    def _stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        matrix = 0.5 * (covariance + covariance.T)
        minimum = float(np.min(np.linalg.eigvalsh(matrix)))
        floor = self.config.covariance_eigenvalue_floor
        if minimum < floor:
            matrix = matrix + np.eye(STATE_DIM) * (floor - minimum)
        return matrix

    @staticmethod
    def _validate_dt(dt: float) -> float:
        value = float(dt)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError('dt must be finite and non-negative')
        return value
