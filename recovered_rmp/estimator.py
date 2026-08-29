"""State estimators matching the recovered constant-acceleration model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class ConstantAccelerationKalman:
    """Independent [position, velocity, acceleration] filters per axis.

    DWARF exposes the original scalar state as ``x: Vector3d``, ``P/F/Q:
    Matrix3d``, ``H: RowVector3d``, ``R`` and ``K: Vector3d``.  The original
    initializer contains dt through dt^5, matching the white-jerk Q below.
    """

    def __init__(self, dimensions: int, dt: float, process_variance: float, measurement_variance: float):
        if dimensions <= 0 or dt <= 0.0:
            raise ValueError("dimensions and dt must be positive")
        self.dimensions = int(dimensions)
        self.dt = float(dt)
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        dt2, dt3 = dt * dt, dt * dt * dt
        dt4, dt5 = dt3 * dt, dt3 * dt * dt
        self.F = np.array([[1.0, dt, 0.5 * dt2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        self.H = np.array([1.0, 0.0, 0.0])
        self.Q = self.process_variance * np.array(
            [[dt5 / 20.0, dt4 / 8.0, dt3 / 6.0], [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0], [dt3 / 6.0, dt2 / 2.0, dt]]
        )
        self.x = np.zeros((self.dimensions, 3), dtype=float)
        self.P = np.zeros((self.dimensions, 3, 3), dtype=float)
        self.P[:, 0, 0] = 0.01
        self.initialized = False

    @property
    def position(self) -> FloatArray:
        return self.x[:, 0].copy()

    @property
    def velocity(self) -> FloatArray:
        return self.x[:, 1].copy()

    @property
    def acceleration(self) -> FloatArray:
        return self.x[:, 2].copy()

    def reset(self, measurement: ArrayLike, *, velocity: ArrayLike | None = None, acceleration: ArrayLike | None = None) -> None:
        value = np.asarray(measurement, dtype=float).reshape(-1)
        if value.size != self.dimensions:
            raise ValueError("measurement has the wrong dimension")
        self.x.fill(0.0)
        self.x[:, 0] = value
        if velocity is not None:
            self.x[:, 1] = np.asarray(velocity, dtype=float).reshape(self.dimensions)
        if acceleration is not None:
            self.x[:, 2] = np.asarray(acceleration, dtype=float).reshape(self.dimensions)
        # Exact initSingleJointFilter initialization: only the measured
        # position state starts with covariance 0.01.  Velocity and
        # acceleration covariance start at zero and are introduced through Q.
        self.P.fill(0.0)
        self.P[:, 0, 0] = 0.01
        self.initialized = True

    def predict(self) -> None:
        if not self.initialized:
            raise RuntimeError("reset or update the filter before predict")
        self.x = self.x @ self.F.T
        self.P = self.F[None, :, :] @ self.P @ self.F.T[None, :, :] + self.Q[None, :, :]

    def decelerate(self, acceleration_limit: ArrayLike) -> None:
        """Predict one tick while braking the estimated velocity to zero.

        The binary switches from ``Predict`` to ``Deceleration`` after 20
        servo ticks without a new target.  Deceleration replaces each filter's
        acceleration state by ``clip(-velocity / dt, +/- limit)`` (and exact
        zero below 1e-6 rad/s), then performs the normal state/covariance
        prediction.
        """

        if not self.initialized:
            raise RuntimeError("reset or update the filter before decelerate")
        limit = np.asarray(acceleration_limit, dtype=float).reshape(-1)
        if limit.size != self.dimensions:
            raise ValueError("acceleration_limit has the wrong dimension")
        braking = np.clip(-self.x[:, 1] / self.dt, -limit, limit)
        self.x[:, 2] = np.where(np.abs(self.x[:, 1]) <= 1.0e-6, 0.0, braking)
        self.predict()

    def update(self, measurement: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        value = np.asarray(measurement, dtype=float).reshape(-1)
        if value.size != self.dimensions:
            raise ValueError("measurement has the wrong dimension")
        if not self.initialized:
            self.reset(value)
            return self.position, self.velocity, self.acceleration
        self.predict()
        innovation = value - self.x[:, 0]
        innovation_cov = self.P[:, 0, 0] + self.measurement_variance
        gain = self.P[:, :, 0] / innovation_cov[:, None]
        self.x += gain * innovation[:, None]
        identity = np.eye(3)[None, :, :]
        kh = gain[:, :, None] * self.H[None, None, :]
        # Joseph form keeps P symmetric/positive in long simulation runs.
        left = identity - kh
        self.P = left @ self.P @ np.transpose(left, (0, 2, 1))
        self.P += self.measurement_variance * gain[:, :, None] * gain[:, None, :]
        return self.position, self.velocity, self.acceleration


@dataclass
class TargetDifferentiator:
    """Moving-average target filter and numerical velocity/acceleration.

    The two window lengths and the three position/velocity/acceleration filter
    buffers are confirmed members of the binary.  Exact edge bookkeeping was
    optimized away, so this is the simulation-equivalent reconstruction.
    """

    dimensions: int
    dt: float
    target_filter_length: int = 10
    derivative_filter_length: int = 35

    def __post_init__(self) -> None:
        if self.dimensions <= 0 or self.dt <= 0.0:
            raise ValueError("dimensions and dt must be positive")
        self._targets: deque[FloatArray] = deque(maxlen=max(1, self.target_filter_length))
        self._velocities: deque[FloatArray] = deque(maxlen=max(1, self.derivative_filter_length))
        self._accelerations: deque[FloatArray] = deque(maxlen=max(1, self.derivative_filter_length))
        self._last_position: FloatArray | None = None
        self._last_velocity = np.zeros(self.dimensions, dtype=float)

    def reset(self, target: ArrayLike) -> None:
        value = np.asarray(target, dtype=float).reshape(self.dimensions)
        self._targets.clear()
        self._velocities.clear()
        self._accelerations.clear()
        self._targets.append(value.copy())
        self._last_position = value.copy()
        self._last_velocity.fill(0.0)

    def update(self, target: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        value = np.asarray(target, dtype=float).reshape(self.dimensions)
        if self._last_position is None:
            self.reset(value)
            return value.copy(), np.zeros_like(value), np.zeros_like(value)
        self._targets.append(value.copy())
        position = np.mean(np.stack(tuple(self._targets)), axis=0)
        velocity_raw = (position - self._last_position) / self.dt
        self._velocities.append(velocity_raw)
        velocity = np.mean(np.stack(tuple(self._velocities)), axis=0)
        acceleration_raw = (velocity - self._last_velocity) / self.dt
        self._accelerations.append(acceleration_raw)
        acceleration = np.mean(np.stack(tuple(self._accelerations)), axis=0)
        self._last_position = position
        self._last_velocity = velocity
        return position.copy(), velocity.copy(), acceleration.copy()
