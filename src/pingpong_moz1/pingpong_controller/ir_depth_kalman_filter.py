"""Causal 3D Kalman filter for synchronized IR-pixel/depth measurements.

The detector must first back-project the IR center and depth from the same
stereo frameset into its output frame.  This filter then estimates position
and velocity without differentiating a sliding position window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IRDepthKalmanConfig:
    """Filter tuning in millimetres and seconds."""

    horizontal_accel_std_mm_s2: float = 2000.0
    vertical_accel_std_mm_s2: float = 8000.0
    lateral_measurement_std_mm: float = 5.0
    depth_measurement_std_mm: float = 10.0
    innovation_gate_sigma: float = 8.0
    reset_after_outliers: int = 3
    max_dt_s: float = 0.05


class IRDepthKalmanFilter:
    """Variable-dt constant-velocity filter with axis-specific uncertainty."""

    def __init__(
        self,
        config: IRDepthKalmanConfig | None = None,
        frame_rotation: np.ndarray | None = None,
    ) -> None:
        self.config = config or IRDepthKalmanConfig()
        self._frame_rotation = self._valid_rotation_or_identity(frame_rotation)
        self._measurement_covariance = self._build_measurement_covariance()
        self.reset()

    @staticmethod
    def _valid_rotation_or_identity(rotation: np.ndarray | None) -> np.ndarray:
        if rotation is None:
            return np.eye(3, dtype=np.float64)
        candidate = np.asarray(rotation, dtype=np.float64)
        if candidate.shape != (3, 3) or not np.all(np.isfinite(candidate)):
            return np.eye(3, dtype=np.float64)
        return candidate.copy()

    def _build_measurement_covariance(self) -> np.ndarray:
        lateral_variance = (
            max(float(self.config.lateral_measurement_std_mm), 1e-6) ** 2)
        depth_variance = (
            max(float(self.config.depth_measurement_std_mm), 1e-6) ** 2)
        camera_covariance = np.diag([
            lateral_variance,
            lateral_variance,
            depth_variance,
        ])
        rotation = self._frame_rotation
        return rotation @ camera_covariance @ rotation.T

    def reset(self) -> None:
        self.initialized = False
        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)
        self.last_timestamp_s: float | None = None
        self.consecutive_outliers = 0
        self.total_updates = 0
        self.outliers_rejected = 0
        self.reinitializations = 0
        self.last_innovation_mahalanobis = 0.0
        self._reject_measurements: list[tuple[np.ndarray, float]] = []

    @staticmethod
    def _transition(dt_s: float) -> np.ndarray:
        transition = np.eye(6, dtype=np.float64)
        transition[:3, 3:] = np.eye(3, dtype=np.float64) * dt_s
        return transition

    def _process_covariance(self, dt_s: float) -> np.ndarray:
        covariance = np.zeros((6, 6), dtype=np.float64)
        acceleration_std = (
            float(self.config.horizontal_accel_std_mm_s2),
            float(self.config.horizontal_accel_std_mm_s2),
            float(self.config.vertical_accel_std_mm_s2),
        )
        base = np.array([
            [0.25 * dt_s ** 4, 0.5 * dt_s ** 3],
            [0.5 * dt_s ** 3, dt_s ** 2],
        ], dtype=np.float64)
        for axis, std in enumerate(acceleration_std):
            indices = [axis, axis + 3]
            covariance[np.ix_(indices, indices)] = max(std, 0.0) ** 2 * base
        return covariance

    def _initialize(
        self,
        position_mm: np.ndarray,
        timestamp_s: float,
        velocity_mm_s: np.ndarray | None = None,
    ) -> None:
        self.state[:3] = position_mm
        self.state[3:] = (
            np.zeros(3, dtype=np.float64)
            if velocity_mm_s is None else velocity_mm_s)
        lateral_var = float(self.config.lateral_measurement_std_mm) ** 2
        depth_var = float(self.config.depth_measurement_std_mm) ** 2
        self.covariance = np.diag([
            lateral_var,
            lateral_var,
            depth_var,
            1.0e6,
            1.0e6,
            1.0e6,
        ]).astype(np.float64)
        self.last_timestamp_s = timestamp_s
        self.initialized = True
        self.consecutive_outliers = 0

    def update(self, position_mm, timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Fuse one same-frame IR+depth 3D point and return position/velocity."""
        measurement = np.asarray(position_mm, dtype=np.float64).reshape(-1)
        timestamp = float(timestamp_s)
        if measurement.shape != (3,) or not np.all(np.isfinite(measurement)):
            return self.position_velocity()
        if not np.isfinite(timestamp):
            return self.position_velocity()

        if not self.initialized or self.last_timestamp_s is None:
            self._initialize(measurement, timestamp)
            self.total_updates += 1
            return self.position_velocity()

        raw_dt = timestamp - self.last_timestamp_s
        if raw_dt <= 1e-6:
            return self.position_velocity()
        dt_s = min(raw_dt, max(float(self.config.max_dt_s), 1e-4))
        transition = self._transition(dt_s)
        predicted_state = transition @ self.state
        predicted_covariance = (
            transition @ self.covariance @ transition.T
            + self._process_covariance(dt_s))

        observation = np.zeros((3, 6), dtype=np.float64)
        observation[:, :3] = np.eye(3, dtype=np.float64)
        innovation = measurement - observation @ predicted_state
        innovation_covariance = (
            observation @ predicted_covariance @ observation.T
            + self._measurement_covariance)
        try:
            inverse_innovation_covariance = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            inverse_innovation_covariance = np.linalg.pinv(
                innovation_covariance)
        mahalanobis_sq = float(
            innovation.T @ inverse_innovation_covariance @ innovation)
        self.last_innovation_mahalanobis = float(np.sqrt(max(
            mahalanobis_sq, 0.0)))

        gate = max(float(self.config.innovation_gate_sigma), 0.0)
        if gate > 0.0 and self.last_innovation_mahalanobis > gate:
            self.consecutive_outliers += 1
            self.outliers_rejected += 1
            self._reject_measurements.append((measurement.copy(), timestamp))
            self.state = predicted_state
            self.covariance = predicted_covariance
            self.last_timestamp_s = timestamp
            if self.consecutive_outliers >= max(
                    1, int(self.config.reset_after_outliers)):
                first_position, first_timestamp = self._reject_measurements[0]
                seed_dt_s = timestamp - first_timestamp
                velocity_seed = (
                    (measurement - first_position) / seed_dt_s
                    if seed_dt_s > 1e-4 else np.zeros(3, dtype=np.float64))
                self._initialize(
                    measurement, timestamp, velocity_mm_s=velocity_seed)
                self._reject_measurements = []
                self.reinitializations += 1
            self.total_updates += 1
            return self.position_velocity()

        self.consecutive_outliers = 0
        self._reject_measurements = []
        kalman_gain = (
            predicted_covariance @ observation.T
            @ inverse_innovation_covariance)
        self.state = predicted_state + kalman_gain @ innovation
        identity_minus_gain = np.eye(6) - kalman_gain @ observation
        self.covariance = (
            identity_minus_gain @ predicted_covariance
            @ identity_minus_gain.T
            + kalman_gain @ self._measurement_covariance @ kalman_gain.T)
        self.last_timestamp_s = timestamp
        self.total_updates += 1
        return self.position_velocity()

    def position_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        return self.state[:3].copy(), self.state[3:].copy()

    def get_stats(self) -> dict[str, float | int | bool]:
        return {
            "initialized": bool(self.initialized),
            "total_updates": int(self.total_updates),
            "outliers_rejected": int(self.outliers_rejected),
            "reinitializations": int(self.reinitializations),
            "innovation_mahalanobis": float(
                self.last_innovation_mahalanobis),
        }
