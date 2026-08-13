#!/usr/bin/env python3
"""Bounded high-FOV ball prediction for the policy observation path.

This module implements the trajectory model in Section III-B of
``IROS2249_ball_bouncing_final_version_peisen_xu_0715.pdf``:

* fit X/Y with first-order least squares and Z with second-order least
  squares over the latest N measured positions;
* propagate ``p_dot = v`` and
  ``v_dot = gravity - k_m * ||v|| * v`` with forward Euler integration.

It is deliberately *not* a general missing-detection tracker.  Ordinary
camera loss is predicted only when a continuous, upward-moving track leaves
the camera near the configured top-of-view height.  A separate caller-owned
    measurement guard may explicitly force the same bounded model after declaring
    sensor depth untrusted, and a caller-provided IR top-edge cue can replace the
    installation-dependent world-height gate while retaining the rising-track
    check.  Normal trusted measurements pass through unchanged.
The optional reacquisition handover still passes visible position through
immediately, but temporarily replaces the unavailable endpoint velocity with
a smooth model-to-detector transition.  In particular, a detector velocity
warm-up is never represented by a valid zero velocity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


# The control loop can observe the same camera frame through a raw timestamp
# path and a telemetry path that historically rounded to microseconds.  Treat
# sub-microsecond differences as the same measurement so a 200 Hz consumer
# cannot add one 90 Hz camera frame to the trajectory fit more than once.
_MEASUREMENT_TIMESTAMP_DUPLICATE_TOLERANCE_S = 1.0e-6


@dataclass(frozen=True)
class HighBallOcclusionConfig:
    """Configuration in SI units.

    The physical defaults reproduce Table I of the paper:
    ``m=2.7e-3 kg``, ``rho=1.29 kg/m^3``, ``S=1.26e-3 m^2`` and
    ``C_D=0.45``, which gives ``k_m=0.13545 1/m``.
    """

    history_frames: int = 10
    activation_z_m: float = 1.42
    min_upward_velocity_m_s: float = 0.20
    max_prediction_s: float = 0.45
    stop_prediction_z_m: float = 1.24
    drag_coefficient_per_m: float = 0.13545
    gravity_m_s2: float = 9.8
    integration_step_s: float = 0.005
    max_history_gap_s: float = 0.050
    min_history_span_s: float = 0.020
    activation_max_gap_s: float = 0.080
    fit_max_rmse_m: float = 0.030
    max_fitted_speed_m_s: float = 15.0
    reacquisition_gate_m: float = 0.18
    enable_reacquisition_velocity_handover: bool = False
    reacquisition_velocity_frames: int = 3
    reacquisition_velocity_blend_s: float = 0.040
    enable_control_time_compensation: bool = False
    control_time_compensation_max_s: float = 0.025
    enable_reacquisition_smoothing: bool = False
    reacquisition_smoothing_frames: int = 2
    reacquisition_smoothing_blend_s: float = 0.025
    enable_optimized_initial_state: bool = False

    def __post_init__(self) -> None:
        if int(self.history_frames) < 3:
            raise ValueError("history_frames must be at least 3")
        positive = {
            "max_prediction_s": self.max_prediction_s,
            "integration_step_s": self.integration_step_s,
            "max_history_gap_s": self.max_history_gap_s,
            "min_history_span_s": self.min_history_span_s,
            "activation_max_gap_s": self.activation_max_gap_s,
            "fit_max_rmse_m": self.fit_max_rmse_m,
            "max_fitted_speed_m_s": self.max_fitted_speed_m_s,
            "reacquisition_gate_m": self.reacquisition_gate_m,
            "reacquisition_velocity_blend_s": (
                self.reacquisition_velocity_blend_s),
            "control_time_compensation_max_s": (
                self.control_time_compensation_max_s),
            "reacquisition_smoothing_blend_s": (
                self.reacquisition_smoothing_blend_s),
        }
        if any(not np.isfinite(value) or value <= 0.0
               for value in positive.values()):
            raise ValueError("prediction time, fit, and speed limits must be positive")
        finite = (
            self.activation_z_m,
            self.min_upward_velocity_m_s,
            self.stop_prediction_z_m,
            self.drag_coefficient_per_m,
            self.gravity_m_s2,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("all physical parameters must be finite")
        if self.drag_coefficient_per_m < 0.0 or self.gravity_m_s2 <= 0.0:
            raise ValueError("drag must be non-negative and gravity must be positive")
        if int(self.reacquisition_velocity_frames) < 2:
            raise ValueError("reacquisition_velocity_frames must be at least 2")
        if int(self.reacquisition_smoothing_frames) < 2:
            raise ValueError("reacquisition_smoothing_frames must be at least 2")


@dataclass(frozen=True)
class HighBallOcclusionResult:
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    valid: bool
    predicted: bool
    timestamp_s: float
    diagnostics: dict[str, object]


def propagate_paper_model(
    position_m: Iterable[float],
    velocity_m_s: Iterable[float],
    elapsed_s: float,
    *,
    drag_coefficient_per_m: float = 0.13545,
    gravity_m_s2: float = 9.8,
    integration_step_s: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-Euler propagation of Eqs. (2) and (4) in the paper."""

    position = np.asarray(position_m, dtype=np.float64).reshape(3).copy()
    velocity = np.asarray(velocity_m_s, dtype=np.float64).reshape(3).copy()
    elapsed = max(0.0, float(elapsed_s))
    if elapsed <= 0.0:
        return position, velocity
    max_step = max(1.0e-5, float(integration_step_s))
    steps = max(1, int(np.ceil(elapsed / max_step)))
    step = elapsed / float(steps)
    gravity = np.array([0.0, 0.0, -float(gravity_m_s2)], dtype=np.float64)
    drag = max(0.0, float(drag_coefficient_per_m))
    for _ in range(steps):
        previous_velocity = velocity.copy()
        acceleration = gravity - drag * np.linalg.norm(previous_velocity) * previous_velocity
        position = position + previous_velocity * step
        velocity = previous_velocity + acceleration * step
    return position, velocity


class HighBallOcclusionPredictor:
    """State machine for a high, rising ball that temporarily leaves the FOV."""

    # Offline fit across all usable high-FOV/depth-fault transitions favored a
    # nine-frame robust gravity-constrained initialization.  Keep its endpoint
    # velocity physically neutral: older per-axis 0.95 scales and a +0.04 m/s
    # vertical bias were ad-hoc calibration terms, not part of the paper model,
    # and regressed both record_new4 and the 2026-08-12 clean handovers.
    OPTIMIZED_HISTORY_FRAMES = 9
    OPTIMIZED_ACTIVATION_Z_M = 1.40
    # Recent 90 Hz recordings contain valid leave-and-return arcs lasting a
    # little over 0.40 s.  The old 0.35 s override expired before the first
    # trustworthy reacquisition frame even though the public configuration
    # allowed 0.45 s.  Keep a small margin for dropped camera frames.
    OPTIMIZED_MAX_PREDICTION_S = 0.50
    OPTIMIZED_REACQUISITION_GATE_M = 0.12
    OPTIMIZED_X_VELOCITY_SCALE = 1.0
    OPTIMIZED_Y_VELOCITY_SCALE = 1.0
    OPTIMIZED_VERTICAL_VELOCITY_BIAS_M_S = 0.0
    # Depth faults have far fewer reliable recorded handovers.  Keep their
    # calibrated initialization neutral instead of importing the high-FOV
    # vertical bias, which regressed the available depth-fault transitions.
    OPTIMIZED_DEPTH_FAULT_X_VELOCITY_SCALE = 1.0
    OPTIMIZED_DEPTH_FAULT_Y_VELOCITY_SCALE = 1.0
    OPTIMIZED_DEPTH_FAULT_VERTICAL_VELOCITY_BIAS_M_S = 0.0
    # A depth guard necessarily fires after the first bad stereo evidence.
    # If the full fit reverses a well-established lateral direction, discard
    # at most the last three samples (about 33 ms at 90 Hz) and extrapolate the
    # clean prefix to the activation time.  This is a bounded fault-only
    # defence, not a longer smoothing window.
    DEPTH_FAULT_TAIL_CHECK_FRAMES = 3
    DEPTH_FAULT_MIN_REFERENCE_SPEED_M_S = 0.08
    DEPTH_FAULT_MIN_SLOPE_DISAGREEMENT_M_S = 0.20

    def __init__(self, config: HighBallOcclusionConfig | None = None) -> None:
        self.config = config or HighBallOcclusionConfig()
        self._history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(
                int(self.config.history_frames),
                (self.OPTIMIZED_HISTORY_FRAMES
                 if self.config.enable_optimized_initial_state else 0)))
        self._active = False
        self._initial_time_s: float | None = None
        self._initial_position_m: np.ndarray | None = None
        self._initial_velocity_m_s: np.ndarray | None = None
        self._fit_rmse_m: np.ndarray | None = None
        self._active_trigger: str | None = None
        self._last_trusted_time_s: float | None = None
        self._last_trusted_position_m: np.ndarray | None = None
        self._last_trusted_velocity_m_s: np.ndarray | None = None
        self._activation_count = 0
        self._reacquisition_count = 0
        self._reacquisition_rejected_count = 0
        self._expiration_count = 0
        self._last_reacquisition_error_m: np.ndarray | None = None
        self._handover_active = False
        self._handover_mode = "velocity"
        self._handover_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=max(
                2,
                int(self.config.reacquisition_velocity_frames),
                int(self.config.reacquisition_smoothing_frames),
            ))
        self._handover_ready_time_s: float | None = None
        self._handover_last_ready_time_s: float | None = None
        self._handover_start_time_s: float | None = None
        self._handover_target_velocity_m_s: np.ndarray | None = None
        self._handover_count = 0
        self._handover_completion_count = 0
        self._velocity_warmup_bridge_count = 0
        self._velocity_warmup_bridge_rejected_count = 0
        self._last_velocity_warmup_rejection_error_m: (
            np.ndarray | None) = None
        self._handover_forced_by_position_only = False
        self._handover_gate_confirmed = False
        self._last_handover_alpha = 0.0
        self._last_fit_tail_trimmed_samples = 0
        self._last_control_time_compensation_s = 0.0
        self._last_reason = "idle"

    @property
    def active(self) -> bool:
        return bool(self._active)

    def reset(self, *, clear_counters: bool = False) -> None:
        self._history.clear()
        self._clear_active()
        self._last_reason = "reset"
        self._last_reacquisition_error_m = None
        self._last_velocity_warmup_rejection_error_m = None
        self._last_handover_alpha = 0.0
        self._last_fit_tail_trimmed_samples = 0
        self._last_trusted_time_s = None
        self._last_trusted_position_m = None
        self._last_trusted_velocity_m_s = None
        if clear_counters:
            self._activation_count = 0
            self._reacquisition_count = 0
            self._reacquisition_rejected_count = 0
            self._expiration_count = 0
            self._velocity_warmup_bridge_count = 0
            self._velocity_warmup_bridge_rejected_count = 0

    def step(
        self,
        position_m: Iterable[float],
        velocity_m_s: Iterable[float],
        measured_valid: bool,
        measurement_timestamp_s: float | None,
        query_timestamp_s: float,
        *,
        force_prediction_reason: str | None = None,
        allow_high_fov_prediction: bool = True,
        measurement_velocity_valid: bool = True,
        measurement_reacquisition_confirmed: bool = False,
    ) -> HighBallOcclusionResult:
        """Pass through a measurement or produce a bounded predicted state.

        ``force_prediction_reason`` is used by an external measurement guard
        after it has already declared the current measurement untrusted.  A
        sensor-depth fault bypasses both high-FOV activation gates; an explicit
        ``high_fov_edge_exit`` bypasses only world height and must still fit a
        rising trajectory.  The measurement itself must still be passed with
        ``measured_valid=False`` so it cannot enter the trusted history.
        """

        query_time = float(query_timestamp_s)
        if not np.isfinite(query_time):
            raise ValueError("query_timestamp_s must be finite")
        position = np.asarray(position_m, dtype=np.float64).reshape(3)
        velocity = np.asarray(velocity_m_s, dtype=np.float64).reshape(3)
        position_valid = bool(
            measured_valid and np.all(np.isfinite(position)))
        velocity_valid = bool(
            measurement_velocity_valid
            and np.all(np.isfinite(velocity)))
        valid = bool(position_valid and velocity_valid)
        measurement_time = self._measurement_time(
            measurement_timestamp_s, query_time)
        forced_reason = (
            str(force_prediction_reason).strip()
            if force_prediction_reason else None)
        self._last_control_time_compensation_s = 0.0

        if position_valid and not velocity_valid:
            if self._handover_active:
                return self._step_reacquisition_handover(
                    position, velocity, False,
                    measurement_time, query_time)
            if self._active:
                predicted_at_measurement, predicted_velocity = (
                    self._predict_at(measurement_time))
                self._last_reacquisition_error_m = (
                    position.copy() - predicted_at_measurement)
                prediction_elapsed_s = max(
                    0.0, query_time - float(self._initial_time_s))
                if (not measurement_reacquisition_confirmed
                        and np.linalg.norm(self._last_reacquisition_error_m)
                        > self._effective_reacquisition_gate_m()):
                    self._reacquisition_rejected_count += 1
                    if (prediction_elapsed_s
                            > self._effective_max_prediction_s()):
                        self._expiration_count += 1
                        self._clear_active()
                        self._history.clear()
                        self._last_reason = (
                            "reacquisition_after_timeout_rejected")
                        return self._result(
                            np.zeros(3), np.zeros(3), False, False,
                            query_time, reason=self._last_reason,
                            elapsed_s=prediction_elapsed_s)
                    predicted_position, predicted_velocity = self._predict_at(
                        query_time)
                    self._last_reason = "reacquisition_outside_gate"
                    return self._result(
                        predicted_position, predicted_velocity,
                        True, True, query_time,
                        reason=self._last_reason,
                        elapsed_s=prediction_elapsed_s)
                self._reacquisition_count += 1
                self._history.clear()
                if self.config.enable_reacquisition_smoothing:
                    # The detector deliberately marks the first reacquired
                    # positions velocity-unready after clearing its segment.
                    # This is exactly the transition where full-state
                    # smoothing is needed most; publishing the fresh position
                    # immediately used to create a visible position jump while
                    # retaining model velocity from a different state.
                    self._start_reacquisition_handover(
                        measurement_time, position, mode="state",
                        forced_position_only=True,
                        gate_confirmed=measurement_reacquisition_confirmed)
                    self._last_reason = (
                        "reacquisition_position_only_state_smoothing")
                    return self._result(
                        predicted_at_measurement, predicted_velocity,
                        True, True, measurement_time,
                        reason=self._last_reason,
                        elapsed_s=prediction_elapsed_s,
                        state="reacquisition_smoothing",
                        position_predicted=True,
                        velocity_predicted=True)
                self._start_reacquisition_handover(
                    measurement_time, position, mode="velocity",
                    forced_position_only=True,
                    gate_confirmed=measurement_reacquisition_confirmed)
                self._last_reason = "reacquisition_position_only"
                return self._result(
                    position, predicted_velocity, True, True,
                    measurement_time, reason=self._last_reason,
                    elapsed_s=prediction_elapsed_s,
                    state="reacquisition_handover",
                    position_predicted=False,
                    velocity_predicted=True)
            # A detector segment can restart even when high-FOV prediction was
            # not activated (ordinary miss, rejected depth, live reconfigure,
            # etc.).  If a complete state exists, keep its causal ballistic
            # velocity until the detector's new segment is actually ready.
            # Previously this path published the detector's placeholder
            # [0, 0, 0] as a valid velocity, producing a nonzero -> zero ->
            # nonzero impulse at every five-frame velocity restart.
            warmup_velocity = self._start_velocity_warmup_bridge(
                measurement_time, position)
            if warmup_velocity is not None:
                self._last_reason = "velocity_warmup_bridge_started"
                return self._result(
                    position, warmup_velocity, True, True, measurement_time,
                    reason=self._last_reason, elapsed_s=0.0,
                    state="reacquisition_handover",
                    position_predicted=False,
                    velocity_predicted=True)

            # At process start there is no safe velocity to bridge.  Keep the
            # trustworthy position for diagnostics/history, but invalidate the
            # incomplete 6D state instead of labelling a zero velocity valid.
            self._append_measurement(measurement_time, position)
            self._last_reason = (
                "velocity_warmup_outside_gate"
                if self._last_velocity_warmup_rejection_error_m is not None
                else "measured_position_only")
            return self._result(
                position, np.zeros(3), False, False, measurement_time,
                reason=self._last_reason, elapsed_s=0.0,
                state="tracking_velocity_warmup",
                position_predicted=False,
                velocity_predicted=False)

        if valid:
            if self._handover_active:
                return self._step_reacquisition_handover(
                    position, velocity, True,
                    measurement_time, query_time)
            if self._active:
                active_trigger = self._active_trigger
                predicted_at_measurement, predicted_velocity = self._predict_at(
                    measurement_time)
                self._last_reacquisition_error_m = (
                    position.copy() - predicted_at_measurement)
                prediction_elapsed_s = max(
                    0.0, query_time - float(self._initial_time_s))
                if (not measurement_reacquisition_confirmed
                        and np.linalg.norm(self._last_reacquisition_error_m)
                        > self._effective_reacquisition_gate_m()):
                    self._reacquisition_rejected_count += 1
                    if (prediction_elapsed_s
                            > self._effective_max_prediction_s()):
                        self._expiration_count += 1
                        self._clear_active()
                        self._history.clear()
                        self._last_reason = "reacquisition_after_timeout_rejected"
                        return self._result(
                            np.zeros(3), np.zeros(3), False, False,
                            query_time,
                            reason=self._last_reason,
                            elapsed_s=prediction_elapsed_s,
                        )
                    predicted_position, predicted_velocity = self._predict_at(
                        query_time)
                    self._last_reason = "reacquisition_outside_gate"
                    return self._result(
                        predicted_position, predicted_velocity,
                        True, True, query_time,
                        reason=self._last_reason,
                        elapsed_s=prediction_elapsed_s,
                    )
                self._reacquisition_count += 1
                self._history.clear()
                if self.config.enable_reacquisition_smoothing:
                    self._start_reacquisition_handover(
                        measurement_time, position, mode="state",
                        gate_confirmed=measurement_reacquisition_confirmed)
                    self._last_reason = "reacquisition_state_smoothing"
                    return self._result(
                        predicted_at_measurement, predicted_velocity,
                        True, True, measurement_time,
                        reason=self._last_reason,
                        elapsed_s=prediction_elapsed_s,
                        state="reacquisition_smoothing",
                        position_predicted=True,
                        velocity_predicted=True,
                    )
                confirmed_safety_handover = bool(
                    measurement_reacquisition_confirmed
                    and not bool(
                        self.config.enable_reacquisition_velocity_handover))
                if (self.config.enable_reacquisition_velocity_handover
                        or measurement_reacquisition_confirmed):
                    self._start_reacquisition_handover(
                        measurement_time, position, mode="velocity",
                        forced_position_only=confirmed_safety_handover,
                        gate_confirmed=measurement_reacquisition_confirmed)
                    self._last_reason = (
                        "reacquisition_confirmed_safety_handover"
                        if confirmed_safety_handover
                        else "reacquisition_velocity_handover")
                    return self._result(
                        position, predicted_velocity, True, True,
                        measurement_time,
                        reason=self._last_reason,
                        elapsed_s=prediction_elapsed_s,
                        state="reacquisition_handover",
                        position_predicted=False,
                        velocity_predicted=True,
                    )
                self._clear_active()
                # Do not fit across the unobserved arc.  The next activation
                # requires a new continuous set of visible samples.
                self._last_reason = (
                    "sensor_depth_reacquired"
                    if active_trigger == "sensor_depth_anomaly"
                    else "reacquired")
            else:
                self._last_reason = "measured"
            self._append_measurement(measurement_time, position)
            self._remember_trusted_state(
                measurement_time, position, velocity)
            if self.config.enable_control_time_compensation:
                horizon_s = min(
                    max(0.0, query_time - measurement_time),
                    float(self.config.control_time_compensation_max_s),
                )
                if horizon_s > _MEASUREMENT_TIMESTAMP_DUPLICATE_TOLERANCE_S:
                    compensated_position, compensated_velocity = (
                        propagate_paper_model(
                            position,
                            velocity,
                            horizon_s,
                            drag_coefficient_per_m=(
                                self.config.drag_coefficient_per_m),
                            gravity_m_s2=self.config.gravity_m_s2,
                            integration_step_s=self.config.integration_step_s,
                        ))
                    self._last_control_time_compensation_s = horizon_s
                    self._last_reason = "measured_control_time_compensated"
                    return self._result(
                        compensated_position, compensated_velocity,
                        True, False, measurement_time + horizon_s,
                        reason=self._last_reason,
                        elapsed_s=0.0,
                        state="tracking_latency_compensation",
                        position_predicted=True,
                        velocity_predicted=True,
                    )
            self._last_control_time_compensation_s = 0.0
            return self._result(
                position, velocity, True, False, measurement_time,
                reason=self._last_reason, elapsed_s=0.0)

        if not self._active:
            if forced_reason is not None:
                edge_exit_trigger = bool(
                    forced_reason == "high_fov_edge_exit")
                activated, reason = self._try_activate(
                    query_time,
                    trigger=forced_reason,
                    # A sensor-depth fault has already invalidated 3D range,
                    # so both high-FOV gates are irrelevant.  A verified 2D
                    # top-edge exit bypasses only the installation-dependent
                    # world-Z gate; its fitted trajectory must still be rising.
                    bypass_high_fov_gates=not edge_exit_trigger,
                    bypass_activation_height=edge_exit_trigger,
                )
            elif allow_high_fov_prediction:
                activated, reason = self._try_activate(query_time)
            else:
                activated, reason = False, "prediction_trigger_disabled"
            if not activated:
                self._last_reason = reason
                return self._result(
                    np.zeros(3), np.zeros(3), False, False, query_time,
                    reason=reason, elapsed_s=None)

        assert self._initial_time_s is not None
        elapsed = max(0.0, query_time - self._initial_time_s)
        if elapsed > self._effective_max_prediction_s() + 1.0e-9:
            self._expiration_count += 1
            self._clear_active()
            self._history.clear()
            self._last_reason = "prediction_timeout"
            return self._result(
                np.zeros(3), np.zeros(3), False, False, query_time,
                reason=self._last_reason, elapsed_s=elapsed)

        predicted_position, predicted_velocity = self._predict_at(query_time)
        if (self._active_trigger not in {
                "sensor_depth_anomaly", "high_fov_edge_exit"}
                and predicted_position[2] < self.config.stop_prediction_z_m):
            self._expiration_count += 1
            self._clear_active()
            self._history.clear()
            self._last_reason = "prediction_below_stop_z"
            return self._result(
                np.zeros(3), np.zeros(3), False, False, query_time,
                reason=self._last_reason, elapsed_s=elapsed)

        if self._active_trigger == "sensor_depth_anomaly":
            self._last_reason = "predicting_sensor_depth_anomaly"
        elif self._active_trigger == "high_fov_edge_exit":
            self._last_reason = "predicting_high_fov_edge_exit"
        else:
            self._last_reason = "predicting_high_fov_occlusion"
        return self._result(
            predicted_position, predicted_velocity, True, True, query_time,
            reason=self._last_reason, elapsed_s=elapsed)

    def diagnostics(self) -> dict[str, object]:
        span = self._history_span_s()
        return {
            "model": (
                "robust_xy_gravity_z_neutral_paper_drag"
                if self.config.enable_optimized_initial_state
                else "iros2249_ls_gravity_quadratic_drag"),
            "active": bool(self._active),
            "activeTrigger": self._active_trigger,
            "historySamples": len(self._history),
            "historySpanS": None if span is None else round(span, 6),
            "historyFrames": int(
                self.OPTIMIZED_HISTORY_FRAMES
                if self.config.enable_optimized_initial_state
                else self.config.history_frames),
            "activationZMinM": float(self._effective_activation_z_m()),
            "minUpwardVelocityMps": float(
                self.config.min_upward_velocity_m_s),
            "maxPredictionS": float(self._effective_max_prediction_s()),
            "stopPredictionZMinM": float(self.config.stop_prediction_z_m),
            "dragCoefficientPerM": float(
                self.config.drag_coefficient_per_m),
            "gravityMps2": float(self.config.gravity_m_s2),
            "integrationStepS": float(self.config.integration_step_s),
            "activationCount": int(self._activation_count),
            "reacquisitionCount": int(self._reacquisition_count),
            "reacquisitionRejectedCount": int(
                self._reacquisition_rejected_count),
            "reacquisitionGateM": float(
                self._effective_reacquisition_gate_m()),
            "optimizedInitialStateEnabled": bool(
                self.config.enable_optimized_initial_state),
            "optimizedHistoryFrames": int(self.OPTIMIZED_HISTORY_FRAMES),
            "optimizedVelocityScale": [
                self.OPTIMIZED_X_VELOCITY_SCALE,
                self.OPTIMIZED_Y_VELOCITY_SCALE,
                1.0,
            ],
            "optimizedVerticalVelocityBiasMps": float(
                self.OPTIMIZED_VERTICAL_VELOCITY_BIAS_M_S),
            "optimizedDepthFaultVelocityScale": [
                self.OPTIMIZED_DEPTH_FAULT_X_VELOCITY_SCALE,
                self.OPTIMIZED_DEPTH_FAULT_Y_VELOCITY_SCALE,
                1.0,
            ],
            "optimizedDepthFaultVerticalVelocityBiasMps": float(
                self.OPTIMIZED_DEPTH_FAULT_VERTICAL_VELOCITY_BIAS_M_S),
            "controlTimeCompensationEnabled": bool(
                self.config.enable_control_time_compensation),
            "controlTimeCompensationMaxS": float(
                self.config.control_time_compensation_max_s),
            "controlTimeCompensationAppliedS": round(
                float(self._last_control_time_compensation_s), 6),
            "reacquisitionSmoothingEnabled": bool(
                self.config.enable_reacquisition_smoothing),
            "reacquisitionSmoothingFrames": int(
                self.config.reacquisition_smoothing_frames),
            "reacquisitionSmoothingBlendS": float(
                self.config.reacquisition_smoothing_blend_s),
            "reacquisitionHandoverMode": (
                self._handover_mode if self._handover_active else None),
            "reacquisitionVelocityHandoverEnabled": bool(
                self.config.enable_reacquisition_velocity_handover),
            "reacquisitionVelocityHandoverActive": bool(
                self._handover_active),
            "reacquisitionPositionOnlySafetyHandover": bool(
                self._handover_active
                and self._handover_forced_by_position_only),
            "reacquisitionMeasurementConfirmed": bool(
                self._handover_active and self._handover_gate_confirmed),
            "reacquisitionVelocityFrames": int(
                self.config.reacquisition_velocity_frames),
            "reacquisitionVelocityBlendS": float(
                self.config.reacquisition_velocity_blend_s),
            "reacquisitionVelocitySamples": len(self._handover_history),
            "reacquisitionVelocityHandoverCount": int(
                self._handover_count),
            "reacquisitionVelocityHandoverCompletionCount": int(
                self._handover_completion_count),
            "velocityWarmupBridgeCount": int(
                self._velocity_warmup_bridge_count),
            "velocityWarmupBridgeRejectedCount": int(
                self._velocity_warmup_bridge_rejected_count),
            "velocityWarmupRejectionErrorMm": self._vector_or_none(
                None
                if self._last_velocity_warmup_rejection_error_m is None
                else self._last_velocity_warmup_rejection_error_m * 1000.0),
            "velocityWarmupRejectionErrorNormMm": (
                None
                if self._last_velocity_warmup_rejection_error_m is None
                else round(float(np.linalg.norm(
                    self._last_velocity_warmup_rejection_error_m))
                    * 1000.0, 6)),
            "reacquisitionVelocityHandoverAlpha": round(
                float(self._last_handover_alpha), 6),
            "reacquisitionFittedVelocityMps": self._vector_or_none(
                self._handover_target_velocity_m_s),
            "reacquisitionTargetVelocityMps": self._vector_or_none(
                self._handover_target_velocity_m_s),
            "reacquisitionVelocitySource": (
                "detector_ready"
                if self._handover_target_velocity_m_s is not None else None),
            "expirationCount": int(self._expiration_count),
            "lastReason": self._last_reason,
            "initialTimeS": self._initial_time_s,
            "initialPositionM": self._vector_or_none(
                self._initial_position_m),
            "initialVelocityMps": self._vector_or_none(
                self._initial_velocity_m_s),
            "fitRmseMm": self._vector_or_none(
                None if self._fit_rmse_m is None
                else self._fit_rmse_m * 1000.0),
            "depthFaultTailTrimmedSamples": int(
                self._last_fit_tail_trimmed_samples),
            "lastReacquisitionErrorMm": self._vector_or_none(
                None if self._last_reacquisition_error_m is None
                else self._last_reacquisition_error_m * 1000.0),
            "lastReacquisitionErrorNormMm": (
                None if self._last_reacquisition_error_m is None
                else round(float(np.linalg.norm(
                    self._last_reacquisition_error_m)) * 1000.0, 6)),
        }

    def _append_measurement(self, timestamp_s: float, position: np.ndarray) -> None:
        if self._history:
            previous_time = self._history[-1][0]
            dt = timestamp_s - previous_time
            if dt <= _MEASUREMENT_TIMESTAMP_DUPLICATE_TOLERANCE_S:
                return
            if dt > self.config.max_history_gap_s + 1.0e-9:
                self._history.clear()
        self._history.append((timestamp_s, position.copy()))

    def _remember_trusted_state(
        self,
        timestamp_s: float,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
    ) -> None:
        self._last_trusted_time_s = float(timestamp_s)
        self._last_trusted_position_m = np.asarray(
            position_m, dtype=np.float64).reshape(3).copy()
        self._last_trusted_velocity_m_s = np.asarray(
            velocity_m_s, dtype=np.float64).reshape(3).copy()

    def _try_activate(
        self,
        query_time_s: float,
        *,
        trigger: str = "high_fov_occlusion",
        bypass_high_fov_gates: bool = False,
        bypass_activation_height: bool = False,
    ) -> tuple[bool, str]:
        configured_history = (
            min(int(self.config.history_frames), self.OPTIMIZED_HISTORY_FRAMES)
            if self.config.enable_optimized_initial_state
            else int(self.config.history_frames))
        forced_activation = bool(
            bypass_high_fov_gates or bypass_activation_height)
        required_history = (
            min(3, int(self.config.history_frames))
            if forced_activation
            else configured_history)
        if len(self._history) < required_history:
            if bypass_high_fov_gates:
                return self._activate_from_last_trusted_state(
                    query_time_s, trigger)
            return False, "insufficient_history"
        last_time, last_position = self._history[-1]
        gap_s = query_time_s - last_time
        activation_gap_limit_s = float(self.config.activation_max_gap_s)
        if forced_activation:
            # A late explicit fault/edge cue may arrive after several missed
            # frames.  It is still safe to initialize from the last trajectory
            # while it remains inside the bounded prediction horizon.
            activation_gap_limit_s = max(
                activation_gap_limit_s,
                self._effective_max_prediction_s())
        if gap_s < -1.0e-9 or gap_s > activation_gap_limit_s:
            return False, "activation_gap_out_of_range"
        span = self._history_span_s()
        if span is None or span < self.config.min_history_span_s:
            return False, "history_span_too_short"
        if (not bypass_high_fov_gates
                and not bypass_activation_height
                and last_position[2] < self._effective_activation_z_m()):
            return False, "last_measurement_below_activation_z"

        initial_position, initial_velocity, fit_rmse = self._fit_initial_state(
            trigger=trigger)
        if np.max(fit_rmse) > self.config.fit_max_rmse_m:
            return False, "trajectory_fit_residual_too_large"
        if np.linalg.norm(initial_velocity) > self.config.max_fitted_speed_m_s:
            return False, "fitted_speed_too_large"
        if (not bypass_high_fov_gates
                and initial_velocity[2]
                < self.config.min_upward_velocity_m_s):
            return False, "ball_not_rising_at_loss"

        self._active = True
        self._initial_time_s = last_time
        self._initial_position_m = initial_position
        self._initial_velocity_m_s = initial_velocity
        self._fit_rmse_m = fit_rmse
        self._active_trigger = str(trigger)
        self._activation_count += 1
        return True, "activated"

    def _activate_from_last_trusted_state(
        self,
        query_time_s: float,
        trigger: str,
    ) -> tuple[bool, str]:
        """Seed a forced short dropout from the last complete 6D state.

        Adaptive recovery already requires three clean depth frames, but a
        second fault can occur before the configured ten-position LS window
        has refilled.  The last trusted detector velocity keeps prediction
        available in that case without admitting the rejected depth frame.
        """
        if (self._last_trusted_time_s is None
                or self._last_trusted_position_m is None
                or self._last_trusted_velocity_m_s is None):
            return False, "insufficient_history"
        gap_s = float(query_time_s - self._last_trusted_time_s)
        activation_gap_limit_s = max(
            float(self.config.activation_max_gap_s),
            self._effective_max_prediction_s())
        if gap_s < -1.0e-9 or gap_s > activation_gap_limit_s:
            return False, "activation_gap_out_of_range"
        if (not np.all(np.isfinite(self._last_trusted_position_m))
                or not np.all(np.isfinite(self._last_trusted_velocity_m_s))
                or np.linalg.norm(self._last_trusted_velocity_m_s)
                > self.config.max_fitted_speed_m_s):
            return False, "last_trusted_state_invalid"
        self._active = True
        self._initial_time_s = float(self._last_trusted_time_s)
        self._initial_position_m = self._last_trusted_position_m.copy()
        self._initial_velocity_m_s = self._last_trusted_velocity_m_s.copy()
        self._fit_rmse_m = None
        self._active_trigger = str(trigger)
        self._activation_count += 1
        return True, "activated_from_last_trusted_state"

    def _fit_initial_state(
        self,
        trigger: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = list(self._history)
        if self.config.enable_optimized_initial_state:
            samples = samples[-self.OPTIMIZED_HISTORY_FRAMES:]
        activation_time_s = float(samples[-1][0])
        self._last_fit_tail_trimmed_samples = 0
        if (self.config.enable_optimized_initial_state
                and str(trigger) == "sensor_depth_anomaly"):
            trim = int(self.DEPTH_FAULT_TAIL_CHECK_FRAMES)
            reference_samples = samples[:-trim]
            if len(reference_samples) >= 5:
                full_times = np.asarray(
                    [sample[0] for sample in samples], dtype=float)
                full_times -= activation_time_s
                reference_times = full_times[:-trim]
                direction_reversed = False
                for axis in (0, 1):
                    full_values = np.asarray(
                        [sample[1][axis] for sample in samples], dtype=float)
                    full_slope = float(self._robust_line_fit(
                        full_times, full_values)[0])
                    reference_slope = float(self._robust_line_fit(
                        reference_times, full_values[:-trim])[0])
                    if (full_slope * reference_slope < 0.0
                            and abs(full_slope)
                            >= self.DEPTH_FAULT_MIN_REFERENCE_SPEED_M_S
                            and abs(reference_slope)
                            >= self.DEPTH_FAULT_MIN_REFERENCE_SPEED_M_S
                            and abs(full_slope - reference_slope)
                            >= self.DEPTH_FAULT_MIN_SLOPE_DISAGREEMENT_M_S):
                        direction_reversed = True
                        break
                if direction_reversed:
                    samples = reference_samples
                    self._last_fit_tail_trimmed_samples = trim
        times = np.asarray([sample[0] for sample in samples], dtype=float)
        positions = np.asarray([sample[1] for sample in samples], dtype=float)
        # Keep the state timestamp at the most recent trusted frame even when
        # a contaminated tail was removed.  The prefix fit therefore
        # extrapolates only the bounded discarded interval instead of making
        # the prediction artificially older.
        relative_time = times - activation_time_s

        linear_design = np.column_stack((relative_time, np.ones_like(relative_time)))
        initial_position = np.zeros(3, dtype=np.float64)
        initial_velocity = np.zeros(3, dtype=np.float64)
        residual = np.zeros_like(positions)
        for axis in (0, 1):
            if self.config.enable_optimized_initial_state:
                coefficient = self._robust_line_fit(
                    relative_time, positions[:, axis])
            else:
                coefficient, *_ = np.linalg.lstsq(
                    linear_design, positions[:, axis], rcond=None)
            initial_velocity[axis] = coefficient[0]
            initial_position[axis] = coefficient[1]
            residual[:, axis] = positions[:, axis] - linear_design @ coefficient
        if self.config.enable_optimized_initial_state:
            gravity_compensated_z = (
                positions[:, 2]
                + 0.5 * float(self.config.gravity_m_s2)
                * relative_time ** 2)
            vertical_coefficient = self._robust_line_fit(
                relative_time, gravity_compensated_z)
            is_depth_fault = str(trigger) == "sensor_depth_anomaly"
            x_scale = (
                self.OPTIMIZED_DEPTH_FAULT_X_VELOCITY_SCALE
                if is_depth_fault else self.OPTIMIZED_X_VELOCITY_SCALE)
            y_scale = (
                self.OPTIMIZED_DEPTH_FAULT_Y_VELOCITY_SCALE
                if is_depth_fault else self.OPTIMIZED_Y_VELOCITY_SCALE)
            vertical_bias = (
                self.OPTIMIZED_DEPTH_FAULT_VERTICAL_VELOCITY_BIAS_M_S
                if is_depth_fault
                else self.OPTIMIZED_VERTICAL_VELOCITY_BIAS_M_S)
            initial_velocity[0] *= x_scale
            initial_velocity[1] *= y_scale
            initial_velocity[2] = (
                vertical_coefficient[0]
                + vertical_bias)
            initial_position[2] = vertical_coefficient[1]
            fitted_z = (
                linear_design @ vertical_coefficient
                - 0.5 * float(self.config.gravity_m_s2)
                * relative_time ** 2)
            residual[:, 2] = positions[:, 2] - fitted_z
        else:
            vertical_design = np.column_stack((
                relative_time * relative_time,
                relative_time,
                np.ones_like(relative_time),
            ))
            vertical_coefficient, *_ = np.linalg.lstsq(
                vertical_design, positions[:, 2], rcond=None)
            initial_velocity[2] = vertical_coefficient[1]
            initial_position[2] = vertical_coefficient[2]
            residual[:, 2] = (
                positions[:, 2] - vertical_design @ vertical_coefficient)
        rmse = np.sqrt(np.mean(residual * residual, axis=0))
        return initial_position, initial_velocity, rmse

    @staticmethod
    def _robust_line_fit(relative_time: np.ndarray, values: np.ndarray) -> np.ndarray:
        design = np.column_stack((relative_time, np.ones_like(relative_time)))
        coefficient, *_ = np.linalg.lstsq(design, values, rcond=None)
        for _ in range(4):
            residual = values - design @ coefficient
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            scale = max(1.0e-3, 1.4826 * mad)
            cutoff = 1.5 * scale
            absolute = np.abs(residual)
            weights = np.ones_like(absolute)
            mask = absolute > cutoff
            weights[mask] = cutoff / np.maximum(absolute[mask], 1.0e-12)
            root = np.sqrt(weights)
            updated, *_ = np.linalg.lstsq(
                design * root[:, None], values * root, rcond=None)
            coefficient = updated
        return coefficient

    def _effective_activation_z_m(self) -> float:
        if self.config.enable_optimized_initial_state:
            return float(self.OPTIMIZED_ACTIVATION_Z_M)
        return float(self.config.activation_z_m)

    def _effective_max_prediction_s(self) -> float:
        if self.config.enable_optimized_initial_state:
            return float(self.OPTIMIZED_MAX_PREDICTION_S)
        return float(self.config.max_prediction_s)

    def _effective_reacquisition_gate_m(self) -> float:
        if self.config.enable_optimized_initial_state:
            return float(self.OPTIMIZED_REACQUISITION_GATE_M)
        return float(self.config.reacquisition_gate_m)

    def _predict_at(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
        assert self._initial_time_s is not None
        assert self._initial_position_m is not None
        assert self._initial_velocity_m_s is not None
        return propagate_paper_model(
            self._initial_position_m,
            self._initial_velocity_m_s,
            max(0.0, float(timestamp_s) - self._initial_time_s),
            drag_coefficient_per_m=self.config.drag_coefficient_per_m,
            gravity_m_s2=self.config.gravity_m_s2,
            integration_step_s=self.config.integration_step_s,
        )

    def _start_reacquisition_handover(
        self,
        timestamp_s: float,
        position_m: np.ndarray,
        *,
        mode: str = "velocity",
        forced_position_only: bool = False,
        gate_confirmed: bool = False,
    ) -> None:
        """Keep model velocity until the detector owns velocity again."""
        self._handover_active = True
        self._handover_mode = str(mode)
        self._handover_history.clear()
        self._handover_ready_time_s = None
        self._handover_last_ready_time_s = None
        self._handover_start_time_s = float(timestamp_s)
        self._handover_target_velocity_m_s = None
        self._handover_forced_by_position_only = bool(
            forced_position_only)
        self._handover_gate_confirmed = bool(gate_confirmed)
        self._last_handover_alpha = 0.0
        self._handover_count += 1
        self._append_handover_measurement(timestamp_s, position_m)

    def _start_velocity_warmup_bridge(
        self,
        timestamp_s: float,
        position_m: np.ndarray,
    ) -> np.ndarray | None:
        """Bridge an incomplete detector state from the last trusted 6D state.

        This is intentionally independent of the high-FOV activation gates.
        The position is already a trusted, visible measurement; only velocity
        is unavailable while the detector refills its causal fit window.
        """
        self._last_velocity_warmup_rejection_error_m = None
        if (self._last_trusted_time_s is None
                or self._last_trusted_position_m is None
                or self._last_trusted_velocity_m_s is None):
            return None
        gap_s = float(timestamp_s - self._last_trusted_time_s)
        if (gap_s < -_MEASUREMENT_TIMESTAMP_DUPLICATE_TOLERANCE_S
                or gap_s > self._effective_max_prediction_s()):
            return None
        if (not np.all(np.isfinite(self._last_trusted_position_m))
                or not np.all(np.isfinite(self._last_trusted_velocity_m_s))):
            return None
        predicted_position, predicted_velocity = propagate_paper_model(
            self._last_trusted_position_m,
            self._last_trusted_velocity_m_s,
            max(0.0, gap_s),
            drag_coefficient_per_m=self.config.drag_coefficient_per_m,
            gravity_m_s2=self.config.gravity_m_s2,
            integration_step_s=self.config.integration_step_s,
        )
        innovation = np.asarray(
            position_m, dtype=np.float64).reshape(3) - predicted_position
        if np.linalg.norm(innovation) > self._effective_reacquisition_gate_m():
            # Ordinary misses do not have the detector's two-frame confirmed
            # reacquisition contract.  Never bridge from a stale 6D state when
            # the new visible position is incompatible with it: the most
            # common cause is an unreported impact, for which propagating the
            # old downward velocity is actively worse than declaring velocity
            # warm-up incomplete for a few 90 Hz frames.
            self._last_velocity_warmup_rejection_error_m = innovation.copy()
            self._velocity_warmup_bridge_rejected_count += 1
            return None
        self._last_velocity_warmup_rejection_error_m = None
        self._active = True
        self._initial_time_s = float(self._last_trusted_time_s)
        self._initial_position_m = self._last_trusted_position_m.copy()
        self._initial_velocity_m_s = self._last_trusted_velocity_m_s.copy()
        self._fit_rmse_m = None
        self._active_trigger = "velocity_warmup"
        self._start_reacquisition_handover(
            timestamp_s, position_m, mode="velocity",
            forced_position_only=True, gate_confirmed=True)
        self._velocity_warmup_bridge_count += 1
        return predicted_velocity

    def _append_handover_measurement(
        self,
        timestamp_s: float,
        position_m: np.ndarray,
    ) -> bool:
        if self._handover_history:
            dt = float(timestamp_s - self._handover_history[-1][0])
            if dt <= _MEASUREMENT_TIMESTAMP_DUPLICATE_TOLERANCE_S:
                return False
            if dt > self.config.max_history_gap_s + 1.0e-9:
                self._handover_history.clear()
                self._handover_ready_time_s = None
                self._handover_last_ready_time_s = None
                self._handover_target_velocity_m_s = None
                self._last_handover_alpha = 0.0
        self._handover_history.append(
            (float(timestamp_s), np.asarray(position_m, dtype=np.float64).copy()))
        return True

    def _step_reacquisition_handover(
        self,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        measurement_velocity_valid: bool,
        measurement_time_s: float,
        query_time_s: float,
    ) -> HighBallOcclusionResult:
        if self._handover_mode == "state":
            return self._step_reacquisition_state_smoothing(
                position_m, velocity_m_s, measurement_velocity_valid,
                measurement_time_s, query_time_s)
        assert self._active
        predicted_position, predicted_velocity = self._predict_at(
            measurement_time_s)
        self._last_reacquisition_error_m = (
            np.asarray(position_m, dtype=np.float64) - predicted_position)
        prediction_elapsed_s = max(
            0.0, float(query_time_s) - float(self._initial_time_s))
        if (not self._handover_gate_confirmed
                and np.linalg.norm(self._last_reacquisition_error_m)
                > self._effective_reacquisition_gate_m()):
            self._reacquisition_rejected_count += 1
            if prediction_elapsed_s > self._effective_max_prediction_s():
                self._expiration_count += 1
                self._clear_active()
                self._history.clear()
                self._last_reason = "reacquisition_after_timeout_rejected"
                return self._result(
                    np.zeros(3), np.zeros(3), False, False, query_time_s,
                    reason=self._last_reason,
                    elapsed_s=prediction_elapsed_s)
            query_position, query_velocity = self._predict_at(query_time_s)
            self._last_reason = "reacquisition_handover_outside_gate"
            return self._result(
                query_position, query_velocity, True, True, query_time_s,
                reason=self._last_reason,
                elapsed_s=prediction_elapsed_s,
                state="predicting",
                position_predicted=True,
                velocity_predicted=True,
            )

        appended = self._append_handover_measurement(
            measurement_time_s, position_m)
        if (appended and bool(measurement_velocity_valid)
                and len(self._handover_history) >= int(
                    self.config.reacquisition_velocity_frames)):
            target_velocity = np.asarray(
                velocity_m_s, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(target_velocity)):
                # The detector's velocity_ready flag is the ownership boundary.
                # Do not replace its robust five-frame result with a duplicate
                # two/three-position fit here: millimetre position noise over
                # 11 ms created the frame-000234 lateral velocity spike.
                self._handover_target_velocity_m_s = target_velocity.copy()
                if self._handover_ready_time_s is None:
                    self._handover_ready_time_s = float(measurement_time_s)
                self._handover_last_ready_time_s = float(measurement_time_s)

        if (self._handover_ready_time_s is None
                or self._handover_target_velocity_m_s is None):
            self._last_handover_alpha = 0.0
            self._last_reason = "reacquisition_velocity_collecting"
            return self._result(
                position_m, predicted_velocity, True, True,
                measurement_time_s,
                reason=self._last_reason,
                elapsed_s=prediction_elapsed_s,
                state="reacquisition_handover",
                position_predicted=False,
                velocity_predicted=True,
            )

        assert self._handover_last_ready_time_s is not None
        linear_alpha = np.clip(
            (self._handover_last_ready_time_s - self._handover_ready_time_s)
            / float(self.config.reacquisition_velocity_blend_s),
            0.0,
            1.0,
        )
        # Smoothstep keeps acceleration finite at both ends of the handover.
        alpha = float(linear_alpha * linear_alpha * (3.0 - 2.0 * linear_alpha))
        self._last_handover_alpha = alpha
        target_velocity = self._handover_target_velocity_m_s.copy()
        blended_velocity = (
            (1.0 - alpha) * predicted_velocity + alpha * target_velocity)
        if linear_alpha < 1.0:
            self._last_reason = "reacquisition_velocity_blending"
            return self._result(
                position_m, blended_velocity, True, True,
                measurement_time_s,
                reason=self._last_reason,
                elapsed_s=prediction_elapsed_s,
                state="reacquisition_handover",
                position_predicted=False,
                velocity_predicted=True,
            )

        fresh_samples = list(self._handover_history)
        self._handover_completion_count += 1
        self._clear_active()
        self._history.clear()
        for timestamp_s, sample_position in fresh_samples:
            self._append_measurement(timestamp_s, sample_position)
        self._remember_trusted_state(
            measurement_time_s, position_m, target_velocity)
        self._last_handover_alpha = 1.0
        self._last_reason = "reacquisition_velocity_handover_complete"
        return self._result(
            position_m, target_velocity, True, False, measurement_time_s,
            reason=self._last_reason,
            elapsed_s=prediction_elapsed_s,
            state="tracking",
            position_predicted=False,
            velocity_predicted=False,
        )

    def _step_reacquisition_state_smoothing(
        self,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        measurement_velocity_valid: bool,
        measurement_time_s: float,
        query_time_s: float,
    ) -> HighBallOcclusionResult:
        """Confirm and blend the full 6D state after camera reacquisition."""
        assert self._active
        predicted_position, predicted_velocity = self._predict_at(
            measurement_time_s)
        error = np.asarray(position_m, dtype=np.float64) - predicted_position
        self._last_reacquisition_error_m = error.copy()
        prediction_elapsed_s = max(
            0.0, float(query_time_s) - float(self._initial_time_s))
        if (not self._handover_gate_confirmed
                and np.linalg.norm(error)
                > self._effective_reacquisition_gate_m()):
            self._reacquisition_rejected_count += 1
            if prediction_elapsed_s > self._effective_max_prediction_s():
                self._expiration_count += 1
                self._clear_active()
                self._history.clear()
                self._last_reason = "reacquisition_after_timeout_rejected"
                return self._result(
                    np.zeros(3), np.zeros(3), False, False, query_time_s,
                    reason=self._last_reason,
                    elapsed_s=prediction_elapsed_s,
                )
            query_position, query_velocity = self._predict_at(query_time_s)
            self._last_reason = "reacquisition_smoothing_outside_gate"
            return self._result(
                query_position, query_velocity, True, True, query_time_s,
                reason=self._last_reason,
                elapsed_s=prediction_elapsed_s,
                state="predicting",
                position_predicted=True,
                velocity_predicted=True,
            )

        appended = self._append_handover_measurement(
            measurement_time_s, position_m)
        required = int(self.config.reacquisition_smoothing_frames)
        if (appended and bool(measurement_velocity_valid)
                and len(self._handover_history) >= required):
            target_velocity = np.asarray(
                velocity_m_s, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(target_velocity)):
                self._handover_target_velocity_m_s = target_velocity.copy()
                if self._handover_ready_time_s is None:
                    self._handover_ready_time_s = float(measurement_time_s)
                self._handover_last_ready_time_s = float(measurement_time_s)

        if (len(self._handover_history) < required
                or self._handover_start_time_s is None):
            self._last_handover_alpha = 0.0
            self._last_reason = "reacquisition_smoothing_confirming"
            return self._result(
                predicted_position, predicted_velocity, True, True,
                measurement_time_s,
                reason=self._last_reason,
                elapsed_s=prediction_elapsed_s,
                state="reacquisition_smoothing",
                position_predicted=True,
                velocity_predicted=True,
            )

        position_linear_alpha = float(np.clip(
            (float(measurement_time_s) - self._handover_start_time_s)
            / float(self.config.reacquisition_smoothing_blend_s),
            0.0,
            1.0,
        ))
        position_alpha = float(
            position_linear_alpha * position_linear_alpha
            * (3.0 - 2.0 * position_linear_alpha))
        blended_position = (
            (1.0 - position_alpha) * predicted_position
            + position_alpha * np.asarray(position_m, dtype=np.float64))

        # Position recovery and velocity ownership are separate contracts.
        # Position may finish smoothing while the detector is still filling
        # its velocity window; keep model velocity until velocity_ready, then
        # advance the blend only on subsequent fresh velocity-ready frames.
        velocity_linear_alpha = 0.0
        if (self._handover_ready_time_s is not None
                and self._handover_last_ready_time_s is not None
                and self._handover_target_velocity_m_s is not None):
            velocity_linear_alpha = float(np.clip(
                (self._handover_last_ready_time_s
                 - self._handover_ready_time_s)
                / float(self.config.reacquisition_velocity_blend_s),
                0.0,
                1.0,
            ))
        velocity_alpha = float(
            velocity_linear_alpha * velocity_linear_alpha
            * (3.0 - 2.0 * velocity_linear_alpha))
        self._last_handover_alpha = velocity_alpha
        target_velocity = (
            predicted_velocity
            if self._handover_target_velocity_m_s is None
            else self._handover_target_velocity_m_s)
        blended_velocity = (
            (1.0 - velocity_alpha) * predicted_velocity
            + velocity_alpha * target_velocity)
        if (position_linear_alpha < 1.0
                or velocity_linear_alpha < 1.0):
            self._last_reason = (
                "reacquisition_velocity_waiting"
                if self._handover_target_velocity_m_s is None
                else "reacquisition_state_blending")
            return self._result(
                blended_position, blended_velocity, True, True,
                measurement_time_s,
                reason=self._last_reason,
                elapsed_s=prediction_elapsed_s,
                state="reacquisition_smoothing",
                position_predicted=bool(position_linear_alpha < 1.0),
                velocity_predicted=True,
            )

        fresh_samples = list(self._handover_history)
        self._handover_completion_count += 1
        self._clear_active()
        self._history.clear()
        for timestamp_s, sample_position in fresh_samples:
            self._append_measurement(timestamp_s, sample_position)
        self._remember_trusted_state(
            measurement_time_s, position_m, target_velocity)
        self._last_handover_alpha = 1.0
        self._last_reason = "reacquisition_state_smoothing_complete"
        return self._result(
            position_m, target_velocity, True, False, measurement_time_s,
            reason=self._last_reason,
            elapsed_s=prediction_elapsed_s,
            state="tracking",
            position_predicted=False,
            velocity_predicted=False,
        )

    def _clear_active(self) -> None:
        self._active = False
        self._initial_time_s = None
        self._initial_position_m = None
        self._initial_velocity_m_s = None
        self._fit_rmse_m = None
        self._active_trigger = None
        self._handover_active = False
        self._handover_mode = "velocity"
        self._handover_history.clear()
        self._handover_ready_time_s = None
        self._handover_last_ready_time_s = None
        self._handover_start_time_s = None
        self._handover_target_velocity_m_s = None
        self._handover_forced_by_position_only = False
        self._handover_gate_confirmed = False

    def _history_span_s(self) -> float | None:
        if len(self._history) < 2:
            return None
        return max(0.0, float(self._history[-1][0] - self._history[0][0]))

    def _result(
        self,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        valid: bool,
        predicted: bool,
        timestamp_s: float,
        *,
        reason: str,
        elapsed_s: float | None,
        state: str | None = None,
        position_predicted: bool | None = None,
        velocity_predicted: bool | None = None,
    ) -> HighBallOcclusionResult:
        diagnostics = self.diagnostics()
        diagnostics.update({
            "state": state or (
                "predicting" if predicted
                else ("tracking" if valid else "idle")),
            "reason": reason,
            "predicted": bool(predicted),
            "positionPredicted": bool(
                predicted if position_predicted is None
                else position_predicted),
            "velocityPredicted": bool(
                predicted if velocity_predicted is None
                else velocity_predicted),
            "valid": bool(valid),
            "elapsedS": (
                None if elapsed_s is None else round(float(elapsed_s), 6)),
            "positionM": (
                self._vector_or_none(position_m) if valid else None),
            "velocityMps": (
                self._vector_or_none(velocity_m_s) if valid else None),
        })
        return HighBallOcclusionResult(
            position_m=np.asarray(position_m, dtype=np.float32).reshape(3),
            velocity_m_s=np.asarray(velocity_m_s, dtype=np.float32).reshape(3),
            valid=bool(valid),
            predicted=bool(predicted),
            timestamp_s=float(timestamp_s),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _measurement_time(timestamp_s: float | None, fallback_s: float) -> float:
        if timestamp_s is None:
            return float(fallback_s)
        value = float(timestamp_s)
        return value if np.isfinite(value) else float(fallback_s)

    @staticmethod
    def _vector_or_none(vector: np.ndarray | None) -> list[float] | None:
        if vector is None:
            return None
        array = np.asarray(vector, dtype=float).reshape(3)
        return [round(float(value), 6) for value in array]
