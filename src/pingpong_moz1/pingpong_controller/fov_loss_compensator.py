#!/home/jacky/miniconda3/envs/gmr/bin/python
"""FOV-loss compensator for ping-pong ball juggling.

When the ball flies above the camera FOV during the upward phase of a
juggling cycle, ``PingPongBallDetector.get_observation()`` returns
``detected=False`` and the controller falls back to its hold-last
behavior. That is fine for sub-100 ms dropouts but starves the policy
on a typical apex-and-back of 300-500 ms.

This module sits between the detector and the controller's
``BallState`` assembly. While vision keeps producing valid detections it
is a passthrough; once detection drops, it propagates the last good
state forward with a closed-form ballistic model (gravity + optional
linear drag) and emits a synthesized observation tagged
``predicted=True``. When detection comes back, a Mahalanobis-style gate
on the position residual decides whether the new measurement is the
same ball before re-locking onto it.

The compensator is opt-in: build one per node lifetime and call
``update(observation, observation_time)`` once per detector frame. With
the controller switch off, the module is never imported into the
detection path.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


# --------------------------------------------------------------------------
# State labels (exposed as plain strings so callers can compare without
# importing this module if they do not need to).
# --------------------------------------------------------------------------
STATE_TRACKING = 'tracking'
STATE_PREDICTING = 'predicting'
STATE_REACQUIRING = 'reacquiring'
STATE_LOST = 'lost'


# Gravity magnitude in mm/s^2 (positive scalar).
_G_MM_S2 = 9810.0


def gravity_for_base_link() -> Tuple[float, float, float]:
    """Gravity vector when (position, velocity) live in the robot base_link
    frame with Z pointing up.
    """
    return (0.0, 0.0, -_G_MM_S2)


def gravity_for_camera_frame(camera_tilt_deg: float) -> Tuple[float, float, float]:
    """Gravity vector in the camera optical frame (X right, Y down, Z forward).

    Matches the convention used by ``ball_detector.KalmanFilter._compute_gravity``:
    a positive ``camera_tilt_deg`` means the camera is tilted to look downward.
    """
    rad = float(np.deg2rad(camera_tilt_deg))
    return (0.0, _G_MM_S2 * float(np.cos(rad)), _G_MM_S2 * float(np.sin(rad)))


@dataclass
class FovLossCompensatorConfig:
    """Tunable parameters for ``FovLossCompensator``.

    All distances in mm, velocities in mm/s, times in seconds.
    """

    # Gravity vector in the same frame as the (position, velocity) being
    # passed to ``update()``. Default assumes base_link with Z up.
    gravity_mm_s2: Tuple[float, float, float] = (0.0, 0.0, -_G_MM_S2)

    # State-machine timing.
    miss_frames_to_predict: int = 2
    max_predict_duration_s: float = 0.8
    reacquire_consistent_frames: int = 1

    # Only enter PREDICTING if the ball was clearly going upward at the
    # moment of FOV loss (i.e. velocity component anti-parallel to gravity
    # is at least ``min_upward_velocity_mm_s``). Avoids fabricating
    # trajectories from random frame drops while the ball is on the way
    # back down where it should remain inside the FOV.
    require_upward_to_predict: bool = True
    min_upward_velocity_mm_s: float = 200.0

    # History window used to seed the exit state at TRACKING -> PREDICTING.
    history_window_s: float = 0.15
    min_history_samples: int = 3

    # Gating during REACQUIRING: a Mahalanobis-style scalar test on the 3D
    # position residual, with sigma growing in time so brief losses use a
    # tight gate and long predicts use a loose one.
    gate_sigma_floor_mm: float = 40.0
    gate_sigma_per_s_mm: float = 250.0
    gate_max_mahalanobis: float = 4.0

    # Optional linear-drag coefficient (k = 1/tau, units 1/s). 0 disables
    # drag and falls back to the closed-form ballistic model.
    drag_coeff_per_s: float = 0.0


def _propagate(position_mm: np.ndarray,
               velocity_mm_s: np.ndarray,
               gravity_mm_s2: np.ndarray,
               dt: float,
               drag_coeff_per_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form ballistic propagation, with optional linear drag.

    With drag coefficient ``k`` (1/s), the model integrates
    ``a = g - k * v`` exactly so even longish predicts stay numerically
    clean. With ``k == 0`` it collapses to ``p + v dt + 0.5 g dt^2``.
    """
    dt = max(float(dt), 0.0)
    if dt <= 0.0:
        return position_mm.copy(), velocity_mm_s.copy()

    if drag_coeff_per_s <= 1e-6:
        new_pos = position_mm + velocity_mm_s * dt + 0.5 * gravity_mm_s2 * (dt ** 2)
        new_vel = velocity_mm_s + gravity_mm_s2 * dt
        return new_pos, new_vel

    k = float(drag_coeff_per_s)
    # Terminal velocity component: v_inf = g / k. Then
    #   v(t) = v_inf + (v0 - v_inf) * exp(-k t)
    #   x(t) = x0 + v_inf * t + (v0 - v_inf) * (1 - exp(-k t)) / k
    v_inf = gravity_mm_s2 / k
    decay = float(np.exp(-k * dt))
    new_vel = v_inf + (velocity_mm_s - v_inf) * decay
    new_pos = position_mm + v_inf * dt + (velocity_mm_s - v_inf) * (1.0 - decay) / k
    return new_pos, new_vel


def _seed_state_from_history(history: deque,
                             current_pos: np.ndarray,
                             current_vel: np.ndarray,
                             window_s: float,
                             min_samples: int
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the exit (position, velocity) from the last detector samples.

    Linear least squares on the most recent ``window_s`` window — more
    robust than the per-frame velocity, which on a 60 Hz feed can be
    noisy precisely at the edge of the FOV where motion blur is worst.
    Falls back to the current detector outputs if not enough history.
    """
    if len(history) < max(2, min_samples):
        return current_pos.copy(), current_vel.copy()

    samples = list(history)
    t_latest = samples[-1][0]
    cutoff = t_latest - max(window_s, 1e-3)
    window = [s for s in samples if s[0] >= cutoff]
    if len(window) < max(2, min_samples):
        window = samples[-max(2, min_samples):]

    times = np.array([s[0] for s in window], dtype=float)
    positions = np.array([s[1] for s in window], dtype=float)

    t0 = times[0]
    t = times - t0

    # Per-axis linear fit p(t) = a + b t. b is the velocity estimate;
    # a + b * (t_latest - t0) is the smoothed exit position.
    A = np.stack([np.ones_like(t), t], axis=1)
    try:
        coeffs, *_ = np.linalg.lstsq(A, positions, rcond=None)
    except np.linalg.LinAlgError:
        return current_pos.copy(), current_vel.copy()

    a = coeffs[0]
    b = coeffs[1]
    exit_pos = a + b * (t_latest - t0)
    exit_vel = b
    if not (np.all(np.isfinite(exit_pos)) and np.all(np.isfinite(exit_vel))):
        return current_pos.copy(), current_vel.copy()
    return exit_pos.astype(float), exit_vel.astype(float)


class FovLossCompensator:
    """State machine that fills in the ball state across short FOV losses.

    Usage (intended to live in the vision callback):

        comp = FovLossCompensator(config)
        ...
        out = comp.update(observation, observation_time=stamp_sec)
        # out is a dict with the same shape as `observation` plus:
        #   'predicted': bool, 'state': str, 'predict_age_s': float
        # When predicted=True, position/velocity are propagated, NOT
        # measured, but valid for downstream as if detected=True.

    The compensator never raises on bad inputs; it logs nothing of its
    own. Callers can read ``last_state``/``last_event`` for diagnostics.
    """

    def __init__(self, config: Optional[FovLossCompensatorConfig] = None):
        self.config = config if config is not None else FovLossCompensatorConfig()
        self._gravity = np.array(self.config.gravity_mm_s2, dtype=float)

        self._state = STATE_TRACKING
        self._history: deque = deque(maxlen=64)  # (t, position_mm)
        self._last_pos: Optional[np.ndarray] = None
        self._last_vel: Optional[np.ndarray] = None
        self._last_time: Optional[float] = None
        self._predict_start_time: Optional[float] = None
        self._predict_pos: Optional[np.ndarray] = None
        self._predict_vel: Optional[np.ndarray] = None
        self._consecutive_misses = 0
        self._reacquire_consistent = 0

        # Counters / diagnostics.
        self.predicts_emitted = 0
        self.reacquire_accepted = 0
        self.reacquire_rejected = 0
        self.last_event: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    def reset(self) -> None:
        """Drop all internal state. The next observation seeds tracking."""
        self._state = STATE_TRACKING
        self._history.clear()
        self._last_pos = None
        self._last_vel = None
        self._last_time = None
        self._predict_start_time = None
        self._predict_pos = None
        self._predict_vel = None
        self._consecutive_misses = 0
        self._reacquire_consistent = 0
        self.last_event = 'reset'

    def update(self, observation: dict,
               observation_time: float) -> dict:
        """Wrap an observation; emit a (possibly synthesized) one back.

        ``observation`` must follow the shape returned by
        ``PingPongBallDetector.get_observation()`` (keys ``detected``,
        ``position``, ``velocity``, optional ``raw_measurement``,
        ``reject_reason``).

        Returns a new dict with the same keys plus:
            ``predicted`` (bool) — True if position/velocity came from the
                                   ballistic propagator, not the detector.
            ``state`` (str)      — current state-machine label.
            ``predict_age_s``    — seconds since the FOV-loss start (0
                                   while tracking).
        """
        t = float(observation_time)
        detected = bool(observation.get('detected', False))

        if detected:
            return self._handle_detection(observation, t)
        return self._handle_miss(observation, t)

    # ------------------------------------------------------------------
    # Internals: detection branch
    # ------------------------------------------------------------------

    def _handle_detection(self, observation: dict, t: float) -> dict:
        pos = np.asarray(observation.get('position'), dtype=float).reshape(-1)
        vel = np.asarray(observation.get('velocity'), dtype=float).reshape(-1)
        if pos.shape != (3,) or vel.shape != (3,) or not (
                np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            # Treat malformed detection as a miss so we don't poison history.
            return self._handle_miss(observation, t)

        if self._state == STATE_PREDICTING or self._state == STATE_REACQUIRING:
            accepted = self._gate_reacquire(pos, t)
            if not accepted:
                # Stay in REACQUIRING; emit the propagated state for one
                # more tick and wait for a consistent measurement.
                self.reacquire_rejected += 1
                self._state = STATE_REACQUIRING
                self.last_event = 'reacquire_rejected'
                return self._emit_predicted(observation, t)

            self._reacquire_consistent += 1
            if self._reacquire_consistent < max(
                    1, int(self.config.reacquire_consistent_frames)):
                # Soft accept: still emit the propagated value for this
                # frame, but record the measurement to confirm next tick.
                self._state = STATE_REACQUIRING
                self.last_event = 'reacquire_pending'
                return self._emit_predicted(observation, t)

            # Hard re-lock.
            self.reacquire_accepted += 1
            self.last_event = 'reacquire_locked'

        # TRACKING (or just-locked): pass the detector outputs straight
        # through and refresh history for the next FOV-loss event.
        self._state = STATE_TRACKING
        self._last_pos = pos.copy()
        self._last_vel = vel.copy()
        self._last_time = t
        self._predict_start_time = None
        self._predict_pos = None
        self._predict_vel = None
        self._consecutive_misses = 0
        self._reacquire_consistent = 0
        self._history.append((t, pos.copy()))
        # Trim history to the configured window.
        cutoff = t - max(self.config.history_window_s * 4.0, 0.5)
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        out = dict(observation)
        out['predicted'] = False
        out['state'] = STATE_TRACKING
        out['predict_age_s'] = 0.0
        return out

    # ------------------------------------------------------------------
    # Internals: miss branch
    # ------------------------------------------------------------------

    def _handle_miss(self, observation: dict, t: float) -> dict:
        self._consecutive_misses += 1
        self._reacquire_consistent = 0

        if self._state == STATE_TRACKING:
            should_predict = (
                self._last_pos is not None
                and self._last_vel is not None
                and self._consecutive_misses
                >= max(1, int(self.config.miss_frames_to_predict))
                and self._is_eligible_for_predict()
            )
            if not should_predict:
                # Not yet eligible: pass detection failure through unchanged.
                return self._passthrough_miss(observation, t)
            self._enter_predicting(t)

        if self._state in (STATE_PREDICTING, STATE_REACQUIRING):
            assert self._predict_start_time is not None
            if (t - self._predict_start_time
                    > self.config.max_predict_duration_s):
                # Give up; controller falls back to its own dropout logic.
                self._state = STATE_LOST
                self.last_event = 'predict_timeout'
                return self._passthrough_miss(observation, t)
            return self._emit_predicted(observation, t)

        # STATE_LOST or anything else: passthrough until the next valid
        # detection rebuilds tracking.
        return self._passthrough_miss(observation, t)

    def _is_eligible_for_predict(self) -> bool:
        if not self.config.require_upward_to_predict:
            return True
        if self._last_vel is None:
            return False
        # Upward = velocity component anti-parallel to gravity.
        gnorm = float(np.linalg.norm(self._gravity))
        if gnorm < 1e-6:
            return True
        upward_dir = -self._gravity / gnorm
        upward_speed = float(np.dot(self._last_vel, upward_dir))
        return upward_speed >= float(self.config.min_upward_velocity_mm_s)

    def _enter_predicting(self, t: float) -> None:
        exit_pos, exit_vel = _seed_state_from_history(
            self._history,
            self._last_pos if self._last_pos is not None else np.zeros(3),
            self._last_vel if self._last_vel is not None else np.zeros(3),
            window_s=self.config.history_window_s,
            min_samples=self.config.min_history_samples,
        )
        # Roll forward from the last detection time to "now" so the first
        # emitted prediction lines up with the current frame timestamp
        # rather than the moment we lost the ball.
        if self._last_time is not None and t > self._last_time:
            exit_pos, exit_vel = _propagate(
                exit_pos, exit_vel, self._gravity,
                t - self._last_time,
                self.config.drag_coeff_per_s)
        self._predict_start_time = (
            self._last_time if self._last_time is not None else t)
        self._predict_pos = exit_pos
        self._predict_vel = exit_vel
        self._state = STATE_PREDICTING
        self.last_event = 'predict_started'

    def _emit_predicted(self, observation: dict, t: float) -> dict:
        if self._predict_pos is None or self._predict_vel is None:
            return self._passthrough_miss(observation, t)

        # Step the propagator forward from its last advance to ``t``.
        anchor_t = (self._last_time
                    if self._last_time is not None
                    else self._predict_start_time)
        if anchor_t is None:
            anchor_t = t
        dt = max(t - anchor_t, 0.0)
        new_pos, new_vel = _propagate(
            self._predict_pos, self._predict_vel, self._gravity,
            dt, self.config.drag_coeff_per_s)
        self._predict_pos = new_pos
        self._predict_vel = new_vel
        self._last_time = t

        self.predicts_emitted += 1
        predict_age = (t - self._predict_start_time
                       if self._predict_start_time is not None else 0.0)

        out = dict(observation)
        out['detected'] = True
        out['position'] = new_pos.tolist()
        out['velocity'] = new_vel.tolist()
        out['predicted'] = True
        out['state'] = self._state
        out['predict_age_s'] = float(predict_age)
        # Tag raw_measurement so downstream logging can tell predictions
        # from real detections without re-checking ``predicted``.
        raw = dict(observation.get('raw_measurement') or {})
        raw['fov_predicted'] = True
        raw['fov_predict_age_s'] = float(predict_age)
        raw['fov_state'] = self._state
        out['raw_measurement'] = raw
        return out

    def _passthrough_miss(self, observation: dict, t: float) -> dict:
        out = dict(observation)
        out['predicted'] = False
        out['state'] = self._state
        if self._predict_start_time is not None and self._state in (
                STATE_PREDICTING, STATE_REACQUIRING, STATE_LOST):
            out['predict_age_s'] = float(t - self._predict_start_time)
        else:
            out['predict_age_s'] = 0.0
        return out

    def _gate_reacquire(self, measurement_pos: np.ndarray, t: float) -> bool:
        if self._predict_pos is None:
            return True
        residual = measurement_pos - self._predict_pos
        norm = float(np.linalg.norm(residual))
        age = (t - self._predict_start_time
               if self._predict_start_time is not None else 0.0)
        sigma = max(
            self.config.gate_sigma_floor_mm,
            self.config.gate_sigma_floor_mm
            + self.config.gate_sigma_per_s_mm * max(age, 0.0))
        if sigma <= 1e-6:
            return True
        mahalanobis = norm / sigma
        return mahalanobis <= float(self.config.gate_max_mahalanobis)

