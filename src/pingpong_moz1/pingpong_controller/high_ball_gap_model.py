"""Causal, bounded prediction of a short missing section near a ball's apex.

Only measured positions enter the fit. This module has no mocap dependency
and never renews its time budget with an extrapolated sample.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HighBallGapConfig:
    history_frames: int = 10
    history_window_s: float = 0.120
    min_samples: int = 5
    min_span_s: float = 0.040
    max_sample_gap_s: float = 0.060
    max_prediction_s: float = 0.200
    min_seed_z_m: float = 1.350
    min_apex_z_m: float = 1.400
    stop_z_m: float = 1.350
    max_fit_error_m: float = 0.004
    max_last_residual_m: float = 0.012
    max_uncertainty_m: float = 0.025
    noise_floor_m: float = 0.0015
    acceleration_uncertainty_m_s2: float = 0.5
    gravity_m_s2: float = 9.8
    max_speed_m_s: float = 5.0

    def __post_init__(self):
        if not 3 <= self.min_samples <= self.history_frames:
            raise ValueError("Require 3 <= min_samples <= history_frames")
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_span_s > self.history_window_s:
            raise ValueError("min_span_s exceeds history_window_s")


class HighBallGapModel:
    def __init__(self, config=None):
        self.config = config or HighBallGapConfig()
        self.history = deque(maxlen=self.config.history_frames)
        self.reason = "insufficient_history"
        self.fit = None

    def reset(self):
        self.history.clear()
        self.fit = None
        self.reason = "insufficient_history"

    def observe(self, timestamp_s, position_m):
        t = float(timestamp_s)
        p = np.asarray(position_m, dtype=float).reshape(3)
        if not np.isfinite(t) or not np.isfinite(p).all():
            return
        if self.history:
            dt = t - self.history[-1][0]
            if dt <= 1e-6:
                return
            if dt > self.config.max_sample_gap_s + 1e-9:
                self.reset()
        self.history.append((t, p.copy()))
        self.fit = None

    def _fit(self):
        cfg = self.config
        if len(self.history) < cfg.min_samples:
            self.reason = "insufficient_history"
            return None
        end = self.history[-1][0]
        samples = [s for s in self.history if end-s[0] <= cfg.history_window_s]
        if len(samples) < cfg.min_samples:
            self.reason = "insufficient_recent_history"
            return None
        times = np.array([s[0]-end for s in samples])
        if np.ptp(times) < cfg.min_span_s:
            self.reason = "history_span_too_short"
            return None
        points = np.array([s[1] for s in samples])
        if np.ptp(points[:, 2]) < .004:
            self.reason = "stationary_or_ambiguous_motion"
            return None
        # Fixed gravity removes the ill-conditioned free quadratic acceleration
        # estimate from a 40-120 ms window. Vector Huber weights reduce the
        # effect of an isolated range perturbation on all three axes together.
        target = points.copy()
        target[:, 2] += .5*cfg.gravity_m_s2*times**2
        design = np.column_stack((np.ones(len(times)), times))
        weights = np.ones(len(times))
        for _ in range(3):
            root = np.sqrt(weights)
            coef = np.linalg.lstsq(design*root[:, None], target*root[:, None], rcond=None)[0]
            residual = np.linalg.norm(target-design@coef, axis=1)
            weights = np.minimum(1., .004/np.maximum(residual, 1e-12))
        rmse = float(np.sqrt(np.mean(residual**2)))
        if rmse > cfg.max_fit_error_m or residual[-1] > cfg.max_last_residual_m:
            self.reason = "trajectory_fit_residual_too_large"
            return None
        pos, vel = coef
        if np.linalg.norm(vel) > cfg.max_speed_m_s:
            self.reason = "fitted_speed_too_large"
            return None
        apex = pos[2] + vel[2]**2/(2*cfg.gravity_m_s2)
        if pos[2] < cfg.min_seed_z_m or apex < cfg.min_apex_z_m:
            self.reason = "trajectory_below_high_band"
            return None
        # A descending near-apex track is also usable, but a falling low ball
        # cannot trigger this model from its projected *past* apex.
        if vel[2] < 0 and pos[2] < cfg.min_apex_z_m:
            self.reason = "descending_below_high_band"
            return None
        return dict(t=end, position=pos.copy(), velocity=vel.copy(), rmse=rmse,
                    count=len(times), mean_time=float(times.mean()),
                    time_energy=float(np.sum((times-times.mean())**2)))

    def predict(self, timestamp_s):
        now = float(timestamp_s)
        if not np.isfinite(now):
            self.reason = "invalid_query_time"
            return None
        if self.fit is None:
            self.fit = self._fit()
        if self.fit is None:
            return None
        return self.predict_fit(self.fit, now)

    def predict_fit(self, fit, timestamp_s):
        """Evaluate a frozen seed; reacquisition cannot extend this budget."""
        now, cfg = float(timestamp_s), self.config
        if not np.isfinite(now):
            self.reason = "invalid_query_time"
            return None
        dt = now-fit['t']
        if dt < 0 or dt > cfg.max_prediction_s+1e-9:
            self.reason = "prediction_timeout"
            return None
        uncertainty = (2*max(cfg.noise_floor_m, fit['rmse'])*np.sqrt(
            1/fit['count']+(dt-fit['mean_time'])**2/fit['time_energy'])
            + .5*cfg.acceleration_uncertainty_m_s2*dt**2)
        if uncertainty > cfg.max_uncertainty_m:
            self.reason = "prediction_uncertainty_limit"
            return None
        p = fit['position'] + dt*fit['velocity']
        p[2] -= .5*cfg.gravity_m_s2*dt**2
        v = fit['velocity'].copy()
        v[2] -= cfg.gravity_m_s2*dt
        if p[2] < cfg.stop_z_m:
            self.reason = "prediction_below_stop_z"
            return None
        self.reason = "predicting_high_ball_gap"
        return p, v, dict(elapsed_s=dt, uncertainty_m=float(uncertainty),
                         fit_rmse_m=fit['rmse'], seed_time_s=fit['t'])
