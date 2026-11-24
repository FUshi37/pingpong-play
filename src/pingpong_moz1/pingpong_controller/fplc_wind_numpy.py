"""NumPy deployment port of fixed_plane_wind_history / fixed_plane_wind_actor.

Only measured camera/mocap positions and their sample ages enter the history.
Simulator wind truth never enters the robot actor. The u416 adapter separately
derives its estimated-wind inputs from this measured history.
"""
from __future__ import annotations

import numpy as np

WIND_HISTORY_LAGS = (0, 10, 20, 30, 40, 60, 80, 100)
WIND_HISTORY_SIZE = 101


class WindStartupAuthority:
    """Five qualified launches, then the frozen sin² 1 s authority ramp.

    Hardware has no MuJoCo contact flag. A directly observed near-racket
    approach arms a pending launch; the native clearance, predicted apex,
    confirmation deadline and minimum count interval qualify it. This is an
    explicit sensor estimate, not the camera's unqualified impact counter.
    """

    def __init__(self, cfg):
        self._validate_cfg(cfg)
        self.cfg = cfg
        self.confirmed_hits = 0
        self.started_tick = None
        self.last_hit_tick = None
        self.pending_tick = None
        self.dt = .005
        self.gravity_m_s2 = max(abs(float(np.mean(cfg['dr_gravity_z_range']))), 1e-6)

    def _validate_cfg(self, cfg):
        if (cfg['wind_config']['startup_no_wind_hits'] != 5
                or cfg['wind_config']['startup_ramp_s'] != 1.
                or cfg['wind_pose']['start_deg'] != 20.
                or cfg['wind_pose']['target_deg'] != 20.
                or cfg['wind_pose']['ramp_updates'] != 0):
            raise RuntimeError('V325 startup authority requires the frozen five-hit/1s/full20 contract')

    def observe(self, tick, position, velocity, racket_position, *, observed):
        if not observed:
            # Do not join a pending contact across missing measurements.
            self.pending_tick = None
            return
        cfg = self.cfg
        first = self.confirmed_hits == 0
        clearance = max(float(cfg['hit_confirm_rel_height']), .04)
        deadline = int(cfg['hit_confirm_max_steps'])
        if first:
            clearance = float(cfg['fixed_plane_v107_first_confirm_rel_height_m']) or clearance
            deadline = int(cfg['fixed_plane_v107_first_confirm_max_steps']) or deadline
        relative = np.asarray(position) - np.asarray(racket_position)
        if self.pending_tick is not None and tick - self.pending_tick >= deadline:
            self.pending_tick = None
        # XY proximity follows the existing measured recording counter.
        near = (np.linalg.norm(relative[:2]) <= .095
                and -.035 <= relative[2] <= float(cfg['hit_rearm_distance']))
        if self.pending_tick is None and near and (first or velocity[2] <= 0.):
            self.pending_tick = tick
        if self.pending_tick is None:
            return
        schedule = cfg['fixed_plane_limit_cycle_apex_schedule_z_m']
        desired_apex = schedule[min(self.confirmed_hits, len(schedule) - 1)]
        quality_floor = max(
            desired_apex - float(cfg['fixed_plane_limit_cycle_hit_quality_apex_margin_m']),
            float(cfg['fixed_contact_surface_z_m']) + .020 + .030)
        predicted_apex = (float(position[2]) + max(float(velocity[2]), 0.) ** 2
                          / (2. * self.gravity_m_s2))
        interval_ok = (self.last_hit_tick is None or (tick - self.last_hit_tick) * self.dt
                       >= float(cfg['hit_min_count_interval']))
        if relative[2] >= clearance and velocity[2] > 0. and predicted_apex >= quality_floor:
            self.pending_tick = None
            self.last_hit_tick = tick
            if interval_ok:
                self.confirmed_hits += 1
                if self.confirmed_hits >= 5 and self.started_tick is None:
                    self.started_tick = tick

    def limits_deg(self, tick):
        # fixed_plane_wind_pose.effective_degrees uses the startup shelter
        # even when the saved curriculum start_deg and target_deg are both20.
        fraction = (0. if self.started_tick is None else
                    np.clip((tick - self.started_tick) * self.dt, 0., 1.))
        gain = np.sin(.5 * np.pi * fraction) ** 2
        target = float(self.cfg['wind_pose']['target_deg'])
        return np.full(2, 2. + gain * (target - 2.), np.float32)

    def diagnostics(self, tick):
        bounds = self.limits_deg(tick)
        return {'confirmedHits': self.confirmed_hits,
                'rampStartedTick': self.started_tick,
                'pendingLaunch': self.pending_tick is not None,
                'relativeRollPitchLimitDeg': bounds.tolist(),
                'residualGate': float(np.clip((min(bounds) - 2.) / .25, 0., 1.)),
                'confirmationMethod': 'direct_sensor_near_racket_clearance_apex_estimate'}


class WindBallHistory:
    def __init__(self):
        self.positions = np.zeros((WIND_HISTORY_SIZE, 3), dtype=np.float32)
        self.sample_times = np.zeros(WIND_HISTORY_SIZE, dtype=np.float64)
        self.valid = np.zeros(WIND_HISTORY_SIZE, dtype=bool)
        self.tick = 0

    def advance(self, position, sample_age_s, observed, bounds_deg):
        position = np.asarray(position, dtype=np.float32)
        valid = bool(observed and np.isfinite(position).all())
        now = self.tick * .005
        self.positions[1:] = self.positions[:-1].copy()
        self.sample_times[1:] = self.sample_times[:-1].copy()
        self.valid[1:] = self.valid[:-1].copy()
        self.positions[0] = position if valid else 0.
        self.sample_times[0] = now - sample_age_s
        self.valid[0] = valid
        self.tick += 1
        return self.suffix(now, bounds_deg)

    def suffix(self, now_s, bounds_deg):
        indices = np.asarray(WIND_HISTORY_LAGS)
        positions = self.positions[indices]
        # Match the training float32 age comparison at the 0.6 s boundary.
        ages = np.maximum(now_s - self.sample_times[indices], 0.).astype(np.float32)
        valid = self.valid[indices] & (ages <= .6)
        delta_valid = valid & valid[0]
        deltas = np.where(delta_valid[:, None], positions - positions[:1], 0.)
        normalized_age = np.where(valid, np.clip(ages / .6, 0., 1.), 0.)
        return np.concatenate((deltas.reshape(24), normalized_age, valid,
                               np.asarray(bounds_deg) / 20.)).astype(np.float32)


def wind_policy_mean(observation, teacher_mean, residual_mean, residual_scale, gate_span_deg):
    """Exact deterministic teacher + authority-gated bounded residual mean."""
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (114,):
        raise ValueError('Wind authority gate requires the registered 114-D observation')
    authority_deg = np.min(observation[-2:]) * np.float32(20.)
    gate = np.clip((authority_deg - np.float32(2.))
                   / np.maximum(gate_span_deg, np.float32(1e-6)), 0., 1.)
    baseline = teacher_mean + gate * (np.clip(teacher_mean, -1., 1.) - teacher_mean)
    correction = gate * residual_scale * np.tanh(residual_mean)
    return np.asarray(baseline + correction, dtype=np.float32)
