"""V325 u416 NumPy runtime, paired with the authenticated 20261124_r1 source.

observed_force functions below are unchanged from observed_force.py.
No training code, JAX, teacher or learned wind estimator executes here.
"""
import math
from types import SimpleNamespace
import numpy as np
from pingpong_controller.fplc_wind_numpy import WindStartupAuthority



def select_consistent_wind(winds, valid, ages_s, *, xp):
    """Use a consistent recent pair, otherwise a robust valid medoid."""
    count = xp.sum(valid, axis=-1)
    ordered = xp.sort(xp.where(valid[..., None], winds, xp.inf), axis=-2)
    lo = xp.maximum((count-1)//2, 0)
    hi = xp.maximum(count//2, 0)
    median = .5*(xp.take_along_axis(ordered, lo[..., None, None], axis=-2)[..., 0, :]
                  + xp.take_along_axis(ordered, hi[..., None, None], axis=-2)[..., 0, :])
    distance = xp.sum((winds-median[..., None, :])**2, axis=-1)
    robust_index = xp.argmin(xp.where(valid, distance, xp.inf), axis=-1)
    indices = xp.broadcast_to(xp.arange(6), valid.shape)
    first = xp.min(xp.where(valid, indices, 6), axis=-1)
    second = xp.min(xp.where(valid & (indices > first[..., None]), indices, 6), axis=-1)
    first, second = xp.minimum(first, 5), xp.minimum(second, 5)
    newest = xp.take_along_axis(winds, first[..., None, None], axis=-2)[..., 0, :]
    previous = xp.take_along_axis(winds, second[..., None, None], axis=-2)[..., 0, :]
    speed0 = xp.sqrt(xp.sum(newest**2, axis=-1))
    speed1 = xp.sqrt(xp.sum(previous**2, axis=-1))
    aligned = xp.sum(newest*previous, axis=-1) > .9*speed0*speed1
    fast = (count >= 2) & (speed0 > .02) & (speed1 > .02) & aligned
    selected = xp.where(fast, first, robust_index)
    chosen = xp.take_along_axis(winds, selected[..., None, None], axis=-2)[..., 0, :]
    age = xp.take_along_axis(ages_s, selected[..., None], axis=-1)[..., 0]
    enough = count >= 2
    age = xp.where(xp.sum(chosen**2, axis=-1) < .05**2,
                   xp.take_along_axis(ages_s, first[..., None], axis=-1)[..., 0], age)
    chosen = xp.where((enough & (xp.sum(chosen**2, axis=-1) >= .05**2))[..., None], chosen, 0.)
    return chosen, xp.where(enough, age, .6), enough


def observed_force_features(observations, *, xp):
    """Return (...,4): bounded wind XY/1.6, estimate age/0.6 s, validity."""
    if observations.shape[-1] < 112:
        raise ValueError('Observed-force features require the registered history prefix')
    positions = observations[..., 72:96].reshape(observations.shape[:-1] + (8, 3))
    ages = .6 * observations[..., 96:104]
    finite = xp.all(xp.isfinite(positions) & (xp.abs(positions) < 5.), axis=-1)
    valid = (observations[..., 104:112] > .5) & finite & xp.isfinite(ages)
    positions = xp.where(finite[..., None], positions, 0.)
    ages = xp.where(xp.isfinite(ages), ages, 0.)
    candidates, candidate_ages, candidate_valid = [], [], []
    coefficient_m_inv = .5 * 1.2 * .47 * math.pi * .02**2 / .004
    for index in range(6):
        dt01 = ages[..., index+1] - ages[..., index]
        dt12 = ages[..., index+2] - ages[..., index+1]
        safe01, safe12 = xp.maximum(dt01, 1.e-6), xp.maximum(dt12, 1.e-6)
        v01 = (positions[..., index, :] - positions[..., index+1, :]) / safe01[..., None]
        v12 = (positions[..., index+1, :] - positions[..., index+2, :]) / safe12[..., None]
        acceleration = 2. * (v01-v12) / (safe01+safe12)[..., None]
        velocity = v01 - .5 * safe12[..., None] * acceleration
        usable = (xp.all(valid[..., index:index+3], axis=-1)
                  & (dt01 > .02) & (dt12 > .02)
                  & (acceleration[..., 2] > -11.5) & (acceleration[..., 2] < -8.5))
        # b = |w-v| (w_xy-v_xy); solve its scalar magnitude, then its direction.
        speed = xp.sqrt(xp.sum(velocity**2, axis=-1))
        b = acceleration[..., :2] / coefficient_m_inv - speed[..., None] * velocity[..., :2]
        b_sq = xp.sum(b**2, axis=-1)
        vz_sq = velocity[..., 2]**2
        relative_speed = xp.sqrt(xp.maximum(
            .5*(vz_sq + xp.sqrt(vz_sq**2 + 4.*b_sq)), 1.e-12))
        wind = velocity[..., :2] + b / relative_speed[..., None]
        wind = xp.stack([wind[..., 0], xp.minimum(wind[..., 1], 0.)], axis=-1)
        norm = xp.sqrt(xp.sum(wind**2, axis=-1))
        wind = wind * xp.minimum(1., 1.6/xp.maximum(norm, 1.e-6))[..., None]
        candidates.append(wind)
        candidate_ages.append(xp.clip(.5*(ages[..., index]+ages[..., index+2]), 0., .6))
        candidate_valid.append(usable)
    winds = xp.stack(candidates, axis=-2)
    usable = xp.stack(candidate_valid, axis=-1)
    estimate, estimate_age, chosen = select_consistent_wind(
        winds, usable, xp.stack(candidate_ages, axis=-1), xp=xp)
    mature = xp.any(valid & (ages >= .25), axis=-1)
    chosen = chosen & mature
    estimate = xp.where(chosen[..., None], estimate/1.6, 0.)
    estimate_age = xp.where(chosen, estimate_age/.6, 1.)
    return xp.concatenate([estimate, estimate_age[..., None],
                           chosen.astype(observations.dtype)[..., None]], axis=-1)


def acquisition_features(state, *, xp):
    """single5_runtime.acquisition_features: ordinary inputs, no action gate."""
    elapsed = state.step_count * .005 - state.fifth_hit_s
    acquired = (state.wind_startup_hit_count >= 5) & (state.fifth_hit_s >= 0.)
    phase = xp.clip(elapsed / 3., 0., 1.)
    progress = xp.where(acquired, xp.sin(.5 * math.pi * phase) ** 2, 0.)
    return xp.stack([progress, xp.clip(72. * progress, 0., 1.)], axis=-1)


def encode_high_wind(obs, *, xp):
    """wind_features.encode, preserving the first 118 values exactly."""
    if obs.shape[-1] != 118:
        raise ValueError('Expected the registered 118-D raw actor observation')
    estimated = obs[..., 114:116]
    speed_m_s = 1.6 * xp.sqrt(xp.sum(estimated * estimated, axis=-1))
    gain = xp.clip((speed_m_s - 1.25) / .35, 0., 1.)
    gain = xp.where(obs[..., 112] > 0., gain, 0.)
    return xp.concatenate([obs, estimated * gain[..., None]], axis=-1)


def encode_single5(prefix112, phase):
    prefix112 = np.asarray(prefix112, dtype=np.float32)
    # Invalid candidate medians can contain inf internally; source masks them.
    with np.errstate(invalid='ignore', over='ignore'):
        estimate = observed_force_features(prefix112, xp=np)
    raw = np.concatenate([prefix112, np.asarray(phase, dtype=np.float32), estimate], axis=-1)
    return encode_high_wind(raw, xp=np).astype(np.float32)


def single5_policy_mean(weights, obs):
    """Same three-hidden-layer MLP; split l1 to retain zero-input identity."""
    obs = np.asarray(obs, dtype=np.float32)
    old_affine = obs[..., :118] @ weights['l1_w'][:118] + weights['l1_b']
    extra = obs[..., 118:] @ weights['l1_w'][118:]
    active = np.any(obs[..., 118:] != 0., axis=-1, keepdims=True)
    x = np.tanh(np.where(active, old_affine + extra, old_affine))
    for layer in ('l2', 'l3'):
        x = np.tanh(x @ weights[layer + '_w'] + weights[layer + '_b'])
    return x @ weights['out_w'] + weights['out_b']


class FixedWindAcquisition(WindStartupAuthority):
    """Share measured launch confirmation; attitude is always five degrees.

    The actor clock starts at the estimated fifth qualified launch. Hardware
    lacks the simulator's contact flag, so this remains a recorded estimate.
    No random training wind-onset delay is sampled or fed to the robot actor.
    """
    def _validate_cfg(self, cfg):
        if (cfg['wind_config']['startup_no_wind_hits'] != 5
                or cfg['wind_pose']['start_deg'] != 5.
                or cfg['wind_pose']['target_deg'] != 5.
                or cfg['wind_pose']['ramp_updates'] != 0):
            raise RuntimeError('u416 requires fixed five-degree pose and five-hit acquisition')

    def limits_deg(self, tick):
        return np.full(2, 5., dtype=np.float32)

    def features(self, tick):
        state = SimpleNamespace(step_count=np.asarray([tick], dtype=np.float32),
            fifth_hit_s=np.asarray([-1. if self.started_tick is None else self.started_tick * self.dt], dtype=np.float32),
            wind_startup_hit_count=np.asarray([self.confirmed_hits], dtype=np.int32))
        return acquisition_features(state, xp=np)[0].astype(np.float32)

    def diagnostics(self, tick):
        return {'confirmedHits': self.confirmed_hits, 'fifthHitTick': self.started_tick,
                'pendingLaunch': self.pending_tick is not None,
                'relativeRollPitchLimitDeg': [5., 5.],
                'acquisitionFeatures': self.features(tick).tolist(),
                'phaseDurationS': 3., 'fixedPose': True,
                'confirmationMethod': 'direct_sensor_near_racket_clearance_apex_estimate'}
