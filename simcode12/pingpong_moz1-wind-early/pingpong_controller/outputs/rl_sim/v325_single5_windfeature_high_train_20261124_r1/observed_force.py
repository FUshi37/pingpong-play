"""Causal four-value actor feature from the existing observed-ball history.

Inputs use only actor columns 72:112: eight observed position differences in
metres, their actual camera-sample ages, and validity. The estimator inverts
the registered incremental quadratic-drag equation with the unchanged 4 g
simulation ball. It never reads simulator wind, mass samples or true velocity.
"""
import math


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
