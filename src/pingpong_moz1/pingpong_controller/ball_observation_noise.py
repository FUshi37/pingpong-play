"""Fixed-size observation noise and output bias calibration.

Configuration is read once at camera-producer construction. No truth,
recordings, history fitting, file I/O or parameter adaptation runs per frame.
"""
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np


class ObservationNoiseProfile:
    VERSION = 'camera-observation-v1.5.0-rc1-20261020'

    def __init__(self, settings=None):
        value = dict(settings or {})
        for key in ('enabled', 'filter_enabled', 'guard_contact'):
            if key in value and not isinstance(value[key], bool):
                raise ValueError('Observation '+key+' must be boolean')
        self.enabled = bool(value.get('enabled', False))
        self.name = str(value.get('name', 'legacy'))
        self.filter_enabled = bool(self.enabled and value.get('filter_enabled', True))
        self.guard_contact = bool(value.get('guard_contact', False))
        self.bias = np.array(value.get('output_bias_m', [0., 0., 0.]), float, copy=True)
        self.reference = np.array(value.get('measurement_covariance_m2',
            np.diag([.0015**2, .0015**2, .003**2])), float)
        self.process = np.array(value.get('process_noise_diagonal_m2_s3',
            [.10**2, .10**2, .15**2]), float)
        self.quality_reference = float(value.get('quality_reference_sigma_m', .003))
        self.robust_nu = float(value.get('robust_nu', 0.))
        self.maximum_inflation = float(value.get('maximum_robust_inflation', 4.))
        if (self.bias.shape != (3,) or not np.isfinite(self.bias).all()
                or np.max(np.abs(self.bias)) > .005):
            raise ValueError('Observation output bias must be a finite XYZ vector within 5 mm')
        if (self.reference.shape != (3, 3) or not np.isfinite(self.reference).all()
                or not np.allclose(self.reference, self.reference.T, atol=1e-15)
                or np.min(np.linalg.eigvalsh(self.reference)) <= 0.
                or np.min(np.diag(self.reference)) < .0001**2
                or np.max(np.diag(self.reference)) > .020**2):
            raise ValueError('Observation measurement covariance must be positive definite in m^2')
        if (self.process.shape != (3,) or not np.isfinite(self.process).all()
                or np.min(self.process) < 0. or np.max(self.process) > 4.):
            raise ValueError('Invalid observation process spectral density')
        if not np.isfinite([self.quality_reference, self.robust_nu, self.maximum_inflation]).all():
            raise ValueError('Finite observation noise settings required')
        if not .001 <= self.quality_reference <= .010:
            raise ValueError('Invalid observation quality reference')
        if not (self.robust_nu == 0. or 2. <= self.robust_nu <= 100.):
            raise ValueError('Observation robust degrees of freedom must be 0 or 2..100')
        if not 1. <= self.maximum_inflation <= 16.:
            raise ValueError('Invalid observation robust inflation cap')
        self.has_bias = bool(self.enabled and np.any(self.bias))
        self.process_matrix = np.diag(self.process)
        for array in (self.bias, self.reference, self.process, self.process_matrix):
            array.setflags(write=False)

    def measurement_noise(self, sigma, near_depth_factor, prior_covariance,
                          innovation, *, robust_allowed=True):
        # Preserve the producer's real depth-quality weight. Reference R is
        # measured at the usual 3 mm scalar quality floor, not at 6 mm gaps.
        scale = max(1., min(2., float(sigma) / self.quality_reference))
        factors = np.array([scale*near_depth_factor, scale*near_depth_factor, scale])
        noise = self.reference * factors[:, None] * factors[None, :]
        inflation = np.ones(3)
        if self.robust_nu and robust_allowed:
            variance = np.maximum(1e-12, np.diag(prior_covariance[:3, :3]) + np.diag(noise))
            normalized = np.asarray(innovation)**2 / variance
            inflation = np.clip((self.robust_nu+normalized)/(self.robust_nu+1.),
                                1., self.maximum_inflation)
            root = np.sqrt(inflation)
            # Congruence scaling preserves positive definiteness and XYZ
            # cross-covariance. It downweights, never rejects, a real sample.
            noise = noise * root[:, None] * root[None, :]
        return noise, float(np.max(inflation))

    def correct_snapshot(self, raw):
        """Output calibration only; never feed corrected predictions to history."""
        if not self.has_bias:
            return
        if raw.get('observation_position_valid') or raw.get('observation_velocity_valid'):
            for i, axis in enumerate('xyz'):
                raw['observation_position_'+axis+'_m'] -= float(self.bias[i])
        if raw.get('observation_bounce_horizon_s') is not None:
            for i, axis in enumerate('xyz'):
                raw['observation_bounce_position_'+axis+'_m'] -= float(self.bias[i])
        # Constant translation has zero time derivative. Velocity, covariance,
        # validity, measured flags and genuine range timestamps are unchanged.


@lru_cache(maxsize=3)
def _load_profile(selection):
    if selection in ('legacy', 'off'):
        return ObservationNoiseProfile()
    if selection != 'default':
        raise ValueError('CAMERA_OBSERVATION_TUNING must be default, legacy or off')
    path = Path(__file__).with_name('ball_observation_calibration.json')
    if not path.is_file():
        return ObservationNoiseProfile()
    calibration = json.loads(path.read_text())
    return ObservationNoiseProfile(calibration.get('observation_tuning'))


def default_profile():
    return _load_profile(os.environ.get('CAMERA_OBSERVATION_TUNING', 'default').strip().lower())
