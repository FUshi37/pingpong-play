"""Bounded camera observation recovery; never marks an invalid state as valid.

An acquired ball may briefly need a new flight/contact fit. During that time
the caller holds a measured arm position, keeps reception/publication alive,
and resumes only after a real measurement and a valid state agree. Prediction
age, model uncertainty, clock and robot protections remain independent.
"""
import math


class CameraObservationContinuity:
    REVISION = 'camera-continuity-v1.4.1-20261019'
    MAX_WAIT_S = .200
    TRANSIENT_REASONS = frozenset({
        'prediction_timeout', 'supported_prediction_timeout',
        'prediction_uncertainty_limit', 'prediction_below_contact_band',
        'contact_pending', 'measured_position_only', 'insufficient_history',
        'new_segment_confirming', 'contact_prediction_limit',
        'contact_confirmation_timeout', 'contact_prediction_timeout',
    })

    def __init__(self):
        self.reset()

    def reset(self):
        self.source_key = None
        self.last_time = None
        self.started = None
        self.acquired = False
        self.last_observed_stamp = None
        self.wait_observed_stamp = None
        self.last_info = dict(revision=self.REVISION, action='inactive', elapsedS=0., maximumWaitS=self.MAX_WAIT_S)

    def step(self, now, *, enabled, source_key, valid, observed, ready, reason,
             measurement_stamp=None):
        if not enabled:
            self.reset()
            return self.last_info
        if self.source_key != source_key:
            self.reset()
            self.source_key = source_key
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            self.last_info = dict(revision=self.REVISION, action='fault', elapsedS=0., maximumWaitS=self.MAX_WAIT_S,
                                  modelReason='continuity_clock_fault')
            return self.last_info
        self.last_time = now
        real_observation = bool(observed and measurement_stamp is not None
                                and math.isfinite(measurement_stamp))
        waiting = self.started is not None
        action = 'running'
        if waiting:
            elapsed = max(0., now-self.started)
            if (elapsed > self.MAX_WAIT_S+1e-6
                    or (not valid and reason not in self.TRANSIENT_REASONS)):
                action = 'fault'
                self.started = None
                self.acquired = False
            elif (valid and real_observation and self.wait_observed_stamp is not None
                  and measurement_stamp > self.wait_observed_stamp+1e-6):
                action = 'resumed'
                self.started = None
                self.last_observed_stamp = measurement_stamp
            else:
                action = 'waiting'
        elif valid:
            elapsed = 0.
            self.acquired = self.acquired or real_observation
            if real_observation:
                self.last_observed_stamp = measurement_stamp
        elif self.acquired and ready and reason in self.TRANSIENT_REASONS:
            self.started = now
            self.wait_observed_stamp = (measurement_stamp if real_observation
                                        else self.last_observed_stamp)
            elapsed, action = 0., 'waiting'
        else:
            elapsed, action = 0., 'fault'
        self.last_info = dict(revision=self.REVISION, action=action, elapsedS=elapsed, maximumWaitS=self.MAX_WAIT_S,
                              modelReason=reason, stateValid=bool(valid), observed=bool(observed))
        return self.last_info
