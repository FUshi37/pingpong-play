"""Bounded-memory ball/racket bounce counting in the recording writer."""

import math
from collections import deque

from pingpong_controller.probe_ball_recording import finite_number


def _vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    result = tuple(finite_number(component) for component in value)
    return None if None in result else result


class RecordingJuggleCounter:
    """Count measured falling-to-rising reversals close to the real racket.

    Consume only the selected physical sensor, once per measurement. Never
    join evidence across invalid observations, source/clock changes or gaps.
    Thresholds follow the existing offline trajectory contact detector.
    """

    WINDOW_S = 0.09
    REFRACTORY_S = 0.16
    MIN_SPEED_M_S = 0.10
    MIN_JUMP_M_S = 0.70

    def __init__(self):
        self.count = 0
        self.by_source = {"camera": 0, "mocap": 0}
        self.measurement_samples = 0
        self.context = None
        self.last_stamp = None
        self.last_control_time = None
        self.last_bounce_time = -math.inf
        self.recent = deque(maxlen=32)

    @staticmethod
    def _near_racket(position, racket):
        if position is None or racket is None:
            return False
        dx, dy, dz = (p - r for p, r in zip(position, racket))
        return math.hypot(dx, dy) <= 0.095 and -0.035 <= dz <= 0.120

    def consume(self, row):
        """Return one auditable event per bounce; retain no event history."""
        source = row.get("ballSource")
        sensors = row.get("ballSensors") or {}
        sensor = sensors.get(source) if isinstance(sensors, dict) else None
        if isinstance(sensor, dict):
            valid = sensor.get("valid", False) and sensor.get("frameAligned", False)
            position = _vector(sensor.get("position"))
            velocity = _vector(sensor.get("velocity"))
            stamp = finite_number(sensor.get("measurementTimestampS"))
            sync = sensor.get("clockSync") or {}
            measurement = sensor.get("measurement") or {}
        else:
            valid = row.get("rawValid", False)
            position = _vector(row.get("measuredBall"))
            velocity = _vector(row.get("measuredVelocity"))
            stamp = finite_number(row.get("cameraStampS"))
            sync = row.get("ballClockSync") or {}
            measurement = row.get("cameraMeasurement") or {}
        context = (source, sync.get("generation"), sync.get("epoch"), measurement.get("markerId"))
        if context != self.context:
            self.recent.clear()
            self.last_stamp = None
            self.context = context
        control_time = finite_number(row.get("sourceTimeS", row.get("time")))
        if (source not in self.by_source or not valid or stamp is None
                or position is None or velocity is None or control_time is None
                or sync.get("ready") is False):
            self.recent.clear()
            return None
        if self.last_control_time is not None and control_time < self.last_control_time:
            self.recent.clear()
            self.last_stamp = None
            self.last_bounce_time = -math.inf
        self.last_control_time = control_time
        if stamp == self.last_stamp:
            return None
        if self.last_stamp is not None and not 0.0 < stamp - self.last_stamp <= self.WINDOW_S:
            self.recent.clear()
        self.last_stamp = stamp
        self.measurement_samples += 1
        ros = row.get("ros") or {}
        racket = _vector(ros.get("jointEe")) or _vector(row.get("racket"))
        while self.recent and stamp - self.recent[0][0] > self.WINDOW_S:
            self.recent.popleft()
        before = min(self.recent, key=lambda item: item[1]) if self.recent else None
        event = None
        if (before is not None and before[1] < -self.MIN_SPEED_M_S
                and velocity[2] > self.MIN_SPEED_M_S
                and velocity[2] - before[1] >= self.MIN_JUMP_M_S
                and control_time - self.last_bounce_time >= self.REFRACTORY_S
                and (self._near_racket(position, racket)
                     or self._near_racket(before[2], before[3]))):
            self.count += 1
            self.by_source[source] += 1
            self.last_bounce_time = control_time
            self.recent.clear()
            event = {
                "event": "juggle_bounce", "count": self.count,
                "ballSource": source, "sourceSeq": row.get("sourceSeq"),
                "sourceTimeS": control_time, "wallTimeS": row.get("wallTimeS"),
                "measurementTimestampS": stamp,
                "velocityBeforeMps": before[1], "velocityAfterMps": velocity[2],
                "ball": position, "racket": racket,
            }
        self.recent.append((stamp, velocity[2], position, racket))
        return event

    def summary(self, *, complete=False, dropped=0):
        return {
            "count": self.count,
            "bySource": dict(self.by_source),
            "measurementSamples": self.measurement_samples,
            "method": "measured_velocity_reversal_near_racket",
            "status": ("recording" if not complete else
                       "no_valid_observations" if not self.measurement_samples else
                       "partial" if dropped else "complete"),
            "recorderDropped": dropped,
            "windowS": self.WINDOW_S, "refractoryS": self.REFRACTORY_S,
            "minSpeedMps": self.MIN_SPEED_M_S, "minVelocityJumpMps": self.MIN_JUMP_M_S,
        }
