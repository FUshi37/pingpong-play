"""Detect a sustained violation of the policy's physical 5 ms clock."""
from collections import deque
import math


class PolicyDeadlineMonitor:
    def __init__(self, dt_s=.005, window=80):
        self.dt_s = dt_s
        self.window = window
        self.intervals = deque(maxlen=window)
        self.previous = None
        self.profile = None

    def reset(self):
        self.intervals.clear()
        self.previous = None
        self.profile = None

    def start(self, now, active, profile):
        if not active:
            self.reset()
            return None
        if self.profile != profile:
            self.reset()
            self.profile = profile
        previous, self.previous = self.previous, now
        if previous is None:
            return None
        interval = now-previous
        if not math.isfinite(interval) or interval <= 0:
            return {'reason': 'invalid_control_clock', 'intervalS': interval}
        self.intervals.append(interval)
        mean = sum(self.intervals)/len(self.intervals)
        if interval > 5*self.dt_s or (len(self.intervals) == self.window and mean > 1.08*self.dt_s):
            return {'reason': 'policy_control_deadline_exceeded',
                    'intervalMs': interval*1000, 'windowMeanMs': mean*1000,
                    'targetHz': 1/self.dt_s, 'windowHz': 1/mean,
                    'windowSamples': len(self.intervals)}
        return None

    def payload(self):
        mean = sum(self.intervals)/len(self.intervals) if self.intervals else None
        return {'activePolicyHz': None if mean is None else 1/mean,
                'windowSamples': len(self.intervals), 'windowSize': self.window,
                'maximumGapMs': self.dt_s*5000,
                'sustainedMinimumHz': 1/(1.08*self.dt_s)}
