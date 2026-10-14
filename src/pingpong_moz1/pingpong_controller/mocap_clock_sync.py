"""Map a one-way mocap clock to the receiver clock, without changing positions.

The minimum arrival-minus-source offset over a recent window removes variable
queueing delay. Fixed capture/transport delay is inseparable from clock offset
in a one-way stream: the resulting observation age is a lower-bound estimate,
not an NTP/PTP measurement of absolute acquisition time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ClockSyncResult:
    accepted: bool
    source_unwrapped_s: float | None
    receiver_timestamp_s: float | None
    ready: bool
    status: str
    excess_delay_s: float | None
    epoch: int


class MocapClockSynchronizer:
    """Rolling offset estimate; duplicate/late packets cannot refresh a sample.

    ``receiver_s`` must use the clock against which the consumer computes age
    (ROS time for ROS nodes). Monotonic receipt time detects receiver clock
    changes and makes the fit independent of wall-clock adjustments.
    """

    def __init__(self, *, min_samples=8, min_span_s=.05, window_s=10.,
                 reset_jump_s=.25, max_gap_s=.5, day_rollover=True):
        if (min_samples < 2 or not all(math.isfinite(v) and v > 0 for v in
                (min_span_s, window_s, reset_jump_s, max_gap_s))
                or window_s < min_span_s):
            raise ValueError('Invalid mocap clock synchronization settings')
        self.min_samples = int(min_samples)
        self.min_span_s = float(min_span_s)
        self.window_s = float(window_s)
        self.reset_jump_s = float(reset_jump_s)
        self.max_gap_s = float(max_gap_s)
        self.day_rollover = bool(day_rollover)
        self.epoch = 0
        self.reset_count = 0
        self.duplicate_count = 0
        self.out_of_order_count = 0
        self.rollover_count = 0
        self.reset('initializing', count=False)

    def reset(self, reason, *, count=True):
        if count:
            self.epoch += 1
            self.reset_count += 1
        self.status = reason
        self.ready = False
        self._samples = deque(maxlen=4096)
        self._previous_source = None
        self._previous_unwrapped = None
        self._previous_receiver = None
        self._previous_monotonic = None
        self._previous_frame = None
        self._previous_mapped_monotonic = None
        self._source_origin = None
        self._day_offset = 0.
        self.offset_s = None
        self.excess_delay_s = None

    def update(self, source_s, receiver_s, monotonic_s, *, frame_idx=None):
        receiver_s, monotonic_s = float(receiver_s), float(monotonic_s)
        if not all(math.isfinite(v) for v in (receiver_s, monotonic_s)):
            self.reset('invalid_receiver_clock')
            return self._result(False, None)
        if source_s is None or not math.isfinite(float(source_s)) or float(source_s) < 0:
            self.reset('missing_source_timestamp')
            # Tracking may still use the position, but control must await sync.
            return self._result(True, None)
        source_s = float(source_s)
        unwrapped = source_s + self._day_offset
        if self._previous_source is not None:
            dm = monotonic_s - self._previous_monotonic
            dr = receiver_s - self._previous_receiver
            if dm < 0 or dr <= 0 or abs(dr - dm) > self.reset_jump_s:
                self.reset('receiver_clock_reset')
            elif dm > self.max_gap_s:
                self.reset('receipt_gap')
            else:
                # Only an actual end-of-day -> start-of-day transition wraps.
                if (self.day_rollover and 86300 <= self._previous_source < 86400
                        and 0 <= source_s <= 100):
                    unwrapped += 86400
                # A late packet from the preceding day must remain late.
                elif (self.day_rollover and self._day_offset and self._previous_source <= 100
                      and 86300 <= source_s < 86400):
                    unwrapped -= 86400
                ds = unwrapped - self._previous_unwrapped
                frame_restart = (frame_idx is not None and self._previous_frame is not None
                                 and self._previous_frame >= 32 and frame_idx <= 4)
                if ds < -self.reset_jump_s or (frame_restart and ds <= 0):
                    # Reacquire rather than stamp a possible restarted stream fresh.
                    self.reset('source_clock_reset')
                elif ds < 0:
                    self.out_of_order_count += 1
                    return self._result(False, unwrapped, status='out_of_order')
                elif ds == 0 or (frame_idx is not None
                                and frame_idx == self._previous_frame):
                    self.duplicate_count += 1
                    return self._result(False, unwrapped, status='duplicate')
                elif abs(ds - dm) > self.reset_jump_s:
                    self.reset('source_clock_jump')
                elif unwrapped - source_s > self._day_offset:
                    self._day_offset += 86400
                    self.rollover_count += 1
        if self._previous_source is None:
            unwrapped = source_s
            self._source_origin = unwrapped

        source_elapsed = unwrapped - self._source_origin
        offset_monotonic = monotonic_s - source_elapsed
        while self._samples and monotonic_s - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        self._samples.append((monotonic_s, source_elapsed, offset_monotonic))
        minimum_offset = min(sample[2] for sample in self._samples)
        mapped_monotonic = source_elapsed + minimum_offset
        if (self._previous_mapped_monotonic is not None
                and mapped_monotonic <= self._previous_mapped_monotonic):
            self.reset('offset_reacquisition')
            return self.update(source_s, receiver_s, monotonic_s, frame_idx=frame_idx)
        self.excess_delay_s = max(0., monotonic_s - mapped_monotonic)
        self.offset_s = receiver_s - self.excess_delay_s - unwrapped
        source_span = source_elapsed - self._samples[0][1]
        receipt_span = monotonic_s - self._samples[0][0]
        self.ready = (len(self._samples) >= self.min_samples
                      and min(source_span, receipt_span) >= self.min_span_s)
        self.status = 'synchronized' if self.ready else 'warming_up'
        self._previous_source = source_s
        self._previous_unwrapped = unwrapped
        self._previous_receiver = receiver_s
        self._previous_monotonic = monotonic_s
        self._previous_frame = frame_idx
        self._previous_mapped_monotonic = mapped_monotonic
        return self._result(True, unwrapped,
                            receiver_s - self.excess_delay_s if self.ready else None)

    def _result(self, accepted, source, receiver=None, *, status=None):
        return ClockSyncResult(accepted, source, receiver, self.ready,
                               status or self.status, self.excess_delay_s, self.epoch)

    def diagnostics(self):
        return dict(ready=self.ready, status=self.status, epoch=self.epoch,
                    sample_count=len(self._samples), offset_s=self.offset_s,
                    excess_delay_s=self.excess_delay_s,
                    reset_count=self.reset_count, duplicate_count=self.duplicate_count,
                    out_of_order_count=self.out_of_order_count,
                    rollover_count=self.rollover_count,
                    method='rolling_minimum_one_way_offset',
                    fixed_latency_calibrated=False)
