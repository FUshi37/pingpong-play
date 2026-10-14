"""Capture clock alignment and ROS-clock transport for sensor-only workers."""
import math
import time

from pingpong_controller.mocap_clock_sync import MocapClockSynchronizer


def receiver_clock_anchor(clock):
    before = time.monotonic()
    receiver = float(clock())
    after = time.monotonic()
    if not math.isfinite(receiver) or after - before > .005:
        raise ValueError('receiver clock anchor is nonfinite or took more than 5 ms')
    return receiver, .5 * (before + after)


class ReceiverClock:
    """System-rate clock anchored in the actual consumer's ROS time domain."""
    def __init__(self, anchor):
        self.anchor = anchor

    def __call__(self):
        receiver_s, monotonic_s = self.anchor
        return receiver_s + time.monotonic() - monotonic_s


class CameraTimestampAligner:
    """Map device captures; validate ROS headers without erasing their delay.

    A device stream uses the same one-way rolling minimum offset as mocap.
    Its fixed transport latency is unidentifiable. ROS image headers already
    belong to the shared ROS clock and retain their original capture time.
    """
    def __init__(self):
        self.fit = MocapClockSynchronizer(day_rollover=False)
        self.domain = None
        self.generation = 0
        self.latest = self.diagnostics()

    def reset(self, reason='source_selected'):
        self.generation += 1
        self.fit.reset(reason)
        self.latest = self.diagnostics()

    def diagnostics(self):
        return dict(self.fit.diagnostics(), generation=self.generation,
                    sourceDomain=self.domain, receiverDomain='ROS',
                    required=True)

    def update(self, source_s, receiver_s, monotonic_s, *, domain, frame_idx=None, system_s=None):
        if self.domain is not None and domain != self.domain:
            self.reset('source_clock_domain_changed')
        self.domain = domain
        system_capture = domain.endswith(('/timestamp_domain.global_time', '/timestamp_domain.system_time'))
        capture_now = system_s if system_capture else receiver_s
        minimum_delay = -.025 if system_capture else 0.
        if (domain == 'ros_header' or system_capture) and (
                source_s is None or not math.isfinite(source_s) or source_s <= 0.
                or capture_now is None or not math.isfinite(capture_now)
                or not minimum_delay <= capture_now - source_s < .5):
            self.fit.reset('invalid_system_capture_clock' if system_capture else 'invalid_ros_header_clock')
            self.latest = self.diagnostics()
            self.latest.update(sourceTimestampS=source_s, receiverTimestampS=receiver_s,
                               systemTimestampS=system_s, measurementTimestampS=None,
                               receiptMonotonicS=monotonic_s)
            return None, dict(self.latest)
        result = self.fit.update(source_s, receiver_s, monotonic_s, frame_idx=frame_idx)
        info = self.diagnostics()
        mapped = result.receiver_timestamp_s
        if domain == 'ros_header':
            mapped = source_s if result.ready else None
            info.update(method='shared_ROS_header_clock_validation', offset_s=0.,
                        captureLatencyPreserved=True)
        elif system_capture:
            # RealSense global/system timestamps already use this host's
            # wall clock. Translate that known domain to ROS; do not subtract
            # minimum USB latency and make delayed captures appear fresher.
            known_offset = receiver_s - system_s
            # Global timestamps are estimates of the host clock and can lead
            # it slightly. Calibrate only toward the past. Positive USB delay
            # is never removed, and mapped frames can never be in the future.
            offset = min(known_offset, self.fit.offset_s) if self.fit.offset_s is not None else known_offset
            mapped = source_s + offset if result.ready else None
            info.update(method='system_capture_to_ROS_clock', offset_s=offset,
                        captureLatencyPreserved=True,
                        futureClockCorrectionS=offset-known_offset)
        info.update(sourceTimestampS=source_s, receiverTimestampS=receiver_s,
                    measurementTimestampS=mapped, receiptMonotonicS=monotonic_s,
                    accepted=result.accepted, status=result.status)
        self.latest = info
        return (mapped if result.accepted and result.ready else None), dict(info)


def sample_clock_valid(info, receiver_now, *, monotonic_now=None, timeout_s=.5):
    """Cached metadata cannot become fresh by polling or by a clock jump."""
    if not info or not info.get('ready'):
        return False
    try:
        now_m = time.monotonic() if monotonic_now is None else monotonic_now
        age = receiver_now - info['measurementTimestampS']
        arrival_age = now_m - info['receiptMonotonicS']
        receiver_elapsed = receiver_now - info['receiverTimestampS']
        return (all(math.isfinite(x) for x in (age, arrival_age, receiver_elapsed))
                and 0. <= age < timeout_s and 0. <= arrival_age < timeout_s
                and abs(receiver_elapsed - arrival_age) <= .25)
    except (KeyError, TypeError, ValueError):
        return False
