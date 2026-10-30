"""Receive/transform mocap outside the policy process; cache atomic observations.

Uses the existing UDP parser, calibration, velocity estimator and clock fit.
The policy never waits for their lock. Polling cannot refresh source timestamps
or validity; cached observations age in the consumer's clock domain.
"""
import multiprocessing as mp
import os
from queue import Empty
import signal
import threading
import time

import numpy as np

from pingpong_controller.camera_process import replace_latest
from pingpong_controller.observation_clock import ReceiverClock, receiver_clock_anchor
from pingpong_controller.probe_ball_recording import make_sensor_recording_sample


class _StopFlag:
    """A lock-free byte remains writable after an externally killed worker."""
    def __init__(self, ctx):
        self.value = ctx.RawValue('b', False)

    def set(self):
        self.value.value = True

    def is_set(self):
        return bool(self.value.value)

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while not self.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            time.sleep(min(.01, left))
        return True


def mocap_worker(kwargs, input_q, state_q, stop, parent_pid, anchor):
    from pingpong_controller.mocap_input import MocapBallInput
    from pingpong_controller.tools.interactive_policy_safety_probe import _install_parent_death_signal
    _install_parent_death_signal(signal.SIGKILL)
    clock = ReceiverClock(anchor)
    source = MocapBallInput(**kwargs, receiver_clock=clock)
    previous = None
    sync_request = 0
    try:
        if not source.start():
            raise RuntimeError(str(source._last_start_error))
        replace_latest(state_q, {'kind': 'ready'})
        while not stop.is_set() and os.getppid() == parent_pid:
            pending = None
            while True:
                try:
                    pending = input_q.get_nowait()
                except Empty:
                    break
            if pending is not None:
                clock.anchor = pending['clockAnchor']
                if pending['syncRequest'] != sync_request:
                    source.resynchronize()
                    sync_request = pending['syncRequest']
                marker = int(pending['markerId'])
                if marker != source.marker_id:
                    with source.detector._lock:
                        source.marker_id = source.detector.marker_id = marker
                        # A different marker needs its own source-clock/velocity
                        # history; never relabel a cached previous marker fresh.
                        source._resynchronize_locked('marker_changed')
            observed_monotonic = time.monotonic()
            p, v, valid, feedback = source.get_control_state()
            feedback['syncRequest'] = sync_request
            key = (sync_request, feedback.get('packetCount'), feedback.get('parseErrorCount'), valid,
                   feedback.get('udp', {}).get('received_count'),
                   feedback.get('udp', {}).get('receiver_error_count'),
                   feedback.get('udp', {}).get('status'),
                   feedback.get('markerId'), tuple(sorted(feedback.get('clockSync', {}).items())))
            if key != previous:
                packet = dict(kind='sample', position=p, velocity=v, valid=valid,
                              feedback=feedback, observedMonotonicS=observed_monotonic,
                              recording=(make_sensor_recording_sample(
                                  'mocap', p, v, valid, feedback.get('measurementTimestampS'), feedback),
                                  sync_request, observed_monotonic - float(feedback.get('ageS') or 0.)))
                if replace_latest(state_q, packet):
                    previous = key
            stop.wait(.002)
    except Exception as exc:
        replace_latest(state_q, {'kind': 'error', 'error': str(exc)})
    finally:
        source.stop()


class MocapInputProcess:
    def __init__(self, *, receiver_clock=None, worker_target=mocap_worker, **kwargs):
        self.receiver_clock = receiver_clock or time.time
        self.sync_request = 0
        self.kwargs = kwargs
        self.marker_id = int(kwargs.get('marker_id', 1))
        self.timeout_s = float(kwargs.get('timeout_s', .5))
        self.worker_target = worker_target
        self.ctx = mp.get_context('spawn')
        self.process = None
        self.thread = None
        self._sample = None
        self.error = None
        self.closed = False
        self._last_start_error = None
        self.ready = threading.Event()
        self.pump_stop = threading.Event()

    def _anchor(self):
        return receiver_clock_anchor(self.receiver_clock)

    def start(self):
        if self.process is not None:
            return self.process.is_alive() and not self.closed
        self.input_q = self.ctx.Queue(maxsize=1)
        self.state_q = self.ctx.Queue(maxsize=2)
        self.stop_event = _StopFlag(self.ctx)
        self.process = self.ctx.Process(target=self.worker_target,
            args=(self.kwargs, self.input_q, self.state_q, self.stop_event, os.getpid(), self._anchor()),
            name='policy_probe_mocap', daemon=False)
        self.process.start()
        self.thread = threading.Thread(target=self._pump, name='mocap_cache_receiver', daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 90.
        while not self.ready.wait(.05):
            if self.error or not self.process.is_alive() or time.monotonic() > deadline:
                self._last_start_error = RuntimeError(self.error or 'mocap process did not become ready')
                self.stop()
                return False
        return True

    def _pump(self):
        next_clock = 0.
        while not self.pump_stop.is_set():
            now = time.monotonic()
            if now >= next_clock:
                try:
                    replace_latest(self.input_q, {'clockAnchor': self._anchor(),
                        'markerId': self.marker_id, 'syncRequest': self.sync_request})
                except ValueError:
                    pass  # Retry an imprecise clock read; preserve source stamps.
                except Exception as exc:
                    self.error = f'consumer clock unavailable: {exc}'
                    break
                next_clock = now + .01
            try:
                packet = self.state_q.get(timeout=.002)
                if packet['kind'] == 'ready':
                    self.ready.set()
                elif packet['kind'] == 'sample':
                    self._sample = packet
                elif packet['kind'] == 'error':
                    self.error = packet['error']
            except Empty:
                pass
            except (EOFError, OSError) as exc:
                self.error = f'mocap transport failed: {exc}'
                break
            if not self.process.is_alive():
                self.error = self.error or 'mocap process exited'
                break

    def set_marker_id(self, marker_id):
        marker_id = int(marker_id)
        if marker_id != self.marker_id:
            self.marker_id = marker_id
            self.resynchronize()

    def resynchronize(self):
        self.sync_request += 1

    def get_control_state(self):
        sample = self._sample
        feedback = (dict(clockSync=dict(required=True, ready=False, status='waiting_for_packet'))
                    if sample is None else dict(sample['feedback']))
        valid = False
        if sample is not None:
            # Both timestamps remain those of the actual UDP observation.
            elapsed = time.monotonic() - sample['observedMonotonicS']
            udp = dict(feedback.get('udp') or {})
            if udp.get('last_packet_age_s') is not None:
                udp['last_packet_age_s'] += elapsed
                if udp['last_packet_age_s'] >= self.timeout_s:
                    udp['status'] = 'udp_timeout'
            if udp:
                feedback['udp'] = udp
            arrival_age = float(feedback.get('ageS') or 0.) + elapsed
            stamp = feedback.get('measurementTimestampS')
            receiver_stamp = feedback.get('receiverTimestampS')
            try:
                now = float(self.receiver_clock())
            except Exception:
                now = float('nan')
            measurement_age = None if stamp is None else now - float(stamp)
            clock_ok = bool(receiver_stamp is not None
                and abs((now - float(receiver_stamp)) - arrival_age) <= .25)
            sync = dict(feedback.get('clockSync') or {})
            if receiver_stamp is None:
                # Before the first accepted ball observation there is no
                # receipt clock to compare. Keep control blocked, but do not
                # report a clock jump that has never been observed.
                sync.update(ready=False, status='waiting_for_first_observation')
            elif not clock_ok:
                sync.update(ready=False, status='receiver_clock_changed')
            if feedback.get('syncRequest', 0) != self.sync_request:
                sync.update(ready=False, status='awaiting_resynchronization')
            elif sync.get('ready') and (measurement_age is None
                    or not 0. <= measurement_age < self.timeout_s
                    or not 0. <= arrival_age < self.timeout_s):
                sync.update(ready=False, status='waiting_for_fresh_packet')
            tracked = bool(feedback.get('tracked', False) and 0. <= arrival_age < self.timeout_s)
            valid = bool(sample['valid'] and tracked and sync.get('ready') and measurement_age is not None
                and 0. <= measurement_age < self.timeout_s
                and feedback.get('markerId') == self.marker_id)
            feedback.update(ageS=arrival_age, measurementAgeS=measurement_age,
                            clockSync=sync, tracked=tracked, detected=valid)
        alive = bool(self.process is not None and self.process.is_alive())
        valid = bool(valid and alive and not self.closed and not self.error)
        if not valid:
            feedback.update(detected=False, controlBall=None, controlVelocity=None)
        if self.error or self.closed or not alive:
            feedback['error'] = self.error or 'mocap process stopped'
            sync = dict(feedback.get('clockSync') or {})
            sync['ready'] = False
            feedback['clockSync'] = sync
        feedback['mocapProcess'] = dict(enabled=True, alive=alive,
            pid=None if self.process is None else self.process.pid)
        return (np.asarray(sample['position'], np.float32).copy(),
                np.asarray(sample['velocity'], np.float32).copy(), True, feedback) if valid else (
                    np.zeros(3, np.float32), np.zeros(3, np.float32), False, feedback)

    def recording_snapshot(self):
        sample = self._sample
        return (None if sample is None else sample.get('recording'),
                self.error, self.closed, self.sync_request, self.timeout_s)

    def stop(self):
        if self.closed:
            return
        self.closed = True
        self.pump_stop.set()
        if self.process is not None:
            self.stop_event.set()
            self.process.join(timeout=4.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2.)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=1.)
        if self.thread is not None:
            self.thread.join(timeout=1.)
        for name in ('input_q', 'state_q'):
            queue = getattr(self, name, None)
            if queue is not None:
                queue.cancel_join_thread()
                queue.close()
