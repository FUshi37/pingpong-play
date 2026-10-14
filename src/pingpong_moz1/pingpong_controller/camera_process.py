"""Camera-only process and latest-value cache for the control loop.

The worker has no ROS node, robot publisher or policy. Sensor timestamps and
invalid observations survive transport unchanged. The controller never waits
for image processing, JPEG encoding, or a request/reply from this worker.
"""
from dataclasses import asdict
import logging
import multiprocessing as mp
import os
from queue import Empty, Full
import threading
import time
from types import SimpleNamespace

import numpy as np
from pingpong_controller.observation_clock import ReceiverClock, receiver_clock_anchor, sample_clock_valid


def replace_latest(queue, value):
    try:
        queue.put_nowait(value)
        return True
    except Full:
        pass
    try:
        queue.get_nowait()
    except Empty:
        return False
    try:
        queue.put_nowait(value)
        return True
    except Full:
        return False


class _Clock:
    def __init__(self, anchor):
        self.receiver = ReceiverClock(anchor)

    def now(self):
        stamp = int(self.receiver() * 1e9)
        return SimpleNamespace(
            nanoseconds=stamp,
            to_msg=lambda: SimpleNamespace(sec=stamp // 10**9, nanosec=stamp % 10**9))


class _Logger:
    def __init__(self):
        self.logger = logging.getLogger('policy_probe_camera_process')

    def info(self, message, **_):
        self.logger.info(message)

    def warn(self, message, **_):
        self.logger.warning(message)

    def error(self, message, **_):
        self.logger.error(message)


def camera_worker(cfg_values, input_queue, state_queue, frame_queue, stop, parent_pid):
    # Import here: spawn isolates camera libraries and their Python threads.
    from pingpong_controller.tools.interactive_policy_safety_probe import (
        CameraSource, CameraRuntimeOptions, _install_parent_death_signal)
    import signal
    _install_parent_death_signal(signal.SIGKILL)
    cfg = SimpleNamespace(**cfg_values)
    cfg.camera_subprocess = False
    cfg.ros_bridge = False
    cfg.ros_publish_commands = False
    cfg.enable_mocap_ball = False
    source = CameraSource()
    source.cfg = cfg
    source.camera_lock = threading.RLock()
    source.camera_rebuild_lock = threading.Lock()
    source.get_logger = lambda: _Logger()
    clock = _Clock(cfg_values['_receiver_clock_anchor'])
    source.get_clock = lambda: clock
    source._initialize_camera_state(cfg)
    previous_sample = previous_frame = None
    sync_request = 0
    previous_revision = -1
    try:
        source._init_direct_realsense_camera_source()
        replace_latest(state_queue, {'kind': 'ready'})
        while not stop.is_set() and os.getppid() == parent_pid:
            pending = None
            while True:
                try:
                    pending = input_queue.get_nowait()
                except Empty:
                    break
            if pending is not None:
                clock.receiver.anchor = pending['clockAnchor']
                if pending['syncRequest'] != sync_request:
                    source.resynchronize_camera_clock()
                    sync_request = pending['syncRequest']
                changed = pending.get('revision') != previous_revision
                previous_revision = pending.get('revision')
                options = pending.get('options') if changed else None
                if options is not None:
                    options = CameraRuntimeOptions(**options)
                    if options != source.camera_options:
                        source.set_camera_runtime_options(options)
                viz = pending.get('viz') if changed else None
                if viz is not None:
                    # Preserve the original feedback time; transport must not
                    # make an old racket pose look newly measured.
                    source.set_policy_viz_state(
                        *viz['args'], racket_state_valid=viz['valid'],
                        stamp_wall=viz['stamp'], stamp_ros=viz['stampRos'])
            sync = source.latest_camera_clock_sync
            key = (source.camera_frame_count, source.camera_error, sync_request,
                   sync.get('epoch'), sync.get('status'), sync.get('ready'))
            if key != previous_sample:
                with source.camera_lock:
                    pos, vel, valid, feedback = source.camera_ball_state()
                    packet = {
                        'kind': 'sample', 'position': pos, 'velocity': vel,
                        'valid': valid, 'feedback': feedback,
                        'observationWallS': source.latest_camera_wall,
                        'frameStatus': source.camera_control_frame_status(),
                        'rotation': source.camera_frame_rotation(),
                        'syncRequest': sync_request,
                    }
                sent = replace_latest(state_queue, packet)
                if sent:
                    previous_sample = key
            frame, info = source.latest_camera_visualization()
            if frame is not None and info['seq'] != previous_frame:
                if replace_latest(frame_queue, {'frame': frame, 'info': info}):
                    previous_frame = info['seq']
            stop.wait(.002)
    except Exception as exc:
        replace_latest(state_queue, {'kind': 'error', 'error': str(exc)})
    finally:
        source.shutdown_camera_source()


class CameraProcess:
    def __init__(self, cfg, *, receiver_clock=None, worker_target=camera_worker, startup_timeout_s=90.):
        self.receiver_clock = receiver_clock or time.time
        self.sync_request = 0
        self.ctx = mp.get_context('spawn')
        self.input_queue = self.ctx.Queue(maxsize=1)
        self.state_queue = self.ctx.Queue(maxsize=2)
        self.frame_queue = self.ctx.Queue(maxsize=1)
        self.stop_event = self.ctx.Event()
        self._pump_stop = threading.Event()
        self._ready = threading.Event()
        self._pending = {}
        self._revision = 0
        self._sample = None
        self._frame = None
        self.error = None
        self.closed = False
        values = asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else dict(vars(cfg))
        values.update(ros_bridge=False, ros_publish_commands=False, enable_mocap_ball=False)
        values['_receiver_clock_anchor'] = receiver_clock_anchor(self.receiver_clock)
        self.process = self.ctx.Process(
            target=worker_target,
            args=(values, self.input_queue, self.state_queue, self.frame_queue,
                  self.stop_event, os.getpid()),
            name='policy_probe_camera', daemon=False)
        self.process.start()
        self.thread = threading.Thread(target=self._pump, name='camera_cache_receiver', daemon=True)
        self.thread.start()
        deadline = time.monotonic() + startup_timeout_s
        while not self._ready.wait(.05):
            if self.error or not self.process.is_alive() or time.monotonic() > deadline:
                message = self.error or 'camera process did not become ready'
                self.close()
                raise RuntimeError(message)

    def _pump(self):
        sent_revision = -1
        next_clock = 0.
        while not self._pump_stop.is_set():
            now = time.monotonic()
            if self._revision != sent_revision or now >= next_clock:
                revision = self._revision
                try:
                    pending = dict(self._pending, clockAnchor=receiver_clock_anchor(self.receiver_clock),
                                   syncRequest=self.sync_request, revision=revision)
                    if replace_latest(self.input_queue, pending):
                        sent_revision = revision
                        next_clock = now + .01
                except ValueError:
                    # Discard an imprecise anchor and retry, never relabel a
                    # delayed clock read as the time of this sensor frame.
                    next_clock = now + .01
                except Exception as exc:
                    self.error = f'consumer clock unavailable: {exc}'
                    break
            try:
                packet = self.state_queue.get(timeout=.004)
                if packet['kind'] == 'ready':
                    self._ready.set()
                elif packet['kind'] == 'error':
                    self.error = packet['error']
                elif packet['kind'] == 'sample':
                    # One immutable packet is the atomic position/velocity/
                    # source-timestamp envelope read by the controller.
                    self._sample = packet
            except Empty:
                pass
            except (EOFError, OSError) as exc:
                self.error = f'camera transport failed: {exc}'
                break
            try:
                self._frame = self.frame_queue.get_nowait()
            except Empty:
                pass
            except (EOFError, OSError) as exc:
                self.error = f'camera frame transport failed: {exc}'
                break
            if not self.process.is_alive():
                self.error = self.error or 'camera process exited'
                break

    def set_options(self, options):
        self._pending = {**self._pending, 'options': asdict(options)}
        self._revision += 1

    def resynchronize(self):
        self.sync_request += 1
        self._revision += 1

    def set_viz(self, args, valid, stamp, *, stamp_ros=None):
        self._pending = {**self._pending, 'viz': {
            'args': tuple(np.asarray(x).copy() if isinstance(x, np.ndarray) else x for x in args),
            'valid': valid, 'stamp': stamp, 'stampRos': stamp if stamp_ros is None else stamp_ros}}
        self._revision += 1

    def measurement(self):
        return self._measurement(self._sample)

    def _measurement(self, sample):
        if sample is None:
            return np.zeros(3, np.float32), np.zeros(3, np.float32), False, None, {
                'clockSync': dict(required=True, ready=False, status='waiting_for_capture')}
        feedback = sample['feedback']
        valid = bool(sample['valid'] and not self.error and not self.closed and self.process.is_alive())
        sync = self._clock_info(sample)
        if sync is not None:
            valid = bool(valid and sync.get('ready'))
        elif self.sync_request:
            valid = False
        measurement = dict(feedback.get('measurement') or {})
        if sync is not None:
            measurement['clockSync'] = dict(sync)
        return (np.asarray(sample['position'], np.float32).copy(),
                np.asarray(sample['velocity'], np.float32).copy(), valid,
                feedback.get('stampS'), measurement)

    def _clock_info(self, sample):
        original = sample['feedback'].get('clockSync')
        if self.error or self.closed or not self.process.is_alive():
            return dict(original or {}, required=True, ready=False, status='camera_process_unavailable')
        if original is None:
            return (dict(required=True, ready=False, status='awaiting_resynchronization')
                    if self.sync_request else None)
        sync = dict(original)
        if sample.get('syncRequest', 0) != self.sync_request:
            sync.update(ready=False, status='awaiting_resynchronization')
        elif sync.get('ready'):
            try:
                valid = sample_clock_valid(sync, self.receiver_clock())
            except Exception:
                valid = False
            if not valid:
                sync.update(ready=False, status='waiting_for_fresh_capture_or_clock_reset')
        return sync

    def state(self):
        sample = self._sample
        if sample is None:
            feedback = {'enabled': True, 'ready': False, 'detected': False, 'error': self.error,
                        'clockSync': dict(required=True, ready=False, status='waiting_for_capture')}
        else:
            feedback = dict(sample['feedback'])
            wall = sample['observationWallS']
            feedback['ageS'] = None if wall is None else max(0., time.time()-wall)
            sync = self._clock_info(sample) or {}
            feedback['clockSync'] = sync
            stamp = feedback.get('stampS')
            try:
                feedback['measurementAgeS'] = None if stamp is None else self.receiver_clock() - stamp
            except Exception:
                feedback['measurementAgeS'] = None
        feedback['cameraProcess'] = {'enabled': True, 'alive': self.process.is_alive(), 'pid': self.process.pid}
        if self.error:
            feedback.update(detected=False, ready=False, error=self.error)
        pos, vel, valid, _, _ = self._measurement(sample)
        feedback['detected'] = valid
        if feedback.get('clockSync', {}).get('required'):
            feedback['ready'] = bool(feedback.get('ready') and feedback['clockSync'].get('ready'))
        return pos, vel, valid, feedback

    def frame_status(self):
        sample = self._sample
        return (False, 'camera') if sample is None else tuple(sample['frameStatus'])

    def rotation(self):
        sample = self._sample
        r = None if sample is None else sample['rotation']
        return None if r is None else np.asarray(r).copy()

    def visualization(self):
        frame = self._frame
        if frame is None:
            return None, {'seq': 0, 'ageS': None, 'enabled': False}
        info = dict(frame['info'])
        wall = info.get('wallTimeS')
        info['ageS'] = None if wall is None else max(0., time.time()-wall)
        return frame['frame'], info

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.process.join(timeout=8.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.)
        self._pump_stop.set()
        self.thread.join(timeout=1.)
        for queue in (self.input_queue, self.state_queue, self.frame_queue):
            queue.cancel_join_thread()
            queue.close()
