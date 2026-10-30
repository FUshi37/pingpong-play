"""Best-effort raw camera sidecar. Never called from the control/receiver loop.

offer() only replaces one Python reference. A low-priority copy thread moves
selected frames into a fixed shared-memory pool; a separate low-priority
process writes them. Full pools, slow storage and errors drop recording data.
Images never use the camera/control IPC queues. Start/stop requests carry
only a directory string through the existing camera management channel.
"""
from dataclasses import asdict, is_dataclass
import json
import hashlib
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
from queue import Empty, Full
import threading
import time
import shutil

import numpy as np


def low_priority(*, idle=False):
    """Linux priorities are per thread. Never change the capture thread."""
    try:
        tid = threading.get_native_id()
        os.setpriority(os.PRIO_PROCESS, tid, 19 if idle else 10)
        # Never put a Python thread sharing the detector's GIL in SCHED_IDLE:
        # starvation while it owns the GIL could delay camera reception.
        if idle and hasattr(os, 'SCHED_IDLE'):
            os.sched_setscheduler(tid, os.SCHED_IDLE, os.sched_param(0))
    except (OSError, AttributeError):
        pass


def json_value(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                               allow_nan=False)+'\n')
    temp.replace(path)


def writer_main(shm_name, slot_bytes, ready, free, status, stop, parent_pid,
                write_delay_s=0.):
    """Only this process creates directories, serializes arrays or writes disk."""
    low_priority(idle=True)
    shm = SharedMemory(name=shm_name)
    manifest = directory = None
    counts = dict(written=0, bytesWritten=0, writeErrors=0, discardedAfterError=0)
    error = None
    started = time.monotonic()

    def publish(state):
        message = dict(counts, state=state, error=error,
                       directory=str(directory) if directory else None)
        try:
            status.put_nowait(message)
        except Full:
            pass

    def finish(extra=None):
        nonlocal manifest
        if manifest is not None:
            manifest.close()
            manifest = None
        if directory is not None:
            atomic_json(directory/'summary.json', dict(
                counts, error=error, state='error' if error else 'complete',
                durationS=time.monotonic()-started, producer=extra or {}))
        publish('error' if error else 'complete')

    try:
        while not stop.is_set() and os.getppid() == parent_pid:
            try:
                message = ready.get(timeout=.1)
            except Empty:
                continue
            kind = message['kind']
            if kind == 'start':
                if manifest is not None:
                    finish()
                directory = Path(message['directory'])
                counts = dict(written=0, bytesWritten=0, writeErrors=0, discardedAfterError=0)
                error = None
                started = time.monotonic()
                try:
                    # A session is immutable: never silently append to an old
                    # run after a restart or a repeated UI request.
                    directory.mkdir(parents=True, exist_ok=False)
                    (directory/'frames').mkdir()
                    metadata=dict(message['metadata'])
                    source=Path(__file__).with_name('hybrid_ball_detector.py')
                    metadata['detectorSourceSha256']=hashlib.sha256(source.read_bytes()).hexdigest()
                    assets={}
                    for key in ('depth_correction_model_path','yolo_model_path'):
                        source=Path(metadata.get('detectorKwargs',{}).get(key,'__absent__'))
                        if source.is_file():
                            (directory/'assets').mkdir(exist_ok=True)
                            dest=directory/'assets'/(key+source.suffix)
                            shutil.copyfile(source,dest)
                            assets[key]=dict(path=str(dest.relative_to(directory)),
                                sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
                    metadata['assets']=assets
                    atomic_json(directory/'metadata.json', metadata)
                    atomic_json(directory/'summary.json', {'state': 'recording'})
                    manifest = (directory/'frames.jsonl').open('w', buffering=1)
                    publish('recording')
                except Exception as exc:
                    error = str(exc)
                    manifest = None
                    publish('error')
                    directory = None  # Never touch an existing/unowned directory.
            elif kind == 'end':
                try:
                    finish(message.get('producer'))
                except Exception as exc:
                    error = str(exc)
                    publish('error')
                directory = None
            elif kind == 'frame':
                slot = message['slot']
                try:
                    if manifest is None or error:
                        counts['discardedAfterError'] += 1
                        continue
                    if write_delay_s:
                        time.sleep(write_delay_s)  # fault injection, never producer-side
                    arrays = {}
                    for name, spec in message['arrays'].items():
                        arrays[name] = np.ndarray(tuple(spec['shape']), dtype=spec['dtype'],
                            buffer=shm.buf, offset=slot*slot_bytes+spec['offset'])
                    filename = f"frames/{message['sequence']:09d}.npz"
                    dest = directory/filename
                    temp = dest.with_suffix('.npz.tmp')
                    with temp.open('wb') as stream:
                        np.savez(stream, **arrays)  # lossless; no compression CPU load
                    temp.replace(dest)
                    row = dict(message['frame'], sequence=message['sequence'], path=filename)
                    manifest.write(json.dumps(json_value(row), allow_nan=False)+'\n')
                    counts['written'] += 1
                    counts['bytesWritten'] += dest.stat().st_size
                    if counts['written'] % 10 == 0:
                        publish('recording')
                except Exception as exc:
                    counts['writeErrors'] += 1
                    error = str(exc)
                    publish('error')
                finally:
                    arrays = None
                    free.put(slot)
    finally:
        if manifest is not None:
            try:
                finish({'shutdown': True})
            except Exception:
                pass
        shm.close()


def unpack_realsense(packet):
    """Runs on the copy thread; keeps frames alive until the copy completes."""
    frames, rgb, stamp, clock_info, options, capture_count, racket_state = packet[:7]
    racket_context = packet[7] if len(packet)>7 else None
    ir = frames.get_infrared_frame(1)
    depth = frames.get_depth_frame()
    arrays = {'ir': np.asanyarray(ir.get_data())}
    if depth:
        arrays['depth'] = np.asanyarray(depth.get_data())
    if rgb is not None:
        arrays['rgb'] = rgb
    return arrays, dict(timestampS=stamp, clockSync=clock_info,
        sourceFrameNumber=int(ir.get_frame_number()), sourceTimestampMs=float(ir.get_timestamp()),
        depthFrameNumber=int(depth.get_frame_number()) if depth else None,
        depthTimestampMs=float(depth.get_timestamp()) if depth else None,
        captureCount=capture_count, runtimeOptions=json_value(options),
        racketState=json_value(racket_state), racketContext=json_value(racket_context))


class RawCameraRecorder:
    def __init__(self, *, slot_bytes, slots=4, max_duration_s=60., max_bytes=8*1024**3,
                 unpack=unpack_realsense, write_delay_s=0.):
        if not 1 <= slots <= 32 or slot_bytes <= 0 or max_duration_s <= 0 or max_bytes <= 0:
            raise ValueError('invalid raw recorder memory/time/size limits')
        self.slot_bytes = int(slot_bytes)
        self.max_duration_s = float(max_duration_s)
        self.max_bytes = int(max_bytes)
        self.unpack = unpack
        self.ctx = mp.get_context('spawn')
        self.shm = SharedMemory(create=True, size=self.slot_bytes*slots)
        self.free = self.ctx.Queue(maxsize=slots)
        self.ready = self.ctx.Queue(maxsize=slots+2)
        self.status_queue = self.ctx.Queue(maxsize=16)
        self.writer_stop = self.ctx.Event()
        self.thread_stop = threading.Event()
        self.request = None
        self.pending = None
        self.accepting = False
        self.active_session = None
        self.closed = False
        self.offered = 0
        self.stats = dict(offered=0, submitted=0, overwritten=0, poolFull=0,
                          copyErrors=0, unprocessedAtStop=0, bytesSubmitted=0, state='idle', error=None)
        self.cached_status = dict(self.stats)
        self.process = self.ctx.Process(target=writer_main,
            args=(self.shm.name,self.slot_bytes,self.ready,self.free,self.status_queue,
                  self.writer_stop,os.getpid(),write_delay_s),
            name='camera_raw_disk_writer', daemon=True)
        try:
            self.process.start()
            for slot in range(slots):
                self.free.put_nowait(slot)
            self.thread = threading.Thread(target=self._pump, name='camera_raw_copy', daemon=True)
            self.thread.start()
        except Exception:
            self.shm.close()
            self.shm.unlink()
            raise

    def set_session(self, directory, metadata=None):
        # UI/camera-management path: no mkdir, flushing, spawn or IPC waits.
        self.accepting = False
        self.active_session = None
        self.request = (str(directory), metadata) if directory else None

    def offer(self, packet):
        # CPython reference replacement only. No image conversion, copy,
        # mutex, queue, clock read, serialization, file access or waiting.
        session = self.active_session
        if session is not None:
            self.offered += 1
            self.pending = (self.offered, packet, session)

    def status(self):
        return self.cached_status

    def _send(self, message):
        while not self.thread_stop.is_set() and self.process.is_alive():
            try:
                self.ready.put(message, timeout=.05)
                return True
            except Full:
                pass
        return False

    def _pump(self):
        low_priority()
        current = None
        previous = 0
        started = 0.
        writer_status = {}
        ended_request = None
        while not self.thread_stop.wait(.002):
            while True:
                try:
                    writer_status = self.status_queue.get_nowait()
                except Empty:
                    break
            if not self.process.is_alive():
                self.accepting = False
                self.active_session = None
                self.stats.update(state='error', error='raw writer exited')
                self.cached_status = dict(self.stats, writer=writer_status)
                break
            requested = self.request
            desired = None if requested is ended_request else requested
            if current is not desired:
                self.accepting = False
                self.active_session = None
                self.pending = None
                if current is not None:
                    self.stats['offered'] = self.offered
                    self.stats['unprocessedAtStop'] += max(0,self.offered-previous)
                    if not self._send(dict(kind='end', producer=dict(self.stats))):
                        break
                current = desired
                if current is None:
                    self.stats['state'] = 'stopping' if started else 'idle'
                if current is not None and current is not ended_request:
                    self.offered = previous = 0
                    self.stats = dict(offered=0, submitted=0, overwritten=0, poolFull=0,
                        copyErrors=0, unprocessedAtStop=0, bytesSubmitted=0, state='starting', error=None)
                    if not self._send(dict(kind='start', directory=current[0], metadata=current[1])):
                        break
                    started = time.monotonic()
                    writer_status = {}
                    self.accepting = True
                    self.active_session = current
                    self.stats['state'] = 'recording'
            if current is not None and self.accepting:
                reason = None
                if time.monotonic()-started >= self.max_duration_s:
                    reason = 'duration_limit'
                elif self.stats['bytesSubmitted']+self.slot_bytes > self.max_bytes:
                    reason = 'size_limit'
                elif writer_status.get('error') and writer_status.get('directory') == current[0]:
                    reason = 'writer_error'
                if reason:
                    self.accepting = False
                    self.active_session = None
                    self.pending = None
                    self.stats.update(state=reason, error=writer_status.get('error'))
                    self.stats['offered'] = self.offered
                    self.stats['unprocessedAtStop'] += max(0,self.offered-previous)
                    self._send(dict(kind='end', producer=dict(self.stats)))
                    ended_request = current
                    current = None
            packet = self.pending
            if (self.accepting and packet is not None and packet[0] != previous
                    and packet[2] is current):
                sequence, reference, _ = packet
                self.stats['overwritten'] += max(0, sequence-previous-1)
                previous = sequence
                slot = None
                try:
                    slot = self.free.get_nowait()
                except Empty:
                    self.stats['poolFull'] += 1
                if slot is not None:
                    try:
                        arrays, info = self.unpack(reference)
                        offset = 0
                        specs = {}
                        for name, array in arrays.items():
                            array = np.asarray(array)
                            if array.dtype.hasobject:
                                raise ValueError('object arrays cannot be recorded')
                            if offset+array.nbytes > self.slot_bytes:
                                raise ValueError('frame exceeds fixed slot size')
                            target = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf,
                                                offset=slot*self.slot_bytes+offset)
                            np.copyto(target, array)
                            specs[name] = dict(shape=array.shape, dtype=array.dtype.str, offset=offset)
                            offset += array.nbytes
                        self.ready.put_nowait(dict(kind='frame',slot=slot,sequence=sequence,
                                                    arrays=specs,frame=json_value(info)))
                        self.stats['submitted'] += 1
                        self.stats['bytesSubmitted'] += offset
                    except Exception as exc:
                        self.stats['copyErrors'] += 1
                        self.stats['error'] = str(exc)
                        self.free.put_nowait(slot)
                reference = arrays = target = packet = None
            self.stats['offered'] = self.offered
            if requested is None and writer_status.get('state') in {'complete','error'}:
                self.stats['state'] = writer_status['state']
            self.cached_status = dict(self.stats, writer=writer_status,
                                      directory=requested[0] if requested else None)
        self.accepting = False
        self.active_session = None
        self.pending = None

    def close(self, timeout_s=2.):
        if self.closed:
            return
        self.closed = True
        self.set_session(None)
        deadline = time.monotonic()+timeout_s
        # Flush runs only during shutdown, never from the policy thread.
        while time.monotonic()<deadline and self.thread.is_alive():
            if (self.cached_status.get('directory') is None
                    and self.cached_status.get('writer',{}).get('state') in {'complete','error'}):
                break
            time.sleep(.01)
        self.thread_stop.set()
        self.thread.join(timeout=.3)
        self.writer_stop.set()
        self.process.join(timeout=.3)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.3)
        # A blocked writer leaves summary.state=recording, explicitly incomplete.
        self.shm.close()
        self.shm.unlink()
        for queue in (self.free,self.ready,self.status_queue):
            queue.cancel_join_thread()
            queue.close()
