"""Opt-in wall/CPU measurements; no policy or actuator behavior changes.

Set PROBE_STAGE_TIMINGS_PATH to a JSON output path before launching. Samples
are bounded, kept in memory, and written only after the control loop stops.
Nested stages overlap and must not be added together.
"""
from collections import defaultdict, deque
from functools import wraps
import json
from pathlib import Path
import time

import numpy as np


class ProbeStageTimings:
    def __init__(self, probe, output_path):
        self.probe = probe
        self.output_path = Path(output_path)
        self.samples = defaultdict(lambda: deque(maxlen=16000))
        self.last_start = None
        for name in ('_run_backend_control_tick', '_run_control_tick_locked',
                     '_run_one_control_step', '_predict_policy_with_feedback',
                     '_publish_control_command', '_step_sim_mirror',
                     '_record_control_sample', '_workspace_safety_racket_pose',
                     '_joint_feedback_for_policy', 'snapshot', 'step'):
            self._wrap(probe, name)
        self._wrap(probe.controller, 'predict', 'actor_and_projection')
        self._wrap(probe.controller, '_pose', 'fplc_fk_and_jacobian')
        self._wrap(probe.controller, '_policy_action', 'network')
        self._wrap(probe.controller, 'pose_is_safe', 'physical_pose_check')
        bridge_node = getattr(getattr(probe, 'ros_bridge', None), 'node', None)
        if bridge_node is not None:
            self._wrap(bridge_node, 'publish_right_arm_command', 'ros_command_publish')
        self._wrap(probe.recorder, 'record_snapshot', 'record_snapshot')

    def _wrap(self, target, name, label=None):
        original = getattr(target, name, None)
        if not callable(original):
            return
        label = label or name

        @wraps(original)
        def measured(*args, **kwargs):
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            interval = None
            if name == '_run_backend_control_tick':
                if self.last_start is not None:
                    interval = (start - self.last_start) / 1e6
                self.last_start = start
            try:
                return original(*args, **kwargs)
            finally:
                wall_ms = (time.perf_counter_ns() - start) / 1e6
                cpu_ms = (time.thread_time_ns() - cpu) / 1e6
                mode = str(self.probe.last_control_mode or 'initializing')
                if self.probe.command_stream_active:
                    mode += ':send'
                self.samples[(label, mode)].append((wall_ms, cpu_ms))
                if interval is not None:
                    self.samples[('control_start_interval', mode)].append((interval, 0.0))

        setattr(target, name, measured)

    def write(self):
        result = {}
        for (stage, mode), values in list(self.samples.items()):
            array = np.asarray(list(values))
            result[f'{mode}/{stage}'] = dict(
                count=len(array), wallMeanMs=float(array[:, 0].mean()),
                wallMedianMs=float(np.median(array[:, 0])),
                wallP95Ms=float(np.percentile(array[:, 0], 95)),
                wallP99Ms=float(np.percentile(array[:, 0], 99)),
                wallMaxMs=float(array[:, 0].max()),
                cpuMeanMs=float(array[:, 1].mean()),
                cpuMaxMs=float(array[:, 1].max()),
                cpuAtMaximumWallMs=float(array[np.argmax(array[:, 0]), 1]))
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2) + '\n')
