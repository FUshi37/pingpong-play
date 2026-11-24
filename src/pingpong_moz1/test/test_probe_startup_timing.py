"""Idle mirror preparation, cheap recording, and persistent timing-stop evidence."""

import dataclasses
import json
import pickle
import queue
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.recording_payload import (
    RecordingVector, recording_camera_measurement, finish_recording_payload,
)
from pingpong_controller.tools import interactive_policy_safety_probe as probe
from test_simcode9_fplc_contract import probe as fplc_probe, _policy_options


@pytest.mark.parametrize("digits", [3, 6, 9])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_recorded_vectors_preserve_legacy_rounding_and_freeze_queue_inputs(digits, dtype):
    source = np.array([1.2345675, -.0000005, .0100005, 7.123456789], dtype=dtype)
    expected = probe._rounded_vector(source, digits)
    encoded = RecordingVector(source, digits)
    source[:] = 99
    decoded = pickle.loads(pickle.dumps(encoded))
    assert finish_recording_payload({"nested": [decoded]}, probe._camera_measurement_payload) == {"nested": [expected]}
    assert not encoded.values.flags.writeable


def test_camera_scalar_formatting_is_deferred_and_matches_legacy():
    source = dict(cx=np.float32(123.4567), observation_position_variance=1.23456789e-7,
                  depth_mm=float("nan"), velocity_ready=True)
    expected = probe._camera_measurement_payload(source)
    encoded = recording_camera_measurement(source)
    source["cx"] = 9
    assert finish_recording_payload(encoded, probe._camera_measurement_payload) == expected


def test_camera_formatting_does_not_enqueue_raw_images_or_nested_diagnostics():
    raw = dict(cx=4.5, image=np.ones((10,10)), udp={"large":[0]*100},
               observation_position_variance=np.ones(3))
    encoded = recording_camera_measurement(raw, probe._CAMERA_MEASUREMENT_SCALAR_KEY_SET)
    assert "image" not in encoded.values and "udp" not in encoded.values
    assert finish_recording_payload(encoded, probe._camera_measurement_payload) == probe._camera_measurement_payload(raw)


def test_writer_finishes_vectors_before_json_and_bounce_counting(tmp_path):
    events, status = queue.Queue(), queue.Queue(maxsize=1)
    for stamp, vz in ((.1, -1.), (.11, 1.)):
        events.put(dict(kind="control", sample=dict(
            sourceSeq=round(stamp*200), sourceTimeS=stamp, time=stamp, wallTimeS=1000.+stamp,
            ballSource="mocap", rawValid=True, cameraStampS=stamp,
            ballClockSync=dict(ready=True,epoch=0),
            measuredBall=RecordingVector([0.,0.,1.04]),
            measuredVelocity=RecordingVector([0.,0.,vz]), racket=RecordingVector([0.,0.,1.]),
            policyObs=RecordingVector(np.arange(72)/1000.),
        )))
    events.put(dict(kind="stop",dropped=0))
    events.put(dict(kind="shutdown"))
    probe._recording_writer_main(str(tmp_path), events, status, {"recording":{"saveCameraImages":False}})
    assert status.get_nowait()["error"] is None
    summary = json.loads((tmp_path/"summary.json").read_text())
    assert summary["juggle"]["count"] == 1
    rows = [json.loads(line) for line in (tmp_path/"control_samples.jsonl").open()]
    assert rows[0]["policyObs"] == probe._rounded_vector(np.arange(72)/1000.)


def test_mirror_switch_is_prepared_during_release_gate_and_real_reset_is_preserved(monkeypatch):
    calls = []
    class Mirror:
        policy_profile = "fplc_gpu1_v172"
        policy_model = "fplc_gpu1_v172"
        policy_generation = 0
        def reset(self, **kw):
            calls.append(("reset", kw))
            self.policy_profile = kw["policy_profile"]
            self.policy_model = kw["policy_model"]
            self.policy_generation = kw["policy_generation"]
            return self.snapshot()
        def snapshot(self):
            return dict(available=True,sync=dict(ready=True))
        def drain_completed_samples(self):
            return []
        def step(self, *args, **kw):
            calls.append(("step",kw))
            return self.snapshot()
    monkeypatch.setattr(probe, "SimMirrorSubprocess", Mirror)
    p = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    p.cfg = SimpleNamespace(enable_sim_mirror=True)
    p.sim_mirror = Mirror()
    p.sim_mirror_error = None
    p.sim_needs_policy_reset = True
    p.policy_reentry_reset_q = np.ones(7)
    p.active_policy_profile = p.active_policy_model = "fplc_v321_s4_u2097_20261029_r1"
    p.policy_generation = 1
    p.recorder = SimpleNamespace(record_sim_control_samples=lambda _: None)
    p._sim_compensation_options = lambda options: options
    p._reset_policy_dq_adapter = lambda reason: None
    def step(active, seq):
        return p._step_sim_mirror(np.array([0.,-.49,1.04]), np.zeros(3),ball_valid=True,age_s=.001,
            options=SimpleNamespace(enable_sim_mirror=True),source_seq=seq,source_time_s=seq*.005,
            ball_observation_seq=seq,camera_stamp_s=1.,control_mode="policy" if active else "release_gate",
            policy_active=active)
    for seq in range(10):
        step(False,seq)
    assert len(calls) == 1 and calls[0][0] == "reset"
    assert calls[0][1]["policy_profile"] == p.active_policy_profile
    assert p.sim_needs_policy_reset
    p.policy_reentry_reset_q = np.full(7, .25)
    step(True,10)
    assert [kind for kind,_ in calls] == ["reset","reset","step"]
    np.testing.assert_array_equal(calls[1][1]["arm_q"],np.full(7,.25))
    assert not p.sim_needs_policy_reset
    step(False,11)
    assert len(calls) == 3  # A later hold does not restart the mirror episode.


def test_timing_stop_blocks_dry_actor_and_preserves_reason_when_ball_falls(fplc_probe, monkeypatch):
    p = fplc_probe
    options = dataclasses.replace(_policy_options(p),publish_commands=False)
    p.runtime_options = options
    p._require_publish_rearm("policy_control_timing_fault",details=dict(windowHz=183.9035))
    monkeypatch.setattr(p, "_predict_policy_with_feedback", lambda *a,**kw: pytest.fail("actor ran after timing stop"))
    monkeypatch.setattr(p, "_workspace_safety_racket_pose", lambda options: (np.array([0.,-.49,1.05]),None,True))
    p.current_policy_joint_sample = object()  # A previous actor sample is not a new hold-tick sample.
    for ball in ([0.,-.49,1.15],[.3,-.1,.1]):
        mode, reason, _, _, executed = p._run_one_control_step(np.array(ball),np.zeros(3),True,.001,options)
        assert mode == "policy_timing_hold" and not executed
        assert "policy_control_timing_fault" in reason
        assert p.publish_rearm_reason == "policy_control_timing_fault"
        assert p.publish_rearm_details == dict(windowHz=183.9035)
        assert p.last_policy_execution_trace is None
        assert p.current_policy_joint_sample is None
    # Operator reset has priority over the stop and can still run its own
    # supervised recovery, rather than requiring a whole process restart.
    monkeypatch.setattr(p, "_reset_recovery_active", lambda: True)
    mode, *_ = p._run_one_control_step(np.array([0.,-.49,1.05]),np.zeros(3),True,.001,options)
    assert mode == "reset"


def test_timing_stop_workspace_preflight_uses_physical_feedback_even_after_send_off():
    p = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    p.publish_rearm_required = True
    p.publish_rearm_reason = "policy_control_timing_fault"
    command = np.array([0.,-.2,1.05])
    physical = np.array([0.,-.49,1.05])
    p._racket_pose = lambda: (command,np.array([0.,0.,0.,1.]),True)
    p.controller = SimpleNamespace(_compute_racket_pose_quat_base=lambda q: (physical,np.array([0.,0.,0.,1.]),True))
    p.ros_bridge = SimpleNamespace(get_joint_feedback_with_age=lambda: (np.zeros(7),np.zeros(7),.001,"joint_state"))
    position, _, valid = p._workspace_safety_racket_pose(SimpleNamespace(publish_commands=False))
    np.testing.assert_allclose(position,physical,rtol=0,atol=1e-7)
    assert valid and p.last_workspace_safety_pose_source == "joint_state_fk"


def test_trajectory_mode_keeps_the_first_timing_stop_event():
    recorder = probe.AsyncProbeRecorder.__new__(probe.AsyncProbeRecorder)
    recorder.trajectory_only = True
    sent = []
    recorder._put = sent.append
    event = dict(event="control_safety_stop",reason="policy_control_timing_fault",details=dict(windowHz=183.9))
    recorder.record_event(event)
    assert sent == [dict(kind="event",event=event)]


def test_sim_reset_never_waits_for_a_full_command_queue():
    class FullQueue:
        def get_nowait(self):
            raise queue.Empty
        def put_nowait(self, item):
            raise queue.Full
        def put(self, *args, **kw):
            pytest.fail("sim reset blocked the control thread")
    mirror = probe.SimMirrorSubprocess.__new__(probe.SimMirrorSubprocess)
    mirror.lock = threading.Lock()
    mirror.sim_cfg = SimpleNamespace()
    mirror.policy_profile = mirror.policy_model = "fplc_gpu1_v172"
    mirror.policy_generation = mirror.episode = 0
    mirror.completed_samples = deque()
    mirror.command_q = FullQueue()
    mirror.snapshot = lambda: dict(valid=mirror.stream_valid)
    assert mirror.reset(arm_q=np.zeros(7)) == dict(valid=False)
    assert mirror.command_drops == 1


def test_deferred_diagnostic_snapshot_waits_for_its_next_30hz_slot(monkeypatch):
    p = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    p.lock = threading.Lock()
    p.runtime_options = SimpleNamespace(publish_commands=False)
    p.active_policy_profile = "fplc_gpu1_v172"
    p.policy_tick_active = False
    p.policy_deadline_monitor = SimpleNamespace(start=lambda *args: None)
    p.web_drag_ball_pos = p.web_drag_ball_vel = np.zeros(3)
    p.web_drag_valid = True
    p.web_noise_std_m = 0.
    p.ros_bridge = None
    p.backend_next_snapshot_wall = 10.
    p.backend_snapshot_period_s = 1/30
    attempted = []
    p._run_control_tick_locked = lambda *args, **kw: attempted.append(kw['make_snapshot'])
    # None means the control budget deferred expensive snapshot packaging.
    # The next six policy ticks must not repeat diagnostic preparation.
    clock = [10.]
    monkeypatch.setattr(probe.time, "perf_counter", lambda: clock[0])
    for tick in range(9):
        clock[0] = 10.+tick*.005
        p._run_backend_control_tick()
    assert attempted == [True, False, False, False, False, False, False, True, False]

