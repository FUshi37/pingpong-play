"""Dual-source recording tests with cached synthetic observations; no devices."""
import json
import queue
import socket
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pingpong_controller.tools import interactive_policy_safety_probe as probe
from pingpong_controller.camera_process import CameraProcess
from pingpong_controller.mocap_process import MocapInputProcess
from pingpong_controller.probe_ball_recording import (
    make_sensor_recording_sample, complete_ball_sensor_recording_snapshot,
)


def sensor_bridge(*, camera_valid=True, mocap_valid=True):
    camera_stamp = 1234.123456789
    mocap_stamp = 1234.125
    camera_info = {
        "observation_state": "tracking" if camera_valid else "lost",
        "clockSync": {
            "required": True, "ready": True, "epoch": 2, "generation": 3,
            "sourceTimestampS": 12.123456789, "receiverTimestampS": 1234.13,
        },
    }
    mocap_info = {
        "clockSync": {"required": True, "ready": True, "epoch": 5},
        "sourceTimestampS": 98.125, "receiverTimestampS": 1234.135,
        "frameIdx": 42, "markerId": 1, "tracked": mocap_valid,
    }
    camera = CameraProcess.__new__(CameraProcess)
    mocap = MocapInputProcess.__new__(MocapInputProcess)
    camera._sample = {"recording": (make_sensor_recording_sample(
        "camera", [0.1, -0.5, 1.2], [0.2, 0.3, -0.4], camera_valid, camera_stamp,
        camera_info, detector_frame="base_link"), 0, 99.98)}
    mocap._sample = {"recording": (make_sensor_recording_sample(
        "mocap", [0.11, -0.51, 1.21], [0.21, 0.31, -0.41], mocap_valid, mocap_stamp,
        mocap_info), 0, 99.985)}
    for client in (camera, mocap):
        client.error = None
        client.closed = False
        client.sync_request = 0
        client.process = SimpleNamespace(is_alive=Mock(side_effect=AssertionError("process polling in control thread")))
    mocap.timeout_s = .5
    # Slow readers are deliberately forbidden: the high-rate path may only
    # use the immutable packets, even if the sensor's lock is unavailable.
    node = SimpleNamespace(
        cfg=SimpleNamespace(enable_camera_ball=True, enable_mocap_ball=True),
        now_sec=lambda: 1234.15, camera_process=camera, mocap_input=mocap,
        camera_ball_measurement_with_diagnostics=Mock(side_effect=AssertionError("sensor read in control thread")),
        mocap_ball_measurement_with_diagnostics=Mock(side_effect=AssertionError("sensor read in control thread")),
    )
    bridge = probe.RosBridge.__new__(probe.RosBridge)
    bridge.node = node
    return bridge


def collect(bridge, cfg=None):
    if bridge is None:
        return complete_ball_sensor_recording_snapshot((None, 100., None, None, True, True))
    if cfg is not None:
        bridge.node.cfg = cfg
    snapshot = bridge.ball_sensor_recording_snapshot()
    return complete_ball_sensor_recording_snapshot((snapshot[0], 100., *snapshot[2:]))


CFG = SimpleNamespace(enable_camera_ball=True, enable_mocap_ball=True)


def test_both_sources_preserve_independent_states_and_clock_precision():
    bridge = sensor_bridge()
    payload = collect(bridge, CFG)
    assert payload["bothValid"] and payload["comparisonReady"]
    camera, mocap = payload["camera"], payload["mocap"]
    np.testing.assert_allclose(camera["position"], [0.1, -0.5, 1.2])
    np.testing.assert_allclose(mocap["position"], [0.11, -0.51, 1.21])
    np.testing.assert_allclose(camera["velocity"], [0.2, 0.3, -0.4])
    np.testing.assert_allclose(mocap["velocity"], [0.21, 0.31, -0.41])
    assert camera["measurementTimestampS"] == 1234.123456789
    assert camera["sourceTimestampS"] == 12.123456789
    assert mocap["sourceTimestampS"] == 98.125
    assert mocap["measurement"]["frameIdx"] == 42
    assert camera["measurementAgeS"] == 1234.15 - 1234.123456789
    assert payload["measurementTimeDeltaS"] == 1234.123456789 - 1234.125
    camera["clockSync"]["ready"] = False
    assert bridge.node.camera_process._sample["recording"][0]["clockSync"]["ready"]
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("source", ["camera", "mocap"])
@pytest.mark.parametrize("failure", ["lost", "unsynchronized", "unavailable", "no_stamp", "expired", "resync", "clock_jump"])
def test_one_invalid_source_does_not_replace_or_hide_the_other(source, failure):
    bridge = sensor_bridge()
    client = bridge.node.camera_process if source == "camera" else bridge.node.mocap_input
    sensor = client._sample["recording"][0]
    if failure == "lost":
        sensor["valid"] = False
    elif failure == "unsynchronized":
        sensor["clockSync"]["ready"] = False
    elif failure == "unavailable":
        client.error = "sensor unavailable"
    elif failure == "no_stamp":
        sensor["measurementTimestampS"] = None
    elif failure == "expired":
        sensor["measurementTimestampS"] = 1200.
    elif failure == "resync":
        client.sync_request += 1
    else:
        sensor["receiverTimestampS"] = 0.
    payload = collect(bridge, CFG)
    failed = payload[source]
    other = payload["mocap" if source == "camera" else "camera"]
    assert not failed["valid"]
    assert failed["position"] is None and failed["velocity"] is None
    assert other["valid"] and other["position"] is not None
    assert not payload["bothValid"] and not payload["comparisonReady"]
    json.dumps(payload, allow_nan=False)


def test_disabled_or_missing_bridge_never_reads_sensors():
    bridge = sensor_bridge()
    cfg = SimpleNamespace(enable_camera_ball=True, enable_mocap_ball=False)
    payload = collect(bridge, cfg)
    assert payload["camera"]["valid"]
    assert not payload["mocap"]["enabled"] and not payload["mocap"]["available"]
    bridge.node.mocap_ball_measurement_with_diagnostics.assert_not_called()
    payload = collect(None)
    assert payload["receiverTimeS"] is None
    assert not payload["camera"]["available"] and not payload["mocap"]["available"]


def test_unaligned_camera_frame_is_not_ready_for_comparison():
    bridge = sensor_bridge()
    bridge.node.camera_process._sample["recording"][0]["frameAligned"] = False
    payload = collect(bridge, CFG)
    assert payload["bothValid"]
    assert not payload["camera"]["frameAligned"]
    assert not payload["comparisonReady"]


def test_polling_preserves_sensor_stamp_and_increases_age():
    bridge = sensor_bridge()
    old = collect(bridge, CFG)
    bridge.node.now_sec = lambda: 1234.17
    new = collect(bridge, CFG)
    for source in ("camera", "mocap"):
        assert new[source]["measurementTimestampS"] == old[source]["measurementTimestampS"]
        assert new[source]["measurementAgeS"] > old[source]["measurementAgeS"]


@pytest.mark.parametrize("source", ["camera", "mocap"])
def test_producer_rejects_nonfinite_vectors(source):
    sample = make_sensor_recording_sample(source, [np.nan, 0., 1.], [0., 0., 0.],
                                         True, 1234., {"clockSync": {"ready": True}})
    assert not sample["valid"]
    assert sample["position"] is None and sample["velocity"] is None
    json.dumps(sample, allow_nan=False)


def test_direct_camera_producer_prepares_cache_before_control_reads():
    cfg = probe.config_from_args(probe.build_arg_parser().parse_args(["--camera-ball", "--disable-camera-subprocess"]))
    source = probe.CameraSource()
    source.cfg = cfg
    source.camera_lock = threading.RLock()
    source._initialize_camera_state(cfg)
    source.camera_output_frame = "base_link"
    source.latest_camera_clock_sync = {
        "ready": True, "required": True, "generation": source.camera_clock.generation,
        "receiverTimestampS": 1234.13, "receiptMonotonicS": 99.98}
    source._store_camera_observation({"detected": True, "position": [100., -500., -100.],
        "velocity": [200., 300., -400.], "raw_measurement": {"observation_state": "tracking"}},
        1234.123456789)
    prepared = source.camera_recording_packet[0]
    assert prepared["valid"] and prepared["measurement"]["observation_state"] == "tracking"
    np.testing.assert_allclose(prepared["position"],
        [.1 + cfg.sim_cartesian_x_offset_m, -.5 + cfg.sim_cartesian_y_offset_m,
         -.1 + cfg.sim_base_height_m], atol=1e-7)
    np.testing.assert_allclose(prepared["velocity"], [.2, .3, -.4])
    source.now_sec = lambda: 1234.15
    source.camera_error = None
    bridge = probe.RosBridge.__new__(probe.RosBridge)
    bridge.node = source
    snapshot = bridge.ball_sensor_recording_snapshot()
    assert snapshot[2][0] is source.camera_recording_packet
    row = complete_ball_sensor_recording_snapshot((snapshot[0], 100., *snapshot[2:]))
    assert row["camera"]["valid"]
    source.latest_camera_clock_sync = dict(source.latest_camera_clock_sync, epoch=1, ready=False)
    snapshot = bridge.ball_sensor_recording_snapshot()
    row = complete_ball_sensor_recording_snapshot((snapshot[0], 100., *snapshot[2:]))
    assert not row["camera"]["valid"]
    assert row["camera"]["clockSync"]["status"] == "awaiting_resynchronization"


def test_writer_delay_does_not_expire_a_snapshot_that_was_fresh_at_the_tick(monkeypatch):
    bridge = sensor_bridge()
    raw = bridge.ball_sensor_recording_snapshot()
    snapshot = (raw[0], 100., *raw[2:])
    monkeypatch.setattr(probe.time, "monotonic", lambda: 1000.)
    monkeypatch.setattr(probe.time, "time", lambda: 99999.)
    payload = complete_ball_sensor_recording_snapshot(snapshot)
    assert payload["comparisonReady"]
    assert payload["camera"]["measurementAgeS"] == 1234.15 - 1234.123456789


def test_direct_mocap_receiver_updates_cache_and_fences_old_markers(monkeypatch):
    from pingpong_controller.mocap_input import MocapBallInput
    clock = [100.]
    source = MocapBallInput(enable_recording_cache=True, position_unit="m",
                           receiver_clock=lambda: 1234. + clock[0] - 100.)
    detector = source.detector
    frames = iter(range(12))

    def recvfrom(_size):
        try:
            idx = next(frames)
        except StopIteration:
            detector._running = False
            raise socket.timeout
        clock[0] = 100. + idx * .01
        data = dict(marker_frmIdx=100+idx, marker_fTimestamp=50.+idx*.01,
                    marker_1_px=.1, marker_1_py=-.5, marker_1_pz=-.2)
        return json.dumps(data).encode(), ("127.0.0.1", 1)

    detector._socket = SimpleNamespace(recvfrom=recvfrom)
    detector._running = True
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    detector._receiver_loop()
    packet = source.recording_snapshot()[0]
    assert packet[0]["valid"] and packet[0]["measurement"]["frameIdx"] == 111
    np.testing.assert_allclose(packet[0]["position"],
        [.1, -.5 + source.sim_cartesian_y_offset_m, -.2 + source.sim_base_height_m], atol=1e-7)
    source.set_marker_id(2)
    snapshot = (1234.11, clock[0], None, source.recording_snapshot(), False, True)
    row = complete_ball_sensor_recording_snapshot(snapshot)
    assert not row["mocap"]["valid"] and row["mocap"]["position"] is None
    assert row["mocap"]["clockSync"]["status"] == "awaiting_resynchronization"


@pytest.mark.parametrize("selected_source", ["camera", "mocap", "drag"])
def test_control_record_collects_both_sensors_in_every_source_mode(selected_source, monkeypatch):
    """Exercise the actual control recording entry point, including holds."""
    samples = []
    zero = np.zeros(7, dtype=np.float32)
    command = {"position": zero, "velocity": zero, "acceleration": zero}
    runtime = SimpleNamespace(
        policy_profile=probe.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
        control_mode="policy", policy_model="synthetic",
        enable_policy_dq_adapter=False, use_ros_state=False,
        publish_commands=False, enable_compensation=False,
        compensation_parameter_target="real", use_ball_estimator=False,
        estimator_hybrid_ekf9=False, estimator_prediction=False,
        estimator_output_mode="continuous", high_ball_occlusion_prediction=False,
        enable_ball_release_gate=False, ball_release_use_estimator_state=False,
        ball_observation_rate_hz=60., policy_ball_handover_mode="legacy_force_fresh",
        workspace_safety_mode="position_soft", policy_observation_scales=[1.] * 67,
    )
    bridge = sensor_bridge()
    # Neither sensor formatting nor completion may happen on this thread.
    monkeypatch.setattr(probe, "make_sensor_recording_sample", Mock(side_effect=AssertionError("formatting in control thread")))
    monkeypatch.setattr(probe, "complete_ball_sensor_recording_snapshot", Mock(side_effect=AssertionError("completion in control thread")))
    instance = SimpleNamespace(
        recorder=SimpleNamespace(active=True, record_control_sample=samples.append),
        controller=SimpleNamespace(
            latest_command_state=lambda: command,
            latest_obs_and_action=lambda: {
                "obs": np.zeros(67, dtype=np.float32), "action": zero}),
        cfg=SimpleNamespace(enable_camera_ball=True, enable_mocap_ball=True,
                            done_ball_required_consecutive_ticks=5, policy_dq_mode="raw_coherent"),
        ros_bridge=bridge, now_s=.1, control_loop_tick=20,
        last_policy_execution_trace=None, last_estimator_info={},
        last_high_ball_occlusion_info={}, last_done_ball_pos=np.ones(3),
        last_done_ball_valid=False, ball_done_raw_conditions={}, ball_done_streaks={},
        last_policy_dq_adapter_info={}, last_contact_action_shield_info={},
        last_policy_plant_matcher_info={}, last_policy_ball_handover_info={},
        last_release_gate_info={}, active_policy_profile=runtime.policy_profile,
        active_policy_model="synthetic", policy_generation=1,
        reset_initial_pose_id="default_2_horizontal",
        _workspace_safety_pose_payload=lambda: {},
        _hard_workspace_stop_payload=lambda: {},
        _soft_workspace_reset_payload=lambda: {},
        _reset_recovery_payload=lambda: {},
        _real_compensation_options=lambda options: options,
        _compensation_payload=lambda options: {},
    )
    probe.PolicySafetyProbe._record_control_sample(
        instance, source_seq=20, source_time_s=.1, ball_observation_seq=3,
        camera_stamp_s=1234.0, camera_measurement={}, mode="hold",
        policy_executed=False, reason="idle", conditions={},
        raw_ball_pos=np.zeros(3), measured_ball_pos=np.zeros(3),
        measured_ball_vel=np.zeros(3), observed_ball_pos=np.zeros(3),
        observed_ball_vel=np.zeros(3), raw_valid=False, control_valid=False,
        age_s=0., noise_std_m=0., options=runtime, ball_source=selected_source,
        command_published=False,
    )
    assert len(samples) == 1
    sample = samples[0]
    assert sample["ballSource"] == selected_source
    assert not sample["rawValid"] and not sample["policyExecuted"]
    # The independent sensor data remains valid even when source gating
    # hides the ball from the actor or the robot stays in hold.
    snapshot = sample["_ballSensorSnapshot"]
    assert "ballSensors" not in sample
    assert snapshot[2][0] is bridge.node.camera_process._sample["recording"]
    assert snapshot[3][0] is bridge.node.mocap_input._sample["recording"]
    bridge.node.camera_ball_measurement_with_diagnostics.assert_not_called()
    bridge.node.mocap_ball_measurement_with_diagnostics.assert_not_called()


def test_writer_persists_both_sources_with_control_alignment(tmp_path):
    events, status = queue.Queue(), queue.Queue(maxsize=1)
    raw_snapshot = sensor_bridge().ball_sensor_recording_snapshot()
    snapshot = (raw_snapshot[0], 100., *raw_snapshot[2:])
    for seq, source in enumerate(("camera", "mocap", "drag")):
        events.put({"kind": "control", "sample": {
            "schemaVersion": probe.RECORDING_SCHEMA_VERSION,
            "sourceSeq": seq, "sourceTimeS": seq * .005,
            "tick": seq, "wallTimeS": 2000.0 + seq * .005,
            "time": seq * .005, "ballSource": source, "_ballSensorSnapshot": snapshot,
        }})
    events.put({"kind": "stop", "dropped": 0})
    events.put({"kind": "shutdown"})
    probe._recording_writer_main(str(tmp_path), events, status, {})
    controls = probe._read_jsonl(tmp_path / "control_samples.jsonl")
    sensors = probe._read_jsonl(tmp_path / "ball_sensor_samples.jsonl")
    assert len(controls) == len(sensors) == 3
    for control, row in zip(controls, sensors):
        assert row["sourceSeq"] == control["sourceSeq"]
        assert row["sourceTimeS"] == control["sourceTimeS"]
        assert row["ballSource"] == control["ballSource"]
        assert row["camera"] == control["ballSensors"]["camera"]
        assert row["mocap"] == control["ballSensors"]["mocap"]
        assert row["comparisonReady"]
        assert "_ballSensorSnapshot" not in control
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["counts"]["ballSensors"] == summary["counts"]["control"] == 3
    assert summary["dropped"] == 0
    assert status.get_nowait()["error"] is None


def test_full_recording_queue_drops_sample_without_waiting():
    class FullQueue:
        def put(self, item, block):
            assert block is False
            raise queue.Full
    recorder = probe.AsyncProbeRecorder.__new__(probe.AsyncProbeRecorder)
    recorder.cfg = SimpleNamespace(recording_control_rate_hz=200., control_rate_hz=200.)
    recorder.lock = threading.Lock()
    recorder.active = True
    recorder.frozen_source_seq = None
    recorder.event_q = FullQueue()
    recorder.dropped = 0
    recorder.record_control_sample({"sourceSeq": 1, "_ballSensorSnapshot": sensor_bridge().ball_sensor_recording_snapshot()})
    assert recorder.dropped == 1
