"""Trajectory-only recording, bounce parity and bounded offline analysis."""

import copy
import csv
import json
import os
import queue
import re
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pingpong_controller.ball_trajectory_recording import (
    compact_trajectory_sample, export_ball_trajectory,
)
from pingpong_controller.juggle_recording import RecordingJuggleCounter
from pingpong_controller.tools import interactive_policy_safety_probe as probe
from test_probe_recording_options import sample


@pytest.mark.parametrize("images", [False, True])
def test_trajectory_writer_omits_other_streams_and_counts_bounces(tmp_path, images):
    events, status = queue.Queue(), queue.Queue(maxsize=1)
    rows = [sample(.1, -1.), sample(.11, 1.), sample(.3, -1.), sample(.31, 1.)]
    for row in rows:
        row.update(observedBall=row["measuredBall"], observedVelocity=row["measuredVelocity"],
                   policyObs=[42] * 117, policyAction=[42] * 7,
                   command={"position": [42] * 7}, runtime={"diagnostics": "large"})
        events.put(dict(kind="control", sample=row))
    for kind, field in [("snapshot", "snapshot"), ("view", "view"), ("sim_control_batch", "samples")]:
        events.put(dict(kind=kind, **{field: rows}))
    for kind in ("frame", "sim_frame"):
        events.put(dict(kind=kind, frame=b"JPEG", info={"seq": 1}))
    events.put(dict(kind="stop", dropped=0))
    events.put(dict(kind="shutdown"))
    probe._recording_writer_main(str(tmp_path), events, status,
                                 {"recording": {"trajectoryOnly": True, "saveCameraImages": images}})
    assert status.get_nowait()["error"] is None
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["trajectoryOnly"] and summary["juggle"]["count"] == 2
    assert summary["counts"]["control"] == 4
    assert summary["counts"]["frames"] == int(images)
    assert all(summary["counts"][name] == 0 for name in ("simControl", "simFrames", "snapshots", "views"))
    assert not any((tmp_path / name).exists() for name in (
        "ball_sensor_samples.jsonl", "sim_control_samples.jsonl", "snapshots.jsonl", "browser_view.jsonl", "sim_frames"))
    controls = probe._read_jsonl(tmp_path / "control_samples.jsonl")
    assert all("policyObs" not in row and "command" not in row and "runtime" not in row for row in controls)
    assert len(probe._read_jsonl(tmp_path / "events.jsonl")) == 2
    assert rows[0]["policyObs"] == [42] * 117  # No mutation of shared samples.


def test_sensor_compaction_preserves_counting_and_clock_boundaries():
    full, compact = RecordingJuggleCounter(), RecordingJuggleCounter()
    rows = [sample(.1, -1.), sample(.11, 1.), sample(.3, -1.), sample(.31, 1.),
            sample(.5, -1.), sample(.51, 1.)]
    for index, row in enumerate(rows):
        row["ballSensors"] = {"receiverTimeS": 1000. + row["time"], "camera": {
            "valid": True, "frameAligned": True, "position": row["measuredBall"],
            "velocity": row["measuredVelocity"], "measurementTimestampS": row["cameraStampS"],
            "clockSync": {"ready": True, "epoch": int(index == 5), "diagnostics": [42] * 100},
            "measurement": {"markerId": 2, "unrelated": [42] * 100}}}
        before = copy.deepcopy(row)
        slim = compact_trajectory_sample(row)
        assert full.consume(row) == compact.consume(slim)
        assert row == before
        assert "diagnostics" not in slim["ballSensors"]["camera"]["clockSync"]
        assert "unrelated" not in slim["ballSensors"]["camera"]["measurement"]
    assert compact.count == full.count == 2


def test_recorder_skips_non_ball_queue_messages_and_preserves_per_run_defaults(tmp_path, monkeypatch):
    cfg = probe.config_from_args(probe.build_arg_parser().parse_args(["--recording-output-root", str(tmp_path)]))
    monkeypatch.setattr(probe.AsyncProbeRecorder, "_spawn_standby_writer", lambda self: None)
    recorder = probe.AsyncProbeRecorder(cfg)
    recorder.event_q, recorder.process = queue.Queue(), Mock()
    recorder.process.is_alive.return_value = True
    recorder.start({"largePolicyRegistry": [42] * 1000}, initial_view={"obs": [42] * 117},
                   trajectory_only=True, save_camera_images=False)
    recorder.record_control_sample(dict(sample(.1, -1.), policyObs=[42] * 117))
    recorder.record_snapshot({"obs": [42] * 117})
    recorder.record_view({"obs": [42] * 117})
    recorder.record_sim_control_samples([{"obs": [42] * 117}])
    recorder.record_sim_frame(b"JPEG", {"seq": 1})
    recorder.record_camera_frame(b"JPEG", {"seq": 1})
    queued = list(recorder.event_q.queue)
    assert [item["kind"] for item in queued] == ["configure", "event", "control"]
    assert queued[0]["metadata"]["recording"]["trajectoryOnly"]
    assert queued[0]["metadata"]["recording"]["ballSensorFile"] == "control_samples.jsonl"
    assert "largePolicyRegistry" not in queued[0]["metadata"]["config"]
    assert "policyObs" not in queued[-1]["sample"]
    assert not cfg.recording_trajectory_only and cfg.recording_save_camera_images
    recorder.active = False
    recorder.start({})
    assert not recorder.trajectory_only and recorder.save_camera_images


def test_actual_control_entry_never_builds_full_policy_telemetry():
    rows = []
    controller = SimpleNamespace(last_obs=np.zeros(72),
                                 latest_obs_and_action=Mock(side_effect=AssertionError("full observation requested")),
                                 latest_command_state=Mock(side_effect=AssertionError("joint command requested")))
    controller.last_obs[52:55] = [0., 0., 1.]
    backend = SimpleNamespace(recorder=SimpleNamespace(active=True, trajectory_only=True, record_control_sample=rows.append),
                              controller=controller, ros_bridge=None, now_s=.1, control_loop_tick=20,
                              cfg=SimpleNamespace(enable_camera_ball=False, enable_mocap_ball=False))
    probe.PolicySafetyProbe._record_control_sample(backend, source_seq=20, source_time_s=.1,
        ball_observation_seq=1, camera_stamp_s=.1, camera_measurement={"clockSync": {"ready": True}},
        mode="policy", policy_executed=False, reason="", conditions={},
        raw_ball_pos=np.array([0., 0., 1.04]), measured_ball_pos=np.array([0., 0., 1.04]),
        measured_ball_vel=np.array([0., 0., -1.]), observed_ball_pos=np.array([0., 0., 1.04]),
        observed_ball_vel=np.array([0., 0., -1.]), raw_valid=True, control_valid=True,
        age_s=0., noise_std_m=0., options=None, ball_source="camera", command_published=False)
    assert len(rows) == 1 and rows[0]["racket"] == [0., 0., 1.]
    assert "policyObs" not in rows[0] and "command" not in rows[0]


def test_streaming_export_keeps_all_csv_rows_with_bounded_preview(tmp_path):
    with (tmp_path / "control_samples.jsonl").open("w") as handle:
        for index in range(12000):
            row = sample(index * .005, -.5)
            row["ballSensors"] = {"camera": {"valid": False, "position": [9, 9, 9]}, "mocap": {}}
            handle.write(json.dumps(row) + "\n")
        handle.write('{"interrupted":')
    samples, report = export_ball_trajectory(tmp_path, tmp_path, max_plot_samples=31)
    assert report["sampleCount"] == 12000 and len(samples) <= 31
    assert samples[0]["sourceSeq"] == 0 and samples[-1]["sourceSeq"] == 11999
    with (tmp_path / "ball_trajectory.csv").open() as handle:
        rows = csv.DictReader(handle)
        assert all(row["camera_x_m"] == "" for row in rows)
        assert rows.line_num == 12001


@pytest.mark.parametrize("comparison_only", [False, True])
def test_trajectory_analysis_generates_plots_without_loading_whole_logs(tmp_path, monkeypatch, comparison_only):
    (tmp_path / "metadata.json").write_text(json.dumps({"recording": {"trajectoryOnly": True}, "config": {}}))
    (tmp_path / "summary.json").write_text(json.dumps({"juggle": {"count": 7}}))
    rows = [dict(sample(i * .005, -1.), observedBall=[0., 0., 1.04], observedVelocity=[0., 0., -1.]) for i in range(6)]
    (tmp_path / "control_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(probe, "_read_jsonl", Mock(side_effect=AssertionError("unbounded JSON loading")))
    status = queue.Queue(maxsize=1)
    probe._recording_analysis_main(str(tmp_path), status, 60., ball_sensors_only=comparison_only)
    result = status.get_nowait()
    assert result["phase"] == "done", result
    analysis = tmp_path / "analysis"
    assert len(list(analysis.glob("ball_*_curve.png"))) == 6
    data = json.loads((analysis / "analysis_data.json").read_text())
    assert data["metadata"]["sampleCount"] == 6 and data["metadata"]["trajectoryOnly"]
    assert data["juggle"]["count"] == 7 and not data["obsNames"]


def test_trajectory_launch_cli_and_invalid_browser_payload():
    parser = probe.build_arg_parser()
    assert not probe.config_from_args(parser.parse_args([])).recording_trajectory_only
    assert probe.config_from_args(parser.parse_args(["--recording-trajectory-only"])).recording_trajectory_only
    backend = SimpleNamespace(cfg=probe.config_from_args(parser.parse_args([])), recorder=SimpleNamespace(active=False))
    with pytest.raises(ValueError, match="trajectoryOnly must be a boolean"):
        probe.PolicySafetyProbe.start_recording(backend, {"trajectoryOnly": "false"})


@pytest.mark.parametrize("analyze", [False, True])
def test_stop_trajectory_recording_skips_simulation_flush_and_still_analyzes(tmp_path, analyze):
    backend = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    backend.cfg = probe.config_from_args(probe.build_arg_parser().parse_args(["--no-save-camera-images"]))
    backend.control_loop_tick = 20
    backend.ros_bridge = None
    backend.sim_mirror = probe.SimMirrorSubprocess.__new__(probe.SimMirrorSubprocess)
    backend.sim_mirror.flush = Mock(side_effect=AssertionError("unneeded simulation flush"))
    backend.recorder = Mock(trajectory_only=True)
    status = {"runDir": str(tmp_path), "error": None}
    backend.recorder.stop.return_value = dict(status)
    backend.recorder.start_analysis.return_value = dict(status)
    result = backend.stop_recording({"analyze": analyze})
    backend.recorder.freeze.assert_called_once_with(19)
    backend.sim_mirror.flush.assert_not_called()
    backend.recorder.stop.assert_called_once_with(final_view=None, analyze=analyze)
    if not analyze:
        backend.recorder.start_analysis.assert_called_once_with(str(tmp_path), ball_sensors_only=True)
    assert not result.get("error")


def test_launcher_exposes_trajectory_default_without_devices():
    env = {**os.environ, "PROBE_CONFIG_ONLY": "true"}
    script = probe.REPO_ROOT / "RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh"
    run = subprocess.run(["bash", str(script), "--recording-trajectory-only", "--no-save-camera-images"],
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "recording trajectory_only=true" in run.stdout and "recording save_camera_images=false" in run.stdout


def test_frontend_trajectory_switch_preserves_independent_image_choice(tmp_path):
    function = re.search(r"function syncRecordingModeControls\(\) \{.*?\n\}", probe.HTML_TEMPLATE, re.S).group()
    script = tmp_path / "mode_controls.js"
    script.write_text("""
const analyzeAfterRecordEl = {checked:true};
const recordingTrajectoryOnlyEl = {checked:true};
const recordingButtonEl = {};
const saveCameraImagesEl = {checked:false};
let recordingActive = false;
""" + function + """
syncRecordingModeControls();
if (!analyzeAfterRecordEl.disabled || !recordingButtonEl.textContent.includes('仅球轨迹')) process.exit(1);
saveCameraImagesEl.checked = true;
syncRecordingModeControls();
if (!recordingButtonEl.textContent.includes('图像') || !saveCameraImagesEl.checked) process.exit(2);
recordingTrajectoryOnlyEl.checked = false;
syncRecordingModeControls();
if (analyzeAfterRecordEl.disabled || !saveCameraImagesEl.checked) process.exit(3);
""")
    subprocess.run(["node", str(script)], capture_output=True, check=True)
