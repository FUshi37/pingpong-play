"""Recording options and bounded bounce statistics, without live hardware."""

import copy
import json
import queue
import threading
from unittest.mock import Mock

import pytest

from pingpong_controller.juggle_recording import RecordingJuggleCounter
from pingpong_controller.tools import interactive_policy_safety_probe as probe


def sample(stamp, vz, *, source="camera", near=True):
    return {
        "sourceSeq": round(stamp * 200), "sourceTimeS": stamp,
        "time": stamp, "wallTimeS": 1000. + stamp,
        "ballSource": source, "cameraStampS": stamp, "rawValid": True,
        "ballClockSync": {"ready": True, "epoch": 0},
        "measuredBall": [0., 0., 1.04 if near else 1.6],
        "measuredVelocity": [0., 0., vz], "racket": [0., 0., 1.],
    }


def write_recording(path, samples, *, save_images=True, dropped=0):
    events, status = queue.Queue(), queue.Queue(maxsize=1)
    for row in samples:
        events.put({"kind": "control", "sample": row})
    events.put({"kind": "frame", "frame": b"synthetic JPEG", "info": {"seq": 1}})
    events.put({"kind": "stop", "dropped": dropped})
    events.put({"kind": "shutdown"})
    probe._recording_writer_main(str(path), events, status,
                                 {"recording": {"saveCameraImages": save_images}})
    result = status.get_nowait()
    assert result["error"] is None
    return result, json.loads((path / "summary.json").read_text())


@pytest.mark.parametrize("save_images", [False, True])
def test_writer_saves_statistics_without_full_analysis_or_images(tmp_path, save_images):
    rows = [sample(.1, -1.), sample(.11, 1.), sample(.3, -1.), sample(.31, 1.)]
    status, summary = write_recording(tmp_path, rows, save_images=save_images)
    assert summary["juggle"]["count"] == status["juggle"]["count"] == 2
    assert summary["juggle"]["status"] == "complete"
    assert summary["juggle"]["bySource"] == {"camera": 2, "mocap": 0}
    assert summary["saveCameraImages"] == save_images
    assert summary["counts"]["control"] == 4
    assert summary["counts"]["frames"] == int(save_images)
    assert len(list(tmp_path.glob("frames/*.jpg"))) == int(save_images)
    events = probe._read_jsonl(tmp_path / "events.jsonl")
    assert [row["count"] for row in events] == [1, 2]
    assert not (tmp_path / "analysis").exists()


def test_counter_deduplicates_and_enforces_refractory_period():
    counter = RecordingJuggleCounter()
    down, up = sample(.1, -1.), sample(.11, 1.)
    for row in [down, down, up, up, sample(.15, -1.), sample(.16, 1.),
                sample(.3, -1.), sample(.31, 1.)]:
        counter.consume(row)
    assert counter.count == 2
    assert counter.measurement_samples == 6


@pytest.mark.parametrize("failure", ["far", "invalid", "predicted", "gap", "source", "clock", "backward", "nan", "drag", "weak"])
def test_counter_rejects_false_contacts(failure):
    counter = RecordingJuggleCounter()
    down, up = sample(.1, -1.), sample(.11, 1.)
    if failure == "far":
        down["measuredBall"] = up["measuredBall"] = [0., 0., 1.6]
    elif failure in ("invalid", "predicted"):
        up.update(rawValid=False, valid=True, predictedBallIsPredicted=True)
    elif failure == "gap":
        up = sample(.3, 1.)
    elif failure == "source":
        up["ballSource"] = "mocap"
    elif failure == "clock":
        up["ballClockSync"]["epoch"] = 1
    elif failure == "backward":
        up["cameraStampS"] = .09
    elif failure == "nan":
        up["measuredVelocity"][2] = float("nan")
    elif failure == "drag":
        down["ballSource"] = up["ballSource"] = "drag"
    elif failure == "weak":
        down["measuredVelocity"][2], up["measuredVelocity"][2] = -.15, .15
    counter.consume(down)
    counter.consume(up)
    assert counter.count == 0


def test_sensor_cache_counts_selected_source_once_and_requires_alignment():
    counter = RecordingJuggleCounter()
    for row in [sample(.1, -1.), sample(.11, 1.)]:
        sensor = {
            "valid": True, "frameAligned": True,
            "position": row["measuredBall"], "velocity": row["measuredVelocity"],
            "measurementTimestampS": row["cameraStampS"], "clockSync": {"ready": True},
        }
        row["ballSensors"] = {"camera": sensor, "mocap": copy.deepcopy(sensor)}
        row["measuredVelocity"] = [0., 0., 0.]
        counter.consume(row)
        counter.consume(row)
    assert counter.count == 1
    assert counter.measurement_samples == 2
    sensor["frameAligned"] = False
    counter.consume(row)
    assert not counter.recent


def test_writer_reports_missing_observations_and_drops(tmp_path):
    _, empty = write_recording(tmp_path / "empty", [])
    assert empty["juggle"]["status"] == "no_valid_observations"
    _, partial = write_recording(tmp_path / "partial", [sample(.1, -1.), sample(.11, 1.)], dropped=3)
    assert partial["juggle"]["status"] == "partial"
    assert partial["juggle"]["recorderDropped"] == 3


def test_disabled_images_never_enter_recording_queue():
    recorder = probe.AsyncProbeRecorder.__new__(probe.AsyncProbeRecorder)
    recorder.save_camera_images = False
    recorder._put = Mock()
    recorder.record_camera_frame(b"JPEG", {"seq": 1})
    recorder._put.assert_not_called()


def test_launch_default_and_cli_option():
    parser = probe.build_arg_parser()
    assert probe.config_from_args(parser.parse_args([])).recording_save_camera_images
    assert not probe.config_from_args(parser.parse_args(["--no-save-camera-images"])).recording_save_camera_images


@pytest.mark.parametrize("save_images", [False, True])
def test_recorder_applies_per_run_option_to_metadata_and_frame_queue(tmp_path, monkeypatch, save_images):
    cfg = probe.config_from_args(probe.build_arg_parser().parse_args([
        "--recording-output-root", str(tmp_path)]))
    monkeypatch.setattr(probe.AsyncProbeRecorder, "_spawn_standby_writer", lambda self: None)
    recorder = probe.AsyncProbeRecorder(cfg)
    recorder.event_q = queue.Queue()
    recorder.process = Mock()
    recorder.process.is_alive.return_value = True
    recorder.start({}, save_camera_images=save_images)
    recorder.record_control_sample(sample(.1, -1.))
    recorder.record_camera_frame(b"JPEG", {"seq": 1})
    queued = []
    while not recorder.event_q.empty():
        queued.append(recorder.event_q.get_nowait())
    assert queued[0]["metadata"]["recording"]["saveCameraImages"] == save_images
    assert sum(item["kind"] == "control" for item in queued) == 1
    assert sum(item["kind"] == "frame" for item in queued) == int(save_images)


@pytest.mark.parametrize("trajectory_only", [False, True])
def test_per_recording_choice_controls_raw_images_and_preserves_defaults(tmp_path, trajectory_only):
    cfg = probe.config_from_args(probe.build_arg_parser().parse_args(["--enable-raw-camera-recording"]))
    backend = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    backend.cfg = cfg
    backend.lock = threading.RLock()
    backend.recorder = Mock(active=False)
    backend.recorder.start.return_value = {"active": True, "runDir": str(tmp_path), "error": None}
    backend.browser_config = lambda: {}
    backend.ros_bridge = Mock()
    status = backend.start_recording({"saveCameraImages": False, "trajectoryOnly": trajectory_only})
    backend.recorder.start.assert_called_once_with({}, initial_view=None, save_camera_images=False,
                                                   trajectory_only=trajectory_only)
    backend.ros_bridge.set_raw_camera_recording.assert_not_called()
    assert status["rawCamera"] == {"enabled": False, "state": "disabled"}
    assert status["trajectoryOnly"] == trajectory_only
    assert not cfg.recording_trajectory_only
    assert cfg.recording_save_camera_images  # Per-run choice never rewrites defaults.
    backend.start_recording({"saveCameraImages": True})
    backend.ros_bridge.set_raw_camera_recording.assert_called_once_with(str(tmp_path / "raw_camera"))


def test_counter_history_stays_bounded_for_long_recordings():
    counter = RecordingJuggleCounter()
    for i in range(10000):
        counter.consume(sample(.1 + i * .005, -1.))
    assert len(counter.recent) <= 32
    assert counter.count == 0
