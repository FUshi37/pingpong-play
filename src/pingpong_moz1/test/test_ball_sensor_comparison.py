"""Timestamp matching and post-record plot generation without devices."""
import json
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pingpong_controller.ball_sensor_comparison import (
    collect_observations, align_positions, analyze_ball_sensor_comparison,
)
from pingpong_controller.tools import interactive_policy_safety_probe as probe


def sample(stamp, bias=(0., 0., 0.)):
    return dict(valid=True, frameAligned=True, clockSync=dict(ready=True, epoch=0),
                measurementTimestampS=stamp,
                position=(np.array([stamp, 2. * stamp, -stamp]) + bias).tolist())


def rows():
    return [dict(receiverTimeS=100., time=i*.01, sourceSeq=i,
                 camera=sample(i*.01+.005, (.01, -.02, .005)), mocap=sample(i*.01)) for i in range(4)]


def test_differences_use_capture_timestamps_and_never_extrapolate():
    observations, report = collect_observations(rows())
    matched = align_positions(observations)
    assert len(matched) == 3
    np.testing.assert_allclose([r["difference"] for r in matched], [[.01, -.02, .005]] * 3)
    assert report["sources"]["camera"]["validRows"] == 4


def test_cached_captures_are_not_counted_as_new_measurements():
    data = rows()
    data[1]["camera"] = data[0]["camera"]
    observations, _ = collect_observations(data)
    assert len(observations["camera"]) == 3
    assert len(align_positions(observations)) == 2


def test_camera_calibration_changes_comparison_time_without_rewriting_capture_stamp():
    data=rows()
    for row in data:
        stamp=row['camera']['measurementTimestampS']
        row['camera']=sample(stamp+.004)
        row['camera']['measurementTimestampS']=stamp
        row['camera']['measurement']=dict(ball_calibration_applied=True,ball_calibration_time_shift_s=.004)
    observations,_=collect_observations(data)
    matched=align_positions(observations)
    assert len(matched)==3
    np.testing.assert_allclose([r['difference'] for r in matched],np.zeros((3,3)),atol=1e-12)
    assert data[0]['camera']['measurementTimestampS']==.005


def test_mocap_loss_and_clock_reset_prevent_interpolation_across_boundary():
    for failure in ("lost", "epoch"):
        data = rows()
        if failure == "lost":
            data[1]["mocap"]["valid"] = False
        else:
            for row in data[1:]:
                row["mocap"]["clockSync"]["epoch"] = 1
        observations, _ = collect_observations(data)
        matched = align_positions(observations)
        assert all(r["stamp"] != data[0]["camera"]["measurementTimestampS"] for r in matched)
        if failure == "lost":
            assert all(r["stamp"] != data[1]["camera"]["measurementTimestampS"] for r in matched)


def test_sparse_mocap_gap_is_excluded_even_when_rows_remain_valid():
    data = rows()
    data[1]["mocap"] = data[0]["mocap"]
    data[2]["mocap"] = sample(.2)
    data[3]["mocap"] = sample(.21)
    observations, _ = collect_observations(data)
    assert not align_positions(observations, max_gap_s=.05)


def test_unknown_camera_frame_is_not_compared():
    data = rows()
    for row in data:
        row["camera"]["frameAligned"] = False
    observations, report = collect_observations(data)
    assert not observations["camera"] and not align_positions(observations)
    assert report["sources"]["camera"]["rejectedFrameRows"] == 4


def test_comparison_plots_csv_and_statistics_are_generated(tmp_path):
    data = rows()
    (tmp_path / "ball_sensor_samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in data))
    report = analyze_ball_sensor_comparison(tmp_path, control_samples=[dict(sourceSeq=i, mode="policy") for i in range(4)])
    assert report["matchedCaptures"] == report["policy"]["matchedCaptures"] == 3
    np.testing.assert_allclose(report["biasMm"], [10., -20., 5.])
    np.testing.assert_allclose(report["rmseMm"], [10., 20., 5.])
    np.testing.assert_allclose(report["positionRmseMm"], np.sqrt(525.))
    for filename in report["outputs"].values():
        assert (tmp_path / "analysis" / filename).stat().st_size > 0


def test_missing_sensors_produce_labeled_plots_and_no_fake_zero_errors(tmp_path):
    report = analyze_ball_sensor_comparison(tmp_path)
    assert report["status"] == "insufficient_synchronized_data"
    assert report["matchedCaptures"] == 0 and report["rmseMm"] is None
    assert (tmp_path / "analysis" / "ball_sensor_position_comparison.png").is_file()


def test_stop_requests_comparison_even_when_full_analysis_unchecked(tmp_path):
    recorder = SimpleNamespace(
        freeze=Mock(), record_event=Mock(),
        stop=Mock(return_value={"active": False, "runDir": str(tmp_path), "error": None}),
        start_analysis=Mock(return_value={"analysis": {"active": True}}))
    instance = SimpleNamespace(recorder=recorder, sim_mirror=None, control_loop_tick=12,
                               raw_camera_recording_status=Mock(return_value={"enabled": False}))
    status = probe.PolicySafetyProbe.stop_recording(instance, {"analyze": False})
    recorder.stop.assert_called_once_with(final_view=None, analyze=False)
    recorder.start_analysis.assert_called_once_with(str(tmp_path), ball_sensors_only=True)
    assert status["analysis"]["active"]


def test_comparison_only_worker_does_not_build_video_or_load_large_policy_log(tmp_path, monkeypatch):
    (tmp_path / "ball_sensor_samples.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows()))
    monkeypatch.setattr(probe, "_build_camera_video", Mock(side_effect=AssertionError("unexpected full analysis")))
    monkeypatch.setattr(probe, "_read_jsonl", Mock(side_effect=AssertionError("unexpected policy log load")))
    status = queue.Queue(maxsize=1)
    probe._recording_analysis_main(str(tmp_path), status, 60., ball_sensors_only=True)
    result = status.get_nowait()
    assert result["phase"] == "done" and not result.get("error")
    assert "ballSensorPositionComparison" in result["outputs"]
    summary = json.loads((tmp_path / "analysis" / "analysis_summary.json").read_text())
    assert summary["analysisSchemaVersion"] == 0


def test_previous_recording_analysis_does_not_suppress_new_recording_plots(tmp_path):
    old_process = SimpleNamespace(is_alive=lambda: True)
    child = SimpleNamespace(start=Mock())
    factory = Mock(return_value=child)
    recorder = probe.AsyncProbeRecorder.__new__(probe.AsyncProbeRecorder)
    recorder.lock = threading.Lock()
    recorder.cfg = SimpleNamespace(recording_analysis_video_fps=60.)
    recorder.ctx = SimpleNamespace(Queue=lambda **kw: queue.Queue(maxsize=1), Process=factory)
    recorder.analysis_process = old_process
    recorder.analysis_status_q = queue.Queue(maxsize=1)
    recorder.analysis_status = {"active": True, "analysisDir": str(tmp_path / "old" / "analysis")}
    recorder.retired_analysis_workers = []
    recorder._drain_status_locked = lambda: None
    recorder.status_locked = lambda: recorder.analysis_status
    recorder.start_analysis(str(tmp_path / "new"), ball_sensors_only=True)
    assert factory.call_args.kwargs["args"][-1] is True
    child.start.assert_called_once()
    assert recorder.retired_analysis_workers[0][0] is old_process
    assert recorder.analysis_status["analysisDir"] == str(tmp_path / "new" / "analysis")
