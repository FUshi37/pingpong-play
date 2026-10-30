"""Offline camera/mocap position plots; never called by the control loop."""
import csv
import json
import os
from pathlib import Path

import numpy as np


def collect_observations(rows):
    """Deduplicate actual captures and split at losses and clock changes."""
    observations = {source: [] for source in ("camera", "mocap")}
    counts = {source: dict(validRows=0, invalidRows=0, rejectedFrameRows=0) for source in observations}
    state = {source: dict(epoch=None, segment=0, valid=False, last_stamp=None, seen=set()) for source in observations}
    origin = None
    total = 0
    for index, row in enumerate(rows):
        total += 1
        sensors = row.get("ballSensors", row)
        if origin is None and sensors.get("receiverTimeS") is not None:
            origin = float(sensors["receiverTimeS"])
        for source, points in observations.items():
            sensor = sensors.get(source) or {}
            sync = sensor.get("clockSync") or {}
            previous = state[source]
            epoch = (sync.get("generation"), sync.get("epoch"), sync.get("sourceDomain"), sync.get("receiverDomain"))
            valid = bool(sensor.get("valid") and sync.get("ready") and sensor.get("frameAligned"))
            stamp = sensor.get("measurementTimestampS")
            # Preserve original capture stamps in the recording. Only this
            # offline comparison applies the producer's calibrated ball time.
            measurement=sensor.get('measurement') or {}
            if source=='camera' and measurement.get('ball_calibration_applied') and stamp is not None:
                shift=measurement.get('ball_calibration_time_shift_s',0.)
                try:
                    shift=float(shift)
                    if not np.isfinite(shift) or abs(shift)>.020:
                        valid=False
                    else:
                        stamp=float(stamp)+shift
                except (TypeError,ValueError):
                    valid=False
            position = np.asarray(sensor.get("position") or [np.nan] * 3, dtype=float)
            valid = bool(valid and stamp is not None and np.isfinite(stamp)
                         and position.shape == (3,) and np.isfinite(position).all())
            if not valid:
                counts[source]["invalidRows"] += 1
                if sensor.get("valid") and not sensor.get("frameAligned"):
                    counts[source]["rejectedFrameRows"] += 1
                previous["valid"] = False
                continue
            counts[source]["validRows"] += 1
            if not previous["valid"] or epoch != previous["epoch"]:
                previous["segment"] += 1
            if previous["last_stamp"] is not None and stamp < previous["last_stamp"]:
                previous["segment"] += 1
            previous.update(valid=True, epoch=epoch, last_stamp=stamp)
            key = (epoch, stamp)
            if key in previous["seen"]:
                continue
            previous["seen"].add(key)
            points.append(dict(stamp=float(stamp), position=position,
                               segment=previous["segment"], rowIndex=index,
                               controlTimeS=row.get("time"), sourceSeq=row.get("sourceSeq")))
    if origin is None:
        stamps = [p["stamp"] for pts in observations.values() for p in pts]
        origin = min(stamps) if stamps else 0.
    return observations, dict(rows=total, originTimestampS=origin, sources=counts)


def align_positions(observations, max_gap_s=.05):
    """Interpolate mocap at unique camera capture times within valid segments.

    Never extrapolate or bridge a lost/unsynchronized mocap interval. Row
    ranges also prevent matching old and new ROS-clock epochs that overlap
    numerically after a backwards clock jump.
    """
    camera, mocap = observations["camera"], observations["mocap"]
    matches = []
    segments = {}
    for point in mocap:
        segments.setdefault(point["segment"], []).append(point)
    for segment in segments.values():
        segment = sorted(segment, key=lambda p: p["stamp"])
        times = np.asarray([p["stamp"] for p in segment])
        positions = np.asarray([p["position"] for p in segment])
        row_min, row_max = min(p["rowIndex"] for p in segment), max(p["rowIndex"] for p in segment)
        for point in camera:
            if not row_min <= point["rowIndex"] <= row_max:
                continue
            stamp = point["stamp"]
            right = int(np.searchsorted(times, stamp))
            if right < len(times) and times[right] == stamp:
                value = positions[right]
                gap = 0.
            elif right == 0 or right == len(times):
                continue
            else:
                gap = times[right] - times[right - 1]
                if gap <= 0. or gap > max_gap_s:
                    continue
                fraction = (stamp - times[right - 1]) / gap
                value = positions[right - 1] * (1. - fraction) + positions[right] * fraction
            matches.append(dict(point, segment=(point["segment"], segment[0]["segment"]),
                                mocapPosition=value, difference=point["position"] - value,
                                interpolationGapS=float(gap)))
    return sorted(matches, key=lambda p: p["rowIndex"])


def _statistics(matches):
    if not matches:
        return dict(matchedCaptures=0, biasMm=None, rmseMm=None, positionRmseMm=None,
                    normMedianMm=None, normP95Mm=None, normMaxMm=None)
    differences = 1000. * np.asarray([p["difference"] for p in matches])
    norms = np.linalg.norm(differences, axis=1)
    return dict(matchedCaptures=len(matches), biasMm=differences.mean(axis=0).tolist(),
                rmseMm=np.sqrt(np.mean(differences ** 2, axis=0)).tolist(),
                positionRmseMm=float(np.sqrt(np.mean(norms ** 2))),
                normMedianMm=float(np.median(norms)), normP95Mm=float(np.percentile(norms, 95)),
                normMaxMm=float(np.max(norms)))


def _plot_arrays(points, origin, field, max_gap_s):
    times, values = [], []
    previous = None
    for point in points:
        if previous is not None and (point["segment"] != previous["segment"]
                                     or point["stamp"] - previous["stamp"] > max_gap_s):
            times.append(np.nan)
            values.append(np.full(3, np.nan))
        times.append(point["stamp"] - origin)
        values.append(point[field])
        previous = point
    return np.asarray(times), np.asarray(values).reshape(-1, 3)


def analyze_ball_sensor_comparison(run_dir, analysis_dir=None, *, max_gap_s=.05, control_samples=None):
    """Write comparison/difference PNGs and their timestamp-aligned CSV data."""
    run_dir = Path(run_dir)
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "ball_sensor_samples.jsonl"
    if not path.is_file():
        path = run_dir / "control_samples.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            observations, report = collect_observations(json.loads(line) for line in handle if line.strip())
    else:
        observations, report = collect_observations(())
    matches = align_positions(observations, max_gap_s=max_gap_s)
    report.update(_statistics(matches))
    report.update(status="ok" if matches else "insufficient_synchronized_data",
                  differenceDefinition="camera minus mocap interpolated at camera capture time",
                  positionUnit="m", differenceUnit="mm", maxInterpolationGapS=max_gap_s,
                  timingCorrectionApplied=False,
                  timingLimitation="Device-to-ROS synchronization does not calibrate fixed sensor latency.")
    for source in observations:
        report["sources"][source]["uniqueCaptures"] = len(observations[source])
    if control_samples:
        by_seq = {s.get("sourceSeq"): s for s in control_samples}
        policy_matches = [p for p in matches if by_seq.get(p.get("sourceSeq"), {}).get("mode") == "policy"]
        report["policy"] = _statistics(policy_matches)
    csv_path = analysis_dir / "ball_sensor_position_difference.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "measurement_timestamp_s", "control_time_s", "source_seq",
                         "camera_x_m", "camera_y_m", "camera_z_m", "mocap_x_m", "mocap_y_m", "mocap_z_m",
                         "difference_x_mm", "difference_y_mm", "difference_z_mm", "difference_norm_mm", "mocap_interpolation_gap_s"])
        for point in matches:
            delta = 1000. * point["difference"]
            writer.writerow([point["stamp"] - report["originTimestampS"], point["stamp"],
                             point["controlTimeS"], point["sourceSeq"], *point["position"],
                             *point["mocapPosition"], *delta, np.linalg.norm(delta), point["interpolationGapS"]])
    source_path = analysis_dir / "ball_sensor_positions.csv"
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "time_s", "measurement_timestamp_s", "control_time_s", "source_seq", "segment", "x_m", "y_m", "z_m"])
        for source, points in observations.items():
            for point in points:
                writer.writerow([source, point["stamp"] - report["originTimestampS"], point["stamp"],
                                 point["controlTimeS"], point["sourceSeq"], point["segment"], *point["position"]])

    # Import plotting only inside the low-priority analysis process.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
        for source, color in (("mocap", "#0072B2"), ("camera", "#D55E00")):
            times, positions = _plot_arrays(observations[source], report["originTimestampS"], "position", max_gap_s)
            for index, axis in enumerate(axes):
                axis.plot(times, positions[:, index], color=color, linewidth=1., label=source.capitalize())
                axis.set_ylabel(f"{'XYZ'[index]} position (m)")
                axis.grid(alpha=.2)
        axes[0].legend(loc="upper left", ncol=2)
        axes[0].set_title("Ball position: camera and motion capture (own capture timestamps)")
        axes[-1].set_xlabel("Time since recording start (s)")
        if not observations["camera"] or not observations["mocap"]:
            missing = ", ".join(s for s, pts in observations.items() if not pts)
            axes[0].text(.5, .5, f"No valid {missing} observations", ha="center", transform=axes[0].transAxes)
        fig.savefig(analysis_dir / "ball_sensor_position_comparison.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
        times, differences = _plot_arrays(matches, report["originTimestampS"], "difference", max_gap_s)
        for index, axis in enumerate(axes[:3]):
            axis.plot(times, 1000. * differences[:, index], color="#D55E00", linewidth=1.)
            axis.axhline(0., color=".45", linewidth=.6)
            axis.set_ylabel(f"Delta {'XYZ'[index]} (mm)")
            axis.grid(alpha=.2)
        axes[-1].plot(times, 1000. * np.linalg.norm(differences, axis=1), color="#009E73", linewidth=1.)
        axes[-1].set_ylabel("3D difference (mm)")
        axes[-1].grid(alpha=.2)
        axes[-1].set_xlabel("Time since recording start (s)")
        axes[0].set_title("Camera minus motion capture, aligned at camera capture time")
        if not matches:
            axes[0].text(.5, .5, "No synchronized position pairs", ha="center", transform=axes[0].transAxes)
        fig.savefig(analysis_dir / "ball_sensor_position_difference.png", dpi=160)
        plt.close(fig)
    outputs = dict(ballSensorPositionComparison="ball_sensor_position_comparison.png",
                   ballSensorPositionDifference="ball_sensor_position_difference.png",
                   ballSensorPositionDifferenceCsv=csv_path.name, ballSensorPositionsCsv=source_path.name,
                   ballSensorComparisonReport="ball_sensor_comparison_report.json")
    report["outputs"] = outputs
    (analysis_dir / outputs["ballSensorComparisonReport"]).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
