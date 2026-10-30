"""Compact trajectory records and bounded-memory offline export."""

import csv
import json
import math
from pathlib import Path


TRAJECTORY_FIELDS = (
    "schemaVersion", "sourceSeq", "sourceTimeS", "wallTimeS", "time", "tick",
    "mode", "ballSource", "ballObservationSeq", "cameraStampS",
    "rawValid", "valid", "age", "rawBall", "measuredBall", "measuredVelocity",
    "observedBall", "observedVelocity", "predictedBall", "predictedVelocity",
    "predictedBallIsPredicted", "racket",
)
SYNC_FIELDS = ("ready", "generation", "epoch", "sourceDomain", "receiverDomain")
MEASUREMENT_FIELDS = ("markerId", "ball_calibration_applied", "ball_calibration_time_shift_s")


def compact_trajectory_sample(row):
    """Whitelist ball fields; do not mutate the live sample or sensor caches."""
    output = {key: row[key] for key in TRAJECTORY_FIELDS if key in row}
    for key, fields in (("ballClockSync", SYNC_FIELDS), ("cameraMeasurement", MEASUREMENT_FIELDS)):
        if isinstance(row.get(key), dict):
            output[key] = {name: row[key][name] for name in fields if name in row[key]}
    # The counter needs measured racket XYZ, but no joint data or commands.
    if isinstance(row.get("ros"), dict) and row["ros"].get("jointEe") is not None:
        output["racket"] = row["ros"]["jointEe"]
    if "_ballSensorSnapshot" in row:
        output["_ballSensorSnapshot"] = row["_ballSensorSnapshot"]
    sensors = row.get("ballSensors")
    if isinstance(sensors, dict):
        compact = {key: sensors[key] for key in (
            "receiverTimeS", "bothValid", "comparisonReady", "measurementTimeDeltaS") if key in sensors}
        for source in ("camera", "mocap"):
            sensor = sensors.get(source) or {}
            compact[source] = {key: sensor[key] for key in (
                "enabled", "available", "valid", "frameAligned", "controlFrame",
                "position", "velocity", "measurementTimestampS", "measurementAgeS",
            ) if key in sensor}
            for key, fields in (("clockSync", SYNC_FIELDS), ("measurement", MEASUREMENT_FIELDS)):
                value = sensor.get(key) or {}
                compact[source][key] = {name: value[name] for name in fields if name in value}
        output["ballSensors"] = compact
    return output


def iter_trajectory_rows(path):
    """Read one JSON record at a time, including interrupted final lines."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def preview_jsonl(path, max_rows=3000):
    """Bound metadata previews too, when image recording is independently on."""
    if not Path(path).is_file():
        return []
    total = sum(1 for _ in iter_trajectory_rows(path))
    stride = max(1, math.ceil(max(0, total - 1) / (max_rows - 1)))
    return [row for index, row in enumerate(iter_trajectory_rows(path))
            if index % stride == 0 or index == total - 1]


def export_ball_trajectory(run_dir, analysis_dir, *, max_plot_samples=8000):
    """Export every position to CSV; retain at most max_plot_samples for UI.

    Two streaming passes give an even preview with the first and last sample.
    Neither pass holds the complete trajectory or a growing deduplication set.
    The per-sensor columns preserve invalid observations as missing values.
    """
    if max_plot_samples < 2:
        raise ValueError("max_plot_samples must be at least 2")
    path = Path(run_dir) / "control_samples.jsonl"
    total = sum(1 for _ in iter_trajectory_rows(path))
    stride = max(1, math.ceil(max(0, total - 1) / (max_plot_samples - 1)))
    selected = []
    vectors = ("rawBall", "measuredBall", "observedBall", "predictedBall", "racket")
    csv_path = Path(analysis_dir) / "ball_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_seq", "time_s", "source_time_s", "wall_time_s", "ball_source", "valid",
                         *[f"{key}_{axis}_m" for key in vectors for axis in "xyz"],
                         *[f"{source}_{field}" for source in ("camera", "mocap") for field in (
                             "valid", "measurement_timestamp_s", "x_m", "y_m", "z_m")]])
        for index, raw in enumerate(iter_trajectory_rows(path)):
            row = compact_trajectory_sample(raw)
            positions = []
            for key in vectors:
                vector = row.get(key)
                positions.extend(vector if isinstance(vector, (list, tuple)) and len(vector) == 3 else [None] * 3)
            sensors = row.get("ballSensors") or {}
            for source in ("camera", "mocap"):
                sensor = sensors.get(source) or {}
                valid = bool(sensor.get("valid") and sensor.get("frameAligned")
                             and (sensor.get("clockSync") or {}).get("ready"))
                vector = sensor.get("position") if valid else None
                positions.extend([valid, sensor.get("measurementTimestampS"),
                                  *(vector if isinstance(vector, (list, tuple)) and len(vector) == 3 else [None] * 3)])
            writer.writerow([row.get("sourceSeq"), row.get("time"), row.get("sourceTimeS"),
                             row.get("wallTimeS"), row.get("ballSource"), row.get("valid"), *positions])
            if index % stride == 0 or index == total - 1:
                selected.append(row)
    return selected, dict(sampleCount=total, plotSampleCount=len(selected),
                          maxPlotSamples=max_plot_samples, plotStride=stride,
                          fullResolutionFile=csv_path.name, reading="streaming_two_pass")
