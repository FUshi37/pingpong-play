#!/usr/bin/env python3
"""Replay synchronized IR+depth measurements through IRDepthKalmanFilter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pingpong_controller.ir_depth_kalman_filter import IRDepthKalmanFilter


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def analysis_segment(rows: list[dict]) -> list[dict]:
    end = next((
        index for index, row in enumerate(rows)
        if str(row.get("mode", "")).startswith("done")
    ), len(rows))
    start = 0
    for index in range(1, end):
        if float(rows[index].get("time", 0.0)) < (
                float(rows[index - 1].get("time", 0.0)) - 0.1):
            start = index
    return rows[start:end]


def unique_camera_measurements(rows: list[dict]) -> list[dict]:
    selected = []
    previous_position = None
    for row in rows:
        camera = row.get("cameraMeasurement") or {}
        position = tuple(row.get("measuredBall") or ())
        required = (
            camera.get("cx_raw"),
            camera.get("cy_raw"),
            camera.get("depth_mm"),
            camera.get("sensor_depth_mm"),
        )
        if (not row.get("rawValid") or len(position) != 3
                or position == previous_position
                or any(value is None for value in required)):
            continue
        selected.append(row)
        previous_position = position
    return selected


def fit_recorded_projection(rows: list[dict]) -> np.ndarray:
    """Recover the exact affine IR back-projection used by this recording."""
    features = []
    positions_mm = []
    for row in rows:
        camera = row["cameraMeasurement"]
        depth_mm = float(camera["depth_mm"])
        features.append([
            float(camera["cx_raw"]) * depth_mm,
            float(camera["cy_raw"]) * depth_mm,
            depth_mm,
            1.0,
        ])
        positions_mm.append(
            np.asarray(row["measuredBall"], dtype=np.float64) * 1000.0)
    mapping, _, _, _ = np.linalg.lstsq(
        np.asarray(features), np.asarray(positions_mm), rcond=None)
    return mapping


def synchronized_position_mm(row: dict, mapping: np.ndarray) -> np.ndarray:
    camera = row["cameraMeasurement"]
    depth_mm = float(camera["sensor_depth_mm"])
    feature = np.array([
        float(camera["cx_raw"]) * depth_mm,
        float(camera["cy_raw"]) * depth_mm,
        depth_mm,
        1.0,
    ])
    return feature @ mapping


def metric_summary(
    time_s: np.ndarray,
    original_velocity: np.ndarray,
    filtered_velocity: np.ndarray,
) -> dict[str, float]:
    dt_s = np.maximum(np.diff(time_s), 1e-4)
    original_lateral = np.linalg.norm(original_velocity[:, :2], axis=1)
    filtered_lateral = np.linalg.norm(filtered_velocity[:, :2], axis=1)
    original_acceleration = np.linalg.norm(
        np.diff(original_velocity[:, :2], axis=0) / dt_s[:, None], axis=1)
    filtered_acceleration = np.linalg.norm(
        np.diff(filtered_velocity[:, :2], axis=0) / dt_s[:, None], axis=1)
    return {
        "originalLateralVelocityP99Mps": float(np.percentile(
            original_lateral, 99)),
        "filteredLateralVelocityP99Mps": float(np.percentile(
            filtered_lateral, 99)),
        "originalLateralAccelerationP95Mps2": float(np.percentile(
            original_acceleration, 95)),
        "filteredLateralAccelerationP95Mps2": float(np.percentile(
            filtered_acceleration, 95)),
        "originalAbsVzMaxMps": float(np.max(np.abs(
            original_velocity[:, 2]))),
        "filteredAbsVzMaxMps": float(np.max(np.abs(
            filtered_velocity[:, 2]))),
    }


def analyze_recording(run_dir: Path, frame_rotation: np.ndarray) -> dict:
    rows = unique_camera_measurements(analysis_segment(
        read_jsonl(run_dir / "control_samples.jsonl")))
    mapping = fit_recorded_projection(rows)
    estimator = IRDepthKalmanFilter(frame_rotation=frame_rotation)
    filtered_position = []
    filtered_velocity = []
    for row in rows:
        position_mm, velocity_mm_s = estimator.update(
            synchronized_position_mm(row, mapping),
            float(row["cameraStampS"]),
        )
        filtered_position.append(position_mm / 1000.0)
        filtered_velocity.append(velocity_mm_s / 1000.0)

    policy = np.array([row.get("mode") == "policy" for row in rows])
    time_s = np.asarray([row["time"] for row in rows], dtype=np.float64)[policy]
    original_position = np.asarray([
        row["measuredBall"] for row in rows], dtype=np.float64)[policy]
    original_velocity = np.asarray([
        row["measuredVelocity"] for row in rows], dtype=np.float64)[policy]
    filtered_position_array = np.asarray(filtered_position)[policy]
    filtered_velocity_array = np.asarray(filtered_velocity)[policy]
    position_rmse_mm = np.sqrt(np.mean(
        (filtered_position_array - original_position) ** 2, axis=0)) * 1000.0

    result = {
        "recording": run_dir.name,
        "cameraSamples": int(len(rows)),
        "policySamples": int(np.count_nonzero(policy)),
        **metric_summary(time_s, original_velocity, filtered_velocity_array),
        "positionRmseMm": position_rmse_mm.tolist(),
        "filterStats": estimator.get_stats(),
    }
    if run_dir.name.endswith("210358"):
        window = (time_s >= 13.0) & (time_s <= 14.5)
        result["window13To14_5"] = {
            "originalAbsVelocityMaxMps": np.max(
                np.abs(original_velocity[window]), axis=0).tolist(),
            "filteredAbsVelocityMaxMps": np.max(
                np.abs(filtered_velocity_array[window]), axis=0).tolist(),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument(
        "--calibration",
        required=True,
        type=Path,
        help="Calibration JSON containing T_base_camera.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    frame_rotation = np.asarray(
        calibration["T_base_camera"], dtype=np.float64)[:3, :3]
    results = [
        analyze_recording(path.expanduser().resolve(), frame_rotation)
        for path in args.recordings
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
