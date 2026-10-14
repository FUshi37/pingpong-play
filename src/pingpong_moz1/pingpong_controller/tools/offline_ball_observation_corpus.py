#!/usr/bin/env python3
"""Offline ball-observation corpus analysis for record_new3/4/5.

This script analyzes retained 3D detector telemetry from interactive probe
recordings.  It deliberately does not rerun raw IR contour extraction or
16-bit depth sampling unless those arrays are present in the recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import offline_ball_trajectory_study as traj  # noqa: E402

SERIES_COLORS = {
    "recorded_policy_state": "#2a78d6",
    "raw_finite_difference": "#16a34a",
    "causal_alpha_beta_loose": "#a855f7",
    "causal_alpha_beta_default": "#eab308",
    "causal_alpha_beta_strict": "#ef4444",
    "causal_kalman_like_default": "#0f766e",
    "state_replay_recommended": "#8b5cf6",
}

COHORT_DATES = {
    "record_new3": ["20260810"],
    "record_new4": ["20260811"],
    "record_new5": ["20260812", "20260813"],
}

REFERENCE_GAP_LIMIT_S = 0.080
BALL_RADIUS_M = 0.020
CONTACT_WINDOW_S = 0.120
TERMINAL_WINDOW_S = 0.250
MAX_PREDICTION_HOLD_S = 0.080
MAX_CONSECUTIVE_REJECTIONS = 8
MAX_TRACK_SPEED_MPS = 6.0
MAX_PREDICTED_STEP_M = 0.180


@dataclass(frozen=True)
class ObservationVariant:
    name: str
    kind: str
    innovation_gate_m: float = 0.080
    alpha: float = 0.65
    beta: float = 0.18
    process_noise: float = 0.55
    measurement_noise_m: float = 0.020
    depth_gate_mm: float | None = None


OBSERVATION_VARIANTS = (
    ObservationVariant("recorded_policy_state", "recorded"),
    ObservationVariant("raw_finite_difference", "raw_fd"),
    ObservationVariant(
        "causal_alpha_beta_loose", "alpha_beta",
        innovation_gate_m=0.140, alpha=0.75, beta=0.24),
    ObservationVariant(
        "causal_alpha_beta_default", "alpha_beta",
        innovation_gate_m=0.090, alpha=0.62, beta=0.17),
    ObservationVariant(
        "causal_alpha_beta_strict", "alpha_beta",
        innovation_gate_m=0.055, alpha=0.50, beta=0.12,
        depth_gate_mm=45.0),
)


@dataclass
class RunAnalysis:
    cohort: str
    run_dir: Path
    ticks: list[traj.Tick]
    refreshes: list[traj.Tick]
    phase: traj.Phase | None
    reference: traj.ReferenceTrajectory | None
    raw_rows: list[dict[str, Any]]
    exclusion_reason: str | None = None


@dataclass
class VariantSeries:
    name: str
    times_s: np.ndarray
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    valid: np.ndarray
    diagnostics: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_recordings(roots: Iterable[Path]) -> list[tuple[str, Path]]:
    found: dict[Path, str] = {}
    for root in roots:
        root = root.expanduser().resolve()
        cohort = root.name
        for control in sorted(root.rglob("control_samples.jsonl")):
            run_dir = control.parent.resolve()
            if run_dir.name.startswith("probe_recording_"):
                found.setdefault(run_dir, cohort)
    return sorted(((cohort, path) for path, cohort in found.items()),
                  key=lambda item: (item[0], item[1].name))


def capability_for_run(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    field_counts = Counter()
    camera_keys = Counter()
    predicted_count = 0
    fresh_camera = 0
    depth_delta_values: list[float] = []
    for row in rows:
        field_counts.update(row.keys())
        camera = row.get("cameraMeasurement")
        if isinstance(camera, dict):
            camera_keys.update(camera.keys())
            if row.get("cameraStampS") is not None:
                fresh_camera += 1
            try:
                sensor = float(camera.get("sensor_depth_mm"))
                geom = float(camera.get("geom_depth_mm"))
            except (TypeError, ValueError):
                sensor = geom = float("nan")
            if math.isfinite(sensor) and math.isfinite(geom):
                depth_delta_values.append(sensor - geom)
        prediction = row.get("highBallOcclusionPrediction")
        estimator = row.get("estimator")
        if (row.get("predictedBall") is not None
                or (isinstance(prediction, dict)
                    and prediction.get("predicted", False))
                or (isinstance(estimator, dict)
                    and estimator.get("predicted", False))):
            predicted_count += 1
    raw_files = list(run_dir.glob("*.npy")) + list(run_dir.glob("*.npz"))
    bag_files = list(run_dir.glob("*.bag")) + list(run_dir.glob("*.db3"))
    return {
        "rowCount": len(rows),
        "freshCameraRows": fresh_camera,
        "hasRawIrDepthArrays": bool(raw_files),
        "hasRosBagOrDb3": bool(bag_files),
        "fullRawVisionRerunAvailable": bool(raw_files or bag_files),
        "predictedRows": predicted_count,
        "fieldCounts": dict(field_counts),
        "cameraFieldCounts": dict(camera_keys),
        "sensorGeometryDepthDeltaMeanAbsMm": percentile_abs(
            depth_delta_values, 50.0),
        "sensorGeometryDepthDeltaP95AbsMm": percentile_abs(
            depth_delta_values, 95.0),
    }


def percentile(values: Iterable[float], pct: float) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return round(float(np.percentile(arr, pct)), 6)


def percentile_abs(values: Iterable[float], pct: float) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return round(float(np.percentile(np.abs(arr), pct)), 6)


def mean(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return round(float(np.mean(arr)), 6)


def finite_point(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 3:
        return None
    arr = arr[:3]
    return arr if np.all(np.isfinite(arr)) else None


def finite_velocity(value: Any) -> np.ndarray | None:
    return finite_point(value)


def tick_time(tick: traj.Tick) -> float:
    return float(tick.wall_time_s)


def tick_stamp(tick: traj.Tick) -> float:
    return float(tick.camera_stamp_s or tick.wall_time_s)


def build_run_analysis(cohort: str, run_dir: Path) -> RunAnalysis:
    raw_rows = read_jsonl(run_dir / "control_samples.jsonl")
    ticks = traj.load_ticks(run_dir)
    refreshes = traj.fresh_camera_ticks(ticks)
    phase = traj.select_valid_phase(ticks)
    if phase is None:
        return RunAnalysis(
            cohort, run_dir, ticks, refreshes, None, None, raw_rows,
            exclusion_reason="no_valid_policy_juggling_phase")
    phase_refreshes = [
        tick for tick in refreshes
        if phase.start_s <= tick.wall_time_s <= phase.end_s
    ]
    reference = traj.build_reference(phase_refreshes, phase)
    if reference is None:
        return RunAnalysis(
            cohort, run_dir, ticks, refreshes, phase, None, raw_rows,
            exclusion_reason="no_reference_trajectory")
    return RunAnalysis(cohort, run_dir, ticks, refreshes, phase, reference, raw_rows)


def raw_row_nearest_time(run: RunAnalysis, wall_time_s: float) -> dict[str, Any]:
    best_row: dict[str, Any] = {}
    best_distance = float("inf")
    for row in run.raw_rows:
        try:
            row_time = float(row.get("wallTimeS", row.get("sourceTimeS", "nan")))
        except (TypeError, ValueError):
            continue
        distance = abs(row_time - wall_time_s)
        if math.isfinite(distance) and distance < best_distance:
            best_distance = distance
            best_row = row
    return best_row


def selected_ticks(run: RunAnalysis) -> list[traj.Tick]:
    if run.phase is None:
        return []
    return [
        tick for tick in run.refreshes
        if run.phase.start_s <= tick.wall_time_s <= run.phase.end_s
    ]


def series_recorded(ticks: list[traj.Tick]) -> VariantSeries:
    times = np.asarray([tick_time(t) for t in ticks], dtype=np.float64)
    positions = np.asarray([t.measured_ball_m for t in ticks], dtype=np.float64)
    velocities = np.asarray([t.measured_velocity_m_s for t in ticks], dtype=np.float64)
    valid = np.asarray([
        bool(t.raw_valid)
        and np.all(np.isfinite(t.measured_ball_m))
        and np.all(np.isfinite(t.measured_velocity_m_s))
        for t in ticks
    ], dtype=bool)
    return VariantSeries(
        "recorded_policy_state", times, positions, velocities, valid, {})


def series_raw_fd(ticks: list[traj.Tick]) -> VariantSeries:
    times = np.asarray([tick_time(t) for t in ticks], dtype=np.float64)
    positions = np.asarray([t.raw_ball_m for t in ticks], dtype=np.float64)
    valid = np.asarray([
        bool(t.raw_valid) and np.all(np.isfinite(t.raw_ball_m))
        for t in ticks
    ], dtype=bool)
    velocities = np.full_like(positions, np.nan)
    last_index: int | None = None
    resets = 0
    for index, tick in enumerate(ticks):
        if not valid[index]:
            last_index = None
            resets += 1
            continue
        if last_index is None:
            last_index = index
            continue
        dt = tick_stamp(tick) - tick_stamp(ticks[last_index])
        if not 1e-4 < dt <= 0.075:
            last_index = index
            resets += 1
            continue
        velocity = (positions[index] - positions[last_index]) / dt
        if np.linalg.norm(velocity) > 8.0:
            velocities[index] = np.nan
            resets += 1
        else:
            velocities[index] = velocity
        last_index = index
    return VariantSeries(
        "raw_finite_difference", times, positions, velocities,
        valid & np.all(np.isfinite(velocities), axis=1),
        {"velocityResets": resets})


def camera_depth_delta_mm(tick: traj.Tick) -> float:
    measurement = tick.measurement if isinstance(tick.measurement, dict) else {}
    try:
        sensor = float(measurement.get("sensor_depth_mm"))
        geom = float(measurement.get("geom_depth_mm"))
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(sensor) and math.isfinite(geom)):
        return float("nan")
    return sensor - geom


def measurement_available(tick: traj.Tick, variant: ObservationVariant) -> bool:
    if not (tick.raw_valid and np.all(np.isfinite(tick.raw_ball_m))):
        return False
    if variant.depth_gate_mm is None:
        return True
    delta = camera_depth_delta_mm(tick)
    return (not math.isfinite(delta)) or abs(delta) <= variant.depth_gate_mm


def series_alpha_beta(ticks: list[traj.Tick], variant: ObservationVariant) -> VariantSeries:
    times = np.asarray([tick_time(t) for t in ticks], dtype=np.float64)
    positions = np.full((len(ticks), 3), np.nan, dtype=np.float64)
    velocities = np.full((len(ticks), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(ticks), dtype=bool)
    x: np.ndarray | None = None
    v = np.zeros(3, dtype=np.float64)
    last_stamp: float | None = None
    rejected = 0
    resets = 0
    depth_rejected = 0
    rejected_streak = 0
    held_s = 0.0
    for index, tick in enumerate(ticks):
        stamp = tick_stamp(tick)
        dt = 0.0 if last_stamp is None else stamp - last_stamp
        if x is not None and 1e-4 < dt <= 0.12:
            v[2] -= 9.81 * dt
            x = x + v * dt
        elif x is not None and dt > 0.12:
            x = None
            v = np.zeros(3, dtype=np.float64)
            resets += 1
        if not measurement_available(tick, variant):
            if variant.depth_gate_mm is not None and tick.raw_valid:
                depth_rejected += 1
            if x is not None and 1e-4 < dt <= 0.12:
                held_s += dt
            if (x is not None and held_s <= MAX_PREDICTION_HOLD_S
                    and rejected_streak < MAX_CONSECUTIVE_REJECTIONS):
                positions[index] = x
                velocities[index] = v
                valid[index] = True
            else:
                x = None
                v = np.zeros(3, dtype=np.float64)
                rejected_streak = 0
                held_s = 0.0
                resets += 1
            last_stamp = stamp
            continue
        z = tick.raw_ball_m.copy()
        if x is None:
            x = z.copy()
            v = tick.measured_velocity_m_s.copy()
            if not np.all(np.isfinite(v)):
                v = np.zeros(3, dtype=np.float64)
            valid[index] = True
            rejected_streak = 0
            held_s = 0.0
        else:
            innovation = z - x
            if float(np.linalg.norm(innovation)) > variant.innovation_gate_m:
                rejected += 1
                rejected_streak += 1
                if 1e-4 < dt <= 0.12:
                    held_s += dt
                if (held_s > MAX_PREDICTION_HOLD_S
                        or rejected_streak >= MAX_CONSECUTIVE_REJECTIONS):
                    x = None
                    v = np.zeros(3, dtype=np.float64)
                    rejected_streak = 0
                    held_s = 0.0
                    resets += 1
            else:
                inv_dt = 1.0 / max(dt, 0.005)
                x = x + variant.alpha * innovation
                v = v + variant.beta * innovation * inv_dt
                valid[index] = True
                rejected_streak = 0
                held_s = 0.0
        if x is not None:
            positions[index] = x
            velocities[index] = v
        last_stamp = stamp
    return VariantSeries(
        variant.name, times, positions, velocities, valid,
        {"innovationRejected": rejected, "resets": resets,
         "depthRejected": depth_rejected})


def series_kalman_like(ticks: list[traj.Tick], variant: ObservationVariant) -> VariantSeries:
    times = np.asarray([tick_time(t) for t in ticks], dtype=np.float64)
    positions = np.full((len(ticks), 3), np.nan, dtype=np.float64)
    velocities = np.full((len(ticks), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(ticks), dtype=bool)
    state: np.ndarray | None = None
    covariance = np.eye(6, dtype=np.float64)
    last_stamp: float | None = None
    rejected = 0
    depth_rejected = 0
    resets = 0
    rejected_streak = 0
    held_s = 0.0
    r_var = float(variant.measurement_noise_m) ** 2
    for index, tick in enumerate(ticks):
        stamp = tick_stamp(tick)
        dt = 0.0 if last_stamp is None else stamp - last_stamp
        if state is not None and 1e-4 < dt <= 0.12:
            f = np.eye(6, dtype=np.float64)
            f[0, 3] = dt
            f[1, 4] = dt
            f[2, 5] = dt
            state = f @ state
            state[2] -= 0.5 * 9.81 * dt * dt
            state[5] -= 9.81 * dt
            if (np.linalg.norm(state[3:]) > MAX_TRACK_SPEED_MPS
                    or np.linalg.norm(state[:3] - positions[index - 1])
                    > MAX_PREDICTED_STEP_M):
                state = None
                covariance = np.eye(6, dtype=np.float64)
                rejected_streak = 0
                held_s = 0.0
                resets += 1
            if state is None:
                last_stamp = stamp
                continue
            q = np.eye(6, dtype=np.float64) * (variant.process_noise * dt) ** 2
            covariance = f @ covariance @ f.T + q
        elif state is not None and dt > 0.12:
            state = None
            covariance = np.eye(6, dtype=np.float64)
            resets += 1
        if measurement_available(tick, variant):
            z = tick.raw_ball_m.copy()
            if state is None:
                velocity = tick.measured_velocity_m_s.copy()
                if not np.all(np.isfinite(velocity)):
                    velocity = np.zeros(3, dtype=np.float64)
                state = np.concatenate([z, velocity])
                covariance = np.eye(6, dtype=np.float64) * 0.02
                rejected_streak = 0
                held_s = 0.0
            else:
                innovation = z - state[:3]
                if float(np.linalg.norm(innovation)) > variant.innovation_gate_m:
                    rejected += 1
                    rejected_streak += 1
                    if 1e-4 < dt <= 0.12:
                        held_s += dt
                    if (held_s > MAX_PREDICTION_HOLD_S
                            or rejected_streak >= MAX_CONSECUTIVE_REJECTIONS):
                        state = None
                        covariance = np.eye(6, dtype=np.float64)
                        rejected_streak = 0
                        held_s = 0.0
                        resets += 1
                else:
                    h = np.zeros((3, 6), dtype=np.float64)
                    h[:, :3] = np.eye(3, dtype=np.float64)
                    r = np.eye(3, dtype=np.float64) * r_var
                    s = h @ covariance @ h.T + r
                    k = covariance @ h.T @ np.linalg.inv(s)
                    state = state + k @ innovation
                    covariance = (np.eye(6, dtype=np.float64) - k @ h) @ covariance
                    rejected_streak = 0
                    held_s = 0.0
            valid[index] = state is not None
        else:
            if variant.depth_gate_mm is not None and tick.raw_valid:
                depth_rejected += 1
            if state is not None and 1e-4 < dt <= 0.12:
                held_s += dt
            if state is None or held_s > MAX_PREDICTION_HOLD_S:
                if state is not None:
                    resets += 1
                state = None
                covariance = np.eye(6, dtype=np.float64)
                rejected_streak = 0
                held_s = 0.0
                valid[index] = False
            else:
                valid[index] = True
        if state is not None:
            positions[index] = state[:3]
            velocities[index] = state[3:]
        last_stamp = stamp
    return VariantSeries(
        variant.name, times, positions, velocities, valid,
        {"innovationRejected": rejected, "depthRejected": depth_rejected,
         "resets": resets})


def build_observation_series(
    ticks: list[traj.Tick], variant: ObservationVariant,
) -> VariantSeries:
    if variant.kind == "recorded":
        return series_recorded(ticks)
    if variant.kind == "raw_fd":
        return series_raw_fd(ticks)
    if variant.kind == "alpha_beta":
        return series_alpha_beta(ticks, variant)
    if variant.kind == "kalman_like":
        return series_kalman_like(ticks, variant)
    raise ValueError(f"unsupported variant kind: {variant.kind}")


def metric_with_z(
    series: VariantSeries,
    reference: traj.ReferenceTrajectory,
    contacts_s: list[float],
) -> dict[str, Any]:
    metrics = traj.trajectory_metrics(
        series.times_s, series.positions_m, series.velocities_m_s,
        series.valid, reference, contacts_s)
    ref_position, ref_velocity = traj.interpolate_reference(reference, series.times_s)
    finite = (
        series.valid
        & np.all(np.isfinite(series.positions_m), axis=1)
        & np.all(np.isfinite(series.velocities_m_s), axis=1))
    contact = traj._contact_mask(series.times_s, contacts_s)  # noqa: SLF001
    evaluate = finite & ~contact
    z_error_mm = (series.positions_m[:, 2] - ref_position[:, 2]) * 1000.0
    vz_error = series.velocities_m_s[:, 2] - ref_velocity[:, 2]
    metrics.update({
        "validRatio": round(
            float(np.count_nonzero(finite)) / float(max(1, len(series.times_s))),
            6,
        ),
        "zErrorMedianMm": percentile(z_error_mm[evaluate], 50.0),
        "zErrorP95Mm": percentile_abs(z_error_mm[evaluate], 95.0),
        "vzErrorMedianMps": percentile(vz_error[evaluate], 50.0),
        "vzErrorP95AbsMps": percentile_abs(vz_error[evaluate], 95.0),
    })
    return metrics


def nearest_index(times: np.ndarray, target: float) -> int | None:
    finite = np.isfinite(times)
    if not np.any(finite):
        return None
    distances = np.where(finite, np.abs(times - target), np.inf)
    index = int(np.argmin(distances))
    return index if math.isfinite(float(distances[index])) else None


def contact_metrics_for_variant(
    run: RunAnalysis,
    series: VariantSeries,
) -> list[dict[str, Any]]:
    if run.phase is None:
        return []
    rows = []
    recorded = series_recorded(selected_ticks(run))
    for contact_index, contact_time in enumerate(run.phase.contacts_s):
        index = nearest_index(series.times_s, contact_time)
        rec_index = nearest_index(recorded.times_s, contact_time)
        if index is None:
            continue
        window = np.abs(series.times_s - contact_time) <= CONTACT_WINDOW_S
        pre_window = ((series.times_s >= contact_time - CONTACT_WINDOW_S)
                      & (series.times_s <= contact_time - 0.020))
        post_window = ((series.times_s >= contact_time + 0.020)
                       & (series.times_s <= contact_time + CONTACT_WINDOW_S))
        z_gap_mm = float("nan")
        distance_m = float("nan")
        racket_z_m = float("nan")
        if rec_index is not None:
            ticks = selected_ticks(run)
            if rec_index < len(ticks):
                racket = ticks[rec_index].racket_m
                if np.all(np.isfinite(racket)) and index < len(series.positions_m):
                    racket_z_m = float(racket[2])
                    z_gap_mm = float(
                        (series.positions_m[index, 2] - racket[2]) * 1000.0)
                    distance_m = float(
                        np.linalg.norm(series.positions_m[index] - racket))
        vz = series.velocities_m_s[:, 2]
        rows.append({
            "cohort": run.cohort,
            "run": run.run_dir.name,
            "variant": series.name,
            "contactIndex": contact_index,
            "contactWallTimeS": round(float(contact_time), 6),
            "validSamplesInWindow": int(np.count_nonzero(series.valid & window)),
            "zGapMm": round(z_gap_mm, 6) if math.isfinite(z_gap_mm) else None,
            "ballRacketDistanceM": (
                round(distance_m, 6) if math.isfinite(distance_m) else None),
            "racketZM": round(racket_z_m, 6) if math.isfinite(racket_z_m) else None,
            "vzPreMedianMps": percentile(vz[pre_window], 50.0),
            "vzPostMedianMps": percentile(vz[post_window], 50.0),
            "vzReversalMps": (
                None if percentile(vz[pre_window], 50.0) is None
                or percentile(vz[post_window], 50.0) is None else round(
                    float(percentile(vz[post_window], 50.0))
                    - float(percentile(vz[pre_window], 50.0)), 6)),
        })
    return rows


def terminal_metrics_for_run(
    run: RunAnalysis,
    series_by_name: dict[str, VariantSeries],
) -> dict[str, Any]:
    if run.phase is None:
        return {
            "cohort": run.cohort,
            "run": run.run_dir.name,
            "classification": "insufficient_evidence",
            "reason": run.exclusion_reason,
        }
    recorded = series_by_name.get("recorded_policy_state")
    if recorded is None:
        return {
            "cohort": run.cohort,
            "run": run.run_dir.name,
            "classification": "insufficient_evidence",
            "reason": "missing_recorded_series",
        }
    terminal_time = run.phase.miss_time_s or run.phase.end_s
    window = ((recorded.times_s >= terminal_time - TERMINAL_WINDOW_S)
              & (recorded.times_s <= terminal_time))
    ages: list[float] = []
    depth_delta: list[float] = []
    predicted = 0
    loss_like = 0
    ticks = selected_ticks(run)
    raw_by_time = []
    for row in run.raw_rows:
        try:
            row_time = float(row.get("wallTimeS", row.get("sourceTimeS", "nan")))
        except (TypeError, ValueError):
            row_time = float("nan")
        raw_by_time.append((row_time, row))
    for tick in ticks:
        if not (terminal_time - TERMINAL_WINDOW_S <= tick.wall_time_s <= terminal_time):
            continue
        nearest_row = min(
            raw_by_time,
            key=lambda item: abs(item[0] - tick.wall_time_s)
            if math.isfinite(item[0]) else float("inf"),
            default=(float("nan"), {}),
        )[1]
        try:
            age_value = float(nearest_row.get("age", "nan"))
        except (TypeError, ValueError):
            age_value = float("nan")
        if math.isfinite(age_value):
            ages.append(age_value)
        delta = camera_depth_delta_mm(tick)
        if math.isfinite(delta):
            depth_delta.append(delta)
        pred = tick.prediction if isinstance(tick.prediction, dict) else {}
        if pred.get("predicted", False) or pred.get("positionPredicted", False):
            predicted += 1
        if not tick.raw_valid or not tick.valid:
            loss_like += 1
    valid_ratio = (
        float(np.count_nonzero(recorded.valid & window))
        / float(max(1, np.count_nonzero(window))))
    reference = run.reference
    recorded_metrics = None
    if reference is not None:
        recorded_metrics = metric_with_z(
            recorded, reference, run.phase.contacts_s)
    classification = "observation_not_primary_from_telemetry"
    reasons: list[str] = []
    if valid_ratio < 0.70:
        classification = "observation_likely_contributor"
        reasons.append("low terminal valid ratio")
    if percentile(ages, 95.0) is not None and float(percentile(ages, 95.0)) > 0.060:
        classification = "observation_possible_contributor"
        reasons.append("stale terminal ball observation")
    if percentile_abs(depth_delta, 95.0) is not None and float(percentile_abs(depth_delta, 95.0)) > 60.0:
        if classification == "observation_not_primary_from_telemetry":
            classification = "observation_possible_contributor"
        reasons.append("large sensor-vs-geometry depth disagreement")
    if recorded_metrics and recorded_metrics.get("zErrorP95Mm") is not None:
        if float(recorded_metrics["zErrorP95Mm"]) > 55.0:
            if classification == "observation_not_primary_from_telemetry":
                classification = "observation_possible_contributor"
            reasons.append("large z error against smooth-flight reference")
    if predicted:
        reasons.append("prediction/handover active in terminal window")
    if loss_like:
        reasons.append("invalid/loss-like samples in terminal window")
    if not reasons:
        reasons.append("terminal telemetry did not show a dominant ball-observation fault")
    return {
        "cohort": run.cohort,
        "run": run.run_dir.name,
        "classification": classification,
        "reasons": reasons,
        "phaseEndReason": run.phase.reason,
        "contactCount": len(run.phase.contacts_s),
        "terminalWallTimeS": round(float(terminal_time), 6),
        "terminalValidRatio": round(valid_ratio, 6),
        "terminalAgeP95S": percentile(ages, 95.0),
        "terminalDepthDeltaP95AbsMm": percentile_abs(depth_delta, 95.0),
        "terminalPredictionRows": predicted,
        "terminalInvalidRows": loss_like,
    }


def flatten_metric_row(
    run: RunAnalysis,
    variant_name: str,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    phase = run.phase
    row: dict[str, Any] = {
        "cohort": run.cohort,
        "run": run.run_dir.name,
        "variant": variant_name,
        "eligible": run.exclusion_reason is None,
        "exclusionReason": run.exclusion_reason,
        "phaseDurationS": None if phase is None else round(phase.duration_s, 6),
        "contactCount": 0 if phase is None else len(phase.contacts_s),
    }
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[key] = value
    for key, value in (diagnostics or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"diag_{key}"] = value
    return row


def aggregate_variant_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("eligible"):
            by_variant[str(row["variant"])].append(row)
    summary: dict[str, Any] = {}
    keys = [
        "positionErrorP95Mm", "zErrorP95Mm", "velocityErrorP95Mps",
        "vzErrorP95AbsMps", "positionJumpCount", "velocityJumpCount",
        "validRatio", "positionHighFrequencyP95Mm",
    ]
    for variant, variant_rows in sorted(by_variant.items()):
        entry = {"runCount": len(variant_rows)}
        for key in keys:
            entry[f"{key}Median"] = percentile(
                [row.get(key) for row in variant_rows], 50.0)
        summary[variant] = entry
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_outputs(
    out_dir: Path,
    run_metric_rows: list[dict[str, Any]],
    contact_rows: list[dict[str, Any]],
    terminal_rows: list[dict[str, Any]],
) -> list[str]:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        (out_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")
        return outputs

    variants = sorted({row["variant"] for row in run_metric_rows if row.get("eligible")})
    cohorts = sorted({row["cohort"] for row in run_metric_rows if row.get("eligible")})
    if variants and cohorts:
        matrix = np.full((len(variants), len(cohorts)), np.nan, dtype=np.float64)
        for vi, variant in enumerate(variants):
            for ci, cohort in enumerate(cohorts):
                vals = [
                    row.get("zErrorP95Mm") for row in run_metric_rows
                    if row.get("eligible") and row["variant"] == variant
                    and row["cohort"] == cohort
                ]
                value = percentile(vals, 50.0)
                if value is not None:
                    matrix[vi, ci] = value
        fig, ax = plt.subplots(1, 1, figsize=(10, max(4, 0.42 * len(variants))))
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(cohorts, rotation=20, ha="right")
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        ax.set_title("Median per-run z error p95 by cohort and variant")
        ax.set_ylabel("variant")
        ax.set_xlabel("cohort")
        fig.colorbar(image, ax=ax, label="z p95 error (mm)")
        fig.tight_layout()
        path = figures_dir / "cohort_variant_metric_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.relative_to(out_dir)))

    contact_by_variant = defaultdict(list)
    for row in contact_rows:
        value = row.get("zGapMm")
        if value is not None:
            contact_by_variant[row["variant"]].append(float(value))
    if contact_by_variant:
        labels = sorted(contact_by_variant.keys())
        data = [contact_by_variant[label] for label in labels]
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.boxplot(data, labels=labels, vert=True, showfliers=False)
        ax.axhline(BALL_RADIUS_M * 1000.0, color="#ef4444", linestyle="--",
                   linewidth=1.0, label="ball radius")
        ax.axhline(-BALL_RADIUS_M * 1000.0, color="#ef4444", linestyle="--",
                   linewidth=1.0)
        ax.set_title("Ball-racket z gap at detected contacts")
        ax.set_ylabel("z gap (mm)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        path = figures_dir / "contact_window_z_gap_boxplots.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.relative_to(out_dir)))

    class_counts = Counter(row.get("classification") for row in terminal_rows)
    if class_counts:
        labels = list(class_counts.keys())
        values = [class_counts[label] for label in labels]
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        ax.bar(range(len(labels)), values, color="#2a78d6")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title("Terminal failure observation classification")
        ax.set_ylabel("runs")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = figures_dir / "terminal_failure_classification.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.relative_to(out_dir)))
    return outputs


def select_representative_runs(
    run_metric_rows: list[dict[str, Any]],
    terminal_rows: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    recorded = [
        row for row in run_metric_rows
        if row.get("eligible") and row.get("variant") == "recorded_policy_state"
        and row.get("zErrorP95Mm") is not None
    ]
    recorded.sort(key=lambda row: float(row["zErrorP95Mm"]), reverse=True)
    selected: list[tuple[str, str]] = []
    for row in recorded[:6]:
        selected.append((str(row["cohort"]), str(row["run"])))
    terminal_by_run = {(row.get("cohort"), row.get("run")): row for row in terminal_rows}
    for key, row in terminal_by_run.items():
        if row.get("classification") == "observation_likely_contributor":
            pair = (str(key[0]), str(key[1]))
            if pair not in selected:
                selected.append(pair)
    return selected[:8]


def plot_representative_timelines(
    out_dir: Path,
    analyses: list[RunAnalysis],
    series_by_run: dict[tuple[str, str], dict[str, VariantSeries]],
    selected: list[tuple[str, str]],
) -> list[str]:
    figures_dir = out_dir / "figures"
    outputs: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return outputs
    by_key = {(run.cohort, run.run_dir.name): run for run in analyses}
    for cohort, run_name in selected:
        run = by_key.get((cohort, run_name))
        if run is None or run.phase is None:
            continue
        series_map = series_by_run.get((cohort, run_name), {})
        if not series_map:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        for name in [
                "recorded_policy_state", "raw_finite_difference",
                "causal_alpha_beta_default",
                "state_replay_recommended"]:
            series = series_map.get(name)
            if series is None:
                continue
            color = SERIES_COLORS.get(name, None)
            axes[0].plot(series.times_s, series.positions_m[:, 2],
                         label=name, linewidth=1.0, color=color)
            axes[1].plot(series.times_s, series.velocities_m_s[:, 2],
                         label=name, linewidth=1.0, color=color)
        ticks = selected_ticks(run)
        racket_times = np.asarray([tick.wall_time_s for tick in ticks], dtype=np.float64)
        racket_z = np.asarray([tick.racket_m[2] for tick in ticks], dtype=np.float64)
        axes[0].plot(racket_times, racket_z, label="racket z",
                     color="#0b0b0b", linestyle="--", linewidth=1.0)
        ages = np.asarray([
            float(raw_row_nearest_time(run, tick.wall_time_s).get("age", float("nan")))
            for tick in ticks
        ], dtype=np.float64)
        depth_delta = np.asarray([camera_depth_delta_mm(tick) for tick in ticks], dtype=np.float64)
        axes[2].plot(racket_times, ages * 1000.0, label="age (ms)",
                     color="#2a78d6", linewidth=1.0)
        axes[2].plot(racket_times, np.abs(depth_delta),
                     label="|sensor-geometry depth| (mm)",
                     color="#ef4444", linewidth=1.0)
        for contact in run.phase.contacts_s:
            for ax in axes:
                ax.axvline(contact, color="#a855f7", linewidth=0.9, alpha=0.35)
        if run.phase.miss_time_s is not None:
            for ax in axes:
                ax.axvline(run.phase.miss_time_s, color="#ef4444",
                           linewidth=1.1, linestyle=":", alpha=0.65)
        axes[0].set_ylabel("z position (m)")
        axes[1].set_ylabel("vz (m/s)")
        axes[2].set_ylabel("diagnostic")
        axes[2].set_xlabel("wall time (s)")
        axes[0].set_title(f"{cohort}/{run_name}: ball observation timeline")
        for ax in axes:
            ax.grid(True, alpha=0.22)
        axes[0].legend(loc="upper right", ncol=2, fontsize=7)
        axes[2].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        safe = f"{cohort}_{run_name}_timeline.png"
        path = figures_dir / safe
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.relative_to(out_dir)))
    return outputs


def write_report(
    out_dir: Path,
    manifest: dict[str, Any],
    corpus_summary: dict[str, Any],
    run_metric_rows: list[dict[str, Any]],
    terminal_rows: list[dict[str, Any]],
    figures: list[str],
) -> None:
    aggregate = corpus_summary["variantAggregate"]
    best = sorted(
        aggregate.items(),
        key=lambda item: float("inf") if item[1].get("zErrorP95MmMedian") is None
        else float(item[1]["zErrorP95MmMedian"]),
    )
    classifications = Counter(row.get("classification") for row in terminal_rows)
    lines = [
        "# record_new3-5 球观测离线系统分析",
        "",
        "## 范围与限制",
        "",
        "- 本报告只使用已记录的 `control_samples.jsonl` 等状态遥测。",
        "- 没有修改 `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh`。",
        "- 当前记录未发现可完整重跑 raw IR/depth extraction 的 `.npy/.npz/.bag/.db3` 原始数据；因此这里是 downstream 3D/state/filter counterfactual，不是原始视觉重跑。",
        f"- 发现 {manifest['runCount']} 个 recording，其中 {manifest['eligibleRunCount']} 个有可分析颠球阶段。",
        "",
        "## Variant aggregate",
        "",
        "| variant | runs | z p95 median (mm) | pos p95 median (mm) | vel p95 median (m/s) | valid ratio median | jumps median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in best:
        lines.append(
            f"| {name} | {metrics.get('runCount')} | "
            f"{metrics.get('zErrorP95MmMedian')} | "
            f"{metrics.get('positionErrorP95MmMedian')} | "
            f"{metrics.get('velocityErrorP95MpsMedian')} | "
            f"{metrics.get('validRatioMedian')} | "
            f"{metrics.get('positionJumpCountMedian')} |")
    lines.extend([
        "",
        "## Terminal classification",
        "",
    ])
    for name, count in classifications.most_common():
        lines.append(f"- {name}: {count}")
    lines.extend([
        "",
        "## 初步判断规则",
        "",
        "- 如果 recorded/raw/causal variants 在终端前同时恶化，且 strict/outlier-gated variants 改善明显，则球观测更可能是 contributor。",
        "- 如果 recorded 与 raw/ref 接近，而失败仍发生，应优先检查接触模型、球拍 site/frame、执行器时延或策略动作。",
        "- 如果 sensor depth 与 geometry depth 在终端/接触窗口分歧大，应优先检查 depth 选择和标定，而不是只调后端滤波。",
        "- acausal reference 只用于评分，不能作为可部署方案。",
        "",
        "## Figures",
        "",
    ])
    lines.extend(f"- `{figure}`" for figure in figures)
    lines.extend([
        "",
        "## 输出文件",
        "",
        "- `manifest.json`: run/capability/provenance。",
        "- `corpus_summary.json`: aggregate metrics。",
        "- `run_metrics.jsonl` / `run_metrics.csv`: run × variant 指标。",
        "- `contact_metrics.jsonl` / `contact_metrics.csv`: contact × variant 指标。",
        "- `terminal_failure_metrics.jsonl` / `terminal_failure_metrics.csv`: 终端失败分类。",
        "- `figures/`: 汇总图和代表 run 时间线。",
    ])
    (out_dir / "corpus_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def analyze_corpus(roots: list[Path], out_dir: Path, limit: int = 0) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = discover_recordings(roots)
    if limit > 0:
        recordings = recordings[:limit]
    analyses: list[RunAnalysis] = []
    capabilities: list[dict[str, Any]] = []
    run_metric_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    series_by_run: dict[tuple[str, str], dict[str, VariantSeries]] = {}

    for cohort, run_dir in recordings:
        run = build_run_analysis(cohort, run_dir)
        analyses.append(run)
        cap = capability_for_run(run_dir, run.raw_rows)
        cap.update({
            "cohort": cohort,
            "run": run_dir.name,
            "path": str(run_dir),
            "eligible": run.exclusion_reason is None,
            "exclusionReason": run.exclusion_reason,
            "phaseDurationS": None if run.phase is None else run.phase.duration_s,
            "contactCount": 0 if run.phase is None else len(run.phase.contacts_s),
        })
        capabilities.append(cap)
        if run.exclusion_reason is not None or run.phase is None or run.reference is None:
            terminal_rows.append(terminal_metrics_for_run(run, {}))
            continue
        ticks = selected_ticks(run)
        per_run_series: dict[str, VariantSeries] = {}
        for variant in OBSERVATION_VARIANTS:
            series = build_observation_series(ticks, variant)
            per_run_series[series.name] = series
            metrics = metric_with_z(series, run.reference, run.phase.contacts_s)
            run_metric_rows.append(
                flatten_metric_row(run, series.name, metrics, series.diagnostics))
            contact_rows.extend(contact_metrics_for_variant(run, series))
        for pipeline_variant in traj.PIPELINE_VARIANTS:
            if pipeline_variant.name != "recommended":
                continue
            replay = traj.replay_pipeline(run.refreshes, run.phase, pipeline_variant)
            replay_series = VariantSeries(
                "state_replay_recommended",
                np.asarray([row["timeS"] for row in replay["rows"]], dtype=np.float64),
                np.asarray([
                    row["positionM"] if row.get("positionM") is not None
                    else [np.nan, np.nan, np.nan]
                    for row in replay["rows"]
                ], dtype=np.float64),
                np.asarray([
                    row["velocityMps"] if row.get("velocityMps") is not None
                    else [np.nan, np.nan, np.nan]
                    for row in replay["rows"]
                ], dtype=np.float64),
                np.asarray([bool(row.get("valid", False)) for row in replay["rows"]],
                           dtype=bool),
                {"rejected": replay.get("rejected"), "losses": replay.get("losses")},
            )
            per_run_series[replay_series.name] = replay_series
            metrics = metric_with_z(
                replay_series, run.reference, run.phase.contacts_s)
            run_metric_rows.append(
                flatten_metric_row(run, replay_series.name, metrics,
                                   replay_series.diagnostics))
            contact_rows.extend(contact_metrics_for_variant(run, replay_series))
        series_by_run[(cohort, run_dir.name)] = per_run_series
        terminal_rows.append(terminal_metrics_for_run(run, per_run_series))

    manifest = {
        "schemaVersion": 1,
        "scope": "record_new3-5 downstream ball-observation telemetry replay",
        "roots": [str(path.expanduser().resolve()) for path in roots],
        "runCount": len(recordings),
        "eligibleRunCount": sum(1 for run in analyses if run.exclusion_reason is None),
        "fullRawVisionRerunAvailableRunCount": sum(
            1 for cap in capabilities if cap.get("fullRawVisionRerunAvailable")),
        "variants": [variant.__dict__ for variant in OBSERVATION_VARIANTS]
        + [{"name": "state_replay_recommended", "kind": "existing_detector_state_replay"}],
        "limitations": [
            "Lossy annotated frames are not sufficient to rerun raw IR contour extraction or 16-bit depth sampling.",
            "Acausal robust reference is a scoring aid, not an online deployable trajectory.",
            "Offline forced observation variants indicate telemetry consistency, not guaranteed real-world success.",
        ],
        "runs": capabilities,
    }
    corpus_summary = {
        "schemaVersion": 1,
        "runCount": len(recordings),
        "eligibleRunCount": manifest["eligibleRunCount"],
        "cohorts": sorted(set(cohort for cohort, _ in recordings)),
        "variantAggregate": aggregate_variant_rows(run_metric_rows),
        "terminalClassificationCounts": dict(Counter(
            row.get("classification") for row in terminal_rows)),
        "excludedRuns": [
            {"cohort": run.cohort, "run": run.run_dir.name,
             "reason": run.exclusion_reason}
            for run in analyses if run.exclusion_reason is not None
        ],
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    (out_dir / "corpus_summary.json").write_text(
        json.dumps(corpus_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    write_jsonl(out_dir / "run_metrics.jsonl", run_metric_rows)
    write_csv(out_dir / "run_metrics.csv", run_metric_rows)
    write_jsonl(out_dir / "contact_metrics.jsonl", contact_rows)
    write_csv(out_dir / "contact_metrics.csv", contact_rows)
    write_jsonl(out_dir / "terminal_failure_metrics.jsonl", terminal_rows)
    write_csv(out_dir / "terminal_failure_metrics.csv", terminal_rows)
    figures = plot_outputs(out_dir, run_metric_rows, contact_rows, terminal_rows)
    selected = select_representative_runs(run_metric_rows, terminal_rows)
    figures.extend(plot_representative_timelines(
        out_dir, analyses, series_by_run, selected))
    write_report(out_dir, manifest, corpus_summary, run_metric_rows,
                 terminal_rows, figures)
    return {
        "outDir": str(out_dir),
        "runs": len(recordings),
        "eligible": manifest["eligibleRunCount"],
        "figures": figures,
        "terminalClassificationCounts": corpus_summary[
            "terminalClassificationCounts"],
    }


def default_roots() -> list[Path]:
    base = (PACKAGE_ROOT / "pingpong_controller" / "outputs"
            / "interactive_recordings")
    return [base / "record_new3", base / "record_new4", base / "record_new5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", type=Path, default=default_roots())
    parser.add_argument("--output-dir", type=Path, default=(
        Path("analysis_results") / "ball_observation_corpus_20260810_13"))
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = analyze_corpus(
        [path.expanduser().resolve() for path in args.roots],
        args.output_dir.expanduser().resolve(),
        limit=max(0, int(args.limit or 0)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
