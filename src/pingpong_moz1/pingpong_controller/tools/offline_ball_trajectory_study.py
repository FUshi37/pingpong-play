#!/usr/bin/env python3
"""Audit recorded juggling trajectories and replay detector state variants.

The interactive recorder contains the exact pre-motion-gate transformed 3D
candidate, the detector decision, the policy-facing 6D ball state and racket
FK at every control tick.  This tool uses those fields to:

* discover non-duplicated recordings for one or more dates;
* keep only the policy-controlled juggling interval and cut the terminal miss;
* build an acausal, robust interpolated trajectory used only as a reference;
* replay the production HybridBallDetector motion/velocity state machine with
  several switch combinations; and
* report position/velocity discontinuities against identical intervals.

It deliberately does not claim to rerun IR contour extraction or stereo depth
sampling.  Annotated JPEG recordings are lossy 8-bit composites and do not
contain the original 16-bit depth image.  The downstream 3D/state replay is
deterministic because rejected candidates are recorded before the motion gate.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# Importing the detector normally imports ultralytics/torch even though state
# replay never constructs YOLO.  Keep this CPU-only analysis independent of a
# CUDA installation, matching the detector unit tests.
if "ultralytics" not in sys.modules:
    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = object
    sys.modules["ultralytics"] = ultralytics_stub

from pingpong_controller.hybrid_ball_detector import HybridBallDetector  # noqa: E402


SIM_X_OFFSET_M = 0.0
SIM_Y_OFFSET_M = 0.023583
SIM_BASE_HEIGHT_M = 1.302396
MIN_PHASE_S = 0.25
TERMINAL_TAIL_GUARD_S = 0.12
MAX_REFERENCE_GAP_S = 0.22
CONTACT_EXCLUSION_S = 0.065
POSITION_JUMP_SPEED_M_S = 6.0
VELOCITY_JUMP_M_S = 2.0


def _vector(value: Any, length: int, fill: float = math.nan) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        array = np.zeros(0, dtype=np.float64)
    result = np.full(length, fill, dtype=np.float64)
    count = min(length, array.size)
    if count:
        result[:count] = array[:count]
    return result


def _finite_vector(value: Any, length: int) -> np.ndarray | None:
    vector = _vector(value, length)
    return vector if np.all(np.isfinite(vector)) else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return round(float(np.percentile(array, percentile)), 6)


def _rms(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return round(float(np.sqrt(np.mean(array * array))), 6)


def _rounded_vector(values: np.ndarray, digits: int = 6) -> list[float]:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


@dataclass
class Tick:
    source_time_s: float
    wall_time_s: float
    source_seq: int
    observation_seq: int
    camera_stamp_s: float | None
    mode: str
    policy_executed: bool
    raw_valid: bool
    valid: bool
    raw_ball_m: np.ndarray
    measured_ball_m: np.ndarray
    measured_velocity_m_s: np.ndarray
    observed_ball_m: np.ndarray
    observed_velocity_m_s: np.ndarray
    racket_m: np.ndarray
    arm_q_rad: np.ndarray
    measurement: dict[str, Any]
    prediction: dict[str, Any]


@dataclass
class Phase:
    start_s: float
    end_s: float
    start_source_time_s: float
    end_source_time_s: float
    reason: str
    contacts_s: list[float]
    miss_time_s: float | None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class ReferenceTrajectory:
    times_s: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    input_valid: np.ndarray
    outlier: np.ndarray
    interpolated: np.ndarray


@dataclass(frozen=True)
class PipelineVariant:
    name: str
    clear_on_loss: bool
    motion_gate: bool
    physics_velocity: bool
    position_smoothing: bool
    impact_reset: bool
    velocity_update_gain: float = 0.65
    velocity_correction_limit_mm_s: float = 450.0
    position_correction_limit_mm: float = 12.0
    adaptive_velocity_gain: bool = False
    position_output_speed_limit_mm_s: float = 1.0e12


PIPELINE_VARIANTS = (
    PipelineVariant(
        "legacy_endpoint", False, False, False, False, False),
    PipelineVariant(
        "impact_endpoint", True, False, False, False, True),
    PipelineVariant(
        "physics_only", True, False, True, False, True),
    PipelineVariant(
        "physics_gate", True, True, True, False, True),
    PipelineVariant(
        "current_latest", True, True, True, True, True),
    PipelineVariant(
        "position_smooth_20mm", True, True, True, True, True,
        position_correction_limit_mm=20.0,
        position_output_speed_limit_mm_s=6000.0),
    PipelineVariant(
        "velocity_gain_055", True, True, True, True, True,
        velocity_update_gain=0.55,
        position_output_speed_limit_mm_s=6000.0),
    PipelineVariant(
        "recommended", True, True, True, True, True,
        position_output_speed_limit_mm_s=6000.0),
)


def load_ticks(run_dir: Path) -> list[Tick]:
    path = run_dir / "control_samples.jsonl"
    ticks: list[Tick] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            camera_stamp = row.get("cameraStampS")
            ros = row.get("ros") or {}
            arm_q = _vector(ros.get("jointPosition"), 7)
            if not np.all(np.isfinite(arm_q)):
                arm_q = _vector(row.get("policyObs"), 67)[:7]
            ticks.append(Tick(
                source_time_s=float(row.get("sourceTimeS", 0.0) or 0.0),
                wall_time_s=float(row.get("wallTimeS", 0.0) or 0.0),
                source_seq=int(row.get("sourceSeq", 0) or 0),
                observation_seq=int(row.get("ballObservationSeq", 0) or 0),
                camera_stamp_s=(
                    None if camera_stamp is None else float(camera_stamp)),
                mode=str(row.get("mode", "")),
                policy_executed=bool(row.get("policyExecuted", False)),
                raw_valid=bool(row.get("rawValid", False)),
                valid=bool(row.get("valid", False)),
                raw_ball_m=_vector(row.get("rawBall"), 3),
                measured_ball_m=_vector(row.get("measuredBall"), 3),
                measured_velocity_m_s=_vector(
                    row.get("measuredVelocity"), 3),
                observed_ball_m=_vector(row.get("observedBall"), 3),
                observed_velocity_m_s=_vector(
                    row.get("observedVelocity"), 3),
                racket_m=_vector(row.get("racket"), 3),
                arm_q_rad=arm_q,
                measurement=dict(row.get("cameraMeasurement") or {}),
                prediction=dict(
                    row.get("highBallOcclusionPrediction") or {}),
            ))
    return ticks


def fresh_camera_ticks(ticks: list[Tick]) -> list[Tick]:
    result: list[Tick] = []
    previous_stamp: float | None = None
    for tick in ticks:
        stamp = tick.camera_stamp_s
        if stamp is None:
            continue
        if previous_stamp is not None and abs(stamp - previous_stamp) <= 1e-6:
            continue
        previous_stamp = stamp
        result.append(tick)
    return result


def discover_runs(root: Path, dates: list[str]) -> list[Path]:
    """Discover top-level recordings only; record_new3/new4 are duplicates."""
    runs: list[Path] = []
    for date in dates:
        runs.extend(root.glob(f"probe_recording_{date}_*"))
    return sorted(
        path.resolve() for path in set(runs)
        if path.joinpath("control_samples.jsonl").is_file())


def _camera_position_plausible(position_m: np.ndarray) -> bool:
    if not np.all(np.isfinite(position_m)):
        return False
    x, y, z = position_m
    return bool(-0.80 <= x <= 0.60 and -1.16 <= y <= 0.22
                and 0.20 <= z <= 2.10)


def detect_contacts(refreshes: list[Tick]) -> list[float]:
    """Detect downward-to-upward reversals near the recorded racket FK."""
    trusted = [
        tick for tick in refreshes
        if tick.raw_valid
        and _camera_position_plausible(tick.measured_ball_m)
        and np.all(np.isfinite(tick.measured_velocity_m_s))
    ]
    contacts: list[float] = []
    last_time = -1.0e30
    for index in range(1, len(trusted)):
        after = trusted[index]
        recent = trusted[max(0, index - 5):index]
        if not recent:
            continue
        before = min(recent, key=lambda tick: tick.measured_velocity_m_s[2])
        v_before = before.measured_velocity_m_s
        v_after = after.measured_velocity_m_s
        if not (v_before[2] < -0.10 and v_after[2] > 0.10
                and v_after[2] - v_before[2] >= 0.70):
            continue
        if after.wall_time_s - last_time < 0.16:
            continue
        relative = after.measured_ball_m - after.racket_m
        geometry_ok = bool(
            np.linalg.norm(relative[:2]) <= 0.095
            and -0.035 <= relative[2] <= 0.120)
        before_relative = before.measured_ball_m - before.racket_m
        held_contact = bool(
            after.wall_time_s - before.wall_time_s <= 0.09
            and np.linalg.norm(before_relative) <= 0.12)
        detector_contact = bool(
            after.measurement.get("velocity_impact_detected", False))
        if not (geometry_ok or held_contact or detector_contact):
            continue
        contacts.append(after.wall_time_s)
        last_time = after.wall_time_s
    return contacts


def _terminal_miss_time(
    refreshes: list[Tick], start_s: float, contacts_s: list[float],
) -> float | None:
    """Find the first final downward racket-plane crossing without contact."""
    search_start = contacts_s[-1] + 0.12 if contacts_s else start_s + 0.20
    candidates: list[float] = []
    for tick in refreshes:
        if tick.wall_time_s < search_start or not tick.raw_valid:
            continue
        position = tick.measured_ball_m
        velocity = tick.measured_velocity_m_s
        racket = tick.racket_m
        if not (np.all(np.isfinite(position))
                and np.all(np.isfinite(velocity))
                and np.all(np.isfinite(racket))):
            continue
        relative = position - racket
        if (velocity[2] <= -0.20
                and -0.10 <= relative[2] <= 0.10
                and np.linalg.norm(relative[:2]) >= 0.105):
            candidates.append(tick.wall_time_s)
            break

    # Gross physically impossible observations are terminal even when the
    # detector velocity itself is corrupted and cannot identify the crossing.
    for previous, current in zip(refreshes, refreshes[1:]):
        if current.wall_time_s < search_start:
            continue
        if current.raw_valid and not _camera_position_plausible(
                current.measured_ball_m):
            candidates.append(current.wall_time_s)
            break
        dt = current.wall_time_s - previous.wall_time_s
        if (current.raw_valid and previous.raw_valid and dt > 1e-4
                and np.all(np.isfinite(current.measured_ball_m))
                and np.all(np.isfinite(previous.measured_ball_m))
                and np.linalg.norm(
                    current.measured_ball_m - previous.measured_ball_m) / dt
                > 10.0):
            candidates.append(current.wall_time_s)
            break
    return min(candidates) if candidates else None


def select_valid_phase(ticks: list[Tick]) -> Phase | None:
    policy = [tick for tick in ticks if tick.policy_executed]
    if not policy:
        return None
    start = policy[0]
    final = policy[-1]
    refreshes = [
        tick for tick in fresh_camera_ticks(ticks)
        if start.wall_time_s <= tick.wall_time_s <= final.wall_time_s
    ]
    contacts = detect_contacts(refreshes)
    miss_time = _terminal_miss_time(refreshes, start.wall_time_s, contacts)
    guarded_end = final.wall_time_s - TERMINAL_TAIL_GUARD_S
    if miss_time is not None:
        end_s = min(guarded_end, miss_time)
        reason = "terminal_miss"
    else:
        end_s = guarded_end
        reason = "terminal_tail_guard"
    if end_s - start.wall_time_s < MIN_PHASE_S:
        return None
    end_tick = min(policy, key=lambda tick: abs(tick.wall_time_s - end_s))
    return Phase(
        start_s=start.wall_time_s,
        end_s=end_s,
        start_source_time_s=start.source_time_s,
        end_source_time_s=end_tick.source_time_s,
        reason=reason,
        contacts_s=[
            value for value in contacts if start.wall_time_s <= value <= end_s
        ],
        miss_time_s=miss_time,
    )


def _weighted_polynomial(
    times_s: np.ndarray,
    values: np.ndarray,
    query_s: float,
    degree: int,
) -> tuple[float, float]:
    relative = np.asarray(times_s, dtype=np.float64) - float(query_s)
    target = np.asarray(values, dtype=np.float64)
    if len(relative) <= degree:
        degree = max(0, len(relative) - 1)
    design = np.column_stack([
        relative ** power for power in range(degree + 1)
    ])
    weights = np.ones(len(relative), dtype=np.float64)
    coefficient = np.zeros(degree + 1, dtype=np.float64)
    for _ in range(4):
        root_weight = np.sqrt(np.maximum(weights, 1e-6))
        coefficient, *_ = np.linalg.lstsq(
            design * root_weight[:, None], target * root_weight, rcond=None)
        residual = target - design @ coefficient
        scale = 1.4826 * np.median(np.abs(
            residual - np.median(residual))) + 1e-9
        normalized = np.abs(residual) / (2.5 * scale)
        weights = np.ones_like(normalized)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
    position = float(coefficient[0])
    velocity = float(coefficient[1]) if coefficient.size > 1 else 0.0
    return position, velocity


def _initial_outlier_mask(
    times_s: np.ndarray, positions_m: np.ndarray, valid: np.ndarray,
) -> np.ndarray:
    outlier = ~valid.copy()
    for index, position in enumerate(positions_m):
        if valid[index] and not _camera_position_plausible(position):
            outlier[index] = True
    # Iterative leave-one-out interpolation catches isolated depth/XYZ spikes
    # without classifying a real continuous fast flight as an outlier.
    for _ in range(3):
        good = ~outlier
        changed = False
        for index in range(1, len(times_s) - 1):
            if not good[index]:
                continue
            previous = index - 1
            while previous >= 0 and not good[previous]:
                previous -= 1
            following = index + 1
            while following < len(times_s) and not good[following]:
                following += 1
            if previous < 0 or following >= len(times_s):
                continue
            span = times_s[following] - times_s[previous]
            if not 1e-4 < span <= 0.075:
                continue
            alpha = (times_s[index] - times_s[previous]) / span
            expected = (
                (1.0 - alpha) * positions_m[previous]
                + alpha * positions_m[following])
            residual = float(np.linalg.norm(positions_m[index] - expected))
            speed_in = np.linalg.norm(
                positions_m[index] - positions_m[previous]) / max(
                    times_s[index] - times_s[previous], 1e-6)
            speed_out = np.linalg.norm(
                positions_m[following] - positions_m[index]) / max(
                    times_s[following] - times_s[index], 1e-6)
            if (residual > 0.035
                    and max(speed_in, speed_out) > POSITION_JUMP_SPEED_M_S):
                outlier[index] = True
                changed = True
        if not changed:
            break
    return outlier


def build_reference(
    refreshes: list[Tick], phase: Phase,
) -> ReferenceTrajectory:
    selected = [
        tick for tick in refreshes
        if phase.start_s <= tick.wall_time_s <= phase.end_s
    ]
    times = np.asarray([tick.wall_time_s for tick in selected])
    positions = np.asarray([tick.observed_ball_m for tick in selected])
    valid = np.asarray([
        tick.valid and np.all(np.isfinite(tick.observed_ball_m))
        for tick in selected], dtype=bool)
    outlier = _initial_outlier_mask(times, positions, valid)
    usable = ~outlier
    interpolated = outlier.copy()
    filled = positions.copy()
    for axis in range(3):
        if np.count_nonzero(usable) < 2:
            filled[:, axis] = 0.0
            continue
        filled[:, axis] = np.interp(
            times, times[usable], positions[usable, axis])

    # Robust local endpoint fits smooth camera-frequency noise without a
    # moving-average phase shift.  Contacts split the fit window so the real
    # vertical velocity reversal is retained.
    clean_position = np.zeros_like(filled)
    clean_velocity = np.zeros_like(filled)
    contacts = np.asarray(phase.contacts_s, dtype=np.float64)
    for index, query in enumerate(times):
        same_segment = np.ones(len(times), dtype=bool)
        if contacts.size:
            segment_id = int(np.searchsorted(contacts, query, side="right"))
            sample_ids = np.searchsorted(contacts, times, side="right")
            same_segment = sample_ids == segment_id
        candidates = np.flatnonzero(
            same_segment & (np.abs(times - query) <= 0.045))
        if candidates.size < 5:
            candidates = np.flatnonzero(
                same_segment & (np.abs(times - query) <= 0.075))
        if candidates.size < 2:
            clean_position[index] = filled[index]
            if index:
                dt = times[index] - times[index - 1]
                if dt > 1e-6:
                    clean_velocity[index] = (
                        clean_position[index] - clean_position[index - 1]) / dt
            continue
        for axis in range(3):
            degree = 2 if axis == 2 and candidates.size >= 3 else 1
            position, velocity = _weighted_polynomial(
                times[candidates], filled[candidates, axis], query, degree)
            clean_position[index, axis] = position
            clean_velocity[index, axis] = velocity
    return ReferenceTrajectory(
        times_s=times,
        position_m=clean_position,
        velocity_m_s=clean_velocity,
        input_valid=valid,
        outlier=outlier,
        interpolated=interpolated,
    )


def interpolate_reference(
    reference: ReferenceTrajectory, query_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.column_stack([
        np.interp(query_times_s, reference.times_s,
                  reference.position_m[:, axis])
        for axis in range(3)
    ])
    velocity = np.column_stack([
        np.interp(query_times_s, reference.times_s,
                  reference.velocity_m_s[:, axis])
        for axis in range(3)
    ])
    return position, velocity


def _make_state_detector(variant: PipelineVariant) -> HybridBallDetector:
    detector = HybridBallDetector.__new__(HybridBallDetector)
    detector.output_frame = "base_link"
    detector.fps = 90
    detector.pixel_kalman_enabled = False
    detector.pixel_kalman_process_noise_pos = 0.3
    detector.pixel_kalman_process_noise_vel = 3.0
    detector.pixel_kalman_measurement_noise = 1.2
    detector.pixel_kalman_innovation_gate_px = 15.0
    detector.pixel_kalman_reset_after_outliers = 3
    detector.ir_depth_kalman_enabled = False
    detector.synchronized_depth_enabled = False
    detector.depth_temporal_window_frames = 3
    detector.velocity_least_squares_enabled = False
    detector.velocity_window_frames = 3
    detector.clear_buffers_on_detection_loss_enabled = variant.clear_on_loss
    detector.motion_consistency_gate_enabled = variant.motion_gate
    detector.physics_constrained_velocity_enabled = variant.physics_velocity
    detector.trajectory_position_smoothing_enabled = variant.position_smoothing
    detector.impact_velocity_reset_enabled = variant.impact_reset
    detector.impact_min_downward_velocity_mm_s = 250.0
    detector.impact_min_upward_velocity_mm_s = 250.0
    detector.impact_min_velocity_jump_mm_s = 800.0
    detector.impact_refractory_s = 0.08
    detector.impact_racket_gate_required = True
    detector.FLIGHT_VELOCITY_UPDATE_GAIN = float(
        variant.velocity_update_gain)
    detector.FLIGHT_VELOCITY_MAX_CORRECTION_MM_S = float(
        variant.velocity_correction_limit_mm_s)
    detector.TRAJECTORY_POSITION_MAX_CORRECTION_MM = float(
        variant.position_correction_limit_mm)
    detector.FLIGHT_VELOCITY_ADAPTIVE_GAIN_ENABLED = bool(
        variant.adaptive_velocity_gain)
    detector.TRAJECTORY_POSITION_MAX_OUTPUT_SPEED_MM_S = float(
        variant.position_output_speed_limit_mm_s)
    detector.candidate_filter_enabled = False
    detector.candidate_min_brightness = 0.0
    detector.candidate_max_depth_disagreement_mm = 180.0
    detector.candidate_motion_scale_px = 130.0
    detector.depth_scale = 0.001
    detector.depth_neighborhood_px = 2
    detector.ball_diameter_mm = 40.0
    detector.depth_mode = "adaptive"
    detector.sensor_surface_to_center_depth_enabled = True
    detector.depth_correction_model_path = str(
        PACKAGE_ROOT / "pingpong_controller" / "depth_correction_model.pkl")
    # _init_runtime_state loads the geometry depth model and prints a startup
    # diagnostic intended for the live detector.  The state replay does not
    # use geometry depth; silence the repeated message for corpus runs.
    with contextlib.redirect_stdout(io.StringIO()):
        detector._init_runtime_state()
    return detector


def _pre_gate_position_mm(tick: Tick) -> np.ndarray | None:
    measurement = tick.measurement
    exact = [
        measurement.get(f"transformed_measurement_{axis}_mm")
        for axis in "xyz"
    ]
    if all(value is not None for value in exact):
        position = np.asarray(exact, dtype=np.float64)
        return position if np.all(np.isfinite(position)) else None
    if tick.raw_valid and np.all(np.isfinite(tick.raw_ball_m)):
        detector_m = tick.raw_ball_m.copy()
        detector_m[0] -= SIM_X_OFFSET_M
        detector_m[1] -= SIM_Y_OFFSET_M
        detector_m[2] -= SIM_BASE_HEIGHT_M
        return detector_m * 1000.0
    return None


def _candidate_available(tick: Tick) -> bool:
    measurement = tick.measurement
    state = str(measurement.get("observation_state") or "")
    if state in {"depth_fault", "detection_loss"}:
        return False
    if measurement.get("depth_sensor_rejected", False):
        return False
    return _pre_gate_position_mm(tick) is not None


def replay_pipeline(
    refreshes: list[Tick], phase: Phase, variant: PipelineVariant,
) -> dict[str, Any]:
    detector = _make_state_detector(variant)
    selected = [
        tick for tick in refreshes
        if phase.start_s <= tick.wall_time_s <= phase.end_s
    ]
    rows: list[dict[str, Any]] = []
    rejected = 0
    losses = 0
    for tick in selected:
        stamp = float(tick.camera_stamp_s or tick.wall_time_s)
        if np.all(np.isfinite(tick.racket_m)):
            racket_detector = tick.racket_m.copy()
            racket_detector[0] -= SIM_X_OFFSET_M
            racket_detector[1] -= SIM_Y_OFFSET_M
            racket_detector[2] -= SIM_BASE_HEIGHT_M
            detector.set_racket_state(racket_detector, stamp)
        else:
            detector.clear_racket_state()
        raw_position = _pre_gate_position_mm(tick)
        if raw_position is None or not _candidate_available(tick):
            reason = str(tick.measurement.get("observation_reason") or
                         tick.measurement.get("observation_state") or
                         "recorded_loss")
            detector._handle_observation_loss(reason)
            losses += 1
            rows.append({
                "timeS": tick.wall_time_s,
                "valid": False,
                "positionM": None,
                "velocityMps": None,
                "ready": False,
                "state": "loss",
            })
            continue
        if not detector._motion_measurement_consistent(stamp, raw_position):
            reason = "motion_consistency_" + str(
                detector.motion_last_gate_reason)
            detector._handle_observation_loss(reason, preserve_depth_guard=True)
            rejected += 1
            rows.append({
                "timeS": tick.wall_time_s,
                "valid": False,
                "positionM": None,
                "velocityMps": None,
                "ready": False,
                "state": "motion_outlier",
            })
            continue
        velocity_mm_s = np.asarray(
            detector._append_velocity_sample(stamp, raw_position),
            dtype=np.float64)
        output_mm = (
            detector.last_position_estimate.copy()
            if detector.last_position_estimate is not None
            else raw_position.copy())
        detector._remember_motion_state(
            stamp, output_mm, velocity_mm_s,
            velocity_ready=detector.last_velocity_ready)
        output_m = output_mm * 1e-3
        output_m[0] += SIM_X_OFFSET_M
        output_m[1] += SIM_Y_OFFSET_M
        output_m[2] += SIM_BASE_HEIGHT_M
        rows.append({
            "timeS": tick.wall_time_s,
            "valid": True,
            "positionM": output_m,
            "velocityMps": velocity_mm_s * 1e-3,
            "ready": bool(detector.last_velocity_ready),
            "state": str(detector.velocity_motion_state),
            "impact": bool(detector.last_velocity_impact_detected),
        })
    return {
        "rows": rows,
        "inputSamples": len(selected),
        "validSamples": sum(row["valid"] for row in rows),
        "readySamples": sum(row["ready"] for row in rows),
        "motionRejected": rejected,
        "lossSamples": losses,
        "impactConfirmed": int(detector.velocity_impact_confirmed_count),
        "impactRejected": int(detector.velocity_impact_rejected_count),
        "restartBridges": int(detector.velocity_restart_bridge_count),
    }


def _contact_mask(times_s: np.ndarray, contacts_s: list[float]) -> np.ndarray:
    mask = np.zeros(len(times_s), dtype=bool)
    for contact in contacts_s:
        mask |= np.abs(times_s - contact) <= CONTACT_EXCLUSION_S
    return mask


def trajectory_metrics(
    times_s: np.ndarray,
    positions_m: np.ndarray,
    velocities_m_s: np.ndarray,
    valid: np.ndarray,
    reference: ReferenceTrajectory,
    contacts_s: list[float],
) -> dict[str, Any]:
    ref_position, ref_velocity = interpolate_reference(reference, times_s)
    finite = (
        valid
        & np.all(np.isfinite(positions_m), axis=1)
        & np.all(np.isfinite(velocities_m_s), axis=1))
    contact = _contact_mask(times_s, contacts_s)
    evaluate = finite & ~contact
    position_error_mm = np.linalg.norm(
        positions_m - ref_position, axis=1) * 1000.0
    velocity_error = np.linalg.norm(
        velocities_m_s - ref_velocity, axis=1)
    step_speed: list[float] = []
    position_jump_residual_mm: list[float] = []
    position_outlier_count = 0
    velocity_delta: list[float] = []
    position_second_difference_mm: list[float] = []
    high_frequency_velocity_delta: list[float] = []
    for index in range(1, len(times_s)):
        if not (finite[index] and finite[index - 1]):
            continue
        dt = times_s[index] - times_s[index - 1]
        if dt <= 1e-5 or dt > 0.08:
            continue
        step_speed.append(float(np.linalg.norm(
            positions_m[index] - positions_m[index - 1]) / dt))
        delta = velocities_m_s[index] - velocities_m_s[index - 1]
        delta = delta.copy()
        delta[2] += 9.8 * dt
        velocity_delta.append(float(np.linalg.norm(delta)))
        if not (contact[index] or contact[index - 1]):
            high_frequency_velocity_delta.append(float(np.linalg.norm(delta)))
    for index in range(1, len(times_s) - 1):
        if not (finite[index - 1] and finite[index] and finite[index + 1]):
            continue
        dt0 = times_s[index] - times_s[index - 1]
        dt1 = times_s[index + 1] - times_s[index]
        if min(dt0, dt1) <= 1e-5 or max(dt0, dt1) > 0.08:
            continue
        v0 = (positions_m[index] - positions_m[index - 1]) / dt0
        v1 = (positions_m[index + 1] - positions_m[index]) / dt1
        residual = v1 - v0
        residual[2] += 9.8 * 0.5 * (dt0 + dt1)
        position_second_difference_mm.append(float(
            np.linalg.norm(residual) * 0.5 * (dt0 + dt1) * 1000.0))
        if not (contact[index - 1]
                or contact[index] or contact[index + 1]):
            span = dt0 + dt1
            alpha = dt0 / span
            expected = (
                (1.0 - alpha) * positions_m[index - 1]
                + alpha * positions_m[index + 1])
            local_residual_mm = float(np.linalg.norm(
                positions_m[index] - expected) * 1000.0)
            position_jump_residual_mm.append(local_residual_mm)
            if (local_residual_mm > 35.0
                    and max(np.linalg.norm(v0), np.linalg.norm(v1))
                    > POSITION_JUMP_SPEED_M_S):
                position_outlier_count += 1
    evaluate_position = position_error_mm[evaluate]
    evaluate_velocity = velocity_error[evaluate]
    return {
        "samples": len(times_s),
        "validSamples": int(np.count_nonzero(finite)),
        "evaluatedSamples": int(np.count_nonzero(evaluate)),
        "positionErrorMedianMm": _percentile(evaluate_position, 50.0),
        "positionErrorP95Mm": _percentile(evaluate_position, 95.0),
        "velocityErrorMedianMps": _percentile(evaluate_velocity, 50.0),
        "velocityErrorP95Mps": _percentile(evaluate_velocity, 95.0),
        "positionStepSpeedP99Mps": _percentile(step_speed, 99.0),
        "positionJumpCount": int(position_outlier_count),
        "positionJumpResidualP99Mm": _percentile(
            position_jump_residual_mm, 99.0),
        "velocityDeltaP99Mps": _percentile(velocity_delta, 99.0),
        "velocityJumpCount": int(sum(
            value > VELOCITY_JUMP_M_S for value in high_frequency_velocity_delta)),
        "positionHighFrequencyP95Mm": _percentile(
            position_second_difference_mm, 95.0),
        "velocityHighFrequencyP95Mps": _percentile(
            high_frequency_velocity_delta, 95.0),
    }


def recorded_metrics(
    refreshes: list[Tick], phase: Phase, reference: ReferenceTrajectory,
) -> dict[str, Any]:
    selected = [
        tick for tick in refreshes
        if phase.start_s <= tick.wall_time_s <= phase.end_s
    ]
    times = np.asarray([tick.wall_time_s for tick in selected])
    positions = np.asarray([tick.observed_ball_m for tick in selected])
    velocities = np.asarray([tick.observed_velocity_m_s for tick in selected])
    valid = np.asarray([tick.valid for tick in selected], dtype=bool)
    return trajectory_metrics(
        times, positions, velocities, valid, reference, phase.contacts_s)


def pipeline_metrics(
    replay: dict[str, Any], reference: ReferenceTrajectory, phase: Phase,
) -> dict[str, Any]:
    rows = replay["rows"]
    times = np.asarray([row["timeS"] for row in rows])
    positions = np.asarray([
        row["positionM"] if row["positionM"] is not None
        else [math.nan] * 3 for row in rows
    ])
    velocities = np.asarray([
        row["velocityMps"] if row["velocityMps"] is not None
        else [math.nan] * 3 for row in rows
    ])
    valid = np.asarray([
        row["valid"] and row["ready"] for row in rows], dtype=bool)
    replay_contacts = list(phase.contacts_s)
    replay_contacts.extend(
        row["timeS"] for row in rows if row.get("impact", False))
    return {
        **trajectory_metrics(
            times, positions, velocities, valid, reference, replay_contacts),
        "motionRejected": replay["motionRejected"],
        "lossSamples": replay["lossSamples"],
        "impactConfirmed": replay["impactConfirmed"],
        "impactRejected": replay["impactRejected"],
        "restartBridges": replay["restartBridges"],
    }


def recording_capability(run_dir: Path) -> dict[str, Any]:
    arrays = list(run_dir.rglob("*.npy")) + list(run_dir.rglob("*.npz"))
    bags = list(run_dir.rglob("*.bag")) + list(run_dir.rglob("*.db3"))
    frames = run_dir / "frames"
    return {
        "rawArrayCount": len(arrays),
        "realsenseBagCount": len(bags),
        "annotatedJpegCount": (
            len(list(frames.glob("*.jpg"))) if frames.is_dir() else 0),
        "fullIrDepthReplayAvailable": bool(arrays or bags),
        "stateReplayAvailable": run_dir.joinpath(
            "control_samples.jsonl").is_file(),
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    ticks = load_ticks(run_dir)
    refreshes = fresh_camera_ticks(ticks)
    phase = select_valid_phase(ticks)
    if phase is None:
        return {
            "run": run_dir.name,
            "eligible": False,
            "reason": "no_valid_policy_phase",
            "controlSamples": len(ticks),
            "cameraSamples": len(refreshes),
            "capability": recording_capability(run_dir),
        }
    phase_refreshes = [
        tick for tick in refreshes
        if phase.start_s <= tick.wall_time_s <= phase.end_s
    ]
    if len(phase_refreshes) < 8:
        return {
            "run": run_dir.name,
            "eligible": False,
            "reason": "insufficient_valid_camera_samples",
            "controlSamples": len(ticks),
            "cameraSamples": len(refreshes),
            "phaseCameraSamples": len(phase_refreshes),
            "capability": recording_capability(run_dir),
        }
    if not phase.contacts_s:
        return {
            "run": run_dir.name,
            "eligible": False,
            "reason": "no_confirmed_juggling_contact",
            "controlSamples": len(ticks),
            "cameraSamples": len(refreshes),
            "phaseCameraSamples": len(phase_refreshes),
            "capability": recording_capability(run_dir),
        }
    reference = build_reference(refreshes, phase)
    recorded = recorded_metrics(refreshes, phase, reference)
    variants: dict[str, Any] = {}
    for variant in PIPELINE_VARIANTS:
        replay = replay_pipeline(refreshes, phase, variant)
        variants[variant.name] = pipeline_metrics(replay, reference, phase)
    switches = Counter()
    for tick in phase_refreshes:
        measurement = tick.measurement
        switches.update({
            "syncDepthOn": bool(measurement.get(
                "synchronized_depth_enabled", False)),
            "leastSquaresOn": bool(measurement.get(
                "velocity_least_squares_enabled", False)),
            "motionGateOn": bool(measurement.get(
                "motion_consistency_gate_enabled", False)),
            "positionSmoothingOn": bool(measurement.get(
                "trajectory_position_smoothing_enabled", False)),
            "impactResetOn": bool(measurement.get(
                "impact_velocity_reset_enabled", False)),
            "reacqVelocityOn": bool(tick.prediction.get(
                "reacquisitionVelocityHandoverEnabled", False)),
            "timeCompOn": bool(tick.prediction.get(
                "controlTimeCompensationEnabled", False)),
        })
    return {
        "run": run_dir.name,
        "eligible": True,
        "controlSamples": len(ticks),
        "cameraSamples": len(refreshes),
        "phase": {
            "startWallTimeS": round(phase.start_s, 6),
            "endWallTimeS": round(phase.end_s, 6),
            "startSourceTimeS": round(phase.start_source_time_s, 6),
            "endSourceTimeS": round(phase.end_source_time_s, 6),
            "durationS": round(phase.duration_s, 6),
            "endReason": phase.reason,
            "terminalMissWallTimeS": (
                None if phase.miss_time_s is None
                else round(phase.miss_time_s, 6)),
            "contactsWallTimeS": [round(value, 6)
                                  for value in phase.contacts_s],
            "cameraSamples": len(phase_refreshes),
        },
        "reference": {
            "inputValidSamples": int(np.count_nonzero(reference.input_valid)),
            "outlierOrMissingSamples": int(np.count_nonzero(reference.outlier)),
            "numericOutlierSamples": int(np.count_nonzero(
                reference.outlier & reference.input_valid)),
            "interpolatedSamples": int(np.count_nonzero(
                reference.interpolated)),
        },
        "recorded": recorded,
        "variants": variants,
        "switchTrueSampleCounts": dict(switches),
        "capability": recording_capability(run_dir),
    }


def _aggregate_metric(
    rows: list[dict[str, Any]], path: tuple[str, ...],
) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append(number)
    return {
        "runCount": len(values),
        "medianAcrossRuns": _percentile(values, 50.0),
        "p90AcrossRuns": _percentile(values, 90.0),
        "maxAcrossRuns": max(values) if values else None,
        "sumAcrossRuns": round(float(sum(values)), 6) if values else None,
    }


def summarize(run_rows: list[dict[str, Any]], dates: list[str]) -> dict[str, Any]:
    eligible = [row for row in run_rows if row.get("eligible")]
    metric_names = (
        "positionErrorMedianMm",
        "positionErrorP95Mm",
        "velocityErrorMedianMps",
        "velocityErrorP95Mps",
        "positionStepSpeedP99Mps",
        "positionJumpCount",
        "positionJumpResidualP99Mm",
        "velocityDeltaP99Mps",
        "velocityJumpCount",
        "positionHighFrequencyP95Mm",
        "velocityHighFrequencyP95Mps",
    )
    recorded = {
        name: _aggregate_metric(eligible, ("recorded", name))
        for name in metric_names
    }
    variants = {
        variant.name: {
            name: _aggregate_metric(
                eligible, ("variants", variant.name, name))
            for name in metric_names
        }
        for variant in PIPELINE_VARIANTS
    }
    return {
        "dates": dates,
        "discoveredRunCount": len(run_rows),
        "eligibleRunCount": len(eligible),
        "excludedRunCount": len(run_rows) - len(eligible),
        "effectiveDurationS": round(sum(
            row["phase"]["durationS"] for row in eligible), 6),
        "effectiveCameraSamples": sum(
            row["phase"]["cameraSamples"] for row in eligible),
        "numericOutlierSamples": sum(
            row["reference"]["numericOutlierSamples"] for row in eligible),
        "interpolatedSamples": sum(
            row["reference"]["interpolatedSamples"] for row in eligible),
        "fullIrDepthReplayRunCount": sum(
            row["capability"]["fullIrDepthReplayAvailable"]
            for row in run_rows),
        "recorded": recorded,
        "variants": variants,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Three-day offline ball trajectory study",
        "",
        f"- Dates: {', '.join(summary['dates'])}",
        (f"- Runs: {summary['eligibleRunCount']} eligible / "
         f"{summary['discoveredRunCount']} discovered"),
        f"- Effective juggling duration: {summary['effectiveDurationS']} s",
        f"- Fresh camera samples: {summary['effectiveCameraSamples']}",
        ("- Numeric outliers replaced in the acausal reference: "
         f"{summary['numericOutlierSamples']}"),
        ("- Full raw IR/depth rerun available: "
         f"{summary['fullIrDepthReplayRunCount']}/"
         f"{summary['discoveredRunCount']} runs"),
        "",
        "## Aggregate detector-state replay",
        "",
        ("| chain | position p95 median across runs (mm) | "
         "velocity p95 median across runs (m/s) | "
         "position HF p95 (mm) | velocity HF p95 (m/s) | "
         "position jumps | velocity jumps |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(block: dict[str, Any], metric: str, field: str = "medianAcrossRuns"):
        return block[metric].get(field)

    recorded = summary["recorded"]
    lines.append(
        f"| recorded policy input | {cell(recorded, 'positionErrorP95Mm')} | "
        f"{cell(recorded, 'velocityErrorP95Mps')} | "
        f"{cell(recorded, 'positionHighFrequencyP95Mm')} | "
        f"{cell(recorded, 'velocityHighFrequencyP95Mps')} | "
        f"{cell(recorded, 'positionJumpCount', 'sumAcrossRuns')} | "
        f"{cell(recorded, 'velocityJumpCount', 'sumAcrossRuns')} |")
    for variant in PIPELINE_VARIANTS:
        block = summary["variants"][variant.name]
        lines.append(
            f"| {variant.name} | {cell(block, 'positionErrorP95Mm')} | "
            f"{cell(block, 'velocityErrorP95Mps')} | "
            f"{cell(block, 'positionHighFrequencyP95Mm')} | "
            f"{cell(block, 'velocityHighFrequencyP95Mps')} | "
            f"{cell(block, 'positionJumpCount', 'sumAcrossRuns')} | "
            f"{cell(block, 'velocityJumpCount', 'sumAcrossRuns')} |")
    lines.extend([
        "",
        "`recommended` keeps synchronized IR+depth, least-squares velocity, "
        "reacquisition velocity handover and control-time compensation off. "
        "It enables loss-boundary clearing, the ballistic motion gate, causal "
        "physics velocity, bounded endpoint position smoothing and confirmed "
        "impact reset.",
        "",
        "The clean trajectory is acausal and is used only as an evaluation "
        "reference/counterfactual replay input. It is not deployable online.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=(
        PACKAGE_ROOT / "pingpong_controller" / "outputs"
        / "interactive_recordings"))
    parser.add_argument(
        "--dates", nargs="+", default=["20260810", "20260811", "20260812"])
    parser.add_argument("--output", type=Path, default=(
        Path("analysis_results") / "ball_trajectory_study_20260810_12.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    runs = discover_runs(args.root, args.dates)
    if args.limit > 0:
        runs = runs[:args.limit]
    rows: list[dict[str, Any]] = []
    for index, run_dir in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run_dir.name}", flush=True)
        try:
            rows.append(analyze_run(run_dir))
        except Exception as exc:  # noqa: BLE001 - keep the corpus audit going.
            rows.append({
                "run": run_dir.name,
                "eligible": False,
                "reason": "analysis_error",
                "error": f"{type(exc).__name__}: {exc}",
                "capability": recording_capability(run_dir),
            })
    report = {
        "schemaVersion": 1,
        "scope": {
            "root": str(args.root.resolve()),
            "dates": args.dates,
            "duplicatePolicy": "top-level only; record_new3/new4 excluded",
            "phasePolicy": (
                "policyExecuted interval; cut at detected terminal miss and "
                "always remove final 120 ms"),
        },
        "pipelineContract": {
            "synchronizedIrDepth": False,
            "leastSquaresVelocity": False,
            "reacquisitionVelocityHandover": False,
            "controlTimeCompensation90To200": False,
            "variants": [variant.__dict__ for variant in PIPELINE_VARIANTS],
        },
        "summary": summarize(rows, args.dates),
        "runs": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(output),
        "markdown": str(markdown),
        "runs": report["summary"]["discoveredRunCount"],
        "eligible": report["summary"]["eligibleRunCount"],
        "effectiveDurationS": report["summary"]["effectiveDurationS"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
