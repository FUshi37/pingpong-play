#!/usr/bin/env python3
"""Replay recorded camera-state telemetry and audit prediction handovers.

The interactive recorder stores every detector result, camera timestamp,
trusted/untrusted decision, detector velocity and prediction-state diagnostic.
That is sufficient to replay the 6D prediction/handover state machine and to
compare trajectory initializers.  It is not sufficient to rerun IR contour or
stereo-depth extraction unless a recording also contains IR/depth arrays or a
RealSense bag; the report states that distinction explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.high_ball_occlusion_predictor import (  # noqa: E402
    HighBallOcclusionConfig,
    HighBallOcclusionPredictor,
    propagate_paper_model,
)


HIGH_FOV_TRIGGERS = {"high_fov_occlusion", "high_fov_edge_exit"}
MODEL_TRIGGERS = HIGH_FOV_TRIGGERS | {"sensor_depth_anomaly"}
MAX_REFERENCE_HORIZON_S = 0.55
CLEAN_REFERENCE_ERROR_M = 0.20
WARMUP_GATE_M = HighBallOcclusionPredictor.OPTIMIZED_REACQUISITION_GATE_M


@dataclass
class CameraSample:
    stamp_s: float
    source_time_s: float
    raw_valid: bool
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    measurement: dict[str, Any]
    prediction: dict[str, Any]
    runtime: dict[str, Any]
    racket_m: np.ndarray | None

    @property
    def trusted(self) -> bool:
        trusted = self.measurement.get("measurement_trusted_3d")
        return bool(self.raw_valid and (trusted is None or trusted))


def _vector(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return np.zeros(3, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        return np.zeros(3, dtype=np.float64)
    return result


def load_camera_samples(run_dir: Path) -> list[CameraSample]:
    """Load one sample per fresh camera timestamp, not per 200 Hz tick."""
    path = run_dir / "control_samples.jsonl"
    samples: list[CameraSample] = []
    previous_stamp: float | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            stamp = row.get("cameraStampS")
            if stamp is None:
                continue
            stamp = float(stamp)
            if (previous_stamp is not None
                    and abs(stamp - previous_stamp) <= 1.0e-6):
                continue
            previous_stamp = stamp
            racket = row.get("racket")
            samples.append(CameraSample(
                stamp_s=stamp,
                source_time_s=float(row.get("sourceTimeS", 0.0)),
                raw_valid=bool(row.get("rawValid", False)),
                position_m=_vector(row.get("rawBall")),
                velocity_m_s=_vector(row.get("measuredVelocity")),
                measurement=dict(row.get("cameraMeasurement") or {}),
                prediction=dict(
                    row.get("highBallOcclusionPrediction") or {}),
                runtime=dict(row.get("runtime") or {}),
                racket_m=(None if racket is None else _vector(racket)),
            ))
    return samples


def _contiguous_history(
    samples: list[CameraSample], end_index: int, end_stamp_s: float,
    *, limit: int,
) -> list[tuple[float, np.ndarray]]:
    candidates = [
        (sample.stamp_s, sample.position_m)
        for sample in samples[:end_index + 1]
        if sample.trusted and sample.stamp_s <= end_stamp_s + 1.0e-5
    ]
    tail: list[tuple[float, np.ndarray]] = []
    for item in reversed(candidates):
        if tail and tail[-1][0] - item[0] > 0.051:
            break
        tail.append(item)
    return list(reversed(tail))[-limit:]


def _fit_paper_initial_state(
    history: list[tuple[float, np.ndarray]],
) -> tuple[float, np.ndarray, np.ndarray]:
    history = history[-10:]
    times = np.asarray([item[0] for item in history], dtype=np.float64)
    positions = np.asarray([item[1] for item in history], dtype=np.float64)
    endpoint = float(times[-1])
    relative = times - endpoint
    linear = np.column_stack((relative, np.ones_like(relative)))
    position = np.zeros(3, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)
    for axis in (0, 1):
        coefficient, *_ = np.linalg.lstsq(
            linear, positions[:, axis], rcond=None)
        velocity[axis], position[axis] = coefficient
    vertical = np.column_stack((relative ** 2, relative, np.ones_like(relative)))
    coefficient, *_ = np.linalg.lstsq(
        vertical, positions[:, 2], rcond=None)
    velocity[2], position[2] = coefficient[1], coefficient[2]
    return endpoint, position, velocity


def _fit_revised_initial_state(
    history: list[tuple[float, np.ndarray]], trigger: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    history = history[-HighBallOcclusionPredictor.OPTIMIZED_HISTORY_FRAMES:]
    predictor = HighBallOcclusionPredictor(HighBallOcclusionConfig(
        history_frames=max(3, len(history)),
        enable_optimized_initial_state=True,
    ))
    for stamp, position in history:
        predictor._append_measurement(stamp, position)  # shared offline fit
    position, velocity, _ = predictor._fit_initial_state(trigger=trigger)
    return history[-1][0], position, velocity


def _prediction_error(
    initial_time_s: float,
    initial_position_m: Iterable[float],
    initial_velocity_m_s: Iterable[float],
    recovery: CameraSample,
) -> np.ndarray:
    predicted, _ = propagate_paper_model(
        initial_position_m,
        initial_velocity_m_s,
        recovery.stamp_s - initial_time_s,
    )
    return recovery.position_m - predicted


def extract_model_events(
    run_dir: Path, samples: list[CameraSample],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_active = False
    for index, sample in enumerate(samples):
        diagnostics = sample.prediction
        active = bool(diagnostics.get("active", False))
        trigger = str(diagnostics.get("activeTrigger") or "")
        if active and not previous_active and trigger in MODEL_TRIGGERS:
            initial_time = diagnostics.get("initialTimeS")
            initial_position = diagnostics.get("initialPositionM")
            initial_velocity = diagnostics.get("initialVelocityMps")
            if (initial_time is not None and initial_position is not None
                    and initial_velocity is not None):
                recovery = next((
                    candidate for candidate in samples[index + 1:]
                    if candidate.trusted
                    and candidate.stamp_s > float(initial_time) + 1.0e-6
                ), None)
                history = _contiguous_history(
                    samples, index, float(initial_time), limit=10)
                if recovery is not None and len(history) >= 5:
                    horizon = recovery.stamp_s - float(initial_time)
                    if 0.0 < horizon <= MAX_REFERENCE_HORIZON_S:
                        recorded_error = _prediction_error(
                            float(initial_time), initial_position,
                            initial_velocity, recovery)
                        revised = _fit_revised_initial_state(history, trigger)
                        paper = _fit_paper_initial_state(history)
                        revised_error = _prediction_error(*revised, recovery)
                        paper_error = _prediction_error(*paper, recovery)
                        clean = bool(
                            np.linalg.norm(recorded_error)
                            <= CLEAN_REFERENCE_ERROR_M)
                        events.append({
                            "run": run_dir.name,
                            "trigger": trigger,
                            "activationSourceTimeS": sample.source_time_s,
                            "horizonS": round(float(horizon), 6),
                            "historySamples": len(history),
                            "cleanReference": clean,
                            "recordedErrorMm": np.round(
                                recorded_error * 1000.0, 6).tolist(),
                            "recordedErrorNormMm": round(float(
                                np.linalg.norm(recorded_error) * 1000.0), 6),
                            "revisedErrorMm": np.round(
                                revised_error * 1000.0, 6).tolist(),
                            "revisedErrorNormMm": round(float(
                                np.linalg.norm(revised_error) * 1000.0), 6),
                            "paperErrorMm": np.round(
                                paper_error * 1000.0, 6).tolist(),
                            "paperErrorNormMm": round(float(
                                np.linalg.norm(paper_error) * 1000.0), 6),
                        })
        previous_active = active
    return events


def _metric_block(events: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not events:
        return {"count": 0}
    errors = np.asarray([event[prefix + "ErrorMm"] for event in events])
    norms = np.linalg.norm(errors, axis=1)
    return {
        "count": len(events),
        "medianAbsAxisMm": np.round(
            np.median(np.abs(errors), axis=0), 3).tolist(),
        "medianNormMm": round(float(np.median(norms)), 3),
        "meanNormMm": round(float(np.mean(norms)), 3),
        "p90NormMm": round(float(np.percentile(norms, 90.0)), 3),
    }


def summarize_model_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups = {
        "all": events,
        "highFov": [e for e in events if e["trigger"] in HIGH_FOV_TRIGGERS],
        "depthFault": [
            e for e in events if e["trigger"] == "sensor_depth_anomaly"],
    }
    for name, group in groups.items():
        clean = [event for event in group if event["cleanReference"]]
        result[name] = {
            "usableCount": len(group),
            "cleanCount": len(clean),
            "cleanCriterion": "recorded model recovery error <= 200 mm",
            "recorded": _metric_block(clean, "recorded"),
            "revised": _metric_block(clean, "revised"),
            "paperQuadratic": _metric_block(clean, "paper"),
        }
    return result


def _camera_affine_model(
    samples: list[CameraSample],
) -> tuple[np.ndarray | None, dict[str, float | int | None]]:
    design: list[list[float]] = []
    target: list[np.ndarray] = []
    for sample in samples:
        measurement = sample.measurement
        values = [measurement.get(key) for key in ("cx", "cy", "depth_mm")]
        if not sample.trusted or any(value is None for value in values):
            continue
        cx, cy, depth_mm = (float(value) for value in values)
        depth_m = depth_mm * 1.0e-3
        design.append([cx * depth_m, cy * depth_m, depth_m, 1.0])
        target.append(sample.position_m)
    if len(design) < 12:
        return None, {"samples": len(design), "medianFitErrorMm": None,
                      "p95FitErrorMm": None}
    matrix, *_ = np.linalg.lstsq(
        np.asarray(design), np.asarray(target), rcond=None)
    residual = np.linalg.norm(
        np.asarray(design) @ matrix - np.asarray(target), axis=1) * 1000.0
    return matrix, {
        "samples": len(design),
        "medianFitErrorMm": round(float(np.median(residual)), 3),
        "p95FitErrorMm": round(float(np.percentile(residual, 95.0)), 3),
    }


def _reconstruct_rejected_position(
    sample: CameraSample, matrix: np.ndarray | None,
) -> np.ndarray | None:
    measurement = sample.measurement
    exact = [measurement.get(
        f"transformed_measurement_{axis}_mm") for axis in "xyz"]
    if all(value is not None for value in exact):
        return np.asarray(exact, dtype=np.float64) * 1.0e-3
    if matrix is None:
        return None
    values = [measurement.get(key) for key in ("cx", "cy", "depth_mm")]
    if any(value is None for value in values):
        return None
    cx, cy, depth_mm = (float(value) for value in values)
    depth_m = depth_mm * 1.0e-3
    return np.asarray(
        [cx * depth_m, cy * depth_m, depth_m, 1.0]) @ matrix


def audit_impact_gate_order(
    run_dir: Path, samples: list[CameraSample],
) -> dict[str, Any]:
    """Find contact evidence that the old motion-gate order starved."""
    matrix, fit = _camera_affine_model(samples)
    candidate_starved: list[dict[str, Any]] = []
    confirmation_starved: list[dict[str, Any]] = []

    def racket_ok(sample: CameraSample, position: np.ndarray) -> bool:
        if sample.racket_m is None:
            return not bool(sample.measurement.get(
                "impact_racket_gate_required", False))
        return float(np.linalg.norm(
            position - sample.racket_m) * 1000.0) <= 150.0

    for index in range(1, len(samples)):
        current = samples[index]
        previous = samples[index - 1]
        if (current.measurement.get("observation_state") != "motion_outlier"
                or previous.measurement.get("velocity_motion_state")
                != "flight"):
            continue
        reconstructed = _reconstruct_rejected_position(current, matrix)
        if reconstructed is None:
            continue
        dt = current.stamp_s - previous.stamp_s
        if not 1.0e-4 < dt <= 0.035:
            continue
        pair_velocity = (reconstructed - previous.position_m) / dt
        previous_vz = float(previous.velocity_m_s[2])
        if (previous_vz <= -0.25
                and pair_velocity[2] >= 0.25
                and pair_velocity[2] - previous_vz >= 0.80
                and np.linalg.norm(pair_velocity) <= 6.0
                and racket_ok(current, reconstructed)):
            candidate_starved.append({
                "run": run_dir.name,
                "sourceTimeS": current.source_time_s,
                "pairVelocityMps": np.round(pair_velocity, 6).tolist(),
                "reconstructedPositionM": np.round(
                    reconstructed, 6).tolist(),
            })

    for index in range(1, len(samples) - 1):
        first = samples[index]
        if not first.measurement.get("velocity_impact_candidate", False):
            continue
        pre = samples[index - 1]
        for current in samples[index + 1:index + 3]:
            if current.measurement.get("observation_state") != "motion_outlier":
                continue
            reconstructed = _reconstruct_rejected_position(current, matrix)
            if reconstructed is None:
                continue
            dt = current.stamp_s - first.stamp_s
            if not 1.0e-4 < dt <= 0.035:
                continue
            pair_velocity = (reconstructed - first.position_m) / dt
            total_rise_m = float(reconstructed[2] - pre.position_m[2])
            if (pair_velocity[2] >= 0.25
                    and total_rise_m >= 0.004
                    and np.linalg.norm(pair_velocity) <= 6.0
                    and racket_ok(current, reconstructed)):
                confirmation_starved.append({
                    "run": run_dir.name,
                    "candidateSourceTimeS": first.source_time_s,
                    "confirmationSourceTimeS": current.source_time_s,
                    "pairVelocityMps": np.round(pair_velocity, 6).tolist(),
                    "totalRiseMm": round(total_rise_m * 1000.0, 3),
                })
                break
    return {
        "affineRejectedPointReconstruction": fit,
        "firstCandidateStarved": candidate_starved,
        "pendingConfirmationStarved": confirmation_starved,
    }


def audit_recorded_warmup(
    run_dir: Path, samples: list[CameraSample],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_active = False
    active_event: dict[str, Any] | None = None
    for sample in samples:
        diagnostics = sample.prediction
        active = bool(diagnostics.get("active", False))
        trigger = diagnostics.get("activeTrigger")
        if active and not previous_active and trigger == "velocity_warmup":
            active_event = {
                "run": run_dir.name,
                "sourceTimeS": sample.source_time_s,
                "maxRecordedInnovationMm": 0.0,
            }
        if active_event is not None:
            error = diagnostics.get("lastReacquisitionErrorNormMm")
            if error is not None:
                active_event["maxRecordedInnovationMm"] = max(
                    active_event["maxRecordedInnovationMm"], float(error))
        if active_event is not None and previous_active and not active:
            active_event["revisedBridgeAccepted"] = bool(
                active_event["maxRecordedInnovationMm"]
                <= WARMUP_GATE_M * 1000.0)
            events.append(active_event)
            active_event = None
        previous_active = active
    if active_event is not None:
        active_event["revisedBridgeAccepted"] = bool(
            active_event["maxRecordedInnovationMm"]
            <= WARMUP_GATE_M * 1000.0)
        events.append(active_event)
    return events


def _config_from_recording(samples: list[CameraSample]) -> HighBallOcclusionConfig:
    diagnostics = next((sample.prediction for sample in samples
                        if sample.prediction), {})
    return HighBallOcclusionConfig(
        history_frames=10,
        activation_z_m=float(diagnostics.get("activationZMinM", 1.42)),
        min_upward_velocity_m_s=float(
            diagnostics.get("minUpwardVelocityMps", 0.20)),
        max_prediction_s=float(diagnostics.get("maxPredictionS", 0.45)),
        stop_prediction_z_m=float(
            diagnostics.get("stopPredictionZMinM", 1.24)),
        drag_coefficient_per_m=float(
            diagnostics.get("dragCoefficientPerM", 0.13545)),
        integration_step_s=float(
            diagnostics.get("integrationStepS", 0.005)),
        reacquisition_gate_m=float(
            diagnostics.get("reacquisitionGateM", 0.18)),
        enable_reacquisition_velocity_handover=bool(
            diagnostics.get("reacquisitionVelocityHandoverEnabled", False)),
        reacquisition_velocity_frames=int(
            diagnostics.get("reacquisitionVelocityFrames", 3)),
        reacquisition_velocity_blend_s=float(
            diagnostics.get("reacquisitionVelocityBlendS", 0.04)),
        enable_control_time_compensation=False,
        # The experiment asks how the *revised current method* behaves on old
        # recordings, not how each historical switch combination behaved.
        enable_reacquisition_smoothing=True,
        enable_optimized_initial_state=True,
    )


def replay_predictor(samples: list[CameraSample]) -> dict[str, Any]:
    """Run the revised predictor against recorded detector decisions."""
    if not samples:
        return {"samples": 0}
    predictor = HighBallOcclusionPredictor(_config_from_recording(samples))
    warmup_rejections = 0
    finite_failures = 0
    valid_count = 0
    predicted_count = 0
    previous_impact_stamp: float | None = None
    previous_result = None
    handover_transitions: list[dict[str, Any]] = []
    for sample in samples:
        measurement = sample.measurement
        recorded = sample.prediction
        impact = bool(measurement.get("velocity_impact_detected", False))
        if impact and (previous_impact_stamp is None
                       or abs(sample.stamp_s - previous_impact_stamp) > 1.0e-6):
            predictor.reset(clear_counters=False)
            previous_impact_stamp = sample.stamp_s
        depth_fault = bool(recorded.get(
            "sensorDepthAnomalyDetected",
            measurement.get("depth_sensor_rejected", False)))
        trusted = bool(sample.trusted and not depth_fault)
        force_reason = None
        if depth_fault:
            force_reason = "sensor_depth_anomaly"
        elif bool(recorded.get("topFovExitDetected", False)):
            force_reason = "high_fov_edge_exit"
        result = predictor.step(
            sample.position_m,
            sample.velocity_m_s,
            trusted,
            sample.stamp_s,
            sample.stamp_s,
            force_prediction_reason=force_reason,
            allow_high_fov_prediction=bool(
                recorded.get("highFovEnabled", True)),
            measurement_velocity_valid=bool(
                measurement.get("velocity_ready", True)),
            measurement_reacquisition_confirmed=bool(
                measurement.get("motion_reacquisition_confirmed", False)),
        )
        valid_count += int(result.valid)
        predicted_count += int(result.predicted)
        if not (np.all(np.isfinite(result.position_m))
                and np.all(np.isfinite(result.velocity_m_s))):
            finite_failures += 1
        if (previous_result is not None
                and previous_result.valid and result.valid
                and previous_result.predicted and not result.predicted
                and not impact):
            dt = float(result.timestamp_s - previous_result.timestamp_s)
            if 0.0 <= dt <= 0.060:
                expected_position, expected_velocity = propagate_paper_model(
                    previous_result.position_m,
                    previous_result.velocity_m_s,
                    dt,
                )
                position_step = float(np.linalg.norm(
                    result.position_m - expected_position))
                velocity_step = float(np.linalg.norm(
                    result.velocity_m_s - expected_velocity))
                handover_transitions.append({
                    "sourceTimeS": sample.source_time_s,
                    "positionInnovationMm": round(
                        position_step * 1000.0, 3),
                    "velocityInnovationMps": round(velocity_step, 6),
                    "large": bool(
                        position_step > 0.050 or velocity_step > 0.75),
                })
        previous_result = result
        warmup_rejections = max(
            warmup_rejections,
            int(result.diagnostics.get(
                "velocityWarmupBridgeRejectedCount", 0)))
    final = predictor.diagnostics()
    return {
        "samples": len(samples),
        "validSamples": valid_count,
        "predictedSamples": predicted_count,
        "nonFiniteOutputs": finite_failures,
        "velocityWarmupBridgeCount": int(
            final.get("velocityWarmupBridgeCount", 0)),
        "velocityWarmupBridgeRejectedCount": warmup_rejections,
        "handoverTransitionCount": len(handover_transitions),
        "largeHandoverTransitionCount": sum(
            event["large"] for event in handover_transitions),
        "handoverTransitions": handover_transitions,
    }


def recording_capability(run_dir: Path) -> dict[str, Any]:
    npy = list(run_dir.rglob("*.npy"))
    bags = list(run_dir.rglob("*.bag"))
    frames = run_dir / "frames"
    jpeg_count = len(list(frames.glob("*.jpg"))) if frames.is_dir() else 0
    return {
        "run": run_dir.name,
        "irOrDepthArrayCount": len(npy),
        "realsenseBagCount": len(bags),
        "annotatedJpegCount": jpeg_count,
        "fullIrDepthVisionReplayAvailable": bool(npy or bags),
        "statePipelineReplayAvailable": (
            run_dir.joinpath("control_samples.jsonl").is_file()),
    }


def discover_runs(root: Path, date: str) -> list[Path]:
    candidates = list((root / "record_new4").glob("probe_recording_*"))
    candidates.extend(root.glob(f"probe_recording_{date}_*"))
    return sorted({path.resolve() for path in candidates
                   if path.joinpath("control_samples.jsonl").is_file()})


def _markdown(report: dict[str, Any]) -> str:
    model = report["modelComparison"]
    high = model["highFov"]
    depth = model["depthFault"]
    impact = report["impactGateOrder"]
    replay = report["revisedStateReplay"]
    lines = [
        "# Offline ball-state replay",
        "",
        f"- Runs: {report['runCount']}",
        f"- Fresh camera samples: {report['cameraSampleCount']}",
        ("- Full IR/depth rerun available: "
         f"{report['rawDataCapability']['fullVisionReplayRunCount']}/"
         f"{report['runCount']} runs"),
        "",
        "## Clean recovery model comparison",
        "",
        "| group | events | recorded median norm (mm) | revised median norm (mm) | paper quadratic median norm (mm) |",
        "|---|---:|---:|---:|---:|",
        (f"| high-FOV | {high['cleanCount']} | "
         f"{high['recorded'].get('medianNormMm')} | "
         f"{high['revised'].get('medianNormMm')} | "
         f"{high['paperQuadratic'].get('medianNormMm')} |"),
        (f"| depth fault | {depth['cleanCount']} | "
         f"{depth['recorded'].get('medianNormMm')} | "
         f"{depth['revised'].get('medianNormMm')} | "
         f"{depth['paperQuadratic'].get('medianNormMm')} |"),
        "",
        "## State-transition audit",
        "",
        ("- Recorded stale warm-up bridges rejected by the revised 120 mm "
         f"gate: {report['warmupAudit']['revisedRejectedCount']}"),
        ("- First impact candidates previously starved by motion gating: "
         f"{len(impact['firstCandidateStarved'])}"),
        ("- Pending impact confirmations previously starved by motion gating: "
         f"{len(impact['pendingConfirmationStarved'])}"),
        ("- Revised predictor non-finite outputs: "
         f"{replay['nonFiniteOutputs']}"),
        ("- Revised predictor large prediction-to-sensor handovers "
         "(>50 mm position innovation or >0.75 m/s velocity innovation): "
         f"{replay['largeHandoverTransitionCount']}/"
         f"{replay['handoverTransitionCount']}"),
        ("  - record_new4 (historical detector velocity retained): "
         f"{replay['largeHandoverTransitionRecordNew4Count']}"),
        (f"  - {report['scope']['date']} recordings: "
         f"{replay['largeHandoverTransitionCurrentDateCount']}"),
        "",
        "The state replay uses saved detector telemetry. Annotated JPEGs do not "
        "contain the original 16-bit IR/depth arrays, so runs without `.npy` "
        "or `.bag` files cannot rerun contour extraction or stereo sampling.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=(
        PACKAGE_ROOT / "pingpong_controller" / "outputs"
        / "interactive_recordings"))
    parser.add_argument("--date", default="20260812")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    runs = discover_runs(args.root, args.date)
    all_events: list[dict[str, Any]] = []
    warmup_events: list[dict[str, Any]] = []
    impact_candidate: list[dict[str, Any]] = []
    impact_confirmation: list[dict[str, Any]] = []
    capabilities: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    sample_count = 0

    for run in runs:
        samples = load_camera_samples(run)
        sample_count += len(samples)
        all_events.extend(extract_model_events(run, samples))
        warmup_events.extend(audit_recorded_warmup(run, samples))
        impact = audit_impact_gate_order(run, samples)
        impact_candidate.extend(impact["firstCandidateStarved"])
        impact_confirmation.extend(impact["pendingConfirmationStarved"])
        fit_diagnostics.append({
            "run": run.name,
            **impact["affineRejectedPointReconstruction"],
        })
        capabilities.append(recording_capability(run))
        replay_rows.append({"run": run.name, **replay_predictor(samples)})

    handover_transitions = [
        {"run": row["run"], **event}
        for row in replay_rows
        for event in row.get("handoverTransitions", [])
    ]
    replay_summary = {
        "samples": sum(row.get("samples", 0) for row in replay_rows),
        "validSamples": sum(row.get("validSamples", 0) for row in replay_rows),
        "predictedSamples": sum(
            row.get("predictedSamples", 0) for row in replay_rows),
        "nonFiniteOutputs": sum(
            row.get("nonFiniteOutputs", 0) for row in replay_rows),
        "velocityWarmupBridgeCount": sum(
            row.get("velocityWarmupBridgeCount", 0) for row in replay_rows),
        "velocityWarmupBridgeRejectedCount": sum(
            row.get("velocityWarmupBridgeRejectedCount", 0)
            for row in replay_rows),
        "handoverTransitionCount": len(handover_transitions),
        "largeHandoverTransitionCount": sum(
            event["large"] for event in handover_transitions),
        "largeHandoverTransitionRecordNew4Count": sum(
            event["large"] and args.date not in event["run"]
            for event in handover_transitions),
        "largeHandoverTransitionCurrentDateCount": sum(
            event["large"] and args.date in event["run"]
            for event in handover_transitions),
        "maxHandoverPositionInnovationMm": (
            None if not handover_transitions else max(
                event["positionInnovationMm"]
                for event in handover_transitions)),
        "maxHandoverVelocityInnovationMps": (
            None if not handover_transitions else max(
                event["velocityInnovationMps"]
                for event in handover_transitions)),
        "handoverTransitions": handover_transitions,
        "runs": replay_rows,
    }
    report = {
        "schemaVersion": 1,
        "scope": {
            "recordNew4": str((args.root / "record_new4").resolve()),
            "date": args.date,
        },
        "runCount": len(runs),
        "cameraSampleCount": sample_count,
        "modelComparison": summarize_model_events(all_events),
        "modelEvents": all_events,
        "warmupAudit": {
            "recordedEventCount": len(warmup_events),
            "revisedRejectedCount": sum(
                not event["revisedBridgeAccepted"] for event in warmup_events),
            "gateMm": WARMUP_GATE_M * 1000.0,
            "events": warmup_events,
        },
        "impactGateOrder": {
            "firstCandidateStarved": impact_candidate,
            "pendingConfirmationStarved": impact_confirmation,
            "reconstructionFitByRun": fit_diagnostics,
        },
        "revisedStateReplay": replay_summary,
        "rawDataCapability": {
            "fullVisionReplayRunCount": sum(
                row["fullIrDepthVisionReplayAvailable"]
                for row in capabilities),
            "statePipelineReplayRunCount": sum(
                row["statePipelineReplayAvailable"]
                for row in capabilities),
            "runs": capabilities,
        },
    }

    output = args.output or (
        Path("analysis_results")
        / f"ball_state_offline_replay_{args.date}.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(output),
        "markdown": str(markdown),
        "runs": len(runs),
        "samples": sample_count,
        "modelEvents": len(all_events),
        "revisedNonFiniteOutputs": replay_summary["nonFiniteOutputs"],
    }, ensure_ascii=False, indent=2))

    if args.fail_on_regression:
        high = report["modelComparison"]["highFov"]
        recorded = high["recorded"].get("medianNormMm")
        revised = high["revised"].get("medianNormMm")
        if replay_summary["nonFiniteOutputs"]:
            return 2
        if (recorded is not None and revised is not None
                and revised > recorded + 1.0e-6):
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
