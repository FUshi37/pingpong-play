#!/usr/bin/env python3
"""Identify static-racket impacts from interactive probe recordings.

Unlike ``identify_ball_racket_contact.py``, this entry point does not require an
offline policy/RMP replay trace and does not require ``policyExecuted=true``.
It uses the raw camera timestamps and the ``policyJointSample`` recorded by
schema-v11 probe recordings, so it also supports experiments collected with
``Send command`` disabled.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

import identify_ball_racket_contact as contact


WORKSPACE = Path(__file__).resolve().parents[5]
DEFAULT_RECORD_ROOT = (
    WORKSPACE
    / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "analysis_results/ball_racket_contact_identification_v91_static_20260901"
)
DEFAULT_FK_MODEL = (
    WORKSPACE
    / "simcode7/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
)
DEFAULT_CHECKPOINT = (
    WORKSPACE
    / "simcode7/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/"
    "measured_qvel_rmp_vertical_v91_gpu0_seed20261018_20260829_stage24_"
    "ceilling_gate_online2/26_rmp91_complete_nonexecution_factorized_commit.pkl"
)


class V91RacketKinematics(contact.RealRacketKinematics):
    """Deployment-identical FK/Jacobian using the 57-D V85/V91 adapter."""

    def __init__(self, model_path: Path, checkpoint_path: Path) -> None:
        package_root = WORKSPACE / "src/pingpong_moz1"
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        from pingpong_controller.mjx_rl_policy import (  # noqa: WPS433
            Gpu0QvelRealRmpV85PolicyController,
        )

        self.controller = Gpu0QvelRealRmpV85PolicyController(
            str(checkpoint_path),
            robot_xml_path=str(model_path),
        )


def _finite_vector(value: Any, size: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != size or not np.all(np.isfinite(vector)):
        return None
    return vector


def _load_recording(
    control_path: Path,
) -> tuple[
    list[contact.CameraSample],
    dict[str, np.ndarray],
    dict[str, Any],
    list[float],
]:
    samples: list[contact.CameraSample] = []
    q_rows: list[np.ndarray] = []
    dq_rows: list[np.ndarray] = []
    joint_times: list[float] = []
    all_camera_stamps: list[float] = []
    seen_valid_stamps: set[float] = set()
    row_count = 0
    valid_row_count = 0
    published_count = 0
    policy_executed_count = 0
    inferred_contact_count = 0
    detected_impact_stamps: set[float] = set()
    first_wall: float | None = None
    last_wall: float | None = None
    filter_flags: dict[str, Any] = {}

    with control_path.open("rb") as stream:
        for line in stream:
            row_count += 1
            row = contact._json_loads(line)
            wall_time = float(row.get("wallTimeS", math.nan))
            if np.isfinite(wall_time):
                first_wall = wall_time if first_wall is None else first_wall
                last_wall = wall_time
            published_count += int(bool((row.get("command") or {}).get("published", False)))
            policy_executed_count += int(bool(row.get("policyExecuted", False)))
            inferred_contact_count += int(
                bool((row.get("contactIdentification") or {}).get("inferredContact", False))
            )

            joint = row.get("policyJointSample") or {}
            q = _finite_vector(joint.get("qRaw"), 7)
            dq = _finite_vector(joint.get("dqRaw"), 7)
            if q is not None and dq is not None and np.isfinite(wall_time):
                q_rows.append(q)
                dq_rows.append(dq)
                joint_times.append(wall_time)

            stamp_value = row.get("cameraStampS")
            if stamp_value is None:
                continue
            camera_stamp = float(stamp_value)
            if not np.isfinite(camera_stamp):
                continue
            all_camera_stamps.append(camera_stamp)
            if bool((row.get("cameraMeasurement") or {}).get("velocity_impact_detected", False)):
                detected_impact_stamps.add(round(camera_stamp, 7))
            if not bool(row.get("rawValid", False)):
                continue
            valid_row_count += 1
            position = _finite_vector(row.get("rawBall"), 3)
            if position is None or not contact._plausible_ball(position):
                continue
            stamp_key = round(camera_stamp, 7)
            if stamp_key in seen_valid_stamps:
                continue
            seen_valid_stamps.add(stamp_key)
            measurement = row.get("cameraMeasurement") or {}
            if not filter_flags:
                filter_flags = {
                    "clear_buffers_on_loss": measurement.get(
                        "clear_buffers_on_detection_loss_enabled"
                    ),
                    "motion_consistency_gate": measurement.get(
                        "motion_consistency_gate_enabled"
                    ),
                    "physics_constrained_velocity": measurement.get(
                        "physics_constrained_velocity_enabled"
                    ),
                    "trajectory_position_smoothing": measurement.get(
                        "trajectory_position_smoothing_enabled"
                    ),
                    "impact_velocity_reset": measurement.get(
                        "impact_velocity_reset_enabled"
                    ),
                    "surface_to_center_depth": measurement.get(
                        "sensor_surface_to_center_depth_enabled"
                    ),
                }
            disagreement = measurement.get("depth_disagreement_mm", math.nan)
            disagreement = float(disagreement) if disagreement is not None else math.nan
            latency = wall_time - camera_stamp if np.isfinite(wall_time) else math.nan
            samples.append(
                contact.CameraSample(
                    capture_time_s=camera_stamp,
                    source_time_s=float(row.get("sourceTimeS", math.nan)),
                    source_seq=int(row.get("sourceSeq", 0) or 0),
                    observation_seq=int(row.get("ballObservationSeq", 0) or 0),
                    position_m=position,
                    capture_latency_s=float(latency),
                    depth_disagreement_mm=disagreement,
                    depth_rejected=bool(measurement.get("depth_sensor_rejected", False)),
                )
            )

    samples.sort(key=lambda item: item.capture_time_s)
    if not q_rows:
        trace = {
            "time": np.empty((0,), dtype=np.float64),
            "actual_q": np.empty((0, 7), dtype=np.float64),
            "actual_qd": np.empty((0, 7), dtype=np.float64),
        }
    else:
        order = np.argsort(np.asarray(joint_times))
        trace = {
            "time": np.asarray(joint_times, dtype=np.float64)[order],
            "actual_q": np.asarray(q_rows, dtype=np.float64)[order],
            "actual_qd": np.asarray(dq_rows, dtype=np.float64)[order],
        }

    unique_stamps = np.unique(np.asarray(all_camera_stamps, dtype=np.float64))
    intervals = np.diff(unique_stamps)
    positive_intervals = intervals[intervals > 1.0e-6]
    q_array = trace["actual_q"]
    dq_array = trace["actual_qd"]
    quality = {
        "rows": row_count,
        "duration_s": (
            float(last_wall - first_wall)
            if first_wall is not None and last_wall is not None
            else None
        ),
        "raw_valid_rows": valid_row_count,
        "fresh_valid_camera_samples": len(samples),
        "fresh_camera_rate_hz": (
            float(1.0 / np.median(positive_intervals))
            if positive_intervals.size
            else None
        ),
        "camera_interval_p95_ms": (
            float(1000.0 * np.percentile(positive_intervals, 95))
            if positive_intervals.size
            else None
        ),
        "joint_samples": int(len(q_rows)),
        "joint_position_peak_to_peak_rad": (
            np.ptp(q_array, axis=0).tolist() if q_array.size else None
        ),
        "joint_velocity_rms_rad_s": (
            np.sqrt(np.mean(dq_array * dq_array, axis=0)).tolist()
            if dq_array.size
            else None
        ),
        "joint_velocity_abs_max_rad_s": (
            np.max(np.abs(dq_array), axis=0).tolist() if dq_array.size else None
        ),
        "published_command_rows": published_count,
        "policy_executed_rows": policy_executed_count,
        "inferred_contact_rows": inferred_contact_count,
        "detected_impact_count": len(detected_impact_stamps),
        "camera_filter_flags": filter_flags,
    }
    return samples, trace, quality, sorted(detected_impact_stamps)


def _normal_objective(
    experiment: contact.IsolatedContactExperiment,
    events: list[contact.ImpactEvent],
    damping: float,
) -> float:
    if not events:
        return math.inf
    errors = []
    for event in events:
        prediction = experiment.simulate(
            event,
            contact.SIMCODE7_BASELINE["pair_sliding_friction"],
            contact.SIMCODE7_BASELINE["solref_time_s"],
            damping,
        )
        target = float(event.post_relative_local_m_s[2])
        error = (float(prediction[2]) - target) / 0.18
        squared = error * error
        errors.append(squared if squared <= 4.0 else 4.0 * math.sqrt(squared) - 4.0)
    return float(np.mean(errors))


def _candidate_static_event(
    group: str,
    recording: str,
    pre_samples: list[contact.CameraSample],
    post_samples: list[contact.CameraSample],
    trace: dict[str, np.ndarray],
    kinematics: V91RacketKinematics,
) -> tuple[contact.ImpactEvent | None, str]:
    """Static-racket variant tolerant of the logged uncorrected surface depth.

    These recordings have surface-to-center correction disabled and show a
    sizeable absolute depth bias near the racket.  Absolute contact geometry
    is therefore retained as a diagnostic, not used with the old 12 mm gate.
    Flight fit residuals, continuity, velocity reversal, and a broad 90 mm
    proximity gate still have to pass.
    """
    if not contact._window_is_continuous(pre_samples) or not contact._window_is_continuous(
        post_samples
    ):
        return None, "camera_gap"
    combined = pre_samples + post_samples
    if any(sample.depth_rejected for sample in combined):
        return None, "depth_rejected"
    disagreements = [
        sample.depth_disagreement_mm
        for sample in combined
        if np.isfinite(sample.depth_disagreement_mm)
    ]
    max_disagreement = max(disagreements, default=0.0)
    if max_disagreement > 120.0:
        return None, "depth_disagreement"
    left = pre_samples[-1].capture_time_s
    right = post_samples[0].capture_time_s
    if not 0.006 <= right - left <= 0.024:
        return None, "impact_bracket_gap"
    reference = 0.5 * (left + right)
    pre_fit = contact._fit_flight(pre_samples, reference)
    post_fit = contact._fit_flight(post_samples, reference)
    if pre_fit.rmse_m > 0.006 or post_fit.rmse_m > 0.006:
        return None, "fit_rmse"
    if pre_fit.max_residual_m > 0.014 or post_fit.max_residual_m > 0.014:
        return None, "fit_max_residual"

    result = minimize_scalar(
        lambda query: float(
            np.sum((pre_fit.position(query) - post_fit.position(query)) ** 2)
        ),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1.0e-7},
    )
    impact_time = float(result.x)
    pre_position = pre_fit.position(impact_time)
    post_position = post_fit.position(impact_time)
    continuity = float(np.linalg.norm(pre_position - post_position))
    if continuity > 0.010:
        return None, "position_discontinuity"
    impact_position = 0.5 * (pre_position + post_position)
    pre_velocity = pre_fit.velocity(impact_time)
    post_velocity = post_fit.velocity(impact_time)
    if np.linalg.norm(post_velocity - pre_velocity) < 0.55:
        return None, "weak_velocity_jump"

    racket = kinematics.state(trace, impact_time)
    normal = racket.rotation[:, 2].copy()
    if float(np.dot(impact_position - racket.center_m, normal)) < 0.0:
        normal *= -1.0
    ball_contact = impact_position - normal * contact.BALL_RADIUS_M
    contact_offset = ball_contact - racket.center_m
    normal_surface_error = abs(
        float(np.dot(contact_offset, normal))
        - contact.RACKET_RUBBER_HALF_THICKNESS_M
    )
    tangent_offset = contact_offset - normal * float(np.dot(contact_offset, normal))
    lateral_offset = float(np.linalg.norm(tangent_offset))
    if normal_surface_error > 0.090:
        return None, "broad_normal_geometry"
    if lateral_offset > 0.100:
        return None, "broad_lateral_geometry"

    racket_contact_velocity = racket.linear_velocity_m_s + np.cross(
        racket.angular_velocity_rad_s, contact_offset
    )
    pre_relative = pre_velocity - racket_contact_velocity
    post_relative = post_velocity - racket_contact_velocity
    pre_normal = float(np.dot(pre_relative, normal))
    post_normal = float(np.dot(post_relative, normal))
    if not (pre_normal < -0.10 and post_normal > 0.10):
        return None, "relative_normal_reversal"
    restitution = -post_normal / pre_normal
    if not 0.20 <= restitution <= 1.20:
        return None, "restitution_range"

    delta_velocity = post_velocity - pre_velocity
    delta_normal = float(np.dot(delta_velocity, normal))
    delta_tangent = delta_velocity - normal * delta_normal
    friction_ratio = float(
        np.linalg.norm(delta_tangent) / max(abs(delta_normal), 1.0e-9)
    )
    pre_tangent = pre_relative - normal * pre_normal
    post_tangent = post_relative - normal * post_normal
    tangent_ratio = float(
        np.linalg.norm(post_tangent) / max(np.linalg.norm(pre_tangent), 1.0e-9)
    )
    frame = contact._local_frame(racket.rotation, normal)
    local_pre = frame.T @ pre_relative
    local_post = frame.T @ post_relative
    local_offset = frame.T @ tangent_offset
    score = (
        1000.0 * (pre_fit.rmse_m + post_fit.rmse_m + continuity)
        + 80.0 * normal_surface_error
        + 20.0 * lateral_offset
        + 0.02 * max_disagreement
    )
    return (
        contact.ImpactEvent(
            group=group,
            recording=recording,
            impact_time_s=impact_time,
            impact_position_m=impact_position.tolist(),
            pre_velocity_m_s=pre_velocity.tolist(),
            post_velocity_m_s=post_velocity.tolist(),
            racket_center_m=racket.center_m.tolist(),
            racket_normal=normal.tolist(),
            racket_contact_velocity_m_s=racket_contact_velocity.tolist(),
            tangent_offset_m=local_offset[:2].tolist(),
            pre_relative_local_m_s=local_pre.tolist(),
            post_relative_local_m_s=local_post.tolist(),
            normal_restitution=float(restitution),
            impulse_friction_ratio=friction_ratio,
            tangential_speed_ratio=tangent_ratio,
            pre_fit_rmse_mm=1000.0 * pre_fit.rmse_m,
            post_fit_rmse_mm=1000.0 * post_fit.rmse_m,
            continuity_error_mm=1000.0 * continuity,
            normal_surface_error_mm=1000.0 * normal_surface_error,
            lateral_contact_offset_mm=1000.0 * lateral_offset,
            max_capture_latency_ms=1000.0
            * max(sample.capture_latency_s for sample in combined),
            max_depth_disagreement_mm=float(max_disagreement),
            score=float(score),
        ),
        "accepted",
    )


def _find_static_impacts(
    group: str,
    recording: str,
    samples: list[contact.CameraSample],
    trace: dict[str, np.ndarray],
    kinematics: V91RacketKinematics,
    window_samples: int = 6,
) -> tuple[list[contact.ImpactEvent], Counter[str]]:
    candidates: list[contact.ImpactEvent] = []
    reasons: Counter[str] = Counter()
    for split in range(window_samples, len(samples) - window_samples + 1):
        event, reason = _candidate_static_event(
            group,
            recording,
            samples[split - window_samples : split],
            samples[split : split + window_samples],
            trace,
            kinematics,
        )
        reasons[reason] += 1
        if event is not None:
            candidates.append(event)
    candidates.sort(key=lambda item: (item.score, item.impact_time_s))
    selected: list[contact.ImpactEvent] = []
    for candidate in candidates:
        if any(
            abs(candidate.impact_time_s - kept.impact_time_s) < 0.12
            for kept in selected
        ):
            reasons["duplicate_candidate"] += 1
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item.impact_time_s)
    return selected, reasons


def _event_at_detected_impact(
    group: str,
    recording: str,
    samples: list[contact.CameraSample],
    detected_time_s: float,
    trace: dict[str, np.ndarray],
    kinematics: V91RacketKinematics,
    window_samples: int = 6,
) -> tuple[contact.ImpactEvent | None, str]:
    """Fit flights around a detector event while excluding transition frames."""
    def select_window(before: bool) -> list[contact.CameraSample]:
        if before:
            candidates = [
                sample
                for sample in samples
                if detected_time_s - 0.18
                <= sample.capture_time_s
                <= detected_time_s - 0.025
            ]
        else:
            candidates = [
                sample
                for sample in samples
                if detected_time_s
                <= sample.capture_time_s
                <= detected_time_s + 0.13
            ]
        ranked: list[tuple[float, list[contact.CameraSample]]] = []
        for index in range(len(candidates) - window_samples + 1):
            window = candidates[index : index + window_samples]
            if not contact._window_is_continuous(window):
                continue
            fit = contact._fit_flight(window, detected_time_s)
            normal_velocity = float(fit.velocity(detected_time_s)[2])
            if before and normal_velocity >= -0.10:
                continue
            if not before and normal_velocity <= 0.10:
                continue
            gap = (
                detected_time_s - window[-1].capture_time_s
                if before
                else window[0].capture_time_s - detected_time_s
            )
            score = fit.rmse_m + 0.25 * fit.max_residual_m + 0.100 * gap
            ranked.append((score, window))
        return min(ranked, key=lambda item: item[0])[1] if ranked else []

    pre_samples = select_window(before=True)
    post_samples = select_window(before=False)
    if len(pre_samples) < window_samples or len(post_samples) < window_samples:
        return None, "detected_window_short"
    if not contact._window_is_continuous(pre_samples) or not contact._window_is_continuous(
        post_samples
    ):
        return None, "camera_gap"
    combined = pre_samples + post_samples
    if any(sample.depth_rejected for sample in combined):
        return None, "depth_rejected"
    disagreements = [
        sample.depth_disagreement_mm
        for sample in combined
        if np.isfinite(sample.depth_disagreement_mm)
    ]
    max_disagreement = max(disagreements, default=0.0)
    if max_disagreement > 120.0:
        return None, "depth_disagreement"
    reference = detected_time_s
    pre_fit = contact._fit_flight(pre_samples, reference)
    post_fit = contact._fit_flight(post_samples, reference)
    if pre_fit.rmse_m > 0.006 or post_fit.rmse_m > 0.006:
        return None, "fit_rmse"
    if pre_fit.max_residual_m > 0.014 or post_fit.max_residual_m > 0.014:
        return None, "fit_max_residual"
    left = pre_samples[-1].capture_time_s
    right = post_samples[0].capture_time_s
    if not 0.024 <= right - left <= 0.100:
        return None, "transition_gap"
    result = minimize_scalar(
        lambda query: float(
            np.sum((pre_fit.position(query) - post_fit.position(query)) ** 2)
        ),
        bounds=(left, right),
        method="bounded",
        options={"xatol": 1.0e-7},
    )
    impact_time = float(result.x)
    pre_position = pre_fit.position(impact_time)
    post_position = post_fit.position(impact_time)
    continuity = float(np.linalg.norm(pre_position - post_position))
    # Default collection had trajectory-position smoothing enabled.  At impact
    # the filter introduces a visible 1--2 frame position transition even when
    # the independently fitted pre/post velocities remain clean.  Retain up to
    # 100 mm for first-impact sensitivity analysis and report the discontinuity
    # explicitly; the convergence decision below remains conservative.
    if continuity > 0.100:
        return None, "position_discontinuity"
    impact_position = 0.5 * (pre_position + post_position)
    pre_velocity = pre_fit.velocity(impact_time)
    post_velocity = post_fit.velocity(impact_time)
    racket = kinematics.state(trace, impact_time)
    normal = racket.rotation[:, 2].copy()
    if float(np.dot(impact_position - racket.center_m, normal)) < 0.0:
        normal *= -1.0
    ball_contact = impact_position - normal * contact.BALL_RADIUS_M
    contact_offset = ball_contact - racket.center_m
    tangent_offset = contact_offset - normal * float(np.dot(contact_offset, normal))
    lateral_offset = float(np.linalg.norm(tangent_offset))
    normal_surface_error = abs(
        float(np.dot(contact_offset, normal))
        - contact.RACKET_RUBBER_HALF_THICKNESS_M
    )
    if normal_surface_error > 0.090:
        return None, "broad_normal_geometry"
    if lateral_offset > 0.075:
        return None, "racket_radius_geometry"
    racket_contact_velocity = racket.linear_velocity_m_s + np.cross(
        racket.angular_velocity_rad_s, contact_offset
    )
    pre_relative = pre_velocity - racket_contact_velocity
    post_relative = post_velocity - racket_contact_velocity
    pre_normal = float(np.dot(pre_relative, normal))
    post_normal = float(np.dot(post_relative, normal))
    if not (pre_normal < -0.10 and post_normal > 0.10):
        return None, "relative_normal_reversal"
    restitution = -post_normal / pre_normal
    if not 0.20 <= restitution <= 1.20:
        return None, "restitution_range"
    delta_velocity = post_velocity - pre_velocity
    delta_normal = float(np.dot(delta_velocity, normal))
    delta_tangent = delta_velocity - normal * delta_normal
    pre_tangent = pre_relative - normal * pre_normal
    post_tangent = post_relative - normal * post_normal
    frame = contact._local_frame(racket.rotation, normal)
    local_pre = frame.T @ pre_relative
    local_post = frame.T @ post_relative
    local_offset = frame.T @ tangent_offset
    score = (
        1000.0 * (pre_fit.rmse_m + post_fit.rmse_m + continuity)
        + 80.0 * normal_surface_error
        + 20.0 * lateral_offset
        + 0.02 * max_disagreement
    )
    return (
        contact.ImpactEvent(
            group=group,
            recording=recording,
            impact_time_s=impact_time,
            impact_position_m=impact_position.tolist(),
            pre_velocity_m_s=pre_velocity.tolist(),
            post_velocity_m_s=post_velocity.tolist(),
            racket_center_m=racket.center_m.tolist(),
            racket_normal=normal.tolist(),
            racket_contact_velocity_m_s=racket_contact_velocity.tolist(),
            tangent_offset_m=local_offset[:2].tolist(),
            pre_relative_local_m_s=local_pre.tolist(),
            post_relative_local_m_s=local_post.tolist(),
            normal_restitution=float(restitution),
            impulse_friction_ratio=float(
                np.linalg.norm(delta_tangent) / max(abs(delta_normal), 1.0e-9)
            ),
            tangential_speed_ratio=float(
                np.linalg.norm(post_tangent)
                / max(np.linalg.norm(pre_tangent), 1.0e-9)
            ),
            pre_fit_rmse_mm=1000.0 * pre_fit.rmse_m,
            post_fit_rmse_mm=1000.0 * post_fit.rmse_m,
            continuity_error_mm=1000.0 * continuity,
            normal_surface_error_mm=1000.0 * normal_surface_error,
            lateral_contact_offset_mm=1000.0 * lateral_offset,
            max_capture_latency_ms=1000.0
            * max(sample.capture_latency_s for sample in combined),
            max_depth_disagreement_mm=float(max_disagreement),
            score=float(score),
        ),
        "accepted",
    )


def _find_detected_impacts(
    group: str,
    recording: str,
    samples: list[contact.CameraSample],
    detected_times_s: list[float],
    trace: dict[str, np.ndarray],
    kinematics: V91RacketKinematics,
) -> tuple[list[contact.ImpactEvent], Counter[str]]:
    events: list[contact.ImpactEvent] = []
    reasons: Counter[str] = Counter()
    for detected_time in detected_times_s:
        event, reason = _event_at_detected_impact(
            group,
            recording,
            samples,
            detected_time,
            trace,
            kinematics,
        )
        reasons[reason] += 1
        if event is not None:
            events.append(event)
    return events, reasons


def _fit_damping(
    experiment: contact.IsolatedContactExperiment,
    events: list[contact.ImpactEvent],
) -> tuple[float | None, float | None]:
    if not events:
        return None, None
    result = minimize_scalar(
        lambda value: _normal_objective(experiment, events, float(value)),
        bounds=(0.05, 0.50),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    return float(result.x), float(result.fun)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    params = summary["recommended_parameters"]
    lines = [
        "# V91 静态球拍落球碰撞辨识（2026-09-01）",
        "",
        "## 结论",
        "",
        summary["decision"]["message_zh"],
        "",
        "## 数据审计",
        "",
        "| 录制 | 新鲜有效相机点 | 入选碰撞 | 指令发布行 | 最大关节速度 (rad/s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["records"]:
        maximum = row["quality"].get("joint_velocity_abs_max_rad_s")
        maximum = max(maximum) if maximum else math.nan
        lines.append(
            f"| {row['recording']} | {row['quality']['fresh_valid_camera_samples']} | "
            f"{row['selected_impacts']} | {row['quality']['published_command_rows']} | "
            f"{maximum:.5f} |"
        )
    lines.extend(
        [
            "",
            "## 实物碰撞统计",
            "",
            f"- 入选碰撞数：{aggregate['count']}。",
            f"- 法向恢复系数中位数：{aggregate['normal_restitution']['median']}。",
            f"- 法向恢复系数 P10–P90：{aggregate['normal_restitution']['p10']}–"
            f"{aggregate['normal_restitution']['p90']}。",
            f"- 拟合前/后轨迹 RMSE 中位数："
            f"{aggregate['fit_quality']['pre_rmse_median_mm']} / "
            f"{aggregate['fit_quality']['post_rmse_median_mm']} mm。",
            "",
            "## CPU MuJoCo 代理拟合",
            "",
            f"- `friction={params['pair_sliding_friction']}`（当前数据没有球自旋，保留先验）。",
            f"- CPU 代理候选 `solref=\"{params['solref_time_s']} "
            f"{params['solref_damping_ratio']}\"`，不是生产 MJX 推荐值。",
            f"- `solimp={params['solimp']}`，`margin={params['margin_m']}`，"
            f"`condim={params['condim']}`。",
            "",
            "90 Hz 球轨迹不能独立辨识毫秒级接触时间、刚度和阻尼，也不能把摩擦与未知球自旋分开。"
            "因此只把法向恢复最敏感的 `solref` 阻尼比作为本批数据的候选拟合量。",
            "",
            "## CPU 代理收敛检查",
            "",
            f"- 留一录制拟合阻尼范围：{summary['convergence']['leave_one_recording_out_damping_range']}。",
            f"- 逐录制恢复系数中位数范围：{summary['convergence']['per_recording_restitution_median_range']}。",
            f"- CPU 代理判定：`{summary['decision']['status']}`。",
            "- 每段录制只使用释放球后的首次球拍碰撞，后续反弹全部排除。",
            "- 本批采集开启了轨迹位置平滑，且 surface-to-center 深度修正实际未启用；"
            "绝对接触位置存在明显系统偏差。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--recording", action="append", default=[])
    parser.add_argument("--start", default="probe_recording_20260901_142344")
    parser.add_argument("--end", default="probe_recording_20260901_143346")
    parser.add_argument("--fk-model", type=Path, default=DEFAULT_FK_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_root = args.record_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.recording:
        recording_dirs = [record_root / name for name in args.recording]
    else:
        recording_dirs = [
            path
            for path in sorted(record_root.glob("probe_recording_*"))
            if args.start <= path.name <= args.end
        ]
    if not recording_dirs:
        raise SystemExit("No recordings matched the requested range")

    kinematics = V91RacketKinematics(
        args.fk_model.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
    )
    records: list[dict[str, Any]] = []
    all_events: list[contact.ImpactEvent] = []
    rejection_counts: Counter[str] = Counter()
    for recording_dir in recording_dirs:
        control_path = recording_dir / "control_samples.jsonl"
        if not control_path.is_file():
            records.append({"recording": recording_dir.name, "status": "missing_control"})
            continue
        samples, trace, quality, detected_times = _load_recording(control_path)
        if trace["time"].size == 0:
            records.append(
                {
                    "recording": recording_dir.name,
                    "status": "missing_joint_feedback",
                    "quality": quality,
                }
            )
            continue
        events, reasons = _find_detected_impacts(
            recording_dir.name,
            recording_dir.name,
            samples,
            detected_times[:1],
            trace,
            kinematics,
        )
        all_events.extend(events)
        rejection_counts.update(reasons)
        records.append(
            {
                "recording": recording_dir.name,
                "status": "processed",
                "quality": quality,
                "selected_impacts": len(events),
                "candidate_counts": dict(reasons),
            }
        )
        print(
            f"[{recording_dir.name}] camera={len(samples)} impacts={len(events)}",
            flush=True,
        )

    experiment = contact.IsolatedContactExperiment()
    damping, objective = _fit_damping(experiment, all_events)
    per_recording_events = {
        directory.name: [event for event in all_events if event.recording == directory.name]
        for directory in recording_dirs
    }
    loo_rows = []
    for held_out in per_recording_events:
        fit_events = [
            event for event in all_events if event.recording != held_out
        ]
        value, loss = _fit_damping(experiment, fit_events)
        loo_rows.append(
            {
                "held_out_recording": held_out,
                "fit_event_count": len(fit_events),
                "solref_damping_ratio": value,
                "objective": loss,
            }
        )
    finite_loo = [
        row["solref_damping_ratio"]
        for row in loo_rows
        if row["solref_damping_ratio"] is not None
    ]
    per_record_medians = {
        name: (
            float(np.median([event.normal_restitution for event in events]))
            if events
            else None
        )
        for name, events in per_recording_events.items()
    }
    finite_record_medians = [
        value for value in per_record_medians.values() if value is not None
    ]

    event_count = len(all_events)
    recording_event_count = sum(bool(events) for events in per_recording_events.values())
    loo_span = max(finite_loo) - min(finite_loo) if finite_loo else math.inf
    restitution_span = (
        max(finite_record_medians) - min(finite_record_medians)
        if finite_record_medians
        else math.inf
    )
    converged = bool(
        event_count >= 5
        and recording_event_count >= 4
        and loo_span <= 0.05
        and restitution_span <= 0.20
    )
    decision = {
        "status": "converged_candidate" if converged else "insufficient_or_not_converged",
        "apply_to_production_xml": False,
        "message_zh": (
            f"这 {recording_event_count} 组数据形成了可重复的候选法向碰撞拟合；"
            "仍先在独立 MJX 验证中确认后再改生产 XML。"
            if converged
            else (
                f"这 {recording_event_count} 组数据尚不足以证明碰撞参数收敛，"
                "不应直接覆盖生产 MJX XML。"
            )
        ),
    }
    recommended = {
        "ball_mass_kg": contact.BALL_MASS_KG,
        "pair_sliding_friction": float(
            contact.SIMCODE7_BASELINE["pair_sliding_friction"]
        ),
        "solref_time_s": float(contact.SIMCODE7_BASELINE["solref_time_s"]),
        "solref_damping_ratio": damping,
        "solimp": contact.SIMCODE7_BASELINE["solimp"],
        "margin_m": contact.SIMCODE7_BASELINE["margin_m"],
        "condim": contact.SIMCODE7_BASELINE["condim"],
        "fit_backend": "CPU MuJoCo isolated-contact proxy",
        "fit_scope": "normal restitution only; friction/time retained as priors",
    }
    baseline = (
        float(contact.SIMCODE7_BASELINE["pair_sliding_friction"]),
        float(contact.SIMCODE7_BASELINE["solref_time_s"]),
        float(contact.SIMCODE7_BASELINE["solref_damping_ratio"]),
    )
    fitted = (
        recommended["pair_sliding_friction"],
        recommended["solref_time_s"],
        float(damping if damping is not None else baseline[2]),
    )
    summary = {
        "schema_version": 2,
        "experiment": "static_racket_first_free_drop_v91_send_command_off",
        "records": records,
        "rejection_counts": dict(rejection_counts),
        "aggregate": contact._event_aggregate(all_events),
        "per_recording": {
            name: contact._event_aggregate(events)
            for name, events in per_recording_events.items()
        },
        "recommended_parameters": recommended,
        "optimization": {
            "status": "normal_only_constrained_fit",
            "objective": objective,
            "leave_one_recording_out": loo_rows,
        },
        "convergence": {
            "selected_event_count": event_count,
            "recordings_with_events": recording_event_count,
            "leave_one_recording_out_damping_range": (
                [min(finite_loo), max(finite_loo)] if finite_loo else None
            ),
            "leave_one_recording_out_damping_span": loo_span,
            "per_recording_restitution_median": per_record_medians,
            "per_recording_restitution_median_range": (
                [min(finite_record_medians), max(finite_record_medians)]
                if finite_record_medians
                else None
            ),
            "per_recording_restitution_median_span": restitution_span,
            "criteria": {
                "minimum_events": 5,
                "minimum_recordings_with_events": 4,
                "maximum_loo_damping_span": 0.05,
                "maximum_per_recording_restitution_span": 0.20,
            },
        },
        "simulation_validation": {
            "baseline": contact._event_prediction_metrics(
                experiment, all_events, baseline
            ),
            "fitted": contact._event_prediction_metrics(
                experiment, all_events, fitted
            ),
        },
        "decision": decision,
        "events": [asdict(event) for event in all_events],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "recommended_contact_params.json").write_text(
        json.dumps(recommended, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contact._write_events_csv(output / "selected_impacts.csv", all_events)
    _write_report(output / "STATIC_CONTACT_ANALYSIS_ZH.md", summary)
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_impacts": event_count,
                "recordings_with_events": recording_event_count,
                "recommended_parameters": recommended,
                "decision": decision,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
