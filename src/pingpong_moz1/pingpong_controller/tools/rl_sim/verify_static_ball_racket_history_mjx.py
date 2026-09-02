#!/usr/bin/env python3
"""Reconstruct recorded drop states and replay the first bounce in MJX.

This is the history-based counterpart of
``verify_static_ball_racket_contact_mjx.py``.  It deliberately does not infer a
release height from one fitted impact velocity.  For every recording it:

* finds the release-to-first-impact segment in the fresh camera positions;
* robustly fits the repository's continuous gravity/drag prediction model;
* validates that state with a causal, leave-tail-out position prediction;
* starts MJX from the fitted state at the beginning of that measured segment;
* identifies contact damping from the first rebound position trajectory.

Absolute camera-to-racket depth bias is removed with one rigid translation:
the incoming fitted trajectory at the independently detected first-impact time
is placed at the known racket contact plane.  Neither the release height nor
the initial velocity is reconstructed from the old impact-velocity estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import savgol_filter

import identify_ball_racket_contact as contact
import identify_static_ball_racket_contact as static_contact


WORKSPACE = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = WORKSPACE / "src/pingpong_moz1"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.high_ball_occlusion_predictor import (  # noqa: E402
    propagate_paper_model,
)


CURRENT_PRODUCTION_DAMPING = 0.152540
PAPER_DRAG_PER_M = 0.13545
GRAVITY_M_S2 = 9.8
FIT_INTEGRATION_STEP_S = 0.001
CONTACT_CENTER_Z_M = contact.BALL_RADIUS_M + 0.001
INCOMING_END_MARGIN_S = 0.018
POST_START_MARGIN_S = 0.035
POST_END_MARGIN_S = 0.235
NEXT_IMPACT_SEARCH_START_S = 0.180
NEXT_IMPACT_SEARCH_END_S = 0.800
NEXT_IMPACT_EXCLUSION_MARGIN_S = 0.015


@dataclass(frozen=True)
class HistoryFit:
    """Continuous position/state fit referenced to ``reference_time_s``."""

    model: str
    drag_coefficient_per_m: float
    reference_time_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    sample_count: int
    span_s: float
    rmse_m: float
    axis_rmse_m: np.ndarray
    max_residual_m: float

    def state(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray]:
        return _propagate(
            self.position_m,
            self.velocity_m_s,
            float(timestamp_s) - self.reference_time_s,
            self.drag_coefficient_per_m,
        )

    def positions(self, timestamps_s: Iterable[float]) -> np.ndarray:
        return np.asarray(
            [self.state(float(timestamp))[0] for timestamp in timestamps_s],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "drag_coefficient_per_m": self.drag_coefficient_per_m,
            "reference_time_s": self.reference_time_s,
            "position_m": self.position_m.tolist(),
            "velocity_m_s": self.velocity_m_s.tolist(),
            "sample_count": self.sample_count,
            "span_s": self.span_s,
            "rmse_mm": 1000.0 * self.rmse_m,
            "axis_rmse_mm": (1000.0 * self.axis_rmse_m).tolist(),
            "max_residual_mm": 1000.0 * self.max_residual_m,
        }


@dataclass(frozen=True)
class ReconstructedRecord:
    recording: str
    event: contact.ImpactEvent
    incoming_samples: tuple[contact.CameraSample, ...]
    post_samples: tuple[contact.CameraSample, ...]
    incoming_fit: HistoryFit
    incoming_no_drag_fit: HistoryFit
    causal_fit: HistoryFit
    causal_holdout_rmse_m: float
    causal_holdout_axis_rmse_m: np.ndarray
    no_drag_causal_holdout_rmse_m: float
    post_fit: HistoryFit
    frame: np.ndarray
    contact_center_local_m: np.ndarray
    start_position_local_m: np.ndarray
    start_velocity_local_m_s: np.ndarray
    gravity_local_m_s2: np.ndarray
    target_times_from_start_s: np.ndarray
    target_positions_local_m: np.ndarray
    next_impact_time_s: float | None
    first_rebound_cutoff_time_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(
            "analysis_results/ball_racket_contact_identification_v91_static_20260901"
        ),
    )
    parser.add_argument(
        "--record-root",
        type=Path,
        default=static_contact.DEFAULT_RECORD_ROOT,
    )
    parser.add_argument("--reconstruct-only", action="store_true")
    return parser.parse_args()


def _propagate(
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    elapsed_s: float,
    drag_coefficient_per_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate forward, or analytically reverse the short fit reference."""
    elapsed = float(elapsed_s)
    if elapsed < -1.0e-12:
        # Fits are normally referenced at or before all their samples.  The
        # post-impact fit is referenced at impact, so this path is retained
        # only as a safe gravity-only fallback.
        if drag_coefficient_per_m > 0.0:
            raise ValueError("paper drag propagation does not support negative time")
        gravity = np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)
        position = position_m + velocity_m_s * elapsed + 0.5 * gravity * elapsed**2
        velocity = velocity_m_s + gravity * elapsed
        return position, velocity
    return propagate_paper_model(
        position_m,
        velocity_m_s,
        elapsed,
        drag_coefficient_per_m=drag_coefficient_per_m,
        gravity_m_s2=GRAVITY_M_S2,
        integration_step_s=FIT_INTEGRATION_STEP_S,
    )


def _initial_state_guess(
    times_s: np.ndarray,
    positions_m: np.ndarray,
    reference_time_s: float,
) -> np.ndarray:
    elapsed = times_s - reference_time_s
    design = np.column_stack((np.ones_like(elapsed), elapsed))
    state = np.zeros(6, dtype=np.float64)
    for axis in (0, 1):
        coefficient, *_ = np.linalg.lstsq(design, positions_m[:, axis], rcond=None)
        state[axis] = coefficient[0]
        state[3 + axis] = coefficient[1]
    gravity_compensated_z = positions_m[:, 2] + 0.5 * GRAVITY_M_S2 * elapsed**2
    coefficient, *_ = np.linalg.lstsq(design, gravity_compensated_z, rcond=None)
    state[2] = coefficient[0]
    state[5] = coefficient[1]
    return state


def _fit_history(
    samples: Iterable[contact.CameraSample],
    *,
    reference_time_s: float | None = None,
    drag_coefficient_per_m: float = PAPER_DRAG_PER_M,
) -> HistoryFit:
    rows = tuple(samples)
    if len(rows) < 6:
        raise ValueError(f"at least six history samples are required, got {len(rows)}")
    times = np.asarray([row.capture_time_s for row in rows], dtype=np.float64)
    positions = np.asarray([row.position_m for row in rows], dtype=np.float64)
    reference = float(times[0] if reference_time_s is None else reference_time_s)
    if np.min(times - reference) < -1.0e-9:
        raise ValueError("fit reference must not be later than a measurement")
    initial = _initial_state_guess(times, positions, reference)

    def predict(state: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _propagate(
                    state[:3], state[3:], float(timestamp - reference),
                    drag_coefficient_per_m,
                )[0]
                for timestamp in times
            ],
            dtype=np.float64,
        )

    result = least_squares(
        lambda state: (predict(state) - positions).reshape(-1),
        initial,
        loss="soft_l1",
        f_scale=0.0025,
        max_nfev=250,
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
    )
    prediction = predict(result.x)
    residual = prediction - positions
    norms = np.linalg.norm(residual, axis=1)
    model = "paper_gravity_quadratic_drag" if drag_coefficient_per_m else "gravity_only"
    return HistoryFit(
        model=model,
        drag_coefficient_per_m=float(drag_coefficient_per_m),
        reference_time_s=reference,
        position_m=np.asarray(result.x[:3], dtype=np.float64),
        velocity_m_s=np.asarray(result.x[3:], dtype=np.float64),
        sample_count=len(rows),
        span_s=float(times[-1] - times[0]),
        rmse_m=float(np.sqrt(np.mean(residual * residual))),
        axis_rmse_m=np.sqrt(np.mean(residual * residual, axis=0)),
        max_residual_m=float(np.max(norms)),
    )


def _last_continuous_segment(
    samples: list[contact.CameraSample],
    impact_time_s: float,
) -> list[contact.CameraSample]:
    usable = [
        sample
        for sample in samples
        if impact_time_s - 0.50 <= sample.capture_time_s
        <= impact_time_s - INCOMING_END_MARGIN_S
        and not sample.depth_rejected
        and (
            not np.isfinite(sample.depth_disagreement_mm)
            or sample.depth_disagreement_mm <= 120.0
        )
    ]
    if not usable:
        return []
    start = 0
    for index, gap in enumerate(np.diff([row.capture_time_s for row in usable])):
        if gap > 0.035:
            start = index + 1
    return usable[start:]


def _release_to_impact_samples(
    samples: list[contact.CameraSample],
    impact_time_s: float,
) -> list[contact.CameraSample]:
    segment = _last_continuous_segment(samples, impact_time_s)
    if len(segment) < 10:
        raise ValueError(f"only {len(segment)} continuous pre-impact samples")
    # Detect the release change point by scanning gravity-constrained falling
    # fits.  A last-maximum rule is fragile when a long held-ball plateau has a
    # 1--2 mm depth drift: it can select the beginning of the plateau and make
    # the fitted release velocity spuriously positive.  The score below favors
    # a low-residual ballistic suffix whose velocity at its first sample is
    # near zero.  It therefore selects the plateau-to-fall transition while
    # still supporting recordings that begin after the release.
    candidates: list[tuple[float, int]] = []
    for start in range(0, len(segment) - 9):
        suffix = segment[start:]
        times = np.asarray(
            [row.capture_time_s - suffix[0].capture_time_s for row in suffix],
            dtype=np.float64,
        )
        if times[-1] < 0.10:
            continue
        z = np.asarray([row.position_m[2] for row in suffix], dtype=np.float64)
        design = np.column_stack((np.ones_like(times), times))
        gravity_compensated = z + 0.5 * GRAVITY_M_S2 * times * times
        coefficient, *_ = np.linalg.lstsq(
            design, gravity_compensated, rcond=None
        )
        fitted = design @ coefficient - 0.5 * GRAVITY_M_S2 * times * times
        vertical_rmse = float(np.sqrt(np.mean((fitted - z) ** 2)))
        initial_vertical_speed = float(coefficient[1])
        score = vertical_rmse + 0.003 * abs(initial_vertical_speed)
        candidates.append((score, start))
    start = min(candidates)[1] if candidates else max(0, len(segment) - 10)
    falling = segment[start:]
    if len(falling) < 10 or falling[-1].capture_time_s - falling[0].capture_time_s < 0.10:
        # Short recordings can start after release.  They still contain a
        # usable falling state, so use their complete last continuous segment.
        falling = segment
    return falling


def _next_impact_time(
    samples: list[contact.CameraSample],
    event: contact.ImpactEvent,
) -> float | None:
    """Detect the next racket impact only to establish a hard data boundary."""
    normal = np.asarray(event.racket_normal, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    racket_center = np.asarray(event.racket_center_m, dtype=np.float64)
    rows = [
        sample
        for sample in samples
        if event.impact_time_s + NEXT_IMPACT_SEARCH_START_S
        <= sample.capture_time_s
        <= event.impact_time_s + NEXT_IMPACT_SEARCH_END_S
        and not sample.depth_rejected
    ]
    if len(rows) < 7:
        return None
    projected = np.asarray(
        [float(np.dot(row.position_m - racket_center, normal)) for row in rows],
        dtype=np.float64,
    )
    smooth = savgol_filter(projected, 7, 2)
    first_impact_height = float(
        np.dot(np.asarray(event.impact_position_m) - racket_center, normal)
    )
    for index in range(2, len(rows) - 2):
        is_local_minimum = (
            smooth[index] <= smooth[index - 1]
            and smooth[index] < smooth[index + 1]
            and np.mean(np.diff(smooth[index - 2 : index + 1])) < 0.0
            and np.mean(np.diff(smooth[index : index + 3])) > 0.0
        )
        near_racket = projected[index] <= first_impact_height + 0.025
        if is_local_minimum and near_racket:
            return float(rows[index].capture_time_s)
    return None


def _first_rebound_samples(
    samples: list[contact.CameraSample],
    event: contact.ImpactEvent,
) -> tuple[list[contact.CameraSample], float | None, float]:
    next_impact = _next_impact_time(samples, event)
    cutoff = event.impact_time_s + POST_END_MARGIN_S
    if next_impact is not None:
        cutoff = min(cutoff, next_impact - NEXT_IMPACT_EXCLUSION_MARGIN_S)
    rows = [
        sample
        for sample in samples
        if event.impact_time_s + POST_START_MARGIN_S <= sample.capture_time_s
        <= cutoff
        and not sample.depth_rejected
        and (
            not np.isfinite(sample.depth_disagreement_mm)
            or sample.depth_disagreement_mm <= 120.0
        )
    ]
    if len(rows) < 8:
        raise ValueError(f"only {len(rows)} clean first-rebound samples")
    return rows, next_impact, float(cutoff)


def _causal_holdout(
    samples: list[contact.CameraSample],
    drag_coefficient_per_m: float,
) -> tuple[HistoryFit, float, np.ndarray, int]:
    holdout_count = max(4, min(6, len(samples) // 3))
    train = samples[:-holdout_count]
    holdout = samples[-holdout_count:]
    fit = _fit_history(train, drag_coefficient_per_m=drag_coefficient_per_m)
    times = np.asarray([row.capture_time_s for row in holdout], dtype=np.float64)
    positions = np.asarray([row.position_m for row in holdout], dtype=np.float64)
    residual = fit.positions(times) - positions
    return (
        fit,
        float(np.sqrt(np.mean(residual * residual))),
        np.sqrt(np.mean(residual * residual, axis=0)),
        holdout_count,
    )


def _frame_from_normal(normal_value: Iterable[float]) -> np.ndarray:
    normal = np.asarray(normal_value, dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent_x -= normal * float(np.dot(tangent_x, normal))
    if np.linalg.norm(tangent_x) < 1.0e-6:
        tangent_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent_x -= normal * float(np.dot(tangent_x, normal))
    tangent_x /= np.linalg.norm(tangent_x)
    tangent_y = np.cross(normal, tangent_x)
    tangent_y /= np.linalg.norm(tangent_y)
    return np.column_stack((tangent_x, tangent_y, normal))


def _reconstruct_record(
    event: contact.ImpactEvent,
    samples: list[contact.CameraSample],
) -> ReconstructedRecord:
    incoming_samples = _release_to_impact_samples(samples, event.impact_time_s)
    post_samples, next_impact_time, first_rebound_cutoff = (
        _first_rebound_samples(samples, event)
    )
    incoming_fit = _fit_history(
        incoming_samples, drag_coefficient_per_m=PAPER_DRAG_PER_M
    )
    incoming_no_drag_fit = _fit_history(
        incoming_samples, drag_coefficient_per_m=0.0
    )
    causal_fit, holdout_rmse, holdout_axis_rmse, _ = _causal_holdout(
        incoming_samples, PAPER_DRAG_PER_M
    )
    _, no_drag_holdout_rmse, _, _ = _causal_holdout(incoming_samples, 0.0)
    post_fit = _fit_history(
        post_samples,
        reference_time_s=event.impact_time_s,
        drag_coefficient_per_m=PAPER_DRAG_PER_M,
    )

    frame = _frame_from_normal(event.racket_normal)
    normal = frame[:, 2]
    racket_center = np.asarray(event.racket_center_m, dtype=np.float64)
    incoming_at_impact_position, _ = incoming_fit.state(event.impact_time_s)
    # Use the extrapolated incoming position for the contact's tangential
    # coordinate as well.  The old pre/post average is visibly displaced by
    # the enabled impact smoothing, especially in the low-quality recording.
    # Only the unobservable absolute normal bias is replaced by the known
    # racket plane.
    tangent_world = incoming_at_impact_position - racket_center
    tangent_world -= normal * float(np.dot(tangent_world, normal))
    contact_center = np.array(
        [
            float(np.dot(tangent_world, frame[:, 0])),
            float(np.dot(tangent_world, frame[:, 1])),
            CONTACT_CENTER_Z_M,
        ],
        dtype=np.float64,
    )

    start_time = incoming_fit.reference_time_s
    start_position = contact_center + frame.T @ (
        incoming_fit.position_m - incoming_at_impact_position
    )
    racket_velocity = np.asarray(event.racket_contact_velocity_m_s, dtype=np.float64)
    start_velocity = frame.T @ (incoming_fit.velocity_m_s - racket_velocity)
    gravity_local = frame.T @ np.array([0.0, 0.0, -GRAVITY_M_S2], dtype=np.float64)

    target_times = np.asarray(
        [sample.capture_time_s - start_time for sample in post_samples],
        dtype=np.float64,
    )
    post_at_impact, _ = post_fit.state(event.impact_time_s)
    post_positions = post_fit.positions(
        [sample.capture_time_s for sample in post_samples]
    )
    target_positions = contact_center + (post_positions - post_at_impact) @ frame
    return ReconstructedRecord(
        recording=event.recording,
        event=event,
        incoming_samples=tuple(incoming_samples),
        post_samples=tuple(post_samples),
        incoming_fit=incoming_fit,
        incoming_no_drag_fit=incoming_no_drag_fit,
        causal_fit=causal_fit,
        causal_holdout_rmse_m=holdout_rmse,
        causal_holdout_axis_rmse_m=holdout_axis_rmse,
        no_drag_causal_holdout_rmse_m=no_drag_holdout_rmse,
        post_fit=post_fit,
        frame=frame,
        contact_center_local_m=contact_center,
        start_position_local_m=start_position,
        start_velocity_local_m_s=start_velocity,
        gravity_local_m_s2=gravity_local,
        target_times_from_start_s=target_times,
        target_positions_local_m=target_positions,
        next_impact_time_s=next_impact_time,
        first_rebound_cutoff_time_s=first_rebound_cutoff,
    )


def _reconstruction_row(row: ReconstructedRecord) -> dict[str, Any]:
    event = row.event
    incoming_contact_position, incoming_contact_velocity = row.incoming_fit.state(
        event.impact_time_s
    )
    _, no_drag_contact_velocity = row.incoming_no_drag_fit.state(event.impact_time_s)
    post_contact_position, post_contact_velocity = row.post_fit.state(event.impact_time_s)
    normal = row.frame[:, 2]
    old_incoming = np.asarray(event.pre_velocity_m_s, dtype=np.float64)
    old_outgoing = np.asarray(event.post_velocity_m_s, dtype=np.float64)
    history_incoming_normal = float(np.dot(incoming_contact_velocity, normal))
    history_outgoing_normal = float(np.dot(post_contact_velocity, normal))
    old_incoming_normal = float(np.dot(old_incoming, normal))
    old_outgoing_normal = float(np.dot(old_outgoing, normal))
    target_z = row.target_positions_local_m[:, 2]
    return {
        "recording": row.recording,
        "impact_time_s": event.impact_time_s,
        "incoming_window": {
            "start_s_before_impact": event.impact_time_s
            - row.incoming_samples[0].capture_time_s,
            "end_s_before_impact": event.impact_time_s
            - row.incoming_samples[-1].capture_time_s,
            "sample_count": len(row.incoming_samples),
        },
        "post_first_rebound_window": {
            "start_s_after_impact": row.post_samples[0].capture_time_s
            - event.impact_time_s,
            "end_s_after_impact": row.post_samples[-1].capture_time_s
            - event.impact_time_s,
            "sample_count": len(row.post_samples),
            "hard_cutoff_s_after_first_impact": (
                row.first_rebound_cutoff_time_s - event.impact_time_s
            ),
            "next_impact_detected_s_after_first_impact": (
                None
                if row.next_impact_time_s is None
                else row.next_impact_time_s - event.impact_time_s
            ),
            "uses_samples_at_or_after_next_impact": False,
        },
        "incoming_paper_fit": row.incoming_fit.as_dict(),
        "incoming_gravity_only_fit": row.incoming_no_drag_fit.as_dict(),
        "causal_leave_tail_out": {
            "training_sample_count": row.causal_fit.sample_count,
            "holdout_sample_count": len(row.incoming_samples)
            - row.causal_fit.sample_count,
            "paper_drag_position_rmse_mm": 1000.0 * row.causal_holdout_rmse_m,
            "paper_drag_axis_rmse_mm": (
                1000.0 * row.causal_holdout_axis_rmse_m
            ).tolist(),
            "gravity_only_position_rmse_mm": 1000.0
            * row.no_drag_causal_holdout_rmse_m,
        },
        "post_first_rebound_paper_fit": row.post_fit.as_dict(),
        "history_state_at_impact": {
            "incoming_position_m": incoming_contact_position.tolist(),
            "incoming_velocity_m_s": incoming_contact_velocity.tolist(),
            "incoming_normal_velocity_m_s": history_incoming_normal,
            "outgoing_position_m": post_contact_position.tolist(),
            "outgoing_velocity_m_s": post_contact_velocity.tolist(),
            "outgoing_normal_velocity_m_s": history_outgoing_normal,
            "normal_restitution": -history_outgoing_normal
            / min(history_incoming_normal, -1.0e-9),
            "gravity_only_incoming_normal_velocity_m_s": float(
                np.dot(no_drag_contact_velocity, normal)
            ),
        },
        "old_six_frame_impact_fit_for_diagnostic_only": {
            "incoming_normal_velocity_m_s": old_incoming_normal,
            "outgoing_normal_velocity_m_s": old_outgoing_normal,
            "normal_restitution": -old_outgoing_normal
            / min(old_incoming_normal, -1.0e-9),
        },
        "mjx_initial_state": {
            "time_s_before_impact": event.impact_time_s
            - row.incoming_fit.reference_time_s,
            "position_local_m": row.start_position_local_m.tolist(),
            "velocity_local_m_s": row.start_velocity_local_m_s.tolist(),
            "gravity_local_m_s2": row.gravity_local_m_s2.tolist(),
            "contact_center_local_m": row.contact_center_local_m.tolist(),
        },
        "target_first_rebound_normal_position_range_m": [
            float(np.min(target_z)),
            float(np.max(target_z)),
        ],
    }


def _write_reconstruction(
    analysis_dir: Path,
    reconstructed: list[ReconstructedRecord],
) -> dict[str, Any]:
    rows = [_reconstruction_row(row) for row in reconstructed]
    paper_holdout = np.asarray(
        [row.causal_holdout_rmse_m for row in reconstructed], dtype=np.float64
    )
    no_drag_holdout = np.asarray(
        [row.no_drag_causal_holdout_rmse_m for row in reconstructed], dtype=np.float64
    )
    history_e = np.asarray(
        [row["history_state_at_impact"]["normal_restitution"] for row in rows],
        dtype=np.float64,
    )
    scope_rows = []
    separation_gaps_ms = []
    for row in rows:
        window = row["post_first_rebound_window"]
        next_impact = window["next_impact_detected_s_after_first_impact"]
        gap_ms = (
            None
            if next_impact is None
            else 1000.0 * (next_impact - window["end_s_after_impact"])
        )
        if gap_ms is not None:
            separation_gaps_ms.append(gap_ms)
        scope_rows.append(
            {
                "recording": row["recording"],
                "first_impact_time_s": row["impact_time_s"],
                "used_rebound_start_s_after_first_impact": window[
                    "start_s_after_impact"
                ],
                "used_rebound_end_s_after_first_impact": window[
                    "end_s_after_impact"
                ],
                "next_impact_detected_s_after_first_impact": next_impact,
                "used_to_next_impact_separation_ms": gap_ms,
                "later_juggle_samples_used": False,
            }
        )
    scope_audit = {
        "one_first_impact_selected_per_recording": True,
        "later_juggles_used": False,
        "hard_stop_before_next_detected_impact": True,
        "record_count": len(scope_rows),
        "minimum_used_to_next_impact_separation_ms": (
            float(min(separation_gaps_ms)) if separation_gaps_ms else None
        ),
        "records": scope_rows,
    }
    result = {
        "schema_version": 1,
        "method": (
            "release-to-impact fresh positions -> robust continuous paper gravity/drag "
            "state; causal leave-tail-out validation; first rebound positions only; "
            "hard stop before the next detected impact"
        ),
        "one_event_per_recording": True,
        "later_juggles_used": False,
        "first_bounce_scope_audit": scope_audit,
        "does_not_use_impact_velocity_to_infer_release_height": True,
        "paper_model": {
            "gravity_m_s2": GRAVITY_M_S2,
            "drag_coefficient_per_m": PAPER_DRAG_PER_M,
            "integration_step_s": FIT_INTEGRATION_STEP_S,
        },
        "record_count": len(rows),
        "aggregate": {
            "paper_drag_causal_holdout_rmse_median_mm": float(
                1000.0 * np.median(paper_holdout)
            ),
            "paper_drag_causal_holdout_rmse_max_mm": float(
                1000.0 * np.max(paper_holdout)
            ),
            "gravity_only_causal_holdout_rmse_median_mm": float(
                1000.0 * np.median(no_drag_holdout)
            ),
            "history_restitution_median": float(np.median(history_e)),
            "history_restitution_range": [
                float(np.min(history_e)),
                float(np.max(history_e)),
            ],
        },
        "records": rows,
    }
    (analysis_dir / "history_state_reconstruction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "first_bounce_scope_audit.json").write_text(
        json.dumps(scope_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _run_mjx(
    reconstructed: list[ReconstructedRecord],
    summary: dict[str, Any],
) -> dict[str, Any]:
    import jax
    from jax import lax
    import mujoco
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)
    experiment = contact.IsolatedContactExperiment()
    experiment.model.opt.gravity[:] = (0.0, 0.0, 0.0)
    friction = float(contact.SIMCODE7_BASELINE["pair_sliding_friction"])
    solref_time_s = float(contact.SIMCODE7_BASELINE["solref_time_s"])
    experiment.model.pair_friction[experiment.pair_id, 0] = friction
    experiment.model.pair_solref[experiment.pair_id] = (
        solref_time_s,
        CURRENT_PRODUCTION_DAMPING,
    )
    mjx_model = mjx.put_model(experiment.model)
    base_data = mjx.make_data(mjx_model)
    timestep = float(experiment.model.opt.timestep)
    maximum_duration = max(
        float(row.target_times_from_start_s[-1]) + 0.010 for row in reconstructed
    )
    step_count = max(1, int(math.ceil(maximum_duration / timestep)))

    qpos_rows = []
    qvel_rows = []
    gravity_rows = []
    for row in reconstructed:
        qpos = np.asarray(base_data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(base_data.qvel, dtype=np.float64).copy()
        qpos[experiment.ball_qadr : experiment.ball_qadr + 3] = (
            row.start_position_local_m
        )
        qpos[experiment.ball_qadr + 3 : experiment.ball_qadr + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        qvel[experiment.ball_vadr : experiment.ball_vadr + 3] = (
            row.start_velocity_local_m_s
        )
        qvel[experiment.ball_vadr + 3 : experiment.ball_vadr + 6] = 0.0
        qpos_rows.append(qpos)
        qvel_rows.append(qvel)
        gravity_rows.append(row.gravity_local_m_s2)

    qpos_batch = jax.numpy.asarray(np.asarray(qpos_rows))
    qvel_batch = jax.numpy.asarray(np.asarray(qvel_rows))
    gravity_batch = jax.numpy.asarray(np.asarray(gravity_rows))
    ball_mass = float(contact.BALL_MASS_KG)

    @jax.jit
    def rollout_batch(damping_value, qpos_values, qvel_values, gravity_values):
        model = mjx_model.replace(
            pair_solref=mjx_model.pair_solref.at[experiment.pair_id, 1].set(
                damping_value
            )
        )

        def rollout(qpos, qvel, gravity):
            data = base_data.replace(qpos=qpos, qvel=qvel)

            def step(data, _):
                velocity = data.qvel[
                    experiment.ball_vadr : experiment.ball_vadr + 3
                ]
                acceleration = gravity - PAPER_DRAG_PER_M * (
                    jax.numpy.linalg.norm(velocity) * velocity
                )
                force = ball_mass * acceleration
                xfrc = data.xfrc_applied.at[experiment.ball_body_id, :3].set(force)
                data = data.replace(xfrc_applied=xfrc)
                data = mjx.step(model, data)
                return data, (
                    data.qpos[experiment.ball_qadr : experiment.ball_qadr + 3],
                    data.qvel[experiment.ball_vadr : experiment.ball_vadr + 3],
                )

            _, trajectory = lax.scan(step, data, xs=None, length=step_count)
            return trajectory

        return jax.vmap(rollout)(qpos_values, qvel_values, gravity_values)

    target_indices = [
        np.clip(
            np.rint(row.target_times_from_start_s / timestep).astype(int) - 1,
            0,
            step_count - 1,
        )
        for row in reconstructed
    ]
    high_quality_mask = np.asarray(
        [
            row.event.continuity_error_mm <= 60.0
            and row.event.pre_fit_rmse_mm <= 3.0
            and row.event.post_fit_rmse_mm <= 3.0
            and row.event.max_depth_disagreement_mm <= 100.0
            for row in reconstructed
        ],
        dtype=bool,
    )
    trajectory_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def trajectories(damping: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(damping), 9)
        if key not in trajectory_cache:
            position, velocity = rollout_batch(
                float(damping), qpos_batch, qvel_batch, gravity_batch
            )
            trajectory_cache[key] = (
                np.asarray(position, dtype=np.float64),
                np.asarray(velocity, dtype=np.float64),
            )
        return trajectory_cache[key]

    def event_normal_errors(damping: float) -> list[np.ndarray]:
        position, _ = trajectories(damping)
        return [
            position[index, indices, 2] - row.target_positions_local_m[:, 2]
            for index, (row, indices) in enumerate(zip(reconstructed, target_indices))
        ]

    def objective(damping: float, mask: np.ndarray) -> float:
        losses = []
        for keep, errors in zip(mask, event_normal_errors(float(damping))):
            if not keep:
                continue
            scaled = errors / 0.010
            squared = scaled * scaled
            robust = np.where(
                squared <= 4.0,
                squared,
                4.0 * np.sqrt(squared) - 4.0,
            )
            losses.append(float(np.mean(robust)))
        return float(np.mean(losses)) if losses else math.inf

    damping_grid = np.linspace(0.05, 0.50, 181)

    def fit(mask: np.ndarray, *, include_profile: bool = False) -> dict[str, Any]:
        # MJX contact response is not guaranteed to be unimodal in positive
        # solref damping.  Search the full admissible interval first, then
        # refine only the best grid basin; a direct bounded golden search can
        # otherwise report a local solution as if it were global.
        grid_objective = np.asarray(
            [objective(float(value), mask) for value in damping_grid],
            dtype=np.float64,
        )
        best_index = int(np.argmin(grid_objective))
        lower = float(damping_grid[max(0, best_index - 1)])
        upper = float(damping_grid[min(len(damping_grid) - 1, best_index + 1)])
        if upper - lower > 1.0e-9:
            result = minimize_scalar(
                lambda value: objective(float(value), mask),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 2.0e-6},
            )
            fitted_value = float(result.x)
            fitted_objective = float(result.fun)
            success = bool(result.success)
        else:
            fitted_value = float(damping_grid[best_index])
            fitted_objective = float(grid_objective[best_index])
            success = True
        output = {
            "solref_damping_ratio": fitted_value,
            "objective": fitted_objective,
            "success": success,
            "event_count": int(np.sum(mask)),
            "global_grid_step": float(damping_grid[1] - damping_grid[0]),
            "best_grid_damping_ratio": float(damping_grid[best_index]),
            "best_grid_objective": float(grid_objective[best_index]),
        }
        if include_profile:
            output["profile_damping_grid"] = damping_grid.tolist()
            output["profile_objective"] = grid_objective.tolist()
        return output

    all_mask = np.ones(len(reconstructed), dtype=bool)
    fit_all = fit(all_mask, include_profile=True)
    fit_high_quality = fit(high_quality_mask)
    old_candidate = float(
        summary.get("candidate_mjx_parameters", {}).get(
            "solref_damping_ratio_all_five", 0.35613038706791794
        )
    )
    candidates = {
        "current_production": CURRENT_PRODUCTION_DAMPING,
        "old_contact_state_candidate": old_candidate,
        "history_position_fit_all": fit_all["solref_damping_ratio"],
        "history_position_fit_high_quality": fit_high_quality[
            "solref_damping_ratio"
        ],
    }

    def evaluate(name: str, damping: float) -> dict[str, Any]:
        position, velocity = trajectories(damping)
        errors = event_normal_errors(damping)
        rows = []
        for index, (record, error) in enumerate(zip(reconstructed, errors)):
            impact_index = int(
                round(
                    (record.event.impact_time_s - record.incoming_fit.reference_time_s)
                    / timestep
                )
            )
            start = max(0, impact_index - int(round(0.020 / timestep)))
            end = min(step_count, impact_index + int(round(0.260 / timestep)))
            local_z = position[index, start:end, 2]
            local_vz = velocity[index, start:end, 2]
            minimum_relative = int(
                np.argmin(local_z[: max(1, int(round(0.060 / timestep)))])
            )
            minimum_index = start + minimum_relative
            rebound_end = min(step_count, impact_index + int(round(0.240 / timestep)))
            outgoing_vz = float(np.max(velocity[index, minimum_index:rebound_end, 2]))
            apex_z = float(np.max(position[index, minimum_index:rebound_end, 2]))
            target_apex_z = float(np.max(record.target_positions_local_m[:, 2]))
            target_post_velocity = record.frame.T @ record.post_fit.velocity_m_s
            rows.append(
                {
                    "recording": record.recording,
                    "high_quality": bool(high_quality_mask[index]),
                    "status": (
                        "first_rebound_found"
                        if outgoing_vz > 0.05 and apex_z > CONTACT_CENTER_Z_M + 0.001
                        else "no_first_rebound"
                    ),
                    "normal_position_rmse_mm": float(
                        1000.0 * np.sqrt(np.mean(error * error))
                    ),
                    "normal_position_bias_mm": float(1000.0 * np.mean(error)),
                    "simulated_first_apex_above_contact_mm": 1000.0
                    * (apex_z - CONTACT_CENTER_Z_M),
                    "history_fit_first_apex_above_contact_mm": 1000.0
                    * (target_apex_z - CONTACT_CENTER_Z_M),
                    "simulated_outgoing_normal_velocity_m_s": outgoing_vz,
                    "history_fit_outgoing_normal_velocity_m_s": float(
                        target_post_velocity[2]
                    ),
                    "minimum_normal_velocity_m_s": float(np.min(local_vz)),
                }
            )
        normal_rmse = np.asarray(
            [row["normal_position_rmse_mm"] for row in rows], dtype=np.float64
        )
        high_quality_errors = np.concatenate(
            [
                error
                for keep, error in zip(high_quality_mask, errors)
                if keep
            ]
        )
        high_quality_row_rmse = np.asarray(
            [
                row["normal_position_rmse_mm"]
                for row in rows
                if row["high_quality"]
            ],
            dtype=np.float64,
        )
        apex_error = np.asarray(
            [
                row["simulated_first_apex_above_contact_mm"]
                - row["history_fit_first_apex_above_contact_mm"]
                for row in rows
            ],
            dtype=np.float64,
        )
        return {
            "name": name,
            "parameters": {
                "pair_sliding_friction": friction,
                "solref_time_s": solref_time_s,
                "solref_damping_ratio": damping,
            },
            "first_rebound_count": sum(
                row["status"] == "first_rebound_found" for row in rows
            ),
            "requested_count": len(rows),
            "per_record_normal_position_rmse_median_mm": float(
                np.median(normal_rmse)
            ),
            "all_position_samples_normal_rmse_mm": float(
                1000.0
                * np.sqrt(
                    np.mean(
                        np.concatenate(event_normal_errors(damping)) ** 2
                    )
                )
            ),
            "high_quality_position_samples_normal_rmse_mm": float(
                1000.0
                * np.sqrt(np.mean(high_quality_errors * high_quality_errors))
            ),
            "high_quality_per_record_normal_rmse_max_mm": float(
                np.max(high_quality_row_rmse)
            ),
            "first_apex_error_rmse_mm": float(
                np.sqrt(np.mean(apex_error * apex_error))
            ),
            "rows": rows,
        }

    evaluations = {
        name: evaluate(name, damping) for name, damping in candidates.items()
    }
    loo = []
    for index, record in enumerate(reconstructed):
        mask = all_mask.copy()
        mask[index] = False
        fitted = fit(mask)
        fitted["held_out_recording"] = record.recording
        loo.append(fitted)
    per_recording_fit = []
    for index, record in enumerate(reconstructed):
        mask = np.zeros(len(reconstructed), dtype=bool)
        mask[index] = True
        fitted = fit(mask)
        fitted["recording"] = record.recording
        per_recording_fit.append(fitted)
    loo_values = [row["solref_damping_ratio"] for row in loo]
    per_recording_values = [
        row["solref_damping_ratio"] for row in per_recording_fit
    ]
    candidate_evaluation = evaluations["history_position_fit_all"]
    high_quality_per_recording_values = [
        row["solref_damping_ratio"]
        for row, keep in zip(per_recording_fit, high_quality_mask)
        if keep
    ]
    stable = bool(
        candidate_evaluation["first_rebound_count"] == len(reconstructed)
        and int(np.sum(high_quality_mask)) >= 4
        and max(loo_values) - min(loo_values) <= 0.05
        and abs(
            fit_all["solref_damping_ratio"]
            - fit_high_quality["solref_damping_ratio"]
        )
        <= 0.05
        and max(high_quality_per_recording_values)
        - min(high_quality_per_recording_values)
        <= 0.05
        and candidate_evaluation[
            "high_quality_position_samples_normal_rmse_mm"
        ]
        <= 20.0
        and candidate_evaluation[
            "high_quality_per_record_normal_rmse_max_mm"
        ]
        <= 30.0
    )
    return {
        "schema_version": 1,
        "backend": "MJX history-state full drop and first-rebound position replay",
        "mujoco_version": mujoco.__version__,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "timestep_s": timestep,
        "steps": step_count,
        "duration_s": step_count * timestep,
        "initialization": (
            "fitted state at the first measured falling sample; rigid position-history "
            "alignment at detected impact; no v^2/(2g) release-height inference"
        ),
        "flight_force": (
            "per-record local gravity plus paper quadratic drag applied as MJX xfrc"
        ),
        "target": "continuous first-rebound normal positions, not logged velocity",
        "fit_all": fit_all,
        "fit_high_quality": fit_high_quality,
        "leave_one_recording_out": loo,
        "leave_one_recording_out_damping_range": [min(loo_values), max(loo_values)],
        "leave_one_recording_out_damping_span": max(loo_values) - min(loo_values),
        "all_vs_high_quality_damping_delta": abs(
            fit_all["solref_damping_ratio"]
            - fit_high_quality["solref_damping_ratio"]
        ),
        "per_recording_fit": per_recording_fit,
        "per_recording_damping_range": [
            min(per_recording_values),
            max(per_recording_values),
        ],
        "per_recording_damping_span": max(per_recording_values)
        - min(per_recording_values),
        "high_quality_per_recording_damping_range": [
            min(high_quality_per_recording_values),
            max(high_quality_per_recording_values),
        ],
        "high_quality_per_recording_damping_span": (
            max(high_quality_per_recording_values)
            - min(high_quality_per_recording_values)
        ),
        "stability_thresholds": {
            "leave_one_recording_out_damping_span_max": 0.05,
            "high_quality_per_recording_damping_span_max": 0.05,
            "all_vs_high_quality_damping_delta_max": 0.05,
            "high_quality_position_samples_normal_rmse_max_mm": 20.0,
            "high_quality_per_record_normal_rmse_max_mm": 30.0,
            "first_rebound_required_for_all": True,
        },
        "stable_candidate": stable,
        "evaluations": evaluations,
    }


def _update_outputs(
    analysis_dir: Path,
    summary: dict[str, Any],
    reconstruction: dict[str, Any],
    mjx_result: dict[str, Any],
) -> None:
    (analysis_dir / "mjx_history_replay.json").write_text(
        json.dumps(mjx_result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["historical_fall_reconstruction"] = reconstruction
    summary["mjx_history_replay"] = mjx_result
    old_full_drop = (
        summary.get("mjx_validation", {}).get("full_drop_validation")
    )
    if isinstance(old_full_drop, dict):
        old_full_drop["status"] = "superseded_velocity_derived_sensitivity_only"
        old_full_drop["used_for_decision"] = False
    stable = bool(mjx_result["stable_candidate"])
    fitted = float(mjx_result["fit_all"]["solref_damping_ratio"])
    record_count = int(reconstruction["record_count"])
    high_quality_count = int(mjx_result["fit_high_quality"]["event_count"])
    summary["candidate_mjx_parameters"][
        "solref_damping_ratio_history_position_fit"
    ] = fitted
    summary["decision"]["status"] = (
        "history_mjx_candidate_stable_but_collection_biased"
        if stable
        else "history_mjx_candidate_not_converged"
    )
    summary["decision"]["mjx_candidate_stable"] = stable
    summary["decision"]["apply_to_production_xml"] = False
    summary["decision"]["message_zh"] = (
        f"已从 {record_count} 组连续下落位置历史重建状态并完成 MJX 首次反弹位置回放；"
        f"历史轨迹候选阻尼比为 {fitted:.6f}。候选通过当前数据的稳定性门槛，"
        "但采集时位置平滑开启且一组深度/连续性质量偏低，因此仍不覆盖生产 XML。"
        if stable
        else (
            "已改用连续下落位置历史重建状态并完成 MJX 首次反弹位置回放，"
            "但候选参数仍未通过留一录制稳定性门槛；本轮不覆盖生产 XML。"
        )
    )
    summary_path = analysis_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    evaluation = mjx_result["evaluations"]["history_position_fit_all"]
    current = mjx_result["evaluations"]["current_production"]
    report_path = analysis_dir / "STATIC_CONTACT_ANALYSIS_ZH.md"
    report = report_path.read_text(encoding="utf-8").rstrip()
    report = report.split("\n\n## 基于历史状态的 MJX 回放", maxsplit=1)[0].rstrip()
    report = report.replace(
        "## MJX 完整释放—首次反弹回放\n\n",
        "## 已废弃的速度反推释放敏感性检查\n\n"
        "**本节用 `v²/(2g)` 构造释放高度，不满足历史状态重建要求，"
        "不参与参数结论；其中旧版相邻步过零判据还会漏报缓慢过零的反弹。**\n\n",
    )
    report = report.replace(
        "- 生产 XML 更新：`false`。完整释放回放中候选参数只有 3/5 次产生首次反弹，"
        "不能复现实物五次实验；本轮不覆盖生产 XML。",
        "- 生产 XML 更新：`false`。这里的速度初值直接拟合仅作接触映射诊断；"
        "完整历史状态结论见后文。",
    )
    conclusion_marker = "## 结论\n\n"
    data_marker = "\n\n## 数据审计"
    if conclusion_marker in report and data_marker in report:
        prefix, conclusion_and_rest = report.split(conclusion_marker, maxsplit=1)
        _, rest = conclusion_and_rest.split(data_marker, maxsplit=1)
        report = (
            prefix
            + conclusion_marker
            + summary["decision"]["message_zh"]
            + data_marker
            + rest
        )
    aggregate = reconstruction["aggregate"]
    report += (
        "\n\n## 基于历史状态的 MJX 回放\n\n"
        "上一节按碰撞入射速度换算 `v²/(2g)` 的完整释放试验只保留为敏感性检查，"
        "不再参与结论。这里从每组首次碰撞前的连续 `rawBall` 位置历史拟合仓库中的"
        "重力—二次阻力预测模型，并从最早的实测下落状态启动 MJX；释放高度和初速度"
        "都不由碰撞速度反推。首次反弹目标同样来自碰后 35–235 ms 的连续位置轨迹。\n\n"
        f"- 因果留尾预测 RMSE 中位数/最大值："
        f"`{aggregate['paper_drag_causal_holdout_rmse_median_mm']:.2f}` / "
        f"`{aggregate['paper_drag_causal_holdout_rmse_max_mm']:.2f} mm`。\n"
        f"- 历史位置轨迹拟合阻尼比（{record_count} 组/高质量 {high_quality_count} 组）："
        f"`{mjx_result['fit_all']['solref_damping_ratio']:.6f}` / "
        f"`{mjx_result['fit_high_quality']['solref_damping_ratio']:.6f}`。\n"
        f"- 留一录制范围："
        f"`{mjx_result['leave_one_recording_out_damping_range'][0]:.6f}–"
        f"{mjx_result['leave_one_recording_out_damping_range'][1]:.6f}`。\n"
        f"- 逐录制单独拟合范围："
        f"`{mjx_result['per_recording_damping_range'][0]:.6f}–"
        f"{mjx_result['per_recording_damping_range'][1]:.6f}`。\n"
        f"- 当前生产参数首次反弹：`{current['first_rebound_count']}/{record_count}`；"
        f"历史轨迹候选：`{evaluation['first_rebound_count']}/{record_count}`。\n"
        f"- 当前/历史候选的首次反弹法向位置 RMSE："
        f"`{current['all_position_samples_normal_rmse_mm']:.2f}` / "
        f"`{evaluation['all_position_samples_normal_rmse_mm']:.2f} mm`。\n"
        f"- 生产 XML 更新：`false`。{summary['decision']['message_zh']}\n\n"
        "| 录制 | 下落点数 | 留尾预测 RMSE (mm) | 历史拟合 e | 候选反弹位置 RMSE (mm) | 质量 |\n"
        "|---|---:|---:|---:|---:|---|\n"
    )
    reconstruction_by_name = {
        row["recording"]: row for row in reconstruction["records"]
    }
    for result_row in evaluation["rows"]:
        row = reconstruction_by_name[result_row["recording"]]
        report += (
            f"| {result_row['recording']} | "
            f"{row['incoming_window']['sample_count']} | "
            f"{row['causal_leave_tail_out']['paper_drag_position_rmse_mm']:.2f} | "
            f"{row['history_state_at_impact']['normal_restitution']:.3f} | "
            f"{result_row['normal_position_rmse_mm']:.2f} | "
            f"{'高' if result_row['high_quality'] else '低'} |\n"
        )
    report_path.write_text(report, encoding="utf-8")

    recommendation_path = analysis_dir / "recommended_contact_params.json"
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    recommendation["status"] = (
        "history_candidate_only_do_not_apply"
        if stable
        else "history_candidate_not_converged_do_not_apply"
    )
    recommendation["production_xml_updated"] = False
    recommendation["mjx_candidate_history_position_fit"] = fitted
    recommendation["reason"] = summary["decision"]["message_zh"]
    recommendation["supersedes_velocity_derived_full_drop_conclusion"] = True
    recommendation_path.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    record_root = args.record_root.expanduser().resolve()
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    event_fields = {item.name for item in fields(contact.ImpactEvent)}
    events = [
        contact.ImpactEvent(
            **{key: value for key, value in row.items() if key in event_fields}
        )
        for row in summary["events"]
    ]
    reconstructed = []
    for event in events:
        control_path = record_root / event.recording / "control_samples.jsonl"
        samples, _, _, _ = static_contact._load_recording(control_path)
        row = _reconstruct_record(event, samples)
        reconstructed.append(row)
        print(
            f"[{event.recording}] fall={len(row.incoming_samples)} "
            f"holdout={1000.0 * row.causal_holdout_rmse_m:.2f}mm "
            f"post={len(row.post_samples)}",
            flush=True,
        )
    reconstruction = _write_reconstruction(analysis_dir, reconstructed)
    summary["first_bounce_scope_audit"] = reconstruction[
        "first_bounce_scope_audit"
    ]
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.reconstruct_only:
        print(json.dumps(reconstruction, ensure_ascii=False, indent=2))
        return
    mjx_result = _run_mjx(reconstructed, summary)
    _update_outputs(analysis_dir, summary, reconstruction, mjx_result)
    print(json.dumps(mjx_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
