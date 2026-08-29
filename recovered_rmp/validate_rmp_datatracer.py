#!/usr/bin/env python3
"""Validate the recovered joint RMP against a 1 kHz Movax DataTracer export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovered_rmp.params import load_movax_rmp_config
from recovered_rmp.trajectory import JointRMPPlanner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="NPZ made by extract_movax_datatracer.py")
    parser.add_argument("recording", type=Path, help="recording directory or control_samples.jsonl")
    parser.add_argument("config", type=Path, help="copied Movax model YAML")
    parser.add_argument("--mech-unit", default="RightArm")
    parser.add_argument(
        "--recording-to-tracer-offset-s",
        type=float,
        default=-3.0209988619056563,
        help="add to recording wall-time relative seconds to obtain tracer-relative time",
    )
    parser.add_argument(
        "--command-delay-ms",
        type=float,
        default=3.0,
        help="sub-millisecond-clock alignment from recorded publish to planner receipt",
    )
    parser.add_argument("--evaluation-start-s", type=float, default=1.410)
    parser.add_argument("--evaluation-end-s", type=float, default=5.000)
    parser.add_argument(
        "--include-feedback-reference-baseline",
        action="store_true",
        help=(
            "also compare the recorded RMP reference directly with encoder "
            "feedback after a per-joint continuous delay fit"
        ),
    )
    parser.add_argument("--feedback-delay-min-ms", type=float, default=5.0)
    parser.add_argument("--feedback-delay-max-ms", type=float, default=25.0)
    parser.add_argument("--feedback-delay-step-ms", type=float, default=0.05)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser.parse_args()


def load_events(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = path / "control_samples.jsonl" if path.is_dir() else path
    wall_time: list[float] = []
    target: list[np.ndarray] = []
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            command = row.get("command")
            timestamp = row.get("wallTimeS")
            if not isinstance(command, dict) or not bool(command.get("published")):
                continue
            position = command.get("position")
            if not isinstance(timestamp, (int, float)) or not isinstance(position, list):
                continue
            value = np.asarray(position, dtype=float)
            if value.shape != (7,) or not np.all(np.isfinite(value)):
                continue
            wall_time.append(float(timestamp))
            target.append(value)
    if len(wall_time) < 2:
        raise ValueError(f"no usable published commands in {source}")
    time = np.asarray(wall_time)
    values = np.stack(target)
    monotonic = np.concatenate(([True], np.diff(time) > 0.0))
    return time[monotonic], values[monotonic]


def rms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def _feedback_reference_baseline(
    time: np.ndarray,
    reference_q: np.ndarray,
    reference_qd: np.ndarray,
    reference_qdd: np.ndarray,
    actual_q: np.ndarray,
    actual_qd: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    delay_min_ms: float,
    delay_max_ms: float,
    delay_step_ms: float,
) -> dict[str, object]:
    """Report the irreducible error of a reference-only feedback model.

    This is deliberately not a plant fit: it asks how close an ideal simulator
    could get if it emitted Movax's *recorded final RMP reference* and only
    modeled the observed downstream delay.  Any remaining acceleration error
    belongs to the unobserved servo/mechanical/feedback chain, not to the
    recovered upstream planner.
    """

    if delay_step_ms <= 0.0 or delay_max_ms < delay_min_ms:
        raise ValueError("feedback delay grid must have a positive step and ordered bounds")
    dt = float(np.median(np.diff(time)))
    actual_qdd = np.full_like(actual_qd, np.nan)
    actual_qdd[:-1] = np.diff(actual_qd, axis=0) / np.diff(time)[:, None]
    valid = evaluation_mask.copy()
    valid[-1] = False
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        raise ValueError("feedback baseline has fewer than two valid samples")
    delay_grid_ms = np.arange(
        delay_min_ms,
        delay_max_ms + 0.5 * delay_step_ms,
        delay_step_ms,
    )
    delay_ms: list[float] = []
    q_errors: list[np.ndarray] = []
    qd_errors: list[np.ndarray] = []
    qdd_errors: list[np.ndarray] = []
    for joint in range(reference_q.shape[1]):
        best: tuple[float, float, np.ndarray, np.ndarray, np.ndarray] | None = None
        for delay in delay_grid_ms:
            shifted_time = time[indices] - float(delay) * 1.0e-3
            q_error = np.interp(shifted_time, time, reference_q[:, joint]) - actual_q[indices, joint]
            qd_error = np.interp(shifted_time, time, reference_qd[:, joint]) - actual_qd[indices, joint]
            qdd_error = np.interp(shifted_time, time, reference_qdd[:, joint]) - actual_qdd[indices, joint]
            # Threshold-normalized combined score selects one physical delay
            # for all three channels; it cannot separately time-shift qdd.
            score = (
                rms(q_error) / np.deg2rad(0.1)
            ) ** 2 + (rms(qd_error) / 0.05) ** 2 + rms(qdd_error) ** 2
            if best is None or score < best[0]:
                best = (score, float(delay), q_error, qd_error, qdd_error)
        assert best is not None
        _, selected_delay, q_error, qd_error, qdd_error = best
        delay_ms.append(selected_delay)
        q_errors.append(q_error)
        qd_errors.append(qd_error)
        qdd_errors.append(qdd_error)
    q_error_array = np.stack(q_errors, axis=-1)
    qd_error_array = np.stack(qd_errors, axis=-1)
    qdd_error_array = np.stack(qdd_errors, axis=-1)
    return {
        "definition": (
            "recorded final RMP reference delayed per joint versus encoder "
            "feedback; actual qdd is the 1 kHz forward difference of actual qd"
        ),
        "samples": int(indices.size),
        "feedback_delay_ms_per_joint": delay_ms,
        "position_rms_rad": rms(q_error_array),
        "position_rms_deg": float(np.rad2deg(rms(q_error_array))),
        "velocity_rms_rad_s": rms(qd_error_array),
        "acceleration_rms_rad_s2": rms(qdd_error_array),
        "position_rms_per_joint_rad": np.sqrt(np.mean(q_error_array**2, axis=0)).tolist(),
        "velocity_rms_per_joint_rad_s": np.sqrt(np.mean(qd_error_array**2, axis=0)).tolist(),
        "acceleration_rms_per_joint_rad_s2": np.sqrt(np.mean(qdd_error_array**2, axis=0)).tolist(),
        "sample_period_s": dt,
    }


def main() -> None:
    args = parse_args()
    with np.load(args.trace.expanduser().resolve()) as trace:
        tracer_time = np.asarray(trace["time"], dtype=float)
        reference_q = np.asarray(trace["reference_q"], dtype=float)
        reference_qd = np.asarray(trace["reference_qd"], dtype=float)
        reference_qdd = np.asarray(trace["reference_qdd"], dtype=float)
        actual_q = np.asarray(trace["actual_q"], dtype=float)
        actual_qd = np.asarray(trace["actual_qd"], dtype=float)
    dt = float(np.median(np.diff(tracer_time)))
    relative_time = tracer_time - tracer_time[0]
    wall_time, targets = load_events(args.recording.expanduser().resolve())
    event_time = (
        wall_time
        - wall_time[0]
        + float(args.recording_to_tracer_offset_s)
        + float(args.command_delay_ms) * 1.0e-3
    )
    params, limits = load_movax_rmp_config(
        args.config.expanduser().resolve(), args.mech_unit
    )
    first_event = max(0, int(np.searchsorted(event_time, 0.0, side="right") - 1))
    planner = JointRMPPlanner(
        limits,
        params,
        dt,
        use_safety_bounds=True,
        use_jerk_limit=False,
    )
    planner.reset(
        reference_q[0],
        velocity=reference_qd[0],
        acceleration=reference_qdd[0],
        target_position=targets[first_event],
    )
    predicted_q = np.empty_like(reference_q)
    predicted_qd = np.empty_like(reference_qd)
    predicted_qdd = np.empty_like(reference_qdd)
    previous_event = -1
    for sample, now in enumerate(relative_time):
        event = max(0, int(np.searchsorted(event_time, now, side="right") - 1))
        command = planner.step(
            targets[event],
            target_updated=event != previous_event,
        )
        previous_event = event
        predicted_q[sample] = command.position
        predicted_qd[sample] = command.velocity
        predicted_qdd[sample] = command.acceleration
    mask = (relative_time >= args.evaluation_start_s) & (
        relative_time < args.evaluation_end_s
    )
    errors = (
        predicted_q[mask] - reference_q[mask],
        predicted_qd[mask] - reference_qd[mask],
        predicted_qdd[mask] - reference_qdd[mask],
    )
    overall = [rms(error) for error in errors]
    per_joint = [np.sqrt(np.mean(np.square(error), axis=0)) for error in errors]
    report = {
        "samples": int(np.sum(mask)),
        "dt_s": dt,
        "position_rms_rad": overall[0],
        "position_rms_deg": float(np.rad2deg(overall[0])),
        "velocity_rms_rad_s": overall[1],
        "acceleration_rms_rad_s2": overall[2],
        "position_rms_per_joint_rad": per_joint[0].tolist(),
        "velocity_rms_per_joint_rad_s": per_joint[1].tolist(),
        "acceleration_rms_per_joint_rad_s2": per_joint[2].tolist(),
    }
    if args.include_feedback_reference_baseline:
        report["feedback_reference_baseline"] = _feedback_reference_baseline(
            relative_time,
            reference_q,
            reference_qd,
            reference_qdd,
            actual_q,
            actual_qd,
            mask,
            delay_min_ms=float(args.feedback_delay_min_ms),
            delay_max_ms=float(args.feedback_delay_max_ms),
            delay_step_ms=float(args.feedback_delay_step_ms),
        )
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
