#!/usr/bin/env python3
"""Plot recovered-RMP accuracy and final-RMP-to-feedback tracking separately."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovered_rmp.params import load_movax_rmp_config
from recovered_rmp.trajectory import JointRMPPlanner
from recovered_rmp.validate_rmp_datatracer import load_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="NPZ made by extract_movax_datatracer.py")
    parser.add_argument("recording", type=Path, help="matching recording directory or JSONL")
    parser.add_argument("config", type=Path, help="Movax model YAML")
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("recovered_rmp/calibration/datatracer_203806_rmp_and_feedback_validation.json"),
        help="JSON produced by validate_rmp_datatracer.py with feedback baseline",
    )
    parser.add_argument("--mech-unit", default="RightArm")
    parser.add_argument("--recording-to-tracer-offset-s", type=float, default=-3.0209988619056563)
    parser.add_argument("--command-delay-ms", type=float, default=3.0)
    parser.add_argument("--evaluation-start-s", type=float, default=1.410)
    parser.add_argument("--evaluation-end-s", type=float, default=5.000)
    parser.add_argument(
        "--zoom-s",
        type=float,
        default=1.0,
        help="duration plotted for time-series residual panels from evaluation start",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recovered_rmp/calibration/datatracer_203806_rmp_tracking_dashboard.png"),
    )
    parser.add_argument(
        "--signals-output",
        type=Path,
        help=(
            "optional q/qd/qdd tracking-curve image; defaults beside --output "
            "with _signals appended"
        ),
    )
    return parser.parse_args()


def _rms(error: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(error), axis=0))


def _reconstruct(
    relative_time: np.ndarray,
    reference_q: np.ndarray,
    reference_qd: np.ndarray,
    reference_qdd: np.ndarray,
    recording: Path,
    config: Path,
    *,
    recording_to_tracer_offset_s: float,
    command_delay_ms: float,
    mech_unit: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wall_time, targets = load_events(recording)
    event_time = (
        wall_time
        - wall_time[0]
        + recording_to_tracer_offset_s
        + command_delay_ms * 1.0e-3
    )
    params, limits = load_movax_rmp_config(config, mech_unit)
    dt = float(np.median(np.diff(relative_time)))
    first_event = max(0, int(np.searchsorted(event_time, 0.0, side="right") - 1))
    planner = JointRMPPlanner(
        limits, params, dt, use_safety_bounds=True, use_jerk_limit=False
    )
    planner.reset(
        reference_q[0],
        velocity=reference_qd[0],
        acceleration=reference_qdd[0],
        target_position=targets[first_event],
    )
    predicted = [np.empty_like(reference_q) for _ in range(3)]
    previous_event = -1
    for sample, now in enumerate(relative_time):
        event = max(0, int(np.searchsorted(event_time, now, side="right") - 1))
        command = planner.step(targets[event], target_updated=event != previous_event)
        previous_event = event
        predicted[0][sample] = command.position
        predicted[1][sample] = command.velocity
        predicted[2][sample] = command.acceleration
    return tuple(predicted)  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt

    with np.load(args.trace.expanduser().resolve()) as trace:
        time = np.asarray(trace["time"], dtype=float)
        reference_q = np.asarray(trace["reference_q"], dtype=float)
        reference_qd = np.asarray(trace["reference_qd"], dtype=float)
        reference_qdd = np.asarray(trace["reference_qdd"], dtype=float)
        actual_q = np.asarray(trace["actual_q"], dtype=float)
        actual_qd = np.asarray(trace["actual_qd"], dtype=float)
    relative_time = time - time[0]
    predicted_q, predicted_qd, predicted_qdd = _reconstruct(
        relative_time,
        reference_q,
        reference_qd,
        reference_qdd,
        args.recording.expanduser().resolve(),
        args.config.expanduser().resolve(),
        recording_to_tracer_offset_s=float(args.recording_to_tracer_offset_s),
        command_delay_ms=float(args.command_delay_ms),
        mech_unit=args.mech_unit,
    )
    report = json.loads(args.validation_report.expanduser().resolve().read_text(encoding="utf-8"))
    feedback = report.get("feedback_reference_baseline")
    if not isinstance(feedback, dict):
        raise ValueError("validation report lacks feedback_reference_baseline")
    delays_ms = np.asarray(feedback["feedback_delay_ms_per_joint"], dtype=float)
    if delays_ms.shape != (7,):
        raise ValueError("validation report must provide seven feedback delays")

    actual_qdd = np.full_like(actual_qd, np.nan)
    actual_qdd[:-1] = np.diff(actual_qd, axis=0) / np.diff(relative_time)[:, None]
    evaluation = (relative_time >= args.evaluation_start_s) & (relative_time < args.evaluation_end_s)
    evaluation[-1] = False
    shifted_reference = tuple(
        np.stack(
            [
                np.interp(relative_time - delay * 1.0e-3, relative_time, signal[:, joint])
                for joint, delay in enumerate(delays_ms)
            ],
            axis=-1,
        )
        for signal in (reference_q, reference_qd, reference_qdd)
    )
    recovery_errors = (
        predicted_q - reference_q,
        predicted_qd - reference_qd,
        predicted_qdd - reference_qdd,
    )
    feedback_errors = (
        shifted_reference[0] - actual_q,
        shifted_reference[1] - actual_qd,
        shifted_reference[2] - actual_qdd,
    )
    window = evaluation & (relative_time < args.evaluation_start_s + float(args.zoom_s))
    labels = ("position error [deg]", "velocity error [rad/s]", "acceleration error [rad/s²]")
    thresholds = (0.1, 0.05, 1.0)
    unit_scale = (180.0 / np.pi, 1.0, 1.0)
    colors = plt.get_cmap("tab10")(np.arange(7))
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), constrained_layout=True)
    for row, (recovery_error, feedback_error) in enumerate(zip(recovery_errors, feedback_errors)):
        scale = unit_scale[row]
        for joint in range(7):
            axes[row, 0].plot(
                relative_time[window], recovery_error[window, joint] * scale,
                color=colors[joint], lw=0.9, label=f"J{joint}",
            )
            axes[row, 1].plot(
                relative_time[window], feedback_error[window, joint] * scale,
                color=colors[joint], lw=0.9,
            )
        axes[row, 0].axhline(0.0, color="black", lw=0.6)
        axes[row, 1].axhline(0.0, color="black", lw=0.6)
        recovery_rms = _rms(recovery_error[evaluation]) * scale
        feedback_rms = _rms(feedback_error[evaluation]) * scale
        joints = np.arange(7)
        axes[row, 2].bar(joints - 0.19, recovery_rms, width=0.38, label="recovered RMP vs axis ref")
        axes[row, 2].bar(joints + 0.19, feedback_rms, width=0.38, label="delayed RMP ref vs encoder")
        axes[row, 2].axhline(thresholds[row], color="crimson", ls="--", lw=1.2, label="requested limit")
        axes[row, 2].set_xticks(joints, [f"J{joint}" for joint in joints])
        axes[row, 0].set_ylabel(labels[row])
        axes[row, 0].set_title("Recovered RMP − DataTracer axis reference" if row == 0 else "")
        axes[row, 1].set_title("Delayed final RMP reference − encoder feedback" if row == 0 else "")
        axes[row, 2].set_title("RMS over full stable interval" if row == 0 else "")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 1].grid(alpha=0.25)
        axes[row, 2].grid(axis="y", alpha=0.25)
    axes[0, 0].legend(ncol=4, fontsize=8, loc="upper right")
    axes[0, 2].legend(fontsize=7, loc="upper left")
    for column in range(2):
        axes[2, column].set_xlabel("DataTracer relative time [s]")
    fig.suptitle(
        "RightArm RMP recovery is accurate; reference-to-feedback residual is downstream servo dynamics",
        fontsize=14,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    # A direct signal overlay is intentionally separate from the residual
    # dashboard above: it makes the physical lead/lag easy to read without
    # asking a reader to reconstruct it from an error sign convention.
    signals_output = (
        args.signals_output.expanduser().resolve()
        if args.signals_output is not None
        else output.with_name(f"{output.stem}_signals.png")
    )
    fig, axes = plt.subplots(7, 3, figsize=(20, 23), sharex=True, constrained_layout=True)
    actual_signals = (actual_q, actual_qd, actual_qdd)
    raw_signals = (reference_q, reference_qd, reference_qdd)
    signal_titles = ("Position", "Velocity", "Acceleration")
    signal_units = ("deg", "rad/s", "rad/s²")
    for joint in range(7):
        for column, (raw, aligned, actual) in enumerate(
            zip(raw_signals, shifted_reference, actual_signals)
        ):
            scale = unit_scale[column]
            axis = axes[joint, column]
            # Acceleration feedback is wide and noisy. Draw it first and with
            # lower opacity so the command/reference curves remain visible.
            axis.plot(
                relative_time[evaluation], actual[evaluation, joint] * scale,
                color="#ff7f0e",
                lw=0.65 if column == 2 else 0.9,
                alpha=0.38 if column == 2 else 0.78,
                zorder=1,
                label="encoder feedback" if joint == 0 and column == 0 else None,
            )
            axis.plot(
                relative_time[evaluation], raw[evaluation, joint] * scale,
                color="0.35", ls="--", lw=0.85, zorder=2,
                label="final RMP reference (raw)" if joint == 0 and column == 0 else None,
            )
            axis.plot(
                relative_time[evaluation], aligned[evaluation, joint] * scale,
                color="#1f77b4", lw=1.15, zorder=3,
                label="RMP reference, delayed to feedback" if joint == 0 and column == 0 else None,
            )
            axis.grid(alpha=0.22)
            if joint == 0:
                axis.set_title(f"{signal_titles[column]} [{signal_units[column]}]")
            if column == 0:
                axis.set_ylabel(f"J{joint}")
            if joint == 6:
                axis.set_xlabel("DataTracer relative time [s]")
    axes[0, 0].legend(ncol=3, fontsize=8, loc="upper left")
    fig.suptitle(
        "RightArm final RMP reference and physical encoder feedback (stable 1 kHz interval)",
        fontsize=14,
    )
    fig.savefig(signals_output, dpi=220)
    plt.close(fig)
    print(output)
    print(signals_output)


if __name__ == "__main__":
    main()
