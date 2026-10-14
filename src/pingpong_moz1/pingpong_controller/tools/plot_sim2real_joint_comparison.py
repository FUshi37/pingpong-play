#!/usr/bin/env python3
"""Plot source-sequence-aligned real/sim joint angle and velocity traces.

The interactive recorder writes the real controller stream to
``control_samples.jsonl`` and the MuJoCo mirror stream to
``sim_control_samples.jsonl``.  This utility joins the streams by the canonical
control ``sourceSeq`` and plots the tick-start state seen by each policy
(``policyObs[0:14]``), avoiding a phase mismatch with the simulator's tick-end
snapshot fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


JOINT_NAMES = [f"RightArm-{index}" for index in range(7)]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skipping malformed {path}:{line_number}: {exc}")
                continue
            if isinstance(row, dict):
                yield row


def _source_seq(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("sourceSeq", row.get("tick")))
    except (TypeError, ValueError):
        return None


def _sim_source_seq(row: dict[str, Any]) -> int | None:
    sync = row.get("sync")
    if not isinstance(sync, dict):
        return None
    try:
        return int(sync.get("sourceSeq"))
    except (TypeError, ValueError):
        return None


def _vector(row: dict[str, Any], key: str, size: int) -> np.ndarray | None:
    try:
        value = np.asarray(row.get(key), dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if value.size < size or not np.all(np.isfinite(value[:size])):
        return None
    return value[:size]


def _policy_inputs_match(real: dict[str, Any], sim: dict[str, Any]) -> bool:
    """Mirror the recorder's strict input-envelope equality check."""
    policy_input = real.get("policyInput")
    sync = sim.get("sync")
    if not isinstance(policy_input, dict) or not isinstance(sync, dict):
        return False
    runtime = real.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    try:
        real_ball = np.asarray(
            policy_input.get("ball"), dtype=np.float32).reshape(3)
        sim_ball = np.asarray(sync.get("inputBall"), dtype=np.float32).reshape(3)
        real_velocity = np.asarray(
            policy_input.get("velocity"), dtype=np.float32).reshape(3)
        sim_velocity = np.asarray(
            sync.get("inputVelocity"), dtype=np.float32).reshape(3)
        real_profile = str(runtime.get(
            "activePolicyProfile", runtime.get("policyProfile", "")))
        sim_profile = str(sync.get("policyProfile", ""))
        envelope_match = (
            np.array_equal(real_ball, sim_ball)
            and np.array_equal(real_velocity, sim_velocity)
            and bool(policy_input.get("valid", False))
            == bool(sync.get("inputValid", False))
            and int(policy_input.get("ballObservationSeq", 0) or 0)
            == int(sync.get("ballObservationSeq", 0) or 0)
            and abs(
                float(policy_input.get("age", 0.0))
                - float(sync.get("inputAgeS", 0.0))
            ) <= 1e-9
            and real_profile == sim_profile
            and int(runtime.get("policyGeneration", 0) or 0)
            == int(sync.get("policyGeneration", 0) or 0)
        )
        real_policy_ball = real.get("policyBallObservation")
        sim_policy_ball = sim.get("policyBallObservation")
        if real_policy_ball is None and sim_policy_ball is None:
            return bool(envelope_match)
        policy_ball_match = np.array_equal(
            np.asarray(real_policy_ball, dtype=np.float32).reshape(6),
            np.asarray(sim_policy_ball, dtype=np.float32).reshape(6),
        )
        return bool(envelope_match and policy_ball_match)
    except (TypeError, ValueError):
        return False


def _cohort_key(
    real: dict[str, Any], sim: dict[str, Any]
) -> tuple[str, int, int]:
    runtime = real.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    sync = sim.get("sync")
    sync = sync if isinstance(sync, dict) else {}
    profile = str(runtime.get(
        "activePolicyProfile", runtime.get(
            "policyProfile", sync.get("policyProfile", "unknown"))))
    try:
        generation = int(runtime.get(
            "policyGeneration", sync.get("policyGeneration", 0)) or 0)
    except (TypeError, ValueError):
        generation = 0
    try:
        episode = int(sync.get("episode", 0) or 0)
    except (TypeError, ValueError):
        episode = 0
    return profile, generation, episode


def _load_selected_cohort(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    sim_rows_by_seq: dict[int, dict[str, Any]] = {}
    duplicate_sim = 0
    for row in _read_jsonl(run_dir / "sim_control_samples.jsonl"):
        seq = _sim_source_seq(row)
        if seq is None or _vector(row, "policyObs", 14) is None:
            continue
        if seq in sim_rows_by_seq:
            duplicate_sim += 1
            continue
        sim_rows_by_seq[seq] = row

    matches: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    duplicate_real = 0
    seen_real: set[int] = set()
    for row in _read_jsonl(run_dir / "control_samples.jsonl"):
        seq = _source_seq(row)
        if seq is None or seq not in sim_rows_by_seq:
            continue
        if seq in seen_real:
            duplicate_real += 1
            continue
        seen_real.add(seq)
        if not bool(row.get("policyExecuted", str(row.get("mode", "")) == "policy")):
            continue
        if str(row.get("mode", "")) != "policy":
            continue
        if _vector(row, "policyObs", 14) is None:
            continue
        sim = sim_rows_by_seq[seq]
        if not _policy_inputs_match(row, sim):
            continue
        matches.append((seq, row, sim))

    cohorts: dict[
        tuple[str, int, int], list[tuple[int, dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for match in matches:
        cohorts[_cohort_key(match[1], match[2])].append(match)
    if not cohorts:
        raise ValueError("no exactly aligned policy samples with matching inputs")

    selected_key = max(cohorts, key=lambda key: len(cohorts[key]))
    selected = sorted(cohorts[selected_key], key=lambda item: item[0])
    real_obs = np.asarray(
        [_vector(real, "policyObs", 14) for _, real, _ in selected],
        dtype=np.float64,
    )
    sim_obs = np.asarray(
        [_vector(sim, "policyObs", 14) for _, _, sim in selected],
        dtype=np.float64,
    )
    times = np.asarray(
        [float(real.get("time", index)) for index, (_, real, _) in enumerate(selected)],
        dtype=np.float64,
    )
    elapsed = times - times[0]
    if np.any(~np.isfinite(elapsed)) or np.any(np.diff(elapsed) < 0.0):
        elapsed = np.arange(len(selected), dtype=np.float64)
        time_source = "aligned sample index"
    else:
        time_source = "elapsed control time (s)"

    profile, generation, episode = selected_key
    data = {
        "elapsed": elapsed,
        "real_q": real_obs[:, :7],
        "sim_q": sim_obs[:, :7],
        "real_dq": real_obs[:, 7:14],
        "sim_dq": sim_obs[:, 7:14],
    }
    metadata = {
        "run": run_dir.name,
        "sim_rows": len(sim_rows_by_seq),
        "input_matched_rows": len(matches),
        "selected_rows": len(selected),
        "cohort_count": len(cohorts),
        "profile": profile,
        "generation": generation,
        "episode": episode,
        "duration_s": float(elapsed[-1]) if elapsed.size else 0.0,
        "time_source": time_source,
        "duplicate_real": duplicate_real,
        "duplicate_sim": duplicate_sim,
        "recording_complete": (run_dir / "summary.json").is_file(),
    }
    return data, metadata


def _plot_run(data: dict[str, Any], metadata: dict[str, Any]) -> plt.Figure:
    elapsed = data["elapsed"]
    fig, axes = plt.subplots(7, 2, figsize=(16, 20), sharex=True)
    for index, name in enumerate(JOINT_NAMES):
        angle_axis = axes[index, 0]
        velocity_axis = axes[index, 1]
        angle_axis.plot(elapsed, data["real_q"][:, index], color="#1f77b4",
                        linewidth=1.15, label="real")
        angle_axis.plot(elapsed, data["sim_q"][:, index], color="#d62728",
                        linewidth=1.05, linestyle="--", label="sim")
        velocity_axis.plot(elapsed, data["real_dq"][:, index], color="#1f77b4",
                           linewidth=1.15, label="real")
        velocity_axis.plot(elapsed, data["sim_dq"][:, index], color="#d62728",
                           linewidth=1.05, linestyle="--", label="sim")
        angle_axis.set_ylabel(f"{name}\nangle (rad)")
        velocity_axis.set_ylabel(f"{name}\nvelocity (rad/s)")
        for axis in (angle_axis, velocity_axis):
            axis.grid(True, alpha=0.25)
            axis.margins(x=0.01)
    axes[0, 0].set_title("Joint angle: real vs sim")
    axes[0, 1].set_title("Joint angular velocity: real vs sim")
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[-1, 0].set_xlabel(metadata["time_source"])
    axes[-1, 1].set_xlabel(metadata["time_source"])
    completion = "complete" if metadata["recording_complete"] else "incomplete recording"
    fig.suptitle(
        f"{metadata['run']} — sourceSeq-aligned policy state ({completion})\n"
        f"profile={metadata['profile']}, generation={metadata['generation']}, "
        f"episode={metadata['episode']}, n={metadata['selected_rows']}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig


def _rms(real: np.ndarray, sim: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(real - sim), axis=0))


def _gap_onset_summary(recording_root: Path) -> dict[str, dict[str, float | int]]:
    signal_keys = {
        "joint acceleration > 5 rad/s^2": "jointAccelerationRms5radS2",
        "joint velocity > 0.2 rad/s": "jointVelocityRms200mradS",
        "policy action > 0.20": "policyActionRms020",
        "nominal command > 0.02 rad": "nominalCommandRms20mrad",
        "racket position > 0.01 m": "racketPositionRms10mm",
        "joint position > 0.02 rad": "jointPositionRms20mrad",
    }
    values: dict[str, list[float]] = {label: [] for label in signal_keys}
    for report_path in sorted(
        recording_root.glob("probe_recording_*/analysis/sim_real_comparison.json")
    ):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        diagnostics = report.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
        onset = diagnostics.get("divergenceOnset")
        onset = onset if isinstance(onset, dict) else {}
        for label, key in signal_keys.items():
            crossing = onset.get(key)
            crossing = crossing if isinstance(crossing, dict) else {}
            try:
                elapsed = float(crossing["elapsedS"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(elapsed):
                values[label].append(elapsed)
    return {
        label: {
            "count": len(series),
            "median_s": float(np.median(series)),
            "min_s": float(np.min(series)),
            "max_s": float(np.max(series)),
        }
        for label, series in values.items()
        if series
    }


def _write_gap_summary(
    recording_root: Path,
    output_dir: Path,
    run_data: list[tuple[dict[str, Any], dict[str, Any]]],
    summary_rows: list[dict[str, Any]],
) -> None:
    q_error = np.concatenate([
        data["real_q"] - data["sim_q"] for data, _ in run_data
    ])
    dq_error = np.concatenate([
        data["real_dq"] - data["sim_dq"] for data, _ in run_data
    ])
    real_dq = np.concatenate([data["real_dq"] for data, _ in run_data])
    sim_dq = np.concatenate([data["sim_dq"] for data, _ in run_data])
    q_per_joint = np.sqrt(np.mean(np.square(q_error), axis=0))
    dq_per_joint = np.sqrt(np.mean(np.square(dq_error), axis=0))
    q_overall = float(np.sqrt(np.mean(np.square(q_error))))
    dq_overall = float(np.sqrt(np.mean(np.square(dq_error))))
    real_dq_rms = float(np.sqrt(np.mean(np.square(real_dq))))
    sim_dq_rms = float(np.sqrt(np.mean(np.square(sim_dq))))
    run_q = np.asarray(
        [float(row["angle_overall_rms_rad"]) for row in summary_rows])
    run_dq = np.asarray(
        [float(row["velocity_overall_rms_rad_s"]) for row in summary_rows])

    early_windows: dict[str, dict[str, float | int]] = {}
    for cutoff_s in (0.10, 0.15, 0.20, 0.25):
        q_chunks: list[np.ndarray] = []
        dq_chunks: list[np.ndarray] = []
        for data, _ in run_data:
            mask = data["elapsed"] <= cutoff_s + 1e-9
            q_chunks.append((data["real_q"] - data["sim_q"])[mask])
            dq_chunks.append((data["real_dq"] - data["sim_dq"])[mask])
        q_window = np.concatenate(q_chunks)
        dq_window = np.concatenate(dq_chunks)
        early_windows[f"{cutoff_s:.2f}s"] = {
            "ticks": int(q_window.shape[0]),
            "joint_position_rms_rad": float(
                np.sqrt(np.mean(np.square(q_window)))),
            "joint_velocity_rms_rad_s": float(
                np.sqrt(np.mean(np.square(dq_window)))),
        }

    onset_summary = _gap_onset_summary(recording_root)
    aggregate = {
        "recording_count": len(run_data),
        "complete_recording_count": sum(
            int(bool(metadata["recording_complete"])) for _, metadata in run_data
        ),
        "aligned_tick_count": int(q_error.shape[0]),
        "joint_position": {
            "overall_rms_rad": q_overall,
            "overall_rms_deg": float(np.rad2deg(q_overall)),
            "per_joint_rms_rad": dict(zip(JOINT_NAMES, q_per_joint.tolist())),
            "per_run_rms_rad": {
                "mean": float(np.mean(run_q)),
                "std": float(np.std(run_q)),
                "min": float(np.min(run_q)),
                "max": float(np.max(run_q)),
            },
        },
        "joint_velocity": {
            "overall_rms_rad_s": dq_overall,
            "real_signal_rms_rad_s": real_dq_rms,
            "sim_signal_rms_rad_s": sim_dq_rms,
            "normalized_rms_vs_real": dq_overall / max(real_dq_rms, 1e-12),
            "abs_error_p95_rad_s": float(np.percentile(np.abs(dq_error), 95)),
            "abs_error_p99_rad_s": float(np.percentile(np.abs(dq_error), 99)),
            "abs_error_max_rad_s": float(np.max(np.abs(dq_error))),
            "per_joint_rms_rad_s": dict(zip(JOINT_NAMES, dq_per_joint.tolist())),
            "per_run_rms_rad_s": {
                "mean": float(np.mean(run_dq)),
                "std": float(np.std(run_dq)),
                "min": float(np.min(run_dq)),
                "max": float(np.max(run_dq)),
            },
        },
        "early_windows": early_windows,
        "divergence_onset": onset_summary,
    }
    (output_dir / "gap_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    run_labels = [str(row["run"]).replace("probe_recording_", "")
                  for row in summary_rows]
    x = np.arange(len(run_labels))
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    axes[0, 0].bar(x, np.rad2deg(run_q), color="#4e79a7")
    axes[0, 0].set_ylabel("RMS angle gap (deg)")
    axes[0, 0].set_title("Per-recording joint-angle gap")
    axes[0, 1].bar(x, run_dq, color="#e15759")
    axes[0, 1].set_ylabel("RMS angular-velocity gap (rad/s)")
    axes[0, 1].set_title("Per-recording joint-velocity gap")
    for axis in axes[0]:
        axis.set_xticks(x)
        axis.set_xticklabels(run_labels, rotation=65, ha="right", fontsize=8)
        axis.grid(True, axis="y", alpha=0.25)

    joint_x = np.arange(len(JOINT_NAMES))
    axes[1, 0].bar(
        joint_x - 0.18, np.rad2deg(q_per_joint), width=0.36,
        label="angle gap (deg)", color="#4e79a7")
    velocity_axis = axes[1, 0].twinx()
    velocity_axis.bar(
        joint_x + 0.18, dq_per_joint, width=0.36,
        label="velocity gap (rad/s)", color="#e15759")
    axes[1, 0].set_xticks(joint_x)
    axes[1, 0].set_xticklabels(JOINT_NAMES, rotation=30, ha="right")
    axes[1, 0].set_ylabel("Angle RMS (deg)", color="#4e79a7")
    velocity_axis.set_ylabel("Velocity RMS (rad/s)", color="#e15759")
    axes[1, 0].set_title("Aggregate gap by joint")
    axes[1, 0].grid(True, axis="y", alpha=0.25)

    onset_labels = list(onset_summary)
    onset_medians = [onset_summary[label]["median_s"] for label in onset_labels]
    onset_low = [
        onset_summary[label]["median_s"] - onset_summary[label]["min_s"]
        for label in onset_labels
    ]
    onset_high = [
        onset_summary[label]["max_s"] - onset_summary[label]["median_s"]
        for label in onset_labels
    ]
    onset_y = np.arange(len(onset_labels))
    axes[1, 1].errorbar(
        onset_medians, onset_y, xerr=[onset_low, onset_high], fmt="o",
        color="#59a14f", ecolor="#59a14f", capsize=4)
    axes[1, 1].set_yticks(onset_y)
    axes[1, 1].set_yticklabels(onset_labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("First threshold crossing after cohort start (s)")
    axes[1, 1].set_title("Divergence order (median and range)")
    axes[1, 1].grid(True, axis="x", alpha=0.25)
    fig.suptitle(
        f"Sim2Real joint-gap summary — {len(run_data)} recordings, "
        f"{q_error.shape[0]} exact sourceSeq-aligned ticks")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_dir / "sim2real_joint_gap_summary.png", dpi=160)
    plt.close(fig)


def plot_recordings(recording_root: Path, output_dir: Path) -> int:
    run_dirs = sorted(
        path.parent
        for path in recording_root.glob("probe_recording_*/sim_control_samples.jsonl")
        if (path.parent / "control_samples.jsonl").is_file()
    )
    if not run_dirs:
        raise FileNotFoundError(f"no sim2real recordings found under {recording_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    run_data: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pdf_path = output_dir / "sim2real_joint_angle_velocity_comparisons.pdf"
    with PdfPages(pdf_path) as pdf:
        for run_dir in run_dirs:
            try:
                data, metadata = _load_selected_cohort(run_dir)
            except ValueError as exc:
                print(f"skip {run_dir.name}: {exc}")
                continue
            fig = _plot_run(data, metadata)
            png_path = output_dir / f"{run_dir.name}_joint_angle_velocity.png"
            fig.savefig(png_path, dpi=150)
            pdf.savefig(fig)
            plt.close(fig)

            q_rms = _rms(data["real_q"], data["sim_q"])
            dq_rms = _rms(data["real_dq"], data["sim_dq"])
            summary: dict[str, Any] = dict(metadata)
            summary["angle_overall_rms_rad"] = float(math.sqrt(np.mean(q_rms ** 2)))
            summary["velocity_overall_rms_rad_s"] = float(
                math.sqrt(np.mean(dq_rms ** 2)))
            for index, name in enumerate(JOINT_NAMES):
                summary[f"{name}_angle_rms_rad"] = float(q_rms[index])
                summary[f"{name}_velocity_rms_rad_s"] = float(dq_rms[index])
            summary["plot"] = png_path.name
            summary_rows.append(summary)
            run_data.append((data, metadata))
            print(
                f"wrote {png_path.name}: n={metadata['selected_rows']}, "
                f"q_rms={summary['angle_overall_rms_rad']:.6f} rad, "
                f"dq_rms={summary['velocity_overall_rms_rad_s']:.6f} rad/s"
            )

    if not summary_rows:
        pdf_path.unlink(missing_ok=True)
        raise RuntimeError("recordings were found, but none had plottable aligned data")
    summary_path = output_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_gap_summary(recording_root, output_dir, run_data, summary_rows)
    print(f"wrote {pdf_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {output_dir / 'gap_summary.json'}")
    print(f"wrote {output_dir / 'sim2real_joint_gap_summary.png'}")
    return len(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <recording_root>/sim2real_joint_comparisons",
    )
    args = parser.parse_args()
    recording_root = args.recording_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else recording_root / "sim2real_joint_comparisons"
    )
    count = plot_recordings(recording_root, output_dir)
    print(f"plotted {count} recording(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
