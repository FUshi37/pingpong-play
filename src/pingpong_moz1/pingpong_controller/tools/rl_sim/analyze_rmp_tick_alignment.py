#!/usr/bin/env python3
"""Generate tick-aligned command/feedback tables around divergence onset.

The recorder and sim2real comparison reports already align streams by
``sourceSeq``.  This helper focuses on the onset window and writes a compact CSV
plus a JSON summary so we can answer the causality question directly:
which side starts moving first — ``command.acceleration``,
``publishedSafePosition``, or ``ros.jointAcceleration``?
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RIGHT_ARM_JOINTS = tuple(f"RightArm-{index}" for index in range(7))
EPS = 1.0e-9


@dataclass(frozen=True)
class WindowRow:
    source_seq: int
    source_time_s: float
    published: bool
    command_position: np.ndarray
    command_safe_position: np.ndarray
    command_acceleration: np.ndarray
    ros_velocity: np.ndarray
    ros_acceleration: np.ndarray
    ros_dt_s: float | None


def _as_vector(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 7:
        return None
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        return None
    return result


def _load_run_rows(run_dir: Path) -> dict[int, WindowRow]:
    rows: dict[int, WindowRow] = {}
    path = run_dir / "control_samples.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            command = row.get("command")
            ros = row.get("ros")
            if not isinstance(command, dict) or not isinstance(ros, dict):
                continue
            if not bool(command.get("published")):
                continue
            seq = row.get("sourceSeq", row.get("tick"))
            time_s = row.get("sourceTimeS", row.get("time"))
            if not isinstance(seq, (int, float)) or not isinstance(time_s, (int, float)):
                continue
            position = _as_vector(command.get("position"))
            safe_position = _as_vector(command.get("publishedSafePosition", command.get("position")))
            acceleration = _as_vector(command.get("acceleration"))
            velocity = _as_vector(ros.get("jointVelocity"))
            ros_acceleration = _as_vector(ros.get("jointAcceleration"))
            if position is None or safe_position is None or acceleration is None:
                continue
            if velocity is None:
                velocity = np.zeros(7, dtype=float)
            if ros_acceleration is None:
                ros_acceleration = np.zeros(7, dtype=float)
            rows[int(seq)] = WindowRow(
                source_seq=int(seq),
                source_time_s=float(time_s),
                published=True,
                command_position=position,
                command_safe_position=safe_position,
                command_acceleration=acceleration,
                ros_velocity=velocity,
                ros_acceleration=ros_acceleration,
                ros_dt_s=float(ros.get("jointSampleDtS")) if isinstance(ros.get("jointSampleDtS"), (int, float)) else None,
            )
    if not rows:
        raise ValueError(f"no published samples found in {path}")
    return rows


def _first_nonzero_offset(values: list[np.ndarray], baseline: np.ndarray | None = None) -> int | None:
    if not values:
        return None
    if baseline is None:
        baseline = np.zeros_like(values[0])
    for index, value in enumerate(values):
        if np.max(np.abs(value - baseline)) > EPS:
            return index
    return None


def _divergence_seq(analysis_path: Path) -> int | None:
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    diagnostics = data.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return None
    onset = diagnostics.get("divergenceOnset")
    if not isinstance(onset, dict):
        return None
    acc = onset.get("jointAccelerationRms5radS2")
    if not isinstance(acc, dict):
        return None
    seq = acc.get("sourceSeq")
    return int(seq) if isinstance(seq, (int, float)) else None


def _format_vector(values: np.ndarray) -> str:
    return json.dumps([float(item) for item in values.tolist()], ensure_ascii=False)


def analyze_run(run_dir: Path, output_dir: Path, window: int) -> dict[str, Any] | None:
    analysis_path = run_dir / "analysis" / "sim_real_comparison.json"
    if not analysis_path.exists():
        return None
    onset_seq = _divergence_seq(analysis_path)
    if onset_seq is None:
        return None

    rows = _load_run_rows(run_dir)
    seqs = sorted(rows)
    if onset_seq not in rows:
        nearest = min(seqs, key=lambda seq: abs(seq - onset_seq))
        onset_seq = nearest

    onset_index = seqs.index(onset_seq)
    start = max(0, onset_index - window)
    stop = min(len(seqs), onset_index + window + 1)
    selected = seqs[start:stop]
    selected_rows = [rows[seq] for seq in selected]

    baseline_safe = selected_rows[0].command_safe_position
    baseline_command = selected_rows[0].command_acceleration
    baseline_ros = selected_rows[0].ros_acceleration
    first_command = _first_nonzero_offset([row.command_acceleration for row in selected_rows], baseline_command)
    first_safe = _first_nonzero_offset([row.command_safe_position for row in selected_rows], baseline_safe)
    first_ros = _first_nonzero_offset([row.ros_acceleration for row in selected_rows], baseline_ros)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_dir.name}_tick_alignment.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = [
            "offset",
            "sourceSeq",
            "sourceTimeS",
            "published",
            "command_safe_changed",
            "command_acc_changed",
            "ros_acc_changed",
            "command_position",
            "command_safe_position",
            "command_acceleration",
            "ros_velocity",
            "ros_acceleration",
            "ros_jointSampleDtS",
        ]
        writer.writerow(header)
        for offset, row in enumerate(selected_rows):
            writer.writerow([
                offset - (onset_index - start),
                row.source_seq,
                row.source_time_s,
                row.published,
                bool(np.max(np.abs(row.command_safe_position - baseline_safe)) > EPS),
                bool(np.max(np.abs(row.command_acceleration - baseline_command)) > EPS),
                bool(np.max(np.abs(row.ros_acceleration - baseline_ros)) > EPS),
                _format_vector(row.command_position),
                _format_vector(row.command_safe_position),
                _format_vector(row.command_acceleration),
                _format_vector(row.ros_velocity),
                _format_vector(row.ros_acceleration),
                row.ros_dt_s,
            ])

    summary = {
        "runDir": str(run_dir),
        "analysisPath": str(analysis_path),
        "onsetSeq": onset_seq,
        "window": window,
        "windowSeqs": selected,
        "firstNonzero": {
            "commandAccelerationOffset": first_command,
            "publishedSafePositionOffset": first_safe,
            "rosAccelerationOffset": first_ros,
        },
        "ordering": (
            "command.acceleration"
            if first_command is not None and (first_ros is None or first_command <= first_ros)
            else "ros.jointAcceleration"
        ),
        "csvPath": str(csv_path),
    }
    summary_path = output_dir / f"{run_dir.name}_tick_alignment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def _discover_runs(paths: list[Path]) -> list[Path]:
    runs: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if path.is_dir() and (path / "analysis" / "sim_real_comparison.json").exists():
            runs.append(path)
            continue
        if path.is_dir():
            for run_dir in sorted(path.glob("probe_recording_*")):
                if (run_dir / "analysis" / "sim_real_comparison.json").exists():
                    runs.append(run_dir)
            continue
        raise FileNotFoundError(path)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path(
                "/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/record_new3"
            ),
            Path(
                "/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/record_new4"
            ),
            Path(
                "/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/record_new5"
            ),
        ],
        help="recording roots or probe_recording directories",
    )
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/rmp_tick_alignment"),
    )
    args = parser.parse_args()

    runs = _discover_runs(list(args.paths))
    if not runs:
        raise SystemExit("no recording runs found")
    summaries = [
        summary for summary in (
            analyze_run(run_dir, args.output_dir, args.window) for run_dir in runs
        )
        if summary is not None
    ]
    if not summaries:
        raise SystemExit("no runs with divergence onset found")
    print(json.dumps({"runs": len(summaries), "outputDir": str(args.output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
