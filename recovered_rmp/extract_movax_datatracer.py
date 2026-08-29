#!/usr/bin/env python3
"""Stream Movax DataTracer JSON and export 1 kHz right-arm references/feedback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np


FIELDS = {
    "reference_q": "axispos",
    "actual_q": "axisposres",
    "reference_qd": "axisvel",
    "actual_qd": "axisvelres",
    "reference_qdd": "axisacc",
    "command_torque": "axistrq",
    "feedforward_torque": "axistrqffw",
    "feedback_torque": "axistrqfdb",
    "estimated_torque": "axisestmatetrq",
    "resolver_torque": "axistrqres",
}


def _records(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, object]]:
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    with path.open(encoding="utf-8") as stream:
        while True:
            if position >= len(buffer) - 1:
                buffer = buffer[position:] + stream.read(chunk_size)
                position = 0
                if not buffer:
                    return
            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] in "[,"
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = stream.read(chunk_size)
                if not more:
                    raise
                buffer = buffer[position:] + more
                position = 0
                continue
            if not isinstance(record, dict):
                raise ValueError("DataTracer top-level entries must be objects")
            yield record
            position = end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--right-arm-start", type=int, default=17)
    parser.add_argument("--max-lag-ms", type=int, default=50)
    return parser.parse_args()


def _best_lag(reference: np.ndarray, actual: np.ndarray, maximum: int) -> tuple[int, float]:
    candidates: list[tuple[float, int]] = []
    for lag in range(maximum + 1):
        reference_slice = reference if lag == 0 else reference[:-lag]
        actual_slice = actual if lag == 0 else actual[lag:]
        candidates.append(
            (float(np.sqrt(np.mean((reference_slice - actual_slice) ** 2))), lag)
        )
    error, lag = min(candidates)
    return lag, error


def main() -> None:
    args = parse_args()
    rows: dict[str, list[np.ndarray]] = {name: [] for name in FIELDS}
    times: list[float] = []
    start = int(args.right_arm_start)
    for record in _records(args.trace.expanduser().resolve()):
        axis_data = record.get("data")
        if not isinstance(axis_data, list) or len(axis_data) < start + 7:
            raise ValueError("record does not contain the configured seven right-arm axes")
        right_arm = axis_data[start : start + 7]
        if not all(isinstance(axis, dict) for axis in right_arm):
            raise ValueError("right-arm axis entries must be objects")
        times.append(float(record["time"]))
        for output_name, field_name in FIELDS.items():
            rows[output_name].append(
                np.asarray([float(axis[field_name]) for axis in right_arm])
            )
    if len(times) < 2:
        raise ValueError("trace must contain at least two samples")
    arrays = {name: np.stack(values) for name, values in rows.items()}
    time = np.asarray(times, dtype=float)
    dt = float(np.median(np.diff(time)))
    lag_q: list[int] = []
    lag_q_error: list[float] = []
    lag_qd: list[int] = []
    lag_qd_error: list[float] = []
    for joint in range(7):
        lag, error = _best_lag(
            arrays["reference_q"][:, joint],
            arrays["actual_q"][:, joint],
            int(args.max_lag_ms),
        )
        lag_q.append(lag)
        lag_q_error.append(error)
        lag, error = _best_lag(
            arrays["reference_qd"][:, joint],
            arrays["actual_qd"][:, joint],
            int(args.max_lag_ms),
        )
        lag_qd.append(lag)
        lag_qd_error.append(error)
    report = {
        "source": str(args.trace.expanduser().resolve()),
        "samples": len(time),
        "median_dt_s": dt,
        "right_arm_axis_indices": list(range(start, start + 7)),
        "reference_to_feedback": {
            "q_zero_lag_rms_rad": float(
                np.sqrt(np.mean((arrays["reference_q"] - arrays["actual_q"]) ** 2))
            ),
            "qd_zero_lag_rms_rad_s": float(
                np.sqrt(
                    np.mean((arrays["reference_qd"] - arrays["actual_qd"]) ** 2)
                )
            ),
            "q_best_lag_steps_per_joint": lag_q,
            "q_best_lag_rms_rad_per_joint": lag_q_error,
            "qd_best_lag_steps_per_joint": lag_qd,
            "qd_best_lag_rms_rad_s_per_joint": lag_qd_error,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), time=time, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
