#!/usr/bin/env python3
"""Test whether a causal low-level servo model is identifiable from DataTracer.

The model is deliberately modest: each joint's measured velocity is an ARX
function of its own past measured velocity and of delayed final-RMP reference
velocity.  It is fitted on the first part of a stable trace and recursively
rolled out on the held-out tail.  This prevents teacher forcing from hiding an
unmodelled high-frequency feedback residual.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _integer_list(text: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in text.split(","))
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="NPZ made by extract_movax_datatracer.py")
    parser.add_argument("--start-s", type=float, default=1.410)
    parser.add_argument("--split-s", type=float, default=3.200)
    parser.add_argument("--end-s", type=float, default=5.000)
    parser.add_argument("--ar-orders", type=_integer_list, default=(1, 2, 4, 8, 12, 20))
    parser.add_argument("--input-lags", type=_integer_list, default=(0, 1, 2, 4, 8, 12))
    parser.add_argument("--input-orders", type=_integer_list, default=(0, 1, 2, 4, 8))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def main() -> None:
    args = parse_args()
    with np.load(args.trace.expanduser().resolve()) as trace:
        time = np.asarray(trace["time"], dtype=float)
        actual_q = np.asarray(trace["actual_q"], dtype=float)
        actual_qd = np.asarray(trace["actual_qd"], dtype=float)
        reference_qd = np.asarray(trace["reference_qd"], dtype=float)
    time -= time[0]
    dt = float(np.median(np.diff(time)))
    start = int(np.searchsorted(time, float(args.start_s)))
    split = int(np.searchsorted(time, float(args.split_s)))
    stop = int(np.searchsorted(time, float(args.end_s)))
    if not start < split < stop <= time.size:
        raise ValueError("start, split and end must select a nonempty ordered interval")
    results: list[dict[str, object]] = []
    for joint in range(actual_q.shape[1]):
        best: tuple[float, dict[str, object]] | None = None
        for ar_order in args.ar_orders:
            for input_lag in args.input_lags:
                for input_order in args.input_orders:
                    fit_start = start + max(ar_order, input_lag + input_order)
                    indices = np.arange(fit_start, split)
                    if indices.size < 20:
                        continue
                    features = np.column_stack(
                        [
                            *[
                                actual_qd[indices - lag, joint]
                                for lag in range(1, ar_order + 1)
                            ],
                            *[
                                reference_qd[indices - input_lag - lag, joint]
                                for lag in range(input_order + 1)
                            ],
                            np.ones(indices.size),
                        ]
                    )
                    coefficients = np.linalg.lstsq(
                        features, actual_qd[indices, joint], rcond=None
                    )[0]
                    history = list(actual_qd[split - ar_order : split, joint])
                    predicted_qd = np.empty(stop - split, dtype=float)
                    for output_index, sample in enumerate(range(split, stop)):
                        vector = np.asarray(
                            history[-ar_order:][::-1]
                            + [
                                reference_qd[sample - input_lag - lag, joint]
                                for lag in range(input_order + 1)
                            ]
                            + [1.0]
                        )
                        predicted_qd[output_index] = float(vector @ coefficients)
                        history.append(predicted_qd[output_index])
                    heldout_qd = actual_qd[split:stop, joint]
                    velocity_rms = _rms(predicted_qd - heldout_qd)
                    acceleration_rms = _rms(
                        np.diff(predicted_qd - heldout_qd) / dt
                    )
                    predicted_q = actual_q[split, joint] + np.concatenate(
                        (
                            [0.0],
                            np.cumsum(
                                0.5 * (predicted_qd[:-1] + predicted_qd[1:]) * dt
                            ),
                        )
                    )
                    position_rms = _rms(predicted_q - actual_q[split:stop, joint])
                    score = (
                        (position_rms / np.deg2rad(0.1)) ** 2
                        + (velocity_rms / 0.05) ** 2
                        + acceleration_rms**2
                    )
                    result = {
                        "joint": joint,
                        "ar_order": ar_order,
                        "input_lag_steps": input_lag,
                        "input_order": input_order,
                        "position_rms_rad": position_rms,
                        "position_rms_deg": float(np.rad2deg(position_rms)),
                        "velocity_rms_rad_s": velocity_rms,
                        "acceleration_rms_rad_s2": acceleration_rms,
                    }
                    if best is None or score < best[0]:
                        best = (score, result)
        if best is None:
            raise RuntimeError(f"no model was fitted for joint {joint}")
        results.append(best[1])
    report = {
        "definition": (
            "per-joint causal ARX fitted on the first stable interval and "
            "recursively rolled out without actual-feedback teacher forcing"
        ),
        "train_interval_s": [float(args.start_s), float(args.split_s)],
        "heldout_interval_s": [float(args.split_s), float(args.end_s)],
        "sample_period_s": dt,
        "per_joint": results,
        "overall": {
            "position_rms_rad": _rms(
                np.asarray([item["position_rms_rad"] for item in results])
            ),
            "velocity_rms_rad_s": _rms(
                np.asarray([item["velocity_rms_rad_s"] for item in results])
            ),
            "acceleration_rms_rad_s2": _rms(
                np.asarray([item["acceleration_rms_rad_s2"] for item in results])
            ),
        },
    }
    report["overall"]["position_rms_deg"] = float(
        np.rad2deg(report["overall"]["position_rms_rad"])
    )
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
