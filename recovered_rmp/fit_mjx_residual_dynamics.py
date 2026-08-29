#!/usr/bin/env python3
"""Fit a causal residual map from recovered-RMP/MJX traces to real joints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recovered_rmp.replay import forward_velocity_acceleration


SIGNALS = ("sim_q", "sim_qd", "sim_qacc", "rmp_q", "rmp_qd", "rmp_qdd", "target_q")


def _feature(data: dict[str, np.ndarray], record: int, index: int, history: int) -> np.ndarray:
    values: list[np.ndarray] = []
    for lag in range(history):
        sample = max(0, index - lag)
        values.extend(np.asarray(data[name][record, sample], dtype=float) for name in SIGNALS)
    current_qd = np.asarray(data["sim_qd"][record, index], dtype=float)
    current_error = np.asarray(
        data["rmp_q"][record, index] - data["sim_q"][record, index], dtype=float
    )
    values.extend((current_error, current_qd * np.abs(current_qd), np.ones(1)))
    return np.concatenate(values)


def _record_examples(
    data: dict[str, np.ndarray],
    record: int,
    history: int,
    include_context: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = int(data["lengths"][record])
    policy = np.asarray(data["policy_mask"][record, :length], dtype=bool)
    indices = np.flatnonzero(policy)
    if indices.size == 0:
        raise ValueError("recording has no policy samples")
    if include_context:
        start = max(history - 1, int(indices[0]) - 80)
        stop = min(length, int(indices[-1]) + 21)
        indices = np.arange(start, stop, dtype=np.int64)
    else:
        indices = indices[indices >= history - 1]
    features = np.stack(
        [_feature(data, record, int(index), history) for index in indices]
    )
    q_residual = data["actual_q"][record, indices] - data["sim_q"][record, indices]
    qd_residual = data["actual_qd"][record, indices] - data["sim_qd"][record, indices]
    return features, np.concatenate((q_residual, qd_residual), axis=1), indices


def _fit(features: np.ndarray, targets: np.ndarray, ridge: float) -> dict[str, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-7] = 1.0
    x = (features - mean) / scale
    target_mean = np.mean(targets, axis=0)
    centered = targets - target_mean
    weights = np.linalg.solve(
        x.T @ x + float(ridge) * np.eye(x.shape[1]), x.T @ centered
    )
    return {"mean": mean, "scale": scale, "target_mean": target_mean, "weights": weights}


def _predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    return (
        (features - model["mean"]) / model["scale"] @ model["weights"]
        + model["target_mean"]
    )


def _rms(error: np.ndarray) -> dict[str, object]:
    return {
        "rms": float(np.sqrt(np.mean(error**2))),
        "rms_per_joint": np.sqrt(np.mean(error**2, axis=0)).tolist(),
    }


def _evaluate(
    data: dict[str, np.ndarray],
    record: int,
    history: int,
    model: dict[str, np.ndarray],
) -> dict[str, object]:
    features, _, context_indices = _record_examples(
        data, record, history, include_context=True
    )
    residual = _predict(model, features)
    length = int(data["lengths"][record])
    corrected_q = np.asarray(data["sim_q"][record, :length], dtype=float).copy()
    corrected_qd = np.asarray(data["sim_qd"][record, :length], dtype=float).copy()
    corrected_q[context_indices] += residual[:, :7]
    corrected_qd[context_indices] += residual[:, 7:]
    policy = np.asarray(data["policy_mask"][record, :length], dtype=bool)
    predicted_qdd, predicted_valid = forward_velocity_acceleration(
        corrected_qd,
        np.asarray(data["time"][record, :length], dtype=float),
        np.asarray(data["source_sequence"][record, :length], dtype=np.int64),
    )
    actual_qdd, actual_valid = forward_velocity_acceleration(
        np.asarray(data["actual_qd"][record, :length], dtype=float),
        np.asarray(data["joint_time"][record, :length], dtype=float),
        np.asarray(data["source_sequence"][record, :length], dtype=np.int64),
    )
    acceleration_mask = policy & predicted_valid & actual_valid
    return {
        "q": _rms(corrected_q[policy] - data["actual_q"][record, :length][policy]),
        "qd": _rms(corrected_qd[policy] - data["actual_qd"][record, :length][policy]),
        "qdd": _rms(predicted_qdd[acceleration_mask] - actual_qdd[acceleration_mask]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validation-name", default="probe_recording_20260811_210018")
    parser.add_argument(
        "--training-name",
        action="append",
        default=[],
        help="Recording to use for fitting; repeat to select several (default: all except validation)",
    )
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.dataset) as archive:
        data = {name: archive[name] for name in archive.files}
    names = [str(value) for value in data["recording_names"]]
    validation = names.index(args.validation_name)
    if args.training_name:
        missing = sorted(set(args.training_name) - set(names))
        if missing:
            raise ValueError(f"training recordings not found: {missing}")
        if args.validation_name in args.training_name:
            raise ValueError("validation recording cannot also be a training recording")
        train_records = [names.index(name) for name in args.training_name]
    else:
        train_records = [index for index in range(len(names)) if index != validation]
    examples = [
        _record_examples(data, record, int(args.history), include_context=False)
        for record in train_records
    ]
    features = np.concatenate([value[0] for value in examples], axis=0)
    targets = np.concatenate([value[1] for value in examples], axis=0)
    model = _fit(features, targets, float(args.ridge))
    report = {
        "history": int(args.history),
        "ridge": float(args.ridge),
        "training_recordings": [names[index] for index in train_records],
        "training_samples": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "validation_recording": args.validation_name,
        "training_metrics": {
            names[index]: _evaluate(data, index, int(args.history), model)
            for index in train_records
        },
        "validation_metrics": _evaluate(data, validation, int(args.history), model),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), **model)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
