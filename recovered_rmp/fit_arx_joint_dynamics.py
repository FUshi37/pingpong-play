#!/usr/bin/env python3
"""Fit and validate a causal cross-joint ARX model from control recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recovered_rmp.replay import (  # noqa: E402
    RecordingTrace,
    forward_velocity_acceleration,
    load_control_recording,
)


def _policy_slice(trace: RecordingTrace, warmup: int = 80, cooldown: int = 20) -> slice:
    indices = np.flatnonzero(trace.policy_mask)
    if indices.size == 0:
        return slice(0, trace.time.size)
    return slice(
        max(0, int(indices[0]) - warmup),
        min(trace.time.size, int(indices[-1]) + cooldown + 1),
    )


def _features(
    q: np.ndarray,
    qd: np.ndarray,
    target: np.ndarray,
    index: int,
    history: int,
) -> np.ndarray:
    command_history = np.stack(
        [target[max(0, index - lag)] for lag in range(history)], axis=0
    )
    command_delta = command_history[:-1] - command_history[1:]
    error = command_history[0] - q
    return np.concatenate(
        (
            q,
            qd,
            error,
            command_history.reshape(-1),
            command_delta.reshape(-1),
            qd * np.abs(qd),
            np.tanh(20.0 * qd),
            np.ones(1),
        )
    )


def _examples(
    trace: RecordingTrace,
    history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selection = _policy_slice(trace)
    start = int(selection.start or 0)
    stop = int(selection.stop or trace.time.size)
    qdd, valid = forward_velocity_acceleration(
        trace.actual_qd, trace.joint_time, trace.source_sequence
    )
    rows: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    indices: list[int] = []
    for index in range(start, stop - 1):
        if not valid[index]:
            continue
        rows.append(
            _features(
                trace.actual_q[index],
                trace.actual_qd[index],
                trace.target_q,
                index,
                history,
            )
        )
        outputs.append(qdd[index])
        indices.append(index)
    return np.stack(rows), np.stack(outputs), np.asarray(indices, dtype=np.int64)


def _fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    regularization: float,
) -> dict[str, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1.0e-8] = 1.0
    normalized = (features - mean) / scale
    gram = normalized.T @ normalized
    rhs = normalized.T @ targets
    weights = np.linalg.solve(
        gram + float(regularization) * np.eye(gram.shape[0]), rhs
    )
    bias = np.mean(targets - normalized @ weights, axis=0)
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def _predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    return (
        (features - model["mean"]) / model["scale"] @ model["weights"]
        + model["bias"]
    )


def _metrics(error: np.ndarray) -> dict[str, object]:
    return {
        "rms": float(np.sqrt(np.mean(error**2))),
        "rms_per_joint": np.sqrt(np.mean(error**2, axis=0)).tolist(),
    }


def _teacher_forced_metrics(
    model: dict[str, np.ndarray], trace: RecordingTrace, history: int
) -> dict[str, object]:
    features, actual_qdd, indices = _examples(trace, history)
    predicted_qdd = _predict(model, features)
    qdd_error = predicted_qdd - actual_qdd
    dt = np.diff(trace.joint_time)[indices]
    predicted_qd_next = trace.actual_qd[indices] + dt[:, None] * predicted_qdd
    qd_error = predicted_qd_next - trace.actual_qd[indices + 1]
    predicted_q_next = (
        trace.actual_q[indices]
        + dt[:, None] * trace.actual_qd[indices]
        + 0.5 * dt[:, None] ** 2 * predicted_qdd
    )
    q_error = predicted_q_next - trace.actual_q[indices + 1]
    return {
        "samples": int(indices.size),
        "q_one_step": _metrics(q_error),
        "qd_one_step": _metrics(qd_error),
        "qdd": _metrics(qdd_error),
    }


def _knn_metrics(
    training_features: np.ndarray,
    training_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    neighbors: int,
) -> dict[str, object]:
    from scipy.spatial import cKDTree

    mean = np.mean(training_features, axis=0)
    scale = np.std(training_features, axis=0)
    scale[scale < 1.0e-8] = 1.0
    train = (training_features - mean) / scale
    validation = (validation_features - mean) / scale
    # Saturated command histories otherwise dominate Euclidean distance in
    # high dimension. Clipping preserves local regimes without letting a few
    # out-of-range features hide a close state match.
    train = np.clip(train, -5.0, 5.0)
    validation = np.clip(validation, -5.0, 5.0)
    tree = cKDTree(train)
    distance, indices = tree.query(validation, k=neighbors, workers=-1)
    if neighbors == 1:
        prediction = training_targets[indices]
        distance = distance[:, None]
    else:
        weights = 1.0 / np.maximum(distance, 1.0e-6)
        prediction = np.sum(
            training_targets[indices] * weights[..., None], axis=1
        ) / np.sum(weights, axis=1, keepdims=True)
    result = _metrics(prediction - validation_targets)
    result["mean_neighbor_distance"] = float(np.mean(distance[:, 0]))
    return result


def _rollout_metrics(
    model: dict[str, np.ndarray], trace: RecordingTrace, history: int
) -> dict[str, object]:
    selection = _policy_slice(trace)
    start = int(selection.start or 0)
    stop = int(selection.stop or trace.time.size)
    q = trace.actual_q[start].copy()
    qd = trace.actual_qd[start].copy()
    predicted_q = np.full_like(trace.actual_q, np.nan)
    predicted_qd = np.full_like(trace.actual_qd, np.nan)
    predicted_qdd = np.full_like(trace.actual_qdd, np.nan)
    predicted_q[start] = q
    predicted_qd[start] = qd
    valid = np.zeros(trace.time.size, dtype=bool)
    for index in range(start, stop - 1):
        dt = float(trace.joint_time[index + 1] - trace.joint_time[index])
        if (
            trace.source_sequence[index + 1] != trace.source_sequence[index] + 1
            or not 1.0e-4 <= dt <= 0.10
        ):
            predicted_q[index + 1] = q
            predicted_qd[index + 1] = qd
            continue
        feature = _features(q, qd, trace.target_q, index, history)
        qdd = _predict(model, feature[None, :])[0]
        q = q + dt * qd + 0.5 * dt * dt * qdd
        qd = qd + dt * qdd
        predicted_q[index + 1] = q
        predicted_qd[index + 1] = qd
        predicted_qdd[index] = qdd
        valid[index] = True
    mask = trace.policy_mask & np.all(np.isfinite(predicted_q), axis=1)
    qdd_actual, qdd_valid = forward_velocity_acceleration(
        trace.actual_qd, trace.joint_time, trace.source_sequence
    )
    qdd_mask = mask & valid & qdd_valid
    return {
        "q": _metrics(predicted_q[mask] - trace.actual_q[mask]),
        "qd": _metrics(predicted_qd[mask] - trace.actual_qd[mask]),
        "qdd": _metrics(predicted_qdd[qdd_mask] - qdd_actual[qdd_mask]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--validation-name", default="probe_recording_20260811_210018")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.record_root.glob("record_new[34]/*/control_samples.jsonl"))
    traces: dict[str, RecordingTrace] = {}
    for path in paths:
        try:
            trace = load_control_recording(path)
        except ValueError:
            continue
        if np.count_nonzero(trace.policy_mask) >= 250:
            traces[path.parent.name] = trace
    validation = traces.pop(args.validation_name)
    training_examples = tuple(
        _examples(trace, int(args.history))[:2] for trace in traces.values()
    )
    features = np.concatenate([item[0] for item in training_examples], axis=0)
    targets = np.concatenate([item[1] for item in training_examples], axis=0)
    model = _fit_ridge(features, targets, float(args.ridge))
    validation_features, validation_targets, _ = _examples(
        validation, int(args.history)
    )
    report = {
        "training_recordings": len(traces),
        "training_samples": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "validation_recording": args.validation_name,
        "teacher_forced": _teacher_forced_metrics(
            model, validation, int(args.history)
        ),
        "nonlinear_knn_qdd": {
            str(neighbors): _knn_metrics(
                features,
                targets,
                validation_features,
                validation_targets,
                neighbors,
            )
            for neighbors in (1, 4, 16)
        },
        "free_rollout": _rollout_metrics(model, validation, int(args.history)),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"),
        mean=model["mean"],
        scale=model["scale"],
        weights=model["weights"],
        bias=model["bias"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
