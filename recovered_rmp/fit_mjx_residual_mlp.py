#!/usr/bin/env python3
"""Test whether a causal nonlinear model can identify MJX acceleration residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recovered_rmp.replay import forward_velocity_acceleration


SIGNALS = (
    "sim_q",
    "sim_qd",
    "sim_qacc",
    "rmp_q",
    "rmp_qd",
    "rmp_qdd",
    "target_q",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validation-name", default="probe_recording_20260811_210018")
    parser.add_argument("--training-name", action="append", default=[])
    parser.add_argument("--state-source", choices=("sim", "actual"), default="sim")
    parser.add_argument("--target-mode", choices=("residual", "actual"), default="residual")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _examples(
    data: dict[str, np.ndarray],
    record: int,
    history: int,
    state_source: str,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = int(data["lengths"][record])
    source_sequence = np.asarray(data["source_sequence"][record, :length])
    actual_acceleration, actual_valid = forward_velocity_acceleration(
        np.asarray(data["actual_qd"][record, :length], dtype=float),
        np.asarray(data["joint_time"][record, :length], dtype=float),
        source_sequence,
    )
    sim_acceleration, sim_valid = forward_velocity_acceleration(
        np.asarray(data["sim_qd"][record, :length], dtype=float),
        np.asarray(data["time"][record, :length], dtype=float),
        source_sequence,
    )
    policy = np.asarray(data["policy_mask"][record, :length], dtype=bool)
    valid = policy & actual_valid & sim_valid
    indices = np.flatnonzero(valid & (np.arange(length) >= history - 1))
    features: list[np.ndarray] = []
    for index in indices:
        values: list[np.ndarray] = []
        for lag in range(history):
            sample = index - lag
            for name in SIGNALS:
                source_name = name
                if state_source == "actual" and name == "sim_q":
                    source_name = "actual_q"
                elif state_source == "actual" and name == "sim_qd":
                    source_name = "actual_qd"
                elif state_source == "actual" and name == "sim_qacc":
                    current = np.asarray(
                        data["actual_qd"][record, sample], dtype=float
                    )
                    previous = np.asarray(
                        data["actual_qd"][record, max(0, sample - 1)], dtype=float
                    )
                    dt = float(
                        data["joint_time"][record, sample]
                        - data["joint_time"][record, max(0, sample - 1)]
                    )
                    values.append(
                        np.zeros(7) if dt < 1.0e-4 else (current - previous) / dt
                    )
                    continue
                values.append(np.asarray(data[source_name][record, sample], dtype=float))
        target_delta = np.asarray(
            data["target_q"][record, index]
            - data["target_q"][record, index - 1],
            dtype=float,
        )
        state_q_name = "actual_q" if state_source == "actual" else "sim_q"
        state_qd_name = "actual_qd" if state_source == "actual" else "sim_qd"
        q_error = np.asarray(
            data["rmp_q"][record, index] - data[state_q_name][record, index],
            dtype=float,
        )
        qd = np.asarray(data[state_qd_name][record, index], dtype=float)
        values.extend((target_delta, q_error, qd * np.abs(qd)))
        features.append(np.concatenate(values))
    target = actual_acceleration[indices]
    if target_mode == "residual":
        target = target - sim_acceleration[indices]
    return np.stack(features), target, indices


def _init_mlp(key: jax.Array, sizes: tuple[int, ...]) -> tuple[tuple[jax.Array, jax.Array], ...]:
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for layer_key, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:]):
        weight = jax.random.normal(layer_key, (fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        layers.append((weight, jnp.zeros((fan_out,), dtype=weight.dtype)))
    return tuple(layers)


def _predict(
    parameters: tuple[tuple[jax.Array, jax.Array], ...], features: jax.Array
) -> jax.Array:
    value = features
    for weight, bias in parameters[:-1]:
        value = jax.nn.silu(value @ weight + bias)
    weight, bias = parameters[-1]
    return value @ weight + bias


def _tree_zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def main() -> None:
    args = parse_args()
    with np.load(args.dataset) as archive:
        data = {name: archive[name] for name in archive.files}
    names = [str(value) for value in data["recording_names"]]
    validation_record = names.index(args.validation_name)
    examples = {
        record: _examples(
            data,
            record,
            int(args.history),
            str(args.state_source),
            str(args.target_mode),
        )
        for record in range(len(names))
    }
    if args.training_name:
        missing = sorted(set(args.training_name) - set(names))
        if missing:
            raise ValueError(f"training recordings not found: {missing}")
        training_records = [names.index(name) for name in args.training_name]
    else:
        training_records = [
            record for record in range(len(names)) if record != validation_record
        ]
    training_features = np.concatenate(
        [examples[record][0] for record in training_records]
    )
    training_targets = np.concatenate(
        [examples[record][1] for record in training_records]
    )
    feature_mean = training_features.mean(axis=0)
    feature_scale = training_features.std(axis=0)
    feature_scale[feature_scale < 1.0e-7] = 1.0
    target_mean = training_targets.mean(axis=0)
    target_scale = training_targets.std(axis=0)
    target_scale[target_scale < 1.0e-7] = 1.0
    x = jnp.asarray((training_features - feature_mean) / feature_scale, dtype=jnp.float32)
    y = jnp.asarray((training_targets - target_mean) / target_scale, dtype=jnp.float32)
    parameters = _init_mlp(
        jax.random.PRNGKey(int(args.seed)),
        (x.shape[1], 256, 128, 7),
    )
    first_moment = _tree_zeros_like(parameters)
    second_moment = _tree_zeros_like(parameters)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)

    @jax.jit
    def train_step(parameters, first_moment, second_moment, features, targets, step):
        def loss_fn(candidate):
            prediction = _predict(candidate, features)
            mse = jnp.mean((prediction - targets) ** 2)
            penalty = sum(jnp.sum(weight**2) for weight, _ in candidate)
            return mse + weight_decay * penalty

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        first_moment = jax.tree.map(
            lambda old, gradient: 0.9 * old + 0.1 * gradient,
            first_moment,
            gradients,
        )
        second_moment = jax.tree.map(
            lambda old, gradient: 0.999 * old + 0.001 * gradient**2,
            second_moment,
            gradients,
        )
        correction1 = 1.0 - 0.9**step
        correction2 = 1.0 - 0.999**step
        parameters = jax.tree.map(
            lambda value, first, second: value
            - learning_rate
            * (first / correction1)
            / (jnp.sqrt(second / correction2) + 1.0e-8),
            parameters,
            first_moment,
            second_moment,
        )
        return parameters, first_moment, second_moment, loss

    rng = np.random.default_rng(int(args.seed))
    batch_size = min(int(args.batch_size), x.shape[0])
    final_loss = np.nan
    for step in range(1, int(args.steps) + 1):
        indices = rng.integers(0, x.shape[0], size=batch_size)
        parameters, first_moment, second_moment, loss = train_step(
            parameters,
            first_moment,
            second_moment,
            x[indices],
            y[indices],
            jnp.asarray(step, dtype=jnp.float32),
        )
        if step == 1 or step % 500 == 0:
            final_loss = float(loss)
            print(f"step={step} normalized_loss={final_loss:.6f}", flush=True)

    def evaluate(record: int) -> dict[str, object]:
        features, _, indices = examples[record]
        normalized = jnp.asarray(
            (features - feature_mean) / feature_scale, dtype=jnp.float32
        )
        predicted = (
            np.asarray(_predict(parameters, normalized)) * target_scale + target_mean
        )
        length = int(data["lengths"][record])
        actual_acceleration, _ = forward_velocity_acceleration(
            np.asarray(data["actual_qd"][record, :length], dtype=float),
            np.asarray(data["joint_time"][record, :length], dtype=float),
            np.asarray(data["source_sequence"][record, :length]),
        )
        sim_acceleration, _ = forward_velocity_acceleration(
            np.asarray(data["sim_qd"][record, :length], dtype=float),
            np.asarray(data["time"][record, :length], dtype=float),
            np.asarray(data["source_sequence"][record, :length]),
        )
        base_error = sim_acceleration[indices] - actual_acceleration[indices]
        corrected_acceleration = predicted
        if args.target_mode == "residual":
            corrected_acceleration = sim_acceleration[indices] + predicted
        corrected_error = corrected_acceleration - actual_acceleration[indices]
        return {
            "samples": int(features.shape[0]),
            "base_qdd_rms": float(np.sqrt(np.mean(base_error**2))),
            "corrected_qdd_rms": float(np.sqrt(np.mean(corrected_error**2))),
            "corrected_qdd_rms_per_joint": np.sqrt(
                np.mean(corrected_error**2, axis=0)
            ).tolist(),
        }

    report = {
        "history": int(args.history),
        "state_source": str(args.state_source),
        "target_mode": str(args.target_mode),
        "training_recordings": [names[record] for record in training_records],
        "validation_recording": args.validation_name,
        "training_samples": int(x.shape[0]),
        "feature_dimension": int(x.shape[1]),
        "steps": int(args.steps),
        "final_normalized_batch_loss": final_loss,
        "metrics": {names[record]: evaluate(record) for record in range(len(names))},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    flattened, structure = jax.tree.flatten(parameters)
    np.savez_compressed(
        output.with_suffix(".npz"),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        parameter_structure=np.asarray(str(structure)),
        **{f"parameter_{index}": np.asarray(value) for index, value in enumerate(flattened)},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
