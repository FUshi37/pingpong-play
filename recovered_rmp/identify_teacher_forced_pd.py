#!/usr/bin/env python3
"""Identify low-level RMP PD/feedforward torque from real joint trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recovered_rmp.mujoco_adapter import single_dof_joint_indices
from recovered_rmp.replay import RIGHT_ARM_JOINTS, forward_velocity_acceleration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-name", default="probe_recording_20260811_210018")
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _inverse_torque(
    model,
    qpos_indices: np.ndarray,
    dof_indices: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[qpos_indices] = q
    data.qvel[:] = 0.0
    data.qvel[dof_indices] = qd
    data.qacc[:] = 0.0
    data.qacc[dof_indices] = qdd
    mujoco.mj_inverse(model, data)
    return np.asarray(data.qfrc_inverse[dof_indices], dtype=float).copy()


def _record_examples(
    model,
    data: dict[str, np.ndarray],
    record: int,
    qpos_indices: np.ndarray,
    dof_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = int(data["lengths"][record])
    sequence = np.asarray(data["source_sequence"][record, :length])
    actual_qdd, actual_valid = forward_velocity_acceleration(
        np.asarray(data["actual_qd"][record, :length], dtype=float),
        np.asarray(data["joint_time"][record, :length], dtype=float),
        sequence,
    )
    policy = np.asarray(data["policy_mask"][record, :length], dtype=bool)
    indices = np.flatnonzero(policy & actual_valid)
    torques = np.stack(
        [
            _inverse_torque(
                model,
                qpos_indices,
                dof_indices,
                np.asarray(data["actual_q"][record, index], dtype=float),
                np.asarray(data["actual_qd"][record, index], dtype=float),
                actual_qdd[index],
            )
            for index in indices
        ]
    )
    next_indices = np.minimum(indices + 1, length - 1)
    reference_q = 0.5 * (
        data["rmp_q"][record, indices] + data["rmp_q"][record, next_indices]
    )
    reference_qd = 0.5 * (
        data["rmp_qd"][record, indices] + data["rmp_qd"][record, next_indices]
    )
    reference_qdd = 0.5 * (
        data["rmp_qdd"][record, indices] + data["rmp_qdd"][record, next_indices]
    )
    features = np.stack(
        (
            reference_q - data["actual_q"][record, indices],
            reference_qd - data["actual_qd"][record, indices],
            reference_qdd,
            np.ones_like(reference_q),
        ),
        axis=-1,
    )
    return features, torques, actual_qdd[indices], indices


def main() -> None:
    args = parse_args()
    import mujoco

    with np.load(args.dataset) as archive:
        data = {name: archive[name] for name in archive.files}
    names = [str(value) for value in data["recording_names"]]
    validation = names.index(args.validation_name)
    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser().resolve()))
    qpos_indices, dof_indices = single_dof_joint_indices(model, RIGHT_ARM_JOINTS)
    examples = {
        record: _record_examples(
            model, data, record, qpos_indices, dof_indices
        )
        for record in range(len(names))
    }
    training_records = [record for record in range(len(names)) if record != validation]
    features = np.concatenate([examples[record][0] for record in training_records])
    torques = np.concatenate([examples[record][1] for record in training_records])
    coefficients = np.empty((7, 4), dtype=float)
    for joint in range(7):
        x = features[:, joint, :]
        y = torques[:, joint]
        coefficients[joint] = np.linalg.solve(
            x.T @ x + float(args.ridge) * np.eye(x.shape[1]), x.T @ y
        )

    def evaluate(record: int) -> dict[str, object]:
        x, required_torque, _, _ = examples[record]
        predicted_torque = np.einsum("nji,ji->nj", x, coefficients)
        error = predicted_torque - required_torque
        return {
            "samples": int(x.shape[0]),
            "torque_rms_nm": float(np.sqrt(np.mean(error**2))),
            "torque_rms_per_joint_nm": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "required_torque_rms_nm": float(np.sqrt(np.mean(required_torque**2))),
        }

    report = {
        "training_recordings": [names[record] for record in training_records],
        "validation_recording": args.validation_name,
        "coefficient_columns": ["kp_nm_per_rad", "kd_nm_per_rad_s", "qdd_ff_nm_per_rad_s2", "bias_nm"],
        "coefficients_per_joint": coefficients.tolist(),
        "metrics": {names[record]: evaluate(record) for record in range(len(names))},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
