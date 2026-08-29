#!/usr/bin/env python3
"""Identify an ideal MuJoCo PD/feedforward law from a Movax 1 kHz trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovered_rmp.mujoco_adapter import single_dof_joint_indices
from recovered_rmp.replay import REST_POSTURE_DEG, RIGHT_ARM_JOINTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=1.410)
    parser.add_argument("--end-s", type=float, default=5.000)
    parser.add_argument("--ridge", type=float, default=1.0e-6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import mujoco

    with np.load(args.trace.expanduser().resolve()) as archive:
        time = np.asarray(archive["time"], dtype=float)
        reference_q = np.asarray(archive["reference_q"], dtype=float)
        actual_q = np.asarray(archive["actual_q"], dtype=float)
        reference_qd = np.asarray(archive["reference_qd"], dtype=float)
        actual_qd = np.asarray(archive["actual_qd"], dtype=float)
        reference_qdd = np.asarray(archive["reference_qdd"], dtype=float)
    relative_time = time - time[0]
    dt = np.diff(time)
    actual_qdd = np.zeros_like(actual_qd)
    actual_qdd[:-1] = np.diff(actual_qd, axis=0) / dt[:, None]
    actual_qdd[-1] = actual_qdd[-2]
    selection = (relative_time >= args.start_s) & (relative_time < args.end_s)
    indices = np.flatnonzero(selection)

    model = mujoco.MjModel.from_xml_path(str(args.model.expanduser().resolve()))
    data = mujoco.MjData(model)
    qpos_indices, dof_indices = single_dof_joint_indices(model, RIGHT_ARM_JOINTS)
    posture = model.qpos0.copy()
    for name, degrees in REST_POSTURE_DEG.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            posture[int(model.jnt_qposadr[joint_id])] = np.deg2rad(degrees)
    full_mass = np.empty((model.nv, model.nv), dtype=float)
    required_dynamic: list[np.ndarray] = []
    reference_dynamic: list[np.ndarray] = []
    for index in indices:
        data.qpos[:] = posture
        data.qpos[qpos_indices] = actual_q[index]
        data.qvel[:] = 0.0
        data.qvel[dof_indices] = actual_qd[index]
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        bias = np.asarray(data.qfrc_inverse[dof_indices], dtype=float).copy()
        mujoco.mj_fullM(model, data, full_mass)
        desired_full_acc = np.zeros(model.nv, dtype=float)
        desired_full_acc[dof_indices] = reference_qdd[index]
        reference_dynamic.append((full_mass @ desired_full_acc)[dof_indices])
        data.qacc[dof_indices] = actual_qdd[index]
        mujoco.mj_inverse(model, data)
        required_dynamic.append(
            np.asarray(data.qfrc_inverse[dof_indices], dtype=float).copy() - bias
        )
    required = np.stack(required_dynamic)
    dynamic_ff = np.stack(reference_dynamic)
    features = np.stack(
        (
            reference_q[indices] - actual_q[indices],
            reference_qd[indices] - actual_qd[indices],
            dynamic_ff,
            np.ones_like(dynamic_ff),
        ),
        axis=-1,
    )
    coefficients = np.empty((7, 4), dtype=float)
    for joint in range(7):
        x = features[:, joint]
        y = required[:, joint]
        coefficients[joint] = np.linalg.solve(
            x.T @ x + float(args.ridge) * np.eye(4), x.T @ y
        )
    prediction = np.einsum("nji,ji->nj", features, coefficients)
    error = prediction - required
    report = {
        "samples": int(indices.size),
        "coefficient_columns": [
            "kp_nm_per_rad",
            "kd_nm_per_rad_s",
            "mass_qdd_feedforward_scale",
            "bias_nm",
        ],
        "coefficients_per_joint": coefficients.tolist(),
        "dynamic_torque_rms_nm": float(np.sqrt(np.mean(required**2))),
        "fit_torque_rms_nm": float(np.sqrt(np.mean(error**2))),
        "fit_torque_rms_per_joint_nm": np.sqrt(
            np.mean(error**2, axis=0)
        ).tolist(),
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
