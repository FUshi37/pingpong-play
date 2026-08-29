#!/usr/bin/env python3
"""Replay recorded 1 kHz RMP q/qd/qdd through ideal MuJoCo PD only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovered_rmp.mujoco_adapter import single_dof_joint_indices
from recovered_rmp.replay import REST_POSTURE_DEG, RIGHT_ARM_JOINTS


def _joint_vector(text: str) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split(",")], dtype=float)
    if values.shape not in {(1,), (7,)}:
        raise argparse.ArgumentTypeError("expected one or seven comma-separated values")
    return np.broadcast_to(values, (7,)).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kp-scale", type=_joint_vector, default=np.full(7, 0.02))
    parser.add_argument("--kd-scale", type=_joint_vector, default=np.full(7, 0.10))
    parser.add_argument("--damping-scale", type=_joint_vector, default=np.ones(7))
    parser.add_argument("--armature-scale", type=_joint_vector, default=np.ones(7))
    parser.add_argument(
        "--acceleration-feedforward-s2",
        type=_joint_vector,
        default=np.zeros(7),
    )
    parser.add_argument("--inverse-dynamics-feedforward", type=float, default=0.0)
    parser.add_argument("--qdd-feedforward", type=float, default=0.0)
    parser.add_argument("--reference-delay-steps", type=int, default=0)
    parser.add_argument("--start-s", type=float, default=1.410)
    parser.add_argument("--end-s", type=float, default=5.000)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def replay(
    trace_path: Path,
    model_path: Path,
    *,
    kp_scale: np.ndarray,
    kd_scale: np.ndarray,
    damping_scale: np.ndarray,
    armature_scale: np.ndarray,
    acceleration_feedforward_s2: np.ndarray,
    inverse_dynamics_feedforward: float = 0.0,
    qdd_feedforward: float = 0.0,
    reference_delay_steps: int = 0,
) -> dict[str, np.ndarray]:
    import mujoco

    with np.load(trace_path.expanduser().resolve()) as archive:
        arrays = {
            name: np.asarray(archive[name], dtype=float)
            for name in (
                "time",
                "reference_q",
                "actual_q",
                "reference_qd",
                "actual_qd",
                "reference_qdd",
            )
        }
    time = arrays["time"]
    dt = float(np.median(np.diff(time)))
    model = mujoco.MjModel.from_xml_path(str(model_path.expanduser().resolve()))
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    qpos_indices, dof_indices = single_dof_joint_indices(model, RIGHT_ARM_JOINTS)
    actuator_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in RIGHT_ARM_JOINTS
        ],
        dtype=np.int32,
    )
    base_kp = model.actuator_gainprm[actuator_ids, 0].copy()
    base_kd = -model.actuator_biasprm[actuator_ids, 2].copy()
    kp = base_kp * np.asarray(kp_scale)
    kd = base_kd * np.asarray(kd_scale)
    model.actuator_gainprm[actuator_ids, 0] = kp
    model.actuator_biasprm[actuator_ids, 1] = -kp
    model.actuator_biasprm[actuator_ids, 2] = -kd
    model.dof_damping[dof_indices] *= np.asarray(damping_scale)
    model.dof_armature[dof_indices] *= np.asarray(armature_scale)

    for name, degrees in REST_POSTURE_DEG.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id >= 0:
            data.qpos[int(model.jnt_qposadr[joint_id])] = np.deg2rad(degrees)
        if actuator_id >= 0:
            data.ctrl[actuator_id] = np.deg2rad(degrees)
    data.qpos[qpos_indices] = arrays["actual_q"][0]
    data.ctrl[actuator_ids] = arrays["actual_q"][0]
    for _ in range(700):
        mujoco.mj_step(model, data)
    data.qpos[qpos_indices] = arrays["actual_q"][0]
    data.qvel[dof_indices] = arrays["actual_qd"][0]
    mujoco.mj_forward(model, data)

    sim_q = np.empty_like(arrays["actual_q"])
    sim_qd = np.empty_like(arrays["actual_qd"])
    full_mass = np.empty((model.nv, model.nv), dtype=float)
    for index in range(time.size):
        reference_index = max(0, index - int(reference_delay_steps))
        sim_q[index] = data.qpos[qpos_indices]
        sim_qd[index] = data.qvel[dof_indices]
        compensated_position = (
            arrays["reference_q"][reference_index]
            + kd / np.maximum(kp, 1.0e-9)
            * arrays["reference_qd"][reference_index]
            + np.asarray(acceleration_feedforward_s2)
            * arrays["reference_qdd"][reference_index]
        )
        data.ctrl[actuator_ids] = np.clip(
            compensated_position,
            model.actuator_ctrlrange[actuator_ids, 0],
            model.actuator_ctrlrange[actuator_ids, 1],
        )
        mujoco.mj_forward(model, data)
        mujoco.mj_fullM(model, data, full_mass)
        desired_full_acc = np.zeros(model.nv, dtype=float)
        desired_full_acc[dof_indices] = arrays["reference_qdd"][reference_index]
        dynamic_torque = (full_mass @ desired_full_acc)[dof_indices]
        bias_torque = (
            np.asarray(data.qfrc_bias) - np.asarray(data.qfrc_passive)
        )[dof_indices]
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof_indices] = (
            float(inverse_dynamics_feedforward) * bias_torque
            + float(qdd_feedforward) * dynamic_torque
        )
        mujoco.mj_step(model, data)
    sim_qdd = np.zeros_like(sim_qd)
    sim_qdd[:-1] = np.diff(sim_qd, axis=0) / np.diff(time)[:, None]
    sim_qdd[-1] = sim_qdd[-2]
    actual_qdd = np.zeros_like(arrays["actual_qd"])
    actual_qdd[:-1] = np.diff(arrays["actual_qd"], axis=0) / np.diff(time)[:, None]
    actual_qdd[-1] = actual_qdd[-2]
    return {
        **arrays,
        "sim_q": sim_q,
        "sim_qd": sim_qd,
        "sim_qdd": sim_qdd,
        "actual_qdd": actual_qdd,
        "kp": kp,
        "kd": kd,
    }


def _metrics(result: dict[str, np.ndarray], start: float, end: float) -> dict[str, object]:
    relative_time = result["time"] - result["time"][0]
    mask = (relative_time >= start) & (relative_time < end)
    errors = (
        result["sim_q"][mask] - result["actual_q"][mask],
        result["sim_qd"][mask] - result["actual_qd"][mask],
        result["sim_qdd"][mask] - result["actual_qdd"][mask],
    )
    rms = [np.sqrt(np.mean(error**2, axis=0)) for error in errors]
    return {
        "samples": int(np.sum(mask)),
        "position_rms_rad": float(np.sqrt(np.mean(errors[0] ** 2))),
        "position_rms_deg": float(np.rad2deg(np.sqrt(np.mean(errors[0] ** 2)))),
        "velocity_rms_rad_s": float(np.sqrt(np.mean(errors[1] ** 2))),
        "acceleration_rms_rad_s2": float(np.sqrt(np.mean(errors[2] ** 2))),
        "position_rms_per_joint_rad": rms[0].tolist(),
        "velocity_rms_per_joint_rad_s": rms[1].tolist(),
        "acceleration_rms_per_joint_rad_s2": rms[2].tolist(),
    }


def main() -> None:
    args = parse_args()
    result = replay(
        args.trace,
        args.model,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        damping_scale=args.damping_scale,
        armature_scale=args.armature_scale,
        acceleration_feedforward_s2=args.acceleration_feedforward_s2,
        inverse_dynamics_feedforward=args.inverse_dynamics_feedforward,
        qdd_feedforward=args.qdd_feedforward,
        reference_delay_steps=args.reference_delay_steps,
    )
    report = {
        "actuator_model_enabled": False,
        "control_chain": "DataTracer RMP q/qd/qdd -> ideal MuJoCo PD",
        "kp": result["kp"].tolist(),
        "kd": result["kd"].tolist(),
        "metrics": _metrics(result, args.start_s, args.end_s),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(output / "trace.npz", **result)


if __name__ == "__main__":
    main()
