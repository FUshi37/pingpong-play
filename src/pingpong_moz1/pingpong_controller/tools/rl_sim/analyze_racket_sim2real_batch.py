#!/usr/bin/env python3
"""End-effector (racket site) sim-vs-real error metrics from RMP replay artifacts.

For every ``rmp_mjx_replay_report.json`` / ``rmp_mjx_replay_trace.npz`` pair in
a batch output tree, this computes the racket ``right_ee_site`` kinematics for
the real recording (MuJoCo FK on recorded joint angles and raw ROS joint
velocity, angular velocity via Jacobian, linear/angular acceleration via
Savitzky-Golay on the tip velocity) and for the MJX replay (same operators on
sim joint angles / sim joint velocity), then reports per-group RMS/MAE of the
per-axis differences for:

  position (m), Euler angle (deg), linear velocity (m/s),
  angular velocity (rad/s), linear acceleration (m/s^2),
  angular acceleration (rad/s^2).

Both sides use the replay-trace time grid (``sourceTimeS``); acceleration
filters do not bridge control-time gaps.  Joint acceleration for FK is not
needed because tip acceleration is filtered from tip velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.signal import savgol_filter

RECORD_ROOT = (
    "/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
)
DEFAULT_MODEL = (
    "/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/tools/rl_sim/moz1_rmp_equiv.xml"
)
SITE = "right_ee_site"

SIGNALS = (
    "position",
    "angle",
    "velocity",
    "angular_velocity",
    "acceleration",
    "angular_acceleration",
)
UNITS = {
    "position": "m",
    "angle": "deg",
    "velocity": "m/s",
    "angular_velocity": "rad/s",
    "acceleration": "m/s^2",
    "angular_acceleration": "rad/s^2",
}
SG_WINDOW = 15
SG_ORDER = 3


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "time": data["time"],
            "source_sequence": data["source_sequence"],
            "actual_q": data["actual_q"],
            "actual_qd": data["actual_qd"],
            "sim_q": data["sim_q"],
            "sim_qd": data["sim_qd"],
        }


def _rotmat_euler_xyz(mat: np.ndarray) -> np.ndarray:
    """XYZ fixed-axis Euler angles (rad), wrapped to [-pi, pi] per sample."""
    euler = np.empty((mat.shape[0], 3), dtype=float)
    for i in range(mat.shape[0]):
        m = mat[i]
        sy = np.clip(m[0, 2], -1.0, 1.0)
        pitch = np.arcsin(-sy)
        if np.abs(np.cos(pitch)) > 1.0e-9:
            roll = np.arctan2(m[1, 2], m[2, 2])
            yaw = np.arctan2(m[0, 1], m[0, 0])
        else:
            roll = np.arctan2(-m[2, 1], m[1, 1])
            yaw = 0.0
        euler[i] = (roll, pitch, yaw)
    return euler


def _angular_velocity(model: mujoco.MjModel, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    nv = model.nv
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE)
    dof_ids = [model.jnt_dofadr[j] for j in _RIGHT_ARM_JOINT_IDS(model)]
    jacr = np.zeros((3, nv), dtype=float)
    omega = np.empty((q.shape[0], 3), dtype=float)
    for i in range(q.shape[0]):
        data.qpos[dof_ids] = q[i]
        data.qvel[dof_ids] = qd[i]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_jacSite(model, data, jacr, None, site_id)
        omega[i] = jacr @ data.qvel
    return omega


def _RIGHT_ARM_JOINT_IDS(model: mujoco.MjModel) -> list[int]:
    ids = []
    for name in ("RightArm-0", "RightArm-1", "RightArm-2", "RightArm-3", "RightArm-4", "RightArm-5", "RightArm-6"):
        ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
    return ids


def _fk_positions(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE)
    dof_ids = [model.jnt_dofadr[j] for j in _RIGHT_ARM_JOINT_IDS(model)]
    positions = np.empty((q.shape[0], 3), dtype=float)
    rotations = np.empty((q.shape[0], 3, 3), dtype=float)
    for i in range(q.shape[0]):
        data.qpos[dof_ids] = q[i]
        mujoco.mj_fwdPosition(model, data)
        positions[i] = data.site_xpos[site_id]
        rotations[i] = data.site_xmat[site_id].reshape(3, 3).copy()
    return positions, rotations


def _filter_velocity(velocity: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Per-axis SG first derivative, not bridging control-time gaps."""
    dt = np.diff(time)
    gap = np.flatnonzero((dt < 1.0e-4) | (dt > 0.10) | ~np.isfinite(dt))
    acc = np.full_like(velocity, np.nan, dtype=float)
    starts = [0, *(gap + 1)]
    stops = [*gap, velocity.shape[0]]
    for start, stop in zip(starts, stops):
        length = stop - start
        if length < SG_WINDOW:
            continue
        for axis in range(velocity.shape[1]):
            acc[start:stop, axis] = savgol_filter(
                velocity[start:stop, axis], SG_WINDOW, SG_ORDER, deriv=1, delta=np.mean(dt[start : stop - 1])
            )
    return acc


def _valid_row(arrays: dict[str, np.ndarray]) -> np.ndarray:
    acc = arrays["acceleration"]
    n = acc.shape[0]
    valid = np.ones(n, dtype=bool)
    for key in ("position", "angle", "velocity", "angular_velocity"):
        valid &= np.all(np.isfinite(arrays[key]), axis=1)
    valid &= np.all(np.isfinite(acc), axis=1) & np.all(np.isfinite(arrays["angular_acceleration"]), axis=1)
    return valid


def _per_axis_error(sim: np.ndarray, real: np.ndarray) -> np.ndarray:
    return sim - real


def _signature():
    return {
        "site": SITE,
        "real": "MuJoCo FK on recorded joint angles; raw ros.jointVelocity",
        "sim": "MuJoCo FK on MJX replay joint angles; sim joint velocity",
        "acceleration": "SG window 15 order 3 on tip velocity; gaps not bridged",
    }


def analyze_one(model: mujoco.MjModel, trace_path: Path, report_path: Path) -> dict:
    trace = _load_trace(trace_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    real_pos, real_rot = _fk_positions(model, trace["actual_q"])
    sim_pos, sim_rot = _fk_positions(model, trace["sim_q"])
    real_vel = _filter_velocity(real_pos, trace["time"])
    sim_vel = _filter_velocity(sim_pos, trace["time"])
    real_omega = _angular_velocity(model, trace["actual_q"], trace["actual_qd"])
    sim_omega = _angular_velocity(model, trace["sim_q"], trace["sim_qd"])
    real_alpha = _filter_velocity(real_omega, trace["time"])
    sim_alpha = _filter_velocity(sim_omega, trace["time"])

    real_angle = np.degrees(_rotmat_euler_xyz(real_rot))
    sim_angle = np.degrees(_rotmat_euler_xyz(sim_rot))

    real = {
        "position": real_pos,
        "angle": real_angle,
        "velocity": real_vel,
        "angular_velocity": real_omega,
        "acceleration": _filter_velocity(real_vel, trace["time"]),
        "angular_acceleration": real_alpha,
    }
    sim = {
        "position": sim_pos,
        "angle": sim_angle,
        "velocity": sim_vel,
        "angular_velocity": sim_omega,
        "acceleration": _filter_velocity(sim_vel, trace["time"]),
        "angular_acceleration": sim_alpha,
    }
    valid = _valid_row(sim) & _valid_row(real)
    metrics: dict[str, dict[str, list[float]]] = {}
    stats: dict[str, dict[str, float]] = {}
    for name in SIGNALS:
        error = sim[name][valid] - real[name][valid]
        metrics[name] = {
            "rms": np.sqrt(np.mean(error**2, axis=0)).tolist(),
            "mae": np.mean(np.abs(error), axis=0).tolist(),
            "rms_norm": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "real_std": np.std(real[name][valid], axis=0).tolist(),
            "sim_std": np.std(sim[name][valid], axis=0).tolist(),
        }
        stats[name] = {
            "n_valid": int(np.count_nonzero(valid)),
            "n_total": int(trace["time"].shape[0]),
        }
    return {
        "group": report.get("group", ""),
        "recording_name": report.get("recording_name", ""),
        "metrics": metrics,
        "stats": stats,
        "signature": _signature(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path, help="batch output root containing per-record report dirs")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=Path("racket_sim2real_batch.json"))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    records = []
    for trace_path in sorted(args.batch_root.rglob("rmp_mjx_replay_trace.npz")):
        report_path = trace_path.with_name("rmp_mjx_replay_report.json")
        if not report_path.exists():
            continue
        try:
            records.append(analyze_one(model, trace_path, report_path))
        except Exception as exc:  # noqa: BLE001 - keep per-record failures visible
            records.append({"trace_path": str(trace_path), "error": str(exc)})

    groups: dict[str, dict[str, list]] = {}
    for rec in records:
        if "error" in rec:
            continue
        group = rec["group"]
        groups.setdefault(group, {})
        for name in SIGNALS:
            entry = groups[group].setdefault(name, [])
            entry.append(rec["metrics"][name])

    summary: dict[str, object] = {"groups": {}, "records": records}
    for group, entries in groups.items():
        summary["groups"][group] = {
            name: {
                "rms_mean": np.mean([m["rms"] for m in meters], axis=0).tolist(),
                "mae_mean": np.mean([m["mae"] for m in meters], axis=0).tolist(),
                "rms_norm_mean": float(np.mean([m["rms_norm"] for m in meters])),
                "rms_norm_max": float(max(m["rms_norm"] for m in meters)),
                "n_records": len(meters),
            }
            for name, meters in entries.items()
        }
        # ensure insertion order is stable
        summary["groups"][group]["unit"] = UNITS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "groups": list(summary["groups"].keys())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
