#!/usr/bin/env python3
"""Plot real-vs-MJX/RMP replay curves for the same commanded trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from recovered_rmp.replay import forward_velocity_acceleration, load_control_recording  # noqa: E402

JOINT_NAMES = tuple(f"RightArm-{index}" for index in range(7))


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def _first_2d(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[1] == 1:
        return value[:, 0, :]
    if value.ndim == 2:
        return value
    raise ValueError(f"expected [T,1,7] or [T,7], got {value.shape}")


def _metrics(real: np.ndarray, sim: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is None:
        mask = np.ones(real.shape[0], dtype=bool)
    error = sim[mask] - real[mask]
    return {
        "rms": float(np.sqrt(np.mean(error**2))),
        "rms_per_joint": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "max_abs": float(np.max(np.abs(error))),
        "max_abs_per_joint": np.max(np.abs(error), axis=0).tolist(),
        "samples": int(np.count_nonzero(mask)),
    }


def plot_group(
    output_path: Path,
    time: np.ndarray,
    real: np.ndarray,
    sim: np.ndarray,
    ylabel: str,
    title: str,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        mask = np.ones(time.shape[0], dtype=bool)
    fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
    for joint, axis in enumerate(axes):
        axis.plot(time[mask], real[mask, joint], color="#4e79a7", linewidth=1.6, label="real")
        axis.plot(time[mask], sim[mask, joint], color="#e15759", linewidth=1.3, linestyle="--", label="MJX/RMP")
        axis.set_ylabel(JOINT_NAMES[joint])
        axis.grid(True, alpha=0.22)
        if joint == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.supylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.03, 0.02, 1.0, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error(
    output_path: Path,
    time: np.ndarray,
    real: np.ndarray,
    sim: np.ndarray,
    ylabel: str,
    title: str,
    mask: np.ndarray | None = None,
) -> None:
    if mask is None:
        mask = np.ones(time.shape[0], dtype=bool)
    error = sim - real
    fig, axes = plt.subplots(7, 1, figsize=(13, 15), sharex=True)
    for joint, axis in enumerate(axes):
        axis.plot(time[mask], error[mask, joint], color="#59a14f", linewidth=1.4)
        axis.axhline(0.0, color="#333333", linewidth=0.7, alpha=0.6)
        axis.set_ylabel(JOINT_NAMES[joint])
        axis.grid(True, alpha=0.22)
    axes[-1].set_xlabel("time (s)")
    fig.supylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout(rect=(0.03, 0.02, 1.0, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results/rmp_replay_curves"))
    parser.add_argument("--title", default="Same command: real robot vs MJX/RMP replay")
    parser.add_argument("--policy-only", action="store_true", default=True)
    parser.add_argument("--all-samples", action="store_false", dest="policy_only")
    args = parser.parse_args()

    recording = load_control_recording(args.recording)
    trace = _load_trace(args.trace)
    count = min(recording.time.shape[0], _first_2d(trace["sim_q"]).shape[0])
    time = recording.time[:count] - recording.time[0]
    source_sequence = recording.source_sequence[:count]
    policy_mask = recording.policy_mask[:count].copy()
    if not args.policy_only or not np.any(policy_mask):
        policy_mask = np.ones(count, dtype=bool)

    real_q = recording.actual_q[:count]
    real_qd = recording.actual_qd[:count]
    real_qdd, real_qdd_valid = forward_velocity_acceleration(real_qd, recording.joint_time[:count], source_sequence)
    sim_q = _first_2d(trace["sim_q"])[:count]
    sim_qd = _first_2d(trace["sim_qd"])[:count]
    sim_qdd, sim_qdd_valid = forward_velocity_acceleration(sim_qd, recording.time[:count], source_sequence)
    qdd_mask = real_qdd_valid & sim_qdd_valid & policy_mask

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_group(output / "q_real_vs_mjx.png", time, real_q, sim_q, "q (rad)", f"{args.title} — q", policy_mask)
    plot_group(output / "dq_real_vs_mjx.png", time, real_qd, sim_qd, "dq (rad/s)", f"{args.title} — dq", policy_mask)
    plot_group(output / "ddq_real_vs_mjx.png", time, real_qdd, sim_qdd, "ddq (rad/s²)", f"{args.title} — ddq", qdd_mask)
    plot_error(output / "q_error.png", time, real_q, sim_q, "sim - real q (rad)", "q error", policy_mask)
    plot_error(output / "dq_error.png", time, real_qd, sim_qd, "sim - real dq (rad/s)", "dq error", policy_mask)
    plot_error(output / "ddq_error.png", time, real_qdd, sim_qdd, "sim - real ddq (rad/s²)", "ddq error", qdd_mask)

    summary = {
        "recording": str(args.recording),
        "trace": str(args.trace),
        "samples": count,
        "q": _metrics(real_q, sim_q, policy_mask),
        "dq": _metrics(real_qd, sim_qd, policy_mask),
        "ddq": _metrics(real_qdd, sim_qdd, qdd_mask),
        "plots": {
            "q": str(output / "q_real_vs_mjx.png"),
            "dq": str(output / "dq_real_vs_mjx.png"),
            "ddq": str(output / "ddq_real_vs_mjx.png"),
            "qError": str(output / "q_error.png"),
            "dqError": str(output / "dq_error.png"),
            "ddqError": str(output / "ddq_error.png"),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
