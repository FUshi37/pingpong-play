#!/usr/bin/env python3
"""Generate 2D diagnostics for one real-robot pingpong run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RIGHT_ARM_JOINTS = [f"RightArm-{i}" for i in range(7)]


def as_float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_bool(row: dict, key: str, default: bool = False) -> bool:
    if key not in row or row.get(key, "") == "":
        return default
    try:
        return bool(int(float(row.get(key, int(default)))))
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rel_time(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    finite = out[np.isfinite(out)]
    if len(finite):
        out = out - finite[0]
    return out


def finite_any(arr: np.ndarray) -> bool:
    return arr.size > 0 and bool(np.any(np.isfinite(arr)))


def base_to_sim(
    pos_m: np.ndarray,
    sim_base_height_m: float,
    sim_cartesian_x_offset_m: float,
    sim_cartesian_y_offset_m: float,
) -> np.ndarray:
    if pos_m.size == 0:
        return pos_m.reshape(0, 3)
    out = np.asarray(pos_m, dtype=float).copy()
    out[:, 0] += sim_cartesian_x_offset_m
    out[:, 1] += sim_cartesian_y_offset_m
    out[:, 2] += sim_base_height_m
    return out


def load_trajectory(rows: list[dict], args) -> dict:
    t = []
    pos = []
    vel = []
    for row in rows:
        t.append(as_float(row, "timestamp"))
        pos.append([
            as_float(row, "x_mm") * 1e-3,
            as_float(row, "y_mm") * 1e-3,
            as_float(row, "z_mm") * 1e-3,
        ])
        vel.append([
            as_float(row, "vx_mm_s") * 1e-3,
            as_float(row, "vy_mm_s") * 1e-3,
            as_float(row, "vz_mm_s") * 1e-3,
        ])
    pos_arr = np.asarray(pos, dtype=float).reshape(-1, 3)
    return {
        "t": rel_time(np.asarray(t, dtype=float)),
        "pos_base": pos_arr,
        "pos_sim": base_to_sim(
            pos_arr,
            args.sim_base_height_m,
            args.sim_cartesian_x_offset_m,
            args.sim_cartesian_y_offset_m,
        ),
        "vel": np.asarray(vel, dtype=float).reshape(-1, 3),
    }


def load_estimator(rows: list[dict], args) -> dict:
    t = []
    pos = []
    vel = []
    raw_pos = []
    valid = []
    loss = []
    extrap = []
    for row in rows:
        t.append(as_float(row, "t_now"))
        pos.append([
            as_float(row, "x_mm") * 1e-3,
            as_float(row, "y_mm") * 1e-3,
            as_float(row, "z_mm") * 1e-3,
        ])
        vel.append([
            as_float(row, "vx_mm_s") * 1e-3,
            as_float(row, "vy_mm_s") * 1e-3,
            as_float(row, "vz_mm_s") * 1e-3,
        ])
        raw_pos.append([
            as_float(row, "raw_x_mm") * 1e-3,
            as_float(row, "raw_y_mm") * 1e-3,
            as_float(row, "raw_z_mm") * 1e-3,
        ])
        valid.append(as_bool(row, "valid", False))
        loss.append(as_float(row, "loss_s"))
        extrap.append(as_float(row, "extrapolation_s"))
    pos_arr = np.asarray(pos, dtype=float).reshape(-1, 3)
    raw_arr = np.asarray(raw_pos, dtype=float).reshape(-1, 3)
    return {
        "t": rel_time(np.asarray(t, dtype=float)),
        "pos_sim": base_to_sim(
            pos_arr,
            args.sim_base_height_m,
            args.sim_cartesian_x_offset_m,
            args.sim_cartesian_y_offset_m,
        ),
        "raw_pos_sim": base_to_sim(
            raw_arr,
            args.sim_base_height_m,
            args.sim_cartesian_x_offset_m,
            args.sim_cartesian_y_offset_m,
        ),
        "vel": np.asarray(vel, dtype=float).reshape(-1, 3),
        "valid": np.asarray(valid, dtype=bool),
        "loss": np.asarray(loss, dtype=float),
        "extrap": np.asarray(extrap, dtype=float),
    }


def load_policy(rows: list[dict]) -> dict:
    t = []
    ball_pos = []
    ball_vel = []
    valid = []
    age = []
    obs = []
    actions = []
    arm_q = []
    arm_dq = []
    for row in rows:
        t.append(as_float(row, "time_sec"))
        ball_pos.append([
            as_float(row, "ball_pos_x_m"),
            as_float(row, "ball_pos_y_m"),
            as_float(row, "ball_pos_z_m"),
        ])
        ball_vel.append([
            as_float(row, "ball_vel_x_m_s"),
            as_float(row, "ball_vel_y_m_s"),
            as_float(row, "ball_vel_z_m_s"),
        ])
        valid.append(as_bool(row, "ball_valid", False))
        age.append(as_float(row, "ball_age_s"))
        obs_cols = sorted(k for k in row if k.startswith("obs/"))
        obs.append([as_float(row, k) for k in obs_cols])
        actions.append([as_float(row, f"action/{j}") for j in RIGHT_ARM_JOINTS])
        arm_q.append([as_float(row, f"arm_q/{j}") for j in RIGHT_ARM_JOINTS])
        arm_dq.append([as_float(row, f"arm_dq/{j}") for j in RIGHT_ARM_JOINTS])
    return {
        "t": rel_time(np.asarray(t, dtype=float)),
        "ball_pos": np.asarray(ball_pos, dtype=float).reshape(-1, 3),
        "ball_vel": np.asarray(ball_vel, dtype=float).reshape(-1, 3),
        "valid": np.asarray(valid, dtype=bool),
        "age": np.asarray(age, dtype=float),
        "obs": np.asarray(obs, dtype=float),
        "actions": np.asarray(actions, dtype=float).reshape(-1, 7),
        "arm_q": np.asarray(arm_q, dtype=float).reshape(-1, 7),
        "arm_dq": np.asarray(arm_dq, dtype=float).reshape(-1, 7),
    }


def plot_series(ax, t: np.ndarray, y: np.ndarray, label: str, **kwargs) -> None:
    if len(t) == 0 or y.size == 0:
        return
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) == 0:
        return
    ax.plot(t[mask], y[mask], label=label, **kwargs)


def plot_ball_state(output: Path, traj: dict, est: dict, policy: dict, args) -> None:
    labels = ("x", "y", "z")
    fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=False)
    fig.suptitle("Real Run Ball State")

    for i, name in enumerate(labels):
        ax = axes[i, 0]
        plot_series(ax, traj["t"], traj["pos_sim"][:, i], f"detector {name}_sim", linewidth=1.4)
        plot_series(ax, est["t"], est["pos_sim"][:, i], f"est/control {name}_sim", linewidth=1.1, alpha=0.85)
        plot_series(ax, policy["t"], policy["ball_pos"][:, i], f"policy {name}_sim", linewidth=0.9, alpha=0.75)
        ax.set_ylabel(f"{name} pos (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        ax = axes[i, 1]
        plot_series(ax, traj["t"], traj["vel"][:, i], f"detector v{name}", linewidth=1.4)
        plot_series(ax, est["t"], est["vel"][:, i], f"est/control v{name}", linewidth=1.1, alpha=0.85)
        plot_series(ax, policy["t"], policy["ball_vel"][:, i], f"policy v{name}", linewidth=0.9, alpha=0.75)
        ax.set_ylabel(f"v{name} (m/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[3, 0].axhline(args.done_z_min_m, color="#b91c1c", linestyle="--", linewidth=1.2, label="done z min")
    axes[3, 0].axhline(args.done_z_max_m, color="#b91c1c", linestyle="--", linewidth=1.2, label="done z max")
    plot_series(axes[3, 0], traj["t"], traj["pos_sim"][:, 2], "detector z_sim", linewidth=1.3)
    plot_series(axes[3, 0], policy["t"], policy["ball_pos"][:, 2], "policy z_sim", linewidth=1.0, alpha=0.8)
    axes[3, 0].set_ylabel("z bounds (m)")
    axes[3, 0].set_xlabel("time (s)")
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].legend(loc="best", fontsize=8)

    plot_series(axes[3, 1], policy["t"], policy["age"] * 1e3, "policy ball age", linewidth=1.2)
    plot_series(axes[3, 1], est["t"], est["loss"] * 1e3, "est loss", linewidth=1.0, alpha=0.8)
    plot_series(axes[3, 1], est["t"], est["extrap"] * 1e3, "est extrap", linewidth=1.0, alpha=0.8)
    if len(policy["t"]):
        axes[3, 1].plot(policy["t"], policy["valid"].astype(float) * 10.0, label="policy valid x10", linewidth=0.9)
    axes[3, 1].set_ylabel("ms / valid")
    axes[3, 1].set_xlabel("time (s)")
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"[real2d] wrote: {output}")


def plot_policy_io(output: Path, policy: dict) -> None:
    t = policy["t"]
    obs = policy["obs"]
    fig, axes = plt.subplots(6, 1, figsize=(16, 15), sharex=True)
    fig.suptitle("Real Run Policy Inputs / Outputs")

    def plot_obs_group(ax, start: int, names: tuple[str, ...], title: str) -> None:
        if obs.size == 0 or obs.shape[1] <= start:
            return
        for j, name in enumerate(names):
            idx = start + j
            if idx < obs.shape[1]:
                plot_series(ax, t, obs[:, idx], f"obs/{idx:03d} {name}", linewidth=1.0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8, ncol=3)

    plot_obs_group(axes[0], 20, ("ball_x", "ball_y", "ball_z"), "Policy Input: Ball Position")
    plot_obs_group(axes[1], 23, ("ball_vx", "ball_vy", "ball_vz"), "Policy Input: Ball Velocity")
    plot_obs_group(axes[2], 26, ("racket_x", "racket_y", "racket_z"), "Policy Input: Racket Position")
    plot_obs_group(axes[3], 32, ("rel_x", "rel_y", "rel_z"), "Policy Input: Ball - Racket")
    if obs.size and obs.shape[1] > 49:
        plot_series(axes[3], t, obs[:, 49], "obs/049 ball_age_norm", linewidth=1.2, linestyle="--")
    if obs.size and obs.shape[1] > 66:
        plot_series(axes[3], t, obs[:, 65], "obs/065 contact_t", linewidth=1.0, alpha=0.8)
        plot_series(axes[3], t, obs[:, 66], "obs/066 margin", linewidth=1.0, alpha=0.8)

    for i, joint in enumerate(RIGHT_ARM_JOINTS):
        plot_series(axes[4], t, policy["actions"][:, i], f"action/{joint}", linewidth=0.95)
    axes[4].set_title("Policy Output: Normalized Actions")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="best", fontsize=8, ncol=4)

    for i, joint in enumerate(RIGHT_ARM_JOINTS):
        plot_series(axes[5], t, policy["arm_q"][:, i], f"q/{joint}", linewidth=0.9)
    axes[5].set_title("Right Arm Feedback q")
    axes[5].set_xlabel("time (s)")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend(loc="best", fontsize=8, ncol=4)

    for ax in axes:
        ax.set_ylabel("value")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"[real2d] wrote: {output}")


def write_summary(output: Path, traj_rows: list[dict], est_rows: list[dict], policy_rows: list[dict]) -> None:
    summary = {
        "trajectory_rows": len(traj_rows),
        "estimator_rows": len(est_rows),
        "policy_rows": len(policy_rows),
    }
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[real2d] wrote: {output}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot one real pingpong hardware run.")
    p.add_argument("--trajectory-csv", type=Path, required=True)
    p.add_argument("--policy-trace", type=Path, required=True)
    p.add_argument("--estimator-trace", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sim-base-height-m", type=float, default=0.947396)
    p.add_argument("--sim-cartesian-x-offset-m", type=float, default=0.0)
    p.add_argument("--sim-cartesian-y-offset-m", type=float, default=0.023583)
    p.add_argument("--done-z-min-m", type=float, default=0.8)
    p.add_argument("--done-z-max-m", type=float, default=1.9)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    traj_rows = read_csv_rows(args.trajectory_csv)
    policy_rows = read_csv_rows(args.policy_trace)
    est_rows = read_csv_rows(args.estimator_trace)
    if not traj_rows:
        print(f"[real2d] WARNING: no trajectory rows: {args.trajectory_csv}")
    if not policy_rows:
        print(f"[real2d] WARNING: no policy rows: {args.policy_trace}")
    if not traj_rows and not policy_rows:
        return 1

    traj = load_trajectory(traj_rows, args) if traj_rows else {
        "t": np.array([]), "pos_base": np.empty((0, 3)),
        "pos_sim": np.empty((0, 3)), "vel": np.empty((0, 3)),
    }
    est = load_estimator(est_rows, args) if est_rows else {
        "t": np.array([]), "pos_sim": np.empty((0, 3)),
        "raw_pos_sim": np.empty((0, 3)), "vel": np.empty((0, 3)),
        "valid": np.array([], dtype=bool), "loss": np.array([]),
        "extrap": np.array([]),
    }
    policy = load_policy(policy_rows) if policy_rows else {
        "t": np.array([]), "ball_pos": np.empty((0, 3)),
        "ball_vel": np.empty((0, 3)), "valid": np.array([], dtype=bool),
        "age": np.array([]), "obs": np.empty((0, 0)),
        "actions": np.empty((0, 7)), "arm_q": np.empty((0, 7)),
        "arm_dq": np.empty((0, 7)),
    }

    if finite_any(traj["pos_sim"]) or finite_any(policy["ball_pos"]):
        plot_ball_state(args.output_dir / "ball_state_2d.png", traj, est, policy, args)
    if policy_rows:
        plot_policy_io(args.output_dir / "policy_io_2d.png", policy)
    write_summary(args.output_dir / "run_summary.json", traj_rows, est_rows, policy_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
