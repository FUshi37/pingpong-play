#!/usr/bin/env python3
"""Plot real, recorded baseline mirror, and tuned closed-loop mirror trajectories.

The tuned mirror consumes the exact recorded drag-ball input stream, starts
from the recorded mirror tick-start state, runs the same actor, and differs
only in the recovered-RMP candidate.  A fresh baseline closed-loop replay is
also run as a reproducibility diagnostic; plots use the recorded sim mirror
as the requested before-tuning reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_rmp_three_way_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_drag_ball_sim2real as drag
import analyze_valid_juggle_sim2real as base
import tune_drag_ball_rmp_replay as tuning


COLORS = {
    "real": "#1769aa",
    "baseline": "#e76f51",
    "tuned": "#2a9d8f",
    "ball": "#6c757d",
}
LINE_WIDTHS = {"real": 0.72, "baseline": 0.62, "tuned": 0.62}


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else math.nan


def _matrix(rows: list[dict[str, Any]], key: str, size: int) -> np.ndarray:
    output = []
    for row in rows:
        value = base.vector(row.get(key), size)
        if value is None:
            raise RuntimeError(f"missing {key} in closed-loop replay snapshot")
        output.append(value)
    return np.asarray(output, dtype=np.float64)


def _run_closed_loop(
    recording: Path,
    aligned: list[dict[str, Any]],
    candidate: tuning.Candidate | None,
) -> dict[str, np.ndarray]:
    from pingpong_controller.sim_mirror import RealBallSimMirror

    mirror = RealBallSimMirror(
        drag._mirror_config(recording, base.DEFAULT_CHECKPOINT, base.DEFAULT_XML))
    if candidate is not None:
        tuning.apply_candidate_to_mirror(mirror, candidate)
    first = aligned[0]
    drag._reset_replay_mirror(
        mirror, first["sim_q"], first["sim_dq"], first)
    options = SimpleNamespace(
        rl_use_valid_ball_obs_age=True,
        policy_observation_scales=(),
    )
    q_rows: list[np.ndarray] = []
    dq_rows: list[np.ndarray] = []
    snapshots: list[dict[str, Any]] = []
    try:
        for item in aligned:
            q_rows.append(
                mirror.data.qpos[mirror.arm_qadr].astype(np.float64).copy())
            dq_rows.append(
                mirror.data.qvel[mirror.arm_vadr].astype(np.float64).copy())
            sync = item["sim"].get("sync") or {}
            ball = base.vector(sync.get("inputBall"), 3)
            velocity = base.vector(sync.get("inputVelocity"), 3)
            if ball is None or velocity is None:
                raise RuntimeError(
                    f"missing drag-ball input at sourceSeq={item['seq']}")
            snapshots.append(mirror.step(
                ball.astype(np.float32),
                velocity.astype(np.float32),
                ball_valid=bool(sync.get("inputValid", True)),
                ball_obs_age_s=float(sync.get("inputAgeS", 0.0) or 0.0),
                options=options,
            ))
    finally:
        mirror.close()
    return {
        "q": np.asarray(q_rows),
        "dq": np.asarray(dq_rows),
        "qcmd": _matrix(snapshots, "policyCommand", 7),
        "action": _matrix(snapshots, "policyAction", 7),
    }


def _kinematics(
    controller: Any, q: np.ndarray, dq: np.ndarray
) -> dict[str, np.ndarray]:
    rows = [base.pose_twist(controller, q_row, dq_row)
            for q_row, dq_row in zip(q, dq)]
    result = {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in ("position", "quaternion", "velocity", "angular_velocity")
    }
    result["euler"] = base.quaternion_to_euler_deg(result["quaternion"])
    result["acceleration"] = drag._uniform_difference(result["velocity"])
    result["angular_acceleration"] = drag._uniform_difference(
        result["angular_velocity"])
    return result


def _plot_family(
    output: Path,
    elapsed: np.ndarray,
    real: np.ndarray,
    baseline: np.ndarray,
    tuned: np.ndarray,
    title: str,
    unit: str,
    labels: list[str],
    ball: np.ndarray | None = None,
    *,
    tuned_label: str = "sim mirror after RMP tuning",
) -> None:
    dimensions = real.shape[1]
    height = 18 if dimensions == 7 else 10
    figure, axes = plt.subplots(
        dimensions, 1, figsize=(15, height), sharex=True,
        constrained_layout=True)
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        axis.plot(
            elapsed, baseline[:, index], color=COLORS["baseline"],
            linewidth=LINE_WIDTHS["baseline"],
            label="sim mirror before tuning" if index == 0 else None)
        axis.plot(
            elapsed, tuned[:, index], color=COLORS["tuned"],
            linewidth=LINE_WIDTHS["tuned"],
            label=tuned_label if index == 0 else None)
        axis.plot(
            elapsed, real[:, index], color=COLORS["real"],
            linewidth=LINE_WIDTHS["real"], alpha=0.90,
            label="real hardware" if index == 0 else None)
        if ball is not None:
            axis.plot(
                elapsed, ball[:, index], color=COLORS["ball"],
                linewidth=0.50, alpha=0.72,
                label="drag-ball input" if index == 0 else None)
        axis.set_ylabel(f"{labels[index]}\n[{unit}]")
        axis.grid(alpha=0.23)
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_policy_outputs(
    output: Path,
    elapsed: np.ndarray,
    real_qcmd: np.ndarray,
    baseline_qcmd: np.ndarray,
    tuned_qcmd: np.ndarray,
    real_action: np.ndarray,
    baseline_action: np.ndarray,
    tuned_action: np.ndarray,
    *,
    tuned_label: str = "sim after",
    title: str = "Policy output: real vs sim mirror before/after RMP tuning",
) -> None:
    figure, axes = plt.subplots(
        7, 2, figsize=(17, 19), sharex=True, constrained_layout=True)
    for joint in range(7):
        for column, values in enumerate((
            (real_qcmd, baseline_qcmd, tuned_qcmd, "qcmd rad"),
            (real_action, baseline_action, tuned_action, "action"),
        )):
            real, baseline, tuned, unit = values
            axis = axes[joint, column]
            axis.plot(
                elapsed, baseline[:, joint], color=COLORS["baseline"],
                linewidth=LINE_WIDTHS["baseline"],
                label="sim before" if joint == 0 else None)
            axis.plot(
                elapsed, tuned[:, joint], color=COLORS["tuned"],
                linewidth=LINE_WIDTHS["tuned"],
                label=tuned_label if joint == 0 else None)
            axis.plot(
                elapsed, real[:, joint], color=COLORS["real"],
                linewidth=LINE_WIDTHS["real"], alpha=0.90,
                label="real" if joint == 0 else None)
            axis.set_ylabel(f"J{joint + 1}\n[{unit}]")
            axis.grid(alpha=0.23)
    axes[0, 0].set_title("Nominal policy command position")
    axes[0, 1].set_title("Actor action")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time from policy start [s]")
    axes[-1, 1].set_xlabel("Time from policy start [s]")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_gap_summary(
    output: Path,
    elapsed: np.ndarray,
    signal_triplets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    *,
    after_label: str = "after tuning",
    title: str = "Closed-loop Sim2Real gap before/after RMP tuning",
) -> None:
    figure, axes = plt.subplots(
        len(signal_triplets), 1, figsize=(15, 12), sharex=True,
        constrained_layout=True)
    for axis, (label, real, baseline, tuned) in zip(axes, signal_triplets):
        before = np.sqrt(np.mean((baseline - real) ** 2, axis=1))
        after = np.sqrt(np.mean((tuned - real) ** 2, axis=1))
        axis.plot(
            elapsed, before, color=COLORS["baseline"], linewidth=0.65,
            label="before tuning")
        axis.plot(
            elapsed, after, color=COLORS["tuned"], linewidth=0.65,
            label=after_label)
        axis.set_ylabel(label)
        axis.grid(alpha=0.23)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _metric_windows(
    elapsed: np.ndarray,
    real: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    tuned: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for window, mask in {
        "full": np.ones(len(elapsed), dtype=bool),
        "post_warmup": elapsed >= 1.0,
        "held_out_14_s_to_end": elapsed >= 14.0,
    }.items():
        result[window] = {}
        for name in ("q", "dq", "racket_position", "racket_velocity"):
            before = _rms(baseline[name][mask] - real[name][mask])
            after = _rms(tuned[name][mask] - real[name][mask])
            result[window][name] = {
                "before_rms": before,
                "after_rms": after,
                "relative_change": (
                    (after - before) / before if before > 0.0 else 0.0),
            }
    return result


def _write_report(
    path: Path,
    result: dict[str, Any],
) -> None:
    metrics = result["closed_loop_metrics"]
    validation = result["fresh_baseline_reproduction"]
    lines = [
        "# RMP 调参前后三轨迹对比",
        "",
        "每幅轨迹图同时包含实物、记录中的原始 sim mirror，以及仅替换 RMP 参数后重新运行的独立 sim mirror。后者使用相同初始状态、相同 drag-ball 输入和相同策略。所有曲线均为无 marker 的细实线。",
        "",
        "## 重放语义",
        "",
        "- `before`：记录文件里的原始在线 sim mirror。",
        "- `after`：从记录的首个 sim tick-start 状态启动，逐 tick 输入相同 drag-ball 观测，运行相同 actor，仅替换拟合后的 recovered-RMP 参数。",
        "- `fresh baseline reproduction`：同样重新运行，但不改 RMP，用于评估仅凭六位小数记录状态能否重建原闭环。",
        "",
        "## Fresh baseline 对记录 sim mirror 的重建误差",
        "",
        f"- q RMS：`{validation['q_rms_rad']:.8f} rad`",
        f"- dq RMS：`{validation['dq_rms_rad_s']:.7f} rad/s`",
        f"- qcmd RMS：`{validation['qcmd_rms_rad']:.8f} rad`",
        f"- action RMS：`{validation['action_rms']:.8f}`",
        "",
        "## 闭环 Sim2Real 误差",
        "",
        "| 时间窗 | 信号 | 调参前 RMS | 调参后 RMS | 相对变化（负值=改善） |",
        "|---|---|---:|---:|---:|",
    ]
    units = {
        "q": "rad", "dq": "rad/s", "racket_position": "m",
        "racket_velocity": "m/s",
    }
    names = {
        "q": "关节角", "dq": "关节角速度",
        "racket_position": "球拍位置", "racket_velocity": "球拍速度",
    }
    for window, values in metrics.items():
        for name, item in values.items():
            lines.append(
                f"| {window} | {names[name]} | "
                f"{item['before_rms']:.6f} {units[name]} | "
                f"{item['after_rms']:.6f} {units[name]} | "
                f"{100.0 * item['relative_change']:+.1f}% |")
    lines.extend([
        "",
        "注意：这是把 RMP 参数直接放回策略闭环后的结果；它与同 q_ref 执行器 replay 指标回答的问题不同。参数变化会改变关节状态，随后 actor 的 q_ref 也会发生分岔。",
        "",
        "## 图片",
        "",
        "- `01–03`：关节角、角速度、角加速度。",
        "- `04`：策略 qcmd 和 actor action。",
        "- `05–10`：球拍三维位置、速度、加速度、Euler 角、角速度、角加速度。",
        "- `11`：调参前后闭环误差随时间的变化。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording", nargs="?", type=Path,
        default=base.ROOT / (
            "src/pingpong_moz1/pingpong_controller/outputs/"
            "interactive_recordings/probe_recording_20260829_180205"))
    parser.add_argument("--tuning-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording = args.recording.expanduser().resolve()
    tuning_dir = (
        args.tuning_dir.expanduser().resolve()
        if args.tuning_dir is not None
        else recording / "analysis" / "v91_rmp_tuning_drag_ball")
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else tuning_dir / "three_way_trajectory_comparison")
    output.mkdir(parents=True, exist_ok=True)

    tuning_result = json.loads(
        (tuning_dir / "rmp_tuning_results.json").read_text(encoding="utf-8"))
    selected = dict(tuning_result["selected_candidate"])
    selected["output_delay_per_joint_adjustment"] = tuple(
        int(value) for value in selected["output_delay_per_joint_adjustment"])
    candidate = tuning.Candidate(**selected)

    stage, aligned, _excluded = base.load_alignment(recording)
    controller = base.init_kinematics(base.DEFAULT_CHECKPOINT, base.DEFAULT_XML)
    base.populate_kinematics(stage, aligned, controller)
    elapsed = np.asarray([item["elapsed"] for item in aligned])

    recorded = {
        "q": base.aligned_matrix(aligned, "sim_q"),
        "dq": base.aligned_matrix(aligned, "sim_dq"),
        "qcmd": base.aligned_matrix(aligned, "sim_qcmd"),
        "action": base.aligned_matrix(aligned, "sim_action"),
    }
    real = {
        "q": base.aligned_matrix(aligned, "real_q"),
        "dq": base.aligned_matrix(aligned, "real_dq"),
        "qcmd": base.aligned_matrix(aligned, "real_qcmd"),
        "action": base.aligned_matrix(aligned, "real_action"),
    }

    fresh_baseline = _run_closed_loop(recording, aligned, None)
    tuned = _run_closed_loop(recording, aligned, candidate)
    reproduction = {
        "q_rms_rad": _rms(fresh_baseline["q"] - recorded["q"]),
        "dq_rms_rad_s": _rms(fresh_baseline["dq"] - recorded["dq"]),
        "qcmd_rms_rad": _rms(fresh_baseline["qcmd"] - recorded["qcmd"]),
        "action_rms": _rms(fresh_baseline["action"] - recorded["action"]),
    }

    real_kin = {
        "position": base.aligned_matrix(aligned, "real_kin.position"),
        "velocity": base.aligned_matrix(aligned, "real_kin.velocity"),
        "euler": base.aligned_matrix(aligned, "real_euler"),
        "angular_velocity": base.aligned_matrix(
            aligned, "real_kin.angular_velocity"),
        "acceleration": base.aligned_matrix(
            aligned, "real_racket_acceleration"),
        "angular_acceleration": base.aligned_matrix(
            aligned, "real_racket_angular_acceleration"),
    }
    recorded_kin = {
        "position": base.aligned_matrix(aligned, "sim_kin.position"),
        "velocity": base.aligned_matrix(aligned, "sim_kin.velocity"),
        "euler": base.aligned_matrix(aligned, "sim_euler"),
        "angular_velocity": base.aligned_matrix(
            aligned, "sim_kin.angular_velocity"),
        "acceleration": base.aligned_matrix(
            aligned, "sim_racket_acceleration"),
        "angular_acceleration": base.aligned_matrix(
            aligned, "sim_racket_angular_acceleration"),
    }
    tuned_kin = _kinematics(controller, tuned["q"], tuned["dq"])

    real_ddq = base.aligned_matrix(aligned, "real_ddq")
    recorded_ddq = base.aligned_matrix(aligned, "sim_ddq")
    tuned_ddq = drag._uniform_difference(tuned["dq"])
    joints = [f"J{index}" for index in range(1, 8)]
    axes = ["x", "y", "z"]
    ball = np.asarray([
        base.vector((item["sim"].get("sync") or {}).get("inputBall"), 3)
        for item in aligned], dtype=np.float64)

    _plot_family(
        output / "01_joint_position.png", elapsed,
        real["q"], recorded["q"], tuned["q"],
        "Joint position: real vs sim mirror before/after RMP tuning",
        "rad", joints)
    _plot_family(
        output / "02_joint_velocity.png", elapsed,
        real["dq"], recorded["dq"], tuned["dq"],
        "Joint velocity: real vs sim mirror before/after RMP tuning",
        "rad/s", joints)
    _plot_family(
        output / "03_joint_acceleration.png", elapsed,
        real_ddq, recorded_ddq, tuned_ddq,
        "Joint acceleration: real vs sim mirror before/after RMP tuning",
        "rad/s²", joints)
    _plot_policy_outputs(
        output / "04_policy_command_and_action.png", elapsed,
        real["qcmd"], recorded["qcmd"], tuned["qcmd"],
        real["action"], recorded["action"], tuned["action"])
    for filename, title, unit, key in (
        ("05_racket_position.png", "Racket-center position", "m", "position"),
        ("06_racket_velocity.png", "Racket-center linear velocity", "m/s", "velocity"),
        ("07_racket_acceleration.png", "Racket-center linear acceleration", "m/s²", "acceleration"),
        ("08_racket_orientation.png", "Racket XYZ Euler orientation", "deg", "euler"),
        ("09_racket_angular_velocity.png", "Racket angular velocity in base frame", "rad/s", "angular_velocity"),
        ("10_racket_angular_acceleration.png", "Racket angular acceleration in base frame", "rad/s²", "angular_acceleration"),
    ):
        _plot_family(
            output / filename, elapsed,
            real_kin[key], recorded_kin[key], tuned_kin[key],
            f"{title}: real vs sim mirror before/after RMP tuning",
            unit, axes, ball=ball if key == "position" else None)

    metric_real = {
        "q": real["q"], "dq": real["dq"],
        "racket_position": real_kin["position"],
        "racket_velocity": real_kin["velocity"],
    }
    metric_baseline = {
        "q": recorded["q"], "dq": recorded["dq"],
        "racket_position": recorded_kin["position"],
        "racket_velocity": recorded_kin["velocity"],
    }
    metric_tuned = {
        "q": tuned["q"], "dq": tuned["dq"],
        "racket_position": tuned_kin["position"],
        "racket_velocity": tuned_kin["velocity"],
    }
    _plot_gap_summary(
        output / "11_gap_summary.png", elapsed, [
            ("joint q RMS [rad]", real["q"], recorded["q"], tuned["q"]),
            ("joint dq RMS [rad/s]", real["dq"], recorded["dq"], tuned["dq"]),
            ("racket position RMS [m]", real_kin["position"], recorded_kin["position"], tuned_kin["position"]),
            ("racket velocity RMS [m/s]", real_kin["velocity"], recorded_kin["velocity"], tuned_kin["velocity"]),
        ])

    result = {
        "recording": recording.name,
        "ticks": len(aligned),
        "duration_s": float(elapsed[-1]),
        "comparison_contract": {
            "before": "recorded online sim mirror",
            "after": (
                "fresh closed-loop mirror with identical initial tick-start "
                "state, drag-ball input stream, actor, and tuned RMP only"),
            "line_style": "thin solid lines without markers",
        },
        "selected_candidate": tuning_result["selected_candidate"],
        "fresh_baseline_reproduction": reproduction,
        "closed_loop_metrics": _metric_windows(
            elapsed, metric_real, metric_baseline, metric_tuned),
    }
    (output / "three_way_comparison_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    _write_report(output / "THREE_WAY_TRAJECTORY_COMPARISON_ZH.md", result)
    print(json.dumps({
        "output": str(output),
        "fresh_baseline_reproduction": reproduction,
        "closed_loop_metrics": result["closed_loop_metrics"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
