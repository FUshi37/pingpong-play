#!/usr/bin/env python3
"""Render final real/baseline/final-tuned drag-ball trajectory comparisons."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_rmp_three_way_v2_mpl")

import numpy as np

import analyze_drag_ball_sim2real as drag
import analyze_valid_juggle_sim2real as base
import plot_drag_ball_rmp_three_way as plots
from tune_drag_ball_rmp_replay_v2 import CandidateV2, TunerV2


def _proposal(candidate: dict) -> dict:
    return {
        "recovered_rmp_jnt_kp_scale_per_joint": [1.0] * 7,
        "recovered_rmp_jnt_kd_scale_per_joint": [
            1.0, 1.0, 0.92, 1.0, 1.0, 1.0, 1.0],
        "recovered_rmp_output_filter_length": int(
            candidate["output_filter_length"]),
        "recovered_rmp_target_filter_length": int(
            candidate["target_filter_length"]),
        "recovered_rmp_target_update_delay_steps": int(
            candidate["target_update_delay_steps"]),
        "recovered_rmp_output_delay_steps_per_joint": [
            int(value) for value in
            candidate["output_delay_steps_per_joint"]],
        "recovered_rmp_joint_velocity_feedforward": float(
            candidate["joint_velocity_feedforward"]),
        "recovered_rmp_joint_acceleration_feedforward": float(
            candidate["joint_acceleration_feedforward"]),
        "recovered_rmp_estimator_joint_process_var": 50.0,
        "recovered_rmp_estimator_joint_measure_var": 0.0001,
        "arm_acc_limit_deg_s2": [
            1300.0,
            1300.0,
            1800.0,
            3000.0 * float(
                candidate["acceleration_limit_scale_per_joint"][3]),
            3000.0,
            3000.0,
            3000.0,
        ],
        "recovered_rmp_dof_damping_scale_per_joint": [
            0.6084, 0.9984, 0.9984, 1.92, 0.58, 1.17, 0.78],
        "recovered_rmp_dof_armature_scale_per_joint": [
            0.4524, 1.65, 0.92, 1.22, 0.58, 0.5336, 0.58],
        "recovered_rmp_acceleration_feedforward_s2": [
            float(value) for value in
            candidate["acceleration_feedforward_s2_per_joint"]],
    }


def _write_report(path: Path, result: dict) -> None:
    metrics = result["closed_loop_metrics"]
    before = result["reference_v1_closed_loop_metrics"]
    final_search = result["refined_search_metrics"]
    lines = [
        "# Drag-ball RMP V2 最终三轨迹对比",
        "",
        "所有图片同时包含实物、记录中的原始 sim mirror、V2 调参后独立 sim mirror。调参后镜像保持相同初始状态、相同 drag-ball 输入和相同 actor，仅替换 recovered-RMP/plant 参数；曲线均为无 marker 的细实线。",
        "",
        "## 最终参数要点",
        "",
        f"- target update servo substep index：`{result['selected_candidate']['target_update_delay_steps']}`（每个 5 ms 控制周期含 5 个 1 ms substep）。",
        f"- output delay J1–J7：`{result['selected_candidate']['output_delay_steps_per_joint']} ms`。",
        f"- joint acceleration feedforward：`{result['selected_candidate']['joint_acceleration_feedforward']}`。",
        f"- J4 acceleration limit scale：`{result['selected_candidate']['acceleration_limit_scale_per_joint'][3]}`。",
        f"- low-level qdd feedforward：`{result['selected_candidate']['acceleration_feedforward_s2_per_joint']}`。",
        "- jnt_kp/jnt_kd、filter、estimator、damping、armature 保持 checkpoint 原值。",
        "",
        "## 相对原始 sim mirror 的闭环误差",
        "",
        "| 时间窗 | 信号 | 调参前 RMS | V2 调参后 RMS | 改善 |",
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
            improvement = -float(item["relative_change"])
            lines.append(
                f"| {window} | {names[name]} | "
                f"{item['before_rms']:.6f} {units[name]} | "
                f"{item['after_rms']:.6f} {units[name]} | "
                f"{100.0 * improvement:.1f}% |")
    lines.extend([
        "",
        "## 相对上一版调参",
        "",
        f"- 上一版 post-warmup：q=`{before['post_warmup']['q']['after_rms']:.6f} rad`，dq=`{before['post_warmup']['dq']['after_rms']:.6f} rad/s`。",
        f"- V2 post-warmup：q=`{metrics['post_warmup']['q']['after_rms']:.6f} rad`，dq=`{metrics['post_warmup']['dq']['after_rms']:.6f} rad/s`。",
        f"- V2 未参与选择的 14 s+ 留出段：q=`{final_search['held_out_14_s_to_end']['q_rms_rad']:.6f} rad`，dq=`{final_search['held_out_14_s_to_end']['dq_rms_rad_s']:.6f} rad/s`。",
        "",
        "## 固定同 q_ref 的执行器 replay 复核",
        "",
        f"- 最终闭环候选 post-warmup：q=`{result['final_same_command_metrics']['post_warmup']['q_rms_rad']:.6f} rad`，dq=`{result['final_same_command_metrics']['post_warmup']['dq_rms_rad_s']:.6f} rad/s`。",
        f"- 最终闭环候选 14 s+：q=`{result['final_same_command_metrics']['held_out_14_s_to_end']['q_rms_rad']:.6f} rad`，dq=`{result['final_same_command_metrics']['held_out_14_s_to_end']['dq_rms_rad_s']:.6f} rad/s`。",
        "- 该指标略差于上一版同 q_ref 最优值，说明最终候选优化的是策略闭环整体相位，而不是单独执行器 replay 的唯一最优点；两者不能混为同一目标。",
        "",
        "## 解释限制",
        "",
        "V2 已显著缩小闭环相位分岔，但不会完全重合。实物和独立 sim mirror 使用各自的关节状态运行 actor；任何小状态误差都会改变下一拍 action/q_ref，随后累积为闭环轨迹差异。用单条记录继续直接最小化全段误差会把 RMP 参数拟合成该条策略轨迹的补偿器，而不再是可泛化的执行器模型。因此参数选择没有使用 14 s+ 留出段。",
        "",
        "## 图片",
        "",
        "- `01–03`：关节角、角速度、角加速度。",
        "- `04`：策略 qcmd 和 actor action。",
        "- `05–10`：球拍位置、速度、加速度、Euler 角、角速度、角加速度。",
        "- `11`：调参前后闭环误差随时间变化。",
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
        else recording / "analysis" / "v91_rmp_tuning_drag_ball_v2")
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else tuning_dir / "three_way_trajectory_comparison")
    output.mkdir(parents=True, exist_ok=True)

    refined = json.loads((
        tuning_dir / "closed_loop_refined_results.json").read_text(
            encoding="utf-8"))
    selected_npz = np.load(tuning_dir / "closed_loop_selected_replay.npz")
    previous_metrics = json.loads((
        recording / "analysis" / "v91_rmp_tuning_drag_ball"
        / "three_way_trajectory_comparison"
        / "three_way_comparison_metrics.json").read_text(encoding="utf-8"))

    stage, aligned, _excluded = base.load_alignment(recording)
    controller = base.init_kinematics(base.DEFAULT_CHECKPOINT, base.DEFAULT_XML)
    base.populate_kinematics(stage, aligned, controller)
    elapsed = np.asarray([item["elapsed"] for item in aligned])
    tuned = {
        key: np.asarray(selected_npz[key], dtype=np.float64)
        for key in ("q", "dq", "qcmd", "action")
    }
    if any(len(value) != len(aligned) for value in tuned.values()):
        raise RuntimeError("selected closed-loop replay length mismatch")
    real = {
        "q": base.aligned_matrix(aligned, "real_q"),
        "dq": base.aligned_matrix(aligned, "real_dq"),
        "qcmd": base.aligned_matrix(aligned, "real_qcmd"),
        "action": base.aligned_matrix(aligned, "real_action"),
    }
    recorded = {
        "q": base.aligned_matrix(aligned, "sim_q"),
        "dq": base.aligned_matrix(aligned, "sim_dq"),
        "qcmd": base.aligned_matrix(aligned, "sim_qcmd"),
        "action": base.aligned_matrix(aligned, "sim_action"),
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
    tuned_kin = plots._kinematics(controller, tuned["q"], tuned["dq"])
    real_ddq = base.aligned_matrix(aligned, "real_ddq")
    recorded_ddq = base.aligned_matrix(aligned, "sim_ddq")
    tuned_ddq = drag._uniform_difference(tuned["dq"])
    joints = [f"J{index}" for index in range(1, 8)]
    axes = ["x", "y", "z"]
    ball = np.asarray([
        base.vector((item["sim"].get("sync") or {}).get("inputBall"), 3)
        for item in aligned], dtype=np.float64)

    plots._plot_family(
        output / "01_joint_position.png", elapsed,
        real["q"], recorded["q"], tuned["q"],
        "Joint position: real vs sim mirror before/after RMP V2 tuning",
        "rad", joints)
    plots._plot_family(
        output / "02_joint_velocity.png", elapsed,
        real["dq"], recorded["dq"], tuned["dq"],
        "Joint velocity: real vs sim mirror before/after RMP V2 tuning",
        "rad/s", joints)
    plots._plot_family(
        output / "03_joint_acceleration.png", elapsed,
        real_ddq, recorded_ddq, tuned_ddq,
        "Joint acceleration: real vs sim mirror before/after RMP V2 tuning",
        "rad/s²", joints)
    plots._plot_policy_outputs(
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
        plots._plot_family(
            output / filename, elapsed,
            real_kin[key], recorded_kin[key], tuned_kin[key],
            f"{title}: real vs sim mirror before/after RMP V2 tuning",
            unit, axes, ball=ball if key == "position" else None)
    plots._plot_gap_summary(
        output / "11_gap_summary.png", elapsed, [
            ("joint q RMS [rad]", real["q"], recorded["q"], tuned["q"]),
            ("joint dq RMS [rad/s]", real["dq"], recorded["dq"], tuned["dq"]),
            ("racket position RMS [m]", real_kin["position"], recorded_kin["position"], tuned_kin["position"]),
            ("racket velocity RMS [m/s]", real_kin["velocity"], recorded_kin["velocity"], tuned_kin["velocity"]),
        ])

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
    selected_mapping = dict(refined["selected_candidate"])
    candidate_defaults = CandidateV2()
    for field in candidate_defaults.__dataclass_fields__:
        if field not in selected_mapping:
            selected_mapping[field] = getattr(candidate_defaults, field)
        if isinstance(getattr(candidate_defaults, field), tuple):
            selected_mapping[field] = tuple(selected_mapping[field])
    same_command_candidate = CandidateV2(**selected_mapping)
    commands = np.asarray([
        drag._published_command(item["real"]) for item in aligned])
    same_command_output = tuning_dir / "final_same_command_evaluation"
    same_command_output.mkdir(parents=True, exist_ok=True)
    same_command_tuner = TunerV2(
        recording, aligned, real["q"], real["dq"], commands, elapsed,
        same_command_output)
    try:
        final_same_command = same_command_tuner.evaluate(
            same_command_candidate, search_only=False)
    finally:
        same_command_tuner.close()
    result = {
        "recording": recording.name,
        "ticks": len(aligned),
        "duration_s": float(elapsed[-1]),
        "comparison_contract": {
            "before": "recorded online sim mirror",
            "after": (
                "closed-loop mirror with identical initial state, drag-ball "
                "input, actor, and selected RMP V2 candidate"),
            "selection": (
                "train 1-9s + validation 9-14s; 14s+ held out"),
            "line_style": "thin solid lines without markers",
        },
        "selected_candidate": refined["selected_candidate"],
        "refined_search_metrics": refined["selected_metrics"],
        "reference_v1_closed_loop_metrics": previous_metrics[
            "closed_loop_metrics"],
        "final_same_command_metrics": final_same_command.metrics,
        "closed_loop_metrics": plots._metric_windows(
            elapsed, metric_real, metric_baseline, metric_tuned),
        "proposed_env_cfg_override": _proposal(
            refined["selected_candidate"]),
    }
    (output / "three_way_comparison_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    (tuning_dir / "proposed_rmp_env_cfg_override_closed_loop.json").write_text(
        json.dumps(
            result["proposed_env_cfg_override"], indent=2,
            ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    _write_report(output / "THREE_WAY_TRAJECTORY_COMPARISON_ZH.md", result)
    print(json.dumps({
        "output": str(output),
        "selected_candidate": result["selected_candidate"],
        "closed_loop_metrics": result["closed_loop_metrics"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
