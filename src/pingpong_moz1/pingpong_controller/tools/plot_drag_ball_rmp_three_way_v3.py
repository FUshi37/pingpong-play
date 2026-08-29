#!/usr/bin/env python3
"""Plot the command-driven V3 mirror and write the integrated Chinese report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_rmp_three_way_v3_mpl")

import numpy as np

import analyze_drag_ball_sim2real as drag
import analyze_valid_juggle_sim2real as base
import plot_drag_ball_rmp_three_way as plots


def _pct(change: float) -> str:
    return f"{100.0 * (-float(change)):.1f}%"


def _write_report(path: Path, result: dict[str, Any]) -> None:
    metrics = result["three_way_metrics"]
    same = result["same_command_comparison"]
    candidate = result["selected_candidate"]
    contact = result["contact_parameters"]
    lines = [
        "# V91 Sim2Real 近 1:1 数字镜像 V3 分析",
        "",
        "## 结论",
        "",
        "V3 推荐使用 **command-driven sim mirror**：实物策略产生的同一组 `q_ref` 一路继续只发送给真实机器人，另一路只读复制给 sim mirror。mirror 使用辨识后的 recovered-RMP/执行器参数推进 MuJoCo，不向实物控制链回写任何量。这样既保留独立执行器和接触物理，又消除两个 actor 因微小状态差异而产生的闭环分岔。",
        "",
        "三轨迹图片中的绿色曲线是该 command-driven V3 mirror；橙色仍是记录中的原始独立 sim mirror，蓝色是实物。所有曲线均为无 marker 的细实线。",
        "",
        "## 三轨迹误差",
        "",
        "| 时间窗 | 信号 | 原始 sim mirror | V3 command-driven mirror | 改善 |",
        "|---|---|---:|---:|---:|",
    ]
    units = {
        "q": "rad", "dq": "rad/s", "racket_position": "m",
        "racket_velocity": "m/s",
    }
    labels = {
        "q": "关节角", "dq": "关节角速度",
        "racket_position": "球拍位置", "racket_velocity": "球拍速度",
    }
    for window, values in metrics.items():
        for name, item in values.items():
            lines.append(
                f"| {window} | {labels[name]} | "
                f"{item['before_rms']:.6f} {units[name]} | "
                f"{item['after_rms']:.6f} {units[name]} | "
                f"{_pct(item['relative_change'])} |")
    lines.extend([
        "",
        "## V2 → V3 固定同一 q_ref 识别结果",
        "",
        "| 时间窗 | 指标 | V2 | V3 | 改善 |",
        "|---|---|---:|---:|---:|",
    ])
    for window in ("post_warmup", "held_out_14_s_to_end"):
        v2 = same["v2"][window]
        v3 = same["v3"][window]
        for key, label, unit in (
            ("q_rms_rad", "关节角", "rad"),
            ("dq_rms_rad_s", "关节角速度", "rad/s"),
        ):
            improvement = 1.0 - float(v3[key]) / float(v2[key])
            lines.append(
                f"| {window} | {label} | {v2[key]:.6f} {unit} | "
                f"{v3[key]:.6f} {unit} | {100.0 * improvement:.1f}% |")
    lines.extend([
        "",
        "V3 相对 V2 的数值收益较小但在 14 s+ 留出段仍同向，说明 V2 已接近当前 RMP 结构的执行器辨识极限。三轨迹中的大幅改善主要来自 command-driven 架构消除了独立 actor 状态分岔，而不是把该收益错误归功于几个新增参数。",
        "",
        "## V3 mirror-only 执行器参数",
        "",
        f"- output delay J1–J7：`{candidate['output_delay_steps_per_joint']} ms`；允许 1 ms 相邻 RMP 输出间线性插值。",
        f"- MuJoCo position actuator Kp scale：`{candidate['actuator_kp_scale_per_joint']}`。",
        f"- MuJoCo position actuator Kv scale：`{candidate['actuator_kv_scale_per_joint']}`。",
        f"- RMP joint acceleration feedforward：`{candidate['joint_acceleration_feedforward']}`。",
        f"- J4 acceleration limit scale：`{candidate['acceleration_limit_scale_per_joint'][3]}`。",
        f"- low-level qdd feedforward：`{candidate['acceleration_feedforward_s2_per_joint']}`。",
        f"- dof damping relative scale：`{candidate['dof_damping_scale_per_joint']}`。",
        f"- dof armature relative scale：`{candidate['dof_armature_scale_per_joint']}`。",
        "",
        "这些值只能作为 sim mirror override，不能写回 V91 checkpoint 或实物共享 `env_cfg`。实物接口仍只发布七维 `jnt_pos=q_ref`。",
        "",
        "## 球拍接触层方案",
        "",
        "Drag-ball 记录没有自由球—球拍冲击，不能用来再次拟合接触参数。接触层直接采用独立的 record_new3/4 碰撞辨识结果，并保留 record_new5 为留出验证：",
        "",
        f"- ball mass：`{contact['ball_mass_kg']} kg`；radius：`{contact['ball_radius_m']} m`。",
        f"- pair friction：`{contact['pair_sliding_friction']}`。",
        f"- solref：`{contact['solref_time_s']} {contact['solref_damping_ratio']:.6f}`。",
        f"- solimp：`{contact['solimp']}`；margin：`{contact['margin_m']} m`；condim：`{contact['condim']}`。",
        f"- 训练随机化建议：mass `{contact['suggested_training_dr_ball_mass_range_kg']} kg`，solref damping `{contact['suggested_training_dr_solref_damping_range']}`。",
        "",
        "由于记录没有球自旋观测，胶皮摩擦与入射自旋仍不可分辨；MJX 与 CPU MuJoCo 的高速偏心接触也不是逐步完全等价。因此接触层不能宣称数学上的完全 1:1。",
        "",
        "## 推荐运行结构",
        "",
        "```text",
        "camera / joint_states",
        "          │",
        "          ▼",
        "      real V91 actor",
        "          │ q_ref",
        "          ├────────► 实物控制器（只收关节角）",
        "          │",
        "          └────────► command-driven sim mirror",
        "                         ├─ V3 recovered-RMP/actuator",
        "                         └─ 已辨识 ball-racket contact",
        "```",
        "",
        "自由运行的独立 actor mirror 仍可作为策略稳定性诊断模式，并继续使用 V2 闭环参数；不能把 V3 的执行器最优参数直接用于该模式。本次复核中 V3 参数的自由闭环表现变差，验证了两个模式必须分开。",
        "",
        "## 泛化与上线边界",
        "",
        "- 14 s+ 是同一次记录内的严格时间留出段，不等价于跨日期、跨温度和跨负载验证。",
        "- V3 相对 V2 只改善约 1%，部分 Kp/Kv、damping、armature 比例落在本轮搜索边界；它们应视为离线候选，而不是已经唯一辨识的实物常数。",
        "- 在至少两条新的 drag-ball 记录上复核前，不应自动替换在线 mirror 的默认参数。即使启用，也必须保持 `sim_mirror_only` 和禁止回写实物控制。",
        "- 图中的约 95% 总改善包含 command-driven 架构收益；参数本身的独立增益以 V2→V3 表为准。",
        "- 本轮已实现并测试 mirror 内部的分数毫秒延迟/Kp/Kv 参数能力，但在线 probe 尚未自动加载候选 JSON，也尚未把实物 `q_ref` 分叉接入 mirror；当前结果是确定性离线 replay 验证，不会改变现有部署。",
        "",
        "## 图片与产物",
        "",
        "- `01–03`：关节角、角速度、角加速度。",
        "- `04`：共享 q_ref/action 与原始独立 mirror actor 输出。",
        "- `05–10`：球拍位置、速度、加速度、Euler 角、角速度、角加速度。",
        "- `11`：原始 mirror 与 V3 command-driven mirror 的误差时间序列。",
        "- `proposed_sim_mirror_rmp_override_v3.json`：仅供 mirror 使用的执行器参数。",
        "- `sim_mirror_digital_twin_v3.json`：执行器、接触和架构的组合方案。",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording = args.recording.expanduser().resolve()
    tuning_dir = (
        args.tuning_dir.expanduser().resolve()
        if args.tuning_dir is not None
        else recording / "analysis" / "v91_rmp_tuning_drag_ball_v3")
    output = tuning_dir / "three_way_trajectory_comparison"
    output.mkdir(parents=True, exist_ok=True)

    tuning = json.loads((
        tuning_dir / "rmp_tuning_v3_results.json").read_text(encoding="utf-8"))
    replay = np.load(tuning_dir / "same_command_replay_v3.npz")
    contact_path = base.ROOT / (
        "analysis_results/ball_racket_contact_identification/"
        "recommended_contact_params.json")
    contact = json.loads(contact_path.read_text(encoding="utf-8"))

    stage, aligned, _excluded = base.load_alignment(recording)
    controller = base.init_kinematics(base.DEFAULT_CHECKPOINT, base.DEFAULT_XML)
    base.populate_kinematics(stage, aligned, controller)
    elapsed = np.asarray([item["elapsed"] for item in aligned])
    if len(replay["selected_q"]) != len(elapsed):
        raise RuntimeError("V3 replay length does not match aligned recording")

    real = {
        "q": base.aligned_matrix(aligned, "real_q"),
        "dq": base.aligned_matrix(aligned, "real_dq"),
        "qcmd": base.aligned_matrix(aligned, "real_qcmd"),
        "action": base.aligned_matrix(aligned, "real_action"),
    }
    baseline = {
        "q": base.aligned_matrix(aligned, "sim_q"),
        "dq": base.aligned_matrix(aligned, "sim_dq"),
        "qcmd": base.aligned_matrix(aligned, "sim_qcmd"),
        "action": base.aligned_matrix(aligned, "sim_action"),
    }
    tuned = {
        "q": np.asarray(replay["selected_q"], dtype=np.float64),
        "dq": np.asarray(replay["selected_dq"], dtype=np.float64),
        # The recommended mirror consumes the exact published command.  These
        # are shared inputs, not a second actor prediction.
        "qcmd": np.asarray(replay["command_q_ref"], dtype=np.float64),
        "action": real["action"].copy(),
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
    baseline_kin = {
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
    baseline_ddq = base.aligned_matrix(aligned, "sim_ddq")
    tuned_ddq = drag._uniform_difference(tuned["dq"])
    joints = [f"J{index}" for index in range(1, 8)]
    axes = ["x", "y", "z"]
    ball = np.asarray([
        base.vector((item["sim"].get("sync") or {}).get("inputBall"), 3)
        for item in aligned], dtype=np.float64)

    plots._plot_family(
        output / "01_joint_position.png", elapsed,
        real["q"], baseline["q"], tuned["q"],
        "Joint position: real / original mirror / command-driven V3 mirror",
        "rad", joints, tuned_label="V3 command-driven sim mirror")
    plots._plot_family(
        output / "02_joint_velocity.png", elapsed,
        real["dq"], baseline["dq"], tuned["dq"],
        "Joint velocity: real / original mirror / command-driven V3 mirror",
        "rad/s", joints, tuned_label="V3 command-driven sim mirror")
    plots._plot_family(
        output / "03_joint_acceleration.png", elapsed,
        real_ddq, baseline_ddq, tuned_ddq,
        "Joint acceleration: real / original mirror / command-driven V3 mirror",
        "rad/s²", joints, tuned_label="V3 command-driven sim mirror")
    plots._plot_policy_outputs(
        output / "04_policy_command_and_action.png", elapsed,
        real["qcmd"], baseline["qcmd"], tuned["qcmd"],
        real["action"], baseline["action"], tuned["action"],
        tuned_label="shared real q_ref/action",
        title="Policy output: real/shared command vs original mirror actor")
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
            real_kin[key], baseline_kin[key], tuned_kin[key],
            f"{title}: real / original mirror / command-driven V3 mirror",
            unit, axes, ball=ball if key == "position" else None,
            tuned_label="V3 command-driven sim mirror")
    plots._plot_gap_summary(
        output / "11_gap_summary.png", elapsed, [
            ("joint q RMS [rad]", real["q"], baseline["q"], tuned["q"]),
            ("joint dq RMS [rad/s]", real["dq"], baseline["dq"], tuned["dq"]),
            ("racket position RMS [m]", real_kin["position"], baseline_kin["position"], tuned_kin["position"]),
            ("racket velocity RMS [m/s]", real_kin["velocity"], baseline_kin["velocity"], tuned_kin["velocity"]),
        ], after_label="V3 command-driven mirror",
        title="Sim2Real gap: original vs command-driven V3 mirror")

    metric_real = {
        "q": real["q"], "dq": real["dq"],
        "racket_position": real_kin["position"],
        "racket_velocity": real_kin["velocity"],
    }
    metric_baseline = {
        "q": baseline["q"], "dq": baseline["dq"],
        "racket_position": baseline_kin["position"],
        "racket_velocity": baseline_kin["velocity"],
    }
    metric_tuned = {
        "q": tuned["q"], "dq": tuned["dq"],
        "racket_position": tuned_kin["position"],
        "racket_velocity": tuned_kin["velocity"],
    }
    result = {
        "recording": recording.name,
        "comparison_contract": {
            "real": "recorded hardware q/dq",
            "before": "recorded independent actor sim mirror",
            "after": "V3 plant driven by the real published q_ref",
            "selection": "train 1-9s + validation 9-14s; 14s+ held out",
            "line_style": "thin solid lines without markers",
        },
        "selected_candidate": tuning["selected_candidate"],
        "three_way_metrics": plots._metric_windows(
            elapsed, metric_real, metric_baseline, metric_tuned),
        "same_command_comparison": {
            "v2": tuning["v2_same_command_metrics"],
            "v3": tuning["selected_same_command_metrics"],
        },
        "free_closed_loop_v3_metrics": tuning[
            "selected_free_closed_loop_metrics"],
        "contact_parameters": contact,
    }
    (output / "three_way_comparison_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    combined = {
        "scope": "sim_mirror_only",
        "drive_mode": "real_published_q_ref",
        "no_feedback_to_real_control": True,
        "real_command_interface": "joint_position_q_ref_only",
        "actuator_rmp": tuning["proposed_mirror_env_cfg_override"],
        "ball_racket_contact": contact,
    }
    (tuning_dir / "sim_mirror_digital_twin_v3.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    report = tuning_dir / "SIM2REAL_1TO1_MIRROR_V3_ANALYSIS_ZH.md"
    _write_report(report, result)
    print(json.dumps({
        "output": str(output),
        "report": str(report),
        "metrics": result["three_way_metrics"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
