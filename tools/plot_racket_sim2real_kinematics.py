#!/usr/bin/env python3
"""Plot source-sequence-aligned real/sim racket kinematics during juggling.

The default recording is a long, fully aligned ``record_new4`` episode.  It
intentionally has the policy dq adapter configured, so the script keeps the
measured side of all velocity-derived curves on the raw ``ros.jointVelocity``
field instead of ``policyObs[7:14]``.

Six PNG files are written, each with x/y/z subplots and real/simulation
curves.  Detected physical contacts are marked on both traces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = ROOT / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
CHECKPOINT = ROOT / (
    "simcode5/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/"
    "goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl"
)
XML = ROOT / "simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
DEFAULT_RUN = "record_new4/probe_recording_20260811_203653"
DEFAULT_OUTPUT = ROOT / "analysis_results/racket_kinematics_sim2real_20260811_203653"

COLORS = {
    "real": "#1769aa", "sim": "#e76f51", "raw_ball": "#16857b",
    "processed_ball": "#8a5ea7", "impact": "#6a4c93",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def vector(value: Any, n: int) -> np.ndarray | None:
    try:
        out = np.asarray(value, dtype=np.float64).reshape(-1)[:n]
    except (TypeError, ValueError):
        return None
    return out if out.size == n and np.all(np.isfinite(out)) else None


def source_seq_real(row: dict[str, Any]) -> int | None:
    try:
        seq = int(row.get("sourceSeq", row.get("tick")))
    except (TypeError, ValueError):
        return None
    return seq if seq > 0 else None


def source_seq_sim(row: dict[str, Any]) -> int | None:
    try:
        seq = int((row.get("sync") or {}).get("sourceSeq"))
    except (TypeError, ValueError):
        return None
    return seq if seq > 0 else None


def mirror_contract_error(sim: dict[str, Any]) -> str | None:
    """Reject mirror rows that explicitly declare an invalid plant/stream."""
    if sim.get("backend") != "mujoco_actuator_mirror":
        return "wrong_backend"
    if not bool(sim.get("available", True)):
        return "unavailable"
    if not bool(sim.get("active", True)):
        return "inactive"
    if not bool(sim.get("valid", False)):
        return "invalid_state"
    sync = sim.get("sync") or {}
    if sync.get("streamContinuous") is False:
        return "noncontinuous_stream"
    if sync.get("alignmentValid") is False:
        return "invalid_alignment"
    plant = sim.get("plantContract")
    if isinstance(plant, dict):
        declared = [plant.get("checkpointPlantSettingsAligned"),
                    plant.get("simcode5SettingsAligned")]
        if any(value is False for value in declared):
            return "plant_settings_misaligned"
        if not any(value is True for value in declared):
            return "plant_alignment_not_declared"
        if abs(float(plant.get("controlDtS", 0.005)) - 0.005) > 1e-9:
            return "wrong_control_dt"
        if str(plant.get("actuatorCmdModel", "second_order")) != "second_order":
            return "wrong_actuator_model"
    return None


def inputs_match(real: dict[str, Any], sim: dict[str, Any]) -> bool:
    """Require the recorded policy input envelope and profile to match."""
    policy_input = real.get("policyInput")
    sync = sim.get("sync")
    if not isinstance(policy_input, dict) or not isinstance(sync, dict):
        return False
    try:
        runtime = real.get("runtime") or {}
        real_profile = str(runtime.get(
            "activePolicyProfile", runtime.get("policyProfile", "")))
        sim_profile = str(sync.get("policyProfile", sim.get("policyProfile", "")))
        same = (
            np.array_equal(np.asarray(policy_input["ball"], np.float32),
                           np.asarray(sync["inputBall"], np.float32))
            and np.array_equal(np.asarray(policy_input["velocity"], np.float32),
                               np.asarray(sync["inputVelocity"], np.float32))
            and bool(policy_input.get("valid")) == bool(sync.get("inputValid"))
            and int(policy_input.get("ballObservationSeq", 0) or 0)
            == int(sync.get("ballObservationSeq", 0) or 0)
            and real_profile == sim_profile
            and int(runtime.get("policyGeneration", 0) or 0)
            == int(sync.get("policyGeneration", 0) or 0)
        )
        real_ball = real.get("policyBallObservation")
        sim_ball = sim.get("policyBallObservation")
        if real_ball is not None or sim_ball is not None:
            same = same and np.array_equal(
                np.asarray(real_ball, np.float32), np.asarray(sim_ball, np.float32))
        return bool(same)
    except (KeyError, TypeError, ValueError):
        return False


def detect_contacts(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Detect physical impacts from a ball vz reversal and racket proximity."""
    refresh: list[dict[str, Any]] = []
    last_observation_seq: int | None = None
    for row in rows:
        observation_seq = int(row.get("ballObservationSeq", 0) or 0)
        if observation_seq > 0 and observation_seq != last_observation_seq:
            refresh.append(row)
            last_observation_seq = observation_seq

    contacts: list[dict[str, float]] = []
    last_time = -math.inf
    for index in range(1, len(refresh)):
        after = refresh[index]
        velocity_after = vector(after.get("measuredVelocity"), 3)
        if velocity_after is None or velocity_after[2] <= 0.10:
            continue
        recent = refresh[max(0, index - 5):index]
        candidates = [(row, vector(row.get("measuredVelocity"), 3))
                      for row in recent]
        candidates = [(row, velocity) for row, velocity in candidates
                      if velocity is not None]
        if not candidates:
            continue
        before, velocity_before = min(candidates, key=lambda item: item[1][2])
        if velocity_before[2] >= -0.10:
            continue
        detected_time = float(after.get("wallTimeS", 0.0) or 0.0)
        if detected_time - last_time < 0.18:
            continue
        ball_after, racket_after = (vector(after.get("measuredBall"), 3),
                                    vector(after.get("racket"), 3))
        if ball_after is None or racket_after is None:
            continue
        relative = ball_after - racket_after
        geometry_ok = (np.linalg.norm(relative[:2]) <= 0.085
                       and -0.025 <= relative[2] <= 0.105)
        ball_before, racket_before = (vector(before.get("measuredBall"), 3),
                                      vector(before.get("racket"), 3))
        held_ok = (
            ball_before is not None and racket_before is not None
            and detected_time - float(before.get("wallTimeS", detected_time)) <= 0.09
            and np.linalg.norm(ball_before - racket_before) <= 0.10
        )
        if not (geometry_ok or held_ok):
            continue
        nearby = [row for row in rows if detected_time - 0.085
                  <= float(row.get("wallTimeS", 0.0) or 0.0)
                  <= detected_time + 0.005]
        physical_time = detected_time
        if nearby:
            def ball_racket_distance(row: dict[str, Any]) -> float:
                ball, racket = vector(row.get("measuredBall"), 3), vector(row.get("racket"), 3)
                return float(np.linalg.norm(ball - racket)) if ball is not None and racket is not None else math.inf
            physical_time = float(min(nearby, key=ball_racket_distance).get(
                "wallTimeS", detected_time))
        contacts.append({"detected_t": detected_time, "physical_t": physical_time})
        last_time = detected_time
    return contacts


def load_aligned(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, float]], Counter[str]]:
    """Read one recording and retain only source-sequence-valid sim pairs."""
    control_path = run_dir / "control_samples.jsonl"
    sim_path = run_dir / "sim_control_samples.jsonl"
    if not control_path.exists() or not sim_path.exists():
        raise FileNotFoundError("control_samples.jsonl or sim_control_samples.jsonl is missing")

    real_rows = [row for row in read_jsonl(control_path)
                 if bool(row.get("policyExecuted")) and row.get("mode") == "policy"]
    sim_by_seq: dict[int, dict[str, Any]] = {}
    for sim in read_jsonl(sim_path):
        seq = source_seq_sim(sim)
        if seq is not None and seq not in sim_by_seq:
            sim_by_seq[seq] = sim

    excluded: Counter[str] = Counter()
    aligned: list[dict[str, Any]] = []
    for real in real_rows:
        seq = source_seq_real(real)
        sim = sim_by_seq.get(seq) if seq is not None else None
        if sim is None:
            excluded["no_source_seq_pair"] += 1
            continue
        reason = mirror_contract_error(sim)
        if reason is not None:
            excluded[reason] += 1
            continue
        if not inputs_match(real, sim):
            excluded["policy_input_or_cohort_mismatch"] += 1
            continue

        ros = real.get("ros") or {}
        real_q = vector(ros.get("jointPosition"), 7)
        # This is deliberately the raw ROS measurement.  policyObs[7:14]
        # can be the dq-adapter output in record_new4/5 and is never used.
        real_dq = vector(ros.get("jointVelocity"), 7)
        sim_q = vector(sim.get("jointPosition"), 7)
        sim_dq = vector(sim.get("jointVelocity"), 7)
        # rawBall is the pre-control source value.  observedBall is the
        # post-processing state consumed by the policy and submitted to the
        # actuator mirror through policyInput.
        raw_ball = vector(real.get("rawBall"), 3)
        processed_ball = vector(real.get("observedBall"), 3)
        if (real_q is None or real_dq is None or sim_q is None or sim_dq is None
                or raw_ball is None or processed_ball is None):
            excluded["missing_joint_q_dq_or_ball"] += 1
            continue
        try:
            source_time = float(real.get("time"))
            wall_time = float(real.get("wallTimeS"))
        except (TypeError, ValueError):
            excluded["missing_time"] += 1
            continue
        if not (math.isfinite(source_time) and math.isfinite(wall_time)):
            excluded["invalid_time"] += 1
            continue
        aligned.append({
            "source_seq": seq,
            "source_time_s": source_time,
            "wall_time_s": wall_time,
            "real_q": real_q,
            "real_dq_raw_ros": real_dq,
            "real_dq_source": str(ros.get("jointVelocitySource", "")),
            "raw_ball": raw_ball,
            "sim_q": sim_q,
            "sim_dq": sim_dq,
            "processed_ball": processed_ball,
            "dq_adapter": real.get("policyDqAdapter", {}),
        })

    if not aligned:
        raise RuntimeError("No source-sequence-aligned samples survived validation")
    return aligned, detect_contacts(real_rows), excluded


def calculate_kinematics(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Compute base-frame pose/Jacobian twist from actual and mirror joints."""
    sys.path.insert(0, str(ROOT / "src/pingpong_moz1"))
    from pingpong_controller.mjx_rl_policy import Simcode5PureActuator67PolicyController

    controller = Simcode5PureActuator67PolicyController(
        str(CHECKPOINT), robot_xml_path=str(XML))
    values: dict[str, list[np.ndarray | float | int | str]] = {
        "source_seq": [], "source_time_s": [], "wall_time_s": [],
        "real_position": [], "sim_position": [],
        "raw_ball_position": [], "processed_ball_position": [],
        "real_quaternion": [], "sim_quaternion": [],
        "real_velocity_raw": [], "sim_velocity_raw": [],
        "real_angular_velocity_raw": [], "sim_angular_velocity_raw": [],
        "real_dq_source": [],
    }
    for row in rows:
        real_pose = controller._compute_racket_pose_jacobians_base(row["real_q"])
        sim_pose = controller._compute_racket_pose_jacobians_base(row["sim_q"])
        real_position, real_quat, real_jp, real_jr, real_ok = real_pose
        sim_position, sim_quat, sim_jp, sim_jr, sim_ok = sim_pose
        if not real_ok or not sim_ok:
            raise RuntimeError("MuJoCo FK/Jacobian was unavailable for an aligned sample")
        values["source_seq"].append(int(row["source_seq"]))
        values["source_time_s"].append(float(row["source_time_s"]))
        values["wall_time_s"].append(float(row["wall_time_s"]))
        values["real_position"].append(np.asarray(real_position, dtype=np.float64))
        values["sim_position"].append(np.asarray(sim_position, dtype=np.float64))
        values["raw_ball_position"].append(np.asarray(row["raw_ball"], dtype=np.float64))
        values["processed_ball_position"].append(
            np.asarray(row["processed_ball"], dtype=np.float64))
        values["real_quaternion"].append(np.asarray(real_quat, dtype=np.float64))
        values["sim_quaternion"].append(np.asarray(sim_quat, dtype=np.float64))
        values["real_velocity_raw"].append(np.asarray(real_jp, dtype=np.float64) @ row["real_dq_raw_ros"])
        values["sim_velocity_raw"].append(np.asarray(sim_jp, dtype=np.float64) @ row["sim_dq"])
        values["real_angular_velocity_raw"].append(np.asarray(real_jr, dtype=np.float64) @ row["real_dq_raw_ros"])
        values["sim_angular_velocity_raw"].append(np.asarray(sim_jr, dtype=np.float64) @ row["sim_dq"])
        values["real_dq_source"].append(str(row["real_dq_source"]))

    result: dict[str, np.ndarray] = {}
    for name, value in values.items():
        if name == "real_dq_source":
            result[name] = np.asarray(value, dtype=object)
        else:
            result[name] = np.asarray(value, dtype=np.float64)

    for side in ("real", "sim"):
        euler = Rotation.from_quat(result[f"{side}_quaternion"]).as_euler(
            "xyz", degrees=True)
        result[f"{side}_angle"] = np.rad2deg(np.unwrap(np.deg2rad(euler), axis=0))
    return result


def smoothing_window(n_samples: int) -> int:
    """About 75 ms at 200 Hz; always valid for cubic Savitzky--Golay."""
    window = min(15, n_samples if n_samples % 2 else n_samples - 1)
    return max(5, window) if n_samples >= 5 else 0


def continuous_slices(time_s: np.ndarray, nominal_dt: float) -> list[slice]:
    """Split only at genuine recording gaps before applying fixed-rate filters."""
    boundaries = np.flatnonzero(np.diff(time_s) > 1.5 * nominal_dt) + 1
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, len(time_s)]
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]


def add_smoothed_derivatives(data: dict[str, np.ndarray]) -> tuple[float, int, int]:
    time_s = data["source_time_s"]
    dt = float(np.median(np.diff(time_s)))
    if not math.isfinite(dt) or dt <= 0.0:
        raise RuntimeError("Aligned source time is not strictly increasing")
    slices = continuous_slices(time_s, dt)
    largest_window = 0
    for key in ("velocity", "angular_velocity"):
        for side in ("real", "sim"):
            raw = data[f"{side}_{key}_raw"]
            acc_key = "acceleration" if key == "velocity" else "angular_acceleration"
            filtered = np.empty_like(raw)
            derivative = np.empty_like(raw)
            for segment in slices:
                values = raw[segment]
                segment_time = time_s[segment]
                window = smoothing_window(len(values))
                if window:
                    filtered[segment] = savgol_filter(
                        values, window, 3, axis=0, mode="interp")
                    derivative[segment] = savgol_filter(
                        values, window, 3, deriv=1, delta=dt, axis=0, mode="interp")
                    largest_window = max(largest_window, window)
                elif len(values) >= 2:
                    filtered[segment] = values
                    derivative[segment] = np.gradient(
                        values, segment_time, axis=0, edge_order=1)
                else:
                    filtered[segment] = values
                    derivative[segment] = 0.0
            data[f"{side}_{key}"] = filtered
            data[f"{side}_{acc_key}"] = derivative
    return dt, largest_window, len(slices)


def crop_to_juggling_phase(data: dict[str, np.ndarray], contacts: list[dict[str, float]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Keep the contact-to-contact phase with a short lead-in/out for context."""
    wall_time = data["wall_time_s"]
    contact_indices: list[int] = []
    for contact in contacts:
        index = int(np.argmin(np.abs(wall_time - contact["physical_t"])))
        if abs(float(wall_time[index] - contact["physical_t"])) <= 0.035:
            contact_indices.append(index)
    contact_indices = sorted(set(contact_indices))
    if not contact_indices:
        raise RuntimeError("No detected physical contact has an aligned racket sample")
    first, last = contact_indices[0], contact_indices[-1]
    lower = wall_time[first] - 0.25
    upper = wall_time[last] + 0.35
    mask = (wall_time >= lower) & (wall_time <= upper)
    kept_indices = np.flatnonzero(mask)
    old_to_new = {int(old): new for new, old in enumerate(kept_indices)}
    cropped = {key: (value[mask] if value.shape[0] == mask.shape[0] else value)
               for key, value in data.items()}
    cropped["time_s"] = cropped["source_time_s"] - cropped["source_time_s"][0]
    cropped_contacts = np.asarray([old_to_new[index] for index in contact_indices
                                   if index in old_to_new], dtype=np.int64)
    return cropped, cropped_contacts


def rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=0))


def metrics(data: dict[str, np.ndarray], impact_indices: np.ndarray) -> dict[str, dict[str, list[float]]]:
    output: dict[str, dict[str, list[float]]] = {}
    for name in ("position", "velocity", "acceleration", "angle",
                 "angular_velocity", "angular_acceleration"):
        error = data[f"real_{name}"] - data[f"sim_{name}"]
        output[name] = {
            "rms_gap_xyz": [float(value) for value in rms(error)],
            "mean_gap_xyz": [float(value) for value in np.mean(error, axis=0)],
            "impact_mae_xyz": [float(value) for value in np.mean(
                np.abs(error[impact_indices]), axis=0)],
        }
    return output


def write_csv(path: Path, data: dict[str, np.ndarray]) -> None:
    vector_fields = (
        "position", "velocity", "acceleration", "angle",
        "angular_velocity", "angular_acceleration",
    )
    header = ["time_s", "source_seq", "real_joint_velocity_source"]
    for field in vector_fields:
        for side in ("real", "sim"):
            header.extend(f"{side}_{field}_{axis}" for axis in "xyz")
    header.extend(f"raw_ball_position_{axis}" for axis in "xyz")
    header.extend(f"processed_ball_position_{axis}" for axis in "xyz")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index in range(len(data["time_s"])):
            row: list[Any] = [
                f"{data['time_s'][index]:.6f}",
                int(data["source_seq"][index]),
                data["real_dq_source"][index],
            ]
            for field in vector_fields:
                for side in ("real", "sim"):
                    row.extend(f"{value:.9g}" for value in data[f"{side}_{field}"][index])
            row.extend(f"{value:.9g}" for value in data["raw_ball_position"][index])
            row.extend(
                f"{value:.9g}" for value in data["processed_ball_position"][index])
            writer.writerow(row)


def plot_family(
    output: Path,
    title: str,
    field: str,
    unit: str,
    data: dict[str, np.ndarray],
    impact_indices: np.ndarray,
    run_label: str,
) -> None:
    axis_names = ("x", "y", "z")
    time_s = data["time_s"]
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    for dimension, axis in enumerate(axes):
        axis.plot(time_s, data[f"real_{field}"][:, dimension], color=COLORS["real"],
                  linewidth=1.65, label="Real racket" if dimension == 0 else None)
        axis.plot(time_s, data[f"sim_{field}"][:, dimension], color=COLORS["sim"],
                  linewidth=1.45, linestyle="--", label="Actuator mirror" if dimension == 0 else None)
        if field == "position":
            axis.plot(time_s, data["raw_ball_position"][:, dimension],
                      color=COLORS["raw_ball"], linewidth=1.35,
                      label="Raw ball measurement" if dimension == 0 else None)
            axis.plot(time_s, data["processed_ball_position"][:, dimension],
                      color=COLORS["processed_ball"], linewidth=1.25, linestyle=":",
                      label="Processed control ball" if dimension == 0 else None)
        for impact in impact_indices:
            axis.axvline(time_s[impact], color=COLORS["impact"], alpha=0.22,
                        linewidth=0.9, zorder=0)
        axis.scatter(time_s[impact_indices], data[f"real_{field}"][impact_indices, dimension],
                     s=32, color=COLORS["real"], edgecolor="white", linewidth=0.55,
                     zorder=4, label="Detected impact" if dimension == 0 else None)
        axis.scatter(time_s[impact_indices], data[f"sim_{field}"][impact_indices, dimension],
                     s=48, color=COLORS["sim"], marker="X", edgecolor="white", linewidth=0.45,
                     zorder=4)
        if field == "position":
            axis.scatter(time_s[impact_indices], data["raw_ball_position"][impact_indices, dimension],
                         s=25, color=COLORS["raw_ball"], marker="^", edgecolor="white", linewidth=0.45,
                         zorder=4, label="Ball/racket samples at impact" if dimension == 0 else None)
            axis.scatter(time_s[impact_indices], data["processed_ball_position"][impact_indices, dimension],
                         s=36, color=COLORS["processed_ball"], marker="P", edgecolor="white", linewidth=0.4,
                         zorder=4)
        axis.set_ylabel(f"{axis_names[dimension]} [{unit}]")
        axis.grid(alpha=0.25, linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=True)
    axes[-1].set_xlabel("Time in valid juggling phase [s]")
    figure.suptitle(f"{title}: real vs simulation — {run_label}", fontsize=14, fontweight="bold")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_report(
    output: Path,
    run: str,
    data: dict[str, np.ndarray],
    impact_indices: np.ndarray,
    dt_s: float,
    window: int,
    excluded: Counter[str],
    statistics: dict[str, dict[str, list[float]]],
) -> None:
    adapter_samples = [row for row in data.get("dq_adapter", []) if isinstance(row, dict)]
    del adapter_samples  # Data is retained in JSON metadata rather than in the cropped arrays.
    sources = sorted(set(str(value) for value in data["real_dq_source"] if str(value)))
    line_items = []
    for name, metric in statistics.items():
        line_items.append(
            f"| {name} | {metric['rms_gap_xyz'][0]:.5g}, {metric['rms_gap_xyz'][1]:.5g}, {metric['rms_gap_xyz'][2]:.5g} | "
            f"{metric['impact_mae_xyz'][0]:.5g}, {metric['impact_mae_xyz'][1]:.5g}, {metric['impact_mae_xyz'][2]:.5g} |")
    report = f"""# 实物/仿真球拍运动学曲线

## 数据范围

- 记录：`{run}`。
- 有效颠球阶段：{data['time_s'][0]:.3f}--{data['time_s'][-1]:.3f} s（相对裁剪起点），共 {len(data['time_s'])} 个严格 `sourceSeq` 对齐样本。
- 触球标记：{len(impact_indices)} 个。检测规则是实测球的 `vz` 由下落转上升，并通过球拍邻近几何门限；紫色竖线、蓝色圆点和橙色叉号指向同一个实物检测触球时刻。
- 实物关节速度：只使用原始 `ros.jointVelocity`（来源：{', '.join(sources) or '未声明'}），没有使用 `policyObs[7:14]`。因此 `record_new4/5` 的 `dq_adapter` 不会污染末端速度、加速度、角速度或角加速度。
- 位置图中的球：青绿实线是处理前的 `rawBall`；紫色点线是处理后的 `observedBall`，也就是 `policyInput.ball`。严格对齐时该处理后状态已原样送入 mirror，因此没有重复画一条独立的“mirror ball”。
- 速度/角速度：MuJoCo 同一球拍 site 的基座坐标 Jacobian 乘以各自关节速度；实物用 ROS 原始速度，仿真用 mirror `jointVelocity`。
- 加速度/角加速度：对上述原始末端速度做最多 {window} 点三阶 Savitzky--Golay 微分（中位采样间隔 {dt_s * 1e3:.3f} ms）；若记录中有控制时间缺口，会在缺口处分段而不跨段滤波。位置与姿态未作滤波。
- 角度：球拍 site 相对基座坐标系的 XYZ Euler 角（度），已逐轴 unwrap；角速度/角加速度是基座 XYZ 分量，而不是 Euler 角的逐轴数值微分。

## 图像

- `racket_position.png`（同时叠加处理前 `rawBall` 与处理后 `observedBall` 的位置）
- `racket_velocity.png`
- `racket_acceleration.png`
- `racket_angle.png`
- `racket_angular_velocity.png`
- `racket_angular_acceleration.png`

## 曲线差距（x, y, z）

单位依次为 m、m/s、m/s²、deg、rad/s、rad/s²。`RMS` 为整个有效颠球阶段，`触球 MAE` 仅在突出标记的时刻计算。

| 曲线 | RMS 差距 (x, y, z) | 触球 MAE (x, y, z) |
| --- | --- | --- |
{chr(10).join(line_items)}

## 对齐与过滤

- 同时要求 `sourceSeq`、策略 profile/generation、球位置/速度/有效性、观测序号完全一致。
- mirror 还要求 actuator-mirror backend、有效状态、连续流、正确的 5 ms controlDt 和二阶 actuator model。
- 记录中的排除计数：`{json.dumps(dict(excluded), ensure_ascii=False)}`。

数值明细见 `aligned_racket_kinematics.csv`，机器可读元数据见 `summary.json`。
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="recording relative to interactive_recordings")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = (RECORD_ROOT / args.run).resolve()
    if RECORD_ROOT not in run_dir.parents:
        raise ValueError("--run must stay below interactive_recordings")
    if not CHECKPOINT.exists() or not XML.exists():
        raise FileNotFoundError("Required simcode5 checkpoint or MuJoCo XML is missing")

    aligned, contacts, excluded = load_aligned(run_dir)
    data = calculate_kinematics(aligned)
    dt_s, window, derivative_segments = add_smoothed_derivatives(data)
    data, impact_indices = crop_to_juggling_phase(data, contacts)
    statistics = metrics(data, impact_indices)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_label = f"{args.run} ({len(impact_indices)} detected impacts)"
    plot_specs = (
        ("racket_position.png", "Racket position with raw/processed ball", "position", "m"),
        ("racket_velocity.png", "Racket linear velocity", "velocity", "m/s"),
        ("racket_acceleration.png", "Racket linear acceleration", "acceleration", "m/s²"),
        ("racket_angle.png", "Racket XYZ Euler angle", "angle", "deg"),
        ("racket_angular_velocity.png", "Racket angular velocity", "angular_velocity", "rad/s"),
        ("racket_angular_acceleration.png", "Racket angular acceleration", "angular_acceleration", "rad/s²"),
    )
    for filename, title, field, unit in plot_specs:
        plot_family(output / filename, title, field, unit, data, impact_indices, run_label)
    write_csv(output / "aligned_racket_kinematics.csv", data)

    metadata = {
        "run": args.run,
        "rawRealJointVelocityField": "ros.jointVelocity",
        "policyObsDqUsed": False,
        "ballPositionCurves": {
            "raw": "control_samples.rawBall",
            "processed": "control_samples.observedBall = policyInput.ball = mirror sync input",
        },
        "realJointVelocitySources": sorted(set(str(value) for value in data["real_dq_source"])),
        "alignedSamplesInPlottedPhase": int(len(data["time_s"])),
        "detectedImpactCount": int(len(impact_indices)),
        "sourceTimeMedianDtS": dt_s,
        "savgolWindowSamples": window,
        "savgolPolynomialOrder": 3,
        "derivativeContinuousSegments": derivative_segments,
        "impactSampleTimesS": [float(data["time_s"][index]) for index in impact_indices],
        "excludedSamples": dict(excluded),
        "metrics": statistics,
    }
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, args.run, data, impact_indices, dt_s, window, excluded, statistics)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
