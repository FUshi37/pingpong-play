#!/usr/bin/env python3
"""Four-layer diagnostics for one real ping-pong policy run.

Layers:
  1. perception: ball visibility, estimator age/loss/prediction quality
  2. policy_io: observation/action health and gate-mode timing
  3. task: coarse juggling/contact outcome signals
  4. hardware_safety: joint feedback and command smoothness risk signals
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


JOINTS = [
    "RightArm-0",
    "RightArm-1",
    "RightArm-2",
    "RightArm-3",
    "RightArm-4",
    "RightArm-5",
    "RightArm-6",
]


def read_csv_rows(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def i(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean(values: Iterable[float]) -> float | None:
    vals = finite(values)
    return None if not vals else sum(vals) / len(vals)


def rms(values: Iterable[float]) -> float | None:
    vals = finite(values)
    return None if not vals else math.sqrt(sum(v * v for v in vals) / len(vals))


def percentile(values: Iterable[float], pct: float) -> float | None:
    vals = sorted(finite(values))
    if not vals:
        return None
    k = (len(vals) - 1) * max(0.0, min(100.0, pct)) / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def value_range(values: Iterable[float]) -> dict:
    vals = finite(values)
    if not vals:
        return {"min": None, "max": None}
    return {"min": min(vals), "max": max(vals)}


def duration_from_time(rows: list[dict], key: str, dt_default: float) -> float:
    if not rows:
        return 0.0
    start = f(rows[0], key)
    end = f(rows[-1], key)
    if math.isfinite(start) and math.isfinite(end) and end >= start:
        return float(end - start)
    return max(0.0, (len(rows) - 1) * dt_default)


def max_invalid_streak_s(rows: list[dict], dt: float) -> float:
    streak = 0
    best = 0
    for row in rows:
        if i(row, "ball_valid", 0):
            streak = 0
        else:
            streak += 1
            best = max(best, streak)
    return best * dt


def action_col(row: dict, joint: str, idx: int) -> float:
    if f"action/{joint}" in row:
        return f(row, f"action/{joint}")
    return f(row, f"action/{idx:03d}")


def norm3(x: float, y: float, z: float) -> float:
    if not all(math.isfinite(v) for v in (x, y, z)):
        return float("nan")
    return math.sqrt(x * x + y * y + z * z)


def analyze_perception(
    trajectory: list[dict],
    policy: list[dict],
    estimator: list[dict],
    dt: float,
) -> dict:
    traj_duration = duration_from_time(trajectory, "timestamp", dt)
    traj_dt = [
        f(trajectory[k], "timestamp") - f(trajectory[k - 1], "timestamp")
        for k in range(1, len(trajectory))
    ]
    policy_valid = [i(row, "ball_valid", 0) for row in policy]
    ages = [f(row, "ball_age_s") for row in policy]

    est_valid = [i(row, "valid", 0) for row in estimator]
    est_pred = [i(row, "predicted", 0) for row in estimator]
    est_age = [f(row, "age_s") for row in estimator]
    est_loss = [f(row, "loss_s") for row in estimator]
    est_extrap = [f(row, "extrapolation_s") for row in estimator]
    residuals_mm = []
    for row in estimator:
        raw_age = f(row, "t_now") - f(row, "raw_t_img")
        if not (i(row, "valid", 0)
                and math.isfinite(raw_age)
                and 0.0 <= raw_age <= 0.08):
            continue
        residuals_mm.append(norm3(
            f(row, "x_mm") - f(row, "raw_x_mm"),
            f(row, "y_mm") - f(row, "raw_y_mm"),
            f(row, "z_mm") - f(row, "raw_z_mm"),
        ))
    state_counts = Counter(str(row.get("state", "")) for row in estimator)

    return {
        "trajectory_rows": len(trajectory),
        "trajectory_duration_s": traj_duration,
        "trajectory_effective_hz": (
            len(trajectory) / traj_duration if traj_duration > 1e-9 else None),
        "trajectory_dt_ms": {
            "p50": None if not traj_dt else percentile([v * 1000.0 for v in traj_dt], 50),
            "p95": None if not traj_dt else percentile([v * 1000.0 for v in traj_dt], 95),
        },
        "policy_ball_valid_ratio": (
            sum(policy_valid) / len(policy_valid) if policy_valid else None),
        "policy_ball_max_age_s": percentile(ages, 100),
        "policy_ball_age_p95_s": percentile(ages, 95),
        "policy_ball_max_invalid_streak_s": max_invalid_streak_s(policy, dt),
        "estimator_rows": len(estimator),
        "estimator_valid_ratio": (
            sum(est_valid) / len(est_valid) if est_valid else None),
        "estimator_predicted_ratio": (
            sum(est_pred) / len(est_pred) if est_pred else None),
        "estimator_state_counts": dict(state_counts),
        "estimator_age_s": {
            "mean": mean(est_age),
            "p95": percentile(est_age, 95),
            "max": percentile(est_age, 100),
        },
        "estimator_loss_s": {
            "mean": mean(est_loss),
            "p95": percentile(est_loss, 95),
            "max": percentile(est_loss, 100),
        },
        "estimator_extrapolation_s": {
            "p95": percentile(est_extrap, 95),
            "max": percentile(est_extrap, 100),
        },
        "estimator_vs_latest_raw_residual_mm": {
            "rows": len(finite(residuals_mm)),
            "rms": rms(residuals_mm),
            "p95": percentile(residuals_mm, 95),
            "max": percentile(residuals_mm, 100),
        },
    }


def analyze_policy_io(policy: list[dict], dt: float) -> dict:
    mode_counts = Counter(str(row.get("control_mode", "unknown")) for row in policy)
    action_abs_max = []
    action_delta_norm = []
    prev_action = None
    action_nonfinite = 0
    obs_nonfinite = 0
    obs_cols = [c for c in (policy[0].keys() if policy else []) if c.startswith("obs/")]

    for row in policy:
        obs_vals = [f(row, c) for c in obs_cols]
        if any(not math.isfinite(v) for v in obs_vals):
            obs_nonfinite += 1
        action = [action_col(row, joint, j) for j, joint in enumerate(JOINTS)]
        if any(not math.isfinite(v) for v in action):
            action_nonfinite += 1
            prev_action = None
            continue
        action_abs_max.append(max(abs(v) for v in action))
        if prev_action is not None:
            delta = [a - b for a, b in zip(action, prev_action)]
            action_delta_norm.append(math.sqrt(sum(v * v for v in delta)))
        prev_action = action

    first_policy_t = None
    for row in policy:
        if str(row.get("control_mode", "policy")) == "policy":
            first_policy_t = f(row, "time_sec")
            break

    return {
        "rows": len(policy),
        "duration_s": duration_from_time(policy, "time_sec", dt),
        "control_mode_counts": dict(mode_counts),
        "first_policy_time_s": first_policy_t,
        "obs_nonfinite_rows": obs_nonfinite,
        "action_nonfinite_rows": action_nonfinite,
        "action_abs_max": {
            "mean": mean(action_abs_max),
            "p95": percentile(action_abs_max, 95),
            "max": percentile(action_abs_max, 100),
        },
        "action_delta_norm_per_tick": {
            "rms": rms(action_delta_norm),
            "p95": percentile(action_delta_norm, 95),
            "max": percentile(action_delta_norm, 100),
        },
        "action_delta_norm_per_s": {
            "p95": (
                None if percentile(action_delta_norm, 95) is None
                else percentile(action_delta_norm, 95) / max(dt, 1e-9)),
            "max": (
                None if percentile(action_delta_norm, 100) is None
                else percentile(action_delta_norm, 100) / max(dt, 1e-9)),
        },
    }


def analyze_task(policy: list[dict]) -> dict:
    valid_rows = [row for row in policy if i(row, "ball_valid", 0)]
    z = [f(row, "ball_pos_z_m") for row in valid_rows]
    vz = [f(row, "ball_vel_z_m_s") for row in valid_rows]
    t = [f(row, "time_sec") for row in valid_rows]
    rel_norm = [
        norm3(f(row, "obs/032"), f(row, "obs/033"), f(row, "obs/034"))
        for row in valid_rows
    ]
    contact_like = []
    last_hit_t = -1e9
    for k in range(1, len(valid_rows)):
        if (math.isfinite(vz[k - 1]) and math.isfinite(vz[k])
                and vz[k - 1] < -0.10 and vz[k] > 0.10
                and math.isfinite(t[k]) and t[k] - last_hit_t > 0.08):
            contact_like.append(t[k])
            last_hit_t = t[k]

    min_rel = percentile(rel_norm, 0)
    min_rel_time = None
    if min_rel is not None:
        for row, value in zip(valid_rows, rel_norm):
            if math.isfinite(value) and abs(value - min_rel) < 1e-9:
                min_rel_time = f(row, "time_sec")
                break

    return {
        "valid_duration_s": (
            max(t) - min(t) if len(finite(t)) >= 2 else 0.0),
        "ball_z_m": {
            "min": percentile(z, 0),
            "max": percentile(z, 100),
            "p95": percentile(z, 95),
        },
        "vertical_velocity_reversal_count": len(contact_like),
        "vertical_velocity_reversal_times_s": contact_like[:20],
        "min_ball_racket_relative_distance_m": min_rel,
        "min_ball_racket_relative_distance_time_s": min_rel_time,
    }


def analyze_hardware_safety(policy: list[dict], run_log: Path | None, dt: float) -> dict:
    arm_dq_abs = []
    q_delta_speed = []
    prev_q = None
    for row in policy:
        q = [f(row, f"arm_q/{joint}") for joint in JOINTS]
        dq = [f(row, f"arm_dq/{joint}") for joint in JOINTS]
        arm_dq_abs.extend(abs(v) for v in dq if math.isfinite(v))
        if prev_q is not None and all(math.isfinite(v) for v in q + prev_q):
            q_delta_speed.extend(abs(a - b) / max(dt, 1e-9) for a, b in zip(q, prev_q))
        prev_q = q if all(math.isfinite(v) for v in q) else None

    log_text = ""
    if run_log is not None and run_log.is_file():
        log_text = run_log.read_text(errors="replace")
    safety_enabled = None
    if "Safety limiter ENABLED" in log_text:
        safety_enabled = True
    if "Safety limiter DISABLED" in log_text:
        safety_enabled = False
    clip_totals = Counter()
    for match in re.finditer(
            r"Safety limiter clipped last 1s: invalid=(\d+) pos=(\d+) vel=(\d+) acc=(\d+)",
            log_text):
        clip_totals["invalid"] += int(match.group(1))
        clip_totals["pos"] += int(match.group(2))
        clip_totals["vel"] += int(match.group(3))
        clip_totals["acc"] += int(match.group(4))

    return {
        "safety_limiter_enabled": safety_enabled,
        "safety_limiter_clip_totals": dict(clip_totals),
        "ball_lost_failsafe_events": log_text.count("Ball lost beyond"),
        "done_failsafe_events": log_text.count("Done condition triggered"),
        "release_gate_cleared_events": log_text.count("Ball-release gate cleared"),
        "right_arm_feedback_abs_velocity_rad_s": {
            "p95": percentile(arm_dq_abs, 95),
            "max": percentile(arm_dq_abs, 100),
        },
        "right_arm_position_diff_speed_rad_s": {
            "p95": percentile(q_delta_speed, 95),
            "max": percentile(q_delta_speed, 100),
        },
    }


def diagnosis_hints(summary: dict) -> list[str]:
    hints: list[str] = []
    p = summary["perception"]
    io = summary["policy_io"]
    hw = summary["hardware_safety"]
    task = summary["task"]
    valid_ratio = p.get("policy_ball_valid_ratio")
    if valid_ratio is not None and valid_ratio < 0.85:
        hints.append("感知有效率偏低：优先检查视野、检测阈值、曝光和 estimator loss。")
    age_p95 = (p.get("estimator_age_s") or {}).get("p95")
    if age_p95 is not None and age_p95 > 0.08:
        hints.append("estimator/policy 输入年龄 p95 偏大：优先做延迟补偿和丢帧复刻实验。")
    residual_p95 = (p.get("estimator_vs_latest_raw_residual_mm") or {}).get("p95")
    if residual_p95 is not None and residual_p95 > 80.0:
        hints.append("estimator 与最新 raw 观测差异偏大：检查深度噪声、坐标标定或预测模型。")
    action_max = (io.get("action_abs_max") or {}).get("p95")
    if action_max is not None and action_max > 0.85:
        hints.append("策略动作接近饱和：可能是观测超训练分布或任务初始条件过激。")
    first_policy_t = io.get("first_policy_time_s")
    if first_policy_t is not None and first_policy_t > 1.0:
        hints.append("策略启动较晚：查看 release_gate/capture_gate 是否长期等待或阈值过严。")
    if hw.get("safety_limiter_enabled") is False:
        hints.append("本次 safety_limiter 关闭：硬件安全只能依赖上游状态机和外部控制器。")
    if hw.get("ball_lost_failsafe_events", 0) > 0:
        hints.append("出现丢球 failsafe：需要区分真实飞出视野和 estimator 过早/过晚判失效。")
    if task.get("vertical_velocity_reversal_count", 0) == 0:
        hints.append("没有明显竖直速度反转：可能没有有效击球，或速度估计/坐标轴方向异常。")
    return hints


def analyze(args: argparse.Namespace) -> dict:
    trajectory = read_csv_rows(args.trajectory_csv)
    policy = read_csv_rows(args.policy_trace)
    estimator = read_csv_rows(args.estimator_trace)
    summary = {
        "inputs": {
            "trajectory_csv": str(args.trajectory_csv) if args.trajectory_csv else None,
            "policy_trace": str(args.policy_trace) if args.policy_trace else None,
            "estimator_trace": str(args.estimator_trace) if args.estimator_trace else None,
            "run_log": str(args.run_log) if args.run_log else None,
            "dt": args.dt,
        },
        "perception": analyze_perception(trajectory, policy, estimator, args.dt),
        "policy_io": analyze_policy_io(policy, args.dt),
        "task": analyze_task(policy),
        "hardware_safety": analyze_hardware_safety(policy, args.run_log, args.dt),
    }
    summary["diagnosis_hints"] = diagnosis_hints(summary)
    return summary


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def print_report(summary: dict) -> str:
    p = summary["perception"]
    io = summary["policy_io"]
    task = summary["task"]
    hw = summary["hardware_safety"]
    lines = [
        "Four-layer real-run diagnostics",
        "",
        "[1] Perception",
        f"  policy valid ratio: {fmt(p.get('policy_ball_valid_ratio'))}",
        f"  ball age p95/max: {fmt(p.get('policy_ball_age_p95_s'))} / {fmt(p.get('policy_ball_max_age_s'))} s",
        f"  estimator valid/predicted ratio: {fmt(p.get('estimator_valid_ratio'))} / {fmt(p.get('estimator_predicted_ratio'))}",
        f"  estimator loss p95/max: {fmt((p.get('estimator_loss_s') or {}).get('p95'))} / {fmt((p.get('estimator_loss_s') or {}).get('max'))} s",
        f"  estimator-vs-raw residual p95: {fmt((p.get('estimator_vs_latest_raw_residual_mm') or {}).get('p95'), 1)} mm",
        "",
        "[2] Policy I/O",
        f"  mode counts: {io.get('control_mode_counts')}",
        f"  first policy time: {fmt(io.get('first_policy_time_s'))} s",
        f"  action abs max p95/max: {fmt((io.get('action_abs_max') or {}).get('p95'))} / {fmt((io.get('action_abs_max') or {}).get('max'))}",
        f"  action delta norm p95/max: {fmt((io.get('action_delta_norm_per_tick') or {}).get('p95'))} / {fmt((io.get('action_delta_norm_per_tick') or {}).get('max'))}",
        "",
        "[3] Task",
        f"  ball z min/max: {fmt((task.get('ball_z_m') or {}).get('min'))} / {fmt((task.get('ball_z_m') or {}).get('max'))} m",
        f"  vertical velocity reversals: {task.get('vertical_velocity_reversal_count')}",
        f"  min ball-racket relative distance: {fmt(task.get('min_ball_racket_relative_distance_m'))} m",
        "",
        "[4] Hardware Safety",
        f"  safety limiter enabled: {hw.get('safety_limiter_enabled')}",
        f"  limiter clip totals: {hw.get('safety_limiter_clip_totals')}",
        f"  ball-lost/done failsafe events: {hw.get('ball_lost_failsafe_events')} / {hw.get('done_failsafe_events')}",
        f"  right-arm dq p95/max: {fmt((hw.get('right_arm_feedback_abs_velocity_rad_s') or {}).get('p95'))} / {fmt((hw.get('right_arm_feedback_abs_velocity_rad_s') or {}).get('max'))} rad/s",
        "",
        "Diagnosis hints",
    ]
    hints = summary.get("diagnosis_hints") or []
    if hints:
        lines.extend(f"  - {hint}" for hint in hints)
    else:
        lines.append("  - No obvious high-level warning from these coarse metrics.")
    return "\n".join(lines) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate four-layer diagnostics for one real policy run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--trajectory-csv", type=Path, default=None)
    p.add_argument("--policy-trace", type=Path, required=True)
    p.add_argument("--estimator-trace", type=Path, default=None)
    p.add_argument("--run-log", type=Path, default=None)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-report", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = analyze(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[real-run-metrics] ERROR: {exc}", file=sys.stderr)
        return 1
    report = print_report(summary)
    print(report, end="")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
