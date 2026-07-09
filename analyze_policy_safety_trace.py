#!/usr/bin/env python3
"""Summarize policy replay traces for real-system safety checks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


JOINTS = [
    "RightArm-0",
    "RightArm-1",
    "RightArm-2",
    "RightArm-3",
    "RightArm-4",
    "RightArm-5",
    "RightArm-6",
]


def parse_limits(text: str | None, name: str) -> tuple[float, float] | None:
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"{name} must be min,max, got: {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
        raise argparse.ArgumentTypeError(f"{name} has invalid limits: {text!r}")
    return lo, hi


def as_float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def range_dict(values: list[float]) -> dict:
    vals = finite(values)
    if not vals:
        return {"min": None, "max": None}
    return {"min": min(vals), "max": max(vals)}


def action_col(row: dict, joint: str, idx: int) -> float:
    named = f"action/{joint}"
    numbered = f"action/{idx:03d}"
    if named in row:
        return as_float(row, named)
    return as_float(row, numbered)


def first_anchor(rows: list[dict]) -> tuple[float, float, float]:
    row = rows[0]
    anchor = (
        as_float(row, "obs/026"),
        as_float(row, "obs/027"),
        as_float(row, "obs/028"),
    )
    if all(math.isfinite(v) for v in anchor):
        return anchor
    return (0.0, 0.0, 0.0)


def update_condition(
    summary: dict,
    name: str,
    active: bool,
    time_s: float,
    value: float | None = None,
) -> None:
    item = summary.setdefault(name, {"count": 0, "first_time_s": None, "first_value": None})
    if not active:
        return
    item["count"] += 1
    if item["first_time_s"] is None:
        item["first_time_s"] = time_s
        item["first_value"] = value


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze policy replay trace safety signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("policy_trace", type=Path)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--done-z-min-m", type=float, default=0.8)
    p.add_argument("--done-z-max-m", type=float, default=1.9)
    p.add_argument("--done-x-range-m", type=float, default=0.5)
    p.add_argument("--done-y-range-m", type=float, default=0.5)
    p.add_argument("--done-racket-anchor-distance-m", type=float, default=1.1)
    p.add_argument("--done-racket-z-high-offset-m", type=float, default=0.24)
    p.add_argument("--done-racket-z-low-offset-m", type=float, default=-0.12)
    p.add_argument("--ee-x-limits-m", default=None, help="Optional min,max")
    p.add_argument("--ee-y-limits-m", default=None, help="Optional min,max")
    p.add_argument("--ee-z-limits-m", default=None, help="Optional min,max")
    p.add_argument(
        "--fail-on-nonfinite-action",
        action="store_true",
        help="Exit non-zero if any action is NaN/Inf.",
    )
    return p


def analyze(args: argparse.Namespace) -> dict:
    with args.policy_trace.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty policy trace: {args.policy_trace}")

    anchor = first_anchor(rows)
    ee_x = parse_limits(args.ee_x_limits_m, "--ee-x-limits-m")
    ee_y = parse_limits(args.ee_y_limits_m, "--ee-y-limits-m")
    ee_z = parse_limits(args.ee_z_limits_m, "--ee-z-limits-m")

    condition_summary: dict = {}
    valid_count = 0
    invalid_streak = 0
    max_invalid_streak = 0
    max_ball_age = 0.0
    action_nonfinite = 0
    action_values = {joint: [] for joint in JOINTS}
    action_delta_values = {joint: [] for joint in JOINTS}
    previous_action: list[float] | None = None

    ball_x: list[float] = []
    ball_y: list[float] = []
    ball_z: list[float] = []

    for i, row in enumerate(rows):
        t = as_float(row, "time_sec", i * args.dt)
        ball_valid = bool(as_int(row, "ball_valid", 0))
        age = as_float(row, "ball_age_s", 0.0)
        if math.isfinite(age):
            max_ball_age = max(max_ball_age, age)

        if ball_valid:
            valid_count += 1
            invalid_streak = 0
            bx = as_float(row, "ball_pos_x_m")
            by = as_float(row, "ball_pos_y_m")
            bz = as_float(row, "ball_pos_z_m")
            ball_x.append(bx)
            ball_y.append(by)
            ball_z.append(bz)
            ax, ay, _ = anchor
            update_condition(
                condition_summary, "ball_too_low",
                math.isfinite(bz) and bz < args.done_z_min_m, t, bz)
            update_condition(
                condition_summary, "ball_too_high",
                math.isfinite(bz) and bz > args.done_z_max_m, t, bz)
            update_condition(
                condition_summary, "ball_x_out_of_bounds",
                math.isfinite(bx) and abs(bx - ax) > args.done_x_range_m, t, bx)
            update_condition(
                condition_summary, "ball_y_out_of_bounds",
                math.isfinite(by) and abs(by - ay) > args.done_y_range_m, t, by)
        else:
            invalid_streak += 1
            max_invalid_streak = max(max_invalid_streak, invalid_streak)

        rx = as_float(row, "obs/026")
        ry = as_float(row, "obs/027")
        rz = as_float(row, "obs/028")
        if all(math.isfinite(v) for v in (rx, ry, rz)):
            ax, ay, az = anchor
            dist = math.sqrt((rx - ax) ** 2 + (ry - ay) ** 2 + (rz - az) ** 2)
            z_offset = rz - az
            update_condition(
                condition_summary, "racket_too_far_from_anchor",
                dist > args.done_racket_anchor_distance_m, t, dist)
            update_condition(
                condition_summary, "racket_too_high",
                z_offset > args.done_racket_z_high_offset_m, t, z_offset)
            update_condition(
                condition_summary, "racket_too_low",
                z_offset < args.done_racket_z_low_offset_m, t, z_offset)
            if ee_x is not None:
                update_condition(
                    condition_summary, "ee_x_out_of_workspace",
                    not (ee_x[0] <= rx <= ee_x[1]), t, rx)
            if ee_y is not None:
                update_condition(
                    condition_summary, "ee_y_out_of_workspace",
                    not (ee_y[0] <= ry <= ee_y[1]), t, ry)
            if ee_z is not None:
                update_condition(
                    condition_summary, "ee_z_out_of_workspace",
                    not (ee_z[0] <= rz <= ee_z[1]), t, rz)

        action = [action_col(row, joint, j) for j, joint in enumerate(JOINTS)]
        if not all(math.isfinite(v) for v in action):
            action_nonfinite += 1
        for j, joint in enumerate(JOINTS):
            action_values[joint].append(action[j])
            if previous_action is not None and math.isfinite(action[j]) and math.isfinite(previous_action[j]):
                action_delta_values[joint].append(action[j] - previous_action[j])
        previous_action = action

    duration = as_float(rows[-1], "time_sec", (len(rows) - 1) * args.dt)
    max_invalid_streak_s = max_invalid_streak * args.dt
    valid_ratio = valid_count / max(len(rows), 1)

    max_delta_by_joint = {
        joint: max((abs(v) for v in finite(vals)), default=0.0)
        for joint, vals in action_delta_values.items()
    }
    max_speed_by_joint = {
        joint: value / max(args.dt, 1e-9)
        for joint, value in max_delta_by_joint.items()
    }

    return {
        "policy_trace": str(args.policy_trace),
        "rows": len(rows),
        "duration_s": duration,
        "anchor_m": anchor,
        "ball": {
            "valid_rows": valid_count,
            "invalid_rows": len(rows) - valid_count,
            "valid_ratio": valid_ratio,
            "max_ball_age_s": max_ball_age,
            "max_invalid_streak_s": max_invalid_streak_s,
            "valid_position_range_m": {
                "x": range_dict(ball_x),
                "y": range_dict(ball_y),
                "z": range_dict(ball_z),
            },
        },
        "done_equivalent_conditions": condition_summary,
        "actions": {
            "nonfinite_rows": action_nonfinite,
            "range_rad": {
                joint: range_dict(vals)
                for joint, vals in action_values.items()
            },
            "max_abs_delta_rad_per_tick": max_delta_by_joint,
            "max_abs_speed_rad_s": max_speed_by_joint,
        },
    }


def print_report(summary: dict) -> None:
    print(f"[safety] trace={summary['policy_trace']}")
    print(
        "[safety] rows={rows} duration={duration_s:.3f}s "
        "anchor={anchor_m}".format(**summary)
    )
    ball = summary["ball"]
    print(
        "[safety] ball valid={valid_rows} invalid={invalid_rows} "
        "valid_ratio={valid_ratio:.3f} max_age={max_ball_age_s:.3f}s "
        "max_invalid_streak={max_invalid_streak_s:.3f}s".format(**ball)
    )
    active = {
        k: v for k, v in summary["done_equivalent_conditions"].items()
        if v["count"] > 0
    }
    if active:
        print("[safety] done-equivalent condition hits:")
        for name, item in sorted(active.items()):
            print(
                f"[safety]   {name}: count={item['count']} "
                f"first_t={item['first_time_s']} first_value={item['first_value']}"
            )
    else:
        print("[safety] done-equivalent condition hits: none")
    actions = summary["actions"]
    max_delta = max(actions["max_abs_delta_rad_per_tick"].values(), default=0.0)
    max_speed = max(actions["max_abs_speed_rad_s"].values(), default=0.0)
    print(
        f"[safety] actions nonfinite_rows={actions['nonfinite_rows']} "
        f"max_delta_rad_tick={max_delta:.6f} max_speed_rad_s={max_speed:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = analyze(args)
    except Exception as exc:
        print(f"[safety] ERROR: {exc}", file=sys.stderr)
        return 1
    print_report(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
    if args.fail_on_nonfinite_action and summary["actions"]["nonfinite_rows"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
