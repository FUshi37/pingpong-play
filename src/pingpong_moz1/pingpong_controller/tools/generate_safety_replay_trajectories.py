#!/usr/bin/env python3
"""Generate replay CSVs for real-system policy safety tests.

The output format is accepted by pingpong_node replay mode:

    t_rel,valid,x_mm,y_mm,z_mm,vx_mm_s,vy_mm_s,vz_mm_s

Extra columns are included for offline debugging and ignored by the node.
Internally all trajectories are authored in the MJX/sim frame because the
done bounds are defined there. Use --frame base_link to write base-link
coordinates for tests that should exercise pingpong_node's real->sim transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[4]

DEFAULT_OUTPUT_BASE = PACKAGE_DIR / "outputs" / "rl_sim" / "safety_replay"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUT_BASE
    / f"safety_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_ANCHOR_TRACE = (
    PACKAGE_DIR / "data" / "rl_sim" / "validate_obs_trace"
    / "validate_obs_trace.csv"
)

EXIT_DIRECTIONS = {
    "exit_x_pos": (1.0, 0.0, 0.0),
    "exit_x_neg": (-1.0, 0.0, 0.0),
    "exit_y_pos": (0.0, 1.0, 0.0),
    "exit_y_neg": (0.0, -1.0, 0.0),
    "exit_z_high": (0.0, 0.0, 1.0),
    "exit_z_low": (0.0, 0.0, -1.0),
    "exit_xy_pos_pos": (1.0, 1.0, 0.0),
    "exit_xy_pos_neg": (1.0, -1.0, 0.0),
    "exit_xy_neg_pos": (-1.0, 1.0, 0.0),
    "exit_xy_neg_neg": (-1.0, -1.0, 0.0),
}

ALL_SCENARIOS = [
    "stationary",
    "jitter",
    "lost",
    "lost_after_seen",
    "lost_high_jitter_juggle",
    "exit_x_pos",
    "exit_x_neg",
    "exit_y_pos",
    "exit_y_neg",
    "exit_z_high",
    "exit_z_low",
    "jump",
    "up_occlusion",
    "normal_juggle",
    "wide_juggle",
    "near_floor_juggle",
    "near_ceiling_juggle",
    "edge_x_pos_hold",
    "edge_x_neg_hold",
    "edge_y_pos_hold",
    "edge_y_neg_hold",
    "edge_z_high_hold",
    "edge_z_low_hold",
    "fast_lateral_x",
    "fast_lateral_y",
    "side_arc",
    "spiral",
    "random_walk",
    "intermittent_occlusion",
    "burst_dropout",
    "flicker",
    "reacquire_shift",
    "false_positive_spike",
    "depth_spike",
    "velocity_spike",
    "drop_from_high",
    "low_fast_drop",
    "exit_xy_pos_pos",
    "exit_xy_pos_neg",
    "exit_xy_neg_pos",
    "exit_xy_neg_neg",
]

SCENARIO_CHOICES = ALL_SCENARIOS + ["all"]


@dataclass
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, p: tuple[float, float, float]) -> bool:
        x, y, z = p
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass
class GenerationConfig:
    scenario: str
    frame: str
    duration_s: float
    rate_hz: float
    seed: int
    anchor_m: tuple[float, float, float]
    center_m: tuple[float, float, float]
    done_z_min_m: float
    done_z_max_m: float
    done_x_range_m: float
    done_y_range_m: float
    space: Bounds
    sim_base_height_m: float
    sim_cartesian_x_offset_m: float
    sim_cartesian_y_offset_m: float


def parse_vec3(text: str, name: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{name} must be three comma-separated numbers, got: {text!r}")
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} contains a non-numeric value: {text!r}") from exc


def finite_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"expected a finite float, got {value}")
    return out


def read_anchor_from_policy_trace(path: Path) -> tuple[float, float, float] | None:
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration:
            return None
    keys = ("obs/026", "obs/027", "obs/028")
    if not all(k in row for k in keys):
        return None
    try:
        return tuple(float(row[k]) for k in keys)  # type: ignore[return-value]
    except ValueError:
        return None


def clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        return value
    return min(max(value, lo), hi)


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate synthetic ball replay trajectories for safety tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        choices=SCENARIO_CHOICES,
        default="all",
        help="Scenario to generate. 'all' writes one CSV per scenario.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path, or output directory when --scenario all.",
    )
    p.add_argument("--duration", type=finite_float, default=6.0)
    p.add_argument("--rate-hz", type=finite_float, default=60.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--frame",
        choices=("sim", "base_link"),
        default="sim",
        help="Coordinate frame written to CSV.",
    )

    p.add_argument(
        "--anchor-from-policy-trace",
        default=str(DEFAULT_ANCHOR_TRACE),
        help="Policy trace used to read racket anchor obs/026..028.",
    )
    p.add_argument("--anchor-x-m", type=finite_float, default=None)
    p.add_argument("--anchor-y-m", type=finite_float, default=None)
    p.add_argument("--anchor-z-m", type=finite_float, default=None)
    p.add_argument(
        "--center-m",
        default=None,
        help="Ball nominal center in sim frame as x,y,z. Defaults near anchor.",
    )

    p.add_argument("--done-z-min-m", type=finite_float, default=0.8)
    p.add_argument("--done-z-max-m", type=finite_float, default=1.9)
    p.add_argument("--done-x-range-m", type=finite_float, default=0.5)
    p.add_argument("--done-y-range-m", type=finite_float, default=0.5)

    p.add_argument("--space-x-min-m", type=finite_float, default=None)
    p.add_argument("--space-x-max-m", type=finite_float, default=None)
    p.add_argument("--space-y-min-m", type=finite_float, default=None)
    p.add_argument("--space-y-max-m", type=finite_float, default=None)
    p.add_argument("--space-z-min-m", type=finite_float, default=None)
    p.add_argument("--space-z-max-m", type=finite_float, default=None)
    p.add_argument(
        "--outside-margin-m",
        type=finite_float,
        default=0.15,
        help="How far exit targets go beyond the selected motion space.",
    )
    p.add_argument(
        "--keep-valid-outside-space",
        action="store_true",
        help="Keep observations valid after leaving --space bounds.",
    )
    p.add_argument(
        "--invalid-fill",
        choices=("hold", "zero", "continue"),
        default="hold",
        help="Position/velocity values written on valid=0 rows.",
    )

    p.add_argument(
        "--velocity-mode",
        choices=("auto", "derivative", "zero"),
        default="auto",
        help="How to populate velocity columns.",
    )

    p.add_argument("--jitter-std-mm", type=finite_float, default=5.0)
    p.add_argument("--jitter-sine-mm", type=finite_float, default=3.0)
    p.add_argument("--jitter-sine-hz", type=finite_float, default=7.0)

    p.add_argument("--lost-warmup-s", type=finite_float, default=0.8)
    p.add_argument("--capture-lost-s", type=finite_float, default=1.5)
    p.add_argument("--capture-high-jitter-s", type=finite_float, default=2.0)
    p.add_argument("--capture-transition-s", type=finite_float, default=0.8)
    p.add_argument(
        "--capture-high-z-m",
        type=finite_float,
        default=None,
        help="High visible hold height for lost_high_jitter_juggle. Defaults inside done z bounds.",
    )
    p.add_argument("--exit-start-s", type=finite_float, default=1.0)
    p.add_argument("--exit-transit-s", type=finite_float, default=1.2)

    p.add_argument("--jump-time-s", type=finite_float, default=2.0)
    p.add_argument(
        "--jump-offset-m",
        default="0.35,0.00,0.25",
        help="Jump offset from center in sim frame.",
    )
    p.add_argument(
        "--jump-target-m",
        default=None,
        help="Absolute post-jump target in sim frame as x,y,z.",
    )
    p.add_argument("--jump-velocity-m-s", default="0.0,0.0,-2.0")

    p.add_argument("--throw-vx-m-s", type=finite_float, default=0.0)
    p.add_argument("--throw-vy-m-s", type=finite_float, default=0.0)
    p.add_argument("--throw-vz-m-s", type=finite_float, default=3.6)
    p.add_argument("--gravity-m-s2", type=finite_float, default=-9.81)

    p.add_argument("--juggle-period-s", type=finite_float, default=0.65)
    p.add_argument("--juggle-x-amp-m", type=finite_float, default=0.06)
    p.add_argument("--juggle-y-amp-m", type=finite_float, default=0.04)
    p.add_argument("--juggle-z-min-m", type=finite_float, default=None)
    p.add_argument("--juggle-z-max-m", type=finite_float, default=None)
    p.add_argument(
        "--edge-margin-m",
        type=finite_float,
        default=0.03,
        help="Inside-margin used for edge-hold and edge-juggle scenarios.",
    )
    p.add_argument("--occlusion-period-s", type=finite_float, default=0.75)
    p.add_argument("--occlusion-duty", type=finite_float, default=0.35)
    p.add_argument("--burst-dropout-start-s", type=finite_float, default=1.5)
    p.add_argument("--burst-dropout-duration-s", type=finite_float, default=1.0)
    p.add_argument("--flicker-hz", type=finite_float, default=18.0)
    p.add_argument("--reacquire-lost-start-s", type=finite_float, default=1.0)
    p.add_argument("--reacquire-lost-duration-s", type=finite_float, default=1.0)
    p.add_argument(
        "--reacquire-offset-m",
        default="0.22,-0.18,0.08",
        help="Post-occlusion reacquire offset in sim frame.",
    )
    p.add_argument("--spike-time-s", type=finite_float, default=2.0)
    p.add_argument("--spike-duration-s", type=finite_float, default=0.08)
    p.add_argument(
        "--spike-offset-m",
        default="0.42,0.28,0.35",
        help="False-positive/depth-spike offset from nominal juggle point.",
    )
    p.add_argument(
        "--velocity-spike-m-s",
        default="3.0,-2.0,4.5",
        help="Velocity written during velocity_spike scenario.",
    )
    p.add_argument("--random-walk-step-m", type=finite_float, default=0.018)

    p.add_argument("--sim-base-height-m", type=finite_float, default=1.302396)
    p.add_argument("--sim-cartesian-x-offset-m", type=finite_float, default=0.0)
    p.add_argument("--sim-cartesian-y-offset-m", type=finite_float, default=0.023583)
    p.add_argument("--quiet", action="store_true")
    return p


def resolve_config(args: argparse.Namespace, scenario: str) -> GenerationConfig:
    trace_anchor = None
    trace_path = resolve_path(args.anchor_from_policy_trace)
    if trace_path is not None:
        trace_anchor = read_anchor_from_policy_trace(trace_path)

    fallback_anchor = trace_anchor or (0.0, 0.0, 1.10)
    anchor = (
        args.anchor_x_m if args.anchor_x_m is not None else fallback_anchor[0],
        args.anchor_y_m if args.anchor_y_m is not None else fallback_anchor[1],
        args.anchor_z_m if args.anchor_z_m is not None else fallback_anchor[2],
    )

    if args.center_m:
        center = parse_vec3(args.center_m, "--center-m")
    else:
        z_default = clamp(
            anchor[2] + 0.30,
            args.done_z_min_m + 0.08,
            args.done_z_max_m - 0.08,
        )
        center = (anchor[0], anchor[1], z_default)

    space = Bounds(
        x_min=(
            args.space_x_min_m
            if args.space_x_min_m is not None
            else anchor[0] - args.done_x_range_m
        ),
        x_max=(
            args.space_x_max_m
            if args.space_x_max_m is not None
            else anchor[0] + args.done_x_range_m
        ),
        y_min=(
            args.space_y_min_m
            if args.space_y_min_m is not None
            else anchor[1] - args.done_y_range_m
        ),
        y_max=(
            args.space_y_max_m
            if args.space_y_max_m is not None
            else anchor[1] + args.done_y_range_m
        ),
        z_min=args.space_z_min_m if args.space_z_min_m is not None else args.done_z_min_m,
        z_max=args.space_z_max_m if args.space_z_max_m is not None else args.done_z_max_m,
    )
    if not (
        space.x_min < space.x_max
        and space.y_min < space.y_max
        and space.z_min < space.z_max
    ):
        raise ValueError(f"invalid motion space bounds: {space}")

    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0")
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be > 0")
    if args.done_z_min_m >= args.done_z_max_m:
        raise ValueError("--done-z-min-m must be below --done-z-max-m")

    return GenerationConfig(
        scenario=scenario,
        frame=args.frame,
        duration_s=args.duration,
        rate_hz=args.rate_hz,
        seed=args.seed,
        anchor_m=anchor,
        center_m=center,
        done_z_min_m=args.done_z_min_m,
        done_z_max_m=args.done_z_max_m,
        done_x_range_m=args.done_x_range_m,
        done_y_range_m=args.done_y_range_m,
        space=space,
        sim_base_height_m=args.sim_base_height_m,
        sim_cartesian_x_offset_m=args.sim_cartesian_x_offset_m,
        sim_cartesian_y_offset_m=args.sim_cartesian_y_offset_m,
    )


def make_times(duration_s: float, rate_hz: float) -> list[float]:
    dt = 1.0 / rate_hz
    n = max(2, int(round(duration_s * rate_hz)) + 1)
    return [min(i * dt, duration_s) for i in range(n)]


def vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def lerp(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    alpha: float,
) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * alpha,
        a[1] + (b[1] - a[1]) * alpha,
        a[2] + (b[2] - a[2]) * alpha,
    )


def smoothstep(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def exit_target(
    center: tuple[float, float, float],
    direction: tuple[float, float, float],
    space: Bounds,
    margin: float,
) -> tuple[float, float, float]:
    x, y, z = center
    dx, dy, dz = direction
    if dx > 0:
        x = space.x_max + margin
    elif dx < 0:
        x = space.x_min - margin
    if dy > 0:
        y = space.y_max + margin
    elif dy < 0:
        y = space.y_min - margin
    if dz > 0:
        z = space.z_max + margin
    elif dz < 0:
        z = space.z_min - margin
    return (x, y, z)


def derivative_velocity(
    times: list[float],
    positions: list[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    n = len(times)
    out: list[tuple[float, float, float]] = []
    for i in range(n):
        if i == 0:
            j0, j1 = 0, 1
        elif i == n - 1:
            j0, j1 = n - 2, n - 1
        else:
            j0, j1 = i - 1, i + 1
        dt = max(times[j1] - times[j0], 1e-9)
        p0, p1 = positions[j0], positions[j1]
        out.append(((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt, (p1[2] - p0[2]) / dt))
    return out


def default_juggle_z_bounds(
    args: argparse.Namespace,
    cfg: GenerationConfig,
) -> tuple[float, float]:
    z_min = (
        args.juggle_z_min_m
        if args.juggle_z_min_m is not None
        else max(cfg.done_z_min_m + 0.10, cfg.center_m[2] - 0.18)
    )
    z_max = (
        args.juggle_z_max_m
        if args.juggle_z_max_m is not None
        else min(cfg.done_z_max_m - 0.10, cfg.center_m[2] + 0.22)
    )
    if z_max <= z_min:
        z_min, z_max = cfg.center_m[2] - 0.10, cfg.center_m[2] + 0.10
    return z_min, z_max


def juggle_sample(
    t: float,
    center: tuple[float, float, float],
    period: float,
    x_amp: float,
    y_amp: float,
    z_min: float,
    z_max: float,
    *,
    phase: float = 0.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    period = max(period, 1e-6)
    local_t = t + phase
    u = (local_t % period) / period
    height = z_max - z_min
    z = z_min + 4.0 * height * u * (1.0 - u)
    vz = 4.0 * height * (1.0 - 2.0 * u) / period
    wx = 2.0 * math.pi / (period * 2.0)
    wy = 2.0 * math.pi / (period * 1.5)
    x = center[0] + x_amp * math.sin(wx * local_t)
    y = center[1] + y_amp * math.sin(wy * local_t + 0.7)
    vx = x_amp * wx * math.cos(wx * local_t)
    vy = y_amp * wy * math.cos(wy * local_t + 0.7)
    return (x, y, z), (vx, vy, vz)


def visible_in_space(
    args: argparse.Namespace,
    cfg: GenerationConfig,
    p: tuple[float, float, float],
) -> bool:
    return cfg.space.contains(p) or args.keep_valid_outside_space


def edge_hold_point(
    cfg: GenerationConfig,
    axis: str,
    sign: float,
    margin: float,
) -> tuple[float, float, float]:
    x, y, z = cfg.center_m
    margin = max(0.0, margin)
    if axis == "x":
        x = (cfg.space.x_max - margin) if sign > 0 else (cfg.space.x_min + margin)
    elif axis == "y":
        y = (cfg.space.y_max - margin) if sign > 0 else (cfg.space.y_min + margin)
    elif axis == "z":
        z = (cfg.space.z_max - margin) if sign > 0 else (cfg.space.z_min + margin)
    return (x, y, z)


def dropout_window(t: float, start: float, duration: float) -> bool:
    return start <= t <= start + max(0.0, duration)


def generate_trajectory(
    args: argparse.Namespace,
    cfg: GenerationConfig,
    times: list[float],
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float, float]] | None,
    list[bool],
    list[str],
]:
    rng = random.Random(cfg.seed)
    scenario = cfg.scenario
    center = cfg.center_m
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] | None = None
    valid: list[bool] = []
    phase: list[str] = []

    if scenario == "stationary":
        for _ in times:
            positions.append(center)
            valid.append(True)
            phase.append("hold")
        velocities = [(0.0, 0.0, 0.0) for _ in times]

    elif scenario == "jitter":
        std_m = args.jitter_std_mm * 1e-3
        sine_m = args.jitter_sine_mm * 1e-3
        for t in times:
            wobble = (
                sine_m * math.sin(2.0 * math.pi * args.jitter_sine_hz * t),
                sine_m * math.sin(2.0 * math.pi * (args.jitter_sine_hz * 0.7) * t + 1.1),
                sine_m * math.sin(2.0 * math.pi * (args.jitter_sine_hz * 0.5) * t + 2.2),
            )
            noise = (
                rng.gauss(0.0, std_m),
                rng.gauss(0.0, std_m),
                rng.gauss(0.0, std_m),
            )
            positions.append(vec_add(center, vec_add(wobble, noise)))
            valid.append(True)
            phase.append("jitter")

    elif scenario == "lost":
        for _ in times:
            positions.append(center)
            valid.append(False)
            phase.append("lost")
        velocities = [(0.0, 0.0, 0.0) for _ in times]

    elif scenario == "lost_after_seen":
        warmup = min(max(args.lost_warmup_s, 0.0), cfg.duration_s)
        for t in times:
            positions.append(center)
            is_valid = t <= warmup
            valid.append(is_valid)
            phase.append("warmup_visible" if is_valid else "lost")
        velocities = [(0.0, 0.0, 0.0) for _ in times]

    elif scenario == "lost_high_jitter_juggle":
        lost_s = min(max(args.capture_lost_s, 0.0), cfg.duration_s)
        high_jitter_s = max(args.capture_high_jitter_s, 0.0)
        transition_s = max(args.capture_transition_s, 1.0 / cfg.rate_hz)
        jitter_end = min(lost_s + high_jitter_s, cfg.duration_s)
        juggle_start = min(jitter_end + transition_s, cfg.duration_s)

        high_z = (
            args.capture_high_z_m
            if args.capture_high_z_m is not None
            else min(cfg.done_z_max_m - 0.12, max(center[2] + 0.35, cfg.done_z_max_m - 0.25))
        )
        high_z = clamp(high_z, cfg.done_z_min_m + 0.10, cfg.done_z_max_m - 0.05)
        high_center = (center[0], center[1], high_z)

        period = max(args.juggle_period_s, 1.0 / cfg.rate_hz)
        z_min = (
            args.juggle_z_min_m
            if args.juggle_z_min_m is not None
            else max(cfg.done_z_min_m + 0.10, center[2] - 0.18)
        )
        z_max = (
            args.juggle_z_max_m
            if args.juggle_z_max_m is not None
            else min(cfg.done_z_max_m - 0.10, center[2] + 0.22)
        )
        if z_max <= z_min:
            z_min, z_max = center[2] - 0.10, center[2] + 0.10
        height = z_max - z_min

        def juggle_at(local_t: float) -> tuple[float, float, float]:
            u = (local_t % period) / period
            z = z_min + 4.0 * height * u * (1.0 - u)
            wx = 2.0 * math.pi / (period * 2.0)
            wy = 2.0 * math.pi / (period * 1.5)
            x = center[0] + args.juggle_x_amp_m * math.sin(wx * local_t)
            y = center[1] + args.juggle_y_amp_m * math.sin(wy * local_t + 0.7)
            return (x, y, z)

        jitter_std_m = args.jitter_std_mm * 1e-3
        jitter_sine_m = args.jitter_sine_mm * 1e-3
        transition_target = juggle_at(0.0)
        for t in times:
            if t < lost_s:
                p = high_center
                ok = False
                ph = "lost_before_capture"
            elif t < jitter_end:
                local_t = t - lost_s
                wobble = (
                    jitter_sine_m * math.sin(2.0 * math.pi * args.jitter_sine_hz * local_t),
                    jitter_sine_m * math.sin(2.0 * math.pi * (args.jitter_sine_hz * 0.7) * local_t + 1.1),
                    jitter_sine_m * math.sin(2.0 * math.pi * (args.jitter_sine_hz * 0.5) * local_t + 2.2),
                )
                noise = (
                    rng.gauss(0.0, jitter_std_m),
                    rng.gauss(0.0, jitter_std_m),
                    rng.gauss(0.0, jitter_std_m),
                )
                p = vec_add(high_center, vec_add(wobble, noise))
                ok = True
                ph = "high_jitter"
            elif t < juggle_start:
                alpha = smoothstep((t - jitter_end) / transition_s)
                p = lerp(high_center, transition_target, alpha)
                ok = True
                ph = "transition_to_juggle"
            else:
                p = juggle_at(t - juggle_start)
                ok = True
                ph = "juggle"
            positions.append(p)
            is_visible = ok and (cfg.space.contains(p) or args.keep_valid_outside_space)
            valid.append(is_visible)
            phase.append(ph if is_visible or not ok else "outside_space")

    elif scenario in EXIT_DIRECTIONS:
        target = exit_target(
            center, EXIT_DIRECTIONS[scenario], cfg.space, args.outside_margin_m)
        start = min(max(args.exit_start_s, 0.0), cfg.duration_s)
        transit = max(args.exit_transit_s, 1.0 / cfg.rate_hz)
        for t in times:
            if t < start:
                p = center
                ph = "pre_exit"
            else:
                alpha = smoothstep((t - start) / transit)
                p = lerp(center, target, alpha)
                ph = "exit"
            positions.append(p)
            is_visible = cfg.space.contains(p) or args.keep_valid_outside_space
            valid.append(is_visible)
            phase.append(ph if is_visible else "outside_space")

    elif scenario == "jump":
        jump_time = min(max(args.jump_time_s, 0.0), cfg.duration_s)
        if args.jump_target_m:
            target = parse_vec3(args.jump_target_m, "--jump-target-m")
        else:
            target = vec_add(center, parse_vec3(args.jump_offset_m, "--jump-offset-m"))
        jump_vel = parse_vec3(args.jump_velocity_m_s, "--jump-velocity-m-s")
        velocities = []
        for t in times:
            if t < jump_time:
                p = center
                v = (0.0, 0.0, 0.0)
                ph = "pre_jump"
            else:
                dt = t - jump_time
                p = vec_add(target, vec_scale(jump_vel, dt))
                v = jump_vel
                ph = "post_jump"
            positions.append(p)
            velocities.append(v)
            is_visible = cfg.space.contains(p) or args.keep_valid_outside_space
            valid.append(is_visible)
            phase.append(ph if is_visible else "outside_space")

    elif scenario == "up_occlusion":
        velocities = []
        vx, vy, vz0 = args.throw_vx_m_s, args.throw_vy_m_s, args.throw_vz_m_s
        g = args.gravity_m_s2
        for t in times:
            p = (
                center[0] + vx * t,
                center[1] + vy * t,
                center[2] + vz0 * t + 0.5 * g * t * t,
            )
            v = (vx, vy, vz0 + g * t)
            positions.append(p)
            velocities.append(v)
            is_visible = cfg.space.contains(p) or args.keep_valid_outside_space
            valid.append(is_visible)
            if is_visible:
                phase.append("visible")
            elif p[2] > cfg.space.z_max:
                phase.append("above_space")
            elif p[2] < cfg.space.z_min:
                phase.append("below_space")
            else:
                phase.append("outside_space")

    elif scenario in (
            "normal_juggle",
            "wide_juggle",
            "near_floor_juggle",
            "near_ceiling_juggle",
            "intermittent_occlusion",
            "burst_dropout",
            "flicker",
            "reacquire_shift",
            "false_positive_spike",
            "depth_spike",
            "velocity_spike"):
        velocities = []
        period = max(args.juggle_period_s, 1.0 / cfg.rate_hz)
        x_amp = args.juggle_x_amp_m
        y_amp = args.juggle_y_amp_m
        z_min, z_max = default_juggle_z_bounds(args, cfg)
        local_center = center
        phase_offset = 0.0
        if scenario == "wide_juggle":
            x_amp = min(max(args.juggle_x_amp_m, 0.18), cfg.done_x_range_m * 0.82)
            y_amp = min(max(args.juggle_y_amp_m, 0.14), cfg.done_y_range_m * 0.82)
        elif scenario == "near_floor_juggle":
            z_min = cfg.space.z_min + max(args.edge_margin_m, 0.01)
            z_max = min(cfg.space.z_max - 0.05, z_min + 0.24)
        elif scenario == "near_ceiling_juggle":
            z_max = cfg.space.z_max - max(args.edge_margin_m, 0.01)
            z_min = max(cfg.space.z_min + 0.05, z_max - 0.24)
        elif scenario == "reacquire_shift":
            phase_offset = 0.17
        if z_max <= z_min:
            z_min, z_max = center[2] - 0.10, center[2] + 0.10

        lost_start = max(0.0, args.reacquire_lost_start_s)
        lost_duration = max(0.0, args.reacquire_lost_duration_s)
        reacquire_offset = parse_vec3(args.reacquire_offset_m, "--reacquire-offset-m")
        spike_offset = parse_vec3(args.spike_offset_m, "--spike-offset-m")
        velocity_spike = parse_vec3(args.velocity_spike_m_s, "--velocity-spike-m-s")
        for t in times:
            if scenario == "reacquire_shift" and t > lost_start + lost_duration:
                shifted = vec_add(center, reacquire_offset)
                local_center = (
                    clamp(shifted[0], cfg.space.x_min + 0.04, cfg.space.x_max - 0.04),
                    clamp(shifted[1], cfg.space.y_min + 0.04, cfg.space.y_max - 0.04),
                    clamp(shifted[2], cfg.space.z_min + 0.08, cfg.space.z_max - 0.08),
                )
            p, v = juggle_sample(
                t, local_center, period, x_amp, y_amp, z_min, z_max,
                phase=phase_offset)
            ph = "juggle"
            is_visible = visible_in_space(args, cfg, p)

            if scenario == "intermittent_occlusion":
                occ_period = max(args.occlusion_period_s, 1.0 / cfg.rate_hz)
                duty = clamp(args.occlusion_duty, 0.0, 0.95)
                in_drop = (t % occ_period) < (occ_period * duty)
                is_visible = is_visible and not in_drop
                ph = "occluded" if in_drop else "visible"
            elif scenario == "burst_dropout":
                in_drop = dropout_window(
                    t, args.burst_dropout_start_s, args.burst_dropout_duration_s)
                is_visible = is_visible and not in_drop
                ph = "burst_dropout" if in_drop else "visible"
            elif scenario == "flicker":
                flicker_period = 1.0 / max(args.flicker_hz, 1e-6)
                in_drop = int(t / flicker_period) % 2 == 1
                is_visible = is_visible and not in_drop
                ph = "flicker_lost" if in_drop else "flicker_seen"
            elif scenario == "reacquire_shift":
                in_drop = dropout_window(t, lost_start, lost_duration)
                is_visible = is_visible and not in_drop
                ph = "lost_before_reacquire" if in_drop else (
                    "reacquired_shifted" if t > lost_start + lost_duration else "pre_loss")
            elif scenario == "false_positive_spike" and dropout_window(
                    t, args.spike_time_s, args.spike_duration_s):
                p = vec_add(p, spike_offset)
                v = (0.0, 0.0, 0.0)
                is_visible = True
                ph = "false_positive_spike"
            elif scenario == "depth_spike" and dropout_window(
                    t, args.spike_time_s, args.spike_duration_s):
                p = (p[0], p[1], p[2] + spike_offset[2])
                v = (v[0], v[1], 0.0)
                is_visible = True
                ph = "depth_spike"
            elif scenario == "velocity_spike" and dropout_window(
                    t, args.spike_time_s, args.spike_duration_s):
                v = velocity_spike
                ph = "velocity_spike"

            positions.append(p)
            velocities.append(v)
            valid.append(is_visible)
            phase.append(ph if is_visible else (
                ph if ph in ("occluded", "burst_dropout", "flicker_lost", "lost_before_reacquire")
                else "outside_space"))

    elif scenario.startswith("edge_") and scenario.endswith("_hold"):
        velocities = []
        edge_map = {
            "edge_x_pos_hold": ("x", 1.0),
            "edge_x_neg_hold": ("x", -1.0),
            "edge_y_pos_hold": ("y", 1.0),
            "edge_y_neg_hold": ("y", -1.0),
            "edge_z_high_hold": ("z", 1.0),
            "edge_z_low_hold": ("z", -1.0),
        }
        axis, sign = edge_map[scenario]
        p = edge_hold_point(cfg, axis, sign, args.edge_margin_m)
        for _ in times:
            positions.append(p)
            velocities.append((0.0, 0.0, 0.0))
            valid.append(visible_in_space(args, cfg, p))
            phase.append("edge_hold")

    elif scenario in ("fast_lateral_x", "fast_lateral_y", "side_arc", "spiral"):
        velocities = []
        period = max(args.juggle_period_s, 1.0 / cfg.rate_hz)
        z_min, z_max = default_juggle_z_bounds(args, cfg)
        x_span = max(0.02, cfg.space.x_max - cfg.space.x_min - 2.0 * args.edge_margin_m)
        y_span = max(0.02, cfg.space.y_max - cfg.space.y_min - 2.0 * args.edge_margin_m)
        for t in times:
            u = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / period)
            du = math.pi * math.cos(2.0 * math.pi * t / period) / period
            base_p, base_v = juggle_sample(
                t, center, period, 0.0, 0.0, z_min, z_max)
            if scenario == "fast_lateral_x":
                x = cfg.space.x_min + args.edge_margin_m + x_span * u
                y = center[1]
                vx = x_span * du
                vy = 0.0
                ph = "fast_lateral_x"
            elif scenario == "fast_lateral_y":
                x = center[0]
                y = cfg.space.y_min + args.edge_margin_m + y_span * u
                vx = 0.0
                vy = y_span * du
                ph = "fast_lateral_y"
            elif scenario == "side_arc":
                theta = -0.85 * math.pi + 1.70 * math.pi * u
                theta_dot = 1.70 * math.pi * du
                rx = x_span * 0.42
                ry = y_span * 0.42
                x = center[0] + rx * math.cos(theta)
                y = center[1] + ry * math.sin(theta)
                vx = -rx * math.sin(theta) * theta_dot
                vy = ry * math.cos(theta) * theta_dot
                ph = "side_arc"
            else:
                grow = min(1.0, t / max(cfg.duration_s * 0.75, 1e-6))
                theta = 2.0 * math.pi * t / period
                theta_dot = 2.0 * math.pi / period
                rx = x_span * 0.42 * grow
                ry = y_span * 0.42 * grow
                x = center[0] + rx * math.cos(theta)
                y = center[1] + ry * math.sin(theta)
                vx = -rx * math.sin(theta) * theta_dot
                vy = ry * math.cos(theta) * theta_dot
                ph = "spiral"
            p = (x, y, base_p[2])
            v = (vx, vy, base_v[2])
            positions.append(p)
            velocities.append(v)
            is_visible = visible_in_space(args, cfg, p)
            valid.append(is_visible)
            phase.append(ph if is_visible else "outside_space")

    elif scenario == "random_walk":
        p = list(center)
        velocities = []
        last_p = tuple(p)
        for i, _t in enumerate(times):
            p[0] += rng.gauss(0.0, args.random_walk_step_m)
            p[1] += rng.gauss(0.0, args.random_walk_step_m)
            p[2] += rng.gauss(0.0, args.random_walk_step_m * 0.8)
            p[0] = clamp(p[0], cfg.space.x_min + 0.02, cfg.space.x_max - 0.02)
            p[1] = clamp(p[1], cfg.space.y_min + 0.02, cfg.space.y_max - 0.02)
            p[2] = clamp(p[2], cfg.space.z_min + 0.03, cfg.space.z_max - 0.03)
            cur = (p[0], p[1], p[2])
            dt = 1.0 / cfg.rate_hz if i > 0 else 1.0
            velocities.append((
                (cur[0] - last_p[0]) / dt,
                (cur[1] - last_p[1]) / dt,
                (cur[2] - last_p[2]) / dt,
            ))
            positions.append(cur)
            valid.append(True)
            phase.append("random_walk")
            last_p = cur

    elif scenario in ("drop_from_high", "low_fast_drop"):
        velocities = []
        if scenario == "drop_from_high":
            start = (center[0], center[1], cfg.space.z_max - max(args.edge_margin_m, 0.02))
            v0 = (args.throw_vx_m_s, args.throw_vy_m_s, -0.20)
            ph_name = "drop_from_high"
        else:
            start = (
                center[0], center[1],
                cfg.space.z_min + max(args.edge_margin_m, 0.02) + 0.12)
            v0 = (args.throw_vx_m_s, args.throw_vy_m_s, -1.80)
            ph_name = "low_fast_drop"
        g = args.gravity_m_s2
        for t in times:
            p = (
                start[0] + v0[0] * t,
                start[1] + v0[1] * t,
                start[2] + v0[2] * t + 0.5 * g * t * t,
            )
            v = (v0[0], v0[1], v0[2] + g * t)
            positions.append(p)
            velocities.append(v)
            is_visible = visible_in_space(args, cfg, p)
            valid.append(is_visible)
            phase.append(ph_name if is_visible else (
                "below_space" if p[2] < cfg.space.z_min else "outside_space"))

    else:
        raise ValueError(f"unsupported scenario: {scenario}")

    return positions, velocities, valid, phase


def select_velocities(
    args: argparse.Namespace,
    times: list[float],
    positions: list[tuple[float, float, float]],
    explicit_velocities: list[tuple[float, float, float]] | None,
) -> list[tuple[float, float, float]]:
    if args.velocity_mode == "zero":
        return [(0.0, 0.0, 0.0) for _ in times]
    if args.velocity_mode == "derivative" or explicit_velocities is None:
        return derivative_velocity(times, positions)
    return explicit_velocities


def apply_invalid_fill(
    args: argparse.Namespace,
    positions: list[tuple[float, float, float]],
    velocities: list[tuple[float, float, float]],
    valid: list[bool],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    if args.invalid_fill == "continue":
        return positions, velocities

    out_p: list[tuple[float, float, float]] = []
    out_v: list[tuple[float, float, float]] = []
    last_p: tuple[float, float, float] | None = None
    last_v: tuple[float, float, float] | None = None
    zero = (0.0, 0.0, 0.0)
    for p, v, ok in zip(positions, velocities, valid):
        if ok:
            last_p = p
            last_v = v
            out_p.append(p)
            out_v.append(v)
            continue
        if args.invalid_fill == "zero" or last_p is None or last_v is None:
            out_p.append(zero)
            out_v.append(zero)
        else:
            out_p.append(last_p)
            out_v.append(last_v)
    return out_p, out_v


def to_output_frame(
    cfg: GenerationConfig,
    pos_sim_m: tuple[float, float, float],
    vel_sim_m_s: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if cfg.frame == "sim":
        return pos_sim_m, vel_sim_m_s
    pos_base = (
        pos_sim_m[0] - cfg.sim_cartesian_x_offset_m,
        pos_sim_m[1] - cfg.sim_cartesian_y_offset_m,
        pos_sim_m[2] - cfg.sim_base_height_m,
    )
    return pos_base, vel_sim_m_s


def range_summary(values: Iterable[float]) -> dict[str, float | None]:
    vals = list(values)
    if not vals:
        return {"min": None, "max": None}
    return {"min": min(vals), "max": max(vals)}


def write_csv_and_summary(
    args: argparse.Namespace,
    cfg: GenerationConfig,
    output_csv: Path,
) -> dict:
    times = make_times(cfg.duration_s, cfg.rate_hz)
    positions, explicit_velocities, valid, phase = generate_trajectory(args, cfg, times)
    velocities = select_velocities(args, times, positions, explicit_velocities)
    positions, velocities = apply_invalid_fill(args, positions, velocities, valid)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_rel",
        "valid",
        "x_mm",
        "y_mm",
        "z_mm",
        "vx_mm_s",
        "vy_mm_s",
        "vz_mm_s",
        "scenario",
        "phase",
    ]
    rows = []
    sim_rows = []
    for t, ok, p_sim, v_sim, ph in zip(times, valid, positions, velocities, phase):
        p_out, v_out = to_output_frame(cfg, p_sim, v_sim)
        rows.append({
            "t_rel": f"{t:.6f}",
            "valid": int(ok),
            "x_mm": f"{p_out[0] * 1000.0:.6f}",
            "y_mm": f"{p_out[1] * 1000.0:.6f}",
            "z_mm": f"{p_out[2] * 1000.0:.6f}",
            "vx_mm_s": f"{v_out[0] * 1000.0:.6f}",
            "vy_mm_s": f"{v_out[1] * 1000.0:.6f}",
            "vz_mm_s": f"{v_out[2] * 1000.0:.6f}",
            "scenario": cfg.scenario,
            "phase": ph,
        })
        sim_rows.append((p_sim, v_sim))

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_positions = [p for p, ok in zip(positions, valid) if ok]
    summary = {
        "csv": str(output_csv),
        "metadata_json": str(output_csv.with_suffix(".json")),
        "scenario": cfg.scenario,
        "frame": cfg.frame,
        "replay_data_is_sim_frame": cfg.frame == "sim",
        "duration_s": cfg.duration_s,
        "rate_hz": cfg.rate_hz,
        "rows": len(rows),
        "valid_rows": sum(1 for ok in valid if ok),
        "invalid_rows": sum(1 for ok in valid if not ok),
        "valid_ratio": sum(1 for ok in valid if ok) / max(len(valid), 1),
        "config": {
            **asdict(cfg),
            "space": asdict(cfg.space),
        },
        "all_position_m_range_written_frame": {
            "x": range_summary(float(r["x_mm"]) * 1e-3 for r in rows),
            "y": range_summary(float(r["y_mm"]) * 1e-3 for r in rows),
            "z": range_summary(float(r["z_mm"]) * 1e-3 for r in rows),
        },
        "valid_position_m_range_sim_frame": {
            "x": range_summary(p[0] for p in valid_positions),
            "y": range_summary(p[1] for p in valid_positions),
            "z": range_summary(p[2] for p in valid_positions),
        },
    }
    with output_csv.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def output_for(args: argparse.Namespace, scenario: str) -> Path:
    if args.output is None:
        return DEFAULT_OUTPUT_DIR / f"{scenario}.csv"
    output = args.output.expanduser()
    if args.scenario == "all":
        return output / f"{scenario}.csv"
    if output.suffix.lower() == ".csv":
        return output
    return output / f"{scenario}.csv"


def print_summary(summary: dict) -> None:
    print(f"[generate] scenario={summary['scenario']} frame={summary['frame']}")
    print(f"[generate] csv={summary['csv']}")
    print(
        "[generate] rows={rows} valid={valid_rows} invalid={invalid_rows} "
        "valid_ratio={valid_ratio:.3f}".format(**summary)
    )
    rng = summary["valid_position_m_range_sim_frame"]
    print(
        "[generate] valid sim range: "
        f"x={rng['x']} y={rng['y']} z={rng['z']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    scenarios = ALL_SCENARIOS if args.scenario == "all" else [args.scenario]
    summaries = []
    try:
        for scenario in scenarios:
            cfg = resolve_config(args, scenario)
            summary = write_csv_and_summary(args, cfg, output_for(args, scenario))
            summaries.append(summary)
            if not args.quiet:
                print_summary(summary)

        if args.scenario == "all":
            output_dir = args.output.expanduser() if args.output else DEFAULT_OUTPUT_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = output_dir / "manifest.json"
            with manifest.open("w") as f:
                json.dump({"scenarios": summaries}, f, indent=2, sort_keys=True)
                f.write("\n")
            if not args.quiet:
                print(f"[generate] manifest={manifest}")
    except Exception as exc:
        print(f"[generate] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
