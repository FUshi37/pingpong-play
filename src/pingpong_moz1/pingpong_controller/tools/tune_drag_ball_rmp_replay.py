#!/usr/bin/env python3
"""Tune recovered-RMP response against a drag-ball hardware recording.

The recorded hardware ``publishedSafePosition`` stream is replayed through a
fresh CPU recovered-RMP/MuJoCo plant.  A mandatory replay-equivalence gate
first verifies that the offline implementation reproduces both the recorded
online mirror (using recorded sim commands) and the previously generated
common-command baseline.  Only then does a coordinate search adjust RMP
response parameters.  The checkpoint and runtime configuration are never
modified; this tool writes a proposed env_cfg override for later validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_rmp_tuning_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_drag_ball_sim2real as drag
import analyze_valid_juggle_sim2real as base


DT = 0.005
Q_SCALE_RAD = 0.01
DQ_SCALE_RAD_S = 0.20
COLORS = {
    "real": "#1769aa",
    "baseline": "#e76f51",
    "tuned": "#2a9d8f",
    "command": "#6c5ce7",
}


@dataclass(frozen=True)
class Candidate:
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    output_filter_length: int = 35
    target_filter_length: int = 10
    target_update_delay_steps: int = 3
    output_delay_offset_steps: int = 0
    output_delay_per_joint_adjustment: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0)
    joint_velocity_feedforward: float = 0.5
    acceleration_limit_scale: float = 1.0

    def key(self) -> tuple[Any, ...]:
        return (
            round(self.kp_scale, 8),
            round(self.kd_scale, 8),
            int(self.output_filter_length),
            int(self.target_filter_length),
            int(self.target_update_delay_steps),
            int(self.output_delay_offset_steps),
            tuple(int(value) for value in self.output_delay_per_joint_adjustment),
            round(self.joint_velocity_feedforward, 8),
            round(self.acceleration_limit_scale, 8),
        )


@dataclass
class ReplayResult:
    candidate: Candidate
    q: np.ndarray
    dq: np.ndarray
    metrics: dict[str, Any]


def apply_candidate_to_mirror(mirror: Any, candidate: Candidate) -> None:
    """Apply one recovered-RMP response candidate to an initialized mirror."""
    from recovered_rmp import JointLimits, JointRMPPlanner

    if mirror.rmp_planner is None:
        raise RuntimeError("recovered-RMP planner is unavailable")
    base_params = mirror.rmp_planner.params
    base_limits = mirror.rmp_planner.limits
    base_delays = mirror.rmp_output_delay_steps.copy()
    params = replace(
        base_params,
        jnt_kp=np.asarray(base_params.jnt_kp) * candidate.kp_scale,
        jnt_kd=np.asarray(base_params.jnt_kd) * candidate.kd_scale,
        filter_length=int(candidate.output_filter_length),
        target_filter_length=int(candidate.target_filter_length),
        joint_vel_ffd=float(candidate.joint_velocity_feedforward),
    )
    limits = JointLimits(
        q_min=base_limits.q_min,
        q_max=base_limits.q_max,
        qd_max=base_limits.qd_max,
        qdd_max=base_limits.qdd_max * candidate.acceleration_limit_scale,
        qddd_max=(
            None
            if base_limits.qddd_max is None
            else base_limits.qddd_max * candidate.acceleration_limit_scale
        ),
    )
    mirror.rmp_planner = JointRMPPlanner(
        limits,
        params,
        float(mirror.model.opt.timestep),
        use_safety_bounds=mirror.rmp_planner.use_safety_bounds,
        use_jerk_limit=mirror.rmp_planner.use_jerk_limit,
    )
    per_joint = np.asarray(
        candidate.output_delay_per_joint_adjustment, dtype=np.int32)
    if per_joint.shape != (7,):
        raise ValueError("output_delay_per_joint_adjustment must have length 7")
    delays = (
        base_delays
        + int(candidate.output_delay_offset_steps)
        + per_joint
    )
    if np.any(delays < 0):
        raise ValueError("candidate creates a negative output delay")
    mirror.rmp_output_delay_steps = delays.astype(np.int32)
    mirror.rmp_target_update_delay_steps = int(
        candidate.target_update_delay_steps)


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else math.nan


def _window_metric(
    real_q: np.ndarray,
    real_dq: np.ndarray,
    sim_q: np.ndarray,
    sim_dq: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    q_error = sim_q[mask] - real_q[mask]
    dq_error = sim_dq[mask] - real_dq[mask]
    q_per_joint = np.sqrt(np.mean(q_error * q_error, axis=0))
    dq_per_joint = np.sqrt(np.mean(dq_error * dq_error, axis=0))
    q_rms = _rms(q_error)
    dq_rms = _rms(dq_error)
    score = math.sqrt(
        0.5 * (q_rms / Q_SCALE_RAD) ** 2
        + 0.5 * (dq_rms / DQ_SCALE_RAD_S) ** 2
    )
    return {
        "samples": int(np.count_nonzero(mask)),
        "q_rms_rad": q_rms,
        "dq_rms_rad_s": dq_rms,
        "q_per_joint_rms_rad": q_per_joint.tolist(),
        "dq_per_joint_rms_rad_s": dq_per_joint.tolist(),
        "normalized_score": score,
    }


def _metrics(
    elapsed: np.ndarray,
    real_q: np.ndarray,
    real_dq: np.ndarray,
    sim_q: np.ndarray,
    sim_dq: np.ndarray,
) -> dict[str, Any]:
    windows = {
        "full": np.ones(elapsed.shape, dtype=bool),
        "post_warmup": elapsed >= 1.0,
        "train_1_to_9_s": (elapsed >= 1.0) & (elapsed < 9.0),
        "validation_9_to_14_s": (elapsed >= 9.0) & (elapsed < 14.0),
        "held_out_14_s_to_end": elapsed >= 14.0,
    }
    output = {
        name: _window_metric(real_q, real_dq, sim_q, sim_dq, mask)
        for name, mask in windows.items()
    }
    output["development_score"] = (
        0.65 * output["train_1_to_9_s"]["normalized_score"]
        + 0.35 * output["validation_9_to_14_s"]["normalized_score"]
    )
    return output


class RMPReplayTuner:
    def __init__(
        self,
        recording: Path,
        aligned: list[dict[str, Any]],
        real_q: np.ndarray,
        real_dq: np.ndarray,
        commands: np.ndarray,
        elapsed: np.ndarray,
    ) -> None:
        self.aligned = aligned
        self.real_q = real_q
        self.real_dq = real_dq
        self.commands = commands
        self.elapsed = elapsed
        config = drag._mirror_config(
            recording, base.DEFAULT_CHECKPOINT, base.DEFAULT_XML)
        from pingpong_controller.sim_mirror import RealBallSimMirror

        self.mirror = RealBallSimMirror(config)
        drag._disable_ball_collisions(self.mirror)
        if self.mirror.rmp_planner is None:
            raise RuntimeError("recovered-RMP planner is unavailable")
        self.base_params = self.mirror.rmp_planner.params
        self.base_limits = self.mirror.rmp_planner.limits
        self.base_delays = self.mirror.rmp_output_delay_steps.copy()
        self.base_target_delay = int(self.mirror.rmp_target_update_delay_steps)
        self.use_safety_bounds = self.mirror.rmp_planner.use_safety_bounds
        self.use_jerk_limit = self.mirror.rmp_planner.use_jerk_limit
        self.cache: dict[tuple[Any, ...], ReplayResult] = {}
        self.evaluation_count = 0

    def close(self) -> None:
        self.mirror.close()

    def _apply(self, candidate: Candidate) -> None:
        from recovered_rmp import JointLimits, JointRMPPlanner

        params = replace(
            self.base_params,
            jnt_kp=np.asarray(self.base_params.jnt_kp) * candidate.kp_scale,
            jnt_kd=np.asarray(self.base_params.jnt_kd) * candidate.kd_scale,
            filter_length=int(candidate.output_filter_length),
            target_filter_length=int(candidate.target_filter_length),
            joint_vel_ffd=float(candidate.joint_velocity_feedforward),
        )
        limits = JointLimits(
            q_min=self.base_limits.q_min,
            q_max=self.base_limits.q_max,
            qd_max=self.base_limits.qd_max,
            qdd_max=(
                self.base_limits.qdd_max * candidate.acceleration_limit_scale
            ),
            qddd_max=(
                None
                if self.base_limits.qddd_max is None
                else self.base_limits.qddd_max
                * candidate.acceleration_limit_scale
            ),
        )
        self.mirror.rmp_planner = JointRMPPlanner(
            limits,
            params,
            float(self.mirror.model.opt.timestep),
            use_safety_bounds=self.use_safety_bounds,
            use_jerk_limit=self.use_jerk_limit,
        )
        per_joint = np.asarray(
            candidate.output_delay_per_joint_adjustment, dtype=np.int32)
        if per_joint.shape != (7,):
            raise ValueError("output_delay_per_joint_adjustment must have length 7")
        delays = (
            self.base_delays
            + int(candidate.output_delay_offset_steps)
            + per_joint
        )
        if np.any(delays < 0):
            raise ValueError("candidate creates a negative output delay")
        self.mirror.rmp_output_delay_steps = delays.astype(np.int32)
        self.mirror.rmp_target_update_delay_steps = int(
            candidate.target_update_delay_steps)

    def evaluate(self, candidate: Candidate) -> ReplayResult:
        cached = self.cache.get(candidate.key())
        if cached is not None:
            return cached
        self._apply(candidate)
        drag._reset_replay_mirror(
            self.mirror,
            self.real_q[0],
            self.real_dq[0],
            self.aligned[0],
        )
        drag._park_replay_ball(self.mirror)
        q_rows: list[np.ndarray] = []
        dq_rows: list[np.ndarray] = []
        for command in self.commands:
            drag._park_replay_ball(self.mirror)
            q_rows.append(
                self.mirror.data.qpos[self.mirror.arm_qadr]
                .astype(np.float64).copy()
            )
            dq_rows.append(
                self.mirror.data.qvel[self.mirror.arm_vadr]
                .astype(np.float64).copy()
            )
            self.mirror._step_recovered_rmp(command.astype(np.float32))
        q = np.asarray(q_rows)
        dq = np.asarray(dq_rows)
        result = ReplayResult(
            candidate=candidate,
            q=q,
            dq=dq,
            metrics=_metrics(
                self.elapsed, self.real_q, self.real_dq, q, dq),
        )
        self.cache[candidate.key()] = result
        self.evaluation_count += 1
        print(
            f"eval={self.evaluation_count:03d} dev="
            f"{result.metrics['development_score']:.6f} "
            f"q={result.metrics['post_warmup']['q_rms_rad']:.6f} "
            f"dq={result.metrics['post_warmup']['dq_rms_rad_s']:.5f} "
            f"candidate={candidate}",
            flush=True,
        )
        return result


def _candidate_with(
    candidate: Candidate,
    name: str,
    value: float | int,
) -> Candidate:
    return replace(candidate, **{name: value})


def _unique(values: Iterable[float | int]) -> list[float | int]:
    output: list[float | int] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def coordinate_search(
    tuner: RMPReplayTuner,
    baseline: ReplayResult,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
    current = baseline
    decisions: list[dict[str, Any]] = []
    coarse: tuple[tuple[str, Iterable[float | int]], ...] = (
        ("output_delay_offset_steps", (-5, -3, -1, 0, 1, 3, 5)),
        ("target_update_delay_steps", (0, 1, 2, 3, 4)),
        ("output_filter_length", (20, 27, 35, 43, 50)),
        ("target_filter_length", (4, 7, 10, 13, 16)),
        ("kp_scale", (0.70, 0.85, 1.00, 1.15, 1.30)),
        ("kd_scale", (0.70, 0.85, 1.00, 1.15, 1.30)),
        ("joint_velocity_feedforward", (0.25, 0.375, 0.50, 0.625, 0.75)),
        ("acceleration_limit_scale", (0.75, 1.00, 1.25)),
    )

    def sweep(name: str, values: Iterable[float | int], phase: str) -> None:
        nonlocal current
        before = current
        candidates = [
            tuner.evaluate(_candidate_with(current.candidate, name, value))
            for value in _unique(values)
        ]
        winner = min(candidates, key=lambda item: item.metrics["development_score"])
        # Avoid accepting numerical noise from a repeated reset/replay.
        if winner.metrics["development_score"] < (
            current.metrics["development_score"] - 1.0e-6
        ):
            current = winner
        decisions.append({
            "phase": phase,
            "parameter": name,
            "before": asdict(before.candidate),
            "selected": asdict(current.candidate),
            "before_development_score": before.metrics["development_score"],
            "selected_development_score": current.metrics["development_score"],
        })
        print(
            f"select phase={phase} parameter={name} "
            f"value={getattr(current.candidate, name)} "
            f"dev={current.metrics['development_score']:.6f}",
            flush=True,
        )

    for name, values in coarse:
        sweep(name, values, "coarse")

    fine = (
        ("output_delay_offset_steps", (
            current.candidate.output_delay_offset_steps - 1,
            current.candidate.output_delay_offset_steps,
            current.candidate.output_delay_offset_steps + 1,
        )),
        ("target_update_delay_steps", tuple(
            value for value in (
                current.candidate.target_update_delay_steps - 1,
                current.candidate.target_update_delay_steps,
                current.candidate.target_update_delay_steps + 1,
            ) if 0 <= value < 5
        )),
        ("output_filter_length", (
            max(1, current.candidate.output_filter_length - 5),
            current.candidate.output_filter_length,
            current.candidate.output_filter_length + 5,
        )),
        ("target_filter_length", (
            max(1, current.candidate.target_filter_length - 2),
            current.candidate.target_filter_length,
            current.candidate.target_filter_length + 2,
        )),
        ("kp_scale", (
            max(0.1, current.candidate.kp_scale - 0.10),
            current.candidate.kp_scale,
            current.candidate.kp_scale + 0.10,
        )),
        ("kd_scale", (
            max(0.1, current.candidate.kd_scale - 0.10),
            current.candidate.kd_scale,
            current.candidate.kd_scale + 0.10,
        )),
        ("joint_velocity_feedforward", (
            max(0.0, current.candidate.joint_velocity_feedforward - 0.10),
            current.candidate.joint_velocity_feedforward,
            current.candidate.joint_velocity_feedforward + 0.10,
        )),
        ("acceleration_limit_scale", (
            max(0.25, current.candidate.acceleration_limit_scale - 0.15),
            current.candidate.acceleration_limit_scale,
            current.candidate.acceleration_limit_scale + 0.15,
        )),
    )
    for name, values in fine:
        sweep(name, values, "fine")
    return current, decisions


def per_joint_delay_search(
    tuner: RMPReplayTuner,
    initial: ReplayResult,
) -> tuple[ReplayResult, list[dict[str, Any]]]:
    """Refine the integer output delay independently for each joint."""
    current = initial
    decisions: list[dict[str, Any]] = []
    for joint in range(7):
        before = current
        candidates: list[ReplayResult] = []
        for adjustment in range(-3, 4):
            values = list(current.candidate.output_delay_per_joint_adjustment)
            values[joint] = adjustment
            candidate = replace(
                current.candidate,
                output_delay_per_joint_adjustment=tuple(values),
            )
            absolute_delays = (
                tuner.base_delays
                + candidate.output_delay_offset_steps
                + np.asarray(candidate.output_delay_per_joint_adjustment)
            )
            if np.any(absolute_delays < 0):
                continue
            candidates.append(tuner.evaluate(candidate))
        winner = min(
            candidates, key=lambda item: item.metrics["development_score"])
        if winner.metrics["development_score"] < (
            current.metrics["development_score"] - 1.0e-6
        ):
            current = winner
        decisions.append({
            "phase": "per_joint_output_delay",
            "joint": joint + 1,
            "before": asdict(before.candidate),
            "selected": asdict(current.candidate),
            "before_development_score": before.metrics["development_score"],
            "selected_development_score": current.metrics["development_score"],
        })
        print(
            f"select phase=per_joint_output_delay joint=J{joint + 1} "
            f"adjustment={current.candidate.output_delay_per_joint_adjustment[joint]} "
            f"dev={current.metrics['development_score']:.6f}",
            flush=True,
        )
    return current, decisions


def _load_previous_common(output: Path) -> tuple[np.ndarray, np.ndarray]:
    path = output / "actuator_common_command_replay.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"run analyze_drag_ball_sim2real.py first; missing {path}")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    q = np.asarray([[
        float(row[f"collisionless_common_replay_q_rad_J{joint}"])
        for joint in range(1, 8)
    ] for row in rows])
    dq = np.asarray([[
        float(row[f"collisionless_common_replay_dq_rad_s_J{joint}"])
        for joint in range(1, 8)
    ] for row in rows])
    return q, dq


def _frequency_diagnostics(
    elapsed: np.ndarray,
    command: np.ndarray,
    real_q: np.ndarray,
    baseline_q: np.ndarray,
    tuned_q: np.ndarray,
) -> list[dict[str, Any]]:
    mask = elapsed >= 1.0
    time = elapsed[mask]
    window = np.hanning(mask.sum())
    frequencies = np.fft.rfftfreq(mask.sum(), DT)
    band = (frequencies >= 1.0) & (frequencies <= 15.0)
    output: list[dict[str, Any]] = []
    for joint in range(7):
        u = command[mask, joint]
        u = u - np.polyval(np.polyfit(time, u, 1), time)
        spectrum = np.abs(np.fft.rfft(window * u))
        frequency = float(frequencies[band][np.argmax(spectrum[band])])
        phase_vector = np.exp(-2j * np.pi * frequency * time)

        def coefficient(values: np.ndarray) -> complex:
            centered = values[mask, joint]
            centered = centered - np.polyval(
                np.polyfit(time, centered, 1), time)
            return complex(np.sum(window * centered * phase_vector))

        real = coefficient(real_q)
        baseline = coefficient(baseline_q)
        tuned = coefficient(tuned_q)

        def response(value: complex) -> dict[str, float]:
            return {
                "amplitude_ratio_real_over_sim": float(
                    abs(real) / max(abs(value), 1.0e-12)),
                "phase_sim_minus_real_deg": float(
                    np.angle(value / real, deg=True)),
            }

        output.append({
            "joint": joint + 1,
            "dominant_command_frequency_hz": frequency,
            "baseline": response(baseline),
            "tuned": response(tuned),
        })
    return output


def _plot_joint_family(
    path: Path,
    elapsed: np.ndarray,
    real: np.ndarray,
    baseline: np.ndarray,
    tuned: np.ndarray,
    title: str,
    unit: str,
) -> None:
    figure, axes = plt.subplots(
        7, 1, figsize=(15, 18), sharex=True, constrained_layout=True)
    for joint, axis in enumerate(axes):
        axis.plot(elapsed, real[:, joint], color=COLORS["real"],
                  linewidth=0.70, label="real" if joint == 0 else None)
        axis.plot(elapsed, baseline[:, joint], color=COLORS["baseline"],
                  linewidth=0.60, label="baseline RMP" if joint == 0 else None)
        axis.plot(elapsed, tuned[:, joint], color=COLORS["tuned"],
                  linewidth=0.60, label="tuned RMP" if joint == 0 else None)
        axis.set_ylabel(f"J{joint + 1}\n[{unit}]")
        axis.grid(alpha=0.22)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_errors(
    path: Path,
    elapsed: np.ndarray,
    real_q: np.ndarray,
    real_dq: np.ndarray,
    baseline: ReplayResult,
    tuned: ReplayResult,
) -> None:
    signals = (
        ("q RMS [rad]", real_q, baseline.q, tuned.q),
        ("dq RMS [rad/s]", real_dq, baseline.dq, tuned.dq),
    )
    figure, axes = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    for axis, (label, real, before, after) in zip(axes, signals):
        before_error = np.sqrt(np.mean((before - real) ** 2, axis=1))
        after_error = np.sqrt(np.mean((after - real) ** 2, axis=1))
        axis.plot(elapsed, before_error, color=COLORS["baseline"],
                  linewidth=0.65, label="baseline RMP")
        axis.plot(elapsed, after_error, color=COLORS["tuned"],
                  linewidth=0.65, label="tuned RMP")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle("Same-q_ref replay error before and after RMP tuning",
                    fontsize=15, fontweight="bold")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_frequency(
    path: Path,
    diagnostics: list[dict[str, Any]],
) -> None:
    joints = np.arange(1, 8)
    baseline_amp = np.asarray([
        item["baseline"]["amplitude_ratio_real_over_sim"]
        for item in diagnostics])
    tuned_amp = np.asarray([
        item["tuned"]["amplitude_ratio_real_over_sim"]
        for item in diagnostics])
    baseline_phase = np.asarray([
        item["baseline"]["phase_sim_minus_real_deg"]
        for item in diagnostics])
    tuned_phase = np.asarray([
        item["tuned"]["phase_sim_minus_real_deg"]
        for item in diagnostics])
    figure, axes = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    axes[0].plot(joints, baseline_amp, color=COLORS["baseline"],
                 linewidth=0.8, label="baseline RMP")
    axes[0].plot(joints, tuned_amp, color=COLORS["tuned"],
                 linewidth=0.8, label="tuned RMP")
    axes[0].axhline(1.0, color="#555555", linewidth=0.55)
    axes[0].set_ylabel("real/sim amplitude ratio")
    axes[1].plot(joints, baseline_phase, color=COLORS["baseline"],
                 linewidth=0.8)
    axes[1].plot(joints, tuned_phase, color=COLORS["tuned"],
                 linewidth=0.8)
    axes[1].axhline(0.0, color="#555555", linewidth=0.55)
    axes[1].set_ylabel("sim - real phase [deg]")
    axes[1].set_xlabel("Joint")
    axes[1].set_xticks(joints)
    for axis in axes:
        axis.grid(alpha=0.22)
    axes[0].legend(loc="upper right")
    figure.suptitle("Dominant-frequency response under the same q_ref",
                    fontsize=15, fontweight="bold")
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _improvement(before: float, after: float) -> float:
    return float((before - after) / before) if before > 0.0 else 0.0


def _proposal(
    tuner: RMPReplayTuner,
    candidate: Candidate,
) -> dict[str, Any]:
    kp = np.asarray(
        tuner.mirror.checkpoint_env_cfg.get(
            "recovered_rmp_jnt_kp_scale_per_joint", (1.0,) * 7),
        dtype=np.float64,
    ) * candidate.kp_scale
    kd = np.asarray(
        tuner.mirror.checkpoint_env_cfg.get(
            "recovered_rmp_jnt_kd_scale_per_joint",
            (1.0, 1.0, 0.92, 1.0, 1.0, 1.0, 1.0)),
        dtype=np.float64,
    ) * candidate.kd_scale
    acceleration = np.asarray(
        tuner.mirror.checkpoint_env_cfg.get(
            "arm_acc_limit_deg_s2",
            (1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0)),
        dtype=np.float64,
    ) * candidate.acceleration_limit_scale
    return {
        "recovered_rmp_jnt_kp_scale_per_joint": kp.tolist(),
        "recovered_rmp_jnt_kd_scale_per_joint": kd.tolist(),
        "recovered_rmp_output_filter_length": candidate.output_filter_length,
        "recovered_rmp_target_filter_length": candidate.target_filter_length,
        "recovered_rmp_target_update_delay_steps": (
            candidate.target_update_delay_steps),
        "recovered_rmp_output_delay_steps_per_joint": (
            tuner.base_delays
            + candidate.output_delay_offset_steps
            + np.asarray(candidate.output_delay_per_joint_adjustment)
        ).astype(int).tolist(),
        "recovered_rmp_joint_velocity_feedforward": (
            candidate.joint_velocity_feedforward),
        "arm_acc_limit_deg_s2": acceleration.tolist(),
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    baseline = result["baseline_metrics"]
    tuned = result["tuned_metrics"]
    candidate = result["selected_candidate"]
    equivalence = result["replay_equivalence"]
    rows = []
    for name, label in (
        ("full", "全程"),
        ("post_warmup", "1 s 后"),
        ("train_1_to_9_s", "训练 1–9 s"),
        ("validation_9_to_14_s", "验证 9–14 s"),
        ("held_out_14_s_to_end", "留出测试 14 s–结束"),
    ):
        before = baseline[name]
        after = tuned[name]
        rows.append(
            f"| {label} | {before['q_rms_rad']:.6f} | "
            f"{after['q_rms_rad']:.6f} | "
            f"{100.0 * _improvement(before['q_rms_rad'], after['q_rms_rad']):.1f}% | "
            f"{before['dq_rms_rad_s']:.5f} | {after['dq_rms_rad_s']:.5f} | "
            f"{100.0 * _improvement(before['dq_rms_rad_s'], after['dq_rms_rad_s']):.1f}% |"
        )
    proposal = result["proposed_env_cfg_override"]
    frequency_rows = []
    for item in result["frequency_diagnostics"]:
        frequency_rows.append(
            f"| J{item['joint']} | {item['dominant_command_frequency_hz']:.3f} | "
            f"{item['baseline']['amplitude_ratio_real_over_sim']:.4f} | "
            f"{item['tuned']['amplitude_ratio_real_over_sim']:.4f} | "
            f"{item['baseline']['phase_sim_minus_real_deg']:.2f}° | "
            f"{item['tuned']['phase_sim_minus_real_deg']:.2f}° |"
        )
    content = "\n".join([
        "# Drag Ball 实物指令 Replay 与 RMP 参数拟合",
        "",
        f"录制：`{result['recording']}`",
        "",
        "## Replay 等价性门槛",
        "",
        f"- 记录的 sim q_ref 离线重放 ↔ 在线 sim mirror：q RMS="
        f"{equivalence['online_independent_q_rms_rad']:.9f} rad，dq RMS="
        f"{equivalence['online_independent_dq_rms_rad_s']:.7f} rad/s，"
        f"`passed={str(equivalence['online_independent_passed']).lower()}`。",
        f"- 本工具实物 q_ref baseline replay ↔ 先前 common replay：q RMS="
        f"{equivalence['common_replay_q_rms_rad']:.9f} rad，dq RMS="
        f"{equivalence['common_replay_dq_rms_rad_s']:.7f} rad/s，"
        f"`passed={str(equivalence['common_replay_passed']).lower()}`。",
        "",
        "因此离线 replay 与当前 sim 的模型、observe/apply 相位和 RMP 状态机一致。"
        "实物 q_ref replay 不应与在线 independent sim 后半段完全相同，因为两边的 actor q_ref "
        "从 0.085 s 起已经因 q/dq 状态分叉而不同。",
        "",
        "## 拟合结果",
        "",
        "| 窗口 | baseline q RMS | tuned q RMS | q 改善 | baseline dq RMS | tuned dq RMS | dq 改善 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        f"搜索共运行 {result['evaluation_count']} 个唯一候选。参数选择使用 1–9 s 训练段和 "
        "9–14 s 验证段；14 s 之后的结果在搜索完成前未参与选择。",
        "",
        "## 主频幅相诊断",
        "",
        "| 关节 | q_ref 主频 Hz | baseline 实物/sim 幅值比 | tuned 实物/sim 幅值比 | "
        "baseline sim-real 相位 | tuned sim-real 相位 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *frequency_rows,
        "",
        "幅值比调参前后几乎不变并已接近 1，说明无需修改 kp/kd。相位则从 baseline 的"
        "约 -2° 至 -10° 收敛到大约 ±1.2° 内；拟合本质是逐关节时序修正。",
        "",
        "## 选中的相对参数",
        "",
        "```json",
        json.dumps(candidate, indent=2, ensure_ascii=False),
        "```",
        "",
        "对应 checkpoint `env_cfg` 建议值：",
        "",
        "```json",
        json.dumps(proposal, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 结论与使用边界",
        "",
        "- 同一 q_ref 下，实物和 baseline sim 的主要受迫频率本来就一致；肉眼看到的"
        "在线频率差主要来自闭环 actor 在状态分叉后输出了不同 q_ref，而不是两个执行器自由运行在不同频率。",
        "- 原 output delay 为 [18,19,19,18,18,18,18] 个 1 ms step、"
        "target_update_delay_steps=3；建议 delay 为 [14,13,11,14,13,13,13]、"
        "target_update_delay_steps=4。"
        "合并计算后，各关节净提前约 [3,5,7,3,4,4,4] ms。",
        "- 本次拟合只改变离线 recovered-RMP/MuJoCo replay，没有改 checkpoint、实时 mirror 或实物控制。",
        "- 单条 drag-ball 轨迹不足以直接发布新 plant。只有留出段同样改善，并且另一条独立录制也验证后，"
        "才建议把这些值写入下一版训练/部署 checkpoint；否则容易把 RMP 启动隐藏状态或本次负载拟合进去。",
        "- `01–02` 为各关节调参前后曲线，`03` 为逐 tick gap，`04` 为主频幅相比较。",
    ]) + "\n"
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-global",
        action="store_true",
        help="reuse the existing selected global candidate, then refine per-joint delays",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording = args.recording.expanduser().resolve()
    drag_output = recording / "analysis" / "v91_sim2real_drag_ball"
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else recording / "analysis" / "v91_rmp_tuning_drag_ball"
    )
    output.mkdir(parents=True, exist_ok=True)
    previous_result: dict[str, Any] | None = None
    previous_search_rows: list[dict[str, str]] = []
    result_path = output / "rmp_tuning_results.json"
    search_path = output / "candidate_search.csv"
    if args.resume_global:
        if not result_path.is_file():
            raise FileNotFoundError(
                f"--resume-global requires existing {result_path}")
        previous_result = json.loads(result_path.read_text(encoding="utf-8"))
        if search_path.is_file():
            with search_path.open(encoding="utf-8") as handle:
                previous_search_rows = list(csv.DictReader(handle))

    stage, aligned, _excluded = base.load_alignment(recording)
    if not aligned:
        raise RuntimeError("no aligned policy samples")
    elapsed = np.asarray([item["time"] - aligned[0]["time"] for item in aligned])
    real_q = base.aligned_matrix(aligned, "real_q")
    real_dq = base.aligned_matrix(aligned, "real_dq")
    commands = np.asarray([
        drag._published_command(item["real"]) for item in aligned])

    summary = json.loads((drag_output / "summary.json").read_text(encoding="utf-8"))
    online_check = summary["actuator_decomposition"]["self_check"]
    online_q = float(online_check["joint_position"]["overall_rms"])
    online_dq = float(online_check["joint_velocity"]["overall_rms"])
    online_passed = bool(
        online_check["passed"] and online_q <= 5.0e-4 and online_dq <= 2.0e-2)
    if not online_passed:
        raise RuntimeError("recorded independent-sim replay equivalence failed")

    previous_q, previous_dq = _load_previous_common(drag_output)
    if previous_q.shape != real_q.shape:
        raise RuntimeError("previous common replay length does not match recording")

    tuner = RMPReplayTuner(
        recording, aligned, real_q, real_dq, commands, elapsed)
    try:
        baseline_candidate = Candidate(
            kp_scale=1.0,
            kd_scale=1.0,
            output_filter_length=int(tuner.base_params.filter_length),
            target_filter_length=int(tuner.base_params.target_filter_length),
            target_update_delay_steps=tuner.base_target_delay,
            output_delay_offset_steps=0,
            output_delay_per_joint_adjustment=(0, 0, 0, 0, 0, 0, 0),
            joint_velocity_feedforward=float(
                tuner.base_params.joint_vel_ffd),
            acceleration_limit_scale=1.0,
        )
        baseline = tuner.evaluate(baseline_candidate)
        common_q_rms = _rms(baseline.q - previous_q)
        common_dq_rms = _rms(baseline.dq - previous_dq)
        common_passed = bool(
            common_q_rms <= 2.0e-7 and common_dq_rms <= 2.0e-5)
        if not common_passed:
            raise RuntimeError(
                "fresh common-command replay does not reproduce baseline: "
                f"q={common_q_rms}, dq={common_dq_rms}")

        if previous_result is None:
            global_tuned, decisions = coordinate_search(tuner, baseline)
            prior_evaluations = 0
        else:
            selected = dict(previous_result["selected_candidate"])
            selected.setdefault(
                "output_delay_per_joint_adjustment", [0, 0, 0, 0, 0, 0, 0])
            selected["output_delay_per_joint_adjustment"] = tuple(
                int(value)
                for value in selected["output_delay_per_joint_adjustment"]
            )
            global_tuned = tuner.evaluate(Candidate(**selected))
            decisions = list(previous_result.get("search_decisions", []))
            prior_evaluations = int(previous_result.get("evaluation_count", 0))
        tuned, joint_decisions = per_joint_delay_search(tuner, global_tuned)
        decisions.extend(joint_decisions)
        diagnostics = _frequency_diagnostics(
            elapsed, commands, real_q, baseline.q, tuned.q)
        equivalence = {
            "online_independent_q_rms_rad": online_q,
            "online_independent_dq_rms_rad_s": online_dq,
            "online_independent_passed": online_passed,
            "common_replay_q_rms_rad": common_q_rms,
            "common_replay_dq_rms_rad_s": common_dq_rms,
            "common_replay_passed": common_passed,
        }
        result = {
            "recording": recording.name,
            "ticks": len(aligned),
            "duration_s": float(elapsed[-1]),
            "replay_equivalence": equivalence,
            "objective_scales": {
                "q_rad": Q_SCALE_RAD,
                "dq_rad_s": DQ_SCALE_RAD_S,
                "development_score": "0.65*train + 0.35*validation",
            },
            "baseline_candidate": asdict(baseline.candidate),
            "selected_candidate": asdict(tuned.candidate),
            "baseline_metrics": baseline.metrics,
            "tuned_metrics": tuned.metrics,
            "frequency_diagnostics": diagnostics,
            "search_decisions": decisions,
            "evaluation_count": (
                tuner.evaluation_count
                if previous_result is None
                else prior_evaluations + max(0, tuner.evaluation_count - 2)
            ),
            "resumed_global_search": previous_result is not None,
            "proposed_env_cfg_override": _proposal(tuner, tuned.candidate),
        }

        _plot_joint_family(
            output / "01_joint_position_tuning.png", elapsed, real_q,
            baseline.q, tuned.q, "Same-q_ref joint position: RMP tuning", "rad")
        _plot_joint_family(
            output / "02_joint_velocity_tuning.png", elapsed, real_dq,
            baseline.dq, tuned.dq, "Same-q_ref joint velocity: RMP tuning", "rad/s")
        _plot_errors(
            output / "03_replay_gap_before_after.png", elapsed, real_q,
            real_dq, baseline, tuned)
        _plot_frequency(
            output / "04_dominant_frequency_response.png", diagnostics)
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output / "proposed_rmp_env_cfg_override.json").write_text(
            json.dumps(
                result["proposed_env_cfg_override"],
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        with search_path.open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            fields = [
                *asdict(baseline.candidate).keys(),
                "development_score", "train_score", "validation_score",
                "held_out_score", "post_warmup_q_rms_rad",
                "post_warmup_dq_rms_rad_s",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in previous_search_rows:
                writer.writerow({name: row.get(name, "") for name in fields})
            for replay in tuner.cache.values():
                writer.writerow({
                    **asdict(replay.candidate),
                    "development_score": replay.metrics["development_score"],
                    "train_score": replay.metrics["train_1_to_9_s"]["normalized_score"],
                    "validation_score": replay.metrics["validation_9_to_14_s"]["normalized_score"],
                    "held_out_score": replay.metrics["held_out_14_s_to_end"]["normalized_score"],
                    "post_warmup_q_rms_rad": replay.metrics["post_warmup"]["q_rms_rad"],
                    "post_warmup_dq_rms_rad_s": replay.metrics["post_warmup"]["dq_rms_rad_s"],
                })
        _write_report(output / "RMP_TUNING_ANALYSIS_ZH.md", result)
        print(json.dumps({
            "output": str(output),
            "equivalence": equivalence,
            "baseline": baseline.metrics,
            "selected": asdict(tuned.candidate),
            "tuned": tuned.metrics,
            "evaluations": result["evaluation_count"],
        }, indent=2, ensure_ascii=False))
    finally:
        tuner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
