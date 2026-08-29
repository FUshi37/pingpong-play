#!/usr/bin/env python3
"""Identify a mirror-only V3 RMP plant from the drag-ball command replay."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_rmp_tuning_v3_mpl")

import numpy as np

import analyze_drag_ball_sim2real as drag
import analyze_valid_juggle_sim2real as base
from tune_drag_ball_rmp_closed_loop_v2 import (
    ClosedLoopTuner,
    _candidate_from_mapping,
)
from tune_drag_ball_rmp_replay_v2 import (
    CandidateV2,
    ReplayV2,
    TunerV2,
    _proposal,
)


def _replace_joint_float(
    candidate: CandidateV2,
    field: str,
    joint: int,
    value: float,
) -> CandidateV2:
    values = [float(item) for item in getattr(candidate, field)]
    values[joint] = float(value)
    return replace(candidate, **{field: tuple(values)})


def _unique(values: Iterable[float]) -> list[float]:
    output: list[float] = []
    for value in values:
        rounded = round(float(value), 8)
        if rounded not in output:
            output.append(rounded)
    return output


def _choose(
    current: ReplayV2,
    candidates: list[ReplayV2],
    stage: str,
    decisions: list[dict[str, Any]],
) -> ReplayV2:
    winner = min(candidates, key=lambda item: item.metrics["development_score"])
    before_score = current.metrics["development_score"]
    if winner.metrics["development_score"] < before_score - 1.0e-7:
        current = winner
    decisions.append({
        "stage": stage,
        "before_development_score": before_score,
        "selected_development_score": current.metrics["development_score"],
        "selected_candidate": asdict(current.candidate),
    })
    print(
        f"select {stage} dev={current.metrics['development_score']:.7f}",
        flush=True,
    )
    return current


def _global_offset_sweep(
    tuner: TunerV2,
    current: ReplayV2,
    offsets: Iterable[float],
    decisions: list[dict[str, Any]],
) -> ReplayV2:
    base_delays = np.asarray(
        current.candidate.output_delay_steps_per_joint, dtype=np.float64)
    candidates = [
        tuner.evaluate(replace(
            current.candidate,
            output_delay_steps_per_joint=tuple((base_delays + offset).tolist()),
        ))
        for offset in _unique(offsets)
        if np.all(base_delays + offset >= 0.0)
    ]
    return _choose(current, candidates, "global_fractional_delay", decisions)


def _joint_sweep(
    tuner: TunerV2,
    current: ReplayV2,
    field: str,
    values: Iterable[float],
    decisions: list[dict[str, Any]],
) -> ReplayV2:
    sweep_values = _unique(values)
    for joint in range(7):
        candidates = [
            tuner.evaluate(_replace_joint_float(
                current.candidate, field, joint, value))
            for value in sweep_values
            if field != "output_delay_steps_per_joint" or value >= 0.0
        ]
        current = _choose(
            current, candidates, f"{field}_J{joint + 1}", decisions)
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recording", nargs="?", type=Path,
        default=base.ROOT / (
            "src/pingpong_moz1/pingpong_controller/outputs/"
            "interactive_recordings/probe_recording_20260829_180205"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording = args.recording.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else recording / "analysis" / "v91_rmp_tuning_drag_ball_v3")
    output.mkdir(parents=True, exist_ok=True)

    _stage, aligned, _excluded = base.load_alignment(recording)
    elapsed = np.asarray([
        item["time"] - aligned[0]["time"] for item in aligned])
    real_q = base.aligned_matrix(aligned, "real_q")
    real_dq = base.aligned_matrix(aligned, "real_dq")
    commands = np.asarray([
        drag._published_command(item["real"]) for item in aligned])

    v2_dir = recording / "analysis" / "v91_rmp_tuning_drag_ball_v2"
    v2_result = json.loads((
        v2_dir / "rmp_tuning_v2_results.json").read_text(encoding="utf-8"))
    initial_candidate = _candidate_from_mapping(v2_result["selected_candidate"])

    tuner = TunerV2(
        recording, aligned, real_q, real_dq, commands, elapsed, output)
    decisions: list[dict[str, Any]] = []
    try:
        initial = tuner.evaluate(initial_candidate)
        current = initial

        current = _global_offset_sweep(
            tuner, current, (-0.5, 0.0, 0.5, 1.0, 1.5), decisions)
        delay_center = np.asarray(
            current.candidate.output_delay_steps_per_joint, dtype=np.float64)
        for joint in range(7):
            center = float(delay_center[joint])
            candidates = [
                tuner.evaluate(_replace_joint_float(
                    current.candidate, "output_delay_steps_per_joint", joint,
                    center + delta))
                for delta in (-0.5, -0.25, 0.0, 0.25, 0.5)
                if center + delta >= 0.0
            ]
            current = _choose(
                current, candidates, f"fractional_delay_J{joint + 1}",
                decisions)

        for field, factors in (
            ("actuator_kp_scale_per_joint", (0.85, 1.0, 1.15)),
            ("actuator_kv_scale_per_joint", (0.85, 1.0, 1.15)),
        ):
            for joint in range(7):
                center = float(getattr(current.candidate, field)[joint])
                current = _joint_sweep(
                    tuner, current, field,
                    (center * factor for factor in factors), decisions)
                break

        for joint in range(7):
            center = float(current.candidate.acceleration_feedforward_s2_per_joint[joint])
            candidates = [
                tuner.evaluate(_replace_joint_float(
                    current.candidate,
                    "acceleration_feedforward_s2_per_joint", joint,
                    center + delta))
                for delta in (-1.25e-5, 0.0, 1.25e-5)
            ]
            current = _choose(
                current, candidates, f"low_level_qdd_ff_J{joint + 1}",
                decisions)

        for joint in range(7):
            center = float(current.candidate.output_delay_steps_per_joint[joint])
            candidates = [
                tuner.evaluate(_replace_joint_float(
                    current.candidate, "output_delay_steps_per_joint", joint,
                    center + delta))
                for delta in (-0.125, 0.0, 0.125)
                if center + delta >= 0.0
            ]
            current = _choose(
                current, candidates, f"fine_fractional_delay_J{joint + 1}",
                decisions)

        initial_full = tuner.evaluate(initial.candidate, search_only=False)
        selected_full = tuner.evaluate(current.candidate, search_only=False)
        proposal = _proposal(tuner, selected_full.candidate)
        proposal["application_scope"] = "sim_mirror_only"
        proposal["mirror_drive_mode"] = "real_published_q_ref"

        closed = ClosedLoopTuner(
            recording, aligned, elapsed, real_q, real_dq, output,
            log_name="v3_free_closed_loop_evaluation.jsonl")
        try:
            free_closed_loop = closed.evaluate(
                selected_full.candidate, search_only=False)
        finally:
            closed.close()

        result = {
            "recording": recording.name,
            "ticks": len(aligned),
            "duration_s": float(elapsed[-1]),
            "objective": (
                "real-published-q_ref mirror plant: 0.65*train(1-9s) + "
                "0.35*validation(9-14s); 14s+ held out"),
            "application_scope": "sim_mirror_only",
            "recommended_mirror_drive_mode": "real_published_q_ref",
            "v2_candidate": asdict(initial_full.candidate),
            "selected_candidate": asdict(selected_full.candidate),
            "v2_same_command_metrics": initial_full.metrics,
            "selected_same_command_metrics": selected_full.metrics,
            "selected_free_closed_loop_metrics": free_closed_loop["metrics"],
            "search_decisions": decisions,
            "evaluation_count": tuner.evaluations,
            "proposed_mirror_env_cfg_override": proposal,
        }
        (output / "rmp_tuning_v3_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n", encoding="utf-8")
        (output / "proposed_sim_mirror_rmp_override_v3.json").write_text(
            json.dumps(proposal, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n", encoding="utf-8")
        np.savez_compressed(
            output / "same_command_replay_v3.npz",
            elapsed=elapsed,
            real_q=real_q,
            real_dq=real_dq,
            command_q_ref=commands,
            v2_q=initial_full.q,
            v2_dq=initial_full.dq,
            selected_q=selected_full.q,
            selected_dq=selected_full.dq,
            free_closed_loop_q=free_closed_loop["q"],
            free_closed_loop_dq=free_closed_loop["dq"],
            free_closed_loop_qcmd=free_closed_loop["qcmd"],
            free_closed_loop_action=free_closed_loop["action"],
        )
        print(json.dumps({
            "output": str(output),
            "evaluations": tuner.evaluations,
            "selected_candidate": result["selected_candidate"],
            "v2_post": initial_full.metrics["post_warmup"],
            "selected_post": selected_full.metrics["post_warmup"],
            "selected_held_out": selected_full.metrics[
                "held_out_14_s_to_end"],
            "free_closed_loop_held_out": free_closed_loop["metrics"][
                "held_out_14_s_to_end"],
        }, indent=2, ensure_ascii=False))
    finally:
        tuner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
