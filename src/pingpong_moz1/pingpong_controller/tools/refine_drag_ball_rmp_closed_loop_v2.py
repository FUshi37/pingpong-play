#!/usr/bin/env python3
"""Repeat local closed-loop delay sweeps around the selected V2 candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

import analyze_valid_juggle_sim2real as base
from tune_drag_ball_rmp_closed_loop_v2 import (
    ClosedLoopTuner,
    _candidate_from_mapping,
    _choose,
    _replace_delay,
)


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
        else recording / "analysis" / "v91_rmp_tuning_drag_ball_v2")
    previous = json.loads((
        output / "closed_loop_tuning_results.json").read_text(encoding="utf-8"))
    candidate = _candidate_from_mapping(previous["selected_candidate"])
    _stage, aligned, _excluded = base.load_alignment(recording)
    elapsed = np.asarray([
        item["time"] - aligned[0]["time"] for item in aligned])
    real_q = base.aligned_matrix(aligned, "real_q")
    real_dq = base.aligned_matrix(aligned, "real_dq")
    tuner = ClosedLoopTuner(
        recording, aligned, elapsed, real_q, real_dq, output,
        log_name="closed_loop_refine_search.jsonl")
    decisions: list[dict] = []
    try:
        initial = tuner.evaluate(candidate)
        current = initial
        for sweep in range(2):
            for joint in range(7):
                center = current["candidate"].output_delay_steps_per_joint[joint]
                current = _choose(
                    tuner, current,
                    [_replace_delay(current["candidate"], joint, value)
                     for value in range(max(0, center - 1), center + 2)],
                    f"refine_{sweep + 1}_output_delay_J{joint + 1}",
                    decisions)
        current = _choose(
            tuner, current,
            [replace(current["candidate"], target_update_delay_steps=value)
             for value in (3, 4)],
            "refine_target_update_delay", decisions)
        center_acc = current["candidate"].joint_acceleration_feedforward
        current = _choose(
            tuner, current,
            [replace(
                current["candidate"],
                joint_acceleration_feedforward=max(0.0, center_acc + delta))
             for delta in (-0.025, 0.0, 0.025)],
            "refine_joint_acceleration_feedforward", decisions)
        selected = tuner.evaluate(current["candidate"], search_only=False)
        result = {
            "recording": recording.name,
            "objective": previous["objective"],
            "initial_candidate": asdict(initial["candidate"]),
            "selected_candidate": asdict(selected["candidate"]),
            "initial_search_metrics": initial["metrics"],
            "selected_metrics": selected["metrics"],
            "search_decisions": decisions,
            "evaluation_count": tuner.evaluations,
        }
        (output / "closed_loop_refined_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n", encoding="utf-8")
        np.savez_compressed(
            output / "closed_loop_selected_replay.npz",
            elapsed=elapsed,
            q=selected["q"],
            dq=selected["dq"],
            qcmd=selected["qcmd"],
            action=selected["action"],
        )
        print(json.dumps({
            "output": str(output),
            "evaluations": tuner.evaluations,
            "initial_candidate": result["initial_candidate"],
            "selected_candidate": result["selected_candidate"],
            "initial_development": initial["metrics"]["development_score"],
            "selected_post": selected["metrics"]["post_warmup"],
            "selected_held_out": selected["metrics"][
                "held_out_14_s_to_end"],
        }, indent=2, ensure_ascii=False))
    finally:
        tuner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
