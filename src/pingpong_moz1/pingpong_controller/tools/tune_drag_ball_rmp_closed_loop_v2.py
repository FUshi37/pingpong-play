#!/usr/bin/env python3
"""Select a V2 RMP candidate using a replayed drag-ball policy closed loop."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_rmp_closed_loop_v2_mpl")

import numpy as np

import analyze_drag_ball_sim2real as drag
import analyze_valid_juggle_sim2real as base
from tune_drag_ball_rmp_replay_v2 import (
    CandidateV2,
    apply_candidate_v2_to_mirror,
)


Q_SCALE = 0.05
DQ_SCALE = 1.0


def _rms(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.sqrt(np.mean(finite * finite))) if finite.size else math.nan


def _window(
    real_q: np.ndarray,
    real_dq: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    q_error = q[mask] - real_q[mask]
    dq_error = dq[mask] - real_dq[mask]
    q_rms = _rms(q_error)
    dq_rms = _rms(dq_error)
    return {
        "samples": int(np.count_nonzero(mask)),
        "q_rms_rad": q_rms,
        "dq_rms_rad_s": dq_rms,
        "q_per_joint_rms_rad": np.sqrt(
            np.mean(q_error * q_error, axis=0)).tolist(),
        "dq_per_joint_rms_rad_s": np.sqrt(
            np.mean(dq_error * dq_error, axis=0)).tolist(),
        "normalized_score": math.sqrt(
            0.5 * (q_rms / Q_SCALE) ** 2
            + 0.5 * (dq_rms / DQ_SCALE) ** 2),
    }


def _metrics(
    elapsed: np.ndarray,
    real_q: np.ndarray,
    real_dq: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
) -> dict[str, Any]:
    windows = {
        "full": np.ones(len(elapsed), dtype=bool),
        "post_warmup": elapsed >= 1.0,
        "train_1_to_9_s": (elapsed >= 1.0) & (elapsed < 9.0),
        "validation_9_to_14_s": (elapsed >= 9.0) & (elapsed < 14.0),
        "held_out_14_s_to_end": elapsed >= 14.0,
    }
    output = {
        name: _window(real_q, real_dq, q, dq, mask)
        for name, mask in windows.items()
        if np.any(mask)
    }
    output["development_score"] = (
        0.65 * output["train_1_to_9_s"]["normalized_score"]
        + 0.35 * output["validation_9_to_14_s"]["normalized_score"])
    return output


class ClosedLoopTuner:
    def __init__(
        self,
        recording: Path,
        aligned: list[dict[str, Any]],
        elapsed: np.ndarray,
        real_q: np.ndarray,
        real_dq: np.ndarray,
        output: Path,
        *,
        log_name: str = "closed_loop_candidate_search.jsonl",
    ) -> None:
        config = drag._mirror_config(
            recording, base.DEFAULT_CHECKPOINT, base.DEFAULT_XML)
        from pingpong_controller.sim_mirror import RealBallSimMirror

        self.mirror = RealBallSimMirror(config)
        if self.mirror.rmp_planner is None:
            raise RuntimeError("recovered-RMP planner is unavailable")
        self.base_state = {
            "params": self.mirror.rmp_planner.params,
            "limits": self.mirror.rmp_planner.limits,
            "dof_damping": self.mirror.model.dof_damping[
                self.mirror.arm_vadr].copy(),
            "dof_armature": self.mirror.model.dof_armature[
                self.mirror.arm_vadr].copy(),
            "acceleration_feedforward_s2": (
                self.mirror.rmp_acceleration_feedforward_s2.copy()),
            "actuator_gainprm_0": self.mirror.model.actuator_gainprm[
                self.mirror.arm_aids, 0].copy(),
            "actuator_biasprm_1": self.mirror.model.actuator_biasprm[
                self.mirror.arm_aids, 1].copy(),
            "actuator_biasprm_2": self.mirror.model.actuator_biasprm[
                self.mirror.arm_aids, 2].copy(),
        }
        self.aligned = aligned
        self.elapsed = elapsed
        self.real_q = real_q
        self.real_dq = real_dq
        self.options = SimpleNamespace(
            rl_use_valid_ball_obs_age=True,
            policy_observation_scales=(),
        )
        self.cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.evaluations = 0
        self.log_path = output / log_name
        self.log_path.write_text("", encoding="utf-8")

    def close(self) -> None:
        self.mirror.close()

    def evaluate(
        self,
        candidate: CandidateV2,
        *,
        search_only: bool = True,
    ) -> dict[str, Any]:
        count = int(np.searchsorted(self.elapsed, 14.0, side="left"))
        if not search_only:
            count = len(self.elapsed)
        key = (candidate.key(), count)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        apply_candidate_v2_to_mirror(
            self.mirror, candidate, base_state=self.base_state)
        first = self.aligned[0]
        drag._reset_replay_mirror(
            self.mirror, first["sim_q"], first["sim_dq"], first)
        q_rows: list[np.ndarray] = []
        dq_rows: list[np.ndarray] = []
        qcmd_rows: list[np.ndarray] = []
        action_rows: list[np.ndarray] = []
        for item in self.aligned[:count]:
            q_rows.append(
                self.mirror.data.qpos[self.mirror.arm_qadr]
                .astype(np.float64).copy())
            dq_rows.append(
                self.mirror.data.qvel[self.mirror.arm_vadr]
                .astype(np.float64).copy())
            sync = item["sim"].get("sync") or {}
            ball = base.vector(sync.get("inputBall"), 3)
            velocity = base.vector(sync.get("inputVelocity"), 3)
            if ball is None or velocity is None:
                raise RuntimeError(f"missing ball input at sourceSeq={item['seq']}")
            snapshot = self.mirror.step(
                ball.astype(np.float32),
                velocity.astype(np.float32),
                ball_valid=bool(sync.get("inputValid", True)),
                ball_obs_age_s=float(sync.get("inputAgeS", 0.0) or 0.0),
                options=self.options,
            )
            qcmd = base.vector(snapshot.get("policyCommand"), 7)
            action = base.vector(snapshot.get("policyAction"), 7)
            if qcmd is None or action is None:
                raise RuntimeError("closed-loop snapshot lacks policy output")
            qcmd_rows.append(qcmd)
            action_rows.append(action)
        q = np.asarray(q_rows)
        dq = np.asarray(dq_rows)
        result = {
            "candidate": candidate,
            "q": q,
            "dq": dq,
            "qcmd": np.asarray(qcmd_rows),
            "action": np.asarray(action_rows),
            "metrics": _metrics(
                self.elapsed[:count], self.real_q[:count],
                self.real_dq[:count], q, dq),
        }
        self.cache[key] = result
        self.evaluations += 1
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "evaluation": self.evaluations,
                "search_only": search_only,
                "development_score": result["metrics"]["development_score"],
                "candidate": asdict(candidate),
            }, ensure_ascii=False) + "\n")
        print(
            f"eval={self.evaluations:03d} "
            f"dev={result['metrics']['development_score']:.7f} "
            f"q={result['metrics']['post_warmup']['q_rms_rad']:.5f} "
            f"dq={result['metrics']['post_warmup']['dq_rms_rad_s']:.4f}",
            flush=True,
        )
        return result


def _candidate_from_mapping(value: dict[str, Any]) -> CandidateV2:
    selected = dict(value)
    defaults = CandidateV2()
    for field in defaults.__dataclass_fields__:
        if field not in selected:
            selected[field] = getattr(defaults, field)
        if isinstance(getattr(defaults, field), tuple):
            selected[field] = tuple(selected[field])
    return CandidateV2(**selected)


def _choose(
    tuner: ClosedLoopTuner,
    current: dict[str, Any],
    candidates: Iterable[CandidateV2],
    stage: str,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    before = current
    evaluated = [tuner.evaluate(candidate) for candidate in candidates]
    winner = min(
        evaluated, key=lambda item: item["metrics"]["development_score"])
    if winner["metrics"]["development_score"] < (
        current["metrics"]["development_score"] - 1.0e-6
    ):
        current = winner
    decisions.append({
        "stage": stage,
        "before": asdict(before["candidate"]),
        "selected": asdict(current["candidate"]),
        "before_development_score": before["metrics"]["development_score"],
        "selected_development_score": current["metrics"]["development_score"],
    })
    print(
        f"select {stage} dev={current['metrics']['development_score']:.7f}",
        flush=True,
    )
    return current


def _replace_delay(
    candidate: CandidateV2, joint: int, value: int
) -> CandidateV2:
    delays = list(candidate.output_delay_steps_per_joint)
    delays[joint] = int(value)
    return replace(candidate, output_delay_steps_per_joint=tuple(delays))


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
    output.mkdir(parents=True, exist_ok=True)
    _stage, aligned, _excluded = base.load_alignment(recording)
    elapsed = np.asarray([
        item["time"] - aligned[0]["time"] for item in aligned])
    real_q = base.aligned_matrix(aligned, "real_q")
    real_dq = base.aligned_matrix(aligned, "real_dq")
    recorded_q = base.aligned_matrix(aligned, "sim_q")
    recorded_dq = base.aligned_matrix(aligned, "sim_dq")

    v2_result = json.loads((
        output / "rmp_tuning_v2_results.json").read_text(encoding="utf-8"))
    selected_v2 = _candidate_from_mapping(v2_result["selected_candidate"])
    v1 = CandidateV2()
    original = CandidateV2(
        target_update_delay_steps=3,
        output_delay_steps_per_joint=(18, 19, 19, 18, 18, 18, 18))
    tuner = ClosedLoopTuner(
        recording, aligned, elapsed, real_q, real_dq, output)
    decisions: list[dict[str, Any]] = []
    try:
        original_replay = tuner.evaluate(original)
        v1_replay = tuner.evaluate(v1)
        v2_replay = tuner.evaluate(selected_v2)
        current = min(
            (original_replay, v1_replay, v2_replay),
            key=lambda item: item["metrics"]["development_score"])
        decisions.append({
            "stage": "initial_model_selection",
            "selected": asdict(current["candidate"]),
            "scores": {
                "original": original_replay["metrics"]["development_score"],
                "v1": v1_replay["metrics"]["development_score"],
                "v2": v2_replay["metrics"]["development_score"],
            },
        })

        current = _choose(
            tuner, current,
            [replace(
                current["candidate"],
                output_delay_steps_per_joint=tuple(
                    max(0, int(value) + offset)
                    for value in current["candidate"].output_delay_steps_per_joint))
             for offset in (-2, -1, 0, 1, 2)],
            "global_output_delay", decisions)
        current = _choose(
            tuner, current,
            [replace(current["candidate"], target_update_delay_steps=value)
             for value in (2, 3, 4)],
            "target_update_delay", decisions)
        for joint in range(7):
            center = current["candidate"].output_delay_steps_per_joint[joint]
            current = _choose(
                tuner, current,
                [_replace_delay(current["candidate"], joint, value)
                 for value in range(max(0, center - 1), center + 2)],
                f"output_delay_J{joint + 1}", decisions)
        current = _choose(
            tuner, current,
            [replace(current["candidate"], joint_acceleration_feedforward=value)
             for value in (0.0, 0.025, 0.05, 0.075, 0.10)],
            "joint_acceleration_feedforward", decisions)
        acceleration_ff = np.asarray(
            current["candidate"].acceleration_feedforward_s2_per_joint)
        current = _choose(
            tuner, current,
            [replace(
                current["candidate"],
                acceleration_feedforward_s2_per_joint=tuple(
                    (acceleration_ff * scale).tolist()))
             for scale in (0.0, 0.5, 1.0, 1.5)],
            "low_level_acceleration_feedforward_scale", decisions)
        current = _choose(
            tuner, current,
            [
                current["candidate"],
                replace(
                    current["candidate"],
                    dof_damping_scale_per_joint=(1.0,) * 7),
                replace(
                    current["candidate"],
                    dof_armature_scale_per_joint=(1.0,) * 7),
                replace(
                    current["candidate"],
                    dof_damping_scale_per_joint=(1.0,) * 7,
                    dof_armature_scale_per_joint=(1.0,) * 7),
            ],
            "low_level_damping_armature_ablation", decisions)

        original_full = tuner.evaluate(original, search_only=False)
        v1_full = tuner.evaluate(v1, search_only=False)
        v2_full = tuner.evaluate(selected_v2, search_only=False)
        selected_full = tuner.evaluate(
            current["candidate"], search_only=False)
        reproduction = {
            "q_rms_rad": _rms(original_full["q"] - recorded_q),
            "dq_rms_rad_s": _rms(original_full["dq"] - recorded_dq),
        }
        result = {
            "recording": recording.name,
            "ticks": len(aligned),
            "duration_s": float(elapsed[-1]),
            "objective": (
                "closed-loop real q/dq: 0.65*train(1-9s) + "
                "0.35*validation(9-14s); 14s+ held out"),
            "fresh_original_reproduction": reproduction,
            "original_candidate": asdict(original),
            "v1_candidate": asdict(v1),
            "same_command_v2_candidate": asdict(selected_v2),
            "selected_candidate": asdict(selected_full["candidate"]),
            "original_metrics": original_full["metrics"],
            "v1_metrics": v1_full["metrics"],
            "same_command_v2_metrics": v2_full["metrics"],
            "selected_metrics": selected_full["metrics"],
            "search_decisions": decisions,
            "evaluation_count": tuner.evaluations,
        }
        (output / "closed_loop_tuning_results.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
            + "\n", encoding="utf-8")
        np.savez_compressed(
            output / "closed_loop_selected_replay.npz",
            elapsed=elapsed,
            q=selected_full["q"],
            dq=selected_full["dq"],
            qcmd=selected_full["qcmd"],
            action=selected_full["action"],
        )
        print(json.dumps({
            "output": str(output),
            "evaluations": tuner.evaluations,
            "reproduction": reproduction,
            "initial_scores": decisions[0]["scores"],
            "selected_candidate": result["selected_candidate"],
            "original_post": original_full["metrics"]["post_warmup"],
            "v1_post": v1_full["metrics"]["post_warmup"],
            "v2_post": v2_full["metrics"]["post_warmup"],
            "selected_post": selected_full["metrics"]["post_warmup"],
            "selected_held_out": selected_full["metrics"][
                "held_out_14_s_to_end"],
        }, indent=2, ensure_ascii=False))
    finally:
        tuner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
