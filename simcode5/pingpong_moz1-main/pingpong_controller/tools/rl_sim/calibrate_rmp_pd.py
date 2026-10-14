#!/usr/bin/env python3
"""Jointly calibrate recovered RMP and per-joint low-level PD from recordings."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from recovered_rmp.replay import (  # noqa: E402
    IdealRMPControllerConfig,
    RecordingTrace,
    load_control_recording,
    replay_in_mujoco,
)


DEFAULT_RECORD_ROOT = (
    WORKSPACE
    / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
)
DEFAULT_TRAIN = (
    DEFAULT_RECORD_ROOT
    / "record_new3/probe_recording_20260810_195203/control_samples.jsonl",
    DEFAULT_RECORD_ROOT
    / "record_new4/probe_recording_20260811_203653/control_samples.jsonl",
)
DEFAULT_VALIDATION = (
    DEFAULT_RECORD_ROOT
    / "record_new4/probe_recording_20260811_210018/control_samples.jsonl",
)
DEFAULT_MODEL = Path(__file__).resolve().with_name("moz1_pd.xml")
DEFAULT_MOVAX_CONFIG = (
    WORKSPACE
    / "movax/controller/configuration/model/user/spirit01_base_e_no_neck_hh.yaml"
)


class Candidate(NamedTuple):
    rmp_kp_scale: float
    rmp_kd_scale: float
    kp_scale_per_joint: tuple[float, ...]
    kd_scale_per_joint: tuple[float, ...]
    qdd_feedforward: float = 0.0


_WORKER_TRACES: tuple[RecordingTrace, ...] = ()
_WORKER_MODEL = Path()
_WORKER_MOVAX = Path()
_WORKER_WEIGHTS = (0.40, 0.30, 0.30)
_WORKER_TOLERANCES = (0.02, 0.25, 5.0)


def _slice_policy_region(trace: RecordingTrace, warmup: int, cooldown: int) -> RecordingTrace:
    policy_indices = np.flatnonzero(trace.policy_mask)
    if policy_indices.size == 0:
        return trace
    start = max(0, int(policy_indices[0]) - max(0, int(warmup)))
    stop = min(trace.time.size, int(policy_indices[-1]) + max(0, int(cooldown)) + 1)
    selection = slice(start, stop)
    time = trace.time[selection].copy()
    time -= time[0]
    return RecordingTrace(
        source=trace.source,
        time=time,
        source_sequence=trace.source_sequence[selection].copy(),
        joint_time=trace.joint_time[selection].copy(),
        target_q=trace.target_q[selection].copy(),
        actual_q=trace.actual_q[selection].copy(),
        actual_qd=trace.actual_qd[selection].copy(),
        actual_qdd=trace.actual_qdd[selection].copy(),
        policy_mask=trace.policy_mask[selection].copy(),
        median_dt=trace.median_dt,
    )


def _init_worker(
    recording_paths: tuple[str, ...],
    model_path: str,
    movax_path: str,
    warmup: int,
    cooldown: int,
    weights: tuple[float, float, float],
    tolerances: tuple[float, float, float],
) -> None:
    global _WORKER_TRACES, _WORKER_MODEL, _WORKER_MOVAX
    global _WORKER_WEIGHTS, _WORKER_TOLERANCES
    _WORKER_TRACES = tuple(
        _slice_policy_region(load_control_recording(path), warmup, cooldown)
        for path in recording_paths
    )
    _WORKER_MODEL = Path(model_path)
    _WORKER_MOVAX = Path(movax_path)
    _WORKER_WEIGHTS = weights
    _WORKER_TOLERANCES = tolerances


def _candidate_result(candidate: Candidate, trace: RecordingTrace):
    from recovered_rmp.params import load_movax_rmp_config

    params, _ = load_movax_rmp_config(_WORKER_MOVAX, "RightArm")
    params = replace(
        params,
        jnt_kp=params.jnt_kp * candidate.rmp_kp_scale,
        jnt_kd=params.jnt_kd * candidate.rmp_kd_scale,
    )
    controller = IdealRMPControllerConfig(
        kp_scale_per_joint=candidate.kp_scale_per_joint,
        kd_scale_per_joint=candidate.kd_scale_per_joint,
        qdd_feedforward=candidate.qdd_feedforward,
        inverse_dynamics_feedforward=1.0,
    )
    return replay_in_mujoco(
        trace,
        _WORKER_MODEL,
        _WORKER_MOVAX,
        controller,
        rmp_params=params,
    )


def _score_candidate(candidate: Candidate) -> float:
    if (
        candidate.rmp_kp_scale <= 0.0
        or candidate.rmp_kd_scale <= 0.0
        or min(candidate.kp_scale_per_joint) <= 0.0
        or min(candidate.kd_scale_per_joint) <= 0.0
    ):
        return 1.0e12
    q_weight, qd_weight, qdd_weight = _WORKER_WEIGHTS
    q_tol, qd_tol, qdd_tol = _WORKER_TOLERANCES
    signal_losses: list[tuple[float, float, float]] = []
    try:
        for trace in _WORKER_TRACES:
            result = _candidate_result(candidate, trace)
            mask = trace.policy_mask
            if not np.any(mask):
                mask = np.ones(trace.time.size, dtype=bool)
            q_error = result.sim_q[mask] - trace.actual_q[mask]
            qd_error = result.sim_qd[mask] - trace.actual_qd[mask]
            acceleration_mask = result.acceleration_valid_mask
            qdd_error = (
                result.sim_qdd[acceleration_mask]
                - result.actual_qdd_observed[acceleration_mask]
            )
            if not all(
                np.all(np.isfinite(value))
                for value in (q_error, qd_error, qdd_error)
            ):
                return 1.0e12
            # Average per-joint normalized MSE so high-amplitude shoulder
            # motion cannot hide a poorly reproduced distal joint.
            signal_losses.append(
                (
                    float(np.mean(np.mean(q_error**2, axis=0) / q_tol**2)),
                    float(np.mean(np.mean(qd_error**2, axis=0) / qd_tol**2)),
                    float(np.mean(np.mean(qdd_error**2, axis=0) / qdd_tol**2)),
                )
            )
    except Exception:
        return 1.0e12
    mean_losses = np.mean(np.asarray(signal_losses), axis=0)
    score = (
        q_weight * mean_losses[0]
        + qd_weight * mean_losses[1]
        + qdd_weight * mean_losses[2]
    )
    # Weak log regularization discourages a brittle seven-joint gain spread
    # without preventing data-supported differences.
    kp = np.asarray(candidate.kp_scale_per_joint)
    kd = np.asarray(candidate.kd_scale_per_joint)
    regularization = 0.003 * (
        np.var(np.log(kp)) + np.var(np.log(kd))
    )
    return float(score + regularization)


def _score_item(candidate: Candidate) -> tuple[float, Candidate]:
    return _score_candidate(candidate), candidate


def _evaluate_batch(
    pool: mp.pool.Pool,
    candidates: list[Candidate],
) -> list[tuple[float, Candidate]]:
    return pool.map(_score_item, candidates)


def _multiply_candidate(
    base: Candidate,
    *,
    rmp_kp: float = 1.0,
    rmp_kd: float = 1.0,
    all_kp: float = 1.0,
    all_kd: float = 1.0,
    joint: int | None = None,
    joint_kp: float = 1.0,
    joint_kd: float = 1.0,
) -> Candidate:
    kp = np.asarray(base.kp_scale_per_joint) * all_kp
    kd = np.asarray(base.kd_scale_per_joint) * all_kd
    if joint is not None:
        kp[joint] *= joint_kp
        kd[joint] *= joint_kd
    return Candidate(
        rmp_kp_scale=float(np.clip(base.rmp_kp_scale * rmp_kp, 0.08, 2.0)),
        rmp_kd_scale=float(np.clip(base.rmp_kd_scale * rmp_kd, 0.10, 2.5)),
        kp_scale_per_joint=tuple(float(v) for v in np.clip(kp, 0.003, 0.20)),
        kd_scale_per_joint=tuple(float(v) for v in np.clip(kd, 0.015, 0.80)),
        qdd_feedforward=base.qdd_feedforward,
    )


def _detailed_metrics(candidate: Candidate, traces: tuple[RecordingTrace, ...]) -> dict[str, object]:
    output: dict[str, object] = {}
    for trace in traces:
        result = _candidate_result(candidate, trace)
        mask = trace.policy_mask if np.any(trace.policy_mask) else np.ones(trace.time.size, bool)
        values: dict[str, object] = {}
        for name, simulated, actual, signal_mask in (
            ("q", result.sim_q, trace.actual_q, mask),
            ("qd", result.sim_qd, trace.actual_qd, mask),
            (
                "qdd",
                result.sim_qdd,
                result.actual_qdd_observed,
                result.acceleration_valid_mask,
            ),
        ):
            error = simulated[signal_mask] - actual[signal_mask]
            values[f"{name}_rms"] = float(np.sqrt(np.mean(error**2)))
            values[f"{name}_rms_per_joint"] = np.sqrt(
                np.mean(error**2, axis=0)
            ).tolist()
        output[trace.source.parent.name] = values
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", default=list(DEFAULT_TRAIN))
    parser.add_argument("--validation", type=Path, nargs="+", default=list(DEFAULT_VALIDATION))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--movax-config", type=Path, default=DEFAULT_MOVAX_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("rmp_pd_calibration.json"))
    parser.add_argument("--workers", type=int, default=max(1, min(4, mp.cpu_count())))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-samples", type=int, default=100)
    parser.add_argument("--cooldown-samples", type=int, default=20)
    parser.add_argument("--global-candidates", type=int, default=48)
    parser.add_argument("--joint-sweeps", type=int, default=2)
    parser.add_argument("--q-weight", type=float, default=0.40)
    parser.add_argument("--qd-weight", type=float, default=0.30)
    parser.add_argument("--qdd-weight", type=float, default=0.30)
    parser.add_argument("--q-tolerance", type=float, default=0.02)
    parser.add_argument("--qd-tolerance", type=float, default=0.25)
    parser.add_argument("--qdd-tolerance", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = (args.q_weight, args.qd_weight, args.qdd_weight)
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one signal weight must be positive")
    weights = tuple(float(value / weight_sum) for value in weights)
    tolerances = (args.q_tolerance, args.qd_tolerance, args.qdd_tolerance)
    if min(tolerances) <= 0.0:
        raise ValueError("signal tolerances must be positive")

    initializer_args = (
        tuple(str(path.resolve()) for path in args.train),
        str(args.model.resolve()),
        str(args.movax_config.resolve()),
        args.warmup_samples,
        args.cooldown_samples,
        weights,
        tolerances,
    )
    context = mp.get_context("fork")
    initial = Candidate(
        rmp_kp_scale=0.30,
        rmp_kd_scale=math.sqrt(0.30),
        kp_scale_per_joint=(0.03,) * 7,
        kd_scale_per_joint=(0.10,) * 7,
    )
    rng = np.random.default_rng(args.seed)
    with context.Pool(
        processes=max(1, int(args.workers)),
        initializer=_init_worker,
        initargs=initializer_args,
    ) as pool:
        global_candidates = [initial]
        for _ in range(max(0, int(args.global_candidates) - 1)):
            global_candidates.append(
                Candidate(
                    rmp_kp_scale=float(np.exp(rng.uniform(np.log(0.15), np.log(0.8)))),
                    rmp_kd_scale=float(np.exp(rng.uniform(np.log(0.30), np.log(1.2)))),
                    kp_scale_per_joint=(
                        float(np.exp(rng.uniform(np.log(0.012), np.log(0.08)))),
                    )
                    * 7,
                    kd_scale_per_joint=(
                        float(np.exp(rng.uniform(np.log(0.04), np.log(0.30)))),
                    )
                    * 7,
                )
            )
        scored = _evaluate_batch(pool, global_candidates)
        best_score, best = min(scored, key=lambda item: item[0])
        print(f"global best score={best_score:.6f}: {best}", flush=True)

        broad_factors = (0.60, 0.82, 1.0, 1.25, 1.55)
        narrow_factors = (0.78, 0.92, 1.0, 1.10, 1.28)
        for sweep in range(max(0, int(args.joint_sweeps))):
            factors = broad_factors if sweep == 0 else narrow_factors
            # Refresh the four global scales around the current per-joint fit.
            global_local = [best]
            for _ in range(31):
                global_local.append(
                    _multiply_candidate(
                        best,
                        rmp_kp=float(np.exp(rng.uniform(np.log(0.75), np.log(1.35)))),
                        rmp_kd=float(np.exp(rng.uniform(np.log(0.75), np.log(1.35)))),
                        all_kp=float(np.exp(rng.uniform(np.log(0.75), np.log(1.35)))),
                        all_kd=float(np.exp(rng.uniform(np.log(0.75), np.log(1.35)))),
                    )
                )
            local_scored = _evaluate_batch(pool, global_local)
            local_score, local_best = min(local_scored, key=lambda item: item[0])
            if local_score < best_score:
                best_score, best = local_score, local_best
            for joint in range(7):
                candidates = [best]
                candidates.extend(
                    _multiply_candidate(
                        best,
                        joint=joint,
                        joint_kp=kp_factor,
                        joint_kd=kd_factor,
                    )
                    for kp_factor in factors
                    for kd_factor in factors
                )
                joint_scored = _evaluate_batch(pool, candidates)
                joint_score, joint_best = min(joint_scored, key=lambda item: item[0])
                if joint_score < best_score:
                    best_score, best = joint_score, joint_best
                print(
                    f"sweep={sweep + 1} joint={joint} score={best_score:.6f}",
                    flush=True,
                )

    # Run final train/validation reports in the parent process with full
    # policy regions, using the same worker globals for code reuse.
    all_paths = tuple(str(path.resolve()) for path in (*args.train, *args.validation))
    _init_worker(
        all_paths,
        str(args.model.resolve()),
        str(args.movax_config.resolve()),
        args.warmup_samples,
        args.cooldown_samples,
        weights,
        tolerances,
    )
    details = _detailed_metrics(best, _WORKER_TRACES)
    report = {
        "objective_score": best_score,
        "weights": {"q": weights[0], "qd": weights[1], "qdd": weights[2]},
        "tolerances": {"q": tolerances[0], "qd": tolerances[1], "qdd": tolerances[2]},
        "train_recordings": [str(path.resolve()) for path in args.train],
        "validation_recordings": [str(path.resolve()) for path in args.validation],
        "candidate": best._asdict(),
        "metrics": details,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
