#!/usr/bin/env python3
"""Tune recovered-RMP estimator/filter/feedforward internals in full MJX replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

import calibrate_rmp_pd_mjx as calibration
from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", default=list(calibration.DEFAULT_TRAIN))
    parser.add_argument(
        "--validation",
        type=Path,
        nargs="+",
        default=list(calibration.DEFAULT_VALIDATION),
    )
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).with_name("moz1_pd.xml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-samples", type=int, default=100)
    parser.add_argument("--cooldown-samples", type=int, default=20)
    return parser.parse_args()


def _constant_candidates(value: float, count: int) -> np.ndarray:
    return np.full((count, 7), value, dtype=np.float32)


def _candidate_tuple(best: dict[str, float], count: int) -> tuple[np.ndarray, ...]:
    return (
        _constant_candidates(best["process_var"], count),
        _constant_candidates(best["measure_var"], count),
        _constant_candidates(best["velocity_ffd"], count),
        _constant_candidates(best["acceleration_ffd"], count),
        _constant_candidates(best["acceleration_weight"], count),
        _constant_candidates(best["filter_length"], count).astype(np.int32),
    )


def main() -> None:
    args = parse_args()
    candidate_count = 11
    env = MjxJuggleEnv(
        args.model,
        n_envs=candidate_count,
        cfg=MjxJuggleConfig(
            domain_randomization=False,
            arm_action_limiter=False,
            recovered_rmp_motion_mode=True,
            recovered_rmp_target_filter_length=35,
            recovered_rmp_jnt_kp_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_RMP_KP
            ),
            recovered_rmp_jnt_kd_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_RMP_KD
            ),
            recovered_rmp_dof_damping_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_DAMPING
            ),
            recovered_rmp_dof_armature_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_ARMATURE
            ),
            recovered_rmp_controller_kp_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_KP
            ),
            recovered_rmp_controller_kd_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_KD
            ),
            actuator_cmd_filter=False,
            actuator_compensation_mode="none",
            enable_delay_conditioning=False,
        ),
    )
    replay_scan = calibration._make_scan(env)
    train_traces = tuple(
        calibration._slice_policy_region(
            calibration.load_control_recording(path),
            args.warmup_samples,
            args.cooldown_samples,
        )
        for path in args.train
    )
    validation_traces = tuple(
        calibration._slice_policy_region(
            calibration.load_control_recording(path),
            args.warmup_samples,
            args.cooldown_samples,
        )
        for path in args.validation
    )
    keys = jax.random.split(
        jax.random.PRNGKey(37), len(train_traces) + len(validation_traces)
    )
    train_batches = tuple(
        calibration._make_replay_batch(env, trace, keys[index])
        for index, trace in enumerate(train_traces)
    )
    validation_batches = tuple(
        calibration._make_replay_batch(
            env, trace, keys[len(train_traces) + index]
        )
        for index, trace in enumerate(validation_traces)
    )
    repeated_kp = np.broadcast_to(
        calibration.INITIAL_KP, (candidate_count, 7)
    ).copy()
    repeated_kd = np.broadcast_to(
        calibration.INITIAL_KD, (candidate_count, 7)
    ).copy()
    repeated_rmp_kp = np.broadcast_to(
        calibration.INITIAL_RMP_KP, (candidate_count, 7)
    ).copy()
    repeated_rmp_kd = np.broadcast_to(
        calibration.INITIAL_RMP_KD, (candidate_count, 7)
    ).copy()
    repeated_damping = np.broadcast_to(
        calibration.INITIAL_DAMPING, (candidate_count, 7)
    ).copy()
    repeated_armature = np.broadcast_to(
        calibration.INITIAL_ARMATURE, (candidate_count, 7)
    ).copy()

    best = {
        "process_var": 50.0,
        "measure_var": 1.0e-4,
        "velocity_ffd": 0.5,
        "acceleration_ffd": 0.0,
        "acceleration_weight": 0.001,
        "filter_length": 10.0,
    }
    tolerances = (np.deg2rad(0.1), 0.05, 1.0)
    weights = (1.0 / 3.0,) * 3
    stage_reports: list[dict[str, object]] = []

    def run_stage(
        name: str,
        first_name: str,
        first_values: list[float],
        second_name: str,
        second_values: list[float],
    ) -> None:
        candidates = _candidate_tuple(best, candidate_count)
        candidate_index = 1
        for first in first_values:
            for second in second_values:
                candidates[
                    {
                        "process_var": 0,
                        "measure_var": 1,
                        "velocity_ffd": 2,
                        "acceleration_ffd": 3,
                        "acceleration_weight": 4,
                        "filter_length": 5,
                    }[first_name]
                ][candidate_index, :] = first
                candidates[
                    {
                        "process_var": 0,
                        "measure_var": 1,
                        "velocity_ffd": 2,
                        "acceleration_ffd": 3,
                        "acceleration_weight": 4,
                        "filter_length": 5,
                    }[second_name]
                ][candidate_index, :] = second
                candidate_index += 1
        total_loss = np.zeros(candidate_count, dtype=np.float64)
        for batch in train_batches:
            outputs = calibration._run_candidates(
                env,
                replay_scan,
                batch,
                repeated_kp,
                repeated_kd,
                repeated_rmp_kp,
                repeated_rmp_kd,
                repeated_damping,
                repeated_armature,
                rmp_internal_candidates=candidates,
            )
            total_loss += calibration._candidate_losses(
                outputs, batch, weights, tolerances
            )
        total_loss /= len(train_batches)
        selected = int(np.argmin(total_loss))
        for key, values in zip(best, candidates):
            value = values[selected, 0]
            best[key] = float(int(value)) if key == "filter_length" else float(value)
        stage_reports.append(
            {
                "stage": name,
                "selected_candidate": selected,
                "loss": float(total_loss[selected]),
                "parameters": best.copy(),
            }
        )
        print(
            f"stage={name} candidate={selected} loss={total_loss[selected]:.6f} "
            f"parameters={best}",
            flush=True,
        )

    run_stage(
        "estimator_coarse",
        "process_var",
        [0.5, 50.0, 5000.0],
        "measure_var",
        [1.0e-6, 1.0e-4, 1.0e-2],
    )
    run_stage(
        "filter_velocity_coarse",
        "filter_length",
        [1.0, 10.0, 20.0],
        "velocity_ffd",
        [0.0, 0.5, 1.0],
    )
    run_stage(
        "acceleration_coarse",
        "acceleration_ffd",
        [-1.0, 0.0, 1.0],
        "acceleration_weight",
        [0.0, 0.01, 1.0],
    )
    run_stage(
        "estimator_fine",
        "process_var",
        [best["process_var"] * factor for factor in (0.5, 1.0, 2.0)],
        "measure_var",
        [best["measure_var"] * factor for factor in (0.5, 1.0, 2.0)],
    )
    run_stage(
        "filter_velocity_fine",
        "filter_length",
        [float(np.clip(best["filter_length"] + offset, 1, 35)) for offset in (-2, 0, 2)],
        "velocity_ffd",
        [best["velocity_ffd"] + offset for offset in (-0.2, 0.0, 0.2)],
    )
    run_stage(
        "acceleration_fine",
        "acceleration_ffd",
        [best["acceleration_ffd"] + offset for offset in (-0.5, 0.0, 0.5)],
        "acceleration_weight",
        [0.0, 0.001, 0.1],
    )

    final_candidates = _candidate_tuple(best, candidate_count)
    details: dict[str, object] = {}
    for batch in (*train_batches, *validation_batches):
        outputs = calibration._run_candidates(
            env,
            replay_scan,
            batch,
            repeated_kp,
            repeated_kd,
            repeated_rmp_kp,
            repeated_rmp_kd,
            repeated_damping,
            repeated_armature,
            rmp_internal_candidates=final_candidates,
        )
        details[batch.trace.source.parent.name] = calibration._metrics(
            outputs, batch, 0
        )
    report = {
        "strict_tolerances": {"q_rad": tolerances[0], "qd": tolerances[1], "qdd": tolerances[2]},
        "parameters": best,
        "stages": stage_reports,
        "metrics": details,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
