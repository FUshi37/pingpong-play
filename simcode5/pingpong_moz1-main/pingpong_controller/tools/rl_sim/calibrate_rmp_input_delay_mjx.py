#!/usr/bin/env python3
"""Identify command-publication to recovered-RMP consumption delay in MJX."""

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
        "--validation", type=Path, nargs="+", default=list(calibration.DEFAULT_VALIDATION)
    )
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("moz1_pd.xml"))
    parser.add_argument("--max-delay-steps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("rmp_input_delay_calibration.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_count = 26
    env = MjxJuggleEnv(
        args.model,
        n_envs=candidate_count,
        cfg=MjxJuggleConfig(
            domain_randomization=False,
            arm_action_limiter=False,
            recovered_rmp_motion_mode=True,
            actuator_cmd_filter=False,
            actuator_compensation_mode="none",
            enable_delay_conditioning=False,
        ),
    )
    replay_scan = calibration._make_scan(env)
    traces = tuple(
        calibration._slice_policy_region(
            calibration.load_control_recording(path), 100, 20
        )
        for path in (*args.train, *args.validation)
    )
    keys = jax.random.split(jax.random.PRNGKey(29), len(traces))
    batches = tuple(
        calibration._make_replay_batch(env, trace, key)
        for trace, key in zip(traces, keys)
    )
    train_count = len(args.train)

    def repeated(value: np.ndarray) -> np.ndarray:
        return np.broadcast_to(value, (candidate_count, 7)).copy()

    kp = repeated(calibration.INITIAL_KP)
    kd = repeated(calibration.INITIAL_KD)
    rmp_kp = repeated(calibration.INITIAL_RMP_KP)
    rmp_kd = repeated(calibration.INITIAL_RMP_KD)
    damping = repeated(calibration.INITIAL_DAMPING)
    armature = repeated(calibration.INITIAL_ARMATURE)
    delays = np.zeros((candidate_count, 7), dtype=np.int32)
    choices = list(range(max(0, int(args.max_delay_steps)) + 1))
    for index, delay in enumerate(choices, start=1):
        delays[index, :] = delay
    weights = (0.40, 0.30, 0.30)
    tolerances = (0.02, 0.25, 5.0)
    total_loss = np.zeros(candidate_count, dtype=np.float64)
    cached: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for batch_index, batch in enumerate(batches):
        outputs = calibration._run_candidates(
            env,
            replay_scan,
            batch,
            kp,
            kd,
            rmp_kp,
            rmp_kd,
            damping,
            armature,
            delays,
        )
        cached.append(outputs)
        if batch_index < train_count:
            total_loss += calibration._candidate_losses(
                outputs, batch, weights, tolerances
            )
    total_loss /= max(1, train_count)
    selected = int(np.argmin(total_loss))
    report = {
        "delay_candidates_control_steps": choices,
        "candidate_objective": {
            str(delay): float(total_loss[index])
            for index, delay in enumerate(choices, start=1)
        },
        "selected_delay_control_steps": int(delays[selected, 0]),
        "selected_delay_ms": float(delays[selected, 0] * 5.0),
        "metrics": {
            batch.trace.source.parent.name: calibration._metrics(
                outputs, batch, selected
            )
            for batch, outputs in zip(batches, cached)
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
