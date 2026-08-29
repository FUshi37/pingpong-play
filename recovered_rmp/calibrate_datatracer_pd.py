#!/usr/bin/env python3
"""Fit seven ideal-PD gains to a 1 kHz Movax DataTracer trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from recovered_rmp.replay_datatracer_mujoco import _metrics, replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--start-s", type=float, default=1.410)
    parser.add_argument("--end-s", type=float, default=5.000)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = args.trace.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    with np.load(trace_path) as archive:
        time = np.asarray(archive["time"], dtype=float)
    relative_time = time - time[0]
    mask = (relative_time >= args.start_s) & (relative_time < args.end_s)
    evaluations = 0
    best_loss = np.inf
    best_parameters: np.ndarray | None = None

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        nonlocal evaluations, best_loss, best_parameters
        parameters = np.exp(log_parameters)
        result = replay(
            trace_path,
            model_path,
            kp_scale=parameters[:7],
            kd_scale=parameters[7:],
            damping_scale=np.ones(7),
            armature_scale=np.ones(7),
            acceleration_feedforward_s2=np.zeros(7),
        )
        value = np.concatenate(
            (
                ((result["sim_q"] - result["actual_q"])[mask] / np.deg2rad(0.1)).ravel(),
                ((result["sim_qd"] - result["actual_qd"])[mask] / 0.05).ravel(),
                ((result["sim_qdd"] - result["actual_qdd"])[mask] / 1.0).ravel(),
            )
        )
        value = np.nan_to_num(value, nan=1.0e6, posinf=1.0e6, neginf=-1.0e6)
        loss = float(np.sqrt(np.mean(value**2)))
        evaluations += 1
        if loss < best_loss:
            best_loss = loss
            best_parameters = parameters.copy()
            print(
                f"evaluation={evaluations} normalized_rms={loss:.6f} "
                f"kp={parameters[:7].round(6).tolist()} "
                f"kd={parameters[7:].round(6).tolist()}",
                flush=True,
            )
        return value

    initial = np.concatenate((np.full(7, 0.02), np.full(7, 0.10)))
    fit = least_squares(
        residual,
        np.log(initial),
        bounds=(
            np.log(np.concatenate((np.full(7, 0.001), np.full(7, 0.01)))),
            np.log(np.concatenate((np.full(7, 0.8), np.full(7, 2.0)))),
        ),
        max_nfev=int(args.max_evaluations),
        verbose=2,
    )
    parameters = np.exp(fit.x) if best_parameters is None else best_parameters
    result = replay(
        trace_path,
        model_path,
        kp_scale=parameters[:7],
        kd_scale=parameters[7:],
        damping_scale=np.ones(7),
        armature_scale=np.ones(7),
        acceleration_feedforward_s2=np.zeros(7),
    )
    report = {
        "success": bool(fit.success),
        "message": str(fit.message),
        "function_evaluations": evaluations,
        "normalized_rms": best_loss,
        "kp_scale_per_joint": parameters[:7].tolist(),
        "kd_scale_per_joint": parameters[7:].tolist(),
        "effective_kp": result["kp"].tolist(),
        "effective_kd": result["kd"].tolist(),
        "metrics": _metrics(result, args.start_s, args.end_s),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
