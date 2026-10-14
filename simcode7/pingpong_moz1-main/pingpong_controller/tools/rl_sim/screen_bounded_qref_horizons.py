#!/usr/bin/env python3
"""Screen per-joint bounded-QREF horizons with a fixed vertical stroke."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mjx_juggle_env import MjxJuggleEnv
from train_juggle_mjx_curriculum import (
    GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V14_PROFILE,
    build_curriculum,
)


DEFAULT_CANDIDATES_MS = (
    "uniform17p5=17.5,17.5,17.5,17.5,17.5,17.5,17.5",
    "feedback=13.10,13.10,14.60,13.35,12.35,12.30,12.55",
    "rmp_output=18,19,19,18,18,18,18",
    "combined=31.10,32.10,33.60,31.35,30.35,30.30,30.55",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate(value: str) -> tuple[str, tuple[float, ...]]:
    try:
        name, raw_values = value.split("=", maxsplit=1)
        horizons = tuple(float(item) for item in raw_values.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidate must be NAME=MS0,MS1,...,MS6"
        ) from exc
    if not name or len(horizons) != 7:
        raise argparse.ArgumentTypeError(
            "candidate must have a name and exactly seven horizons"
        )
    if not all(np.isfinite(item) and item > 0.0 for item in horizons):
        raise argparse.ArgumentTypeError(
            "candidate horizons must be positive and finite"
        )
    return name, horizons


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path("moz1_pd.xml"))
    parser.add_argument(
        "--candidate",
        action="append",
        type=_candidate,
        dest="candidates",
        help="Repeat NAME=MS0,...,MS6; defaults compare four evidence hypotheses.",
    )
    parser.add_argument(
        "--ball-mass-kg",
        type=float,
        nargs="+",
        default=(0.0025, 0.0037, 0.0040),
    )
    parser.add_argument(
        "--onset-ms",
        type=float,
        nargs="+",
        default=(80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0),
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-gpu", action="store_true")
    return parser.parse_args()


def _screen_case(
    *,
    xml_path: Path,
    name: str,
    horizons_ms: tuple[float, ...],
    ball_mass_kg: float,
    onset_ms: tuple[float, ...],
    steps: int,
    seed: int,
) -> list[dict[str, Any]]:
    stage = build_curriculum(
        curriculum_profile=GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V14_PROFILE,
        delay_ablation_preset="baseline_current",
        measured_ball_mass_range_kg=(ball_mass_kg, ball_mass_kg),
    )[0]
    cfg = replace(
        stage.cfg,
        recovered_rmp_qvel_reference_error_horizon_s=0.0,
        recovered_rmp_qvel_reference_error_horizon_s_per_joint=tuple(
            value * 1.0e-3 for value in horizons_ms
        ),
        falling_reset_time_to_contact_range_s=(0.24, 0.24),
        falling_reset_apex_height_range_m=(0.24, 0.24),
        dr_ball_mass_range=(ball_mass_kg, ball_mass_kg),
    )
    env = MjxJuggleEnv(xml_path, n_envs=len(onset_ms), cfg=cfg)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(onset_ms))
    state, _ = env.reset(keys)
    step_fn = jax.jit(env.step)
    direction = jnp.sign(
        env.racket_vertical_arm_jacobian * env.arm_vel_limit_rad_s
    )
    onset_steps = jnp.rint(
        jnp.asarray(onset_ms) / (env.dt * 1000.0)
    ).astype(jnp.int32)
    zeros = jnp.zeros((len(onset_ms),), dtype=jnp.float32)
    aggregate = {
        "confirmed_hits": zeros,
        "clearance_crossings": zeros,
        "physical_contacts": zeros,
        "max_ball_vz_m_s": jnp.full_like(zeros, -jnp.inf),
        "max_abs_racket_vz_m_s": zeros,
        "max_qvel_utilization": zeros,
        "max_qacc_utilization": zeros,
        "max_qvel_exceed_fraction": zeros,
        "max_qacc_exceed_fraction": zeros,
        "max_qref_error_norm_rad": zeros,
    }
    previous_racket_z = None
    done_any = jnp.zeros((len(onset_ms),), dtype=bool)
    for step_index in range(steps):
        active = step_index >= onset_steps
        action = jnp.where(active[:, None], direction[None, :], 0.0)
        state, _, _, done, metrics = step_fn(state, action)
        racket_z = metrics["racket_z"]
        if previous_racket_z is not None:
            racket_vz = (racket_z - previous_racket_z) / env.dt
            aggregate["max_abs_racket_vz_m_s"] = jnp.maximum(
                aggregate["max_abs_racket_vz_m_s"], jnp.abs(racket_vz)
            )
        previous_racket_z = racket_z
        aggregate["confirmed_hits"] += metrics["confirmed_hit"]
        aggregate["clearance_crossings"] += metrics[
            "launch_clearance_crossing"
        ]
        aggregate["physical_contacts"] += metrics["physical_contact_edge"]
        for output_name, metric_name in (
            ("max_ball_vz_m_s", "ball_vz"),
            ("max_qvel_utilization", "arm_qvel_limit_utilization_max"),
            ("max_qacc_utilization", "arm_qacc_limit_utilization_max"),
            ("max_qvel_exceed_fraction", "arm_qvel_limit_exceed_fraction"),
            ("max_qacc_exceed_fraction", "arm_qacc_limit_exceed_fraction"),
            ("max_qref_error_norm_rad", "qvel_reference_error_norm"),
        ):
            aggregate[output_name] = jnp.maximum(
                aggregate[output_name], metrics[metric_name]
            )
        done_any |= done

    host = jax.device_get(aggregate)
    host_done = np.asarray(jax.device_get(done_any))
    rows: list[dict[str, Any]] = []
    for index, onset in enumerate(onset_ms):
        row: dict[str, Any] = {
            "candidate": name,
            "horizons_ms": list(horizons_ms),
            "ball_mass_kg": ball_mass_kg,
            "onset_ms": onset,
            "done": bool(host_done[index]),
        }
        for metric_name, values in host.items():
            row[metric_name] = float(np.asarray(values)[index])
        rows.append(row)
    return rows


def main() -> None:
    args = _parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX GPU required, got {jax.devices()}")
    candidates = tuple(args.candidates or map(_candidate, DEFAULT_CANDIDATES_MS))
    masses = tuple(float(value) for value in args.ball_mass_kg)
    onsets = tuple(float(value) for value in args.onset_ms)
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if any(not np.isfinite(value) or value <= 0.0 for value in masses):
        raise ValueError("ball masses must be positive and finite")
    if any(not np.isfinite(value) or value < 0.0 for value in onsets):
        raise ValueError("onset times must be non-negative and finite")

    rows: list[dict[str, Any]] = []
    for name, horizons_ms in candidates:
        for mass in masses:
            print(
                f"screen candidate={name} mass={mass:g} kg "
                f"horizons_ms={horizons_ms}",
                flush=True,
            )
            rows.extend(
                _screen_case(
                    xml_path=xml_path,
                    name=name,
                    horizons_ms=horizons_ms,
                    ball_mass_kg=mass,
                    onset_ms=onsets,
                    steps=int(args.steps),
                    seed=int(args.seed),
                )
            )

    summary: list[dict[str, Any]] = []
    for name, horizons_ms in candidates:
        selected = [row for row in rows if row["candidate"] == name]
        mass_success = {
            f"{mass:.7f}": sum(
                row["confirmed_hits"] > 0.0
                for row in selected
                if row["ball_mass_kg"] == mass
            )
            for mass in masses
        }
        summary.append(
            {
                "candidate": name,
                "horizons_ms": list(horizons_ms),
                "successful_onsets_by_mass": mass_success,
                "minimum_mass_successes": min(mass_success.values()),
                "successful_cases": sum(
                    row["confirmed_hits"] > 0.0 for row in selected
                ),
                "total_cases": len(selected),
                "max_abs_racket_vz_m_s": max(
                    row["max_abs_racket_vz_m_s"] for row in selected
                ),
                "max_qvel_utilization": max(
                    row["max_qvel_utilization"] for row in selected
                ),
                "max_qacc_utilization": max(
                    row["max_qacc_utilization"] for row in selected
                ),
                "max_qvel_exceed_fraction": max(
                    row["max_qvel_exceed_fraction"] for row in selected
                ),
                "max_qacc_exceed_fraction": max(
                    row["max_qacc_exceed_fraction"] for row in selected
                ),
                "max_qref_error_norm_rad": max(
                    row["max_qref_error_norm_rad"] for row in selected
                ),
            }
        )

    rmp_config = (
        Path(__file__).resolve().parents[3]
        / "ReactiveMotionPlanner/rmp-recovery/recovered_rmp/configs/"
        "rmp_equiv_refine_baseline.json"
    )
    feedback_report = (
        Path(__file__).resolve().parents[3]
        / "ReactiveMotionPlanner/rmp-recovery/recovered_rmp/calibration/"
        "datatracer_203806_rmp_and_feedback_validation.json"
    )
    payload = {
        "contract": {
            "profile_source": GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V14_PROFILE,
            "selection_rule": (
                "Require at least 4/7 confirmed onset lanes at every mass, "
                "zero qvel exceedance, qacc exceedance <=0.01; among passing "
                "physically evidenced candidates prefer the componentwise "
                "smallest horizon. Combined delay is diagnostic only."
            ),
            "ball_mass_kg": masses,
            "onset_ms": onsets,
            "steps": int(args.steps),
            "seed": int(args.seed),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "xml": str(xml_path),
            "xml_sha256": _sha256(xml_path),
            "rmp_config": str(rmp_config),
            "rmp_config_sha256": _sha256(rmp_config),
            "feedback_report": str(feedback_report),
            "feedback_report_sha256": _sha256(feedback_report),
        },
        "summary": summary,
        "rows": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=False)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
