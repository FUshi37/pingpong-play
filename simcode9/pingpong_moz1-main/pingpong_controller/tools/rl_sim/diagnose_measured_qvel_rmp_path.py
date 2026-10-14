#!/usr/bin/env python3
"""Diagnose measured-QVEL/RMP authority and first-hit reward credit."""

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
    GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V8_PROFILE,
    GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V12_PROFILE,
    build_curriculum,
)


REWARD_METRICS = (
    "reward/dense_scaled",
    "reward/ball_height",
    "reward/pre_hit_intercept",
    "reward/action_penalty",
    "reward/action_delta_penalty",
    "reward/arm_velocity_usage_penalty",
    "reward/arm_acceleration_usage_penalty",
    "reward/hit_bonus",
    "reward/hit_count_floor_reward",
    "reward/center_flat_hit",
    "reward/post_hit_survival",
    "reward/failed_hit_penalty",
    "reward/ball_miss_termination_penalty",
    "reward/racket_z_limit_termination_penalty",
    "reward/racket_anchor_termination_penalty",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=Path("moz1_pd.xml"))
    parser.add_argument("--ball-mass-kg", type=float, default=0.0037)
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


def _case_config(
    profile: str,
    lead_s: float,
    ball_mass_kg: float,
):
    stage = build_curriculum(
        curriculum_profile=profile,
        delay_ablation_preset="baseline_current",
        measured_ball_mass_range_kg=(ball_mass_kg, ball_mass_kg),
    )[0]
    return replace(
        stage.cfg,
        action_velocity_scale=1.0,
        recovered_rmp_qvel_target_lead_s=lead_s,
        falling_reset_time_to_contact_range_s=(0.24, 0.24),
        falling_reset_apex_height_range_m=(0.24, 0.24),
        dr_ball_mass_range=(ball_mass_kg, ball_mass_kg),
    )


def _run_case(
    *,
    xml_path: Path,
    profile: str,
    lead_s: float,
    action_mode: str,
    ball_mass_kg: float,
    onset_ms: tuple[float, ...],
    steps: int,
    seed: int,
) -> dict[str, Any]:
    cfg = _case_config(profile, lead_s, ball_mass_kg)
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
    n_envs = len(onset_ms)
    zeros = jnp.zeros((n_envs,), dtype=jnp.float32)
    totals = {name: zeros for name in REWARD_METRICS}
    total_reward = zeros
    contacts = zeros
    clearances = zeros
    confirmed_hits = zeros
    done_seen = jnp.zeros((n_envs,), dtype=bool)
    first_done_step = jnp.full((n_envs,), steps, dtype=jnp.int32)
    max_qvel_utilization = zeros
    max_qacc_utilization = zeros
    max_racket_vz = jnp.full((n_envs,), -jnp.inf, dtype=jnp.float32)
    max_ball_vz = jnp.full((n_envs,), -jnp.inf, dtype=jnp.float32)
    max_internal_measured_q_error = zeros
    desired_qvel_norm = jnp.linalg.norm(env.arm_vel_limit_rad_s)
    sum_target_estimator_gain = zeros
    sum_rmp_output_gain = zeros
    sum_actual_qvel_gain = zeros
    gain_samples = zeros

    for step_index in range(steps):
        alive = ~done_seen
        active = step_index >= onset_steps
        if action_mode == "vertical_max":
            action = jnp.where(active[:, None], direction[None, :], 0.0)
        elif action_mode == "zero":
            action = jnp.zeros((n_envs, 7), dtype=jnp.float32)
        else:
            raise ValueError(f"unsupported action mode: {action_mode}")
        previous_racket_z = state.data.site_xpos[:, env.racket_site_id, 2]
        state, _, reward, done, metrics = step_fn(state, action)
        alive_f = alive.astype(jnp.float32)
        total_reward += reward * alive_f
        for name in REWARD_METRICS:
            totals[name] += metrics[name] * alive_f
        contacts += metrics["physical_contact_edge"] * alive_f
        clearances += metrics["launch_clearance_crossing"] * alive_f
        confirmed_hits += metrics["confirmed_hit"] * alive_f
        max_qvel_utilization = jnp.maximum(
            max_qvel_utilization,
            metrics["arm_qvel_limit_utilization_max"] * alive_f,
        )
        max_qacc_utilization = jnp.maximum(
            max_qacc_utilization,
            metrics["arm_qacc_limit_utilization_max"] * alive_f,
        )
        racket_vz = (
            state.data.site_xpos[:, env.racket_site_id, 2]
            - previous_racket_z
        ) / env.dt
        max_racket_vz = jnp.maximum(
            max_racket_vz,
            jnp.where(alive, racket_vz, -jnp.inf),
        )
        max_ball_vz = jnp.maximum(
            max_ball_vz,
            jnp.where(alive, metrics["ball_vz"], -jnp.inf),
        )
        measured_q = state.data.qpos[:, env.arm_qadr]
        measured_qvel = state.data.qvel[:, env.arm_vadr]
        internal_q_error = jnp.linalg.norm(
            state.rmp_state.q - measured_q,
            axis=-1,
        )
        max_internal_measured_q_error = jnp.maximum(
            max_internal_measured_q_error,
            internal_q_error * alive_f,
        )
        gain_active = alive & active & (action_mode == "vertical_max")
        gain_active_f = gain_active.astype(jnp.float32)
        sum_target_estimator_gain += (
            jnp.linalg.norm(state.rmp_state.target_x[..., 1], axis=-1)
            / desired_qvel_norm
            * gain_active_f
        )
        sum_rmp_output_gain += (
            metrics["rmp_qd_norm"] / desired_qvel_norm * gain_active_f
        )
        sum_actual_qvel_gain += (
            jnp.linalg.norm(measured_qvel, axis=-1)
            / desired_qvel_norm
            * gain_active_f
        )
        gain_samples += gain_active_f
        newly_done = alive & done
        first_done_step = jnp.where(
            newly_done,
            jnp.asarray(step_index + 1, dtype=jnp.int32),
            first_done_step,
        )
        done_seen |= done

    def host(values: jax.Array) -> np.ndarray:
        return np.asarray(jax.device_get(values))

    safe_gain_samples = jnp.maximum(gain_samples, 1.0)
    arrays = {
        "return": host(total_reward),
        "physical_contacts": host(contacts),
        "clearance_crossings": host(clearances),
        "confirmed_hits": host(confirmed_hits),
        "episode_steps": host(first_done_step),
        "max_qvel_utilization": host(max_qvel_utilization),
        "max_qacc_utilization": host(max_qacc_utilization),
        "max_racket_vz_m_s": host(max_racket_vz),
        "max_ball_vz_m_s": host(max_ball_vz),
        "max_internal_measured_q_error_rad": host(
            max_internal_measured_q_error
        ),
        "mean_target_estimator_velocity_gain": host(
            sum_target_estimator_gain / safe_gain_samples
        ),
        "mean_rmp_output_velocity_gain": host(
            sum_rmp_output_gain / safe_gain_samples
        ),
        "mean_actual_velocity_gain": host(
            sum_actual_qvel_gain / safe_gain_samples
        ),
    }
    reward_arrays = {name: host(value) for name, value in totals.items()}
    rows: list[dict[str, Any]] = []
    for index, onset in enumerate(onset_ms):
        row = {"onset_ms": onset}
        row.update({name: float(value[index]) for name, value in arrays.items()})
        row["reward_terms"] = {
            name.removeprefix("reward/"): float(value[index])
            for name, value in reward_arrays.items()
        }
        rows.append(row)
    successful = host(confirmed_hits) > 0.0
    return {
        "profile": profile,
        "action_mode": action_mode,
        "configured_lead_ms": lead_s * 1000.0,
        "effective_lead_ms": env.recovered_rmp_qvel_target_lead_s * 1000.0,
        "summary": {
            "successful_onsets": int(successful.sum()),
            "total_onsets": n_envs,
            "mean_return": float(host(total_reward).mean()),
            "mean_return_successful": (
                float(host(total_reward)[successful].mean())
                if successful.any()
                else None
            ),
            "mean_return_unsuccessful": (
                float(host(total_reward)[~successful].mean())
                if (~successful).any()
                else None
            ),
            "max_qvel_utilization": float(host(max_qvel_utilization).max()),
            "max_qacc_utilization": float(host(max_qacc_utilization).max()),
            "max_racket_vz_m_s": float(host(max_racket_vz).max()),
            "max_ball_vz_m_s": float(host(max_ball_vz).max()),
            "max_internal_measured_q_error_rad": float(
                host(max_internal_measured_q_error).max()
            ),
            "mean_target_estimator_velocity_gain": float(
                host(sum_target_estimator_gain).sum()
                / max(float(host(gain_samples).sum()), 1.0)
            ),
            "mean_rmp_output_velocity_gain": float(
                host(sum_rmp_output_gain).sum()
                / max(float(host(gain_samples).sum()), 1.0)
            ),
            "mean_actual_velocity_gain": float(
                host(sum_actual_qvel_gain).sum()
                / max(float(host(gain_samples).sum()), 1.0)
            ),
        },
        "rows": rows,
    }


def main() -> None:
    args = _parse_args()
    xml_path = args.xml.resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"JAX GPU required, got {jax.devices()}")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.ball_mass_kg <= 0.0:
        raise ValueError("--ball-mass-kg must be positive")
    onset_ms = tuple(float(value) for value in args.onset_ms)
    cases = (
        (
            "v8_old_reward_lead17p5",
            GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V8_PROFILE,
            0.0175,
            "vertical_max",
        ),
        (
            "v12_repaired_reward_lead17p5",
            GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V12_PROFILE,
            0.0175,
            "vertical_max",
        ),
        (
            "v12_repaired_reward_effective5",
            GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V12_PROFILE,
            0.0,
            "vertical_max",
        ),
        (
            "v12_repaired_reward_zero_action",
            GOAL_D455_MEASURED_QVEL_RMP_VERTICAL_V12_PROFILE,
            0.0175,
            "zero",
        ),
    )
    results = {}
    for name, profile, lead_s, action_mode in cases:
        print(f"running {name}", flush=True)
        results[name] = _run_case(
            xml_path=xml_path,
            profile=profile,
            lead_s=lead_s,
            action_mode=action_mode,
            ball_mass_kg=float(args.ball_mass_kg),
            onset_ms=onset_ms,
            steps=int(args.steps),
            seed=int(args.seed),
        )
    payload = {
        "contract": {
            "ball_mass_kg": float(args.ball_mass_kg),
            "onset_ms": onset_ms,
            "steps": int(args.steps),
            "seed": int(args.seed),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "xml": str(xml_path),
            "xml_sha256": _sha256(xml_path),
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {name: value["summary"] for name, value in results.items()},
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
