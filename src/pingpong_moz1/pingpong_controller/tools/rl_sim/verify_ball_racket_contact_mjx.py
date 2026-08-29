#!/usr/bin/env python3
"""Cross-check identified ball/racket contacts between CPU MuJoCo and MJX."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

import identify_ball_racket_contact as identification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis_results/ball_racket_contact_identification"),
    )
    return parser.parse_args()


def main() -> None:
    try:
        import jax
        from jax import lax
        from mujoco import mjx
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MJX dependencies are required. Install matching mujoco and "
            "mujoco-mjx versions plus jax[cpu]."
        ) from exc

    jax.config.update("jax_enable_x64", True)
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    event_fields = {item.name for item in fields(identification.ImpactEvent)}
    events = [
        identification.ImpactEvent(**{key: value for key, value in row.items() if key in event_fields})
        for row in summary["events"]
    ]
    params = summary["recommended_parameters"]
    friction = float(params["pair_sliding_friction"])
    solref_time_s = float(params["solref_time_s"])
    damping = float(params["solref_damping_ratio"])

    experiment = identification.IsolatedContactExperiment()
    cpu_proxy_damping = float(params.get("cpu_proxy_solref_damping_ratio", damping))
    experiment.model.pair_friction[experiment.pair_id, 0] = friction
    experiment.model.pair_solref[experiment.pair_id] = (solref_time_s, cpu_proxy_damping)
    mjx_model = mjx.put_model(experiment.model)
    base_data = mjx.make_data(mjx_model)
    step_count = max(1, int(round(0.030 / experiment.model.opt.timestep)))

    @jax.jit
    def rollout_batch(damping_value, qpos_batch, qvel_batch):
        model = mjx_model.replace(
            pair_solref=mjx_model.pair_solref.at[experiment.pair_id, 1].set(damping_value)
        )

        def rollout(qpos, qvel):
            data = base_data.replace(qpos=qpos, qvel=qvel)

            def step(data, _):
                data = mjx.step(model, data)
                return data, None

            data, _ = lax.scan(step, data, xs=None, length=step_count)
            return data.qvel[experiment.ball_vadr:experiment.ball_vadr + 3]

        return jax.vmap(rollout)(qpos_batch, qvel_batch)

    qpos_rows = []
    qvel_rows = []
    for event in events:
        incoming = np.asarray(event.pre_relative_local_m_s, dtype=np.float64)
        lateral = np.asarray(event.tangent_offset_m, dtype=np.float64)
        normal_speed = max(0.10, -float(incoming[2]))
        travel_time = 0.0020
        clearance = normal_speed * travel_time
        qpos = np.asarray(base_data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(base_data.qvel, dtype=np.float64).copy()
        start_xy = lateral - incoming[:2] * travel_time
        qpos[experiment.ball_qadr:experiment.ball_qadr + 3] = (
            start_xy[0],
            start_xy[1],
            identification.BALL_RADIUS_M + 0.001 + clearance,
        )
        qpos[experiment.ball_qadr + 3:experiment.ball_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        qvel[experiment.ball_vadr:experiment.ball_vadr + 3] = incoming
        qvel[experiment.ball_vadr + 3:experiment.ball_vadr + 6] = 0.0

        qpos_rows.append(qpos)
        qvel_rows.append(qvel)

    qpos_batch = jax.numpy.asarray(np.asarray(qpos_rows))
    qvel_batch = jax.numpy.asarray(np.asarray(qvel_rows))
    targets = np.asarray([event.post_relative_local_m_s for event in events], dtype=np.float64)
    groups = np.asarray([event.group for event in events])
    calibration_mask = groups != "record_new5"

    def damping_objective(damping_value: float) -> float:
        predictions = np.asarray(
            rollout_batch(float(damping_value), qpos_batch, qvel_batch), dtype=np.float64
        )
        normal_error = (predictions[:, 2] - targets[:, 2]) / 0.18
        tangent_error = np.linalg.norm(
            predictions[:, :2] - targets[:, :2], axis=1
        ) / 0.25
        squared = normal_error * normal_error + tangent_error * tangent_error
        losses = np.where(squared <= 4.0, squared, 4.0 * np.sqrt(squared) - 4.0)
        return float(np.mean([
            np.mean(losses[groups == group]) for group in ("record_new3", "record_new4")
        ]))

    fit = minimize_scalar(
        damping_objective,
        bounds=(0.10, 0.35),
        method="bounded",
        options={"xatol": 1.0e-6},
    )
    damping = float(fit.x)
    mjx_array = np.asarray(
        rollout_batch(damping, qpos_batch, qvel_batch), dtype=np.float64
    )
    cpu_array = np.asarray([
        experiment.simulate(event, friction, solref_time_s, damping) for event in events
    ])
    rows = []
    for event, target, cpu, mjx_prediction in zip(events, targets, cpu_array, mjx_array):
        rows.append({
            "group": event.group,
            "recording": event.recording,
            "impact_time_s": event.impact_time_s,
            "target_m_s": target.tolist(),
            "cpu_mujoco_m_s": cpu.tolist(),
            "mjx_m_s": mjx_prediction.tolist(),
            "cpu_mjx_delta_m_s": (mjx_prediction - cpu).tolist(),
        })

    delta = mjx_array - cpu_array

    def target_metrics(predictions: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
        error = predictions[mask] - targets[mask]
        incoming = np.asarray([
            event.pre_relative_local_m_s for event in events
        ])[mask]
        incoming_normal = incoming[:, 2]
        velocity_jump = predictions[mask] - incoming
        return {
            "count": int(np.sum(mask)),
            "velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum(error ** 2, axis=1)))),
            "normal_rmse_m_s": float(np.sqrt(np.mean(error[:, 2] ** 2)),),
            "tangent_rmse_m_s": float(
                np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))
            ),
            "predicted_restitution_median": float(
                np.median(-predictions[mask, 2] / incoming_normal)
            ),
            "predicted_effective_impulse_friction_ratio_median": float(
                np.median(
                    np.linalg.norm(velocity_jump[:, :2], axis=1)
                    / np.maximum(np.abs(velocity_jump[:, 2]), 1.0e-9)
                )
            ),
        }

    all_mask = np.ones(len(events), dtype=bool)
    holdout_mask = ~calibration_mask
    result = {
        "schema_version": 1,
        "count": len(events),
        "steps_per_impact": step_count,
        "timestep_s": float(experiment.model.opt.timestep),
        "mujoco_version": mujoco.__version__,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "parameters": {
            "ball_mass_kg": float(params["ball_mass_kg"]),
            "pair_sliding_friction": friction,
            "solref_time_s": solref_time_s,
            "solref_damping_ratio": damping,
        },
        "fit": {
            "backend": "MJX",
            "groups": ["record_new3", "record_new4"],
            "holdout_group": "record_new5",
            "cpu_proxy_solref_damping_ratio": cpu_proxy_damping,
            "objective": float(fit.fun),
            "optimizer_success": bool(fit.success),
            "optimizer_message": str(fit.message),
            "note": (
                "friction and solref time are held at simcode7 priors because ball spin "
                "and contact force/duration are not observed"
            ),
        },
        "cpu_mjx_velocity_rms_delta_m_s": float(np.sqrt(np.mean(delta ** 2))),
        "cpu_mjx_max_component_delta_m_s": float(np.max(np.abs(delta))),
        "cpu_target_velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.sum((cpu_array - targets) ** 2, axis=1)))
        ),
        "mjx_target_velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.sum((mjx_array - targets) ** 2, axis=1)))
        ),
        "mjx_target_metrics": {
            "all": target_metrics(mjx_array, all_mask),
            "calibration": target_metrics(mjx_array, calibration_mask),
            "holdout": target_metrics(mjx_array, holdout_mask),
        },
        "rows": rows,
    }
    (analysis_dir / "mjx_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary["recommended_parameters"]["solref_damping_ratio"] = damping
    summary["recommended_parameters"]["fit_backend"] = "MJX 3.3.7 direct"
    summary["recommended_parameters"]["cpu_proxy_solref_damping_ratio"] = cpu_proxy_damping
    summary["optimization"]["mjx_constrained_damping_fit"] = result["fit"]
    calibration_events = [event for event in events if event.group != "record_new5"]
    holdout_events = [event for event in events if event.group == "record_new5"]
    final_parameters = (friction, solref_time_s, damping)
    summary["simulation_validation"]["fitted"] = {
        "all": identification._event_prediction_metrics(
            experiment, events, final_parameters
        ),
        "calibration": identification._event_prediction_metrics(
            experiment, calibration_events, final_parameters
        ),
        "holdout": identification._event_prediction_metrics(
            experiment, holdout_events, final_parameters
        ),
    }
    summary["mjx_validation"] = {key: value for key, value in result.items() if key != "rows"}
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (analysis_dir / "recommended_contact_params.json").write_text(
        json.dumps(summary["recommended_parameters"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    identification._write_markdown(
        analysis_dir / "SIM2REAL_BALL_RACKET_CONTACT_ANALYSIS_ZH.md", summary
    )
    print(json.dumps(summary["mjx_validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
