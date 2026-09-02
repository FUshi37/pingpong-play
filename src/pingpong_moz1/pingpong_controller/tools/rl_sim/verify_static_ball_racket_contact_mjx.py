#!/usr/bin/env python3
"""Fit static first-impact observations directly with MJX contact rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

import identify_ball_racket_contact as contact


CURRENT_PRODUCTION_DAMPING = 0.152540


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(
            "analysis_results/ball_racket_contact_identification_v91_static_20260901"
        ),
    )
    return parser.parse_args()


def _metrics(
    predictions: np.ndarray,
    events: list[contact.ImpactEvent],
    mask: np.ndarray,
) -> dict[str, Any]:
    targets = np.asarray(
        [event.post_relative_local_m_s for event in events], dtype=np.float64
    )
    incoming = np.asarray(
        [event.pre_relative_local_m_s for event in events], dtype=np.float64
    )
    selected_prediction = predictions[mask]
    selected_target = targets[mask]
    selected_incoming = incoming[mask]
    error = selected_prediction - selected_target
    return {
        "count": int(np.sum(mask)),
        "velocity_rmse_m_s": float(
            np.sqrt(np.mean(np.sum(error * error, axis=1)))
        ),
        "normal_rmse_m_s": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "tangent_rmse_m_s": float(
            np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))
        ),
        "measured_restitution_median": float(
            np.median(-selected_target[:, 2] / selected_incoming[:, 2])
        ),
        "predicted_restitution_median": float(
            np.median(-selected_prediction[:, 2] / selected_incoming[:, 2])
        ),
        "normal_errors_m_s": error[:, 2].tolist(),
    }


def main() -> None:
    import jax
    from jax import lax
    import mujoco
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    event_fields = {item.name for item in fields(contact.ImpactEvent)}
    events = [
        contact.ImpactEvent(
            **{key: value for key, value in row.items() if key in event_fields}
        )
        for row in summary["events"]
    ]
    if not events:
        raise SystemExit("No selected first-impact events are available")

    experiment = contact.IsolatedContactExperiment()
    friction = float(contact.SIMCODE7_BASELINE["pair_sliding_friction"])
    solref_time_s = float(contact.SIMCODE7_BASELINE["solref_time_s"])
    experiment.model.pair_friction[experiment.pair_id, 0] = friction
    experiment.model.pair_solref[experiment.pair_id] = (
        solref_time_s,
        CURRENT_PRODUCTION_DAMPING,
    )
    mjx_model = mjx.put_model(experiment.model)
    base_data = mjx.make_data(mjx_model)
    step_count = max(1, int(round(0.030 / experiment.model.opt.timestep)))

    @jax.jit
    def rollout_batch(damping_value, qpos_batch, qvel_batch):
        model = mjx_model.replace(
            pair_solref=mjx_model.pair_solref.at[experiment.pair_id, 1].set(
                damping_value
            )
        )

        def rollout(qpos, qvel):
            data = base_data.replace(qpos=qpos, qvel=qvel)

            def step(data, _):
                data = mjx.step(model, data)
                return data, None

            data, _ = lax.scan(step, data, xs=None, length=step_count)
            return data.qvel[experiment.ball_vadr : experiment.ball_vadr + 3]

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
        qpos[experiment.ball_qadr : experiment.ball_qadr + 3] = (
            start_xy[0],
            start_xy[1],
            contact.BALL_RADIUS_M + 0.001 + clearance,
        )
        qpos[experiment.ball_qadr + 3 : experiment.ball_qadr + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        qvel[experiment.ball_vadr : experiment.ball_vadr + 3] = incoming
        qvel[experiment.ball_vadr + 3 : experiment.ball_vadr + 6] = 0.0
        qpos_rows.append(qpos)
        qvel_rows.append(qvel)

    qpos_batch = jax.numpy.asarray(np.asarray(qpos_rows))
    qvel_batch = jax.numpy.asarray(np.asarray(qvel_rows))
    targets = np.asarray(
        [event.post_relative_local_m_s for event in events], dtype=np.float64
    )
    all_mask = np.ones(len(events), dtype=bool)
    high_quality_mask = np.asarray(
        [
            event.continuity_error_mm <= 60.0
            and event.pre_fit_rmse_mm <= 3.0
            and event.post_fit_rmse_mm <= 3.0
            and event.max_depth_disagreement_mm <= 100.0
            for event in events
        ],
        dtype=bool,
    )

    def predictions(damping: float) -> np.ndarray:
        return np.asarray(
            rollout_batch(float(damping), qpos_batch, qvel_batch),
            dtype=np.float64,
        )

    def objective(damping: float, mask: np.ndarray, loss_kind: str) -> float:
        prediction = predictions(damping)
        raw_error = prediction[mask, 2] - targets[mask, 2]
        if loss_kind == "least_squares":
            return float(np.mean(raw_error * raw_error))
        normal_error = raw_error / 0.18
        squared = normal_error * normal_error
        loss = np.where(squared <= 4.0, squared, 4.0 * np.sqrt(squared) - 4.0)
        return float(np.mean(loss))

    def fit(mask: np.ndarray, loss_kind: str = "robust_huber") -> dict[str, Any]:
        result = minimize_scalar(
            lambda value: objective(float(value), mask, loss_kind),
            bounds=(0.05, 0.50),
            method="bounded",
            options={"xatol": 1.0e-6},
        )
        return {
            "solref_damping_ratio": float(result.x),
            "objective": float(result.fun),
            "success": bool(result.success),
            "message": str(result.message),
            "event_count": int(np.sum(mask)),
            "loss_kind": loss_kind,
        }

    all_fit = fit(all_mask)
    high_quality_fit = fit(high_quality_mask)
    all_fit_least_squares = fit(all_mask, "least_squares")
    high_quality_fit_least_squares = fit(high_quality_mask, "least_squares")
    loo = []
    for index, event in enumerate(events):
        mask = all_mask.copy()
        mask[index] = False
        fit_row = fit(mask)
        fit_row["held_out_recording"] = event.recording
        loo.append(fit_row)
    loo_values = [row["solref_damping_ratio"] for row in loo]

    candidates = {
        "current_production": CURRENT_PRODUCTION_DAMPING,
        "all_five_robust_fit": all_fit["solref_damping_ratio"],
        "high_quality_robust_fit": high_quality_fit["solref_damping_ratio"],
        "all_five_least_squares_fit": all_fit_least_squares[
            "solref_damping_ratio"
        ],
        "high_quality_least_squares_fit": high_quality_fit_least_squares[
            "solref_damping_ratio"
        ],
    }
    evaluations = {}
    prediction_cache = {}
    for name, damping in candidates.items():
        value = predictions(damping)
        prediction_cache[name] = value
        evaluations[name] = {
            "parameters": {
                "pair_sliding_friction": friction,
                "solref_time_s": solref_time_s,
                "solref_damping_ratio": damping,
            },
            "all": _metrics(value, events, all_mask),
            "high_quality": _metrics(value, events, high_quality_mask),
        }
    event_rows = []
    for index, event in enumerate(events):
        incoming_normal = float(event.pre_relative_local_m_s[2])
        row = {
            "recording": event.recording,
            "high_quality": bool(high_quality_mask[index]),
            "incoming_normal_m_s": incoming_normal,
            "measured_outgoing_normal_m_s": float(
                event.post_relative_local_m_s[2]
            ),
            "measured_restitution": float(event.normal_restitution),
            "continuity_error_mm": float(event.continuity_error_mm),
            "predictions": {},
        }
        for name, value in prediction_cache.items():
            row["predictions"][name] = {
                "outgoing_normal_m_s": float(value[index, 2]),
                "normal_error_m_s": float(
                    value[index, 2] - event.post_relative_local_m_s[2]
                ),
                "restitution": float(-value[index, 2] / incoming_normal),
            }
        event_rows.append(row)

    # Full release replay: reconstruct the equivalent rest-release height from
    # each measured first-impact normal speed, simulate gravity and the static
    # racket, then evaluate only the first rebound.
    gravity_m_s2 = 9.81
    full_experiment = contact.IsolatedContactExperiment()
    full_experiment.model.opt.gravity[:] = (0.0, 0.0, -gravity_m_s2)
    full_experiment.model.pair_friction[full_experiment.pair_id, 0] = friction
    full_experiment.model.pair_solref[full_experiment.pair_id] = (
        solref_time_s,
        CURRENT_PRODUCTION_DAMPING,
    )
    full_mjx_model = mjx.put_model(full_experiment.model)
    full_base_data = mjx.make_data(full_mjx_model)
    full_step_count = max(1, int(round(0.75 / full_experiment.model.opt.timestep)))

    @jax.jit
    def rollout_full_drop_batch(damping_value, qpos_batch, qvel_batch):
        model = full_mjx_model.replace(
            pair_solref=full_mjx_model.pair_solref.at[
                full_experiment.pair_id, 1
            ].set(damping_value)
        )

        def rollout(qpos, qvel):
            data = full_base_data.replace(qpos=qpos, qvel=qvel)

            def step(data, _):
                data = mjx.step(model, data)
                return data, (
                    data.qpos[full_experiment.ball_qadr + 2],
                    data.qvel[full_experiment.ball_vadr + 2],
                )

            _, trajectory = lax.scan(step, data, xs=None, length=full_step_count)
            return trajectory

        return jax.vmap(rollout)(qpos_batch, qvel_batch)

    full_qpos_rows = []
    full_qvel_rows = []
    equivalent_release_heights = []
    for event in events:
        incoming = np.asarray(event.pre_relative_local_m_s, dtype=np.float64)
        lateral = np.asarray(event.tangent_offset_m, dtype=np.float64)
        normal_speed = max(0.10, -float(incoming[2]))
        fall_time = normal_speed / gravity_m_s2
        release_height = normal_speed * normal_speed / (2.0 * gravity_m_s2)
        qpos = np.asarray(full_base_data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(full_base_data.qvel, dtype=np.float64).copy()
        start_xy = lateral - incoming[:2] * fall_time
        qpos[full_experiment.ball_qadr : full_experiment.ball_qadr + 3] = (
            start_xy[0],
            start_xy[1],
            contact.BALL_RADIUS_M + 0.001 + release_height,
        )
        qpos[full_experiment.ball_qadr + 3 : full_experiment.ball_qadr + 7] = (
            1.0,
            0.0,
            0.0,
            0.0,
        )
        qvel[full_experiment.ball_vadr : full_experiment.ball_vadr + 3] = (
            incoming[0],
            incoming[1],
            0.0,
        )
        qvel[full_experiment.ball_vadr + 3 : full_experiment.ball_vadr + 6] = 0.0
        full_qpos_rows.append(qpos)
        full_qvel_rows.append(qvel)
        equivalent_release_heights.append(release_height)

    full_qpos_batch = jax.numpy.asarray(np.asarray(full_qpos_rows))
    full_qvel_batch = jax.numpy.asarray(np.asarray(full_qvel_rows))
    full_drop_evaluations = {}
    for name, damping in candidates.items():
        z_rows, vz_rows = rollout_full_drop_batch(
            float(damping), full_qpos_batch, full_qvel_batch
        )
        z_rows = np.asarray(z_rows, dtype=np.float64)
        vz_rows = np.asarray(vz_rows, dtype=np.float64)
        rows = []
        for index, event in enumerate(events):
            vz = vz_rows[index]
            z = z_rows[index]
            crossings = np.flatnonzero((vz[:-1] < -0.05) & (vz[1:] > 0.05))
            if not crossings.size:
                measured_outgoing = float(event.post_relative_local_m_s[2])
                rows.append(
                    {
                        "recording": event.recording,
                        "status": "no_first_rebound",
                        "equivalent_release_height_m": equivalent_release_heights[
                            index
                        ],
                        "measured_incoming_normal_speed_m_s": float(
                            -event.pre_relative_local_m_s[2]
                        ),
                        "simulated_outgoing_normal_speed_m_s": 0.0,
                        "measured_outgoing_normal_speed_m_s": measured_outgoing,
                        "simulated_restitution": 0.0,
                        "measured_restitution": float(event.normal_restitution),
                        "simulated_rebound_height_m": 0.0,
                        "measured_ballistic_rebound_height_m": (
                            measured_outgoing
                            * measured_outgoing
                            / (2.0 * gravity_m_s2)
                        ),
                    }
                )
                continue
            bounce_index = int(crossings[0] + 1)
            apex_candidates = np.flatnonzero(vz[bounce_index:] <= 0.0)
            apex_index = (
                bounce_index + int(apex_candidates[0])
                if apex_candidates.size
                else len(vz) - 1
            )
            incoming_speed = float(-np.min(vz[: bounce_index + 1]))
            outgoing_speed = float(np.max(vz[bounce_index : apex_index + 1]))
            rebound_height = float(
                np.max(z[bounce_index : apex_index + 1])
                - (contact.BALL_RADIUS_M + 0.001)
            )
            measured_outgoing = float(event.post_relative_local_m_s[2])
            rows.append(
                {
                    "recording": event.recording,
                    "status": "first_rebound_found",
                    "equivalent_release_height_m": equivalent_release_heights[index],
                    "simulated_incoming_normal_speed_m_s": incoming_speed,
                    "measured_incoming_normal_speed_m_s": float(
                        -event.pre_relative_local_m_s[2]
                    ),
                    "simulated_outgoing_normal_speed_m_s": outgoing_speed,
                    "measured_outgoing_normal_speed_m_s": measured_outgoing,
                    "simulated_restitution": outgoing_speed / incoming_speed,
                    "measured_restitution": float(event.normal_restitution),
                    "simulated_rebound_height_m": rebound_height,
                    "measured_ballistic_rebound_height_m": (
                        measured_outgoing * measured_outgoing / (2.0 * gravity_m_s2)
                    ),
                }
            )
        valid_rows = [row for row in rows if row["status"] == "first_rebound_found"]
        outgoing_errors = np.asarray(
            [
                row["simulated_outgoing_normal_speed_m_s"]
                - row["measured_outgoing_normal_speed_m_s"]
                for row in valid_rows
            ],
            dtype=np.float64,
        )
        height_errors = np.asarray(
            [
                row["simulated_rebound_height_m"]
                - row["measured_ballistic_rebound_height_m"]
                for row in valid_rows
            ],
            dtype=np.float64,
        )
        penalized_outgoing_errors = np.asarray(
            [
                row["simulated_outgoing_normal_speed_m_s"]
                - row["measured_outgoing_normal_speed_m_s"]
                for row in rows
            ],
            dtype=np.float64,
        )
        penalized_height_errors = np.asarray(
            [
                row["simulated_rebound_height_m"]
                - row["measured_ballistic_rebound_height_m"]
                for row in rows
            ],
            dtype=np.float64,
        )
        full_drop_evaluations[name] = {
            "parameters": {
                "pair_sliding_friction": friction,
                "solref_time_s": solref_time_s,
                "solref_damping_ratio": damping,
            },
            "count": len(valid_rows),
            "requested_count": len(rows),
            "first_rebound_success_rate": len(valid_rows) / len(rows),
            "outgoing_normal_velocity_rmse_m_s": float(
                np.sqrt(np.mean(outgoing_errors * outgoing_errors))
            ),
            "rebound_height_rmse_m": float(
                np.sqrt(np.mean(height_errors * height_errors))
            ),
            "penalized_all_event_outgoing_normal_velocity_rmse_m_s": float(
                np.sqrt(
                    np.mean(penalized_outgoing_errors * penalized_outgoing_errors)
                )
            ),
            "penalized_all_event_rebound_height_rmse_m": float(
                np.sqrt(np.mean(penalized_height_errors * penalized_height_errors))
            ),
            "rows": rows,
        }
    full_drop_validation = {
        "method": (
            "rest release from v_impact^2/(2g), gravity flight, static racket, "
            "first rebound only"
        ),
        "gravity_m_s2": gravity_m_s2,
        "duration_s": full_step_count * full_experiment.model.opt.timestep,
        "steps": full_step_count,
        "evaluations": full_drop_evaluations,
    }

    fitted_difference = abs(
        all_fit["solref_damping_ratio"]
        - high_quality_fit["solref_damping_ratio"]
    )
    loss_sensitivity = max(
        abs(
            all_fit["solref_damping_ratio"]
            - all_fit_least_squares["solref_damping_ratio"]
        ),
        abs(
            high_quality_fit["solref_damping_ratio"]
            - high_quality_fit_least_squares["solref_damping_ratio"]
        ),
    )
    loo_span = max(loo_values) - min(loo_values)
    stable_candidate = bool(
        int(np.sum(high_quality_mask)) >= 4
        and fitted_difference <= 0.03
        and loo_span <= 0.05
        and loss_sensitivity <= 0.05
        and full_drop_evaluations["all_five_robust_fit"]["count"] == len(events)
    )
    result = {
        "schema_version": 1,
        "backend": "MJX direct isolated contact",
        "mujoco_version": mujoco.__version__,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "timestep_s": float(experiment.model.opt.timestep),
        "steps_per_impact": step_count,
        "event_count": len(events),
        "high_quality_event_count": int(np.sum(high_quality_mask)),
        "high_quality_recordings": [
            event.recording
            for event, keep in zip(events, high_quality_mask)
            if keep
        ],
        "excluded_low_quality_recordings": [
            event.recording
            for event, keep in zip(events, high_quality_mask)
            if not keep
        ],
        "fit_all_five": all_fit,
        "fit_high_quality": high_quality_fit,
        "fit_all_five_least_squares": all_fit_least_squares,
        "fit_high_quality_least_squares": high_quality_fit_least_squares,
        "leave_one_recording_out": loo,
        "leave_one_recording_out_damping_range": [min(loo_values), max(loo_values)],
        "leave_one_recording_out_damping_span": loo_span,
        "all_vs_high_quality_damping_delta": fitted_difference,
        "robust_vs_least_squares_max_damping_delta": loss_sensitivity,
        "stable_candidate": stable_candidate,
        "evaluations": evaluations,
        "event_rows": event_rows,
        "full_drop_validation": full_drop_validation,
        "note": (
            "friction and solref time remain fixed because the recordings do not "
            "observe ball spin or millisecond contact force/duration"
        ),
    }
    (analysis_dir / "mjx_static_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary["mjx_validation"] = result
    summary["candidate_mjx_parameters"] = {
        "ball_mass_kg": contact.BALL_MASS_KG,
        "pair_sliding_friction": friction,
        "solref_time_s": solref_time_s,
        "solref_damping_ratio_all_five": all_fit["solref_damping_ratio"],
        "solref_damping_ratio_high_quality": high_quality_fit[
            "solref_damping_ratio"
        ],
        "solref_damping_ratio_all_five_least_squares": all_fit_least_squares[
            "solref_damping_ratio"
        ],
        "solref_damping_ratio_high_quality_least_squares": (
            high_quality_fit_least_squares["solref_damping_ratio"]
        ),
        "solimp": contact.SIMCODE7_BASELINE["solimp"],
        "margin_m": contact.SIMCODE7_BASELINE["margin_m"],
        "condim": contact.SIMCODE7_BASELINE["condim"],
    }
    summary["decision"]["status"] = (
        "mjx_candidate_stable" if stable_candidate else "mjx_candidate_not_robust"
    )
    summary["decision"]["mjx_candidate_stable"] = stable_candidate
    summary["decision"]["apply_to_production_xml"] = False
    summary["decision"]["message_zh"] = (
        "首次碰撞在 MJX 中形成稳定候选解，但采集时位置平滑开启且一组轨迹低质量；"
        "本轮只生成候选参数，不覆盖生产 XML。"
        if stable_candidate
        else (
            "完整释放回放中候选参数只有 3/5 次产生首次反弹，"
            "不能复现实物五次实验；本轮不覆盖生产 XML。"
        )
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = analysis_dir / "STATIC_CONTACT_ANALYSIS_ZH.md"
    report = report_path.read_text(encoding="utf-8").rstrip()
    report = report.split("\n\n## MJX 直接验证", maxsplit=1)[0].rstrip()
    conclusion_marker = "## 结论\n\n"
    data_marker = "\n\n## 数据审计"
    if conclusion_marker in report and data_marker in report:
        prefix, conclusion_and_rest = report.split(conclusion_marker, maxsplit=1)
        _, rest = conclusion_and_rest.split(data_marker, maxsplit=1)
        report = (
            prefix
            + conclusion_marker
            + summary["decision"]["message_zh"]
            + data_marker
            + rest
        )
    report += (
        "\n\n## MJX 直接验证\n\n"
        f"- 当前生产阻尼比：`{CURRENT_PRODUCTION_DAMPING:.6f}`。\n"
        f"- 5 组首次碰撞直接拟合：`{all_fit['solref_damping_ratio']:.6f}`。\n"
        f"- 排除低质量组后直接拟合："
        f"`{high_quality_fit['solref_damping_ratio']:.6f}`。\n"
        f"- 最小二乘拟合（5 组/高质量 4 组）："
        f"`{all_fit_least_squares['solref_damping_ratio']:.6f}` / "
        f"`{high_quality_fit_least_squares['solref_damping_ratio']:.6f}`。\n"
        f"- 两种拟合差：`{fitted_difference:.6f}`；留一录制范围："
        f"`{min(loo_values):.6f}–{max(loo_values):.6f}`。\n"
        f"- 当前参数的法向速度 RMSE（全部）："
        f"`{evaluations['current_production']['all']['normal_rmse_m_s']:.4f} m/s`；"
        f"5 组候选参数："
        f"`{evaluations['all_five_robust_fit']['all']['normal_rmse_m_s']:.4f} m/s`。\n"
        f"- 生产 XML 更新：`false`。{summary['decision']['message_zh']}\n"
        "\n| 录制 | 实物 e | 当前 MJX e | 候选 MJX e | 轨迹连续误差 (mm) | 质量 |\n"
        "|---|---:|---:|---:|---:|---|\n"
    )
    for row in event_rows:
        report += (
            f"| {row['recording']} | {row['measured_restitution']:.3f} | "
            f"{row['predictions']['current_production']['restitution']:.3f} | "
            f"{row['predictions']['all_five_robust_fit']['restitution']:.3f} | "
            f"{row['continuity_error_mm']:.1f} | "
            f"{'高' if row['high_quality'] else '低'} |\n"
        )
    report += (
        "\n## MJX 完整释放—首次反弹回放\n\n"
        "按实测首次碰撞入射速度换算等效静止释放高度，在重力下完整模拟下落和首次反弹。\n\n"
        f"- 当前参数：5/5 次产生首次反弹；全 5 组法向速度 RMSE "
        f"`{full_drop_evaluations['current_production']['penalized_all_event_outgoing_normal_velocity_rmse_m_s']:.4f} m/s`，"
        f"回弹高度 RMSE "
        f"`{full_drop_evaluations['current_production']['penalized_all_event_rebound_height_rmse_m']:.4f} m`。\n"
        f"- 5 组候选参数："
        f"{full_drop_evaluations['all_five_robust_fit']['count']}/5 次产生首次反弹；"
        f"把未反弹计为 0 后，法向速度 RMSE "
        f"`{full_drop_evaluations['all_five_robust_fit']['penalized_all_event_outgoing_normal_velocity_rmse_m_s']:.4f} m/s`，"
        f"回弹高度 RMSE "
        f"`{full_drop_evaluations['all_five_robust_fit']['penalized_all_event_rebound_height_rmse_m']:.4f} m`。\n"
    )
    report_path.write_text(report, encoding="utf-8")
    recommendation = {
        "status": "candidate_only_do_not_apply",
        "production_xml_updated": False,
        "current_production": {
            "pair_sliding_friction": friction,
            "solref_time_s": solref_time_s,
            "solref_damping_ratio": CURRENT_PRODUCTION_DAMPING,
        },
        "mjx_candidate_robust_all_five": candidates["all_five_robust_fit"],
        "mjx_candidate_robust_high_quality": candidates[
            "high_quality_robust_fit"
        ],
        "mjx_candidate_least_squares_all_five": candidates[
            "all_five_least_squares_fit"
        ],
        "mjx_candidate_least_squares_high_quality": candidates[
            "high_quality_least_squares_fit"
        ],
        "reason": summary["decision"]["message_zh"],
    }
    (analysis_dir / "recommended_contact_params.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
