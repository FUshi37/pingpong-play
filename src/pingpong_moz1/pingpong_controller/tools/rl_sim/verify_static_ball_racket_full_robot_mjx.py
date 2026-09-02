#!/usr/bin/env python3
"""Fit first-bounce contact damping with the fixed-base full MOZ1 MJX model.

The ball state for each run comes from the continuous release-to-impact camera
position fit produced by ``verify_static_ball_racket_history_mjx.py``.  The
robot is held at its recorded impact posture and settled for 700 steps before
the ball is placed at the reconstructed release state.  No impact velocity is
converted to a release height.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

import identify_ball_racket_contact as contact
import identify_static_ball_racket_contact as static_contact
import verify_static_ball_racket_history_mjx as history
from mjx_smoke import _write_mjx_contact_only_xml
from rl_juggle_env_random import (
    RIGHT_ARM_JOINTS,
    TARGET_DEGREES,
    _build_temp_xml_with_ball,
)


WORKSPACE = Path(__file__).resolve().parents[5]
DEFAULT_XML = (
    WORKSPACE
    / "simcode7/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
)
DEFAULT_ANALYSIS_DIR = (
    WORKSPACE
    / "analysis_results/ball_racket_contact_identification_v91_static_"
    "20260901_152749_153517"
)
WARMUP_STEPS = 700
CONTACT_CENTER_M = contact.BALL_RADIUS_M + contact.RACKET_RUBBER_HALF_THICKNESS_M
DAMPING_GRID = np.linspace(0.05, 0.50, 181)


@dataclass(frozen=True)
class PreparedRecord:
    reconstruction: history.ReconstructedRecord
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    target_indices: np.ndarray
    target_normal_positions_m: np.ndarray
    expected_impact_index: int
    diagnostic: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--record-root", type=Path, default=static_contact.DEFAULT_RECORD_ROOT)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.001,
        help="MJX integration step; use 0.0002 for the 3 ms contact-time fit.",
    )
    parser.add_argument(
        "--output-name",
        default="full_robot_mjx_history_replay.json",
    )
    parser.add_argument(
        "--kinematic-robot-hold",
        action="store_true",
        help=(
            "Hold the settled robot qpos/qvel fixed each MJX substep; appropriate "
            "for the static, high-gain-held racket identification experiment."
        ),
    )
    parser.add_argument("--grid-points", type=int, default=181)
    return parser.parse_args()


def _object_id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"missing MuJoCo object: {name}")
    return int(object_id)


def _pair_id_for_geoms(model: Any, first: int, second: int) -> int:
    for pair_id in range(model.npair):
        if {int(model.pair_geom1[pair_id]), int(model.pair_geom2[pair_id])} == {
            first,
            second,
        }:
            return pair_id
    raise ValueError("explicit ball/racket contact pair is missing")


def _interpolate_trace(trace: dict[str, np.ndarray], timestamp_s: float) -> np.ndarray:
    if len(trace["time"]) == 0:
        raise ValueError("recording has no robot joint samples")
    return np.asarray(
        [
            np.interp(timestamp_s, trace["time"], trace["actual_q"][:, axis])
            for axis in range(7)
        ],
        dtype=np.float64,
    )


def _seed_robot_posture(model: Any, data: Any, right_q: np.ndarray) -> None:
    import mujoco

    for joint_name, degrees in TARGET_DEGREES.items():
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        value = float(np.deg2rad(degrees))
        data.qpos[int(model.jnt_qposadr[joint_id])] = value
        data.ctrl[actuator_id] = value
    for joint_name, value in zip(RIGHT_ARM_JOINTS, right_q):
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
        data.ctrl[actuator_id] = float(value)


def _load_reconstructions(
    analysis_dir: Path,
    record_root: Path,
) -> tuple[dict[str, Any], list[history.ReconstructedRecord], dict[str, dict[str, np.ndarray]]]:
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    event_fields = {item.name for item in fields(contact.ImpactEvent)}
    events = [
        contact.ImpactEvent(
            **{key: value for key, value in row.items() if key in event_fields}
        )
        for row in summary["events"]
    ]
    reconstructed: list[history.ReconstructedRecord] = []
    traces: dict[str, dict[str, np.ndarray]] = {}
    for event in events:
        samples, trace, _, _ = static_contact._load_recording(
            record_root / event.recording / "control_samples.jsonl"
        )
        reconstructed.append(history._reconstruct_record(event, samples))
        traces[event.recording] = trace
    return summary, reconstructed, traces


def _build_model(
    xml_path: Path,
    timestep_s: float,
) -> tuple[Any, dict[str, int], Path, float]:
    import mujoco

    ball_xml = _build_temp_xml_with_ball(xml_path)
    fixed_xml = _write_mjx_contact_only_xml(ball_xml, fixed_base=True)
    model = mujoco.MjModel.from_xml_path(str(fixed_xml))
    source_timestep = float(model.opt.timestep)
    # Match the continuous state reconstruction exactly and use the measured
    # 4 g ball prior.  Inertia scales linearly for an unchanged sphere radius.
    model.opt.gravity[:] = (0.0, 0.0, -history.GRAVITY_M_S2)
    model.opt.timestep = float(timestep_s)
    ball_body_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball"
    )
    original_mass = float(model.body_mass[ball_body_id])
    model.body_mass[ball_body_id] = contact.BALL_MASS_KG
    model.body_inertia[ball_body_id] *= contact.BALL_MASS_KG / original_mass
    ball_joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_geom_id = _object_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    racket_geom_id = _object_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "racket_rubber_fore"
    )
    pair_id = _pair_id_for_geoms(model, ball_geom_id, racket_geom_id)
    ids = {
        "ball_body": ball_body_id,
        "ball_geom": ball_geom_id,
        "racket_geom": racket_geom_id,
        "pair": pair_id,
        "ball_qadr": int(model.jnt_qposadr[ball_joint_id]),
        "ball_vadr": int(model.jnt_dofadr[ball_joint_id]),
        "base_z_qadr": int(
            model.jnt_qposadr[
                _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, "base_z")
            ]
        ),
    }
    return model, ids, fixed_xml, source_timestep


def _prepare_records(
    model: Any,
    ids: dict[str, int],
    reconstructed: list[history.ReconstructedRecord],
    traces: dict[str, dict[str, np.ndarray]],
    warmup_steps: int,
) -> list[PreparedRecord]:
    import mujoco

    timestep = float(model.opt.timestep)
    prepared = []
    for row in reconstructed:
        data = mujoco.MjData(model)
        right_q = _interpolate_trace(
            traces[row.recording], row.event.impact_time_s
        )
        _seed_robot_posture(model, data, right_q)
        ball_qadr = ids["ball_qadr"]
        ball_vadr = ids["ball_vadr"]
        data.qpos[ball_qadr : ball_qadr + 3] = (0.0, 0.0, 10.0)
        data.qpos[ball_qadr + 3 : ball_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        initial_base_z = float(data.qpos[ids["base_z_qadr"]])
        for _ in range(warmup_steps):
            mujoco.mj_step(model, data)

        racket_position = np.asarray(
            data.geom_xpos[ids["racket_geom"]], dtype=np.float64
        ).copy()
        racket_rotation = np.asarray(
            data.geom_xmat[ids["racket_geom"]], dtype=np.float64
        ).reshape(3, 3).copy()
        racket_normal = racket_rotation[:, 2]
        recorded_normal = row.frame[:, 2]
        normal_alignment = float(np.dot(racket_normal, recorded_normal))
        if normal_alignment < 0.90:
            raise ValueError(
                f"{row.recording}: recorded/MJX racket normal alignment is only "
                f"{normal_alignment:.3f}"
            )

        incoming_at_impact, _ = row.incoming_fit.state(row.event.impact_time_s)
        recorded_racket_center = np.asarray(
            row.event.racket_center_m, dtype=np.float64
        )
        tangent = incoming_at_impact - recorded_racket_center
        tangent -= recorded_normal * float(np.dot(tangent, recorded_normal))
        # Remove the tiny component created by the warm-settled normal change.
        tangent -= racket_normal * float(np.dot(tangent, racket_normal))
        simulated_contact_center = (
            racket_position + tangent + CONTACT_CENTER_M * racket_normal
        )
        flight_delta = row.incoming_fit.position_m - incoming_at_impact
        start_position = simulated_contact_center + flight_delta

        data.qpos[ball_qadr : ball_qadr + 3] = start_position
        data.qpos[ball_qadr + 3 : ball_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        data.qvel[ball_vadr : ball_vadr + 3] = row.incoming_fit.velocity_m_s
        data.qvel[ball_vadr + 3 : ball_vadr + 6] = 0.0
        data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(model, data)

        target_indices = np.rint(
            row.target_times_from_start_s / timestep
        ).astype(int) - 1
        expected_impact_index = int(
            round(
                (row.event.impact_time_s - row.incoming_fit.reference_time_s)
                / timestep
            )
        )
        drop_height = float(
            np.dot(start_position - simulated_contact_center, racket_normal)
        )
        prepared.append(
            PreparedRecord(
                reconstruction=row,
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                ctrl=data.ctrl.copy(),
                target_indices=target_indices,
                target_normal_positions_m=row.target_positions_local_m[:, 2].copy(),
                expected_impact_index=expected_impact_index,
                diagnostic={
                    "recorded_right_arm_q_rad": right_q.tolist(),
                    "warmup_steps": warmup_steps,
                    "warmup_duration_s": warmup_steps * timestep,
                    "base_z_delta_after_warmup_mm": 1000.0
                    * (float(data.qpos[ids["base_z_qadr"]]) - initial_base_z),
                    "recorded_to_sim_racket_normal_dot": normal_alignment,
                    "release_time_before_impact_s": row.event.impact_time_s
                    - row.incoming_fit.reference_time_s,
                    "release_height_above_contact_along_racket_normal_m": drop_height,
                    "release_position_world_m": start_position.tolist(),
                    "release_velocity_world_m_s": row.incoming_fit.velocity_m_s.tolist(),
                    "simulated_contact_center_world_m": simulated_contact_center.tolist(),
                },
            )
        )
    return prepared


def _run_fit(
    model: Any,
    ids: dict[str, int],
    prepared: list[PreparedRecord],
    source_xml: Path,
    fixed_xml: Path,
    source_timestep_s: float,
    kinematic_robot_hold: bool,
    grid_points: int,
) -> dict[str, Any]:
    import jax
    from jax import lax
    import mujoco
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)
    timestep = float(model.opt.timestep)
    step_count = max(
        int(np.max(row.target_indices)) + 10 for row in prepared
    )
    pair_id = ids["pair"]
    source_damping = float(model.pair_solref[pair_id, 1])
    source_time = float(model.pair_solref[pair_id, 0])
    friction = float(model.pair_friction[pair_id, 0])
    margin = float(model.pair_margin[pair_id])
    mjx_model = mjx.put_model(model)
    base_data = mjx.make_data(mjx_model)
    qpos_batch = jax.numpy.asarray(np.asarray([row.qpos for row in prepared]))
    qvel_batch = jax.numpy.asarray(np.asarray([row.qvel for row in prepared]))
    ctrl_batch = jax.numpy.asarray(np.asarray([row.ctrl for row in prepared]))
    damping_grid = np.linspace(0.05, 0.50, int(grid_points))

    @jax.jit
    def rollout_batch(damping_value, qpos_values, qvel_values, ctrl_values):
        dynamic_model = mjx_model.replace(
            pair_solref=mjx_model.pair_solref.at[pair_id, 1].set(damping_value)
        )

        def rollout(qpos, qvel, ctrl):
            data = base_data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
            held_robot_qpos = qpos[: ids["ball_qadr"]]

            def step(data, _):
                if kinematic_robot_hold:
                    data = data.replace(
                        qpos=data.qpos.at[: ids["ball_qadr"]].set(
                            held_robot_qpos
                        ),
                        qvel=data.qvel.at[: ids["ball_vadr"]].set(0.0),
                    )
                velocity = data.qvel[ids["ball_vadr"] : ids["ball_vadr"] + 3]
                drag_force = -contact.BALL_MASS_KG * history.PAPER_DRAG_PER_M * (
                    jax.numpy.linalg.norm(velocity) * velocity
                )
                xfrc = data.xfrc_applied.at[ids["ball_body"], :3].set(
                    drag_force
                )
                data = mjx.step(dynamic_model, data.replace(xfrc_applied=xfrc))
                normal = data.geom_xmat[ids["racket_geom"], :, 2]
                displacement = (
                    data.qpos[ids["ball_qadr"] : ids["ball_qadr"] + 3]
                    - data.geom_xpos[ids["racket_geom"]]
                )
                relative_normal_position = jax.numpy.dot(displacement, normal)
                ball_normal_velocity = jax.numpy.dot(
                    data.qvel[ids["ball_vadr"] : ids["ball_vadr"] + 3], normal
                )
                return data, (relative_normal_position, ball_normal_velocity)

            _, trajectory = lax.scan(step, data, xs=None, length=step_count)
            return trajectory

        return jax.vmap(rollout)(qpos_values, qvel_values, ctrl_values)

    trajectory_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def trajectories(damping: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(damping), 9)
        if key not in trajectory_cache:
            position, velocity = rollout_batch(
                float(damping), qpos_batch, qvel_batch, ctrl_batch
            )
            trajectory_cache[key] = (
                np.asarray(position, dtype=np.float64),
                np.asarray(velocity, dtype=np.float64),
            )
        return trajectory_cache[key]

    high_quality_mask = np.asarray(
        [
            row.reconstruction.event.continuity_error_mm <= 60.0
            and row.reconstruction.event.pre_fit_rmse_mm <= 3.0
            and row.reconstruction.event.post_fit_rmse_mm <= 3.0
            and row.reconstruction.causal_holdout_rmse_m <= 0.005
            for row in prepared
        ],
        dtype=bool,
    )

    def errors_by_record(damping: float) -> list[np.ndarray]:
        position, _ = trajectories(damping)
        return [
            position[index, row.target_indices] - row.target_normal_positions_m
            for index, row in enumerate(prepared)
        ]

    def objective(damping: float, mask: np.ndarray) -> float:
        losses = []
        for keep, error in zip(mask, errors_by_record(damping)):
            if not keep:
                continue
            scaled_squared = (error / 0.010) ** 2
            robust = np.where(
                scaled_squared <= 4.0,
                scaled_squared,
                4.0 * np.sqrt(scaled_squared) - 4.0,
            )
            losses.append(float(np.mean(robust)))
        return float(np.mean(losses)) if losses else math.inf

    def fit(mask: np.ndarray, include_profile: bool = False) -> dict[str, Any]:
        profile = np.asarray(
            [objective(float(value), mask) for value in damping_grid],
            dtype=np.float64,
        )
        best_index = int(np.argmin(profile))
        lower = float(damping_grid[max(0, best_index - 1)])
        upper = float(damping_grid[min(len(damping_grid) - 1, best_index + 1)])
        if upper > lower:
            optimized = minimize_scalar(
                lambda value: objective(float(value), mask),
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 2.0e-6},
            )
            value = float(optimized.x)
            loss = float(optimized.fun)
            success = bool(optimized.success)
        else:
            value = float(damping_grid[best_index])
            loss = float(profile[best_index])
            success = True
        result = {
            "solref_damping_ratio": value,
            "objective": loss,
            "success": success,
            "record_count": int(np.sum(mask)),
            "best_grid_value": float(damping_grid[best_index]),
            "global_grid_step": float(damping_grid[1] - damping_grid[0]),
        }
        if include_profile:
            result["profile_damping_grid"] = damping_grid.tolist()
            result["profile_objective"] = profile.tolist()
        return result

    all_mask = np.ones(len(prepared), dtype=bool)
    fit_all = fit(all_mask, include_profile=True)
    fit_high_quality = fit(high_quality_mask)
    leave_one_out = []
    per_record = []
    for index, row in enumerate(prepared):
        mask = all_mask.copy()
        mask[index] = False
        result = fit(mask)
        result["held_out_recording"] = row.reconstruction.recording
        leave_one_out.append(result)
        mask = np.zeros(len(prepared), dtype=bool)
        mask[index] = True
        result = fit(mask)
        result["recording"] = row.reconstruction.recording
        per_record.append(result)

    def evaluate(name: str, damping: float) -> dict[str, Any]:
        relative_position, _ = trajectories(damping)
        errors = errors_by_record(damping)
        rows = []
        for index, (prepared_row, error) in enumerate(zip(prepared, errors)):
            reconstruction = prepared_row.reconstruction
            impact = prepared_row.expected_impact_index
            search_start = max(0, impact - int(round(0.030 / timestep)))
            search_end = min(step_count, impact + int(round(0.060 / timestep)))
            minimum_index = search_start + int(
                np.argmin(relative_position[index, search_start:search_end])
            )
            rebound_end = min(step_count, impact + int(round(0.240 / timestep)))
            relative_velocity = np.gradient(
                relative_position[index], timestep
            )
            outgoing_velocity = float(
                np.max(relative_velocity[minimum_index:rebound_end])
            )
            apex = float(
                np.max(relative_position[index, minimum_index:rebound_end])
            )
            target_apex = float(np.max(prepared_row.target_normal_positions_m))
            rows.append(
                {
                    "recording": reconstruction.recording,
                    "high_quality": bool(high_quality_mask[index]),
                    "status": (
                        "first_rebound_found"
                        if outgoing_velocity > 0.05
                        and apex > CONTACT_CENTER_M + 0.001
                        else "no_first_rebound"
                    ),
                    "normal_position_rmse_mm": float(
                        1000.0 * np.sqrt(np.mean(error * error))
                    ),
                    "normal_position_bias_mm": float(1000.0 * np.mean(error)),
                    "simulated_first_apex_above_contact_mm": 1000.0
                    * (apex - CONTACT_CENTER_M),
                    "history_first_apex_above_contact_mm": 1000.0
                    * (target_apex - CONTACT_CENTER_M),
                    "simulated_outgoing_relative_normal_velocity_m_s": outgoing_velocity,
                    "history_normal_restitution": history._reconstruction_row(
                        reconstruction
                    )["history_state_at_impact"]["normal_restitution"],
                }
            )
        all_errors = np.concatenate(errors)
        hq_errors = np.concatenate(
            [error for error, keep in zip(errors, high_quality_mask) if keep]
        )
        row_rmse = np.asarray(
            [row["normal_position_rmse_mm"] for row in rows], dtype=np.float64
        )
        hq_row_rmse = np.asarray(
            [
                row["normal_position_rmse_mm"]
                for row in rows
                if row["high_quality"]
            ],
            dtype=np.float64,
        )
        return {
            "name": name,
            "solref_damping_ratio": float(damping),
            "first_rebound_count": sum(
                row["status"] == "first_rebound_found" for row in rows
            ),
            "requested_count": len(rows),
            "all_position_samples_normal_rmse_mm": float(
                1000.0 * np.sqrt(np.mean(all_errors * all_errors))
            ),
            "high_quality_position_samples_normal_rmse_mm": float(
                1000.0 * np.sqrt(np.mean(hq_errors * hq_errors))
            ),
            "per_record_normal_rmse_median_mm": float(np.median(row_rmse)),
            "high_quality_per_record_normal_rmse_max_mm": float(
                np.max(hq_row_rmse)
            ),
            "rows": rows,
        }

    candidate_damping = float(fit_all["solref_damping_ratio"])
    evaluations = {
        "source_xml_runtime": evaluate("source_xml_runtime", source_damping),
        "full_robot_history_fit_all": evaluate(
            "full_robot_history_fit_all", candidate_damping
        ),
        "full_robot_history_fit_high_quality": evaluate(
            "full_robot_history_fit_high_quality",
            float(fit_high_quality["solref_damping_ratio"]),
        ),
    }
    loo_values = [row["solref_damping_ratio"] for row in leave_one_out]
    per_values = [row["solref_damping_ratio"] for row in per_record]
    hq_per_values = [
        value for value, keep in zip(per_values, high_quality_mask) if keep
    ]
    candidate_evaluation = evaluations["full_robot_history_fit_all"]
    stable = bool(
        candidate_evaluation["first_rebound_count"] == len(prepared)
        and int(np.sum(high_quality_mask)) >= 4
        and max(loo_values) - min(loo_values) <= 0.05
        and abs(
            float(fit_all["solref_damping_ratio"])
            - float(fit_high_quality["solref_damping_ratio"])
        )
        <= 0.05
        and max(hq_per_values) - min(hq_per_values) <= 0.05
        and candidate_evaluation["high_quality_position_samples_normal_rmse_mm"]
        <= 20.0
        and candidate_evaluation["high_quality_per_record_normal_rmse_max_mm"]
        <= 30.0
    )
    return {
        "schema_version": 1,
        "backend": "fixed-base full-robot MJX",
        "mujoco_version": mujoco.__version__,
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "source_xml": str(source_xml),
        "generated_fixed_base_xml": str(fixed_xml),
        "timestep_s": timestep,
        "source_xml_timestep_s": source_timestep_s,
        "warmup_steps": int(prepared[0].diagnostic["warmup_steps"]),
        "rollout_steps": step_count,
        "ball_mass_kg": contact.BALL_MASS_KG,
        "paper_drag_coefficient_per_m": history.PAPER_DRAG_PER_M,
        "contact_parameters_held_fixed": {
            "solref_time_s": source_time,
            "pair_sliding_friction": friction,
            "margin_m": margin,
            "solimp": model.pair_solimp[pair_id].tolist(),
            "condim": int(model.pair_dim[pair_id]),
        },
        "source_xml_runtime_damping_ratio": source_damping,
        "runtime_parameter_inheritance_verified": True,
        "robot_contact_response_mode": (
            "kinematic_hold_at_settled_recorded_pose"
            if kinematic_robot_hold
            else "articulated_pd_hold"
        ),
        "initialization": (
            "continuous release-to-impact position fit, per-record corresponding "
            "height and velocity; no impact-speed-derived release height"
        ),
        "target": "first-rebound relative normal position trajectory",
        "one_first_impact_per_recording": True,
        "later_juggles_used": False,
        "fit_all": fit_all,
        "fit_high_quality": fit_high_quality,
        "high_quality_record_count": int(np.sum(high_quality_mask)),
        "leave_one_recording_out": leave_one_out,
        "leave_one_recording_out_damping_range": [min(loo_values), max(loo_values)],
        "leave_one_recording_out_damping_span": max(loo_values) - min(loo_values),
        "all_vs_high_quality_damping_delta": abs(
            float(fit_all["solref_damping_ratio"])
            - float(fit_high_quality["solref_damping_ratio"])
        ),
        "per_recording_fit": per_record,
        "per_recording_damping_range": [min(per_values), max(per_values)],
        "per_recording_damping_span": max(per_values) - min(per_values),
        "high_quality_per_recording_damping_range": [
            min(hq_per_values),
            max(hq_per_values),
        ],
        "high_quality_per_recording_damping_span": max(hq_per_values)
        - min(hq_per_values),
        "stable_candidate": stable,
        "stability_thresholds": {
            "leave_one_recording_out_damping_span_max": 0.05,
            "high_quality_per_recording_damping_span_max": 0.05,
            "all_vs_high_quality_damping_delta_max": 0.05,
            "high_quality_position_samples_normal_rmse_max_mm": 20.0,
            "high_quality_per_record_normal_rmse_max_mm": 30.0,
            "first_rebound_required_for_all": True,
        },
        "records": [
            {
                "recording": row.reconstruction.recording,
                "high_quality": bool(keep),
                "incoming_position_fit_rmse_mm": 1000.0
                * row.reconstruction.incoming_fit.rmse_m,
                "causal_holdout_rmse_mm": 1000.0
                * row.reconstruction.causal_holdout_rmse_m,
                **row.diagnostic,
            }
            for row, keep in zip(prepared, high_quality_mask)
        ],
        "evaluations": evaluations,
    }


def _write_outputs(
    analysis_dir: Path,
    summary: dict[str, Any],
    result: dict[str, Any],
    output_name: str,
) -> None:
    output_path = analysis_dir / output_name
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    previous = summary.get("fixed_base_full_robot_mjx")
    if (
        isinstance(previous, dict)
        and previous.get("timestep_s") != result.get("timestep_s")
    ):
        previous_step_us = int(round(1.0e6 * float(previous["timestep_s"])))
        previous_mode = previous.get("robot_contact_response_mode", "articulated_pd_hold")
        mode_label = "kinematic" if previous_mode.startswith("kinematic") else "articulated"
        summary[
            f"fixed_base_full_robot_mjx_dt{previous_step_us}us_{mode_label}_diagnostic"
        ] = previous
    elif (
        isinstance(previous, dict)
        and previous.get("robot_contact_response_mode")
        != result.get("robot_contact_response_mode")
    ):
        previous_step_us = int(round(1.0e6 * float(previous["timestep_s"])))
        previous_mode = previous.get("robot_contact_response_mode", "articulated_pd_hold")
        mode_label = "kinematic" if previous_mode.startswith("kinematic") else "articulated"
        summary[
            f"fixed_base_full_robot_mjx_dt{previous_step_us}us_{mode_label}_diagnostic"
        ] = previous
    summary["fixed_base_full_robot_mjx"] = result
    summary.setdefault("recommended_parameters", {})[
        "full_robot_history_solref_damping_ratio"
    ] = result["fit_all"]["solref_damping_ratio"]
    stable = bool(result["stable_candidate"])
    fit_value = float(result["fit_all"]["solref_damping_ratio"])
    count = len(result["records"])
    summary["decision"] = {
        "status": (
            "fixed_base_full_robot_candidate_stable"
            if stable
            else "fixed_base_full_robot_candidate_not_converged"
        ),
        "apply_to_production_xml": False,
        "message_zh": (
            f"已用固定底座完整机器人 MJX 对 {count} 组位置历史做同高度首次反弹回放；"
            f"候选阻尼比为 {fit_value:.6f}，并通过稳定性门槛。为避免在报告确认前"
            "自动覆盖生产模型，本轮只记录候选值。"
            if stable
            else (
                f"已用固定底座完整机器人 MJX 对 {count} 组位置历史做同高度首次反弹"
                f"回放；候选阻尼比为 {fit_value:.6f}，但未通过跨录制稳定性门槛，"
                "不覆盖生产 XML。"
            )
        ),
    }
    (analysis_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    base_audit_path = analysis_dir / "base_stability_audit.json"
    base_audit = json.loads(base_audit_path.read_text(encoding="utf-8"))
    current_base = base_audit["variants"]["current_support_contact"]["mjx"]
    fixed_base = base_audit["variants"]["fixed_base_equality"]["mjx"]
    source_eval = result["evaluations"]["source_xml_runtime"]
    candidate_eval = result["evaluations"]["full_robot_history_fit_all"]
    scope = summary.get("first_bounce_scope_audit", {})
    scope_text = ""
    if scope:
        used_ends = [
            row["used_rebound_end_s_after_first_impact"]
            for row in scope["records"]
        ]
        next_impacts = [
            row["next_impact_detected_s_after_first_impact"]
            for row in scope["records"]
            if row["next_impact_detected_s_after_first_impact"] is not None
        ]
        scope_text = (
            "- 数据范围硬约束：每条录制只选择释放后的第一次撞拍；碰后拟合窗口"
            f"止于第一次撞拍后 `{min(used_ends):.3f}–{max(used_ends):.3f} s`，"
            f"检测到的下一次撞拍位于 `{min(next_impacts):.3f}–"
            f"{max(next_impacts):.3f} s`，最小隔离间隙 "
            f"`{scope['minimum_used_to_next_impact_separation_ms']:.1f} ms`。"
            "后续颠球样本使用数为 `0`。\n"
        )
    backend_audit_path = analysis_dir / "full_robot_contact_backend_audit.json"
    backend_text = ""
    if backend_audit_path.exists():
        backend_audit = json.loads(backend_audit_path.read_text(encoding="utf-8"))
        backend_text = (
            "- 同状态接触审计确认只发生 `ball ↔ racket_rubber_fore` 接触；"
            f"CPU/MJX 源参数全部位置 RMSE 为 "
            f"`{backend_audit['cpu_all_position_samples_normal_rmse_mm']:.2f}` / "
            f"`{backend_audit['mjx_all_position_samples_normal_rmse_mm']:.2f} mm`，"
            f"最大逐组差为 "
            f"`{backend_audit['maximum_per_record_cpu_mjx_rmse_delta_mm']:.2f} mm`，"
            "后端一致性检查未通过。\n"
        )
    coarse = summary.get(
        "fixed_base_full_robot_mjx_dt1000us_articulated_diagnostic"
    ) or summary.get("fixed_base_full_robot_mjx_dt1000us_diagnostic")
    coarse_text = ""
    if isinstance(coarse, dict):
        coarse_text = (
            f"- 生产 `1.0 ms` 步长逐录制阻尼范围："
            f"`{coarse['per_recording_damping_range'][0]:.6f}`–"
            f"`{coarse['per_recording_damping_range'][1]:.6f}`，受 3 ms 接触过程的"
            "离散相位影响而不收敛；以下候选使用 `0.2 ms` 接触辨识步长。\n"
        )
    articulated_fine = summary.get(
        "fixed_base_full_robot_mjx_dt200us_articulated_diagnostic"
    )
    articulated_text = ""
    if isinstance(articulated_fine, dict):
        articulated_text = (
            "- `0.2 ms` 动态机械臂 MJX 与 CPU MuJoCo 接触响应不一致，且逐录制"
            f"范围仍为 `{articulated_fine['per_recording_damping_range'][0]:.6f}`–"
            f"`{articulated_fine['per_recording_damping_range'][1]:.6f}`；最终静态辨识"
            "采用每子步保持预热后实测机器人姿态的回放。\n"
        )
    section = (
        "## 固定底座完整机器人 MJX 位置回放\n\n"
        "本轮不从单个入射速度反推高度。每组都从释放至首次碰撞的连续球位置拟合"
        "重力—二次阻力状态，并按该组对应的释放位置和速度启动完整机器人 MJX。"
        f"机器人先保持实物碰撞时的右臂姿态预热 {result['warmup_steps']} 步"
        f"（{result['warmup_steps'] * result['timestep_s']:.3f} s）。\n\n"
        f"{scope_text}"
        f"- 原 MJX 第 700 步底座 z 漂移："
        f"`{current_base['base_z_delta_at_700_mm']:.6f} mm`；固定后："
        f"`{fixed_base['base_z_delta_at_700_mm']:.6f} mm`。\n"
        "- 修复方式：锁定六个基座关节的等式约束，保持 qpos/dof/actuator 布局不变；"
        "默认可移动底座训练路径仍可选择不启用。\n"
        f"{coarse_text}"
        f"{articulated_text}"
        f"{backend_text}"
        f"- 完整机器人全 {len(result['records'])} 组/高质量 "
        f"{result['high_quality_record_count']} 组拟合阻尼比："
        f"`{result['fit_all']['solref_damping_ratio']:.6f}` / "
        f"`{result['fit_high_quality']['solref_damping_ratio']:.6f}`。\n"
        f"- 留一录制范围：`{result['leave_one_recording_out_damping_range'][0]:.6f}`–"
        f"`{result['leave_one_recording_out_damping_range'][1]:.6f}`；稳定候选："
        f"`{str(result['stable_candidate']).lower()}`。\n"
        f"- 源 XML 参数/拟合候选的全部首次反弹位置 RMSE："
        f"`{source_eval['all_position_samples_normal_rmse_mm']:.2f}` / "
        f"`{candidate_eval['all_position_samples_normal_rmse_mm']:.2f} mm`。\n"
        "- 运行时构建器已改为继承源 XML 的接触参数；此前它会把源 XML 的"
        " `0.152540` 静默重置为 `0.80`。\n"
        f"- 生产 XML 更新：`false`。{summary['decision']['message_zh']}\n\n"
        "| 录制 | 释放高度/法向 (mm) | 释放 vz (m/s) | 留尾预测 RMSE (mm) | "
        "候选反弹位置 RMSE (mm) | 质量 |\n"
        "|---|---:|---:|---:|---:|---|\n"
    )
    candidate_by_name = {
        row["recording"]: row for row in candidate_eval["rows"]
    }
    for row in result["records"]:
        evaluation = candidate_by_name[row["recording"]]
        section += (
            f"| {row['recording']} | "
            f"{1000.0 * row['release_height_above_contact_along_racket_normal_m']:.2f} | "
            f"{row['release_velocity_world_m_s'][2]:.3f} | "
            f"{row['causal_holdout_rmse_mm']:.2f} | "
            f"{evaluation['normal_position_rmse_mm']:.2f} | "
            f"{'高' if row['high_quality'] else '低'} |\n"
        )
    report_path = analysis_dir / "STATIC_CONTACT_ANALYSIS_ZH.md"
    report = report_path.read_text(encoding="utf-8").rstrip()
    marker = "\n\n## 固定底座完整机器人 MJX 位置回放"
    if marker in report:
        report = report.split(marker, maxsplit=1)[0].rstrip()
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
    report_path.write_text(report + "\n\n" + section, encoding="utf-8")

    recommendation_path = analysis_dir / "recommended_contact_params.json"
    recommendation = json.loads(
        recommendation_path.read_text(encoding="utf-8")
    )
    recommendation["status"] = (
        "fixed_base_full_robot_candidate_only_do_not_apply"
        if stable
        else "fixed_base_full_robot_not_converged_do_not_apply"
    )
    recommendation["production_xml_updated"] = False
    recommendation["full_robot_history_solref_damping_ratio"] = fit_value
    recommendation["runtime_builder_inherits_source_contact_parameters"] = True
    recommendation["reason"] = summary["decision"]["message_zh"]
    recommendation_path.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    record_root = args.record_root.expanduser().resolve()
    source_xml = args.xml.expanduser().resolve()
    summary, reconstructed, traces = _load_reconstructions(
        analysis_dir, record_root
    )
    if args.timestep <= 0.0:
        raise SystemExit("--timestep must be positive")
    if args.grid_points < 3:
        raise SystemExit("--grid-points must be at least 3")
    model, ids, fixed_xml, source_timestep = _build_model(
        source_xml, float(args.timestep)
    )
    prepared = _prepare_records(
        model, ids, reconstructed, traces, int(args.warmup_steps)
    )
    for row in prepared:
        print(
            f"[{row.reconstruction.recording}] height="
            f"{1000.0 * row.diagnostic['release_height_above_contact_along_racket_normal_m']:.1f}mm "
            f"holdout={1000.0 * row.reconstruction.causal_holdout_rmse_m:.2f}mm "
            f"base={row.diagnostic['base_z_delta_after_warmup_mm']:.6f}mm",
            flush=True,
        )
    result = _run_fit(
        model,
        ids,
        prepared,
        source_xml,
        fixed_xml,
        source_timestep,
        bool(args.kinematic_robot_hold),
        int(args.grid_points),
    )
    _write_outputs(analysis_dir, summary, result, str(args.output_name))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
