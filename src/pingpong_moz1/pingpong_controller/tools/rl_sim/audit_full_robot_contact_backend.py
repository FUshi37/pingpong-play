#!/usr/bin/env python3
"""Compare CPU MuJoCo and MJX first-bounce contact on identical full-model states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import identify_ball_racket_contact as contact
import verify_static_ball_racket_full_robot_mjx as full_robot
import verify_static_ball_racket_history_mjx as history


DEFAULT_ANALYSIS_DIR = full_robot.DEFAULT_ANALYSIS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--xml", type=Path, default=full_robot.DEFAULT_XML)
    parser.add_argument("--timestep", type=float, default=0.0002)
    parser.add_argument("--warmup-steps", type=int, default=3500)
    parser.add_argument(
        "--mjx-result",
        default="full_robot_mjx_history_replay_dt0p2ms_kinematic.json",
    )
    return parser.parse_args()


def _cpu_record(
    model: mujoco.MjModel,
    ids: dict[str, int],
    row: full_robot.PreparedRecord,
) -> tuple[dict[str, Any], np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = row.qpos
    data.qvel[:] = row.qvel
    data.ctrl[:] = row.ctrl
    mujoco.mj_forward(model, data)
    relative_positions = []
    contact_steps = 0
    contact_pairs: dict[str, int] = {}
    first_contact_step = None
    step_count = int(row.target_indices[-1]) + 10
    for step in range(step_count):
        velocity = data.qvel[ids["ball_vadr"] : ids["ball_vadr"] + 3]
        data.xfrc_applied[ids["ball_body"], :3] = (
            -contact.BALL_MASS_KG
            * history.PAPER_DRAG_PER_M
            * np.linalg.norm(velocity)
            * velocity
        )
        mujoco.mj_step(model, data)
        normal = data.geom_xmat[ids["racket_geom"]].reshape(3, 3)[:, 2]
        displacement = (
            data.qpos[ids["ball_qadr"] : ids["ball_qadr"] + 3]
            - data.geom_xpos[ids["racket_geom"]]
        )
        relative_positions.append(float(np.dot(displacement, normal)))
        for item in data.contact[: data.ncon]:
            if ids["ball_geom"] not in (int(item.geom1), int(item.geom2)):
                continue
            names = sorted(
                [
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom1)
                    ),
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, int(item.geom2)
                    ),
                ]
            )
            label = " <-> ".join(names)
            contact_pairs[label] = contact_pairs.get(label, 0) + 1
            contact_steps += 1
            if first_contact_step is None:
                first_contact_step = step
    position = np.asarray(relative_positions, dtype=np.float64)
    error = position[row.target_indices] - row.target_normal_positions_m
    return (
        {
            "recording": row.reconstruction.recording,
            "normal_position_rmse_mm": float(
                1000.0 * np.sqrt(np.mean(error * error))
            ),
            "normal_position_bias_mm": float(1000.0 * np.mean(error)),
            "first_ball_contact_step": first_contact_step,
            "ball_contact_constraint_samples": contact_steps,
            "ball_contact_geom_pairs": contact_pairs,
        },
        error,
    )


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.expanduser().resolve()
    mjx_result = json.loads(
        (analysis_dir / args.mjx_result).read_text(encoding="utf-8")
    )
    _, reconstructed, traces = full_robot._load_reconstructions(
        analysis_dir, full_robot.static_contact.DEFAULT_RECORD_ROOT
    )
    model, ids, _, _ = full_robot._build_model(
        args.xml.expanduser().resolve(), float(args.timestep)
    )
    source_damping = float(mjx_result["source_xml_runtime_damping_ratio"])
    model.pair_solref[ids["pair"], 1] = source_damping
    prepared = full_robot._prepare_records(
        model, ids, reconstructed, traces, int(args.warmup_steps)
    )
    cpu_rows = []
    cpu_errors = []
    for row in prepared:
        result, error = _cpu_record(model, ids, row)
        cpu_rows.append(result)
        cpu_errors.append(error)
    mjx_rows = {
        row["recording"]: row
        for row in mjx_result["evaluations"]["source_xml_runtime"]["rows"]
    }
    comparisons = []
    for cpu_row in cpu_rows:
        mjx_row = mjx_rows[cpu_row["recording"]]
        comparisons.append(
            {
                "recording": cpu_row["recording"],
                "cpu_normal_position_rmse_mm": cpu_row[
                    "normal_position_rmse_mm"
                ],
                "mjx_normal_position_rmse_mm": mjx_row[
                    "normal_position_rmse_mm"
                ],
                "absolute_rmse_delta_mm": abs(
                    cpu_row["normal_position_rmse_mm"]
                    - mjx_row["normal_position_rmse_mm"]
                ),
            }
        )
    cpu_all_rmse = float(
        1000.0 * np.sqrt(np.mean(np.concatenate(cpu_errors) ** 2))
    )
    mjx_all_rmse = float(
        mjx_result["evaluations"]["source_xml_runtime"][
            "all_position_samples_normal_rmse_mm"
        ]
    )
    expected_pair = "ball <-> racket_rubber_fore"
    only_expected_contact = all(
        set(row["ball_contact_geom_pairs"]) == {expected_pair}
        for row in cpu_rows
    )
    maximum_delta = max(row["absolute_rmse_delta_mm"] for row in comparisons)
    output = {
        "schema_version": 1,
        "purpose": "same-state CPU MuJoCo versus MJX contact response audit",
        "timestep_s": float(args.timestep),
        "solref_damping_ratio": source_damping,
        "only_expected_ball_racket_contact_geoms": only_expected_contact,
        "cpu_all_position_samples_normal_rmse_mm": cpu_all_rmse,
        "mjx_all_position_samples_normal_rmse_mm": mjx_all_rmse,
        "maximum_per_record_cpu_mjx_rmse_delta_mm": maximum_delta,
        "backend_consistency_limit_mm": 5.0,
        "backend_consistency_pass": bool(maximum_delta <= 5.0),
        "cpu_records": cpu_rows,
        "comparisons": comparisons,
    }
    output_path = analysis_dir / "full_robot_contact_backend_audit.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["full_robot_contact_backend_audit"] = output
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
