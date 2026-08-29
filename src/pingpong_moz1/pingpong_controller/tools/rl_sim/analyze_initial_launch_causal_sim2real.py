#!/usr/bin/env python3
"""Causal analysis of the initial ball-on-racket launch in probe 141442.

The recorded online mirror copied the measured ball state into MuJoCo on
every tick.  Consequently it is useful for robot/RMP diagnostics, but it is
not an independent ball-contact experiment.  This script preserves the
recorded V91 checkpoint and recovered-RMP contract while letting the ball
evolve independently for explicit initialization/contact counterfactuals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_mpl_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "src/pingpong_moz1"
REPLAY_SRC = ROOT / "src/pingpong_replay"
for item in (str(SRC), str(REPLAY_SRC), str(ROOT)):
    if item not in sys.path:
        sys.path.insert(0, item)

from pingpong_controller.rl_policy import (  # noqa: E402
    OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
)
from pingpong_controller.sim_mirror import (  # noqa: E402
    RealBallSimMirror,
    SimMirrorConfig,
)


RECORDING_REL = Path(
    "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/"
    "probe_recording_20260829_141442"
)
HISTORICAL_XML_REL = Path(
    "simcode6/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "92b5d39592a81236ee5ee5806499a0a8d4b422f131b312ad87ea1efc90507c66"
)
EXPECTED_XML_SHA256 = (
    "7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92"
)
DT = 0.005
BALL_RADIUS_M = 0.020
RACKET_SURFACE_FROM_SITE_M = 0.002
REST_CENTER_DISTANCE_M = BALL_RADIUS_M + RACKET_SURFACE_FROM_SITE_M
HISTORICAL_MASS_KG = 0.0027
PHYSICAL_MASS_KG = 0.0040
HISTORICAL_SOLREF = (0.003, 0.80)
IDENTIFIED_SOLREF = (0.003, 0.152540)


class QuietLogger:
    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    warning = info
    warn = info
    error = info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value


def vec(value: Any, size: int) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)[:size]
    except (TypeError, ValueError):
        return None
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        return None
    return result


def real_q_dq(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    contact = row.get("contactIdentification") or {}
    ros = row.get("ros") or {}
    q = vec(contact.get("rawJointPosition"), 7)
    dq = vec(contact.get("rawJointVelocity"), 7)
    if q is None:
        q = vec(ros.get("jointPosition"), 7)
    if dq is None:
        dq = vec(ros.get("jointVelocity"), 7)
    if q is None or dq is None:
        raise RuntimeError("launch row lacks coherent q/dq")
    return q, dq


def robust_local_polynomial(
    time: np.ndarray,
    values: np.ndarray,
    half_window_s: float = 0.040,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth camera positions and differentiate without logged velocity."""
    smooth = np.full_like(values, np.nan, dtype=np.float64)
    velocity = np.full_like(values, np.nan, dtype=np.float64)
    for center_index, center in enumerate(time):
        selected = np.flatnonzero(np.abs(time - center) <= half_window_s)
        if selected.size < 5:
            selected = np.argsort(np.abs(time - center))[: min(7, len(time))]
        local_t = time[selected] - center
        local_y = values[selected]
        keep = np.ones(selected.size, dtype=bool)
        for _ in range(2):
            if np.count_nonzero(keep) < 4:
                break
            predictions = np.empty_like(local_y)
            coefficients: list[np.ndarray] = []
            for axis in range(values.shape[1]):
                coefficient = np.polyfit(local_t[keep], local_y[keep, axis], 2)
                coefficients.append(coefficient)
                predictions[:, axis] = np.polyval(coefficient, local_t)
            residual = np.linalg.norm(predictions - local_y, axis=1)
            median = float(np.median(residual[keep]))
            mad = float(np.median(np.abs(residual[keep] - median)))
            threshold = median + max(0.0010, 3.5 * 1.4826 * mad)
            keep = residual <= threshold
        for axis in range(values.shape[1]):
            coefficient = np.polyfit(local_t[keep], local_y[keep, axis], 2)
            smooth[center_index, axis] = coefficient[2]
            velocity[center_index, axis] = coefficient[1]
    return smooth, velocity


def first_major_apex(
    time: np.ndarray,
    z: np.ndarray,
    *,
    low_s: float = 0.10,
    high_s: float = 0.35,
) -> tuple[int, float, float]:
    mask = np.isfinite(z) & (time >= low_s) & (time <= high_s)
    indices = np.flatnonzero(mask)
    if not indices.size:
        raise RuntimeError("no samples in initial-apex window")
    for index in indices[1:-1]:
        if z[index - 1] < z[index] and z[index] >= z[index + 1]:
            return int(index), float(time[index]), float(z[index])
    index = int(indices[np.nanargmax(z[indices])])
    return index, float(time[index]), float(z[index])


def shade_holds(axis: plt.Axes, elapsed: np.ndarray, executed: np.ndarray) -> None:
    active = False
    start = 0.0
    for index, (current, is_executed) in enumerate(zip(elapsed, executed)):
        if not is_executed and not active:
            active = True
            start = float(current) - DT / 2.0
        at_end = index == len(elapsed) - 1
        if active and (is_executed or at_end):
            stop = float(current) + (DT / 2.0 if at_end else -DT / 2.0)
            axis.axvspan(start, stop, color="#b8b8b8", alpha=0.22, lw=0)
            active = False


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    init_mode: str
    mass_kg: float
    solref_damping: float
    use_real_hold_mask: bool = False
    observation_mode: str = "truth"


def mirror_config(checkpoint: Path, historical_xml: Path) -> SimMirrorConfig:
    return SimMirrorConfig(
        model_path=checkpoint,
        policy_xml_path=historical_xml,
        sim_xml_path=historical_xml,
        policy_obs_dim=57,
        control_dt=DT,
        backend="mujoco",
        enable_video_frames=False,
        use_mujoco_renderer=False,
        policy_profile=OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
        policy_model="v91",
        simcode7_model_path=checkpoint,
        simcode7_v91_model_path=checkpoint,
        simcode7_policy_xml_path=historical_xml,
        simcode7_sim_xml_path=historical_xml,
    )


def make_mirror(
    checkpoint: Path,
    historical_xml: Path,
    *,
    mass_kg: float,
    solref_damping: float,
) -> RealBallSimMirror:
    mirror = RealBallSimMirror(
        mirror_config(checkpoint, historical_xml), logger=QuietLogger())
    ball_geom = mj.mj_name2id(
        mirror.model, mj.mjtObj.mjOBJ_GEOM, "ball")
    racket_geom = mj.mj_name2id(
        mirror.model, mj.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")
    original_mass = float(mirror.model.body_mass[mirror.ball_body_id])
    ratio = float(mass_kg) / original_mass
    mirror.model.body_mass[mirror.ball_body_id] = float(mass_kg)
    mirror.model.body_inertia[mirror.ball_body_id] *= ratio
    solref = np.asarray([HISTORICAL_SOLREF[0], solref_damping])
    mirror.model.geom_solref[ball_geom] = solref
    mirror.model.geom_solref[racket_geom] = solref
    if mirror.model.npair:
        mirror.model.pair_solref[: mirror.model.npair] = solref
    return mirror


def point_world_to_base(mirror: RealBallSimMirror, world: np.ndarray) -> np.ndarray:
    base = mirror.data.qpos[mirror.base_qadr]
    c, s = math.cos(float(base[2])), math.sin(float(base[2]))
    delta = np.asarray(world[:2]) - base[:2]
    return np.asarray([
        c * delta[0] + s * delta[1],
        -s * delta[0] + c * delta[1],
        float(world[2]),
    ])


def vector_world_to_base(mirror: RealBallSimMirror, world: np.ndarray) -> np.ndarray:
    base = mirror.data.qpos[mirror.base_qadr]
    c, s = math.cos(float(base[2])), math.sin(float(base[2]))
    return np.asarray([
        c * float(world[0]) + s * float(world[1]),
        -s * float(world[0]) + c * float(world[1]),
        float(world[2]),
    ])


def ball_base_state(mirror: RealBallSimMirror) -> tuple[np.ndarray, np.ndarray]:
    position_world = np.asarray(
        mirror.data.xpos[mirror.ball_body_id], dtype=np.float64)
    position = point_world_to_base(mirror, position_world)
    base_q = mirror.data.qpos[mirror.base_qadr]
    base_dq = mirror.data.qvel[mirror.base_vadr]
    delta = position_world[:2] - base_q[:2]
    velocity_world = np.asarray(
        mirror.data.qvel[mirror.ball_vadr : mirror.ball_vadr + 3],
        dtype=np.float64,
    )
    relative_world = np.asarray([
        velocity_world[0] - base_dq[0] + base_dq[2] * delta[1],
        velocity_world[1] - base_dq[1] - base_dq[2] * delta[0],
        velocity_world[2],
    ])
    return position, vector_world_to_base(mirror, relative_world)


def physical_racket_state(
    mirror: RealBallSimMirror,
) -> tuple[np.ndarray, np.ndarray]:
    site_world = np.asarray(
        mirror.data.site_xpos[mirror.racket_site_id], dtype=np.float64)
    matrix = np.asarray(
        mirror.data.site_xmat[mirror.racket_site_id], dtype=np.float64,
    ).reshape(3, 3)
    normal = vector_world_to_base(mirror, matrix[:, 2])
    normal /= np.linalg.norm(normal)
    return point_world_to_base(mirror, site_world), normal


def actor_racket_position(mirror: RealBallSimMirror) -> np.ndarray:
    q = mirror.data.qpos[mirror.arm_qadr].astype(np.float32)
    position, ok = mirror.policy._compute_racket_pose_base(q)
    if not ok:
        raise RuntimeError("actor FK failed")
    return np.asarray(position, dtype=np.float64)


def set_initial_ball(
    mirror: RealBallSimMirror,
    mode: str,
    context: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    site, normal = physical_racket_state(mirror)
    if mode == "observed":
        position = context["observed_ball"].copy()
        velocity = np.zeros(3)
    elif mode == "physical_on_racket":
        position = (
            site + context["observed_tangent"]
            + normal * REST_CENTER_DISTANCE_M)
        velocity = np.zeros(3)
    elif mode == "mirror_default":
        position = context["default_ball"].copy()
        velocity = context["default_velocity"].copy()
    elif mode == "physical_default":
        position = site + context["default_delta"].copy()
        velocity = context["default_velocity"].copy()
    else:
        raise ValueError(f"unknown initial mode {mode!r}")
    mirror._set_ball_state(position, velocity)
    mj.mj_forward(mirror.model, mirror.data)
    return position, velocity


def contact_active(mirror: RealBallSimMirror) -> bool:
    ball = mj.mj_name2id(mirror.model, mj.mjtObj.mjOBJ_GEOM, "ball")
    racket = mj.mj_name2id(
        mirror.model, mj.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")
    desired = {ball, racket}
    return any(
        {int(mirror.data.contact[index].geom1),
         int(mirror.data.contact[index].geom2)} == desired
        for index in range(mirror.data.ncon)
    )


def snapshot_row(
    mirror: RealBallSimMirror,
    scenario: Scenario,
    elapsed: float,
    executed: bool,
) -> dict[str, Any]:
    ball, ball_velocity = ball_base_state(mirror)
    physical_racket, _normal = physical_racket_state(mirror)
    actor_racket = actor_racket_position(mirror)
    action = np.asarray(
        getattr(mirror.policy, "last_executed_action", np.zeros(7)),
        dtype=np.float64,
    )
    return {
        "scenario": scenario.name,
        "label": scenario.label,
        "elapsed_s": float(elapsed),
        "ball": ball,
        "ball_velocity": ball_velocity,
        "actor_racket": actor_racket,
        "physical_racket": physical_racket,
        "joint_position": mirror.data.qpos[mirror.arm_qadr].copy(),
        "joint_velocity": mirror.data.qvel[mirror.arm_vadr].copy(),
        "policy_action": action,
        "policy_executed": bool(executed),
        "contact": contact_active(mirror),
    }


def run_closed_loop(
    scenario: Scenario,
    checkpoint: Path,
    historical_xml: Path,
    q0: np.ndarray,
    dq0: np.ndarray,
    launch_rows: list[dict[str, Any]],
    context: dict[str, np.ndarray],
    horizon_s: float = 0.40,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mirror = make_mirror(
        checkpoint,
        historical_xml,
        mass_kg=scenario.mass_kg,
        solref_damping=scenario.solref_damping,
    )
    mirror.reset(q0)
    mirror.data.qvel[mirror.arm_vadr] = dq0
    initial_position, initial_velocity = set_initial_ball(
        mirror, scenario.init_mode, context)
    initial_actor_racket = actor_racket_position(mirror)
    initial_physical_racket, _ = physical_racket_state(mirror)
    rows = [snapshot_row(mirror, scenario, 0.0, True)]
    observation_position = initial_position.copy()
    observation_velocity = initial_velocity.copy()
    observation_age = 0.0
    last_target = q0.astype(np.float32).copy()
    step_count = int(round(horizon_s / DT))
    for step in range(step_count):
        ball_position, ball_velocity = ball_base_state(mirror)
        refresh = (
            step == 0
            or int(step * 90 / 200) != int((step - 1) * 90 / 200)
        )
        if refresh:
            observation_position = ball_position.copy()
            observation_velocity = ball_velocity.copy()
            observation_age = 0.0
        else:
            observation_age += DT

        recorded = launch_rows[min(step, len(launch_rows) - 1)]
        execute = (
            bool(recorded.get("policyExecuted"))
            if scenario.use_real_hold_mask else True)
        if execute:
            if scenario.observation_mode == "recorded":
                policy_input = recorded.get("policyInput") or {}
                input_position = vec(
                    policy_input.get("ball", recorded.get("measuredBall")), 3)
                input_velocity = vec(
                    policy_input.get(
                        "velocity", recorded.get("measuredVelocity")), 3)
                try:
                    input_age = float(policy_input.get(
                        "age", recorded.get("age", 0.0)))
                except (TypeError, ValueError):
                    input_age = 0.0
                if (
                    input_position is None
                    or input_velocity is None
                    or not bool(recorded.get("valid", True))
                    or input_age > 0.350
                ):
                    execute = False
            else:
                input_position = observation_position
                input_velocity = observation_velocity
                input_age = observation_age
        if execute:
            q = mirror.data.qpos[mirror.arm_qadr].astype(np.float32)
            dq = mirror.data.qvel[mirror.arm_vadr].astype(np.float32)
            last_target = mirror.policy.predict(
                input_position,
                input_velocity,
                ball_valid=True,
                ball_obs_age_s=float(input_age),
                dt=DT,
                arm_q=q,
                arm_dq=dq,
                observation_scales=(),
            )
        mirror._step_recovered_rmp(last_target)
        rows.append(snapshot_row(
            mirror, scenario, (step + 1) * DT, execute))
        ball_now, ball_velocity_now = ball_base_state(mirror)
        if (
            not np.all(np.isfinite(ball_now))
            or abs(float(ball_now[2])) > 5.0
            or np.linalg.norm(ball_velocity_now) > 25.0
        ):
            break

    time = np.asarray([row["elapsed_s"] for row in rows])
    ball_z = np.asarray([row["ball"][2] for row in rows])
    apex_index, apex_time, apex_z = first_major_apex(time, ball_z)
    result = {
        "scenario": scenario.name,
        "label": scenario.label,
        "init_mode": scenario.init_mode,
        "mass_kg": scenario.mass_kg,
        "solref": [HISTORICAL_SOLREF[0], scenario.solref_damping],
        "observation_mode": scenario.observation_mode,
        "real_hold_mask": scenario.use_real_hold_mask,
        "initial_ball_m": initial_position.tolist(),
        "initial_actor_racket_m": initial_actor_racket.tolist(),
        "initial_physical_racket_m": initial_physical_racket.tolist(),
        "initial_actor_minus_physical_racket_m": (
            initial_actor_racket - initial_physical_racket).tolist(),
        "apex_time_s": apex_time,
        "apex_z_m": apex_z,
        "height_gain_m": apex_z - float(initial_position[2]),
        "apex_vertical_velocity_m_s": float(
            rows[apex_index]["ball_velocity"][2]),
        "maximum_ball_vertical_speed_m_s": float(np.nanmax(np.abs([
            row["ball_velocity"][2] for row in rows
        ]))),
        "maximum_actor_racket_z_m": float(np.nanmax([
            row["actor_racket"][2] for row in rows
        ])),
        "maximum_physical_racket_z_m": float(np.nanmax([
            row["physical_racket"][2] for row in rows
        ])),
        "executed_fraction": float(np.mean([
            row["policy_executed"] for row in rows[1:]
        ])),
        "contact_tick_count": int(sum(row["contact"] for row in rows)),
        "sample_count": len(rows),
    }
    return rows, result


def published_target(row: dict[str, Any]) -> np.ndarray:
    command = row.get("command") or {}
    target = vec(command.get("publishedSafePosition"), 7)
    if target is None:
        policy_command = row.get("policyCommand")
        if isinstance(policy_command, dict):
            policy_command = policy_command.get("position")
        target = vec(policy_command, 7)
    if target is None:
        raise RuntimeError("row lacks published command target")
    return target


def run_command_replay(
    checkpoint: Path,
    historical_xml: Path,
    q0: np.ndarray,
    dq0: np.ndarray,
    launch_rows: list[dict[str, Any]],
    context: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scenario = Scenario(
        "published_command_replay_4g_historical",
        "实物发布目标回放 / 4.0 g / 历史接触",
        "physical_on_racket",
        PHYSICAL_MASS_KG,
        HISTORICAL_SOLREF[1],
    )
    mirror = make_mirror(
        checkpoint, historical_xml,
        mass_kg=scenario.mass_kg,
        solref_damping=scenario.solref_damping)
    mirror.reset(q0)
    mirror.data.qvel[mirror.arm_vadr] = dq0
    set_initial_ball(mirror, scenario.init_mode, context)
    output: list[dict[str, Any]] = []
    for index, real in enumerate(launch_rows):
        item = snapshot_row(
            mirror, scenario, index * DT, bool(real.get("policyExecuted")))
        real_q, real_dq = real_q_dq(real)
        item.update({
            "real_q": real_q,
            "real_dq": real_dq,
            "real_racket": vec(real.get("racket"), 3),
            "target": published_target(real),
            "mode": str(real.get("mode", "")),
        })
        output.append(item)
        mirror._step_recovered_rmp(item["target"])

    simulated_q = np.asarray([row["joint_position"] for row in output])
    real_q = np.asarray([row["real_q"] for row in output])
    simulated_dq = np.asarray([row["joint_velocity"] for row in output])
    real_dq = np.asarray([row["real_dq"] for row in output])
    simulated_racket = np.asarray([row["actor_racket"] for row in output])
    real_racket = np.asarray([row["real_racket"] for row in output])
    summary = {
        "sample_count": len(output),
        "joint_position_rms_rad": float(np.sqrt(np.mean(
            (simulated_q - real_q) ** 2))),
        "joint_position_per_joint_rms_rad": np.sqrt(np.mean(
            (simulated_q - real_q) ** 2, axis=0)).tolist(),
        "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(
            (simulated_dq - real_dq) ** 2))),
        "racket_position_rms_m": float(np.sqrt(np.mean(
            (simulated_racket - real_racket) ** 2))),
        "racket_z_rms_m": float(np.sqrt(np.mean(
            (simulated_racket[:, 2] - real_racket[:, 2]) ** 2))),
        "real_racket_z_max_m": float(np.max(real_racket[:, 2])),
        "sim_racket_z_max_m": float(np.max(simulated_racket[:, 2])),
        "sim_minus_real_peak_racket_z_m": float(
            np.max(simulated_racket[:, 2]) - np.max(real_racket[:, 2])),
    }
    return output, summary


def run_static_hold(
    checkpoint: Path,
    historical_xml: Path,
    q0: np.ndarray,
    context: dict[str, np.ndarray],
    damping: float,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    mirror = make_mirror(
        checkpoint, historical_xml,
        mass_kg=PHYSICAL_MASS_KG, solref_damping=damping)
    mirror.reset(q0)
    position, velocity = set_initial_ball(
        mirror, "mirror_default", context)
    output: list[dict[str, float]] = []
    departure_time: float | None = None
    for step in range(201):
        ball, ball_velocity = ball_base_state(mirror)
        racket, _normal = physical_racket_state(mirror)
        lateral = float(np.linalg.norm((ball - racket)[:2]))
        if departure_time is None and (
            lateral > 0.055 or ball[2] < racket[2]
        ):
            departure_time = step * DT
        output.append({
            "elapsed_s": step * DT,
            "ball_x_m": float(ball[0]),
            "ball_y_m": float(ball[1]),
            "ball_z_m": float(ball[2]),
            "ball_vz_m_s": float(ball_velocity[2]),
            "racket_z_m": float(racket[2]),
            "lateral_offset_m": lateral,
        })
        mirror._step_recovered_rmp(q0)
    summary = {
        "mass_kg": PHYSICAL_MASS_KG,
        "solref": [HISTORICAL_SOLREF[0], damping],
        "initial_ball_m": position.tolist(),
        "initial_velocity_m_s": velocity.tolist(),
        "departure_time_s": departure_time,
        "maximum_ball_z_m": max(row["ball_z_m"] for row in output),
        "minimum_ball_z_m": min(row["ball_z_m"] for row in output),
        "maximum_lateral_offset_m": max(
            row["lateral_offset_m"] for row in output),
    }
    return output, summary


def flatten_sim_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        "scenario": row["scenario"],
        "label": row["label"],
        "elapsed_s": row["elapsed_s"],
        "policy_executed": int(row["policy_executed"]),
        "contact": int(row["contact"]),
    }
    for key, size in (
        ("ball", 3), ("ball_velocity", 3),
        ("actor_racket", 3), ("physical_racket", 3),
        ("joint_position", 7), ("joint_velocity", 7),
        ("policy_action", 7),
    ):
        value = np.asarray(row[key]).reshape(size)
        for index, scalar in enumerate(value):
            result[f"{key}_{index}"] = float(scalar)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_real_launch(
    output: Path,
    elapsed_fresh: np.ndarray,
    measured: np.ndarray,
    smooth: np.ndarray,
    smooth_velocity: np.ndarray,
    real_elapsed: np.ndarray,
    real_racket: np.ndarray,
    real_racket_velocity: np.ndarray,
    executed: np.ndarray,
    apex_time: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].scatter(
        elapsed_fresh, measured[:, 2], s=15, color="#8ecae6",
        label="camera measured z", zorder=2)
    axes[0].plot(
        elapsed_fresh, smooth[:, 2], color="#126782", lw=2.2,
        label="robust local-polynomial z")
    axes[0].plot(
        real_elapsed, real_racket[:, 2], color="#e76f51", lw=2.0,
        label="real racket FK z")
    axes[0].axvline(apex_time, color="#6a4c93", ls="--", lw=1.4)
    axes[0].set_ylabel("height (m)")
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        elapsed_fresh, smooth_velocity[:, 2], color="#126782", lw=2.2,
        label="ball vz from smooth position")
    axes[1].plot(
        real_elapsed, real_racket_velocity[:, 2], color="#e76f51", lw=2.0,
        label="racket vz from encoder/Jacobian")
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_xlabel("elapsed from first policy tick (s)")
    axes[1].set_ylabel("vertical velocity (m/s)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.25)
    for axis in axes:
        shade_holds(axis, real_elapsed, executed)
    fig.suptitle(
        "Real initial launch: continuous camera position fit (gray = feedback hold)",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "01_real_initial_launch.png", dpi=180)
    plt.close(fig)


def plot_initial_geometry(output: Path, summary: dict[str, Any]) -> None:
    labels = ["camera observed", "physical rest", "recorded default"]
    values = [
        1000.0 * summary["initial_state"]["camera_normal_distance_m"],
        1000.0 * REST_CENTER_DISTANCE_M,
        1000.0 * summary["initial_state"]["default_normal_distance_m"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].bar(labels, values, color=["#e76f51", "#2a9d8f", "#457b9d"])
    axes[0].axhline(BALL_RADIUS_M * 1000.0, color="black", ls="--", lw=1,
                    label="ball radius")
    axes[0].set_ylabel("ball-center normal distance (mm)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.25)
    mismatch = summary["original_mirror"]["actor_minus_physical_racket_mean_m"]
    axes[1].bar(["x", "y", "z"], 1000.0 * np.asarray(mismatch),
                color=["#577590", "#43aa8b", "#f94144"])
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].set_ylabel("actor FK - collision racket (mm)")
    axes[1].set_title("Recorded original mirror internal geometry gap")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Initial-state geometry is not a single consistent state",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "02_initial_geometry_contract.png", dpi=180)
    plt.close(fig)


def plot_command_replay(
    output: Path,
    replay: list[dict[str, Any]],
) -> None:
    elapsed = np.asarray([row["elapsed_s"] for row in replay])
    real_racket = np.asarray([row["real_racket"] for row in replay])
    sim_racket = np.asarray([row["actor_racket"] for row in replay])
    real_q = np.asarray([row["real_q"] for row in replay])
    sim_q = np.asarray([row["joint_position"] for row in replay])
    executed = np.asarray([row["policy_executed"] for row in replay])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(elapsed, real_racket[:, 2], lw=2.2, label="real racket z")
    axes[0].plot(elapsed, sim_racket[:, 2], lw=2.2,
                 label="frozen recovered-RMP replay z")
    axes[0].set_ylabel("height (m)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(elapsed, np.linalg.norm(sim_q - real_q, axis=1),
                 color="#6a4c93", lw=2.0)
    axes[1].set_ylabel("||q_sim - q_real|| (rad)")
    axes[1].set_xlabel("elapsed (s)")
    axes[1].grid(alpha=0.25)
    for axis in axes:
        shade_holds(axis, elapsed, executed)
    fig.suptitle(
        "Same publishedSafePosition sequence: real actuator vs frozen mirror RMP",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "03_published_command_rmp_replay.png", dpi=180)
    plt.close(fig)


def plot_simulations(
    output: Path,
    runs: dict[str, list[dict[str, Any]]],
    results: list[dict[str, Any]],
    real_elapsed_fresh: np.ndarray,
    real_smooth: np.ndarray,
    real_apex_z: float,
) -> None:
    plot_labels = {
        "same_observed_init_4g_historical": "camera init, 4g, historical",
        "same_physical_on_racket_4g_historical": "physical on-racket, 4g",
        "same_physical_on_racket_2p7g_historical": "physical on-racket, 2.7g",
        "physical_default_4g_historical": "physical default, 4g",
        "recorded_mirror_default_4g_historical": "recorded mirror default, 4g",
        "physical_default_2p7g_historical": "physical default, 2.7g",
        "same_physical_truth_holdmask": "physical init, truth, hold mask",
        "same_physical_recorded_obs_holdmask": "physical init, real obs, hold mask",
        "same_physical_4g_identified_contact": "physical init, impact-fit contact",
    }
    selected_names = {
        "same_observed_init_4g_historical",
        "same_physical_on_racket_4g_historical",
        "physical_default_4g_historical",
        "recorded_mirror_default_4g_historical",
        "same_physical_recorded_obs_holdmask",
    }
    fig, axis = plt.subplots(figsize=(11, 6.5))
    axis.plot(real_elapsed_fresh, real_smooth[:, 2], color="black", lw=2.6,
              label="real robust trajectory")
    for result in results:
        if result["scenario"] not in selected_names:
            continue
        rows = runs[result["scenario"]]
        axis.plot(
            [row["elapsed_s"] for row in rows],
            [row["ball"][2] for row in rows],
            lw=1.8, label=plot_labels[result["scenario"]])
    axis.set_xlim(0.0, 0.38)
    axis.set_xlabel("elapsed (s)")
    axis.set_ylabel("ball z (m)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    axis.set_title("Independent ball physics with the frozen 141442 RMP")
    fig.tight_layout()
    fig.savefig(output / "04_initial_state_simulation_trajectories.png", dpi=180)
    plt.close(fig)

    result_by_name = {item["scenario"]: item for item in results}
    order = [item["scenario"] for item in results]
    values = [1000.0 * (result_by_name[name]["apex_z_m"] - real_apex_z)
              for name in order]
    labels = [plot_labels[name] for name in order]
    colors = ["#2a9d8f" if abs(value) <= 10 else (
        "#e9c46a" if abs(value) <= 30 else "#e76f51") for value in values]
    fig, axis = plt.subplots(figsize=(12, 7.2))
    positions = np.arange(len(order))
    axis.barh(positions, values, color=colors)
    axis.axvline(0.0, color="black", lw=1.0)
    axis.set_yticks(positions, labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("simulated apex - real robust apex (mm)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_title("Apex counterfactuals (4.0 g unless explicitly historical 2.7 g)")
    fig.tight_layout()
    fig.savefig(output / "05_apex_counterfactuals.png", dpi=180)
    plt.close(fig)


def plot_static_hold(
    output: Path,
    historical: list[dict[str, float]],
    identified: list[dict[str, float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, rows, title in (
        (axes[0], historical, "recording-time solref: 0.003 / 0.80"),
        (axes[1], identified, "impact-identified solref: 0.003 / 0.152540"),
    ):
        time = [row["elapsed_s"] for row in rows]
        axis.plot(time, [row["ball_z_m"] for row in rows], label="ball z")
        axis.plot(time, [row["racket_z_m"] for row in rows], label="racket z")
        axis2 = axis.twinx()
        axis2.plot(time, [1000.0 * row["lateral_offset_m"] for row in rows],
                   color="#6a4c93", alpha=0.65, label="lateral offset")
        axis.set_title(title)
        axis.set_xlabel("static hold time (s)")
        axis.set_ylabel("height (m)")
        axis2.set_ylabel("lateral offset (mm)", color="#6a4c93")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper left")
    fig.suptitle(
        "Placed-ball static hold is outside the validated impact-contact regime",
        fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output / "06_static_ball_on_racket_contact_test.png", dpi=180)
    plt.close(fig)


def write_report(output: Path, summary: dict[str, Any]) -> None:
    real = summary["real_launch"]
    initial = summary["initial_state"]
    executor = summary["executor"]
    replay = summary["command_replay"]
    original = summary["original_mirror"]
    results = {row["scenario"]: row for row in summary["simulations"]}
    static_historical = summary["static_hold"]["historical_contact"]
    static_identified = summary["static_hold"]["identified_contact"]

    def mm(value: float) -> str:
        return f"{1000.0 * value:.1f}"

    report = f"""# probe_recording_20260829_141442 初始起球因果分析

## 结论先行

这次实物起球的平滑球心峰值为 **{real['apex_z_m']:.4f} m**（首个 policy tick 后 {real['apex_time_s']:.3f} s），相对起始观测球心上升 **{mm(real['height_gain_m'])} mm**。判因结果不是“某一个碰撞恢复系数直接导致低起球”，而是一个更靠前的状态合同问题：**实物放拍状态、相机给 actor 的状态、训练/默认 reset 状态和原始 mirror 的碰撞几何不是同一个初态。**

优先级判断：

1. **首要：初态/观测几何合同不一致。** 相机初态的球心法向距离只有 {mm(initial['camera_normal_distance_m'])} mm，小于 20 mm 球半径，几何上不可能；记录默认 reset 是 {mm(initial['default_normal_distance_m'])} mm。actor 因而看到的 `ball-racket rel_z` 比默认状态少约 {mm(initial['default_normal_distance_m'] - initial['camera_normal_distance_m'])} mm。
2. **首要并列：原始 mirror 不能独立验证起球碰撞。** 它每 tick 强制写回实物球状态；同时 actor FK 与实际碰撞拍面 z 平均相差 {mm(original['actor_minus_physical_racket_mean_m'][2])} mm。
3. **执行器不是“实物起得低”的单独主因。** 相同 `publishedSafePosition` 回放下，冻结的历史 RMP 反而比实物更慢/更低：仿真拍面峰值比实物低 {mm(-replay['sim_minus_real_peak_racket_z_m'])} mm。这说明不能用该冷启动 mirror 回放声称实物执行器力度不足。
4. **反馈 hold 不是主因。** 起球至峰值间 policy 实际执行率 {100.0 * executor['executed_fraction_to_apex']:.1f}%，但保留真实 hold 掩码的反事实没有稳定降低峰值；hold 改变闭环相位，影响是非单调的。
5. **碰撞参数仍然重要，但本段不能单独辨识。** 4.0 g 与 2.7 g、初始间隙以及 `solref` 的耦合可改变峰值几十毫米；而只用冲击反弹拟合出的低 damping 参数，在静态放拍实验中产生非实物能量，不能直接用于起球准静态接触。

因此，如果问题表述为“为什么实物首起高度低于期望”，最可信的回答是：**策略从一个训练中不存在、且相机几何不自洽的放拍观测状态启动；原始 mirror 又没有真正模拟这个接触。执行器与 hold 不是现有证据下的第一嫌疑，碰撞参数必须在修正初态/拍面坐标并加入静态预接触状态后再标定。**

## 冻结的历史仿真合同

- V91 checkpoint SHA-256：`{summary['contract']['checkpoint_sha256']}`。
- 记录时 policy XML SHA-256：`{summary['contract']['historical_xml_sha256']}`；脚本使用仓库内仍保留该精确哈希的副本，没有使用当前 simcode7 XML。
- control dt = 5 ms；MuJoCo = 1 ms × 5；RMP target-update delay = {summary['contract']['rmp_target_update_delay_steps']} step。
- RMP output delay：`{summary['contract']['rmp_output_delay_steps_per_joint']}`。
- 历史 nominal 接触：球 2.7 g，`solref=0.003 0.80`。按你的约束，主反事实把球质量改为 **4.0 g**；2.7 g 仅保留为记录时复现基线。
- 原始在线 backend 是 `mujoco_recovered_rmp_mirror`，不是独立 MJX 球物理；本报告用同一 checkpoint 内嵌 RMP 参数做 CPU MuJoCo 独立球物理实验。
- 独立闭环采用确定性的 90 Hz sample-and-hold 真值观测，不开启随机 DR；因此它是单变量判因实验，不是随机训练分布的统计评估。

## 实物起球数据质量

- 分析窗口：sourceTime {real['start_source_time_s']:.3f}–{real['apex_source_time_s']:.3f} s。
- 使用 {real['fresh_camera_sample_count']} 个 cameraStamp 去重后的连续位置样本；最大相邻帧间隔 {1000.0 * real['maximum_camera_gap_s']:.1f} ms。
- 球速度没有采用记录中的逐帧 `measuredVelocity` 作主证据，而是由连续位置做 80 ms 鲁棒局部二次拟合求导；位置拟合 RMS 为 {mm(real['position_fit_rms_m'])} mm。
- 平滑峰值：{real['apex_z_m']:.4f} m；原始位置峰值：{real['raw_apex_z_m']:.4f} m；二者差 {mm(real['raw_apex_z_m'] - real['apex_z_m'])} mm。
- 实物拍面最高 {real['racket_z_max_m']:.4f} m，最大向上速度 {real['racket_vz_max_m_s']:.3f} m/s。

灰色区是 q/dq feedback hold：[`01_real_initial_launch.png`](01_real_initial_launch.png)。

## 初态：手上“同一个球”在四个合同里不是同一个状态

| 状态 | 球心/拍面法向距离 | 说明 |
|---|---:|---|
| 相机稳定初态 | {mm(initial['camera_normal_distance_m'])} mm | 小于球半径，不可作为真实碰撞 qpos |
| 几何约束的实物放拍 | {1000.0 * REST_CENTER_DISTANCE_M:.1f} mm | 20 mm 半径 + 2 mm site→表面 |
| 记录默认 reset | {mm(initial['default_normal_distance_m'])} mm | 含 {mm(initial['default_surface_gap_m'])} mm 自由间隙 |

实物初始关节角与 checkpoint 默认关节角 RMS 只有 {initial['arm_q_default_rms_rad']:.6f} rad，说明**臂姿态本身不是主要初态差异，球相对拍面的状态才是。** 详见 [`02_initial_geometry_contract.png`](02_initial_geometry_contract.png)。

## 原始 mirror 的能力边界

记录中的 {original['sample_count']} 个原始 mirror 样本均由实物 policy ball 输入驱动。代码路径在每个 MuJoCo step 前后调用 `_set_ball_state`，所以：

- 原始文件能回答 q/dq、RMP 延迟和 actor 输出差异；
- 不能回答球是否被拍面弹起、恢复系数是否正确；
- 同一 tick 的 actor 观测拍面与碰撞拍面平均差为 `[{', '.join(f'{1000.0*x:.1f}' for x in original['actor_minus_physical_racket_mean_m'])}] mm`。

这是本次必须另跑独立球物理的原因。

## 执行器/RMP 回放

对每个实物 tick 使用真正发布的 `command.publishedSafePosition`（feedback hold 时它保留上次安全目标，不能误用诊断字段 `policyCommand` 的当前 q）：

- q RMS：{replay['joint_position_rms_rad']:.4f} rad；dq RMS：{replay['joint_velocity_rms_rad_s']:.3f} rad/s。
- 拍面位置 RMS：{mm(replay['racket_position_rms_m'])} mm；z RMS：{mm(replay['racket_z_rms_m'])} mm。
- 实物/冻结 RMP 回放拍面峰值：{replay['real_racket_z_max_m']:.4f} / {replay['sim_racket_z_max_m']:.4f} m。

历史 mirror RMP 对同一发布目标的响应比实物低，而不是更高。这个方向与“实物执行器太弱导致低起球”不一致。详见 [`03_published_command_rmp_replay.png`](03_published_command_rmp_replay.png)。

## 独立球物理反事实

| 仿真 | 初态 | 质量 | 接触 damping | 峰值 z | 相对实物 |
|---|---|---:|---:|---:|---:|
"""
    for item in summary["simulations"]:
        report += (
            f"| {item['label']} | `{item['init_mode']}` | "
            f"{1000.0 * item['mass_kg']:.1f} g | "
            f"{item['solref'][1]:.6f} | {item['apex_z_m']:.4f} m | "
            f"{1000.0 * (item['apex_z_m'] - real['apex_z_m']):+.1f} mm |\n"
        )
    report += f"""

读表时要注意两个非线性结果：

- “相机观测坐标直接当物理 qpos”的 4.0 g 仿真峰值为 {results['same_observed_init_4g_historical']['apex_z_m']:.4f} m，接近实物，但它从球拍穿透/接触边界启动，**数值接近不等于物理正确**。
- 用碰撞拍面构造真正的 4.0 g 放拍状态后，历史接触峰值为 {results['same_physical_on_racket_4g_historical']['apex_z_m']:.4f} m；用真正的碰撞拍面构造默认 gap 后为 {results['physical_default_4g_historical']['apex_z_m']:.4f} m。结果对 gap、质量和接触状态高度敏感。
- 在相同物理放拍初态与相同历史接触下，2.7 g / 4.0 g 峰值为 {results['same_physical_on_racket_2p7g_historical']['apex_z_m']:.4f} / {results['same_physical_on_racket_4g_historical']['apex_z_m']:.4f} m；这正是为什么主结论必须固定你给出的 4.0 g，不能沿用记录时 2.7 g。
- 喂入实物记录观测并保留真实 hold 掩码的 hybrid 仿真峰值为 {results['same_physical_recorded_obs_holdmask']['apex_z_m']:.4f} m，高于 perfect-truth/hold-mask 的 {results['same_physical_truth_holdmask']['apex_z_m']:.4f} m。就本段符号而言，观测没有让 actor 更保守；它让闭环更激进，因此“观测导致命令太小”不成立。

轨迹见 [`04_initial_state_simulation_trajectories.png`](04_initial_state_simulation_trajectories.png)，峰值对比见 [`05_apex_counterfactuals.png`](05_apex_counterfactuals.png)。

## 静态放拍检查：冲击拟合不能直接外推

从记录默认 gap 开始、拍面保持 q0：

- 历史 `0.003/0.80` 下，球在 {static_historical['departure_time_s']:.3f} s 后横向离开有效拍面，说明该模型没有重现实物在起球前长时间静止于拍面的状态。
- 冲击数据拟合的 `0.003/0.152540` 在这个准静态场景中把球推到 {static_identified['maximum_ball_z_m']:.3f} m，出现明显非实物储能/释放。

这不否定冲击反弹辨识，而是说明它只约束了有限入射速度下的恢复行为；放拍启动还需要静态接触、滚动/旋转摩擦、初始 penetration/solver warm state 的约束。见 [`06_static_ball_on_racket_contact_test.png`](06_static_ball_on_racket_contact_test.png)。

## 对四个候选原因的最终判断

| 候选原因 | 判断 | 证据 |
|---|---|---|
| 初始状态 | **主要问题** | 球相对拍面相差约 22 mm；实物是预接触放拍，默认仿真是带 gap 的即时启动 |
| 执行器/RMP | **不是低高度主因，但 mirror 模型不够准** | 同命令回放的历史 RMP 比实物更低、更慢 |
| 球/球拍碰撞 | **重要且当前未被该场景可靠约束** | 质量/gap/solref 强耦合；低 damping 在静态启动中非物理 |
| 球观测 | **状态合同主因之一，但不是“命令变小”** | 初态几何不可行；recorded-observation hybrid 反而更高 |
| feedback hold | **次要/非单调** | {100.0 * (1.0 - executor['executed_fraction_to_apex']):.1f}% 起球 tick 未推理，但 hold 反事实未稳定降低峰值 |

## 建议的下一轮最小实验

1. 在仿真 reset 中增加 `ball_on_racket_settled` 模式：球心以碰撞拍面而非 actor FK 放置，保存 contact solver warm state，起球前至少静置 0.3 s 且横向漂移 < 2 mm。
2. 在线记录同时输出 `actor_racket_pose` 与 `collision_racket_pose`，把当前约 10.7 mm z 差消掉后再做碰撞辨识。
3. 实物起球前拍摄固定球标定：用球半径约束修正法向偏差；actor 可继续用平滑观测，但物理初始化不要直接使用相机球心 qpos。
4. 用 4.0 g 固定质量，分开标定：静态/低速承托参数与 0.5–2.0 m/s 冲击恢复参数；不要只靠一个 `solref damping` 同时拟合两种工况。
5. 修正初态合同后再比较 hold on/off；当前状态差异大于 hold 的可解释效应。

## 产物

- `summary.json`：全部数值结论与冻结合同。
- `real_launch_samples.csv`：cameraStamp 去重、平滑后的实物球/拍数据。
- `simulation_timeseries.csv`：所有独立物理反事实逐 tick 数据。
- `published_command_replay.csv`：相同发布目标的实物/历史 RMP 执行器对比。
- `static_hold_timeseries.csv`：两套接触参数的放拍静态检查。
"""
    (output / "SIM2REAL_INITIAL_LAUNCH_CAUSAL_ANALYSIS_ZH.md").write_text(
        report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, default=ROOT / RECORDING_REL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    recording = args.recording.expanduser().resolve()
    output = (
        args.output.expanduser().resolve() if args.output else
        recording / "analysis/v91_initial_launch_causal"
    )
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((recording / "metadata.json").read_text(
        encoding="utf-8"))
    config = metadata["config"]
    checkpoint = Path(config["policy"]["modelPath"]).resolve()
    historical_xml = (ROOT / HISTORICAL_XML_REL).resolve()
    checkpoint_hash = sha256(checkpoint)
    xml_hash = sha256(historical_xml)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"checkpoint hash changed: {checkpoint_hash}")
    if xml_hash != EXPECTED_XML_SHA256:
        raise RuntimeError(f"historical XML hash changed: {xml_hash}")

    control_rows: list[dict[str, Any]] = []
    for row in read_jsonl(recording / "control_samples.jsonl"):
        time = float(row.get("time", -math.inf))
        if time >= 17.35:
            control_rows.append(row)
        if time > 18.15:
            break
    first_index = next(
        index for index, row in enumerate(control_rows)
        if row.get("mode") == "policy" and bool(row.get("policyExecuted")))
    start_row = control_rows[first_index]
    start_time = float(start_row["time"])
    launch_rows = [
        row for row in control_rows
        if start_time <= float(row["time"]) <= start_time + 0.40
    ]
    q0, dq0 = real_q_dq(start_row)
    racket0 = vec(start_row.get("racket"), 3)
    contact0 = start_row.get("contactIdentification") or {}
    normal0 = vec(contact0.get("racketFaceNormal"), 3)
    if racket0 is None or normal0 is None:
        raise RuntimeError("first launch row lacks racket pose/normal")
    normal0 /= np.linalg.norm(normal0)

    fresh_rows: list[dict[str, Any]] = []
    last_stamp: float | None = None
    for row in control_rows:
        time = float(row.get("time", -math.inf))
        if not start_time - 0.18 <= time <= start_time + 0.40:
            continue
        stamp = row.get("cameraStampS")
        measured = vec(row.get("measuredBall"), 3)
        if stamp is None or measured is None or stamp == last_stamp:
            continue
        fresh_rows.append(row)
        last_stamp = stamp
    fresh_time_absolute = np.asarray([float(row["time"]) for row in fresh_rows])
    fresh_elapsed = fresh_time_absolute - start_time
    measured_ball = np.asarray([vec(row.get("measuredBall"), 3)
                                for row in fresh_rows])
    smooth_ball, smooth_velocity = robust_local_polynomial(
        fresh_elapsed, measured_ball)
    launch_mask = (fresh_elapsed >= 0.10) & (fresh_elapsed <= 0.35)
    apex_fresh_index = np.flatnonzero(launch_mask)[
        np.argmax(smooth_ball[launch_mask, 2])]
    apex_elapsed = float(fresh_elapsed[apex_fresh_index])
    apex_z = float(smooth_ball[apex_fresh_index, 2])
    prelaunch_mask = (fresh_elapsed >= -0.15) & (fresh_elapsed <= 0.015)
    observed_ball0 = np.median(measured_ball[prelaunch_mask], axis=0)
    position_fit_rms = float(np.sqrt(np.mean(
        (measured_ball[launch_mask] - smooth_ball[launch_mask]) ** 2)))

    real_rows = [
        row for row in launch_rows
        if float(row["time"]) <= start_time + apex_elapsed + 0.0026
    ]
    real_elapsed = np.asarray([float(row["time"]) - start_time
                               for row in real_rows])
    executed = np.asarray([bool(row.get("policyExecuted")) for row in real_rows])
    real_racket = np.asarray([vec(row.get("racket"), 3) for row in real_rows])
    real_racket_velocity = np.asarray([
        vec((row.get("contactIdentification") or {}).get(
            "racketLinearVelocity"), 3)
        for row in real_rows
    ])
    modes = Counter(str(row.get("mode", "")) for row in real_rows)
    longest_hold = 0
    current_hold = 0
    for value in executed:
        current_hold = 0 if value else current_hold + 1
        longest_hold = max(longest_hold, current_hold)

    reset_ball = config["policy"]["resetBall"]
    default_ball = np.asarray(reset_ball["position"], dtype=np.float64)
    default_velocity = np.asarray(reset_ball["velocity"], dtype=np.float64)
    default_racket = np.asarray(reset_ball["racketCenter"], dtype=np.float64)
    default_normal = np.asarray(reset_ball["racketNormal"], dtype=np.float64)
    default_normal /= np.linalg.norm(default_normal)
    default_delta = default_ball - default_racket
    camera_normal_distance = float(np.dot(
        observed_ball0 - racket0, normal0))
    default_normal_distance = float(np.dot(default_delta, default_normal))
    observed_delta = observed_ball0 - racket0
    observed_tangent = observed_delta - normal0 * float(np.dot(
        observed_delta, normal0))
    default_q = np.deg2rad(np.asarray(
        config.get("initialArmJointPositionDeg", (
            5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214)),
        dtype=np.float64))
    context = {
        "observed_ball": observed_ball0,
        "observed_tangent": observed_tangent,
        "default_ball": default_ball,
        "default_velocity": default_velocity,
        "default_delta": default_delta,
    }

    original_mirror_rows = list(read_jsonl(
        recording / "sim_control_samples.jsonl"))
    original_gaps = []
    for row in original_mirror_rows:
        observation = vec(row.get("policyObs"), 57)
        physical = vec(row.get("racket"), 3)
        if observation is not None and physical is not None:
            original_gaps.append(observation[26:29] - physical)
    original_gaps_array = np.asarray(original_gaps)

    scenarios = [
        Scenario(
            "same_observed_init_4g_historical",
            "同相机初态 / 4.0g / 历史接触",
            "observed", PHYSICAL_MASS_KG, HISTORICAL_SOLREF[1]),
        Scenario(
            "same_physical_on_racket_4g_historical",
            "同物理放拍初态 / 4.0g / 历史接触",
            "physical_on_racket", PHYSICAL_MASS_KG, HISTORICAL_SOLREF[1]),
        Scenario(
            "same_physical_on_racket_2p7g_historical",
            "同物理放拍初态 / 历史2.7g",
            "physical_on_racket", HISTORICAL_MASS_KG,
            HISTORICAL_SOLREF[1]),
        Scenario(
            "physical_default_4g_historical",
            "碰撞拍面默认初态 / 4.0g / 历史接触",
            "physical_default", PHYSICAL_MASS_KG, HISTORICAL_SOLREF[1]),
        Scenario(
            "recorded_mirror_default_4g_historical",
            "记录 mirror 默认坐标 / 4.0g / 历史接触",
            "mirror_default", PHYSICAL_MASS_KG, HISTORICAL_SOLREF[1]),
        Scenario(
            "physical_default_2p7g_historical",
            "碰撞拍面默认初态 / 历史2.7g",
            "physical_default", HISTORICAL_MASS_KG, HISTORICAL_SOLREF[1]),
        Scenario(
            "same_physical_truth_holdmask",
            "同物理初态 / truth / 实物hold掩码",
            "physical_on_racket", PHYSICAL_MASS_KG, HISTORICAL_SOLREF[1],
            use_real_hold_mask=True),
        Scenario(
            "same_physical_recorded_obs_holdmask",
            "同物理初态 / 实物观测 / 实物hold掩码",
            "physical_on_racket", PHYSICAL_MASS_KG, HISTORICAL_SOLREF[1],
            use_real_hold_mask=True, observation_mode="recorded"),
        Scenario(
            "same_physical_4g_identified_contact",
            "同物理初态 / 4.0g / 冲击辨识接触",
            "physical_on_racket", PHYSICAL_MASS_KG, IDENTIFIED_SOLREF[1]),
    ]
    simulation_runs: dict[str, list[dict[str, Any]]] = {}
    simulation_results: list[dict[str, Any]] = []
    for scenario in scenarios:
        rows, result = run_closed_loop(
            scenario, checkpoint, historical_xml, q0, dq0,
            launch_rows, context)
        simulation_runs[scenario.name] = rows
        simulation_results.append(result)

    command_replay, command_replay_summary = run_command_replay(
        checkpoint, historical_xml, q0, dq0, launch_rows, context)
    static_historical_rows, static_historical_summary = run_static_hold(
        checkpoint, historical_xml, q0, context, HISTORICAL_SOLREF[1])
    static_identified_rows, static_identified_summary = run_static_hold(
        checkpoint, historical_xml, q0, context, IDENTIFIED_SOLREF[1])

    contract_mirror = make_mirror(
        checkpoint, historical_xml,
        mass_kg=HISTORICAL_MASS_KG,
        solref_damping=HISTORICAL_SOLREF[1])
    summary = {
        "contract": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "historical_xml": str(historical_xml),
            "historical_xml_sha256": xml_hash,
            "recorded_backend": "mujoco_recovered_rmp_mirror",
            "control_dt_s": DT,
            "mujoco_timestep_s": float(contract_mirror.model.opt.timestep),
            "frame_skip": contract_mirror.substeps,
            "rmp_target_update_delay_steps": int(
                contract_mirror.rmp_target_update_delay_steps),
            "rmp_output_delay_steps_per_joint": (
                contract_mirror.rmp_output_delay_steps.tolist()),
            "right_arm_controller_kp": contract_mirror.rmp_controller_kp.tolist(),
            "right_arm_controller_kv": contract_mirror.rmp_controller_kv.tolist(),
            "recording_time_ball_mass_kg": HISTORICAL_MASS_KG,
            "physical_ball_mass_kg": PHYSICAL_MASS_KG,
            "recording_time_solref": list(HISTORICAL_SOLREF),
            "identified_impact_solref_sensitivity": list(IDENTIFIED_SOLREF),
        },
        "real_launch": {
            "start_source_time_s": start_time,
            "apex_source_time_s": start_time + apex_elapsed,
            "apex_time_s": apex_elapsed,
            "apex_z_m": apex_z,
            "raw_apex_z_m": float(np.max(measured_ball[launch_mask, 2])),
            "initial_observed_ball_m": observed_ball0.tolist(),
            "height_gain_m": apex_z - float(observed_ball0[2]),
            "fresh_camera_sample_count": int(np.count_nonzero(
                (fresh_elapsed >= 0.0) & (fresh_elapsed <= apex_elapsed))),
            "maximum_camera_gap_s": float(np.max(np.diff(
                fresh_time_absolute))),
            "position_fit_rms_m": position_fit_rms,
            "racket_z_max_m": float(np.max(real_racket[:, 2])),
            "racket_vz_max_m_s": float(np.max(real_racket_velocity[:, 2])),
        },
        "initial_state": {
            "real_arm_q_rad": q0.tolist(),
            "real_arm_dq_rad_s": dq0.tolist(),
            "checkpoint_default_arm_q_rad": default_q.tolist(),
            "arm_q_default_rms_rad": float(np.sqrt(np.mean((q0-default_q)**2))),
            "camera_ball_m": observed_ball0.tolist(),
            "real_racket_m": racket0.tolist(),
            "real_racket_normal": normal0.tolist(),
            "camera_normal_distance_m": camera_normal_distance,
            "camera_tangent_offset_m": observed_tangent.tolist(),
            "camera_tangent_offset_norm_m": float(np.linalg.norm(
                observed_tangent)),
            "default_ball_m": default_ball.tolist(),
            "default_racket_m": default_racket.tolist(),
            "default_normal_distance_m": default_normal_distance,
            "default_surface_gap_m": float(reset_ball["surfaceGapM"]),
        },
        "executor": {
            "mode_counts_to_apex": dict(modes),
            "tick_count_to_apex": len(real_rows),
            "executed_tick_count_to_apex": int(np.count_nonzero(executed)),
            "executed_fraction_to_apex": float(np.mean(executed)),
            "longest_hold_ticks": longest_hold,
            "longest_hold_duration_s": longest_hold * DT,
        },
        "original_mirror": {
            "sample_count": len(original_gaps_array),
            "ball_physics_independent": False,
            "ball_state_forced_each_tick": True,
            "actor_minus_physical_racket_mean_m": np.mean(
                original_gaps_array, axis=0).tolist(),
            "actor_minus_physical_racket_rms_m": np.sqrt(np.mean(
                original_gaps_array ** 2, axis=0)).tolist(),
        },
        "command_replay": command_replay_summary,
        "simulations": simulation_results,
        "static_hold": {
            "historical_contact": static_historical_summary,
            "identified_contact": static_identified_summary,
        },
    }

    real_csv = []
    for index, row in enumerate(fresh_rows):
        real_csv.append({
            "source_time_s": float(row["time"]),
            "elapsed_s": float(fresh_elapsed[index]),
            "camera_stamp_s": float(row["cameraStampS"]),
            **{f"measured_ball_{axis}_m": measured_ball[index, axis]
               for axis in range(3)},
            **{f"smooth_ball_{axis}_m": smooth_ball[index, axis]
               for axis in range(3)},
            **{f"smooth_velocity_{axis}_m_s": smooth_velocity[index, axis]
               for axis in range(3)},
        })
    write_csv(output / "real_launch_samples.csv", real_csv)
    flattened_sim = [
        flatten_sim_row(row)
        for scenario_rows in simulation_runs.values()
        for row in scenario_rows
    ]
    write_csv(output / "simulation_timeseries.csv", flattened_sim)
    replay_csv = []
    for row in command_replay:
        item = flatten_sim_row(row)
        item["mode"] = row["mode"]
        for prefix, value in (
            ("real_q", row["real_q"]),
            ("real_dq", row["real_dq"]),
            ("real_racket", row["real_racket"]),
            ("published_target", row["target"]),
        ):
            for index, scalar in enumerate(value):
                item[f"{prefix}_{index}"] = float(scalar)
        replay_csv.append(item)
    write_csv(output / "published_command_replay.csv", replay_csv)
    static_csv = []
    for name, rows in (
        ("historical_0p80", static_historical_rows),
        ("identified_0p152540", static_identified_rows),
    ):
        static_csv.extend({"contact": name, **row} for row in rows)
    write_csv(output / "static_hold_timeseries.csv", static_csv)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_real_launch(
        output, fresh_elapsed, measured_ball, smooth_ball, smooth_velocity,
        real_elapsed, real_racket, real_racket_velocity, executed, apex_elapsed)
    plot_initial_geometry(output, summary)
    plot_command_replay(output, command_replay)
    plot_simulations(
        output, simulation_runs, simulation_results,
        fresh_elapsed, smooth_ball, apex_z)
    plot_static_hold(
        output, static_historical_rows, static_identified_rows)
    write_report(output, summary)
    print(json.dumps({
        "output": str(output),
        "real_apex_z_m": apex_z,
        "real_apex_elapsed_s": apex_elapsed,
        "camera_normal_distance_mm": 1000.0 * camera_normal_distance,
        "default_normal_distance_mm": 1000.0 * default_normal_distance,
        "command_replay": command_replay_summary,
        "simulation_apices_m": {
            item["scenario"]: item["apex_z_m"] for item in simulation_results
        },
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
