#!/usr/bin/env python3
"""Analyze the valid-ball sim-mirror/real juggling phase of one recording.

The recorder's ``policyObs`` is the tick-start state.  This tool therefore
compares real and mirror joint/racket state at that phase, joins strictly by
``sourceSeq`` and policy input, and never compares a tick-end mirror snapshot
to same-tick hardware feedback.  Acceleration is a finite difference over
contiguous mirror intervals; gaps and mirror episode resets are not bridged.

The generated report also diagnoses physical ball/racket contacts from fresh
camera-position updates.  The mirror ball is an input copy, not an independent
contact simulation, so ball-flight conclusions use the real measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = ROOT / (
    "simcode7/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/"
    "measured_qvel_rmp_vertical_v91_gpu0_seed20261018_20260829_stage24_"
    "ceilling_gate_online2/26_rmp91_complete_nonexecution_factorized_commit.pkl"
)
DEFAULT_XML = ROOT / (
    "simcode7/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
)
JOINT_NAMES = [f"RightArm-{index}" for index in range(7)]
AXES = ("x", "y", "z")
COLORS = {
    "real": "#1769aa",
    "sim": "#e76f51",
    "ball": "#16857b",
    "contact": "#6a4c93",
    "hold": "#9e9e9e",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skip {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            if isinstance(value, dict):
                yield value


def vector(value: Any, size: int) -> np.ndarray | None:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)[:size]
    except (TypeError, ValueError):
        return None
    if result.size != size or not np.all(np.isfinite(result)):
        return None
    return result


def nested(row: Any, *keys: str) -> Any:
    current = row
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def row_seq(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("sourceSeq", row.get("tick")))
    except (TypeError, ValueError):
        return None


def sim_seq(row: dict[str, Any]) -> int | None:
    try:
        return int(nested(row, "sync", "sourceSeq"))
    except (TypeError, ValueError):
        return None


def real_q_dq(row: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    contact = row.get("contactIdentification") or {}
    ros = row.get("ros") or {}
    obs = vector(row.get("policyObs"), 14)
    q = first_vector(
        vector(contact.get("rawJointPosition"), 7),
        vector(ros.get("jointPosition"), 7),
        None if obs is None else obs[:7],
    )
    dq = first_vector(
        vector(contact.get("rawJointVelocity"), 7),
        vector(ros.get("jointVelocity"), 7),
        None if obs is None else obs[7:14],
    )
    return q, dq


def first_vector(*values: np.ndarray | None) -> np.ndarray | None:
    return next((value for value in values if value is not None), None)


def policy_command(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("policyCommand")
    if isinstance(value, dict):
        value = value.get("position")
    return vector(value, 7)


def input_matches(real: dict[str, Any], sim: dict[str, Any]) -> bool:
    policy_input = real.get("policyInput")
    sync = sim.get("sync")
    if not isinstance(policy_input, dict) or not isinstance(sync, dict):
        return False
    runtime = real.get("runtime") or {}
    try:
        exact = (
            np.array_equal(
                np.asarray(policy_input.get("ball"), dtype=np.float32),
                np.asarray(sync.get("inputBall"), dtype=np.float32),
            )
            and np.array_equal(
                np.asarray(policy_input.get("velocity"), dtype=np.float32),
                np.asarray(sync.get("inputVelocity"), dtype=np.float32),
            )
            and bool(policy_input.get("valid")) == bool(sync.get("inputValid"))
            and int(policy_input.get("ballObservationSeq", 0) or 0)
            == int(sync.get("ballObservationSeq", 0) or 0)
            and abs(
                float(policy_input.get("age", 0.0))
                - float(sync.get("inputAgeS", 0.0))
            )
            <= 1.0e-9
            and str(runtime.get("activePolicyProfile", runtime.get("policyProfile", "")))
            == str(sync.get("policyProfile", ""))
            and int(runtime.get("policyGeneration", 0) or 0)
            == int(sync.get("policyGeneration", 0) or 0)
        )
        real_ball = real.get("policyBallObservation")
        sim_ball = sim.get("policyBallObservation")
        if real_ball is None and sim_ball is None:
            return bool(exact)
        return bool(
            exact
            and np.array_equal(
                np.asarray(real_ball, dtype=np.float32),
                np.asarray(sim_ball, dtype=np.float32),
            )
        )
    except (TypeError, ValueError):
        return False


def valid_stage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    start = next(
        index
        for index, row in enumerate(rows)
        if row.get("mode") == "policy" and bool(row.get("policyExecuted"))
    )
    stop = len(rows)
    for index in range(start, len(rows) - 4):
        window = rows[index : index + 5]
        if all(not bool(item.get("valid")) for item in window) and all(
            str(item.get("mode", ""))
            in {"policy_ball_state_hold", "policy_feedback_hold", "done_failsafe"}
            for item in window
        ):
            stop = index
            break
    stage = rows[start:stop]
    if not stage:
        raise RuntimeError("valid-ball policy stage is empty")
    return stage


def load_alignment(
    recording: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    all_real = list(read_jsonl(recording / "control_samples.jsonl"))
    stage = valid_stage(all_real)
    real_by_seq = {row_seq(row): row for row in stage if row_seq(row) is not None}
    aligned: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    seen: set[int] = set()
    for sim in read_jsonl(recording / "sim_control_samples.jsonl"):
        seq = sim_seq(sim)
        if seq is None or seq in seen:
            excluded["missing_or_duplicate_sim_source_seq"] += 1
            continue
        seen.add(seq)
        real = real_by_seq.get(seq)
        if real is None:
            excluded["outside_valid_stage"] += 1
            continue
        if sim.get("backend") != "mujoco_recovered_rmp_mirror":
            excluded["wrong_backend"] += 1
            continue
        if not bool(sim.get("available", True)) or not bool(sim.get("valid", False)):
            excluded["invalid_sim_state"] += 1
            continue
        if real.get("mode") != "policy" or not bool(real.get("policyExecuted")):
            excluded["real_not_policy_executed"] += 1
            continue
        if not input_matches(real, sim):
            excluded["input_or_cohort_mismatch"] += 1
            continue
        q_real, dq_real = real_q_dq(real)
        obs_sim = vector(sim.get("policyObs"), 14)
        if q_real is None or dq_real is None or obs_sim is None:
            excluded["missing_tick_start_state"] += 1
            continue
        aligned.append({
            "seq": seq,
            "time": float(real.get("time")),
            "wall_time": float(real.get("wallTimeS")),
            "joint_stamp": float(nested(real, "ros", "jointSampleStampS")),
            "episode": int(nested(sim, "sync", "episode") or 0),
            "real": real,
            "sim": sim,
            "real_q": q_real,
            "real_dq": dq_real,
            "sim_q": obs_sim[:7],
            "sim_dq": obs_sim[7:14],
        })
    aligned.sort(key=lambda item: item["seq"])
    if not aligned:
        raise RuntimeError("no exact sourceSeq/input-matched mirror rows")
    segment = -1
    previous: dict[str, Any] | None = None
    for item in aligned:
        if (
            previous is None
            or item["episode"] != previous["episode"]
            or item["seq"] != previous["seq"] + 1
        ):
            segment += 1
        item["segment"] = segment
        previous = item
    return stage, aligned, excluded


def init_kinematics(checkpoint: Path, xml: Path):
    sys.path.insert(0, str(ROOT / "src/pingpong_moz1"))
    from pingpong_controller.mjx_rl_policy import Gpu0QvelRealRmpV85PolicyController

    return Gpu0QvelRealRmpV85PolicyController(
        str(checkpoint), dt=0.005, robot_xml_path=str(xml)
    )


def pose_twist(controller: Any, q: np.ndarray, dq: np.ndarray) -> dict[str, np.ndarray]:
    position, quaternion, jacobian_p, jacobian_r, ok = (
        controller._compute_racket_pose_jacobians_base(q)
    )
    if not ok:
        raise RuntimeError("racket FK/Jacobian failed")
    return {
        "position": np.asarray(position, dtype=np.float64),
        "quaternion": np.asarray(quaternion, dtype=np.float64),
        "velocity": np.asarray(jacobian_p, dtype=np.float64) @ dq,
        "angular_velocity": np.asarray(jacobian_r, dtype=np.float64) @ dq,
    }


def quaternion_to_euler_deg(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    x, y, z, w = (q[:, index] for index in range(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.rad2deg(np.unwrap(np.stack([roll, pitch, yaw], axis=1), axis=0))


def normal_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.asarray([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])


def populate_kinematics(
    stage: list[dict[str, Any]], aligned: list[dict[str, Any]], controller: Any
) -> dict[str, np.ndarray]:
    full_q: list[np.ndarray] = []
    full_dq: list[np.ndarray] = []
    full_position: list[np.ndarray] = []
    full_quaternion: list[np.ndarray] = []
    full_velocity: list[np.ndarray] = []
    full_omega: list[np.ndarray] = []
    full_joint_stamp: list[float] = []
    kept_rows: list[dict[str, Any]] = []
    for row in stage:
        q, dq = real_q_dq(row)
        if q is None or dq is None:
            continue
        kin = pose_twist(controller, q, dq)
        kept_rows.append(row)
        full_q.append(q)
        full_dq.append(dq)
        full_position.append(kin["position"])
        full_quaternion.append(kin["quaternion"])
        full_velocity.append(kin["velocity"])
        full_omega.append(kin["angular_velocity"])
        try:
            full_joint_stamp.append(float(nested(row, "ros", "jointSampleStampS")))
        except (TypeError, ValueError):
            full_joint_stamp.append(float("nan"))

    start_time = float(kept_rows[0]["time"])
    result = {
        "rows": np.asarray(kept_rows, dtype=object),
        "time": np.asarray([float(row["time"]) - start_time for row in kept_rows]),
        "wall_time": np.asarray([float(row["wallTimeS"]) for row in kept_rows]),
        "joint_stamp": np.asarray(full_joint_stamp),
        "q": np.asarray(full_q),
        "dq": np.asarray(full_dq),
        "racket_position": np.asarray(full_position),
        "racket_quaternion": np.asarray(full_quaternion),
        "racket_velocity": np.asarray(full_velocity),
        "racket_angular_velocity": np.asarray(full_omega),
    }
    result["racket_angle"] = quaternion_to_euler_deg(result["racket_quaternion"])
    result["ddq"] = finite_difference(result["dq"], result["joint_stamp"], 0.030)
    result["racket_acceleration"] = finite_difference(
        result["racket_velocity"], result["joint_stamp"], 0.030
    )
    result["racket_angular_acceleration"] = finite_difference(
        result["racket_angular_velocity"], result["joint_stamp"], 0.030
    )

    for item in aligned:
        item["real_kin"] = pose_twist(controller, item["real_q"], item["real_dq"])
        item["sim_kin"] = pose_twist(controller, item["sim_q"], item["sim_dq"])
    real_euler = quaternion_to_euler_deg(aligned_matrix(aligned, "real_kin.quaternion"))
    sim_euler = quaternion_to_euler_deg(aligned_matrix(aligned, "sim_kin.quaternion"))
    for index, item in enumerate(aligned):
        item["real_euler"] = real_euler[index]
        item["sim_euler"] = sim_euler[index]
    add_aligned_arrays(aligned, start_time)
    return result


def finite_difference(values: np.ndarray, time: np.ndarray, max_dt: float) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=np.float64)
    for index in range(len(values) - 1):
        dt = float(time[index + 1] - time[index])
        if math.isfinite(dt) and 1.0e-4 <= dt <= max_dt:
            result[index] = (values[index + 1] - values[index]) / dt
    return result


def add_aligned_arrays(aligned: list[dict[str, Any]], start_time: float) -> None:
    for item in aligned:
        item["elapsed"] = item["time"] - start_time
        item["real_qcmd"] = policy_command(item["real"])
        item["sim_qcmd"] = vector(item["sim"].get("policyCommand"), 7)
        item["real_action"] = vector(item["real"].get("policyAction"), 7)
        item["sim_action"] = vector(item["sim"].get("policyAction"), 7)
        item["real_ddq"] = np.full(7, np.nan)
        item["sim_ddq"] = np.full(7, np.nan)
        item["real_racket_acceleration"] = np.full(3, np.nan)
        item["sim_racket_acceleration"] = np.full(3, np.nan)
        item["real_racket_angular_acceleration"] = np.full(3, np.nan)
        item["sim_racket_angular_acceleration"] = np.full(3, np.nan)
    for index in range(len(aligned) - 1):
        current, following = aligned[index], aligned[index + 1]
        if current["segment"] != following["segment"]:
            continue
        dt_sim = float(following["time"] - current["time"])
        dt_real = float(following["joint_stamp"] - current["joint_stamp"])
        if not (1.0e-4 <= dt_sim <= 0.030 and 1.0e-4 <= dt_real <= 0.030):
            continue
        current["real_ddq"] = (following["real_dq"] - current["real_dq"]) / dt_real
        current["sim_ddq"] = (following["sim_dq"] - current["sim_dq"]) / dt_sim
        for key, target in (
            ("velocity", "racket_acceleration"),
            ("angular_velocity", "racket_angular_acceleration"),
        ):
            current[f"real_{target}"] = (
                following["real_kin"][key] - current["real_kin"][key]
            ) / dt_real
            current[f"sim_{target}"] = (
                following["sim_kin"][key] - current["sim_kin"][key]
            ) / dt_sim


def aligned_matrix(aligned: list[dict[str, Any]], path: str) -> np.ndarray:
    values: list[np.ndarray] = []
    for item in aligned:
        current: Any = item
        for part in path.split("."):
            current = current[part]
        values.append(np.asarray(current, dtype=np.float64))
    return np.asarray(values)


def aligned_euler(aligned: list[dict[str, Any]], side: str) -> np.ndarray:
    return quaternion_to_euler_deg(aligned_matrix(aligned, f"{side}_kin.quaternion"))


def fresh_camera_rows(stage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    last_stamp: float | None = None
    for row in stage:
        try:
            stamp = float(row.get("cameraStampS"))
        except (TypeError, ValueError):
            continue
        if last_stamp is not None and stamp == last_stamp:
            continue
        if vector(row.get("measuredBall"), 3) is None:
            continue
        output.append(row)
        last_stamp = stamp
    return output


def detect_contacts(stage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fresh = fresh_camera_rows(stage)
    detections: list[dict[str, Any]] = []
    last_time = -math.inf
    for index in range(1, len(fresh)):
        current = fresh[index]
        current_velocity = vector(current.get("measuredVelocity"), 3)
        if current_velocity is None or current_velocity[2] <= 0.10:
            continue
        recent = fresh[max(0, index - 8) : index]
        candidates = [
            (offset, row, vector(row.get("measuredVelocity"), 3))
            for offset, row in enumerate(recent)
        ]
        candidates = [item for item in candidates if item[2] is not None]
        if not candidates:
            continue
        offset, before, before_velocity = min(candidates, key=lambda item: item[2][2])
        if before_velocity[2] >= -0.10:
            continue
        if any(
            (
                vector(row.get("measuredVelocity"), 3) is not None
                and vector(row.get("measuredVelocity"), 3)[2] > 0.10
            )
            for row in recent[offset + 1 :]
        ):
            continue
        detection_time = float(current["time"])
        if detection_time - last_time < 0.15:
            continue
        ball = vector(current.get("measuredBall"), 3)
        racket = vector(current.get("racket"), 3)
        before_ball = vector(before.get("measuredBall"), 3)
        before_racket = vector(before.get("racket"), 3)
        geometry = (
            ball is not None
            and racket is not None
            and np.linalg.norm(ball - racket) <= 0.10
        ) or (
            before_ball is not None
            and before_racket is not None
            and np.linalg.norm(before_ball - before_racket) <= 0.10
        )
        if not geometry:
            continue
        nearby = [
            row
            for row in stage
            if detection_time - 0.09 <= float(row["time"]) <= detection_time
            and vector(row.get("measuredBall"), 3) is not None
            and vector(row.get("racket"), 3) is not None
        ]
        physical = min(
            nearby,
            key=lambda row: np.linalg.norm(
                vector(row.get("measuredBall"), 3) - vector(row.get("racket"), 3)
            ),
        )
        detections.append({
            "detected": current,
            "before": before,
            "physical": physical,
            "detection_time": detection_time,
            "physical_time": float(physical["time"]),
        })
        last_time = detection_time
    return add_contact_dynamics(stage, fresh, detections)


def fit_flight_velocity(
    rows: list[dict[str, Any]], start: float, stop: float, evaluate: float
) -> tuple[np.ndarray, dict[str, Any]] | None:
    selected = [row for row in rows if start <= float(row["time"]) <= stop]
    if len(selected) < 5:
        return None
    time = np.asarray([float(row["time"]) - evaluate for row in selected])
    position = np.asarray([vector(row.get("measuredBall"), 3) for row in selected])
    velocity = np.zeros(3)
    residuals: list[float] = []
    for axis in range(2):
        coefficients = np.polyfit(time, position[:, axis], 1)
        velocity[axis] = float(coefficients[0])
        residuals.append(float(np.sqrt(np.mean(
            (np.polyval(coefficients, time) - position[:, axis]) ** 2
        ))))
    # Constrain the short free-flight fit to gravity.  An unconstrained
    # quadratic is extremely sensitive to the near-racket depth jitter in this
    # recording and can report physically impossible vertical acceleration.
    gravity = 9.8
    gravity_removed_z = position[:, 2] + 0.5 * gravity * time * time
    coefficients_z = np.polyfit(time, gravity_removed_z, 1)
    velocity[2] = float(coefficients_z[0])
    predicted_z = np.polyval(coefficients_z, time) - 0.5 * gravity * time * time
    residuals.append(float(np.sqrt(np.mean((predicted_z - position[:, 2]) ** 2))))
    return velocity, {
        "samples": len(selected),
        "acceleration_z": -gravity,
        "position_rmse_m": residuals,
        "fit_model": "constant_xy_velocity_and_gravity_z",
    }


def add_contact_dynamics(
    stage: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for index, contact in enumerate(contacts):
        detected = float(contact["detection_time"])
        physical_time = float(contact["physical_time"])
        physical = contact["physical"]
        next_time = (
            float(contacts[index + 1]["detection_time"])
            if index + 1 < len(contacts)
            else float(stage[-1]["time"]) + 0.05
        )
        incoming = fit_flight_velocity(
            fresh,
            max(float(stage[0]["time"]), physical_time - 0.14),
            physical_time - 0.035,
            physical_time,
        )
        outgoing = fit_flight_velocity(
            fresh,
            detected + 0.02,
            min(detected + 0.14, next_time - 0.04),
            physical_time,
        )
        contact["incoming_velocity"] = None if incoming is None else incoming[0]
        contact["incoming_fit"] = None if incoming is None else incoming[1]
        contact["outgoing_velocity"] = None if outgoing is None else outgoing[0]
        contact["outgoing_fit"] = None if outgoing is None else outgoing[1]
        ball = vector(physical.get("measuredBall"), 3)
        racket = vector(physical.get("racket"), 3)
        relative = ball - racket
        contact["relative"] = relative
        contact["lateral_offset"] = float(np.linalg.norm(relative[:2]))
        contact["distance"] = float(np.linalg.norm(relative))
        contact["ball_position"] = ball
        contact["racket_position"] = racket
        contact_info = physical.get("contactIdentification") or {}
        contact["racket_velocity"] = vector(contact_info.get("racketLinearVelocity"), 3)
        contact["racket_normal"] = vector(contact_info.get("racketFaceNormal"), 3)
        if contact["racket_velocity"] is None:
            contact["racket_velocity"] = np.zeros(3)
        if contact["racket_normal"] is None:
            quaternion = vector(contact_info.get("racketQuaternionXyzw"), 4)
            contact["racket_normal"] = (
                np.asarray([0.0, 0.0, 1.0])
                if quaternion is None
                else normal_from_quaternion(quaternion)
            )
        flight = [
            row
            for row in fresh
            if detected <= float(row["time"]) < next_time
        ]
        apex = max((float(vector(row.get("measuredBall"), 3)[2]) for row in flight), default=float(ball[2]))
        contact["apex_z"] = apex
        contact["height_gain"] = apex - float(ball[2])
        if contact["incoming_velocity"] is not None and contact["outgoing_velocity"] is not None:
            normal = contact["racket_normal"] / np.linalg.norm(contact["racket_normal"])
            vin = float(np.dot(contact["incoming_velocity"], normal))
            vout = float(np.dot(contact["outgoing_velocity"], normal))
            vracket = float(np.dot(contact["racket_velocity"], normal))
            contact["incoming_normal_velocity"] = vin
            contact["outgoing_normal_velocity"] = vout
            contact["racket_normal_velocity"] = vracket
            denominator = vracket - vin
            contact["effective_restitution"] = (
                (vout - vracket) / denominator if denominator > 1.0e-6 else float("nan")
            )
            contact["expected_outgoing_normal_e066"] = (
                (1.0 + 0.66) * vracket - 0.66 * vin
            )
        else:
            contact["incoming_normal_velocity"] = float("nan")
            contact["outgoing_normal_velocity"] = float("nan")
            contact["racket_normal_velocity"] = float(np.dot(
                contact["racket_velocity"], contact["racket_normal"]
            ))
            contact["effective_restitution"] = float("nan")
            contact["expected_outgoing_normal_e066"] = float("nan")
    return contacts


def metric(real: np.ndarray, sim: np.ndarray) -> dict[str, Any]:
    real = np.asarray(real, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    valid = np.isfinite(real) & np.isfinite(sim)
    difference = np.where(valid, sim - real, np.nan)
    per_dimension = []
    for index in range(difference.shape[1]):
        values = difference[:, index]
        values = values[np.isfinite(values)]
        per_dimension.append({
            "count": int(values.size),
            "mean": None if not values.size else float(np.mean(values)),
            "rms": None if not values.size else float(np.sqrt(np.mean(values * values))),
            "max_abs": None if not values.size else float(np.max(np.abs(values))),
        })
    values = difference[np.isfinite(difference)]
    return {
        "sample_count": int(np.count_nonzero(np.all(valid, axis=1))),
        "overall_rms": None if not values.size else float(np.sqrt(np.mean(values * values))),
        "per_dimension": per_dimension,
    }


def rotation_angle_error_deg(real: np.ndarray, sim: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(real * sim, axis=1))
    return np.rad2deg(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def compute_summary(
    stage: list[dict[str, Any]],
    aligned: list[dict[str, Any]],
    full: dict[str, np.ndarray],
    contacts: list[dict[str, Any]],
    excluded: Counter[str],
) -> dict[str, Any]:
    modes = Counter(str(row.get("mode", "")) for row in stage)
    elapsed = float(stage[-1]["time"] - stage[0]["time"])
    wall_elapsed = float(stage[-1]["wallTimeS"] - stage[0]["wallTimeS"])
    segments = sorted(set(int(item["segment"]) for item in aligned))
    segment_lengths = [sum(int(item["segment"] == segment) for item in aligned) for segment in segments]
    starts = np.asarray([
        index == 0 or aligned[index - 1]["segment"] != item["segment"]
        for index, item in enumerate(aligned)
    ])
    real_quat = aligned_matrix(aligned, "real_kin.quaternion")
    sim_quat = aligned_matrix(aligned, "sim_kin.quaternion")
    groups = {
        "joint_position_rad": metric(
            aligned_matrix(aligned, "real_q"), aligned_matrix(aligned, "sim_q")
        ),
        "joint_velocity_rad_s": metric(
            aligned_matrix(aligned, "real_dq"), aligned_matrix(aligned, "sim_dq")
        ),
        "joint_acceleration_rad_s2": metric(
            aligned_matrix(aligned, "real_ddq"), aligned_matrix(aligned, "sim_ddq")
        ),
        "policy_command_position_rad": metric(
            aligned_matrix(aligned, "real_qcmd"), aligned_matrix(aligned, "sim_qcmd")
        ),
        "policy_action": metric(
            aligned_matrix(aligned, "real_action"), aligned_matrix(aligned, "sim_action")
        ),
        "racket_position_m": metric(
            aligned_matrix(aligned, "real_kin.position"),
            aligned_matrix(aligned, "sim_kin.position"),
        ),
        "racket_velocity_m_s": metric(
            aligned_matrix(aligned, "real_kin.velocity"),
            aligned_matrix(aligned, "sim_kin.velocity"),
        ),
        "racket_acceleration_m_s2": metric(
            aligned_matrix(aligned, "real_racket_acceleration"),
            aligned_matrix(aligned, "sim_racket_acceleration"),
        ),
        "racket_euler_angle_deg": metric(
            aligned_euler(aligned, "real"), aligned_euler(aligned, "sim")
        ),
        "racket_angular_velocity_rad_s": metric(
            aligned_matrix(aligned, "real_kin.angular_velocity"),
            aligned_matrix(aligned, "sim_kin.angular_velocity"),
        ),
        "racket_angular_acceleration_rad_s2": metric(
            aligned_matrix(aligned, "real_racket_angular_acceleration"),
            aligned_matrix(aligned, "sim_racket_angular_acceleration"),
        ),
    }
    rotation_error = rotation_angle_error_deg(real_quat, sim_quat)
    interval_count = sum(max(0, length - 1) for length in segment_lengths)
    contacts_json = []
    for contact in contacts:
        contacts_json.append({
            "physical_time_s": contact["physical_time"],
            "detected_time_s": contact["detection_time"],
            "detected_mode": str(contact["detected"].get("mode", "")),
            "physical_mode": str(contact["physical"].get("mode", "")),
            "relative_ball_minus_racket_m": contact["relative"].tolist(),
            "lateral_offset_m": contact["lateral_offset"],
            "center_distance_m": contact["distance"],
            "racket_velocity_m_s": contact["racket_velocity"].tolist(),
            "racket_normal": contact["racket_normal"].tolist(),
            "racket_normal_velocity_m_s": contact["racket_normal_velocity"],
            "incoming_ball_velocity_m_s": (
                None if contact["incoming_velocity"] is None else contact["incoming_velocity"].tolist()
            ),
            "outgoing_ball_velocity_m_s": (
                None if contact["outgoing_velocity"] is None else contact["outgoing_velocity"].tolist()
            ),
            "incoming_normal_velocity_m_s": contact["incoming_normal_velocity"],
            "outgoing_normal_velocity_m_s": contact["outgoing_normal_velocity"],
            "expected_outgoing_normal_e066_m_s": contact["expected_outgoing_normal_e066"],
            "effective_restitution": contact["effective_restitution"],
            "apex_z_m": contact["apex_z"],
            "height_gain_m": contact["height_gain"],
            "incoming_fit": contact["incoming_fit"],
            "outgoing_fit": contact["outgoing_fit"],
        })
    return {
        "recording": str(Path(stage[0].get("recording", "")).name),
        "stage": {
            "start_time_s": float(stage[0]["time"]),
            "end_time_s": float(stage[-1]["time"]),
            "source_duration_s": elapsed,
            "wall_duration_s": wall_elapsed,
            "control_ticks": len(stage),
            "wall_rate_hz": (len(stage) - 1) / wall_elapsed,
            "mode_counts": dict(modes),
            "feedback_hold_fraction": modes.get("policy_feedback_hold", 0) / len(stage),
        },
        "mirror": {
            "aligned_samples": len(aligned),
            "aligned_fraction_of_stage_ticks": len(aligned) / len(stage),
            "episodes": len(set(int(item["episode"]) for item in aligned)),
            "segments": len(segments),
            "segment_lengths": segment_lengths,
            "contiguous_interval_count": interval_count,
            "contiguous_simulated_duration_s": interval_count * 0.005,
            "longest_segment_duration_s": (max(segment_lengths) - 1) * 0.005,
            "excluded": dict(excluded),
            "episode_start_joint_position_rms_rad": metric(
                aligned_matrix(aligned, "real_q")[starts],
                aligned_matrix(aligned, "sim_q")[starts],
            )["overall_rms"],
            "episode_start_joint_velocity_rms_rad_s": metric(
                aligned_matrix(aligned, "real_dq")[starts],
                aligned_matrix(aligned, "sim_dq")[starts],
            )["overall_rms"],
            "episode_start_sim_joint_velocity_signal_rms_rad_s": float(
                np.sqrt(np.mean(aligned_matrix(aligned, "sim_dq")[starts] ** 2))
            ),
            "episode_start_real_joint_velocity_signal_rms_rad_s": float(
                np.sqrt(np.mean(aligned_matrix(aligned, "real_dq")[starts] ** 2))
            ),
        },
        "groups": groups,
        "racket_orientation_geodesic_error_deg": {
            "rms": float(np.sqrt(np.mean(rotation_error**2))),
            "max": float(np.max(rotation_error)),
        },
        "contacts": contacts_json,
    }


def add_contact_lines(axes: Iterable[plt.Axes], contacts: list[dict[str, Any]], start: float) -> None:
    for axis in axes:
        for contact in contacts:
            axis.axvline(
                contact["physical_time"] - start,
                color=COLORS["contact"],
                linewidth=0.9,
                alpha=0.35,
            )


def plot_sim_segments(
    axis: plt.Axes,
    time: np.ndarray,
    values: np.ndarray,
    segments: np.ndarray,
    dimension: int,
    label: str | None,
) -> None:
    first = True
    for segment in sorted(set(int(value) for value in segments)):
        mask = segments == segment
        valid = mask & np.isfinite(values[:, dimension])
        if not np.any(valid):
            continue
        axis.plot(
            time[valid],
            values[valid, dimension],
            color=COLORS["sim"],
            linestyle="-",
            linewidth=0.65,
            label=label if first else None,
        )
        first = False


def plot_joint_family(
    output: Path,
    title: str,
    unit: str,
    full_time: np.ndarray,
    full_real: np.ndarray,
    aligned: list[dict[str, Any]],
    sim_path: str,
    contacts: list[dict[str, Any]],
    start_time: float,
) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    segments = np.asarray([item["segment"] for item in aligned])
    sim_label = (
        "sim mirror" if np.unique(segments).size == 1 else "sim mirror segments"
    )
    sim = aligned_matrix(aligned, sim_path)
    figure, axes = plt.subplots(7, 1, figsize=(15, 18), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.plot(full_time, full_real[:, index], color=COLORS["real"], linewidth=0.70,
                  label="real hardware" if index == 0 else None)
        plot_sim_segments(
            axis, time, sim, segments, index, sim_label if index == 0 else None
        )
        axis.set_ylabel(f"J{index + 1}\n[{unit}]")
        axis.grid(alpha=0.25)
    add_contact_lines(axes, contacts, start_time)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Control time from valid-stage start [s]")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_racket_family(
    output: Path,
    title: str,
    unit: str,
    full: dict[str, np.ndarray],
    full_key: str,
    aligned: list[dict[str, Any]],
    sim_path: str,
    contacts: list[dict[str, Any]],
    ball: bool = False,
) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    segments = np.asarray([item["segment"] for item in aligned])
    sim_label = (
        "sim mirror" if np.unique(segments).size == 1 else "sim mirror segments"
    )
    sim = aligned_matrix(aligned, sim_path)
    figure, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.plot(full["time"], full[full_key][:, index], color=COLORS["real"], linewidth=0.70,
                  label="real racket" if index == 0 else None)
        plot_sim_segments(
            axis, time, sim, segments, index, sim_label if index == 0 else None
        )
        if ball:
            ball_values = np.asarray([
                vector(row.get("measuredBall"), 3) for row in full["rows"]
            ])
            axis.plot(full["time"], ball_values[:, index], color=COLORS["ball"],
                      linestyle="-", linewidth=0.55,
                      label="measured ball" if index == 0 else None)
        axis.set_ylabel(f"{AXES[index]} [{unit}]")
        axis.grid(alpha=0.25)
    add_contact_lines(axes, contacts, float(full["rows"][0]["time"]))
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Control time from valid-stage start [s]")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_policy_outputs(
    output: Path, aligned: list[dict[str, Any]], contacts: list[dict[str, Any]], start: float
) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    segments = np.asarray([item["segment"] for item in aligned])
    real_command = aligned_matrix(aligned, "real_qcmd")
    sim_command = aligned_matrix(aligned, "sim_qcmd")
    real_action = aligned_matrix(aligned, "real_action")
    sim_action = aligned_matrix(aligned, "sim_action")
    figure, axes = plt.subplots(7, 2, figsize=(17, 19), sharex=True, constrained_layout=True)
    for joint in range(7):
        for column, (real, sim, unit) in enumerate((
            (real_command, sim_command, "qcmd rad"),
            (real_action, sim_action, "action"),
        )):
            axis = axes[joint, column]
            axis.plot(time, real[:, joint], color=COLORS["real"], linewidth=0.70,
                      label="real" if joint == 0 else None)
            plot_sim_segments(axis, time, sim, segments, joint, "sim" if joint == 0 else None)
            axis.set_ylabel(f"J{joint + 1}\n[{unit}]")
            axis.grid(alpha=0.25)
    add_contact_lines(axes.reshape(-1), contacts, start)
    axes[0, 0].set_title("Nominal policy command position")
    axes[0, 1].set_title("Actor action")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Control time from valid-stage start [s]")
    axes[-1, 1].set_xlabel("Control time from valid-stage start [s]")
    figure.suptitle("Policy output after closed-loop state divergence", fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_gap_summary(output: Path, aligned: list[dict[str, Any]], contacts: list[dict[str, Any]], start: float) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    segments = np.asarray([item["segment"] for item in aligned])
    signals = (
        ("joint q RMS [rad]", aligned_matrix(aligned, "real_q"), aligned_matrix(aligned, "sim_q")),
        ("joint dq RMS [rad/s]", aligned_matrix(aligned, "real_dq"), aligned_matrix(aligned, "sim_dq")),
        ("racket position RMS [m]", aligned_matrix(aligned, "real_kin.position"), aligned_matrix(aligned, "sim_kin.position")),
        ("racket velocity RMS [m/s]", aligned_matrix(aligned, "real_kin.velocity"), aligned_matrix(aligned, "sim_kin.velocity")),
    )
    figure, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    for axis, (label, real, sim) in zip(axes, signals):
        values = np.sqrt(np.mean((sim - real) ** 2, axis=1))
        for segment in sorted(set(int(value) for value in segments)):
            mask = segments == segment
            axis.plot(time[mask], values[mask], color="#4e79a7", linewidth=0.70)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    add_contact_lines(axes, contacts, start)
    axes[-1].set_xlabel("Control time from valid-stage start [s]")
    figure.suptitle("Sim2Real gap on exact sourceSeq/input-matched ticks", fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def hold_spans(stage: list[dict[str, Any]], start: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    active: float | None = None
    for row in stage:
        is_hold = row.get("mode") != "policy"
        time = float(row["time"])
        if is_hold and active is None:
            active = time - start
        elif not is_hold and active is not None:
            spans.append((active, time - start))
            active = None
    if active is not None:
        spans.append((active, float(stage[-1]["time"]) - start + 0.005))
    return spans


def plot_contact_dynamics(
    output: Path, full: dict[str, np.ndarray], contacts: list[dict[str, Any]]
) -> None:
    start = float(full["rows"][0]["time"])
    ball_position = np.asarray([vector(row.get("measuredBall"), 3) for row in full["rows"]])
    ball_velocity = np.asarray([vector(row.get("measuredVelocity"), 3) for row in full["rows"]])
    relative = ball_position - full["racket_position"]
    figure, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    axes[0].plot(full["time"], ball_position[:, 2], color=COLORS["ball"], label="ball z")
    axes[0].plot(full["time"], full["racket_position"][:, 2], color=COLORS["real"], label="racket z")
    axes[0].set_ylabel("height [m]")
    axes[1].plot(full["time"], ball_velocity[:, 2], color=COLORS["ball"], label="measured ball vz")
    axes[1].plot(full["time"], full["racket_velocity"][:, 2], color=COLORS["real"], label="racket vz")
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_ylabel("vertical velocity [m/s]")
    axes[2].plot(full["time"], np.linalg.norm(relative[:, :2], axis=1) * 1000.0,
                 color="#f28e2b", label="lateral center offset")
    axes[2].plot(full["time"], relative[:, 2] * 1000.0,
                 color="#59a14f", label="ball z - racket z")
    axes[2].set_ylabel("relative geometry [mm]")
    axes[2].set_xlabel("Control time from valid-stage start [s]")
    for axis in axes:
        for lower, upper in hold_spans(list(full["rows"]), start):
            axis.axvspan(lower, upper, color=COLORS["hold"], alpha=0.10)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    add_contact_lines(axes, contacts, start)
    figure.suptitle(
        "Real ball/racket contact dynamics (gray = policy hold)", fontsize=15, fontweight="bold"
    )
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_aligned_csv(output: Path, aligned: list[dict[str, Any]]) -> None:
    scalar_vectors = (
        ("q_rad", "real_q", "sim_q", JOINT_NAMES),
        ("dq_rad_s", "real_dq", "sim_dq", JOINT_NAMES),
        ("ddq_rad_s2", "real_ddq", "sim_ddq", JOINT_NAMES),
        ("qcmd_rad", "real_qcmd", "sim_qcmd", JOINT_NAMES),
        ("action", "real_action", "sim_action", JOINT_NAMES),
        ("racket_position_m", "real_kin.position", "sim_kin.position", AXES),
        ("racket_velocity_m_s", "real_kin.velocity", "sim_kin.velocity", AXES),
        ("racket_acceleration_m_s2", "real_racket_acceleration", "sim_racket_acceleration", AXES),
        ("racket_angular_velocity_rad_s", "real_kin.angular_velocity", "sim_kin.angular_velocity", AXES),
        ("racket_angular_acceleration_rad_s2", "real_racket_angular_acceleration", "sim_racket_angular_acceleration", AXES),
    )
    header = ["elapsed_s", "source_seq", "episode", "segment"]
    for label, _real, _sim, names in scalar_vectors:
        for side in ("real", "sim"):
            header.extend(f"{side}_{label}_{name}" for name in names)
    real_euler = aligned_euler(aligned, "real")
    sim_euler = aligned_euler(aligned, "sim")
    header.extend(f"real_racket_euler_deg_{axis}" for axis in AXES)
    header.extend(f"sim_racket_euler_deg_{axis}" for axis in AXES)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, item in enumerate(aligned):
            row: list[Any] = [item["elapsed"], item["seq"], item["episode"], item["segment"]]
            for _label, real_path, sim_path, _names in scalar_vectors:
                row.extend(aligned_matrix([item], real_path)[0].tolist())
                row.extend(aligned_matrix([item], sim_path)[0].tolist())
            row.extend(real_euler[index].tolist())
            row.extend(sim_euler[index].tolist())
            writer.writerow(row)


def write_contact_csv(output: Path, summary: dict[str, Any], stage_start: float) -> None:
    fields = [
        "contact", "physical_elapsed_s", "detected_elapsed_s", "detected_mode",
        "physical_mode", "lateral_offset_mm", "center_distance_mm",
        "racket_vz_m_s", "racket_normal_velocity_m_s", "incoming_normal_velocity_m_s",
        "outgoing_normal_velocity_m_s", "expected_outgoing_normal_e066_m_s",
        "effective_restitution", "apex_z_m", "height_gain_mm",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, contact in enumerate(summary["contacts"], start=1):
            writer.writerow({
                "contact": index,
                "physical_elapsed_s": contact["physical_time_s"] - stage_start,
                "detected_elapsed_s": contact["detected_time_s"] - stage_start,
                "detected_mode": contact["detected_mode"],
                "physical_mode": contact["physical_mode"],
                "lateral_offset_mm": 1000.0 * contact["lateral_offset_m"],
                "center_distance_mm": 1000.0 * contact["center_distance_m"],
                "racket_vz_m_s": contact["racket_velocity_m_s"][2],
                "racket_normal_velocity_m_s": contact["racket_normal_velocity_m_s"],
                "incoming_normal_velocity_m_s": contact["incoming_normal_velocity_m_s"],
                "outgoing_normal_velocity_m_s": contact["outgoing_normal_velocity_m_s"],
                "expected_outgoing_normal_e066_m_s": contact["expected_outgoing_normal_e066_m_s"],
                "effective_restitution": contact["effective_restitution"],
                "apex_z_m": contact["apex_z_m"],
                "height_gain_mm": 1000.0 * contact["height_gain_m"],
            })


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}f}"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_report(output: Path, recording: Path, summary: dict[str, Any]) -> None:
    stage = summary["stage"]
    mirror = summary["mirror"]
    groups = summary["groups"]
    joint_q = groups["joint_position_rad"]
    joint_dq = groups["joint_velocity_rad_s"]
    racket_pos = groups["racket_position_m"]
    racket_vel = groups["racket_velocity_m_s"]
    lines = [
        "# V91 有效颠球阶段 Sim2Real 与球拍/球接触分析",
        "",
        f"录制：`{recording.name}`",
        "",
        "## 结论",
        "",
        "这段数据里最明显的问题不是球输入在 real/sim 两边不一致，而是闭环执行状态没有连续对齐：",
        "",
        f"1. 有效阶段共 {stage['control_ticks']} 个控制周期，其中 `policy_feedback_hold` "
        f"有 {stage['mode_counts'].get('policy_feedback_hold', 0)} 次（{100.0 * stage['feedback_hold_fraction']:.1f}%）。"
        f"墙钟频率只有 {stage['wall_rate_hz']:.1f} Hz，而名义频率是 200 Hz。",
        f"2. sim mirror 只有 {mirror['aligned_samples']} 个严格对齐样本，被切成 {mirror['segments']} 个连续短片段；"
        f"合计真正连续推进 {mirror['contiguous_simulated_duration_s']:.3f} s，最长单段只有 "
        f"{mirror['longest_segment_duration_s']:.3f} s。",
        f"3. 每个 mirror 片段开头，关节位置 RMS 只差 {mirror['episode_start_joint_position_rms_rad']:.6f} rad，"
        f"但关节速度 RMS 已差 {mirror['episode_start_joint_velocity_rms_rad_s']:.3f} rad/s。"
        f"实物起始速度信号 RMS 是 {mirror['episode_start_real_joint_velocity_signal_rms_rad_s']:.3f} rad/s，"
        f"mirror 只有 {mirror['episode_start_sim_joint_velocity_signal_rms_rad_s']:.3f} rad/s。"
        "这说明 mirror 重启时主要对齐了 q，却没有把实物 dq 和执行器/RMP 历史状态一起恢复。",
        f"4. 全部严格对齐点上，关节 q gap RMS 为 {joint_q['overall_rms']:.4f} rad，"
        f"dq gap RMS 为 {joint_dq['overall_rms']:.3f} rad/s；球拍中心位置 gap RMS 为 "
        f"{1000.0 * racket_pos['overall_rms']:.1f} mm，线速度 gap RMS 为 {racket_vel['overall_rms']:.3f} m/s。",
        "5. 离线从新鲜相机位置反转识别到 3 次实物触球。第一次和第三次的检测周期恰好处于 "
        "`policy_feedback_hold`；第三次触球时球拍法向速度很小且横向偏心最大，因而最后一跳只获得很低的高度。",
        "",
        "因此当前日志首先证明的是：频繁 hold + mirror 速度/历史状态重置破坏了时序与闭环一致性。"
        "在得到连续 mirror 记录前，不能把现有曲线简单解释成一组可直接拟合的纯机械参数误差。",
        "",
        "## 数据范围与对齐语义",
        "",
        f"- 有效球阶段：控制时间 `{stage['start_time_s']:.3f}–{stage['end_time_s']:.3f} s`；"
        f"源时钟跨度 {stage['source_duration_s']:.3f} s，实际墙钟跨度 {stage['wall_duration_s']:.3f} s。",
        f"- 模式计数：`{json.dumps(stage['mode_counts'], ensure_ascii=False)}`。",
        "- real/sim 同时要求 `sourceSeq`、球位置/速度/有效性/age/观测序号、policy profile/generation 完全一致。",
        "- q、dq、球拍位置/速度/姿态均比较同一 tick 开始状态。real dq 使用原始关节反馈，"
        "sim dq 使用 mirror `policyObs[7:14]`。",
        "- ddq、球拍线加速度和角加速度只在同一 mirror 片段内作有限差分；不跨 episode 或序号缺口。",
        "- 图上的橙色 sim 曲线是短片段，不应把两个分离片段之间视觉上连成连续仿真。",
        "",
        "## 图表",
        "",
        "- `01_joint_position.png`：7 个关节角。",
        "- `02_joint_velocity.png`：7 个关节原始角速度。",
        "- `03_joint_acceleration.png`：7 个关节有限差分角加速度。",
        "- `04_policy_command_and_action.png`：同 tick 策略关节目标与动作。",
        "- `05_racket_position.png`：球拍中心位置，并叠加实测球位置。",
        "- `06_racket_velocity.png` / `07_racket_acceleration.png`：球拍中心线速度/线加速度。",
        "- `08_racket_orientation.png`：球拍 XYZ Euler 姿态。",
        "- `09_racket_angular_velocity.png` / `10_racket_angular_acceleration.png`：角速度/角加速度。",
        "- `11_gap_summary.png`：严格对齐点上的 q、dq、球拍位置和线速度 RMS gap。",
        "- `12_ball_contact_dynamics.png`：实物球/球拍高度、竖直速度和接触偏心；灰色为非 policy 周期。",
        "",
        "## 各参数误差摘要",
        "",
        "| 参数 | 总体 RMS gap | 有效差分样本 |",
        "| --- | ---: | ---: |",
    ]
    group_labels = (
        ("关节角", "joint_position_rad", "rad"),
        ("关节角速度", "joint_velocity_rad_s", "rad/s"),
        ("关节角加速度", "joint_acceleration_rad_s2", "rad/s²"),
        ("策略关节目标", "policy_command_position_rad", "rad"),
        ("策略动作", "policy_action", "normalized"),
        ("球拍中心位置", "racket_position_m", "m"),
        ("球拍中心线速度", "racket_velocity_m_s", "m/s"),
        ("球拍中心线加速度", "racket_acceleration_m_s2", "m/s²"),
        ("球拍 Euler 姿态", "racket_euler_angle_deg", "deg"),
        ("球拍角速度", "racket_angular_velocity_rad_s", "rad/s"),
        ("球拍角加速度", "racket_angular_acceleration_rad_s2", "rad/s²"),
    )
    for label, key, unit in group_labels:
        item = groups[key]
        lines.append(
            f"| {label} | {fmt(item['overall_rms'], 5)} {unit} | {item['sample_count']} |"
        )
    orientation = summary["racket_orientation_geodesic_error_deg"]
    lines.extend([
        f"| 球拍三维姿态旋转角 | {orientation['rms']:.4f} deg | {mirror['aligned_samples']} |",
        "",
        "逐关节/逐轴 RMS、mean、max 见 `summary.json`；逐 tick 数值见 `aligned_sim2real.csv`。",
        "",
        "## 实物触球诊断",
        "",
        "触球时刻由新鲜相机位置的下落→上升反转并结合球拍距离离线识别。"
        "记录内在线 `inferredContact` 只标出后两次；下表把较弱的第一次反弹也纳入。",
        "",
        "| 次数 | 物理时刻 | 检测周期模式 | 横向偏心 | 球拍法向速度 | 入射/出射法向速度 | 后续增高 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for index, contact in enumerate(summary["contacts"], start=1):
        lines.append(
            f"| {index} | {contact['physical_time_s']:.3f} s | `{contact['detected_mode']}` | "
            f"{1000.0 * contact['lateral_offset_m']:.1f} mm | "
            f"{fmt(contact['racket_normal_velocity_m_s'])} m/s | "
            f"{fmt(contact['incoming_normal_velocity_m_s'])} / {fmt(contact['outgoing_normal_velocity_m_s'])} m/s | "
            f"{1000.0 * contact['height_gain_m']:.1f} mm |"
        )
    lines.extend([
        "",
        "这里的入/出射速度来自接触前后球位置的局部飞行拟合；相机速度是滑窗量，不能直接当瞬时碰撞速度。"
        "最后一段在 18.645 s 就失去球状态，因此第三次后续增高是可观测下界。",
        "",
        "## 为什么球颠不高",
        "",
        "### 1. 接触附近的 policy 被反馈 hold 打断",
        "",
        f"有效阶段 {100.0 * stage['feedback_hold_fraction']:.1f}% 的周期没有运行策略。"
        "第一次、第三次反弹的检测周期都是 `policy_feedback_hold`。第三次附近球拍 z 速度几乎为零，"
        "没有按来球速度补足上行动量；这与最后一跳很低直接一致。",
        "",
        "### 2. 第三次触球明显偏心，能量被送到横向",
        "",
        "球从初始位置持续向 +y 漂移。第三次触球横向中心偏差显著增大，"
        "同时实测球 y 速度继续增大；球拍即使发生碰撞，也没有把主要冲量稳定地送到竖直方向。",
        "",
        "### 3. mirror 没有保持真实执行器状态，不能提前预测接触相位",
        "",
        "mirror 每个短片段开头的位置看似对齐，但速度和 RMP/延迟缓存没有对齐。"
        "这会让 sim 球拍相位、速度和策略后续动作一开始就不同。当前 mirror 也只复制球观测输入，"
        "并不在这份日志中形成可与实物比较的独立球碰撞轨迹。",
        "",
        "### 4. 没有竖直能量闭环兜底",
        "",
        "本次 `contactActionShield.enabled=false`，`verticalEnergyEnabled=false`。"
        "策略一旦因 hold、相位差或偏心击球造成出球速度不足，没有根据目标 apex 对接触前球拍法向速度做补偿。",
        "",
        "## 建议的排查/修正顺序",
        "",
        "1. 先让实物控制流稳定：消除 `source_sequence_gap` 导致的频繁 hold，确认墙钟接近 200 Hz，"
        "并保证接触前后至少数百毫秒不切 mode。",
        "2. sim mirror 重同步时同时注入 q、dq、当前/延迟目标、RMP 输出延迟队列；"
        "否则应把该片段标成 warm-up，不计入 sim2real 指标。",
        "3. 录一段至少 3–5 次连续触球、无 mirror reset 的数据，再做 command→q/dq 的延迟、增益、"
        "阻尼和摩擦辨识；当前碎片不足以唯一分离这些机械参数。",
        "4. 接触控制加入 apex/竖直能量闭环：根据入射法向速度与目标高度计算接触前球拍法向速度，"
        "同时限制横向速度和偏心。先 shadow 记录，再逐步启用。",
        "5. 把离线识别到的第一次弱反弹加入在线 contact 标记测试，避免控制器少算一次触球。",
        "",
        "## 限制",
        "",
        "- sim mirror 仅有短片段，结论能确认状态重置/时序问题，但不能从本次数据唯一反推完整实物动力学参数。",
        "- 球碰撞只存在于实物测量侧；sim 侧球位置/速度是同一策略输入副本，不能用来比较恢复系数。",
        "- 第三次触球后的球检测很快失效，其实际 apex 可能略高于记录内最大值。",
    ])
    (output / "SIM2REAL_VALID_JUGGLE_ANALYSIS_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording = args.recording.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    xml = args.xml.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else recording / "analysis" / "v91_sim2real_valid_juggle"
    )
    for path in (recording / "control_samples.jsonl", recording / "sim_control_samples.jsonl", checkpoint, xml):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    stage, aligned, excluded = load_alignment(recording)
    controller = init_kinematics(checkpoint, xml)
    full = populate_kinematics(stage, aligned, controller)
    contacts = detect_contacts(stage)
    summary = compute_summary(stage, aligned, full, contacts, excluded)
    summary["recording"] = recording.name
    summary["contracts"] = {
        "checkpoint": str(checkpoint),
        "xml": str(xml),
        "state_phase": "policyObs tick-start",
        "real_joint_velocity": "contactIdentification.rawJointVelocity, fallback ros.jointVelocity",
        "sim_joint_velocity": "sim.policyObs[7:14]",
        "acceleration": "finite difference within contiguous sim segment only",
        "mirror_ball": "policy input copy; not independent collision dynamics",
    }
    summary = json_safe(summary)

    start = float(stage[0]["time"])
    plot_joint_family(output / "01_joint_position.png", "Joint position: real vs sim mirror", "rad",
                      full["time"], full["q"], aligned, "sim_q", contacts, start)
    plot_joint_family(output / "02_joint_velocity.png", "Joint velocity: real raw feedback vs sim mirror", "rad/s",
                      full["time"], full["dq"], aligned, "sim_dq", contacts, start)
    plot_joint_family(output / "03_joint_acceleration.png", "Joint acceleration: contiguous finite difference", "rad/s²",
                      full["time"], full["ddq"], aligned, "sim_ddq", contacts, start)
    plot_policy_outputs(output / "04_policy_command_and_action.png", aligned, contacts, start)
    plot_racket_family(output / "05_racket_position.png", "Racket-center position: real vs sim mirror", "m",
                       full, "racket_position", aligned, "sim_kin.position", contacts, ball=True)
    plot_racket_family(output / "06_racket_velocity.png", "Racket-center linear velocity", "m/s",
                       full, "racket_velocity", aligned, "sim_kin.velocity", contacts)
    plot_racket_family(output / "07_racket_acceleration.png", "Racket-center linear acceleration", "m/s²",
                       full, "racket_acceleration", aligned, "sim_racket_acceleration", contacts)
    plot_racket_family(output / "08_racket_orientation.png", "Racket XYZ Euler orientation", "deg",
                       full, "racket_angle", aligned, "sim_euler", contacts)
    plot_racket_family(output / "09_racket_angular_velocity.png", "Racket angular velocity in base frame", "rad/s",
                       full, "racket_angular_velocity", aligned, "sim_kin.angular_velocity", contacts)
    plot_racket_family(output / "10_racket_angular_acceleration.png", "Racket angular acceleration in base frame", "rad/s²",
                       full, "racket_angular_acceleration", aligned, "sim_racket_angular_acceleration", contacts)
    plot_gap_summary(output / "11_gap_summary.png", aligned, contacts, start)
    plot_contact_dynamics(output / "12_ball_contact_dynamics.png", full, contacts)

    write_aligned_csv(output / "aligned_sim2real.csv", aligned)
    write_contact_csv(output / "contact_diagnostics.csv", summary, start)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(output, recording, summary)
    print(json.dumps({
        "output": str(output),
        "stage": summary["stage"],
        "mirror": summary["mirror"],
        "contacts": summary["contacts"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
