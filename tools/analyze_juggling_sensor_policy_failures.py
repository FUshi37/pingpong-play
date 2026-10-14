#!/usr/bin/env python3
"""Analyze camera/mocap juggling observations and full 67D->7D failure chains.

This analysis is intentionally specific to the nine 2026-08-05/09 probe runs.
It keeps the causal cut points explicit: the normal interval ends at the last
plausible contact (or contact-time reacquisition) that starts the terminal
flight/chase.  Data after that cut is used for failure attribution, but is not
allowed to contaminate the normal observation/noise comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import types
from collections import Counter, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


RUN_SPECS = (
    # run, sensor, first observation sequence belonging to terminal event
    ("probe_recording_20260805_155031", "camera", 126),
    ("probe_recording_20260805_155253", "camera", 228),
    ("probe_recording_20260809_121703", "camera", 178),
    ("probe_recording_20260809_121734", "camera", 81),
    ("probe_recording_20260809_121831", "camera", 259),
    ("probe_recording_20260809_154527", "mocap", 100),
    ("probe_recording_20260809_154656", "mocap", 51),
    ("probe_recording_20260809_161246", "mocap", 212),
    ("probe_recording_20260809_161432", "mocap", 182),
)

ACTION_NAMES = [f"RightArm-{i}" for i in range(7)]


@dataclass
class RunData:
    name: str
    sensor: str
    causal_seq: int
    metadata: dict[str, Any]
    policy: list[dict[str, Any]]
    done: dict[str, Any]
    sim: dict[int, dict[str, Any]]
    normal: list[dict[str, Any]]
    failure: list[dict[str, Any]]
    terminal: list[dict[str, Any]]
    onset: dict[str, Any]
    contacts: list[dict[str, Any]]


def finite_array(values: Iterable[Any]) -> np.ndarray:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return np.asarray(out, dtype=np.float64)


def vector(value: Any, size: int, fill: float = math.nan) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        arr = np.zeros(0, dtype=np.float64)
    out = np.full(size, fill, dtype=np.float64)
    count = min(size, arr.size)
    if count:
        out[:count] = arr[:count]
    return out


def nested(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def compact_control(row: dict[str, Any]) -> dict[str, Any]:
    policy_command = row.get("policyCommand") or {}
    command = row.get("command") or {}
    ros = row.get("ros") or {}
    workspace = row.get("workspaceSafety") or {}
    return {
        "sourceSeq": int(row.get("sourceSeq", 0) or 0),
        "sourceTimeS": float(row.get("sourceTimeS", 0.0) or 0.0),
        "wallTimeS": float(row.get("wallTimeS", 0.0) or 0.0),
        "mode": str(row.get("mode", "")),
        "reason": str(row.get("reason", "")),
        "conditions": dict(row.get("conditions") or {}),
        "policyExecuted": bool(row.get("policyExecuted", False)),
        "rawValid": bool(row.get("rawValid", False)),
        "valid": bool(row.get("valid", False)),
        "age": float(row.get("age", math.nan)),
        "ballObservationSeq": int(row.get("ballObservationSeq", 0) or 0),
        "measuredBall": vector(row.get("measuredBall"), 3),
        "measuredVelocity": vector(row.get("measuredVelocity"), 3),
        "observedBall": vector(row.get("observedBall"), 3),
        "observedVelocity": vector(row.get("observedVelocity"), 3),
        "policyObs": vector(row.get("policyObs"), 67),
        "policyAction": vector(row.get("policyAction"), 7),
        "policyCommandPosition": vector(
            policy_command.get("position") if isinstance(policy_command, dict)
            else policy_command,
            7,
        ),
        "policyCommandVelocity": vector(
            policy_command.get("velocity") if isinstance(policy_command, dict)
            else None,
            7,
        ),
        "policyCommandAcceleration": vector(
            policy_command.get("acceleration") if isinstance(policy_command, dict)
            else None,
            7,
        ),
        "publishedPosition": vector(
            command.get("publishedSafePosition", command.get("position")), 7),
        "nominalPosition": vector(
            command.get("policyNominalPosition", policy_command.get("position")), 7),
        "actualQ": vector(ros.get("jointPosition"), 7),
        "actualDq": vector(ros.get("jointVelocity"), 7),
        "actualDdq": vector(ros.get("jointAcceleration"), 7),
        "policyRacket": vector(row.get("racket"), 3),
        "actualEe": vector(ros.get("jointEe"), 3),
        "workspaceEe": vector(workspace.get("eePosition"), 3),
        "workspacePoseSource": str(workspace.get("poseSource", "")),
        "workspaceJointAgeS": workspace.get("jointAgeS"),
        "cameraMeasurement": row.get("cameraMeasurement"),
    }


def compact_sim(row: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    sync = row.get("sync") or {}
    source_seq = int(sync.get("sourceSeq", 0) or 0)
    if source_seq <= 0:
        return None
    command = row.get("policyCommand")
    return source_seq, {
        "obs": vector(row.get("policyObs"), 67),
        "action": vector(row.get("policyAction"), 7),
        "q": vector(row.get("jointPosition"), 7),
        "dq": vector(row.get("jointVelocity"), 7),
        "command": vector(command, 7),
        "racket": vector(row.get("racket"), 3),
        "racketQuatXyzw": vector(row.get("racketQuatXyzw"), 4),
        "ball": vector(row.get("ball"), 3),
    }


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def unique_refreshes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    last_seq: int | None = None
    for row in rows:
        seq = row["ballObservationSeq"]
        if seq > 0 and seq != last_seq:
            out.append(row)
            last_seq = seq
    return out


def detect_contacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect measured downward-to-upward reversals with racket geometry checks.

    The deployed camera velocity is a trailing five-position secant.  Around a
    real impact it can therefore move from a clearly downward value to -0.096
    m/s and then to a positive value.  Requiring the immediately preceding
    sample to be below -0.10 m/s misses that impact.  Search the last five
    refreshes for the downward leg, but still require the first subsequent
    positive crossing and racket proximity so noise spikes are not counted as
    contacts.
    """
    refresh = unique_refreshes(rows)
    contacts: list[dict[str, Any]] = []
    last_time = -1e30
    for idx in range(1, len(refresh)):
        after = refresh[idx]
        recent = refresh[max(0, idx - 5):idx]
        before = min(recent, key=lambda row: row["measuredVelocity"][2])
        vb = before["measuredVelocity"]
        va = after["measuredVelocity"]
        if not (np.all(np.isfinite(vb)) and np.all(np.isfinite(va))):
            continue
        if not (vb[2] < -0.10 and va[2] > 0.10):
            continue
        before_idx = refresh.index(before, max(0, idx - 5), idx)
        if any(
            row["measuredVelocity"][2] > 0.10
            for row in refresh[before_idx + 1:idx]
        ):
            continue
        event_t = after["wallTimeS"]
        if event_t - last_time < 0.18:
            continue
        ball = after["measuredBall"]
        racket = after["policyRacket"]
        rel = ball - racket
        geometry_ok = bool(
            np.linalg.norm(rel[:2]) <= 0.085 and -0.025 <= rel[2] <= 0.105)
        # Mocap can hold the last pre-impact state and reacquire only after the
        # ball has already separated. Accept that pattern if the held state was
        # close to the racket and the refresh gap spans the contact.
        prev_rel = before["measuredBall"] - before["policyRacket"]
        held_contact = bool(
            after["wallTimeS"] - before["wallTimeS"] <= 0.09
            and np.linalg.norm(prev_rel) <= 0.10
        )
        if not (geometry_ok or held_contact):
            continue
        last_time = event_t
        contacts.append({
            "seq": after["ballObservationSeq"],
            "wallTimeS": event_t,
            "position": ball.copy(),
            "racket": racket.copy(),
            "relative": rel.copy(),
            "velocityBefore": vb.copy(),
            "velocityAfter": va.copy(),
            "heldContact": held_contact and not geometry_ok,
        })
    return contacts


def load_run(root: Path, name: str, sensor: str, causal_seq: int) -> RunData:
    run_dir = root / name
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    policy: list[dict[str, Any]] = []
    done: dict[str, Any] | None = None
    for raw in load_jsonl(run_dir / "control_samples.jsonl"):
        if raw.get("policyExecuted"):
            policy.append(compact_control(raw))
        elif done is None and raw.get("mode") == "done_failsafe":
            done = compact_control(raw)
    if not policy or done is None:
        raise RuntimeError(f"{name}: missing policy or done samples")

    onset = next(
        (row for row in policy if row["ballObservationSeq"] >= causal_seq),
        policy[-1],
    )
    onset_t = onset["wallTimeS"]
    normal = [row for row in policy if row["wallTimeS"] < onset_t]
    failure = [row for row in policy if row["wallTimeS"] >= onset_t]
    terminal_start = max(policy[0]["wallTimeS"], policy[-1]["wallTimeS"] - 0.30)
    terminal = [row for row in policy if row["wallTimeS"] >= terminal_start]

    sim: dict[int, dict[str, Any]] = {}
    sim_path = run_dir / "sim_control_samples.jsonl"
    if sim_path.is_file():
        for raw in load_jsonl(sim_path):
            item = compact_sim(raw)
            if item is not None:
                sim[item[0]] = item[1]

    return RunData(
        name=name,
        sensor=sensor,
        causal_seq=causal_seq,
        metadata=metadata,
        policy=policy,
        done=done,
        sim=sim,
        normal=normal,
        failure=failure,
        terminal=terminal,
        onset=onset,
        contacts=detect_contacts(policy),
    )


def fit_post_contact_flight(
    run: RunData,
    contact_index: int,
) -> dict[str, Any] | None:
    """Fit the clean flight after a detected impact and extrapolate to it.

    x/y use constant velocity; z uses a quadratic.  The first 55 ms after the
    detector's reversal are excluded because the deployed five-position
    velocity window still straddles the impact.  The last 75 ms before the
    next impact are excluded for the same reason.
    """
    contact = run.contacts[contact_index]
    refresh = unique_refreshes(run.policy)
    start = contact["wallTimeS"] + 0.055
    if contact_index + 1 < len(run.contacts):
        end = run.contacts[contact_index + 1]["wallTimeS"] - 0.075
    else:
        end = min(
            run.policy[-1]["wallTimeS"], contact["wallTimeS"] + 0.28)
    fit_rows = [
        row for row in refresh
        if start <= row["wallTimeS"] <= end and row["rawValid"]
    ]
    if len(fit_rows) < 8:
        return None
    tt = np.asarray([
        row["wallTimeS"] - contact["wallTimeS"] for row in fit_rows
    ])
    position = rows_matrix(fit_rows, "measuredBall", 3)
    velocity: list[float] = []
    event_position: list[float] = []
    acceleration_z = math.nan
    for dim in range(3):
        degree = 1 if dim < 2 else 2
        coeff = np.polyfit(tt, position[:, dim], degree)
        velocity.append(float(np.polyval(np.polyder(coeff), 0.0)))
        event_position.append(float(np.polyval(coeff, 0.0)))
        if dim == 2:
            acceleration_z = float(2.0 * coeff[0])
    return {
        "n": len(fit_rows),
        "velocity": np.asarray(velocity, dtype=np.float64),
        "eventPosition": np.asarray(event_position, dtype=np.float64),
        "accelerationZ": acceleration_z,
        "ballistic": bool(-13.0 <= acceleration_z <= -7.0),
    }


def _nearest_policy_row(
    rows: list[dict[str, Any]], target_time_s: float,
) -> dict[str, Any]:
    return min(rows, key=lambda row: abs(row["wallTimeS"] - target_time_s))


def contact_and_cycle_dynamics(
    runs: list[RunData],
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build impact-phase and inter-impact loop diagnostics.

    The optional MuJoCo FK path evaluates the *actual feedback q* and command q
    with exactly the deployment model.  Failure to initialize FK leaves those
    columns NaN but does not prevent the sensor/trajectory analysis.
    """
    kinematics = None
    normal_from_quaternion = None
    try:
        from pingpong_controller.mjx_rl_policy import (
            Simcode5PureActuator67PolicyController,
        )
        from pingpong_controller.racket_juggle_mpc import (
            racket_normal_from_quaternion_xyzw,
        )
        xml_path = checkpoint.parents[3] / "tools" / "rl_sim" / "moz1_pd.xml"
        kinematics = Simcode5PureActuator67PolicyController(
            str(checkpoint), robot_xml_path=str(xml_path))
        normal_from_quaternion = racket_normal_from_quaternion_xyzw
    except Exception as exc:  # pragma: no cover - optional analysis backend.
        print(f"Warning: contact FK diagnostics disabled: {exc}", file=sys.stderr)

    calibration_rotations: dict[str, np.ndarray] = {}
    contact_rows: list[dict[str, Any]] = []
    cycle_rows: list[dict[str, Any]] = []
    for run in runs:
        camera_rotation = None
        if run.sensor == "camera":
            calibration_path = nested(
                run.metadata, "config", "cameraBall", "calibrationJsonPath")
            if calibration_path:
                key = str(calibration_path)
                if key not in calibration_rotations:
                    try:
                        payload = json.loads(
                            Path(key).read_text(encoding="utf-8"))
                        calibration_rotations[key] = np.asarray(
                            payload["T_base_camera"], dtype=np.float64)[:3, :3]
                    except Exception:
                        calibration_rotations[key] = np.full((3, 3), math.nan)
                camera_rotation = calibration_rotations[key]

        dynamics_by_index: dict[int, dict[str, Any] | None] = {}
        for index, contact in enumerate(run.contacts):
            dynamics = fit_post_contact_flight(run, index)
            dynamics_by_index[index] = dynamics
            event_candidates = [
                row for row in run.policy
                if row["ballObservationSeq"] == contact["seq"]
            ]
            event_row = event_candidates[0] if event_candidates else _nearest_policy_row(
                run.policy, contact["wallTimeS"])
            near_contact = [
                row for row in run.policy
                if contact["wallTimeS"] - 0.085 <= row["wallTimeS"]
                <= contact["wallTimeS"] + 0.005
            ]
            physical_row = min(
                near_contact,
                key=lambda row: np.linalg.norm(
                    row["measuredBall"] - row["policyRacket"]),
            )

            actual_normal = np.full(3, math.nan)
            command_normal = np.full(3, math.nan)
            jacobian_action = np.full(3, math.nan)
            if kinematics is not None and normal_from_quaternion is not None:
                try:
                    (_pos, quat, jacobian, _jr, ok) = (
                        kinematics._compute_racket_pose_jacobians_base(
                            event_row["actualQ"]))
                    if ok:
                        actual_normal = normal_from_quaternion(quat)
                        jacobian_action = jacobian @ event_row["policyAction"]
                    _pos, quat, ok = kinematics._compute_racket_pose_quat_base(
                        event_row["publishedPosition"])
                    if ok:
                        command_normal = normal_from_quaternion(quat)
                except Exception:
                    pass

            sim_normal = np.full(3, math.nan)
            sim = run.sim.get(event_row["sourceSeq"])
            if (sim is not None and normal_from_quaternion is not None
                    and np.all(np.isfinite(sim["racketQuatXyzw"]))):
                sim_normal = normal_from_quaternion(sim["racketQuatXyzw"])

            measurement = event_row.get("cameraMeasurement")
            used_minus_sensor_depth_mm = math.nan
            depth_bias_base_mm = np.full(3, math.nan)
            if isinstance(measurement, dict):
                used = measurement.get("depth_mm")
                sensor = measurement.get("sensor_depth_mm")
                try:
                    used_minus_sensor_depth_mm = float(used) - float(sensor)
                except (TypeError, ValueError):
                    used_minus_sensor_depth_mm = math.nan
                if (camera_rotation is not None
                        and math.isfinite(used_minus_sensor_depth_mm)):
                    depth_bias_base_mm = (
                        camera_rotation[:, 2] * used_minus_sensor_depth_mm)

            fit_velocity = (
                dynamics["velocity"] if dynamics is not None
                else np.full(3, math.nan))
            fit_position = (
                dynamics["eventPosition"] if dynamics is not None
                else np.full(3, math.nan))
            contact_rows.append({
                "run": run.name,
                "sensor": run.sensor,
                "contact_index": index + 1,
                "ball_observation_seq": contact["seq"],
                "time_from_policy_start_s": (
                    contact["wallTimeS"] - run.policy[0]["wallTimeS"]),
                "detector_delay_from_closest_approach_ms": 1000.0 * (
                    contact["wallTimeS"] - physical_row["wallTimeS"]),
                "closest_rel_x_mm": 1000.0 * (
                    physical_row["measuredBall"][0]
                    - physical_row["policyRacket"][0]),
                "closest_rel_y_mm": 1000.0 * (
                    physical_row["measuredBall"][1]
                    - physical_row["policyRacket"][1]),
                "closest_rel_z_mm": 1000.0 * (
                    physical_row["measuredBall"][2]
                    - physical_row["policyRacket"][2]),
                **{
                    f"event_ball_v{axis}_m_s": event_row["measuredVelocity"][dim]
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"postflight_fit_v{axis}_m_s": fit_velocity[dim]
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"event_velocity_error_v{axis}_m_s": (
                        event_row["measuredVelocity"][dim] - fit_velocity[dim])
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"event_position_minus_postfit_{axis}_mm": 1000.0 * (
                        event_row["measuredBall"][dim] - fit_position[dim])
                    for dim, axis in enumerate("xyz")
                },
                "postflight_fit_acceleration_z_m_s2": (
                    dynamics["accelerationZ"] if dynamics is not None
                    else math.nan),
                "postflight_fit_ballistic": bool(
                    dynamics is not None and dynamics["ballistic"]),
                "used_minus_sensor_depth_mm": used_minus_sensor_depth_mm,
                **{
                    f"depth_temporal_bias_base_{axis}_mm": depth_bias_base_mm[dim]
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"event_action_{dim}": event_row["policyAction"][dim]
                    for dim in range(7)
                },
                **{
                    f"event_actual_minus_command_q{dim}_deg": math.degrees(
                        event_row["actualQ"][dim]
                        - event_row["publishedPosition"][dim])
                    for dim in range(7)
                },
                **{
                    f"event_jacobian_action_{axis}": jacobian_action[dim]
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"event_racket_v{axis}_m_s": event_row["policyObs"][29 + dim]
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"actual_normal_{axis}": actual_normal[dim]
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"command_normal_{axis}": command_normal[dim]
                    for dim, axis in enumerate("xyz")
                },
                **{
                    f"sim_normal_{axis}": sim_normal[dim]
                    for dim, axis in enumerate("xyz")
                },
            })

        for index, (start_contact, end_contact) in enumerate(
                zip(run.contacts[:-1], run.contacts[1:])):
            interval = [
                row for row in run.policy
                if start_contact["wallTimeS"] <= row["wallTimeS"]
                <= end_contact["wallTimeS"]
            ]
            time = np.asarray([row["wallTimeS"] for row in interval])
            real_racket = rows_matrix(interval, "policyRacket", 3)

            def loop_metrics(position: np.ndarray) -> dict[str, float]:
                start_position = position[0]
                path = float(np.sum(np.linalg.norm(
                    np.diff(position[:, [0, 2]], axis=0), axis=1)))
                chord = float(np.linalg.norm(
                    position[-1, [0, 2]] - start_position[[0, 2]]))
                out = {
                    "xz_path_length_m": path,
                    "xz_chord_m": chord,
                    "xz_path_chord_ratio": path / max(chord, 0.001),
                    "x_retreat_m": float(
                        start_position[0] - np.min(position[:, 0])),
                    "x_range_m": float(np.ptp(position[:, 0])),
                }
                for delay_ms in (100, 200):
                    sample = int(np.argmin(np.abs(
                        time - (start_contact["wallTimeS"]
                                + delay_ms * 1e-3))))
                    out[f"dx_{delay_ms}ms_m"] = float(
                        position[sample, 0] - start_position[0])
                return out

            real_metrics = loop_metrics(real_racket)
            sim_interval = [
                row for row in interval if row["sourceSeq"] in run.sim
            ]
            sim_metrics = {key: math.nan for key in real_metrics}
            if len(sim_interval) >= 5:
                sim_position = np.asarray([
                    run.sim[row["sourceSeq"]]["racket"]
                    for row in sim_interval
                ])
                # The policy/control timestamps are shared, so use the subset
                # time vector for the nearest 100/200-ms lookups.
                old_time = time
                time = np.asarray([row["wallTimeS"] for row in sim_interval])
                sim_metrics = loop_metrics(sim_position)
                time = old_time
            cycle_rows.append({
                "run": run.name,
                "sensor": run.sensor,
                "cycle_index": index + 1,
                "start_contact_seq": start_contact["seq"],
                "end_contact_seq": end_contact["seq"],
                "duration_s": (
                    end_contact["wallTimeS"] - start_contact["wallTimeS"]),
                **{f"real_{key}": value for key, value in real_metrics.items()},
                **{f"sim_{key}": value for key, value in sim_metrics.items()},
            })
    return contact_rows, cycle_rows


def stats(values: Iterable[Any]) -> dict[str, float | int]:
    arr = finite_array(values)
    if arr.size == 0:
        return {key: math.nan for key in (
            "mean", "std", "median", "mad", "p01", "p05", "p95", "p99", "min", "max")
        } | {"n": 0}
    median = float(np.median(arr))
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": median,
        "mad": float(np.median(np.abs(arr - median))),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(finite_array(a))
    b = np.sort(finite_array(b))
    if not a.size or not b.size:
        return math.nan
    grid = np.sort(np.concatenate((a, b)))
    ca = np.searchsorted(a, grid, side="right") / a.size
    cb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def wasserstein_quantile(a: np.ndarray, b: np.ndarray) -> float:
    a = finite_array(a)
    b = finite_array(b)
    if not a.size or not b.size:
        return math.nan
    q = np.linspace(0.0, 1.0, 1001)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = finite_array(a)
    b = finite_array(b)
    if a.size < 2 or b.size < 2:
        return math.nan
    pooled = math.sqrt(max(1e-18, (float(np.var(a)) + float(np.var(b))) / 2.0))
    return float((np.mean(b) - np.mean(a)) / pooled)


def balanced_matrix(runs: list[RunData], sensor: str, key: str, width: int,
                    points_per_run: int = 200) -> np.ndarray:
    blocks: list[np.ndarray] = []
    for run in runs:
        if run.sensor != sensor or not run.normal:
            continue
        matrix = np.asarray([vector(row[key], width) for row in run.normal])
        indices = np.linspace(0, len(matrix) - 1, points_per_run).round().astype(int)
        blocks.append(matrix[indices])
    return np.concatenate(blocks, axis=0) if blocks else np.empty((0, width))


def robust_scale(normal: np.ndarray, fallback: float = 1e-6) -> float:
    arr = finite_array(normal)
    if not arr.size:
        return fallback
    mad = float(np.median(np.abs(arr - np.median(arr)))) * 1.4826
    std = float(np.std(arr))
    return max(mad, std * 0.25, fallback)


def max_invalid_streak(rows: list[dict[str, Any]], key: str) -> float:
    start: float | None = None
    best = 0.0
    last_t: float | None = None
    for row in rows:
        t = row["wallTimeS"]
        if not bool(row[key]):
            start = t if start is None else start
        elif start is not None:
            best = max(best, t - start)
            start = None
        last_t = t
    if start is not None and last_t is not None:
        best = max(best, last_t - start)
    return best


def refresh_quality(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    refresh = unique_refreshes(rows)
    duration = max(1e-9, rows[-1]["wallTimeS"] - rows[0]["wallTimeS"]) if rows else 0.0
    if not refresh:
        return {
            "refreshCount": 0, "refreshHz": 0.0, "changedCount": 0,
            "changedHz": 0.0, "heldRefreshFraction": math.nan,
            "maxHeldS": math.nan, "grossJumpCount": 0,
        }
    held = 0
    changed = 1
    max_held = 0.0
    held_start: float | None = None
    changed_rows = [refresh[0]]
    for previous, current in zip(refresh[:-1], refresh[1:]):
        same = bool(np.linalg.norm(
            current["measuredBall"] - previous["measuredBall"]) <= 1e-7)
        if same:
            held += 1
            held_start = previous["wallTimeS"] if held_start is None else held_start
            max_held = max(max_held, current["wallTimeS"] - held_start)
        else:
            held_start = None
            changed += 1
            changed_rows.append(current)
    gross_jumps = 0
    for previous, current in zip(changed_rows[:-1], changed_rows[1:]):
        dt = current["wallTimeS"] - previous["wallTimeS"]
        if 1e-4 < dt <= 0.10:
            speed = np.linalg.norm(current["measuredBall"] - previous["measuredBall"]) / dt
            if speed > 12.0:
                gross_jumps += 1
    return {
        "refreshCount": len(refresh),
        "refreshHz": len(refresh) / duration if duration > 0 else math.nan,
        "changedCount": changed,
        "changedHz": changed / duration if duration > 0 else math.nan,
        "heldRefreshFraction": held / max(1, len(refresh) - 1),
        "maxHeldS": max_held,
        "grossJumpCount": gross_jumps,
    }


def local_ballistic_noise(run: RunData) -> dict[str, Any]:
    """Leave-one-out local quadratic residuals, excluding contact neighborhoods."""
    refresh = unique_refreshes(run.normal)
    changed: list[dict[str, Any]] = []
    last_pos: np.ndarray | None = None
    for row in refresh:
        pos = row["measuredBall"]
        if last_pos is None or np.linalg.norm(pos - last_pos) > 1e-7:
            changed.append(row)
            last_pos = pos.copy()
    contact_times = [c["wallTimeS"] for c in run.contacts]
    pos_residuals: list[np.ndarray] = []
    vel_residuals: list[np.ndarray] = []
    for idx, row in enumerate(changed):
        t0 = row["wallTimeS"]
        if any(abs(t0 - tc) < 0.075 for tc in contact_times):
            continue
        neighbor_indices = [
            j for j, other in enumerate(changed)
            if j != idx and abs(other["wallTimeS"] - t0) <= 0.085
            and not any(abs(other["wallTimeS"] - tc) < 0.055 for tc in contact_times)
        ]
        if len(neighbor_indices) < 6:
            continue
        tt = np.asarray([changed[j]["wallTimeS"] - t0 for j in neighbor_indices])
        yy = np.asarray([changed[j]["measuredBall"] for j in neighbor_indices])
        design = np.column_stack((np.ones(tt.size), tt, tt * tt))
        try:
            coeff, *_ = np.linalg.lstsq(design, yy, rcond=None)
        except np.linalg.LinAlgError:
            continue
        predicted_pos = coeff[0]
        predicted_vel = coeff[1]
        pos_residuals.append(row["measuredBall"] - predicted_pos)
        vel_residuals.append(row["measuredVelocity"] - predicted_vel)
    pos_arr = np.asarray(pos_residuals, dtype=np.float64).reshape(-1, 3)
    vel_arr = np.asarray(vel_residuals, dtype=np.float64).reshape(-1, 3)

    def residual_summary(arr: np.ndarray, scale: float) -> dict[str, Any]:
        if arr.size == 0:
            return {"n": 0, "robustSigma": [math.nan] * 3,
                    "rms": [math.nan] * 3, "p95Abs": [math.nan] * 3}
        median = np.median(arr, axis=0)
        sigma = 1.4826 * np.median(np.abs(arr - median), axis=0)
        return {
            "n": int(arr.shape[0]),
            "robustSigma": (sigma * scale).tolist(),
            "rms": (np.sqrt(np.mean(arr * arr, axis=0)) * scale).tolist(),
            "p95Abs": (np.percentile(np.abs(arr), 95, axis=0) * scale).tolist(),
        }

    return {
        "positionMm": residual_summary(pos_arr, 1000.0),
        "velocityMps": residual_summary(vel_arr, 1.0),
    }


def load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # The checkpoint's optimizer NamedTuple points at a training module that
    # imports JAX. A tiny schema-compatible stand-in lets this NumPy-only
    # analysis read the checkpoint without requiring JAX.
    optim_state = namedtuple("OptimState", "m v t")
    optim_state.__module__ = "train_juggle_mjx_ppo"
    fake = types.ModuleType("train_juggle_mjx_ppo")
    fake.OptimState = optim_state
    previous = sys.modules.get("train_juggle_mjx_ppo")
    sys.modules["train_juggle_mjx_ppo"] = fake
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    finally:
        if previous is None:
            sys.modules.pop("train_juggle_mjx_ppo", None)
        else:
            sys.modules["train_juggle_mjx_ppo"] = previous
    return payload["params"], payload["env_cfg"], payload


def actor_mean(params: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    pi = params["pi"]
    x = np.asarray(obs, dtype=np.float64)
    single = x.ndim == 1
    x = x[None, :] if single else x
    h1 = np.tanh(x @ np.asarray(pi["l1"]["w"]) + np.asarray(pi["l1"]["b"]))
    h2 = np.tanh(h1 @ np.asarray(pi["l2"]["w"]) + np.asarray(pi["l2"]["b"]))
    out = h2 @ np.asarray(pi["out"]["w"]) + np.asarray(pi["out"]["b"])
    return out[0] if single else out


def actor_jacobian(params: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    pi = params["pi"]
    w1 = np.asarray(pi["l1"]["w"], dtype=np.float64)
    w2 = np.asarray(pi["l2"]["w"], dtype=np.float64)
    w3 = np.asarray(pi["out"]["w"], dtype=np.float64)
    b1 = np.asarray(pi["l1"]["b"], dtype=np.float64)
    b2 = np.asarray(pi["l2"]["b"], dtype=np.float64)
    h1 = np.tanh(np.asarray(obs, dtype=np.float64) @ w1 + b1)
    h2 = np.tanh(h1 @ w2 + b2)
    return ((w1 * (1.0 - h1 * h1)) @ w2 * (1.0 - h2 * h2)) @ w3


def sample_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    idx = np.linspace(0, len(rows) - 1, count).round().astype(int)
    return [rows[i] for i in idx]


def observation_names(runs: list[RunData]) -> list[str]:
    names = nested(runs[0].metadata, "config", "policy", "observationNames", default=[])
    if isinstance(names, list) and len(names) == 67:
        return [str(name) for name in names]
    return [f"obs_{i:03d}" for i in range(67)]


def obs_group(index: int) -> str:
    groups = (
        (0, 7, "arm_q"), (7, 14, "arm_dq"), (14, 17, "base_q"),
        (17, 20, "base_dq"), (20, 23, "ball_pos"),
        (23, 26, "ball_vel"), (26, 29, "racket_pos"),
        (29, 32, "racket_vel"), (32, 35, "ball_minus_racket"),
        (35, 42, "prev_action"), (42, 49, "arm_cmd_error"),
        (49, 50, "ball_age"), (50, 51, "actuator_tau"),
        (51, 58, "arm_cmd_qvel"), (58, 65, "active_cmd_error"),
        (65, 67, "contact_phase"),
    )
    return next(name for lo, hi, name in groups if lo <= index < hi)


def rows_matrix(rows: list[dict[str, Any]], key: str, width: int) -> np.ndarray:
    return np.asarray([vector(row[key], width) for row in rows], dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def analyze(runs: list[RunData], checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = observation_names(runs)
    params, env_cfg, checkpoint_payload = load_checkpoint(checkpoint)
    camera_obs = balanced_matrix(runs, "camera", "policyObs", 67)
    mocap_obs = balanced_matrix(runs, "mocap", "policyObs", 67)
    camera_action = balanced_matrix(runs, "camera", "policyAction", 7)
    mocap_action = balanced_matrix(runs, "mocap", "policyAction", 7)
    all_normal_obs = np.concatenate((camera_obs, mocap_obs), axis=0)

    # Actor sensitivities and nonlinear one-coordinate counterfactuals.
    jacobian_abs = np.zeros((67, 7), dtype=np.float64)
    jacobian_count = 0
    counterfactual_abs = np.zeros((67, 7), dtype=np.float64)
    counterfactual_count = 0
    actor_errors: list[float] = []
    for run in runs:
        reference = np.nanmedian(rows_matrix(run.normal, "policyObs", 67), axis=0)
        for row in sample_rows(run.normal + run.failure, 300):
            obs = row["policyObs"]
            predicted = np.clip(actor_mean(params, obs), -1.0, 1.0)
            actor_errors.extend((predicted - row["policyAction"]).tolist())
        for row in sample_rows(run.normal, 150):
            jacobian_abs += np.abs(actor_jacobian(params, row["policyObs"]))
            jacobian_count += 1
        for row in sample_rows(run.failure, 100):
            obs = row["policyObs"]
            action = np.clip(actor_mean(params, obs), -1.0, 1.0)
            for dim in range(67):
                cf = obs.copy()
                cf[dim] = reference[dim]
                cf_action = np.clip(actor_mean(params, cf), -1.0, 1.0)
                counterfactual_abs[dim] += np.abs(action - cf_action)
            counterfactual_count += 1
    jacobian_abs /= max(1, jacobian_count)
    counterfactual_abs /= max(1, counterfactual_count)
    actor_rmse = float(np.sqrt(np.mean(np.square(finite_array(actor_errors)))))

    observation_rows: list[dict[str, Any]] = []
    observation_per_run: list[dict[str, Any]] = []
    failure_effect_matrix = np.full((len(runs), 67), np.nan)
    failure_ood_matrix = np.full((len(runs), 67), np.nan)
    global_p01 = np.nanpercentile(all_normal_obs, 1, axis=0)
    global_p99 = np.nanpercentile(all_normal_obs, 99, axis=0)
    global_scale = np.asarray([
        robust_scale(all_normal_obs[:, dim]) for dim in range(67)])
    global_median = np.nanmedian(all_normal_obs, axis=0)

    for dim, name in enumerate(names):
        c = camera_obs[:, dim]
        m = mocap_obs[:, dim]
        cs, ms = stats(c), stats(m)
        per_run_effects: list[float] = []
        per_run_oods: list[float] = []
        for run_index, run in enumerate(runs):
            normal = rows_matrix(run.normal, "policyObs", 67)[:, dim]
            failure = rows_matrix(run.failure, "policyObs", 67)[:, dim]
            ns, fs = stats(normal), stats(failure)
            scale = max(robust_scale(normal), global_scale[dim] * 0.25, 1e-7)
            effect = (float(fs["mean"]) - float(ns["mean"])) / scale
            peak = float(np.max(np.abs(failure - float(ns["median"]))) / scale)
            ood = float(np.mean(
                (failure < global_p01[dim]) | (failure > global_p99[dim])))
            failure_effect_matrix[run_index, dim] = effect
            failure_ood_matrix[run_index, dim] = ood
            per_run_effects.append(effect)
            per_run_oods.append(ood)
            observation_per_run.append({
                "run": run.name,
                "sensor": run.sensor,
                "index": dim,
                "name": name,
                "group": obs_group(dim),
                "normal_n": ns["n"],
                "normal_mean": ns["mean"],
                "normal_std": ns["std"],
                "normal_p01": ns["p01"],
                "normal_p99": ns["p99"],
                "failure_n": fs["n"],
                "failure_mean": fs["mean"],
                "failure_std": fs["std"],
                "failure_min": fs["min"],
                "failure_max": fs["max"],
                "failure_effect_robust_sigma": effect,
                "failure_peak_robust_sigma": peak,
                "failure_ood_fraction": ood,
            })
        action_sensitivity = float(np.linalg.norm(jacobian_abs[dim]))
        cf_impact = float(np.linalg.norm(counterfactual_abs[dim]))
        observation_rows.append({
            "index": dim,
            "name": name,
            "group": obs_group(dim),
            "camera_normal_mean": cs["mean"],
            "camera_normal_std": cs["std"],
            "camera_normal_p01": cs["p01"],
            "camera_normal_p99": cs["p99"],
            "mocap_normal_mean": ms["mean"],
            "mocap_normal_std": ms["std"],
            "mocap_normal_p01": ms["p01"],
            "mocap_normal_p99": ms["p99"],
            "mocap_minus_camera_smd": standardized_mean_difference(c, m),
            "camera_mocap_ks": ks_statistic(c, m),
            "camera_mocap_wasserstein": wasserstein_quantile(c, m),
            "failure_effect_abs_median_runs": float(np.median(np.abs(per_run_effects))),
            "failure_effect_max_abs_runs": float(np.max(np.abs(per_run_effects))),
            "failure_ood_mean_fraction": float(np.mean(per_run_oods)),
            "actor_local_jacobian_l2": action_sensitivity,
            "actor_local_1sigma_effect_l2": action_sensitivity * global_scale[dim],
            "failure_counterfactual_action_l2": cf_impact,
        })

    attribution_rows: list[dict[str, Any]] = []
    for obs_idx, name in enumerate(names):
        for action_idx, action_name in enumerate(ACTION_NAMES):
            attribution_rows.append({
                "obs_index": obs_idx,
                "obs_name": name,
                "obs_group": obs_group(obs_idx),
                "action_index": action_idx,
                "action_name": action_name,
                "mean_abs_local_derivative": jacobian_abs[obs_idx, action_idx],
                "normal_1sigma_local_action_effect": (
                    jacobian_abs[obs_idx, action_idx] * global_scale[obs_idx]),
                "failure_counterfactual_abs_action_change": (
                    counterfactual_abs[obs_idx, action_idx]),
            })

    action_rows: list[dict[str, Any]] = []
    action_per_run: list[dict[str, Any]] = []
    action_effect_matrix = np.full((len(runs), 7), np.nan)
    for dim, name in enumerate(ACTION_NAMES):
        cs, ms = stats(camera_action[:, dim]), stats(mocap_action[:, dim])
        effects: list[float] = []
        for run_index, run in enumerate(runs):
            normal = rows_matrix(run.normal, "policyAction", 7)[:, dim]
            failure = rows_matrix(run.failure, "policyAction", 7)[:, dim]
            ns, fs = stats(normal), stats(failure)
            scale = robust_scale(normal, 1e-5)
            effect = (float(fs["mean"]) - float(ns["mean"])) / scale
            effects.append(effect)
            action_effect_matrix[run_index, dim] = effect
            normal_delta = np.diff(normal)
            failure_delta = np.diff(failure)
            sim_gaps_normal = []
            sim_gaps_failure = []
            for phase_rows, sink in ((run.normal, sim_gaps_normal),
                                     (run.failure, sim_gaps_failure)):
                for row in phase_rows:
                    sim = run.sim.get(row["sourceSeq"])
                    if sim is not None:
                        sink.append(row["policyAction"][dim] - sim["action"][dim])
            action_per_run.append({
                "run": run.name,
                "sensor": run.sensor,
                "action_index": dim,
                "action_name": name,
                "normal_mean": ns["mean"],
                "normal_std": ns["std"],
                "normal_p01": ns["p01"],
                "normal_p99": ns["p99"],
                "normal_saturation_fraction": float(np.mean(np.abs(normal) >= 0.99)),
                "normal_delta_rms": float(np.sqrt(np.mean(normal_delta**2))) if normal_delta.size else math.nan,
                "failure_mean": fs["mean"],
                "failure_std": fs["std"],
                "failure_min": fs["min"],
                "failure_max": fs["max"],
                "failure_saturation_fraction": float(np.mean(np.abs(failure) >= 0.99)),
                "failure_delta_rms": float(np.sqrt(np.mean(failure_delta**2))) if failure_delta.size else math.nan,
                "failure_effect_robust_sigma": effect,
                "real_sim_gap_normal_rms": (
                    float(np.sqrt(np.mean(np.square(sim_gaps_normal))))
                    if sim_gaps_normal else math.nan),
                "real_sim_gap_failure_rms": (
                    float(np.sqrt(np.mean(np.square(sim_gaps_failure))))
                    if sim_gaps_failure else math.nan),
            })
        action_rows.append({
            "action_index": dim,
            "action_name": name,
            "camera_normal_mean": cs["mean"],
            "camera_normal_std": cs["std"],
            "camera_normal_saturation_fraction": float(np.mean(np.abs(camera_action[:, dim]) >= 0.99)),
            "mocap_normal_mean": ms["mean"],
            "mocap_normal_std": ms["std"],
            "mocap_normal_saturation_fraction": float(np.mean(np.abs(mocap_action[:, dim]) >= 0.99)),
            "mocap_minus_camera_smd": standardized_mean_difference(
                camera_action[:, dim], mocap_action[:, dim]),
            "camera_mocap_ks": ks_statistic(camera_action[:, dim], mocap_action[:, dim]),
            "failure_effect_abs_median_runs": float(np.median(np.abs(effects))),
            "failure_effect_max_abs_runs": float(np.max(np.abs(effects))),
        })

    sensor_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    for run in runs:
        q_normal = refresh_quality(run.normal)
        q_failure = refresh_quality(run.failure)
        noise = local_ballistic_noise(run)
        raw_valid_normal = float(np.mean([row["rawValid"] for row in run.normal]))
        raw_valid_failure = float(np.mean([row["rawValid"] for row in run.failure]))
        policy_valid_normal = float(np.mean([row["valid"] for row in run.normal]))
        policy_valid_failure = float(np.mean([row["valid"] for row in run.failure]))
        age_norm = rows_matrix(run.policy, "policyObs", 67)[:, 49]
        initial_end = run.normal[0]["wallTimeS"] + 0.15
        initial_rows = [
            row for row in run.normal if row["wallTimeS"] <= initial_end
        ] or run.normal[:1]
        initial_ball = np.nanmedian(
            rows_matrix(initial_rows, "measuredBall", 3), axis=0)
        initial_racket = np.nanmedian(
            rows_matrix(initial_rows, "policyRacket", 3), axis=0)
        initial_relative = initial_ball - initial_racket
        sensor_rows.append({
            "run": run.name,
            "sensor": run.sensor,
            "normal_duration_s": run.onset["wallTimeS"] - run.policy[0]["wallTimeS"],
            "normal_raw_valid_fraction": raw_valid_normal,
            "normal_policy_valid_fraction": policy_valid_normal,
            "normal_max_raw_invalid_streak_s": max_invalid_streak(run.normal, "rawValid"),
            "normal_refresh_hz": q_normal["refreshHz"],
            "normal_changed_position_hz": q_normal["changedHz"],
            "normal_held_refresh_fraction": q_normal["heldRefreshFraction"],
            "normal_max_held_s": q_normal["maxHeldS"],
            "normal_gross_jump_count": q_normal["grossJumpCount"],
            "normal_position_noise_sigma_x_mm": noise["positionMm"]["robustSigma"][0],
            "normal_position_noise_sigma_y_mm": noise["positionMm"]["robustSigma"][1],
            "normal_position_noise_sigma_z_mm": noise["positionMm"]["robustSigma"][2],
            "normal_velocity_residual_sigma_x_m_s": noise["velocityMps"]["robustSigma"][0],
            "normal_velocity_residual_sigma_y_m_s": noise["velocityMps"]["robustSigma"][1],
            "normal_velocity_residual_sigma_z_m_s": noise["velocityMps"]["robustSigma"][2],
            "initial_150ms_ball_x_m": initial_ball[0],
            "initial_150ms_ball_y_m": initial_ball[1],
            "initial_150ms_ball_z_m": initial_ball[2],
            "initial_150ms_racket_y_m": initial_racket[1],
            "initial_150ms_racket_z_m": initial_racket[2],
            "initial_150ms_ball_minus_racket_y_m": initial_relative[1],
            "initial_150ms_ball_minus_racket_z_m": initial_relative[2],
            "failure_duration_s": run.policy[-1]["wallTimeS"] - run.onset["wallTimeS"],
            "failure_raw_valid_fraction": raw_valid_failure,
            "failure_policy_valid_fraction": policy_valid_failure,
            "failure_max_raw_invalid_streak_s": max_invalid_streak(run.failure, "rawValid"),
            "failure_refresh_hz": q_failure["refreshHz"],
            "failure_changed_position_hz": q_failure["changedHz"],
            "failure_held_refresh_fraction": q_failure["heldRefreshFraction"],
            "failure_max_held_s": q_failure["maxHeldS"],
            "failure_gross_jump_count": q_failure["grossJumpCount"],
            "policy_ball_age_norm_max_all": float(np.max(age_norm)),
        })

        done_y = run.done["workspaceEe"][1]
        y_lo = float(nested(run.metadata, "config", "eeBox", "y0"))
        y_hi = float(nested(run.metadata, "config", "eeBox", "y1"))
        side = "lower" if done_y < y_lo else "upper"
        onset_ball = run.onset["measuredBall"]
        onset_vel = run.onset["measuredVelocity"]
        onset_racket = run.onset["policyRacket"]
        normal_ee_y = finite_array(row["workspaceEe"][1] for row in run.normal)
        failure_ee_y = finite_array(row["workspaceEe"][1] for row in run.failure)
        normal_margin = np.minimum(normal_ee_y - y_lo, y_hi - normal_ee_y)
        failure_margin = np.minimum(failure_ee_y - y_lo, y_hi - failure_ee_y)
        onset_ee_y = float(run.onset["workspaceEe"][1])
        onset_ee_y_margin = min(onset_ee_y - y_lo, y_hi - onset_ee_y)
        train_ball_z = (float(env_cfg["ball_low_termination_z_m"]),
                        float(env_cfg["ball_high_termination_z_m"]))
        failure_ball = rows_matrix(run.failure, "policyObs", 67)[:, 20:26]
        failure_training_z_ood = float(np.mean(
            (failure_ball[:, 2] < train_ball_z[0])
            | (failure_ball[:, 2] > train_ball_z[1])))
        run_rows.append({
            "run": run.name,
            "sensor": run.sensor,
            "policy_duration_s": run.policy[-1]["wallTimeS"] - run.policy[0]["wallTimeS"],
            "normal_duration_s": run.onset["wallTimeS"] - run.policy[0]["wallTimeS"],
            "failure_duration_s": run.policy[-1]["wallTimeS"] - run.onset["wallTimeS"],
            "detected_contacts_total": len(run.contacts),
            "contacts_before_or_at_causal": sum(
                c["wallTimeS"] <= run.onset["wallTimeS"] + 1e-6 for c in run.contacts),
            "causal_ball_observation_seq": run.causal_seq,
            "causal_time_from_policy_start_s": run.onset["wallTimeS"] - run.policy[0]["wallTimeS"],
            "causal_ball_x_m": onset_ball[0],
            "causal_ball_y_m": onset_ball[1],
            "causal_ball_z_m": onset_ball[2],
            "causal_ball_vx_m_s": onset_vel[0],
            "causal_ball_vy_m_s": onset_vel[1],
            "causal_ball_vz_m_s": onset_vel[2],
            "causal_ball_vxy_m_s": float(np.linalg.norm(onset_vel[:2])),
            "causal_racket_x_m": onset_racket[0],
            "causal_racket_y_m": onset_racket[1],
            "causal_racket_z_m": onset_racket[2],
            "causal_workspace_ee_y_m": onset_ee_y,
            "causal_workspace_ee_y_margin_m": onset_ee_y_margin,
            "normal_min_ee_y_margin_m": float(np.min(normal_margin)),
            "failure_min_ee_y_margin_m": float(np.min(failure_margin)),
            "done_reason": run.done["reason"],
            "done_y_side": side,
            "done_workspace_pose_source": run.done["workspacePoseSource"],
            "done_ee_y_m": done_y,
            "ee_y_low_m": y_lo,
            "ee_y_high_m": y_hi,
            "failure_ball_z_outside_training_fraction": failure_training_z_ood,
            "failure_top_obs_effect_index": int(np.nanargmax(np.abs(
                failure_effect_matrix[[r.name for r in runs].index(run.name)]))),
            "failure_top_action_effect_index": int(np.nanargmax(np.abs(
                action_effect_matrix[[r.name for r in runs].index(run.name)]))),
        })
        for contact_index, contact in enumerate(run.contacts, start=1):
            contact_rows.append({
                "run": run.name,
                "sensor": run.sensor,
                "contact_index": contact_index,
                "ball_observation_seq": contact["seq"],
                "time_from_policy_start_s": contact["wallTimeS"] - run.policy[0]["wallTimeS"],
                "is_causal_or_later": contact["wallTimeS"] >= run.onset["wallTimeS"] - 1e-6,
                "held_contact_reacquisition": contact["heldContact"],
                "ball_x_m": contact["position"][0],
                "ball_y_m": contact["position"][1],
                "ball_z_m": contact["position"][2],
                "rel_x_m": contact["relative"][0],
                "rel_y_m": contact["relative"][1],
                "rel_z_m": contact["relative"][2],
                "vz_before_m_s": contact["velocityBefore"][2],
                "vx_after_m_s": contact["velocityAfter"][0],
                "vy_after_m_s": contact["velocityAfter"][1],
                "vz_after_m_s": contact["velocityAfter"][2],
                "vxy_after_m_s": float(np.linalg.norm(contact["velocityAfter"][:2])),
            })

    execution_rows: list[dict[str, Any]] = []
    for joint, name in enumerate(ACTION_NAMES):
        for run in runs:
            for phase_name, phase in (("normal", run.normal), ("failure", run.failure)):
                actual_q = rows_matrix(phase, "actualQ", 7)[:, joint]
                command_q = rows_matrix(phase, "publishedPosition", 7)[:, joint]
                actual_dq = rows_matrix(phase, "actualDq", 7)[:, joint]
                actual_ddq = rows_matrix(phase, "actualDdq", 7)[:, joint]
                error = command_q - actual_q
                sim_q_gap: list[float] = []
                for row in phase:
                    sim = run.sim.get(row["sourceSeq"])
                    if sim is not None:
                        sim_q_gap.append(row["actualQ"][joint] - sim["q"][joint])
                execution_rows.append({
                    "run": run.name,
                    "sensor": run.sensor,
                    "phase": phase_name,
                    "joint_index": joint,
                    "joint_name": name,
                    "command_tracking_error_rms_rad": float(np.sqrt(np.nanmean(error**2))),
                    "command_tracking_error_p95_abs_rad": float(np.nanpercentile(np.abs(error), 95)),
                    "command_tracking_error_max_abs_rad": float(np.nanmax(np.abs(error))),
                    "actual_velocity_p95_abs_rad_s": float(np.nanpercentile(np.abs(actual_dq), 95)),
                    "actual_velocity_max_abs_rad_s": float(np.nanmax(np.abs(actual_dq)),),
                    "actual_acceleration_p95_abs_rad_s2": float(np.nanpercentile(np.abs(actual_ddq), 95)),
                    "actual_acceleration_max_abs_rad_s2": float(np.nanmax(np.abs(actual_ddq))),
                    "real_sim_joint_gap_rms_rad": (
                        float(np.sqrt(np.mean(np.square(sim_q_gap)))) if sim_q_gap else math.nan),
                })

    # Correct a NumPy scalar construction typo defensively before serialization.
    for row in execution_rows:
        value = row["actual_velocity_max_abs_rad_s"]
        if isinstance(value, tuple):
            row["actual_velocity_max_abs_rad_s"] = value[0]

    contact_phase_rows, cycle_rows = contact_and_cycle_dynamics(
        runs, checkpoint)

    write_csv(output_dir / "run_summary.csv", run_rows)
    write_csv(output_dir / "sensor_quality.csv", sensor_rows)
    write_csv(output_dir / "contacts.csv", contact_rows)
    write_csv(output_dir / "observation_dimensions.csv", observation_rows)
    write_csv(output_dir / "observation_per_run.csv", observation_per_run)
    write_csv(output_dir / "observation_action_attribution.csv", attribution_rows)
    write_csv(output_dir / "action_dimensions.csv", action_rows)
    write_csv(output_dir / "action_per_run.csv", action_per_run)
    write_csv(output_dir / "joint_execution_per_run.csv", execution_rows)
    write_csv(output_dir / "contact_phase_dynamics.csv", contact_phase_rows)
    write_csv(output_dir / "inter_contact_cycles.csv", cycle_rows)

    summary = {
        "runs": run_rows,
        "sensorQuality": sensor_rows,
        "observationDimensions": observation_rows,
        "actionDimensions": action_rows,
        "contactPhaseDynamics": contact_phase_rows,
        "interContactCycles": cycle_rows,
        "actorReplayRmse": actor_rmse,
        "checkpoint": {
            "path": str(checkpoint),
            "stageIndex": checkpoint_payload.get("stage_index"),
            "stageName": checkpoint_payload.get("stage_name"),
            "trainingMeanHits": nested(checkpoint_payload, "last_row", "mean_hits"),
            "ballObservationRateHz": env_cfg.get("ball_obs_rate_hz"),
            "ballPositionNoiseStdM": env_cfg.get("ball_obs_pos_noise_std"),
            "ballVelocityNoiseStdMps": env_cfg.get("ball_obs_vel_noise_std"),
            "dropoutProbability": env_cfg.get("ball_obs_dropout_prob"),
            "cameraMissingProbability": env_cfg.get("ball_obs_camera_missing_prob"),
            "ballZTerminationM": [
                env_cfg.get("ball_low_termination_z_m"),
                env_cfg.get("ball_high_termination_z_m"),
            ],
            "ballViewYM": env_cfg.get("ball_view_y_bounds_m"),
            "ballViewXM": env_cfg.get("ball_view_x_bounds_m"),
            "terminateOnBallViewBounds": env_cfg.get("terminate_on_ball_view_bounds"),
            "racketAnchorTerminationDistanceM": 1.1,
            "physicalEeYWorkspaceTrained": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    make_plots(
        runs, output_dir, names, sensor_rows,
        failure_effect_matrix, action_effect_matrix,
    )
    report = build_report(
        runs=runs,
        run_rows=run_rows,
        sensor_rows=sensor_rows,
        observation_rows=observation_rows,
        action_rows=action_rows,
        attribution_rows=attribution_rows,
        action_per_run=action_per_run,
        execution_rows=execution_rows,
        contact_phase_rows=contact_phase_rows,
        cycle_rows=cycle_rows,
        names=names,
        env_cfg=env_cfg,
        checkpoint_payload=checkpoint_payload,
        actor_rmse=actor_rmse,
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def make_plots(
    runs: list[RunData],
    output_dir: Path,
    names: list[str],
    sensor_rows: list[dict[str, Any]],
    obs_effects: np.ndarray,
    action_effects: np.ndarray,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pingpong_mpl_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=False)
    for ax, run in zip(axes.flat, runs):
        t0 = run.policy[0]["wallTimeS"]
        t = np.asarray([row["wallTimeS"] - t0 for row in run.policy])
        ball_y = np.asarray([row["policyObs"][21] for row in run.policy])
        ee_y = np.asarray([row["workspaceEe"][1] for row in run.policy])
        raw_valid = np.asarray([row["rawValid"] for row in run.policy])
        y_lo = float(nested(run.metadata, "config", "eeBox", "y0"))
        y_hi = float(nested(run.metadata, "config", "eeBox", "y1"))
        ax.plot(t, ball_y, label="policy ball y", linewidth=1.1)
        ax.plot(t, ee_y, label="workspace EE y", linewidth=1.1)
        ax.axhline(y_lo, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(y_hi, color="tab:red", linestyle="--", linewidth=0.8)
        onset = run.onset["wallTimeS"] - t0
        ax.axvline(onset, color="black", linestyle=":", linewidth=1.0)
        if np.any(~raw_valid):
            ax.fill_between(t, y_lo, y_hi, where=~raw_valid,
                            color="tab:orange", alpha=0.12, step="mid")
        ax.set_title(f"{run.name[-15:]} ({run.sensor})")
        ax.grid(True, alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Ball/EE lateral trajectory; dotted line = terminal-event cut")
    fig.tight_layout()
    fig.savefig(output_dir / "ball_ee_y_timelines.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(20, 6))
    image = ax.imshow(np.clip(obs_effects, -5, 5), aspect="auto",
                      cmap="coolwarm", vmin=-5, vmax=5)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([run.name[-15:] for run in runs], fontsize=8)
    ax.set_xticks(range(67))
    ax.set_xticklabels([f"{i}:{names[i]}" for i in range(67)],
                       rotation=90, fontsize=6)
    ax.set_title("Failure mean shift vs within-run normal (robust sigma, clipped ±5)")
    fig.colorbar(image, ax=ax, label="robust sigma")
    fig.tight_layout()
    fig.savefig(output_dir / "observation_failure_heatmap.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(np.clip(action_effects, -5, 5), aspect="auto",
                      cmap="coolwarm", vmin=-5, vmax=5)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([run.name[-15:] for run in runs], fontsize=8)
    ax.set_xticks(range(7))
    ax.set_xticklabels(ACTION_NAMES, rotation=35, ha="right")
    ax.set_title("7D action failure shift vs normal (robust sigma, clipped ±5)")
    fig.colorbar(image, ax=ax, label="robust sigma")
    fig.tight_layout()
    fig.savefig(output_dir / "action_failure_heatmap.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [row["run"][-6:] for row in sensor_rows]
    x = np.arange(len(labels))
    held = [100 * float(row["normal_held_refresh_fraction"]) for row in sensor_rows]
    valid = [100 * float(row["normal_raw_valid_fraction"]) for row in sensor_rows]
    axes[0].bar(x, held, color=["tab:blue" if row["sensor"] == "camera" else "tab:green" for row in sensor_rows])
    axes[0].set_title("Normal interval: unchanged refreshes marked new")
    axes[0].set_ylabel("percent")
    axes[0].set_xticks(x, labels, rotation=45)
    axes[1].bar(x, valid, color=["tab:blue" if row["sensor"] == "camera" else "tab:green" for row in sensor_rows])
    axes[1].set_title("Normal interval: rawValid")
    axes[1].set_ylabel("percent")
    axes[1].set_xticks(x, labels, rotation=45)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "sensor_freshness_comparison.png", dpi=150)
    plt.close(fig)


def build_report(
    *, runs: list[RunData], run_rows: list[dict[str, Any]],
    sensor_rows: list[dict[str, Any]], observation_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    action_per_run: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    contact_phase_rows: list[dict[str, Any]],
    cycle_rows: list[dict[str, Any]],
    names: list[str], env_cfg: dict[str, Any], checkpoint_payload: dict[str, Any],
    actor_rmse: float,
) -> str:
    lines: list[str] = []
    lines += [
        "# 2026-08-05/09 相机与动捕颠球：观测质量及 67D→7D 失败分析",
        "",
        "## 结论",
        "",
        "1. **共同根因不是最后一次 done，而是每次碰撞的横向闭环不稳定**：球拍实际法向/切向速度产生横向冲量，球 6D 随后驱动策略追赶；上一动作、命令速度和跟踪误差又把这次追赶带入下一次触球，直到球和 EE 一起漂到 y 工作区边界。九条记录最后都由 `ee_y_out_of_workspace` 退出只是该积累过程的末端结果。",
        "2. **相机特有的主要放大器发生在触球相位，而不是正常飞行噪声**：当前运行使用未同步的 3 帧深度中值和 5 点后向速度。触球检测帧的深度相对当前 sensor depth 中位滞后约 10.5 mm，经相机外参会成为约 -8 mm 的 base-y 偏差；后向速度又把冲击前后状态混合，给策略的 vx/vy/vz 与触球后弹道拟合相比中位偏差约 -0.143/-0.351/-0.978 m/s。",
        "3. **‘画圈’确实由策略轨迹产生，而不是机器人随机抖动**：相机组两次触球间 x-z 路径/首尾弦长中位数 2.90，动捕 1.96；触球后 200 ms 的 x 位移中位数分别为 -11.5 mm 和 +12.0 mm。把同一实测球输入喂给仿真镜像，方向和差异仍然存在，说明球状态触发的动作序列是主因；实机执行误差继续放大触球结果。",
        "4. **实机碰撞姿态没有复现策略所依据的命令/仿真姿态**：在可通过重力弹道自检的相机触球 event，实际拍面法向中位数约为 (-0.070,-0.016,0.994)，命令 FK 为 (-0.092,+0.026,0.990)，仿真镜像为 (-0.020,+0.067,0.994)。尤其 y 法向实际为负、命令和仿真为正；RightArm-5 的触球处跟踪误差是主要贡献之一，因此会把本应回中的碰撞打成切球。",
        "5. **漏检不是五条相机记录的共同根因，但会造成灾难性特例**：前四条在大量有效观测下也会逐次漂移；`121831` 则在最后误检/丢失后继续向策略提供约 10 s 的恒定 6D 球状态，且 `valid=true, age=0`，最终连关节反馈也变陈旧。",
        "6. **动捕 age=0 正常，但当前动捕点/标定仍不可信**：四条动捕 rawValid 均为 100%，所以第 49 维 age=0 符合当前配置。动捕起球 z 比相机稳定低约 21 mm；live path 又直接使用 `marker_5`，没有 marker→球心换算。",
        "7. **训练—部署还缺少失效与工作区契约**：checkpoint 的 dropout/missing 概率为 0，训练球 z<0.98 m 即终止，并且没有实机 `ee_y∈[-0.602,-0.340] m` 约束。因此策略不会识别陈旧球，也不会在实机边界前主动放弃追赶或回中。",
        "",
        "## 切段与口径",
        "",
        "“正常段”从 policy 开始，到人工审阅后选定的漂移/失效因果切点之前；切点以后即使仍碰到球，也计入误差积累的失败段，而不是把最后一次 done 当成根因。触球检测使用最近 5 个刷新中的下降速度和首次正向跨越，并做拍球距离门控，避免后向速度恰好经过 -0.096 m/s 时漏计。噪声使用正常段、剔除触球 ±75 ms 后的局部二次轨迹留一残差；组间分布按 run 等权重采样。",
        "",
        "## 每条记录的失败链",
        "",
        "| run | 源 | 正常/失败 s | 触球数 | 因果点球 y / vy / vxy | 因果点 EE-y 余量 | 退出侧 | rawValid 正常→失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sensor_by_run = {row["run"]: row for row in sensor_rows}
    for row in run_rows:
        sq = sensor_by_run[row["run"]]
        lines.append(
            f"| {row['run']} | {row['sensor']} | "
            f"{fmt(row['normal_duration_s'],2)}/{fmt(row['failure_duration_s'],2)} | "
            f"{row['contacts_before_or_at_causal']} | "
            f"{fmt(row['causal_ball_y_m'])} / {fmt(row['causal_ball_vy_m_s'])} / "
            f"{fmt(row['causal_ball_vxy_m_s'])} | "
            f"{fmt(row['causal_workspace_ee_y_margin_m']*1000,1)} mm | "
            f"{row['done_y_side']} | "
            f"{fmt(100*sq['normal_raw_valid_fraction'],1)}%→{fmt(100*sq['failure_raw_valid_fraction'],1)}% |"
        )

    lines += ["", "逐条解释：", ""]
    explanations = {
        "probe_recording_20260805_155031": "第 3 次回球进入持续漂移段，之后虽又接到 3 次，ball-y/拍面切向误差没有回中；最终实测 EE-y 穿过 -0.602 m。感知短时失效不是共同触发器。",
        "probe_recording_20260805_155253": "经历多次低幅触球后球/拍已到 y 下边缘；最后一次触球发生在 ball-y≈-0.584 m，几乎没有剩余恢复空间，随后实测 EE 越下界。",
        "probe_recording_20260809_121703": "后段回球的横向速度把球送向 y 下界，策略追球至实测 EE-y=-0.6046 m。正常段已有较多 raw miss，但被 valid hold 掩盖。",
        "probe_recording_20260809_121734": "第二次触球已在 y 下边缘附近，约 0.21 s 后 EE 穿下界；属于策略可达工作区与实机工作区不匹配。",
        "probe_recording_20260809_121831": "第 7 次可靠触球后发生连续误检和丢失；球状态最终冻结在 z=0.7785 m、vx=2.5246 m/s，策略仍运行约 10 s。训练在 z<0.98 m 已终止，故这是明确 OOD；最终 joint-state 也不可用，安全监控退到 command-FK 并越上界。",
        "probe_recording_20260809_154527": "触球附近 marker 坐标连续若干帧变化很小，随后状态跳到 vz≈5.09 m/s、vy≈0.98 m/s；全程 rawValid=true，不能称为漏检。策略动作同时出现饱和，该高能回球把拍带向 y 上界，之后即使再次接到球也无恢复余量。",
        "probe_recording_20260809_154656": "首个有效回球后球逐渐漂到 y 上边界，策略追随至实测 EE-y=-0.3381 m。没有相机式漏检，但也没有横向回中约束。",
        "probe_recording_20260809_161246": "多次触球后末次回球向 y 下界运动，策略追随，约 0.30 s 后实测 EE-y=-0.6047 m。",
        "probe_recording_20260809_161432": "末次触球发生时球和拍已靠近 y 上边界；约 0.07 s 后实测 EE-y=-0.3396 m。更低观测噪声无法补偿剩余工作区为零。",
    }
    for run in runs:
        lines.append(f"- `{run.name}`：{explanations[run.name]}")

    camera_phase = [
        row for row in contact_phase_rows
        if row["sensor"] == "camera" and row["postflight_fit_ballistic"]
    ]
    mocap_phase = [
        row for row in contact_phase_rows
        if row["sensor"] == "mocap" and row["postflight_fit_ballistic"]
    ]
    camera_cycles = [row for row in cycle_rows if row["sensor"] == "camera"]
    mocap_cycles = [row for row in cycle_rows if row["sensor"] == "mocap"]

    def phase_median(rows: list[dict[str, Any]], key: str) -> float:
        return float(np.nanmedian(finite_array(row.get(key) for row in rows)))

    def vector_median(
        rows: list[dict[str, Any]], prefix: str, suffixes: Iterable[Any],
    ) -> np.ndarray:
        return np.asarray([
            phase_median(rows, f"{prefix}{suffix}") for suffix in suffixes
        ])

    camera_velocity_error = vector_median(
        camera_phase, "event_velocity_error_v",
        ["x_m_s", "y_m_s", "z_m_s"])
    mocap_velocity_error = vector_median(
        mocap_phase, "event_velocity_error_v",
        ["x_m_s", "y_m_s", "z_m_s"])
    camera_event_velocity = vector_median(
        camera_phase, "event_ball_v", ["x_m_s", "y_m_s", "z_m_s"])
    mocap_event_velocity = vector_median(
        mocap_phase, "event_ball_v", ["x_m_s", "y_m_s", "z_m_s"])
    camera_action = vector_median(camera_phase, "event_action_", range(7))
    mocap_action = vector_median(mocap_phase, "event_action_", range(7))
    camera_jac_action = vector_median(
        camera_phase, "event_jacobian_action_", "xyz")
    mocap_jac_action = vector_median(
        mocap_phase, "event_jacobian_action_", "xyz")
    camera_actual_normal = vector_median(
        camera_phase, "actual_normal_", "xyz")
    camera_command_normal = vector_median(
        camera_phase, "command_normal_", "xyz")
    camera_sim_normal = vector_median(
        camera_phase, "sim_normal_", "xyz")
    camera_q_error_deg = vector_median(
        camera_phase, "event_actual_minus_command_q", [
            f"{index}_deg" for index in range(7)])

    lines += [
        "",
        "除 `121831` 的陈旧状态特例外，退出方向与末次有效回球的横向状态一致：5 个下界退出的因果点 `vy` 全为负；动捕上界退出中，`154527` 的 `vy=+0.982 m/s`，`161432` 虽已反向但触球时球已到 y=-0.343 m、EE-y 仅余约 26 mm，70 ms 内无法回中。末次回球 `vxy` 中位数相机为 "
        f"{fmt(np.median([row['causal_ball_vxy_m_s'] for row in run_rows if row['sensor']=='camera']),3)} m/s，"
        "动捕反而为 "
        f"{fmt(np.median([row['causal_ball_vxy_m_s'] for row in run_rows if row['sensor']=='mocap']),3)} m/s。"
        "因此本批动捕实验没有增加次数，并不是位置抖动没有下降，而是末次触球产生的横向冲量/落点漂移没有下降。",
        "",
        "## 相机‘画圈’与触球相位根因",
        "",
        "修正后共识别到相机 26 次、动捕 15 次可信速度反转；其中触球后弹道 z 加速度落在 [-13,-7] m/s² 的分别为 "
        f"{len(camera_phase)} 和 {len(mocap_phase)} 次。以下速度比较只使用这些可由重力弹道自检的触球。",
        "",
        "| 指标（组中位数） | camera | mocap |",
        "|---|---:|---:|",
        f"| 触球反转检测相对最近距离时刻的延迟 | {fmt(phase_median(camera_phase, 'detector_delay_from_closest_approach_ms'),1)} ms | {fmt(phase_median(mocap_phase, 'detector_delay_from_closest_approach_ms'),1)} ms |",
        f"| event 球速度 xyz (m/s) | {fmt(camera_event_velocity[0])}/{fmt(camera_event_velocity[1])}/{fmt(camera_event_velocity[2])} | {fmt(mocap_event_velocity[0])}/{fmt(mocap_event_velocity[1])}/{fmt(mocap_event_velocity[2])} |",
        f"| event 速度减去触球后弹道速度 xyz (m/s) | {fmt(camera_velocity_error[0])}/{fmt(camera_velocity_error[1])}/{fmt(camera_velocity_error[2])} | {fmt(mocap_velocity_error[0])}/{fmt(mocap_velocity_error[1])}/{fmt(mocap_velocity_error[2])} |",
        f"| event 动作 0..6 | {'/'.join(fmt(value,2) for value in camera_action)} | {'/'.join(fmt(value,2) for value in mocap_action)} |",
        f"| J(q)·action 的局部 xyz 方向 | {fmt(camera_jac_action[0])}/{fmt(camera_jac_action[1])}/{fmt(camera_jac_action[2])} | {fmt(mocap_jac_action[0])}/{fmt(mocap_jac_action[1])}/{fmt(mocap_jac_action[2])} |",
        f"| 两触球间实机 x-z 路径/弦长 | {fmt(phase_median(camera_cycles, 'real_xz_path_chord_ratio'),2)} | {fmt(phase_median(mocap_cycles, 'real_xz_path_chord_ratio'),2)} |",
        f"| 触球后 200 ms 实机 Δx | {fmt(1000*phase_median(camera_cycles, 'real_dx_200ms_m'),1)} mm | {fmt(1000*phase_median(mocap_cycles, 'real_dx_200ms_m'),1)} mm |",
        f"| 触球后 200 ms 仿真镜像 Δx | {fmt(1000*phase_median(camera_cycles, 'sim_dx_200ms_m'),1)} mm | {fmt(1000*phase_median(mocap_cycles, 'sim_dx_200ms_m'),1)} mm |",
        "",
        "相机 event 的 vx/vy 并不是后续整段飞行速度：与 55 ms 后开始的弹道拟合相比，横向中位偏差达到 "
        f"{fmt(camera_velocity_error[0],3)}/{fmt(camera_velocity_error[1],3)} m/s，而动捕是 "
        f"{fmt(mocap_velocity_error[0],3)}/{fmt(mocap_velocity_error[1],3)} m/s。"
        "这会把 event 动作 0 从动捕附近的中值推到相机的正饱和，同时让动作 2/3 更负。"
        "按触球处平移 Jacobian，动作 0 主要产生 -y/+z，动作 2/3 同时影响 +x/+y/z；合成的局部方向从动捕的 "
        f"({fmt(mocap_jac_action[0])},{fmt(mocap_jac_action[1])},{fmt(mocap_jac_action[2])}) 变为相机的 "
        f"({fmt(camera_jac_action[0])},{fmt(camera_jac_action[1])},{fmt(camera_jac_action[2])})。"
        "这正对应相机触球后先向 -x/-y 撤、再回来接球的闭环。",
        "",
        "相机深度链给出了可直接定位的来源：`synchronizedDepth=false`，重建用 3 帧深度中值，速度再取 5 个位置端点差。"
        "触球 event 中 used depth 比当前 sensor depth 中位大 "
        f"{fmt(phase_median(camera_phase, 'used_minus_sensor_depth_mm'),1)} mm；"
        "按当前 `T_base_camera` 的相机深度轴，这等价于 base xyz 偏置约 "
        f"{fmt(phase_median(camera_phase, 'depth_temporal_bias_base_x_mm'),1)}/"
        f"{fmt(phase_median(camera_phase, 'depth_temporal_bias_base_y_mm'),1)}/"
        f"{fmt(phase_median(camera_phase, 'depth_temporal_bias_base_z_mm'),1)} mm。"
        "所以相机在正常飞行段只有毫米级噪声，并不代表碰撞附近的 6D 状态也正确。",
        "",
        "实机姿态又把该动作差异变成真实横向冲量。相机可信触球处实际法向中位数为 "
        f"({fmt(camera_actual_normal[0],3)},{fmt(camera_actual_normal[1],3)},{fmt(camera_actual_normal[2],3)})，"
        "命令 FK 为 "
        f"({fmt(camera_command_normal[0],3)},{fmt(camera_command_normal[1],3)},{fmt(camera_command_normal[2],3)})，"
        "仿真镜像为 "
        f"({fmt(camera_sim_normal[0],3)},{fmt(camera_sim_normal[1],3)},{fmt(camera_sim_normal[2],3)})。"
        "实际 y 法向与命令/仿真符号相反，说明执行器相位/跟踪误差不是普通位置 RMS，而是会直接改变反弹方向的碰撞误差。",
        "",
        "## 正常段传感器质量",
        "",
        "| run | raw 有效率 | 刷新 Hz | 坐标变化 Hz | 重复坐标刷新 | 最长相同坐标 ms | 局部位置噪声 σ xyz (mm) | 速度残差 σ xyz (m/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sensor_rows:
        lines.append(
            f"| {row['run']} | {fmt(100*row['normal_raw_valid_fraction'],1)}% | "
            f"{fmt(row['normal_refresh_hz'],1)} | {fmt(row['normal_changed_position_hz'],1)} | "
            f"{fmt(100*row['normal_held_refresh_fraction'],1)}% | {fmt(1000*row['normal_max_held_s'],1)} | "
            f"{fmt(row['normal_position_noise_sigma_x_mm'],2)}/"
            f"{fmt(row['normal_position_noise_sigma_y_mm'],2)}/"
            f"{fmt(row['normal_position_noise_sigma_z_mm'],2)} | "
            f"{fmt(row['normal_velocity_residual_sigma_x_m_s'],3)}/"
            f"{fmt(row['normal_velocity_residual_sigma_y_m_s'],3)}/"
            f"{fmt(row['normal_velocity_residual_sigma_z_m_s'],3)} |"
        )

    camera_sensor = [row for row in sensor_rows if row["sensor"] == "camera"]
    mocap_sensor = [row for row in sensor_rows if row["sensor"] == "mocap"]
    def group_median(rows: list[dict[str, Any]], key: str) -> float:
        return float(np.nanmedian(finite_array(row[key] for row in rows)))

    lines += [
        "",
        "按 run 取中位数后，动捕相对相机的 x/y 位置残差由 "
        f"{fmt(group_median(camera_sensor, 'normal_position_noise_sigma_x_mm'),2)}/"
        f"{fmt(group_median(camera_sensor, 'normal_position_noise_sigma_y_mm'),2)} mm 降到 "
        f"{fmt(group_median(mocap_sensor, 'normal_position_noise_sigma_x_mm'),2)}/"
        f"{fmt(group_median(mocap_sensor, 'normal_position_noise_sigma_y_mm'),2)} mm；"
        "但 z 残差几乎没有改善（"
        f"{fmt(group_median(camera_sensor, 'normal_position_noise_sigma_z_mm'),2)} vs "
        f"{fmt(group_median(mocap_sensor, 'normal_position_noise_sigma_z_mm'),2)} mm），"
        "速度 z 残差还略高（"
        f"{fmt(group_median(camera_sensor, 'normal_velocity_residual_sigma_z_m_s'),3)} vs "
        f"{fmt(group_median(mocap_sensor, 'normal_velocity_residual_sigma_z_m_s'),3)} m/s）。"
        "动捕 rawValid 为 100%；重复坐标刷新中位数为 "
        f"{fmt(100*group_median(mocap_sensor, 'normal_held_refresh_fraction'),1)}%，"
        f"最长连续相同坐标中位数 {fmt(1000*group_median(mocap_sensor, 'normal_max_held_s'),1)} ms。"
        "该指标描述数值更新特性，不等价于漏检率。",
        "",
        "首个 150 ms 的组中位数（用于检查放球/坐标偏置）：",
        "",
        "| source | ball xyz (m) | racket yz (m) | ball-racket yz (mm) |",
        "|---|---:|---:|---:|",
    ]
    for sensor_name, rows in (("camera", camera_sensor), ("mocap", mocap_sensor)):
        lines.append(
            f"| {sensor_name} | "
            f"{fmt(group_median(rows, 'initial_150ms_ball_x_m'))}/"
            f"{fmt(group_median(rows, 'initial_150ms_ball_y_m'))}/"
            f"{fmt(group_median(rows, 'initial_150ms_ball_z_m'))} | "
            f"{fmt(group_median(rows, 'initial_150ms_racket_y_m'))}/"
            f"{fmt(group_median(rows, 'initial_150ms_racket_z_m'))} | "
            f"{fmt(1000*group_median(rows, 'initial_150ms_ball_minus_racket_y_m'),1)}/"
            f"{fmt(1000*group_median(rows, 'initial_150ms_ball_minus_racket_z_m'),1)} |"
        )

    initial_delta_mm = np.asarray([
        group_median(mocap_sensor, f"initial_150ms_ball_{axis}_m")
        - group_median(camera_sensor, f"initial_150ms_ball_{axis}_m")
        for axis in "xyz"
    ]) * 1000.0
    camera_z = finite_array(row["initial_150ms_ball_z_m"] for row in camera_sensor)
    mocap_z = finite_array(row["initial_150ms_ball_z_m"] for row in mocap_sensor)
    racket_z_all = finite_array(row["initial_150ms_racket_z_m"] for row in sensor_rows)
    lines += [
        "",
        "这里的 `ball-racket` 只是观测球坐标减去 FK 的球拍参考点坐标，不是测得的球面—拍面物理间隙。"
        f"动捕减相机的起球点中位数差为 xyz={fmt(initial_delta_mm[0],1)}/"
        f"{fmt(initial_delta_mm[1],1)}/{fmt(initial_delta_mm[2],1)} mm。"
        f"相机 z 范围 {fmt(np.min(camera_z),4)}–{fmt(np.max(camera_z),4)} m，"
        f"动捕 z 范围 {fmt(np.min(mocap_z),4)}–{fmt(np.max(mocap_z),4)} m；"
        f"与此同时九次 FK 球拍 z 只跨 {fmt(1000*(np.max(racket_z_all)-np.min(racket_z_all)),2)} mm。"
        "因此约 21 mm 的 z 差不能由机器人起始姿态解释。",
        "",
        "逐 run 起球坐标：",
        "",
        "| run | source | ball xyz (m) | ball-racket yz (mm) |",
        "|---|---|---:|---:|",
    ]
    for row in sensor_rows:
        lines.append(
            f"| {row['run']} | {row['sensor']} | "
            f"{fmt(row['initial_150ms_ball_x_m'])}/"
            f"{fmt(row['initial_150ms_ball_y_m'])}/"
            f"{fmt(row['initial_150ms_ball_z_m'])} | "
            f"{fmt(1000*row['initial_150ms_ball_minus_racket_y_m'],1)}/"
            f"{fmt(1000*row['initial_150ms_ball_minus_racket_z_m'],1)} |"
        )

    lines += [
        "",
        "`policy_ball_age_norm_max_all` 对九条记录全部为 0。对动捕，这是因为 rawValid 始终为真且配置 `rlUseValidBallObsAge=false`，属于正常设计；不能用它推断漏检。对相机，rawValid=false 时外层仍可能把 last-known 状态作为 policy valid，因此第 49 维同样不增长，这才是需要单独处理的失效提示缺口。",
        "",
        "## 球 6D 正常分布：相机 vs 动捕",
        "",
        "| 维度 | camera mean±std | mocap mean±std | SMD (mocap-camera) | KS | Wasserstein |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in observation_rows[20:26]:
        lines.append(
            f"| {row['index']} {row['name']} | {fmt(row['camera_normal_mean'])}±{fmt(row['camera_normal_std'])} | "
            f"{fmt(row['mocap_normal_mean'])}±{fmt(row['mocap_normal_std'])} | "
            f"{fmt(row['mocap_minus_camera_smd'],2)} | {fmt(row['camera_mocap_ks'],2)} | "
            f"{fmt(row['camera_mocap_wasserstein'],4)} |"
        )
    lines += [
        "",
        "SMD 绝对值 >0.5 已是明显分布差异；这里的正常飞行分布差异同时包含传感器偏置和两组闭环轨迹不同，不能全部解释成测量 bias。起球点则有更强的可比性：约 21 mm 的稳定 z 差说明至少存在固定参考点/标定差，但仅凭现有非同步记录还不能唯一归因到 marker-to-center 或某一个传感器。",
        "",
        "## 67 维输入逐维结果",
        "",
        "下表中的 failure σ 是九条 run 的失败均值偏移绝对值中位数；OOD 是失败样本落到所有正常样本 1–99% 区间外的比例；counterfactual 是把该维单独替换为同 run 正常中位数后，7D 动作变化的 L2 均值。",
        "",
        "| idx | name | group | C μ±σ | M μ±σ | C→M SMD | fail σ | fail OOD | local 1σ→action | fail counterfactual |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in observation_rows:
        lines.append(
            f"| {row['index']} | {row['name']} | {row['group']} | "
            f"{fmt(row['camera_normal_mean'])}±{fmt(row['camera_normal_std'])} | "
            f"{fmt(row['mocap_normal_mean'])}±{fmt(row['mocap_normal_std'])} | "
            f"{fmt(row['mocap_minus_camera_smd'],2)} | "
            f"{fmt(row['failure_effect_abs_median_runs'],2)} | "
            f"{fmt(100*row['failure_ood_mean_fraction'],1)}% | "
            f"{fmt(row['actor_local_1sigma_effect_l2'],4)} | "
            f"{fmt(row['failure_counterfactual_action_l2'],4)} |"
        )

    top_cf = sorted(observation_rows,
                    key=lambda row: row["failure_counterfactual_action_l2"],
                    reverse=True)[:12]
    top_failure = sorted(observation_rows,
                         key=lambda row: row["failure_effect_abs_median_runs"],
                         reverse=True)[:12]
    lines += [
        "",
        "失败期对动作贡献最高的输入维度：",
        "",
        ", ".join(
            f"{row['index']}:{row['name']} ({fmt(row['failure_counterfactual_action_l2'],4)})"
            for row in top_cf),
        "",
        "跨 run 失败偏移最一致/最大的维度：",
        "",
        ", ".join(
            f"{row['index']}:{row['name']} ({fmt(row['failure_effect_abs_median_runs'],2)}σ)"
            for row in top_failure),
        "",
        "## 7 维策略输出逐维结果",
        "",
        "| action | C μ±σ / sat | M μ±σ / sat | C→M SMD | fail σ median/max |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in action_rows:
        lines.append(
            f"| {row['action_index']} {row['action_name']} | "
            f"{fmt(row['camera_normal_mean'])}±{fmt(row['camera_normal_std'])} / "
            f"{fmt(100*row['camera_normal_saturation_fraction'],1)}% | "
            f"{fmt(row['mocap_normal_mean'])}±{fmt(row['mocap_normal_std'])} / "
            f"{fmt(100*row['mocap_normal_saturation_fraction'],1)}% | "
            f"{fmt(row['mocap_minus_camera_smd'],2)} | "
            f"{fmt(row['failure_effect_abs_median_runs'],2)}/"
            f"{fmt(row['failure_effect_max_abs_runs'],2)} |"
        )

    lines += [
        "",
        "每个输出维度的主要失败期输入驱动（单维恢复到本 run 正常中位数时，该 action 的平均绝对变化；这是条件反事实，相关输入之间不能相加）：",
        "",
        "| action | top-5 input drivers | 最大失败偏移 run（方向，σ） |",
        "|---|---|---:|",
    ]
    for action_idx, action_name in enumerate(ACTION_NAMES):
        drivers = sorted(
            (row for row in attribution_rows if row["action_index"] == action_idx),
            key=lambda row: row["failure_counterfactual_abs_action_change"],
            reverse=True,
        )[:5]
        candidates = [
            item for item in action_per_run if item["action_index"] == action_idx
        ]
        largest = max(
            candidates,
            key=lambda item: abs(float(item["failure_effect_robust_sigma"])),
        )
        driver_text = ", ".join(
            f"{item['obs_index']}:{item['obs_name']} "
            f"({fmt(item['failure_counterfactual_abs_action_change'],3)})"
            for item in drivers
        )
        direction = "+" if largest["failure_effect_robust_sigma"] >= 0 else "−"
        lines.append(
            f"| {action_idx} {action_name} | {driver_text} | "
            f"{largest['run'][-6:]} ({direction}{fmt(abs(largest['failure_effect_robust_sigma']),2)}) |"
        )

    lines += [
        "",
        "`observation_action_attribution.csv` 给出完整 67×7 映射；`action_per_run.csv` 给出每一条记录、每个 action 的正常/失败分布、饱和、逐 tick 突变和 real-sim 动作差。checkpoint actor 用记录的 67D observation 重算动作的 RMSE 为 "
        f"`{actor_rmse:.3e}`，因此逐维敏感度/反事实归因确实对应本次部署 actor。",
        "",
        "## 关节执行链",
        "",
    ]
    normal_exec = [row for row in execution_rows if row["phase"] == "normal"]
    failure_exec = [row for row in execution_rows if row["phase"] == "failure"]
    lines += [
        "| joint | normal cmd-q RMS (median/max run, deg) | failure cmd-q RMS (median/max run, deg) | normal real-sim q RMS (median, deg) | failure real-sim q RMS (median, deg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for joint, name in enumerate(ACTION_NAMES):
        nrows = [row for row in normal_exec if row["joint_index"] == joint]
        frows = [row for row in failure_exec if row["joint_index"] == joint]
        nerr = np.rad2deg(finite_array(row["command_tracking_error_rms_rad"] for row in nrows))
        ferr = np.rad2deg(finite_array(row["command_tracking_error_rms_rad"] for row in frows))
        nsim = np.rad2deg(finite_array(row["real_sim_joint_gap_rms_rad"] for row in nrows))
        fsim = np.rad2deg(finite_array(row["real_sim_joint_gap_rms_rad"] for row in frows))
        lines.append(
            f"| {name} | {fmt(np.median(nerr),2)}/{fmt(np.max(nerr),2)} | "
            f"{fmt(np.median(ferr),2)}/{fmt(np.max(ferr),2)} | "
            f"{fmt(np.median(nsim),2)} | {fmt(np.median(fsim),2)} |"
        )

    lines += [
        "",
        "执行误差会改变拍速、拍面和触球冲量，但它不能单独解释九次都由同一个实机 EE-y 约束退出。尤其 `121831` 的失败段包含约 10 s 的关节反馈陈旧，导致上表 max 被异常放大，因此应优先看跨 run 中位数。更可靠的解释是：执行残差放大了回球横向速度误差，而感知新鲜度和 workspace contract 缺口决定了策略会继续追球直到保护停车。",
        "",
        "## 与 checkpoint 训练条件的量化对照",
        "",
        f"- checkpoint: stage {checkpoint_payload.get('stage_index')} `{checkpoint_payload.get('stage_name')}`；训练日志 mean_hits={fmt(nested(checkpoint_payload, 'last_row', 'mean_hits'),2)}。",
        f"- 训练球观测刷新 {fmt(env_cfg.get('ball_obs_rate_hz'),1)} Hz，位置噪声 σ={fmt(1000*float(env_cfg.get('ball_obs_pos_noise_std',0)),1)} mm，速度噪声 σ={fmt(env_cfg.get('ball_obs_vel_noise_std'),2)} m/s。",
        f"- dropout_prob={env_cfg.get('ball_obs_dropout_prob')}、camera_missing_prob={env_cfg.get('ball_obs_camera_missing_prob')}、missing_episode_prob={env_cfg.get('ball_obs_missing_episode_coherent_prob')}。",
        f"- 训练球 z 终止范围 [{fmt(env_cfg.get('ball_low_termination_z_m'),2)}, {fmt(env_cfg.get('ball_high_termination_z_m'),2)}] m；部署 probe 使用 [0.30, 1.90] m。",
        f"- 训练 ball-view y 名义范围 {env_cfg.get('ball_view_y_bounds_m')} m，但 `terminate_on_ball_view_bounds={env_cfg.get('terminate_on_ball_view_bounds')}`；实机 EE-y 是硬约束 [-0.602, -0.340] m，训练 termination 中不存在该约束。",
        "",
        "## 不重训、只调整实物部署端的优先顺序",
        "",
        "1. **先消掉相机深度相位误差，再谈策略**。当前外参的相机深度轴有约 0.77 的分量落在 base-y，因此 10 mm 深度滞后会直接伪造成约 8 mm 横向偏差。若允许改运行参数，先启用同一 IR/depth frameset 的 synchronized depth（此时 3 帧中值不参与控制重建），其余滤波保持不变做 A/B；若严格只能移动硬件，则把相机安装得更接近俯视，使 `|R_base_camera[y,2]|` 从 0.77 降到 0.3 以下，同时保证球拍不遮挡触球点。验收不是看静止噪声，而是看 `event_velocity_error_vx/vy` 是否降到各 <0.10 m/s。",
        "2. **给球拍做固定小角度垫片，直接抵消实际碰撞法向**。相机触球实际法向约等于 x/y 倾角 -2.5°/-0.9°，首次触球约 -3.0°/-3.5°。建议从‘法向朝 +x 3°、朝 +y 2°’开始，按 0.5° 步进调；目标是多次触球的 outgoing vx/vy 中位数都在 ±0.10 m/s，而不是只把静止拍面调水平。固定垫片是对动态误差的折中，最终角度应以飞行拟合结果为准。",
        "3. **优先校正 RightArm-5 的动态跟踪，而不是整体提高所有关节增益**。相机可信触球处 actual-command 关节误差中，RightArm-5 中位数约 "
        f"{fmt(camera_q_error_deg[5],1)}°；其旋转 Jacobian 会把该误差主要投到负 y 法向。先用当前颠球命令做空载跟踪，分别检查慢速和触球速度：慢速也偏说明编码器零位/装配偏置，只有快速偏说明时延、速度前馈或驱动增益不匹配。目标是触球时 RightArm-5 误差 <1.5°、实际与命令拍面 x/y 倾角差各 <1°。",
        "4. **把起球夹具固定到相机所见的球心，而不是按动捕点调**。相机起球相对 FK 拍点的组中位数约 x/y/z=+2/+5/+27.5 mm，z 已接近训练 reset 的 26.3 mm，不要按动捕的 6.4 mm 去降低。把夹具 x/y 调到 0±2 mm（按当前中位数约需 -2 mm x、-5 mm y），并把释放水平速度控制在 0.02 m/s 内。",
        "5. **若仍有切球，再换低摩擦、均匀且刚性的拍面覆盖层**。高摩擦胶皮会把拍面的切向速度和旋转误差传给球；光滑硬片能降低这种放大。当前触球离拍心只有数毫米，先加大拍面并不能解决根因，反而会改变训练中的碰撞几何，因此不作为首选。",
        "6. **按单次→两次→五次逐级验收**。每种配置先做至少 20 次单次碰撞，记录触球后 55–250 ms 弹道拟合的 vx/vy、实际拍面法向、触球后 200 ms 的 Δx；单次横向速度未收敛前不要用‘偶尔多颠了几次’判断优劣。当前初始 EE-y=-0.466 m 已接近工作区中心 -0.471 m，平移整个起球区不会消除逐次漂移。",
        "",
        "## 若以后允许修改部署逻辑或重训",
        "",
        "1. **分别处理两种 age 语义**：动捕 rawValid=100% 时维持 age=0 没有问题；相机 rawValid=false 后的 last-known hold 应增长 age，并在超时后 policy valid=false。若以后希望策略感知亚帧延迟，再把 mocap source frame/timestamp age 接入第 49 维，但这不是本批动捕失败的先决修复。",
        "2. **把部署 done 与训练对齐**：至少把真实 EE xyz 工作区（尤其 y）纳入训练 termination/reward，或在策略外加入不会继续追出边界的可恢复 target projection/recenter。球 z 下限也应与训练 0.98 m 对齐，避免低于训练支持域仍运行。",
        "3. **先判定 marker 点还是坐标标定偏置**：固定球心并旋转球；若报告点随球旋转约一个半径，则 `marker_5` 不是球心且单 marker 无法在飞行旋转时用固定世界偏移修正，应使用刚体中心/球心解算；若偏差始终为固定世界向量，则优先修正 `T_robot_mocap` 的平移。",
        "4. **训练真实相机失效模式**：加入 30–100 ms last-value hold、错误 `valid`、视野丢失以及速度重建尖峰；动捕则按实测坐标噪声、速度残差和可能的球心偏差建模，不再把重复坐标直接标成漏检。",
        "5. **触球稳定性再训练/校准**：对每次触球的 outgoing vxy、拍面法向和接触点做闭环约束，特别惩罚把下一次落点送到 EE-y 边界的回球。训练已有 hit-vxy reward，但没有真实 EE-y 可达域，目标仍不闭合。",
        "6. **最后处理执行器模型残差**：依据 `joint_execution_per_run.csv` 中各关节 real-sim gap 更新二阶模型；用相同 ball trace 做 HIL replay，确认修感知/workspace 后剩余的回球漂移有多少由关节响应造成。",
        "",
        "## 产物索引",
        "",
        "- `run_summary.csv`：逐 run 因果切点与安全边界。",
        "- `sensor_quality.csv`：rawValid、重复坐标、起球点、位置/速度残差。",
        "- `contacts.csv`：逐触球输入/输出速度与拍球相对位置。",
        "- `contact_phase_dynamics.csv`：触球相位的深度滞后、弹道速度、7D 动作、拍面法向和 real/sim 差异。",
        "- `inter_contact_cycles.csv`：逐个触球间循环的实机/仿真 x-z 路径、回撤和 100/200 ms 位移。",
        "- `observation_dimensions.csv` / `observation_per_run.csv`：67D 汇总及 9×67 明细。",
        "- `observation_action_attribution.csv`：完整 67×7 actor Jacobian 与失败反事实贡献。",
        "- `action_dimensions.csv` / `action_per_run.csv`：7D 输出汇总及逐 run 明细。",
        "- `joint_execution_per_run.csv`：7 关节命令、反馈、速度/加速度和 real-sim gap。",
        "- PNG：球/EE-y 时间线、67D/7D 失败热图和传感器新鲜度对比。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-root",
        type=Path,
        default=workspace / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=workspace / (
            "simcode5/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/"
            "goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "analysis_results/juggling_camera_mocap_20260809",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = [load_run(args.recording_root, *spec) for spec in RUN_SPECS]
    summary = analyze(runs, args.checkpoint, args.output_dir)
    print(f"Wrote analysis to {args.output_dir}")
    print(f"Actor replay RMSE: {summary['actorReplayRmse']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
