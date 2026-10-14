#!/usr/bin/env python3
"""Replay recorded and robust-cleaned real ball trajectories through policy.

Both branches use the same simcode5 checkpoint, MuJoCo plant, initial arm pose,
control ticks, and valid juggling interval.  The raw branch receives the exact
recorded policy-facing 6D ball state.  The clean branch receives the acausal
robust/interpolated reference produced by offline_ball_trajectory_study.py.

The ball trajectory is forced, so this is a counterfactual policy-sensitivity
test rather than a closed-loop estimate of how many future real catches would
succeed.  Catch geometry is always evaluated against the same clean reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.rl_policy import (  # noqa: E402
    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
)
from pingpong_controller.sim_mirror import (  # noqa: E402
    RealBallSimMirror,
    SimMirrorConfig,
)
from pingpong_controller.tools.offline_ball_trajectory_study import (  # noqa: E402
    TERMINAL_TAIL_GUARD_S,
    build_reference,
    detect_contacts,
    discover_runs,
    fresh_camera_ticks,
    interpolate_reference,
    load_ticks,
    recording_capability,
    select_valid_phase,
)


CONTROL_DT_S = 0.005
ANOMALY_POSITION_M = 0.035
ANOMALY_VELOCITY_M_S = 1.5
ANOMALY_RESPONSE_LAG_S = 0.10
DESCENDING_SPEED_M_S = -0.15
RACKET_PLANE_HALF_WIDTH_M = 0.12
CATCH_GEOMETRY_THRESHOLD_M = 0.10


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return round(float(np.percentile(array, percentile)), 6)


def _rms(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return None
    return round(float(np.sqrt(np.mean(array * array))), 6)


def _vec(row: dict[str, Any], key: str, length: int) -> np.ndarray:
    try:
        value = np.asarray(row.get(key), dtype=np.float64).reshape(length)
    except (TypeError, ValueError):
        value = np.full(length, math.nan, dtype=np.float64)
    return value


@dataclass
class ReplayArrays:
    action: np.ndarray
    command: np.ndarray
    joint: np.ndarray
    racket: np.ndarray


class QuietLogger:
    def info(self, _message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        print(f"[SimMirror][WARN] {message}", flush=True)

    def error(self, message: str) -> None:
        print(f"[SimMirror][ERROR] {message}", flush=True)


def make_mirror() -> RealBallSimMirror:
    legacy_model = (
        PACKAGE_ROOT / "pingpong_controller" / "outputs" / "rl_sim"
        / "inverse_compensation"
        / "18_stage4g_strong_contact_dr_inverse_mpc.pkl")
    legacy_xml = (
        PACKAGE_ROOT / "pingpong_controller" / "tools" / "rl_sim"
        / "moz1_pd.xml")
    simcode5_root = WORKSPACE_ROOT / "simcode5" / "pingpong_moz1-main"
    simcode5_model = (
        simcode5_root / "pingpong_controller" / "outputs" / "rl_sim"
        / "goal_d455_gpu1_pure_stable_v5_recovery_20260802"
        / "mjx_curriculum_best.pkl")
    simcode5_xml = (
        simcode5_root / "pingpong_controller" / "tools" / "rl_sim"
        / "moz1_pd.xml")
    required = [legacy_model, legacy_xml, simcode5_model, simcode5_xml]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing mirror inputs: " + ", ".join(missing))
    cfg = SimMirrorConfig(
        model_path=legacy_model,
        policy_xml_path=legacy_xml,
        sim_xml_path=legacy_xml,
        policy_obs_dim=67,
        control_dt=CONTROL_DT_S,
        backend="mujoco",
        enable_video_frames=False,
        use_mujoco_renderer=False,
        enable_compensation=False,
        rl_use_valid_ball_obs_age=False,
        policy_profile=OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
        simcode5_model_path=simcode5_model,
        simcode5_policy_xml_path=simcode5_xml,
        simcode5_sim_xml_path=simcode5_xml,
    )
    return RealBallSimMirror(cfg, logger=QuietLogger())


def mirror_options() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        enable_compensation=False,
        rl_use_valid_ball_obs_age=False,
        policy_observation_scales=(),
        comp_delay_ms=0.0,
        comp_actuator_tau_s=0.074,
        comp_actuator_gain=1.0,
        actuator_use_delay=True,
        actuator_pure_second_order=True,
    )


def _run_branch(
    mirror: RealBallSimMirror,
    arm_q: np.ndarray,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
    valid: np.ndarray,
) -> ReplayArrays:
    mirror.reset(
        arm_q=arm_q,
        ball_pos=position_m[0],
        ball_vel=velocity_m_s[0],
        ball_valid=bool(valid[0]),
        ball_obs_age_s=0.0,
    )
    action: list[np.ndarray] = []
    command: list[np.ndarray] = []
    joint: list[np.ndarray] = []
    racket: list[np.ndarray] = []
    options = mirror_options()
    for ball, velocity, is_valid in zip(position_m, velocity_m_s, valid):
        snap = mirror.step(
            ball, velocity,
            ball_valid=bool(is_valid),
            ball_obs_age_s=0.0,
            options=options,
        )
        action.append(_vec(snap, "policyAction", 7))
        command.append(_vec(snap, "policyCommand", 7))
        joint.append(_vec(snap, "jointPosition", 7))
        racket.append(_vec(snap, "racket", 3))
    return ReplayArrays(
        action=np.asarray(action),
        command=np.asarray(command),
        joint=np.asarray(joint),
        racket=np.asarray(racket),
    )


def _roughness(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    result: list[float] = []
    for index in range(1, len(values)):
        dt = times_s[index] - times_s[index - 1]
        if 1e-5 < dt <= 0.02:
            result.append(float(np.linalg.norm(
                values[index] - values[index - 1]) / dt))
    return np.asarray(result, dtype=np.float64)


def _response_mask(times_s: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(times_s), dtype=bool)
    for anomaly_time in times_s[anomaly]:
        mask |= (
            (times_s >= anomaly_time)
            & (times_s <= anomaly_time + ANOMALY_RESPONSE_LAG_S))
    return mask


def _descent_geometry(
    times_s: np.ndarray,
    truth_position: np.ndarray,
    truth_velocity: np.ndarray,
    racket: np.ndarray,
    contacts_s: list[float],
) -> dict[str, Any]:
    boundaries = [times_s[0], *contacts_s, times_s[-1] + 1e-6]
    best_xy: list[float] = []
    best_3d: list[float] = []
    best_vertical: list[float] = []
    for start, end in zip(boundaries, boundaries[1:]):
        descending = (
            (times_s >= start)
            & (times_s < end)
            & (truth_velocity[:, 2] <= DESCENDING_SPEED_M_S))
        if not np.any(descending):
            continue
        indices = np.flatnonzero(descending)
        vertical = np.abs(
            truth_position[indices, 2] - racket[indices, 2])
        closest = indices[int(np.argmin(vertical))]
        delta = truth_position[closest] - racket[closest]
        best_vertical.append(float(abs(delta[2])))
        best_xy.append(float(np.linalg.norm(delta[:2])))
        best_3d.append(float(np.linalg.norm(delta)))
    return {
        "descentCount": len(best_xy),
        "bestXyErrorM": best_xy,
        "best3dErrorM": best_3d,
        "catchGeometryCount": int(sum(
            xy <= CATCH_GEOMETRY_THRESHOLD_M
            and vertical <= RACKET_PLANE_HALF_WIDTH_M
            for xy, vertical in zip(best_xy, best_vertical))),
    }


def analyze_run(
    run_dir: Path, mirror: RealBallSimMirror,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    ticks = load_ticks(run_dir)
    phase = select_valid_phase(ticks)
    if phase is None:
        return ({
            "run": run_dir.name,
            "eligible": False,
            "reason": "no_valid_policy_phase",
            "capability": recording_capability(run_dir),
        }, {})
    refreshes = fresh_camera_ticks(ticks)
    phase_refreshes = [
        tick for tick in refreshes
        if phase.start_s <= tick.wall_time_s <= phase.end_s]
    if len(phase_refreshes) < 8 or not phase.contacts_s:
        return ({
            "run": run_dir.name,
            "eligible": False,
            "reason": (
                "no_confirmed_juggling_contact"
                if not phase.contacts_s else "insufficient_camera_samples"),
            "capability": recording_capability(run_dir),
        }, {})
    reference = build_reference(refreshes, phase)
    selected = [
        tick for tick in ticks
        if tick.policy_executed
        and phase.start_s <= tick.wall_time_s <= phase.end_s]
    if len(selected) < 20:
        return ({
            "run": run_dir.name,
            "eligible": False,
            "reason": "insufficient_policy_samples",
            "capability": recording_capability(run_dir),
        }, {})
    times = np.asarray([tick.wall_time_s for tick in selected])
    raw_position = np.asarray([tick.observed_ball_m for tick in selected])
    raw_velocity = np.asarray([
        tick.observed_velocity_m_s for tick in selected])
    raw_valid = np.asarray([
        tick.valid
        and np.all(np.isfinite(tick.observed_ball_m))
        and np.all(np.isfinite(tick.observed_velocity_m_s))
        for tick in selected], dtype=bool)
    clean_position, clean_velocity = interpolate_reference(reference, times)
    raw_position = np.where(
        np.isfinite(raw_position), raw_position, clean_position)
    raw_velocity = np.where(
        np.isfinite(raw_velocity), raw_velocity, clean_velocity)
    arm_q = selected[0].arm_q_rad
    if not np.all(np.isfinite(arm_q)):
        raise ValueError("first policy sample has no finite arm pose")

    raw = _run_branch(
        mirror, arm_q, raw_position, raw_velocity, raw_valid)
    clean = _run_branch(
        mirror, arm_q, clean_position, clean_velocity,
        np.ones(len(selected), dtype=bool))

    input_position_delta = np.linalg.norm(
        raw_position - clean_position, axis=1)
    input_velocity_delta = np.linalg.norm(
        raw_velocity - clean_velocity, axis=1)
    anomaly = (
        ~raw_valid
        | (input_position_delta > ANOMALY_POSITION_M)
        | (input_velocity_delta > ANOMALY_VELOCITY_M_S))
    exposed = _response_mask(times, anomaly)
    action_delta = np.linalg.norm(raw.action - clean.action, axis=1)
    command_delta = np.linalg.norm(raw.command - clean.command, axis=1)
    joint_delta = np.linalg.norm(raw.joint - clean.joint, axis=1)
    racket_delta = np.linalg.norm(raw.racket - clean.racket, axis=1)
    raw_geometry = _descent_geometry(
        times, clean_position, clean_velocity, raw.racket, phase.contacts_s)
    clean_geometry = _descent_geometry(
        times, clean_position, clean_velocity, clean.racket, phase.contacts_s)
    raw_action_roughness = _roughness(raw.action, times)
    clean_action_roughness = _roughness(clean.action, times)
    raw_command_roughness = _roughness(raw.command, times)
    clean_command_roughness = _roughness(clean.command, times)

    row = {
        "run": run_dir.name,
        "eligible": True,
        "phase": {
            "startSourceTimeS": round(phase.start_source_time_s, 6),
            "endSourceTimeS": round(phase.end_source_time_s, 6),
            "durationS": round(phase.duration_s, 6),
            "endReason": phase.reason,
            "terminalTailGuardS": TERMINAL_TAIL_GUARD_S,
            "contacts": len(phase.contacts_s),
        },
        "policySamples": len(selected),
        "numericOutlierCameraSamples": int(np.count_nonzero(
            reference.outlier & reference.input_valid)),
        "interpolatedCameraSamples": int(np.count_nonzero(
            reference.interpolated)),
        "invalidRawPolicySamples": int(np.count_nonzero(~raw_valid)),
        "anomalousInputPolicySamples": int(np.count_nonzero(anomaly)),
        "anomalyResponseSamples": int(np.count_nonzero(exposed)),
        "input": {
            "positionDeltaP95Mm": _percentile(
                input_position_delta * 1000.0, 95.0),
            "positionDeltaMaxMm": _percentile(
                input_position_delta * 1000.0, 100.0),
            "velocityDeltaP95Mps": _percentile(
                input_velocity_delta, 95.0),
            "velocityDeltaMaxMps": _percentile(
                input_velocity_delta, 100.0),
        },
        "response": {
            "actionDeltaRms": _rms(action_delta),
            "actionDeltaP95": _percentile(action_delta, 95.0),
            "commandDeltaRmsRad": _rms(command_delta),
            "commandDeltaP95Rad": _percentile(command_delta, 95.0),
            "jointDeltaRmsRad": _rms(joint_delta),
            "jointDeltaP95Rad": _percentile(joint_delta, 95.0),
            "racketDeltaRmsMm": _rms(racket_delta * 1000.0),
            "racketDeltaP95Mm": _percentile(
                racket_delta * 1000.0, 95.0),
            "racketDeltaMaxMm": _percentile(
                racket_delta * 1000.0, 100.0),
            "exposedActionDeltaP95": _percentile(
                action_delta[exposed], 95.0),
            "baselineActionDeltaP95": _percentile(
                action_delta[~exposed], 95.0),
            "exposedRacketDeltaP95Mm": _percentile(
                racket_delta[exposed] * 1000.0, 95.0),
            "baselineRacketDeltaP95Mm": _percentile(
                racket_delta[~exposed] * 1000.0, 95.0),
        },
        "roughness": {
            "rawActionRateP95": _percentile(raw_action_roughness, 95.0),
            "cleanActionRateP95": _percentile(clean_action_roughness, 95.0),
            "rawCommandRateP95RadS": _percentile(
                raw_command_roughness, 95.0),
            "cleanCommandRateP95RadS": _percentile(
                clean_command_roughness, 95.0),
        },
        "rawGeometry": raw_geometry,
        "cleanGeometry": clean_geometry,
    }
    corpus = {
        "inputPositionDeltaMm": (input_position_delta * 1000.0).tolist(),
        "inputVelocityDeltaMps": input_velocity_delta.tolist(),
        "actionDelta": action_delta.tolist(),
        "commandDeltaRad": command_delta.tolist(),
        "jointDeltaRad": joint_delta.tolist(),
        "racketDeltaMm": (racket_delta * 1000.0).tolist(),
        "exposedActionDelta": action_delta[exposed].tolist(),
        "baselineActionDelta": action_delta[~exposed].tolist(),
        "exposedRacketDeltaMm": (
            racket_delta[exposed] * 1000.0).tolist(),
        "baselineRacketDeltaMm": (
            racket_delta[~exposed] * 1000.0).tolist(),
        "rawActionRate": raw_action_roughness.tolist(),
        "cleanActionRate": clean_action_roughness.tolist(),
        "rawCommandRateRadS": raw_command_roughness.tolist(),
        "cleanCommandRateRadS": clean_command_roughness.tolist(),
        "rawBestXyM": raw_geometry["bestXyErrorM"],
        "cleanBestXyM": clean_geometry["bestXyErrorM"],
    }
    return row, corpus


def summarize(
    rows: list[dict[str, Any]], corpus: dict[str, list[float]],
    dates: list[str],
) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible")]
    raw_catches = sum(row["rawGeometry"]["catchGeometryCount"]
                      for row in eligible)
    clean_catches = sum(row["cleanGeometry"]["catchGeometryCount"]
                        for row in eligible)
    descents = sum(row["rawGeometry"]["descentCount"] for row in eligible)
    return {
        "dates": dates,
        "discoveredRunCount": len(rows),
        "eligibleRunCount": len(eligible),
        "excludedRunCount": len(rows) - len(eligible),
        "effectiveDurationS": round(sum(
            row["phase"]["durationS"] for row in eligible), 6),
        "policySamples": sum(row["policySamples"] for row in eligible),
        "numericOutlierCameraSamples": sum(
            row["numericOutlierCameraSamples"] for row in eligible),
        "interpolatedCameraSamples": sum(
            row["interpolatedCameraSamples"] for row in eligible),
        "invalidRawPolicySamples": sum(
            row["invalidRawPolicySamples"] for row in eligible),
        "anomalousInputPolicySamples": sum(
            row["anomalousInputPolicySamples"] for row in eligible),
        "input": {
            "positionDeltaP95Mm": _percentile(
                corpus["inputPositionDeltaMm"], 95.0),
            "positionDeltaMaxMm": _percentile(
                corpus["inputPositionDeltaMm"], 100.0),
            "velocityDeltaP95Mps": _percentile(
                corpus["inputVelocityDeltaMps"], 95.0),
            "velocityDeltaMaxMps": _percentile(
                corpus["inputVelocityDeltaMps"], 100.0),
        },
        "policyResponse": {
            "actionDeltaRms": _rms(corpus["actionDelta"]),
            "actionDeltaP95": _percentile(corpus["actionDelta"], 95.0),
            "commandDeltaRmsRad": _rms(corpus["commandDeltaRad"]),
            "commandDeltaP95Rad": _percentile(
                corpus["commandDeltaRad"], 95.0),
            "jointDeltaRmsRad": _rms(corpus["jointDeltaRad"]),
            "racketDeltaRmsMm": _rms(corpus["racketDeltaMm"]),
            "racketDeltaP95Mm": _percentile(
                corpus["racketDeltaMm"], 95.0),
            "racketDeltaMaxMm": _percentile(
                corpus["racketDeltaMm"], 100.0),
            "exposedActionDeltaP95": _percentile(
                corpus["exposedActionDelta"], 95.0),
            "baselineActionDeltaP95": _percentile(
                corpus["baselineActionDelta"], 95.0),
            "exposedRacketDeltaP95Mm": _percentile(
                corpus["exposedRacketDeltaMm"], 95.0),
            "baselineRacketDeltaP95Mm": _percentile(
                corpus["baselineRacketDeltaMm"], 95.0),
        },
        "roughness": {
            "rawActionRateP95": _percentile(
                corpus["rawActionRate"], 95.0),
            "cleanActionRateP95": _percentile(
                corpus["cleanActionRate"], 95.0),
            "rawCommandRateP95RadS": _percentile(
                corpus["rawCommandRateRadS"], 95.0),
            "cleanCommandRateP95RadS": _percentile(
                corpus["cleanCommandRateRadS"], 95.0),
        },
        "catchGeometry": {
            "descentCount": descents,
            "rawWithin100mmCount": raw_catches,
            "cleanWithin100mmCount": clean_catches,
            "rawBestXyP50Mm": _percentile(
                np.asarray(corpus["rawBestXyM"]) * 1000.0, 50.0),
            "cleanBestXyP50Mm": _percentile(
                np.asarray(corpus["cleanBestXyM"]) * 1000.0, 50.0),
            "rawBestXyP95Mm": _percentile(
                np.asarray(corpus["rawBestXyM"]) * 1000.0, 95.0),
            "cleanBestXyP95Mm": _percentile(
                np.asarray(corpus["cleanBestXyM"]) * 1000.0, 95.0),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    response = summary["policyResponse"]
    roughness = summary["roughness"]
    geometry = summary["catchGeometry"]
    return "\n".join([
        "# Raw vs cleaned real-trajectory policy replay",
        "",
        f"- Dates: {', '.join(summary['dates'])}",
        (f"- Runs: {summary['eligibleRunCount']} eligible / "
         f"{summary['discoveredRunCount']} discovered"),
        f"- Effective juggling duration: {summary['effectiveDurationS']} s",
        f"- Policy ticks per branch: {summary['policySamples']}",
        ("- Camera numeric outliers robustly replaced: "
         f"{summary['numericOutlierCameraSamples']}"),
        ("- Policy ticks with anomalous/missing raw input: "
         f"{summary['anomalousInputPolicySamples']}"),
        "",
        "## Paired policy response",
        "",
        "| metric | raw vs clean |",
        "|---|---:|",
        f"| action difference RMS | {response['actionDeltaRms']} |",
        f"| action difference p95 | {response['actionDeltaP95']} |",
        f"| command difference RMS (rad) | {response['commandDeltaRmsRad']} |",
        f"| racket difference RMS (mm) | {response['racketDeltaRmsMm']} |",
        f"| racket difference p95 (mm) | {response['racketDeltaP95Mm']} |",
        f"| racket difference max (mm) | {response['racketDeltaMaxMm']} |",
        ("| action difference p95, 0-100 ms after anomaly | "
         f"{response['exposedActionDeltaP95']} |"),
        ("| action difference p95, baseline | "
         f"{response['baselineActionDeltaP95']} |"),
        ("| racket difference p95, 0-100 ms after anomaly (mm) | "
         f"{response['exposedRacketDeltaP95Mm']} |"),
        ("| racket difference p95, baseline (mm) | "
         f"{response['baselineRacketDeltaP95Mm']} |"),
        "",
        "## Smoothness and catch geometry",
        "",
        "| metric | raw | clean |",
        "|---|---:|---:|",
        (f"| action-rate p95 | {roughness['rawActionRateP95']} | "
         f"{roughness['cleanActionRateP95']} |"),
        (f"| command-rate p95 (rad/s) | "
         f"{roughness['rawCommandRateP95RadS']} | "
         f"{roughness['cleanCommandRateP95RadS']} |"),
        (f"| best descending XY error p50 (mm) | "
         f"{geometry['rawBestXyP50Mm']} | "
         f"{geometry['cleanBestXyP50Mm']} |"),
        (f"| best descending XY error p95 (mm) | "
         f"{geometry['rawBestXyP95Mm']} | "
         f"{geometry['cleanBestXyP95Mm']} |"),
        (f"| descents within 100 mm | "
         f"{geometry['rawWithin100mmCount']}/{geometry['descentCount']} | "
         f"{geometry['cleanWithin100mmCount']}/{geometry['descentCount']} |"),
        "",
        "The ball is forced and catch geometry is evaluated against the same "
        "clean reference in both branches. Counts are sensitivity indicators, "
        "not predicted real-world success rates.",
    ]) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=(
        PACKAGE_ROOT / "pingpong_controller" / "outputs"
        / "interactive_recordings"))
    parser.add_argument(
        "--dates", nargs="+", default=["20260810", "20260811", "20260812"])
    parser.add_argument("--output", type=Path, default=(
        Path("analysis_results")
        / "policy_trajectory_replay_20260810_12.json"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    runs = discover_runs(args.root, args.dates)
    if args.limit > 0:
        runs = runs[:args.limit]
    mirror = make_mirror()
    rows: list[dict[str, Any]] = []
    corpus: dict[str, list[float]] = {}
    for index, run_dir in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run_dir.name}", flush=True)
        try:
            row, values = analyze_run(run_dir, mirror)
        except Exception as exc:  # noqa: BLE001 - report corpus failures.
            row, values = ({
                "run": run_dir.name,
                "eligible": False,
                "reason": "replay_error",
                "error": f"{type(exc).__name__}: {exc}",
            }, {})
        rows.append(row)
        for key, items in values.items():
            corpus.setdefault(key, []).extend(items)
    required_keys = (
        "inputPositionDeltaMm", "inputVelocityDeltaMps", "actionDelta",
        "commandDeltaRad", "jointDeltaRad", "racketDeltaMm",
        "exposedActionDelta", "baselineActionDelta",
        "exposedRacketDeltaMm", "baselineRacketDeltaMm",
        "rawActionRate", "cleanActionRate", "rawCommandRateRadS",
        "cleanCommandRateRadS", "rawBestXyM", "cleanBestXyM",
    )
    for key in required_keys:
        corpus.setdefault(key, [])
    report = {
        "schemaVersion": 1,
        "scope": {
            "root": str(args.root.resolve()),
            "dates": args.dates,
            "duplicatePolicy": "top-level only; record_new3/new4 excluded",
            "phasePolicy": (
                "policyExecuted interval; cut at detected terminal miss and "
                "always remove final 120 ms"),
            "rawBranch": "exact recorded policy-facing position/velocity/valid",
            "cleanBranch": "acausal robust interpolation/local polynomial",
            "policyProfile": OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
            "comparisonContract": (
                "same initial arm pose, checkpoint, MuJoCo plant and ticks; "
                "catch geometry evaluated against the clean reference"),
        },
        "summary": summarize(rows, corpus, args.dates),
        "runs": rows,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = output.with_suffix(".md")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(output),
        "markdown": str(md_path),
        "runs": report["summary"]["discoveredRunCount"],
        "eligible": report["summary"]["eligibleRunCount"],
        "policySamples": report["summary"]["policySamples"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
