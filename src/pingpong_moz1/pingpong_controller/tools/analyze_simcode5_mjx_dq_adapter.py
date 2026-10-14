#!/usr/bin/env python3
"""Replay recorded ball observations through simcode5 MJX and assess dq adapters.

This is an offline analysis tool.  It does not change the controller or the
training environment.  It joins real and recorded mirror samples by the
lossless ``sourceSeq`` envelope, replays the exact policy-facing ball input
through the checkpoint-authoritative simcode5 MJX mirror, then evaluates
causal real-side joint-velocity estimators against ``MJX policyObs[7:14]``.

The calibrated adapter is deliberately simple enough to deploy safely:

    dq_policy[j] = gain[j] * delay(EMA(dq_joint_state[j]))

Gains are constrained to pass through zero.  Delay and EMA time constant are
selected per joint with leave-one-recording-out cross-validation, so the
reported score is not an in-sample curve fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.rl_policy import (  # noqa: E402
    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
    RIGHT_ARM_JOINTS,
)
from pingpong_controller.sim_mirror import (  # noqa: E402
    RealBallMjxJaxSimMirror,
    SimMirrorConfig,
)


OBS_DIM = 67
ARM_DIM = 7
DEFAULT_DT_S = 0.005
EMA_TAU_GRID_S = (0.0, 0.005, 0.010, 0.020, 0.030, 0.040)
DELAY_GRID_STEPS = tuple(range(7))
FIT_WINDOW_S = 0.50
METRIC_WINDOWS_S = (0.10, 0.25, 0.50)


class AnalysisError(RuntimeError):
    """Raised when the recording cannot support a trustworthy replay."""


class _SilentLogger:
    def info(self, _message: str) -> None:
        pass

    def warn(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass


@dataclass(frozen=True)
class Frame:
    run: str
    source_seq: int
    source_time_s: float
    episode: int
    stream_seq: int
    ball: np.ndarray
    velocity: np.ndarray
    valid: bool
    age_s: float
    real_obs: np.ndarray
    cpu_obs: np.ndarray
    real_action: np.ndarray
    published_command: np.ndarray
    policy_nominal_command: np.ndarray
    compensated_command: np.ndarray
    command_published: bool
    scales: np.ndarray
    joint_stamp_s: float


@dataclass(frozen=True)
class Segment:
    run: str
    episode: int
    frames: tuple[Frame, ...]


@dataclass(frozen=True)
class ReplayData:
    frames: tuple[Frame, ...]
    run_id: np.ndarray
    segment_id: np.ndarray
    segment_position: np.ndarray
    elapsed_s: np.ndarray
    real_obs: np.ndarray
    cpu_obs: np.ndarray
    mjx_obs: np.ndarray
    real_action: np.ndarray
    mjx_action: np.ndarray
    joint_stamp_s: np.ndarray


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"malformed JSON at {path}:{line_number}") from exc
            if isinstance(row, dict):
                yield row


def _vector(value: Any, width: int, label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(width)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be a length-{width} vector") from exc
    if not np.all(np.isfinite(result)):
        raise AnalysisError(f"{label} contains non-finite values")
    return result


def _source_seq(row: dict[str, Any], *, sim: bool) -> int | None:
    raw: Any
    if sim:
        sync = row.get("sync")
        raw = sync.get("sourceSeq") if isinstance(sync, dict) else None
    else:
        raw = row.get("sourceSeq", row.get("tick"))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _policy_executed(row: dict[str, Any]) -> bool:
    return bool(row.get("policyExecuted", str(row.get("mode", "")) == "policy"))


def _float32_equal(left: Any, right: Any, width: int) -> bool:
    try:
        a = np.asarray(left, dtype=np.float32).reshape(width)
        b = np.asarray(right, dtype=np.float32).reshape(width)
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(a, b))


def _compact_sim_row(row: dict[str, Any]) -> dict[str, Any] | None:
    sync = row.get("sync")
    if not isinstance(sync, dict):
        return None
    if str(sync.get("policyProfile", "")) != OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67:
        return None
    if not bool(sync.get("streamContinuous", False)):
        return None
    if int(sync.get("resultDrops", 0) or 0) != 0:
        return None
    try:
        return {
            "episode": int(sync.get("episode", 0) or 0),
            "stream_seq": int(sync.get("streamSeq", 0) or 0),
            "source_time_s": float(sync.get("sourceTimeS", 0.0) or 0.0),
            "ball": _vector(sync.get("inputBall"), 3, "sync.inputBall"),
            "velocity": _vector(sync.get("inputVelocity"), 3, "sync.inputVelocity"),
            "valid": bool(sync.get("inputValid", False)),
            "age_s": float(sync.get("inputAgeS", 0.0) or 0.0),
            "obs": _vector(row.get("policyObs"), OBS_DIM, "sim.policyObs"),
            "policy_ball": _vector(
                row.get("policyBallObservation"), 6, "sim.policyBallObservation"),
        }
    except AnalysisError:
        return None


def _load_run(run_dir: Path) -> tuple[list[Segment], dict[str, Any]]:
    sim_path = run_dir / "sim_control_samples.jsonl"
    real_path = run_dir / "control_samples.jsonl"
    if not sim_path.is_file() or not real_path.is_file():
        return [], {"run": run_dir.name, "reason": "missing streams"}

    sim_by_seq: dict[int, dict[str, Any]] = {}
    duplicate_sim = 0
    for row in _read_jsonl(sim_path):
        seq = _source_seq(row, sim=True)
        compact = _compact_sim_row(row)
        if seq is None or compact is None:
            continue
        if seq in sim_by_seq:
            duplicate_sim += 1
            continue
        sim_by_seq[seq] = compact

    frames: list[Frame] = []
    executed_count = 0
    unmatched_count = 0
    duplicate_real = 0
    seen_real: set[int] = set()
    for row in _read_jsonl(real_path):
        if not _policy_executed(row) or str(row.get("mode", "")) != "policy":
            continue
        runtime = row.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        if str(runtime.get("activePolicyProfile", "")) != OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67:
            continue
        executed_count += 1
        seq = _source_seq(row, sim=False)
        if seq is None:
            continue
        if seq in seen_real:
            duplicate_real += 1
            continue
        seen_real.add(seq)
        sim = sim_by_seq.get(seq)
        if sim is None:
            unmatched_count += 1
            continue
        policy_input = row.get("policyInput")
        if not isinstance(policy_input, dict):
            unmatched_count += 1
            continue
        envelope_ok = (
            _float32_equal(policy_input.get("ball"), sim["ball"], 3)
            and _float32_equal(policy_input.get("velocity"), sim["velocity"], 3)
            and bool(policy_input.get("valid", False)) == bool(sim["valid"])
            and abs(float(policy_input.get("age", 0.0) or 0.0) - sim["age_s"]) <= 1e-9
            and _float32_equal(
                row.get("policyBallObservation"), sim["policy_ball"], 6)
        )
        if not envelope_ok:
            unmatched_count += 1
            continue
        try:
            real_obs = _vector(row.get("policyObs"), OBS_DIM, "real.policyObs")
            real_action = _vector(row.get("policyAction"), ARM_DIM, "real.policyAction")
            command = row.get("command")
            if not isinstance(command, dict):
                raise AnalysisError("real.command is missing")
            published_command = _vector(
                command.get("publishedSafePosition", command.get("position")),
                ARM_DIM,
                "real.command.publishedSafePosition",
            )
            policy_nominal_command = _vector(
                command.get("policyNominalPosition", command.get("position")),
                ARM_DIM,
                "real.command.policyNominalPosition",
            )
            compensated_command = _vector(
                command.get("compensatedReferencePosition", command.get("position")),
                ARM_DIM,
                "real.command.compensatedReferencePosition",
            )
        except AnalysisError:
            unmatched_count += 1
            continue
        scales_raw = runtime.get("policyObservationScales")
        try:
            scales = _vector(scales_raw, OBS_DIM, "policyObservationScales")
        except AnalysisError:
            scales = np.ones(OBS_DIM, dtype=np.float64)
        ros = row.get("ros")
        ros = ros if isinstance(ros, dict) else {}
        try:
            joint_stamp_s = float(ros.get("jointSampleStampS"))
        except (TypeError, ValueError):
            joint_stamp_s = math.nan
        frames.append(Frame(
            run=run_dir.name,
            source_seq=seq,
            source_time_s=float(sim["source_time_s"]),
            episode=int(sim["episode"]),
            stream_seq=int(sim["stream_seq"]),
            ball=np.asarray(sim["ball"], dtype=np.float64),
            velocity=np.asarray(sim["velocity"], dtype=np.float64),
            valid=bool(sim["valid"]),
            age_s=float(max(0.0, sim["age_s"])),
            real_obs=real_obs,
            cpu_obs=np.asarray(sim["obs"], dtype=np.float64),
            real_action=real_action,
            published_command=published_command,
            policy_nominal_command=policy_nominal_command,
            compensated_command=compensated_command,
            command_published=bool(command.get("published", False)),
            scales=scales,
            joint_stamp_s=joint_stamp_s,
        ))

    frames.sort(key=lambda item: item.source_seq)
    segments: list[Segment] = []
    current: list[Frame] = []
    for frame in frames:
        if current:
            previous = current[-1]
            continuous = (
                frame.episode == previous.episode
                and frame.stream_seq == previous.stream_seq + 1
                and frame.source_seq == previous.source_seq + 1
                and np.array_equal(frame.scales.astype(np.float32), previous.scales.astype(np.float32))
            )
            if not continuous:
                segments.append(Segment(
                    run=run_dir.name,
                    episode=current[0].episode,
                    frames=tuple(current),
                ))
                current = []
        current.append(frame)
    if current:
        segments.append(Segment(
            run=run_dir.name,
            episode=current[0].episode,
            frames=tuple(current),
        ))
    segments = [segment for segment in segments if len(segment.frames) >= 2]
    audit = {
        "run": run_dir.name,
        "executedPolicyRows": executed_count,
        "alignedRows": sum(len(segment.frames) for segment in segments),
        "unmatchedRows": unmatched_count,
        "simRows": len(sim_by_seq),
        "segments": len(segments),
        "duplicateReal": duplicate_real,
        "duplicateSim": duplicate_sim,
    }
    return segments, audit


def _load_recording_root(root: Path) -> tuple[list[Segment], list[dict[str, Any]]]:
    segments: list[Segment] = []
    audit: list[dict[str, Any]] = []
    for sim_path in sorted(root.glob("probe_recording_*/sim_control_samples.jsonl")):
        run_segments, run_audit = _load_run(sim_path.parent)
        segments.extend(run_segments)
        audit.append(run_audit)
    if not segments:
        raise AnalysisError(f"no aligned simcode5 policy segments found under {root}")
    return segments, audit


def _metadata_paths(root: Path) -> tuple[Path, Path, float]:
    for metadata_path in sorted(root.glob("probe_recording_*/metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        config = metadata.get("config")
        config = config if isinstance(config, dict) else {}
        sim = config.get("simMirror")
        sim = sim if isinstance(sim, dict) else {}
        if str(sim.get("policyProfile", "")) != OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67:
            continue
        model_raw = sim.get("modelPath")
        xml_raw = sim.get("xmlPath", sim.get("policyXmlPath"))
        if model_raw and xml_raw:
            rate_hz = float(config.get("controlRateHz", 200.0) or 200.0)
            return Path(model_raw).expanduser().resolve(), Path(xml_raw).expanduser().resolve(), 1.0 / rate_hz
    raise AnalysisError("simcode5 checkpoint/XML paths are missing from recording metadata")


def _make_mirror(model: Path, xml: Path, dt_s: float) -> RealBallMjxJaxSimMirror:
    if not model.is_file() or not xml.is_file():
        raise AnalysisError(f"missing checkpoint or XML: {model}, {xml}")
    cfg = SimMirrorConfig(
        model_path=model,
        policy_xml_path=xml,
        sim_xml_path=xml,
        policy_obs_dim=OBS_DIM,
        control_dt=dt_s,
        backend="mjx",
        enable_video_frames=False,
        use_mujoco_renderer=False,
        enable_compensation=False,
        policy_profile=OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
        simcode5_model_path=model,
        simcode5_policy_xml_path=xml,
        simcode5_sim_xml_path=xml,
    )
    return RealBallMjxJaxSimMirror(cfg, logger=_SilentLogger())


def _pad(values: np.ndarray, width: int) -> np.ndarray:
    if values.shape[0] >= width:
        return values
    return np.concatenate([
        values,
        np.repeat(values[-1:], width - values.shape[0], axis=0),
    ], axis=0)


def _replay(
    mirror: RealBallMjxJaxSimMirror,
    segments: list[Segment],
    *,
    batch_size: int,
) -> ReplayData:
    flat_frames: list[Frame] = []
    mjx_obs_rows: list[np.ndarray] = []
    mjx_action_rows: list[np.ndarray] = []
    run_names = sorted({segment.run for segment in segments})
    run_ids = {name: index for index, name in enumerate(run_names)}
    run_id_rows: list[int] = []
    segment_id_rows: list[int] = []
    segment_position_rows: list[int] = []
    elapsed_rows: list[float] = []

    mirror.prewarm(batch_size)
    for segment_index, segment in enumerate(segments):
        frames = segment.frames
        first = frames[0]
        mirror.reset(
            arm_q=first.real_obs[:ARM_DIM].astype(np.float32),
            ball_pos=first.ball.astype(np.float32),
            ball_vel=first.velocity.astype(np.float32),
            ball_valid=first.valid,
            ball_obs_age_s=first.age_s,
        )
        options = SimpleNamespace(
            policy_observation_scales=tuple(first.scales.astype(np.float32)))
        for start in range(0, len(frames), batch_size):
            chunk = frames[start:start + batch_size]
            count = len(chunk)
            ball = _pad(np.asarray([frame.ball for frame in chunk], dtype=np.float32), batch_size)
            velocity = _pad(
                np.asarray([frame.velocity for frame in chunk], dtype=np.float32), batch_size)
            valid = _pad(
                np.asarray([frame.valid for frame in chunk], dtype=bool)[:, None], batch_size
            ).reshape(-1)
            age = _pad(
                np.asarray([frame.age_s for frame in chunk], dtype=np.float32)[:, None], batch_size
            ).reshape(-1)
            snap = mirror.step_sequence(
                ball,
                velocity,
                ball_valid_seq=valid,
                ball_obs_age_seq=age,
                options=options,
            )
            samples = snap.get("_batchSamples", [])
            if len(samples) < count:
                raise AnalysisError("MJX mirror returned an incomplete batch")
            for offset, (frame, sample) in enumerate(zip(chunk, samples[:count])):
                flat_frames.append(frame)
                mjx_obs_rows.append(_vector(
                    sample.get("policyObs"), OBS_DIM, "MJX policyObs"))
                mjx_action_rows.append(_vector(
                    sample.get("policyAction"), ARM_DIM, "MJX policyAction"))
                run_id_rows.append(run_ids[segment.run])
                segment_id_rows.append(segment_index)
                segment_position_rows.append(start + offset)
                elapsed_rows.append((start + offset) * float(mirror.dt))
        print(
            f"replayed {segment.run} episode={segment.episode} "
            f"ticks={len(frames)}")

    return ReplayData(
        frames=tuple(flat_frames),
        run_id=np.asarray(run_id_rows, dtype=np.int32),
        segment_id=np.asarray(segment_id_rows, dtype=np.int32),
        segment_position=np.asarray(segment_position_rows, dtype=np.int32),
        elapsed_s=np.asarray(elapsed_rows, dtype=np.float64),
        real_obs=np.asarray([frame.real_obs for frame in flat_frames], dtype=np.float64),
        cpu_obs=np.asarray([frame.cpu_obs for frame in flat_frames], dtype=np.float64),
        mjx_obs=np.asarray(mjx_obs_rows, dtype=np.float64),
        real_action=np.asarray([frame.real_action for frame in flat_frames], dtype=np.float64),
        mjx_action=np.asarray(mjx_action_rows, dtype=np.float64),
        joint_stamp_s=np.asarray([frame.joint_stamp_s for frame in flat_frames], dtype=np.float64),
    )


def _causal_q_polynomial(data: ReplayData, window: int) -> np.ndarray:
    q = data.real_obs[:, :ARM_DIM]
    raw = data.real_obs[:, ARM_DIM:2 * ARM_DIM]
    result = raw.copy()
    for segment_id in np.unique(data.segment_id):
        indices = np.flatnonzero(data.segment_id == segment_id)
        history_t: list[float] = []
        history_q: list[np.ndarray] = []
        held = raw[indices[0]].copy()
        last_stamp: float | None = None
        for index in indices:
            stamp = float(data.joint_stamp_s[index])
            if not math.isfinite(stamp):
                result[index] = held
                continue
            if last_stamp is None or stamp > last_stamp + 1e-9:
                history_t.append(stamp)
                history_q.append(q[index].copy())
                history_t = history_t[-window:]
                history_q = history_q[-window:]
                last_stamp = stamp
                if len(history_t) >= 2:
                    relative = np.asarray(history_t, dtype=np.float64) - history_t[-1]
                    values = np.asarray(history_q, dtype=np.float64)
                    degree = min(2, len(history_t) - 1)
                    design = np.stack(
                        [relative ** power for power in range(degree + 1)], axis=1)
                    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
                    held = coefficients[1].copy()
            result[index] = held
    return result


def _segment_ema(values: np.ndarray, segment_id: np.ndarray, tau_s: float, dt_s: float) -> np.ndarray:
    if tau_s <= 0.0:
        return values.copy()
    alpha = 1.0 - math.exp(-dt_s / tau_s)
    output = np.empty_like(values)
    for sid in np.unique(segment_id):
        indices = np.flatnonzero(segment_id == sid)
        state = values[indices[0]].copy()
        for index in indices:
            state = state + alpha * (values[index] - state)
            output[index] = state
    return output


def _segment_delay(values: np.ndarray, segment_id: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return values.copy()
    output = np.empty_like(values)
    for sid in np.unique(segment_id):
        indices = np.flatnonzero(segment_id == sid)
        for position, index in enumerate(indices):
            source = indices[max(0, position - steps)]
            output[index] = values[source]
    return output


def _fit_zero_gain(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip(np.dot(x, y) / denominator, 0.25, 2.0))


def _fit_calibrated_adapter(
    data: ReplayData,
    raw_dq: np.ndarray,
    target: np.ndarray,
    *,
    search_tau: bool,
    cross_validate: bool,
    dt_s: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    taus = EMA_TAU_GRID_S if search_tau else (0.0,)
    delays = DELAY_GRID_STEPS if search_tau else (0,)
    filtered = {
        tau: _segment_ema(raw_dq, data.segment_id, tau, dt_s)
        for tau in taus
    }
    candidates = {
        (tau, delay): _segment_delay(values, data.segment_id, delay)
        for tau, values in filtered.items()
        for delay in delays
    }
    prediction = np.zeros_like(target)
    parameter_accumulator: list[list[dict[str, float]]] = [
        [] for _ in range(ARM_DIM)]
    folds = np.unique(data.run_id) if cross_validate else np.asarray([-1])
    for held_run in folds:
        test = np.ones(data.run_id.shape, dtype=bool) if held_run < 0 else data.run_id == held_run
        train = np.ones(data.run_id.shape, dtype=bool) if held_run < 0 else data.run_id != held_run
        train &= data.elapsed_s <= FIT_WINDOW_S + 1e-12
        if not np.any(train) or not np.any(test):
            continue
        for joint in range(ARM_DIM):
            best: tuple[float, float, int, float] | None = None
            for (tau, delay), values in candidates.items():
                valid = train & (data.segment_position >= delay)
                gain = _fit_zero_gain(values[valid, joint], target[valid, joint])
                error = gain * values[valid, joint] - target[valid, joint]
                mse = float(np.mean(np.square(error)))
                choice = (mse, tau, delay, gain)
                if best is None or choice < best:
                    best = choice
            if best is None:
                raise AnalysisError("unable to fit dq adapter")
            _mse, tau, delay, gain = best
            prediction[test, joint] = gain * candidates[(tau, delay)][test, joint]
            parameter_accumulator[joint].append({
                "tauS": float(tau),
                "delaySteps": int(delay),
                "delayMs": float(delay * dt_s * 1000.0),
                "gain": float(gain),
            })
    parameters: list[dict[str, Any]] = []
    for joint, values in enumerate(parameter_accumulator):
        if not values:
            continue
        parameters.append({
            "joint": RIGHT_ARM_JOINTS[joint],
            "foldCount": len(values),
            "tauSMedian": float(np.median([value["tauS"] for value in values])),
            "delayStepsMedian": int(round(float(np.median([
                value["delaySteps"] for value in values])))),
            "gainMedian": float(np.median([value["gain"] for value in values])),
            "foldParameters": values if cross_validate else [],
        })
    return prediction, parameters


def _actor_batch(obs: np.ndarray, weights: dict[str, Any]) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    x = np.tanh(x @ np.asarray(weights["l1_w"], dtype=np.float32)
                + np.asarray(weights["l1_b"], dtype=np.float32))
    x = np.tanh(x @ np.asarray(weights["l2_w"], dtype=np.float32)
                + np.asarray(weights["l2_b"], dtype=np.float32))
    action = x @ np.asarray(weights["out_w"], dtype=np.float32) \
        + np.asarray(weights["out_b"], dtype=np.float32)
    return np.clip(action, -1.0, 1.0).astype(np.float64)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _candidate_metrics(
    name: str,
    dq: np.ndarray,
    data: ReplayData,
    weights: dict[str, Any],
) -> dict[str, Any]:
    target = data.mjx_obs[:, 7:14]
    adapted_obs = data.real_obs.copy()
    adapted_obs[:, 7:14] = dq
    action = _actor_batch(adapted_obs, weights)
    result: dict[str, Any] = {"name": name, "windows": {}}
    windows: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(data.elapsed_s.shape, dtype=bool))]
    windows.extend([
        (f"first_{window:.2f}s", data.elapsed_s <= window + 1e-12)
        for window in METRIC_WINDOWS_S
    ])
    for label, mask in windows:
        error = dq[mask] - target[mask]
        action_error = action[mask] - data.mjx_action[mask]
        action_change = action[mask] - _actor_batch(data.real_obs[mask], weights)
        per_joint = np.sqrt(np.mean(np.square(error), axis=0))
        correlations: list[float | None] = []
        for joint in range(ARM_DIM):
            left = dq[mask, joint]
            right = target[mask, joint]
            if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
                correlations.append(None)
            else:
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
        result["windows"][label] = {
            "samples": int(np.count_nonzero(mask)),
            "dqRmsRadS": _rms(error),
            "dqPerJointRmsRadS": per_joint.tolist(),
            "dqCorrelationPerJoint": correlations,
            "actionToMjxRms": _rms(action_error),
            "actionChangeFromRawRealRms": _rms(action_change),
        }
    return result


def _write_replay_jsonl(path: Path, data: ReplayData) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, frame in enumerate(data.frames):
            row = {
                "run": frame.run,
                "sourceSeq": frame.source_seq,
                "episode": frame.episode,
                "segment": int(data.segment_id[index]),
                "segmentPosition": int(data.segment_position[index]),
                "elapsedS": float(data.elapsed_s[index]),
                "ball": frame.ball.tolist(),
                "velocity": frame.velocity.tolist(),
                "valid": frame.valid,
                "ageS": frame.age_s,
                "realPolicyObs": data.real_obs[index].tolist(),
                "cpuMirrorPolicyObs": data.cpu_obs[index].tolist(),
                "mjxMirrorPolicyObs": data.mjx_obs[index].tolist(),
                "realPolicyAction": data.real_action[index].tolist(),
                "mjxMirrorPolicyAction": data.mjx_action[index].tolist(),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _plot_summary(
    path: Path,
    metrics: list[dict[str, Any]],
) -> None:
    labels = [item["name"] for item in metrics]
    early = [item["windows"]["first_0.50s"]["dqRmsRadS"] for item in metrics]
    overall = [item["windows"]["all"]["dqRmsRadS"] for item in metrics]
    actions = [item["windows"]["first_0.50s"]["actionToMjxRms"] for item in metrics]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    width = 0.38
    axes[0].bar(x - width / 2, early, width, label="first 0.50 s")
    axes[0].bar(x + width / 2, overall, width, label="all")
    axes[0].set_ylabel("dq RMS to MJX (rad/s)")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].bar(x, actions, 0.6, color="#59a14f")
    axes[1].set_ylabel("actor action RMS to MJX\n(first 0.50 s)")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Execution-side dq adapter candidates (held-recording-out scores)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_representative(
    path: Path,
    data: ReplayData,
    candidates: dict[str, np.ndarray],
) -> None:
    counts = np.bincount(data.segment_id)
    representative = int(np.argmax(counts))
    indices = np.flatnonzero(data.segment_id == representative)
    elapsed = data.elapsed_s[indices]
    fig, axes = plt.subplots(ARM_DIM, 1, figsize=(15, 18), sharex=True)
    colors = {
        "real_joint_state": "#1f77b4",
        "q_causal_poly5": "#59a14f",
        "calibrated_gain_delay_ema_cv": "#f28e2b",
    }
    for joint, axis in enumerate(axes):
        axis.plot(
            elapsed,
            data.mjx_obs[indices, 7 + joint],
            color="#d62728",
            linewidth=1.5,
            label="MJX target",
        )
        for name in ("real_joint_state", "q_causal_poly5", "calibrated_gain_delay_ema_cv"):
            axis.plot(
                elapsed,
                candidates[name][indices, joint],
                color=colors[name],
                linewidth=1.0,
                alpha=0.9,
                label=name,
            )
        axis.set_ylabel(f"{RIGHT_ARM_JOINTS[joint]}\nrad/s")
        axis.grid(True, alpha=0.22)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("time since mirror reset (s)")
    first = data.frames[indices[0]]
    fig.suptitle(
        f"Representative causal dq estimates — {first.run}, episode {first.episode}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_mjx_comparison(path: Path, data: ReplayData) -> None:
    counts = np.bincount(data.segment_id)
    representative = int(np.argmax(counts))
    indices = np.flatnonzero(data.segment_id == representative)
    elapsed = data.elapsed_s[indices]
    fig, axes = plt.subplots(ARM_DIM, 2, figsize=(16, 20), sharex=True)
    for joint in range(ARM_DIM):
        for label, obs, color in (
            ("real", data.real_obs, "#1f77b4"),
            ("CPU mirror", data.cpu_obs, "#9467bd"),
            ("MJX mirror", data.mjx_obs, "#d62728"),
        ):
            axes[joint, 0].plot(elapsed, obs[indices, joint], color=color, linewidth=1.0, label=label)
            axes[joint, 1].plot(elapsed, obs[indices, 7 + joint], color=color, linewidth=1.0, label=label)
        axes[joint, 0].set_ylabel(f"{RIGHT_ARM_JOINTS[joint]}\nrad")
        axes[joint, 1].set_ylabel("rad/s")
        axes[joint, 0].grid(True, alpha=0.22)
        axes[joint, 1].grid(True, alpha=0.22)
    axes[0, 0].set_title("joint angle")
    axes[0, 1].set_title("joint velocity")
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("time since mirror reset (s)")
    axes[-1, 1].set_xlabel("time since mirror reset (s)")
    fig.suptitle("Recorded real vs CPU mirror vs exact simcode5 MJX mirror")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze(root: Path, output_dir: Path, batch_size: int) -> dict[str, Any]:
    segments, input_audit = _load_recording_root(root)
    model, xml, dt_s = _metadata_paths(root)
    mirror = _make_mirror(model, xml, dt_s)
    try:
        data = _replay(mirror, segments, batch_size=batch_size)
        weights = dict(mirror.policy._pi_weights or {})
        env_contract = {
            "obsDim": int(mirror.env.obs_dim),
            "dtS": float(mirror.dt),
            "rightArmPdProfile": str(getattr(mirror.env_cfg, "right_arm_pd_profile", "")),
            "actuatorCommandModel": str(getattr(mirror.env_cfg, "actuator_cmd_model", "")),
            "domainRandomization": bool(getattr(mirror.env_cfg, "domain_randomization", False)),
        }
    finally:
        close = getattr(mirror, "close", None)
        if callable(close):
            close()

    raw = data.real_obs[:, 7:14]
    target = data.mjx_obs[:, 7:14]
    q_poly3 = _causal_q_polynomial(data, 3)
    q_poly5 = _causal_q_polynomial(data, 5)
    q_poly9 = _causal_q_polynomial(data, 9)
    gain_cv, gain_cv_params = _fit_calibrated_adapter(
        data, raw, target, search_tau=False, cross_validate=True, dt_s=dt_s)
    calibrated_cv, calibrated_cv_params = _fit_calibrated_adapter(
        data, raw, target, search_tau=True, cross_validate=True, dt_s=dt_s)
    _calibrated_full, deployment_params = _fit_calibrated_adapter(
        data, raw, target, search_tau=True, cross_validate=False, dt_s=dt_s)

    candidates = {
        "real_joint_state": raw,
        "q_causal_poly3": q_poly3,
        "q_causal_poly5": q_poly5,
        "q_causal_poly9": q_poly9,
        "calibrated_gain_only_cv": gain_cv,
        "calibrated_gain_delay_ema_cv": calibrated_cv,
        "online_mjx_shadow_upper_bound": target,
    }
    metrics = [
        _candidate_metrics(name, values, data, weights)
        for name, values in candidates.items()
    ]
    actor_real = _actor_batch(data.real_obs, weights)
    actor_mjx = _actor_batch(data.mjx_obs, weights)
    replay_audit = {
        "sampleCount": len(data.frames),
        "segmentCount": int(np.unique(data.segment_id).size),
        "runCount": int(np.unique(data.run_id).size),
        "realActorReproductionRms": _rms(actor_real - data.real_action),
        "realActorReproductionMax": float(np.max(np.abs(actor_real - data.real_action))),
        "mjxActorReproductionRms": _rms(actor_mjx - data.mjx_action),
        "mjxActorReproductionMax": float(np.max(np.abs(actor_mjx - data.mjx_action))),
        "realToCpuDqRmsRadS": _rms(raw - data.cpu_obs[:, 7:14]),
        "realToMjxDqRmsRadS": _rms(raw - target),
        "cpuToMjxDqRmsRadS": _rms(data.cpu_obs[:, 7:14] - target),
        "realToMjxQRmsRad": _rms(data.real_obs[:, :7] - data.mjx_obs[:, :7]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_replay_jsonl(output_dir / "mjx_replay_samples.jsonl", data)
    _plot_summary(output_dir / "dq_adapter_summary.png", metrics)
    _plot_representative(output_dir / "dq_adapter_representative_trace.png", data, candidates)
    _plot_mjx_comparison(output_dir / "real_cpu_mjx_joint_comparison.png", data)

    with (output_dir / "dq_candidate_summary.csv").open(
        "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate", "dq_rms_first_0_10s", "dq_rms_first_0_25s",
            "dq_rms_first_0_50s", "dq_rms_all", "action_rms_first_0_50s",
            "action_change_first_0_50s",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow({
                "candidate": item["name"],
                "dq_rms_first_0_10s": item["windows"]["first_0.10s"]["dqRmsRadS"],
                "dq_rms_first_0_25s": item["windows"]["first_0.25s"]["dqRmsRadS"],
                "dq_rms_first_0_50s": item["windows"]["first_0.50s"]["dqRmsRadS"],
                "dq_rms_all": item["windows"]["all"]["dqRmsRadS"],
                "action_rms_first_0_50s": item["windows"]["first_0.50s"]["actionToMjxRms"],
                "action_change_first_0_50s": item["windows"]["first_0.50s"]["actionChangeFromRawRealRms"],
            })

    report = {
        "analysis": "simcode5_mjx_execution_side_dq_adapter",
        "recordingRoot": str(root),
        "checkpoint": str(model),
        "xml": str(xml),
        "envContract": env_contract,
        "inputAudit": input_audit,
        "replayAudit": replay_audit,
        "fitContract": {
            "causal": True,
            "crossValidation": "leave-one-recording-out",
            "fitWindowS": FIT_WINDOW_S,
            "emaTauGridS": list(EMA_TAU_GRID_S),
            "delayGridSteps": list(DELAY_GRID_STEPS),
            "gainConstraint": [0.25, 2.0],
            "zeroIntercept": True,
        },
        "candidateMetrics": metrics,
        "gainOnlyCrossValidationParameters": gain_cv_params,
        "calibratedCrossValidationParameters": calibrated_cv_params,
        "deploymentParametersFitAllRecordings": deployment_params,
        "warnings": [
            "The MJX target is an independently evolving closed-loop state, not a sensor ground truth for the real robot.",
            "A dq-only adapter makes q and dq internally inconsistent after the real and MJX trajectories diverge.",
            "Raw joint velocity must remain the source for safety limits and diagnostics.",
        ],
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: <recording_root>/simcode5_mjx_dq_adapter",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    root = args.recording_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else root / "simcode5_mjx_dq_adapter")
    report = analyze(root, output_dir, max(1, int(args.batch_size)))
    audit = report["replayAudit"]
    print(
        f"wrote {output_dir} | runs={audit['runCount']} "
        f"segments={audit['segmentCount']} samples={audit['sampleCount']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
