#!/usr/bin/env python3
"""Offline common-command ideal-PD shadow analysis for probe recordings.

The tool never changes the real-time controller.  It joins the recorded real
and independent-simulation streams by ``sourceSeq``, drives a fresh launch19
XML-PD plant with the *recorded real nominal policy command*, and compares the
three resulting 67-D observation streams:

* real vs common-command ideal-PD shadow,
* common-command shadow vs independent ideal-PD actor, and
* real vs independent ideal-PD actor.

Before reporting gaps, an independent-command replay must reproduce the
recorded independent mirror.  This fail-closed self-check protects the report
from command/observation timing drift in this offline implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller import (  # noqa: E402
    mjx_rl_policy as mjx_rl_policy_module,
    rl_policy as rl_policy_module,
    sim_mirror as sim_mirror_module,
    simcode2_xml as simcode2_xml_module,
)
from pingpong_controller.rl_policy import (  # noqa: E402
    OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
)
from pingpong_controller.sim_mirror import (  # noqa: E402
    RealBallSimMirror,
    SimMirrorConfig,
)


ALGORITHM_NAME = "common_command_ideal_pd_shadow"
ALGORITHM_VERSION = 1
PRIMARY_METRIC_VARIANT = "actualRecordedWindows"
OBS_DIM = 67
ARM_DIM = 7
TRAINING_DT_S = 0.005
NOMINAL_CONTINUITY_ATOL_RAD = 5e-5

OBS_GROUPS: dict[str, tuple[int, int, str]] = {
    "q": (0, 7, "rad"),
    "dq": (7, 14, "rad/s"),
    "base_q": (14, 17, "mixed"),
    "base_dq": (17, 20, "mixed/s"),
    "ball_position": (20, 23, "m"),
    "ball_velocity": (23, 26, "m/s"),
    "racket_position": (26, 29, "m"),
    "racket_velocity": (29, 32, "m/s"),
    "ball_racket_relative": (32, 35, "m"),
    "previous_action": (35, 42, "unitless"),
    "command_error": (42, 49, "rad"),
    "ball_age_normalized": (49, 50, "unitless"),
    "actuator_delay_normalized": (50, 51, "unitless"),
    "command_velocity": (51, 58, "rad/s"),
    "active_command_error": (58, 65, "rad"),
    "contact_phase": (65, 67, "s"),
}

ACTUATOR_GROUPS = (
    "q",
    "dq",
    "racket_position",
    "racket_velocity",
    "ball_racket_relative",
    "command_error",
    "active_command_error",
    "contact_phase",
)

WINDOWS_S = (0.1, 0.25, 0.5)
SELF_CHECK_LIMITS = {
    "qRmsRad": 2e-5,
    "qMaxRad": 2e-4,
    "dqRmsRadS": 2e-3,
    "dqMaxRadS": 2e-2,
    "racketPositionRmsM": 2e-5,
    "racketPositionMaxM": 2e-4,
    "racketVelocityRmsMps": 2e-3,
    "racketVelocityMaxMps": 2e-2,
}


class AnalysisError(RuntimeError):
    """Raised when the recording cannot support a reliable shadow replay."""


class _SilentLogger:
    def info(self, _message: str) -> None:
        pass

    def warn(self, _message: str) -> None:
        pass

    def error(self, _message: str) -> None:
        pass


@dataclass(frozen=True)
class Inputs:
    recording_dir: Path
    metadata_path: Path
    real_path: Path
    sim_path: Path
    checkpoint_path: Path
    xml_path: Path
    training_trace_path: Path
    control_dt_s: float


@dataclass(frozen=True)
class Frame:
    source_seq: int
    real: dict[str, Any]
    independent: dict[str, Any]
    episode: int
    stream_seq: int


@dataclass(frozen=True)
class ReplayPlan:
    frames: tuple[Frame, ...]
    reset_boundaries: frozenset[int]
    asymmetric_tail_start: int | None
    mask_reasons: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TrainingScale:
    mean: np.ndarray
    std: np.ndarray
    rms: np.ndarray
    scale: np.ndarray
    rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "sha256": _sha256(resolved),
    }


def _git_record() -> dict[str, Any]:
    result: dict[str, Any] = {"repository": str(REPO_ROOT)}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        result.update({
            "commit": commit,
            "dirty": bool(status.strip()),
        })
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    return result


def _runtime_provenance() -> dict[str, Any]:
    modules = {
        "mjx_rl_policy": mjx_rl_policy_module,
        "rl_policy": rl_policy_module,
        "sim_mirror": sim_mirror_module,
        "simcode2_xml": simcode2_xml_module,
    }
    dependencies: dict[str, Any] = {}
    for name, module in modules.items():
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            raise AnalysisError(f"cannot locate runtime dependency {name}")
        dependencies[name] = _file_record(Path(raw_path))
    mujoco_version = getattr(sim_mirror_module.mj, "__version__", None)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mujoco": mujoco_version,
        "dependencies": dependencies,
        "git": _git_record(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AnalysisError(
                        f"invalid JSONL {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise AnalysisError(
                        f"expected JSON object at {path}:{line_number}")
                rows.append(value)
    except OSError as exc:
        raise AnalysisError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AnalysisError(f"{label} not found: {resolved}")
    return resolved


def _resolve_inputs(
    recording_dir: Path,
    *,
    checkpoint_override: Path | None,
    xml_override: Path | None,
    trace_override: Path | None,
) -> Inputs:
    run_dir = recording_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise AnalysisError(f"recording directory not found: {run_dir}")
    metadata_path = _require_file(run_dir / "metadata.json", "metadata")
    real_path = _require_file(
        run_dir / "control_samples.jsonl", "real control stream")
    sim_path = _require_file(
        run_dir / "sim_control_samples.jsonl", "simulation control stream")
    metadata = _read_json(metadata_path)
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise AnalysisError("metadata.config is missing")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise AnalysisError("metadata.config.policy is missing")
    sim_cfg = config.get("simMirror")
    sim_cfg = sim_cfg if isinstance(sim_cfg, dict) else {}

    profile = str(policy.get("activeProfile", sim_cfg.get("policyProfile", "")))
    if profile != OBSERVATION_PROFILE_SIMCODE2_IDEALPD67:
        raise AnalysisError(
            "common-command analysis requires simcode2_idealpd67; "
            f"recording profile is {profile!r}")
    try:
        obs_dim = int(policy.get("observationDim", 0))
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid policy observationDim") from exc
    if obs_dim != OBS_DIM:
        raise AnalysisError(f"expected 67-D policy, got {obs_dim}-D")

    model_raw = checkpoint_override or policy.get("modelPath")
    xml_raw = xml_override or policy.get(
        "policyXmlPath", sim_cfg.get("policyXmlPath"))
    if model_raw is None or xml_raw is None:
        raise AnalysisError("checkpoint or policy XML missing from metadata")
    checkpoint = _require_file(Path(model_raw), "launch19 checkpoint")
    xml = _require_file(Path(xml_raw), "launch19 policy XML")
    trace = _require_file(
        trace_override or checkpoint.parent / "obs_trace.csv",
        "launch19 training observation trace",
    )

    try:
        control_rate_hz = float(config.get("controlRateHz", 200.0))
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid controlRateHz") from exc
    if not math.isfinite(control_rate_hz) or control_rate_hz <= 0.0:
        raise AnalysisError(f"invalid controlRateHz: {control_rate_hz}")
    dt_s = 1.0 / control_rate_hz
    if abs(dt_s - TRAINING_DT_S) > 1e-9:
        raise AnalysisError(
            "launch19 training rvel contract requires exactly 0.005 s; "
            f"recording dt is {dt_s:.9f} s")

    return Inputs(
        recording_dir=run_dir,
        metadata_path=metadata_path,
        real_path=real_path,
        sim_path=sim_path,
        checkpoint_path=checkpoint,
        xml_path=xml,
        training_trace_path=trace,
        control_dt_s=dt_s,
    )


def _source_seq(row: dict[str, Any]) -> int:
    raw = row.get("sourceSeq", row.get("tick"))
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"real sample has invalid sourceSeq: {raw!r}") from exc


def _sim_source_seq(row: dict[str, Any]) -> int:
    sync = row.get("sync")
    raw = sync.get("sourceSeq") if isinstance(sync, dict) else None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"sim sample has invalid sync.sourceSeq: {raw!r}") from exc


def _vector(value: Any, width: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(width)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be a length-{width} vector") from exc
    if not np.all(np.isfinite(array)):
        raise AnalysisError(f"{label} contains non-finite values")
    return array


def _obs(row: dict[str, Any], label: str) -> np.ndarray:
    return _vector(row.get("policyObs"), OBS_DIM, f"{label}.policyObs")


def _real_command(row: dict[str, Any]) -> np.ndarray:
    command = row.get("policyCommand")
    if not isinstance(command, dict):
        raise AnalysisError("real sample policyCommand is missing")
    return _vector(command.get("position"), ARM_DIM, "real policy command")


def _sim_command(row: dict[str, Any]) -> np.ndarray:
    raw = row.get("policyCommand")
    if raw is None:
        command = row.get("command")
        raw = command.get("position") if isinstance(command, dict) else None
    return _vector(raw, ARM_DIM, "independent policy command")


def _policy_executed(row: dict[str, Any]) -> bool:
    if "policyExecuted" in row:
        return bool(row.get("policyExecuted"))
    return str(row.get("mode", "")) == "policy"


def _float32_equal(left: Any, right: Any, width: int) -> bool:
    try:
        a = np.asarray(left, dtype=np.float32).reshape(width)
        b = np.asarray(right, dtype=np.float32).reshape(width)
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(a, b))


def _validate_input_envelope(real: dict[str, Any], sim: dict[str, Any]) -> None:
    policy_input = real.get("policyInput")
    sync = sim.get("sync")
    if not isinstance(policy_input, dict) or not isinstance(sync, dict):
        raise AnalysisError("policyInput or sim sync envelope is missing")
    checks = {
        "ball": _float32_equal(
            policy_input.get("ball"), sync.get("inputBall"), 3),
        "velocity": _float32_equal(
            policy_input.get("velocity"), sync.get("inputVelocity"), 3),
        "valid": bool(policy_input.get("valid", False))
        == bool(sync.get("inputValid", False)),
        "ballObservationSeq": int(
            policy_input.get("ballObservationSeq", 0) or 0)
        == int(sync.get("ballObservationSeq", 0) or 0),
        "age": abs(
            float(policy_input.get("age", 0.0) or 0.0)
            - float(sync.get("inputAgeS", 0.0) or 0.0)
        ) <= 1e-9,
        "policyBallObservation": _float32_equal(
            real.get("policyBallObservation"),
            sim.get("policyBallObservation"),
            6,
        ),
    }
    runtime = real.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    checks["profile"] = str(runtime.get("activePolicyProfile", "")) == str(
        sync.get("policyProfile", ""))
    checks["generation"] = int(runtime.get("policyGeneration", 0) or 0) == int(
        sync.get("policyGeneration", 0) or 0)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AnalysisError(
            "real/sim policy input mismatch for "
            f"sourceSeq={sync.get('sourceSeq')}: {', '.join(failed)}")
    if not bool(sync.get("streamContinuous", False)):
        raise AnalysisError(
            f"sim stream discontinuity at sourceSeq={sync.get('sourceSeq')}")
    if int(sync.get("resultDrops", 0) or 0) != 0:
        raise AnalysisError(
            f"sim result drop at sourceSeq={sync.get('sourceSeq')}")
    if not bool(sync.get("runtimeConfigMatch", True)):
        raise AnalysisError(
            f"sim runtime config mismatch at sourceSeq={sync.get('sourceSeq')}")


def _align_frames(
    real_rows: Iterable[dict[str, Any]],
    sim_rows: Iterable[dict[str, Any]],
) -> tuple[Frame, ...]:
    real_by_seq: dict[int, dict[str, Any]] = {}
    for row in real_rows:
        if not _policy_executed(row):
            continue
        seq = _source_seq(row)
        if seq in real_by_seq:
            raise AnalysisError(f"duplicate real sourceSeq={seq}")
        real_by_seq[seq] = row
    sim_by_seq: dict[int, dict[str, Any]] = {}
    for row in sim_rows:
        seq = _sim_source_seq(row)
        if seq in sim_by_seq:
            raise AnalysisError(f"duplicate sim sourceSeq={seq}")
        sim_by_seq[seq] = row
    if not real_by_seq:
        raise AnalysisError("recording contains no executed policy samples")
    missing = sorted(set(real_by_seq) - set(sim_by_seq))
    if missing:
        raise AnalysisError(
            f"missing {len(missing)} aligned sim samples; first={missing[:10]}")

    frames: list[Frame] = []
    for seq in sorted(real_by_seq):
        real = real_by_seq[seq]
        sim = sim_by_seq[seq]
        _validate_input_envelope(real, sim)
        _obs(real, f"real[{seq}]")
        _obs(sim, f"sim[{seq}]")
        _real_command(real)
        _sim_command(sim)
        sync = sim["sync"]
        frames.append(Frame(
            source_seq=seq,
            real=real,
            independent=sim,
            episode=int(sync.get("episode", 0) or 0),
            stream_seq=int(sync.get("streamSeq", 0) or 0),
        ))
    for previous, current in zip(frames, frames[1:]):
        same_episode = current.episode == previous.episode
        if same_episode and current.stream_seq != previous.stream_seq + 1:
            raise AnalysisError(
                "sim streamSeq gap inside episode at sourceSeq="
                f"{current.source_seq}")
        if same_episode and current.source_seq != previous.source_seq + 1:
            raise AnalysisError(
                "real sourceSeq gap inside sim episode at sourceSeq="
                f"{current.source_seq}")
    return tuple(frames)


def _command_before(obs: np.ndarray) -> np.ndarray:
    return obs[0:7] + obs[42:49]


def _build_replay_plan(frames: tuple[Frame, ...]) -> ReplayPlan:
    first_command = frames[0].real.get("command")
    first_compensation = (
        first_command.get("compensation")
        if isinstance(first_command, dict) else None)
    if (
        not isinstance(first_compensation, dict)
        or not bool(first_compensation.get("active", False))
    ):
        raise AnalysisError(
            "compensation is not active on the first aligned policy tick; "
            "this is not an active real-control cohort")
    boundaries = {0}
    masks: list[dict[str, Any]] = [{
        "startIndex": 0,
        "startSourceSeq": frames[0].source_seq,
        "endIndexExclusive": 1,
        "reason": "initial_reset_boundary",
    }]
    tail_start: int | None = None
    for index in range(1, len(frames)):
        previous = frames[index - 1]
        current = frames[index]
        current_command = current.real.get("command")
        current_compensation = (
            current_command.get("compensation")
            if isinstance(current_command, dict) else None)
        if (
            isinstance(current_compensation, dict)
            and not bool(current_compensation.get("active", False))
        ):
            tail_start = index
            masks.append({
                "startIndex": index,
                "startSourceSeq": current.source_seq,
                "endIndexExclusive": len(frames),
                "endSourceSeq": frames[-1].source_seq,
                "reason": "real_compensation_inactive_tail",
            })
            break
        episode_reset = current.episode != previous.episode
        real_discontinuity = not np.allclose(
            _command_before(_obs(current.real, "real")),
            _real_command(previous.real),
            rtol=0.0,
            atol=NOMINAL_CONTINUITY_ATOL_RAD,
        )
        sim_discontinuity = not np.allclose(
            _command_before(_obs(current.independent, "independent")),
            _sim_command(previous.independent),
            rtol=0.0,
            atol=NOMINAL_CONTINUITY_ATOL_RAD,
        )
        if episode_reset or (real_discontinuity and sim_discontinuity):
            boundaries.add(index)
            masks.append({
                "startIndex": index,
                "startSourceSeq": current.source_seq,
                "endIndexExclusive": index + 1,
                "reason": (
                    "sim_episode_reset" if episode_reset
                    else "symmetric_nominal_history_reset"),
            })
            continue
        if real_discontinuity != sim_discontinuity:
            tail_start = index
            masks.append({
                "startIndex": index,
                "startSourceSeq": current.source_seq,
                "endIndexExclusive": len(frames),
                "endSourceSeq": frames[-1].source_seq,
                "reason": (
                    "asymmetric_real_nominal_reset_tail"
                    if real_discontinuity
                    else "asymmetric_independent_nominal_reset_tail"),
            })
            break
    return ReplayPlan(
        frames=frames,
        reset_boundaries=frozenset(boundaries),
        asymmetric_tail_start=tail_start,
        mask_reasons=tuple(masks),
    )


COMPENSATION_PARAMETER_FIELDS = (
    "mode",
    "delayMs",
    "tauS",
    "gain",
    "mpcBeta",
    "mpcDelayScale",
    "mpcTauScale",
    "mpcHorizonSteps",
    "mpcTrackingWeight",
    "mpcNominalWeight",
    "mpcDeltaWeight",
    "mpcMaxDeltaRad",
    "bufferLen",
)

COMPENSATION_STATUS_FIELDS = (
    "enabled",
    "active",
    "available",
    "error",
)


def _compensation_at(frame: Frame) -> dict[str, Any]:
    command = frame.real.get("command")
    compensation = (
        command.get("compensation") if isinstance(command, dict) else None)
    if not isinstance(compensation, dict):
        raise AnalysisError(
            "actual per-tick command.compensation is missing at sourceSeq="
            f"{frame.source_seq}; metadata defaults are intentionally not used")
    missing = [
        name for name in (
            COMPENSATION_PARAMETER_FIELDS + COMPENSATION_STATUS_FIELDS)
        if name not in compensation
    ]
    if missing:
        raise AnalysisError(
            "per-tick compensation parameters missing at sourceSeq="
            f"{frame.source_seq}: {', '.join(missing)}")
    return compensation


def _compensation_parameters(compensation: dict[str, Any]) -> dict[str, Any]:
    return {
        name: compensation[name]
        for name in COMPENSATION_PARAMETER_FIELDS
    }


def _signature(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _segments(
    frames: tuple[Frame, ...],
    values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    start = 0
    for index in range(1, len(values) + 1):
        changed = (
            index == len(values)
            or frames[index].source_seq != frames[index - 1].source_seq + 1
            or frames[index].episode != frames[index - 1].episode
            or _signature(values[index]) != _signature(values[start])
        )
        if not changed:
            continue
        output.append({
            "startIndex": start,
            "endIndexExclusive": index,
            "startSourceSeq": frames[start].source_seq,
            "endSourceSeq": frames[index - 1].source_seq,
            "sampleCount": index - start,
            "value": values[start],
        })
        start = index
    return output


def _parameter_cohorts(
    frames: tuple[Frame, ...],
    parameters: list[dict[str, Any]],
) -> dict[str, Any]:
    parameter_segments = _segments(frames, parameters)
    cohorts_by_signature: dict[str, dict[str, Any]] = {}
    for segment in parameter_segments:
        value = segment["value"]
        signature = _signature(value)
        cohort = cohorts_by_signature.setdefault(signature, {
            "parameters": value,
            "sampleCount": 0,
            "ranges": [],
        })
        cohort["sampleCount"] += int(segment["sampleCount"])
        cohort["ranges"].append({
            key: segment[key]
            for key in (
                "startIndex",
                "endIndexExclusive",
                "startSourceSeq",
                "endSourceSeq",
                "sampleCount",
            )
        })
    return {
        "tickCount": len(frames),
        "cohortCount": len(cohorts_by_signature),
        "mixedCohort": len(cohorts_by_signature) > 1,
        "cohorts": list(cohorts_by_signature.values()),
        "segments": parameter_segments,
        "changeCount": max(0, len(parameter_segments) - 1),
    }


def _compensation_audit(
    frames: tuple[Frame, ...],
    plan: ReplayPlan,
) -> dict[str, Any]:
    per_tick = [_compensation_at(frame) for frame in frames]
    parameters = [_compensation_parameters(item) for item in per_tick]
    aligned = _parameter_cohorts(frames, parameters)
    metric_limit = (
        len(frames)
        if plan.asymmetric_tail_start is None
        else plan.asymmetric_tail_start
    )
    metric_indices = [
        index for index in range(metric_limit)
        if index not in plan.reset_boundaries
    ]
    metric_frames = tuple(frames[index] for index in metric_indices)
    metric_parameters = [parameters[index] for index in metric_indices]
    metric = _parameter_cohorts(metric_frames, metric_parameters)
    status_values = [{
        "enabled": bool(item.get("enabled", False)),
        "active": bool(item.get("active", False)),
        "available": bool(item.get("available", False)),
        "error": item.get("error"),
    } for item in per_tick]
    status_segments = _segments(frames, status_values)
    return {
        "source": "control_samples.jsonl[].command.compensation",
        "metadataDefaultsUsed": False,
        "tickCount": len(frames),
        "alignedTuning": aligned,
        "metricUnmaskedTuning": metric,
        "runtimeStatusSegments": status_segments,
        "runtimeStatusChangeCount": max(0, len(status_segments) - 1),
    }


def _load_training_scale(path: Path) -> TrainingScale:
    values: list[list[float]] = []
    columns = [f"obs/{index:03d}" for index in range(OBS_DIM)]
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            missing = [name for name in columns if name not in fieldnames]
            if missing:
                raise AnalysisError(
                    "training trace lacks observation columns: "
                    + ", ".join(missing[:5]))
            for line_number, row in enumerate(reader, start=2):
                try:
                    item = [float(row[name]) for name in columns]
                except (TypeError, ValueError) as exc:
                    raise AnalysisError(
                        f"invalid observation at {path}:{line_number}") from exc
                values.append(item)
    except OSError as exc:
        raise AnalysisError(f"cannot read training trace {path}: {exc}") from exc
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != OBS_DIM or not array.shape[0]:
        raise AnalysisError(f"empty or malformed training trace: {path}")
    if not np.all(np.isfinite(array)):
        raise AnalysisError(f"training trace contains non-finite observations: {path}")
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    rms = np.sqrt(np.mean(np.square(array), axis=0))
    scale = np.maximum(np.maximum(std, 0.05 * rms), 1e-4)
    return TrainingScale(
        mean=mean,
        std=std,
        rms=rms,
        scale=scale,
        rows=int(array.shape[0]),
    )


class IdealPdCommandShadow:
    """Observe-before-command wrapper around the launch19 CPU XML-PD plant."""

    def __init__(self, inputs: Inputs) -> None:
        cfg = SimMirrorConfig(
            model_path=inputs.checkpoint_path,
            policy_xml_path=inputs.xml_path,
            sim_xml_path=inputs.xml_path,
            policy_obs_dim=OBS_DIM,
            control_dt=inputs.control_dt_s,
            enable_video_frames=False,
            use_mujoco_renderer=False,
            enable_compensation=False,
            policy_profile=OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
            simcode2_model_path=inputs.checkpoint_path,
            simcode2_policy_xml_path=inputs.xml_path,
            simcode2_sim_xml_path=inputs.xml_path,
        )
        self.mirror = RealBallSimMirror(cfg, logger=_SilentLogger())
        self.dt = float(self.mirror.dt)
        self.previous_racket_position: np.ndarray | None = None

    @property
    def patched_xml_path(self) -> Path:
        return Path(self.mirror.patched_xml_path)

    @property
    def policy(self):
        return self.mirror.policy

    def reset(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ball_position: np.ndarray,
        ball_velocity: np.ndarray,
    ) -> None:
        self.mirror.reset(
            np.asarray(q, dtype=np.float32),
            ball_pos=np.asarray(ball_position, dtype=np.float32),
            ball_vel=np.asarray(ball_velocity, dtype=np.float32),
            ball_valid=True,
            ball_obs_age_s=0.0,
        )
        self.mirror.data.qpos[self.mirror.arm_qadr] = np.asarray(
            q, dtype=np.float64)
        self.mirror.data.qvel[self.mirror.arm_vadr] = np.asarray(
            dq, dtype=np.float64)
        self.mirror.data.ctrl[self.mirror.arm_aids] = np.asarray(
            q, dtype=np.float64)
        self._forward()
        self.previous_racket_position = self._racket_position(q)

    def _forward(self) -> None:
        # Keep the MuJoCo dependency encapsulated in the existing mirror.
        from pingpong_controller import sim_mirror  # noqa: PLC0415

        sim_mirror.mj.mj_forward(self.mirror.model, self.mirror.data)

    def _racket_position(self, q: np.ndarray) -> np.ndarray:
        position, ok = self.policy._compute_racket_pose_base(
            np.asarray(q, dtype=np.float32))
        result = np.asarray(position, dtype=np.float64).reshape(3)
        if not ok or not np.all(np.isfinite(result)):
            raise AnalysisError("launch19 FK failed during shadow replay")
        return result

    def observe(
        self,
        ball_position: np.ndarray,
        ball_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.mirror._set_ball_state(ball_position, ball_velocity)
        self._forward()
        q = np.asarray(
            self.mirror.data.qpos[self.mirror.arm_qadr],
            dtype=np.float64,
        ).copy()
        dq = np.asarray(
            self.mirror.data.qvel[self.mirror.arm_vadr],
            dtype=np.float64,
        ).copy()
        rpos = self._racket_position(q)
        if self.previous_racket_position is None:
            rvel = np.zeros(3, dtype=np.float64)
        else:
            rvel = (rpos - self.previous_racket_position) / self.dt
        self.previous_racket_position = rpos.copy()
        return q, dq, rpos, rvel

    def advance(self, nominal_command: np.ndarray) -> None:
        from pingpong_controller import sim_mirror  # noqa: PLC0415

        self.mirror.data.ctrl[:] = self.mirror.default_ctrl
        self.mirror.data.ctrl[self.mirror.arm_aids] = np.asarray(
            nominal_command, dtype=np.float64)
        for _ in range(self.mirror.substeps):
            sim_mirror.mj.mj_step(self.mirror.model, self.mirror.data)

    def close(self) -> None:
        self.mirror.close()


def _age_seconds(frame: Frame, stream: str) -> float:
    if stream == "real":
        envelope = frame.real.get("policyInput")
        raw = envelope.get("age", 0.0) if isinstance(envelope, dict) else 0.0
    else:
        sync = frame.independent.get("sync")
        raw = sync.get("inputAgeS", 0.0) if isinstance(sync, dict) else 0.0
    try:
        age = float(raw or 0.0)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid ball observation age: {raw!r}") from exc
    if not math.isfinite(age) or age < 0.0:
        raise AnalysisError(f"invalid ball observation age: {age}")
    return age


def _contact_phase(
    obs: np.ndarray,
    *,
    age_s: float,
    policy: Any,
) -> tuple[float, float]:
    gravity_z = -9.81
    rel_z_now = (
        float(obs[34])
        + float(obs[25]) * age_s
        + 0.5 * gravity_z * age_s * age_s
    )
    ball_vz_now = float(obs[25]) + gravity_z * age_s
    contact = float(policy._estimate_contact_time(
        rel_z_now,
        ball_vz_now,
        float(obs[31]),
        ball_obs_age_s=age_s,
        max_time=policy.max_contact_time_s,
        contact_height_offset=policy.contact_height_offset_m,
        gravity=abs(gravity_z),
        lost_ball_timeout_ms=policy.contact_lost_ball_timeout_s * 1000.0,
    ))
    return contact, contact - float(policy.delay_conditioning_tau_s)


def _canonical_recorded_obs(
    row: dict[str, Any],
    *,
    label: str,
    previous_rpos: np.ndarray | None,
    dt_s: float,
    age_s: float,
    policy: Any,
) -> tuple[np.ndarray, np.ndarray]:
    obs = _obs(row, label).copy()
    rpos = obs[26:29].copy()
    obs[29:32] = (
        np.zeros(3, dtype=np.float64)
        if previous_rpos is None else (rpos - previous_rpos) / dt_s
    )
    obs[32:35] = obs[20:23] - obs[26:29]
    obs[65:67] = _contact_phase(obs, age_s=age_s, policy=policy)
    return obs, rpos


def _common_obs(
    real_obs: np.ndarray,
    *,
    q: np.ndarray,
    dq: np.ndarray,
    rpos: np.ndarray,
    rvel: np.ndarray,
    age_s: float,
    policy: Any,
) -> np.ndarray:
    nominal_before = real_obs[0:7] + real_obs[42:49]
    active_before = real_obs[0:7] + real_obs[58:65]
    common = real_obs.copy()
    common[0:7] = q
    common[7:14] = dq
    common[14:20] = 0.0
    common[26:29] = rpos
    common[29:32] = rvel
    common[32:35] = common[20:23] - rpos
    common[42:49] = nominal_before - q
    common[58:65] = active_before - q
    common[65:67] = _contact_phase(common, age_s=age_s, policy=policy)
    return common


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _self_check(
    replayed: np.ndarray,
    independent: np.ndarray,
) -> dict[str, Any]:
    delta = replayed - independent
    result = {
        "qRmsRad": _rms(delta[:, 0:7]),
        "qMaxRad": _max_abs(delta[:, 0:7]),
        "dqRmsRadS": _rms(delta[:, 7:14]),
        "dqMaxRadS": _max_abs(delta[:, 7:14]),
        "racketPositionRmsM": _rms(delta[:, 26:29]),
        "racketPositionMaxM": _max_abs(delta[:, 26:29]),
        "racketVelocityRmsMps": _rms(delta[:, 29:32]),
        "racketVelocityMaxMps": _max_abs(delta[:, 29:32]),
    }
    failures = [
        key for key, limit in SELF_CHECK_LIMITS.items()
        if result[key] > limit
    ]
    result.update({
        "passed": not failures,
        "limits": dict(SELF_CHECK_LIMITS),
        "failedMetrics": failures,
        "meaning": (
            "Recorded independent nominal commands replayed through the same "
            "observe-before-command launch19 XML-PD plant must reproduce the "
            "recorded independent q/dq/racket state after strict-5-ms "
            "canonicalization. Historical raw policyObs contract differences "
            "are reported separately by canonicalizationAudit."
        ),
    })
    return result


def _selected_indices() -> np.ndarray:
    indices: list[int] = []
    for name in ACTUATOR_GROUPS:
        start, stop, _unit = OBS_GROUPS[name]
        indices.extend(range(start, stop))
    return np.asarray(indices, dtype=np.int64)


def _gap_metrics(
    left: np.ndarray,
    right: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    delta = np.asarray(left - right, dtype=np.float64)
    normalized = delta / scale[None, :]
    selected = _selected_indices()
    groups: dict[str, Any] = {}
    for name, (start, stop, unit) in OBS_GROUPS.items():
        groups[name] = {
            "dimensions": list(range(start, stop)),
            "unit": unit,
            "rawRms": _rms(delta[:, start:stop]),
            "normalizedRms": _rms(normalized[:, start:stop]),
        }
    actuator_group_values = np.asarray([
        groups[name]["normalizedRms"] for name in ACTUATOR_GROUPS
    ], dtype=np.float64)
    raw_per_dimension = np.sqrt(np.mean(np.square(delta), axis=0))
    normalized_per_dimension = np.sqrt(
        np.mean(np.square(normalized), axis=0))
    return {
        "sampleCount": int(delta.shape[0]),
        "full67": {
            "rawRms": _rms(delta),
            "normalizedRms": _rms(normalized),
        },
        "selectedActuator39": {
            "dimensions": selected.tolist(),
            "rawRms": _rms(delta[:, selected]),
            "normalizedRms": _rms(normalized[:, selected]),
        },
        "equalWeightActuatorGroupNormalizedRms": _rms(
            actuator_group_values),
        "groups": groups,
        "perDimensionRawRms": raw_per_dimension.tolist(),
        "perDimensionNormalizedRms": normalized_per_dimension.tolist(),
    }


def _window_report(
    source_seq: np.ndarray,
    comparisons: dict[str, tuple[np.ndarray, np.ndarray]],
    scale: np.ndarray,
    dt_s: float,
) -> dict[str, Any]:
    if not comparisons:
        raise AnalysisError("metric comparison set is empty")
    first_left, _first_right = next(iter(comparisons.values()))
    count = int(first_left.shape[0])
    if count <= 0:
        raise AnalysisError("no unmasked policy samples remain for metrics")
    for name, (left, right) in comparisons.items():
        if left.shape != (count, OBS_DIM) or right.shape != (count, OBS_DIM):
            raise AnalysisError(
                f"comparison {name} does not contain aligned 67-D arrays")
    windows: list[tuple[str, int]] = []
    for duration in WINDOWS_S:
        requested = int(round(duration / dt_s))
        windows.append((f"{duration:g}s", min(requested, count)))
    windows.append(("full", count))
    output: dict[str, Any] = {}
    for name, sample_count in windows:
        report = {
            "sampleCount": sample_count,
            "durationS": sample_count * dt_s,
            "startSourceSeq": int(source_seq[0]),
            "endSourceSeq": int(source_seq[sample_count - 1]),
            "comparisons": {},
        }
        for comparison, (left, right) in comparisons.items():
            report["comparisons"][comparison] = _gap_metrics(
                left[:sample_count], right[:sample_count], scale)
        output[name] = report
    return output


def _canonicalization_summary(windows: dict[str, Any]) -> dict[str, Any]:
    full = windows["full"]["comparisons"]
    thresholds = {
        "racketVelocityRawRmsMps": 0.005,
        "contactPhaseRawRmsS": 0.001,
    }
    streams: dict[str, Any] = {}
    material = False
    for name, comparison in full.items():
        rvel = float(comparison["groups"]["racket_velocity"]["rawRms"])
        contact = float(comparison["groups"]["contact_phase"]["rawRms"])
        mismatch = bool(
            rvel > thresholds["racketVelocityRawRmsMps"]
            or contact > thresholds["contactPhaseRawRmsS"]
        )
        material = material or mismatch
        streams[name] = {
            "materialMismatch": mismatch,
            "racketVelocityRawRmsMps": rvel,
            "contactPhaseRawRmsS": contact,
            "full67NormalizedRms": float(
                comparison["full67"]["normalizedRms"]),
        }
    return {
        "materialMismatch": material,
        "thresholds": thresholds,
        "streams": streams,
        "canonicalizedDimensions": list(range(29, 35)) + [65, 66],
        "meaning": (
            "Historical actualRecorded observations are preserved separately. "
            "The canonical variant replaces racket velocity with the strict "
            "5 ms training secant, then recomputes relative position and "
            "contact phase from the canonical fields."
        ),
    }


def _validate_initial_contract(first: Frame) -> dict[str, Any]:
    real = _obs(first.real, "real initial")
    independent = _obs(first.independent, "independent initial")
    delta_q = real[0:7] - independent[0:7]
    delta_dq = real[7:14] - independent[7:14]
    delta_ball = real[20:26] - independent[20:26]
    report = {
        "realQ": real[0:7].tolist(),
        "independentQ": independent[0:7].tolist(),
        "qMaxAbsDifferenceRad": _max_abs(delta_q),
        "realDq": real[7:14].tolist(),
        "independentDq": independent[7:14].tolist(),
        "dqMaxAbsDifferenceRadS": _max_abs(delta_dq),
        "ballStateMaxAbsDifference": _max_abs(delta_ball),
    }
    if (
        report["qMaxAbsDifferenceRad"] > 2e-5
        or report["dqMaxAbsDifferenceRadS"] > 2e-4
        or report["ballStateMaxAbsDifference"] > 2e-6
    ):
        raise AnalysisError(
            "real/common and independent streams do not share the same "
            f"initial q/dq/ball state: {report}")
    return report


def _run_replay(
    inputs: Inputs,
    plan: ReplayPlan,
) -> tuple[dict[str, Any], Path]:
    common_plant = IdealPdCommandShadow(inputs)
    check_plant = IdealPdCommandShadow(inputs)
    patched_xml = common_plant.patched_xml_path
    try:
        initial_contract = _validate_initial_contract(plan.frames[0])
        actual_real_rows: list[np.ndarray] = []
        actual_independent_rows: list[np.ndarray] = []
        real_rows: list[np.ndarray] = []
        common_rows: list[np.ndarray] = []
        independent_rows: list[np.ndarray] = []
        check_rows: list[np.ndarray] = []
        valid_source_seq: list[int] = []
        previous_real_rpos: np.ndarray | None = None
        previous_independent_rpos: np.ndarray | None = None
        replay_limit = (
            len(plan.frames)
            if plan.asymmetric_tail_start is None
            else plan.asymmetric_tail_start
        )
        for index, frame in enumerate(plan.frames[:replay_limit]):
            boundary = index in plan.reset_boundaries
            raw_real = _obs(frame.real, "real")
            raw_independent = _obs(frame.independent, "independent")
            real_age = _age_seconds(frame, "real")
            independent_age = _age_seconds(frame, "independent")
            ball_position = raw_real[20:23]
            ball_velocity = raw_real[23:26]
            if boundary:
                common_plant.reset(
                    raw_real[0:7],
                    raw_real[7:14],
                    ball_position,
                    ball_velocity,
                )
                check_plant.reset(
                    raw_independent[0:7],
                    raw_independent[7:14],
                    raw_independent[20:23],
                    raw_independent[23:26],
                )
                previous_real_rpos = None
                previous_independent_rpos = None

            real_obs, previous_real_rpos = _canonical_recorded_obs(
                frame.real,
                label="real",
                previous_rpos=previous_real_rpos,
                dt_s=inputs.control_dt_s,
                age_s=real_age,
                policy=common_plant.policy,
            )
            independent_obs, previous_independent_rpos = (
                _canonical_recorded_obs(
                    frame.independent,
                    label="independent",
                    previous_rpos=previous_independent_rpos,
                    dt_s=inputs.control_dt_s,
                    age_s=independent_age,
                    policy=common_plant.policy,
                )
            )
            common_q, common_dq, common_rpos, common_rvel = (
                common_plant.observe(ball_position, ball_velocity))
            common_obs = _common_obs(
                real_obs,
                q=common_q,
                dq=common_dq,
                rpos=common_rpos,
                rvel=common_rvel,
                age_s=real_age,
                policy=common_plant.policy,
            )
            check_q, check_dq, check_rpos, check_rvel = check_plant.observe(
                raw_independent[20:23], raw_independent[23:26])
            check_obs = independent_obs.copy()
            check_obs[0:7] = check_q
            check_obs[7:14] = check_dq
            check_obs[26:29] = check_rpos
            check_obs[29:32] = check_rvel
            check_obs[32:35] = check_obs[20:23] - check_rpos
            nominal_before = (
                independent_obs[0:7] + independent_obs[42:49])
            active_before = (
                independent_obs[0:7] + independent_obs[58:65])
            check_obs[42:49] = nominal_before - check_q
            check_obs[58:65] = active_before - check_q
            check_obs[65:67] = _contact_phase(
                check_obs,
                age_s=independent_age,
                policy=check_plant.policy,
            )

            if not boundary:
                valid_source_seq.append(frame.source_seq)
                actual_real_rows.append(raw_real.copy())
                actual_independent_rows.append(raw_independent.copy())
                real_rows.append(real_obs)
                common_rows.append(common_obs)
                independent_rows.append(independent_obs)
                check_rows.append(check_obs)

            common_plant.advance(_real_command(frame.real))
            check_plant.advance(_sim_command(frame.independent))

        actual_real_array = np.asarray(actual_real_rows, dtype=np.float64)
        actual_independent_array = np.asarray(
            actual_independent_rows, dtype=np.float64)
        real_array = np.asarray(real_rows, dtype=np.float64)
        common_array = np.asarray(common_rows, dtype=np.float64)
        independent_array = np.asarray(independent_rows, dtype=np.float64)
        check_array = np.asarray(check_rows, dtype=np.float64)
        if real_array.ndim != 2 or real_array.shape[1] != OBS_DIM:
            raise AnalysisError("no valid 67-D replay rows were generated")
        timing_check = _self_check(check_array, independent_array)
        if not timing_check["passed"]:
            raise AnalysisError(
                "independent XML-PD timing self-check failed: "
                + ", ".join(timing_check["failedMetrics"]))
        scale = _load_training_scale(inputs.training_trace_path)
        report = {
            "initialStateContract": initial_contract,
            "selfCheck": timing_check,
            "trainingScale": scale,
            "sourceSeq": np.asarray(valid_source_seq, dtype=np.int64),
            "actualReal": actual_real_array,
            "actualIndependent": actual_independent_array,
            "real": real_array,
            "common": common_array,
            "independent": independent_array,
        }
        return report, patched_xml
    finally:
        common_plant.close()
        check_plant.close()


def _normalization_report(scale: TrainingScale) -> dict[str, Any]:
    return {
        "trainingRows": scale.rows,
        "formula": (
            "scale_i=max(std_train_i, "
            "0.05*sqrt(mean_train(obs_i^2)), 1e-4); "
            "normalized_rms=sqrt(mean((left-right)^2/scale_i^2))"
        ),
        "stdDof": 0,
        "absoluteFloor": 1e-4,
        "rmsFractionFloor": 0.05,
        "mean": scale.mean.tolist(),
        "std": scale.std.tolist(),
        "rms": scale.rms.tolist(),
        "scale": scale.scale.tolist(),
        "diagnosticOnly": True,
    }


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise AnalysisError("report contains non-finite numeric output")
        return number
    return value


def analyze(
    recording_dir: Path,
    *,
    checkpoint: Path | None = None,
    xml: Path | None = None,
    training_trace: Path | None = None,
) -> dict[str, Any]:
    inputs = _resolve_inputs(
        recording_dir,
        checkpoint_override=checkpoint,
        xml_override=xml,
        trace_override=training_trace,
    )
    real_rows = _read_jsonl(inputs.real_path)
    sim_rows = _read_jsonl(inputs.sim_path)
    frames = _align_frames(real_rows, sim_rows)
    plan = _build_replay_plan(frames)
    compensation = _compensation_audit(frames, plan)
    replay, patched_xml_path = _run_replay(inputs, plan)
    scale: TrainingScale = replay["trainingScale"]
    source_seq = replay["sourceSeq"]
    canonical_metrics = _window_report(
        source_seq,
        {
            "canonicalRealToCanonicalCommon": (
                replay["real"], replay["common"]),
            "canonicalCommonToCanonicalIndependent": (
                replay["common"], replay["independent"]),
            "canonicalRealToCanonicalIndependent": (
                replay["real"], replay["independent"]),
        },
        scale.scale,
        inputs.control_dt_s,
    )
    actual_metrics = _window_report(
        source_seq,
        {
            "actualRealToCanonicalCommon": (
                replay["actualReal"], replay["common"]),
            "canonicalCommonToActualIndependent": (
                replay["common"], replay["actualIndependent"]),
            "actualRealToActualIndependent": (
                replay["actualReal"], replay["actualIndependent"]),
        },
        scale.scale,
        inputs.control_dt_s,
    )
    canonicalization_windows = _window_report(
        source_seq,
        {
            "actualRealToCanonicalReal": (
                replay["actualReal"], replay["real"]),
            "actualIndependentToCanonicalIndependent": (
                replay["actualIndependent"], replay["independent"]),
        },
        scale.scale,
        inputs.control_dt_s,
    )
    canonicalization_summary = _canonicalization_summary(
        canonicalization_windows)
    masked_count = len(plan.reset_boundaries)
    if plan.asymmetric_tail_start is not None:
        masked_count += len(frames) - plan.asymmetric_tail_start
    warnings: list[str] = []
    if canonicalization_summary["materialMismatch"]:
        warnings.append(
            "actualRecorded policyObs materially differs from the strict "
            "5 ms training-contract canonicalization; inspect "
            "canonicalizationAudit and do not describe canonical metrics as "
            "the exact historical actor input")
    if compensation["metricUnmaskedTuning"]["mixedCohort"]:
        warnings.append(
            "multiple per-tick compensation tuning cohorts occur in this "
            "recording; full-window metrics mix those cohorts")
    if plan.asymmetric_tail_start is not None:
        warnings.append(
            "an asymmetric reset or inactive-compensation tail was masked; "
            "see mask.reasons")
    report = {
        "schemaVersion": 1,
        "status": "ok_with_warnings" if warnings else "ok",
        "warnings": warnings,
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "source": _file_record(SCRIPT_PATH),
            "runtime": _runtime_provenance(),
            "observationTiming": "observe_then_apply_current_nominal_command",
            "racketVelocityContract": (
                "Generated common shadow and canonical diagnostic use strict "
                "consecutive racket-position secant divided by 0.005s; raw "
                "historical policyObs remains unchanged in the primary metric"
            ),
        },
        "recording": {
            "directory": str(inputs.recording_dir),
            "metadata": _file_record(inputs.metadata_path),
            "realControlStream": _file_record(inputs.real_path),
            "independentControlStream": _file_record(inputs.sim_path),
        },
        "provenance": {
            "checkpoint": _file_record(inputs.checkpoint_path),
            "sourceXml": _file_record(inputs.xml_path),
            "patchedTrainingXml": _file_record(patched_xml_path),
            "trainingObservationTrace": _file_record(
                inputs.training_trace_path),
        },
        "contract": {
            "policyProfile": OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
            "observationDimensions": OBS_DIM,
            "controlDtS": inputs.control_dt_s,
            "initialQdQBallShared": True,
            "ballInputMatchedBySourceSeq": True,
            "commonCommandSource": "real policyCommand.position",
            "independentCommandSource": "sim policyCommand",
            "recordedNumericPrecisionNote": (
                "Commands and observations are replayed from the numeric "
                "precision stored in the probe JSONL."
            ),
        },
        "metricVariants": {
            "primary": PRIMARY_METRIC_VARIANT,
            "canonicalTrainingContractWindows": (
                "All three streams use strict 5 ms training-contract racket "
                "velocity and dependent relative/contact fields. This is a "
                "secondary diagnostic/counterfactual contract-corrected view, "
                "not the primary acceptance metric and not necessarily the "
                "exact 67-D vector consumed by the historical actor."
            ),
            "actualRecordedWindows": (
                "Real and independent observations are the exact recorded "
                "historical policyObs vectors. The common shadow remains a "
                "strict training-contract generated stream. This is the "
                "primary acceptance metric for actual 67-D observation "
                "alignment."
            ),
        },
        "alignment": {
            "method": "exact sourceSeq join plus policy input envelope equality",
            "eventualSequenceAlignmentRequired": True,
            "realtimeCompletionAlignmentRequired": False,
            "alignedPolicySamples": len(frames),
            "firstSourceSeq": frames[0].source_seq,
            "lastSourceSeq": frames[-1].source_seq,
        },
        "mask": {
            "maskedSamples": masked_count,
            "unmaskedSamples": int(source_seq.shape[0]),
            "reasons": list(plan.mask_reasons),
            "rule": (
                "Reset-boundary ticks are excluded. If only one actor's "
                "nominal history resets, that unmatched tail is excluded."
            ),
        },
        "actualCompensation": compensation,
        "initialStateContract": replay["initialStateContract"],
        "selfCheck": replay["selfCheck"],
        "canonicalizationAudit": {
            "summary": canonicalization_summary,
            "windows": canonicalization_windows,
        },
        "normalization": _normalization_report(scale),
        "observationGroups": {
            name: {
                "start": start,
                "stopExclusive": stop,
                "unit": unit,
            }
            for name, (start, stop, unit) in OBS_GROUPS.items()
        },
        "actuatorSelectedGroups": list(ACTUATOR_GROUPS),
        "canonicalTrainingContractWindows": canonical_metrics,
        "actualRecordedWindows": actual_metrics,
    }
    return _finite_json(report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "JSON output path. Default: "
            "<recording>/analysis/common_command_shadow.json; use '-' for stdout"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--training-trace", type=Path, default=None)
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = analyze(
            args.recording_dir,
            checkpoint=args.checkpoint,
            xml=args.xml,
            training_trace=args.training_trace,
        )
        payload = json.dumps(
            report,
            indent=max(0, int(args.indent)),
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        if args.output == "-":
            sys.stdout.write(payload)
        else:
            output = (
                Path(args.output).expanduser().resolve()
                if args.output else (
                    args.recording_dir.expanduser().resolve()
                    / "analysis" / "common_command_shadow.json")
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(payload, encoding="utf-8")
            print(output)
        return 0
    except AnalysisError as exc:
        print(f"common-command shadow analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
