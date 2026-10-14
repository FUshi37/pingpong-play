#!/usr/bin/env python3
"""Replay recorded physical angle commands through the simcode5 MJX plant.

Unlike a same-ball closed-loop replay, this tool injects the exact recorded
``command.publishedSafePosition`` at every control tick through
``MjxJuggleEnv.step_q_ref``.  The command bypasses the policy integrator but
still traverses the checkpoint's command delay, second-order actuator model,
sport PD gains, and MJX substeps.  Real and MJX states are compared at the
tick-start instant immediately before the same tick's command is applied.

The tool then evaluates causal execution-side velocity adapters against this
common-command MJX response with leave-one-recording-out validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_simcode5_mjx_dq_adapter import (
    ARM_DIM,
    DELAY_GRID_STEPS,
    EMA_TAU_GRID_S,
    FIT_WINDOW_S,
    METRIC_WINDOWS_S,
    OBS_DIM,
    AnalysisError,
    Frame,
    ReplayData,
    Segment,
    _actor_batch,
    _causal_q_polynomial,
    _fit_calibrated_adapter,
    _load_recording_root,
    _make_mirror,
    _metadata_paths,
    _pad,
    _rms,
    _vector,
)
from pingpong_controller.rl_policy import RIGHT_ARM_JOINTS


NO_CONTACT_BALL_BASE_M = np.asarray([10.0, 10.0, 10.0], dtype=np.float32)


@dataclass(frozen=True)
class CommonReplay:
    data: ReplayData
    command: np.ndarray
    sim_qacc: np.ndarray
    initial_state_error: dict[str, float]


class CommonCommandMjxReplay:
    """Fixed-batch JAX scan around the environment's calibration entry point."""

    def __init__(self, mirror: Any) -> None:
        self.mirror = mirror
        self.jax = mirror.jax
        self.jnp = mirror.jnp
        self._scan_jit = self.jax.jit(self._scan)

    def _scan(self, state: Any, command_sequence: Any) -> tuple[Any, dict[str, Any]]:
        def one_step(carry: Any, command: Any) -> tuple[Any, dict[str, Any]]:
            data = carry.data
            sample = {
                "q": data.qpos[0, self.mirror.env.arm_qadr],
                "dq": data.qvel[0, self.mirror.env.arm_vadr],
                "qacc": data.qacc[0, self.mirror.env.arm_vadr],
                "ctrl": data.ctrl[0, self.mirror.env.arm_aids_j],
                "active_command": carry.arm_q_ref_active[0],
                "applied_command": carry.arm_applied_q[0],
            }
            action_shape = (1, ARM_DIM)
            next_state, _obs, _reward, _done, _metrics = self.mirror.env.step_q_ref(
                carry,
                self.jnp.asarray(command, dtype=self.jnp.float32).reshape(action_shape),
            )
            return next_state, sample

        return self.jax.lax.scan(
            one_step,
            state,
            self.jnp.asarray(command_sequence, dtype=self.jnp.float32),
        )

    def reset(self, frame: Frame) -> dict[str, float]:
        real_q = frame.real_obs[:ARM_DIM].astype(np.float32)
        real_dq = frame.real_obs[ARM_DIM:2 * ARM_DIM].astype(np.float32)
        self.mirror.reset(
            arm_q=real_q,
            ball_pos=NO_CONTACT_BALL_BASE_M,
            ball_vel=np.zeros(3, dtype=np.float32),
            ball_valid=False,
            ball_obs_age_s=0.0,
        )
        # Match the measured plant state at the replay boundary.  The command
        # delay/filter histories remain at a stationary hold, exactly as they
        # do after mirror.reset; only physical generalized velocity is seeded.
        state = self.mirror.state
        qvel = state.data.qvel.at[:, self.mirror.env.arm_vadr].set(
            self.jnp.asarray(real_dq, dtype=self.jnp.float32)[None, :])
        data = state.data.replace(qvel=qvel)
        data = self.mirror.env.batched_forward(state.model, data)
        self.mirror.state = state._replace(
            data=data,
            prev_arm_qvel=self.jnp.asarray(real_dq, dtype=self.jnp.float32)[None, :],
        )
        q0 = np.asarray(
            self.jax.device_get(self.mirror.state.data.qpos[0, self.mirror.env.arm_qadr]),
            dtype=np.float64,
        )
        dq0 = np.asarray(
            self.jax.device_get(self.mirror.state.data.qvel[0, self.mirror.env.arm_vadr]),
            dtype=np.float64,
        )
        return {
            "qMaxAbsRad": float(np.max(np.abs(q0 - real_q))),
            "dqMaxAbsRadS": float(np.max(np.abs(dq0 - real_dq))),
        }

    def run_batch(self, commands: np.ndarray) -> dict[str, np.ndarray]:
        self.mirror.state, output = self._scan_jit(
            self.mirror.state,
            self.jnp.asarray(commands, dtype=self.jnp.float32),
        )
        return {
            key: np.asarray(self.jax.device_get(value), dtype=np.float64)
            for key, value in output.items()
        }


def _published_segments(segments: list[Segment]) -> tuple[list[Segment], dict[str, int]]:
    output: list[Segment] = []
    unpublished = 0
    split_count = 0
    for segment in segments:
        current: list[Frame] = []
        for frame in segment.frames:
            if not frame.command_published:
                unpublished += 1
                if len(current) >= 2:
                    output.append(Segment(segment.run, segment.episode, tuple(current)))
                    split_count += 1
                current = []
                continue
            current.append(frame)
        if len(current) >= 2:
            output.append(Segment(segment.run, segment.episode, tuple(current)))
    return output, {
        "unpublishedPolicyTicksExcluded": unpublished,
        "splitsAtUnpublishedTicks": split_count,
    }


def _common_replay(
    mirror: Any,
    segments: list[Segment],
    *,
    batch_size: int,
) -> CommonReplay:
    runner = CommonCommandMjxReplay(mirror)
    run_names = sorted({segment.run for segment in segments})
    run_ids = {name: index for index, name in enumerate(run_names)}
    flat_frames: list[Frame] = []
    run_id_rows: list[int] = []
    segment_id_rows: list[int] = []
    segment_position_rows: list[int] = []
    elapsed_rows: list[float] = []
    command_rows: list[np.ndarray] = []
    sim_q_rows: list[np.ndarray] = []
    sim_dq_rows: list[np.ndarray] = []
    sim_qacc_rows: list[np.ndarray] = []
    initial_errors: list[dict[str, float]] = []

    for segment_index, segment in enumerate(segments):
        frames = segment.frames
        initial_errors.append(runner.reset(frames[0]))
        for start in range(0, len(frames), batch_size):
            chunk = frames[start:start + batch_size]
            count = len(chunk)
            commands = _pad(
                np.asarray([frame.published_command for frame in chunk], dtype=np.float32),
                batch_size,
            )
            output = runner.run_batch(commands)
            for offset, frame in enumerate(chunk):
                flat_frames.append(frame)
                run_id_rows.append(run_ids[segment.run])
                segment_id_rows.append(segment_index)
                segment_position_rows.append(start + offset)
                elapsed_rows.append((start + offset) * float(mirror.dt))
                command_rows.append(frame.published_command.copy())
                sim_q_rows.append(output["q"][offset].copy())
                sim_dq_rows.append(output["dq"][offset].copy())
                sim_qacc_rows.append(output["qacc"][offset].copy())
        print(
            f"common-command replay {segment.run} episode={segment.episode} "
            f"ticks={len(frames)}")

    real_obs = np.asarray([frame.real_obs for frame in flat_frames], dtype=np.float64)
    common_obs = real_obs.copy()
    common_obs[:, :ARM_DIM] = np.asarray(sim_q_rows, dtype=np.float64)
    common_obs[:, ARM_DIM:2 * ARM_DIM] = np.asarray(sim_dq_rows, dtype=np.float64)
    data = ReplayData(
        frames=tuple(flat_frames),
        run_id=np.asarray(run_id_rows, dtype=np.int32),
        segment_id=np.asarray(segment_id_rows, dtype=np.int32),
        segment_position=np.asarray(segment_position_rows, dtype=np.int32),
        elapsed_s=np.asarray(elapsed_rows, dtype=np.float64),
        real_obs=real_obs,
        cpu_obs=np.asarray([frame.cpu_obs for frame in flat_frames], dtype=np.float64),
        mjx_obs=common_obs,
        real_action=np.asarray([frame.real_action for frame in flat_frames], dtype=np.float64),
        mjx_action=np.zeros((len(flat_frames), ARM_DIM), dtype=np.float64),
        joint_stamp_s=np.asarray([frame.joint_stamp_s for frame in flat_frames], dtype=np.float64),
    )
    return CommonReplay(
        data=data,
        command=np.asarray(command_rows, dtype=np.float64),
        sim_qacc=np.asarray(sim_qacc_rows, dtype=np.float64),
        initial_state_error={
            "qMaxAbsRad": max(item["qMaxAbsRad"] for item in initial_errors),
            "dqMaxAbsRadS": max(item["dqMaxAbsRadS"] for item in initial_errors),
        },
    )


def _metrics(
    name: str,
    dq: np.ndarray,
    replay: CommonReplay,
    weights: dict[str, Any],
) -> dict[str, Any]:
    data = replay.data
    target = data.mjx_obs[:, 7:14]
    adapted_obs = data.real_obs.copy()
    adapted_obs[:, 7:14] = dq
    raw_action = _actor_batch(data.real_obs, weights)
    adapted_action = _actor_batch(adapted_obs, weights)
    output: dict[str, Any] = {"name": name, "windows": {}}
    windows: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(data.elapsed_s.shape, dtype=bool))]
    windows.extend([
        (f"first_{window:.2f}s", data.elapsed_s <= window + 1e-12)
        for window in METRIC_WINDOWS_S
    ])
    for label, mask in windows:
        error = dq[mask] - target[mask]
        per_joint = np.sqrt(np.mean(np.square(error), axis=0))
        correlations: list[float | None] = []
        for joint in range(ARM_DIM):
            left = dq[mask, joint]
            right = target[mask, joint]
            if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
                correlations.append(None)
            else:
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
        action_delta = adapted_action[mask] - raw_action[mask]
        output["windows"][label] = {
            "samples": int(np.count_nonzero(mask)),
            "dqRmsRadS": _rms(error),
            "dqPerJointRmsRadS": per_joint.tolist(),
            "dqCorrelationPerJoint": correlations,
            "policyActionChangeRms": _rms(action_delta),
            "policyActionChangeMax": float(np.max(np.abs(action_delta))),
        }
    return output


def _write_samples(path: Path, replay: CommonReplay, calibrated: np.ndarray) -> None:
    data = replay.data
    with path.open("w", encoding="utf-8") as handle:
        for index, frame in enumerate(data.frames):
            row = {
                "run": frame.run,
                "sourceSeq": frame.source_seq,
                "episode": frame.episode,
                "segment": int(data.segment_id[index]),
                "segmentPosition": int(data.segment_position[index]),
                "elapsedS": float(data.elapsed_s[index]),
                "publishedAngleCommandRad": replay.command[index].tolist(),
                "realQRad": data.real_obs[index, :7].tolist(),
                "realDqRadS": data.real_obs[index, 7:14].tolist(),
                "mjxCommonCommandQRad": data.mjx_obs[index, :7].tolist(),
                "mjxCommonCommandDqRadS": data.mjx_obs[index, 7:14].tolist(),
                "mjxCommonCommandQaccRadS2": replay.sim_qacc[index].tolist(),
                "calibratedRealDqRadS": calibrated[index].tolist(),
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _plot_joint_comparison(path: Path, replay: CommonReplay) -> None:
    data = replay.data
    counts = np.bincount(data.segment_id)
    sid = int(np.argmax(counts))
    indices = np.flatnonzero(data.segment_id == sid)
    elapsed = data.elapsed_s[indices]
    fig, axes = plt.subplots(ARM_DIM, 2, figsize=(16, 20), sharex=True)
    for joint in range(ARM_DIM):
        axes[joint, 0].plot(
            elapsed, data.real_obs[indices, joint], label="real", linewidth=1.1)
        axes[joint, 0].plot(
            elapsed, data.mjx_obs[indices, joint], label="MJX same command", linewidth=1.1)
        axes[joint, 0].plot(
            elapsed, replay.command[indices, joint], label="published command", linewidth=0.9, alpha=0.7)
        axes[joint, 1].plot(
            elapsed, data.real_obs[indices, 7 + joint], label="real", linewidth=1.1)
        axes[joint, 1].plot(
            elapsed, data.mjx_obs[indices, 7 + joint], label="MJX same command", linewidth=1.1)
        axes[joint, 0].set_ylabel(f"{RIGHT_ARM_JOINTS[joint]}\nrad")
        axes[joint, 1].set_ylabel("rad/s")
        axes[joint, 0].grid(True, alpha=0.22)
        axes[joint, 1].grid(True, alpha=0.22)
    axes[0, 0].set_title("angle response to identical published command")
    axes[0, 1].set_title("velocity response to identical published command")
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 1].legend(ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("time since common-command reset (s)")
    axes[-1, 1].set_xlabel("time since common-command reset (s)")
    first = data.frames[indices[0]]
    fig.suptitle(f"MJX common-command replay — {first.run}, episode {first.episode}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_adapter(
    path: Path,
    replay: CommonReplay,
    candidates: dict[str, np.ndarray],
) -> None:
    data = replay.data
    counts = np.bincount(data.segment_id)
    sid = int(np.argmax(counts))
    indices = np.flatnonzero(data.segment_id == sid)
    elapsed = data.elapsed_s[indices]
    fig, axes = plt.subplots(ARM_DIM, 1, figsize=(15, 18), sharex=True)
    for joint, axis in enumerate(axes):
        axis.plot(
            elapsed,
            data.mjx_obs[indices, 7 + joint],
            color="#d62728",
            linewidth=1.5,
            label="MJX same-command target",
        )
        axis.plot(
            elapsed,
            candidates["real_joint_state"][indices, joint],
            color="#1f77b4",
            linewidth=1.0,
            label="real joint_state",
        )
        axis.plot(
            elapsed,
            candidates["q_causal_poly5"][indices, joint],
            color="#59a14f",
            linewidth=0.9,
            label="q causal poly5",
        )
        axis.plot(
            elapsed,
            candidates["calibrated_gain_delay_ema_cv"][indices, joint],
            color="#f28e2b",
            linewidth=1.0,
            label="calibrated causal adapter",
        )
        axis.set_ylabel(f"{RIGHT_ARM_JOINTS[joint]}\nrad/s")
        axis.grid(True, alpha=0.22)
    axes[0].legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("time since common-command reset (s)")
    fig.suptitle("Causal dq adapters against MJX response to identical angle commands")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_summary(path: Path, metrics: list[dict[str, Any]]) -> None:
    labels = [item["name"] for item in metrics]
    early = [item["windows"]["first_0.50s"]["dqRmsRadS"] for item in metrics]
    overall = [item["windows"]["all"]["dqRmsRadS"] for item in metrics]
    action = [item["windows"]["all"]["policyActionChangeRms"] for item in metrics]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    width = 0.38
    axes[0].bar(x - width / 2, early, width, label="first 0.50 s")
    axes[0].bar(x + width / 2, overall, width, label="all")
    axes[0].set_ylabel("dq RMS to same-command MJX (rad/s)")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, action, 0.6, color="#59a14f")
    axes[1].set_ylabel("change to deployed policy action RMS")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Common-command execution-side dq candidates")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def analyze(root: Path, output_dir: Path, batch_size: int) -> dict[str, Any]:
    loaded_segments, input_audit = _load_recording_root(root)
    segments, publish_audit = _published_segments(loaded_segments)
    if not segments:
        raise AnalysisError("no continuous published-command segments remain")
    model, xml, dt_s = _metadata_paths(root)
    mirror = _make_mirror(model, xml, dt_s)
    try:
        replay = _common_replay(mirror, segments, batch_size=batch_size)
        weights = dict(mirror.policy._pi_weights or {})
        env_contract = {
            "entryPoint": "MjxJuggleEnv.step_q_ref",
            "obsDim": int(mirror.env.obs_dim),
            "dtS": float(mirror.dt),
            "rightArmPdProfile": str(getattr(mirror.env_cfg, "right_arm_pd_profile", "")),
            "actuatorCommandModel": str(getattr(mirror.env_cfg, "actuator_cmd_model", "")),
            "domainRandomization": bool(getattr(mirror.env_cfg, "domain_randomization", False)),
            "commandSource": "control_samples.jsonl[].command.publishedSafePosition",
            "comparisonPhase": "tick-start state before applying same tick command",
        }
    finally:
        close = getattr(mirror, "close", None)
        if callable(close):
            close()

    data = replay.data
    raw = data.real_obs[:, 7:14]
    target = data.mjx_obs[:, 7:14]
    q_poly3 = _causal_q_polynomial(data, 3)
    q_poly5 = _causal_q_polynomial(data, 5)
    q_poly9 = _causal_q_polynomial(data, 9)
    gain_cv, gain_cv_params = _fit_calibrated_adapter(
        data, raw, target, search_tau=False, cross_validate=True, dt_s=dt_s)
    calibrated_cv, calibrated_cv_params = _fit_calibrated_adapter(
        data, raw, target, search_tau=True, cross_validate=True, dt_s=dt_s)
    calibrated_full, deployment_params = _fit_calibrated_adapter(
        data, raw, target, search_tau=True, cross_validate=False, dt_s=dt_s)
    candidates = {
        "real_joint_state": raw,
        "q_causal_poly3": q_poly3,
        "q_causal_poly5": q_poly5,
        "q_causal_poly9": q_poly9,
        "calibrated_gain_only_cv": gain_cv,
        "calibrated_gain_delay_ema_cv": calibrated_cv,
    }
    metrics = [
        _metrics(name, values, replay, weights)
        for name, values in candidates.items()
    ]

    command_delta = replay.command - data.real_obs[:, :7]
    replay_audit = {
        "sampleCount": len(data.frames),
        "segmentCount": int(np.unique(data.segment_id).size),
        "runCount": int(np.unique(data.run_id).size),
        "initialStateError": replay.initial_state_error,
        "realToMjxQRmsRad": _rms(data.real_obs[:, :7] - data.mjx_obs[:, :7]),
        "realToMjxDqRmsRadS": _rms(raw - target),
        "realDqRmsRadS": _rms(raw),
        "mjxDqRmsRadS": _rms(target),
        "publishedCommandTrackingErrorRealRmsRad": _rms(command_delta),
        "publishedVsPolicyNominalMaxAbsRad": float(np.max(np.abs(
            replay.command - np.asarray(
                [frame.policy_nominal_command for frame in data.frames],
                dtype=np.float64,
            )
        ))),
        "publishedVsCompensatedMaxAbsRad": float(np.max(np.abs(
            replay.command - np.asarray(
                [frame.compensated_command for frame in data.frames],
                dtype=np.float64,
            )
        ))),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_samples(output_dir / "common_command_samples.jsonl", replay, calibrated_full)
    _plot_joint_comparison(output_dir / "common_command_joint_comparison.png", replay)
    _plot_adapter(output_dir / "common_command_dq_adapter_trace.png", replay, candidates)
    _plot_summary(output_dir / "common_command_dq_summary.png", metrics)

    with (output_dir / "candidate_summary.csv").open(
        "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate", "dq_rms_first_0_10s", "dq_rms_first_0_25s",
            "dq_rms_first_0_50s", "dq_rms_all", "policy_action_change_rms_all",
            "policy_action_change_max_all",
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
                "policy_action_change_rms_all": item["windows"]["all"]["policyActionChangeRms"],
                "policy_action_change_max_all": item["windows"]["all"]["policyActionChangeMax"],
            })

    report = {
        "analysis": "simcode5_mjx_common_published_angle_command_replay",
        "recordingRoot": str(root),
        "checkpoint": str(model),
        "xml": str(xml),
        "envContract": env_contract,
        "inputAudit": input_audit,
        "publishAudit": publish_audit,
        "replayAudit": replay_audit,
        "fitContract": {
            "causal": True,
            "crossValidation": "leave-one-recording-out",
            "fitWindowS": FIT_WINDOW_S,
            "emaTauGridS": list(EMA_TAU_GRID_S),
            "delayGridSteps": list(DELAY_GRID_STEPS),
            "zeroIntercept": True,
            "gainConstraint": [0.25, 2.0],
        },
        "candidateMetrics": metrics,
        "gainOnlyCrossValidationParameters": gain_cv_params,
        "calibratedCrossValidationParameters": calibrated_cv_params,
        "deploymentParametersFitAllRecordings": deployment_params,
        "warnings": [
            "The calibrated dq is policy input only; raw joint_state velocity must remain authoritative for safety.",
            "This replay isolates command-to-joint response and intentionally removes ball contact by placing the MJX ball far away.",
            "Parameters should first run in shadow logging mode before they can affect real commands.",
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
        help="default: <recording_root>/simcode5_mjx_common_command",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    root = args.recording_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir else root / "simcode5_mjx_common_command")
    report = analyze(root, output_dir, max(1, int(args.batch_size)))
    audit = report["replayAudit"]
    print(
        f"wrote {output_dir} | runs={audit['runCount']} "
        f"segments={audit['segmentCount']} samples={audit['sampleCount']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
