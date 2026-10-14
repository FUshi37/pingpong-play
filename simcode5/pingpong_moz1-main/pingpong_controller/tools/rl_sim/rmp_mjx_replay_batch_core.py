#!/usr/bin/env python3
"""Vectorized MJX replay helpers for recovered RMP recording validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from recovered_rmp.replay import (  # noqa: E402
    RecordingTrace,
    forward_velocity_acceleration,
    load_control_recording,
)

CONTROL_CHAIN = (
    "recorded q target -> recovered RMP q/qd/qdd at 1 kHz -> "
    "XML low-level PD -> MJX at 1 kHz"
)
ACCELERATION_MEASUREMENT = (
    "forward velocity difference; real dt=ros.jointSampleStampS, "
    "sim dt=sourceTimeS, contiguous sourceSeq only"
)
RACKET_METRIC_KEYS = (
    "center_pos_m",
    "center_quat_wxyz",
    "center_rotmat_rowmajor",
    "center_vel_m_s",
    "center_acc_m_s2",
    "center_angular_velocity_world_rad_s",
    "center_angular_velocity_local_rad_s",
    "center_angular_acceleration_world_rad_s2",
    "center_angular_acceleration_local_rad_s2",
)


@dataclass(frozen=True)
class ReplayConfigResolved:
    calibration_profile: str
    rmp_jnt_kp_scale: float
    rmp_jnt_kd_scale: float
    controller_kp_scale: float
    controller_kd_scale: float
    controller_kp_scale_per_joint: tuple[float, ...]
    controller_kd_scale_per_joint: tuple[float, ...]
    rmp_jnt_kp_scale_per_joint: tuple[float, ...]
    rmp_jnt_kd_scale_per_joint: tuple[float, ...]
    dof_damping_scale_per_joint: tuple[float, ...]
    dof_armature_scale_per_joint: tuple[float, ...]
    acceleration_feedforward_s2: float
    rmp_output_delay_steps_per_joint: tuple[int, ...]
    env_cfg: MjxJuggleConfig

    def report_fields(self) -> dict[str, object]:
        return {
            "calibration_profile": self.calibration_profile,
            "rmp_jnt_kp_scale": self.rmp_jnt_kp_scale,
            "rmp_jnt_kd_scale": self.rmp_jnt_kd_scale,
            "rmp_jnt_kp_scale_per_joint": list(self.rmp_jnt_kp_scale_per_joint),
            "rmp_jnt_kd_scale_per_joint": list(self.rmp_jnt_kd_scale_per_joint),
            "dof_damping_scale_per_joint": list(self.dof_damping_scale_per_joint),
            "dof_armature_scale_per_joint": list(self.dof_armature_scale_per_joint),
            "controller_kp_scale": self.controller_kp_scale,
            "controller_kd_scale": self.controller_kd_scale,
            "controller_kp_scale_per_joint": list(self.controller_kp_scale_per_joint),
            "controller_kd_scale_per_joint": list(self.controller_kd_scale_per_joint),
            "acceleration_feedforward_s2": self.acceleration_feedforward_s2,
            "rmp_output_delay_steps_per_joint": list(self.rmp_output_delay_steps_per_joint),
        }


@dataclass(frozen=True)
class PreparedReplay:
    group: str
    recording_path: Path
    recording_name: str
    trace: RecordingTrace
    phase_info: dict[str, object]
    sample_count: int


@dataclass(frozen=True)
class ReplayBatch:
    records: tuple[PreparedReplay, ...]
    targets: jax.Array
    active_mask: jax.Array
    lengths: np.ndarray
    initial_actual_q: jax.Array
    initial_actual_qd: jax.Array
    initial_actual_qdd: jax.Array
    initial_target_q: jax.Array


@dataclass(frozen=True)
class BatchedReplayResult:
    final_state: Any
    values: dict[str, jax.Array]
    active_mask: jax.Array
    lengths: np.ndarray
    device_seconds: float


def joint_scales(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if len(result) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated values")
    return result


def joint_delay_steps(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 7 or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected seven comma-separated nonnegative integers"
        )
    return result


def add_replay_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--segment", choices=("longest", "first", "last"), default="longest")
    parser.add_argument(
        "--valid-phase-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim each replay trace to the contiguous valid juggling phase only.",
    )
    parser.add_argument(
        "--calibration-profile",
        choices=(
            "recovered_yaml_pd_fit",
            "record_new3_4",
            "record_new3_4_q_qd_qdd",
        ),
        default="recovered_yaml_pd_fit",
    )
    parser.add_argument("--rmp-jnt-kp-scale", type=float)
    parser.add_argument("--rmp-jnt-kd-scale", type=float)
    parser.add_argument("--controller-kp-scale", type=float)
    parser.add_argument("--controller-kd-scale", type=float)
    parser.add_argument("--controller-kp-scale-per-joint", type=joint_scales, default=())
    parser.add_argument("--controller-kd-scale-per-joint", type=joint_scales, default=())
    parser.add_argument("--rmp-jnt-kp-scale-per-joint", type=joint_scales, default=())
    parser.add_argument("--rmp-jnt-kd-scale-per-joint", type=joint_scales, default=())
    parser.add_argument("--dof-damping-scale-per-joint", type=joint_scales, default=())
    parser.add_argument("--dof-armature-scale-per-joint", type=joint_scales, default=())
    parser.add_argument("--rmp-target-filter-length", type=int, default=10)
    parser.add_argument("--rmp-estimator-process-var", type=float, default=50.0)
    parser.add_argument("--rmp-estimator-measure-var", type=float, default=0.0001)
    parser.add_argument("--rmp-velocity-feedforward", type=float, default=0.5)
    parser.add_argument("--rmp-acceleration-feedforward", type=float, default=0.0)
    parser.add_argument("--rmp-acceleration-weight", type=float, default=0.001)
    parser.add_argument(
        "--rmp-output-delay-steps-per-joint",
        type=joint_delay_steps,
        default=(),
        help="pure 1 kHz RMP-output-to-PD transport delay; does not enable actuator model",
    )
    parser.add_argument(
        "--acceleration-feedforward-s2",
        type=float,
        default=0.0,
        help="qdd-to-equivalent-position coefficient; zero best fit the current logs",
    )
    parser.add_argument("--max-samples", type=int, default=0)


def _selected(value: float | None, default: float) -> float:
    return default if value is None else float(value)


def resolve_replay_config(args: argparse.Namespace) -> ReplayConfigResolved:
    profile = {
        "recovered_yaml_pd_fit": (1.0, 1.0, 0.02, 0.10),
        "record_new3_4": (0.30, math.sqrt(0.30), 0.03, 0.10),
        "record_new3_4_q_qd_qdd": (1.0, 1.0, 1.0, 1.0),
    }[args.calibration_profile]
    rmp_kp_scale = _selected(args.rmp_jnt_kp_scale, profile[0])
    rmp_kd_scale = _selected(args.rmp_jnt_kd_scale, profile[1])
    controller_kp_scale = _selected(args.controller_kp_scale, profile[2])
    controller_kd_scale = _selected(args.controller_kd_scale, profile[3])
    fitted_kp_per_joint = (
        0.009803635068237782,
        0.0072898827493190765,
        0.012868792749941349,
        0.011012702248990536,
        0.004711992107331753,
        0.011036773212254047,
        0.005395024083554745,
    )
    fitted_kd_per_joint = (
        0.054490312933921814,
        0.06669235229492188,
        0.02183920331299305,
        0.03845234215259552,
        0.005404237192124128,
        0.009526701644062996,
        0.005226867739111185,
    )
    fitted_rmp_kp_per_joint = (
        0.33092403411865234,
        0.42426156997680664,
        0.3956700265407562,
        0.46115389466285706,
        0.5072692632675171,
        0.3956700265407562,
        0.5072692632675171,
    )
    fitted_rmp_kd_per_joint = (
        0.40992820262908936,
        0.3360067307949066,
        0.40992820262908936,
        0.36960741877555847,
        0.36960741877555847,
        0.36960741877555847,
        0.36960741877555847,
    )
    fitted_damping_per_joint = (1.92, 1.92, 1.92, 1.92, 1.22, 1.92, 1.10)
    fitted_armature_per_joint = (1.92,) * 7
    kp_scale_per_joint = args.controller_kp_scale_per_joint
    kd_scale_per_joint = args.controller_kd_scale_per_joint
    rmp_kp_scale_per_joint = args.rmp_jnt_kp_scale_per_joint
    rmp_kd_scale_per_joint = args.rmp_jnt_kd_scale_per_joint
    damping_scale_per_joint = args.dof_damping_scale_per_joint
    armature_scale_per_joint = args.dof_armature_scale_per_joint
    if args.calibration_profile == "record_new3_4_q_qd_qdd":
        kp_scale_per_joint = kp_scale_per_joint or fitted_kp_per_joint
        kd_scale_per_joint = kd_scale_per_joint or fitted_kd_per_joint
        rmp_kp_scale_per_joint = rmp_kp_scale_per_joint or fitted_rmp_kp_per_joint
        rmp_kd_scale_per_joint = rmp_kd_scale_per_joint or fitted_rmp_kd_per_joint
        damping_scale_per_joint = damping_scale_per_joint or fitted_damping_per_joint
        armature_scale_per_joint = armature_scale_per_joint or fitted_armature_per_joint
    acceleration_ff = float(args.acceleration_feedforward_s2)
    delay_steps = tuple(args.rmp_output_delay_steps_per_joint or (0,) * 7)
    env_cfg = MjxJuggleConfig(
        domain_randomization=False,
        arm_action_limiter=False,
        recovered_rmp_motion_mode=True,
        recovered_rmp_jnt_kp_scale=rmp_kp_scale,
        recovered_rmp_jnt_kd_scale=rmp_kd_scale,
        recovered_rmp_target_filter_length=int(args.rmp_target_filter_length),
        recovered_rmp_estimator_joint_process_var=float(args.rmp_estimator_process_var),
        recovered_rmp_estimator_joint_measure_var=float(args.rmp_estimator_measure_var),
        recovered_rmp_joint_velocity_feedforward=float(args.rmp_velocity_feedforward),
        recovered_rmp_joint_acceleration_feedforward=float(args.rmp_acceleration_feedforward),
        recovered_rmp_acceleration_weight=float(args.rmp_acceleration_weight),
        recovered_rmp_output_delay_steps_per_joint=delay_steps,
        recovered_rmp_controller_kp_scale=controller_kp_scale,
        recovered_rmp_controller_kd_scale=controller_kd_scale,
        recovered_rmp_controller_kp_scale_per_joint=kp_scale_per_joint,
        recovered_rmp_controller_kd_scale_per_joint=kd_scale_per_joint,
        recovered_rmp_jnt_kp_scale_per_joint=rmp_kp_scale_per_joint,
        recovered_rmp_jnt_kd_scale_per_joint=rmp_kd_scale_per_joint,
        recovered_rmp_dof_damping_scale_per_joint=damping_scale_per_joint,
        recovered_rmp_dof_armature_scale_per_joint=armature_scale_per_joint,
        recovered_rmp_acceleration_feedforward_s2=(acceleration_ff,) * 7,
        actuator_cmd_filter=False,
        actuator_compensation_mode="none",
        enable_delay_conditioning=False,
    )
    return ReplayConfigResolved(
        calibration_profile=args.calibration_profile,
        rmp_jnt_kp_scale=rmp_kp_scale,
        rmp_jnt_kd_scale=rmp_kd_scale,
        controller_kp_scale=controller_kp_scale,
        controller_kd_scale=controller_kd_scale,
        controller_kp_scale_per_joint=tuple(kp_scale_per_joint),
        controller_kd_scale_per_joint=tuple(kd_scale_per_joint),
        rmp_jnt_kp_scale_per_joint=tuple(rmp_kp_scale_per_joint),
        rmp_jnt_kd_scale_per_joint=tuple(rmp_kd_scale_per_joint),
        dof_damping_scale_per_joint=tuple(damping_scale_per_joint),
        dof_armature_scale_per_joint=tuple(armature_scale_per_joint),
        acceleration_feedforward_s2=acceleration_ff,
        rmp_output_delay_steps_per_joint=delay_steps,
        env_cfg=env_cfg,
    )


def _valid_phase(trace: RecordingTrace) -> tuple[RecordingTrace, dict[str, object]]:
    indices = np.flatnonzero(trace.policy_mask)
    if indices.size == 0:
        raise ValueError(f"no valid policy phase in {trace.source}")
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    selection = slice(start, stop)
    trace_time = trace.time[selection].copy()
    trace_time -= trace_time[0]
    phase = RecordingTrace(
        source=trace.source,
        time=trace_time,
        source_sequence=trace.source_sequence[selection].copy(),
        joint_time=trace.joint_time[selection].copy(),
        target_q=trace.target_q[selection].copy(),
        actual_q=trace.actual_q[selection].copy(),
        actual_qd=trace.actual_qd[selection].copy(),
        actual_qdd=trace.actual_qdd[selection].copy(),
        policy_mask=np.ones(stop - start, dtype=bool),
        median_dt=trace.median_dt,
    )
    return phase, {
        "valid_phase_only": True,
        "source_start_index": start,
        "source_stop_index": stop,
        "source_policy_samples": int(indices.size),
        "source_samples": int(trace.time.size),
        "source_start_s": float(trace.time[start]),
        "source_stop_s": float(trace.time[stop - 1]),
        "source_valid_duration_s": (
            float(trace.time[stop - 1] - trace.time[start]) if indices.size > 1 else 0.0
        ),
    }


def prepare_recording(
    recording: Path,
    *,
    group: str = "recording",
    segment: str = "longest",
    valid_phase_only: bool = True,
    max_samples: int = 0,
) -> PreparedReplay:
    trace = load_control_recording(recording, segment=segment)
    phase_info: dict[str, object] = {"valid_phase_only": False}
    if valid_phase_only:
        trace, phase_info = _valid_phase(trace)
    count = int(trace.time.size)
    if int(max_samples) > 0:
        count = min(count, int(max_samples))
        selection = slice(0, count)
        trace = RecordingTrace(
            source=trace.source,
            time=trace.time[selection].copy(),
            source_sequence=trace.source_sequence[selection].copy(),
            joint_time=trace.joint_time[selection].copy(),
            target_q=trace.target_q[selection].copy(),
            actual_q=trace.actual_q[selection].copy(),
            actual_qd=trace.actual_qd[selection].copy(),
            actual_qdd=trace.actual_qdd[selection].copy(),
            policy_mask=trace.policy_mask[selection].copy(),
            median_dt=trace.median_dt,
        )
    if count < 2:
        raise ValueError(f"recording has fewer than two replay samples: {recording}")
    for name, arr in (
        ("target_q", trace.target_q),
        ("actual_q", trace.actual_q),
        ("actual_qd", trace.actual_qd),
        ("actual_qdd", trace.actual_qdd),
    ):
        if arr.shape != (count, 7):
            raise ValueError(f"{name} must have shape {(count, 7)}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
    path = Path(recording).expanduser().resolve()
    name = path.name if path.is_dir() else path.parent.name
    return PreparedReplay(
        group=group,
        recording_path=path,
        recording_name=name,
        trace=trace,
        phase_info=phase_info,
        sample_count=count,
    )


def build_replay_batch(records: Sequence[PreparedReplay]) -> ReplayBatch:
    if not records:
        raise ValueError("cannot build an empty replay batch")
    n_envs = len(records)
    lengths = np.asarray([record.sample_count for record in records], dtype=np.int32)
    max_steps = int(np.max(lengths))
    targets = np.zeros((max_steps, n_envs, 7), dtype=np.float32)
    active = np.zeros((max_steps, n_envs), dtype=bool)
    initial_q = np.zeros((n_envs, 7), dtype=np.float32)
    initial_qd = np.zeros((n_envs, 7), dtype=np.float32)
    initial_qdd = np.zeros((n_envs, 7), dtype=np.float32)
    initial_target = np.zeros((n_envs, 7), dtype=np.float32)
    for lane, record in enumerate(records):
        length = int(lengths[lane])
        targets[:length, lane] = record.trace.target_q[:length]
        if length < max_steps:
            targets[length:, lane] = record.trace.target_q[length - 1]
        active[:length, lane] = True
        initial_q[lane] = record.trace.actual_q[0]
        initial_qd[lane] = record.trace.actual_qd[0]
        initial_qdd[lane] = record.trace.actual_qdd[0]
        initial_target[lane] = record.trace.target_q[0]
    return ReplayBatch(
        records=tuple(records),
        targets=jnp.asarray(targets, dtype=jnp.float32),
        active_mask=jnp.asarray(active),
        lengths=lengths,
        initial_actual_q=jnp.asarray(initial_q, dtype=jnp.float32),
        initial_actual_qd=jnp.asarray(initial_qd, dtype=jnp.float32),
        initial_actual_qdd=jnp.asarray(initial_qdd, dtype=jnp.float32),
        initial_target_q=jnp.asarray(initial_target, dtype=jnp.float32),
    )


def initialize_batch_state(env: MjxJuggleEnv, batch: ReplayBatch, seed: int = 0) -> Any:
    n_envs = len(batch.records)
    state, _ = env.reset(jax.random.split(jax.random.PRNGKey(seed), n_envs))
    ball_pos = jnp.tile(jnp.asarray([[0.0, 0.0, 5.0]], dtype=jnp.float32), (n_envs, 1))
    qpos = state.data.qpos.at[:, env.ball_qadr : env.ball_qadr + 3].set(ball_pos)
    qvel = state.data.qvel.at[:, env.ball_vadr : env.ball_vadr + 6].set(0.0)
    replay_data = env.batched_forward(state.model, state.data.replace(qpos=qpos, qvel=qvel))
    state = state._replace(data=replay_data)
    return env.reset_arm_replay_state(
        state,
        batch.initial_actual_q,
        measured_qd=batch.initial_actual_qd,
        measured_qdd=batch.initial_actual_qdd,
        position_target=batch.initial_target_q,
    )


def _is_batched_leaf(value: Any, n_envs: int) -> bool:
    return hasattr(value, "shape") and getattr(value, "ndim", 0) >= 1 and int(value.shape[0]) == int(n_envs)


def select_active_lanes(old_state: Any, new_state: Any, active_mask: jax.Array, n_envs: int) -> Any:
    active = jnp.asarray(active_mask, dtype=bool)

    def choose(old_leaf: Any, new_leaf: Any) -> Any:
        if _is_batched_leaf(new_leaf, n_envs):
            mask = active.reshape((n_envs,) + (1,) * (new_leaf.ndim - 1))
            return jnp.where(mask, new_leaf, old_leaf)
        return new_leaf

    return jax.tree_util.tree_map(choose, old_state, new_state)


def run_batched_replay(env: MjxJuggleEnv, initial_state: Any, batch: ReplayBatch) -> BatchedReplayResult:
    n_envs = len(batch.records)

    def replay_scan(state: Any, inputs: tuple[jax.Array, jax.Array]) -> tuple[Any, dict[str, jax.Array]]:
        target, active = inputs
        racket_mat = state.data.site_xmat[:, env.racket_site_id].reshape((-1, 3, 3))
        output = {
            "sim_q": state.data.qpos[:, env.arm_qadr],
            "sim_qd": state.data.qvel[:, env.arm_vadr],
            "sim_qacc": state.data.qacc[:, env.arm_vadr],
            "rmp_q": state.rmp_q,
            "rmp_qd": state.rmp_qd,
            "rmp_qdd": state.rmp_qdd,
            "racket_pos": state.data.site_xpos[:, env.racket_site_id],
            "racket_mat": racket_mat,
            "racket_omega_world": state.data.cvel[:, env.racket_body_id, :3],
        }
        candidate_state, _, _, _, _ = env.step_q_ref(state, target)
        next_state = select_active_lanes(state, candidate_state, active, n_envs)
        return next_state, output

    @jax.jit
    def run(initial: Any, targets: jax.Array, active_mask: jax.Array) -> tuple[Any, dict[str, jax.Array]]:
        return jax.lax.scan(replay_scan, initial, (targets, active_mask))

    start = time.perf_counter()
    final_state, values = run(initial_state, batch.targets, batch.active_mask)
    jax.block_until_ready(final_state.data.qpos)
    elapsed = time.perf_counter() - start
    return BatchedReplayResult(
        final_state=final_state,
        values=values,
        active_mask=batch.active_mask,
        lengths=batch.lengths,
        device_seconds=elapsed,
    )


def _rotmat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float32)
    q = np.empty((m.shape[0], 4), dtype=np.float32)
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    positive = tr > 0.0
    s = np.sqrt(tr[positive] + 1.0) * 2.0
    q[positive, 0] = 0.25 * s
    q[positive, 1] = (m[positive, 2, 1] - m[positive, 1, 2]) / s
    q[positive, 2] = (m[positive, 0, 2] - m[positive, 2, 0]) / s
    q[positive, 3] = (m[positive, 1, 0] - m[positive, 0, 1]) / s
    for i in np.where(~positive)[0]:
        if m[i, 0, 0] > m[i, 1, 1] and m[i, 0, 0] > m[i, 2, 2]:
            s = np.sqrt(1.0 + m[i, 0, 0] - m[i, 1, 1] - m[i, 2, 2]) * 2.0
            q[i, 0] = (m[i, 2, 1] - m[i, 1, 2]) / s
            q[i, 1] = 0.25 * s
            q[i, 2] = (m[i, 0, 1] + m[i, 1, 0]) / s
            q[i, 3] = (m[i, 0, 2] + m[i, 2, 0]) / s
        elif m[i, 1, 1] > m[i, 2, 2]:
            s = np.sqrt(1.0 + m[i, 1, 1] - m[i, 0, 0] - m[i, 2, 2]) * 2.0
            q[i, 0] = (m[i, 0, 2] - m[i, 2, 0]) / s
            q[i, 1] = (m[i, 0, 1] + m[i, 1, 0]) / s
            q[i, 2] = 0.25 * s
            q[i, 3] = (m[i, 1, 2] + m[i, 2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[i, 2, 2] - m[i, 0, 0] - m[i, 1, 1]) * 2.0
            q[i, 0] = (m[i, 1, 0] - m[i, 0, 1]) / s
            q[i, 1] = (m[i, 0, 2] + m[i, 2, 0]) / s
            q[i, 2] = (m[i, 1, 2] + m[i, 2, 1]) / s
            q[i, 3] = 0.25 * s
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return q / norms


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(arr, dtype=float) ** 2)))


def _lane_value(values: dict[str, jax.Array], key: str, lane: int, length: int) -> np.ndarray:
    return np.asarray(values[key])[:length, lane]


def make_record_artifacts(
    record: PreparedReplay,
    result: BatchedReplayResult,
    lane: int,
    env_dt: float,
    config: ReplayConfigResolved,
    *,
    batch_index: int = 0,
    padded_batch_samples: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    count = int(result.lengths[lane])
    trace = record.trace
    sim_q = _lane_value(result.values, "sim_q", lane, count)
    sim_qd = _lane_value(result.values, "sim_qd", lane, count)
    sim_qacc = _lane_value(result.values, "sim_qacc", lane, count)
    rmp_q = _lane_value(result.values, "rmp_q", lane, count)
    rmp_qd = _lane_value(result.values, "rmp_qd", lane, count)
    rmp_qdd = _lane_value(result.values, "rmp_qdd", lane, count)
    racket_pos = _lane_value(result.values, "racket_pos", lane, count)
    racket_mat = _lane_value(result.values, "racket_mat", lane, count).reshape(count, 3, 3)
    racket_omega_world = _lane_value(result.values, "racket_omega_world", lane, count)
    dt = max(float(env_dt), 1.0e-6)
    racket_vel = np.diff(racket_pos, axis=0, prepend=racket_pos[:1]) / dt
    racket_acc = np.diff(racket_vel, axis=0, prepend=racket_vel[:1]) / dt
    racket_omega_local = np.einsum("nij,nj->ni", np.swapaxes(racket_mat, 1, 2), racket_omega_world)
    racket_alpha_world = np.diff(racket_omega_world, axis=0, prepend=racket_omega_world[:1]) / dt
    racket_alpha_local = np.einsum("nij,nj->ni", np.swapaxes(racket_mat, 1, 2), racket_alpha_world)
    racket_quat = _rotmat_to_quat_wxyz(racket_mat)

    policy_mask = trace.policy_mask[:count]
    if not np.any(policy_mask):
        policy_mask = np.ones(count, dtype=bool)
    actual_q = trace.actual_q[:count]
    actual_qd = trace.actual_qd[:count]
    position_error = sim_q[policy_mask] - actual_q[policy_mask]
    velocity_error = sim_qd[policy_mask] - actual_qd[policy_mask]
    actual_qdd, actual_qdd_valid = forward_velocity_acceleration(
        actual_qd,
        trace.joint_time[:count],
        trace.source_sequence[:count],
    )
    sim_qdd, sim_qdd_valid = forward_velocity_acceleration(
        sim_qd,
        trace.time[:count],
        trace.source_sequence[:count],
    )
    acceleration_mask = policy_mask & actual_qdd_valid & sim_qdd_valid
    if not np.any(acceleration_mask):
        raise ValueError(f"{record.recording_name} has no valid aligned acceleration samples")
    acceleration_error = sim_qdd[acceleration_mask] - actual_qdd[acceleration_mask]
    rmp_error = rmp_q[policy_mask] - actual_q[policy_mask]
    metrics = {
        "samples": count,
        "policy_samples": int(np.count_nonzero(policy_mask)),
        "acceleration_samples": int(np.count_nonzero(acceleration_mask)),
        "position_rms_rad": _rms(position_error),
        "position_rms_deg": float(np.degrees(_rms(position_error))),
        "position_rms_per_joint_rad": np.sqrt(np.mean(position_error**2, axis=0)).tolist(),
        "velocity_rms_rad_s": _rms(velocity_error),
        "velocity_rms_per_joint_rad_s": np.sqrt(np.mean(velocity_error**2, axis=0)).tolist(),
        "acceleration_rms_rad_s2": _rms(acceleration_error),
        "acceleration_rms_per_joint_rad_s2": np.sqrt(np.mean(acceleration_error**2, axis=0)).tolist(),
        "rmp_position_rms_rad": _rms(rmp_error),
        "maximum_position_error_rad": float(np.max(np.abs(position_error))),
    }
    arrays = {
        "time": trace.time[:count],
        "source_sequence": trace.source_sequence[:count],
        "joint_time": trace.joint_time[:count],
        "target_q": trace.target_q[:count],
        "actual_q": actual_q,
        "actual_qd": actual_qd,
        "actual_qdd": actual_qdd,
        "actual_qdd_raw": trace.actual_qdd[:count],
        "policy_mask": trace.policy_mask[:count],
        "rmp_q": rmp_q,
        "rmp_qd": rmp_qd,
        "rmp_qdd": rmp_qdd,
        "sim_q": sim_q,
        "sim_qd": sim_qd,
        "sim_qdd": sim_qdd,
        "sim_qacc": sim_qacc,
        "acceleration_valid_mask": acceleration_mask,
        "racket_pos": racket_pos,
        "racket_mat": racket_mat,
        "racket_vel": racket_vel,
        "racket_acc": racket_acc,
        "racket_omega_world": racket_omega_world,
        "racket_omega_local": racket_omega_local,
        "racket_alpha_world": racket_alpha_world,
        "racket_alpha_local": racket_alpha_local,
    }
    report = {
        "recording": str(trace.source),
        "group": record.group,
        "recording_name": record.recording_name,
        "control_chain": CONTROL_CHAIN,
        **record.phase_info,
        "actuator_model_enabled": False,
        "acceleration_measurement": ACCELERATION_MEASUREMENT,
        **config.report_fields(),
        "batch_index": int(batch_index),
        "environment_index": int(lane),
        "active_samples": count,
        "padded_batch_samples": int(padded_batch_samples if padded_batch_samples is not None else count),
        "racket_metrics": {
            "center_pos_m": racket_pos.tolist(),
            "center_quat_wxyz": racket_quat.tolist(),
            "center_rotmat_rowmajor": racket_mat.reshape(count, 9).tolist(),
            "center_vel_m_s": racket_vel.tolist(),
            "center_acc_m_s2": racket_acc.tolist(),
            "center_angular_velocity_world_rad_s": racket_omega_world.tolist(),
            "center_angular_velocity_local_rad_s": racket_omega_local.tolist(),
            "center_angular_acceleration_world_rad_s2": racket_alpha_world.tolist(),
            "center_angular_acceleration_local_rad_s2": racket_alpha_local.tolist(),
        },
        "metrics": metrics,
    }
    return arrays, report


def write_record_artifacts(output_dir: Path, arrays: dict[str, np.ndarray], report: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "rmp_mjx_replay_trace.npz"
    report_path = output_dir / "rmp_mjx_replay_report.json"
    np.savez_compressed(trace_path, **arrays)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return trace_path, report_path


def record_summary_row(
    report: dict[str, object],
    *,
    trace_path: Path,
    report_path: Path,
    output_dir: Path,
    status: str = "pass",
    reason: str = "",
) -> dict[str, object]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    racket = report.get("racket_metrics") if isinstance(report.get("racket_metrics"), dict) else {}
    return {
        "group": report.get("group", ""),
        "recording_name": report.get("recording_name", ""),
        "recording": report.get("recording", ""),
        "status": status,
        "reason": reason,
        "batch_index": report.get("batch_index", 0),
        "environment_index": report.get("environment_index", 0),
        "active_samples": report.get("active_samples", metrics.get("samples", 0)),
        "padded_batch_samples": report.get("padded_batch_samples", 0),
        "trace_dir": str(output_dir),
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "position_rms_rad": metrics.get("position_rms_rad"),
        "velocity_rms_rad_s": metrics.get("velocity_rms_rad_s"),
        "acceleration_rms_rad_s2": metrics.get("acceleration_rms_rad_s2"),
        "rmp_position_rms_rad": metrics.get("rmp_position_rms_rad"),
        "maximum_position_error_rad": metrics.get("maximum_position_error_rad"),
        "racket_metrics": {
            "available": all(key in racket for key in RACKET_METRIC_KEYS),
            "keys": sorted(racket.keys()),
        },
    }


def summarize_records(
    *,
    model: Path,
    record_root: Path,
    include_groups: Sequence[str],
    calibration_profile: str,
    output_root: Path,
    records: Sequence[dict[str, object]],
    discovered_count: int,
    loaded_count: int,
    batch_size_requested: int,
    vectorized_batch_count: int,
    max_padded_samples: int,
    total_active_samples: int,
    device_seconds: float,
    wall_seconds: float,
    env_dt: float | None,
) -> dict[str, object]:
    status_counts: dict[str, int] = {"pass": 0, "skipped": 0, "failed": 0, "pending": 0}
    for row in records:
        status = str(row.get("status", "pending"))
        status_counts[status] = status_counts.get(status, 0) + 1
    groups: dict[str, dict[str, object]] = {}
    for group in include_groups:
        rows = [row for row in records if row.get("group") == group]
        pass_rows = [row for row in rows if row.get("status") == "pass"]
        group_metrics: dict[str, object] = {
            "records": len(rows),
            "pass": len(pass_rows),
            "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
            "failed": sum(1 for row in rows if row.get("status") == "failed"),
        }
        for key in (
            "position_rms_rad",
            "velocity_rms_rad_s",
            "acceleration_rms_rad_s2",
            "maximum_position_error_rad",
        ):
            vals = [float(row[key]) for row in pass_rows if row.get(key) is not None]
            group_metrics[key] = float(np.mean(vals)) if vals else None
        groups[group] = group_metrics
    return {
        "model": str(model),
        "execution_mode": "vectorized_mjx",
        "record_root": str(record_root),
        "include_groups": list(include_groups),
        "calibration_profile": calibration_profile,
        "output_root": str(output_root),
        "batch_size_requested": int(batch_size_requested),
        "vectorized_batch_count": int(vectorized_batch_count),
        "count": len(records),
        "discovered_count": int(discovered_count),
        "loaded_count": int(loaded_count),
        "status_counts": status_counts,
        "max_padded_samples": int(max_padded_samples),
        "total_active_samples": int(total_active_samples),
        "device_seconds": float(device_seconds),
        "wall_seconds": float(wall_seconds),
        "mjx_control_dt_s": None if env_dt is None else float(env_dt),
        "groups": groups,
        "records": list(records),
    }


def _fmt_float(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{digits}g}"


def _md_row(items: Sequence[object]) -> str:
    return "| " + " | ".join(str(item) for item in items) + " |"


def render_markdown_report(summary: dict[str, object]) -> str:
    status_counts = summary.get("status_counts", {})
    groups = summary.get("groups", {})
    records = summary.get("records", [])
    lines = [
        "# Vectorized MJX RMP Replay Validation",
        "",
        "## Run configuration",
        f"- Execution mode: `{summary.get('execution_mode', 'unknown')}`",
        f"- XML model: `{summary.get('model', '')}`",
        f"- Calibration profile: `{summary.get('calibration_profile', '')}`",
        f"- Recording root: `{summary.get('record_root', '')}`",
        f"- Selected groups: {', '.join(summary.get('include_groups', []))}",
        f"- Batch size requested: {summary.get('batch_size_requested', 0)}",
        f"- Vectorized waves: {summary.get('vectorized_batch_count', 0)}",
        f"- MJX control timestep: {_fmt_float(summary.get('mjx_control_dt_s'))} s",
        f"- Device execution time: {_fmt_float(summary.get('device_seconds'))} s",
        f"- Wall time: {_fmt_float(summary.get('wall_seconds'))} s",
        "",
        "## Overall result",
        f"- Discovered recordings: {summary.get('discovered_count', 0)}",
        f"- Loaded recordings: {summary.get('loaded_count', 0)}",
        f"- Passed: {status_counts.get('pass', 0)}",
        f"- Skipped: {status_counts.get('skipped', 0)}",
        f"- Failed: {status_counts.get('failed', 0)}",
        f"- Total active samples: {summary.get('total_active_samples', 0)}",
        f"- Maximum padded batch length: {summary.get('max_padded_samples', 0)}",
        "",
        "## Group metrics",
        _md_row(["Group", "Records", "Pass", "Skip", "Fail", "q RMS rad", "dq RMS rad/s", "qdd RMS rad/s²", "max q error rad"]),
        _md_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]
    for group, data in groups.items():
        lines.append(_md_row([
            group,
            data.get("records", 0),
            data.get("pass", 0),
            data.get("skipped", 0),
            data.get("failed", 0),
            _fmt_float(data.get("position_rms_rad")),
            _fmt_float(data.get("velocity_rms_rad_s")),
            _fmt_float(data.get("acceleration_rms_rad_s2")),
            _fmt_float(data.get("maximum_position_error_rad")),
        ]))
    lines.extend([
        "",
        "## Per-record results",
        _md_row(["Group", "Recording", "Wave", "Lane", "Active samples", "q RMS rad", "dq RMS rad/s", "qdd RMS rad/s²", "max q error rad", "Status"]),
        _md_row(["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
    ])
    for row in records:
        lines.append(_md_row([
            row.get("group", ""),
            row.get("recording_name", ""),
            row.get("batch_index", ""),
            row.get("environment_index", ""),
            row.get("active_samples", ""),
            _fmt_float(row.get("position_rms_rad")),
            _fmt_float(row.get("velocity_rms_rad_s")),
            _fmt_float(row.get("acceleration_rms_rad_s2")),
            _fmt_float(row.get("maximum_position_error_rad")),
            row.get("status", ""),
        ]))
    missing = [row for row in records if row.get("status") == "pass" and not (row.get("racket_metrics") or {}).get("available")]
    lines.extend(["", "## Racket metric coverage"])
    if missing:
        lines.append("The following passing records are missing at least one expected racket metric block:")
        for row in missing:
            lines.append(f"- {row.get('group', '')}/{row.get('recording_name', '')}")
    else:
        lines.append("All passing reports include racket center position, orientation, linear velocity/acceleration, and angular velocity/acceleration metrics.")
    failed = [row for row in records if row.get("status") in {"failed", "skipped"}]
    lines.extend(["", "## Failed and skipped recordings"])
    if failed:
        lines.extend([
            _md_row(["Recording", "Status", "Reason"]),
            _md_row(["---", "---", "---"]),
        ])
        for row in failed:
            lines.append(_md_row([
                f"{row.get('group', '')}/{row.get('recording_name', '')}",
                row.get("status", ""),
                row.get("reason", ""),
            ]))
    else:
        lines.append("No failed or skipped recordings.")
    lines.append("")
    return "\n".join(lines)
