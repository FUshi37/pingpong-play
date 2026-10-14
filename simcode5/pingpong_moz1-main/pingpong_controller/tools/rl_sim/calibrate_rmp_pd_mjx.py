#!/usr/bin/env python3
"""Refine seven low-level PD pairs directly in the full MJX replay path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import NamedTuple

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


DEFAULT_RECORD_ROOT = (
    WORKSPACE
    / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
)
DEFAULT_TRAIN = (
    DEFAULT_RECORD_ROOT
    / "record_new3/probe_recording_20260810_195203/control_samples.jsonl",
    DEFAULT_RECORD_ROOT
    / "record_new4/probe_recording_20260811_203653/control_samples.jsonl",
)
DEFAULT_VALIDATION = (
    DEFAULT_RECORD_ROOT
    / "record_new4/probe_recording_20260811_210018/control_samples.jsonl",
)

# Start from the high-bandwidth ideal-PD region identified against the 1 kHz
# DataTracer reference.  The former tiny values and non-unit RMP gains were fit
# before the runtime RMP wrapper was recovered and are intentionally retired.
INITIAL_KP = np.full(7, 3.0, dtype=np.float32)
INITIAL_KD = np.full(7, 10.0, dtype=np.float32)
INITIAL_RMP_KP = np.ones(7, dtype=np.float32)
INITIAL_RMP_KD = np.ones(7, dtype=np.float32)
INITIAL_DAMPING = np.ones(7, dtype=np.float32)
INITIAL_ARMATURE = np.ones(7, dtype=np.float32)
INITIAL_OUTPUT_DELAY = np.asarray((13, 13, 13, 14, 12, 12, 13), dtype=np.int32)
MAX_OUTPUT_DELAY_STEPS = 25


def _float_vector(text: str) -> np.ndarray:
    values = np.asarray([float(item) for item in text.split(",")], dtype=np.float32)
    if values.shape == (1,):
        values = np.full(7, values.item(), dtype=np.float32)
    if values.shape != (7,) or np.any(values <= 0.0):
        raise argparse.ArgumentTypeError(
            "expected one or seven comma-separated positive values"
        )
    return values


def _delay_vector(text: str) -> np.ndarray:
    values = np.asarray([int(item) for item in text.split(",")], dtype=np.int32)
    if values.shape == (1,):
        values = np.full(7, values.item(), dtype=np.int32)
    if values.shape != (7,) or np.any(values < 0) or np.any(values > MAX_OUTPUT_DELAY_STEPS):
        raise argparse.ArgumentTypeError(
            f"expected one or seven integers in [0, {MAX_OUTPUT_DELAY_STEPS}]"
        )
    return values


class ReplayBatch(NamedTuple):
    trace: RecordingTrace
    targets: jax.Array
    actual_q: np.ndarray
    actual_qd: np.ndarray
    actual_qdd: np.ndarray
    policy_mask: np.ndarray
    base_state: object


def _slice_policy_region(trace: RecordingTrace, warmup: int, cooldown: int) -> RecordingTrace:
    indices = np.flatnonzero(trace.policy_mask)
    if indices.size == 0:
        return trace
    start = max(0, int(indices[0]) - max(0, warmup))
    stop = min(trace.time.size, int(indices[-1]) + max(0, cooldown) + 1)
    selection = slice(start, stop)
    time = trace.time[selection].copy()
    time -= time[0]
    return RecordingTrace(
        source=trace.source,
        time=time,
        source_sequence=trace.source_sequence[selection].copy(),
        joint_time=trace.joint_time[selection].copy(),
        target_q=trace.target_q[selection].copy(),
        actual_q=trace.actual_q[selection].copy(),
        actual_qd=trace.actual_qd[selection].copy(),
        actual_qdd=trace.actual_qdd[selection].copy(),
        policy_mask=trace.policy_mask[selection].copy(),
        median_dt=trace.median_dt,
    )


def _make_replay_batch(
    env: MjxJuggleEnv,
    trace: RecordingTrace,
    reset_key: jax.Array,
) -> ReplayBatch:
    state, _ = env.reset(jax.random.split(reset_key, env.n_envs))
    qpos = state.data.qpos.at[:, env.ball_qadr : env.ball_qadr + 3].set(
        jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 5.0], dtype=jnp.float32),
            (env.n_envs, 3),
        )
    )
    qvel = state.data.qvel.at[:, env.ball_vadr : env.ball_vadr + 6].set(0.0)
    data = env.batched_forward(
        state.model, state.data.replace(qpos=qpos, qvel=qvel)
    )
    state = state._replace(data=data)
    return ReplayBatch(
        trace=trace,
        targets=jnp.asarray(trace.target_q, dtype=jnp.float32),
        actual_q=trace.actual_q,
        actual_qd=trace.actual_qd,
        actual_qdd=trace.actual_qdd,
        policy_mask=trace.policy_mask,
        base_state=state,
    )


def _make_scan(env: MjxJuggleEnv):
    def replay_scan(initial_state, targets):
        def one_sample(state, target):
            values = (
                state.data.qpos[:, env.arm_qadr],
                state.data.qvel[:, env.arm_vadr],
                state.data.qacc[:, env.arm_vadr],
            )
            next_state, _, _, _, _ = env.step_q_ref(state, target)
            return next_state, values

        return jax.lax.scan(one_sample, initial_state, targets)

    return jax.jit(replay_scan)


def _run_candidates(
    env: MjxJuggleEnv,
    replay_scan,
    batch: ReplayBatch,
    kp_candidates: np.ndarray,
    kd_candidates: np.ndarray,
    rmp_kp_candidates: np.ndarray,
    rmp_kd_candidates: np.ndarray,
    damping_candidates: np.ndarray,
    armature_candidates: np.ndarray,
    output_delay_candidates: np.ndarray | None = None,
    target_delay_candidates: np.ndarray | None = None,
    rmp_internal_candidates: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = env.set_arm_pd_scales(
        batch.base_state,
        jnp.asarray(kp_candidates, dtype=jnp.float32),
        jnp.asarray(kd_candidates, dtype=jnp.float32),
    )
    state = env.set_rmp_joint_gain_scales(
        state,
        jnp.asarray(rmp_kp_candidates, dtype=jnp.float32),
        jnp.asarray(rmp_kd_candidates, dtype=jnp.float32),
    )
    state = env.set_arm_dynamics_scales(
        state,
        jnp.asarray(damping_candidates, dtype=jnp.float32),
        jnp.asarray(armature_candidates, dtype=jnp.float32),
    )
    if output_delay_candidates is not None:
        state = env.set_rmp_output_delay_steps(
            state,
            jnp.asarray(output_delay_candidates, dtype=jnp.int32),
        )
    if rmp_internal_candidates is not None:
        if len(rmp_internal_candidates) != 6:
            raise ValueError("rmp_internal_candidates must contain six arrays")
        state = env.set_rmp_internal_parameters(
            state,
            *(jnp.asarray(value) for value in rmp_internal_candidates),
        )
    state = env.reset_arm_replay_state(
        state,
        jnp.broadcast_to(
            jnp.asarray(batch.actual_q[0], dtype=jnp.float32),
            (env.n_envs, env.act_dim),
        ),
        measured_qd=jnp.broadcast_to(
            jnp.asarray(batch.actual_qd[0], dtype=jnp.float32),
            (env.n_envs, env.act_dim),
        ),
        measured_qdd=jnp.broadcast_to(
            jnp.asarray(batch.actual_qdd[0], dtype=jnp.float32),
            (env.n_envs, env.act_dim),
        ),
        position_target=jnp.broadcast_to(
            jnp.asarray(batch.trace.target_q[0], dtype=jnp.float32),
            (env.n_envs, env.act_dim),
        ),
    )
    if target_delay_candidates is None:
        targets = jnp.broadcast_to(
            batch.targets[:, None, :],
            (batch.targets.shape[0], env.n_envs, env.act_dim),
        )
    else:
        delays = jnp.asarray(target_delay_candidates, dtype=jnp.int32)
        expected_shape = (env.n_envs, env.act_dim)
        if delays.shape != expected_shape:
            raise ValueError(
                f"target delays must have shape {expected_shape}, got {delays.shape}"
            )
        sample_indices = jnp.maximum(
            0,
            jnp.arange(batch.targets.shape[0])[:, None, None]
            - delays[None, :, :],
        )
        joint_indices = jnp.arange(env.act_dim)[None, None, :]
        targets = batch.targets[sample_indices, joint_indices]
    final_state, values = replay_scan(state, targets)
    jax.block_until_ready(final_state.data.qpos)
    return tuple(np.asarray(value) for value in values)


def _candidate_losses(
    outputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch: ReplayBatch,
    weights: tuple[float, float, float],
    tolerances: tuple[float, float, float],
) -> np.ndarray:
    mask = batch.policy_mask
    if not np.any(mask):
        mask = np.ones(batch.targets.shape[0], dtype=bool)
    losses = np.zeros(outputs[0].shape[1], dtype=np.float64)
    for weight, tolerance, simulated, actual in (
        (weights[0], tolerances[0], outputs[0], batch.actual_q),
        (weights[1], tolerances[1], outputs[1], batch.actual_qd),
    ):
        error = simulated[mask] - actual[mask, None, :]
        losses += weight * np.mean(error**2, axis=(0, 2)) / tolerance**2
    actual_qdd, actual_valid = forward_velocity_acceleration(
        batch.actual_qd, batch.trace.joint_time, batch.trace.source_sequence
    )
    sample_count, candidate_count, joint_count = outputs[1].shape
    sim_qdd_flat, sim_valid = forward_velocity_acceleration(
        outputs[1].reshape(sample_count, candidate_count * joint_count),
        batch.trace.time,
        batch.trace.source_sequence,
    )
    sim_qdd = sim_qdd_flat.reshape(outputs[1].shape)
    acceleration_mask = mask & actual_valid & sim_valid
    qdd_error = (
        sim_qdd[acceleration_mask]
        - actual_qdd[acceleration_mask, None, :]
    )
    losses += (
        weights[2]
        * np.mean(qdd_error**2, axis=(0, 2))
        / tolerances[2] ** 2
    )
    losses[~np.isfinite(losses)] = 1.0e12
    return losses


def _metrics(
    outputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch: ReplayBatch,
    candidate_index: int,
) -> dict[str, object]:
    mask = batch.policy_mask
    values: dict[str, object] = {}
    for name, simulated, actual in (
        ("q", outputs[0], batch.actual_q),
        ("qd", outputs[1], batch.actual_qd),
    ):
        error = simulated[mask, candidate_index, :] - actual[mask]
        values[f"{name}_rms"] = float(np.sqrt(np.mean(error**2)))
        values[f"{name}_rms_per_joint"] = np.sqrt(
            np.mean(error**2, axis=0)
        ).tolist()
    actual_qdd, actual_valid = forward_velocity_acceleration(
        batch.actual_qd, batch.trace.joint_time, batch.trace.source_sequence
    )
    sim_qdd, sim_valid = forward_velocity_acceleration(
        outputs[1][:, candidate_index, :],
        batch.trace.time,
        batch.trace.source_sequence,
    )
    acceleration_mask = mask & actual_valid & sim_valid
    qdd_error = sim_qdd[acceleration_mask] - actual_qdd[acceleration_mask]
    values["qdd_rms"] = float(np.sqrt(np.mean(qdd_error**2)))
    values["qdd_rms_per_joint"] = np.sqrt(
        np.mean(qdd_error**2, axis=0)
    ).tolist()
    values["qdd_samples"] = int(np.count_nonzero(acceleration_mask))
    native_qacc_error = (
        outputs[2][acceleration_mask, candidate_index, :]
        - actual_qdd[acceleration_mask]
    )
    values["native_qacc_rms_diagnostic"] = float(
        np.sqrt(np.mean(native_qacc_error**2))
    )
    return values


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", default=list(DEFAULT_TRAIN))
    parser.add_argument("--validation", type=Path, nargs="+", default=list(DEFAULT_VALIDATION))
    parser.add_argument("--model", type=Path, default=here / "moz1_pd.xml")
    parser.add_argument("--output", type=Path, default=Path("rmp_pd_mjx_calibration.json"))
    parser.add_argument("--warmup-samples", type=int, default=100)
    parser.add_argument("--cooldown-samples", type=int, default=20)
    parser.add_argument("--joint-sweeps", type=int, default=2)
    parser.add_argument("--delay-joint-sweeps", type=int, default=2)
    parser.add_argument("--rmp-sweeps", type=int, default=0)
    parser.add_argument("--rmp-joint-sweeps", type=int, default=0)
    parser.add_argument("--dynamics-joint-sweeps", type=int, default=2)
    parser.add_argument("--fine-search", action="store_true")
    parser.add_argument(
        "--initial-controller-kp", type=_float_vector,
        help="one or seven XML position-PD Kp scales",
    )
    parser.add_argument(
        "--initial-controller-kd", type=_float_vector,
        help="one or seven XML position-PD Kv scales",
    )
    parser.add_argument(
        "--initial-damping", type=_float_vector,
        help="one or seven MuJoCo dof-damping scales",
    )
    parser.add_argument(
        "--initial-armature", type=_float_vector,
        help="one or seven MuJoCo dof-armature scales",
    )
    parser.add_argument(
        "--initial-output-delay", type=_delay_vector,
        help="one or seven 1 kHz RMP-output-to-PD delays",
    )
    parser.add_argument("--q-weight", type=float, default=0.40)
    parser.add_argument("--qd-weight", type=float, default=0.30)
    parser.add_argument("--qdd-weight", type=float, default=0.30)
    parser.add_argument(
        "--q-tolerance", type=float, default=float(np.deg2rad(0.1))
    )
    parser.add_argument("--qd-tolerance", type=float, default=0.05)
    parser.add_argument("--qdd-tolerance", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_raw = (args.q_weight, args.qd_weight, args.qdd_weight)
    total_weight = sum(weights_raw)
    weights = tuple(float(value / total_weight) for value in weights_raw)
    tolerances = (args.q_tolerance, args.qd_tolerance, args.qdd_tolerance)
    factors_by_sweep = (
        (0.58, 0.78, 1.0, 1.22, 1.50),
        (0.78, 0.92, 1.0, 1.10, 1.28),
    )
    if args.fine_search:
        factors_by_sweep = (
            (0.78, 0.92, 1.0, 1.10, 1.28),
            (0.90, 0.97, 1.0, 1.04, 1.12),
        )
    candidate_count = 1 + 5 * 5
    env = MjxJuggleEnv(
        args.model,
        n_envs=candidate_count,
        cfg=MjxJuggleConfig(
            domain_randomization=False,
            arm_action_limiter=False,
            recovered_rmp_motion_mode=True,
            recovered_rmp_jnt_kp_scale=1.0,
            recovered_rmp_jnt_kd_scale=1.0,
            recovered_rmp_target_update_delay_steps=3,
            # Allocate enough history for all delay candidates.  Every replay
            # state receives its actual candidate values below.
            recovered_rmp_output_delay_steps_per_joint=(
                MAX_OUTPUT_DELAY_STEPS,
            ) * 7,
            actuator_cmd_filter=False,
            actuator_compensation_mode="none",
            enable_delay_conditioning=False,
        ),
    )
    replay_scan = _make_scan(env)
    train_traces = tuple(
        _slice_policy_region(
            load_control_recording(path), args.warmup_samples, args.cooldown_samples
        )
        for path in args.train
    )
    validation_traces = tuple(
        _slice_policy_region(
            load_control_recording(path), args.warmup_samples, args.cooldown_samples
        )
        for path in args.validation
    )
    keys = jax.random.split(
        jax.random.PRNGKey(19), len(train_traces) + len(validation_traces)
    )
    train_batches = tuple(
        _make_replay_batch(env, trace, keys[index])
        for index, trace in enumerate(train_traces)
    )
    validation_batches = tuple(
        _make_replay_batch(env, trace, keys[len(train_traces) + index])
        for index, trace in enumerate(validation_traces)
    )

    best_kp = (
        INITIAL_KP.copy()
        if args.initial_controller_kp is None
        else args.initial_controller_kp.copy()
    )
    best_kd = (
        INITIAL_KD.copy()
        if args.initial_controller_kd is None
        else args.initial_controller_kd.copy()
    )
    best_rmp_kp = INITIAL_RMP_KP.copy()
    best_rmp_kd = INITIAL_RMP_KD.copy()
    best_damping = (
        INITIAL_DAMPING.copy()
        if args.initial_damping is None
        else args.initial_damping.copy()
    )
    best_armature = (
        INITIAL_ARMATURE.copy()
        if args.initial_armature is None
        else args.initial_armature.copy()
    )
    best_output_delay = (
        INITIAL_OUTPUT_DELAY.copy()
        if args.initial_output_delay is None
        else args.initial_output_delay.copy()
    )
    best_loss = np.inf
    total_sweeps = max(
        int(args.joint_sweeps),
        int(args.rmp_sweeps),
        int(args.rmp_joint_sweeps),
        int(args.dynamics_joint_sweeps),
        int(args.delay_joint_sweeps),
        0,
    )
    for sweep in range(total_sweeps):
        factors = factors_by_sweep[min(sweep, len(factors_by_sweep) - 1)]
        if sweep < max(0, int(args.delay_joint_sweeps)):
            delay_offsets = (-6, -3, -1, 0, 1, 3, 6)
            for joint in range(7):
                kp_candidates = np.broadcast_to(
                    best_kp, (candidate_count, 7)
                ).copy()
                kd_candidates = np.broadcast_to(
                    best_kd, (candidate_count, 7)
                ).copy()
                rmp_kp_candidates = np.broadcast_to(
                    best_rmp_kp, (candidate_count, 7)
                ).copy()
                rmp_kd_candidates = np.broadcast_to(
                    best_rmp_kd, (candidate_count, 7)
                ).copy()
                damping_candidates = np.broadcast_to(
                    best_damping, (candidate_count, 7)
                ).copy()
                armature_candidates = np.broadcast_to(
                    best_armature, (candidate_count, 7)
                ).copy()
                output_delay_candidates = np.broadcast_to(
                    best_output_delay, (candidate_count, 7)
                ).copy()
                for candidate_index, offset in enumerate(delay_offsets, start=1):
                    output_delay_candidates[candidate_index, joint] = np.clip(
                        best_output_delay[joint] + offset,
                        0,
                        MAX_OUTPUT_DELAY_STEPS,
                    )
                total_losses = np.zeros(candidate_count, dtype=np.float64)
                for batch in train_batches:
                    outputs = _run_candidates(
                        env,
                        replay_scan,
                        batch,
                        kp_candidates,
                        kd_candidates,
                        rmp_kp_candidates,
                        rmp_kd_candidates,
                        damping_candidates,
                        armature_candidates,
                        output_delay_candidates=output_delay_candidates,
                    )
                    total_losses += _candidate_losses(
                        outputs, batch, weights, tolerances
                    )
                total_losses /= len(train_batches)
                selected = int(np.argmin(total_losses))
                best_output_delay = output_delay_candidates[selected].copy()
                best_loss = float(total_losses[selected])
                print(
                    f"sweep={sweep + 1} delay_joint={joint} "
                    f"loss={best_loss:.6f} candidate={selected} "
                    f"delay={best_output_delay[joint]}",
                    flush=True,
                )
        if sweep < max(0, int(args.rmp_sweeps)):
            kp_candidates = np.broadcast_to(best_kp, (candidate_count, 7)).copy()
            kd_candidates = np.broadcast_to(best_kd, (candidate_count, 7)).copy()
            rmp_kp_candidates = np.full(
                (candidate_count, 7), 1.0, dtype=np.float32
            )
            rmp_kd_candidates = np.full(
                (candidate_count, 7), 1.0, dtype=np.float32
            )
            rmp_kp_candidates *= best_rmp_kp[None, :]
            rmp_kd_candidates *= best_rmp_kd[None, :]
            damping_candidates = np.broadcast_to(
                best_damping, (candidate_count, 7)
            ).copy()
            armature_candidates = np.broadcast_to(
                best_armature, (candidate_count, 7)
            ).copy()
            candidate_index = 1
            for kp_factor in factors:
                for kd_factor in factors:
                    rmp_kp_candidates[candidate_index] *= kp_factor
                    rmp_kd_candidates[candidate_index] *= kd_factor
                    candidate_index += 1
            total_losses = np.zeros(candidate_count, dtype=np.float64)
            for batch in train_batches:
                outputs = _run_candidates(
                    env,
                    replay_scan,
                    batch,
                    kp_candidates,
                    kd_candidates,
                    rmp_kp_candidates,
                    rmp_kd_candidates,
                    damping_candidates,
                    armature_candidates,
                    output_delay_candidates=np.broadcast_to(
                        best_output_delay, (candidate_count, 7)
                    ),
                )
                total_losses += _candidate_losses(
                    outputs, batch, weights, tolerances
                )
            total_losses /= len(train_batches)
            selected = int(np.argmin(total_losses))
            best_rmp_kp = rmp_kp_candidates[selected].copy()
            best_rmp_kd = rmp_kd_candidates[selected].copy()
            best_loss = float(total_losses[selected])
            print(
                f"sweep={sweep + 1} rmp loss={best_loss:.6f} "
                f"candidate={selected}",
                flush=True,
            )
        if sweep < max(0, int(args.rmp_joint_sweeps)):
            for joint in range(7):
                kp_candidates = np.broadcast_to(
                    best_kp, (candidate_count, 7)
                ).copy()
                kd_candidates = np.broadcast_to(
                    best_kd, (candidate_count, 7)
                ).copy()
                rmp_kp_candidates = np.broadcast_to(
                    best_rmp_kp, (candidate_count, 7)
                ).copy()
                rmp_kd_candidates = np.broadcast_to(
                    best_rmp_kd, (candidate_count, 7)
                ).copy()
                damping_candidates = np.broadcast_to(
                    best_damping, (candidate_count, 7)
                ).copy()
                armature_candidates = np.broadcast_to(
                    best_armature, (candidate_count, 7)
                ).copy()
                candidate_index = 1
                for kp_factor in factors:
                    for kd_factor in factors:
                        rmp_kp_candidates[candidate_index, joint] *= kp_factor
                        rmp_kd_candidates[candidate_index, joint] *= kd_factor
                        candidate_index += 1
                total_losses = np.zeros(candidate_count, dtype=np.float64)
                for batch in train_batches:
                    outputs = _run_candidates(
                        env,
                        replay_scan,
                        batch,
                        kp_candidates,
                        kd_candidates,
                        rmp_kp_candidates,
                        rmp_kd_candidates,
                        damping_candidates,
                        armature_candidates,
                        output_delay_candidates=np.broadcast_to(
                            best_output_delay, (candidate_count, 7)
                        ),
                    )
                    total_losses += _candidate_losses(
                        outputs, batch, weights, tolerances
                    )
                total_losses /= len(train_batches)
                selected = int(np.argmin(total_losses))
                best_rmp_kp = rmp_kp_candidates[selected].copy()
                best_rmp_kd = rmp_kd_candidates[selected].copy()
                best_loss = float(total_losses[selected])
                print(
                    f"sweep={sweep + 1} rmp_joint={joint} "
                    f"loss={best_loss:.6f} candidate={selected}",
                    flush=True,
                )
        if sweep < max(0, int(args.dynamics_joint_sweeps)):
            for joint in range(7):
                kp_candidates = np.broadcast_to(
                    best_kp, (candidate_count, 7)
                ).copy()
                kd_candidates = np.broadcast_to(
                    best_kd, (candidate_count, 7)
                ).copy()
                rmp_kp_candidates = np.broadcast_to(
                    best_rmp_kp, (candidate_count, 7)
                ).copy()
                rmp_kd_candidates = np.broadcast_to(
                    best_rmp_kd, (candidate_count, 7)
                ).copy()
                damping_candidates = np.broadcast_to(
                    best_damping, (candidate_count, 7)
                ).copy()
                armature_candidates = np.broadcast_to(
                    best_armature, (candidate_count, 7)
                ).copy()
                candidate_index = 1
                for damping_factor in factors:
                    for armature_factor in factors:
                        damping_candidates[candidate_index, joint] *= damping_factor
                        armature_candidates[candidate_index, joint] *= armature_factor
                        candidate_index += 1
                total_losses = np.zeros(candidate_count, dtype=np.float64)
                for batch in train_batches:
                    outputs = _run_candidates(
                        env,
                        replay_scan,
                        batch,
                        kp_candidates,
                        kd_candidates,
                        rmp_kp_candidates,
                        rmp_kd_candidates,
                        damping_candidates,
                        armature_candidates,
                        output_delay_candidates=np.broadcast_to(
                            best_output_delay, (candidate_count, 7)
                        ),
                    )
                    total_losses += _candidate_losses(
                        outputs, batch, weights, tolerances
                    )
                total_losses /= len(train_batches)
                selected = int(np.argmin(total_losses))
                best_damping = damping_candidates[selected].copy()
                best_armature = armature_candidates[selected].copy()
                best_loss = float(total_losses[selected])
                print(
                    f"sweep={sweep + 1} dynamics_joint={joint} "
                    f"loss={best_loss:.6f} candidate={selected}",
                    flush=True,
                )
        if sweep >= max(0, int(args.joint_sweeps)):
            continue
        for joint in range(7):
            kp_candidates = np.broadcast_to(best_kp, (candidate_count, 7)).copy()
            kd_candidates = np.broadcast_to(best_kd, (candidate_count, 7)).copy()
            rmp_kp_candidates = np.full(
                (candidate_count, 7), 1.0, dtype=np.float32
            )
            rmp_kd_candidates = np.full(
                (candidate_count, 7), 1.0, dtype=np.float32
            )
            rmp_kp_candidates *= best_rmp_kp[None, :]
            rmp_kd_candidates *= best_rmp_kd[None, :]
            damping_candidates = np.broadcast_to(
                best_damping, (candidate_count, 7)
            ).copy()
            armature_candidates = np.broadcast_to(
                best_armature, (candidate_count, 7)
            ).copy()
            candidate_index = 1
            for kp_factor in factors:
                for kd_factor in factors:
                    kp_candidates[candidate_index, joint] *= kp_factor
                    kd_candidates[candidate_index, joint] *= kd_factor
                    candidate_index += 1
            total_losses = np.zeros(candidate_count, dtype=np.float64)
            for batch in train_batches:
                outputs = _run_candidates(
                    env,
                    replay_scan,
                    batch,
                    kp_candidates,
                    kd_candidates,
                    rmp_kp_candidates,
                    rmp_kd_candidates,
                    damping_candidates,
                    armature_candidates,
                    output_delay_candidates=np.broadcast_to(
                        best_output_delay, (candidate_count, 7)
                    ),
                )
                total_losses += _candidate_losses(
                    outputs, batch, weights, tolerances
                )
            total_losses /= len(train_batches)
            selected = int(np.argmin(total_losses))
            best_kp = kp_candidates[selected]
            best_kd = kd_candidates[selected]
            best_loss = float(total_losses[selected])
            print(
                f"sweep={sweep + 1} joint={joint} loss={best_loss:.6f} "
                f"candidate={selected}",
                flush=True,
            )

    repeated_kp = np.broadcast_to(best_kp, (candidate_count, 7)).copy()
    repeated_kd = np.broadcast_to(best_kd, (candidate_count, 7)).copy()
    repeated_rmp_kp = np.full(
        (candidate_count, 7), 1.0, dtype=np.float32
    )
    repeated_rmp_kd = np.full(
        (candidate_count, 7), 1.0, dtype=np.float32
    )
    repeated_rmp_kp *= best_rmp_kp[None, :]
    repeated_rmp_kd *= best_rmp_kd[None, :]
    repeated_damping = np.broadcast_to(
        best_damping, (candidate_count, 7)
    ).copy()
    repeated_armature = np.broadcast_to(
        best_armature, (candidate_count, 7)
    ).copy()
    details: dict[str, object] = {}
    for batch in (*train_batches, *validation_batches):
        outputs = _run_candidates(
            env,
            replay_scan,
            batch,
            repeated_kp,
            repeated_kd,
            repeated_rmp_kp,
            repeated_rmp_kd,
            repeated_damping,
            repeated_armature,
            output_delay_candidates=np.broadcast_to(
                best_output_delay, (candidate_count, 7)
            ),
        )
        details[batch.trace.source.parent.name] = _metrics(outputs, batch, 0)
    report = {
        "objective_loss": best_loss,
        "weights": {"q": weights[0], "qd": weights[1], "qdd": weights[2]},
        "tolerances": {"q": tolerances[0], "qd": tolerances[1], "qdd": tolerances[2]},
        "rmp_kp_scale_per_joint": best_rmp_kp.tolist(),
        "rmp_kd_scale_per_joint": best_rmp_kd.tolist(),
        "dof_damping_scale_per_joint": best_damping.tolist(),
        "dof_armature_scale_per_joint": best_armature.tolist(),
        "rmp_output_delay_steps_per_joint": best_output_delay.tolist(),
        "controller_kp_scale_per_joint": best_kp.tolist(),
        "controller_kd_scale_per_joint": best_kd.tolist(),
        "metrics": details,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {output}")


if __name__ == "__main__":
    main()
