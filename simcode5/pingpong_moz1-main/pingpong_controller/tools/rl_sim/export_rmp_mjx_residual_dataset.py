#!/usr/bin/env python3
"""Export aligned real/final-RMP-MJX traces for residual identification."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import calibrate_rmp_pd_mjx as calibration
from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument(
        "--model", type=Path, default=Path(__file__).with_name("moz1_pd.xml")
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = tuple(calibration.load_control_recording(path) for path in args.recordings)
    n_envs = len(traces)
    maximum_length = max(trace.time.size for trace in traces)
    env = MjxJuggleEnv(
        args.model,
        n_envs=n_envs,
        cfg=MjxJuggleConfig(
            domain_randomization=False,
            arm_action_limiter=False,
            recovered_rmp_motion_mode=True,
            recovered_rmp_jnt_kp_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_RMP_KP
            ),
            recovered_rmp_jnt_kd_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_RMP_KD
            ),
            recovered_rmp_dof_damping_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_DAMPING
            ),
            recovered_rmp_dof_armature_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_ARMATURE
            ),
            recovered_rmp_controller_kp_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_KP
            ),
            recovered_rmp_controller_kd_scale_per_joint=tuple(
                float(value) for value in calibration.INITIAL_KD
            ),
            actuator_cmd_filter=False,
            actuator_compensation_mode="none",
            enable_delay_conditioning=False,
        ),
    )
    state, _ = env.reset(jax.random.split(jax.random.PRNGKey(31), n_envs))
    qpos = state.data.qpos.at[:, env.ball_qadr : env.ball_qadr + 3].set(
        jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 5.0], dtype=jnp.float32), (n_envs, 3)
        )
    )
    qvel = state.data.qvel.at[:, env.ball_vadr : env.ball_vadr + 6].set(0.0)
    data = env.batched_forward(state.model, state.data.replace(qpos=qpos, qvel=qvel))
    state = state._replace(data=data)
    initial_q = np.stack([trace.actual_q[0] for trace in traces])
    initial_qd = np.stack([trace.actual_qd[0] for trace in traces])
    initial_qdd = np.stack([trace.actual_qdd[0] for trace in traces])
    initial_target = np.stack([trace.target_q[0] for trace in traces])
    state = env.reset_arm_replay_state(
        state,
        jnp.asarray(initial_q, dtype=jnp.float32),
        measured_qd=jnp.asarray(initial_qd, dtype=jnp.float32),
        measured_qdd=jnp.asarray(initial_qdd, dtype=jnp.float32),
        position_target=jnp.asarray(initial_target, dtype=jnp.float32),
    )
    targets = np.empty((maximum_length, n_envs, 7), dtype=np.float32)
    for env_index, trace in enumerate(traces):
        targets[: trace.time.size, env_index] = trace.target_q
        targets[trace.time.size :, env_index] = trace.target_q[-1]

    def replay_scan(initial_state, target_history):
        def one_sample(replay_state, target):
            output = (
                replay_state.data.qpos[:, env.arm_qadr],
                replay_state.data.qvel[:, env.arm_vadr],
                replay_state.data.qacc[:, env.arm_vadr],
                replay_state.rmp_q,
                replay_state.rmp_qd,
                replay_state.rmp_qdd,
            )
            next_state, _, _, _, _ = env.step_q_ref(replay_state, target)
            return next_state, output

        return jax.lax.scan(one_sample, initial_state, target_history)

    state, values = jax.jit(replay_scan)(state, jnp.asarray(targets))
    jax.block_until_ready(state.data.qpos)
    arrays = tuple(np.asarray(value) for value in values)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        recording_names=np.asarray([trace.source.parent.name for trace in traces]),
        lengths=np.asarray([trace.time.size for trace in traces], dtype=np.int32),
        time=np.stack(
            [np.pad(trace.time, (0, maximum_length - trace.time.size)) for trace in traces]
        ),
        joint_time=np.stack(
            [
                np.pad(trace.joint_time, (0, maximum_length - trace.time.size))
                for trace in traces
            ]
        ),
        source_sequence=np.stack(
            [
                np.pad(trace.source_sequence, (0, maximum_length - trace.time.size))
                for trace in traces
            ]
        ),
        policy_mask=np.stack(
            [
                np.pad(trace.policy_mask, (0, maximum_length - trace.time.size))
                for trace in traces
            ]
        ),
        target_q=np.swapaxes(targets, 0, 1),
        actual_q=np.stack(
            [
                np.pad(
                    trace.actual_q,
                    ((0, maximum_length - trace.time.size), (0, 0)),
                    constant_values=np.nan,
                )
                for trace in traces
            ]
        ),
        actual_qd=np.stack(
            [
                np.pad(
                    trace.actual_qd,
                    ((0, maximum_length - trace.time.size), (0, 0)),
                    constant_values=np.nan,
                )
                for trace in traces
            ]
        ),
        sim_q=np.swapaxes(arrays[0], 0, 1),
        sim_qd=np.swapaxes(arrays[1], 0, 1),
        sim_qacc=np.swapaxes(arrays[2], 0, 1),
        rmp_q=np.swapaxes(arrays[3], 0, 1),
        rmp_qd=np.swapaxes(arrays[4], 0, 1),
        rmp_qdd=np.swapaxes(arrays[5], 0, 1),
    )
    print(f"exported {n_envs} recordings x {maximum_length} samples to {output}")


if __name__ == "__main__":
    main()
