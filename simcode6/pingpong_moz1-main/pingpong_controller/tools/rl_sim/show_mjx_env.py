"""Visualize the MJX Stage 1a environment initialization and zero-action motion."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np

RL_SIM_DIR = Path(__file__).resolve().parent
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv


def copy_env0_to_mujoco_data(env: MjxJuggleEnv, env_state, data: mj.MjData) -> None:
    data.qpos[:] = np.asarray(jax.device_get(env_state.data.qpos[0]))
    data.qvel[:] = np.asarray(jax.device_get(env_state.data.qvel[0]))
    data.ctrl[:] = np.asarray(jax.device_get(env_state.data.ctrl[0]))
    mj.mj_forward(env.mj_model, data)


def print_state(env: MjxJuggleEnv, env_state, label: str) -> None:
    data = env_state.data
    ball = np.asarray(jax.device_get(data.xpos[0, env.ball_body_id]))
    racket = np.asarray(jax.device_get(data.site_xpos[0, env.racket_site_id]))
    base = np.asarray(
        jax.device_get(
            jnp.asarray(
                [
                    data.qpos[0, env.base_x_qadr],
                    data.qpos[0, env.base_y_qadr],
                    data.qpos[0, env.base_yaw_qadr],
                ]
            )
        )
    )
    arm_q = np.asarray(jax.device_get(data.qpos[0, env.arm_qadr]))
    print(f"[show_mjx_env] {label}")
    print(f"  base[x,y,yaw] = {base}")
    print(f"  racket = {racket}")
    print(f"  ball = {ball}")
    print(f"  ball-racket = {ball - racket}")
    print(f"  right_arm_q(rad) = {arm_q}")


def main() -> None:
    p = argparse.ArgumentParser(description="Show MJX env init and zero-action behavior.")
    p.add_argument("--xml", type=Path, default=RL_SIM_DIR / "moz1_pd.xml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--pause-initial-sec", type=float, default=2.0)
    p.add_argument("--headless", action="store_true", help="Print init state and exit without viewer.")
    p.add_argument("--realtime", action="store_true", default=True)
    p.add_argument("--slowmo", type=float, default=1.0)
    args = p.parse_args()

    env = MjxJuggleEnv(args.xml, n_envs=1, cfg=MjxJuggleConfig())
    key = jax.random.PRNGKey(args.seed)[None, :]
    env_state, _ = jax.jit(env.reset)(key)
    print(f"[show_mjx_env] JAX devices: {jax.devices()}")
    print(f"[show_mjx_env] XML: {args.xml}")
    print(f"[show_mjx_env] MJX XML: {env.mjx_xml}")
    print(f"[show_mjx_env] dt={env.dt:.4f}, max_steps={env.max_steps}")
    print_state(env, env_state, "initial")

    if args.headless:
        return

    import mujoco.viewer

    render_data = mj.MjData(env.mj_model)
    copy_env0_to_mujoco_data(env, env_state, render_data)
    zero_action = jnp.zeros((1, env.act_dim), dtype=jnp.float32)
    step_fn = jax.jit(env.step)

    with mujoco.viewer.launch_passive(env.mj_model, render_data) as viewer:
        if args.pause_initial_sec > 0:
            t_end = time.time() + float(args.pause_initial_sec)
            while viewer.is_running() and time.time() < t_end:
                viewer.sync()
                time.sleep(0.03)

        for step_idx in range(int(args.steps)):
            if not viewer.is_running():
                break
            env_state, _, _, done, _ = step_fn(env_state, zero_action)
            copy_env0_to_mujoco_data(env, env_state, render_data)
            viewer.sync()
            if args.realtime:
                time.sleep(max(0.0, env.dt * float(args.slowmo)))
            if bool(np.asarray(jax.device_get(done[0]))):
                print_state(env, env_state, f"done at step {step_idx + 1}")
                break


if __name__ == "__main__":
    main()

