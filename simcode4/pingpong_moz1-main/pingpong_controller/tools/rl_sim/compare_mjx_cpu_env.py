"""Compare CPU MuJoCo/Gym and MJX/JAX Stage 1a environment behavior.

This is a migration guardrail.  It does not train; it runs both environments
from the same Stage 1a-style reset and action sequence, then reports numerical
drift in observations, ball/racket positions, rewards, and termination flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

RL_SIM_DIR = Path(__file__).resolve().parent
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv
from rl_juggle_env_random import JuggleConfig, JuggleEnv


def make_cpu_stage1a_cfg() -> JuggleConfig:
    return JuggleConfig(
        domain_randomization=False,
        frame_skip=5,
        action_acc_scale=1.5,
        ball_launch_height=0.30,
        ball_spawn_cube_size=0.0,
        ball_spawn_xy_jitter=0.0,
        ball_spawn_z_jitter=0.0,
        ball_init_vxy_max=0.0,
        ball_init_vz=-0.28,
        ball_obs_rate_hz=100.0,
        ball_obs_pos_noise_std=0.0,
        ball_obs_vel_noise_std=0.0,
        ball_obs_dropout_prob=0.0,
        ball_obs_dropout_max_steps=1,
        ball_obs_dropout_burst_prob=0.0,
        ball_obs_dropout_burst_max_steps=1,
        ball_obs_age_clip=0.20,
        target_height=0.34,
        posture_weight=0.02,
        base_pose_weight=0.0,
        ball_base_x_penalty_weight=0.0,
        ball_base_x_soft_limit=0.20,
        ball_base_vxy_penalty_weight=0.0,
        torque_penalty_weight=0.00005,
        arm_action_limiter=True,
        arm_vel_limit_deg_s=(210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0),
        arm_acc_limit_deg_s2=(1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0),
        arm_vel_limit_penalty_weight=0.0,
        arm_acc_limit_penalty_weight=0.002,
        arm_limiter_penalty_weight=0.0,
    )


def get_mjx_positions(env: MjxJuggleEnv, state) -> tuple[np.ndarray, np.ndarray]:
    data = state.data
    ball = np.asarray(jax.device_get(data.xpos[0, env.ball_body_id]))
    racket = np.asarray(jax.device_get(data.site_xpos[0, env.racket_site_id]))
    return ball, racket


def max_abs(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare CPU MuJoCo env with MJX/JAX env on Stage 1a.")
    p.add_argument("--xml", type=Path, default=RL_SIM_DIR / "moz1_pd.xml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--random-actions", action="store_true", help="Use seeded random actions instead of zero actions.")
    p.add_argument("--action-scale", type=float, default=0.25, help="Scale for random actions before clipping.")
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--strict", action="store_true", help="Exit nonzero if tolerances are exceeded.")
    p.add_argument("--obs-tol", type=float, default=1e-3)
    p.add_argument("--pos-tol", type=float, default=2e-3)
    p.add_argument("--reward-tol", type=float, default=5e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cpu_env = JuggleEnv(xml_path=str(args.xml), cfg=make_cpu_stage1a_cfg())
    cpu_obs, _ = cpu_env.reset(seed=args.seed)

    mjx_env = MjxJuggleEnv(args.xml, n_envs=1, cfg=MjxJuggleConfig())
    key = jax.random.PRNGKey(args.seed)[None, :]
    mjx_state, mjx_obs = jax.jit(mjx_env.reset)(key)
    mjx_step = jax.jit(mjx_env.step)

    cpu_ball = cpu_env._ball_pos()
    cpu_racket = cpu_env._racket_pos()
    mjx_ball, mjx_racket = get_mjx_positions(mjx_env, mjx_state)
    print(f"[compare] JAX devices: {jax.devices()}")
    print(f"[compare] CPU obs dim={cpu_obs.shape[0]}, MJX obs dim={mjx_obs.shape[-1]}")
    print(f"[compare] init ball delta   : {cpu_ball - mjx_ball}")
    print(f"[compare] init racket delta : {cpu_racket - mjx_racket}")
    print(f"[compare] init obs max_abs  : {max_abs(cpu_obs - np.asarray(jax.device_get(mjx_obs[0]))):.6g}")

    maxima = {
        "obs": 0.0,
        "ball": 0.0,
        "racket": 0.0,
        "reward": 0.0,
        "done_mismatch": 0.0,
    }

    for step in range(1, int(args.steps) + 1):
        if args.random_actions:
            action = np.clip(rng.normal(size=cpu_env.action_space.shape).astype(np.float32) * float(args.action_scale), -1.0, 1.0)
        else:
            action = np.zeros(cpu_env.action_space.shape, dtype=np.float32)

        cpu_obs, cpu_reward, cpu_terminated, cpu_truncated, _ = cpu_env.step(action)
        mjx_state, mjx_obs, mjx_reward, mjx_done, _ = mjx_step(mjx_state, jnp.asarray(action[None, :], dtype=jnp.float32))

        cpu_ball = cpu_env._ball_pos()
        cpu_racket = cpu_env._racket_pos()
        mjx_ball, mjx_racket = get_mjx_positions(mjx_env, mjx_state)
        obs_err = max_abs(cpu_obs - np.asarray(jax.device_get(mjx_obs[0])))
        ball_err = max_abs(cpu_ball - mjx_ball)
        racket_err = max_abs(cpu_racket - mjx_racket)
        reward_err = abs(float(cpu_reward) - float(jax.device_get(mjx_reward[0])))
        done_err = float(bool(cpu_terminated or cpu_truncated) != bool(jax.device_get(mjx_done[0])))

        maxima["obs"] = max(maxima["obs"], obs_err)
        maxima["ball"] = max(maxima["ball"], ball_err)
        maxima["racket"] = max(maxima["racket"], racket_err)
        maxima["reward"] = max(maxima["reward"], reward_err)
        maxima["done_mismatch"] = max(maxima["done_mismatch"], done_err)

        if args.print_every > 0 and step % int(args.print_every) == 0:
            print(
                f"[compare] step={step} "
                f"obs={obs_err:.3g} ball={ball_err:.3g} racket={racket_err:.3g} "
                f"reward={reward_err:.3g} done_mismatch={done_err:.0f}"
            )

        if cpu_terminated or cpu_truncated or bool(jax.device_get(mjx_done[0])):
            print(f"[compare] stopped at step={step} because at least one env ended")
            break

    print("[compare] maxima:")
    for key_name, value in maxima.items():
        print(f"  {key_name}: {value:.6g}")

    failed = (
        maxima["obs"] > float(args.obs_tol)
        or maxima["ball"] > float(args.pos_tol)
        or maxima["racket"] > float(args.pos_tol)
        or maxima["reward"] > float(args.reward_tol)
        or maxima["done_mismatch"] > 0.0
    )
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

