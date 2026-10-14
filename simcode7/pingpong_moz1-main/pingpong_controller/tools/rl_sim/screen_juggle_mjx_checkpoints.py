"""Batch-screen same-shape MJX juggling checkpoints with one JIT per environment condition."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

RL_SIM_DIR = Path(__file__).resolve().parent
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv
from validate_juggle_mjx_ppo import load_checkpoint, make_eval_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick deterministic Pareto screening of archived MJX juggling checkpoints."
    )
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--env-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint supplying shared XML/environment config for every policy.",
    )
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("--missing-probs", nargs="+", required=True, type=float)
    parser.add_argument(
        "--dropout-probs",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional per-condition refresh-dropout probabilities. The list "
            "must match --missing-probs. Omit it to preserve the environment "
            "checkpoint value."
        ),
    )
    parser.add_argument(
        "--dropout-max-steps",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional per-condition dropout lengths. The list must match "
            "--missing-probs. Omit it to preserve the environment checkpoint "
            "value."
        ),
    )
    parser.add_argument("--coherent-prob", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--first-episode-per-env",
        action="store_true",
        help=(
            "Collect exactly the first episode from each environment lane. "
            "Requires --episodes == --n-envs and enables paired lane-wise "
            "checkpoint comparisons without fast-failure resampling bias."
        ),
    )
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument(
        "--episode-results-csv",
        type=Path,
        default=None,
        help=(
            "Optional per-episode output with every reward-component sum and "
            "terminal reason for frozen-policy reward attribution."
        ),
    )
    args = parser.parse_args()
    if args.episodes <= 0 or args.n_envs <= 0:
        parser.error("--episodes and --n-envs must be positive")
    if not 0.0 <= args.coherent_prob <= 1.0:
        parser.error("--coherent-prob must be within [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in args.missing_probs):
        parser.error("every --missing-probs value must be within [0, 1]")
    if args.dropout_probs is not None:
        if len(args.dropout_probs) != len(args.missing_probs):
            parser.error("--dropout-probs must match --missing-probs length")
        if any(not 0.0 <= value <= 1.0 for value in args.dropout_probs):
            parser.error("every --dropout-probs value must be within [0, 1]")
    if args.dropout_max_steps is not None:
        if len(args.dropout_max_steps) != len(args.missing_probs):
            parser.error("--dropout-max-steps must match --missing-probs length")
        if any(value <= 0 for value in args.dropout_max_steps):
            parser.error("every --dropout-max-steps value must be positive")
    if args.first_episode_per_env and args.episodes != args.n_envs:
        parser.error(
            "--first-episode-per-env requires --episodes == --n-envs"
        )
    return args


def checkpoint_cfg(
    payload: dict,
    missing_prob: float,
    coherent_prob: float,
    *,
    dropout_prob: float | None = None,
    dropout_max_steps: int | None = None,
) -> MjxJuggleConfig:
    raw = payload.get("env_cfg") or {}
    valid_fields = {field.name for field in fields(MjxJuggleConfig)}
    kwargs = {key: value for key, value in raw.items() if key in valid_fields}
    if "virtual_camera_pose_mode" not in raw:
        kwargs["virtual_camera_pose_mode"] = "body_mount"
    overrides: dict[str, object] = {
        "ball_obs_camera_missing_prob": float(missing_prob),
        "ball_obs_view_bounds_missing_prob": float(missing_prob),
        "ball_obs_missing_episode_coherent_prob": float(coherent_prob),
    }
    if dropout_prob is not None:
        overrides["ball_obs_dropout_prob"] = float(dropout_prob)
    if dropout_max_steps is not None:
        overrides["ball_obs_dropout_max_steps"] = int(dropout_max_steps)
    return replace(
        MjxJuggleConfig(**kwargs),
        **overrides,
    )


def policy_parameters(params: dict[str, object]) -> dict[str, object]:
    """Return only parameter leaves consumed by ``policy_mean``.

    Frozen policy comparison must accept checkpoints that append critic-only
    state, such as V65's CMDP cost heads and dual variables.  Actor structure
    and shape remain strict, including residual-teacher actor leaves when
    present; unrelated value state cannot change the deterministic action.
    """

    policy_keys = ("pi", "log_std", "teacher_pi", "residual_action_scale")
    policy_params = {key: params[key] for key in policy_keys if key in params}
    if "pi" not in policy_params or "log_std" not in policy_params:
        raise ValueError("checkpoint is missing policy parameters")
    return policy_params


def policy_parameter_shapes(params: dict[str, object]) -> object:
    return jax.tree_util.tree_map(np.shape, policy_parameters(params))


def screen_one(
    *,
    eval_step,
    env: MjxJuggleEnv,
    params,
    episodes: int,
    n_envs: int,
    seed: int,
    first_episode_per_env: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, n_envs)
    env_state, obs = jax.jit(env.reset)(reset_keys)
    running_return = jnp.zeros((n_envs,), dtype=jnp.float32)
    running_length = jnp.zeros((n_envs,), dtype=jnp.int32)
    dq_adapter_ema = obs[:, 7:14]
    dq_adapter_history = jnp.broadcast_to(
        dq_adapter_ema[:, None, :],
        (n_envs, 4, dq_adapter_ema.shape[-1]),
    )
    hits: list[float] = []
    lengths: list[int] = []
    full: list[float] = []
    view_rates: list[float] = []
    hit_apexes: list[float] = []
    next_anchor_errors: list[float] = []
    running_view_sum = np.zeros((n_envs,), dtype=np.float64)
    running_hit_events = np.zeros((n_envs,), dtype=np.float64)
    running_hit_apex_sum = np.zeros((n_envs,), dtype=np.float64)
    running_next_anchor_sum = np.zeros((n_envs,), dtype=np.float64)
    running_hit_ball_z_sum = np.zeros((n_envs,), dtype=np.float64)
    running_hit_racket_z_sum = np.zeros((n_envs,), dtype=np.float64)
    running_hit_apex_lift_sum = np.zeros((n_envs,), dtype=np.float64)
    running_hit_racket_vxy_sum = np.zeros((n_envs,), dtype=np.float64)
    running_hit_racket_vxy_sq_sum = np.zeros((n_envs,), dtype=np.float64)
    running_hit_racket_full_angular_speed_sum = np.zeros(
        (n_envs,), dtype=np.float64
    )
    running_qvel_exceed_sum = np.zeros((n_envs,), dtype=np.float64)
    running_qacc_exceed_sum = np.zeros((n_envs,), dtype=np.float64)
    running_counted_interval_events = np.zeros((n_envs,), dtype=np.float64)
    running_counted_interval_sum_s = np.zeros((n_envs,), dtype=np.float64)
    running_reward_sums: dict[str, np.ndarray] = {}
    episode_rows: list[dict[str, float]] = []
    completed_lanes = np.zeros((n_envs,), dtype=bool)
    max_steps = max(
        env.max_steps,
        int(np.ceil(episodes / max(1, n_envs)) * env.max_steps * 2),
    )
    for _ in range(max_steps):
        (
            env_state,
            obs,
            rng,
            running_return,
            running_length,
            dq_adapter_ema,
            dq_adapter_history,
            metrics,
        ) = eval_step(
            params,
            env_state,
            obs,
            rng,
            running_return,
            running_length,
            dq_adapter_ema,
            dq_adapter_history,
        )
        host = jax.device_get(metrics)
        running_view_sum += np.asarray(host["ball_view_in_bounds"], dtype=np.float64)
        running_hit_events += np.asarray(host["hit_event_count"], dtype=np.float64)
        running_hit_apex_sum += np.asarray(
            host["hit_apex_rel_height_sum"], dtype=np.float64
        )
        running_next_anchor_sum += np.asarray(
            host["hit_next_contact_anchor_err_sum"], dtype=np.float64
        )
        running_hit_ball_z_sum += np.asarray(
            host["hit_ball_z_sum"], dtype=np.float64
        )
        running_hit_racket_z_sum += np.asarray(
            host["hit_racket_z_sum"], dtype=np.float64
        )
        running_hit_apex_lift_sum += np.asarray(
            host["hit_apex_lift_sum"], dtype=np.float64
        )
        running_hit_racket_vxy_sum += np.asarray(
            host["hit_racket_vxy_sum"], dtype=np.float64
        )
        running_hit_racket_vxy_sq_sum += np.asarray(
            host["hit_racket_vxy_sq_sum"], dtype=np.float64
        )
        running_hit_racket_full_angular_speed_sum += np.asarray(
            host["hit_racket_full_angular_speed_rad_s"],
            dtype=np.float64,
        )
        # ``make_eval_step`` exposes the measured arm velocity and finite-
        # difference acceleration arrays, but intentionally does not forward
        # the environment's scalar ``metric/arm_*_exceed_fraction`` aliases.
        # Recompute the same pointwise fractions here instead of extending the
        # shared validator metric contract solely for this screening tool.
        arm_qvel = np.asarray(host["arm_qvel"], dtype=np.float64)
        arm_qacc = np.asarray(host["arm_qacc"], dtype=np.float64)
        qvel_limit = np.asarray(env.arm_vel_limit_rad_s, dtype=np.float64)
        qacc_limit = np.asarray(env.arm_acc_limit_rad_s2, dtype=np.float64)
        running_qvel_exceed_sum += np.mean(
            np.abs(arm_qvel) > qvel_limit[None, :], axis=-1
        )
        running_qacc_exceed_sum += np.mean(
            np.abs(arm_qacc) > qacc_limit[None, :], axis=-1
        )
        running_counted_interval_events += np.asarray(
            host["counted_hit_interval_event"], dtype=np.float64
        )
        running_counted_interval_sum_s += np.asarray(
            host["counted_hit_interval_sum_s"], dtype=np.float64
        )
        reward_keys = sorted(key for key in host if key.startswith("reward/"))
        for key in reward_keys:
            if key not in running_reward_sums:
                running_reward_sums[key] = np.zeros(
                    (n_envs,), dtype=np.float64
                )
            running_reward_sums[key] += np.asarray(
                host[key], dtype=np.float64
            )
        done_indices = np.flatnonzero(np.asarray(host["done"], dtype=bool))
        for env_i in done_indices:
            if first_episode_per_env and completed_lanes[env_i]:
                continue
            episode_length = max(1, int(host["episode_length"][env_i]))
            event_count = float(running_hit_events[env_i])
            hits.append(float(host["hit_count"][env_i]))
            lengths.append(episode_length)
            full.append(float(host["truncated"][env_i]) > 0.5)
            view_rates.append(float(running_view_sum[env_i] / episode_length))
            hit_apexes.append(
                float(running_hit_apex_sum[env_i] / event_count)
                if event_count > 0.0
                else float("nan")
            )
            next_anchor_errors.append(
                float(running_next_anchor_sum[env_i] / event_count)
                if event_count > 0.0
                else float("nan")
            )
            interval_events = float(running_counted_interval_events[env_i])
            episode_row = {
                "episode_index": float(len(hits) - 1),
                "environment_index": float(env_i),
                "hits": float(host["hit_count"][env_i]),
                "length": float(episode_length),
                "length_fraction": float(episode_length / env.max_steps),
                "full": float(host["truncated"][env_i]) > 0.5,
                "return": float(host["episode_return"][env_i]),
                "ball_view_in_bounds": float(
                    running_view_sum[env_i] / episode_length
                ),
                "mean_hit_ball_z_m": (
                    float(running_hit_ball_z_sum[env_i] / event_count)
                    if event_count > 0.0
                    else float("nan")
                ),
                "mean_hit_racket_z_m": (
                    float(running_hit_racket_z_sum[env_i] / event_count)
                    if event_count > 0.0
                    else float("nan")
                ),
                "mean_hit_apex_lift_m": (
                    float(running_hit_apex_lift_sum[env_i] / event_count)
                    if event_count > 0.0
                    else float("nan")
                ),
                "mean_hit_racket_vxy_m_s": (
                    float(running_hit_racket_vxy_sum[env_i] / event_count)
                    if event_count > 0.0
                    else float("nan")
                ),
                "rms_hit_racket_vxy_m_s": (
                    float(
                        np.sqrt(
                            max(
                                0.0,
                                running_hit_racket_vxy_sq_sum[env_i]
                                / event_count,
                            )
                        )
                    )
                    if event_count > 0.0
                    else float("nan")
                ),
                "mean_hit_racket_full_angular_speed_rad_s": (
                    float(
                        running_hit_racket_full_angular_speed_sum[env_i]
                        / event_count
                    )
                    if event_count > 0.0
                    else float("nan")
                ),
                "mean_arm_qvel_limit_exceed_fraction": float(
                    running_qvel_exceed_sum[env_i] / episode_length
                ),
                "mean_arm_qacc_limit_exceed_fraction": float(
                    running_qacc_exceed_sum[env_i] / episode_length
                ),
                "mean_counted_hit_interval_s": (
                    float(
                        running_counted_interval_sum_s[env_i]
                        / interval_events
                    )
                    if interval_events > 0.0
                    else float("nan")
                ),
            }
            episode_row["mean_hit_apex_abs_z_m"] = (
                episode_row["mean_hit_ball_z_m"]
                + episode_row["mean_hit_apex_lift_m"]
            )
            for key in sorted(key for key in host if key.startswith("done/")):
                episode_row[key] = float(host[key][env_i])
            for key, values in running_reward_sums.items():
                episode_row[f"{key}/episode_sum"] = float(values[env_i])
                episode_row[f"{key}/step_mean"] = float(
                    values[env_i] / episode_length
                )
            episode_rows.append(episode_row)
            completed_lanes[env_i] = True
            if len(hits) >= episodes:
                break
        running_view_sum[done_indices] = 0.0
        running_hit_events[done_indices] = 0.0
        running_hit_apex_sum[done_indices] = 0.0
        running_next_anchor_sum[done_indices] = 0.0
        running_hit_ball_z_sum[done_indices] = 0.0
        running_hit_racket_z_sum[done_indices] = 0.0
        running_hit_apex_lift_sum[done_indices] = 0.0
        running_hit_racket_vxy_sum[done_indices] = 0.0
        running_hit_racket_vxy_sq_sum[done_indices] = 0.0
        running_hit_racket_full_angular_speed_sum[done_indices] = 0.0
        running_qvel_exceed_sum[done_indices] = 0.0
        running_qacc_exceed_sum[done_indices] = 0.0
        running_counted_interval_events[done_indices] = 0.0
        running_counted_interval_sum_s[done_indices] = 0.0
        for values in running_reward_sums.values():
            values[done_indices] = 0.0
        if len(hits) >= episodes:
            break
    if len(hits) < episodes:
        raise RuntimeError(
            f"completed only {len(hits)}/{episodes} episodes within {max_steps} environment steps"
        )
    hit_array = np.asarray(hits[:episodes], dtype=np.float64)
    length_array = np.asarray(lengths[:episodes], dtype=np.float64)
    full_array = np.asarray(full[:episodes], dtype=np.float64)
    view_array = np.asarray(view_rates[:episodes], dtype=np.float64)
    apex_array = np.asarray(hit_apexes[:episodes], dtype=np.float64)
    anchor_array = np.asarray(next_anchor_errors[:episodes], dtype=np.float64)
    angular_array = np.asarray(
        [
            row["mean_hit_racket_full_angular_speed_rad_s"]
            for row in episode_rows[:episodes]
        ],
        dtype=np.float64,
    )
    racket_vxy_array = np.asarray(
        [row["mean_hit_racket_vxy_m_s"] for row in episode_rows[:episodes]],
        dtype=np.float64,
    )
    racket_vxy_rms_array = np.asarray(
        [row["rms_hit_racket_vxy_m_s"] for row in episode_rows[:episodes]],
        dtype=np.float64,
    )
    qvel_array = np.asarray(
        [
            row["mean_arm_qvel_limit_exceed_fraction"]
            for row in episode_rows[:episodes]
        ],
        dtype=np.float64,
    )
    qacc_array = np.asarray(
        [
            row["mean_arm_qacc_limit_exceed_fraction"]
            for row in episode_rows[:episodes]
        ],
        dtype=np.float64,
    )
    return {
        "mean_hits": float(hit_array.mean()),
        "max_hits": float(hit_array.max()),
        "mean_length": float(length_array.mean()),
        "full_rate": float(full_array.mean()),
        "mean_ball_view_in_bounds": float(view_array.mean()),
        "mean_hit_apex_rel_height_m": float(np.nanmean(apex_array)),
        "mean_hit_next_contact_anchor_err": float(np.nanmean(anchor_array)),
        "mean_hit_racket_full_angular_speed_rad_s": float(
            np.nanmean(angular_array)
        ),
        "mean_hit_racket_vxy_m_s": float(np.nanmean(racket_vxy_array)),
        "rms_hit_racket_vxy_m_s": float(np.nanmean(racket_vxy_rms_array)),
        "mean_arm_qvel_limit_exceed_fraction": float(np.nanmean(qvel_array)),
        "mean_arm_qacc_limit_exceed_fraction": float(np.nanmean(qacc_array)),
    }, episode_rows


def main() -> None:
    args = parse_args()
    payloads = [(path, load_checkpoint(path)) for path in args.checkpoints]
    env_payload = (
        load_checkpoint(args.env_checkpoint)
        if args.env_checkpoint is not None
        else payloads[0][1]
    )
    reference_shapes = policy_parameter_shapes(payloads[0][1]["params"])
    for path, payload in payloads[1:]:
        shapes = policy_parameter_shapes(payload["params"])
        if jax.tree_util.tree_structure(shapes) != jax.tree_util.tree_structure(reference_shapes):
            raise SystemExit(f"parameter tree mismatch: {path}")
        shape_equal = jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda left, right: left == right, shapes, reference_shapes)
        )
        if not shape_equal:
            raise SystemExit(f"parameter shape mismatch: {path}")

    rows: list[dict[str, object]] = []
    all_episode_rows: list[dict[str, object]] = []
    print(f"[screen_mjx] JAX devices: {jax.devices()}")
    for condition_index, missing_prob in enumerate(args.missing_probs):
        dropout_prob = (
            None
            if args.dropout_probs is None
            else float(args.dropout_probs[condition_index])
        )
        dropout_max_steps = (
            None
            if args.dropout_max_steps is None
            else int(args.dropout_max_steps[condition_index])
        )
        cfg = checkpoint_cfg(
            env_payload,
            missing_prob,
            args.coherent_prob,
            dropout_prob=dropout_prob,
            dropout_max_steps=dropout_max_steps,
        )
        env = MjxJuggleEnv(args.xml, n_envs=args.n_envs, cfg=cfg)
        eval_step = make_eval_step(env, deterministic=True, action_gain=1.0)
        print(
            f"[screen_mjx] condition p={missing_prob:.4f} q={args.coherent_prob:.4f} "
            f"dropout={cfg.ball_obs_dropout_prob:.4f}x{cfg.ball_obs_dropout_max_steps} "
            f"episodes={args.episodes} n_envs={args.n_envs}"
        )
        for checkpoint, payload in payloads:
            params = jax.tree_util.tree_map(
                jnp.asarray, policy_parameters(payload["params"])
            )
            result, episode_rows = screen_one(
                eval_step=eval_step,
                env=env,
                params=params,
                episodes=args.episodes,
                n_envs=args.n_envs,
                seed=args.seed,
                first_episode_per_env=bool(args.first_episode_per_env),
            )
            row = {
                "checkpoint": str(checkpoint),
                "missing_prob": float(missing_prob),
                "coherent_prob": float(args.coherent_prob),
                "dropout_prob": float(cfg.ball_obs_dropout_prob),
                "dropout_max_steps": int(cfg.ball_obs_dropout_max_steps),
                "episodes": int(args.episodes),
                "n_envs": int(args.n_envs),
                "seed": int(args.seed),
                **result,
            }
            rows.append(row)
            all_episode_rows.extend(
                {
                    "checkpoint": str(checkpoint),
                    "missing_prob": float(missing_prob),
                    "coherent_prob": float(args.coherent_prob),
                    "dropout_prob": float(cfg.ball_obs_dropout_prob),
                    "dropout_max_steps": int(cfg.ball_obs_dropout_max_steps),
                    "seed": int(args.seed),
                    **episode_row,
                }
                for episode_row in episode_rows
            )
            print(
                f"[screen_mjx] {checkpoint.name} p={missing_prob:.4f} "
                f"dropout={cfg.ball_obs_dropout_prob:.4f}x"
                f"{cfg.ball_obs_dropout_max_steps} "
                f"hits={result['mean_hits']:.3f} full={result['full_rate']:.3f} "
                f"len={result['mean_length']:.1f} view={result['mean_ball_view_in_bounds']:.3f} "
                f"apex={result['mean_hit_apex_rel_height_m']:.3f} "
                f"anchor={result['mean_hit_next_contact_anchor_err']:.3f} "
                f"angular={result['mean_hit_racket_full_angular_speed_rad_s']:.3f} "
                f"racket_vxy={result['mean_hit_racket_vxy_m_s']:.3f} "
                f"qacc={result['mean_arm_qacc_limit_exceed_fraction']:.5f} "
                f"max={result['max_hits']:.0f}"
            )

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[screen_mjx] wrote: {args.results_csv}")
    if args.episode_results_csv is not None:
        args.episode_results_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_episode_rows[0])
        with args.episode_results_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episode_rows)
        print(f"[screen_mjx] wrote: {args.episode_results_csv}")


if __name__ == "__main__":
    main()
