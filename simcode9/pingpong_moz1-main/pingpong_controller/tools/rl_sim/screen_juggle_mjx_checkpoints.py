"""Batch-screen same-shape MJX juggling checkpoints with one JIT per environment condition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

RL_SIM_DIR = Path(__file__).resolve().parent
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from mjx_juggle_env import (
    MjxBallRacketContactPairConfig,
    MjxBaseModeConfig,
    MjxFixedContactPlaneVisionConfig,
    MjxJuggleConfig,
    MjxJuggleEnv,
    MjxPredictiveContactCurriculumConfig,
)
from mjx_smoke import SIMULATION_BASE_MODES
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
    parser.add_argument(
        "--simulation-base-mode",
        choices=SIMULATION_BASE_MODES,
        default=None,
        help=(
            "Optional diagnostic override. Omit it to preserve the environment "
            "checkpoint mode (or legacy_floating for old checkpoints)."
        ),
    )
    parser.add_argument(
        "--ball-mass-range-kg",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional diagnostic ball-mass range override in kg.",
    )
    parser.add_argument(
        "--racket-launch-surface-gap-range-m",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Optional diagnostic racket-launch surface-gap range in metres.",
    )
    parser.add_argument(
        "--env-overrides-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON object of strict MjxJuggleConfig field overrides. "
            "This is intended for frozen-policy causal ablations; unknown "
            "fields and simulation_base_mode fail closed."
        ),
    )
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
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional multi-seed sweep. When supplied, it replaces --seed "
            "and writes one summary row per seed."
        ),
    )
    parser.add_argument(
        "--horizon-seconds",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional validation horizons, for example 0.5 1.5 3 6. "
            "Omit to preserve the checkpoint horizon."
        ),
    )
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
        "--summary-json",
        type=Path,
        default=None,
        help="Optional machine-readable sweep summary and command manifest.",
    )
    parser.add_argument(
        "--ttc-trace-csv",
        type=Path,
        default=None,
        help=(
            "Optional per-tick first-episode predictive-contact trace. "
            "Requires --first-episode-per-env."
        ),
    )
    parser.add_argument(
        "--contact-edge-csv",
        type=Path,
        default=None,
        help=(
            "Optional physical contact-edge/outcome table joined by lane and "
            "edge index. Requires --first-episode-per-env."
        ),
    )
    parser.add_argument(
        "--trace-environment-indices",
        nargs="+",
        type=int,
        default=(0,),
        help="Environment lanes exported by the optional trace tables.",
    )
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
    if args.seeds is not None and not args.seeds:
        parser.error("--seeds must contain at least one integer")
    if args.horizon_seconds is not None and any(
        not np.isfinite(value) or value <= 0.0
        for value in args.horizon_seconds
    ):
        parser.error("every --horizon-seconds value must be positive and finite")
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
    if (args.ttc_trace_csv is not None or args.contact_edge_csv is not None):
        if not args.first_episode_per_env:
            parser.error(
                "trace CSV export requires --first-episode-per-env"
            )
        if any(
            index < 0 or index >= args.n_envs
            for index in args.trace_environment_indices
        ):
            parser.error(
                "--trace-environment-indices must be within [0, n_envs)"
            )
    if args.ball_mass_range_kg is not None:
        low, high = args.ball_mass_range_kg
        if low <= 0.0 or high < low:
            parser.error("--ball-mass-range-kg must be positive and ordered")
    if args.racket_launch_surface_gap_range_m is not None:
        low, high = args.racket_launch_surface_gap_range_m
        if low < 0.0 or high < low:
            parser.error(
                "--racket-launch-surface-gap-range-m must be non-negative "
                "and ordered"
            )
    return args


def checkpoint_cfg(
    payload: dict,
    missing_prob: float,
    coherent_prob: float,
    *,
    dropout_prob: float | None = None,
    dropout_max_steps: int | None = None,
    simulation_base_mode: str | None = None,
    ball_mass_range_kg: tuple[float, float] | None = None,
    racket_launch_surface_gap_range_m: tuple[float, float] | None = None,
    env_overrides: dict[str, object] | None = None,
) -> MjxJuggleConfig:
    raw = payload.get("env_cfg") or {}
    overrides: dict[str, object] = {
        "ball_obs_camera_missing_prob": float(missing_prob),
        "ball_obs_view_bounds_missing_prob": float(missing_prob),
        "ball_obs_missing_episode_coherent_prob": float(coherent_prob),
    }
    if dropout_prob is not None:
        overrides["ball_obs_dropout_prob"] = float(dropout_prob)
    if dropout_max_steps is not None:
        overrides["ball_obs_dropout_max_steps"] = int(dropout_max_steps)
    if ball_mass_range_kg is not None:
        overrides["dr_ball_mass_range"] = tuple(
            float(value) for value in ball_mass_range_kg
        )
    if racket_launch_surface_gap_range_m is not None:
        overrides["racket_launch_surface_gap_range_m"] = tuple(
            float(value) for value in racket_launch_surface_gap_range_m
        )
    resolved_base_mode = (
        str(simulation_base_mode)
        if simulation_base_mode is not None
        else str(raw.get("simulation_base_mode", "legacy_floating"))
    )
    if resolved_base_mode not in SIMULATION_BASE_MODES:
        raise ValueError(
            f"unknown simulation_base_mode={resolved_base_mode!r}; expected "
            f"one of {list(SIMULATION_BASE_MODES)}"
        )
    if (
        "fixed_contact_skill_mode" in raw
        or "scripted_airborne_handoff_enabled" in raw
    ):
        config_type = MjxPredictiveContactCurriculumConfig
    elif bool(raw.get("fixed_contact_plane_training_enabled", False)):
        config_type = MjxFixedContactPlaneVisionConfig
    elif "ball_racket_pair_override_enabled" in raw:
        config_type = MjxBallRacketContactPairConfig
    elif (
        "simulation_base_mode" in raw
        or resolved_base_mode != "legacy_floating"
    ):
        config_type = MjxBaseModeConfig
    else:
        config_type = MjxJuggleConfig
    valid_fields = {field.name for field in fields(config_type)}
    kwargs = {key: value for key, value in raw.items() if key in valid_fields}
    # Checkpoints normally serialize configs through ``asdict``.  Rebuild
    # nested dataclasses before constructing an environment; retaining only
    # the outer predictive subclass would otherwise leave the contact target
    # and high-ball predictor as plain dicts and fail constructor validation.
    default_cfg = config_type()
    for name, value in tuple(kwargs.items()):
        default_value = getattr(default_cfg, name)
        if is_dataclass(default_value) and isinstance(value, dict):
            nested_type = type(default_value)
            nested_fields = {
                field.name for field in fields(nested_type)
            }
            unknown_nested = sorted(set(value) - nested_fields)
            if unknown_nested:
                raise ValueError(
                    f"unknown nested field(s) for {name}: "
                    + ", ".join(unknown_nested)
                )
            kwargs[name] = nested_type(**value)
    if "virtual_camera_pose_mode" not in raw:
        kwargs["virtual_camera_pose_mode"] = "body_mount"
    if issubclass(config_type, MjxBaseModeConfig):
        kwargs["simulation_base_mode"] = resolved_base_mode
    base_cfg = config_type(**kwargs)
    if env_overrides:
        unknown = sorted(set(env_overrides) - valid_fields)
        if unknown:
            raise ValueError(
                "unknown environment override field(s): " + ", ".join(unknown)
            )
        if "simulation_base_mode" in env_overrides:
            raise ValueError(
                "simulation_base_mode must use --simulation-base-mode, not "
                "--env-overrides-json"
            )
        normalized: dict[str, object] = {}
        for name, value in env_overrides.items():
            current = getattr(base_cfg, name)
            if is_dataclass(current) and isinstance(value, dict):
                nested_type = type(current)
                nested_fields = {
                    field.name for field in fields(nested_type)
                }
                unknown_nested = sorted(set(value) - nested_fields)
                if unknown_nested:
                    raise ValueError(
                        f"unknown nested override field(s) for {name}: "
                        + ", ".join(unknown_nested)
                    )
                normalized[name] = nested_type(**value)
            else:
                normalized[name] = (
                    tuple(value) if isinstance(current, tuple) else value
                )
        base_cfg = replace(base_cfg, **normalized)
    return replace(
        base_cfg,
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
    trace_environment_indices: tuple[int, ...] = (),
    ttc_trace_rows: list[dict[str, object]] | None = None,
    contact_edge_rows: list[dict[str, object]] | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, n_envs)
    env_state, obs = jax.jit(env.reset)(reset_keys)
    base_qadr = np.asarray(
        [
            env.base_x_qadr,
            env.base_y_qadr,
            env.base_z_qadr,
            env.base_roll_qadr,
            env.base_pitch_qadr,
            env.base_yaw_qadr,
        ],
        dtype=np.int32,
    )
    base_vadr = np.asarray(
        [
            env.base_x_vadr,
            env.base_y_vadr,
            env.base_z_vadr,
            env.base_roll_vadr,
            env.base_pitch_vadr,
            env.base_yaw_vadr,
        ],
        dtype=np.int32,
    )
    reset_qpos = np.asarray(jax.device_get(env_state.data.qpos))
    reset_qvel = np.asarray(jax.device_get(env_state.data.qvel))
    reset_base_qpos = reset_qpos[:, base_qadr]
    reset_base_qvel = reset_qvel[:, base_vadr]
    reset_arm_qpos = reset_qpos[:, np.asarray(env.arm_qadr_np)]
    reset_arm_qvel = reset_qvel[:, np.asarray(env.arm_vadr_np)]
    reset_racket_pos = np.asarray(
        jax.device_get(env_state.data.site_xpos[:, env.racket_site_id])
    )
    reset_racket_rot = np.asarray(
        jax.device_get(env_state.data.site_xmat[:, env.racket_site_id])
    ).reshape((n_envs, 9))
    reset_racket_cvel = np.asarray(
        jax.device_get(env_state.data.cvel[:, env.racket_body_id])
    )
    reset_ball_pos = np.asarray(
        jax.device_get(env_state.data.xpos[:, env.ball_body_id])
    )
    reset_ball_vel = reset_qvel[:, env.ball_vadr : env.ball_vadr + 3]
    reset_ball_minus_racket = reset_ball_pos - reset_racket_pos
    reset_surface_gap = np.asarray(
        jax.device_get(env_state.reset_ball_surface_gap)
    )
    reset_mass = np.asarray(jax.device_get(env_state.dr_ball_mass))
    reset_normalized_inertia = np.asarray(
        jax.device_get(env_state.dr_ball_normalized_inertia)
    )
    model_inertia = np.asarray(
        jax.device_get(env_state.model.body_inertia[:, env.ball_body_id, :])
    )
    ball_radius = np.asarray(
        jax.device_get(env_state.model.geom_size[:, env.ball_geom_id, 0])
    )
    realized_normalized_inertia = model_inertia / (
        reset_mass[:, None] * ball_radius[:, None] * ball_radius[:, None]
    )
    normalized_inertia_error = np.max(
        np.abs(realized_normalized_inertia - reset_normalized_inertia[:, None])
    )
    max_abs_base_qpos = np.max(np.abs(reset_base_qpos), axis=0)
    max_abs_base_qvel = np.max(np.abs(reset_base_qvel), axis=0)
    max_first_200ms_ball_speed_m_s = float(
        np.linalg.norm(reset_ball_vel, axis=-1).max()
    )
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
    contact_edge_count_by_lane = np.zeros((n_envs,), dtype=np.int32)
    pending_contact_rows: dict[int, dict[str, object]] = {}
    max_steps = max(
        env.max_steps,
        int(np.ceil(episodes / max(1, n_envs)) * env.max_steps * 2),
    )
    for step_index in range(max_steps):
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

        def lane_scalar(key: str, lane: int) -> float:
            value = host.get(key)
            if value is None:
                return float("nan")
            return float(np.asarray(value)[lane])

        def lane_vector(key: str, lane: int, size: int) -> np.ndarray:
            value = host.get(key)
            if value is None:
                return np.full((size,), np.nan, dtype=np.float64)
            return np.asarray(value, dtype=np.float64)[lane].reshape(-1)

        if ttc_trace_rows is not None:
            actor_obs = np.asarray(host["obs"], dtype=np.float64)
            for lane in trace_environment_indices:
                if completed_lanes[lane]:
                    continue
                ball_world = lane_vector(
                    "trace_ball_position_world_m", lane, 3
                )
                ball_velocity_world = lane_vector(
                    "trace_ball_velocity_world_m_s", lane, 3
                )
                racket_world = lane_vector(
                    "trace_racket_position_world_m", lane, 3
                )
                racket_rotation_world = lane_vector(
                    "trace_racket_rotation_world", lane, 9
                )
                racket_cvel = lane_vector(
                    "trace_racket_spatial_velocity_world", lane, 6
                )
                row: dict[str, object] = {
                    "environment_index": int(lane),
                    "global_screen_step": int(step_index),
                    "episode_step": int(lane_scalar("episode_step", lane)),
                    "episode_time_s": float(
                        lane_scalar("episode_step", lane) * env.dt
                    ),
                    "done": lane_scalar("done", lane),
                    "fixed_contact_skill_mode_id": lane_scalar(
                        "fixed_contact_skill_mode_id", lane
                    ),
                    "fixed_contact_phase_id": lane_scalar(
                        "fixed_contact_phase_id", lane
                    ),
                    "fixed_contact_crossing_valid": lane_scalar(
                        "fixed_contact_crossing_valid", lane
                    ),
                    "time_to_contact_s": lane_scalar(
                        "fixed_contact_time_to_contact_s", lane
                    ),
                    "ball_position_world_x_m": ball_world[0],
                    "ball_position_world_y_m": ball_world[1],
                    "ball_position_world_z_m": ball_world[2],
                    "ball_velocity_world_x_m_s": ball_velocity_world[0],
                    "ball_velocity_world_y_m_s": ball_velocity_world[1],
                    "ball_velocity_world_z_m_s": ball_velocity_world[2],
                    "racket_position_world_x_m": racket_world[0],
                    "racket_position_world_y_m": racket_world[1],
                    "racket_position_world_z_m": racket_world[2],
                    "racket_angular_velocity_world_x_rad_s": racket_cvel[0],
                    "racket_angular_velocity_world_y_rad_s": racket_cvel[1],
                    "racket_angular_velocity_world_z_rad_s": racket_cvel[2],
                    "racket_linear_velocity_world_x_m_s": racket_cvel[3],
                    "racket_linear_velocity_world_y_m_s": racket_cvel[4],
                    "racket_linear_velocity_world_z_m_s": racket_cvel[5],
                    "physical_contact_edge": lane_scalar(
                        "physical_contact_edge", lane
                    ),
                    "in_contact": lane_scalar("in_contact", lane),
                    "new_counted_hit": lane_scalar("new_hit", lane),
                }
                for metric_name in (
                    "fixed_contact_tracking_error_m",
                    "fixed_contact_target_position_error_m",
                    "fixed_contact_target_orientation_error_rad",
                    "fixed_contact_s0_xy_error_sum",
                    "fixed_contact_s0_z_error_sum",
                    "fixed_contact_s0_pose_error_sum",
                    "fixed_contact_s0_vxy_sum",
                    "fixed_contact_s0_omega_sum",
                    "fixed_contact_s0_q_xy_sum",
                    "fixed_contact_s0_q_z_sum",
                    "fixed_contact_s0_q_pose_sum",
                    "fixed_contact_s0_q_vxy_sum",
                    "fixed_contact_s0_q_omega_sum",
                    "fixed_contact_s0_ready_sum",
                    "fixed_contact_s0_arrival_window",
                    "fixed_contact_s0_arrival_ready_sum",
                    "fixed_contact_ready_100ms_window",
                    "fixed_contact_ready_100ms_sum",
                    "fixed_contact_ready_80ms_window",
                    "fixed_contact_ready_80ms_sum",
                    "fixed_contact_ready_60ms_window",
                    "fixed_contact_ready_60ms_sum",
                    "fixed_contact_ready_hold_window",
                    "fixed_contact_ready_hold_sum",
                ):
                    row[metric_name] = lane_scalar(metric_name, lane)
                for rotation_index, value in enumerate(
                    racket_rotation_world
                ):
                    row[
                        "racket_rotation_world_r"
                        f"{rotation_index // 3}{rotation_index % 3}"
                    ] = value
                for row_index in range(3):
                    for column_index in range(3):
                        row[
                            "contact_target_rotation_world_r"
                            f"{row_index}{column_index}"
                        ] = lane_scalar(
                            "fixed_contact_target_rotation_world_"
                            f"r{row_index}{column_index}",
                            lane,
                        )
                for axis in ("x", "y", "z"):
                    row[f"contact_target_world_{axis}_m"] = lane_scalar(
                        f"fixed_contact_target_position_world_{axis}_m",
                        lane,
                    )
                    row[
                        f"predicted_ball_contact_world_{axis}_m"
                    ] = lane_scalar(
                        "fixed_contact_predicted_ball_contact_world_"
                        f"{axis}_m",
                        lane,
                    )
                    row[
                        f"incoming_ball_velocity_world_{axis}_m_s"
                    ] = lane_scalar(
                        "fixed_contact_incoming_ball_velocity_world_"
                        f"{axis}_m_s",
                        lane,
                    )
                row["next_crossing_target_world_x_m"] = lane_scalar(
                    "fixed_contact_next_crossing_target_world_x_m", lane
                )
                row["next_crossing_target_world_y_m"] = lane_scalar(
                    "fixed_contact_next_crossing_target_world_y_m", lane
                )
                if actor_obs.shape[-1] >= 43:
                    for offset, axis in enumerate(("x", "y", "z")):
                        row[
                            f"actor_obs_ball_position_base_{axis}_m"
                        ] = float(actor_obs[lane, 28 + offset])
                        row[
                            f"actor_obs_ball_velocity_base_{axis}_m_s"
                        ] = float(actor_obs[lane, 31 + offset])
                        row[
                            f"actor_obs_predicted_contact_base_{axis}_m"
                        ] = float(actor_obs[lane, 34 + offset])
                        row[
                            f"actor_obs_incoming_velocity_base_{axis}_m_s"
                        ] = float(actor_obs[lane, 37 + offset])
                    row["actor_obs_time_to_contact_s"] = float(
                        actor_obs[lane, 40]
                    )
                    row["actor_obs_prediction_valid"] = float(
                        actor_obs[lane, 41]
                    )
                    row["actor_obs_prediction_extrapolated"] = float(
                        actor_obs[lane, 42]
                    )
                    for offset, axis in enumerate(("x", "y", "z")):
                        row[
                            f"actor_obs_racket_position_base_{axis}_m"
                        ] = float(actor_obs[lane, 43 + offset])
                        row[
                            f"actor_obs_contact_target_base_{axis}_m"
                        ] = float(actor_obs[lane, 58 + offset])
                    for offset in range(6):
                        row[
                            f"actor_obs_racket_rotation6d_base_{offset}"
                        ] = float(actor_obs[lane, 46 + offset])
                        row[
                            "actor_obs_contact_target_rotation6d_base_"
                            f"{offset}"
                        ] = float(actor_obs[lane, 61 + offset])
                ttc_trace_rows.append(row)

        if contact_edge_rows is not None:
            for lane in trace_environment_indices:
                if completed_lanes[lane]:
                    continue
                if lane_scalar("physical_contact_edge", lane) > 0.5:
                    previous_pending = pending_contact_rows.pop(lane, None)
                    if previous_pending is not None:
                        previous_pending["resolution"] = (
                            "superseded_by_new_contact_edge"
                        )
                        previous_pending["resolution_episode_step"] = int(
                            lane_scalar("episode_step", lane)
                        )
                    edge_index = int(contact_edge_count_by_lane[lane])
                    contact_edge_count_by_lane[lane] += 1
                    pre_ball = lane_vector(
                        "trace_contact_edge_pre_ball_position_world_m",
                        lane,
                        3,
                    )
                    pre_velocity = lane_vector(
                        "trace_contact_edge_pre_ball_velocity_world_m_s",
                        lane,
                        3,
                    )
                    racket_position = lane_vector(
                        "trace_contact_edge_racket_position_world_m",
                        lane,
                        3,
                    )
                    racket_rotation = lane_vector(
                        "trace_contact_edge_racket_rotation_world",
                        lane,
                        9,
                    )
                    racket_linear_velocity = lane_vector(
                        "trace_contact_edge_racket_linear_velocity_world_m_s",
                        lane,
                        3,
                    )
                    racket_angular_velocity = lane_vector(
                        "trace_contact_edge_racket_angular_velocity_world_rad_s",
                        lane,
                        3,
                    )
                    impact_offset = lane_vector(
                        "trace_contact_edge_impact_offset_racket_xy_m",
                        lane,
                        2,
                    )
                    edge_row: dict[str, object] = {
                        "environment_index": int(lane),
                        "contact_edge_index": edge_index,
                        "contact_edge_episode_step": int(
                            lane_scalar("episode_step", lane)
                        ),
                        "contact_edge_time_s": lane_scalar(
                            "trace_contact_edge_time_s", lane
                        ),
                        "pre_ball_position_world_x_m": pre_ball[0],
                        "pre_ball_position_world_y_m": pre_ball[1],
                        "pre_ball_position_world_z_m": pre_ball[2],
                        "pre_ball_velocity_world_x_m_s": pre_velocity[0],
                        "pre_ball_velocity_world_y_m_s": pre_velocity[1],
                        "pre_ball_velocity_world_z_m_s": pre_velocity[2],
                        "racket_position_world_x_m": racket_position[0],
                        "racket_position_world_y_m": racket_position[1],
                        "racket_position_world_z_m": racket_position[2],
                        "racket_linear_velocity_world_x_m_s": (
                            racket_linear_velocity[0]
                        ),
                        "racket_linear_velocity_world_y_m_s": (
                            racket_linear_velocity[1]
                        ),
                        "racket_linear_velocity_world_z_m_s": (
                            racket_linear_velocity[2]
                        ),
                        "racket_angular_velocity_world_x_rad_s": (
                            racket_angular_velocity[0]
                        ),
                        "racket_angular_velocity_world_y_rad_s": (
                            racket_angular_velocity[1]
                        ),
                        "racket_angular_velocity_world_z_rad_s": (
                            racket_angular_velocity[2]
                        ),
                        "impact_offset_racket_x_m": impact_offset[0],
                        "impact_offset_racket_y_m": impact_offset[1],
                        "resolution": "pending",
                    }
                    for rotation_index, value in enumerate(
                        racket_rotation
                    ):
                        edge_row[
                            "racket_rotation_world_r"
                            f"{rotation_index // 3}{rotation_index % 3}"
                        ] = value
                    contact_edge_rows.append(edge_row)
                    pending_contact_rows[lane] = edge_row
                pending_row = pending_contact_rows.get(lane)
                if pending_row is not None:
                    if lane_scalar("contact_duration_event", lane) > 0.5:
                        pending_row["contact_duration_s"] = lane_scalar(
                            "contact_duration_event_sum_s", lane
                        )
                    counted_hit = lane_scalar("new_hit", lane) > 0.5
                    failed_hit = lane_scalar("failed_hit", lane) > 0.5
                    if counted_hit or failed_hit:
                        post_velocity = lane_vector(
                            "trace_step_end_ball_velocity_world_m_s",
                            lane,
                            3,
                        )
                        pending_row.update(
                            {
                                "resolution": (
                                    "counted_hit"
                                    if counted_hit
                                    else "failed_hit"
                                ),
                                "resolution_episode_step": int(
                                    lane_scalar("episode_step", lane)
                                ),
                                "post_ball_velocity_world_x_m_s": (
                                    post_velocity[0]
                                ),
                                "post_ball_velocity_world_y_m_s": (
                                    post_velocity[1]
                                ),
                                "post_ball_velocity_world_z_m_s": (
                                    post_velocity[2]
                                ),
                                "predicted_apex_abs_z_m": (
                                    lane_scalar("hit_ball_z_sum", lane)
                                    + lane_scalar(
                                        "hit_apex_lift_sum", lane
                                    )
                                    if counted_hit
                                    else float("nan")
                                ),
                                "next_contact_anchor_error_m": (
                                    lane_scalar(
                                        "hit_next_contact_anchor_err_sum",
                                        lane,
                                    )
                                    if counted_hit
                                    else float("nan")
                                ),
                            }
                        )
                        pending_contact_rows.pop(lane, None)
        current_qpos = np.asarray(jax.device_get(env_state.data.qpos))
        current_qvel = np.asarray(jax.device_get(env_state.data.qvel))
        max_abs_base_qpos = np.maximum(
            max_abs_base_qpos,
            np.max(np.abs(current_qpos[:, base_qadr]), axis=0),
        )
        max_abs_base_qvel = np.maximum(
            max_abs_base_qvel,
            np.max(np.abs(current_qvel[:, base_vadr]), axis=0),
        )
        if step_index < int(np.ceil(0.200 / env.dt)):
            max_first_200ms_ball_speed_m_s = max(
                max_first_200ms_ball_speed_m_s,
                float(
                    np.linalg.norm(
                        current_qvel[:, env.ball_vadr : env.ball_vadr + 3],
                        axis=-1,
                    ).max()
                ),
            )
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
            pending_row = pending_contact_rows.pop(env_i, None)
            if pending_row is not None:
                pending_row["resolution"] = (
                    "right_censored_contact"
                    if lane_scalar("in_contact", env_i) > 0.5
                    else "unresolved_terminal"
                )
                pending_row["resolution_episode_step"] = int(
                    lane_scalar("episode_step", env_i)
                )
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
    result = {
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
        "reset_surface_gap_min_m": float(reset_surface_gap.min()),
        "reset_surface_gap_mean_m": float(reset_surface_gap.mean()),
        "reset_surface_gap_max_m": float(reset_surface_gap.max()),
        "reset_ball_mass_min_kg": float(reset_mass.min()),
        "reset_ball_mass_mean_kg": float(reset_mass.mean()),
        "reset_ball_mass_max_kg": float(reset_mass.max()),
        "reset_normalized_inertia_max_abs_error": float(
            normalized_inertia_error
        ),
        "max_first_200ms_ball_speed_m_s": max_first_200ms_ball_speed_m_s,
    }
    for threshold in (1, 2, 3, 5, 6, 12):
        result[f"hit{threshold}_rate"] = float(
            np.mean(hit_array >= float(threshold))
        )
    terminal_reason_keys = sorted(
        {
            key
            for episode_row in episode_rows[:episodes]
            for key in episode_row
            if key.startswith("done/")
        }
    )
    for key in terminal_reason_keys:
        result[f"failure_rate/{key[5:]}"] = float(
            np.mean(
                [
                    float(episode_row.get(key, 0.0))
                    for episode_row in episode_rows[:episodes]
                ]
            )
        )
    vector_metrics = {
        "reset_base_qpos": reset_base_qpos.mean(axis=0),
        "reset_base_qvel": reset_base_qvel.mean(axis=0),
        "max_abs_base_qpos": max_abs_base_qpos,
        "max_abs_base_qvel": max_abs_base_qvel,
        "reset_arm_qpos": reset_arm_qpos.mean(axis=0),
        "reset_arm_qvel": reset_arm_qvel.mean(axis=0),
        "reset_racket_pos": reset_racket_pos.mean(axis=0),
        "reset_racket_rot": reset_racket_rot.mean(axis=0),
        "reset_racket_cvel": reset_racket_cvel.mean(axis=0),
        "reset_ball_pos": reset_ball_pos.mean(axis=0),
        "reset_ball_vel": reset_ball_vel.mean(axis=0),
        "reset_ball_minus_racket": reset_ball_minus_racket.mean(axis=0),
    }
    for metric_name, values in vector_metrics.items():
        for index, value in enumerate(np.asarray(values).reshape(-1)):
            result[f"{metric_name}_{index}"] = float(value)
    return result, episode_rows


def main() -> None:
    args = parse_args()
    env_overrides: dict[str, object] | None = None
    if args.env_overrides_json is not None:
        loaded_overrides = json.loads(
            args.env_overrides_json.read_text(encoding="utf-8")
        )
        if not isinstance(loaded_overrides, dict):
            raise SystemExit("--env-overrides-json must contain one JSON object")
        env_overrides = loaded_overrides
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
    all_ttc_trace_rows: list[dict[str, object]] = []
    all_contact_edge_rows: list[dict[str, object]] = []
    trace_enabled = bool(
        args.ttc_trace_csv is not None or args.contact_edge_csv is not None
    )
    print(f"[screen_mjx] JAX devices: {jax.devices()}")
    sweep_seeds = [int(args.seed)] if args.seeds is None else args.seeds
    sweep_horizons: list[float | None] = (
        [None]
        if args.horizon_seconds is None
        else [float(value) for value in args.horizon_seconds]
    )
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
            simulation_base_mode=args.simulation_base_mode,
            ball_mass_range_kg=(
                None
                if args.ball_mass_range_kg is None
                else tuple(args.ball_mass_range_kg)
            ),
            racket_launch_surface_gap_range_m=(
                None
                if args.racket_launch_surface_gap_range_m is None
                else tuple(args.racket_launch_surface_gap_range_m)
            ),
            env_overrides=env_overrides,
        )
        for horizon_seconds in sweep_horizons:
            horizon_cfg = (
                cfg
                if horizon_seconds is None
                else replace(cfg, horizon_sec=float(horizon_seconds))
            )
            env = MjxJuggleEnv(
                args.xml, n_envs=args.n_envs, cfg=horizon_cfg
            )
            eval_step = make_eval_step(
                env,
                deterministic=True,
                action_gain=1.0,
                include_predictive_contact_trace=trace_enabled,
            )
            print(
                f"[screen_mjx] condition p={missing_prob:.4f} "
                f"q={args.coherent_prob:.4f} "
                f"dropout={horizon_cfg.ball_obs_dropout_prob:.4f}x"
                f"{horizon_cfg.ball_obs_dropout_max_steps} "
                f"base={getattr(horizon_cfg, 'simulation_base_mode', 'legacy_floating')} "
                f"mass={horizon_cfg.dr_ball_mass_range} "
                f"surface_gap={horizon_cfg.racket_launch_surface_gap_range_m} "
                f"horizon={horizon_cfg.horizon_sec:.3f}s "
                f"episodes={args.episodes} n_envs={args.n_envs}"
            )
            for seed in sweep_seeds:
                for checkpoint, payload in payloads:
                    params = jax.tree_util.tree_map(
                        jnp.asarray, policy_parameters(payload["params"])
                    )
                    ttc_trace_rows = (
                        [] if args.ttc_trace_csv is not None else None
                    )
                    contact_edge_rows = (
                        [] if args.contact_edge_csv is not None else None
                    )
                    result, episode_rows = screen_one(
                        eval_step=eval_step,
                        env=env,
                        params=params,
                        episodes=args.episodes,
                        n_envs=args.n_envs,
                        seed=int(seed),
                        first_episode_per_env=bool(
                            args.first_episode_per_env
                        ),
                        trace_environment_indices=tuple(
                            int(value)
                            for value in args.trace_environment_indices
                        ),
                        ttc_trace_rows=ttc_trace_rows,
                        contact_edge_rows=contact_edge_rows,
                    )
                    trace_identity = {
                        "checkpoint": str(checkpoint),
                        "missing_prob": float(missing_prob),
                        "coherent_prob": float(args.coherent_prob),
                        "dropout_prob": float(
                            horizon_cfg.ball_obs_dropout_prob
                        ),
                        "dropout_max_steps": int(
                            horizon_cfg.ball_obs_dropout_max_steps
                        ),
                        "horizon_seconds": float(horizon_cfg.horizon_sec),
                        "seed": int(seed),
                    }
                    if ttc_trace_rows is not None:
                        all_ttc_trace_rows.extend(
                            {**trace_identity, **trace_row}
                            for trace_row in ttc_trace_rows
                        )
                    if contact_edge_rows is not None:
                        all_contact_edge_rows.extend(
                            {**trace_identity, **edge_row}
                            for edge_row in contact_edge_rows
                        )
                    row = {
                        "checkpoint": str(checkpoint),
                        "missing_prob": float(missing_prob),
                        "coherent_prob": float(args.coherent_prob),
                        "dropout_prob": float(
                            horizon_cfg.ball_obs_dropout_prob
                        ),
                        "dropout_max_steps": int(
                            horizon_cfg.ball_obs_dropout_max_steps
                        ),
                        "simulation_base_mode": getattr(
                            horizon_cfg,
                            "simulation_base_mode",
                            "legacy_floating",
                        ),
                        "horizon_seconds": float(horizon_cfg.horizon_sec),
                        "ball_mass_range_low_kg": float(
                            horizon_cfg.dr_ball_mass_range[0]
                        ),
                        "ball_mass_range_high_kg": float(
                            horizon_cfg.dr_ball_mass_range[1]
                        ),
                        "racket_launch_surface_gap_low_m": float(
                            horizon_cfg.racket_launch_surface_gap_range_m[0]
                        ),
                        "racket_launch_surface_gap_high_m": float(
                            horizon_cfg.racket_launch_surface_gap_range_m[1]
                        ),
                        "episodes": int(args.episodes),
                        "n_envs": int(args.n_envs),
                        "seed": int(seed),
                        **result,
                    }
                    rows.append(row)
                    all_episode_rows.extend(
                        {
                            "checkpoint": str(checkpoint),
                            "missing_prob": float(missing_prob),
                            "coherent_prob": float(args.coherent_prob),
                            "dropout_prob": float(
                                horizon_cfg.ball_obs_dropout_prob
                            ),
                            "dropout_max_steps": int(
                                horizon_cfg.ball_obs_dropout_max_steps
                            ),
                            "horizon_seconds": float(
                                horizon_cfg.horizon_sec
                            ),
                            "seed": int(seed),
                            **episode_row,
                        }
                        for episode_row in episode_rows
                    )
                    print(
                        f"[screen_mjx] {checkpoint.name} "
                        f"p={missing_prob:.4f} seed={seed} "
                        f"horizon={horizon_cfg.horizon_sec:.3f}s "
                        f"hits={result['mean_hits']:.3f} "
                        f"P12={result['hit12_rate']:.3f} "
                        f"full={result['full_rate']:.3f} "
                        f"len={result['mean_length']:.1f} "
                        f"view={result['mean_ball_view_in_bounds']:.3f} "
                        f"apex={result['mean_hit_apex_rel_height_m']:.3f} "
                        f"anchor={result['mean_hit_next_contact_anchor_err']:.3f} "
                        f"angular={result['mean_hit_racket_full_angular_speed_rad_s']:.3f} "
                        f"racket_vxy={result['mean_hit_racket_vxy_m_s']:.3f} "
                        f"qacc={result['mean_arm_qacc_limit_exceed_fraction']:.5f} "
                        f"max={result['max_hits']:.0f}"
                    )

    group_fields = (
        "checkpoint",
        "missing_prob",
        "coherent_prob",
        "dropout_prob",
        "dropout_max_steps",
        "simulation_base_mode",
        "horizon_seconds",
        "ball_mass_range_low_kg",
        "ball_mass_range_high_kg",
        "racket_launch_surface_gap_low_m",
        "racket_launch_surface_gap_high_m",
    )
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(
            tuple(row[field] for field in group_fields), []
        ).append(row)
    lower_is_worse = (
        "hit1_rate",
        "hit2_rate",
        "hit3_rate",
        "hit5_rate",
        "hit6_rate",
        "hit12_rate",
        "full_rate",
        "mean_ball_view_in_bounds",
    )
    upper_is_worse = (
        "mean_hit_next_contact_anchor_err",
        "mean_hit_racket_full_angular_speed_rad_s",
        "mean_hit_racket_vxy_m_s",
        "rms_hit_racket_vxy_m_s",
        "mean_arm_qvel_limit_exceed_fraction",
        "mean_arm_qacc_limit_exceed_fraction",
    )

    def finite_min_mean(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan"), float("nan")
        return float(finite.min()), float(finite.mean())

    def finite_max_mean(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan"), float("nan")
        return float(finite.max()), float(finite.mean())

    for group_rows in groups.values():
        aggregate: dict[str, float] = {
            "seed_count": float(len(group_rows))
        }
        for metric in lower_is_worse:
            values = np.asarray(
                [float(row[metric]) for row in group_rows], dtype=np.float64
            )
            worst, pooled = finite_min_mean(values)
            aggregate[f"seed_worst_min/{metric}"] = worst
            aggregate[f"seed_pooled_mean/{metric}"] = pooled
        for metric in upper_is_worse:
            values = np.asarray(
                [float(row[metric]) for row in group_rows], dtype=np.float64
            )
            worst, pooled = finite_max_mean(values)
            aggregate[f"seed_worst_max/{metric}"] = worst
            aggregate[f"seed_pooled_mean/{metric}"] = pooled
        failure_metrics = sorted(
            key
            for key in group_rows[0]
            if key.startswith("failure_rate/")
        )
        for metric in failure_metrics:
            aggregate[f"seed_worst_max/{metric}"] = max(
                float(row[metric]) for row in group_rows
            )
        for row in group_rows:
            row.update(aggregate)

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[screen_mjx] wrote: {args.results_csv}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        json_rows = [
            {
                key: (
                    None
                    if isinstance(value, (float, np.floating))
                    and not np.isfinite(value)
                    else value
                )
                for key, value in row.items()
            }
            for row in rows
        ]
        args.summary_json.write_text(
            json.dumps(
                {
                    "command": [str(value) for value in sys.argv],
                    "rows": json_rows,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(f"[screen_mjx] wrote: {args.summary_json}")
    if args.episode_results_csv is not None:
        args.episode_results_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_episode_rows[0])
        with args.episode_results_csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_episode_rows)
        print(f"[screen_mjx] wrote: {args.episode_results_csv}")

    def write_trace_csv(
        path: Path | None,
        trace_rows: list[dict[str, object]],
    ) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        identity_fields = [
            "checkpoint",
            "missing_prob",
            "coherent_prob",
            "dropout_prob",
            "dropout_max_steps",
            "horizon_seconds",
            "seed",
            "environment_index",
        ]
        remaining = sorted(
            {
                key
                for trace_row in trace_rows
                for key in trace_row
                if key not in identity_fields
            }
        )
        fieldnames = identity_fields + remaining
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(trace_rows)
        print(f"[screen_mjx] wrote: {path}")

    write_trace_csv(args.ttc_trace_csv, all_ttc_trace_rows)
    write_trace_csv(args.contact_edge_csv, all_contact_edge_rows)


if __name__ == "__main__":
    main()
