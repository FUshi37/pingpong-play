from __future__ import annotations

import argparse
from collections import deque
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from .hil_env import HilSafetyConfig, build_simcode4_hil_config, load_simcode4_env, make_hil_env_class
from .ros_plant import PlantConfig, PlantSafetyError, RosPlantRuntime


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "hil_flashsac.yaml"
DEFAULT_SIMCODE4_RL = (
    WORKSPACE / "simcode4/pingpong_moz1-main/pingpong_controller/tools/rl_sim"
)
DEFAULT_XML = DEFAULT_SIMCODE4_RL / "moz1_pd.xml"
FLASH_ROOT = WORKSPACE / "third_party/FlashSAC"
FLASH_UPSTREAM = "https://github.com/Holiday-Robot/FlashSAC"


def _flashsac_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(FLASH_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _episode_meets_stage_goal(stage: dict[str, Any], metrics: dict[str, Any]) -> bool:
    """Competence gate used for curriculum promotion and regression."""
    mode = str(stage["success_mode"])
    hits = int(metrics.get("max_hits", 0))
    if mode == "launch_progress":
        return bool(
            hits >= 1
            or (
                bool(metrics.get("contact_detected", False))
                and float(metrics.get("max_ball_vz", 0.0)) >= 0.20
                and float(metrics.get("max_predicted_apex_rel_z", 0.0)) >= 0.05
            )
        )
    required_hits = {"first_hit": 1, "two_hits": 2, "three_hits": 3}.get(mode)
    if required_hits is None:
        raise ValueError(f"unknown curriculum success_mode={mode!r}")
    return hits >= required_hits


class CompetenceCurriculum:
    """Episode-window curriculum that cannot advance on elapsed steps alone."""

    def __init__(self, stages: list[dict[str, Any]]) -> None:
        if not stages:
            raise ValueError("curriculum must contain at least one stage")
        self.stages = stages
        self.index = 0
        self.episodes_in_stage = 0
        self.eligible_episodes_in_stage = 0
        self._outcomes: deque[bool] = deque(maxlen=int(stages[0]["window_episodes"]))

    @property
    def stage(self) -> dict[str, Any]:
        return self.stages[self.index]

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._outcomes)) if self._outcomes else 0.0

    def _switch(self, new_index: int, reason: str, rate: float) -> dict[str, Any]:
        previous = self.stage["name"]
        self.index = int(new_index)
        self.episodes_in_stage = 0
        self.eligible_episodes_in_stage = 0
        self._outcomes = deque(maxlen=int(self.stage["window_episodes"]))
        return {
            "type": reason,
            "from": previous,
            "to": self.stage["name"],
            "trigger_success_rate": float(rate),
        }

    def observe_episode(self, metrics: dict[str, Any]) -> tuple[bool, dict[str, Any] | None]:
        success = _episode_meets_stage_goal(self.stage, metrics)
        self.episodes_in_stage += 1
        eligible = bool(metrics.get("evaluation_episode", False)) or not bool(
            self.stage.get("promotion_requires_evaluation", False)
        )
        if not eligible:
            return success, None
        self._outcomes.append(success)
        self.eligible_episodes_in_stage += 1
        rate = self.success_rate
        minimum = int(self.stage["min_episodes"])
        if (
            self.index + 1 < len(self.stages)
            and self.eligible_episodes_in_stage >= minimum
            and len(self._outcomes) >= minimum
            and rate >= float(self.stage["promote_success_rate"])
        ):
            return success, self._switch(self.index + 1, "promotion", rate)
        demote_rate = float(self.stage.get("demote_success_rate", 0.0))
        if (
            self.index > 0
            and demote_rate > 0.0
            and self.eligible_episodes_in_stage >= int(self.stage["window_episodes"])
            and rate < demote_rate
        ):
            return success, self._switch(self.index - 1, "regression", rate)
        return success, None

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_name": str(self.stage["name"]),
            "index": int(self.index),
            "episodes_in_stage": int(self.episodes_in_stage),
            "eligible_episodes_in_stage": int(self.eligible_episodes_in_stage),
            "outcomes": [bool(value) for value in self._outcomes],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stage_name = str(state["stage_name"])
        names = [str(stage["name"]) for stage in self.stages]
        if stage_name not in names:
            raise ValueError(
                f"saved curriculum stage {stage_name!r} is absent from current config"
            )
        self.index = names.index(stage_name)
        self.episodes_in_stage = int(state.get("episodes_in_stage", 0))
        self.eligible_episodes_in_stage = int(
            state.get("eligible_episodes_in_stage", 0)
        )
        self._outcomes = deque(
            (bool(value) for value in state.get("outcomes", ())),
            maxlen=int(self.stage["window_episodes"]),
        )

    def reset_to_stage(self, stage_name: str) -> None:
        """Select a revised curriculum stage without reusing stale outcomes."""
        names = [str(stage["name"]) for stage in self.stages]
        if stage_name not in names:
            raise ValueError(
                f"resume curriculum stage {stage_name!r} is absent from current config"
            )
        self.index = names.index(stage_name)
        self.episodes_in_stage = 0
        self.eligible_episodes_in_stage = 0
        self._outcomes = deque(maxlen=int(self.stage["window_episodes"]))


def _is_known_simulator_idle_pose(q: np.ndarray, plant_cfg: Any) -> bool:
    if not bool(plant_cfg.allow_simulator_idle_recovery):
        return False
    idle = np.deg2rad(np.asarray(plant_cfg.simulator_idle_degrees, dtype=np.float64))
    tolerance = np.deg2rad(float(plant_cfg.simulator_idle_tolerance_deg))
    return bool(np.max(np.abs(np.asarray(q, dtype=np.float64) - idle)) <= tolerance)


def _resolve_run_dir(output_dir: str | Path, resume: str | Path | None) -> Path:
    """Append resumed metrics/TensorBoard events to the original run."""
    if resume:
        checkpoint = Path(resume).expanduser().resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint directory not found: {checkpoint}")
        return checkpoint.parent
    return (WORKSPACE / Path(output_dir) / time.strftime("%Y%m%d-%H%M%S")).resolve()


def _truncated_zeta_cdf(mu: float, max_n: int) -> np.ndarray:
    """NumPy equivalent of official FlashSAC's truncated-zeta CDF."""
    ns = np.arange(1, int(max_n) + 1, dtype=np.float64)
    pmf = np.power(ns, -float(mu))
    return np.cumsum(pmf / np.sum(pmf))


def _sample_truncated_zeta(cdf: np.ndarray, uniform_sample: float | None = None) -> int:
    """Sample 1..N exactly as FlashSAC, without clipping a Zipf sample."""
    u = float(np.random.random() if uniform_sample is None else uniform_sample)
    return int(np.searchsorted(np.asarray(cdf), u, side="right") + 1)


def _agent_config(
    raw: Any, total_steps: int, seed: int, updates_per_step: float = 1.0
) -> Any:
    # FlashSAC advances its LR scheduler inside agent.update(), so scheduler
    # lengths are update counts, not environment interaction counts.
    total_updates = max(1, int(float(total_steps) * float(updates_per_step)))
    return OmegaConf.create(
        {
            "agent_type": "flashSAC",
            "seed": seed,
            "normalize_reward": bool(raw.normalize_reward),
            "normalized_G_max": float(raw.normalized_G_max),
            "asymmetric_observation": False,
            "device_type": str(raw.device_type),
            "buffer_max_length": int(raw.buffer_max_length),
            "buffer_min_length": int(raw.buffer_min_length),
            "buffer_device_type": str(raw.buffer_device_type),
            "sample_batch_size": int(raw.sample_batch_size),
            "learning_rate_init": float(raw.learning_rate_init),
            "learning_rate_peak": float(raw.learning_rate_peak),
            "learning_rate_end": float(raw.learning_rate_end),
            "learning_rate_warmup_rate": 0.01,
            "learning_rate_warmup_step": max(1, int(total_updates * 0.01)),
            "learning_rate_decay_rate": 1.0,
            "learning_rate_decay_step": total_updates,
            "actor_num_blocks": int(raw.actor_num_blocks),
            "actor_hidden_dim": int(raw.actor_hidden_dim),
            "actor_bc_alpha": 0.0,
            "actor_noise_zeta_mu": float(raw.actor_noise_zeta_mu),
            "actor_noise_zeta_max": int(raw.actor_noise_zeta_max),
            "actor_update_period": int(raw.actor_update_period),
            "critic_num_blocks": int(raw.critic_num_blocks),
            "critic_hidden_dim": int(raw.critic_hidden_dim),
            "critic_num_bins": int(raw.critic_num_bins),
            "critic_min_v": -float(raw.normalized_G_max),
            "critic_max_v": float(raw.normalized_G_max),
            "critic_target_update_tau": float(raw.critic_target_update_tau),
            "temp_initial_value": float(raw.temp_initial_value),
            "temp_target_sigma": float(raw.temp_target_sigma),
            "temp_target_entropy": None,
            "gamma": float(raw.gamma),
            "n_step": int(raw.n_step),
            "use_compile": bool(raw.use_compile),
            "compile_mode": str(raw.compile_mode),
            "use_amp": bool(raw.use_amp),
            "load_optimizer": True,
            "load_reward_normalizer": True,
        }
    )


def _assert_contract(cfg: Any) -> None:
    forbidden = ("actuator_model", "compensation", "servo_target_planner")
    enabled = [name for name in forbidden if bool(cfg.plant[name])]
    if enabled:
        raise ValueError(f"HIL contract forbids synthetic plant features: {enabled}")
    if int(cfg.ros.causal_guard_samples) < 1:
        raise ValueError("causal_guard_samples must be >= 1")
    if not 0 <= int(cfg.ros.causal_timeout_retries) <= 2:
        raise ValueError("causal_timeout_retries must be in [0, 2]")
    if int(cfg.plant.command_history_steps) < 24:
        raise ValueError("command history must cover measured 100-120ms lag (>=24 samples at 200Hz)")
    if float(cfg.flashsac.gamma) < 0.995:
        raise ValueError("200Hz HIL gamma must cover the measured delay and ~1s launch cycle")
    if int(cfg.flashsac.buffer_min_length) < int(cfg.flashsac.sample_batch_size):
        raise ValueError("FlashSAC replay warm-up must cover at least one sample batch")
    if int(cfg.flashsac.min_confirmed_hits_before_learning) < 1:
        raise ValueError("FlashSAC learning must wait for at least one confirmed hit")
    for stage in cfg.curriculum:
        stage_dict = OmegaConf.to_container(stage, resolve=True)
        _episode_meets_stage_goal(stage_dict, {})
        if stage_dict["success_mode"] == "launch_progress":
            raise ValueError("HIL curriculum promotion must require confirmed hits")
        shaping_cap = sum(
            float(stage_dict.get(key, 0.0))
            for key in (
                "pre_hit_flatness_weight",
                "pre_hit_lift_weight",
                "pre_hit_launch_weight",
                "pre_hit_impulse_weight",
            )
        )
        if shaping_cap > 6.0 + 1e-9:
            raise ValueError("pre-hit shaping cap must stay below one valid-hit reward")
        if float(stage_dict.get("valid_hit_bonus", 0.0)) <= shaping_cap:
            raise ValueError("valid-hit bonus must dominate bounded pre-hit shaping")
        if bool(stage_dict.get("promotion_requires_evaluation", False)) and int(
            stage_dict.get("min_episodes", 0)
        ) > int(stage_dict.get("window_episodes", 0)):
            raise ValueError("evaluation curriculum min_episodes must fit its outcome window")
        if float(stage_dict.get("exploration_vertical_amplitude_scale", 1.0)) <= 0.0:
            raise ValueError("exploration vertical amplitude scale must be positive")
        if float(stage_dict.get("exploration_frequency_scale", 1.0)) <= 0.0:
            raise ValueError("exploration frequency scale must be positive")
    exploration = cfg.structured_exploration
    if not 0.0 <= float(exploration.final_probability) <= float(exploration.initial_probability) <= 1.0:
        raise ValueError("structured exploration probabilities must satisfy 0 <= final <= initial <= 1")
    if not 0.0 <= float(exploration.primitive_blend) <= 1.0:
        raise ValueError("primitive_blend must be in [0, 1]")
    if not 0.05 <= float(exploration.task_action_limit) <= 1.0:
        raise ValueError("task_action_limit must be in [0.05, 1]")
    if not 0.0 <= float(exploration.normal_up_blend) <= 1.0:
        raise ValueError("normal_up_blend must be in [0, 1]")
    if int(exploration.warmup_steps) < 0 or int(exploration.decay_steps) <= 0:
        raise ValueError("structured exploration warmup/decay steps are invalid")
    if int(exploration.deterministic_evaluation_interval_episodes) < 2:
        raise ValueError("deterministic evaluation interval must be >= 2 episodes")


def _measure_no_action_baseline(env: Any, episodes: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for episode in range(max(0, int(episodes))):
        _obs, _reset_info = env.reset()
        episode_return = 0.0
        max_hits = 0
        contact = False
        while True:
            _obs, reward, terminated, truncated, info = env.step(zero)
            episode_return += float(reward)
            max_hits = max(max_hits, int(info.get("hit_count", 0)))
            contact = bool(
                contact
                or int(info.get("contact_hold_steps", 0)) > 0
                or bool(info.get("pending_hit", False))
            )
            if terminated or truncated:
                records.append(
                    {
                        "episode": episode,
                        "return": episode_return,
                        "hits": max_hits,
                        "contact": contact,
                        "termination_reasons": list(info.get("termination_reasons", ())),
                    }
                )
                break
    return records


def _load_hil_training_state(checkpoint: str | Path | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    root = Path(checkpoint)
    state_path = root / "hil_training_state.json"
    if state_path.is_file():
        return dict(json.loads(state_path.read_text(encoding="utf-8")))
    metadata_path = root / "hil_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"interaction_step": int(metadata.get("interaction_step", 0))}
    return {}


def _save(
    agent: Any,
    run_dir: Path,
    step: int,
    cfg: Any,
    save_buffer: bool,
    hil_state: dict[str, Any] | None = None,
) -> Path:
    target = run_dir / f"step{step:09d}"
    agent.save(str(target))
    if save_buffer:
        agent.save_replay_buffer(str(target))
    saved_hil_state = dict(hil_state or {})
    saved_hil_state["interaction_step"] = int(step)
    (target / "hil_training_state.json").write_text(
        json.dumps(saved_hil_state, indent=2), encoding="utf-8"
    )
    metadata = {
        "interaction_step": step,
        "created_unix_s": time.time(),
        "algorithm": "official Holiday-Robot/FlashSAC",
        "flashsac_upstream": FLASH_UPSTREAM,
        "flashsac_commit": _flashsac_commit(),
        "single_environment": True,
        "actuator_model": False,
        "compensation": False,
        "servo_target_planner": False,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "hil_training_state": saved_hil_state,
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "hil_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return target


def _write_tensorboard_metrics(writer: Any, record: dict[str, Any], step: int) -> None:
    mapping = {
        "task/reward": "reward",
        "task/episode_return": "episode_return",
        "task/hits": "hits",
        "task/episode_max_hits": "episode_max_hits",
        "task/confirmed_hit_count": "confirmed_hit_count",
        "task/contact_detected": "episode_contact_detected",
        "task/max_ball_vz": "episode_max_ball_vz",
        "task/max_predicted_apex_rel_z": "episode_max_predicted_apex_rel_z",
        "task/pre_hit_scaffold": "pre_hit_scaffold",
        "task/pre_hit_launch_progress": "pre_hit_launch_progress",
        "task/pre_hit_impulse_progress": "pre_hit_impulse_progress",
        "task/contact_delta_vz": "contact_delta_vz",
        "task/episode_impulse_progress": "episode_pre_hit_impulse_progress",
        "task/episode_max_contact_delta_vz": "episode_max_contact_delta_vz",
        "task/valid_hit_bonus": "hil_valid_hit_bonus",
        "task/episode_valid_hit_bonus": "episode_valid_hit_bonus",
        "task/termination_penalty": "hil_termination_penalty",
        "curriculum/success_rate": "curriculum_success_rate",
        "curriculum/episodes_in_stage": "curriculum_episodes_in_stage",
        "curriculum/eligible_evaluation_episodes": "curriculum_eligible_episodes_in_stage",
        "curriculum/gravity_z": "gravity_z",
        "safety/tracking_error_deg": "tracking_error_deg",
        "safety/episode_max_tracking_error_deg": "episode_max_tracking_error_deg",
        "safety/max_feedback_dq_deg_s": "max_feedback_dq_deg_s",
        "safety/episode_max_feedback_dq_deg_s": "episode_max_feedback_dq_deg_s",
        "timing/source_dt_ms": "source_dt_ms",
        "timing/steps_per_second": "steps_per_second",
        "timing/learner_queue": "learner_queue",
        "exploration/structured_probability": "structured_exploration_probability",
        "exploration/structured_episode": "structured_exploration_episode",
        "exploration/deterministic_policy_evaluation": "policy_evaluation_episode",
        "exploration/stage_schedule_step": "stage_exploration_step",
        "exploration/recovery_count": "exploration_recovery_count",
        "exploration/task_action_norm": "structured_task_action_norm",
        "exploration/applied_action_norm": "applied_action_norm",
        "diagnostics/no_hit_streak": "no_hit_streak",
        "diagnostics/passive_return_margin": "passive_return_margin",
        "diagnostics/passive_collapse": "passive_collapse",
        "diagnostics/launch_collapse": "launch_collapse",
        "diagnostics/learning_gate_ready": "learning_gate_ready",
        "diagnostics/confirmed_hits_before_learning": "total_confirmed_hits_for_learning",
    }
    for tag, key in mapping.items():
        value = record.get(key)
        if isinstance(value, (bool, int, float)) and np.isfinite(float(value)):
            writer.add_scalar(tag, float(value), step)
    for key, value in record.items():
        if "/" in key and isinstance(value, (int, float)) and np.isfinite(float(value)):
            writer.add_scalar(f"learner/{key}", float(value), step)
    if bool(record.get("episode_done", False)):
        reasons = set(record.get("termination_reasons", ()))
        for reason in (
            "racket_too_low",
            "racket_too_high",
            "racket_too_far_from_anchor",
            "ball_too_low",
            "ball_too_high",
            "ball_x_out_of_bounds",
            "ball_y_out_of_bounds",
            "time_limit",
        ):
            writer.add_scalar(f"termination/{reason}", float(reason in reasons), step)
        writer.add_text("termination/latest_reasons", ", ".join(sorted(reasons)) or "none", step)


def run(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    if args.smoke_steps is not None:
        smoke_steps = max(2, int(args.smoke_steps))
        cfg.total_steps = smoke_steps
        cfg.flashsac.buffer_max_length = max(1024, smoke_steps + 1)
        cfg.flashsac.buffer_min_length = 64 if args.smoke_learn else max(1024, smoke_steps + 1)
        if args.smoke_learn:
            cfg.flashsac.sample_batch_size = min(
                64, int(cfg.flashsac.sample_batch_size)
            )
        cfg.checkpoint_every = smoke_steps + 1
        cfg.save_replay_every = None
    _assert_contract(cfg)
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)

    if not FLASH_ROOT.is_dir():
        raise FileNotFoundError(f"official FlashSAC checkout missing: {FLASH_ROOT}")
    sys.path.insert(0, str(FLASH_ROOT))
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from flash_rl.agents import create_agent

    if (
        not args.reset_only
        and str(cfg.flashsac.device_type).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("FlashSAC is configured for CUDA but torch.cuda.is_available() is false")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    simcode = load_simcode4_env(args.simcode4_rl_dir)
    env_cfg = build_simcode4_hil_config(simcode)
    plant_cfg = PlantConfig(
        joint_topic=str(cfg.ros.joint_topic),
        command_topic=str(cfg.ros.command_topic),
        right_arm_mu_idx=int(cfg.ros.right_arm_mu_idx),
        control_hz=float(cfg.ros.control_hz),
        state_timeout_s=float(cfg.ros.state_timeout_s),
        command_watchdog_s=float(cfg.ros.command_watchdog_s),
        causal_guard_samples=int(cfg.ros.causal_guard_samples),
        causal_timeout_retries=int(cfg.ros.causal_timeout_retries),
        allow_existing_command_publisher=bool(args.allow_existing_command_publisher),
    )
    runtime = RosPlantRuntime(plant_cfg)
    env = None
    agent = None
    stop = False

    def request_stop(_signum=None, _frame=None):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    resume_hil_state = _load_hil_training_state(args.resume)
    start_step = int(resume_hil_state.get("interaction_step", 0))
    run_dir = _resolve_run_dir(str(cfg.output_dir), args.resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    tensorboard_writer = SummaryWriter(
        log_dir=str(run_dir / "tensorboard"),
        flush_secs=5,
        purge_step=(start_step + 1 if start_step > 0 else None),
    )
    tensorboard_writer.add_text("run/config", OmegaConf.to_yaml(cfg), start_step)

    try:
        startup_sample = runtime.plant.wait_for_state(timeout_s=2.0)
        runtime.plant.wait_for_command_subscriber(timeout_s=2.0)
        reset_q = np.deg2rad(np.asarray(cfg.plant.reset_degrees, dtype=np.float64))
        startup_error_by_joint_deg = np.rad2deg(np.abs(startup_sample.q - reset_q))
        recovery_limit_by_joint_deg = np.asarray(
            cfg.plant.max_pose_deviation_deg,
            dtype=np.float64,
        )
        startup_error_deg = float(np.max(startup_error_by_joint_deg))
        outside = np.flatnonzero(startup_error_by_joint_deg > recovery_limit_by_joint_deg)
        if outside.size and _is_known_simulator_idle_pose(startup_sample.q, cfg.plant):
            print(
                "measured simulator idle pose detected; performing configured "
                "slow simulator-only recovery to D455",
                flush=True,
            )
            startup_sample = runtime.plant.move_to_pose(
                reset_q,
                speed_deg_s=float(cfg.plant.simulator_idle_recovery_speed_deg_s),
                tolerance_deg=float(cfg.plant.reset_tolerance_deg),
                timeout_s=float(cfg.plant.simulator_idle_recovery_timeout_s),
            )
            startup_error_by_joint_deg = np.rad2deg(np.abs(startup_sample.q - reset_q))
            startup_error_deg = float(np.max(startup_error_by_joint_deg))
            outside = np.flatnonzero(startup_error_by_joint_deg > recovery_limit_by_joint_deg)
        if outside.size:
            details = ", ".join(
                f"j{idx}={startup_error_by_joint_deg[idx]:.2f}>{recovery_limit_by_joint_deg[idx]:.2f}deg"
                for idx in outside
            )
            raise PlantSafetyError(
                f"startup pose outside automatic recovery window: {details}"
            )
        # Every run starts from the same feedback-confirmed D455 pose.  The
        # wider per-joint window above only decides whether automatic recovery
        # is safe; it must not silently accept a several-degree policy residue.
        if startup_error_deg > float(cfg.plant.reset_tolerance_deg):
            startup_sample = runtime.plant.move_to_pose(
                reset_q,
                speed_deg_s=float(cfg.plant.reset_speed_deg_s),
                tolerance_deg=float(cfg.plant.reset_tolerance_deg),
                timeout_s=float(cfg.plant.reset_timeout_s),
            )
        if args.reset_only:
            final_error_deg = np.rad2deg(startup_sample.q - reset_q)
            print(
                json.dumps(
                    {
                        "reset_only": True,
                        "feedback_q_deg": np.rad2deg(startup_sample.q).round(6).tolist(),
                        "error_deg": final_error_deg.round(6).tolist(),
                        "max_abs_error_deg": float(np.max(np.abs(final_error_deg))),
                    },
                    sort_keys=True,
                )
            )
            return
        HilEnv = make_hil_env_class(simcode)
        hil_cfg = HilSafetyConfig(
            command_history_steps=int(cfg.plant.command_history_steps),
            max_command_delta_deg=float(cfg.plant.max_command_delta_deg),
            max_tracking_error_deg=float(cfg.plant.max_tracking_error_deg),
            max_feedback_velocity_deg_s=float(cfg.plant.max_feedback_velocity_deg_s),
            reset_degrees=tuple(float(v) for v in cfg.plant.reset_degrees),
            max_pose_deviation_deg=tuple(float(v) for v in cfg.plant.max_pose_deviation_deg),
            episode_reset_threshold_deg=float(cfg.plant.episode_reset_threshold_deg),
            reset_speed_deg_s=float(cfg.plant.reset_speed_deg_s),
            reset_tolerance_deg=float(cfg.plant.reset_tolerance_deg),
            reset_timeout_s=float(cfg.plant.reset_timeout_s),
        )
        env = HilEnv(xml_path=str(args.xml), plant=runtime.plant, cfg=env_cfg, hil_cfg=hil_cfg)
        obs, env_info = env.reset(seed=seed)
        # Arm the watchdog with an exact feedback-position hold while CUDA
        # networks and optional torch.compile state are initialized.
        runtime.plant.publish(env._last_plant_sample.q)
        agent = create_agent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env_info=env_info,
            cfg=_agent_config(
                cfg.flashsac,
                int(cfg.total_steps),
                seed,
                float(cfg.updates_per_step),
            ),
        )
        if args.resume:
            agent.load(args.resume)
        if args.resume_buffer:
            agent.load_replay_buffer(args.resume_buffer)
        checkpoint_state: dict[str, Any] = dict(resume_hil_state)

        # Keep learning off the 200 Hz HIL path. The learner owns the official
        # agent and replay buffer; control uses a small actor-only snapshot.
        learner_actor = agent._actor.network  # official FlashSAC implementation
        learner_actor = getattr(learner_actor, "_orig_mod", learner_actor)
        actor_mirror = copy.deepcopy(learner_actor).to(agent._device)
        # The official agent may replace this instance method with a
        # torch.compile wrapper. The real-time snapshot must call the original
        # class implementation rather than nesting Dynamo/FX wrappers.
        actor_mirror.get_mean_and_std = type(actor_mirror).get_mean_and_std.__get__(
            actor_mirror,
            type(actor_mirror),
        )
        actor_mirror.eval()
        inference_stream = torch.cuda.Stream(priority=-1) if agent._device.type == "cuda" else None
        actor_lock = threading.Lock()
        transition_queue: queue.Queue[tuple[int, dict[str, np.ndarray]]] = queue.Queue(maxsize=20_000)
        learner_stop = threading.Event()
        learning_gate = threading.Event()
        if args.smoke_learn:
            learning_gate.set()
        training_ready = threading.Event()
        learner_errors: list[BaseException] = []
        learner_stats: dict[str, Any] = {}
        learner_stats_lock = threading.Lock()

        def sync_actor() -> None:
            source = agent._actor.network
            source = getattr(source, "_orig_mod", source)
            state = {key: value.detach().clone() for key, value in source.state_dict().items()}
            with actor_lock:
                actor_mirror.load_state_dict(state)
                if inference_stream is not None:
                    weights_ready = torch.cuda.Event()
                    weights_ready.record(torch.cuda.current_stream(agent._device))
                    inference_stream.wait_event(weights_ready)

        def learner_loop() -> None:
            update_credit_local = 0.0
            updates = 0
            try:
                while not learner_stop.is_set() or not transition_queue.empty():
                    try:
                        item_step, item_transition = transition_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    agent.process_transition(item_transition)
                    if agent.can_start_training() and learning_gate.is_set():
                        update_credit_local += float(cfg.updates_per_step)
                        while update_credit_local >= 1.0:
                            update_result = agent.update()
                            with learner_stats_lock:
                                learner_stats.clear()
                                learner_stats.update(update_result)
                            update_credit_local -= 1.0
                            updates += 1
                            if updates == 1 or updates % 20 == 0:
                                sync_actor()
                            training_ready.set()
                    if item_step % int(cfg.checkpoint_every) == 0:
                        _save(
                            agent, run_dir, item_step, cfg, save_buffer=False,
                            hil_state=checkpoint_state,
                        )
                    if item_step == int(cfg.flashsac.buffer_min_length):
                        _save(
                            agent, run_dir, item_step, cfg, save_buffer=True,
                            hil_state=checkpoint_state,
                        )
                    if cfg.save_replay_every and item_step % int(cfg.save_replay_every) == 0:
                        _save(
                            agent, run_dir, item_step, cfg, save_buffer=True,
                            hil_state=checkpoint_state,
                        )
                    transition_queue.task_done()
            except BaseException as exc:
                learner_errors.append(exc)

        learner_thread = threading.Thread(target=learner_loop, name="flashsac_learner", daemon=True)
        learner_thread.start()

        exploration_noise = np.zeros(7, dtype=np.float32)
        exploration_repeat_remaining = 0
        zeta_cdf = _truncated_zeta_cdf(
            float(cfg.flashsac.actor_noise_zeta_mu),
            int(cfg.flashsac.actor_noise_zeta_max),
        )

        def mirror_action(observation: np.ndarray, *, stochastic: bool = True) -> np.ndarray:
            nonlocal exploration_noise, exploration_repeat_remaining
            if stochastic and exploration_repeat_remaining <= 0:
                exploration_noise = np.random.randn(7).astype(np.float32)
                exploration_repeat_remaining = _sample_truncated_zeta(zeta_cdf)
            stream_context = torch.cuda.stream(inference_stream) if inference_stream is not None else nullcontext()
            with actor_lock, torch.no_grad(), stream_context:
                tensor_obs = torch.as_tensor(observation[None, :], dtype=torch.float32, device=agent._device)
                mean, std = actor_mirror.get_mean_and_std(tensor_obs, training=False)
                if stochastic:
                    noise = torch.as_tensor(exploration_noise[None, :], device=agent._device)
                    result = torch.tanh(mean + std * noise)
                else:
                    # Matches official FlashSAC sample_actions(training=False).
                    result = torch.tanh(mean)
            if stochastic:
                exploration_repeat_remaining -= 1
            return result[0].cpu().numpy().astype(np.float32)

        stages = [OmegaConf.to_container(stage, resolve=True) for stage in cfg.curriculum]
        curriculum = CompetenceCurriculum(stages)
        if "curriculum" in resume_hil_state:
            curriculum.load_state_dict(dict(resume_hil_state["curriculum"]))
        if args.resume_stage:
            curriculum.reset_to_stage(str(args.resume_stage))
        stage_idx = curriculum.index
        if args.smoke_steps is not None:
            stages[0]["init_vz"] = 0.0
            stages[0]["horizon_sec"] = max(1.0, int(cfg.total_steps) / float(cfg.ros.control_hz))
        env.set_curriculum_stage(stages[stage_idx])
        no_action_baseline_return = 0.0
        if args.smoke_steps is not None:
            # Keep the diagnostic ball clear of the robot so it cannot reset
            # the command integrator during the one-second arm round trip.
            env.cfg.ball_launch_height = 0.60
            env.model.geom_contype[env.ball_geom_id] = 0
            env.model.geom_conaffinity[env.ball_geom_id] = 0
            obs, env_info = env.reset()
            env.model.opt.gravity[2] = 0.0
        else:
            # Apply stage-specific gravity/contact geometry to the very first
            # learning episode; the constructor reset used only base defaults.
            env.set_evaluation_mode(True)
            obs, env_info = env.reset()
            baseline_records = _measure_no_action_baseline(
                env, int(cfg.diagnostics.no_action_baseline_episodes)
            )
            if baseline_records:
                no_action_baseline_return = float(
                    np.mean([record["return"] for record in baseline_records])
                )
                (run_dir / "no_action_baseline.json").write_text(
                    json.dumps(
                        {
                            "mean_return": no_action_baseline_return,
                            "episodes": baseline_records,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                tensorboard_writer.add_scalar(
                    "diagnostics/no_action_baseline_return", no_action_baseline_return, 0
                )
                for baseline in baseline_records:
                    tensorboard_writer.add_scalar(
                        "diagnostics/no_action_episode_return",
                        float(baseline["return"]),
                        int(baseline["episode"]),
                    )
            env.set_evaluation_mode(False)
            obs, env_info = env.reset()

        exploration_cfg = cfg.structured_exploration

        stage_schedule_origin_step = int(
            resume_hil_state.get("stage_schedule_origin_step", start_step)
        )
        exploration_recovery_count = int(
            resume_hil_state.get("exploration_recovery_count", 0)
        )

        def structured_probability(stage_interaction_step: int) -> float:
            if not bool(exploration_cfg.enabled) or args.smoke_steps is not None:
                return 0.0
            decay_step = max(
                0.0,
                float(stage_interaction_step) - float(exploration_cfg.warmup_steps),
            )
            fraction = min(
                1.0,
                decay_step / max(1.0, float(exploration_cfg.decay_steps)),
            )
            return float(
                float(exploration_cfg.initial_probability)
                + fraction
                * (
                    float(exploration_cfg.final_probability)
                    - float(exploration_cfg.initial_probability)
                )
            )

        def sample_exploration_episode(
            stage_interaction_step: int, *, evaluation_episode: bool = False
        ) -> tuple[bool, dict[str, Any]]:
            probability = structured_probability(stage_interaction_step)
            enabled = bool(not evaluation_episode and np.random.random() < probability)
            stage = curriculum.stage
            frequency_scale = float(stage.get("exploration_frequency_scale", 1.0))
            amplitude_scale = float(
                stage.get("exploration_vertical_amplitude_scale", 1.0)
            )
            frequency_hz = float(
                np.random.uniform(
                    float(exploration_cfg.frequency_min_hz),
                    float(exploration_cfg.frequency_max_hz),
                )
            ) * frequency_scale
            parameters = {
                "frequency_hz": frequency_hz,
                "vertical_amplitude_m": float(
                    np.random.uniform(
                        float(exploration_cfg.vertical_amplitude_min_m),
                        float(exploration_cfg.vertical_amplitude_max_m),
                    )
                ) * amplitude_scale,
                "vertical_phase": 0.0
                if np.random.random() < float(exploration_cfg.upward_phase_probability)
                else float(np.random.uniform(0.0, 2.0 * np.pi)),
                "xy_amplitude_m": np.random.uniform(
                    0.0, float(exploration_cfg.xy_amplitude_max_m), size=2
                ).astype(np.float64),
                "xy_frequency_hz": frequency_hz
                * np.random.uniform(
                    float(exploration_cfg.xy_frequency_ratio_min),
                    float(exploration_cfg.xy_frequency_ratio_max),
                    size=2,
                ),
                "xy_phase": np.random.uniform(0.0, 2.0 * np.pi, size=2),
                "tilt_amplitude": np.tan(
                    np.deg2rad(
                        np.random.uniform(
                            -float(exploration_cfg.normal_tilt_max_deg),
                            float(exploration_cfg.normal_tilt_max_deg),
                            size=2,
                        )
                    )
                ),
                "tilt_phase": np.random.uniform(0.0, 2.0 * np.pi, size=2),
            }
            return enabled, parameters

        def task_space_noise(
            parameters: dict[str, Any], elapsed_s: float
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            vertical_omega = 2.0 * np.pi * float(parameters["frequency_hz"])
            vertical_phase0 = float(parameters["vertical_phase"])
            vertical_phase = vertical_omega * elapsed_s + vertical_phase0
            vertical_amplitude = float(parameters["vertical_amplitude_m"])
            position = np.zeros(3, dtype=np.float64)
            velocity = np.zeros(3, dtype=np.float64)
            acceleration = np.zeros(3, dtype=np.float64)
            position[2] = vertical_amplitude * (
                np.sin(vertical_phase) - np.sin(vertical_phase0)
            )
            velocity[2] = vertical_amplitude * vertical_omega * np.cos(vertical_phase)
            acceleration[2] = -vertical_amplitude * vertical_omega**2 * np.sin(
                vertical_phase
            )
            xy_amplitude = np.asarray(parameters["xy_amplitude_m"], dtype=np.float64)
            xy_omega = 2.0 * np.pi * np.asarray(
                parameters["xy_frequency_hz"], dtype=np.float64
            )
            xy_phase0 = np.asarray(parameters["xy_phase"], dtype=np.float64)
            xy_phase = xy_omega * elapsed_s + xy_phase0
            position[:2] = xy_amplitude * (np.sin(xy_phase) - np.sin(xy_phase0))
            velocity[:2] = xy_amplitude * xy_omega * np.cos(xy_phase)
            acceleration[:2] = -xy_amplitude * xy_omega**2 * np.sin(xy_phase)

            tilt_amplitude = np.asarray(parameters["tilt_amplitude"], dtype=np.float64)
            tilt_phase = vertical_omega * elapsed_s + np.asarray(
                parameters["tilt_phase"], dtype=np.float64
            )
            normal_tilt = tilt_amplitude * np.sin(tilt_phase)
            normal_tilt_velocity = tilt_amplitude * vertical_omega * np.cos(tilt_phase)
            normal_tilt_acceleration = -tilt_amplitude * vertical_omega**2 * np.sin(
                tilt_phase
            )
            angular_velocity = np.asarray(
                [-normal_tilt_velocity[1], normal_tilt_velocity[0], 0.0]
            )
            angular_acceleration = np.asarray(
                [-normal_tilt_acceleration[1], normal_tilt_acceleration[0], 0.0]
            )
            return (
                position,
                velocity,
                acceleration,
                normal_tilt,
                angular_velocity,
                angular_acceleration,
            )

        policy_evaluation_episode = False
        structured_episode, structured_parameters = sample_exploration_episode(
            max(0, start_step - stage_schedule_origin_step)
        )
        exploration_episode_step = 0
        diagnostic_window = max(
            1, int(cfg.diagnostics.passive_collapse_window_episodes)
        )
        recent_episode_returns: deque[float] = deque(maxlen=diagnostic_window)
        recent_episode_hits: deque[int] = deque(maxlen=diagnostic_window)
        recent_episode_max_vz: deque[float] = deque(maxlen=diagnostic_window)
        recent_episode_contacts: deque[bool] = deque(maxlen=diagnostic_window)
        no_hit_streak = int(resume_hil_state.get("no_hit_streak", 0))
        passive_return_margin = 0.0
        passive_collapse = False
        launch_collapse = False
        collapse_recovery_active = False
        total_confirmed_hits = int(
            resume_hil_state.get("total_confirmed_hits_for_learning", 0)
        )
        if total_confirmed_hits >= int(cfg.flashsac.min_confirmed_hits_before_learning):
            learning_gate.set()
        if args.resume:
            # The loaded actor is already trained and safe to use immediately;
            # learner updates still wait for replay readiness and the hit gate.
            sync_actor()
            training_ready.set()
        episode_return = 0.0
        episode_length = 0
        episode_max_tracking_error_deg = 0.0
        episode_max_feedback_dq_deg_s = 0.0
        episode_q_min_deg = np.full(7, np.inf, dtype=np.float64)
        episode_q_max_deg = np.full(7, -np.inf, dtype=np.float64)
        episode_contact_detected = False
        episode_max_ball_vz = -np.inf
        episode_max_predicted_apex_rel_z = -np.inf
        episode_max_hits = 0
        episode_valid_hit_bonus = 0.0
        episode_pre_hit_impulse_progress = 0.0
        episode_max_contact_delta_vz = 0.0
        start = time.monotonic()

        def refresh_checkpoint_state(interaction_step: int) -> None:
            nonlocal checkpoint_state
            checkpoint_state = {
                "interaction_step": int(interaction_step),
                "curriculum": curriculum.state_dict(),
                "stage_schedule_origin_step": int(stage_schedule_origin_step),
                "exploration_recovery_count": int(exploration_recovery_count),
                "total_confirmed_hits_for_learning": int(total_confirmed_hits),
                "no_hit_streak": int(no_hit_streak),
            }

        refresh_checkpoint_state(start_step)

        for step in range(start_step + 1, int(cfg.total_steps) + 1):
            if stop:
                break
            if learner_errors:
                raise RuntimeError("FlashSAC learner thread failed") from learner_errors[0]
            if training_ready.is_set():
                action = mirror_action(obs, stochastic=not policy_evaluation_episode)
                if args.smoke_steps is not None:
                    action *= np.float32(0.10)
            else:
                # Bounded startup exploration; the environment additionally
                # enforces per-command and feedback safety limits.
                if args.smoke_steps is not None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    phase = step / float(cfg.total_steps)
                    action[-1] = 0.05 if phase <= 0.25 or phase > 0.75 else -0.05
                else:
                    # Temporally correlated bootstrap exploration is much more
                    # useful to a delayed arm than independent 200Hz jitter.
                    if exploration_repeat_remaining <= 0:
                        exploration_noise = np.random.uniform(
                            -0.45, 0.45, size=env.action_space.shape
                        ).astype(np.float32)
                        exploration_repeat_remaining = _sample_truncated_zeta(zeta_cdf)
                    action = exploration_noise.copy()
                    exploration_repeat_remaining -= 1

            stage_interaction_step = max(0, step - stage_schedule_origin_step)
            current_structured_probability = structured_probability(stage_interaction_step)
            structured_task_action_norm = 0.0
            if structured_episode and args.smoke_steps is None:
                trajectory = task_space_noise(
                    structured_parameters,
                    exploration_episode_step / max(float(cfg.ros.control_hz), 1.0),
                )
                if bool(getattr(env, "pending_hit", False)):
                    # Once contact creates a candidate, pull the paddle away
                    # so the ball can satisfy the existing 4 cm separation
                    # confirmation gate. Continuing the upward harmonic made
                    # the paddle chase the ball and hid otherwise good launch
                    # samples from FlashSAC.
                    trajectory = tuple(np.asarray(value).copy() for value in trajectory)
                    trajectory[0][2] = min(float(trajectory[0][2]), -0.015)
                    trajectory[1][2] = min(float(trajectory[1][2]), -0.30)
                    trajectory[2][2] = min(float(trajectory[2][2]), -1.5)
                primitive_action = env.coordinated_exploration_action(
                    position_offset_m=trajectory[0],
                    linear_velocity_m_s=trajectory[1],
                    linear_acceleration_m_s2=trajectory[2],
                    normal_tilt_xy=trajectory[3],
                    angular_velocity_rad_s=trajectory[4],
                    angular_acceleration_rad_s2=trajectory[5],
                    position_kp=float(exploration_cfg.position_kp),
                    position_kd=float(exploration_cfg.position_kd),
                    orientation_kp=float(exploration_cfg.orientation_kp),
                    orientation_kd=float(exploration_cfg.orientation_kd),
                    normal_up_blend=float(exploration_cfg.normal_up_blend),
                    max_action_abs=float(exploration_cfg.task_action_limit),
                )
                structured_task_action_norm = float(np.linalg.norm(primitive_action))
                blend = float(exploration_cfg.primitive_blend)
                action = np.clip(
                    blend * primitive_action + (1.0 - blend) * action,
                    -1.0,
                    1.0,
                ).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            exploration_episode_step += 1
            transition = {
                "observation": obs[None, :],
                "action": action[None, :],
                "reward": np.asarray([reward], dtype=np.float32),
                "terminated": np.asarray([terminated], dtype=np.bool_),
                "truncated": np.asarray([truncated], dtype=np.bool_),
                "next_observation": next_obs[None, :],
            }
            try:
                transition_queue.put_nowait((step, transition))
            except queue.Full as exc:
                raise RuntimeError("learner fell more than 20000 samples behind real-time control") from exc
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1
            episode_max_tracking_error_deg = max(
                episode_max_tracking_error_deg,
                float(info["hil/tracking_error_deg"]),
            )
            episode_max_feedback_dq_deg_s = max(
                episode_max_feedback_dq_deg_s,
                float(info["hil/max_feedback_dq_deg_s"]),
            )
            feedback_q_deg = np.asarray(info["hil/feedback_q_deg"], dtype=np.float64)
            episode_q_min_deg = np.minimum(episode_q_min_deg, feedback_q_deg)
            episode_q_max_deg = np.maximum(episode_q_max_deg, feedback_q_deg)
            episode_contact_detected = bool(
                episode_contact_detected
                or int(info.get("contact_hold_steps", 0)) > 0
                or bool(info.get("pending_hit", False))
            )
            episode_max_ball_vz = max(episode_max_ball_vz, float(info.get("ball_vz", -np.inf)))
            episode_max_predicted_apex_rel_z = max(
                episode_max_predicted_apex_rel_z,
                float(info.get("predicted_apex_rel_z", -np.inf)),
            )
            episode_max_hits = max(
                episode_max_hits,
                int(info.get("hit_count", info.get("confirmed_hit_count", 0))),
            )
            episode_valid_hit_bonus += float(info.get("hil/valid_hit_bonus", 0.0))
            episode_pre_hit_impulse_progress += float(
                info.get("hil/pre_hit_impulse_progress", 0.0)
            )
            episode_max_contact_delta_vz = max(
                episode_max_contact_delta_vz,
                float(info.get("hil/contact_delta_vz", 0.0)),
            )
            done = bool(terminated or truncated)
            if done:
                recent_episode_returns.append(float(episode_return))
                recent_episode_hits.append(int(episode_max_hits))
                recent_episode_max_vz.append(float(episode_max_ball_vz))
                recent_episode_contacts.append(bool(episode_contact_detected))
                total_confirmed_hits += int(episode_max_hits)
                if total_confirmed_hits >= int(
                    cfg.flashsac.min_confirmed_hits_before_learning
                ):
                    learning_gate.set()
                no_hit_streak = 0 if episode_max_hits > 0 else no_hit_streak + 1
                if episode_max_hits > 0:
                    collapse_recovery_active = False
                passive_return_margin = float(
                    np.mean(recent_episode_returns) - no_action_baseline_return
                )
                passive_collapse = bool(
                    len(recent_episode_returns) >= diagnostic_window
                    and sum(recent_episode_hits) == 0
                    and passive_return_margin >= -0.5
                )
                launch_collapse = bool(
                    len(recent_episode_max_vz) >= diagnostic_window
                    and sum(recent_episode_hits) == 0
                    and float(np.mean(recent_episode_contacts))
                    >= float(cfg.diagnostics.launch_collapse_min_contact_rate)
                    and float(np.mean(recent_episode_max_vz))
                    < float(cfg.diagnostics.launch_collapse_max_vz_m_s)
                )
                if launch_collapse and not collapse_recovery_active:
                    # Restore full task-space coverage once per collapse.  A
                    # later confirmed hit rearms this detector.
                    stage_schedule_origin_step = step
                    exploration_recovery_count += 1
                    collapse_recovery_active = True
            if step % int(cfg.log_every) == 0 or done:
                with learner_stats_lock:
                    update_snapshot = dict(learner_stats)
                record = {
                    "step": step,
                    "stage": stages[stage_idx]["name"],
                    "reward": float(reward),
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "hits": int(info.get("hit_count", info.get("confirmed_hit_count", 0))),
                    "new_hit": bool(info.get("new_hit", False)),
                    "confirmed_hit_count": int(info.get("confirmed_hit_count", 0)),
                    "pending_hit": bool(info.get("pending_hit", False)),
                    "contact_hold_steps": int(info.get("contact_hold_steps", 0)),
                    "racket_up_cos": float(info.get("racket_up_cos", 0.0)),
                    "contact_center_dist": float(info.get("contact_center_dist", 0.0)),
                    "predicted_apex_z": float(info.get("predicted_apex_z", 0.0)),
                    "ball_vz": float(info.get("ball_vz", 0.0)),
                    "pre_hit_scaffold": float(info.get("hil/pre_hit_scaffold", 0.0)),
                    "pre_hit_launch_progress": float(info.get("hil/pre_hit_launch_progress", 0.0)),
                    "pre_hit_impulse_progress": float(info.get("hil/pre_hit_impulse_progress", 0.0)),
                    "contact_delta_vz": float(info.get("hil/contact_delta_vz", 0.0)),
                    "episode_pre_hit_impulse_progress": episode_pre_hit_impulse_progress,
                    "episode_max_contact_delta_vz": episode_max_contact_delta_vz,
                    "hil_valid_hit_bonus": float(info.get("hil/valid_hit_bonus", 0.0)),
                    "episode_valid_hit_bonus": episode_valid_hit_bonus,
                    "hil_termination_penalty": float(info.get("hil/termination_penalty", 0.0)),
                    "ball_termination_min_z": float(info.get("hil/ball_termination_min_z", 0.0)),
                    "ball_termination_max_z": float(info.get("hil/ball_termination_max_z", 0.0)),
                    "gravity_z": float(info.get("hil/gravity_z", 0.0)),
                    "causal_timeout_recoveries": int(
                        info.get("hil/causal_timeout_recoveries", 0)
                    ),
                    "structured_exploration_probability": current_structured_probability,
                    "structured_exploration_episode": bool(structured_episode),
                    "policy_evaluation_episode": bool(policy_evaluation_episode),
                    "stage_exploration_step": int(stage_interaction_step),
                    "exploration_recovery_count": int(exploration_recovery_count),
                    "structured_task_action_norm": structured_task_action_norm,
                    "applied_action_norm": float(np.linalg.norm(action)),
                    "no_action_baseline_return": no_action_baseline_return,
                    "no_hit_streak": no_hit_streak,
                    "passive_return_margin": passive_return_margin,
                    "passive_collapse": passive_collapse,
                    "launch_collapse": launch_collapse,
                    "learning_gate_ready": learning_gate.is_set(),
                    "total_confirmed_hits_for_learning": total_confirmed_hits,
                    "episode_contact_detected": episode_contact_detected,
                    "episode_max_ball_vz": episode_max_ball_vz,
                    "episode_max_predicted_apex_rel_z": episode_max_predicted_apex_rel_z,
                    "episode_max_hits": episode_max_hits,
                    "curriculum_success_rate": curriculum.success_rate,
                    "curriculum_episodes_in_stage": curriculum.episodes_in_stage,
                    "curriculum_eligible_episodes_in_stage": (
                        curriculum.eligible_episodes_in_stage
                    ),
                    "termination_reasons": list(info.get("termination_reasons", ())),
                    "episode_done": done,
                    "source_dt_ms": 1000.0 * float(info["hil/source_dt_s"]),
                    "max_qdd_ratio": float(info["hil/max_qdd_ratio"]),
                    "tracking_error_deg": float(info["hil/tracking_error_deg"]),
                    "max_feedback_dq_deg_s": float(info["hil/max_feedback_dq_deg_s"]),
                    "episode_max_tracking_error_deg": episode_max_tracking_error_deg,
                    "episode_max_feedback_dq_deg_s": episode_max_feedback_dq_deg_s,
                    "episode_feedback_span_deg": (episode_q_max_deg - episode_q_min_deg).round(6).tolist(),
                    "steps_per_second": (step - start_step)
                    / max(time.monotonic() - start, 1e-6),
                    "learner_queue": transition_queue.qsize(),
                    **{k: float(v) for k, v in update_snapshot.items()},
                }
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                _write_tensorboard_metrics(tensorboard_writer, record, step)
                print(json.dumps(record, sort_keys=True), flush=True)

            if done:
                success, curriculum_event = curriculum.observe_episode(
                    {
                        "contact_detected": episode_contact_detected,
                        "max_ball_vz": episode_max_ball_vz,
                        "max_predicted_apex_rel_z": episode_max_predicted_apex_rel_z,
                        "max_hits": episode_max_hits,
                        "evaluation_episode": policy_evaluation_episode,
                    }
                )
                if curriculum_event is not None:
                    stage_idx = curriculum.index
                    env.set_curriculum_stage(curriculum.stage)
                    stage_schedule_origin_step = step
                    collapse_recovery_active = False
                    launch_collapse = False
                    recent_episode_returns.clear()
                    recent_episode_hits.clear()
                    recent_episode_max_vz.clear()
                    recent_episode_contacts.clear()
                    env.set_evaluation_mode(True)
                    baseline_records = _measure_no_action_baseline(
                        env, int(cfg.diagnostics.no_action_baseline_episodes)
                    )
                    if baseline_records:
                        no_action_baseline_return = float(
                            np.mean([entry["return"] for entry in baseline_records])
                        )
                        (run_dir / f"no_action_baseline_{curriculum.stage['name']}_{step}.json").write_text(
                            json.dumps(
                                {
                                    "stage": curriculum.stage["name"],
                                    "step": step,
                                    "mean_return": no_action_baseline_return,
                                    "episodes": baseline_records,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        tensorboard_writer.add_scalar(
                            "diagnostics/no_action_baseline_return",
                            no_action_baseline_return,
                            step,
                        )
                    event_record = {
                        "step": step,
                        "curriculum_event": curriculum_event,
                        "episode_success": success,
                    }
                    with metrics_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(event_record, sort_keys=True) + "\n")
                    tensorboard_writer.add_scalar("curriculum/stage_index", stage_idx, step)
                    tensorboard_writer.add_text(
                        "curriculum/events", json.dumps(curriculum_event, sort_keys=True), step
                    )
                    print(json.dumps(event_record, sort_keys=True), flush=True)
                evaluation_interval = int(
                    exploration_cfg.deterministic_evaluation_interval_episodes
                )
                policy_evaluation_episode = bool(
                    training_ready.is_set()
                    and (curriculum.episodes_in_stage + 1) % evaluation_interval == 0
                )
                env.set_evaluation_mode(policy_evaluation_episode)
                obs, env_info = env.reset()
                episode_return = 0.0
                episode_length = 0
                episode_max_tracking_error_deg = 0.0
                episode_max_feedback_dq_deg_s = 0.0
                episode_q_min_deg.fill(np.inf)
                episode_q_max_deg.fill(-np.inf)
                episode_contact_detected = False
                episode_max_ball_vz = -np.inf
                episode_max_predicted_apex_rel_z = -np.inf
                episode_max_hits = 0
                episode_valid_hit_bonus = 0.0
                episode_pre_hit_impulse_progress = 0.0
                episode_max_contact_delta_vz = 0.0
                structured_episode, structured_parameters = sample_exploration_episode(
                    max(0, step - stage_schedule_origin_step),
                    evaluation_episode=policy_evaluation_episode,
                )
                exploration_episode_step = 0
                refresh_checkpoint_state(step)

        final_step = step if "step" in locals() else 0
        learner_stop.set()
        learner_thread.join(timeout=120.0)
        if learner_thread.is_alive():
            raise RuntimeError("learner did not drain its transition queue within 120s")
        if learner_errors:
            raise RuntimeError("FlashSAC learner thread failed") from learner_errors[0]
        if agent is not None:
            saved = _save(
                agent,
                run_dir,
                final_step,
                cfg,
                save_buffer=bool(args.save_final_buffer),
                hil_state=checkpoint_state,
            )
            print(f"final checkpoint: {saved}")
    except PlantSafetyError as exc:
        print(f"HIL SAFETY STOP: {exc}", file=sys.stderr)
        if "learner_stop" in locals():
            learner_stop.set()
        if "learner_thread" in locals() and learner_thread.is_alive():
            learner_thread.join(timeout=30.0)
        learner_stopped = "learner_thread" not in locals() or not learner_thread.is_alive()
        if agent is not None and learner_stopped:
            _save(
                agent, run_dir, locals().get("step", 0), cfg,
                save_buffer=False, hil_state=locals().get("checkpoint_state", {}),
            )
        elif agent is not None:
            print("learner still active; skipped racy emergency checkpoint", file=sys.stderr)
        raise
    except BaseException:
        if "learner_stop" in locals():
            learner_stop.set()
        if "learner_thread" in locals() and learner_thread.is_alive():
            learner_thread.join(timeout=30.0)
        learner_stopped = "learner_thread" not in locals() or not learner_thread.is_alive()
        if agent is not None and learner_stopped:
            _save(
                agent, run_dir, locals().get("step", 0), cfg,
                save_buffer=False, hil_state=locals().get("checkpoint_state", {}),
            )
        raise
    finally:
        if "learner_stop" in locals():
            learner_stop.set()
        if "learner_thread" in locals() and learner_thread.is_alive():
            learner_thread.join(timeout=5.0)
        if env is not None:
            env.close()
        if "tensorboard_writer" in locals():
            tensorboard_writer.flush()
            tensorboard_writer.close()
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train official FlashSAC against the ROS2 arm plant")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--simcode4-rl-dir", type=Path, default=DEFAULT_SIMCODE4_RL)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--resume")
    parser.add_argument("--resume-buffer")
    parser.add_argument(
        "--resume-stage",
        help="restart competence accounting at this stage after revising the curriculum",
    )
    parser.add_argument("--save-final-buffer", action="store_true")
    parser.add_argument(
        "--reset-only",
        action="store_true",
        help="move the right arm to the D455 reset with the low-speed feedback ramp, then exit",
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="short active HIL check with +/-0.05 startup actions and no learner updates",
    )
    parser.add_argument(
        "--smoke-learn",
        action="store_true",
        help="with --smoke-steps, start FlashSAC updates after 64 transitions",
    )
    parser.add_argument(
        "--allow-existing-command-publisher",
        action="store_true",
        help="unsafe expert override; default is exclusive ownership",
    )
    args = parser.parse_args()
    if args.smoke_learn and args.smoke_steps is None:
        parser.error("--smoke-learn requires --smoke-steps")
    run(args)


if __name__ == "__main__":
    main()
