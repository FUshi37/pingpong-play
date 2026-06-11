"""Run a multi-stage MJX/JAX PPO curriculum in one process.

This is separate from ``train_juggle_mjx_ppo.py``.  It keeps one policy and
optimizer state, then walks through named curriculum stages while rebuilding
the MJX environment for each stage config.  Within a stage, each parallel env
can carry its own randomized MJX Model for per-episode DR.

The stage names mirror ``training.md`` and use the same reward, observation,
latency, camera, and DR knobs exposed by ``MjxJuggleConfig``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import pickle
import signal
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv
from train_juggle_mjx_ppo import (
    OptimState,
    RunnerState,
    TrainState,
    adam_init,
    append_progress,
    init_params,
    make_train_fns,
    policy_value,
    save_checkpoint,
)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    total_steps: int
    cfg: MjxJuggleConfig
    notes: str = ""
    target_mean_hits: float = 2.0
    target_mean_len_frac: float = 0.20
    min_updates: int = 5
    min_recent_mean_return: float | None = None
    target_camera_visible: float | None = None
    min_recent_camera_reward_dense: float | None = None


class StopRequest:
    def __init__(self) -> None:
        self.requested = False
        self.reason = ""

    def handle_signal(self, signum, _frame) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        self.requested = True
        self.reason = f"received {name}"
        print(
            f"\n[mjx_curriculum] {self.reason}; will save a checkpoint and stop after the current update.",
            flush=True,
        )


def install_stop_handlers() -> StopRequest:
    stop = StopRequest()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop.handle_signal)
    return stop


def _strict_gate_overrides() -> dict[str, dict[str, float]]:
    """V7-derived convergence gates.

    The values are intentionally below the best observed v7 peaks and closer to
    stable plateaus.  They keep stages from advancing on short lucky windows
    while leaving room for the next stage to introduce genuinely new difficulty.
    """

    return {
        "stage1a_fixed_ball_hit_discovery": {
            "target_mean_hits": 2.0,
            "target_mean_len_frac": 0.15,
            "min_recent_mean_return": 8.0,
        },
        "stage1b_small_ball_init_randomization": {
            "target_mean_hits": 2.0,
            "target_mean_len_frac": 0.16,
            "min_recent_mean_return": 10.0,
        },
        "stage1c_center_aware_obs_noise_curriculum": {
            "target_mean_hits": 2.0,
            "target_mean_len_frac": 0.17,
            "min_recent_mean_return": 11.0,
        },
        "stage1d_active_hit_transition": {
            "target_mean_hits": 2.0,
            "target_mean_len_frac": 0.16,
            "min_recent_mean_return": 10.0,
        },
        "stage1e_hit_consolidation": {
            "target_mean_hits": 3.5,
            "target_mean_len_frac": 0.30,
            "min_recent_mean_return": 25.0,
        },
        "stage1f_hit_cadence_consolidation": {
            "target_mean_hits": 3.5,
            "target_mean_len_frac": 0.30,
            "min_recent_mean_return": 25.0,
        },
        "stage2a_gentle_centering_transition": {
            "target_mean_hits": 3.5,
            "target_mean_len_frac": 0.30,
            "min_recent_mean_return": 25.0,
        },
        "stage2b_centered_hit_consolidation": {
            "target_mean_hits": 3.5,
            "target_mean_len_frac": 0.30,
            "min_recent_mean_return": 25.0,
        },
        "stage2c_base_x_recenter_with_mild_posture": {
            "target_mean_hits": 4.0,
            "target_mean_len_frac": 0.32,
            "min_recent_mean_return": 28.0,
        },
        "stage3a_smooth_hardware_limited_action": {
            "target_mean_hits": 4.0,
            "target_mean_len_frac": 0.32,
            "min_recent_mean_return": 12.0,
        },
        "stage3b_light_camera_constraint": {
            "target_mean_hits": 9.0,
            "target_mean_len_frac": 0.75,
            "min_recent_mean_return": 30.0,
            "target_camera_visible": 0.80,
            "min_recent_camera_reward_dense": -0.02,
        },
        "stage4a_ball_only_light_dr": {
            "target_mean_hits": 6.5,
            "target_mean_len_frac": 0.45,
            "min_recent_mean_return": 10.0,
            "target_camera_visible": 0.80,
            "min_recent_camera_reward_dense": -0.02,
        },
        "stage4b_contact_dr": {
            "target_mean_hits": 9.0,
            "target_mean_len_frac": 0.65,
            "min_recent_mean_return": 15.0,
            "target_camera_visible": 0.80,
            "min_recent_camera_reward_dense": -0.02,
        },
        "stage4c_lite_actuator_dr": {
            "target_mean_hits": 8.0,
            "target_mean_len_frac": 0.60,
            "min_recent_mean_return": 14.0,
            "target_camera_visible": 0.80,
            "min_recent_camera_reward_dense": -0.02,
        },
        "stage4d_latency_dr": {
            "target_mean_hits": 4.5,
            "target_mean_len_frac": 0.30,
            "min_recent_mean_return": 3.0,
            "target_camera_visible": 0.83,
            "min_recent_camera_reward_dense": -0.02,
        },
        "stage4e_racket_mount_dr": {
            "target_mean_hits": 4.5,
            "target_mean_len_frac": 0.30,
            "min_recent_mean_return": 3.2,
            "target_camera_visible": 0.83,
            "min_recent_camera_reward_dense": -0.02,
        },
        "stage4f_final_dr_camera_dropout": {
            "target_mean_hits": 4.5,
            "target_mean_len_frac": 0.30,
            "min_recent_mean_return": 3.2,
            "target_camera_visible": 0.85,
            "min_recent_camera_reward_dense": -0.02,
        },
        "stage4g_strong_contact_dr": {
            "target_mean_hits": 4.8,
            "target_mean_len_frac": 0.32,
            "min_recent_mean_return": 3.5,
            "target_camera_visible": 0.85,
            "min_recent_camera_reward_dense": -0.02,
        },
    }


def build_curriculum(stage_steps_override: int | None = None, gate_preset: str = "v7_strict") -> list[CurriculumStage]:
    base = MjxJuggleConfig(domain_randomization=False, arm_action_limiter=True)

    stages = [
        CurriculumStage(
            "stage1a_fixed_ball_hit_discovery",
            1_000_000,
            replace(base, ball_launch_height=0.30, ball_spawn_xy_jitter=0.0, ball_spawn_z_jitter=0.0, ball_init_vxy_max=0.0, target_height=0.34),
            target_mean_hits=2.0,
            target_mean_len_frac=0.10,
        ),
        CurriculumStage(
            "stage1b_small_ball_init_randomization",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.4,
                ball_spawn_xy_jitter=0.005,
                ball_spawn_z_jitter=0.005,
                ball_init_vxy_max=0.003,
                target_height=0.36,
                posture_weight=0.05,
                arm_acc_limit_penalty_weight=0.005,
            ),
            target_mean_hits=2.0,
            target_mean_len_frac=0.10,
        ),
        CurriculumStage(
            "stage1c_center_aware_obs_noise_curriculum",
            500_000,
            replace(
                base,
                action_acc_scale=1.25,
                action_penalty_weight=0.0010,
                action_delta_penalty_weight=0.0004,
                ball_spawn_xy_jitter=0.005,
                ball_spawn_z_jitter=0.005,
                ball_init_vxy_max=0.003,
                target_height=0.38,
                posture_weight=0.10,
                ball_base_x_penalty_weight=0.30,
                ball_base_x_soft_limit=0.20,
                ball_base_vxy_penalty_weight=0.06,
                torque_penalty_weight=0.00008,
                arm_vel_limit_penalty_weight=0.03,
                arm_acc_limit_penalty_weight=0.03,
                arm_limiter_penalty_weight=0.01,
            ),
            "",
            target_mean_hits=2.0,
            target_mean_len_frac=0.12,
        ),
        CurriculumStage(
            "stage1d_active_hit_transition",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.25,
                action_penalty_weight=0.0010,
                action_delta_penalty_weight=0.0004,
                ball_launch_height=0.31,
                ball_spawn_xy_jitter=0.012,
                ball_spawn_z_jitter=0.015,
                ball_init_vxy_max=0.006,
                target_height=0.40,
                posture_weight=0.12,
                base_pose_weight=0.03,
                ball_base_x_penalty_weight=0.70,
                ball_base_x_soft_limit=0.15,
                ball_base_vxy_penalty_weight=0.08,
                torque_penalty_weight=0.00003,
                arm_vel_limit_penalty_weight=0.02,
                arm_acc_limit_penalty_weight=0.03,
                arm_limiter_penalty_weight=0.01,
            ),
            target_mean_hits=2.0,
            target_mean_len_frac=0.15,
        ),
        CurriculumStage(
            "stage1e_hit_consolidation",
            500_000,
            replace(
                base,
                action_acc_scale=0.95,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0010,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.42,
                posture_weight=0.35,
                base_pose_weight=0.08,
                ball_base_x_penalty_weight=1.50,
                ball_base_x_soft_limit=0.10,
                ball_base_vxy_penalty_weight=0.12,
                torque_penalty_weight=0.0003,
                arm_vel_limit_penalty_weight=0.10,
                arm_acc_limit_penalty_weight=0.12,
                arm_limiter_penalty_weight=0.04,
            ),
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
        ),
        CurriculumStage(
            "stage1f_hit_cadence_consolidation",
            2_000_000,
            replace(
                base,
                action_acc_scale=0.95,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0010,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.42,
                posture_weight=0.35,
                base_pose_weight=0.08,
                ball_base_x_penalty_weight=1.20,
                ball_base_x_soft_limit=0.12,
                ball_base_vxy_penalty_weight=0.12,
                torque_penalty_weight=0.0003,
                arm_vel_limit_penalty_weight=0.10,
                arm_acc_limit_penalty_weight=0.12,
                arm_limiter_penalty_weight=0.04,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
        ),
        CurriculumStage(
            "stage2a_gentle_centering_transition",
            500_000,
            replace(
                base,
                action_acc_scale=0.95,
                action_penalty_weight=0.0015,
                action_delta_penalty_weight=0.0010,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.42,
                posture_weight=0.60,
                base_pose_weight=0.10,
                ball_base_x_penalty_weight=2.5,
                ball_base_x_soft_limit=0.09,
                ball_base_vxy_penalty_weight=0.40,
                torque_penalty_weight=0.0003,
                arm_vel_limit_penalty_weight=0.05,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.03,
            ),
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
        ),
        CurriculumStage(
            "stage2b_centered_hit_consolidation",
            500_000,
            replace(
                base,
                action_acc_scale=0.95,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.42,
                posture_weight=0.90,
                base_pose_weight=0.20,
                ball_base_x_penalty_weight=4.0,
                ball_base_x_soft_limit=0.07,
                ball_base_vxy_penalty_weight=0.80,
                torque_penalty_weight=0.0004,
                arm_vel_limit_penalty_weight=0.05,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.04,
            ),
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
        ),
        CurriculumStage(
            "stage2c_base_x_recenter_with_mild_posture",
            500_000,
            replace(
                base,
                action_acc_scale=0.95,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.42,
                posture_weight=1.00,
                base_pose_weight=0.25,
                ball_anchor_xy_penalty_weight=0.40,
                ball_base_x_penalty_weight=6.0,
                ball_base_x_soft_limit=0.05,
                ball_base_vxy_penalty_weight=1.0,
                ball_vxy_penalty_weight=0.10,
                torque_penalty_weight=0.0004,
                racket_xy_gauss_reward_weight=0.20,
                racket_xy_gauss_penalty_weight=0.20,
                arm_vel_limit_penalty_weight=0.05,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.04,
            ),
            "MJX base recenter terms are partial compared with CPU env.",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
        ),
        CurriculumStage(
            "stage3a_smooth_hardware_limited_action",
            3_000_000,
            replace(
                base,
                action_acc_scale=0.95,
                action_penalty_weight=0.0020,
                action_delta_penalty_weight=0.0014,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.42,
                posture_weight=0.70,
                base_pose_weight=0.10,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=8.0,
                ball_base_x_soft_limit=0.05,
                ball_base_vxy_penalty_weight=1.50,
                torque_penalty_weight=0.0005,
                hit_reward_base=1.2,
                hit_reward_combo=0.25,
                center_flat_hit_reward_weight=1.2,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
        ),
        CurriculumStage(
            "stage3b_light_camera_constraint",
            3_000_000,
            replace(
                base,
                action_acc_scale=0.975,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.42,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=1.2,
                hit_reward_combo=0.25,
                center_flat_hit_reward_weight=1.2,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.10,
        ),
        CurriculumStage(
            "stage4a_ball_only_light_dr",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.0,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.28,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=0.5,
                hit_reward_combo=0.02,
                center_flat_hit_reward_weight=0.8,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.10,
        ),
        CurriculumStage(
            "stage4b_contact_dr",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.0,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.28,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=0.5,
                hit_reward_combo=0.02,
                center_flat_hit_reward_weight=0.8,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.10,
        ),
        CurriculumStage(
            "stage4c_lite_actuator_dr",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.0,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.28,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=0.5,
                hit_reward_combo=0.02,
                center_flat_hit_reward_weight=0.8,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.10,
        ),
        CurriculumStage(
            "stage4d_latency_dr",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.0,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.28,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=0.5,
                hit_reward_combo=0.02,
                center_flat_hit_reward_weight=0.8,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.10,
        ),
        CurriculumStage(
            "stage4e_racket_mount_dr",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.0,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.28,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=0.5,
                hit_reward_combo=0.02,
                center_flat_hit_reward_weight=0.8,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.10,
        ),
        CurriculumStage(
            "stage4f_final_dr_camera_dropout",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.0,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.28,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=0.5,
                hit_reward_combo=0.02,
                center_flat_hit_reward_weight=0.8,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.85,
            min_recent_camera_reward_dense=-0.10,
        ),
        CurriculumStage(
            "stage4g_strong_contact_dr",
            1_000_000,
            replace(
                base,
                action_acc_scale=1.0,
                action_penalty_weight=0.0018,
                action_delta_penalty_weight=0.0012,
                ball_launch_height=0.32,
                ball_spawn_xy_jitter=0.025,
                ball_spawn_z_jitter=0.035,
                ball_init_vxy_max=0.012,
                target_height=0.28,
                posture_weight=0.80,
                base_pose_weight=0.15,
                ball_anchor_xy_penalty_weight=0.60,
                ball_base_x_penalty_weight=1.0,
                ball_base_x_soft_limit=0.025,
                ball_base_vxy_penalty_weight=6.0,
                torque_penalty_weight=0.0005,
                hit_reward_base=0.5,
                hit_reward_combo=0.02,
                center_flat_hit_reward_weight=0.8,
                arm_vel_limit_penalty_weight=0.06,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.08,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            min_recent_mean_return=0.0,
            target_camera_visible=0.85,
            min_recent_camera_reward_dense=-0.10,
        ),
    ]

    cadence_1f = dict(
        hit_cadence_reward_weight=0.50,
        hit_cadence_target_interval=0.65,
        hit_cadence_sigma=0.18,
        hit_min_interval_penalty_weight=1.00,
        hit_min_interval=0.40,
        hit_min_count_interval=0.32,
        fast_hit_penalty_weight=0.30,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.65,
    )
    cadence_mid = dict(
        hit_cadence_reward_weight=0.30,
        hit_cadence_target_interval=0.65,
        hit_cadence_sigma=0.20,
        hit_min_interval_penalty_weight=0.80,
        hit_min_interval=0.40,
        hit_min_count_interval=0.32,
        fast_hit_penalty_weight=0.30,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.65,
    )
    cadence_3a = dict(
        hit_cadence_reward_weight=0.15,
        hit_cadence_target_interval=0.65,
        hit_cadence_sigma=0.20,
        hit_min_interval_penalty_weight=1.00,
        hit_min_interval=0.40,
        hit_min_count_interval=0.32,
        fast_hit_penalty_weight=0.30,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.65,
    )
    cadence_fast_032 = dict(
        hit_cadence_reward_weight=0.05,
        hit_cadence_target_interval=0.32,
        hit_cadence_sigma=0.10,
        hit_min_interval_penalty_weight=1.50,
        hit_min_interval=0.24,
        hit_min_count_interval=0.22,
        fast_hit_penalty_weight=0.80,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.32,
    )
    cadence_fast_030 = dict(
        hit_cadence_reward_weight=0.05,
        hit_cadence_target_interval=0.30,
        hit_cadence_sigma=0.10,
        hit_min_interval_penalty_weight=1.50,
        hit_min_interval=0.24,
        hit_min_count_interval=0.20,
        fast_hit_penalty_weight=0.80,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.30,
    )
    camera_stage3 = dict(
        camera_visibility_mode="pixel",
        camera_center_weight=0.25,
        camera_visibility_penalty_weight=1.0,
        camera_depth_penalty_weight=0.5,
        camera_pixel_margin=80.0,
        camera_min_depth=0.15,
        camera_max_depth=2.50,
        racket_chest_xy_penalty_weight=0.55,
        racket_chest_z_penalty_weight=0.35,
    )
    camera_stage4 = dict(
        camera_visibility_mode="pixel",
        camera_center_weight=0.5,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
        camera_pixel_margin=80.0,
        camera_min_depth=0.15,
        camera_max_depth=2.50,
        racket_chest_xy_penalty_weight=0.55,
        racket_chest_z_penalty_weight=0.35,
    )

    patched_stages = []
    for stage in stages:
        cfg = stage.cfg
        name = stage.name
        if name.startswith("stage1f"):
            cfg = replace(cfg, **cadence_1f)
        elif name.startswith(("stage2a", "stage2b", "stage2c", "stage3b")):
            cfg = replace(cfg, **cadence_mid)
        elif name.startswith("stage3a"):
            cfg = replace(cfg, **cadence_3a)
        elif name.startswith(("stage4a", "stage4b")):
            cfg = replace(cfg, **cadence_fast_032, **camera_stage4)
        elif name.startswith(("stage4c", "stage4d", "stage4e")):
            cfg = replace(cfg, **cadence_fast_030, **camera_stage4)
        elif name.startswith(("stage4f", "stage4g")):
            cfg = replace(cfg, **cadence_fast_032, **camera_stage4)

        if name.startswith("stage3b"):
            cfg = replace(cfg, **camera_stage3)
        elif name.startswith("stage4a"):
            cfg = replace(
                cfg,
                domain_randomization=True,
                dr_randomize_ball=True,
                dr_randomize_contact=False,
                dr_randomize_actuator=False,
                dr_randomize_latency=False,
            )
        elif name.startswith("stage4b"):
            cfg = replace(
                cfg,
                domain_randomization=True,
                dr_randomize_ball=True,
                dr_randomize_contact=True,
                dr_randomize_actuator=False,
                dr_randomize_latency=False,
            )
        elif name.startswith("stage4c"):
            cfg = replace(
                cfg,
                domain_randomization=True,
                dr_randomize_ball=True,
                dr_randomize_contact=True,
                dr_randomize_actuator=True,
                dr_randomize_latency=False,
                dr_action_scale_mult_range=(0.93, 1.07),
                dr_damping_mult_range=(0.85, 1.15),
                dr_armature_mult_range=(0.90, 1.10),
            )
        elif name.startswith(("stage4d", "stage4e", "stage4f", "stage4g")):
            cfg = replace(
                cfg,
                domain_randomization=True,
                dr_randomize_ball=True,
                dr_randomize_contact=True,
                dr_randomize_actuator=True,
                dr_randomize_latency=True,
                dr_action_scale_mult_range=(0.93, 1.07),
                dr_damping_mult_range=(0.85, 1.15),
                dr_armature_mult_range=(0.90, 1.10),
            )
        if name.startswith(("stage4e", "stage4f", "stage4g")):
            cfg = replace(
                cfg,
                dr_randomize_racket_mount=True,
                dr_racket_pos_offset_m=0.003,
                dr_racket_rot_offset_rad=float(np.deg2rad(1.0)),
                dr_racket_radius_offset_m=0.002,
            )
        if name.startswith("stage4g"):
            cfg = replace(
                cfg,
                dr_ball_friction_range=(0.08, 0.45),
                dr_racket_friction_range=(0.18, 0.75),
                dr_ball_solref_time_range=(0.0015, 0.010),
                dr_ball_solref_damping_range=(0.55, 1.10),
            )
        gate_kwargs = _strict_gate_overrides().get(name, {}) if gate_preset == "v7_strict" else {}
        patched_stages.append(replace(stage, cfg=cfg, notes="", **gate_kwargs))
    stages = patched_stages

    if stage_steps_override is not None:
        stages = [replace(stage, total_steps=int(stage_steps_override)) for stage in stages]
    return stages


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run all MJX-compatible curriculum stages in one training process.")
    p.add_argument("--xml", type=Path, default=here / "moz1_pd.xml")
    p.add_argument("--save-dir", type=Path, default=here.parents[1] / "outputs" / "rl_sim" / "logs_mjx_curriculum")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="Rollout steps per env before each PPO update. 64 is fast; 128-256 is usually better for juggling credit assignment.",
    )
    p.add_argument("--stage-steps", type=int, default=None, help="Override total steps for every stage, useful for smoke tests.")
    p.add_argument("--max-stages", type=int, default=0, help="Run only the first N stages. 0 means all stages.")
    p.add_argument(
        "--curriculum-gate-preset",
        choices=["v7_strict", "legacy"],
        default="v7_strict",
        help="v7_strict uses data-driven, higher/stabler convergence gates; legacy keeps the old easier gates.",
    )
    p.add_argument("--resume-from", type=Path, default=None, help="MJX curriculum .pkl checkpoint to continue from.")
    p.add_argument(
        "--resume-start-stage",
        type=str,
        default="auto",
        help=(
            "Stage to run after loading --resume-from. Use a 1-based index, a stage name, or auto. "
            "auto starts at the next stage for files named NN_stage_name.pkl."
        ),
    )
    p.add_argument(
        "--advance-mode",
        choices=["converged", "fixed"],
        default="converged",
        help="converged gates each stage on recent performance; fixed advances after stage-steps/total_steps.",
    )
    p.add_argument("--convergence-window", type=int, default=20, help="Number of recent updates used for stage convergence.")
    p.add_argument("--convergence-min-episodes", type=int, default=32, help="Ignore updates with fewer completed episodes.")
    p.add_argument("--min-stage-updates", type=int, default=30, help="Minimum updates before a stage can be marked converged.")
    p.add_argument(
        "--allow-unconverged-advance",
        action="store_true",
        help="With --advance-mode converged, continue to the next stage when --max-stage-updates is exhausted.",
    )
    p.add_argument(
        "--max-stage-updates",
        type=int,
        default=0,
        help="Safety cap per stage in converged mode. 0 means train the stage until convergence.",
    )
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--save-every-updates", type=int, default=10)
    p.add_argument(
        "--advance-validation-mode",
        choices=["off", "warn", "block"],
        default="block",
        help=(
            "Run a next-stage validation probe when a stage converges. "
            "block prevents advancing if the next-stage probe collapses; warn only logs."
        ),
    )
    p.add_argument(
        "--no-advance-validation",
        dest="advance_validation_mode",
        action="store_const",
        const="off",
        help="Shortcut for --advance-validation-mode off.",
    )
    p.add_argument("--advance-eval-n-envs", type=int, default=128, help="Parallel envs for next-stage validation probes.")
    p.add_argument(
        "--advance-eval-retry-updates",
        type=int,
        default=10,
        help="After a blocking validation failure, wait this many updates before probing again.",
    )
    p.add_argument(
        "--advance-eval-steps",
        type=int,
        default=0,
        help="Steps for validation probes. 0 uses one full episode horizon of the probe stage.",
    )
    p.add_argument("--advance-eval-min-episodes", type=int, default=32)
    p.add_argument("--advance-eval-hit-ratio", type=float, default=0.25)
    p.add_argument("--advance-eval-min-hits", type=float, default=1.5)
    p.add_argument("--advance-eval-len-ratio", type=float, default=0.25)
    p.add_argument("--advance-eval-min-len-frac", type=float, default=0.10)
    p.add_argument("--advance-eval-min-return", type=float, default=-2.0)
    p.add_argument("--advance-eval-camera-margin", type=float, default=0.10)
    p.add_argument("--advance-eval-camera-reward-margin", type=float, default=0.02)
    p.add_argument(
        "--advance-eval-stochastic",
        dest="advance_eval_deterministic",
        action="store_false",
        help="Use stochastic actions for validation probes. By default probes use deterministic policy means.",
    )
    p.set_defaults(advance_eval_deterministic=True)
    p.add_argument(
        "--no-safe-stop",
        dest="safe_stop",
        action="store_false",
        help="Disable automatic safety stops for non-finite or obviously exploded training metrics.",
    )
    p.set_defaults(safe_stop=True)
    p.add_argument(
        "--max-abs-mean-return",
        type=float,
        default=1e6,
        help="Stop safely if |mean_return| exceeds this value. Use <=0 to disable this guard.",
    )
    p.add_argument(
        "--max-loss",
        type=float,
        default=1e8,
        help="Stop safely if loss or value_loss exceeds this value. Use <=0 to disable this guard.",
    )
    p.add_argument(
        "--max-grad-norm-alert",
        type=float,
        default=1e6,
        help="Stop safely if the unclipped gradient norm exceeds this value. Use <=0 to disable this guard.",
    )
    p.add_argument(
        "--max-abs-reward-metric",
        type=float,
        default=1e4,
        help="Stop safely if any per-step reward/* metric exceeds this absolute value. Use <=0 to disable.",
    )
    p.add_argument(
        "--gpu-max-temp-c",
        type=float,
        default=0.0,
        help="Optional GPU temperature safety stop in Celsius using nvidia-smi. 0 disables this guard.",
    )
    p.add_argument("--gpu-check-every-updates", type=int, default=5)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="pingpong-mjx")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default="mjx-curriculum")
    p.add_argument("--wandb-tags", nargs="*", default=["curriculum"])
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument(
        "--sb3-parity",
        action="store_true",
        help="Override rollout/batch/network defaults to the CPU SB3 PPO reference: 8 envs, 2048 steps, batch 256, 10 epochs, hidden 64.",
    )
    args = p.parse_args()
    if args.sb3_parity:
        args.n_envs = 8
        args.n_steps = 2048
        args.minibatch_size = 256
        args.update_epochs = 10
        args.hidden_dim = 64
    return args


def mean_rollout_metrics(transitions) -> dict[str, float]:
    metrics = {}
    host_metrics = jax.device_get(transitions.metrics)
    for key, value in host_metrics.items():
        arr = np.asarray(value)
        if arr.dtype.kind in "fbiu":
            metrics[key] = float(np.mean(arr))
    return metrics


def _finite_float(row: dict[str, object], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _recent_mean(recent: list[dict[str, object]], key: str) -> float:
    values = []
    for row in recent:
        value = _finite_float(row, key)
        if value is not None:
            values.append(value)
    if not values:
        return float("nan")
    return float(np.mean(values))


def convergence_status(
    history: list[dict[str, object]],
    stage: CurriculumStage,
    env: MjxJuggleEnv,
    args: argparse.Namespace,
    stage_update: int,
) -> dict[str, float]:
    eligible = []
    for row in history:
        if int(row.get("episodes", 0)) < int(args.convergence_min_episodes):
            continue
        mean_hits = _finite_float(row, "mean_hits")
        mean_len = _finite_float(row, "mean_len")
        if mean_hits is None or mean_len is None:
            continue
        eligible.append(row)

    window = max(1, int(args.convergence_window))
    recent = eligible[-window:]
    if recent:
        recent_mean_hits = _recent_mean(recent, "mean_hits")
        recent_mean_len = _recent_mean(recent, "mean_len")
        recent_mean_return = _recent_mean(recent, "mean_return")
        recent_len_frac = recent_mean_len / max(1, int(env.max_steps))
    else:
        recent_mean_hits = float("nan")
        recent_mean_len = float("nan")
        recent_mean_return = float("nan")
        recent_len_frac = float("nan")
    recent_camera_visible = _recent_mean(recent, "camera_visible")
    recent_camera_reward_dense = _recent_mean(recent, "reward/camera_reward_dense")

    required_updates = max(int(stage.min_updates), int(args.min_stage_updates))
    enough_updates = stage_update >= required_updates
    enough_window = len(recent) >= window
    hit_ok = recent_mean_hits >= float(stage.target_mean_hits) if np.isfinite(recent_mean_hits) else False
    len_ok = recent_len_frac >= float(stage.target_mean_len_frac) if np.isfinite(recent_len_frac) else False
    if stage.min_recent_mean_return is None:
        return_ok = True
    else:
        return_ok = bool(np.isfinite(recent_mean_return) and recent_mean_return >= float(stage.min_recent_mean_return))
    if stage.target_camera_visible is None:
        camera_visible_ok = True
    else:
        camera_visible_ok = bool(np.isfinite(recent_camera_visible) and recent_camera_visible >= float(stage.target_camera_visible))
    if stage.min_recent_camera_reward_dense is None:
        camera_reward_ok = True
    else:
        camera_reward_ok = bool(
            np.isfinite(recent_camera_reward_dense)
            and recent_camera_reward_dense >= float(stage.min_recent_camera_reward_dense)
        )
    converged = bool(
        args.advance_mode == "converged"
        and enough_updates
        and enough_window
        and hit_ok
        and len_ok
        and return_ok
        and camera_visible_ok
        and camera_reward_ok
    )
    return {
        "convergence/stage_converged": float(converged),
        "convergence/recent_updates": float(len(recent)),
        "convergence/recent_mean_hits": recent_mean_hits,
        "convergence/recent_mean_len": recent_mean_len,
        "convergence/recent_mean_len_frac": recent_len_frac,
        "convergence/recent_mean_return": recent_mean_return,
        "convergence/recent_camera_visible": recent_camera_visible,
        "convergence/recent_camera_reward_dense": recent_camera_reward_dense,
        "convergence/target_mean_hits": float(stage.target_mean_hits),
        "convergence/target_mean_len_frac": float(stage.target_mean_len_frac),
        "convergence/min_recent_mean_return": (
            float(stage.min_recent_mean_return) if stage.min_recent_mean_return is not None else 0.0
        ),
        "convergence/target_camera_visible": (
            float(stage.target_camera_visible) if stage.target_camera_visible is not None else 0.0
        ),
        "convergence/min_recent_camera_reward_dense": (
            float(stage.min_recent_camera_reward_dense) if stage.min_recent_camera_reward_dense is not None else 0.0
        ),
        "convergence/return_ok": float(return_ok),
        "convergence/camera_visible_ok": float(camera_visible_ok),
        "convergence/camera_reward_ok": float(camera_reward_ok),
        "convergence/min_updates": float(required_updates),
    }


def advance_validation_defaults(
    args: argparse.Namespace,
    stage_idx: int,
    stages: list[CurriculumStage],
) -> dict[str, float]:
    probe_stage = stages[stage_idx] if stage_idx < len(stages) else None
    required = bool(args.advance_validation_mode != "off" and probe_stage is not None)
    if probe_stage is None:
        target_hits = float("nan")
        target_len_frac = float("nan")
        target_return = float("nan")
        target_camera_visible = float("nan")
        target_camera_reward_dense = float("nan")
        probe_stage_index = 0.0
    else:
        thresholds = advance_validation_thresholds(args, probe_stage)
        target_hits = thresholds["target_mean_hits"]
        target_len_frac = thresholds["target_mean_len_frac"]
        target_return = thresholds["min_mean_return"]
        target_camera_visible = thresholds["target_camera_visible"]
        target_camera_reward_dense = thresholds["min_camera_reward_dense"]
        probe_stage_index = float(stage_idx + 1)
    return {
        "advance_eval/required": float(required),
        "advance_eval/ran": 0.0,
        "advance_eval/skipped_cooldown": 0.0,
        "advance_eval/passed": float(not required),
        "advance_eval/blocking": float(required and args.advance_validation_mode == "block"),
        "advance_eval/probe_stage_index": probe_stage_index,
        "advance_eval/target_mean_hits": target_hits,
        "advance_eval/target_mean_len_frac": target_len_frac,
        "advance_eval/min_mean_return": target_return,
        "advance_eval/target_camera_visible": target_camera_visible,
        "advance_eval/min_camera_reward_dense": target_camera_reward_dense,
        "advance_eval/episodes": float("nan"),
        "advance_eval/mean_return": float("nan"),
        "advance_eval/mean_len": float("nan"),
        "advance_eval/mean_len_frac": float("nan"),
        "advance_eval/mean_hits": float("nan"),
        "advance_eval/camera_visible": float("nan"),
        "advance_eval/camera_reward_dense": float("nan"),
        "advance_eval/terminated": float("nan"),
        "advance_eval/truncated": float("nan"),
        "advance_eval/racket_too_high": float("nan"),
        "advance_eval/ball_too_low": float("nan"),
        "advance_eval/hit_ok": 0.0,
        "advance_eval/len_ok": 0.0,
        "advance_eval/return_ok": 0.0,
        "advance_eval/camera_visible_ok": 0.0,
        "advance_eval/camera_reward_ok": 0.0,
        "advance_eval/enough_episodes": 0.0,
    }


def advance_validation_thresholds(args: argparse.Namespace, probe_stage: CurriculumStage) -> dict[str, float]:
    target_hits = max(
        float(args.advance_eval_min_hits),
        float(probe_stage.target_mean_hits) * float(args.advance_eval_hit_ratio),
    )
    target_len_frac = max(
        float(args.advance_eval_min_len_frac),
        float(probe_stage.target_mean_len_frac) * float(args.advance_eval_len_ratio),
    )
    if probe_stage.target_camera_visible is None:
        target_camera_visible = float("nan")
    else:
        target_camera_visible = max(0.0, float(probe_stage.target_camera_visible) - float(args.advance_eval_camera_margin))
    if probe_stage.min_recent_camera_reward_dense is None:
        min_camera_reward = float("nan")
    else:
        min_camera_reward = float(probe_stage.min_recent_camera_reward_dense) - float(args.advance_eval_camera_reward_margin)
    return {
        "target_mean_hits": target_hits,
        "target_mean_len_frac": target_len_frac,
        "min_mean_return": float(args.advance_eval_min_return),
        "target_camera_visible": target_camera_visible,
        "min_camera_reward_dense": min_camera_reward,
    }


def make_eval_rollout(env: MjxJuggleEnv, n_steps: int, deterministic: bool = True):
    def eval_rollout(params, rng: jax.Array):
        reset_keys = jax.random.split(rng, env.n_envs)
        env_state, obs = env.reset(reset_keys)
        running_return = jnp.zeros((env.n_envs,), dtype=jnp.float32)
        running_length = jnp.zeros((env.n_envs,), dtype=jnp.int32)

        def rollout_step(carry, _):
            env_state, obs, rng, running_return, running_length = carry
            rng, action_key, reset_key = jax.random.split(rng, 3)
            mean, _value = policy_value(params, obs)
            if deterministic:
                raw_action = mean
            else:
                log_std = params["log_std"]
                raw_action = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
            env_action = jnp.clip(raw_action, -1.0, 1.0)
            next_env_state, next_obs, reward, done, metrics = env.step(env_state, env_action)

            completed_return = running_return + reward
            completed_length = running_length + 1
            reset_keys = jax.random.split(reset_key, env.n_envs)
            next_env_state, next_obs = env.reset_done(next_env_state, next_obs, done, reset_keys)
            next_running_return = jnp.where(done, 0.0, completed_return)
            next_running_length = jnp.where(done, 0, completed_length)
            output = {
                "done": done,
                "episode_return": completed_return,
                "episode_length": completed_length,
                "hit_count": metrics["hit_count"],
                "metrics": metrics,
            }
            return (next_env_state, next_obs, rng, next_running_return, next_running_length), output

        _carry, outputs = jax.lax.scan(
            rollout_step,
            (env_state, obs, rng, running_return, running_length),
            None,
            length=int(n_steps),
        )
        return outputs

    return jax.jit(eval_rollout)


def summarize_eval_outputs(outputs, env: MjxJuggleEnv) -> dict[str, float]:
    host = jax.device_get(outputs)
    done = np.asarray(host["done"]).astype(bool)
    ep_ret = np.asarray(host["episode_return"])
    ep_len = np.asarray(host["episode_length"])
    hit_count = np.asarray(host["hit_count"])
    done_count = int(done.sum())
    metrics = host["metrics"]

    def metric_mean(key: str) -> float:
        value = metrics.get(key)
        if value is None:
            return float("nan")
        arr = np.asarray(value)
        if arr.dtype.kind not in "fbiu":
            return float("nan")
        return float(np.mean(arr))

    mean_len = float(ep_len[done].mean()) if done_count > 0 else float("nan")
    return {
        "advance_eval/ran": 1.0,
        "advance_eval/episodes": float(done_count),
        "advance_eval/mean_return": float(ep_ret[done].mean()) if done_count > 0 else float("nan"),
        "advance_eval/mean_len": mean_len,
        "advance_eval/mean_len_frac": mean_len / max(1, int(env.max_steps)) if np.isfinite(mean_len) else float("nan"),
        "advance_eval/mean_hits": float(hit_count[done].mean()) if done_count > 0 else float("nan"),
        "advance_eval/camera_visible": metric_mean("camera_visible"),
        "advance_eval/camera_reward_dense": metric_mean("reward/camera_reward_dense"),
        "advance_eval/terminated": metric_mean("terminated"),
        "advance_eval/truncated": metric_mean("truncated"),
        "advance_eval/racket_too_high": metric_mean("done/racket_too_high"),
        "advance_eval/ball_too_low": metric_mean("done/ball_too_low"),
    }


def run_advance_validation(
    args: argparse.Namespace,
    stage_idx: int,
    stages: list[CurriculumStage],
    params,
    rng: jax.Array,
) -> dict[str, float]:
    if args.advance_validation_mode == "off" or stage_idx >= len(stages):
        return {}
    probe_stage = stages[stage_idx]
    n_eval_envs = min(int(args.n_envs), max(1, int(args.advance_eval_n_envs)))
    probe_env = MjxJuggleEnv(args.xml, n_envs=n_eval_envs, cfg=probe_stage.cfg)
    n_eval_steps = int(args.advance_eval_steps) if int(args.advance_eval_steps) > 0 else int(probe_env.max_steps)
    eval_rollout = make_eval_rollout(
        probe_env,
        n_steps=n_eval_steps,
        deterministic=bool(args.advance_eval_deterministic),
    )
    outputs = eval_rollout(params, rng)
    jax.block_until_ready(outputs["done"])
    result = summarize_eval_outputs(outputs, probe_env)
    thresholds = advance_validation_thresholds(args, probe_stage)
    result.update(
        {
            "advance_eval/probe_stage_index": float(stage_idx + 1),
            "advance_eval/target_mean_hits": thresholds["target_mean_hits"],
            "advance_eval/target_mean_len_frac": thresholds["target_mean_len_frac"],
            "advance_eval/min_mean_return": thresholds["min_mean_return"],
            "advance_eval/target_camera_visible": thresholds["target_camera_visible"],
            "advance_eval/min_camera_reward_dense": thresholds["min_camera_reward_dense"],
        }
    )

    episodes = result["advance_eval/episodes"]
    mean_hits = result["advance_eval/mean_hits"]
    mean_len_frac = result["advance_eval/mean_len_frac"]
    mean_return = result["advance_eval/mean_return"]
    camera_visible = result["advance_eval/camera_visible"]
    camera_reward = result["advance_eval/camera_reward_dense"]

    enough_episodes = bool(np.isfinite(episodes) and episodes >= int(args.advance_eval_min_episodes))
    hit_ok = bool(np.isfinite(mean_hits) and mean_hits >= thresholds["target_mean_hits"])
    len_ok = bool(np.isfinite(mean_len_frac) and mean_len_frac >= thresholds["target_mean_len_frac"])
    return_ok = bool(np.isfinite(mean_return) and mean_return >= thresholds["min_mean_return"])
    if probe_stage.target_camera_visible is None:
        camera_visible_ok = True
    else:
        camera_visible_ok = bool(np.isfinite(camera_visible) and camera_visible >= thresholds["target_camera_visible"])
    if probe_stage.min_recent_camera_reward_dense is None:
        camera_reward_ok = True
    else:
        camera_reward_ok = bool(np.isfinite(camera_reward) and camera_reward >= thresholds["min_camera_reward_dense"])
    passed = enough_episodes and hit_ok and len_ok and return_ok and camera_visible_ok and camera_reward_ok
    result.update(
        {
            "advance_eval/enough_episodes": float(enough_episodes),
            "advance_eval/hit_ok": float(hit_ok),
            "advance_eval/len_ok": float(len_ok),
            "advance_eval/return_ok": float(return_ok),
            "advance_eval/camera_visible_ok": float(camera_visible_ok),
            "advance_eval/camera_reward_ok": float(camera_reward_ok),
            "advance_eval/passed": float(passed),
        }
    )
    return result


def stage_update_cap(stage: CurriculumStage, args: argparse.Namespace, batch_steps: int) -> int | None:
    if args.advance_mode == "fixed":
        return max(1, int(stage.total_steps) // max(1, int(batch_steps)))
    if int(args.max_stage_updates) > 0:
        return int(args.max_stage_updates)
    return None


def metric_safety_stop_reason(row: dict[str, object], args: argparse.Namespace) -> str | None:
    if not bool(args.safe_stop):
        return None

    episodes = int(row.get("episodes", 0) or 0)
    for key, value in row.items():
        if isinstance(value, str):
            continue
        if episodes == 0 and key in {"mean_return", "mean_len", "mean_hits"}:
            continue
        if key.startswith("convergence/recent_"):
            continue
        if key.startswith("advance_eval/"):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            return f"non-finite metric: {key}={numeric}"

    max_abs_return = float(args.max_abs_mean_return)
    mean_return = _finite_float(row, "mean_return")
    if max_abs_return > 0.0 and mean_return is not None and abs(mean_return) > max_abs_return:
        return f"|mean_return|={abs(mean_return):.3g} exceeded --max-abs-mean-return={max_abs_return:.3g}"

    max_loss = float(args.max_loss)
    if max_loss > 0.0:
        for key in ("loss", "value_loss"):
            value = _finite_float(row, key)
            if value is not None and abs(value) > max_loss:
                return f"|{key}|={abs(value):.3g} exceeded --max-loss={max_loss:.3g}"

    max_grad_norm = float(args.max_grad_norm_alert)
    grad_norm = _finite_float(row, "grad_norm")
    if max_grad_norm > 0.0 and grad_norm is not None and grad_norm > max_grad_norm:
        return f"grad_norm={grad_norm:.3g} exceeded --max-grad-norm-alert={max_grad_norm:.3g}"

    max_abs_reward = float(args.max_abs_reward_metric)
    if max_abs_reward > 0.0:
        for key, value in row.items():
            if not key.startswith("reward/"):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric) and abs(numeric) > max_abs_reward:
                return f"|{key}|={abs(numeric):.3g} exceeded --max-abs-reward-metric={max_abs_reward:.3g}"
    return None


def gpu_temperature_stop_reason(args: argparse.Namespace, global_update: int) -> str | None:
    limit_c = float(args.gpu_max_temp_c)
    if not bool(args.safe_stop) or limit_c <= 0.0:
        return None
    every = max(1, int(args.gpu_check_every_updates))
    if int(global_update) % every != 0:
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    hottest: tuple[int, float] | None = None
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpu_idx = int(parts[0])
            temp_c = float(parts[1])
        except ValueError:
            continue
        if hottest is None or temp_c > hottest[1]:
            hottest = (gpu_idx, temp_c)
    if hottest is not None and hottest[1] >= limit_c:
        return f"GPU {hottest[0]} temperature {hottest[1]:.0f}C exceeded --gpu-max-temp-c={limit_c:.0f}C"
    return None


def _to_jax_tree(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if hasattr(x, "shape") or np.isscalar(x) else x, tree)


def load_train_state(path: Path) -> tuple[TrainState, dict[str, object]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    params = _to_jax_tree(payload["params"])
    opt_payload = payload.get("opt")
    if opt_payload is None:
        opt = adam_init(params)
    else:
        opt = OptimState(
            m=_to_jax_tree(opt_payload.m),
            v=_to_jax_tree(opt_payload.v),
            t=jnp.asarray(opt_payload.t),
        )
    return TrainState(params=params, opt=opt), payload


def resolve_resume_start_stage(args: argparse.Namespace, stages: list[CurriculumStage]) -> int:
    if args.resume_from is None:
        return 1
    token = str(args.resume_start_stage).strip()
    if token == "auto":
        prefix = args.resume_from.name.split("_", 1)[0]
        if prefix.isdigit():
            return min(int(prefix) + 1, len(stages) + 1)
        return 1
    if token.isdigit():
        return int(token)
    for idx, stage in enumerate(stages, start=1):
        if stage.name == token:
            return idx
    raise SystemExit(f"[mjx_curriculum] unknown --resume-start-stage: {token}")


def finish_wandb_run(wandb_run, args: argparse.Namespace, progress_path: Path) -> None:
    if wandb_run is None:
        return
    import wandb

    last_ckpt = args.save_dir / "mjx_curriculum_last.pkl"
    if last_ckpt.exists():
        wandb.save(str(last_ckpt))
    interrupted_ckpt = args.save_dir / "mjx_curriculum_interrupted.pkl"
    if interrupted_ckpt.exists():
        wandb.save(str(interrupted_ckpt))
    safety_ckpt = args.save_dir / "mjx_curriculum_safety_stop_bad.pkl"
    if safety_ckpt.exists():
        wandb.save(str(safety_ckpt))
    if progress_path.exists():
        wandb.save(str(progress_path))
    wandb_run.finish()


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    stages = build_curriculum(args.stage_steps, args.curriculum_gate_preset)
    if args.max_stages > 0:
        stages = stages[: int(args.max_stages)]
    progress_path = args.save_dir / "curriculum_progress.csv"
    wandb_run = None

    if args.wandb:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise SystemExit("wandb is not installed. Install with: python -m pip install wandb") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "jax_devices": [str(d) for d in jax.devices()],
                "stages": [
                    {
                        "name": stage.name,
                        "total_steps": stage.total_steps,
                        "cfg": stage.cfg.__dict__,
                        "notes": stage.notes,
                        "target_mean_hits": stage.target_mean_hits,
                        "target_mean_len_frac": stage.target_mean_len_frac,
                        "min_updates": stage.min_updates,
                        "min_recent_mean_return": stage.min_recent_mean_return,
                        "target_camera_visible": stage.target_camera_visible,
                        "min_recent_camera_reward_dense": stage.min_recent_camera_reward_dense,
                    }
                    for stage in stages
                ],
            },
        )

    print(f"[mjx_curriculum] JAX devices: {jax.devices()}")
    rng = jax.random.PRNGKey(args.seed)
    rng, params_key = jax.random.split(rng)
    train_state: TrainState | None = None
    resume_payload: dict[str, object] | None = None
    global_step = 0
    global_update = 0
    stop_request = install_stop_handlers()
    start_stage_idx = resolve_resume_start_stage(args, stages)
    if start_stage_idx < 1 or start_stage_idx > len(stages):
        raise SystemExit(f"[mjx_curriculum] --resume-start-stage resolved to {start_stage_idx}, outside 1..{len(stages)}")
    if args.resume_from is not None:
        train_state, resume_payload = load_train_state(args.resume_from)
        global_step = int(resume_payload.get("step", 0))
        print(
            f"[mjx_curriculum] resumed from {args.resume_from} "
            f"at global_step={global_step}; starting stage {start_stage_idx}/{len(stages)}: "
            f"{stages[start_stage_idx - 1].name}"
        )

    for stage_idx, stage in enumerate(stages[start_stage_idx - 1 :], start=start_stage_idx):
        print(f"[mjx_curriculum] stage {stage_idx}/{len(stages)}: {stage.name}")
        if stage.notes:
            print(f"[mjx_curriculum] note: {stage.notes}")
        env = MjxJuggleEnv(args.xml, n_envs=args.n_envs, cfg=stage.cfg)
        print(f"[mjx_curriculum] MJX XML: {env.mjx_xml}")
        print(f"[mjx_curriculum] episode_max_steps={env.max_steps}, dt={env.dt:.4f}s")
        if resume_payload is not None:
            ckpt_obs_dim = int(resume_payload.get("obs_dim", env.obs_dim))
            ckpt_act_dim = int(resume_payload.get("act_dim", env.act_dim))
            if ckpt_obs_dim != int(env.obs_dim) or ckpt_act_dim != int(env.act_dim):
                raise SystemExit(
                    "[mjx_curriculum] resume checkpoint dimensions do not match this env: "
                    f"checkpoint obs/act={ckpt_obs_dim}/{ckpt_act_dim}, env obs/act={env.obs_dim}/{env.act_dim}"
                )

        if train_state is None:
            params = init_params(params_key, env.obs_dim, env.act_dim, args.hidden_dim)
            train_state = TrainState(params=params, opt=adam_init(params))

        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, args.n_envs)
        env_state, obs = jax.jit(env.reset)(reset_keys)
        runner = RunnerState(
            env_state=env_state,
            obs=obs,
            rng=rng,
            running_return=jnp.zeros((args.n_envs,), dtype=jnp.float32),
            running_length=jnp.zeros((args.n_envs,), dtype=jnp.int32),
        )
        collect_rollout, update = make_train_fns(
            env=env,
            n_steps=args.n_steps,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
        )
        batch_steps = int(args.n_envs) * int(args.n_steps)
        stage_updates = stage_update_cap(stage, args, batch_steps)
        stage_history: list[dict[str, object]] = []
        stage_converged = args.advance_mode == "fixed"
        last_advance_eval_update = -10**9

        stage_update = 0
        while True:
            if stop_request.requested:
                reason = stop_request.reason or "stop requested"
                extra = {
                    "stop_reason": reason,
                    "stop_kind": "signal",
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "stage_update": stage_update,
                    "global_update": global_update,
                }
                save_checkpoint(args.save_dir / "mjx_curriculum_last.pkl", train_state, args, env, global_step, extra=extra)
                save_checkpoint(args.save_dir / "mjx_curriculum_interrupted.pkl", train_state, args, env, global_step, extra=extra)
                finish_wandb_run(wandb_run, args, progress_path)
                print(f"[mjx_curriculum] stopped safely: {reason}. checkpoint={args.save_dir / 'mjx_curriculum_interrupted.pkl'}")
                return
            if stage_updates is not None and stage_update >= stage_updates:
                break
            stage_update += 1
            t0 = time.perf_counter()
            runner, transitions = collect_rollout(train_state.params, runner)
            train_state, losses = update(train_state, runner, transitions)
            jax.block_until_ready(losses["loss"])
            elapsed = time.perf_counter() - t0
            global_step += batch_steps
            global_update += 1

            done = np.asarray(jax.device_get(transitions.done)).astype(bool)
            ep_ret = np.asarray(jax.device_get(transitions.episode_return))
            ep_len = np.asarray(jax.device_get(transitions.episode_length))
            hit_count = np.asarray(jax.device_get(transitions.hit_count))
            done_count = int(done.sum())
            row = {
                "stage_index": stage_idx,
                "stage_name": stage.name,
                "stage_update": stage_update,
                "global_update": global_update,
                "global_step": global_step,
                "sps": float(batch_steps / max(elapsed, 1e-9)),
                "episodes": done_count,
                "mean_return": float(ep_ret[done].mean()) if done_count > 0 else float("nan"),
                "mean_len": float(ep_len[done].mean()) if done_count > 0 else float("nan"),
                "mean_hits": float(hit_count[done].mean()) if done_count > 0 else float("nan"),
                **{k: float(v) for k, v in jax.device_get(losses).items()},
                **mean_rollout_metrics(transitions),
            }
            stage_history.append(row)
            status = convergence_status(stage_history, stage, env, args, stage_update)
            row.update(status)
            row.update(advance_validation_defaults(args, stage_idx, stages))
            if bool(row["convergence/stage_converged"]) and bool(row["advance_eval/required"]):
                retry_updates = max(1, int(args.advance_eval_retry_updates))
                if stage_update - last_advance_eval_update < retry_updates:
                    row["convergence/stage_converged"] = 0.0
                    row["advance_eval/skipped_cooldown"] = 1.0
                else:
                    last_advance_eval_update = stage_update
                    eval_key, runner_rng = jax.random.split(runner.rng)
                    runner = runner._replace(rng=runner_rng)
                    eval_result = run_advance_validation(args, stage_idx, stages, train_state.params, eval_key)
                    row.update(eval_result)
                if bool(row["advance_eval/ran"]) and not bool(row["advance_eval/passed"]):
                    row["convergence/stage_converged"] = 0.0
                    if args.advance_validation_mode == "warn":
                        row["convergence/stage_converged"] = 1.0
            append_progress(progress_path, row)
            if wandb_run is not None:
                import wandb

                wandb.log(row, step=global_step)
            update_label = str(stage_update) if stage_updates is None else f"{stage_update}/{stage_updates}"
            camera_label = ""
            if stage.target_camera_visible is not None:
                camera_label = (
                    f" cam={row['convergence/recent_camera_visible']:.2f}/{stage.target_camera_visible:.2f}"
                    f" cam_rew={row['convergence/recent_camera_reward_dense']:.3f}/"
                    f"{stage.min_recent_camera_reward_dense:.3f}"
                )
            print(
                f"[mjx_curriculum] {stage.name} update={update_label} "
                f"global_step={global_step} sps={row['sps']:,.0f} "
                f"episodes={done_count} return={row['mean_return']:.3f} hits={row['mean_hits']:.2f} "
                f"conv_hits={row['convergence/recent_mean_hits']:.2f}/{stage.target_mean_hits:.2f} "
                f"conv_len={row['convergence/recent_mean_len_frac']:.2f}/{stage.target_mean_len_frac:.2f}"
                f"{camera_label}"
            )

            metric_stop = metric_safety_stop_reason(row, args)
            if metric_stop is not None:
                extra = {
                    "stop_reason": metric_stop,
                    "stop_kind": "metric_safety",
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "last_row": row,
                }
                save_checkpoint(
                    args.save_dir / "mjx_curriculum_safety_stop_bad.pkl",
                    train_state,
                    args,
                    env,
                    global_step,
                    extra=extra,
                )
                if wandb_run is not None:
                    import wandb

                    wandb.log({"safe_stop/triggered": 1.0, "safe_stop/metric_guard": 1.0}, step=global_step)
                finish_wandb_run(wandb_run, args, progress_path)
                raise SystemExit(
                    f"[mjx_curriculum] safety stop: {metric_stop}. "
                    f"Bad diagnostic checkpoint saved to {args.save_dir / 'mjx_curriculum_safety_stop_bad.pkl'}; "
                    "mjx_curriculum_last.pkl was left at the previous periodic/stage checkpoint."
                )

            temp_stop = gpu_temperature_stop_reason(args, global_update)
            if temp_stop is not None:
                extra = {
                    "stop_reason": temp_stop,
                    "stop_kind": "gpu_temperature",
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "last_row": row,
                }
                save_checkpoint(args.save_dir / "mjx_curriculum_last.pkl", train_state, args, env, global_step, extra=extra)
                save_checkpoint(args.save_dir / "mjx_curriculum_interrupted.pkl", train_state, args, env, global_step, extra=extra)
                if wandb_run is not None:
                    import wandb

                    wandb.log({"safe_stop/triggered": 1.0, "safe_stop/gpu_temperature": 1.0}, step=global_step)
                finish_wandb_run(wandb_run, args, progress_path)
                print(f"[mjx_curriculum] stopped safely: {temp_stop}. checkpoint={args.save_dir / 'mjx_curriculum_interrupted.pkl'}")
                return

            if stage_update % max(1, int(args.save_every_updates)) == 0:
                save_checkpoint(args.save_dir / "mjx_curriculum_last.pkl", train_state, args, env, global_step)

            if stop_request.requested:
                reason = stop_request.reason or "stop requested"
                extra = {
                    "stop_reason": reason,
                    "stop_kind": "signal",
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "last_row": row,
                }
                save_checkpoint(args.save_dir / "mjx_curriculum_last.pkl", train_state, args, env, global_step, extra=extra)
                save_checkpoint(args.save_dir / "mjx_curriculum_interrupted.pkl", train_state, args, env, global_step, extra=extra)
                finish_wandb_run(wandb_run, args, progress_path)
                print(f"[mjx_curriculum] stopped safely: {reason}. checkpoint={args.save_dir / 'mjx_curriculum_interrupted.pkl'}")
                return

            if bool(row["convergence/stage_converged"]):
                stage_converged = True
                print(
                    f"[mjx_curriculum] stage converged: {stage.name} "
                    f"recent_hits={row['convergence/recent_mean_hits']:.2f}, "
                    f"recent_len_frac={row['convergence/recent_mean_len_frac']:.2f}"
                )
                break
            if bool(row["advance_eval/ran"]) and not bool(row["advance_eval/passed"]):
                print(
                    f"[mjx_curriculum] advance validation failed for next stage "
                    f"{int(row['advance_eval/probe_stage_index'])}/{len(stages)}: "
                    f"hits={row['advance_eval/mean_hits']:.2f}/{row['advance_eval/target_mean_hits']:.2f}, "
                    f"len_frac={row['advance_eval/mean_len_frac']:.2f}/{row['advance_eval/target_mean_len_frac']:.2f}, "
                    f"return={row['advance_eval/mean_return']:.2f}/{row['advance_eval/min_mean_return']:.2f}, "
                    f"cam={row['advance_eval/camera_visible']:.2f}/{row['advance_eval/target_camera_visible']:.2f}. "
                    "Continuing current stage."
                )

        stage_ckpt = args.save_dir / f"{stage_idx:02d}_{stage.name}.pkl"
        save_checkpoint(stage_ckpt, train_state, args, env, global_step)
        save_checkpoint(args.save_dir / "mjx_curriculum_last.pkl", train_state, args, env, global_step)
        if args.advance_mode == "converged" and not stage_converged:
            if not stage_history:
                raise SystemExit(f"[mjx_curriculum] no updates were run for stage: {stage.name}")
            message = (
                f"[mjx_curriculum] stage did not converge before --max-stage-updates: {stage.name}. "
                f"last_recent_hits={stage_history[-1]['convergence/recent_mean_hits']:.2f}, "
                f"target_hits={stage.target_mean_hits:.2f}, "
                f"last_recent_len_frac={stage_history[-1]['convergence/recent_mean_len_frac']:.2f}, "
                f"target_len_frac={stage.target_mean_len_frac:.2f}, "
                f"last_recent_return={stage_history[-1]['convergence/recent_mean_return']:.2f}, "
                f"min_return={stage.min_recent_mean_return}, "
                f"last_camera_visible={stage_history[-1]['convergence/recent_camera_visible']:.2f}, "
                f"target_camera_visible={stage.target_camera_visible}, "
                f"last_camera_reward={stage_history[-1]['convergence/recent_camera_reward_dense']:.3f}, "
                f"min_camera_reward={stage.min_recent_camera_reward_dense}"
            )
            if args.allow_unconverged_advance:
                print(message + " Continuing because --allow-unconverged-advance is set.")
            else:
                finish_wandb_run(wandb_run, args, progress_path)
                raise SystemExit(message)

    finish_wandb_run(wandb_run, args, progress_path)
    print(f"[mjx_curriculum] finished: {args.save_dir}")


if __name__ == "__main__":
    main()
