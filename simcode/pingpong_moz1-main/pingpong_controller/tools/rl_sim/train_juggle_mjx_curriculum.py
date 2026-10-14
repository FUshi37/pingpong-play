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
    policy_mean,
    save_checkpoint,
)


DELAY_ABLATION_PRESETS = (
    "baseline_current",
    "smooth_no_delay_command_state_phase",
    "delay_tau_only",
    "delay_command_state",
    "delay_command_state_phase",
    "delay_command_state_phase_smoothing",
    "delay_command_state_phase_smoothing_antiwindup",
    "real_actuator_replay_hidden50",
    "real_actuator_replay_fit",
    "real_actuator_replay_dr",
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


def _actuator_response_dr_kwargs(level: str = "real") -> dict[str, object]:
    if level == "mild":
        tau_range = (0.030, 0.060)
        gain_range = (0.990, 1.010)
    elif level == "medium":
        tau_range = (0.045, 0.080)
        gain_range = (0.980, 1.020)
    elif level == "real":
        tau_range = (0.060, 0.090)
        gain_range = (0.970, 1.030)
    else:
        raise ValueError(f"unknown actuator response DR level: {level}")
    return {
        "actuator_cmd_filter": True,
        "actuator_cmd_tau": 0.074,
        "actuator_cmd_gain": 1.0,
        "dr_randomize_actuator_cmd_filter": True,
        "dr_actuator_cmd_tau_range": tau_range,
        "dr_actuator_cmd_gain_range": gain_range,
    }


def _with_wide_polish_dr(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    return replace(
        cfg,
        domain_randomization=True,
        dr_randomize_actuator=True,
        dr_randomize_pd=True,
        dr_action_scale_mult_range=(0.85, 1.15),
        dr_damping_mult_range=(0.70, 1.30),
        dr_armature_mult_range=(0.80, 1.20),
        dr_pd_kp_mult_range=(0.85, 1.15),
        dr_pd_kv_mult_range=(0.80, 1.20),
        dr_pd_per_joint=True,
        **_actuator_response_dr_kwargs("real"),
    )


def _with_actuator_safe_early_curriculum(stages: list[CurriculumStage]) -> list[CurriculumStage]:
    """Retune early stages for the real delay + command-filter actuator path."""
    patched: list[CurriculumStage] = []
    for stage in stages:
        cfg = stage.cfg
        target_mean_len_frac = stage.target_mean_len_frac

        common = dict(
            command_tracking_error_penalty_weight=max(float(cfg.command_tracking_error_penalty_weight), 0.05),
            delay_action_jerk_penalty_weight=max(float(cfg.delay_action_jerk_penalty_weight), 3.0e-7),
        )
        limit_common = dict(
            racket_z_limit_termination_penalty_base=max(float(cfg.racket_z_limit_termination_penalty_base), 2.5),
            racket_z_limit_termination_penalty_per_hit=max(float(cfg.racket_z_limit_termination_penalty_per_hit), 1.0),
        )

        if stage.name.startswith("stage1a"):
            cfg = replace(
                cfg,
                action_acc_scale=1.05,
                action_acc_limit=1.0,
                action_penalty_weight=max(float(cfg.action_penalty_weight), 0.0035),
                action_delta_penalty_weight=max(float(cfg.action_delta_penalty_weight), 0.0012),
                arm_vel_limit_penalty_weight=max(float(cfg.arm_vel_limit_penalty_weight), 0.025),
                arm_acc_limit_penalty_weight=max(float(cfg.arm_acc_limit_penalty_weight), 0.04),
                arm_limiter_penalty_weight=max(float(cfg.arm_limiter_penalty_weight), 0.02),
                racket_z_band_down=0.03,
                racket_z_band_up=0.12,
                racket_z_soft_penalty_weight=5.0,
                racket_up_drift_penalty_weight=0.80,
                racket_z_hard_limit_down=0.16,
                racket_z_hard_limit_up=0.32,
                racket_z_limit_termination_penalty_base=4.0,
                racket_z_limit_termination_penalty_per_hit=2.0,
                post_hit_survival_reward_weight=max(float(cfg.post_hit_survival_reward_weight), 1.8),
                hit_reward_base=min(float(cfg.hit_reward_base), 2.5),
                hit_reward_combo=min(float(cfg.hit_reward_combo), 0.80),
                center_flat_hit_reward_weight=min(float(cfg.center_flat_hit_reward_weight), 1.60),
                **common,
            )
            target_mean_len_frac = max(float(target_mean_len_frac), 0.11)
        elif stage.name.startswith("stage1b"):
            cfg = replace(
                cfg,
                action_acc_scale=1.05,
                action_acc_limit=1.0,
                action_penalty_weight=max(float(cfg.action_penalty_weight), 0.0035),
                action_delta_penalty_weight=max(float(cfg.action_delta_penalty_weight), 0.0012),
                arm_vel_limit_penalty_weight=max(float(cfg.arm_vel_limit_penalty_weight), 0.03),
                arm_acc_limit_penalty_weight=max(float(cfg.arm_acc_limit_penalty_weight), 0.05),
                arm_limiter_penalty_weight=max(float(cfg.arm_limiter_penalty_weight), 0.025),
                racket_z_band_down=0.03,
                racket_z_band_up=0.12,
                racket_z_soft_penalty_weight=5.5,
                racket_up_drift_penalty_weight=0.90,
                racket_z_hard_limit_down=0.15,
                racket_z_hard_limit_up=0.30,
                racket_z_limit_termination_penalty_base=3.5,
                racket_z_limit_termination_penalty_per_hit=1.5,
                post_hit_survival_reward_weight=max(float(cfg.post_hit_survival_reward_weight), 1.8),
                hit_reward_combo=min(float(cfg.hit_reward_combo), 0.90),
                **common,
            )
            target_mean_len_frac = max(float(target_mean_len_frac), 0.12)
        elif stage.name.startswith(("stage1c", "stage1d")):
            cfg = replace(
                cfg,
                action_acc_scale=1.0,
                action_acc_limit=1.0,
                action_penalty_weight=max(float(cfg.action_penalty_weight), 0.0035),
                action_delta_penalty_weight=max(float(cfg.action_delta_penalty_weight), 0.0012),
                arm_vel_limit_penalty_weight=max(float(cfg.arm_vel_limit_penalty_weight), 0.04),
                arm_acc_limit_penalty_weight=max(float(cfg.arm_acc_limit_penalty_weight), 0.05),
                arm_limiter_penalty_weight=max(float(cfg.arm_limiter_penalty_weight), 0.03),
                racket_z_band_down=0.03,
                racket_z_band_up=0.12,
                racket_z_soft_penalty_weight=5.5,
                racket_up_drift_penalty_weight=0.90,
                racket_z_hard_limit_down=0.14,
                racket_z_hard_limit_up=0.28,
                racket_z_limit_termination_penalty_base=3.0,
                racket_z_limit_termination_penalty_per_hit=1.2,
                post_hit_survival_reward_weight=max(float(cfg.post_hit_survival_reward_weight), 1.8),
                **common,
            )
            target_mean_len_frac = max(float(target_mean_len_frac), 0.16)
        elif stage.name.startswith(("stage1e", "stage1f", "stage2")):
            cfg = replace(
                cfg,
                action_acc_limit=0.95,
                racket_z_band_down=0.02,
                racket_z_band_up=0.12,
                racket_z_soft_penalty_weight=max(float(cfg.racket_z_soft_penalty_weight), 5.0),
                racket_up_drift_penalty_weight=max(float(cfg.racket_up_drift_penalty_weight), 0.8),
                racket_z_hard_limit_down=0.13,
                racket_z_hard_limit_up=0.26,
                **common,
                **limit_common,
            )
        else:
            cfg = replace(cfg, **common, **limit_common)

        note = "Actuator-safe early curriculum profile."
        notes = f"{stage.notes} {note}".strip() if stage.notes else note
        patched.append(replace(stage, cfg=cfg, notes=notes, target_mean_len_frac=target_mean_len_frac))
    return patched


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


def _sim2real_real_stages(
    stage4g_cfg: MjxJuggleConfig,
    *,
    nominal_pos_bias_base: tuple[float, float, float] | None = None,
    nominal_vel_bias_base: tuple[float, float, float] | None = None,
) -> list[CurriculumStage]:
    """Continuation stages for the first real-robot observations.

    These stages intentionally start after stage4g and ramp the real-world
    mismatch in pieces: camera cadence/dropout, large command latency,
    actuator target lag, then residual hand-eye/frame calibration error.
    """

    pos_bias = nominal_pos_bias_base or (0.0, 0.0, 0.0)
    vel_bias = nominal_vel_bias_base or (0.0, 0.0, 0.0)
    camera_real = dict(
        ball_obs_rate_hz=60.0,
        ball_obs_fractional_rate=True,
        ball_obs_age_tracks_stale=True,
        ball_obs_dropout_on_refresh_only=True,
        ball_obs_require_camera_visible=True,
        ball_obs_pos_noise_std=0.006,
        ball_obs_vel_noise_std=0.08,
        ball_obs_noise_warmup_ratio=0.0,
        ball_obs_noise_ramp_ratio=0.05,
        ball_obs_nominal_pos_bias_base=tuple(float(v) for v in pos_bias),
        ball_obs_nominal_vel_bias_base=tuple(float(v) for v in vel_bias),
        domain_randomization=True,
        dr_randomize_latency=True,
    )
    dropout_mild = dict(
        ball_obs_dropout_prob=0.02,
        ball_obs_dropout_max_steps=6,
        ball_obs_dropout_burst_prob=0.004,
        ball_obs_dropout_burst_max_steps=24,
    )
    dropout_real = dict(
        ball_obs_dropout_prob=0.04,
        ball_obs_dropout_max_steps=10,
        ball_obs_dropout_burst_prob=0.010,
        ball_obs_dropout_burst_max_steps=48,
    )
    actuator_lag = _actuator_response_dr_kwargs("real")
    obs_frame_dr = dict(
        dr_randomize_ball_obs_frame=True,
        dr_ball_obs_pos_bias_base_m=(0.030, 0.030, 0.040),
        dr_ball_obs_rot_bias_deg=(2.0, 2.0, 3.0),
        dr_ball_obs_vel_bias_base_m_s=(0.05, 0.05, 0.08),
        dr_ball_obs_scale_range=(0.97, 1.03),
    )

    return [
        CurriculumStage(
            "stage5a_real_camera_60hz_age",
            1_500_000,
            replace(
                stage4g_cfg,
                **camera_real,
                dr_obs_latency_steps_range=(1, 6),
                dr_action_latency_steps_range=(0, 4),
            ),
            "60Hz camera cadence with stale-observation age, no heavy dropout yet.",
            target_mean_hits=4.0,
            target_mean_len_frac=0.25,
            min_updates=20,
            min_recent_mean_return=2.0,
            target_camera_visible=0.82,
            min_recent_camera_reward_dense=-0.04,
        ),
        CurriculumStage(
            "stage5b_real_camera_fov_dropout",
            2_000_000,
            replace(
                stage4g_cfg,
                **camera_real,
                **dropout_mild,
                dr_obs_latency_steps_range=(2, 8),
                dr_action_latency_steps_range=(0, 6),
            ),
            "Camera visibility gates the ball observation; detector dropouts hold the last valid ball state.",
            target_mean_hits=3.8,
            target_mean_len_frac=0.24,
            min_updates=25,
            min_recent_mean_return=1.5,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.05,
        ),
        CurriculumStage(
            "stage5c_real_action_latency_ramp",
            2_500_000,
            replace(
                stage4g_cfg,
                **camera_real,
                **dropout_real,
                dr_obs_latency_steps_range=(3, 10),
                dr_action_latency_steps_range=(8, 18),
            ),
            "Ramp toward real command delay before exposing the full 120-150ms range.",
            target_mean_hits=3.2,
            target_mean_len_frac=0.22,
            min_updates=30,
            min_recent_mean_return=0.5,
            target_camera_visible=0.78,
            min_recent_camera_reward_dense=-0.06,
        ),
        CurriculumStage(
            "stage5d_real_action_latency_120_170ms",
            3_000_000,
            replace(
                stage4g_cfg,
                **camera_real,
                **dropout_real,
                dr_obs_latency_steps_range=(3, 12),
                dr_action_latency_steps_range=(20, 34),
            ),
            "Real-scale 200Hz command delay: 20-34 control steps is about 100-170ms.",
            target_mean_hits=2.8,
            target_mean_len_frac=0.20,
            min_updates=35,
            min_recent_mean_return=0.0,
            target_camera_visible=0.76,
            min_recent_camera_reward_dense=-0.07,
        ),
        CurriculumStage(
            "stage5e_real_actuator_tracking_lag",
            3_000_000,
            replace(
                stage4g_cfg,
                **camera_real,
                **dropout_real,
                **actuator_lag,
                dr_obs_latency_steps_range=(3, 12),
                dr_action_latency_steps_range=(20, 34),
            ),
            "Adds joint target low-pass tracking and gain error on top of the real-scale command delay.",
            target_mean_hits=2.6,
            target_mean_len_frac=0.18,
            min_updates=40,
            min_recent_mean_return=-0.5,
            target_camera_visible=0.74,
            min_recent_camera_reward_dense=-0.08,
        ),
        CurriculumStage(
            "stage5f_real_calibration_residual_dr",
            4_000_000,
            replace(
                stage4g_cfg,
                **camera_real,
                **dropout_real,
                **actuator_lag,
                **obs_frame_dr,
                dr_obs_latency_steps_range=(3, 12),
                dr_action_latency_steps_range=(20, 34),
            ),
            "Residual hand-eye/base-vs-chest frame error after gross coordinate alignment has been fixed.",
            target_mean_hits=2.5,
            target_mean_len_frac=0.18,
            min_updates=45,
            min_recent_mean_return=-1.0,
            target_camera_visible=0.72,
            min_recent_camera_reward_dense=-0.09,
        ),
    ]


def _sim2real_kf_stages(
    stage4g_cfg: MjxJuggleConfig,
    *,
    nominal_pos_bias_base: tuple[float, float, float] | None = None,
    nominal_vel_bias_base: tuple[float, float, float] | None = None,
) -> list[CurriculumStage]:
    """Continuation stages for a real pipeline with KF prediction at control rate.

    The camera detector may still be 60Hz, but the policy sees the estimator
    output at 200Hz.  Therefore raw camera cadence and FOV dropout are not
    mandatory training stages; they are better kept as held-out stress tests.
    """

    pos_bias = nominal_pos_bias_base or (0.0, 0.0, 0.0)
    vel_bias = nominal_vel_bias_base or (0.0, 0.0, 0.0)
    kf_obs = dict(
        ball_obs_rate_hz=200.0,
        ball_obs_fractional_rate=False,
        ball_obs_age_tracks_stale=False,
        ball_obs_dropout_on_refresh_only=False,
        ball_obs_require_camera_visible=False,
        ball_obs_dropout_prob=0.0,
        ball_obs_dropout_max_steps=1,
        ball_obs_dropout_burst_prob=0.0,
        ball_obs_dropout_burst_max_steps=1,
        ball_obs_pos_noise_std=0.006,
        ball_obs_vel_noise_std=0.08,
        ball_obs_noise_warmup_ratio=0.0,
        ball_obs_noise_ramp_ratio=0.05,
        ball_obs_nominal_pos_bias_base=tuple(float(v) for v in pos_bias),
        ball_obs_nominal_vel_bias_base=tuple(float(v) for v in vel_bias),
        domain_randomization=True,
        dr_randomize_latency=True,
    )
    actuator_lag = _actuator_response_dr_kwargs("real")
    obs_frame_dr = dict(
        dr_randomize_ball_obs_frame=True,
        dr_ball_obs_pos_bias_base_m=(0.030, 0.030, 0.040),
        dr_ball_obs_rot_bias_deg=(2.0, 2.0, 3.0),
        dr_ball_obs_vel_bias_base_m_s=(0.05, 0.05, 0.08),
        dr_ball_obs_scale_range=(0.97, 1.03),
    )

    return [
        CurriculumStage(
            "stage5a_kf_latency_warmup_0_4",
            2_500_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                dr_obs_latency_steps_range=(0, 1),
                dr_action_latency_steps_range=(0, 4),
            ),
            "KF-predicted 200Hz ball observation with only mild command-delay randomization.",
            target_mean_hits=8.0,
            target_mean_len_frac=0.55,
            min_updates=25,
            min_recent_mean_return=12.0,
            target_camera_visible=0.84,
            min_recent_camera_reward_dense=-0.03,
        ),
        CurriculumStage(
            "stage5b_kf_latency_ramp_2_8",
            3_000_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                dr_obs_latency_steps_range=(0, 2),
                dr_action_latency_steps_range=(2, 8),
            ),
            "Ramp command delay into the range where timing adaptation starts to matter.",
            target_mean_hits=6.0,
            target_mean_len_frac=0.42,
            min_updates=30,
            min_recent_mean_return=8.0,
            target_camera_visible=0.82,
            min_recent_camera_reward_dense=-0.04,
        ),
        CurriculumStage(
            "stage5c_kf_latency_ramp_5_12",
            3_000_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                dr_obs_latency_steps_range=(0, 3),
                dr_action_latency_steps_range=(5, 12),
            ),
            "Mid-delay adaptation before exposing the old 8-18 step cliff.",
            target_mean_hits=4.8,
            target_mean_len_frac=0.34,
            min_updates=35,
            min_recent_mean_return=4.0,
            target_camera_visible=0.80,
            min_recent_camera_reward_dense=-0.05,
        ),
        CurriculumStage(
            "stage5d_kf_latency_ramp_8_18",
            3_500_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                dr_obs_latency_steps_range=(0, 3),
                dr_action_latency_steps_range=(8, 18),
            ),
            "The previous first sim-to-real delay is now introduced after three easier adaptation stages.",
            target_mean_hits=3.6,
            target_mean_len_frac=0.26,
            min_updates=40,
            min_recent_mean_return=1.0,
            target_camera_visible=0.78,
            min_recent_camera_reward_dense=-0.06,
        ),
        CurriculumStage(
            "stage5e_kf_latency_ramp_14_26",
            4_000_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                dr_obs_latency_steps_range=(0, 4),
                dr_action_latency_steps_range=(14, 26),
            ),
            "Bridge from medium delay to the measured real-robot command-delay range.",
            target_mean_hits=3.0,
            target_mean_len_frac=0.22,
            min_updates=45,
            min_recent_mean_return=0.0,
            target_camera_visible=0.76,
            min_recent_camera_reward_dense=-0.07,
        ),
        CurriculumStage(
            "stage5f_kf_latency_120_170ms",
            4_500_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                dr_obs_latency_steps_range=(0, 4),
                dr_action_latency_steps_range=(20, 34),
            ),
            "Real-scale 200Hz command delay: 20-34 control steps is about 100-170ms.",
            target_mean_hits=2.6,
            target_mean_len_frac=0.20,
            min_updates=50,
            min_recent_mean_return=-0.5,
            target_camera_visible=0.75,
            min_recent_camera_reward_dense=-0.08,
        ),
        CurriculumStage(
            "stage5g_kf_actuator_tracking_lag",
            4_500_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                **actuator_lag,
                dr_obs_latency_steps_range=(0, 4),
                dr_action_latency_steps_range=(20, 34),
            ),
            "Adds joint target low-pass tracking and gain error after the policy can survive real-scale delay.",
            target_mean_hits=2.4,
            target_mean_len_frac=0.18,
            min_updates=55,
            min_recent_mean_return=-1.0,
            target_camera_visible=0.74,
            min_recent_camera_reward_dense=-0.08,
        ),
        CurriculumStage(
            "stage5h_kf_calibration_residual_dr",
            5_000_000,
            replace(
                stage4g_cfg,
                **kf_obs,
                **actuator_lag,
                **obs_frame_dr,
                dr_obs_latency_steps_range=(0, 4),
                dr_action_latency_steps_range=(20, 34),
            ),
            "Residual hand-eye/base-vs-chest frame error after gross coordinate alignment has been fixed.",
            target_mean_hits=2.2,
            target_mean_len_frac=0.16,
            min_updates=60,
            min_recent_mean_return=-1.5,
            target_camera_visible=0.72,
            min_recent_camera_reward_dense=-0.09,
        ),
    ]


def _high_latency_obs_kwargs(
    *,
    enabled: bool,
    history_frames: int,
    obs_history_frames: int | None = None,
    action_history_frames: int | None = None,
    prediction_time_clip: float,
) -> dict[str, object]:
    obs_frames = history_frames if obs_history_frames is None else obs_history_frames
    action_frames = history_frames if action_history_frames is None else action_history_frames
    return {
        "high_latency_obs": bool(enabled),
        "high_latency_history_frames": int(history_frames),
        "high_latency_obs_history_frames": int(obs_frames),
        "high_latency_action_history_frames": int(action_frames),
        "high_latency_prediction_time_clip": float(prediction_time_clip),
        "high_latency_prediction_include_obs_latency": True,
        "high_latency_prediction_include_ball_age": True,
        "high_latency_prediction_include_actuator_tau": True,
    }


def _delay_conditioned_control_kwargs(preset: str) -> dict[str, object]:
    """Ablation presets for the low-risk command-buffer delay controller."""
    if preset == "baseline_current":
        return {
            "enable_delay_conditioning": False,
            "include_tau_act_norm": False,
            "include_command_state": False,
            "include_phase_features": False,
            "include_active_command_error": False,
            "action_filter_tau_ms": 0.0,
            "action_jerk_limit": 0.0,
            "enable_anti_windup": False,
        }
    if preset not in DELAY_ABLATION_PRESETS:
        raise ValueError(f"unknown delay ablation preset: {preset}")

    kwargs: dict[str, object] = {
        "enable_delay_conditioning": True,
        "delay_min_ms": 0.0,
        "delay_max_ms": 150.0,
        "delay_bin_edges_ms": (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0),
        "delay_jitter_ms": 5.0,
        "delay_sampling_mode": "balanced_bins",
        "include_tau_act_norm": True,
        "include_command_state": False,
        "include_phase_features": False,
        "include_active_command_error": False,
        "action_filter_tau_ms": 0.0,
        "action_jerk_limit": 0.0,
        "action_acc_limit": 1.0,
        "enable_anti_windup": False,
        "anti_windup_error_threshold": 0.35,
        "anti_windup_min_scale": 0.25,
        "command_tracking_error_penalty_weight": 0.0,
        "delay_action_jerk_penalty_weight": 0.0,
        "command_buffer_extra_steps": 4,
        "use_delay_embedding": False,
        "delay_embedding_dim": 0,
        "use_delay_bin_value_heads": False,
        "contact_height_offset": 0.0,
        "max_contact_time": 0.50,
        "lost_ball_timeout_ms": 150.0,
        # Avoid stacking the legacy raw-action latency on top of q_ref delay.
        "dr_randomize_latency": False,
        "dr_action_latency_steps_range": (0, 0),
    }
    if preset == "smooth_no_delay_command_state_phase":
        kwargs.update(
            delay_max_ms=0.0,
            delay_bin_edges_ms=(0.0, 0.0),
            delay_jitter_ms=0.0,
            delay_sampling_mode="uniform",
        )
    if preset == "real_actuator_replay_hidden50":
        kwargs.update(
            delay_min_ms=72.0,
            delay_max_ms=72.0,
            delay_bin_edges_ms=(72.0, 72.0),
            delay_jitter_ms=0.0,
            delay_sampling_mode="uniform",
            include_tau_act_norm=False,
            include_command_state=False,
            include_active_command_error=False,
            include_phase_features=False,
            actuator_cmd_filter=True,
            actuator_cmd_tau=0.074,
            actuator_cmd_gain=1.0,
            dr_randomize_actuator_cmd_filter=False,
            dr_actuator_cmd_tau_range=(0.074, 0.074),
            dr_actuator_cmd_gain_range=(1.0, 1.0),
        )
    if preset == "real_actuator_replay_fit":
        kwargs.update(
            delay_min_ms=72.0,
            delay_max_ms=72.0,
            delay_bin_edges_ms=(72.0, 72.0),
            delay_jitter_ms=0.0,
            delay_sampling_mode="uniform",
            include_command_state=True,
            include_active_command_error=True,
            include_phase_features=True,
            actuator_cmd_filter=True,
            actuator_cmd_tau=0.074,
            actuator_cmd_gain=1.0,
            dr_randomize_actuator_cmd_filter=False,
            dr_actuator_cmd_tau_range=(0.074, 0.074),
            dr_actuator_cmd_gain_range=(1.0, 1.0),
        )
    if preset == "real_actuator_replay_dr":
        kwargs.update(
            delay_min_ms=60.0,
            delay_max_ms=85.0,
            delay_bin_edges_ms=(60.0, 65.0, 70.0, 75.0, 80.0, 85.0),
            delay_jitter_ms=3.0,
            delay_sampling_mode="balanced_bins",
            include_command_state=True,
            include_active_command_error=True,
            include_phase_features=True,
            actuator_cmd_filter=True,
            actuator_cmd_tau=0.074,
            actuator_cmd_gain=1.0,
            dr_randomize_actuator_cmd_filter=True,
            dr_actuator_cmd_tau_range=(0.060, 0.090),
            dr_actuator_cmd_gain_range=(0.97, 1.03),
        )
    if preset in {
        "smooth_no_delay_command_state_phase",
        "delay_command_state",
        "delay_command_state_phase",
        "delay_command_state_phase_smoothing",
        "delay_command_state_phase_smoothing_antiwindup",
    }:
        kwargs.update(include_command_state=True, include_active_command_error=True)
    if preset in {
        "smooth_no_delay_command_state_phase",
        "delay_command_state_phase",
        "delay_command_state_phase_smoothing",
        "delay_command_state_phase_smoothing_antiwindup",
    }:
        kwargs.update(include_phase_features=True)
    if preset in {
        "smooth_no_delay_command_state_phase",
        "delay_command_state_phase_smoothing",
        "delay_command_state_phase_smoothing_antiwindup",
    }:
        kwargs.update(action_filter_tau_ms=15.0, action_jerk_limit=60.0)
    if preset == "delay_command_state_phase_smoothing_antiwindup":
        kwargs.update(enable_anti_windup=True)
    return kwargs


def _delay_bin_edges_for_range(delay_min_ms: float, delay_max_ms: float) -> tuple[float, ...]:
    lo_ms = float(min(delay_min_ms, delay_max_ms))
    hi_ms = float(max(delay_min_ms, delay_max_ms))
    if hi_ms <= lo_ms:
        return (lo_ms, hi_ms)
    step_ms = 25.0
    edges = list(np.arange(lo_ms, hi_ms, step_ms, dtype=np.float32))
    if not edges or abs(float(edges[0]) - lo_ms) > 1e-6:
        edges.insert(0, lo_ms)
    if abs(float(edges[-1]) - hi_ms) > 1e-6:
        edges.append(hi_ms)
    return tuple(float(x) for x in edges)


def _apply_delay_cli_overrides(
    kwargs: dict[str, object],
    *,
    delay_min_ms: float | None,
    delay_max_ms: float | None,
    delay_jitter_ms: float | None,
    delay_sampling_mode: str | None,
) -> dict[str, object]:
    if not bool(kwargs.get("enable_delay_conditioning", False)):
        return kwargs
    patched = dict(kwargs)
    range_changed = False
    if delay_min_ms is not None:
        patched["delay_min_ms"] = float(delay_min_ms)
        range_changed = True
    if delay_max_ms is not None:
        patched["delay_max_ms"] = float(delay_max_ms)
        range_changed = True
    if range_changed:
        patched["delay_bin_edges_ms"] = _delay_bin_edges_for_range(
            float(patched["delay_min_ms"]),
            float(patched["delay_max_ms"]),
        )
    if delay_jitter_ms is not None:
        patched["delay_jitter_ms"] = float(delay_jitter_ms)
    if delay_sampling_mode is not None:
        patched["delay_sampling_mode"] = str(delay_sampling_mode)
    return patched


def _apply_actuator_cli_overrides(
    kwargs: dict[str, object],
    *,
    actuator_cmd_filter: bool | None,
    actuator_cmd_tau: float | None,
    actuator_cmd_gain: float | None,
    actuator_compensation_mode: str | None,
    actuator_lead_compensation: bool | None,
    actuator_lead_beta: float | None,
    actuator_lead_delay_scale: float | None,
    actuator_lead_tau_scale: float | None,
    actuator_lead_max_delta_deg: float | None,
    actuator_inverse_beta: float | None,
    actuator_inverse_delay_scale: float | None,
    actuator_inverse_tau_scale: float | None,
    actuator_inverse_max_delta_deg: float | None,
    actuator_mpc_beta: float | None,
    actuator_mpc_delay_scale: float | None,
    actuator_mpc_tau_scale: float | None,
    actuator_mpc_horizon_steps: int | None,
    actuator_mpc_tracking_weight: float | None,
    actuator_mpc_nominal_weight: float | None,
    actuator_mpc_delta_weight: float | None,
    actuator_mpc_max_delta_deg: float | None,
    dr_randomize_actuator_cmd_filter: bool | None,
    dr_actuator_cmd_tau_range: tuple[float, float] | None,
    dr_actuator_cmd_gain_range: tuple[float, float] | None,
) -> dict[str, object]:
    patched = dict(kwargs)
    if actuator_cmd_filter is not None:
        patched["actuator_cmd_filter"] = bool(actuator_cmd_filter)
        if not bool(actuator_cmd_filter):
            patched["dr_randomize_actuator_cmd_filter"] = False
    if actuator_cmd_tau is not None:
        tau = float(actuator_cmd_tau)
        patched["actuator_cmd_tau"] = tau
        if dr_actuator_cmd_tau_range is None:
            patched["dr_actuator_cmd_tau_range"] = (tau, tau)
    if actuator_cmd_gain is not None:
        gain = float(actuator_cmd_gain)
        patched["actuator_cmd_gain"] = gain
        if dr_actuator_cmd_gain_range is None:
            patched["dr_actuator_cmd_gain_range"] = (gain, gain)
    if actuator_compensation_mode is not None:
        patched["actuator_compensation_mode"] = str(actuator_compensation_mode)
    if actuator_lead_compensation is not None:
        patched["actuator_lead_compensation"] = bool(actuator_lead_compensation)
        if bool(actuator_lead_compensation) and actuator_compensation_mode is None:
            patched["actuator_compensation_mode"] = "lead"
        elif not bool(actuator_lead_compensation) and actuator_compensation_mode is None:
            patched["actuator_compensation_mode"] = "none"
    if actuator_lead_beta is not None:
        patched["actuator_lead_beta"] = float(actuator_lead_beta)
    if actuator_lead_delay_scale is not None:
        patched["actuator_lead_delay_scale"] = float(actuator_lead_delay_scale)
    if actuator_lead_tau_scale is not None:
        patched["actuator_lead_tau_scale"] = float(actuator_lead_tau_scale)
    if actuator_lead_max_delta_deg is not None:
        patched["actuator_lead_max_delta_rad"] = float(np.deg2rad(float(actuator_lead_max_delta_deg)))
    if actuator_inverse_beta is not None:
        patched["actuator_inverse_beta"] = float(actuator_inverse_beta)
    if actuator_inverse_delay_scale is not None:
        patched["actuator_inverse_delay_scale"] = float(actuator_inverse_delay_scale)
    if actuator_inverse_tau_scale is not None:
        patched["actuator_inverse_tau_scale"] = float(actuator_inverse_tau_scale)
    if actuator_inverse_max_delta_deg is not None:
        patched["actuator_inverse_max_delta_rad"] = float(np.deg2rad(float(actuator_inverse_max_delta_deg)))
    if actuator_mpc_beta is not None:
        patched["actuator_mpc_beta"] = float(actuator_mpc_beta)
    if actuator_mpc_delay_scale is not None:
        patched["actuator_mpc_delay_scale"] = float(actuator_mpc_delay_scale)
    if actuator_mpc_tau_scale is not None:
        patched["actuator_mpc_tau_scale"] = float(actuator_mpc_tau_scale)
    if actuator_mpc_horizon_steps is not None:
        patched["actuator_mpc_horizon_steps"] = int(actuator_mpc_horizon_steps)
    if actuator_mpc_tracking_weight is not None:
        patched["actuator_mpc_tracking_weight"] = float(actuator_mpc_tracking_weight)
    if actuator_mpc_nominal_weight is not None:
        patched["actuator_mpc_nominal_weight"] = float(actuator_mpc_nominal_weight)
    if actuator_mpc_delta_weight is not None:
        patched["actuator_mpc_delta_weight"] = float(actuator_mpc_delta_weight)
    if actuator_mpc_max_delta_deg is not None:
        patched["actuator_mpc_max_delta_rad"] = float(np.deg2rad(float(actuator_mpc_max_delta_deg)))
    if dr_actuator_cmd_tau_range is not None:
        tau_lo, tau_hi = dr_actuator_cmd_tau_range
        patched["dr_actuator_cmd_tau_range"] = (float(tau_lo), float(tau_hi))
        if dr_randomize_actuator_cmd_filter is None:
            patched["dr_randomize_actuator_cmd_filter"] = True
    if dr_actuator_cmd_gain_range is not None:
        gain_lo, gain_hi = dr_actuator_cmd_gain_range
        patched["dr_actuator_cmd_gain_range"] = (float(gain_lo), float(gain_hi))
        if dr_randomize_actuator_cmd_filter is None:
            patched["dr_randomize_actuator_cmd_filter"] = True
    if dr_randomize_actuator_cmd_filter is not None:
        patched["dr_randomize_actuator_cmd_filter"] = bool(dr_randomize_actuator_cmd_filter)
    return patched


def _sim2real_kf_high_latency_stages(
    stage4g_cfg: MjxJuggleConfig,
    *,
    nominal_pos_bias_base: tuple[float, float, float] | None = None,
    nominal_vel_bias_base: tuple[float, float, float] | None = None,
    high_latency_obs: bool = False,
    high_latency_history_frames: int = 3,
    high_latency_obs_history_frames: int | None = None,
    high_latency_action_history_frames: int | None = None,
    high_latency_prediction_time_clip: float = 0.30,
) -> list[CurriculumStage]:
    """High-delay continuation for learning 120-150ms juggling in simulation."""

    pos_bias = nominal_pos_bias_base or (0.0, 0.0, 0.0)
    vel_bias = nominal_vel_bias_base or (0.0, 0.0, 0.0)
    kf_obs = dict(
        ball_obs_rate_hz=200.0,
        ball_obs_fractional_rate=False,
        ball_obs_age_tracks_stale=False,
        ball_obs_dropout_on_refresh_only=False,
        ball_obs_require_camera_visible=False,
        ball_obs_dropout_prob=0.0,
        ball_obs_dropout_max_steps=1,
        ball_obs_dropout_burst_prob=0.0,
        ball_obs_dropout_burst_max_steps=1,
        ball_obs_pos_noise_std=0.004,
        ball_obs_vel_noise_std=0.05,
        ball_obs_noise_warmup_ratio=0.0,
        ball_obs_noise_ramp_ratio=0.05,
        ball_obs_nominal_pos_bias_base=tuple(float(v) for v in pos_bias),
        ball_obs_nominal_vel_bias_base=tuple(float(v) for v in vel_bias),
        domain_randomization=True,
        dr_randomize_latency=True,
        **_high_latency_obs_kwargs(
            enabled=high_latency_obs,
            history_frames=high_latency_history_frames,
            obs_history_frames=high_latency_obs_history_frames,
            action_history_frames=high_latency_action_history_frames,
            prediction_time_clip=high_latency_prediction_time_clip,
        ),
    )
    latency_schedule = [
        ("stage5a_hl_latency_0_3", (0, 3), 11.8, 0.78, 19.0, 35),
        ("stage5b_hl_latency_1_4", (1, 4), 11.6, 0.76, 17.0, 40),
        ("stage5c_hl_latency_2_6", (2, 6), 11.0, 0.72, 15.0, 45),
        ("stage5d_hl_latency_3_8", (3, 8), 10.6, 0.68, 13.0, 50),
        ("stage5e_hl_latency_4_10", (4, 10), 10.2, 0.64, 11.0, 55),
        ("stage5f_hl_latency_5_12", (5, 12), 9.8, 0.60, 9.0, 60),
        ("stage5g_hl_latency_6_14", (6, 14), 9.6, 0.58, 8.0, 65),
        ("stage5h_hl_latency_8_16", (8, 16), 9.4, 0.56, 7.0, 70),
        ("stage5i_hl_latency_10_20", (10, 20), 9.2, 0.54, 6.0, 75),
        ("stage5j_hl_latency_12_24", (12, 24), 9.0, 0.52, 5.0, 80),
        ("stage5k_hl_latency_16_30", (16, 30), 8.8, 0.50, 4.0, 85),
        ("stage5l_hl_latency_24_30_120_150ms", (24, 30), 8.5, 0.48, 3.0, 90),
        ("stage5m_hl_latency_24_30_polish", (24, 30), 11.0, 0.70, 10.0, 100),
    ]
    stages: list[CurriculumStage] = []
    for name, action_range, target_hits, target_len, min_return, min_updates in latency_schedule:
        polish = name.endswith("_polish")
        cfg = replace(
            stage4g_cfg,
            **kf_obs,
            **_actuator_response_dr_kwargs("real"),
            dr_obs_latency_steps_range=(0, 2),
            dr_action_latency_steps_range=action_range,
            ball_spawn_xy_jitter=0.020 if polish else stage4g_cfg.ball_spawn_xy_jitter,
            ball_spawn_z_jitter=0.025 if polish else stage4g_cfg.ball_spawn_z_jitter,
            ball_init_vxy_max=0.010 if polish else stage4g_cfg.ball_init_vxy_max,
        )
        if polish:
            cfg = _with_wide_polish_dr(cfg)
        stages.append(
            CurriculumStage(
                name,
                5_000_000 if not polish else 8_000_000,
                cfg,
                (
                    "High-latency policy polish at 120-150ms."
                    if polish
                    else "Fine-grained high-latency ramp with predicted ball observation support."
                ),
                target_mean_hits=target_hits,
                target_mean_len_frac=target_len,
                min_updates=min_updates,
                min_recent_mean_return=min_return,
                target_camera_visible=0.84 if not polish else 0.86,
                min_recent_camera_reward_dense=-0.035 if not polish else -0.025,
            )
        )
    return stages


def build_curriculum(
    stage_steps_override: int | None = None,
    gate_preset: str = "v7_strict",
    curriculum_profile: str = "standard",
    real_ball_obs_nominal_pos_bias_base: tuple[float, float, float] | None = None,
    real_ball_obs_nominal_vel_bias_base: tuple[float, float, float] | None = None,
    high_latency_obs: bool = False,
    high_latency_history_frames: int = 3,
    high_latency_obs_history_frames: int | None = None,
    high_latency_action_history_frames: int | None = None,
    high_latency_prediction_time_clip: float = 0.30,
    delay_ablation_preset: str = "baseline_current",
    delay_min_ms: float | None = None,
    delay_max_ms: float | None = None,
    delay_jitter_ms: float | None = None,
    delay_sampling_mode: str | None = None,
    actuator_cmd_filter: bool | None = None,
    actuator_cmd_tau: float | None = None,
    actuator_cmd_gain: float | None = None,
    actuator_compensation_mode: str | None = None,
    actuator_lead_compensation: bool | None = None,
    actuator_lead_beta: float | None = None,
    actuator_lead_delay_scale: float | None = None,
    actuator_lead_tau_scale: float | None = None,
    actuator_lead_max_delta_deg: float | None = None,
    actuator_inverse_beta: float | None = None,
    actuator_inverse_delay_scale: float | None = None,
    actuator_inverse_tau_scale: float | None = None,
    actuator_inverse_max_delta_deg: float | None = None,
    actuator_mpc_beta: float | None = None,
    actuator_mpc_delay_scale: float | None = None,
    actuator_mpc_tau_scale: float | None = None,
    actuator_mpc_horizon_steps: int | None = None,
    actuator_mpc_tracking_weight: float | None = None,
    actuator_mpc_nominal_weight: float | None = None,
    actuator_mpc_delta_weight: float | None = None,
    actuator_mpc_max_delta_deg: float | None = None,
    dr_randomize_actuator_cmd_filter: bool | None = None,
    dr_actuator_cmd_tau_range: tuple[float, float] | None = None,
    dr_actuator_cmd_gain_range: tuple[float, float] | None = None,
    wide_polish_dr: bool = False,
    asymmetric_critic: bool = False,
    critic_command_history_steps: int = 4,
) -> list[CurriculumStage]:
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
                dr_randomize_pd=True,
                dr_pd_kp_mult_range=(0.97, 1.03),
                dr_pd_kv_mult_range=(0.95, 1.05),
                dr_pd_per_joint=True,
                **_actuator_response_dr_kwargs("mild"),
            )
        elif name.startswith(("stage4d", "stage4e", "stage4f", "stage4g")):
            actuator_response_level = "real" if name.startswith("stage4g") else "medium"
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
                dr_randomize_pd=True,
                dr_pd_kp_mult_range=(0.95, 1.05),
                dr_pd_kv_mult_range=(0.90, 1.10),
                dr_pd_per_joint=True,
                **_actuator_response_dr_kwargs(actuator_response_level),
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

    if curriculum_profile == "sim2real_real":
        stages = stages + _sim2real_real_stages(
            stages[-1].cfg,
            nominal_pos_bias_base=real_ball_obs_nominal_pos_bias_base,
            nominal_vel_bias_base=real_ball_obs_nominal_vel_bias_base,
        )
    elif curriculum_profile == "sim2real_kf":
        stages = stages + _sim2real_kf_stages(
            stages[-1].cfg,
            nominal_pos_bias_base=real_ball_obs_nominal_pos_bias_base,
            nominal_vel_bias_base=real_ball_obs_nominal_vel_bias_base,
        )
    elif curriculum_profile == "sim2real_kf_high_latency":
        stages = stages + _sim2real_kf_high_latency_stages(
            stages[-1].cfg,
            nominal_pos_bias_base=real_ball_obs_nominal_pos_bias_base,
            nominal_vel_bias_base=real_ball_obs_nominal_vel_bias_base,
            high_latency_obs=high_latency_obs,
            high_latency_history_frames=high_latency_history_frames,
            high_latency_obs_history_frames=high_latency_obs_history_frames,
            high_latency_action_history_frames=high_latency_action_history_frames,
            high_latency_prediction_time_clip=high_latency_prediction_time_clip,
        )
    elif curriculum_profile not in ("standard", "actuator_safe"):
        raise ValueError(f"unknown curriculum_profile={curriculum_profile!r}")

    delay_kwargs = _apply_delay_cli_overrides(
        _delay_conditioned_control_kwargs(delay_ablation_preset),
        delay_min_ms=delay_min_ms,
        delay_max_ms=delay_max_ms,
        delay_jitter_ms=delay_jitter_ms,
        delay_sampling_mode=delay_sampling_mode,
    )
    delay_kwargs = _apply_actuator_cli_overrides(
        delay_kwargs,
        actuator_cmd_filter=actuator_cmd_filter,
        actuator_cmd_tau=actuator_cmd_tau,
        actuator_cmd_gain=actuator_cmd_gain,
        actuator_compensation_mode=actuator_compensation_mode,
        actuator_lead_compensation=actuator_lead_compensation,
        actuator_lead_beta=actuator_lead_beta,
        actuator_lead_delay_scale=actuator_lead_delay_scale,
        actuator_lead_tau_scale=actuator_lead_tau_scale,
        actuator_lead_max_delta_deg=actuator_lead_max_delta_deg,
        actuator_inverse_beta=actuator_inverse_beta,
        actuator_inverse_delay_scale=actuator_inverse_delay_scale,
        actuator_inverse_tau_scale=actuator_inverse_tau_scale,
        actuator_inverse_max_delta_deg=actuator_inverse_max_delta_deg,
        actuator_mpc_beta=actuator_mpc_beta,
        actuator_mpc_delay_scale=actuator_mpc_delay_scale,
        actuator_mpc_tau_scale=actuator_mpc_tau_scale,
        actuator_mpc_horizon_steps=actuator_mpc_horizon_steps,
        actuator_mpc_tracking_weight=actuator_mpc_tracking_weight,
        actuator_mpc_nominal_weight=actuator_mpc_nominal_weight,
        actuator_mpc_delta_weight=actuator_mpc_delta_weight,
        actuator_mpc_max_delta_deg=actuator_mpc_max_delta_deg,
        dr_randomize_actuator_cmd_filter=dr_randomize_actuator_cmd_filter,
        dr_actuator_cmd_tau_range=dr_actuator_cmd_tau_range,
        dr_actuator_cmd_gain_range=dr_actuator_cmd_gain_range,
    )
    if delay_ablation_preset != "baseline_current":
        stages = [replace(stage, cfg=replace(stage.cfg, **delay_kwargs)) for stage in stages]

    if curriculum_profile == "actuator_safe":
        stages = _with_actuator_safe_early_curriculum(stages)

    if bool(wide_polish_dr):
        widened = []
        for stage in stages:
            if stage.name.startswith("stage4g") or stage.name.endswith("_polish"):
                widened.append(
                    replace(
                        stage,
                        cfg=_with_wide_polish_dr(stage.cfg),
                        notes=(stage.notes + " " if stage.notes else "") + "Wide actuator/PD DR enabled for polish.",
                    )
                )
            else:
                widened.append(stage)
        stages = widened

    if bool(asymmetric_critic):
        stages = [
            replace(
                stage,
                cfg=replace(
                    stage.cfg,
                    asymmetric_critic=True,
                    critic_command_history_steps=int(critic_command_history_steps),
                ),
            )
            for stage in stages
        ]

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
    p.add_argument(
        "--curriculum-profile",
        choices=["standard", "actuator_safe", "sim2real_real", "sim2real_kf", "sim2real_kf_high_latency"],
        default="standard",
        help=(
            "standard keeps the original 18 stages; actuator_safe retunes early stages for the real delay/filter actuator; "
            "sim2real_real appends raw-detector camera/dropout stages; "
            "sim2real_kf assumes KF-predicted 200Hz ball observations and skips FOV dropout training; "
            "sim2real_kf_high_latency adds finer 120-150ms latency stages."
        ),
    )
    p.add_argument(
        "--delay-ablation-preset",
        choices=DELAY_ABLATION_PRESETS,
        default="baseline_current",
        help=(
            "Enable delay-conditioned command-buffer ablations without changing act_dim. "
            "baseline_current preserves the legacy action path."
        ),
    )
    p.add_argument(
        "--delay-min-ms",
        type=float,
        default=None,
        help="Override the delay-conditioned command-buffer minimum execution delay in milliseconds.",
    )
    p.add_argument(
        "--delay-max-ms",
        type=float,
        default=None,
        help="Override the delay-conditioned command-buffer maximum execution delay in milliseconds.",
    )
    p.add_argument(
        "--delay-jitter-ms",
        type=float,
        default=None,
        help="Override per-step execution delay jitter in milliseconds.",
    )
    p.add_argument(
        "--delay-sampling-mode",
        choices=["uniform", "balanced_bins"],
        default=None,
        help="Override delay sampling mode for delay-conditioned command-buffer training.",
    )
    p.add_argument(
        "--actuator-cmd-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the real-actuator command low-pass filter. Use --no-actuator-cmd-filter to disable it.",
    )
    p.add_argument(
        "--actuator-cmd-tau",
        type=float,
        default=None,
        help="Override the real-actuator command low-pass time constant in seconds.",
    )
    p.add_argument(
        "--actuator-cmd-gain",
        type=float,
        default=None,
        help="Override the real-actuator command gain.",
    )
    p.add_argument(
        "--actuator-compensation-mode",
        choices=["none", "lead", "inverse_smith", "inverse_mpc"],
        default=None,
        help="Output-side actuator compensation before the command delay/filter path.",
    )
    p.add_argument(
        "--actuator-lead-compensation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable conservative output-side lead compensation before the command delay/filter path.",
    )
    p.add_argument(
        "--actuator-lead-beta",
        type=float,
        default=None,
        help="Scale for the lead term qdot*T + 0.5*qdd*T^2. Start conservatively around 0.3-0.5.",
    )
    p.add_argument(
        "--actuator-lead-delay-scale",
        type=float,
        default=None,
        help="Scale applied to the pure command delay when computing lead time.",
    )
    p.add_argument(
        "--actuator-lead-tau-scale",
        type=float,
        default=None,
        help="Scale applied to actuator_cmd_tau when computing lead time.",
    )
    p.add_argument(
        "--actuator-lead-max-delta-deg",
        type=float,
        default=None,
        help="Per-joint absolute cap for lead compensation in degrees.",
    )
    p.add_argument("--actuator-inverse-beta", type=float, default=None)
    p.add_argument(
        "--actuator-inverse-delay-scale",
        type=float,
        default=None,
        help="Scale applied to delay_steps for inverse Smith prediction.",
    )
    p.add_argument(
        "--actuator-inverse-tau-scale",
        type=float,
        default=None,
        help="Scale applied to actuator_cmd_tau inside the inverse model.",
    )
    p.add_argument(
        "--actuator-inverse-max-delta-deg",
        type=float,
        default=None,
        help="Per-joint absolute cap for inverse Smith compensation in degrees.",
    )
    p.add_argument("--actuator-mpc-beta", type=float, default=None)
    p.add_argument("--actuator-mpc-delay-scale", type=float, default=None)
    p.add_argument("--actuator-mpc-tau-scale", type=float, default=None)
    p.add_argument("--actuator-mpc-horizon-steps", type=int, default=None)
    p.add_argument("--actuator-mpc-tracking-weight", type=float, default=None)
    p.add_argument("--actuator-mpc-nominal-weight", type=float, default=None)
    p.add_argument("--actuator-mpc-delta-weight", type=float, default=None)
    p.add_argument(
        "--actuator-mpc-max-delta-deg",
        type=float,
        default=None,
        help="Per-joint absolute cap for regularized inverse-MPC compensation in degrees.",
    )
    p.add_argument(
        "--dr-randomize-actuator-cmd-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override domain randomization of the real-actuator command filter.",
    )
    p.add_argument(
        "--dr-actuator-cmd-tau-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Override the domain-randomized actuator command tau range in seconds.",
    )
    p.add_argument(
        "--dr-actuator-cmd-gain-range",
        nargs=2,
        type=float,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Override the domain-randomized actuator command gain range.",
    )
    p.add_argument(
        "--wide-polish-dr",
        action="store_true",
        help=(
            "Use wider actuator/PD domain randomization on stage4g and named polish stages. "
            "Useful when resuming a converged policy for robustness polish."
        ),
    )
    p.add_argument(
        "--high-latency-obs",
        action="store_true",
        help=(
            "Enable predicted-ball, explicit latency, and observation/action history features. "
            "This increases obs_dim; old 50D checkpoints are migrated with a predicted-ball warm start."
        ),
    )
    p.add_argument("--high-latency-history-frames", type=int, default=3)
    p.add_argument(
        "--high-latency-obs-history-frames",
        type=int,
        default=None,
        help="Override only the observation history frame count; defaults to --high-latency-history-frames.",
    )
    p.add_argument(
        "--high-latency-action-history-frames",
        type=int,
        default=None,
        help="Override only the raw policy action history frame count; defaults to --high-latency-history-frames.",
    )
    p.add_argument("--high-latency-prediction-time-clip", type=float, default=0.30)
    p.add_argument(
        "--allow-obs-dim-migration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow warm-starting/remapping actor/critic first-layer weights when obs_dim changes.",
    )
    p.add_argument(
        "--real-ball-obs-nominal-pos-bias-base",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional nominal bias added to the policy's ball position observation in base coordinates for sim2real_real. "
            "If real detections are chest-frame values fed as base-frame values, use approximately -T_base_chest here."
        ),
    )
    p.add_argument(
        "--real-ball-obs-nominal-vel-bias-base",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "VZ"),
        help="Optional nominal bias added to the policy's ball velocity observation in base coordinates for sim2real_real.",
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
    p.add_argument(
        "--asymmetric-critic",
        action="store_true",
        help="Train the value network with simulator-only privileged observations; actor observations stay unchanged.",
    )
    p.add_argument(
        "--critic-command-history-steps",
        type=int,
        default=4,
        help="Number of recent q_ref command-buffer entries appended only to privileged critic observations.",
    )
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
            mean = policy_mean(params, obs)
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


def _cfg_value(cfg: object | None, name: str, default: object = None) -> object:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _high_latency_input_layout(obs_dim: int, cfg: object | None) -> dict[str, object]:
    base_dim = 50
    act_dim = 7
    prefix_dim = base_dim + 16
    high_latency_obs = bool(_cfg_value(cfg, "high_latency_obs", int(obs_dim) > base_dim))
    if int(obs_dim) <= base_dim or not high_latency_obs:
        return {
            "high_latency": False,
            "prefix_dim": min(base_dim, int(obs_dim)),
            "obs_start": 0,
            "obs_prev": 0,
            "action_start": 0,
            "action_prev": 0,
            "base_dim": base_dim,
            "act_dim": act_dim,
        }

    legacy_frames = max(1, int(_cfg_value(cfg, "high_latency_history_frames", 1) or 1))
    obs_frames_raw = _cfg_value(cfg, "high_latency_obs_history_frames", None)
    action_frames_raw = _cfg_value(cfg, "high_latency_action_history_frames", None)
    obs_frames = legacy_frames if obs_frames_raw is None else max(1, int(obs_frames_raw))
    action_frames = legacy_frames if action_frames_raw is None else max(1, int(action_frames_raw))
    obs_prev = max(0, obs_frames - 1)
    action_prev = max(0, action_frames - 1)
    expected_dim = prefix_dim + obs_prev * base_dim + action_prev * act_dim

    if expected_dim != int(obs_dim):
        extra_dim = int(obs_dim) - prefix_dim
        if extra_dim >= 0 and extra_dim % (base_dim + act_dim) == 0:
            prev_frames = extra_dim // (base_dim + act_dim)
            obs_prev = prev_frames
            action_prev = prev_frames
        else:
            obs_prev = 0
            action_prev = 0

    obs_start = prefix_dim
    action_start = obs_start + obs_prev * base_dim
    return {
        "high_latency": True,
        "prefix_dim": min(prefix_dim, int(obs_dim)),
        "obs_start": obs_start,
        "obs_prev": obs_prev,
        "action_start": action_start,
        "action_prev": action_prev,
        "base_dim": base_dim,
        "act_dim": act_dim,
    }


def _copy_history_weights(
    new_w: jax.Array,
    old_w: jax.Array,
    old_layout: dict[str, object],
    new_layout: dict[str, object],
    *,
    kind: str,
    block_dim: int,
) -> jax.Array:
    old_prev = int(old_layout[f"{kind}_prev"])
    new_prev = int(new_layout[f"{kind}_prev"])
    common = min(old_prev, new_prev)
    if common <= 0:
        return new_w
    old_start = int(old_layout[f"{kind}_start"])
    new_start = int(new_layout[f"{kind}_start"])
    for idx in range(common):
        old_frame = old_prev - common + idx
        new_frame = new_prev - common + idx
        old_slice = slice(old_start + old_frame * block_dim, old_start + (old_frame + 1) * block_dim)
        new_slice = slice(new_start + new_frame * block_dim, new_start + (new_frame + 1) * block_dim)
        new_w = new_w.at[new_slice, :].set(old_w[old_slice, :])
    return new_w


def warm_start_high_latency_l1_weights(
    old_w: jax.Array,
    new_obs_dim: int,
    old_env_cfg: object | None = None,
    new_env_cfg: object | None = None,
) -> jax.Array:
    """Remap policy/value input weights across high-latency observation layouts."""

    old_obs_dim = int(old_w.shape[0])
    new_obs_dim = int(new_obs_dim)
    old_layout = _high_latency_input_layout(old_obs_dim, old_env_cfg)
    new_layout = _high_latency_input_layout(new_obs_dim, new_env_cfg)
    new_w = jnp.zeros((new_obs_dim, old_w.shape[1]), dtype=old_w.dtype)

    prefix_common = min(int(old_layout["prefix_dim"]), int(new_layout["prefix_dim"]), old_obs_dim, new_obs_dim)
    if prefix_common > 0:
        new_w = new_w.at[:prefix_common, :].set(old_w[:prefix_common, :])

    if old_obs_dim == 50 and bool(new_layout["high_latency"]) and new_obs_dim >= 66:
        # High-latency obs layout appends predicted ball pos/vel/rel at rows 50:59.
        # Split original ball-related weights between delayed and predicted rows.
        blend = jnp.asarray(0.5, dtype=old_w.dtype)
        feature_pairs = (
            (slice(20, 23), slice(50, 53)),  # ball position
            (slice(23, 26), slice(53, 56)),  # ball velocity
            (slice(32, 35), slice(56, 59)),  # ball-racket relative position
        )
        for src, dst in feature_pairs:
            src_w = old_w[src, :] * blend
            new_w = new_w.at[src, :].set(src_w)
            new_w = new_w.at[dst, :].set(src_w)

    if bool(old_layout["high_latency"]) and bool(new_layout["high_latency"]):
        base_dim = int(new_layout["base_dim"])
        act_dim = int(new_layout["act_dim"])
        new_w = _copy_history_weights(new_w, old_w, old_layout, new_layout, kind="obs", block_dim=base_dim)
        new_w = _copy_history_weights(new_w, old_w, old_layout, new_layout, kind="action", block_dim=act_dim)
    return new_w


def warm_start_prefix_l1_weights(old_w: jax.Array, new_obs_dim: int) -> jax.Array:
    old_obs_dim = int(old_w.shape[0])
    new_obs_dim = int(new_obs_dim)
    new_w = jnp.zeros((new_obs_dim, old_w.shape[1]), dtype=old_w.dtype)
    common = min(old_obs_dim, new_obs_dim)
    if common > 0:
        new_w = new_w.at[:common, :].set(old_w[:common, :])
    return new_w


def migrate_train_state_obs_dim(
    train_state: TrainState,
    new_obs_dim: int,
    old_env_cfg: object | None = None,
    new_env_cfg: object | None = None,
    new_critic_obs_dim: int | None = None,
) -> TrainState:
    params = dict(train_state.params)
    migrated_params = dict(params)
    for net_name in ("pi", "v"):
        net = dict(migrated_params[net_name])
        l1 = dict(net["l1"])
        old_w = jnp.asarray(l1["w"])
        old_obs_dim = int(old_w.shape[0])
        target_obs_dim = int(new_obs_dim if net_name == "pi" or new_critic_obs_dim is None else new_critic_obs_dim)
        if old_obs_dim == target_obs_dim:
            continue
        if net_name == "v" and target_obs_dim != int(new_obs_dim):
            actor_prefix_w = warm_start_high_latency_l1_weights(old_w, int(new_obs_dim), old_env_cfg, new_env_cfg)
            l1["w"] = warm_start_prefix_l1_weights(actor_prefix_w, target_obs_dim)
        else:
            l1["w"] = warm_start_high_latency_l1_weights(old_w, target_obs_dim, old_env_cfg, new_env_cfg)
        net["l1"] = l1
        migrated_params[net_name] = net
    return TrainState(params=migrated_params, opt=adam_init(migrated_params))


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
    stages = build_curriculum(
        args.stage_steps,
        args.curriculum_gate_preset,
        args.curriculum_profile,
        tuple(args.real_ball_obs_nominal_pos_bias_base) if args.real_ball_obs_nominal_pos_bias_base is not None else None,
        tuple(args.real_ball_obs_nominal_vel_bias_base) if args.real_ball_obs_nominal_vel_bias_base is not None else None,
        bool(args.high_latency_obs),
        int(args.high_latency_history_frames),
        args.high_latency_obs_history_frames,
        args.high_latency_action_history_frames,
        float(args.high_latency_prediction_time_clip),
        str(args.delay_ablation_preset),
        args.delay_min_ms,
        args.delay_max_ms,
        args.delay_jitter_ms,
        args.delay_sampling_mode,
        args.actuator_cmd_filter,
        args.actuator_cmd_tau,
        args.actuator_cmd_gain,
        args.actuator_compensation_mode,
        args.actuator_lead_compensation,
        args.actuator_lead_beta,
        args.actuator_lead_delay_scale,
        args.actuator_lead_tau_scale,
        args.actuator_lead_max_delta_deg,
        args.actuator_inverse_beta,
        args.actuator_inverse_delay_scale,
        args.actuator_inverse_tau_scale,
        args.actuator_inverse_max_delta_deg,
        args.actuator_mpc_beta,
        args.actuator_mpc_delay_scale,
        args.actuator_mpc_tau_scale,
        args.actuator_mpc_horizon_steps,
        args.actuator_mpc_tracking_weight,
        args.actuator_mpc_nominal_weight,
        args.actuator_mpc_delta_weight,
        args.actuator_mpc_max_delta_deg,
        args.dr_randomize_actuator_cmd_filter,
        tuple(args.dr_actuator_cmd_tau_range) if args.dr_actuator_cmd_tau_range is not None else None,
        tuple(args.dr_actuator_cmd_gain_range) if args.dr_actuator_cmd_gain_range is not None else None,
        bool(args.wide_polish_dr),
        bool(args.asymmetric_critic),
        int(args.critic_command_history_steps),
    )
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
    train_state_env_cfg: object | None = None
    global_step = 0
    global_update = 0
    stop_request = install_stop_handlers()
    start_stage_idx = resolve_resume_start_stage(args, stages)
    if start_stage_idx < 1 or start_stage_idx > len(stages):
        raise SystemExit(f"[mjx_curriculum] --resume-start-stage resolved to {start_stage_idx}, outside 1..{len(stages)}")
    if args.resume_from is not None:
        train_state, resume_payload = load_train_state(args.resume_from)
        train_state_env_cfg = resume_payload.get("env_cfg")
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
        print(
            f"[mjx_curriculum] episode_max_steps={env.max_steps}, dt={env.dt:.4f}s, "
            f"obs_dim={env.obs_dim}, high_latency_obs={stage.cfg.high_latency_obs}, "
            f"hl_obs_history={getattr(env, 'high_latency_obs_history_frames', 1)}, "
            f"hl_action_history={getattr(env, 'high_latency_action_history_frames', 1)}, "
            f"delay_preset={args.delay_ablation_preset}, "
            f"delay_conditioning={stage.cfg.enable_delay_conditioning}, "
            f"actuator_cmd_filter={stage.cfg.actuator_cmd_filter}, "
            f"actuator_cmd_tau={stage.cfg.actuator_cmd_tau:.4f}, "
            f"actuator_cmd_gain={stage.cfg.actuator_cmd_gain:.3f}, "
            f"comp_mode={stage.cfg.actuator_compensation_mode}, "
            f"lead_comp={stage.cfg.actuator_lead_compensation}, "
            f"lead_beta={stage.cfg.actuator_lead_beta:.3f}, "
            f"lead_max_deg={np.rad2deg(stage.cfg.actuator_lead_max_delta_rad):.2f}, "
            f"inverse_beta={stage.cfg.actuator_inverse_beta:.3f}, "
            f"inverse_max_deg={np.rad2deg(stage.cfg.actuator_inverse_max_delta_rad):.2f}, "
            f"mpc_beta={stage.cfg.actuator_mpc_beta:.3f}, "
            f"mpc_horizon={stage.cfg.actuator_mpc_horizon_steps}, "
            f"mpc_max_deg={np.rad2deg(stage.cfg.actuator_mpc_max_delta_rad):.2f}, "
            f"delay_extra_dim={getattr(env, 'delay_extra_dim', 0)}, "
            f"asymmetric_critic={getattr(env, 'asymmetric_critic', False)}, "
            f"critic_obs_dim={getattr(env, 'critic_obs_dim', env.obs_dim)}"
        )
        if resume_payload is not None:
            ckpt_obs_dim = int(resume_payload.get("obs_dim", env.obs_dim))
            ckpt_critic_obs_dim = int(resume_payload.get("critic_obs_dim", ckpt_obs_dim))
            ckpt_act_dim = int(resume_payload.get("act_dim", env.act_dim))
            if (
                ckpt_obs_dim != int(env.obs_dim)
                or ckpt_critic_obs_dim != int(getattr(env, "critic_obs_dim", env.obs_dim))
                or ckpt_act_dim != int(env.act_dim)
            ):
                can_migrate_obs = (
                    bool(args.allow_obs_dim_migration)
                    and ckpt_act_dim == int(env.act_dim)
                    and train_state is not None
                )
                if not can_migrate_obs:
                    raise SystemExit(
                        "[mjx_curriculum] resume checkpoint dimensions do not match this env: "
                        f"checkpoint obs/critic/act={ckpt_obs_dim}/{ckpt_critic_obs_dim}/{ckpt_act_dim}, "
                        f"env obs/critic/act={env.obs_dim}/{getattr(env, 'critic_obs_dim', env.obs_dim)}/{env.act_dim}"
                    )
                train_state = migrate_train_state_obs_dim(
                    train_state,
                    int(env.obs_dim),
                    train_state_env_cfg,
                    stage.cfg,
                    int(getattr(env, "critic_obs_dim", env.obs_dim)),
                )
                resume_payload["obs_dim"] = int(env.obs_dim)
                resume_payload["critic_obs_dim"] = int(getattr(env, "critic_obs_dim", env.obs_dim))
                resume_payload["env_cfg"] = stage.cfg.__dict__
                train_state_env_cfg = stage.cfg.__dict__
                print(
                    "[mjx_curriculum] migrated checkpoint input layer: "
                    f"obs_dim {ckpt_obs_dim} -> {env.obs_dim}, "
                    f"critic_obs_dim {ckpt_critic_obs_dim} -> {getattr(env, 'critic_obs_dim', env.obs_dim)}; "
                    "high-latency rows warm-started/remapped; "
                    "optimizer state reinitialized"
                )

        if train_state is None:
            params = init_params(
                params_key,
                env.obs_dim,
                env.act_dim,
                args.hidden_dim,
                int(getattr(env, "critic_obs_dim", env.obs_dim)),
            )
            train_state = TrainState(params=params, opt=adam_init(params))
        else:
            param_obs_dim = int(train_state.params["pi"]["l1"]["w"].shape[0])
            param_critic_obs_dim = int(train_state.params["v"]["l1"]["w"].shape[0])
            param_act_dim = int(train_state.params["pi"]["out"]["b"].shape[0])
            if (
                param_obs_dim != int(env.obs_dim)
                or param_critic_obs_dim != int(getattr(env, "critic_obs_dim", env.obs_dim))
                or param_act_dim != int(env.act_dim)
            ):
                can_migrate_obs = (
                    bool(args.allow_obs_dim_migration)
                    and param_act_dim == int(env.act_dim)
                )
                if not can_migrate_obs:
                    raise SystemExit(
                        "[mjx_curriculum] current policy dimensions do not match this env: "
                        f"policy obs/critic/act={param_obs_dim}/{param_critic_obs_dim}/{param_act_dim}, "
                        f"env obs/critic/act={env.obs_dim}/{getattr(env, 'critic_obs_dim', env.obs_dim)}/{env.act_dim}"
                    )
                train_state = migrate_train_state_obs_dim(
                    train_state,
                    int(env.obs_dim),
                    train_state_env_cfg,
                    stage.cfg,
                    int(getattr(env, "critic_obs_dim", env.obs_dim)),
                )
                print(
                    "[mjx_curriculum] migrated current policy input layer: "
                    f"obs_dim {param_obs_dim} -> {env.obs_dim}, "
                    f"critic_obs_dim {param_critic_obs_dim} -> {getattr(env, 'critic_obs_dim', env.obs_dim)}; "
                    "high-latency rows warm-started/remapped; "
                    "optimizer state reinitialized"
                )
        train_state_env_cfg = stage.cfg.__dict__

        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, args.n_envs)
        env_state, obs = jax.jit(env.reset)(reset_keys)
        critic_obs = env.get_critic_obs(env_state, obs)
        runner = RunnerState(
            env_state=env_state,
            obs=obs,
            critic_obs=critic_obs,
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
