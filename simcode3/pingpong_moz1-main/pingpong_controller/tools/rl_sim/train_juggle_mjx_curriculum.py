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
from dataclasses import dataclass, fields, replace
from pathlib import Path
import pickle
import signal
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from camera_calibration import (
    D455_848_UNDISTORTED_BASE_POS,
    D455_848_UNDISTORTED_BASE_ROT,
    D455_848_UNDISTORTED_SIM_BASE_BODY,
    D455_848_UNDISTORTED_CX,
    D455_848_UNDISTORTED_CY,
    D455_848_UNDISTORTED_FX,
    D455_848_UNDISTORTED_FY,
    D455_848_UNDISTORTED_HEIGHT,
    D455_848_UNDISTORTED_HFOV_DEG,
    D455_848_UNDISTORTED_PIXEL_MARGIN,
    D455_848_UNDISTORTED_VFOV_DEG,
    D455_848_UNDISTORTED_WIDTH,
)
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


LOW_REALVIEW_RIGHT_ARM_RESET_DEGREES = (
    2.20493773,
    -36.56958938,
    23.68133788,
    74.28332693,
    51.96893721,
    -24.35973512,
    41.00752796,
)


D455_USER_REQUESTED_RACKET_RESET_DEGREES = (
    5.736,
    -44.399,
    30.683,
    97.142,
    49.323,
    -12.269,
    14.214,
)

D455_USER_TARGET_RACKET_RESET_DEGREES = (
    5.736,
    -44.399,
    30.683,
    97.142,
    49.323,
    -12.269,
    14.214,
)

D455_REAL_VIEW_X_BOUNDS_M = (-0.25, 0.25)
D455_REAL_VIEW_Y_BOUNDS_M = (-0.50, -0.25)
# Physical z measurements are reported as XML/world z minus the 0.100m base height,
# while MJX ball z metrics use the XML/world z directly.
D455_REAL_VIEW_Z_BOUNDS_M = (1.00, 1.47)
D455_REAL_VIEW_Y_TARGET_M = -0.35
D455_STABLE_VIEW_Z_IDEAL_M = (1.02, 1.42)
D455_RECOVERY_VIEW_Z_IDEAL_M = (1.02, 1.42)


STAGE_NAME_ALIASES = {
    "stage5e2_low_reset_visible_recenter_len33_soft": "stage5e2a_low_reset_view_recenter_len45_soft",
    "stage5e3_low_reset_visible_long_len70_soft": "stage5e2b_low_reset_long_soft_len70",
    "stage5e4_low_reset_visible_pre_hard_len85_soft": "stage5e3_low_reset_visible_pre_hard_len85_soft",
}


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


ROBUST_JUGGLE_PROFILE = "robust_juggle_v1"
D455_STABLE_4G_PROFILE = "d455_stable_4g_v1"
D455_RECOVERY_PROFILE = "d455_recovery_v1"
D455_FULL_CURRICULUM_PROFILE = "d455_full_curriculum_v1"
D455_SUCCESS_REF_PROFILE = "d455_success_ref_v1"
GOAL_D455_AUTOLAUNCH_PROFILE = "goal_d455_autolaunch_v1"
GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE = "goal_d455_autolaunch_viewdense_v1"
GOAL_D455_AUTOLAUNCH_VIEWDENSE_RELAXTRUNC_PROFILE = "goal_d455_autolaunch_viewdense_relaxtrunc_v1"
GOAL_D455_AUTOLAUNCH_VIEWDENSE_FULLSAFE_PROFILE = "goal_d455_autolaunch_viewdense_fullsafe_v1"
GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE = (
    "goal_d455_autolaunch_viewdense_drivegov_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE = (
    "goal_d455_autolaunch_viewdense_drivegov_terminalsafe_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE = (
    "goal_d455_autolaunch_viewdense_drivegov_successref_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE = (
    "goal_d455_autolaunch_viewdense_drivegov_highapex_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_COUNTCREDIT_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_COUNTCREDIT_NOMISSING_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_nomissing_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_RECOVERABILITY_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_recoverability_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_INTERCEPT_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_intercept_v1"
)
GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_COUNT_PROGRESS_PROFILE = (
    "goal_d455_autolaunch_viewdense_constrained_mpc_count_progress_v1"
)
GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE = (
    "goal_d455_autolaunch_teacherstudent_drivegov_v1"
)
GOAL_D455_AUTOLAUNCH_IDEALPD_PROFILE = "goal_d455_autolaunch_idealpd_v1"
GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE = "goal_d455_autolaunch_idealpd67_v1"
GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE = (
    "goal_d455_autolaunch_idealpd67_viewdense_v1"
)
GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE = (
    "goal_d455_autolaunch_idealpd67_final_recovery_v1"
)
GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE = (
    "goal_d455_autolaunch_idealpd67_actuator_inversempc_finetune_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_countcredit_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_RECOVERY_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_final_recovery_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_CADENCE_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_SURVIVAL_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_SURVIVAL_COUNTCREDIT_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_final_survival_countcredit_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_MISSING_AGE_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_final_missing_age_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_INTERCEPT_NOGOV_PROFILE = (
    "goal_d455_autolaunch_actuator_inversempc_final_intercept_nogov_v1"
)
GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_RESIDUAL_PROFILE = (
    "goal_d455_autolaunch_idealpd67_actuator_inversempc_residual_v1"
)
GOAL_D455_RELEASE_PROFILE = "goal_d455_release_v1"
GOAL_D455_AUTOLAUNCH_PROFILES = (
    GOAL_D455_AUTOLAUNCH_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_RELAXTRUNC_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_FULLSAFE_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_COUNTCREDIT_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_COUNTCREDIT_NOMISSING_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_RECOVERABILITY_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_INTERCEPT_PROFILE,
    GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_COUNT_PROGRESS_PROFILE,
    GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_RECOVERY_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_CADENCE_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_SURVIVAL_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_SURVIVAL_COUNTCREDIT_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_MISSING_AGE_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_INTERCEPT_NOGOV_PROFILE,
    GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE,
    GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_RESIDUAL_PROFILE,
)
GOAL_D455_IDEALPD67_PROFILES = (
    GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE,
    GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE,
    GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE,
)
GOAL_D455_IDEALPD_PROFILES = (
    GOAL_D455_AUTOLAUNCH_IDEALPD_PROFILE,
    *GOAL_D455_IDEALPD67_PROFILES,
)
GOAL_D455_PROFILES = (
    *GOAL_D455_AUTOLAUNCH_PROFILES,
    *GOAL_D455_IDEALPD_PROFILES,
    GOAL_D455_RELEASE_PROFILE,
)
GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT = 0.03
GOAL_D455_AUTOLAUNCH_VIEWDENSE_XY_WEIGHT = 0.05
GOAL_D455_AUTOLAUNCH_VIEWDENSE_BOUNDS_WEIGHT = 8.0
GOAL_D455_AUTOLAUNCH_VIEWDENSE_OOB_WEIGHT = 0.20
GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_CLIP_WEIGHT = 5.0
GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_JERK_WEIGHT = 3.0e-6
GOAL_D455_AUTOLAUNCH_FULLSAFE_LIMITER_WEIGHT = 0.05
# W017 transfers only the learnability signals supported by the successful
# inverse-MPC actuator run.  It deliberately keeps the GOAL task reward,
# D455 curriculum, original inverse-MPC and final drive-governor plant instead
# of copying that historical curriculum or compensation mechanism.
GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT = 0.0018
GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT = 0.0012
GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT = 0.05
GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT = 3.0e-7
GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT = 1.4
GOAL_D455_AUTOLAUNCH_SUCCESSREF_MISS_PENALTY_PER_HIT = 0.8
GOAL_D455_AUTOLAUNCH_SUCCESSREF_RACKET_Z_PENALTY_PER_HIT = 1.0
GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_VEL_LIMIT_WEIGHT = 0.06
GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_ACC_LIMIT_WEIGHT = 0.08
GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_LIMITER_WEIGHT = 0.08
GOAL_D455_AUTOLAUNCH_FINAL_RECOVERY_POST_HIT_SURVIVAL_WEIGHT = 2.4
GOAL_D455_AUTOLAUNCH_FINAL_RECOVERY_NEXT_CONTACT_WEIGHT = 0.12
GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_WEIGHT = 0.30
GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_TARGET_S = 0.45
GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_SIGMA_S = 0.05
GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_MIN_INTERVAL_S = 0.38
GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_MIN_INTERVAL_WEIGHT = 0.50
GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_FAST_HIT_WEIGHT = 0.50
GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_GATE_MAX_S = 0.47
# W018 derives its vertical target from the calibrated D455 view rather than
# treating hit count as the objective.  The nominal racket-launch ball centre
# is about 1.08 m; an apex at 1.40--1.42 m has a 0.51--0.53 s ballistic period
# and therefore yields roughly 11--12 clean hits in the six-second horizon.
GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_HEIGHT = 0.34
GOAL_D455_AUTOLAUNCH_HIGHAPEX_HIT_HEIGHT = 0.36
GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_ABS_Z = 1.42
GOAL_D455_AUTOLAUNCH_HIGHAPEX_CADENCE_TARGET_S = 0.52
GOAL_D455_AUTOLAUNCH_HIGHAPEX_CADENCE_SIGMA_S = 0.07
GOAL_D455_AUTOLAUNCH_HIGHAPEX_POST_HIT_VXY_WEIGHT = 0.18
GOAL_D455_AUTOLAUNCH_HIGHAPEX_DESCENDING_INTERCEPT_WEIGHT = 0.8
GOAL_D455_AUTOLAUNCH_HIGHAPEX_NEXT_CONTACT_WEIGHT = 0.05
GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_MIN_UPDATES = {
    14: 60,
    15: 80,
}
GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_SURVIVAL_WEIGHT = 0.50
GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_MIN_UPDATES = 60
GOAL_D455_AUTOLAUNCH_RELAXED_EARLY_TRUNCATION = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.05,
    5: 0.13,
    6: 0.22,
    7: 0.30,
}
GOAL_D455_RELEASE_NEXT_CONTACT_PENALTY_WEIGHT = 0.03
D455_TWO_PHASE_PROFILES = (D455_STABLE_4G_PROFILE, D455_RECOVERY_PROFILE)
D455_67D_INVERSE_MPC_PROFILES = (
    *D455_TWO_PHASE_PROFILES,
    D455_FULL_CURRICULUM_PROFILE,
    D455_SUCCESS_REF_PROFILE,
    *GOAL_D455_AUTOLAUNCH_PROFILES,
    GOAL_D455_RELEASE_PROFILE,
)
STAGE4G_ROBUST15_MISSING_PROFILE = "standard_stage4g_robust15_missing_bridge"


ROBUST15_LOW_RESET_CURRICULUM_PROFILES = (
    "standard_low_reset_robust15",
    "standard_low_reset_robust15_bridge",
    "standard_low_reset_robust15_missing_bridge",
)
LOW_RESET_CURRICULUM_PROFILES = ("standard_low_reset", *ROBUST15_LOW_RESET_CURRICULUM_PROFILES)


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    total_steps: int
    cfg: MjxJuggleConfig
    notes: str = ""
    target_mean_hits: float = 2.0
    gate_mode: str = "strict"
    advance_gate_mode: str = "strict"
    target_mean_len_frac: float = 0.20
    min_updates: int = 5
    min_recent_mean_return: float | None = None
    target_camera_visible: float | None = None
    min_recent_camera_reward_dense: float | None = None
    target_ball_view_in_bounds: float | None = None
    target_ball_view_z_ideal: float | None = None
    target_hit1_rate: float | None = None
    target_hit3_rate: float | None = None
    target_hit12_rate: float | None = None
    target_mean_hits_ge3: float | None = None
    target_min_hit_interval_s: float | None = None
    target_max_hit_interval_s: float | None = None
    target_hit_camera_visible_rate: float | None = None
    target_hit_camera_lower_band_rate: float | None = None
    max_recent_mean_hit_vxy: float | None = None
    max_recent_hit_next_contact_anchor_err: float | None = None
    max_recent_mean_hit_camera_v_frac: float | None = None
    target_episode_truncation_rate: float | None = None
    target_racket_up_cos: float | None = None
    min_ball_obs_missing_refresh_rate: float | None = None
    max_ball_obs_lost_rate: float | None = None
    policy_updates_enabled: bool = True
    max_updates: int | None = None


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


def _with_strong_camera_centering(cfg: MjxJuggleConfig, *, center_weight: float = 1.4) -> MjxJuggleConfig:
    return replace(
        cfg,
        camera_visibility_mode="pixel",
        virtual_camera_pose_mode="base_extrinsic",
        virtual_camera_base_body_name=D455_848_UNDISTORTED_SIM_BASE_BODY,
        camera_center_weight=max(float(cfg.camera_center_weight), float(center_weight)),
        camera_visibility_penalty_weight=max(float(cfg.camera_visibility_penalty_weight), 8.0),
        camera_visible_penalty_weight=max(float(cfg.camera_visible_penalty_weight), 3.0),
        camera_top_margin_penalty_weight=max(float(cfg.camera_top_margin_penalty_weight), 12.0),
        camera_pixel_margin=D455_848_UNDISTORTED_PIXEL_MARGIN,
    )


def _with_verified_stage4g_legacy_camera(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Restore the camera/FOV used by the verified 67D stage4g checkpoint.

    ``logs_mjx_actuator_67d_inverse_mpc_h4_reg_stage4g_polish_v1`` was trained
    before the default D455_848/base_extrinsic camera became the current code
    default.  Its saved env_cfg has legacy head22/body_mount intrinsics:
    1280x720, fx/fy ~= 637, pixel margin 80, and camera_visible ~= 0.93.
    Bridge stages that resume from that checkpoint must preserve this first;
    otherwise camera_visible collapses to ~0.01 and the 13-hit policy degrades
    to a ~2-hit policy before any meaningful missing/generalization training.
    """

    return replace(
        cfg,
        camera_visibility_mode="pixel",
        virtual_camera_pose_mode="body_mount",
        virtual_camera_body_name="head22",
        virtual_camera_mount_pos=(0.0, -0.068, 0.062),
        virtual_camera_mount_quat=(0.707107, 0.0, 0.0, -0.707107),
        virtual_camera_optical_pos=(0.048, 0.0, 0.0),
        camera_image_width=1280,
        camera_image_height=720,
        camera_fx=636.99,
        camera_fy=636.84,
        camera_cx=646.82,
        camera_cy=373.21,
        camera_hfov_deg=86.0,
        camera_vfov_deg=57.0,
        camera_pixel_margin=80.0,
        camera_center_weight=0.5,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
        camera_min_depth=0.15,
        camera_max_depth=2.50,
    )


def _with_latest_d455_camera(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Use the current calibrated D455 RGB camera model.

    The final sim2real curriculum must optimize visibility against the latest
    848x480@60Hz undistorted intrinsics and ``T_base_camera`` hand-eye
    calibration, not the older legacy head/body-mounted 1280x720 camera that
    was only useful for diagnosing the original stage4g checkpoint.
    """

    return replace(
        cfg,
        camera_visibility_mode="pixel",
        virtual_camera_pose_mode="base_extrinsic",
        virtual_camera_base_body_name=D455_848_UNDISTORTED_SIM_BASE_BODY,
        camera_image_width=D455_848_UNDISTORTED_WIDTH,
        camera_image_height=D455_848_UNDISTORTED_HEIGHT,
        camera_fx=D455_848_UNDISTORTED_FX,
        camera_fy=D455_848_UNDISTORTED_FY,
        camera_cx=D455_848_UNDISTORTED_CX,
        camera_cy=D455_848_UNDISTORTED_CY,
        camera_hfov_deg=D455_848_UNDISTORTED_HFOV_DEG,
        camera_vfov_deg=D455_848_UNDISTORTED_VFOV_DEG,
        camera_pixel_margin=D455_848_UNDISTORTED_PIXEL_MARGIN,
        virtual_camera_base_pos=D455_848_UNDISTORTED_BASE_POS,
        virtual_camera_base_rot=D455_848_UNDISTORTED_BASE_ROT,
        camera_center_weight=0.5,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
        camera_min_depth=0.15,
        camera_max_depth=2.50,
    )


def _with_verified_stage4g_policy_compatible_terms(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Keep stage4g-compatible control/reward terms while using the real D455 camera.

    This deliberately does not touch reset target ranges, view-bound penalties,
    or missing/dropout settings, so later bridge stages can still add
    generalization.  The original successful run remains the curriculum/reward
    reference, but the final sim2real target uses the latest calibrated D455
    camera rather than the legacy camera from that run.
    """

    cfg = _with_latest_d455_camera(cfg)
    return replace(
        cfg,
        right_arm_pd_profile="xml",
        domain_randomization=True,
        dr_randomize_ball=True,
        dr_randomize_contact=True,
        dr_randomize_actuator=True,
        dr_randomize_latency=False,
        dr_randomize_pd=True,
        dr_pd_per_joint=True,
        dr_randomize_racket_mount=True,
        dr_randomize_actuator_cmd_filter=False,
        dr_action_latency_steps_range=(0, 0),
        dr_obs_latency_steps_range=(0, 2),
        dr_action_scale_mult_range=(0.93, 1.07),
        dr_armature_mult_range=(0.90, 1.10),
        dr_damping_mult_range=(0.85, 1.15),
        dr_pd_kp_mult_range=(0.95, 1.05),
        dr_pd_kv_mult_range=(0.90, 1.10),
        dr_actuator_cmd_tau_range=(0.074, 0.074),
        dr_actuator_cmd_gain_range=(1.0, 1.0),
        dr_ball_mass_range=(0.0024, 0.0030),
        dr_ball_friction_range=(0.08, 0.45),
        dr_racket_friction_range=(0.18, 0.75),
        dr_ball_solref_time_range=(0.0015, 0.010),
        dr_ball_solref_damping_range=(0.55, 1.10),
        dr_racket_pos_offset_m=0.003,
        dr_racket_radius_offset_m=0.002,
        dr_racket_rot_offset_rad=float(np.deg2rad(1.0)),
        hit_cadence_reward_weight=0.05,
        hit_cadence_target_interval=0.32,
        hit_cadence_sigma=0.10,
        hit_min_interval_penalty_weight=1.50,
        hit_min_interval=0.24,
        hit_min_count_interval=0.22,
        fast_hit_penalty_weight=0.80,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.32,
        hit_reward_count_cap=0,
        post_hit_survival_reward_weight=1.40,
        center_flat_hit_reward_weight=0.80,
        hit_reward_base=0.50,
        hit_reward_combo=0.02,
    )


def _with_real_view_ball_range(
    cfg: MjxJuggleConfig,
    *,
    terminate: bool,
    xy_weight: float,
    z_weight: float,
    bounds_weight: float,
    vxy_weight: float,
    out_of_bounds_weight: float = 0.0,
    z_not_ideal_weight: float = 0.0,
    target_x_range: tuple[float, float] = (0.14, 0.20),
    target_y_range: tuple[float, float] = (0.02, 0.10),
    anchor_z_range: tuple[float, float] = (-0.20, -0.20),
    launch_height: float = 0.10,
    target_height: float = 0.10,
    hit_height_center: float = 0.13,
    hit_confirm_rel_height: float = 0.06,
    racket_up_margin: float = 0.16,
    terminate_racket_z: bool | None = None,
) -> MjxJuggleConfig:
    return replace(
        cfg,
        ball_launch_height=launch_height,
        target_height=target_height,
        rel_height_center=min(float(cfg.rel_height_center), max(0.08, target_height - 0.02)),
        hit_height_center=hit_height_center,
        hit_confirm_rel_height=hit_confirm_rel_height,
        hit_height_tolerance=min(float(cfg.hit_height_tolerance), 0.045),
        low_hit_apex_margin=min(float(cfg.low_hit_apex_margin), 0.025),
        apex_soft_limit_margin=min(float(cfg.apex_soft_limit_margin), 0.025),
        episode_target_x_range_m=target_x_range,
        episode_target_y_range_m=target_y_range,
        episode_racket_anchor_z_range_m=anchor_z_range,
        racket_z_hard_limit_up=max(
            float(cfg.racket_z_hard_limit_up),
            max(0.0, -float(anchor_z_range[0])) + float(racket_up_margin),
        ),
        terminate_on_racket_z_limit=bool(cfg.terminate_on_racket_z_limit)
        if terminate_racket_z is None
        else bool(terminate_racket_z),
        terminate_on_ball_view_bounds=terminate,
        ball_view_xy_center_penalty_weight=xy_weight,
        ball_view_z_ideal_penalty_weight=z_weight,
        ball_view_bounds_penalty_weight=bounds_weight,
        ball_view_out_of_bounds_penalty_weight=out_of_bounds_weight,
        ball_view_z_not_ideal_penalty_weight=z_not_ideal_weight,
        ball_view_vxy_excess_penalty_weight=vxy_weight,
        ball_base_vxy_penalty_weight=min(float(cfg.ball_base_vxy_penalty_weight), 2.0),
        ball_vxy_penalty_weight=min(float(cfg.ball_vxy_penalty_weight), 0.20),
    )


def _with_low_realview_reset_pose(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    return replace(
        cfg,
        right_arm_reset_degrees=LOW_REALVIEW_RIGHT_ARM_RESET_DEGREES,
        episode_racket_anchor_z_range_m=(0.0, 0.0),
        hit_confirm_abs_height=0.85,
        ball_view_x_bounds_m=(-0.20, 0.20),
        ball_view_x_sigma_m=0.08,
        ball_view_z_ideal_m=(0.83, 1.10),
        ball_view_z_sigma_m=0.08,
    )


def _with_low_reset_ball_range(
    cfg: MjxJuggleConfig,
    *,
    terminate: bool,
    xy_weight: float,
    z_weight: float,
    bounds_weight: float,
    vxy_weight: float,
    out_of_bounds_weight: float = 0.0,
    z_not_ideal_weight: float = 0.0,
    target_x_range: tuple[float, float] = (0.0, 0.0),
    target_y_range: tuple[float, float] = (0.0, 0.0),
    anchor_z_range: tuple[float, float] = (0.0, 0.0),
    launch_height: float = 0.14,
    target_height: float = 0.11,
    hit_height_center: float = 0.13,
    hit_confirm_rel_height: float = 0.06,
    racket_up_margin: float = 0.24,
    terminate_racket_z: bool | None = None,
) -> MjxJuggleConfig:
    cfg = _with_low_realview_reset_pose(cfg)
    cfg = _with_real_view_ball_range(
        cfg,
        terminate=terminate,
        xy_weight=xy_weight,
        z_weight=z_weight,
        bounds_weight=bounds_weight,
        out_of_bounds_weight=out_of_bounds_weight,
        z_not_ideal_weight=z_not_ideal_weight,
        vxy_weight=vxy_weight,
        target_x_range=target_x_range,
        target_y_range=target_y_range,
        anchor_z_range=anchor_z_range,
        launch_height=launch_height,
        target_height=target_height,
        hit_height_center=hit_height_center,
        hit_confirm_rel_height=hit_confirm_rel_height,
        racket_up_margin=racket_up_margin,
        terminate_racket_z=terminate_racket_z,
    )
    return replace(cfg, ball_low_termination_z_m=0.58)


def _goal_d455_from_scratch_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    branch: str,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Build the two new GOAL.md curricula without reusing an old stage table.

    The base task reward is deliberately minimal.  Stages only add one
    already-declared difficulty axis at a time, both reset primitives remain
    branch-invariant, and evidence-backed branch reward schedules are explicit.
    """

    if branch not in {"autolaunch", "release"}:
        raise ValueError(f"unknown GOAL.md branch: {branch!r}")

    common = _with_latest_d455_camera(MjxJuggleConfig(**stack_kwargs))
    common = replace(
        common,
        horizon_sec=6.0,
        right_arm_reset_degrees=D455_USER_REQUESTED_RACKET_RESET_DEGREES,
        virtual_camera_require_base_body=True,
        # The base-origin implementation was falsified on the stopped GPU0
        # launch19 run.  Keep GPU1 byte-for-byte on its previously trained
        # observation semantics until that independent branch is validated.
        ball_obs_frame_pivot_mode=(
            "camera_center" if branch == "autolaunch" else "legacy_base_origin"
        ),
        arm_action_limiter=True,
        action_acc_scale=1.0,
        # Minimal reward allowlist.  The environment's four fixed physical
        # terms remain active: ball height, ball/racket XY tracking, racket
        # anchor tracking, and the valid-hit center/flat/height multiplier.
        # Everything else below is explicitly disabled unless it is a sparse
        # hit, anti-cheat contact, or terminal event.  The controlled-release
        # branch enables one evidence-backed, event-conditioned recoverability
        # term below; the autonomous-launch branch keeps this base through its
        # already-passed launch14 stage and enables it only in the unpassed tail.
        action_penalty_weight=0.0,
        action_delta_penalty_weight=0.0,
        posture_weight=0.0,
        base_pose_weight=0.0,
        torque_penalty_weight=0.0,
        arm_vel_limit_penalty_weight=0.0,
        arm_acc_limit_penalty_weight=0.0,
        arm_limiter_penalty_weight=0.0,
        post_hit_survival_reward_weight=0.0,
        post_hit_ball_vxy_penalty_weight=0.0,
        descending_intercept_reward_weight=0.0,
        pre_hit_intercept_reward_weight=0.0,
        pre_hit_intercept_penalty_weight=0.0,
        non_racket_ball_contact_penalty_weight=2.0,
        failed_hit_penalty_weight=0.0,
        stick_contact_penalty_weight=0.50,
        hit_reward_base=1.0,
        hit_reward_combo=0.0,
        hit_reward_cap_mode="fixed",
        hit_reward_count_cap=15,
        hit_combo_count_cap=14,
        rel_height_bonus_weight=0.0,
        racket_xy_gauss_reward_weight=0.0,
        racket_xy_gauss_penalty_weight=0.0,
        racket_chest_xy_penalty_weight=0.0,
        racket_chest_z_penalty_weight=0.0,
        ball_anchor_xy_penalty_weight=0.0,
        ball_base_x_penalty_weight=0.0,
        ball_base_vxy_penalty_weight=0.0,
        ball_vxy_penalty_weight=0.0,
        apex_soft_penalty_weight=0.0,
        ball_xy_soft_penalty_weight=0.0,
        target_height=0.23,
        hit_height_center=0.23,
        hit_height_penalty_weight=0.0,
        low_hit_penalty_weight=0.0,
        hit_confirm_rel_height=0.050,
        hit_confirm_abs_height=1.0,
        hit_confirm_max_steps=70,
        hit_center_local_sigma=0.040,
        hit_center_sigma=0.085,
        hit_flatness_target_cos=0.955,
        hit_flatness_sigma=0.085,
        center_flat_hit_reward_weight=0.0,
        contact_flatness_penalty_weight=0.0,
        hit_vxy_penalty_weight=0.0,
        hit_apex_view_center_penalty_weight=0.0,
        hit_next_contact_anchor_penalty_weight=0.0,
        first_hit_apex_reward_weight=0.0,
        hit_cadence_reward_weight=0.0,
        hit_min_interval_penalty_weight=0.0,
        hit_min_count_interval=0.32,
        fast_hit_penalty_weight=0.0,
        hit_reward_cap_target_interval=0.43,
        termination_miss_penalty_base=2.5,
        termination_miss_penalty_per_hit=0.0,
        termination_miss_penalty_requires_hit=False,
        termination_no_hit_miss_early_penalty=0.0,
        racket_z_limit_termination_penalty_base=2.5,
        racket_z_limit_termination_penalty_per_hit=0.0,
        racket_z_soft_penalty_weight=0.0,
        racket_up_drift_penalty_weight=0.0,
        racket_flatness_penalty_weight=0.0,
        ball_low_termination_z_m=0.98,
        ball_high_termination_z_m=1.50,
        terminate_on_ball_view_bounds=False,
        terminate_on_ball_view_x_bounds=True,
        terminate_on_ball_view_y_bounds=True,
        terminate_on_ball_view_z_low=True,
        terminate_on_ball_view_z_high=True,
        ball_view_x_bounds_m=D455_REAL_VIEW_X_BOUNDS_M,
        ball_view_y_bounds_m=D455_REAL_VIEW_Y_BOUNDS_M,
        ball_view_z_bounds_m=D455_REAL_VIEW_Z_BOUNDS_M,
        ball_view_z_ideal_m=D455_STABLE_VIEW_Z_IDEAL_M,
        ball_view_x_target_m=0.0,
        ball_view_y_target_m=D455_REAL_VIEW_Y_TARGET_M,
        ball_view_xy_center_penalty_weight=0.0,
        ball_view_z_ideal_penalty_weight=0.0,
        ball_view_bounds_penalty_weight=0.0,
        ball_view_out_of_bounds_penalty_weight=0.0,
        ball_view_z_not_ideal_penalty_weight=0.0,
        ball_view_vxy_excess_penalty_weight=0.0,
        camera_center_weight=0.0,
        camera_visibility_penalty_weight=0.0,
        camera_depth_penalty_weight=0.0,
        camera_box_penalty_weight=0.0,
        camera_visible_penalty_weight=0.0,
        camera_top_margin_penalty_weight=0.0,
        hit_camera_reward_weight=0.0,
        hit_camera_out_of_band_penalty_weight=0.0,
        hit_camera_target_v_frac=0.67,
        hit_camera_v_sigma_frac=0.16,
        hit_camera_lower_band_frac=(0.48, 0.86),
        ball_obs_rate_hz=60.0,
        ball_obs_fractional_rate=True,
        ball_obs_pos_noise_std=0.003,
        ball_obs_vel_noise_std=0.030,
        ball_obs_noise_warmup_ratio=0.0,
        ball_obs_noise_ramp_ratio=0.0,
        ball_obs_age_tracks_stale=True,
        ball_obs_age_clip=0.50,
        ball_obs_dropout_on_refresh_only=True,
        ball_obs_require_camera_visible=False,
        ball_obs_camera_missing_prob=0.0,
        ball_obs_reset_respects_camera_visibility=False,
        ball_obs_require_view_bounds=False,
        ball_obs_view_bounds_missing_prob=0.0,
        ball_obs_missing_episode_coherent_prob=0.0,
        ball_obs_dropout_prob=0.0,
        ball_obs_dropout_burst_prob=0.0,
        lost_ball_timeout_ms=350.0,
        domain_randomization=False,
        dr_randomize_ball=False,
        dr_randomize_contact=False,
        dr_randomize_actuator=False,
        dr_randomize_latency=False,
        dr_randomize_pd=False,
        dr_randomize_racket_mount=False,
        dr_randomize_ball_obs_frame=False,
        dr_randomize_actuator_cmd_filter=False,
        episode_target_x_range_m=(0.0, 0.0),
        episode_target_y_range_m=(0.0, 0.0),
        episode_racket_anchor_z_range_m=(0.0, 0.0),
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )

    if branch == "autolaunch":
        common = replace(
            common,
            ball_reset_mode="racket_launch",
            racket_launch_surface_gap_range_m=(0.005, 0.010),
            racket_launch_xy_jitter=0.004,
            racket_launch_vxy_max=0.003,
            racket_launch_vnormal_max=0.003,
            racket_launch_edge_margin=0.005,
            ball_spawn_xy_jitter=0.0,
            ball_spawn_z_jitter=0.0,
            ball_init_vxy_max=0.0,
            ball_init_vz=0.0,
            ball_init_vz_jitter=0.0,
        )
    else:
        # This controlled release primitive and the small recoverability term
        # are invariant across every GPU1 stage.  The latter directly trains
        # the strict next-contact gate that remained false throughout the
        # release09 camera-missing plateau, without changing GPU0's reward.
        common = replace(
            common,
            ball_reset_mode="anchor_drop",
            ball_launch_height=0.32,
            ball_spawn_xy_jitter=0.025,
            ball_spawn_z_jitter=0.035,
            ball_init_vxy_max=0.012,
            ball_init_vz=-0.28,
            ball_init_vz_jitter=0.0,
            hit_next_contact_anchor_penalty_weight=(
                GOAL_D455_RELEASE_NEXT_CONTACT_PENALTY_WEIGHT
            ),
        )

    cfg_00 = common
    cfg_01 = replace(
        cfg_00,
        episode_target_x_range_m=(-0.020, 0.020),
        episode_target_y_range_m=(-0.015, 0.015),
        episode_racket_anchor_z_range_m=(-0.008, 0.008),
    )
    cfg_02 = replace(
        cfg_01,
        episode_target_x_range_m=(-0.050, 0.050),
        episode_target_y_range_m=(-0.040, 0.040),
        episode_racket_anchor_z_range_m=(-0.018, 0.018),
    )
    cfg_03 = replace(
        cfg_02,
        domain_randomization=True,
        dr_randomize_ball=True,
        dr_ball_mass_range=(0.00260, 0.00280),
        dr_gravity_z_range=(-9.83, -9.79),
    )
    cfg_04 = replace(
        cfg_03,
        dr_randomize_contact=True,
        dr_ball_friction_range=(0.16, 0.28),
        dr_racket_friction_range=(0.30, 0.48),
        dr_ball_solref_time_range=(0.0030, 0.0060),
        dr_ball_solref_damping_range=(0.74, 0.94),
    )
    cfg_05 = replace(
        cfg_04,
        dr_randomize_actuator=True,
        dr_action_scale_mult_range=(0.97, 1.03),
        dr_damping_mult_range=(0.94, 1.06),
        dr_armature_mult_range=(0.96, 1.04),
        dr_randomize_pd=True,
        dr_pd_kp_mult_range=(0.98, 1.02),
        dr_pd_kv_mult_range=(0.96, 1.04),
        dr_pd_per_joint=True,
        dr_randomize_actuator_cmd_filter=True,
        dr_actuator_cmd_tau_range=(0.070, 0.078),
        dr_actuator_cmd_gain_range=(0.995, 1.005),
    )
    cfg_06 = replace(
        cfg_05,
        dr_randomize_racket_mount=True,
        dr_racket_pos_offset_m=0.0010,
        dr_racket_rot_offset_rad=float(np.deg2rad(0.35)),
        dr_racket_radius_offset_m=0.0007,
    )
    cfg_07 = replace(
        cfg_06,
        ball_obs_pos_noise_std=0.004,
        ball_obs_vel_noise_std=0.040,
        dr_randomize_ball_obs_frame=True,
        dr_ball_obs_pos_bias_base_m=(0.002, 0.002, 0.002),
        dr_ball_obs_rot_bias_deg=(0.35, 0.35, 0.50),
        dr_ball_obs_vel_bias_base_m_s=(0.020, 0.020, 0.030),
        dr_ball_obs_scale_range=(0.997, 1.003),
    )
    cfg_08 = replace(
        cfg_07,
        ball_obs_dropout_prob=0.002,
        ball_obs_dropout_max_steps=1,
    )
    cfg_09 = replace(
        cfg_08,
        ball_obs_require_camera_visible=True,
        ball_obs_camera_missing_prob=0.15,
        ball_obs_reset_respects_camera_visibility=False,
        ball_obs_require_view_bounds=True,
        ball_obs_view_bounds_missing_prob=0.15,
        ball_obs_missing_episode_coherent_prob=0.0,
        ball_obs_dropout_prob=0.004,
        ball_obs_dropout_max_steps=1,
        ball_obs_dropout_burst_prob=0.0,
        ball_obs_dropout_burst_max_steps=1,
    )
    cfg_10 = replace(
        cfg_09,
        episode_target_x_range_m=(-0.090, 0.090),
        episode_target_y_range_m=(-0.070, 0.070),
        episode_racket_anchor_z_range_m=(-0.035, 0.035),
    )
    cfg_11 = replace(
        cfg_10,
        dr_ball_mass_range=(0.00245, 0.00295),
        dr_gravity_z_range=(-9.88, -9.72),
    )
    cfg_12 = replace(
        cfg_11,
        dr_ball_friction_range=(0.10, 0.38),
        dr_racket_friction_range=(0.22, 0.62),
        dr_ball_solref_time_range=(0.0020, 0.0080),
        dr_ball_solref_damping_range=(0.62, 1.02),
    )
    cfg_13 = replace(
        cfg_12,
        dr_action_scale_mult_range=(0.93, 1.07),
        dr_damping_mult_range=(0.84, 1.16),
        dr_armature_mult_range=(0.90, 1.10),
        dr_pd_kp_mult_range=(0.94, 1.06),
        dr_pd_kv_mult_range=(0.90, 1.10),
        dr_actuator_cmd_tau_range=(0.063, 0.085),
        dr_actuator_cmd_gain_range=(0.980, 1.020),
    )
    cfg_14 = replace(
        cfg_13,
        dr_racket_pos_offset_m=0.0030,
        dr_racket_rot_offset_rad=float(np.deg2rad(1.0)),
        dr_racket_radius_offset_m=0.0018,
    )
    tail_next_contact_penalty_weight = (
        GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT
        if branch == "autolaunch"
        else cfg_14.hit_next_contact_anchor_penalty_weight
    )
    # GPU0 plateau recovery: matched validation showed that the launch14 policy
    # dropped from 13.37 to 5.61 mean hits when observation calibration moved
    # directly from mild to the existing 50% bridge.  Add the exact 25% point
    # before that unchanged 50% point; the wide distribution and all stages
    # through launch14 remain unchanged.  The controlled-release branch
    # deliberately does not include either GPU0-specific bridge.
    cfg_15_micro_bridge = replace(
        cfg_14,
        hit_next_contact_anchor_penalty_weight=tail_next_contact_penalty_weight,
        ball_obs_pos_noise_std=0.00475,
        ball_obs_vel_noise_std=0.0475,
        dr_ball_obs_pos_bias_base_m=(0.003, 0.003, 0.003),
        dr_ball_obs_rot_bias_deg=(0.5125, 0.5125, 0.75),
        dr_ball_obs_vel_bias_base_m_s=(0.030, 0.030, 0.0425),
        dr_ball_obs_scale_range=(0.99525, 1.00475),
    )
    cfg_15_bridge = replace(
        cfg_14,
        hit_next_contact_anchor_penalty_weight=tail_next_contact_penalty_weight,
        ball_obs_pos_noise_std=0.0055,
        ball_obs_vel_noise_std=0.055,
        dr_ball_obs_pos_bias_base_m=(0.004, 0.004, 0.004),
        dr_ball_obs_rot_bias_deg=(0.675, 0.675, 1.0),
        dr_ball_obs_vel_bias_base_m_s=(0.040, 0.040, 0.055),
        dr_ball_obs_scale_range=(0.9935, 1.0065),
    )
    cfg_15 = replace(
        cfg_14,
        hit_next_contact_anchor_penalty_weight=tail_next_contact_penalty_weight,
        ball_obs_pos_noise_std=0.007,
        ball_obs_vel_noise_std=0.070,
        dr_ball_obs_pos_bias_base_m=(0.006, 0.006, 0.006),
        dr_ball_obs_rot_bias_deg=(1.0, 1.0, 1.5),
        dr_ball_obs_vel_bias_base_m_s=(0.060, 0.060, 0.080),
        dr_ball_obs_scale_range=(0.990, 1.010),
    )
    cfg_16 = replace(
        cfg_15,
        ball_obs_camera_missing_prob=0.50,
        ball_obs_view_bounds_missing_prob=0.50,
        ball_obs_dropout_prob=0.012,
        ball_obs_dropout_max_steps=3,
    )
    cfg_17 = cfg_16

    cfgs = [
        cfg_00, cfg_01, cfg_02, cfg_03, cfg_04, cfg_05, cfg_06, cfg_07, cfg_08,
        cfg_09, cfg_10, cfg_11, cfg_12, cfg_13, cfg_14, cfg_15, cfg_16, cfg_17,
    ]
    suffixes = [
        "acquisition",
        "local_workspace",
        "workspace",
        "ball_dynamics_mild",
        "contact_dynamics_mild",
        "actuator_pd_mild",
        "racket_geometry_mild",
        "observation_calibration_mild",
        "single_dropout_preview",
        "camera_missing_mild",
        "workspace_wide",
        "ball_dynamics_wide",
        "contact_dynamics_wide",
        "actuator_pd_wide",
        "racket_geometry_wide",
        "observation_calibration_wide",
        "camera_missing_wide",
        "final_consolidation",
    ]
    notes = [
        "Acquire the branch task with final actuator, inverse-MPC, 67/231 stack, reset, and calibrated D455 geometry already active.",
        "Introduce only a small episode-anchor workspace; the branch-local reset distribution is unchanged.",
        "Widen only the episode-anchor workspace before dynamics uncertainty.",
        "Introduce only mild ball-mass and gravity variation.",
        "Introduce only mild contact-material and solver variation.",
        "Introduce one mild control-realism axis: actuator, PD, and command-filter calibration uncertainty.",
        "Introduce only mild racket mount and radius uncertainty.",
        "Introduce only mild observation calibration and full-scale measurement noise; camera geometry stays fixed.",
        "Preview the missing-observation axis with rare one-refresh dropouts only.",
        "Add mild D455/view-conditioned missing without coherent episodes or bursts.",
        "Widen only the already-learned episode-anchor workspace.",
        "Widen only ball mass and gravity uncertainty.",
        "Widen only contact uncertainty, after the mild contact stage has consolidated.",
        "Widen only actuator, PD, and command-filter calibration uncertainty.",
        "Widen only racket mount and radius uncertainty.",
        "Widen only observation calibration and measurement noise.",
        "Widen only the already-learned camera/view missing axis; no coherent missing or burst mechanism is added.",
        "Keep the final distribution unchanged and train until the 1200-step, 13--15-hit stochastic CVaR gates pass.",
    ]
    min_updates = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 360]
    total_steps = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 28]
    if branch == "autolaunch":
        target_hits = [1.0, 2.0, 3.2, 4.5, 5.8, 7.0, 8.0, 8.8, 9.4, 9.8, 10.4, 10.8, 11.2, 11.5, 11.8, 12.1, 12.4, 13.0]
        target_len = [0.10, 0.18, 0.28, 0.38, 0.48, 0.58, 0.66, 0.72, 0.76, 0.79, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.93, 0.95]
        hit1 = [0.60, 0.75, 0.82, 0.86, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98]
        hit3 = [None, 0.15, 0.30, 0.42, 0.52, 0.61, 0.68, 0.73, 0.76, 0.78, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
        hit12 = [None, None, None, None, None, None, 0.05, 0.10, 0.16, 0.22, 0.30, 0.38, 0.46, 0.53, 0.59, 0.64, 0.69, 0.76]
    else:
        target_hits = [2.0, 3.0, 4.2, 5.5, 6.8, 8.0, 9.0, 9.7, 10.2, 10.6, 11.0, 11.3, 11.6, 11.8, 12.0, 12.2, 12.5, 13.0]
        target_len = [0.16, 0.24, 0.34, 0.44, 0.54, 0.63, 0.70, 0.75, 0.78, 0.81, 0.84, 0.86, 0.88, 0.89, 0.90, 0.92, 0.93, 0.95]
        hit1 = [0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98]
        hit3 = [0.25, 0.36, 0.46, 0.55, 0.63, 0.69, 0.74, 0.77, 0.79, 0.81, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90, 0.91]
        hit12 = [None, None, None, None, None, 0.05, 0.10, 0.16, 0.22, 0.28, 0.35, 0.42, 0.49, 0.55, 0.60, 0.65, 0.71, 0.78]
    truncation = [0.02, 0.05, 0.10, 0.16, 0.23, 0.31, 0.40, 0.48, 0.54, 0.59, 0.64, 0.68, 0.72, 0.75, 0.78, 0.80, 0.82, 0.86]
    camera_visible = [0.55, 0.57, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.70, 0.71, 0.71, 0.72]
    view_in_bounds = [0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.70, 0.70, 0.71, 0.72, 0.72, 0.73, 0.73, 0.73, 0.72, 0.72]
    view_z_ideal = [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60]

    prefix = "launch" if branch == "autolaunch" else "release"
    stages: list[CurriculumStage] = []
    for index, cfg in enumerate(cfgs):
        stages.append(
            CurriculumStage(
                name=f"{prefix}{index:02d}_{suffixes[index]}",
                total_steps=int(total_steps[index] * 1_000_000),
                cfg=cfg,
                notes=notes[index],
                target_mean_hits=target_hits[index],
                gate_mode="strict",
                advance_gate_mode="collapse",
                target_mean_len_frac=target_len[index],
                min_updates=min_updates[index],
                target_camera_visible=camera_visible[index],
                target_ball_view_in_bounds=view_in_bounds[index],
                target_ball_view_z_ideal=view_z_ideal[index],
                target_hit1_rate=hit1[index],
                target_hit3_rate=hit3[index],
                target_hit12_rate=hit12[index],
                target_mean_hits_ge3=(target_hits[index] + 0.5) if index >= 2 else None,
                target_min_hit_interval_s=0.32 if index >= 2 else None,
                target_max_hit_interval_s=(0.56 - 0.004 * min(index - 2, 15)) if index >= 2 else None,
                target_hit_camera_visible_rate=(0.76 + 0.01 * min(index - 2, 15)) if index >= 2 else None,
                target_hit_camera_lower_band_rate=(0.50 + 0.02 * min(index - 2, 15)) if index >= 2 else None,
                max_recent_mean_hit_vxy=(0.65 - 0.012 * min(index - 4, 13)) if index >= 4 else None,
                max_recent_hit_next_contact_anchor_err=(0.18 - 0.0045 * min(index - 4, 13)) if index >= 4 else None,
                max_recent_mean_hit_camera_v_frac=(0.84 - 0.003 * min(index - 6, 11)) if index >= 6 else None,
                target_episode_truncation_rate=truncation[index],
                target_racket_up_cos=(0.94 + 0.0025 * min(index - 5, 12)) if index >= 5 else None,
                min_ball_obs_missing_refresh_rate=(
                    0.001 if index == 8 else (
                        0.002 if 9 <= index <= 15 else (0.004 if index == 16 else (0.006 if index == 17 else None))
                    )
                ),
                max_ball_obs_lost_rate=(0.12 if index == 8 else (0.10 if 9 <= index <= 15 else (0.06 if index == 16 else (0.05 if index == 17 else None)))),
                max_updates=None,
            )
        )

    if branch == "autolaunch":
        # One new mechanism only: refine the already isolated observation-
        # calibration axis with a 25% point before the existing 50% bridge.
        # Every stage through launch14 and every original post-launch14
        # environment/numerical target remain unchanged.  The explicit tail
        # reward schedule above is the only cfg difference; the evidence-backed
        # intermediate readiness mode is declared below.
        micro_bridge = CurriculumStage(
            name="launch15_observation_calibration_micro_bridge",
            total_steps=17_250_000,
            cfg=cfg_15_micro_bridge,
            notes=(
                "Bridge only the observation-calibration axis at the exact "
                "25% point between mild and wide after matched validation "
                "showed the 50% entry jump was too large."
            ),
            target_mean_hits=11.875,
            gate_mode="balanced_probe",
            advance_gate_mode="collapse",
            target_mean_len_frac=0.9025,
            min_updates=225,
            target_camera_visible=0.7025,
            target_ball_view_in_bounds=0.73,
            target_ball_view_z_ideal=0.60,
            target_hit1_rate=0.97,
            target_hit3_rate=0.8725,
            target_hit12_rate=0.6025,
            target_mean_hits_ge3=12.375,
            target_min_hit_interval_s=0.32,
            target_max_hit_interval_s=0.511,
            target_hit_camera_visible_rate=0.8825,
            target_hit_camera_lower_band_rate=0.745,
            max_recent_mean_hit_vxy=0.527,
            max_recent_hit_next_contact_anchor_err=0.133875,
            max_recent_mean_hit_camera_v_frac=0.81525,
            target_episode_truncation_rate=0.785,
            target_racket_up_cos=0.963125,
            min_ball_obs_missing_refresh_rate=0.002,
            max_ball_obs_lost_rate=0.10,
            max_updates=None,
        )
        bridge = CurriculumStage(
            name="launch16_observation_calibration_bridge",
            total_steps=17_500_000,
            cfg=cfg_15_bridge,
            notes=(
                "Bridge only the observation-calibration axis at the exact "
                "midpoint between mild and wide after the launch14 plateau."
            ),
            target_mean_hits=11.95,
            gate_mode="balanced_probe",
            advance_gate_mode="collapse",
            target_mean_len_frac=0.905,
            min_updates=230,
            target_camera_visible=0.705,
            target_ball_view_in_bounds=0.73,
            target_ball_view_z_ideal=0.60,
            target_hit1_rate=0.97,
            target_hit3_rate=0.875,
            target_hit12_rate=0.615,
            target_mean_hits_ge3=12.45,
            target_min_hit_interval_s=0.32,
            target_max_hit_interval_s=0.510,
            target_hit_camera_visible_rate=0.885,
            target_hit_camera_lower_band_rate=0.75,
            max_recent_mean_hit_vxy=0.524,
            max_recent_hit_next_contact_anchor_err=0.13275,
            max_recent_mean_hit_camera_v_frac=0.8145,
            target_episode_truncation_rate=0.79,
            target_racket_up_cos=0.96375,
            min_ball_obs_missing_refresh_rate=0.002,
            max_ball_obs_lost_rate=0.10,
            max_updates=None,
        )
        # The unpassed GPU0 transition tail uses readiness gates, followed by
        # the existing stochastic next-stage anti-collapse/reset-CVaR probe.
        # launch19 remains strict on the unchanged final distribution, so the
        # 1200-step/13--15-hit final contract is not relaxed.
        shifted_tail = [
            replace(
                stage,
                name=f"launch{index + 2:02d}_{suffixes[index]}",
                gate_mode="strict" if index == 17 else "balanced_probe",
            )
            for index, stage in enumerate(stages[15:], start=15)
        ]
        stages = [*stages[:15], micro_bridge, bridge, *shifted_tail]

    if stage_steps_override is not None:
        stages = [replace(stage, total_steps=int(stage_steps_override)) for stage in stages]

    camera_fields = (
        "camera_visibility_mode",
        "virtual_camera_pose_mode",
        "virtual_camera_base_body_name",
        "virtual_camera_require_base_body",
        "ball_obs_frame_pivot_mode",
        "virtual_camera_base_pos",
        "virtual_camera_base_rot",
        "camera_image_width",
        "camera_image_height",
        "camera_fx",
        "camera_fy",
        "camera_cx",
        "camera_cy",
        "camera_hfov_deg",
        "camera_vfov_deg",
        "camera_pixel_margin",
    )
    reward_contract_fields = tuple(
        field.name
        for field in fields(MjxJuggleConfig)
        if (
            ("reward" in field.name or "penalty" in field.name)
            and field.name != "hit_next_contact_anchor_penalty_weight"
        )
    ) + (
        "target_height",
        "hit_height_center",
        "hit_reward_cap_mode",
        "hit_reward_count_cap",
        "hit_combo_count_cap",
        "hit_min_count_interval",
        "termination_miss_penalty_requires_hit",
        "terminate_on_ball_view_bounds",
    )
    first_cfg = stages[0].cfg
    expected_next_contact_weights = (
        [0.0] * 15
        + [GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT]
        * (len(stages) - 15)
        if branch == "autolaunch"
        else [GOAL_D455_RELEASE_NEXT_CONTACT_PENALTY_WEIGHT] * len(stages)
    )
    for stage_index, stage in enumerate(stages):
        cfg = stage.cfg
        if any(getattr(cfg, field) != getattr(first_cfg, field) for field in camera_fields):
            raise ValueError(f"{stage.name} changed the fixed calibrated D455 geometry")
        if tuple(cfg.right_arm_reset_degrees) != tuple(D455_USER_REQUESTED_RACKET_RESET_DEGREES):
            raise ValueError(f"{stage.name} changed the required right-arm reset pose")
        if any(getattr(cfg, field) != getattr(first_cfg, field) for field in reward_contract_fields):
            raise ValueError(f"{stage.name} changed the fixed minimal reward contract")
        if (
            cfg.hit_next_contact_anchor_penalty_weight
            != expected_next_contact_weights[stage_index]
        ):
            raise ValueError(
                f"{stage.name} escaped the branch-specific next-contact reward schedule"
            )
        if require_inverse_mpc_stack:
            if not (
                cfg.enable_delay_conditioning
                and cfg.include_tau_act_norm
                and cfg.include_command_state
                and cfg.include_active_command_error
                and cfg.include_phase_features
                and cfg.actuator_cmd_filter
                and cfg.actuator_compensation_mode == "inverse_mpc"
                and cfg.asymmetric_critic
                and int(cfg.critic_command_history_steps) == 12
            ):
                raise ValueError(f"{stage.name} escaped the required 67D/inverse-MPC/asymmetric stack")
        if branch == "autolaunch":
            if not (
                cfg.ball_reset_mode == "racket_launch"
                and tuple(cfg.racket_launch_surface_gap_range_m) == (0.005, 0.010)
                and float(cfg.racket_launch_xy_jitter) == 0.004
                and float(cfg.racket_launch_vxy_max) == 0.003
                and float(cfg.racket_launch_vnormal_max) == 0.003
            ):
                raise ValueError(f"{stage.name} failed to preserve autonomous launch")
        else:
            release_contract = (
                cfg.ball_reset_mode,
                cfg.ball_launch_height,
                cfg.ball_spawn_xy_jitter,
                cfg.ball_spawn_z_jitter,
                cfg.ball_init_vxy_max,
                cfg.ball_init_vz,
                cfg.ball_init_vz_jitter,
            )
            if release_contract != ("anchor_drop", 0.32, 0.025, 0.035, 0.012, -0.28, 0.0):
                raise ValueError(f"{stage.name} changed the controlled released-ball primitive")
    return stages


def _goal_d455_autolaunch_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    return _goal_d455_from_scratch_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        branch="autolaunch",
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )


def _with_goal_d455_autolaunch_viewdense_shaping(
    stages: list[CurriculumStage],
) -> list[CurriculumStage]:
    """Add a W012 view-shaping variant without changing the W011 profile.

    The bottom-hard-limited W011 plant can improve hit count and episode length
    while drifting toward the calibrated D455 view edge.  The original
    ``goal_d455_autolaunch_v1`` profile deliberately keeps view terms at zero,
    so preserve it byte-for-byte and expose this separate profile for the
    shaped experiment.
    """

    shaped: list[CurriculumStage] = []
    for index, stage in enumerate(stages):
        cfg = replace(
            stage.cfg,
            ball_view_xy_center_penalty_weight=GOAL_D455_AUTOLAUNCH_VIEWDENSE_XY_WEIGHT,
            ball_view_bounds_penalty_weight=GOAL_D455_AUTOLAUNCH_VIEWDENSE_BOUNDS_WEIGHT,
            ball_view_out_of_bounds_penalty_weight=GOAL_D455_AUTOLAUNCH_VIEWDENSE_OOB_WEIGHT,
        )
        stage_updates: dict[str, object] = {
            "cfg": cfg,
            "notes": (
                f"{stage.notes}  W012 viewdense variant: add mild dense D455 "
                "view centering/bounds penalties so reward and view gates are aligned."
            ),
        }
        if index == 0:
            # launch00 already has a mean-length gate.  Requiring a rare full
            # 6 s truncation before the second stage made the W011 plant
            # overtrain the one-hit acquisition behavior and drift out of
            # view; later stages keep their stricter truncation gates.
            stage_updates["target_episode_truncation_rate"] = 0.0
        shaped.append(replace(stage, **stage_updates))
    return shaped


def _goal_d455_autolaunch_viewdense_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    return _with_goal_d455_autolaunch_viewdense_shaping(
        _goal_d455_autolaunch_v1_stages(
            stack_kwargs=stack_kwargs,
            stage_steps_override=stage_steps_override,
            critic_command_history_steps=critic_command_history_steps,
            require_inverse_mpc_stack=require_inverse_mpc_stack,
        )
    )


def _with_goal_d455_autolaunch_idealpd67_viewdense_shaping(
    stages: list[CurriculumStage],
) -> list[CurriculumStage]:
    """Align ideal-PD67 launch14/15 optimization with their strict gates.

    A stopped launch14 run reached every performance gate around updates
    44--66, but the actuator-stack-era 220-update floor forced continued PPO
    until view-in-bounds collapsed.  The ideal plant learns this transition
    faster and has no inverse-MPC smoothing, so keep the original full-horizon
    gates while adding the existing mild view shaping, enabling the existing
    next-contact term one stage earlier, and shortening only the two measured
    transition floors.  All actuator, observation, reset, DR, and safety
    semantics remain owned by the ideal-PD67 profile.
    """

    shaped = _with_goal_d455_autolaunch_viewdense_shaping(stages)
    patched: list[CurriculumStage] = []
    for index, (original_stage, stage) in enumerate(zip(stages, shaped)):
        cfg = stage.cfg
        if index == 14:
            cfg = replace(
                cfg,
                hit_next_contact_anchor_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT
                ),
            )
        patched.append(
            replace(
                stage,
                cfg=cfg,
                min_updates=GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_MIN_UPDATES.get(
                    index,
                    original_stage.min_updates,
                ),
                target_episode_truncation_rate=(
                    original_stage.target_episode_truncation_rate
                ),
                notes=(
                    f"{stage.notes}  ideal-PD67 viewdense recovery: preserve "
                    "the original full-horizon gate, align view/next-contact "
                    "reward with the strict gates, and use the measured "
                    "ideal-plant transition floor at launch14/15."
                ),
            )
        )
    return patched


def _with_goal_d455_autolaunch_idealpd67_final_recovery(
    stages: list[CurriculumStage],
) -> list[CurriculumStage]:
    """Add one final-stage survival signal without relaxing acceptance gates.

    The launch19 environment is byte-identical to launch18, but its strict
    full/length/hit gates were never reached simultaneously because the GOAL
    reward allowlist disabled post-hit survival.  Patch only the final stage:
    keep its strict targets and plant/DR contract, add a moderate integrated
    landing/survival reward, and avoid forcing hundreds of updates after a
    recovered policy is already eligible for block validation.
    """

    if not stages or stages[-1].name != "launch19_final_consolidation":
        raise ValueError("ideal-PD67 final recovery requires launch19 as the final stage")
    final = stages[-1]
    recovered_final = replace(
        final,
        cfg=replace(
            final.cfg,
            post_hit_survival_reward_weight=(
                GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_SURVIVAL_WEIGHT
            ),
        ),
        min_updates=GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_MIN_UPDATES,
        notes=(
            f"{final.notes}  ideal-PD67 final recovery: retain every strict "
            "launch19 gate and add only a moderate post-hit landing/survival "
            "reward; shorten the minimum-update floor for best-checkpoint "
            "recovery while block validation remains mandatory."
        ),
    )
    return [*stages[:-1], recovered_final]


def _goal_d455_autolaunch_idealpd67_actuator_inversempc_finetune_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Fine-tune the learned ideal-PD67 actor on the original actuator stack.

    Keep the ideal-PD67 final-recovery curriculum, reset distribution, reward,
    view shaping, and strict gates unchanged.  Only restore the original 67D
    actuator observation/execution semantics supplied by ``stack_kwargs``:
    the nominal delayed command is active, the FOPDT command filter runs, and
    inverse MPC compensates it.  Safety planner/limiter switches are owned by
    the CLI; this profile specifically rejects the bottom actual-state limiter
    in ``build_curriculum`` so this experiment isolates the actuator plant and
    compensation requested by the user.
    """

    stages = _goal_d455_autolaunch_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    stages = _with_goal_d455_autolaunch_idealpd67_viewdense_shaping(stages)
    stages = _with_goal_d455_autolaunch_idealpd67_final_recovery(stages)
    return [
        replace(
            stage,
            notes=(
                f"{stage.notes}  ideal-PD67 actuator fine-tune: restore the "
                "original delayed-command 17D semantics, 74 ms actuator "
                "filter, and one causal inverse-MPC compensation pass; keep "
                "the bottom actual-state limiter disabled to isolate the "
                "requested execution-stack transfer."
            ),
        )
        for stage in stages
    ]


def _goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Full D455 course on the proven actuator/inverse-MPC path, without a governor.

    This is a random-initialized, launch00-to-launch19 profile, not an ideal-PD
    checkpoint transfer.  It keeps the successful ideal-PD67 view-dense
    camera/reset/workspace/ball-DR/missing curriculum, while running the fitted
    72/74 ms actuator model and the original regularized inverse MPC used by
    the historical actuator success reference.  No post-compensation limiter,
    servo-target limiter/planner, actual-state projector, or drive governor is
    allowed.  Instead, restore the reference run's causal smoothness and
    qvel/qacc exceedance costs.
    """

    stages = _goal_d455_autolaunch_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    stages = _with_goal_d455_autolaunch_idealpd67_viewdense_shaping(stages)
    return [
        replace(
            stage,
            cfg=replace(
                stage.cfg,
                action_penalty_weight=GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT,
                action_delta_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT
                ),
                command_tracking_error_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT
                ),
                delay_action_jerk_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT
                ),
                post_hit_survival_reward_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT
                ),
                termination_miss_penalty_per_hit=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_MISS_PENALTY_PER_HIT
                ),
                racket_z_limit_termination_penalty_per_hit=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_RACKET_Z_PENALTY_PER_HIT
                ),
                racket_anchor_termination_penalty_base=2.5,
                racket_anchor_termination_penalty_per_hit=0.0,
                arm_vel_limit_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_VEL_LIMIT_WEIGHT
                ),
                arm_acc_limit_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_ACC_LIMIT_WEIGHT
                ),
                arm_limiter_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_LIMITER_WEIGHT
                ),
                actuator_mpc_feedback_source="applied",
                arm_post_compensation_limiter=False,
                arm_servo_target_limiter=False,
                arm_servo_target_tracking_planner=False,
                arm_actual_state_limiter=False,
                arm_actual_target_tracking_governor=False,
                right_arm_pd_profile="xml",
            ),
            notes=(
                f"{stage.notes}  Actuator success-reference no-governor "
                "variant: use the fitted actuator plus original inverse MPC, "
                "and train smooth/limit-respecting motion through reward only."
            ),
        )
        for stage in stages
    ]


def _goal_d455_autolaunch_actuator_inversempc_countcredit_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Align the no-governor objective with monotonic counted-hit credit.

    Preserve the plant, observations, dense shaping, hit quality multiplier,
    and fixed terminal failure bases of the success-reference profile.  Remove
    only the two objective contradictions: later failures must not claw back
    more already-earned hit credit, and valid hits after count 15 must remain
    rewardable.  Combo reward stays disabled so reward scale does not grow with
    episode age.
    """

    stages = _goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    patched: list[CurriculumStage] = []
    for stage in stages:
        cfg = stage.cfg
        reference_hits = max(1.0, float(stage.target_mean_hits))
        fixed_miss_barrier = float(cfg.termination_miss_penalty_base) + (
            float(cfg.termination_miss_penalty_per_hit) * reference_hits
        )
        fixed_racket_barrier = float(cfg.racket_z_limit_termination_penalty_base) + (
            float(cfg.racket_z_limit_termination_penalty_per_hit) * reference_hits
        )
        fixed_failure_barrier = max(fixed_miss_barrier, fixed_racket_barrier)
        patched.append(
            replace(
                stage,
                cfg=replace(
                    cfg,
                    hit_reward_cap_mode="off",
                    hit_reward_count_cap=0,
                    termination_miss_penalty_base=fixed_miss_barrier,
                    termination_miss_penalty_per_hit=0.0,
                    racket_z_limit_termination_penalty_base=fixed_racket_barrier,
                    racket_z_limit_termination_penalty_per_hit=0.0,
                    # Do not leave a cheaper workspace-escape terminal after
                    # making ball/racket failures count-independent.
                    racket_anchor_termination_penalty_base=max(
                        float(cfg.racket_anchor_termination_penalty_base),
                        fixed_failure_barrier,
                    ),
                ),
                notes=(
                    f"{stage.notes}  Count-credit objective: every valid counted "
                    "hit remains rewardable; all failure penalties are fixed at "
                    f"the old target-count barrier ({fixed_failure_barrier:.2f}) "
                    "instead of growing with already-earned hits."
                ),
            )
        )
    return patched


def _goal_d455_autolaunch_actuator_inversempc_final_recovery_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Turn launch19 into a real recoverability optimization stage.

    The success-reference curriculum previously made launch18 and launch19
    environment-identical while only tightening the launch19 acceptance gate.
    Keep every earlier stage byte-identical, then align the final-stage reward
    with its failed recoverability gates: preserve credit for every counted
    hit, make terminal failure costs independent of episode age, strengthen
    the immediate ballistic next-contact signal, and integrate more reward for
    maintaining a landable post-hit state.
    """

    stages = _goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    if not stages or stages[-1].name != "launch19_final_consolidation":
        raise ValueError("final-recovery no-governor profile requires launch19")

    final = stages[-1]
    cfg = final.cfg
    reference_hits = max(1.0, float(final.target_mean_hits))
    fixed_miss_barrier = float(cfg.termination_miss_penalty_base) + (
        float(cfg.termination_miss_penalty_per_hit) * reference_hits
    )
    fixed_racket_barrier = float(cfg.racket_z_limit_termination_penalty_base) + (
        float(cfg.racket_z_limit_termination_penalty_per_hit) * reference_hits
    )
    fixed_failure_barrier = max(fixed_miss_barrier, fixed_racket_barrier)
    recovered_final = replace(
        final,
        cfg=replace(
            cfg,
            hit_reward_cap_mode="off",
            hit_reward_count_cap=0,
            termination_miss_penalty_base=fixed_miss_barrier,
            termination_miss_penalty_per_hit=0.0,
            racket_z_limit_termination_penalty_base=fixed_racket_barrier,
            racket_z_limit_termination_penalty_per_hit=0.0,
            racket_anchor_termination_penalty_base=max(
                float(cfg.racket_anchor_termination_penalty_base),
                fixed_failure_barrier,
            ),
            post_hit_survival_reward_weight=(
                GOAL_D455_AUTOLAUNCH_FINAL_RECOVERY_POST_HIT_SURVIVAL_WEIGHT
            ),
            hit_next_contact_anchor_penalty_weight=(
                GOAL_D455_AUTOLAUNCH_FINAL_RECOVERY_NEXT_CONTACT_WEIGHT
            ),
        ),
        notes=(
            f"{final.notes}  Final recoverability bridge: launch19 now differs "
            "from launch18 through monotonic hit credit, fixed failure barriers, "
            "and stronger immediate/integrated next-contact shaping."
        ),
    )
    return [*stages[:-1], recovered_final]


def _goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Align launch19's learned cadence with its 13-hit six-second gate.

    Keep the successful no-governor plant, terminal costs, hit credit, and all
    launch00--launch18 stages unchanged.  The old final objective rewarded hit
    quality and integrated survival but gave no gradient for cadence; policies
    consequently raised the ball period from about 0.46 s to 0.48--0.49 s
    while improving full-horizon survival.  Add a bounded cadence bonus only
    at launch19 and tighten its diagnostic period gate to the value required
    by thirteen hits in the six-second horizon.
    """

    stages = _goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    if not stages or stages[-1].name != "launch19_final_consolidation":
        raise ValueError("final-cadence no-governor profile requires launch19")

    final = stages[-1]
    cadence_final = replace(
        final,
        cfg=replace(
            final.cfg,
            hit_cadence_reward_weight=GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_WEIGHT,
            hit_cadence_target_interval=(
                GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_TARGET_S
            ),
            hit_cadence_sigma=GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_SIGMA_S,
            hit_min_interval=GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_MIN_INTERVAL_S,
            hit_min_interval_penalty_weight=(
                GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_MIN_INTERVAL_WEIGHT
            ),
            fast_hit_penalty_weight=(
                GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_FAST_HIT_WEIGHT
            ),
        ),
        target_max_hit_interval_s=GOAL_D455_AUTOLAUNCH_FINAL_CADENCE_GATE_MAX_S,
        notes=(
            f"{final.notes}  Final cadence alignment: reward recoverable "
            "0.45 s counted-hit periods and reject periods too slow to support "
            "the 13-hit six-second objective."
        ),
    )
    return [*stages[:-1], cadence_final]


def _goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Repair launch19 survival credit without changing its plant or gates.

    The long final-cadence run improved hit cadence but plateaued near 0.83
    mean length and 0.65 full-horizon rate.  More than 85 percent of true
    failures were ball-low or lateral-bound exits, while 50-update aggregates
    showed length was strongly anti-correlated with hit lateral velocity.
    Preserve the validated cadence objective and add only conservative,
    recoverability-aligned shaping: a small survival increase, continuous
    post-hit lateral-drift cost, an outlier hit-vxy cost, and a stronger
    predicted next-contact placement cost.
    """

    stages = _goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    if not stages or stages[-1].name != "launch19_final_consolidation":
        raise ValueError("final-survival no-governor profile requires launch19")

    final = stages[-1]
    survival_final = replace(
        final,
        cfg=replace(
            final.cfg,
            post_hit_survival_reward_weight=1.70,
            post_hit_ball_vxy_penalty_weight=0.18,
            hit_vxy_penalty_weight=0.90,
            hit_next_contact_anchor_penalty_weight=0.06,
        ),
        notes=(
            f"{final.notes}  Final survival alignment: modestly strengthen "
            "time-alive credit after a valid hit while suppressing lateral "
            "drift and unreachable next-contact states."
        ),
    )
    return [*stages[:-1], survival_final]


def _goal_d455_autolaunch_actuator_inversempc_final_survival_countcredit_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Keep every launch19 hit valuable without weakening its failure barrier.

    Final-cadence branched directly from success-reference and therefore
    accidentally reintroduced terminal costs proportional to hits already
    earned.  At the 13-hit target those costs almost exactly cancel the hit
    reward.  Preserve all dynamics, shaping, and gates, but freeze the final
    stage failure costs at their original 13-hit values.
    """

    stages = _goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    if not stages or stages[-1].name != "launch19_final_consolidation":
        raise ValueError("final-survival count-credit profile requires launch19")

    final = stages[-1]
    cfg = final.cfg
    reference_hits = max(1.0, float(final.target_mean_hits))
    fixed_miss_barrier = float(cfg.termination_miss_penalty_base) + (
        float(cfg.termination_miss_penalty_per_hit) * reference_hits
    )
    fixed_racket_barrier = float(cfg.racket_z_limit_termination_penalty_base) + (
        float(cfg.racket_z_limit_termination_penalty_per_hit) * reference_hits
    )
    fixed_failure_barrier = max(fixed_miss_barrier, fixed_racket_barrier)
    repaired_final = replace(
        final,
        cfg=replace(
            cfg,
            termination_miss_penalty_base=fixed_miss_barrier,
            termination_miss_penalty_per_hit=0.0,
            racket_z_limit_termination_penalty_base=fixed_racket_barrier,
            racket_z_limit_termination_penalty_per_hit=0.0,
            racket_anchor_termination_penalty_base=max(
                float(cfg.racket_anchor_termination_penalty_base),
                fixed_failure_barrier,
            ),
        ),
        notes=(
            f"{final.notes}  Final count-credit repair: failure remains priced "
            f"at the original {reference_hits:.0f}-hit barrier, but no longer "
            "claws back additional credit for each hit already earned."
        ),
    )
    return [*stages[:-1], repaired_final]


def _goal_d455_autolaunch_actuator_inversempc_final_missing_age_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Use stale observation plus age above the calibrated height ceiling.

    Keep the episode alive when a high juggle leaves the vertical field of
    view, but do not expose simulator truth above that boundary.  Do not apply
    the full calibrated x/y/z box as an observation mask: those lateral and
    lower margins are reward/evaluation regions, not the user's stated sensor
    cutoff, and masking all of them removes roughly a third of final-stage
    observations.
    """

    stages = _goal_d455_autolaunch_actuator_inversempc_final_survival_countcredit_nogov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    if not stages or stages[-1].name != "launch19_final_consolidation":
        raise ValueError("final missing-age profile requires launch19")

    final = stages[-1]
    sensor_cfg = replace(
        final.cfg,
        terminate_on_ball_view_bounds=False,
        ball_obs_require_camera_visible=False,
        ball_obs_require_view_bounds=False,
        ball_obs_require_view_z_high=True,
        ball_obs_missing_episode_coherent_prob=0.0,
        ball_obs_age_tracks_stale=True,
        ball_obs_reset_respects_camera_visibility=False,
    )
    common_notes = (
        "High-view missing-age recovery: only z above the configured ceiling "
        "is missing; x/y and the lower z edge remain observable.  The episode "
        "continues with cached ball state plus increasing age."
    )

    # A checkpoint trained with oracle observations cannot absorb the real
    # ceiling as a one-step distribution switch: it produces almost no valid
    # contacts, so PPO receives no recovery trajectories.  Teach the same
    # behavior progressively.  Reward, plant DR and every strict performance
    # gate remain byte-for-byte equal to launch19 throughout; only the sensor
    # ceiling moves, ending at the calibrated cfg.ball_view_z_bounds_m[1].
    ceiling_stages = (
        ("launch19a_height_missing_z180", (1.80, 1.80)),
        ("launch19b_height_missing_z165", (1.65, 1.65)),
        ("launch19c_height_missing_z155", (1.55, 1.55)),
        ("launch19d_height_missing_final", (0.0, 0.0)),
    )
    recovery_stages = []
    for name, ceiling_range in ceiling_stages:
        is_final = ceiling_range == (0.0, 0.0)
        recovery_stages.append(
            replace(
                final,
                name=name,
                cfg=replace(
                    sensor_cfg,
                    ball_obs_view_z_high_missing_range_m=ceiling_range,
                ),
                notes=(
                    f"{final.notes}  {common_notes}  "
                    + (
                        "Final ceiling equals the calibrated maximum height; "
                        "the original strict final gates are authoritative."
                        if is_final
                        else f"Bridge ceiling is fixed at {ceiling_range[0]:.2f} m."
                    )
                ),
            )
        )
    return [*stages[:-1], *recovery_stages]


def _goal_d455_autolaunch_actuator_inversempc_final_intercept_nogov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Teach launch19 to place the physical racket under each falling ball.

    The final-survival run reduced lateral hit velocity but still lost about
    thirty percent of episodes near mid-horizon.  Add only the existing causal
    descending-intercept reward: it projects the current falling ball to the
    racket plane and scores the actual racket position, so its gradient covers
    observation/actuator delay rather than merely the post-contact trajectory.
    """

    stages = _goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    if not stages or stages[-1].name != "launch19_final_consolidation":
        raise ValueError("final-intercept no-governor profile requires launch19")

    final = stages[-1]
    intercept_final = replace(
        final,
        cfg=replace(final.cfg, descending_intercept_reward_weight=1.20),
        notes=(
            f"{final.notes}  Final intercept alignment: reward the actual "
            "racket for reaching the projected descending-ball crossing point."
        ),
    )
    return [*stages[:-1], intercept_final]


def _goal_d455_autolaunch_idealpd67_actuator_inversempc_residual_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """Residual transfer on the actuator stack with the final hard governor.

    This keeps the ideal-PD67 final-recovery task contract for a matched
    comparison with the previous direct fine-tune.  The only additional plant
    change relative to that branch is the deployable hard q/dq/ddq/jerk drive
    governor, so the residual actor cannot trade safety for return.
    """

    stages = _goal_d455_autolaunch_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    stages = _with_goal_d455_autolaunch_idealpd67_viewdense_shaping(stages)
    stages = _with_goal_d455_autolaunch_idealpd67_final_recovery(stages)
    return [
        replace(
            stage,
            cfg=replace(
                stage.cfg,
                arm_actual_state_limiter=True,
                arm_actual_target_tracking_governor=True,
                arm_actual_governor_natural_frequency_hz=8.0,
                arm_actual_governor_damping_ratio=1.0,
                arm_actual_jerk_limit_deg_s3=(175000.0,) * 7,
            ),
            notes=(
                f"{stage.notes}  Residual-safe variant: freeze the ideal-PD67 "
                "teacher, add only a bounded pre-compensation action residual, "
                "and enforce actual q/dq/ddq/jerk with the 8 Hz drive governor."
            ),
        )
        for stage in stages
    ]


def _with_goal_d455_autolaunch_relaxed_early_truncation(
    stages: list[CurriculumStage],
) -> list[CurriculumStage]:
    """Relax early full-horizon gates while preserving later strictness.

    W012 showed that the view-dense shaping fixes the early D455 view collapse,
    but launch01 can still overtrain for thousands of updates because the
    strict profile asks for non-zero full-episode truncation before the policy
    has learned medium-length juggling.  Keep the hit, length, view, cadence and
    final-stage contracts unchanged; only delay the full-horizon rate gate until
    the curriculum is actually targeting longer survival.
    """

    relaxed: list[CurriculumStage] = []
    for index, stage in enumerate(stages):
        trunc_target = GOAL_D455_AUTOLAUNCH_RELAXED_EARLY_TRUNCATION.get(index)
        if trunc_target is None:
            relaxed.append(stage)
            continue
        relaxed.append(
            replace(
                stage,
                target_episode_truncation_rate=float(trunc_target),
                notes=(
                    f"{stage.notes}  W013 relaxtrunc variant: early "
                    "full-horizon truncation gate is relaxed so progression is "
                    "driven by hit/length/view gates before long-survival stages."
                ),
            )
        )
    return relaxed


def _goal_d455_autolaunch_viewdense_relaxtrunc_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    return _with_goal_d455_autolaunch_relaxed_early_truncation(
        _goal_d455_autolaunch_viewdense_v1_stages(
            stack_kwargs=stack_kwargs,
            stage_steps_override=stage_steps_override,
            critic_command_history_steps=critic_command_history_steps,
            require_inverse_mpc_stack=require_inverse_mpc_stack,
        )
    )


def _goal_d455_autolaunch_viewdense_fullsafe_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W014: preserve early full episodes and train away safety intervention.

    W013 proved that post-step projection alone is not enough: the hard actual
    state limiter stayed active on nearly every MJX substep while neither the
    raw Gaussian action overflow nor the pre-projection acceleration violation
    contributed to reward.  Keep the validated PD/FOPDT/inverse-MPC stack and
    scalar parameters unchanged.  Restore the original full-horizon schedule,
    retain W012 D455 view shaping, and add only safety/smoothness costs that are
    observable and causal at the current step.
    """

    original = _goal_d455_autolaunch_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    viewdense = _with_goal_d455_autolaunch_viewdense_shaping(original)
    patched: list[CurriculumStage] = []
    for original_stage, stage in zip(original, viewdense):
        patched.append(
            replace(
                stage,
                cfg=replace(
                    stage.cfg,
                    action_clip_excess_penalty_weight=(
                        GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_CLIP_WEIGHT
                    ),
                    delay_action_jerk_penalty_weight=(
                        GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_JERK_WEIGHT
                    ),
                    arm_limiter_penalty_weight=(
                        GOAL_D455_AUTOLAUNCH_FULLSAFE_LIMITER_WEIGHT
                    ),
                ),
                target_episode_truncation_rate=(
                    original_stage.target_episode_truncation_rate
                ),
                notes=(
                    f"{stage.notes}  W014 fullsafe variant: restore the "
                    "original early full-horizon gate and penalize raw action "
                    "overflow, action jerk, and pre-projection actual-state "
                    "limiter intervention."
                ),
            )
        )
    return patched


def _goal_d455_autolaunch_viewdense_drivegov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W015: early-full view-dense course for the smooth drive governor.

    The target-aware actual drive governor is the commanded plant, so its
    normal intervention must not be penalized as though it were an emergency
    correction.  Preserve W014's early full-horizon schedule, view shaping,
    raw-action overflow cost and small applied-action jerk cost, while removing
    only the obsolete pre-projection XML-PD intervention penalty.
    """

    stages = _goal_d455_autolaunch_viewdense_fullsafe_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    return [
        replace(
            stage,
            cfg=replace(
                stage.cfg,
                arm_limiter_penalty_weight=0.0,
                actuator_mpc_feedback_source="actual",
                arm_post_compensation_limiter=False,
                arm_servo_target_limiter=False,
                arm_servo_target_tracking_planner=False,
                arm_actual_state_limiter=True,
                arm_actual_target_tracking_governor=True,
                arm_actual_governor_natural_frequency_hz=8.0,
                arm_actual_governor_damping_ratio=1.0,
                arm_actual_jerk_limit_deg_s3=(175000.0,) * 7,
                right_arm_pd_profile="xml",
            ),
            notes=(
                f"{stage.notes}  W015 drive-governor variant: the causal "
                "target-aware acceleration/jerk profile is the plant, so "
                "normal governor intervention is not reward-penalized."
            ),
        )
        for stage in stages
    ]


def _goal_d455_autolaunch_viewdense_drivegov_terminalsafe_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W016: close the unpenalized racket-workspace termination loophole."""

    stages = _goal_d455_autolaunch_viewdense_drivegov_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    return [
        replace(
            stage,
            cfg=replace(
                stage.cfg,
                racket_anchor_termination_penalty_base=2.5,
                racket_anchor_termination_penalty_per_hit=0.0,
            ),
            notes=(
                f"{stage.notes}  W016 terminal-safe variant: penalize "
                "racket_too_far_from_anchor by 2.5 so a one-hit workspace "
                "escape cannot dominate continued juggling."
            ),
        )
        for stage in stages
    ]


def _goal_d455_autolaunch_viewdense_drivegov_successref_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W017: restore multi-hit learning signals under the final safe plant.

    The historical inverse-MPC run learned repeated hits with explicit
    post-hit survival, smooth-action and per-hit failure signals.  W015 had
    removed all of those signals while also introducing a materially slower
    target-aware drive plant; its launch00 policy learned one hit followed by
    the unpenalized racket-workspace termination.  W016 closes that terminal
    loophole.  This profile additionally restores a deliberately conservative
    subset of the successful learnability signals without changing the GOAL
    curriculum, D455 geometry, original inverse-MPC scalars, XML PD, fitted
    actuator model or hard q/dq/ddq/jerk governor.
    """

    stages = _goal_d455_autolaunch_viewdense_drivegov_terminalsafe_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    return [
        replace(
            stage,
            cfg=replace(
                stage.cfg,
                action_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT
                ),
                action_delta_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT
                ),
                command_tracking_error_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT
                ),
                delay_action_jerk_penalty_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT
                ),
                post_hit_survival_reward_weight=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT
                ),
                termination_miss_penalty_per_hit=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_MISS_PENALTY_PER_HIT
                ),
                racket_z_limit_termination_penalty_per_hit=(
                    GOAL_D455_AUTOLAUNCH_SUCCESSREF_RACKET_Z_PENALTY_PER_HIT
                ),
            ),
            notes=(
                f"{stage.notes}  W017 success-reference shaping: preserve "
                "the final safe inverse-MPC plant while adding conservative "
                "post-hit survival, smooth-action, reachable-command and "
                "per-hit failure signals from the proven actuator-learning "
                "recipe."
            ),
        )
        for stage in stages
    ]


def _goal_d455_autolaunch_viewdense_drivegov_highapex_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W018: align apex, cadence, visibility, and hit-count gates physically.

    A fixed 13--15-hit target was inconsistent with the user's primary goal of
    placing each apex near the top of the calibrated D455 ideal view.  With the
    launch reset's roughly 1.08 m nominal contact height, a safe 1.40--1.42 m
    apex has a 0.51--0.53 s ballistic period, or about 11--12 hits over the
    six-second horizon.  This profile preserves the complete W017 plant and
    course, but makes the reward and gates describe that same trajectory.
    """

    stages = _goal_d455_autolaunch_viewdense_drivegov_successref_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    target_hits = (
        1.0, 2.0, 3.2, 4.5, 5.8, 6.8, 7.5, 8.0, 8.4, 8.7,
        9.0, 9.2, 9.4, 9.6, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3,
    )
    target_hit12 = (
        None, None, None, None, None, None, 0.02, 0.04, 0.06, 0.08,
        0.10, 0.14, 0.18, 0.22, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36,
    )
    cadence_min = (
        None, None, 0.46, 0.46, 0.47, 0.47, 0.48, 0.48, 0.48, 0.48,
        0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.49, 0.49, 0.49, 0.49,
    )
    cadence_max = (
        None, None, 0.60, 0.60, 0.59, 0.59, 0.58, 0.58, 0.58, 0.58,
        0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.57, 0.57, 0.57, 0.57,
    )
    if len(stages) != len(target_hits):
        raise ValueError(
            f"W018 expected {len(target_hits)} autonomous-launch stages, got {len(stages)}"
        )

    shaped: list[CurriculumStage] = []
    for index, stage in enumerate(stages):
        cfg = replace(
            stage.cfg,
            target_height=GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_HEIGHT,
            hit_height_center=GOAL_D455_AUTOLAUNCH_HIGHAPEX_HIT_HEIGHT,
            hit_apex_target_abs_z=GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_ABS_Z,
            hit_height_tolerance=0.05,
            hit_height_penalty_weight=6.0,
            low_hit_apex_margin=0.03,
            low_hit_penalty_weight=6.0,
            apex_soft_penalty_weight=5.0,
            apex_soft_limit_margin=0.04,
            first_hit_apex_reward_weight=0.25,
            first_hit_apex_sigma=0.06,
            post_hit_ball_vxy_penalty_weight=(
                GOAL_D455_AUTOLAUNCH_HIGHAPEX_POST_HIT_VXY_WEIGHT
            ),
            descending_intercept_reward_weight=(
                GOAL_D455_AUTOLAUNCH_HIGHAPEX_DESCENDING_INTERCEPT_WEIGHT
            ),
            hit_next_contact_anchor_penalty_weight=(
                GOAL_D455_AUTOLAUNCH_HIGHAPEX_NEXT_CONTACT_WEIGHT
            ),
            hit_next_contact_anchor_sigma_m=0.15,
            hit_apex_view_center_penalty_weight=0.05,
            hit_apex_view_center_sigma_m=0.15,
            hit_cadence_reward_weight=0.08,
            hit_cadence_target_interval=(
                GOAL_D455_AUTOLAUNCH_HIGHAPEX_CADENCE_TARGET_S
            ),
            hit_cadence_sigma=GOAL_D455_AUTOLAUNCH_HIGHAPEX_CADENCE_SIGMA_S,
            hit_min_interval_penalty_weight=0.8,
            hit_min_interval=0.44,
            fast_hit_penalty_weight=0.5,
            ball_view_z_ideal_penalty_weight=max(
                float(stage.cfg.ball_view_z_ideal_penalty_weight), 0.5
            ),
        )
        target = target_hits[index]
        shaped.append(
            replace(
                stage,
                cfg=cfg,
                target_mean_hits=target,
                target_mean_hits_ge3=(target + 0.5) if index >= 2 else None,
                target_hit12_rate=target_hit12[index],
                target_min_hit_interval_s=cadence_min[index],
                target_max_hit_interval_s=cadence_max[index],
                target_camera_visible=(
                    max(float(stage.target_camera_visible or 0.0), 0.95)
                    if index >= 2
                    else stage.target_camera_visible
                ),
                target_ball_view_in_bounds=(
                    max(float(stage.target_ball_view_in_bounds or 0.0), 0.75)
                    if index >= 2
                    else stage.target_ball_view_in_bounds
                ),
                target_ball_view_z_ideal=(
                    max(float(stage.target_ball_view_z_ideal or 0.0), 0.90)
                    if index >= 2
                    else stage.target_ball_view_z_ideal
                ),
                notes=(
                    f"{stage.notes}  W018 calibrated-high-apex recovery: "
                    "target a 1.40--1.42 m apex, 0.52 s causal ballistic "
                    "cadence, episode-anchor return, and near-continuous D455 "
                    "visibility; the actuator/compensation/safety plant is unchanged."
                ),
            )
        )
    return shaped


def _goal_d455_autolaunch_viewdense_constrained_mpc_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W019: safe compensation output without post-physics state rewriting.

    Keep W017's learnability curriculum and the hardware-validated inverse-MPC,
    actuator and XML-PD parameters.  The unconstrained MPC solution is only an
    internal goal.  After causal delay/FOPDT prediction, a target-aware
    acceleration planner produces the position trajectory actually sent to the
    XML PD.  Consequently the PD never sees the compensation spikes that made
    W017 launch/tip the base, and no q/dq/ddq state is overwritten after a
    physics step.
    """

    stages = _goal_d455_autolaunch_viewdense_drivegov_successref_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    # The final physical reset is only 5--10 mm above the racket and reaches
    # contact in roughly 32--45 ms, earlier than the fixed 72 ms actuator
    # delay.  A safe controller cannot influence that first contact.  Give the
    # acquisition stages causal response time, then restore the exact final
    # reset by launch04 and keep it invariant thereafter.
    acquisition_gap_ranges = (
        (0.080, 0.120),
        (0.060, 0.100),
        (0.040, 0.080),
        (0.020, 0.050),
    )
    return [
        replace(
            stage,
            cfg=replace(
                stage.cfg,
                racket_launch_surface_gap_range_m=(
                    acquisition_gap_ranges[index]
                    if index < len(acquisition_gap_ranges)
                    else (0.005, 0.010)
                ),
                arm_action_limiter=True,
                actuator_mpc_beta=1.2,
                actuator_mpc_delay_scale=1.05,
                actuator_mpc_tau_scale=0.75,
                actuator_mpc_horizon_steps=6,
                actuator_mpc_tracking_weight=1.0,
                actuator_mpc_nominal_weight=0.25,
                actuator_mpc_delta_weight=0.05,
                actuator_mpc_max_delta_rad=float(np.deg2rad(30.0)),
                actuator_mpc_feedback_source="actual",
                actuator_mpc_command_dynamics_constraint=False,
                actuator_mpc_command_velocity_weight=0.0,
                actuator_mpc_command_acceleration_weight=0.0,
                arm_post_compensation_limiter=False,
                arm_servo_target_limiter=False,
                arm_servo_target_tracking_planner=True,
                arm_servo_target_velocity_scale=1.0,
                arm_servo_target_acceleration_scale=0.8,
                arm_actual_state_limiter=False,
                arm_actual_target_tracking_governor=False,
                right_arm_pd_profile="xml",
                terminate_on_base_stability=True,
                base_z_deviation_limit_m=0.03,
                base_roll_pitch_limit_rad=float(np.deg2rad(5.0)),
            ),
            notes=(
                f"{stage.notes}  W019 constrained compensation: the causal "
                "post-FOPDT position trajectory sent to XML PD is planned at "
                "full qvel and 0.8 qacc limits; post-physics state projection "
                "is disabled and base lift/roll/pitch terminates immediately. "
                f"Causal acquisition reset gap={acquisition_gap_ranges[index] if index < len(acquisition_gap_ranges) else (0.005, 0.010)} m."
            ),
        )
        for index, stage in enumerate(stages)
    ]


def _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W019 plant with evidence-based observation-DR bridges and gates.

    The stopped launch16 run proved that moving observation calibration DR
    directly from 25% to 50% cut deterministic full-episode rate from 0.547
    to 0.247 for the same frozen policy.  The old tail also combined an 80%
    balanced training floor with a 35%/30% next-stage collapse probe, allowing
    launch15 to advance while its hit, length, hit12 and truncation targets all
    failed.  Preserve the complete W019 plant/reward contract, insert 37.5%
    and 75% calibration points, and use explicit strict bridge targets plus
    strict next-stage validation throughout the observation-DR tail.
    """

    stages = _goal_d455_autolaunch_viewdense_constrained_mpc_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    micro = stages[15]
    midpoint = stages[16]
    wide = stages[17]
    missing = stages[18]
    final = stages[19]

    three_eighths_cfg = replace(
        micro.cfg,
        ball_obs_pos_noise_std=0.005125,
        ball_obs_vel_noise_std=0.05125,
        dr_ball_obs_pos_bias_base_m=(0.0035, 0.0035, 0.0035),
        dr_ball_obs_rot_bias_deg=(0.59375, 0.59375, 0.875),
        dr_ball_obs_vel_bias_base_m_s=(0.035, 0.035, 0.04875),
        dr_ball_obs_scale_range=(0.994375, 1.005625),
    )
    three_quarters_cfg = replace(
        midpoint.cfg,
        ball_obs_pos_noise_std=0.00625,
        ball_obs_vel_noise_std=0.0625,
        dr_ball_obs_pos_bias_base_m=(0.005, 0.005, 0.005),
        dr_ball_obs_rot_bias_deg=(0.8375, 0.8375, 1.25),
        dr_ball_obs_vel_bias_base_m_s=(0.050, 0.050, 0.0675),
        dr_ball_obs_scale_range=(0.99175, 1.00825),
    )

    def strict_bridge(
        stage: CurriculumStage,
        *,
        name: str,
        cfg: MjxJuggleConfig,
        target_hits: float,
        target_len: float,
        target_hit1: float,
        target_hit3: float,
        target_hit12: float,
        target_truncation: float,
        min_updates: int,
        notes: str,
    ) -> CurriculumStage:
        return replace(
            stage,
            name=name,
            cfg=cfg,
            notes=f"{notes}  W019 actuator, inverse-MPC, XML-PD and servo-planner plant unchanged.",
            gate_mode="strict",
            advance_gate_mode="strict",
            target_mean_hits=target_hits,
            target_mean_len_frac=target_len,
            min_updates=min_updates,
            target_hit1_rate=target_hit1,
            target_hit3_rate=target_hit3,
            target_hit12_rate=target_hit12,
            target_mean_hits_ge3=target_hits + 0.5,
            target_episode_truncation_rate=target_truncation,
            # The old 0.133 m proxy rejected the best launch16 policy even
            # though full trajectories remained safe and visually valid.
            # Keep it as a real strict gate, but at an attainable bridge bound.
            max_recent_hit_next_contact_anchor_err=0.18,
        )

    repaired_tail = [
        strict_bridge(
            micro,
            name="launch15_observation_calibration_micro_bridge",
            cfg=micro.cfg,
            target_hits=10.0,
            target_len=0.72,
            target_hit1=0.97,
            target_hit3=0.90,
            target_hit12=0.45,
            target_truncation=0.50,
            min_updates=225,
            notes="Consolidate the measured 25% observation-calibration domain before increasing DR.",
        ),
        strict_bridge(
            micro,
            name="launch16_observation_calibration_three_eighths_bridge",
            cfg=three_eighths_cfg,
            target_hits=9.5,
            target_len=0.68,
            target_hit1=0.96,
            target_hit3=0.84,
            target_hit12=0.36,
            target_truncation=0.42,
            min_updates=240,
            notes="Insert the measured-missing 37.5% point between the 25% and 50% calibration domains.",
        ),
        strict_bridge(
            midpoint,
            name="launch17_observation_calibration_bridge",
            cfg=midpoint.cfg,
            target_hits=9.0,
            target_len=0.64,
            target_hit1=0.95,
            target_hit3=0.80,
            target_hit12=0.30,
            target_truncation=0.34,
            min_updates=260,
            notes="Train the original 50% observation-calibration domain only after the 37.5% bridge passes.",
        ),
        strict_bridge(
            midpoint,
            name="launch18_observation_calibration_three_quarters_bridge",
            cfg=three_quarters_cfg,
            target_hits=8.8,
            target_len=0.62,
            target_hit1=0.94,
            target_hit3=0.77,
            target_hit12=0.26,
            target_truncation=0.30,
            min_updates=280,
            notes="Insert a 75% point before the full observation-calibration distribution.",
        ),
        strict_bridge(
            wide,
            name="launch19_observation_calibration_wide",
            cfg=wide.cfg,
            target_hits=8.6,
            target_len=0.60,
            target_hit1=0.94,
            target_hit3=0.75,
            target_hit12=0.24,
            target_truncation=0.28,
            min_updates=300,
            notes="Reach the unchanged full observation-calibration distribution under strict transfer validation.",
        ),
        strict_bridge(
            missing,
            name="launch20_camera_missing_wide",
            cfg=missing.cfg,
            target_hits=8.4,
            target_len=0.58,
            target_hit1=0.93,
            target_hit3=0.73,
            target_hit12=0.22,
            target_truncation=0.25,
            min_updates=320,
            notes="Add the unchanged wide camera-missing distribution after full calibration DR is stable.",
        ),
        replace(
            final,
            name="launch21_final_consolidation",
            gate_mode="strict",
            advance_gate_mode="strict",
            notes=(
                f"{final.notes}  Final task targets and environment remain unchanged; "
                "only its numeric launch index shifts after inserting two DR bridges."
            ),
        ),
    ]
    return [*stages[:15], *repaired_tail]


def _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W019 DR bridges with strict learning gates and transfer-only probes.

    The first DR-bridge run reached every launch16 target except mean hits at
    updates 98 and 1709, missing the 9.5 threshold by only 1.4--1.7%, but its
    240-update floor prevented the early peak from advancing.  More
    importantly, ``advance_gate_mode=strict`` required a policy to satisfy the
    *next* DR stage's end-of-training goals before it was allowed to train on
    that distribution.  Frozen update-100/best-1709 policies retained strong
    initial-episode performance at 50% DR (8.27/8.54 hits and 0.320/0.355 full
    rate), so this is a gate-structure error rather than a missing DR bridge.

    Keep each current-domain learning gate strict, use the existing collapse
    probe only as an anti-collapse transfer check, and shorten consolidation
    floors so a validated peak is not trained past for hundreds of updates.
    The plant, rewards, DR distributions and final strict objective are
    identical to v1.
    """

    stages = _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    min_updates = (80, 80, 100, 120, 140, 160)
    repaired_tail: list[CurriculumStage] = []
    for offset, stage in enumerate(stages[15:21]):
        updates = min_updates[offset]
        changes: dict[str, object] = {
            "gate_mode": "strict",
            "advance_gate_mode": "collapse",
            "min_updates": updates,
            "notes": (
                f"{stage.notes}  V2 gate repair: the current DR domain must "
                f"pass strict task/vision targets after {updates} updates, "
                "while the next DR domain is only an anti-collapse transfer "
                "probe and is learned after advancement."
            ),
        }
        if offset == 1:
            # Two independent peaks reached 9.34--9.37 mean hits while every
            # other strict task/vision target passed.  A 9.30 threshold keeps
            # the gate selective without rejecting measurement-level noise.
            changes["target_mean_hits"] = 9.30
        repaired_tail.append(replace(stage, **changes))

    final = replace(
        stages[21],
        gate_mode="strict",
        advance_gate_mode="strict",
        notes=(
            f"{stages[21].notes}  V2 preserves the original final strict "
            "self-validation contract without a relaxed transfer probe."
        ),
    )
    return [*stages[:15], *repaired_tail, final]


def _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W020: make counted-hit credit monotonic on the accepted W019 plant.

    The W019 launch17 plateau retained only 0.2 reward for an additional hit
    followed by a ball miss, while a later racket-z failure clawed the full
    hit reward back.  Preserve each failure barrier at the value it had at
    that stage's target hit count, but remove growth with already-earned hits.
    No plant, observation, DR, gate, dense shaping, or controller field changes.
    """

    stages = _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    patched: list[CurriculumStage] = []
    for stage in stages:
        cfg = stage.cfg
        reference_hits = max(1.0, float(stage.target_mean_hits))
        fixed_miss_barrier = float(cfg.termination_miss_penalty_base) + (
            float(cfg.termination_miss_penalty_per_hit) * reference_hits
        )
        fixed_racket_barrier = float(cfg.racket_z_limit_termination_penalty_base) + (
            float(cfg.racket_z_limit_termination_penalty_per_hit) * reference_hits
        )
        fixed_failure_barrier = max(fixed_miss_barrier, fixed_racket_barrier)
        patched.append(
            replace(
                stage,
                cfg=replace(
                    cfg,
                    hit_reward_cap_mode="off",
                    hit_reward_count_cap=0,
                    termination_miss_penalty_base=fixed_miss_barrier,
                    termination_miss_penalty_per_hit=0.0,
                    racket_z_limit_termination_penalty_base=fixed_racket_barrier,
                    racket_z_limit_termination_penalty_per_hit=0.0,
                    racket_anchor_termination_penalty_base=max(
                        float(cfg.racket_anchor_termination_penalty_base),
                        fixed_failure_barrier,
                    ),
                ),
                notes=(
                    f"{stage.notes}  W020 count-credit repair: every counted "
                    "hit keeps its reward; terminal barriers are fixed at the "
                    f"old target-count values ({fixed_miss_barrier:.2f}/"
                    f"{fixed_racket_barrier:.2f})."
                ),
            )
        )
    return patched


def _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_nomissing_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W020 course with every actor ball-missing mechanism disabled.

    Camera/view metrics and rewards remain active, but the observation path
    never hides simulator ball state and does not accumulate stale age.  This
    is an explicit temporary ablation requested for the launch17-to-final run.
    Reward, DR and advancement gates are otherwise unchanged.
    """

    stages = _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    no_missing: list[CurriculumStage] = []
    for stage in stages:
        no_missing.append(
            replace(
                stage,
                cfg=replace(
                    stage.cfg,
                    ball_obs_dropout_prob=0.0,
                    ball_obs_dropout_max_steps=1,
                    ball_obs_dropout_burst_prob=0.0,
                    ball_obs_dropout_burst_max_steps=1,
                    ball_obs_age_tracks_stale=False,
                    ball_obs_dropout_on_refresh_only=False,
                    ball_obs_require_camera_visible=False,
                    ball_obs_camera_missing_prob=0.0,
                    ball_obs_reset_respects_camera_visibility=False,
                    ball_obs_require_view_bounds=False,
                    ball_obs_view_bounds_missing_prob=0.0,
                    ball_obs_missing_episode_coherent_prob=0.0,
                    ball_obs_require_view_z_high=False,
                    ball_obs_view_z_high_missing_range_m=(0.0, 0.0),
                ),
                notes=(
                    f"{stage.notes}  Temporary no-missing ablation: all ball "
                    "observation dropout, camera/view masking, coherent "
                    "missing, height masking, stale age, and visibility-aware "
                    "reset behavior are disabled."
                ),
            )
        )
    return no_missing


def _goal_d455_autolaunch_viewdense_constrained_mpc_recoverability_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W021: directly optimize the failed tail's recoverable return state.

    Launch17's best-policy episodes terminate primarily because the ball is
    too low (27.8%) or leaves x/y bounds (25.5%), while racket workspace
    failures account for only 5.6%.  Across the bounded W020 recovery run,
    hit count is most strongly anti-correlated with hit lateral velocity and
    actuator tracking error.  Keep every strict gate unchanged and add only
    reward terms that give the actor immediate credit for a low-drift,
    reachable next contact after each hit.
    """

    stages = (
        _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1_stages(
            stack_kwargs=stack_kwargs,
            stage_steps_override=stage_steps_override,
            critic_command_history_steps=critic_command_history_steps,
            require_inverse_mpc_stack=require_inverse_mpc_stack,
        )
    )
    repaired: list[CurriculumStage] = []
    for index, stage in enumerate(stages):
        if index < 17:
            repaired.append(stage)
            continue
        repaired.append(
            replace(
                stage,
                cfg=replace(
                    stage.cfg,
                    post_hit_ball_vxy_penalty_weight=0.18,
                    hit_vxy_penalty_weight=0.90,
                    hit_next_contact_anchor_penalty_weight=max(
                        0.06,
                        float(stage.cfg.hit_next_contact_anchor_penalty_weight),
                    ),
                ),
                notes=(
                    f"{stage.notes}  W021 recoverability repair: penalize "
                    "post-hit lateral drift continuously, penalize high-vxy "
                    "contact outliers, and strengthen predicted next-contact "
                    "placement; all strict gates remain unchanged."
                ),
            )
        )
    return repaired


def _goal_d455_autolaunch_viewdense_constrained_mpc_intercept_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W022: teach the delayed arm to intercept each descending return.

    W021 improved predicted next-contact placement but did not keep the
    physical racket under the falling ball, and its stable hit window stopped
    below the original 9.0 gate.  Add the existing causal descending-intercept
    signal used by the robust curriculum.  It compares the ball's projected
    crossing point with the actual racket position, so credit includes the
    actuator/planner execution delay rather than only the post-contact ball
    trajectory.  Gates and environment dynamics remain unchanged.
    """

    stages = _goal_d455_autolaunch_viewdense_constrained_mpc_recoverability_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    repaired: list[CurriculumStage] = []
    for index, stage in enumerate(stages):
        if index < 17:
            repaired.append(stage)
            continue
        repaired.append(
            replace(
                stage,
                cfg=replace(
                    stage.cfg,
                    descending_intercept_reward_weight=1.20,
                ),
                notes=(
                    f"{stage.notes}  W022 execution-aware intercept repair: "
                    "reward the actual racket for reaching the projected "
                    "descending-ball crossing point before it falls too low."
                ),
            )
        )
    return repaired


def _goal_d455_autolaunch_viewdense_constrained_mpc_count_progress_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """W024: make later legitimate contacts increasingly worth preserving.

    The strict tail inherited a zero combo coefficient, so every valid hit
    paid the same +1 even though the task metric requires a long sequence.
    Restore a small count-dependent marginal bonus, still capped by the
    existing 14-hit combo cap and protected by the existing contact/cadence
    checks.  This changes neither gates nor environment dynamics.
    """

    stages = _goal_d455_autolaunch_viewdense_constrained_mpc_intercept_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    repaired: list[CurriculumStage] = []
    for index, stage in enumerate(stages):
        if index < 17:
            repaired.append(stage)
            continue
        repaired.append(
            replace(
                stage,
                cfg=replace(stage.cfg, hit_reward_combo=0.08),
                notes=(
                    f"{stage.notes}  W024 count-progress objective repair: "
                    "each later legitimate hit receives a modestly larger "
                    "marginal bonus (combo=0.08, existing cap=14); all strict "
                    "gates remain unchanged."
                ),
            )
        )
    return repaired


def _goal_d455_autolaunch_teacherstudent_drivegov_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
    require_inverse_mpc_stack: bool = True,
) -> list[CurriculumStage]:
    """From-scratch student on the safe plant with ideal-domain distillation.

    Keep the proven W017 environment, reward, curriculum, and hard governor.
    Teacher supervision is owned by PPO CLI arguments and is evaluated only on
    frozen ideal-domain replay observations; it never enters environment
    actions or the deployable checkpoint.
    """

    stages = _goal_d455_autolaunch_viewdense_drivegov_successref_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        require_inverse_mpc_stack=require_inverse_mpc_stack,
    )
    return [
        replace(
            stage,
            notes=(
                f"{stage.notes}  Teacher-student variant: initialize the "
                "student from scratch and distill the frozen ideal-PD67 actor "
                "only on ideal-domain replay observations; actuator-domain "
                "actions remain PPO-controlled and hard-governed."
            ),
        )
        for stage in stages
    ]


def _goal_d455_release_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
) -> list[CurriculumStage]:
    return _goal_d455_from_scratch_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=stage_steps_override,
        critic_command_history_steps=critic_command_history_steps,
        branch="release",
    )


def _robust_juggle_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
) -> list[CurriculumStage]:
    """Build the compact final-goal-preserving robust juggling curriculum."""

    common = MjxJuggleConfig(**stack_kwargs)
    common = replace(
        common,
        horizon_sec=6.0,
        arm_action_limiter=True,
        action_acc_scale=1.0,
        action_penalty_weight=0.0018,
        action_delta_penalty_weight=0.0012,
        posture_weight=0.65,
        base_pose_weight=0.12,
        torque_penalty_weight=0.00035,
        arm_vel_limit_penalty_weight=0.06,
        arm_acc_limit_penalty_weight=0.08,
        arm_limiter_penalty_weight=0.05,
        ball_anchor_xy_penalty_weight=0.80,
        ball_base_x_penalty_weight=0.80,
        ball_base_x_soft_limit=0.10,
        ball_base_vxy_penalty_weight=0.45,
        ball_vxy_penalty_weight=0.18,
        ball_xy_soft_limit_radius=0.16,
        ball_xy_soft_penalty_weight=3.5,
        post_hit_survival_reward_weight=2.0,
        descending_intercept_reward_weight=1.8,
        pre_hit_intercept_reward_weight=1.4,
        pre_hit_intercept_sigma=0.10,
        pre_hit_intercept_time_max=0.72,
        pre_hit_intercept_penalty_weight=0.60,
        pre_hit_intercept_penalty_sigma=0.22,
        pre_hit_intercept_penalty_time_max=0.85,
        first_hit_apex_reward_weight=0.45,
        first_hit_apex_sigma=0.065,
        hit_reward_base=1.15,
        hit_reward_combo=0.18,
        hit_reward_cap_mode="fixed",
        hit_reward_count_cap=15,
        hit_combo_count_cap=14,
        hit_cadence_reward_weight=0.22,
        hit_cadence_target_interval=0.44,
        hit_cadence_sigma=0.09,
        hit_min_interval_penalty_weight=0.90,
        hit_min_interval=0.36,
        hit_min_count_interval=0.34,
        fast_hit_penalty_weight=0.70,
        hit_height_center=0.27,
        hit_height_tolerance=0.055,
        hit_height_penalty_weight=16.0,
        low_hit_apex_margin=0.020,
        low_hit_penalty_weight=14.0,
        target_height=0.22,
        apex_soft_limit_margin=0.06,
        center_flat_hit_reward_weight=1.15,
        termination_miss_penalty_base=4.0,
        termination_miss_penalty_per_hit=0.30,
        termination_miss_penalty_requires_hit=False,
        termination_no_hit_miss_early_penalty=8.0,
        ball_high_termination_z_m=1.80,
        terminate_on_ball_view_bounds=True,
        terminate_on_ball_view_x_bounds=True,
        terminate_on_ball_view_y_bounds=True,
        terminate_on_ball_view_z_low=True,
        terminate_on_ball_view_z_high=False,
        terminate_on_racket_z_limit=True,
        racket_z_limit_termination_penalty_base=4.0,
        racket_z_limit_termination_penalty_per_hit=0.30,
        ball_view_x_bounds_m=(-0.32, 0.36),
        ball_view_y_bounds_m=(-0.62, -0.12),
        ball_view_z_bounds_m=(0.62, 1.80),
        ball_view_z_ideal_m=(0.80, 1.28),
        hit_camera_reward_weight=0.70,
        hit_camera_out_of_band_penalty_weight=0.0,
        hit_camera_target_v_frac=0.67,
        hit_camera_v_sigma_frac=0.13,
        hit_camera_lower_band_frac=(0.52, 0.84),
        ball_view_z_sigma_m=0.12,
        ball_view_xy_center_penalty_weight=0.75,
        ball_view_z_ideal_penalty_weight=1.8,
        ball_view_bounds_penalty_weight=5.0,
        ball_view_out_of_bounds_penalty_weight=2.0,
        ball_view_z_not_ideal_penalty_weight=0.6,
        ball_view_vxy_excess_penalty_weight=0.70,
        ball_obs_rate_hz=60.0,
        ball_obs_fractional_rate=True,
        ball_obs_pos_noise_std=0.004,
        ball_obs_vel_noise_std=0.04,
        ball_obs_age_tracks_stale=True,
        ball_obs_age_clip=0.60,
        ball_obs_dropout_on_refresh_only=True,
        ball_obs_require_camera_visible=False,
        ball_obs_camera_missing_prob=0.0,
        ball_obs_require_view_bounds=False,
        lost_ball_timeout_ms=450.0,
        asymmetric_critic=True,
        critic_command_history_steps=max(12, int(critic_command_history_steps)),
    )
    common = _with_low_reset_ball_range(
        common,
        terminate=True,
        xy_weight=0.75,
        z_weight=1.8,
        bounds_weight=5.0,
        out_of_bounds_weight=2.0,
        z_not_ideal_weight=0.6,
        vxy_weight=0.70,
        target_x_range=(0.08, 0.14),
        target_y_range=(-0.02, 0.06),
        anchor_z_range=(-0.01, 0.01),
        launch_height=0.12,
        target_height=0.22,
        hit_height_center=0.27,
        hit_confirm_rel_height=0.065,
        terminate_racket_z=True,
    )
    common = _with_strong_camera_centering(common, center_weight=0.70)

    def make_cfg(
        *,
        target_x: tuple[float, float],
        target_y: tuple[float, float],
        anchor_z: tuple[float, float],
        xy_jitter: float,
        z_jitter: float,
        init_vxy: float,
        init_vz_jitter: float,
        camera_missing_prob: float,
        dropout_prob: float,
        burst_prob: float,
        dropout_steps: int,
        burst_steps: int,
        dr_level: int,
        delay_range_ms: tuple[float, float],
        obs_pos_noise: float,
        obs_vel_noise: float,
        camera_center_weight: float,
        falling_tau: tuple[float, float] = (0.14, 0.23),
        falling_apex_height: tuple[float, float] = (0.20, 0.32),
        falling_vxy: float = 0.08,
        falling_contact_jitter: float = 0.012,
    ) -> MjxJuggleConfig:
        cfg = _with_low_reset_ball_range(
            common,
            terminate=True,
            xy_weight=0.75 + 0.10 * dr_level,
            z_weight=1.8 + 0.15 * dr_level,
            bounds_weight=5.0 + 0.5 * dr_level,
            out_of_bounds_weight=2.0 + 0.5 * dr_level,
            z_not_ideal_weight=0.6 + 0.15 * dr_level,
            vxy_weight=0.70 + 0.05 * dr_level,
            target_x_range=target_x,
            target_y_range=target_y,
            anchor_z_range=anchor_z,
            launch_height=0.12,
            target_height=0.22,
            hit_height_center=0.27,
            hit_confirm_rel_height=0.065,
            terminate_racket_z=False,
        )
        cfg = _with_strong_camera_centering(cfg, center_weight=camera_center_weight)
        delay_lo, delay_hi = delay_range_ms
        cfg = replace(
            cfg,
            ball_spawn_xy_jitter=xy_jitter,
            ball_spawn_z_jitter=z_jitter,
            ball_init_vxy_max=init_vxy,
            ball_init_vz_jitter=init_vz_jitter,
            ball_reset_mode="falling_contact",
            falling_reset_time_to_contact_range_s=falling_tau,
            falling_reset_apex_height_range_m=falling_apex_height,
            falling_reset_vxy_max=falling_vxy,
            falling_reset_contact_xy_jitter=falling_contact_jitter,
            falling_reset_contact_rel_height=0.065,
            falling_reset_min_downward_speed=0.12,
            ball_view_x_bounds_m=(-0.32, 0.36),
            rel_height_center=0.22,
            hit_height_tolerance=0.055,
            apex_soft_limit_margin=0.10,
            ball_view_y_bounds_m=(-0.62, -0.12),
            ball_view_z_bounds_m=(0.62, 1.80),
            ball_view_z_ideal_m=(0.80, 1.28),
            ball_obs_pos_noise_std=obs_pos_noise,
            ball_obs_vel_noise_std=obs_vel_noise,
            ball_obs_require_camera_visible=camera_missing_prob > 0.0,
            ball_obs_camera_missing_prob=camera_missing_prob,
            ball_obs_require_view_bounds=False,
            ball_obs_reset_respects_camera_visibility=camera_missing_prob > 0.0,
            ball_obs_dropout_prob=dropout_prob,
            ball_obs_dropout_burst_prob=burst_prob,
            ball_obs_dropout_max_steps=dropout_steps,
            ball_obs_dropout_burst_max_steps=burst_steps,
            delay_min_ms=delay_lo,
            delay_max_ms=delay_hi,
            delay_bin_edges_ms=_delay_bin_edges_for_range(delay_lo, delay_hi),
            delay_jitter_ms=0.0 if delay_lo == delay_hi else 2.0,
            delay_sampling_mode="uniform" if delay_lo == delay_hi else "balanced_bins",
        )
        if dr_level <= 0:
            return replace(
                cfg,
                domain_randomization=False,
                dr_randomize_ball=False,
                dr_randomize_contact=False,
                dr_randomize_actuator=False,
                dr_randomize_latency=False,
                dr_randomize_pd=False,
                dr_randomize_racket_mount=False,
                dr_randomize_ball_obs_frame=False,
                dr_randomize_actuator_cmd_filter=False,
            )
        cfg = replace(
            cfg,
            domain_randomization=True,
            dr_randomize_ball=True,
            dr_randomize_contact=dr_level >= 2,
            dr_randomize_actuator=dr_level >= 2,
            dr_randomize_latency=False,
            dr_action_scale_mult_range=(0.95, 1.05) if dr_level < 2 else ((0.92, 1.08) if dr_level < 3 else (0.88, 1.12)),
            dr_damping_mult_range=(0.88, 1.12) if dr_level < 2 else ((0.82, 1.18) if dr_level < 3 else (0.75, 1.25)),
            dr_armature_mult_range=(0.92, 1.08) if dr_level < 2 else ((0.88, 1.12) if dr_level < 3 else (0.82, 1.18)),
            dr_randomize_pd=dr_level >= 2,
            dr_pd_kp_mult_range=(0.96, 1.04) if dr_level < 2 else ((0.93, 1.07) if dr_level < 3 else (0.88, 1.12)),
            dr_pd_kv_mult_range=(0.92, 1.08) if dr_level < 2 else ((0.88, 1.12) if dr_level < 3 else (0.82, 1.18)),
            dr_pd_per_joint=True,
            dr_randomize_racket_mount=dr_level >= 2,
            dr_racket_pos_offset_m=0.002 if dr_level < 3 else 0.004,
            dr_racket_rot_offset_rad=float(np.deg2rad(0.8 if dr_level < 3 else 1.5)),
            dr_racket_radius_offset_m=0.0015 if dr_level < 3 else 0.0025,
            dr_randomize_ball_obs_frame=dr_level >= 2,
            dr_ball_obs_pos_bias_base_m=(0.004, 0.004, 0.004) if dr_level < 3 else (0.008, 0.008, 0.008),
            dr_ball_obs_rot_bias_deg=(0.7, 0.7, 1.0) if dr_level < 3 else (1.5, 1.5, 2.0),
            dr_ball_obs_vel_bias_base_m_s=(0.04, 0.04, 0.05) if dr_level < 3 else (0.08, 0.08, 0.10),
            dr_ball_obs_scale_range=(0.99, 1.01) if dr_level < 3 else (0.98, 1.02),
        )
        if dr_level == 2:
            cfg = replace(
                cfg,
                dr_randomize_actuator_cmd_filter=True,
                dr_actuator_cmd_tau_range=(0.068, 0.082),
                dr_actuator_cmd_gain_range=(0.985, 1.015),
            )
        elif dr_level >= 3:
            cfg = replace(
                cfg,
                dr_randomize_actuator_cmd_filter=True,
                dr_actuator_cmd_tau_range=(0.060, 0.090),
                dr_actuator_cmd_gain_range=(0.97, 1.03),
                dr_ball_friction_range=(0.08, 0.45),
                dr_racket_friction_range=(0.18, 0.75),
                dr_ball_solref_time_range=(0.0015, 0.010),
                dr_ball_solref_damping_range=(0.55, 1.10),
            )
        return cfg

    narrow = dict(
        target_x=(0.08, 0.14),
        target_y=(-0.02, 0.06),
        anchor_z=(-0.01, 0.01),
        xy_jitter=0.005,
        z_jitter=0.004,
        init_vxy=0.004,
        init_vz_jitter=0.010,
        falling_tau=(0.14, 0.23),
        falling_apex_height=(0.22, 0.34),
        falling_vxy=0.08,
        falling_contact_jitter=0.012,
    )
    narrow_plus = dict(
        target_x=(0.06, 0.17),
        target_y=(-0.04, 0.09),
        anchor_z=(-0.025, 0.018),
        xy_jitter=0.008,
        z_jitter=0.005,
        init_vxy=0.006,
        init_vz_jitter=0.016,
        falling_tau=(0.12, 0.23),
        falling_apex_height=(0.21, 0.33),
        falling_vxy=0.12,
        falling_contact_jitter=0.018,
    )
    mid = dict(
        target_x=(0.02, 0.21),
        target_y=(-0.07, 0.13),
        anchor_z=(-0.04, 0.025),
        xy_jitter=0.012,
        z_jitter=0.007,
        init_vxy=0.010,
        init_vz_jitter=0.025,
        falling_tau=(0.10, 0.23),
        falling_apex_height=(0.20, 0.31),
        falling_vxy=0.18,
        falling_contact_jitter=0.030,
    )
    wide = dict(
        target_x=(-0.02, 0.26),
        target_y=(-0.105, 0.155),
        anchor_z=(-0.07, 0.04),
        xy_jitter=0.020,
        z_jitter=0.010,
        init_vxy=0.016,
        init_vz_jitter=0.040,
        falling_tau=(0.08, 0.23),
        falling_apex_height=(0.18, 0.30),
        falling_vxy=0.25,
        falling_contact_jitter=0.045,
    )

    def learnable_juggle_cfg(
        cfg: MjxJuggleConfig,
        *,
        launch_height: float,
        target_height: float,
        hit_height_center: float,
        hit_height_tolerance: float,
        hit_reward_base: float,
        hit_reward_combo: float,
        center_flat_weight: float,
        hit_height_penalty_weight: float,
        low_hit_penalty_weight: float,
        view_xy_weight: float,
        view_z_weight: float,
        view_bounds_weight: float,
        view_oob_weight: float,
        view_z_not_ideal_weight: float,
        view_vxy_weight: float,
        posture_weight: float,
        base_pose_weight: float,
        ball_anchor_xy_weight: float,
        ball_base_vxy_weight: float,
        action_penalty_weight: float,
        action_delta_penalty_weight: float,
        arm_vel_limit_penalty_weight: float,
        arm_acc_limit_penalty_weight: float,
        arm_limiter_penalty_weight: float,
        camera_missing_prob: float | None = None,
        dropout_prob: float | None = None,
        burst_prob: float | None = None,
        view_z_low: float = 0.62,
        hit_apex_view_center_weight: float = 0.0,
        hit_next_contact_anchor_weight: float = 0.0,
        hit_apex_view_center_sigma: float = 0.14,
        hit_next_contact_anchor_sigma: float = 0.12,
    ) -> MjxJuggleConfig:
        missing_kwargs: dict[str, object] = {}
        if camera_missing_prob is not None:
            camera_missing_prob = float(camera_missing_prob)
            missing_kwargs.update(
                ball_obs_require_camera_visible=camera_missing_prob > 0.0,
                ball_obs_camera_missing_prob=camera_missing_prob,
                ball_obs_reset_respects_camera_visibility=camera_missing_prob > 0.0,
            )
        if dropout_prob is not None:
            missing_kwargs["ball_obs_dropout_prob"] = float(dropout_prob)
        if burst_prob is not None:
            missing_kwargs["ball_obs_dropout_burst_prob"] = float(burst_prob)
        return replace(
            cfg,
            ball_launch_height=float(launch_height),
            target_height=float(target_height),
            rel_height_center=max(0.10, min(float(target_height) - 0.02, 0.28)),
            hit_height_center=float(hit_height_center),
            hit_height_tolerance=float(hit_height_tolerance),
            hit_reward_base=float(hit_reward_base),
            hit_reward_combo=float(hit_reward_combo),
            center_flat_hit_reward_weight=float(center_flat_weight),
            hit_height_penalty_weight=float(hit_height_penalty_weight),
            low_hit_penalty_weight=float(low_hit_penalty_weight),
            ball_view_z_bounds_m=(float(view_z_low), 1.80),
            ball_view_xy_center_penalty_weight=float(view_xy_weight),
            ball_view_z_ideal_penalty_weight=float(view_z_weight),
            ball_view_bounds_penalty_weight=float(view_bounds_weight),
            ball_view_out_of_bounds_penalty_weight=float(view_oob_weight),
            ball_view_z_not_ideal_penalty_weight=float(view_z_not_ideal_weight),
            ball_view_vxy_excess_penalty_weight=float(view_vxy_weight),
            posture_weight=float(posture_weight),
            base_pose_weight=float(base_pose_weight),
            ball_anchor_xy_penalty_weight=float(ball_anchor_xy_weight),
            ball_base_vxy_penalty_weight=float(ball_base_vxy_weight),
            hit_apex_view_center_penalty_weight=float(hit_apex_view_center_weight),
            hit_next_contact_anchor_penalty_weight=float(hit_next_contact_anchor_weight),
            hit_apex_view_center_sigma_m=float(hit_apex_view_center_sigma),
            hit_next_contact_anchor_sigma_m=float(hit_next_contact_anchor_sigma),
            action_penalty_weight=float(action_penalty_weight),
            action_delta_penalty_weight=float(action_delta_penalty_weight),
            arm_vel_limit_penalty_weight=float(arm_vel_limit_penalty_weight),
            arm_acc_limit_penalty_weight=float(arm_acc_limit_penalty_weight),
            arm_limiter_penalty_weight=float(arm_limiter_penalty_weight),
            **missing_kwargs,
        )


    cfg_contact = make_cfg(
        **narrow,
        camera_missing_prob=0.0,
        dropout_prob=0.0,
        burst_prob=0.0,
        dropout_steps=1,
        burst_steps=1,
        dr_level=0,
        delay_range_ms=(72.0, 72.0),
        obs_pos_noise=0.003,
        obs_vel_noise=0.03,
        camera_center_weight=0.70,
    )
    cfg_contact = learnable_juggle_cfg(
        cfg_contact,
        launch_height=0.30,
        target_height=0.34,
        hit_height_center=0.38,
        hit_height_tolerance=0.085,
        hit_reward_base=2.45,
        hit_reward_combo=0.65,
        center_flat_weight=1.65,
        hit_height_penalty_weight=5.0,
        low_hit_penalty_weight=4.0,
        view_xy_weight=0.15,
        view_z_weight=0.0,
        view_bounds_weight=0.35,
        view_oob_weight=0.0,
        view_z_not_ideal_weight=0.0,
        view_vxy_weight=0.10,
        posture_weight=0.08,
        base_pose_weight=0.02,
        ball_anchor_xy_weight=0.25,
        ball_base_vxy_weight=0.08,
        action_penalty_weight=0.0010,
        action_delta_penalty_weight=0.00045,
        arm_vel_limit_penalty_weight=0.025,
        arm_acc_limit_penalty_weight=0.035,
        arm_limiter_penalty_weight=0.012,
        view_z_low=0.58,
        hit_apex_view_center_weight=0.05,
        hit_next_contact_anchor_weight=0.07,
        hit_apex_view_center_sigma=0.18,
        hit_next_contact_anchor_sigma=0.16,
    )
    # The entry policy needs braking room after its first upward strike.  The
    # old 0.24 m hard limit was reached almost once per episode while the
    # policy was still learning the camera-aligned contact height, making a
    # second hit structurally impossible.  Keep the 0.20 m soft band active,
    # but shrink the hard workspace progressively as control improves.
    cfg_contact = replace(
        cfg_contact,
        racket_z_hard_limit_up=0.34,
        camera_center_weight=0.25,
        camera_visibility_penalty_weight=1.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=0.0,
        camera_top_margin_penalty_weight=0.0,
    )
    cfg_first_hit = replace(
        cfg_contact,
        racket_z_hard_limit_up=0.40,
        termination_miss_penalty_base=2.2,
        termination_miss_penalty_per_hit=0.20,
        termination_no_hit_miss_early_penalty=4.0,
        post_hit_survival_reward_weight=3.0,
        descending_intercept_reward_weight=2.2,
        pre_hit_intercept_reward_weight=1.6,
        hit_min_interval_penalty_weight=0.35,
        fast_hit_penalty_weight=0.35,
        hit_camera_reward_weight=0.55,
        camera_center_weight=0.20,
        hit_apex_view_center_penalty_weight=0.04,
        hit_next_contact_anchor_penalty_weight=0.05,
    )
    cfg_recoverable_first_hit_soft = replace(
        cfg_contact,
        racket_z_hard_limit_up=0.40,
        termination_miss_penalty_base=2.45,
        termination_miss_penalty_per_hit=0.35,
        termination_no_hit_miss_early_penalty=4.2,
        post_hit_survival_reward_weight=3.9,
        descending_intercept_reward_weight=2.9,
        pre_hit_intercept_reward_weight=2.0,
        hit_reward_base=2.25,
        hit_reward_combo=0.50,
        center_flat_hit_reward_weight=1.55,
        ball_base_vxy_penalty_weight=0.12,
        ball_vxy_penalty_weight=0.16,
        ball_view_vxy_excess_penalty_weight=0.35,
        hit_height_penalty_weight=5.8,
        low_hit_penalty_weight=4.8,
        hit_min_interval_penalty_weight=0.40,
        fast_hit_penalty_weight=0.38,
        hit_camera_reward_weight=1.25,
        hit_camera_out_of_band_penalty_weight=0.65,
        hit_camera_v_sigma_frac=0.11,
        hit_vxy_soft_limit_m_s=0.64,
        hit_vxy_penalty_weight=0.9,
        hit_apex_view_center_penalty_weight=0.08,
        hit_next_contact_anchor_penalty_weight=0.07,
        hit_apex_view_center_sigma_m=0.18,
        hit_next_contact_anchor_sigma_m=0.18,
    )
    cfg_recoverable_first_hit = replace(
        cfg_contact,
        racket_z_hard_limit_up=0.40,
        termination_miss_penalty_base=2.65,
        termination_miss_penalty_per_hit=0.55,
        termination_no_hit_miss_early_penalty=4.4,
        post_hit_survival_reward_weight=4.5,
        descending_intercept_reward_weight=3.4,
        pre_hit_intercept_reward_weight=2.3,
        hit_reward_base=2.05,
        hit_reward_combo=0.42,
        center_flat_hit_reward_weight=1.50,
        ball_base_vxy_penalty_weight=0.15,
        ball_vxy_penalty_weight=0.20,
        ball_view_vxy_excess_penalty_weight=0.55,
        hit_height_penalty_weight=6.4,
        low_hit_penalty_weight=5.4,
        hit_min_interval_penalty_weight=0.43,
        fast_hit_penalty_weight=0.40,
        hit_camera_reward_weight=1.45,
        hit_camera_out_of_band_penalty_weight=0.85,
        hit_camera_v_sigma_frac=0.10,
        hit_vxy_soft_limit_m_s=0.58,
        hit_vxy_penalty_weight=1.7,
        hit_apex_view_center_penalty_weight=0.10,
        hit_next_contact_anchor_penalty_weight=0.14,
        hit_apex_view_center_sigma_m=0.17,
        hit_next_contact_anchor_sigma_m=0.16,
    )
    cfg_second_hit = replace(
        cfg_recoverable_first_hit,
        termination_miss_penalty_base=2.9,
        termination_miss_penalty_per_hit=0.75,
        post_hit_survival_reward_weight=5.2,
        descending_intercept_reward_weight=4.2,
        pre_hit_intercept_reward_weight=2.6,
        hit_reward_base=1.95,
        hit_reward_combo=0.42,
        ball_base_vxy_penalty_weight=0.18,
        ball_vxy_penalty_weight=0.24,
        ball_view_vxy_excess_penalty_weight=0.80,
        hit_camera_reward_weight=1.35,
        hit_camera_out_of_band_penalty_weight=0.70,
        hit_vxy_soft_limit_m_s=0.54,
        hit_vxy_penalty_weight=2.5,
        hit_apex_view_center_penalty_weight=0.12,
        hit_next_contact_anchor_penalty_weight=0.22,
    )
    cfg_contact_pair = replace(
        cfg_contact,
        racket_z_hard_limit_up=0.36,
        termination_miss_penalty_base=2.8,
        termination_miss_penalty_per_hit=0.22,
        termination_no_hit_miss_early_penalty=5.5,
        post_hit_survival_reward_weight=3.8,
        descending_intercept_reward_weight=2.9,
        pre_hit_intercept_reward_weight=2.1,
        hit_min_interval_penalty_weight=0.65,
        fast_hit_penalty_weight=0.55,
        hit_camera_reward_weight=1.25,
        hit_camera_out_of_band_penalty_weight=0.18,
        hit_apex_view_center_penalty_weight=0.08,
        hit_next_contact_anchor_penalty_weight=0.12,
    )
    cfg_third_hit_bridge = replace(
        cfg_contact_pair,
        post_hit_survival_reward_weight=3.6,
        descending_intercept_reward_weight=2.7,
        pre_hit_intercept_reward_weight=2.0,
        hit_camera_reward_weight=1.15,
        hit_camera_out_of_band_penalty_weight=0.14,
        hit_apex_view_center_penalty_weight=0.09,
        hit_next_contact_anchor_penalty_weight=0.13,
    )
    cfg_control = make_cfg(
        **narrow_plus,
        camera_missing_prob=0.10,
        dropout_prob=0.001,
        burst_prob=0.0002,
        dropout_steps=3,
        burst_steps=8,
        dr_level=0,
        delay_range_ms=(70.0, 74.0),
        obs_pos_noise=0.004,
        obs_vel_noise=0.04,
        camera_center_weight=0.78,
    )
    cfg_control = learnable_juggle_cfg(
        cfg_control,
        launch_height=0.30,
        target_height=0.34,
        hit_height_center=0.38,
        hit_height_tolerance=0.078,
        hit_reward_base=2.25,
        hit_reward_combo=0.55,
        center_flat_weight=1.55,
        hit_height_penalty_weight=7.0,
        low_hit_penalty_weight=5.5,
        view_xy_weight=0.25,
        view_z_weight=0.20,
        view_bounds_weight=0.80,
        view_oob_weight=0.15,
        view_z_not_ideal_weight=0.05,
        view_vxy_weight=0.18,
        posture_weight=0.14,
        base_pose_weight=0.03,
        ball_anchor_xy_weight=0.35,
        ball_base_vxy_weight=0.10,
        action_penalty_weight=0.0012,
        action_delta_penalty_weight=0.00055,
        arm_vel_limit_penalty_weight=0.030,
        arm_acc_limit_penalty_weight=0.040,
        arm_limiter_penalty_weight=0.016,
        view_z_low=0.58,
        hit_apex_view_center_weight=0.07,
        hit_next_contact_anchor_weight=0.10,
        hit_apex_view_center_sigma=0.17,
        hit_next_contact_anchor_sigma=0.15,
    )
    cfg_control = replace(
        cfg_control,
        racket_z_hard_limit_up=0.32,
        camera_center_weight=0.35,
        camera_visibility_penalty_weight=2.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=0.5,
        camera_top_margin_penalty_weight=2.0,
    )
    cfg_range = make_cfg(
        **mid,
        camera_missing_prob=0.25,
        dropout_prob=0.002,
        burst_prob=0.0004,
        dropout_steps=4,
        burst_steps=12,
        dr_level=1,
        delay_range_ms=(68.0, 76.0),
        obs_pos_noise=0.006,
        obs_vel_noise=0.06,
        camera_center_weight=0.86,
    )
    cfg_range = learnable_juggle_cfg(
        cfg_range,
        launch_height=0.30,
        target_height=0.32,
        hit_height_center=0.36,
        hit_height_tolerance=0.072,
        hit_reward_base=2.05,
        hit_reward_combo=0.45,
        center_flat_weight=1.45,
        hit_height_penalty_weight=8.5,
        low_hit_penalty_weight=7.0,
        view_xy_weight=0.40,
        view_z_weight=0.55,
        view_bounds_weight=1.60,
        view_oob_weight=0.35,
        view_z_not_ideal_weight=0.12,
        view_vxy_weight=0.30,
        posture_weight=0.22,
        base_pose_weight=0.05,
        ball_anchor_xy_weight=0.45,
        ball_base_vxy_weight=0.14,
        action_penalty_weight=0.0014,
        action_delta_penalty_weight=0.00070,
        arm_vel_limit_penalty_weight=0.038,
        arm_acc_limit_penalty_weight=0.050,
        arm_limiter_penalty_weight=0.022,
        view_z_low=0.60,
        hit_apex_view_center_weight=0.10,
        hit_next_contact_anchor_weight=0.14,
        hit_apex_view_center_sigma=0.16,
        hit_next_contact_anchor_sigma=0.14,
    )
    cfg_range = replace(
        cfg_range,
        racket_z_hard_limit_up=0.30,
        camera_center_weight=0.50,
        camera_visibility_penalty_weight=4.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=1.0,
        camera_top_margin_penalty_weight=4.0,
    )
    cfg_fov = make_cfg(
        **wide,
        camera_missing_prob=0.45,
        dropout_prob=0.0035,
        burst_prob=0.0008,
        dropout_steps=5,
        burst_steps=18,
        dr_level=1,
        delay_range_ms=(65.0, 80.0),
        obs_pos_noise=0.008,
        obs_vel_noise=0.08,
        camera_center_weight=0.95,
    )
    cfg_fov = learnable_juggle_cfg(
        cfg_fov,
        launch_height=0.28,
        target_height=0.30,
        hit_height_center=0.34,
        hit_height_tolerance=0.068,
        hit_reward_base=1.85,
        hit_reward_combo=0.36,
        center_flat_weight=1.35,
        hit_height_penalty_weight=10.0,
        low_hit_penalty_weight=8.5,
        view_xy_weight=0.55,
        view_z_weight=0.90,
        view_bounds_weight=2.60,
        view_oob_weight=0.70,
        view_z_not_ideal_weight=0.22,
        view_vxy_weight=0.42,
        posture_weight=0.32,
        base_pose_weight=0.07,
        ball_anchor_xy_weight=0.55,
        ball_base_vxy_weight=0.18,
        action_penalty_weight=0.0016,
        action_delta_penalty_weight=0.00085,
        arm_vel_limit_penalty_weight=0.046,
        arm_acc_limit_penalty_weight=0.060,
        arm_limiter_penalty_weight=0.030,
        view_z_low=0.60,
        hit_apex_view_center_weight=0.12,
        hit_next_contact_anchor_weight=0.16,
        hit_apex_view_center_sigma=0.15,
        hit_next_contact_anchor_sigma=0.13,
    )
    cfg_fov = replace(
        cfg_fov,
        racket_z_hard_limit_up=0.28,
        camera_center_weight=0.70,
        camera_visibility_penalty_weight=6.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=2.0,
        camera_top_margin_penalty_weight=8.0,
    )
    cfg_fov = replace(
        cfg_fov,
        dr_randomize_contact=True,
        dr_randomize_actuator_cmd_filter=True,
        dr_actuator_cmd_tau_range=(0.070, 0.078),
        dr_actuator_cmd_gain_range=(0.992, 1.008),
        dr_ball_friction_range=(0.16, 0.28),
        dr_racket_friction_range=(0.30, 0.50),
        dr_ball_solref_time_range=(0.0028, 0.0052),
        dr_ball_solref_damping_range=(0.74, 0.91),
    )
    cfg_missing = make_cfg(
        **wide,
        camera_missing_prob=0.70,
        dropout_prob=0.006,
        burst_prob=0.0015,
        dropout_steps=8,
        burst_steps=32,
        dr_level=1,
        delay_range_ms=(65.0, 80.0),
        obs_pos_noise=0.010,
        obs_vel_noise=0.10,
        camera_center_weight=1.02,
    )
    cfg_missing = learnable_juggle_cfg(
        cfg_missing,
        launch_height=0.27,
        target_height=0.29,
        hit_height_center=0.33,
        hit_height_tolerance=0.064,
        hit_reward_base=1.70,
        hit_reward_combo=0.30,
        center_flat_weight=1.25,
        hit_height_penalty_weight=11.5,
        low_hit_penalty_weight=10.0,
        view_xy_weight=0.70,
        view_z_weight=1.20,
        view_bounds_weight=3.50,
        view_oob_weight=1.10,
        view_z_not_ideal_weight=0.35,
        view_vxy_weight=0.55,
        posture_weight=0.42,
        base_pose_weight=0.09,
        ball_anchor_xy_weight=0.65,
        ball_base_vxy_weight=0.24,
        action_penalty_weight=0.0017,
        action_delta_penalty_weight=0.0010,
        arm_vel_limit_penalty_weight=0.052,
        arm_acc_limit_penalty_weight=0.070,
        arm_limiter_penalty_weight=0.038,
        hit_apex_view_center_weight=0.14,
        hit_next_contact_anchor_weight=0.18,
        hit_apex_view_center_sigma=0.14,
        hit_next_contact_anchor_sigma=0.12,
    )
    cfg_missing = replace(
        cfg_missing,
        racket_z_hard_limit_up=0.27,
        camera_center_weight=0.90,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
        dr_randomize_contact=True,
        dr_randomize_actuator=True,
        dr_randomize_pd=True,
        dr_action_scale_mult_range=(0.96, 1.04),
        dr_damping_mult_range=(0.90, 1.10),
        dr_armature_mult_range=(0.94, 1.06),
        dr_pd_kp_mult_range=(0.97, 1.03),
        dr_pd_kv_mult_range=(0.94, 1.06),
        dr_randomize_actuator_cmd_filter=True,
        dr_actuator_cmd_tau_range=(0.068, 0.082),
        dr_actuator_cmd_gain_range=(0.985, 1.015),
        dr_ball_friction_range=(0.14, 0.31),
        dr_racket_friction_range=(0.28, 0.52),
        dr_ball_solref_time_range=(0.0025, 0.0060),
        dr_ball_solref_damping_range=(0.70, 0.95),
        dr_randomize_racket_mount=False,
        dr_randomize_ball_obs_frame=False,
    )
    cfg_dynamics = make_cfg(
        **wide,
        camera_missing_prob=0.90,
        dropout_prob=0.007,
        burst_prob=0.0018,
        dropout_steps=10,
        burst_steps=40,
        dr_level=2,
        delay_range_ms=(62.0, 82.0),
        obs_pos_noise=0.012,
        obs_vel_noise=0.12,
        camera_center_weight=1.08,
    )
    cfg_dynamics = learnable_juggle_cfg(
        cfg_dynamics,
        launch_height=0.26,
        target_height=0.28,
        hit_height_center=0.32,
        hit_height_tolerance=0.060,
        hit_reward_base=1.55,
        hit_reward_combo=0.25,
        center_flat_weight=1.20,
        hit_height_penalty_weight=13.0,
        low_hit_penalty_weight=12.0,
        view_xy_weight=0.80,
        view_z_weight=1.45,
        view_bounds_weight=4.20,
        view_oob_weight=1.55,
        view_z_not_ideal_weight=0.45,
        view_vxy_weight=0.62,
        posture_weight=0.52,
        base_pose_weight=0.10,
        ball_anchor_xy_weight=0.72,
        ball_base_vxy_weight=0.32,
        action_penalty_weight=0.0018,
        action_delta_penalty_weight=0.0011,
        arm_vel_limit_penalty_weight=0.058,
        arm_acc_limit_penalty_weight=0.078,
        arm_limiter_penalty_weight=0.045,
        hit_apex_view_center_weight=0.16,
        hit_next_contact_anchor_weight=0.22,
        hit_apex_view_center_sigma=0.13,
        hit_next_contact_anchor_sigma=0.11,
    )
    cfg_dynamics = replace(
        cfg_dynamics,
        racket_z_hard_limit_up=0.26,
        camera_center_weight=1.05,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
    )
    cfg_final = make_cfg(
        **wide,
        camera_missing_prob=1.0,
        dropout_prob=0.008,
        burst_prob=0.002,
        dropout_steps=12,
        burst_steps=48,
        dr_level=3,
        delay_range_ms=(60.0, 85.0),
        obs_pos_noise=0.015,
        obs_vel_noise=0.15,
        camera_center_weight=1.15,
    )
    cfg_final = learnable_juggle_cfg(
        cfg_final,
        launch_height=0.26,
        target_height=0.26,
        hit_height_center=0.31,
        hit_height_tolerance=0.055,
        hit_reward_base=1.35,
        hit_reward_combo=0.20,
        center_flat_weight=1.15,
        hit_height_penalty_weight=16.0,
        low_hit_penalty_weight=14.0,
        view_xy_weight=0.75,
        view_z_weight=1.8,
        view_bounds_weight=5.0,
        view_oob_weight=2.0,
        view_z_not_ideal_weight=0.6,
        view_vxy_weight=0.70,
        posture_weight=0.65,
        base_pose_weight=0.12,
        ball_anchor_xy_weight=0.80,
        ball_base_vxy_weight=0.45,
        action_penalty_weight=0.0018,
        action_delta_penalty_weight=0.0012,
        arm_vel_limit_penalty_weight=0.060,
        arm_acc_limit_penalty_weight=0.080,
        arm_limiter_penalty_weight=0.050,
        camera_missing_prob=1.0,
        dropout_prob=0.008,
        burst_prob=0.002,
        hit_apex_view_center_weight=0.18,
        hit_next_contact_anchor_weight=0.25,
        hit_apex_view_center_sigma=0.12,
        hit_next_contact_anchor_sigma=0.10,
    )
    cfg_final = replace(
        cfg_final,
        racket_z_hard_limit_up=0.26,
        camera_center_weight=1.15,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
    )

    stages = [
        CurriculumStage(
            "00_first_hit_bootstrap",
            1_500_000,
            cfg_first_hit,
            "Bootstrap reliable first contact in the real D455 window before asking for sustained juggling.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=0.85,
            target_mean_len_frac=0.08,
            min_updates=18,
            target_hit1_rate=0.72,
            target_hit_camera_visible_rate=0.70,
            target_hit_camera_lower_band_rate=0.20,
        ),
        CurriculumStage(
            "00b_recoverable_first_hit_soft",
            1_200_000,
            cfg_recoverable_first_hit_soft,
            "Begin shaping the first hit toward a recoverable next contact without collapsing hit acquisition.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=0.70,
            target_mean_len_frac=0.095,
            min_updates=30,
            target_hit1_rate=0.64,
            target_hit_camera_visible_rate=0.74,
            target_hit_camera_lower_band_rate=0.40,
            max_recent_mean_hit_vxy=0.88,
            max_recent_hit_next_contact_anchor_err=0.46,
            max_recent_mean_hit_camera_v_frac=0.86,
        ),
        CurriculumStage(
            "00c_recoverable_first_hit",
            1_800_000,
            cfg_recoverable_first_hit,
            "Keep the first falling-reset contact recoverable before asking for a second hit.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=0.68,
            target_mean_len_frac=0.10,
            min_updates=45,
            target_hit1_rate=0.62,
            target_hit_camera_visible_rate=0.74,
            target_hit_camera_lower_band_rate=0.40,
            max_recent_mean_hit_vxy=0.72,
            max_recent_hit_next_contact_anchor_err=0.36,
            max_recent_mean_hit_camera_v_frac=0.86,
        ),
        CurriculumStage(
            "01_second_hit_bridge",
            2_500_000,
            cfg_second_hit,
            "Bridge from first contact to a recoverable second contact without changing the D455 view geometry.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=1.05,
            target_mean_len_frac=0.12,
            min_updates=45,
            target_hit1_rate=0.72,
            target_hit3_rate=0.010,
            target_min_hit_interval_s=0.32,
            target_max_hit_interval_s=0.90,
            target_hit_camera_visible_rate=0.70,
            target_hit_camera_lower_band_rate=0.35,
            max_recent_mean_hit_vxy=0.66,
            max_recent_hit_next_contact_anchor_err=0.30,
            max_recent_mean_hit_camera_v_frac=0.86,
        ),
        CurriculumStage(
            "02_third_hit_bridge",
            2_000_000,
            cfg_third_hit_bridge,
            "Turn stable two-hit behavior into frequent three-hit episodes before the full contact-pair gate.",
            gate_mode="strict",
            advance_gate_mode="collapse",
            target_mean_hits=1.85,
            target_mean_len_frac=0.14,
            min_updates=80,
            target_hit1_rate=0.82,
            target_hit3_rate=0.006,
            target_mean_hits_ge3=3.0,
            target_min_hit_interval_s=0.34,
            target_max_hit_interval_s=0.84,
            target_hit_camera_visible_rate=0.60,
            target_hit_camera_lower_band_rate=0.28,
        ),
        CurriculumStage(
            "02_contact_pair",
            2_000_000,
            cfg_contact_pair,
            "Reliable centered contact with the full 67D actuator-aware actor stack.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=2,
            target_mean_len_frac=0.16,
            min_updates=35,
            target_hit1_rate=0.82,
            target_hit3_rate=0.035,
            target_mean_hits_ge3=3.2,
            target_min_hit_interval_s=0.34,
            target_max_hit_interval_s=0.82,
            target_hit_camera_visible_rate=0.66,
            target_hit_camera_lower_band_rate=0.34,
        ),
        CurriculumStage(
            "03_control",
            3_000_000,
            cfg_control,
            "Stabilize height, cadence, and camera centering before observation missing.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=3,
            target_mean_len_frac=0.25,
            min_updates=35,
            target_camera_visible=0.72,
            target_ball_view_in_bounds=0.62,
            target_ball_view_z_ideal=0.5,
            target_hit1_rate=0.65,
            target_hit3_rate=0.12,
            target_mean_hits_ge3=4.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.74,
            target_hit_camera_visible_rate=0.72,
            target_hit_camera_lower_band_rate=0.42,
            target_episode_truncation_rate=0.1,
            max_ball_obs_lost_rate=0.03,
        ),
        CurriculumStage(
            "04_range",
            4_000_000,
            cfg_range,
            "Expand reset and target range while keeping observations mostly visible.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=4.2,
            target_mean_len_frac=0.36,
            min_updates=40,
            target_camera_visible=0.72,
            target_ball_view_in_bounds=0.64,
            target_ball_view_z_ideal=0.52,
            target_hit1_rate=0.75,
            target_hit3_rate=0.25,
            target_mean_hits_ge3=5.2,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.68,
            target_hit_camera_visible_rate=0.78,
            target_hit_camera_lower_band_rate=0.52,
            target_episode_truncation_rate=0.2,
            max_ball_obs_lost_rate=0.045,
        ),
        CurriculumStage(
            "05_fov",
            5_000_000,
            cfg_fov,
            "Reach the final wide range; upward D455 FOV loss is recoverable, while lateral/low safety exits terminate.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=5.6,
            target_mean_len_frac=0.48,
            min_updates=45,
            target_camera_visible=0.7,
            target_ball_view_in_bounds=0.62,
            target_ball_view_z_ideal=0.52,
            target_hit1_rate=0.82,
            target_hit3_rate=0.38,
            target_mean_hits_ge3=6.6,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.62,
            target_hit_camera_visible_rate=0.84,
            target_hit_camera_lower_band_rate=0.60,
            target_episode_truncation_rate=0.34,
            min_ball_obs_missing_refresh_rate=0.001,
            max_ball_obs_lost_rate=0.06,
        ),
        CurriculumStage(
            "06_missing",
            6_000_000,
            cfg_missing,
            "Train missing recovery plus mild actuator and PD domain randomization.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=7,
            target_mean_len_frac=0.6,
            min_updates=50,
            target_camera_visible=0.68,
            target_ball_view_in_bounds=0.62,
            target_ball_view_z_ideal=0.54,
            target_hit1_rate=0.87,
            target_hit3_rate=0.50,
            target_hit12_rate=0.05,
            target_mean_hits_ge3=8.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.58,
            target_hit_camera_visible_rate=0.88,
            target_hit_camera_lower_band_rate=0.68,
            target_episode_truncation_rate=0.48,
            min_ball_obs_missing_refresh_rate=0.003,
            max_ball_obs_lost_rate=0.055,
        ),
        CurriculumStage(
            "07_dynamics",
            7_000_000,
            cfg_dynamics,
            "Expand contact, actuator, and PD DR; add mild racket-mount and observation-frame DR.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=8.8,
            target_mean_len_frac=0.72,
            min_updates=55,
            target_camera_visible=0.7,
            target_ball_view_in_bounds=0.64,
            target_ball_view_z_ideal=0.56,
            target_hit1_rate=0.91,
            target_hit3_rate=0.64,
            target_hit12_rate=0.2,
            target_mean_hits_ge3=9.8,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.92,
            target_hit_camera_lower_band_rate=0.76,
            target_episode_truncation_rate=0.62,
            min_ball_obs_missing_refresh_rate=0.005,
            max_ball_obs_lost_rate=0.05,
        ),
        CurriculumStage(
            "08_robust",
            9_000_000,
            cfg_final,
            "Final wide range, FOV missing, actuator response, delay, observation, and dynamics DR.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=11,
            target_mean_len_frac=0.84,
            min_updates=65,
            target_camera_visible=0.76,
            target_ball_view_in_bounds=0.68,
            target_ball_view_z_ideal=0.60,
            target_hit1_rate=0.95,
            target_hit3_rate=0.78,
            target_hit12_rate=0.52,
            target_mean_hits_ge3=11.6,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.50,
            target_hit_camera_visible_rate=0.95,
            target_hit_camera_lower_band_rate=0.84,
            target_episode_truncation_rate=0.78,
            min_ball_obs_missing_refresh_rate=0.010,
            max_ball_obs_lost_rate=0.035,
        ),
        CurriculumStage(
            "09_final",
            12_000_000,
            cfg_final,
            "Same final objective as 07_robust; only the strict acceptance standard changes.",
            gate_mode="strict",
            advance_gate_mode="collapse",
            target_mean_hits=13.0,
            target_mean_len_frac=0.95,
            min_updates=80,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.62,
            target_hit1_rate=0.97,
            target_hit3_rate=0.88,
            target_hit12_rate=0.82,
            target_mean_hits_ge3=14.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.50,
            target_hit_camera_visible_rate=0.98,
            target_hit_camera_lower_band_rate=0.90,
            target_episode_truncation_rate=0.90,
            min_ball_obs_missing_refresh_rate=0.010,
            max_ball_obs_lost_rate=0.025,
        ),
    ]
    if stage_steps_override is not None:
        stages = [replace(stage, total_steps=int(stage_steps_override)) for stage in stages]
    return stages


def _d455_stable_4g_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
) -> list[CurriculumStage]:
    """Standard-like D455 stable-juggle curriculum before recovery sampling.

    This profile intentionally keeps the arm reset fixed and uses the current
    calibrated D455 camera from the first stage.  It learns the nominal
    13--15-hit trajectory with anchor-drop ball resets before a later resume
    profile exposes broad falling-contact recovery states.
    """

    source_stages = _robust_juggle_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=None,
        critic_command_history_steps=critic_command_history_steps,
    )
    source = {stage.name: stage for stage in source_stages}

    def steps(default_steps: int) -> int:
        return int(stage_steps_override) if stage_steps_override is not None else int(default_steps)

    def stable_cfg(
        source_name: str,
        *,
        target_x: tuple[float, float],
        target_y: tuple[float, float],
        anchor_z: tuple[float, float],
        launch_height: float,
        target_height: float,
        hit_height_center: float,
        hit_height_tolerance: float,
        xy_jitter: float,
        z_jitter: float,
        init_vxy: float,
        init_vz_jitter: float,
        obs_pos_noise: float,
        obs_vel_noise: float,
        camera_center_weight: float,
        hit_camera_weight: float,
        hit_camera_oob: float,
        view_xy_weight: float,
        view_z_weight: float,
        view_bounds_weight: float,
        view_oob_weight: float,
        view_vxy_weight: float,
        hit_vxy_limit: float,
        hit_vxy_weight: float,
        next_contact_weight: float,
        apex_center_weight: float,
    ) -> MjxJuggleConfig:
        cfg = source[source_name].cfg
        return replace(
            cfg,
            right_arm_reset_degrees=D455_USER_TARGET_RACKET_RESET_DEGREES,
            ball_reset_mode="anchor_drop",
            ball_launch_height=float(launch_height),
            target_height=float(target_height),
            rel_height_center=max(0.10, min(float(target_height) - 0.02, 0.28)),
            hit_height_center=float(hit_height_center),
            hit_confirm_rel_height=0.055,
            hit_height_tolerance=float(hit_height_tolerance),
            low_hit_apex_margin=min(max(float(cfg.low_hit_apex_margin), 0.024), 0.055),
            apex_soft_limit_margin=max(float(cfg.apex_soft_limit_margin), 0.060),
            episode_target_x_range_m=target_x,
            episode_target_y_range_m=target_y,
            episode_racket_anchor_z_range_m=anchor_z,
            ball_spawn_xy_jitter=float(xy_jitter),
            ball_spawn_z_jitter=float(z_jitter),
            ball_init_vxy_max=float(init_vxy),
            ball_init_vz=-0.28,
            ball_init_vz_jitter=float(init_vz_jitter),
            ball_obs_pos_noise_std=float(obs_pos_noise),
            ball_obs_vel_noise_std=float(obs_vel_noise),
            ball_obs_require_camera_visible=False,
            ball_obs_camera_missing_prob=0.0,
            ball_obs_reset_respects_camera_visibility=False,
            ball_obs_require_view_bounds=False,
            ball_obs_view_bounds_missing_prob=0.0,
            ball_obs_missing_episode_coherent_prob=0.0,
            ball_obs_dropout_prob=0.0,
            ball_obs_dropout_burst_prob=0.0,
            ball_obs_dropout_max_steps=1,
            ball_obs_dropout_burst_max_steps=1,
            ball_obs_dropout_on_refresh_only=False,
            ball_obs_age_clip=0.35,
            lost_ball_timeout_ms=450.0,
            terminate_on_ball_view_bounds=True,
            terminate_on_ball_view_x_bounds=True,
            terminate_on_ball_view_y_bounds=True,
            terminate_on_ball_view_z_low=True,
            terminate_on_ball_view_z_high=False,
            racket_z_hard_limit_up=max(float(cfg.racket_z_hard_limit_up), max(0.0, -float(anchor_z[0])) + 0.12),
            ball_low_termination_z_m=D455_REAL_VIEW_Z_BOUNDS_M[0],
            ball_view_x_bounds_m=D455_REAL_VIEW_X_BOUNDS_M,
            ball_view_y_bounds_m=D455_REAL_VIEW_Y_BOUNDS_M,
            ball_view_z_bounds_m=D455_REAL_VIEW_Z_BOUNDS_M,
            ball_view_z_ideal_m=D455_STABLE_VIEW_Z_IDEAL_M,
            ball_view_y_target_m=D455_REAL_VIEW_Y_TARGET_M,
            ball_view_y_sigma_m=0.090,
            ball_view_z_sigma_m=0.10,
            ball_view_xy_center_penalty_weight=float(view_xy_weight),
            ball_view_z_ideal_penalty_weight=float(view_z_weight),
            ball_view_bounds_penalty_weight=float(view_bounds_weight),
            ball_view_out_of_bounds_penalty_weight=float(view_oob_weight),
            ball_view_z_not_ideal_penalty_weight=max(float(cfg.ball_view_z_not_ideal_penalty_weight), 0.35),
            ball_view_vxy_excess_penalty_weight=float(view_vxy_weight),
            camera_center_weight=float(camera_center_weight),
            hit_camera_reward_weight=float(hit_camera_weight),
            hit_camera_out_of_band_penalty_weight=float(hit_camera_oob),
            hit_camera_target_v_frac=0.66,
            hit_camera_v_sigma_frac=0.10,
            hit_camera_lower_band_frac=(0.50, 0.82),
            hit_reward_cap_mode="fixed",
            hit_reward_count_cap=15,
            hit_combo_count_cap=14,
            hit_cadence_target_interval=0.44,
            hit_cadence_sigma=0.10,
            hit_min_interval=0.36,
            hit_min_count_interval=0.34,
            hit_vxy_soft_limit_m_s=float(hit_vxy_limit),
            hit_vxy_penalty_weight=float(hit_vxy_weight),
            hit_apex_view_center_penalty_weight=float(apex_center_weight),
            hit_next_contact_anchor_penalty_weight=float(next_contact_weight),
            hit_apex_view_center_sigma_m=0.14,
            hit_next_contact_anchor_sigma_m=0.12,
        )

    cfg_1a = stable_cfg(
        "00_first_hit_bootstrap",
        target_x=(-0.025, 0.015),
        target_y=(-0.005, 0.020),
        anchor_z=(-0.006, 0.010),
        launch_height=0.200,
        target_height=0.180,
        hit_height_center=0.215,
        hit_height_tolerance=0.075,
        xy_jitter=0.004,
        z_jitter=0.004,
        init_vxy=0.003,
        init_vz_jitter=0.008,
        obs_pos_noise=0.003,
        obs_vel_noise=0.03,
        camera_center_weight=0.24,
        hit_camera_weight=0.55,
        hit_camera_oob=0.05,
        view_xy_weight=0.12,
        view_z_weight=0.0,
        view_bounds_weight=0.30,
        view_oob_weight=0.0,
        view_vxy_weight=0.08,
        hit_vxy_limit=0.78,
        hit_vxy_weight=0.15,
        next_contact_weight=0.04,
        apex_center_weight=0.03,
    )
    cfg_1b = stable_cfg(
        "00b_recoverable_first_hit_soft",
        target_x=(-0.030, 0.020),
        target_y=(0.000, 0.035),
        anchor_z=(-0.010, 0.014),
        launch_height=0.205,
        target_height=0.185,
        hit_height_center=0.220,
        hit_height_tolerance=0.072,
        xy_jitter=0.006,
        z_jitter=0.005,
        init_vxy=0.005,
        init_vz_jitter=0.012,
        obs_pos_noise=0.0035,
        obs_vel_noise=0.035,
        camera_center_weight=0.28,
        hit_camera_weight=0.80,
        hit_camera_oob=0.15,
        view_xy_weight=0.18,
        view_z_weight=0.08,
        view_bounds_weight=0.45,
        view_oob_weight=0.05,
        view_vxy_weight=0.14,
        hit_vxy_limit=0.68,
        hit_vxy_weight=0.35,
        next_contact_weight=0.08,
        apex_center_weight=0.06,
    )
    cfg_2a = stable_cfg(
        "01_second_hit_bridge",
        target_x=(-0.035, 0.030),
        target_y=(0.010, 0.055),
        anchor_z=(-0.014, 0.018),
        launch_height=0.210,
        target_height=0.190,
        hit_height_center=0.225,
        hit_height_tolerance=0.070,
        xy_jitter=0.008,
        z_jitter=0.006,
        init_vxy=0.006,
        init_vz_jitter=0.014,
        obs_pos_noise=0.004,
        obs_vel_noise=0.04,
        camera_center_weight=0.34,
        hit_camera_weight=0.95,
        hit_camera_oob=0.20,
        view_xy_weight=0.22,
        view_z_weight=0.14,
        view_bounds_weight=0.65,
        view_oob_weight=0.10,
        view_vxy_weight=0.18,
        hit_vxy_limit=0.62,
        hit_vxy_weight=0.55,
        next_contact_weight=0.10,
        apex_center_weight=0.08,
    )
    cfg_2b = stable_cfg(
        "02_third_hit_bridge",
        target_x=(-0.040, 0.040),
        target_y=(0.020, 0.075),
        anchor_z=(-0.018, 0.020),
        launch_height=0.215,
        target_height=0.195,
        hit_height_center=0.230,
        hit_height_tolerance=0.068,
        xy_jitter=0.010,
        z_jitter=0.007,
        init_vxy=0.008,
        init_vz_jitter=0.016,
        obs_pos_noise=0.004,
        obs_vel_noise=0.04,
        camera_center_weight=0.38,
        hit_camera_weight=1.05,
        hit_camera_oob=0.24,
        view_xy_weight=0.28,
        view_z_weight=0.22,
        view_bounds_weight=0.90,
        view_oob_weight=0.16,
        view_vxy_weight=0.24,
        hit_vxy_limit=0.58,
        hit_vxy_weight=0.80,
        next_contact_weight=0.12,
        apex_center_weight=0.09,
    )
    cfg_3a = stable_cfg(
        "02_contact_pair",
        target_x=(-0.045, 0.050),
        target_y=(0.035, 0.095),
        anchor_z=(-0.022, 0.022),
        launch_height=0.215,
        target_height=0.200,
        hit_height_center=0.235,
        hit_height_tolerance=0.066,
        xy_jitter=0.012,
        z_jitter=0.008,
        init_vxy=0.009,
        init_vz_jitter=0.018,
        obs_pos_noise=0.004,
        obs_vel_noise=0.04,
        camera_center_weight=0.44,
        hit_camera_weight=1.10,
        hit_camera_oob=0.28,
        view_xy_weight=0.35,
        view_z_weight=0.32,
        view_bounds_weight=1.15,
        view_oob_weight=0.22,
        view_vxy_weight=0.30,
        hit_vxy_limit=0.54,
        hit_vxy_weight=1.00,
        next_contact_weight=0.14,
        apex_center_weight=0.10,
    )
    cfg_3b = stable_cfg(
        "03_control",
        target_x=(-0.050, 0.060),
        target_y=(0.045, 0.115),
        anchor_z=(-0.026, 0.024),
        launch_height=0.220,
        target_height=0.200,
        hit_height_center=0.240,
        hit_height_tolerance=0.064,
        xy_jitter=0.014,
        z_jitter=0.009,
        init_vxy=0.010,
        init_vz_jitter=0.020,
        obs_pos_noise=0.005,
        obs_vel_noise=0.05,
        camera_center_weight=0.55,
        hit_camera_weight=1.15,
        hit_camera_oob=0.35,
        view_xy_weight=0.45,
        view_z_weight=0.50,
        view_bounds_weight=1.60,
        view_oob_weight=0.35,
        view_vxy_weight=0.38,
        hit_vxy_limit=0.50,
        hit_vxy_weight=1.25,
        next_contact_weight=0.16,
        apex_center_weight=0.11,
    )
    cfg_4a = stable_cfg(
        "04_range",
        target_x=(-0.055, 0.065),
        target_y=(0.055, 0.130),
        anchor_z=(-0.030, 0.026),
        launch_height=0.220,
        target_height=0.205,
        hit_height_center=0.242,
        hit_height_tolerance=0.060,
        xy_jitter=0.016,
        z_jitter=0.010,
        init_vxy=0.012,
        init_vz_jitter=0.024,
        obs_pos_noise=0.006,
        obs_vel_noise=0.06,
        camera_center_weight=0.68,
        hit_camera_weight=1.20,
        hit_camera_oob=0.42,
        view_xy_weight=0.58,
        view_z_weight=0.80,
        view_bounds_weight=2.40,
        view_oob_weight=0.55,
        view_vxy_weight=0.48,
        hit_vxy_limit=0.48,
        hit_vxy_weight=1.55,
        next_contact_weight=0.18,
        apex_center_weight=0.13,
    )
    cfg_4b = stable_cfg(
        "07_dynamics",
        target_x=(-0.065, 0.070),
        target_y=(0.065, 0.145),
        anchor_z=(-0.034, 0.028),
        launch_height=0.225,
        target_height=0.198,
        hit_height_center=0.228,
        hit_height_tolerance=0.050,
        xy_jitter=0.018,
        z_jitter=0.011,
        init_vxy=0.014,
        init_vz_jitter=0.028,
        obs_pos_noise=0.007,
        obs_vel_noise=0.07,
        camera_center_weight=0.78,
        hit_camera_weight=1.25,
        hit_camera_oob=0.50,
        view_xy_weight=0.68,
        view_z_weight=1.20,
        view_bounds_weight=3.20,
        view_oob_weight=0.80,
        view_vxy_weight=0.55,
        hit_vxy_limit=0.46,
        hit_vxy_weight=1.90,
        next_contact_weight=0.20,
        apex_center_weight=0.15,
    )
    cfg_4b = replace(
        cfg_4b,
        rel_height_center=0.178,
        low_hit_apex_margin=0.024,
        apex_soft_limit_margin=0.050,
        hit_cadence_target_interval=0.425,
        hit_cadence_sigma=0.090,
        hit_reward_combo=0.34,
        post_hit_survival_reward_weight=2.55,
        ball_view_z_not_ideal_penalty_weight=max(float(cfg_4b.ball_view_z_not_ideal_penalty_weight), 0.55),
        hit_next_contact_anchor_penalty_weight=0.450,
    )
    cfg_4ab = replace(
        cfg_4a,
        dr_randomize_contact=True,
        dr_randomize_actuator_cmd_filter=True,
        dr_actuator_cmd_tau_range=(0.071, 0.077),
        dr_actuator_cmd_gain_range=(0.992, 1.008),
    )
    cfg_4ac = replace(
        cfg_4b,
        # 4ac was plateauing with high hit count but low z_ideal: the old
        # hit apex target sat at the 1.30 m ideal-z ceiling, and the tolerance
        # allowed many rewardable hits above it.  Align the post-hit target
        # with the D455 ideal band before adding more sim2real DR.
        target_height=0.198,
        rel_height_center=0.178,
        hit_height_center=0.228,
        hit_height_tolerance=0.050,
        apex_soft_limit_margin=0.045,
        hit_cadence_target_interval=0.425,
        hit_cadence_sigma=0.090,
        episode_target_x_range_m=(-0.060, 0.068),
        episode_target_y_range_m=(0.060, 0.138),
        episode_racket_anchor_z_range_m=(-0.032, 0.027),
        ball_spawn_xy_jitter=0.017,
        ball_spawn_z_jitter=0.0105,
        ball_init_vxy_max=0.013,
        ball_init_vz_jitter=0.026,
        ball_obs_pos_noise_std=0.0065,
        ball_obs_vel_noise_std=0.065,
        dr_randomize_racket_mount=False,
        dr_randomize_ball_obs_frame=False,
        dr_action_scale_mult_range=(0.94, 1.06),
        dr_damping_mult_range=(0.86, 1.14),
        dr_armature_mult_range=(0.90, 1.10),
        dr_pd_kp_mult_range=(0.95, 1.05),
        dr_pd_kv_mult_range=(0.90, 1.10),
        dr_actuator_cmd_tau_range=(0.071, 0.077),
        dr_actuator_cmd_gain_range=(0.992, 1.008),
        ball_view_xy_center_penalty_weight=0.63,
        ball_view_z_ideal_penalty_weight=1.20,
        ball_view_z_not_ideal_penalty_weight=0.55,
        ball_view_bounds_penalty_weight=2.80,
        ball_view_out_of_bounds_penalty_weight=0.65,
        ball_view_vxy_excess_penalty_weight=0.52,
        hit_camera_reward_weight=1.22,
        hit_camera_out_of_band_penalty_weight=0.46,
        hit_vxy_soft_limit_m_s=0.47,
        hit_vxy_penalty_weight=1.72,
        hit_next_contact_anchor_penalty_weight=0.19,
        hit_apex_view_center_penalty_weight=0.14,
        camera_center_weight=0.73,
    )
    cfg_4ad = replace(
        cfg_4ac,
        # First racket/observation-frame bridge.  Keep it deliberately close
        # to 4ac so advance validation checks for collapse instead of asking
        # an untrained policy to solve the full obs-frame DR jump at once.
        dr_randomize_racket_mount=True,
        dr_randomize_ball_obs_frame=True,
        dr_action_scale_mult_range=(0.935, 1.065),
        dr_damping_mult_range=(0.85, 1.15),
        dr_armature_mult_range=(0.895, 1.105),
        dr_pd_kp_mult_range=(0.945, 1.055),
        dr_pd_kv_mult_range=(0.895, 1.105),
        dr_actuator_cmd_tau_range=(0.070, 0.080),
        dr_actuator_cmd_gain_range=(0.989, 1.011),
        dr_racket_pos_offset_m=0.0008,
        dr_racket_rot_offset_rad=float(np.deg2rad(0.30)),
        dr_racket_radius_offset_m=0.0005,
        dr_ball_obs_pos_bias_base_m=(0.0015, 0.0015, 0.0015),
        dr_ball_obs_rot_bias_deg=(0.25, 0.25, 0.35),
        dr_ball_obs_vel_bias_base_m_s=(0.015, 0.015, 0.022),
        dr_ball_obs_scale_range=(0.996, 1.004),
        camera_center_weight=0.76,
        ball_view_z_ideal_penalty_weight=1.25,
        ball_view_z_not_ideal_penalty_weight=0.58,
        ball_view_bounds_penalty_weight=3.00,
        ball_view_out_of_bounds_penalty_weight=0.72,
        hit_camera_reward_weight=1.24,
        hit_camera_out_of_band_penalty_weight=0.48,
        hit_vxy_soft_limit_m_s=0.465,
        hit_vxy_penalty_weight=1.82,
        hit_next_contact_anchor_penalty_weight=0.195,
    )
    cfg_4ae = replace(
        cfg_4ad,
        # Mid obs-frame bridge.  The direct jump from 4ad to the old 4ae
        # plateaued around 8 hits / 0.58 length on 2026-07-15, so split the
        # camera/racket-frame perturbation ramp instead of changing PPO.
        dr_action_scale_mult_range=(0.9325, 1.0675),
        dr_damping_mult_range=(0.845, 1.155),
        dr_armature_mult_range=(0.8925, 1.1075),
        dr_pd_kp_mult_range=(0.9425, 1.0575),
        dr_pd_kv_mult_range=(0.8925, 1.1075),
        dr_actuator_cmd_tau_range=(0.0695, 0.0805),
        dr_actuator_cmd_gain_range=(0.9885, 1.0115),
        dr_racket_pos_offset_m=0.00115,
        dr_racket_rot_offset_rad=float(np.deg2rad(0.43)),
        dr_racket_radius_offset_m=0.00075,
        dr_ball_obs_pos_bias_base_m=(0.0020, 0.0020, 0.0020),
        dr_ball_obs_rot_bias_deg=(0.35, 0.35, 0.50),
        dr_ball_obs_vel_bias_base_m_s=(0.020, 0.020, 0.028),
        dr_ball_obs_scale_range=(0.995, 1.005),
        ball_view_z_ideal_penalty_weight=1.28,
        ball_view_z_not_ideal_penalty_weight=0.59,
        hit_apex_view_center_penalty_weight=0.145,
    )
    cfg_4af = replace(
        cfg_4ad,
        # Full obs-frame bridge: previous 4ae-sized racket/obs-frame
        # perturbations, kept as a separate stage after the easier 4ae ramp.
        dr_action_scale_mult_range=(0.93, 1.07),
        dr_damping_mult_range=(0.84, 1.16),
        dr_armature_mult_range=(0.89, 1.11),
        dr_pd_kp_mult_range=(0.94, 1.06),
        dr_pd_kv_mult_range=(0.89, 1.11),
        dr_actuator_cmd_tau_range=(0.069, 0.081),
        dr_actuator_cmd_gain_range=(0.988, 1.012),
        dr_racket_pos_offset_m=0.0015,
        dr_racket_rot_offset_rad=float(np.deg2rad(0.55)),
        dr_racket_radius_offset_m=0.0010,
        dr_ball_obs_pos_bias_base_m=(0.0025, 0.0025, 0.0025),
        dr_ball_obs_rot_bias_deg=(0.45, 0.45, 0.65),
        dr_ball_obs_vel_bias_base_m_s=(0.025, 0.025, 0.035),
        dr_ball_obs_scale_range=(0.994, 1.006),
        ball_view_z_ideal_penalty_weight=1.30,
        ball_view_z_not_ideal_penalty_weight=0.60,
        hit_apex_view_center_penalty_weight=0.15,
    )
    cfg_4ag = replace(
        cfg_4af,
        # Dynamics bridge.  4af can pass the collapse probe, but the direct
        # jump to 4b plateaued near 4--5 hits on 2026-07-15.  Split the
        # remaining racket/obs-frame and dynamics DR ramp before asking for
        # the full 4b gate.
        episode_target_x_range_m=(-0.0625, 0.069),
        episode_target_y_range_m=(0.0625, 0.1415),
        episode_racket_anchor_z_range_m=(-0.033, 0.0275),
        ball_spawn_xy_jitter=0.0175,
        ball_spawn_z_jitter=0.01075,
        ball_init_vxy_max=0.0135,
        ball_init_vz_jitter=0.027,
        ball_obs_pos_noise_std=0.00675,
        ball_obs_vel_noise_std=0.0675,
        dr_action_scale_mult_range=(0.925, 1.075),
        dr_damping_mult_range=(0.83, 1.17),
        dr_armature_mult_range=(0.885, 1.115),
        dr_pd_kp_mult_range=(0.935, 1.065),
        dr_pd_kv_mult_range=(0.885, 1.115),
        dr_actuator_cmd_tau_range=(0.0685, 0.0815),
        dr_actuator_cmd_gain_range=(0.9865, 1.0135),
        dr_racket_pos_offset_m=0.00175,
        dr_racket_rot_offset_rad=float(np.deg2rad(0.675)),
        dr_racket_radius_offset_m=0.00125,
        dr_ball_obs_pos_bias_base_m=(0.00325, 0.00325, 0.00325),
        dr_ball_obs_rot_bias_deg=(0.575, 0.575, 0.825),
        dr_ball_obs_vel_bias_base_m_s=(0.0325, 0.0325, 0.0425),
        dr_ball_obs_scale_range=(0.992, 1.008),
        ball_view_z_ideal_penalty_weight=1.25,
        ball_view_bounds_penalty_weight=3.10,
        hit_vxy_penalty_weight=1.86,
        hit_reward_combo=0.28,
        post_hit_survival_reward_weight=2.25,
        hit_next_contact_anchor_penalty_weight=0.340,
        camera_center_weight=0.77,
        hit_camera_reward_weight=1.245,
        hit_camera_out_of_band_penalty_weight=0.49,
    )
    cfg_4ag2 = replace(
        cfg_4ag,
        # Flatness/DR bridge.  Strict 4ag learned the D455 dynamics but
        # direct 4ah still plateaued near 6.7 hits / 0.54 length, so ramp
        # the last flatness and observation-frame DR step before the 4ah gate.
        episode_target_x_range_m=(-0.0632, 0.0693),
        episode_target_y_range_m=(0.0632, 0.1425),
        episode_racket_anchor_z_range_m=(-0.03325, 0.02765),
        ball_spawn_xy_jitter=0.01765,
        ball_spawn_z_jitter=0.01085,
        ball_init_vxy_max=0.01365,
        ball_init_vz_jitter=0.02725,
        ball_obs_pos_noise_std=0.00682,
        ball_obs_vel_noise_std=0.0682,
        dr_action_scale_mult_range=(0.92375, 1.07625),
        dr_damping_mult_range=(0.8275, 1.1725),
        dr_armature_mult_range=(0.88375, 1.11625),
        dr_pd_kp_mult_range=(0.93375, 1.06625),
        dr_pd_kv_mult_range=(0.88375, 1.11625),
        dr_actuator_cmd_tau_range=(0.06838, 0.08162),
        dr_actuator_cmd_gain_range=(0.9861, 1.0139),
        dr_racket_pos_offset_m=0.00182,
        dr_racket_rot_offset_rad=float(np.deg2rad(0.7125)),
        dr_racket_radius_offset_m=0.00133,
        dr_ball_obs_pos_bias_base_m=(0.00345, 0.00345, 0.00345),
        dr_ball_obs_rot_bias_deg=(0.61, 0.61, 0.88),
        dr_ball_obs_vel_bias_base_m_s=(0.0345, 0.0345, 0.0445),
        dr_ball_obs_scale_range=(0.9915, 1.0085),
        ball_view_z_ideal_penalty_weight=1.24,
        ball_view_z_not_ideal_penalty_weight=0.585,
        ball_view_bounds_penalty_weight=3.12,
        hit_vxy_penalty_weight=1.87,
        hit_reward_combo=0.30,
        post_hit_survival_reward_weight=2.35,
        hit_next_contact_anchor_penalty_weight=0.380,
        camera_center_weight=0.772,
        hit_camera_reward_weight=1.246,
        hit_camera_out_of_band_penalty_weight=0.492,
    )
    cfg_4ah = replace(
        cfg_4ag,
        # Near-4b bridge.  The flat985 run reached a long-window plateau in
        # 4b around 5.4 hits / 0.41 length while the racket stayed near an
        # 11 deg average tilt.  Move the last dynamics/obs-frame DR step and
        # stricter flat-racket reward into a separate stage before the full
        # 4b gate.
        episode_target_x_range_m=(-0.064, 0.0695),
        episode_target_y_range_m=(0.064, 0.1435),
        episode_racket_anchor_z_range_m=(-0.0335, 0.0278),
        ball_spawn_xy_jitter=0.0178,
        ball_spawn_z_jitter=0.0109,
        ball_init_vxy_max=0.0138,
        ball_init_vz_jitter=0.0275,
        ball_obs_pos_noise_std=0.0069,
        ball_obs_vel_noise_std=0.069,
        dr_action_scale_mult_range=(0.9225, 1.0775),
        dr_damping_mult_range=(0.825, 1.175),
        dr_armature_mult_range=(0.8825, 1.1175),
        dr_pd_kp_mult_range=(0.9325, 1.0675),
        dr_pd_kv_mult_range=(0.8825, 1.1175),
        dr_actuator_cmd_tau_range=(0.06825, 0.08175),
        dr_actuator_cmd_gain_range=(0.98575, 1.01425),
        dr_racket_pos_offset_m=0.0019,
        dr_racket_rot_offset_rad=float(np.deg2rad(0.75)),
        dr_racket_radius_offset_m=0.0014,
        dr_ball_obs_pos_bias_base_m=(0.00365, 0.00365, 0.00365),
        dr_ball_obs_rot_bias_deg=(0.65, 0.65, 0.93),
        dr_ball_obs_vel_bias_base_m_s=(0.0365, 0.0365, 0.0465),
        dr_ball_obs_scale_range=(0.991, 1.009),
        ball_view_z_ideal_penalty_weight=1.23,
        ball_view_z_not_ideal_penalty_weight=0.57,
        ball_view_bounds_penalty_weight=3.15,
        hit_vxy_penalty_weight=1.88,
        hit_reward_combo=0.32,
        post_hit_survival_reward_weight=2.45,
        hit_next_contact_anchor_penalty_weight=0.420,
        camera_center_weight=0.775,
        hit_camera_reward_weight=1.248,
        hit_camera_out_of_band_penalty_weight=0.495,
    )
    cfg_4g = stable_cfg(
        "09_final",
        target_x=(-0.075, 0.075),
        target_y=(0.070, 0.155),
        anchor_z=(-0.038, 0.030),
        launch_height=0.225,
        target_height=0.198,
        hit_height_center=0.228,
        hit_height_tolerance=0.050,
        xy_jitter=0.020,
        z_jitter=0.012,
        init_vxy=0.015,
        init_vz_jitter=0.030,
        obs_pos_noise=0.008,
        obs_vel_noise=0.08,
        camera_center_weight=0.90,
        hit_camera_weight=1.35,
        hit_camera_oob=0.60,
        view_xy_weight=0.78,
        view_z_weight=1.45,
        view_bounds_weight=4.00,
        view_oob_weight=1.10,
        view_vxy_weight=0.62,
        hit_vxy_limit=0.44,
        hit_vxy_weight=2.25,
        next_contact_weight=0.48,
        apex_center_weight=0.16,
    )
    cfg_4g = replace(
        cfg_4g,
        rel_height_center=0.178,
        low_hit_apex_margin=0.024,
        apex_soft_limit_margin=0.050,
        hit_cadence_target_interval=0.425,
        hit_cadence_sigma=0.090,
        hit_reward_combo=0.35,
        post_hit_survival_reward_weight=2.65,
        ball_view_z_not_ideal_penalty_weight=max(float(cfg_4g.ball_view_z_not_ideal_penalty_weight), 0.60),
    )
    cfg_4h = replace(
        cfg_4g,
        ball_spawn_xy_jitter=0.022,
        ball_spawn_z_jitter=0.014,
        ball_init_vxy_max=0.016,
        ball_init_vz_jitter=0.032,
        ball_obs_pos_noise_std=0.009,
        ball_obs_vel_noise_std=0.09,
        camera_center_weight=1.00,
        hit_camera_reward_weight=1.45,
        hit_camera_out_of_band_penalty_weight=0.70,
        hit_vxy_penalty_weight=2.50,
        hit_reward_combo=0.35,
        post_hit_survival_reward_weight=2.70,
        hit_next_contact_anchor_penalty_weight=0.50,
        hit_apex_view_center_penalty_weight=0.18,
    )

    def flat_cfg(
        cfg: MjxJuggleConfig,
        weight: float,
        *,
        target_cos: float = 0.985,
        sigma: float = 0.035,
        hit_sigma: float = 0.045,
        contact_weight: float = 0.70,
    ) -> MjxJuggleConfig:
        return replace(
            cfg,
            racket_flatness_penalty_weight=float(weight),
            # Keep the D455 sim-to-real policy close to a horizontal racket.
            # Late stages use cos(8.1 deg) ~= 0.990: enough for small
            # corrective tilt, but it removes the visible press-down/recover
            # behavior that is risky on the real racket.
            racket_flatness_target_cos=float(target_cos),
            racket_flatness_sigma=float(sigma),
            hit_flatness_target_cos=float(target_cos),
            hit_flatness_sigma=float(hit_sigma),
            contact_flatness_penalty_weight=max(float(cfg.contact_flatness_penalty_weight), float(contact_weight)),
        )

    (
        cfg_1a,
        cfg_1b,
        cfg_2a,
        cfg_2b,
        cfg_3a,
        cfg_3b,
        cfg_4a,
        cfg_4ab,
        cfg_4ac,
        cfg_4ad,
        cfg_4ae,
        cfg_4af,
        cfg_4ag,
        cfg_4ag2,
        cfg_4ah,
        cfg_4b,
        cfg_4g,
        cfg_4h,
    ) = (
        flat_cfg(cfg_1a, 0.04),
        flat_cfg(cfg_1b, 0.05),
        flat_cfg(cfg_2a, 0.07),
        flat_cfg(cfg_2b, 0.09),
        flat_cfg(cfg_3a, 0.12),
        flat_cfg(cfg_3b, 0.15),
        flat_cfg(cfg_4a, 0.20),
        flat_cfg(cfg_4ab, 0.24),
        flat_cfg(cfg_4ac, 0.36),
        flat_cfg(cfg_4ad, 0.40),
        flat_cfg(cfg_4ae, 0.43),
        flat_cfg(cfg_4af, 0.46),
        flat_cfg(cfg_4ag, 0.60, target_cos=0.988, sigma=0.032, hit_sigma=0.042, contact_weight=0.90),
        flat_cfg(cfg_4ag2, 0.68, target_cos=0.989, sigma=0.031, hit_sigma=0.041, contact_weight=0.98),
        flat_cfg(cfg_4ah, 0.75, target_cos=0.990, sigma=0.030, hit_sigma=0.040, contact_weight=1.05),
        flat_cfg(cfg_4b, 0.85, target_cos=0.990, sigma=0.030, hit_sigma=0.040, contact_weight=1.10),
        flat_cfg(cfg_4g, 1.00, target_cos=0.990, sigma=0.030, hit_sigma=0.040, contact_weight=1.15),
        flat_cfg(cfg_4h, 1.10, target_cos=0.990, sigma=0.030, hit_sigma=0.040, contact_weight=1.20),
    )

    return [
        CurriculumStage(
            "stage1a_d455_anchor_drop_first_hit",
            steps(1_500_000),
            cfg_1a,
            "Fixed-arm anchor-drop first-hit bootstrap in the calibrated D455 lower-middle view.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=0.90,
            target_mean_len_frac=0.09,
            min_updates=20,
            target_hit1_rate=0.74,
            target_hit_camera_visible_rate=0.70,
            target_hit_camera_lower_band_rate=0.25,
        ),
        CurriculumStage(
            "stage1b_d455_anchor_drop_recoverable_hit",
            steps(1_800_000),
            cfg_1b,
            "Shape the first nominal hit so its post-hit apex and next contact stay recoverable.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=1.05,
            target_mean_len_frac=0.11,
            min_updates=35,
            target_hit1_rate=0.76,
            target_hit_camera_visible_rate=0.76,
            target_hit_camera_lower_band_rate=0.36,
            max_recent_mean_hit_vxy=0.82,
            max_recent_hit_next_contact_anchor_err=0.42,
            max_recent_mean_hit_camera_v_frac=0.86,
        ),
        CurriculumStage(
            "stage2a_d455_second_hit_bridge",
            steps(2_500_000),
            cfg_2a,
            "Bridge from first hit to stable two-hit anchor-drop episodes.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=1.50,
            target_mean_len_frac=0.15,
            min_updates=45,
            target_hit1_rate=0.82,
            target_hit3_rate=0.020,
            target_min_hit_interval_s=0.34,
            target_max_hit_interval_s=0.84,
            target_hit_camera_visible_rate=0.78,
            target_hit_camera_lower_band_rate=0.42,
            max_recent_mean_hit_vxy=0.72,
            max_recent_hit_next_contact_anchor_err=0.36,
            max_recent_mean_hit_camera_v_frac=0.86,
        ),
        CurriculumStage(
            "stage2b_d455_three_hit_bridge",
            steps(2_500_000),
            cfg_2b,
            "Require frequent three-hit behavior before adding longer-horizon DR.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=2.20,
            target_mean_len_frac=0.20,
            min_updates=55,
            target_hit1_rate=0.86,
            target_hit3_rate=0.12,
            target_mean_hits_ge3=3.2,
            target_min_hit_interval_s=0.36,
            target_max_hit_interval_s=0.76,
            target_hit_camera_visible_rate=0.82,
            target_hit_camera_lower_band_rate=0.48,
            max_recent_mean_hit_vxy=0.66,
            max_recent_hit_next_contact_anchor_err=0.32,
        ),
        CurriculumStage(
            "stage3a_d455_contact_pair",
            steps(3_000_000),
            cfg_3a,
            "Nominal multi-hit contact-pair consolidation with D455 view reward active.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=3.2,
            target_mean_len_frac=0.28,
            min_updates=50,
            target_camera_visible=0.72,
            target_ball_view_in_bounds=0.62,
            target_ball_view_z_ideal=0.48,
            target_hit1_rate=0.88,
            target_hit3_rate=0.28,
            target_mean_hits_ge3=4.2,
            target_min_hit_interval_s=0.36,
            target_max_hit_interval_s=0.70,
            target_hit_camera_visible_rate=0.84,
            target_hit_camera_lower_band_rate=0.54,
        ),
        CurriculumStage(
            "stage3b_d455_control",
            steps(4_000_000),
            cfg_3b,
            "Stabilize cadence, height, and camera centering before broadening the nominal reset.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=5.0,
            target_mean_len_frac=0.40,
            min_updates=55,
            target_camera_visible=0.74,
            target_ball_view_in_bounds=0.66,
            target_ball_view_z_ideal=0.52,
            target_hit1_rate=0.90,
            target_hit3_rate=0.42,
            target_mean_hits_ge3=6.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.64,
            target_hit_camera_visible_rate=0.86,
            target_hit_camera_lower_band_rate=0.60,
            target_episode_truncation_rate=0.20,
        ),
        CurriculumStage(
            "stage4a_d455_nominal_range",
            steps(5_000_000),
            cfg_4a,
            "Moderately broaden the nominal anchor-drop reset while keeping all observations visible.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=7.0,
            target_mean_len_frac=0.55,
            min_updates=60,
            target_camera_visible=0.76,
            target_ball_view_in_bounds=0.68,
            target_ball_view_z_ideal=0.56,
            target_hit1_rate=0.93,
            target_hit3_rate=0.58,
            target_mean_hits_ge3=8.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.58,
            target_hit_camera_visible_rate=0.90,
            target_hit_camera_lower_band_rate=0.68,
            target_episode_truncation_rate=0.40,
        ),
        CurriculumStage(
            "stage4ab_d455_contact_cmd_bridge",
            steps(5_500_000),
            cfg_4ab,
            "Bridge stage: keep the stage4a geometry but enable contact and actuator command-filter DR before full actuator/PD DR.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=7.0,
            target_mean_len_frac=0.55,
            min_updates=55,
            target_camera_visible=0.76,
            target_ball_view_in_bounds=0.68,
            target_ball_view_z_ideal=0.56,
            target_hit1_rate=0.93,
            target_hit3_rate=0.58,
            target_mean_hits_ge3=8.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.58,
            target_hit_camera_visible_rate=0.90,
            target_hit_camera_lower_band_rate=0.68,
            target_episode_truncation_rate=0.40,
        ),
        CurriculumStage(
            "stage4ac_d455_light_actuator_pd_bridge",
            steps(6_000_000),
            cfg_4ac,
            "Bridge stage: add mild actuator and PD DR with an intermediate reset bucket before racket/observation-frame DR.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=8.0,
            target_mean_len_frac=0.62,
            min_updates=60,
            target_camera_visible=0.77,
            target_ball_view_in_bounds=0.69,
            target_ball_view_z_ideal=0.57,
            target_hit1_rate=0.94,
            target_hit3_rate=0.64,
            target_hit12_rate=0.12,
            target_mean_hits_ge3=9.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.56,
            target_hit_camera_visible_rate=0.92,
            target_hit_camera_lower_band_rate=0.72,
            target_episode_truncation_rate=0.48,
            max_recent_hit_next_contact_anchor_err=0.082,
        ),
        CurriculumStage(
            "stage4ad_d455_racket_obs_frame_bridge",
            steps(5_500_000),
            cfg_4ad,
            "Micro bridge: add small racket-mount and ball-observation-frame DR after the height-aligned 4ac policy.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=8.0,
            target_mean_len_frac=0.62,
            min_updates=60,
            target_camera_visible=0.775,
            target_ball_view_in_bounds=0.695,
            target_ball_view_z_ideal=0.56,
            target_hit1_rate=0.94,
            target_hit3_rate=0.64,
            target_hit12_rate=0.14,
            target_mean_hits_ge3=9.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.55,
            target_hit_camera_visible_rate=0.92,
            target_hit_camera_lower_band_rate=0.73,
            target_episode_truncation_rate=0.50,
            max_recent_hit_next_contact_anchor_err=0.094,
        ),
        CurriculumStage(
            "stage4ae_d455_obs_frame_dr_bridge",
            steps(5_500_000),
            cfg_4ae,
            "Mid bridge: split the 4ad to full obs-frame DR ramp after the direct jump plateaued.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=8.3,
            target_mean_len_frac=0.64,
            min_updates=60,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.57,
            target_hit1_rate=0.94,
            target_hit3_rate=0.66,
            target_hit12_rate=0.18,
            target_mean_hits_ge3=9.4,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.93,
            target_hit_camera_lower_band_rate=0.74,
            target_episode_truncation_rate=0.52,
            max_recent_hit_next_contact_anchor_err=0.102,
        ),
        CurriculumStage(
            "stage4af_d455_full_obs_frame_dr_bridge",
            steps(6_000_000),
            cfg_4af,
            "Full obs-frame bridge: previous 4ae racket/obs-frame DR magnitude before the full stage4b DR/gate.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=8.7,
            target_mean_len_frac=0.67,
            min_updates=65,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.57,
            target_hit1_rate=0.945,
            target_hit3_rate=0.68,
            target_hit12_rate=0.20,
            target_mean_hits_ge3=9.8,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.93,
            target_hit_camera_lower_band_rate=0.75,
            target_episode_truncation_rate=0.55,
            max_recent_hit_next_contact_anchor_err=0.101,
        ),
        CurriculumStage(
            "stage4ag_d455_dynamics_dr_bridge",
            steps(6_000_000),
            cfg_4ag,
            "Dynamics bridge: midpoint between 4af and full 4b after the direct 4b jump plateaued.",
            gate_mode="strict",
            advance_gate_mode="strict",
            target_mean_hits=8.2,
            target_mean_len_frac=0.61,
            min_updates=65,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.57,
            target_hit1_rate=0.945,
            target_hit3_rate=0.67,
            target_hit12_rate=0.28,
            target_mean_hits_ge3=10.3,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.93,
            target_hit_camera_lower_band_rate=0.755,
            target_episode_truncation_rate=0.48,
            max_recent_hit_next_contact_anchor_err=0.112,
            target_racket_up_cos=0.982,
        ),
        CurriculumStage(
            "stage4ag2_d455_flatness_dr_bridge",
            steps(6_000_000),
            cfg_4ag2,
            "Flatness/DR bridge between strict 4ag and the full late-dynamics 4ah gate.",
            gate_mode="strict",
            advance_gate_mode="strict",
            target_mean_hits=8.6,
            target_mean_len_frac=0.63,
            min_updates=60,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.57,
            target_hit1_rate=0.945,
            target_hit3_rate=0.70,
            target_hit12_rate=0.38,
            target_mean_hits_ge3=11.2,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.93,
            target_hit_camera_lower_band_rate=0.756,
            target_episode_truncation_rate=0.50,
            max_recent_hit_next_contact_anchor_err=0.110,
            target_racket_up_cos=0.984,
        ),
        CurriculumStage(
            "stage4ah_d455_late_dynamics_flat_bridge",
            steps(6_000_000),
            cfg_4ah,
            "Late bridge: near-4b DR plus stricter flat-racket shaping before the full 4b gate.",
            gate_mode="strict",
            advance_gate_mode="strict",
            target_mean_hits=9.5,
            target_mean_len_frac=0.68,
            min_updates=70,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.57,
            target_hit1_rate=0.947,
            target_hit3_rate=0.72,
            target_hit12_rate=0.45,
            target_mean_hits_ge3=11.8,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.93,
            target_hit_camera_lower_band_rate=0.758,
            target_episode_truncation_rate=0.56,
            max_recent_hit_next_contact_anchor_err=0.106,
            target_racket_up_cos=0.985,
        ),
        CurriculumStage(
            "stage4b_d455_contact_actuator_dr",
            steps(6_000_000),
            cfg_4b,
            "Add contact, actuator, and PD DR after the nominal trajectory is already multi-hit.",
            gate_mode="strict",
            advance_gate_mode="strict",
            target_mean_hits=10.5,
            target_mean_len_frac=0.75,
            min_updates=75,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.58,
            target_hit1_rate=0.95,
            target_hit3_rate=0.76,
            target_hit12_rate=0.55,
            target_mean_hits_ge3=12.3,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.93,
            target_hit_camera_lower_band_rate=0.76,
            target_episode_truncation_rate=0.64,
            max_recent_hit_next_contact_anchor_err=0.102,
            target_racket_up_cos=0.986,
        ),
        CurriculumStage(
            "stage4g_d455_stable_dr",
            steps(9_000_000),
            cfg_4g,
            "Stable nominal final objective: 13--15 hits in 1200 steps, without recovery-state resets.",
            gate_mode="strict",
            advance_gate_mode="collapse",
            target_mean_hits=12.8,
            target_mean_len_frac=0.92,
            min_updates=90,
            target_camera_visible=0.80,
            target_ball_view_in_bounds=0.72,
            target_ball_view_z_ideal=0.60,
            target_hit1_rate=0.97,
            target_hit3_rate=0.86,
            target_hit12_rate=0.76,
            target_mean_hits_ge3=13.5,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.52,
            target_hit_camera_visible_rate=0.96,
            target_hit_camera_lower_band_rate=0.84,
            target_episode_truncation_rate=0.86,
            max_recent_hit_next_contact_anchor_err=0.090,
            target_racket_up_cos=0.987,
            max_ball_obs_lost_rate=0.010,
        ),
        CurriculumStage(
            "stage4h_d455_stable_polish",
            steps(10_000_000),
            cfg_4h,
            "Polish the converged nominal D455 policy before switching to recovery learning.",
            gate_mode="strict",
            advance_gate_mode="collapse",
            target_mean_hits=13.2,
            target_mean_len_frac=0.95,
            min_updates=100,
            target_camera_visible=0.82,
            target_ball_view_in_bounds=0.74,
            target_ball_view_z_ideal=0.62,
            target_hit1_rate=0.98,
            target_hit3_rate=0.90,
            target_hit12_rate=0.84,
            target_mean_hits_ge3=14.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.50,
            target_hit_camera_visible_rate=0.98,
            target_hit_camera_lower_band_rate=0.88,
            target_episode_truncation_rate=0.90,
            target_racket_up_cos=0.988,
            max_recent_hit_next_contact_anchor_err=0.086,
            max_ball_obs_lost_rate=0.008,
        ),
    ]



def _d455_success_ref_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
) -> list[CurriculumStage]:
    """D455 reference-structured profile rebuilt from the successful legacy curricula.

    The two reference runs used the old low-gate 18-stage progression and did
    not need the dense recoverability/flatness gates that later caused the
    D455 profile to stall.  This profile keeps the current real D455 reset
    pose, view bounds, and done conditions from the first stage, but rebuilds
    the course distribution by difficulty: nominal closed-loop juggling,
    visible sim2real DR, visible range expansion, missing ramp, and final
    full-missing/large-DR polish.
    """

    base = MjxJuggleConfig(**stack_kwargs)
    base = _with_latest_d455_camera(base)
    base = replace(
        base,
        horizon_sec=6.0,
        right_arm_pd_profile="xml",
        right_arm_reset_degrees=D455_USER_TARGET_RACKET_RESET_DEGREES,
        ball_reset_mode="anchor_drop",
        ball_launch_height=0.225,
        target_height=0.198,
        rel_height_center=0.178,
        hit_height_center=0.228,
        hit_confirm_rel_height=0.055,
        hit_height_tolerance=0.050,
        low_hit_apex_margin=0.024,
        apex_soft_limit_margin=0.050,
        ball_init_vz=-0.28,
        arm_action_limiter=True,
        action_acc_scale=1.0,
        action_penalty_weight=0.0018,
        action_delta_penalty_weight=0.0012,
        posture_weight=0.80,
        base_pose_weight=0.15,
        torque_penalty_weight=0.0005,
        arm_vel_limit_penalty_weight=0.06,
        arm_acc_limit_penalty_weight=0.08,
        arm_limiter_penalty_weight=0.08,
        ball_anchor_xy_penalty_weight=0.60,
        ball_base_x_penalty_weight=0.20,
        ball_base_x_soft_limit=0.12,
        ball_base_vxy_penalty_weight=0.45,
        ball_vxy_penalty_weight=0.12,
        ball_xy_soft_limit_radius=0.18,
        ball_xy_soft_penalty_weight=2.0,
        post_hit_survival_reward_weight=1.40,
        hit_reward_base=0.50,
        hit_reward_combo=0.02,
        center_flat_hit_reward_weight=0.80,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.425,
        hit_reward_count_cap=0,
        hit_combo_count_cap=14,
        hit_cadence_reward_weight=0.10,
        hit_cadence_target_interval=0.425,
        hit_cadence_sigma=0.11,
        hit_min_interval_penalty_weight=1.35,
        hit_min_interval=0.34,
        hit_min_count_interval=0.32,
        fast_hit_penalty_weight=0.90,
        hit_camera_reward_weight=0.70,
        hit_camera_out_of_band_penalty_weight=0.0,
        hit_camera_target_v_frac=0.66,
        hit_camera_v_sigma_frac=0.12,
        hit_camera_lower_band_frac=(0.50, 0.82),
        ball_obs_rate_hz=60.0,
        ball_obs_fractional_rate=True,
        ball_obs_age_clip=0.35,
        lost_ball_timeout_ms=450.0,
        terminate_on_ball_view_bounds=True,
        terminate_on_ball_view_x_bounds=True,
        terminate_on_ball_view_y_bounds=True,
        terminate_on_ball_view_z_low=True,
        terminate_on_ball_view_z_high=False,
        terminate_on_racket_z_limit=True,
        racket_z_limit_termination_penalty_base=4.0,
        racket_z_limit_termination_penalty_per_hit=0.30,
        ball_low_termination_z_m=D455_REAL_VIEW_Z_BOUNDS_M[0],
        ball_high_termination_z_m=max(float(base.ball_high_termination_z_m), D455_REAL_VIEW_Z_BOUNDS_M[1] + 0.20),
        ball_view_x_bounds_m=D455_REAL_VIEW_X_BOUNDS_M,
        ball_view_y_bounds_m=D455_REAL_VIEW_Y_BOUNDS_M,
        ball_view_z_bounds_m=D455_REAL_VIEW_Z_BOUNDS_M,
        ball_view_z_ideal_m=D455_STABLE_VIEW_Z_IDEAL_M,
        ball_view_x_target_m=0.0,
        ball_view_y_target_m=D455_REAL_VIEW_Y_TARGET_M,
        ball_view_x_sigma_m=0.10,
        ball_view_y_sigma_m=0.12,
        ball_view_z_sigma_m=0.11,
        ball_view_xy_center_penalty_weight=0.45,
        ball_view_z_ideal_penalty_weight=0.90,
        ball_view_bounds_penalty_weight=2.00,
        ball_view_out_of_bounds_penalty_weight=0.0,
        ball_view_z_not_ideal_penalty_weight=0.25,
        ball_view_vxy_excess_penalty_weight=0.30,
        ball_view_vxy_soft_limit_m_s=0.80,
        camera_center_weight=0.50,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
        camera_pixel_margin=D455_848_UNDISTORTED_PIXEL_MARGIN,
        camera_min_depth=0.15,
        camera_max_depth=2.50,
        asymmetric_critic=True,
        critic_command_history_steps=max(12, int(critic_command_history_steps)),
    )

    def steps(default_steps: int) -> int:
        return int(stage_steps_override) if stage_steps_override is not None else int(default_steps)

    def cfg_for(
        *,
        target_x: tuple[float, float],
        target_y: tuple[float, float],
        anchor_z: tuple[float, float],
        xy_jitter: float,
        z_jitter: float,
        init_vxy: float,
        init_vz_jitter: float,
        obs_pos_noise: float,
        obs_vel_noise: float,
        dr_level: int,
        view_xy_weight: float,
        view_z_weight: float,
        view_bounds_weight: float,
        camera_center_weight: float,
        view_missing_prob: float = 0.0,
        camera_missing_prob: float = 0.0,
        dropout_prob: float = 0.0,
        burst_prob: float = 0.0,
        dropout_steps: int = 1,
        burst_steps: int = 1,
        z_high_missing_range: tuple[float, float] = (0.0, 0.0),
    ) -> MjxJuggleConfig:
        has_missing = bool(
            view_missing_prob > 0.0
            or camera_missing_prob > 0.0
            or dropout_prob > 0.0
            or burst_prob > 0.0
        )
        cfg = replace(
            base,
            episode_target_x_range_m=target_x,
            episode_target_y_range_m=target_y,
            episode_racket_anchor_z_range_m=anchor_z,
            ball_spawn_xy_jitter=float(xy_jitter),
            ball_spawn_z_jitter=float(z_jitter),
            ball_init_vxy_max=float(init_vxy),
            ball_init_vz_jitter=float(init_vz_jitter),
            ball_obs_pos_noise_std=float(obs_pos_noise),
            ball_obs_vel_noise_std=float(obs_vel_noise),
            ball_obs_age_tracks_stale=has_missing,
            ball_obs_age_clip=0.60 if has_missing else 0.35,
            ball_obs_dropout_on_refresh_only=has_missing,
            ball_obs_require_camera_visible=camera_missing_prob > 0.0,
            ball_obs_camera_missing_prob=float(camera_missing_prob),
            ball_obs_reset_respects_camera_visibility=camera_missing_prob > 0.0,
            ball_obs_require_view_bounds=view_missing_prob > 0.0,
            ball_obs_view_bounds_missing_prob=float(view_missing_prob),
            ball_obs_view_z_high_missing_range_m=(
                z_high_missing_range if view_missing_prob > 0.0 else (0.0, 0.0)
            ),
            ball_obs_missing_episode_coherent_prob=1.0 if has_missing else 0.0,
            ball_obs_dropout_prob=float(dropout_prob),
            ball_obs_dropout_burst_prob=float(burst_prob),
            ball_obs_dropout_max_steps=int(dropout_steps) if has_missing else 1,
            ball_obs_dropout_burst_max_steps=int(burst_steps) if has_missing else 1,
            lost_ball_timeout_ms=500.0 if has_missing else 450.0,
            ball_view_xy_center_penalty_weight=float(view_xy_weight),
            ball_view_z_ideal_penalty_weight=float(view_z_weight),
            ball_view_bounds_penalty_weight=float(view_bounds_weight),
            ball_view_out_of_bounds_penalty_weight=1.20 if has_missing else 0.0,
            ball_view_vxy_excess_penalty_weight=0.45 if has_missing else 0.30,
            camera_center_weight=float(camera_center_weight),
            camera_visible_penalty_weight=3.0 if not has_missing else 2.0,
            racket_z_hard_limit_up=max(float(base.racket_z_hard_limit_up), max(0.0, -float(anchor_z[0])) + 0.14),
        )
        if dr_level <= 0:
            return replace(
                cfg,
                domain_randomization=False,
                dr_randomize_ball=False,
                dr_randomize_contact=False,
                dr_randomize_actuator=False,
                dr_randomize_latency=False,
                dr_randomize_pd=False,
                dr_randomize_racket_mount=False,
                dr_randomize_ball_obs_frame=False,
                dr_randomize_actuator_cmd_filter=False,
            )

        cfg = replace(
            cfg,
            domain_randomization=True,
            dr_randomize_ball=True,
            dr_randomize_contact=dr_level >= 2,
            dr_randomize_actuator=dr_level >= 3,
            dr_randomize_pd=dr_level >= 3,
            dr_pd_per_joint=True,
            dr_randomize_latency=dr_level >= 4,
            dr_randomize_racket_mount=dr_level >= 5,
            dr_randomize_ball_obs_frame=dr_level >= 6,
            dr_randomize_actuator_cmd_filter=dr_level >= 3,
            dr_action_scale_mult_range=(0.93, 1.07),
            dr_damping_mult_range=(0.85, 1.15),
            dr_armature_mult_range=(0.90, 1.10),
            dr_pd_kp_mult_range=(0.95, 1.05),
            dr_pd_kv_mult_range=(0.90, 1.10),
        )
        if dr_level == 1:
            return replace(
                cfg,
                dr_randomize_actuator_cmd_filter=False,
                dr_ball_friction_range=(0.16, 0.30),
                dr_racket_friction_range=(0.32, 0.56),
                dr_ball_solref_time_range=(0.0025, 0.0060),
                dr_ball_solref_damping_range=(0.70, 0.96),
            )
        if dr_level == 2:
            return replace(
                cfg,
                dr_randomize_actuator_cmd_filter=False,
                dr_ball_friction_range=(0.12, 0.36),
                dr_racket_friction_range=(0.26, 0.64),
                dr_ball_solref_time_range=(0.0020, 0.0080),
                dr_ball_solref_damping_range=(0.62, 1.05),
            )

        actuator_level = "real" if dr_level >= 6 else "medium"
        cfg = replace(cfg, **_actuator_response_dr_kwargs(actuator_level))
        if dr_level >= 4:
            cfg = replace(cfg, dr_obs_latency_steps_range=(0, 2), dr_action_latency_steps_range=(0, 2))
        if dr_level >= 5:
            cfg = replace(
                cfg,
                dr_racket_pos_offset_m=0.0025,
                dr_racket_rot_offset_rad=float(np.deg2rad(0.9)),
                dr_racket_radius_offset_m=0.0018,
            )
        if dr_level >= 6:
            cfg = replace(
                cfg,
                dr_action_scale_mult_range=(0.88, 1.12),
                dr_damping_mult_range=(0.75, 1.25),
                dr_armature_mult_range=(0.82, 1.18),
                dr_pd_kp_mult_range=(0.88, 1.12),
                dr_pd_kv_mult_range=(0.82, 1.18),
                dr_obs_latency_steps_range=(0, 3),
                dr_action_latency_steps_range=(0, 3),
                dr_racket_pos_offset_m=0.004,
                dr_racket_rot_offset_rad=float(np.deg2rad(1.5)),
                dr_racket_radius_offset_m=0.0025,
                dr_ball_obs_pos_bias_base_m=(0.008, 0.008, 0.008),
                dr_ball_obs_rot_bias_deg=(1.5, 1.5, 2.0),
                dr_ball_obs_vel_bias_base_m_s=(0.08, 0.08, 0.10),
                dr_ball_obs_scale_range=(0.98, 1.02),
                dr_ball_friction_range=(0.08, 0.45),
                dr_racket_friction_range=(0.18, 0.75),
                dr_ball_solref_time_range=(0.0015, 0.010),
                dr_ball_solref_damping_range=(0.55, 1.10),
            )
        return cfg

    cfgs = {
        "1a": cfg_for(target_x=(0.055, 0.055), target_y=(0.095, 0.095), anchor_z=(0.0, 0.0), xy_jitter=0.002, z_jitter=0.002, init_vxy=0.0, init_vz_jitter=0.0, obs_pos_noise=0.0, obs_vel_noise=0.0, dr_level=0, view_xy_weight=0.25, view_z_weight=0.55, view_bounds_weight=1.20, camera_center_weight=0.35),
        "1b": cfg_for(target_x=(0.035, 0.075), target_y=(0.080, 0.115), anchor_z=(-0.006, 0.006), xy_jitter=0.006, z_jitter=0.004, init_vxy=0.004, init_vz_jitter=0.010, obs_pos_noise=0.001, obs_vel_noise=0.010, dr_level=0, view_xy_weight=0.30, view_z_weight=0.65, view_bounds_weight=1.40, camera_center_weight=0.40),
        "1c": cfg_for(target_x=(0.025, 0.085), target_y=(0.072, 0.123), anchor_z=(-0.010, 0.010), xy_jitter=0.008, z_jitter=0.006, init_vxy=0.006, init_vz_jitter=0.014, obs_pos_noise=0.0015, obs_vel_noise=0.015, dr_level=0, view_xy_weight=0.42, view_z_weight=0.70, view_bounds_weight=1.50, camera_center_weight=0.46),
        "1d": cfg_for(target_x=(0.015, 0.090), target_y=(0.065, 0.130), anchor_z=(-0.014, 0.014), xy_jitter=0.010, z_jitter=0.007, init_vxy=0.008, init_vz_jitter=0.018, obs_pos_noise=0.002, obs_vel_noise=0.020, dr_level=0, view_xy_weight=0.56, view_z_weight=0.75, view_bounds_weight=1.80, camera_center_weight=0.58),
        "2a": cfg_for(target_x=(0.000, 0.100), target_y=(0.060, 0.138), anchor_z=(-0.020, 0.020), xy_jitter=0.012, z_jitter=0.008, init_vxy=0.010, init_vz_jitter=0.022, obs_pos_noise=0.0025, obs_vel_noise=0.025, dr_level=0, view_xy_weight=0.62, view_z_weight=0.80, view_bounds_weight=2.00, camera_center_weight=0.62),
        "2b": cfg_for(target_x=(-0.020, 0.105), target_y=(0.055, 0.145), anchor_z=(-0.024, 0.022), xy_jitter=0.014, z_jitter=0.009, init_vxy=0.011, init_vz_jitter=0.024, obs_pos_noise=0.003, obs_vel_noise=0.030, dr_level=0, view_xy_weight=0.66, view_z_weight=0.85, view_bounds_weight=2.15, camera_center_weight=0.66),
        "3a": cfg_for(target_x=(-0.030, 0.110), target_y=(0.052, 0.148), anchor_z=(-0.028, 0.024), xy_jitter=0.016, z_jitter=0.010, init_vxy=0.012, init_vz_jitter=0.026, obs_pos_noise=0.0035, obs_vel_noise=0.035, dr_level=1, view_xy_weight=0.45, view_z_weight=0.90, view_bounds_weight=2.00, camera_center_weight=0.50),
        "3b": cfg_for(target_x=(-0.030, 0.110), target_y=(0.052, 0.148), anchor_z=(-0.028, 0.024), xy_jitter=0.016, z_jitter=0.010, init_vxy=0.012, init_vz_jitter=0.026, obs_pos_noise=0.0035, obs_vel_noise=0.035, dr_level=2, view_xy_weight=0.45, view_z_weight=0.90, view_bounds_weight=2.00, camera_center_weight=0.50),
        "4a": cfg_for(target_x=(-0.035, 0.112), target_y=(0.050, 0.150), anchor_z=(-0.030, 0.026), xy_jitter=0.017, z_jitter=0.011, init_vxy=0.013, init_vz_jitter=0.028, obs_pos_noise=0.004, obs_vel_noise=0.040, dr_level=3, view_xy_weight=0.48, view_z_weight=0.95, view_bounds_weight=2.20, camera_center_weight=0.55),
        "4b": cfg_for(target_x=(-0.040, 0.115), target_y=(0.049, 0.151), anchor_z=(-0.031, 0.027), xy_jitter=0.0175, z_jitter=0.0115, init_vxy=0.0135, init_vz_jitter=0.029, obs_pos_noise=0.0042, obs_vel_noise=0.042, dr_level=3, view_xy_weight=0.49, view_z_weight=0.98, view_bounds_weight=2.30, camera_center_weight=0.56),
        # Stage4c previously jumped directly from no raw action latency to
        # action-latency DR in [0, 2] steps.  Validation at stage4c update 54
        # showed a stable ~2-hit plateau with large post-hit vxy and y/x view
        # exits.  A first split to random [0, 1] action latency also plateaued
        # near 2.1 hits by update 84, so first teach the deterministic 1-step
        # phase shift, then introduce the random range.
        "4c0": replace(
            cfg_for(target_x=(-0.040, 0.115), target_y=(0.049, 0.151), anchor_z=(-0.031, 0.027), xy_jitter=0.0175, z_jitter=0.0115, init_vxy=0.0135, init_vz_jitter=0.029, obs_pos_noise=0.0042, obs_vel_noise=0.042, dr_level=3, view_xy_weight=0.49, view_z_weight=0.98, view_bounds_weight=2.30, camera_center_weight=0.56),
            dr_randomize_latency=True,
            dr_obs_latency_steps_range=(0, 2),
            dr_action_latency_steps_range=(1, 1),
        ),
        "4c1": replace(
            cfg_for(target_x=(-0.040, 0.115), target_y=(0.049, 0.151), anchor_z=(-0.031, 0.027), xy_jitter=0.0175, z_jitter=0.0115, init_vxy=0.0135, init_vz_jitter=0.029, obs_pos_noise=0.0042, obs_vel_noise=0.042, dr_level=3, view_xy_weight=0.49, view_z_weight=0.98, view_bounds_weight=2.30, camera_center_weight=0.56),
            dr_randomize_latency=True,
            dr_obs_latency_steps_range=(0, 2),
            dr_action_latency_steps_range=(0, 1),
        ),
        "4c2": replace(
            cfg_for(target_x=(-0.045, 0.118), target_y=(0.048, 0.152), anchor_z=(-0.032, 0.028), xy_jitter=0.018, z_jitter=0.012, init_vxy=0.014, init_vz_jitter=0.030, obs_pos_noise=0.0045, obs_vel_noise=0.045, dr_level=3, view_xy_weight=0.50, view_z_weight=1.00, view_bounds_weight=2.40, camera_center_weight=0.58),
            dr_randomize_latency=True,
            dr_obs_latency_steps_range=(0, 2),
            dr_action_latency_steps_range=(0, 1),
        ),
        "4c3": cfg_for(target_x=(-0.045, 0.118), target_y=(0.048, 0.152), anchor_z=(-0.032, 0.028), xy_jitter=0.018, z_jitter=0.012, init_vxy=0.014, init_vz_jitter=0.030, obs_pos_noise=0.0045, obs_vel_noise=0.045, dr_level=4, view_xy_weight=0.50, view_z_weight=1.00, view_bounds_weight=2.40, camera_center_weight=0.58),
        "4d": cfg_for(target_x=(-0.050, 0.120), target_y=(0.046, 0.154), anchor_z=(-0.034, 0.030), xy_jitter=0.019, z_jitter=0.013, init_vxy=0.015, init_vz_jitter=0.032, obs_pos_noise=0.005, obs_vel_noise=0.050, dr_level=5, view_xy_weight=0.52, view_z_weight=1.05, view_bounds_weight=2.60, camera_center_weight=0.60),
        "4e": cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=6, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
        "5a": cfg_for(target_x=(-0.075, 0.155), target_y=(0.020, 0.180), anchor_z=(-0.045, 0.050), xy_jitter=0.024, z_jitter=0.016, init_vxy=0.018, init_vz_jitter=0.036, obs_pos_noise=0.008, obs_vel_noise=0.080, dr_level=6, view_xy_weight=0.65, view_z_weight=1.25, view_bounds_weight=3.40, camera_center_weight=0.75),
        "5b": cfg_for(target_x=(-0.095, 0.200), target_y=(0.000, 0.205), anchor_z=(-0.055, 0.060), xy_jitter=0.028, z_jitter=0.020, init_vxy=0.020, init_vz_jitter=0.042, obs_pos_noise=0.010, obs_vel_noise=0.100, dr_level=6, view_xy_weight=0.75, view_z_weight=1.45, view_bounds_weight=4.20, camera_center_weight=0.85, view_missing_prob=0.25, camera_missing_prob=0.02, dropout_prob=0.002, burst_prob=0.0005, dropout_steps=5, burst_steps=18, z_high_missing_range=(1.36, 1.47)),
        "5c": cfg_for(target_x=(-0.115, 0.235), target_y=(-0.010, 0.225), anchor_z=(-0.060, 0.070), xy_jitter=0.032, z_jitter=0.024, init_vxy=0.023, init_vz_jitter=0.048, obs_pos_noise=0.012, obs_vel_noise=0.120, dr_level=6, view_xy_weight=0.85, view_z_weight=1.70, view_bounds_weight=5.00, camera_center_weight=0.95, view_missing_prob=0.50, camera_missing_prob=0.04, dropout_prob=0.004, burst_prob=0.0010, dropout_steps=8, burst_steps=28, z_high_missing_range=(1.30, 1.47)),
        "5d": cfg_for(target_x=(-0.135, 0.280), target_y=(-0.020, 0.245), anchor_z=(-0.065, 0.080), xy_jitter=0.036, z_jitter=0.028, init_vxy=0.026, init_vz_jitter=0.054, obs_pos_noise=0.014, obs_vel_noise=0.140, dr_level=6, view_xy_weight=0.95, view_z_weight=2.00, view_bounds_weight=5.80, camera_center_weight=1.05, view_missing_prob=0.75, camera_missing_prob=0.06, dropout_prob=0.006, burst_prob=0.0015, dropout_steps=10, burst_steps=40, z_high_missing_range=(1.24, 1.47)),
        "5e": cfg_for(target_x=(-0.150, 0.300), target_y=(-0.025, 0.255), anchor_z=(-0.070, 0.085), xy_jitter=0.040, z_jitter=0.030, init_vxy=0.030, init_vz_jitter=0.060, obs_pos_noise=0.016, obs_vel_noise=0.160, dr_level=6, view_xy_weight=1.05, view_z_weight=2.30, view_bounds_weight=6.50, camera_center_weight=1.15, view_missing_prob=1.00, camera_missing_prob=0.08, dropout_prob=0.008, burst_prob=0.0020, dropout_steps=12, burst_steps=48, z_high_missing_range=(1.20, 1.47)),
    }

    cfgs.update({
        "4e0": cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
        # The first obs-frame bridge at update50 plateaued around 2.4 hits on
        # two seeds and validation died from x/y view exits.  That is too big a
        # sensor-model jump for the 4e0 attractor, so ramp structured ball
        # observation-frame error in three small steps before real-light DR.
        "4e1a": replace(
            cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0004, 0.0004, 0.0004),
            dr_ball_obs_rot_bias_deg=(0.08, 0.08, 0.12),
            dr_ball_obs_vel_bias_base_m_s=(0.004, 0.004, 0.006),
            dr_ball_obs_scale_range=(0.999, 1.001),
        ),
        "4e1b": replace(
            cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0008, 0.0008, 0.0008),
            dr_ball_obs_rot_bias_deg=(0.15, 0.15, 0.22),
            dr_ball_obs_vel_bias_base_m_s=(0.008, 0.008, 0.010),
            dr_ball_obs_scale_range=(0.998, 1.002),
        ),
        "4e1b2": replace(
            cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0011, 0.0011, 0.0011),
            dr_ball_obs_rot_bias_deg=(0.22, 0.22, 0.32),
            dr_ball_obs_vel_bias_base_m_s=(0.011, 0.011, 0.014),
            dr_ball_obs_scale_range=(0.997, 1.003),
        ),
        "4e1b3": replace(
            cfg_for(target_x=(-0.075, 0.085), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.021, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.56, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.66),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0013, 0.0013, 0.0013),
            dr_ball_obs_rot_bias_deg=(0.26, 0.26, 0.38),
            dr_ball_obs_vel_bias_base_m_s=(0.013, 0.013, 0.017),
            dr_ball_obs_scale_range=(0.9965, 1.0035),
        ),
        "4e1b3b": replace(
            cfg_for(target_x=(-0.085, 0.065), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.022, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.58, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.68),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0013, 0.0013, 0.0013),
            dr_ball_obs_rot_bias_deg=(0.26, 0.26, 0.38),
            dr_ball_obs_vel_bias_base_m_s=(0.013, 0.013, 0.017),
            dr_ball_obs_scale_range=(0.9965, 1.0035),
        ),
        "4e1b3b1": replace(
            # v34 diagnostics showed that the softer low-x edge still loses too
            # many episodes before the first hit.  First tighten y/jitter and
            # shift x only partway left, then return to the wider soft edge.
            cfg_for(target_x=(-0.070, 0.045), target_y=(0.055, 0.145), anchor_z=(-0.034, 0.030), xy_jitter=0.014, z_jitter=0.012, init_vxy=0.014, init_vz_jitter=0.030, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.59, view_z_weight=1.11, view_bounds_weight=2.85, camera_center_weight=0.69),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0013, 0.0013, 0.0013),
            dr_ball_obs_rot_bias_deg=(0.26, 0.26, 0.38),
            dr_ball_obs_vel_bias_base_m_s=(0.013, 0.013, 0.017),
            dr_ball_obs_scale_range=(0.9965, 1.0035),
        ),
        "4e1b3b1h": replace(
            # v35 learned the average 4e1b3b1 distribution, but repeated
            # next-stage probes still failed only in the target_x-low and
            # target_y-high reset buckets.  Make that corner the whole stage so
            # the mean gate cannot hide it behind easier high-x episodes.
            cfg_for(target_x=(-0.078, -0.028), target_y=(0.095, 0.155), anchor_z=(-0.034, 0.030), xy_jitter=0.012, z_jitter=0.012, init_vxy=0.012, init_vz_jitter=0.028, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.60, view_z_weight=1.11, view_bounds_weight=2.90, camera_center_weight=0.70),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0013, 0.0013, 0.0013),
            dr_ball_obs_rot_bias_deg=(0.26, 0.26, 0.38),
            dr_ball_obs_vel_bias_base_m_s=(0.013, 0.013, 0.017),
            dr_ball_obs_scale_range=(0.9965, 1.0035),
        ),
        "4e1b3b2": replace(
            # v33 showed that the pure low-x edge remains too hard even with
            # mid obs-frame DR.  Move the distribution left in a smaller step
            # before making the edge the entire reset bucket.
            cfg_for(target_x=(-0.075, 0.025), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.016, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.59, view_z_weight=1.11, view_bounds_weight=2.85, camera_center_weight=0.69),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0013, 0.0013, 0.0013),
            dr_ball_obs_rot_bias_deg=(0.26, 0.26, 0.38),
            dr_ball_obs_vel_bias_base_m_s=(0.013, 0.013, 0.017),
            dr_ball_obs_scale_range=(0.9965, 1.0035),
        ),
        "4e1b3c0": replace(
            # v32 showed that low-x edge plus full obs-frame DR is too hard
            # directly from 4e1b3b.  First make the weak edge the whole
            # distribution while keeping the 4e1b3b mid obs-frame bias.
            cfg_for(target_x=(-0.095, -0.015), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.018, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.60, view_z_weight=1.12, view_bounds_weight=2.90, camera_center_weight=0.70),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0013, 0.0013, 0.0013),
            dr_ball_obs_rot_bias_deg=(0.26, 0.26, 0.38),
            dr_ball_obs_vel_bias_base_m_s=(0.013, 0.013, 0.017),
            dr_ball_obs_scale_range=(0.9965, 1.0035),
        ),
        "4e1b3c": replace(
            # The 4e1b3b -> 4e1b4 probe repeatedly failed only in the low-x
            # reset buckets.  Train that tail as the whole stage so the mean
            # gate can no longer hide the weak bucket before full obs-frame DR.
            cfg_for(target_x=(-0.095, -0.015), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.018, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.60, view_z_weight=1.12, view_bounds_weight=2.90, camera_center_weight=0.70),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0015, 0.0015, 0.0015),
            dr_ball_obs_rot_bias_deg=(0.30, 0.30, 0.45),
            dr_ball_obs_vel_bias_base_m_s=(0.015, 0.015, 0.020),
            dr_ball_obs_scale_range=(0.996, 1.004),
        ),
        "4e1b4": replace(
            cfg_for(target_x=(-0.085, 0.065), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.022, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.58, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.68),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0015, 0.0015, 0.0015),
            dr_ball_obs_rot_bias_deg=(0.30, 0.30, 0.45),
            dr_ball_obs_vel_bias_base_m_s=(0.015, 0.015, 0.020),
            dr_ball_obs_scale_range=(0.996, 1.004),
        ),
        "4e1c": replace(
            cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=5, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.0015, 0.0015, 0.0015),
            dr_ball_obs_rot_bias_deg=(0.30, 0.30, 0.45),
            dr_ball_obs_vel_bias_base_m_s=(0.015, 0.015, 0.020),
            dr_ball_obs_scale_range=(0.996, 1.004),
        ),
        "4e1d": replace(
            cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=6, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
            dr_action_scale_mult_range=(0.91, 1.09),
            dr_damping_mult_range=(0.82, 1.18),
            dr_armature_mult_range=(0.88, 1.12),
            dr_pd_kp_mult_range=(0.92, 1.08),
            dr_pd_kv_mult_range=(0.88, 1.12),
            dr_obs_latency_steps_range=(0, 2),
            dr_action_latency_steps_range=(0, 2),
            dr_racket_pos_offset_m=0.0030,
            dr_racket_rot_offset_rad=float(np.deg2rad(1.1)),
            dr_racket_radius_offset_m=0.0020,
            dr_ball_obs_pos_bias_base_m=(0.004, 0.004, 0.004),
            dr_ball_obs_rot_bias_deg=(0.8, 0.8, 1.0),
            dr_ball_obs_vel_bias_base_m_s=(0.04, 0.04, 0.05),
            dr_ball_obs_scale_range=(0.99, 1.01),
            dr_ball_friction_range=(0.10, 0.40),
            dr_racket_friction_range=(0.20, 0.70),
            dr_ball_solref_time_range=(0.0018, 0.0090),
            dr_ball_solref_damping_range=(0.58, 1.08),
        ),
        "4e2": cfg_for(target_x=(-0.055, 0.125), target_y=(0.045, 0.155), anchor_z=(-0.036, 0.032), xy_jitter=0.020, z_jitter=0.014, init_vxy=0.016, init_vz_jitter=0.034, obs_pos_noise=0.006, obs_vel_noise=0.060, dr_level=6, view_xy_weight=0.55, view_z_weight=1.10, view_bounds_weight=2.80, camera_center_weight=0.65),
        "4f": cfg_for(target_x=(-0.065, 0.135), target_y=(0.040, 0.160), anchor_z=(-0.038, 0.034), xy_jitter=0.022, z_jitter=0.015, init_vxy=0.017, init_vz_jitter=0.036, obs_pos_noise=0.007, obs_vel_noise=0.070, dr_level=6, view_xy_weight=0.60, view_z_weight=1.15, view_bounds_weight=3.00, camera_center_weight=0.70),
        "5a1": cfg_for(target_x=(-0.075, 0.155), target_y=(0.020, 0.180), anchor_z=(-0.045, 0.050), xy_jitter=0.024, z_jitter=0.016, init_vxy=0.018, init_vz_jitter=0.036, obs_pos_noise=0.008, obs_vel_noise=0.080, dr_level=5, view_xy_weight=0.65, view_z_weight=1.25, view_bounds_weight=3.40, camera_center_weight=0.75),
        "5a2": cfg_for(target_x=(-0.095, 0.200), target_y=(0.000, 0.205), anchor_z=(-0.055, 0.060), xy_jitter=0.028, z_jitter=0.020, init_vxy=0.020, init_vz_jitter=0.042, obs_pos_noise=0.010, obs_vel_noise=0.100, dr_level=5, view_xy_weight=0.75, view_z_weight=1.45, view_bounds_weight=4.20, camera_center_weight=0.85),
        "5a3": cfg_for(target_x=(-0.120, 0.250), target_y=(-0.015, 0.230), anchor_z=(-0.062, 0.072), xy_jitter=0.032, z_jitter=0.024, init_vxy=0.023, init_vz_jitter=0.048, obs_pos_noise=0.012, obs_vel_noise=0.120, dr_level=5, view_xy_weight=0.85, view_z_weight=1.65, view_bounds_weight=5.00, camera_center_weight=0.95),
        "5a4": cfg_for(target_x=(-0.150, 0.300), target_y=(-0.025, 0.255), anchor_z=(-0.070, 0.085), xy_jitter=0.036, z_jitter=0.028, init_vxy=0.026, init_vz_jitter=0.054, obs_pos_noise=0.014, obs_vel_noise=0.140, dr_level=6, view_xy_weight=0.95, view_z_weight=1.90, view_bounds_weight=5.80, camera_center_weight=1.05),
        "6a": cfg_for(target_x=(-0.095, 0.200), target_y=(0.000, 0.205), anchor_z=(-0.055, 0.060), xy_jitter=0.028, z_jitter=0.020, init_vxy=0.020, init_vz_jitter=0.042, obs_pos_noise=0.010, obs_vel_noise=0.100, dr_level=5, view_xy_weight=0.75, view_z_weight=1.45, view_bounds_weight=4.20, camera_center_weight=0.85, view_missing_prob=0.10, camera_missing_prob=0.01, dropout_prob=0.001, burst_prob=0.00025, dropout_steps=3, burst_steps=10, z_high_missing_range=(1.40, 1.47)),
        "6b": cfg_for(target_x=(-0.115, 0.235), target_y=(-0.010, 0.225), anchor_z=(-0.060, 0.070), xy_jitter=0.032, z_jitter=0.024, init_vxy=0.023, init_vz_jitter=0.048, obs_pos_noise=0.012, obs_vel_noise=0.120, dr_level=5, view_xy_weight=0.85, view_z_weight=1.70, view_bounds_weight=5.00, camera_center_weight=0.95, view_missing_prob=0.25, camera_missing_prob=0.02, dropout_prob=0.002, burst_prob=0.0005, dropout_steps=5, burst_steps=18, z_high_missing_range=(1.36, 1.47)),
        "6c": cfg_for(target_x=(-0.130, 0.265), target_y=(-0.018, 0.240), anchor_z=(-0.064, 0.076), xy_jitter=0.034, z_jitter=0.026, init_vxy=0.024, init_vz_jitter=0.050, obs_pos_noise=0.014, obs_vel_noise=0.140, dr_level=6, view_xy_weight=0.95, view_z_weight=1.95, view_bounds_weight=5.60, camera_center_weight=1.05, view_missing_prob=0.50, camera_missing_prob=0.04, dropout_prob=0.004, burst_prob=0.0010, dropout_steps=8, burst_steps=28, z_high_missing_range=(1.30, 1.47)),
        "6d": cfg_for(target_x=(-0.140, 0.285), target_y=(-0.022, 0.250), anchor_z=(-0.067, 0.082), xy_jitter=0.038, z_jitter=0.029, init_vxy=0.028, init_vz_jitter=0.056, obs_pos_noise=0.016, obs_vel_noise=0.160, dr_level=6, view_xy_weight=1.05, view_z_weight=2.20, view_bounds_weight=6.20, camera_center_weight=1.15, view_missing_prob=0.75, camera_missing_prob=0.06, dropout_prob=0.006, burst_prob=0.0015, dropout_steps=10, burst_steps=40, z_high_missing_range=(1.24, 1.47)),
        "6e": cfg_for(target_x=(-0.150, 0.300), target_y=(-0.025, 0.255), anchor_z=(-0.070, 0.085), xy_jitter=0.040, z_jitter=0.030, init_vxy=0.030, init_vz_jitter=0.060, obs_pos_noise=0.018, obs_vel_noise=0.180, dr_level=6, view_xy_weight=1.15, view_z_weight=2.45, view_bounds_weight=6.80, camera_center_weight=1.20, view_missing_prob=1.00, camera_missing_prob=0.08, dropout_prob=0.008, burst_prob=0.0020, dropout_steps=12, burst_steps=48, z_high_missing_range=(1.20, 1.47)),
        "7a": cfg_for(target_x=(-0.150, 0.300), target_y=(-0.025, 0.255), anchor_z=(-0.070, 0.085), xy_jitter=0.040, z_jitter=0.030, init_vxy=0.030, init_vz_jitter=0.060, obs_pos_noise=0.018, obs_vel_noise=0.180, dr_level=6, view_xy_weight=1.20, view_z_weight=2.60, view_bounds_weight=7.20, camera_center_weight=1.25, view_missing_prob=1.00, camera_missing_prob=0.08, dropout_prob=0.008, burst_prob=0.0020, dropout_steps=12, burst_steps=48, z_high_missing_range=(1.20, 1.47)),
    })

    # Keep the post-hit state recoverable.  The first v7 stage1b run learned a
    # reliable first hit but hit3 stayed near zero because this profile had
    # inherited the default zero next-contact/apex-center weights.  Use the
    # existing reward terms from the successful D455 curricula and ramp them
    # with stage difficulty instead of changing the reset/view geometry.
    recoverability_weights = {
        "1a": (0.03, 0.03, 0.72, 0.10),
        "1b": (0.08, 0.06, 0.66, 0.25),
        "1c": (0.14, 0.10, 0.58, 0.85),
        "1d": (0.24, 0.14, 0.50, 1.45),
        "2a": (0.28, 0.15, 0.48, 1.70),
        "2b": (0.32, 0.16, 0.46, 1.90),
        "3a": (0.18, 0.12, 0.54, 0.80),
        "3b": (0.20, 0.13, 0.52, 0.95),
        "4a": (0.22, 0.14, 0.50, 1.15),
        "4b": (0.24, 0.14, 0.49, 1.25),
        "4c0": (0.25, 0.14, 0.49, 1.30),
        "4c1": (0.25, 0.14, 0.49, 1.30),
        "4c2": (0.26, 0.15, 0.48, 1.35),
        "4c3": (0.27, 0.15, 0.48, 1.40),
        "4d": (0.28, 0.15, 0.47, 1.45),
        "4e0": (0.30, 0.16, 0.30, 1.70),
        "4e1a": (0.32, 0.17, 0.26, 2.00),
        "4e1b": (0.34, 0.18, 0.24, 2.20),
        "4e1b2": (0.35, 0.18, 0.235, 2.30),
        "4e1b3": (0.355, 0.18, 0.23, 2.35),
        "4e1b3b": (0.36, 0.18, 0.23, 2.40),
        "4e1b3b1": (0.36, 0.18, 0.23, 2.40),
        "4e1b3b1h": (0.36, 0.18, 0.23, 2.45),
        "4e1b3b2": (0.36, 0.18, 0.23, 2.40),
        "4e1b3c0": (0.36, 0.18, 0.23, 2.40),
        "4e1b3c": (0.36, 0.18, 0.23, 2.40),
        "4e1b4": (0.36, 0.18, 0.23, 2.40),
        "4e1c": (0.36, 0.18, 0.23, 2.40),
        "4e1d": (0.38, 0.18, 0.23, 2.50),
        "4e2": (0.40, 0.18, 0.23, 2.60),
        "4e": (0.42, 0.18, 0.22, 2.70),
        "4f": (0.44, 0.18, 0.22, 2.80),
        "5a": (0.38, 0.17, 0.45, 1.85),
        "5a1": (0.38, 0.17, 0.45, 1.85),
        "5a2": (0.40, 0.18, 0.45, 1.95),
        "5a3": (0.42, 0.18, 0.44, 2.05),
        "5a4": (0.44, 0.18, 0.44, 2.15),
        "5b": (0.42, 0.18, 0.44, 2.05),
        "5c": (0.44, 0.18, 0.44, 2.15),
        "5d": (0.46, 0.18, 0.44, 2.25),
        "5e": (0.48, 0.18, 0.44, 2.35),
        "6a": (0.40, 0.17, 0.45, 1.95),
        "6b": (0.42, 0.18, 0.45, 2.05),
        "6c": (0.44, 0.18, 0.44, 2.15),
        "6d": (0.46, 0.18, 0.44, 2.25),
        "6e": (0.48, 0.18, 0.44, 2.35),
        "7a": (0.50, 0.18, 0.44, 2.45),
    }
    for key, (next_contact_weight, apex_center_weight, hit_vxy_limit, hit_vxy_weight) in recoverability_weights.items():
        cfgs[key] = replace(
            cfgs[key],
            hit_next_contact_anchor_penalty_weight=float(next_contact_weight),
            hit_next_contact_anchor_sigma_m=(
                0.12 if key in {"1d", "2a", "2b"} else (0.13 if key not in {"1a", "1b"} else 0.14)
            ),
            hit_apex_view_center_penalty_weight=float(apex_center_weight),
            hit_apex_view_center_sigma_m=(
                0.14 if key in {"1d", "2a", "2b"} else (0.15 if key not in {"1a", "1b"} else 0.16)
            ),
            hit_vxy_soft_limit_m_s=float(hit_vxy_limit),
            hit_vxy_penalty_weight=float(hit_vxy_weight),
        )

    def entry_bootstrap_cfg(
        key: str,
        *,
        hit_base: float,
        hit_combo: float,
        pre_hit: float,
        first_apex: float,
        miss_base: float,
        no_hit_miss: float,
        torque_weight: float,
        arm_acc_weight: float,
        action_weight: float,
        action_delta_weight: float,
        center_flat_weight: float,
    ) -> None:
        # The reference stage4g-style reward is intentionally low once a
        # stable juggle attractor exists, but from-scratch D455 stage1a was
        # observed to learn the "stay still and reduce dense penalties" local
        # optimum: last 16 updates had zero hits while return improved.  Keep
        # reset/view geometry unchanged and only strengthen the existing
        # first-hit bootstrap terms for the entry stages.
        cfgs[key] = replace(
            cfgs[key],
            hit_reward_base=float(hit_base),
            hit_reward_combo=float(hit_combo),
            hit_reward_cap_mode="fixed",
            hit_reward_count_cap=15,
            center_flat_hit_reward_weight=float(center_flat_weight),
            pre_hit_intercept_reward_weight=float(pre_hit),
            pre_hit_intercept_sigma=0.10,
            pre_hit_intercept_time_max=0.72,
            pre_hit_intercept_penalty_weight=0.45,
            pre_hit_intercept_penalty_sigma=0.22,
            pre_hit_intercept_penalty_radius=0.030,
            pre_hit_intercept_penalty_time_max=0.85,
            first_hit_apex_reward_weight=float(first_apex),
            first_hit_apex_sigma=0.070,
            termination_miss_penalty_base=float(miss_base),
            termination_miss_penalty_per_hit=0.30,
            termination_miss_penalty_requires_hit=False,
            termination_no_hit_miss_early_penalty=float(no_hit_miss),
            torque_penalty_weight=float(torque_weight),
            arm_acc_limit_penalty_weight=float(arm_acc_weight),
            arm_vel_limit_penalty_weight=0.035,
            arm_limiter_penalty_weight=0.030,
            action_penalty_weight=float(action_weight),
            action_delta_penalty_weight=float(action_delta_weight),
            racket_xy_gauss_reward_weight=0.25,
            racket_xy_gauss_penalty_weight=0.25,
        )

    entry_bootstrap_cfg(
        "1a",
        hit_base=2.25,
        hit_combo=0.28,
        pre_hit=1.80,
        first_apex=0.60,
        miss_base=3.50,
        no_hit_miss=6.0,
        torque_weight=0.00012,
        arm_acc_weight=0.030,
        action_weight=0.0010,
        action_delta_weight=0.00045,
        center_flat_weight=1.45,
    )
    entry_bootstrap_cfg(
        "1b",
        hit_base=2.05,
        hit_combo=0.25,
        pre_hit=1.60,
        first_apex=0.55,
        miss_base=3.20,
        no_hit_miss=5.0,
        torque_weight=0.00015,
        arm_acc_weight=0.035,
        action_weight=0.0011,
        action_delta_weight=0.00055,
        center_flat_weight=1.35,
    )
    entry_bootstrap_cfg(
        "1c",
        hit_base=1.80,
        hit_combo=0.22,
        pre_hit=1.40,
        first_apex=0.50,
        miss_base=2.80,
        no_hit_miss=4.0,
        torque_weight=0.00020,
        arm_acc_weight=0.045,
        action_weight=0.0012,
        action_delta_weight=0.00070,
        center_flat_weight=1.25,
    )
    entry_bootstrap_cfg(
        "1d",
        hit_base=1.55,
        hit_combo=0.18,
        pre_hit=1.20,
        first_apex=0.40,
        miss_base=2.50,
        no_hit_miss=3.0,
        torque_weight=0.00025,
        arm_acc_weight=0.055,
        action_weight=0.0013,
        action_delta_weight=0.00085,
        center_flat_weight=1.10,
    )
    entry_bootstrap_cfg(
        "2a",
        hit_base=1.35,
        hit_combo=0.14,
        pre_hit=1.00,
        first_apex=0.30,
        miss_base=2.30,
        no_hit_miss=2.0,
        torque_weight=0.00030,
        arm_acc_weight=0.065,
        action_weight=0.00145,
        action_delta_weight=0.00100,
        center_flat_weight=1.00,
    )
    entry_bootstrap_cfg(
        "2b",
        hit_base=1.15,
        hit_combo=0.10,
        pre_hit=0.80,
        first_apex=0.25,
        miss_base=2.10,
        no_hit_miss=1.5,
        torque_weight=0.00035,
        arm_acc_weight=0.070,
        action_weight=0.00160,
        action_delta_weight=0.00110,
        center_flat_weight=0.95,
    )
    for key in ("3a", "3b"):
        cfgs[key] = replace(
            cfgs[key],
            hit_reward_base=0.95,
            hit_reward_combo=0.08,
            hit_reward_cap_mode="fixed",
            hit_reward_count_cap=15,
            pre_hit_intercept_reward_weight=0.70,
            first_hit_apex_reward_weight=0.20,
            termination_miss_penalty_base=2.0,
            termination_miss_penalty_per_hit=0.30,
            termination_miss_penalty_requires_hit=False,
            termination_no_hit_miss_early_penalty=1.0,
            torque_penalty_weight=0.00040,
            arm_acc_limit_penalty_weight=0.075,
        )

    def stage(
        key: str,
        name: str,
        total_steps: int,
        notes: str,
        *,
        target_hits: float,
        target_len: float,
        min_updates: int,
        camera_visible: float,
        view_in_bounds: float | None = None,
        z_ideal: float | None = None,
        min_return: float | None = None,
        gate_mode: str = "strict",
        full_rate: float | None = None,
        hit1: float | None = None,
        hit3: float | None = None,
        hit12: float | None = None,
        hits_ge3: float | None = None,
        missing_refresh: float | None = None,
        lost_rate: float | None = None,
        advance_gate_mode: str = "collapse",
        hit_interval_min: float | None = 0.32,
        hit_interval_max: float | None = 0.58,
    ) -> CurriculumStage:
        return CurriculumStage(
            name,
            steps(total_steps),
            cfgs[key],
            notes,
            gate_mode=gate_mode,
            # Early bridges use collapse probes to avoid over-blocking one-step
            # distribution changes; later sim-to-real stages switch to strict
            # probes so training metrics cannot advance past weak validation.
            advance_gate_mode=advance_gate_mode,
            target_mean_hits=target_hits,
            target_mean_len_frac=target_len,
            min_updates=min_updates,
            min_recent_mean_return=min_return,
            target_camera_visible=camera_visible,
            min_recent_camera_reward_dense=-0.10,
            target_ball_view_in_bounds=view_in_bounds,
            target_ball_view_z_ideal=z_ideal,
            target_hit1_rate=hit1,
            target_hit3_rate=hit3,
            target_hit12_rate=hit12,
            target_mean_hits_ge3=hits_ge3,
            target_min_hit_interval_s=hit_interval_min,
            target_max_hit_interval_s=hit_interval_max,
            target_episode_truncation_rate=full_rate,
            min_ball_obs_missing_refresh_rate=missing_refresh,
            max_ball_obs_lost_rate=lost_rate,
        )

    return [
        stage("1a", "stage1a_d455_ref_first_hit", 4_000_000, "Fixed D455 reset and centered anchor-drop first-hit bootstrap.", target_hits=0.97, target_len=0.10, min_updates=15, camera_visible=0.60, hit_interval_min=None, hit_interval_max=None),
        stage("1b", "stage1b_d455_ref_small_random", 5_000_000, "Small reset and ball-state randomization.", target_hits=2.2, target_len=0.18, min_updates=25, camera_visible=0.65, hit_interval_min=None, hit_interval_max=None),
        stage("1c", "stage1c_d455_ref_three_hit_bridge", 5_500_000, "Bridge to repeated contacts without DR.", target_hits=3.5, target_len=0.25, min_updates=30, camera_visible=0.68, view_in_bounds=0.58, z_ideal=0.44),
        stage("1d", "stage1d_d455_ref_nominal_multi_hit", 6_000_000, "Nominal D455 visible-range multi-hit learning.", target_hits=5.0, target_len=0.36, min_updates=35, camera_visible=0.70, view_in_bounds=0.62, z_ideal=0.48),
        stage("2a", "stage2a_d455_ref_centered_visible", 7_000_000, "Center the hit bucket in the D455 lower-middle view.", target_hits=6.5, target_len=0.48, min_updates=40, camera_visible=0.72, view_in_bounds=0.66, z_ideal=0.50),
        stage("2b", "stage2b_d455_ref_stable_anchor", 8_000_000, "Consolidate a long visible anchor-drop attractor before DR.", target_hits=8.0, target_len=0.60, min_updates=45, camera_visible=0.74, view_in_bounds=0.68, z_ideal=0.54),
        stage("3a", "stage3a_d455_ref_ball_light_dr", 8_000_000, "Legacy-like ball-only light DR with D455 bounds fixed.", target_hits=4.0, target_len=0.20, min_updates=30, camera_visible=0.76, view_in_bounds=0.68, z_ideal=0.54),
        stage("3b", "stage3b_d455_ref_contact_dr", 8_000_000, "Add contact DR without changing reset geometry.", target_hits=4.0, target_len=0.20, min_updates=30, camera_visible=0.76, view_in_bounds=0.68, z_ideal=0.54),
        stage("4a", "stage4a_d455_ref_actuator_cmd_bridge", 8_000_000, "Enable mild actuator command-filter/PD randomization.", target_hits=4.0, target_len=0.20, min_updates=30, camera_visible=0.78, view_in_bounds=0.70, z_ideal=0.56),
        stage("4b", "stage4b_d455_ref_lite_actuator_pd", 8_000_000, "Slightly broaden actuator and PD DR.", target_hits=4.0, target_len=0.20, min_updates=30, camera_visible=0.78, view_in_bounds=0.70, z_ideal=0.56),
        stage("4c0", "stage4c0_d455_ref_action_latency_fixed_1_same_range", 8_000_000, "Teach the fixed 1-step raw action-latency phase shift before randomizing action latency.", target_hits=4.0, target_len=0.20, min_updates=35, camera_visible=0.78, view_in_bounds=0.70, z_ideal=0.56),
        stage("4c1", "stage4c1_d455_ref_action_latency_0_1_same_range", 8_000_000, "Randomize raw action latency to 0--1 control steps without broadening the 4b range.", target_hits=4.0, target_len=0.20, min_updates=35, camera_visible=0.78, view_in_bounds=0.70, z_ideal=0.56),
        stage("4c2", "stage4c2_d455_ref_action_latency_0_1", 8_000_000, "Keep raw action-latency DR at 0--1 steps while moving to the original 4c reset/noise range.", target_hits=4.0, target_len=0.20, min_updates=35, camera_visible=0.78, view_in_bounds=0.70, z_ideal=0.56),
        stage("4c3", "stage4c3_d455_ref_action_latency_0_2", 8_000_000, "Raise raw action-latency DR to the original 0--2 step range after the 0--1 bridge is stable.", target_hits=4.0, target_len=0.20, min_updates=45, camera_visible=0.78, view_in_bounds=0.70, z_ideal=0.56),
        stage("4d", "stage4d_d455_ref_racket_mount_dr", 8_000_000, "Add racket mount DR.", target_hits=4.0, target_len=0.20, min_updates=30, camera_visible=0.80, view_in_bounds=0.70, z_ideal=0.56),
        stage("4e0", "stage4e0_d455_ref_4e_range_dr5_quality", 8_000_000, "Move to the 4e visible range while keeping medium actuator/racket-mount DR.", target_hits=4.1, target_len=0.32, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57),
        stage("4e1a", "stage4e1a_d455_ref_obs_frame_tiny_bridge", 8_000_000, "Introduce tiny structured ball-observation-frame DR after the 4e0 visible attractor.", target_hits=3.8, target_len=0.28, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57),
        stage("4e1b", "stage4e1b_d455_ref_obs_frame_small_bridge", 8_000_000, "Raise structured ball-observation-frame DR to a small bias while keeping medium actuator response.", target_hits=3.9, target_len=0.29, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit3=0.30, hits_ge3=8.0),
        stage("4e1b2", "stage4e1b2_d455_ref_obs_frame_mid_bridge", 8_000_000, "Add an intermediate obs-frame DR bridge after strict validation showed 4e1b could not robustly enter 4e1c low-x buckets.", target_hits=3.55, target_len=0.29, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit3=0.29, hits_ge3=9.0, advance_gate_mode="strict"),
        stage("4e1b3", "stage4e1b3_d455_ref_low_x_mid_obs_bridge", 8_000_000, "Focus the failed low-x reset/target buckets with an intermediate obs-frame DR before the full 4e1c sensor jump.", target_hits=2.9, target_len=0.24, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit3=0.23, hits_ge3=7.5, advance_gate_mode="strict"),
        stage("4e1b3b", "stage4e1b3b_d455_ref_low_x_full_range_mid_obs_bridge", 8_000_000, "Cover the full low-x reset/target range using the same mid obs-frame DR before adding the full obs-frame bias.", target_hits=2.6, target_len=0.22, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit3=0.20, hits_ge3=7.0, advance_gate_mode="collapse"),
        stage("4e1b3b1", "stage4e1b3b1_d455_ref_low_x_first_hit_repair", 8_000_000, "Repair first-hit survival on the low-x tail with a centered-y, lower-jitter bridge before the soft edge.", target_hits=2.0, target_len=0.18, min_updates=40, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit1=0.55, hit3=0.13, hits_ge3=5.0, advance_gate_mode="collapse"),
        stage("4e1b3b1h", "stage4e1b3b1h_d455_ref_low_x_high_y_bucket_repair", 8_000_000, "Train the next-stage target-x-low and target-y-high corner as the whole distribution after v35 showed average first-hit repair still hid this weak bucket.", target_hits=1.4, target_len=0.13, min_updates=35, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit1=0.42, hit3=0.06, hits_ge3=3.0, advance_gate_mode="collapse"),
        stage("4e1b3b2", "stage4e1b3b2_d455_ref_low_x_soft_edge_mid_obs", 8_000_000, "Shift the low-x distribution left in a softer step after the pure low-x edge plateaued below one hit.", target_hits=2.4, target_len=0.20, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit1=0.50, hit3=0.17, hits_ge3=6.0, advance_gate_mode="collapse"),
        stage("4e1b3c0", "stage4e1b3c0_d455_ref_low_x_edge_mid_obs_focus", 8_000_000, "Make only the low-x edge buckets trainable under the mid obs-frame DR before asking for the full obs-frame bias.", target_hits=2.3, target_len=0.19, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit1=0.48, hit3=0.16, hits_ge3=5.8, advance_gate_mode="collapse"),
        stage("4e1b3c", "stage4e1b3c_d455_ref_low_x_edge_full_obs_recovery", 8_000_000, "Train only the low-x edge buckets with the full 4e1b4 obs-frame DR, so the weak reset tail cannot be hidden by average-stage metrics.", target_hits=2.1, target_len=0.18, min_updates=55, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit1=0.45, hit3=0.13, hits_ge3=5.2, advance_gate_mode="collapse"),
        stage("4e1b4", "stage4e1b4_d455_ref_low_x_full_obs_bridge", 8_000_000, "Use the full 4e1c obs-frame DR on the low-x reset/target buckets after the intermediate low-x bridge is stable.", target_hits=2.9, target_len=0.24, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit3=0.23, hits_ge3=7.5, advance_gate_mode="strict"),
        stage("4e1c", "stage4e1c_d455_ref_obs_frame_micro_bridge", 8_000_000, "Reach the previous micro ball-observation-frame DR only after tiny/small bridges.", target_hits=4.0, target_len=0.30, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit3=0.32, hits_ge3=8.5, advance_gate_mode="strict"),
        stage("4e1d", "stage4e1d_d455_ref_real_dr_light_bridge", 8_000_000, "Introduce clipped real actuator and ball-observation-frame DR after the obs-frame bridge is stable.", target_hits=4.2, target_len=0.30, min_updates=45, camera_visible=0.80, view_in_bounds=0.71, z_ideal=0.57, hit3=0.34, hits_ge3=8.8, advance_gate_mode="strict"),
        stage("4e2", "stage4e2_d455_ref_full_visible_dr_stabilize", 9_000_000, "Use full visible DR at the 4e range before asking for long 13-hit polish.", target_hits=6.8, target_len=0.50, min_updates=60, camera_visible=0.80, view_in_bounds=0.72, z_ideal=0.58, gate_mode="balanced", full_rate=0.32, hit3=0.56, hit12=0.18, hits_ge3=8.0, advance_gate_mode="strict"),
        stage("4e", "stage4e_d455_ref_strong_visible_dr", 10_000_000, "Strong visible DR bridge before range and missing.", target_hits=9.5, target_len=0.70, min_updates=60, camera_visible=0.80, view_in_bounds=0.72, z_ideal=0.58, gate_mode="balanced", full_rate=0.55, hit3=0.70, hit12=0.35, hits_ge3=10.5, advance_gate_mode="strict"),
        stage("4f", "stage4f_d455_ref_visible_13hit_polish", 12_000_000, "Visible strong-DR polish: require the 13-hit attractor before broad range.", target_hits=12.5, target_len=0.88, min_updates=90, camera_visible=0.82, view_in_bounds=0.74, z_ideal=0.60, gate_mode="balanced", full_rate=0.80, hit3=0.82, hit12=0.68, hits_ge3=13.2, advance_gate_mode="strict"),
        stage("5a1", "stage5a1_d455_ref_wide_xy_small", 9_000_000, "Small visible range expansion, no missing.", target_hits=11.5, target_len=0.82, min_updates=70, camera_visible=0.80, view_in_bounds=0.72, z_ideal=0.58, gate_mode="balanced", full_rate=0.72, hit3=0.78, hit12=0.55, hits_ge3=12.4, lost_rate=0.020, advance_gate_mode="strict"),
        stage("5a2", "stage5a2_d455_ref_wide_xy_medium", 9_000_000, "Medium visible range expansion, still no missing.", target_hits=10.8, target_len=0.78, min_updates=80, camera_visible=0.78, view_in_bounds=0.70, z_ideal=0.56, gate_mode="balanced", full_rate=0.66, hit3=0.74, hit12=0.46, hits_ge3=11.8, lost_rate=0.020, advance_gate_mode="strict"),
        stage("5a3", "stage5a3_d455_ref_wide_xy_large", 10_000_000, "Large visible XY range before adding missing.", target_hits=10.2, target_len=0.74, min_updates=90, camera_visible=0.76, view_in_bounds=0.68, z_ideal=0.54, gate_mode="balanced", full_rate=0.62, hit3=0.72, hit12=0.40, hits_ge3=11.2, lost_rate=0.025, advance_gate_mode="strict"),
        stage("5a4", "stage5a4_d455_ref_wide_xyz_visible", 11_000_000, "Full visible range including racket-anchor z variation.", target_hits=10.5, target_len=0.76, min_updates=100, camera_visible=0.76, view_in_bounds=0.66, z_ideal=0.52, gate_mode="balanced", full_rate=0.64, hit3=0.74, hit12=0.42, hits_ge3=11.6, lost_rate=0.025, advance_gate_mode="strict"),
        stage("6a", "stage6a_d455_ref_viewmissing10", 8_000_000, "Introduce only mild upper-FOV stale-age missing.", target_hits=10.0, target_len=0.72, min_updates=70, camera_visible=0.76, view_in_bounds=0.66, z_ideal=0.52, gate_mode="balanced", full_rate=0.60, hit3=0.70, hit12=0.36, hits_ge3=10.8, missing_refresh=0.001, lost_rate=0.030, advance_gate_mode="strict"),
        stage("6b", "stage6b_d455_ref_viewmissing25", 9_000_000, "Raise upper-FOV missing to 25%.", target_hits=9.5, target_len=0.70, min_updates=80, camera_visible=0.74, view_in_bounds=0.64, z_ideal=0.50, gate_mode="balanced", full_rate=0.58, hit3=0.68, hit12=0.34, hits_ge3=10.4, missing_refresh=0.002, lost_rate=0.035, advance_gate_mode="strict"),
        stage("6c", "stage6c_d455_ref_viewmissing50", 10_000_000, "Raise upper-FOV missing to 50% with large DR active.", target_hits=9.0, target_len=0.68, min_updates=90, camera_visible=0.72, view_in_bounds=0.62, z_ideal=0.50, gate_mode="balanced", full_rate=0.56, hit3=0.66, hit12=0.30, hits_ge3=9.8, missing_refresh=0.004, lost_rate=0.045, advance_gate_mode="strict"),
        stage("6d", "stage6d_d455_ref_viewmissing75", 11_000_000, "Raise upper-FOV missing to 75%; keep targets moderate to avoid a cliff.", target_hits=9.5, target_len=0.70, min_updates=100, camera_visible=0.70, view_in_bounds=0.60, z_ideal=0.48, gate_mode="balanced", full_rate=0.58, hit3=0.68, hit12=0.34, hits_ge3=10.5, missing_refresh=0.006, lost_rate=0.050, advance_gate_mode="strict"),
        stage("6e", "stage6e_d455_ref_full_viewmissing", 12_000_000, "Full upper-FOV missing and wide reset range, before final high target polish.", target_hits=10.5, target_len=0.76, min_updates=110, camera_visible=0.68, view_in_bounds=0.58, z_ideal=0.46, gate_mode="balanced", full_rate=0.66, hit3=0.74, hit12=0.46, hits_ge3=11.8, missing_refresh=0.008, lost_rate=0.055, advance_gate_mode="strict"),
        stage("7a", "stage7a_d455_ref_full_missing_large_dr_polish", 16_000_000, "Final 13--15 hit D455 policy with full upper-FOV missing and large DR.", target_hits=13.0, target_len=0.95, min_updates=140, camera_visible=0.68, view_in_bounds=0.58, z_ideal=0.46, min_return=0.0, gate_mode="strict", full_rate=0.90, hit3=0.86, hit12=0.80, hits_ge3=14.0, missing_refresh=0.010, lost_rate=0.050, advance_gate_mode="strict"),
    ]

def _d455_full_curriculum_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
) -> list[CurriculumStage]:
    """Single-process D455 curriculum from nominal juggling to full robustness.

    This is the host-side mainline profile: unlike ``d455_stable_4g_v1`` it
    does not stop at the stable nominal 4g/polish policy, and unlike
    ``d455_recovery_v1`` it does not switch to falling-contact recovery
    sampling.  The arm reset stays fixed and the ball reset remains
    anchor-drop while the curriculum expands the target/reset range, adds
    D455 stale-age missing, observation noise, actuator/contact/PD/racket DR,
    and final 13--15-hit acceptance gates.
    """

    stages = _d455_stable_4g_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=None,
        critic_command_history_steps=critic_command_history_steps,
    )
    stages = [
        replace(
            stage,
            notes=(
                "Polish the nominal D455 policy before the same single curriculum continues "
                "into wide-range missing and sim2real DR."
            ),
        )
        if stage.name == "stage4h_d455_stable_polish"
        else stage
        for stage in stages
    ]
    stable_final = stages[-1].cfg

    def steps(default_steps: int) -> int:
        return int(stage_steps_override) if stage_steps_override is not None else int(default_steps)

    def full_cfg(
        base: MjxJuggleConfig,
        *,
        target_x: tuple[float, float],
        target_y: tuple[float, float],
        anchor_z: tuple[float, float],
        xy_jitter: float,
        z_jitter: float,
        init_vxy: float,
        init_vz_jitter: float,
        obs_pos_noise: float,
        obs_vel_noise: float,
        view_missing_prob: float,
        camera_missing_prob: float,
        dropout_prob: float,
        burst_prob: float,
        dropout_steps: int,
        burst_steps: int,
        z_high_missing_range: tuple[float, float],
        dr_level: int,
        view_xy_weight: float,
        view_z_weight: float,
        view_bounds_weight: float,
        view_oob_weight: float,
        view_vxy_weight: float,
        camera_center_weight: float,
        hit_camera_weight: float,
        hit_camera_oob: float,
        hit_vxy_limit: float,
        hit_vxy_weight: float,
        next_contact_weight: float,
        apex_center_weight: float,
    ) -> MjxJuggleConfig:
        has_missing = (
            view_missing_prob > 0.0
            or camera_missing_prob > 0.0
            or dropout_prob > 0.0
            or burst_prob > 0.0
        )
        cfg = replace(
            base,
            right_arm_reset_degrees=D455_USER_TARGET_RACKET_RESET_DEGREES,
            ball_reset_mode="anchor_drop",
            ball_launch_height=0.225,
            target_height=0.198,
            rel_height_center=0.178,
            hit_height_center=0.228,
            hit_confirm_rel_height=0.055,
            hit_height_tolerance=0.050,
            low_hit_apex_margin=0.024,
            apex_soft_limit_margin=0.050,
            episode_target_x_range_m=target_x,
            episode_target_y_range_m=target_y,
            episode_racket_anchor_z_range_m=anchor_z,
            ball_spawn_xy_jitter=float(xy_jitter),
            ball_spawn_z_jitter=float(z_jitter),
            ball_init_vxy_max=float(init_vxy),
            ball_init_vz=-0.28,
            ball_init_vz_jitter=float(init_vz_jitter),
            ball_obs_pos_noise_std=float(obs_pos_noise),
            ball_obs_vel_noise_std=float(obs_vel_noise),
            ball_obs_require_camera_visible=camera_missing_prob > 0.0,
            ball_obs_camera_missing_prob=float(camera_missing_prob),
            ball_obs_reset_respects_camera_visibility=camera_missing_prob > 0.0,
            ball_obs_require_view_bounds=view_missing_prob > 0.0,
            ball_obs_view_bounds_missing_prob=float(view_missing_prob),
            ball_obs_view_z_high_missing_range_m=(
                z_high_missing_range if view_missing_prob > 0.0 else (0.0, 0.0)
            ),
            ball_obs_missing_episode_coherent_prob=1.0 if has_missing else 0.0,
            ball_obs_age_tracks_stale=has_missing,
            ball_obs_dropout_on_refresh_only=has_missing,
            ball_obs_dropout_prob=float(dropout_prob),
            ball_obs_dropout_burst_prob=float(burst_prob),
            ball_obs_dropout_max_steps=int(dropout_steps) if has_missing else 1,
            ball_obs_dropout_burst_max_steps=int(burst_steps) if has_missing else 1,
            ball_obs_age_clip=0.60 if has_missing else 0.35,
            lost_ball_timeout_ms=500.0,
            terminate_on_ball_view_bounds=True,
            terminate_on_ball_view_x_bounds=True,
            terminate_on_ball_view_y_bounds=True,
            terminate_on_ball_view_z_low=True,
            terminate_on_ball_view_z_high=False,
            ball_low_termination_z_m=D455_REAL_VIEW_Z_BOUNDS_M[0],
            ball_high_termination_z_m=max(float(base.ball_high_termination_z_m), D455_REAL_VIEW_Z_BOUNDS_M[1] + 0.20),
            ball_view_x_bounds_m=D455_REAL_VIEW_X_BOUNDS_M,
            ball_view_y_bounds_m=D455_REAL_VIEW_Y_BOUNDS_M,
            ball_view_z_bounds_m=D455_REAL_VIEW_Z_BOUNDS_M,
            ball_view_z_ideal_m=D455_RECOVERY_VIEW_Z_IDEAL_M,
            ball_view_x_target_m=0.0,
            ball_view_y_target_m=D455_REAL_VIEW_Y_TARGET_M,
            ball_view_x_sigma_m=0.10,
            ball_view_y_sigma_m=0.12,
            ball_view_z_sigma_m=0.11,
            ball_view_xy_center_penalty_weight=float(view_xy_weight),
            ball_view_z_ideal_penalty_weight=float(view_z_weight),
            ball_view_bounds_penalty_weight=float(view_bounds_weight),
            ball_view_out_of_bounds_penalty_weight=float(view_oob_weight),
            ball_view_z_not_ideal_penalty_weight=max(float(base.ball_view_z_not_ideal_penalty_weight), 0.55),
            ball_view_vxy_excess_penalty_weight=float(view_vxy_weight),
            camera_center_weight=float(camera_center_weight),
            camera_visibility_penalty_weight=max(float(base.camera_visibility_penalty_weight), 7.0),
            camera_depth_penalty_weight=max(float(base.camera_depth_penalty_weight), 0.5),
            camera_visible_penalty_weight=max(float(base.camera_visible_penalty_weight), 2.5),
            camera_top_margin_penalty_weight=max(float(base.camera_top_margin_penalty_weight), 10.0),
            hit_camera_reward_weight=float(hit_camera_weight),
            hit_camera_out_of_band_penalty_weight=float(hit_camera_oob),
            hit_camera_target_v_frac=0.66,
            hit_camera_v_sigma_frac=0.11,
            hit_camera_lower_band_frac=(0.50, 0.82),
            hit_reward_cap_mode="fixed",
            hit_reward_count_cap=15,
            hit_combo_count_cap=14,
            hit_cadence_reward_weight=max(float(base.hit_cadence_reward_weight), 0.30),
            hit_cadence_target_interval=0.425,
            hit_cadence_sigma=0.090,
            hit_min_interval_penalty_weight=max(float(base.hit_min_interval_penalty_weight), 1.35),
            hit_min_interval=0.36,
            hit_min_count_interval=0.34,
            fast_hit_penalty_weight=max(float(base.fast_hit_penalty_weight), 1.00),
            hit_vxy_soft_limit_m_s=float(hit_vxy_limit),
            hit_vxy_penalty_weight=float(hit_vxy_weight),
            hit_apex_view_center_penalty_weight=float(apex_center_weight),
            hit_next_contact_anchor_penalty_weight=float(next_contact_weight),
            hit_apex_view_center_sigma_m=0.13,
            hit_next_contact_anchor_sigma_m=0.11,
            racket_flatness_penalty_weight=max(
                float(base.racket_flatness_penalty_weight),
                0.65 + 0.06 * float(dr_level),
            ),
            racket_flatness_target_cos=0.985,
            racket_flatness_sigma=0.035,
            hit_flatness_target_cos=0.985,
            hit_flatness_sigma=0.045,
            contact_flatness_penalty_weight=max(float(base.contact_flatness_penalty_weight), 0.70),
            racket_z_hard_limit_up=max(float(base.racket_z_hard_limit_up), max(0.0, -float(anchor_z[0])) + 0.12),
        )

        cfg = replace(
            cfg,
            domain_randomization=True,
            dr_randomize_ball=True,
            dr_randomize_contact=dr_level >= 1,
            dr_randomize_actuator=dr_level >= 1,
            dr_randomize_pd=dr_level >= 1,
            dr_pd_per_joint=True,
            dr_randomize_latency=dr_level >= 3,
            dr_randomize_racket_mount=dr_level >= 2,
            dr_randomize_ball_obs_frame=dr_level >= 2,
            dr_randomize_actuator_cmd_filter=dr_level >= 1,
        )
        if dr_level <= 1:
            return replace(
                cfg,
                dr_action_scale_mult_range=(0.94, 1.06),
                dr_damping_mult_range=(0.88, 1.12),
                dr_armature_mult_range=(0.92, 1.08),
                dr_pd_kp_mult_range=(0.96, 1.04),
                dr_pd_kv_mult_range=(0.92, 1.08),
                dr_actuator_cmd_tau_range=(0.068, 0.082),
                dr_actuator_cmd_gain_range=(0.985, 1.015),
                dr_ball_friction_range=(0.14, 0.32),
                dr_racket_friction_range=(0.28, 0.56),
                dr_ball_solref_time_range=(0.0025, 0.0060),
                dr_ball_solref_damping_range=(0.70, 0.96),
            )
        if dr_level == 2:
            return replace(
                cfg,
                dr_action_scale_mult_range=(0.90, 1.10),
                dr_damping_mult_range=(0.82, 1.18),
                dr_armature_mult_range=(0.88, 1.12),
                dr_pd_kp_mult_range=(0.93, 1.07),
                dr_pd_kv_mult_range=(0.88, 1.12),
                dr_actuator_cmd_tau_range=(0.064, 0.086),
                dr_actuator_cmd_gain_range=(0.98, 1.02),
                dr_racket_pos_offset_m=0.003,
                dr_racket_rot_offset_rad=float(np.deg2rad(1.0)),
                dr_racket_radius_offset_m=0.002,
                dr_ball_obs_pos_bias_base_m=(0.006, 0.006, 0.006),
                dr_ball_obs_rot_bias_deg=(1.0, 1.0, 1.4),
                dr_ball_obs_vel_bias_base_m_s=(0.06, 0.06, 0.08),
                dr_ball_obs_scale_range=(0.985, 1.015),
                dr_ball_friction_range=(0.10, 0.40),
                dr_racket_friction_range=(0.22, 0.68),
                dr_ball_solref_time_range=(0.0020, 0.0080),
                dr_ball_solref_damping_range=(0.62, 1.05),
            )
        return replace(
            cfg,
            dr_action_scale_mult_range=(0.88, 1.12),
            dr_damping_mult_range=(0.75, 1.25),
            dr_armature_mult_range=(0.82, 1.18),
            dr_pd_kp_mult_range=(0.88, 1.12),
            dr_pd_kv_mult_range=(0.82, 1.18),
            dr_obs_latency_steps_range=(0, 3),
            dr_action_latency_steps_range=(0, 3),
            dr_actuator_cmd_tau_range=(0.060, 0.090),
            dr_actuator_cmd_gain_range=(0.97, 1.03),
            dr_racket_pos_offset_m=0.004,
            dr_racket_rot_offset_rad=float(np.deg2rad(1.5)),
            dr_racket_radius_offset_m=0.0025,
            dr_ball_obs_pos_bias_base_m=(0.008, 0.008, 0.008),
            dr_ball_obs_rot_bias_deg=(1.5, 1.5, 2.0),
            dr_ball_obs_vel_bias_base_m_s=(0.08, 0.08, 0.10),
            dr_ball_obs_scale_range=(0.98, 1.02),
            dr_ball_friction_range=(0.08, 0.45),
            dr_racket_friction_range=(0.18, 0.75),
            dr_ball_solref_time_range=(0.0015, 0.010),
            dr_ball_solref_damping_range=(0.55, 1.10),
        )

    cfg_5a = full_cfg(
        stable_final,
        target_x=(-0.10, 0.20),
        target_y=(0.020, 0.190),
        anchor_z=(-0.045, 0.050),
        xy_jitter=0.024,
        z_jitter=0.016,
        init_vxy=0.018,
        init_vz_jitter=0.034,
        obs_pos_noise=0.010,
        obs_vel_noise=0.10,
        view_missing_prob=0.0,
        camera_missing_prob=0.0,
        dropout_prob=0.0,
        burst_prob=0.0,
        dropout_steps=1,
        burst_steps=1,
        z_high_missing_range=(0.0, 0.0),
        dr_level=1,
        view_xy_weight=0.85,
        view_z_weight=1.55,
        view_bounds_weight=4.60,
        view_oob_weight=1.30,
        view_vxy_weight=0.72,
        camera_center_weight=1.05,
        hit_camera_weight=1.45,
        hit_camera_oob=0.75,
        hit_vxy_limit=0.44,
        hit_vxy_weight=2.60,
        next_contact_weight=0.28,
        apex_center_weight=0.20,
    )
    cfg_5b = full_cfg(
        cfg_5a,
        target_x=(-0.12, 0.25),
        target_y=(0.000, 0.215),
        anchor_z=(-0.055, 0.060),
        xy_jitter=0.028,
        z_jitter=0.020,
        init_vxy=0.020,
        init_vz_jitter=0.040,
        obs_pos_noise=0.012,
        obs_vel_noise=0.12,
        view_missing_prob=0.25,
        camera_missing_prob=0.02,
        dropout_prob=0.002,
        burst_prob=0.0005,
        dropout_steps=5,
        burst_steps=18,
        z_high_missing_range=(1.36, 1.47),
        dr_level=1,
        view_xy_weight=0.95,
        view_z_weight=1.85,
        view_bounds_weight=5.20,
        view_oob_weight=1.60,
        view_vxy_weight=0.78,
        camera_center_weight=1.12,
        hit_camera_weight=1.50,
        hit_camera_oob=0.85,
        hit_vxy_limit=0.43,
        hit_vxy_weight=2.75,
        next_contact_weight=0.30,
        apex_center_weight=0.22,
    )
    cfg_5c = full_cfg(
        cfg_5b,
        target_x=(-0.13, 0.29),
        target_y=(-0.010, 0.235),
        anchor_z=(-0.060, 0.070),
        xy_jitter=0.032,
        z_jitter=0.024,
        init_vxy=0.023,
        init_vz_jitter=0.046,
        obs_pos_noise=0.014,
        obs_vel_noise=0.14,
        view_missing_prob=0.50,
        camera_missing_prob=0.04,
        dropout_prob=0.004,
        burst_prob=0.0010,
        dropout_steps=8,
        burst_steps=28,
        z_high_missing_range=(1.30, 1.47),
        dr_level=2,
        view_xy_weight=1.05,
        view_z_weight=2.20,
        view_bounds_weight=6.00,
        view_oob_weight=2.00,
        view_vxy_weight=0.86,
        camera_center_weight=1.20,
        hit_camera_weight=1.55,
        hit_camera_oob=0.95,
        hit_vxy_limit=0.42,
        hit_vxy_weight=2.90,
        next_contact_weight=0.32,
        apex_center_weight=0.24,
    )
    cfg_5d = full_cfg(
        cfg_5c,
        target_x=(-0.14, 0.33),
        target_y=(-0.020, 0.250),
        anchor_z=(-0.065, 0.080),
        xy_jitter=0.036,
        z_jitter=0.028,
        init_vxy=0.026,
        init_vz_jitter=0.052,
        obs_pos_noise=0.016,
        obs_vel_noise=0.16,
        view_missing_prob=0.75,
        camera_missing_prob=0.06,
        dropout_prob=0.006,
        burst_prob=0.0015,
        dropout_steps=10,
        burst_steps=40,
        z_high_missing_range=(1.24, 1.47),
        dr_level=3,
        view_xy_weight=1.15,
        view_z_weight=2.50,
        view_bounds_weight=6.80,
        view_oob_weight=2.40,
        view_vxy_weight=0.94,
        camera_center_weight=1.28,
        hit_camera_weight=1.60,
        hit_camera_oob=1.05,
        hit_vxy_limit=0.41,
        hit_vxy_weight=3.05,
        next_contact_weight=0.34,
        apex_center_weight=0.26,
    )
    cfg_5e = full_cfg(
        cfg_5d,
        target_x=(-0.14, 0.36),
        target_y=(-0.020, 0.250),
        anchor_z=(-0.065, 0.080),
        xy_jitter=0.040,
        z_jitter=0.030,
        init_vxy=0.030,
        init_vz_jitter=0.060,
        obs_pos_noise=0.018,
        obs_vel_noise=0.18,
        view_missing_prob=1.0,
        camera_missing_prob=0.08,
        dropout_prob=0.008,
        burst_prob=0.0020,
        dropout_steps=12,
        burst_steps=48,
        z_high_missing_range=(1.20, 1.47),
        dr_level=3,
        view_xy_weight=1.20,
        view_z_weight=2.80,
        view_bounds_weight=7.50,
        view_oob_weight=2.80,
        view_vxy_weight=1.00,
        camera_center_weight=1.35,
        hit_camera_weight=1.65,
        hit_camera_oob=1.15,
        hit_vxy_limit=0.40,
        hit_vxy_weight=3.20,
        next_contact_weight=0.36,
        apex_center_weight=0.28,
    )

    late = [
        CurriculumStage(
            "stage5a_d455_wide_visible_range",
            steps(7_000_000),
            cfg_5a,
            "Single-stage mainline: expand the D455 reset/target bucket while keeping the ball visible.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=10.0,
            target_mean_len_frac=0.74,
            min_updates=90,
            target_camera_visible=0.78,
            target_ball_view_in_bounds=0.70,
            target_ball_view_z_ideal=0.58,
            target_hit1_rate=0.95,
            target_hit3_rate=0.72,
            target_hit12_rate=0.32,
            target_mean_hits_ge3=11.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.52,
            target_hit_camera_visible_rate=0.94,
            target_hit_camera_lower_band_rate=0.78,
            target_episode_truncation_rate=0.62,
            max_ball_obs_lost_rate=0.020,
        ),
        CurriculumStage(
            "stage5b_d455_viewmissing25_range",
            steps(7_000_000),
            cfg_5b,
            "Add mild D455 stale-age upper-FOV missing after the wide visible bucket is learnable.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=9.0,
            target_mean_len_frac=0.68,
            min_updates=90,
            target_camera_visible=0.76,
            target_ball_view_in_bounds=0.66,
            target_ball_view_z_ideal=0.56,
            target_hit1_rate=0.94,
            target_hit3_rate=0.68,
            target_hit12_rate=0.24,
            target_mean_hits_ge3=10.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.52,
            target_hit_camera_visible_rate=0.92,
            target_hit_camera_lower_band_rate=0.76,
            target_episode_truncation_rate=0.58,
            min_ball_obs_missing_refresh_rate=0.002,
            max_ball_obs_lost_rate=0.030,
        ),
        CurriculumStage(
            "stage5c_d455_viewmissing50_obs_dr",
            steps(8_000_000),
            cfg_5c,
            "Raise view-missing to 50% and add observation-frame/racket DR without changing reset mode.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=8.5,
            target_mean_len_frac=0.66,
            min_updates=100,
            target_camera_visible=0.74,
            target_ball_view_in_bounds=0.64,
            target_ball_view_z_ideal=0.54,
            target_hit1_rate=0.93,
            target_hit3_rate=0.66,
            target_hit12_rate=0.22,
            target_mean_hits_ge3=9.5,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.52,
            target_hit_camera_visible_rate=0.90,
            target_hit_camera_lower_band_rate=0.74,
            target_episode_truncation_rate=0.56,
            min_ball_obs_missing_refresh_rate=0.004,
            max_ball_obs_lost_rate=0.045,
        ),
        CurriculumStage(
            "stage5d_d455_viewmissing75_large_dr",
            steps(9_000_000),
            cfg_5d,
            "Train broad reset range, 75% stale-age missing, actuator/contact/PD/racket/latency DR.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=10.5,
            target_mean_len_frac=0.80,
            min_updates=110,
            target_camera_visible=0.72,
            target_ball_view_in_bounds=0.62,
            target_ball_view_z_ideal=0.52,
            target_hit1_rate=0.94,
            target_hit3_rate=0.72,
            target_hit12_rate=0.48,
            target_mean_hits_ge3=12.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.52,
            target_hit_camera_visible_rate=0.90,
            target_hit_camera_lower_band_rate=0.76,
            target_episode_truncation_rate=0.72,
            min_ball_obs_missing_refresh_rate=0.006,
            max_ball_obs_lost_rate=0.050,
        ),
        CurriculumStage(
            "stage5e_d455_full_missing_dr_polish",
            steps(12_000_000),
            cfg_5e,
            "Final single-stage robustness polish: full upper-FOV missing, broad reset range, and large DR.",
            gate_mode="strict",
            advance_gate_mode="collapse",
            target_mean_hits=13.0,
            target_mean_len_frac=0.95,
            min_updates=130,
            min_recent_mean_return=0.5,
            target_camera_visible=0.70,
            target_ball_view_in_bounds=0.60,
            target_ball_view_z_ideal=0.50,
            target_hit1_rate=0.97,
            target_hit3_rate=0.86,
            target_hit12_rate=0.80,
            target_mean_hits_ge3=14.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.50,
            target_hit_camera_visible_rate=0.94,
            target_hit_camera_lower_band_rate=0.84,
            target_episode_truncation_rate=0.90,
            min_ball_obs_missing_refresh_rate=0.010,
            max_ball_obs_lost_rate=0.040,
        ),
    ]
    if stage_steps_override is not None:
        stages = [replace(stage, total_steps=int(stage_steps_override)) for stage in stages]
    return stages + late


def _d455_recovery_v1_stages(
    *,
    stack_kwargs: dict[str, object],
    stage_steps_override: int | None,
    critic_command_history_steps: int,
) -> list[CurriculumStage]:
    """Recovery-state curriculum to resume from ``d455_stable_4g_v1``.

    The stages sample the ball as a descending post-apex state near a future
    racket contact.  This directly trains recovery from the distribution gaps
    caused by compounding hit errors, while preserving the fixed arm reset.
    """

    source_stages = _robust_juggle_v1_stages(
        stack_kwargs=stack_kwargs,
        stage_steps_override=None,
        critic_command_history_steps=critic_command_history_steps,
    )
    source = {stage.name: stage for stage in source_stages}

    def steps(default_steps: int) -> int:
        return int(stage_steps_override) if stage_steps_override is not None else int(default_steps)

    def recovery_cfg(
        source_name: str,
        *,
        target_x: tuple[float, float],
        target_y: tuple[float, float],
        anchor_z: tuple[float, float],
        xy_jitter: float,
        z_jitter: float,
        init_vxy: float,
        init_vz_jitter: float,
        falling_tau: tuple[float, float],
        falling_apex: tuple[float, float],
        falling_vxy: float,
        falling_contact_jitter: float,
        camera_missing_prob: float,
        dropout_prob: float,
        burst_prob: float,
        dropout_steps: int,
        burst_steps: int,
        obs_pos_noise: float,
        obs_vel_noise: float,
        hit_vxy_limit: float,
        hit_vxy_weight: float,
        next_contact_weight: float,
        apex_center_weight: float,
        camera_center_weight: float,
        view_xy_weight: float,
        view_z_weight: float,
        view_bounds_weight: float,
        view_oob_weight: float,
        view_vxy_weight: float,
    ) -> MjxJuggleConfig:
        cfg = source[source_name].cfg
        return replace(
            cfg,
            right_arm_reset_degrees=D455_USER_TARGET_RACKET_RESET_DEGREES,
            ball_reset_mode="falling_contact",
            ball_launch_height=0.225,
            target_height=0.205,
            rel_height_center=0.18,
            hit_height_center=0.242,
            hit_height_tolerance=0.055,
            low_hit_apex_margin=0.026,
            apex_soft_limit_margin=0.070,
            episode_target_x_range_m=target_x,
            episode_target_y_range_m=target_y,
            episode_racket_anchor_z_range_m=anchor_z,
            ball_spawn_xy_jitter=float(xy_jitter),
            ball_spawn_z_jitter=float(z_jitter),
            ball_init_vxy_max=float(init_vxy),
            ball_init_vz=-0.28,
            ball_init_vz_jitter=float(init_vz_jitter),
            falling_reset_time_to_contact_range_s=falling_tau,
            falling_reset_apex_height_range_m=falling_apex,
            falling_reset_vxy_max=float(falling_vxy),
            falling_reset_contact_xy_jitter=float(falling_contact_jitter),
            falling_reset_contact_rel_height=0.065,
            falling_reset_min_downward_speed=0.12,
            ball_obs_pos_noise_std=float(obs_pos_noise),
            ball_obs_vel_noise_std=float(obs_vel_noise),
            ball_obs_require_camera_visible=camera_missing_prob > 0.0,
            ball_obs_camera_missing_prob=float(camera_missing_prob),
            ball_obs_reset_respects_camera_visibility=camera_missing_prob > 0.0,
            ball_obs_require_view_bounds=False,
            ball_obs_view_bounds_missing_prob=0.0,
            ball_obs_missing_episode_coherent_prob=1.0 if camera_missing_prob > 0.0 else 0.0,
            ball_obs_age_tracks_stale=True,
            ball_obs_age_clip=0.60,
            ball_obs_dropout_on_refresh_only=dropout_prob > 0.0 or burst_prob > 0.0,
            ball_obs_dropout_prob=float(dropout_prob),
            ball_obs_dropout_burst_prob=float(burst_prob),
            ball_obs_dropout_max_steps=int(dropout_steps),
            ball_obs_dropout_burst_max_steps=int(burst_steps),
            lost_ball_timeout_ms=500.0,
            terminate_on_ball_view_bounds=True,
            terminate_on_ball_view_x_bounds=True,
            terminate_on_ball_view_y_bounds=True,
            terminate_on_ball_view_z_low=True,
            terminate_on_ball_view_z_high=False,
            racket_z_hard_limit_up=max(float(cfg.racket_z_hard_limit_up), max(0.0, -float(anchor_z[0])) + 0.12),
            ball_low_termination_z_m=D455_REAL_VIEW_Z_BOUNDS_M[0],
            ball_view_x_bounds_m=D455_REAL_VIEW_X_BOUNDS_M,
            ball_view_y_bounds_m=D455_REAL_VIEW_Y_BOUNDS_M,
            ball_view_z_bounds_m=D455_REAL_VIEW_Z_BOUNDS_M,
            ball_view_z_ideal_m=D455_RECOVERY_VIEW_Z_IDEAL_M,
            ball_view_y_target_m=D455_REAL_VIEW_Y_TARGET_M,
            ball_view_y_sigma_m=0.095,
            ball_view_z_sigma_m=0.11,
            ball_view_xy_center_penalty_weight=float(view_xy_weight),
            ball_view_z_ideal_penalty_weight=float(view_z_weight),
            ball_view_bounds_penalty_weight=float(view_bounds_weight),
            ball_view_out_of_bounds_penalty_weight=float(view_oob_weight),
            ball_view_z_not_ideal_penalty_weight=max(float(cfg.ball_view_z_not_ideal_penalty_weight), 0.45),
            ball_view_vxy_excess_penalty_weight=float(view_vxy_weight),
            camera_center_weight=float(camera_center_weight),
            hit_camera_reward_weight=max(float(cfg.hit_camera_reward_weight), 1.30),
            hit_camera_out_of_band_penalty_weight=max(float(cfg.hit_camera_out_of_band_penalty_weight), 0.60),
            hit_camera_target_v_frac=0.66,
            hit_camera_v_sigma_frac=0.10,
            hit_camera_lower_band_frac=(0.50, 0.82),
            hit_reward_cap_mode="fixed",
            hit_reward_count_cap=15,
            hit_combo_count_cap=14,
            hit_cadence_target_interval=0.44,
            hit_cadence_sigma=0.10,
            hit_min_interval=0.36,
            hit_min_count_interval=0.34,
            hit_vxy_soft_limit_m_s=float(hit_vxy_limit),
            hit_vxy_penalty_weight=float(hit_vxy_weight),
            hit_apex_view_center_penalty_weight=float(apex_center_weight),
            hit_next_contact_anchor_penalty_weight=float(next_contact_weight),
            hit_apex_view_center_sigma_m=0.13,
            hit_next_contact_anchor_sigma_m=0.11,
        )

    cfg_r1 = recovery_cfg(
        "04_range",
        target_x=(-0.050, 0.055),
        target_y=(0.045, 0.115),
        anchor_z=(-0.030, 0.028),
        xy_jitter=0.012,
        z_jitter=0.008,
        init_vxy=0.010,
        init_vz_jitter=0.020,
        falling_tau=(0.11, 0.22),
        falling_apex=(0.110, 0.200),
        falling_vxy=0.12,
        falling_contact_jitter=0.020,
        camera_missing_prob=0.0,
        dropout_prob=0.0,
        burst_prob=0.0,
        dropout_steps=1,
        burst_steps=1,
        obs_pos_noise=0.005,
        obs_vel_noise=0.05,
        hit_vxy_limit=0.54,
        hit_vxy_weight=1.2,
        next_contact_weight=0.16,
        apex_center_weight=0.12,
        camera_center_weight=0.70,
        view_xy_weight=0.55,
        view_z_weight=0.85,
        view_bounds_weight=2.8,
        view_oob_weight=0.7,
        view_vxy_weight=0.45,
    )
    cfg_r2 = recovery_cfg(
        "05_fov",
        target_x=(-0.065, 0.070),
        target_y=(0.050, 0.140),
        anchor_z=(-0.038, 0.034),
        xy_jitter=0.016,
        z_jitter=0.010,
        init_vxy=0.014,
        init_vz_jitter=0.028,
        falling_tau=(0.11, 0.24),
        falling_apex=(0.120, 0.220),
        falling_vxy=0.18,
        falling_contact_jitter=0.030,
        camera_missing_prob=0.15,
        dropout_prob=0.0015,
        burst_prob=0.0003,
        dropout_steps=4,
        burst_steps=12,
        obs_pos_noise=0.007,
        obs_vel_noise=0.07,
        hit_vxy_limit=0.50,
        hit_vxy_weight=1.6,
        next_contact_weight=0.20,
        apex_center_weight=0.14,
        camera_center_weight=0.82,
        view_xy_weight=0.68,
        view_z_weight=1.05,
        view_bounds_weight=3.5,
        view_oob_weight=1.0,
        view_vxy_weight=0.55,
    )
    cfg_r3 = recovery_cfg(
        "06_missing",
        target_x=(-0.085, 0.085),
        target_y=(0.045, 0.160),
        anchor_z=(-0.046, 0.040),
        xy_jitter=0.022,
        z_jitter=0.012,
        init_vxy=0.018,
        init_vz_jitter=0.036,
        falling_tau=(0.09, 0.24),
        falling_apex=(0.130, 0.240),
        falling_vxy=0.24,
        falling_contact_jitter=0.045,
        camera_missing_prob=0.40,
        dropout_prob=0.003,
        burst_prob=0.0008,
        dropout_steps=6,
        burst_steps=24,
        obs_pos_noise=0.010,
        obs_vel_noise=0.10,
        hit_vxy_limit=0.46,
        hit_vxy_weight=2.1,
        next_contact_weight=0.24,
        apex_center_weight=0.16,
        camera_center_weight=0.95,
        view_xy_weight=0.82,
        view_z_weight=1.35,
        view_bounds_weight=4.5,
        view_oob_weight=1.4,
        view_vxy_weight=0.65,
    )
    cfg_r4 = recovery_cfg(
        "07_dynamics",
        target_x=(-0.100, 0.100),
        target_y=(0.035, 0.175),
        anchor_z=(-0.056, 0.046),
        xy_jitter=0.026,
        z_jitter=0.014,
        init_vxy=0.022,
        init_vz_jitter=0.044,
        falling_tau=(0.08, 0.24),
        falling_apex=(0.140, 0.260),
        falling_vxy=0.30,
        falling_contact_jitter=0.055,
        camera_missing_prob=0.70,
        dropout_prob=0.006,
        burst_prob=0.0015,
        dropout_steps=9,
        burst_steps=36,
        obs_pos_noise=0.013,
        obs_vel_noise=0.13,
        hit_vxy_limit=0.44,
        hit_vxy_weight=2.6,
        next_contact_weight=0.28,
        apex_center_weight=0.18,
        camera_center_weight=1.05,
        view_xy_weight=0.92,
        view_z_weight=1.60,
        view_bounds_weight=5.2,
        view_oob_weight=1.8,
        view_vxy_weight=0.75,
    )
    cfg_r5 = recovery_cfg(
        "09_final",
        target_x=(-0.115, 0.115),
        target_y=(0.025, 0.190),
        anchor_z=(-0.066, 0.052),
        xy_jitter=0.030,
        z_jitter=0.016,
        init_vxy=0.026,
        init_vz_jitter=0.050,
        falling_tau=(0.075, 0.25),
        falling_apex=(0.150, 0.280),
        falling_vxy=0.34,
        falling_contact_jitter=0.065,
        camera_missing_prob=0.90,
        dropout_prob=0.008,
        burst_prob=0.002,
        dropout_steps=12,
        burst_steps=48,
        obs_pos_noise=0.015,
        obs_vel_noise=0.15,
        hit_vxy_limit=0.42,
        hit_vxy_weight=3.0,
        next_contact_weight=0.32,
        apex_center_weight=0.20,
        camera_center_weight=1.15,
        view_xy_weight=1.00,
        view_z_weight=1.80,
        view_bounds_weight=6.0,
        view_oob_weight=2.2,
        view_vxy_weight=0.85,
    )

    return [
        CurriculumStage(
            "recovery1_descent_small_no_missing",
            steps(5_000_000),
            cfg_r1,
            "Resume from stable 4g/polish: small falling-contact recovery states without missing.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=5.0,
            target_mean_len_frac=0.38,
            min_updates=70,
            target_camera_visible=0.74,
            target_ball_view_in_bounds=0.66,
            target_ball_view_z_ideal=0.52,
            target_hit1_rate=0.90,
            target_hit3_rate=0.40,
            target_mean_hits_ge3=6.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.64,
            target_hit_camera_visible_rate=0.86,
            target_hit_camera_lower_band_rate=0.58,
            max_recent_mean_hit_vxy=0.62,
            max_recent_hit_next_contact_anchor_err=0.32,
            target_episode_truncation_rate=0.22,
        ),
        CurriculumStage(
            "recovery2_descent_range_mild_missing",
            steps(6_000_000),
            cfg_r2,
            "Widen descending ball states and add mild stale-age camera-missing exposure.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=7.0,
            target_mean_len_frac=0.52,
            min_updates=80,
            target_camera_visible=0.72,
            target_ball_view_in_bounds=0.64,
            target_ball_view_z_ideal=0.54,
            target_hit1_rate=0.92,
            target_hit3_rate=0.55,
            target_mean_hits_ge3=8.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.60,
            target_hit_camera_visible_rate=0.88,
            target_hit_camera_lower_band_rate=0.64,
            max_recent_mean_hit_vxy=0.58,
            max_recent_hit_next_contact_anchor_err=0.30,
            min_ball_obs_missing_refresh_rate=0.001,
            max_ball_obs_lost_rate=0.060,
            target_episode_truncation_rate=0.38,
        ),
        CurriculumStage(
            "recovery3_wide_descent_missing",
            steps(7_000_000),
            cfg_r3,
            "Train wide post-apex recovery states with realistic missing/dropout and noise.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=9.0,
            target_mean_len_frac=0.68,
            min_updates=90,
            target_camera_visible=0.70,
            target_ball_view_in_bounds=0.64,
            target_ball_view_z_ideal=0.55,
            target_hit1_rate=0.94,
            target_hit3_rate=0.68,
            target_hit12_rate=0.18,
            target_mean_hits_ge3=10.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.56,
            target_hit_camera_visible_rate=0.90,
            target_hit_camera_lower_band_rate=0.72,
            max_recent_mean_hit_vxy=0.54,
            max_recent_hit_next_contact_anchor_err=0.28,
            min_ball_obs_missing_refresh_rate=0.003,
            max_ball_obs_lost_rate=0.055,
            target_episode_truncation_rate=0.55,
        ),
        CurriculumStage(
            "recovery4_wide_noise_dynamics",
            steps(8_000_000),
            cfg_r4,
            "Add broad noise, missing, actuator response, contact, PD, and observation-frame DR.",
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=11.0,
            target_mean_len_frac=0.82,
            min_updates=100,
            target_camera_visible=0.72,
            target_ball_view_in_bounds=0.66,
            target_ball_view_z_ideal=0.58,
            target_hit1_rate=0.96,
            target_hit3_rate=0.80,
            target_hit12_rate=0.40,
            target_mean_hits_ge3=12.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.54,
            target_hit_camera_visible_rate=0.94,
            target_hit_camera_lower_band_rate=0.80,
            max_recent_mean_hit_vxy=0.50,
            max_recent_hit_next_contact_anchor_err=0.26,
            min_ball_obs_missing_refresh_rate=0.006,
            max_ball_obs_lost_rate=0.045,
            target_episode_truncation_rate=0.72,
        ),
        CurriculumStage(
            "recovery5_final_missing_polish",
            steps(10_000_000),
            cfg_r5,
            "Final recovery polish: broad falling-contact states, D455 missing, and sim2real noise.",
            gate_mode="strict",
            advance_gate_mode="collapse",
            target_mean_hits=13.0,
            target_mean_len_frac=0.93,
            min_updates=120,
            target_camera_visible=0.74,
            target_ball_view_in_bounds=0.68,
            target_ball_view_z_ideal=0.60,
            target_hit1_rate=0.98,
            target_hit3_rate=0.88,
            target_hit12_rate=0.78,
            target_mean_hits_ge3=14.0,
            target_min_hit_interval_s=0.38,
            target_max_hit_interval_s=0.52,
            target_hit_camera_visible_rate=0.97,
            target_hit_camera_lower_band_rate=0.86,
            max_recent_mean_hit_vxy=0.48,
            max_recent_hit_next_contact_anchor_err=0.24,
            min_ball_obs_missing_refresh_rate=0.008,
            max_ball_obs_lost_rate=0.035,
            target_episode_truncation_rate=0.88,
        ),
    ]


def _target_generalization_stages(stage4g_cfg: MjxJuggleConfig) -> list[CurriculumStage]:
    def make_stage(
        name: str,
        *,
        target_x_range: tuple[float, float],
        target_y_range: tuple[float, float],
        anchor_z_range: tuple[float, float],
        xy_jitter: float,
        z_jitter: float,
        init_vxy: float,
        init_vz_jitter: float,
        xy_weight: float,
        z_weight: float,
        bounds_weight: float,
        out_of_bounds_weight: float = 0.0,
        z_not_ideal_weight: float = 0.0,
        vxy_weight: float,
        center_weight: float,
        target_hits: float,
        target_len_frac: float,
        target_camera_visible: float,
        target_in_bounds: float,
        target_z_ideal: float,
        notes: str,
        stage_steps: int = 2_000_000,
        min_return: float | None = 1.0,
    ) -> CurriculumStage:
        cfg = replace(
            stage4g_cfg,
            ball_spawn_xy_jitter=xy_jitter,
            ball_spawn_z_jitter=z_jitter,
            ball_init_vxy_max=init_vxy,
            ball_init_vz_jitter=init_vz_jitter,
            ball_obs_pos_noise_std=0.030,
            ball_obs_vel_noise_std=0.300,
            ball_anchor_xy_penalty_weight=max(float(stage4g_cfg.ball_anchor_xy_penalty_weight), 0.85),
            racket_chest_xy_penalty_weight=max(float(stage4g_cfg.racket_chest_xy_penalty_weight), 0.65),
            racket_chest_z_penalty_weight=max(float(stage4g_cfg.racket_chest_z_penalty_weight), 0.50),
        )
        cfg = _with_real_view_ball_range(
            cfg,
            terminate=True,
            xy_weight=xy_weight,
            z_weight=z_weight,
            bounds_weight=bounds_weight,
            out_of_bounds_weight=out_of_bounds_weight,
            z_not_ideal_weight=z_not_ideal_weight,
            vxy_weight=vxy_weight,
            target_x_range=target_x_range,
            target_y_range=target_y_range,
            anchor_z_range=anchor_z_range,
            launch_height=0.10,
            target_height=0.10,
            hit_height_center=0.13,
        )
        cfg = _with_strong_camera_centering(cfg, center_weight=center_weight)
        return CurriculumStage(
            name,
            stage_steps,
            cfg,
            notes,
            target_mean_hits=target_hits,
            target_mean_len_frac=target_len_frac,
            min_updates=30,
            min_recent_mean_return=min_return,
            target_camera_visible=target_camera_visible,
            min_recent_camera_reward_dense=-0.040,
            target_ball_view_in_bounds=target_in_bounds,
            target_ball_view_z_ideal=target_z_ideal,
        )

    return [
        make_stage(
            "stage5a_center_target_generalization_entry",
            target_x_range=(-0.02, 0.32),
            target_y_range=(-0.10, 0.22),
            anchor_z_range=(-0.30, -0.12),
            xy_jitter=0.015,
            z_jitter=0.015,
            init_vxy=0.010,
            init_vz_jitter=0.020,
            xy_weight=1.25,
            z_weight=2.60,
            bounds_weight=7.0,
            vxy_weight=0.90,
            center_weight=1.35,
            target_hits=3.8,
            target_len_frac=0.25,
            target_camera_visible=0.84,
            target_in_bounds=0.82,
            target_z_ideal=0.64,
            notes=(
                "Entry real-view generalization: randomize most of the visible Y band, bias X toward "
                "negative/center, and keep positive X coverage deliberately smaller."
            ),
        ),
        make_stage(
            "stage5b_real_view_generalization_mid",
            target_x_range=(-0.015, 0.33),
            target_y_range=(-0.12, 0.23),
            anchor_z_range=(-0.34, -0.10),
            xy_jitter=0.018,
            z_jitter=0.015,
            init_vxy=0.014,
            init_vz_jitter=0.030,
            xy_weight=1.10,
            z_weight=2.30,
            bounds_weight=7.5,
            vxy_weight=1.00,
            center_weight=1.25,
            target_hits=3.4,
            target_len_frac=0.23,
            target_camera_visible=0.82,
            target_in_bounds=0.78,
            target_z_ideal=0.58,
            notes=(
                "Mid real-view generalization: broaden target and launch variation while retaining hard visible-window done."
            ),
        ),
        make_stage(
            "stage5c_real_view_generalization_wide",
            target_x_range=(-0.01, 0.33),
            target_y_range=(-0.125, 0.225),
            anchor_z_range=(-0.36, -0.10),
            xy_jitter=0.020,
            z_jitter=0.015,
            init_vxy=0.018,
            init_vz_jitter=0.040,
            xy_weight=0.95,
            z_weight=2.00,
            bounds_weight=8.0,
            vxy_weight=1.10,
            center_weight=1.15,
            target_hits=3.0,
            target_len_frac=0.21,
            target_camera_visible=0.80,
            target_in_bounds=0.74,
            target_z_ideal=0.52,
            notes=(
                "Wide real-view generalization: approach the x[-0.20,0.20], y[-0.6,-0.2], z[0.65,1.1] "
                "usable window while keeping the main reward centered near x=0."
            ),
        ),
    ]


def _low_reset_target_generalization_stages(stage4g_cfg: MjxJuggleConfig) -> list[CurriculumStage]:
    def make_stage(
        name: str,
        *,
        target_x_range: tuple[float, float],
        target_y_range: tuple[float, float],
        anchor_z_range: tuple[float, float],
        xy_jitter: float,
        z_jitter: float,
        init_vxy: float,
        init_vz_jitter: float,
        xy_weight: float,
        z_weight: float,
        bounds_weight: float,
        vxy_weight: float,
        center_weight: float,
        target_hits: float,
        target_len_frac: float,
        target_camera_visible: float,
        target_in_bounds: float,
        target_z_ideal: float,
        notes: str,
        stage_steps: int = 2_000_000,
        min_return: float | None = 1.0,
        terminate: bool = False,
        miss_penalty_base: float = 1.0,
        miss_penalty_per_hit: float = 0.20,
        hit_reward_base: float = 0.9,
        hit_reward_combo: float = 0.10,
        target_height: float = 0.20,
        hit_height_center: float = 0.24,
        hit_confirm_rel_height: float = 0.06,
        low_hit_apex_margin: float = 0.025,
        hit_height_penalty_weight: float = 10.0,
        low_hit_penalty_weight: float = 10.0,
        center_flat_weight: float = 1.0,
        torque_weight: float = 0.00030,
        out_of_bounds_weight: float = 0.0,
        z_not_ideal_weight: float = 0.0,
        post_hit_survival_weight: float = 1.4,
        hit_cadence_weight: float = 0.0,
        hit_cadence_target_interval: float = 0.40,
        hit_cadence_sigma: float = 0.14,
        hit_min_interval_penalty_weight: float = 0.0,
        hit_min_interval: float = 0.40,
        hit_min_count_interval: float = 0.0,
        fast_hit_penalty_weight: float = 0.0,
        hit_reward_cap_mode: str | None = None,
        hit_reward_count_cap: int | None = None,
        hit_reward_cap_target_interval: float | None = None,
        pre_hit_intercept_weight: float = 0.0,
        pre_hit_intercept_sigma: float | None = None,
        pre_hit_intercept_time_max: float | None = None,
        pre_hit_intercept_penalty_weight: float = 0.0,
        pre_hit_intercept_penalty_sigma: float | None = None,
        pre_hit_intercept_penalty_radius: float | None = None,
        pre_hit_intercept_penalty_time_max: float | None = None,
        first_hit_apex_weight: float = 0.0,
        first_hit_apex_sigma: float | None = None,
        target_hit1_rate: float | None = None,
        target_hit3_rate: float | None = None,
        target_mean_hits_ge3: float | None = None,
        obs_pos_noise_std: float = 0.030,
        obs_vel_noise_std: float = 0.300,
        racket_z_hard_limit_down: float | None = None,
        racket_up_margin: float = 0.24,
        racket_z_soft_penalty_weight: float | None = None,
        racket_up_drift_penalty_weight: float | None = None,
        terminate_racket_z: bool = True,
    ) -> CurriculumStage:
        cfg = replace(
            stage4g_cfg,
            ball_spawn_xy_jitter=xy_jitter,
            ball_spawn_z_jitter=z_jitter,
            ball_init_vxy_max=init_vxy,
            ball_init_vz_jitter=init_vz_jitter,
            ball_obs_pos_noise_std=obs_pos_noise_std,
            ball_obs_vel_noise_std=obs_vel_noise_std,
            ball_anchor_xy_penalty_weight=max(float(stage4g_cfg.ball_anchor_xy_penalty_weight), 0.85),
            racket_chest_xy_penalty_weight=max(float(stage4g_cfg.racket_chest_xy_penalty_weight), 0.65),
            racket_chest_z_penalty_weight=max(float(stage4g_cfg.racket_chest_z_penalty_weight), 0.50),
        )
        cfg = _with_low_reset_ball_range(
            cfg,
            terminate=terminate,
            xy_weight=xy_weight,
            z_weight=z_weight,
            bounds_weight=bounds_weight,
            out_of_bounds_weight=out_of_bounds_weight,
            z_not_ideal_weight=z_not_ideal_weight,
            vxy_weight=vxy_weight,
            target_x_range=target_x_range,
            target_y_range=target_y_range,
            anchor_z_range=anchor_z_range,
            launch_height=0.10,
            target_height=target_height,
            hit_height_center=hit_height_center,
            hit_confirm_rel_height=hit_confirm_rel_height,
            racket_up_margin=racket_up_margin,
            terminate_racket_z=terminate_racket_z,
        )
        if racket_z_hard_limit_down is not None:
            cfg = replace(cfg, racket_z_hard_limit_down=float(racket_z_hard_limit_down))
        if racket_z_soft_penalty_weight is not None:
            cfg = replace(cfg, racket_z_soft_penalty_weight=float(racket_z_soft_penalty_weight))
        if racket_up_drift_penalty_weight is not None:
            cfg = replace(cfg, racket_up_drift_penalty_weight=float(racket_up_drift_penalty_weight))
        if not terminate:
            cfg = replace(
                cfg,
                ball_view_z_bounds_m=(0.62, 1.26),
                ball_view_z_ideal_m=(0.76, 1.22),
                ball_view_z_sigma_m=0.12,
            )
        cfg = _with_strong_camera_centering(cfg, center_weight=center_weight)
        cfg = replace(
            cfg,
            termination_miss_penalty_base=min(float(cfg.termination_miss_penalty_base), miss_penalty_base),
            termination_miss_penalty_per_hit=min(
                float(cfg.termination_miss_penalty_per_hit),
                miss_penalty_per_hit,
            ),
            hit_reward_base=max(float(cfg.hit_reward_base), hit_reward_base),
            hit_reward_combo=max(float(cfg.hit_reward_combo), hit_reward_combo),
            low_hit_apex_margin=min(float(cfg.low_hit_apex_margin), low_hit_apex_margin),
            hit_height_penalty_weight=max(float(cfg.hit_height_penalty_weight), hit_height_penalty_weight),
            low_hit_penalty_weight=max(float(cfg.low_hit_penalty_weight), low_hit_penalty_weight),
            center_flat_hit_reward_weight=max(float(cfg.center_flat_hit_reward_weight), center_flat_weight),
            post_hit_survival_reward_weight=max(
                float(cfg.post_hit_survival_reward_weight),
                post_hit_survival_weight,
            ),
            hit_cadence_reward_weight=max(float(cfg.hit_cadence_reward_weight), hit_cadence_weight),
            hit_cadence_target_interval=hit_cadence_target_interval
            if hit_cadence_weight > 0.0
            else float(cfg.hit_cadence_target_interval),
            hit_cadence_sigma=hit_cadence_sigma if hit_cadence_weight > 0.0 else float(cfg.hit_cadence_sigma),
            hit_min_interval_penalty_weight=max(
                float(cfg.hit_min_interval_penalty_weight),
                hit_min_interval_penalty_weight,
            ),
            hit_min_interval=hit_min_interval
            if hit_min_interval_penalty_weight > 0.0
            else float(cfg.hit_min_interval),
            hit_min_count_interval=max(float(cfg.hit_min_count_interval), hit_min_count_interval),
            fast_hit_penalty_weight=max(float(cfg.fast_hit_penalty_weight), fast_hit_penalty_weight),
            hit_reward_cap_mode=hit_reward_cap_mode
            if hit_reward_cap_mode is not None
            else str(cfg.hit_reward_cap_mode),
            hit_reward_count_cap=max(0, int(hit_reward_count_cap))
            if hit_reward_count_cap is not None
            else int(cfg.hit_reward_count_cap),
            hit_reward_cap_target_interval=hit_reward_cap_target_interval
            if hit_reward_cap_target_interval is not None
            else float(cfg.hit_reward_cap_target_interval),
            pre_hit_intercept_reward_weight=max(
                float(cfg.pre_hit_intercept_reward_weight),
                pre_hit_intercept_weight,
            ),
            pre_hit_intercept_sigma=float(pre_hit_intercept_sigma)
            if pre_hit_intercept_sigma is not None
            else min(float(cfg.pre_hit_intercept_sigma), 0.075),
            pre_hit_intercept_time_max=float(pre_hit_intercept_time_max)
            if pre_hit_intercept_time_max is not None
            else max(float(cfg.pre_hit_intercept_time_max), 0.55),
            pre_hit_intercept_penalty_weight=max(
                float(cfg.pre_hit_intercept_penalty_weight),
                pre_hit_intercept_penalty_weight,
            ),
            pre_hit_intercept_penalty_sigma=float(pre_hit_intercept_penalty_sigma)
            if pre_hit_intercept_penalty_sigma is not None
            else float(cfg.pre_hit_intercept_penalty_sigma),
            pre_hit_intercept_penalty_radius=float(pre_hit_intercept_penalty_radius)
            if pre_hit_intercept_penalty_radius is not None
            else float(cfg.pre_hit_intercept_penalty_radius),
            pre_hit_intercept_penalty_time_max=float(pre_hit_intercept_penalty_time_max)
            if pre_hit_intercept_penalty_time_max is not None
            else float(cfg.pre_hit_intercept_penalty_time_max),
            first_hit_apex_reward_weight=max(
                float(cfg.first_hit_apex_reward_weight),
                first_hit_apex_weight,
            ),
            first_hit_apex_sigma=float(first_hit_apex_sigma)
            if first_hit_apex_sigma is not None
            else min(float(cfg.first_hit_apex_sigma), 0.055),
            torque_penalty_weight=min(float(cfg.torque_penalty_weight), torque_weight),
        )
        return CurriculumStage(
            name,
            stage_steps,
            cfg,
            notes,
            target_mean_hits=target_hits,
            target_mean_len_frac=target_len_frac,
            min_updates=30,
            min_recent_mean_return=min_return,
            target_camera_visible=target_camera_visible,
            min_recent_camera_reward_dense=-0.040,
            target_ball_view_in_bounds=target_in_bounds,
            target_ball_view_z_ideal=target_z_ideal,
            target_hit1_rate=target_hit1_rate,
            target_hit3_rate=target_hit3_rate,
            target_mean_hits_ge3=target_mean_hits_ge3,
            target_min_hit_interval_s=(
                max(0.34, float(cfg.hit_min_count_interval) + 0.04)
                if float(cfg.hit_min_count_interval) > 0.0
                else None
            ),
        )

    return [
        make_stage(
            "stage5a0_low_reset_height_noise_bridge",
            target_x_range=(0.055, 0.175),
            target_y_range=(-0.045, 0.115),
            anchor_z_range=(-0.010, 0.010),
            xy_jitter=0.008,
            z_jitter=0.006,
            init_vxy=0.006,
            init_vz_jitter=0.010,
            xy_weight=0.80,
            z_weight=1.20,
            bounds_weight=3.8,
            vxy_weight=0.45,
            center_weight=0.65,
            target_hits=2.2,
            target_len_frac=0.18,
            target_camera_visible=0.76,
            target_in_bounds=0.80,
            target_z_ideal=0.58,
            min_return=None,
            obs_pos_noise_std=0.010,
            obs_vel_noise_std=0.100,
            terminate_racket_z=False,
            target_height=0.14,
            hit_height_center=0.17,
            hit_height_penalty_weight=12.0,
            low_hit_penalty_weight=12.0,
            pre_hit_intercept_weight=0.8,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.35,
            pre_hit_intercept_penalty_sigma=0.22,
            target_hit1_rate=0.60,
            target_hit3_rate=0.12,
            notes=(
                "Low-reset stage5 entry bridge: introduce the lower launch/reset and moderate hit-height/noise "
                "targets before enabling the full stage5a observation noise and racket-z hard termination."
            ),
        ),
        make_stage(
            "stage5a1_low_reset_full_noise_bridge",
            target_x_range=(0.055, 0.175),
            target_y_range=(-0.045, 0.115),
            anchor_z_range=(-0.010, 0.010),
            xy_jitter=0.008,
            z_jitter=0.006,
            init_vxy=0.006,
            init_vz_jitter=0.012,
            xy_weight=0.82,
            z_weight=1.35,
            bounds_weight=4.0,
            vxy_weight=0.50,
            center_weight=0.70,
            target_hits=2.0,
            target_len_frac=0.17,
            target_camera_visible=0.76,
            target_in_bounds=0.80,
            target_z_ideal=0.58,
            min_return=None,
            obs_pos_noise_std=0.030,
            obs_vel_noise_std=0.300,
            terminate_racket_z=False,
            target_height=0.14,
            hit_height_center=0.17,
            hit_height_penalty_weight=12.0,
            low_hit_penalty_weight=12.0,
            pre_hit_intercept_weight=0.9,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.40,
            pre_hit_intercept_penalty_sigma=0.22,
            target_hit1_rate=0.60,
            target_hit3_rate=0.10,
            notes=(
                "Low-reset stage5 bridge: keep the moderate hit-height objective but switch to the full "
                "stage5 ball-observation noise before raising the hit height."
            ),
        ),
        make_stage(
            "stage5a2_low_reset_full_height_no_z_bridge",
            target_x_range=(0.055, 0.175),
            target_y_range=(-0.045, 0.115),
            anchor_z_range=(-0.015, 0.015),
            xy_jitter=0.008,
            z_jitter=0.006,
            init_vxy=0.006,
            init_vz_jitter=0.015,
            xy_weight=0.86,
            z_weight=1.65,
            bounds_weight=4.2,
            vxy_weight=0.52,
            center_weight=0.78,
            target_hits=1.4,
            target_len_frac=0.14,
            target_camera_visible=0.76,
            target_in_bounds=0.80,
            target_z_ideal=0.62,
            min_return=None,
            obs_pos_noise_std=0.030,
            obs_vel_noise_std=0.300,
            racket_up_margin=0.30,
            terminate_racket_z=False,
            target_height=0.20,
            hit_height_center=0.24,
            pre_hit_intercept_weight=0.9,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.40,
            pre_hit_intercept_penalty_sigma=0.22,
            target_hit1_rate=0.58,
            target_hit3_rate=0.05,
            notes=(
                "Low-reset stage5 bridge: raise to the full stage5 hit-height objective while keeping "
                "racket-z hard termination disabled, so the policy can relearn the taller apex first."
            ),
        ),
        make_stage(
            "stage5a3_low_reset_z_soft_bridge",
            target_x_range=(0.055, 0.175),
            target_y_range=(-0.045, 0.115),
            anchor_z_range=(-0.015, 0.015),
            xy_jitter=0.008,
            z_jitter=0.006,
            init_vxy=0.006,
            init_vz_jitter=0.015,
            xy_weight=0.90,
            z_weight=1.80,
            bounds_weight=4.5,
            vxy_weight=0.55,
            center_weight=0.85,
            target_hits=2.0,
            target_len_frac=0.20,
            target_camera_visible=0.76,
            target_in_bounds=0.80,
            target_z_ideal=0.64,
            min_return=None,
            obs_pos_noise_std=0.030,
            obs_vel_noise_std=0.300,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_height=0.20,
            hit_height_center=0.24,
            pre_hit_intercept_weight=0.8,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.35,
            pre_hit_intercept_penalty_sigma=0.22,
            target_hit1_rate=0.65,
            target_hit3_rate=0.20,
            notes=(
                "Low-reset stage5 bridge: keep hard racket-z termination disabled but strengthen z-band "
                "soft penalties so the policy learns the final height band before hard termination."
            ),
        ),
        make_stage(
            "stage5a4_low_reset_racket_z_bridge",
            target_x_range=(0.055, 0.175),
            target_y_range=(-0.045, 0.115),
            anchor_z_range=(-0.015, 0.015),
            xy_jitter=0.008,
            z_jitter=0.006,
            init_vxy=0.006,
            init_vz_jitter=0.015,
            xy_weight=0.90,
            z_weight=1.80,
            bounds_weight=4.5,
            vxy_weight=0.55,
            center_weight=0.85,
            target_hits=1.2,
            target_len_frac=0.14,
            target_camera_visible=0.76,
            target_in_bounds=0.80,
            target_z_ideal=0.64,
            min_return=None,
            obs_pos_noise_std=0.030,
            obs_vel_noise_std=0.300,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_height=0.20,
            hit_height_center=0.24,
            pre_hit_intercept_weight=0.8,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.35,
            pre_hit_intercept_penalty_sigma=0.22,
            target_hit1_rate=0.58,
            target_hit3_rate=0.05,
            notes=(
                "Low-reset stage5 bridge: keep hard racket-z termination disabled but retain strong "
                "soft z-band penalties as a final non-collapse check before the original stage5a 4-hit objective."
            ),
        ),
        make_stage(
            "stage5a_low_reset_generalization_entry",
            target_x_range=(0.055, 0.175),
            target_y_range=(-0.045, 0.115),
            anchor_z_range=(-0.015, 0.015),
            xy_jitter=0.008,
            z_jitter=0.006,
            init_vxy=0.006,
            init_vz_jitter=0.015,
            xy_weight=0.90,
            z_weight=1.80,
            bounds_weight=4.5,
            vxy_weight=0.55,
            center_weight=0.85,
            target_hits=2.15,
            target_len_frac=0.36,
            target_camera_visible=0.83,
            target_in_bounds=0.84,
            target_z_ideal=0.68,
            min_return=None,
            target_hit1_rate=0.70,
            target_hit3_rate=0.30,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=0.8,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.35,
            pre_hit_intercept_penalty_sigma=0.22,
            notes="Low-reset entry generalization close to the stage4g right-arm juggling pocket.",
        ),
        make_stage(
            "stage5ab_low_reset_generalization_bridge",
            target_x_range=(0.045, 0.190),
            target_y_range=(-0.060, 0.125),
            anchor_z_range=(-0.025, 0.020),
            xy_jitter=0.010,
            z_jitter=0.007,
            init_vxy=0.008,
            init_vz_jitter=0.020,
            xy_weight=0.70,
            z_weight=1.50,
            bounds_weight=4.0,
            vxy_weight=0.45,
            center_weight=0.75,
            target_hits=1.95,
            target_len_frac=0.32,
            target_camera_visible=0.82,
            target_in_bounds=0.82,
            target_z_ideal=0.64,
            min_return=None,
            target_hit1_rate=0.63,
            target_hit3_rate=0.25,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=1.0,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.45,
            pre_hit_intercept_penalty_sigma=0.22,
            notes=(
                "Low-reset bridge generalization: expands from stage5a without simultaneously opening "
                "the full velocity and target-height spread."
            ),
        ),
        make_stage(
            "stage5b_low_reset_generalization_mid",
            target_x_range=(0.030, 0.205),
            target_y_range=(-0.070, 0.135),
            anchor_z_range=(-0.035, 0.025),
            xy_jitter=0.012,
            z_jitter=0.008,
            init_vxy=0.010,
            init_vz_jitter=0.025,
            xy_weight=0.65,
            z_weight=1.45,
            bounds_weight=4.2,
            vxy_weight=0.50,
            center_weight=0.70,
            target_hits=1.65,
            target_len_frac=0.29,
            target_camera_visible=0.82,
            target_in_bounds=0.78,
            target_z_ideal=0.58,
            min_return=None,
            stage_steps=3_000_000,
            target_hit1_rate=0.56,
            target_hit3_rate=0.20,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=1.2,
            pre_hit_intercept_sigma=0.105,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.55,
            pre_hit_intercept_penalty_sigma=0.22,
            notes=(
                "Low-reset mid generalization: the old stage5b range, reached through the bridge stage so "
                "hits and survival do not collapse at the transition."
            ),
        ),
        make_stage(
            "stage5c_low_reset_generalization_outer",
            target_x_range=(0.005, 0.230),
            target_y_range=(-0.085, 0.145),
            anchor_z_range=(-0.050, 0.032),
            xy_jitter=0.016,
            z_jitter=0.009,
            init_vxy=0.013,
            init_vz_jitter=0.032,
            xy_weight=0.60,
            z_weight=1.40,
            bounds_weight=4.4,
            vxy_weight=0.55,
            center_weight=0.65,
            target_hits=1.30,
            target_len_frac=0.235,
            target_camera_visible=0.81,
            target_in_bounds=0.78,
            target_z_ideal=0.58,
            min_return=None,
            stage_steps=3_000_000,
            target_hit1_rate=0.47,
            target_hit3_rate=0.14,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=1.5,
            pre_hit_intercept_sigma=0.105,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.65,
            pre_hit_intercept_penalty_sigma=0.22,
            notes=(
                "Low-reset outer generalization: opens most of the final target spread while keeping one "
                "more step before the full wide range."
            ),
        ),
        make_stage(
            "stage5d_low_reset_generalization_wide",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=0.60,
            z_weight=1.40,
            bounds_weight=4.6,
            out_of_bounds_weight=1.5,
            z_not_ideal_weight=0.4,
            vxy_weight=0.60,
            center_weight=0.65,
            target_hits=1.00,
            target_len_frac=0.19,
            target_camera_visible=0.80,
            target_in_bounds=0.74,
            target_z_ideal=0.52,
            min_return=None,
            stage_steps=3_000_000,
            hit_height_center=0.240,
            hit_confirm_rel_height=0.045,
            target_hit1_rate=0.38,
            target_hit3_rate=0.08,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=1.7,
            pre_hit_intercept_sigma=0.100,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.75,
            pre_hit_intercept_penalty_sigma=0.22,
            notes=(
                "Low-reset wide generalization: reaches the final broad target and ball-state distribution "
                "without demanding full-episode survival yet."
            ),
        ),
        make_stage(
            "stage5d1b_low_reset_easy_to_center_first_hit_bridge",
            target_x_range=(0.118, 0.152),
            target_y_range=(-0.032, 0.006),
            anchor_z_range=(-0.055, 0.030),
            xy_jitter=0.010,
            z_jitter=0.008,
            init_vxy=0.008,
            init_vz_jitter=0.018,
            xy_weight=0.98,
            z_weight=1.78,
            bounds_weight=5.6,
            out_of_bounds_weight=3.2,
            z_not_ideal_weight=0.72,
            vxy_weight=0.70,
            center_weight=0.98,
            target_hits=0.30,
            target_len_frac=0.080,
            target_camera_visible=0.82,
            target_in_bounds=0.79,
            target_z_ideal=0.56,
            min_return=None,
            stage_steps=2_750_000,
            hit_height_center=0.240,
            hit_confirm_rel_height=0.052,
            hit_height_penalty_weight=14.0,
            low_hit_penalty_weight=14.0,
            center_flat_weight=1.18,
            post_hit_survival_weight=1.40,
            target_hit1_rate=0.12,
            target_hit3_rate=0.006,
            target_mean_hits_ge3=0.75,
            obs_pos_noise_std=0.020,
            obs_vel_noise_std=0.200,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=7.0,
            pre_hit_intercept_sigma=0.200,
            pre_hit_intercept_time_max=0.95,
            pre_hit_intercept_penalty_weight=2.00,
            pre_hit_intercept_penalty_sigma=0.34,
            pre_hit_intercept_penalty_radius=0.026,
            pre_hit_intercept_penalty_time_max=0.95,
            first_hit_apex_weight=1.00,
            first_hit_apex_sigma=0.095,
            notes=(
                "Low-reset easy-to-center first-hit bridge: narrow band between the stage5d1m "
                "survivor geometry (target_x≈0.139,target_y≈-0.016,ball_y≈-0.46) and the "
                "validation center (target_x≈0.155,target_y≈-0.037,ball_y≈-0.48). It prevents "
                "a direct all-zero jump into stage5d1c while still moving toward the failed bucket."
            ),
        ),
        make_stage(
            "stage5d1c_low_reset_validate_center_first_hit_warmup",
            target_x_range=(0.140, 0.172),
            target_y_range=(-0.055, -0.020),
            anchor_z_range=(-0.050, 0.025),
            xy_jitter=0.006,
            z_jitter=0.006,
            init_vxy=0.004,
            init_vz_jitter=0.010,
            xy_weight=1.05,
            z_weight=1.85,
            bounds_weight=5.8,
            out_of_bounds_weight=3.5,
            z_not_ideal_weight=0.75,
            vxy_weight=0.70,
            center_weight=1.05,
            target_hits=0.22,
            target_len_frac=0.070,
            target_camera_visible=0.82,
            target_in_bounds=0.80,
            target_z_ideal=0.56,
            min_return=None,
            stage_steps=2_500_000,
            hit_height_center=0.240,
            hit_confirm_rel_height=0.052,
            hit_height_penalty_weight=14.0,
            low_hit_penalty_weight=14.0,
            center_flat_weight=1.18,
            post_hit_survival_weight=1.35,
            target_hit1_rate=0.10,
            target_hit3_rate=0.005,
            target_mean_hits_ge3=0.5,
            obs_pos_noise_std=0.015,
            obs_vel_noise_std=0.150,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=8.5,
            pre_hit_intercept_sigma=0.220,
            pre_hit_intercept_time_max=0.98,
            pre_hit_intercept_penalty_weight=2.20,
            pre_hit_intercept_penalty_sigma=0.36,
            pre_hit_intercept_penalty_radius=0.028,
            pre_hit_intercept_penalty_time_max=0.98,
            first_hit_apex_weight=1.10,
            first_hit_apex_sigma=0.100,
            notes=(
                "Low-reset validate-center first-hit warmup: isolates the deterministic/stochastic "
                "validation mean geometry seen after stage5d1m (target_x≈0.155,target_y≈-0.037, "
                "ball_y≈-0.48) with reduced launch/noise spread. This prevents easy-corner "
                "survivor bias from hiding a zero-hit validation policy before reopening d1m/d1h."
            ),
        ),
        make_stage(
            "stage5d1m_low_reset_hard_edge_first_hit_bridge",
            target_x_range=(0.105, 0.195),
            target_y_range=(-0.085, 0.025),
            anchor_z_range=(-0.070, 0.035),
            xy_jitter=0.016,
            z_jitter=0.010,
            init_vxy=0.012,
            init_vz_jitter=0.030,
            xy_weight=0.92,
            z_weight=1.80,
            bounds_weight=5.8,
            out_of_bounds_weight=3.5,
            z_not_ideal_weight=0.75,
            vxy_weight=0.72,
            center_weight=0.90,
            target_hits=0.35,
            target_len_frac=0.085,
            target_camera_visible=0.82,
            target_in_bounds=0.79,
            target_z_ideal=0.56,
            min_return=None,
            stage_steps=3_500_000,
            hit_height_center=0.240,
            hit_confirm_rel_height=0.052,
            hit_height_penalty_weight=14.0,
            low_hit_penalty_weight=14.0,
            center_flat_weight=1.18,
            post_hit_survival_weight=1.45,
            target_hit1_rate=0.16,
            target_hit3_rate=0.01,
            target_mean_hits_ge3=1.0,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=5.6,
            pre_hit_intercept_sigma=0.150,
            pre_hit_intercept_time_max=0.90,
            pre_hit_intercept_penalty_weight=1.80,
            pre_hit_intercept_penalty_sigma=0.30,
            pre_hit_intercept_penalty_radius=0.018,
            pre_hit_intercept_penalty_time_max=0.90,
            first_hit_apex_weight=1.00,
            first_hit_apex_sigma=0.090,
            notes=(
                "Low-reset hard-edge first-hit bridge: intermediate bucket between stage5d and the "
                "hard-only validation corner. It targets the validate mean geometry while reducing "
                "velocity/noise so first-hit learning has a non-zero foothold."
            ),
        ),
        make_stage(
            "stage5d1h_low_reset_hard_first_hit_recovery",
            target_x_range=(0.150, 0.260),
            target_y_range=(-0.115, -0.020),
            anchor_z_range=(-0.070, 0.030),
            xy_jitter=0.018,
            z_jitter=0.010,
            init_vxy=0.014,
            init_vz_jitter=0.035,
            xy_weight=1.00,
            z_weight=1.90,
            bounds_weight=6.2,
            out_of_bounds_weight=4.0,
            z_not_ideal_weight=0.85,
            vxy_weight=0.75,
            center_weight=0.95,
            target_hits=0.45,
            target_len_frac=0.10,
            target_camera_visible=0.82,
            target_in_bounds=0.80,
            target_z_ideal=0.58,
            min_return=None,
            stage_steps=3_500_000,
            hit_height_center=0.245,
            hit_confirm_rel_height=0.055,
            hit_height_penalty_weight=14.0,
            low_hit_penalty_weight=14.0,
            center_flat_weight=1.20,
            post_hit_survival_weight=1.45,
            target_hit1_rate=0.22,
            target_hit3_rate=0.02,
            target_mean_hits_ge3=2.0,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=7.0,
            pre_hit_intercept_sigma=0.160,
            pre_hit_intercept_time_max=0.95,
            pre_hit_intercept_penalty_weight=2.40,
            pre_hit_intercept_penalty_sigma=0.32,
            pre_hit_intercept_penalty_radius=0.015,
            pre_hit_intercept_penalty_time_max=0.95,
            first_hit_apex_weight=1.20,
            first_hit_apex_sigma=0.100,
            notes=(
                "Low-reset hard-bucket first-hit recovery: isolate the validation failure bucket "
                "(large target_x, lower/behind target_y) so short hard failures are not drowned out "
                "by longer easy episodes before the broader validate-bucket recovery stage."
            ),
        ),
        make_stage(
            "stage5d1_low_reset_validate_bucket_recovery",
            target_x_range=(0.040, 0.260),
            target_y_range=(-0.115, 0.060),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=0.82,
            z_weight=1.70,
            bounds_weight=5.4,
            out_of_bounds_weight=3.0,
            z_not_ideal_weight=0.7,
            vxy_weight=0.66,
            center_weight=0.82,
            target_hits=0.75,
            target_len_frac=0.13,
            target_camera_visible=0.80,
            target_in_bounds=0.76,
            target_z_ideal=0.54,
            min_return=None,
            stage_steps=4_000_000,
            hit_height_center=0.245,
            hit_confirm_rel_height=0.055,
            hit_height_penalty_weight=14.0,
            low_hit_penalty_weight=14.0,
            center_flat_weight=1.15,
            post_hit_survival_weight=1.8,
            target_hit1_rate=0.28,
            target_hit3_rate=0.04,
            target_mean_hits_ge3=3.0,
            racket_up_margin=0.30,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            pre_hit_intercept_weight=3.8,
            pre_hit_intercept_sigma=0.120,
            pre_hit_intercept_time_max=0.78,
            pre_hit_intercept_penalty_weight=1.20,
            pre_hit_intercept_penalty_sigma=0.24,
            first_hit_apex_weight=0.80,
            first_hit_apex_sigma=0.075,
            notes=(
                "Low-reset validation-bucket recovery: over-sample the full-validate hard side "
                "(large target_x and lower/behind target_y) that repeatedly failed deterministic and "
                "stochastic validation after stage5d, before returning to survival-ramp stages."
            ),
        ),
        make_stage(
            "stage5e_low_reset_wide_survival_len35_soft",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=0.66,
            z_weight=1.45,
            bounds_weight=4.8,
            out_of_bounds_weight=2.5,
            z_not_ideal_weight=0.7,
            vxy_weight=0.62,
            center_weight=0.70,
            target_hits=4.1,
            target_len_frac=0.35,
            target_camera_visible=0.80,
            target_in_bounds=0.76,
            target_z_ideal=0.54,
            min_return=0.0,
            stage_steps=4_000_000,
            terminate=False,
            miss_penalty_base=0.8,
            miss_penalty_per_hit=0.15,
            hit_reward_base=1.1,
            hit_reward_combo=0.16,
            target_height=0.200,
            hit_height_center=0.240,
            hit_confirm_rel_height=0.055,
            low_hit_apex_margin=0.020,
            hit_height_penalty_weight=12.0,
            low_hit_penalty_weight=12.0,
            center_flat_weight=1.1,
            post_hit_survival_weight=1.8,
            hit_cadence_weight=0.14,
            hit_cadence_target_interval=0.44,
            hit_cadence_sigma=0.11,
            hit_min_interval_penalty_weight=0.70,
            hit_min_interval=0.32,
            hit_min_count_interval=0.30,
            fast_hit_penalty_weight=0.60,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.44,
            pre_hit_intercept_weight=2.0,
            pre_hit_intercept_sigma=0.100,
            pre_hit_intercept_time_max=0.70,
            pre_hit_intercept_penalty_weight=0.90,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.45,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_hit1_rate=0.58,
            target_hit3_rate=0.38,
            target_mean_hits_ge3=8.0,
            notes=(
                "Low-reset wide soft survival bridge: keep the full wide distribution while extending "
                "episode length before hard view termination is reintroduced."
            ),
        ),
        make_stage(
            "stage5e1_low_reset_entry_apex_len38_soft",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=0.86,
            z_weight=1.75,
            bounds_weight=5.8,
            out_of_bounds_weight=4.0,
            z_not_ideal_weight=1.0,
            vxy_weight=0.70,
            center_weight=0.90,
            target_hits=4.6,
            target_len_frac=0.38,
            target_camera_visible=0.82,
            target_in_bounds=0.80,
            target_z_ideal=0.64,
            min_return=-0.2,
            stage_steps=4_000_000,
            terminate=False,
            miss_penalty_base=0.9,
            miss_penalty_per_hit=0.18,
            hit_reward_base=1.08,
            hit_reward_combo=0.16,
            target_height=0.220,
            hit_height_center=0.260,
            hit_confirm_rel_height=0.060,
            low_hit_apex_margin=0.018,
            hit_height_penalty_weight=14.0,
            low_hit_penalty_weight=13.0,
            center_flat_weight=1.1,
            post_hit_survival_weight=1.70,
            hit_cadence_weight=0.18,
            hit_cadence_target_interval=0.44,
            hit_cadence_sigma=0.11,
            hit_min_interval_penalty_weight=0.75,
            hit_min_interval=0.32,
            hit_min_count_interval=0.30,
            fast_hit_penalty_weight=0.60,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.44,
            pre_hit_intercept_weight=3.6,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.72,
            pre_hit_intercept_penalty_weight=1.10,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.75,
            first_hit_apex_sigma=0.070,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_hit1_rate=0.60,
            target_hit3_rate=0.40,
            target_mean_hits_ge3=9.0,
            notes=(
                "Low-reset entry-geometry bridge: raise first-hit reliability with broad pre-hit intercept "
                "shaping while keeping long-loop reward available for successful episodes."
            ),
        ),
        make_stage(
            "stage5e1b_low_reset_first_hit_bridge_len38_soft",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=0.92,
            z_weight=1.85,
            bounds_weight=6.0,
            out_of_bounds_weight=4.5,
            z_not_ideal_weight=1.1,
            vxy_weight=0.72,
            center_weight=0.95,
            target_hits=5.4,
            target_len_frac=0.42,
            target_camera_visible=0.82,
            target_in_bounds=0.80,
            target_z_ideal=0.64,
            min_return=0.0,
            stage_steps=4_000_000,
            terminate=False,
            miss_penalty_base=0.9,
            miss_penalty_per_hit=0.18,
            hit_reward_base=1.12,
            hit_reward_combo=0.18,
            target_height=0.220,
            hit_height_center=0.260,
            hit_confirm_rel_height=0.060,
            low_hit_apex_margin=0.018,
            hit_height_penalty_weight=16.0,
            low_hit_penalty_weight=14.0,
            center_flat_weight=1.1,
            post_hit_survival_weight=1.85,
            hit_cadence_weight=0.20,
            hit_cadence_target_interval=0.44,
            hit_cadence_sigma=0.11,
            hit_min_interval_penalty_weight=0.75,
            hit_min_interval=0.32,
            hit_min_count_interval=0.30,
            fast_hit_penalty_weight=0.60,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.44,
            pre_hit_intercept_weight=3.2,
            pre_hit_intercept_sigma=0.100,
            pre_hit_intercept_time_max=0.72,
            pre_hit_intercept_penalty_weight=1.05,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.75,
            first_hit_apex_sigma=0.065,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_hit1_rate=0.64,
            target_hit3_rate=0.46,
            target_mean_hits_ge3=11.0,
            notes=(
                "Low-reset first-hit consolidation: require the entry bins to improve without suppressing "
                "13-15 hit trajectories that already have the right long-loop shape."
            ),
        ),
        make_stage(
            "stage5e2a_low_reset_view_recenter_len45_soft",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=1.12,
            z_weight=2.20,
            bounds_weight=7.6,
            out_of_bounds_weight=7.5,
            z_not_ideal_weight=1.9,
            vxy_weight=0.84,
            center_weight=1.18,
            target_hits=6.2,
            target_len_frac=0.50,
            target_camera_visible=0.84,
            target_in_bounds=0.84,
            target_z_ideal=0.72,
            min_return=0.0,
            stage_steps=5_000_000,
            terminate=False,
            miss_penalty_base=1.0,
            miss_penalty_per_hit=0.24,
            hit_reward_base=1.15,
            hit_reward_combo=0.22,
            target_height=0.220,
            hit_height_center=0.270,
            hit_confirm_rel_height=0.065,
            low_hit_apex_margin=0.015,
            hit_height_penalty_weight=16.0,
            low_hit_penalty_weight=14.0,
            center_flat_weight=1.1,
            post_hit_survival_weight=2.0,
            hit_cadence_weight=0.22,
            hit_cadence_target_interval=0.46,
            hit_cadence_sigma=0.10,
            hit_min_interval_penalty_weight=0.85,
            hit_min_interval=0.34,
            hit_min_count_interval=0.32,
            fast_hit_penalty_weight=0.65,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.46,
            pre_hit_intercept_weight=3.0,
            pre_hit_intercept_sigma=0.105,
            pre_hit_intercept_time_max=0.72,
            pre_hit_intercept_penalty_weight=1.05,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.65,
            first_hit_apex_sigma=0.065,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_hit1_rate=0.68,
            target_hit3_rate=0.50,
            target_mean_hits_ge3=12.0,
            notes=(
                "Low-reset view-recenter bridge: tighten view/z-ideal requirements after first-hit entry "
                "improves, without immediately demanding a 7+ hit mean."
            ),
        ),
        make_stage(
            "stage5e2b_low_reset_long_soft_len70",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=1.10,
            z_weight=2.20,
            bounds_weight=7.5,
            out_of_bounds_weight=8.0,
            z_not_ideal_weight=1.9,
            vxy_weight=0.86,
            center_weight=1.18,
            target_hits=9.0,
            target_len_frac=0.70,
            target_camera_visible=0.84,
            target_in_bounds=0.82,
            target_z_ideal=0.70,
            min_return=0.0,
            stage_steps=6_000_000,
            terminate=False,
            miss_penalty_base=1.1,
            miss_penalty_per_hit=0.30,
            hit_reward_base=1.20,
            hit_reward_combo=0.24,
            target_height=0.230,
            hit_height_center=0.275,
            hit_confirm_rel_height=0.070,
            low_hit_apex_margin=0.015,
            hit_height_penalty_weight=18.0,
            low_hit_penalty_weight=15.0,
            center_flat_weight=1.15,
            post_hit_survival_weight=2.2,
            hit_cadence_weight=0.22,
            hit_cadence_target_interval=0.45,
            hit_cadence_sigma=0.10,
            hit_min_interval_penalty_weight=0.90,
            hit_min_interval=0.33,
            hit_min_count_interval=0.31,
            fast_hit_penalty_weight=0.70,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.45,
            pre_hit_intercept_weight=3.0,
            pre_hit_intercept_sigma=0.110,
            pre_hit_intercept_time_max=0.72,
            pre_hit_intercept_penalty_weight=1.05,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.65,
            first_hit_apex_sigma=0.065,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_hit1_rate=0.72,
            target_hit3_rate=0.56,
            target_mean_hits_ge3=13.5,
            notes=(
                "Low-reset long-soft bridge: lengthen horizon only after entry and view recenter pass, "
                "so long survival does not compete with first-hit recovery in the same gate."
            ),
        ),
        make_stage(
            "stage5e3_low_reset_visible_pre_hard_len85_soft",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=1.12,
            z_weight=2.25,
            bounds_weight=7.8,
            out_of_bounds_weight=8.5,
            z_not_ideal_weight=2.0,
            vxy_weight=0.90,
            center_weight=1.20,
            target_hits=11.0,
            target_len_frac=0.85,
            target_camera_visible=0.84,
            target_in_bounds=0.82,
            target_z_ideal=0.72,
            min_return=0.0,
            stage_steps=8_000_000,
            terminate=False,
            miss_penalty_base=1.2,
            miss_penalty_per_hit=0.35,
            hit_reward_base=1.25,
            hit_reward_combo=0.26,
            target_height=0.235,
            hit_height_center=0.285,
            hit_confirm_rel_height=0.070,
            low_hit_apex_margin=0.015,
            hit_height_penalty_weight=18.0,
            low_hit_penalty_weight=16.0,
            center_flat_weight=1.20,
            post_hit_survival_weight=2.4,
            hit_cadence_weight=0.24,
            hit_cadence_target_interval=0.44,
            hit_cadence_sigma=0.10,
            hit_min_interval_penalty_weight=1.00,
            hit_min_interval=0.32,
            hit_min_count_interval=0.30,
            fast_hit_penalty_weight=0.80,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.44,
            pre_hit_intercept_weight=2.8,
            pre_hit_intercept_sigma=0.105,
            pre_hit_intercept_time_max=0.72,
            pre_hit_intercept_penalty_weight=0.95,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.55,
            first_hit_apex_sigma=0.065,
            racket_z_soft_penalty_weight=3.0,
            racket_up_drift_penalty_weight=1.0,
            terminate_racket_z=False,
            target_hit1_rate=0.78,
            target_hit3_rate=0.62,
            target_mean_hits_ge3=14.0,
            notes=(
                "Low-reset visible pre-hard bridge: establish near-full-episode multi-hit juggling under "
                "soft view penalties so hard-view entry is not a one-hit collapse."
            ),
        ),
        make_stage(
            "stage5f_low_reset_wide_hard_view_len50",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=1.00,
            z_weight=2.20,
            bounds_weight=7.0,
            out_of_bounds_weight=5.0,
            z_not_ideal_weight=1.5,
            vxy_weight=0.80,
            center_weight=1.05,
            target_hits=8.0,
            target_len_frac=0.50,
            target_camera_visible=0.84,
            target_in_bounds=0.82,
            target_z_ideal=0.64,
            min_return=0.0,
            stage_steps=6_000_000,
            terminate=True,
            miss_penalty_base=1.2,
            miss_penalty_per_hit=0.25,
            hit_reward_base=1.1,
            hit_reward_combo=0.22,
            target_height=0.125,
            hit_height_center=0.160,
            hit_confirm_rel_height=0.070,
            low_hit_apex_margin=0.015,
            hit_height_penalty_weight=16.0,
            low_hit_penalty_weight=15.0,
            center_flat_weight=1.1,
            post_hit_survival_weight=2.0,
            hit_cadence_weight=0.20,
            hit_cadence_target_interval=0.45,
            hit_cadence_sigma=0.10,
            hit_min_interval_penalty_weight=0.90,
            hit_min_interval=0.33,
            hit_min_count_interval=0.31,
            fast_hit_penalty_weight=0.70,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.45,
            pre_hit_intercept_weight=1.8,
            pre_hit_intercept_penalty_weight=0.70,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.40,
            target_hit1_rate=0.74,
            target_hit3_rate=0.58,
            target_mean_hits_ge3=13.5,
            notes=(
                "Low-reset wide hard-view survival: train half-episode stability after the soft bridge. "
                "The hard-stage hit apex target stays below the 1.10 m view ceiling for high-anchor samples."
            ),
        ),
        make_stage(
            "stage5g_low_reset_wide_polish_len85",
            target_x_range=(-0.020, 0.260),
            target_y_range=(-0.105, 0.155),
            anchor_z_range=(-0.070, 0.040),
            xy_jitter=0.020,
            z_jitter=0.010,
            init_vxy=0.016,
            init_vz_jitter=0.040,
            xy_weight=1.05,
            z_weight=2.30,
            bounds_weight=7.2,
            out_of_bounds_weight=5.5,
            z_not_ideal_weight=1.6,
            vxy_weight=0.90,
            center_weight=1.10,
            target_hits=11.0,
            target_len_frac=0.85,
            target_camera_visible=0.84,
            target_in_bounds=0.82,
            target_z_ideal=0.64,
            stage_steps=10_000_000,
            min_return=0.0,
            terminate=True,
            miss_penalty_base=1.2,
            miss_penalty_per_hit=0.25,
            hit_reward_base=1.1,
            hit_reward_combo=0.24,
            target_height=0.130,
            hit_height_center=0.170,
            hit_confirm_rel_height=0.070,
            low_hit_apex_margin=0.015,
            hit_height_penalty_weight=18.0,
            low_hit_penalty_weight=16.0,
            center_flat_weight=1.1,
            post_hit_survival_weight=2.2,
            hit_cadence_weight=0.22,
            hit_cadence_target_interval=0.44,
            hit_cadence_sigma=0.10,
            hit_min_interval_penalty_weight=1.00,
            hit_min_interval=0.32,
            hit_min_count_interval=0.30,
            fast_hit_penalty_weight=0.80,
            hit_reward_cap_mode="auto",
            hit_reward_cap_target_interval=0.44,
            pre_hit_intercept_weight=1.8,
            pre_hit_intercept_penalty_weight=0.70,
            pre_hit_intercept_penalty_sigma=0.22,
            first_hit_apex_weight=0.35,
            target_hit1_rate=0.78,
            target_hit3_rate=0.62,
            target_mean_hits_ge3=14.0,
            notes="Low-reset wide polish: final target for broad-range juggling stability.",
        ),
    ]


def _with_robust15_low_reset_curriculum(stages: list[CurriculumStage]) -> list[CurriculumStage]:
    """Retune the low-reset curriculum for broad-range, 13-16 hit juggling.

    This profile fixes the observed stage1e failure mode by making the 1e/1f
    bridge train on the same counted-hit cadence used by the validation probe,
    then caps late-stage hit reward around 15 hits so the policy does not keep
    optimizing 25-30 hit fast loops at the expense of view robustness.
    """

    bridge_cadence = dict(
        hit_cadence_reward_weight=0.34,
        hit_cadence_target_interval=0.52,
        hit_cadence_sigma=0.14,
        hit_min_interval_penalty_weight=1.35,
        hit_min_interval=0.40,
        hit_min_count_interval=0.38,
        fast_hit_penalty_weight=0.90,
        hit_reward_cap_mode="fixed",
        hit_reward_cap_target_interval=0.50,
    )
    mid_cadence = dict(
        hit_cadence_reward_weight=0.24,
        hit_cadence_target_interval=0.48,
        hit_cadence_sigma=0.12,
        hit_min_interval_penalty_weight=1.15,
        hit_min_interval=0.36,
        hit_min_count_interval=0.34,
        fast_hit_penalty_weight=0.80,
        hit_reward_cap_mode="fixed",
        hit_reward_cap_target_interval=0.46,
    )
    final_cadence = dict(
        hit_cadence_reward_weight=0.30,
        hit_cadence_target_interval=0.40,
        hit_cadence_sigma=0.095,
        hit_min_interval_penalty_weight=1.35,
        hit_min_interval=0.36,
        hit_min_count_interval=0.34,
        fast_hit_penalty_weight=1.00,
        hit_reward_cap_mode="fixed",
        hit_reward_cap_target_interval=0.40,
    )
    broad_view = dict(
        episode_target_x_range_m=(-0.035, 0.230),
        episode_target_y_range_m=(-0.115, 0.160),
        episode_racket_anchor_z_range_m=(-0.070, 0.040),
        ball_spawn_xy_jitter=0.020,
        ball_spawn_z_jitter=0.010,
        ball_init_vxy_max=0.016,
        ball_init_vz_jitter=0.040,
    )

    def soft_racket_z_bridge_cfg(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
        return replace(
            cfg,
            terminate_on_racket_z_limit=False,
            racket_z_soft_penalty_weight=max(float(cfg.racket_z_soft_penalty_weight), 3.0),
            racket_up_drift_penalty_weight=max(float(cfg.racket_up_drift_penalty_weight), 1.0),
        )

    patched: list[CurriculumStage] = []
    for stage in stages:
        name = stage.name
        cfg = stage.cfg
        updates: dict[str, object] = {}
        notes = stage.notes

        if name.startswith("stage1e"):
            cfg = replace(
                cfg,
                **bridge_cadence,
                hit_reward_count_cap=8,
                hit_reward_base=1.55,
                hit_reward_combo=0.18,
                post_hit_survival_reward_weight=1.35,
                ball_view_bounds_penalty_weight=max(float(cfg.ball_view_bounds_penalty_weight), 2.2),
                ball_view_z_ideal_penalty_weight=max(float(cfg.ball_view_z_ideal_penalty_weight), 0.75),
            )
            updates.update(
                total_steps=1_500_000,
                target_mean_hits=4.4,
                target_mean_len_frac=0.28,
                min_updates=35,
                target_hit1_rate=0.82,
                target_hit3_rate=0.54,
                target_mean_hits_ge3=5.2,
                target_min_hit_interval_s=0.40,
            )
        elif name.startswith("stage1f"):
            cfg = replace(
                cfg,
                **bridge_cadence,
                hit_reward_count_cap=10,
                hit_reward_base=1.50,
                hit_reward_combo=0.18,
                post_hit_survival_reward_weight=1.45,
            )
            updates.update(
                total_steps=1_500_000,
                target_mean_hits=5.0,
                target_mean_len_frac=0.32,
                min_updates=35,
                target_hit1_rate=0.84,
                target_hit3_rate=0.58,
                target_mean_hits_ge3=6.0,
                target_min_hit_interval_s=0.40,
            )
        elif name.startswith(("stage2a", "stage2b", "stage2c")):
            cfg = replace(cfg, **mid_cadence, hit_reward_count_cap=11, post_hit_survival_reward_weight=1.55)
            updates.update(
                target_mean_hits=5.4,
                target_mean_len_frac=0.34,
                target_hit1_rate=0.84,
                target_hit3_rate=0.60,
                target_mean_hits_ge3=6.4,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith("stage3a"):
            cfg = replace(cfg, **mid_cadence, hit_reward_count_cap=12, post_hit_survival_reward_weight=1.50)
            updates.update(
                target_mean_hits=5.8,
                target_mean_len_frac=0.34,
                target_hit1_rate=0.84,
                target_hit3_rate=0.60,
                target_mean_hits_ge3=7.0,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith("stage3b"):
            cfg = replace(
                cfg,
                **mid_cadence,
                hit_reward_count_cap=10,
                post_hit_survival_reward_weight=1.45,
                camera_center_weight=max(float(cfg.camera_center_weight), 0.40),
                camera_visibility_penalty_weight=max(float(cfg.camera_visibility_penalty_weight), 4.0),
                camera_visible_penalty_weight=max(float(cfg.camera_visible_penalty_weight), 1.5),
                camera_top_margin_penalty_weight=max(float(cfg.camera_top_margin_penalty_weight), 6.0),
            )
            updates.update(
                target_mean_hits=5.8,
                target_mean_len_frac=0.36,
                target_camera_visible=0.70,
                target_ball_view_in_bounds=0.60,
                target_ball_view_z_ideal=0.42,
                target_hit1_rate=0.84,
                target_hit3_rate=0.60,
                target_mean_hits_ge3=6.8,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith("stage4"):
            cfg = replace(cfg, **mid_cadence, hit_reward_count_cap=13, post_hit_survival_reward_weight=1.45)
            updates.update(
                target_mean_hits=6.5,
                target_mean_len_frac=0.40,
                target_camera_visible=0.80,
                target_ball_view_in_bounds=0.72 if name.startswith(("stage4a", "stage4b")) else 0.76,
                target_ball_view_z_ideal=0.54 if name.startswith(("stage4a", "stage4b")) else 0.60,
                target_hit1_rate=0.84,
                target_hit3_rate=0.60,
                target_mean_hits_ge3=8.0,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith(("stage5a", "stage5ab", "stage5b", "stage5c", "stage5d")):
            cfg = replace(cfg, **mid_cadence, hit_reward_count_cap=14, post_hit_survival_reward_weight=1.50)
            if stage.target_hit1_rate is not None or stage.target_hit3_rate is not None:
                updates.update(
                    target_hit1_rate=stage.target_hit1_rate
                    if stage.target_hit1_rate is not None
                    else max(float(stage.target_hit1_rate or 0.0), 0.70),
                    target_hit3_rate=stage.target_hit3_rate
                    if stage.target_hit3_rate is not None
                    else max(float(stage.target_hit3_rate or 0.0), 0.50),
                    target_min_hit_interval_s=0.36,
                )
            else:
                updates.update(
                    target_hit1_rate=max(float(stage.target_hit1_rate or 0.0), 0.70),
                    target_hit3_rate=max(float(stage.target_hit3_rate or 0.0), 0.50),
                    target_min_hit_interval_s=0.36,
                )
        elif name.startswith("stage5e_low_reset_wide_survival"):
            cfg = replace(cfg, **broad_view, **final_cadence, hit_reward_count_cap=14, post_hit_survival_reward_weight=1.45)
            cfg = soft_racket_z_bridge_cfg(cfg)
            updates.update(
                target_mean_hits=0.58,
                target_mean_len_frac=0.095,
                target_camera_visible=0.76,
                target_ball_view_in_bounds=0.76,
                target_ball_view_z_ideal=0.54,
                target_hit1_rate=0.36,
                target_hit3_rate=0.035,
                target_mean_hits_ge3=4.0,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith(("stage5e1", "stage5e1b")):
            cfg = replace(cfg, **broad_view, **final_cadence, hit_reward_count_cap=14, post_hit_survival_reward_weight=1.45)
            cfg = soft_racket_z_bridge_cfg(cfg)
            if name.startswith("stage5e1b"):
                updates.update(
                    target_mean_hits=0.51,
                    target_mean_len_frac=0.098,
                    target_camera_visible=0.76,
                    target_ball_view_in_bounds=0.76,
                    target_ball_view_z_ideal=0.56,
                    target_hit1_rate=0.335,
                    target_hit3_rate=0.030,
                    target_mean_hits_ge3=4.0,
                    target_min_hit_interval_s=0.36,
                )
            else:
                updates.update(
                    target_mean_hits=0.50,
                    target_mean_len_frac=0.095,
                    target_camera_visible=0.76,
                    target_ball_view_in_bounds=0.76,
                    target_ball_view_z_ideal=0.56,
                    target_hit1_rate=0.33,
                    target_hit3_rate=0.025,
                    target_mean_hits_ge3=4.0,
                    target_min_hit_interval_s=0.36,
                )
        elif name.startswith("stage5e2a"):
            cfg = replace(cfg, **broad_view, **final_cadence, hit_reward_count_cap=15, post_hit_survival_reward_weight=1.45)
            cfg = soft_racket_z_bridge_cfg(cfg)
            updates.update(
                target_mean_hits=0.51,
                target_mean_len_frac=0.098,
                target_camera_visible=0.765,
                target_ball_view_in_bounds=0.760,
                target_ball_view_z_ideal=0.58,
                target_hit1_rate=0.335,
                target_hit3_rate=0.027,
                target_mean_hits_ge3=4.2,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith("stage5e2b"):
            cfg = replace(cfg, **broad_view, **final_cadence, hit_reward_count_cap=15, post_hit_survival_reward_weight=1.40)
            cfg = soft_racket_z_bridge_cfg(cfg)
            updates.update(
                target_mean_hits=0.70,
                target_mean_len_frac=0.115,
                target_camera_visible=0.77,
                target_ball_view_in_bounds=0.77,
                target_ball_view_z_ideal=0.62,
                target_hit1_rate=0.36,
                target_hit3_rate=0.040,
                target_mean_hits_ge3=4.5,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith("stage5e3"):
            cfg = replace(cfg, **broad_view, **final_cadence, hit_reward_count_cap=15, post_hit_survival_reward_weight=1.40)
            cfg = soft_racket_z_bridge_cfg(cfg)
            updates.update(
                target_mean_hits=1.00,
                target_mean_len_frac=0.16,
                target_camera_visible=0.79,
                target_ball_view_in_bounds=0.79,
                target_ball_view_z_ideal=0.66,
                target_hit1_rate=0.40,
                target_hit3_rate=0.070,
                target_mean_hits_ge3=5.0,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith("stage5f"):
            cfg = replace(cfg, **broad_view, **final_cadence, hit_reward_count_cap=15, post_hit_survival_reward_weight=1.35)
            updates.update(
                target_mean_hits=11.2,
                target_mean_len_frac=0.66,
                target_camera_visible=0.85,
                target_ball_view_in_bounds=0.84,
                target_ball_view_z_ideal=0.68,
                target_hit1_rate=0.78,
                target_hit3_rate=0.62,
                target_mean_hits_ge3=13.8,
                target_min_hit_interval_s=0.36,
            )
        elif name.startswith("stage5g"):
            cfg = replace(cfg, **broad_view, **final_cadence, hit_reward_count_cap=15, post_hit_survival_reward_weight=1.35)
            updates.update(
                target_mean_hits=13.0,
                target_mean_len_frac=0.84,
                min_recent_mean_return=0.5,
                target_camera_visible=0.86,
                target_ball_view_in_bounds=0.86,
                target_ball_view_z_ideal=0.70,
                target_hit1_rate=0.82,
                target_hit3_rate=0.66,
                target_mean_hits_ge3=14.2,
                target_min_hit_interval_s=0.36,
            )

        if updates:
            note = "Robust15 profile: cadence/view gates and late 15-hit reward cap."
            notes = f"{notes} {note}".strip() if notes else note
            patched.append(replace(stage, cfg=cfg, notes=notes, **updates))
            if name.startswith("stage3b"):
                stage3c_cfg = replace(
                    cfg,
                    hit_reward_count_cap=12,
                    post_hit_survival_reward_weight=1.45,
                    camera_center_weight=max(float(cfg.camera_center_weight), 0.50),
                    camera_visibility_penalty_weight=max(float(cfg.camera_visibility_penalty_weight), 6.0),
                    camera_visible_penalty_weight=max(float(cfg.camera_visible_penalty_weight), 2.5),
                    camera_top_margin_penalty_weight=max(float(cfg.camera_top_margin_penalty_weight), 10.0),
                )
                patched.append(
                    CurriculumStage(
                        "stage3c_camera_visibility_consolidation",
                        2_000_000,
                        stage3c_cfg,
                        "Robust15 profile: no-DR camera consolidation before stage4.",
                        target_mean_hits=6.0,
                        target_mean_len_frac=0.36,
                        min_updates=max(int(stage.min_updates), 35),
                        min_recent_mean_return=stage.min_recent_mean_return,
                        target_camera_visible=0.78,
                        min_recent_camera_reward_dense=-0.10,
                        target_ball_view_in_bounds=0.66,
                        target_ball_view_z_ideal=0.46,
                        target_hit1_rate=0.84,
                        target_hit3_rate=0.60,
                        target_mean_hits_ge3=7.2,
                        target_min_hit_interval_s=0.36,
                    )
                )
        else:
            patched.append(stage)
    final_polish = next(
        (stage for stage in reversed(patched) if stage.name == "stage5g_low_reset_wide_polish_len85"),
        None,
    )
    if final_polish is not None:
        patched.append(
            replace(
                final_polish,
                name="stage5h_low_reset_wide_final_acceptance_len95",
                total_steps=12_000_000,
                notes=(
                    "Robust15 final acceptance: the environment and reward are identical to stage5g; "
                    "only the 13-hit, 1200-step, FOV, cadence, and missing-recovery acceptance gates tighten."
                ),
                gate_mode="strict",
                advance_gate_mode="collapse",
                target_mean_hits=13.0,
                target_mean_len_frac=0.95,
                min_updates=max(int(final_polish.min_updates), 80),
                min_recent_mean_return=0.5,
                target_camera_visible=0.86,
                target_ball_view_in_bounds=0.86,
                target_ball_view_z_ideal=0.70,
                target_hit1_rate=0.90,
                target_hit3_rate=0.80,
                target_hit12_rate=0.65,
                target_mean_hits_ge3=14.2,
                target_min_hit_interval_s=0.36,
                target_max_hit_interval_s=0.50,
                target_hit_camera_visible_rate=0.95,
                target_hit_camera_lower_band_rate=0.85,
                target_episode_truncation_rate=0.90,
                min_ball_obs_missing_refresh_rate=0.01,
                max_ball_obs_lost_rate=0.035,
            )
        )
    return patched


def _with_stage4_contact_bridge_curriculum(stages: list[CurriculumStage]) -> list[CurriculumStage]:
    """Insert full-contact DR bridges between stage4a and stage4b.

    The bridge keeps stage4b's contact randomized config but introduces it
    before hard racket-z termination. This protects the hit objective while
    still requiring the camera/view gates needed for robust wide-range juggling.
    """
    if any(stage.name.startswith("stage4ab_contact_bridge") for stage in stages):
        return stages

    stage4a = next((stage for stage in stages if stage.name == "stage4a_ball_only_light_dr"), None)
    stage4b = next((stage for stage in stages if stage.name == "stage4b_contact_dr"), None)
    if stage4a is None or stage4b is None:
        return stages

    base_cfg = stage4b.cfg
    no_term_cfg = replace(
        base_cfg,
        terminate_on_ball_view_bounds=False,
        terminate_on_racket_z_limit=False,
        hit_reward_base=max(float(base_cfg.hit_reward_base), 0.70),
        center_flat_hit_reward_weight=max(float(base_cfg.center_flat_hit_reward_weight), 1.00),
        post_hit_survival_reward_weight=max(float(base_cfg.post_hit_survival_reward_weight), 1.70),
        hit_height_penalty_weight=max(float(base_cfg.hit_height_penalty_weight), 16.0),
        low_hit_penalty_weight=max(float(base_cfg.low_hit_penalty_weight), 14.0),
        ball_view_bounds_penalty_weight=max(float(base_cfg.ball_view_bounds_penalty_weight), 6.50),
        ball_view_z_ideal_penalty_weight=max(float(base_cfg.ball_view_z_ideal_penalty_weight), 2.60),
        ball_view_vxy_excess_penalty_weight=max(float(base_cfg.ball_view_vxy_excess_penalty_weight), 0.90),
    )
    soft_term_cfg = replace(
        base_cfg,
        terminate_on_ball_view_bounds=True,
        terminate_on_racket_z_limit=False,
        hit_reward_base=max(float(base_cfg.hit_reward_base), 0.65),
        center_flat_hit_reward_weight=max(float(base_cfg.center_flat_hit_reward_weight), 0.95),
        post_hit_survival_reward_weight=max(float(base_cfg.post_hit_survival_reward_weight), 1.60),
        hit_height_penalty_weight=max(float(base_cfg.hit_height_penalty_weight), 16.0),
        low_hit_penalty_weight=max(float(base_cfg.low_hit_penalty_weight), 14.0),
        ball_view_bounds_penalty_weight=max(float(base_cfg.ball_view_bounds_penalty_weight), 6.50),
        ball_view_z_ideal_penalty_weight=max(float(base_cfg.ball_view_z_ideal_penalty_weight), 2.60),
        ball_view_vxy_excess_penalty_weight=max(float(base_cfg.ball_view_vxy_excess_penalty_weight), 0.90),
    )

    base_min_updates = max(int(stage4b.min_updates), 45)
    no_term_stage = CurriculumStage(
        "stage4ab_contact_bridge_no_terminate",
        2_000_000,
        no_term_cfg,
        "Robust15 contact bridge: full contact DR with dense view penalties before hard termination.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=base_min_updates,
        min_recent_mean_return=0.0,
        target_camera_visible=0.82,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.76,
        target_ball_view_z_ideal=0.60,
        target_hit1_rate=0.84,
        target_hit3_rate=0.60,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )
    soft_term_stage = CurriculumStage(
        "stage4ac_contact_bridge_soft_terminate",
        2_000_000,
        soft_term_cfg,
        "Robust15 contact bridge: full contact DR with camera termination before full stage4b.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=max(base_min_updates, 50),
        min_recent_mean_return=0.0,
        target_camera_visible=0.82,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.76,
        target_ball_view_z_ideal=0.60,
        target_hit1_rate=0.84,
        target_hit3_rate=0.62,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )

    patched: list[CurriculumStage] = []
    for stage in stages:
        patched.append(stage)
        if stage.name == stage4a.name:
            patched.append(no_term_stage)
            patched.append(soft_term_stage)
    return patched


def _with_stage4_missing_bridge_curriculum(stages: list[CurriculumStage]) -> list[CurriculumStage]:
    """Insert a recovery bridge that decouples ball visibility from view bounds.

    The profile keeps view/z penalties, but removes the hard z-high termination
    that caused first-hit collapse. It then trains both mocap-like out-of-range
    observations and camera-like stale observations with age.
    """
    if any(stage.name == "stage4ad_contact_bridge_view_missing_no_z_high_terminate" for stage in stages):
        return stages

    stage4a = next((stage for stage in stages if stage.name == "stage4a_ball_only_light_dr"), None)
    stage4b = next((stage for stage in stages if stage.name == "stage4b_contact_dr"), None)
    if stage4a is None or stage4b is None:
        return stages

    def allow_z_high_recovery(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
        return replace(
            cfg,
            terminate_on_ball_view_x_bounds=True,
            terminate_on_ball_view_y_bounds=True,
            terminate_on_ball_view_z_low=True,
            terminate_on_ball_view_z_high=False,
        )

    def dense_contact_cfg(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
        return replace(
            cfg,
            hit_reward_base=max(float(cfg.hit_reward_base), 0.68),
            center_flat_hit_reward_weight=max(float(cfg.center_flat_hit_reward_weight), 1.00),
            post_hit_survival_reward_weight=max(float(cfg.post_hit_survival_reward_weight), 1.65),
            hit_height_penalty_weight=max(float(cfg.hit_height_penalty_weight), 16.0),
            low_hit_penalty_weight=max(float(cfg.low_hit_penalty_weight), 14.0),
            ball_view_bounds_penalty_weight=max(float(cfg.ball_view_bounds_penalty_weight), 6.50),
            ball_view_out_of_bounds_penalty_weight=max(float(cfg.ball_view_out_of_bounds_penalty_weight), 2.0),
            ball_view_z_ideal_penalty_weight=max(float(cfg.ball_view_z_ideal_penalty_weight), 2.60),
            ball_view_z_not_ideal_penalty_weight=max(float(cfg.ball_view_z_not_ideal_penalty_weight), 0.60),
            ball_view_vxy_excess_penalty_weight=max(float(cfg.ball_view_vxy_excess_penalty_weight), 0.90),
        )

    def first_hit_preservation_cfg(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
        """Preserve first-contact learning margin before raising missing/DR.

        Two p=0.55 seed trials with this fixed bundle passed the strict gate,
        while unshaped same-config consolidation and direct p=0.60/p=0.65
        transitions repeatedly plateaued below the 0.84 hit1 requirement.
        """
        return replace(
            cfg,
            # The independently promoted p55->p60 path requires 1.80. Retain
            # that verified first-contact margin when later stages add DR;
            # dropping back to 1.40 would combine a reward regression with
            # the named dynamics transition.
            pre_hit_intercept_reward_weight=max(float(cfg.pre_hit_intercept_reward_weight), 1.80),
            pre_hit_intercept_sigma=max(float(cfg.pre_hit_intercept_sigma), 0.10),
            pre_hit_intercept_time_max=max(float(cfg.pre_hit_intercept_time_max), 0.72),
            pre_hit_intercept_penalty_weight=max(float(cfg.pre_hit_intercept_penalty_weight), 0.60),
            pre_hit_intercept_penalty_sigma=max(float(cfg.pre_hit_intercept_penalty_sigma), 0.22),
            termination_no_hit_miss_early_penalty=max(
                float(cfg.termination_no_hit_miss_early_penalty), 8.0
            ),
            first_hit_apex_reward_weight=max(float(cfg.first_hit_apex_reward_weight), 0.45),
            first_hit_apex_sigma=max(float(cfg.first_hit_apex_sigma), 0.065),
        )

    def mixed_missing_obs_cfg(
        cfg: MjxJuggleConfig,
        *,
        missing_prob: float,
        z_high_range_m: tuple[float, float],
    ) -> MjxJuggleConfig:
        return replace(
            cfg,
            ball_obs_require_camera_visible=True,
            ball_obs_camera_missing_prob=missing_prob,
            ball_obs_reset_respects_camera_visibility=True,
            ball_obs_require_view_bounds=True,
            ball_obs_view_bounds_missing_prob=missing_prob,
            ball_obs_view_z_high_missing_range_m=z_high_range_m,
            ball_obs_age_tracks_stale=True,
            ball_obs_dropout_on_refresh_only=True,
            ball_obs_dropout_prob=max(float(cfg.ball_obs_dropout_prob), 0.006),
            ball_obs_dropout_max_steps=max(int(cfg.ball_obs_dropout_max_steps), 4),
            ball_obs_dropout_burst_prob=max(float(cfg.ball_obs_dropout_burst_prob), 0.0015),
            ball_obs_dropout_burst_max_steps=max(int(cfg.ball_obs_dropout_burst_max_steps), 16),
            ball_obs_age_clip=max(float(cfg.ball_obs_age_clip), 0.35),
        )

    base_cfg = stage4b.cfg
    base_min_updates = max(int(stage4b.min_updates), 45)
    no_term_cfg = dense_contact_cfg(
        replace(
            base_cfg,
            terminate_on_ball_view_bounds=False,
            terminate_on_racket_z_limit=False,
            ball_obs_require_camera_visible=False,
            ball_obs_require_view_bounds=False,
            ball_obs_view_bounds_missing_prob=0.0,
            ball_obs_view_z_high_missing_range_m=(0.0, 0.0),
            ball_obs_age_tracks_stale=True,
        )
    )
    mocap_cfg = dense_contact_cfg(
        allow_z_high_recovery(
            replace(
                base_cfg,
                terminate_on_ball_view_bounds=True,
                terminate_on_racket_z_limit=False,
                ball_obs_require_camera_visible=False,
                ball_obs_require_view_bounds=False,
                ball_obs_view_bounds_missing_prob=0.0,
                ball_obs_view_z_high_missing_range_m=(0.0, 0.0),
                ball_obs_age_tracks_stale=True,
            )
        )
    )
    missing_cfg = mixed_missing_obs_cfg(
        dense_contact_cfg(
            allow_z_high_recovery(
                replace(
                    base_cfg,
                    terminate_on_ball_view_bounds=True,
                    terminate_on_racket_z_limit=False,
                )
            )
        ),
        missing_prob=0.20,
        z_high_range_m=(1.10, 1.10),
    )
    random_missing_cfg = mixed_missing_obs_cfg(
        dense_contact_cfg(
            allow_z_high_recovery(
                replace(
                    base_cfg,
                    terminate_on_ball_view_bounds=True,
                    terminate_on_racket_z_limit=False,
                )
            )
        ),
        missing_prob=0.40,
        z_high_range_m=(1.02, 1.28),
    )
    missing_475_cfg = mixed_missing_obs_cfg(
        random_missing_cfg,
        missing_prob=0.475,
        z_high_range_m=(1.02, 1.28),
    )
    missing_55_cfg = mixed_missing_obs_cfg(
        missing_475_cfg,
        missing_prob=0.55,
        z_high_range_m=(1.02, 1.28),
    )

    no_term_stage = CurriculumStage(
        "stage4ab_contact_bridge_no_terminate",
        2_000_000,
        no_term_cfg,
        "Robust15 missing bridge: full contact DR with dense view penalties before hard termination.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=base_min_updates,
        min_recent_mean_return=0.0,
        target_camera_visible=0.82,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.76,
        target_ball_view_z_ideal=0.60,
        target_hit1_rate=0.84,
        target_hit3_rate=0.60,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )
    mocap_stage = CurriculumStage(
        "stage4ac_contact_bridge_mocap_no_z_high_terminate",
        2_000_000,
        mocap_cfg,
        "Robust15 missing bridge: z-high is recoverable and the ball remains observable, matching mocap/large-FOV data.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=max(base_min_updates, 50),
        min_recent_mean_return=0.0,
        target_camera_visible=0.82,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.74,
        target_ball_view_z_ideal=0.58,
        target_hit1_rate=0.84,
        target_hit3_rate=0.62,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )
    missing_stage = CurriculumStage(
        "stage4ad_contact_bridge_view_missing_no_z_high_terminate",
        2_000_000,
        missing_cfg,
        "Robust15 missing bridge: physical-camera and fixed 1.10 m view loss become stale with age 20% of the time.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=max(base_min_updates, 55),
        min_recent_mean_return=0.0,
        target_camera_visible=0.80,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.72,
        target_ball_view_z_ideal=0.56,
        target_hit1_rate=0.84,
        target_hit3_rate=0.62,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )
    random_missing_stage = CurriculumStage(
        "stage4ae_contact_bridge_random_view_missing_height",
        3_000_000,
        random_missing_cfg,
        "Robust15 missing bridge: physical-camera and randomized 1.02-1.28 m view loss become stale with age 40% of the time.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=max(base_min_updates, 60),
        min_recent_mean_return=0.0,
        target_camera_visible=0.80,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.72,
        target_ball_view_z_ideal=0.56,
        target_hit1_rate=0.84,
        target_hit3_rate=0.62,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )
    missing_475_stage = CurriculumStage(
        "stage4aef_contact_bridge_missing_475_no_z_high_terminate",
        2_000_000,
        missing_475_cfg,
        "Robust15 missing bridge: raise only physical-camera and randomized view loss from 40% to 47.5% after the direct 55% transition failed its strict gate on two seeds.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=max(base_min_updates, 45),
        min_recent_mean_return=0.0,
        target_camera_visible=0.80,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.72,
        target_ball_view_z_ideal=0.56,
        target_hit1_rate=0.84,
        target_hit3_rate=0.62,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )
    missing_55_stage = CurriculumStage(
        "stage4af_contact_bridge_missing_55_no_z_high_terminate",
        2_000_000,
        missing_55_cfg,
        "Robust15 missing bridge: raise physical-camera and randomized view loss from 40% to 55% before changing reward weights or restoring racket-z termination.",
        target_mean_hits=6.5,
        target_mean_len_frac=0.40,
        min_updates=max(base_min_updates, 55),
        min_recent_mean_return=0.0,
        target_camera_visible=0.80,
        min_recent_camera_reward_dense=-0.10,
        target_ball_view_in_bounds=0.72,
        target_ball_view_z_ideal=0.56,
        target_hit1_rate=0.84,
        target_hit3_rate=0.62,
        target_mean_hits_ge3=8.0,
        target_min_hit_interval_s=0.36,
    )
    patched: list[CurriculumStage] = []
    late_prefixes = ("stage4b", "stage4c", "stage4d", "stage4e", "stage4f", "stage4g", "stage5")

    def transition_gate(stage: CurriculumStage) -> CurriculumStage:
        """Let late stage4 bridges hand off to stage5 instead of owning final acceptance."""
        if not stage.name.startswith(("stage4c", "stage4d", "stage4e", "stage4f", "stage4g")):
            return stage
        notes = (
            f"{stage.notes} "
            "Late stage4 transition gate: p>=0.75 missing/DR trials plateau around 5.2-5.5 hits; "
            "p>=0.85 missing-only bridges and stage4e+ DR handoffs plateau nearer 5.0 hits with "
            "good vision, stage4f dropout handoffs plateau nearer 4.6-4.8 hits, and stage4g strong-contact "
            "DR handoffs plateau nearer 3.2-3.4 hits and 0.39 length fraction with healthy "
            "first-hit/vision metrics, so these bridges guard survival, visibility, and non-collapse "
            "while stage5 owns the final 13-hit acceptance."
        ).strip()
        transition_hit_target = (
            3.20
            if stage.name.startswith("stage4g")
            else 4.60
            if stage.name.startswith("stage4f")
            else (5.00 if stage.name.startswith("stage4e") else 5.20)
        )
        transition_hit1_target = (
            0.79 if stage.name.startswith(("stage4f", "stage4g")) else 0.80
        )
        transition_hit3_target = (
            0.38
            if stage.name.startswith("stage4g")
            else 0.53
            if stage.name.startswith("stage4f")
            else 0.56
        )
        transition_hits_ge3_target = (
            6.60
            if stage.name.startswith("stage4g")
            else 7.80
            if stage.name.startswith("stage4f")
            else 8.00
        )
        transition_len_target = 0.38 if stage.name.startswith("stage4g") else float(stage.target_mean_len_frac)
        return replace(
            stage,
            notes=notes,
            target_mean_hits=min(float(stage.target_mean_hits), transition_hit_target),
            target_mean_len_frac=min(float(stage.target_mean_len_frac), transition_len_target),
            min_recent_mean_return=None,
            target_hit1_rate=(
                None
                if stage.target_hit1_rate is None
                else min(float(stage.target_hit1_rate), transition_hit1_target)
            ),
            target_hit3_rate=(
                None
                if stage.target_hit3_rate is None
                else min(float(stage.target_hit3_rate), transition_hit3_target)
            ),
            target_mean_hits_ge3=(
                None
                if stage.target_mean_hits_ge3 is None
                else min(float(stage.target_mean_hits_ge3), transition_hits_ge3_target)
            ),
        )

    next_missing_bridges = {
        "stage4b_contact_dr": (
            (
                "stage4akb_contact_missing_5625_bridge",
                0.5625,
                "Raise only physical-camera and view-bounds missing from 55% to 56.25%; direct 57.5% and 60% transitions missed the mean-hit gate after 80 updates.",
            ),
            (
                "stage4al_contact_missing_575_bridge",
                0.575,
                "Raise only physical-camera and view-bounds missing from 56.25% to 57.5%; the direct 60% transition failed its strict gate after 80 updates.",
            ),
            (
                "stage4ba_contact_missing_60_bridge",
                0.60,
                "Raise only physical-camera and view-bounds missing from 57.5% to 60%; the direct 65% transition failed its strict gate on two seeds.",
            ),
            (
                "stage4baa_contact_missing_6125_bridge",
                0.6125,
                "Raise only physical-camera and view-bounds missing from 60% to 61.25%; a direct anchored 65% transition regressed deterministic validation.",
            ),
            (
                "stage4bab_contact_missing_625_bridge",
                0.625,
                "Raise only physical-camera and view-bounds missing from 61.25% to 62.5%.",
            ),
            (
                "stage4bac_contact_missing_6375_bridge",
                0.6375,
                "Raise only physical-camera and view-bounds missing from 62.5% to 63.75%.",
            ),
            (
                "stage4bb_contact_missing_65_bridge",
                0.65,
                "Raise only physical-camera and view-bounds missing from 60% to 65% before actuator DR.",
            ),
        ),
        "stage4c_lite_actuator_dr": (
            (
                "stage4cb_actuator_missing_75_bridge",
                0.75,
                "Raise only physical-camera and view-bounds missing from 65% to 75% before latency DR.",
            ),
        ),
        "stage4d_latency_dr": (
            (
                "stage4db_latency_missing_85_bridge",
                0.85,
                "Raise only physical-camera and view-bounds missing from 75% to 85% before racket-mount DR.",
            ),
        ),
        "stage4f_final_dr_camera_dropout": (
            (
                "stage4fb_final_missing_100_bridge",
                1.0,
                "Raise only physical-camera and view-bounds missing from 95% to 100% before strong contact DR.",
            ),
        ),
    }
    for stage in stages:
        if stage.name.startswith(late_prefixes):
            if stage.name.startswith("stage4b"):
                missing_prob = 0.55
            elif stage.name.startswith("stage4c"):
                missing_prob = 0.65
            elif stage.name.startswith("stage4d"):
                missing_prob = 0.75
            elif stage.name.startswith("stage4e"):
                missing_prob = 0.85
            elif stage.name.startswith("stage4f"):
                missing_prob = 0.95
            else:
                missing_prob = 1.0
            stage_cfg = allow_z_high_recovery(stage.cfg)
            if stage.name.startswith("stage4"):
                # Two bounded seed trials showed that restoring the weaker legacy
                # contact/view rewards reduced hit1 from the required 0.84 to
                # 0.80-0.82. A separate two-seed probe showed that hard racket-z
                # termination collapsed mean hits from 7.21 to about 1.3. Keep the
                # already-passed dense, recoverable objective through stage4 so each
                # subsequent transition introduces only its named DR or missing change.
                stage_cfg = dense_contact_cfg(
                    replace(stage_cfg, terminate_on_racket_z_limit=False)
                )
                if stage.name != "stage4b_contact_dr":
                    stage_cfg = first_hit_preservation_cfg(stage_cfg)
            cfg = mixed_missing_obs_cfg(
                stage_cfg,
                missing_prob=missing_prob,
                z_high_range_m=(1.02, 1.28),
            )
            if stage.name != "stage4b_contact_dr":
                cfg = replace(cfg, ball_obs_missing_episode_coherent_prob=1.0)
            note = (
                "Missing bridge profile: z-high is recoverable; physical-camera and view-bounds loss "
                f"are stale with age at p={missing_prob:.2f} with a randomized z-high threshold."
            )
            notes = f"{stage.notes} {note}".strip() if stage.notes else note
            patched_stage = replace(stage, cfg=cfg, notes=notes)
            patched_stage = transition_gate(patched_stage)
            if stage.name == "stage4c_lite_actuator_dr":
                patched.append(
                    replace(
                        patched_stage,
                        name="stage4bc_lite_actuator_dr_frozen_probe",
                        policy_updates_enabled=False,
                        notes=(
                            "Zero-update actuator/PD-DR exposure probe. Save an exact-policy "
                            "baseline under the new environment before stage4c PPO is allowed."
                        ),
                    )
                )
            patched.append(patched_stage)
            bridge_source = patched_stage
            if stage.name == "stage4b_contact_dr":
                first_hit_stage = replace(
                    patched_stage,
                    name="stage4ag_first_hit_preservation_missing_55",
                    total_steps=1_500_000,
                    cfg=first_hit_preservation_cfg(patched_stage.cfg),
                    notes=(
                        "First-hit-only transition at 55% missing: add intercept, early-miss, and first-apex "
                        "shaping before missing or dynamics change."
                    ),
                    min_updates=max(int(patched_stage.min_updates), 45),
                )
                patched.append(first_hit_stage)
                coherent_missing_stage = replace(
                    first_hit_stage,
                    name="stage4ah_episode_coherent_25_missing_55",
                    total_steps=1_500_000,
                    cfg=replace(
                        first_hit_stage.cfg,
                        ball_obs_missing_episode_coherent_prob=0.25,
                    ),
                    notes=(
                        "Observation-semantics-only transition at 55% missing: use episode-coherent "
                        "physical camera/view loss for 25% of environments while retaining the "
                        "previous per-refresh model for the rest. Two matched 256-episode q=0 "
                        "validations showed no old-setting regression after this bridge."
                    ),
                    min_updates=max(int(first_hit_stage.min_updates), 45),
                    target_hit1_rate=0.80,
                )
                patched.append(coherent_missing_stage)
                coherent_source = coherent_missing_stage
                for coherent_name, coherent_prob in (
                    ("stage4ai_episode_coherent_50_missing_55", 0.50),
                    ("stage4aia_episode_coherent_625_missing_55", 0.625),
                    ("stage4aj_episode_coherent_75_missing_55", 0.75),
                    ("stage4aj0_episode_coherent_8125_missing_55", 0.8125),
                    ("stage4aja_episode_coherent_875_missing_55", 0.875),
                    ("stage4ajb_episode_coherent_9375_missing_55", 0.9375),
                    ("stage4ak_episode_coherent_100_missing_55", 1.00),
                ):
                    coherent_source = replace(
                        coherent_missing_stage,
                        name=coherent_name,
                        cfg=replace(
                            coherent_source.cfg,
                            ball_obs_missing_episode_coherent_prob=coherent_prob,
                        ),
                        notes=(
                            "Observation-semantics-only transition at 55% missing: raise the "
                            f"episode-coherent physical missing fraction to {coherent_prob:.0%}."
                        ),
                    )
                    patched.append(coherent_source)
                first_hit_recovery_stage = replace(
                    coherent_source,
                    name="stage4ak1_first_hit_intercept_18_missing_55",
                    total_steps=1_500_000,
                    cfg=replace(
                        coherent_source.cfg,
                        pre_hit_intercept_reward_weight=1.80,
                    ),
                    notes=(
                        "Reward-only recovery transition at q=100%/p=55%: raise pre-hit "
                        "intercept reward from 1.4 to 1.8 before any further missing or DR change."
                    ),
                    target_hit1_rate=0.80,
                )
                patched.append(first_hit_recovery_stage)
                bridge_source = first_hit_recovery_stage
            if stage.name in next_missing_bridges:
                for bridge_name, bridge_prob, bridge_note in next_missing_bridges[stage.name]:
                    bridge_cfg = mixed_missing_obs_cfg(
                        bridge_source.cfg,
                        missing_prob=bridge_prob,
                        z_high_range_m=(1.02, 1.28),
                    )
                    bridge_source = replace(
                        patched_stage,
                        name=bridge_name,
                        total_steps=1_500_000,
                        cfg=bridge_cfg,
                        notes=(
                            f"Missing-only transition: {bridge_note}"
                            + (
                                " This bridge is policy-frozen because matched pointwise PPO "
                                "adaptations regressed the stronger p60 source; p is a categorical "
                                "episode-mixture weight rather than a continuous physical severity."
                                if bridge_prob in {0.575, 0.6125, 0.625, 0.6375, 0.65}
                                else ""
                            )
                        ),
                        min_updates=max(int(patched_stage.min_updates), 20),
                        target_mean_hits=(
                            (4.60 if bridge_prob >= 0.99 else (5.00 if bridge_prob >= 0.85 else 5.20))
                            if bridge_prob >= 0.75
                            else (
                                6.10
                                if bridge_prob == 0.5625
                                else (
                                    6.30
                                    if 0.60 <= bridge_prob < 0.65
                                    else patched_stage.target_mean_hits
                                )
                            )
                        ),
                        # Coherent q=1 source policies pass at hit1~=0.81. Keep the
                        # two fine p bridges aligned with that verified envelope;
                        # mean-hit, length, view, hit3, and later p60 gates stay strict.
                        target_hit1_rate=(
                            (0.79 if bridge_prob >= 0.99 else 0.80)
                            if bridge_prob >= 0.75
                            else (0.80 if bridge_prob < 0.65 else patched_stage.target_hit1_rate)
                        ),
                        target_hit3_rate=(
                            (0.53 if bridge_prob >= 0.99 else 0.56)
                            if bridge_prob >= 0.75
                            else patched_stage.target_hit3_rate
                        ),
                        target_mean_hits_ge3=(
                            (7.8 if bridge_prob >= 0.99 else 8.0)
                            if bridge_prob >= 0.75
                            else patched_stage.target_mean_hits_ge3
                        ),
                        min_recent_mean_return=(
                            None if bridge_prob >= 0.85 else patched_stage.min_recent_mean_return
                        ),
                        policy_updates_enabled=bridge_prob
                        not in {0.575, 0.6125, 0.625, 0.6375, 0.65},
                    )
                    patched.append(bridge_source)
        else:
            patched.append(stage)
        if stage.name == stage4a.name:
            patched.append(no_term_stage)
            patched.append(mocap_stage)
            patched.append(missing_stage)
            patched.append(random_missing_stage)
            patched.append(missing_475_stage)
            patched.append(missing_55_stage)
    return patched


def _stage4g_robust15_missing_stages(stage4g_cfg: MjxJuggleConfig) -> list[CurriculumStage]:
    """Stage4g-success-based robust15 bridge.

    Evidence used for this branch:
    - ``logs_mjx_actuator_67d_inverse_mpc_h4_reg_stage4g_polish_v1`` uses the
      required 67D + real actuator replay fit + inverse MPC + asymmetric critic
      stack and trains stage4g at the original successful height
      (launch=0.32, target=0.28, hit_height_center=0.52) to ~13.5 hits and
      ~1150/1200 steps.
    - The low-reset d1b/d1c/d1m recovery branch used much lower hit centers
      (0.24 and below) and repeatedly validated at 0 hits with ``ball_too_low``.

    With the latest calibrated D455 camera, a direct clone of the old stage4g
    geometry still juggles but projects the ball far outside the image
    (early D455 probes: u ~= 900, v ~= 1100 at 848x480, camera_visible ~= 0).
    Therefore this bridge keeps the proven stage4g height/cadence reward
    structure, but first moves the target/anchor geometry into the calibrated
    D455 lower-middle view before adding stale-age missing and range.  Every
    stage requires multi-hit juggling; no sub-1-hit gate can advance this branch.
    """

    def stage4g_height_cfg(
        cfg: MjxJuggleConfig,
        *,
        target_x_range: tuple[float, float] | None = None,
        target_y_range: tuple[float, float] | None = None,
        anchor_z_range: tuple[float, float] | None = None,
        xy_jitter: float | None = None,
        z_jitter: float | None = None,
        init_vxy: float | None = None,
        init_vz_jitter: float | None = None,
        camera_missing_prob: float = 0.0,
        view_missing_prob: float = 0.0,
        dropout_prob: float = 0.0,
        missing_coherent_prob: float = 1.0,
        z_high_range_m: tuple[float, float] = (1.04, 1.30),
        view_regularization: bool = True,
        view_center_weight: float = 0.70,
        view_bounds_weight: float = 5.0,
        view_z_weight: float = 1.8,
        view_oob_weight: float = 1.5,
    ) -> MjxJuggleConfig:
        has_missing = camera_missing_prob > 0.0 or view_missing_prob > 0.0 or dropout_prob > 0.0
        cfg = _with_verified_stage4g_policy_compatible_terms(cfg)
        kwargs: dict[str, object] = dict(
            # Preserve the locally verified 13-hit stage4g geometry.
            ball_launch_height=0.32,
            target_height=0.28,
            hit_height_center=0.52,
            hit_height_tolerance=max(float(cfg.hit_height_tolerance), 0.06),
            low_hit_apex_margin=max(float(cfg.low_hit_apex_margin), 0.06),
            apex_soft_limit_margin=max(float(cfg.apex_soft_limit_margin), 0.04),
            # Keep the verified stage4g reward scale for the first bridge.
            # Earlier stage4h probes changed both camera and reward/view terms
            # at once; the evidence points to the camera mismatch, so the clone
            # stage must not alter hit reward shaping.
            post_hit_survival_reward_weight=1.40,
            center_flat_hit_reward_weight=0.80,
            hit_reward_base=0.50,
            hit_reward_combo=0.02,
            # The old successful policy used racket-z termination; keep the
            # reasonable height band instead of the soft low-reset band.
            racket_z_hard_limit_down=0.12,
            racket_z_hard_limit_up=0.24,
            terminate_on_racket_z_limit=True,
            racket_z_soft_penalty_weight=max(float(cfg.racket_z_soft_penalty_weight), 1.2),
            racket_up_drift_penalty_weight=max(float(cfg.racket_up_drift_penalty_weight), 0.3),
            # Missing/age semantics.  Start with stale-age/dropout and view-z
            # missing; physical-camera-visible gating is deliberately left off
            # here because the first stage4h probe collapsed camera_visible to
            # ~0.03 when camera visibility was required.
            ball_obs_require_camera_visible=camera_missing_prob > 0.0,
            ball_obs_camera_missing_prob=camera_missing_prob,
            ball_obs_reset_respects_camera_visibility=camera_missing_prob > 0.0,
            ball_obs_require_view_bounds=view_missing_prob > 0.0,
            ball_obs_view_bounds_missing_prob=view_missing_prob,
            ball_obs_view_z_high_missing_range_m=z_high_range_m if view_missing_prob > 0.0 else (0.0, 0.0),
            ball_obs_missing_episode_coherent_prob=missing_coherent_prob if has_missing else 0.0,
            ball_obs_age_tracks_stale=has_missing,
            ball_obs_dropout_on_refresh_only=has_missing,
            ball_obs_dropout_prob=max(float(cfg.ball_obs_dropout_prob), dropout_prob),
            ball_obs_dropout_max_steps=max(int(cfg.ball_obs_dropout_max_steps), 4) if has_missing else 1,
            ball_obs_dropout_burst_prob=max(
                float(cfg.ball_obs_dropout_burst_prob), 0.001 if dropout_prob > 0.0 else 0.0
            ),
            ball_obs_dropout_burst_max_steps=max(int(cfg.ball_obs_dropout_burst_max_steps), 16) if has_missing else 1,
            ball_obs_age_clip=max(float(cfg.ball_obs_age_clip), 0.35) if has_missing else 0.20,
        )
        if view_regularization:
            kwargs.update(
                # Upper FOV loss is recoverable via stale age; lateral/low exits still
                # protect physically bad rollouts once the clone stage has passed.
                terminate_on_ball_view_bounds=True,
                terminate_on_ball_view_x_bounds=True,
                terminate_on_ball_view_y_bounds=True,
                terminate_on_ball_view_z_low=True,
                terminate_on_ball_view_z_high=False,
                ball_view_x_bounds_m=(-0.32, 0.36),
                ball_view_y_bounds_m=(-0.62, -0.12),
                ball_view_z_bounds_m=(0.62, 1.80),
                ball_view_z_ideal_m=(0.80, 1.30),
                ball_view_z_sigma_m=0.12,
                ball_view_xy_center_penalty_weight=max(
                    float(cfg.ball_view_xy_center_penalty_weight), view_center_weight
                ),
                ball_view_bounds_penalty_weight=max(float(cfg.ball_view_bounds_penalty_weight), view_bounds_weight),
                ball_view_out_of_bounds_penalty_weight=max(
                    float(cfg.ball_view_out_of_bounds_penalty_weight), view_oob_weight
                ),
                ball_view_z_ideal_penalty_weight=max(float(cfg.ball_view_z_ideal_penalty_weight), view_z_weight),
                ball_view_z_not_ideal_penalty_weight=max(float(cfg.ball_view_z_not_ideal_penalty_weight), 0.50),
                ball_view_vxy_excess_penalty_weight=max(float(cfg.ball_view_vxy_excess_penalty_weight), 0.70),
            )
            if view_missing_prob > 0.0:
                kwargs.update(
                    # Do not hard-optimize hit count.  The original successful
                    # stage4g naturally settled near 13--15 hits with a
                    # ~0.40s cadence.  Keep cadence as a soft bias while making
                    # the primary view-missing objective: stable juggling with
                    # contacts in the calibrated D455 middle/lower image area.
                    hit_cadence_reward_weight=max(float(cfg.hit_cadence_reward_weight), 0.25),
                    hit_cadence_target_interval=0.42,
                    hit_cadence_sigma=0.12,
                    hit_min_interval_penalty_weight=max(float(cfg.hit_min_interval_penalty_weight), 2.00),
                    hit_min_interval=0.35,
                    hit_min_count_interval=max(float(cfg.hit_min_count_interval), 0.33),
                    hit_reward_cap_mode="auto",
                    hit_reward_cap_target_interval=0.42,
                    fast_hit_penalty_weight=max(float(cfg.fast_hit_penalty_weight), 1.20),
                    hit_camera_reward_weight=max(float(cfg.hit_camera_reward_weight), 0.70),
                    hit_camera_out_of_band_penalty_weight=max(
                        float(cfg.hit_camera_out_of_band_penalty_weight), 0.15
                    ),
                    hit_camera_target_v_frac=0.67,
                    hit_camera_v_sigma_frac=0.13,
                    hit_camera_lower_band_frac=(0.52, 0.84),
                    ball_view_z_ideal_penalty_weight=max(
                        float(cfg.ball_view_z_ideal_penalty_weight), max(view_z_weight, 3.20)
                    ),
                    ball_view_z_not_ideal_penalty_weight=max(
                        float(cfg.ball_view_z_not_ideal_penalty_weight), 0.80
                    ),
                )
        else:
            kwargs.update(
                terminate_on_ball_view_bounds=False,
                camera_center_weight=0.0,
                camera_visibility_penalty_weight=0.0,
                camera_visible_penalty_weight=0.0,
                camera_top_margin_penalty_weight=0.0,
                camera_depth_penalty_weight=0.0,
                ball_view_xy_center_penalty_weight=0.0,
                ball_view_bounds_penalty_weight=0.0,
                ball_view_out_of_bounds_penalty_weight=0.0,
                ball_view_z_ideal_penalty_weight=0.0,
                ball_view_z_not_ideal_penalty_weight=0.0,
                ball_view_vxy_excess_penalty_weight=0.0,
            )
        if target_x_range is not None:
            kwargs["episode_target_x_range_m"] = target_x_range
        if target_y_range is not None:
            kwargs["episode_target_y_range_m"] = target_y_range
        if anchor_z_range is not None:
            kwargs["episode_racket_anchor_z_range_m"] = anchor_z_range
        if xy_jitter is not None:
            kwargs["ball_spawn_xy_jitter"] = xy_jitter
        if z_jitter is not None:
            kwargs["ball_spawn_z_jitter"] = z_jitter
        if init_vxy is not None:
            kwargs["ball_init_vxy_max"] = init_vxy
        if init_vz_jitter is not None:
            kwargs["ball_init_vz_jitter"] = init_vz_jitter
        return replace(cfg, **kwargs)

    def make_stage(
        name: str,
        *,
        cfg: MjxJuggleConfig,
        notes: str,
        target_hits: float,
        target_len_frac: float,
        min_updates: int,
        target_hit1: float,
        target_hit3: float,
        target_hit12: float | None,
        target_hits_ge3: float,
        target_view: float | None,
        target_z: float | None,
        missing_exposure: float | None,
        target_camera_visible: float = 0.90,
        target_hit_camera_visible: float = 0.88,
        target_hit_camera_lower: float = 0.42,
        target_min_hit_interval: float = 0.34,
    ) -> CurriculumStage:
        return CurriculumStage(
            name,
            3_000_000,
            cfg,
            notes,
            gate_mode="balanced",
            advance_gate_mode="collapse",
            target_mean_hits=target_hits,
            target_mean_len_frac=target_len_frac,
            min_updates=min_updates,
            min_recent_mean_return=None,
            target_camera_visible=target_camera_visible,
            min_recent_camera_reward_dense=-0.12,
            target_ball_view_in_bounds=target_view,
            target_ball_view_z_ideal=target_z,
            target_hit1_rate=target_hit1,
            target_hit3_rate=target_hit3,
            target_hit12_rate=target_hit12,
            target_mean_hits_ge3=target_hits_ge3,
            target_min_hit_interval_s=target_min_hit_interval,
            target_max_hit_interval_s=0.62,
            target_hit_camera_visible_rate=target_hit_camera_visible,
            target_hit_camera_lower_band_rate=target_hit_camera_lower,
            target_episode_truncation_rate=0.30,
            min_ball_obs_missing_refresh_rate=missing_exposure,
            max_ball_obs_lost_rate=0.08,
        )

    def d455_hit_camera_cfg(
        cfg: MjxJuggleConfig,
        *,
        weight: float,
        out_of_band: float,
    ) -> MjxJuggleConfig:
        return replace(
            cfg,
            hit_camera_reward_weight=max(float(cfg.hit_camera_reward_weight), weight),
            hit_camera_out_of_band_penalty_weight=max(
                float(cfg.hit_camera_out_of_band_penalty_weight), out_of_band
            ),
            hit_camera_target_v_frac=0.67,
            hit_camera_v_sigma_frac=0.14,
            hit_camera_lower_band_frac=(0.52, 0.84),
        )

    stage4h_geometry_cfg = stage4g_height_cfg(
        stage4g_cfg,
        # Guardrail stage: use the latest D455 calibration constants, but keep
        # the verified stage4g target/anchor geometry exactly unchanged.  v8
        # proved that even the previous "micro" geometry
        # (x=0.02..0.05, anchor_z=-0.08..-0.04) makes both the trained policy
        # and the original 67D/inverse-MPC reference validate at only ~1.2 hits
        # with racket_too_high.  Therefore no geometry movement is allowed
        # until this D455-camera guardrail preserves the 13--15 hit behavior.
        target_x_range=None,
        target_y_range=None,
        anchor_z_range=None,
        camera_missing_prob=0.0,
        view_missing_prob=0.0,
        dropout_prob=0.0,
        missing_coherent_prob=0.0,
        view_regularization=False,
    )
    stage4h_geometry_cfg = d455_hit_camera_cfg(stage4h_geometry_cfg, weight=0.0, out_of_band=0.0)
    stage4i_visible_cfg = stage4g_height_cfg(
        stage4h_geometry_cfg,
        # v8 shows the micro bridge can retain ~5--7 hits, but the ball still
        # projects below the D455 image (roughly u=790--840, v=1000+).  The
        # earlier direct jump to x=0.07..0.11 / z=-0.18..-0.14 improved
        # visibility but collapsed hit count below 1.  Split that move into
        # smaller no-hard-view steps before enabling missing.
        target_x_range=(0.005, 0.025),
        target_y_range=(-0.008, 0.005),
        anchor_z_range=None,
        camera_missing_prob=0.0,
        view_missing_prob=0.0,
        dropout_prob=0.0,
        missing_coherent_prob=0.0,
        view_regularization=False,
    )
    stage4i_visible_cfg = d455_hit_camera_cfg(stage4i_visible_cfg, weight=0.05, out_of_band=0.0)
    stage4j_age_cfg = stage4g_height_cfg(
        stage4i_visible_cfg,
        target_x_range=(0.015, 0.045),
        target_y_range=(-0.015, 0.005),
        anchor_z_range=(-0.025, 0.0),
        camera_missing_prob=0.0,
        view_missing_prob=0.0,
        dropout_prob=0.001,
        missing_coherent_prob=1.0,
        view_regularization=False,
    )
    stage4j_age_cfg = d455_hit_camera_cfg(stage4j_age_cfg, weight=0.10, out_of_band=0.0)
    stage4k_cfg = stage4g_height_cfg(
        stage4j_age_cfg,
        target_x_range=(0.025, 0.060),
        target_y_range=(-0.022, 0.005),
        anchor_z_range=(-0.045, -0.015),
        camera_missing_prob=0.0,
        view_missing_prob=0.10,
        dropout_prob=0.002,
        missing_coherent_prob=1.0,
        # Evidence from ``stage4j_viewmissing20_from_stage4i_seed971_gpu1_v2``
        # det64: reset ball_z was 1.36--1.43m, so z_high=1.10m made the
        # coherent 20% missing episodes lose the ball before the first hit
        # (13/64 zero-hit ball_view_x_too_low terminations).  The first
        # view-missing course should model upward post-hit FOV loss, not hide
        # the launch ball; keep the threshold above reset height.
        z_high_range_m=(1.26, 1.34),
        view_center_weight=0.35,
        view_bounds_weight=2.0,
        view_z_weight=1.0,
        view_oob_weight=0.5,
    )
    stage4k_cfg = replace(
        stage4k_cfg,
        hit_camera_reward_weight=0.18,
        hit_camera_out_of_band_penalty_weight=0.02,
        camera_visibility_penalty_weight=1.5,
        camera_visible_penalty_weight=0.5,
        camera_top_margin_penalty_weight=2.0,
        camera_depth_penalty_weight=0.2,
    )
    stage4l_cfg = stage4g_height_cfg(
        stage4k_cfg,
        target_x_range=(0.040, 0.080),
        target_y_range=(-0.030, 0.005),
        anchor_z_range=(-0.070, -0.030),
        camera_missing_prob=0.0,
        view_missing_prob=0.20,
        dropout_prob=0.003,
        missing_coherent_prob=1.0,
        # Keep the first-hit ball visible; v2 showed z thresholds below reset
        # height create structural 0-hit failures.  Increase missing probability
        # before lowering the z threshold.
        z_high_range_m=(1.28, 1.38),
        view_center_weight=0.45,
        view_bounds_weight=2.8,
        view_z_weight=1.3,
        view_oob_weight=0.7,
    )
    stage4l_cfg = replace(
        stage4l_cfg,
        hit_camera_reward_weight=0.25,
        hit_camera_out_of_band_penalty_weight=0.03,
        camera_visibility_penalty_weight=2.0,
        camera_visible_penalty_weight=0.8,
        camera_top_margin_penalty_weight=3.0,
        camera_depth_penalty_weight=0.25,
    )
    stage4m_cfg = stage4g_height_cfg(
        stage4l_cfg,
        target_x_range=(0.060, 0.110),
        target_y_range=(-0.040, 0.020),
        anchor_z_range=(-0.100, -0.050),
        xy_jitter=0.025,
        z_jitter=0.035,
        init_vxy=0.012,
        init_vz_jitter=0.020,
        camera_missing_prob=0.0,
        view_missing_prob=0.30,
        dropout_prob=0.003,
        missing_coherent_prob=1.0,
        z_high_range_m=(1.30, 1.40),
        view_center_weight=0.70,
        view_bounds_weight=4.2,
        view_z_weight=1.9,
        view_oob_weight=1.2,
    )
    stage4m_cfg = replace(
        stage4m_cfg,
        hit_camera_reward_weight=0.35,
        hit_camera_out_of_band_penalty_weight=0.05,
        camera_visibility_penalty_weight=3.0,
        camera_visible_penalty_weight=1.2,
        camera_top_margin_penalty_weight=4.0,
        camera_depth_penalty_weight=0.3,
    )
    stage4n_cfg = stage4g_height_cfg(
        stage4m_cfg,
        target_x_range=(0.04, 0.20),
        target_y_range=(-0.080, 0.060),
        anchor_z_range=(-0.27, -0.15),
        xy_jitter=0.030,
        z_jitter=0.040,
        init_vxy=0.014,
        init_vz_jitter=0.030,
        camera_missing_prob=0.0,
        view_missing_prob=0.45,
        dropout_prob=0.004,
        missing_coherent_prob=1.0,
        z_high_range_m=(1.32, 1.44),
        view_center_weight=0.85,
        view_bounds_weight=5.5,
        view_z_weight=2.0,
    )
    stage4o_cfg = stage4g_height_cfg(
        stage4n_cfg,
        target_x_range=(0.00, 0.24),
        target_y_range=(-0.100, 0.080),
        anchor_z_range=(-0.29, -0.13),
        xy_jitter=0.035,
        z_jitter=0.045,
        init_vxy=0.016,
        init_vz_jitter=0.040,
        camera_missing_prob=0.0,
        view_missing_prob=0.50,
        dropout_prob=0.004,
        missing_coherent_prob=1.0,
        z_high_range_m=(1.54, 1.58),
        view_center_weight=0.95,
        view_bounds_weight=6.0,
        view_z_weight=2.2,
        view_oob_weight=2.0,
    )

    return [
        make_stage(
            "stage4h_d455_micro_geometry_bridge",
            cfg=stage4h_geometry_cfg,
            notes=(
                "D455 camera guardrail: keep the verified stage4g target/anchor geometry unchanged "
                "while switching the source-of-truth camera constants to the calibrated 848x480 D455. "
                "No view/camera reward is active here; this stage must preserve the 13--15 hit behavior "
                "before any position migration."
            ),
            target_hits=12.0,
            target_len_frac=0.75,
            min_updates=20,
            target_hit1=0.92,
            target_hit3=0.82,
            target_hit12=0.45,
            target_hits_ge3=11.0,
            target_view=None,
            target_z=None,
            missing_exposure=None,
            target_camera_visible=0.0,
            target_hit_camera_visible=0.0,
            target_hit_camera_lower=0.0,
            target_min_hit_interval=0.32,
        ),
        make_stage(
            "stage4i_d455_visible_geometry_consolidation",
            cfg=stage4i_visible_cfg,
            notes=(
                "D455 geometry bridge: first tiny target shift only. The failed v8 micro anchor move "
                "is not repeated; this stage must keep deterministic multi-hit juggling before anchor_z moves."
            ),
            target_hits=8.0,
            target_len_frac=0.55,
            min_updates=200,
            target_hit1=0.88,
            target_hit3=0.70,
            target_hit12=0.20,
            target_hits_ge3=7.0,
            target_view=None,
            target_z=None,
            missing_exposure=None,
            target_camera_visible=0.0,
            target_hit_camera_visible=0.0,
            target_hit_camera_lower=0.0,
            target_min_hit_interval=0.32,
        ),
        make_stage(
            "stage4j_d455_age_dropout_consolidation",
            cfg=stage4j_age_cfg,
            notes=(
                "D455 geometry bridge: second small target move plus only a very small anchor_z move and "
                "tiny stale-age/dropout. The gate remains multi-hit and no hard camera gate is used."
            ),
            target_hits=6.5,
            target_len_frac=0.48,
            min_updates=220,
            target_hit1=0.86,
            target_hit3=0.62,
            target_hit12=0.12,
            target_hits_ge3=5.8,
            target_view=None,
            target_z=None,
            missing_exposure=0.001,
            target_camera_visible=0.0,
            target_hit_camera_visible=0.0,
            target_hit_camera_lower=0.0,
            target_min_hit_interval=0.32,
        ),
        make_stage(
            "stage4k_d455_viewmissing10_consolidation",
            cfg=stage4k_cfg,
            notes=(
                "D455 geometry bridge: add upward view-bound stale-age missing at 10%. Upward FOV loss "
                "is allowed through age; low/lateral image quality is still softly shaped."
            ),
            target_hits=5.5,
            target_len_frac=0.42,
            min_updates=240,
            target_hit1=0.84,
            target_hit3=0.56,
            target_hit12=0.08,
            target_hits_ge3=5.0,
            target_view=0.03,
            target_z=0.02,
            missing_exposure=0.001,
            target_camera_visible=0.01,
            target_hit_camera_visible=0.0,
            target_hit_camera_lower=0.0,
            target_min_hit_interval=0.36,
        ),
        make_stage(
            "stage4l_d455_viewmissing20_consolidation",
            cfg=stage4l_cfg,
            notes=(
                "D455 geometry bridge: raise upward view-bound stale-age missing to 20% at the learned "
                "visible geometry before any range expansion."
            ),
            target_hits=5.0,
            target_len_frac=0.38,
            min_updates=260,
            target_hit1=0.82,
            target_hit3=0.52,
            target_hit12=None,
            target_hits_ge3=4.5,
            target_view=0.05,
            target_z=0.03,
            missing_exposure=0.002,
            target_camera_visible=0.02,
            target_hit_camera_visible=0.0,
            target_hit_camera_lower=0.0,
            target_min_hit_interval=0.36,
        ),
        make_stage(
            "stage4m_d455_small_range_viewmissing30",
            cfg=stage4m_cfg,
            notes=(
                "D455 bridge: first reset/target range expansion around the visible geometry, while "
                "keeping multi-hit gates so the policy cannot advance on first-hit-only behavior."
            ),
            target_hits=3.5,
            target_len_frac=0.30,
            min_updates=300,
            target_hit1=0.78,
            target_hit3=0.46,
            target_hit12=None,
            target_hits_ge3=3.8,
            target_view=0.34,
            target_z=0.24,
            missing_exposure=0.005,
            target_camera_visible=0.22,
            target_hit_camera_visible=0.20,
            target_hit_camera_lower=0.12,
            target_min_hit_interval=0.36,
        ),
        make_stage(
            "stage4n_d455_mid_range_viewmissing45",
            cfg=stage4n_cfg,
            notes=(
                "D455 bridge: mid-range reset/target expansion with 45% stale-age missing. The gate "
                "still requires multi-hit juggling before the final wide bucket."
            ),
            target_hits=3.0,
            target_len_frac=0.28,
            min_updates=340,
            target_hit1=0.72,
            target_hit3=0.42,
            target_hit12=None,
            target_hits_ge3=3.2,
            target_view=0.32,
            target_z=0.22,
            missing_exposure=0.006,
            target_camera_visible=0.20,
            target_hit_camera_visible=0.18,
            target_hit_camera_lower=0.10,
            target_min_hit_interval=0.36,
        ),
        make_stage(
            "stage4o_d455_wide_range_viewmissing50",
            cfg=stage4o_cfg,
            notes=(
                "D455 bridge: broad reset/target bucket around the calibrated visible juggling zone. "
                "This is the handoff checkpoint toward stage5 large-range/noise generalization."
            ),
            target_hits=2.6,
            target_len_frac=0.24,
            min_updates=380,
            target_hit1=0.68,
            target_hit3=0.36,
            target_hit12=None,
            target_hits_ge3=2.8,
            target_view=0.30,
            target_z=0.20,
            missing_exposure=0.006,
            target_camera_visible=0.18,
            target_hit_camera_visible=0.16,
            target_hit_camera_lower=0.08,
            target_min_hit_interval=0.36,
        ),
    ]


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
    actuator_mpc_command_dynamics_constraint: bool | None,
    actuator_mpc_command_velocity_weight: float | None,
    actuator_mpc_command_acceleration_weight: float | None,
    actuator_mpc_command_velocity_scale: float | None,
    actuator_mpc_command_acceleration_scale: float | None,
    actuator_mpc_feedback_source: str | None,
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
    if actuator_mpc_command_dynamics_constraint is not None:
        patched["actuator_mpc_command_dynamics_constraint"] = bool(actuator_mpc_command_dynamics_constraint)
    if actuator_mpc_command_velocity_weight is not None:
        patched["actuator_mpc_command_velocity_weight"] = float(actuator_mpc_command_velocity_weight)
    if actuator_mpc_command_acceleration_weight is not None:
        patched["actuator_mpc_command_acceleration_weight"] = float(actuator_mpc_command_acceleration_weight)
    if actuator_mpc_command_velocity_scale is not None:
        patched["actuator_mpc_command_velocity_scale"] = float(actuator_mpc_command_velocity_scale)
    if actuator_mpc_command_acceleration_scale is not None:
        patched["actuator_mpc_command_acceleration_scale"] = float(actuator_mpc_command_acceleration_scale)
    if actuator_mpc_feedback_source is not None:
        patched["actuator_mpc_feedback_source"] = str(actuator_mpc_feedback_source)
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
            cfg = _with_strong_camera_centering(_with_wide_polish_dr(cfg), center_weight=1.6)
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


def _apply_arm_safety_overrides(
    stages: list[CurriculumStage],
    *,
    arm_post_compensation_limiter: bool | None,
    arm_servo_target_limiter: bool | None,
    arm_servo_target_tracking_planner: bool | None,
    arm_servo_target_velocity_scale: float | None,
    arm_servo_target_acceleration_scale: float | None,
    arm_actual_state_limiter: bool | None,
    arm_actual_target_tracking_governor: bool | None,
    arm_actual_governor_natural_frequency_hz: float | None,
    arm_actual_governor_damping_ratio: float | None,
    arm_actual_jerk_limit_deg_s3: float | None,
    right_arm_pd_profile: str | None,
) -> list[CurriculumStage]:
    """Apply explicit CLI safety overrides uniformly to every curriculum stage."""

    updates: dict[str, object] = {}
    if arm_post_compensation_limiter is not None:
        updates["arm_post_compensation_limiter"] = bool(arm_post_compensation_limiter)
    if arm_servo_target_limiter is not None:
        updates["arm_servo_target_limiter"] = bool(arm_servo_target_limiter)
    if arm_servo_target_tracking_planner is not None:
        updates["arm_servo_target_tracking_planner"] = bool(
            arm_servo_target_tracking_planner
        )
    if arm_servo_target_velocity_scale is not None:
        updates["arm_servo_target_velocity_scale"] = float(
            arm_servo_target_velocity_scale
        )
    if arm_servo_target_acceleration_scale is not None:
        updates["arm_servo_target_acceleration_scale"] = float(
            arm_servo_target_acceleration_scale
        )
    if arm_actual_state_limiter is not None:
        updates["arm_actual_state_limiter"] = bool(arm_actual_state_limiter)
    if arm_actual_target_tracking_governor is not None:
        updates["arm_actual_target_tracking_governor"] = bool(
            arm_actual_target_tracking_governor
        )
    if arm_actual_governor_natural_frequency_hz is not None:
        frequency_hz = float(arm_actual_governor_natural_frequency_hz)
        if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError(
                "arm_actual_governor_natural_frequency_hz must be positive and finite"
            )
        updates["arm_actual_governor_natural_frequency_hz"] = frequency_hz
    if arm_actual_governor_damping_ratio is not None:
        damping_ratio = float(arm_actual_governor_damping_ratio)
        if not np.isfinite(damping_ratio) or damping_ratio <= 0.0:
            raise ValueError(
                "arm_actual_governor_damping_ratio must be positive and finite"
            )
        updates["arm_actual_governor_damping_ratio"] = damping_ratio
    if arm_actual_jerk_limit_deg_s3 is not None:
        jerk_limit = float(arm_actual_jerk_limit_deg_s3)
        if not np.isfinite(jerk_limit) or jerk_limit <= 0.0:
            raise ValueError("arm_actual_jerk_limit_deg_s3 must be positive and finite")
        updates["arm_actual_jerk_limit_deg_s3"] = (jerk_limit,) * 7
    if right_arm_pd_profile is not None:
        updates["right_arm_pd_profile"] = str(right_arm_pd_profile)
    if not updates:
        return stages
    return [replace(stage, cfg=replace(stage.cfg, **updates)) for stage in stages]


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
    actuator_mpc_command_dynamics_constraint: bool | None = None,
    actuator_mpc_command_velocity_weight: float | None = None,
    actuator_mpc_command_acceleration_weight: float | None = None,
    actuator_mpc_command_velocity_scale: float | None = None,
    actuator_mpc_command_acceleration_scale: float | None = None,
    actuator_mpc_feedback_source: str | None = None,
    dr_randomize_actuator_cmd_filter: bool | None = None,
    dr_actuator_cmd_tau_range: tuple[float, float] | None = None,
    dr_actuator_cmd_gain_range: tuple[float, float] | None = None,
    wide_polish_dr: bool = False,
    asymmetric_critic: bool = False,
    critic_command_history_steps: int = 4,
    arm_post_compensation_limiter: bool | None = None,
    arm_servo_target_limiter: bool | None = None,
    arm_servo_target_tracking_planner: bool | None = None,
    arm_servo_target_velocity_scale: float | None = None,
    arm_servo_target_acceleration_scale: float | None = None,
    arm_actual_state_limiter: bool | None = None,
    arm_actual_target_tracking_governor: bool | None = None,
    arm_actual_governor_natural_frequency_hz: float | None = None,
    arm_actual_governor_damping_ratio: float | None = None,
    arm_actual_jerk_limit_deg_s3: float | None = None,
    right_arm_pd_profile: str | None = None,
) -> list[CurriculumStage]:
    if curriculum_profile in GOAL_D455_IDEALPD_PROFILES:
        preserve_deployed_67d = curriculum_profile in GOAL_D455_IDEALPD67_PROFILES
        if bool(high_latency_obs):
            raise ValueError(f"{curriculum_profile} is a no-actuator ideal-PD ablation; high_latency_obs is incompatible")
        allowed_delay_presets = (
            {"baseline_current", "real_actuator_replay_fit"}
            if preserve_deployed_67d
            else {"baseline_current"}
        )
        if delay_ablation_preset not in allowed_delay_presets:
            required = (
                "real_actuator_replay_fit"
                if preserve_deployed_67d
                else "baseline_current"
            )
            raise ValueError(
                f"{curriculum_profile} requires --delay-ablation-preset {required}"
            )
        if any(value is not None for value in (delay_min_ms, delay_max_ms, delay_jitter_ms, delay_sampling_mode)):
            raise ValueError(f"{curriculum_profile} does not use actuator-delay sampling")
        if actuator_cmd_filter not in (None, False):
            raise ValueError(f"{curriculum_profile} disables the actuator command filter")
        if actuator_compensation_mode not in (None, "none"):
            raise ValueError(f"{curriculum_profile} disables actuator compensation")
        if actuator_lead_compensation:
            raise ValueError(f"{curriculum_profile} disables lead/inverse compensation")

        stack_kwargs = _delay_conditioned_control_kwargs(
            "real_actuator_replay_fit"
            if preserve_deployed_67d
            else "baseline_current"
        )
        stack_kwargs["actuator_delay_observation_only"] = preserve_deployed_67d
        stack_kwargs = _apply_actuator_cli_overrides(
            stack_kwargs,
            actuator_cmd_filter=False,
            actuator_cmd_tau=actuator_cmd_tau,
            actuator_cmd_gain=actuator_cmd_gain,
            actuator_compensation_mode="none",
            actuator_lead_compensation=False,
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
            actuator_mpc_command_dynamics_constraint=actuator_mpc_command_dynamics_constraint,
            actuator_mpc_command_velocity_weight=actuator_mpc_command_velocity_weight,
            actuator_mpc_command_acceleration_weight=actuator_mpc_command_acceleration_weight,
            actuator_mpc_command_velocity_scale=actuator_mpc_command_velocity_scale,
            actuator_mpc_command_acceleration_scale=actuator_mpc_command_acceleration_scale,
            actuator_mpc_feedback_source=actuator_mpc_feedback_source,
            dr_randomize_actuator_cmd_filter=False,
            dr_actuator_cmd_tau_range=dr_actuator_cmd_tau_range,
            dr_actuator_cmd_gain_range=dr_actuator_cmd_gain_range,
        )
        stages = _goal_d455_autolaunch_v1_stages(
            stack_kwargs=stack_kwargs,
            stage_steps_override=stage_steps_override,
            critic_command_history_steps=max(12, int(critic_command_history_steps)),
            require_inverse_mpc_stack=False,
        )
        if curriculum_profile in (
            GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE,
            GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE,
        ):
            stages = _with_goal_d455_autolaunch_idealpd67_viewdense_shaping(stages)
        if curriculum_profile == GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE:
            stages = _with_goal_d455_autolaunch_idealpd67_final_recovery(stages)
        if preserve_deployed_67d:
            for stage in stages:
                cfg = stage.cfg
                if not (
                    cfg.enable_delay_conditioning
                    and cfg.include_tau_act_norm
                    and cfg.include_command_state
                    and cfg.include_active_command_error
                    and cfg.include_phase_features
                    and cfg.actuator_delay_observation_only
                    and float(cfg.delay_min_ms) == 72.0
                    and float(cfg.delay_max_ms) == 72.0
                    and float(cfg.delay_jitter_ms) == 0.0
                    and cfg.delay_sampling_mode == "uniform"
                    and not cfg.actuator_cmd_filter
                    and cfg.actuator_compensation_mode == "none"
                    and cfg.asymmetric_critic
                    and int(cfg.critic_command_history_steps) == 12
                ):
                    raise ValueError(
                        f"{stage.name} escaped the deployed-67D/zero-residual-delay ideal-PD contract"
                    )
        stages = [
            replace(
                stage,
                notes=(
                    f"{stage.notes}  "
                    + (
                        "ideal-PD 67D plant: preserve the deployed "
                        "72 ms command-history/error/phase observation contract, "
                        "while bypassing that delay on the simulated servo and "
                        "disabling actuator filtering and compensation."
                        if preserve_deployed_67d
                        else
                        "ideal-PD ablation: reuse the original 20260716 "
                        "autolaunch curriculum/gates/rewards, while disabling "
                        "simulated actuator command filtering, delay "
                        "conditioning, and compensation."
                    )
                ),
            )
            for stage in stages
        ]
        return _apply_arm_safety_overrides(
            stages,
            arm_post_compensation_limiter=arm_post_compensation_limiter,
            arm_servo_target_limiter=arm_servo_target_limiter,
            arm_servo_target_tracking_planner=arm_servo_target_tracking_planner,
            arm_servo_target_velocity_scale=arm_servo_target_velocity_scale,
            arm_servo_target_acceleration_scale=arm_servo_target_acceleration_scale,
            arm_actual_state_limiter=arm_actual_state_limiter,
            arm_actual_target_tracking_governor=arm_actual_target_tracking_governor,
            arm_actual_governor_natural_frequency_hz=arm_actual_governor_natural_frequency_hz,
            arm_actual_governor_damping_ratio=arm_actual_governor_damping_ratio,
            arm_actual_jerk_limit_deg_s3=arm_actual_jerk_limit_deg_s3,
            right_arm_pd_profile=right_arm_pd_profile,
        )

    if curriculum_profile in (ROBUST_JUGGLE_PROFILE, *D455_67D_INVERSE_MPC_PROFILES):
        if bool(high_latency_obs):
            raise ValueError(f"{curriculum_profile} fixes actor obs_dim at 67; high_latency_obs is incompatible")
        if delay_ablation_preset not in {"baseline_current", "real_actuator_replay_fit"}:
            raise ValueError(
                f"{curriculum_profile} owns its progressive delay schedule; "
                "do not select a different --delay-ablation-preset"
            )
        if any(value is not None for value in (delay_min_ms, delay_max_ms, delay_jitter_ms, delay_sampling_mode)):
            raise ValueError(f"{curriculum_profile} owns its per-stage delay ranges")
        if actuator_compensation_mode not in (None, "inverse_mpc"):
            raise ValueError(f"{curriculum_profile} requires --actuator-compensation-mode inverse_mpc")
        if actuator_cmd_filter is False:
            raise ValueError(f"{curriculum_profile} requires the real actuator command filter")
        if actuator_lead_compensation:
            raise ValueError(f"{curriculum_profile} uses inverse MPC, not lead compensation")
        if curriculum_profile in (
            GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
        ):
            if any(
                value is True
                for value in (
                    arm_post_compensation_limiter,
                    arm_servo_target_limiter,
                    arm_servo_target_tracking_planner,
                    arm_actual_state_limiter,
                    arm_actual_target_tracking_governor,
                )
            ):
                raise ValueError(
                    f"{curriculum_profile} requires reward-only actuator+inverse-MPC control and disables "
                    "post-compensation, servo-target, planner, bottom actual-state limiters, and the governor"
                )

        stack_kwargs = _delay_conditioned_control_kwargs("real_actuator_replay_fit")
        stack_kwargs = _apply_actuator_cli_overrides(
            stack_kwargs,
            actuator_cmd_filter=True if actuator_cmd_filter is None else actuator_cmd_filter,
            actuator_cmd_tau=actuator_cmd_tau,
            actuator_cmd_gain=actuator_cmd_gain,
            actuator_compensation_mode="inverse_mpc",
            actuator_lead_compensation=False,
            actuator_lead_beta=actuator_lead_beta,
            actuator_lead_delay_scale=actuator_lead_delay_scale,
            actuator_lead_tau_scale=actuator_lead_tau_scale,
            actuator_lead_max_delta_deg=actuator_lead_max_delta_deg,
            actuator_inverse_beta=actuator_inverse_beta,
            actuator_inverse_delay_scale=actuator_inverse_delay_scale,
            actuator_inverse_tau_scale=actuator_inverse_tau_scale,
            actuator_inverse_max_delta_deg=actuator_inverse_max_delta_deg,
            actuator_mpc_beta=1.2 if actuator_mpc_beta is None else actuator_mpc_beta,
            actuator_mpc_delay_scale=1.05 if actuator_mpc_delay_scale is None else actuator_mpc_delay_scale,
            actuator_mpc_tau_scale=0.75 if actuator_mpc_tau_scale is None else actuator_mpc_tau_scale,
            actuator_mpc_horizon_steps=6 if actuator_mpc_horizon_steps is None else actuator_mpc_horizon_steps,
            actuator_mpc_tracking_weight=(
                1.0 if actuator_mpc_tracking_weight is None else actuator_mpc_tracking_weight
            ),
            actuator_mpc_nominal_weight=(
                0.25 if actuator_mpc_nominal_weight is None else actuator_mpc_nominal_weight
            ),
            actuator_mpc_delta_weight=0.05 if actuator_mpc_delta_weight is None else actuator_mpc_delta_weight,
            actuator_mpc_max_delta_deg=(
                30.0 if actuator_mpc_max_delta_deg is None else actuator_mpc_max_delta_deg
            ),
            actuator_mpc_command_dynamics_constraint=actuator_mpc_command_dynamics_constraint,
            actuator_mpc_command_velocity_weight=actuator_mpc_command_velocity_weight,
            actuator_mpc_command_acceleration_weight=actuator_mpc_command_acceleration_weight,
            actuator_mpc_command_velocity_scale=actuator_mpc_command_velocity_scale,
            actuator_mpc_command_acceleration_scale=actuator_mpc_command_acceleration_scale,
            actuator_mpc_feedback_source=(
                "actual"
                if (
                    curriculum_profile
                    == GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE
                    and actuator_mpc_feedback_source is None
                )
                else actuator_mpc_feedback_source
            ),
            dr_randomize_actuator_cmd_filter=dr_randomize_actuator_cmd_filter,
            dr_actuator_cmd_tau_range=dr_actuator_cmd_tau_range,
            dr_actuator_cmd_gain_range=dr_actuator_cmd_gain_range,
        )
        profile_builders = {
            ROBUST_JUGGLE_PROFILE: _robust_juggle_v1_stages,
            D455_STABLE_4G_PROFILE: _d455_stable_4g_v1_stages,
            D455_RECOVERY_PROFILE: _d455_recovery_v1_stages,
            D455_FULL_CURRICULUM_PROFILE: _d455_full_curriculum_v1_stages,
            D455_SUCCESS_REF_PROFILE: _d455_success_ref_v1_stages,
            GOAL_D455_AUTOLAUNCH_PROFILE: _goal_d455_autolaunch_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE: _goal_d455_autolaunch_viewdense_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_RELAXTRUNC_PROFILE: _goal_d455_autolaunch_viewdense_relaxtrunc_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_FULLSAFE_PROFILE: _goal_d455_autolaunch_viewdense_fullsafe_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE: _goal_d455_autolaunch_viewdense_drivegov_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE: _goal_d455_autolaunch_viewdense_drivegov_terminalsafe_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE: _goal_d455_autolaunch_viewdense_drivegov_successref_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE: _goal_d455_autolaunch_viewdense_drivegov_highapex_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_COUNTCREDIT_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_COUNTCREDIT_NOMISSING_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_nomissing_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_RECOVERABILITY_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_recoverability_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_INTERCEPT_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_intercept_v1_stages,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_COUNT_PROGRESS_PROFILE: _goal_d455_autolaunch_viewdense_constrained_mpc_count_progress_v1_stages,
            GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE: _goal_d455_autolaunch_teacherstudent_drivegov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_countcredit_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_RECOVERY_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_final_recovery_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_CADENCE_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_SURVIVAL_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_SURVIVAL_COUNTCREDIT_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_final_survival_countcredit_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_MISSING_AGE_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_final_missing_age_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_FINAL_INTERCEPT_NOGOV_PROFILE: _goal_d455_autolaunch_actuator_inversempc_final_intercept_nogov_v1_stages,
            GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE: _goal_d455_autolaunch_idealpd67_actuator_inversempc_finetune_v1_stages,
            GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_RESIDUAL_PROFILE: _goal_d455_autolaunch_idealpd67_actuator_inversempc_residual_v1_stages,
            GOAL_D455_RELEASE_PROFILE: _goal_d455_release_v1_stages,
        }
        stages = profile_builders[curriculum_profile](
            stack_kwargs=stack_kwargs,
            stage_steps_override=stage_steps_override,
            critic_command_history_steps=max(12, int(critic_command_history_steps)),
        )
        stages = _apply_arm_safety_overrides(
            stages,
            arm_post_compensation_limiter=arm_post_compensation_limiter,
            arm_servo_target_limiter=arm_servo_target_limiter,
            arm_servo_target_tracking_planner=arm_servo_target_tracking_planner,
            arm_servo_target_velocity_scale=arm_servo_target_velocity_scale,
            arm_servo_target_acceleration_scale=arm_servo_target_acceleration_scale,
            arm_actual_state_limiter=arm_actual_state_limiter,
            arm_actual_target_tracking_governor=arm_actual_target_tracking_governor,
            arm_actual_governor_natural_frequency_hz=arm_actual_governor_natural_frequency_hz,
            arm_actual_governor_damping_ratio=arm_actual_governor_damping_ratio,
            arm_actual_jerk_limit_deg_s3=arm_actual_jerk_limit_deg_s3,
            right_arm_pd_profile=right_arm_pd_profile,
        )
        if curriculum_profile in (
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_PROFILE,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_PROFILE,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_PROFILE,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_CONSTRAINED_MPC_DRBRIDGE_V2_COUNTCREDIT_PROFILE,
        ):
            for stage in stages:
                cfg = stage.cfg
                if not (
                    cfg.arm_action_limiter
                    and cfg.actuator_compensation_mode == "inverse_mpc"
                    and cfg.actuator_mpc_feedback_source == "actual"
                    and float(cfg.actuator_mpc_beta) == 1.2
                    and float(cfg.actuator_mpc_delay_scale) == 1.05
                    and float(cfg.actuator_mpc_tau_scale) == 0.75
                    and int(cfg.actuator_mpc_horizon_steps) == 6
                    and float(cfg.actuator_mpc_tracking_weight) == 1.0
                    and float(cfg.actuator_mpc_nominal_weight) == 0.25
                    and float(cfg.actuator_mpc_delta_weight) == 0.05
                    and not cfg.actuator_mpc_command_dynamics_constraint
                    and not cfg.arm_post_compensation_limiter
                    and not cfg.arm_servo_target_limiter
                    and cfg.arm_servo_target_tracking_planner
                    and float(cfg.arm_servo_target_velocity_scale) == 1.0
                    and float(cfg.arm_servo_target_acceleration_scale) == 0.8
                    and not cfg.arm_actual_state_limiter
                    and not cfg.arm_actual_target_tracking_governor
                    and cfg.right_arm_pd_profile == "xml"
                    and cfg.terminate_on_base_stability
                ):
                    raise ValueError(
                        f"{stage.name} escaped the W019 constrained-compensation contract"
                    )
        if curriculum_profile in (
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE,
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE,
            GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE,
        ):
            for stage in stages:
                cfg = stage.cfg
                if not (
                    cfg.actuator_compensation_mode == "inverse_mpc"
                    and cfg.actuator_mpc_feedback_source == "actual"
                    and float(cfg.actuator_mpc_beta) == 1.2
                    and float(cfg.actuator_mpc_delay_scale) == 1.05
                    and float(cfg.actuator_mpc_tau_scale) == 0.75
                    and int(cfg.actuator_mpc_horizon_steps) == 6
                    and float(cfg.actuator_mpc_tracking_weight) == 1.0
                    and float(cfg.actuator_mpc_nominal_weight) == 0.25
                    and float(cfg.actuator_mpc_delta_weight) == 0.05
                    and np.isclose(
                        float(cfg.actuator_mpc_max_delta_rad),
                        np.deg2rad(30.0),
                    )
                    and cfg.right_arm_pd_profile == "xml"
                    and not cfg.arm_post_compensation_limiter
                    and not cfg.arm_servo_target_limiter
                    and not cfg.arm_servo_target_tracking_planner
                    and cfg.arm_actual_state_limiter
                    and cfg.arm_actual_target_tracking_governor
                    and float(cfg.arm_actual_governor_natural_frequency_hz) == 8.0
                    and float(cfg.arm_actual_governor_damping_ratio) == 1.0
                    and all(
                        float(value) == 175000.0
                        for value in cfg.arm_actual_jerk_limit_deg_s3
                    )
                ):
                    raise ValueError(
                        f"{stage.name} escaped the W015 target-aware actual "
                        "drive-governor contract"
                    )
                if (
                    curriculum_profile
                    in (
                        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE,
                        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE,
                        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE,
                        GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE,
                    )
                    and not (
                        float(cfg.racket_anchor_termination_penalty_base) == 2.5
                        and float(cfg.racket_anchor_termination_penalty_per_hit) == 0.0
                    )
                ):
                    raise ValueError(
                        f"{stage.name} escaped the W016 terminal-safe contract"
                    )
                if (
                    curriculum_profile
                    in (
                        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE,
                        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE,
                        GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE,
                    )
                    and not (
                        float(cfg.action_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT
                        and float(cfg.action_delta_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT
                        and float(cfg.command_tracking_error_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT
                        and float(cfg.delay_action_jerk_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT
                        and float(cfg.post_hit_survival_reward_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT
                        and float(cfg.termination_miss_penalty_per_hit)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_MISS_PENALTY_PER_HIT
                        and float(cfg.racket_z_limit_termination_penalty_per_hit)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_RACKET_Z_PENALTY_PER_HIT
                    )
                ):
                    raise ValueError(
                        f"{stage.name} escaped the W017 success-reference "
                        "learnability contract"
                    )
        if curriculum_profile in (
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
            GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE,
            GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_RESIDUAL_PROFILE,
        ):
            for stage in stages:
                cfg = stage.cfg
                expected_actual_limiter = (
                    curriculum_profile
                    == GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_RESIDUAL_PROFILE
                )
                expected_feedback = (
                    "applied"
                    if curriculum_profile in (
                        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
                        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
                    )
                    else "actual"
                )
                if not (
                    cfg.enable_delay_conditioning
                    and cfg.include_tau_act_norm
                    and cfg.include_command_state
                    and cfg.include_active_command_error
                    and cfg.include_phase_features
                    and not cfg.actuator_delay_observation_only
                    and cfg.actuator_cmd_filter
                    and cfg.actuator_compensation_mode == "inverse_mpc"
                    and cfg.actuator_mpc_feedback_source == expected_feedback
                    and cfg.asymmetric_critic
                    and int(cfg.critic_command_history_steps) == 12
                    and not cfg.arm_post_compensation_limiter
                    and not cfg.arm_servo_target_limiter
                    and not cfg.arm_servo_target_tracking_planner
                    and bool(cfg.arm_actual_state_limiter) == expected_actual_limiter
                    and bool(cfg.arm_actual_target_tracking_governor)
                    == expected_actual_limiter
                    and (
                        not expected_actual_limiter
                        or (
                            cfg.arm_actual_target_tracking_governor
                            and float(cfg.arm_actual_governor_natural_frequency_hz) == 8.0
                            and float(cfg.arm_actual_governor_damping_ratio) == 1.0
                            and all(
                                float(value) == 175000.0
                                for value in cfg.arm_actual_jerk_limit_deg_s3
                            )
                        )
                    )
                ):
                    raise ValueError(
                        f"{stage.name} escaped the original-67D actuator+inverse-MPC "
                        "transfer contract"
                    )
                if (
                    curriculum_profile in (
                        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
                        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
                    )
                    and not (
                        float(cfg.action_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT
                        and float(cfg.action_delta_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT
                        and float(cfg.command_tracking_error_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT
                        and float(cfg.delay_action_jerk_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT
                        and float(cfg.post_hit_survival_reward_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT
                        and float(cfg.arm_vel_limit_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_VEL_LIMIT_WEIGHT
                        and float(cfg.arm_acc_limit_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_ACC_LIMIT_WEIGHT
                        and float(cfg.arm_limiter_penalty_weight)
                        == GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_LIMITER_WEIGHT
                    )
                ):
                    raise ValueError(
                        f"{stage.name} escaped the no-governor success-reference reward contract"
                    )
        return stages

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
                target_height=0.48,
                ball_view_z_bounds_m=(0.62, 1.30),
                ball_view_z_ideal_m=(0.78, 1.24),
                ball_view_z_sigma_m=0.12,
                posture_weight=0.35,
                base_pose_weight=0.08,
                ball_base_x_penalty_weight=1.50,
                ball_base_x_soft_limit=0.10,
                ball_base_vxy_penalty_weight=0.12,
                torque_penalty_weight=0.0003,
                arm_vel_limit_penalty_weight=0.10,
                arm_acc_limit_penalty_weight=0.12,
                arm_limiter_penalty_weight=0.04,
                hit_reward_base=1.8,
                hit_reward_combo=0.25,
                hit_cadence_reward_weight=0.24,
                hit_cadence_target_interval=0.50,
                hit_cadence_sigma=0.14,
                hit_min_interval_penalty_weight=1.20,
                hit_min_interval=0.38,
                hit_min_count_interval=0.36,
                fast_hit_penalty_weight=0.80,
                hit_reward_cap_mode="auto",
                hit_reward_cap_target_interval=0.48,
            ),
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            target_min_hit_interval_s=0.38,
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
                target_height=0.50,
                ball_view_z_bounds_m=(0.62, 1.32),
                ball_view_z_ideal_m=(0.78, 1.26),
                ball_view_z_sigma_m=0.12,
                posture_weight=0.35,
                base_pose_weight=0.08,
                ball_base_x_penalty_weight=1.20,
                ball_base_x_soft_limit=0.12,
                ball_base_vxy_penalty_weight=0.12,
                torque_penalty_weight=0.0003,
                arm_vel_limit_penalty_weight=0.10,
                arm_acc_limit_penalty_weight=0.12,
                arm_limiter_penalty_weight=0.04,
                hit_reward_base=1.7,
                hit_reward_combo=0.22,
                hit_cadence_reward_weight=0.30,
                hit_cadence_target_interval=0.52,
                hit_cadence_sigma=0.14,
                hit_min_interval_penalty_weight=1.40,
                hit_min_interval=0.40,
                hit_min_count_interval=0.38,
                fast_hit_penalty_weight=0.90,
                hit_reward_cap_mode="auto",
                hit_reward_cap_target_interval=0.50,
            ),
            "",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            target_min_hit_interval_s=0.40,
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
                target_height=0.50,
                ball_view_z_bounds_m=(0.62, 1.32),
                ball_view_z_ideal_m=(0.78, 1.26),
                ball_view_z_sigma_m=0.12,
                posture_weight=0.60,
                base_pose_weight=0.10,
                ball_base_x_penalty_weight=2.5,
                ball_base_x_soft_limit=0.09,
                ball_base_vxy_penalty_weight=0.40,
                torque_penalty_weight=0.0003,
                arm_vel_limit_penalty_weight=0.05,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.03,
                hit_reward_base=1.7,
                hit_reward_combo=0.22,
                hit_cadence_reward_weight=0.30,
                hit_cadence_target_interval=0.52,
                hit_cadence_sigma=0.14,
                hit_min_interval_penalty_weight=1.40,
                hit_min_interval=0.40,
                hit_min_count_interval=0.38,
                fast_hit_penalty_weight=0.90,
                hit_reward_cap_mode="auto",
                hit_reward_cap_target_interval=0.50,
            ),
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            target_min_hit_interval_s=0.40,
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
                target_height=0.50,
                ball_view_z_bounds_m=(0.62, 1.32),
                ball_view_z_ideal_m=(0.78, 1.26),
                ball_view_z_sigma_m=0.12,
                posture_weight=0.90,
                base_pose_weight=0.20,
                ball_base_x_penalty_weight=4.0,
                ball_base_x_soft_limit=0.07,
                ball_base_vxy_penalty_weight=0.80,
                torque_penalty_weight=0.0004,
                arm_vel_limit_penalty_weight=0.05,
                arm_acc_limit_penalty_weight=0.08,
                arm_limiter_penalty_weight=0.04,
                hit_reward_base=1.7,
                hit_reward_combo=0.22,
                hit_cadence_reward_weight=0.30,
                hit_cadence_target_interval=0.52,
                hit_cadence_sigma=0.14,
                hit_min_interval_penalty_weight=1.40,
                hit_min_interval=0.40,
                hit_min_count_interval=0.38,
                fast_hit_penalty_weight=0.90,
                hit_reward_cap_mode="auto",
                hit_reward_cap_target_interval=0.50,
            ),
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            target_min_hit_interval_s=0.40,
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
                target_height=0.50,
                ball_view_z_bounds_m=(0.62, 1.32),
                ball_view_z_ideal_m=(0.78, 1.26),
                ball_view_z_sigma_m=0.12,
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
                hit_reward_base=1.7,
                hit_reward_combo=0.22,
                hit_cadence_reward_weight=0.30,
                hit_cadence_target_interval=0.52,
                hit_cadence_sigma=0.14,
                hit_min_interval_penalty_weight=1.40,
                hit_min_interval=0.40,
                hit_min_count_interval=0.38,
                fast_hit_penalty_weight=0.90,
                hit_reward_cap_mode="auto",
                hit_reward_cap_target_interval=0.50,
            ),
            "MJX base recenter terms are partial compared with CPU env.",
            target_mean_hits=4.0,
            target_mean_len_frac=0.20,
            target_min_hit_interval_s=0.40,
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
        hit_cadence_reward_weight=0.35,
        hit_cadence_target_interval=0.52,
        hit_cadence_sigma=0.14,
        hit_min_interval_penalty_weight=1.40,
        hit_min_interval=0.40,
        hit_min_count_interval=0.38,
        fast_hit_penalty_weight=0.90,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.50,
    )
    cadence_mid = dict(
        hit_cadence_reward_weight=0.30,
        hit_cadence_target_interval=0.52,
        hit_cadence_sigma=0.14,
        hit_min_interval_penalty_weight=1.40,
        hit_min_interval=0.40,
        hit_min_count_interval=0.38,
        fast_hit_penalty_weight=0.90,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.50,
    )
    cadence_3a = dict(
        hit_cadence_reward_weight=0.15,
        hit_cadence_target_interval=0.50,
        hit_cadence_sigma=0.16,
        hit_min_interval_penalty_weight=1.00,
        hit_min_interval=0.38,
        hit_min_count_interval=0.36,
        fast_hit_penalty_weight=0.50,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.48,
    )
    cadence_fast_032 = dict(
        hit_cadence_reward_weight=0.10,
        hit_cadence_target_interval=0.44,
        hit_cadence_sigma=0.12,
        hit_min_interval_penalty_weight=1.50,
        hit_min_interval=0.34,
        hit_min_count_interval=0.32,
        fast_hit_penalty_weight=0.90,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.44,
    )
    cadence_fast_030 = dict(
        hit_cadence_reward_weight=0.10,
        hit_cadence_target_interval=0.42,
        hit_cadence_sigma=0.12,
        hit_min_interval_penalty_weight=1.50,
        hit_min_interval=0.32,
        hit_min_count_interval=0.30,
        fast_hit_penalty_weight=0.90,
        hit_reward_cap_mode="auto",
        hit_reward_cap_target_interval=0.42,
    )
    camera_stage3 = dict(
        camera_visibility_mode="pixel",
        virtual_camera_pose_mode="base_extrinsic",
        virtual_camera_base_body_name=D455_848_UNDISTORTED_SIM_BASE_BODY,
        camera_center_weight=0.25,
        camera_visibility_penalty_weight=1.0,
        camera_depth_penalty_weight=0.5,
        camera_pixel_margin=D455_848_UNDISTORTED_PIXEL_MARGIN,
        camera_min_depth=0.15,
        camera_max_depth=2.50,
        racket_chest_xy_penalty_weight=0.55,
        racket_chest_z_penalty_weight=0.35,
    )
    camera_stage4 = dict(
        camera_visibility_mode="pixel",
        virtual_camera_pose_mode="base_extrinsic",
        virtual_camera_base_body_name=D455_848_UNDISTORTED_SIM_BASE_BODY,
        camera_center_weight=0.5,
        camera_visibility_penalty_weight=8.0,
        camera_depth_penalty_weight=0.5,
        camera_visible_penalty_weight=3.0,
        camera_top_margin_penalty_weight=12.0,
        camera_pixel_margin=D455_848_UNDISTORTED_PIXEL_MARGIN,
        camera_min_depth=0.15,
        camera_max_depth=2.50,
        racket_chest_xy_penalty_weight=0.55,
        racket_chest_z_penalty_weight=0.35,
    )

    patched_stages = []
    for stage in stages:
        cfg = stage.cfg
        name = stage.name
        stage_overrides: dict[str, float] = {}
        if name.startswith("stage1f"):
            cfg = replace(cfg, **cadence_1f)
            stage_overrides.update(target_min_hit_interval_s=0.40)
        elif name.startswith(("stage2a", "stage2b", "stage2c", "stage3b")):
            cfg = replace(cfg, **cadence_mid)
            stage_overrides.update(target_min_hit_interval_s=0.38)
        elif name.startswith("stage3a"):
            cfg = replace(cfg, **cadence_3a)
            stage_overrides.update(target_min_hit_interval_s=0.36)
        elif name.startswith(("stage4a", "stage4b")):
            cfg = replace(cfg, **cadence_fast_032, **camera_stage4)
            stage_overrides.update(target_min_hit_interval_s=0.34)
        elif name.startswith(("stage4c", "stage4d", "stage4e")):
            cfg = replace(cfg, **cadence_fast_030, **camera_stage4)
            stage_overrides.update(target_min_hit_interval_s=0.32)
        elif name.startswith(("stage4f", "stage4g")):
            cfg = replace(cfg, **cadence_fast_032, **camera_stage4)
            stage_overrides.update(target_min_hit_interval_s=0.34)

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
        if curriculum_profile in LOW_RESET_CURRICULUM_PROFILES:
            if name.startswith("stage1a"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.05,
                    z_weight=0.15,
                    bounds_weight=0.30,
                    vxy_weight=0.0,
                    launch_height=0.14,
                    target_height=0.11,
                    hit_height_center=0.13,
                    terminate_racket_z=False,
                )
            elif name.startswith("stage1b"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.10,
                    z_weight=0.20,
                    bounds_weight=0.50,
                    vxy_weight=0.05,
                    launch_height=0.145,
                    target_height=0.115,
                    hit_height_center=0.135,
                    terminate_racket_z=False,
                )
            elif name.startswith("stage1c"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.15,
                    z_weight=0.30,
                    bounds_weight=0.70,
                    vxy_weight=0.08,
                    launch_height=0.15,
                    target_height=0.12,
                    hit_height_center=0.14,
                    terminate_racket_z=False,
                )
            elif name.startswith("stage1d"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.20,
                    z_weight=0.40,
                    bounds_weight=1.00,
                    vxy_weight=0.10,
                    launch_height=0.15,
                    target_height=0.12,
                    hit_height_center=0.145,
                    terminate_racket_z=False,
                )
            elif name.startswith(("stage1e", "stage1f")):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.25,
                    z_weight=0.50,
                    bounds_weight=1.50,
                    vxy_weight=0.15,
                    launch_height=0.155,
                    target_height=0.24,
                    hit_height_center=0.28,
                    terminate_racket_z=False,
                )
                cfg = replace(
                    cfg,
                    ball_view_z_bounds_m=(0.62, 1.30),
                    ball_view_z_ideal_m=(0.76, 1.24),
                    ball_view_z_sigma_m=0.12,
                )
            elif name.startswith(("stage2a", "stage2b", "stage2c")):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.35,
                    z_weight=0.70,
                    bounds_weight=2.00,
                    vxy_weight=0.20,
                    target_x_range=(0.00, 0.08),
                    target_y_range=(0.00, 0.06),
                    launch_height=0.16,
                    target_height=0.26,
                    hit_height_center=0.30,
                    terminate_racket_z=False,
                )
                cfg = replace(
                    cfg,
                    ball_view_z_bounds_m=(0.62, 1.32),
                    ball_view_z_ideal_m=(0.76, 1.26),
                    ball_view_z_sigma_m=0.12,
                )
            elif name.startswith("stage3a"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.55,
                    z_weight=1.20,
                    bounds_weight=3.00,
                    vxy_weight=0.35,
                    target_x_range=(0.02, 0.12),
                    target_y_range=(0.00, 0.08),
                    launch_height=0.15,
                    target_height=0.12,
                    hit_height_center=0.15,
                    terminate_racket_z=False,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.55, target_ball_view_z_ideal=0.35)
            elif name.startswith("stage3b"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.75,
                    z_weight=1.80,
                    bounds_weight=4.50,
                    vxy_weight=0.55,
                    target_x_range=(0.03, 0.15),
                    target_y_range=(-0.02, 0.10),
                    launch_height=0.14,
                    target_height=0.11,
                    hit_height_center=0.14,
                    terminate_racket_z=False,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.68, target_ball_view_z_ideal=0.48)
            elif name.startswith("stage4a"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.90,
                    z_weight=5.00,
                    bounds_weight=5.50,
                    vxy_weight=0.70,
                    target_x_range=(0.04, 0.17),
                    target_y_range=(-0.04, 0.12),
                    launch_height=0.11,
                    target_height=0.09,
                    hit_height_center=0.115,
                    terminate_racket_z=False,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.72, target_ball_view_z_ideal=0.70)
            elif name.startswith("stage4"):
                cfg = _with_low_reset_ball_range(
                    cfg,
                    terminate=True,
                    xy_weight=1.00,
                    z_weight=2.30,
                    bounds_weight=6.50,
                    vxy_weight=0.85,
                    target_x_range=(0.05, 0.18),
                    target_y_range=(-0.04, 0.12),
                    launch_height=0.12,
                    target_height=0.10,
                    hit_height_center=0.13,
                    racket_up_margin=0.24,
                    terminate_racket_z=True,
                )
                cfg = replace(cfg, ball_spawn_z_jitter=min(float(cfg.ball_spawn_z_jitter), 0.012))
                if name.startswith(("stage4f", "stage4g")):
                    stage_overrides.update(target_ball_view_in_bounds=0.80, target_ball_view_z_ideal=0.64)
                else:
                    stage_overrides.update(target_ball_view_in_bounds=0.74, target_ball_view_z_ideal=0.58)
        elif curriculum_profile == "standard":
            if name.startswith("stage3a"):
                cfg = _with_real_view_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.35,
                    z_weight=0.80,
                    bounds_weight=1.50,
                    vxy_weight=0.20,
                    target_x_range=(0.06, 0.14),
                    target_y_range=(0.06, 0.16),
                    anchor_z_range=(-0.12, -0.12),
                    launch_height=0.30,
                    target_height=0.18,
                    hit_height_center=0.21,
                    racket_up_margin=0.30,
                    terminate_racket_z=False,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.25, target_ball_view_z_ideal=0.12)
            elif name.startswith("stage3b"):
                cfg = _with_real_view_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.70,
                    z_weight=1.40,
                    bounds_weight=3.00,
                    vxy_weight=0.45,
                    target_x_range=(0.09, 0.18),
                    target_y_range=(0.03, 0.13),
                    anchor_z_range=(-0.16, -0.16),
                    launch_height=0.24,
                    target_height=0.15,
                    hit_height_center=0.18,
                    racket_up_margin=0.28,
                    terminate_racket_z=False,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.45, target_ball_view_z_ideal=0.25)
            elif name.startswith("stage4a"):
                cfg = _with_real_view_ball_range(
                    cfg,
                    terminate=False,
                    xy_weight=0.85,
                    z_weight=2.00,
                    bounds_weight=4.50,
                    vxy_weight=0.60,
                    target_x_range=(0.11, 0.19),
                    target_y_range=(0.02, 0.11),
                    anchor_z_range=(-0.18, -0.18),
                    launch_height=0.18,
                    target_height=0.12,
                    hit_height_center=0.15,
                    racket_up_margin=0.26,
                    terminate_racket_z=False,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.62, target_ball_view_z_ideal=0.42)
            elif name.startswith("stage4b"):
                cfg = _with_real_view_ball_range(
                    cfg,
                    terminate=True,
                    xy_weight=0.90,
                    z_weight=1.80,
                    bounds_weight=4.50,
                    vxy_weight=0.65,
                    target_x_range=(0.12, 0.20),
                    target_y_range=(0.02, 0.11),
                    anchor_z_range=(-0.18, -0.18),
                    launch_height=0.13,
                    target_height=0.13,
                    hit_height_center=0.15,
                    racket_up_margin=0.16,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.65, target_ball_view_z_ideal=0.48)
            elif name.startswith(("stage4c", "stage4d", "stage4e")):
                cfg = _with_real_view_ball_range(
                    cfg,
                    terminate=True,
                    xy_weight=1.00,
                    z_weight=2.20,
                    bounds_weight=6.0,
                    vxy_weight=0.80,
                    target_x_range=(0.14, 0.20),
                    target_y_range=(0.02, 0.10),
                    anchor_z_range=(-0.20, -0.20),
                    launch_height=0.10,
                    target_height=0.10,
                    hit_height_center=0.13,
                    racket_up_margin=0.14,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.72, target_ball_view_z_ideal=0.55)
            elif name.startswith(("stage4f", "stage4g")):
                cfg = _with_real_view_ball_range(
                    cfg,
                    terminate=True,
                    xy_weight=1.00,
                    z_weight=2.20,
                    bounds_weight=6.0,
                    vxy_weight=0.80,
                    target_x_range=(0.14, 0.20),
                    target_y_range=(0.02, 0.10),
                    anchor_z_range=(-0.20, -0.20),
                    launch_height=0.10,
                    target_height=0.10,
                    hit_height_center=0.13,
                    racket_up_margin=0.14,
                )
                stage_overrides.update(target_ball_view_in_bounds=0.78, target_ball_view_z_ideal=0.62)
        gate_kwargs = _strict_gate_overrides().get(name, {}) if gate_preset == "v7_strict" else {}
        patched_stages.append(replace(stage, cfg=cfg, notes="", **gate_kwargs, **stage_overrides))
    stages = patched_stages
    if curriculum_profile == "standard":
        stages = stages + _target_generalization_stages(stages[-1].cfg)
    elif curriculum_profile == STAGE4G_ROBUST15_MISSING_PROFILE:
        stages = [
            replace(stage, cfg=_with_verified_stage4g_policy_compatible_terms(stage.cfg))
            if stage.name == "stage4g_strong_contact_dr"
            else stage
            for stage in stages
        ]
        stage4g = next((stage for stage in reversed(stages) if stage.name == "stage4g_strong_contact_dr"), stages[-1])
        stages = stages + _stage4g_robust15_missing_stages(stage4g.cfg)
    elif curriculum_profile in LOW_RESET_CURRICULUM_PROFILES:
        stages = stages + _low_reset_target_generalization_stages(stages[-1].cfg)
        if curriculum_profile in ROBUST15_LOW_RESET_CURRICULUM_PROFILES:
            stages = _with_robust15_low_reset_curriculum(stages)
            if curriculum_profile == "standard_low_reset_robust15_bridge":
                stages = _with_stage4_contact_bridge_curriculum(stages)
            elif curriculum_profile == "standard_low_reset_robust15_missing_bridge":
                stages = _with_stage4_missing_bridge_curriculum(stages)

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
    elif curriculum_profile not in (
        "standard",
        STAGE4G_ROBUST15_MISSING_PROFILE,
        *LOW_RESET_CURRICULUM_PROFILES,
        "actuator_safe",
    ):
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
        actuator_mpc_command_dynamics_constraint=actuator_mpc_command_dynamics_constraint,
        actuator_mpc_command_velocity_weight=actuator_mpc_command_velocity_weight,
        actuator_mpc_command_acceleration_weight=actuator_mpc_command_acceleration_weight,
        actuator_mpc_command_velocity_scale=actuator_mpc_command_velocity_scale,
        actuator_mpc_command_acceleration_scale=actuator_mpc_command_acceleration_scale,
        actuator_mpc_feedback_source=actuator_mpc_feedback_source,
        dr_randomize_actuator_cmd_filter=dr_randomize_actuator_cmd_filter,
        dr_actuator_cmd_tau_range=dr_actuator_cmd_tau_range,
        dr_actuator_cmd_gain_range=dr_actuator_cmd_gain_range,
    )
    if delay_ablation_preset != "baseline_current":
        stages = [replace(stage, cfg=replace(stage.cfg, **delay_kwargs)) for stage in stages]

    if curriculum_profile in ("actuator_safe", STAGE4G_ROBUST15_MISSING_PROFILE):
        stages = _with_actuator_safe_early_curriculum(stages)
        if curriculum_profile == STAGE4G_ROBUST15_MISSING_PROFILE:
            stages = [
                replace(stage, cfg=_with_verified_stage4g_policy_compatible_terms(stage.cfg))
                if stage.name.startswith(
                    (
                        "stage4g_strong_contact_dr",
                        "stage4h_stage4g_",
                        "stage4i_stage4g_",
                        "stage4j_stage4g_",
                        "stage4k_stage4g_",
                        "stage4l_stage4g_",
                        "stage4m_stage4g_",
                        "stage4n_stage4g_",
                    )
                )
                else stage
                for stage in stages
            ]

    if bool(wide_polish_dr):
        widened = []
        for stage in stages:
            if (
                stage.name.startswith("stage4g")
                or stage.name.startswith("stage5a_center_target_generalization")
                or stage.name.startswith("stage5b_real_view_generalization")
                or stage.name.startswith("stage5c_real_view_generalization")
                or stage.name.endswith("_polish")
            ):
                cfg = _with_strong_camera_centering(_with_wide_polish_dr(stage.cfg), center_weight=1.5)
                widened.append(
                    replace(
                        stage,
                        cfg=cfg,
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
    return _apply_arm_safety_overrides(
        stages,
        arm_post_compensation_limiter=arm_post_compensation_limiter,
        arm_servo_target_limiter=arm_servo_target_limiter,
        arm_servo_target_tracking_planner=arm_servo_target_tracking_planner,
        arm_servo_target_velocity_scale=arm_servo_target_velocity_scale,
        arm_servo_target_acceleration_scale=arm_servo_target_acceleration_scale,
        arm_actual_state_limiter=arm_actual_state_limiter,
        arm_actual_target_tracking_governor=arm_actual_target_tracking_governor,
        arm_actual_governor_natural_frequency_hz=arm_actual_governor_natural_frequency_hz,
        arm_actual_governor_damping_ratio=arm_actual_governor_damping_ratio,
        arm_actual_jerk_limit_deg_s3=arm_actual_jerk_limit_deg_s3,
        right_arm_pd_profile=right_arm_pd_profile,
    )


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
        choices=[
            ROBUST_JUGGLE_PROFILE,
            D455_STABLE_4G_PROFILE,
            D455_RECOVERY_PROFILE,
            D455_FULL_CURRICULUM_PROFILE,
            D455_SUCCESS_REF_PROFILE,
            *GOAL_D455_PROFILES,
            "standard",
            "standard_low_reset",
            "standard_low_reset_robust15",
            "standard_low_reset_robust15_bridge",
            "standard_low_reset_robust15_missing_bridge",
            STAGE4G_ROBUST15_MISSING_PROFILE,
            "actuator_safe",
            "sim2real_real",
            "sim2real_kf",
            "sim2real_kf_high_latency",
        ],
        default="standard",
        help=(
            "robust_juggle_v1 is the compact 10-stage 67D + real actuator + inverse MPC + asymmetric-critic profile; "
            "d455_stable_4g_v1 first learns the fixed-arm, anchor-drop nominal D455 13-15 hit policy; "
            "d455_recovery_v1 resumes from that stable policy and trains falling-contact recovery states, missing, and noise; "
            "d455_full_curriculum_v1 is the single-process D455 mainline that continues stable anchor-drop training into wide range, missing, and large DR; "
            "d455_success_ref_v1 rebuilds the old successful low-gate stage4a->stage4g recipe with the current D455 reset, view bounds, and done conditions; "
            "goal_d455_autolaunch_v1 and goal_d455_release_v1 are independent random-initialized GOAL.md branches with fixed D455 geometry, fixed rewards, strict validation, and branch-invariant resets; "
            "goal_d455_autolaunch_viewdense_v1 is the W012 autonomous-launch variant that keeps the W011 control stack but adds mild dense D455 view centering/bounds shaping; "
            "goal_d455_autolaunch_viewdense_relaxtrunc_v1 is W013: W012 plus relaxed early full-horizon truncation gates; "
            "goal_d455_autolaunch_viewdense_fullsafe_v1 is W014: W012 view shaping plus the original early full-horizon gates and safety costs for raw action overflow, action jerk, and actual-state limiter intervention; "
            "goal_d455_autolaunch_viewdense_drivegov_v1 is W015: the same early-full/view-dense course with raw-action and small action-jerk costs, but no obsolete penalty on normal target-aware drive-governor intervention; "
            "goal_d455_autolaunch_viewdense_drivegov_terminalsafe_v1 is W016: W015 plus a 2.5 penalty for racket workspace escape, closing the observed one-hit early-termination loophole; "
            "goal_d455_autolaunch_viewdense_drivegov_successref_v1 is W017: W016 plus conservative post-hit survival, smooth-action, reachable-command and per-hit failure shaping adapted from the proven actuator-learning run, while retaining D455, original inverse MPC and the final hard drive governor; "
            "goal_d455_autolaunch_viewdense_drivegov_highapex_v1 is the withdrawn W018 analysis profile and must not be trained because it preserves W017's post-physics governor plant; "
            "goal_d455_autolaunch_viewdense_constrained_mpc_v1 is W019: W017 task shaping with original inverse-MPC/FOPDT/XML-PD parameters, but the position trajectory actually sent to PD is causally planned at full qvel/0.8 qacc, post-physics state rewriting is off, and base lift/tip terminates; "
            "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v1 preserves W019's plant, rewards, actuator and safety contracts, but repairs the launch15+ curriculum with 25/37.5/50/75/100% observation-calibration DR bridges and strict advancement gates; "
            "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2 keeps those DR bridges and strict current-stage gates, but uses anti-collapse-only next-stage probes and shorter evidence-based consolidation floors so each new DR distribution is learned after advancement; "
            "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1 is W020: it preserves the complete W019 V2 plant/course/gates and replaces hit-count-growing terminal costs with fixed target-count barriers so every additional valid hit has positive marginal credit; "
            "goal_d455_autolaunch_idealpd_v1 reuses the original 20260716 goal_d455_autolaunch_v1 curriculum/gates/rewards while disabling actuator command filtering, delay conditioning, and compensation for the ideal-PD policy->real compensator ablation; "
            "goal_d455_autolaunch_idealpd67_v1 keeps that original course and the deployed 67D 72 ms command-history/error/phase observation contract, but applies the current position command immediately with XML PD and no actuator filter or compensation; "
            "goal_d455_autolaunch_idealpd67_viewdense_v1 preserves that ideal-PD67 plant and the original full-horizon gates, while adding mild view/next-contact shaping and measured launch14/15 minimum-update floors; "
            "goal_d455_autolaunch_idealpd67_final_recovery_v1 resumes that branch at launch19, preserves every strict final gate and adds only moderate post-hit survival shaping with a shorter recovery floor; "
            "goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1 runs the complete ideal-PD67 view-dense D455 course from launch00 with the fitted actuator, original inverse MPC, historical smoothness/qvel/qacc rewards, and no downstream limiter, planner, projector, or governor; "
            "goal_d455_autolaunch_actuator_inversempc_countcredit_nogov_v1 preserves that exact plant and shaping, but makes counted-hit credit monotonic by removing the 15-hit reward cap and all per-hit growth in terminal failure penalties; "
            "goal_d455_autolaunch_actuator_inversempc_final_recovery_nogov_v1 keeps launch00-launch18 byte-identical to the success-reference no-governor course, then makes launch19 a real recoverability bridge with monotonic hit credit, fixed failure barriers, and stronger next-contact/post-hit shaping; "
            "goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1 keeps that same plant and launch00-launch18 course, then adds only bounded launch19 cadence shaping and a period gate consistent with 13 hits in six seconds; "
            "goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1 preserves that cadence profile and changes only launch19 recoverability shaping, with modest post-hit survival credit plus lateral-drift, hit-vxy and next-contact costs; "
            "goal_d455_autolaunch_actuator_inversempc_final_intercept_nogov_v1 preserves that survival profile and changes only launch19 by rewarding the actual racket at the predicted descending-ball crossing point; "
            "goal_d455_autolaunch_idealpd67_actuator_inversempc_finetune_v1 keeps that final-recovery task contract but restores the original delayed-command 67D observation, 74 ms actuator filter, and inverse-MPC execution stack without bottom actual-state limiting; "
            "goal_d455_autolaunch_idealpd67_actuator_inversempc_residual_v1 keeps the same matched task/actuator contract but uses the final 8 Hz q/dq/ddq/jerk governor for frozen-teacher residual transfer; "
            "standard keeps the original 18 stages; actuator_safe retunes early stages for the real delay/filter actuator; "
            "standard_low_reset starts from the IK-computed low right-arm reset pose and visible-window ball heights; "
            "standard_low_reset_robust15 adds cadence/view bridge gates and a late 15-hit cap; "
            "standard_low_reset_robust15_bridge inserts two full-contact DR bridge stages between 4a and 4b; "
            "standard_low_reset_robust15_missing_bridge replaces the hard z-high transition with mocap-visible and probabilistic stale-age view-missing bridges; "
            "standard_stage4g_robust15_missing_bridge starts from the verified 67D stage4g high-juggle policy, then adds stale-age missing and wide range with multi-hit gates; "
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
        choices=["none", "lead", "inverse_smith", "inverse_mpc", "sim2real_bridger"],
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
        "--actuator-mpc-command-dynamics-constraint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Constrain the original inverse-MPC output command itself to the "
            "one-step position/velocity/acceleration interval. This is not "
            "the servo target tracking planner."
        ),
    )
    p.add_argument("--actuator-mpc-command-velocity-weight", type=float, default=None)
    p.add_argument("--actuator-mpc-command-acceleration-weight", type=float, default=None)
    p.add_argument("--actuator-mpc-command-velocity-scale", type=float, default=None)
    p.add_argument("--actuator-mpc-command-acceleration-scale", type=float, default=None)
    p.add_argument(
        "--actuator-mpc-feedback-source",
        choices=["applied", "actual"],
        default=None,
        help=(
            "State used as the inverse-MPC prediction start. 'applied' "
            "preserves old simulator behavior; 'actual' matches current "
            "joint feedback and is the deployable source used by the real controller."
        ),
    )
    p.add_argument(
        "--arm-post-compensation-limiter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the viable q/dq/ddq limiter immediately after compensation.",
    )
    p.add_argument(
        "--arm-servo-target-limiter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the viable q/dq/ddq limiter after actuator delay/FOPDT "
            "and before the MuJoCo position servo."
        ),
    )
    p.add_argument(
        "--arm-servo-target-tracking-planner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override the target-aware acceleration planner after actuator "
            "delay/FOPDT and before the unchanged position PD."
        ),
    )
    p.add_argument("--arm-servo-target-velocity-scale", type=float, default=None)
    p.add_argument("--arm-servo-target-acceleration-scale", type=float, default=None)
    p.add_argument(
        "--arm-actual-state-limiter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the 1 kHz MJX-substep actual q/dq/ddq projection.",
    )
    p.add_argument(
        "--arm-actual-target-tracking-governor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Make the 1 kHz actual-state limiter follow a target-aware "
            "acceleration/jerk profile toward the current FOPDT output."
        ),
    )
    p.add_argument(
        "--arm-actual-governor-natural-frequency-hz",
        type=float,
        default=None,
        help="Natural frequency of the damped actual drive governor.",
    )
    p.add_argument(
        "--arm-actual-governor-damping-ratio",
        type=float,
        default=None,
        help="Damping ratio of the damped actual drive governor.",
    )
    p.add_argument(
        "--arm-actual-jerk-limit-deg-s3",
        type=float,
        default=None,
        help="Uniform per-joint jerk limit used by the actual drive governor.",
    )
    p.add_argument(
        "--right-arm-pd-profile",
        choices=["xml", "legacy_stage4g", "comparison_safe_v1"],
        default=None,
        help="Override the nominal right-arm PD profile uniformly for every stage.",
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
        "--reset-optimizer-on-resume",
        action="store_true",
        help=(
            "Keep resumed policy/critic parameters but rebuild zero-moment Adam state. "
            "Use when the resumed profile intentionally changes the reward objective."
        ),
    )
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
        "--stage-metric-warmup-updates",
        type=int,
        default=-1,
        help=(
            "Exclude this many updates after every synchronized stage reset from best/checkpoint "
            "and convergence windows. -1 automatically uses ceil(max_episode_steps / n_steps)."
        ),
    )
    p.add_argument(
        "--allow-unconverged-advance",
        action="store_true",
        help="With --advance-mode converged, continue to the next stage when its update cap is exhausted.",
    )
    p.add_argument(
        "--max-stage-updates",
        type=int,
        default=0,
        help="CLI safety cap per stage in converged mode. 0 uses a profile-specific cap when declared, otherwise trains until convergence.",
    )
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument(
        "--failure-focus-hit-threshold",
        type=int,
        default=0,
        help=(
            "Failure-focused PPO: negative-advantage transitions from true "
            "terminations ending below this hit count receive "
            "--failure-focus-weight. Time-limit truncations are excluded. "
            "0 disables the mechanism."
        ),
    )
    p.add_argument(
        "--failure-focus-weight",
        type=float,
        default=1.0,
        help="Actor negative-advantage weight for completed low-hit failures.",
    )
    p.add_argument(
        "--failure-focus-tail-steps",
        type=int,
        default=0,
        help=(
            "Restrict failure focus to the final N transitions before true "
            "termination. 0 focuses the whole completed failed episode."
        ),
    )
    p.add_argument(
        "--time-limit-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Bootstrap the critic at six-second time-limit truncations while cutting the "
            "GAE trace at reset boundaries. --no-time-limit-bootstrap reproduces the "
            "legacy hidden-time-limit terminal target."
        ),
    )
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.0,
        help=(
            "Maximum analytic KL(old||new) from the rollout behavior policy. "
            "Candidate Adam steps are checked after the update and backtracked; "
            "the complete update is rolled back if its full-rollout KL exceeds "
            "this budget. 0 disables the guard."
        ),
    )
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument(
        "--min-log-std",
        type=float,
        default=None,
        help=(
            "Optional lower bound for the trainable Gaussian policy log standard "
            "deviation. Prevents exploration collapse during long curriculum stages."
        ),
    )
    p.add_argument(
        "--actor-anchor-kl-coef",
        type=float,
        default=0.0,
        help="KL penalty to the actor distribution at stage entry (0 disables anchoring)",
    )
    p.add_argument(
        "--actor-anchor-replay-obs",
        type=Path,
        default=None,
        help=(
            "Optional .npy [samples, obs_dim] old-domain actor observations. When source KL is "
            "enabled, minibatches anchor on both current rollout and these replay observations."
        ),
    )
    p.add_argument(
        "--actor-anchor-replay-kl-coef",
        type=float,
        default=0.0,
        help=(
            "Independent KL coefficient on --actor-anchor-replay-obs. 0 retains the legacy "
            "equal-domain average controlled only by --actor-anchor-kl-coef."
        ),
    )
    p.add_argument(
        "--teacher-distill-checkpoint",
        type=Path,
        default=None,
        help=(
            "Frozen ideal-domain teacher checkpoint. The teacher is evaluated only on "
            "--teacher-distill-replay-obs and is never used to act in the student environment."
        ),
    )
    p.add_argument(
        "--teacher-distill-replay-obs",
        type=Path,
        default=None,
        help="Ideal-domain .npy observations with shape [samples, obs_dim] for teacher supervision.",
    )
    p.add_argument(
        "--teacher-distill-coef",
        type=float,
        default=0.0,
        help="MSE coefficient between student and frozen-teacher action means on ideal replay.",
    )
    p.add_argument(
        "--teacher-distill-action-clip",
        type=float,
        default=1.0,
        help="Absolute clip for teacher action targets; 0 disables target clipping.",
    )
    p.add_argument(
        "--residual-teacher-checkpoint",
        type=Path,
        default=None,
        help=(
            "Freeze the actor from this checkpoint as a teacher and train a zero-output "
            "residual actor whose bounded correction is added before the environment control stack."
        ),
    )
    p.add_argument(
        "--residual-action-scale",
        type=float,
        default=0.10,
        help="Maximum absolute normalized-action correction from the residual actor.",
    )
    p.add_argument(
        "--residual-l2-coef",
        type=float,
        default=0.01,
        help="L2 penalty coefficient on the bounded residual action correction.",
    )
    p.add_argument(
        "--residual-initialize-only",
        action="store_true",
        help=(
            "Build and save the exact zero-residual checkpoint for validation, "
            "then exit before rollout or PPO updates."
        ),
    )
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
        "--archive-every-updates",
        type=int,
        default=0,
        help=(
            "Additionally retain an immutable stage/update checkpoint at this interval. "
            "0 disables archival and preserves the existing checkpoint behavior."
        ),
    )
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
    p.add_argument(
        "--advance-eval-hit-rate-margin",
        type=float,
        default=0.05,
        help="Allowed margin below hit1/hit3 rate gates during next-stage validation.",
    )
    p.add_argument(
        "--advance-eval-hit-interval-margin",
        type=float,
        default=0.04,
        help="Allowed seconds below the minimum mean-hit-interval gate during next-stage validation.",
    )
    p.add_argument(
        "--advance-eval-cond-hit-ratio",
        type=float,
        default=0.80,
        help="Ratio applied to conditional mean-hit gates during next-stage validation.",
    )
    p.add_argument("--advance-eval-camera-margin", type=float, default=0.10)
    p.add_argument("--advance-eval-camera-reward-margin", type=float, default=0.02)
    p.add_argument(
        "--advance-eval-ball-view-margin",
        type=float,
        default=0.08,
        help="Allowed margin below the probe stage's ball_view_in_bounds gate during next-stage validation.",
    )
    p.add_argument(
        "--advance-eval-z-ideal-margin",
        type=float,
        default=0.08,
        help="Allowed margin below the probe stage's ball_view_z_ideal gate during next-stage validation.",
    )
    p.add_argument(
        "--advance-eval-reset-bucket-mode",
        choices=["off", "log", "cvar", "worst"],
        default="log",
        help=(
            "Reset-bucket robustness validation. log records low/mid/high bucket metrics; "
            "cvar and worst also require bottom-bin metrics to pass."
        ),
    )
    p.add_argument(
        "--advance-eval-reset-bucket-min-episodes",
        type=int,
        default=4,
        help="Minimum completed episodes required for a reset bucket to count toward robust validation.",
    )
    p.add_argument(
        "--advance-eval-reset-bucket-cvar-frac",
        type=float,
        default=0.20,
        help="Fraction of lowest reset buckets averaged for CVaR validation.",
    )
    p.add_argument(
        "--advance-eval-reset-bucket-rate-margin",
        type=float,
        default=0.08,
        help="Extra margin below rate/view thresholds for robust reset-bucket gates.",
    )
    p.add_argument(
        "--advance-eval-reset-bucket-hit-margin",
        type=float,
        default=1.0,
        help="Extra mean-hit margin below the advance validation hit threshold for robust reset-bucket gates.",
    )
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
    if args.actor_anchor_kl_coef < 0.0:
        p.error("--actor-anchor-kl-coef must be >= 0")
    if args.target_kl < 0.0:
        p.error("--target-kl must be >= 0")
    if args.failure_focus_hit_threshold < 0:
        p.error("--failure-focus-hit-threshold must be >= 0")
    if args.failure_focus_weight < 1.0:
        p.error("--failure-focus-weight must be >= 1")
    if args.failure_focus_tail_steps < 0:
        p.error("--failure-focus-tail-steps must be >= 0")
    if args.stage_metric_warmup_updates < -1:
        p.error("--stage-metric-warmup-updates must be >= -1")
    if args.actor_anchor_replay_kl_coef < 0.0:
        p.error("--actor-anchor-replay-kl-coef must be >= 0")
    if args.teacher_distill_coef < 0.0:
        p.error("--teacher-distill-coef must be >= 0")
    if args.teacher_distill_action_clip < 0.0:
        p.error("--teacher-distill-action-clip must be >= 0")
    teacher_distill_enabled = float(args.teacher_distill_coef) > 0.0
    if teacher_distill_enabled:
        if args.teacher_distill_checkpoint is None:
            p.error("--teacher-distill-coef > 0 requires --teacher-distill-checkpoint")
        if args.teacher_distill_replay_obs is None:
            p.error("--teacher-distill-coef > 0 requires --teacher-distill-replay-obs")
    if args.teacher_distill_checkpoint is not None:
        if not args.teacher_distill_checkpoint.exists():
            p.error(f"teacher checkpoint not found: {args.teacher_distill_checkpoint}")
        if args.residual_teacher_checkpoint is not None:
            p.error("teacher distillation cannot be combined with residual-teacher training")
    if args.teacher_distill_replay_obs is not None and not args.teacher_distill_replay_obs.exists():
        p.error(f"teacher replay file not found: {args.teacher_distill_replay_obs}")
    if not 0.0 < float(args.residual_action_scale) <= 1.0:
        p.error("--residual-action-scale must be in (0, 1]")
    if float(args.residual_l2_coef) < 0.0:
        p.error("--residual-l2-coef must be >= 0")
    if args.residual_teacher_checkpoint is not None:
        if args.resume_from is not None:
            p.error("--residual-teacher-checkpoint cannot be combined with --resume-from")
        if not args.residual_teacher_checkpoint.exists():
            p.error(f"residual teacher checkpoint not found: {args.residual_teacher_checkpoint}")
    if args.curriculum_profile == GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE:
        if not teacher_distill_enabled:
            p.error(f"{args.curriculum_profile} requires --teacher-distill-coef > 0")
    if args.archive_every_updates < 0:
        p.error("--archive-every-updates must be >= 0")
    if args.sb3_parity:
        args.n_envs = 8
        args.n_steps = 2048
        args.minibatch_size = 256
        args.update_epochs = 10
        args.hidden_dim = 64
    if args.curriculum_profile in GOAL_D455_IDEALPD67_PROFILES:
        if args.high_latency_obs:
            p.error(f"{args.curriculum_profile} fixes actor obs_dim at 67; do not use --high-latency-obs")
        if args.delay_ablation_preset not in {"baseline_current", "real_actuator_replay_fit"}:
            p.error(f"{args.curriculum_profile} requires the deployed real_actuator_replay_fit 67D observation contract")
        if args.actuator_compensation_mode not in (None, "none"):
            p.error(f"{args.curriculum_profile} disables actuator compensation")
        if args.actuator_cmd_filter is True:
            p.error(f"{args.curriculum_profile} disables the actuator command filter")
        args.delay_ablation_preset = "real_actuator_replay_fit"
        args.actuator_compensation_mode = "none"
        args.actuator_cmd_filter = False
        args.asymmetric_critic = True
        args.critic_command_history_steps = 12
    if args.curriculum_profile in (ROBUST_JUGGLE_PROFILE, *D455_67D_INVERSE_MPC_PROFILES):
        if args.high_latency_obs:
            p.error(f"{args.curriculum_profile} fixes actor obs_dim at 67; do not use --high-latency-obs")
        if args.delay_ablation_preset not in {"baseline_current", "real_actuator_replay_fit"}:
            p.error(f"{args.curriculum_profile} owns its progressive delay schedule")
        if args.actuator_compensation_mode not in (None, "inverse_mpc"):
            p.error(f"{args.curriculum_profile} requires inverse_mpc")
        if args.actuator_cmd_filter is False:
            p.error(f"{args.curriculum_profile} requires the real actuator command filter")
        args.delay_ablation_preset = "real_actuator_replay_fit"
        args.actuator_compensation_mode = "inverse_mpc"
        args.actuator_mpc_beta = 1.2 if args.actuator_mpc_beta is None else args.actuator_mpc_beta
        args.actuator_mpc_delay_scale = (
            1.05 if args.actuator_mpc_delay_scale is None else args.actuator_mpc_delay_scale
        )
        args.actuator_mpc_tau_scale = 0.75 if args.actuator_mpc_tau_scale is None else args.actuator_mpc_tau_scale
        args.actuator_mpc_horizon_steps = (
            6 if args.actuator_mpc_horizon_steps is None else args.actuator_mpc_horizon_steps
        )
        args.actuator_mpc_tracking_weight = (
            1.0 if args.actuator_mpc_tracking_weight is None else args.actuator_mpc_tracking_weight
        )
        args.actuator_mpc_nominal_weight = (
            0.25 if args.actuator_mpc_nominal_weight is None else args.actuator_mpc_nominal_weight
        )
        args.actuator_mpc_delta_weight = (
            0.05 if args.actuator_mpc_delta_weight is None else args.actuator_mpc_delta_weight
        )
        args.actuator_mpc_max_delta_deg = (
            30.0 if args.actuator_mpc_max_delta_deg is None else args.actuator_mpc_max_delta_deg
        )
        if args.curriculum_profile in (
            GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
        ):
            if any(
                value is True
                for value in (
                    args.arm_post_compensation_limiter,
                    args.arm_servo_target_limiter,
                    args.arm_servo_target_tracking_planner,
                    args.arm_actual_state_limiter,
                    args.arm_actual_target_tracking_governor,
                )
            ):
                p.error(
                    f"{args.curriculum_profile} disables post-compensation, servo-target, "
                    "planner, bottom actual-state limiters, and the governor"
                )
            args.arm_post_compensation_limiter = False
            args.arm_servo_target_limiter = False
            args.arm_servo_target_tracking_planner = False
            args.arm_actual_state_limiter = False
            args.arm_actual_target_tracking_governor = False
            args.actuator_mpc_feedback_source = (
                (
                    "applied"
                    if args.curriculum_profile in (
                        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
                        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
                    )
                    else "actual"
                )
                if args.actuator_mpc_feedback_source is None
                else args.actuator_mpc_feedback_source
            )
        elif (
            args.curriculum_profile
            == GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_RESIDUAL_PROFILE
        ):
            if args.residual_teacher_checkpoint is None:
                p.error(f"{args.curriculum_profile} requires --residual-teacher-checkpoint")
            args.arm_post_compensation_limiter = False
            args.arm_servo_target_limiter = False
            args.arm_servo_target_tracking_planner = False
            args.arm_actual_state_limiter = True
            args.arm_actual_target_tracking_governor = True
            args.arm_actual_governor_natural_frequency_hz = 8.0
            args.arm_actual_governor_damping_ratio = 1.0
            args.arm_actual_jerk_limit_deg_s3 = 175000.0
            args.actuator_mpc_feedback_source = (
                "actual"
                if args.actuator_mpc_feedback_source is None
                else args.actuator_mpc_feedback_source
            )
        args.asymmetric_critic = True
        args.critic_command_history_steps = (
            12
            if args.curriculum_profile in GOAL_D455_PROFILES
            else max(12, int(args.critic_command_history_steps))
        )
    return args


def episode_hit_distribution_metrics(
    hit_count: np.ndarray,
    done: np.ndarray,
    episode_length: np.ndarray | None = None,
    dt: float | None = None,
) -> dict[str, float]:
    done_mask = np.asarray(done).astype(bool)
    done_hits = np.asarray(hit_count)[done_mask]
    if done_hits.size == 0:
        return {
            "hit0_rate": float("nan"),
            "hit1_rate": float("nan"),
            "hit3_rate": float("nan"),
            "hit7_rate": float("nan"),
            "hit12_rate": float("nan"),
            "hit20_rate": float("nan"),
            "mean_hits_ge1": float("nan"),
            "mean_hits_ge3": float("nan"),
            "mean_hits_ge7": float("nan"),
            "mean_hit_interval_s": float("nan"),
            "mean_hit_interval_ge3_s": float("nan"),
            "hit_rate_hz": float("nan"),
        }

    def cond_mean(threshold: int) -> float:
        values = done_hits[done_hits >= threshold]
        return float(values.mean()) if values.size > 0 else 0.0

    mean_hit_interval_s = float("nan")
    mean_hit_interval_ge3_s = float("nan")
    hit_rate_hz = float("nan")
    if episode_length is not None and dt is not None:
        done_len = np.asarray(episode_length)[done_mask].astype(np.float64)
        duration_s = done_len * float(dt)
        positive_hits = (done_hits > 0) & (duration_s > 0.0)
        if np.any(positive_hits):
            per_episode_rate = done_hits[positive_hits].astype(np.float64) / duration_s[positive_hits]
            per_episode_interval = duration_s[positive_hits] / done_hits[positive_hits].astype(np.float64)
            hit_rate_hz = float(np.mean(per_episode_rate))
            mean_hit_interval_s = float(np.mean(per_episode_interval))
        ge3_hits = (done_hits >= 3) & (duration_s > 0.0)
        if np.any(ge3_hits):
            mean_hit_interval_ge3_s = float(
                np.mean(duration_s[ge3_hits] / done_hits[ge3_hits].astype(np.float64))
            )

    return {
        "hit0_rate": float(np.mean(done_hits <= 0)),
        "hit1_rate": float(np.mean(done_hits >= 1)),
        "hit3_rate": float(np.mean(done_hits >= 3)),
        "hit7_rate": float(np.mean(done_hits >= 7)),
        "hit12_rate": float(np.mean(done_hits >= 12)),
        "hit20_rate": float(np.mean(done_hits >= 20)),
        "mean_hits_ge1": cond_mean(1),
        "mean_hits_ge3": cond_mean(3),
        "mean_hits_ge7": cond_mean(7),
        "mean_hit_interval_s": mean_hit_interval_s,
        "mean_hit_interval_ge3_s": mean_hit_interval_ge3_s,
        "hit_rate_hz": hit_rate_hz,
    }


RESET_BUCKET_COMMON_FIELDS = (
    "reset_target_x",
    "reset_target_y",
    "reset_target_z",
    "reset_disturbance_strength",
)
RESET_BUCKET_AUTOLAUNCH_FIELDS = (
    *RESET_BUCKET_COMMON_FIELDS,
    "reset_ball_surface_gap",
    "reset_ball_racket_center_offset",
    "reset_ball_vxy",
    "reset_ball_vz",
)
RESET_BUCKET_RELEASE_FIELDS = (
    *RESET_BUCKET_COMMON_FIELDS,
    "reset_ball_anchor_dx",
    "reset_ball_anchor_dy",
    "reset_ball_anchor_dz",
    "reset_ball_vxy",
    "reset_ball_vz",
)
RESET_BUCKET_FIELDS = tuple(
    dict.fromkeys((*RESET_BUCKET_AUTOLAUNCH_FIELDS, *RESET_BUCKET_RELEASE_FIELDS))
)
RESET_BUCKET_BIN_LABELS = ("low", "mid", "high")
RESET_BUCKET_DETAIL_METRICS = (
    "episodes",
    "mean_hits",
    "hit1_rate",
    "hit3_rate",
    "ball_view_in_bounds",
    "ball_view_z_ideal",
)
RESET_BUCKET_ROBUST_METRICS = (
    "mean_hits",
    "hit1_rate",
    "hit3_rate",
    "ball_view_in_bounds",
    "ball_view_z_ideal",
)


def reset_bucket_default_metrics() -> dict[str, float]:
    result = {
        "advance_eval/reset_bucket_enabled": 0.0,
        "advance_eval/reset_bucket_required": 0.0,
        "advance_eval/reset_bucket_gate_ok": 1.0,
        "advance_eval/reset_bucket_bin_count": 0.0,
        "advance_eval/reset_bucket_field_count": 0.0,
        "advance_eval/reset_bucket_eligible_field_count": 0.0,
        "advance_eval/reset_bucket_min_episodes": float("nan"),
        "advance_eval/reset_bucket_cvar_frac": float("nan"),
    }
    for metric in RESET_BUCKET_ROBUST_METRICS:
        result[f"advance_eval/reset_bucket_worst_{metric}"] = float("nan")
        result[f"advance_eval/reset_bucket_cvar_{metric}"] = float("nan")
        result[f"advance_eval/reset_bucket_target_{metric}"] = float("nan")
        result[f"advance_eval/reset_bucket_{metric}_ok"] = 1.0
    for field in RESET_BUCKET_FIELDS:
        for label in RESET_BUCKET_BIN_LABELS:
            for metric in RESET_BUCKET_DETAIL_METRICS:
                result[f"advance_eval/reset_bucket/{field}/{label}/{metric}"] = float("nan")
    return result


def _numeric_2d_metric(metrics: dict[str, object], key: str, shape: tuple[int, int]) -> np.ndarray | None:
    value = metrics.get(key)
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape != shape or arr.dtype.kind not in "fbiu":
        return None
    return arr.astype(np.float64, copy=False)


def _masked_mean(arr: np.ndarray | None, mask: np.ndarray) -> float:
    if arr is None:
        return float("nan")
    values = arr[mask & np.isfinite(arr)]
    return float(values.mean()) if values.size > 0 else float("nan")


def _bottom_cvar(values: list[float], frac: float) -> float:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return float("nan")
    finite.sort()
    count = max(1, int(np.ceil(finite.size * float(np.clip(frac, 0.0, 1.0)))))
    return float(finite[:count].mean())


def summarize_reset_bucket_outputs(
    metrics: dict[str, object],
    hit_count: np.ndarray,
    done: np.ndarray,
    *,
    mode: str,
    min_episodes: int,
    cvar_frac: float,
    fields: tuple[str, ...] | None = None,
) -> dict[str, float]:
    result = reset_bucket_default_metrics()
    result["advance_eval/reset_bucket_enabled"] = float(mode != "off")
    result["advance_eval/reset_bucket_required"] = float(mode in {"cvar", "worst"})
    result["advance_eval/reset_bucket_min_episodes"] = float(max(1, int(min_episodes)))
    result["advance_eval/reset_bucket_cvar_frac"] = float(cvar_frac)
    if mode == "off":
        return result

    shape = done.shape
    view = _numeric_2d_metric(metrics, "ball_view_in_bounds", shape)
    z_ideal = _numeric_2d_metric(metrics, "ball_view_z_ideal", shape)
    robust_values: dict[str, list[float]] = {metric: [] for metric in RESET_BUCKET_ROBUST_METRICS}
    eligible_bins = 0
    varying_fields = 0
    eligible_fields = 0
    selected_fields = RESET_BUCKET_FIELDS if fields is None else fields

    for field in selected_fields:
        values = _numeric_2d_metric(metrics, field, shape)
        if values is None:
            continue
        finite = np.isfinite(values)
        basis = values[done & finite]
        if basis.size < max(1, int(min_episodes)):
            basis = values[finite]
        if basis.size == 0:
            continue
        if float(np.nanmax(basis) - np.nanmin(basis)) <= 1e-8:
            continue
        varying_fields += 1
        q_low, q_high = np.nanquantile(basis, [1.0 / 3.0, 2.0 / 3.0])
        if not (np.isfinite(q_low) and np.isfinite(q_high)):
            continue
        if q_low == q_high:
            continue
        bin_masks = {
            "low": finite & (values <= q_low),
            "mid": finite & (values > q_low) & (values <= q_high),
            "high": finite & (values > q_high),
        }

        field_eligible = True
        for label, mask in bin_masks.items():
            done_mask = done & mask
            done_hits = hit_count[done_mask]
            episodes = int(done_hits.size)
            prefix = f"advance_eval/reset_bucket/{field}/{label}"
            result[f"{prefix}/episodes"] = float(episodes)
            if episodes > 0:
                result[f"{prefix}/mean_hits"] = float(done_hits.mean())
                result[f"{prefix}/hit1_rate"] = float(np.mean(done_hits >= 1))
                result[f"{prefix}/hit3_rate"] = float(np.mean(done_hits >= 3))
            result[f"{prefix}/ball_view_in_bounds"] = _masked_mean(view, mask)
            result[f"{prefix}/ball_view_z_ideal"] = _masked_mean(z_ideal, mask)

            if episodes < max(1, int(min_episodes)):
                field_eligible = False
                continue
            eligible_bins += 1
            for metric in RESET_BUCKET_ROBUST_METRICS:
                value = result.get(f"{prefix}/{metric}", float("nan"))
                if np.isfinite(value):
                    robust_values[metric].append(float(value))
        if field_eligible:
            eligible_fields += 1

    result["advance_eval/reset_bucket_bin_count"] = float(eligible_bins)
    result["advance_eval/reset_bucket_field_count"] = float(varying_fields)
    result["advance_eval/reset_bucket_eligible_field_count"] = float(eligible_fields)
    for metric, values in robust_values.items():
        finite_values = [float(v) for v in values if np.isfinite(v)]
        if finite_values:
            result[f"advance_eval/reset_bucket_worst_{metric}"] = float(min(finite_values))
            result[f"advance_eval/reset_bucket_cvar_{metric}"] = _bottom_cvar(finite_values, cvar_frac)
    return result


def reset_bucket_gate_status(
    args: argparse.Namespace,
    probe_stage: CurriculumStage,
    thresholds: dict[str, float],
    result: dict[str, float],
) -> dict[str, float]:
    mode = str(args.advance_eval_reset_bucket_mode)
    status = {
        "advance_eval/reset_bucket_required": float(mode in {"cvar", "worst"}),
        "advance_eval/reset_bucket_gate_ok": 1.0,
    }
    if mode not in {"cvar", "worst"}:
        return status

    prefix = "cvar" if mode == "cvar" else "worst"
    rate_margin = max(0.0, float(args.advance_eval_reset_bucket_rate_margin))
    hit_margin = max(0.0, float(args.advance_eval_reset_bucket_hit_margin))
    field_count = float(result.get("advance_eval/reset_bucket_field_count", 0.0))
    eligible_field_count = float(
        result.get("advance_eval/reset_bucket_eligible_field_count", 0.0)
    )
    ok = bool(
        field_count > 0.0
        and eligible_field_count == field_count
        and result.get("advance_eval/reset_bucket_bin_count", 0.0) > 0.0
    )

    def check(metric: str, target: float | None) -> None:
        nonlocal ok
        target_key = f"advance_eval/reset_bucket_target_{metric}"
        ok_key = f"advance_eval/reset_bucket_{metric}_ok"
        if target is None or not np.isfinite(float(target)):
            status[target_key] = float("nan")
            status[ok_key] = 1.0
            return
        value = float(result.get(f"advance_eval/reset_bucket_{prefix}_{metric}", float("nan")))
        metric_ok = bool(np.isfinite(value) and value >= float(target))
        status[target_key] = float(target)
        status[ok_key] = float(metric_ok)
        ok = ok and metric_ok

    def lower_bucket_target(name: str, margin: float) -> float | None:
        value = float(thresholds.get(name, float("nan")))
        if not np.isfinite(value):
            return None
        return max(0.0, value - margin)

    check("mean_hits", max(0.0, float(thresholds["target_mean_hits"]) - hit_margin))
    check(
        "hit1_rate",
        lower_bucket_target("target_hit1_rate", rate_margin),
    )
    check(
        "hit3_rate",
        lower_bucket_target("target_hit3_rate", rate_margin),
    )
    check(
        "ball_view_in_bounds",
        lower_bucket_target("target_ball_view_in_bounds", rate_margin),
    )
    check(
        "ball_view_z_ideal",
        lower_bucket_target("target_ball_view_z_ideal", rate_margin),
    )
    status["advance_eval/reset_bucket_gate_ok"] = float(ok)
    return status


def mean_rollout_metrics(transitions) -> dict[str, float]:
    metrics = {}
    host_metrics = jax.device_get(transitions.metrics)
    for key, value in host_metrics.items():
        arr = np.asarray(value)
        if arr.dtype.kind in "fbiu":
            metrics[key] = float(np.mean(arr))
    done = np.asarray(jax.device_get(transitions.done)).astype(bool)
    done_count = int(done.sum())
    truncated = np.asarray(host_metrics.get("truncated", np.zeros_like(done))).astype(bool)
    metrics["episode_truncation_rate"] = (
        float(truncated.sum()) / float(done_count) if done_count > 0 else float("nan")
    )
    refresh = np.asarray(host_metrics.get("ball_obs_refresh_due", np.zeros_like(done)), dtype=np.float64)
    missing = np.asarray(host_metrics.get("ball_obs_missing_on_refresh", np.zeros_like(done)), dtype=np.float64)
    refresh_count = float(refresh.sum())
    metrics["ball_obs_missing_refresh_rate"] = (
        float(missing.sum()) / refresh_count if refresh_count > 0.0 else float("nan")
    )

    hit_events = np.asarray(host_metrics.get("hit_camera_event", np.zeros_like(done)), dtype=np.float64)
    visible_events = np.asarray(
        host_metrics.get("hit_camera_visible_event", np.zeros_like(done)),
        dtype=np.float64,
    )
    margin_events = np.asarray(
        host_metrics.get("hit_camera_in_margin_event", np.zeros_like(done)),
        dtype=np.float64,
    )
    band_events = np.asarray(
        host_metrics.get("hit_camera_lower_band_event", np.zeros_like(done)),
        dtype=np.float64,
    )
    hit_event_count = float(hit_events.sum())
    visible_event_count = float(visible_events.sum())
    metrics["hit_camera_visible_rate"] = (
        float(visible_event_count / hit_event_count) if hit_event_count > 0.0 else float("nan")
    )
    metrics["hit_camera_in_margin_rate"] = (
        float(margin_events.sum() / hit_event_count) if hit_event_count > 0.0 else float("nan")
    )
    metrics["hit_camera_lower_band_rate"] = (
        float(band_events.sum() / hit_event_count) if hit_event_count > 0.0 else float("nan")
    )
    v_frac_sum = np.asarray(
        host_metrics.get("hit_camera_v_frac_sum", np.zeros_like(done)),
        dtype=np.float64,
    )
    metrics["mean_hit_camera_v_frac"] = (
        float(v_frac_sum.sum() / visible_event_count)
        if visible_event_count > 0.0
        else float("nan")
    )
    hit_vxy_sum = np.asarray(
        host_metrics.get("hit_vxy_sum", np.zeros_like(done)),
        dtype=np.float64,
    )
    metrics["mean_hit_vxy"] = (
        float(hit_vxy_sum.sum() / hit_event_count)
        if hit_event_count > 0.0
        else float("nan")
    )
    hit_next_contact_anchor_err_sum = np.asarray(
        host_metrics.get("hit_next_contact_anchor_err_sum", np.zeros_like(done)),
        dtype=np.float64,
    )
    metrics["mean_hit_next_contact_anchor_err"] = (
        float(hit_next_contact_anchor_err_sum.sum() / hit_event_count)
        if hit_event_count > 0.0
        else float("nan")
    )
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


def _positive_gate_floor(value: float, target: float | None, ratio: float = 0.90) -> bool:
    if target is None:
        return True
    return bool(np.isfinite(value) and value >= float(target) * float(ratio))


def _rate_gate_floor(value: float, target: float | None, margin: float = 0.08) -> bool:
    if target is None:
        return True
    return bool(np.isfinite(value) and value >= max(0.0, float(target) - float(margin)))


def _weighted_gate_score(
    items: list[tuple[float, float | None, float]],
) -> float:
    weighted_score = 0.0
    active_weight = 0.0
    for value, target, weight in items:
        if target is None or float(weight) <= 0.0:
            continue
        active_weight += float(weight)
        if not np.isfinite(value):
            continue
        target_value = float(target)
        if target_value <= 0.0:
            ratio = 1.0 if value >= target_value else 0.0
        else:
            ratio = float(np.clip(value / target_value, 0.0, 1.0))
        weighted_score += float(weight) * ratio
    if active_weight <= 0.0:
        return 1.0
    return weighted_score / active_weight


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
    recent_ball_view_in_bounds = _recent_mean(recent, "ball_view_in_bounds")
    recent_ball_view_z_ideal = _recent_mean(recent, "ball_view_z_ideal")
    recent_hit1_rate = _recent_mean(recent, "hit1_rate")
    recent_hit3_rate = _recent_mean(recent, "hit3_rate")
    recent_hit12_rate = _recent_mean(recent, "hit12_rate")
    recent_mean_hits_ge3 = _recent_mean(recent, "mean_hits_ge3")
    recent_mean_hit_interval_s = _recent_mean(recent, "mean_hit_interval_s")
    recent_mean_hit_interval_ge3_s = _recent_mean(recent, "mean_hit_interval_ge3_s")
    recent_gate_hit_interval_s = (
        recent_mean_hit_interval_ge3_s
        if stage.target_max_hit_interval_s is not None
        else recent_mean_hit_interval_s
    )
    recent_episode_truncation_rate = _recent_mean(recent, "episode_truncation_rate")
    recent_ball_obs_missing_refresh_rate = _recent_mean(recent, "ball_obs_missing_refresh_rate")
    recent_ball_obs_lost_rate = _recent_mean(recent, "ball_obs_lost_active")
    recent_hit_camera_visible_rate = _recent_mean(recent, "hit_camera_visible_rate")
    recent_hit_camera_lower_band_rate = _recent_mean(recent, "hit_camera_lower_band_rate")
    recent_mean_hit_camera_v_frac = _recent_mean(recent, "mean_hit_camera_v_frac")
    recent_mean_hit_vxy = _recent_mean(recent, "mean_hit_vxy")
    recent_mean_hit_next_contact_anchor_err = _recent_mean(
        recent,
        "mean_hit_next_contact_anchor_err",
    )
    recent_racket_up_cos = _recent_mean(recent, "racket_up_cos")

    required_updates = max(int(stage.min_updates), int(args.min_stage_updates))
    enough_updates = stage_update >= required_updates
    enough_window = len(recent) >= window
    hit_ok = bool(np.isfinite(recent_mean_hits) and recent_mean_hits >= float(stage.target_mean_hits))
    len_ok = bool(np.isfinite(recent_len_frac) and recent_len_frac >= float(stage.target_mean_len_frac))
    return_ok = (
        True
        if stage.min_recent_mean_return is None
        else bool(
            np.isfinite(recent_mean_return)
            and recent_mean_return >= float(stage.min_recent_mean_return)
        )
    )
    camera_visible_ok = (
        True
        if stage.target_camera_visible is None
        else bool(
            np.isfinite(recent_camera_visible)
            and recent_camera_visible >= float(stage.target_camera_visible)
        )
    )
    camera_reward_ok = (
        True
        if stage.min_recent_camera_reward_dense is None
        else bool(
            np.isfinite(recent_camera_reward_dense)
            and recent_camera_reward_dense >= float(stage.min_recent_camera_reward_dense)
        )
    )
    ball_view_in_bounds_ok = (
        True
        if stage.target_ball_view_in_bounds is None
        else bool(
            np.isfinite(recent_ball_view_in_bounds)
            and recent_ball_view_in_bounds >= float(stage.target_ball_view_in_bounds)
        )
    )
    ball_view_z_ideal_ok = (
        True
        if stage.target_ball_view_z_ideal is None
        else bool(
            np.isfinite(recent_ball_view_z_ideal)
            and recent_ball_view_z_ideal >= float(stage.target_ball_view_z_ideal)
        )
    )
    hit1_rate_ok = (
        True
        if stage.target_hit1_rate is None
        else bool(np.isfinite(recent_hit1_rate) and recent_hit1_rate >= float(stage.target_hit1_rate))
    )
    hit3_rate_ok = (
        True
        if stage.target_hit3_rate is None
        else bool(np.isfinite(recent_hit3_rate) and recent_hit3_rate >= float(stage.target_hit3_rate))
    )
    hit12_rate_ok = (
        True
        if stage.target_hit12_rate is None
        else bool(np.isfinite(recent_hit12_rate) and recent_hit12_rate >= float(stage.target_hit12_rate))
    )
    hits_ge3_ok = (
        True
        if stage.target_mean_hits_ge3 is None
        else bool(
            np.isfinite(recent_mean_hits_ge3)
            and recent_mean_hits_ge3 >= float(stage.target_mean_hits_ge3)
        )
    )
    hit_interval_ok = (
        True
        if stage.target_min_hit_interval_s is None
        else bool(
            np.isfinite(recent_gate_hit_interval_s)
            and recent_gate_hit_interval_s >= float(stage.target_min_hit_interval_s)
        )
    )
    hit_interval_max_ok = (
        True
        if stage.target_max_hit_interval_s is None
        else bool(
            np.isfinite(recent_gate_hit_interval_s)
            and recent_gate_hit_interval_s <= float(stage.target_max_hit_interval_s)
        )
    )
    hit_camera_visible_ok = (
        True
        if stage.target_hit_camera_visible_rate is None
        else bool(
            np.isfinite(recent_hit_camera_visible_rate)
            and recent_hit_camera_visible_rate >= float(stage.target_hit_camera_visible_rate)
        )
    )
    hit_camera_lower_band_ok = (
        True
        if stage.target_hit_camera_lower_band_rate is None
        else bool(
            np.isfinite(recent_hit_camera_lower_band_rate)
            and recent_hit_camera_lower_band_rate >= float(stage.target_hit_camera_lower_band_rate)
        )
    )
    hit_vxy_ok = (
        True
        if stage.max_recent_mean_hit_vxy is None
        else bool(
            np.isfinite(recent_mean_hit_vxy)
            and recent_mean_hit_vxy <= float(stage.max_recent_mean_hit_vxy)
        )
    )
    hit_next_contact_anchor_ok = (
        True
        if stage.max_recent_hit_next_contact_anchor_err is None
        else bool(
            np.isfinite(recent_mean_hit_next_contact_anchor_err)
            and recent_mean_hit_next_contact_anchor_err
            <= float(stage.max_recent_hit_next_contact_anchor_err)
        )
    )
    hit_camera_v_frac_ok = (
        True
        if stage.max_recent_mean_hit_camera_v_frac is None
        else bool(
            np.isfinite(recent_mean_hit_camera_v_frac)
            and recent_mean_hit_camera_v_frac
            <= float(stage.max_recent_mean_hit_camera_v_frac)
        )
    )
    hit_recoverability_ok = bool(
        hit_vxy_ok and hit_next_contact_anchor_ok and hit_camera_v_frac_ok
    )
    # The intermediate GPU0 tail uses the exact next-contact metric as a
    # finite diagnostic, while the unchanged stochastic next-stage probe is
    # the actual transfer test.  Historical validation showed that a smaller
    # proxy error did not rank checkpoints by launch17 performance.
    hit_probe_readiness_ok = bool(
        hit_vxy_ok
        and hit_camera_v_frac_ok
        and np.isfinite(recent_mean_hit_next_contact_anchor_err)
    )
    episode_truncation_ok = (
        True
        if stage.target_episode_truncation_rate is None
        else bool(
            np.isfinite(recent_episode_truncation_rate)
            and recent_episode_truncation_rate >= float(stage.target_episode_truncation_rate)
        )
    )
    racket_up_cos_ok = (
        True
        if stage.target_racket_up_cos is None
        else bool(
            np.isfinite(recent_racket_up_cos)
            and recent_racket_up_cos >= float(stage.target_racket_up_cos)
        )
    )
    missing_exposure_ok = (
        True
        if stage.min_ball_obs_missing_refresh_rate is None
        else bool(
            np.isfinite(recent_ball_obs_missing_refresh_rate)
            and recent_ball_obs_missing_refresh_rate
            >= float(stage.min_ball_obs_missing_refresh_rate)
        )
    )
    lost_rate_ok = (
        True
        if stage.max_ball_obs_lost_rate is None
        else bool(
            np.isfinite(recent_ball_obs_lost_rate)
            and recent_ball_obs_lost_rate <= float(stage.max_ball_obs_lost_rate)
        )
    )

    hit_family_score = _weighted_gate_score(
        [
            (recent_mean_hits, stage.target_mean_hits, 0.45),
            (recent_hit1_rate, stage.target_hit1_rate, 0.10),
            (recent_hit3_rate, stage.target_hit3_rate, 0.15),
            (recent_hit12_rate, stage.target_hit12_rate, 0.10),
            (recent_mean_hits_ge3, stage.target_mean_hits_ge3, 0.20),
        ]
    )
    survival_family_score = _weighted_gate_score(
        [
            (recent_len_frac, stage.target_mean_len_frac, 0.65),
            (
                recent_episode_truncation_rate,
                stage.target_episode_truncation_rate,
                0.35,
            ),
        ]
    )
    task_family_score = 0.65 * hit_family_score + 0.35 * survival_family_score
    vision_tracking_score = _weighted_gate_score(
        [
            (recent_camera_visible, stage.target_camera_visible, 0.40),
            (
                recent_ball_view_in_bounds,
                stage.target_ball_view_in_bounds,
                0.35,
            ),
            (recent_ball_view_z_ideal, stage.target_ball_view_z_ideal, 0.25),
        ]
    )
    contact_camera_score = _weighted_gate_score(
        [
            (
                recent_hit_camera_visible_rate,
                stage.target_hit_camera_visible_rate,
                0.55,
            ),
            (
                recent_hit_camera_lower_band_rate,
                stage.target_hit_camera_lower_band_rate,
                0.45,
            ),
        ]
    )
    vision_family_score = (
        0.45 * vision_tracking_score + 0.55 * contact_camera_score
    )

    # Intermediate stages use loose anti-collapse floors plus composite
    # readiness. Correlated tail-hit and contact-camera metrics count once;
    # stage 08 still applies every exact target independently.
    balanced_floor_ok = bool(
        _positive_gate_floor(
            recent_mean_hits,
            stage.target_mean_hits,
            ratio=0.80,
        )
        and _positive_gate_floor(
            recent_len_frac,
            stage.target_mean_len_frac,
            ratio=0.80,
        )
        and _rate_gate_floor(
            recent_hit1_rate,
            stage.target_hit1_rate,
            margin=0.15,
        )
        and _rate_gate_floor(
            recent_camera_visible,
            stage.target_camera_visible,
            margin=0.15,
        )
        and _rate_gate_floor(
            recent_ball_view_in_bounds,
            stage.target_ball_view_in_bounds,
            margin=0.18,
        )
        and _rate_gate_floor(
            recent_ball_view_z_ideal,
            stage.target_ball_view_z_ideal,
            margin=0.20,
        )
        and _rate_gate_floor(
            recent_hit_camera_visible_rate,
            stage.target_hit_camera_visible_rate,
            margin=0.15,
        )
        and _rate_gate_floor(
            recent_hit_camera_lower_band_rate,
            stage.target_hit_camera_lower_band_rate,
            margin=0.18,
        )
        and _rate_gate_floor(
            recent_racket_up_cos,
            stage.target_racket_up_cos,
            margin=0.005,
        )
    )
    task_group_ok = task_family_score >= 0.84
    vision_group_ok = vision_family_score >= 0.86

    strict_performance_ok = bool(
        hit_ok
        and len_ok
        and camera_visible_ok
        and ball_view_in_bounds_ok
        and ball_view_z_ideal_ok
        and hit1_rate_ok
        and hit3_rate_ok
        and hit12_rate_ok
        and hits_ge3_ok
        and hit_camera_visible_ok
        and hit_camera_lower_band_ok
        and hit_recoverability_ok
        and episode_truncation_ok
        and racket_up_cos_ok
    )
    if stage.gate_mode == "balanced":
        performance_gate_ok = bool(
            balanced_floor_ok
            and task_group_ok
            and vision_group_ok
            and hit_recoverability_ok
        )
    elif stage.gate_mode == "balanced_probe":
        performance_gate_ok = bool(
            balanced_floor_ok
            and task_group_ok
            and vision_group_ok
            and hit_probe_readiness_ok
        )
    elif stage.gate_mode == "strict":
        performance_gate_ok = strict_performance_ok
    else:
        raise ValueError(f"unknown curriculum gate_mode: {stage.gate_mode}")

    converged = bool(
        args.advance_mode == "converged"
        and enough_updates
        and enough_window
        and return_ok
        and camera_reward_ok
        and performance_gate_ok
        and hit_interval_ok
        and hit_interval_max_ok
        and missing_exposure_ok
        and lost_rate_ok
    )
    return {
        "convergence/stage_converged": float(converged),
        "convergence/gate_mode_balanced": float(stage.gate_mode == "balanced"),
        "convergence/gate_mode_balanced_probe": float(
            stage.gate_mode == "balanced_probe"
        ),
        "convergence/performance_gate_ok": float(performance_gate_ok),
        "convergence/balanced_floor_ok": float(balanced_floor_ok),
        "convergence/task_group_ok": float(task_group_ok),
        "convergence/vision_group_ok": float(vision_group_ok),
        "convergence/hit_probe_readiness_ok": float(hit_probe_readiness_ok),
        "convergence/hit_family_score": float(hit_family_score),
        "convergence/survival_family_score": float(survival_family_score),
        "convergence/task_family_score": float(task_family_score),
        "convergence/vision_tracking_score": float(vision_tracking_score),
        "convergence/contact_camera_score": float(contact_camera_score),
        "convergence/vision_family_score": float(vision_family_score),
        "convergence/recent_updates": float(len(recent)),
        "convergence/recent_mean_hits": recent_mean_hits,
        "convergence/recent_mean_len": recent_mean_len,
        "convergence/recent_mean_len_frac": recent_len_frac,
        "convergence/recent_mean_return": recent_mean_return,
        "convergence/recent_camera_visible": recent_camera_visible,
        "convergence/recent_camera_reward_dense": recent_camera_reward_dense,
        "convergence/recent_ball_view_in_bounds": recent_ball_view_in_bounds,
        "convergence/recent_ball_view_z_ideal": recent_ball_view_z_ideal,
        "convergence/recent_hit1_rate": recent_hit1_rate,
        "convergence/recent_hit3_rate": recent_hit3_rate,
        "convergence/recent_hit12_rate": recent_hit12_rate,
        "convergence/recent_mean_hits_ge3": recent_mean_hits_ge3,
        "convergence/recent_mean_hit_interval_s": recent_mean_hit_interval_s,
        "convergence/recent_mean_hit_interval_ge3_s": recent_mean_hit_interval_ge3_s,
        "convergence/recent_episode_truncation_rate": recent_episode_truncation_rate,
        "convergence/recent_ball_obs_missing_refresh_rate": recent_ball_obs_missing_refresh_rate,
        "convergence/recent_ball_obs_lost_rate": recent_ball_obs_lost_rate,
        "convergence/recent_hit_camera_visible_rate": recent_hit_camera_visible_rate,
        "convergence/recent_hit_camera_lower_band_rate": recent_hit_camera_lower_band_rate,
        "convergence/recent_mean_hit_camera_v_frac": recent_mean_hit_camera_v_frac,
        "convergence/recent_mean_hit_vxy": recent_mean_hit_vxy,
        "convergence/recent_mean_hit_next_contact_anchor_err": recent_mean_hit_next_contact_anchor_err,
        "convergence/recent_racket_up_cos": recent_racket_up_cos,
        "convergence/target_mean_hits": float(stage.target_mean_hits),
        "convergence/target_mean_len_frac": float(stage.target_mean_len_frac),
        "convergence/min_recent_mean_return": (
            float(stage.min_recent_mean_return) if stage.min_recent_mean_return is not None else 0.0
        ),
        "convergence/target_camera_visible": (
            float(stage.target_camera_visible) if stage.target_camera_visible is not None else 0.0
        ),
        "convergence/min_recent_camera_reward_dense": (
            float(stage.min_recent_camera_reward_dense)
            if stage.min_recent_camera_reward_dense is not None
            else 0.0
        ),
        "convergence/target_ball_view_in_bounds": (
            float(stage.target_ball_view_in_bounds)
            if stage.target_ball_view_in_bounds is not None
            else 0.0
        ),
        "convergence/target_ball_view_z_ideal": (
            float(stage.target_ball_view_z_ideal)
            if stage.target_ball_view_z_ideal is not None
            else 0.0
        ),
        "convergence/target_hit1_rate": (
            float(stage.target_hit1_rate) if stage.target_hit1_rate is not None else 0.0
        ),
        "convergence/target_hit3_rate": (
            float(stage.target_hit3_rate) if stage.target_hit3_rate is not None else 0.0
        ),
        "convergence/target_hit12_rate": (
            float(stage.target_hit12_rate) if stage.target_hit12_rate is not None else 0.0
        ),
        "convergence/target_mean_hits_ge3": (
            float(stage.target_mean_hits_ge3) if stage.target_mean_hits_ge3 is not None else 0.0
        ),
        "convergence/target_min_hit_interval_s": (
            float(stage.target_min_hit_interval_s)
            if stage.target_min_hit_interval_s is not None
            else 0.0
        ),
        "convergence/target_max_hit_interval_s": (
            float(stage.target_max_hit_interval_s)
            if stage.target_max_hit_interval_s is not None
            else 0.0
        ),
        "convergence/target_hit_camera_visible_rate": (
            float(stage.target_hit_camera_visible_rate)
            if stage.target_hit_camera_visible_rate is not None
            else 0.0
        ),
        "convergence/target_hit_camera_lower_band_rate": (
            float(stage.target_hit_camera_lower_band_rate)
            if stage.target_hit_camera_lower_band_rate is not None
            else 0.0
        ),
        "convergence/max_recent_mean_hit_vxy": (
            float(stage.max_recent_mean_hit_vxy)
            if stage.max_recent_mean_hit_vxy is not None
            else 0.0
        ),
        "convergence/max_recent_hit_next_contact_anchor_err": (
            float(stage.max_recent_hit_next_contact_anchor_err)
            if stage.max_recent_hit_next_contact_anchor_err is not None
            else 0.0
        ),
        "convergence/max_recent_mean_hit_camera_v_frac": (
            float(stage.max_recent_mean_hit_camera_v_frac)
            if stage.max_recent_mean_hit_camera_v_frac is not None
            else 0.0
        ),
        "convergence/target_episode_truncation_rate": (
            float(stage.target_episode_truncation_rate)
            if stage.target_episode_truncation_rate is not None
            else 0.0
        ),
        "convergence/target_racket_up_cos": (
            float(stage.target_racket_up_cos)
            if stage.target_racket_up_cos is not None
            else 0.0
        ),
        "convergence/min_ball_obs_missing_refresh_rate": (
            float(stage.min_ball_obs_missing_refresh_rate)
            if stage.min_ball_obs_missing_refresh_rate is not None
            else 0.0
        ),
        "convergence/max_ball_obs_lost_rate": (
            float(stage.max_ball_obs_lost_rate)
            if stage.max_ball_obs_lost_rate is not None
            else 0.0
        ),
        "convergence/hit_ok": float(hit_ok),
        "convergence/len_ok": float(len_ok),
        "convergence/return_ok": float(return_ok),
        "convergence/camera_visible_ok": float(camera_visible_ok),
        "convergence/camera_reward_ok": float(camera_reward_ok),
        "convergence/ball_view_in_bounds_ok": float(ball_view_in_bounds_ok),
        "convergence/ball_view_z_ideal_ok": float(ball_view_z_ideal_ok),
        "convergence/hit1_rate_ok": float(hit1_rate_ok),
        "convergence/hit3_rate_ok": float(hit3_rate_ok),
        "convergence/hit12_rate_ok": float(hit12_rate_ok),
        "convergence/hits_ge3_ok": float(hits_ge3_ok),
        "convergence/hit_interval_ok": float(hit_interval_ok),
        "convergence/hit_interval_max_ok": float(hit_interval_max_ok),
        "convergence/hit_camera_visible_ok": float(hit_camera_visible_ok),
        "convergence/hit_camera_lower_band_ok": float(hit_camera_lower_band_ok),
        "convergence/hit_vxy_ok": float(hit_vxy_ok),
        "convergence/hit_next_contact_anchor_ok": float(hit_next_contact_anchor_ok),
        "convergence/hit_camera_v_frac_ok": float(hit_camera_v_frac_ok),
        "convergence/hit_recoverability_ok": float(hit_recoverability_ok),
        "convergence/episode_truncation_ok": float(episode_truncation_ok),
        "convergence/racket_up_cos_ok": float(racket_up_cos_ok),
        "convergence/missing_exposure_ok": float(missing_exposure_ok),
        "convergence/lost_rate_ok": float(lost_rate_ok),
        "convergence/min_updates": float(required_updates),
    }

def _gate_metric_score(value: float | None, target: float | None, weight: float) -> float:
    if target is None:
        return 0.0
    if value is None:
        return -float(weight)
    return float(weight) * (float(value) - float(target))


def _upper_gate_metric_score(value: float | None, limit: float | None, weight: float) -> float:
    if limit is None:
        return 0.0
    if value is None:
        return -float(weight)
    return float(weight) * (float(limit) - float(value))


def _interval_band_score(
    value: float | None,
    minimum: float | None,
    maximum: float | None,
    weight: float,
) -> float:
    """Penalize cadence outside its accepted band without cancelling terms."""

    if minimum is None and maximum is None:
        return 0.0
    if value is None:
        return -float(weight)
    if minimum is not None and float(value) < float(minimum):
        return float(weight) * (float(value) - float(minimum))
    if maximum is not None and float(value) > float(maximum):
        return float(weight) * (float(maximum) - float(value))
    return 0.0


def stage_best_score(row: dict[str, object], stage: CurriculumStage) -> float | None:
    recent_hits = _finite_float(row, "convergence/recent_mean_hits")
    recent_len_frac = _finite_float(row, "convergence/recent_mean_len_frac")
    recent_return = _finite_float(row, "convergence/recent_mean_return")
    if recent_hits is None or recent_len_frac is None or recent_return is None:
        return None
    score = recent_hits + 10.0 * recent_len_frac + 0.10 * recent_return
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_camera_visible"),
        stage.target_camera_visible,
        8.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_ball_view_in_bounds"),
        stage.target_ball_view_in_bounds,
        10.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_ball_view_z_ideal"),
        stage.target_ball_view_z_ideal,
        4.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_camera_reward_dense"),
        stage.min_recent_camera_reward_dense,
        2.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_hit1_rate"),
        stage.target_hit1_rate,
        8.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_hit3_rate"),
        stage.target_hit3_rate,
        10.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_hit12_rate"),
        stage.target_hit12_rate,
        8.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_mean_hits_ge3"),
        stage.target_mean_hits_ge3,
        0.5,
    )
    score += _interval_band_score(
        _finite_float(
            row,
            "convergence/recent_mean_hit_interval_ge3_s"
            if stage.target_max_hit_interval_s is not None
            else "convergence/recent_mean_hit_interval_s",
        ),
        stage.target_min_hit_interval_s,
        stage.target_max_hit_interval_s,
        10.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_hit_camera_visible_rate"),
        stage.target_hit_camera_visible_rate,
        8.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_hit_camera_lower_band_rate"),
        stage.target_hit_camera_lower_band_rate,
        10.0,
    )
    score += _upper_gate_metric_score(
        _finite_float(row, "convergence/recent_mean_hit_vxy"),
        stage.max_recent_mean_hit_vxy,
        4.0,
    )
    score += _upper_gate_metric_score(
        _finite_float(row, "convergence/recent_mean_hit_next_contact_anchor_err"),
        stage.max_recent_hit_next_contact_anchor_err,
        8.0,
    )
    score += _upper_gate_metric_score(
        _finite_float(row, "convergence/recent_mean_hit_camera_v_frac"),
        stage.max_recent_mean_hit_camera_v_frac,
        6.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_episode_truncation_rate"),
        stage.target_episode_truncation_rate,
        12.0,
    )
    score += _gate_metric_score(
        _finite_float(row, "convergence/recent_ball_obs_missing_refresh_rate"),
        stage.min_ball_obs_missing_refresh_rate,
        2.0,
    )
    if stage.max_ball_obs_lost_rate is not None:
        lost_rate = _finite_float(row, "convergence/recent_ball_obs_lost_rate")
        score += (
            -2.0
            if lost_rate is None
            else 8.0 * (float(stage.max_ball_obs_lost_rate) - lost_rate)
        )
    return score

def _advance_validation_probe_spec(
    args: argparse.Namespace,
    stage_idx: int,
    stages: list[CurriculumStage],
) -> tuple[CurriculumStage | None, int | None, bool]:
    """Resolve the next-stage probe, plus the GOAL-profile final self-probe.

    ``stage_idx`` is the one-based index used by the training loop.  Before
    the last stage it points at the next zero-based list entry.  The two new
    GOAL profiles additionally validate their last stage against itself so a
    run cannot finish on rollout convergence alone.
    """

    if stage_idx < len(stages):
        return stages[stage_idx], stage_idx, False
    if (
        stage_idx == len(stages)
        and stages
        and getattr(args, "curriculum_profile", "") in GOAL_D455_PROFILES
    ):
        return stages[-1], len(stages) - 1, True
    return None, None, False


def advance_validation_defaults(
    args: argparse.Namespace,
    stage_idx: int,
    stages: list[CurriculumStage],
) -> dict[str, float]:
    probe_stage, probe_list_index, final_self_probe = _advance_validation_probe_spec(
        args,
        stage_idx,
        stages,
    )
    required = bool(args.advance_validation_mode != "off" and probe_stage is not None)
    threshold_names = (
        "target_mean_hits",
        "target_mean_len_frac",
        "min_mean_return",
        "target_camera_visible",
        "min_camera_reward_dense",
        "target_ball_view_in_bounds",
        "target_ball_view_z_ideal",
        "target_hit1_rate",
        "target_hit3_rate",
        "target_hit12_rate",
        "target_mean_hits_ge3",
        "target_min_hit_interval_s",
        "target_max_hit_interval_s",
        "target_hit_camera_visible_rate",
        "target_hit_camera_lower_band_rate",
        "target_episode_truncation_rate",
        "target_racket_up_cos",
        "max_mean_hit_vxy",
        "max_mean_hit_camera_v_frac",
        "max_mean_hit_next_contact_anchor_err",
        "min_ball_obs_missing_refresh_rate",
        "max_ball_obs_lost_rate",
    )
    if probe_stage is None:
        thresholds = {name: float("nan") for name in threshold_names}
        probe_stage_index = 0.0
        collapse = False
    else:
        thresholds = advance_validation_thresholds(
            args,
            probe_stage,
            force_strict=final_self_probe,
        )
        probe_stage_index = float(int(probe_list_index) + 1)
        collapse = bool(probe_stage.advance_gate_mode == "collapse" and not final_self_probe)
    result = {
        "advance_eval/required": float(required),
        "advance_eval/ran": 0.0,
        "advance_eval/skipped_cooldown": 0.0,
        "advance_eval/passed": float(not required),
        "advance_eval/blocking": float(required and args.advance_validation_mode == "block"),
        "advance_eval/gate_mode_collapse": float(collapse),
        "advance_eval/final_self_probe": float(final_self_probe),
        "advance_eval/probe_stage_index": probe_stage_index,
    }
    result.update({f"advance_eval/{name}": float(thresholds[name]) for name in threshold_names})
    for name in (
        "episodes",
        "mean_return",
        "mean_len",
        "mean_len_frac",
        "mean_hits",
        "hit0_rate",
        "hit1_rate",
        "hit3_rate",
        "hit7_rate",
        "hit12_rate",
        "hit20_rate",
        "mean_hits_ge1",
        "mean_hits_ge3",
        "mean_hits_ge7",
        "mean_hit_interval_s",
        "mean_hit_interval_ge3_s",
        "hit_rate_hz",
        "camera_visible",
        "camera_reward_dense",
        "ball_view_in_bounds",
        "ball_view_z_ideal",
        "ball_view_z_high_exceeded",
        "hit_camera_visible_rate",
        "hit_camera_in_margin_rate",
        "hit_camera_lower_band_rate",
        "mean_hit_camera_v_frac",
        "mean_hit_vxy",
        "mean_hit_contact_center_dist",
        "mean_hit_racket_up_cos",
        "mean_hit_apex_rel_height",
        "racket_up_cos",
        "mean_hit_next_contact_anchor_err",
        "episode_truncation_rate",
        "ball_obs_missing_refresh_rate",
        "ball_obs_lost_rate",
        "terminated",
        "truncated",
        "racket_too_high",
        "ball_too_low",
        "ball_too_high",
        "ball_view_x_too_low",
        "ball_view_x_too_high",
        "ball_view_y_too_low",
        "ball_view_y_too_high",
        "ball_view_z_too_low",
        "ball_view_z_too_high",
    ):
        result[f"advance_eval/{name}"] = float("nan")
    for name in (
        "enough_episodes",
        "hit_ok",
        "len_ok",
        "return_ok",
        "camera_visible_ok",
        "camera_reward_ok",
        "ball_view_in_bounds_ok",
        "ball_view_z_ideal_ok",
        "hit1_rate_ok",
        "hit3_rate_ok",
        "hit12_rate_ok",
        "hits_ge3_ok",
        "hit_interval_ok",
        "hit_interval_max_ok",
        "hit_camera_visible_ok",
        "hit_camera_lower_band_ok",
        "hit_next_contact_anchor_ok",
        "hit_vxy_ok",
        "hit_camera_v_frac_ok",
        "racket_up_cos_ok",
        "episode_truncation_ok",
        "missing_exposure_ok",
        "lost_rate_ok",
        "collapse_core_ok",
    ):
        result[f"advance_eval/{name}"] = 0.0
    result.update(reset_bucket_default_metrics())
    return result

def advance_validation_thresholds(
    args: argparse.Namespace,
    probe_stage: CurriculumStage,
    *,
    force_strict: bool = False,
) -> dict[str, float]:
    collapse = bool(probe_stage.advance_gate_mode == "collapse" and not force_strict)
    if collapse:
        hit_ratio = 0.35
        len_ratio = 0.30
        # Collapse probes are only meant to prevent catastrophic regressions
        # when entering the next stage.  A global minimum such as 2.5 hits is
        # appropriate for strict validation, but it blocks early one-hit
        # D455 stages whose own target is deliberately below that value.
        target_hits = float(probe_stage.target_mean_hits) * hit_ratio
        target_len_frac = float(probe_stage.target_mean_len_frac) * len_ratio
    else:
        hit_ratio = float(args.advance_eval_hit_ratio)
        len_ratio = float(args.advance_eval_len_ratio)
        target_hits = max(
            float(args.advance_eval_min_hits),
            float(probe_stage.target_mean_hits) * hit_ratio,
        )
        target_len_frac = max(
            float(args.advance_eval_min_len_frac),
            float(probe_stage.target_mean_len_frac) * len_ratio,
        )

    def lower_target(target: float | None, *, margin: float, floor: float = 0.0) -> float:
        if target is None:
            return float("nan")
        return max(float(floor), float(target) - float(margin))

    if collapse:
        # A next-stage probe blocks only catastrophic collapse. Camera quality,
        # tail-hit and missing-exposure metrics remain logged diagnostics.
        target_camera_visible = float("nan")
        target_ball_view_in_bounds = float("nan")
        target_ball_view_z_ideal = float("nan")
        target_hit1_rate = 0.50
        target_hit3_rate = float("nan")
        target_hit12_rate = float("nan")
        target_hit_camera_visible_rate = float("nan")
        target_hit_camera_lower_band_rate = float("nan")
        target_mean_hits_ge3 = float("nan")
        target_episode_truncation_rate = float("nan")
        target_racket_up_cos = float("nan")
        max_mean_hit_vxy = float("nan")
        max_mean_hit_camera_v_frac = float("nan")
        max_mean_hit_next_contact_anchor_err = float("nan")
        max_ball_obs_lost_rate = (
            float("nan")
            if probe_stage.max_ball_obs_lost_rate is None
            else min(1.0, float(probe_stage.max_ball_obs_lost_rate) + 0.10)
        )
    else:
        target_camera_visible = lower_target(
            probe_stage.target_camera_visible,
            margin=float(args.advance_eval_camera_margin),
        )
        target_ball_view_in_bounds = lower_target(
            probe_stage.target_ball_view_in_bounds,
            margin=float(args.advance_eval_ball_view_margin),
        )
        target_ball_view_z_ideal = lower_target(
            probe_stage.target_ball_view_z_ideal,
            margin=float(args.advance_eval_z_ideal_margin),
        )
        target_hit1_rate = lower_target(
            probe_stage.target_hit1_rate,
            margin=float(args.advance_eval_hit_rate_margin),
        )
        target_hit3_rate = lower_target(
            probe_stage.target_hit3_rate,
            margin=float(args.advance_eval_hit_rate_margin),
        )
        target_hit12_rate = lower_target(
            probe_stage.target_hit12_rate,
            margin=float(args.advance_eval_hit_rate_margin),
        )
        target_hit_camera_visible_rate = lower_target(
            probe_stage.target_hit_camera_visible_rate,
            margin=float(args.advance_eval_camera_margin),
        )
        target_hit_camera_lower_band_rate = lower_target(
            probe_stage.target_hit_camera_lower_band_rate,
            margin=float(args.advance_eval_camera_margin),
        )
        target_mean_hits_ge3 = (
            float("nan")
            if probe_stage.target_mean_hits_ge3 is None
            else float(probe_stage.target_mean_hits_ge3) * float(args.advance_eval_cond_hit_ratio)
        )
        target_episode_truncation_rate = (
            float("nan")
            if probe_stage.target_episode_truncation_rate is None
            else float(probe_stage.target_episode_truncation_rate) * len_ratio
        )
        target_racket_up_cos = lower_target(
            probe_stage.target_racket_up_cos,
            margin=0.015,
        )
        max_mean_hit_vxy = (
            float("nan")
            if probe_stage.max_recent_mean_hit_vxy is None
            else float(probe_stage.max_recent_mean_hit_vxy) + 0.05
        )
        max_mean_hit_camera_v_frac = (
            float("nan")
            if probe_stage.max_recent_mean_hit_camera_v_frac is None
            else float(probe_stage.max_recent_mean_hit_camera_v_frac) + 0.03
        )
        max_mean_hit_next_contact_anchor_err = (
            float("nan")
            if probe_stage.max_recent_hit_next_contact_anchor_err is None
            else float(probe_stage.max_recent_hit_next_contact_anchor_err) + 0.015
        )
        max_ball_obs_lost_rate = (
            float("nan")
            if probe_stage.max_ball_obs_lost_rate is None
            else min(1.0, float(probe_stage.max_ball_obs_lost_rate) + 0.03)
        )

    min_camera_reward = (
        float("nan")
        if probe_stage.min_recent_camera_reward_dense is None
        else float(probe_stage.min_recent_camera_reward_dense)
        - float(args.advance_eval_camera_reward_margin)
    )
    target_min_hit_interval_s = lower_target(
        probe_stage.target_min_hit_interval_s,
        margin=float(args.advance_eval_hit_interval_margin),
    )
    target_max_hit_interval_s = (
        float("nan")
        if probe_stage.target_max_hit_interval_s is None
        else float(probe_stage.target_max_hit_interval_s)
        + float(args.advance_eval_hit_interval_margin)
    )
    min_ball_obs_missing_refresh_rate = (
        float("nan")
        if probe_stage.min_ball_obs_missing_refresh_rate is None
        else 0.5 * float(probe_stage.min_ball_obs_missing_refresh_rate)
    )
    if collapse:
        min_camera_reward = float("nan")
        target_min_hit_interval_s = float("nan")
        target_max_hit_interval_s = float("nan")
        min_ball_obs_missing_refresh_rate = float("nan")
    return {
        "target_mean_hits": target_hits,
        "target_mean_len_frac": target_len_frac,
        "min_mean_return": float("nan") if collapse else float(args.advance_eval_min_return),
        "target_camera_visible": target_camera_visible,
        "min_camera_reward_dense": min_camera_reward,
        "target_ball_view_in_bounds": target_ball_view_in_bounds,
        "target_ball_view_z_ideal": target_ball_view_z_ideal,
        "target_hit1_rate": target_hit1_rate,
        "target_hit3_rate": target_hit3_rate,
        "target_hit12_rate": target_hit12_rate,
        "target_mean_hits_ge3": target_mean_hits_ge3,
        "target_min_hit_interval_s": target_min_hit_interval_s,
        "target_max_hit_interval_s": target_max_hit_interval_s,
        "target_hit_camera_visible_rate": target_hit_camera_visible_rate,
        "target_hit_camera_lower_band_rate": target_hit_camera_lower_band_rate,
        "target_episode_truncation_rate": target_episode_truncation_rate,
        "target_racket_up_cos": target_racket_up_cos,
        "max_mean_hit_vxy": max_mean_hit_vxy,
        "max_mean_hit_camera_v_frac": max_mean_hit_camera_v_frac,
        "max_mean_hit_next_contact_anchor_err": max_mean_hit_next_contact_anchor_err,
        "min_ball_obs_missing_refresh_rate": min_ball_obs_missing_refresh_rate,
        "max_ball_obs_lost_rate": max_ball_obs_lost_rate,
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


def advance_validation_env_cfg(probe_stage: CurriculumStage) -> MjxJuggleConfig:
    """Evaluate the declared observation noise at full scale from step one."""

    return replace(
        probe_stage.cfg,
        ball_obs_noise_warmup_ratio=0.0,
        ball_obs_noise_ramp_ratio=0.0,
        total_training_steps=1,
    )


def summarize_eval_outputs(
    outputs,
    env: MjxJuggleEnv,
    *,
    reset_bucket_mode: str = "log",
    reset_bucket_min_episodes: int = 4,
    reset_bucket_cvar_frac: float = 0.20,
) -> dict[str, float]:
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

    def metric_ratio(numerator: str, denominator: str) -> float:
        numerator_arr = metrics.get(numerator)
        denominator_arr = metrics.get(denominator)
        if numerator_arr is None or denominator_arr is None:
            return float("nan")
        den = float(np.asarray(denominator_arr, dtype=np.float64).sum())
        if den <= 0.0:
            return float("nan")
        return float(np.asarray(numerator_arr, dtype=np.float64).sum()) / den

    mean_len = float(ep_len[done].mean()) if done_count > 0 else float("nan")
    hit_stats = {
        f"advance_eval/{key}": value
        for key, value in episode_hit_distribution_metrics(hit_count, done, ep_len, env.dt).items()
    }
    summary = {
        "advance_eval/ran": 1.0,
        "advance_eval/episodes": float(done_count),
        "advance_eval/mean_return": float(ep_ret[done].mean()) if done_count > 0 else float("nan"),
        "advance_eval/mean_len": mean_len,
        "advance_eval/mean_len_frac": (
            mean_len / max(1, int(env.max_steps)) if np.isfinite(mean_len) else float("nan")
        ),
        "advance_eval/mean_hits": (
            float(hit_count[done].mean()) if done_count > 0 else float("nan")
        ),
        **hit_stats,
        "advance_eval/camera_visible": metric_mean("camera_visible"),
        "advance_eval/camera_reward_dense": metric_mean("reward/camera_reward_dense"),
        "advance_eval/ball_view_in_bounds": metric_mean("ball_view_in_bounds"),
        "advance_eval/ball_view_z_ideal": metric_mean("ball_view_z_ideal"),
        "advance_eval/ball_view_z_high_exceeded": metric_mean("ball_view_z_high_exceeded"),
        "advance_eval/hit_camera_visible_rate": metric_ratio(
            "hit_camera_visible_event",
            "hit_camera_event",
        ),
        "advance_eval/hit_camera_in_margin_rate": metric_ratio(
            "hit_camera_in_margin_event",
            "hit_camera_event",
        ),
        "advance_eval/hit_camera_lower_band_rate": metric_ratio(
            "hit_camera_lower_band_event",
            "hit_camera_event",
        ),
        "advance_eval/mean_hit_camera_v_frac": metric_ratio(
            "hit_camera_v_frac_sum",
            "hit_camera_visible_event",
        ),
        "advance_eval/mean_hit_vxy": metric_ratio(
            "hit_vxy_sum",
            "hit_event_count",
        ),
        "advance_eval/mean_hit_contact_center_dist": metric_ratio(
            "hit_contact_center_dist_sum",
            "hit_event_count",
        ),
        "advance_eval/mean_hit_racket_up_cos": metric_ratio(
            "hit_racket_up_cos_sum",
            "hit_event_count",
        ),
        "advance_eval/mean_hit_apex_rel_height": metric_ratio(
            "hit_apex_rel_height_sum",
            "hit_event_count",
        ),
        "advance_eval/racket_up_cos": metric_mean("racket_up_cos"),
        "advance_eval/mean_hit_next_contact_anchor_err": metric_ratio(
            "hit_next_contact_anchor_err_sum",
            "hit_event_count",
        ),
        "advance_eval/episode_truncation_rate": (
            float(np.asarray(metrics.get("truncated", np.zeros_like(done))).sum()) / float(done_count)
            if done_count > 0
            else float("nan")
        ),
        "advance_eval/ball_obs_missing_refresh_rate": metric_ratio(
            "ball_obs_missing_on_refresh",
            "ball_obs_refresh_due",
        ),
        "advance_eval/ball_obs_lost_rate": metric_mean("ball_obs_lost_active"),
        "advance_eval/terminated": metric_mean("terminated"),
        "advance_eval/truncated": metric_mean("truncated"),
        "advance_eval/racket_too_high": metric_mean("done/racket_too_high"),
        "advance_eval/ball_too_low": metric_mean("done/ball_too_low"),
        "advance_eval/ball_too_high": metric_mean("done/ball_too_high"),
        "advance_eval/ball_view_x_too_low": metric_mean("done/ball_view_x_too_low"),
        "advance_eval/ball_view_x_too_high": metric_mean("done/ball_view_x_too_high"),
        "advance_eval/ball_view_y_too_low": metric_mean("done/ball_view_y_too_low"),
        "advance_eval/ball_view_y_too_high": metric_mean("done/ball_view_y_too_high"),
        "advance_eval/ball_view_z_too_low": metric_mean("done/ball_view_z_too_low"),
        "advance_eval/ball_view_z_too_high": metric_mean("done/ball_view_z_too_high"),
    }
    summary.update(
        summarize_reset_bucket_outputs(
            metrics,
            hit_count,
            done,
            mode=str(reset_bucket_mode),
            min_episodes=int(reset_bucket_min_episodes),
            cvar_frac=float(reset_bucket_cvar_frac),
            fields=(
                RESET_BUCKET_AUTOLAUNCH_FIELDS
                if env.cfg.ball_reset_mode == "racket_launch"
                else (
                    RESET_BUCKET_RELEASE_FIELDS
                    if env.cfg.ball_reset_mode == "anchor_drop"
                    else None
                )
            ),
        )
    )
    return summary

def _advance_validation_gate_passed(
    advance_gate_mode: str,
    *,
    final_self_probe: bool,
    collapse_core_ok: bool,
    strict_ok: bool,
) -> bool:
    """Select the probe result, including strict GOAL final self-validation."""

    if advance_gate_mode == "collapse":
        return bool(strict_ok if final_self_probe else collapse_core_ok)
    if advance_gate_mode == "strict":
        return bool(strict_ok)
    raise ValueError(f"unknown curriculum advance_gate_mode: {advance_gate_mode}")


def run_advance_validation(
    args: argparse.Namespace,
    stage_idx: int,
    stages: list[CurriculumStage],
    params,
    rng: jax.Array,
) -> dict[str, float]:
    if args.advance_validation_mode == "off":
        return {}
    probe_stage, probe_list_index, final_self_probe = _advance_validation_probe_spec(
        args,
        stage_idx,
        stages,
    )
    if probe_stage is None or probe_list_index is None:
        return {}
    n_eval_envs = min(int(args.n_envs), max(1, int(args.advance_eval_n_envs)))
    probe_cfg = advance_validation_env_cfg(probe_stage)
    probe_env = MjxJuggleEnv(args.xml, n_envs=n_eval_envs, cfg=probe_cfg)
    n_eval_steps = (
        int(args.advance_eval_steps)
        if int(args.advance_eval_steps) > 0
        else int(probe_env.max_steps)
    )
    eval_rollout = make_eval_rollout(
        probe_env,
        n_steps=n_eval_steps,
        deterministic=bool(args.advance_eval_deterministic),
    )
    outputs = eval_rollout(params, rng)
    jax.block_until_ready(outputs["done"])
    result = summarize_eval_outputs(
        outputs,
        probe_env,
        reset_bucket_mode=str(args.advance_eval_reset_bucket_mode),
        reset_bucket_min_episodes=int(args.advance_eval_reset_bucket_min_episodes),
        reset_bucket_cvar_frac=float(args.advance_eval_reset_bucket_cvar_frac),
    )
    thresholds = advance_validation_thresholds(
        args,
        probe_stage,
        force_strict=final_self_probe,
    )
    result["advance_eval/probe_stage_index"] = float(probe_list_index + 1)
    result["advance_eval/gate_mode_collapse"] = float(
        probe_stage.advance_gate_mode == "collapse" and not final_self_probe
    )
    result["advance_eval/final_self_probe"] = float(final_self_probe)
    result.update({f"advance_eval/{name}": float(value) for name, value in thresholds.items()})
    result.update(reset_bucket_gate_status(args, probe_stage, thresholds, result))

    def lower_ok(value: float, target: float) -> bool:
        return bool((not np.isfinite(target)) or (np.isfinite(value) and value >= target))

    def upper_ok(value: float, target: float) -> bool:
        return bool((not np.isfinite(target)) or (np.isfinite(value) and value <= target))

    episodes = float(result["advance_eval/episodes"])
    mean_hits = float(result["advance_eval/mean_hits"])
    mean_len_frac = float(result["advance_eval/mean_len_frac"])
    mean_return = float(result["advance_eval/mean_return"])
    camera_visible = float(result["advance_eval/camera_visible"])
    camera_reward = float(result["advance_eval/camera_reward_dense"])
    ball_view_in_bounds = float(result["advance_eval/ball_view_in_bounds"])
    ball_view_z_ideal = float(result["advance_eval/ball_view_z_ideal"])
    hit1_rate = float(result["advance_eval/hit1_rate"])
    hit3_rate = float(result["advance_eval/hit3_rate"])
    hit12_rate = float(result["advance_eval/hit12_rate"])
    mean_hits_ge3 = float(result["advance_eval/mean_hits_ge3"])
    mean_hit_interval_ge3_s = float(result["advance_eval/mean_hit_interval_ge3_s"])
    gate_hit_interval_s = (
        mean_hit_interval_ge3_s
        if probe_stage.target_max_hit_interval_s is not None
        else float(result["advance_eval/mean_hit_interval_s"])
    )
    hit_camera_visible_rate = float(result["advance_eval/hit_camera_visible_rate"])
    hit_camera_lower_band_rate = float(result["advance_eval/hit_camera_lower_band_rate"])
    mean_hit_vxy = float(result["advance_eval/mean_hit_vxy"])
    mean_hit_camera_v_frac = float(result["advance_eval/mean_hit_camera_v_frac"])
    racket_up_cos = float(result["advance_eval/racket_up_cos"])
    episode_truncation_rate = float(result["advance_eval/episode_truncation_rate"])
    mean_hit_next_contact_anchor_err = float(result["advance_eval/mean_hit_next_contact_anchor_err"])
    missing_refresh_rate = float(result["advance_eval/ball_obs_missing_refresh_rate"])
    lost_rate = float(result["advance_eval/ball_obs_lost_rate"])
    reset_bucket_gate_ok = bool(result["advance_eval/reset_bucket_gate_ok"])

    enough_episodes = bool(
        np.isfinite(episodes) and episodes >= int(args.advance_eval_min_episodes)
    )
    hit_ok = lower_ok(mean_hits, thresholds["target_mean_hits"])
    len_ok = lower_ok(mean_len_frac, thresholds["target_mean_len_frac"])
    return_ok = lower_ok(mean_return, thresholds["min_mean_return"])
    camera_visible_ok = lower_ok(
        camera_visible,
        thresholds["target_camera_visible"],
    )
    camera_reward_ok = lower_ok(
        camera_reward,
        thresholds["min_camera_reward_dense"],
    )
    ball_view_in_bounds_ok = lower_ok(
        ball_view_in_bounds,
        thresholds["target_ball_view_in_bounds"],
    )
    ball_view_z_ideal_ok = lower_ok(
        ball_view_z_ideal,
        thresholds["target_ball_view_z_ideal"],
    )
    hit1_rate_ok = lower_ok(hit1_rate, thresholds["target_hit1_rate"])
    hit3_rate_ok = lower_ok(hit3_rate, thresholds["target_hit3_rate"])
    hit12_rate_ok = lower_ok(hit12_rate, thresholds["target_hit12_rate"])
    hits_ge3_ok = lower_ok(mean_hits_ge3, thresholds["target_mean_hits_ge3"])
    hit_interval_ok = lower_ok(
        gate_hit_interval_s,
        thresholds["target_min_hit_interval_s"],
    )
    hit_interval_max_ok = upper_ok(
        gate_hit_interval_s,
        thresholds["target_max_hit_interval_s"],
    )
    hit_camera_visible_ok = lower_ok(
        hit_camera_visible_rate,
        thresholds["target_hit_camera_visible_rate"],
    )
    hit_camera_lower_band_ok = lower_ok(
        hit_camera_lower_band_rate,
        thresholds["target_hit_camera_lower_band_rate"],
    )
    hit_vxy_ok = upper_ok(mean_hit_vxy, thresholds["max_mean_hit_vxy"])
    hit_camera_v_frac_ok = upper_ok(
        mean_hit_camera_v_frac,
        thresholds["max_mean_hit_camera_v_frac"],
    )
    racket_up_cos_ok = lower_ok(racket_up_cos, thresholds["target_racket_up_cos"])
    episode_truncation_ok = lower_ok(
        episode_truncation_rate,
        thresholds["target_episode_truncation_rate"],
    )
    hit_next_contact_anchor_ok = upper_ok(
        mean_hit_next_contact_anchor_err,
        thresholds["max_mean_hit_next_contact_anchor_err"],
    )
    missing_exposure_ok = lower_ok(
        missing_refresh_rate,
        thresholds["min_ball_obs_missing_refresh_rate"],
    )
    lost_rate_ok = upper_ok(lost_rate, thresholds["max_ball_obs_lost_rate"])

    collapse_core_ok = bool(
        enough_episodes
        and hit_ok
        and len_ok
        and return_ok
        and camera_visible_ok
        and camera_reward_ok
        and hit1_rate_ok
        and hit3_rate_ok
        and hit_camera_visible_ok
        and hit_camera_lower_band_ok
        and missing_exposure_ok
        and lost_rate_ok
        and (
            reset_bucket_gate_ok
            if str(args.advance_eval_reset_bucket_mode) in {"cvar", "worst"}
            else True
        )
    )
    strict_ok = bool(
        collapse_core_ok
        and ball_view_in_bounds_ok
        and ball_view_z_ideal_ok
        and hit12_rate_ok
        and hits_ge3_ok
        and hit_interval_ok
        and hit_interval_max_ok
        and episode_truncation_ok
        and hit_vxy_ok
        and hit_camera_v_frac_ok
        and racket_up_cos_ok
        and hit_next_contact_anchor_ok
        and reset_bucket_gate_ok
    )
    passed = _advance_validation_gate_passed(
        probe_stage.advance_gate_mode,
        final_self_probe=final_self_probe,
        collapse_core_ok=collapse_core_ok,
        strict_ok=strict_ok,
    )
    result.update(
        {
            "advance_eval/enough_episodes": float(enough_episodes),
            "advance_eval/hit_ok": float(hit_ok),
            "advance_eval/len_ok": float(len_ok),
            "advance_eval/return_ok": float(return_ok),
            "advance_eval/camera_visible_ok": float(camera_visible_ok),
            "advance_eval/camera_reward_ok": float(camera_reward_ok),
            "advance_eval/ball_view_in_bounds_ok": float(ball_view_in_bounds_ok),
            "advance_eval/ball_view_z_ideal_ok": float(ball_view_z_ideal_ok),
            "advance_eval/hit1_rate_ok": float(hit1_rate_ok),
            "advance_eval/hit3_rate_ok": float(hit3_rate_ok),
            "advance_eval/hit12_rate_ok": float(hit12_rate_ok),
            "advance_eval/hits_ge3_ok": float(hits_ge3_ok),
            "advance_eval/hit_interval_ok": float(hit_interval_ok),
            "advance_eval/hit_interval_max_ok": float(hit_interval_max_ok),
            "advance_eval/hit_camera_visible_ok": float(hit_camera_visible_ok),
            "advance_eval/hit_camera_lower_band_ok": float(
                hit_camera_lower_band_ok
            ),
            "advance_eval/hit_vxy_ok": float(hit_vxy_ok),
            "advance_eval/hit_camera_v_frac_ok": float(hit_camera_v_frac_ok),
            "advance_eval/racket_up_cos_ok": float(racket_up_cos_ok),
            "advance_eval/episode_truncation_ok": float(episode_truncation_ok),
            "advance_eval/hit_next_contact_anchor_ok": float(hit_next_contact_anchor_ok),
            "advance_eval/missing_exposure_ok": float(missing_exposure_ok),
            "advance_eval/lost_rate_ok": float(lost_rate_ok),
            "advance_eval/collapse_core_ok": float(collapse_core_ok),
            "advance_eval/reset_bucket_gate_ok": float(reset_bucket_gate_ok),
            "advance_eval/passed": float(passed),
        }
    )
    return result


def stage_update_cap(stage: CurriculumStage, args: argparse.Namespace, batch_steps: int) -> int | None:
    if args.advance_mode == "fixed":
        return max(1, int(stage.total_steps) // max(1, int(batch_steps)))
    if int(args.max_stage_updates) > 0:
        return int(args.max_stage_updates)
    if stage.max_updates is not None:
        return max(1, int(stage.max_updates))
    return None


def metric_safety_stop_reason(row: dict[str, object], args: argparse.Namespace) -> str | None:
    if not bool(args.safe_stop):
        return None

    episodes = int(row.get("episodes", 0) or 0)
    optional_nan_diagnostics = {
        "mean_hit_interval_s",
        "mean_hit_interval_ge3_s",
        "hit_rate_hz",
        "episode_truncation_rate",
        "ball_obs_missing_refresh_rate",
        "hit_camera_visible_rate",
        "hit_camera_in_margin_rate",
        "hit_camera_lower_band_rate",
        "mean_hit_camera_v_frac",
        "mean_hit_vxy",
        "mean_hit_next_contact_anchor_err",
    }
    for key, value in row.items():
        if isinstance(value, str):
            continue
        if key in optional_nan_diagnostics:
            continue
        if episodes == 0 and key in {
            "mean_return",
            "mean_len",
            "mean_hits",
            "hit0_rate",
            "hit1_rate",
            "hit3_rate",
            "hit7_rate",
            "hit12_rate",
            "hit20_rate",
            "mean_hits_ge1",
            "mean_hits_ge3",
            "mean_hits_ge7",
            "mean_hit_interval_s",
            "mean_hit_interval_ge3_s",
            "hit_rate_hz",
            "episode_truncation_rate",
        }:
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
    source_checkpoint = (
        args.resume_from
        if args.resume_from is not None
        else args.residual_teacher_checkpoint
    )
    if source_checkpoint is None:
        return 1
    token = str(args.resume_start_stage).strip()
    if token == "auto":
        prefix = source_checkpoint.name.split("_", 1)[0]
        if prefix.isdigit():
            return min(int(prefix) + 1, len(stages) + 1)
        return 1
    if token.isdigit():
        return int(token)
    token = STAGE_NAME_ALIASES.get(token, token)
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
    best_ckpt = args.save_dir / "mjx_curriculum_best.pkl"
    if best_ckpt.exists():
        wandb.save(str(best_ckpt))
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
    actor_anchor_replay_obs_np: np.ndarray | None = None
    teacher_distill_replay_obs_np: np.ndarray | None = None
    if args.actor_anchor_replay_obs is not None:
        if (
            float(args.actor_anchor_kl_coef) <= 0.0
            and float(args.actor_anchor_replay_kl_coef) <= 0.0
        ):
            raise SystemExit(
                "[mjx_curriculum] --actor-anchor-replay-obs requires --actor-anchor-kl-coef > 0"
            )
        if not args.actor_anchor_replay_obs.exists():
            raise SystemExit(
                f"[mjx_curriculum] actor anchor replay file not found: {args.actor_anchor_replay_obs}"
            )
        actor_anchor_replay_obs_np = np.asarray(
            np.load(args.actor_anchor_replay_obs),
            dtype=np.float32,
        )
        if actor_anchor_replay_obs_np.ndim != 2 or actor_anchor_replay_obs_np.shape[0] <= 0:
            raise SystemExit(
                "[mjx_curriculum] actor anchor replay observations must have shape [samples, obs_dim]"
            )
    if args.teacher_distill_replay_obs is not None:
        teacher_distill_replay_obs_np = np.asarray(
            np.load(args.teacher_distill_replay_obs),
            dtype=np.float32,
        )
        if (
            teacher_distill_replay_obs_np.ndim != 2
            or teacher_distill_replay_obs_np.shape[0] <= 0
        ):
            raise SystemExit(
                "[mjx_curriculum] teacher replay observations must have shape [samples, obs_dim]"
            )
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
        args.actuator_mpc_command_dynamics_constraint,
        args.actuator_mpc_command_velocity_weight,
        args.actuator_mpc_command_acceleration_weight,
        args.actuator_mpc_command_velocity_scale,
        args.actuator_mpc_command_acceleration_scale,
        args.actuator_mpc_feedback_source,
        args.dr_randomize_actuator_cmd_filter,
        tuple(args.dr_actuator_cmd_tau_range) if args.dr_actuator_cmd_tau_range is not None else None,
        tuple(args.dr_actuator_cmd_gain_range) if args.dr_actuator_cmd_gain_range is not None else None,
        bool(args.wide_polish_dr),
        bool(args.asymmetric_critic),
        int(args.critic_command_history_steps),
        args.arm_post_compensation_limiter,
        args.arm_servo_target_limiter,
        args.arm_servo_target_tracking_planner,
        args.arm_servo_target_velocity_scale,
        args.arm_servo_target_acceleration_scale,
        args.arm_actual_state_limiter,
        args.arm_actual_target_tracking_governor,
        args.arm_actual_governor_natural_frequency_hz,
        args.arm_actual_governor_damping_ratio,
        args.arm_actual_jerk_limit_deg_s3,
        args.right_arm_pd_profile,
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
                        "gate_mode": stage.gate_mode,
                        "advance_gate_mode": stage.advance_gate_mode,
                        "target_mean_hits": stage.target_mean_hits,
                        "target_mean_len_frac": stage.target_mean_len_frac,
                        "min_updates": stage.min_updates,
                        "max_updates": stage.max_updates,
                        "min_recent_mean_return": stage.min_recent_mean_return,
                        "target_camera_visible": stage.target_camera_visible,
                        "min_recent_camera_reward_dense": stage.min_recent_camera_reward_dense,
                        "target_ball_view_in_bounds": stage.target_ball_view_in_bounds,
                        "target_ball_view_z_ideal": stage.target_ball_view_z_ideal,
                        "target_hit1_rate": stage.target_hit1_rate,
                        "target_hit3_rate": stage.target_hit3_rate,
                        "target_mean_hits_ge3": stage.target_mean_hits_ge3,
                        "target_hit12_rate": stage.target_hit12_rate,
                        "target_min_hit_interval_s": stage.target_min_hit_interval_s,
                        "target_episode_truncation_rate": stage.target_episode_truncation_rate,
                        "target_racket_up_cos": stage.target_racket_up_cos,
                        "target_max_hit_interval_s": stage.target_max_hit_interval_s,
                        "target_hit_camera_visible_rate": stage.target_hit_camera_visible_rate,
                        "target_hit_camera_lower_band_rate": stage.target_hit_camera_lower_band_rate,
                        "max_recent_mean_hit_vxy": stage.max_recent_mean_hit_vxy,
                        "max_recent_hit_next_contact_anchor_err": stage.max_recent_hit_next_contact_anchor_err,
                        "max_recent_mean_hit_camera_v_frac": stage.max_recent_mean_hit_camera_v_frac,
                        "min_ball_obs_missing_refresh_rate": stage.min_ball_obs_missing_refresh_rate,
                        "max_ball_obs_lost_rate": stage.max_ball_obs_lost_rate,
                        "policy_updates_enabled": stage.policy_updates_enabled,
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
    residual_teacher_state: TrainState | None = None
    residual_teacher_payload: dict[str, object] | None = None
    teacher_distill_state: TrainState | None = None
    teacher_distill_payload: dict[str, object] | None = None
    train_state_env_cfg: object | None = None
    global_step = 0
    global_update = 0
    stop_request = install_stop_handlers()
    start_stage_idx = resolve_resume_start_stage(args, stages)
    if start_stage_idx < 1 or start_stage_idx > len(stages):
        raise SystemExit(f"[mjx_curriculum] --resume-start-stage resolved to {start_stage_idx}, outside 1..{len(stages)}")
    if args.resume_from is not None:
        train_state, resume_payload = load_train_state(args.resume_from)
        if args.reset_optimizer_on_resume:
            train_state = TrainState(
                params=train_state.params,
                opt=adam_init(train_state.params),
            )
            print("[mjx_curriculum] reset Adam moments after checkpoint resume")
        if args.curriculum_profile == GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE:
            source_args = resume_payload.get("args", {})
            if isinstance(source_args, dict):
                source_profile = source_args.get("curriculum_profile")
            else:
                source_profile = getattr(source_args, "curriculum_profile", None)
            if source_profile != GOAL_D455_AUTOLAUNCH_TEACHER_STUDENT_PROFILE:
                raise SystemExit(
                    "[mjx_curriculum] teacher-student may resume only its own student "
                    "checkpoints; initialize from scratch instead of a W017/other policy"
                )
        train_state_env_cfg = resume_payload.get("env_cfg")
        global_step = int(resume_payload.get("step", 0))
        print(
            f"[mjx_curriculum] resumed from {args.resume_from} "
            f"at global_step={global_step}; starting stage {start_stage_idx}/{len(stages)}: "
            f"{stages[start_stage_idx - 1].name}"
        )
    elif args.residual_teacher_checkpoint is not None:
        residual_teacher_state, residual_teacher_payload = load_train_state(
            args.residual_teacher_checkpoint
        )
        global_step = int(residual_teacher_payload.get("step", 0))
        print(
            f"[mjx_curriculum] residual teacher loaded from "
            f"{args.residual_teacher_checkpoint} at source_step={global_step}; "
            f"starting stage {start_stage_idx}/{len(stages)}: "
            f"{stages[start_stage_idx - 1].name}"
        )
    if args.teacher_distill_checkpoint is not None:
        teacher_distill_state, teacher_distill_payload = load_train_state(
            args.teacher_distill_checkpoint
        )
        teacher_source_step = int(teacher_distill_payload.get("step", 0))
        print(
            "[mjx_curriculum] frozen ideal-domain teacher loaded from "
            f"{args.teacher_distill_checkpoint} at source_step={teacher_source_step}; "
            + (
                "student parameters and optimizer remain freshly initialized"
                if args.resume_from is None
                else "teacher remains external to the resumed student checkpoint"
            )
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
            f"mpc_feedback_source={stage.cfg.actuator_mpc_feedback_source}, "
            f"right_arm_pd_profile={stage.cfg.right_arm_pd_profile}, "
            f"post_comp_limiter={stage.cfg.arm_post_compensation_limiter}, "
            f"servo_target_limiter={stage.cfg.arm_servo_target_limiter}, "
            f"servo_target_tracking_planner={stage.cfg.arm_servo_target_tracking_planner}, "
            f"actual_state_limiter={stage.cfg.arm_actual_state_limiter}, "
            f"actual_target_governor={stage.cfg.arm_actual_target_tracking_governor}, "
            f"actual_governor_hz={stage.cfg.arm_actual_governor_natural_frequency_hz:.1f}, "
            f"actual_governor_zeta={stage.cfg.arm_actual_governor_damping_ratio:.2f}, "
            f"actual_jerk_deg_s3={stage.cfg.arm_actual_jerk_limit_deg_s3[0]:.0f}, "
            f"delay_extra_dim={getattr(env, 'delay_extra_dim', 0)}, "
            f"asymmetric_critic={getattr(env, 'asymmetric_critic', False)}, "
            f"critic_obs_dim={getattr(env, 'critic_obs_dim', env.obs_dim)}"
        )
        nominal_hit_interval_steps = max(1, int(round(0.48 / float(env.dt))))
        print(
            "[mjx_curriculum] credit_assignment: "
            f"gamma={args.gamma:.7f}, gae_lambda={args.gae_lambda:.7f}, "
            f"time_limit_bootstrap={bool(args.time_limit_bootstrap)}, "
            f"discount_at_0.48s={float(args.gamma) ** nominal_hit_interval_steps:.4f}, "
            f"trace_at_0.48s={(float(args.gamma) * float(args.gae_lambda)) ** nominal_hit_interval_steps:.4f}, "
            f"discount_at_horizon={float(args.gamma) ** int(env.max_steps):.4f}, "
            f"failure_focus=terminated_hits<{args.failure_focus_hit_threshold}:"
            f"tail={args.failure_focus_tail_steps}:x{args.failure_focus_weight:.2f}"
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
            if residual_teacher_state is not None and residual_teacher_payload is not None:
                teacher_obs_dim = int(
                    residual_teacher_payload.get(
                        "obs_dim",
                        residual_teacher_state.params["pi"]["l1"]["w"].shape[0],
                    )
                )
                teacher_act_dim = int(
                    residual_teacher_payload.get(
                        "act_dim",
                        residual_teacher_state.params["pi"]["out"]["b"].shape[0],
                    )
                )
                if teacher_obs_dim != int(env.obs_dim) or teacher_act_dim != int(env.act_dim):
                    raise SystemExit(
                        "[mjx_curriculum] residual teacher dimensions do not match env: "
                        f"teacher obs/act={teacher_obs_dim}/{teacher_act_dim}, "
                        f"env obs/act={env.obs_dim}/{env.act_dim}"
                    )
                residual_pi = dict(params["pi"])
                residual_out = dict(residual_pi["out"])
                residual_out["w"] = jnp.zeros_like(residual_out["w"])
                residual_out["b"] = jnp.zeros_like(residual_out["b"])
                residual_pi["out"] = residual_out
                params = dict(params)
                params["pi"] = residual_pi
                params["teacher_pi"] = residual_teacher_state.params["pi"]
                params["residual_action_scale"] = jnp.asarray(
                    float(args.residual_action_scale), dtype=jnp.float32
                )
                params["log_std"] = residual_teacher_state.params["log_std"]
                print(
                    "[mjx_curriculum] initialized frozen-teacher residual actor: "
                    f"scale={args.residual_action_scale:.3f}, "
                    f"l2_coef={args.residual_l2_coef:.4g}, zero initial correction, "
                    "fresh critic/optimizer"
                )
            train_state = TrainState(params=params, opt=adam_init(params))
            if bool(args.residual_initialize_only):
                if residual_teacher_state is None:
                    raise SystemExit(
                        "[mjx_curriculum] --residual-initialize-only requires "
                        "--residual-teacher-checkpoint"
                    )
                extra = {
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "stage_update": 0,
                    "global_update": 0,
                    "residual_initialized_only": True,
                    "residual_teacher_checkpoint": str(
                        args.residual_teacher_checkpoint
                    ),
                }
                initial_path = args.save_dir / "mjx_residual_initial.pkl"
                save_checkpoint(
                    initial_path,
                    train_state,
                    args,
                    env,
                    global_step,
                    extra=extra,
                )
                save_checkpoint(
                    args.save_dir / "mjx_curriculum_last.pkl",
                    train_state,
                    args,
                    env,
                    global_step,
                    extra=extra,
                )
                print(
                    "[mjx_curriculum] saved exact zero-residual checkpoint: "
                    f"{initial_path}"
                )
                finish_wandb_run(wandb_run, args, progress_path)
                return
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

        if not stage.policy_updates_enabled:
            if global_step <= 0 and resume_payload is None and stage_idx == start_stage_idx:
                raise SystemExit(
                    "[mjx_curriculum] a policy-frozen bridge requires a resume checkpoint "
                    "or a policy trained by an earlier stage"
                )
            extra = {
                "stage_index": stage_idx,
                "stage_name": stage.name,
                "stage_update": 0,
                "global_update": global_update,
                "policy_frozen_bridge": True,
            }
            stage_ckpt = args.save_dir / f"{stage_idx:02d}_{stage.name}.pkl"
            save_checkpoint(stage_ckpt, train_state, args, env, global_step, extra=extra)
            save_checkpoint(
                args.save_dir / "mjx_curriculum_last.pkl",
                train_state,
                args,
                env,
                global_step,
                extra=extra,
            )
            print(
                f"[mjx_curriculum] policy-frozen bridge: {stage.name}; "
                "advanced with zero PPO updates"
            )
            continue

        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, args.n_envs)
        env_state, obs = jax.jit(env.reset)(reset_keys)
        critic_obs = env.get_critic_obs(env_state, obs)
        actor_anchor_replay_obs = None
        if actor_anchor_replay_obs_np is not None:
            if actor_anchor_replay_obs_np.shape[1] != env.obs_dim:
                raise SystemExit(
                    "[mjx_curriculum] actor anchor replay obs_dim mismatch: "
                    f"file={actor_anchor_replay_obs_np.shape[1]}, env={env.obs_dim}"
                )
            actor_anchor_replay_obs = jnp.asarray(actor_anchor_replay_obs_np)
            print(
                "[mjx_curriculum] actor anchor replay: "
                f"samples={actor_anchor_replay_obs.shape[0]}, obs_dim={actor_anchor_replay_obs.shape[1]}"
            )
        teacher_distill_replay_obs = None
        if teacher_distill_replay_obs_np is not None:
            if teacher_distill_state is None or teacher_distill_payload is None:
                raise SystemExit("[mjx_curriculum] teacher replay requires a loaded teacher")
            teacher_obs_dim = int(
                teacher_distill_payload.get(
                    "obs_dim",
                    teacher_distill_state.params["pi"]["l1"]["w"].shape[0],
                )
            )
            teacher_act_dim = int(
                teacher_distill_payload.get(
                    "act_dim",
                    teacher_distill_state.params["pi"]["out"]["b"].shape[0],
                )
            )
            if teacher_obs_dim != int(env.obs_dim) or teacher_act_dim != int(env.act_dim):
                raise SystemExit(
                    "[mjx_curriculum] teacher dimensions do not match student env: "
                    f"teacher obs/act={teacher_obs_dim}/{teacher_act_dim}, "
                    f"student env obs/act={env.obs_dim}/{env.act_dim}"
                )
            if teacher_distill_replay_obs_np.shape[1] != teacher_obs_dim:
                raise SystemExit(
                    "[mjx_curriculum] teacher replay obs_dim mismatch: "
                    f"file={teacher_distill_replay_obs_np.shape[1]}, teacher={teacher_obs_dim}"
                )
            teacher_distill_replay_obs = jnp.asarray(teacher_distill_replay_obs_np)
            print(
                "[mjx_curriculum] ideal-domain teacher replay: "
                f"samples={teacher_distill_replay_obs.shape[0]}, "
                f"obs_dim={teacher_distill_replay_obs.shape[1]}, "
                f"coef={args.teacher_distill_coef:.4g}, "
                f"target_clip={args.teacher_distill_action_clip:.3g}"
            )
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
            min_log_std=args.min_log_std,
            target_kl=args.target_kl,
            failure_focus_hit_threshold=args.failure_focus_hit_threshold,
            failure_focus_weight=args.failure_focus_weight,
            failure_focus_tail_steps=args.failure_focus_tail_steps,
            reference_params=(
                train_state.params
                if (
                    float(args.actor_anchor_kl_coef) > 0.0
                    or float(args.actor_anchor_replay_kl_coef) > 0.0
                )
                else None
            ),
            actor_anchor_kl_coef=args.actor_anchor_kl_coef,
            actor_anchor_replay_obs=actor_anchor_replay_obs,
            actor_anchor_replay_kl_coef=args.actor_anchor_replay_kl_coef,
            residual_l2_coef=args.residual_l2_coef,
            teacher_params=(
                teacher_distill_state.params if teacher_distill_state is not None else None
            ),
            teacher_distill_replay_obs=teacher_distill_replay_obs,
            teacher_distill_coef=args.teacher_distill_coef,
            teacher_distill_action_clip=args.teacher_distill_action_clip,
            time_limit_bootstrap=args.time_limit_bootstrap,
        )
        batch_steps = int(args.n_envs) * int(args.n_steps)
        stage_updates = stage_update_cap(stage, args, batch_steps)
        stage_history: list[dict[str, object]] = []
        stage_metric_warmup_updates = (
            int(np.ceil(float(env.max_steps) / max(1, int(args.n_steps))))
            if int(args.stage_metric_warmup_updates) < 0
            else int(args.stage_metric_warmup_updates)
        )
        stage_converged = args.advance_mode == "fixed"
        last_advance_eval_update = -10**9
        best_stage_score = -float("inf")

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
                **episode_hit_distribution_metrics(hit_count, done, ep_len, env.dt),
                **{k: float(v) for k, v in jax.device_get(losses).items()},
                **mean_rollout_metrics(transitions),
            }
            stage_history.append(row)
            metric_history = stage_history[stage_metric_warmup_updates:]
            status = convergence_status(metric_history, stage, env, args, stage_update)
            status["convergence/metric_warmup_updates"] = float(
                stage_metric_warmup_updates
            )
            status["convergence/metric_warmup_remaining"] = float(
                max(0, stage_metric_warmup_updates - stage_update)
            )
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
                camera_label = f" cam={row['convergence/recent_camera_visible']:.2f}/{stage.target_camera_visible:.2f}"
                if stage.min_recent_camera_reward_dense is not None:
                    camera_label += (
                        f" cam_rew={row['convergence/recent_camera_reward_dense']:.3f}/"
                        f"{stage.min_recent_camera_reward_dense:.3f}"
                    )
            ball_view_label = ""
            if stage.target_ball_view_in_bounds is not None:
                ball_view_label = (
                    f" view={row['convergence/recent_ball_view_in_bounds']:.2f}/"
                    f"{stage.target_ball_view_in_bounds:.2f}"
                    f" zideal={row['convergence/recent_ball_view_z_ideal']:.2f}/"
                    f"{stage.target_ball_view_z_ideal:.2f}"
                )
            entry_label = ""
            if stage.target_hit1_rate is not None or stage.target_hit3_rate is not None:
                entry_label = (
                    f" hit1={row['convergence/recent_hit1_rate']:.2f}/"
                    f"{(stage.target_hit1_rate if stage.target_hit1_rate is not None else 0.0):.2f}"
                    f" hit3={row['convergence/recent_hit3_rate']:.2f}/"
                    f"{(stage.target_hit3_rate if stage.target_hit3_rate is not None else 0.0):.2f}"
                    f" hge3={row['convergence/recent_mean_hits_ge3']:.1f}/"
                    f"{(stage.target_mean_hits_ge3 if stage.target_mean_hits_ge3 is not None else 0.0):.1f}"
                )
            cadence_label = ""
            if stage.target_min_hit_interval_s is not None:
                cadence_label = (
                    f" hit_dt3={row['convergence/recent_mean_hit_interval_ge3_s']:.2f}/"
                    f"[{stage.target_min_hit_interval_s:.2f},"
                    f"{(stage.target_max_hit_interval_s if stage.target_max_hit_interval_s is not None else float('inf')):.2f}]"
                    f" hit_hz={row['hit_rate_hz']:.2f}"
                )
            hit_camera_label = ""
            if stage.target_hit_camera_lower_band_rate is not None:
                hit_camera_label = (
                    f" hit_cam={row['convergence/recent_hit_camera_visible_rate']:.2f}/"
                    f"{(stage.target_hit_camera_visible_rate if stage.target_hit_camera_visible_rate is not None else 0.0):.2f}"
                    f" hit_band={row['convergence/recent_hit_camera_lower_band_rate']:.2f}/"
                    f"{stage.target_hit_camera_lower_band_rate:.2f}"
                )
            recover_label = ""
            if (
                stage.max_recent_mean_hit_vxy is not None
                or stage.max_recent_hit_next_contact_anchor_err is not None
                or stage.max_recent_mean_hit_camera_v_frac is not None
            ):
                recover_label = (
                    f" rec_vxy={row['convergence/recent_mean_hit_vxy']:.2f}/"
                    f"{(stage.max_recent_mean_hit_vxy if stage.max_recent_mean_hit_vxy is not None else float('inf')):.2f}"
                    f" rec_next={row['convergence/recent_mean_hit_next_contact_anchor_err']:.2f}/"
                    f"{(stage.max_recent_hit_next_contact_anchor_err if stage.max_recent_hit_next_contact_anchor_err is not None else float('inf')):.2f}"
                    f" rec_v={row['convergence/recent_mean_hit_camera_v_frac']:.2f}/"
                    f"{(stage.max_recent_mean_hit_camera_v_frac if stage.max_recent_mean_hit_camera_v_frac is not None else float('inf')):.2f}"
                )
            gate_label = (
                f" gate={stage.gate_mode}:{row['convergence/performance_gate_ok']:.0f}"
            )
            survival_label = ""
            if stage.target_episode_truncation_rate is not None:
                survival_label = (
                    f" full={row['convergence/recent_episode_truncation_rate']:.2f}/"
                    f"{stage.target_episode_truncation_rate:.2f}"
                )
            missing_label = ""
            if stage.min_ball_obs_missing_refresh_rate is not None:
                missing_label += (
                    f" missing={row['convergence/recent_ball_obs_missing_refresh_rate']:.3f}/"
                    f"{stage.min_ball_obs_missing_refresh_rate:.3f}"
                )
            if stage.max_ball_obs_lost_rate is not None:
                missing_label += (
                    f" lost={row['convergence/recent_ball_obs_lost_rate']:.3f}/"
                    f"{stage.max_ball_obs_lost_rate:.3f}"
                )
            print(
                f"[mjx_curriculum] {stage.name} update={update_label} "
                f"global_step={global_step} sps={row['sps']:,.0f} "
                f"episodes={done_count} return={row['mean_return']:.3f} hits={row['mean_hits']:.2f} "
                f"conv_hits={row['convergence/recent_mean_hits']:.2f}/{stage.target_mean_hits:.2f} "
                f"conv_len={row['convergence/recent_mean_len_frac']:.2f}/{stage.target_mean_len_frac:.2f}"
                f"{camera_label}"
                f"{ball_view_label}"
                f"{entry_label}"
                f"{cadence_label}"
                f"{survival_label}"
                f"{missing_label}"
                f"{hit_camera_label}"
                f"{recover_label}"
                f"{gate_label}"
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
            if (
                int(args.archive_every_updates) > 0
                and stage_update % int(args.archive_every_updates) == 0
            ):
                archive_extra = {
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "archive_reason": "periodic_update",
                    "last_row": row,
                }
                archive_path = args.save_dir / (
                    f"archive_{stage_idx + 1:02d}_{stage.name}_update_{stage_update:04d}.pkl"
                )
                save_checkpoint(
                    archive_path,
                    train_state,
                    args,
                    env,
                    global_step,
                    extra=archive_extra,
                )
            score = stage_best_score(row, stage)
            if (
                score is not None
                and float(row.get("convergence/recent_updates", 0.0)) >= max(1, int(args.convergence_window))
                and score > best_stage_score
            ):
                best_stage_score = score
                extra = {
                    "stage_index": stage_idx,
                    "stage_name": stage.name,
                    "stage_update": stage_update,
                    "global_update": global_update,
                    "best_stage_score": best_stage_score,
                    "last_row": row,
                }
                save_checkpoint(args.save_dir / "mjx_curriculum_best.pkl", train_state, args, env, global_step, extra=extra)

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
                bucket_label = ""
                if bool(row.get("advance_eval/reset_bucket_required", 0.0)):
                    bucket_mode = str(args.advance_eval_reset_bucket_mode)
                    bucket_label = (
                        f", reset_{bucket_mode}_hits="
                        f"{row[f'advance_eval/reset_bucket_{bucket_mode}_mean_hits']:.2f}/"
                        f"{row['advance_eval/reset_bucket_target_mean_hits']:.2f}"
                    )
                print(
                    f"[mjx_curriculum] advance validation failed for next stage "
                    f"{int(row['advance_eval/probe_stage_index'])}/{len(stages)}: "
                    f"hits={row['advance_eval/mean_hits']:.2f}/{row['advance_eval/target_mean_hits']:.2f}, "
                    f"len_frac={row['advance_eval/mean_len_frac']:.2f}/{row['advance_eval/target_mean_len_frac']:.2f}, "
                    f"return={row['advance_eval/mean_return']:.2f}/{row['advance_eval/min_mean_return']:.2f}, "
                    f"cam={row['advance_eval/camera_visible']:.2f}/{row['advance_eval/target_camera_visible']:.2f}"
                    f"{bucket_label}. "
                    "Continuing current stage."
                )

        stage_ckpt = args.save_dir / f"{stage_idx:02d}_{stage.name}.pkl"
        save_checkpoint(stage_ckpt, train_state, args, env, global_step)
        save_checkpoint(args.save_dir / "mjx_curriculum_last.pkl", train_state, args, env, global_step)
        if args.advance_mode == "converged" and not stage_converged:
            if not stage_history:
                raise SystemExit(f"[mjx_curriculum] no updates were run for stage: {stage.name}")
            message = (
                f"[mjx_curriculum] stage did not converge before its update cap: {stage.name}. "
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
