"""Validate a Stage 1a MJX/JAX PPO juggling checkpoint."""

from __future__ import annotations

import argparse
import csv
import pickle
import subprocess
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np

RL_SIM_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = RL_SIM_DIR.parents[1]
OUTPUT_DIR = PACKAGE_DIR / "outputs" / "rl_sim"
DEFAULT_CHECKPOINT = OUTPUT_DIR / "logs_mjx_stage1a" / "mjx_ppo_last.pkl"
DEFAULT_RESULTS = OUTPUT_DIR / "logs_mjx_stage1a" / "validation.csv"

if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from mjx_juggle_env import (
    BALL_OBS_FRAME_PIVOT_MODES,
    MjxJuggleConfig,
    MjxJuggleEnv,
)
from rl_juggle_env_random import RIGHT_ARM_JOINTS
from train_juggle_mjx_ppo import policy_mean


JOINT_PLOT_COLORS = {
    "RightArm-0": "#005AB5",  # blue
    "RightArm-1": "#DC3220",  # red
    "RightArm-2": "#00A08A",  # teal
    "RightArm-3": "#F2AD00",  # gold
    "RightArm-4": "#7A3E9D",  # purple
    "RightArm-5": "#00B7EB",  # cyan
    "RightArm-6": "#5D4037",  # dark brown
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a MJX/JAX PPO juggling checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--env-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optionally load XML/environment configuration from a different checkpoint while "
            "keeping policy parameters from --checkpoint; useful for old-setting guardrails."
        ),
    )
    p.add_argument("--xml", type=Path, default=None, help="Override XML path. Defaults to the checkpoint XML.")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument(
        "--one-episode-per-env",
        action="store_true",
        help=(
            "Record only the first completed episode from each batched environment. "
            "This avoids completion-time bias when a reset parameter changes episode length; "
            "use --episodes equal to --n-envs for a matched reset sweep."
        ),
    )
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true", help="Use policy mean instead of sampling.")
    p.add_argument("--action-gain", type=float, default=1.0, help="Multiply policy action before clipping.")
    p.add_argument(
        "--racket-launch-hold-time-s",
        type=float,
        default=None,
        help="Evaluation-only racket-launch hold duration override.",
    )
    p.add_argument(
        "--racket-launch-hold-mode",
        choices=["racket_relative", "world_fixed"],
        default=None,
        help="Evaluation-only launch hold semantics override.",
    )
    p.add_argument(
        "--racket-launch-hold-time-range-s",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="Evaluation-only per-episode launch hold duration range.",
    )
    p.add_argument(
        "--racket-launch-pre-release-control-mode",
        choices=["policy", "hold_command"],
        default=None,
        help=(
            "Evaluation-only release gate: hold_command executes neutral "
            "commands/feedback until the world-fixed ball is released."
        ),
    )
    p.add_argument(
        "--ball-obs-rate-hz",
        type=float,
        default=None,
        help="Evaluation-only override for the valid ball-state refresh frequency.",
    )
    p.add_argument("--ball-obs-pos-noise-std", type=float, default=None)
    p.add_argument("--ball-obs-vel-noise-std", type=float, default=None)
    p.add_argument("--ball-obs-vel-xy-noise-std", type=float, default=None)
    p.add_argument(
        "--proprio-dq-obs-noise-std",
        type=float,
        default=None,
        help="Evaluation-only AR(1) arm-velocity observation noise standard deviation.",
    )
    p.add_argument(
        "--proprio-racket-vel-obs-noise-std",
        type=float,
        default=None,
        help="Evaluation-only AR(1) racket-velocity observation noise standard deviation.",
    )
    p.add_argument(
        "--proprio-obs-noise-rho",
        type=float,
        default=None,
        help="Evaluation-only correlation coefficient for proprioceptive observation noise.",
    )
    p.add_argument(
        "--ball-obs-velocity-observer-mode",
        choices=[
            "raw",
            "ema_xy",
            "innovation_clip_xy",
            "alpha_beta_xy",
            "confidence_gate_xy",
            "prospective_signal_xy",
        ],
        default=None,
        help="Evaluation-only causal lateral ball-velocity observer override.",
    )
    p.add_argument(
        "--ball-obs-velocity-observer-tau-ms",
        type=float,
        default=None,
        help="Physical-time EMA constant for --ball-obs-velocity-observer-mode ema_xy.",
    )
    p.add_argument(
        "--ball-obs-velocity-observer-max-innovation-m-s",
        type=float,
        default=None,
        help="Per-fresh-sample lateral innovation limit for innovation_clip_xy.",
    )
    p.add_argument("--ball-obs-joint-observer-alpha", type=float, default=None)
    p.add_argument("--ball-obs-joint-observer-beta", type=float, default=None)
    p.add_argument(
        "--ball-obs-joint-observer-raw-velocity-gain",
        type=float,
        default=None,
    )
    p.add_argument("--ball-obs-consistency-gate-threshold-m-s", type=float, default=None)
    p.add_argument("--ball-obs-consistency-gate-direction-cosine", type=float, default=None)
    p.add_argument("--ball-obs-consistency-gate-min-samples", type=int, default=None)
    p.add_argument("--ball-obs-consistency-gate-correction-gain", type=float, default=None)
    p.add_argument("--ball-obs-consistency-gate-max-correction-m-s", type=float, default=None)
    p.add_argument("--ball-obs-consistency-gate-contact-guard-s", type=float, default=None)
    p.add_argument("--ball-obs-prospective-window-samples", type=int, default=None)
    p.add_argument("--ball-obs-prospective-prediction-margin-m", type=float, default=None)
    p.add_argument(
        "--ball-obs-prospective-velocity-disagreement-m-s",
        type=float,
        default=None,
    )
    p.add_argument("--ball-obs-prospective-candidate-gain", type=float, default=None)
    p.add_argument(
        "--ball-obs-prospective-max-correction-m-s",
        type=float,
        default=None,
    )
    p.add_argument("--ball-obs-prospective-max-sample-gap-s", type=float, default=None)
    p.add_argument("--ball-obs-prospective-contact-guard-s", type=float, default=None)
    p.add_argument(
        "--dr-actuator-cmd-tau-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
    )
    p.add_argument(
        "--dr-actuator-cmd-gain-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
    )
    p.add_argument(
        "--dr-second-order-frequency-scale-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
    )
    p.add_argument(
        "--dr-second-order-damping-scale-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
    )
    p.add_argument(
        "--dr-ball-mass-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW_KG", "HIGH_KG"),
        help=(
            "Evaluation-only ball mass DR override in kilograms. Use equal "
            "bounds for a fixed measured ball mass."
        ),
    )
    p.add_argument(
        "--dr-ball-solref-time-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW_S", "HIGH_S"),
        help="Evaluation-only racket-ball contact solref time-constant range.",
    )
    p.add_argument(
        "--dr-ball-solref-damping-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Evaluation-only racket-ball contact solref damping-ratio range; "
            "larger positive values produce a less elastic collision."
        ),
    )
    p.add_argument(
        "--actuator-compensation-mode",
        choices=[
            "none",
            "sport_horizon_inverse",
            "sport_analytic_inverse",
            "sport_regularized_inverse",
            "sport_safe_analytic_inverse",
        ],
        default=None,
        help="Evaluation-only actuator compensation override.",
    )
    p.add_argument(
        "--arm-post-compensation-limiter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluation-only override for the position-aware velocity/acceleration "
            "projection after inverse-MPC compensation. The default preserves the checkpoint."
        ),
    )
    p.add_argument(
        "--arm-servo-target-limiter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluation-only override for the position-aware trajectory projection "
            "after actuator delay/FOPDT and before the MuJoCo position servo."
        ),
    )
    p.add_argument(
        "--arm-servo-target-tracking-planner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluation-only override for the target-aware acceleration "
            "planner after delay/FOPDT and before the unchanged position PD."
        ),
    )
    p.add_argument(
        "--arm-servo-planner-before-actuator-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluation-only ordering override: inverse MPC -> predictive "
            "servo governor -> delay/FOPDT -> XML position PD."
        ),
    )
    p.add_argument("--arm-servo-target-velocity-scale", type=float, default=None)
    p.add_argument("--arm-servo-target-acceleration-scale", type=float, default=None)
    p.add_argument(
        "--actuator-mpc-command-dynamics-constraint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluation-only override for the original inverse-MPC output "
            "position/velocity/acceleration command constraint."
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
        help="Evaluation-only override for inverse-MPC feedback source.",
    )
    p.add_argument(
        "--arm-actual-state-limiter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Evaluation-only override for the actual q/dq/ddq projection "
            "after every MJX 1 ms substep."
        ),
    )
    p.add_argument(
        "--arm-actual-target-tracking-governor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Evaluation-only override for the target-aware actual drive governor.",
    )
    p.add_argument(
        "--arm-actual-governor-natural-frequency-hz",
        type=float,
        default=None,
        help="Evaluation-only natural frequency for the damped drive governor.",
    )
    p.add_argument(
        "--arm-actual-governor-damping-ratio",
        type=float,
        default=None,
        help="Evaluation-only damping ratio for the damped drive governor.",
    )
    p.add_argument(
        "--arm-actual-jerk-limit-deg-s3",
        type=float,
        default=None,
        help="Evaluation-only uniform jerk limit for the actual drive governor.",
    )
    p.add_argument("--max-env-steps", type=int, default=0, help="0 means auto from episodes, envs, and horizon.")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument(
        "--gpu-max-temp-c",
        type=float,
        default=80.0,
        help="Stop frozen validation if any visible host GPU reaches this temperature; <=0 disables.",
    )
    p.add_argument("--gpu-check-every-steps", type=int, default=100)
    p.add_argument("--log-hit-events", action="store_true")
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--no-save-csv", action="store_true")
    p.add_argument("--racket-z-hard-limit-down", type=float, default=None)
    p.add_argument("--racket-z-hard-limit-up", type=float, default=None)
    p.add_argument(
        "--virtual-camera-base-body-name",
        type=str,
        default=None,
        help=(
            "Override virtual_camera_base_body_name for evaluation only. "
            "Useful for checking calibrated base_extrinsic frame mappings "
            "without changing the checkpoint or training curriculum."
        ),
    )
    p.add_argument(
        "--ball-obs-frame-pivot-mode",
        choices=BALL_OBS_FRAME_PIVOT_MODES,
        default=None,
        help=(
            "Override the position pivot for ball observation frame rotation/scale during "
            "evaluation only. The default preserves the checkpoint setting, including the "
            "legacy base-origin behavior of checkpoints that predate this option."
        ),
    )
    p.add_argument(
        "--right-arm-pd-profile",
        choices=["xml", "legacy_stage4g", "comparison_safe_v1"],
        default=None,
        help="Override the right-arm actuator KP/KV profile in the temporary MJX XML.",
    )
    p.add_argument(
        "--ball-obs-missing-episode-coherent-prob",
        type=float,
        default=None,
        help=(
            "Override the fraction of environments using episode-coherent physical "
            "camera/view missing; useful for matched q=0 versus q>0 checkpoint comparisons."
        ),
    )
    p.add_argument(
        "--ball-obs-missing-prob",
        type=float,
        default=None,
        help=(
            "Override both physical-camera and randomized view-bounds missing probabilities; "
            "useful for matched p sweeps of one checkpoint."
        ),
    )
    p.add_argument("--no-terminate-on-racket-z-limit", action="store_true")
    p.add_argument(
        "--no-terminate-on-ball-view-xy-bounds",
        action="store_true",
        help=(
            "Disable only lateral view-bound termination while preserving physical ball, "
            "z-low, and other task terminations; useful for missing-age recovery ablations."
        ),
    )
    p.add_argument(
        "--ignore-early-done",
        action="store_true",
        help="Do not reset/stop on early task termination; only horizon truncation ends the rollout.",
    )
    p.add_argument(
        "--realistic-s2r",
        action="store_true",
        help="Override the checkpoint env with real-camera/latency/actuator-lag sim-to-real stress settings.",
    )
    p.add_argument(
        "--realistic-s2r-profile",
        choices=["detector", "kf"],
        default="kf",
        help=(
            "kf assumes the policy receives a 200Hz Kalman-filter prediction; "
            "detector uses raw 60Hz camera/FOV/dropout stress settings."
        ),
    )
    p.add_argument(
        "--ball-obs-nominal-pos-bias-base",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Optional nominal ball observation bias in base coordinates. "
            "If real detections are chest-frame values fed as base-frame values, use approximately -T_base_chest."
        ),
    )
    p.add_argument(
        "--ball-obs-nominal-vel-bias-base",
        type=float,
        nargs=3,
        default=None,
        metavar=("VX", "VY", "VZ"),
        help="Optional nominal ball velocity observation bias in base coordinates.",
    )
    p.add_argument(
        "--disable-ball-state-dr",
        action="store_true",
        help=(
            "Evaluation-only ablation: use a nominal deterministic ball reset, disable "
            "ball dynamics/contact DR, and remove ball observation noise, dropout, bias, "
            "rotation, and scale DR. Actuator, PD, and racket-mount DR remain unchanged."
        ),
    )
    p.add_argument(
        "--disable-ball-physics-dr",
        action="store_true",
        help=(
            "Evaluation-only ablation: nominalize ball reset, ball dynamics, and "
            "racket-ball contact while preserving the checkpoint's ball observation DR."
        ),
    )
    p.add_argument(
        "--disable-ball-observation-dr",
        action="store_true",
        help=(
            "Evaluation-only ablation: remove ball observation noise/dropout/frame DR "
            "while preserving ball reset, dynamics, and contact DR."
        ),
    )
    p.add_argument(
        "--disable-ball-observation-position-dr",
        action="store_true",
        help=(
            "Evaluation-only ablation: remove ball position noise and position-frame "
            "bias while preserving velocity noise and physical/contact DR."
        ),
    )
    p.add_argument(
        "--disable-ball-observation-velocity-dr",
        action="store_true",
        help=(
            "Evaluation-only ablation: remove ball velocity noise and velocity-frame "
            "bias while preserving position noise and physical/contact DR."
        ),
    )
    p.add_argument("--render", action="store_true", help="Render env 0 with MuJoCo viewer.")
    p.add_argument("--realtime", action="store_true", help="Sleep according to env.dt while rendering.")
    p.add_argument("--slowmo", type=float, default=1.0)
    p.add_argument("--video-out", type=Path, default=None, help="Save env 0 validation render to an mp4/gif file.")
    p.add_argument("--video-fps", type=int, default=30, help="Playback FPS for --video-out.")
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument(
        "--video-env",
        type=int,
        default=0,
        help="Batched environment index to render into --video-out.",
    )
    p.add_argument(
        "--video-slowmo",
        type=float,
        default=1.0,
        help="Playback slow-motion factor. 1.0 is realtime, 2.0 is half speed.",
    )
    p.add_argument(
        "--video-camera",
        type=str,
        default=None,
        help="Optional MuJoCo camera name/id for video rendering. Defaults to the free camera.",
    )
    p.add_argument("--trace-env", type=int, default=0, help="Environment index to save in the action/joint trace.")
    p.add_argument("--action-trace-csv", type=Path, default=None, help="Save per-control-step policy action and joint trace CSV.")
    p.add_argument("--action-plot-out", type=Path, default=None, help="Save a PNG plot of policy action and joint trajectories.")
    p.add_argument("--obs-trace-csv", type=Path, default=None, help="Save the exact per-control-step policy observation CSV.")
    p.add_argument(
        "--freeze-ball-at-reset",
        action="store_true",
        help=(
            "Diagnostic ablation: keep the true MuJoCo ball and the valid policy "
            "ball observation fixed at the episode reset position with zero velocity."
        ),
    )
    p.add_argument(
        "--ball-replay-csv",
        type=Path,
        default=None,
        help=(
            "Diagnostic input replay: inject observed_ball_{x,y,z} and "
            "observed_vel_{x,y,z} from active policy rows of a real "
            "policy_trace.csv into both MuJoCo and the policy observation."
        ),
    )
    p.add_argument(
        "--ball-replay-obs-ablation",
        choices=(
            "none",
            "zero_vx",
            "zero_vy",
            "zero_vxy",
            "freeze_x",
            "freeze_y",
            "freeze_xy",
            "center_rel_x",
            "center_rel_y",
            "center_rel_xy",
            "vertical_xy",
            "zero_phase",
        ),
        default="none",
        help=(
            "Actor-only causal ablation for --ball-replay-csv. The MuJoCo "
            "ball continues to follow the exact real trajectory while only "
            "the selected 67-D policy observation components are replaced."
        ),
    )
    return p.parse_args(argv)


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    if "params" not in payload:
        raise SystemExit(f"Checkpoint does not contain policy params: {path}")
    return payload


def load_real_ball_replay(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load active-policy ball position/velocity samples from a real trace."""
    required = (
        "observed_ball_x",
        "observed_ball_y",
        "observed_ball_z",
        "observed_vel_x",
        "observed_vel_y",
        "observed_vel_z",
    )
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    with path.expanduser().open("r", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in required if name not in (reader.fieldnames or ())]
        if missing:
            raise SystemExit(f"Ball replay CSV missing columns: {missing}")
        for row in reader:
            policy_executed = str(row.get("policyExecuted", "true")).strip().lower()
            if policy_executed not in {"1", "true", "yes"}:
                continue
            try:
                position = [float(row[name]) for name in required[:3]]
                velocity = [float(row[name]) for name in required[3:]]
            except (TypeError, ValueError):
                continue
            if np.all(np.isfinite(position)) and np.all(np.isfinite(velocity)):
                positions.append(position)
                velocities.append(velocity)
    if not positions:
        raise SystemExit(f"Ball replay CSV has no finite active-policy rows: {path}")
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(velocities, dtype=np.float32),
    )


def save_episode_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_trace_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _midpoint(value_range: tuple[float, float]) -> float:
    return 0.5 * (float(value_range[0]) + float(value_range[1]))


def disable_ball_physics_dr(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Nominalize ball reset/dynamics/contact while retaining sensor DR."""

    target_x = _midpoint(cfg.episode_target_x_range_m)
    target_y = _midpoint(cfg.episode_target_y_range_m)
    anchor_z = _midpoint(cfg.episode_racket_anchor_z_range_m)
    launch_gap = _midpoint(cfg.racket_launch_surface_gap_range_m)
    falling_time = _midpoint(cfg.falling_reset_time_to_contact_range_s)
    falling_apex = _midpoint(cfg.falling_reset_apex_height_range_m)
    return replace(
        cfg,
        # Keep the checkpoint's reset mode, but collapse every ball/reset range
        # to its nominal midpoint and remove all spawn/launch perturbations.
        episode_target_x_range_m=(target_x, target_x),
        episode_target_y_range_m=(target_y, target_y),
        episode_racket_anchor_z_range_m=(anchor_z, anchor_z),
        ball_spawn_xy_jitter=0.0,
        ball_spawn_z_jitter=0.0,
        ball_init_vxy_max=0.0,
        ball_init_vz_jitter=0.0,
        falling_reset_time_to_contact_range_s=(falling_time, falling_time),
        falling_reset_apex_height_range_m=(falling_apex, falling_apex),
        falling_reset_vxy_max=0.0,
        falling_reset_contact_xy_jitter=0.0,
        racket_launch_surface_gap_range_m=(launch_gap, launch_gap),
        racket_launch_xy_jitter=0.0,
        racket_launch_vxy_max=0.0,
        racket_launch_vnormal_max=0.0,
        # Nominal MuJoCo ball parameters/contact replace episode DR samples.
        dr_randomize_ball=False,
        dr_randomize_contact=False,
    )


def disable_ball_observation_dr(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Make refreshed ball observations exact while retaining physical DR."""

    return replace(
        cfg,
        # Preserve the checkpoint's sensor update rate/sample-and-hold timing,
        # but make every refreshed six-dimensional ball state exact.
        ball_obs_pos_noise_std=0.0,
        ball_obs_vel_noise_std=0.0,
        ball_obs_noise_warmup_ratio=0.0,
        ball_obs_noise_ramp_ratio=0.0,
        ball_obs_dropout_prob=0.0,
        ball_obs_dropout_max_steps=1,
        ball_obs_dropout_burst_prob=0.0,
        ball_obs_dropout_burst_max_steps=1,
        ball_obs_require_camera_visible=False,
        ball_obs_camera_missing_prob=0.0,
        ball_obs_reset_respects_camera_visibility=False,
        ball_obs_require_view_bounds=False,
        ball_obs_view_bounds_missing_prob=0.0,
        ball_obs_missing_episode_coherent_prob=0.0,
        ball_obs_require_view_z_high=False,
        ball_obs_view_z_high_missing_range_m=(0.0, 0.0),
        ball_obs_nominal_pos_bias_base=(0.0, 0.0, 0.0),
        ball_obs_nominal_vel_bias_base=(0.0, 0.0, 0.0),
        dr_randomize_ball_obs_frame=False,
        dr_ball_obs_pos_bias_base_m=(0.0, 0.0, 0.0),
        dr_ball_obs_rot_bias_deg=(0.0, 0.0, 0.0),
        dr_ball_obs_vel_bias_base_m_s=(0.0, 0.0, 0.0),
        dr_ball_obs_scale_range=(1.0, 1.0),
    )


def disable_ball_observation_position_dr(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Remove only position-side ball observation uncertainty."""

    return replace(
        cfg,
        ball_obs_pos_noise_std=0.0,
        ball_obs_nominal_pos_bias_base=(0.0, 0.0, 0.0),
        dr_ball_obs_pos_bias_base_m=(0.0, 0.0, 0.0),
        dr_ball_obs_rot_bias_deg=(0.0, 0.0, 0.0),
        dr_ball_obs_scale_range=(1.0, 1.0),
    )


def disable_ball_observation_velocity_dr(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Remove only velocity-side ball observation uncertainty."""

    return replace(
        cfg,
        ball_obs_vel_noise_std=0.0,
        ball_obs_nominal_vel_bias_base=(0.0, 0.0, 0.0),
        dr_ball_obs_vel_bias_base_m_s=(0.0, 0.0, 0.0),
    )


def disable_ball_state_dr(cfg: MjxJuggleConfig) -> MjxJuggleConfig:
    """Nominalize the complete ball chain while retaining robot/actuator DR."""

    return disable_ball_observation_dr(disable_ball_physics_dr(cfg))


def env_config_from_checkpoint(payload: dict, args: argparse.Namespace) -> MjxJuggleConfig:
    cfg_payload = payload.get("env_cfg") or {}
    valid_fields = {f.name for f in fields(MjxJuggleConfig)}
    cfg_kwargs = {k: v for k, v in cfg_payload.items() if k in valid_fields}
    if "virtual_camera_pose_mode" not in cfg_payload:
        cfg_kwargs["virtual_camera_pose_mode"] = "body_mount"
    cfg = MjxJuggleConfig(**cfg_kwargs)
    racket_launch_hold_time_s = getattr(args, "racket_launch_hold_time_s", None)
    if racket_launch_hold_time_s is not None:
        if float(racket_launch_hold_time_s) < 0.0:
            raise ValueError("racket_launch_hold_time_s must be non-negative")
        cfg = replace(
            cfg,
            racket_launch_hold_time_s=float(racket_launch_hold_time_s),
            racket_launch_hold_time_range_s=None,
        )
    racket_launch_hold_time_range_s = getattr(
        args, "racket_launch_hold_time_range_s", None
    )
    if racket_launch_hold_time_range_s is not None:
        hold_range = tuple(float(value) for value in racket_launch_hold_time_range_s)
        if min(hold_range) < 0.0:
            raise ValueError("racket_launch_hold_time_range_s must be non-negative")
        cfg = replace(cfg, racket_launch_hold_time_range_s=hold_range)
    racket_launch_hold_mode = getattr(args, "racket_launch_hold_mode", None)
    if racket_launch_hold_mode is not None:
        cfg = replace(cfg, racket_launch_hold_mode=str(racket_launch_hold_mode))
    racket_launch_pre_release_control_mode = getattr(
        args, "racket_launch_pre_release_control_mode", None
    )
    if racket_launch_pre_release_control_mode is not None:
        cfg = replace(
            cfg,
            racket_launch_pre_release_control_mode=str(
                racket_launch_pre_release_control_mode
            ),
        )
    scalar_overrides = {
        "ball_obs_pos_noise_std": getattr(args, "ball_obs_pos_noise_std", None),
        "ball_obs_vel_noise_std": getattr(args, "ball_obs_vel_noise_std", None),
        "ball_obs_vel_xy_noise_std": getattr(
            args, "ball_obs_vel_xy_noise_std", None
        ),
        "proprio_dq_obs_noise_std": getattr(
            args, "proprio_dq_obs_noise_std", None
        ),
        "proprio_racket_vel_obs_noise_std": getattr(
            args, "proprio_racket_vel_obs_noise_std", None
        ),
        "ball_obs_velocity_observer_tau_ms": getattr(
            args, "ball_obs_velocity_observer_tau_ms", None
        ),
        "ball_obs_velocity_observer_max_innovation_m_s": getattr(
            args, "ball_obs_velocity_observer_max_innovation_m_s", None
        ),
        "ball_obs_joint_observer_alpha": getattr(
            args, "ball_obs_joint_observer_alpha", None
        ),
        "ball_obs_joint_observer_beta": getattr(
            args, "ball_obs_joint_observer_beta", None
        ),
        "ball_obs_joint_observer_raw_velocity_gain": getattr(
            args, "ball_obs_joint_observer_raw_velocity_gain", None
        ),
        "ball_obs_consistency_gate_threshold_m_s": getattr(
            args, "ball_obs_consistency_gate_threshold_m_s", None
        ),
        "ball_obs_consistency_gate_correction_gain": getattr(
            args, "ball_obs_consistency_gate_correction_gain", None
        ),
        "ball_obs_consistency_gate_max_correction_m_s": getattr(
            args, "ball_obs_consistency_gate_max_correction_m_s", None
        ),
        "ball_obs_consistency_gate_contact_guard_s": getattr(
            args, "ball_obs_consistency_gate_contact_guard_s", None
        ),
        "ball_obs_prospective_prediction_margin_m": getattr(
            args, "ball_obs_prospective_prediction_margin_m", None
        ),
        "ball_obs_prospective_velocity_disagreement_m_s": getattr(
            args, "ball_obs_prospective_velocity_disagreement_m_s", None
        ),
        "ball_obs_prospective_candidate_gain": getattr(
            args, "ball_obs_prospective_candidate_gain", None
        ),
        "ball_obs_prospective_max_correction_m_s": getattr(
            args, "ball_obs_prospective_max_correction_m_s", None
        ),
        "ball_obs_prospective_max_sample_gap_s": getattr(
            args, "ball_obs_prospective_max_sample_gap_s", None
        ),
        "ball_obs_prospective_contact_guard_s": getattr(
            args, "ball_obs_prospective_contact_guard_s", None
        ),
    }
    for field_name, value in scalar_overrides.items():
        if value is None:
            continue
        if float(value) < 0.0:
            raise ValueError(f"{field_name} must be non-negative")
        cfg = replace(cfg, **{field_name: float(value)})
    proprio_obs_noise_rho = getattr(args, "proprio_obs_noise_rho", None)
    if proprio_obs_noise_rho is not None:
        value = float(proprio_obs_noise_rho)
        if not 0.0 <= value < 1.0:
            raise ValueError("proprio_obs_noise_rho must be in [0, 1)")
        cfg = replace(cfg, proprio_obs_noise_rho=value)
    observer_mode = getattr(args, "ball_obs_velocity_observer_mode", None)
    if observer_mode is not None:
        cfg = replace(cfg, ball_obs_velocity_observer_mode=str(observer_mode))
    normalized_observer_mode = str(cfg.ball_obs_velocity_observer_mode).strip().lower()
    if (
        normalized_observer_mode == "ema_xy"
        and float(cfg.ball_obs_velocity_observer_tau_ms) <= 0.0
    ):
        raise ValueError("ema_xy requires a positive observer tau")
    if (
        normalized_observer_mode == "innovation_clip_xy"
        and float(cfg.ball_obs_velocity_observer_max_innovation_m_s) <= 0.0
    ):
        raise ValueError("innovation_clip_xy requires a positive innovation limit")
    if normalized_observer_mode == "alpha_beta_xy":
        for field_name in (
            "ball_obs_joint_observer_alpha",
            "ball_obs_joint_observer_beta",
            "ball_obs_joint_observer_raw_velocity_gain",
        ):
            value = float(getattr(cfg, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
    consistency_direction_cosine = getattr(
        args, "ball_obs_consistency_gate_direction_cosine", None
    )
    if consistency_direction_cosine is not None:
        value = float(consistency_direction_cosine)
        if not -1.0 <= value <= 1.0:
            raise ValueError(
                "ball_obs_consistency_gate_direction_cosine must be in [-1, 1]"
            )
        cfg = replace(cfg, ball_obs_consistency_gate_direction_cosine=value)
    consistency_min_samples = getattr(
        args, "ball_obs_consistency_gate_min_samples", None
    )
    if consistency_min_samples is not None:
        if int(consistency_min_samples) < 2:
            raise ValueError("ball_obs_consistency_gate_min_samples must be >= 2")
        cfg = replace(
            cfg,
            ball_obs_consistency_gate_min_samples=int(consistency_min_samples),
        )
    if normalized_observer_mode == "confidence_gate_xy":
        if float(cfg.ball_obs_velocity_observer_max_innovation_m_s) <= 0.0:
            raise ValueError(
                "confidence_gate_xy requires a positive checkpoint innovation limit"
            )
        if not 0.0 < float(cfg.ball_obs_consistency_gate_correction_gain) <= 1.0:
            raise ValueError(
                "ball_obs_consistency_gate_correction_gain must be in (0, 1]"
            )
    prospective_window_samples = getattr(
        args, "ball_obs_prospective_window_samples", None
    )
    if prospective_window_samples is not None:
        if int(prospective_window_samples) not in {4, 5, 6}:
            raise ValueError(
                "ball_obs_prospective_window_samples must be 4, 5, or 6"
            )
        cfg = replace(
            cfg,
            ball_obs_prospective_window_samples=int(prospective_window_samples),
        )
    if normalized_observer_mode == "prospective_signal_xy":
        if float(cfg.ball_obs_velocity_observer_max_innovation_m_s) <= 0.0:
            raise ValueError(
                "prospective_signal_xy requires a positive checkpoint innovation limit"
            )
        if int(cfg.ball_obs_prospective_window_samples) not in {4, 5, 6}:
            raise ValueError(
                "ball_obs_prospective_window_samples must be 4, 5, or 6"
            )
        if not 0.0 < float(cfg.ball_obs_prospective_candidate_gain) <= 1.0:
            raise ValueError(
                "ball_obs_prospective_candidate_gain must be in (0, 1]"
            )
        for field_name in (
            "ball_obs_prospective_velocity_disagreement_m_s",
            "ball_obs_prospective_max_correction_m_s",
            "ball_obs_prospective_max_sample_gap_s",
        ):
            if float(getattr(cfg, field_name)) <= 0.0:
                raise ValueError(f"{field_name} must be positive")
    range_overrides = {
        "dr_actuator_cmd_tau_range": getattr(
            args, "dr_actuator_cmd_tau_range", None
        ),
        "dr_actuator_cmd_gain_range": getattr(
            args, "dr_actuator_cmd_gain_range", None
        ),
        "dr_second_order_frequency_scale_range": getattr(
            args, "dr_second_order_frequency_scale_range", None
        ),
        "dr_second_order_damping_scale_range": getattr(
            args, "dr_second_order_damping_scale_range", None
        ),
        "dr_ball_mass_range": getattr(args, "dr_ball_mass_range", None),
        "dr_ball_solref_time_range": getattr(
            args, "dr_ball_solref_time_range", None
        ),
        "dr_ball_solref_damping_range": getattr(
            args, "dr_ball_solref_damping_range", None
        ),
    }
    for field_name, value in range_overrides.items():
        if value is None:
            continue
        low, high = (float(item) for item in value)
        if low <= 0.0 or high < low:
            raise ValueError(
                f"{field_name} must satisfy 0 < LOW <= HIGH, got {(low, high)}"
            )
        cfg = replace(cfg, **{field_name: (low, high)})
    arm_post_compensation_limiter = getattr(
        args,
        "arm_post_compensation_limiter",
        None,
    )
    if arm_post_compensation_limiter is not None:
        cfg = replace(
            cfg,
            arm_post_compensation_limiter=bool(arm_post_compensation_limiter),
        )
    arm_servo_target_limiter = getattr(args, "arm_servo_target_limiter", None)
    if arm_servo_target_limiter is not None:
        cfg = replace(
            cfg,
            arm_servo_target_limiter=bool(arm_servo_target_limiter),
        )
    arm_servo_target_tracking_planner = getattr(
        args,
        "arm_servo_target_tracking_planner",
        None,
    )
    if arm_servo_target_tracking_planner is not None:
        cfg = replace(
            cfg,
            arm_servo_target_tracking_planner=bool(arm_servo_target_tracking_planner),
        )
    arm_servo_planner_before_actuator_model = getattr(
        args,
        "arm_servo_planner_before_actuator_model",
        None,
    )
    if arm_servo_planner_before_actuator_model is not None:
        cfg = replace(
            cfg,
            arm_servo_planner_before_actuator_model=bool(
                arm_servo_planner_before_actuator_model
            ),
        )
    arm_servo_target_velocity_scale = getattr(
        args,
        "arm_servo_target_velocity_scale",
        None,
    )
    if arm_servo_target_velocity_scale is not None:
        cfg = replace(
            cfg,
            arm_servo_target_velocity_scale=float(arm_servo_target_velocity_scale),
        )
    arm_servo_target_acceleration_scale = getattr(
        args,
        "arm_servo_target_acceleration_scale",
        None,
    )
    if arm_servo_target_acceleration_scale is not None:
        cfg = replace(
            cfg,
            arm_servo_target_acceleration_scale=float(arm_servo_target_acceleration_scale),
        )
    actuator_mpc_command_dynamics_constraint = getattr(
        args,
        "actuator_mpc_command_dynamics_constraint",
        None,
    )
    if actuator_mpc_command_dynamics_constraint is not None:
        cfg = replace(
            cfg,
            actuator_mpc_command_dynamics_constraint=bool(actuator_mpc_command_dynamics_constraint),
        )
    actuator_mpc_command_velocity_weight = getattr(args, "actuator_mpc_command_velocity_weight", None)
    if actuator_mpc_command_velocity_weight is not None:
        cfg = replace(
            cfg,
            actuator_mpc_command_velocity_weight=float(actuator_mpc_command_velocity_weight),
        )
    actuator_mpc_command_acceleration_weight = getattr(
        args,
        "actuator_mpc_command_acceleration_weight",
        None,
    )
    if actuator_mpc_command_acceleration_weight is not None:
        cfg = replace(
            cfg,
            actuator_mpc_command_acceleration_weight=float(actuator_mpc_command_acceleration_weight),
        )
    actuator_mpc_command_velocity_scale = getattr(args, "actuator_mpc_command_velocity_scale", None)
    if actuator_mpc_command_velocity_scale is not None:
        cfg = replace(
            cfg,
            actuator_mpc_command_velocity_scale=float(actuator_mpc_command_velocity_scale),
        )
    actuator_mpc_command_acceleration_scale = getattr(
        args,
        "actuator_mpc_command_acceleration_scale",
        None,
    )
    if actuator_mpc_command_acceleration_scale is not None:
        cfg = replace(
            cfg,
            actuator_mpc_command_acceleration_scale=float(actuator_mpc_command_acceleration_scale),
        )
    actuator_mpc_feedback_source = getattr(args, "actuator_mpc_feedback_source", None)
    if actuator_mpc_feedback_source is not None:
        cfg = replace(
            cfg,
            actuator_mpc_feedback_source=str(actuator_mpc_feedback_source),
        )
    arm_actual_state_limiter = getattr(args, "arm_actual_state_limiter", None)
    if arm_actual_state_limiter is not None:
        cfg = replace(
            cfg,
            arm_actual_state_limiter=bool(arm_actual_state_limiter),
        )
    arm_actual_target_tracking_governor = getattr(
        args,
        "arm_actual_target_tracking_governor",
        None,
    )
    if arm_actual_target_tracking_governor is not None:
        cfg = replace(
            cfg,
            arm_actual_target_tracking_governor=bool(
                arm_actual_target_tracking_governor
            ),
        )
    governor_frequency_hz = getattr(
        args,
        "arm_actual_governor_natural_frequency_hz",
        None,
    )
    if governor_frequency_hz is not None:
        governor_frequency_hz = float(governor_frequency_hz)
        if not np.isfinite(governor_frequency_hz) or governor_frequency_hz <= 0.0:
            raise ValueError(
                "arm_actual_governor_natural_frequency_hz must be positive and finite"
            )
        cfg = replace(
            cfg,
            arm_actual_governor_natural_frequency_hz=governor_frequency_hz,
        )
    governor_damping_ratio = getattr(
        args,
        "arm_actual_governor_damping_ratio",
        None,
    )
    if governor_damping_ratio is not None:
        governor_damping_ratio = float(governor_damping_ratio)
        if not np.isfinite(governor_damping_ratio) or governor_damping_ratio <= 0.0:
            raise ValueError(
                "arm_actual_governor_damping_ratio must be positive and finite"
            )
        cfg = replace(
            cfg,
            arm_actual_governor_damping_ratio=governor_damping_ratio,
        )
    arm_actual_jerk_limit_deg_s3 = getattr(
        args,
        "arm_actual_jerk_limit_deg_s3",
        None,
    )
    if arm_actual_jerk_limit_deg_s3 is not None:
        jerk_limit = float(arm_actual_jerk_limit_deg_s3)
        if not np.isfinite(jerk_limit) or jerk_limit <= 0.0:
            raise ValueError("arm_actual_jerk_limit_deg_s3 must be positive and finite")
        cfg = replace(
            cfg,
            arm_actual_jerk_limit_deg_s3=(jerk_limit,) * 7,
        )
    if args.virtual_camera_base_body_name is not None:
        cfg = replace(cfg, virtual_camera_base_body_name=str(args.virtual_camera_base_body_name))
    ball_obs_frame_pivot_mode = getattr(args, "ball_obs_frame_pivot_mode", None)
    if ball_obs_frame_pivot_mode is not None:
        cfg = replace(
            cfg,
            ball_obs_frame_pivot_mode=str(ball_obs_frame_pivot_mode),
        )
    if args.right_arm_pd_profile is not None:
        cfg = replace(cfg, right_arm_pd_profile=str(args.right_arm_pd_profile))
    actuator_compensation_mode = getattr(args, "actuator_compensation_mode", None)
    if actuator_compensation_mode is not None:
        cfg = replace(
            cfg,
            actuator_compensation_mode=str(actuator_compensation_mode),
        )
    if args.racket_z_hard_limit_down is not None:
        cfg = replace(cfg, racket_z_hard_limit_down=float(args.racket_z_hard_limit_down))
    if args.racket_z_hard_limit_up is not None:
        cfg = replace(cfg, racket_z_hard_limit_up=float(args.racket_z_hard_limit_up))
    if args.ball_obs_missing_episode_coherent_prob is not None:
        coherent_prob = float(args.ball_obs_missing_episode_coherent_prob)
        if not 0.0 <= coherent_prob <= 1.0:
            raise SystemExit(
                "--ball-obs-missing-episode-coherent-prob must be within [0, 1]"
            )
        cfg = replace(
            cfg,
            ball_obs_missing_episode_coherent_prob=coherent_prob,
        )
    if args.ball_obs_missing_prob is not None:
        missing_prob = float(args.ball_obs_missing_prob)
        if not 0.0 <= missing_prob <= 1.0:
            raise SystemExit("--ball-obs-missing-prob must be within [0, 1]")
        cfg = replace(
            cfg,
            ball_obs_camera_missing_prob=missing_prob,
            ball_obs_view_bounds_missing_prob=missing_prob,
        )
    if args.no_terminate_on_racket_z_limit:
        cfg = replace(cfg, terminate_on_racket_z_limit=False)
    if args.no_terminate_on_ball_view_xy_bounds:
        cfg = replace(
            cfg,
            terminate_on_ball_view_x_bounds=False,
            terminate_on_ball_view_y_bounds=False,
        )
    if args.realistic_s2r and args.realistic_s2r_profile == "detector":
        cfg = replace(
            cfg,
            ball_obs_rate_hz=60.0,
            ball_obs_fractional_rate=True,
            ball_obs_age_tracks_stale=True,
            ball_obs_dropout_on_refresh_only=True,
            ball_obs_require_camera_visible=True,
            ball_obs_pos_noise_std=0.006,
            ball_obs_vel_noise_std=0.08,
            ball_obs_noise_warmup_ratio=0.0,
            ball_obs_noise_ramp_ratio=0.05,
            ball_obs_dropout_prob=0.04,
            ball_obs_dropout_max_steps=10,
            ball_obs_dropout_burst_prob=0.010,
            ball_obs_dropout_burst_max_steps=48,
            domain_randomization=True,
            dr_randomize_latency=True,
            dr_obs_latency_steps_range=(3, 12),
            dr_action_latency_steps_range=(20, 34),
            actuator_cmd_filter=True,
            dr_randomize_actuator_cmd_filter=True,
            dr_actuator_cmd_tau_range=(0.04, 0.10),
            dr_actuator_cmd_gain_range=(0.75, 1.00),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.030, 0.030, 0.040),
            dr_ball_obs_rot_bias_deg=(2.0, 2.0, 3.0),
            dr_ball_obs_vel_bias_base_m_s=(0.05, 0.05, 0.08),
            dr_ball_obs_scale_range=(0.97, 1.03),
        )
    elif args.realistic_s2r and args.realistic_s2r_profile == "kf":
        cfg = replace(
            cfg,
            ball_obs_rate_hz=200.0,
            ball_obs_fractional_rate=False,
            ball_obs_age_tracks_stale=False,
            ball_obs_dropout_on_refresh_only=False,
            ball_obs_require_camera_visible=False,
            ball_obs_pos_noise_std=0.006,
            ball_obs_vel_noise_std=0.08,
            ball_obs_noise_warmup_ratio=0.0,
            ball_obs_noise_ramp_ratio=0.05,
            ball_obs_dropout_prob=0.0,
            ball_obs_dropout_max_steps=1,
            ball_obs_dropout_burst_prob=0.0,
            ball_obs_dropout_burst_max_steps=1,
            domain_randomization=True,
            dr_randomize_latency=True,
            dr_obs_latency_steps_range=(0, 4),
            dr_action_latency_steps_range=(20, 34),
            actuator_cmd_filter=True,
            dr_randomize_actuator_cmd_filter=True,
            dr_actuator_cmd_tau_range=(0.04, 0.10),
            dr_actuator_cmd_gain_range=(0.75, 1.00),
            dr_randomize_ball_obs_frame=True,
            dr_ball_obs_pos_bias_base_m=(0.030, 0.030, 0.040),
            dr_ball_obs_rot_bias_deg=(2.0, 2.0, 3.0),
            dr_ball_obs_vel_bias_base_m_s=(0.05, 0.05, 0.08),
            dr_ball_obs_scale_range=(0.97, 1.03),
        )
    ball_obs_rate_hz = getattr(args, "ball_obs_rate_hz", None)
    if ball_obs_rate_hz is not None:
        ball_obs_rate_hz = float(ball_obs_rate_hz)
        if not np.isfinite(ball_obs_rate_hz) or ball_obs_rate_hz <= 0.0:
            raise ValueError("ball_obs_rate_hz must be positive and finite")
        cfg = replace(
            cfg,
            ball_obs_rate_hz=ball_obs_rate_hz,
            ball_obs_fractional_rate=True,
        )
    if args.ball_obs_nominal_pos_bias_base is not None:
        cfg = replace(
            cfg,
            ball_obs_nominal_pos_bias_base=tuple(float(v) for v in args.ball_obs_nominal_pos_bias_base),
        )
    if args.ball_obs_nominal_vel_bias_base is not None:
        cfg = replace(
            cfg,
            ball_obs_nominal_vel_bias_base=tuple(float(v) for v in args.ball_obs_nominal_vel_bias_base),
        )
    if getattr(args, "disable_ball_physics_dr", False):
        cfg = disable_ball_physics_dr(cfg)
    if getattr(args, "disable_ball_observation_dr", False):
        cfg = disable_ball_observation_dr(cfg)
    if getattr(args, "disable_ball_observation_position_dr", False):
        cfg = disable_ball_observation_position_dr(cfg)
    if getattr(args, "disable_ball_observation_velocity_dr", False):
        cfg = disable_ball_observation_velocity_dr(cfg)
    if getattr(args, "disable_ball_state_dr", False):
        cfg = disable_ball_state_dr(cfg)
    return cfg


def copy_env_to_mujoco_data(
    env: MjxJuggleEnv,
    env_state,
    render_data: mj.MjData,
    env_index: int = 0,
) -> None:
    """Copy one batched MJX environment into MuJoCo render data."""

    qpos = np.asarray(jax.device_get(env_state.data.qpos[env_index]))
    qvel = np.asarray(jax.device_get(env_state.data.qvel[env_index]))
    step_count = int(np.asarray(jax.device_get(env_state.step_count[env_index])))
    render_data.qpos[:] = qpos
    render_data.qvel[:] = qvel
    render_data.ctrl[:] = np.asarray(jax.device_get(env_state.data.ctrl[env_index]))
    render_data.time = step_count * env.dt
    mj.mj_forward(env.mj_model, render_data)


def append_video_frame(
    writer,
    renderer: mj.Renderer,
    env: MjxJuggleEnv,
    env_state,
    render_data: mj.MjData,
    camera,
    env_index: int = 0,
) -> None:
    copy_env_to_mujoco_data(env, env_state, render_data, env_index)
    if camera is None:
        renderer.update_scene(render_data)
    else:
        renderer.update_scene(render_data, camera=camera)
    writer.append_data(np.asarray(renderer.render(), dtype=np.uint8))


# Backward-compatible helper name for local callers/tests.
copy_env0_to_mujoco_data = copy_env_to_mujoco_data


def ensure_offscreen_framebuffer(model: mj.MjModel, width: int, height: int) -> tuple[int, int, int, int]:
    old_width = int(model.vis.global_.offwidth)
    old_height = int(model.vis.global_.offheight)
    model.vis.global_.offwidth = max(old_width, int(width))
    model.vis.global_.offheight = max(old_height, int(height))
    return old_width, old_height, int(model.vis.global_.offwidth), int(model.vis.global_.offheight)


def camera_trace_metrics(env: MjxJuggleEnv, env_state) -> dict[str, jax.Array]:
    """Return true ball position in the virtual camera frame for trace CSVs."""
    data = env_state.data
    n = data.xpos.shape[0]
    zeros = jnp.zeros((n,), dtype=jnp.float32)
    out = {
        "camera_available": zeros,
        "camera_in_front": zeros,
        "camera_in_depth": zeros,
        "camera_in_frustum": zeros,
        "camera_in_image": zeros,
        "camera_visible": zeros,
        "ball_cam_x": zeros,
        "ball_cam_y": zeros,
        "ball_cam_z": zeros,
        "ball_cam_distance": zeros,
        "ball_pixel_u": zeros,
        "ball_pixel_v": zeros,
        "camera_world_x": zeros,
        "camera_world_y": zeros,
        "camera_world_z": zeros,
        "ball_world_x": zeros,
        "ball_world_y": zeros,
        "ball_world_z": zeros,
    }
    if env.ball_body_id < 0:
        return out

    bpos = data.xpos[:, env.ball_body_id]
    cam_pos, cam_R, camera_available = env._virtual_camera_pose(data)
    if not camera_available:
        return out
    p_cam = jnp.einsum("nij,nj->ni", jnp.swapaxes(cam_R, 1, 2), bpos - cam_pos)

    x = p_cam[:, 0]
    y = p_cam[:, 1]
    z = p_cam[:, 2]
    has_projection = z > 1e-6
    z_safe = jnp.where(has_projection, z, 1.0)
    width = float(env.cfg.camera_image_width)
    height = float(env.cfg.camera_image_height)
    fx = float(env.cfg.camera_fx)
    fy = float(env.cfg.camera_fy)
    cx = float(env.cfg.camera_cx)
    cy = float(env.cfg.camera_cy)
    u = fx * (x / z_safe) + cx
    v = cy + fy * (y / z_safe)
    h_half = float(np.deg2rad(float(env.cfg.camera_hfov_deg) * 0.5))
    v_half = float(np.deg2rad(float(env.cfg.camera_vfov_deg) * 0.5))
    x_angle = jnp.arctan2(x, z_safe)
    y_angle = jnp.arctan2(y, z_safe)
    in_front = has_projection
    in_depth = has_projection & (z >= float(env.cfg.camera_min_depth)) & (z <= float(env.cfg.camera_max_depth))
    in_frustum = has_projection & (jnp.abs(x_angle) <= h_half) & (jnp.abs(y_angle) <= v_half)
    in_image = has_projection & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    visible = in_front & in_depth & in_frustum & in_image

    return {
        "camera_available": jnp.ones((n,), dtype=jnp.float32),
        "camera_in_front": in_front.astype(jnp.float32),
        "camera_in_depth": in_depth.astype(jnp.float32),
        "camera_in_frustum": in_frustum.astype(jnp.float32),
        "camera_in_image": in_image.astype(jnp.float32),
        "camera_visible": visible.astype(jnp.float32),
        "ball_cam_x": x,
        "ball_cam_y": y,
        "ball_cam_z": z,
        "ball_cam_distance": jnp.linalg.norm(p_cam, axis=-1),
        "ball_pixel_u": jnp.where(has_projection, u, 0.0),
        "ball_pixel_v": jnp.where(has_projection, v, 0.0),
        "camera_world_x": cam_pos[:, 0],
        "camera_world_y": cam_pos[:, 1],
        "camera_world_z": cam_pos[:, 2],
        "ball_world_x": bpos[:, 0],
        "ball_world_y": bpos[:, 1],
        "ball_world_z": bpos[:, 2],
    }


def override_ball_state(
    env: MjxJuggleEnv,
    env_state,
    ball_position: jax.Array,
    ball_velocity: jax.Array,
    *,
    exact_observation_frame: bool = False,
):
    """Override physics and policy-facing ball state for diagnostic replays."""
    ball_position = jnp.asarray(ball_position, dtype=jnp.float32)
    ball_velocity = jnp.asarray(ball_velocity, dtype=jnp.float32)
    data = env_state.data
    if exact_observation_frame:
        base_x = data.qpos[:, env.base_x_qadr]
        base_y = data.qpos[:, env.base_y_qadr]
        base_yaw = data.qpos[:, env.base_yaw_qadr]
        base_vx = data.qvel[:, env.base_x_vadr]
        base_vy = data.qvel[:, env.base_y_vadr]
        base_yaw_rate = data.qvel[:, env.base_yaw_vadr]
        c = jnp.cos(base_yaw)
        s = jnp.sin(base_yaw)
        rel_x = c * ball_position[:, 0] - s * ball_position[:, 1]
        rel_y = s * ball_position[:, 0] + c * ball_position[:, 1]
        vel_rel_x = c * ball_velocity[:, 0] - s * ball_velocity[:, 1]
        vel_rel_y = s * ball_velocity[:, 0] + c * ball_velocity[:, 1]
        ball_position = jnp.stack(
            [base_x + rel_x, base_y + rel_y, ball_position[:, 2]],
            axis=-1,
        )
        ball_velocity = jnp.stack(
            [
                base_vx + vel_rel_x - base_yaw_rate * rel_y,
                base_vy + vel_rel_y + base_yaw_rate * rel_x,
                ball_velocity[:, 2],
            ],
            axis=-1,
        )
    qpos = data.qpos.at[:, env.ball_qadr : env.ball_qadr + 3].set(
        ball_position
    )
    qpos = qpos.at[:, env.ball_qadr + 3 : env.ball_qadr + 7].set(
        jnp.broadcast_to(
            jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
            (env.n_envs, 4),
        )
    )
    qvel = data.qvel.at[:, env.ball_vadr : env.ball_vadr + 3].set(ball_velocity)
    qvel = qvel.at[:, env.ball_vadr + 3 : env.ball_vadr + 6].set(0.0)
    qacc = data.qacc.at[:, env.ball_vadr : env.ball_vadr + 6].set(0.0)
    data = data.replace(qpos=qpos, qvel=qvel, qacc=qacc)
    data = env.batched_forward(env_state.model, data)
    state_updates = dict(
        data=data,
        reset_ball_pos=ball_position,
        reset_ball_vel=ball_velocity,
        prev_ball_pos=ball_position,
        cached_ball_obs_pos=ball_position,
        cached_ball_obs_vel=ball_velocity,
        ball_obs_valid_pos=ball_position,
        ball_obs_valid_vel=ball_velocity,
        ball_obs_age_seconds=jnp.zeros((env.n_envs,), dtype=jnp.float32),
        ball_obs_missing_since_sample=jnp.zeros((env.n_envs,), dtype=bool),
        ball_obs_dropout_remaining=jnp.zeros((env.n_envs,), dtype=jnp.int32),
        last_ball_obs_step=env_state.step_count,
    )
    if exact_observation_frame:
        state_updates.update(
            ball_obs_pos_bias_base=jnp.zeros_like(env_state.ball_obs_pos_bias_base),
            ball_obs_rot_bias_rpy=jnp.zeros_like(env_state.ball_obs_rot_bias_rpy),
            ball_obs_vel_bias_base=jnp.zeros_like(env_state.ball_obs_vel_bias_base),
            ball_obs_scale=jnp.ones_like(env_state.ball_obs_scale),
        )
    env_state = env_state._replace(**state_updates)
    return env_state, env.observe(env_state)


def ablate_ball_replay_observation(
    obs: jax.Array,
    mode: str,
    reference_ball_xy_base: jax.Array,
) -> jax.Array:
    """Apply an actor-only counterfactual to a forced real-ball replay.

    The base 50-D observation layout is fixed by ``MjxJuggleEnv._make_obs``:
    ball position 20:23, ball velocity 23:26, racket position 26:29,
    ball-minus-racket position 32:35, and ball age 49.  The deployed 67-D
    contract appends phase features at 65:67.  Keeping MuJoCo and the replayed
    ball untouched while changing only these actor inputs distinguishes an
    observation-triggered chase from plant/contact dynamics.
    """

    mode = str(mode)
    if mode == "none":
        return obs
    out = obs
    if mode in {"zero_vx", "zero_vxy", "vertical_xy"}:
        out = out.at[:, 23].set(0.0)
    if mode in {"zero_vy", "zero_vxy", "vertical_xy"}:
        out = out.at[:, 24].set(0.0)

    freeze_x = mode in {"freeze_x", "freeze_xy", "vertical_xy"}
    freeze_y = mode in {"freeze_y", "freeze_xy", "vertical_xy"}
    center_x = mode in {"center_rel_x", "center_rel_xy"}
    center_y = mode in {"center_rel_y", "center_rel_xy"}
    if freeze_x:
        ball_x = reference_ball_xy_base[:, 0]
        out = out.at[:, 20].set(ball_x)
        out = out.at[:, 32].set(ball_x - out[:, 26])
    if freeze_y:
        ball_y = reference_ball_xy_base[:, 1]
        out = out.at[:, 21].set(ball_y)
        out = out.at[:, 33].set(ball_y - out[:, 27])
    if center_x:
        out = out.at[:, 20].set(out[:, 26])
        out = out.at[:, 32].set(0.0)
    if center_y:
        out = out.at[:, 21].set(out[:, 27])
        out = out.at[:, 33].set(0.0)
    if mode == "zero_phase":
        if out.shape[1] < 67:
            raise ValueError("zero_phase requires the deployed 67-D observation")
        out = out.at[:, 65:67].set(0.0)
    return out


def freeze_ball_state(
    env: MjxJuggleEnv,
    env_state,
    fixed_ball_position: jax.Array,
):
    """Clamp ball state at reset position with a valid zero-velocity observation."""
    return override_ball_state(
        env,
        env_state,
        fixed_ball_position,
        jnp.zeros_like(fixed_ball_position),
    )


def make_eval_step(
    env: MjxJuggleEnv,
    deterministic: bool,
    action_gain: float,
    ignore_early_done: bool = False,
    fixed_ball_position: jax.Array | None = None,
    ball_replay_positions: jax.Array | None = None,
    ball_replay_velocities: jax.Array | None = None,
    ball_replay_obs_ablation: str = "none",
    reference_ball_xy_base: jax.Array | None = None,
):
    base_joint_names = ("base_x", "base_y", "base_z", "base_roll", "base_pitch", "base_yaw")
    base_joint_ids = [
        mj.mj_name2id(env.mj_model, mj.mjtObj.mjOBJ_JOINT, name)
        for name in base_joint_names
    ]
    base_qadrs = np.asarray(
        [env.mj_model.jnt_qposadr[joint_id] for joint_id in base_joint_ids],
        dtype=np.int32,
    )
    base_vadrs = np.asarray(
        [env.mj_model.jnt_dofadr[joint_id] for joint_id in base_joint_ids],
        dtype=np.int32,
    )

    def eval_step(params, env_state, obs, rng, running_return, running_length):
        rng, action_key, reset_key = jax.random.split(rng, 3)
        mean = policy_mean(params, obs)
        if deterministic:
            raw_action = mean
        else:
            log_std = params["log_std"]
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
        action = jnp.clip(raw_action * float(action_gain), -1.0, 1.0)

        prev_arm_qvel = env_state.data.qvel[:, env.arm_vadr]
        next_env_state, next_obs, reward, done, metrics = env.step(env_state, action)
        if fixed_ball_position is not None:
            next_env_state, next_obs = freeze_ball_state(
                env,
                next_env_state,
                fixed_ball_position,
            )
        elif ball_replay_positions is not None:
            replay_index = jnp.clip(
                next_env_state.step_count,
                0,
                ball_replay_positions.shape[0] - 1,
            )
            next_env_state, next_obs = override_ball_state(
                env,
                next_env_state,
                ball_replay_positions[replay_index],
                ball_replay_velocities[replay_index],
                exact_observation_frame=True,
            )
            next_obs = ablate_ball_replay_observation(
                next_obs,
                ball_replay_obs_ablation,
                reference_ball_xy_base,
            )
        if bool(ignore_early_done):
            done = metrics["truncated"].astype(bool)
        completed_return = running_return + reward
        completed_length = running_length + 1
        effective_action = next_env_state.prev_action
        arm_q = next_env_state.data.qpos[:, env.arm_qadr]
        arm_qvel = next_env_state.data.qvel[:, env.arm_vadr]
        arm_qacc = (arm_qvel - prev_arm_qvel) / max(env.dt, 1e-6)
        arm_qacc_mj = next_env_state.data.qacc[:, env.arm_vadr]
        base_q6 = next_env_state.data.qpos[:, base_qadrs]
        base_qvel6 = next_env_state.data.qvel[:, base_vadrs]
        arm_cmd_q = next_env_state.arm_cmd_q
        arm_cmd_qvel = next_env_state.arm_cmd_qvel
        arm_raw_comp_q = next_env_state.arm_actuator_q_ref_latest
        arm_safe_q = next_env_state.arm_safe_q_ref_latest
        arm_safe_qvel = next_env_state.arm_safe_qvel
        arm_safe_qacc = metrics["ddq_post_comp_safe_latest"]
        arm_servo_target_unlimited = metrics["q_servo_target_unlimited"]
        arm_applied_q = next_env_state.arm_applied_q
        arm_applied_qvel = next_env_state.arm_applied_qvel
        arm_applied_qacc = metrics["ddq_servo_target"]
        desired_qdd_raw = (
            effective_action
            * env.arm_acc_limit_rad_s2[None, :]
            * float(env.cfg.action_acc_scale)
            * next_env_state.action_scale_mult[:, None]
        )
        if bool(env.cfg.arm_action_limiter):
            desired_qdd = jnp.clip(desired_qdd_raw, -env.arm_acc_limit_rad_s2[None, :], env.arm_acc_limit_rad_s2[None, :])
        else:
            desired_qdd = desired_qdd_raw

        reset_keys = jax.random.split(reset_key, env.n_envs)
        camera_metrics = camera_trace_metrics(env, next_env_state)
        next_env_state, next_obs = env.reset_done(next_env_state, next_obs, done, reset_keys)
        if fixed_ball_position is not None:
            next_env_state, next_obs = freeze_ball_state(
                env,
                next_env_state,
                fixed_ball_position,
            )
        elif ball_replay_positions is not None:
            replay_index = jnp.clip(
                next_env_state.step_count,
                0,
                ball_replay_positions.shape[0] - 1,
            )
            next_env_state, next_obs = override_ball_state(
                env,
                next_env_state,
                ball_replay_positions[replay_index],
                ball_replay_velocities[replay_index],
                exact_observation_frame=True,
            )
            next_obs = ablate_ball_replay_observation(
                next_obs,
                ball_replay_obs_ablation,
                reference_ball_xy_base,
            )
        next_running_return = jnp.where(done, 0.0, completed_return)
        next_running_length = jnp.where(done, 0, completed_length)

        step_metrics = {
            "done": done,
            "reward": reward,
            "episode_return": completed_return,
            "episode_length": completed_length,
            "hit_count": metrics["hit_count"],
            "new_hit": metrics["new_hit"],
            "in_contact": metrics["in_contact"],
            "ball_z": metrics["ball_z"],
            "racket_z": metrics["racket_z"],
            "racket_z_rel": metrics["racket_z_rel"],
            "action_norm": jnp.linalg.norm(action, axis=-1),
            "terminated": metrics["terminated"],
            "truncated": metrics["truncated"],
            "episode_step": metrics["episode_step"],
            "obs": obs,
            "policy_mean": mean,
            "raw_action": raw_action,
            "applied_action": action,
            "effective_action": effective_action,
            "desired_qdd_raw": desired_qdd_raw,
            "desired_qdd": desired_qdd,
            "arm_cmd_q": arm_cmd_q,
            "arm_cmd_qvel": arm_cmd_qvel,
            "arm_cmd_qdd": desired_qdd,
            "arm_raw_comp_q": arm_raw_comp_q,
            "arm_safe_q": arm_safe_q,
            "arm_safe_qvel": arm_safe_qvel,
            "arm_safe_qacc": arm_safe_qacc,
            "arm_servo_target_unlimited": arm_servo_target_unlimited,
            "arm_applied_q": arm_applied_q,
            "arm_applied_qvel": arm_applied_qvel,
            "arm_applied_qacc": arm_applied_qacc,
            "arm_q": arm_q,
            "arm_qvel": arm_qvel,
            "arm_qacc": arm_qacc,
            "arm_qacc_mj": arm_qacc_mj,
            "base_q6": base_q6,
            "base_qvel6": base_qvel6,
        }
        step_metrics.update(camera_metrics)
        for key, value in metrics.items():
            if (
                key.startswith("done/")
                or key.startswith("reward/")
                or key.startswith("arm_post_comp_safety_")
                or key.startswith("arm_servo_safety_")
                or (
                    key.startswith("arm_actual_safety_")
                    and key != "arm_actual_safety_clip_count"
                )
                or key.startswith("raw_action_clip_")
                or key.startswith("ball_obs_")
                or key.startswith("hit_camera_")
                or key.startswith("hit_cycle_")
                or key.startswith("hit_racket_vxy_")
                or key.startswith("racket_cycle_")
                or key.startswith("racket_angular_velocity_")
                or key == "racket_tilt_angular_speed_rad_s"
                or key == "racket_spin_angular_speed_rad_s"
                or key
                in {
                    "physical_contact_edge",
                    "launch_clearance_crossing",
                    "low_survival_launch",
                    "subfloor_launch",
                    "confirmed_hit",
                    "failed_hit",
                    "ignored_fast_hit",
                    "hit_event_count",
                    "hit_vxy_sum",
                    "hit_vxy_sq_sum",
                    "hit_contact_center_dist_sum",
                    "hit_racket_up_cos_sum",
                    "hit_apex_rel_height_sum",
                    "hit_apex_view_x_sum",
                    "hit_apex_view_y_sum",
                    "hit_next_contact_anchor_err_sum",
                    "hit_posterior_contact_event",
                    "hit_posterior_contact_anchor_err_sum",
                    "hit_contact_anchor_contraction_sum",
                }
                or key.startswith("reset_")
                or key.startswith("dr_")
                or key
                in {
                    "action_scale_mult",
                    "delay_bin_id",
                    "delay_steps",
                    "tau_act_ms",
                    "servo_execution_delay_steps",
                    "servo_execution_delay_ms",
                }
                or key in {"ball_view_z_high_exceeded", "ball_view_in_bounds", "ball_view_z_ideal"}
            ):
                step_metrics[key] = value
        return next_env_state, next_obs, rng, next_running_return, next_running_length, step_metrics

    return jax.jit(eval_step)


def aggregate_episode_metrics(
    rows: list[dict[str, float]],
) -> dict[str, float]:
    def ratio(numerator_key: str, denominator_key: str) -> float:
        numerator = sum(float(row.get(numerator_key, 0.0)) for row in rows)
        denominator = sum(float(row.get(denominator_key, 0.0)) for row in rows)
        return numerator / denominator if denominator > 0.0 else float("nan")

    return {
        "hit_camera_visible_rate": ratio(
            "hit_camera_visible_events",
            "hit_camera_events",
        ),
        "hit_camera_in_margin_rate": ratio(
            "hit_camera_in_margin_events",
            "hit_camera_events",
        ),
        "hit_camera_lower_band_rate": ratio(
            "hit_camera_lower_band_events",
            "hit_camera_events",
        ),
        "mean_hit_camera_v_frac": ratio(
            "hit_camera_v_frac_sum",
            "hit_camera_visible_events",
        ),
        "mean_hit_vxy": ratio("hit_vxy_sum", "hit_camera_events"),
        "camera_visible_rate": ratio("camera_visible_steps", "length"),
        "ball_view_in_bounds_rate": ratio("ball_view_in_bounds_steps", "length"),
        "ball_view_z_ideal_rate": ratio("ball_view_z_ideal_steps", "length"),
        "ball_view_z_high_exceeded_rate": ratio(
            "ball_view_z_high_exceeded_steps",
            "length",
        ),
        "ball_obs_missing_refresh_rate": ratio(
            "ball_obs_missing_refresh_count",
            "ball_obs_refresh_count",
        ),
        "ball_obs_lost_rate": ratio("ball_obs_lost_steps", "length"),
        "ball_obs_stale_rate": ratio("ball_obs_stale_steps", "length"),
        "ball_obs_dropout_rate": ratio("ball_obs_dropout_steps", "length"),
        "ball_obs_reacquired_after_missing_rate": ratio(
            "ball_obs_reacquired_after_missing_events",
            "ball_obs_missing_exposure_count",
        ),
        "ball_obs_reacquired_after_lost_rate": ratio(
            "ball_obs_reacquired_after_lost_events",
            "ball_obs_lost_exposure_count",
        ),
    }
def add_terminal_step_metrics(
    row: dict[str, float],
    host: dict[str, np.ndarray],
    env_i: int,
) -> None:
    for key, value in host.items():
        if not (
            key.startswith("done/")
            or key.startswith("reward/")
            or key.startswith("ball_obs_")
            or key.startswith("reset_")
            or key.startswith("dr_")
            or key.startswith("hit_camera_")
            or key.startswith("hit_cycle_")
            or key.startswith("hit_racket_vxy_")
            or key.startswith("racket_cycle_")
            or key
            in {
                "action_scale_mult",
                "delay_bin_id",
                "delay_steps",
                "tau_act_ms",
                "servo_execution_delay_steps",
                "servo_execution_delay_ms",
                "ball_view_z_high_exceeded",
                "ball_view_in_bounds",
                "ball_view_z_ideal",
            }
        ):
            continue
        output_key = key if key not in row else f"last/{key}"
        row[output_key] = float(value[env_i])


def accumulate_named_episode_metrics(
    accumulators: dict[str, np.ndarray],
    host: dict[str, np.ndarray],
    metric_names: tuple[str, ...] | None = None,
    prefix: str | None = None,
) -> None:
    """Accumulate selected per-step metrics without losing terminal values."""

    names = metric_names or tuple(
        name for name in host if prefix is not None and name.startswith(prefix)
    )
    for name in names:
        value = np.asarray(host.get(name, 0.0), dtype=np.float64)
        if name not in accumulators:
            if value.ndim != 1:
                raise ValueError(f"episode metric {name!r} must be one-dimensional")
            accumulators[name] = np.zeros_like(value, dtype=np.float64)
        accumulators[name] += value





def summarize(rows: list[dict[str, float]]) -> str:
    if not rows:
        return "no completed episodes"
    returns = np.asarray([r["return"] for r in rows], dtype=np.float32)
    lengths = np.asarray([r["length"] for r in rows], dtype=np.float32)
    hits = np.asarray([r["hits"] for r in rows], dtype=np.float32)
    full_rate = float(np.mean([float(r.get("truncated", 0.0)) > 0.5 for r in rows]))
    aggregate = aggregate_episode_metrics(rows)
    return (
        f"episodes={len(rows)} "
        f"return_mean={returns.mean():.3f} return_std={returns.std():.3f} "
        f"len_mean={lengths.mean():.1f} full_rate={full_rate:.3f} hits_mean={hits.mean():.2f} "
        f"hits_max={hits.max():.0f} "
        f"hit_cam={aggregate['hit_camera_visible_rate']:.3f} "
        f"hit_band={aggregate['hit_camera_lower_band_rate']:.3f} "
        f"hit_vxy={aggregate['mean_hit_vxy']:.3f} "
        f"camera={aggregate['camera_visible_rate']:.3f} "
        f"zideal={aggregate['ball_view_z_ideal_rate']:.3f} "
        f"missing={aggregate['ball_obs_missing_refresh_rate']:.3f} "
        f"lost={aggregate['ball_obs_lost_rate']:.3f}"
    )

def done_reason_summary(rows: list[dict[str, float]]) -> str:
    if not rows:
        return "none"
    keys = sorted({k for row in rows for k in row if k.startswith("done/")})
    parts = []
    for key in keys:
        count = sum(1 for row in rows if float(row.get(key, 0.0)) > 0.5)
        if count > 0:
            parts.append(f"{key.removeprefix('done/')}={count}")
    truncated = sum(1 for row in rows if float(row.get("truncated", 0.0)) > 0.5)
    if truncated > 0:
        parts.append(f"truncated={truncated}")
    return ", ".join(parts) if parts else "none"


def row_done_reasons(row: dict[str, float]) -> str:
    reasons = [k.removeprefix("done/") for k, v in row.items() if k.startswith("done/") and float(v) > 0.5]
    if float(row.get("truncated", 0.0)) > 0.5:
        reasons.append("truncated")
    return ",".join(reasons) if reasons else "none"


def trace_row_from_host(
    host: dict[str, np.ndarray],
    *,
    env_i: int,
    step_idx: int,
    episode_idx: int,
    dt: float,
) -> dict[str, float]:
    row: dict[str, float] = {
        "step": int(step_idx),
        "time_sec": float(step_idx * dt),
        "env": int(env_i),
        "episode": int(episode_idx),
        "episode_step": float(host["episode_step"][env_i]),
        "reward": float(host["reward"][env_i]),
        "return": float(host["episode_return"][env_i]),
        "hits": float(host["hit_count"][env_i]),
        "new_hit": float(host["new_hit"][env_i]),
        "in_contact": float(host["in_contact"][env_i]),
        "done": float(host["done"][env_i]),
        "terminated": float(host["terminated"][env_i]),
        "truncated": float(host["truncated"][env_i]),
        "ball_z": float(host["ball_z"][env_i]),
        "racket_z": float(host["racket_z"][env_i]),
        "racket_z_rel": float(host["racket_z_rel"][env_i]),
        "action_norm": float(host["action_norm"][env_i]),
    }
    add_camera_columns(row, host, env_i)
    # Keep task-space motion in the compact action trace as well.  This makes
    # paired real-ball replays directly usable for circle/path diagnostics
    # without requiring a second, much wider observation trace.
    obs = np.asarray(host["obs"][env_i], dtype=np.float32).reshape(-1)
    for axis_i, axis in enumerate(("x", "y", "z")):
        row[f"obs_ball_pos_base_m/{axis}"] = float(obs[20 + axis_i])
        row[f"obs_ball_vel_base_m_s/{axis}"] = float(obs[23 + axis_i])
        row[f"obs_racket_pos_base_m/{axis}"] = float(obs[26 + axis_i])
        row[f"obs_racket_vel_base_m_s/{axis}"] = float(obs[29 + axis_i])
    vector_keys = [
        "policy_mean",
        "raw_action",
        "applied_action",
        "effective_action",
        "desired_qdd_raw",
        "desired_qdd",
        "arm_cmd_q",
        "arm_cmd_qvel",
        "arm_cmd_qdd",
        "arm_raw_comp_q",
        "arm_safe_q",
        "arm_safe_qvel",
        "arm_safe_qacc",
        "arm_servo_target_unlimited",
        "arm_applied_q",
        "arm_applied_qvel",
        "arm_applied_qacc",
        "arm_q",
        "arm_qvel",
        "arm_qacc",
        "arm_qacc_mj",
    ]
    for i, joint in enumerate(RIGHT_ARM_JOINTS):
        safe_joint = joint.replace("/", "_")
        for key in vector_keys:
            value = float(host[key][env_i, i])
            row[f"{key}/{safe_joint}"] = value
            if key in {
                "arm_cmd_q",
                "arm_raw_comp_q",
                "arm_safe_q",
                "arm_servo_target_unlimited",
                "arm_applied_q",
                "arm_q",
            }:
                row[f"{key}_deg/{safe_joint}"] = float(np.rad2deg(value))
            elif key in {"arm_cmd_qvel", "arm_safe_qvel", "arm_applied_qvel", "arm_qvel"}:
                row[f"{key}_deg_s/{safe_joint}"] = float(np.rad2deg(value))
            elif key in {
                "desired_qdd_raw",
                "desired_qdd",
                "arm_cmd_qdd",
                "arm_safe_qacc",
                "arm_applied_qacc",
                "arm_qacc",
                "arm_qacc_mj",
            }:
                row[f"{key}_deg_s2/{safe_joint}"] = float(np.rad2deg(value))
    base_names = ("x", "y", "z", "roll", "pitch", "yaw")
    for i, name in enumerate(base_names):
        q_value = float(host["base_q6"][env_i, i])
        dq_value = float(host["base_qvel6"][env_i, i])
        row[f"base_q6/{name}"] = q_value
        row[f"base_qvel6/{name}"] = dq_value
        if name in {"roll", "pitch", "yaw"}:
            row[f"base_q6_deg/{name}"] = float(np.rad2deg(q_value))
            row[f"base_qvel6_deg_s/{name}"] = float(np.rad2deg(dq_value))
    for key, value in host.items():
        if (
            key.startswith("done/")
            or key.startswith("arm_post_comp_safety_")
            or key.startswith("arm_servo_safety_")
            or (
                key.startswith("arm_actual_safety_")
                and key != "arm_actual_safety_clip_count"
            )
            or key.startswith("raw_action_clip_")
            or key.startswith("reset_")
            or key.startswith("ball_obs_")
            or key.startswith("hit_camera_")
            or key
            in {
                "racket_up_cos",
                "hit_racket_up_cos_sum",
                "racket_flatness_pen",
                "racket_tilt_angular_speed_rad_s",
                "racket_spin_angular_speed_rad_s",
            }
            or key.startswith("racket_angular_velocity_")
            or key in {"ball_view_z_high_exceeded", "ball_view_in_bounds", "ball_view_z_ideal"}
        ):
            row[key] = float(value[env_i])
    return row


def obs_row_from_host(
    host: dict[str, np.ndarray],
    *,
    env_i: int,
    step_idx: int,
    episode_idx: int,
    dt: float,
) -> dict[str, float]:
    obs = np.asarray(host["obs"][env_i], dtype=np.float32).reshape(-1)
    if obs.shape[0] < 50:
        raise RuntimeError(f"expected obs dim at least 50, got {obs.shape[0]}")

    row: dict[str, float] = {
        "step": int(step_idx),
        "time_sec": float(step_idx * dt),
        "env": int(env_i),
        "episode": int(episode_idx),
        "episode_step": float(host["episode_step"][env_i]),
        "reward": float(host["reward"][env_i]),
        "return": float(host["episode_return"][env_i]),
        "hits": float(host["hit_count"][env_i]),
        "done": float(host["done"][env_i]),
        "terminated": float(host["terminated"][env_i]),
        "truncated": float(host["truncated"][env_i]),
    }
    add_camera_columns(row, host, env_i)
    for key, value in host.items():
        if key.startswith("ball_obs_") or key.startswith("reset_") or key.startswith("hit_camera_") or key in {"ball_view_z_high_exceeded", "ball_view_in_bounds", "ball_view_z_ideal"}:
            row[key] = float(value[env_i])
    for i, value in enumerate(obs):
        row[f"obs/{i:03d}"] = float(value)
        if i >= 50:
            row[f"obs_extra/{i - 50:03d}"] = float(value)

    axes = ("x", "y", "z")
    base_names = ("x", "y", "yaw")
    base_vel_names = ("vx", "vy", "yaw_rate")

    def add_joint_block(prefix: str, start: int, *, deg_prefix: str | None = None) -> None:
        for j, joint in enumerate(RIGHT_ARM_JOINTS):
            value = float(obs[start + j])
            row[f"{prefix}/{joint}"] = value
            if deg_prefix is not None:
                row[f"{deg_prefix}/{joint}"] = float(np.rad2deg(value))

    add_joint_block("obs_arm_q_rad", 0, deg_prefix="obs_arm_q_deg")
    add_joint_block("obs_arm_dq_rad_s", 7, deg_prefix="obs_arm_dq_deg_s")
    for j, name in enumerate(base_names):
        row[f"obs_base_q/{name}"] = float(obs[14 + j])
    row["obs_base_q_yaw_deg"] = float(np.rad2deg(obs[16]))
    for j, name in enumerate(base_vel_names):
        row[f"obs_base_dq/{name}"] = float(obs[17 + j])
    row["obs_base_dq_yaw_rate_deg_s"] = float(np.rad2deg(obs[19]))
    for j, axis in enumerate(axes):
        row[f"obs_ball_pos_base_m/{axis}"] = float(obs[20 + j])
        row[f"obs_ball_vel_base_m_s/{axis}"] = float(obs[23 + j])
        row[f"obs_racket_pos_base_m/{axis}"] = float(obs[26 + j])
        row[f"obs_racket_vel_base_m_s/{axis}"] = float(obs[29 + j])
        row[f"obs_rel_base_m/{axis}"] = float(obs[32 + j])
    add_joint_block("obs_prev_action", 35)
    add_joint_block("obs_arm_cmd_error_rad", 42, deg_prefix="obs_arm_cmd_error_deg")
    row["obs_ball_age_norm"] = float(obs[49])
    return row


def add_camera_columns(row: dict[str, float], host: dict[str, np.ndarray], env_i: int) -> None:
    scalar_keys = [
        "camera_available",
        "camera_in_front",
        "camera_in_depth",
        "camera_in_frustum",
        "camera_in_image",
        "camera_visible",
        "ball_cam_x",
        "ball_cam_y",
        "ball_cam_z",
        "ball_cam_distance",
        "ball_pixel_u",
        "ball_pixel_v",
        "camera_world_x",
        "camera_world_y",
        "camera_world_z",
        "ball_world_x",
        "ball_world_y",
        "ball_world_z",
    ]
    for key in scalar_keys:
        if key in host:
            row[key] = float(host[key][env_i])


def plot_trace_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required for --action-plot-out. Install with: python -m pip install matplotlib") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray([row["time_sec"] for row in rows], dtype=float)
    done_t = [row["time_sec"] for row in rows if float(row.get("done", 0.0)) > 0.5]
    fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

    for joint in RIGHT_ARM_JOINTS:
        name = joint.replace("/", "_")
        color = JOINT_PLOT_COLORS.get(joint)
        axes[0].plot(
            t,
            [row[f"raw_action/{name}"] for row in rows],
            linewidth=0.7,
            linestyle=":",
            alpha=0.35,
            color=color,
        )
        axes[0].plot(
            t,
            [row[f"applied_action/{name}"] for row in rows],
            linewidth=1.25,
            color=color,
            label=joint,
        )
    axes[0].set_ylabel("policy action")
    axes[0].axhline(1.0, color="black", linewidth=0.9, alpha=0.65)
    axes[0].axhline(-1.0, color="black", linewidth=0.9, alpha=0.65)
    axes[0].set_title(
        "Raw Gaussian samples (:), applied normalized acceleration actions (solid)"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    acc_limits_deg_s2 = np.asarray(
        [1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0],
        dtype=float,
    )
    vel_limits_deg_s = np.asarray(
        [210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0],
        dtype=float,
    )

    for joint_idx, joint in enumerate(RIGHT_ARM_JOINTS):
        name = joint.replace("/", "_")
        color = JOINT_PLOT_COLORS.get(joint)
        acc_limit = float(acc_limits_deg_s2[joint_idx])
        axes[1].plot(
            t,
            [row[f"desired_qdd_deg_s2/{name}"] / acc_limit for row in rows],
            linewidth=1.25,
            color=color,
            label=joint,
        )
        qacc_key = f"arm_qacc_deg_s2/{name}"
        applied_qacc_key = f"arm_applied_qacc_deg_s2/{name}"
        if applied_qacc_key in rows[0]:
            axes[1].plot(
                t,
                [row[applied_qacc_key] / acc_limit for row in rows],
                linewidth=0.75,
                color=color,
                linestyle=":",
                alpha=0.65,
            )
        if qacc_key in rows[0]:
            axes[1].plot(
                t,
                [row[qacc_key] / acc_limit for row in rows],
                linewidth=0.9,
                color=color,
                linestyle="--",
                alpha=0.75,
            )
    axes[1].axhline(1.0, color="black", linewidth=0.9, alpha=0.65)
    axes[1].axhline(-1.0, color="black", linewidth=0.9, alpha=0.65)
    axes[1].set_ylabel("acceleration / joint limit")
    axes[1].set_title(
        "Nominal qdd (solid), servo target (:), simulated 200 Hz actual (--); raw compensation is in CSV"
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    for joint_idx, joint in enumerate(RIGHT_ARM_JOINTS):
        name = joint.replace("/", "_")
        color = JOINT_PLOT_COLORS.get(joint)
        vel_limit = float(vel_limits_deg_s[joint_idx])
        axes[2].plot(
            t,
            [row[f"arm_cmd_qvel_deg_s/{name}"] / vel_limit for row in rows],
            linewidth=1.25,
            color=color,
            label=joint,
        )
        applied_qvel_key = f"arm_applied_qvel_deg_s/{name}"
        if applied_qvel_key in rows[0]:
            axes[2].plot(
                t,
                [row[applied_qvel_key] / vel_limit for row in rows],
                linewidth=0.75,
                color=color,
                linestyle=":",
                alpha=0.65,
            )
        axes[2].plot(
            t,
            [row[f"arm_qvel_deg_s/{name}"] / vel_limit for row in rows],
            linewidth=0.9,
            color=color,
            linestyle="--",
            alpha=0.75,
        )
    axes[2].axhline(1.0, color="black", linewidth=0.9, alpha=0.65)
    axes[2].axhline(-1.0, color="black", linewidth=0.9, alpha=0.65)
    axes[2].set_ylabel("velocity / joint limit")
    axes[2].set_title(
        "Nominal velocity (solid), servo target (:), simulated 200 Hz actual (--)"
    )
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)

    for joint in RIGHT_ARM_JOINTS:
        name = joint.replace("/", "_")
        color = JOINT_PLOT_COLORS.get(joint)
        axes[3].plot(
            t,
            [row[f"arm_cmd_q_deg/{name}"] for row in rows],
            linewidth=1.25,
            color=color,
            label=joint,
        )
        axes[3].plot(
            t,
            [row[f"arm_q_deg/{name}"] for row in rows],
            linewidth=0.9,
            color=color,
            linestyle="--",
            alpha=0.75,
        )
        applied_key = f"arm_applied_q_deg/{name}"
        if applied_key in rows[0]:
            axes[3].plot(
                t,
                [row[applied_key] for row in rows],
                linewidth=0.8,
                color=color,
                linestyle=":",
                alpha=0.85,
            )
    axes[3].set_ylabel("joint angle deg")
    axes[3].set_title("Commanded targets (solid), actuator-applied targets (dotted), simulated joint angles (dashed)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)

    axes[4].plot(t, [row["ball_z"] for row in rows], label="ball_z", linewidth=1.4, color="#D55E00")
    axes[4].plot(t, [row["racket_z"] for row in rows], label="racket_z", linewidth=1.4, color="#0072B2")
    axes[4].plot(t, [row["racket_z_rel"] for row in rows], label="racket_z_rel", linewidth=1.2, color="#009E73")
    axes[4].set_xlabel("time sec")
    axes[4].set_ylabel("height m")
    axes[4].set_title("Ball/racket height")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right", fontsize=8)

    for ax in axes:
        for x in done_t:
            ax.axvline(x, color="black", alpha=0.15, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gpu_temperature_stop_reason(limit_c: float) -> str | None:
    if limit_c <= 0.0:
        return None
    try:
        process = subprocess.run(
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
    if process.returncode != 0:
        return None
    for line in process.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpu_index = int(parts[0])
            temperature_c = float(parts[1])
        except ValueError:
            continue
        if temperature_c >= limit_c:
            return (
                f"GPU {gpu_index} temperature {temperature_c:.0f}C reached "
                f"--gpu-max-temp-c={limit_c:.0f}C"
            )
    return None


def main() -> None:
    args = parse_args()
    if args.one_episode_per_env and args.episodes != args.n_envs:
        raise SystemExit(
            "--one-episode-per-env requires --episodes equal to --n-envs so every "
            "reset sample is represented exactly once"
        )
    if args.freeze_ball_at_reset and args.ball_replay_csv is not None:
        raise SystemExit("Use only one of --freeze-ball-at-reset and --ball-replay-csv")
    if args.ball_replay_obs_ablation != "none" and args.ball_replay_csv is None:
        raise SystemExit("--ball-replay-obs-ablation requires --ball-replay-csv")
    replay_positions_np = None
    replay_velocities_np = None
    if args.ball_replay_csv is not None:
        if args.n_envs != 1:
            raise SystemExit("--ball-replay-csv currently requires --n-envs 1")
        replay_positions_np, replay_velocities_np = load_real_ball_replay(
            args.ball_replay_csv
        )
        if args.max_env_steps <= 0:
            args.max_env_steps = int(replay_positions_np.shape[0])
        elif args.max_env_steps > replay_positions_np.shape[0]:
            raise SystemExit(
                f"--max-env-steps={args.max_env_steps} exceeds replay length "
                f"{replay_positions_np.shape[0]}"
            )
    payload = load_checkpoint(args.checkpoint)
    env_payload = (
        load_checkpoint(args.env_checkpoint)
        if args.env_checkpoint is not None
        else payload
    )
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    xml_path = args.xml or Path(env_payload.get("xml", RL_SIM_DIR / "moz1_pd.xml"))
    cfg = env_config_from_checkpoint(env_payload, args)
    results_csv = args.results_csv
    if args.results_csv == DEFAULT_RESULTS:
        results_csv = args.checkpoint.parent / "validation.csv"

    if args.render and args.n_envs != 1:
        print("[validate_mjx] interactive render shows env 0; use --n-envs 1 for smoother visual validation.")
    if args.video_out is not None and args.n_envs != 1:
        print(
            f"[validate_mjx] video records batched env {int(args.video_env)}; "
            "use --n-envs 1 for smoother visual validation."
        )
    trace_enabled = (
        args.action_trace_csv is not None
        or args.action_plot_out is not None
        or args.obs_trace_csv is not None
    )
    if trace_enabled and not (0 <= int(args.trace_env) < int(args.n_envs)):
        raise SystemExit(f"--trace-env must be in [0, {args.n_envs - 1}]")
    if args.video_out is not None and not (0 <= int(args.video_env) < int(args.n_envs)):
        raise SystemExit(f"--video-env must be in [0, {args.n_envs - 1}]")

    env = MjxJuggleEnv(xml_path, n_envs=args.n_envs, cfg=cfg)
    print(f"[validate_mjx] JAX devices: {jax.devices()}")
    print(f"[validate_mjx] checkpoint: {args.checkpoint}")
    if args.env_checkpoint is not None:
        print(f"[validate_mjx] environment checkpoint: {args.env_checkpoint}")
    print(f"[validate_mjx] XML: {xml_path}")
    print(f"[validate_mjx] MJX XML: {env.mjx_xml}")
    print(
        f"[validate_mjx] episodes={args.episodes}, n_envs={args.n_envs}, "
        f"deterministic={args.deterministic}, action_gain={args.action_gain}"
    )
    print(
        "[validate_mjx] env_cfg: "
        f"horizon_sec={cfg.horizon_sec}, max_steps={env.max_steps}, "
        f"mujoco_timestep={env.timestep:.4f}s, frame_skip={cfg.frame_skip}, "
        f"control_dt={env.dt:.4f}s, control_hz={1.0 / env.dt:.1f}Hz, "
        f"right_arm_pd_profile={cfg.right_arm_pd_profile}, "
        f"racket_z_limit_down={cfg.racket_z_hard_limit_down}, "
        f"racket_z_limit_up={cfg.racket_z_hard_limit_up}, "
        f"terminate_on_racket_z_limit={cfg.terminate_on_racket_z_limit}"
    )
    print(
        "[validate_mjx] obs/latency_cfg: "
        f"realistic_s2r_profile={args.realistic_s2r_profile if args.realistic_s2r else 'checkpoint'}, "
        f"obs_dim={env.obs_dim}, high_latency_obs={cfg.high_latency_obs}, "
        f"history_frames={cfg.high_latency_history_frames}, "
        f"ball_obs_rate_hz={cfg.ball_obs_rate_hz}, fractional={cfg.ball_obs_fractional_rate}, "
        f"age_tracks_stale={cfg.ball_obs_age_tracks_stale}, require_camera_visible={cfg.ball_obs_require_camera_visible}, "
        f"camera_missing_prob={cfg.ball_obs_camera_missing_prob}, "
        f"view_missing_prob={cfg.ball_obs_view_bounds_missing_prob}, "
        f"episode_coherent_prob={cfg.ball_obs_missing_episode_coherent_prob}, "
        f"frame_pivot_mode={cfg.ball_obs_frame_pivot_mode}, "
        f"velocity_observer={cfg.ball_obs_velocity_observer_mode}, "
        f"observer_tau_ms={cfg.ball_obs_velocity_observer_tau_ms}, "
        f"observer_max_innovation={cfg.ball_obs_velocity_observer_max_innovation_m_s}, "
        f"joint_observer=({cfg.ball_obs_joint_observer_alpha}, "
        f"{cfg.ball_obs_joint_observer_beta}, "
        f"{cfg.ball_obs_joint_observer_raw_velocity_gain}), "
        f"consistency_gate=({cfg.ball_obs_consistency_gate_threshold_m_s}, "
        f"{cfg.ball_obs_consistency_gate_direction_cosine}, "
        f"{cfg.ball_obs_consistency_gate_min_samples}, "
        f"{cfg.ball_obs_consistency_gate_correction_gain}, "
        f"{cfg.ball_obs_consistency_gate_max_correction_m_s}, "
        f"{cfg.ball_obs_consistency_gate_contact_guard_s}), "
        f"prospective_signal=({cfg.ball_obs_prospective_window_samples}, "
        f"{cfg.ball_obs_prospective_prediction_margin_m}, "
        f"{cfg.ball_obs_prospective_velocity_disagreement_m_s}, "
        f"{cfg.ball_obs_prospective_candidate_gain}, "
        f"{cfg.ball_obs_prospective_max_correction_m_s}, "
        f"{cfg.ball_obs_prospective_max_sample_gap_s}, "
        f"{cfg.ball_obs_prospective_contact_guard_s}), "
        f"obs_latency_steps={cfg.dr_obs_latency_steps_range}, action_latency_steps={cfg.dr_action_latency_steps_range}, "
        f"actuator_cmd_filter={cfg.actuator_cmd_filter}, tau_range={cfg.dr_actuator_cmd_tau_range}, "
        f"gain_range={cfg.dr_actuator_cmd_gain_range}, pos_bias={cfg.ball_obs_nominal_pos_bias_base}"
    )
    print(
        "[validate_mjx] ball_physics_cfg: "
        f"mass_range={cfg.dr_ball_mass_range}, "
        f"solref_time_range={cfg.dr_ball_solref_time_range}, "
        f"solref_damping_range={cfg.dr_ball_solref_damping_range}, "
        f"ball_dr={cfg.dr_randomize_ball}, contact_dr={cfg.dr_randomize_contact}"
    )
    if args.disable_ball_state_dr:
        print(
            "[validate_mjx] ball_state_dr=OFF: "
            f"reset_mode={cfg.ball_reset_mode}, "
            f"target_x={cfg.episode_target_x_range_m}, "
            f"target_y={cfg.episode_target_y_range_m}, "
            f"launch_gap={cfg.racket_launch_surface_gap_range_m}, "
            f"ball_dynamics_dr={cfg.dr_randomize_ball}, "
            f"contact_dr={cfg.dr_randomize_contact}, "
            f"obs_frame_dr={cfg.dr_randomize_ball_obs_frame}, "
            f"obs_noise=({cfg.ball_obs_pos_noise_std}, {cfg.ball_obs_vel_noise_std}); "
            f"preserved actuator_dr={cfg.dr_randomize_actuator}, "
            f"pd_dr={cfg.dr_randomize_pd}, "
            f"racket_mount_dr={cfg.dr_randomize_racket_mount}"
        )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, args.n_envs)
    env_state, obs = jax.jit(env.reset)(reset_keys)
    fixed_ball_position = None
    ball_replay_positions = None
    ball_replay_velocities = None
    reference_ball_xy_base = None
    if args.freeze_ball_at_reset:
        fixed_ball_position = env_state.reset_ball_pos
        env_state, obs = jax.jit(
            lambda state: freeze_ball_state(env, state, fixed_ball_position)
        )(env_state)
        print(
            "[validate_mjx] fixed-ball diagnostic: "
            f"position={np.asarray(jax.device_get(fixed_ball_position))[0].round(6).tolist()}, "
            "velocity=[0, 0, 0], valid observation held constant"
        )
    elif replay_positions_np is not None:
        ball_replay_positions = jnp.asarray(replay_positions_np)
        ball_replay_velocities = jnp.asarray(replay_velocities_np)
        env_state, obs = jax.jit(
            lambda state: override_ball_state(
                env,
                state,
                ball_replay_positions[:1],
                ball_replay_velocities[:1],
                exact_observation_frame=True,
            )
        )(env_state)
        reference_ball_xy_base = obs[:, 20:22]
        obs = ablate_ball_replay_observation(
            obs,
            args.ball_replay_obs_ablation,
            reference_ball_xy_base,
        )
        print(
            "[validate_mjx] real-ball replay: "
            f"csv={args.ball_replay_csv}, samples={replay_positions_np.shape[0]}, "
            f"duration={(replay_positions_np.shape[0] - 1) * env.dt:.3f}s, "
            f"initial_position={replay_positions_np[0].round(6).tolist()}, "
            f"initial_velocity={replay_velocities_np[0].round(6).tolist()}"
        )
        if args.ball_replay_obs_ablation != "none":
            print(
                "[validate_mjx] actor-only replay observation ablation: "
                f"mode={args.ball_replay_obs_ablation}; physical replay unchanged"
            )
    running_return = jnp.zeros((args.n_envs,), dtype=jnp.float32)
    running_length = jnp.zeros((args.n_envs,), dtype=jnp.int32)
    eval_step = make_eval_step(
        env,
        args.deterministic,
        args.action_gain,
        args.ignore_early_done,
        fixed_ball_position=fixed_ball_position,
        ball_replay_positions=ball_replay_positions,
        ball_replay_velocities=ball_replay_velocities,
        ball_replay_obs_ablation=args.ball_replay_obs_ablation,
        reference_ball_xy_base=reference_ball_xy_base,
    )

    viewer_ctx = None
    viewer = None
    render_data = None
    video_writer = None
    video_renderer = None
    next_video_time = 0.0
    video_frame_interval = 0.0
    if args.render:
        import mujoco.viewer

        render_data = mj.MjData(env.mj_model)
        copy_env0_to_mujoco_data(env, env_state, render_data)
        viewer_ctx = mujoco.viewer.launch_passive(env.mj_model, render_data)
        viewer = viewer_ctx.__enter__()
    if args.video_out is not None:
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:
            raise SystemExit("imageio is required for --video-out. Install with: python -m pip install imageio imageio-ffmpeg") from exc

        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            video_writer = imageio.get_writer(str(args.video_out), fps=max(1, int(args.video_fps)))
        except ValueError as exc:
            raise SystemExit(
                f"Could not open video writer for {args.video_out}. "
                "For mp4 install a backend with: python -m pip install imageio-ffmpeg "
                "or conda install -c conda-forge imageio-ffmpeg. "
                "Alternatively save a .gif file."
            ) from exc
        if render_data is None:
            render_data = mj.MjData(env.mj_model)
        video_width = max(64, int(args.video_width))
        video_height = max(64, int(args.video_height))
        old_w, old_h, off_w, off_h = ensure_offscreen_framebuffer(env.mj_model, video_width, video_height)
        if off_w != old_w or off_h != old_h:
            print(f"[validate_mjx] resized offscreen framebuffer: {old_w}x{old_h} -> {off_w}x{off_h}")
        video_renderer = mj.Renderer(
            env.mj_model,
            height=video_height,
            width=video_width,
        )
        video_frame_interval = 1.0 / max(1e-6, float(args.video_fps) * max(1e-6, float(args.video_slowmo)))
        append_video_frame(
            video_writer,
            video_renderer,
            env,
            env_state,
            render_data,
            args.video_camera,
            int(args.video_env),
        )
        next_video_time = video_frame_interval
        print(
            f"[validate_mjx] recording video: {args.video_out} "
            f"(env={int(args.video_env)}, {args.video_width}x{args.video_height}, "
            f"fps={args.video_fps}, slowmo={args.video_slowmo})"
        )

    episode_rows: list[dict[str, float]] = []
    trace_rows: list[dict[str, float]] = []
    obs_trace_rows: list[dict[str, float]] = []
    env_episode_counts = np.zeros((args.n_envs,), dtype=np.int32)
    first_episode_recorded = np.zeros((args.n_envs,), dtype=bool)
    episode_hit_camera_events = np.zeros((args.n_envs,), dtype=np.float64)
    episode_hit_camera_visible = np.zeros((args.n_envs,), dtype=np.float64)
    episode_hit_camera_in_margin = np.zeros((args.n_envs,), dtype=np.float64)
    episode_hit_camera_in_band = np.zeros((args.n_envs,), dtype=np.float64)
    episode_hit_camera_v_frac_sum = np.zeros((args.n_envs,), dtype=np.float64)
    episode_hit_metric_sources = (
        "hit_vxy_sum",
        "hit_vxy_sq_sum",
        "hit_racket_vxy_sum",
        "hit_racket_vxy_sq_sum",
        "hit_cycle_eligible",
        "hit_cycle_racket_xy_path_excess_m",
        "hit_cycle_racket_xy_area_m2",
        "racket_cycle_motion_active",
        "racket_cycle_vxy_m_s",
        "hit_contact_center_dist_sum",
        "hit_racket_up_cos_sum",
        "hit_apex_rel_height_sum",
        "hit_apex_view_x_sum",
        "hit_apex_view_y_sum",
        "hit_next_contact_anchor_err_sum",
        "hit_posterior_contact_event",
        "hit_posterior_contact_anchor_err_sum",
        "hit_contact_anchor_contraction_sum",
    )
    episode_hit_metric_sums = {
        name: np.zeros((args.n_envs,), dtype=np.float64)
        for name in episode_hit_metric_sources
    }
    episode_hit_metric_sums.update(
        {
            name: np.zeros((args.n_envs,), dtype=np.float64)
            for name in (
                "first_hit_event",
                "first_hit_racket_vxy_sum",
                "first_hit_racket_vxy_sq_sum",
                "first_hit_ball_vxy_sum",
                "recurrent_hit_event",
                "recurrent_hit_racket_vxy_sum",
                "recurrent_hit_racket_vxy_sq_sum",
                "recurrent_hit_ball_vxy_sum",
            )
        }
    )
    # Deployment diagnostic: measure the first successful lift separately from
    # the later steady-state juggling hits.  The predicted relative apex is the
    # same quantity used by the environment at the hit event; the observed apex
    # is the maximum true world-z reached before the second counted hit.
    episode_first_hit_predicted_apex_rel = np.full(
        (args.n_envs,), np.nan, dtype=np.float64
    )
    episode_first_hit_contact_ball_z = np.full(
        (args.n_envs,), np.nan, dtype=np.float64
    )
    episode_first_hit_observed_apex_z = np.full(
        (args.n_envs,), np.nan, dtype=np.float64
    )
    # Populated lazily from the first JIT result because reward term names are
    # configuration-dependent.  These are true episode sums; terminal values
    # remain available under last/reward/*.
    episode_reward_sums: dict[str, np.ndarray] = {}
    episode_step_metric_sources = {
        "physical_contact_edges": "physical_contact_edge",
        "launch_clearance_crossings": "launch_clearance_crossing",
        "low_survival_launches": "low_survival_launch",
        "subfloor_launches": "subfloor_launch",
        "confirmed_hit_events": "confirmed_hit",
        "failed_hit_events": "failed_hit",
        "ignored_fast_hit_events": "ignored_fast_hit",
        "camera_visible_steps": "camera_visible",
        "ball_view_in_bounds_steps": "ball_view_in_bounds",
        "ball_view_z_ideal_steps": "ball_view_z_ideal",
        "ball_view_z_high_exceeded_steps": "ball_view_z_high_exceeded",
        "ball_obs_refresh_count": "ball_obs_refresh_due",
        "ball_obs_missing_refresh_count": "ball_obs_missing_on_refresh",
        "ball_obs_missing_streak_start_count": "ball_obs_missing_streak_started",
        "ball_obs_lost_steps": "ball_obs_lost_active",
        "ball_obs_lost_entry_count": "ball_obs_lost_entered",
        "ball_obs_stale_steps": "ball_obs_stale_active",
        "ball_obs_dropout_steps": "ball_obs_dropout_active",
        "ball_obs_reacquired_after_missing_events": (
            "ball_obs_reacquired_after_missing"
        ),
        "ball_obs_reacquired_after_lost_events": "ball_obs_reacquired_after_lost",
        "ball_obs_consistency_gate_active_steps": (
            "ball_obs_consistency_gate_active"
        ),
        "ball_obs_consistency_correction_vxy_sum": (
            "ball_obs_consistency_correction_vxy"
        ),
        "ball_obs_consistency_innovation_vxy_sum": (
            "ball_obs_consistency_innovation_vxy"
        ),
        "ball_obs_consistency_contact_guard_steps": (
            "ball_obs_consistency_contact_guard"
        ),
        "ball_obs_consistency_oracle_pre_error_vxy_sum": (
            "ball_obs_consistency_oracle_pre_error_vxy"
        ),
        "ball_obs_consistency_oracle_post_error_vxy_sum": (
            "ball_obs_consistency_oracle_post_error_vxy"
        ),
        "ball_obs_consistency_oracle_helped_events": (
            "ball_obs_consistency_oracle_helped"
        ),
        "ball_obs_consistency_oracle_harmed_events": (
            "ball_obs_consistency_oracle_harmed"
        ),
        "ball_obs_prospective_score_valid_events": (
            "ball_obs_prospective_score_valid"
        ),
        "ball_obs_prospective_proposal_events": (
            "ball_obs_prospective_proposal"
        ),
        "ball_obs_prospective_raw_prediction_error_m_sum": (
            "ball_obs_prospective_raw_prediction_error_m"
        ),
        "ball_obs_prospective_model_prediction_error_m_sum": (
            "ball_obs_prospective_model_prediction_error_m"
        ),
        "ball_obs_prospective_model_advantage_m_sum": (
            "ball_obs_prospective_model_advantage_m"
        ),
        "ball_obs_prospective_correction_vxy_sum": (
            "ball_obs_prospective_correction_vxy"
        ),
        "ball_obs_prospective_oracle_pre_error_vxy_sum": (
            "ball_obs_prospective_oracle_pre_error_vxy"
        ),
        "ball_obs_prospective_oracle_candidate_error_vxy_sum": (
            "ball_obs_prospective_oracle_candidate_error_vxy"
        ),
        "ball_obs_prospective_oracle_model_error_vxy_sum": (
            "ball_obs_prospective_oracle_model_error_vxy"
        ),
        "ball_obs_prospective_oracle_helped_events": (
            "ball_obs_prospective_oracle_helped"
        ),
        "ball_obs_prospective_oracle_harmed_events": (
            "ball_obs_prospective_oracle_harmed"
        ),
        "ball_obs_prospective_oracle_direction_aligned_events": (
            "ball_obs_prospective_oracle_direction_aligned"
        ),
        "ball_obs_prospective_oracle_wrong_direction_events": (
            "ball_obs_prospective_oracle_wrong_direction"
        ),
        "ball_obs_prospective_oracle_aligned_but_harmed_events": (
            "ball_obs_prospective_oracle_aligned_but_harmed"
        ),
        "ball_obs_prospective_oracle_model_helped_events": (
            "ball_obs_prospective_oracle_model_helped"
        ),
        "ball_obs_prospective_oracle_model_harmed_events": (
            "ball_obs_prospective_oracle_model_harmed"
        ),
        "ball_obs_prospective_oracle_pre_error_lt_correction_events": (
            "ball_obs_prospective_oracle_pre_error_lt_correction"
        ),
        "ball_obs_prospective_oracle_pre_error_lt_half_correction_events": (
            "ball_obs_prospective_oracle_pre_error_lt_half_correction"
        ),
        "ball_obs_prospective_proposal_before_first_hit_events": (
            "ball_obs_prospective_proposal_before_first_hit"
        ),
        "ball_obs_prospective_proposal_posthit_060_120ms_events": (
            "ball_obs_prospective_proposal_posthit_060_120ms"
        ),
        "ball_obs_prospective_proposal_posthit_120_250ms_events": (
            "ball_obs_prospective_proposal_posthit_120_250ms"
        ),
        "ball_obs_prospective_proposal_posthit_after_250ms_events": (
            "ball_obs_prospective_proposal_posthit_after_250ms"
        ),
        "ball_obs_prospective_proposal_ascending_events": (
            "ball_obs_prospective_proposal_ascending"
        ),
        "ball_obs_prospective_proposal_descending_events": (
            "ball_obs_prospective_proposal_descending"
        ),
    }
    episode_step_sums = {
        name: np.zeros((args.n_envs,), dtype=np.float64)
        for name in episode_step_metric_sources
    }
    max_steps = args.max_env_steps
    if max_steps <= 0:
        max_steps = int(np.ceil(args.episodes / max(1, args.n_envs)) * env.max_steps * 2)
        max_steps = max(max_steps, env.max_steps)

    try:
        t0 = time.perf_counter()
        for step_idx in range(1, max_steps + 1):
            env_state, obs, rng, running_return, running_length, metrics = eval_step(
                params,
                env_state,
                obs,
                rng,
                running_return,
                running_length,
            )
            host = jax.device_get(metrics)
            observation_host = np.asarray(jax.device_get(obs))
            if not np.all(np.isfinite(observation_host)):
                bad_count = int(np.size(observation_host) - np.isfinite(observation_host).sum())
                raise FloatingPointError(
                    f"non-finite actor observation at validation step {step_idx}: "
                    f"{bad_count} values"
                )
            for critical_name in (
                "reward/total",
                "action_norm",
                "ball_z",
                "racket_z",
                "hit_count",
            ):
                if critical_name not in host:
                    continue
                critical_value = np.asarray(host[critical_name])
                if not np.all(np.isfinite(critical_value)):
                    raise FloatingPointError(
                        f"non-finite {critical_name} at validation step {step_idx}"
                    )
            if step_idx % max(1, int(args.gpu_check_every_steps)) == 0:
                temperature_reason = gpu_temperature_stop_reason(
                    float(args.gpu_max_temp_c)
                )
                if temperature_reason is not None:
                    raise SystemExit(
                        f"[validate_mjx] thermal safety stop: {temperature_reason}"
                    )
            hit_count = np.asarray(host["hit_count"], dtype=np.float64)
            new_hit = np.asarray(host["new_hit"], dtype=np.float64) > 0.5
            first_hit = new_hit & (hit_count >= 0.5) & (hit_count < 1.5)
            recurrent_hit = new_hit & (hit_count >= 1.5)
            hit_racket_vxy_step = np.asarray(
                host.get("hit_racket_vxy_sum", 0.0), dtype=np.float64
            )
            hit_racket_vxy_sq_step = np.asarray(
                host.get("hit_racket_vxy_sq_sum", 0.0), dtype=np.float64
            )
            hit_ball_vxy_step = np.asarray(
                host.get("hit_vxy_sum", 0.0), dtype=np.float64
            )
            for prefix, phase_mask in (
                ("first_hit", first_hit),
                ("recurrent_hit", recurrent_hit),
            ):
                episode_hit_metric_sums[f"{prefix}_event"] += phase_mask
                episode_hit_metric_sums[f"{prefix}_racket_vxy_sum"] += np.where(
                    phase_mask, hit_racket_vxy_step, 0.0
                )
                episode_hit_metric_sums[f"{prefix}_racket_vxy_sq_sum"] += np.where(
                    phase_mask, hit_racket_vxy_sq_step, 0.0
                )
                episode_hit_metric_sums[f"{prefix}_ball_vxy_sum"] += np.where(
                    phase_mask, hit_ball_vxy_step, 0.0
                )
            episode_hit_camera_events += np.asarray(host.get("hit_camera_event", 0.0), dtype=np.float64)
            episode_hit_camera_visible += np.asarray(host.get("hit_camera_visible_event", 0.0), dtype=np.float64)
            episode_hit_camera_in_margin += np.asarray(host.get("hit_camera_in_margin_event", 0.0), dtype=np.float64)
            episode_hit_camera_in_band += np.asarray(host.get("hit_camera_lower_band_event", 0.0), dtype=np.float64)
            episode_hit_camera_v_frac_sum += np.asarray(host.get("hit_camera_v_frac_sum", 0.0), dtype=np.float64)
            accumulate_named_episode_metrics(
                episode_hit_metric_sums,
                host,
                metric_names=episode_hit_metric_sources,
            )
            accumulate_named_episode_metrics(
                episode_reward_sums,
                host,
                prefix="reward/",
            )
            ball_z = np.asarray(host["ball_z"], dtype=np.float64)
            episode_first_hit_predicted_apex_rel[first_hit] = np.asarray(
                host.get("hit_apex_rel_height_sum", np.nan), dtype=np.float64
            )[first_hit]
            episode_first_hit_contact_ball_z[first_hit] = ball_z[first_hit]
            tracking_first_apex = (hit_count >= 0.5) & (hit_count < 1.5)
            current_apex = episode_first_hit_observed_apex_z[tracking_first_apex]
            episode_first_hit_observed_apex_z[tracking_first_apex] = np.where(
                np.isnan(current_apex),
                ball_z[tracking_first_apex],
                np.maximum(current_apex, ball_z[tracking_first_apex]),
            )
            for accumulator_name, metric_name in episode_step_metric_sources.items():
                episode_step_sums[accumulator_name] += np.asarray(
                    host.get(metric_name, 0.0),
                    dtype=np.float64,
                )
            done = np.asarray(host["done"], dtype=bool)
            if trace_enabled:
                trace_env = int(args.trace_env)
                if args.action_trace_csv is not None or args.action_plot_out is not None:
                    trace_rows.append(
                        trace_row_from_host(
                            host,
                            env_i=trace_env,
                            step_idx=step_idx,
                            episode_idx=int(env_episode_counts[trace_env]) + 1,
                            dt=env.dt,
                        )
                    )
                if args.obs_trace_csv is not None:
                    obs_trace_rows.append(
                        obs_row_from_host(
                            host,
                            env_i=trace_env,
                            step_idx=step_idx,
                            episode_idx=int(env_episode_counts[trace_env]) + 1,
                            dt=env.dt,
                        )
                    )

            if args.log_hit_events:
                hit_envs = np.flatnonzero(np.asarray(host["new_hit"]) > 0.5)
                for env_i in hit_envs:
                    print(
                        f"[hit] step={step_idx} env={env_i} "
                        f"hits={float(host['hit_count'][env_i]):.0f} "
                        f"ball_z={float(host['ball_z'][env_i]):.3f}"
                    )

            for env_i in np.flatnonzero(done):
                if args.one_episode_per_env and first_episode_recorded[env_i]:
                    continue
                first_episode_recorded[env_i] = True
                env_episode_counts[env_i] += 1
                row = {
                    "episode": len(episode_rows) + 1,
                    "env": int(env_i),
                    "env_episode": int(env_episode_counts[env_i]),
                    "return": float(host["episode_return"][env_i]),
                    "length": int(host["episode_length"][env_i]),
                    "hits": float(host["hit_count"][env_i]),
                    "last_reward": float(host["reward"][env_i]),
                    "action_norm": float(host["action_norm"][env_i]),
                    "ball_z": float(host["ball_z"][env_i]),
                    "racket_z": float(host["racket_z"][env_i]),
                    "racket_z_rel": float(host["racket_z_rel"][env_i]),
                    "terminated": float(host["terminated"][env_i]),
                    "truncated": float(host["truncated"][env_i]),
                    "episode_step": float(host["episode_step"][env_i]),
                }
                hit_camera_events = float(episode_hit_camera_events[env_i])
                hit_camera_visible = float(episode_hit_camera_visible[env_i])
                hit_camera_in_margin = float(episode_hit_camera_in_margin[env_i])
                hit_camera_in_band = float(episode_hit_camera_in_band[env_i])
                hit_camera_v_frac_sum = float(
                    episode_hit_camera_v_frac_sum[env_i]
                )
                row["hit_camera_events"] = hit_camera_events
                row["hit_camera_visible_events"] = hit_camera_visible
                row["hit_camera_in_margin_events"] = hit_camera_in_margin
                row["hit_camera_lower_band_events"] = hit_camera_in_band
                row["hit_camera_v_frac_sum"] = hit_camera_v_frac_sum
                row["first_hit_predicted_apex_rel_height"] = float(
                    episode_first_hit_predicted_apex_rel[env_i]
                )
                row["first_hit_contact_ball_z"] = float(
                    episode_first_hit_contact_ball_z[env_i]
                )
                row["first_hit_observed_apex_z"] = float(
                    episode_first_hit_observed_apex_z[env_i]
                )
                row["first_hit_observed_lift_m"] = float(
                    episode_first_hit_observed_apex_z[env_i]
                    - episode_first_hit_contact_ball_z[env_i]
                )
                for metric_name, values in episode_hit_metric_sums.items():
                    row[metric_name] = float(values[env_i])
                for metric_name, values in episode_reward_sums.items():
                    row[metric_name] = float(values[env_i])
                row["hit_camera_visible_rate"] = (
                    hit_camera_visible / hit_camera_events
                    if hit_camera_events > 0.0
                    else float("nan")
                )
                row["hit_camera_in_margin_rate"] = (
                    hit_camera_in_margin / hit_camera_events
                    if hit_camera_events > 0.0
                    else float("nan")
                )
                row["hit_camera_lower_band_rate"] = (
                    hit_camera_in_band / hit_camera_events
                    if hit_camera_events > 0.0
                    else float("nan")
                )
                row["mean_hit_camera_v_frac"] = (
                    hit_camera_v_frac_sum / hit_camera_visible
                    if hit_camera_visible > 0.0
                    else float("nan")
                )
                row["mean_hit_vxy"] = (
                    row["hit_vxy_sum"] / hit_camera_events
                    if hit_camera_events > 0.0
                    else float("nan")
                )
                row["rms_hit_vxy"] = (
                    np.sqrt(row["hit_vxy_sq_sum"] / hit_camera_events)
                    if hit_camera_events > 0.0
                    else float("nan")
                )
                row["mean_hit_racket_vxy"] = (
                    row["hit_racket_vxy_sum"] / hit_camera_events
                    if hit_camera_events > 0.0
                    else float("nan")
                )
                row["rms_hit_racket_vxy"] = (
                    np.sqrt(row["hit_racket_vxy_sq_sum"] / hit_camera_events)
                    if hit_camera_events > 0.0
                    else float("nan")
                )
                posterior_events = float(row["hit_posterior_contact_event"])
                row["mean_hit_posterior_contact_anchor_err"] = (
                    row["hit_posterior_contact_anchor_err_sum"]
                    / posterior_events
                    if posterior_events > 0.0
                    else float("nan")
                )
                row["mean_hit_contact_anchor_contraction"] = (
                    row["hit_contact_anchor_contraction_sum"]
                    / posterior_events
                    if posterior_events > 0.0
                    else float("nan")
                )
                for prefix in ("first_hit", "recurrent_hit"):
                    event_count = float(row[f"{prefix}_event"])
                    row[f"mean_{prefix}_racket_vxy"] = (
                        row[f"{prefix}_racket_vxy_sum"] / event_count
                        if event_count > 0.0
                        else float("nan")
                    )
                    row[f"rms_{prefix}_racket_vxy"] = (
                        np.sqrt(row[f"{prefix}_racket_vxy_sq_sum"] / event_count)
                        if event_count > 0.0
                        else float("nan")
                    )
                    row[f"mean_{prefix}_ball_vxy"] = (
                        row[f"{prefix}_ball_vxy_sum"] / event_count
                        if event_count > 0.0
                        else float("nan")
                    )
                cycle_events = float(row["hit_cycle_eligible"])
                row["mean_hit_cycle_racket_xy_path_excess_m"] = (
                    row["hit_cycle_racket_xy_path_excess_m"] / cycle_events
                    if cycle_events > 0.0
                    else float("nan")
                )
                row["mean_hit_cycle_racket_xy_area_m2"] = (
                    row["hit_cycle_racket_xy_area_m2"] / cycle_events
                    if cycle_events > 0.0
                    else float("nan")
                )
                cycle_motion_steps = float(row["racket_cycle_motion_active"])
                row["mean_racket_cycle_vxy"] = (
                    row["racket_cycle_vxy_m_s"] / cycle_motion_steps
                    if cycle_motion_steps > 0.0
                    else float("nan")
                )
                hit_event_count = max(
                    1.0,
                    float(row.get("hits", hit_camera_events)),
                )
                for sum_name, mean_name in (
                    ("hit_contact_center_dist_sum", "mean_hit_contact_center_dist"),
                    ("hit_racket_up_cos_sum", "mean_hit_racket_up_cos"),
                    ("hit_apex_rel_height_sum", "mean_hit_apex_rel_height"),
                    ("hit_apex_view_x_sum", "mean_hit_apex_view_x"),
                    ("hit_apex_view_y_sum", "mean_hit_apex_view_y"),
                    (
                        "hit_next_contact_anchor_err_sum",
                        "mean_hit_next_contact_anchor_err",
                    ),
                ):
                    row[mean_name] = row[sum_name] / hit_event_count
                for accumulator_name, values in episode_step_sums.items():
                    row[accumulator_name] = float(values[env_i])
                row["quality_hits"] = float(row["hits"])
                row["effective_hits"] = (
                    float(row["hits"])
                    + float(row.get("low_survival_launches", 0.0))
                )
                row["quality_fraction_of_effective_hits"] = (
                    row["quality_hits"] / row["effective_hits"]
                    if row["effective_hits"] > 0.0
                    else float("nan")
                )
                episode_length = max(1.0, float(row["length"]))
                refresh_count = row["ball_obs_refresh_count"]
                row["camera_visible_rate"] = (
                    row["camera_visible_steps"] / episode_length
                )
                row["ball_view_in_bounds_rate"] = (
                    row["ball_view_in_bounds_steps"] / episode_length
                )
                row["ball_view_z_ideal_rate"] = (
                    row["ball_view_z_ideal_steps"] / episode_length
                )
                row["ball_view_z_high_exceeded_rate"] = (
                    row["ball_view_z_high_exceeded_steps"] / episode_length
                )
                row["ball_view_z_high_exceeded_ever"] = float(
                    row["ball_view_z_high_exceeded_steps"] > 0.0
                )
                row["ball_obs_missing_refresh_rate"] = (
                    row["ball_obs_missing_refresh_count"] / refresh_count
                    if refresh_count > 0.0
                    else float("nan")
                )
                row["ball_obs_lost_rate"] = (
                    row["ball_obs_lost_steps"] / episode_length
                )
                row["ball_obs_stale_rate"] = (
                    row["ball_obs_stale_steps"] / episode_length
                )
                row["ball_obs_dropout_rate"] = (
                    row["ball_obs_dropout_steps"] / episode_length
                )
                add_terminal_step_metrics(row, host, env_i)
                reset_missing_exposure = float(
                    row.get("reset_ball_obs_missing", 0.0)
                )
                lost_timeout_s = (
                    max(0.0, float(env.cfg.lost_ball_timeout_ms)) * 1e-3
                )
                reset_missing_age = min(
                    float(env.cfg.ball_obs_age_clip),
                    max(float(env.dt), lost_timeout_s),
                )
                reset_lost_exposure = (
                    reset_missing_exposure
                    if lost_timeout_s > 0.0
                    and reset_missing_age >= lost_timeout_s
                    else 0.0
                )
                row["ball_obs_missing_exposure_count"] = (
                    row["ball_obs_missing_streak_start_count"]
                    + reset_missing_exposure
                )
                row["ball_obs_lost_exposure_count"] = (
                    row["ball_obs_lost_entry_count"] + reset_lost_exposure
                )
                missing_exposure_count = row[
                    "ball_obs_missing_exposure_count"
                ]
                lost_exposure_count = row["ball_obs_lost_exposure_count"]
                row["ball_obs_reacquired_after_missing_rate"] = (
                    row["ball_obs_reacquired_after_missing_events"]
                    / missing_exposure_count
                    if missing_exposure_count > 0.0
                    else float("nan")
                )
                row["ball_obs_reacquired_after_lost_rate"] = (
                    row["ball_obs_reacquired_after_lost_events"]
                    / lost_exposure_count
                    if lost_exposure_count > 0.0
                    else float("nan")
                )
                episode_rows.append(row)
                episode_hit_camera_events[env_i] = 0.0
                episode_hit_camera_visible[env_i] = 0.0
                episode_hit_camera_in_margin[env_i] = 0.0
                episode_hit_camera_in_band[env_i] = 0.0
                episode_hit_camera_v_frac_sum[env_i] = 0.0
                episode_first_hit_predicted_apex_rel[env_i] = np.nan
                episode_first_hit_contact_ball_z[env_i] = np.nan
                episode_first_hit_observed_apex_z[env_i] = np.nan
                for values in episode_hit_metric_sums.values():
                    values[env_i] = 0.0
                for values in episode_reward_sums.values():
                    values[env_i] = 0.0
                for values in episode_step_sums.values():
                    values[env_i] = 0.0
                print(
                    f"[episode] ep={row['episode']} env={row['env']} "
                    f"len={row['length']} return={row['return']:.3f} "
                    f"hits={row['hits']:.0f} done={row_done_reasons(row)} "
                    f"last_reward={row['last_reward']:.3f}"
                )
                if len(episode_rows) >= args.episodes:
                    break

            if args.render and viewer is not None and render_data is not None:
                copy_env0_to_mujoco_data(env, env_state, render_data)
                viewer.sync()
                if args.realtime:
                    time.sleep(max(0.0, env.dt * float(args.slowmo)))

            if video_writer is not None and video_renderer is not None and render_data is not None:
                sim_time = step_idx * env.dt
                while sim_time + 1e-9 >= next_video_time:
                    append_video_frame(
                        video_writer,
                        video_renderer,
                        env,
                        env_state,
                        render_data,
                        args.video_camera,
                        int(args.video_env),
                    )
                    next_video_time += video_frame_interval

            if args.print_every > 0 and step_idx % args.print_every == 0:
                done_count = len(episode_rows)
                mean_reward = float(np.mean(np.asarray(host["reward"])))
                mean_hits = float(np.mean(np.asarray(host["hit_count"])))
                elapsed = time.perf_counter() - t0
                sps = step_idx * args.n_envs / max(elapsed, 1e-9)
                print(
                    f"[validate_mjx] step={step_idx} completed={done_count}/{args.episodes} "
                    f"sps={sps:,.0f} reward={mean_reward:.4f} hits={mean_hits:.2f}"
                )

            if len(episode_rows) >= args.episodes:
                break
        else:
            print(f"[validate_mjx] stopped at max_env_steps={max_steps} before completing all episodes")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
        if video_writer is not None:
            video_writer.close()
        if video_renderer is not None:
            video_renderer.close()

    episode_rows = episode_rows[: args.episodes]
    print(f"[validate_mjx] {summarize(episode_rows)}")
    print(f"[validate_mjx] done reasons: {done_reason_summary(episode_rows)}")
    if not args.no_save_csv:
        save_episode_rows(results_csv, episode_rows)
        print(f"[validate_mjx] wrote: {results_csv}")
    if args.action_trace_csv is not None:
        save_trace_rows(args.action_trace_csv, trace_rows)
        print(f"[validate_mjx] wrote action trace: {args.action_trace_csv}")
    if args.obs_trace_csv is not None:
        save_trace_rows(args.obs_trace_csv, obs_trace_rows)
        print(f"[validate_mjx] wrote observation trace: {args.obs_trace_csv}")
    if args.action_plot_out is not None:
        try:
            plot_trace_rows(args.action_plot_out, trace_rows)
        except Exception as exc:
            print(f"[validate_mjx] warning: failed to write action plot {args.action_plot_out}: {exc}", file=sys.stderr)
        else:
            print(f"[validate_mjx] wrote action plot: {args.action_plot_out}")
    if args.video_out is not None:
        print(f"[validate_mjx] wrote video: {args.video_out}")


if __name__ == "__main__":
    main()
