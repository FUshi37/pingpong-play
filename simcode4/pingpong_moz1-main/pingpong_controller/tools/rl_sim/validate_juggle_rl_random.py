from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco as mj
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

# Make sibling modules importable when the script is launched via absolute
# path (e.g. `python3 /path/to/tools/rl_sim/validate_juggle_rl_random.py`).
_RL_SIM_DIR = Path(__file__).resolve().parent
if str(_RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_SIM_DIR))

from rl_juggle_env_random import JuggleEnv, JuggleConfig, RIGHT_ARM_JOINTS

try:
    from config_utils import apply_yaml_defaults
except ModuleNotFoundError:
    # Fallback: no config_utils.py alongside the script. Skip YAML defaults
    # silently; CLI flags still work end-to-end.
    def apply_yaml_defaults(parser, argv, *, section, default_config_path):
        cfg_path = Path(default_config_path)
        if cfg_path.exists():
            print(
                f"[validate] WARN: config_utils.py missing; "
                f"ignoring {cfg_path}",
                file=sys.stderr)
        return None


# Non-ROS standalone validation script. All defaults live under the
# pingpong_controller package; users can still override any path via CLI.
RL_SIM_DIR = _RL_SIM_DIR
PACKAGE_DIR = RL_SIM_DIR.parents[1]  # pingpong_controller/pingpong_controller
OUTPUT_DIR = PACKAGE_DIR / "outputs" / "rl_sim"
DEFAULT_XML = RL_SIM_DIR / "moz1_pd.xml"
DEFAULT_CHECKPOINT = OUTPUT_DIR / "best" / "best_model.zip"
DEFAULT_VIDEO_DIR = OUTPUT_DIR / "videos"
DEFAULT_TRAJ_DIR = OUTPUT_DIR / "traj"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual validate juggling policy with domain randomization")
    p.add_argument("--config", type=Path, default=RL_SIM_DIR / "config.yaml")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help=f"Path to PPO zip (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=1000, help="Base seed for deterministic env resets (matches train eval when train seed=1)")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--slowmo", type=float, default=1.0, help="Realtime slowdown factor (>1.0 is slower, e.g. 2.0)")
    p.add_argument("--print-every", type=int, default=20)
    p.add_argument("--log-hit-events", action="store_true", help="Print a line whenever a new ball-racket hit is detected")
    p.add_argument("--hit-freeze-sec", type=float, default=0.12, help="Extra pause time after each detected hit to make contact easier to observe")
    p.add_argument("--horizon-sec", type=float, default=6.0)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--action-scale-arm-rad", type=float, default=0.025)
    p.add_argument("--action-scale-base-xy", type=float, default=0.020, help="Unused by the right-arm-only policy; kept for CLI compatibility")
    p.add_argument("--action-scale-base-yaw", type=float, default=0.030, help="Unused by the right-arm-only policy; kept for CLI compatibility")
    p.add_argument("--action-gain", type=float, default=1.0, help="Multiply policy action (use >1.0 only for debugging)")
    p.add_argument("--min-action-norm", type=float, default=0.05, help="If action norm is below this, optional fallback motion can be used")
    p.add_argument("--fallback-sine", action="store_true", help="Inject small sinusoidal motion when policy output is near zero")
    p.add_argument("--ball-launch-height", type=float, default=0.32)
    p.add_argument("--ball-spawn-cube-size", type=float, default=0.10, help="Ball spawn cube size (m)")
    p.add_argument("--ball-spawn-xy-jitter", type=float, default=0.025, help="Ball spawn XY jitter (m)")
    p.add_argument("--ball-spawn-z-jitter", type=float, default=0.035, help="Ball spawn Z jitter (m)")
    p.add_argument("--ball-init-vxy-max", type=float, default=0.012, help="Ball initial XY velocity max (m/s)")
    p.add_argument("--ball-init-vz", type=float, default=-0.28, help="Ball initial Z velocity (m/s)")
    p.add_argument("--target-height", type=float, default=0.42)
    p.add_argument("--posture-weight", type=float, default=1.00)
    p.add_argument("--base-pose-weight", type=float, default=0.35)
    p.add_argument("--ball-base-x-penalty-weight", type=float, default=2.0)
    p.add_argument("--ball-base-x-soft-limit", type=float, default=0.04)
    p.add_argument("--ball-base-vxy-penalty-weight", type=float, default=0.60)
    p.add_argument("--torque-penalty-weight", type=float, default=0.0005)
    p.add_argument(
        "--ball-obs-rate-hz",
        type=float,
        default=50.0,
        help="Ball observation refresh rate inside the environment",
    )
    p.add_argument("--ball-obs-pos-noise-std", type=float, default=0.003)
    p.add_argument("--ball-obs-vel-noise-std", type=float, default=0.03)
    p.add_argument("--total-training-steps", type=int, default=10_000_000)
    p.add_argument("--ball-obs-noise-warmup-ratio", type=float, default=0.10)
    p.add_argument("--ball-obs-noise-ramp-ratio", type=float, default=0.20)
    p.add_argument("--camera-visibility-mode", type=str, default="off", choices=["off", "box", "frustum", "pixel"])
    p.add_argument("--camera-center-weight", type=float, default=0.0)
    p.add_argument("--camera-visibility-penalty-weight", type=float, default=0.0)
    p.add_argument("--camera-depth-penalty-weight", type=float, default=0.0)
    p.add_argument("--camera-box-penalty-weight", type=float, default=0.0)
    p.add_argument("--camera-visible-penalty-weight", type=float, default=0.0)
    p.add_argument("--camera-top-margin-penalty-weight", type=float, default=0.0)
    p.add_argument("--camera-pixel-margin", type=float, default=80.0)
    p.add_argument("--camera-min-depth", type=float, default=0.15)
    p.add_argument("--camera-max-depth", type=float, default=2.50)
    p.add_argument("--camera-box-half-width", type=float, default=0.35)
    p.add_argument("--camera-box-half-height", type=float, default=0.35)
    p.add_argument("--camera-box-depth-min", type=float, default=0.20)
    p.add_argument("--camera-box-depth-max", type=float, default=1.50)
    p.add_argument("--no-domain-randomization", action="store_true", help="Disable domain randomization")
    # DR preset for curriculum stages
    p.add_argument("--dr-preset", type=str, default="full", choices=["none", "ball", "ball_contact", "ball_contact_actuator", "full"],
                   help="DR preset: none=off, ball=ball+gravity, ball_contact=+friction+solref, ball_contact_actuator=+action_scale+damping+armature, full=+latency")
    p.add_argument("--dr-ball", dest="dr_randomize_ball", action="store_true", default=None, help="Enable ball DR (ball mass + gravity)")
    p.add_argument("--no-dr-ball", dest="dr_randomize_ball", action="store_false", help="Disable ball DR")
    p.add_argument("--dr-contact", dest="dr_randomize_contact", action="store_true", default=None, help="Enable contact DR (friction + solref)")
    p.add_argument("--no-dr-contact", dest="dr_randomize_contact", action="store_false", help="Disable contact DR")
    p.add_argument("--dr-actuator", dest="dr_randomize_actuator", action="store_true", default=None, help="Enable actuator DR (action scale + damping + armature)")
    p.add_argument("--no-dr-actuator", dest="dr_randomize_actuator", action="store_false", help="Disable actuator DR")
    p.add_argument("--dr-latency", dest="dr_randomize_latency", action="store_true", default=None, help="Enable latency DR (obs + action latency)")
    p.add_argument("--no-dr-latency", dest="dr_randomize_latency", action="store_false", help="Disable latency DR")
    p.add_argument("--dr-ball-mass-min", type=float, default=0.0024)
    p.add_argument("--dr-ball-mass-max", type=float, default=0.0030)
    p.add_argument("--dr-gravity-z-min", type=float, default=-9.90)
    p.add_argument("--dr-gravity-z-max", type=float, default=-9.70)
    p.add_argument("--dr-ball-friction-min", type=float, default=0.12)
    p.add_argument("--dr-ball-friction-max", type=float, default=0.35)
    p.add_argument("--dr-racket-friction-min", type=float, default=0.25)
    p.add_argument("--dr-racket-friction-max", type=float, default=0.55)
    p.add_argument("--dr-ball-solref-time-min", type=float, default=0.002)
    p.add_argument("--dr-ball-solref-time-max", type=float, default=0.006)
    p.add_argument("--dr-ball-solref-damping-min", type=float, default=0.70)
    p.add_argument("--dr-ball-solref-damping-max", type=float, default=0.95)
    p.add_argument("--dr-action-scale-mult-min", type=float, default=0.85)
    p.add_argument("--dr-action-scale-mult-max", type=float, default=1.15)
    p.add_argument("--dr-armature-mult-min", type=float, default=0.80)
    p.add_argument("--dr-armature-mult-max", type=float, default=1.20)
    p.add_argument("--dr-damping-mult-min", type=float, default=0.70)
    p.add_argument("--dr-damping-mult-max", type=float, default=1.30)
    p.add_argument("--dr-obs-latency-min", type=int, default=0)
    p.add_argument("--dr-obs-latency-max", type=int, default=2)
    p.add_argument("--dr-action-latency-min", type=int, default=0)
    p.add_argument("--dr-action-latency-max", type=int, default=2)
    p.add_argument("--log-dr-params", action="store_true", help="Print domain randomization parameters at episode start")
    p.add_argument("--arm-action-limiter", action="store_true", help="Enable hard right-arm action velocity/acceleration limiter")
    p.add_argument("--action-acc-scale", type=float, default=1.0, help="Acceleration-command action scale (controls policy acceleration command magnitude)")
    p.add_argument("--arm-vel-limit-penalty-weight", type=float, default=0.0, help="Penalty weight for arm velocity limit violations")
    p.add_argument("--arm-acc-limit-penalty-weight", type=float, default=0.0, help="Penalty weight for arm acceleration limit violations")
    p.add_argument("--arm-limiter-penalty-weight", type=float, default=0.0, help="Penalty weight for limiter usage (penalizes policy hitting the hard limiter)")
    p.add_argument("--action-penalty-weight", type=float, default=0.003, help="Penalty weight for action magnitude (sum of squared actions)")
    p.add_argument("--action-delta-penalty-weight", type=float, default=0.001, help="Penalty weight for action delta magnitude (sum of squared action changes)")
    p.add_argument("--hit-cadence-reward-weight", type=float, default=0.0, help="Reward weight for hit cadence (Gaussian around target interval)")
    p.add_argument("--hit-cadence-target-interval", type=float, default=0.65, help="Target hit interval in seconds")
    p.add_argument("--hit-cadence-sigma", type=float, default=0.18, help="Hit cadence Gaussian sigma in seconds")
    p.add_argument("--hit-min-interval-penalty-weight", type=float, default=0.0, help="Penalty weight for hits that are too close together")
    p.add_argument("--hit-min-interval", type=float, default=0.40, help="Minimum hit interval in seconds")
    p.add_argument("--hit-min-count-interval", type=float, default=0.0, help="Minimum interval (s) between counted hits. Confirmed hits faster than this are ignored (no hit_count increment, no hit_bonus). 0 disables.")
    p.add_argument("--fast-hit-penalty-weight", type=float, default=0.0, help="Penalty weight for ignored fast hits (confirmed but too close to previous counted hit)")
    p.add_argument("--hit-reward-cap-mode", type=str, default="off", choices=["off", "auto", "fixed"], help="Hit reward cap mode: off (no cap), auto (cap based on episode duration and target interval), fixed (use hit-reward-count-cap value)")
    p.add_argument("--hit-reward-count-cap", type=int, default=0, help="Fixed hit reward count cap (only used when hit-reward-cap-mode=fixed). <=0 means no cap.")
    p.add_argument("--hit-reward-cap-target-interval", type=float, default=0.65, help="Target interval for auto hit reward cap calculation (seconds)")
    p.add_argument("--hit-reward-base", type=float, default=2.5, help="Base hit reward (default 2.5)")
    p.add_argument("--hit-reward-combo", type=float, default=1.2, help="Hit combo reward multiplier (default 1.2)")
    p.add_argument("--center-flat-hit-reward-weight", type=float, default=1.8, help="Center flat hit reward weight (default 1.8)")
    p.add_argument("--ball-obs-dropout-prob", type=float, default=0.0, help="Per-step ball observation dropout probability")
    p.add_argument("--ball-obs-dropout-max-steps", type=int, default=1, help="Max duration of single dropout event (policy steps)")
    p.add_argument("--ball-obs-dropout-burst-prob", type=float, default=0.0, help="Probability of burst (longer) dropout event")
    p.add_argument("--ball-obs-dropout-burst-max-steps", type=int, default=1, help="Max duration of burst dropout (policy steps)")
    p.add_argument("--ball-obs-age-clip", type=float, default=0.20, help="Ball obs age normalization clip (seconds)")
    p.add_argument("--racket-chest-xy-penalty-weight", type=float, default=1.0, help="Penalty weight for racket XY distance from chest target")
    p.add_argument("--racket-chest-z-penalty-weight", type=float, default=0.8, help="Penalty weight for racket Z distance from chest target")
    p.add_argument("--racket-z-soft-penalty-weight", type=float, default=1.2, help="Penalty weight for racket Z band violation")
    p.add_argument("--racket-up-drift-penalty-weight", type=float, default=0.3, help="Penalty weight for racket upward drift")
    p.add_argument("--racket-xy-gauss-reward-weight", type=float, default=0.50, help="Reward weight for racket XY Gaussian anchor")
    p.add_argument("--racket-xy-gauss-penalty-weight", type=float, default=0.60, help="Penalty weight for racket XY Gaussian deviation")
    p.add_argument("--ball-anchor-xy-penalty-weight", type=float, default=0.7, help="Penalty weight for ball XY distance from chest anchor")
    p.add_argument("--ball-vxy-penalty-weight", type=float, default=0.40, help="Penalty weight for ball XY velocity")
    p.add_argument("--arm-vel-limit-deg-s", nargs=7, type=float, default=[210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0], help="Arm velocity limits in deg/s for 7 joints")
    p.add_argument("--arm-acc-limit-deg-s2", nargs=7, type=float, default=[1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0], help="Arm acceleration limits in deg/s^2 for 7 joints")
    p.add_argument("--dr-racket-mount", action="store_true", help="Enable racket mount DR")
    p.add_argument("--dr-racket-pos-offset-mm", type=float, default=0.0, help="Racket position offset in mm")
    p.add_argument("--dr-racket-rot-offset-deg", type=float, default=0.0, help="Racket rotation offset in degrees")
    p.add_argument("--dr-racket-radius-offset-mm", type=float, default=0.0, help="Racket radius offset in mm")
    p.add_argument("--plot-trajectory", action="store_true", help="Plot 3D ball trajectory after each episode")
    p.add_argument("--plot-height", action="store_true", help="Plot ball height (z) over time after each episode")
    p.add_argument("--save-trajectory-dir", type=Path, default=DEFAULT_TRAJ_DIR,
                   help=f"Directory for trajectory figures (default: {DEFAULT_TRAJ_DIR})")
    p.add_argument("--no-show-plot", action="store_true", help="Do not open plot window (useful for headless runs)")
    p.add_argument("--record-video", action="store_true", help="Record each episode as GIF")
    p.add_argument("--plot-right-arm-joints", action="store_true", help="Save right arm joint angle curves after each episode")
    p.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR,
                   help=f"Directory for episode GIFs/MP4s (default: {DEFAULT_VIDEO_DIR})")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-width", type=int, default=640)
    p.add_argument("--video-height", type=int, default=480)
    p.add_argument("--headless", action="store_true", help="Run without live MuJoCo viewer (recommended when recording)")
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="SB3 inference device")
    apply_yaml_defaults(
        p,
        argv,
        section="validate_juggle_rl",
        default_config_path=Path(__file__).resolve().parent / "config.yaml",
    )
    args = p.parse_args(argv)

    DR_PRESETS = {
        "none": (False, False, False, False),
        "ball": (True, False, False, False),
        "ball_contact": (True, True, False, False),
        "ball_contact_actuator": (True, True, True, False),
        "full": (True, True, True, True),
    }

    if args.dr_preset == "none":
        args.no_domain_randomization = True

    preset_ball, preset_contact, preset_actuator, preset_latency = DR_PRESETS[args.dr_preset]

    if args.dr_randomize_ball is None:
        args.dr_randomize_ball = preset_ball
    if args.dr_randomize_contact is None:
        args.dr_randomize_contact = preset_contact
    if args.dr_randomize_actuator is None:
        args.dr_randomize_actuator = preset_actuator
    if args.dr_randomize_latency is None:
        args.dr_randomize_latency = preset_latency

    return args


def _plot_ball_trajectory_3d(
    traj_xyz: list[np.ndarray],
    paddle_xyz: list[np.ndarray],
    ep_index: int,
    target_height: float,
    save_dir: Path | None,
    show_plot: bool,
) -> None:
    if len(traj_xyz) == 0:
        return

    xyz = np.asarray(traj_xyz, dtype=np.float32)
    pxyz = np.asarray(paddle_xyz, dtype=np.float32) if len(paddle_xyz) > 0 else None
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=2.0, color="#ff7f0e", label="ball trajectory")
    if pxyz is not None and pxyz.shape[0] > 1:
        ax.plot(pxyz[:, 0], pxyz[:, 1], pxyz[:, 2], lw=1.8, color="#17becf", alpha=0.9, label="paddle trajectory")
    ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], c="g", s=40, label="start")
    ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], c="r", s=40, label="end")

    x_min, x_max = float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0]))
    y_min, y_max = float(np.min(xyz[:, 1])), float(np.max(xyz[:, 1]))
    z_min, z_max = float(np.min(xyz[:, 2])), float(np.max(xyz[:, 2]))
    pad = 0.03
    xx, yy = np.meshgrid([x_min - pad, x_max + pad], [y_min - pad, y_max + pad])
    zz = np.full_like(xx, float(target_height))
    ax.plot_surface(xx, yy, zz, alpha=0.15, color="#1f77b4", linewidth=0)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Episode {ep_index} Ball 3D Trajectory")
    ax.legend(loc="upper right")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"ball_traj_ep{ep_index:03d}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[TRAJ] saved: {out}")

    if show_plot:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def _plot_ball_height(
    time_s: np.ndarray,
    z_m: np.ndarray,
    paddle_z_m: np.ndarray,
    ep_index: int,
    target_height: float,
    save_dir: Path | None,
    show_plot: bool,
) -> None:
    if time_s.size == 0 or z_m.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(time_s, z_m, lw=2.0, color="#ff7f0e", label="ball z")
    if paddle_z_m.size == time_s.size:
        ax.plot(time_s, paddle_z_m, lw=1.8, color="#17becf", alpha=0.9, label="paddle z")
    ax.axhline(float(target_height), color="#1f77b4", ls="--", lw=1.6, label="target z")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height z (m)")
    ax.set_title(f"Episode {ep_index} Ball Height vs Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"ball_height_ep{ep_index:03d}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[HEIGHT] saved: {out}")

    if show_plot:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def _plot_right_arm_joints(
    time_s: np.ndarray,
    arm_qpos_deg: np.ndarray,
    ep_index: int,
    save_dir: Path | None,
    show_plot: bool,
) -> None:
    if time_s.size == 0 or arm_qpos_deg.shape[0] == 0:
        return

    n_joints = arm_qpos_deg.shape[1]
    fig, axes = plt.subplots(n_joints, 1, figsize=(8.0, 2.2 * n_joints), sharex=True)
    for j in range(n_joints):
        ax = axes[j]
        ax.plot(time_s, arm_qpos_deg[:, j], lw=1.5, color="#1f77b4")
        ax.set_ylabel("angle (deg)")
        ax.set_title(RIGHT_ARM_JOINTS[j])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Episode {ep_index} Right Arm Joint Angles", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"right_arm_joints_ep{ep_index:03d}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[ARM_JOINTS] saved: {out}")

        csv_path = save_dir / f"right_arm_joints_ep{ep_index:03d}.csv"
        header = "time_s," + ",".join(RIGHT_ARM_JOINTS)
        csv_data = np.column_stack([time_s, arm_qpos_deg])
        np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="")
        print(f"[ARM_JOINTS_DATA] saved: {csv_path}")

        npz_path = save_dir / f"right_arm_joints_ep{ep_index:03d}.npz"
        np.savez_compressed(
            npz_path,
            time_s=time_s,
            arm_qpos_deg=arm_qpos_deg,
            joint_names=np.array(RIGHT_ARM_JOINTS),
        )
        print(f"[ARM_JOINTS_DATA] saved: {npz_path}")

    if show_plot:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = JuggleConfig(
        horizon_sec=args.horizon_sec,
        frame_skip=args.frame_skip,
        action_scale_arm_rad=args.action_scale_arm_rad,
        action_scale_base_xy=args.action_scale_base_xy,
        action_scale_base_yaw=args.action_scale_base_yaw,
        ball_launch_height=args.ball_launch_height,
        ball_spawn_cube_size=args.ball_spawn_cube_size,
        ball_spawn_xy_jitter=args.ball_spawn_xy_jitter,
        ball_spawn_z_jitter=args.ball_spawn_z_jitter,
        ball_init_vxy_max=args.ball_init_vxy_max,
        ball_init_vz=args.ball_init_vz,
        ball_obs_rate_hz=args.ball_obs_rate_hz,
        ball_obs_pos_noise_std=args.ball_obs_pos_noise_std,
        ball_obs_vel_noise_std=args.ball_obs_vel_noise_std,
        total_training_steps=args.total_training_steps,
        ball_obs_noise_warmup_ratio=args.ball_obs_noise_warmup_ratio,
        ball_obs_noise_ramp_ratio=args.ball_obs_noise_ramp_ratio,
        target_height=args.target_height,
        posture_weight=args.posture_weight,
        base_pose_weight=args.base_pose_weight,
        ball_base_x_penalty_weight=args.ball_base_x_penalty_weight,
        ball_base_x_soft_limit=args.ball_base_x_soft_limit,
        ball_base_vxy_penalty_weight=args.ball_base_vxy_penalty_weight,
        torque_penalty_weight=args.torque_penalty_weight,
        domain_randomization=not args.no_domain_randomization,
        dr_randomize_ball=args.dr_randomize_ball,
        dr_randomize_contact=args.dr_randomize_contact,
        dr_randomize_actuator=args.dr_randomize_actuator,
        dr_randomize_latency=args.dr_randomize_latency,
        dr_ball_mass_range=(args.dr_ball_mass_min, args.dr_ball_mass_max),
        dr_ball_friction_range=(args.dr_ball_friction_min, args.dr_ball_friction_max),
        dr_racket_friction_range=(args.dr_racket_friction_min, args.dr_racket_friction_max),
        dr_ball_solref_time_range=(args.dr_ball_solref_time_min, args.dr_ball_solref_time_max),
        dr_ball_solref_damping_range=(args.dr_ball_solref_damping_min, args.dr_ball_solref_damping_max),
        dr_gravity_z_range=(args.dr_gravity_z_min, args.dr_gravity_z_max),
        dr_action_scale_mult_range=(args.dr_action_scale_mult_min, args.dr_action_scale_mult_max),
        dr_armature_mult_range=(args.dr_armature_mult_min, args.dr_armature_mult_max),
        dr_damping_mult_range=(args.dr_damping_mult_min, args.dr_damping_mult_max),
        dr_obs_latency_steps_range=(args.dr_obs_latency_min, args.dr_obs_latency_max),
        dr_action_latency_steps_range=(args.dr_action_latency_min, args.dr_action_latency_max),
        camera_visibility_mode=args.camera_visibility_mode,
        camera_center_weight=args.camera_center_weight,
        camera_visibility_penalty_weight=args.camera_visibility_penalty_weight,
        camera_depth_penalty_weight=args.camera_depth_penalty_weight,
        camera_box_penalty_weight=args.camera_box_penalty_weight,
        camera_visible_penalty_weight=args.camera_visible_penalty_weight,
        camera_top_margin_penalty_weight=args.camera_top_margin_penalty_weight,
        camera_pixel_margin=args.camera_pixel_margin,
        camera_min_depth=args.camera_min_depth,
        camera_max_depth=args.camera_max_depth,
        camera_box_half_width=args.camera_box_half_width,
        camera_box_half_height=args.camera_box_half_height,
        camera_box_depth_min=args.camera_box_depth_min,
        camera_box_depth_max=args.camera_box_depth_max,
        arm_action_limiter=args.arm_action_limiter,
        action_acc_scale=args.action_acc_scale,
        arm_vel_limit_penalty_weight=args.arm_vel_limit_penalty_weight,
        arm_acc_limit_penalty_weight=args.arm_acc_limit_penalty_weight,
        arm_limiter_penalty_weight=args.arm_limiter_penalty_weight,
        action_penalty_weight=args.action_penalty_weight,
        action_delta_penalty_weight=args.action_delta_penalty_weight,
        racket_chest_xy_penalty_weight=args.racket_chest_xy_penalty_weight,
        racket_chest_z_penalty_weight=args.racket_chest_z_penalty_weight,
        racket_z_soft_penalty_weight=args.racket_z_soft_penalty_weight,
        racket_up_drift_penalty_weight=args.racket_up_drift_penalty_weight,
        racket_xy_gauss_reward_weight=args.racket_xy_gauss_reward_weight,
        racket_xy_gauss_penalty_weight=args.racket_xy_gauss_penalty_weight,
        ball_anchor_xy_penalty_weight=args.ball_anchor_xy_penalty_weight,
        ball_vxy_penalty_weight=args.ball_vxy_penalty_weight,
        arm_vel_limit_deg_s=tuple(args.arm_vel_limit_deg_s),
        arm_acc_limit_deg_s2=tuple(args.arm_acc_limit_deg_s2),
        dr_randomize_racket_mount=args.dr_racket_mount,
        dr_racket_pos_offset_m=args.dr_racket_pos_offset_mm / 1000.0,
        dr_racket_rot_offset_rad=np.deg2rad(args.dr_racket_rot_offset_deg),
        dr_racket_radius_offset_m=args.dr_racket_radius_offset_mm / 1000.0,
        hit_cadence_reward_weight=args.hit_cadence_reward_weight,
        hit_cadence_target_interval=args.hit_cadence_target_interval,
        hit_cadence_sigma=args.hit_cadence_sigma,
        hit_min_interval_penalty_weight=args.hit_min_interval_penalty_weight,
        hit_min_interval=args.hit_min_interval,
        hit_min_count_interval=args.hit_min_count_interval,
        fast_hit_penalty_weight=args.fast_hit_penalty_weight,
        hit_reward_cap_mode=args.hit_reward_cap_mode,
        hit_reward_count_cap=args.hit_reward_count_cap,
        hit_reward_cap_target_interval=args.hit_reward_cap_target_interval,
        hit_reward_base=args.hit_reward_base,
        hit_reward_combo=args.hit_reward_combo,
        center_flat_hit_reward_weight=args.center_flat_hit_reward_weight,
        ball_obs_dropout_prob=args.ball_obs_dropout_prob,
        ball_obs_dropout_max_steps=args.ball_obs_dropout_max_steps,
        ball_obs_dropout_burst_prob=args.ball_obs_dropout_burst_prob,
        ball_obs_dropout_burst_max_steps=args.ball_obs_dropout_burst_max_steps,
        ball_obs_age_clip=args.ball_obs_age_clip,
    )

    env = JuggleEnv(xml_path=str(args.xml), cfg=cfg)
    model = PPO.load(str(args.checkpoint), env=env, device=args.device)
    episode_hits: list[int] = []
    hit_interval_mean_vals: list[float] = []
    hit_interval_min_vals: list[float] = []
    short_hit_interval_frac_vals: list[float] = []
    hit_rate_hz_mean_vals: list[float] = []
    hit_cadence_rew_mean_vals: list[float] = []
    hit_min_interval_pen_mean_vals: list[float] = []
    ignored_fast_hit_frac_vals: list[float] = []
    ignored_fast_hit_count_vals: list[float] = []
    fast_hit_pen_vals: list[float] = []
    counted_hit_interval_mean_vals: list[float] = []
    counted_hit_interval_min_vals: list[float] = []
    hit_reward_count_cap_vals: list[float] = []
    rewarded_hit_count_vals: list[float] = []
    unrewarded_extra_hit_count_vals: list[float] = []
    hit_reward_cap_reached_vals: list[float] = []
    arm_cmd_tracking_error_norm_vals: list[float] = []
    arm_cmd_tracking_error_max_abs_vals: list[float] = []
    arm_cmd_acc_ratio_mean_vals: list[float] = []
    arm_cmd_acc_ratio_max_vals: list[float] = []
    arm_limiter_clip_frac_vals: list[float] = []
    arm_limiter_pen_vals: list[float] = []
    arm_limiter_penalty_vals: list[float] = []
    arm_vel_ratio_max_vals: list[float] = []
    arm_acc_ratio_max_vals: list[float] = []
    print(f"[INFO] validate seed base={args.seed}", flush=True)
    print(
        f"[INFO] env ball obs rate={float(args.ball_obs_rate_hz):.3f} Hz, "
        f"pos_noise_std={float(args.ball_obs_pos_noise_std):.4f}, "
        f"vel_noise_std={float(args.ball_obs_vel_noise_std):.4f}",
        flush=True,
    )
    print(f"[INFO] domain_randomization={'ENABLED' if cfg.domain_randomization else 'DISABLED'}", flush=True)
    renderer = None
    if args.record_video:
        renderer = mj.Renderer(env.model, height=max(64, int(args.video_height)), width=max(64, int(args.video_width)))

    viewer_ctx = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(env.model, env.data)
    with viewer_ctx as viewer:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=int(args.seed) + ep)

            if args.log_dr_params and cfg.domain_randomization:
                print(f"[EP {ep+1} DR] DR_PRESET: ball={cfg.dr_randomize_ball} contact={cfg.dr_randomize_contact} "
                      f"actuator={cfg.dr_randomize_actuator} latency={cfg.dr_randomize_latency} "
                      f"racket_mount={cfg.dr_randomize_racket_mount}")
                print(f"[EP {ep+1} DR] ball_mass={env.dr_ball_mass:.6f} ball_friction={env.dr_ball_friction:.3f} "
                      f"racket_friction={env.dr_racket_friction:.3f} gravity_z={env.dr_gravity_z:.3f} "
                      f"action_scale_mult={env.action_scale_mult:.3f} "
                      f"obs_latency={env.dr_obs_latency_steps} action_latency={env.dr_action_latency_steps}")
                if cfg.dr_randomize_racket_mount:
                    pos_norm = float(np.linalg.norm(env.dr_racket_pos_offset))
                    rot_norm = float(np.linalg.norm(env.dr_racket_rot_offset))
                    print(f"[EP {ep+1} DR] racket_mount: pos_offset_norm={pos_norm:.6f}m "
                          f"rot_offset_norm={rot_norm:.6f}rad radius_offset={env.dr_racket_radius_offset:.6f}m")

            done = False
            trunc = False
            total_r = 0.0
            step_i = 0
            video_frames: list[np.ndarray] = []
            ball_traj: list[np.ndarray] = []
            paddle_traj: list[np.ndarray] = []
            ball_z: list[float] = []
            paddle_z: list[float] = []
            sim_t: list[float] = []
            arm_qpos_history: list[np.ndarray] = []
            while (viewer is None or viewer.is_running()) and not (done or trunc):
                t0 = time.perf_counter()
                policy_obs = np.asarray(obs, dtype=np.float32)

                action, _ = model.predict(policy_obs, deterministic=args.deterministic)
                action = np.asarray(action, dtype=np.float32).reshape(-1)

                act_norm = float(np.linalg.norm(action))
                if args.fallback_sine and act_norm < float(args.min_action_norm):
                    s = np.zeros_like(action)
                    fallback_indices = [0, 2, 4]
                    fallback_phases = [0.0, 0.6, 1.2]
                    fallback_scales = [0.8, 0.6, 0.4]
                    fallback_freqs = [0.12, 0.17, 0.09]
                    for idx, phase, scale, freq in zip(fallback_indices, fallback_phases, fallback_scales, fallback_freqs):
                        if idx < s.shape[0]:
                            s[idx] = scale * np.sin(freq * step_i + phase)
                    action = s

                action = np.clip(action * float(args.action_gain), -1.0, 1.0)
                act_norm_used = float(np.linalg.norm(action))

                obs, reward, done, trunc, info = env.step(action)
                if "arm_cmd_tracking_error_norm" in info:
                    arm_cmd_tracking_error_norm_vals.append(float(info["arm_cmd_tracking_error_norm"]))
                if "arm_cmd_tracking_error_max_abs" in info:
                    arm_cmd_tracking_error_max_abs_vals.append(float(info["arm_cmd_tracking_error_max_abs"]))
                if "arm_cmd_acc_ratio_mean" in info:
                    arm_cmd_acc_ratio_mean_vals.append(float(info["arm_cmd_acc_ratio_mean"]))
                if "arm_cmd_acc_ratio_max" in info:
                    arm_cmd_acc_ratio_max_vals.append(float(info["arm_cmd_acc_ratio_max"]))
                if "arm_limiter_clip_frac" in info:
                    arm_limiter_clip_frac_vals.append(float(info["arm_limiter_clip_frac"]))
                if "arm_limiter_pen" in info:
                    arm_limiter_pen_vals.append(float(info["arm_limiter_pen"]))
                if "arm_limiter_penalty" in info:
                    arm_limiter_penalty_vals.append(float(info["arm_limiter_penalty"]))
                if "arm_vel_ratio_max" in info:
                    arm_vel_ratio_max_vals.append(float(info["arm_vel_ratio_max"]))
                if "arm_acc_ratio_max" in info:
                    arm_acc_ratio_max_vals.append(float(info["arm_acc_ratio_max"]))
                if "hit_interval_mean" in info:
                    hit_interval_mean_vals.append(float(info["hit_interval_mean"]))
                if "hit_interval_min" in info and float(info["hit_interval_min"]) > 0:
                    hit_interval_min_vals.append(float(info["hit_interval_min"]))
                if "short_hit_interval_frac" in info:
                    short_hit_interval_frac_vals.append(float(info["short_hit_interval_frac"]))
                if "hit_rate_hz_mean" in info:
                    hit_rate_hz_mean_vals.append(float(info["hit_rate_hz_mean"]))
                if "hit_cadence_rew_mean" in info:
                    hit_cadence_rew_mean_vals.append(float(info["hit_cadence_rew_mean"]))
                if "hit_min_interval_pen_mean" in info:
                    hit_min_interval_pen_mean_vals.append(float(info["hit_min_interval_pen_mean"]))
                if "ignored_fast_hit_frac" in info:
                    ignored_fast_hit_frac_vals.append(float(info["ignored_fast_hit_frac"]))
                if "ignored_fast_hit_count" in info:
                    ignored_fast_hit_count_vals.append(float(info["ignored_fast_hit_count"]))
                if "fast_hit_pen" in info:
                    fast_hit_pen_vals.append(float(info["fast_hit_pen"]))
                if "counted_hit_interval_mean" in info:
                    counted_hit_interval_mean_vals.append(float(info["counted_hit_interval_mean"]))
                if "counted_hit_interval_min" in info and float(info["counted_hit_interval_min"]) > 0:
                    counted_hit_interval_min_vals.append(float(info["counted_hit_interval_min"]))
                if "hit_reward_count_cap" in info:
                    hit_reward_count_cap_vals.append(float(info["hit_reward_count_cap"]))
                if "rewarded_hit_count" in info:
                    rewarded_hit_count_vals.append(float(info["rewarded_hit_count"]))
                if "unrewarded_extra_hit_count" in info:
                    unrewarded_extra_hit_count_vals.append(float(info["unrewarded_extra_hit_count"]))
                if "hit_reward_cap_reached" in info:
                    hit_reward_cap_reached_vals.append(float(info["hit_reward_cap_reached"]))
                total_r += float(reward)
                step_i += 1
                ball_traj.append(np.asarray(info["ball_pos"], dtype=np.float32).copy())
                paddle_traj.append(np.asarray(info["racket_pos"], dtype=np.float32).copy())
                ball_z.append(float(info["ball_pos"][2]))
                paddle_z.append(float(info["racket_pos"][2]))
                sim_t.append(step_i * env.dt)
                arm_qpos_history.append(np.array([env.data.qpos[i] for i in env.arm_qadr], dtype=np.float32))

                if args.log_hit_events and bool(info.get("new_hit", False)):
                    print(
                        f"[EP {ep+1} step {step_i}] HIT! "
                        f"hits={info.get('hit_count', 0)} vz={info.get('ball_vz', 0.0):.3f} vxy={info.get('ball_vxy', 0.0):.3f}"
                    )
                    if args.realtime and args.hit_freeze_sec > 0.0:
                        time.sleep(float(args.hit_freeze_sec))

                if args.print_every > 0 and step_i % args.print_every == 0:
                    lim_clip = info.get('arm_limiter_clip_frac', 0.0)
                    lim_dclip = info.get('arm_limiter_delta_clip_frac', info.get('arm_cmd_vel_clip_frac', 0.0))
                    lim_aclip = info.get('arm_limiter_acc_clip_frac', 0.0)
                    lim_pen = info.get('arm_limiter_pen', 0.0)
                    lim_penalty = info.get('arm_limiter_penalty', 0.0)
                    target_delta_norm = info.get('arm_cmd_tracking_error_norm', 0.0)
                    target_delta_change_norm = info.get('arm_cmd_tracking_error_max_abs', 0.0)
                    print(
                        f"[EP {ep+1} step {step_i}] R={total_r:.2f} "
                        f"xy_err={info['xy_err']:.3f} z_err={info['z_err']:.3f} vz={info['ball_vz']:.3f} vxy={info.get('ball_vxy', 0.0):.3f} "
                        f"ball_base_x={info.get('ball_base_x', 0.0):.3f} base_vxy_pen={info.get('ball_base_vxy_pen', 0.0):.3f} "
                        f"cam_vis={int(bool(info.get('camera_visible', False)))} "
                        f"cam_margin={int(bool(info.get('camera_in_margin', False)))} "
                        f"cam_top_pen={info.get('camera_top_margin_pen', 0.0):.4f} "
                        f"uv=({info.get('ball_pixel_u', 0.0):.1f},{info.get('ball_pixel_v', 0.0):.1f}) "
                        f"cam_z={info.get('ball_cam_z', 0.0):.3f} "
                        f"hits={info.get('hit_count', 0)} new_hit={int(bool(info.get('new_hit', False)))} "
                        f"hit_int_mean={info.get('hit_interval_mean', 0.0):.3f} "
                        f"short_frac={info.get('short_hit_interval_frac', 0.0):.2f} "
                        f"hit_rate={info.get('hit_rate_hz_mean', 0.0):.2f} "
                        f"counted_hit={int(bool(info.get('counted_hit', False)))} "
                        f"ignored_fast_hit={int(bool(info.get('ignored_fast_hit', False)))} "
                        f"ignored_fast_frac={info.get('ignored_fast_hit_frac', 0.0):.2f} "
                        f"counted_int_mean={info.get('counted_hit_interval_mean', 0.0):.3f} "
                        f"reward_cap={info.get('hit_reward_count_cap', 0)} "
                        f"rewarded_hits={info.get('rewarded_hit_count', 0)} "
                        f"extra_hits={info.get('unrewarded_extra_hit_count', 0)} "
                        f"cap_reached={int(bool(info.get('hit_reward_cap_reached', False)))} "
                        f"act_norm(raw/used)={act_norm:.3f}/{act_norm_used:.3f} "
                        f"arm_vel_ratio_max={info.get('arm_vel_ratio_max', 0.0):.2f} arm_acc_ratio_max={info.get('arm_acc_ratio_max', 0.0):.2f} "
                        f"lim_clip={lim_clip:.2f} lim_dclip={lim_dclip:.2f} lim_aclip={lim_aclip:.2f} "
                        f"lim_pen={lim_pen:.3f} lim_penalty={lim_penalty:.3f} "
                        f"target_delta_norm={target_delta_norm:.4f} tracking_err_max={target_delta_change_norm:.4f}"
                    )

                if renderer is not None:
                    renderer.update_scene(env.data)
                    frame = renderer.render()
                    video_frames.append(np.asarray(frame, dtype=np.uint8).copy())

                if viewer is not None:
                    viewer.sync()
                if args.realtime:
                    dt = env.dt
                    slowmo = max(0.05, float(args.slowmo))
                    elapsed = time.perf_counter() - t0
                    target = dt * slowmo
                    if elapsed < target:
                        time.sleep(target - elapsed)

            ep_hits = int(info.get("hit_count", 0))
            episode_hits.append(ep_hits)
            print(f"[EP {ep+1}] total_reward={total_r:.2f} hits={ep_hits}")
            print(f"[EP {ep+1}] JUGGLE_COUNT={ep_hits}")
            if cfg.domain_randomization:
                print(f"[EP {ep+1}] DR_PARAMS: mass={info.get('dr_ball_mass', 0.0):.6f} "
                      f"ball_fric={info.get('dr_ball_friction', 0.0):.3f} "
                      f"racket_fric={info.get('dr_racket_friction', 0.0):.3f} "
                      f"grav_z={info.get('dr_gravity_z', 0.0):.3f} "
                      f"act_mult={info.get('dr_action_scale_mult', 0.0):.3f}")
            if args.plot_trajectory:
                _plot_ball_trajectory_3d(
                    traj_xyz=ball_traj,
                    paddle_xyz=paddle_traj,
                    ep_index=ep + 1,
                    target_height=cfg.target_height,
                    save_dir=args.save_trajectory_dir,
                    show_plot=not args.no_show_plot,
                )
            if args.plot_height:
                _plot_ball_height(
                    time_s=np.asarray(sim_t, dtype=np.float32),
                    z_m=np.asarray(ball_z, dtype=np.float32),
                    paddle_z_m=np.asarray(paddle_z, dtype=np.float32),
                    ep_index=ep + 1,
                    target_height=cfg.target_height,
                    save_dir=args.save_trajectory_dir,
                    show_plot=not args.no_show_plot,
                )
            if args.record_video and len(video_frames) > 0:
                vdir = args.video_dir or args.save_trajectory_dir or DEFAULT_VIDEO_DIR
                vdir.mkdir(parents=True, exist_ok=True)
                vout = vdir / f"juggle_dr_ep{ep+1:03d}.gif"
                imageio.mimsave(vout, video_frames, fps=max(1, int(args.video_fps)))
                print(f"[VIDEO] saved: {vout}")

            if (args.record_video or args.plot_right_arm_joints) and len(arm_qpos_history) > 0:
                arm_save_dir = args.video_dir or args.save_trajectory_dir or DEFAULT_VIDEO_DIR
                _plot_right_arm_joints(
                    time_s=np.asarray(sim_t, dtype=np.float32),
                    arm_qpos_deg=np.rad2deg(np.array(arm_qpos_history, dtype=np.float32)),
                    ep_index=ep + 1,
                    save_dir=arm_save_dir,
                    show_plot=False,
                )

            if viewer is not None and (not viewer.is_running()):
                break

    if len(episode_hits) > 0:
        avg_hits = float(np.mean(np.asarray(episode_hits, dtype=np.float32)))
        max_hits = int(np.max(np.asarray(episode_hits, dtype=np.int32)))
        min_hits = int(np.min(np.asarray(episode_hits, dtype=np.int32)))
        print(f"[SUMMARY] hits_per_episode={episode_hits}")
        print(f"[SUMMARY] avg_hits={avg_hits:.2f} min_hits={min_hits} max_hits={max_hits}")
        if len(arm_vel_ratio_max_vals) > 0:
            print(f"[SUMMARY] arm_vel_ratio_max_mean={float(np.mean(arm_vel_ratio_max_vals)):.3f} max={float(np.max(arm_vel_ratio_max_vals)):.3f}")
        if len(arm_acc_ratio_max_vals) > 0:
            print(f"[SUMMARY] arm_acc_ratio_max_mean={float(np.mean(arm_acc_ratio_max_vals)):.3f} max={float(np.max(arm_acc_ratio_max_vals)):.3f}")
        if len(arm_cmd_acc_ratio_mean_vals) > 0:
            print(f"[SUMMARY] arm_cmd_acc_ratio_mean_mean={float(np.mean(arm_cmd_acc_ratio_mean_vals)):.3f}")
        if len(arm_cmd_acc_ratio_max_vals) > 0:
            print(f"[SUMMARY] arm_cmd_acc_ratio_max_max={float(np.max(arm_cmd_acc_ratio_max_vals)):.3f}")
        if len(arm_limiter_clip_frac_vals) > 0:
            print(f"[SUMMARY] arm_limiter_clip_frac_mean={float(np.mean(arm_limiter_clip_frac_vals)):.3f}")
        if len(arm_limiter_pen_vals) > 0:
            print(f"[SUMMARY] arm_limiter_pen_mean={float(np.mean(arm_limiter_pen_vals)):.6f}")
        if len(arm_limiter_penalty_vals) > 0:
            print(f"[SUMMARY] arm_limiter_penalty_mean={float(np.mean(arm_limiter_penalty_vals)):.6f}")
    if len(hit_interval_mean_vals) > 0:
        print(f"[SUMMARY] hit_interval_mean_mean={float(np.mean(hit_interval_mean_vals)):.4f}")
    if len(hit_interval_min_vals) > 0:
        print(f"[SUMMARY] hit_interval_min_min={float(np.min(hit_interval_min_vals)):.4f}")
    if len(short_hit_interval_frac_vals) > 0:
        print(f"[SUMMARY] short_hit_interval_frac_mean={float(np.mean(short_hit_interval_frac_vals)):.4f}")
    if len(hit_rate_hz_mean_vals) > 0:
        print(f"[SUMMARY] hit_rate_hz_mean={float(np.mean(hit_rate_hz_mean_vals)):.4f}")
    if len(hit_cadence_rew_mean_vals) > 0:
        print(f"[SUMMARY] hit_cadence_rew_mean={float(np.mean(hit_cadence_rew_mean_vals)):.6f}")
    if len(hit_min_interval_pen_mean_vals) > 0:
        print(f"[SUMMARY] hit_min_interval_pen_mean={float(np.mean(hit_min_interval_pen_mean_vals)):.6f}")
    if len(ignored_fast_hit_frac_vals) > 0:
        print(f"[SUMMARY] ignored_fast_hit_frac_mean={float(np.mean(ignored_fast_hit_frac_vals)):.4f}")
    if len(ignored_fast_hit_count_vals) > 0:
        print(f"[SUMMARY] ignored_fast_hit_count_mean={float(np.mean(ignored_fast_hit_count_vals)):.4f}")
    if len(fast_hit_pen_vals) > 0:
        print(f"[SUMMARY] fast_hit_pen_mean={float(np.mean(fast_hit_pen_vals)):.6f}")
    if len(counted_hit_interval_mean_vals) > 0:
        print(f"[SUMMARY] counted_hit_interval_mean_mean={float(np.mean(counted_hit_interval_mean_vals)):.4f}")
    if len(counted_hit_interval_min_vals) > 0:
        print(f"[SUMMARY] counted_hit_interval_min_min={float(np.min(counted_hit_interval_min_vals)):.4f}")
    if len(hit_reward_count_cap_vals) > 0:
        print(f"[SUMMARY] hit_reward_count_cap_mean={float(np.mean(hit_reward_count_cap_vals)):.2f}")
    if len(rewarded_hit_count_vals) > 0:
        print(f"[SUMMARY] rewarded_hit_count_mean={float(np.mean(rewarded_hit_count_vals)):.2f}")
    if len(unrewarded_extra_hit_count_vals) > 0:
        print(f"[SUMMARY] unrewarded_extra_hit_count_mean={float(np.mean(unrewarded_extra_hit_count_vals)):.2f}")
    if len(hit_reward_cap_reached_vals) > 0:
        print(f"[SUMMARY] hit_reward_cap_reached_frac={float(np.mean(hit_reward_cap_reached_vals)):.4f}")
    if len(arm_cmd_tracking_error_norm_vals) > 0:
        print(f"[SUMMARY] arm_cmd_tracking_error_norm_mean={float(np.mean(arm_cmd_tracking_error_norm_vals)):.6f}")
    if len(arm_cmd_tracking_error_max_abs_vals) > 0:
        print(f"[SUMMARY] arm_cmd_tracking_error_max_abs_mean={float(np.mean(arm_cmd_tracking_error_max_abs_vals)):.6f}")


if __name__ == "__main__":
    main()
