from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

# Make sibling modules importable when the script is launched via absolute
# path (e.g. `python3 /path/to/tools/rl_sim/train_juggle_rl_random.py`).
_RL_SIM_DIR = Path(__file__).resolve().parent
if str(_RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_SIM_DIR))

from rl_juggle_env_random import JuggleEnv, JuggleConfig

try:
    from config_utils import apply_yaml_defaults
except ModuleNotFoundError:
    # Fallback: no config_utils.py alongside the script. Skip YAML defaults
    # silently; CLI flags still work end-to-end.
    def apply_yaml_defaults(parser, argv, *, section, default_config_path):
        cfg_path = Path(default_config_path)
        if cfg_path.exists():
            print(
                f"[train] WARN: config_utils.py missing; ignoring {cfg_path}",
                file=sys.stderr)
        return None


# Non-ROS standalone training script. Resolve defaults inside the workspace
# so the XML and all generated outputs stay under pingpong_controller.
RL_SIM_DIR = _RL_SIM_DIR
PACKAGE_DIR = RL_SIM_DIR.parents[1]  # pingpong_controller/pingpong_controller
OUTPUT_DIR = PACKAGE_DIR / "outputs" / "rl_sim"
DEFAULT_XML = RL_SIM_DIR / "moz1_pd.xml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO for wheel-humanoid ping-pong juggling with domain randomization")
    p.add_argument("--config", type=Path, default=RL_SIM_DIR / "config.yaml")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--total-steps", type=int, default=10_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--checkpoint-freq", type=int, default=20_000)
    p.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    p.add_argument("--horizon-sec", type=float, default=6.0)
    p.add_argument("--frame-skip", type=int, default=5)
    p.add_argument("--action-scale-arm-rad", type=float, default=0.025)
    p.add_argument("--action-scale-base-xy", type=float, default=0.020, help="Unused by the right-arm-only policy; kept for CLI compatibility")
    p.add_argument("--action-scale-base-yaw", type=float, default=0.030, help="Unused by the right-arm-only policy; kept for CLI compatibility")
    p.add_argument("--ball-launch-height", type=float, default=0.32)
    p.add_argument("--ball-spawn-cube-size", type=float, default=0.10, help="Ball spawn cube size (m)")
    p.add_argument("--ball-spawn-xy-jitter", type=float, default=0.025, help="Ball spawn XY jitter (m)")
    p.add_argument("--ball-spawn-z-jitter", type=float, default=0.035, help="Ball spawn Z jitter (m)")
    p.add_argument("--ball-init-vxy-max", type=float, default=0.012, help="Ball initial XY velocity max (m/s)")
    p.add_argument("--ball-init-vz", type=float, default=-0.28, help="Ball initial Z velocity (m/s)")
    p.add_argument("--ball-obs-rate-hz", type=float, default=50.0)
    p.add_argument("--ball-obs-pos-noise-std", type=float, default=0.003)
    p.add_argument("--ball-obs-vel-noise-std", type=float, default=0.03)
    p.add_argument("--ball-obs-noise-warmup-ratio", type=float, default=0.10)
    p.add_argument("--ball-obs-noise-ramp-ratio", type=float, default=0.20)
    p.add_argument("--target-height", type=float, default=0.42)
    p.add_argument("--posture-weight", type=float, default=1.00)
    p.add_argument("--base-pose-weight", type=float, default=0.35)
    p.add_argument("--ball-base-x-penalty-weight", type=float, default=2.0)
    p.add_argument("--ball-base-x-soft-limit", type=float, default=0.04)
    p.add_argument("--ball-base-vxy-penalty-weight", type=float, default=0.60)
    p.add_argument("--torque-penalty-weight", type=float, default=0.0005)
    p.add_argument("--no-domain-randomization", action="store_true", help="Disable domain randomization")
    # DR preset for curriculum stages
    p.add_argument("--dr-preset", type=str, default="full", choices=["none", "ball", "ball_contact", "ball_contact_actuator", "full"],
                   help="DR preset: none=off, ball=ball+gravity, ball_contact=+friction+solref, ball_contact_actuator=+action_scale+damping+armature, full=+latency")
    # DR category switches (override preset)
    p.add_argument("--dr-ball", dest="dr_randomize_ball", action="store_true", default=None, help="Enable ball DR (ball mass + gravity)")
    p.add_argument("--no-dr-ball", dest="dr_randomize_ball", action="store_false", help="Disable ball DR")
    p.add_argument("--dr-contact", dest="dr_randomize_contact", action="store_true", default=None, help="Enable contact DR (friction + solref)")
    p.add_argument("--no-dr-contact", dest="dr_randomize_contact", action="store_false", help="Disable contact DR")
    p.add_argument("--dr-actuator", dest="dr_randomize_actuator", action="store_true", default=None, help="Enable actuator DR (action scale + damping + armature)")
    p.add_argument("--no-dr-actuator", dest="dr_randomize_actuator", action="store_false", help="Disable actuator DR")
    p.add_argument("--dr-latency", dest="dr_randomize_latency", action="store_true", default=None, help="Enable latency DR (obs + action latency)")
    p.add_argument("--no-dr-latency", dest="dr_randomize_latency", action="store_false", help="Disable latency DR")
    # DR range parameters
    p.add_argument("--dr-ball-mass-min", type=float, default=0.0024, help="Min ball mass (kg)")
    p.add_argument("--dr-ball-mass-max", type=float, default=0.0030, help="Max ball mass (kg)")
    p.add_argument("--dr-gravity-z-min", type=float, default=-9.90, help="Min gravity Z (m/s^2)")
    p.add_argument("--dr-gravity-z-max", type=float, default=-9.70, help="Max gravity Z (m/s^2)")
    p.add_argument("--dr-ball-friction-min", type=float, default=0.12, help="Min ball friction")
    p.add_argument("--dr-ball-friction-max", type=float, default=0.35, help="Max ball friction")
    p.add_argument("--dr-racket-friction-min", type=float, default=0.25, help="Min racket friction")
    p.add_argument("--dr-racket-friction-max", type=float, default=0.55, help="Max racket friction")
    p.add_argument("--dr-ball-solref-time-min", type=float, default=0.002, help="Min ball solref time")
    p.add_argument("--dr-ball-solref-time-max", type=float, default=0.006, help="Max ball solref time")
    p.add_argument("--dr-ball-solref-damping-min", type=float, default=0.70, help="Min ball solref damping")
    p.add_argument("--dr-ball-solref-damping-max", type=float, default=0.95, help="Max ball solref damping")
    p.add_argument("--dr-action-scale-mult-min", type=float, default=0.85, help="Min action scale multiplier")
    p.add_argument("--dr-action-scale-mult-max", type=float, default=1.15, help="Max action scale multiplier")
    p.add_argument("--dr-armature-mult-min", type=float, default=0.80, help="Min armature multiplier")
    p.add_argument("--dr-armature-mult-max", type=float, default=1.20, help="Max armature multiplier")
    p.add_argument("--dr-damping-mult-min", type=float, default=0.70, help="Min damping multiplier")
    p.add_argument("--dr-damping-mult-max", type=float, default=1.30, help="Max damping multiplier")
    p.add_argument("--dr-obs-latency-min", type=int, default=0, help="Min obs latency steps")
    p.add_argument("--dr-obs-latency-max", type=int, default=2, help="Max obs latency steps")
    p.add_argument("--dr-action-latency-min", type=int, default=0, help="Min action latency steps")
    p.add_argument("--dr-action-latency-max", type=int, default=2, help="Max action latency steps")
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
    p.add_argument("--racket-xy-gauss-reward-weight", type=float, default=0.05, help="Reward weight for racket XY Gaussian anchor")
    p.add_argument("--racket-xy-gauss-penalty-weight", type=float, default=0.05, help="Penalty weight for racket XY Gaussian deviation")
    p.add_argument("--ball-anchor-xy-penalty-weight", type=float, default=0.7, help="Penalty weight for ball XY distance from chest anchor")
    p.add_argument("--ball-vxy-penalty-weight", type=float, default=0.40, help="Penalty weight for ball XY velocity")
    p.add_argument("--arm-vel-limit-deg-s", nargs=7, type=float, default=[210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0], help="Arm velocity limits in deg/s for 7 joints")
    p.add_argument("--arm-acc-limit-deg-s2", nargs=7, type=float, default=[1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0], help="Arm acceleration limits in deg/s^2 for 7 joints")
    p.add_argument("--dr-racket-mount", action="store_true", help="Enable racket mount DR")
    p.add_argument("--dr-racket-pos-offset-mm", type=float, default=0.0, help="Racket position offset in mm")
    p.add_argument("--dr-racket-rot-offset-deg", type=float, default=0.0, help="Racket rotation offset in degrees")
    p.add_argument("--dr-racket-radius-offset-mm", type=float, default=0.0, help="Racket radius offset in mm")
    p.add_argument("--resume-from", type=Path, default=None, help="Path to existing PPO zip checkpoint to continue training")
    p.add_argument("--reset-num-timesteps", action="store_true", help="Reset timestep counter when resuming (default: keep counting)")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--ppo-n-steps", type=int, default=2048)
    p.add_argument("--ppo-batch-size", type=int, default=256)
    p.add_argument("--ppo-gamma", type=float, default=0.995)
    p.add_argument("--ppo-gae-lambda", type=float, default=0.95)
    p.add_argument("--ppo-ent-coef", type=float, default=0.0)
    p.add_argument("--ppo-clip-range", type=float, default=0.2)
    apply_yaml_defaults(
        p,
        argv,
        section="train_juggle_rl",
        default_config_path=Path(__file__).resolve().parent / "config.yaml",
    )
    args = p.parse_args(argv)

    # Apply DR preset
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

    # Apply preset, then allow explicit overrides
    if args.dr_randomize_ball is None:
        args.dr_randomize_ball = preset_ball
    if args.dr_randomize_contact is None:
        args.dr_randomize_contact = preset_contact
    if args.dr_randomize_actuator is None:
        args.dr_randomize_actuator = preset_actuator
    if args.dr_randomize_latency is None:
        args.dr_randomize_latency = preset_latency

    return args


class RolloutHitsTensorboardCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self._buf: dict[str, list[float]] = {}

    def _append(self, key: str, val) -> None:
        if key not in self._buf:
            self._buf[key] = []
        self._buf[key].append(float(val))

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "ball_base_x" in info:
                self._append("ball_base_x", info["ball_base_x"])
            if "ball_base_vxy_pen" in info:
                self._append("ball_base_vxy", info["ball_base_vxy_pen"])
            if "ball_vxy_pen" in info:
                self._append("ball_vxy", info["ball_vxy_pen"])
            if "ball_anchor_xy_pen" in info:
                self._append("ball_anchor_xy_dist", np.sqrt(float(info["ball_anchor_xy_pen"])))
            if "arm_cmd_acc_clip_frac" in info:
                self._append("arm_cmd_acc_clip_frac", info["arm_cmd_acc_clip_frac"])
            if "arm_cmd_vel_clip_frac" in info:
                self._append("arm_cmd_vel_clip_frac", info["arm_cmd_vel_clip_frac"])
            if "arm_limiter_clip_frac" in info:
                self._append("arm_limiter_clip_frac", info["arm_limiter_clip_frac"])
            if "arm_acc_ratio_mean" in info:
                self._append("arm_acc_ratio_mean", info["arm_acc_ratio_mean"])
            if "arm_acc_ratio_max" in info:
                self._append("arm_acc_ratio_max", info["arm_acc_ratio_max"])
            if "arm_vel_ratio_mean" in info:
                self._append("arm_vel_ratio_mean", info["arm_vel_ratio_mean"])
            if "arm_vel_ratio_max" in info:
                self._append("arm_vel_ratio_max", info["arm_vel_ratio_max"])
            if "ball_obs_age" in info:
                self._append("ball_obs_age", info["ball_obs_age"])
            if "ball_obs_dropout_active" in info:
                self._append("ball_obs_dropout_active", float(bool(info["ball_obs_dropout_active"])))
            if "camera_visible" in info:
                self._append("camera_visible", float(bool(info["camera_visible"])))
            if "camera_pixel_margin_pen" in info:
                self._append("camera_pixel_margin", info["camera_pixel_margin_pen"])
            if "camera_top_margin_pen" in info:
                self._append("camera_top_margin_pen", info["camera_top_margin_pen"])
            if "dr_action_scale_mult" in info:
                self._append("dr_action_scale_mult", info["dr_action_scale_mult"])
            if "dr_obs_latency_steps" in info:
                self._append("dr_obs_latency_steps", info["dr_obs_latency_steps"])
            if "dr_action_latency_steps" in info:
                self._append("dr_action_latency_steps", info["dr_action_latency_steps"])
            if "dr_racket_pos_offset" in info:
                self._append("dr_racket_pos_offset_norm", float(np.linalg.norm(info["dr_racket_pos_offset"])))
            if "dr_racket_rot_offset" in info:
                self._append("dr_racket_rot_offset_norm", float(np.linalg.norm(info["dr_racket_rot_offset"])))
            if "dr_racket_radius_offset" in info:
                self._append("dr_racket_radius_offset_abs", abs(float(info["dr_racket_radius_offset"])))
            if "ball_obs_burst_count" in info:
                self._append("ball_obs_burst_count", info["ball_obs_burst_count"])
            # Per-step action norms
            a = self.locals.get("actions")
            if a is not None and len(a) > 0:
                self._append("action_norm", float(np.linalg.norm(a[0])))
        # Action delta norm (difference from previous action)
        new_actions = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        if actions is not None and len(actions) > 0:
            if not hasattr(self, "_prev_action"):
                self._prev_action = np.zeros_like(actions[0])
            self._append("action_delta_norm", float(np.linalg.norm(actions[0] - self._prev_action)))
            self._prev_action = actions[0].copy()
        return True

    def _on_rollout_end(self) -> None:
        buf = list(self.model.ep_info_buffer)
        hits = [float(ep["hit_count"]) for ep in buf if "hit_count" in ep]
        if len(hits) > 0:
            self.logger.record("rollout/ep_hits_mean", float(np.mean(hits)))
            self.logger.record("rollout/ep_hits_max", float(np.max(hits)))

        # Episode-level metrics from ep_info_buffer
        for key, tb_name, agg in [
            ("counted_hit_interval_mean", "rollout/counted_hit_interval_mean", "mean"),
            ("counted_hit_interval_min", "rollout/counted_hit_interval_min", "min"),
            ("short_hit_interval_frac", "rollout/short_hit_interval_frac", "mean"),
            ("ignored_fast_hit_frac", "rollout/ignored_fast_hit_frac", "mean"),
            ("rewarded_hit_count", "rollout/rewarded_hit_count_mean", "mean"),
            ("unrewarded_extra_hit_count", "rollout/unrewarded_extra_hit_count_mean", "mean"),
            ("hit_reward_cap_reached", "rollout/hit_reward_cap_reached_frac", "mean"),
        ]:
            vals = [float(ep[key]) for ep in buf if key in ep and (key != "counted_hit_interval_min" or float(ep[key]) > 0)]
            if vals:
                if agg == "mean":
                    self.logger.record(tb_name, float(np.mean(vals)))
                elif agg == "min":
                    self.logger.record(tb_name, float(np.min(vals)))

        # Step-level metrics from _buf
        _mean_keys = [
            ("ball_base_x", "rollout/ball_base_x_abs_mean", "abs_mean"),
            ("ball_base_x", "rollout/ball_base_x_abs_max", "abs_max"),
            ("ball_base_vxy", "rollout/ball_base_vxy_mean", "mean"),
            ("ball_vxy", "rollout/ball_vxy_mean", "mean"),
            ("ball_anchor_xy_dist", "rollout/ball_anchor_xy_dist_mean", "mean"),
            ("arm_cmd_acc_clip_frac", "rollout/arm_cmd_acc_clip_frac_mean", "mean"),
            ("arm_cmd_vel_clip_frac", "rollout/arm_cmd_vel_clip_frac_mean", "mean"),
            ("arm_limiter_clip_frac", "rollout/arm_limiter_clip_frac_mean", "mean"),
            ("arm_acc_ratio_mean", "rollout/arm_acc_ratio_mean", "mean"),
            ("arm_acc_ratio_max", "rollout/arm_acc_ratio_max", "max"),
            ("arm_vel_ratio_mean", "rollout/arm_vel_ratio_mean", "mean"),
            ("arm_vel_ratio_max", "rollout/arm_vel_ratio_max", "max"),
            ("action_norm", "rollout/action_norm_mean", "mean"),
            ("action_delta_norm", "rollout/action_delta_norm_mean", "mean"),
            ("ball_obs_age", "rollout/ball_obs_age_mean", "mean"),
            ("ball_obs_age", "rollout/ball_obs_age_max", "max"),
            ("ball_obs_dropout_active", "rollout/ball_obs_dropout_frac", "mean"),
            ("ball_obs_burst_count", "rollout/ball_obs_burst_count_mean", "mean"),
            ("camera_visible", "rollout/camera_visible_mean", "mean"),
            ("camera_pixel_margin", "rollout/camera_pixel_margin_mean", "mean"),
            ("camera_top_margin_pen", "rollout/camera_top_margin_pen_mean", "mean"),
            ("dr_action_scale_mult", "rollout/dr_action_scale_mult_mean", "mean"),
            ("dr_obs_latency_steps", "rollout/dr_obs_latency_steps_mean", "mean"),
            ("dr_action_latency_steps", "rollout/dr_action_latency_steps_mean", "mean"),
            ("dr_racket_pos_offset_norm", "rollout/dr_racket_pos_offset_norm_mean", "mean"),
            ("dr_racket_rot_offset_norm", "rollout/dr_racket_rot_offset_norm_mean", "mean"),
            ("dr_racket_radius_offset_abs", "rollout/dr_racket_radius_offset_abs_mean", "mean"),
        ]
        for buf_key, tb_name, agg in _mean_keys:
            vals = self._buf.get(buf_key)
            if vals:
                arr = np.asarray(vals, dtype=np.float32)
                if agg == "mean":
                    self.logger.record(tb_name, float(np.mean(arr)))
                elif agg == "max":
                    self.logger.record(tb_name, float(np.max(arr)))
                elif agg == "abs_mean":
                    self.logger.record(tb_name, float(np.mean(np.abs(arr))))
                elif agg == "abs_max":
                    self.logger.record(tb_name, float(np.max(np.abs(arr))))

        self._buf.clear()


class EvalHitsTensorboardCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 5, deterministic: bool = True) -> None:
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        obs = self.eval_env.reset()
        ep_hits: list[float] = []
        last_hits = 0.0
        step_buf: dict[str, list[float]] = {}

        def _app(k: str, v) -> None:
            if k not in step_buf:
                step_buf[k] = []
            step_buf[k].append(float(v))

        _prev_eval_action = None
        while len(ep_hits) < self.n_eval_episodes:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            _app("action_norm", float(np.linalg.norm(action[0])))
            if _prev_eval_action is not None:
                _app("action_delta_norm", float(np.linalg.norm(action[0] - _prev_eval_action)))
            _prev_eval_action = action[0].copy()
            obs, _, dones, infos = self.eval_env.step(action)

            if len(infos) > 0:
                info0 = infos[0]
                if "hit_count" in info0:
                    last_hits = float(info0["hit_count"])
                if "ball_base_x" in info0:
                    _app("ball_base_x", info0["ball_base_x"])
                if "ball_base_vxy_pen" in info0:
                    _app("ball_base_vxy", info0["ball_base_vxy_pen"])
                if "ball_vxy_pen" in info0:
                    _app("ball_vxy", info0["ball_vxy_pen"])
                if "ball_anchor_xy_pen" in info0:
                    _app("ball_anchor_xy_dist", np.sqrt(float(info0["ball_anchor_xy_pen"])))
                if "arm_cmd_acc_clip_frac" in info0:
                    _app("arm_cmd_acc_clip_frac", info0["arm_cmd_acc_clip_frac"])
                if "arm_cmd_vel_clip_frac" in info0:
                    _app("arm_cmd_vel_clip_frac", info0["arm_cmd_vel_clip_frac"])
                if "arm_limiter_clip_frac" in info0:
                    _app("arm_limiter_clip_frac", info0["arm_limiter_clip_frac"])
                if "arm_acc_ratio_mean" in info0:
                    _app("arm_acc_ratio_mean", info0["arm_acc_ratio_mean"])
                if "arm_acc_ratio_max" in info0:
                    _app("arm_acc_ratio_max", info0["arm_acc_ratio_max"])
                if "arm_vel_ratio_mean" in info0:
                    _app("arm_vel_ratio_mean", info0["arm_vel_ratio_mean"])
                if "arm_vel_ratio_max" in info0:
                    _app("arm_vel_ratio_max", info0["arm_vel_ratio_max"])
                if "ball_obs_age" in info0:
                    _app("ball_obs_age", info0["ball_obs_age"])
                if "ball_obs_dropout_active" in info0:
                    _app("ball_obs_dropout_active", float(bool(info0["ball_obs_dropout_active"])))
                if "camera_visible" in info0:
                    _app("camera_visible", float(bool(info0["camera_visible"])))
                if "camera_pixel_margin_pen" in info0:
                    _app("camera_pixel_margin", info0["camera_pixel_margin_pen"])
                if "camera_top_margin_pen" in info0:
                    _app("camera_top_margin_pen", info0["camera_top_margin_pen"])
                if "dr_action_scale_mult" in info0:
                    _app("dr_action_scale_mult", info0["dr_action_scale_mult"])
                if "dr_obs_latency_steps" in info0:
                    _app("dr_obs_latency_steps", info0["dr_obs_latency_steps"])
                if "dr_action_latency_steps" in info0:
                    _app("dr_action_latency_steps", info0["dr_action_latency_steps"])
                if "dr_racket_pos_offset" in info0:
                    _app("dr_racket_pos_offset_norm", float(np.linalg.norm(info0["dr_racket_pos_offset"])))
                if "dr_racket_rot_offset" in info0:
                    _app("dr_racket_rot_offset_norm", float(np.linalg.norm(info0["dr_racket_rot_offset"])))
                if "dr_racket_radius_offset" in info0:
                    _app("dr_racket_radius_offset_abs", abs(float(info0["dr_racket_radius_offset"])))
                if "ball_obs_burst_count" in info0:
                    _app("ball_obs_burst_count", info0["ball_obs_burst_count"])
                if "counted_hit_interval_mean" in info0:
                    _app("counted_hit_interval_mean", info0["counted_hit_interval_mean"])
                if "counted_hit_interval_min" in info0 and float(info0["counted_hit_interval_min"]) > 0:
                    _app("counted_hit_interval_min", info0["counted_hit_interval_min"])
                if "short_hit_interval_frac" in info0:
                    _app("short_hit_interval_frac", info0["short_hit_interval_frac"])
                if "ignored_fast_hit_frac" in info0:
                    _app("ignored_fast_hit_frac", info0["ignored_fast_hit_frac"])
                if "rewarded_hit_count" in info0:
                    _app("rewarded_hit_count", info0["rewarded_hit_count"])
                if "unrewarded_extra_hit_count" in info0:
                    _app("unrewarded_extra_hit_count", info0["unrewarded_extra_hit_count"])
                if "hit_reward_cap_reached" in info0:
                    _app("hit_reward_cap_reached", float(bool(info0["hit_reward_cap_reached"])))
            else:
                info0 = {}

            if bool(dones[0]):
                ep = info0.get("episode") if isinstance(info0, dict) else None
                if isinstance(ep, dict) and ("hit_count" in ep):
                    hit_val = float(ep["hit_count"])
                else:
                    hit_val = float(info0.get("hit_count", last_hits)) if isinstance(info0, dict) else float(last_hits)
                ep_hits.append(hit_val)
                last_hits = 0.0

        self.logger.record("eval/mean_hits", float(np.mean(np.asarray(ep_hits, dtype=np.float32))))

        _eval_metrics = [
            ("ball_base_x", "eval/ball_base_x_abs_mean", "abs_mean"),
            ("ball_base_x", "eval/ball_base_x_abs_max", "abs_max"),
            ("ball_base_vxy", "eval/ball_base_vxy_mean", "mean"),
            ("ball_vxy", "eval/ball_vxy_mean", "mean"),
            ("ball_anchor_xy_dist", "eval/ball_anchor_xy_dist_mean", "mean"),
            ("arm_cmd_acc_clip_frac", "eval/arm_cmd_acc_clip_frac_mean", "mean"),
            ("arm_cmd_vel_clip_frac", "eval/arm_cmd_vel_clip_frac_mean", "mean"),
            ("arm_limiter_clip_frac", "eval/arm_limiter_clip_frac_mean", "mean"),
            ("arm_acc_ratio_mean", "eval/arm_acc_ratio_mean", "mean"),
            ("arm_acc_ratio_max", "eval/arm_acc_ratio_max", "max"),
            ("arm_vel_ratio_mean", "eval/arm_vel_ratio_mean", "mean"),
            ("arm_vel_ratio_max", "eval/arm_vel_ratio_max", "max"),
            ("action_norm", "eval/action_norm_mean", "mean"),
            ("action_delta_norm", "eval/action_delta_norm_mean", "mean"),
            ("ball_obs_age", "eval/ball_obs_age_mean", "mean"),
            ("ball_obs_age", "eval/ball_obs_age_max", "max"),
            ("ball_obs_dropout_active", "eval/ball_obs_dropout_frac", "mean"),
            ("ball_obs_burst_count", "eval/ball_obs_burst_count_mean", "mean"),
            ("camera_visible", "eval/camera_visible_mean", "mean"),
            ("camera_pixel_margin", "eval/camera_pixel_margin_mean", "mean"),
            ("camera_top_margin_pen", "eval/camera_top_margin_pen_mean", "mean"),
            ("dr_action_scale_mult", "eval/dr_action_scale_mult_mean", "mean"),
            ("dr_obs_latency_steps", "eval/dr_obs_latency_steps_mean", "mean"),
            ("dr_action_latency_steps", "eval/dr_action_latency_steps_mean", "mean"),
            ("dr_racket_pos_offset_norm", "eval/dr_racket_pos_offset_norm_mean", "mean"),
            ("dr_racket_rot_offset_norm", "eval/dr_racket_rot_offset_norm_mean", "mean"),
            ("dr_racket_radius_offset_abs", "eval/dr_racket_radius_offset_abs_mean", "mean"),
            ("counted_hit_interval_mean", "eval/counted_hit_interval_mean", "mean"),
            ("counted_hit_interval_min", "eval/counted_hit_interval_min", "min"),
            ("short_hit_interval_frac", "eval/short_hit_interval_frac", "mean"),
            ("ignored_fast_hit_frac", "eval/ignored_fast_hit_frac", "mean"),
            ("rewarded_hit_count", "eval/rewarded_hit_count_mean", "mean"),
            ("unrewarded_extra_hit_count", "eval/unrewarded_extra_hit_count_mean", "mean"),
            ("hit_reward_cap_reached", "eval/hit_reward_cap_reached_frac", "mean"),
        ]
        for buf_key, tb_name, agg in _eval_metrics:
            vals = step_buf.get(buf_key)
            if vals:
                arr = np.asarray(vals, dtype=np.float32)
                if agg == "mean":
                    self.logger.record(tb_name, float(np.mean(arr)))
                elif agg == "max":
                    self.logger.record(tb_name, float(np.max(arr)))
                elif agg == "min":
                    self.logger.record(tb_name, float(np.min(arr)))
                elif agg == "abs_mean":
                    self.logger.record(tb_name, float(np.mean(np.abs(arr))))
                elif agg == "abs_max":
                    self.logger.record(tb_name, float(np.max(np.abs(arr))))
        return True


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

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
        total_training_steps=args.total_steps,
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

    def _mk():
        return JuggleEnv(xml_path=str(args.xml), cfg=cfg)

    monitor_kwargs = {
        "info_keywords": (
            "hit_count",
            "termination_reason",
            "ball_rel_z",
            "predicted_apex_rel_z",
            "ball_base_x",
            "ball_base_vxy_pen",
            "dr_ball_mass",
            "dr_ball_friction",
            "dr_gravity_z",
            "dr_action_scale_mult",
            "camera_visible",
            "camera_in_margin",
            "camera_frustum_pen",
            "camera_top_margin_pen",
            "ball_pixel_u",
            "ball_pixel_v",
            "ball_cam_z",
            "hit_interval_mean",
            "hit_interval_min",
            "short_hit_interval_frac",
            "hit_rate_hz_mean",
            "hit_cadence_rew_mean",
            "hit_min_interval_pen_mean",
            "ignored_fast_hit_frac",
            "ignored_fast_hit_count",
            "fast_hit_pen",
            "counted_hit_interval_mean",
            "counted_hit_interval_min",
            "hit_reward_count_cap",
            "rewarded_hit_count",
            "unrewarded_extra_hit_count",
            "hit_reward_cap_reached",
            "rewardable_hit",
        ),
    }
    train_env = make_vec_env(_mk, n_envs=args.n_envs, seed=args.seed, monitor_kwargs=monitor_kwargs)
    eval_env = make_vec_env(_mk, n_envs=1, seed=args.seed + 999, monitor_kwargs=monitor_kwargs)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(args.save_dir / "best"),
        log_path=str(args.save_dir / "eval"),
        eval_freq=max(1, int(args.eval_freq)),
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, int(args.checkpoint_freq)),
        save_path=str(args.save_dir / "checkpoints"),
        name_prefix="ppo_juggle_dr",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    rollout_hits_cb = RolloutHitsTensorboardCallback()
    eval_hits_cb = EvalHitsTensorboardCallback(eval_env=eval_env, eval_freq=max(1, int(args.eval_freq)), n_eval_episodes=5, deterministic=True)
    cbs = CallbackList([ckpt_cb, eval_cb, rollout_hits_cb, eval_hits_cb])

    if args.resume_from is not None:
        from stable_baselines3.common.utils import get_schedule_fn
        model = PPO.load(str(args.resume_from), env=train_env, device=args.device)
        model.tensorboard_log = str(args.save_dir / "tb")
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.clip_range = get_schedule_fn(args.ppo_clip_range)
        model.ent_coef = args.ppo_ent_coef
        print(f"[INFO] Resuming from: {args.resume_from}")
        print(f"[INFO] Applied: learning_rate={args.learning_rate}, clip_range={args.ppo_clip_range}, ent_coef={args.ppo_ent_coef}")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.ppo_n_steps,
            batch_size=args.ppo_batch_size,
            gamma=args.ppo_gamma,
            gae_lambda=args.ppo_gae_lambda,
            ent_coef=args.ppo_ent_coef,
            clip_range=args.ppo_clip_range,
            tensorboard_log=str(args.save_dir / "tb"),
            device=args.device,
            seed=args.seed,
        )

    model.learn(
        total_timesteps=args.total_steps,
        callback=cbs,
        progress_bar=True,
        reset_num_timesteps=bool(args.reset_num_timesteps),
    )
    model.save(str(args.save_dir / "ppo_juggle_dr_last"))
    print(f"[OK] Training finished: {args.save_dir}")


if __name__ == "__main__":
    main()
