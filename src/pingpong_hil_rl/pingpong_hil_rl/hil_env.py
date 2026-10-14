from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any

from gymnasium import spaces
import mujoco as mj
import numpy as np

from .ros_plant import JointSample, PlantSafetyError, RosArmPlant


def bounded_contact_impulse_reward(
    contact_entry_ball_vz: float,
    current_ball_vz: float,
    *,
    active_contact_candidate: bool,
    target_delta_vz: float,
    weight: float,
) -> tuple[float, float]:
    """Return bounded impulse potential accumulated over a contact candidate."""
    delta_vz = max(0.0, float(current_ball_vz) - float(contact_entry_ball_vz))
    if not active_contact_candidate or weight <= 0.0:
        return 0.0, delta_vz
    progress = float(np.clip(delta_vz / max(float(target_delta_vz), 1e-6), 0.0, 1.0))
    return float(weight) * progress, delta_vz


@dataclass(frozen=True)
class HilSafetyConfig:
    command_history_steps: int = 40
    max_command_delta_deg: float = 1.25
    max_tracking_error_deg: float = 15.0
    max_feedback_velocity_deg_s: float = 340.0
    reset_degrees: tuple[float, ...] = (5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214)
    max_pose_deviation_deg: tuple[float, ...] = (15.0, 15.0, 15.0, 20.0, 20.0, 25.0, 20.0)
    episode_reset_threshold_deg: float = 5.0
    reset_speed_deg_s: float = 10.0
    reset_tolerance_deg: float = 0.25
    reset_timeout_s: float = 20.0


def load_simcode4_env(simcode4_rl_dir: str | Path) -> ModuleType:
    """Load simcode4 by path without installing or mutating its package."""
    root = Path(simcode4_rl_dir).resolve()
    env_file = root / "rl_juggle_env_random.py"
    if not env_file.is_file():
        raise FileNotFoundError(f"simcode4 environment not found: {env_file}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("pingpong_simcode4_hil_base", env_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {env_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_hil_env_class(simcode4_module: ModuleType):
    base_class = simcode4_module.JuggleEnv

    class HilJuggleEnv(base_class):
        """MuJoCo ball/racket contact with the right arm prescribed by ROS2."""

        metadata = {"render_modes": []}

        def __init__(self, *, xml_path: str, plant: RosArmPlant, cfg: Any, hil_cfg: HilSafetyConfig) -> None:
            self.plant = plant
            self.hil_cfg = hil_cfg
            self._command_history: deque[np.ndarray] = deque(maxlen=max(1, hil_cfg.command_history_steps))
            self._last_plant_sample: JointSample | None = None
            self._last_qdd = np.zeros(7, dtype=np.float32)
            self._last_source_dt_s = 1.0 / max(plant.cfg.control_hz, 1.0)
            self._last_command: np.ndarray | None = None
            self._last_tracking_error_deg = 0.0
            self._pre_hit_flatness_weight = 0.0
            self._pre_hit_lift_weight = 0.0
            self._pre_hit_launch_weight = 0.0
            self._pre_hit_impulse_weight = 0.0
            self._pre_hit_impulse_target_delta_vz = 1.0
            self._workspace_termination_penalty = 0.0
            self._zero_hit_miss_penalty = 0.0
            self._valid_hit_bonus = 0.0
            self._ball_upper_margin_m = 0.15
            self._ball_lower_margin_m = 0.35
            self._prev_scaffold_potential = 0.0
            self._prev_launch_potential = 0.0
            self._previous_ball_vz = 0.0
            self._previous_pending_hit = False
            self._contact_entry_ball_vz: float | None = None
            self._prev_impulse_potential = 0.0
            self._training_gravity_z_range = (-9.81, -9.81)
            self._evaluation_gravity_z = -9.81
            self._evaluation_mode = False
            self._exploration_origin_pos: np.ndarray | None = None
            self._exploration_origin_normal: np.ndarray | None = None
            self._previous_exploration_jacobian: np.ndarray | None = None
            self._fixed_qpos_idx: np.ndarray | None = None
            self._fixed_qvel_idx: np.ndarray | None = None
            self._fixed_qpos: np.ndarray | None = None
            self._fixed_qvel: np.ndarray | None = None
            super().__init__(xml_path=xml_path, cfg=cfg)
            self.posture_targets[:7] = np.deg2rad(
                np.asarray(hil_cfg.reset_degrees, dtype=np.float32)
            )
            # Five plant/task context values prevent replay aliasing when the
            # competence curriculum changes gravity, target height, action
            # scale or racket geometry.  Without these, identical observations
            # from adjacent stages have incompatible transitions and rewards.
            extra_dim = 7 * self._command_history.maxlen + 7 + 2 + 5
            base_dim = int(self.observation_space.shape[0])
            self._base_observation_dim = base_dim
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(base_dim + extra_dim,), dtype=np.float32)

        def _inject_arm(self, q: np.ndarray, dq: np.ndarray) -> None:
            self.data.qpos[self.arm_qadr] = np.asarray(q, dtype=np.float64)
            self.data.qvel[self.arm_vadr] = np.asarray(dq, dtype=np.float64)

        def _inject_prescribed_robot(self, q: np.ndarray, dq: np.ndarray) -> None:
            if self._fixed_qpos_idx is not None:
                self.data.qpos[self._fixed_qpos_idx] = self._fixed_qpos
                self.data.qvel[self._fixed_qvel_idx] = self._fixed_qvel
            self._inject_arm(q, dq)

        def _synchronize_arm_after_reset(self) -> None:
            sample = self.plant.wait_for_state()
            self._inject_arm(sample.q, sample.dq)
            mj.mj_forward(self.model, self.data)
            qpos_mask = np.ones(self.model.nq, dtype=bool)
            qpos_mask[self.arm_qadr] = False
            qpos_mask[self.ball_qadr : self.ball_qadr + 7] = False
            qvel_mask = np.ones(self.model.nv, dtype=bool)
            qvel_mask[self.arm_vadr] = False
            qvel_mask[self.ball_vadr : self.ball_vadr + 6] = False
            self._fixed_qpos_idx = np.flatnonzero(qpos_mask)
            self._fixed_qvel_idx = np.flatnonzero(qvel_mask)
            self._fixed_qpos = self.data.qpos[self._fixed_qpos_idx].copy()
            self._fixed_qvel = self.data.qvel[self._fixed_qvel_idx].copy()
            self._last_plant_sample = sample
            self._last_command = sample.q.copy()
            self._last_qdd[:] = 0.0

        @staticmethod
        def _hermite(
            q0: np.ndarray,
            dq0: np.ndarray,
            q1: np.ndarray,
            dq1: np.ndarray,
            duration_s: float,
            alpha: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            a = float(np.clip(alpha, 0.0, 1.0))
            a2, a3 = a * a, a * a * a
            h00 = 2.0 * a3 - 3.0 * a2 + 1.0
            h10 = a3 - 2.0 * a2 + a
            h01 = -2.0 * a3 + 3.0 * a2
            h11 = a3 - a2
            q = h00 * q0 + h10 * duration_s * dq0 + h01 * q1 + h11 * duration_s * dq1
            dh00 = 6.0 * a2 - 6.0 * a
            dh10 = 3.0 * a2 - 4.0 * a + 1.0
            dh01 = -dh00
            dh11 = 3.0 * a2 - 2.0 * a
            dq = (dh00 * q0 + dh01 * q1) / duration_s + dh10 * dq0 + dh11 * dq1
            return q, dq

        def _advance_physics(self) -> None:
            target = np.asarray(self.arm_q_target, dtype=np.float64).copy()
            if self._last_command is None:
                self._last_command = self.plant.wait_for_state().q
            max_delta = np.deg2rad(float(self.hil_cfg.max_command_delta_deg))
            target = np.clip(target, self._last_command - max_delta, self._last_command + max_delta)
            target = np.clip(target, self.arm_lo, self.arm_hi)
            reset_q = np.deg2rad(np.asarray(self.hil_cfg.reset_degrees, dtype=np.float64))
            pose_window = np.deg2rad(
                np.asarray(self.hil_cfg.max_pose_deviation_deg, dtype=np.float64)
            )
            target = np.clip(target, reset_q - pose_window, reset_q + pose_window)
            self.arm_q_target = target.astype(np.float32)
            self.arm_cmd_q = self.arm_q_target.copy()

            before, after = self.plant.publish_and_wait(target)
            source_dt = (after.source_stamp_ns - before.source_stamp_ns) * 1e-9
            if not (0.001 <= source_dt <= 0.025):
                source_dt = (after.receive_stamp_ns - before.receive_stamp_ns) * 1e-9
            if not (0.001 <= source_dt <= 0.025):
                raise PlantSafetyError(f"invalid JointState interval {source_dt:.6f}s")
            max_velocity = np.deg2rad(float(self.hil_cfg.max_feedback_velocity_deg_s))
            if float(np.max(np.abs(after.dq))) > max_velocity:
                raise PlantSafetyError("feedback velocity exceeded configured HIL safety limit")
            tracking_error = float(np.max(np.abs(target - after.q)))
            self._last_tracking_error_deg = float(np.rad2deg(tracking_error))
            if tracking_error > np.deg2rad(float(self.hil_cfg.max_tracking_error_deg)):
                self.plant.publish(after.q)
                raise PlantSafetyError(
                    f"tracking error {np.rad2deg(tracking_error):.2f}deg exceeded limit"
                )

            qdd = (after.dq - before.dq) / source_dt
            qdd_limit = np.deg2rad(np.asarray(self.arm_acc_limit_deg_s2, dtype=np.float64))
            self._last_qdd = np.clip(qdd / np.maximum(qdd_limit, 1e-6), -2.0, 2.0).astype(np.float32)
            self._last_source_dt_s = float(source_dt)

            self.advance_measured_interval(before, after, source_dt)
            self._last_plant_sample = after
            self._last_command = target.copy()
            self._command_history.append(target.astype(np.float32))

        def advance_measured_interval(
            self,
            before: JointSample,
            after: JointSample,
            source_dt: float | None = None,
        ) -> None:
            """Advance ball/contact using an externally measured arm interval.

            This is shared by the live ROS path and the offline equivalence
            experiment, ensuring the experiment tests the production HIL state
            replacement code rather than a second implementation.
            """
            if source_dt is None:
                source_dt = (after.source_stamp_ns - before.source_stamp_ns) * 1e-9
            source_dt = float(source_dt)
            if not np.isfinite(source_dt) or source_dt <= 0.0:
                raise ValueError(f"invalid measured interval {source_dt!r}")

            # The arm is a measured, prescribed trajectory; MuJoCo only evolves
            # the ball/contact. Endpoints are interpolated at the XML physics dt.
            for substep in range(1, int(self.cfg.frame_skip) + 1):
                alpha = substep / float(self.cfg.frame_skip)
                q, dq = self._hermite(before.q, before.dq, after.q, after.dq, source_dt, alpha)
                self._inject_prescribed_robot(q, dq)
                self.data.ctrl[:] = self.default_ctrl
                for i, aid in enumerate(self.arm_aids):
                    if aid >= 0:
                        self.data.ctrl[aid] = q[i]
                for aid in self.base_aids:
                    if aid >= 0:
                        self.data.ctrl[aid] = 0.0
                mj.mj_step(self.model, self.data)
            self._inject_prescribed_robot(after.q, after.dq)
            mj.mj_forward(self.model, self.data)

        def _augment_observation(self, obs: np.ndarray) -> np.ndarray:
            q = np.asarray([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
            fill = self._last_command.astype(np.float32) if self._last_command is not None else q
            while len(self._command_history) < self._command_history.maxlen:
                self._command_history.appendleft(fill.copy())
            relative_history = np.concatenate([cmd - q for cmd in self._command_history]).astype(np.float32)
            receive_age = 0.0
            if self._last_plant_sample is not None:
                receive_age = (time.monotonic_ns() - self._last_plant_sample.receive_stamp_ns) * 1e-9
            timing = np.asarray(
                [
                    self._last_source_dt_s * self.plant.cfg.control_hz,
                    receive_age / max(self.plant.cfg.state_timeout_s, 1e-6),
                ],
                dtype=np.float32,
            )
            stage_context = np.asarray(
                [
                    float(self.model.opt.gravity[2]) / 9.81,
                    float(self.cfg.target_height) / 0.42,
                    float(self.cfg.action_acc_scale) / 0.90,
                    float(self.cfg.dr_racket_pos_offset_m) / 0.004,
                    float(self.cfg.dr_racket_rot_offset_rad) / np.deg2rad(1.0),
                ],
                dtype=np.float32,
            )
            return np.concatenate(
                [np.asarray(obs, dtype=np.float32), relative_history, self._last_qdd, timing, stage_context]
            )

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if self._last_plant_sample is not None:
                reset_q = np.deg2rad(np.asarray(self.hil_cfg.reset_degrees, dtype=np.float64))
                error_deg = float(
                    np.rad2deg(np.max(np.abs(self._last_plant_sample.q - reset_q)))
                )
                if error_deg > float(self.hil_cfg.episode_reset_threshold_deg):
                    self._last_plant_sample = self.plant.move_to_pose(
                        reset_q,
                        speed_deg_s=self.hil_cfg.reset_speed_deg_s,
                        tolerance_deg=self.hil_cfg.reset_tolerance_deg,
                        timeout_s=self.hil_cfg.reset_timeout_s,
                    )
            obs, info = super().reset(seed=seed, options=options)
            # The base environment restores nominal gravity whenever domain
            # randomization is disabled. Curriculum gravity is a task
            # parameter, not actuator/domain randomization, so enforce the
            # requested fixed value independently after every base reset.
            if not bool(self.cfg.domain_randomization and self.cfg.dr_randomize_ball):
                low, high = (float(value) for value in self.cfg.dr_gravity_z_range)
                gravity_z = low if low == high else float(self.np_random.uniform(low, high))
                self.model.opt.gravity[2] = gravity_z
                self.dr_gravity_z = gravity_z
            mj.mj_forward(self.model, self.data)
            # Base reset establishes the feedback-derived D455 racket anchor;
            # bounds must be resolved afterwards, not while the MuJoCo data
            # still contains its pre-reset zero state.
            reset_racket_z = float(self._racket_pos()[2])
            self.cfg.ball_termination_max_z = (
                reset_racket_z + float(self.cfg.target_height) + self._ball_upper_margin_m
            )
            self.cfg.ball_termination_min_z = reset_racket_z - self._ball_lower_margin_m
            assert self._last_plant_sample is not None
            current = self._last_plant_sample.q.astype(np.float32)
            self._command_history.clear()
            for _ in range(self._command_history.maxlen):
                self._command_history.append(current.copy())
            self.arm_cmd_q = current.copy()
            self.arm_q_target = current.copy()
            self.arm_cmd_qvel[:] = 0.0
            self.prev_action[:] = 0.0
            self._prev_scaffold_potential = 0.0
            self._prev_launch_potential = 0.0
            self._previous_ball_vz = float(self.data.qvel[self.ball_vadr + 2])
            self._previous_pending_hit = False
            self._contact_entry_ball_vz = None
            self._prev_impulse_potential = 0.0
            self._exploration_origin_pos = self._racket_pos().astype(np.float64)
            self._exploration_origin_normal = self._racket_xmat()[:, 2].astype(np.float64)
            self._previous_exploration_jacobian = None
            info.update({"hil": True, "actuator_model": False, "compensation": False})
            return self._augment_observation(obs), info

        def step(self, action: np.ndarray):
            obs, reward, terminated, truncated, info = super().step(action)
            # A single real environment does not get the accidental first-hit
            # coverage that thousands of vectorized simulators provide.  The
            # D455 reset paddle is also strongly inclined (measured up-cos
            # ~=0.653), so the unshaped optimum is to preserve posture and let
            # the ball glance away.  During the early curriculum only, expose
            # the two prerequisites for a valid launch: flatten the face, then
            # add upward paddle velocity shortly before contact.  The real hit
            # bonus remains dominant and this term vanishes after the first hit.
            pre_hit_scaffold = 0.0
            pre_hit_launch_progress = 0.0
            pre_hit_impulse_progress = 0.0
            contact_delta_vz = 0.0
            ball_vz = float(info.get("ball_vz", 0.0))
            pending = bool(info.get("pending_hit", False))
            contact_edge = bool(pending and not self._previous_pending_hit)
            if contact_edge:
                self._contact_entry_ball_vz = self._previous_ball_vz
            active_candidate = bool(
                self._contact_entry_ball_vz is not None
                and (pending or self._previous_pending_hit)
                and int(info.get("hit_count", 0)) <= 1
            )
            if active_candidate:
                impulse_potential, contact_delta_vz = bounded_contact_impulse_reward(
                    float(self._contact_entry_ball_vz),
                    ball_vz,
                    active_contact_candidate=True,
                    target_delta_vz=self._pre_hit_impulse_target_delta_vz,
                    weight=self._pre_hit_impulse_weight,
                )
                pre_hit_impulse_progress = max(
                    0.0, impulse_potential - self._prev_impulse_potential
                )
                self._prev_impulse_potential = max(
                    self._prev_impulse_potential, impulse_potential
                )
                reward = float(reward + pre_hit_impulse_progress)
            if int(info.get("hit_count", 0)) == 0:
                if ball_vz < 0.0:
                    ball_pos = np.asarray(info.get("ball_pos", np.zeros(3)), dtype=np.float64)
                    racket_pos = np.asarray(info.get("racket_pos", np.zeros(3)), dtype=np.float64)
                    if ball_pos[2] > racket_pos[2]:
                        up_cos = float(np.clip((float(info.get("racket_up_cos", 0.0)) - 0.55) / 0.40, 0.0, 1.0))
                        racket_vz = float(np.asarray(info.get("racket_vel", np.zeros(3)))[2])
                        lift = float(np.clip(racket_vz / 0.70, 0.0, 1.0))
                        time_to_plane = float(
                            (ball_pos[2] - racket_pos[2])
                            / max(-ball_vz, 1e-5)
                        )
                        contact_gate = float(np.clip((0.35 - time_to_plane) / 0.25, 0.0, 1.0))
                        scaffold_potential = contact_gate * (
                            self._pre_hit_flatness_weight * up_cos
                            + self._pre_hit_lift_weight * lift
                        )
                        # Potential differences keep the dense prerequisite
                        # signal, but prevent the agent collecting posture
                        # reward on every 200 Hz tick while doing nothing.
                        pre_hit_scaffold = max(
                            0.0, scaffold_potential - self._prev_scaffold_potential
                        )
                        self._prev_scaffold_potential = scaffold_potential
                        reward = float(reward + pre_hit_scaffold)
                # Only a candidate created by a racket-contact edge can earn
                # launch shaping. The previous absolute per-step term rewarded
                # any upward motion (including other-body collisions) on every
                # tick and was therefore exploitable. This bounded potential
                # increment pays at most one stage weight per launch attempt.
                upward = float(np.clip(ball_vz / 0.60, 0.0, 1.0))
                apex_rel = float(info.get("predicted_apex_rel_z", 0.0))
                apex_progress = float(
                    np.clip(apex_rel / max(float(self.cfg.target_height), 0.05), 0.0, 1.0)
                )
                # An additive bounded potential gives credit as soon as a
                # contact produces upward velocity.  The previous product was
                # nearly zero unless both velocity and predicted apex were
                # already good, making the signal almost as sparse as a hit.
                potential = (
                    0.5 * upward + 0.5 * apex_progress
                    if pending
                    else 0.0
                )
                pre_hit_launch_progress = self._pre_hit_launch_weight * max(
                    0.0, potential - self._prev_launch_potential
                )
                self._prev_launch_potential = potential
                reward = float(reward + pre_hit_launch_progress)
            else:
                self._prev_scaffold_potential = 0.0
                self._prev_launch_potential = 0.0

            self._previous_ball_vz = float(info.get("ball_vz", 0.0))
            self._previous_pending_hit = bool(info.get("pending_hit", False))

            valid_hit_bonus = self._valid_hit_bonus * float(bool(info.get("new_hit", False)))
            reward = float(reward + valid_hit_bonus)

            termination_reasons = set(info.get("termination_reasons", ()))
            workspace_failure = bool(
                termination_reasons.intersection(
                    {"racket_too_low", "racket_too_high", "racket_too_far_from_anchor"}
                )
            )
            # Every completed zero-hit episode is a task failure, including a
            # time limit with no explicit base-environment reason. Do not
            # double-charge workspace failures; they have their own penalty.
            zero_hit_miss = bool(
                (terminated or truncated)
                and int(info.get("hit_count", 0)) == 0
                and not workspace_failure
            )
            hil_termination_penalty = (
                self._workspace_termination_penalty * float(workspace_failure)
                + self._zero_hit_miss_penalty * float(zero_hit_miss)
            )
            reward = float(reward - hil_termination_penalty)
            sample = self._last_plant_sample
            info.update(
                {
                    "hil/source_dt_s": self._last_source_dt_s,
                    "hil/state_sequence": -1 if sample is None else sample.sequence,
                    "hil/velocity_source": "none" if sample is None else sample.velocity_source,
                    "hil/max_qdd_ratio": float(np.max(np.abs(self._last_qdd))),
                    "hil/tracking_error_deg": self._last_tracking_error_deg,
                    "hil/max_feedback_dq_deg_s": float(np.rad2deg(np.max(np.abs(sample.dq)))) if sample is not None else 0.0,
                    "hil/causal_timeout_recoveries": int(
                        getattr(self.plant, "causal_timeout_recoveries", 0)
                    ),
                    "hil/feedback_q_deg": np.rad2deg(sample.q).astype(np.float32) if sample is not None else np.zeros(7, dtype=np.float32),
                    "hil/pre_hit_scaffold": float(pre_hit_scaffold),
                    "hil/pre_hit_launch_progress": float(pre_hit_launch_progress),
                    "hil/pre_hit_impulse_progress": float(pre_hit_impulse_progress),
                    "hil/contact_delta_vz": float(contact_delta_vz),
                    "hil/valid_hit_bonus": float(valid_hit_bonus),
                    "hil/termination_penalty": float(hil_termination_penalty),
                    "hil/ball_termination_min_z": float(self.cfg.ball_termination_min_z),
                    "hil/ball_termination_max_z": float(self.cfg.ball_termination_max_z),
                    "hil/gravity_z": float(self.model.opt.gravity[2]),
                }
            )
            return self._augment_observation(obs), reward, terminated, truncated, info

        def set_evaluation_mode(self, enabled: bool) -> None:
            """Use target gravity for evaluation and a continuation band for training."""
            self._evaluation_mode = bool(enabled)
            if self._evaluation_mode:
                gravity_range = (self._evaluation_gravity_z, self._evaluation_gravity_z)
            else:
                gravity_range = self._training_gravity_z_range
            self.cfg.dr_gravity_z_range = tuple(float(value) for value in gravity_range)

        def set_curriculum_stage(self, stage: dict[str, Any]) -> None:
            self.cfg.action_acc_scale = float(stage["action_acc_scale"])
            self.cfg.ball_spawn_xy_jitter = float(stage["spawn_xy_jitter"])
            self.cfg.ball_spawn_z_jitter = float(stage.get("spawn_z_jitter", 0.0))
            self.cfg.ball_init_vz = float(stage["init_vz"])
            self.cfg.horizon_sec = float(stage["horizon_sec"])
            gravity_z = float(stage.get("gravity_z", -9.81))
            gravity_train_from_z = float(stage.get("gravity_train_from_z", gravity_z))
            self._evaluation_gravity_z = gravity_z
            self._training_gravity_z_range = (
                min(gravity_z, gravity_train_from_z),
                max(gravity_z, gravity_train_from_z),
            )
            self.set_evaluation_mode(self._evaluation_mode)
            self.cfg.dr_randomize_ball = bool(stage.get("ball_dr", False))
            self.cfg.dr_randomize_contact = bool(stage.get("contact_dr", False))
            self.cfg.dr_randomize_racket_mount = any(
                float(stage.get(name, 0.0)) > 0.0
                for name in ("racket_pos_dr_m", "racket_rot_dr_deg", "racket_radius_dr_m")
            )
            self.cfg.domain_randomization = bool(
                self.cfg.dr_randomize_ball
                or self.cfg.dr_randomize_contact
                or self.cfg.dr_randomize_racket_mount
            )
            self.cfg.dr_racket_pos_offset_m = float(stage.get("racket_pos_dr_m", 0.0))
            self.cfg.dr_racket_rot_offset_rad = float(
                np.deg2rad(float(stage.get("racket_rot_dr_deg", 0.0)))
            )
            self.cfg.dr_racket_radius_offset_m = float(stage.get("racket_radius_dr_m", 0.0))
            self.cfg.posture_weight = float(stage.get("posture_weight", self.cfg.posture_weight))
            self.cfg.racket_z_soft_penalty_weight = float(
                stage.get("racket_z_soft_penalty_weight", self.cfg.racket_z_soft_penalty_weight)
            )
            self.cfg.failed_hit_penalty_weight = float(
                stage.get("failed_hit_penalty_weight", self.cfg.failed_hit_penalty_weight)
            )
            self.cfg.action_penalty_weight = float(
                stage.get("action_penalty_weight", self.cfg.action_penalty_weight)
            )
            self.cfg.action_delta_penalty_weight = float(
                stage.get(
                    "action_delta_penalty_weight",
                    self.cfg.action_delta_penalty_weight,
                )
            )
            if "target_height" in stage:
                self.cfg.target_height = float(stage["target_height"])
            # The D455 HIL anchor is much higher than the legacy free-base
            # model. Absolute legacy z bounds left only ~10 cm above the
            # paddle and terminated a valid launch before the 4 cm
            # ball--racket separation gate could confirm it.
            self._ball_upper_margin_m = float(stage.get("ball_upper_margin_m", 0.15))
            self._ball_lower_margin_m = float(stage.get("ball_lower_margin_m", 0.35))
            current_racket_z = float(self._racket_pos()[2])
            if current_racket_z != 0.0:
                self.cfg.ball_termination_max_z = (
                    current_racket_z + float(self.cfg.target_height) + self._ball_upper_margin_m
                )
                self.cfg.ball_termination_min_z = current_racket_z - self._ball_lower_margin_m
            self._pre_hit_flatness_weight = float(stage.get("pre_hit_flatness_weight", 0.0))
            self._pre_hit_lift_weight = float(stage.get("pre_hit_lift_weight", 0.0))
            self._pre_hit_launch_weight = float(stage.get("pre_hit_launch_weight", 0.0))
            self._pre_hit_impulse_weight = float(stage.get("pre_hit_impulse_weight", 0.0))
            self._pre_hit_impulse_target_delta_vz = float(
                stage.get("pre_hit_impulse_target_delta_vz", 1.0)
            )
            self._workspace_termination_penalty = float(
                stage.get("workspace_termination_penalty", 0.0)
            )
            self._zero_hit_miss_penalty = float(stage.get("zero_hit_miss_penalty", 0.0))
            self._valid_hit_bonus = float(stage.get("valid_hit_bonus", 0.0))
            self.max_steps = max(1, int(self.cfg.horizon_sec / self.dt))

        def coordinated_exploration_action(
            self,
            *,
            position_offset_m: np.ndarray,
            linear_velocity_m_s: np.ndarray,
            linear_acceleration_m_s2: np.ndarray,
            normal_tilt_xy: np.ndarray,
            angular_velocity_rad_s: np.ndarray,
            angular_acceleration_rad_s2: np.ndarray,
            position_kp: float = 80.0,
            position_kd: float = 16.0,
            orientation_kp: float = 30.0,
            orientation_kd: float = 8.0,
            normal_up_blend: float = 0.0,
            max_action_abs: float = 0.75,
        ) -> np.ndarray:
            """Map smooth racket-centre trajectory noise to joint acceleration.

            This uses only kinematics at the latest prescribed ROS feedback
            state. It is an exploration proposal, not an actuator model or a
            command compensator. The returned action still passes through the
            normal command integrator and all HIL safety limits.
            """
            jac_pos = np.zeros((3, self.model.nv), dtype=np.float64)
            jac_rot = np.zeros((3, self.model.nv), dtype=np.float64)
            mj.mj_jacSite(self.model, self.data, jac_pos, jac_rot, self.racket_site_id)
            arm_dofs = np.asarray(self.arm_vadr, dtype=np.int64)
            j_pos = jac_pos[:, arm_dofs]
            j_rot = jac_rot[:, arm_dofs]
            current_q = np.asarray([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float64)
            current_qvel = np.asarray([self.data.qvel[i] for i in self.arm_vadr], dtype=np.float64)
            racket_pos = self._racket_pos().astype(np.float64)
            if self._exploration_origin_pos is None:
                self._exploration_origin_pos = racket_pos.copy()

            rotation_scale = 0.20
            jacobian = np.vstack((j_pos, rotation_scale * j_rot))
            current_twist = jacobian @ current_qvel
            target_pos = self._exploration_origin_pos + np.asarray(position_offset_m, dtype=np.float64).reshape(3)
            target_linear_velocity = np.asarray(linear_velocity_m_s, dtype=np.float64).reshape(3)
            feedforward_linear_acceleration = np.asarray(
                linear_acceleration_m_s2, dtype=np.float64
            ).reshape(3)
            desired_linear_acceleration = (
                feedforward_linear_acceleration
                + float(position_kp) * (target_pos - racket_pos)
                + float(position_kd) * (target_linear_velocity - current_twist[:3])
            )

            tilt = np.asarray(normal_tilt_xy, dtype=np.float64).reshape(2)
            if self._exploration_origin_normal is None:
                self._exploration_origin_normal = self._racket_xmat()[:, 2].astype(np.float64)
            up_blend = float(np.clip(normal_up_blend, 0.0, 1.0))
            base_normal = (
                (1.0 - up_blend) * self._exploration_origin_normal
                + up_blend * np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            )
            base_normal /= max(np.linalg.norm(base_normal), 1e-9)
            rotation_noise = np.asarray([tilt[0], tilt[1], 0.0], dtype=np.float64)
            target_normal = base_normal + np.cross(rotation_noise, base_normal)
            target_normal /= max(np.linalg.norm(target_normal), 1e-9)
            current_normal = self._racket_xmat()[:, 2].astype(np.float64)
            orientation_error = np.cross(current_normal, target_normal)
            target_angular_velocity = np.asarray(
                angular_velocity_rad_s, dtype=np.float64
            ).reshape(3)
            desired_angular_acceleration = (
                np.asarray(angular_acceleration_rad_s2, dtype=np.float64).reshape(3)
                + float(orientation_kp) * orientation_error
                + float(orientation_kd)
                * (target_angular_velocity - current_twist[3:] / rotation_scale)
            )
            desired_task_acceleration = np.concatenate(
                (desired_linear_acceleration, rotation_scale * desired_angular_acceleration)
            )

            if self._previous_exploration_jacobian is None:
                jdot_qdot = np.zeros(6, dtype=np.float64)
            else:
                jdot_qdot = (
                    (jacobian - self._previous_exploration_jacobian)
                    / max(self._last_source_dt_s, self.dt)
                ) @ current_qvel
            self._previous_exploration_jacobian = jacobian.copy()
            damping = 2.5e-3
            jacobian_pinv = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), np.eye(6)
            )
            desired_qdd = jacobian_pinv @ (desired_task_acceleration - jdot_qdot)
            posture_q = np.deg2rad(np.asarray(self.hil_cfg.reset_degrees, dtype=np.float64))
            nullspace = np.eye(len(arm_dofs)) - jacobian_pinv @ jacobian
            desired_qdd += nullspace @ (
                4.0 * (posture_q - current_q) - 1.5 * current_qvel
            )
            action_denominator = (
                np.deg2rad(np.asarray(self.arm_acc_limit_deg_s2, dtype=np.float64))
                * max(float(self.cfg.action_acc_scale), 1e-3)
            )
            limit = float(np.clip(max_action_abs, 0.05, 1.0))
            return np.clip(desired_qdd / action_denominator, -limit, limit).astype(np.float32)

    return HilJuggleEnv


def build_simcode4_hil_config(simcode4_module: ModuleType):
    """Retain ball/contact DR, while prohibiting synthetic plant dynamics."""
    cfg = simcode4_module.JuggleConfig()
    cfg.domain_randomization = True
    cfg.dr_randomize_ball = True
    cfg.dr_randomize_contact = True
    cfg.dr_randomize_actuator = False
    cfg.dr_randomize_latency = False
    cfg.dr_randomize_racket_mount = True
    cfg.arm_action_limiter = True
    cfg.action_acc_scale = 0.18
    # Start at the racket centre with a 5--10 mm surface gap.  The environment
    # adds the measured MuJoCo ball radius along the racket face normal.
    cfg.ball_spawn_on_racket = True
    cfg.ball_surface_clearance_min = 0.005
    cfg.ball_surface_clearance_max = 0.010
    cfg.ball_spawn_xy_jitter = 0.0
    cfg.ball_spawn_z_jitter = 0.0
    cfg.ball_termination_min_z = 0.35
    cfg.ball_termination_max_z = 1.90
    cfg.hit_confirm_abs_height = 0.68
    cfg.ball_obs_rate_hz = 50.0
    return cfg
