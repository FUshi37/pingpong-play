"""MJX/JAX juggling environment mirroring ``rl_juggle_env_random.JuggleEnv``.

The environment keeps the CPU task's 50-dimensional observation layout and
right-arm acceleration-command action interface while running batched MJX steps.
Each parallel environment carries its own MJX Model pytree so domain
randomization can change model fields such as mass, contact parameters,
damping, armature, gravity, and racket mount geometry per episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from mjx_smoke import _write_mjx_contact_only_xml
from rl_juggle_env_random import RIGHT_ARM_JOINTS, TARGET_DEGREES, _build_temp_xml_with_ball


BASE_ACTS = ("Base-X", "Base-Y", "Base-Yaw")


@dataclass(frozen=True)
class MjxJuggleConfig:
    horizon_sec: float = 6.0
    frame_skip: int = 5
    action_scale_arm_rad: float = 0.03
    action_scale_base_xy: float = 0.020
    action_scale_base_yaw: float = 0.030
    action_acc_scale: float = 1.5
    ball_launch_height: float = 0.30
    ball_spawn_cube_size: float = 0.10
    ball_spawn_xy_jitter: float = 0.0
    ball_spawn_z_jitter: float = 0.0
    ball_init_vxy_max: float = 0.0
    ball_init_vz: float = -0.28
    ball_obs_rate_hz: float = 50.0
    ball_obs_pos_noise_std: float = 0.003
    ball_obs_vel_noise_std: float = 0.03
    total_training_steps: int = 10_000_000
    ball_obs_noise_warmup_ratio: float = 0.10
    ball_obs_noise_ramp_ratio: float = 0.20
    target_height: float = 0.34
    posture_weight: float = 0.02
    base_pose_weight: float = 0.0
    torque_penalty_weight: float = 0.00005
    post_hit_survival_reward_weight: float = 1.4
    post_hit_ball_xy_sigma: float = 0.12
    post_hit_ball_vxy_penalty_weight: float = 0.18
    descending_intercept_reward_weight: float = 1.6
    descending_intercept_sigma: float = 0.10
    non_racket_ball_contact_penalty_weight: float = 1.5
    failed_hit_penalty_weight: float = 1.0
    sticky_contact_penalty_growth: float = 0.6
    hit_reward_base: float = 2.5
    hit_reward_combo: float = 1.2
    rel_height_center: float = 0.18
    rel_height_sigma: float = 0.06
    rel_height_bonus_weight: float = 0.45
    racket_xy_gauss_sigma: float = 0.041
    racket_xy_gauss_reward_weight: float = 0.50
    racket_xy_gauss_penalty_weight: float = 0.60
    racket_chest_xy_penalty_weight: float = 1.0
    racket_chest_z_penalty_weight: float = 0.8
    ball_anchor_xy_penalty_weight: float = 0.7
    ball_base_x_penalty_weight: float = 0.0
    ball_base_x_soft_limit: float = 0.20
    ball_base_vxy_penalty_weight: float = 0.0
    ball_vxy_penalty_weight: float = 0.40
    apex_soft_limit_margin: float = 0.04
    apex_soft_penalty_weight: float = 5.0
    ball_xy_soft_limit_radius: float = 0.14
    ball_xy_soft_penalty_weight: float = 3.0
    racket_z_band_down: float = 0.00
    racket_z_band_up: float = 0.20
    racket_z_soft_penalty_weight: float = 1.2
    racket_up_drift_penalty_weight: float = 0.3
    racket_up_drift_vel_thresh: float = 0.02
    racket_z_hard_limit_down: float = 0.12
    racket_z_hard_limit_up: float = 0.24
    terminate_on_racket_z_limit: bool = True
    racket_z_limit_termination_penalty_base: float = 0.0
    racket_z_limit_termination_penalty_per_hit: float = 0.0
    action_penalty_weight: float = 0.003
    action_delta_penalty_weight: float = 0.001
    termination_miss_penalty_base: float = 2.5
    termination_miss_penalty_per_hit: float = 0.8
    hit_rearm_no_contact_steps: int = 2
    hit_rearm_distance: float = 0.035
    stick_contact_penalty_weight: float = 0.60
    stick_rel_speed_thresh: float = 0.25
    stick_rel_dist_thresh: float = 0.040
    stick_min_contact_steps: int = 4
    hit_confirm_rel_height: float = 0.06
    hit_confirm_abs_height: float = 1.00
    hit_confirm_max_steps: int = 70
    hit_confirm_use_spawn_cube_band: bool = False
    hit_confirm_spawn_band_margin: float = 0.0
    hit_center_local_sigma: float = 0.035
    hit_center_sigma: float = 0.08
    hit_flatness_target_cos: float = 0.96
    hit_flatness_sigma: float = 0.08
    center_flat_hit_reward_weight: float = 1.8
    contact_flatness_penalty_weight: float = 0.45
    hit_height_center: float = 0.52
    hit_height_tolerance: float = 0.06
    hit_height_penalty_weight: float = 10.0
    low_hit_apex_margin: float = 0.06
    low_hit_penalty_weight: float = 10.0
    domain_randomization: bool = True
    dr_randomize_ball: bool = True
    dr_randomize_contact: bool = True
    dr_randomize_actuator: bool = True
    dr_randomize_latency: bool = True
    dr_ball_mass_range: tuple[float, float] = (0.0024, 0.0030)
    dr_ball_friction_range: tuple[float, float] = (0.12, 0.35)
    dr_racket_friction_range: tuple[float, float] = (0.25, 0.55)
    dr_ball_solref_time_range: tuple[float, float] = (0.002, 0.006)
    dr_ball_solref_damping_range: tuple[float, float] = (0.70, 0.95)
    dr_gravity_z_range: tuple[float, float] = (-9.90, -9.70)
    dr_action_scale_mult_range: tuple[float, float] = (0.85, 1.15)
    dr_armature_mult_range: tuple[float, float] = (0.80, 1.20)
    dr_damping_mult_range: tuple[float, float] = (0.70, 1.30)
    dr_obs_latency_steps_range: tuple[int, int] = (0, 2)
    dr_action_latency_steps_range: tuple[int, int] = (0, 2)
    camera_visibility_mode: str = "off"
    virtual_camera_body_name: str = "head22"
    virtual_camera_mount_pos: tuple[float, float, float] = (0.0, -0.068, 0.062)
    virtual_camera_mount_quat: tuple[float, float, float, float] = (0.707107, 0.0, 0.0, -0.707107)
    virtual_camera_optical_pos: tuple[float, float, float] = (0.048, 0.0, 0.0)
    camera_image_width: int = 1280
    camera_image_height: int = 720
    camera_fx: float = 636.99
    camera_fy: float = 636.84
    camera_cx: float = 646.82
    camera_cy: float = 373.21
    camera_hfov_deg: float = 86.0
    camera_vfov_deg: float = 57.0
    camera_min_depth: float = 0.15
    camera_max_depth: float = 2.50
    camera_pixel_margin: float = 80.0
    camera_center_weight: float = 0.0
    camera_visibility_penalty_weight: float = 0.0
    camera_depth_penalty_weight: float = 0.0
    camera_box_penalty_weight: float = 0.0
    camera_visible_penalty_weight: float = 0.0
    camera_top_margin_penalty_weight: float = 0.0
    camera_dense_penalty_clip: float = 20.0
    camera_box_half_width: float = 0.35
    camera_box_half_height: float = 0.35
    camera_box_depth_min: float = 0.20
    camera_box_depth_max: float = 1.50
    arm_action_limiter: bool = False
    arm_vel_limit_deg_s: tuple[float, ...] = (210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0)
    arm_acc_limit_deg_s2: tuple[float, ...] = (1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0)
    arm_vel_limit_penalty_weight: float = 0.0
    arm_acc_limit_penalty_weight: float = 0.002
    arm_limiter_penalty_weight: float = 0.0
    dr_randomize_racket_mount: bool = False
    dr_racket_pos_offset_m: float = 0.0
    dr_racket_rot_offset_rad: float = 0.0
    dr_racket_radius_offset_m: float = 0.0
    hit_cadence_reward_weight: float = 0.0
    hit_cadence_target_interval: float = 0.65
    hit_cadence_sigma: float = 0.18
    hit_min_interval_penalty_weight: float = 0.0
    hit_min_interval: float = 0.40
    hit_min_count_interval: float = 0.0
    fast_hit_penalty_weight: float = 0.0
    hit_reward_cap_mode: str = "off"
    hit_reward_count_cap: int = 0
    hit_reward_cap_target_interval: float = 0.65
    ball_obs_dropout_prob: float = 0.0
    ball_obs_dropout_max_steps: int = 1
    ball_obs_dropout_burst_prob: float = 0.0
    ball_obs_dropout_burst_max_steps: int = 1
    ball_obs_age_clip: float = 0.20


class EnvState(NamedTuple):
    model: object
    data: object
    rng: jax.Array
    step_count: jax.Array
    racket_anchor: jax.Array
    chest_target_offset: jax.Array
    arm_cmd_q: jax.Array
    arm_cmd_qvel: jax.Array
    prev_action: jax.Array
    prev_arm_qvel: jax.Array
    prev_ball_pos: jax.Array
    prev_racket_pos: jax.Array
    prev_contact: jax.Array
    hit_armed: jax.Array
    no_contact_steps: jax.Array
    contact_hold_steps: jax.Array
    pending_hit: jax.Array
    pending_hit_steps: jax.Array
    hit_count: jax.Array
    action_buffer: jax.Array
    action_latency_steps: jax.Array
    obs_buffer: jax.Array
    obs_latency_steps: jax.Array
    cached_ball_obs_pos: jax.Array
    cached_ball_obs_vel: jax.Array
    last_ball_obs_step: jax.Array
    ball_obs_valid_pos: jax.Array
    ball_obs_valid_vel: jax.Array
    ball_obs_age_seconds: jax.Array
    ball_obs_dropout_remaining: jax.Array
    ball_obs_dropout_steps_total: jax.Array
    ball_obs_burst_count: jax.Array
    total_env_steps: jax.Array
    action_scale_mult: jax.Array
    dr_gravity_z: jax.Array
    dr_ball_mass: jax.Array
    dr_ball_friction: jax.Array
    dr_racket_friction: jax.Array
    dr_ball_solref_time: jax.Array
    dr_ball_solref_damping: jax.Array
    dr_damping_mult: jax.Array
    dr_armature_mult: jax.Array
    last_hit_time: jax.Array
    last_counted_hit_time: jax.Array
    last_count_gate_hit_time: jax.Array
    confirmed_hit_count: jax.Array
    ignored_fast_hit_count: jax.Array
    rewarded_hit_count: jax.Array
    unrewarded_extra_hit_count: jax.Array
    dr_racket_pos_offset: jax.Array
    dr_racket_rot_offset: jax.Array
    dr_racket_radius_offset: jax.Array


def _deg_to_rad_map(deg_map: dict[str, float]) -> dict[str, float]:
    return {k: float(np.deg2rad(v)) for k, v in deg_map.items()}


def _quat_wxyz_to_mat_np(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.asarray(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ],
        dtype=np.float32,
    )


def _quat_mul_wxyz_jax(q1: jax.Array, q2: jax.Array) -> jax.Array:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = jnp.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=jnp.float32,
    )
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-8)


def _euler_xyz_to_quat_wxyz_jax(euler_xyz: jax.Array) -> jax.Array:
    roll, pitch, yaw = euler_xyz
    cr, sr = jnp.cos(roll * 0.5), jnp.sin(roll * 0.5)
    cp, sp = jnp.cos(pitch * 0.5), jnp.sin(pitch * 0.5)
    cy, sy = jnp.cos(yaw * 0.5), jnp.sin(yaw * 0.5)
    q = jnp.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=jnp.float32,
    )
    return q / jnp.maximum(jnp.linalg.norm(q), 1e-8)


def _batch_tree(tree, n_envs: int):
    def batch_leaf(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jnp.broadcast_to(x, (n_envs,) + tuple(x.shape))
        return x

    return jax.tree_util.tree_map(batch_leaf, tree)


class MjxJuggleEnv:
    obs_dim = 50
    act_dim = 7

    def __init__(self, xml_path: str | Path, n_envs: int, cfg: MjxJuggleConfig = MjxJuggleConfig()) -> None:
        self.xml_path = Path(xml_path).resolve()
        self.n_envs = int(n_envs)
        self.cfg = cfg

        patched_xml = _build_temp_xml_with_ball(self.xml_path)
        self.mjx_xml = _write_mjx_contact_only_xml(patched_xml)
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.mjx_xml))
        self.model = mjx.put_model(self.mj_model)

        self.timestep = float(self.mj_model.opt.timestep)
        self.dt = float(self.timestep * cfg.frame_skip)
        self.max_steps = max(1, int(cfg.horizon_sec / self.dt))

        self.arm_jids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_ARM_JOINTS]
        self.arm_aids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in RIGHT_ARM_JOINTS]
        self.base_aids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in BASE_ACTS]
        self.arm_qadr = jnp.asarray([int(self.mj_model.jnt_qposadr[j]) for j in self.arm_jids], dtype=jnp.int32)
        self.arm_vadr = jnp.asarray([int(self.mj_model.jnt_dofadr[j]) for j in self.arm_jids], dtype=jnp.int32)
        self.arm_aids_j = jnp.asarray(self.arm_aids, dtype=jnp.int32)
        self.base_aids_j = jnp.asarray(self.base_aids, dtype=jnp.int32)
        self.arm_lo = jnp.asarray([self.mj_model.jnt_range[j, 0] for j in self.arm_jids], dtype=jnp.float32)
        self.arm_hi = jnp.asarray([self.mj_model.jnt_range[j, 1] for j in self.arm_jids], dtype=jnp.float32)

        self.ball_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pingpong_ball")
        self.ball_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        self.racket_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")
        self.racket_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_racket")
        self.racket_wood_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "racket_wood")
        self.racket_rubber_geom_id = self.racket_geom_id
        self.racket_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")
        self.waist_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "waist03")
        self.base_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.virtual_camera_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, str(cfg.virtual_camera_body_name)
        )
        non_racket_gids = []
        for gid in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if name and name.startswith("mjx_ball_contact_"):
                non_racket_gids.append(int(gid))
        self.non_racket_geom_ids = jnp.asarray(non_racket_gids or [-1], dtype=jnp.int32)
        ball_racket_pair_ids = []
        for pid in range(self.mj_model.npair):
            g1 = int(self.mj_model.pair_geom1[pid])
            g2 = int(self.mj_model.pair_geom2[pid])
            if {g1, g2} == {self.ball_geom_id, self.racket_geom_id}:
                ball_racket_pair_ids.append(pid)
        self.has_ball_racket_pair = bool(ball_racket_pair_ids)
        self.ball_racket_pair_ids = jnp.asarray(ball_racket_pair_ids or [-1], dtype=jnp.int32)
        self.ball_qadr = int(self.mj_model.jnt_qposadr[self.ball_joint_id])
        self.ball_vadr = int(self.mj_model.jnt_dofadr[self.ball_joint_id])

        bx = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_x")
        by = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_y")
        byaw = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "base_yaw")
        self.base_x_qadr = int(self.mj_model.jnt_qposadr[bx])
        self.base_y_qadr = int(self.mj_model.jnt_qposadr[by])
        self.base_yaw_qadr = int(self.mj_model.jnt_qposadr[byaw])
        self.base_x_vadr = int(self.mj_model.jnt_dofadr[bx])
        self.base_y_vadr = int(self.mj_model.jnt_dofadr[by])
        self.base_yaw_vadr = int(self.mj_model.jnt_dofadr[byaw])

        target_rad = _deg_to_rad_map(TARGET_DEGREES)
        posture_names = list(target_rad.keys())
        posture_jids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in posture_names]
        posture_aids = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in posture_names]
        self.posture_qadr = jnp.asarray([int(self.mj_model.jnt_qposadr[j]) for j in posture_jids], dtype=jnp.int32)
        self.posture_targets = jnp.asarray([target_rad[n] for n in posture_names], dtype=jnp.float32)

        default_ctrl = np.zeros(self.mj_model.nu, dtype=np.float32)
        for n, aid in zip(posture_names, posture_aids):
            if aid >= 0:
                default_ctrl[aid] = np.float32(target_rad[n])
        for aid in self.base_aids:
            if aid >= 0:
                default_ctrl[aid] = 0.0
        self.default_ctrl = jnp.asarray(default_ctrl, dtype=jnp.float32)

        warm = mujoco.MjData(self.mj_model)
        warm.ctrl[:] = default_ctrl
        for _ in range(700):
            mujoco.mj_step(self.mj_model, warm)
        mujoco.mj_forward(self.mj_model, warm)

        self.warm_qpos = jnp.asarray(warm.qpos, dtype=jnp.float32)
        self.warm_qvel = jnp.asarray(warm.qvel, dtype=jnp.float32)
        self.warm_ctrl = jnp.asarray(default_ctrl, dtype=jnp.float32)
        self.warm_arm_q = self.warm_qpos[self.arm_qadr]
        self.warm_arm_qvel = self.warm_qvel[self.arm_vadr]
        self.racket_anchor = jnp.asarray(warm.site_xpos[self.racket_site_id], dtype=jnp.float32)
        if self.waist_body_id >= 0:
            self.chest_target_offset = jnp.asarray(
                warm.site_xpos[self.racket_site_id] - warm.xpos[self.waist_body_id],
                dtype=jnp.float32,
            )
        else:
            self.chest_target_offset = jnp.zeros((3,), dtype=jnp.float32)
        self.initial_base_pose = jnp.asarray(
            [
                warm.qpos[self.base_x_qadr],
                warm.qpos[self.base_y_qadr],
                warm.qpos[self.base_yaw_qadr],
            ],
            dtype=jnp.float32,
        )

        self.arm_vel_limit_rad_s = jnp.deg2rad(jnp.asarray(cfg.arm_vel_limit_deg_s, dtype=jnp.float32))
        self.arm_acc_limit_rad_s2 = jnp.deg2rad(jnp.asarray(cfg.arm_acc_limit_deg_s2, dtype=jnp.float32))
        self.default_gravity_z = float(self.mj_model.opt.gravity[2])
        self.gravity_mag = float(np.linalg.norm(self.mj_model.opt.gravity))
        self.original_ball_mass = float(self.mj_model.body_mass[self.ball_body_id]) if self.ball_body_id >= 0 else 0.0027
        self.original_ball_inertia = (
            jnp.asarray(self.mj_model.body_inertia[self.ball_body_id], dtype=jnp.float32)
            if self.ball_body_id >= 0
            else jnp.ones((3,), dtype=jnp.float32)
        )
        self.original_ball_friction = float(self.mj_model.geom_friction[self.ball_geom_id, 0]) if self.ball_geom_id >= 0 else 0.20
        self.original_ball_solref_time = float(self.mj_model.geom_solref[self.ball_geom_id, 0]) if self.ball_geom_id >= 0 else 0.003
        self.original_ball_solref_damping = float(self.mj_model.geom_solref[self.ball_geom_id, 1]) if self.ball_geom_id >= 0 else 0.80
        self.original_racket_friction = (
            float(self.mj_model.geom_friction[self.racket_geom_id, 0]) if self.racket_geom_id >= 0 else 0.35
        )
        self.original_dof_damping = jnp.asarray(self.mj_model.dof_damping, dtype=jnp.float32)
        self.original_dof_armature = jnp.asarray(self.mj_model.dof_armature, dtype=jnp.float32)
        self.original_racket_body_pos = (
            jnp.asarray(self.mj_model.body_pos[self.racket_body_id], dtype=jnp.float32)
            if self.racket_body_id >= 0
            else jnp.zeros((3,), dtype=jnp.float32)
        )
        self.original_racket_body_quat = (
            jnp.asarray(self.mj_model.body_quat[self.racket_body_id], dtype=jnp.float32)
            if self.racket_body_id >= 0
            else jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
        )
        self.racket_mount_geom_ids = jnp.asarray(
            [gid for gid in (self.racket_wood_geom_id, self.racket_rubber_geom_id) if gid >= 0] or [-1],
            dtype=jnp.int32,
        )
        self.has_racket_mount_geoms = bool([gid for gid in (self.racket_wood_geom_id, self.racket_rubber_geom_id) if gid >= 0])
        self.original_racket_mount_geom_sizes = jnp.asarray(
            [
                self.mj_model.geom_size[gid]
                for gid in (self.racket_wood_geom_id, self.racket_rubber_geom_id)
                if gid >= 0
            ]
            or [np.zeros(3, dtype=np.float32)],
            dtype=jnp.float32,
        )
        self.ball_obs_every = 1
        if float(cfg.ball_obs_rate_hz) > 0.0:
            self.ball_obs_every = max(1, int(round(1.0 / (float(cfg.ball_obs_rate_hz) * self.dt))))
        self.max_obs_latency_steps = max(0, int(cfg.dr_obs_latency_steps_range[1])) if cfg.domain_randomization else 0
        self.max_action_latency_steps = max(0, int(cfg.dr_action_latency_steps_range[1])) if cfg.domain_randomization else 0
        self.hit_reward_count_cap_active = self._get_hit_reward_count_cap()
        self.vc_mount_R = jnp.asarray(_quat_wxyz_to_mat_np(cfg.virtual_camera_mount_quat), dtype=jnp.float32)
        self.vc_mount_pos = jnp.asarray(cfg.virtual_camera_mount_pos, dtype=jnp.float32)
        self.vc_optical_pos = jnp.asarray(cfg.virtual_camera_optical_pos, dtype=jnp.float32)
        self.vc_mount_to_camera_R = jnp.asarray(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=jnp.float32,
        )
        self.base_data = mjx.make_data(self.model).replace(
            qpos=self.warm_qpos,
            qvel=self.warm_qvel,
            ctrl=self.warm_ctrl,
        )
        self.base_data = mjx.forward(self.model, self.base_data)
        self.batched_step = jax.vmap(lambda model, data: mjx.step(model, data))
        self.batched_forward = jax.vmap(lambda model, data: mjx.forward(model, data))

    def _get_hit_reward_count_cap(self) -> int:
        if self.cfg.hit_reward_cap_mode == "off":
            return 0
        if self.cfg.hit_reward_cap_mode == "fixed":
            return max(0, int(self.cfg.hit_reward_count_cap))
        if self.cfg.hit_reward_cap_mode == "auto":
            episode_total_time = float(self.max_steps) * self.dt
            cap = int(np.floor(episode_total_time / max(1e-6, float(self.cfg.hit_reward_cap_target_interval))))
            return max(1, cap)
        raise ValueError(
            f"Invalid hit_reward_cap_mode={self.cfg.hit_reward_cap_mode!r}; expected 'off', 'auto', or 'fixed'."
        )

    def _make_batched_model(
        self,
        n_envs: int,
        dr_gravity_z: jax.Array,
        dr_ball_mass: jax.Array,
        dr_ball_friction: jax.Array,
        dr_racket_friction: jax.Array,
        dr_ball_solref_time: jax.Array,
        dr_ball_solref_damping: jax.Array,
        dr_damping_mult: jax.Array,
        dr_armature_mult: jax.Array,
        dr_racket_pos_offset: jax.Array,
        dr_racket_rot_offset: jax.Array,
        dr_racket_radius_offset: jax.Array,
    ):
        model = _batch_tree(self.model, n_envs)

        opt = model.opt.replace(gravity=model.opt.gravity.at[:, 2].set(dr_gravity_z))
        model = model.replace(opt=opt)

        if self.ball_body_id >= 0:
            mass_mult = dr_ball_mass / max(self.original_ball_mass, 1e-9)
            body_mass = model.body_mass.at[:, self.ball_body_id].set(dr_ball_mass)
            body_inertia = model.body_inertia.at[:, self.ball_body_id, :].set(
                self.original_ball_inertia[None, :] * mass_mult[:, None]
            )
            model = model.replace(body_mass=body_mass, body_inertia=body_inertia)

        dof_damping = self.original_dof_damping[None, :] * dr_damping_mult[:, None]
        dof_armature = self.original_dof_armature[None, :] * dr_armature_mult[:, None]
        model = model.replace(dof_damping=dof_damping, dof_armature=dof_armature)

        geom_friction = model.geom_friction
        geom_solref = model.geom_solref
        if self.ball_geom_id >= 0:
            geom_friction = geom_friction.at[:, self.ball_geom_id, 0].set(dr_ball_friction)
            geom_solref = geom_solref.at[:, self.ball_geom_id, 0].set(dr_ball_solref_time)
            geom_solref = geom_solref.at[:, self.ball_geom_id, 1].set(dr_ball_solref_damping)
        if self.racket_geom_id >= 0:
            geom_friction = geom_friction.at[:, self.racket_geom_id, 0].set(dr_racket_friction)
        model = model.replace(geom_friction=geom_friction, geom_solref=geom_solref)

        if self.has_ball_racket_pair:
            pair_friction = model.pair_friction.at[:, self.ball_racket_pair_ids, 0].set(
                dr_racket_friction[:, None]
            )
            pair_solref = model.pair_solref.at[:, self.ball_racket_pair_ids, 0].set(
                dr_ball_solref_time[:, None]
            )
            pair_solref = pair_solref.at[:, self.ball_racket_pair_ids, 1].set(
                dr_ball_solref_damping[:, None]
            )
            model = model.replace(pair_friction=pair_friction, pair_solref=pair_solref)

        if self.racket_body_id >= 0:
            rot_quat = jax.vmap(_euler_xyz_to_quat_wxyz_jax)(dr_racket_rot_offset)
            racket_quat = jax.vmap(lambda q: _quat_mul_wxyz_jax(self.original_racket_body_quat, q))(rot_quat)
            body_pos = model.body_pos.at[:, self.racket_body_id, :].set(
                self.original_racket_body_pos[None, :] + dr_racket_pos_offset
            )
            body_quat = model.body_quat.at[:, self.racket_body_id, :].set(racket_quat)
            model = model.replace(body_pos=body_pos, body_quat=body_quat)

        if self.has_racket_mount_geoms:
            new_sizes = jnp.broadcast_to(
                self.original_racket_mount_geom_sizes[None, :, :],
                (n_envs,) + tuple(self.original_racket_mount_geom_sizes.shape),
            )
            radius = jnp.maximum(0.03, new_sizes[:, :, 0] + dr_racket_radius_offset[:, None])
            new_sizes = new_sizes.at[:, :, 0].set(radius)
            geom_size = model.geom_size.at[:, self.racket_mount_geom_ids, :].set(new_sizes)
            model = model.replace(geom_size=geom_size)

        return model

    def reset(self, keys: jax.Array) -> tuple[EnvState, jax.Array]:
        keys = jnp.asarray(keys)
        n_envs = keys.shape[0]
        data = _batch_tree(self.base_data, n_envs)

        split_keys = jax.vmap(lambda k: jax.random.split(k, 17))(keys)
        next_keys = split_keys[:, 0]
        key_xy = split_keys[:, 1]
        key_z = split_keys[:, 2]
        key_vel = split_keys[:, 3]
        key_action_scale = split_keys[:, 4]
        key_gravity = split_keys[:, 5]
        key_ball_mass = split_keys[:, 6]
        key_ball_friction = split_keys[:, 7]
        key_racket_friction = split_keys[:, 8]
        key_solref = split_keys[:, 9]
        key_obs_latency = split_keys[:, 10]
        key_action_latency = split_keys[:, 11]
        key_damping = split_keys[:, 12]
        key_armature = split_keys[:, 13]
        key_racket_pos = split_keys[:, 14]
        key_racket_rot = split_keys[:, 15]
        key_racket_radius = split_keys[:, 16]
        xy_jitter = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (2,),
                minval=-float(self.cfg.ball_spawn_xy_jitter),
                maxval=float(self.cfg.ball_spawn_xy_jitter),
            )
        )(key_xy)
        z_jitter = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (),
                minval=-float(self.cfg.ball_spawn_z_jitter),
                maxval=float(self.cfg.ball_spawn_z_jitter),
            )
        )(key_z)
        vxy = jax.vmap(
            lambda k: jax.random.uniform(
                k,
                (2,),
                minval=-float(self.cfg.ball_init_vxy_max),
                maxval=float(self.cfg.ball_init_vxy_max),
            )
        )(key_vel)

        zero_action = jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32)
        zero_ball_vel = jnp.zeros((n_envs, 3), dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_actuator):
            action_scale_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_action_scale_mult_range[0]),
                    maxval=float(self.cfg.dr_action_scale_mult_range[1]),
                )
            )(key_action_scale)
            dr_damping_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_damping_mult_range[0]),
                    maxval=float(self.cfg.dr_damping_mult_range[1]),
                )
            )(key_damping)
            dr_armature_mult = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_armature_mult_range[0]),
                    maxval=float(self.cfg.dr_armature_mult_range[1]),
                )
            )(key_armature)
        else:
            action_scale_mult = jnp.ones((n_envs,), dtype=jnp.float32)
            dr_damping_mult = jnp.ones((n_envs,), dtype=jnp.float32)
            dr_armature_mult = jnp.ones((n_envs,), dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_ball):
            dr_gravity_z = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_gravity_z_range[0]),
                    maxval=float(self.cfg.dr_gravity_z_range[1]),
                )
            )(key_gravity)
            dr_ball_mass = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_ball_mass_range[0]),
                    maxval=float(self.cfg.dr_ball_mass_range[1]),
                )
            )(key_ball_mass)
        else:
            dr_gravity_z = jnp.full((n_envs,), self.default_gravity_z, dtype=jnp.float32)
            dr_ball_mass = jnp.full((n_envs,), self.original_ball_mass, dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_contact):
            dr_ball_friction = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_ball_friction_range[0]),
                    maxval=float(self.cfg.dr_ball_friction_range[1]),
                )
            )(key_ball_friction)
            dr_racket_friction = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=float(self.cfg.dr_racket_friction_range[0]),
                    maxval=float(self.cfg.dr_racket_friction_range[1]),
                )
            )(key_racket_friction)
            solref_samples = jax.vmap(
                lambda k: jnp.asarray(
                    [
                        jax.random.uniform(
                            k,
                            (),
                            minval=float(self.cfg.dr_ball_solref_time_range[0]),
                            maxval=float(self.cfg.dr_ball_solref_time_range[1]),
                        ),
                        jax.random.uniform(
                            jax.random.fold_in(k, 1),
                            (),
                            minval=float(self.cfg.dr_ball_solref_damping_range[0]),
                            maxval=float(self.cfg.dr_ball_solref_damping_range[1]),
                        ),
                    ],
                    dtype=jnp.float32,
                )
            )(key_solref)
            dr_ball_solref_time = solref_samples[:, 0]
            dr_ball_solref_damping = solref_samples[:, 1]
        else:
            dr_ball_friction = jnp.full((n_envs,), self.original_ball_friction, dtype=jnp.float32)
            dr_racket_friction = jnp.full((n_envs,), self.original_racket_friction, dtype=jnp.float32)
            dr_ball_solref_time = jnp.full((n_envs,), self.original_ball_solref_time, dtype=jnp.float32)
            dr_ball_solref_damping = jnp.full((n_envs,), self.original_ball_solref_damping, dtype=jnp.float32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_latency):
            obs_low, obs_high = [int(v) for v in self.cfg.dr_obs_latency_steps_range]
            act_low, act_high = [int(v) for v in self.cfg.dr_action_latency_steps_range]
            obs_latency_steps = jax.vmap(
                lambda k: jax.random.randint(k, (), minval=obs_low, maxval=max(obs_low + 1, obs_high + 1), dtype=jnp.int32)
            )(key_obs_latency)
            action_latency_steps = jax.vmap(
                lambda k: jax.random.randint(k, (), minval=act_low, maxval=max(act_low + 1, act_high + 1), dtype=jnp.int32)
            )(key_action_latency)
        else:
            obs_latency_steps = jnp.zeros((n_envs,), dtype=jnp.int32)
            action_latency_steps = jnp.zeros((n_envs,), dtype=jnp.int32)

        if bool(self.cfg.domain_randomization and self.cfg.dr_randomize_racket_mount):
            dr_racket_pos_offset = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (3,),
                    minval=-float(self.cfg.dr_racket_pos_offset_m),
                    maxval=float(self.cfg.dr_racket_pos_offset_m),
                )
            )(key_racket_pos)
            dr_racket_rot_offset = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (3,),
                    minval=-float(self.cfg.dr_racket_rot_offset_rad),
                    maxval=float(self.cfg.dr_racket_rot_offset_rad),
                )
            )(key_racket_rot)
            dr_racket_radius_offset = jax.vmap(
                lambda k: jax.random.uniform(
                    k,
                    (),
                    minval=-float(self.cfg.dr_racket_radius_offset_m),
                    maxval=float(self.cfg.dr_racket_radius_offset_m),
                )
            )(key_racket_radius)
        else:
            dr_racket_pos_offset = jnp.zeros((n_envs, 3), dtype=jnp.float32)
            dr_racket_rot_offset = jnp.zeros((n_envs, 3), dtype=jnp.float32)
            dr_racket_radius_offset = jnp.zeros((n_envs,), dtype=jnp.float32)

        model = self._make_batched_model(
            n_envs=n_envs,
            dr_gravity_z=dr_gravity_z,
            dr_ball_mass=dr_ball_mass,
            dr_ball_friction=dr_ball_friction,
            dr_racket_friction=dr_racket_friction,
            dr_ball_solref_time=dr_ball_solref_time,
            dr_ball_solref_damping=dr_ball_solref_damping,
            dr_damping_mult=dr_damping_mult,
            dr_armature_mult=dr_armature_mult,
            dr_racket_pos_offset=dr_racket_pos_offset,
            dr_racket_rot_offset=dr_racket_rot_offset,
            dr_racket_radius_offset=dr_racket_radius_offset,
        )
        data = self.batched_forward(model, data)
        reset_racket_anchor = data.site_xpos[:, self.racket_site_id]
        if self.waist_body_id >= 0:
            chest_target_offset = reset_racket_anchor - data.xpos[:, self.waist_body_id]
        else:
            chest_target_offset = jnp.zeros((n_envs, 3), dtype=jnp.float32)

        ball_init = jnp.concatenate(
            [
                reset_racket_anchor[:, :2] + xy_jitter,
                (reset_racket_anchor[:, 2] + float(self.cfg.ball_launch_height) + z_jitter)[:, None],
            ],
            axis=-1,
        )
        qpos = data.qpos
        qvel = data.qvel
        qpos = qpos.at[:, self.ball_qadr : self.ball_qadr + 3].set(ball_init)
        qpos = qpos.at[:, self.ball_qadr + 3 : self.ball_qadr + 7].set(
            jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32), (n_envs, 4))
        )
        qvel = qvel.at[:, self.ball_vadr : self.ball_vadr + 6].set(0.0)
        qvel = qvel.at[:, self.ball_vadr : self.ball_vadr + 2].set(vxy)
        qvel = qvel.at[:, self.ball_vadr + 2].set(float(self.cfg.ball_init_vz))
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.broadcast_to(self.default_ctrl, (n_envs, self.mj_model.nu)))
        data = self.batched_forward(model, data)

        bpos = data.xpos[:, self.ball_body_id]
        rpos = data.site_xpos[:, self.racket_site_id]

        state = EnvState(
            model=model,
            data=data,
            rng=next_keys,
            step_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            racket_anchor=reset_racket_anchor,
            chest_target_offset=chest_target_offset,
            arm_cmd_q=jnp.broadcast_to(self.warm_arm_q, (n_envs, self.act_dim)),
            arm_cmd_qvel=jnp.zeros((n_envs, self.act_dim), dtype=jnp.float32),
            prev_action=zero_action,
            prev_arm_qvel=jnp.broadcast_to(self.warm_arm_qvel, (n_envs, self.act_dim)),
            prev_ball_pos=bpos,
            prev_racket_pos=rpos,
            prev_contact=jnp.zeros((n_envs,), dtype=bool),
            hit_armed=jnp.ones((n_envs,), dtype=bool),
            no_contact_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            contact_hold_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            pending_hit=jnp.zeros((n_envs,), dtype=bool),
            pending_hit_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            action_buffer=jnp.zeros((n_envs, self.max_action_latency_steps + 1, self.act_dim), dtype=jnp.float32),
            action_latency_steps=action_latency_steps,
            obs_buffer=jnp.zeros((n_envs, self.max_obs_latency_steps + 1, self.obs_dim), dtype=jnp.float32),
            obs_latency_steps=obs_latency_steps,
            cached_ball_obs_pos=bpos,
            cached_ball_obs_vel=zero_ball_vel,
            last_ball_obs_step=jnp.zeros((n_envs,), dtype=jnp.int32),
            ball_obs_valid_pos=bpos,
            ball_obs_valid_vel=zero_ball_vel,
            ball_obs_age_seconds=jnp.zeros((n_envs,), dtype=jnp.float32),
            ball_obs_dropout_remaining=jnp.zeros((n_envs,), dtype=jnp.int32),
            ball_obs_dropout_steps_total=jnp.zeros((n_envs,), dtype=jnp.int32),
            ball_obs_burst_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            total_env_steps=jnp.zeros((n_envs,), dtype=jnp.int32),
            action_scale_mult=action_scale_mult,
            dr_gravity_z=dr_gravity_z,
            dr_ball_mass=dr_ball_mass,
            dr_ball_friction=dr_ball_friction,
            dr_racket_friction=dr_racket_friction,
            dr_ball_solref_time=dr_ball_solref_time,
            dr_ball_solref_damping=dr_ball_solref_damping,
            dr_damping_mult=dr_damping_mult,
            dr_armature_mult=dr_armature_mult,
            last_hit_time=jnp.full((n_envs,), -1.0, dtype=jnp.float32),
            last_counted_hit_time=jnp.full((n_envs,), -1.0, dtype=jnp.float32),
            last_count_gate_hit_time=jnp.full((n_envs,), -1.0, dtype=jnp.float32),
            confirmed_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            ignored_fast_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            rewarded_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            unrewarded_extra_hit_count=jnp.zeros((n_envs,), dtype=jnp.int32),
            dr_racket_pos_offset=dr_racket_pos_offset,
            dr_racket_rot_offset=dr_racket_rot_offset,
            dr_racket_radius_offset=dr_racket_radius_offset,
        )
        obs = self.observe(state)
        state = state._replace(obs_buffer=jnp.broadcast_to(obs[:, None, :], (n_envs, self.max_obs_latency_steps + 1, self.obs_dim)))
        return state, obs

    def observe(self, state: EnvState) -> jax.Array:
        return self._make_obs(
            state,
            state.ball_obs_valid_pos,
            state.ball_obs_valid_vel,
            state.ball_obs_age_seconds,
        )

    def _make_obs(
        self,
        state: EnvState,
        ball_obs_pos: jax.Array,
        ball_obs_vel: jax.Array,
        ball_obs_age_seconds: jax.Array,
    ) -> jax.Array:
        data = state.data
        q = data.qpos[:, self.arm_qadr]
        dq = data.qvel[:, self.arm_vadr]
        base_q = jnp.stack(
            [
                data.qpos[:, self.base_x_qadr],
                data.qpos[:, self.base_y_qadr],
                data.qpos[:, self.base_yaw_qadr],
            ],
            axis=-1,
        )
        base_dq = jnp.stack(
            [
                data.qvel[:, self.base_x_vadr],
                data.qvel[:, self.base_y_vadr],
                data.qvel[:, self.base_yaw_vadr],
            ],
            axis=-1,
        )
        rpos = data.site_xpos[:, self.racket_site_id]
        rvel = (rpos - state.prev_racket_pos) / max(self.dt, 1e-6)
        bpos_base = self._point_to_base(ball_obs_pos, base_q)
        rpos_base = self._point_to_base(rpos, base_q)
        bvel_base = self._vel_to_base(ball_obs_vel, ball_obs_pos, base_q, base_dq)
        rvel_base = self._vel_to_base(rvel, rpos, base_q, base_dq)
        rel_base = bpos_base - rpos_base
        arm_cmd_error = state.arm_cmd_q - q
        age = jnp.clip(ball_obs_age_seconds / max(1e-6, float(self.cfg.ball_obs_age_clip)), 0.0, 1.0)[:, None]
        return jnp.concatenate(
            [
                q,
                dq,
                base_q,
                base_dq,
                bpos_base,
                bvel_base,
                rpos_base,
                rvel_base,
                rel_base,
                state.prev_action,
                arm_cmd_error,
                age,
            ],
            axis=-1,
        )

    def step(self, state: EnvState, action: jax.Array) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        policy_action = jnp.clip(action, -1.0, 1.0)
        action_buffer = jnp.concatenate([state.action_buffer[:, 1:, :], policy_action[:, None, :]], axis=1)
        action_idx = (self.max_action_latency_steps - state.action_latency_steps).astype(jnp.int32)
        action = action_buffer[jnp.arange(action_buffer.shape[0]), action_idx]
        da = action - state.prev_action

        desired_qdd_raw = action * self.arm_acc_limit_rad_s2 * float(self.cfg.action_acc_scale) * state.action_scale_mult[:, None]
        if bool(self.cfg.arm_action_limiter):
            desired_qdd = jnp.clip(desired_qdd_raw, -self.arm_acc_limit_rad_s2, self.arm_acc_limit_rad_s2)
        else:
            desired_qdd = desired_qdd_raw
        raw_cmd_qvel = state.arm_cmd_qvel + desired_qdd * self.dt
        if bool(self.cfg.arm_action_limiter):
            cmd_qvel = jnp.clip(raw_cmd_qvel, -self.arm_vel_limit_rad_s, self.arm_vel_limit_rad_s)
        else:
            cmd_qvel = raw_cmd_qvel
        arm_cmd_q = jnp.clip(state.arm_cmd_q + cmd_qvel * self.dt, self.arm_lo, self.arm_hi)

        acc_clip_diff = desired_qdd_raw - desired_qdd
        vel_clip_diff = raw_cmd_qvel - cmd_qvel
        arm_limiter_pen = jnp.mean(
            vel_clip_diff**2 / (self.arm_vel_limit_rad_s**2 + 1e-8)
            + acc_clip_diff**2 / (self.arm_acc_limit_rad_s2**2 + 1e-8),
            axis=-1,
        )

        ctrl = jnp.broadcast_to(self.default_ctrl, (action.shape[0], self.mj_model.nu))
        ctrl = ctrl.at[:, self.arm_aids_j].set(arm_cmd_q)
        ctrl = ctrl.at[:, self.base_aids_j].set(0.0)
        data = state.data.replace(ctrl=ctrl)

        def one_substep(_, d):
            return self.batched_step(state.model, d.replace(ctrl=ctrl))

        data = jax.lax.fori_loop(0, int(self.cfg.frame_skip), one_substep, data)

        step_count = state.step_count + 1
        bpos = data.xpos[:, self.ball_body_id]
        rpos = data.site_xpos[:, self.racket_site_id]
        rmat = data.site_xmat[:, self.racket_site_id].reshape((-1, 3, 3))
        racket_normal = rmat[:, :, 2]
        bvel = (bpos - state.prev_ball_pos) / max(self.dt, 1e-6)
        rvel = (rpos - state.prev_racket_pos) / max(self.dt, 1e-6)
        rel = bpos - rpos
        rel_local = jnp.einsum("nij,nj->ni", jnp.swapaxes(rmat, 1, 2), rel)
        in_contact, other_ball_contact = self._ball_contact_flags(data)

        sep_dist = jnp.linalg.norm(rel, axis=-1)
        no_contact_steps = jnp.where(in_contact, 0, state.no_contact_steps + 1)
        contact_hold_steps = jnp.where(in_contact, state.contact_hold_steps + 1, 0)
        hit_armed = jnp.where(
            (~in_contact)
            & (no_contact_steps >= int(self.cfg.hit_rearm_no_contact_steps))
            & (sep_dist >= float(self.cfg.hit_rearm_distance)),
            True,
            state.hit_armed,
        )

        hit_edge = in_contact & (~state.prev_contact) & hit_armed & (~state.pending_hit)
        pending_hit = state.pending_hit | hit_edge
        pending_steps = jnp.where(pending_hit, state.pending_hit_steps + 1, 0)
        hit_armed = jnp.where(hit_edge, False, hit_armed)

        upward_vz = jnp.maximum(0.0, bvel[:, 2])
        gravity_mag = jnp.maximum(jnp.abs(state.dr_gravity_z), 1e-6)
        predicted_apex_z = bpos[:, 2] + (upward_vz * upward_vz) / (2.0 * gravity_mag)
        min_launch_rel_z = max(float(self.cfg.hit_confirm_rel_height), 0.04)
        min_launch_apex_z = state.racket_anchor[:, 2] + max(0.70 * float(self.cfg.target_height), min_launch_rel_z + 0.06)
        launched_upward_raw = (
            pending_hit
            & (~in_contact)
            & (rel[:, 2] >= min_launch_rel_z)
            & (bvel[:, 2] > 0.0)
            & (predicted_apex_z >= min_launch_apex_z)
        )
        current_time = step_count.astype(jnp.float32) * self.dt
        count_gate_interval = current_time - state.last_count_gate_hit_time
        counted_hit = launched_upward_raw & (
            (float(self.cfg.hit_min_count_interval) <= 0.0)
            | (state.last_count_gate_hit_time < 0.0)
            | (count_gate_interval >= float(self.cfg.hit_min_count_interval))
        )
        ignored_fast_hit = launched_upward_raw & (~counted_hit)
        cap = int(self.hit_reward_count_cap_active)
        rewardable_hit = counted_hit & ((cap <= 0) | (state.rewarded_hit_count < cap))
        unrewarded_extra_hit = counted_hit & (~rewardable_hit)
        launched_upward = counted_hit
        failed_hit = pending_hit & (pending_steps >= int(self.cfg.hit_confirm_max_steps)) & (~launched_upward_raw)
        hit_count = state.hit_count + counted_hit.astype(jnp.int32)
        confirmed_hit_count = state.confirmed_hit_count + launched_upward_raw.astype(jnp.int32)
        ignored_fast_hit_count = state.ignored_fast_hit_count + ignored_fast_hit.astype(jnp.int32)
        rewarded_hit_count = state.rewarded_hit_count + rewardable_hit.astype(jnp.int32)
        unrewarded_extra_hit_count = state.unrewarded_extra_hit_count + unrewarded_extra_hit.astype(jnp.int32)
        last_count_gate_hit_time = jnp.where(launched_upward_raw, current_time, state.last_count_gate_hit_time)
        last_counted_hit_time = jnp.where(counted_hit, current_time, state.last_counted_hit_time)
        hit_interval = current_time - state.last_hit_time
        has_prev_hit = state.last_hit_time >= 0.0
        cadence_eligible = launched_upward_raw & counted_hit & rewardable_hit & has_prev_hit
        hit_cadence_reward = jnp.where(
            cadence_eligible & (float(self.cfg.hit_cadence_reward_weight) > 0.0),
            float(self.cfg.hit_cadence_reward_weight)
            * jnp.exp(
                -0.5
                * (
                    (hit_interval - float(self.cfg.hit_cadence_target_interval))
                    / max(1e-6, float(self.cfg.hit_cadence_sigma))
                )
                ** 2
            ),
            0.0,
        )
        hit_min_interval_penalty = jnp.where(
            cadence_eligible
            & (float(self.cfg.hit_min_interval_penalty_weight) > 0.0)
            & (hit_interval < float(self.cfg.hit_min_interval)),
            float(self.cfg.hit_min_interval_penalty_weight)
            * (
                (float(self.cfg.hit_min_interval) - hit_interval)
                / max(1e-6, float(self.cfg.hit_min_interval))
            )
            ** 2,
            0.0,
        )
        fast_hit_penalty = jnp.where(
            ignored_fast_hit
            & (float(self.cfg.fast_hit_penalty_weight) > 0.0)
            & (float(self.cfg.hit_min_count_interval) > 0.0),
            float(self.cfg.fast_hit_penalty_weight)
            * (
                (float(self.cfg.hit_min_count_interval) - count_gate_interval)
                / max(1e-6, float(self.cfg.hit_min_count_interval))
            )
            ** 2,
            0.0,
        )
        last_hit_time = jnp.where(launched_upward_raw, current_time, state.last_hit_time)
        pending_hit = jnp.where(launched_upward_raw | failed_hit, False, pending_hit)
        pending_steps = jnp.where(launched_upward_raw | failed_hit, 0, pending_steps)

        reward, reward_terms = self._reward(
            data=data,
            action=action,
            da=da,
            arm_limiter_pen=arm_limiter_pen,
            bpos=bpos,
            bvel=bvel,
            rpos=rpos,
            rvel=rvel,
            rel=rel,
            rel_local=rel_local,
            racket_normal=racket_normal,
            predicted_apex_z=predicted_apex_z,
            hit_count=hit_count,
            new_hit=launched_upward,
            rewardable_hit=rewardable_hit,
            failed_hit=failed_hit,
            ignored_fast_hit=ignored_fast_hit,
            hit_cadence_reward=hit_cadence_reward,
            hit_min_interval_penalty=hit_min_interval_penalty,
            fast_hit_penalty=fast_hit_penalty,
            other_ball_contact=other_ball_contact,
            in_contact=in_contact,
            contact_hold_steps=contact_hold_steps,
            rel_speed=jnp.linalg.norm(bvel - rvel, axis=-1),
            cmd_qvel=cmd_qvel,
            prev_arm_qvel=state.prev_arm_qvel,
            racket_anchor=state.racket_anchor,
            chest_target_offset=state.chest_target_offset,
        )

        arm_qvel = data.qvel[:, self.arm_vadr]
        terminated, done_terms = self._termination_terms(data, bpos, rpos, state.racket_anchor)
        ball_miss = (
            done_terms["ball_too_low"]
            | done_terms["ball_too_high"]
            | done_terms["ball_x_out_of_bounds"]
            | done_terms["ball_y_out_of_bounds"]
        )
        racket_limit_done = done_terms["racket_too_high"] | done_terms["racket_too_low"]
        ball_miss_penalty = jnp.where(
            ball_miss & (hit_count > 0),
            -(
                float(self.cfg.termination_miss_penalty_base)
                + float(self.cfg.termination_miss_penalty_per_hit) * hit_count.astype(jnp.float32)
            ),
            0.0,
        )
        racket_limit_penalty = jnp.where(
            racket_limit_done,
            -(
                float(self.cfg.racket_z_limit_termination_penalty_base)
                + float(self.cfg.racket_z_limit_termination_penalty_per_hit) * hit_count.astype(jnp.float32)
            ),
            0.0,
        )
        reward = reward + ball_miss_penalty + racket_limit_penalty
        truncated = step_count >= self.max_steps
        done = terminated | truncated

        next_state = EnvState(
            model=state.model,
            data=data,
            rng=state.rng,
            step_count=step_count,
            racket_anchor=state.racket_anchor,
            chest_target_offset=state.chest_target_offset,
            arm_cmd_q=arm_cmd_q,
            arm_cmd_qvel=cmd_qvel,
            prev_action=action,
            prev_arm_qvel=arm_qvel,
            prev_ball_pos=bpos,
            prev_racket_pos=rpos,
            prev_contact=in_contact,
            hit_armed=hit_armed,
            no_contact_steps=no_contact_steps,
            contact_hold_steps=contact_hold_steps,
            pending_hit=pending_hit,
            pending_hit_steps=pending_steps,
            hit_count=hit_count,
            action_buffer=action_buffer,
            action_latency_steps=state.action_latency_steps,
            obs_buffer=state.obs_buffer,
            obs_latency_steps=state.obs_latency_steps,
            cached_ball_obs_pos=state.cached_ball_obs_pos,
            cached_ball_obs_vel=state.cached_ball_obs_vel,
            last_ball_obs_step=state.last_ball_obs_step,
            ball_obs_valid_pos=state.ball_obs_valid_pos,
            ball_obs_valid_vel=state.ball_obs_valid_vel,
            ball_obs_age_seconds=state.ball_obs_age_seconds,
            ball_obs_dropout_remaining=state.ball_obs_dropout_remaining,
            ball_obs_dropout_steps_total=state.ball_obs_dropout_steps_total,
            ball_obs_burst_count=state.ball_obs_burst_count,
            total_env_steps=state.total_env_steps + 1,
            action_scale_mult=state.action_scale_mult,
            dr_gravity_z=state.dr_gravity_z,
            dr_ball_mass=state.dr_ball_mass,
            dr_ball_friction=state.dr_ball_friction,
            dr_racket_friction=state.dr_racket_friction,
            dr_ball_solref_time=state.dr_ball_solref_time,
            dr_ball_solref_damping=state.dr_ball_solref_damping,
            dr_damping_mult=state.dr_damping_mult,
            dr_armature_mult=state.dr_armature_mult,
            last_hit_time=last_hit_time,
            last_counted_hit_time=last_counted_hit_time,
            last_count_gate_hit_time=last_count_gate_hit_time,
            confirmed_hit_count=confirmed_hit_count,
            ignored_fast_hit_count=ignored_fast_hit_count,
            rewarded_hit_count=rewarded_hit_count,
            unrewarded_extra_hit_count=unrewarded_extra_hit_count,
            dr_racket_pos_offset=state.dr_racket_pos_offset,
            dr_racket_rot_offset=state.dr_racket_rot_offset,
            dr_racket_radius_offset=state.dr_racket_radius_offset,
        )
        next_state, obs = self._apply_observation_pipeline(next_state, bpos, bvel)
        metrics = {
            "hit_count": hit_count.astype(jnp.float32),
            "new_hit": launched_upward.astype(jnp.float32),
            "confirmed_hit": launched_upward_raw.astype(jnp.float32),
            "rewardable_hit": rewardable_hit.astype(jnp.float32),
            "ignored_fast_hit": ignored_fast_hit.astype(jnp.float32),
            "other_ball_contact": other_ball_contact.astype(jnp.float32),
            "ball_z": bpos[:, 2],
            "racket_z": rpos[:, 2],
            "racket_z_rel": rpos[:, 2] - state.racket_anchor[:, 2],
            "in_contact": in_contact.astype(jnp.float32),
            "action_scale_mult": state.action_scale_mult,
            "dr_gravity_z": state.dr_gravity_z,
            "dr_ball_mass": state.dr_ball_mass,
            "dr_ball_friction": state.dr_ball_friction,
            "dr_racket_friction": state.dr_racket_friction,
            "dr_ball_solref_time": state.dr_ball_solref_time,
            "dr_ball_solref_damping": state.dr_ball_solref_damping,
            "dr_damping_mult": state.dr_damping_mult,
            "dr_armature_mult": state.dr_armature_mult,
            "dr_racket_pos_offset_norm": jnp.linalg.norm(state.dr_racket_pos_offset, axis=-1),
            "dr_racket_rot_offset_norm": jnp.linalg.norm(state.dr_racket_rot_offset, axis=-1),
            "dr_racket_radius_offset": state.dr_racket_radius_offset,
            "dr_obs_latency_steps": state.obs_latency_steps.astype(jnp.float32),
            "dr_action_latency_steps": state.action_latency_steps.astype(jnp.float32),
            "ball_obs_age": next_state.ball_obs_age_seconds,
            "ball_obs_dropout_active": (next_state.ball_obs_age_seconds > 0.0).astype(jnp.float32),
            "terminated": terminated.astype(jnp.float32),
            "truncated": truncated.astype(jnp.float32),
            "episode_step": step_count.astype(jnp.float32),
        }
        for name, value in reward_terms.items():
            if name.startswith("metric/"):
                metrics[name[len("metric/") :]] = value
            else:
                metrics[f"reward/{name}"] = value
        metrics["reward/ball_miss_termination_penalty"] = ball_miss_penalty
        metrics["reward/racket_z_limit_termination_penalty"] = racket_limit_penalty
        metrics["reward/total"] = reward
        metrics.update({f"done/{name}": value.astype(jnp.float32) for name, value in done_terms.items()})
        return next_state, obs, reward, done, metrics

    def reset_done(self, state: EnvState, obs: jax.Array, done: jax.Array, keys: jax.Array) -> tuple[EnvState, jax.Array]:
        old_total_env_steps = state.total_env_steps
        reset_state, reset_obs = self.reset(keys)

        def select(reset_leaf, leaf):
            if not hasattr(leaf, "shape") or leaf.shape[:1] != done.shape[:1]:
                return leaf
            mask_shape = (done.shape[0],) + (1,) * (leaf.ndim - 1)
            return jnp.where(done.reshape(mask_shape), reset_leaf, leaf)

        state = jax.tree_util.tree_map(select, reset_state, state)
        state = state._replace(total_env_steps=old_total_env_steps)
        obs = jnp.where(done[:, None], reset_obs, obs)
        return state, obs

    def _apply_observation_pipeline(
        self,
        state: EnvState,
        true_bpos: jax.Array,
        true_bvel: jax.Array,
    ) -> tuple[EnvState, jax.Array]:
        split_keys = jax.vmap(lambda k: jax.random.split(k, 6))(state.rng)
        next_rng = split_keys[:, 0]
        key_pos_noise = split_keys[:, 1]
        key_vel_noise = split_keys[:, 2]
        key_dropout = split_keys[:, 3]
        key_dropout_duration = split_keys[:, 4]
        key_burst_duration = split_keys[:, 5]

        total_steps_cfg = max(1, int(self.cfg.total_training_steps))
        warmup = max(0, int(round(total_steps_cfg * float(self.cfg.ball_obs_noise_warmup_ratio))))
        ramp = max(1, int(round(total_steps_cfg * float(self.cfg.ball_obs_noise_ramp_ratio))))
        noise_scale = jnp.clip((state.total_env_steps.astype(jnp.float32) - float(warmup)) / float(ramp), 0.0, 1.0)
        pos_std = float(self.cfg.ball_obs_pos_noise_std) * noise_scale
        vel_std = float(self.cfg.ball_obs_vel_noise_std) * noise_scale
        pos_noise = jax.vmap(lambda k: jax.random.normal(k, (3,), dtype=jnp.float32))(key_pos_noise) * pos_std[:, None]
        vel_noise = jax.vmap(lambda k: jax.random.normal(k, (3,), dtype=jnp.float32))(key_vel_noise) * vel_std[:, None]

        refresh = (state.step_count - state.last_ball_obs_step) >= int(self.ball_obs_every)
        sampled_pos = true_bpos + pos_noise
        sampled_vel = true_bvel + vel_noise
        cached_pos = jnp.where(refresh[:, None], sampled_pos, state.cached_ball_obs_pos)
        cached_vel = jnp.where(refresh[:, None], sampled_vel, state.cached_ball_obs_vel)
        last_ball_obs_step = jnp.where(refresh, state.step_count, state.last_ball_obs_step)

        still_dropout = state.ball_obs_dropout_remaining > 0
        u = jax.vmap(lambda k: jax.random.uniform(k, (), dtype=jnp.float32))(key_dropout)
        burst_start = (~still_dropout) & (u < float(self.cfg.ball_obs_dropout_burst_prob))
        single_start = (
            (~still_dropout)
            & (~burst_start)
            & (u < float(self.cfg.ball_obs_dropout_burst_prob) + float(self.cfg.ball_obs_dropout_prob))
        )
        single_duration = jax.vmap(
            lambda k: jax.random.randint(
                k,
                (),
                minval=1,
                maxval=max(2, int(self.cfg.ball_obs_dropout_max_steps) + 1),
                dtype=jnp.int32,
            )
        )(key_dropout_duration)
        burst_duration = jax.vmap(
            lambda k: jax.random.randint(
                k,
                (),
                minval=1,
                maxval=max(2, int(self.cfg.ball_obs_dropout_burst_max_steps) + 1),
                dtype=jnp.int32,
            )
        )(key_burst_duration)
        start_dropout = burst_start | single_start
        new_duration = jnp.where(burst_start, burst_duration, single_duration)
        dropout_remaining = jnp.where(
            still_dropout,
            jnp.maximum(0, state.ball_obs_dropout_remaining - 1),
            jnp.where(start_dropout, jnp.maximum(0, new_duration - 1), 0),
        )
        hold_obs = still_dropout | start_dropout
        valid_pos = jnp.where(hold_obs[:, None], state.ball_obs_valid_pos, cached_pos)
        valid_vel = jnp.where(hold_obs[:, None], state.ball_obs_valid_vel, cached_vel)
        age_seconds = jnp.where(hold_obs, state.ball_obs_age_seconds + self.dt, 0.0)
        dropout_steps_total = state.ball_obs_dropout_steps_total + hold_obs.astype(jnp.int32)
        burst_count = state.ball_obs_burst_count + burst_start.astype(jnp.int32)

        state = state._replace(
            rng=next_rng,
            cached_ball_obs_pos=cached_pos,
            cached_ball_obs_vel=cached_vel,
            last_ball_obs_step=last_ball_obs_step,
            ball_obs_valid_pos=valid_pos,
            ball_obs_valid_vel=valid_vel,
            ball_obs_age_seconds=age_seconds,
            ball_obs_dropout_remaining=dropout_remaining,
            ball_obs_dropout_steps_total=dropout_steps_total,
            ball_obs_burst_count=burst_count,
        )
        raw_obs = self._make_obs(state, valid_pos, valid_vel, age_seconds)
        obs_buffer = jnp.concatenate([state.obs_buffer[:, 1:, :], raw_obs[:, None, :]], axis=1)
        obs_idx = (self.max_obs_latency_steps - state.obs_latency_steps).astype(jnp.int32)
        obs = obs_buffer[jnp.arange(obs_buffer.shape[0]), obs_idx]
        state = state._replace(obs_buffer=obs_buffer)
        return state, obs

    def _ball_contact_flags(self, data) -> tuple[jax.Array, jax.Array]:
        geom = data.contact.geom
        dist = data.contact.dist
        if geom.ndim == 2:
            geom = geom[None, ...]
            dist = dist[None, ...]
        g0 = geom[..., 0]
        g1 = geom[..., 1]
        pair = ((g0 == self.ball_geom_id) & (g1 == self.racket_geom_id)) | (
            (g0 == self.racket_geom_id) & (g1 == self.ball_geom_id)
        )
        ball_pair = (g0 == self.ball_geom_id) | (g1 == self.ball_geom_id)
        racket_pair = ((g0 == self.racket_geom_id) | (g1 == self.racket_geom_id))
        proxy0 = jnp.any(g0[..., None] == self.non_racket_geom_ids, axis=-1)
        proxy1 = jnp.any(g1[..., None] == self.non_racket_geom_ids, axis=-1)
        non_racket_pair = ball_pair & (~racket_pair) & (proxy0 | proxy1)
        close = dist <= 0.002
        return jnp.any(pair & close, axis=-1), jnp.any(non_racket_pair & close, axis=-1)

    def _reward(
        self,
        data,
        action: jax.Array,
        da: jax.Array,
        arm_limiter_pen: jax.Array,
        bpos: jax.Array,
        bvel: jax.Array,
        rpos: jax.Array,
        rvel: jax.Array,
        rel: jax.Array,
        rel_local: jax.Array,
        racket_normal: jax.Array,
        predicted_apex_z: jax.Array,
        hit_count: jax.Array,
        new_hit: jax.Array,
        rewardable_hit: jax.Array,
        failed_hit: jax.Array,
        ignored_fast_hit: jax.Array,
        hit_cadence_reward: jax.Array,
        hit_min_interval_penalty: jax.Array,
        fast_hit_penalty: jax.Array,
        other_ball_contact: jax.Array,
        in_contact: jax.Array,
        contact_hold_steps: jax.Array,
        rel_speed: jax.Array,
        cmd_qvel: jax.Array,
        prev_arm_qvel: jax.Array,
        racket_anchor: jax.Array,
        chest_target_offset: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del cmd_qvel
        target_ball_z = racket_anchor[:, 2] + float(self.cfg.target_height)
        target_hit_apex_z = racket_anchor[:, 2] + float(self.cfg.hit_height_center)
        upward_vz = jnp.maximum(0.0, bvel[:, 2])
        dz_up = predicted_apex_z - target_ball_z
        dz_down = bpos[:, 2] - target_ball_z
        ball_height_reward = jnp.where(
            upward_vz > 0.0,
            jnp.exp(-7.0 * dz_up * dz_up),
            jnp.where(rel[:, 2] > 0.05, 0.25 * jnp.exp(-10.0 * dz_down * dz_down), 0.0),
        )
        xy_track_pen = jnp.sum(rel[:, :2] ** 2, axis=-1)
        racket_center_pen = jnp.sum((rpos - racket_anchor) ** 2, axis=-1)
        racket_xy_dist = jnp.linalg.norm((rpos - racket_anchor)[:, :2], axis=-1)
        racket_xy_gauss = jnp.exp(-0.5 * (racket_xy_dist / max(1e-6, float(self.cfg.racket_xy_gauss_sigma))) ** 2)
        racket_xy_gauss_pen = 1.0 - racket_xy_gauss
        waist_pos = data.xpos[:, self.waist_body_id] if self.waist_body_id >= 0 else racket_anchor
        chest_target = waist_pos + chest_target_offset
        racket_chest_xy_pen = jnp.sum((rpos[:, :2] - chest_target[:, :2]) ** 2, axis=-1)
        racket_chest_z_pen = (rpos[:, 2] - chest_target[:, 2]) ** 2
        ball_anchor_xy_pen = jnp.sum((bpos[:, :2] - chest_target[:, :2]) ** 2, axis=-1)
        rel_height_bonus = jnp.exp(
            -0.5 * ((rel[:, 2] - float(self.cfg.rel_height_center)) / max(1e-6, float(self.cfg.rel_height_sigma))) ** 2
        )

        posture_q = data.qpos[:, self.posture_qadr]
        posture_pen = jnp.mean((posture_q - self.posture_targets) ** 2, axis=-1)
        base_pose = jnp.stack(
            [
                data.qpos[:, self.base_x_qadr],
                data.qpos[:, self.base_y_qadr],
                data.qpos[:, self.base_yaw_qadr],
            ],
            axis=-1,
        )
        base_pose_err = base_pose - self.initial_base_pose
        base_pose_err = base_pose_err.at[:, 2].set((base_pose_err[:, 2] + jnp.pi) % (2.0 * jnp.pi) - jnp.pi)
        base_pose_pen = jnp.sum(base_pose_err**2, axis=-1)
        base_to_ball_world = bpos[:, :2] - base_pose[:, :2]
        base_yaw = base_pose[:, 2]
        c_yaw = jnp.cos(base_yaw)
        s_yaw = jnp.sin(base_yaw)
        ball_base_x = c_yaw * base_to_ball_world[:, 0] + s_yaw * base_to_ball_world[:, 1]
        ball_base_x_excess = jnp.maximum(0.0, jnp.abs(ball_base_x) - float(self.cfg.ball_base_x_soft_limit))
        ball_base_x_pen = ball_base_x_excess * ball_base_x_excess
        ball_base_vx = c_yaw * bvel[:, 0] + s_yaw * bvel[:, 1]
        ball_base_vy = -s_yaw * bvel[:, 0] + c_yaw * bvel[:, 1]
        ball_base_vxy_pen = ball_base_vx * ball_base_vx + ball_base_vy * ball_base_vy
        ball_vxy_pen = jnp.sum(bvel[:, :2] ** 2, axis=-1)
        post_hit_ball_xy_dist = jnp.linalg.norm(bpos[:, :2] - chest_target[:, :2], axis=-1)
        apex_soft_excess = jnp.maximum(0.0, predicted_apex_z - (target_ball_z + float(self.cfg.apex_soft_limit_margin)))
        apex_soft_pen = float(self.cfg.apex_soft_penalty_weight) * apex_soft_excess * apex_soft_excess
        ball_xy_soft_excess = jnp.maximum(0.0, post_hit_ball_xy_dist - float(self.cfg.ball_xy_soft_limit_radius))
        ball_xy_soft_pen = jnp.where(
            (hit_count > 0) | (upward_vz > 0.0) | (bpos[:, 2] > racket_anchor[:, 2]),
            float(self.cfg.ball_xy_soft_penalty_weight) * ball_xy_soft_excess * ball_xy_soft_excess,
            0.0,
        )
        post_hit_ball_xy_score = jnp.exp(
            -0.5 * (post_hit_ball_xy_dist / max(1e-6, float(self.cfg.post_hit_ball_xy_sigma))) ** 2
        )
        post_hit_survival_reward = jnp.where(
            (hit_count > 0) & (bpos[:, 2] >= racket_anchor[:, 2] - 0.02),
            post_hit_ball_xy_score - float(self.cfg.post_hit_ball_vxy_penalty_weight) * ball_vxy_pen,
            0.0,
        )
        drop_dist = bpos[:, 2] - rpos[:, 2]
        vz_abs = jnp.maximum(1e-5, -bvel[:, 2])
        time_to_racket = drop_dist / vz_abs
        projected_ball_xy = bpos[:, :2] + bvel[:, :2] * time_to_racket[:, None]
        descending_intercept_xy_err = jnp.linalg.norm(projected_ball_xy - rpos[:, :2], axis=-1)
        descending_intercept_reward = jnp.where(
            (hit_count > 0) & (bvel[:, 2] < -1e-4) & (bpos[:, 2] > rpos[:, 2]),
            jnp.exp(
                -0.5
                * (descending_intercept_xy_err / max(1e-6, float(self.cfg.descending_intercept_sigma))) ** 2
            ),
            0.0,
        )
        torque_pen = jnp.mean(data.actuator_force[:, self.arm_aids_j] ** 2, axis=-1)
        sep_dist = jnp.linalg.norm(rel, axis=-1)
        sticky_contact = (
            in_contact
            & (contact_hold_steps >= int(self.cfg.stick_min_contact_steps))
            & (sep_dist <= float(self.cfg.stick_rel_dist_thresh))
            & (rel_speed <= float(self.cfg.stick_rel_speed_thresh))
        )
        stick_hold_excess = 1.0 + jnp.maximum(0, contact_hold_steps - int(self.cfg.stick_min_contact_steps)).astype(jnp.float32)
        stick_pen = jnp.where(sticky_contact, stick_hold_excess * float(self.cfg.sticky_contact_penalty_growth), 0.0)
        non_racket_contact_pen = jnp.where(
            other_ball_contact,
            float(self.cfg.non_racket_ball_contact_penalty_weight),
            0.0,
        )

        racket_z_rel = rpos[:, 2] - racket_anchor[:, 2]
        z_excess_up = jnp.maximum(0.0, racket_z_rel - float(self.cfg.racket_z_band_up))
        z_excess_down = jnp.maximum(0.0, -float(self.cfg.racket_z_band_down) - racket_z_rel)
        racket_z_band_pen = z_excess_up * z_excess_up + z_excess_down * z_excess_down
        up_drift_pen = jnp.where(
            (racket_z_rel > 0.0) & (rvel[:, 2] > float(self.cfg.racket_up_drift_vel_thresh)),
            racket_z_rel * jnp.maximum(0.0, rvel[:, 2]),
            0.0,
        )
        camera_terms = self._camera_reward_terms(data, bpos)

        arm_qvel = data.qvel[:, self.arm_vadr]
        arm_vel_ratio = jnp.abs(arm_qvel) / jnp.maximum(self.arm_vel_limit_rad_s, 1e-6)
        arm_vel_exceed = jnp.maximum(arm_vel_ratio - 1.0, 0.0)
        arm_vel_limit_pen = jnp.mean(arm_vel_exceed**2, axis=-1)
        arm_qacc = (arm_qvel - prev_arm_qvel) / max(self.dt, 1e-6)
        arm_acc_ratio = jnp.abs(arm_qacc) / jnp.maximum(self.arm_acc_limit_rad_s2, 1e-6)
        arm_acc_exceed = jnp.maximum(arm_acc_ratio - 1.0, 0.0)
        arm_acc_limit_pen = jnp.mean(arm_acc_exceed**2, axis=-1)

        term_ball_height = 1.2 * ball_height_reward
        term_rel_height = float(self.cfg.rel_height_bonus_weight) * rel_height_bonus
        term_xy_track_penalty = -1.4 * xy_track_pen
        term_racket_center_penalty = -0.35 * racket_center_pen
        term_posture_penalty = -float(self.cfg.posture_weight) * posture_pen
        term_base_pose_penalty = -float(self.cfg.base_pose_weight) * base_pose_pen
        term_torque_penalty = -float(self.cfg.torque_penalty_weight) * torque_pen
        term_stick_penalty = -float(self.cfg.stick_contact_penalty_weight) * stick_pen
        term_non_racket_contact_penalty = -non_racket_contact_pen
        term_racket_chest_xy_penalty = -float(self.cfg.racket_chest_xy_penalty_weight) * racket_chest_xy_pen
        term_racket_chest_z_penalty = -float(self.cfg.racket_chest_z_penalty_weight) * racket_chest_z_pen
        term_ball_anchor_xy_penalty = -float(self.cfg.ball_anchor_xy_penalty_weight) * ball_anchor_xy_pen
        term_ball_base_x_penalty = -float(self.cfg.ball_base_x_penalty_weight) * ball_base_x_pen
        term_ball_base_vxy_penalty = -float(self.cfg.ball_base_vxy_penalty_weight) * ball_base_vxy_pen
        term_ball_vxy_penalty = -float(self.cfg.ball_vxy_penalty_weight) * ball_vxy_pen
        term_apex_soft_penalty = -apex_soft_pen
        term_ball_xy_soft_penalty = -ball_xy_soft_pen
        term_post_hit_survival = float(self.cfg.post_hit_survival_reward_weight) * post_hit_survival_reward
        term_descending_intercept = float(self.cfg.descending_intercept_reward_weight) * descending_intercept_reward
        term_racket_xy_reward = float(self.cfg.racket_xy_gauss_reward_weight) * racket_xy_gauss
        term_racket_xy_penalty = -float(self.cfg.racket_xy_gauss_penalty_weight) * racket_xy_gauss_pen
        term_racket_z_penalty = -float(self.cfg.racket_z_soft_penalty_weight) * racket_z_band_pen
        term_racket_up_drift_penalty = -float(self.cfg.racket_up_drift_penalty_weight) * up_drift_pen
        racket_up_cos = jnp.maximum(0.0, jnp.sum(racket_normal * jnp.asarray([0.0, 0.0, 1.0]), axis=-1))
        flatness_err = jnp.maximum(0.0, float(self.cfg.hit_flatness_target_cos) - racket_up_cos)
        flatness_score = jnp.exp(-0.5 * (flatness_err / max(1e-6, float(self.cfg.hit_flatness_sigma))) ** 2)
        flat_contact_pen = jnp.where(
            in_contact,
            float(self.cfg.contact_flatness_penalty_weight) * jnp.maximum(0.0, 1.0 - flatness_score),
            0.0,
        )
        term_contact_flatness_penalty = -flat_contact_pen
        term_action_penalty = -float(self.cfg.action_penalty_weight) * jnp.sum(action**2, axis=-1)
        term_action_delta_penalty = -float(self.cfg.action_delta_penalty_weight) * jnp.sum(da**2, axis=-1)
        term_arm_vel_penalty = -float(self.cfg.arm_vel_limit_penalty_weight) * arm_vel_limit_pen
        term_arm_acc_penalty = -float(self.cfg.arm_acc_limit_penalty_weight) * arm_acc_limit_pen
        term_arm_limiter_penalty = -float(self.cfg.arm_limiter_penalty_weight) * arm_limiter_pen

        dense_reward = (
            term_ball_height
            + term_rel_height
            + term_xy_track_penalty
            + term_racket_center_penalty
            + term_posture_penalty
            + term_base_pose_penalty
            + term_torque_penalty
            + term_stick_penalty
            + term_non_racket_contact_penalty
            + term_racket_chest_xy_penalty
            + term_racket_chest_z_penalty
            + term_ball_anchor_xy_penalty
            + term_ball_base_x_penalty
            + term_ball_base_vxy_penalty
            + term_ball_vxy_penalty
            + term_apex_soft_penalty
            + term_ball_xy_soft_penalty
            + term_post_hit_survival
            + term_descending_intercept
            + term_racket_xy_reward
            + term_racket_xy_penalty
            + term_racket_z_penalty
            + term_racket_up_drift_penalty
            + term_contact_flatness_penalty
            + camera_terms["camera_reward_dense"]
            + term_action_penalty
            + term_action_delta_penalty
            + term_arm_vel_penalty
            + term_arm_acc_penalty
            + term_arm_limiter_penalty
        )
        reward = dense_reward * self.dt

        contact_center_dist = jnp.linalg.norm(rel_local[:, :2], axis=-1)
        center_gain = jnp.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_sigma))) ** 2)
        local_center_gain = jnp.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_local_sigma))) ** 2)
        hit_bonus = float(self.cfg.hit_reward_base) + float(self.cfg.hit_reward_combo) * jnp.minimum(hit_count.astype(jnp.float32), 12.0)
        hit_bonus = hit_bonus * jnp.maximum(0.2, center_gain * flatness_score)
        hit_height_err = jnp.abs(predicted_apex_z - target_hit_apex_z)
        hit_height_excess = jnp.maximum(0.0, hit_height_err - float(self.cfg.hit_height_tolerance))
        hit_height_pen = float(self.cfg.hit_height_penalty_weight) * hit_height_excess * hit_height_excess
        low_hit_deficit = jnp.maximum(0.0, (target_ball_z - float(self.cfg.low_hit_apex_margin)) - predicted_apex_z)
        low_hit_pen = float(self.cfg.low_hit_penalty_weight) * low_hit_deficit * low_hit_deficit
        center_flat = float(self.cfg.center_flat_hit_reward_weight) * local_center_gain * flatness_score
        height_bonus = jnp.where(
            predicted_apex_z >= target_ball_z,
            0.35 * jnp.exp(-10.0 * (predicted_apex_z - target_ball_z) * (predicted_apex_z - target_ball_z)),
            0.0,
        )
        hit_reward_mask = new_hit & rewardable_hit
        term_hit_bonus = jnp.where(hit_reward_mask, hit_bonus, 0.0)
        term_center_flat_hit = jnp.where(hit_reward_mask, center_flat, 0.0)
        term_hit_height_bonus = jnp.where(hit_reward_mask, height_bonus, 0.0)
        term_hit_cadence_reward = jnp.where(hit_reward_mask, hit_cadence_reward, 0.0)
        term_hit_min_interval_penalty = jnp.where(hit_reward_mask, -hit_min_interval_penalty, 0.0)
        term_hit_height_penalty = jnp.where(hit_reward_mask, -hit_height_pen, 0.0)
        term_low_hit_penalty = jnp.where(hit_reward_mask, -low_hit_pen, 0.0)
        term_failed_hit_penalty = jnp.where(failed_hit, -float(self.cfg.failed_hit_penalty_weight), 0.0)
        term_fast_hit_penalty = jnp.where(ignored_fast_hit, -fast_hit_penalty, 0.0)
        reward = (
            reward
            + term_hit_bonus
            + term_center_flat_hit
            + term_hit_height_bonus
            + term_hit_cadence_reward
            + term_hit_min_interval_penalty
            + term_hit_height_penalty
            + term_low_hit_penalty
            + term_failed_hit_penalty
            + term_fast_hit_penalty
        )
        terms = {
            "total": reward,
            "dense_scaled": dense_reward * self.dt,
            "ball_height": term_ball_height * self.dt,
            "rel_height": term_rel_height * self.dt,
            "xy_track_penalty": term_xy_track_penalty * self.dt,
            "racket_center_penalty": term_racket_center_penalty * self.dt,
            "posture_penalty": term_posture_penalty * self.dt,
            "base_pose_penalty": term_base_pose_penalty * self.dt,
            "torque_penalty": term_torque_penalty * self.dt,
            "stick_penalty": term_stick_penalty * self.dt,
            "non_racket_contact_penalty": term_non_racket_contact_penalty * self.dt,
            "racket_chest_xy_penalty": term_racket_chest_xy_penalty * self.dt,
            "racket_chest_z_penalty": term_racket_chest_z_penalty * self.dt,
            "ball_anchor_xy_penalty": term_ball_anchor_xy_penalty * self.dt,
            "ball_base_x_penalty": term_ball_base_x_penalty * self.dt,
            "ball_base_vxy_penalty": term_ball_base_vxy_penalty * self.dt,
            "ball_vxy_penalty": term_ball_vxy_penalty * self.dt,
            "apex_soft_penalty": term_apex_soft_penalty * self.dt,
            "ball_xy_soft_penalty": term_ball_xy_soft_penalty * self.dt,
            "post_hit_survival": term_post_hit_survival * self.dt,
            "descending_intercept": term_descending_intercept * self.dt,
            "racket_xy_reward": term_racket_xy_reward * self.dt,
            "racket_xy_penalty": term_racket_xy_penalty * self.dt,
            "racket_z_penalty": term_racket_z_penalty * self.dt,
            "racket_up_drift_penalty": term_racket_up_drift_penalty * self.dt,
            "contact_flatness_penalty": term_contact_flatness_penalty * self.dt,
            "camera_reward_dense": camera_terms["camera_reward_dense"] * self.dt,
            "camera_pixel_center_penalty": camera_terms["camera_pixel_center_penalty"] * self.dt,
            "camera_visibility_penalty": camera_terms["camera_visibility_penalty"] * self.dt,
            "camera_depth_penalty": camera_terms["camera_depth_penalty"] * self.dt,
            "camera_box_penalty": camera_terms["camera_box_penalty"] * self.dt,
            "camera_visible_penalty": camera_terms["camera_visible_penalty"] * self.dt,
            "camera_top_margin_penalty": camera_terms["camera_top_margin_penalty"] * self.dt,
            "action_penalty": term_action_penalty * self.dt,
            "action_delta_penalty": term_action_delta_penalty * self.dt,
            "arm_vel_penalty": term_arm_vel_penalty * self.dt,
            "arm_acc_penalty": term_arm_acc_penalty * self.dt,
            "arm_limiter_penalty": term_arm_limiter_penalty * self.dt,
            "hit_bonus": term_hit_bonus,
            "center_flat_hit": term_center_flat_hit,
            "hit_height_bonus": term_hit_height_bonus,
            "hit_cadence_reward": term_hit_cadence_reward,
            "hit_min_interval_penalty": term_hit_min_interval_penalty,
            "hit_height_penalty": term_hit_height_penalty,
            "low_hit_penalty": term_low_hit_penalty,
            "failed_hit_penalty": term_failed_hit_penalty,
            "fast_hit_penalty": term_fast_hit_penalty,
        }
        terms.update({name: value for name, value in camera_terms.items() if name.startswith("metric/")})
        return reward, terms

    def _camera_reward_terms(self, data, bpos: jax.Array) -> dict[str, jax.Array]:
        n = bpos.shape[0]
        zeros = jnp.zeros((n,), dtype=jnp.float32)
        terms = {
            "camera_reward_dense": zeros,
            "camera_pixel_center_penalty": zeros,
            "camera_visibility_penalty": zeros,
            "camera_depth_penalty": zeros,
            "camera_box_penalty": zeros,
            "camera_visible_penalty": zeros,
            "camera_top_margin_penalty": zeros,
            "metric/camera_available": zeros,
            "metric/camera_in_front": zeros,
            "metric/camera_in_depth": zeros,
            "metric/camera_in_frustum": zeros,
            "metric/camera_in_image": zeros,
            "metric/camera_in_margin": zeros,
            "metric/camera_visible": zeros,
            "metric/camera_pixel_center_pen": zeros,
            "metric/camera_pixel_margin_pen": zeros,
            "metric/camera_top_margin_pen": zeros,
            "metric/camera_depth_pen": zeros,
            "metric/ball_cam_x": zeros,
            "metric/ball_cam_y": zeros,
            "metric/ball_cam_z": zeros,
            "metric/ball_pixel_u": zeros,
            "metric/ball_pixel_v": zeros,
        }
        if self.cfg.camera_visibility_mode == "off" or self.virtual_camera_body_id < 0:
            return terms

        body_pos = data.xpos[:, self.virtual_camera_body_id]
        body_R = data.xmat[:, self.virtual_camera_body_id].reshape((n, 3, 3))
        mount_offset = self.vc_mount_pos + self.vc_mount_R @ self.vc_optical_pos
        cam_pos = body_pos + jnp.einsum("nij,j->ni", body_R, mount_offset)
        cam_R = jnp.einsum("nij,jk->nik", body_R, self.vc_mount_R @ self.vc_mount_to_camera_R)
        p_cam = jnp.einsum("nij,nj->ni", jnp.swapaxes(cam_R, 1, 2), bpos - cam_pos)
        x = p_cam[:, 0]
        y = p_cam[:, 1]
        z = p_cam[:, 2]
        has_projection = z > 1e-6
        z_safe = jnp.where(has_projection, z, 1.0)

        width = float(self.cfg.camera_image_width)
        height = float(self.cfg.camera_image_height)
        fx = float(self.cfg.camera_fx)
        fy = float(self.cfg.camera_fy)
        cx = float(self.cfg.camera_cx)
        cy = float(self.cfg.camera_cy)
        margin = float(self.cfg.camera_pixel_margin)
        u = fx * (x / z_safe) + cx
        v = cy - fy * (y / z_safe)

        in_front = has_projection
        in_depth = has_projection & (z >= float(self.cfg.camera_min_depth)) & (z <= float(self.cfg.camera_max_depth))
        x_angle = jnp.arctan2(x, z_safe)
        y_angle = jnp.arctan2(y, z_safe)
        h_half = float(np.deg2rad(float(self.cfg.camera_hfov_deg) * 0.5))
        v_half = float(np.deg2rad(float(self.cfg.camera_vfov_deg) * 0.5))
        x_angle_excess = jnp.maximum(0.0, jnp.abs(x_angle) - h_half)
        y_angle_excess = jnp.maximum(0.0, jnp.abs(y_angle) - v_half)
        in_frustum = has_projection & (x_angle_excess <= 0.0) & (y_angle_excess <= 0.0)
        in_image = has_projection & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        in_margin = has_projection & (u >= margin) & (u <= (width - margin)) & (v >= margin) & (v <= (height - margin))
        visible = in_front & in_depth & in_frustum & in_image

        du = (u - cx) / max(0.5 * width, 1e-6)
        dv = (v - cy) / max(0.5 * height, 1e-6)
        center_pen = du * du + dv * dv
        depth_low = jnp.maximum(0.0, float(self.cfg.camera_min_depth) - z)
        depth_high = jnp.maximum(0.0, z - float(self.cfg.camera_max_depth))
        depth_pen = (depth_low / max(float(self.cfg.camera_min_depth), 1e-6)) ** 2 + (
            depth_high / max(float(self.cfg.camera_max_depth), 1e-6)
        ) ** 2
        u_low = jnp.maximum(0.0, margin - u)
        u_high = jnp.maximum(0.0, u - (width - margin))
        v_low = jnp.maximum(0.0, margin - v)
        v_high = jnp.maximum(0.0, v - (height - margin))
        margin_pen = (
            (u_low / max(width, 1e-6)) ** 2
            + (u_high / max(width, 1e-6)) ** 2
            + (v_low / max(height, 1e-6)) ** 2
            + (v_high / max(height, 1e-6)) ** 2
        )
        frustum_pen = (x_angle_excess / max(h_half, 1e-6)) ** 2 + (y_angle_excess / max(v_half, 1e-6)) ** 2
        x_excess = jnp.maximum(0.0, jnp.abs(x) - float(self.cfg.camera_box_half_width))
        y_excess = jnp.maximum(0.0, jnp.abs(y) - float(self.cfg.camera_box_half_height))
        z_low_excess = jnp.maximum(0.0, float(self.cfg.camera_box_depth_min) - z)
        z_high_excess = jnp.maximum(0.0, z - float(self.cfg.camera_box_depth_max))
        box_pen = x_excess * x_excess + y_excess * y_excess + z_low_excess * z_low_excess + z_high_excess * z_high_excess
        top_margin_pen = (jnp.maximum(0.0, margin - v) / max(height, 1e-6)) ** 2

        # Match the CPU env: if the ball is behind/on the optical plane,
        # geometric camera penalties are zero and only the optional visible
        # fixed penalty can apply.
        center_pen = jnp.where(has_projection, center_pen, 0.0)
        depth_pen = jnp.where(has_projection, depth_pen, 0.0)
        margin_pen = jnp.where(has_projection, margin_pen, 0.0)
        frustum_pen = jnp.where(has_projection, frustum_pen, 0.0)
        box_pen = jnp.where(has_projection, box_pen, 0.0)
        top_margin_pen = jnp.where(has_projection, top_margin_pen, 0.0)
        terms.update(
            {
                "metric/camera_available": jnp.ones((n,), dtype=jnp.float32),
                "metric/camera_in_front": in_front.astype(jnp.float32),
                "metric/camera_in_depth": in_depth.astype(jnp.float32),
                "metric/camera_in_frustum": in_frustum.astype(jnp.float32),
                "metric/camera_in_image": in_image.astype(jnp.float32),
                "metric/camera_in_margin": in_margin.astype(jnp.float32),
                "metric/camera_visible": visible.astype(jnp.float32),
                "metric/camera_pixel_center_pen": center_pen,
                "metric/camera_pixel_margin_pen": margin_pen,
                "metric/camera_top_margin_pen": top_margin_pen,
                "metric/camera_depth_pen": depth_pen,
                "metric/ball_cam_x": x,
                "metric/ball_cam_y": y,
                "metric/ball_cam_z": z,
                "metric/ball_pixel_u": jnp.where(has_projection, u, 0.0),
                "metric/ball_pixel_v": jnp.where(has_projection, v, 0.0),
            }
        )

        if self.cfg.camera_visibility_mode == "box":
            box_term = -float(self.cfg.camera_box_penalty_weight) * box_pen
            dense, (box_term,) = self._clip_camera_dense_terms((box_term,))
            terms.update({"camera_reward_dense": dense, "camera_box_penalty": box_term})
        elif self.cfg.camera_visibility_mode == "frustum":
            vis_term = -float(self.cfg.camera_visibility_penalty_weight) * frustum_pen
            depth_term = -float(self.cfg.camera_depth_penalty_weight) * depth_pen
            dense, (vis_term, depth_term) = self._clip_camera_dense_terms((vis_term, depth_term))
            terms.update(
                {
                    "camera_reward_dense": dense,
                    "camera_visibility_penalty": vis_term,
                    "camera_depth_penalty": depth_term,
                }
            )
        elif self.cfg.camera_visibility_mode == "pixel":
            center_term = -float(self.cfg.camera_center_weight) * center_pen
            vis_term = -float(self.cfg.camera_visibility_penalty_weight) * margin_pen
            depth_term = -float(self.cfg.camera_depth_penalty_weight) * depth_pen
            top_term = -float(self.cfg.camera_top_margin_penalty_weight) * top_margin_pen
            visible_term = jnp.where(
                (float(self.cfg.camera_visible_penalty_weight) > 0.0) & (~visible),
                -float(self.cfg.camera_visible_penalty_weight),
                0.0,
            )
            dense, (center_term, vis_term, depth_term, top_term, visible_term) = self._clip_camera_dense_terms(
                (center_term, vis_term, depth_term, top_term, visible_term)
            )
            terms.update(
                {
                    "camera_reward_dense": dense,
                    "camera_pixel_center_penalty": center_term,
                    "camera_visibility_penalty": vis_term,
                    "camera_depth_penalty": depth_term,
                    "camera_visible_penalty": visible_term,
                    "camera_top_margin_penalty": top_term,
                }
            )
        return terms

    def _clip_camera_dense_terms(self, terms: tuple[jax.Array, ...]) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        dense = sum(terms)
        clip = float(self.cfg.camera_dense_penalty_clip)
        if clip <= 0.0:
            return dense, terms
        dense_clipped = jnp.maximum(dense, -clip)
        scale = jnp.where(dense < -clip, dense_clipped / jnp.minimum(dense, -1e-6), 1.0)
        return dense_clipped, tuple(term * scale for term in terms)

    def _termination_terms(
        self,
        data,
        bpos: jax.Array,
        rpos: jax.Array,
        racket_anchor: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        racket_z_rel = rpos[:, 2] - racket_anchor[:, 2]
        racket_too_high = racket_z_rel > float(self.cfg.racket_z_hard_limit_up)
        racket_too_low = racket_z_rel < -float(self.cfg.racket_z_hard_limit_down)
        if not bool(self.cfg.terminate_on_racket_z_limit):
            racket_too_high = jnp.zeros_like(racket_too_high, dtype=bool)
            racket_too_low = jnp.zeros_like(racket_too_low, dtype=bool)
        terms = {
            "ball_too_low": bpos[:, 2] < 0.8,
            "ball_too_high": bpos[:, 2] > 1.9,
            "ball_x_out_of_bounds": jnp.abs(bpos[:, 0] - racket_anchor[:, 0]) > 0.5,
            "ball_y_out_of_bounds": jnp.abs(bpos[:, 1] - racket_anchor[:, 1]) > 0.5,
            "base_x_out_of_bounds": jnp.abs(data.qpos[:, self.base_x_qadr]) > 2.6,
            "base_y_out_of_bounds": jnp.abs(data.qpos[:, self.base_y_qadr]) > 2.6,
            "racket_too_far_from_anchor": jnp.linalg.norm(rpos - racket_anchor, axis=-1) > 1.1,
            "racket_too_high": racket_too_high,
            "racket_too_low": racket_too_low,
        }
        terminated = jnp.zeros_like(terms["ball_too_low"], dtype=bool)
        for value in terms.values():
            terminated = terminated | value
        return terminated, terms

    def _point_to_base(self, point: jax.Array, base_q: jax.Array) -> jax.Array:
        dx = point[:, 0] - base_q[:, 0]
        dy = point[:, 1] - base_q[:, 1]
        yaw = base_q[:, 2]
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        return jnp.stack([c * dx + s * dy, -s * dx + c * dy, point[:, 2]], axis=-1)

    def _vel_to_base(self, vel: jax.Array, point: jax.Array, base_q: jax.Array, base_dq: jax.Array) -> jax.Array:
        rel_x = point[:, 0] - base_q[:, 0]
        rel_y = point[:, 1] - base_q[:, 1]
        yaw_rate = base_dq[:, 2]
        rel_vx = vel[:, 0] - base_dq[:, 0] + yaw_rate * rel_y
        rel_vy = vel[:, 1] - base_dq[:, 1] - yaw_rate * rel_x
        yaw = base_q[:, 2]
        c = jnp.cos(yaw)
        s = jnp.sin(yaw)
        return jnp.stack([c * rel_vx + s * rel_vy, -s * rel_vx + c * rel_vy, vel[:, 2]], axis=-1)
