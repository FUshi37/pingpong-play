"""Independent 72/117/156-D FPLC adapters; publish q_ref to the robot's RMP.

MuJoCo is used only for FK/Jacobians and the frozen 0.7 s reference posture.
No simulated RMP, PD loop, delay queue or actuator compensation executes
on the hardware command path. An external intervention discards this policy
state and requires an explicit reentry reset; it never rewrites q_ref silently.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
try:
    import mujoco as mj
except ImportError:
    mj = None

from pingpong_controller.fplc_deployment import (
    FPLC_CONTRACTS, FPLC_REAL_ROLL_PITCH_LIMIT_DEG, fplc_assets,
)
from pingpong_controller.fplc_numpy import (
    ContactTargetConfig, advance_position_history, compute_dynamic_contact_target,
    make_history_actor_observation, make_predicted_contact_context,
    make_simple_position_observation, make_v224_history_context, matrix_to_euler_xyz_numpy,
    project_cartesian_pose_qvel_numpy, propagate_ballistic_state_to_age,
    relative_xyz_euler_rate_jacobian_numpy, rotation_matrix_to_6d,
    warm_fplc_projection, V224_PREFIX_HISTORY_LENGTH,
)
from pingpong_controller.mjx_rl_policy import MjxRLPolicyController
from pingpong_controller.policy_release_manifest import sha256_file
from pingpong_controller.policy_ball_handover import is_fractional_refresh
from pingpong_controller.simcode2_xml import build_simcode2_training_xml


def _finite(value, size, name):
    output = np.asarray(value, dtype=np.float32).reshape(-1)
    if output.shape != (size,) or not np.all(np.isfinite(output)):
        raise RuntimeError(f"FPLC {name} must be a finite length-{size} vector")
    return output


class FPLCPolicyController(MjxRLPolicyController):
    LOST_BALL_TIMEOUT_S = 0.35

    def __init__(self, model_path=None, *, observation_profile, robot_xml_path=None, **kwargs):
        if mj is None:
            raise RuntimeError("FPLC requires MuJoCo for FK and Jacobians")
        if observation_profile not in FPLC_CONTRACTS:
            raise RuntimeError(f"unknown FPLC profile {observation_profile!r}")
        self.contract = FPLC_CONTRACTS[observation_profile]
        self.projection_runtime = warm_fplc_projection(require_compiled=True)
        self.policy_model_id = observation_profile
        self.fplc_profile = observation_profile
        default_model, default_xml = fplc_assets(observation_profile)
        model_path = str(model_path or default_model)
        robot_xml_path = str(robot_xml_path or default_xml)
        # Authenticate bytes before unpickling or preprocessing an XML.
        if sha256_file(Path(model_path)) != self.contract['sha256']:
            raise RuntimeError("FPLC checkpoint SHA-256 does not match selected release")
        if sha256_file(Path(robot_xml_path)) != self.contract['policy_xml_sha256']:
            raise RuntimeError("FPLC XML SHA-256 does not match selected release")
        self._checkpoint_sha256 = self.contract['sha256']
        if not np.isclose(float(kwargs.get('dt', .005)), .005, rtol=0, atol=1e-12):
            raise RuntimeError("FPLC requires exactly 200 Hz")
        prepared_xml = build_simcode2_training_xml(
            robot_xml_path, right_arm_pd_profile='recovered_rmp_rmpmd_v2',
            simulation_base_mode='aligned_fixed')
        kwargs.update(expected_obs_dim=self.contract['actor_dim'],
                      enable_delay_conditioning=False, use_valid_ball_obs_age=True)
        super().__init__(model_path, observation_profile=observation_profile,
                         robot_xml_path=str(prepared_xml), **kwargs)
        if self.mj_model is None or self.mj_data is None:
            raise RuntimeError("FPLC requires calibrated right_ee_site FK")
        self.initial_arm_q_rad = np.deg2rad(np.asarray(
            self._checkpoint_env_cfg['right_arm_reset_degrees'], dtype=np.float32))
        self._warm_fk_model_to_training_posture()
        self.reference_rotation = np.asarray(
            self.mj_data.site_xmat[self._racket_site_id], dtype=np.float32).reshape(3, 3).copy()
        self.episode_target_xy = np.asarray(
            self._checkpoint_env_cfg['fixed_contact_workspace_center_xy_m'],
            dtype=np.float32)
        self.warm_arm_q = np.asarray([
            self.mj_data.qpos[int(self.mj_model.jnt_qposadr[j])]
            for j in self._arm_jids], dtype=np.float32)
        cfg = self._checkpoint_env_cfg
        self.position_lower = np.asarray(cfg['cartesian_pose_position_lower_m'], dtype=np.float32)
        self.position_upper = np.asarray(cfg['cartesian_pose_position_upper_m'], dtype=np.float32)
        self.rp_outer = np.deg2rad(np.array([cfg['cartesian_pose_roll_limit_deg'],
                                          cfg['cartesian_pose_pitch_limit_deg']], dtype=np.float32))
        self.rp_inner = self.rp_outer - np.deg2rad(np.float32(cfg['cartesian_pose_orientation_margin_deg']))
        self.real_rp_outer = np.deg2rad(np.asarray(FPLC_REAL_ROLL_PITCH_LIMIT_DEG, dtype=np.float32))
        # right_arm_reset_degrees seeds the reference warm-up, not the actual
        # racket-launch episode reset. The frozen environment then solves a
        # pose-preserving XY/surface-Z follow IK before installing q_ref.
        self.warm_seed_arm_q = self.initial_arm_q_rad.copy()
        self.native_reset_arm_q = self._make_native_reset_pose()
        self.initial_arm_q_rad = self.native_reset_arm_q.copy()
        self.reset()
        self._logger.info(f"FPLC projection runtime: {self.projection_runtime}")
        self._logger.info(
            "FPLC hardware path: 24-sweep Cartesian projection -> persistent q_ref -> robot RMP; "
            "RightArmCommandSafetyLimiter disabled, compensation/delay/PD/RMP software loops absent. "
            "MuJoCo is FK/reference-warm only; checkpoint arm_action_limiter is a training flag.")

    def _make_native_reset_pose(self):
        """Port mjx_juggle_env.solve_follow_xy_pose for the frozen zero-offset lane."""
        cfg = self._checkpoint_env_cfg
        expected = {
            'ball_reset_mode': 'racket_launch',
            'racket_launch_racket_follow_episode_target_xy': True,
            'racket_launch_racket_follow_fixed_contact_surface_z': True,
            'falling_reset_racket_follow_xy_canonical_ik': True,
            'falling_reset_racket_follow_episode_target_xy_blend': 1.0,
            'falling_reset_racket_follow_episode_target_xy_blend_range': None,
            'falling_reset_racket_follow_episode_target_xy_blend_choices': None,
            'fixed_contact_cartesian_reset_blend_enabled': False,
        }
        if any(cfg.get(key) != value for key, value in expected.items()):
            raise RuntimeError('FPLC native reset requires the frozen full-follow racket-launch lane')
        for key in ('episode_target_x_range_m', 'episode_target_y_range_m',
                    'episode_racket_anchor_z_range_m'):
            if tuple(cfg[key]) != (0.0, 0.0):
                # V319 was trained across target offsets. Hardware reset
                # deliberately selects its authenticated nominal lane; never
                # sample training DR or move the episode target implicitly.
                offsets = self.contract['native_reset_contract'].get('deployment_episode_offsets_m')
                if offsets != [0.0, 0.0, 0.0] or not cfg[key][0] <= 0.0 <= cfg[key][1]:
                    raise RuntimeError('FPLC native reset requires the frozen zero-offset lane')
        desired = np.array([*cfg['fixed_contact_workspace_center_xy_m'],
                            cfg['fixed_contact_surface_z_m']], dtype=np.float64)
        qadr = [int(self.mj_model.jnt_qposadr[j]) for j in self._arm_jids]
        vadr = [int(self.mj_model.jnt_dofadr[j]) for j in self._arm_jids]
        warm_q = self.warm_arm_q.astype(np.float64)
        q = warm_q.copy()
        reference = self.mj_data.site_xmat[self._racket_site_id].reshape(3, 3).copy()
        jp, jr = np.zeros((3, self.mj_model.nv)), np.zeros((3, self.mj_model.nv))
        weight = float(cfg['falling_reset_racket_follow_ik_orientation_weight'])
        for _ in range(240):
            self.mj_data.qpos[qadr] = q
            self.mj_data.qvel[vadr] = 0.0
            mj.mj_forward(self.mj_model, self.mj_data)
            position_error = desired - self.mj_data.site_xpos[self._racket_site_id]
            rotation = self.mj_data.site_xmat[self._racket_site_id].reshape(3, 3)
            orientation_error = 0.5 * sum(
                np.cross(rotation[:, axis], reference[:, axis]) for axis in range(3))
            if np.linalg.norm(position_error) < 2.5e-5 and np.linalg.norm(orientation_error) < 5.0e-4:
                break
            mj.mj_jacSite(self.mj_model, self.mj_data, jp, jr, self._racket_site_id)
            jacobian = np.concatenate([jp[:, vadr], weight * jr[:, vadr]], axis=0)
            error = np.concatenate([position_error, weight * orientation_error])
            delta = np.linalg.solve(
                jacobian.T @ jacobian + (2.0e-5 + 2.0e-4) * np.eye(7),
                jacobian.T @ error - 2.0e-4 * (q - warm_q))
            norm = float(np.linalg.norm(delta))
            if norm > 0.10:
                delta *= 0.10 / norm
            q = np.clip(q + delta, self.arm_q_low, self.arm_q_high)
        q = q.astype(np.float32)
        position = self._pose(q, with_jacobian=False)[0]
        error_m = float(np.linalg.norm(position - desired))
        if error_m > float(cfg['falling_reset_racket_follow_max_error_m']) or not self.pose_is_safe(q):
            raise RuntimeError('FPLC native reset IK failed the frozen position/orientation contract')
        self.native_reset_target_m = desired.astype(np.float32)
        self.native_reset_position_m = position.copy()
        self.native_reset_error_m = error_m
        self._logger.info(f'FPLC native racket-launch reset: site={position.tolist()} m, '
                          f'position_error={error_m:.6f} m; reference orientation unchanged')
        return q

    def _validate_checkpoint_contract(self, payload):
        if (int(payload.get('obs_dim', 0)) != self.contract['actor_dim']
                or int(payload.get('critic_obs_dim', 0)) != self.contract['critic_dim']
                or int(payload.get('act_dim', 0)) != 7):
            raise RuntimeError("FPLC actor/critic/action schema mismatch")
        cfg = payload.get('env_cfg', {})
        if self.contract['actor_dim'] not in (72, 117, 156):
            raise RuntimeError("FPLC actor schema is not implemented")
        if self.contract['actor_dim'] == 117:
            if (not cfg.get('v224_history_enabled')
                    or cfg.get('v224_actor_schema') != 'v224_v149_prefix72_observed_local_history45'
                    or cfg.get('v224_history_masked') is not False
                    or not cfg.get('fixed_plane_v149_simple_obs_enabled')):
                raise RuntimeError("FPLC V224 requires its independent 72+45-D history schema")
        for key, expected in self.contract['env_contract'].items():
            actual = cfg.get(key)
            # JSON arrays and checkpoint tuples have identical semantics.
            if isinstance(expected, list):
                equal = actual is not None and np.array_equal(np.asarray(actual), np.asarray(expected))
            else:
                equal = actual == expected
            if not equal:
                raise RuntimeError(f"FPLC checkpoint configuration mismatch: {key}")
        if vars(cfg['fixed_contact_target']) != self.contract['contact_target_config']:
            raise RuntimeError("FPLC contact target configuration mismatch")
        if self._pi_weights['l1_w'].shape != (self.contract['actor_dim'], 256):
            raise RuntimeError("FPLC requires its frozen 256/256 actor")

    def reset(self):
        super().reset()
        self._fplc_requires_measured_reset = True
        self.ball_history = None
        self.racket_history = None
        self.age_history = None
        self.prefix_history = None
        self._held_direct_position = None
        self._held_direct_stamp = None
        self._held_direct_age = 0.0
        self._direct_observed = False
        self._dropout_reference_hold_active = False
        self._fplc_tick = 0
        self._ball_state_rate_hz = None
        self._held_state_position = None
        self._held_state_age = 0.
        self._held_state_measurement_age = 0.
        self.ball_state_refresh_count = 0
        self.last_ball_state_refresh = False
        self.last_ball_state_predicted = False
        self.last_projection = {}

    def invalidate_after_intervention(self):
        """Fence the next inference; never install a safety output as q_ref."""
        self.reset()

    def _update_kinematics(self, q, *, with_jacobian=False):
        # FK does not need contacts, constraints, accelerations or the RMP.
        # comPos supplies the motion axes/COM data required by mj_jacSite;
        # kinematics alone supplies the site/geom/body poses. Never step time.
        for jid, value in zip(self._arm_jids, q):
            self.mj_data.qpos[int(self.mj_model.jnt_qposadr[jid])] = value
        mj.mj_kinematics(self.mj_model, self.mj_data)
        if with_jacobian:
            mj.mj_comPos(self.mj_model, self.mj_data)

    def _compute_racket_pose_base(self, arm_q):
        self._update_kinematics(arm_q)
        position = self.mj_data.site_xpos[self._racket_site_id].astype(np.float32)
        return self._world_point_to_base(position), True

    def _compute_racket_pose_quat_base(self, arm_q):
        self._update_kinematics(arm_q)
        position = self.mj_data.site_xpos[self._racket_site_id].astype(np.float32)
        rotation = self.mj_data.site_xmat[self._racket_site_id].astype(np.float32).reshape(3, 3)
        return (self._world_point_to_base(position),
                self._rotmat_to_quat_xyzw(self._world_rot_to_base_yaw(rotation)), True)

    def _pose(self, q, *, with_jacobian=True):
        q = _finite(q, 7, 'FK q')
        self._update_kinematics(q, with_jacobian=with_jacobian)
        position_world = self.mj_data.site_xpos[self._racket_site_id].astype(np.float32).copy()
        rotation = self.mj_data.site_xmat[self._racket_site_id].reshape(3, 3).astype(np.float32).copy()
        position_policy = self._world_point_to_base(position_world)
        rpy = matrix_to_euler_xyz_numpy(self.reference_rotation.T @ rotation)
        if not with_jacobian:
            return position_world, rotation, position_policy, rpy, None, None
        jp, jr = np.zeros((3, self.mj_model.nv)), np.zeros((3, self.mj_model.nv))
        mj.mj_jacSite(self.mj_model, self.mj_data, jp, jr, self._racket_site_id)
        columns = [int(self.mj_model.jnt_dofadr[j]) for j in self._arm_jids]
        yaw = float(self.mj_data.qpos[self._base_yaw_qadr])
        c, s = np.cos(yaw), np.sin(yaw)
        transform = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]], dtype=np.float32)
        rpy_jac = relative_xyz_euler_rate_jacobian_numpy(
            rpy, self.reference_rotation, jr[:, columns].astype(np.float32))
        return position_world, rotation, position_policy, rpy, transform @ jp[:, columns].astype(np.float32), rpy_jac

    def pose_is_safe(self, q, limits=None):
        """Check joints/attitude and either the training or explicit real box."""
        try:
            q = _finite(q, 7, 'safety q')
            if np.any(q < self.arm_q_low) or np.any(q > self.arm_q_high):
                return False
            _, _, p, rpy, _, _ = self._pose(q, with_jacobian=False)
            lower, upper, rp_limit = self.position_lower, self.position_upper, self.rp_outer
            if limits is not None:
                # The real stop envelope may extend beyond training Z.  The
                # actor projection below still uses its authenticated box.
                real_limits = np.asarray(limits, dtype=np.float64)
                if (real_limits.shape != (3, 2)
                        or not np.all(np.isfinite(real_limits))
                        or np.any(real_limits[:, 0] >= real_limits[:, 1])):
                    return False
                lower, upper = real_limits[:, 0], real_limits[:, 1]
                rp_limit = self.real_rp_outer
            return bool(np.all(p >= lower) and np.all(p <= upper)
                        and np.all(np.abs(rpy[:2]) <= rp_limit)
                        and np.all(np.isfinite(rpy)))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return False

    def _project_reference(self, measured_q, nominal_dq):
        cfg = self._checkpoint_env_cfg
        _, _, position, rpy, jacp, jacr = self._pose(self.arm_cmd_q)
        horizon = np.asarray(cfg['recovered_rmp_qvel_reference_error_horizon_s_per_joint'], dtype=np.float32)
        error = self.arm_vel_limit_rad_s * horizon
        lower_q = np.maximum(self.arm_q_low, measured_q - error)
        upper_q = np.minimum(self.arm_q_high, measured_q + error)
        lower = np.maximum(-self.arm_vel_limit_rad_s, (lower_q - self.arm_cmd_q) / np.float32(self.dt))
        upper = np.minimum(self.arm_vel_limit_rad_s, (upper_q - self.arm_cmd_q) / np.float32(self.dt))
        margin = np.float32(cfg['cartesian_pose_position_margin_m'])
        projection = project_cartesian_pose_qvel_numpy(
            nominal_qvel=nominal_dq[None], position_m=position[None],
            relative_rpy_rad=rpy[None], translation_jacobian=jacp[None],
            relative_rpy_jacobian=jacr[None], position_lower_m=self.position_lower + margin,
            position_upper_m=self.position_upper - margin, roll_pitch_limit_rad=self.rp_inner,
            velocity_lower=lower[None], velocity_upper=upper[None], velocity_scale=self.arm_vel_limit_rad_s,
            position_gain_s=cfg['cartesian_pose_position_cbf_gain_s'],
            orientation_gain_s=cfg['cartesian_pose_orientation_cbf_gain_s'],
            projection_sweeps=cfg['cartesian_pose_projection_sweeps'])
        candidate = self.arm_cmd_q + projection.qvel[0] * np.float32(self.dt)
        _, _, candidate_position, candidate_rpy, _, _ = self._pose(candidate, with_jacobian=False)
        tolerance = cfg['cartesian_pose_projection_tolerance']
        def safe(p, angles):
            return bool(max(np.max(np.maximum(self.position_lower - p, p - self.position_upper)),
                            np.max(np.abs(angles[:2]) - self.rp_outer), 0.0) <= tolerance)
        current_safe, candidate_safe = safe(position, rpy), safe(candidate_position, candidate_rpy)
        accepted = current_safe and candidate_safe and float(projection.max_violation[0]) <= tolerance
        q_next = candidate if accepted else self.arm_cmd_q.copy()
        dq = (q_next - self.arm_cmd_q) / np.float32(self.dt)
        self.last_projection = {
            'accepted': bool(accepted), 'currentSafe': current_safe,
            'candidateSafe': candidate_safe, 'fallbackToHold': not accepted,
            'nonlinearBackstop': current_safe and not candidate_safe,
            'maxLinearViolation': float(projection.max_violation[0]),
            'correctionNormRadS': float(np.linalg.norm(dq - nominal_dq)),
        }
        return q_next.astype(np.float32), dq.astype(np.float32)

    def predict(self, ball_pos_m, ball_vel_m_s, *, ball_valid, ball_obs_age_s=0.0,
                dt=None, arm_q=None, arm_dq=None, arm_sample_stamp_s=None,
                observation_scales=None, confirmed_hit_count=0, ball_observed=None,
                direct_ball_position_m=None, direct_ball_age_s=None,
                direct_ball_stamp_s=None, ball_state_already_propagated=False,
                ball_observation_rate_hz=90., ball_history_position_m=None,
                ball_history_age_s=None, ball_history_measurement_age_s=None,
                episode_target_xy_m=None, hold_reference_on_dropout=False,
                enable_dropout_reference_hold=True):
        del arm_sample_stamp_s  # History advances at 200 Hz, including repeated encoder samples.
        state_rate = float(ball_observation_rate_hz)
        if state_rate not in (90., 120., 200.):
            raise RuntimeError('FPLC ball state rate must be 90, 120 or 200 Hz')
        if (not self._fplc_requires_measured_reset
                and self._ball_state_rate_hz != state_rate):
            raise RuntimeError('FPLC ball state rate change requires an explicit policy reset')
        step_dt = self.dt if dt is None else float(dt)
        age = float(ball_obs_age_s)
        if not np.isclose(step_dt, .005, rtol=0, atol=1e-12):
            raise RuntimeError('FPLC requires an exact 5 ms policy tick')
        if not ball_valid or not np.isfinite(age) or age < 0 or age > self.LOST_BALL_TIMEOUT_S:
            self.invalidate_after_intervention()
            raise RuntimeError('FPLC ball state invalid or older than 350 ms')
        if arm_q is None or arm_dq is None:
            raise RuntimeError('FPLC requires one fresh atomic measured q/dq sample')
        q, dq = _finite(arm_q, 7, 'measured q'), _finite(arm_dq, 7, 'measured dq')
        if np.any(q < self.arm_q_low) or np.any(q > self.arm_q_high):
            raise RuntimeError('FPLC measured q outside joint limits')
        ball, velocity = _finite(ball_pos_m, 3, 'ball position'), _finite(ball_vel_m_s, 3, 'ball velocity')
        if observation_scales is not None:
            scales = np.asarray(observation_scales)
            if scales.size and (scales.shape != (self._expected_obs_dim,) or not np.all(scales == 1)):
                raise RuntimeError('FPLC forbids actor observation scaling')
        observed = bool(ball_valid) if ball_observed is None else bool(ball_observed)
        direct = ball if direct_ball_position_m is None else _finite(direct_ball_position_m, 3, 'direct ball position')
        direct_age = age if direct_ball_age_s is None else float(direct_ball_age_s)
        if not np.isfinite(direct_age) or direct_age < 0:
            raise RuntimeError('FPLC direct sample age is invalid')
        history_position = None
        history_age = None
        history_measurement_age = None
        if ball_history_position_m is not None:
            history_position = _finite(ball_history_position_m, 3, 'predicted history position')
            history_age = float(ball_history_age_s)
            history_measurement_age = float(ball_history_measurement_age_s if ball_history_measurement_age_s is not None else history_age)
            if (not np.isfinite([history_age, history_measurement_age]).all()
                    or not 0. <= history_age <= self.LOST_BALL_TIMEOUT_S
                    or not 0. <= history_measurement_age <= self.LOST_BALL_TIMEOUT_S):
                raise RuntimeError('FPLC predicted history measurement age is invalid or older than 350 ms')
        use_state_history = bool(state_rate > 90. or history_position is not None)
        self.last_ball_state_refresh = False
        position, rotation, _, _, _, _ = self._pose(q, with_jacobian=False)
        pose = np.concatenate((position, rotation_matrix_to_6d(rotation[None])[0])).astype(np.float32)
        if self._fplc_requires_measured_reset:
            if not observed:
                raise RuntimeError('FPLC reset requires a direct ball sample')
            self.arm_cmd_q, self.arm_cmd_qvel, self.prev_action = q.copy(), np.zeros(7, np.float32), np.zeros(7, np.float32)
            self.ball_history = np.broadcast_to(direct, (1, 8, 3)).copy()
            self.racket_history = np.broadcast_to(pose, (1, 8, 9)).copy()
            self.age_history = np.ones((1, 8), dtype=np.float32)
            self.age_history[0, 0] = direct_age
            self._held_direct_position, self._held_direct_age = direct.copy(), direct_age
            self._held_direct_stamp = direct_ball_stamp_s
            self._direct_observed = True
            self._ball_state_rate_hz = state_rate
            self._held_state_position = direct.copy()
            self._held_state_age = direct_age
            self._held_state_measurement_age = direct_age
            self.ball_state_refresh_count = 1
            self.last_ball_state_refresh = True
            self.last_ball_state_predicted = False
            self._fplc_requires_measured_reset = False
            self._fplc_tick = 0
        else:
            # Raw-source 90 Hz retains its original contract. A camera model
            # can explicitly supply a bounded current-time history at any
            # selected rate, keeping real range age separate from state time.
            tick = self._fplc_tick
            refresh = is_fractional_refresh(tick, state_rate, self.dt)
            self._held_direct_age += self.dt
            if use_state_history:
                self._held_state_age += self.dt
                self._held_state_measurement_age += self.dt
            if refresh:
                self._direct_observed = observed
                if observed:
                    self._held_direct_position = direct.copy()
                    self._held_direct_age = direct_age
                    self._held_direct_stamp = direct_ball_stamp_s
                if use_state_history and (history_position is not None or observed):
                    self._held_state_position = (history_position if history_position is not None else direct).copy()
                    self._held_state_age = history_age if history_position is not None else direct_age
                    self._held_state_measurement_age = history_measurement_age if history_position is not None else direct_age
                    self.last_ball_state_predicted = history_position is not None
                self.ball_state_refresh_count += 1
                self.last_ball_state_refresh = True
            self.ball_history, self.racket_history, self.age_history = advance_position_history(
                ball_positions=self.ball_history, racket_poses=self.racket_history, ball_ages=self.age_history,
                ball_position_now=(self._held_state_position if use_state_history else self._held_direct_position)[None], racket_pose_now=pose[None],
                ball_age_now=np.asarray([self._held_state_age if use_state_history else self._held_direct_age], dtype=np.float32))
        self._fplc_tick += 1
        target_xy = self.episode_target_xy if episode_target_xy_m is None else _finite(episode_target_xy_m, 2, 'episode target')
        common = dict(q=q[None], dq=dq[None], q_ref_error=(self.arm_cmd_q-q)[None],
                      previous_action=self.prev_action[None], ball_positions=self.ball_history,
                      ball_ages=self.age_history, target_xy_m=target_xy[None])
        if self._expected_obs_dim in (72, 117):
            observation = make_simple_position_observation(**common, racket_pose_now=pose[None],
                                                          ball_position_valid=np.asarray([self._direct_observed]))[0]
            if self._expected_obs_dim == 117:
                # Advance exactly once per policy tick, including held camera
                # samples and dropout holds. Reset pads the cache, while the
                # tick count masks all unavailable 50/100/200 ms lags.
                if self.prefix_history is None:
                    self.prefix_history = np.broadcast_to(
                        observation, (1, V224_PREFIX_HISTORY_LENGTH, 72)).copy()
                else:
                    self.prefix_history = np.concatenate((
                        observation[None, None], self.prefix_history[:, :-1]), axis=1)
                context = make_v224_history_context(
                    self.prefix_history, step_count=self._fplc_tick - 1)
                observation = np.concatenate((observation, context[0]))
        elif self._expected_obs_dim == 156:
            if int(confirmed_hit_count) < 0 or int(confirmed_hit_count) != confirmed_hit_count:
                raise RuntimeError('FPLC confirmed hit count must be nonnegative and integral')
            cfg = self._checkpoint_env_cfg
            schedule = cfg['fixed_plane_limit_cycle_apex_schedule_z_m']
            apex = np.float32(schedule[min(int(confirmed_hit_count), len(schedule)-1)])
            history = make_history_actor_observation(**common, racket_poses=self.racket_history,
                desired_apex_z_m=np.asarray([apex]), hit_count=np.asarray([confirmed_hit_count]),
                ball_valid=np.asarray([self._direct_observed and ball_valid]))
            predicted_ball, predicted_velocity = ball[None], velocity[None]
            if cfg['fixed_contact_propagate_held_ball_state_to_query_time'] and not ball_state_already_propagated:
                predicted_ball, predicted_velocity = propagate_ballistic_state_to_age(
                    predicted_ball, predicted_velocity, np.asarray([age]))
            target = compute_dynamic_contact_target(predicted_ball, predicted_velocity, self.reference_rotation,
                workspace_center_xy=tuple(cfg['fixed_contact_workspace_center_xy_m']),
                contact_surface_z_m=cfg['fixed_contact_surface_z_m'], ball_radius_m=.02,
                cfg=ContactTargetConfig(**self.contract['contact_target_config']), target_apex_z_m=np.asarray([apex]))
            # The training actor's surface feature is separate from real site FK.
            # V98 shifts only the simulated collision geom by +3 mm and uses
            # geom_size[0] along the site normal. Never modify real FK geometry.
            # The measured-q site/geom poses above remain current: history and
            # ballistic/contact calculations do not mutate mj_data.
            normal = rotation[:, 2] * (1 if rotation[2, 2] >= 0 else -1)
            surface = self.mj_data.geom_xpos[self._racket_geom_id].astype(np.float32) + np.array([0, 0, .003], np.float32) @ rotation.T + normal * np.float32(self.mj_model.geom_size[self._racket_geom_id, 0])
            reachable = all(low <= target.position_m[0, axis] <= high for axis, (low, high) in enumerate(
                (cfg['fixed_contact_racket_x_bounds_m'], cfg['fixed_contact_racket_y_bounds_m'], cfg['fixed_contact_racket_z_bounds_m'])))
            context = make_predicted_contact_context(predicted_ball_position_m=target.predicted_ball_position_at_contact_m,
                time_to_contact_s=target.time_to_contact_s, contact_valid=target.valid & reachable & bool(ball_valid),
                racket_target_position_m=target.position_m, racket_target_rotation=target.rotation_matrix,
                racket_surface_position_m=surface[None], next_crossing_xy_m=target.next_crossing_target_xy_m)
            observation = np.concatenate((history, context), axis=-1)[0]
        observation = _finite(observation, self._expected_obs_dim, 'observation')
        self._dropout_reference_hold_active = bool(
            enable_dropout_reference_hold and (
                hold_reference_on_dropout
                or (self._dropout_reference_hold_active and not self._direct_observed)))
        if self._dropout_reference_hold_active:
            # The clean actors were not trained on low-ball detector loss.
            # Update truthful history/age, but do not infer a new QVEL from an
            # out-of-distribution observed flag. Keep the existing q_ref until
            # a trusted direct observation reaches the selected state sampler.
            # This is an explicit zero-QVEL hold, not a rewritten reference.
            if not self.pose_is_safe(self.arm_cmd_q):
                raise RuntimeError('FPLC dropout reference outside training pose bounds')
            action = np.zeros(7, np.float32)
            q_next = self.arm_cmd_q.copy()
            projected_dq = np.zeros(7, np.float32)
            self.last_projection = {
                'accepted': True, 'currentSafe': True, 'candidateSafe': True,
                'fallbackToHold': False, 'nonlinearBackstop': False,
                'maxLinearViolation': 0.0, 'correctionNormRadS': 0.0,
                'ballDropoutHold': True, 'actorExecuted': False,
            }
        else:
            action = _finite(self._policy_action(observation), 7, 'actor output')
            nominal = np.clip(action, -1, 1) * self.arm_vel_limit_rad_s
            q_next, projected_dq = self._project_reference(q, nominal)
        previous_dq = self.arm_cmd_qvel.copy()
        self.arm_cmd_q, self.arm_cmd_qvel = q_next.copy(), projected_dq.copy()
        self.prev_action = np.clip(projected_dq / self.arm_vel_limit_rad_s, -1, 1).astype(np.float32)
        self.last_obs, self.last_obs_unscaled = observation.copy(), observation.copy()
        self.last_action, self.last_executed_action = action.copy(), self.prev_action.copy()
        self.last_cmd_q, self.last_cmd_qvel = q_next.copy(), projected_dq.copy()
        self.last_cmd_qacc = (projected_dq - previous_dq) / np.float32(self.dt)
        return q_next.copy()
