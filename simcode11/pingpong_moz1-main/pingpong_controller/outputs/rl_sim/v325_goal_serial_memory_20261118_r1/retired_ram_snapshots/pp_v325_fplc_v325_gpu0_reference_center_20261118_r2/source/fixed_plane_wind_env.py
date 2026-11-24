"""Private, source-authenticated wind hooks for the two frozen native stacks."""
from __future__ import annotations

from collections import namedtuple
import inspect
from functools import lru_cache
from pathlib import Path
import textwrap

import jax.numpy as jnp

from fixed_plane_v323_ball_obs_noise import digest_file
from fixed_plane_v325_wind import (
    WIND_FIELDS, advance_wind, critic_features, incremental_force,
    initialize_wind, nominal_wind_velocity, startup_gain, wind_velocity,
)


def wrap_substep(env, state, native_substep):
    """Apply the force only on the ball, before each native1ms solver step."""
    cfg = env.cfg.wind_config
    if not cfg.enabled:
        return native_substep
    tick_time_s = state.step_count * env.dt
    nominal_velocity = nominal_wind_velocity(state, tick_time_s, cfg)
    sheltered = cfg.startup_no_wind_hits > 0 or cfg.startup_ramp_s > 0.
    substep_dt_s = float(env.model.opt.timestep) if sheltered else 0.

    def substep(index, carry):
        data = carry[0]
        velocity = (nominal_velocity * startup_gain(
            state, tick_time_s + index*substep_dt_s, cfg)[:, None]
            if sheltered else nominal_velocity)
        original = data.xfrc_applied
        force, _ = incremental_force(
            data.qvel[:, env.ball_vadr:env.ball_vadr+3], velocity, cfg)
        forced = data.replace(xfrc_applied=original.at[:, env.ball_body_id, :3].add(force))
        result = native_substep(index, (forced, *carry[1:]))
        # Preserve an independently supplied external wrench and never retain
        # this tick's wind as a force to accumulate at the next tick.
        return (result[0].replace(xfrc_applied=original), *result[1:])

    return substep


def transport_wind(env, before, after):
    import jax.numpy as jnp
    cfg = env.cfg.wind_config
    time_s = after.step_count * env.dt
    advanced = advance_wind(before, before.rng, after.step_count * env.dt,
                            env.dt, cfg)
    hits = before.wind_startup_hit_count + jnp.maximum(after.hit_count-before.hit_count, 0)
    if cfg.startup_no_wind_hits:
        began = (before.wind_startup_started_s < 0.) & (hits >= cfg.startup_no_wind_hits)
        advanced = advanced._replace(
            wind_startup_started_s=jnp.where(began, time_s, before.wind_startup_started_s),
            wind_gust_start_s=jnp.where(began,
                before.wind_gust_start_s + time_s-before.wind_sampled_s,
                advanced.wind_gust_start_s))
    advanced = advanced._replace(wind_startup_hit_count=hits)
    extra = tuple(getattr(before, name) for name in ('wind_pose_ramp_steps',
        'wind_pose_ramp_enabled') if name in before._fields)
    return type(before)(*after, *(getattr(advanced, name) for name in WIND_FIELDS), *extra)


@lru_cache(maxsize=4)
def _state_type(version, native_type):
    extra = ('wind_pose_ramp_steps','wind_pose_ramp_enabled') if version in (325,326) else ()
    return namedtuple(f'V{version}WindState', (*native_type._fields, *WIND_FIELDS, *extra))


def make_wind_base(base: type, spec) -> type:
    """Retain all native state fields, including M13's six reset-cohort fields."""
    if digest_file(Path(inspect.getfile(base))) != spec.origin.native_env_sha256:
        raise ValueError('Wind adapter requires its own authenticated frozen environment')
    source = textwrap.dedent(inspect.getsource(base._step_impl))
    replacements = (
        ('    (\n        data,\n        in_contact,',
         '    one_substep = _wrap_wind_substep(self, state, one_substep)\n'
         '    (\n        data,\n        in_contact,'),
        ('    obs_state = next_state._replace(prev_racket_pos=state.prev_racket_pos)\n',
         '    next_state = _transport_wind(self, state, next_state)\n'
         '    obs_state = next_state._replace(prev_racket_pos=state.prev_racket_pos)\n'),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Frozen wind substep/state transport anchor changed')
        source = source.replace(old, new)
    pose_source = None
    if spec.version in (325, 326):
        from fixed_plane_wind_pose import build_pose_hooks
        pose_projection, pose_source, pose_reward = build_pose_hooks(base, source)
    namespace = {**base._step_impl.__globals__,
                 '_wrap_wind_substep': wrap_substep, '_transport_wind': transport_wind}
    exec(compile(source, base._step_impl.__code__.co_filename, 'exec'), namespace)
    legacy_step = namespace['_step_impl']
    if pose_source is not None:
        exec(compile(pose_source, '<wind_pose_native_step>', 'exec'), namespace)
        pose_step = namespace['_step_impl']

    class WindEnv(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.critic_obs_dim != 393 or self.obs_dim != 72 or self.dt != .005:
                raise ValueError('Wind requires the unchanged72/393/7 native200Hz stack')
            if float(self.model.opt.density) != 0. or float(self.model.opt.viscosity) != 0.:
                raise ValueError('Do not stack native fluid forces with wind increments')
            self.critic_extra_dim += 18
            self.critic_obs_dim += 18

        def _step_impl(self, state, action):
            if hasattr(self.cfg, 'wind_pose'):
                return pose_step(self, state, action)
            return legacy_step(self, state, action)

        def _project_cartesian_pose_reference(self, state, measured_q, nominal_qvel):
            if hasattr(self.cfg, 'wind_pose'):
                return pose_projection(self, state, measured_q, nominal_qvel)
            return super()._project_cartesian_pose_reference(state, measured_q, nominal_qvel)

        def _reward(self, *args, wind_pose_state=None, **kwargs):
            if wind_pose_state is not None:
                if self.cfg.wind_pose.strong_wind_reward:
                    from fixed_plane_wind_strong import outgoing_reference
                    target, tilt, peak = outgoing_reference(self, wind_pose_state)
                    physical_velocity = kwargs['bvel']
                    kwargs['bvel'] = physical_velocity.at[:, :2].add(-target)
                    reward, terms = pose_reward(self, wind_pose_state, tilt, *args, **kwargs)
                    hit = terms['metric/hit_event_count']
                    physical_xy = jnp.linalg.norm(physical_velocity[:, :2],axis=-1)
                    terms['metric/hit_vxy_sum'] = hit * physical_xy
                    terms['metric/hit_vxy_sq_sum'] = hit * physical_xy**2
                    terms['metric/v34_outgoing_vxy_m_s'] = hit * physical_xy
                    terms['metric/wind/required_outgoing_vxy_m_s'] = jnp.linalg.norm(target,axis=-1)
                    terms['metric/wind/required_tilt_deg'] = tilt
                    terms['metric/wind/reference_raw_force_peak_n'] = peak
                    terms['metric/wind/compensated_hit_vxy_sum'] = hit*jnp.linalg.norm(kwargs['bvel'][:,:2],axis=-1)
                    return reward, terms
                return pose_reward(self, wind_pose_state, jnp.zeros_like(wind_pose_state.step_count,dtype=jnp.float32), *args, **kwargs)
            return super()._reward(*args, **kwargs)

        def wind_pose_limits(self, state):
            from fixed_plane_wind_pose import effective_degrees
            degrees = effective_degrees(state, self.dt, self.cfg.wind_pose, xp=jnp)
            outer = jnp.repeat(jnp.deg2rad(degrees)[:, None], 2, axis=1)
            inner = jnp.repeat(jnp.deg2rad(degrees-float(
                self.cfg.cartesian_pose_orientation_margin_deg))[:, None], 2, axis=1)
            return inner, outer

        def wind_pose_flatness_cost(self, state, up_cos, reference_tilt_deg=None):
            from fixed_plane_wind_pose import flatness_cost
            return flatness_cost(self, state, up_cos,reference_tilt_deg)

        def wind_pose_deadband(self, state, old_rad,reference_tilt_deg=None):
            from fixed_plane_wind_pose import (flatness_deadband_rad, pose_clock_steps,
                                               schedule_fraction, shelter_gain)
            cfg = self.cfg.wind_pose
            if not cfg.strong_wind_reward:
                return flatness_deadband_rad(state,self.dt,old_rad,xp=jnp,schedule=cfg)
            from fixed_plane_wind_strong import outgoing_reference
            tilt=reference_tilt_deg
            if tilt is None:
                _,tilt,_=outgoing_reference(self,state)
            fraction = schedule_fraction(pose_clock_steps(state,cfg),cfg,xp=jnp)
            allowance = fraction*(tilt+.5*jnp.clip(tilt/.15,0.,1.))
            return old_rad+jnp.deg2rad(allowance)*shelter_gain(state,self.dt,xp=jnp)

        def wind_pose_quality_error(self,state,reference_tilt_deg,pose_error,up_cos,*,uses_normal_tilt):
            from fixed_contact_plane_training import contact_quality_pose_error
            original=contact_quality_pose_error(pose_error,up_cos,uses_normal_tilt=uses_normal_tilt)
            if not self.cfg.wind_pose.strong_wind_reward or not uses_normal_tilt:
                return original
            return jnp.maximum(original-self.wind_pose_deadband(state,0.,reference_tilt_deg),0.)

        def wind_pose_hit_cos(self, state,reference_tilt_deg=None):
            import math
            from fixed_plane_wind_pose import flatness_deadband_rad, shelter_gain
            gain = shelter_gain(state, self.dt, xp=jnp)
            new = jnp.cos(self.wind_pose_deadband(state,
                         math.acos(float(self.cfg.hit_flatness_target_cos)),reference_tilt_deg))
            return jnp.where(gain > 0., new, float(self.cfg.hit_flatness_target_cos))

        def reset(self, keys):
            state, obs = super().reset(keys)
            state_type = _state_type(spec.version, type(state))
            extra = ((jnp.zeros(self.n_envs,jnp.int32),jnp.zeros(self.n_envs,bool))
                     if spec.version in (325,326) else ())
            extended = state_type(*state, *initialize_wind(keys, self.cfg.wind_config), *extra)
            return extended, obs

        def reset_done(self, state, obs, done, keys):
            after, next_obs = super().reset_done(state,obs,done,keys)
            if spec.version in (325,326):
                after=after._replace(wind_pose_ramp_steps=state.wind_pose_ramp_steps,
                                     wind_pose_ramp_enabled=state.wind_pose_ramp_enabled)
            return after,next_obs

        def get_critic_obs(self, state, obs):
            prefix = super().get_critic_obs(state, obs)
            return jnp.concatenate((prefix, critic_features(
                state, state.step_count * self.dt, self.cfg.wind_config)), axis=-1)

        def step(self, state, action):
            after, obs, reward, done, metrics = super().step(state, action)
            if spec.version in (325,326):
                after = after._replace(
                    wind_pose_ramp_steps=state.wind_pose_ramp_steps+state.wind_pose_ramp_enabled.astype(jnp.int32),
                    wind_pose_ramp_enabled=state.wind_pose_ramp_enabled)
            velocity = wind_velocity(state, state.step_count * self.dt, self.cfg.wind_config)
            force, raw = incremental_force(state.data.qvel[
                :, self.ball_vadr:self.ball_vadr+3], velocity, self.cfg.wind_config)
            metrics.update({
                'wind/mode': state.wind_mode.astype(jnp.float32),
                'wind/speed_m_s': jnp.linalg.norm(velocity, axis=-1),
                'wind/entry_force_n': jnp.linalg.norm(force, axis=-1),
                'wind/entry_raw_force_n': jnp.linalg.norm(raw,axis=-1),
                'wind/entry_force_z_n': force[:, 2],
                'wind/entry_force_clipped': (jnp.linalg.norm(raw, axis=-1)
                    > self.cfg.wind_config.force_limit_n).astype(jnp.float32),
                'wind/startup_gain': startup_gain(state, state.step_count*self.dt,
                                                  self.cfg.wind_config),
                'wind/startup_confirmed_hits': state.wind_startup_hit_count.astype(jnp.float32),
            })
            if getattr(self.cfg.wind_config,'wind_y_nonpositive',False):
                metrics.update({'wind/velocity_x_m_s':velocity[:,0],
                    'wind/velocity_y_m_s':velocity[:,1],
                    'wind/positive_y_violation':(velocity[:,1]>1e-8).astype(jnp.float32),
                    'wind/y_nonpositive_domain':jnp.ones_like(velocity[:,1])})
            if hasattr(self.cfg, 'wind_pose'):
                from fixed_plane_wind_pose import effective_degrees, scheduled_degrees, pose_clock_steps
                metrics.update({
                    'pose/outer_limit_deg': effective_degrees(state, self.dt, self.cfg.wind_pose, xp=jnp),
                    'pose/scheduled_limit_deg': scheduled_degrees(pose_clock_steps(state,self.cfg.wind_pose), self.cfg.wind_pose, xp=jnp),
                    'pose/flatness_deadband_deg': jnp.rad2deg(self.wind_pose_deadband(state,
                        self.cfg.racket_flatness_angle_deadband_rad,metrics.get('wind/required_tilt_deg'))),
                    'pose/credited_updates': (state.wind_pose_ramp_steps+self.cfg.wind_pose.elapsed_steps_offset)/128.,
                    'pose/growth_enabled': state.wind_pose_ramp_enabled.astype(jnp.float32),
                })
            return after, obs, reward, done, metrics
    return WindEnv
