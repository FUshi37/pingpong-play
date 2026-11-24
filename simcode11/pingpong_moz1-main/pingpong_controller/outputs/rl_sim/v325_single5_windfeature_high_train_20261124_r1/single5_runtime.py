"""Private V325 single-MLP, constant five-degree training revision 1."""
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from observed_force import observed_force_features
from precision_maturity import restore_precision_context
from randomized_onset import RandomOnsetEnv,acquisition_time
from targeted_reward import targeted_terms,FIELDS as TARGETED_FIELDS
from productive_reward import safe_return_site_z, productive_reward

A = Path(__file__).resolve().parent
MIGRATION = A.parent / 'v325_single_actor_migration_20261120_r1'
sys.path.insert(0, str(A.parent / 'v325_single_actor_ablation_20261120_r1'))
from single_actor import ActorOptim
from feature_optimizer import feature_adam_factory as adam_factory
MODES = ('direct',)
REVISION = 'v325_single5_r27_maturerehearsal'


def constant_degrees(state, dt_s, cfg, *, xp):
    """Physical RP bound is five degrees before and after every contact."""
    return xp.ones_like(state.step_count, dtype=xp.float32)*5.


def configure(cfg):
    result=replace(cfg, cartesian_pose_roll_limit_deg=5., cartesian_pose_pitch_limit_deg=5.,
        wind_pose=replace(cfg.wind_pose, start_deg=5., target_deg=5., ramp_updates=0,
                          quality_gated=False, origin_steps=0, elapsed_steps_offset=0))
    return restore_precision_context(result,enabled=json.loads((A/'TRAIN_PLAN.json').read_text())['reward_maturity']['enabled'])


def install_pose_mode():
    import fixed_plane_wind_pose
    fixed_plane_wind_pose.effective_degrees = constant_degrees
    # Guidance previously compared world tilt with a relative-Euler bound.
    # Patch only this opt-in process, keeping the authenticated source intact.
    import fixed_plane_wind_dynamics_guidance as guidance
    if not getattr(guidance.apply_guidance,'_single5_relative_reference',False):
        import inspect
        import textwrap
        source=textwrap.dedent(inspect.getsource(guidance.apply_guidance))
        old="normal_feasible = contact['tilt_deg'] <= available_tilt"
        assert source.count(old)==1
        source=source.replace(old,"normal_feasible = relative_normal_feasible(contact['normal'], env.fixed_contact_racket_reference_rotation, env.wind_pose_limits(after)[0], xp=jnp)")
        height_line = 'desired_site_z = .5 * (bounds[0] + bounds[1])'
        assert source.count(height_line) == 1
        if json.loads((A/'TRAIN_PLAN.json').read_text())['reward_revision']['safe_return_reference']:
            source=source.replace(height_line, "desired_site_z = safe_return_site_z(before.wind_mode, before.wind_background_m_s, before.racket_anchor[:,2], gain, bounds, xp=jnp)")
        namespace={**guidance.apply_guidance.__globals__,'relative_normal_feasible':relative_normal_feasible,
                   'safe_return_site_z':safe_return_site_z}
        exec(compile(source,'<v325_single5_relative_normal_r1>','exec'),namespace)
        guidance.apply_guidance=namespace['apply_guidance']
        guidance.apply_guidance._single5_relative_reference=True


def relative_normal_feasible(normal_world, reference_rotation, limits_rad, *, xp):
    relative=xp.einsum('ji,...j->...i',reference_rotation,normal_world)
    pitch=xp.arcsin(xp.clip(relative[...,0],-1.,1.))
    roll=xp.arctan2(-relative[...,1],relative[...,2])
    return xp.all(xp.abs(xp.stack([roll,pitch],axis=-1))<=limits_rad,axis=-1)


def acquisition_features(state, *, xp):
    """Two causal time bases are ordinary inputs; neither scales actions."""
    elapsed = state.step_count*.005 - acquisition_time(state)
    acquired = (state.wind_startup_hit_count >= 5) & (acquisition_time(state) >= 0.)
    phase = xp.clip(elapsed/3., 0., 1.)
    progress = xp.where(acquired, xp.sin(.5*math.pi*phase)**2, 0.)
    return xp.stack([progress, xp.clip(72.*progress, 0., 1.)], axis=-1)


def migrate_actor(old):
    """Fold two old authority inputs into one ordinary acquisition time basis."""
    if old['l1']['w'].shape != (115, 282):
        raise ValueError('Expected the selected complete-S15 migrated single MLP')
    new = jax.tree.map(lambda x: np.asarray(x).copy(), old)
    weights = np.asarray(old['l1']['w'], np.float64)
    summed = weights[112]+weights[113]
    new['l1']['w'] = np.concatenate([weights[:112], (.9*summed)[None], weights[114:115]], axis=0).astype(np.float32)
    new['l1']['b'] = (np.asarray(old['l1']['b'], np.float64)+.1*summed).astype(np.float32)
    return augment_actor(new)


def augment_actor(pi):
    """Append four zero rows; preserve every old actor parameter."""
    new = jax.tree.map(lambda x: np.asarray(x).copy(), pi)
    if new["l1"]["w"].shape == (114,282):
        new["l1"]["w"] = np.pad(new["l1"]["w"],((0,4),(0,0)))
    elif new["l1"]["w"].shape != (118,282):
        raise ValueError("Expected the pinned projected 114-D single actor")
    return new


def migrate_critic(old):
    if old['l1']['w'].shape != (453, 256):
        raise ValueError('Expected the original privileged wind critic layout')
    new = jax.tree.map(lambda x: np.asarray(x).copy(), old)
    weights = np.asarray(old['l1']['w'], np.float64)
    summed = weights[451]+weights[452]
    new['l1']['w'][451] = (.9*summed).astype(np.float32)
    new['l1']['w'][452] = 0.
    new['l1']['b'] = (np.asarray(old['l1']['b'], np.float64)+.1*summed).astype(np.float32)
    return new


def assert_single_actor(params):
    if set(params) != {'pi', 'v', 'log_std'} or set(params['pi']) != {'l1','l2','l3','out'}:
        raise ValueError('One ordinary MLP only; no teacher, residual or authority keys')
    for name, shape in [('l1',(118,282)),('l2',(282,282)),('l3',(282,282)),('out',(282,7))]:
        if params['pi'][name]['w'].shape != shape or params['pi'][name]['b'].shape != (shape[1],):
            raise ValueError('Invalid single actor layer '+name)


def actor_parameter_count(pi):
    return sum(int(x.size) for x in jax.tree.leaves(pi))


def apply_single_mlp(pi, obs):
    x = obs
    for name in ('l1','l2','l3'):
        x = jnp.tanh(jnp.matmul(x,pi[name]['w'],precision=jax.lax.Precision.HIGHEST)+pi[name]['b'])
    return jnp.matmul(x,pi['out']['w'],precision=jax.lax.Precision.HIGHEST)+pi['out']['b']


def apply_factory(native_apply, *, enabled=False):
    if not enabled:
        return native_apply
    def apply(pi, obs):
        if pi['l1']['w'].shape == (118,282):
            return apply_single_mlp(pi,obs)
        return native_apply(pi,obs)
    return apply


def initialize(ppo, parent, *, mode, model_path=None):
    """Changed observation/reward contract starts fresh Adam, retaining weights."""
    if mode != 'direct':
        raise ValueError('Residual training is not available in this revision')
    import pickle
    from checkpoint_numpy import NumpyCheckpoint
    plan = json.loads((A/'TRAIN_PLAN.json').read_text())
    with Path(plan['single_parent']).open('rb') as stream:
        saved = NumpyCheckpoint(stream).load()
    assert set(saved['params']) == {'pi','v','log_std'} and saved['actual_updates'] == 64
    params = dict(pi=migrate_actor(saved['params']['pi']), v=migrate_critic(saved['params']['v']),
                  log_std=np.asarray(saved['params']['log_std']).copy())
    if plan.get('projected_initialization'):
        import hashlib
        initialization=Path(plan['projected_initialization'])
        with initialization.open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==plan['projected_initialization_sha256']
        with initialization.open('rb') as stream:initialized=NumpyCheckpoint(stream).load()
        assert initialized['schema']=='v325_single5_projected_actions_init_v1'
        assert initialized['PPO_updates']==0 and initialized['source_networks_embedded'] is False
        params=initialized['params']
    params = {**params, "pi": augment_actor(params["pi"])}
    params = jax.tree.map(jnp.asarray, params)
    assert_single_actor(params)
    zeros = jax.tree.map(jnp.zeros_like, params)
    opt = ActorOptim(zeros, jax.tree.map(jnp.zeros_like,params), jnp.asarray(0,jnp.int32), jnp.asarray(0,jnp.int32))
    anchor=jax.tree.map(jnp.array,params)
    if plan.get('continuation_parent'):
        import hashlib
        path=Path(plan['continuation_parent'])
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==plan['continuation_parent_sha256']
        with path.open('rb') as stream:continued=NumpyCheckpoint(stream).load()
        assert continued['actual_updates']==9607 and continued['stage']==15
        assert_single_actor(continued['params'])
        params=jax.tree.map(jnp.asarray,continued['params'])
        prior=continued['opt']
        opt=ActorOptim(jax.tree.map(jnp.asarray,prior.m),jax.tree.map(jnp.asarray,prior.v),jnp.asarray(prior.t),jnp.asarray(prior.actor_t))
    return ppo.TrainState(params,opt),anchor


def outgoing_loss(error_m_s, *, xp):
    """Retain known precision costs; soften the previously constant tails."""
    x=xp.abs(error_m_s/xp.asarray([.035,.035,.030]))
    huber=.5*xp.minimum(x[...,:2],1.)**2+xp.maximum(x[...,:2]-1.,0.)
    horizontal_raw=xp.sum(huber,axis=-1)
    horizontal=xp.minimum(horizontal_raw,4.)+.25*xp.log1p(xp.maximum(horizontal_raw-4.,0.))
    vertical=.5*xp.minimum(x[...,2],4.)**2+.5*xp.log1p(xp.maximum(x[...,2]-4.,0.))
    return 4.*(horizontal+vertical)


def reward_adjustments(metrics, wind_gain, *, enabled, xp):
    if not enabled:
        return {}
    error = xp.stack([metrics[f'dynamics/outgoing_error_v{axis}_m_s'] for axis in 'xyz'],axis=-1)
    release = metrics['dynamics/release_event']
    active = (metrics['terminated'] < .5) & (metrics['dynamics/reference_valid'] > .5)
    new_outgoing = xp.where(active,-wind_gain*release*outgoing_loss(error,xp=xp),
                            metrics['reward/dynamics_outgoing'])
    center = metrics['hit_contact_center_dist_sum']/xp.maximum(metrics['hit_event_count'],1.)
    return {
        'reward/single5_outgoing_replace':new_outgoing-metrics['reward/dynamics_outgoing'],
        'reward/single5_apex_precision':xp.zeros_like(release),
        'reward/single5_center':xp.zeros_like(release),
    }


class Single5Env:
    """Drop permission observations; preserve causal ball history and wind critic."""
    def __init__(self, inner, reward_enabled=False, goal_reward_enabled=False, random_onset_enabled=False,targeted_reward_enabled=False,weak_release_floor_enabled=False,weak_release_floor_weight=4.):
        self.inner=RandomOnsetEnv(inner,enabled=random_onset_enabled)
        self.weak_release_floor_enabled=weak_release_floor_enabled
        self.weak_release_floor_weight=weak_release_floor_weight
        self.reward_enabled=reward_enabled
        self.targeted_reward_enabled=targeted_reward_enabled
        self.targeted_variant=json.loads((A/'TRAIN_PLAN.json').read_text())['reward_revision']['targeted_variant']
        self.targeted_failure_reserve=json.loads((A/'TRAIN_PLAN.json').read_text())['reward_revision'].get('failure_reserve',0.)
        self.goal_reward_enabled=goal_reward_enabled
        if reward_enabled and goal_reward_enabled:
            raise ValueError('Choose exactly one private reward revision')
        self.obs_dim,self.critic_obs_dim,self.act_dim=118,453,7
        if hasattr(inner,"cfg"):
            assert tuple(inner.cfg.dr_ball_mass_range) == (.004,.004)
            cfg = inner.cfg.wind_config
            assert (cfg.air_density_kg_m3,cfg.drag_coefficient,cfg.radius_m) == (1.2,.47,.02)

    def __getattr__(self,name):
        return getattr(self.inner,name)

    def _encode(self,state,old_obs):
        return jnp.concatenate([old_obs[:,:112],acquisition_features(state,xp=jnp),observed_force_features(old_obs,xp=jnp)],axis=-1)

    def _legacy_obs(self,obs):
        return jnp.concatenate([obs[:,:112],jnp.full_like(obs[:,:2],.25)],axis=-1)

    def _obs(self,state,native_obs):
        return self._encode(state,self.inner._obs(state,native_obs))

    def reset(self,keys):
        state,obs=self.inner.reset(keys)
        return state,self._encode(state,obs)

    def reset_done(self,state,obs,done,keys):
        state,obs=self.inner.reset_done(state,self._legacy_obs(obs),done,keys)
        return state,self._encode(state,obs)

    def get_critic_obs(self,state,obs):
        old=self.inner.get_critic_obs(state,self._legacy_obs(obs))
        return jnp.concatenate([old[:,:451],acquisition_features(state,xp=jnp)],axis=-1)

    def step(self,state,action):
        after,obs,reward,done,metrics=self.inner.step(state,action)
        if self.reward_enabled:
            from fixed_plane_wind_pose import shelter_gain
            gain=shelter_gain(state,self.dt,xp=jnp)*(state.wind_mode>0)
            terms=reward_adjustments(metrics,gain,enabled=True,xp=jnp)
            reward=reward+sum(terms.values())
            metrics.update(terms)
        else:
            for name in ('reward/single5_outgoing_replace','reward/single5_apex_precision','reward/single5_center'):
                metrics[name] = jnp.zeros_like(reward)
        if self.goal_reward_enabled:
            from fixed_plane_wind_pose import shelter_gain
            gain=shelter_gain(state,self.dt,xp=jnp)*(state.wind_mode>0)
            reward,terms=productive_reward(reward,metrics,gain,dt_s=self.dt,xp=jnp)
            metrics.update({'reward/goal_'+key:value for key,value in terms.items()})
        if not self.goal_reward_enabled:
            for name in ('alive','hit','outgoing','normal','height_margin','braking','apex','failure'):
                metrics['reward/goal_'+name]=jnp.zeros_like(reward)
        if self.targeted_reward_enabled:
            from fixed_plane_wind_pose import shelter_gain
            wind_gain=shelter_gain(state,self.dt,xp=jnp)*(state.wind_mode>0)
            elapsed=state.step_count*self.dt-acquisition_time(state)
            clean_gain=jnp.sin(.5*jnp.pi*jnp.clip(elapsed,0.,1.))**2*(state.wind_mode==0)*(state.fifth_hit_s>=0.)
            phase_hits=jnp.where(state.wind_mode==0,jnp.maximum(state.wind_startup_hit_count-5,0),state.post_onset_hits)
            added=targeted_terms(metrics,wind_gain,clean_gain,phase_hits,enabled=True,variant=self.targeted_variant,failure_reserve=self.targeted_failure_reserve,xp=jnp)
            reward=reward+sum(added.values());metrics.update(added)
        else:
            metrics.update({name:jnp.zeros_like(reward) for name in TARGETED_FIELDS})
        if hasattr(state,'wind_background_m_s'):
            metrics['single5/wind_angle_rad']=jnp.arctan2(-jnp.abs(state.wind_background_m_s[:,1]),state.wind_background_m_s[:,0])
        from weak_release_floor import FIELDS as RELEASE_FLOOR_FIELDS,release_floor_terms
        if self.weak_release_floor_enabled:
            floor_terms=release_floor_terms(
                ball_z_m=after.data.qpos[:,self.ball_qadr+2],
                ball_vz_m_s=after.data.qvel[:,self.ball_vadr+2],
                gravity_m_s2=jnp.abs(after.dr_gravity_z),clean=after.wind_mode==0,
                confirmed_hits=after.wind_startup_hit_count,
                released=metrics['dynamics/release_event']>.5,
                terminated=metrics['terminated']>.5,
                enabled=True,weight=self.weak_release_floor_weight,xp=jnp)
            reward=reward+floor_terms['reward/weak_calm_release_floor']
        else:
            floor_terms={key:jnp.zeros_like(reward) for key in RELEASE_FLOOR_FIELDS}
        metrics.update(floor_terms)
        encoded = self._encode(after,obs)
        if hasattr(after,'wind_background_m_s'):
            from fixed_plane_v325_wind import wind_velocity
            actual = wind_velocity(after,after.step_count*self.dt,self.cfg.wind_config)[:,:2]
            estimated = encoded[:,114:116]*1.6
            metrics['observed_force/valid'] = encoded[:,117]
            metrics['observed_force/age_s'] = encoded[:,116]*.6
            metrics['observed_force/error_sq_m2_s2'] = jnp.sum((estimated-actual)**2,axis=-1)
        if hasattr(after,'total_env_steps'):
            physical_steps=after.total_env_steps.astype(jnp.float32)
            offset=8192. if getattr(getattr(self.inner,'cfg',None),'wind_mature_precision_reward',False) else 0.
            metrics['precision_context/physical_steps']=physical_steps
            metrics['precision_context/reward_steps']=physical_steps+offset
        metrics['reward/total']=reward
        return after,encoded,reward,done,metrics
