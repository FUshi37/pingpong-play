"""One delayed wind onset per episode; physical and confirmation clocks are distinct."""
from collections import namedtuple
import jax
import jax.numpy as jnp


def acquisition_time(state):
    return getattr(state, 'fifth_hit_s', state.wind_startup_started_s)


def schedule_after_fifth(fifth_s, old_start_s, now_s, count, delay_steps, mode, *, xp):
    began=(fifth_s<0.)&(count>=5)
    fifth=xp.where(began,now_s,fifth_s)
    start=xp.where(began,now_s+xp.where(mode>0,delay_steps*.005,0.),old_start_s)
    return start,fifth


def confirmation_gap_limit(period_s, windy, *, xp):
    period=xp.where(xp.isfinite(period_s)&(period_s>0.),period_s,.4)
    return xp.where(windy,xp.clip(3.*period,1.2,1.8),1.2)


_FIELDS=('core fifth_hit_s delay_steps last_raw_step last_confirm_step has_raw has_confirm '
         'physical_max_ratio confirmed_max_ratio physical_max_gap_s confirmed_max_gap_s '
         'post_onset_hits')


class OnsetState(namedtuple('RandomOnsetStateBase',_FIELDS)):
    __slots__=()
    def __getattr__(self,name):return getattr(self.core,name)
    def _replace(self,**values):
        local={key:value for key,value in values.items() if key in self._fields}
        forwarded={key:value for key,value in values.items() if key not in self._fields}
        if forwarded:local['core']=self.core._replace(**forwarded)
        return super()._replace(**local)


TIMING_FIELDS=tuple('timing/'+key for key in (
    'fifth_hit_s','scheduled_start_s','delay_steps','after_onset_s','post_onset_hits',
    'physical_gap_s','confirmed_gap_s','physical_max_gap_s','confirmed_max_gap_s',
    'confirmation_limit_s','physical_max_ratio','confirmed_max_ratio','physical_recent',
    'long_confirmed_gap_event','actual_wind_x_m_s','actual_wind_y_m_s'))
PREVIEW_FIELDS=tuple('diagnostic/'+key for key in (
    'predicted_center_m','precontact','racket_z_preview_m','racket_roll_deg',
    'racket_pitch_deg','racket_vz_m_s','episode_seconds','confirmed_hits','racket_az_m_s2'))


class RandomOnsetEnv:
    def __init__(self,inner,enabled=False):
        self.inner=inner;self.enabled=enabled
    def __getattr__(self,name):return getattr(self.inner,name)

    def _fresh(self,core,keys):
        count=core.step_count.shape[0]
        delay=jax.vmap(lambda key:jax.random.randint(jax.random.fold_in(key,2026112305),(),5,101))(keys)
        zero=jnp.zeros(count,jnp.float32);step=jnp.zeros(count,jnp.int32);flag=jnp.zeros(count,bool)
        return OnsetState(core,zero-1.,delay,step,step,flag,flag,zero,zero,zero,zero,step)

    def reset(self,keys):
        core,obs=self.inner.reset(keys)
        return (self._fresh(core,keys),obs) if self.enabled else (core,obs)

    def reset_done(self,state,obs,done,keys):
        if not self.enabled:return self.inner.reset_done(state,obs,done,keys)
        core,obs=self.inner.reset_done(state.core,obs,done,keys)
        fresh=self._fresh(core,keys)
        choose=lambda new,old:jnp.where(done,new,old)
        return OnsetState(core,*jax.tree.map(choose,tuple(fresh)[1:],tuple(state)[1:])),obs

    def get_critic_obs(self,state,obs):
        return self.inner.get_critic_obs(state.core if self.enabled else state,obs)

    def _obs(self,state,native_obs):
        return self.inner._obs(state.core if self.enabled else state,native_obs)

    def step(self,state,action):
        if not self.enabled:return self.inner.step(state,action)
        core,obs,reward,done,m=self.inner.step(state.core,action)
        now=core.step_count*self.dt
        start,fifth=schedule_after_fifth(state.fifth_hit_s,core.wind_startup_started_s,
            now,core.wind_startup_hit_count,state.delay_steps,core.wind_mode,xp=jnp)
        core=core._replace(wind_startup_started_s=start)
        raw=m['raw_physical_contact_edge']>.5
        confirmed=m['hit_event_count']>.5
        raw_gap=jnp.where(state.has_raw,(core.step_count-state.last_raw_step)*self.dt,0.)
        confirm_gap=jnp.where(state.has_confirm,(core.step_count-state.last_confirm_step)*self.dt,0.)
        active=(core.wind_mode>0)&(start>=0.)&(now>start)
        limit=confirmation_gap_limit(m['dynamics/period_s'],active,xp=jnp)
        raw_ratio=jnp.maximum(state.physical_max_ratio,raw_gap/1.2)
        confirmed_ratio=jnp.maximum(state.confirmed_max_ratio,confirm_gap/limit)
        raw_max=jnp.maximum(state.physical_max_gap_s,raw_gap)
        confirm_max=jnp.maximum(state.confirmed_max_gap_s,confirm_gap)
        after=OnsetState(core,fifth,state.delay_steps,
            jnp.where(raw,core.step_count,state.last_raw_step),
            jnp.where(confirmed,core.step_count,state.last_confirm_step),
            state.has_raw|raw,state.has_confirm|confirmed,raw_ratio,confirmed_ratio,
            raw_max,confirm_max,state.post_onset_hits+(confirmed&active).astype(jnp.int32))
        from fixed_plane_v325_wind import wind_velocity
        wind=wind_velocity(core,now,self.cfg.wind_config)
        values=(fifth,start,state.delay_steps, jnp.where(active,now-start,0.),after.post_onset_hits,
            raw_gap,confirm_gap,raw_max,confirm_max,limit,raw_ratio,confirmed_ratio,
            (after.has_raw&(raw_gap<=1.2)).astype(jnp.float32),
            (confirmed&state.has_confirm&(confirm_gap>1.2)).astype(jnp.float32),wind[:,0],wind[:,1])
        m.update({key:jnp.asarray(value,jnp.float32) for key,value in zip(TIMING_FIELDS,values)})
        ball=core.data.qpos[:,self.ball_qadr:self.ball_qadr+3]
        velocity=core.data.qvel[:,self.ball_vadr:self.ball_vadr+3]
        racket=core.data.site_xpos[:,self.racket_site_id]
        rotation=core.data.site_xmat[:,self.racket_site_id].reshape((-1,3,3))
        relative=jnp.einsum('ij,njk->nik',self.fixed_contact_racket_reference_rotation.T,rotation)
        pitch=jnp.rad2deg(jnp.arcsin(jnp.clip(-relative[:,2,0],-1.,1.)))
        roll=jnp.rad2deg(jnp.arctan2(relative[:,2,1],relative[:,2,2]))
        # A short causal preview for reward/diagnostics only; never an action correction.
        offset=core.reset_ball_pos[:,2]-core.reset_ball_surface_gap-core.racket_anchor[:,2]
        distance=jnp.maximum(ball[:,2]-racket[:,2]-offset,0.)
        ttc=(velocity[:,2]+jnp.sqrt(jnp.maximum(velocity[:,2]**2+2.*9.81*distance,0.)))/9.81
        from fixed_plane_v325_wind import incremental_force
        force,_=incremental_force(velocity,wind,self.cfg.wind_config)
        mass=core.model.body_mass[:,self.ball_body_id]
        xy=ball[:,:2]+velocity[:,:2]*ttc[:,None]+.5*force[:,:2]/mass[:,None]*ttc[:,None]**2
        racket_xy=racket[:,:2]+core.prev_racket_vel[:,:2]*ttc[:,None]
        center=jnp.linalg.norm(xy-racket_xy,axis=-1)
        precontact=(velocity[:,2]<-.1)&(ttc>0.)&(ttc<.12)&~core.prev_contact
        az=(core.prev_racket_vel[:,2]-state.core.prev_racket_vel[:,2])/self.dt
        preview=(center,precontact.astype(jnp.float32),racket[:,2]+.08*core.prev_racket_vel[:,2],
                 roll,pitch,core.prev_racket_vel[:,2],now,core.wind_startup_hit_count,az)
        m.update({key:jnp.asarray(value,jnp.float32) for key,value in zip(PREVIEW_FIELDS,preview)})
        return after,obs,reward,done,m
