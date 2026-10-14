"""Reward-only late-impact squared-motion alignment after negative V157 evidence."""
from dataclasses import replace
import math
import jax.numpy as jnp
from fixed_plane_v157_gpu1 import validate_resume_payload as validate_previous
from fixed_plane_v156 import REWARD_FIELDS as BASE_REWARD_FIELDS

PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu1_v158'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu1_v156'
PARENT_SHA256='72898c7fc3d0ff0c1ebd2ece11f921d64efaca8acbd3e9fb88bc39372efc8149'
REWARD_FIELDS=(*BASE_REWARD_FIELDS,'fixed_plane_v132_early_ball_xy_motion_weight',
    'hit_racket_angular_speed_reward_target_rad_s','hit_racket_angular_speed_reward_sigma_rad_s',
    'fixed_plane_v158_late_squared_vxy_weight','fixed_plane_v158_late_squared_omega_weight')

def late_impact_squared_motion_penalties(*,event,hit_count,vxy_m_s,omega_rad_s,vxy_weight,omega_weight):
    """Use exactly the fourth-plus counted-impact population of the RMS gates."""
    if not all(math.isfinite(float(x)) and x>=0 for x in (vxy_weight,omega_weight)):
        raise ValueError('Impact squared weights must be finite and nonnegative')
    active=jnp.asarray(event,dtype=bool)&(jnp.asarray(hit_count)>=4)
    xy=jnp.asarray(vxy_m_s);omega=jnp.asarray(omega_rad_s)
    if active.shape!=xy.shape or active.shape!=omega.shape:
        raise ValueError('Impact quality arrays must share shape')
    return (jnp.where(active,-vxy_weight*(xy/.053)**2,0.),
            jnp.where(active,-omega_weight*(omega/.25)**2,0.))

def build_stages(source):
    from mjx_juggle_env import MjxFixedPlaneV158RewardConfig
    result=[]
    for index,stage in enumerate(source):
        name=stage.name.replace('v156_','v158_',1)
        cfg=replace(stage.cfg,fixed_contact_training_phase=name)
        gates={}
        if index>=7:
            level=min(index-7,2)
            cfg=MjxFixedPlaneV158RewardConfig(**vars(cfg))
            cfg=replace(cfg,fixed_plane_v158_late_squared_vxy_weight=(1.5,2.,2.5)[level],
                fixed_plane_v158_late_squared_omega_weight=(1.,1.5,2.)[level],
                fixed_plane_v156_adjacent_height_penalty_multiplier=(2.,2.5,3.)[level],
                fixed_plane_v132_early_ball_xy_motion_weight=stage.cfg.fixed_plane_v132_early_ball_xy_motion_weight*1.25,
                hit_racket_angular_speed_reward_target_rad_s=.1,
                hit_racket_angular_speed_reward_sigma_rad_s=.25)
            if level==2:
                gates=dict(max_recent_rapid_steady_racket_vxy_rms_m_s=.053,
                    max_recent_rapid_steady_racket_omega_rms_rad_s=.25,
                    max_recent_mean_hit_contact_center_dist_m=.003,
                    max_recent_late_adjacent_jump_m=.007)
        result.append(replace(stage,name=name,cfg=cfg,**gates,notes=(
            'V158 preserves V156 dense/center rewards and adds uncapped squared '
            'late counted-impact motion costs matching measured RMS gates. '
            'No network/control/physics/reset/DR/PPO change.')))
    return result

def validate_resume_payload(payload,**kwargs):
    proxy=dict(payload)
    if payload.get('args',{}).get('curriculum_profile')==PROFILE:
        proxy['args']=dict(payload['args'],curriculum_profile='fixed_plane_limit_cycle_qvel_rmp_gpu1_v157')
        proxy['stage_name']=payload['stage_name'].replace('v158_','v157_',1)
        kwargs=dict(kwargs,requested_start_stage=kwargs['requested_start_stage'].replace('v158_','v157_',1))
        if not kwargs['resume_curriculum_state']:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=kwargs['requested_start_stage'].replace('v157_','v158_',1))
            return
    else:
        kwargs=dict(kwargs,requested_start_stage=kwargs['requested_start_stage'].replace('v158_','v157_',1))
    validate_previous(proxy,**kwargs)
