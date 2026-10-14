"""Continue measured motion progress with mean-aligned center/height losses."""
from dataclasses import replace
import math
import jax.numpy as jnp
from fixed_plane_v158_gpu1 import REWARD_FIELDS as PREVIOUS_REWARD_FIELDS

PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu1_v159'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu1_v158'
PARENT_STEP=1023016960
PARENT_SHA256='bde2ed9f60afe82949564095ac6b1890e74bf1473bbb096f87da7f6e087633d8'
REWARD_FIELDS=(*PREVIOUS_REWARD_FIELDS,'fixed_plane_v159_adjacent_height_linear',
              'fixed_plane_v159_mean_center_penalty_weight')

def mean_center_penalty(*,event,distance_m,weight):
    if not math.isfinite(float(weight)) or weight<0:
        raise ValueError('Center weight must be finite and nonnegative')
    return jnp.where(jnp.asarray(event,dtype=bool),-weight*jnp.asarray(distance_m)/.003,0.)

def build_stages(source):
    from mjx_juggle_env import MjxFixedPlaneV159RewardConfig
    result=[]
    for index,stage in enumerate(source):
        name=stage.name.replace('v158_','v159_',1)
        cfg=replace(stage.cfg,fixed_contact_training_phase=name)
        if index>=7:
            cfg=MjxFixedPlaneV159RewardConfig(**vars(cfg))
            cfg=replace(cfg,fixed_plane_v159_adjacent_height_linear=True,
                fixed_plane_v159_mean_center_penalty_weight=(.75,1.,1.5)[min(index-7,2)])
        result.append(replace(stage,name=name,cfg=cfg,notes=(
            'V159 continues experimentally measured V158 motion improvements; '
            'linear adjacent loss directly matches mean absolute jump, and '
            'direct center-distance cost retains gradient inside old deadband. '
            'All motion/height/survival, control, observation, physics and DR retained.')))
    return result

def validate_resume_payload(payload,*,requested_start_stage,resume_curriculum_state,reset_optimizer,reset_critic):
    if (reset_optimizer or reset_critic or payload.get('opt') is None
        or (payload.get('obs_dim'),payload.get('critic_obs_dim'),payload.get('act_dim'))!=(72,393,7)
        or not {'pi','v'}.issubset(payload.get('params',{}))):
        raise ValueError('V159 requires full72/393/7 actor/critic/Adam')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        if (payload.get('stage_name')!='fplc_v158_gpu1_s8_motion_entry'
            or payload.get('step')!=PARENT_STEP or payload.get('stage_update')!=128
            or resume_curriculum_state or requested_start_stage!='fplc_v159_gpu1_s8_motion_entry'):
            raise ValueError('V159 pins V158u128 with fresh mean-reward history')
    elif source==PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=requested_start_stage)
        elif payload.get('stage_name')!=requested_start_stage:
            raise ValueError('V159 continuation requires exact stage/history')
    else:
        raise ValueError('V159 rejects another profile')
