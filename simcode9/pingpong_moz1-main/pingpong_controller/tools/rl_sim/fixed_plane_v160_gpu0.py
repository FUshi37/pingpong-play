"""Gentle mean-error and impact-quality continuation for the working GPU0 actor."""
from dataclasses import replace
from fixed_plane_v159_gpu1 import REWARD_FIELDS

PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu0_v160'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu0_v154'
PARENT_STEP=2253783040
PARENT_SHA256='e770d59ede246481ac6e31198c262475e4d832ab099abdc763b7193133bb4410'

def build_stages(source):
    from mjx_juggle_env import MjxFixedPlaneV159RewardConfig
    result=[]
    for index,stage in enumerate(source):
        name=stage.name.replace('v154_','v160_',1)
        cfg=replace(stage.cfg,fixed_contact_training_phase=name)
        gates={}
        if index>=6:
            cfg=MjxFixedPlaneV159RewardConfig(**vars(cfg))
            cfg=replace(cfg,fixed_plane_v158_late_squared_vxy_weight=.35,
                fixed_plane_v158_late_squared_omega_weight=.35,
                fixed_plane_v159_adjacent_height_linear=True,
                fixed_plane_v159_mean_center_penalty_weight=.5,
                fixed_plane_v156_adjacent_height_penalty_multiplier=1.25)
            gates=dict(min_recent_late_physical_in_band_rate=.995,
                max_recent_late_physical_below_floor_rate=.0035,
                max_recent_late_physical_above_ceiling_rate=.0035,
                max_recent_rapid_steady_racket_vxy_rms_m_s=.055,
                max_recent_rapid_steady_racket_omega_rms_rad_s=.26,
                max_recent_mean_hit_contact_center_dist_m=.0032,
                max_recent_late_adjacent_jump_m=.0075,
                advance_validation_probe_mode='self',advance_eval_deterministic=False)
        result.append(replace(stage,name=name,cfg=cfg,**gates,notes=(
            'V160 preserves the working GPU0 reward/control/domain; adds gentle '
            'mean-center, linear adjacent-apex and squared late-impact motion costs. '
            'All clean and DR stages retain current-quality ceilings and99.5% physical band.')))
    return result

def validate_resume_payload(payload,*,requested_start_stage,resume_curriculum_state,reset_optimizer,reset_critic):
    if (reset_optimizer or reset_critic or payload.get('opt') is None
        or (payload.get('obs_dim'),payload.get('critic_obs_dim'),payload.get('act_dim'))!=(156,477,7)
        or not {'pi','v'}.issubset(payload.get('params',{}))):
        raise ValueError('V160 requires full156/477/7 actor/critic/Adam')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        if (payload.get('stage_name')!='fplc_v154_gpu0_s7_clean_stability'
            or payload.get('step')!=PARENT_STEP or payload.get('stage_update')!=501
            or resume_curriculum_state or requested_start_stage!='fplc_v160_gpu0_s7_clean_stability'):
            raise ValueError('V160 pins GPU0V154u501 with fresh reward history')
    elif source==PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=requested_start_stage)
        elif payload.get('stage_name')!=requested_start_stage:
            raise ValueError('V160 continuation requires exact stage/history')
    else:
        raise ValueError('V160 rejects another profile')
