"""Joint reward refinement following measured XY/angular response in V156."""
from dataclasses import replace

from fixed_plane_v156 import REWARD_FIELDS as V156_REWARD_FIELDS

PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu1_v157'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_gpu1_v156'
PARENT_STAGE='fplc_v156_gpu1_s8_motion_entry'
PARENT_STEP=1006239744
PARENT_UPDATE=48
PARENT_SHA256='72898c7fc3d0ff0c1ebd2ece11f921d64efaca8acbd3e9fb88bc39372efc8149'
REWARD_FIELDS=(*V156_REWARD_FIELDS,'fixed_plane_v132_early_ball_xy_motion_weight')


def build_stages(source):
    """Keep the physical course; strengthen all four coupled quality losses."""
    result=[]
    for index,stage in enumerate(source):
        name=stage.name.replace('v156_','v157_',1)
        cfg=replace(stage.cfg,fixed_contact_training_phase=name)
        gates={}
        if index>=7:
            level=min(index-7,2)
            cfg=replace(cfg,racket_cycle_vxy_penalty_weight=(18.,22.5,27.)[level],
                racket_stability_angular_speed_penalty_weight=(10.,12.5,15.)[level],
                hit_contact_center_excess_penalty_weight=(176.,208.,240.)[level],
                hit_contact_center_excess_radius_m=(.0025,.002,.001)[level],
                fixed_plane_v156_adjacent_height_penalty_multiplier=(2.,2.5,3.)[level],
                fixed_plane_v132_early_ball_xy_motion_weight=(
                    stage.cfg.fixed_plane_v132_early_ball_xy_motion_weight*1.25))
            if level==2:
                gates=dict(max_recent_rapid_steady_racket_vxy_rms_m_s=.053,
                    max_recent_rapid_steady_racket_omega_rms_rad_s=.25,
                    max_recent_mean_hit_contact_center_dist_m=.003,
                    max_recent_late_adjacent_jump_m=.007)
        result.append(replace(stage,name=name,cfg=cfg,**gates,notes=(
            'V157 keeps measured V156 XY/angular improvement; adds paired center '
            'and adjacent-height gradient strength plus early ball XY control. '
            'Clean commit and every DR lesson require current GPU0-level quality. '
            'Every positive/launch reward, physics, observation and control is unchanged.')))
    return result


def validate_resume_payload(payload,*,requested_start_stage,resume_curriculum_state,
                            reset_optimizer,reset_critic):
    """Restore the proven motion source and full actor/critic/Adam state."""
    if (reset_optimizer or reset_critic or payload.get('opt') is None
            or (payload.get('obs_dim'),payload.get('critic_obs_dim'),payload.get('act_dim'))!=(72,393,7)
            or not isinstance(payload.get('params'),dict)
            or not {'pi','v'}.issubset(payload['params'])):
        raise ValueError('V157 requires72/393/7 actor/critic and full Adam')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        if (payload.get('stage_name')!=PARENT_STAGE or payload.get('step')!=PARENT_STEP
                or payload.get('stage_update')!=PARENT_UPDATE or resume_curriculum_state
                or requested_start_stage!='fplc_v157_gpu1_s8_motion_entry'):
            raise ValueError('V157 requires pinned V156u48 with fresh reward history')
    elif source==PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=requested_start_stage)
        elif (payload.get('stage_name')!=requested_start_stage
                or not requested_start_stage.startswith('fplc_v157_gpu1_')
                or int(payload.get('stage_update',-1))<1):
            raise ValueError('V157 own continuation requires exact stage/history')
    else:
        raise ValueError('V157 rejects another profile')
