"""Continue accepted releases with contact quality aligned to horizontal pose."""
from dataclasses import replace
import math
from typing import Any

from fixed_plane_v153 import SOURCE_SHA256, validate_resume_payload as validate_v153

PROFILES = tuple(f'fixed_plane_limit_cycle_qvel_rmp_gpu{g}_v154' for g in (0,1))


def build_stages(source: list[Any], *, gpu: int) -> list[Any]:
    from mjx_juggle_env import MjxFixedPlaneV154ContactConfig
    result = []
    for index,stage in enumerate(source):
        name = stage.name.replace(f'fplc_v153_gpu{gpu}_',f'fplc_v154_gpu{gpu}_',1)
        cfg = replace(stage.cfg,fixed_contact_training_phase=name)
        gates = {}
        if index>=3:
            cfg = MjxFixedPlaneV154ContactConfig(**vars(cfg))
            cfg = replace(cfg,fixed_contact_quality_pose_uses_normal_tilt=True)
            old_floors = stage.min_recent_fixed_contact_quality_components
            gates['min_recent_fixed_contact_quality_components'] = (
                old_floors[0],math.exp(-.5*(3/2)**2),*old_floors[2:])
        result.append(replace(stage,name=name,cfg=cfg,**gates,
            notes='V154 retains V153 height/precision/small-DR values; contact pose '
                  'reward and admission use physical normal tilt with 2deg sigma '
                  'and a 3deg boundary floor. Full angular-speed gates/costs and '
                  'all legacy behavior remain unchanged. Resume at clean Stage5.'))
    return result


def validate_resume_payload(payload: dict[str,Any], *, gpu: int,
    requested_start_stage: str, resume_curriculum_state: bool,
    reset_optimizer: bool, reset_critic: bool) -> None:
    """Pin accepted full-state releases, or restore exact V154 course history."""
    if gpu not in (0,1):
        raise ValueError('V154 requires GPU0/GPU1 identity')
    args = payload.get('args',{})
    source = args.get('curriculum_profile') if isinstance(args,dict) else getattr(args,'curriculum_profile',None)
    if source == PROFILES[gpu]:
        if not requested_start_stage.startswith(f'fplc_v154_gpu{gpu}_'):
            raise ValueError('V154 continuation requires its exact own stage identity')
        translated = {**payload,'args':{'curriculum_profile':f'fixed_plane_limit_cycle_qvel_rmp_gpu{gpu}_v153'},
                      'stage_name':str(payload.get('stage_name','')).replace('v154_','v153_',1)}
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_v153(translated, gpu=gpu,
                requested_start_stage=translated['stage_name'], resume_curriculum_state=True,
                reset_optimizer=reset_optimizer, reset_critic=reset_critic)
            validate_completed_stage(payload, next_stage=requested_start_stage)
            return
        validate_v153(translated,gpu=gpu,
            requested_start_stage=requested_start_stage.replace('v154_','v153_',1),
            resume_curriculum_state=resume_curriculum_state,reset_optimizer=reset_optimizer,
            reset_critic=reset_critic)
    else:
        if requested_start_stage!=f'fplc_v154_gpu{gpu}_s5_steady_quality':
            raise ValueError('V154 accepted releases must start at clean quality Stage5')
        target = f'fplc_v153_gpu{gpu}_'+('s8_obs_bridge_2mm' if gpu==0 else 's5_steady_quality')
        validate_v153(payload,gpu=gpu,requested_start_stage=target,
            resume_curriculum_state=resume_curriculum_state,reset_optimizer=reset_optimizer,
            reset_critic=reset_critic)
