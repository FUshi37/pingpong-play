"""GPU1 continuation that enters at its accepted reward before precision ramps."""
from dataclasses import replace
from typing import Any

from fixed_plane_v151 import PRECISION_COST_FIELDS
from fixed_plane_v153 import validate_resume_payload as validate_v153

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v155'
PRECISION_FIELDS = (*PRECISION_COST_FIELDS, 'hit_contact_center_excess_penalty_weight',
                    'hit_contact_center_excess_radius_m')


def build_stages(source: list[Any], legacy: list[Any]) -> list[Any]:
    """Retain source precision at S4; insert a 17.5% bridge before 25%."""
    result = []
    for index, stage in enumerate(source):
        number = index + 1 + int(index >= 4)
        suffix = stage.name.split(f'_s{index+1}_', 1)[1]
        name = f'fplc_v155_gpu1_s{number}_{suffix}'
        changes = dict(fixed_contact_training_phase=name)
        if 3 <= index <= 6:
            changes.update({field: getattr(legacy[index].cfg, field)
                            for field in PRECISION_FIELDS})
        new = replace(stage, name=name, cfg=replace(stage.cfg, **changes),
                      notes='V155 starts at accepted S4 precision; one 17.5% bridge '
                            'precedes the proven 25/50/100% precision course. '
                            'Height/normal statistics and small DR retain V154.')
        if index == 3:
            new = replace(new, min_updates=60)
        result.append(new)
        if index == 3:
            bridge_name = 'fplc_v155_gpu1_s5_precision_bridge_0175'
            bridge_fields = {field: 1.75 * getattr(new.cfg, field)
                             for field in PRECISION_COST_FIELDS}
            bridge_fields.update(
                hit_contact_center_excess_penalty_weight=(
                    1.75 * new.cfg.hit_contact_center_excess_penalty_weight),
                hit_contact_center_excess_radius_m=.0065,
                fixed_contact_training_phase=bridge_name)
            result.append(replace(new, name=bridge_name,
                                  cfg=replace(new.cfg, **bridge_fields), min_updates=60))
    return result


def validate_resume_payload(payload: dict[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool) -> None:
    """Restore the pinned release or an exact-history V155 continuation."""
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PROFILE:
        if not requested_start_stage.startswith('fplc_v155_gpu1_'):
            raise ValueError('V155 requires its exact own stage identity')
        translated = {**payload, 'args': {'curriculum_profile':
            'fixed_plane_limit_cycle_qvel_rmp_gpu1_v153'},
            'stage_name': str(payload.get('stage_name', '')).replace('v155_', 'v153_', 1)}
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_v153(translated, gpu=1,
                requested_start_stage=translated['stage_name'], resume_curriculum_state=True,
                reset_optimizer=reset_optimizer, reset_critic=reset_critic)
            validate_completed_stage(payload, next_stage=requested_start_stage)
            return
        validate_v153(translated, gpu=1,
            requested_start_stage=requested_start_stage.replace('v155_', 'v153_', 1),
            resume_curriculum_state=resume_curriculum_state,
            reset_optimizer=reset_optimizer, reset_critic=reset_critic)
    else:
        if requested_start_stage != 'fplc_v155_gpu1_s4_fourth_entry':
            raise ValueError('V155 must enter at the accepted S4 reward')
        validate_v153(payload, gpu=1, requested_start_stage='fplc_v153_gpu1_s5_steady_quality',
            resume_curriculum_state=resume_curriculum_state,
            reset_optimizer=reset_optimizer, reset_critic=reset_critic)
