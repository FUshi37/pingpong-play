"""GPU1 reward-only motion refinement before the preserved small-DR tail."""
from dataclasses import replace
from typing import Any

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v156'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v155'
PARENT_STAGE = 'fplc_v155_gpu1_s8_clean_stability'
PARENT_STEP = 999948288
PARENT_UPDATE = 153
PARENT_SHA256 = '7e33de9a88bb0138f9baef94647d19d980d4f3f8eb15b451445a7205ade65206'
REWARD_FIELDS = (
    'racket_cycle_vxy_penalty_weight', 'racket_stability_angular_speed_penalty_weight',
    'fixed_plane_v128_early_xy_motion_weight', 'racket_flatness_penalty_weight',
    'hit_contact_center_excess_penalty_weight', 'hit_contact_center_excess_radius_m',
    'fixed_plane_v156_adjacent_height_penalty_multiplier',
)


def refined_reward(cfg: Any, level: int) -> Any:
    """Ramp existing additive costs; keep every positive/launch reward fixed."""
    from mjx_juggle_env import MjxFixedPlaneV156RewardConfig
    result = MjxFixedPlaneV156RewardConfig(**vars(cfg))
    return replace(result,
        racket_cycle_vxy_penalty_weight=(13.5, 18.0, 27.0)[level],
        racket_stability_angular_speed_penalty_weight=(7.5, 10.0, 15.0)[level],
        fixed_plane_v128_early_xy_motion_weight=(600.0, 720.0, 840.0)[level],
        racket_flatness_penalty_weight=cfg.racket_flatness_penalty_weight * (1.2, 1.4, 1.6)[level],
        hit_contact_center_excess_penalty_weight=(160.0, 176.0, 200.0)[level],
        hit_contact_center_excess_radius_m=(.003, .0025, .0015)[level],
        fixed_plane_v156_adjacent_height_penalty_multiplier=(1.5, 2.0, 2.5)[level])


def quality_stage(stage: Any, *, name: str, level: int) -> Any:
    """Require simultaneous longevity, centered contact and physical quality."""
    return replace(stage, name=name,
        cfg=replace(refined_reward(stage.cfg, level), fixed_contact_training_phase=name),
        min_updates=(80, 120, 160)[level], convergence_hold_updates=(24, 24, 32)[level],
        target_mean_hits=12.9, target_mean_len_frac=.995,
        target_episode_truncation_rate=.995, target_hit12_rate=.99,
        max_recent_rapid_steady_racket_vxy_rms_m_s=(.073, .065, .055)[level],
        max_recent_rapid_steady_racket_omega_rms_rad_s=(.48, .36, .28)[level],
        max_recent_mean_hit_contact_center_dist_m=(.0066, .005, .0033)[level],
        max_recent_late_adjacent_jump_m=(.0095, .0085, .0075)[level],
        min_recent_late_physical_in_band_rate=(.990, .993, .995)[level],
        max_recent_late_physical_below_floor_rate=(.007, .005, .0035)[level],
        max_recent_late_physical_above_ceiling_rate=(.006, .004, .0035)[level],
        advance_validation_probe_mode='self', advance_eval_deterministic=False,
        notes='V156 reward-only refinement; all late physical impacts enter height gates. '
              'Preserve survival/launch credit and progressively price XY/angular motion, '
              'center error and adjacent height changes. No control/physics/actor change.')


def build_stages(source: list[Any]) -> list[Any]:
    """Retain prior lessons; insert two clean repairs before all seven DR lessons."""
    result = []
    for index, stage in enumerate(source[:7]):
        name = stage.name.replace('v155_', 'v156_', 1)
        result.append(replace(stage, name=name, cfg=replace(stage.cfg,
            fixed_contact_training_phase=name)))
    for level, suffix in enumerate(('motion_entry', 'motion_bridge', 'clean_commit')):
        result.append(quality_stage(source[7],
            name=f'fplc_v156_gpu1_s{8+level}_{suffix}', level=level))
    for index, stage in enumerate(source[8:], 11):
        suffix = stage.name.split('_', 4)[-1]
        result.append(quality_stage(stage,
            name=f'fplc_v156_gpu1_s{index}_{suffix}', level=2))
    return result


def validate_resume_payload(payload: dict[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool) -> None:
    """Pin the stopped source; preserve full state on every own continuation."""
    if (reset_optimizer or reset_critic or payload.get('opt') is None
            or (payload.get('obs_dim'), payload.get('critic_obs_dim'), payload.get('act_dim')) != (72,393,7)
            or not isinstance(payload.get('params'), dict)
            or not {'pi', 'v'}.issubset(payload['params'])):
        raise ValueError('V156 requires72/393/7 actor/critic and full Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (payload.get('stage_name') != PARENT_STAGE or payload.get('step') != PARENT_STEP
                or payload.get('stage_update') != PARENT_UPDATE
                or requested_start_stage != 'fplc_v156_gpu1_s8_motion_entry'
                or resume_curriculum_state):
            raise ValueError('V156 requires pinned V155 S8-u153; fresh reward history')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
            return
        if (payload.get('stage_name') != requested_start_stage
                or not requested_start_stage.startswith('fplc_v156_gpu1_')
                or int(payload.get('stage_update', -1)) < 1):
            raise ValueError('V156 own continuation requires exact stage/history')
    else:
        raise ValueError('V156 rejects another profile')
