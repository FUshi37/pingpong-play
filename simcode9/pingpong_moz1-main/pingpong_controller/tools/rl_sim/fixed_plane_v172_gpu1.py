"""Continue the improving position-only actor with measured advancement gates."""
from dataclasses import replace

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v172'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v159'
PARENT_SHA256 = 'f7a2faeaaa146e8ddc4b27520e012752343899efc7a52e7e5d85ea9f89e3c1a0'
GATE_FIELDS = ('max_recent_mean_hit_contact_center_dist_m',
    'max_recent_rapid_steady_racket_vxy_rms_m_s',
    'max_recent_rapid_steady_racket_omega_rms_rad_s',
    'max_recent_late_adjacent_jump_m', 'min_recent_late_physical_in_band_rate')


def build_stages(source):
    """Keep17 original lessons, rewards, domains and stage identities."""
    result = []
    for index, stage in enumerate(source):
        gates = {}
        if index == 8:
            gates = dict(max_recent_mean_hit_contact_center_dist_m=.0053,
                max_recent_rapid_steady_racket_vxy_rms_m_s=.0525,
                max_recent_rapid_steady_racket_omega_rms_rad_s=.280,
                max_recent_late_adjacent_jump_m=.009,
                min_recent_late_physical_in_band_rate=.994)
        elif index >= 9:
            gates = dict(max_recent_mean_hit_contact_center_dist_m=.0055,
                max_recent_rapid_steady_racket_omega_rms_rad_s=.280,
                max_recent_late_adjacent_jump_m=.009,
                min_recent_late_physical_in_band_rate=.994)
        result.append(replace(stage, **gates, notes=(
            'V172 retains the exact improving V15972-D position-only actor/rewards. '
            'Only user-authorized advancement gates change; the calibrated S9 '
            'domain is independently retained throughout later noise and tiny DR.')))
    return result


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic):
    """Restore full state and unchanged-reward causal history."""
    if (reset_optimizer or reset_critic
            or payload.get('opt') is None
            or (payload.get('obs_dim'),payload.get('critic_obs_dim'),
                payload.get('act_dim')) != (72,393,7)
            or not {'pi','v'}.issubset(payload.get('params',{}))):
        raise ValueError('V172 requires full72/393/7 and exact causal history')
    source = payload.get('args',{}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (not resume_curriculum_state
                or payload.get('stage_name') != requested_start_stage
                or payload.get('stage_name') != 'fplc_v159_gpu1_s9_motion_bridge'
                or payload.get('stage_update') != 1820
                or payload.get('step') != 1275068416):
            raise ValueError('V172 pins improving V159u1820')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
        elif payload.get('stage_name') != requested_start_stage:
            raise ValueError('V172 continuation requires exact stage/history')
    else:
        raise ValueError('V172 rejects another source profile')
