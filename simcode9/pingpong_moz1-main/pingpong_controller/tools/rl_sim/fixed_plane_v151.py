"""V151 reward-only precision ramp with sustained-survival incentives."""
from dataclasses import replace


PRECISION_COST_FIELDS = (
    "fixed_plane_v128_early_xy_motion_weight",
    "early_racket_xy_anchor_penalty_weight",
    "early_racket_xy_anchor_transition_weight",
    "fixed_plane_v134_late_xy_anchor_penalty_weight",
    "fixed_plane_v146_ball_anchor_xy_penalty_weight",
    "fixed_plane_v132_early_ball_xy_motion_weight",
    "racket_cycle_vxy_penalty_weight",
    "racket_flatness_penalty_weight",
    "racket_stability_angular_speed_penalty_weight",
)
THIRD_CREDIT_FIELDS = (
    "fixed_plane_v124_edge_z_penalty_weight",
    "fixed_plane_v124_edge_vz_reward_weight",
    "fixed_plane_v122_third_release_vz_weight",
    "fixed_plane_v115_third_hit_bonus_weight",
)


def reward_bridge_config(cfg, *, precision_fraction, center_radius_m,
                         survival_weight=40.0, terminal_base=192.0,
                         third_scale=0.25):
    """Change additive reward leaves without touching bounded hit budgets."""
    changes = {name: getattr(cfg, name) * precision_fraction
               for name in PRECISION_COST_FIELDS}
    changes.update({name: getattr(cfg, name) * third_scale
                    for name in THIRD_CREDIT_FIELDS})
    return replace(
        cfg, **changes,
        hit_contact_center_excess_penalty_weight=(
            cfg.hit_contact_center_excess_penalty_weight * precision_fraction),
        hit_contact_center_excess_radius_m=center_radius_m,
        fixed_contact_survival_reward_weight=survival_weight,
        racket_z_limit_termination_penalty_base=terminal_base,
        fixed_contact_workspace_z_barrier_downward_lookahead_s=0.020,
    )


def build_survival_bridge_stages(source):
    """Preserve all seven gates; acquire continuity before final precision."""
    result = []
    fractions = (0.10, 0.25, 0.50, 1.0)
    radii_m = (0.008, 0.005, 0.003, 0.003)
    for index, stage in enumerate(source):
        name = stage.name.replace("fplc_v150_gpu1_", "fplc_v151_gpu1_", 1)
        cfg = replace(stage.cfg, fixed_contact_training_phase=name)
        if index >= 3:
            cfg = reward_bridge_config(
                cfg, precision_fraction=fractions[index - 3],
                center_radius_m=radii_m[index - 3])
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            "V151 restores the successful V150 stage3-u1700 actor/critic/Adam. "
            "Stages1--3 are preserved; stages4--7 ramp additive precision costs "
            "10/25/50/100%, reduce one-time third-hit credit to25%, reward "
            "post-hit survival40/s and penalize workspace termination192. "
            "A reward-only20ms downward barrier forecast encourages braking. "
            "All gates,72-D observations,90Hz camera,1ms contact, bounded "
            "reference/RMP/PD, workspace, resets and DR remain unchanged.")))
    return result
