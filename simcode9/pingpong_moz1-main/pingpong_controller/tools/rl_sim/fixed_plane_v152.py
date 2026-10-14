"""V152 contact-height gate calibration and bounded candidate DR lessons."""
from dataclasses import replace
from typing import Any


SOURCE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu0_v149"
PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu0_v152"
SOURCE_SHA256 = "308b47da1b29ad370b02cd7ddad0f4883f603d35f0c518c5c378c2045051982d"
CONTACT_HEIGHT_GATE_M = .060
LATE_RACKET_XY_RMS_GATE_M_S = .105
TAIL_LABELS = (
    "obs_bridge_2mm", "obs_bridge_4mm", "obs_registered_8mm",
    "rmp_pd_bridge", "rmp_pd_registered", "mass_registered",
    "target_hold_bridge", "target_hold_registered", "dr_consolidation",
)


def _observation_dr(cfg: Any, fraction: float) -> Any:
    """Ramp within the registered V16 in-view observation candidate bounds."""
    return replace(
        cfg, ball_obs_pos_noise_std=.008 * fraction,
        ball_obs_vel_noise_std=.080 * fraction,
        dr_randomize_ball_obs_frame=True,
        dr_ball_obs_pos_bias_base_m=(.007 * fraction,) * 3,
        dr_ball_obs_rot_bias_deg=(fraction, fraction, 1.5 * fraction),
        dr_ball_obs_vel_bias_base_m_s=(.070 * fraction, .070 * fraction, .100 * fraction),
        dr_ball_obs_scale_range=(1.0 - .012 * fraction, 1.0 + .012 * fraction),
    )


def _plant_dr(cfg: Any, registered: Any, fraction: float) -> Any:
    """Use V149's existing candidate ranges, initially at half width."""
    changes = dict(dr_randomize_pd=True, dr_randomize_recovered_rmp=True)
    for key in (
        "dr_pd_kp_mult_range", "dr_pd_kv_mult_range",
        "dr_rmp_jnt_kp_mult_range", "dr_rmp_jnt_kd_mult_range",
        "dr_rmp_estimator_process_mult_range", "dr_rmp_estimator_measure_mult_range",
        "dr_rmp_velocity_feedforward_range", "dr_rmp_acceleration_weight_mult_range",
        "dr_rmp_target_filter_length_range", "dr_rmp_output_delay_offset_steps_range",
    ):
        bounds = getattr(registered, key)
        nominal = getattr(cfg, key)[0]
        values = tuple(nominal + fraction * (v - nominal) for v in bounds)
        changes[key] = (tuple(int(v) for v in values)
                        if key.endswith("steps_range") or key.endswith("length_range")
                        else values)
    return replace(cfg, **changes)


def build_progression_stages(source: list[Any]) -> list[Any]:
    """Keep seven clean lessons and append nine independently named DR lessons."""
    result = []
    for index, stage in enumerate(source):
        name = stage.name.replace("fplc_v149_gpu0_", "fplc_v152_gpu0_", 1)
        gate = stage.max_recent_rapid_steady_surface_z_abs_error_m
        result.append(replace(
            stage, name=name,
            cfg=replace(stage.cfg, fixed_contact_training_phase=name),
            max_recent_rapid_steady_surface_z_abs_error_m=(
                gate if index < 3 else CONTACT_HEIGHT_GATE_M),
            max_recent_rapid_steady_racket_vxy_rms_m_s=(
                stage.max_recent_rapid_steady_racket_vxy_rms_m_s if index < 5
                else LATE_RACKET_XY_RMS_GATE_M_S),
            notes="V152 calibrates contact-height admission to60mm and late XY RMS "
                  "admission to0.105m/s; all "
                  "apex, center, survival and safety rewards/gates retain V149. "
                  "The pinned source passes rolling and independent block validation "
                  "before Stage5 continuation. Stages8--16 add bounded candidate DR; "
                  "calibrated contact, actor156 and the RMP control contract remain exact.",
        ))
    base = result[-1]
    cfg = base.cfg
    for number, label in enumerate(TAIL_LABELS, 8):
        if number <= 10:
            cfg = _observation_dr(cfg, (0.25, 0.50, 1.0)[number - 8])
        elif number in (11, 12):
            # Interpolate from nominal each time rather than from the prior bridge.
            plant_nominal = replace(cfg, **{
                k: v for k, v in vars(base.cfg).items()
                if k.startswith(("dr_pd_", "dr_rmp_")) and k.endswith("_range")
            })
            cfg = _plant_dr(plant_nominal, source[3].cfg, .5 if number == 11 else 1.0)
        elif number == 13:
            cfg = replace(cfg, dr_ball_mass_range=(.0039, .0041))
        elif number in (14, 15):
            cfg = replace(
                cfg, dr_rmp_target_update_hold_probability=(.050 if number == 14 else .080),
                dr_rmp_target_update_hold_tail_probability=(.011 if number == 14 else .022),
            )
        name = f"fplc_v152_gpu0_s{number}_{label}"
        result.append(replace(
            base, name=name, cfg=replace(cfg, fixed_contact_training_phase=name),
            notes="V152 graduated in-view observation, registered RMP/PD, attested "
                  "3.9--4.1g mass and measured pre-RMP target-publication holds. "
                  "Contact/spin/mount/reset ranges stay fixed. These are experimental "
                  "candidate ranges; every original Stage7 quality gate remains active "
                  "with60mm contact-height/0.105m/s XY RMS admission and no forced advancement.",
        ))
    return result


def validate_progression_resume_payload(
    payload: dict[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
) -> None:
    """Preserve the pinned156/477/7 actor/critic/Adam; require exact own history."""
    args = payload.get("args", {})
    source = (args.get("curriculum_profile") if isinstance(args, dict)
              else getattr(args, "curriculum_profile", None))
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or payload.get("obs_dim") != 156 or payload.get("critic_obs_dim") != 477
            or payload.get("act_dim") != 7 or not isinstance(payload.get("params"), dict)
            or not {"pi", "v"}.issubset(payload["params"])):
        raise ValueError("V152 requires its156/477/7 actor, critic and full Adam state")
    if source == SOURCE_PROFILE:
        if (payload.get("stage_name") != "fplc_v149_gpu0_s4_fourth_entry"
                or payload.get("stage_update") != 2299 or payload.get("step") != 1747714048
                or payload.get("stop_reason") != "received SIGINT"
                or resume_curriculum_state
                or requested_start_stage != "fplc_v152_gpu0_s5_steady_quality"):
            raise ValueError("V152 requires pinned V149u2299 and reviewed Stage5 admission")
    elif source == PROFILE:
        if (not resume_curriculum_state or payload.get("stage_name") != requested_start_stage
                or not requested_start_stage.startswith("fplc_v152_gpu0_")
                or int(payload.get("stage_update", -1)) < 1):
            raise ValueError("V152 continuation requires exact stage and complete history")
    else:
        raise ValueError("V152 rejects another checkpoint/control profile")
