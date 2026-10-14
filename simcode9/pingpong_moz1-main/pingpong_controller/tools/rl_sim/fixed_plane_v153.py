"""Released-policy continuation with consistent height admission and small DR."""
from dataclasses import replace
import math
from typing import Any

from fixed_plane_v151 import PRECISION_COST_FIELDS
from fixed_plane_v152 import _observation_dr, _plant_dr

PROFILES = tuple(f"fixed_plane_limit_cycle_qvel_rmp_gpu{g}_v153" for g in (0, 1))
HEIGHT_ERROR_GATE_M = .070
HEIGHT_QUALITY_FLOOR = .35
HEIGHT_SIGMA_M = HEIGHT_ERROR_GATE_M / math.sqrt(-2 * math.log(HEIGHT_QUALITY_FLOOR))
SOURCE_SHA256 = (
    "5b98879f9a6f16e61ce2f32a3c5ac2164e9433afc2050fc4293e5c2e25277ffe",
    "b34d33f66efe579bfc4c42d18226c1a05b3a273f0eec5b2722625b1c4d4b8c5f",
)
TAIL_LABELS = ("obs_bridge_2mm", "obs_small_4mm", "plant_quarter",
               "plant_half", "mass_small", "target_hold_small", "small_dr_consolidation")


def contact_quality_gate_status(stage: Any, result: dict[str, float]) -> dict[str, float]:
    """Fail closed on every declared quality floor in opt-in independent blocks."""
    enabled = bool(stage.advance_validate_contact_quality_components)
    floors = stage.min_recent_fixed_contact_quality_components
    checks = {}
    for index, name in enumerate(("z", "pose", "vxy", "omega", "offset")):
        value = float(result.get(f"advance_eval/fixed_contact_q_{name}", float("nan")))
        checks[f"advance_eval/fixed_contact_q_{name}_ok"] = float(
            not enabled or floors is None or
            (math.isfinite(value) and value >= float(floors[index])))
    return {**checks, "advance_eval/fixed_contact_quality_ok": float(all(checks.values()))}


def build_stages(source: list[Any], *, gpu: int, registered_plant: Any,
                 nominal_plant: Any) -> list[Any]:
    """Keep seven clean lessons; use seven graduated, bounded DR lessons."""
    if gpu not in (0, 1):
        raise ValueError("V153 requires physical GPU0 or GPU1 course identity")
    old_prefix = f"fplc_v{152 if gpu == 0 else 151}_gpu{gpu}_"
    result = []
    for index, stage in enumerate(source[:7]):
        name = stage.name.replace(old_prefix, f"fplc_v153_gpu{gpu}_", 1)
        cfg = replace(stage.cfg, fixed_contact_training_phase=name)
        gates = {}
        if index >= 3:
            from mjx_juggle_env import MjxFixedPlaneV153ContactConfig
            cfg = MjxFixedPlaneV153ContactConfig(**vars(cfg))
            cfg = replace(cfg, fixed_contact_quality_surface_z_sigma_m=HEIGHT_SIGMA_M,
                          fixed_contact_outcome_max_surface_z_error_m=HEIGHT_ERROR_GATE_M)
            gates = dict(
                max_recent_rapid_steady_surface_z_abs_error_m=HEIGHT_ERROR_GATE_M,
                min_recent_fixed_contact_quality_components=(
                    HEIGHT_QUALITY_FLOOR, .16,
                    *stage.min_recent_fixed_contact_quality_components[2:]),
            )
            if gpu == 1:
                # Source fractions are .10/.25/.50/1.0. Raise precision gradually.
                ratio = (.25 / .10, .40 / .25, .65 / .50, 1.0)[index - 3]
                cfg = replace(cfg, **{k: getattr(cfg, k) * ratio
                                      for k in PRECISION_COST_FIELDS},
                              hit_contact_center_excess_penalty_weight=(
                                  cfg.hit_contact_center_excess_penalty_weight * ratio),
                              hit_contact_center_excess_radius_m=(.005, .004, .003, .003)[index - 3])
                gates["max_recent_rapid_steady_racket_vxy_rms_m_s"] = (
                    .120, .100, .080, .065)[index - 3]
        result.append(replace(
            stage, name=name, cfg=cfg, **gates,
            advance_eval_deterministic=False,
            advance_validate_contact_quality_components=True,
            notes="V153 aligns70mm height admission, height-quality0.35 and outcome "
                  "height cost. SO3 admission0.16 retains hard level/full-angular gates. "
                  "GPU1 alone ramps precision25/40/65/100%; calibrated contact, actor, "
                  "bounded reference, RMP/PD and reset remain unchanged."))
    base = result[-1]
    cfg = base.cfg
    nominal_bounds = {k: v for k, v in vars(nominal_plant).items()
                      if k.startswith(("dr_pd_", "dr_rmp_")) and k.endswith("_range")}
    for number, label in enumerate(TAIL_LABELS, 8):
        if number in (8, 9):
            cfg = _observation_dr(cfg, .25 if number == 8 else .50)
        elif number in (10, 11):
            cfg = _plant_dr(replace(cfg, **nominal_bounds), registered_plant,
                            .25 if number == 10 else .50)
        elif number == 12:
            cfg = replace(cfg, dr_ball_mass_range=(.00395, .00405))
        elif number == 13:
            cfg = replace(cfg, dr_rmp_target_update_hold_probability=.025,
                          dr_rmp_target_update_hold_tail_probability=.0055)
        name = f"fplc_v153_gpu{gpu}_s{number}_{label}"
        result.append(replace(
            base, name=name, cfg=replace(cfg, fixed_contact_training_phase=name),
            max_recent_rapid_steady_racket_vxy_rms_m_s=(.105 if gpu == 0 else .080),
            notes="V153 small generalization within registered candidate bounds: "
                  "2/4mm observations, quarter/half plant ranges,3.95--4.05g mass, "
                  "2.5% pre-RMP target holds. Contact/spin/reset/dropout stay fixed. "
                  "All survival, height, center, view and safety gates remain active."))
    return result


def validate_resume_payload(payload: dict[str, Any], *, gpu: int,
                            requested_start_stage: str, resume_curriculum_state: bool,
                            reset_optimizer: bool, reset_critic: bool) -> None:
    """Require the pinned release or an exact-history own-profile continuation."""
    if gpu not in (0, 1):
        raise ValueError("V153 GPU identity is required")
    args = payload.get("args", {})
    source = args.get("curriculum_profile") if isinstance(args, dict) else getattr(args, "curriculum_profile", None)
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or payload.get("obs_dim") != (156 if gpu == 0 else 72)
            or payload.get("critic_obs_dim") != (477 if gpu == 0 else 393)
            or payload.get("act_dim") != 7 or not isinstance(payload.get("params"), dict)
            or not {"pi", "v"}.issubset(payload["params"])):
        raise ValueError("V153 requires the released actor/critic and full Adam state")
    parent = f"fixed_plane_limit_cycle_qvel_rmp_gpu{gpu}_v{152 if gpu == 0 else 151}"
    if source == parent:
        source_stage = ("fplc_v152_gpu0_s5_steady_quality" if gpu == 0
                        else "fplc_v151_gpu1_s4_fourth_entry")
        target = f"fplc_v153_gpu{gpu}_" + ("s8_obs_bridge_2mm" if gpu == 0 else "s5_steady_quality")
        if (payload.get("stage_name") != source_stage
                or payload.get("stage_update") != (2786 if gpu == 0 else 3187)
                or payload.get("step") != (2112880640 if gpu == 0 else 883949568)
                or payload.get("stop_reason") != "received SIGINT"
                or resume_curriculum_state or requested_start_stage != target):
            raise ValueError("V153 requires the pinned release and reviewed stage admission")
    elif source == PROFILES[gpu]:
        if (not resume_curriculum_state or payload.get("stage_name") != requested_start_stage
                or not requested_start_stage.startswith(f"fplc_v153_gpu{gpu}_")
                or int(payload.get("stage_update", -1)) < 1):
            raise ValueError("V153 own continuation requires exact stage and history")
    else:
        raise ValueError("V153 rejects another checkpoint/control profile")
