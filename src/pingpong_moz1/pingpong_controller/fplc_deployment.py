"""Versioned FPLC asset and real-workspace contracts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "config" / "fplc_model_contracts.json").is_file())
CONTRACT_PATH = REPO_ROOT / "config" / "fplc_model_contracts.json"
FPLC_CONTRACTS = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
SIMCODE10_CONTRACT_PATH = REPO_ROOT / "config" / "fplc_simcode10_model_contracts.json"
SIMCODE10_CONTRACTS = json.loads(SIMCODE10_CONTRACT_PATH.read_text(encoding="utf-8"))
if FPLC_CONTRACTS.keys() & SIMCODE10_CONTRACTS.keys():
    raise ValueError("simcode10 must not replace an existing FPLC model identity")
FPLC_CONTRACTS.update(SIMCODE10_CONTRACTS)
SIMCODE10_PROFILES = tuple(SIMCODE10_CONTRACTS)
SIMCODE11_CONTRACT_PATH = REPO_ROOT / "config" / "fplc_simcode11_model_contracts.json"
SIMCODE11_CONTRACTS = json.loads(SIMCODE11_CONTRACT_PATH.read_text(encoding="utf-8"))
SIMCODE11_U416_CONTRACT_PATH = REPO_ROOT / "config" / "fplc_simcode11_u416_model_contracts.json"
SIMCODE11_U416_CONTRACTS = json.loads(SIMCODE11_U416_CONTRACT_PATH.read_text(encoding="utf-8"))
if SIMCODE11_CONTRACTS.keys() & SIMCODE11_U416_CONTRACTS.keys():
    raise ValueError("u416 must not replace an existing simcode11 model identity")
SIMCODE11_CONTRACTS.update(SIMCODE11_U416_CONTRACTS)
if FPLC_CONTRACTS.keys() & SIMCODE11_CONTRACTS.keys():
    raise ValueError("simcode11 must not replace an existing FPLC model identity")
FPLC_CONTRACTS.update(SIMCODE11_CONTRACTS)
SIMCODE11_PROFILES = tuple(SIMCODE11_CONTRACTS)
SIMCODE12_CONTRACTS = {}
for filename in ("fplc_simcode12_model_contracts.json",
                 "fplc_simcode12_u7447_model_contracts.json",
                 "fplc_simcode12_u7832_model_contracts.json"):
    contracts = json.loads((REPO_ROOT / "config" / filename).read_text(encoding="utf-8"))
    if FPLC_CONTRACTS.keys() & contracts.keys():
        raise ValueError(f"{filename} must not replace an existing FPLC model identity")
    FPLC_CONTRACTS.update(contracts)
    SIMCODE12_CONTRACTS.update(contracts)
SIMCODE12_PROFILES = tuple(SIMCODE12_CONTRACTS)
FPLC_PROFILES = tuple(FPLC_CONTRACTS)
DEFAULT_OBS_WIND_MAX_M_S = 1.6
# Authenticated training bounds; the actor's projection keeps these unchanged.
FPLC_LOWER_M = (-0.348, -0.700, 1.000)
FPLC_UPPER_M = (0.348, -0.280, 1.150)
# Operator-requested real envelope, independent of the training projection.
# Explicit physical limits may tighten it; reference orientation is unchanged.
FPLC_REAL_LOWER_M = (-0.150, -0.600, 0.950)
FPLC_REAL_UPPER_M = (0.150, -0.350, 1.200)
FPLC_REAL_ROLL_PITCH_LIMIT_DEG = (4.0, 4.0)


# Running feedback tolerates a bounded receiver delay without waiting away a
# control period. Arming/reset/stop-hold retain their separate 10 ms checks.
def joint_feedback_max_age_s(value: str | float = "15") -> float:
    try:
        milliseconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("FPLC_JOINT_FEEDBACK_MAX_AGE_MS must be between 10 and 20") from exc
    if not math.isfinite(milliseconds) or not 10.0 <= milliseconds <= 20.0:
        raise ValueError("FPLC_JOINT_FEEDBACK_MAX_AGE_MS must be between 10 and 20")
    return milliseconds / 1000.0


FPLC_JOINT_FEEDBACK_MAX_AGE_S = joint_feedback_max_age_s(
    os.environ.get("FPLC_JOINT_FEEDBACK_MAX_AGE_MS", "15"))
FPLC_JOINT_FEEDBACK_WAIT_S = 0.005
FPLC_JOINT_FEEDBACK_MAX_GAP_S = (
    FPLC_JOINT_FEEDBACK_MAX_AGE_S + FPLC_JOINT_FEEDBACK_WAIT_S)


def fplc_assets(profile: str, repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    contract = FPLC_CONTRACTS[profile]
    return (repo_root / contract["checkpoint_path"],
            repo_root / contract["policy_xml_path"])


def fplc_obs_wind_max_m_s(profile: str) -> float:
    return float(FPLC_CONTRACTS.get(profile, {}).get(
        'obs_wind_max_m_s', DEFAULT_OBS_WIND_MAX_M_S))


def prepare_fplc_checkpoint_payload(payload: dict, model_path: str) -> dict:
    """Restore single5 checkpoint metadata from authenticated frozen manifests.

    Shared by the actor and CPU mirror, so neither silently uses plant defaults.
    The training optimizer is never imported or executed.
    """
    runtime = payload.get('requires_actor_runtime')
    if runtime is None:
        return payload
    from pingpong_controller.policy_release_manifest import sha256_file
    checkpoint_hash = sha256_file(Path(model_path))
    matches = [c for c in FPLC_CONTRACTS.values()
               if c.get('required_actor_runtime') == runtime
               and c['sha256'] == checkpoint_hash]
    if len(matches) != 1:
        raise RuntimeError('Unregistered FPLC actor runtime/checkpoint identity')
    contract = matches[0]
    path = REPO_ROOT / contract['checkpoint_metadata_path']
    if sha256_file(path) != contract['checkpoint_metadata_sha256']:
        raise RuntimeError('FPLC checkpoint metadata SHA-256 mismatch')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    dimensions = [contract['actor_dim'], contract['critic_dim'], contract['action_dim']]
    architecture = manifest.get('architecture', manifest)
    if (payload.get('schema') != contract.get('checkpoint_schema', 'v325_single5_training_v1')
            or payload.get('mode') != 'direct'
            or payload.get('actual_updates') != contract['global_update']
            or manifest['actual_update'] != contract['global_update']
            or payload.get('native_actor_obs_dim') != dimensions[0]
            or payload.get('single_actor_input_dim') != dimensions[0]
            or architecture['actor_critic_action'] != dimensions
            or architecture['residual_network'] is not False):
        raise RuntimeError('FPLC single5 schema/runtime mismatch')
    if 'physical_metadata_path' in contract:
        # The wind-early stack checks the nominal config against this manifest
        # before its training-only wind, physics and observation perturbations.
        physical_path = REPO_ROOT / contract['physical_metadata_path']
        if sha256_file(physical_path) != contract['physical_metadata_sha256']:
            raise RuntimeError('FPLC physical metadata SHA-256 mismatch')
        if (architecture['requires_actor_runtime'] != runtime
                or manifest['files']['model.pkl']['sha256'] != checkpoint_hash
                or payload.get('step') != manifest['true_step']
                or payload.get('stage') != manifest['stage']
                or payload.get('plan_sha256') != contract['checkpoint_plan_sha256']):
            raise RuntimeError('FPLC wind-early checkpoint provenance mismatch')
        s2r_stage = manifest.get('s2r_curriculum_stage')
        if s2r_stage is not None and s2r_stage != {
            'id': payload.get('s2r_stage_id'),
            'name': payload.get('s2r_stage_name'),
            'warm_w': payload.get('s2r_warm_w'),
            'dr_scale': payload.get('s2r_dr_scale'),
        }:
            raise RuntimeError('FPLC Sim2Real curriculum stage mismatch')
        high_wind_stage = manifest.get('high_wind_curriculum_stage')
        if high_wind_stage is not None and (
            payload.get('source_sha256') != contract['parent_sha256']
            or payload.get('obs_wind_max_m_s') != architecture['obs_wind_max_m_s']
            or payload.get('obs_wind_max_m_s') != contract['obs_wind_max_m_s']
            or payload.get('pose_limit_deg') != manifest['control_and_physics']['RP_limit_deg']
            or payload.get('pose_limit_deg') != contract['fixed_pose_override_deg']
            or payload.get('active_max_speed_m_s') != high_wind_stage['active_max_speed_m_s']
            or payload.get('s2r_dr_scale') != high_wind_stage['dr_scale']
            or payload.get('last_summary', {}).get('stage_idx') != high_wind_stage['id']
            or architecture['wind_feature_deadband_m_s'] != contract['wind_feature_deadband_m_s']
            or architecture['wind_feature_activation_width_m_s'] != contract['wind_feature_activation_width_m_s']
        ):
            raise RuntimeError('FPLC high-wind checkpoint configuration mismatch')
        manifest = json.loads(physical_path.read_text(encoding='utf-8'))
    cfg = {**manifest['source_physical_cfg'], **contract.get('physical_config_overrides', {})}
    return {**payload, 'env_cfg': cfg,
            **dict(zip(('obs_dim', 'critic_obs_dim', 'act_dim'), dimensions))}


def fplc_real_roll_pitch_limit_deg(profile: str) -> tuple[float, float]:
    """Read the selected model's real envelope; older models retain 4 degrees."""
    limits = FPLC_CONTRACTS[profile].get(
        'real_roll_pitch_limit_deg', FPLC_REAL_ROLL_PITCH_LIMIT_DEG)
    if (len(limits) != 2
            or any(not math.isfinite(float(v)) or float(v) <= 0 for v in limits)):
        raise ValueError('FPLC real roll/pitch limits must be two finite positive angles')
    return tuple(map(float, limits))


def fplc_real_workspace_limits(profile: str | None = None):
    """Return the selected model's real box in the policy Cartesian frame."""
    contract = {} if profile is None else FPLC_CONTRACTS[profile]
    mode = contract.get('real_workspace_mode', 'operator')
    margin = 0.0
    if mode == 'training':
        cfg = contract['env_contract']
        lower = cfg['cartesian_pose_position_lower_m']
        upper = cfg['cartesian_pose_position_upper_m']
        margin = float(contract.get('real_workspace_margin_m', 0.0))
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError('FPLC real workspace margin must be finite and nonnegative')
    elif mode == 'operator':
        lower, upper = FPLC_REAL_LOWER_M, FPLC_REAL_UPPER_M
    else:
        raise ValueError(f'Unsupported FPLC real workspace mode: {mode!r}')
    if (len(lower) != 3 or len(upper) != 3
            or any(not math.isfinite(float(v)) for v in (*lower, *upper))
            or any(float(low) >= float(high) for low, high in zip(lower, upper))):
        raise ValueError('FPLC real workspace must contain three finite increasing pairs')
    # Normalize decimal metre bounds after adding the configured outer margin.
    return tuple((round(float(low) - margin, 12), round(float(high) + margin, 12))
                 for low, high in zip(lower, upper))


def fplc_workspace_for_profile(profile: str, configured_limits):
    """Select a model box independently of the previous/general profile box.

    Training-derived releases use the model box plus its explicit outer margin;
    general YAML/CLI bounds remain available when switching to older models.
    """
    if FPLC_CONTRACTS[profile].get('real_workspace_mode', 'operator') == 'training':
        return fplc_real_workspace_limits(profile)
    return fplc_workspace_intersection(configured_limits, profile=profile)


def fplc_workspace_intersection(limits, *, profile: str | None = None):
    """Intersect explicit limits with the selected model's real envelope."""
    envelope = fplc_real_workspace_limits(profile)
    if len(limits) != 3:
        raise ValueError("FPLC requires exactly three real workspace limits")
    output = []
    for axis, limit in enumerate(limits):
        if limit is None or len(limit) != 2:
            raise ValueError("FPLC requires all three real workspace limits")
        low, high = map(float, limit)
        if not all(math.isfinite(v) for v in (low, high)) or low >= high:
            raise ValueError("FPLC workspace limits must be finite increasing pairs")
        low = max(low, envelope[axis][0])
        high = min(high, envelope[axis][1])
        if low >= high:
            raise ValueError("real workspace has no intersection with the FPLC workspace")
        output.append((low, high))
    return tuple(output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=FPLC_PROFILES)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    for axis in "xyz":
        parser.add_argument(f"--{axis}-limits")
    args = parser.parse_args()
    model, xml = fplc_assets(args.profile, args.repo_root)
    print(f"FPLC_MODEL_PATH\t{model}")
    print(f"FPLC_POLICY_XML_PATH\t{xml}")
    print(f"FPLC_OBS_DIM\t{FPLC_CONTRACTS[args.profile]['actor_dim']}")
    print(f"FPLC_ACTOR_AGE_MODE\t{FPLC_CONTRACTS[args.profile].get('actor_ball_age_mode', 'measurement_age')}")
    training = FPLC_CONTRACTS[args.profile]['env_contract']
    training_rp = [training[key] for key in ('cartesian_pose_roll_limit_deg', 'cartesian_pose_pitch_limit_deg')]
    print(f"FPLC_TRAINING_ROLL_PITCH_LIMITS_DEG\t{','.join(map(str, training_rp))}")
    real_rp = fplc_real_roll_pitch_limit_deg(args.profile)
    print(f"FPLC_REAL_ROLL_PITCH_LIMITS_DEG\t{','.join(map(str, real_rp))}")
    if args.x_limits is not None:
        pairs = [tuple(map(float, value.split(','))) for value in (args.x_limits, args.y_limits, args.z_limits)]
        for axis, pair in zip("XYZ", fplc_workspace_for_profile(args.profile, pairs)):
            print(f"DONE_EE_{axis}_LIMITS_M\t{pair[0]},{pair[1]}")
