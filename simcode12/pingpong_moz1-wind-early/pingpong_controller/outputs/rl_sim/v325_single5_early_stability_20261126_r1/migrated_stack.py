"""Clean repo-relative stack loader for V325 on migrated hosts.

Preserves the exact frozen M04 mjx_juggle_env.py, XML, 76/76 source files,
V325 course[14] config, and verifies physical_cfg parity against u416 MANIFEST.
"""
from dataclasses import asdict, fields, is_dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import numpy as np

REPO = Path(__file__).resolve().parents[4]
RL_SIM_OUT = REPO / "pingpong_controller/outputs/rl_sim"
TOOLS_RL_SIM = REPO / "pingpong_controller/tools/rl_sim"
ORIG_RUN = RL_SIM_OUT / "v325_single5_windfeature_high_train_20261124_r1"
S15_POLISH = RL_SIM_OUT / "v325_s15_constant_polish_20261119_r1"
SINGLE_ACT = RL_SIM_OUT / "v325_single_actor_ablation_20261120_r1"
SOURCE = RL_SIM_OUT / "v325_structural_ablations_20261119_r1/source"
PROVISIONAL = RL_SIM_OUT / "v325_single5_provisional416_20261124_r1"
S15_CKPT = RL_SIM_OUT / "fplc_v325_gpu0_reference_center_20261118_r2/converged_s15_u00014400.pkl"
V172_RUN = RL_SIM_OUT / "fplc_v172_gpu1_quality_gates_formal_20261009_r1"
V172_SNAP = V172_RUN / "source_snapshot/pingpong_controller/tools/rl_sim"


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, default=lambda x:
        asdict(x) if is_dataclass(x) else np.asarray(x).tolist())


def _construct_dataclass(dc_type, payload: dict):
    default_obj = dc_type()
    valid = {f.name for f in fields(dc_type)}
    kwargs = {}
    for k, v in payload.items():
        if k not in valid:
            continue
        dv = getattr(default_obj, k)
        if is_dataclass(dv) and isinstance(v, dict):
            kwargs[k] = _construct_dataclass(type(dv), v)
        elif isinstance(dv, tuple) and isinstance(v, list):
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    return dc_type(**kwargs)


def setup_sys_path():
    # Put snapshot BEFORE tools/rl_sim so frozen M04 mjx_juggle_env.py is imported,
    # while unchanged helper modules and XML/meshes resolve from tools/rl_sim.
    for p in (TOOLS_RL_SIM, V172_SNAP, SOURCE, S15_POLISH, SINGLE_ACT, ORIG_RUN):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


def load_migrated_v325_stack(*, verify_source_manifest: bool = True) -> SimpleNamespace:
    setup_sys_path()
    if verify_source_manifest:
        sp = json.loads((RL_SIM_OUT / "v325_structural_ablations_20261119_r1/PLAN.json").read_text())
        for name, expected in sp["source_files"].items():
            if sha256_file(SOURCE / name) != expected:
                raise ValueError(f"V325 source file hash mismatch: {name}")

    from fixed_plane_v321_long_episode import PARENT_SHA256, XML_SHA256
    import fixed_plane_v321_long_episode as helpers
    import train_fixed_plane_v321 as v321_module
    from fixed_plane_v323_ball_obs_noise import spec_for as spec323_for, validate_parent_payload
    from train_fixed_plane_wind import registered_spec, course_config, restore_training_state

    v172_ckpt = V172_RUN / "10_fplc_v159_gpu1_s10_clean_commit.pkl"
    xml = TOOLS_RL_SIM / "moz1_pd_1ms_v98.xml"
    if sha256_file(v172_ckpt) != PARENT_SHA256:
        raise ValueError("M04 checkpoint hash mismatch")
    if sha256_file(xml) != XML_SHA256:
        raise ValueError("XML hash mismatch")

    ppo_spec = importlib.util.spec_from_file_location("train_juggle_mjx_ppo", SOURCE / "train_juggle_mjx_ppo.py")
    ppo = importlib.util.module_from_spec(ppo_spec)
    sys.modules[ppo_spec.name] = ppo
    ppo_spec.loader.exec_module(ppo)

    native = importlib.import_module("mjx_juggle_env")
    spec323 = spec323_for(323)
    if sha256_file(Path(native.__file__)) != spec323.native_env_sha256:
        raise ValueError(f"Wrong frozen M04 mjx_juggle_env.py: {native.__file__}")

    with v172_ckpt.open("rb") as f:
        ancestor = pickle.load(f)
    native_cfg = _construct_dataclass(native.MjxFixedPlaneV159RewardConfig, ancestor["env_cfg"])
    restored = asdict(native_cfg)
    for k, v in ancestor["env_cfg"].items():
        if k not in restored or canonical_json(v) != canonical_json(restored[k]):
            raise ValueError(f"M04 saved config mismatch: {k}")

    parent_path = REPO / spec323.parent_checkpoint
    with parent_path.open("rb") as f:
        parent = pickle.load(f)
    validate_parent_payload(parent, sha256_file(parent_path), spec323)
    saved_contract = parent["v321_contract"]

    for old_path, expected in saved_contract["replay_sha256"].items():
        rel = old_path.replace("/home/yangzhe/Project/pingpong_controller/", "")
        if sha256_file(REPO / rel) != expected:
            raise ValueError(f"Replay provenance hash mismatch: {rel}")

    native_cfg = helpers.make_config(native_cfg)
    if canonical_json(asdict(native_cfg)) != canonical_json(parent["env_cfg"]):
        raise ValueError("Native config mismatch vs V321 parent")

    spec325 = registered_spec(325)
    from fixed_plane_wind_env import make_wind_base
    from fixed_plane_wind_reward import reward_environment
    from fixed_plane_wind_flight_outcomes import flight_environment
    from fixed_plane_wind_reference_constraints import reference_environment
    from fixed_plane_wind_history import history_environment

    base = make_wind_base(native.MjxJuggleEnv, spec325)
    env_class = v321_module.environment_class(base)
    env_class = reward_environment(env_class)
    env_class = flight_environment(env_class)
    env_class = reference_environment(env_class)
    env_class = history_environment(env_class)

    stack = SimpleNamespace(
        spec=spec325,
        default_spec=spec325,
        ppo=ppo,
        parent=parent,
        cfg=None,
        native_cfg=native_cfg,
        xml=xml,
        snapshot=V172_SNAP,
        env_class=env_class,
        module=v321_module,
        helpers=helpers,
        parent_path=parent_path,
        saved_contract=saved_contract,
        repo_root=REPO,
    )
    if sha256_file(S15_CKPT) != "fd9def8f5fdb511e1543cbcca264e0021738ff07c6425e8d393cb1c240918e97":
        raise ValueError("S15 source checkpoint hash mismatch")
    with S15_CKPT.open("rb") as f:
        s15_payload = pickle.load(f)
    saved_boot = s15_payload.get("wind_contract", {}).get("actor_bootstrap")
    stack.actor_bootstrap = saved_boot
    stack.actor_bootstrap_offset = int(saved_boot["extra_optimizer_steps"]) if saved_boot else 0
    stack.actor_bootstrap_requires_probe = False
    stack.bootstrap_online_recipe = None
    stack.bootstrap_stage_anchor = None
    stack.bootstrap_anchor_stage = None
    stack.online_recipe_migration = None
    from fixed_plane_wind_online_recipe import configure_recipe, prepare_recipe_migration
    from fixed_plane_wind_clean_rehearsal import checkpoint_mix
    s15_payload = prepare_recipe_migration(stack, s15_payload, S15_CKPT)
    configure_recipe(stack, s15_payload)
    stack.wind_training_mix = checkpoint_mix(s15_payload, stack.spec.version)
    stack.s15_payload = s15_payload
    stack.cfg = course_config(stack, stack.spec.course[14])

    from single5_runtime import configure, apply_factory
    from feature_training import feature_apply_factory
    stack.cfg = configure(stack.cfg)
    stack.ppo.apply_mlp = feature_apply_factory(apply_factory(stack.ppo.apply_mlp, enabled=True))

    # Strict key-by-key parity check against u416 MANIFEST source_physical_cfg
    manifest = json.loads((PROVISIONAL / "MANIFEST.json").read_text())
    expected_cfg = manifest["source_physical_cfg"]
    actual_cfg = asdict(stack.cfg)
    diffs = []
    for k in sorted(set(expected_cfg) | set(actual_cfg)):
        if k not in actual_cfg:
            diffs.append(f"missing:{k}")
        elif k not in expected_cfg:
            diffs.append(f"extra:{k}")
        elif canonical_json(actual_cfg[k]) != canonical_json(expected_cfg[k]):
            diffs.append(f"diff:{k}")
    if diffs:
        raise ValueError(f"Physical config mismatch vs u416 MANIFEST: {diffs[:10]}")

    return stack
