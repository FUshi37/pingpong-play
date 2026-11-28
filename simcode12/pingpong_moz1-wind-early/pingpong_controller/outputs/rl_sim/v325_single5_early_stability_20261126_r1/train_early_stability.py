"""Migrated V325 single-MLP early-stability continuation from u416 on RTX 4090.

Preserves exact V325 physics/control/reward/observation/PPO contract and frozen
rows 0:118 backbone, while adding stratified 20s launch & early-juggling rehearsal
across all 27 wind cells so PPO continuously trains on the 3-20s onset failure zone.
"""
import argparse
import ctypes
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np

from migrated_stack import (
    load_migrated_v325_stack, setup_sys_path, sha256_file,
    REPO, PROVISIONAL,
)
setup_sys_path()

from single5_runtime import adam_factory, install_pose_mode, Single5Env, actor_parameter_count, REVISION
from learning_stability import critic_warmup_adam, cap_actor_update
from train_s15_fresh_velocity_r31 import ActualContactEnv, CONTACT_FIELDS
from capture_margin import CaptureEnv
from xwind_sampling import XWindEnv
from randomized_onset import TIMING_FIELDS, PREVIEW_FIELDS
from targeted_reward import FIELDS as TARGETED_FIELDS
from fixed_plane_v321_long_episode import QUALITY_FIELDS
from fixed_plane_wind_flight_outcomes import HEIGHT_FIELDS
from fixed_plane_wind_reference_constraints import REFERENCE_FIELDS, make_reference_gate
from fixed_plane_wind_strong import quality_gate
from fixed_plane_wind_bootstrap_restore import learner_settings_for_stack
from train_fixed_plane_wind import make_train_functions, strong_arrays, numeric_digest
from bounded_validation_memory import release_caches
from adaptive_progress import AdaptiveProgress as FullEpisode99Progress, gate99
from legacy_wind_console import LegacyWindConsole
from critic_diagnostics import make_critic_diagnostics
from training_snapshot import save_snapshot, load_snapshot, prune_owned_snapshots, state_digest
from termination_diagnostics import TERMINATION_FIELDS
from precision_maturity import PRECISION_FIELDS, verify_mature_batch
from weak_release_floor import FIELDS as RELEASE_FLOOR_FIELDS
from feature_training import frozen_base_equal, assert_supported_actor
from wind_features import WindFeatureEnv
from stratified_rehearsal import (
    StratifiedStartupRehearsalEnv, StratifiedStartupProgress, EarlyHitQualityTracker,
    masked_rollout, FIELDS as REHEARSAL_FIELDS,
)
from early_stability_reward import EarlyStabilityRewardEnv, EARLY_REWARD_FIELDS
from wide_onset import StratifiedWideOnsetEnv


def save_json(path: Path, value: dict):
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


class BestAndLatestRetention:
    """Periodically prune periodic checkpoints, keeping only best and latest."""

    def __init__(self, directory: Path, cleanup_every: int = 5):
        self.directory = Path(directory)
        self.cleanup_every = max(1, int(cleanup_every))
        self.best_score = None
        self.best_path = None
        self.latest_path = None
        self._since_cleanup = 0

    def register(self, path: Path, score):
        path = Path(path)
        self.latest_path = path
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_path = path
        self._since_cleanup += 1
        if self._since_cleanup >= self.cleanup_every:
            self.prune()
            self._since_cleanup = 0

    def prune(self):
        keep = {p.resolve() for p in (self.best_path, self.latest_path) if p is not None}
        for ckpt in sorted(self.directory.glob("periodic_u*.pkl")):
            if ckpt.resolve() not in keep:
                ckpt.unlink(missing_ok=True)


def load_u416_provisional(env, stack, *, load_runner: bool = True):
    """Verify u416 MANIFEST hashes and restore learner (+ 1024-env runner when requested)."""
    manifest = json.loads((PROVISIONAL / "MANIFEST.json").read_text())
    model_path = PROVISIONAL / "model.pkl"
    zip_path = PROVISIONAL / "trainer_state.zip"
    assert sha256_file(model_path) == manifest["files"]["model.pkl"]["sha256"], "u416 model.pkl hash mismatch"
    assert sha256_file(zip_path) == manifest["files"]["trainer_state.zip"]["sha256"], "u416 trainer_state.zip hash mismatch"

    with model_path.open("rb") as f:
        saved = pickle.load(f)
    assert saved["schema"] == "v325_single5_training_v1"
    assert saved["actual_updates"] == 416
    assert saved["params"]["pi"]["l1"]["w"].shape == (120, 282)
    assert_supported_actor(saved["params"])

    learner_template = stack.ppo.TrainState(
        jax.tree.map(jnp.asarray, saved["params"]),
        jax.tree.map(jnp.asarray, saved["opt"]),
    )

    if not load_runner:
        return (
            learner_template,
            None,
            int(saved["actual_updates"]),
            int(saved["step"]),
            saved["initial_single_actor_digest"],
        )

    inner_env = env.inner
    init_rng, key = jax.random.split(jax.random.PRNGKey(2026112101))
    keys = jax.random.split(key, inner_env.n_envs)
    st0, ob0 = jax.eval_shape(inner_env.reset, keys)
    co0 = jax.eval_shape(inner_env.get_critic_obs, st0, ob0)
    runner_template = stack.ppo.RunnerState(
        st0, ob0, co0, init_rng,
        jax.ShapeDtypeStruct((inner_env.n_envs,), jnp.float32),
        jax.ShapeDtypeStruct((inner_env.n_envs,), jnp.int32),
    )

    with zipfile.ZipFile(zip_path) as zf:
        zmeta = json.loads(zf.read("manifest.json"))
        assert zmeta["schema"] == "v325_full_training_state_v1"
        assert zmeta["metadata"]["actual_updates"] == 416

        def restore_tree(name, template):
            pairs, treedef = jax.tree_util.tree_flatten_with_path(template)
            expected_specs = zmeta["trees"][name]
            assert len(pairs) == len(expected_specs), f"{name} tree length mismatch"
            leaves = []
            for idx, ((path, leaf), spec) in enumerate(zip(pairs, expected_specs)):
                assert jax.tree_util.keystr(path) == spec["path"], f"{name}[{idx}] path mismatch"
                with zf.open(f"{name}_{idx:05d}.npy") as s:
                    arr = np.lib.format.read_array(s, allow_pickle=False)
                assert list(arr.shape) == spec["shape"] and arr.dtype.str == spec["dtype"]
                leaves.append(jnp.asarray(arr))
            return jax.tree_util.tree_unflatten(treedef, leaves)

        restored_learner = restore_tree("learner", learner_template)
        restored_inner_runner = restore_tree("runner", runner_template)

    assert numeric_digest(restored_learner) == numeric_digest(learner_template), "u416 model.pkl vs zip learner mismatch"
    return (
        restored_learner,
        restored_inner_runner,
        int(zmeta["metadata"]["actual_updates"]),
        int(zmeta["metadata"]["true_step"]),
        saved["initial_single_actor_digest"],
    )


def build_train_functions(stack, env, args, plan):
    """Wrap native PPO make_train_fns with exact V325 feature optimizer + stability guard."""
    original = stack.ppo.make_train_fns
    captured = {}
    effective = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return None, None

    stack.ppo.make_train_fns = capture
    try:
        make_train_functions(stack, env, args)
    finally:
        stack.ppo.make_train_fns = original

    extra_fields = (
        "v321/finite", "wind/mode", "wind/positive_y_violation", "wind/speed_m_s",
        "terminated", "truncated", "new_hit", "hit_count",
        "polish/background_speed_m_s", "polish/wind_sector", "polish/peak_assigned",
        "polish/physical_contact_frames", "polish/edge_contact_frames",
        "polish/contact_face_deviation_max_deg",
        "observed_force/valid", "observed_force/age_s", "observed_force/error_sq_m2_s2",
        "single5/wind_angle_rad", "pose/outer_limit_deg",
        "hit_racket_vxy_sq_sum",
        "reward/dynamics_outgoing", "reward/single5_outgoing_replace",
        "reward/single5_apex_precision", "reward/single5_center", "reward/total",
    )
    goal_fields = tuple("reward/goal_" + k for k in ("alive", "hit", "outgoing", "normal", "height_margin", "braking", "apex", "failure"))
    all_metrics = (
        tuple(captured["full_rollout_metric_names"])
        + QUALITY_FIELDS + HEIGHT_FIELDS + REFERENCE_FIELDS
        + CONTACT_FIELDS + TARGETED_FIELDS + TIMING_FIELDS + PREVIEW_FIELDS
        + TERMINATION_FIELDS + PRECISION_FIELDS + RELEASE_FLOOR_FIELDS
        + REHEARSAL_FIELDS + EARLY_REWARD_FIELDS + goal_fields + extra_fields
    )
    captured["full_rollout_metric_names"] = tuple(dict.fromkeys(all_metrics))
    captured["time_limit_bootstrap"] = True
    captured["apply_every_minibatch"] = False
    captured["residual_l2_coef"] = 0.0
    captured["learning_rate"] = plan["ppo"]["learning_rate"]
    captured["clip_range"] = plan["ppo"]["clip_range"]
    captured["target_kl"] = plan["ppo"]["target_kl"]
    captured["gamma"] = plan["ppo"]["gamma"]
    captured["gae_lambda"] = plan["ppo"]["gae_lambda"]
    captured["vf_coef"] = plan["ppo"]["vf_coef"]
    captured["ent_coef"] = plan["ppo"]["ent_coef"]
    captured["max_grad_norm"] = plan["ppo"]["max_grad_norm"]
    captured["min_log_std"] = plan["ppo"]["min_log_std"]
    captured["max_log_std"] = plan["ppo"]["max_log_std"]
    captured["update_epochs"] = plan["ppo"]["update_epochs"]
    captured["minibatch_size"] = plan["ppo"]["minibatch_size"]
    captured["actor_anchor_kl_coef"] = plan["ppo"]["actor_anchor_kl_coef"]
    captured["actor_anchor_replay_obs"] = None
    captured["actor_anchor_replay_kl_coef"] = 0.0
    captured["actor_anchor_replay_max_kl"] = 0.0
    captured["counterfactual_replay_obs"] = None
    captured["counterfactual_replay_actions"] = None
    captured["counterfactual_supervision_coef"] = 0.0
    captured["reference_params"] = stack.single_actor_anchor_params

    effective.update({k: v for k, v in captured.items() if isinstance(v, (str, int, float, bool, tuple))})

    current = stack.ppo.adam_step
    epsilon = getattr(current, "_wind_history_xy_epsilon", 1e-5)
    multiplier = getattr(current, "_wind_history_xy_lr_multiplier", 1.0)
    base = getattr(current, "_wind_history_optimizer_base", current)
    opt_plan = plan["feature_optimizer"]
    stack.ppo.adam_step = adam_factory(
        base,
        history_xy_epsilon=epsilon,
        history_xy_lr_multiplier=multiplier,
        feature_lr_multiplier=opt_plan["feature_row_learning_rate_multiplier"],
        warmup_minibatches=opt_plan["warmup_accepted_minibatches"],
        independent_actor_critic_clipping=opt_plan["independent_actor_critic_clipping"],
        permanent_wind_rows_only=opt_plan["permanent_wind_rows_only"],
    )
    stab = plan["learning_stability"]
    warmup_start_t = int(stab["critic_warmup_start_t"])
    stack.ppo.adam_step = critic_warmup_adam(
        stack.ppo.adam_step,
        start_t=warmup_start_t,
        minibatches=int(stab["critic_only_minibatches"]),
    )
    effective["critic_warmup_start_t"] = warmup_start_t
    effective["critic_only_minibatches"] = int(stab["critic_only_minibatches"])
    effective["permanent_wind_rows_only"] = stack.ppo.adam_step._permanent_wind_rows_only
    effective["feature_lr_multiplier"] = stack.ppo.adam_step._feature_lr_multiplier

    collect, update = original(**captured)
    update = cap_actor_update(
        update,
        stack.ppo.policy_mean,
        maximum_kl=stab["maximum_sampled_cohort_step_kl"],
        sample_size=stab["step_observations"],
    )
    return collect, update, effective


def log_failure_events(stream, update, metrics, done):
    selected = np.asarray(done, bool) | (np.asarray(metrics["timing/long_confirmed_gap_event"]) > 0.5)
    fields = TIMING_FIELDS + PREVIEW_FIELDS + TERMINATION_FIELDS + (
        "terminated", "truncated", "single5/wind_angle_rad",
        "polish/background_speed_m_s", "wind/speed_m_s", "wind/mode",
        "sampling/rehearsal_lane", "sampling/rehearsal_cutoff",
    )
    for t, lane in zip(*np.nonzero(selected)):
        censor = float(metrics["sampling/rehearsal_cutoff"][t, lane]) > 0.5
        if censor:
            continue
        rec = {name: float(metrics[name][t, lane]) for name in fields if name in metrics}
        rec.update(
            update=update,
            lane=int(lane),
            event="episode_end" if done[t, lane] else "long_confirmation_gap",
            rehearsal_lane=bool(metrics["sampling/rehearsal_lane"][t, lane] > 0.5),
            recognized_termination=any(rec.get(tag, 0.0) > 0.5 for tag in TERMINATION_FIELDS),
        )
        stream.write(json.dumps(rec) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=HERE / "TRAIN_PLAN_EARLY.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--update-limit", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--resume-model", type=Path, default=None)
    parser.add_argument("--resume-full", type=Path, default=None)
    opts = parser.parse_args()

    plan = json.loads(opts.plan.read_text())
    plan_sha = sha256_file(opts.plan)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == opts.gpu_uuid and opts.gpu_uuid in plan["allowed_gpu_uuids"]
    assert jax.default_backend() == "gpu" and len(jax.devices()) == 1
    assert bool(opts.resume_model) == bool(opts.resume_full)

    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
        def _reloc(val):
            if isinstance(val, (str, Path)) and "/home/yangzhe/Project/pingpong_controller/" in str(val):
                new_s = str(REPO / str(val).replace("/home/yangzhe/Project/pingpong_controller/", ""))
                return Path(new_s) if isinstance(val, Path) else new_s
            return val
        for k, v in list(stack.parent["args"].items()):
            stack.parent["args"][k] = _reloc(v)
        args = argparse.Namespace(**stack.parent["args"])
        vars(args).update({k: _reloc(v) for k, v in learner_settings_for_stack(stack).items()})
        n_envs = int(plan.get("n_envs", 1024))
        n_steps = int(plan.get("n_steps", 128))
        minibatch_size = int(plan["ppo"].get("minibatch_size", 16384))
        batch_size = n_envs * n_steps
        max_minibatches = int(plan["ppo"].get("update_epochs", 4)) * max(1, batch_size // minibatch_size)
        args.n_envs, args.n_steps, args.minibatch_size = n_envs, n_steps, minibatch_size
        args.seed = plan["seed"]
        args.wind_training_mix = stack.wind_training_mix

    install_pose_mode()
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=stack.cfg)
    env = XWindEnv(
        CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
        half_angle_deg=plan["wind_domain"]["x_axis_half_angle_deg"],
        enabled=True,
        calm_only=False,
    )
    env = Single5Env(
        env,
        reward_enabled=False,
        goal_reward_enabled=plan["reward_revision"]["goal_reward_enabled"],
        random_onset_enabled=True,
        targeted_reward_enabled=plan["reward_revision"]["targeted_reward_enabled"],
        weak_release_floor_enabled=plan["reward_revision"]["weak_calm_release_floor"]["enabled"],
        weak_release_floor_weight=plan["reward_revision"]["weak_calm_release_floor"]["weight"],
    )
    if "targeted_variant" in plan["reward_revision"]:
        env.targeted_variant = str(plan["reward_revision"]["targeted_variant"])
    if "failure_reserve" in plan["reward_revision"]:
        env.targeted_failure_reserve = float(plan["reward_revision"]["failure_reserve"])
    onset_cfg = plan.get("onset", {})
    if onset_cfg.get("mode") == "stratified_wide":
        from stratified_rehearsal import stratified_rehearsal_mask
        _reh_mask = stratified_rehearsal_mask(n_envs, fraction=plan["stratified_rehearsal"]["fraction"])
        env.inner = StratifiedWideOnsetEnv(env.inner.inner, rehearsal_mask=_reh_mask, onset_cfg=onset_cfg, enabled=True)
        print(
            f"[wind-early] Enabled StratifiedWideOnsetEnv: early={onset_cfg.get('early_prob',0.2):.0%}"
            f"[{onset_cfg.get('delay_steps_min',5)}..{onset_cfg.get('delay_steps_max',100)}], "
            f"mid=[{onset_cfg.get('mid_delay_steps_min',500)}..{onset_cfg.get('mid_delay_steps_max',2000)}], "
            f"late(long)=[{onset_cfg.get('late_delay_steps_min',2000)}..{onset_cfg.get('late_delay_steps_max',15000)}]",
            flush=True,
        )
    env = WindFeatureEnv(env)
    early_rew_cfg = plan.get("early_stability_reward", {})
    env = EarlyStabilityRewardEnv(env, early_rew_cfg)
    rehearsal_cfg = plan["stratified_rehearsal"]
    env = StratifiedStartupRehearsalEnv(
        env,
        fraction=rehearsal_cfg["fraction"],
        seconds=rehearsal_cfg["seconds"],
        stagger_min_steps=rehearsal_cfg["stagger_min_steps"],
        stagger_seed=rehearsal_cfg["stagger_seed"],
    )
    assert (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) == (120, 453, 7, 240000)
    assert env.cfg.wind_config.max_speed_m_s == 1.6 and env.cfg.wind_config.wind_y_nonpositive

    perm_wind_only = bool(plan["feature_optimizer"]["permanent_wind_rows_only"])
    u416_learner, u416_inner_runner, completed, true_step, init_actor_digest = load_u416_provisional(
        env, stack, load_runner=(n_envs == 1024 and not opts.resume_model)
    )
    if plan.get("parent_checkpoint") and not opts.resume_model:
        parent_ckpt_path = REPO / plan["parent_checkpoint"]
        with parent_ckpt_path.open("rb") as f:
            p_saved = pickle.load(f)
        assert_supported_actor(p_saved["params"])
        if perm_wind_only or plan["feature_optimizer"].get("verify_parent_frozen_base", True):
            assert bool(frozen_base_equal(jax.tree.map(jnp.asarray, p_saved["params"]), u416_learner.params)), "Parent checkpoint altered frozen 0:118 backbone!"
        p_params = jax.tree.map(jnp.asarray, p_saved["params"])
        p_opt = jax.tree.map(jnp.asarray, p_saved["opt"])
        if plan["feature_optimizer"].get("reset_actor_moments_on_start", False):
            zero_pi = jax.tree.map(jnp.zeros_like, p_params["pi"])
            p_opt = p_opt._replace(
                m={**p_opt.m, "pi": zero_pi},
                v={**p_opt.v, "pi": zero_pi},
                actor_t=jnp.asarray(0, dtype=jnp.int32),
            )
            print("[wind-early] Reset actor Adam moments (m/v['pi']=0, actor_t=0) for clean full-actor unfreeze; critic moments & opt.t preserved.", flush=True)
        u416_learner = stack.ppo.TrainState(p_params, p_opt)
        completed = int(p_saved["actual_updates"])
        true_step = int(p_saved["step"])
        print(f"[wind-early] Loaded parent_checkpoint {parent_ckpt_path.name} (u{completed}, step={true_step}, opt.t={int(u416_learner.opt.t)}, actor_t={int(u416_learner.opt.actor_t)})", flush=True)
    initial_learner_params = jax.tree.map(jnp.array, u416_learner.params)
    stack.single_actor_anchor_params = initial_learner_params
    stack.bootstrap_stage_anchor = initial_learner_params

    native_components = stack.ppo.policy_components
    direct_trace_calls = [0]

    def direct_components(params, obs):
        assert_supported_actor(params)
        direct_trace_calls[0] += 1
        return native_components(params, obs)

    stack.ppo.policy_components = direct_components

    fields = tuple(dict.fromkeys(QUALITY_FIELDS + HEIGHT_FIELDS + REFERENCE_FIELDS))
    progress = FullEpisode99Progress(env.n_long, fields)
    progress.half_angle_deg = plan["wind_domain"]["x_axis_half_angle_deg"]
    startup_tracker = StratifiedStartupProgress(
        seconds=rehearsal_cfg["seconds"],
        capacity=rehearsal_cfg["metric_capacity"],
        half_angle_deg=plan["wind_domain"]["x_axis_half_angle_deg"],
    )
    console = LegacyWindConsole(n_envs, fields, 180.0)
    early_quality = EarlyHitQualityTracker(window_updates=16)
    restored_hold = 0

    contract = dict(
        schema="v325_single5_early_stability_v1",
        mode="direct",
        source_u416_model_sha256="97c89251fe8a54b193f7a9348799cf3fab58b487322a5cdfc27cc51fe2a49dce",
        source_u416_state_sha256="788975ba322906a463af28cf75e79539d6daafbba5474a9080284fee90d731f1",
        initial_single_actor_digest=init_actor_digest,
        actor_critic_action=[120, 453, 7],
        seed=plan["seed"],
        n_envs=n_envs,
        n_steps=n_steps,
        minibatch_size=minibatch_size,
        early_stability_reward=early_rew_cfg,
        stage=15,
        plan_sha256=plan_sha,
        pose_limit_deg=5.0,
        revision=REVISION,
        feature_optimizer=plan["feature_optimizer"],
        learning_stability=plan["learning_stability"],
        stratified_rehearsal=rehearsal_cfg,
        n_rehearsal=env.n_rehearsal,
        n_long=env.n_long,
        reward_revision=plan["reward_revision"],
        effective_targeted_variant=env.inner.targeted_variant if hasattr(env.inner, "targeted_variant") else getattr(env, "targeted_variant", None),
        effective_failure_reserve=float(getattr(env, "targeted_failure_reserve", 0.0)),
        wind_domain=plan["wind_domain"],
        onset=plan["onset"],
    )

    if opts.resume_model:
        with opts.resume_model.open("rb") as f:
            resumed = pickle.load(f)
        assert resumed["schema"] == "v325_single5_early_stability_v1"
        learner = stack.ppo.TrainState(
            jax.tree.map(jnp.asarray, resumed["params"]),
            jax.tree.map(jnp.asarray, resumed["opt"]),
        )
        init_rng, key = jax.random.split(jax.random.PRNGKey(plan["seed"]))
        keys = jax.random.split(key, n_envs)
        st0, ob0 = jax.eval_shape(env.reset, keys)
        co0 = jax.eval_shape(env.get_critic_obs, st0, ob0)
        r_tmpl = stack.ppo.RunnerState(
            st0, ob0, co0, init_rng,
            jax.ShapeDtypeStruct((n_envs,), jnp.float32),
            jax.ShapeDtypeStruct((n_envs,), jnp.int32),
        )
        learner, runner, aux, meta = load_snapshot(opts.resume_full, learner, r_tmpl, contract)
        progress, console = aux["progress"], aux["console"]
        startup_tracker = aux["startup_tracker"]
        early_quality = aux.get("early_quality", early_quality)
        restored_hold = int(aux["hold"])
        completed, true_step = int(meta["actual_updates"]), int(meta["true_step"])
    else:
        learner = u416_learner
        init_rng, reset_key = jax.random.split(jax.random.PRNGKey(plan["seed"] + 416))
        reset_keys = jax.random.split(reset_key, n_envs)
        fresh_state, fresh_obs = jax.jit(env.reset)(reset_keys)
        fresh_cobs = jax.jit(env.get_critic_obs)(fresh_state, fresh_obs)
        if n_envs == 1024 and u416_inner_runner is not None:
            u416_wrapped_state = env.wrap_state(u416_inner_runner.env_state, staggered=True)
            role = env.role

            def blend(fresh, old):
                if hasattr(fresh, "shape") and fresh.ndim >= 1 and fresh.shape[0] == 1024:
                    m = role.reshape((1024,) + (1,) * (fresh.ndim - 1))
                    return jnp.where(m, fresh, old)
                return old

            mixed_state = jax.tree.map(blend, fresh_state, u416_wrapped_state)
            mixed_obs = jnp.where(role[:, None], fresh_obs, u416_inner_runner.obs)
            mixed_cobs = jnp.where(role[:, None], fresh_cobs, u416_inner_runner.critic_obs)
            mixed_ret = jnp.where(role, 0.0, u416_inner_runner.running_return)
            mixed_len = jnp.where(role, 0, u416_inner_runner.running_length)
            runner = stack.ppo.RunnerState(
                mixed_state, mixed_obs, mixed_cobs, u416_inner_runner.rng, mixed_ret, mixed_len
            )
        else:
            runner = stack.ppo.RunnerState(
                fresh_state,
                fresh_obs,
                fresh_cobs,
                init_rng,
                jnp.zeros((n_envs,), dtype=jnp.float32),
                jnp.zeros((n_envs,), dtype=jnp.int32),
            )

    learner, runner = strong_arrays((learner, runner))
    release_caches()

    if plan["learning_stability"]["critic_warmup_start_t"] == "auto":
        plan["learning_stability"]["critic_warmup_start_t"] = int(learner.opt.t)
        contract["learning_stability"] = plan["learning_stability"]

    collect, update, settings = build_train_functions(stack, env, args, plan)
    diagnostics = make_critic_diagnostics(stack.ppo, settings)

    opts.output.mkdir(parents=True, exist_ok=False)
    save_json(
        opts.output / "PREFLIGHT.json",
        dict(
            timestamp=datetime.now().astimezone().isoformat(),
            **contract,
            gpu_uuid=opts.gpu_uuid,
            xml_sha256=sha256_file(stack.xml),
            initial_native_runner_digest=state_digest(runner),
            actor_total_parameters=actor_parameter_count(learner.params["pi"]),
            trainable_actor_parameters=plan["direct_trainable_actor_parameters"],
            start_update=completed,
            start_true_step=true_step,
            start_opt_t=int(learner.opt.t),
            start_actor_t=int(learner.opt.actor_t),
            ppo=settings,
        ),
    )

    import wandb
    run = wandb.init(
        entity=plan["wandb"]["entity"],
        project=plan["wandb"]["project"],
        id=opts.run_id,
        name=opts.run_id,
        resume="never",
        mode=opts.wandb_mode,
        dir=os.environ.get("WANDB_DIR", "/dev/shm/wandb_wind_early"),
        settings=wandb.Settings(x_disable_stats=True),
        config=dict(contract, update_budget=opts.update_limit, migrated_host=os.environ.get("MIGRATED_HOST", "192.168.14.52")),
    )

    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))

    ret_cfg = plan.get("checkpoint_retention", {})
    periodic_every = int(ret_cfg.get("periodic_every_updates", 8))
    cleanup_every_saves = int(ret_cfg.get("cleanup_every_saves", 5))
    retention = BestAndLatestRetention(opts.output, cleanup_every=cleanup_every_saves)
    gate = make_reference_gate(quality_gate(325, stack.spec.course[14], stack.helpers.quality_passed), 325)
    status = {}
    score = (0.0, -1e6, -1e6, -1e6)
    best_pareto = (-1.0, -1.0, -1.0)
    hold = restored_hold
    started = time.monotonic()
    session = 0
    committed = True
    normal_completion = False
    milestones = set(plan.get("phase_milestones", []))
    rehearsal_mask = env.role_np
    long_mask = ~rehearsal_mask

    def checkpoint(kind):
        path = opts.output / f"{kind}_u{completed:08d}.pkl"
        if path.exists():
            return path
        assert_supported_actor(learner.params)
        payload = dict(
            schema="v325_single5_early_stability_v1",
            mode="direct",
            params=jax.device_get(learner.params),
            opt=jax.device_get(learner.opt),
            initial_single_actor_digest=init_actor_digest,
            source_sha256="fd9def8f5fdb511e1543cbcca264e0021738ff07c6425e8d393cb1c240918e97",
            plan_sha256=plan_sha,
            step=true_step,
            actual_updates=completed,
            stage=15,
            BC=0,
            last_summary=status,
            requires_actor_runtime=plan["observation"]["runtime"],
            native_actor_obs_dim=120,
            single_actor_input_dim=120,
            simulation_only=True,
            final1200s99accepted=False,
        )
        with path.open("xb") as f:
            pickle.dump(payload, f)
        if kind == "periodic":
            retention.register(path, score)
        return path

    def recovery():
        path = opts.output / f"trainer_state_u{completed:08d}.zip"
        if not path.exists():
            save_snapshot(
                path,
                learner,
                runner,
                dict(progress=progress, console=console, startup_tracker=startup_tracker, early_quality=early_quality, hold=hold),
                contract,
                dict(actual_updates=completed, true_step=true_step, run_id=opts.run_id),
            )
            if completed in milestones:
                best = opts.output / f"milestone_state_u{completed:08d}.zip"
                if not best.exists():
                    os.link(path, best)
                    os.link(path.with_suffix(".zip.json"), best.with_suffix(".zip.json"))
            prune_owned_snapshots(opts.output, path)

    print(
        f"[wind-early] Starting from u{completed} (n_envs={n_envs}, n_steps={n_steps}, mb={minibatch_size}, "
        f"perm_wind_only={perm_wind_only}, early_rew={early_rew_cfg.get('enabled', False)}, "
        f"step={true_step}, opt.t={int(learner.opt.t)}, "
        f"rehearsal={env.n_rehearsal}x{rehearsal_cfg['seconds']}s, long={env.n_long}x1200s, wandb={opts.wandb_mode})",
        flush=True,
    )

    try:
        with (opts.output / "updates.jsonl").open("x") as stream, \
             (opts.output / "failure_events.jsonl").open("x") as event_stream:
            while (opts.update_limit is None or completed < opts.update_limit) and not stopped[0] and not (opts.output / "STOP").exists():
                committed = False
                t_upd = time.monotonic()
                runner, transitions = collect(learner.params, runner)
                verify_mature_batch(transitions.metrics)
                if session == 0:
                    gc.collect()
                    ctypes.CDLL(None).malloc_trim(0)

                explained, variance = diagnostics(learner.params, runner, transitions)
                new, loss = update(learner, runner, transitions)

                warmup_end_t = int(plan["learning_stability"]["critic_warmup_start_t"]) + int(plan["learning_stability"]["critic_only_minibatches"])
                in_warmup = int(learner.opt.t) < warmup_end_t
                if perm_wind_only or in_warmup:
                    assert bool(frozen_base_equal(new.params, initial_learner_params)), "Frozen 0:118 backbone changed!"
                assert bool(jnp.allclose(new.params["log_std"], initial_learner_params["log_std"], atol=1e-6)), "Fixed log_std drifted!"
                loss = {k: float(v) for k, v in jax.device_get(loss).items()}
                assert loss["learning_stability/sampled_max_step_kl"] <= plan["learning_stability"]["maximum_sampled_cohort_step_kl"] + 1e-7
                assert all(np.isfinite(list(loss.values())))
                minis = int(loss["ppo_minibatches_applied"])
                assert 0 <= minis <= max_minibatches
                assert int(new.opt.t) == int(learner.opt.t) + minis
                assert_supported_actor(new.params)

                metrics = jax.device_get(transitions.metrics)
                dones = jax.device_get(transitions.done)
                assert np.all(metrics["v321/finite"] == 1.0)
                assert not np.any(metrics["wind/positive_y_violation"])
                assert np.max(metrics["wind/speed_m_s"]) <= 1.60001
                np.testing.assert_allclose(metrics["pose/outer_limit_deg"], 5.0, atol=1e-5)

                cutoff = metrics["sampling/rehearsal_cutoff"] > 0.5
                if np.any(cutoff):
                    assert np.all(np.asarray(transitions.truncated)[cutoff])
                    assert not np.any(np.asarray(transitions.terminated)[cutoff])
                    assert np.isfinite(np.asarray(transitions.timeout_value)[cutoff]).all()

                log_failure_events(event_stream, completed + 1, metrics, dones)

                long_metrics, long_done = masked_rollout(metrics, dones, long_mask)
                progress.add(long_metrics, long_done)

                reh_metrics, reh_done = masked_rollout(metrics, dones, rehearsal_mask)
                startup_tracker.add(reh_metrics, reh_done, TERMINATION_FIELDS)

                console.add(metrics, dones)
                early_quality.add(metrics)

                learner = new
                session += 1
                completed += 1
                true_step += batch_size
                committed = True

                summaries = progress.summaries()
                final_gate = gate99(summaries, gate, minimum_n=100)
                startup_summary = startup_tracker.summary()
                hold = hold + 1 if final_gate["passed"] else 0

                w_rows = np.asarray(jax.device_get(learner.params["pi"]["l1"]["w"][118:120]))
                w_l2 = float(np.linalg.norm(w_rows))
                w_max = float(np.max(np.abs(w_rows)))
                bb_sq = 0.0
                for lname, layer in learner.params["pi"].items():
                    for pkey, pval in layer.items():
                        diff = np.asarray(jax.device_get(pval - initial_learner_params["pi"][lname][pkey]))
                        if lname == "l1" and pkey == "w":
                            diff = diff[:118]
                        bb_sq += float(np.sum(diff * diff))
                bb_delta_l2 = float(np.sqrt(max(bb_sq, 0.0)))
                early_q = early_quality.summary()
                upd_sec = time.monotonic() - t_upd

                status = dict(
                    update=completed,
                    actual_update=completed,
                    stage=15,
                    mode="direct",
                    true_step=true_step,
                    actor_local_Adam_t=int(learner.opt.actor_t),
                    applied_minibatches_total=int(learner.opt.t),
                    **loss,
                    **{
                        "critic/explained_variance": float(explained),
                        "critic/return_target_variance": float(variance),
                        "wind_rows/l2_norm": w_l2,
                        "wind_rows/max_abs": w_max,
                        "backbone/delta_l2_norm": bb_delta_l2,
                        "reward/early_center_sum": float(np.sum(metrics["reward/early_center"])),
                        "reward/early_hit_vxy_sum": float(np.sum(metrics["reward/early_hit_vxy"])),
                        "reward/early_step_vxy_per_s": float(np.mean(metrics["reward/early_step_vxy"]) / 0.005),
                        "reward/early_apex_sum": float(np.sum(metrics["reward/early_apex"])),
                        "perf/update_seconds": upd_sec,
                        "sampling/rehearsal_censors": int(cutoff.sum()),
                        "sampling/rehearsal_terminations": int(np.sum(np.asarray(reh_metrics["terminated"]) > 0.5)),
                        "sampling/long_terminations": int(np.sum(np.asarray(long_metrics["terminated"]) > 0.5)),
                        "reward/targeted_brake_per_s": float(np.mean(metrics["reward/targeted_brake"]) / 0.005),
                        "reward/targeted_term_reserve_sum": float(np.sum(metrics["reward/targeted_termination_reserve"])),
                        "reward/targeted_outgoing_xy_per_s": float(np.mean(metrics["reward/targeted_outgoing_xy"]) / 0.005),
                        "reward/targeted_outgoing_z_per_s": float(np.mean(metrics["reward/targeted_outgoing_z"]) / 0.005),
                        "learning_stability/critic_only_warmup": int(int(learner.opt.t) < int(plan["learning_stability"]["critic_warmup_start_t"]) + int(plan["learning_stability"]["critic_only_minibatches"])),
                    },
                    startup_rehearsal=startup_summary,
                    early_hit_quality=early_q,
                    groups=summaries,
                    hold=hold,
                    final_gate=final_gate,
                    wall_seconds=time.monotonic() - started,
                )
                for tag in ("done/racket_z_out_of_bounds", "done/pending_hit_recontact",
                            "done/ball_too_low", "done/ball_too_high", "done/ball_view_z_too_low",
                            "done/failed_hit_confirmation", "done/late_weak_streak"):
                    if tag in metrics:
                        status["term_count/" + tag.split("/", 1)[1]] = int(np.sum(metrics[tag] > 0.5))

                stream.write(json.dumps(status) + "\n")
                stream.flush()
                event_stream.flush()

                flat = {k: v for k, v in status.items() if isinstance(v, (int, float))}
                for k, v in startup_summary.items():
                    if isinstance(v, (int, float)):
                        flat["rehearsal20/" + k] = v
                for k, v in early_q.items():
                    if isinstance(v, (int, float)):
                        flat["early_hits/" + k] = v
                for pop, summ in summaries.items():
                    for hor in ("short30", "stage", "full1200"):
                        avail = summ[hor].get("n", 0) > 0
                        flat[f"population/{pop}/{hor}/available"] = int(avail)
                        for k, v in summ[hor].items():
                            if isinstance(v, (int, float)) and (avail or k != "survival"):
                                flat[f"population/{pop}/{hor}/{k}"] = v
                    for f_name in ("center_rms_m", "apex_error_rms_m", "in_band", "adjacent_jump_m", "racket_vxy_excess_rms_m_s"):
                        if f_name in summ.get("quality", {}):
                            flat[f"population/{pop}/quality/{f_name}"] = summ["quality"][f_name]
                for f_name in ("observed_force/valid", "observed_force/age_s", "observed_force/error_sq_m2_s2"):
                    flat[f_name] = float(np.mean(metrics[f_name]))
                run.log(flat, step=completed)

                print(
                    f"u{completed:05d} | {upd_sec:5.1f}s | Adam+{minis:2d} | EV={float(explained):.4f} | "
                    f"kl={loss['learning_stability/sampled_max_step_kl']:.5f}(s={loss['learning_stability/step_scale']:.2f}) | "
                    f"w118={w_l2:.4f} bb={bb_delta_l2:.4f} | "
                    f"reh20(n={startup_summary['n']:4d} surv={startup_summary['survival']*100:5.1f}% "
                    f"clnP={startup_summary.get('clean_phys_survival',0)*100:5.1f}% "
                    f"peak={startup_summary.get('peak_survival',0)*100:5.1f}% "
                    f"negX={startup_summary.get('negx_peak_survival',0)*100:5.1f}% "
                    f"dir08={startup_summary.get('dir08_survival',0)*100:5.1f}%) | "
                    f"h1_5(c={early_q['h1_5_clean_center_rms_mm']:4.1f}mm v={early_q['h1_5_clean_vxy_rms_m_s']:.3f}) "
                    f"h6_15(c={early_q['h6_15_clean_center_rms_mm']:4.1f}mm v={early_q['h6_15_clean_vxy_rms_m_s']:.3f}) | "
                    f"rehTerm={status['sampling/rehearsal_terminations']} longTerm={status['sampling/long_terminations']}",
                    flush=True,
                )

                dir08_band = float(summaries.get("xpeak_dir08", {}).get("quality", {}).get("in_band", 0.0))
                dir00_band = float(summaries.get("xpeak_dir00", {}).get("quality", {}).get("in_band", 0.0))
                clean_phys = float(startup_summary.get("clean_phys_survival", 0.0))
                clean_ok = float(startup_summary.get("n", 0) >= 256 and clean_phys >= 0.99)
                score = (
                    float(final_gate["passed"]),
                    clean_ok,
                    float(startup_summary.get("negx_peak_survival", 0.0)),
                    float(startup_summary.get("peak_survival", 0.0)),
                    float(startup_summary.get("survival", 0.0)),
                )
                if (startup_summary.get("n", 0) >= 1024 and clean_phys >= 0.995
                        and dir08_band >= 0.985 and dir00_band >= 0.985):
                    peak_s = float(startup_summary.get("peak_survival", 0.0))
                    negx_s = float(startup_summary.get("negx_peak_survival", 0.0))
                    dir08_s = float(startup_summary.get("dir08_survival", 0.0))
                    if early_rew_cfg.get("enabled", False):
                        if peak_s >= 0.95 and negx_s >= 0.93 and early_q["h6_15_clean_n"] >= 100 and early_q["h1_5_clean_n"] >= 50:
                            c15 = early_q["h1_5_clean_center_rms_mm"]
                            c615 = early_q["h6_15_clean_center_rms_mm"]
                            v615 = early_q["h6_15_clean_vxy_rms_m_s"]
                            stab_pen = (
                                0.015 * max(c15 - 9.6, 0.0)
                                + 0.025 * max(c615 - 6.0, 0.0)
                                + 2.5 * max(v615 - 0.040, 0.0)
                            )
                            joint_score = round(0.40 * negx_s + 0.30 * peak_s + 0.30 * dir08_s - stab_pen, 4)
                            cand_pareto = (joint_score, float(startup_summary.get("survival", 0.0)), -c615)
                            if cand_pareto > best_pareto:
                                best_pareto = cand_pareto
                                bp = checkpoint("best_pareto")
                                for old_bp in opts.output.glob("best_pareto_u*.pkl"):
                                    if old_bp.resolve() != bp.resolve():
                                        old_bp.unlink(missing_ok=True)
                                print(
                                    f"  [BEST PARETO] saved {bp.name} joint={joint_score:.4f} "
                                    f"peak={peak_s*100:.1f}% negX={negx_s*100:.1f}% dir08={dir08_s*100:.1f}% "
                                    f"h1_5_c={c15:.1f}mm h6_15_c={c615:.1f}mm h6_15_v={v615:.3f}m/s cleanP={clean_phys*100:.1f}%",
                                    flush=True,
                                )
                    else:
                        comp = round(0.40 * negx_s + 0.30 * peak_s + 0.30 * dir08_s, 4)
                        cand_pareto = (comp, float(startup_summary.get("survival", 0.0)), dir08_s)
                        if cand_pareto > best_pareto:
                            best_pareto = cand_pareto
                            bp = checkpoint("best_pareto")
                            for old_bp in opts.output.glob("best_pareto_u*.pkl"):
                                if old_bp.resolve() != bp.resolve():
                                    old_bp.unlink(missing_ok=True)
                            print(f"  [BEST PARETO] saved {bp.name} comp={comp*100:.2f}% peak={peak_s*100:.1f}% negX={negx_s*100:.1f}% dir08={dir08_s*100:.1f}% dir08_band={dir08_band*100:.1f}% clean_phys={clean_phys*100:.1f}%", flush=True)
                if completed % periodic_every == 0:
                    checkpoint("periodic")
                if completed in milestones:
                    checkpoint("milestone")
                    recovery()
                elif completed % 32 == 0:
                    recovery()
                gc.collect()
                ctypes.CDLL(None).malloc_trim(0)
                if hold >= 16:
                    save_json(
                        opts.output / "READY_FOR_FIXED_POLICY_1200S.json",
                        dict(update=completed, gate=final_gate, training_readiness_only=True, final1200s99accepted=False),
                    )
                    break

        checkpoint("interrupted" if (stopped[0] or (opts.output / "STOP").exists()) else "bounded_final")
        retention.prune()
        recovery()
        assert direct_trace_calls[0] > 0
        run.summary["actual_updates"] = completed
        run.summary["final1200s99accepted"] = False
        normal_completion = True
    finally:
        if committed and not normal_completion:
            checkpoint("interrupted" if stopped[0] else "last")
            retention.prune()
            recovery()
        run.finish()
    print(f"[wind-early] Completed at u{completed}", flush=True)


if __name__ == "__main__":
    main()
