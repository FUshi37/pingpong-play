"""V325 Sim2Real + High-Wind (`s2r_r6`) Trainer: 4.0 m/s wind & +/-15 deg pose from archived S2R u7447.

Starts from `v325_wind_early_s2r_u7447_20261128/model.pkl` with exact linear input-weight
rescaling (`1.6 -> 4.5 m/s`), retains `u7447`'s proven Sim2Real physics & sensing DR
(`P_warm=1.0`, `dr_scale=0.65`, asymmetric clean Critic), and auto-advances a zero-recompile
high-wind curriculum: `2.2 -> 2.6 -> 3.0 -> 3.4 -> 3.8 -> 4.0 m/s`.
"""
import argparse
import ctypes
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import pickle
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np

from migrated_stack import REPO, load_migrated_v325_stack, setup_sys_path, sha256_file
setup_sys_path()

from early_stability_reward import EARLY_REWARD_FIELDS
from high_wind_r6 import (
    HighWindFeatureEnv,
    HighWindSingle5Env,
    install_high_wind_pose_mode,
    rescale_actor_wind_inputs,
)
from sim2real_r6 import (
    HighWindEarlyStabilityRewardEnv,
    S2RR6RehearsalEnv,
    S2RR6StartupProgress,
    S2RR6WindSamplingEnv,
    S2R_R6_METRIC_FIELDS,
    Sim2RealR6PhysicsAndObsEnv,
    configure_s2r_r6_cfg,
    s2r_r6_rehearsal_mask,
)
from stratified_rehearsal import (
    EarlyHitQualityTracker,
    FIELDS as REHEARSAL_FIELDS,
    masked_rollout,
)
from wide_onset import StratifiedWideOnsetEnv

from train_s15_fresh_velocity_r31 import ActualContactEnv, CONTACT_FIELDS
from capture_margin import CaptureEnv
from critic_diagnostics import make_critic_diagnostics
from fixed_plane_v321_long_episode import QUALITY_FIELDS
from fixed_plane_wind_flight_outcomes import HEIGHT_FIELDS
from fixed_plane_wind_reference_constraints import REFERENCE_FIELDS
from train_fixed_plane_wind import make_train_functions, strong_arrays
from bounded_validation_memory import release_caches
from learning_stability import cap_actor_update, critic_warmup_adam
from precision_maturity import PRECISION_FIELDS, verify_mature_batch
from randomized_onset import PREVIEW_FIELDS, TIMING_FIELDS
from single5_runtime import REVISION, actor_parameter_count, adam_factory
from targeted_reward import FIELDS as TARGETED_FIELDS
from termination_diagnostics import TERMINATION_FIELDS
from training_snapshot import prune_owned_snapshots, save_snapshot
from weak_release_floor import FIELDS as RELEASE_FLOOR_FIELDS
from feature_training import assert_supported_actor


def save_json(path: Path, value: dict):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)


class BestAndLatestRetention:
    def __init__(self, out_dir: Path, cleanup_every: int = 5):
        self.out_dir = out_dir
        self.cleanup_every = max(1, int(cleanup_every))
        self.saves_since_cleanup = 0
        self.best_path: Path | None = None
        self.best_score: tuple | None = None
        self.latest_path: Path | None = None

    def register(self, path: Path, score: tuple):
        self.latest_path = path
        if self.best_score is None or score >= self.best_score:
            self.best_score = score
            self.best_path = path
        self.saves_since_cleanup += 1
        if self.saves_since_cleanup >= self.cleanup_every:
            self.prune()

    def prune(self):
        keep = {p.resolve() for p in (self.best_path, self.latest_path) if p is not None}
        for p in sorted(self.out_dir.glob("periodic_u*.pkl")):
            if p.resolve() not in keep:
                p.unlink(missing_ok=True)
        self.saves_since_cleanup = 0


def build_train_functions(stack, env, args, plan):
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
        "wind/entry_force_clipped", "wind/required_tilt_deg",
        "terminated", "truncated", "new_hit", "hit_count",
        "polish/background_speed_m_s", "polish/wind_sector", "polish/peak_assigned",
        "observed_force/valid", "observed_force/age_s", "observed_force/error_sq_m2_s2",
        "single5/wind_angle_rad", "pose/outer_limit_deg",
        "hit_racket_vxy_sq_sum", "reward/total",
    )
    goal_fields = tuple(
        "reward/goal_" + k
        for k in ("alive", "hit", "outgoing", "normal", "height_margin", "braking", "apex", "failure")
    )
    all_metrics = (
        tuple(captured["full_rollout_metric_names"])
        + QUALITY_FIELDS + HEIGHT_FIELDS + REFERENCE_FIELDS
        + CONTACT_FIELDS + TARGETED_FIELDS + TIMING_FIELDS + PREVIEW_FIELDS
        + TERMINATION_FIELDS + PRECISION_FIELDS + RELEASE_FLOOR_FIELDS
        + REHEARSAL_FIELDS + EARLY_REWARD_FIELDS + S2R_R6_METRIC_FIELDS + goal_fields + extra_fields
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
    kl_sigma = float(stab.get("kl_sigma", math.exp(float(plan["ppo"]["min_log_std"]))))
    collect, update = original(**captured)
    update = cap_actor_update(
        update,
        stack.ppo.policy_mean,
        maximum_kl=stab["maximum_sampled_cohort_step_kl"],
        sample_size=stab["step_observations"],
        sigma=kl_sigma,
    )
    effective["critic_warmup_start_t"] = warmup_start_t
    effective["critic_only_minibatches"] = int(stab["critic_only_minibatches"])
    effective["kl_sigma"] = kl_sigma
    return collect, update, effective


def update_runner_wind_stage(runner, new_speed_m_s: float, new_stage_id: int):
    """Update `curr_stage_id` (S2RR6ObsState) and `active_max_speed` (S2RR6WindState) zero-recompile."""
    def _walk(st):
        fields = getattr(st, "_fields", ())
        updates = {}
        if "core" in fields:
            updates["core"] = _walk(st.core)
        if "curr_stage_id" in fields:
            updates["curr_stage_id"] = jnp.asarray(int(new_stage_id), dtype=jnp.int32)
        if "active_max_speed" in fields:
            updates["active_max_speed"] = jnp.full_like(st.active_max_speed, float(new_speed_m_s))
        return st._replace(**updates) if updates else st

    return runner._replace(env_state=_walk(runner.env_state))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=HERE / "TRAIN_PLAN_S2R_R6.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--update-limit", type=int, default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="offline")
    opts = parser.parse_args()

    plan = json.loads(opts.plan.read_text())
    plan_sha = sha256_file(opts.plan)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == opts.gpu_uuid and opts.gpu_uuid in plan["allowed_gpu_uuids"]
    assert jax.default_backend() == "gpu" and len(jax.devices()) == 1

    pose_deg = float(plan["pose"]["limit_deg"])
    max_speed = float(plan["wind_domain"]["max_speed_m_s"])
    force_limit_n = float(plan["wind_domain"].get("force_limit_n", 0.025))
    obs_wind_max = float(plan["observation"]["obs_wind_max_m_s"])
    old_obs_wind_max = float(plan["observation"].get("parent_obs_wind_max_m_s", 1.6))
    s2r_cfg = plan["sim2real"]
    dr_scale = float(s2r_cfg.get("dr_scale", 0.65))

    curr_cfg = plan["wind_curriculum"]
    stages = [float(s) for s in curr_cfg["speeds_m_s"]]
    stage_idx = 0
    active_speed = stages[stage_idx]
    promo_streak = 0
    stage_updates = 0

    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
        for k, v in list(stack.parent["args"].items()):
            if isinstance(v, (str, Path)) and "/home/yangzhe/Project/pingpong_controller/" in str(v):
                new_s = str(REPO / str(v).replace("/home/yangzhe/Project/pingpong_controller/", ""))
                stack.parent["args"][k] = Path(new_s) if isinstance(v, Path) else new_s
        stack.cfg = configure_s2r_r6_cfg(
            stack.cfg, pose_deg=pose_deg, max_speed_m_s=max_speed, force_limit_n=force_limit_n, s2r_cfg=s2r_cfg
        )
        args = argparse.Namespace(**stack.parent["args"])
        n_envs = int(plan.get("n_envs", 4096))
        n_steps = int(plan.get("n_steps", 128))
        minibatch_size = int(plan["ppo"].get("minibatch_size", 65536))
        batch_size = n_envs * n_steps
        max_minibatches = int(plan["ppo"].get("update_epochs", 4)) * max(1, batch_size // minibatch_size)
        args.n_envs, args.n_steps, args.minibatch_size = n_envs, n_steps, minibatch_size
        args.seed = plan["seed"]
        args.wind_training_mix = stack.wind_training_mix

    install_high_wind_pose_mode(pose_deg)
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=stack.cfg)
    wind_env = S2RR6WindSamplingEnv(
        CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
        half_angle_deg=float(plan["wind_domain"]["x_axis_half_angle_deg"]),
        init_max_speed=active_speed,
    )
    real_twin_np = wind_env.real_twin_np

    env = HighWindSingle5Env(
        wind_env,
        obs_wind_max_m_s=obs_wind_max,
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

    rehearsal_cfg = plan["stratified_rehearsal"]
    onset_cfg = plan["onset"]
    reh_mask = s2r_r6_rehearsal_mask(n_envs, fraction=float(rehearsal_cfg["fraction"]))
    env.inner = StratifiedWideOnsetEnv(env.inner.inner, rehearsal_mask=reh_mask, onset_cfg=onset_cfg, enabled=True)
    env = HighWindFeatureEnv(env, obs_wind_max_m_s=obs_wind_max)
    env = Sim2RealR6PhysicsAndObsEnv(
        env,
        real_twin_mask=real_twin_np,
        lane_mode=wind_env.layout[0],
        s2r_cfg=s2r_cfg,
        obs_wind_max_m_s=obs_wind_max,
        dr_scale=dr_scale,
        init_stage_id=stage_idx,
    )
    early_rew_cfg = plan.get("early_stability_reward", {})
    env = HighWindEarlyStabilityRewardEnv(env, early_rew_cfg)
    env = S2RR6RehearsalEnv(
        env,
        fraction=float(rehearsal_cfg["fraction"]),
        seconds=float(rehearsal_cfg["seconds"]),
        stagger_min_steps=int(rehearsal_cfg["stagger_min_steps"]),
        stagger_seed=int(rehearsal_cfg["stagger_seed"]),
    )
    assert (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) == (120, 453, 7, 240000)

    # Load parent checkpoint and apply scale-preserving input weight rescaling ONLY if scale changed
    parent_path = REPO / plan["parent_checkpoint"]
    parent_sha = sha256_file(parent_path)
    with parent_path.open("rb") as f:
        p_saved = pickle.load(f)
    assert_supported_actor(p_saved["params"])
    if p_saved.get("obs_wind_max_m_s") is not None:
        assert abs(float(p_saved["obs_wind_max_m_s"]) - old_obs_wind_max) < 1e-6, (
            f"Parent obs_wind_max_m_s={p_saved['obs_wind_max_m_s']} != plan parent_obs_wind_max_m_s={old_obs_wind_max}"
        )
    p_params = jax.tree.map(jnp.asarray, p_saved["params"])
    p_params = rescale_actor_wind_inputs(p_params, old_max_m_s=old_obs_wind_max, new_max_m_s=obs_wind_max)

    target_log_std = float(plan["ppo"]["min_log_std"])
    p_params = {**p_params, "log_std": jnp.full_like(p_params["log_std"], target_log_std)}

    p_opt = jax.tree.map(jnp.asarray, p_saved["opt"])
    if plan["feature_optimizer"].get("reset_all_moments_on_start", True):
        zero_all = jax.tree.map(jnp.zeros_like, p_params)
        p_opt = p_opt._replace(m=zero_all, v=zero_all, actor_t=jnp.asarray(0, dtype=jnp.int32))
        print("[s2r_r6] Reset actor+critic Adam moments (m=0, v=0, actor_t=0).", flush=True)

    learner = stack.ppo.TrainState(p_params, p_opt)
    completed = int(p_saved["actual_updates"])
    true_step = int(p_saved["step"])
    init_actor_digest = p_saved.get("initial_single_actor_digest", parent_sha)
    print(
        f"[s2r_r6] Loaded {parent_path.name} (u{completed}, rescaled wind obs {old_obs_wind_max}->{obs_wind_max} m/s, "
        f"pose=+/-{pose_deg}deg, S2R P_warm+dr={dr_scale:.2f}, sigma={math.exp(target_log_std):.4f}, stages={stages})",
        flush=True,
    )

    initial_learner_params = jax.tree.map(jnp.array, learner.params)
    stack.single_actor_anchor_params = initial_learner_params
    stack.bootstrap_stage_anchor = initial_learner_params

    init_rng, reset_key = jax.random.split(jax.random.PRNGKey(plan["seed"] + 7447))
    reset_keys = jax.random.split(reset_key, n_envs)
    fresh_state, fresh_obs = jax.jit(env.reset)(reset_keys)
    fresh_cobs = jax.jit(env.get_critic_obs)(fresh_state, fresh_obs)
    runner = stack.ppo.RunnerState(
        fresh_state, fresh_obs, fresh_cobs, init_rng,
        jnp.zeros((n_envs,), dtype=jnp.float32),
        jnp.zeros((n_envs,), dtype=jnp.int32),
    )

    learner, runner = strong_arrays((learner, runner))
    release_caches()

    if plan["learning_stability"]["critic_warmup_start_t"] == "auto":
        plan["learning_stability"]["critic_warmup_start_t"] = int(learner.opt.t)

    contract = dict(
        schema="v325_s2r_high_wind_r6_v1",
        mode="direct",
        parent_checkpoint=plan["parent_checkpoint"],
        parent_sha256=parent_sha,
        actor_critic_action=[120, 453, 7],
        pose_limit_deg=pose_deg,
        obs_wind_max_m_s=obs_wind_max,
        max_speed_m_s=max_speed,
        curriculum_speeds_m_s=stages,
        sim2real=s2r_cfg,
        plan_sha256=plan_sha,
        revision=plan["revision"],
        n_envs=n_envs,
        n_steps=n_steps,
        minibatch_size=minibatch_size,
        learning_stability=plan["learning_stability"],
        early_stability_reward=early_rew_cfg,
    )

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
            actor_total_parameters=actor_parameter_count(learner.params["pi"]),
            start_update=completed,
            start_true_step=true_step,
            start_opt_t=int(learner.opt.t),
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
        dir=os.environ.get("WANDB_DIR", "/dev/shm/wandb_wind_s2r_r6"),
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
    startup_tracker = S2RR6StartupProgress(
        seconds=float(rehearsal_cfg["seconds"]),
        capacity=int(rehearsal_cfg["metric_capacity"]),
        active_stage_id=stage_idx,
    )
    early_quality = EarlyHitQualityTracker(window_updates=16)

    status = {}
    score = (0, 0.0, 0.0)
    best_pareto = (-1, -1.0, -1.0)
    started = time.monotonic()
    session = 0
    committed = True
    normal_completion = False
    rehearsal_mask = env.role_np
    long_mask = ~rehearsal_mask

    def checkpoint(kind: str):
        path = opts.output / f"{kind}_u{completed:08d}.pkl"
        if path.exists():
            return path
        assert_supported_actor(learner.params)
        payload = dict(
            schema="v325_s2r_high_wind_r6_v1",
            mode="direct",
            params=jax.device_get(learner.params),
            opt=jax.device_get(learner.opt),
            initial_single_actor_digest=init_actor_digest,
            source_sha256=parent_sha,
            plan_sha256=plan_sha,
            step=true_step,
            actual_updates=completed,
            stage=16,
            active_max_speed_m_s=active_speed,
            pose_limit_deg=pose_deg,
            obs_wind_max_m_s=obs_wind_max,
            s2r_dr_scale=dr_scale,
            BC=0,
            last_summary=status,
            requires_actor_runtime=plan["observation"]["runtime"],
            native_actor_obs_dim=120,
            single_actor_input_dim=120,
            simulation_only=False,
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
                path, learner, runner,
                dict(startup_tracker=startup_tracker, early_quality=early_quality, stage_idx=stage_idx, active_speed=active_speed, stage_updates=stage_updates),
                contract,
                dict(actual_updates=completed, true_step=true_step, run_id=opts.run_id, stage_idx=stage_idx),
            )
            prune_owned_snapshots(opts.output, path)

    print(
        f"[s2r_r6] Starting from u{completed} (n_envs={n_envs}, n_steps={n_steps}, mb={minibatch_size}, "
        f"pose=+/-{pose_deg}deg, obs_max={obs_wind_max}m/s, active_wind={active_speed:.1f}m/s, "
        f"realTwin={int(real_twin_np.sum())} envs, dr={dr_scale:.2f}, rehearsal={env.n_rehearsal}x25s, long={env.n_long}x1200s)",
        flush=True,
    )

    try:
        with (opts.output / "updates.jsonl").open("x") as stream:
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

                loss = {k: float(v) for k, v in jax.device_get(loss).items()}
                assert loss["learning_stability/sampled_max_step_kl"] <= plan["learning_stability"]["maximum_sampled_cohort_step_kl"] + 1e-6
                assert all(np.isfinite(list(loss.values())))
                minis = int(loss["ppo_minibatches_applied"])
                assert 0 <= minis <= max_minibatches
                assert_supported_actor(new.params)

                metrics = jax.device_get(transitions.metrics)
                dones = jax.device_get(transitions.done)
                assert np.all(metrics["v321/finite"] == 1.0)
                assert not np.any(metrics["wind/positive_y_violation"])
                assert float(np.max(metrics["wind/speed_m_s"])) <= active_speed + 0.05
                np.testing.assert_allclose(metrics["pose/outer_limit_deg"], pose_deg, atol=1e-4)

                reh_metrics, reh_dones = masked_rollout(metrics, dones, rehearsal_mask)
                long_metrics, long_dones = masked_rollout(metrics, dones, long_mask)
                startup_tracker.add(reh_metrics, reh_dones, active_max_speed=active_speed)
                early_quality.add(metrics)

                learner = new
                completed += 1
                session += 1
                stage_updates += 1
                true_step += batch_size
                committed = True

                startup_summary = startup_tracker.summary()
                early_q = early_quality.summary()
                upd_sec = time.monotonic() - t_upd

                warmup_end_t = int(plan["learning_stability"]["critic_warmup_start_t"]) + int(plan["learning_stability"]["critic_only_minibatches"])
                in_warmup = int(learner.opt.t) < warmup_end_t
                obs_rmse = math.sqrt(max(float(np.mean(metrics["observed_force/error_sq_m2_s2"])), 0.0))
                req_tilt_max = float(np.max(metrics.get("wind/required_tilt_deg", np.zeros(1))))
                tilt_p95 = float(np.percentile(metrics["s2r/tilt_deg"], 95))

                status = dict(
                    update=completed,
                    true_step=true_step,
                    opt_t=int(learner.opt.t),
                    actor_t=int(learner.opt.actor_t),
                    active_max_speed_m_s=active_speed,
                    stage_idx=stage_idx,
                    stage_updates=stage_updates,
                    in_critic_warmup=int(in_warmup),
                    explained_variance=float(explained),
                    return_variance=float(variance),
                    obs_wind_rmse_m_s=obs_rmse,
                    required_tilt_max_deg=req_tilt_max,
                    tilt_p95_deg=tilt_p95,
                    update_wall_s=upd_sec,
                    reh_terms=int(np.sum(np.asarray(reh_metrics["terminated"]) > 0.5)),
                    long_terms=int(np.sum(np.asarray(long_metrics["terminated"]) > 0.5)),
                    **loss,
                    startup_rehearsal=startup_summary,
                    early_hit_quality=early_q,
                    wall_seconds=time.monotonic() - started,
                )
                stream.write(json.dumps(status) + "\n")
                stream.flush()

                flat = {k: v for k, v in status.items() if isinstance(v, (int, float))}
                for k, v in startup_summary.items():
                    if isinstance(v, (int, float)):
                        flat["rehearsal25/" + k] = v
                for k, v in early_q.items():
                    if isinstance(v, (int, float)):
                        flat["early_hits/" + k] = v
                run.log(flat, step=completed)

                w_tag = "CRIT-WU" if in_warmup else f"W={active_speed:.1f}m/s"
                print(
                    f"u{completed:05d} | {upd_sec:5.1f}s | {w_tag:8s} | EV={float(explained):.4f} | "
                    f"kl={loss['learning_stability/sampled_max_step_kl']:.5f}(s={loss['learning_stability/step_scale']:.2f}) | "
                    f"wRMSE={obs_rmse:.2f} tilt95={tilt_p95:4.1f}d | "
                    f"reh25(n={startup_summary['n']:4d} surv={startup_summary['survival']*100:5.1f}%[str={startup_summary['strict_survival']*100:4.1f}%] "
                    f"twin={startup_summary['real_twin_survival']*100:5.1f}% "
                    f"clnP={startup_summary['clean_phys_survival']*100:5.1f}% "
                    f"peak={startup_summary['peak_survival']*100:5.1f}% "
                    f"negX={startup_summary['negx_peak_survival']*100:5.1f}% "
                    f"dir08={startup_summary['dir08_survival']*100:5.1f}% "
                    f"dyn={startup_summary['dyn_survival']*100:5.1f}%) | "
                    f"term={status['reh_terms']}/{status['long_terms']}",
                    flush=True,
                )

                twin_s = float(startup_summary["real_twin_survival"])
                clean_s = float(startup_summary["clean_phys_survival"])
                peak_s = float(startup_summary["peak_survival"])
                negx_s = float(startup_summary["negx_peak_survival"])
                dir08_s = float(startup_summary["dir08_survival"])
                dyn_s = float(startup_summary["dyn_survival"])
                surv_s = float(startup_summary["survival"])
                comp = round(0.25 * twin_s + 0.25 * peak_s + 0.20 * negx_s + 0.10 * dir08_s + 0.20 * dyn_s, 4)
                score = (stage_idx, comp, surv_s)

                if not in_warmup and startup_summary["n"] >= 1024 and clean_s >= 0.92 and twin_s >= 0.85:
                    cand_pareto = (stage_idx, comp, surv_s)
                    if cand_pareto > best_pareto:
                        best_pareto = cand_pareto
                        bp = checkpoint(f"best_pareto_w{round(active_speed*10):02d}")
                        for old_bp in opts.output.glob(f"best_pareto_w{round(active_speed*10):02d}_u*.pkl"):
                            if old_bp.resolve() != bp.resolve():
                                old_bp.unlink(missing_ok=True)
                        print(
                            f"  [BEST W={active_speed:.1f}m/s] saved {bp.name} comp={comp*100:.2f}% "
                            f"twin={twin_s*100:.1f}% clnP={clean_s*100:.1f}% peak={peak_s*100:.1f}% "
                            f"negX={negx_s*100:.1f}% dir08={dir08_s*100:.1f}% dyn={dyn_s*100:.1f}%",
                            flush=True,
                        )

                # Stage-pure auto-advancing curriculum promotion
                if (
                    not in_warmup
                    and stage_idx + 1 < len(stages)
                    and stage_updates >= int(curr_cfg.get("min_stage_updates", 24))
                    and startup_summary["n"] >= int(curr_cfg.get("min_rehearsal_n", 1536))
                ):
                    passed = (
                        twin_s >= float(curr_cfg.get("promote_twin_min", 0.88))
                        and clean_s >= float(curr_cfg.get("promote_clean_min", 0.94))
                        and peak_s >= float(curr_cfg.get("promote_peak_min", 0.90))
                        and dyn_s >= float(curr_cfg.get("promote_dyn_min", 0.88))
                    )
                    promo_streak = (promo_streak + 1) if passed else 0
                    if promo_streak >= int(curr_cfg.get("promote_streak", 2)):
                        ms = checkpoint(f"stage_graduated_w{round(active_speed*10):02d}")
                        stage_idx += 1
                        active_speed = stages[stage_idx]
                        promo_streak = 0
                        stage_updates = 0
                        runner = update_runner_wind_stage(runner, active_speed, stage_idx)
                        startup_tracker.set_stage(stage_idx)
                        print(
                            f"  >>> [CURRICULUM PROMOTED] graduated {ms.name} -> W={active_speed:.1f} m/s "
                            f"(stage {stage_idx+1}/{len(stages)}, ZERO XLA recompile!) <<<",
                            flush=True,
                        )

                if completed % periodic_every == 0:
                    checkpoint("periodic")
                if completed % 32 == 0:
                    recovery()
                gc.collect()
                ctypes.CDLL(None).malloc_trim(0)

        checkpoint("interrupted" if (stopped[0] or (opts.output / "STOP").exists()) else "bounded_final")
        retention.prune()
        recovery()
        normal_completion = True
    finally:
        if committed and not normal_completion:
            checkpoint("interrupted" if stopped[0] else "last")
            retention.prune()
        run.finish()
        print(f"[s2r_r6] Completed at u{completed} (stage W={active_speed:.1f} m/s)", flush=True)


if __name__ == "__main__":
    main()
