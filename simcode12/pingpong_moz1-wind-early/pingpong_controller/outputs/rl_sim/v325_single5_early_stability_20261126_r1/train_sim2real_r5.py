"""V325 Sim2Real Auto-Curriculum Trainer (`s2r_v2`) — 100% Drop-In Real-Robot Model.

Continues from archived `u6247` (`120-D`, `120->282->282->282->7`, `+/-5.0 deg`, `1.6 m/s`)
through a zero-recompile 11-stage Sim2Real curriculum:
  - Phase 1 (Stages 0..7): Gain-spaced PSD P_cold -> P_warm ramp (`w: 0.25 -> 1.00`)
  - Phase 2 (Stages 8..10): Full real-robot physical & sensing DR (`dr_scale: 0.35 -> 0.65 -> 1.00`)
Each stage requires stage-pure 25s survival >= 85-88% and RealTwin >= 85-88% before
advancing, keeping EV > 0.90 and avoiding 0%-survival local minima.
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

from early_stability_reward import EARLY_REWARD_FIELDS, EarlyStabilityRewardEnv
from sim2real_r5 import (
    DEFAULT_STAGES,
    SIM2REAL_REWARD_FIELDS,
    Sim2RealPhysicsAndObsEnv,
    Sim2RealRehearsalEnv,
    Sim2RealStartupProgress,
    Sim2RealWindSamplingEnv,
    configure_sim2real_cfg,
    sim2real_rehearsal_mask,
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
from single5_runtime import REVISION, Single5Env, actor_parameter_count, adam_factory, install_pose_mode
from targeted_reward import FIELDS as TARGETED_FIELDS
from termination_diagnostics import TERMINATION_FIELDS
from training_snapshot import (
    prune_owned_snapshots,
    save_snapshot,
)
from weak_release_floor import FIELDS as RELEASE_FLOOR_FIELDS
from feature_training import assert_supported_actor
from wind_features import WindFeatureEnv


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
        + REHEARSAL_FIELDS + EARLY_REWARD_FIELDS + SIM2REAL_REWARD_FIELDS + goal_fields + extra_fields
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=HERE / "TRAIN_PLAN_S2R.json")
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

    s2r_cfg = plan["sim2real"]
    stages = tuple(plan.get("curriculum_stages", DEFAULT_STAGES))
    stage_idx = 0
    cur_stage = stages[stage_idx]

    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
        for k, v in list(stack.parent["args"].items()):
            if isinstance(v, (str, Path)) and "/home/yangzhe/Project/pingpong_controller/" in str(v):
                new_s = str(REPO / str(v).replace("/home/yangzhe/Project/pingpong_controller/", ""))
                stack.parent["args"][k] = Path(new_s) if isinstance(v, Path) else new_s
        stack.cfg = configure_sim2real_cfg(stack.cfg, s2r_cfg)
        args = argparse.Namespace(**stack.parent["args"])
        n_envs = int(plan.get("n_envs", 4096))
        n_steps = int(plan.get("n_steps", 128))
        minibatch_size = int(plan["ppo"].get("minibatch_size", 65536))
        batch_size = n_envs * n_steps
        max_minibatches = int(plan["ppo"].get("update_epochs", 4)) * max(1, batch_size // minibatch_size)
        args.n_envs, args.n_steps, args.minibatch_size = n_envs, n_steps, minibatch_size
        args.seed = plan["seed"]
        args.wind_training_mix = stack.wind_training_mix

    install_pose_mode()
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=stack.cfg)
    wind_env = Sim2RealWindSamplingEnv(
        CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
        half_angle_deg=float(plan["wind_domain"]["x_axis_half_angle_deg"]),
        max_speed=float(plan["wind_domain"]["max_speed_m_s"]),
    )
    real_twin_np = wind_env.real_twin_np

    env = Single5Env(
        wind_env,
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
    reh_mask = sim2real_rehearsal_mask(n_envs, fraction=float(rehearsal_cfg["fraction"]))
    env.inner = StratifiedWideOnsetEnv(env.inner.inner, rehearsal_mask=reh_mask, onset_cfg=onset_cfg, enabled=True)
    env = WindFeatureEnv(env)
    env = Sim2RealPhysicsAndObsEnv(
        env,
        real_twin_mask=real_twin_np,
        s2r_cfg=s2r_cfg,
        init_warm_w=float(cur_stage["warm_w"]),
        init_dr_scale=float(cur_stage["dr_scale"]),
        init_stage_id=int(cur_stage["id"]),
    )
    early_rew_cfg = plan.get("early_stability_reward", {})
    env = EarlyStabilityRewardEnv(env, early_rew_cfg)
    env = Sim2RealRehearsalEnv(
        env,
        fraction=float(rehearsal_cfg["fraction"]),
        seconds=float(rehearsal_cfg["seconds"]),
        stagger_min_steps=int(rehearsal_cfg["stagger_min_steps"]),
        stagger_seed=int(rehearsal_cfg["stagger_seed"]),
    )
    assert (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) == (120, 453, 7, 240000)

    parent_path = REPO / plan["parent_checkpoint"]
    parent_sha = sha256_file(parent_path)
    with parent_path.open("rb") as f:
        p_saved = pickle.load(f)
    assert_supported_actor(p_saved["params"])
    p_params = jax.tree.map(jnp.asarray, p_saved["params"])
    target_log_std = float(plan["ppo"]["min_log_std"])
    p_params = {**p_params, "log_std": jnp.full_like(p_params["log_std"], target_log_std)}

    p_opt = jax.tree.map(jnp.asarray, p_saved["opt"])
    if plan["feature_optimizer"].get("reset_all_moments_on_start", True):
        zero_all = jax.tree.map(jnp.zeros_like, p_params)
        p_opt = p_opt._replace(m=zero_all, v=zero_all, actor_t=jnp.asarray(0, dtype=jnp.int32))
        print("[s2r_v2] Reset actor+critic Adam moments (m=0, v=0, actor_t=0).", flush=True)

    learner = stack.ppo.TrainState(p_params, p_opt)
    completed = int(p_saved["actual_updates"])
    true_step = int(p_saved["step"])
    init_actor_digest = p_saved.get("initial_single_actor_digest", parent_sha)
    print(
        f"[s2r_v2] Loaded {parent_path.name} (u{completed}, 120-D 1.6m/s +/-5deg drop-in, "
        f"{len(stages)}-stage curriculum starting at S{cur_stage['id']}:{cur_stage['name']} "
        f"[warm_w={cur_stage['warm_w']:.3f}, dr={cur_stage['dr_scale']:.2f}], "
        f"sigma={math.exp(target_log_std):.4f})",
        flush=True,
    )

    initial_learner_params = jax.tree.map(jnp.array, learner.params)
    stack.single_actor_anchor_params = initial_learner_params
    stack.bootstrap_stage_anchor = initial_learner_params

    init_rng, reset_key = jax.random.split(jax.random.PRNGKey(plan["seed"] + 6247))
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
        schema="v325_single5_early_stability_v1",
        mode="direct",
        parent_checkpoint=plan["parent_checkpoint"],
        parent_sha256=parent_sha,
        actor_critic_action=[120, 453, 7],
        pose_limit_deg=5.0,
        max_speed_m_s=1.6,
        plan_sha256=plan_sha,
        revision=plan["revision"],
        n_envs=n_envs,
        n_steps=n_steps,
        minibatch_size=minibatch_size,
        sim2real=s2r_cfg,
        curriculum_stages=list(stages),
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
        dir=os.environ.get("WANDB_DIR", "/dev/shm/wandb_wind_s2r_v2"),
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
    startup_tracker = Sim2RealStartupProgress(
        seconds=float(rehearsal_cfg["seconds"]),
        capacity=int(rehearsal_cfg["metric_capacity"]),
        active_stage_id=int(cur_stage["id"]),
    )
    early_quality = EarlyHitQualityTracker(window_updates=16)

    status = {}
    score = (0, 0.0, 0.0)
    best_pareto = (-1, -1.0, -1.0, -1.0)
    started = time.monotonic()
    session = 0
    stage_updates = 0
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
            s2r_stage_id=int(cur_stage["id"]),
            s2r_stage_name=str(cur_stage["name"]),
            s2r_warm_w=float(cur_stage["warm_w"]),
            s2r_dr_scale=float(cur_stage["dr_scale"]),
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
                dict(startup_tracker=startup_tracker, early_quality=early_quality, stage_idx=stage_idx, stage_updates=stage_updates),
                contract,
                dict(actual_updates=completed, true_step=true_step, run_id=opts.run_id, stage_idx=stage_idx),
            )
            prune_owned_snapshots(opts.output, path)

    print(
        f"[s2r_v2] Starting from u{completed} (n_envs={n_envs}, n_steps={n_steps}, mb={minibatch_size}, "
        f"rehearsal={env.n_rehearsal}x{rehearsal_cfg['seconds']}s, long={env.n_long}x1200s, "
        f"realTwin={int(real_twin_np.sum())} envs, wandb={opts.wandb_mode})",
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
                assert float(np.max(metrics["wind/speed_m_s"])) <= 1.6001
                np.testing.assert_allclose(metrics["pose/outer_limit_deg"], 5.0, atol=1e-4)

                reh_metrics, reh_dones = masked_rollout(metrics, dones, rehearsal_mask)
                long_metrics, long_dones = masked_rollout(metrics, dones, long_mask)
                startup_tracker.add(reh_metrics, reh_dones)
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
                calm_mask = metrics["wind/mode"] < 0.5
                calm_tilt_p95 = float(np.percentile(metrics["s2r/tilt_deg"][calm_mask], 95)) if np.any(calm_mask) else 0.0

                status = dict(
                    update=completed,
                    true_step=true_step,
                    s2r_stage_id=int(cur_stage["id"]),
                    s2r_stage_name=str(cur_stage["name"]),
                    s2r_warm_w=float(cur_stage["warm_w"]),
                    s2r_dr_scale=float(cur_stage["dr_scale"]),
                    stage_updates=stage_updates,
                    opt_t=int(learner.opt.t),
                    actor_t=int(learner.opt.actor_t),
                    in_critic_warmup=int(in_warmup),
                    explained_variance=float(explained),
                    return_variance=float(variance),
                    calm_tilt_p95_deg=calm_tilt_p95,
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

                tag = "CRIT-WU" if in_warmup else cur_stage["name"]
                print(
                    f"u{completed:05d} | {upd_sec:5.1f}s | {tag:8s} | EV={float(explained):.4f} | "
                    f"kl={loss['learning_stability/sampled_max_step_kl']:.5f}(s={loss['learning_stability/step_scale']:.2f}) | "
                    f"cTilt95={calm_tilt_p95:4.2f}d | "
                    f"reh25(n={startup_summary['n']:4d} surv={startup_summary['survival']*100:5.1f}% "
                    f"twin={startup_summary['real_twin_survival']*100:5.1f}% "
                    f"clnP={startup_summary['clean_phys_survival']*100:5.1f}% "
                    f"peak={startup_summary['peak_survival']*100:5.1f}% "
                    f"dir08={startup_summary['dir08_survival']*100:5.1f}% "
                    f"dyn={startup_summary['dyn_survival']*100:5.1f}%) | "
                    f"s={startup_summary['mean_seconds']:5.2f}s h={startup_summary['mean_hits']:4.1f} | "
                    f"term={status['reh_terms']}/{status['long_terms']}",
                    flush=True,
                )

                twin_s = float(startup_summary["real_twin_survival"])
                clean_s = float(startup_summary["clean_phys_survival"])
                peak_s = float(startup_summary["peak_survival"])
                dir08_s = float(startup_summary["dir08_survival"])
                dyn_s = float(startup_summary["dyn_survival"])
                surv_s = float(startup_summary["survival"])
                joint = round(0.35 * twin_s + 0.20 * clean_s + 0.20 * peak_s + 0.10 * dir08_s + 0.15 * dyn_s, 4)
                score = (int(cur_stage["id"]), joint, surv_s)

                if not in_warmup and startup_summary["n"] >= 1024:
                    cand_pareto = (int(cur_stage["id"]), joint, twin_s, -early_q["h6_15_clean_center_rms_mm"])
                    if cand_pareto > best_pareto:
                        best_pareto = cand_pareto
                        bp = checkpoint(f"best_pareto_S{cur_stage['id']:02d}")
                        for old_bp in opts.output.glob("best_pareto_*.pkl"):
                            if old_bp.resolve() != bp.resolve():
                                old_bp.unlink(missing_ok=True)
                        print(
                            f"  [BEST S2R] saved {bp.name} stage={cur_stage['name']} joint={joint*100:.2f}% "
                            f"twin={twin_s*100:.1f}% clnP={clean_s*100:.1f}% peak={peak_s*100:.1f}% "
                            f"dir08={dir08_s*100:.1f}% dyn={dyn_s*100:.1f}% cTilt95={calm_tilt_p95:.2f}deg",
                            flush=True,
                        )

                # Zero-recompile curriculum stage advancement check
                if (
                    not in_warmup
                    and stage_idx + 1 < len(stages)
                    and stage_updates >= int(cur_stage["min_u"])
                    and startup_summary["n"] >= int(cur_stage["min_n"])
                    and surv_s >= float(cur_stage["req_surv"])
                    and clean_s >= float(cur_stage["req_clean"])
                    and twin_s >= float(cur_stage["req_twin"])
                ):
                    if cur_stage.get("save_clear", False):
                        mp = checkpoint(f"milestone_S{cur_stage['id']:02d}_{cur_stage['name']}")
                        print(f"  [MILESTONE SAVED] {mp.name}", flush=True)
                    stage_idx += 1
                    cur_stage = stages[stage_idx]
                    stage_updates = 0
                    startup_tracker.set_stage(int(cur_stage["id"]))
                    new_s2r_core = runner.env_state.core._replace(
                        curr_warm_w=jnp.asarray(float(cur_stage["warm_w"]), dtype=jnp.float32),
                        curr_dr_scale=jnp.asarray(float(cur_stage["dr_scale"]), dtype=jnp.float32),
                        curr_stage_id=jnp.asarray(int(cur_stage["id"]), dtype=jnp.int32),
                    )
                    runner = runner._replace(env_state=runner.env_state._replace(core=new_s2r_core))
                    print(
                        f"  >>> [STAGE ADVANCE] u{completed:05d}: cleared S{stage_idx-1:02d} -> "
                        f"S{cur_stage['id']:02d}:{cur_stage['name']} "
                        f"(warm_w={cur_stage['warm_w']:.3f}, dr_scale={cur_stage['dr_scale']:.2f}, ZERO XLA recompile) <<<",
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
        print(f"[s2r_v2] Completed at u{completed} (stage S{cur_stage['id']:02d}:{cur_stage['name']})", flush=True)


if __name__ == "__main__":
    main()
