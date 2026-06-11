"""Train juggling with MJX/JAX PPO.

For the full multi-stage schedule use ``train_juggle_mjx_curriculum.py``.  This
script is a compact single-stage entrypoint for quick MJX PPO experiments.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv


LOG_2PI = float(np.log(2.0 * np.pi))


class OptimState(NamedTuple):
    m: object
    v: object
    t: jax.Array


class TrainState(NamedTuple):
    params: object
    opt: OptimState


class RunnerState(NamedTuple):
    env_state: object
    obs: jax.Array
    rng: jax.Array
    running_return: jax.Array
    running_length: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    new_hit: jax.Array
    hit_count: jax.Array
    metrics: dict[str, jax.Array]


class PpoBatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    old_logp: jax.Array
    advantages: jax.Array
    returns: jax.Array
    old_values: jax.Array


def init_layer(key: jax.Array, in_dim: int, out_dim: int, scale: float = np.sqrt(2.0)) -> dict[str, jax.Array]:
    shape = (int(in_dim), int(out_dim))
    if in_dim < out_dim:
        a = jax.random.normal(key, (out_dim, in_dim), dtype=jnp.float32)
        q, r = jnp.linalg.qr(a)
        sign = jnp.sign(jnp.diag(r))
        q = (q * sign).T
    else:
        a = jax.random.normal(key, shape, dtype=jnp.float32)
        q, r = jnp.linalg.qr(a)
        sign = jnp.sign(jnp.diag(r))
        q = q * sign
    w = q[: shape[0], : shape[1]] * float(scale)
    b = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": w, "b": b}


def init_mlp(key: jax.Array, in_dim: int, hidden_dim: int, out_dim: int, out_scale: float) -> dict[str, dict[str, jax.Array]]:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "l1": init_layer(k1, in_dim, hidden_dim),
        "l2": init_layer(k2, hidden_dim, hidden_dim),
        "out": init_layer(k3, hidden_dim, out_dim, out_scale),
    }


def init_params(key: jax.Array, obs_dim: int, act_dim: int, hidden_dim: int) -> dict[str, object]:
    k_pi, k_v = jax.random.split(key)
    return {
        "pi": init_mlp(k_pi, obs_dim, hidden_dim, act_dim, 0.01),
        "v": init_mlp(k_v, obs_dim, hidden_dim, 1, 1.0),
        "log_std": jnp.full((act_dim,), -0.5, dtype=jnp.float32),
    }


def apply_mlp(params: dict[str, dict[str, jax.Array]], obs: jax.Array) -> jax.Array:
    x = jnp.tanh(obs @ params["l1"]["w"] + params["l1"]["b"])
    x = jnp.tanh(x @ params["l2"]["w"] + params["l2"]["b"])
    return x @ params["out"]["w"] + params["out"]["b"]


def policy_value(params: dict[str, object], obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    mean = apply_mlp(params["pi"], obs)
    value = apply_mlp(params["v"], obs).squeeze(-1)
    return mean, value


def normal_logprob(action: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    inv_std = jnp.exp(-log_std)
    return -0.5 * jnp.sum(((action - mean) * inv_std) ** 2 + 2.0 * log_std + LOG_2PI, axis=-1)


def normal_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * (1.0 + LOG_2PI))


def adam_init(params) -> OptimState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return OptimState(m=zeros, v=zeros, t=jnp.asarray(0, dtype=jnp.int32))


def tree_global_norm(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves]))


def adam_step(
    params,
    grads,
    opt: OptimState,
    learning_rate: float,
    max_grad_norm: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-5,
) -> tuple[object, OptimState, jax.Array]:
    grad_norm = tree_global_norm(grads)
    scale = jnp.minimum(1.0, float(max_grad_norm) / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    t = opt.t + 1
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1.0 - beta1) * g, opt.m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: beta2 * v_ + (1.0 - beta2) * (g * g), opt.v, grads)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**t), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**t), v)
    params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - float(learning_rate) * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, OptimState(m=m, v=v, t=t), grad_norm


def ppo_loss(
    params,
    batch: PpoBatch,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    mean, value = policy_value(params, batch.obs)
    log_std = params["log_std"]
    logp = normal_logprob(batch.action, mean, log_std)
    ratio = jnp.exp(logp - batch.old_logp)
    pg1 = ratio * batch.advantages
    pg2 = jnp.clip(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * batch.advantages
    policy_loss = -jnp.mean(jnp.minimum(pg1, pg2))
    value_loss = 0.5 * jnp.mean((batch.returns - value) ** 2)
    entropy = normal_entropy(log_std)
    loss = policy_loss + float(vf_coef) * value_loss - float(ent_coef) * entropy
    approx_kl = jnp.mean(batch.old_logp - logp)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > float(clip_range)).astype(jnp.float32))
    aux = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
    }
    return loss, aux


def compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    def scan_fn(carry, xs):
        next_adv, next_value = carry
        reward, done, value = xs
        nonterminal = 1.0 - done.astype(jnp.float32)
        delta = reward + float(gamma) * nonterminal * next_value - value
        adv = delta + float(gamma) * float(gae_lambda) * nonterminal * next_adv
        return (adv, value), adv

    init = (jnp.zeros_like(last_value), last_value)
    _, adv_rev = jax.lax.scan(scan_fn, init, (rewards[::-1], dones[::-1], values[::-1]))
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns


def flatten_time_env(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def make_train_fns(
    env: MjxJuggleEnv,
    n_steps: int,
    update_epochs: int,
    minibatch_size: int,
    gamma: float,
    gae_lambda: float,
    learning_rate: float,
    clip_range: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
):
    batch_size = env.n_envs * int(n_steps)
    num_minibatches = max(1, batch_size // int(minibatch_size))
    used_batch_size = num_minibatches * int(minibatch_size)

    def collect_rollout(params, runner: RunnerState) -> tuple[RunnerState, Transition]:
        def rollout_step(carry: RunnerState, _):
            env_state, obs, rng, running_return, running_length = carry
            rng, action_key, reset_key = jax.random.split(rng, 3)
            mean, value = policy_value(params, obs)
            log_std = params["log_std"]
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
            env_action = jnp.clip(raw_action, -1.0, 1.0)
            logp = normal_logprob(raw_action, mean, log_std)
            next_env_state, next_obs, reward, done, metrics = env.step(env_state, env_action)

            completed_return = running_return + reward
            completed_length = running_length + 1
            reset_keys = jax.random.split(reset_key, env.n_envs)
            next_env_state, next_obs = env.reset_done(next_env_state, next_obs, done, reset_keys)
            next_running_return = jnp.where(done, 0.0, completed_return)
            next_running_length = jnp.where(done, 0, completed_length)

            transition = Transition(
                obs=obs,
                action=raw_action,
                logp=logp,
                value=value,
                reward=reward,
                done=done,
                episode_return=completed_return,
                episode_length=completed_length,
                new_hit=metrics["new_hit"],
                hit_count=metrics["hit_count"],
                metrics=metrics,
            )
            return (
                RunnerState(next_env_state, next_obs, rng, next_running_return, next_running_length),
                transition,
            )

        return jax.lax.scan(rollout_step, runner, None, length=int(n_steps))

    def update(train_state: TrainState, runner: RunnerState, transitions: Transition) -> tuple[TrainState, dict[str, jax.Array]]:
        _, last_value = policy_value(train_state.params, runner.obs)
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.done,
            transitions.value,
            last_value,
            gamma,
            gae_lambda,
        )
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        batch = PpoBatch(
            obs=flatten_time_env(transitions.obs),
            action=flatten_time_env(transitions.action),
            old_logp=flatten_time_env(transitions.logp),
            advantages=flatten_time_env(advantages),
            returns=flatten_time_env(returns),
            old_values=flatten_time_env(transitions.value),
        )

        def take_minibatch(b: PpoBatch, idx: jax.Array) -> PpoBatch:
            return jax.tree_util.tree_map(lambda x: x[idx], b)

        def update_minibatch(state: TrainState, idx: jax.Array):
            mini = take_minibatch(batch, idx)
            (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                state.params,
                mini,
                clip_range,
                vf_coef,
                ent_coef,
            )
            params, opt, grad_norm = adam_step(
                state.params,
                grads,
                state.opt,
                learning_rate,
                max_grad_norm,
            )
            aux = dict(aux)
            aux["grad_norm"] = grad_norm
            aux["loss"] = loss
            return TrainState(params=params, opt=opt), aux

        def update_epoch(carry, epoch_key):
            perm = jax.random.permutation(epoch_key, batch_size)[:used_batch_size]
            mb_idx = perm.reshape((num_minibatches, int(minibatch_size)))
            return jax.lax.scan(update_minibatch, carry, mb_idx)

        rng, epoch_rng = jax.random.split(runner.rng)
        epoch_keys = jax.random.split(epoch_rng, int(update_epochs))
        train_state, aux = jax.lax.scan(update_epoch, train_state, epoch_keys)
        aux_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), aux)
        aux_mean["explained_var"] = 1.0 - jnp.var(flatten_time_env(returns) - batch.old_values) / (
            jnp.var(flatten_time_env(returns)) + 1e-8
        )
        return train_state, aux_mean

    return jax.jit(collect_rollout), jax.jit(update)


def save_checkpoint(
    path: Path,
    train_state: TrainState,
    args: argparse.Namespace,
    env: MjxJuggleEnv,
    step: int,
    extra: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jax.device_get(train_state.params),
        "opt": jax.device_get(train_state.opt),
        "step": int(step),
        "args": vars(args),
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "env_cfg": env.cfg.__dict__,
        "xml": str(env.xml_path),
        "mjx_xml": str(env.mjx_xml),
    }
    if extra:
        payload.update(extra)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f)
    tmp_path.replace(path)


def append_progress(path: Path, row: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Train Stage 1a juggling with MJX/JAX PPO.")
    p.add_argument("--xml", type=Path, default=here / "moz1_pd.xml")
    p.add_argument("--save-dir", type=Path, default=here.parents[1] / "outputs" / "rl_sim" / "logs_mjx_stage1a")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-envs", type=int, default=1024)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument(
        "--n-steps",
        type=int,
        default=64,
        help="Rollout steps per env before each PPO update. 64 is fast; 128-256 is usually better for juggling credit assignment.",
    )
    p.add_argument("--minibatch-size", type=int, default=8192)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--save-every-updates", type=int, default=10)
    p.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases.")
    p.add_argument("--wandb-project", type=str, default="pingpong-mjx")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument(
        "--sb3-parity",
        action="store_true",
        help="Override rollout/batch/network defaults to the CPU SB3 PPO reference: 8 envs, 2048 steps, batch 256, 10 epochs, hidden 64.",
    )
    args = p.parse_args()
    if args.sb3_parity:
        args.n_envs = 8
        args.n_steps = 2048
        args.minibatch_size = 256
        args.update_epochs = 10
        args.hidden_dim = 64
    return args


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = None

    print(f"[mjx_ppo] JAX devices: {jax.devices()}")
    env = MjxJuggleEnv(args.xml, n_envs=args.n_envs, cfg=MjxJuggleConfig(domain_randomization=False, arm_action_limiter=True))
    print(f"[mjx_ppo] MJX XML: {env.mjx_xml}")
    print(f"[mjx_ppo] n_envs={args.n_envs}, n_steps={args.n_steps}, batch={args.n_envs * args.n_steps}")
    print(f"[mjx_ppo] episode_max_steps={env.max_steps}, dt={env.dt:.4f}s, horizon={env.max_steps * env.dt:.2f}s")

    if args.wandb:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise SystemExit("wandb is not installed. Install with: python -m pip install wandb") from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            mode=args.wandb_mode,
            config={
                **vars(args),
                "mjx_xml": str(env.mjx_xml),
                "dt": env.dt,
                "max_steps": env.max_steps,
                "obs_dim": env.obs_dim,
                "act_dim": env.act_dim,
                "jax_devices": [str(d) for d in jax.devices()],
            },
        )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key, params_key = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_key, args.n_envs)
    env_state, obs = jax.jit(env.reset)(reset_keys)
    params = init_params(params_key, env.obs_dim, env.act_dim, args.hidden_dim)
    train_state = TrainState(params=params, opt=adam_init(params))
    runner = RunnerState(
        env_state=env_state,
        obs=obs,
        rng=rng,
        running_return=jnp.zeros((args.n_envs,), dtype=jnp.float32),
        running_length=jnp.zeros((args.n_envs,), dtype=jnp.int32),
    )

    collect_rollout, update = make_train_fns(
        env=env,
        n_steps=args.n_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
    )

    total_updates = max(1, int(args.total_steps) // (int(args.n_envs) * int(args.n_steps)))
    progress_path = args.save_dir / "progress.csv"
    global_step = 0

    for update_idx in range(1, total_updates + 1):
        t0 = time.perf_counter()
        runner, transitions = collect_rollout(train_state.params, runner)
        train_state, losses = update(train_state, runner, transitions)
        jax.block_until_ready(losses["loss"])
        elapsed = time.perf_counter() - t0
        global_step += args.n_envs * args.n_steps

        done = np.asarray(jax.device_get(transitions.done)).astype(bool)
        ep_ret = np.asarray(jax.device_get(transitions.episode_return))
        ep_len = np.asarray(jax.device_get(transitions.episode_length))
        hit_count = np.asarray(jax.device_get(transitions.hit_count))
        rollout_metrics = {
            key: float(jnp.mean(value))
            for key, value in jax.device_get(transitions.metrics).items()
            if value.dtype.kind in "fbiu"
        }
        done_count = int(done.sum())
        mean_return = float(ep_ret[done].mean()) if done_count > 0 else float("nan")
        mean_len = float(ep_len[done].mean()) if done_count > 0 else float("nan")
        mean_hits = float(hit_count[done].mean()) if done_count > 0 else float("nan")
        sps = float(args.n_envs * args.n_steps / max(elapsed, 1e-9))
        loss_host = {k: float(v) for k, v in jax.device_get(losses).items()}
        row = {
            "update": update_idx,
            "global_step": global_step,
            "sps": sps,
            "episodes": done_count,
            "mean_return": mean_return,
            "mean_len": mean_len,
            "mean_hits": mean_hits,
            **loss_host,
            **rollout_metrics,
        }
        append_progress(progress_path, row)
        if wandb_run is not None:
            import wandb

            wandb.log(row, step=global_step)
        print(
            f"[mjx_ppo] update={update_idx}/{total_updates} "
            f"step={global_step} sps={sps:,.0f} episodes={done_count} "
            f"return={mean_return:.3f} hits={mean_hits:.2f} "
            f"loss={loss_host['loss']:.4f} kl={loss_host['approx_kl']:.5f}"
        )

        if update_idx % max(1, int(args.save_every_updates)) == 0:
            save_checkpoint(args.save_dir / "mjx_ppo_last.pkl", train_state, args, env, global_step)

    save_checkpoint(args.save_dir / "mjx_ppo_last.pkl", train_state, args, env, global_step)
    if wandb_run is not None:
        import wandb

        wandb.save(str(args.save_dir / "mjx_ppo_last.pkl"))
        wandb.save(str(progress_path))
        wandb_run.finish()
    print(f"[mjx_ppo] finished: {args.save_dir}")


if __name__ == "__main__":
    main()
