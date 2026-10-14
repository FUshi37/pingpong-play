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
PPO_KL_BACKTRACK_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
PPO_ABSOLUTE_KL_BACKTRACK_SCALES = (
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
    0.001953125,
    0.0009765625,
)

# CMDP costs deliberately describe episode outcomes rather than additional
# reward terms.  Keeping stable channel identities makes their value heads,
# dual variables, checkpoints, and logs auditable across continuation runs.
CMDP_COST_NAMES = (
    "failure",
    "shortfall",
    "ball_too_low",
    "ball_too_high",
    "racket_too_low",
    "racket_too_high",
)


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
    critic_obs: jax.Array
    rng: jax.Array
    running_return: jax.Array
    running_length: jax.Array


class Transition(NamedTuple):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    timeout_value: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    new_hit: jax.Array
    hit_count: jax.Array
    metrics: dict[str, jax.Array]


class PpoBatch(NamedTuple):
    obs: jax.Array
    critic_obs: jax.Array
    action: jax.Array
    old_logp: jax.Array
    advantages: jax.Array
    returns: jax.Array
    old_values: jax.Array


class CmdpBatch(NamedTuple):
    cost_returns: jax.Array


def backtrack_params_to_kl_limit(
    previous_params,
    candidate_params,
    kl_fn,
    max_kl: float,
    scales: tuple[float, ...] = PPO_ABSOLUTE_KL_BACKTRACK_SCALES,
):
    """Select the largest interpolated parameter step inside an absolute KL.

    The old replay guard discarded a complete PPO update whenever the final
    candidate crossed the source-policy KL boundary.  Once an actor reached
    that boundary, even a mildly useful update was therefore rejected in its
    entirety.  Backtracking preserves the same hard absolute limit while
    accepting the largest safe fraction of the already-computed update.

    ``kl_fn`` is intentionally supplied by the caller so this helper can be
    used for frozen replay-state KLs without coupling it to a policy layout.
    The returned ``found`` flag concerns a *positive* step; the unchanged
    previous parameters remain the fallback when no outward fraction is safe.
    """

    scale_values = jnp.asarray(scales, dtype=jnp.float32)

    def try_scale(carry, step_scale):
        selected_params, selected_kl, selected_scale, found = carry
        scaled_params = jax.tree_util.tree_map(
            lambda previous, candidate: previous
            + step_scale * (candidate - previous),
            previous_params,
            candidate_params,
        )
        scaled_kl = kl_fn(scaled_params)
        choose = (~found) & (scaled_kl <= float(max_kl))
        selected_params = jax.tree_util.tree_map(
            lambda selected, candidate: jnp.where(choose, candidate, selected),
            selected_params,
            scaled_params,
        )
        selected_kl = jnp.where(choose, scaled_kl, selected_kl)
        selected_scale = jnp.where(choose, step_scale, selected_scale)
        return (
            selected_params,
            selected_kl,
            selected_scale,
            found | choose,
        ), scaled_kl

    initial = (
        previous_params,
        kl_fn(previous_params),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(False),
    )
    (selected_params, selected_kl, selected_scale, found), trial_kls = (
        jax.lax.scan(try_scale, initial, scale_values)
    )
    return selected_params, selected_kl, selected_scale, found, trial_kls


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


def init_params(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int,
    critic_obs_dim: int | None = None,
) -> dict[str, object]:
    k_pi, k_v = jax.random.split(key)
    value_obs_dim = int(obs_dim if critic_obs_dim is None else critic_obs_dim)
    return {
        "pi": init_mlp(k_pi, obs_dim, hidden_dim, act_dim, 0.01),
        "v": init_mlp(k_v, value_obs_dim, hidden_dim, 1, 1.0),
        "log_std": jnp.full((act_dim,), -0.5, dtype=jnp.float32),
    }


def apply_mlp(params: dict[str, dict[str, jax.Array]], obs: jax.Array) -> jax.Array:
    x = jnp.tanh(obs @ params["l1"]["w"] + params["l1"]["b"])
    x = jnp.tanh(x @ params["l2"]["w"] + params["l2"]["b"])
    return x @ params["out"]["w"] + params["out"]["b"]


def policy_components(
    params: dict[str, object],
    obs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return composite mean, frozen-teacher mean, and bounded correction.

    Legacy checkpoints contain only ``pi`` and therefore return the original
    policy unchanged.  Residual checkpoints additionally contain
    ``teacher_pi`` and ``residual_action_scale``; ``pi`` then denotes the
    trainable residual MLP.  Keeping this dispatch in the shared policy helper
    makes training, validation, and deployment consume identical actions.
    """

    residual_or_mean = apply_mlp(params["pi"], obs)
    if "teacher_pi" not in params:
        zeros = jnp.zeros_like(residual_or_mean)
        return residual_or_mean, zeros, zeros
    teacher_mean = jax.lax.stop_gradient(apply_mlp(params["teacher_pi"], obs))
    residual_scale = jax.lax.stop_gradient(jnp.asarray(params["residual_action_scale"]))
    correction = residual_scale * jnp.tanh(residual_or_mean)
    return teacher_mean + correction, teacher_mean, correction


def policy_mean(params: dict[str, object], obs: jax.Array) -> jax.Array:
    mean, _, _ = policy_components(params, obs)
    return mean


def value_fn(params: dict[str, object], critic_obs: jax.Array) -> jax.Array:
    return apply_mlp(params["v"], critic_obs).squeeze(-1)


def cmdp_cost_value_fn(
    params: dict[str, object], critic_obs: jax.Array
) -> jax.Array:
    """Predict the undiscounted episodic costs used by constrained PPO."""

    return apply_mlp(params["cmdp_cost_v"], critic_obs)


def initialize_cmdp_train_state(
    train_state: TrainState,
    key: jax.Array,
    *,
    critic_obs_dim: int,
    hidden_dim: int,
    initial_duals: tuple[float, ...],
) -> TrainState:
    """Append zero-output cost critics while preserving legacy PPO state.

    V29 owns a useful actor, reward critic, and nonzero Adam moments.  A CMDP
    continuation must not reset those leaves merely because the source did
    not contain cost heads.  New cost-head moments start at zero while the
    optimizer step counter and every existing moment remain exact.
    """

    if len(initial_duals) != len(CMDP_COST_NAMES):
        raise ValueError(
            "CMDP initial dual count must match the registered cost channels"
        )
    if "cmdp_cost_v" in train_state.params or "cmdp_dual" in train_state.params:
        if "cmdp_cost_v" not in train_state.params or "cmdp_dual" not in train_state.params:
            raise ValueError("checkpoint contains an incomplete CMDP parameter set")
        output_dim = int(train_state.params["cmdp_cost_v"]["out"]["b"].shape[0])
        dual_dim = int(jnp.asarray(train_state.params["cmdp_dual"]).shape[0])
        if output_dim != len(CMDP_COST_NAMES) or dual_dim != len(CMDP_COST_NAMES):
            raise ValueError("checkpoint CMDP channel count is incompatible")
        return train_state

    cost_value = init_mlp(
        key,
        int(critic_obs_dim),
        int(hidden_dim),
        len(CMDP_COST_NAMES),
        1.0,
    )
    # An arbitrary initial risk baseline would contaminate the first actor
    # update.  Hidden features may learn immediately, but V_c(s)=0 exactly at
    # migration and the output layer is fitted only from observed outcomes.
    cost_out = dict(cost_value["out"])
    cost_out["w"] = jnp.zeros_like(cost_out["w"])
    cost_out["b"] = jnp.zeros_like(cost_out["b"])
    cost_value = dict(cost_value)
    cost_value["out"] = cost_out

    params = dict(train_state.params)
    params["cmdp_cost_v"] = cost_value
    params["cmdp_dual"] = jnp.asarray(initial_duals, dtype=jnp.float32)

    zero_cost_value = jax.tree_util.tree_map(jnp.zeros_like, cost_value)
    zero_dual = jnp.zeros((len(CMDP_COST_NAMES),), dtype=jnp.float32)
    opt_m = dict(train_state.opt.m)
    opt_v = dict(train_state.opt.v)
    opt_m["cmdp_cost_v"] = zero_cost_value
    opt_v["cmdp_cost_v"] = zero_cost_value
    opt_m["cmdp_dual"] = zero_dual
    opt_v["cmdp_dual"] = zero_dual
    opt = OptimState(m=opt_m, v=opt_v, t=train_state.opt.t)
    return TrainState(params=params, opt=opt)


def policy_value(
    params: dict[str, object],
    obs: jax.Array,
    critic_obs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if critic_obs is None:
        critic_obs = obs
    mean = policy_mean(params, obs)
    value = value_fn(params, critic_obs)
    return mean, value


def normal_logprob(action: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    inv_std = jnp.exp(-log_std)
    return -0.5 * jnp.sum(((action - mean) * inv_std) ** 2 + 2.0 * log_std + LOG_2PI, axis=-1)


def normal_entropy(log_std: jax.Array) -> jax.Array:
    return jnp.sum(log_std + 0.5 * (1.0 + LOG_2PI))


def diagonal_gaussian_kl(
    old_mean: jax.Array,
    old_log_std: jax.Array,
    new_mean: jax.Array,
    new_log_std: jax.Array,
) -> jax.Array:
    """Mean analytic KL(old || new) for diagonal Gaussian policies."""

    old_var = jnp.exp(2.0 * old_log_std)
    new_var = jnp.exp(2.0 * new_log_std)
    per_sample = jnp.sum(
        new_log_std
        - old_log_std
        + (old_var + jnp.square(old_mean - new_mean)) / (2.0 * new_var)
        - 0.5,
        axis=-1,
    )
    return jnp.mean(per_sample)


def effective_log_std(
    log_std: jax.Array,
    min_log_std: float | None,
    max_log_std: float | None = None,
) -> jax.Array:
    """Return the exploration scale used by rollout and PPO likelihoods.

    A trainable Gaussian scale can otherwise drift toward zero during very
    long curriculum stages.  Applying the same floor to action sampling and
    likelihood evaluation keeps the PPO old/new distributions consistent.
    """

    result = log_std
    if min_log_std is not None:
        result = jnp.maximum(
            result, jnp.asarray(min_log_std, dtype=log_std.dtype)
        )
    if max_log_std is not None:
        result = jnp.minimum(
            result, jnp.asarray(max_log_std, dtype=log_std.dtype)
        )
    return result


def project_policy_log_std(
    params,
    min_log_std: float | None,
    max_log_std: float | None = None,
):
    """Project the stored policy scale to the configured exploration interval."""

    if min_log_std is None and max_log_std is None:
        return params
    projected = dict(params)
    projected["log_std"] = effective_log_std(
        params["log_std"], min_log_std, max_log_std
    )
    return projected


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
    reference_params=None,
    actor_anchor_kl_coef: float = 0.0,
    actor_anchor_obs: jax.Array | None = None,
    actor_anchor_replay_kl_coef: float = 0.0,
    residual_l2_coef: float = 0.0,
    teacher_params=None,
    teacher_distill_obs: jax.Array | None = None,
    teacher_distill_coef: float = 0.0,
    teacher_distill_action_clip: float = 1.0,
    min_log_std: float | None = None,
    max_log_std: float | None = None,
    counterfactual_replay_obs: jax.Array | None = None,
    counterfactual_replay_actions: jax.Array | None = None,
    counterfactual_supervision_coef: float = 0.0,
    counterfactual_focus_tail_rows: int = 0,
    counterfactual_focus_prob: float = 0.0,
    counterfactual_vxy_weight_mode: str = "high_vxy",
    noise_invariance_clean_obs: jax.Array | None = None,
    noise_invariance_noisy_obs: jax.Array | None = None,
    noise_invariance_coef: float = 0.0,
    action_feedback_sensitivity_coef: float = 0.0,
    action_feedback_perturb_scale: float = 0.0,
    action_feedback_obs_starts: tuple[int, ...] = (),
    cmdp_batch: CmdpBatch | None = None,
    cmdp_cost_vf_coef: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    mean, _, residual_correction = policy_components(params, batch.obs)
    value = value_fn(params, batch.critic_obs)
    log_std = effective_log_std(params["log_std"], min_log_std, max_log_std)
    logp = normal_logprob(batch.action, mean, log_std)
    ratio = jnp.exp(logp - batch.old_logp)
    pg1 = ratio * batch.advantages
    pg2 = jnp.clip(ratio, 1.0 - float(clip_range), 1.0 + float(clip_range)) * batch.advantages
    policy_loss = -jnp.mean(jnp.minimum(pg1, pg2))
    value_loss = 0.5 * jnp.mean((batch.returns - value) ** 2)
    cmdp_cost_value_loss = jnp.asarray(0.0, dtype=value.dtype)
    if cmdp_batch is not None:
        cost_value = cmdp_cost_value_fn(params, batch.critic_obs)
        cmdp_cost_value_loss = 0.5 * jnp.mean(
            jnp.square(cmdp_batch.cost_returns - cost_value)
        )
    entropy = normal_entropy(log_std)
    actor_anchor_kl = jnp.asarray(0.0, dtype=mean.dtype)
    actor_anchor_current_kl = jnp.asarray(0.0, dtype=mean.dtype)
    actor_anchor_replay_kl = jnp.asarray(0.0, dtype=mean.dtype)
    actor_anchor_regularization = jnp.asarray(0.0, dtype=mean.dtype)
    residual_rms = jnp.sqrt(jnp.mean(jnp.square(residual_correction)))
    residual_abs_max = jnp.max(jnp.abs(residual_correction))
    residual_regularization = float(residual_l2_coef) * jnp.mean(
        jnp.square(residual_correction)
    )
    teacher_distill_mse = jnp.asarray(0.0, dtype=mean.dtype)
    teacher_distill_regularization = jnp.asarray(0.0, dtype=mean.dtype)
    teacher_target_clip_fraction = jnp.asarray(0.0, dtype=mean.dtype)
    counterfactual_supervision_mse = jnp.asarray(0.0, dtype=mean.dtype)
    counterfactual_supervision_regularization = jnp.asarray(0.0, dtype=mean.dtype)
    noise_invariance_mse = jnp.asarray(0.0, dtype=mean.dtype)
    noise_invariance_regularization = jnp.asarray(0.0, dtype=mean.dtype)
    action_feedback_sensitivity_mse = jnp.asarray(0.0, dtype=mean.dtype)
    action_feedback_sensitivity_regularization = jnp.asarray(
        0.0, dtype=mean.dtype
    )
    if (
        counterfactual_replay_obs is not None
        and counterfactual_replay_actions is not None
        and float(counterfactual_supervision_coef) > 0.0
    ):
        counterfactual_mean = policy_mean(params, counterfactual_replay_obs)
        counterfactual_targets = jax.lax.stop_gradient(
            jnp.clip(counterfactual_replay_actions, -1.0, 1.0)
        )
        per_sample_sq = jnp.sum(
            jnp.square(counterfactual_mean - counterfactual_targets),
            axis=-1,
        )
        if counterfactual_replay_obs.shape[-1] >= 25:
            ball_vxy = jnp.linalg.norm(
                counterfactual_replay_obs[..., 23:25],
                axis=-1,
            )
            if counterfactual_vxy_weight_mode == "low_vxy":
                sample_weight = 1.0 + jnp.clip((0.10 - ball_vxy) / 0.10, 0.0, 2.0)
            else:
                sample_weight = 1.0 + jnp.clip(ball_vxy / 0.06, 0.0, 3.0)
            counterfactual_supervision_mse = jnp.mean(sample_weight * per_sample_sq)
        else:
            counterfactual_supervision_mse = jnp.mean(per_sample_sq)
        counterfactual_supervision_regularization = (
            float(counterfactual_supervision_coef) * counterfactual_supervision_mse
        )
    if (
        noise_invariance_clean_obs is not None
        and noise_invariance_noisy_obs is not None
        and float(noise_invariance_coef) > 0.0
    ):
        # The clean member is a moving stop-gradient target.  PPO and the
        # counterfactual labels retain the useful physical response; this term
        # only suppresses the local action change caused by the measured
        # AR(1)/post-hit lateral-velocity disturbance of the same state.
        clean_mean = jax.lax.stop_gradient(
            policy_mean(params, noise_invariance_clean_obs)
        )
        noisy_mean = policy_mean(params, noise_invariance_noisy_obs)
        noise_invariance_mse = jnp.mean(
            jnp.sum(jnp.square(noisy_mean - clean_mean), axis=-1)
        )
        noise_invariance_regularization = (
            float(noise_invariance_coef) * noise_invariance_mse
        )
    if (
        float(action_feedback_sensitivity_coef) > 0.0
        and float(action_feedback_perturb_scale) > 0.0
        and action_feedback_obs_starts
    ):
        # Preserve the actual rollout/deployment observation exactly.  The
        # paired branches add a small coherent per-joint offset to every copy
        # of previous-action/action-history in the temporal observation.  A
        # moving stop-gradient target suppresses only the local actor
        # sensitivity to that audited feedback channel; it does not imitate a
        # teacher, recorded action, or altered observation distribution.
        rows = jnp.arange(batch.obs.shape[0], dtype=jnp.int32)[:, None]
        joints = jnp.arange(7, dtype=jnp.int32)[None, :]
        signs = jnp.where(
            jnp.bitwise_and(jnp.right_shift(rows, joints), 1) == 0,
            -1.0,
            1.0,
        ).astype(batch.obs.dtype)
        perturb = float(action_feedback_perturb_scale) * signs
        plus_obs = batch.obs
        minus_obs = batch.obs
        for start in action_feedback_obs_starts:
            plus_obs = plus_obs.at[:, int(start) : int(start) + 7].add(perturb)
            minus_obs = minus_obs.at[:, int(start) : int(start) + 7].add(-perturb)
        clean_mean = jax.lax.stop_gradient(mean)
        plus_mean = policy_mean(params, plus_obs)
        minus_mean = policy_mean(params, minus_obs)
        action_feedback_sensitivity_mse = 0.5 * jnp.mean(
            jnp.sum(
                jnp.square(plus_mean - clean_mean)
                + jnp.square(minus_mean - clean_mean),
                axis=-1,
            )
        )
        action_feedback_sensitivity_regularization = (
            float(action_feedback_sensitivity_coef)
            * action_feedback_sensitivity_mse
        )
    if (
        teacher_params is not None
        and teacher_distill_obs is not None
        and float(teacher_distill_coef) > 0.0
    ):
        teacher_mean = jax.lax.stop_gradient(
            policy_mean(teacher_params, teacher_distill_obs)
        )
        action_clip = float(teacher_distill_action_clip)
        if action_clip > 0.0:
            teacher_target_clip_fraction = jnp.mean(
                (jnp.abs(teacher_mean) > action_clip).astype(jnp.float32)
            )
            teacher_mean = jnp.clip(teacher_mean, -action_clip, action_clip)
        student_teacher_domain_mean = policy_mean(params, teacher_distill_obs)
        teacher_distill_mse = jnp.mean(
            jnp.square(student_teacher_domain_mean - teacher_mean)
        )
        teacher_distill_regularization = (
            float(teacher_distill_coef) * teacher_distill_mse
        )
    if reference_params is not None and (
        float(actor_anchor_kl_coef) > 0.0
        or float(actor_anchor_replay_kl_coef) > 0.0
    ):
        # Compare the two distributions under the same exploration bounds used
        # by rollout/PPO.  Without this, a configured log-std bound creates a
        # large non-zero "anchor" KL even when params and reference_params are
        # exactly identical.
        reference_log_std = jax.lax.stop_gradient(
            effective_log_std(reference_params["log_std"], min_log_std, max_log_std)
        )
        reference_var = jnp.exp(2.0 * reference_log_std)
        current_var = jnp.exp(2.0 * log_std)

        def anchor_kl(obs: jax.Array) -> jax.Array:
            current_mean = policy_mean(params, obs)
            reference_mean = jax.lax.stop_gradient(policy_mean(reference_params, obs))
            return jnp.mean(jnp.sum(
                log_std
                - reference_log_std
                + (reference_var + jnp.square(reference_mean - current_mean))
                / (2.0 * current_var)
                - 0.5,
                axis=-1,
            ))

        actor_anchor_current_kl = anchor_kl(batch.obs)
        actor_anchor_kl = actor_anchor_current_kl
        actor_anchor_regularization = (
            float(actor_anchor_kl_coef) * actor_anchor_current_kl
        )
        if actor_anchor_obs is not None:
            actor_anchor_replay_kl = anchor_kl(actor_anchor_obs)
            if float(actor_anchor_replay_kl_coef) > 0.0:
                actor_anchor_regularization = (
                    actor_anchor_regularization
                    + float(actor_anchor_replay_kl_coef) * actor_anchor_replay_kl
                )
            else:
                # Preserve the original equal-domain averaging behavior when
                # no independent replay coefficient is requested.
                actor_anchor_kl = 0.5 * (
                    actor_anchor_current_kl + actor_anchor_replay_kl
                )
                actor_anchor_regularization = (
                    float(actor_anchor_kl_coef) * actor_anchor_kl
                )
    loss = (
        policy_loss
        + float(vf_coef) * value_loss
        + float(cmdp_cost_vf_coef) * cmdp_cost_value_loss
        - float(ent_coef) * entropy
        + actor_anchor_regularization
        + residual_regularization
        + teacher_distill_regularization
        + counterfactual_supervision_regularization
        + noise_invariance_regularization
        + action_feedback_sensitivity_regularization
    )
    approx_kl = jnp.mean(batch.old_logp - logp)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > float(clip_range)).astype(jnp.float32))
    aux = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "cmdp/cost_value_loss": cmdp_cost_value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "actor_anchor_kl": actor_anchor_kl,
        "actor_anchor_current_kl": actor_anchor_current_kl,
        "actor_anchor_replay_kl": actor_anchor_replay_kl,
        "actor_anchor_regularization": actor_anchor_regularization,
        "residual_rms": residual_rms,
        "residual_abs_max": residual_abs_max,
        "residual_regularization": residual_regularization,
        "teacher_distill_mse": teacher_distill_mse,
        "teacher_distill_regularization": teacher_distill_regularization,
        "teacher_target_clip_fraction": teacher_target_clip_fraction,
        "counterfactual_supervision_mse": counterfactual_supervision_mse,
        "counterfactual_supervision_regularization": counterfactual_supervision_regularization,
        "noise_invariance_mse": noise_invariance_mse,
        "noise_invariance_regularization": noise_invariance_regularization,
        "action_feedback_sensitivity_mse": action_feedback_sensitivity_mse,
        "action_feedback_sensitivity_regularization": action_feedback_sensitivity_regularization,
    }
    return loss, aux


def compute_gae(
    rewards: jax.Array,
    dones: jax.Array,
    values: jax.Array,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
    *,
    terminated: jax.Array | None = None,
    truncated: jax.Array | None = None,
    timeout_values: jax.Array | None = None,
    time_limit_bootstrap: bool = False,
) -> tuple[jax.Array, jax.Array]:
    if time_limit_bootstrap:
        if terminated is None or truncated is None or timeout_values is None:
            raise ValueError(
                "time-limit bootstrap requires terminated, truncated, and timeout_values"
            )
        terminated = terminated.astype(bool)
        truncated = truncated.astype(bool)
    else:
        terminated = dones.astype(bool)
        truncated = jnp.zeros_like(dones, dtype=bool)
        timeout_values = jnp.zeros_like(values)

    def scan_fn(carry, xs):
        next_adv, next_value = carry
        reward, done, is_terminated, is_truncated, timeout_value, value = xs
        # A time-limit boundary ends the GAE trace so advantages never leak
        # into the reset episode, but it is not an MDP terminal: bootstrap the
        # value of the physical pre-reset state.  True task failures retain a
        # zero bootstrap target.
        bootstrap_value = jnp.where(is_truncated, timeout_value, next_value)
        bootstrap_mask = 1.0 - is_terminated.astype(jnp.float32)
        trace_mask = 1.0 - done.astype(jnp.float32)
        delta = reward + float(gamma) * bootstrap_mask * bootstrap_value - value
        adv = delta + float(gamma) * float(gae_lambda) * trace_mask * next_adv
        return (adv, value), adv

    init = (jnp.zeros_like(last_value), last_value)
    _, adv_rev = jax.lax.scan(
        scan_fn,
        init,
        (
            rewards[::-1],
            dones[::-1],
            terminated[::-1],
            truncated[::-1],
            timeout_values[::-1],
            values[::-1],
        ),
    )
    advantages = adv_rev[::-1]
    returns = advantages + values
    return advantages, returns


def cmdp_transition_costs(
    transitions: Transition,
    *,
    max_episode_steps: int,
) -> jax.Array:
    """Build terminal CMDP costs with shape ``[time, env, channel]``.

    ``failure`` constrains the probability of any true task termination.
    ``shortfall`` constrains expected episode length exactly because its
    undiscounted episodic sum is ``(H - T) / H`` on a failure and zero on a
    full time-limit completion.  Four direction-specific channels prevent an
    optimizer from satisfying the aggregate bound by exchanging high-ball
    failures for low-ball or racket-height failures.
    """

    if int(max_episode_steps) <= 0:
        raise ValueError("max_episode_steps must be positive")
    terminated = transitions.terminated.astype(jnp.float32)
    remaining = jnp.clip(
        (float(max_episode_steps) - transitions.episode_length.astype(jnp.float32))
        / float(max_episode_steps),
        0.0,
        1.0,
    )

    def terminal_reason(name: str) -> jax.Array:
        key = f"done/{name}"
        if key not in transitions.metrics:
            raise ValueError(f"CMDP termination metric is missing: {key}")
        return terminated * transitions.metrics[key].astype(jnp.float32)

    return jnp.stack(
        (
            terminated,
            terminated * remaining,
            terminal_reason("ball_too_low"),
            terminal_reason("ball_too_high"),
            terminal_reason("racket_too_low"),
            terminal_reason("racket_too_high"),
        ),
        axis=-1,
    )


def flatten_time_env(x: jax.Array) -> jax.Array:
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def completed_failure_focus_mask(
    dones: jax.Array,
    terminated: jax.Array,
    hit_counts: jax.Array,
    *,
    hit_threshold: int,
    tail_steps: int = 0,
) -> jax.Array:
    """Select transitions preceding a completed low-hit true termination.

    A rollout can contain true task terminations, time-limit truncations, and
    an unfinished suffix.  Only true terminations are failures.  Walking the
    rollout backwards labels each transition with the outcome at the next
    episode boundary; every ``done`` boundary replaces (rather than inherits)
    that outcome so a truncation cannot borrow a later episode's failure.

    ``tail_steps <= 0`` keeps the legacy whole-episode scope.  A positive
    value restricts focus to the final N transitions, where actions can still
    be causally related to the miss and are not dominated by already-stable
    early contacts.
    """

    def propagate_outcome(carry, xs):
        final_hits, final_terminated, steps_to_end, final_valid = carry
        done_t, terminated_t, hits_t = xs
        steps_to_end = jnp.where(final_valid, steps_to_end + 1, steps_to_end)
        final_hits = jnp.where(done_t, hits_t, final_hits)
        final_terminated = jnp.where(done_t, terminated_t, final_terminated)
        steps_to_end = jnp.where(done_t, 0, steps_to_end)
        final_valid = jnp.where(done_t, True, final_valid)
        return (
            final_hits,
            final_terminated,
            steps_to_end,
            final_valid,
        ), (
            final_hits,
            final_terminated,
            steps_to_end,
            final_valid,
        )

    init = (
        jnp.zeros_like(hit_counts[-1]),
        jnp.zeros_like(terminated[-1], dtype=bool),
        jnp.zeros_like(hit_counts[-1], dtype=jnp.int32),
        jnp.zeros_like(dones[-1], dtype=bool),
    )
    _, (episode_hits, episode_terminated, steps_to_end, episode_valid) = jax.lax.scan(
        propagate_outcome,
        init,
        (dones, terminated, hit_counts),
        reverse=True,
    )
    in_tail = (int(tail_steps) <= 0) | (steps_to_end < int(tail_steps))
    return (
        episode_valid
        & episode_terminated
        & (episode_hits < int(hit_threshold))
        & in_tail
    )


def completed_success_focus_mask(
    dones: jax.Array,
    truncated: jax.Array,
    hit_counts: jax.Array,
    *,
    hit_threshold: int,
    tail_steps: int = 0,
) -> jax.Array:
    """Select transitions preceding a completed full-horizon success.

    A time-limit truncation is the juggling task's full-horizon success rather
    than a failure.  As with :func:`completed_failure_focus_mask`, the outcome
    is propagated only within the current rollout and cannot leak across a
    different episode boundary or into an unfinished suffix.
    """

    def propagate_outcome(carry, xs):
        final_hits, final_truncated, steps_to_end, final_valid = carry
        done_t, truncated_t, hits_t = xs
        steps_to_end = jnp.where(final_valid, steps_to_end + 1, steps_to_end)
        final_hits = jnp.where(done_t, hits_t, final_hits)
        final_truncated = jnp.where(done_t, truncated_t, final_truncated)
        steps_to_end = jnp.where(done_t, 0, steps_to_end)
        final_valid = jnp.where(done_t, True, final_valid)
        return (
            final_hits,
            final_truncated,
            steps_to_end,
            final_valid,
        ), (
            final_hits,
            final_truncated,
            steps_to_end,
            final_valid,
        )

    init = (
        jnp.zeros_like(hit_counts[-1]),
        jnp.zeros_like(truncated[-1], dtype=bool),
        jnp.zeros_like(hit_counts[-1], dtype=jnp.int32),
        jnp.zeros_like(dones[-1], dtype=bool),
    )
    _, (episode_hits, episode_truncated, steps_to_end, episode_valid) = jax.lax.scan(
        propagate_outcome,
        init,
        (dones, truncated, hit_counts),
        reverse=True,
    )
    in_tail = (int(tail_steps) <= 0) | (steps_to_end < int(tail_steps))
    return (
        episode_valid
        & episode_truncated
        & (episode_hits >= int(hit_threshold))
        & in_tail
    )


def apply_outcome_balanced_advantage_focus(
    advantages: jax.Array,
    failure_focus: jax.Array,
    success_focus: jax.Array,
    *,
    failure_mass: float = 0.0,
    success_mass: float = 0.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Give rare completed outcomes a population-independent actor share.

    Ordinary sample multipliers make an outcome group's influence vanish as
    its rollout fraction becomes small.  Here each configured ``mass`` adds
    that much *batch-average* actor weight to the corresponding completed
    outcome group, independent of whether the group occupies 0.5% or 5% of
    the rollout.  Only negative advantages from true failures and positive
    advantages from time-limit successes are eligible.  The final RMS
    normalization preserves the PPO optimizer's established overall scale.
    """

    if float(failure_mass) < 0.0 or float(success_mass) < 0.0:
        raise ValueError("outcome focus masses must be >= 0")
    focused_failure = failure_focus & (advantages < 0.0)
    focused_success = success_focus & (advantages > 0.0)
    failure_fraction = jnp.mean(focused_failure.astype(jnp.float32))
    success_fraction = jnp.mean(focused_success.astype(jnp.float32))
    failure_applied = jnp.where(
        failure_fraction > 0.0,
        jnp.asarray(float(failure_mass), dtype=advantages.dtype),
        jnp.asarray(0.0, dtype=advantages.dtype),
    )
    success_applied = jnp.where(
        success_fraction > 0.0,
        jnp.asarray(float(success_mass), dtype=advantages.dtype),
        jnp.asarray(0.0, dtype=advantages.dtype),
    )
    sample_weight = jnp.ones_like(advantages)
    sample_weight = sample_weight + jnp.where(
        focused_failure,
        failure_applied / jnp.maximum(failure_fraction, 1.0e-8),
        0.0,
    )
    sample_weight = sample_weight + jnp.where(
        focused_success,
        success_applied / jnp.maximum(success_fraction, 1.0e-8),
        0.0,
    )
    focused_advantages = advantages * sample_weight
    focused_advantages = focused_advantages / jnp.sqrt(
        jnp.mean(focused_advantages * focused_advantages) + 1.0e-8
    )
    return focused_advantages, {
        "failure_focus_fraction": failure_fraction,
        "success_focus_fraction": success_fraction,
        "failure_focus_balanced_mass_applied": failure_applied,
        "success_focus_balanced_mass_applied": success_applied,
        "outcome_focus_sample_weight_mean": jnp.mean(sample_weight),
    }


def hard_lane_focus_mask(
    metrics: dict[str, jax.Array],
    conditions: tuple[tuple[str, str, float], ...],
    *,
    min_conditions: int,
) -> tuple[jax.Array, jax.Array]:
    """Identify rollout samples lying in multiple preregistered hard tails.

    Each condition is ``(metric_name, direction, threshold)`` where direction
    is ``"high"`` or ``"low"``.  Requiring more than one independently drawn
    tail avoids turning a single noisy attribution into a privileged training
    label.  The environment and actor observations remain unchanged; these
    metrics are used only to allocate actor-gradient mass.
    """

    if not conditions:
        raise ValueError("hard-lane focus requires at least one condition")
    if int(min_conditions) <= 0 or int(min_conditions) > len(conditions):
        raise ValueError("hard-lane min_conditions must be in [1, condition count]")
    reference = metrics[conditions[0][0]]
    condition_count = jnp.zeros_like(reference, dtype=jnp.int32)
    for metric_name, direction, threshold in conditions:
        if metric_name not in metrics:
            raise ValueError(f"hard-lane focus metric is missing: {metric_name}")
        if direction == "high":
            active = metrics[metric_name] >= float(threshold)
        elif direction == "low":
            active = metrics[metric_name] <= float(threshold)
        else:
            raise ValueError(
                f"hard-lane focus direction must be high/low, got {direction!r}"
            )
        condition_count = condition_count + active.astype(jnp.int32)
    return condition_count >= int(min_conditions), condition_count


def minimum_group_actor_weights(
    group_mask: jax.Array,
    *,
    minimum_mass: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Allocate a minimum normalized actor-sample mass to a rare group.

    When the observed group fraction is below ``minimum_mass``, group and
    complement weights are chosen so their mean remains exactly one and the
    group's normalized weight becomes ``minimum_mass``. Empty, complete, or
    already-sufficient batches are exact no-ops. These weights are intended
    for actor advantages only; critic targets must remain population-uniform.
    """

    if not 0.0 <= float(minimum_mass) <= 1.0:
        raise ValueError("minimum_mass must be in [0, 1]")
    mask = jnp.asarray(group_mask, dtype=bool)
    dtype = jnp.float32
    fraction = jnp.mean(mask.astype(dtype))
    target = jnp.asarray(float(minimum_mass), dtype=dtype)
    applied = (
        (target > 0.0)
        & (fraction > 0.0)
        & (fraction < 1.0)
        & (fraction < target)
    )
    group_weight = jnp.where(
        applied,
        target / jnp.maximum(fraction, 1.0e-8),
        jnp.asarray(1.0, dtype=dtype),
    )
    main_weight = jnp.where(
        applied,
        (1.0 - target) / jnp.maximum(1.0 - fraction, 1.0e-8),
        jnp.asarray(1.0, dtype=dtype),
    )
    weights = jnp.where(mask, group_weight, main_weight)
    weighted_fraction = jnp.sum(weights * mask.astype(dtype)) / jnp.maximum(
        jnp.sum(weights), 1.0e-8
    )
    return weights, {
        "raw_fraction": fraction,
        "minimum_mass": target,
        "weighted_fraction": weighted_fraction,
        "group_weight": group_weight,
        "main_weight": main_weight,
        "applied": applied.astype(dtype),
    }


def mix_hard_episode_reset_keys(
    random_reset_keys: jax.Array,
    *,
    selector_key: jax.Array,
    hard_reset_keys: jax.Array | None,
    episode_mass: float,
    eligible: jax.Array | None = None,
) -> jax.Array:
    """Replace eligible episode resets from a fixed-probability key mixture.

    The pool stores complete JAX reset keys, so replaying one recreates the
    associated reset and episode-constant DR tuple without changing any DR
    interval. Selection is made only among lanes that actually reset, so
    episode length cannot change the per-reset mixture probability. A zero
    mass is an exact legacy no-op.
    """

    if not 0.0 <= float(episode_mass) <= 1.0:
        raise ValueError("episode_mass must be in [0, 1]")
    if float(episode_mass) == 0.0:
        return random_reset_keys
    if hard_reset_keys is None:
        raise ValueError("positive episode_mass requires hard_reset_keys")
    hard_reset_keys = jnp.asarray(hard_reset_keys, dtype=jnp.uint32)
    if hard_reset_keys.ndim != 2 or hard_reset_keys.shape[1] != 2:
        raise ValueError("hard_reset_keys must have shape [pool, 2]")
    if hard_reset_keys.shape[0] <= 0:
        raise ValueError("hard_reset_keys cannot be empty")

    lane_count = int(random_reset_keys.shape[0])
    if eligible is None:
        eligible = jnp.ones((lane_count,), dtype=bool)
    else:
        eligible = jnp.asarray(eligible, dtype=bool)
    if eligible.shape != (lane_count,):
        raise ValueError("eligible must have shape [n_envs]")

    selection_key, pool_key = jax.random.split(selector_key)
    selected = eligible & jax.random.bernoulli(
        selection_key,
        p=float(episode_mass),
        shape=(lane_count,),
    )

    pool_indices = jax.random.randint(
        pool_key,
        (lane_count,),
        minval=0,
        maxval=hard_reset_keys.shape[0],
    )
    replay_keys = hard_reset_keys[pool_indices]
    return jnp.where(selected[:, None], replay_keys, random_reset_keys)


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
    reference_params=None,
    actor_anchor_kl_coef: float = 0.0,
    actor_anchor_replay_obs: jax.Array | None = None,
    actor_anchor_replay_kl_coef: float = 0.0,
    actor_anchor_replay_max_kl: float = 0.0,
    residual_l2_coef: float = 0.0,
    teacher_params=None,
    teacher_distill_replay_obs: jax.Array | None = None,
    teacher_distill_coef: float = 0.0,
    teacher_distill_action_clip: float = 1.0,
    time_limit_bootstrap: bool = True,
    min_log_std: float | None = None,
    max_log_std: float | None = None,
    target_kl: float | None = None,
    failure_focus_hit_threshold: int = 0,
    failure_focus_weight: float = 1.0,
    failure_focus_tail_steps: int = 0,
    failure_focus_balanced_mass: float = 0.0,
    success_focus_hit_threshold: int = 0,
    success_focus_weight: float = 1.0,
    success_focus_tail_steps: int = 0,
    success_focus_balanced_mass: float = 0.0,
    hard_lane_focus_conditions: tuple[tuple[str, str, float], ...] = (),
    hard_lane_focus_min_conditions: int = 1,
    hard_lane_focus_weight: float = 1.0,
    reset_family_min_actor_mass: float = 0.0,
    counterfactual_replay_obs: jax.Array | None = None,
    counterfactual_replay_actions: jax.Array | None = None,
    counterfactual_supervision_coef: float = 0.0,
    counterfactual_focus_tail_rows: int = 0,
    counterfactual_focus_prob: float = 0.0,
    counterfactual_vxy_weight_mode: str = "high_vxy",
    noise_invariance_clean_obs: jax.Array | None = None,
    noise_invariance_noisy_obs: jax.Array | None = None,
    noise_invariance_coef: float = 0.0,
    action_feedback_sensitivity_coef: float = 0.0,
    action_feedback_perturb_scale: float = 0.0,
    action_feedback_obs_starts: tuple[int, ...] = (),
    cmdp_enabled: bool = False,
    cmdp_cost_limits: tuple[float, ...] = (),
    cmdp_cost_gae_lambda: float = 0.99,
    cmdp_cost_vf_coef: float = 0.5,
    cmdp_dual_learning_rate: float = 0.1,
    cmdp_dual_max: float = 2.0,
    hard_reset_keys: jax.Array | None = None,
    hard_reset_episode_mass: float = 0.0,
):
    if (
        min_log_std is not None
        and max_log_std is not None
        and float(min_log_std) > float(max_log_std)
    ):
        raise ValueError("min_log_std must be <= max_log_std")
    if int(failure_focus_hit_threshold) < 0 or int(failure_focus_tail_steps) < 0:
        raise ValueError("failure focus thresholds/tail must be >= 0")
    if float(failure_focus_weight) < 1.0:
        raise ValueError("failure_focus_weight must be >= 1")
    if not 0.0 <= float(failure_focus_balanced_mass) <= 1.0:
        raise ValueError("failure_focus_balanced_mass must be in [0, 1]")
    if int(success_focus_hit_threshold) < 0 or int(success_focus_tail_steps) < 0:
        raise ValueError("success focus thresholds/tail must be >= 0")
    if float(success_focus_weight) < 1.0:
        raise ValueError("success_focus_weight must be >= 1")
    if not 0.0 <= float(success_focus_balanced_mass) <= 1.0:
        raise ValueError("success_focus_balanced_mass must be in [0, 1]")
    if float(hard_lane_focus_weight) < 1.0:
        raise ValueError("hard_lane_focus_weight must be >= 1")
    if not 0.0 <= float(reset_family_min_actor_mass) <= 1.0:
        raise ValueError("reset_family_min_actor_mass must be in [0, 1]")
    if hard_lane_focus_conditions and (
        int(hard_lane_focus_min_conditions) <= 0
        or int(hard_lane_focus_min_conditions) > len(hard_lane_focus_conditions)
    ):
        raise ValueError(
            "hard_lane_focus_min_conditions must be in [1, condition count]"
        )
    if float(hard_lane_focus_weight) > 1.0 and not hard_lane_focus_conditions:
        raise ValueError(
            "hard_lane_focus_weight > 1 requires hard-lane conditions"
        )
    for metric_name, direction, _threshold in hard_lane_focus_conditions:
        if not metric_name:
            raise ValueError("hard-lane focus metric name cannot be empty")
        if direction not in {"high", "low"}:
            raise ValueError(
                f"hard-lane focus direction must be high/low, got {direction!r}"
            )
    if (
        float(actor_anchor_kl_coef) > 0.0
        or float(actor_anchor_replay_kl_coef) > 0.0
    ) and reference_params is None:
        raise ValueError("actor anchor KL requires reference_params")
    if float(actor_anchor_replay_kl_coef) > 0.0 and actor_anchor_replay_obs is None:
        raise ValueError("actor_anchor_replay_kl_coef > 0 requires replay observations")
    if float(actor_anchor_replay_max_kl) < 0.0:
        raise ValueError("actor_anchor_replay_max_kl must be >= 0")
    if float(actor_anchor_replay_max_kl) > 0.0:
        if reference_params is None:
            raise ValueError("actor anchor replay KL guard requires reference_params")
        if actor_anchor_replay_obs is None:
            raise ValueError("actor anchor replay KL guard requires replay observations")
    if float(teacher_distill_coef) > 0.0:
        if teacher_params is None:
            raise ValueError("teacher_distill_coef > 0 requires teacher_params")
        if teacher_distill_replay_obs is None:
            raise ValueError(
                "teacher_distill_coef > 0 requires teacher replay observations"
            )
    if float(counterfactual_supervision_coef) > 0.0:
        if counterfactual_replay_obs is None or counterfactual_replay_actions is None:
            raise ValueError(
                "counterfactual_supervision_coef > 0 requires replay observations and actions"
            )
        if counterfactual_replay_obs.shape[0] != counterfactual_replay_actions.shape[0]:
            raise ValueError("counterfactual replay observations/actions must have equal rows")
        if not 0 <= int(counterfactual_focus_tail_rows) < counterfactual_replay_obs.shape[0]:
            raise ValueError("counterfactual_focus_tail_rows must be in [0, replay_rows)")
        if not 0.0 <= float(counterfactual_focus_prob) <= 1.0:
            raise ValueError("counterfactual_focus_prob must be in [0, 1]")
        if float(counterfactual_focus_prob) > 0.0 and int(counterfactual_focus_tail_rows) == 0:
            raise ValueError("counterfactual_focus_prob > 0 requires focus tail rows")
    if float(noise_invariance_coef) > 0.0:
        if noise_invariance_clean_obs is None or noise_invariance_noisy_obs is None:
            raise ValueError(
                "noise_invariance_coef > 0 requires clean and noisy replay observations"
            )
        if noise_invariance_clean_obs.shape != noise_invariance_noisy_obs.shape:
            raise ValueError("noise invariance clean/noisy observations must have equal shape")
        if noise_invariance_clean_obs.shape[0] <= 0:
            raise ValueError("noise invariance replay cannot be empty")
    if float(action_feedback_sensitivity_coef) < 0.0:
        raise ValueError("action_feedback_sensitivity_coef must be >= 0")
    if float(action_feedback_perturb_scale) < 0.0:
        raise ValueError("action_feedback_perturb_scale must be >= 0")
    if float(action_feedback_sensitivity_coef) > 0.0:
        if float(action_feedback_perturb_scale) <= 0.0:
            raise ValueError(
                "action feedback sensitivity requires a positive perturb scale"
            )
        if not action_feedback_obs_starts:
            raise ValueError(
                "action feedback sensitivity requires observation slice starts"
            )
        if any(
            int(start) < 0 or int(start) + 7 > int(env.obs_dim)
            for start in action_feedback_obs_starts
        ):
            raise ValueError(
                "action feedback sensitivity observation slice is outside obs_dim"
            )
    if bool(cmdp_enabled):
        if len(cmdp_cost_limits) != len(CMDP_COST_NAMES):
            raise ValueError(
                "CMDP cost limits must match the registered cost channels"
            )
        if not 0.0 <= float(cmdp_cost_gae_lambda) <= 1.0:
            raise ValueError("cmdp_cost_gae_lambda must be in [0, 1]")
        if float(cmdp_cost_vf_coef) <= 0.0:
            raise ValueError("cmdp_cost_vf_coef must be positive")
        if float(cmdp_dual_learning_rate) <= 0.0:
            raise ValueError("cmdp_dual_learning_rate must be positive")
        if float(cmdp_dual_max) <= 0.0:
            raise ValueError("cmdp_dual_max must be positive")
        if any(float(limit) < 0.0 for limit in cmdp_cost_limits):
            raise ValueError("CMDP cost limits must be nonnegative")
    if not 0.0 <= float(hard_reset_episode_mass) <= 1.0:
        raise ValueError("hard_reset_episode_mass must be in [0, 1]")
    if float(hard_reset_episode_mass) > 0.0:
        if hard_reset_keys is None:
            raise ValueError(
                "hard_reset_episode_mass > 0 requires hard_reset_keys"
            )
        if hard_reset_keys.ndim != 2 or hard_reset_keys.shape[1] != 2:
            raise ValueError("hard_reset_keys must have shape [pool, 2]")
        if hard_reset_keys.shape[0] <= 0:
            raise ValueError("hard_reset_keys cannot be empty")
    batch_size = env.n_envs * int(n_steps)
    num_minibatches = max(1, batch_size // int(minibatch_size))
    used_batch_size = num_minibatches * int(minibatch_size)

    def collect_rollout(params, runner: RunnerState) -> tuple[RunnerState, Transition]:
        def rollout_step(carry: RunnerState, _):
            env_state, obs, critic_obs, rng, running_return, running_length = carry
            rng, action_key, reset_key = jax.random.split(rng, 3)
            mean, value = policy_value(params, obs, critic_obs)
            log_std = effective_log_std(
                params["log_std"], min_log_std, max_log_std
            )
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
            env_action = jnp.clip(raw_action, -1.0, 1.0)
            logp = normal_logprob(raw_action, mean, log_std)
            next_env_state, next_obs, reward, done, metrics = env.step(env_state, env_action)

            # Capture V(s_{t+1}) before reset.  This is used only for time-limit
            # truncations; evaluating it here avoids storing a full privileged
            # next observation for every rollout step.
            terminal_critic_obs = env.get_critic_obs(next_env_state, next_obs)
            timeout_value = value_fn(params, terminal_critic_obs)

            completed_return = running_return + reward
            completed_length = running_length + 1
            reset_keys = jax.random.split(reset_key, env.n_envs)
            reset_keys = mix_hard_episode_reset_keys(
                reset_keys,
                selector_key=reset_key,
                hard_reset_keys=hard_reset_keys,
                episode_mass=hard_reset_episode_mass,
                eligible=done,
            )
            next_env_state, next_obs = env.reset_done(next_env_state, next_obs, done, reset_keys)
            next_critic_obs = env.get_critic_obs(next_env_state, next_obs)
            next_running_return = jnp.where(done, 0.0, completed_return)
            next_running_length = jnp.where(done, 0, completed_length)

            transition = Transition(
                obs=obs,
                critic_obs=critic_obs,
                action=raw_action,
                logp=logp,
                value=value,
                reward=reward,
                done=done,
                terminated=metrics["terminated"].astype(bool),
                truncated=metrics["truncated"].astype(bool),
                timeout_value=timeout_value,
                episode_return=completed_return,
                episode_length=completed_length,
                new_hit=metrics["new_hit"],
                hit_count=metrics["hit_count"],
                metrics=metrics,
            )
            return (
                RunnerState(next_env_state, next_obs, next_critic_obs, rng, next_running_return, next_running_length),
                transition,
            )

        return jax.lax.scan(rollout_step, runner, None, length=int(n_steps))

    def update(train_state: TrainState, runner: RunnerState, transitions: Transition) -> tuple[TrainState, dict[str, jax.Array]]:
        # This is the immutable behavior policy that generated the rollout.
        # Every KL safety decision below is measured against this snapshot.
        behavior_train_state = train_state
        last_value = value_fn(train_state.params, runner.critic_obs)
        advantages, returns = compute_gae(
            transitions.reward,
            transitions.done,
            transitions.value,
            last_value,
            gamma,
            gae_lambda,
            terminated=transitions.terminated,
            truncated=transitions.truncated,
            timeout_values=transitions.timeout_value,
            time_limit_bootstrap=time_limit_bootstrap,
        )
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
        cmdp_cost_returns = None
        cmdp_cost_old_values = None
        cmdp_observed_costs = jnp.zeros(
            (len(CMDP_COST_NAMES),), dtype=jnp.float32
        )
        cmdp_dual_before = jnp.zeros(
            (len(CMDP_COST_NAMES),), dtype=jnp.float32
        )
        cmdp_completed_count = jnp.asarray(0.0, dtype=jnp.float32)
        if bool(cmdp_enabled):
            cost_values = cmdp_cost_value_fn(
                train_state.params, transitions.critic_obs
            )
            last_cost_value = cmdp_cost_value_fn(
                train_state.params, runner.critic_obs
            )
            costs = cmdp_transition_costs(
                transitions,
                max_episode_steps=int(env.max_steps),
            )
            channel_advantages = []
            channel_returns = []
            for channel_index in range(len(CMDP_COST_NAMES)):
                cost_advantage, cost_return = compute_gae(
                    costs[..., channel_index],
                    transitions.done,
                    cost_values[..., channel_index],
                    last_cost_value[..., channel_index],
                    1.0,
                    float(cmdp_cost_gae_lambda),
                )
                channel_advantages.append(cost_advantage)
                channel_returns.append(cost_return)
            cost_advantages = jnp.stack(channel_advantages, axis=-1)
            cmdp_cost_returns = jnp.stack(channel_returns, axis=-1)
            cmdp_cost_old_values = cost_values
            cost_advantages = (
                cost_advantages
                - jnp.mean(cost_advantages, axis=(0, 1), keepdims=True)
            ) / (
                jnp.std(cost_advantages, axis=(0, 1), keepdims=True) + 1.0e-8
            )
            cmdp_dual_before = jax.lax.stop_gradient(
                jnp.asarray(train_state.params["cmdp_dual"])
            )
            advantages = advantages - jnp.sum(
                cmdp_dual_before[None, None, :] * cost_advantages,
                axis=-1,
            )
            advantages = (
                advantages - jnp.mean(advantages)
            ) / (jnp.std(advantages) + 1.0e-8)
            cmdp_completed_count = jnp.sum(
                transitions.done.astype(jnp.float32)
            )
            cmdp_observed_costs = jnp.where(
                cmdp_completed_count > 0.0,
                jnp.sum(costs, axis=(0, 1))
                / jnp.maximum(cmdp_completed_count, 1.0),
                jnp.asarray(cmdp_cost_limits, dtype=jnp.float32),
            )
        failure_focus_fraction = jnp.asarray(0.0, dtype=jnp.float32)
        success_focus_fraction = jnp.asarray(0.0, dtype=jnp.float32)
        failure_focus_balanced_mass_applied = jnp.asarray(0.0, dtype=jnp.float32)
        success_focus_balanced_mass_applied = jnp.asarray(0.0, dtype=jnp.float32)
        outcome_focus_sample_weight_mean = jnp.asarray(1.0, dtype=jnp.float32)
        hard_lane_focus_fraction = jnp.asarray(0.0, dtype=jnp.float32)
        hard_lane_focus_mean_condition_count = jnp.asarray(
            0.0, dtype=jnp.float32
        )
        hard_lane_focus_advantage_abs_fraction = jnp.asarray(
            0.0, dtype=jnp.float32
        )
        reset_family_raw_fraction = jnp.asarray(0.0, dtype=jnp.float32)
        reset_family_weighted_fraction = jnp.asarray(0.0, dtype=jnp.float32)
        reset_family_group_weight = jnp.asarray(1.0, dtype=jnp.float32)
        reset_family_main_weight = jnp.asarray(1.0, dtype=jnp.float32)
        reset_family_mass_applied = jnp.asarray(0.0, dtype=jnp.float32)
        advantage_weight = jnp.ones_like(advantages)
        focus_enabled = False
        failure_focus = jnp.zeros_like(transitions.done, dtype=bool)
        success_focus = jnp.zeros_like(transitions.done, dtype=bool)
        if int(failure_focus_hit_threshold) > 0 and (
            float(failure_focus_weight) > 1.0
            or float(failure_focus_balanced_mass) > 0.0
        ):
            failure_focus = completed_failure_focus_mask(
                transitions.done,
                transitions.terminated,
                transitions.hit_count,
                hit_threshold=failure_focus_hit_threshold,
                tail_steps=failure_focus_tail_steps,
            )
            # Only amplify actions the critic already judges worse than its
            # baseline.  Weighting positive advantages from a failed episode
            # would reinforce the locally good first contacts even though the
            # same trajectory still terminates before the third hit.
            focused_negative = failure_focus & (advantages < 0.0)
            if float(failure_focus_weight) > 1.0:
                advantage_weight = jnp.where(
                    focused_negative,
                    advantage_weight * float(failure_focus_weight),
                    advantage_weight,
                )
            failure_focus_fraction = jnp.mean(focused_negative.astype(jnp.float32))
            focus_enabled = True
        if int(success_focus_hit_threshold) > 0 and (
            float(success_focus_weight) > 1.0
            or float(success_focus_balanced_mass) > 0.0
        ):
            success_focus = completed_success_focus_mask(
                transitions.done,
                transitions.truncated,
                transitions.hit_count,
                hit_threshold=success_focus_hit_threshold,
                tail_steps=success_focus_tail_steps,
            )
            # Reinforce only actions the critic already regards as better than
            # baseline.  Negative advantages in a successful episode still
            # carry their ordinary corrective signal.
            focused_positive = success_focus & (advantages > 0.0)
            if float(success_focus_weight) > 1.0:
                advantage_weight = jnp.where(
                    focused_positive,
                    advantage_weight * float(success_focus_weight),
                    advantage_weight,
                )
            success_focus_fraction = jnp.mean(focused_positive.astype(jnp.float32))
            focus_enabled = True
        if hard_lane_focus_conditions and float(hard_lane_focus_weight) > 1.0:
            hard_lane_focus, hard_lane_condition_count = hard_lane_focus_mask(
                transitions.metrics,
                hard_lane_focus_conditions,
                min_conditions=hard_lane_focus_min_conditions,
            )
            hard_lane_focus_fraction = jnp.mean(
                hard_lane_focus.astype(jnp.float32)
            )
            hard_lane_focus_mean_condition_count = jnp.mean(
                hard_lane_condition_count.astype(jnp.float32)
            )
            abs_advantage = jnp.abs(advantages)
            hard_lane_focus_advantage_abs_fraction = jnp.sum(
                abs_advantage * hard_lane_focus.astype(abs_advantage.dtype)
            ) / jnp.maximum(jnp.sum(abs_advantage), 1.0e-8)
            advantage_weight = jnp.where(
                hard_lane_focus,
                advantage_weight * float(hard_lane_focus_weight),
                advantage_weight,
            )
            focus_enabled = True
        if focus_enabled:
            advantages = advantages * advantage_weight
            if (
                float(failure_focus_balanced_mass) > 0.0
                or float(success_focus_balanced_mass) > 0.0
            ):
                advantages, balanced_focus_metrics = (
                    apply_outcome_balanced_advantage_focus(
                        advantages,
                        failure_focus,
                        success_focus,
                        failure_mass=failure_focus_balanced_mass,
                        success_mass=success_focus_balanced_mass,
                    )
                )
                failure_focus_fraction = balanced_focus_metrics[
                    "failure_focus_fraction"
                ]
                success_focus_fraction = balanced_focus_metrics[
                    "success_focus_fraction"
                ]
                failure_focus_balanced_mass_applied = balanced_focus_metrics[
                    "failure_focus_balanced_mass_applied"
                ]
                success_focus_balanced_mass_applied = balanced_focus_metrics[
                    "success_focus_balanced_mass_applied"
                ]
                outcome_focus_sample_weight_mean = balanced_focus_metrics[
                    "outcome_focus_sample_weight_mean"
                ]
            else:
                advantages = advantages / jnp.sqrt(
                    jnp.mean(advantages * advantages) + 1e-8
                )
        if float(reset_family_min_actor_mass) > 0.0:
            reset_family_weights, reset_family_metrics = (
                minimum_group_actor_weights(
                    transitions.metrics["reset_falling_contact_active"] >= 0.5,
                    minimum_mass=reset_family_min_actor_mass,
                )
            )
            advantages = advantages * reset_family_weights
            advantages = advantages / jnp.sqrt(
                jnp.mean(advantages * advantages) + 1e-8
            )
            reset_family_raw_fraction = reset_family_metrics["raw_fraction"]
            reset_family_weighted_fraction = reset_family_metrics[
                "weighted_fraction"
            ]
            reset_family_group_weight = reset_family_metrics["group_weight"]
            reset_family_main_weight = reset_family_metrics["main_weight"]
            reset_family_mass_applied = reset_family_metrics["applied"]

        batch = PpoBatch(
            obs=flatten_time_env(transitions.obs),
            critic_obs=flatten_time_env(transitions.critic_obs),
            action=flatten_time_env(transitions.action),
            old_logp=flatten_time_env(transitions.logp),
            advantages=flatten_time_env(advantages),
            returns=flatten_time_env(returns),
            old_values=flatten_time_env(transitions.value),
        )
        cmdp_batch = (
            CmdpBatch(cost_returns=flatten_time_env(cmdp_cost_returns))
            if cmdp_cost_returns is not None
            else None
        )

        def take_minibatch(b: PpoBatch, idx: jax.Array) -> PpoBatch:
            return jax.tree_util.tree_map(lambda x: x[idx], b)

        def update_minibatch(state: TrainState, idx: jax.Array):
            mini = take_minibatch(batch, idx)
            mini_cmdp = (
                CmdpBatch(cost_returns=cmdp_batch.cost_returns[idx])
                if cmdp_batch is not None
                else None
            )
            mini_anchor_obs = None
            if actor_anchor_replay_obs is not None:
                replay_idx = jnp.mod(idx, actor_anchor_replay_obs.shape[0])
                mini_anchor_obs = actor_anchor_replay_obs[replay_idx]
            mini_teacher_obs = None
            if teacher_distill_replay_obs is not None:
                teacher_replay_idx = jnp.mod(
                    idx, teacher_distill_replay_obs.shape[0]
                )
                mini_teacher_obs = teacher_distill_replay_obs[teacher_replay_idx]
            mini_counterfactual_obs = None
            mini_counterfactual_actions = None
            if counterfactual_replay_obs is not None:
                replay_rows = counterfactual_replay_obs.shape[0]
                focus_rows = int(counterfactual_focus_tail_rows)
                focus_count = int(round(int(minibatch_size) * float(counterfactual_focus_prob)))
                if focus_rows > 0 and focus_count > 0:
                    nonfocus_rows = replay_rows - focus_rows
                    sample_rank = jnp.arange(idx.shape[0])
                    nonfocus_idx = jnp.mod(idx, nonfocus_rows)
                    focus_idx = nonfocus_rows + jnp.mod(idx, focus_rows)
                    counterfactual_idx = jnp.where(
                        sample_rank < focus_count, focus_idx, nonfocus_idx
                    )
                else:
                    counterfactual_idx = jnp.mod(idx, replay_rows)
                mini_counterfactual_obs = counterfactual_replay_obs[counterfactual_idx]
                mini_counterfactual_actions = counterfactual_replay_actions[counterfactual_idx]
            mini_noise_invariance_clean_obs = None
            mini_noise_invariance_noisy_obs = None
            if noise_invariance_clean_obs is not None:
                noise_pair_idx = jnp.mod(idx, noise_invariance_clean_obs.shape[0])
                mini_noise_invariance_clean_obs = noise_invariance_clean_obs[noise_pair_idx]
                mini_noise_invariance_noisy_obs = noise_invariance_noisy_obs[noise_pair_idx]
            (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                state.params,
                mini,
                clip_range,
                vf_coef,
                ent_coef,
                reference_params,
                actor_anchor_kl_coef,
                mini_anchor_obs,
                actor_anchor_replay_kl_coef,
                residual_l2_coef,
                teacher_params,
                mini_teacher_obs,
                teacher_distill_coef,
                teacher_distill_action_clip,
                min_log_std,
                max_log_std,
                mini_counterfactual_obs,
                mini_counterfactual_actions,
                counterfactual_supervision_coef,
                counterfactual_focus_tail_rows,
                counterfactual_focus_prob,
                counterfactual_vxy_weight_mode,
                mini_noise_invariance_clean_obs,
                mini_noise_invariance_noisy_obs,
                noise_invariance_coef,
                action_feedback_sensitivity_coef,
                action_feedback_perturb_scale,
                action_feedback_obs_starts,
                mini_cmdp,
                cmdp_cost_vf_coef,
            )
            params, opt, grad_norm = adam_step(
                state.params,
                grads,
                state.opt,
                learning_rate,
                max_grad_norm,
            )
            params = project_policy_log_std(params, min_log_std, max_log_std)
            aux = dict(aux)
            aux["grad_norm"] = grad_norm
            aux["loss"] = loss
            return TrainState(params=params, opt=opt), aux

        old_log_std = effective_log_std(
            behavior_train_state.params["log_std"], min_log_std, max_log_std
        )

        def candidate_exact_kl(candidate_params, mini: PpoBatch) -> jax.Array:
            old_mean = policy_mean(behavior_train_state.params, mini.obs)
            new_mean = policy_mean(candidate_params, mini.obs)
            new_log_std = effective_log_std(
                candidate_params["log_std"], min_log_std, max_log_std
            )
            return diagonal_gaussian_kl(
                old_mean, old_log_std, new_mean, new_log_std
            )

        def update_epoch(carry, epoch_key):
            state, update_active = carry
            perm = jax.random.permutation(epoch_key, batch_size)[:used_batch_size]
            mb_idx = perm.reshape((num_minibatches, int(minibatch_size)))

            def guarded_minibatch(minibatch_carry, idx):
                minibatch_state, minibatch_active = minibatch_carry
                full_candidate_state, aux = update_minibatch(minibatch_state, idx)
                if target_kl is None or float(target_kl) <= 0.0:
                    apply_candidate = minibatch_active
                    next_active = minibatch_active
                    candidate_state = full_candidate_state
                    accepted_scale = jnp.where(
                        apply_candidate,
                        jnp.asarray(1.0, dtype=jnp.float32),
                        jnp.asarray(0.0, dtype=jnp.float32),
                    )
                    post_update_kl = jnp.asarray(0.0, dtype=jnp.float32)
                else:
                    # Evaluate the policy *after* the Adam candidate step.  A
                    # pre-update KL check always accepts the first minibatch
                    # because its KL is zero, even when that one step moves the
                    # policy far beyond the trust-region budget.  Backtrack the
                    # parameter displacement while keeping the same candidate
                    # Adam moments, and select the largest safe step.
                    mini = take_minibatch(batch, idx)
                    scales = jnp.asarray(
                        PPO_KL_BACKTRACK_SCALES, dtype=jnp.float32
                    )

                    def try_scale(scale_carry, step_scale):
                        selected_state, selected_kl, selected_scale, found = scale_carry
                        scaled_params = jax.tree_util.tree_map(
                            lambda previous, candidate: previous
                            + step_scale * (candidate - previous),
                            minibatch_state.params,
                            full_candidate_state.params,
                        )
                        scaled_params = project_policy_log_std(
                            scaled_params, min_log_std, max_log_std
                        )
                        scaled_state = TrainState(
                            params=scaled_params,
                            opt=full_candidate_state.opt,
                        )
                        scaled_kl = candidate_exact_kl(scaled_params, mini)
                        choose = (~found) & (scaled_kl <= float(target_kl))
                        selected_state = jax.tree_util.tree_map(
                            lambda selected, candidate: jnp.where(
                                choose, candidate, selected
                            ),
                            selected_state,
                            scaled_state,
                        )
                        selected_kl = jnp.where(choose, scaled_kl, selected_kl)
                        selected_scale = jnp.where(
                            choose, step_scale, selected_scale
                        )
                        return (
                            selected_state,
                            selected_kl,
                            selected_scale,
                            found | choose,
                        ), scaled_kl

                    initial_scale_carry = (
                        minibatch_state,
                        candidate_exact_kl(minibatch_state.params, mini),
                        jnp.asarray(0.0, dtype=jnp.float32),
                        jnp.asarray(False),
                    )
                    (
                        candidate_state,
                        post_update_kl,
                        accepted_scale,
                        found_safe_step,
                    ), _trial_kls = jax.lax.scan(
                        try_scale, initial_scale_carry, scales
                    )
                    apply_candidate = minibatch_active & found_safe_step
                    next_active = minibatch_active & found_safe_step
                next_state = jax.tree_util.tree_map(
                    lambda candidate, previous: jnp.where(
                        apply_candidate, candidate, previous
                    ),
                    candidate_state,
                    minibatch_state,
                )
                aux = dict(aux)
                aux["ppo_minibatch_applied"] = apply_candidate.astype(jnp.float32)
                aux["ppo_safe_step_scale"] = jnp.where(
                    apply_candidate, accepted_scale, 0.0
                )
                aux["ppo_candidate_exact_kl"] = jnp.where(
                    apply_candidate, post_update_kl, 0.0
                )
                aux["ppo_candidate_rejected"] = (
                    minibatch_active & (~apply_candidate)
                ).astype(jnp.float32)
                return (next_state, next_active), aux

            (next_state, next_active), minibatch_aux = jax.lax.scan(
                guarded_minibatch, (state, update_active), mb_idx
            )
            minibatch_applied = minibatch_aux.pop("ppo_minibatch_applied")
            candidate_rejected = minibatch_aux.pop("ppo_candidate_rejected")
            applied_count = jnp.sum(minibatch_applied)
            safe_count = jnp.maximum(applied_count, 1.0)
            epoch_aux = jax.tree_util.tree_map(
                lambda x: jnp.sum(x * minibatch_applied) / safe_count,
                minibatch_aux,
            )
            epoch_aux = dict(epoch_aux)
            epoch_aux["ppo_minibatches_applied"] = applied_count
            epoch_aux["ppo_candidates_rejected"] = jnp.sum(candidate_rejected)
            return (next_state, next_active), epoch_aux

        rng, epoch_rng = jax.random.split(runner.rng)
        epoch_keys = jax.random.split(epoch_rng, int(update_epochs))
        (train_state, _epoch_active), epoch_aux = jax.lax.scan(
            update_epoch,
            (train_state, jnp.asarray(True)),
            epoch_keys,
        )
        minibatches_applied = epoch_aux.pop("ppo_minibatches_applied")
        candidates_rejected = epoch_aux.pop("ppo_candidates_rejected")
        applied_count = jnp.maximum(jnp.sum(minibatches_applied), 1.0)
        aux_mean = jax.tree_util.tree_map(
            lambda x: jnp.sum(x * minibatches_applied) / applied_count,
            epoch_aux,
        )
        aux_mean["ppo_minibatches_applied"] = jnp.sum(minibatches_applied)
        aux_mean["ppo_candidates_rejected"] = jnp.sum(candidates_rejected)
        aux_mean["ppo_epochs_applied"] = (
            jnp.sum(minibatches_applied) / float(num_minibatches)
        )
        aux_mean["ppo_kl_guard_triggered"] = (
            jnp.sum(minibatches_applied)
            < float(int(update_epochs) * num_minibatches)
        ).astype(jnp.float32)
        # Audit the final policy displacement on the complete rollout.  The
        # minibatch check above is necessarily sampled; if it underestimates
        # the rollout-wide displacement, roll back both parameters and Adam
        # state transactionally so no over-budget update can reach a
        # checkpoint or the next rollout.
        candidate_final_mean = policy_mean(train_state.params, batch.obs)
        candidate_final_log_std = effective_log_std(
            train_state.params["log_std"], min_log_std, max_log_std
        )
        candidate_final_logp = normal_logprob(
            batch.action, candidate_final_mean, candidate_final_log_std
        )
        candidate_final_approx_kl = jnp.mean(
            batch.old_logp - candidate_final_logp
        )
        candidate_final_exact_kl = diagonal_gaussian_kl(
            policy_mean(behavior_train_state.params, batch.obs),
            old_log_std,
            candidate_final_mean,
            candidate_final_log_std,
        )
        target_kl_rollback = jnp.asarray(False)
        if target_kl is not None and float(target_kl) > 0.0:
            target_kl_rollback = candidate_final_exact_kl > float(target_kl)

        # A small trust-region budget per PPO update does not bound the
        # cumulative displacement over an uncapped continuation.  Audit the
        # candidate actor on the frozen source-state replay set.  If the full
        # update crosses the absolute boundary, interpolate the parameter
        # displacement and retain the largest safe fraction instead of
        # discarding all minibatches.  This preserves the same hard source
        # behavior bound without deadlocking adaptation at its edge.
        actor_anchor_replay_candidate_kl = jnp.asarray(0.0, dtype=jnp.float32)
        actor_anchor_replay_post_kl = jnp.asarray(0.0, dtype=jnp.float32)
        actor_anchor_replay_guard_triggered = jnp.asarray(False)
        actor_anchor_replay_projection_triggered = jnp.asarray(False)
        actor_anchor_replay_projection_scale = jnp.asarray(
            1.0, dtype=jnp.float32
        )
        actor_anchor_replay_no_safe_step = jnp.asarray(False)
        if float(actor_anchor_replay_max_kl) > 0.0:
            reference_replay_mean = jax.lax.stop_gradient(
                policy_mean(reference_params, actor_anchor_replay_obs)
            )
            reference_replay_log_std = jax.lax.stop_gradient(
                effective_log_std(
                    reference_params["log_std"], min_log_std, max_log_std
                )
            )

            def replay_kl(params):
                return diagonal_gaussian_kl(
                    reference_replay_mean,
                    reference_replay_log_std,
                    policy_mean(params, actor_anchor_replay_obs),
                    effective_log_std(
                        params["log_std"], min_log_std, max_log_std
                    ),
                )

            actor_anchor_replay_candidate_kl = replay_kl(train_state.params)
            actor_anchor_replay_guard_triggered = (
                actor_anchor_replay_candidate_kl
                > float(actor_anchor_replay_max_kl)
            )
            (
                projected_params,
                projected_replay_kl,
                projected_scale,
                found_safe_step,
                _replay_trial_kls,
            ) = backtrack_params_to_kl_limit(
                behavior_train_state.params,
                train_state.params,
                replay_kl,
                float(actor_anchor_replay_max_kl),
            )
            actor_anchor_replay_projection_triggered = (
                actor_anchor_replay_guard_triggered & found_safe_step
            )
            actor_anchor_replay_no_safe_step = (
                actor_anchor_replay_guard_triggered & (~found_safe_step)
            )
            actor_anchor_replay_projection_scale = jnp.where(
                actor_anchor_replay_guard_triggered,
                projected_scale,
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            selected_params = jax.tree_util.tree_map(
                lambda candidate, projected: jnp.where(
                    actor_anchor_replay_guard_triggered,
                    projected,
                    candidate,
                ),
                train_state.params,
                projected_params,
            )
            train_state = TrainState(params=selected_params, opt=train_state.opt)
            actor_anchor_replay_post_kl = jnp.where(
                actor_anchor_replay_guard_triggered,
                projected_replay_kl,
                actor_anchor_replay_candidate_kl,
            )

        rollback_update = target_kl_rollback | actor_anchor_replay_no_safe_step
        if (target_kl is not None and float(target_kl) > 0.0) or (
            float(actor_anchor_replay_max_kl) > 0.0
        ):
            train_state = jax.tree_util.tree_map(
                lambda candidate, previous: jnp.where(
                    rollback_update, previous, candidate
                ),
                train_state,
                behavior_train_state,
            )

        cmdp_dual_after = cmdp_dual_before
        if bool(cmdp_enabled):
            cmdp_dual_after = jnp.clip(
                cmdp_dual_before
                + float(cmdp_dual_learning_rate)
                * (
                    cmdp_observed_costs
                    - jnp.asarray(cmdp_cost_limits, dtype=jnp.float32)
                ),
                0.0,
                float(cmdp_dual_max),
            )
            updated_params = dict(train_state.params)
            updated_params["cmdp_dual"] = cmdp_dual_after
            train_state = TrainState(params=updated_params, opt=train_state.opt)

        accepted_final_mean = policy_mean(train_state.params, batch.obs)
        accepted_final_log_std = effective_log_std(
            train_state.params["log_std"], min_log_std, max_log_std
        )
        accepted_final_logp = normal_logprob(
            batch.action, accepted_final_mean, accepted_final_log_std
        )
        accepted_final_ratio = jnp.exp(accepted_final_logp - batch.old_logp)
        effective_approx_kl = jnp.where(
            rollback_update,
            0.0,
            jnp.mean(batch.old_logp - accepted_final_logp),
        )
        effective_exact_kl = jnp.where(
            rollback_update,
            0.0,
            diagonal_gaussian_kl(
                policy_mean(behavior_train_state.params, batch.obs),
                old_log_std,
                accepted_final_mean,
                accepted_final_log_std,
            ),
        )
        if float(actor_anchor_replay_max_kl) > 0.0:
            actor_anchor_replay_post_kl = replay_kl(train_state.params)
        aux_mean["approx_kl"] = effective_approx_kl
        aux_mean["ppo_exact_kl"] = effective_exact_kl
        aux_mean["ppo_pre_rollback_exact_kl"] = candidate_final_exact_kl
        aux_mean["ppo_update_rolled_back"] = rollback_update.astype(jnp.float32)
        aux_mean["actor_anchor_replay_candidate_kl"] = (
            actor_anchor_replay_candidate_kl
        )
        aux_mean["actor_anchor_replay_post_kl"] = actor_anchor_replay_post_kl
        aux_mean["actor_anchor_replay_guard_triggered"] = (
            actor_anchor_replay_guard_triggered.astype(jnp.float32)
        )
        aux_mean["actor_anchor_replay_projection_triggered"] = (
            actor_anchor_replay_projection_triggered.astype(jnp.float32)
        )
        aux_mean["actor_anchor_replay_projection_scale"] = (
            actor_anchor_replay_projection_scale
        )
        aux_mean["actor_anchor_replay_no_safe_step"] = (
            actor_anchor_replay_no_safe_step.astype(jnp.float32)
        )
        aux_mean["ppo_minibatches_applied"] = jnp.where(
            rollback_update, 0.0, aux_mean["ppo_minibatches_applied"]
        )
        aux_mean["ppo_epochs_applied"] = jnp.where(
            rollback_update, 0.0, aux_mean["ppo_epochs_applied"]
        )
        aux_mean["ppo_safe_step_scale"] = jnp.where(
            rollback_update,
            0.0,
            aux_mean["ppo_safe_step_scale"]
            * actor_anchor_replay_projection_scale,
        )
        aux_mean["ppo_candidate_exact_kl"] = jnp.where(
            rollback_update, 0.0, aux_mean["ppo_candidate_exact_kl"]
        )
        aux_mean["clip_frac"] = jnp.where(
            rollback_update,
            0.0,
            jnp.mean(
                (jnp.abs(accepted_final_ratio - 1.0) > float(clip_range)).astype(
                    jnp.float32
                )
            ),
        )
        aux_mean["entropy"] = normal_entropy(accepted_final_log_std)
        if bool(cmdp_enabled):
            aux_mean["cmdp/completed_episodes"] = cmdp_completed_count
            for channel_index, channel_name in enumerate(CMDP_COST_NAMES):
                cost_returns_flat = flatten_time_env(cmdp_cost_returns)[
                    :, channel_index
                ]
                cost_values_flat = flatten_time_env(cmdp_cost_old_values)[
                    :, channel_index
                ]
                aux_mean[f"cmdp/observed_cost/{channel_name}"] = (
                    cmdp_observed_costs[channel_index]
                )
                aux_mean[f"cmdp/cost_limit/{channel_name}"] = jnp.asarray(
                    float(cmdp_cost_limits[channel_index]), dtype=jnp.float32
                )
                aux_mean[f"cmdp/dual_before/{channel_name}"] = (
                    cmdp_dual_before[channel_index]
                )
                aux_mean[f"cmdp/dual/{channel_name}"] = cmdp_dual_after[
                    channel_index
                ]
                aux_mean[f"cmdp/cost_explained_var/{channel_name}"] = (
                    1.0
                    - jnp.var(cost_returns_flat - cost_values_flat)
                    / (jnp.var(cost_returns_flat) + 1.0e-8)
                )
        aux_mean["failure_focus_fraction"] = failure_focus_fraction
        aux_mean["success_focus_fraction"] = success_focus_fraction
        aux_mean["failure_focus_balanced_mass_applied"] = (
            failure_focus_balanced_mass_applied
        )
        aux_mean["success_focus_balanced_mass_applied"] = (
            success_focus_balanced_mass_applied
        )
        aux_mean["outcome_focus_sample_weight_mean"] = (
            outcome_focus_sample_weight_mean
        )
        aux_mean["hard_lane_focus_fraction"] = hard_lane_focus_fraction
        aux_mean["hard_lane_focus_mean_condition_count"] = (
            hard_lane_focus_mean_condition_count
        )
        aux_mean["hard_lane_focus_advantage_abs_fraction"] = (
            hard_lane_focus_advantage_abs_fraction
        )
        aux_mean["hard_lane_focus_weight"] = jnp.asarray(
            float(hard_lane_focus_weight), dtype=jnp.float32
        )
        aux_mean["reset_family_raw_fraction"] = reset_family_raw_fraction
        aux_mean["reset_family_min_actor_mass"] = jnp.asarray(
            float(reset_family_min_actor_mass), dtype=jnp.float32
        )
        aux_mean["reset_family_weighted_fraction"] = (
            reset_family_weighted_fraction
        )
        aux_mean["reset_family_group_weight"] = reset_family_group_weight
        aux_mean["reset_family_main_weight"] = reset_family_main_weight
        aux_mean["reset_family_mass_applied"] = reset_family_mass_applied
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
        "critic_obs_dim": getattr(env, "critic_obs_dim", env.obs_dim),
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
    row_fields = list(row.keys())
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_fields)
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        try:
            fieldnames = next(reader)
        except StopIteration:
            fieldnames = []

    if not fieldnames:
        fieldnames = row_fields

    missing_fields = [name for name in row_fields if name not in fieldnames]
    if missing_fields:
        old_fieldnames = fieldnames
        fieldnames = [*fieldnames, *missing_fields]
        tmp_path = path.with_name(path.name + ".tmp")
        with path.open("r", newline="") as src, tmp_path.open("w", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            try:
                next(reader)
            except StopIteration:
                pass
            writer.writerow(fieldnames)
            for old_row in reader:
                padded = old_row[: len(old_fieldnames)]
                if len(padded) < len(old_fieldnames):
                    padded.extend([""] * (len(old_fieldnames) - len(padded)))
                writer.writerow([*padded, *([""] * len(missing_fields))])
        tmp_path.replace(path)

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
    p.add_argument(
        "--time-limit-bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Bootstrap V(s) at horizon_sec truncations while still cutting the GAE trace "
            "at the episode boundary. Disable only to reproduce legacy runs that treated "
            "the hidden time limit as a true terminal state."
        ),
    )
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.0,
        help=(
            "Maximum analytic KL(old||new) from the rollout behavior policy. "
            "Candidate Adam steps are checked after the update and backtracked; "
            "the complete update is rolled back if its full-rollout KL exceeds "
            "this budget. 0 disables the guard."
        ),
    )
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument(
        "--min-log-std",
        type=float,
        default=None,
        help=(
            "Optional lower bound for the trainable Gaussian policy log standard "
            "deviation. Prevents exploration collapse during long training stages."
        ),
    )
    p.add_argument(
        "--max-log-std",
        type=float,
        default=None,
        help=(
            "Optional upper bound for Gaussian policy log standard deviation. "
            "Useful for low-noise policy polishing after task acquisition."
        ),
    )
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument(
        "--asymmetric-critic",
        action="store_true",
        help="Train the value network with simulator-only privileged observations; actor observations stay unchanged.",
    )
    p.add_argument(
        "--critic-command-history-steps",
        type=int,
        default=4,
        help="Number of recent q_ref command-buffer entries appended only to privileged critic observations.",
    )
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
    env = MjxJuggleEnv(
        args.xml,
        n_envs=args.n_envs,
        cfg=MjxJuggleConfig(
            domain_randomization=False,
            arm_action_limiter=True,
            asymmetric_critic=bool(args.asymmetric_critic),
            critic_command_history_steps=int(args.critic_command_history_steps),
        ),
    )
    print(f"[mjx_ppo] MJX XML: {env.mjx_xml}")
    print(f"[mjx_ppo] n_envs={args.n_envs}, n_steps={args.n_steps}, batch={args.n_envs * args.n_steps}")
    print(
        f"[mjx_ppo] episode_max_steps={env.max_steps}, dt={env.dt:.4f}s, "
        f"horizon={env.max_steps * env.dt:.2f}s, obs_dim={env.obs_dim}, "
        f"critic_obs_dim={getattr(env, 'critic_obs_dim', env.obs_dim)}, "
        f"asymmetric_critic={getattr(env, 'asymmetric_critic', False)}"
    )

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
                "critic_obs_dim": getattr(env, "critic_obs_dim", env.obs_dim),
                "act_dim": env.act_dim,
                "jax_devices": [str(d) for d in jax.devices()],
            },
        )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key, params_key = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_key, args.n_envs)
    env_state, obs = jax.jit(env.reset)(reset_keys)
    critic_obs = env.get_critic_obs(env_state, obs)
    params = init_params(params_key, env.obs_dim, env.act_dim, args.hidden_dim, getattr(env, "critic_obs_dim", env.obs_dim))
    train_state = TrainState(params=params, opt=adam_init(params))
    runner = RunnerState(
        env_state=env_state,
        obs=obs,
        critic_obs=critic_obs,
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
        min_log_std=args.min_log_std,
        max_log_std=args.max_log_std,
        target_kl=args.target_kl,
        time_limit_bootstrap=args.time_limit_bootstrap,
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
