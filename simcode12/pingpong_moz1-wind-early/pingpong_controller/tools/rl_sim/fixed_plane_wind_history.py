"""Causal ball-camera history for the explicitly selected V325/V326 tail.

Native 72-D actor and 411-D critic prefixes retain their original ordering.
History contains observed positions/timestamps only, never privileged wind.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field, make_dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp

LAGS = (0, 10, 20, 30, 40, 60, 80, 100)
BUFFER_SIZE = 101
EXTRA_DIM = 42


@dataclass(frozen=True)
class HistoryConfig:
    enabled: bool = False
    revision: str = 'observed_ball_500ms_age_valid_bound_20261113'


def history_contract() -> dict:
    return dict(revision=HistoryConfig().revision, lags_ms=[v * 5 for v in LAGS],
                suffix_dim=EXTRA_DIM, actor_critic_action=[114, 453, 7],
                native_prefixes=[72, 411], position='world observed position minus latest observed position; metres',
                age='actual camera sample age / 0.6s, clipped at 1',
                valid='observed, finite, age <= 0.6s; missing past remains invalid',
                controller_bounds='current roll/pitch authority divided by 20 degrees',
                privileged_wind_in_actor=False, reset='clear completed lanes only',
                migration='append zero input rows to actor, critic and both Adam moments')


@lru_cache(maxsize=4)
def _config_type(parent_type: type) -> type:
    return make_dataclass('WindHistoryConfig',
        [('wind_history', HistoryConfig, field(default_factory=HistoryConfig))],
        bases=(parent_type,), frozen=True)


def with_history_config(cfg, lesson):
    if not getattr(lesson, 'history_enabled', False):
        return cfg
    return _config_type(type(cfg))(**vars(cfg), wind_history=HistoryConfig(enabled=True))


def dimensions_for_config(cfg) -> tuple[int, int, int]:
    return ((114, 453, 7) if getattr(getattr(cfg, 'wind_history', None), 'enabled', False)
            else (72, 411, 7))


class HistoryState(namedtuple('HistoryStateBase',
                             'core ball_positions ball_sample_steps ball_valid')):
    """Forward native state access without expanding native observation buffers."""
    __slots__ = ()

    def __getattr__(self, name):
        return getattr(self.core, name)

    def _replace(self, **kwargs):
        own = {k: v for k, v in kwargs.items() if k in self._fields}
        native = {k: v for k, v in kwargs.items() if k not in self._fields}
        if native:
            own['core'] = own.get('core', self.core)._replace(**native)
        return super()._replace(**own)


def _sample(core):
    pos = core.cached_ball_obs_pos
    valid = core.fixed_contact_ball_observed & jnp.all(jnp.isfinite(pos), axis=-1)
    return jnp.where(valid[:, None], pos, 0.), core.last_ball_obs_step, valid


def initialize_history(core) -> HistoryState:
    pos, steps, valid = _sample(core)
    n = pos.shape[0]
    return HistoryState(core,
        jnp.zeros((n, BUFFER_SIZE, 3), pos.dtype).at[:, 0].set(pos),
        jnp.zeros((n, BUFFER_SIZE), steps.dtype).at[:, 0].set(steps),
        jnp.zeros((n, BUFFER_SIZE), jnp.bool_).at[:, 0].set(valid))


def advance_history(state: HistoryState, core) -> HistoryState:
    pos, steps, valid = _sample(core)
    return HistoryState(core,
        jnp.concatenate((pos[:, None], state.ball_positions[:, :-1]), axis=1),
        jnp.concatenate((steps[:, None], state.ball_sample_steps[:, :-1]), axis=1),
        jnp.concatenate((valid[:, None], state.ball_valid[:, :-1]), axis=1))


def history_suffix(state: HistoryState, bounds_deg) -> jax.Array:
    indices = jnp.asarray(LAGS)
    positions = state.ball_positions[:, indices]
    age_s = jnp.maximum(state.core.step_count[:, None]
                        - state.ball_sample_steps[:, indices], 0) * .005
    valid = state.ball_valid[:, indices] & (age_s <= .6)
    delta_valid = valid & valid[:, :1]
    deltas = jnp.where(delta_valid[:, :, None], positions - positions[:, :1], 0.)
    ages = jnp.where(valid, jnp.clip(age_s / .6, 0., 1.), 0.)
    return jnp.concatenate((deltas.reshape((positions.shape[0], 24)), ages,
                            valid.astype(positions.dtype), bounds_deg / 20.), axis=-1)


def history_environment(inner_class):
    """Wrap only opted-in tail stages; leave legacy environments untouched."""
    class HistoryEnv:
        def __init__(self, inner):
            self.inner = inner
            self.obs_dim, self.critic_obs_dim, self.act_dim = 114, 453, 7

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def _obs(self, state, native_obs):
            bounds = jnp.rad2deg(self.inner.wind_pose_limits(state.core)[1])
            return jnp.concatenate((native_obs, history_suffix(state, bounds)), axis=-1)

        def attach(self, core, native_obs):
            prepare = getattr(self.inner, 'attach_training_state', None)
            if prepare is not None:
                core = prepare(core)
            state = initialize_history(core)
            return state, self._obs(state, native_obs)

        def reset(self, keys):
            core, obs = self.inner.reset(keys)
            return self.attach(core, obs)

        def step(self, state, action):
            core, obs, reward, done, metrics = self.inner.step(state.core, action)
            state = advance_history(state, core)
            return state, self._obs(state, obs), reward, done, metrics

        def reset_done(self, state, obs, done, keys):
            core, native_obs = self.inner.reset_done(state.core, obs[:, :72], done, keys)
            fresh = initialize_history(core)
            select = lambda new, old: jnp.where(
                done.reshape((done.shape[0],) + (1,) * (new.ndim - 1)), new, old)
            state = HistoryState(core,
                select(fresh.ball_positions, state.ball_positions),
                select(fresh.ball_sample_steps, state.ball_sample_steps),
                select(fresh.ball_valid, state.ball_valid))
            return state, self._obs(state, native_obs)

        def get_critic_obs(self, state, obs):
            return jnp.concatenate((self.inner.get_critic_obs(state.core, obs[:, :72]),
                                    obs[:, 72:]), axis=-1)

    def factory(*args, **kwargs):
        inner = inner_class(*args, **kwargs)
        return HistoryEnv(inner) if dimensions_for_config(inner.cfg)[0] == 114 else inner
    return factory


def expand_reference_params(params, *, actor_only: bool = False):
    """Zero-pad an authenticated 72-D actor anchor or complete network tree."""
    result = dict(params)
    for name, old_width in (('pi', 72), ('v', 411)):
        if actor_only and name != 'pi':
            continue
        if name not in result:
            continue
        net = result[name]
        width = net['l1']['w'].shape[0]
        if width not in (old_width, old_width + EXTRA_DIM):
            raise ValueError('Unsupported history network input layout')
        if width == old_width:
            result[name] = {**net, 'l1': {**net['l1'],
                'w': jnp.pad(jnp.asarray(net['l1']['w']), ((0, EXTRA_DIM), (0, 0)))}}
    return result


def expand_history_learner(state):
    return state._replace(params=expand_reference_params(state.params),
        opt=state.opt._replace(m=expand_reference_params(state.opt.m),
                               v=expand_reference_params(state.opt.v)))


def pad_replay_observations(obs):
    """Old replay has no past frames; known legacy authority is two degrees."""
    obs = jnp.asarray(obs)
    if obs.shape[-1] == 114:
        return obs
    if obs.shape[-1] != 72:
        raise ValueError('History replay must have 72 or 114 columns')
    extra = jnp.zeros((*obs.shape[:-1], EXTRA_DIM), obs.dtype).at[..., -2:].set(.1)
    return jnp.concatenate((obs, extra), axis=-1)
