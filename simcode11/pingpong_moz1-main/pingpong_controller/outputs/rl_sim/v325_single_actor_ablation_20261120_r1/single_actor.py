"""Explicit single-MLP initialization and a same-mean zero-residual control."""
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


MODES = ('direct', 'residual_control')
HISTORY_XY_ROWS = tuple(72 + 3 * lag + axis for lag in range(1, 8) for axis in (0, 1))


class ActorOptim(NamedTuple):
    m: Any
    v: Any
    t: Any
    actor_t: Any


def initialize(ppo, parent, *, mode):
    if mode not in MODES:
        raise ValueError('Explicit single-actor comparison mode required')
    if parent.params['teacher_pi']['l1']['w'].shape != (114, 256):
        raise ValueError('Pinned114-D frozen S9 actor required')
    if parent.params['v']['l1']['w'].shape != (453, 256):
        raise ValueError('Pinned453-D critic required')
    copy = lambda x: jnp.array(x)
    actor = jax.tree.map(copy, parent.params['teacher_pi'])
    params = dict(pi=actor, v=jax.tree.map(copy, parent.params['v']),
                  log_std=copy(parent.params['log_std']))
    anchor = jax.tree.map(copy, params)
    if mode == 'residual_control':
        params['teacher_pi'] = jax.tree.map(copy, actor)
        params['pi'] = {**actor, 'out': jax.tree.map(jnp.zeros_like, actor['out'])}
        params['residual_action_scale'] = jnp.full((7,), 2., jnp.float32)
    moments = []
    for old in (parent.opt.m, parent.opt.v):
        current = jax.tree.map(jnp.zeros_like, params)
        current['v'] = jax.tree.map(copy, old['v'])
        current['log_std'] = copy(old['log_std'])
        moments.append(current)
    opt = ActorOptim(*moments, copy(parent.opt.t), jnp.asarray(0, jnp.int32))
    if mode == 'direct':
        assert_single_actor(params)
    return ppo.TrainState(params, opt), anchor


def assert_single_actor(params):
    """A direct artifact has exactly one actor and no residual dispatch metadata."""
    if set(params) != {'pi', 'v', 'log_std'}:
        raise ValueError('DirectactorallowsONLYpi,v,log_std;no residual/source/teacher branches')
    if set(params['pi']) != {'l1', 'l2', 'out'}:
        raise ValueError('Directactorrequiresoneordinarytwo-hidden-layerMLP')
    if params['pi']['l1']['w'].shape != (114, 256) or params['pi']['out']['w'].shape != (256, 7):
        raise ValueError('Directactorinput/outputcontractchanged')


def adam_factory(previous_adam, *, history_xy_epsilon=1.e-5, history_xy_lr_multiplier=1.):
    """Keep old critic updates; use local bias correction for each new actor."""
    def step(params, grads, opt, learning_rate, max_grad_norm, beta1=.9, beta2=.999, eps=1.e-5):
        updated, after, norm = previous_adam(params, grads, opt, learning_rate, max_grad_norm, beta1, beta2, eps)
        if not isinstance(opt, ActorOptim):
            return updated, after, norm
        active = sum(jnp.sum(x*x) for x in jax.tree.leaves(grads['pi'])) > 0.
        actor_t = opt.actor_t + active.astype(jnp.int32)
        age = jnp.maximum(actor_t, 1)
        first = jax.tree.map(lambda x: x/(1.-beta1**age), after.m['pi'])
        second = jax.tree.map(lambda x: x/(1.-beta2**age), after.v['pi'])
        candidate = jax.tree.map(lambda p, m, v: p-float(learning_rate)*m/(jnp.sqrt(v)+float(eps)),
                                 params['pi'], first, second)
        if history_xy_epsilon != eps or history_xy_lr_multiplier != 1.:
            rows = jnp.asarray(HISTORY_XY_ROWS)
            weights = params['pi']['l1']['w']
            selected = weights[rows] - float(learning_rate)*history_xy_lr_multiplier*first['l1']['w'][rows]/(
                jnp.sqrt(second['l1']['w'][rows]) + history_xy_epsilon)
            candidate = {**candidate, 'l1': {**candidate['l1'], 'w': candidate['l1']['w'].at[rows].set(selected)}}
        actor = jax.tree.map(lambda new, old: jnp.where(active, new, old), candidate, params['pi'])
        m = {**after.m, 'pi': jax.tree.map(lambda new, old: jnp.where(active, new, old), after.m['pi'], opt.m['pi'])}
        v = {**after.v, 'pi': jax.tree.map(lambda new, old: jnp.where(active, new, old), after.v['pi'], opt.v['pi'])}
        return {**updated, 'pi': actor}, ActorOptim(m, v, after.t, actor_t), norm
    step._single_actor_ablation = True
    return step
