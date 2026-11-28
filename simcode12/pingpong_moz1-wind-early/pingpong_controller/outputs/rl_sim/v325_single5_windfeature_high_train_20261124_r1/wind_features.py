"""Opt-in input columns in one ordinary MLP; never sum policy outputs."""
import jax
import jax.numpy as jnp
import numpy as np

OLD_DIM = 118
NEW_DIM = 120
DEADBAND_M_S = 1.25
ACTIVATION_WIDTH_M_S = .35


def encode(obs, *, xp):
    """Append two causal transforms of the existing estimated wind XY.

    Neither true wind, wind population labels nor the future onset schedule
    are inputs. The first118 values are copied exactly. Small estimated wind
    and all pre-fifth-hit states retain zero additional features.
    """
    if obs.shape[-1] != OLD_DIM:
        raise ValueError('Expected the registered118-D raw actor observation')
    estimated = obs[..., 114:116]
    speed_m_s = 1.6 * xp.sqrt(xp.sum(estimated * estimated, axis=-1))
    gain = xp.clip((speed_m_s-DEADBAND_M_S)/ACTIVATION_WIDTH_M_S, 0., 1.)
    gain = xp.where(obs[..., 112] > 0., gain, 0.)
    return xp.concatenate([obs, estimated * gain[..., None]], axis=-1)


def migrate_params(params):
    """Preserve every old value; append two zero first-layer input rows."""
    if set(params) != {'pi', 'v', 'log_std'}:
        raise ValueError('Expected one ordinary actor, critic and log sigma')
    if params['pi']['l1']['w'].shape[0] != OLD_DIM:
        raise ValueError('Only the pinned118-D single-actor layout may migrate')
    result = jax.tree.map(lambda value: jnp.asarray(np.asarray(value).copy()), params)
    first = result['pi']['l1']
    result['pi']['l1'] = {**first, 'w': jnp.pad(first['w'], ((0, 2), (0, 0)))}
    return result


def apply_actor(pi, obs):
    """Evaluate the same dense MLP with an explicit zero-input identity.

    Splitting the first matrix by its input columns preserves legacy floating
    point evaluation order. Both parts feed the SAME first hidden activation;
    there is no extra network, action sum, teacher, BC or residual policy.
    Selecting the old affine value for zero new inputs prevents a compiler
    reassociation from changing the preserved path.
    """
    if obs.shape[-1] != NEW_DIM or pi['l1']['w'].shape[0] != NEW_DIM:
        raise ValueError('Expected the opt-in120-D ordinary actor')
    old_affine = jnp.matmul(obs[..., :OLD_DIM], pi['l1']['w'][:OLD_DIM],
                           precision=jax.lax.Precision.HIGHEST) + pi['l1']['b']
    extra = jnp.matmul(obs[..., OLD_DIM:], pi['l1']['w'][OLD_DIM:],
                       precision=jax.lax.Precision.HIGHEST)
    active = jnp.any(obs[..., OLD_DIM:] != 0., axis=-1, keepdims=True)
    x = jnp.tanh(jnp.where(active, old_affine + extra, old_affine))
    for name in ('l2', 'l3'):
        x = jnp.tanh(jnp.matmul(x, pi[name]['w'],
                               precision=jax.lax.Precision.HIGHEST) + pi[name]['b'])
    return jnp.matmul(x, pi['out']['w'], precision=jax.lax.Precision.HIGHEST) + pi['out']['b']


class WindFeatureEnv:
    """Stateless actor-only suffix; native dynamics/reward/critic unchanged."""

    def __init__(self, inner):
        if inner.obs_dim != OLD_DIM or inner.critic_obs_dim != 453:
            raise ValueError('Unsupported native observation contract')
        self.inner = inner
        self.obs_dim, self.critic_obs_dim, self.act_dim = NEW_DIM, 453, inner.act_dim

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self, keys):
        state, raw = self.inner.reset(keys)
        return state, encode(raw, xp=jnp)

    def reset_done(self, state, obs, done, keys):
        state, raw = self.inner.reset_done(state, obs[..., :OLD_DIM], done, keys)
        return state, encode(raw, xp=jnp)

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state, obs[..., :OLD_DIM])

    def _obs(self, state, native_obs):
        return encode(self.inner._obs(state, native_obs), xp=jnp)

    def step(self, state, action):
        after, raw, reward, done, metrics = self.inner.step(state, action)
        return after, encode(raw, xp=jnp), reward, done, metrics
