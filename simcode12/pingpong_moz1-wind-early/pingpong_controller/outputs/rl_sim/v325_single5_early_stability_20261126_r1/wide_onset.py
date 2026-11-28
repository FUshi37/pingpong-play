"""Stratified wide wind-onset distribution for V325 r5 robustness training.

Extends RandomOnsetEnv by overriding only `_fresh` so OnsetState pytree layout,
causal acquisition_time (fifth_hit_s), and all timing/preview diagnostics remain
100% compatible with existing checkpoints and PPO code.
"""
import jax
import jax.numpy as jnp
from randomized_onset import OnsetState, RandomOnsetEnv


class StratifiedWideOnsetEnv(RandomOnsetEnv):
    """Sample per-episode post-hit-5 wind onset delay from a stratified wide distribution.

    - Rehearsal lanes (e.g. 25s ~= 48 hits):
        early_prob       -> [early_min, early_max] steps after hit 5  (default 5-100: hits 5-6)
        1 - early_prob   -> [mid_min, mid_max] steps after hit 5      (default 500-2000: hits 10-25,
                            leaving >= 12s post-onset wind testing inside a 25s rehearsal)
    - Long lanes (1200s ~= 2300 hits):
        early_prob             -> [early_min, early_max] steps        (hits 5-6)
        long_mid_prob          -> [mid_min, mid_max] steps            (hits 10-25)
        1 - early - long_mid   -> [late_min, late_max] steps          (default 2000-15000: hits 25-150)
    """

    def __init__(self, inner, *, rehearsal_mask, onset_cfg: dict, enabled: bool = True):
        super().__init__(inner, enabled=enabled)
        self.rehearsal_mask = jnp.asarray(rehearsal_mask, dtype=jnp.bool_)
        self.early_prob = float(onset_cfg.get("early_prob", 0.20))
        self.long_mid_prob = float(onset_cfg.get("long_mid_prob", 0.50))
        self.early_min = int(onset_cfg.get("delay_steps_min", 5))
        self.early_max = int(onset_cfg.get("delay_steps_max", 100))
        self.mid_min = int(onset_cfg.get("mid_delay_steps_min", 500))
        self.mid_max = int(onset_cfg.get("mid_delay_steps_max", 2000))
        self.late_min = int(onset_cfg.get("late_delay_steps_min", 2000))
        self.late_max = int(onset_cfg.get("late_delay_steps_max", 15000))
        self.onset_salt = int(onset_cfg.get("salt", 2026112705))
        assert 0.0 <= self.early_prob <= 1.0
        assert 0.0 <= self.long_mid_prob <= 1.0 - self.early_prob + 1e-9
        assert 0 <= self.early_min <= self.early_max
        assert 0 <= self.mid_min <= self.mid_max
        assert 0 <= self.late_min <= self.late_max

    def _fresh(self, core, keys):
        count = core.step_count.shape[0]

        def _sample_one(key, is_reh):
            k0 = jax.random.fold_in(key, self.onset_salt)
            ku, k1, k2, k3 = jax.random.split(k0, 4)
            u = jax.random.uniform(ku, (), dtype=jnp.float32)
            d_early = jax.random.randint(k1, (), self.early_min, self.early_max + 1, dtype=jnp.int32)
            d_mid = jax.random.randint(k2, (), self.mid_min, self.mid_max + 1, dtype=jnp.int32)
            d_late = jax.random.randint(k3, (), self.late_min, self.late_max + 1, dtype=jnp.int32)
            reh_delay = jnp.where(u < self.early_prob, d_early, d_mid)
            long_delay = jnp.where(
                u < self.early_prob,
                d_early,
                jnp.where(u < (self.early_prob + self.long_mid_prob), d_mid, d_late),
            )
            return jnp.where(is_reh, reh_delay, long_delay)

        delay = jax.vmap(_sample_one)(keys, self.rehearsal_mask)
        zero = jnp.zeros(count, jnp.float32)
        step = jnp.zeros(count, jnp.int32)
        flag = jnp.zeros(count, bool)
        return OnsetState(core, zero - 1.0, delay, step, step, flag, flag, zero, zero, zero, zero, step)
