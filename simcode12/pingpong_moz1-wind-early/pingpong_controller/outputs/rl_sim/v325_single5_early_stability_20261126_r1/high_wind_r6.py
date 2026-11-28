"""High-wind (up to 4.0 m/s, +/-15 deg pose) opt-in stack for V325 Stage 6 (r6).

Preserves all frozen V325 sources untouched while providing:
  1. Unclipped causal wind-force estimation up to `obs_wind_max_m_s` (default 4.5 m/s)
     using the exact 4.0 g pingpong quadratic drag equation (zero privileged inputs).
  2. Exact scale-preserving actor warm-start from 1.6 m/s checkpoints (`u6247`):
     multiplying `pi['l1']['w'][114:116]` and `pi['l1']['w'][118:120]` by `new_max / 1.6`
     yields mathematically identical outputs on 0..1.6 m/s at step 0 while keeping all
     actor inputs cleanly normalized in [-1, 1] up to 4.5 m/s.
  3. Relaxed constant pose authority (`pose_limit_deg`, default 15.0 deg) wired through
     `fixed_plane_wind_pose.effective_degrees` and `WindConfig(max_speed_m_s=4.0)`.
  4. Mixed Clean (10%) + Constant X-cone (70%: 25% peak + 45% 4 bands) + Dynamic (20%:
     10% Slow OU + 10% Gust) sampling with a dynamic `active_max_speed` curriculum
     field (2.4 -> 2.8 -> 3.2 -> 3.6 -> 4.0 m/s) requiring ZERO XLA recompilation.
"""
from __future__ import annotations

from collections import deque, namedtuple
from dataclasses import replace
import math
import numpy as np
import jax
import jax.numpy as jnp

from migrated_stack import setup_sys_path
setup_sys_path()

from fixed_plane_v325_wind import wind_velocity
from fixed_plane_wind_direction_domain import constrain_wind
from observed_force import select_consistent_wind
from single5_runtime import Single5Env, install_pose_mode
from wind_features import WindFeatureEnv, OLD_DIM, NEW_DIM, DEADBAND_M_S, ACTIVATION_WIDTH_M_S

DEFAULT_OBS_WIND_MAX_M_S = 4.5


def make_constant_degrees(pose_deg: float):
    deg = float(pose_deg)
    if not (2.0 <= deg <= 20.0):
        raise ValueError(f"pose_deg must be in [2.0, 20.0], got {deg}")

    def _const_deg(state, dt_s, cfg, *, xp):
        return xp.ones_like(state.step_count, dtype=xp.float32) * deg

    return _const_deg


def configure_high_wind_cfg(
    cfg,
    *,
    pose_deg: float = 15.0,
    max_speed_m_s: float = 4.0,
    force_limit_n: float = 0.025,
):
    """Return updated physical cfg with relaxed pose limit and high-wind ceiling."""
    wcfg = replace(
        cfg.wind_config,
        max_speed_m_s=float(max_speed_m_s),
        force_limit_n=float(force_limit_n),
        mode_weights=(0.0, 0.0, 0.5, 0.5),
        wind_y_nonpositive=True,
        startup_no_wind_hits=5,
        startup_ramp_s=1.0,
    )
    wpose = replace(
        cfg.wind_pose,
        start_deg=float(pose_deg),
        target_deg=float(pose_deg),
        ramp_updates=0,
        quality_gated=False,
        origin_steps=0,
        elapsed_steps_offset=0,
    )
    return replace(
        cfg,
        cartesian_pose_roll_limit_deg=float(pose_deg),
        cartesian_pose_pitch_limit_deg=float(pose_deg),
        wind_pose=wpose,
        wind_config=wcfg,
    )


def install_high_wind_pose_mode(pose_deg: float = 15.0):
    """Install relative-normal dynamics guidance patch and constant `pose_deg` limit."""
    install_pose_mode()
    import fixed_plane_wind_pose

    fixed_plane_wind_pose.effective_degrees = make_constant_degrees(pose_deg)


def observed_force_features_r6(
    observations,
    *,
    max_wind_m_s: float = DEFAULT_OBS_WIND_MAX_M_S,
    ball_mass_kg: float = 0.004,
    xp,
):
    """Causal 4-D actor wind estimate (..., 4): wind_xy / max_wind_m_s, age / 0.6, valid.

    Identical to `observed_force.observed_force_features` except the clamp & normalization
    scale `max_wind_m_s` is configurable (4.5 m/s for r6 instead of hardcoded 1.6 m/s).
    """
    if observations.shape[-1] < 112:
        raise ValueError("Observed-force features require the registered 112-D history prefix")
    wmax = float(max_wind_m_s)
    positions = observations[..., 72:96].reshape(observations.shape[:-1] + (8, 3))
    ages = 0.6 * observations[..., 96:104]
    finite = xp.all(xp.isfinite(positions) & (xp.abs(positions) < 5.0), axis=-1)
    valid = (observations[..., 104:112] > 0.5) & finite & xp.isfinite(ages)
    positions = xp.where(finite[..., None], positions, 0.0)
    ages = xp.where(xp.isfinite(ages), ages, 0.0)

    candidates, candidate_ages, candidate_valid = [], [], []
    coefficient_m_inv = 0.5 * 1.2 * 0.47 * math.pi * (0.02 ** 2) / float(ball_mass_kg)
    for index in range(6):
        dt01 = ages[..., index + 1] - ages[..., index]
        dt12 = ages[..., index + 2] - ages[..., index + 1]
        safe01, safe12 = xp.maximum(dt01, 1.0e-6), xp.maximum(dt12, 1.0e-6)
        v01 = (positions[..., index, :] - positions[..., index + 1, :]) / safe01[..., None]
        v12 = (positions[..., index + 1, :] - positions[..., index + 2, :]) / safe12[..., None]
        acceleration = 2.0 * (v01 - v12) / (safe01 + safe12)[..., None]
        velocity = v01 - 0.5 * safe12[..., None] * acceleration
        usable = (
            xp.all(valid[..., index : index + 3], axis=-1)
            & (dt01 > 0.02)
            & (dt12 > 0.02)
            & (acceleration[..., 2] > -11.5)
            & (acceleration[..., 2] < -8.5)
        )
        speed = xp.sqrt(xp.sum(velocity ** 2, axis=-1))
        b = acceleration[..., :2] / coefficient_m_inv - speed[..., None] * velocity[..., :2]
        b_sq = xp.sum(b ** 2, axis=-1)
        vz_sq = velocity[..., 2] ** 2
        relative_speed = xp.sqrt(xp.maximum(0.5 * (vz_sq + xp.sqrt(vz_sq ** 2 + 4.0 * b_sq)), 1.0e-12))
        wind = velocity[..., :2] + b / relative_speed[..., None]
        wind = xp.stack([wind[..., 0], xp.minimum(wind[..., 1], 0.0)], axis=-1)
        norm = xp.sqrt(xp.sum(wind ** 2, axis=-1))
        wind = wind * xp.minimum(1.0, wmax / xp.maximum(norm, 1.0e-6))[..., None]
        candidates.append(wind)
        candidate_ages.append(xp.clip(0.5 * (ages[..., index] + ages[..., index + 2]), 0.0, 0.6))
        candidate_valid.append(usable)

    winds = xp.stack(candidates, axis=-2)
    usable = xp.stack(candidate_valid, axis=-1)
    estimate, estimate_age, chosen = select_consistent_wind(
        winds, usable, xp.stack(candidate_ages, axis=-1), xp=xp
    )
    mature = xp.any(valid & (ages >= 0.25), axis=-1)
    chosen = chosen & mature
    estimate = xp.where(chosen[..., None], estimate / wmax, 0.0)
    estimate_age = xp.where(chosen, estimate_age / 0.6, 1.0)
    return xp.concatenate(
        [estimate, estimate_age[..., None], chosen.astype(observations.dtype)[..., None]], axis=-1
    )


def encode_r6(
    obs,
    *,
    max_wind_m_s: float = DEFAULT_OBS_WIND_MAX_M_S,
    deadband_m_s: float = DEADBAND_M_S,
    activation_width_m_s: float = ACTIVATION_WIDTH_M_S,
    xp,
):
    """Append 2-D deadbanded strong-wind suffix using configurable `max_wind_m_s`."""
    if obs.shape[-1] != OLD_DIM:
        raise ValueError("Expected 118-D actor observation")
    estimated = obs[..., 114:116]
    speed_m_s = float(max_wind_m_s) * xp.sqrt(xp.sum(estimated * estimated, axis=-1))
    gain = xp.clip((speed_m_s - float(deadband_m_s)) / float(activation_width_m_s), 0.0, 1.0)
    gain = xp.where(obs[..., 112] > 0.0, gain, 0.0)
    return xp.concatenate([obs, estimated * gain[..., None]], axis=-1)


def rescale_actor_wind_inputs(params: dict, *, old_max_m_s: float = 1.6, new_max_m_s: float = DEFAULT_OBS_WIND_MAX_M_S):
    """Exact linear input-row rescaling so 0..old_max_m_s predictions are preserved identically."""
    ratio = float(new_max_m_s) / float(old_max_m_s)
    if abs(ratio - 1.0) < 1e-9:
        return params
    pi = dict(params["pi"])
    l1 = dict(pi["l1"])
    w = jnp.asarray(l1["w"], dtype=jnp.float32)
    w = w.at[114:116].multiply(ratio).at[118:120].multiply(ratio)
    l1["w"] = w
    pi["l1"] = l1
    return {**params, "pi": pi}


class HighWindSingle5Env(Single5Env):
    """Single5Env subclass with configurable unclipped `obs_wind_max_m_s`."""

    def __init__(self, inner, *, obs_wind_max_m_s: float = DEFAULT_OBS_WIND_MAX_M_S, **kwargs):
        super().__init__(inner, **kwargs)
        self.obs_wind_max_m_s = float(obs_wind_max_m_s)

    def _encode(self, state, old_obs):
        from single5_runtime import acquisition_features

        return jnp.concatenate(
            [
                old_obs[:, :112],
                acquisition_features(state, xp=jnp),
                observed_force_features_r6(old_obs, max_wind_m_s=self.obs_wind_max_m_s, xp=jnp),
            ],
            axis=-1,
        )

    def step(self, state, action):
        after, encoded, reward, done, metrics = super().step(state, action)
        if hasattr(after, "wind_background_m_s"):
            actual = wind_velocity(after, after.step_count * self.dt, self.cfg.wind_config)[:, :2]
            estimated = encoded[:, 114:116] * self.obs_wind_max_m_s
            metrics["observed_force/error_sq_m2_s2"] = jnp.sum((estimated - actual) ** 2, axis=-1)
        return after, encoded, reward, done, metrics


class HighWindFeatureEnv(WindFeatureEnv):
    """WindFeatureEnv subclass decoding estimated wind speed with `obs_wind_max_m_s`."""

    def __init__(self, inner, *, obs_wind_max_m_s: float = DEFAULT_OBS_WIND_MAX_M_S):
        super().__init__(inner)
        self.obs_wind_max_m_s = float(obs_wind_max_m_s)

    def _enc(self, raw):
        return encode_r6(raw, max_wind_m_s=self.obs_wind_max_m_s, xp=jnp)

    def reset(self, keys):
        state, raw = self.inner.reset(keys)
        return state, self._enc(raw)

    def reset_done(self, state, obs, done, keys):
        state, raw = self.inner.reset_done(state, obs[..., :OLD_DIM], done, keys)
        return state, self._enc(raw)

    def _obs(self, state, native_obs):
        return self._enc(self.inner._obs(state, native_obs))

    def step(self, state, action):
        after, raw, reward, done, metrics = self.inner.step(state, action)
        return after, self._enc(raw), reward, done, metrics


# ---------------------------------------------------------------------------
# High-Wind Stratified Sampling (Clean 10% + Constant 70% + Dynamic 20%)
# with dynamic `active_max_speed` curriculum (zero XLA recompilation!)
# ---------------------------------------------------------------------------

def high_wind_layout(n_envs: int, *, clean_frac: float = 0.10, dyn_frac: float = 0.20, peak_frac: float = 0.25):
    """Assign every lane to a stratified (mode, sector, bin, axis, hard) cell.

    - 10% Clean (mode=0)
    - 20% Dynamic: 10% Slow OU (mode=2) + 10% Gust (mode=3)
    - 70% Constant X-cone (mode=1):
        - 25% of total envs at exact active peak (bins=4; 62.5% 4-sector, 37.5% pure +/-X axis)
        - 45% of total envs stratified across 4 uniform speed bands [0..active_max_speed]
          (75% representative across 4 sectors x 4 bins, 25% hard outer X-axis bin 3)
    """
    clean_n = round(clean_frac * n_envs)
    dyn_n = round(dyn_frac * n_envs)
    ou_n = dyn_n // 2
    gust_n = dyn_n - ou_n
    peak_n = round(peak_frac * n_envs)

    mode = np.ones(n_envs, np.int32)
    sector = np.zeros(n_envs, np.int32)
    bins = np.zeros(n_envs, np.int32)
    axis = np.full(n_envs, -1, np.int32)
    hard = np.zeros(n_envs, bool)

    p = 0
    mode[p : p + clean_n] = 0
    p += clean_n

    mode[p : p + ou_n] = 2
    p += ou_n

    mode[p : p + gust_n] = 3
    p += gust_n

    # Peak constant lanes
    uniform = peak_n * 5 // 8
    sector[p : p + peak_n] = np.arange(peak_n) % 4
    bins[p : p + peak_n] = 4
    axis[p + uniform : p + peak_n] = np.arange(peak_n - uniform) % 2
    p += peak_n

    # Remaining stratified constant lanes (bins 0..3)
    rem = n_envs - p
    representative = (rem * 3 // 4) // 16 * 16
    slots = np.arange(representative)
    sector[p : p + representative] = slots % 4
    bins[p : p + representative] = (slots // 4) % 4
    hard_start = p + representative
    sector[hard_start:] = np.where(np.arange(n_envs - hard_start) % 2 == 0, 0, 3)
    bins[hard_start:] = 3
    hard[hard_start:] = True
    return mode, sector, bins, axis, hard


def high_wind_rehearsal_mask(n_envs: int, fraction: float = 0.65) -> np.ndarray:
    """Select `fraction` of lanes within every exact high_wind_layout cell."""
    mode, sector, bins, axis, hard = high_wind_layout(n_envs)
    groups = {}
    for lane in range(n_envs):
        key = (int(mode[lane]), int(sector[lane]), int(bins[lane]), int(axis[lane]), bool(hard[lane]))
        groups.setdefault(key, []).append(lane)
    mask = np.zeros(n_envs, dtype=bool)
    for key in sorted(groups):
        lanes = groups[key]
        k = max(1, min(len(lanes) - 1, int(round(len(lanes) * fraction)))) if len(lanes) > 1 else 0
        chosen_idx = np.linspace(0, len(lanes) - 1, k, dtype=int) if k > 0 else []
        for idx in chosen_idx:
            mask[lanes[idx]] = True
    return mask


def constant_vectors_scaled(uniforms, assignments, half_angle_deg: float, max_speed, *, xp):
    """Sample constant X-cone wind vectors scaled to dynamic per-lane `max_speed`."""
    mode, sector, bins, axis, hard = [xp.asarray(x) for x in assignments]
    width = float(half_angle_deg) * math.pi / 180.0
    band_w = max_speed / 4.0
    low = xp.minimum(bins, 3).astype(xp.float32) * band_w
    high = low + band_w
    radius = xp.where(bins == 4, max_speed, xp.sqrt(low * low + uniforms[:, 0] * (high * high - low * low)))
    offset = (sector % 2 + uniforms[:, 1]) * (width / 2.0)
    angle = xp.where(sector < 2, -math.pi + offset, -width + offset)
    hard_angle = xp.where(sector < 2, -math.pi + uniforms[:, 1] * width / 4.0, -uniforms[:, 1] * width / 4.0)
    angle = xp.where(hard, hard_angle, angle)
    angle = xp.where(axis == 0, -math.pi, xp.where(axis == 1, 0.0, angle))
    vec = xp.stack([radius * xp.cos(angle), xp.minimum(radius * xp.sin(angle), 0.0), xp.zeros_like(radius)], axis=-1)
    return vec


HighWindStateBase = namedtuple("HighWindStateBase", ("core", "active_max_speed"))


class HighWindState(HighWindStateBase):
    __slots__ = ()

    def __getattr__(self, name):
        return getattr(self.core, name)

    def _replace(self, **values):
        local = {k: v for k, v in values.items() if k in self._fields}
        fwd = {k: v for k, v in values.items() if k not in self._fields}
        if fwd:
            local["core"] = self.core._replace(**fwd)
        return super()._replace(**local)


def _bound_vec(vec, limit):
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    return vec * jnp.minimum(1.0, limit[:, None] / jnp.maximum(norm, 1.0e-12))


class HighWindSamplingEnv:
    """Stratified Clean + Constant X-cone + Dynamic OU/Gust with dynamic `active_max_speed`."""

    def __init__(self, inner, *, half_angle_deg: float = 45.0, init_max_speed: float = 2.4):
        self.inner = inner
        self.half_angle_deg = float(half_angle_deg)
        self.init_max_speed = float(init_max_speed)
        self.assignment = high_wind_layout(inner.n_envs)
        self.lane_mode = jnp.asarray(self.assignment[0], dtype=jnp.int32)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _assign(self, core, keys, selected, active_max_speed):
        n = self.n_envs
        u = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 2026112760), (6,), dtype=jnp.float32))(keys)
        norm_g = jax.vmap(lambda k: jax.random.normal(jax.random.fold_in(k, 2026112761), (3,), dtype=jnp.float32))(keys)

        # Constant X-cone vectors
        const_bg = constant_vectors_scaled(u[:, :2], self.assignment, self.half_angle_deg, active_max_speed, xp=jnp)

        # Dynamic Slow OU (mode=2) & Gust (mode=3) vectors in wind_y <= 0 half-plane
        def _disk_y_neg(u2, rad):
            ang = -jnp.pi * u2[:, 0]  # [-pi, 0] -> sin(ang) <= 0
            length = rad * jnp.sqrt(u2[:, 1])
            return jnp.stack([length * jnp.cos(ang), jnp.minimum(length * jnp.sin(ang), 0.0), jnp.zeros_like(length)], axis=-1)

        peak_dyn = u[:, 4] < 0.25
        dir_ang = -jnp.pi * u[:, 2]
        unit_dir = jnp.stack([jnp.cos(dir_ang), jnp.minimum(jnp.sin(dir_ang), 0.0), jnp.zeros_like(dir_ang)], axis=-1)

        ou_bg = _disk_y_neg(u[:, 2:4], 0.35 * active_max_speed)
        ou_bg = jnp.where(peak_dyn[:, None], 0.75 * active_max_speed[:, None] * unit_dir, ou_bg)
        ou_vec = _bound_vec(norm_g.at[:, 2].set(0.0) * (0.25 * active_max_speed[:, None]), 0.65 * active_max_speed)
        ou_vec = jnp.where(peak_dyn[:, None], 0.25 * active_max_speed[:, None] * unit_dir, ou_vec)

        gust_bg = _disk_y_neg(u[:, 2:4], 0.15 * active_max_speed)
        gust_vec = _disk_y_neg(u[:, 4:6], 0.85 * active_max_speed)
        gust_vec = jnp.where(peak_dyn[:, None], active_max_speed[:, None] * unit_dir - gust_bg, gust_vec)
        gust_vec = constrain_wind(gust_vec, self.cfg.wind_config, xp=jnp)

        lm = self.lane_mode
        new_bg = jnp.where(
            (lm == 0)[:, None],
            0.0,
            jnp.where((lm == 1)[:, None], const_bg, jnp.where((lm == 2)[:, None], ou_bg, gust_bg)),
        )
        new_ou = jnp.where((lm == 2)[:, None], ou_vec, 0.0)
        new_gust = jnp.where((lm == 3)[:, None], gust_vec, 0.0)

        return core._replace(
            wind_mode=jnp.where(selected, lm, core.wind_mode),
            wind_background_m_s=jnp.where(selected[:, None], new_bg, core.wind_background_m_s),
            wind_ou_m_s=jnp.where(selected[:, None], new_ou, core.wind_ou_m_s),
            wind_gust_vector_m_s=jnp.where(selected[:, None], new_gust, core.wind_gust_vector_m_s),
        )

    def reset(self, keys):
        core, obs = self.inner.reset(keys)
        active = jnp.full((self.n_envs,), self.init_max_speed, dtype=jnp.float32)
        core = self._assign(core, keys, jnp.ones(self.n_envs, dtype=jnp.bool_), active)
        return HighWindState(core, active), obs

    def reset_done(self, state, obs, done, keys):
        core, obs = self.inner.reset_done(state.core, obs, done, keys)
        core = self._assign(core, keys, done, state.active_max_speed)
        return HighWindState(core, state.active_max_speed), obs

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state.core, obs)

    def _obs(self, state, native_obs):
        return self.inner._obs(state.core, native_obs)

    def step(self, state, action):
        after_core, obs, reward, done, metrics = self.inner.step(state.core, action)
        # Keep ongoing OU / rescheduled gusts strictly within current curriculum `active_max_speed`
        active = state.active_max_speed
        bg_norm = jnp.linalg.norm(after_core.wind_background_m_s, axis=-1)
        ou_cap = jnp.minimum(0.65 * active, jnp.maximum(active - bg_norm, 0.1))
        gust_cap = jnp.minimum(0.85 * active, jnp.maximum(active - bg_norm, 0.1))
        after_core = after_core._replace(
            wind_ou_m_s=jnp.where(
                (after_core.wind_mode == 2)[:, None],
                _bound_vec(after_core.wind_ou_m_s, ou_cap),
                0.0,
            ),
            wind_gust_vector_m_s=jnp.where(
                (after_core.wind_mode == 3)[:, None],
                _bound_vec(after_core.wind_gust_vector_m_s, gust_cap),
                0.0,
            ),
        )
        metrics = {**metrics, "curriculum/active_max_speed_m_s": active}
        return HighWindState(after_core, active), obs, reward, done, metrics


# ---------------------------------------------------------------------------
# High-Wind Rehearsal & Progress Trackers (arbitrary max_speed + modes 0..3)
# ---------------------------------------------------------------------------

class HighWindRehearsalEnv:
    """Stratified 25s rehearsal wrapper using `high_wind_rehearsal_mask`."""

    def __init__(self, inner, *, fraction: float = 0.65, seconds: float = 25.0,
                 stagger_min_steps: int = 512, stagger_seed: int = 2026112766):
        from stratified_rehearsal import _RehearsalStateWrapper, initial_staggered_cutoffs

        self._wrap_cls = _RehearsalStateWrapper
        self.inner = inner
        self.seconds = float(seconds)
        self.cutoff_steps = int(round(seconds / inner.dt))
        self.role_np = high_wind_rehearsal_mask(inner.n_envs, fraction=fraction)
        self.role = jnp.asarray(self.role_np, dtype=jnp.bool_)
        self.n_rehearsal = int(self.role_np.sum())
        self.n_long = int((~self.role_np).sum())
        self.init_cutoffs = jnp.asarray(
            initial_staggered_cutoffs(self.role_np, self.cutoff_steps, min_steps=stagger_min_steps, seed=stagger_seed),
            dtype=jnp.int32,
        )
        self.full_cutoffs = jnp.full((inner.n_envs,), self.cutoff_steps, dtype=jnp.int32)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self, keys):
        core, obs = self.inner.reset(keys)
        return self._wrap_cls(core, self.init_cutoffs), obs

    def reset_done(self, state, obs, done, keys):
        core, obs = self.inner.reset_done(state.core, obs, done, keys)
        next_cutoffs = jnp.where(done, self.full_cutoffs, state.rehearsal_cutoff_steps)
        return self._wrap_cls(core, next_cutoffs), obs

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state.core, obs)

    def _obs(self, state, native_obs):
        return self.inner._obs(state.core, native_obs)

    def step(self, state, action):
        after_core, obs, reward, done, metrics = self.inner.step(state.core, action)
        cutoff = self.role & (after_core.step_count >= state.rehearsal_cutoff_steps) & ~done
        full_horizon = (state.rehearsal_cutoff_steps >= self.cutoff_steps).astype(jnp.float32)
        metrics = {
            **metrics,
            "truncated": jnp.where(cutoff, jnp.ones_like(metrics["truncated"]), metrics["truncated"]),
            "sampling/rehearsal_lane": self.role.astype(jnp.float32),
            "sampling/rehearsal_cutoff": cutoff.astype(jnp.float32),
            "sampling/rehearsal_full_horizon": full_horizon,
        }
        return self._wrap_cls(after_core, state.rehearsal_cutoff_steps), obs, reward, done | cutoff, metrics


class HighWindStartupProgress:
    """Sliding-window 25s survival tracker supporting Clean, Constant (arbitrary peak), and Dynamic."""

    def __init__(self, *, seconds: float = 25.0, capacity: int = 4096):
        self.seconds = float(seconds)
        self.outcomes = deque(maxlen=capacity)
        self.completed_episodes = 0

    def clear(self):
        self.outcomes.clear()

    def add(self, metrics: dict, done: np.ndarray, active_max_speed: float):
        done_b = np.asarray(done, bool)
        if not np.any(done_b):
            return
        mode = np.rint(np.asarray(metrics["wind/mode"])).astype(np.int32)
        bg_speed = np.asarray(metrics["polish/background_speed_m_s"], dtype=np.float32)
        angle = np.asarray(metrics["single5/wind_angle_rad"], dtype=np.float32)
        for t, lane in zip(*np.nonzero(done_b)):
            full_h = float(metrics["sampling/rehearsal_full_horizon"][t, lane]) > 0.5
            terminated = float(metrics["terminated"][t, lane]) > 0.5
            cutoff = float(metrics["sampling/rehearsal_cutoff"][t, lane]) > 0.5
            if not full_h and not terminated:
                continue
            duration = float(metrics["diagnostic/episode_seconds"][t, lane])
            hits = float(metrics["hit_count"][t, lane])
            conf_ratio = float(metrics["timing/confirmed_max_ratio"][t, lane])
            phys_ratio = float(metrics["timing/physical_max_ratio"][t, lane])
            success = (
                full_h
                and cutoff
                and not terminated
                and duration >= self.seconds - 1e-5
                and hits >= math.ceil(1.2 * self.seconds)
                and conf_ratio <= 1.0 + 1e-6
                and phys_ratio <= 1.0 + 1e-6
            )
            m = int(mode[t, lane])
            spd = float(bg_speed[t, lane])
            ang = float(angle[t, lane])
            is_peak = (m == 1) and (spd >= 0.98 * active_max_speed)
            is_negx_peak = is_peak and (ang < -math.pi / 2.0)
            is_dir08 = is_peak and (abs(abs(ang) - math.pi) < 1e-3)
            is_dyn = m in (2, 3)
            rz_fail = bool(terminated and float(metrics.get("done/racket_z_out_of_bounds", np.zeros_like(done_b))[t, lane]) > 0.5)
            self.outcomes.append((duration, hits, success, terminated, m, is_peak, is_negx_peak, is_dir08, is_dyn, rz_fail))
            self.completed_episodes += 1

    def summary(self) -> dict:
        n = len(self.outcomes)
        if n == 0:
            return dict(
                n=0, survival=0.0, clean_phys_survival=0.0, peak_survival=0.0,
                negx_peak_survival=0.0, dir08_survival=0.0, dyn_survival=0.0,
                mean_seconds=0.0, mean_hits=0.0, completed_episodes=self.completed_episodes,
            )

        def rate(pred, idx=2):
            rows = [r for r in self.outcomes if pred(r)]
            return (len(rows), float(np.mean([r[idx] for r in rows])) if rows else 0.0)

        clean_rows = [r for r in self.outcomes if r[4] == 0]
        clean_phys = float(np.mean([not r[3] for r in clean_rows])) if clean_rows else 1.0
        peak_n, peak_s = rate(lambda r: r[5])
        negx_n, negx_s = rate(lambda r: r[6])
        dir08_n, dir08_s = rate(lambda r: r[7])
        dyn_n, dyn_s = rate(lambda r: r[8])
        fails = [r for r in self.outcomes if r[3]]
        rz_frac = float(np.mean([r[9] for r in fails])) if fails else 0.0
        return dict(
            n=n,
            survival=float(np.mean([r[2] for r in self.outcomes])),
            mean_seconds=float(np.mean([r[0] for r in self.outcomes])),
            mean_hits=float(np.mean([r[1] for r in self.outcomes])),
            completed_episodes=self.completed_episodes,
            clean_n=len(clean_rows),
            clean_phys_survival=clean_phys,
            peak_n=peak_n,
            peak_survival=peak_s,
            negx_peak_n=negx_n,
            negx_peak_survival=negx_s,
            dir08_n=dir08_n,
            dir08_survival=dir08_s,
            dyn_n=dyn_n,
            dyn_survival=dyn_s,
            fail_count=len(fails),
            fail_racket_z_fraction=rz_frac,
        )
