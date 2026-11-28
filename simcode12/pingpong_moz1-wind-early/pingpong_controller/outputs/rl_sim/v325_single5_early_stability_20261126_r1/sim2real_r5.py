"""Sim2Real Curriculum Wrapper for V325 Wind-Early (`s2r_v2`, 120-D Drop-In Compatible).

Solves the real-robot failure mechanisms identified on the 2026-11-27 MOZ1 probe
recordings WITHOUT any physical robot or deployment-code changes (preserves
exact 120-D obs, 120->282->282->282->7 MLP, 1.6 m/s scale, and +/-5.0 deg pose):

Uses a dynamic, zero-recompile 11-stage curriculum stored directly inside
`Sim2RealState` (`curr_warm_w`, `curr_dr_scale`, `curr_stage_id`) so the policy
never hits a 0%-survival cliff and maintains EV > 0.90 throughout:
  - Phase 1 (Stages 0..7, `dr_scale = 0.0`): Logarithmically gain-spaced PSD
    convex interpolation `(1-w)*P_cold + w*P_warm` from `w=0.25` (where u6247
    already survives ~85-95%) through `0.55 -> 0.78 -> 0.90 -> 0.96 -> 0.985 ->
    0.995 -> 1.000` (exact 200 Hz steady-state `P_warm`).
  - Phase 2 (Stages 8..10, `warm_w = 1.0`): Progressive physical & sensing DR
    `dr_scale = 0.35 -> 0.65 -> 1.00` scaling ALL real-robot gaps together:
      * Omnidirectional gravity / frame tilt incl. `+Y` (+0.08..+0.20 m/s^2)
      * Contact restitution DR (`pair_solref` damping 0.228 -> [0.21..0.36])
      * Hidden mocap latency (0 -> 0..20 ms, RealTwin 10..20 ms)
      * Static mocap XYZ bias (0 -> bz in [-16, +4] mm, RealTwin [-15, -8] mm)
      * Per-step mocap jitter (0 -> 0.6/0.6/0.8 mm) propagated through
        `observed_force_features` and `wind_features.encode`
      * RMP `kp/kd` DR (1.0 -> [0.90, 1.10])
      * Gentle calm/launch tilt & action-rate smoothness (scaled by `dr_scale`)
  - True Asymmetric Actor-Critic: Critic always receives clean unlagged/unbiased
    120-D inner observation (`state.clean_obs`).
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

from fixed_plane_wind_direction_domain import constrain_wind
from observed_force import observed_force_features
from wind_features import encode as wind_features_encode


# Exact 200 Hz (servo_steps=5, dt=0.001, Q=50.0, R=1e-4) steady-state Kalman
# covariance computed directly from `recovered_rmp_jax.joint_rmp_step`.
P_WARM_3X3 = np.array(
    [
        [2.3074470e-05, 4.9841561e-04, 5.3961379e-03],
        [4.9841159e-04, 1.5797192e-02, 2.2570074e-01],
        [5.3961705e-03, 2.2582090e-01, 4.7765627e+00],
    ],
    dtype=np.float32,
)

P_COLD_3X3 = np.diag([0.01, 0.0, 0.0]).astype(np.float32)

SIM2REAL_REWARD_FIELDS = (
    "reward/s2r_calm_tilt",
    "reward/s2r_action_smooth",
    "reward/s2r_total",
    "s2r/real_twin_lane",
    "s2r/tilt_deg",
    "s2r/gravity_ay",
    "s2r/lag_steps",
    "s2r/ep_stage_id",
)

DEFAULT_STAGES = (
    dict(id=0, name="W0_pw25",  warm_w=0.250, dr_scale=0.00, min_u=18, min_n=1536, req_surv=0.88, req_clean=0.92, req_twin=0.85, save_clear=False),
    dict(id=1, name="W1_pw55",  warm_w=0.550, dr_scale=0.00, min_u=18, min_n=1536, req_surv=0.88, req_clean=0.92, req_twin=0.85, save_clear=False),
    dict(id=2, name="W2_pw78",  warm_w=0.780, dr_scale=0.00, min_u=18, min_n=1536, req_surv=0.88, req_clean=0.92, req_twin=0.85, save_clear=False),
    dict(id=3, name="W3_pw90",  warm_w=0.900, dr_scale=0.00, min_u=18, min_n=1536, req_surv=0.88, req_clean=0.92, req_twin=0.85, save_clear=False),
    dict(id=4, name="W4_pw96",  warm_w=0.960, dr_scale=0.00, min_u=18, min_n=1536, req_surv=0.88, req_clean=0.92, req_twin=0.85, save_clear=False),
    dict(id=5, name="W5_pw985", warm_w=0.985, dr_scale=0.00, min_u=18, min_n=1536, req_surv=0.88, req_clean=0.92, req_twin=0.85, save_clear=False),
    dict(id=6, name="W6_pw995", warm_w=0.995, dr_scale=0.00, min_u=18, min_n=1536, req_surv=0.88, req_clean=0.92, req_twin=0.85, save_clear=False),
    dict(id=7, name="W7_PW100", warm_w=1.000, dr_scale=0.00, min_u=24, min_n=2048, req_surv=0.88, req_clean=0.92, req_twin=0.88, save_clear=True),
    dict(id=8, name="D1_dr35",  warm_w=1.000, dr_scale=0.35, min_u=24, min_n=2048, req_surv=0.85, req_clean=0.88, req_twin=0.85, save_clear=True),
    dict(id=9, name="D2_dr65",  warm_w=1.000, dr_scale=0.65, min_u=24, min_n=2048, req_surv=0.85, req_clean=0.88, req_twin=0.85, save_clear=True),
    dict(id=10, name="D3_FULL", warm_w=1.000, dr_scale=1.00, min_u=999999, min_n=999999, req_surv=2.0, req_clean=2.0, req_twin=2.0, save_clear=False),
)


def configure_sim2real_cfg(cfg, s2r_cfg: dict):
    """Keep native MJX config clean at Stage 0; RMP kp/kd & physical DR are scaled by dr_scale."""
    return replace(
        cfg,
        ball_spawn_xy_jitter=float(s2r_cfg.get("ball_spawn_xy_jitter_m", 0.006)),
        dr_randomize_recovered_rmp=False,
        dr_rmp_target_update_hold_probability=float(s2r_cfg.get("dr_rmp_target_update_hold_probability", 0.01)),
        wind_config=replace(
            cfg.wind_config,
            max_speed_m_s=1.6,
            mode_weights=(0.0, 0.0, 0.5, 0.5),
            wind_y_nonpositive=True,
            startup_no_wind_hits=5,
            startup_ramp_s=1.0,
        ),
    )


# ---------------------------------------------------------------------------
# Stratified Layout:
#   30% Clean: 10% Real-Robot Digital Twin + 20% Full Omnidirectional Clean DR
#   50% Constant X-cone (0..1.6 m/s): 20% Peak 1.6 m/s + 30% 4 speed bands
#   20% Dynamic (0..1.6 m/s, Y<=0): 10% Slow OU + 10% Gust
# ---------------------------------------------------------------------------

def sim2real_layout(
    n_envs: int,
    *,
    real_twin_frac: float = 0.10,
    clean_other_frac: float = 0.20,
    dyn_frac: float = 0.20,
    peak_frac: float = 0.20,
):
    """Return (mode, real_twin, sector, bins, axis, hard) arrays of length n_envs."""
    twin_n = round(real_twin_frac * n_envs)
    clean_other_n = round(clean_other_frac * n_envs)
    dyn_n = round(dyn_frac * n_envs)
    ou_n = dyn_n // 2
    gust_n = dyn_n - ou_n
    peak_n = round(peak_frac * n_envs)

    mode = np.ones(n_envs, np.int32)
    real_twin = np.zeros(n_envs, bool)
    sector = np.zeros(n_envs, np.int32)
    bins = np.zeros(n_envs, np.int32)
    axis = np.full(n_envs, -1, np.int32)
    hard = np.zeros(n_envs, bool)

    p = 0
    mode[p : p + twin_n] = 0
    real_twin[p : p + twin_n] = True
    p += twin_n

    mode[p : p + clean_other_n] = 0
    p += clean_other_n

    mode[p : p + ou_n] = 2
    p += ou_n

    mode[p : p + gust_n] = 3
    p += gust_n

    uniform = peak_n * 5 // 8
    sector[p : p + peak_n] = np.arange(peak_n) % 4
    bins[p : p + peak_n] = 4
    axis[p + uniform : p + peak_n] = np.arange(peak_n - uniform) % 2
    p += peak_n

    rem = n_envs - p
    representative = (rem * 3 // 4) // 16 * 16
    slots = np.arange(representative)
    sector[p : p + representative] = slots % 4
    bins[p : p + representative] = (slots // 4) % 4
    hard_start = p + representative
    sector[hard_start:] = np.where(np.arange(n_envs - hard_start) % 2 == 0, 0, 3)
    bins[hard_start:] = 3
    hard[hard_start:] = True
    return mode, real_twin, sector, bins, axis, hard


def sim2real_rehearsal_mask(n_envs: int, fraction: float = 0.65) -> np.ndarray:
    """Stratified 65% rehearsal mask preserving exact proportion inside every cell."""
    mode, real_twin, sector, bins, axis, hard = sim2real_layout(n_envs)
    groups = {}
    for lane in range(n_envs):
        key = (int(mode[lane]), bool(real_twin[lane]), int(sector[lane]), int(bins[lane]), int(axis[lane]), bool(hard[lane]))
        groups.setdefault(key, []).append(lane)
    mask = np.zeros(n_envs, dtype=bool)
    for key in sorted(groups):
        lanes = groups[key]
        k = max(1, min(len(lanes) - 1, int(round(len(lanes) * fraction)))) if len(lanes) > 1 else 0
        chosen_idx = np.linspace(0, len(lanes) - 1, k, dtype=int) if k > 0 else []
        for idx in chosen_idx:
            mask[lanes[idx]] = True
    return mask


def _bound_vec(vec, limit):
    norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    return vec * jnp.minimum(1.0, limit / jnp.maximum(norm, 1.0e-12))


class Sim2RealWindSamplingEnv:
    """Assigns Clean (30%: 10% RealTwin + 20% CleanDR), Constant X-cone (50%), Dynamic (20%)."""

    def __init__(self, inner, *, half_angle_deg: float = 45.0, max_speed: float = 1.6):
        self.inner = inner
        self.half_angle_deg = float(half_angle_deg)
        self.max_speed = float(max_speed)
        self.layout = sim2real_layout(inner.n_envs)
        self.lane_mode = jnp.asarray(self.layout[0], dtype=jnp.int32)
        self.real_twin_np = self.layout[1]
        self.real_twin = jnp.asarray(self.real_twin_np, dtype=jnp.bool_)
        self.sector = jnp.asarray(self.layout[2], dtype=jnp.int32)
        self.bins = jnp.asarray(self.layout[3], dtype=jnp.int32)
        self.axis = jnp.asarray(self.layout[4], dtype=jnp.int32)
        self.hard = jnp.asarray(self.layout[5], dtype=jnp.bool_)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _assign(self, state, keys, selected):
        u = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 2026112801), (6,), dtype=jnp.float32))(keys)
        norm_g = jax.vmap(lambda k: jax.random.normal(jax.random.fold_in(k, 2026112802), (3,), dtype=jnp.float32))(keys)

        width = self.half_angle_deg * math.pi / 180.0
        band_w = self.max_speed / 4.0
        low = jnp.minimum(self.bins, 3).astype(jnp.float32) * band_w
        high = low + band_w
        rad = jnp.where(self.bins == 4, self.max_speed, jnp.sqrt(low * low + u[:, 0] * (high * high - low * low)))
        offset = (self.sector % 2 + u[:, 1]) * (width / 2.0)
        angle = jnp.where(self.sector < 2, -math.pi + offset, -width + offset)
        hard_ang = jnp.where(self.sector < 2, -math.pi + u[:, 1] * width / 4.0, -u[:, 1] * width / 4.0)
        angle = jnp.where(self.hard, hard_ang, angle)
        angle = jnp.where(self.axis == 0, -math.pi, jnp.where(self.axis == 1, 0.0, angle))
        const_bg = jnp.stack([rad * jnp.cos(angle), jnp.minimum(rad * jnp.sin(angle), 0.0), jnp.zeros_like(rad)], axis=-1)

        def _disk_y_neg(u2, r):
            a = -jnp.pi * u2[:, 0]
            l = r * jnp.sqrt(u2[:, 1])
            return jnp.stack([l * jnp.cos(a), jnp.minimum(l * jnp.sin(a), 0.0), jnp.zeros_like(l)], axis=-1)

        peak_dyn = u[:, 4] < 0.25
        dir_a = -jnp.pi * u[:, 2]
        udir = jnp.stack([jnp.cos(dir_a), jnp.minimum(jnp.sin(dir_a), 0.0), jnp.zeros_like(dir_a)], axis=-1)

        ou_bg = _disk_y_neg(u[:, 2:4], 0.35 * self.max_speed)
        ou_bg = jnp.where(peak_dyn[:, None], 0.75 * self.max_speed * udir, ou_bg)
        ou_vec = _bound_vec(norm_g.at[:, 2].set(0.0) * (0.25 * self.max_speed), 0.65 * self.max_speed)
        ou_vec = jnp.where(peak_dyn[:, None], 0.25 * self.max_speed * udir, ou_vec)

        gust_bg = _disk_y_neg(u[:, 2:4], 0.15 * self.max_speed)
        gust_vec = _disk_y_neg(u[:, 4:6], 0.85 * self.max_speed)
        gust_vec = jnp.where(peak_dyn[:, None], self.max_speed * udir - gust_bg, gust_vec)
        gust_vec = constrain_wind(gust_vec, self.cfg.wind_config, xp=jnp)

        lm = self.lane_mode
        new_bg = jnp.where(
            (lm == 0)[:, None],
            0.0,
            jnp.where((lm == 1)[:, None], const_bg, jnp.where((lm == 2)[:, None], ou_bg, gust_bg)),
        )
        new_ou = jnp.where((lm == 2)[:, None], ou_vec, 0.0)
        new_gust = jnp.where((lm == 3)[:, None], gust_vec, 0.0)

        return state._replace(
            wind_mode=jnp.where(selected, lm, state.wind_mode),
            wind_background_m_s=jnp.where(selected[:, None], new_bg, state.wind_background_m_s),
            wind_ou_m_s=jnp.where(selected[:, None], new_ou, state.wind_ou_m_s),
            wind_gust_vector_m_s=jnp.where(selected[:, None], new_gust, state.wind_gust_vector_m_s),
        )

    def reset(self, keys):
        st, ob = self.inner.reset(keys)
        return self._assign(st, keys, jnp.ones(self.n_envs, dtype=jnp.bool_)), ob

    def reset_done(self, state, obs, done, keys):
        st, ob = self.inner.reset_done(state, obs, done, keys)
        return self._assign(st, keys, done), ob

    def step(self, state, action):
        return self.inner.step(state, action)


# ---------------------------------------------------------------------------
# Sim2Real Physics + Observation + Curriculum Wrapper (Zero-Recompile)
# ---------------------------------------------------------------------------

Sim2RealStateBase = namedtuple(
    "Sim2RealStateBase",
    (
        "core",
        "clean_obs",        # (N, 120) uncorrupted obs for true asymmetric critic
        "obs_ring_28_52",   # (N, 5, 24) ring buffer for 0..20ms mocap lag
        "obs_ring_72_112",  # (N, 5, 40) ring buffer for 0..20ms mocap lag
        "lag_steps",        # (N,) int32 in [0..4]
        "ball_bias_xyz",    # (N, 3) static mocap XYZ calibration bias (m)
        "noise_std_xyz",    # (N, 3) per-step mocap jitter std (m)
        "gravity_xyz",      # (N, 3) per-episode tilted gravity (m/s^2)
        "prev_action",      # (N, 7) previous action for smoothness regularization
        "noise_rng",        # (N, 2) per-lane PRNG key for step obs jitter
        "curr_warm_w",      # () float32 active curriculum P_warm convex weight in [0, 1]
        "curr_dr_scale",    # () float32 active curriculum DR scale in [0, 1]
        "curr_stage_id",    # () int32 active curriculum stage index
        "ep_stage_id",      # (N,) int32 stage index at which each lane's episode reset
    ),
)


class Sim2RealState(Sim2RealStateBase):
    __slots__ = ()

    def __getattr__(self, name):
        return getattr(self.core, name)

    def _replace(self, **values):
        local = {k: v for k, v in values.items() if k in self._fields}
        fwd = {k: v for k, v in values.items() if k not in self._fields}
        if fwd:
            local["core"] = self.core._replace(**fwd)
        return super()._replace(**local)


def _huber(x):
    return jnp.sqrt(1.0 + x * x) - 1.0


class Sim2RealPhysicsAndObsEnv:
    """Dynamic-curriculum Sim2Real wrapper preserving exact 120-D WindFeatureEnv contract."""

    MAX_LAG = 4
    NOMINAL_DAMP = 0.228086

    def __init__(
        self,
        inner,
        *,
        real_twin_mask: np.ndarray,
        s2r_cfg: dict,
        init_warm_w: float = 0.25,
        init_dr_scale: float = 0.0,
        init_stage_id: int = 0,
    ):
        self.inner = inner
        self.s2r_cfg = dict(s2r_cfg)
        self.real_twin = jnp.asarray(real_twin_mask, dtype=jnp.bool_)
        self.p_warm = jnp.asarray(P_WARM_3X3, dtype=jnp.float32)
        self.p_cold = jnp.asarray(P_COLD_3X3, dtype=jnp.float32)
        self.init_warm_w = float(init_warm_w)
        self.init_dr_scale = float(init_dr_scale)
        self.init_stage_id = int(init_stage_id)

        self.pair_id = 0
        if hasattr(inner, "ball_racket_pair_ids") and len(inner.ball_racket_pair_ids) > 0:
            self.pair_id = int(inner.ball_racket_pair_ids[0])

        self.nominal_rmp_kp = getattr(inner, "recovered_rmp_jnt_kp", None)
        self.nominal_rmp_kd = getattr(inner, "recovered_rmp_jnt_kd", None)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _sample_episode_dr(self, keys, core, warm_w, dr_scale):
        u = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 2026112810), (14,), dtype=jnp.float32))(keys)
        twin = self.real_twin
        s = jnp.clip(dr_scale, 0.0, 1.0)
        w_cur = jnp.clip(warm_w, 0.0, 1.0)
        nominal = (~twin) & (u[:, 0] < float(self.s2r_cfg.get("nominal_clean_fraction", 0.15)))

        # 1. PSD convex P_cold -> P_warm interpolation at active curriculum weight `w_cur`:
        #    RealTwin: 100% exact `w_cur`
        #    Others:   80% exact `w_cur`, 15% uniform in [0.5*w_cur, w_cur], 5% exact P_cold anchor
        other_w = jnp.where(
            u[:, 1] < 0.80,
            w_cur,
            jnp.where(u[:, 1] < 0.95, w_cur * (0.5 + 0.5 * u[:, 2]), 0.0),
        )
        lane_w = jnp.where(twin, w_cur, other_w)
        old_cov = core.rmp_state.target_covariance
        p_warm_b = jnp.broadcast_to(self.p_warm[None, None, :, :], old_cov.shape)
        p_cold_b = jnp.broadcast_to(self.p_cold[None, None, :, :], old_cov.shape)
        cov = lane_w[:, None, None, None] * p_warm_b + (1.0 - lane_w[:, None, None, None]) * p_cold_b

        # 2. Omnidirectional Gravity / Calibration Frame Tilt (scaled by `s`)
        max_tilt_acc = float(self.s2r_cfg.get("max_horizontal_gravity_m_s2", 0.22)) * s
        twin_gx = (u[:, 3] * 2.0 - 1.0) * 0.04 * s
        twin_gy = (0.08 + u[:, 4] * 0.12) * s
        twin_gz = -9.81 + ((-9.76 + u[:, 5] * 0.16) - (-9.81)) * s

        ang_full = (u[:, 3] * 2.0 - 1.0) * math.pi
        ang_pos_y = u[:, 3] * math.pi  # +Y hemisphere
        ang = jnp.where(u[:, 6] < 0.40, ang_pos_y, ang_full)
        mag = max_tilt_acc * jnp.sqrt(u[:, 4])
        gen_gx = jnp.where(nominal, 0.0, mag * jnp.cos(ang))
        gen_gy = jnp.where(nominal, 0.0, mag * jnp.sin(ang))
        gen_gz = jnp.where(nominal, -9.81, -9.81 + ((-9.85 + u[:, 5] * 0.25) - (-9.81)) * s)

        gravity_xyz = jnp.stack(
            [jnp.where(twin, twin_gx, gen_gx), jnp.where(twin, twin_gy, gen_gy), jnp.where(twin, twin_gz, gen_gz)],
            axis=-1,
        )

        # 3. Contact restitution DR via pair_solref[:, pair_id, 1] (scaled by `s`)
        twin_damp_full = 0.24 + u[:, 7] * 0.12
        gen_damp_full = jnp.where(nominal, self.NOMINAL_DAMP, 0.21 + u[:, 7] * 0.15)
        damp_full = jnp.where(twin, twin_damp_full, gen_damp_full)
        damp = self.NOMINAL_DAMP + s * (damp_full - self.NOMINAL_DAMP)

        # 4. Static mocap ball position bias (scaled by `s`)
        twin_bx = (u[:, 8] * 2.0 - 1.0) * 0.004 * s
        twin_by = (u[:, 9] * 2.0 - 1.0) * 0.004 * s
        twin_bz = (-0.015 + u[:, 10] * 0.007) * s
        gen_bx = jnp.where(nominal, 0.0, (u[:, 8] * 2.0 - 1.0) * 0.004 * s)
        gen_by = jnp.where(nominal, 0.0, (u[:, 9] * 2.0 - 1.0) * 0.004 * s)
        gen_bz = jnp.where(nominal, 0.0, (-0.016 + u[:, 10] * 0.020) * s)
        ball_bias = jnp.stack(
            [jnp.where(twin, twin_bx, gen_bx), jnp.where(twin, twin_by, gen_by), jnp.where(twin, twin_bz, gen_bz)],
            axis=-1,
        )

        # 5. Hidden mocap observation latency (0..4 steps = 0..20 ms, scaled by `s`)
        p_lag = u[:, 11]
        twin_lag_full = jnp.where(p_lag < 0.45, 2.0, jnp.where(p_lag < 0.85, 3.0, 4.0))
        gen_lag_full = jnp.where(
            nominal,
            0.0,
            jnp.where(p_lag < 0.15, 0.0, jnp.where(p_lag < 0.35, 1.0, jnp.where(p_lag < 0.65, 2.0, jnp.where(p_lag < 0.90, 3.0, 4.0)))),
        )
        lag_steps = jnp.rint(jnp.where(twin, twin_lag_full, gen_lag_full) * s).astype(jnp.int32)
        lag_steps = jnp.clip(lag_steps, 0, self.MAX_LAG)

        # 6. Per-step mocap noise std (scaled by `s`)
        base_std = jnp.asarray(self.s2r_cfg.get("mocap_noise_std_m", [0.0006, 0.0006, 0.0008]), dtype=jnp.float32) * s
        noise_std = jnp.where(nominal[:, None], 0.0, base_std[None, :])

        # 7. RMP kp/kd multipliers (1.0 +/- 0.10 * s)
        kp_half = float(self.s2r_cfg.get("rmp_kp_half_range", 0.10)) * s
        kd_half = float(self.s2r_cfg.get("rmp_kd_half_range", 0.10)) * s
        kp_mult = jnp.where(nominal, 1.0, 1.0 + (u[:, 12] * 2.0 - 1.0) * kp_half)
        kd_mult = jnp.where(nominal, 1.0, 1.0 + (u[:, 13] * 2.0 - 1.0) * kd_half)

        return cov, gravity_xyz, damp, ball_bias, lag_steps, noise_std, kp_mult, kd_mult

    def _apply_physics_dr(self, core, selected, cov, gravity_xyz, damp, kp_mult, kd_mult):
        old_rmp = core.rmp_state
        new_cov = jnp.where(selected[:, None, None, None], cov, old_rmp.target_covariance)
        new_rmp = old_rmp._replace(target_covariance=new_cov)

        old_g = core.model.opt.gravity
        new_g = jnp.where(selected[:, None], gravity_xyz, old_g)
        new_opt = core.model.opt.replace(gravity=new_g)

        new_model = core.model.replace(opt=new_opt)
        if hasattr(new_model, "pair_solref") and new_model.pair_solref.ndim == 3 and new_model.pair_solref.shape[1] > self.pair_id:
            old_d = new_model.pair_solref[:, self.pair_id, 1]
            new_d = jnp.where(selected, damp, old_d)
            new_model = new_model.replace(pair_solref=new_model.pair_solref.at[:, self.pair_id, 1].set(new_d))

        updates = dict(rmp_state=new_rmp, model=new_model)
        if self.nominal_rmp_kp is not None and hasattr(core, "rmp_jnt_kp"):
            target_kp = self.nominal_rmp_kp[None, :] * kp_mult[:, None]
            updates["rmp_jnt_kp"] = jnp.where(selected[:, None], target_kp, core.rmp_jnt_kp)
        if self.nominal_rmp_kd is not None and hasattr(core, "rmp_jnt_kd"):
            target_kd = self.nominal_rmp_kd[None, :] * kd_mult[:, None]
            updates["rmp_jnt_kd"] = jnp.where(selected[:, None], target_kd, core.rmp_jnt_kd)

        return core._replace(**updates)

    def _corrupt_obs_step(self, clean_obs, ball_bias, noise_std, rng):
        keys = jax.vmap(lambda k: jax.random.split(k, 2))(rng)
        next_rng = keys[:, 0]
        draw_k = keys[:, 1]

        eps_all = jax.vmap(lambda k: jax.random.normal(k, (16, 3), dtype=jnp.float32))(draw_k)
        eps_recent = eps_all[:, :8, :] * noise_std[:, None, :]
        eps_hist = eps_all[:, 8:, :] * (noise_std[:, None, :] * math.sqrt(2.0))
        eps_hist = eps_hist.at[:, 0, :].set(0.0)

        pos_28_52 = clean_obs[:, 28:52].reshape((-1, 8, 3))
        pos_28_52 = pos_28_52 + ball_bias[:, None, :] + eps_recent
        noisy_28_52 = pos_28_52.reshape((-1, 24))

        valid_8 = clean_obs[:, 104:112] > 0.5
        hist_72_96 = clean_obs[:, 72:96].reshape((-1, 8, 3))
        hist_72_96 = jnp.where(valid_8[:, :, None], hist_72_96 + eps_hist, hist_72_96)
        noisy_72_112 = jnp.concatenate([hist_72_96.reshape((-1, 24)), clean_obs[:, 96:112]], axis=-1)
        return noisy_28_52, noisy_72_112, next_rng

    def _assemble_obs(self, clean_obs, ring_28_52, ring_72_112, lag_steps):
        idx = jnp.arange(self.n_envs)
        slot = self.MAX_LAG - jnp.clip(lag_steps, 0, self.MAX_LAG)
        sel_28_52 = ring_28_52[idx, slot]
        sel_72_112 = ring_72_112[idx, slot]

        obs112 = clean_obs[:, :112].at[:, 28:52].set(sel_28_52).at[:, 72:112].set(sel_72_112)
        obs_force = observed_force_features(obs112, xp=jnp)
        obs118 = jnp.concatenate([obs112, clean_obs[:, 112:114], obs_force], axis=-1)
        return wind_features_encode(obs118, xp=jnp)

    def reset(self, keys):
        core, clean_obs = self.inner.reset(keys)
        all_sel = jnp.ones(self.n_envs, dtype=jnp.bool_)
        curr_warm_w = jnp.asarray(self.init_warm_w, dtype=jnp.float32)
        curr_dr_scale = jnp.asarray(self.init_dr_scale, dtype=jnp.float32)
        curr_stage_id = jnp.asarray(self.init_stage_id, dtype=jnp.int32)
        ep_stage_id = jnp.full((self.n_envs,), self.init_stage_id, dtype=jnp.int32)

        cov, gravity_xyz, damp, ball_bias, lag_steps, noise_std, kp_mult, kd_mult = self._sample_episode_dr(
            keys, core, curr_warm_w, curr_dr_scale
        )
        core = self._apply_physics_dr(core, all_sel, cov, gravity_xyz, damp, kp_mult, kd_mult)

        noise_rng = jax.vmap(lambda k: jax.random.fold_in(k, 2026112899))(keys)
        n28, n72, noise_rng = self._corrupt_obs_step(clean_obs, ball_bias, noise_std, noise_rng)
        ring28 = jnp.repeat(n28[:, None, :], self.MAX_LAG + 1, axis=1)
        ring72 = jnp.repeat(n72[:, None, :], self.MAX_LAG + 1, axis=1)
        obs = self._assemble_obs(clean_obs, ring28, ring72, lag_steps)
        prev_act = jnp.zeros((self.n_envs, self.act_dim), dtype=jnp.float32)
        st = Sim2RealState(
            core, clean_obs, ring28, ring72, lag_steps, ball_bias, noise_std, gravity_xyz,
            prev_act, noise_rng, curr_warm_w, curr_dr_scale, curr_stage_id, ep_stage_id,
        )
        return st, obs

    def reset_done(self, state, obs, done, keys):
        core, clean_obs = self.inner.reset_done(state.core, state.clean_obs, done, keys)
        cov, gravity_xyz, damp, ball_bias, lag_steps, noise_std, kp_mult, kd_mult = self._sample_episode_dr(
            keys, core, state.curr_warm_w, state.curr_dr_scale
        )
        core = self._apply_physics_dr(core, done, cov, gravity_xyz, damp, kp_mult, kd_mult)

        new_bias = jnp.where(done[:, None], ball_bias, state.ball_bias_xyz)
        new_std = jnp.where(done[:, None], noise_std, state.noise_std_xyz)
        new_lag = jnp.where(done, lag_steps, state.lag_steps)
        new_grav = jnp.where(done[:, None], gravity_xyz, state.gravity_xyz)
        new_prev = jnp.where(done[:, None], 0.0, state.prev_action)
        new_ep_stage = jnp.where(done, state.curr_stage_id, state.ep_stage_id)

        fresh_rng = jax.vmap(lambda k: jax.random.fold_in(k, 2026112899))(keys)
        n28, n72, stepped_fresh_rng = self._corrupt_obs_step(clean_obs, new_bias, new_std, fresh_rng)
        new_rng = jnp.where(done[:, None], stepped_fresh_rng, state.noise_rng)
        fresh_r28 = jnp.repeat(n28[:, None, :], self.MAX_LAG + 1, axis=1)
        fresh_r72 = jnp.repeat(n72[:, None, :], self.MAX_LAG + 1, axis=1)
        new_r28 = jnp.where(done[:, None, None], fresh_r28, state.obs_ring_28_52)
        new_r72 = jnp.where(done[:, None, None], fresh_r72, state.obs_ring_72_112)

        corrupted_reset_obs = self._assemble_obs(clean_obs, new_r28, new_r72, new_lag)
        out_obs = jnp.where(done[:, None], corrupted_reset_obs, obs)
        st = Sim2RealState(
            core, clean_obs, new_r28, new_r72, new_lag, new_bias, new_std, new_grav,
            new_prev, new_rng, state.curr_warm_w, state.curr_dr_scale, state.curr_stage_id, new_ep_stage,
        )
        return st, out_obs

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state.core, state.clean_obs)

    def _obs(self, state, native_obs):
        return self.inner._obs(state.core, native_obs)

    def step(self, state, action):
        after_core, clean_obs, reward, done, metrics = self.inner.step(state.core, action)

        n28, n72, next_rng = self._corrupt_obs_step(clean_obs, state.ball_bias_xyz, state.noise_std_xyz, state.noise_rng)
        ring28 = jnp.concatenate([state.obs_ring_28_52[:, 1:, :], n28[:, None, :]], axis=1)
        ring72 = jnp.concatenate([state.obs_ring_72_112[:, 1:, :], n72[:, None, :]], axis=1)
        obs = self._assemble_obs(clean_obs, ring28, ring72, state.lag_steps)

        # Gentle calm/launch tilt & action smoothness scaled by `curr_dr_scale`
        # (strictly 0.0 during Phase 1 P_warm curriculum so launch adaptation has zero distortion)
        s = jnp.clip(state.curr_dr_scale, 0.0, 1.0)
        live = metrics["terminated"] < 0.5
        wind_spd = metrics["wind/speed_m_s"]
        hits = after_core.wind_startup_hit_count.astype(jnp.float32)
        calm_w = jnp.where(hits < 5.0, 1.0, jnp.clip(1.0 - wind_spd / 0.8, 0.0, 1.0))
        tilt_deg = jnp.hypot(metrics["diagnostic/racket_roll_deg"], metrics["diagnostic/racket_pitch_deg"])
        tilt_dead = float(self.s2r_cfg.get("calm_tilt_deadband_deg", 1.5))
        tilt_scale = float(self.s2r_cfg.get("calm_tilt_scale_deg", 1.0))
        tilt_rate = float(self.s2r_cfg.get("calm_tilt_rate_per_s", 6.0)) * s
        tilt_exc = jnp.maximum(tilt_deg - tilt_dead, 0.0) / tilt_scale
        r_tilt = jnp.where(live, -tilt_rate * self.dt * calm_w * jnp.minimum(_huber(tilt_exc), 6.0), 0.0)

        has_prev = after_core.step_count > 1
        act_diff_rms = jnp.sqrt(jnp.mean((action - state.prev_action) ** 2, axis=-1) + 1e-12)
        act_dead = float(self.s2r_cfg.get("action_smooth_deadband", 0.06))
        act_scale = float(self.s2r_cfg.get("action_smooth_scale", 0.08))
        act_rate = float(self.s2r_cfg.get("action_smooth_rate_per_s", 1.0)) * s
        act_exc = jnp.maximum(act_diff_rms - act_dead, 0.0) / act_scale
        r_smooth = jnp.where(live & has_prev, -act_rate * self.dt * jnp.minimum(_huber(act_exc), 6.0), 0.0)

        r_s2r = r_tilt + r_smooth
        reward = reward + r_s2r
        metrics = {
            **metrics,
            "reward/s2r_calm_tilt": r_tilt,
            "reward/s2r_action_smooth": r_smooth,
            "reward/s2r_total": r_s2r,
            "reward/total": reward,
            "s2r/real_twin_lane": self.real_twin.astype(jnp.float32),
            "s2r/tilt_deg": tilt_deg,
            "s2r/gravity_ay": state.gravity_xyz[:, 1],
            "s2r/lag_steps": state.lag_steps.astype(jnp.float32),
            "s2r/ep_stage_id": state.ep_stage_id.astype(jnp.float32),
        }
        next_st = Sim2RealState(
            after_core, clean_obs, ring28, ring72, state.lag_steps, state.ball_bias_xyz,
            state.noise_std_xyz, state.gravity_xyz, action, next_rng,
            state.curr_warm_w, state.curr_dr_scale, state.curr_stage_id, state.ep_stage_id,
        )
        return next_st, obs, reward, done, metrics


# ---------------------------------------------------------------------------
# Rehearsal Wrapper & Stage-Pure Progress Tracker
# ---------------------------------------------------------------------------

class Sim2RealRehearsalEnv:
    def __init__(self, inner, *, fraction: float = 0.65, seconds: float = 25.0,
                 stagger_min_steps: int = 512, stagger_seed: int = 2026112806):
        from stratified_rehearsal import _RehearsalStateWrapper, initial_staggered_cutoffs

        self._wrap_cls = _RehearsalStateWrapper
        self.inner = inner
        self.seconds = float(seconds)
        self.cutoff_steps = int(round(seconds / inner.dt))
        self.role_np = sim2real_rehearsal_mask(inner.n_envs, fraction=fraction)
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


class Sim2RealStartupProgress:
    """Stage-pure 25s survival tracker: only counts episodes started at `active_stage_id`."""

    def __init__(self, *, seconds: float = 25.0, capacity: int = 4096, active_stage_id: int = 0):
        self.seconds = float(seconds)
        self.active_stage_id = int(active_stage_id)
        self.outcomes = deque(maxlen=capacity)
        self.completed_episodes = 0

    def set_stage(self, stage_id: int):
        self.active_stage_id = int(stage_id)
        self.outcomes.clear()

    def add(self, metrics: dict, done: np.ndarray):
        done_b = np.asarray(done, bool)
        if not np.any(done_b):
            return
        mode = np.rint(np.asarray(metrics["wind/mode"])).astype(np.int32)
        bg_speed = np.asarray(metrics["polish/background_speed_m_s"], dtype=np.float32)
        angle = np.asarray(metrics["single5/wind_angle_rad"], dtype=np.float32)
        twin = np.asarray(metrics["s2r/real_twin_lane"]) > 0.5
        ep_stg = np.rint(np.asarray(metrics["s2r/ep_stage_id"])).astype(np.int32)
        for t, lane in zip(*np.nonzero(done_b)):
            if int(ep_stg[t, lane]) != self.active_stage_id:
                continue
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
            is_twin = bool(twin[t, lane])
            is_peak = (m == 1) and (spd >= 1.56)
            is_negx = is_peak and (ang < -math.pi / 2.0)
            is_dir08 = is_peak and (abs(abs(ang) - math.pi) < 1e-3)
            is_dyn = m in (2, 3)
            self.outcomes.append((duration, hits, success, terminated, m, is_twin, is_peak, is_negx, is_dir08, is_dyn))
            self.completed_episodes += 1

    def summary(self) -> dict:
        n = len(self.outcomes)
        if n == 0:
            return dict(
                n=0, survival=0.0, real_twin_survival=0.0, clean_phys_survival=0.0,
                peak_survival=0.0, negx_peak_survival=0.0, dir08_survival=0.0, dyn_survival=0.0,
                mean_seconds=0.0, mean_hits=0.0, completed_episodes=self.completed_episodes,
                real_twin_n=0, clean_n=0, peak_n=0, negx_peak_n=0, dir08_n=0, dyn_n=0, fail_count=0,
            )

        def rate(pred, idx=2):
            rows = [r for r in self.outcomes if pred(r)]
            return (len(rows), float(np.mean([r[idx] for r in rows])) if rows else 0.0)

        twin_n, twin_s = rate(lambda r: r[5])
        clean_rows = [r for r in self.outcomes if r[4] == 0]
        clean_phys = float(np.mean([not r[3] for r in clean_rows])) if clean_rows else 1.0
        peak_n, peak_s = rate(lambda r: r[6])
        negx_n, negx_s = rate(lambda r: r[7])
        dir08_n, dir08_s = rate(lambda r: r[8])
        dyn_n, dyn_s = rate(lambda r: r[9])
        fails = [r for r in self.outcomes if r[3]]
        return dict(
            n=n,
            survival=float(np.mean([r[2] for r in self.outcomes])),
            mean_seconds=float(np.mean([r[0] for r in self.outcomes])),
            mean_hits=float(np.mean([r[1] for r in self.outcomes])),
            completed_episodes=self.completed_episodes,
            real_twin_n=twin_n,
            real_twin_survival=twin_s,
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
        )
