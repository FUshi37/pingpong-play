"""Standalone Pure-NumPy Inference & Feature Encoder for V325 High-Wind r6 (`single5_highwind120_4p5ms_v1`).

Zero JAX dependency. Matches `high_wind_r6.observed_force_features_r6`, `high_wind_r6.encode_r6`,
and `stack.ppo.policy_mean` to < 1e-6 across all valid 120-D observations (0..4.5 m/s unclipped wind).

Key contract differences vs legacy 1.6 m/s `single5_windfeature120_high_v2` (u6247 / u7447):
  1. `obs_wind_max_m_s = 4.5` (instead of 1.6):
     - `obs[114:116] = estimated_wind_xy / 4.5` (norm clipped at 4.5 m/s instead of 1.6 m/s)
     - `obs[118:120] = obs[114:116] * clip((|est_wind_m_s| - 1.25) / 0.35, 0, 1)` where
       `|est_wind_m_s| = 4.5 * ||obs[114:116]||` (gated to 0 when `obs[112] <= 0`)
  2. `relativeRollPitchLimitDeg = [15.0, 15.0]` (`fixedPose = True`, vs `[5.0, 5.0]` for u7447).
  3. Actor MLP architecture is identical: `120 -> 282 (tanh) -> 282 (tanh) -> 282 (tanh) -> 7 (linear)`.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
import numpy as np

DEFAULT_OBS_WIND_MAX_M_S = 4.5
DEADBAND_M_S = 1.25
ACTIVATION_WIDTH_M_S = 0.35
BALL_MASS_KG = 0.004


def select_consistent_wind_np(winds: np.ndarray, valid: np.ndarray, ages_s: np.ndarray):
    """Exact NumPy mirror of `observed_force.select_consistent_wind`."""
    count = np.sum(valid, axis=-1)
    ordered = np.sort(np.where(valid[..., None], winds, np.inf), axis=-2)
    lo = np.maximum((count - 1) // 2, 0)
    hi = np.maximum(count // 2, 0)
    median = 0.5 * (
        np.take_along_axis(ordered, lo[..., None, None], axis=-2)[..., 0, :]
        + np.take_along_axis(ordered, hi[..., None, None], axis=-2)[..., 0, :]
    )
    distance = np.sum((winds - median[..., None, :]) ** 2, axis=-1)
    robust_index = np.argmin(np.where(valid, distance, np.inf), axis=-1)
    indices = np.broadcast_to(np.arange(6), valid.shape)
    first = np.min(np.where(valid, indices, 6), axis=-1)
    second = np.min(np.where(valid & (indices > first[..., None]), indices, 6), axis=-1)
    first = np.minimum(first, 5)
    second = np.minimum(second, 5)
    newest = np.take_along_axis(winds, first[..., None, None], axis=-2)[..., 0, :]
    previous = np.take_along_axis(winds, second[..., None, None], axis=-2)[..., 0, :]
    speed0 = np.sqrt(np.sum(newest ** 2, axis=-1))
    speed1 = np.sqrt(np.sum(previous ** 2, axis=-1))
    aligned = np.sum(newest * previous, axis=-1) > 0.9 * speed0 * speed1
    fast = (count >= 2) & (speed0 > 0.02) & (speed1 > 0.02) & aligned
    selected = np.where(fast, first, robust_index)
    chosen = np.take_along_axis(winds, selected[..., None, None], axis=-2)[..., 0, :]
    age = np.take_along_axis(ages_s, selected[..., None], axis=-1)[..., 0]
    enough = count >= 2
    age = np.where(
        np.sum(chosen ** 2, axis=-1) < 0.05 ** 2,
        np.take_along_axis(ages_s, first[..., None], axis=-1)[..., 0],
        age,
    )
    chosen = np.where((enough & (np.sum(chosen ** 2, axis=-1) >= 0.05 ** 2))[..., None], chosen, 0.0)
    return chosen, np.where(enough, age, 0.6), enough


def observed_force_features_r6_np(
    observations: np.ndarray,
    *,
    max_wind_m_s: float = DEFAULT_OBS_WIND_MAX_M_S,
    ball_mass_kg: float = BALL_MASS_KG,
) -> np.ndarray:
    """Compute causal 4-D observed wind estimate `(..., 4)`: `[wx/wmax, wy/wmax, age/0.6, valid]`."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.shape[-1] < 112:
        raise ValueError(f"Expected >= 112-D observation prefix, got {obs.shape[-1]}")
    wmax = float(max_wind_m_s)
    positions = obs[..., 72:96].reshape(obs.shape[:-1] + (8, 3))
    ages = np.float32(0.6) * obs[..., 96:104]
    finite = np.all(np.isfinite(positions) & (np.abs(positions) < 5.0), axis=-1)
    valid = (obs[..., 104:112] > 0.5) & finite & np.isfinite(ages)
    positions = np.where(finite[..., None], positions, np.float32(0.0))
    ages = np.where(np.isfinite(ages), ages, np.float32(0.0))

    candidates, candidate_ages, candidate_valid = [], [], []
    coeff_m_inv = np.float32(0.5 * 1.2 * 0.47 * math.pi * (0.02 ** 2) / float(ball_mass_kg))
    for index in range(6):
        dt01 = ages[..., index + 1] - ages[..., index]
        dt12 = ages[..., index + 2] - ages[..., index + 1]
        safe01 = np.maximum(dt01, np.float32(1.0e-6))
        safe12 = np.maximum(dt12, np.float32(1.0e-6))
        v01 = (positions[..., index, :] - positions[..., index + 1, :]) / safe01[..., None]
        v12 = (positions[..., index + 1, :] - positions[..., index + 2, :]) / safe12[..., None]
        acceleration = np.float32(2.0) * (v01 - v12) / (safe01 + safe12)[..., None]
        velocity = v01 - np.float32(0.5) * safe12[..., None] * acceleration
        usable = (
            np.all(valid[..., index : index + 3], axis=-1)
            & (dt01 > 0.02)
            & (dt12 > 0.02)
            & (acceleration[..., 2] > -11.5)
            & (acceleration[..., 2] < -8.5)
        )
        speed = np.sqrt(np.sum(velocity ** 2, axis=-1))
        b = acceleration[..., :2] / coeff_m_inv - speed[..., None] * velocity[..., :2]
        b_sq = np.sum(b ** 2, axis=-1)
        vz_sq = velocity[..., 2] ** 2
        rel_speed = np.sqrt(np.maximum(np.float32(0.5) * (vz_sq + np.sqrt(vz_sq ** 2 + np.float32(4.0) * b_sq)), np.float32(1.0e-12)))
        wind = velocity[..., :2] + b / rel_speed[..., None]
        wind = np.stack([wind[..., 0], np.minimum(wind[..., 1], np.float32(0.0))], axis=-1)
        norm = np.sqrt(np.sum(wind ** 2, axis=-1))
        wind = wind * np.minimum(np.float32(1.0), np.float32(wmax) / np.maximum(norm, np.float32(1.0e-6)))[..., None]
        candidates.append(wind.astype(np.float32))
        candidate_ages.append(np.clip(np.float32(0.5) * (ages[..., index] + ages[..., index + 2]), np.float32(0.0), np.float32(0.6)))
        candidate_valid.append(usable)

    winds = np.stack(candidates, axis=-2)
    usable_all = np.stack(candidate_valid, axis=-1)
    estimate, estimate_age, chosen = select_consistent_wind_np(winds, usable_all, np.stack(candidate_ages, axis=-1))
    mature = np.any(valid & (ages >= 0.25), axis=-1)
    chosen = chosen & mature
    estimate = np.where(chosen[..., None], estimate / np.float32(wmax), np.float32(0.0))
    estimate_age = np.where(chosen, estimate_age / np.float32(0.6), np.float32(1.0))
    return np.concatenate(
        [estimate.astype(np.float32), estimate_age[..., None].astype(np.float32), chosen.astype(np.float32)[..., None]],
        axis=-1,
    )


def encode_r6_np(
    obs118: np.ndarray,
    *,
    max_wind_m_s: float = DEFAULT_OBS_WIND_MAX_M_S,
    deadband_m_s: float = DEADBAND_M_S,
    activation_width_m_s: float = ACTIVATION_WIDTH_M_S,
) -> np.ndarray:
    """Append 2-D deadbanded strong-wind suffix `(..., 118) -> (..., 120)`."""
    obs = np.asarray(obs118, dtype=np.float32)
    if obs.shape[-1] != 118:
        raise ValueError(f"Expected 118-D actor observation, got {obs.shape[-1]}")
    estimated = obs[..., 114:116]
    speed_m_s = np.float32(max_wind_m_s) * np.sqrt(np.sum(estimated * estimated, axis=-1))
    gain = np.clip((speed_m_s - np.float32(deadband_m_s)) / np.float32(activation_width_m_s), np.float32(0.0), np.float32(1.0))
    gain = np.where(obs[..., 112] > 0.0, gain, np.float32(0.0))
    return np.concatenate([obs, (estimated * gain[..., None]).astype(np.float32)], axis=-1)


def build_actor_obs_120_r6_np(
    obs112: np.ndarray,
    acquisition_features_2d: np.ndarray,
    *,
    max_wind_m_s: float = DEFAULT_OBS_WIND_MAX_M_S,
) -> np.ndarray:
    """Build full 120-D actor observation from `[obs112, acquisition_2d]` with `max_wind_m_s=4.5`."""
    o112 = np.asarray(obs112, dtype=np.float32)
    acq = np.asarray(acquisition_features_2d, dtype=np.float32)
    obs_force = observed_force_features_r6_np(o112, max_wind_m_s=max_wind_m_s)
    obs118 = np.concatenate([o112, acq, obs_force], axis=-1)
    return encode_r6_np(obs118, max_wind_m_s=max_wind_m_s)


class _SafeDeployUnpickler(pickle.Unpickler):
    """Allow loading model.pkl on pure-NumPy robot hosts without JAX or training optimizer modules."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Return a lightweight placeholder class for optimizer-only state (not used at inference)
            return type(name, (), {"__setstate__": lambda self, s: None, "__new__": lambda cls, *a, **kw: object.__new__(cls)})


class Single5HighWindR6Actor:
    """Pure-NumPy deterministic actor for `single5_highwind120_4p5ms_v1` checkpoints."""

    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        with self.checkpoint_path.open("rb") as f:
            payload = _SafeDeployUnpickler(f).load()
        self.obs_wind_max_m_s = float(payload.get("obs_wind_max_m_s", DEFAULT_OBS_WIND_MAX_M_S))
        self.pose_limit_deg = float(payload.get("pose_limit_deg", 15.0))
        self.runtime_schema = str(payload.get("requires_actor_runtime", "single5_highwind120_4p5ms_v1"))
        pi = payload["params"]["pi"]
        self.w1 = np.asarray(pi["l1"]["w"], dtype=np.float32)
        self.b1 = np.asarray(pi["l1"]["b"], dtype=np.float32)
        self.w2 = np.asarray(pi["l2"]["w"], dtype=np.float32)
        self.b2 = np.asarray(pi["l2"]["b"], dtype=np.float32)
        self.w3 = np.asarray(pi["l3"]["w"], dtype=np.float32)
        self.b3 = np.asarray(pi["l3"]["b"], dtype=np.float32)
        self.wo = np.asarray(pi["out"]["w"], dtype=np.float32)
        self.bo = np.asarray(pi["out"]["b"], dtype=np.float32)
        assert self.w1.shape == (120, 282) and self.wo.shape == (282, 7)

    def policy_mean(self, obs120: np.ndarray) -> np.ndarray:
        x = np.asarray(obs120, dtype=np.float32)
        h = np.tanh(x @ self.w1 + self.b1)
        h = np.tanh(h @ self.w2 + self.b2)
        h = np.tanh(h @ self.w3 + self.b3)
        return (h @ self.wo + self.bo).astype(np.float32)

    def infer_from_prefix(self, obs112: np.ndarray, acquisition_features_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs120 = build_actor_obs_120_r6_np(obs112, acquisition_features_2d, max_wind_m_s=self.obs_wind_max_m_s)
        return self.policy_mean(obs120), obs120
