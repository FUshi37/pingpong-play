#!/usr/bin/env python3
"""Pure-function, ROS-independent excitation waveform library for MOZ1 base ID.

All quantities use SI units (s, m, rad, m/s, rad/s, m/s^2, rad/s^2).
Every motion waveform:
  1. Begins with a zero-velocity idle window (>= 0.4 s) and ends with a zero-velocity
     settling window (>= 0.8 s).
  2. Uses SE(2)-reversible positive/negative or time-reversed pairs so the nominal
     trajectory returns to the starting pose (x=0, y=0, yaw=0).
  3. Computes its analytical SE(2) world-frame pose envelope for pre-flight geofence checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

VALID_STAGES: Tuple[str, ...] = (
    "s0_idle_noise",
    "s1_kinematic_low",
    "s2_step_ramp_med",
    "s3_step_brake_high",
    "s4_combined_turn",
    "s5_arm_concurrency",
)


@dataclass(frozen=True)
class TierLimits:
    """Per-tier kinematic safety envelope (SI units)."""

    name: str
    v_max_m_s: float
    w_max_rad_s: float
    a_max_m_s2: float
    alpha_max_rad_s2: float
    max_disp_m: float


DEFAULT_TIERS: Dict[str, TierLimits] = {
    "low": TierLimits("low", 0.20, 0.30, 0.80, 1.20, 0.30),
    "med": TierLimits("med", 0.50, 0.80, 1.50, 2.50, 0.60),
    "high": TierLimits("high", 0.90, 1.40, 2.50, 4.00, 0.90),
}


@dataclass(frozen=True)
class WaveformTrial:
    """One complete, self-contained excitation trial with analytical SE(2) bounds."""

    trial_id: str
    waveform_name: str
    stage: str
    tier: str
    dt_s: float
    repeat_index: int
    split_role: str  # "train" | "heldout" | "audit"
    primary_axis: str  # "none" | "vx" | "vy" | "wz" | "multi"
    time_s: np.ndarray
    cmd_vx_m_s: np.ndarray
    cmd_vy_m_s: np.ndarray
    cmd_wz_rad_s: np.ndarray
    phase_labels: Tuple[str, ...]
    theoretical_pose_x_m: np.ndarray
    theoretical_pose_y_m: np.ndarray
    theoretical_pose_yaw_rad: np.ndarray
    max_theoretical_radius_m: float
    max_theoretical_path_m: float
    max_theoretical_abs_yaw_rad: float
    end_pose_radius_m: float
    end_abs_yaw_rad: float
    max_abs_vx_m_s: float
    max_abs_vy_m_s: float
    max_abs_wz_rad_s: float
    arm_concurrency_enabled: bool = False

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1]) if len(self.time_s) > 0 else 0.0

    def summary_dict(self) -> Dict[str, object]:
        return {
            "trial_id": self.trial_id,
            "waveform_name": self.waveform_name,
            "stage": self.stage,
            "tier": self.tier,
            "dt_s": self.dt_s,
            "duration_s": round(self.duration_s, 3),
            "n_samples": int(len(self.time_s)),
            "repeat_index": self.repeat_index,
            "split_role": self.split_role,
            "primary_axis": self.primary_axis,
            "max_abs_vx_m_s": round(self.max_abs_vx_m_s, 4),
            "max_abs_vy_m_s": round(self.max_abs_vy_m_s, 4),
            "max_abs_wz_rad_s": round(self.max_abs_wz_rad_s, 4),
            "max_theoretical_radius_m": round(self.max_theoretical_radius_m, 4),
            "max_theoretical_path_m": round(self.max_theoretical_path_m, 4),
            "max_theoretical_abs_yaw_rad": round(self.max_theoretical_abs_yaw_rad, 4),
            "end_pose_radius_m": round(self.end_pose_radius_m, 6),
            "arm_concurrency_enabled": self.arm_concurrency_enabled,
        }


def tier_limits_from_contract(
    contract_tiers: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Dict[str, TierLimits]:
    """Parse ``safety.tier_limits`` from contract YAML into ``TierLimits`` objects."""
    if not contract_tiers:
        return dict(DEFAULT_TIERS)
    out: Dict[str, TierLimits] = {}
    for name in ("low", "med", "high"):
        src = contract_tiers.get(name)
        if src is None:
            out[name] = DEFAULT_TIERS[name]
        else:
            out[name] = TierLimits(
                name=name,
                v_max_m_s=float(src["v_max_m_s"]),
                w_max_rad_s=float(src["w_max_rad_s"]),
                a_max_m_s2=float(src["a_max_m_s2"]),
                alpha_max_rad_s2=float(src["alpha_max_rad_s2"]),
                max_disp_m=float(src["max_disp_m"]),
            )
    return out


# MOZ1 OmniWheelBase geometry (wheel radius r = 0.10 m, half-diagonal l = sqrt(2)*0.2423 m)
# and MonitorTask wheel-velocity safety bound (original_limit = 5.445 rad/s, fault = 5.990 rad/s).
OMNI_WHEEL_RADIUS_M: float = 0.10
OMNI_WHEEL_HALF_DIAG_M: float = math.sqrt(2.0) * 0.2423
SAFE_MAX_WHEEL_VEL_RAD_S: float = 4.95


def max_omni_wheel_speed_rad_s(
    vx_m_s: np.ndarray,
    vy_m_s: np.ndarray,
    wz_rad_s: np.ndarray,
    *,
    wheel_radius_m: float = OMNI_WHEEL_RADIUS_M,
    wheel_half_diag_m: float = OMNI_WHEEL_HALF_DIAG_M,
) -> float:
    """Return peak single-wheel angular speed (rad/s) for 45-deg X-omni kinematics."""
    vx = np.asarray(vx_m_s, dtype=np.float64)
    vy = np.asarray(vy_m_s, dtype=np.float64)
    wz = np.asarray(wz_rad_s, dtype=np.float64)
    c = math.sqrt(0.5) / wheel_radius_m
    k_w = wheel_half_diag_m / wheel_radius_m
    w0 = -c * vx + c * vy + k_w * wz
    w1 = -c * vx - c * vy + k_w * wz
    w2 = c * vx - c * vy + k_w * wz
    w3 = c * vx + c * vy + k_w * wz
    return float(max(
        np.max(np.abs(w0)), np.max(np.abs(w1)), np.max(np.abs(w2)), np.max(np.abs(w3))
    )) if len(vx) > 0 else 0.0


def integrate_body_twist_se2(
    dt_s: float,
    vx_m_s: np.ndarray,
    vy_m_s: np.ndarray,
    wz_rad_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate body-frame ``(vx, vy, wz)`` in world SE(2) via midpoint integration."""
    n = len(vx_m_s)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    yaw = np.zeros(n, dtype=np.float64)
    path_m = 0.0
    for i in range(n - 1):
        vx_mid = 0.5 * (float(vx_m_s[i]) + float(vx_m_s[i + 1]))
        vy_mid = 0.5 * (float(vy_m_s[i]) + float(vy_m_s[i + 1]))
        wz_mid = 0.5 * (float(wz_rad_s[i]) + float(wz_rad_s[i + 1]))
        yaw_mid = yaw[i] + 0.5 * wz_mid * dt_s
        c = math.cos(yaw_mid)
        s = math.sin(yaw_mid)
        dx = (vx_mid * c - vy_mid * s) * dt_s
        dy = (vx_mid * s + vy_mid * c) * dt_s
        x[i + 1] = x[i] + dx
        y[i + 1] = y[i] + dy
        yaw[i + 1] = yaw[i] + wz_mid * dt_s
        path_m += math.hypot(dx, dy)
    return x, y, yaw, path_m


def _finalize_trial(
    *,
    trial_id: str,
    waveform_name: str,
    stage: str,
    tier: str,
    dt_s: float,
    repeat_index: int,
    split_role: str,
    primary_axis: str,
    vx: np.ndarray,
    vy: np.ndarray,
    wz: np.ndarray,
    phase_labels: Sequence[str],
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    vx_arr = np.asarray(vx, dtype=np.float64)
    vy_arr = np.asarray(vy, dtype=np.float64)
    wz_arr = np.asarray(wz, dtype=np.float64)
    n = len(vx_arr)
    if not (len(vy_arr) == n and len(wz_arr) == n and len(phase_labels) == n):
        raise ValueError("Waveform arrays and phase_labels must share equal length.")
    time_s = np.arange(n, dtype=np.float64) * float(dt_s)
    x, y, yaw, path_m = integrate_body_twist_se2(float(dt_s), vx_arr, vy_arr, wz_arr)
    radii = np.hypot(x, y)
    return WaveformTrial(
        trial_id=trial_id,
        waveform_name=waveform_name,
        stage=stage,
        tier=tier,
        dt_s=float(dt_s),
        repeat_index=int(repeat_index),
        split_role=split_role,
        primary_axis=primary_axis,
        time_s=time_s,
        cmd_vx_m_s=vx_arr,
        cmd_vy_m_s=vy_arr,
        cmd_wz_rad_s=wz_arr,
        phase_labels=tuple(phase_labels),
        theoretical_pose_x_m=x,
        theoretical_pose_y_m=y,
        theoretical_pose_yaw_rad=yaw,
        max_theoretical_radius_m=float(np.max(radii)) if n > 0 else 0.0,
        max_theoretical_path_m=float(path_m),
        max_theoretical_abs_yaw_rad=float(np.max(np.abs(yaw))) if n > 0 else 0.0,
        end_pose_radius_m=float(radii[-1]) if n > 0 else 0.0,
        end_abs_yaw_rad=float(abs(yaw[-1])) if n > 0 else 0.0,
        max_abs_vx_m_s=float(np.max(np.abs(vx_arr))) if n > 0 else 0.0,
        max_abs_vy_m_s=float(np.max(np.abs(vy_arr))) if n > 0 else 0.0,
        max_abs_wz_rad_s=float(np.max(np.abs(wz_arr))) if n > 0 else 0.0,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def _steps(duration_s: float, dt_s: float) -> int:
    return max(1, int(round(duration_s / dt_s)))


def _make_step_axis_pos_neg(
    *,
    axis: str,
    amp: float,
    tier_cfg: TierLimits,
    dt_s: float,
    stage: str,
    trial_id: str,
    repeat_index: int = 0,
    split_role: str = "train",
    pre_idle_s: float = 0.6,
    hold_s: Optional[float] = None,
    mid_idle_s: float = 1.0,
    post_idle_s: float = 1.0,
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    """Build a symmetric +amp -> hard zero brake -> settle -> -amp -> hard zero brake trial."""
    amp = abs(float(amp))
    if axis in ("vx", "vy"):
        if amp > tier_cfg.v_max_m_s + 1e-9:
            raise ValueError(
                f"{axis} amplitude {amp:.3f} m/s exceeds tier '{tier_cfg.name}' "
                f"limit {tier_cfg.v_max_m_s:.3f} m/s."
            )
        # Auto-size hold duration so nominal single-leg displacement <= 0.85 * max_disp_m.
        max_hold_s = (0.85 * tier_cfg.max_disp_m) / max(amp, 1e-6)
        eff_hold_s = min(1.2, max_hold_s) if hold_s is None else min(float(hold_s), max_hold_s)
    elif axis == "wz":
        if amp > tier_cfg.w_max_rad_s + 1e-9:
            raise ValueError(
                f"wz amplitude {amp:.3f} rad/s exceeds tier '{tier_cfg.name}' "
                f"limit {tier_cfg.w_max_rad_s:.3f} rad/s."
            )
        max_yaw_rad = min(1.6, 0.75 * math.pi)
        eff_hold_s = min(1.3, max_yaw_rad / max(amp, 1e-6)) if hold_s is None else float(hold_s)
    else:
        raise ValueError(f"Unsupported axis {axis!r}")

    n_pre = _steps(pre_idle_s, dt_s)
    n_hold = _steps(eff_hold_s, dt_s)
    n_mid = _steps(mid_idle_s, dt_s)
    n_post = _steps(post_idle_s, dt_s)

    seq_vals = np.concatenate([
        np.zeros(n_pre, dtype=np.float64),
        np.full(n_hold, +amp, dtype=np.float64),
        np.zeros(n_mid, dtype=np.float64),
        np.full(n_hold, -amp, dtype=np.float64),
        np.zeros(n_post, dtype=np.float64),
    ])
    labels: List[str] = (
        ["pre_idle"] * n_pre
        + ["pos_step"] * n_hold
        + ["pos_brake_idle"] * n_mid
        + ["neg_step"] * n_hold
        + ["neg_brake_idle"] * n_post
    )

    zeros = np.zeros_like(seq_vals)
    vx = seq_vals if axis == "vx" else zeros
    vy = seq_vals if axis == "vy" else zeros
    wz = seq_vals if axis == "wz" else zeros
    return _finalize_trial(
        trial_id=trial_id,
        waveform_name=f"step_{axis}_pos_neg",
        stage=stage,
        tier=tier_cfg.name,
        dt_s=dt_s,
        repeat_index=repeat_index,
        split_role=split_role,
        primary_axis=axis,
        vx=vx,
        vy=vy,
        wz=wz,
        phase_labels=labels,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def idle_noise(
    *,
    duration_s: float = 30.0,
    dt_s: float = 0.02,
    stage: str = "s0_idle_noise",
    trial_id: str = "s0_idle_noise_r0",
) -> WaveformTrial:
    """Generate a zero-command baseline recording waveform."""
    n = _steps(duration_s, dt_s)
    zeros = np.zeros(n, dtype=np.float64)
    return _finalize_trial(
        trial_id=trial_id,
        waveform_name="idle_noise",
        stage=stage,
        tier="low",
        dt_s=dt_s,
        repeat_index=0,
        split_role="audit",
        primary_axis="none",
        vx=zeros,
        vy=zeros,
        wz=zeros,
        phase_labels=["idle"] * n,
    )


def step_vx_pos_neg(
    v_target_m_s: float,
    *,
    tier_cfg: TierLimits,
    dt_s: float = 0.02,
    stage: str = "s1_kinematic_low",
    trial_id: str = "step_vx_pos_neg",
    repeat_index: int = 0,
    split_role: str = "train",
    hold_s: Optional[float] = None,
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    """Pure longitudinal (+X then -X) step & emergency-brake trial."""
    return _make_step_axis_pos_neg(
        axis="vx",
        amp=v_target_m_s,
        tier_cfg=tier_cfg,
        dt_s=dt_s,
        stage=stage,
        trial_id=trial_id,
        repeat_index=repeat_index,
        split_role=split_role,
        hold_s=hold_s,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def step_vy_pos_neg(
    v_target_m_s: float,
    *,
    tier_cfg: TierLimits,
    dt_s: float = 0.02,
    stage: str = "s1_kinematic_low",
    trial_id: str = "step_vy_pos_neg",
    repeat_index: int = 0,
    split_role: str = "train",
    hold_s: Optional[float] = None,
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    """Pure lateral (+Y then -Y) step & emergency-brake trial."""
    return _make_step_axis_pos_neg(
        axis="vy",
        amp=v_target_m_s,
        tier_cfg=tier_cfg,
        dt_s=dt_s,
        stage=stage,
        trial_id=trial_id,
        repeat_index=repeat_index,
        split_role=split_role,
        hold_s=hold_s,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def step_wz_pos_neg(
    w_target_rad_s: float,
    *,
    tier_cfg: TierLimits,
    dt_s: float = 0.02,
    stage: str = "s1_kinematic_low",
    trial_id: str = "step_wz_pos_neg",
    repeat_index: int = 0,
    split_role: str = "train",
    hold_s: Optional[float] = None,
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    """Pure yaw (+Yaw then -Yaw) step & emergency-brake trial."""
    return _make_step_axis_pos_neg(
        axis="wz",
        amp=w_target_rad_s,
        tier_cfg=tier_cfg,
        dt_s=dt_s,
        stage=stage,
        trial_id=trial_id,
        repeat_index=repeat_index,
        split_role=split_role,
        hold_s=hold_s,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def ramp_accel_sweep(
    axis: str,
    accel_levels: Sequence[float],
    *,
    v_peak: float,
    tier_cfg: TierLimits,
    dt_s: float = 0.02,
    stage: str = "s2_step_ramp_med",
    trial_id: str = "ramp_accel_sweep",
    repeat_index: int = 0,
    split_role: str = "train",
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    """Multi-slope symmetric ramp sweep to pinpoint linear vs acceleration-saturated knee.

    Each slope runs a positive triangular/trapezoidal pulse followed by an exact negative
    return pulse so net displacement returns to zero after every acceleration level.
    """
    v_cap = tier_cfg.v_max_m_s if axis in ("vx", "vy") else tier_cfg.w_max_rad_s
    v_req = min(abs(float(v_peak)), v_cap)
    disp_budget = 0.85 * tier_cfg.max_disp_m if axis in ("vx", "vy") else 1.5

    chunks: List[np.ndarray] = [np.zeros(_steps(0.5, dt_s), dtype=np.float64)]
    labels: List[str] = ["pre_idle"] * len(chunks[0])

    for acc_raw in accel_levels:
        acc = abs(float(acc_raw))
        if acc <= 1e-6:
            raise ValueError(f"Ramp acceleration must be positive, got {acc_raw}")
        # Triangular pulse displacement is v_p^2 / acc. Bound v_p so it never exceeds disp_budget.
        v_max_for_disp = math.sqrt(max(1e-6, disp_budget * acc * 0.90))
        v_p = min(v_req, v_max_for_disp)
        t_ramp = v_p / acc
        n_ramp = max(2, _steps(t_ramp, dt_s))
        # Optional brief plateau if remaining displacement budget permits.
        tri_disp = v_p * (n_ramp * dt_s)
        rem_disp = max(0.0, disp_budget - tri_disp)
        t_hold = min(0.25, rem_disp / max(v_p, 1e-6))
        n_hold = _steps(t_hold, dt_s) if t_hold >= dt_s else 0

        up = np.linspace(0.0, v_p, n_ramp, endpoint=False, dtype=np.float64)
        hold = np.full(n_hold, v_p, dtype=np.float64) if n_hold > 0 else np.empty(0, dtype=np.float64)
        down = np.linspace(v_p, 0.0, n_ramp, endpoint=True, dtype=np.float64)
        pos_pulse = np.concatenate([up, hold, down])
        neg_pulse = -pos_pulse

        pause = np.zeros(_steps(0.6, dt_s), dtype=np.float64)
        chunks.extend([pos_pulse, pause, neg_pulse, pause])
        tag = f"{acc:.2f}".replace(".", "p")
        labels.extend(
            [f"ramp_pos_a{tag}"] * len(pos_pulse)
            + ["ramp_pause"] * len(pause)
            + [f"ramp_neg_a{tag}"] * len(neg_pulse)
            + ["ramp_pause"] * len(pause)
        )

    post = np.zeros(_steps(0.8, dt_s), dtype=np.float64)
    chunks.append(post)
    labels.extend(["post_idle"] * len(post))

    seq_vals = np.concatenate(chunks)
    zeros = np.zeros_like(seq_vals)
    vx = seq_vals if axis == "vx" else zeros
    vy = seq_vals if axis == "vy" else zeros
    wz = seq_vals if axis == "wz" else zeros
    return _finalize_trial(
        trial_id=trial_id,
        waveform_name=f"ramp_accel_sweep_{axis}",
        stage=stage,
        tier=tier_cfg.name,
        dt_s=dt_s,
        repeat_index=repeat_index,
        split_role=split_role,
        primary_axis=axis,
        vx=vx,
        vy=vy,
        wz=wz,
        phase_labels=labels,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def combined_diag_and_arc(
    *,
    v_diag_m_s: float,
    v_arc_m_s: float,
    w_arc_rad_s: float,
    tier_cfg: TierLimits,
    dt_s: float = 0.02,
    stage: str = "s4_combined_turn",
    trial_id: str = "combined_diag_and_arc",
    repeat_index: int = 0,
    split_role: str = "train",
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    """Combined 45-deg diagonal translation + SE(2)-reversible translate-and-turn arc.

    Using exact time-reversal ``(-vx(T-t), -vy(T-t), -wz(T-t))`` for the return leg
    guarantees exact return to (x=0, y=0, yaw=0) under SE(2) kinematics.
    """
    v_diag = min(abs(float(v_diag_m_s)), tier_cfg.v_max_m_s)
    v_arc = min(abs(float(v_arc_m_s)), tier_cfg.v_max_m_s)
    w_arc = min(abs(float(w_arc_rad_s)), tier_cfg.w_max_rad_s)

    diag_comp = v_diag / math.sqrt(2.0)
    t_diag = min(1.0, (0.80 * tier_cfg.max_disp_m) / max(v_diag, 1e-6))
    n_diag = _steps(t_diag, dt_s)

    # Arc leg: 90 deg turn or bounded by displacement budget.
    t_arc_yaw = (0.5 * math.pi) / max(w_arc, 1e-6)
    t_arc_disp = (0.80 * tier_cfg.max_disp_m) / max(v_arc, 1e-6)
    t_arc = min(1.6, t_arc_yaw, t_arc_disp)
    n_arc = _steps(t_arc, dt_s)
    n_pre = _steps(0.6, dt_s)
    n_pause = _steps(0.9, dt_s)
    n_post = _steps(1.0, dt_s)

    # Smooth cosine edges on arc leg so reversal is continuous and clean.
    t_rel = np.linspace(0.0, 1.0, n_arc, dtype=np.float64)
    env = np.minimum(1.0, np.minimum(t_rel / 0.15, (1.0 - t_rel) / 0.15))
    arc_vx_out = v_arc * 0.85 * env
    arc_vy_out = v_arc * 0.45 * env
    arc_wz_out = w_arc * env

    # Enforce 4-wheel omni velocity polytope so combined (vx, vy, wz) never trips
    # mc_core MonitorTask wheel velocity limit (5.445 rad/s).
    diag_w_peak = max_omni_wheel_speed_rad_s(
        np.array([diag_comp]), np.array([diag_comp]), np.array([0.0])
    )
    if diag_w_peak > SAFE_MAX_WHEEL_VEL_RAD_S:
        diag_comp *= SAFE_MAX_WHEEL_VEL_RAD_S / diag_w_peak
    arc_w_peak = max_omni_wheel_speed_rad_s(arc_vx_out, arc_vy_out, arc_wz_out)
    if arc_w_peak > SAFE_MAX_WHEEL_VEL_RAD_S:
        scale_arc = SAFE_MAX_WHEEL_VEL_RAD_S / arc_w_peak
        arc_vx_out *= scale_arc
        arc_vy_out *= scale_arc
        arc_wz_out *= scale_arc

    # Exact SE(2) time-reversed return leg.
    arc_vx_back = -arc_vx_out[::-1]
    arc_vy_back = -arc_vy_out[::-1]
    arc_wz_back = -arc_wz_out[::-1]

    z_pre = np.zeros(n_pre, dtype=np.float64)
    z_pause = np.zeros(n_pause, dtype=np.float64)
    z_post = np.zeros(n_post, dtype=np.float64)
    z_diag = np.zeros(n_diag, dtype=np.float64)

    vx = np.concatenate([
        z_pre,
        np.full(n_diag, +diag_comp),
        z_pause,
        np.full(n_diag, -diag_comp),
        z_pause,
        arc_vx_out,
        z_pause,
        arc_vx_back,
        z_post,
    ])
    vy = np.concatenate([
        z_pre,
        np.full(n_diag, +diag_comp),
        z_pause,
        np.full(n_diag, -diag_comp),
        z_pause,
        arc_vy_out,
        z_pause,
        arc_vy_back,
        z_post,
    ])
    wz = np.concatenate([
        z_pre,
        z_diag,
        z_pause,
        z_diag,
        z_pause,
        arc_wz_out,
        z_pause,
        arc_wz_back,
        z_post,
    ])
    labels: List[str] = (
        ["pre_idle"] * n_pre
        + ["diag_out"] * n_diag
        + ["pause"] * n_pause
        + ["diag_back"] * n_diag
        + ["pause"] * n_pause
        + ["arc_out"] * n_arc
        + ["pause"] * n_pause
        + ["arc_back"] * n_arc
        + ["post_idle"] * n_post
    )

    return _finalize_trial(
        trial_id=trial_id,
        waveform_name="combined_diag_and_arc",
        stage=stage,
        tier=tier_cfg.name,
        dt_s=dt_s,
        repeat_index=repeat_index,
        split_role=split_role,
        primary_axis="multi",
        vx=vx,
        vy=vy,
        wz=wz,
        phase_labels=labels,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def prbs_or_multisine_small(
    *,
    v_amp_m_s: float,
    w_amp_rad_s: float,
    tier_cfg: TierLimits,
    duration_s: float = 12.0,
    dt_s: float = 0.02,
    stage: str = "s2_step_ramp_med",
    trial_id: str = "prbs_or_multisine_small",
    repeat_index: int = 0,
    split_role: str = "heldout",
    arm_concurrency_enabled: bool = False,
) -> WaveformTrial:
    """Zero-net-displacement multi-sine excitation for bandwidth & held-out validation.

    Constructs a half-length multi-sine with smooth raised-cosine taper and appends its
    exact SE(2) time-reversal so the robot returns to (0, 0, 0) while keeping radial
    excursion strictly below ``0.80 * tier_cfg.max_disp_m``.
    """
    v_lim = min(abs(float(v_amp_m_s)), 0.70 * tier_cfg.v_max_m_s)
    w_lim = min(abs(float(w_amp_rad_s)), 0.70 * tier_cfg.w_max_rad_s)

    n_pre = _steps(0.5, dt_s)
    n_post = _steps(0.9, dt_s)
    active_s = max(4.0, float(duration_s) - 1.4)
    n_half = max(20, _steps(0.5 * active_s, dt_s))
    t_half = np.arange(n_half, dtype=np.float64) * dt_s
    t_span = max(float(t_half[-1]), dt_s)

    # Raised-cosine edge taper (10% at each end) ensures exact zero velocity at boundaries.
    u = t_half / t_span
    taper = np.ones_like(u)
    edge = 0.12
    left = u < edge
    right = u > (1.0 - edge)
    taper[left] = 0.5 * (1.0 - np.cos(math.pi * u[left] / edge))
    taper[right] = 0.5 * (1.0 - np.cos(math.pi * (1.0 - u[right]) / edge))
    taper[0] = 0.0
    taper[-1] = 0.0

    # Prime-like frequencies across 0.35 - 1.85 Hz with distinct phases per axis.
    raw_vx = (
        0.45 * np.sin(2.0 * math.pi * 0.41 * t_half + 0.0)
        + 0.35 * np.sin(2.0 * math.pi * 0.89 * t_half + 0.7)
        + 0.20 * np.sin(2.0 * math.pi * 1.57 * t_half + 1.9)
    ) * taper
    raw_vy = (
        0.45 * np.sin(2.0 * math.pi * 0.47 * t_half + 1.2)
        + 0.35 * np.sin(2.0 * math.pi * 1.03 * t_half + 2.4)
        + 0.20 * np.sin(2.0 * math.pi * 1.73 * t_half + 0.4)
    ) * taper
    raw_wz = (
        0.50 * np.sin(2.0 * math.pi * 0.37 * t_half + 0.5)
        + 0.30 * np.sin(2.0 * math.pi * 0.83 * t_half + 1.8)
        + 0.20 * np.sin(2.0 * math.pi * 1.49 * t_half + 2.9)
    ) * taper

    # Normalize amplitude so combined planar speed <= v_lim and |wz| <= w_lim.
    planar_peak = float(np.max(np.hypot(raw_vx, raw_vy)))
    if planar_peak > 1e-9:
        raw_vx = raw_vx * (v_lim / planar_peak)
        raw_vy = raw_vy * (v_lim / planar_peak)
    wz_peak = float(np.max(np.abs(raw_wz)))
    if wz_peak > 1e-9:
        raw_wz = raw_wz * (w_lim / wz_peak)

    # Concatenate forward half + exact SE(2) time-reversed return half.
    vx_act = np.concatenate([raw_vx, -raw_vx[::-1]])
    vy_act = np.concatenate([raw_vy, -raw_vy[::-1]])
    wz_act = np.concatenate([raw_wz, -raw_wz[::-1]])

    # Enforce strict displacement budget by uniform scaling if needed.
    x, y, _, _ = integrate_body_twist_se2(dt_s, vx_act, vy_act, wz_act)
    peak_r = float(np.max(np.hypot(x, y)))
    disp_cap = 0.75 * tier_cfg.max_disp_m
    if peak_r > disp_cap and peak_r > 1e-9:
        scale = disp_cap / peak_r
        vx_act *= scale
        vy_act *= scale

    # Enforce 4-wheel omni velocity polytope (<5.445 rad/s MonitorTask limit).
    w_peak = max_omni_wheel_speed_rad_s(vx_act, vy_act, wz_act)
    if w_peak > SAFE_MAX_WHEEL_VEL_RAD_S and w_peak > 1e-9:
        w_scale = SAFE_MAX_WHEEL_VEL_RAD_S / w_peak
        vx_act *= w_scale
        vy_act *= w_scale
        wz_act *= w_scale

    z_pre = np.zeros(n_pre, dtype=np.float64)
    z_post = np.zeros(n_post, dtype=np.float64)
    vx = np.concatenate([z_pre, vx_act, z_post])
    vy = np.concatenate([z_pre, vy_act, z_post])
    wz = np.concatenate([z_pre, wz_act, z_post])
    labels = ["pre_idle"] * n_pre + ["multisine"] * len(vx_act) + ["post_idle"] * n_post

    return _finalize_trial(
        trial_id=trial_id,
        waveform_name="prbs_or_multisine_small",
        stage=stage,
        tier=tier_cfg.name,
        dt_s=dt_s,
        repeat_index=repeat_index,
        split_role=split_role,
        primary_axis="multi",
        vx=vx,
        vy=vy,
        wz=wz,
        phase_labels=labels,
        arm_concurrency_enabled=arm_concurrency_enabled,
    )


def verify_plan_within_geofence(
    trials: Sequence[WaveformTrial],
    *,
    geofence_radius_m: float,
    geofence_yaw_rad: float,
    tier_limits: Mapping[str, TierLimits],
) -> None:
    """Strictly validate every trial against zero-boundary, tier caps, and geofence."""
    seen_ids: set[str] = set()
    for tr in trials:
        if tr.trial_id in seen_ids:
            raise ValueError(f"Duplicate trial_id detected in stage plan: {tr.trial_id!r}")
        seen_ids.add(tr.trial_id)
        if len(tr.time_s) < 2:
            raise ValueError(f"Trial {tr.trial_id} has fewer than 2 samples.")
        if not (
            np.all(np.isfinite(tr.cmd_vx_m_s))
            and np.all(np.isfinite(tr.cmd_vy_m_s))
            and np.all(np.isfinite(tr.cmd_wz_rad_s))
        ):
            raise ValueError(f"Trial {tr.trial_id} contains non-finite velocity commands.")
        # Zero initial & final velocity invariant.
        if (
            abs(float(tr.cmd_vx_m_s[0])) > 1e-12
            or abs(float(tr.cmd_vy_m_s[0])) > 1e-12
            or abs(float(tr.cmd_wz_rad_s[0])) > 1e-12
        ):
            raise ValueError(f"Trial {tr.trial_id} does not start at zero velocity.")
        if (
            abs(float(tr.cmd_vx_m_s[-1])) > 1e-12
            or abs(float(tr.cmd_vy_m_s[-1])) > 1e-12
            or abs(float(tr.cmd_wz_rad_s[-1])) > 1e-12
        ):
            raise ValueError(f"Trial {tr.trial_id} does not end at zero velocity.")

        tcfg = tier_limits[tr.tier]
        planar_max = float(np.max(np.hypot(tr.cmd_vx_m_s, tr.cmd_vy_m_s)))
        if planar_max > tcfg.v_max_m_s + 1e-6:
            raise ValueError(
                f"Trial {tr.trial_id} planar speed {planar_max:.4f} m/s exceeds "
                f"tier '{tr.tier}' limit {tcfg.v_max_m_s:.4f} m/s."
            )
        if tr.max_abs_wz_rad_s > tcfg.w_max_rad_s + 1e-6:
            raise ValueError(
                f"Trial {tr.trial_id} yaw rate {tr.max_abs_wz_rad_s:.4f} rad/s exceeds "
                f"tier '{tr.tier}' limit {tcfg.w_max_rad_s:.4f} rad/s."
            )
        if tr.max_theoretical_radius_m > min(tcfg.max_disp_m, geofence_radius_m) + 1e-6:
            raise ValueError(
                f"Trial {tr.trial_id} theoretical radius {tr.max_theoretical_radius_m:.4f} m "
                f"exceeds bound min(tier.max_disp={tcfg.max_disp_m:.3f}, "
                f"geofence={geofence_radius_m:.3f}) m."
            )
        if tr.max_theoretical_abs_yaw_rad > geofence_yaw_rad + 1e-6:
            raise ValueError(
                f"Trial {tr.trial_id} theoretical yaw {tr.max_theoretical_abs_yaw_rad:.4f} rad "
                f"exceeds geofence_yaw_rad={geofence_yaw_rad:.3f} rad."
            )


def build_stage_plan(
    stage: str,
    *,
    tier_limits: Optional[Mapping[str, TierLimits]] = None,
    geofence_radius_m: float = 1.0,
    geofence_yaw_rad: float = 3.5,
    dt_s: float = 0.02,
    trial_filter: Optional[Sequence[str]] = None,
) -> List[WaveformTrial]:
    """Build and geofence-verify the ordered list of trials for a test stage."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {VALID_STAGES}")
    tiers = dict(tier_limits or DEFAULT_TIERS)
    trials: List[WaveformTrial] = []

    def _clamped_unique(levels: Sequence[float], cap: float) -> List[float]:
        out: List[float] = []
        for lv in levels:
            val = round(min(float(lv), float(cap)), 4)
            if not any(abs(val - prev) < 1e-4 for prev in out):
                out.append(val)
        return out

    if stage == "s0_idle_noise":
        trials.append(idle_noise(duration_s=30.0, dt_s=dt_s, stage=stage))

    elif stage == "s1_kinematic_low":
        low = tiers["low"]
        v_levels = _clamped_unique((0.15, 0.20), low.v_max_m_s)
        w_levels = _clamped_unique((0.20, 0.30), low.w_max_rad_s)
        for idx, v in enumerate(v_levels):
            tag = f"{v:.2f}".replace(".", "p")
            trials.append(step_vx_pos_neg(
                v, tier_cfg=low, dt_s=dt_s, stage=stage,
                trial_id=f"s1_step_vx_{tag}_r{idx}", repeat_index=idx,
                split_role="heldout" if idx == len(v_levels) - 1 else "train",
            ))
        for idx, v in enumerate(v_levels):
            tag = f"{v:.2f}".replace(".", "p")
            trials.append(step_vy_pos_neg(
                v, tier_cfg=low, dt_s=dt_s, stage=stage,
                trial_id=f"s1_step_vy_{tag}_r{idx}", repeat_index=idx,
                split_role="heldout" if idx == len(v_levels) - 1 else "train",
            ))
        for idx, w in enumerate(w_levels):
            tag = f"{w:.2f}".replace(".", "p")
            trials.append(step_wz_pos_neg(
                w, tier_cfg=low, dt_s=dt_s, stage=stage,
                trial_id=f"s1_step_wz_{tag}_r{idx}", repeat_index=idx,
                split_role="heldout" if idx == len(w_levels) - 1 else "train",
            ))

    elif stage == "s2_step_ramp_med":
        med = tiers["med"]
        v_levels = _clamped_unique((0.25, 0.40, 0.50), med.v_max_m_s)
        w_levels = _clamped_unique((0.40, 0.60, 0.80), med.w_max_rad_s)
        for v in v_levels:
            tag = f"{v:.2f}".replace(".", "p")
            for rep in range(3):
                role = "heldout" if rep == 2 else "train"
                trials.append(step_vx_pos_neg(
                    v, tier_cfg=med, dt_s=dt_s, stage=stage,
                    trial_id=f"s2_step_vx_{tag}_r{rep}", repeat_index=rep, split_role=role,
                ))
        for v in v_levels:
            tag = f"{v:.2f}".replace(".", "p")
            for rep in range(3):
                role = "heldout" if rep == 2 else "train"
                trials.append(step_vy_pos_neg(
                    v, tier_cfg=med, dt_s=dt_s, stage=stage,
                    trial_id=f"s2_step_vy_{tag}_r{rep}", repeat_index=rep, split_role=role,
                ))
        for w in w_levels:
            tag = f"{w:.2f}".replace(".", "p")
            for rep in range(3):
                role = "heldout" if rep == 2 else "train"
                trials.append(step_wz_pos_neg(
                    w, tier_cfg=med, dt_s=dt_s, stage=stage,
                    trial_id=f"s2_step_wz_{tag}_r{rep}", repeat_index=rep, split_role=role,
                ))
        trials.append(ramp_accel_sweep(
            "vx", [0.3, 0.6, 1.0], v_peak=min(0.45, med.v_max_m_s),
            tier_cfg=med, dt_s=dt_s, stage=stage, trial_id="s2_ramp_vx_r0",
        ))
        trials.append(ramp_accel_sweep(
            "vy", [0.3, 0.6, 1.0], v_peak=min(0.45, med.v_max_m_s),
            tier_cfg=med, dt_s=dt_s, stage=stage, trial_id="s2_ramp_vy_r0",
        ))
        trials.append(ramp_accel_sweep(
            "wz", [0.5, 1.0, 1.5], v_peak=min(0.75, med.w_max_rad_s),
            tier_cfg=med, dt_s=dt_s, stage=stage, trial_id="s2_ramp_wz_r0",
        ))
        trials.append(prbs_or_multisine_small(
            v_amp_m_s=min(0.30, 0.65 * med.v_max_m_s),
            w_amp_rad_s=min(0.45, 0.65 * med.w_max_rad_s),
            tier_cfg=med, duration_s=12.0, dt_s=dt_s, stage=stage,
            trial_id="s2_multisine_heldout_r0", split_role="heldout",
        ))

    elif stage == "s3_step_brake_high":
        high = tiers["high"]
        v_levels = _clamped_unique((0.65, 0.80, 0.90), high.v_max_m_s)
        w_levels = _clamped_unique((1.00, 1.20, 1.40), high.w_max_rad_s)
        for v in v_levels:
            tag = f"{v:.2f}".replace(".", "p")
            for rep in range(2):
                role = "heldout" if rep == 1 else "train"
                trials.append(step_vx_pos_neg(
                    v, tier_cfg=high, dt_s=dt_s, stage=stage,
                    trial_id=f"s3_step_vx_{tag}_r{rep}", repeat_index=rep, split_role=role,
                ))
        for v in v_levels:
            tag = f"{v:.2f}".replace(".", "p")
            for rep in range(2):
                role = "heldout" if rep == 1 else "train"
                trials.append(step_vy_pos_neg(
                    v, tier_cfg=high, dt_s=dt_s, stage=stage,
                    trial_id=f"s3_step_vy_{tag}_r{rep}", repeat_index=rep, split_role=role,
                ))
        for w in w_levels:
            tag = f"{w:.2f}".replace(".", "p")
            for rep in range(2):
                role = "heldout" if rep == 1 else "train"
                trials.append(step_wz_pos_neg(
                    w, tier_cfg=high, dt_s=dt_s, stage=stage,
                    trial_id=f"s3_step_wz_{tag}_r{rep}", repeat_index=rep, split_role=role,
                ))

    elif stage == "s4_combined_turn":
        low = tiers["low"]
        med = tiers["med"]
        for rep in range(2):
            trials.append(combined_diag_and_arc(
                v_diag_m_s=min(0.18, low.v_max_m_s),
                v_arc_m_s=min(0.18, low.v_max_m_s),
                w_arc_rad_s=min(0.28, low.w_max_rad_s),
                tier_cfg=low, dt_s=dt_s, stage=stage,
                trial_id=f"s4_combined_low_r{rep}", repeat_index=rep,
                split_role="heldout" if rep == 1 else "train",
            ))
        for rep in range(2):
            trials.append(combined_diag_and_arc(
                v_diag_m_s=min(0.45, med.v_max_m_s),
                v_arc_m_s=min(0.42, med.v_max_m_s),
                w_arc_rad_s=min(0.75, med.w_max_rad_s),
                tier_cfg=med, dt_s=dt_s, stage=stage,
                trial_id=f"s4_combined_med_r{rep}", repeat_index=rep,
                split_role="heldout" if rep == 1 else "train",
            ))
        trials.append(prbs_or_multisine_small(
            v_amp_m_s=min(0.32, 0.65 * med.v_max_m_s),
            w_amp_rad_s=min(0.50, 0.65 * med.w_max_rad_s),
            tier_cfg=med, duration_s=14.0, dt_s=dt_s, stage=stage,
            trial_id="s4_multisine_med_r0", split_role="heldout",
        ))

    elif stage == "s5_arm_concurrency":
        low = tiers["low"]
        med = tiers["med"]
        for rep in range(2):
            trials.append(prbs_or_multisine_small(
                v_amp_m_s=min(0.14, 0.70 * low.v_max_m_s),
                w_amp_rad_s=min(0.20, 0.70 * low.w_max_rad_s),
                tier_cfg=low, duration_s=16.0, dt_s=dt_s, stage=stage,
                trial_id=f"s5_concurrent_low_r{rep}", repeat_index=rep,
                split_role="audit", arm_concurrency_enabled=True,
            ))
        for rep in range(2):
            trials.append(prbs_or_multisine_small(
                v_amp_m_s=min(0.32, 0.65 * med.v_max_m_s),
                w_amp_rad_s=min(0.50, 0.65 * med.w_max_rad_s),
                tier_cfg=med, duration_s=16.0, dt_s=dt_s, stage=stage,
                trial_id=f"s5_concurrent_med_r{rep}", repeat_index=rep,
                split_role="audit", arm_concurrency_enabled=True,
            ))

    if trial_filter:
        wanted = {tok.strip() for tok in trial_filter if tok.strip()}
        trials = [
            tr for tr in trials
            if tr.trial_id in wanted or tr.waveform_name in wanted
        ]
        if not trials:
            raise ValueError(
                f"--trials filter {sorted(wanted)} matched zero trials in stage {stage!r}."
            )

    verify_plan_within_geofence(
        trials,
        geofence_radius_m=geofence_radius_m,
        geofence_yaw_rad=geofence_yaw_rad,
        tier_limits=tiers,
    )
    return trials
