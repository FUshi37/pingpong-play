"""Reference helpers for delay-conditioned acceleration command control.

The MJX environment uses JAX equivalents of these formulas inside jit-compiled
steps.  This module keeps the same math in small NumPy functions so deployment
and unit tests can exercise the control law without constructing MJX state.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


DEFAULT_DELAY_BIN_EDGES_MS = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0)


def delay_steps_from_tau(tau_s: float, dt: float) -> int:
    """Return rounded control-delay steps with non-negative clamping."""
    if not np.isfinite(tau_s) or not np.isfinite(dt) or dt <= 0.0:
        return 0
    return max(0, int(round(float(tau_s) / float(dt))))


def command_buffer_length(delay_max_ms: float, dt: float, extra_steps: int = 0) -> int:
    """Length needed to retrieve a delayed command plus latest command."""
    max_steps = delay_steps_from_tau(max(0.0, float(delay_max_ms)) * 1e-3, dt)
    return max(1, max_steps + max(0, int(extra_steps)) + 1)


def delay_bin_id(tau_ms: float, edges_ms: Iterable[float] = DEFAULT_DELAY_BIN_EDGES_MS) -> int:
    """Return the interval index containing tau_ms for monotonically sorted edges."""
    edges = np.asarray(tuple(edges_ms), dtype=np.float32)
    if edges.size < 2:
        return 0
    tau = float(np.clip(tau_ms, edges[0], edges[-1]))
    return int(np.clip(np.searchsorted(edges[1:], tau, side="right"), 0, edges.size - 2))


def push_command_buffer(
    command_buffer: np.ndarray,
    q_ref_latest: np.ndarray,
    delay_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append latest q_ref and return the delay-selected active command."""
    buf = np.asarray(command_buffer, dtype=np.float32)
    q = np.asarray(q_ref_latest, dtype=np.float32)
    if buf.ndim != 2:
        raise ValueError("command_buffer must have shape (T, act_dim)")
    if q.shape != (buf.shape[1],):
        raise ValueError(f"q_ref_latest must have shape ({buf.shape[1]},)")
    new_buf = np.concatenate([buf[1:], q[None, :]], axis=0)
    steps = int(np.clip(delay_steps, 0, new_buf.shape[0] - 1))
    active_idx = new_buf.shape[0] - 1 - steps
    return new_buf, new_buf[active_idx].copy()


def smooth_action(
    a_raw: np.ndarray,
    a_prev: np.ndarray,
    *,
    dt: float,
    action_acc_limit: float = 1.0,
    action_filter_tau_ms: float = 0.0,
    action_jerk_limit: float = 0.0,
    e_active: np.ndarray | None = None,
    enable_anti_windup: bool = False,
    anti_windup_error_threshold: float = 0.5,
    anti_windup_min_scale: float = 0.2,
) -> tuple[np.ndarray, float]:
    """Clip, low-pass filter, jerk-limit, and optionally anti-windup scale action."""
    raw = np.asarray(a_raw, dtype=np.float32)
    prev = np.asarray(a_prev, dtype=np.float32)
    if raw.shape != prev.shape:
        raise ValueError("a_raw and a_prev must have the same shape")
    dt_safe = max(float(dt), 1e-9)
    acc_limit = float(action_acc_limit)
    if acc_limit > 0.0 and np.isfinite(acc_limit):
        a_clip = np.clip(raw, -acc_limit, acc_limit)
    else:
        a_clip = raw.copy()

    tau_s = max(0.0, float(action_filter_tau_ms)) * 1e-3
    alpha = 1.0 if tau_s <= 1e-9 else dt_safe / (tau_s + dt_safe)
    a_lpf = alpha * a_clip + (1.0 - alpha) * prev

    jerk_limit = float(action_jerk_limit)
    if jerk_limit > 0.0 and np.isfinite(jerk_limit):
        max_delta = jerk_limit * dt_safe
        a_final = prev + np.clip(a_lpf - prev, -max_delta, max_delta)
    else:
        a_final = a_lpf

    scale = 1.0
    if enable_anti_windup and e_active is not None:
        threshold = max(float(anti_windup_error_threshold), 1e-9)
        min_scale = float(np.clip(anti_windup_min_scale, 0.0, 1.0))
        err = float(np.linalg.norm(np.asarray(e_active, dtype=np.float32)))
        if np.isfinite(err):
            scale = float(np.clip(1.0 - err / threshold, min_scale, 1.0))
            a_final = a_final * np.float32(scale)
    return a_final.astype(np.float32), scale


def lead_compensate_q_ref(
    q_cmd: np.ndarray,
    qdot_cmd: np.ndarray,
    qdd_cmd: np.ndarray,
    *,
    dt: float,
    delay_steps: int,
    actuator_tau_s: float = 0.0,
    beta: float = 0.0,
    delay_scale: float = 1.0,
    tau_scale: float = 1.0,
    max_delta_rad: float = 0.0,
    q_low: np.ndarray | None = None,
    q_high: np.ndarray | None = None,
) -> np.ndarray:
    """Return a conservative phase-lead joint target for delayed actuators.

    The policy still integrates a nominal q_ref.  This helper optionally sends a
    small look-ahead version into the command delay/filter path:

        q_ref = q_cmd + beta * (T * qdot_cmd + 0.5 * T^2 * qdd_cmd)

    ``max_delta_rad`` should usually be finite and modest when deploying on a
    real robot; it bounds the compensation independently per joint.
    """
    q = np.asarray(q_cmd, dtype=np.float32)
    qdot = np.asarray(qdot_cmd, dtype=np.float32)
    qdd = np.asarray(qdd_cmd, dtype=np.float32)
    if q.shape != qdot.shape or q.shape != qdd.shape:
        raise ValueError("q_cmd, qdot_cmd, and qdd_cmd must have the same shape")

    beta_f = float(beta)
    if not np.isfinite(beta_f) or beta_f == 0.0:
        out = q.copy()
    else:
        dt_safe = max(float(dt), 1e-9)
        delay_s = max(0.0, float(delay_steps)) * dt_safe
        tau_s = max(0.0, float(actuator_tau_s))
        lead_time = max(0.0, float(delay_scale)) * delay_s + max(0.0, float(tau_scale)) * tau_s
        delta = beta_f * (lead_time * qdot + 0.5 * lead_time * lead_time * qdd)
        limit = float(max_delta_rad)
        if np.isfinite(limit) and limit > 0.0:
            delta = np.clip(delta, -limit, limit)
        out = q + delta.astype(np.float32)

    if q_low is not None or q_high is not None:
        lo = -np.inf if q_low is None else np.asarray(q_low, dtype=np.float32)
        hi = np.inf if q_high is None else np.asarray(q_high, dtype=np.float32)
        out = np.clip(out, lo, hi)
    return out.astype(np.float32)


def predict_filtered_command_before_delay(
    applied_q: np.ndarray,
    command_buffer: np.ndarray,
    *,
    dt: float,
    delay_steps: int,
    actuator_tau_s: float,
    actuator_gain: float = 1.0,
    warm_q: np.ndarray | None = None,
    append_placeholder: np.ndarray | None = None,
) -> np.ndarray:
    """Predict filtered joint output before a newly appended command is active."""
    y = np.asarray(applied_q, dtype=np.float32).copy()
    buf = np.asarray(command_buffer, dtype=np.float32)
    if buf.ndim != 2:
        raise ValueError("command_buffer must have shape (T, act_dim)")
    if y.shape != (buf.shape[1],):
        raise ValueError(f"applied_q must have shape ({buf.shape[1]},)")
    if append_placeholder is None:
        append = buf[-1]
    else:
        append = np.asarray(append_placeholder, dtype=np.float32)
        if append.shape != y.shape:
            raise ValueError("append_placeholder must match applied_q shape")
    pred_buf = np.concatenate([buf[1:], append[None, :]], axis=0)
    steps = int(np.clip(delay_steps, 0, pred_buf.shape[0] - 1))
    if steps <= 0:
        return y

    dt_safe = max(float(dt), 1e-9)
    tau_s = max(0.0, float(actuator_tau_s))
    alpha = 1.0 if tau_s <= 1e-9 else dt_safe / (tau_s + dt_safe)
    base = np.zeros_like(y) if warm_q is None else np.asarray(warm_q, dtype=np.float32)
    gain = float(actuator_gain)
    for s in range(steps):
        idx = pred_buf.shape[0] - 1 - steps + s
        active = pred_buf[int(np.clip(idx, 0, pred_buf.shape[0] - 1))]
        target = base + gain * (active - base)
        y = y + np.float32(alpha) * (target - y)
    return y.astype(np.float32)


def inverse_smith_compensate_q_ref(
    q_cmd: np.ndarray,
    qdot_cmd: np.ndarray,
    qdd_cmd: np.ndarray,
    applied_q: np.ndarray,
    command_buffer: np.ndarray,
    *,
    dt: float,
    delay_steps: int,
    actuator_tau_s: float,
    actuator_gain: float = 1.0,
    beta: float = 1.0,
    delay_scale: float = 1.0,
    tau_scale: float = 1.0,
    max_delta_rad: float = 0.0,
    warm_q: np.ndarray | None = None,
    q_low: np.ndarray | None = None,
    q_high: np.ndarray | None = None,
) -> np.ndarray:
    """Invert the delayed first-order actuator model with a Smith predictor.

    The current command will become active after ``delay_steps`` control ticks.
    We first predict where the filtered actuator output will be just before that
    tick using the already queued commands, then solve the one-step first-order
    inverse needed to land near a short-horizon nominal target.
    """
    q = np.asarray(q_cmd, dtype=np.float32)
    qdot = np.asarray(qdot_cmd, dtype=np.float32)
    qdd = np.asarray(qdd_cmd, dtype=np.float32)
    y_now = np.asarray(applied_q, dtype=np.float32)
    if q.shape != qdot.shape or q.shape != qdd.shape or q.shape != y_now.shape:
        raise ValueError("q_cmd, qdot_cmd, qdd_cmd, and applied_q must have the same shape")

    beta_f = float(beta)
    if not np.isfinite(beta_f) or beta_f == 0.0:
        out = q.copy()
    else:
        dt_safe = max(float(dt), 1e-9)
        comp_delay_steps = int(np.clip(round(float(delay_steps) * max(0.0, float(delay_scale))), 0, command_buffer.shape[0] - 1))
        horizon = float(comp_delay_steps) * dt_safe
        tau_est = max(0.0, float(actuator_tau_s) * max(0.0, float(tau_scale)))
        alpha = 1.0 if tau_est <= 1e-9 else dt_safe / (tau_est + dt_safe)
        y_pre = predict_filtered_command_before_delay(
            y_now,
            command_buffer,
            dt=dt_safe,
            delay_steps=comp_delay_steps,
            actuator_tau_s=tau_est,
            actuator_gain=actuator_gain,
            warm_q=warm_q,
            append_placeholder=q,
        )
        target_future = q + horizon * qdot + 0.5 * horizon * horizon * qdd
        base = np.zeros_like(q) if warm_q is None else np.asarray(warm_q, dtype=np.float32)
        gain = float(actuator_gain)
        if not np.isfinite(gain) or abs(gain) < 1e-6:
            gain = 1.0
        filter_target = (target_future - (1.0 - alpha) * y_pre) / max(alpha, 1e-6)
        inv = base + (filter_target - base) / gain
        delta = beta_f * (inv - q)
        limit = float(max_delta_rad)
        if np.isfinite(limit) and limit > 0.0:
            delta = np.clip(delta, -limit, limit)
        out = q + delta.astype(np.float32)

    if q_low is not None or q_high is not None:
        lo = -np.inf if q_low is None else np.asarray(q_low, dtype=np.float32)
        hi = np.inf if q_high is None else np.asarray(q_high, dtype=np.float32)
        out = np.clip(out, lo, hi)
    return out.astype(np.float32)


def inverse_mpc_compensate_q_ref(
    q_cmd: np.ndarray,
    qdot_cmd: np.ndarray,
    qdd_cmd: np.ndarray,
    applied_q: np.ndarray,
    command_buffer: np.ndarray,
    *,
    dt: float,
    delay_steps: int,
    actuator_tau_s: float,
    actuator_gain: float = 1.0,
    beta: float = 1.0,
    delay_scale: float = 1.0,
    tau_scale: float = 1.0,
    horizon_steps: int = 4,
    tracking_weight: float = 1.0,
    nominal_weight: float = 0.25,
    delta_weight: float = 0.08,
    max_delta_rad: float = 0.0,
    warm_q: np.ndarray | None = None,
    q_low: np.ndarray | None = None,
    q_high: np.ndarray | None = None,
) -> np.ndarray:
    """Regularized inverse-MPC command for a delayed first-order actuator.

    The optimizer assumes the newly sent command will become active after the
    queued delay, then remain roughly constant for ``horizon_steps`` control
    ticks.  For each joint this makes the horizon tracking objective quadratic,
    so the regularized optimum has a closed form:

        min_u w_t || actuator_rollout(u) - q_des_future ||^2
              + w_n || u - q_cmd ||^2
              + w_d || u - u_last ||^2

    This is deliberately softer than a pure inverse Smith step.  The nominal and
    delta terms keep the policy-facing action interface smooth while still
    compensating the low-level delay/filter.
    """
    q = np.asarray(q_cmd, dtype=np.float32)
    qdot = np.asarray(qdot_cmd, dtype=np.float32)
    qdd = np.asarray(qdd_cmd, dtype=np.float32)
    y_now = np.asarray(applied_q, dtype=np.float32)
    buf = np.asarray(command_buffer, dtype=np.float32)
    if q.shape != qdot.shape or q.shape != qdd.shape or q.shape != y_now.shape:
        raise ValueError("q_cmd, qdot_cmd, qdd_cmd, and applied_q must have the same shape")
    if buf.ndim != 2 or buf.shape[1] != q.shape[0]:
        raise ValueError("command_buffer must have shape (T, act_dim)")

    beta_f = float(beta)
    if not np.isfinite(beta_f) or beta_f == 0.0:
        out = q.copy()
    else:
        dt_safe = max(float(dt), 1e-9)
        comp_delay_steps = int(np.clip(round(float(delay_steps) * max(0.0, float(delay_scale))), 0, buf.shape[0] - 1))
        h_steps = max(1, int(horizon_steps))
        tau_est = max(0.0, float(actuator_tau_s) * max(0.0, float(tau_scale)))
        alpha = 1.0 if tau_est <= 1e-9 else dt_safe / (tau_est + dt_safe)
        y_pre = predict_filtered_command_before_delay(
            y_now,
            buf,
            dt=dt_safe,
            delay_steps=comp_delay_steps,
            actuator_tau_s=tau_est,
            actuator_gain=actuator_gain,
            warm_q=warm_q,
            append_placeholder=q,
        )

        total_horizon = (float(comp_delay_steps) + float(h_steps)) * dt_safe
        q_des = q + total_horizon * qdot + 0.5 * total_horizon * total_horizon * qdd
        decay = float(np.clip(1.0 - alpha, 0.0, 1.0)) ** h_steps
        response = 1.0 - decay
        base = np.zeros_like(q) if warm_q is None else np.asarray(warm_q, dtype=np.float32)
        gain = float(actuator_gain)
        if not np.isfinite(gain) or abs(gain) < 1e-6:
            gain = 1.0
        k = response * gain
        b = decay * y_pre + response * (1.0 - gain) * base
        u_last = buf[-1]

        wt = max(0.0, float(tracking_weight))
        wn = max(0.0, float(nominal_weight))
        wd = max(0.0, float(delta_weight))
        denom = wt * k * k + wn + wd
        if denom <= 1e-9 or not np.isfinite(denom):
            u_star = q.copy()
        else:
            u_star = (wt * k * (q_des - b) + wn * q + wd * u_last) / denom
        delta = beta_f * (u_star - q)
        limit = float(max_delta_rad)
        if np.isfinite(limit) and limit > 0.0:
            delta = np.clip(delta, -limit, limit)
        out = q + delta.astype(np.float32)

    if q_low is not None or q_high is not None:
        lo = -np.inf if q_low is None else np.asarray(q_low, dtype=np.float32)
        hi = np.inf if q_high is None else np.asarray(q_high, dtype=np.float32)
        out = np.clip(out, lo, hi)
    return out.astype(np.float32)


def compensate_q_ref(
    mode: str,
    q_cmd: np.ndarray,
    qdot_cmd: np.ndarray,
    qdd_cmd: np.ndarray,
    *,
    dt: float,
    delay_steps: int,
    actuator_tau_s: float,
    actuator_gain: float = 1.0,
    lead_beta: float = 0.0,
    lead_delay_scale: float = 1.0,
    lead_tau_scale: float = 1.0,
    lead_max_delta_rad: float = 0.0,
    inverse_beta: float = 1.0,
    inverse_delay_scale: float = 1.0,
    inverse_tau_scale: float = 1.0,
    inverse_max_delta_rad: float = 0.0,
    mpc_beta: float = 1.0,
    mpc_delay_scale: float = 1.0,
    mpc_tau_scale: float = 1.0,
    mpc_horizon_steps: int = 4,
    mpc_tracking_weight: float = 1.0,
    mpc_nominal_weight: float = 0.25,
    mpc_delta_weight: float = 0.08,
    mpc_max_delta_rad: float = 0.0,
    applied_q: np.ndarray | None = None,
    command_buffer: np.ndarray | None = None,
    warm_q: np.ndarray | None = None,
    q_low: np.ndarray | None = None,
    q_high: np.ndarray | None = None,
) -> np.ndarray:
    """Shared NumPy dispatcher for actuator command compensation."""
    mode_norm = str(mode or "none").strip().lower().replace("-", "_")
    if mode_norm in {"none", "off", "false", "0"}:
        out = np.asarray(q_cmd, dtype=np.float32).copy()
        if q_low is not None or q_high is not None:
            lo = -np.inf if q_low is None else np.asarray(q_low, dtype=np.float32)
            hi = np.inf if q_high is None else np.asarray(q_high, dtype=np.float32)
            out = np.clip(out, lo, hi)
        return out.astype(np.float32)
    if mode_norm == "lead":
        return lead_compensate_q_ref(
            q_cmd,
            qdot_cmd,
            qdd_cmd,
            dt=dt,
            delay_steps=delay_steps,
            actuator_tau_s=actuator_tau_s,
            beta=lead_beta,
            delay_scale=lead_delay_scale,
            tau_scale=lead_tau_scale,
            max_delta_rad=lead_max_delta_rad,
            q_low=q_low,
            q_high=q_high,
        )
    if mode_norm in {"inverse_smith", "smith", "inverse"}:
        if applied_q is None or command_buffer is None:
            raise ValueError("inverse_smith compensation requires applied_q and command_buffer")
        return inverse_smith_compensate_q_ref(
            q_cmd,
            qdot_cmd,
            qdd_cmd,
            applied_q,
            command_buffer,
            dt=dt,
            delay_steps=delay_steps,
            actuator_tau_s=actuator_tau_s,
            actuator_gain=actuator_gain,
            beta=inverse_beta,
            delay_scale=inverse_delay_scale,
            tau_scale=inverse_tau_scale,
            max_delta_rad=inverse_max_delta_rad,
            warm_q=warm_q,
            q_low=q_low,
            q_high=q_high,
        )
    if mode_norm in {"inverse_mpc", "regularized_inverse_mpc", "mpc"}:
        if applied_q is None or command_buffer is None:
            raise ValueError("inverse_mpc compensation requires applied_q and command_buffer")
        return inverse_mpc_compensate_q_ref(
            q_cmd,
            qdot_cmd,
            qdd_cmd,
            applied_q,
            command_buffer,
            dt=dt,
            delay_steps=delay_steps,
            actuator_tau_s=actuator_tau_s,
            actuator_gain=actuator_gain,
            beta=mpc_beta,
            delay_scale=mpc_delay_scale,
            tau_scale=mpc_tau_scale,
            horizon_steps=mpc_horizon_steps,
            tracking_weight=mpc_tracking_weight,
            nominal_weight=mpc_nominal_weight,
            delta_weight=mpc_delta_weight,
            max_delta_rad=mpc_max_delta_rad,
            warm_q=warm_q,
            q_low=q_low,
            q_high=q_high,
        )
    raise ValueError(f"unknown actuator compensation mode: {mode}")


def estimate_contact_time(
    z_rel: float,
    vz_rel: float,
    *,
    gravity: float = 9.81,
    contact_height_offset: float = 0.0,
    max_contact_time: float = 0.5,
    ball_lost: bool = False,
) -> float:
    """Estimate time until the ball reaches a contact height relative to racket."""
    max_t = max(0.0, float(max_contact_time))
    z = float(z_rel)
    vz = float(vz_rel)
    g = max(abs(float(gravity)), 1e-9)
    if ball_lost or not all(math.isfinite(v) for v in (z, vz, g)):
        return max_t
    if abs(vz) > 50.0 or abs(z) > 10.0:
        return max_t

    h = z - float(contact_height_offset)
    disc = vz * vz + 2.0 * g * h
    if disc < 0.0 or not math.isfinite(disc):
        return max_t
    root = math.sqrt(disc)
    candidates = [(vz + root) / g, (vz - root) / g]
    positive = [t for t in candidates if t >= 0.0 and math.isfinite(t)]
    if not positive:
        return max_t
    return float(np.clip(min(positive), 0.0, max_t))
