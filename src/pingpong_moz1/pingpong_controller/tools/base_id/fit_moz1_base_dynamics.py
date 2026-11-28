#!/usr/bin/env python3
"""Offline system identification, audit, Chinese report, and figure generator for MOZ1 base.

100% offline (zero ROS runtime dependency). Consumes one or more run directories produced
by ``run_base_id_experiment.py`` and writes four auditable artifacts into a NEW output
directory (refusing to overwrite existing directories):
  1. ``DATA_AUDIT.json`` — per-trial acceptance/rejection reasons & clock-offset audit.
  2. ``moz1_base_identified_params_v1.json`` — standardized simulation input contract.
  3. ``BASE_ID_REPORT_CN.md`` — Chinese technical report + MuJoCo/MJX modeling guide.
  4. ``fig_moz1_base_id_summary.pdf`` and ``.png`` — 4-panel publication-grade summary plot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, nnls

PARAMS_SCHEMA_VERSION = "moz1_base_identified_params_v1"
AXES: Tuple[str, ...] = ("vx", "vy", "wz")


class OutputDirExistsError(FileExistsError):
    """Raised when ``--out`` already exists."""


@dataclass
class LoadedTrial:
    """One audited trial loaded from a run directory."""

    run_dir: Path
    stage: str
    trial_id: str
    waveform_name: str
    tier: str
    split_role: str
    primary_axis: str
    status: str
    t_s: np.ndarray
    cmd: Dict[str, np.ndarray]  # keys: vx, vy, wz
    meas: Dict[str, np.ndarray]  # keys: vx, vy, wz
    odom_x_m: np.ndarray
    odom_y_m: np.ndarray
    odom_yaw_rad: np.ndarray
    racket_pos_m: Optional[np.ndarray]
    racket_normal: Optional[np.ndarray]
    imu_rpy_rad: Optional[np.ndarray]
    phase_labels: np.ndarray
    base_raw_wall_s: Optional[np.ndarray]
    base_raw_epoch_s: Optional[np.ndarray]
    base_raw_header_s: Optional[np.ndarray]
    arm_cmd_wall_s: Optional[np.ndarray]
    arm_raw_wall_s: Optional[np.ndarray]
    arm_raw_epoch_s: Optional[np.ndarray]
    arm_raw_header_s: Optional[np.ndarray]
    mocap_raw_wall_s: Optional[np.ndarray]
    mocap_raw_epoch_s: Optional[np.ndarray]
    mocap_raw_header_s: Optional[np.ndarray]
    mocap_pos_xyz_m: Optional[np.ndarray]
    mocap_yaw_rad: Optional[np.ndarray]
    mocap_rpy_rad: Optional[np.ndarray] = None
    mocap_quat_xyzw: Optional[np.ndarray] = None


def simulate_fopdt_asymmetric(
    t_s: np.ndarray,
    u_cmd: np.ndarray,
    *,
    delay_s: float,
    time_constant_s: float,
    gain: float,
    a_acc_max: float,
    a_brake_max: float,
    v_max: float = 1e6,
    v0: float = 0.0,
    max_substep_s: float = 0.004,
) -> np.ndarray:
    """Simulate a delayed first-order velocity plant with asymmetric accel/brake clamps.

    Continuous-time dynamics:
        u_eff(t) = clip(gain * u_cmd(t - delay_s), -v_max, +v_max)
        a_des    = (u_eff(t) - v) / time_constant_s
        |a| <= a_brake_max  when braking (v * a_des < 0 and |v| > 1e-4)
        |a| <= a_acc_max    otherwise
        dv/dt = a

    Sub-stepping at ``<= 4 ms`` with linear interpolation of ``u_cmd(t - delay_s)``
    makes the simulated trajectory smooth in ``delay_s`` for sub-millisecond fitting.
    """
    t = np.asarray(t_s, dtype=np.float64)
    u = np.asarray(u_cmd, dtype=np.float64)
    n = len(t)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    tau = max(0.0, float(delay_s))
    tc = max(1e-3, float(time_constant_s))
    k_gain = float(gain)
    a_acc = max(1e-3, float(a_acc_max))
    a_brk = max(1e-3, float(a_brake_max))
    vmax = max(1e-3, float(v_max))

    y = np.empty(n, dtype=np.float64)
    v = float(v0)
    y[0] = v
    t0 = float(t[0])
    u0 = float(u[0])

    # Precompute uniform or per-step subgrid for fast vectorized interpolation of u(t - tau).
    for i in range(n - 1):
        dt_total = float(t[i + 1] - t[i])
        if dt_total <= 0.0:
            y[i + 1] = v
            continue
        n_sub = max(1, int(math.ceil(dt_total / max_substep_s)))
        dt = dt_total / n_sub
        t_cur = float(t[i])
        for s_idx in range(n_sub):
            t_mid = t_cur + (s_idx + 0.5) * dt
            t_q = t_mid - tau
            if t_q <= t0:
                u_del = u0
            else:
                u_del = float(np.interp(t_q, t, u))
            v_ref = k_gain * u_del
            if v_ref > vmax:
                v_ref = vmax
            elif v_ref < -vmax:
                v_ref = -vmax

            err = v_ref - v
            a_des = err / tc
            is_braking = (abs(v) > 1e-4) and ((v * a_des) < 0.0)
            a_lim = a_brk if is_braking else a_acc
            if a_des > a_lim:
                a_eff = a_lim
            elif a_des < -a_lim:
                a_eff = -a_lim
            else:
                a_eff = a_des

            v_next = v + a_eff * dt
            # Prevent zero-crossing chatter during braking toward 0.
            if is_braking and abs(v_ref) < 1e-6 and (v * v_next < 0.0):
                v_next = 0.0
            v = v_next
        y[i + 1] = v
    return y


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _estimate_clock_offset(
    header_s: Optional[np.ndarray],
    epoch_s: Optional[np.ndarray],
) -> Optional[Dict[str, Any]]:
    if header_s is None or epoch_s is None:
        return None
    h = np.asarray(header_s, dtype=np.float64)
    e = np.asarray(epoch_s, dtype=np.float64)
    m = min(len(h), len(e))
    if m == 0:
        return None
    mask = np.isfinite(h[:m]) & np.isfinite(e[:m])
    if not np.any(mask):
        return None
    deltas = h[:m][mask] - e[:m][mask]
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med)))
    return {
        "n_samples": int(deltas.size),
        "median_header_minus_epoch_s": round(med, 6),
        "mad_s": round(mad, 6),
        "large_bias_detected": bool(abs(med) > 0.5),
    }


def load_and_audit_runs(
    run_dirs: Sequence[Path],
) -> Tuple[List[LoadedTrial], Dict[str, Any], Dict[str, str]]:
    """Load one or more run directories and perform strict per-trial data auditing."""
    accepted: List[LoadedTrial] = []
    accepted_audit: List[Dict[str, Any]] = []
    rejected_audit: List[Dict[str, Any]] = []
    clock_offsets_by_run: Dict[str, Any] = {}
    source_hashes: Dict[str, str] = {}

    for rdir in run_dirs:
        run_path = Path(rdir).resolve()
        manifest_path = run_path / "MANIFEST.json"
        npz_path = run_path / "aligned_timeseries.npz"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing MANIFEST.json in run directory: {run_path}")
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing aligned_timeseries.npz in run directory: {run_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hashes[run_path.name] = _sha256_file(manifest_path)
        clock_offsets_by_run[run_path.name] = dict(manifest.get("clock_offset_summary_s") or {})
        stage = str(manifest.get("stage", "unknown"))

        npz = np.load(str(npz_path), allow_pickle=False)
        for tmeta in manifest.get("trials", []):
            tid = str(tmeta["trial_id"])
            status = str(tmeta.get("status", "completed"))
            reasons: List[str] = []

            t_key = f"{tid}__t_rel_s"
            if t_key not in npz.files:
                reasons.append("missing_aligned_timeseries_in_npz")
                rejected_audit.append({
                    "run_dir": run_path.name,
                    "trial_id": tid,
                    "stage": stage,
                    "status": status,
                    "reasons": reasons,
                })
                continue

            t_s = np.asarray(npz[t_key], dtype=np.float64)
            cmd = {
                "vx": np.asarray(npz[f"{tid}__cmd_vx_m_s"], dtype=np.float64),
                "vy": np.asarray(npz[f"{tid}__cmd_vy_m_s"], dtype=np.float64),
                "wz": np.asarray(npz[f"{tid}__cmd_wz_rad_s"], dtype=np.float64),
            }
            meas = {
                "vx": np.asarray(npz[f"{tid}__meas_vx_m_s"], dtype=np.float64),
                "vy": np.asarray(npz[f"{tid}__meas_vy_m_s"], dtype=np.float64),
                "wz": np.asarray(npz[f"{tid}__meas_wz_rad_s"], dtype=np.float64),
            }

            if len(t_s) < 10:
                reasons.append(f"too_few_samples ({len(t_s)} < 10)")
            dt_arr = np.diff(t_s) if len(t_s) >= 2 else np.array([0.0])
            min_dt = float(np.min(dt_arr))
            med_dt = float(np.median(dt_arr)) if len(dt_arr) > 0 else 0.02
            max_dt = float(np.max(dt_arr))
            if min_dt < -1e-6:
                reasons.append(f"wall_time_backward (min_dt={min_dt:.6f}s)")
            if med_dt > 0.0 and max_dt > max(0.30, 10.0 * med_dt):
                reasons.append(f"excessive_frame_gap (max_dt={max_dt:.3f}s, med_dt={med_dt:.4f}s)")

            if status not in ("completed",):
                reasons.append(f"trial_truncated_or_aborted (status={status})")

            cmd_nonfinite = sum(int(np.sum(~np.isfinite(cmd[ax]))) for ax in AXES)
            if cmd_nonfinite > 0:
                reasons.append(f"cmd_nonfinite_count={cmd_nonfinite}")

            meas_nan_frac = max(
                float(np.mean(~np.isfinite(meas[ax]))) if len(meas[ax]) > 0 else 1.0
                for ax in AXES
            )
            if meas_nan_frac > 0.25:
                reasons.append(f"meas_missing_fraction={meas_nan_frac:.3f} (>0.25)")

            def _opt_arr(suffix: str) -> Optional[np.ndarray]:
                k = f"{tid}__{suffix}"
                return np.asarray(npz[k]) if k in npz.files else None

            base_raw_wall = _opt_arr("base_raw_wall_s")
            base_raw_epoch = _opt_arr("base_raw_epoch_s")
            base_raw_header = _opt_arr("base_raw_header_s")
            arm_raw_wall = _opt_arr("arm_raw_wall_s")
            arm_raw_epoch = _opt_arr("arm_raw_epoch_s")
            arm_raw_header = _opt_arr("arm_raw_header_s")
            mocap_raw_wall = _opt_arr("mocap_raw_wall_s")
            mocap_raw_epoch = _opt_arr("mocap_raw_epoch_s")
            mocap_raw_header = _opt_arr("mocap_raw_header_s")

            # Check header timestamp monotonicity after constant-bias check.
            base_clk = _estimate_clock_offset(base_raw_header, base_raw_epoch)
            arm_clk = _estimate_clock_offset(arm_raw_header, arm_raw_epoch)
            mocap_clk = _estimate_clock_offset(mocap_raw_header, mocap_raw_epoch)

            if base_raw_header is not None and len(base_raw_header) >= 2:
                finite_h = base_raw_header[np.isfinite(base_raw_header)]
                if len(finite_h) >= 2 and float(np.min(np.diff(finite_h))) < -1e-4:
                    reasons.append("base_header_timestamp_backward")

            trial_clock_info = {
                k: v for k, v in (
                    ("base_state", base_clk),
                    ("arm_state", arm_clk),
                    ("mocap_base", mocap_clk),
                ) if v is not None
            }

            if reasons:
                rejected_audit.append({
                    "run_dir": run_path.name,
                    "trial_id": tid,
                    "stage": stage,
                    "tier": str(tmeta.get("tier", "unknown")),
                    "status": status,
                    "n_samples": int(len(t_s)),
                    "reasons": reasons,
                    "clock_offsets": trial_clock_info,
                })
                continue

            # Interpolate rare edge NaNs (<25%) cleanly along valid measured interior.
            for ax in AXES:
                arr = meas[ax].copy()
                good = np.isfinite(arr)
                if not np.all(good) and np.any(good):
                    arr[~good] = np.interp(t_s[~good], t_s[good], arr[good])
                    meas[ax] = arr

            odom_x = np.asarray(npz[f"{tid}__odom_x_m"], dtype=np.float64) if f"{tid}__odom_x_m" in npz.files else np.zeros_like(t_s)
            odom_y = np.asarray(npz[f"{tid}__odom_y_m"], dtype=np.float64) if f"{tid}__odom_y_m" in npz.files else np.zeros_like(t_s)
            odom_yaw = np.asarray(npz[f"{tid}__odom_yaw_rad"], dtype=np.float64) if f"{tid}__odom_yaw_rad" in npz.files else np.zeros_like(t_s)
            for _arr in (odom_x, odom_y, odom_yaw):
                _good = np.isfinite(_arr)
                if not np.all(_good) and np.any(_good):
                    _arr[~_good] = np.interp(t_s[~_good], t_s[_good], _arr[_good])
            phases = np.asarray(npz[f"{tid}__phase_labels"]) if f"{tid}__phase_labels" in npz.files else np.array([""] * len(t_s))

            loaded = LoadedTrial(
                run_dir=run_path,
                stage=stage,
                trial_id=tid,
                waveform_name=str(tmeta.get("waveform_name", tid)),
                tier=str(tmeta.get("tier", "med")),
                split_role=str(tmeta.get("split_role", "train")),
                primary_axis=str(tmeta.get("primary_axis", "multi")),
                status=status,
                t_s=t_s,
                cmd=cmd,
                meas=meas,
                odom_x_m=odom_x,
                odom_y_m=odom_y,
                odom_yaw_rad=odom_yaw,
                racket_pos_m=_opt_arr("racket_pos_m"),
                racket_normal=_opt_arr("racket_normal"),
                imu_rpy_rad=_opt_arr("imu_rpy_rad"),
                phase_labels=phases,
                base_raw_wall_s=base_raw_wall,
                base_raw_epoch_s=base_raw_epoch,
                base_raw_header_s=base_raw_header,
                arm_cmd_wall_s=_opt_arr("arm_cmd_wall_s"),
                arm_raw_wall_s=arm_raw_wall,
                arm_raw_epoch_s=arm_raw_epoch,
                arm_raw_header_s=arm_raw_header,
                mocap_raw_wall_s=mocap_raw_wall,
                mocap_raw_epoch_s=mocap_raw_epoch,
                mocap_raw_header_s=mocap_raw_header,
                mocap_pos_xyz_m=_opt_arr("mocap_pos_xyz_m"),
                mocap_yaw_rad=_opt_arr("mocap_yaw_rad"),
                mocap_rpy_rad=_opt_arr("mocap_rpy_rad"),
                mocap_quat_xyzw=_opt_arr("mocap_quat_xyzw"),
            )
            accepted.append(loaded)
            accepted_audit.append({
                "run_dir": run_path.name,
                "trial_id": tid,
                "stage": stage,
                "tier": loaded.tier,
                "split_role": loaded.split_role,
                "primary_axis": loaded.primary_axis,
                "n_samples": int(len(t_s)),
                "duration_s": round(float(t_s[-1] - t_s[0]), 3),
                "med_dt_ms": round(med_dt * 1000.0, 3),
                "max_dt_ms": round(max_dt * 1000.0, 3),
                "meas_nan_fraction": round(meas_nan_frac, 4),
                "clock_offsets": trial_clock_info,
            })

    # Aggregate global clock offset summary across all accepted trials.
    global_clocks: Dict[str, List[float]] = {}
    for row in accepted_audit:
        for src, info in (row.get("clock_offsets") or {}).items():
            global_clocks.setdefault(src, []).append(float(info["median_header_minus_epoch_s"]))
    global_clock_summary = {
        src: {
            "median_s": round(float(np.median(vals)), 6),
            "min_s": round(float(np.min(vals)), 6),
            "max_s": round(float(np.max(vals)), 6),
            "large_bias_separated": bool(abs(float(np.median(vals))) > 0.5),
        }
        for src, vals in sorted(global_clocks.items())
    }

    audit_doc: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_run_dirs": len(run_dirs),
        "n_accepted_trials": len(accepted_audit),
        "n_rejected_trials": len(rejected_audit),
        "clock_offset_audit_s": global_clock_summary,
        "clock_offsets_by_run": clock_offsets_by_run,
        "accepted_trials": accepted_audit,
        "rejected_trials": rejected_audit,
    }
    return accepted, audit_doc, source_hashes


def _estimate_delay_xcorr_and_onset10(
    trials: Sequence[LoadedTrial],
    axis: str,
) -> Tuple[float, float, Dict[str, float]]:
    """Estimate pure delay via cross-correlation and 10% step onset, plus brake/overshoot stats."""
    xcorr_lags: List[float] = []
    onset10_lags: List[float] = []
    brake_times: List[float] = []
    brake_dists: List[float] = []
    overshoots: List[float] = []
    steady_speeds: List[float] = []

    for tr in trials:
        t = tr.t_s
        u = tr.cmd[axis]
        y = tr.meas[axis]
        if len(t) < 20 or float(np.max(np.abs(u))) < 0.05:
            continue
        dt = float(np.median(np.diff(t)))
        if dt <= 0.0:
            continue

        # 1. Cross-correlation on finite-differenced edges (minimizes first-order lag bias).
        du = np.diff(u)
        dy = np.diff(y)
        if np.std(du) > 1e-6 and np.std(dy) > 1e-6:
            max_lag_steps = min(len(du) // 3, int(round(0.45 / dt)))
            best_lag = 0
            best_corr = -1e18
            for lag in range(0, max_lag_steps + 1):
                a = du[: len(du) - lag] if lag > 0 else du
                b = dy[lag:]
                c = float(np.dot(a, b))
                if c > best_corr:
                    best_corr = c
                    best_lag = lag
            xcorr_lags.append(best_lag * dt)

        # 2. Step-edge onset (10% of steady-state), overshoot, and emergency-brake metrics.
        u_peak = float(np.max(np.abs(u)))
        thresh = 0.5 * u_peak
        active = np.abs(u) >= thresh
        edges_up = np.where((~active[:-1]) & active[1:])[0] + 1
        edges_dn = np.where(active[:-1] & (~active[1:]))[0] + 1

        for idx_up in edges_up:
            # Find matching end of step hold.
            dn_candidates = edges_dn[edges_dn > idx_up]
            idx_end = int(dn_candidates[0]) if len(dn_candidates) > 0 else len(t) - 1
            if idx_end - idx_up < max(6, int(round(0.25 / dt))):
                continue
            sign = 1.0 if u[idx_up] >= 0.0 else -1.0
            seg_y = sign * y[idx_up:idx_end]
            late_start = max(1, int(0.65 * len(seg_y)))
            steady = float(np.median(seg_y[late_start:]))
            if steady > 0.04:
                steady_speeds.append(steady)
                cross = np.where(seg_y >= 0.10 * steady)[0]
                if len(cross) > 0:
                    onset10_lags.append(float(t[idx_up + int(cross[0])] - t[idx_up]))
                peak = float(np.max(seg_y))
                overshoots.append(max(0.0, (peak - steady) / steady))

        for idx_dn in edges_dn:
            # Emergency brake: step back to ~0.
            if abs(u[idx_dn]) > 0.05 * u_peak:
                continue
            look_end = min(len(t), idx_dn + int(round(1.5 / dt)))
            if look_end - idx_dn < 5:
                continue
            v_start = abs(float(y[idx_dn]))
            if v_start < 0.05:
                continue
            seg_abs = np.abs(y[idx_dn:look_end])
            stop_idx = np.where(seg_abs <= max(0.015, 0.05 * v_start))[0]
            end_rel = int(stop_idx[0]) if len(stop_idx) > 0 else len(seg_abs) - 1
            brake_times.append(float(t[idx_dn + end_rel] - t[idx_dn]))
            t_seg = t[idx_dn : idx_dn + end_rel + 1]
            v_seg = seg_abs[: end_rel + 1]
            brake_dists.append(float(np.trapz(v_seg, t_seg)))

    delay_xcorr = float(np.median(xcorr_lags)) if xcorr_lags else 0.0
    delay_onset10 = float(np.median(onset10_lags)) if onset10_lags else delay_xcorr
    extras = {
        "v_max": float(np.max(steady_speeds)) if steady_speeds else 0.0,
        "overshoot_frac": float(np.median(overshoots)) if overshoots else 0.0,
        "brake_time_s": float(np.median(brake_times)) if brake_times else 0.0,
        "brake_distance": float(np.median(brake_dists)) if brake_dists else 0.0,
    }
    return delay_xcorr, delay_onset10, extras


def _eval_axis_predictions(
    trials: Sequence[LoadedTrial],
    axis: str,
    params: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Simulate fitted FOPDT+asymmetric model across trials and compute error statistics."""
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for tr in trials:
        yhat = simulate_fopdt_asymmetric(
            tr.t_s,
            tr.cmd[axis],
            delay_s=params["delay_s"],
            time_constant_s=params["time_constant_s"],
            gain=params["gain"],
            a_acc_max=params["a_acc_max"],
            a_brake_max=params["a_brake_max"],
            v_max=float(params.get("v_max", 1e6)),
            v0=float(tr.meas[axis][0]),
        )
        preds.append(yhat)
        trues.append(tr.meas[axis])

    if not preds:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            {"rmse": 0.0, "p95": 0.0, "max_err": 0.0, "r2": 0.0},
        )
    y_hat = np.concatenate(preds)
    y_true = np.concatenate(trues)
    err = y_hat - y_true
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err * err)))
    p95 = float(np.percentile(abs_err, 95))
    max_err = float(np.max(abs_err))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return y_true, y_hat, {"rmse": rmse, "p95": p95, "max_err": max_err, "r2": r2}


def fit_single_axis(
    all_trials: Sequence[LoadedTrial],
    axis: str,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Identify per-axis FOPDT + asymmetric accel/brake parameters with held-out validation."""
    exc_trials = [
        tr for tr in all_trials
        if tr.stage != "s0_idle_noise" and float(np.max(np.abs(tr.cmd[axis]))) >= 0.05
    ]
    if not exc_trials:
        return (
            {
                "identified": False,
                "n_train_trials": 0,
                "n_heldout_trials": 0,
                "delay_s": 0.0,
                "delay_xcorr_s": 0.0,
                "delay_onset10_s": 0.0,
                "time_constant_s": 0.1,
                "gain": 1.0,
                "v_max": 0.0,
                "a_acc_max": 0.0,
                "a_brake_max": 0.0,
                "overshoot_frac": 0.0,
                "brake_time_s": 0.0,
                "brake_distance": 0.0,
                "train_rmse": 0.0,
                "heldout_rmse": 0.0,
                "heldout_p95": 0.0,
                "heldout_max_err": 0.0,
                "heldout_r2": 0.0,
            },
            None,
        )

    # Split train vs heldout. Prefer dedicated single-axis trials for training, plus multi-axis.
    train_trials = [tr for tr in exc_trials if tr.split_role == "train"]
    heldout_trials = [tr for tr in exc_trials if tr.split_role == "heldout"]
    if not train_trials:
        train_trials = list(exc_trials)
    if not heldout_trials:
        if len(train_trials) >= 2:
            heldout_trials = [train_trials[-1]]
            train_trials = train_trials[:-1]
        else:
            heldout_trials = list(train_trials)

    delay_xcorr, delay_onset10, emp_stats = _estimate_delay_xcorr_and_onset10(exc_trials, axis)

    v_max_obs = max(
        emp_stats["v_max"],
        float(np.max([np.max(np.abs(tr.meas[axis])) for tr in exc_trials])),
    )
    u_max_obs = float(np.max([np.max(np.abs(tr.cmd[axis])) for tr in exc_trials]))

    # Initial gain guess from unsaturated steady-state samples (below observed speed ceiling).
    u_all = np.concatenate([tr.cmd[axis] for tr in train_trials])
    y_all = np.concatenate([tr.meas[axis] for tr in train_trials])
    unsat_cap = 0.92 * v_max_obs if v_max_obs > 0.1 else u_max_obs
    steady_mask = (np.abs(u_all) > 0.15 * max(u_max_obs, 1e-6)) & (np.abs(u_all) <= max(unsat_cap, 0.2))
    if not np.any(steady_mask):
        steady_mask = np.abs(u_all) > 0.25 * max(u_max_obs, 1e-6)
    if np.any(steady_mask):
        gain_init = float(
            np.clip(
                np.dot(u_all[steady_mask], y_all[steady_mask])
                / max(np.dot(u_all[steady_mask], u_all[steady_mask]), 1e-9),
                0.15,
                1.35,
            )
        )
    else:
        gain_init = 1.0

    # Empirical acceleration scale for sensible initialization.
    emp_slopes: List[float] = []
    for tr in train_trials:
        dt = float(np.median(np.diff(tr.t_s)))
        if dt > 0.0:
            dy = np.abs(np.gradient(tr.meas[axis], tr.t_s))
            emp_slopes.append(float(np.percentile(dy, 98)))
    a_init = float(np.clip(np.median(emp_slopes) if emp_slopes else 1.5, 0.3, 8.0))
    v_sat_init = max(v_max_obs * 1.02, 0.25)

    def _residuals(theta: np.ndarray) -> np.ndarray:
        delay_s, tc_s, gain, a_acc, a_brk, v_sat = (
            float(theta[0]),
            float(theta[1]),
            float(theta[2]),
            float(theta[3]),
            float(theta[4]),
            float(theta[5]),
        )
        res_list: List[np.ndarray] = []
        for tr in train_trials:
            yhat = simulate_fopdt_asymmetric(
                tr.t_s,
                tr.cmd[axis],
                delay_s=delay_s,
                time_constant_s=tc_s,
                gain=gain,
                a_acc_max=a_acc,
                a_brake_max=a_brk,
                v_max=v_sat,
                v0=float(tr.meas[axis][0]),
            )
            res_list.append(yhat - tr.meas[axis])
        return np.concatenate(res_list)

    # Stage 1: Coarse grid over delay & time constant to avoid local minima.
    candidates: List[Tuple[float, np.ndarray]] = []
    tau_grid = np.unique(
        np.clip(
            np.concatenate([
                np.linspace(0.0, 0.28, 15),
                np.array([delay_xcorr, delay_onset10]),
            ]),
            0.0,
            0.40,
        )
    )
    for tau_c in tau_grid:
        for tc_c in (0.05, 0.14, 0.28):
            th = np.array([tau_c, tc_c, gain_init, a_init, a_init * 1.15, v_sat_init], dtype=np.float64)
            r = _residuals(th)
            cost = float(np.dot(r, r))
            candidates.append((cost, th))
    candidates.sort(key=lambda item: item[0])

    # Stage 2: Bounded nonlinear least-squares from top-3 coarse seeds.
    v_sat_lb = max(0.10, 0.50 * v_max_obs)
    v_sat_ub = max(3.0, 2.5 * max(u_max_obs, v_max_obs))
    lb = np.array([0.0, 0.005, 0.05, 0.10, 0.10, v_sat_lb], dtype=np.float64)
    ub = np.array([0.50, 1.50, 1.50, 15.0, 15.0, v_sat_ub], dtype=np.float64)
    best_opt = None
    for _, seed in candidates[:3]:
        seed_clipped = np.clip(seed, lb + 1e-4, ub - 1e-4)
        opt = least_squares(
            _residuals,
            seed_clipped,
            bounds=(lb, ub),
            method="trf",
            diff_step=np.array([1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3]),
            max_nfev=150,
        )
        if best_opt is None or opt.cost < best_opt.cost:
            best_opt = opt

    assert best_opt is not None
    fitted = {
        "delay_s": float(best_opt.x[0]),
        "time_constant_s": float(best_opt.x[1]),
        "gain": float(best_opt.x[2]),
        "a_acc_max": float(best_opt.x[3]),
        "a_brake_max": float(best_opt.x[4]),
        "v_max": float(best_opt.x[5]),
    }

    _, _, train_metrics = _eval_axis_predictions(train_trials, axis, fitted)
    _, _, held_metrics = _eval_axis_predictions(heldout_trials, axis, fitted)

    # Choose representative step trial for publication figure Panel A/B.
    step_candidates = [
        tr for tr in (heldout_trials + train_trials)
        if tr.primary_axis == axis and "step" in tr.waveform_name
    ]
    plot_trial = step_candidates[0] if step_candidates else heldout_trials[0]
    plot_pred = simulate_fopdt_asymmetric(
        plot_trial.t_s,
        plot_trial.cmd[axis],
        delay_s=fitted["delay_s"],
        time_constant_s=fitted["time_constant_s"],
        gain=fitted["gain"],
        a_acc_max=fitted["a_acc_max"],
        a_brake_max=fitted["a_brake_max"],
        v_max=fitted["v_max"],
        v0=float(plot_trial.meas[axis][0]),
    )
    plot_bundle = {
        "trial_id": plot_trial.trial_id,
        "t_s": plot_trial.t_s,
        "cmd": plot_trial.cmd[axis],
        "meas": plot_trial.meas[axis],
        "pred": plot_pred,
        "fitted": fitted,
        "heldout_rmse": held_metrics["rmse"],
    }

    v_max_report = min(v_max_obs, fitted["v_max"]) if u_max_obs > v_max_obs * 1.05 else v_max_obs
    axis_summary = {
        "identified": True,
        "n_train_trials": len(train_trials),
        "n_heldout_trials": len(heldout_trials),
        "delay_s": round(fitted["delay_s"], 6),
        "delay_xcorr_s": round(delay_xcorr, 6),
        "delay_onset10_s": round(delay_onset10, 6),
        "time_constant_s": round(fitted["time_constant_s"], 6),
        "gain": round(fitted["gain"], 5),
        "v_max": round(v_max_report, 4),
        "a_acc_max": round(fitted["a_acc_max"], 4),
        "a_brake_max": round(fitted["a_brake_max"], 4),
        "overshoot_frac": round(emp_stats["overshoot_frac"], 5),
        "brake_time_s": round(emp_stats["brake_time_s"], 4),
        "brake_distance": round(emp_stats["brake_distance"], 4),
        "train_rmse": round(train_metrics["rmse"], 5),
        "heldout_rmse": round(held_metrics["rmse"], 5),
        "heldout_p95": round(held_metrics["p95"], 5),
        "heldout_max_err": round(held_metrics["max_err"], 5),
        "heldout_r2": round(held_metrics["r2"], 5),
    }
    return axis_summary, plot_bundle


def estimate_cross_coupling_and_kinematics(
    trials: Sequence[LoadedTrial],
    axes_params: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[List[float]], str, Dict[str, float]]:
    """Regress delay-aligned measured [vx, vy, wz] onto commanded [ux, uy, uz]."""
    u_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []

    delays = {ax: float(axes_params[ax].get("delay_s", 0.0)) for ax in AXES}
    mean_delay = float(np.mean([d for d in delays.values() if d >= 0.0]))

    for tr in trials:
        if tr.stage == "s0_idle_noise" or len(tr.t_s) < 10:
            continue
        t = tr.t_s
        u_shift = np.column_stack([
            np.clip(
                np.interp(t - mean_delay, t, tr.cmd[ax], left=0.0, right=0.0),
                -max(float(axes_params[ax].get("v_max", 1e6)), 1e-3),
                max(float(axes_params[ax].get("v_max", 1e6)), 1e-3),
            )
            for ax in AXES
        ])
        y_mat = np.column_stack([tr.meas[ax] for ax in AXES])
        # Exclude sharp transient steps so steady/tracking gain dominates coupling matrix.
        du_norm = np.linalg.norm(np.gradient(u_shift, t, axis=0), axis=1)
        u_norm = np.linalg.norm(u_shift, axis=1)
        keep = (u_norm > 0.04) & (du_norm < 0.8)
        if np.sum(keep) >= 5:
            u_rows.append(u_shift[keep])
            y_rows.append(y_mat[keep])

    if not u_rows:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "insufficient_data", {}

    U = np.vstack(u_rows)
    Y = np.vstack(y_rows)
    reg = 1e-4 * np.eye(3)
    # Solve M^T = (U^T U + reg)^-1 U^T Y  => Y ≈ U M^T => y ≈ M u.
    M_T = np.linalg.solve(U.T @ U + reg, U.T @ Y)
    M = M_T.T

    g_x = float(M[0, 0])
    g_y = float(M[1, 1])
    g_w = float(M[2, 2])
    off_diag_max = float(np.max(np.abs(M - np.diag(np.diag(M)))))

    if g_x > 0.4 and g_y > 0.4:
        lat_delay_diff = abs(delays["vy"] - delays["vx"])
        kinematic_type = (
            "swerve_with_steering_lag" if lat_delay_diff >= 0.12 else "omnidirectional_4wheel"
        )
    elif g_x > 0.4 and g_y < 0.25:
        kinematic_type = "nonholonomic_differential"
    else:
        kinematic_type = "uncertain_verify_override"

    diagnostics = {
        "gain_vx": round(g_x, 4),
        "gain_vy": round(g_y, 4),
        "gain_wz": round(g_w, 4),
        "max_off_diagonal_coupling": round(off_diag_max, 4),
    }
    mat_list = [[round(float(M[r, c]), 5) for c in range(3)] for r in range(3)]
    return mat_list, kinematic_type, diagnostics


def analyze_racket_jitter(
    trials: Sequence[LoadedTrial],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Quantify racket normal tilt (deg) and position jitter (mm) vs base (|a|, |alpha|)."""
    idle_tilts_deg: List[np.ndarray] = []
    idle_pos_mm: List[np.ndarray] = []
    tier_tilts: Dict[str, List[np.ndarray]] = {"low": [], "med": [], "high": []}
    tier_pos: Dict[str, List[np.ndarray]] = {"low": [], "med": [], "high": []}

    feat_lin_acc: List[float] = []
    feat_ang_acc: List[float] = []
    target_tilt_deg: List[float] = []
    imu_used_any = False
    r_ref_mocap = _shared_mocap_ref_rotmat(trials)

    for tr in trials:
        if tr.racket_normal is None or tr.racket_pos_m is None:
            continue
        rn = np.asarray(tr.racket_normal, dtype=np.float64)
        rp = np.asarray(tr.racket_pos_m, dtype=np.float64)
        if rn.ndim != 2 or rn.shape[1] != 3 or rp.shape != rn.shape:
            continue
        valid = np.all(np.isfinite(rn), axis=1) & np.all(np.isfinite(rp), axis=1)
        if np.sum(valid) < 10:
            continue

        t = tr.t_s[valid]
        rn_v = rn[valid]
        rp_v = rp[valid]
        norms = np.linalg.norm(rn_v, axis=1, keepdims=True)
        rn_v = rn_v / np.maximum(norms, 1e-12)

        # Reference idle posture from initial 0.4 s (or first 10 samples).
        n_ref = max(5, int(np.sum((t - t[0]) <= 0.4)))
        n0 = np.mean(rn_v[:n_ref], axis=0)
        n0 /= max(float(np.linalg.norm(n0)), 1e-12)
        p0 = np.mean(rp_v[:n_ref], axis=0)

        dots = np.clip(rn_v @ n0, -1.0, 1.0)
        tilt_arm_deg = np.rad2deg(np.arccos(dots))
        pos_mm = 1000.0 * np.linalg.norm(rp_v - p0[None, :], axis=1)

        # Fold chassis roll/pitch tilt into total normal tilt (prefer IMU, fallback to mocap rigid body).
        tilt_total_deg = tilt_arm_deg
        chassis_rpy_aligned: Optional[np.ndarray] = None
        if tr.imu_rpy_rad is not None and tr.imu_rpy_rad.shape == rn.shape:
            imu_v = tr.imu_rpy_rad[valid]
            if np.all(np.isfinite(imu_v[:, :2])):
                imu_used_any = True
                chassis_rpy_aligned = imu_v
        if (
            chassis_rpy_aligned is None
            and tr.mocap_raw_wall_s is not None
            and len(tr.mocap_raw_wall_s) >= 10
        ):
            rpy_m = _trial_leveled_rpy(tr, r_ref_mocap)
            if rpy_m is not None and rpy_m.ndim == 2 and rpy_m.shape[1] >= 2:
                good_m = np.all(np.isfinite(rpy_m[:, :2]), axis=1)
                if np.sum(good_m) >= 10:
                    tm = tr.mocap_raw_wall_s[good_m] - tr.mocap_raw_wall_s[0]
                    roll_i = np.interp(t, tm, rpy_m[good_m, 0])
                    pitch_i = np.interp(t, tm, rpy_m[good_m, 1])
                    chassis_rpy_aligned = np.column_stack([roll_i, pitch_i])
                    imu_used_any = True

        if chassis_rpy_aligned is not None:
            r0 = float(np.mean(chassis_rpy_aligned[:n_ref, 0]))
            p_c0 = float(np.mean(chassis_rpy_aligned[:n_ref, 1]))
            chassis_tilt_deg = np.rad2deg(
                np.hypot(chassis_rpy_aligned[:, 0] - r0, chassis_rpy_aligned[:, 1] - p_c0)
            )
            tilt_total_deg = np.hypot(tilt_arm_deg, chassis_tilt_deg)

        if tr.stage == "s0_idle_noise":
            idle_tilts_deg.append(tilt_total_deg)
            idle_pos_mm.append(pos_mm)
            continue

        # Also harvest pre-idle baseline if no s0_idle_noise was passed.
        if not idle_tilts_deg:
            idle_tilts_deg.append(tilt_total_deg[:n_ref])
            idle_pos_mm.append(pos_mm[:n_ref])

        if tr.tier in tier_tilts:
            tier_tilts[tr.tier].append(tilt_total_deg)
            tier_pos[tr.tier].append(pos_mm)

        # Windowed (0.25 s) regression of tilt RMS against base |a| and |alpha|.
        vx_v = tr.meas["vx"][valid]
        vy_v = tr.meas["vy"][valid]
        wz_v = tr.meas["wz"][valid]
        ax = np.gradient(vx_v, t)
        ay = np.gradient(vy_v, t)
        aw = np.gradient(wz_v, t)
        a_lin = np.hypot(ax, ay)
        a_ang = np.abs(aw)

        dt = float(np.median(np.diff(t)))
        win = max(5, int(round(0.25 / max(dt, 1e-3))))
        for start in range(0, len(t) - win + 1, max(1, win // 2)):
            sl = slice(start, start + win)
            feat_lin_acc.append(float(np.percentile(a_lin[sl], 90)))
            feat_ang_acc.append(float(np.percentile(a_ang[sl], 90)))
            target_tilt_deg.append(float(np.sqrt(np.mean(tilt_total_deg[sl] ** 2))))

    idle_rms = (
        float(np.sqrt(np.mean(np.concatenate(idle_tilts_deg) ** 2)))
        if idle_tilts_deg else 0.0
    )
    idle_pos_rms = (
        float(np.sqrt(np.mean(np.concatenate(idle_pos_mm) ** 2)))
        if idle_pos_mm else 0.0
    )

    c_lin = 0.0
    c_ang = 0.0
    if len(target_tilt_deg) >= 4:
        X = np.column_stack([
            np.asarray(feat_lin_acc, dtype=np.float64),
            np.asarray(feat_ang_acc, dtype=np.float64),
        ])
        y_exc = np.maximum(0.0, np.asarray(target_tilt_deg, dtype=np.float64) - idle_rms)
        coef, _ = nnls(X, y_exc)
        c_lin = float(coef[0])
        c_ang = float(coef[1])

    norm_p95_by_tier: Dict[str, float] = {}
    norm_rms_by_tier: Dict[str, float] = {}
    norm_max_by_tier: Dict[str, float] = {}
    pos_p95_by_tier: Dict[str, float] = {}
    for tier_name in ("low", "med", "high"):
        if tier_tilts[tier_name]:
            arr_t = np.concatenate(tier_tilts[tier_name])
            arr_p = np.concatenate(tier_pos[tier_name])
            norm_p95_by_tier[tier_name] = round(float(np.percentile(arr_t, 95)), 4)
            norm_rms_by_tier[tier_name] = round(float(np.sqrt(np.mean(arr_t * arr_t))), 4)
            norm_max_by_tier[tier_name] = round(float(np.max(arr_t)), 4)
            pos_p95_by_tier[tier_name] = round(float(np.percentile(arr_p, 95)), 4)
        else:
            norm_p95_by_tier[tier_name] = 0.0
            norm_rms_by_tier[tier_name] = 0.0
            norm_max_by_tier[tier_name] = 0.0
            pos_p95_by_tier[tier_name] = 0.0

    model = {
        "imu_included": imu_used_any,
        "normal_deg_rms_idle": round(idle_rms, 5),
        "pos_mm_rms_idle": round(idle_pos_rms, 4),
        "normal_deg_per_mps2": round(c_lin, 5),
        "normal_deg_per_radps2": round(c_ang, 5),
        "normal_deg_p95_by_tier": norm_p95_by_tier,
        "normal_deg_rms_by_tier": norm_rms_by_tier,
        "normal_deg_max_by_tier": norm_max_by_tier,
        "pos_mm_p95_by_tier": pos_p95_by_tier,
    }
    scatter_data = {
        "a_lin_mps2": np.asarray(feat_lin_acc, dtype=np.float64),
        "a_ang_radps2": np.asarray(feat_ang_acc, dtype=np.float64),
        "tilt_rms_deg": np.asarray(target_tilt_deg, dtype=np.float64),
    }
    return model, scatter_data


def analyze_arm_base_concurrency(
    trials: Sequence[LoadedTrial],
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Compute arm (200 Hz) & base publish period P50/P95/P99, drop/ZOH rate, and blocking."""
    arm_dts_ms: List[np.ndarray] = []
    base_dts_ms: List[np.ndarray] = []
    zoh_total = 0
    zoh_dups = 0

    s5_trials = [tr for tr in trials if tr.stage == "s5_arm_concurrency"]
    target_trials = s5_trials if s5_trials else list(trials)

    for tr in target_trials:
        if len(tr.t_s) >= 2:
            dt_b = np.diff(tr.t_s) * 1000.0
            base_dts_ms.append(dt_b[dt_b > 0.0])
        if tr.arm_cmd_wall_s is not None and len(tr.arm_cmd_wall_s) >= 2:
            dt_a = np.diff(tr.arm_cmd_wall_s) * 1000.0
            arm_dts_ms.append(dt_a[dt_a > 0.0])
        elif tr.arm_raw_wall_s is not None and len(tr.arm_raw_wall_s) >= 2:
            dt_a = np.diff(tr.arm_raw_wall_s) * 1000.0
            arm_dts_ms.append(dt_a[dt_a > 0.0])

        # ZOH / duplicate feedback check on base measured twist during motion.
        active = np.hypot(tr.cmd["vx"], tr.cmd["vy"]) + np.abs(tr.cmd["wz"]) > 0.05
        if np.sum(active) >= 10:
            v_act = np.column_stack([tr.meas["vx"][active], tr.meas["vy"][active], tr.meas["wz"][active]])
            diffs = np.linalg.norm(np.diff(v_act, axis=0), axis=1)
            zoh_total += int(len(diffs))
            zoh_dups += int(np.sum(diffs < 1e-9))

    arm_all = np.concatenate(arm_dts_ms) if arm_dts_ms else np.empty(0, dtype=np.float64)
    base_all = np.concatenate(base_dts_ms) if base_dts_ms else np.empty(0, dtype=np.float64)

    def _pct(arr: np.ndarray, q: float) -> float:
        return round(float(np.percentile(arr, q)), 4) if len(arr) > 0 else 0.0

    arm_p50 = _pct(arm_all, 50)
    arm_p95 = _pct(arm_all, 95)
    arm_p99 = _pct(arm_all, 99)
    base_p50 = _pct(base_all, 50)
    base_p95 = _pct(base_all, 95)
    base_p99 = _pct(base_all, 99)

    arm_drop = float(np.mean(arm_all > max(1.5 * arm_p50, 7.5))) if len(arm_all) > 0 and arm_p50 > 0 else 0.0
    base_drop = float(np.mean(base_all > max(1.5 * base_p50, 30.0))) if len(base_all) > 0 and base_p50 > 0 else 0.0
    zoh_ratio = float(zoh_dups / zoh_total) if zoh_total > 0 else 0.0
    drop_rate = max(arm_drop, base_drop)
    blocking = bool(
        (arm_p50 > 0.0 and arm_p99 > max(2.5 * arm_p50, 12.5))
        or (base_p50 > 0.0 and base_p99 > max(2.5 * base_p50, 50.0))
        or drop_rate > 0.02
    )

    summary = {
        "has_s5_concurrency_trials": bool(s5_trials),
        "arm_dt_p50_ms": arm_p50,
        "arm_dt_p95_ms": arm_p95,
        "arm_dt_p99_ms": arm_p99,
        "base_dt_p50_ms": base_p50,
        "base_dt_p95_ms": base_p95,
        "base_dt_p99_ms": base_p99,
        "drop_rate": round(drop_rate, 5),
        "zoh_hold_ratio": round(zoh_ratio, 5),
        "blocking_detected": blocking,
    }
    return summary, {"arm_dt_ms": arm_all, "base_dt_ms": base_all}


def _quat_xyzw_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _rotmat_to_rpy_rad(R: np.ndarray) -> Tuple[float, float, float]:
    sy = max(-1.0, min(1.0, -float(R[2, 0])))
    pitch = math.asin(sy)
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return (roll, pitch, yaw)


def _shared_mocap_ref_rotmat(trials: Sequence[LoadedTrial]) -> Optional[np.ndarray]:
    """Return one shared reference rotation ``R_ref`` from the first valid quaternion.

    Right-multiplying every sample's robot-frame orientation by ``R_ref^T`` cancels any
    arbitrary fixed mounting angle of a K-shaped rigid body (even 90-deg vertical on a
    chassis side wall), eliminating Euler gimbal lock consistently across all trials/runs.
    """
    for tr in trials:
        if tr.mocap_quat_xyzw is None:
            continue
        q_arr = np.asarray(tr.mocap_quat_xyzw, dtype=np.float64)
        if q_arr.ndim != 2 or q_arr.shape[1] != 4:
            continue
        for row in q_arr:
            if np.all(np.isfinite(row)) and float(np.linalg.norm(row)) > 1e-6:
                return _quat_xyzw_to_rotmat(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    return None


def _trial_leveled_rpy(
    tr: LoadedTrial,
    r_ref: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return gimbal-lock-free ``(N, 3)`` ``[roll, pitch, yaw]`` for a trial's mocap rigid body."""
    if tr.mocap_quat_xyzw is not None and r_ref is not None:
        q_arr = np.asarray(tr.mocap_quat_xyzw, dtype=np.float64)
        if q_arr.ndim == 2 and q_arr.shape[1] == 4:
            out = np.full((len(q_arr), 3), np.nan, dtype=np.float64)
            r_ref_t = r_ref.T
            for idx, row in enumerate(q_arr):
                if np.all(np.isfinite(row)) and float(np.linalg.norm(row)) > 1e-6:
                    r_cur = _quat_xyzw_to_rotmat(float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                    out[idx] = _rotmat_to_rpy_rad(r_cur @ r_ref_t)
            return out
    if tr.mocap_rpy_rad is not None:
        arr = np.asarray(tr.mocap_rpy_rad, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr
    if tr.mocap_yaw_rad is not None:
        yaw = np.asarray(tr.mocap_yaw_rad, dtype=np.float64).reshape(-1)
        out = np.full((len(yaw), 3), np.nan, dtype=np.float64)
        out[:, 2] = yaw
        return out
    return None


def estimate_rigid_body_lever_arm(
    trials: Sequence[LoadedTrial],
) -> Dict[str, Any]:
    """Solve 2D rigid-body lever arm ``r_b = [rx, ry]`` from pure-yaw (``wz``) trials.

    For a rigid body mounted at unknown offset ``r_b`` from the chassis rotation center:
        p_rb(t) = c_k + R_2(yaw(t)) @ r_b
    Subtracting each trial's mean removes the per-trial center ``c_k``, leaving an
    exact linear least-squares problem for ``r_b`` with sub-millimeter precision.
    """
    A_rows: List[np.ndarray] = []
    b_rows: List[np.ndarray] = []
    n_trials_used = 0
    r_ref = _shared_mocap_ref_rotmat(trials)

    for tr in trials:
        if tr.primary_axis != "wz":
            continue
        if (
            tr.mocap_raw_wall_s is None
            or tr.mocap_pos_xyz_m is None
            or len(tr.mocap_raw_wall_s) < 15
        ):
            continue
        rpy_m = _trial_leveled_rpy(tr, r_ref)
        if rpy_m is None:
            continue
        yaw_raw = rpy_m[:, 2]
        pos_xy = np.asarray(tr.mocap_pos_xyz_m[:, :2], dtype=np.float64)
        valid = np.isfinite(yaw_raw) & np.all(np.isfinite(pos_xy), axis=1)
        if np.sum(valid) < 15:
            continue
        yaw = np.unwrap(yaw_raw[valid])
        if float(np.ptp(yaw)) < 0.15:
            continue
        p = pos_xy[valid]
        c = np.cos(yaw)
        s = np.sin(yaw)
        # R_2(yaw) = [[c, -s], [s, c]] -> subtract trial mean to eliminate center c_k.
        c_dm = c - np.mean(c)
        s_dm = s - np.mean(s)
        p_dm = p - np.mean(p, axis=0, keepdims=True)
        # Stack [x_dm; y_dm] = [[c_dm, -s_dm]; [s_dm, c_dm]] @ [rx; ry]
        A_x = np.column_stack([c_dm, -s_dm])
        A_y = np.column_stack([s_dm, c_dm])
        A_rows.append(np.vstack([A_x, A_y]))
        b_rows.append(np.concatenate([p_dm[:, 0], p_dm[:, 1]]))
        n_trials_used += 1

    if not A_rows:
        return {
            "calibrated": False,
            "n_wz_trials_used": 0,
            "lever_arm_xy_m": [0.0, 0.0],
            "lever_arm_radius_m": 0.0,
            "fit_residual_mm": 0.0,
        }

    A = np.vstack(A_rows)
    b = np.concatenate(b_rows)
    r_hat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ r_hat - b
    rmse_mm = float(np.sqrt(np.mean(resid * resid)) * 1000.0)
    rx, ry = float(r_hat[0]), float(r_hat[1])
    return {
        "calibrated": True,
        "n_wz_trials_used": n_trials_used,
        "lever_arm_xy_m": [round(rx, 5), round(ry, 5)],
        "lever_arm_radius_m": round(float(math.hypot(rx, ry)), 5),
        "fit_residual_mm": round(rmse_mm, 3),
    }


def recover_chassis_center_xy(
    pos_xy_m: np.ndarray,
    yaw_rad: np.ndarray,
    lever_arm_xy_m: Sequence[float],
) -> np.ndarray:
    """Subtract ``R_2(yaw) @ lever_arm_xy_m`` from rigid-body XY to recover chassis center."""
    p = np.asarray(pos_xy_m, dtype=np.float64).copy()
    rx, ry = float(lever_arm_xy_m[0]), float(lever_arm_xy_m[1])
    if abs(rx) < 1e-12 and abs(ry) < 1e-12:
        return p
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    p[:, 0] -= (c * rx - s * ry)
    p[:, 1] -= (s * rx + c * ry)
    return p


def analyze_odom_vs_mocap(trials: Sequence[LoadedTrial]) -> Dict[str, Any]:
    """Compare internal wheel odometry against external motion capture when present.

    Supports both off-center 4-ball rigid bodies (with gimbal-lock-free orientation
    leveling + automatic closed-form lever-arm calibration from pure-yaw trials -> exact
    chassis-center position, linear velocity, and yaw rate) and single markers.
    """
    lever_info = estimate_rigid_body_lever_arm(trials)
    lever_xy = lever_info["lever_arm_xy_m"] if lever_info["calibrated"] else [0.0, 0.0]
    r_ref = _shared_mocap_ref_rotmat(trials)

    scale_ratios: List[float] = []
    yaw_scale_ratios: List[float] = []
    drift_rates: List[float] = []
    rel_delays: List[float] = []
    has_mocap_yaw = False

    for tr in trials:
        if tr.mocap_raw_wall_s is None or tr.mocap_pos_xyz_m is None or len(tr.mocap_raw_wall_s) < 15:
            continue
        t_raw = tr.mocap_raw_wall_s - tr.mocap_raw_wall_s[0]
        pos_raw = np.asarray(tr.mocap_pos_xyz_m, dtype=np.float64)
        rpy_m = _trial_leveled_rpy(tr, r_ref)
        yaw_raw = None if rpy_m is None else rpy_m[:, 2]

        valid = np.isfinite(t_raw) & np.all(np.isfinite(pos_raw[:, :2]), axis=1)
        trial_has_yaw = bool(yaw_raw is not None and np.sum(valid & np.isfinite(yaw_raw)) >= 15)
        if trial_has_yaw:
            assert yaw_raw is not None
            valid = valid & np.isfinite(yaw_raw)
        if np.sum(valid) < 15:
            continue

        t_m = t_raw[valid]
        pos_m = pos_raw[valid]
        order = np.argsort(t_m, kind="mergesort")
        t_m = t_m[order]
        pos_m = pos_m[order]
        if float(t_m[-1] - t_m[0]) < 0.4:
            continue

        # Resample onto a uniform 50 Hz grid so UDP micro-bursts (dt ~ 0.15 ms) do not
        # amplify position/angle noise during differentiation, while masking >0.15 s gaps.
        dt_u = 0.02
        t_u = np.arange(float(t_m[0]), float(t_m[-1]) + 1e-9, dt_u, dtype=np.float64)
        if len(t_u) < 15:
            continue

        dts_raw = np.diff(t_m)
        gap_ok_u = np.ones(len(t_u), dtype=bool)
        for gap_idx in np.where(dts_raw > 0.15)[0]:
            gap_ok_u &= ~((t_u >= t_m[gap_idx] - 0.05) & (t_u <= t_m[gap_idx + 1] + 0.05))

        def _centered_deriv(y_u: np.ndarray, dt_step: float, half_win: int = 2) -> np.ndarray:
            d = np.gradient(y_u, dt_step)
            if len(y_u) > 2 * half_win:
                d[half_win:-half_win] = (y_u[2 * half_win :] - y_u[: -2 * half_win]) / (2.0 * half_win * dt_step)
            return d

        if trial_has_yaw:
            has_mocap_yaw = True
            assert yaw_raw is not None
            yaw_unwrapped_raw = np.unwrap(yaw_raw[valid][order])
            xy_center_raw = recover_chassis_center_xy(pos_m[:, :2], yaw_unwrapped_raw, lever_xy)
            yaw_u = np.interp(t_u, t_m, yaw_unwrapped_raw)
            xy_center = np.column_stack([
                np.interp(t_u, t_m, xy_center_raw[:, 0]),
                np.interp(t_u, t_m, xy_center_raw[:, 1]),
            ])
            wz_m = _centered_deriv(yaw_u, dt_u, half_win=2)
            wz_o_signed = np.interp(t_u, tr.t_s, tr.meas["wz"])
            rotating = (np.abs(wz_m) > 0.12) & (np.abs(wz_o_signed) > 0.12) & gap_ok_u
            if np.sum(rotating) >= 10:
                yaw_scale_ratios.append(
                    float(np.median(np.abs(wz_o_signed[rotating]) / np.maximum(np.abs(wz_m[rotating]), 1e-6)))
                )
        else:
            xy_center = np.column_stack([
                np.interp(t_u, t_m, pos_m[:, 0]),
                np.interp(t_u, t_m, pos_m[:, 1]),
            ])

        vx_m = _centered_deriv(xy_center[:, 0], dt_u, half_win=2)
        vy_m = _centered_deriv(xy_center[:, 1], dt_u, half_win=2)
        spd_m = np.hypot(vx_m, vy_m)

        # Interpolate odom speed & yaw rate onto uniform mocap timestamps.
        spd_o = np.hypot(
            np.interp(t_u, tr.t_s, tr.meas["vx"]),
            np.interp(t_u, tr.t_s, tr.meas["vy"]),
        )
        wz_o = np.abs(np.interp(t_u, tr.t_s, tr.meas["wz"]))
        # Only restrict to pure translation when lever-arm compensation is unavailable.
        compensated = trial_has_yaw and lever_info["calibrated"]
        pure_trans = np.ones_like(spd_m, dtype=bool) if compensated else (wz_o < 0.05)
        moving = (spd_m > 0.08) & (spd_o > 0.08) & pure_trans & gap_ok_u
        if np.sum(moving) >= 10:
            ratio = float(np.median(spd_o[moving] / np.maximum(spd_m[moving], 1e-6)))
            scale_ratios.append(ratio)

        # End-to-end chassis-center drift per meter traveled.
        if compensated or float(np.max(wz_o)) < 0.08:
            ox_finite = np.isfinite(tr.odom_x_m) & np.isfinite(tr.odom_y_m)
            if np.sum(ox_finite) >= 10:
                ox_v = tr.odom_x_m[ox_finite]
                oy_v = tr.odom_y_m[ox_finite]
                ts_v = tr.t_s[ox_finite]
                odom_xy = np.column_stack([
                    np.interp(t_u, ts_v, ox_v - ox_v[0]),
                    np.interp(t_u, ts_v, oy_v - oy_v[0]),
                ])
                mocap_xy = xy_center - xy_center[0:1, :]
                path_len = float(np.sum(np.linalg.norm(np.diff(mocap_xy, axis=0), axis=1)))
                if path_len > 0.15 and np.all(np.isfinite(odom_xy[-1])) and np.all(np.isfinite(mocap_xy[-1])):
                    pos_err = float(np.linalg.norm(odom_xy[-1] - mocap_xy[-1]))
                    drift_rates.append(pos_err / path_len)

        # Relative lag via cross-correlation of speed magnitude.
        if (compensated or float(np.max(wz_o)) < 0.08) and np.std(spd_o) > 1e-4 and np.std(spd_m) > 1e-4:
            max_lag = min(len(spd_m) // 4, int(round(0.25 / dt_u)))
            best_lag = 0
            best_c = -1e18
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    a, b = spd_o[: len(spd_o) - lag], spd_m[lag:]
                else:
                    a, b = spd_o[-lag:], spd_m[: len(spd_m) + lag]
                c = float(np.dot(a, b))
                if c > best_c:
                    best_c = c
                    best_lag = lag
            rel_delays.append(best_lag * dt_u)

    if not scale_ratios and not drift_rates and not yaw_scale_ratios:
        return {
            "available": False,
            "mocap_mode": "none",
            "velocity_scale_error_frac": 0.0,
            "yaw_rate_scale_error_frac": 0.0,
            "position_drift_m_per_m": 0.0,
            "relative_delay_mocap_minus_odom_s": 0.0,
            "lever_arm_calibration": lever_info,
            "recommended_primary_source": "mx_robot_state_odom_twist",
        }

    scale_err = abs(float(np.median(scale_ratios)) - 1.0) if scale_ratios else 0.0
    yaw_scale_err = abs(float(np.median(yaw_scale_ratios)) - 1.0) if yaw_scale_ratios else 0.0
    drift = float(np.median(drift_rates)) if drift_rates else 0.0
    rel_d = float(np.median(rel_delays)) if rel_delays else 0.0
    rec = (
        "mocap_preferred_for_position_odom_for_velocity"
        if drift > 0.05 or scale_err > 0.05
        else "mx_robot_state_odom_verified_against_mocap"
    )
    return {
        "available": True,
        "mocap_mode": "rigid_body_lever_arm_compensated" if (has_mocap_yaw and lever_info["calibrated"]) else (
            "rigid_body" if has_mocap_yaw else "single_marker_translation_only"
        ),
        "velocity_scale_error_frac": round(scale_err, 5),
        "yaw_rate_scale_error_frac": round(yaw_scale_err, 5),
        "position_drift_m_per_m": round(drift, 5),
        "relative_delay_mocap_minus_odom_s": round(rel_d, 5),
        "lever_arm_calibration": lever_info,
        "recommended_primary_source": rec,
    }


def generate_summary_figure(
    *,
    plot_bundles: Mapping[str, Optional[Dict[str, Any]]],
    jitter_model: Mapping[str, Any],
    jitter_scatter: Mapping[str, np.ndarray],
    concurrency_summary: Mapping[str, Any],
    concurrency_dts: Mapping[str, np.ndarray],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Generate 4-panel publication-grade summary figure (.pdf and .png)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=200)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: vx / vy / wz step measured vs FOPDT+asymmetric model fit.
    colors = {"vx": "#1f77b4", "vy": "#2ca02c", "wz": "#d62728"}
    labels_axis = {"vx": "vx (m/s)", "vy": "vy (m/s)", "wz": "wz (rad/s)"}
    any_a = False
    for axis in AXES:
        bundle = plot_bundles.get(axis)
        if bundle is None:
            continue
        any_a = True
        t = bundle["t_s"]
        ax_a.plot(t, bundle["cmd"], linestyle=":", color=colors[axis], alpha=0.55, lw=1.0)
        ax_a.plot(
            t, bundle["meas"], color=colors[axis], lw=1.4, alpha=0.85,
            label=f"{labels_axis[axis]} meas",
        )
        ax_a.plot(
            t, bundle["pred"], linestyle="--", color=colors[axis], lw=1.6,
            label=f"{axis} fit (RMSE={bundle['heldout_rmse']:.3f})",
        )
    ax_a.set_title("A. Step Response: Measured vs FOPDT+Asymmetric Model")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Velocity (m/s or rad/s)")
    ax_a.grid(True, alpha=0.3)
    if any_a:
        ax_a.legend(fontsize=7.5, ncol=2, loc="best")
    else:
        ax_a.text(0.5, 0.5, "No step excitation trials", ha="center", va="center", transform=ax_a.transAxes)

    # Panel B: Zoom-in on acceleration & emergency-braking edge with delay annotation.
    zoom_bundle = plot_bundles.get("vx") or plot_bundles.get("vy") or plot_bundles.get("wz")
    if zoom_bundle is not None:
        t = zoom_bundle["t_s"]
        u = zoom_bundle["cmd"]
        y = zoom_bundle["meas"]
        yhat = zoom_bundle["pred"]
        fit = zoom_bundle["fitted"]
        ax_b.plot(t, u, ":", color="#444444", lw=1.3, label="Command u(t)")
        ax_b.plot(t, y, color="#1f77b4", lw=1.6, label="Measured v(t)")
        ax_b.plot(t, yhat, "--", color="#ff7f0e", lw=1.6, label="Model fit")
        # Focus on the first positive step + brake window.
        up_idx = np.where(np.abs(u) > 0.5 * max(float(np.max(np.abs(u))), 1e-6))[0]
        if len(up_idx) > 0:
            t_start = max(float(t[0]), float(t[up_idx[0]]) - 0.25)
            # Find end of first contiguous positive pulse.
            gaps = np.where(np.diff(up_idx) > 2)[0]
            end_i = int(up_idx[gaps[0]]) if len(gaps) > 0 else int(up_idx[-1])
            t_end = min(float(t[-1]), float(t[end_i]) + 0.85)
            ax_b.set_xlim(t_start, t_end)
            t_edge = float(t[up_idx[0]])
            ax_b.axvline(t_edge, color="#666666", linestyle="-.", lw=0.9)
            ax_b.axvline(t_edge + fit["delay_s"], color="#d62728", linestyle="-.", lw=0.9)
            ax_b.annotate(
                f"tau={fit['delay_s']*1000:.1f} ms\n"
                f"T={fit['time_constant_s']*1000:.1f} ms\n"
                f"a_acc={fit['a_acc_max']:.2f}, a_brk={fit['a_brake_max']:.2f}",
                xy=(t_edge + fit["delay_s"], 0.1 * float(np.max(np.abs(u)))),
                xytext=(0.56, 0.16),
                textcoords="axes fraction",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9),
            )
        ax_b.legend(fontsize=8, loc="upper right")
    else:
        ax_b.text(0.5, 0.5, "No step/brake data", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_title("B. Acceleration, Pure Delay & Emergency Braking Zoom")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Velocity (m/s or rad/s)")
    ax_b.grid(True, alpha=0.3)

    # Panel C: Base acceleration vs racket normal tilt jitter.
    a_lin = jitter_scatter.get("a_lin_mps2", np.empty(0))
    tilt = jitter_scatter.get("tilt_rms_deg", np.empty(0))
    if len(a_lin) > 0:
        ax_c.scatter(a_lin, tilt, s=16, alpha=0.55, color="#9467bd", label="0.25 s windows")
        a_grid = np.linspace(0.0, max(float(np.max(a_lin)) * 1.05, 0.5), 100)
        y_fit = jitter_model["normal_deg_rms_idle"] + jitter_model["normal_deg_per_mps2"] * a_grid
        ax_c.plot(a_grid, y_fit, color="#d62728", lw=1.8, label="Linear fit (|a|)")
        # Overlay tier P95 horizontal markers.
        p95_map = jitter_model.get("normal_deg_p95_by_tier") or {}
        p95_txt = ", ".join(f"{k}:{v:.2f} deg" for k, v in p95_map.items() if v > 0)
        if p95_txt:
            ax_c.text(
                0.03, 0.92, f"P95 tilt -> {p95_txt}",
                transform=ax_c.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85),
            )
        ax_c.legend(fontsize=8, loc="lower right")
    else:
        ax_c.text(0.5, 0.5, "No racket FK jitter data", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.set_title("C. Base Acceleration vs Racket Normal Tilt Jitter")
    ax_c.set_xlabel("Base Linear Acceleration |a| (m/s^2)")
    ax_c.set_ylabel("Racket Normal Tilt RMS (deg)")
    ax_c.grid(True, alpha=0.3)

    # Panel D: Arm & Base publish loop period distribution.
    arm_dt = concurrency_dts.get("arm_dt_ms", np.empty(0))
    base_dt = concurrency_dts.get("base_dt_ms", np.empty(0))
    if len(arm_dt) > 0:
        ax_d.hist(arm_dt, bins=35, alpha=0.65, color="#1f77b4", label=f"Arm dt (P95={concurrency_summary['arm_dt_p95_ms']:.2f} ms)")
    if len(base_dt) > 0:
        ax_d.hist(base_dt, bins=35, alpha=0.55, color="#ff7f0e", label=f"Base dt (P95={concurrency_summary['base_dt_p95_ms']:.2f} ms)")
    if len(arm_dt) > 0 or len(base_dt) > 0:
        ax_d.legend(fontsize=8, loc="best")
        ax_d.text(
            0.03, 0.85,
            f"drop_rate={concurrency_summary['drop_rate']:.4f}\n"
            f"blocking={concurrency_summary['blocking_detected']}",
            transform=ax_d.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85),
        )
    else:
        ax_d.text(0.5, 0.5, "No timing samples", ha="center", va="center", transform=ax_d.transAxes)
    ax_d.set_title("D. Arm (200 Hz) & Base Control Period Distribution")
    ax_d.set_xlabel("Loop Period dt (ms)")
    ax_d.set_ylabel("Count")
    ax_d.grid(True, alpha=0.3)

    fig.tight_layout()
    pdf_path = out_dir / "fig_moz1_base_id_summary.pdf"
    png_path = out_dir / "fig_moz1_base_id_summary.png"
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=200)
    plt.close(fig)
    return pdf_path, png_path


def render_chinese_report(
    params: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> str:
    """Render concise Chinese technical report ``BASE_ID_REPORT_CN.md``."""
    lines: List[str] = [
        "# MOZ1 实机底盘动力学辨识与拍面抖动标定报告",
        "",
        f"- 生成时间 (UTC)：`{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- 参数合同版本：`{params['schema_version']}`",
        f"- 轮型与运动学结论：**`{params['kinematic_type']}`**",
        f"- 数据审计：共扫描 `{audit['n_run_dirs']}` 个采集目录，接受 `{audit['n_accepted_trials']}` 条试次，剔除 `{audit['n_rejected_trials']}` 条试次（详见 `DATA_AUDIT.json`）。",
        "",
        "## 1. 三轴辨识动力学参数（SI 单位）",
        "",
        "| 轴 | 纯延迟 $\\tau$ (s) | 互相关延迟 (s) | 10%起动延迟 (s) | 时间常数 $T$ (s) | 稳态增益 $K$ | 实测速度上限 | 加速上限 $a_{\\text{acc}}$ | 制动上限 $a_{\\text{brake}}$ | 过冲比 | 制动滑行 | 留出集 RMSE | 留出集 $R^2$ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ax in AXES:
        p = params["axes"][ax]
        unit_v = "m/s" if ax != "wz" else "rad/s"
        unit_a = "m/s²" if ax != "wz" else "rad/s²"
        unit_d = "m" if ax != "wz" else "rad"
        lines.append(
            f"| `{ax}` | {p['delay_s']:.4f} | {p['delay_xcorr_s']:.4f} | {p['delay_onset10_s']:.4f} | "
            f"{p['time_constant_s']:.4f} | {p['gain']:.3f} | {p['v_max']:.3f} {unit_v} | "
            f"{p['a_acc_max']:.3f} {unit_a} | {p['a_brake_max']:.3f} {unit_a} | "
            f"{p['overshoot_frac']*100:.1f}% | {p['brake_distance']:.3f} {unit_d} | "
            f"{p['heldout_rmse']:.4f} | {p['heldout_r2']:.4f} |"
        )

    M = params["cross_coupling_matrix_3x3"]
    lines.extend([
        "",
        "## 2. 3×3 轴间串扰增益矩阵 (`meas = M @ cmd`)",
        "",
        "| 实测 \\ 指令 | `cmd_vx` | `cmd_vy` | `cmd_wz` |",
        "|---|---:|---:|---:|",
        f"| `meas_vx` | {M[0][0]:.4f} | {M[0][1]:.4f} | {M[0][2]:.4f} |",
        f"| `meas_vy` | {M[1][0]:.4f} | {M[1][1]:.4f} | {M[1][2]:.4f} |",
        f"| `meas_wz` | {M[2][0]:.4f} | {M[2][1]:.4f} | {M[2][2]:.4f} |",
        "",
        "## 3. 拍面法向倾角与位置抖动耦合",
        "",
    ])
    rj = params["racket_jitter_model"]
    lines.extend([
        f"- 静止本底法向抖动 RMS：`{rj['normal_deg_rms_idle']:.4f} deg`（位置 RMS：`{rj['pos_mm_rms_idle']:.3f} mm`）",
        f"- 经验线性斜率：$\\sigma_\\theta(|a|, |\\alpha|) = {rj['normal_deg_rms_idle']:.4f} + "
        f"{rj['normal_deg_per_mps2']:.4f}\\,|a| + {rj['normal_deg_per_radps2']:.4f}\\,|\\alpha|$ (deg)",
        f"- 是否叠加底盘 IMU 姿态：`{rj['imu_included']}`",
        f"- 各档位法向抖动 P95 (deg)：`{rj['normal_deg_p95_by_tier']}`",
        f"- 各档位拍面位置抖动 P95 (mm)：`{rj['pos_mm_p95_by_tier']}`",
        "",
        "## 4. 200 Hz 手臂与底盘并发同步性",
        "",
    ])
    cc = params["arm_base_concurrency"]
    lines.extend([
        f"- 手臂控制周期 P50 / P95 / P99：`{cc['arm_dt_p50_ms']:.3f} / {cc['arm_dt_p95_ms']:.3f} / {cc['arm_dt_p99_ms']:.3f} ms`",
        f"- 底盘控制周期 P50 / P95 / P99：`{cc['base_dt_p50_ms']:.3f} / {cc['base_dt_p95_ms']:.3f} / {cc['base_dt_p99_ms']:.3f} ms`",
        f"- 丢帧比例：`{cc['drop_rate']:.5f}`，ZOH 重复保持比例：`{cc['zoh_hold_ratio']:.5f}`，是否检测到阻塞：**`{cc['blocking_detected']}`**",
        "",
        "## 5. 时钟偏置审计与里程计 vs 动捕比对",
        "",
    ])
    clk_audit = audit.get("clock_offset_audit_s") or {}
    if clk_audit:
        for src, info in clk_audit.items():
            lines.append(
                f"- `{src}`：`header_time - epoch_time` 中位数 = `{info['median_s']:.6f} s`"
                f"（大偏置已剥离：`{info['large_bias_separated']}`）"
            )
    else:
        lines.append("- 无有效 header 时钟偏置记录（全部使用单调墙钟对齐）。")

    om = params["odom_vs_mocap"]
    lines.extend([
        f"- 动捕比对可用：`{om['available']}`，速度标度误差：`{om['velocity_scale_error_frac']*100:.2f}%`，"
        f"位置漂移率：`{om['position_drift_m_per_m']*100:.2f}%`，相对延迟：`{om['relative_delay_mocap_minus_odom_s']*1000:.2f} ms`",
        f"- 推荐闭环底盘状态源：**`{om['recommended_primary_source']}`**",
        "",
        "## 6. MuJoCo / MJX 仿真底盘建模对接建议",
        "",
        "在 MuJoCo/MJX 中对 MOZ1 全向底盘（`base_x`, `base_y`, `base_yaw` 平面 3-DOF 或 6-DOF 浮动基座）建模时，可直接采用本报告辨识出的**延迟一阶 + 不对称加/制动饱和速度执行器**：",
        "",
        "1. **指令队列纯延迟**：按各轴 `delay_s` 维护环形缓冲区读取 $u(t - \\tau_i)$（例如控制步长 $\\Delta t$ 下延迟 $d_i = \\text{round}(\\tau_i / \\Delta t)$ 步，或线性分数步插值）。",
        "2. **稳态增益与串扰**：机体系参考速度 $v_{\\text{ref}} = M_{3\\times 3}\\,u(t - \\tau)$，并按各轴 `v_max` 限幅。",
        "3. **一阶动态 + 不对称加减速钳位**：$a_{\\text{des}} = (v_{\\text{ref}} - v)/T_i$；当 $v\\cdot a_{\\text{des}} < 0$（制动）时钳位至 `a_brake_max`，否则钳位至 `a_acc_max`，积分更新机体系速度后经 $\\text{SE}(2)$ 旋转映射到世界系 $(\\dot{x}, \\dot{y}, \\dot{\\psi})$。",
        "4. **底盘机动诱发拍面扰动**：在域随机化 / 鲁棒控制器中，按 `racket_jitter_model` 注入随 $(|a|, |\\alpha|)$ 线性增长的机身/拍面姿态扰动。",
        "",
        "## 7. 剔除试次与异常记录",
        "",
    ])
    rejected = audit.get("rejected_trials") or []
    if rejected:
        for row in rejected:
            lines.append(f"- `{row['run_dir']}/{row['trial_id']}`：`{row['reasons']}`")
    else:
        lines.append("- 本次全部试次通过数据质量审计，无剔除记录。")

    return "\n".join(lines) + "\n"


def run_offline_fit(
    run_dirs: Sequence[Path],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run the complete offline audit, system ID, JSON contract, report, and plot pipeline."""
    out_path = Path(out_dir)
    if out_path.exists():
        raise OutputDirExistsError(f"Refusing to overwrite existing output directory: {out_path}")
    out_path.mkdir(parents=True, exist_ok=False)

    accepted, audit_doc, source_hashes = load_and_audit_runs(run_dirs)
    (out_path / "DATA_AUDIT.json").write_text(
        json.dumps(audit_doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    axes_params: Dict[str, Any] = {}
    plot_bundles: Dict[str, Optional[Dict[str, Any]]] = {}
    for axis in AXES:
        ax_summary, bundle = fit_single_axis(accepted, axis)
        axes_params[axis] = ax_summary
        plot_bundles[axis] = bundle

    coupling_mat, kin_type, coupling_diag = estimate_cross_coupling_and_kinematics(
        accepted, axes_params
    )
    jitter_model, jitter_scatter = analyze_racket_jitter(accepted)
    concurrency_summary, concurrency_dts = analyze_arm_base_concurrency(accepted)
    odom_vs_mocap = analyze_odom_vs_mocap(accepted)

    params_doc: Dict[str, Any] = {
        "schema_version": PARAMS_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kinematic_type": kin_type,
        "axes": axes_params,
        "cross_coupling_matrix_3x3": coupling_mat,
        "cross_coupling_diagnostics": coupling_diag,
        "racket_jitter_model": jitter_model,
        "arm_base_concurrency": concurrency_summary,
        "odom_vs_mocap": odom_vs_mocap,
        "clock_offsets_s": audit_doc["clock_offset_audit_s"],
        "source_run_hashes": source_hashes,
        "simulation_modeling_hints": {
            "actuator_model": "fopdt_with_asymmetric_accel_brake_clamp",
            "frame": "body_frame_se2_twist_to_world",
            "omni_wheel_half_xy_m": [0.2423, 0.2423],
            "recommended_sim_dt_s": 0.002,
            "recommended_control_dt_s": 0.02,
        },
    }

    params_path = out_path / "moz1_base_identified_params_v1.json"
    params_path.write_text(
        json.dumps(params_doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    report_md = render_chinese_report(params_doc, audit_doc)
    (out_path / "BASE_ID_REPORT_CN.md").write_text(report_md, encoding="utf-8")

    generate_summary_figure(
        plot_bundles=plot_bundles,
        jitter_model=jitter_model,
        jitter_scatter=jitter_scatter,
        concurrency_summary=concurrency_summary,
        concurrency_dts=concurrency_dts,
        out_dir=out_path,
    )
    return params_doc


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline MOZ1 base dynamics identification, audit, report, and plotting."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="One or more experiment run directories produced by run_base_id_experiment.py.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="New output directory for fitted params, audit JSON, report, and summary plots.",
    )
    args = parser.parse_args(argv)

    try:
        params = run_offline_fit(args.run_dirs, args.out)
    except (OutputDirExistsError, FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] Offline fit failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"[OK] Identification complete -> {args.out}\n"
        f"  kinematic_type : {params['kinematic_type']}\n"
        f"  vx             : delay={params['axes']['vx']['delay_s']*1000:.1f} ms, "
        f"T={params['axes']['vx']['time_constant_s']*1000:.1f} ms, "
        f"a_acc={params['axes']['vx']['a_acc_max']:.2f}, a_brk={params['axes']['vx']['a_brake_max']:.2f}, "
        f"heldout_rmse={params['axes']['vx']['heldout_rmse']:.4f}\n"
        f"  vy             : delay={params['axes']['vy']['delay_s']*1000:.1f} ms, "
        f"T={params['axes']['vy']['time_constant_s']*1000:.1f} ms, "
        f"a_acc={params['axes']['vy']['a_acc_max']:.2f}, a_brk={params['axes']['vy']['a_brake_max']:.2f}, "
        f"heldout_rmse={params['axes']['vy']['heldout_rmse']:.4f}\n"
        f"  wz             : delay={params['axes']['wz']['delay_s']*1000:.1f} ms, "
        f"T={params['axes']['wz']['time_constant_s']*1000:.1f} ms, "
        f"a_acc={params['axes']['wz']['a_acc_max']:.2f}, a_brk={params['axes']['wz']['a_brake_max']:.2f}, "
        f"heldout_rmse={params['axes']['wz']['heldout_rmse']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
