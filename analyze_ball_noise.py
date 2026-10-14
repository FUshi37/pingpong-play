#!/usr/bin/env python3
"""Analyze ball state noise characteristics from the original replay CSV.

This analyzes:
1. Position noise (x, y, z mm)
2. Velocity noise (vx, vy, vz mm/s)
3. Temporal consistency (jitter in consecutive frames)
4. Done-bound violations for +0.5m shift
"""
import csv
import numpy as np
import math

SIM_BASE_HEIGHT_M = 1.302396
BASE = "src/pingpong_moz1/pingpong_controller/outputs/rl_sim"
ORIGINAL = f"{BASE}/real_ball_replay_untransformed.csv"
SHIFTED_05 = f"{BASE}/real_ball_replay_untransformed_z_plus_0.5.csv"
OBS_TRACE = f"{BASE}/simulation_replay/policy_obs_trace.csv"

# Done bounds
BALL_Z_LOW, BALL_Z_HIGH = 0.8, 1.9
BALL_XY_HALF = 0.5


def read_anchor(path):
    """racket_anchor from obs trace first step."""
    with open(path) as f:
        r = csv.DictReader(f)
        row = next(r)
        return (float(row["obs/026"]), float(row["obs/027"]), float(row["obs/028"]))


def analyze_noise_and_bounds(path, anchor):
    """Analyze noise and done violations."""
    ax, ay, az = anchor

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n = len(rows)

    # Extract data arrays
    t = np.array([float(r["t_rel"]) for r in rows])
    x_mm = np.array([float(r["x_mm"]) for r in rows])
    y_mm = np.array([float(r["y_mm"]) for r in rows])
    z_mm = np.array([float(r["z_mm"]) for r in rows])
    vx = np.array([float(r["vx_mm_s"]) for r in rows])
    vy = np.array([float(r["vy_mm_s"]) for r in rows])
    vz = np.array([float(r["vz_mm_s"]) for r in rows])

    # Convert to sim frame
    x_m = x_mm / 1000.0
    y_m = y_mm / 1000.0
    z_m = z_mm / 1000.0 + SIM_BASE_HEIGHT_M

    # Position noise (frame-to-frame jitter in consecutive measurements)
    # dt is ~0.005s (200Hz), so frame-to-frame diff / dt gives instantaneous velocity
    dt = np.diff(t)
    dx_mm = np.diff(x_mm)
    dy_mm = np.diff(y_mm)
    dz_mm = np.diff(z_mm)

    # Velocity from position diff
    vx_from_pos = dx_mm / dt
    vy_from_pos = dy_mm / dt
    vz_from_pos = dz_mm / dt

    # Velocity noise (difference between reported velocity and position-derived velocity)
    vx_noise = vx[:-1] - vx_from_pos
    vy_noise = vy[:-1] - vy_from_pos
    vz_noise = vz[:-1] - vz_from_pos

    # Position jitter (absolute frame-to-frame change)
    pos_jitter_mm = np.sqrt(dx_mm**2 + dy_mm**2 + dz_mm**2)

    # Acceleration (change in velocity)
    dvx = np.diff(vx)
    dvy = np.diff(vy)
    dvz = np.diff(vz)
    # dt has n-1 elements, dvx/dvy/dvz also have n-1, so we need dt for acceleration
    accel_mag = np.sqrt(dvx**2 + dvy**2 + dvz**2) / dt  # mm/s^2

    # Done violations
    viol = {
        "ball_too_low": np.sum(z_m < BALL_Z_LOW),
        "ball_too_high": np.sum(z_m > BALL_Z_HIGH),
        "ball_x_oob": np.sum(np.abs(x_m - ax) > BALL_XY_HALF),
        "ball_y_oob": np.sum(np.abs(y_m - ay) > BALL_XY_HALF),
    }

    # Find first violation
    first = {}
    for i in range(n):
        if z_m[i] < BALL_Z_LOW and "ball_too_low" not in first:
            first["ball_too_low"] = (t[i], z_m[i])
        if z_m[i] > BALL_Z_HIGH and "ball_too_high" not in first:
            first["ball_too_high"] = (t[i], z_m[i])
        if abs(x_m[i] - ax) > BALL_XY_HALF and "ball_x_oob" not in first:
            first["ball_x_oob"] = (t[i], x_m[i])
        if abs(y_m[i] - ay) > BALL_XY_HALF and "ball_y_oob" not in first:
            first["ball_y_oob"] = (t[i], y_m[i])

    return {
        "n": n,
        "duration": t[-1],
        # Position stats (sim frame, meters)
        "x_range": (x_m.min(), x_m.max()),
        "y_range": (y_m.min(), y_m.max()),
        "z_range": (z_m.min(), z_m.max()),
        "x_std": x_m.std(),
        "y_std": y_m.std(),
        "z_std": z_m.std(),
        # Velocity stats (mm/s)
        "vx_range": (vx.min(), vx.max()),
        "vy_range": (vy.min(), vy.max()),
        "vz_range": (vz.min(), vz.max()),
        "vx_std": vx.std(),
        "vy_std": vy.std(),
        "vz_std": vz.std(),
        # Noise characteristics
        "pos_jitter_mean": pos_jitter_mm.mean(),
        "pos_jitter_std": pos_jitter_mm.std(),
        "pos_jitter_max": pos_jitter_mm.max(),
        "vx_noise_std": vx_noise.std(),
        "vy_noise_std": vy_noise.std(),
        "vz_noise_std": vz_noise.std(),
        "accel_mean": accel_mag.mean(),
        "accel_std": accel_mag.std(),
        "accel_max": accel_mag.max(),
        # Done violations
        "viol": viol,
        "first": first,
        "dx_range": (x_m.min() - ax, x_m.max() - ax),
        "dy_range": (y_m.min() - ay, y_m.max() - ay),
    }


def print_analysis(label, stats):
    print(f"\n{'='*72}")
    print(f"{label}")
    print(f"{'='*72}")
    print(f"Frames: {stats['n']}, Duration: {stats['duration']:.2f}s")

    print(f"\nPosition (sim frame, meters):")
    print(f"  x: [{stats['x_range'][0]:+.4f}, {stats['x_range'][1]:+.4f}]  std={stats['x_std']:.4f}")
    print(f"  y: [{stats['y_range'][0]:+.4f}, {stats['y_range'][1]:+.4f}]  std={stats['y_std']:.4f}")
    print(f"  z: [{stats['z_range'][0]:+.4f}, {stats['z_range'][1]:+.4f}]  std={stats['z_std']:.4f}")

    print(f"\nVelocity (mm/s):")
    print(f"  vx: [{stats['vx_range'][0]:+.1f}, {stats['vx_range'][1]:+.1f}]  std={stats['vx_std']:.1f}")
    print(f"  vy: [{stats['vy_range'][0]:+.1f}, {stats['vy_range'][1]:+.1f}]  std={stats['vy_std']:.1f}")
    print(f"  vz: [{stats['vz_range'][0]:+.1f}, {stats['vz_range'][1]:+.1f}]  std={stats['vz_std']:.1f}")

    print(f"\nNoise Characteristics:")
    print(f"  Position jitter (frame-to-frame Δ):")
    print(f"    mean: {stats['pos_jitter_mean']:.2f} mm")
    print(f"    std:  {stats['pos_jitter_std']:.2f} mm")
    print(f"    max:  {stats['pos_jitter_max']:.2f} mm")
    print(f"  Velocity noise (reported - position-derived, std):")
    print(f"    vx: {stats['vx_noise_std']:.1f} mm/s")
    print(f"    vy: {stats['vy_noise_std']:.1f} mm/s")
    print(f"    vz: {stats['vz_noise_std']:.1f} mm/s")
    print(f"  Acceleration magnitude:")
    print(f"    mean: {stats['accel_mean']:.0f} mm/s²")
    print(f"    std:  {stats['accel_std']:.0f} mm/s²")
    print(f"    max:  {stats['accel_max']:.0f} mm/s²")

    print(f"\nDone Bound Violations:")
    print(f"  ball_z range vs [0.8, 1.9]: [{stats['z_range'][0]:.3f}, {stats['z_range'][1]:.3f}]")
    print(f"  ball_x-anchor vs ±0.5: [{stats['dx_range'][0]:+.3f}, {stats['dx_range'][1]:+.3f}]")
    print(f"  ball_y-anchor vs ±0.5: [{stats['dy_range'][0]:+.3f}, {stats['dy_range'][1]:+.3f}]")
    for k, v in stats['viol'].items():
        pct = 100.0 * v / stats['n']
        print(f"    {k:16s}: {v:4d} frames ({pct:4.1f}%)", end="")
        if k in stats['first']:
            t, val = stats['first'][k]
            print(f"  first@ t={t:.3f}s val={val:.3f}")
        else:
            print()


# Main
anchor = read_anchor(OBS_TRACE)
print(f"racket_anchor = ({anchor[0]:.4f}, {anchor[1]:.4f}, {anchor[2]:.4f}) m")
print(f"SIM_BASE_HEIGHT_M = {SIM_BASE_HEIGHT_M}")

orig = analyze_noise_and_bounds(ORIGINAL, anchor)
print_analysis("ORIGINAL (untransformed, trimmed)", orig)

shift05 = analyze_noise_and_bounds(SHIFTED_05, anchor)
print_analysis("SHIFTED +0.5m in z", shift05)

print(f"\n{'='*72}")
print("SUMMARY:")
print(f"  Original z_min = {orig['z_range'][0]:.3f}m  (needs +{0.8-orig['z_range'][0]:.3f}m to reach 0.8)")
print(f"  +0.5m shift brings z to [{shift05['z_range'][0]:.3f}, {shift05['z_range'][1]:.3f}]")
print(f"  Position jitter: {orig['pos_jitter_mean']:.2f} ± {orig['pos_jitter_std']:.2f} mm")
print(f"  Velocity noise: vx={orig['vx_noise_std']:.1f}, vy={orig['vy_noise_std']:.1f}, vz={orig['vz_noise_std']:.1f} mm/s")
print(f"{'='*72}")
