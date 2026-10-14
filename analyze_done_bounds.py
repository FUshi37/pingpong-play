#!/usr/bin/env python3
"""Analyze whether the real ball replay trajectory exceeds the sim done bounds.

Transform (from pingpong_node._transform_ball_pos_to_sim_frame):
    sim_x = x_mm/1000                    (base at origin, yaw=0 in replay)
    sim_y = y_mm/1000
    sim_z = z_mm/1000 + SIM_BASE_HEIGHT_M
racket_anchor = right-arm EE initial pos in sim frame, read from policy obs trace
(obs/026-028 at step 0).
"""
import csv
import math

SIM_BASE_HEIGHT_M = 1.302396

BASE = "src/pingpong_moz1/pingpong_controller/outputs/rl_sim"
FILES = {
    "ORIGINAL (untransformed)": f"{BASE}/real_ball_replay_untransformed.csv",
    "SHIFTED (+0.3m in z)":      f"{BASE}/real_ball_replay_untransformed_z_plus_0.3.csv",
}
OBS_TRACE = f"{BASE}/simulation_replay/policy_obs_trace.csv"

# done thresholds
BALL_Z_LOW, BALL_Z_HIGH = 0.8, 1.9
BALL_XY_HALF = 0.5


def read_anchor(path):
    """racket_anchor = racket pos (obs/026-028) at first step."""
    with open(path) as f:
        r = csv.DictReader(f)
        row = next(r)
        return (float(row["obs/026"]), float(row["obs/027"]), float(row["obs/028"]))


def read_racket_traj(path):
    """racket pos over time (obs/026-028) and its z-rel vs anchor."""
    xs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            xs.append((float(row["obs/026"]), float(row["obs/027"]), float(row["obs/028"])))
    return xs


def analyze_ball(path, anchor):
    ax, ay, az = anchor
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            x = float(row["x_mm"]) / 1000.0
            y = float(row["y_mm"]) / 1000.0
            z = float(row["z_mm"]) / 1000.0 + SIM_BASE_HEIGHT_M
            t = float(row["t_rel"])
            rows.append((t, x, y, z))

    n = len(rows)
    viol = {"ball_too_low": 0, "ball_too_high": 0, "ball_x_oob": 0, "ball_y_oob": 0}
    first = {k: None for k in viol}
    zs = [r[3] for r in rows]
    dxs = [r[1] - ax for r in rows]
    dys = [r[2] - ay for r in rows]

    for (t, x, y, z) in rows:
        if z < BALL_Z_LOW:
            viol["ball_too_low"] += 1
            if first["ball_too_low"] is None:
                first["ball_too_low"] = (t, z)
        if z > BALL_Z_HIGH:
            viol["ball_too_high"] += 1
            if first["ball_too_high"] is None:
                first["ball_too_high"] = (t, z)
        if abs(x - ax) > BALL_XY_HALF:
            viol["ball_x_oob"] += 1
            if first["ball_x_oob"] is None:
                first["ball_x_oob"] = (t, x)
        if abs(y - ay) > BALL_XY_HALF:
            viol["ball_y_oob"] += 1
            if first["ball_y_oob"] is None:
                first["ball_y_oob"] = (t, y)

    # earliest done across all ball terms
    firsts = [(v[0], k, v[1]) for k, v in first.items() if v is not None]
    earliest = min(firsts) if firsts else None

    return {
        "n": n, "viol": viol, "first": first, "earliest": earliest,
        "z_min": min(zs), "z_max": max(zs),
        "dx_min": min(dxs), "dx_max": max(dxs),
        "dy_min": min(dys), "dy_max": max(dys),
    }


def pct(c, n):
    return f"{c} ({100.0*c/n:.1f}%)"


anchor = read_anchor(OBS_TRACE)
print("=" * 72)
print(f"racket_anchor (right-arm EE init, sim frame) = "
      f"({anchor[0]:.4f}, {anchor[1]:.4f}, {anchor[2]:.4f}) m")
print(f"SIM_BASE_HEIGHT_M = {SIM_BASE_HEIGHT_M}")
print("Ball done windows:")
print(f"  z in [{BALL_Z_LOW}, {BALL_Z_HIGH}]")
print(f"  x in [{anchor[0]-BALL_XY_HALF:.3f}, {anchor[0]+BALL_XY_HALF:.3f}]")
print(f"  y in [{anchor[1]-BALL_XY_HALF:.3f}, {anchor[1]+BALL_XY_HALF:.3f}]")
print("=" * 72)

for label, path in FILES.items():
    try:
        a = analyze_ball(path, anchor)
    except FileNotFoundError:
        print(f"\n### {label}: FILE NOT FOUND ({path})")
        continue
    n = a["n"]
    print(f"\n### {label}   ({n} frames)")
    print(f"  ball_z range: [{a['z_min']:.3f}, {a['z_max']:.3f}]  (limit 0.8..1.9)")
    print(f"  ball_x-anchor range: [{a['dx_min']:+.3f}, {a['dx_max']:+.3f}]  (limit +-0.5)")
    print(f"  ball_y-anchor range: [{a['dy_min']:+.3f}, {a['dy_max']:+.3f}]  (limit +-0.5)")
    print("  Violations (frames out of bounds):")
    for k in ["ball_too_low", "ball_too_high", "ball_x_oob", "ball_y_oob"]:
        print(f"    {k:16s}: {pct(a['viol'][k], n)}", end="")
        if a["first"][k]:
            print(f"   first@ t={a['first'][k][0]:.3f}s val={a['first'][k][1]:.3f}")
        else:
            print()
    if a["earliest"]:
        t, k, v = a["earliest"]
        print(f"  >>> EARLIEST DONE: '{k}' at t={t:.3f}s (val={v:.3f})")
    else:
        print("  >>> No ball-bound done triggered.")

# racket own bounds during the actual (+0.3) replay run
print("\n" + "=" * 72)
print("RACKET bounds during the logged replay run (policy_obs_trace = +0.3 run):")
rack = read_racket_traj(OBS_TRACE)
az = anchor[2]
z_rel = [r[2] - az for r in rack]
far = [math.dist(r, anchor) for r in rack]
n = len(rack)
too_high = sum(1 for z in z_rel if z > 0.24)
too_low = sum(1 for z in z_rel if z < -0.12)
too_far = sum(1 for d in far if d > 1.1)
print(f"  racket_z_rel range: [{min(z_rel):+.3f}, {max(z_rel):+.3f}]  "
      f"(limit -0.12..+0.24)")
print(f"    too_high (>+0.24): {pct(too_high, n)}")
print(f"    too_low  (<-0.12): {pct(too_low, n)}")
print(f"  racket dist from anchor range: [{min(far):.3f}, {max(far):.3f}]  (limit 1.1)")
print(f"    too_far  (>1.1):  {pct(too_far, n)}")
print("=" * 72)
