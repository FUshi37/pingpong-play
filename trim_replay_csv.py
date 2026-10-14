#!/usr/bin/env python3
"""Trim the stuck/pre-detection frames from the replay CSVs.

The first ~18 rows are frozen at the detector's stale init value
(x=-77.26, z=-351.73) with t_rel 0.0->0.085, then real detection begins at
row 20 with t_rel~0.82. We drop every row before the jump and re-zero t_rel
so replay's (t_rel - elapsed).abs().argmin() timing and duration stay correct.

Detection of the cut point is data-driven: find the first row whose (x_mm, z_mm)
differs from row 0 by more than a small threshold -- that is the first real
detection. Everything before it is dropped.
"""
import csv

BASE = "src/pingpong_moz1/pingpong_controller/outputs/rl_sim"
FILES = [
    f"{BASE}/real_ball_replay_untransformed.csv",
    f"{BASE}/real_ball_replay_untransformed_z_plus_0.3.csv",
]
POS_JUMP_MM = 5.0  # a real detection moves the ball by > this vs the frozen init


def trim(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = rows[1:]

    xi = header.index("x_mm")
    zi = header.index("z_mm")
    ti = header.index("t_rel")

    x0 = float(data[0][xi])
    z0 = float(data[0][zi])

    cut = None
    for i, r in enumerate(data):
        dx = abs(float(r[xi]) - x0)
        dz = abs(float(r[zi]) - z0)
        if dx > POS_JUMP_MM or dz > POS_JUMP_MM:
            cut = i
            break
    if cut is None:
        print(f"[SKIP] {path}: no jump found")
        return

    kept = data[cut:]
    t_base = float(kept[0][ti])
    for r in kept:
        r[ti] = str(float(r[ti]) - t_base)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    print(f"[OK] {path.split('/')[-1]}")
    print(f"     dropped {cut} frozen rows (t_rel 0..{t_base:.4f}s), "
          f"kept {len(kept)} rows")
    print(f"     new t_rel range: 0.000 .. {float(kept[-1][ti]):.4f}s, "
          f"first real pos x={kept[0][xi]}, z={kept[0][zi]}")


for p in FILES:
    trim(p)
