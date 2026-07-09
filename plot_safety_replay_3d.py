#!/usr/bin/env python3
"""Plot 3D ball and racket/end-effector trajectories for safety replays."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def as_float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_bool(row: dict, key: str, default: bool = False) -> bool:
    if key not in row or row.get(key, "") == "":
        return default
    try:
        return bool(int(float(row.get(key, int(default)))))
    except (TypeError, ValueError):
        return default


def parse_limits_arg(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned or cleaned == "[]":
        return None
    cleaned = cleaned.strip("[]")
    if not cleaned:
        return None
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected 'min,max' limits, got {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise argparse.ArgumentTypeError(
            f"invalid limits {text!r}; expected finite min < max")
    return (lo, hi)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def finite_xyz(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    mask = np.all(np.isfinite(points), axis=1)
    return points[mask]


def replay_to_sim(
    replay_rows: list[dict],
    frame: str,
    sim_base_height_m: float,
    sim_cartesian_x_offset_m: float,
    sim_cartesian_y_offset_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    valid = []
    times = []
    for row in replay_rows:
        p = np.array([
            as_float(row, "x_mm") * 1e-3,
            as_float(row, "y_mm") * 1e-3,
            as_float(row, "z_mm") * 1e-3,
        ], dtype=float)
        if frame == "base_link":
            p[0] += sim_cartesian_x_offset_m
            p[1] += sim_cartesian_y_offset_m
            p[2] += sim_base_height_m
        points.append(p)
        # Synthetic replays have explicit valid/t_rel columns. Real hardware
        # trajectory CSVs only contain valid detections and use timestamp.
        valid.append(as_bool(row, "valid", True))
        t = as_float(row, "t_rel")
        if not math.isfinite(t):
            t = as_float(row, "timestamp")
        times.append(t)
    times_arr = np.asarray(times, dtype=float)
    finite_t = times_arr[np.isfinite(times_arr)]
    if len(finite_t):
        times_arr = times_arr - finite_t[0]
    return times_arr, np.asarray(points), np.asarray(valid, dtype=bool)


def policy_trace_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    times = []
    ball = []
    ball_valid = []
    ball_age = []
    racket = []
    for row in rows:
        times.append(as_float(row, "time_sec"))
        obs_ball = [
            as_float(row, "obs/020"),
            as_float(row, "obs/021"),
            as_float(row, "obs/022"),
        ]
        if not all(math.isfinite(v) for v in obs_ball):
            obs_ball = [
                as_float(row, "ball_pos_x_m"),
                as_float(row, "ball_pos_y_m"),
                as_float(row, "ball_pos_z_m"),
            ]
        ball.append(obs_ball)
        ball_valid.append(as_bool(row, "ball_valid", False))
        ball_age.append(as_float(row, "ball_age_s"))
        racket.append([
            as_float(row, "obs/026"),
            as_float(row, "obs/027"),
            as_float(row, "obs/028"),
        ])
    return {
        "time": np.asarray(times, dtype=float),
        "ball": np.asarray(ball, dtype=float),
        "ball_valid": np.asarray(ball_valid, dtype=bool),
        "ball_age": np.asarray(ball_age, dtype=float),
        "racket": np.asarray(racket, dtype=float),
    }


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def plot_masked_line(
    ax,
    points: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 2.0,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    first = True
    start = None
    for i, ok in enumerate(mask.tolist() + [False]):
        if ok and start is None:
            start = i
        if (not ok) and start is not None:
            segment = finite_xyz(points[start:i])
            if len(segment) >= 2:
                ax.plot(
                    segment[:, 0], segment[:, 1], segment[:, 2],
                    color=color, linewidth=linewidth, alpha=alpha,
                    linestyle=linestyle, label=label if first else None)
                first = False
            elif len(segment) == 1:
                ax.scatter(
                    segment[:, 0], segment[:, 1], segment[:, 2],
                    color=color, s=18, alpha=alpha,
                    label=label if first else None)
                first = False
            start = None


def draw_box(
    ax,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    *,
    color: str = "0.55",
    alpha: float = 0.35,
) -> None:
    xs = xlim
    ys = ylim
    zs = zlim
    corners = np.array([
        [xs[0], ys[0], zs[0]], [xs[1], ys[0], zs[0]],
        [xs[1], ys[1], zs[0]], [xs[0], ys[1], zs[0]],
        [xs[0], ys[0], zs[1]], [xs[1], ys[0], zs[1]],
        [xs[1], ys[1], zs[1]], [xs[0], ys[1], zs[1]],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color=color, linewidth=1.0, alpha=alpha)


def set_equal_axes(ax, point_sets: list[np.ndarray]) -> None:
    pts = [finite_xyz(p) for p in point_sets if p.size > 0]
    pts = [p for p in pts if len(p) > 0]
    if not pts:
        return
    all_pts = np.vstack(pts)
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center = (mins + maxs) * 0.5
    span = np.max(maxs - mins)
    if not math.isfinite(float(span)) or span < 1e-3:
        span = 0.4
    radius = span * 0.58 + 0.04
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def mark_endpoint(ax, points: np.ndarray, *, color: str, prefix: str) -> None:
    pts = finite_xyz(points)
    if len(pts) == 0:
        return
    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color=color, marker="o", s=38, label=f"{prefix} start")
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=color, marker="^", s=48, label=f"{prefix} end")


def finite_records(
    times: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray | None = None,
    age: np.ndarray | None = None,
    max_points: int = 1800,
) -> list[dict]:
    mask = np.all(np.isfinite(points), axis=1) & np.isfinite(times)
    if valid is not None:
        mask &= np.isfinite(valid.astype(float))
    idx = np.flatnonzero(mask)
    if max_points > 0 and len(idx) > max_points:
        idx = np.linspace(idx[0], idx[-1], max_points).astype(int)
    out = []
    for i in idx:
        item = {
            "t": round(float(times[i]), 6),
            "x": round(float(points[i, 0]), 6),
            "y": round(float(points[i, 1]), 6),
            "z": round(float(points[i, 2]), 6),
        }
        if valid is not None:
            item["valid"] = bool(valid[i])
        if age is not None:
            item["age"] = round(float(age[i]), 6)
        out.append(item)
    return out


def axis_limits_for_html(point_sets: list[np.ndarray]) -> dict:
    pts = [finite_xyz(p) for p in point_sets if p.size > 0]
    pts = [p for p in pts if len(p) > 0]
    if not pts:
        return {"center": [0.0, 0.0, 1.0], "radius": 1.0}
    all_pts = np.vstack(pts)
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center = (mins + maxs) * 0.5
    span = float(np.max(maxs - mins))
    if not math.isfinite(span) or span < 1e-3:
        span = 0.4
    radius = span * 0.62 + 0.08
    return {
        "center": [round(float(v), 6) for v in center],
        "radius": round(float(radius), 6),
    }


def write_interactive_html(
    path: Path,
    *,
    scenario: str,
    replay_time: np.ndarray,
    replay_ball: np.ndarray,
    replay_valid: np.ndarray,
    policy_time: np.ndarray,
    policy_ball: np.ndarray,
    policy_ball_valid: np.ndarray,
    policy_ball_age: np.ndarray,
    racket: np.ndarray,
    anchor: np.ndarray | None,
    done_z_min_m: float,
    done_z_max_m: float,
    done_x_range_m: float,
    done_y_range_m: float,
    ee_x_limits_m: tuple[float, float] | None,
    ee_y_limits_m: tuple[float, float] | None,
    ee_z_limits_m: tuple[float, float] | None,
    max_points: int,
) -> None:
    if len(policy_time) == len(racket):
        racket_time = policy_time
    else:
        racket_time = np.arange(len(racket), dtype=float) * 0.005

    extra = []
    ball_box = None
    if anchor is not None and np.all(np.isfinite(anchor)):
        ball_box = {
            "x0": round(float(anchor[0] - done_x_range_m), 6),
            "x1": round(float(anchor[0] + done_x_range_m), 6),
            "y0": round(float(anchor[1] - done_y_range_m), 6),
            "y1": round(float(anchor[1] + done_y_range_m), 6),
            "z0": round(float(done_z_min_m), 6),
            "z1": round(float(done_z_max_m), 6),
        }
        extra.append(np.array([
            [ball_box["x0"], ball_box["y0"], ball_box["z0"]],
            [ball_box["x1"], ball_box["y1"], ball_box["z1"]],
        ]))
    ee_box = None
    if ee_x_limits_m and ee_y_limits_m and ee_z_limits_m:
        ee_box = {
            "x0": round(float(ee_x_limits_m[0]), 6),
            "x1": round(float(ee_x_limits_m[1]), 6),
            "y0": round(float(ee_y_limits_m[0]), 6),
            "y1": round(float(ee_y_limits_m[1]), 6),
            "z0": round(float(ee_z_limits_m[0]), 6),
            "z1": round(float(ee_z_limits_m[1]), 6),
        }
        extra.append(np.array([
            [ee_box["x0"], ee_box["y0"], ee_box["z0"]],
            [ee_box["x1"], ee_box["y1"], ee_box["z1"]],
        ]))

    limits = axis_limits_for_html([replay_ball, policy_ball, racket, *extra])
    t_max = 0.0
    for arr in (replay_time, policy_time, racket_time):
        vals = arr[np.isfinite(arr)]
        if len(vals):
            t_max = max(t_max, float(np.max(vals)))

    payload = {
        "scenario": scenario,
        "tMax": round(t_max, 6),
        "limits": limits,
        "box": ball_box,
        "ballBox": ball_box,
        "eeBox": ee_box,
        "anchor": (
            [round(float(v), 6) for v in anchor]
            if anchor is not None and np.all(np.isfinite(anchor)) else None
        ),
        "replayBall": finite_records(
            replay_time, replay_ball, replay_valid, max_points=max_points),
        "policyBall": finite_records(
            policy_time, policy_ball, policy_ball_valid,
            age=policy_ball_age, max_points=max_points),
        "racket": finite_records(
            racket_time, racket, None, max_points=max_points),
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    title = html.escape(
        f"Safety replay interactive 3D: {scenario}"
        if scenario else "Safety replay interactive 3D")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  body {{
    margin: 0;
    background: #f6f7f9;
    color: #1f2933;
  }}
  .wrap {{
    display: grid;
    grid-template-rows: auto minmax(420px, 1fr) auto;
    min-height: 100vh;
  }}
  header {{
    padding: 14px 18px 8px;
    border-bottom: 1px solid #d8dde6;
    background: white;
  }}
  h1 {{
    margin: 0;
    font-size: 18px;
    font-weight: 650;
  }}
  .meta {{
    margin-top: 5px;
    color: #65758b;
    font-size: 13px;
  }}
  #stage {{
    width: 100%;
    height: 100%;
    min-height: 560px;
    cursor: grab;
    background: linear-gradient(#ffffff, #eef2f7);
    touch-action: none;
  }}
  #stage.dragging {{
    cursor: grabbing;
  }}
  .controls {{
    display: grid;
    grid-template-columns: auto 1fr auto auto auto;
    gap: 12px;
    align-items: center;
    padding: 12px 18px 14px;
    border-top: 1px solid #d8dde6;
    background: white;
  }}
  button {{
    border: 1px solid #b8c2d1;
    background: #fff;
    border-radius: 6px;
    padding: 7px 13px;
    font-size: 14px;
    cursor: pointer;
  }}
  button:hover {{
    background: #f0f4f8;
  }}
  input[type="range"] {{
    width: 100%;
  }}
  .readout {{
    min-width: 135px;
    font-variant-numeric: tabular-nums;
    font-size: 14px;
    color: #273445;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 13px;
    color: #455468;
  }}
  .item::before {{
    content: "";
    display: inline-block;
    width: 18px;
    height: 3px;
    margin-right: 6px;
    vertical-align: middle;
    background: var(--c);
  }}
  .hint {{
    font-size: 13px;
    color: #65758b;
  }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{title}</h1>
    <div class="meta">Drag to rotate, mouse wheel to zoom. Use Play/Pause or the time slider to inspect motion.</div>
  </header>
  <canvas id="stage"></canvas>
  <div class="controls">
    <button id="play">Play</button>
    <input id="slider" type="range" min="0" max="1000" value="0">
    <div class="readout" id="time">t=0.000s</div>
    <label class="hint">speed <input id="speed" type="number" min="0.1" max="20" step="0.1" value="1" style="width:58px"></label>
    <div class="legend">
      <span class="item" style="--c:#f28e2b">replay ball valid</span>
      <span class="item" style="--c:#9aa3af">replay ball invalid/FOV lost</span>
      <span class="item" style="--c:#e15759">policy ball obs valid</span>
      <span class="item" style="--c:#8b5cf6">policy ball obs held/stale</span>
      <span class="item" style="--c:#4e79a7">racket/EE trajectory</span>
      <span class="item" style="--c:#111827">ball done bounds</span>
      <span class="item" style="--c:#0891b2">EE workspace</span>
      <span class="item" style="--c:#111827">racket anchor</span>
      <span class="item" style="--c:#b45309">ball start/end</span>
      <span class="item" style="--c:#1d4ed8">racket start/end</span>
    </div>
  </div>
</div>
<script>
const DATA = {payload_json};
const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const timeEl = document.getElementById('time');
const speedEl = document.getElementById('speed');

let yaw = -0.75;
let pitch = 0.45;
let zoom = 1.0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let playing = false;
let currentT = 0;
let lastFrame = performance.now();

function resize() {{
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}

function rotate(p) {{
  const c = DATA.limits.center;
  let x = p.x - c[0], y = p.y - c[1], z = p.z - c[2];
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const z1 = z;
  const y2 = cp * y1 - sp * z1;
  const z2 = sp * y1 + cp * z1;
  return {{x:x1, y:y2, z:z2}};
}}

function project(p) {{
  const rect = canvas.getBoundingClientRect();
  const r = rotate(p);
  const scale = Math.min(rect.width, rect.height) * 0.40 * zoom / DATA.limits.radius;
  return {{
    x: rect.width * 0.50 + r.x * scale,
    y: rect.height * 0.53 - r.y * scale,
    depth: r.z
  }};
}}

function pointAtOrBefore(arr, t) {{
  let best = null;
  for (const p of arr) {{
    if (p.t <= t) best = p;
    else break;
  }}
  return best;
}}

function pointsUntil(arr, t) {{
  const out = [];
  for (const p of arr) {{
    if (p.t <= t) out.push(p);
    else break;
  }}
  return out;
}}

function drawLine(points, color, width, alpha=1, dashed=false) {{
  if (points.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dashed ? [7, 7] : []);
  ctx.beginPath();
  let moved = false;
  for (const p of points) {{
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {{
      moved = false;
      continue;
    }}
    const q = project(p);
    if (!moved) {{
      ctx.moveTo(q.x, q.y);
      moved = true;
    }} else {{
      ctx.lineTo(q.x, q.y);
    }}
  }}
  ctx.stroke();
  ctx.restore();
}}

function drawSegmented(points, colorValid, colorInvalid, width, alpha) {{
  let seg = [];
  let state = null;
  function flush() {{
    if (seg.length > 1) drawLine(seg, state ? colorValid : colorInvalid, width, alpha, !state);
    seg = [];
  }}
  for (const p of points) {{
    if (state === null) state = !!p.valid;
    if (!!p.valid !== state) {{
      flush();
      state = !!p.valid;
    }}
    seg.push(p);
  }}
  flush();
}}

function isFinitePoint(p) {{
  return p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z);
}}

function firstWhere(arr, predicate) {{
  for (const p of arr) {{
    if (predicate(p)) return p;
  }}
  return null;
}}

function lastWhere(arr, predicate) {{
  for (let i = arr.length - 1; i >= 0; i--) {{
    const p = arr[i];
    if (predicate(p)) return p;
  }}
  return null;
}}

function drawDot(p, color, radius, stroke='#ffffff', alpha=1) {{
  if (!isFinitePoint(p)) return;
  const q = project(p);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(q.x, q.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}}

function drawTriangle(p, color, size, stroke='#ffffff', alpha=1) {{
  if (!isFinitePoint(p)) return;
  const q = project(p);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(q.x, q.y - size);
  ctx.lineTo(q.x + size * 0.9, q.y + size * 0.65);
  ctx.lineTo(q.x - size * 0.9, q.y + size * 0.65);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}}

function drawEndpointMarkers() {{
  const ballStart = firstWhere(DATA.replayBall, p => p.valid && isFinitePoint(p));
  const ballEnd = lastWhere(DATA.replayBall, p => p.valid && isFinitePoint(p));
  const racketStart = firstWhere(DATA.racket, isFinitePoint);
  const racketEnd = lastWhere(DATA.racket, isFinitePoint);
  if (ballStart) {{
    drawDot(ballStart, '#f28e2b', 4, '#ffffff', 0.80);
  }}
  if (ballEnd && ballEnd !== ballStart) {{
    drawTriangle(ballEnd, '#f28e2b', 5, '#ffffff', 0.82);
  }}
  if (racketStart) {{
    drawDot(racketStart, '#4e79a7', 4, '#ffffff', 0.80);
  }}
  if (racketEnd && racketEnd !== racketStart) {{
    drawTriangle(racketEnd, '#4e79a7', 5, '#ffffff', 0.82);
  }}
}}

function drawStaticLegendPanel() {{
  const items = [
    {{label:'replay ball valid', color:'#f28e2b', kind:'line'}},
    {{label:'replay ball invalid/FOV lost', color:'#9aa3af', kind:'dash'}},
    {{label:'policy ball obs valid', color:'#e15759', kind:'line'}},
    {{label:'policy ball obs held/stale', color:'#8b5cf6', kind:'dash'}},
    {{label:'racket/EE trajectory', color:'#4e79a7', kind:'line'}},
    {{label:'ball done bounds', color:'#111827', kind:'line'}},
    {{label:'EE workspace', color:'#0891b2', kind:'line'}},
    {{label:'racket anchor', color:'#111827', kind:'marker'}},
    {{label:'ball start/end', color:'#b45309', kind:'marker'}},
    {{label:'racket start/end', color:'#1d4ed8', kind:'marker'}},
  ];
  const x = 18, y = 18, w = 275, titleH = 24, rowH = 18;
  const h = titleH + items.length * rowH + 12;
  ctx.save();
  ctx.font = '12px sans-serif';
  ctx.globalAlpha = 0.86;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(x, y, w, h);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#d1d5db';
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = '#273445';
  ctx.font = '13px sans-serif';
  ctx.fillText('Trajectory labels', x + 14, y + 18);
  ctx.font = '12px sans-serif';
  items.forEach((item, i) => {{
    const yy = y + titleH + 12 + i * rowH;
    ctx.strokeStyle = item.color;
    ctx.fillStyle = item.color;
    ctx.lineWidth = 2.4;
    ctx.setLineDash(item.kind === 'dash' ? [6, 5] : []);
    if (item.kind === 'marker') {{
      ctx.beginPath();
      ctx.arc(x + 21, yy - 4, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(x + 35, yy - 9);
      ctx.lineTo(x + 40, yy);
      ctx.lineTo(x + 30, yy);
      ctx.closePath();
      ctx.fill();
    }} else {{
      ctx.beginPath();
      ctx.moveTo(x + 12, yy - 4);
      ctx.lineTo(x + 42, yy - 4);
      ctx.stroke();
    }}
    ctx.setLineDash([]);
    ctx.fillStyle = '#273445';
    ctx.fillText(item.label, x + 52, yy);
  }});
  ctx.restore();
  return y + h + 10;
}}

function drawBox(box, color, alpha=0.22, width=1.5) {{
  const b = box;
  if (!b) return;
  const pts = [
    {{x:b.x0,y:b.y0,z:b.z0}},{{x:b.x1,y:b.y0,z:b.z0}},{{x:b.x1,y:b.y1,z:b.z0}},{{x:b.x0,y:b.y1,z:b.z0}},
    {{x:b.x0,y:b.y0,z:b.z1}},{{x:b.x1,y:b.y0,z:b.z1}},{{x:b.x1,y:b.y1,z:b.z1}},{{x:b.x0,y:b.y1,z:b.z1}}
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  for (const [a,b2] of edges) {{
    const pa = project(pts[a]), pb = project(pts[b2]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }}
  ctx.restore();
}}

function drawAxes() {{
  const c = DATA.limits.center;
  const len = DATA.limits.radius * 0.45;
  const origin = {{x:c[0], y:c[1], z:c[2]}};
  const axes = [
    [{{x:c[0]+len,y:c[1],z:c[2]}}, '#c2410c', 'x'],
    [{{x:c[0],y:c[1]+len,z:c[2]}}, '#15803d', 'y'],
    [{{x:c[0],y:c[1],z:c[2]+len}}, '#1d4ed8', 'z']
  ];
  for (const [end, color, label] of axes) {{
    drawLine([origin, end], color, 1.5, 0.7);
    const q = project(end);
    ctx.fillStyle = color;
    ctx.font = '13px sans-serif';
    ctx.fillText(label, q.x + 5, q.y - 5);
  }}
}}

function drawTextPanel(y = 18) {{
  const rb = pointAtOrBefore(DATA.replayBall, currentT);
  const pb = pointAtOrBefore(DATA.policyBall, currentT);
  const rk = pointAtOrBefore(DATA.racket, currentT);
  const lines = [
    `t = ${{currentT.toFixed(3)}} s`,
    rb ? `replay ball: (${{rb.x.toFixed(3)}}, ${{rb.y.toFixed(3)}}, ${{rb.z.toFixed(3)}}) valid=${{rb.valid ? 1 : 0}}` : 'replay ball: none',
    pb ? `policy ball obs: (${{pb.x.toFixed(3)}}, ${{pb.y.toFixed(3)}}, ${{pb.z.toFixed(3)}}) valid=${{pb.valid ? 1 : 0}} age=${{Number.isFinite(pb.age) ? pb.age.toFixed(3) : 'nan'}}` : 'policy ball obs: none',
    rk ? `racket/EE: (${{rk.x.toFixed(3)}}, ${{rk.y.toFixed(3)}}, ${{rk.z.toFixed(3)}})` : 'racket/EE: none'
  ];
  ctx.save();
  ctx.font = '13px sans-serif';
  const w = 430, h = 22 + lines.length * 20;
  ctx.globalAlpha = 0.82;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(18, y, w, h);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = '#d1d5db';
  ctx.strokeRect(18, y, w, h);
  ctx.fillStyle = '#273445';
  lines.forEach((line, i) => ctx.fillText(line, 32, y + 27 + i * 20));
  ctx.restore();
}}

function draw() {{
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  drawAxes();
  drawBox(DATA.ballBox || DATA.box, '#111827', 0.22, 1.5);
  drawBox(DATA.eeBox, '#0891b2', 0.42, 2.0);

  drawSegmented(DATA.replayBall, '#f28e2b', '#9aa3af', 1.3, 0.30);
  drawSegmented(DATA.policyBall, '#e15759', '#8b5cf6', 1.1, 0.18);
  drawLine(DATA.racket, '#4e79a7', 1.2, 0.20);

  const replayHist = pointsUntil(DATA.replayBall, currentT);
  drawSegmented(replayHist, '#f28e2b', '#9aa3af', 3.0, 0.95);
  drawSegmented(pointsUntil(DATA.policyBall, currentT), '#e15759', '#8b5cf6', 2.0, 0.72);
  drawLine(pointsUntil(DATA.racket, currentT), '#4e79a7', 3.0, 0.90);

  drawEndpointMarkers();
  if (DATA.anchor) drawDot({{x:DATA.anchor[0], y:DATA.anchor[1], z:DATA.anchor[2]}}, '#111827', 6, '#ffffff');
  drawDot(pointAtOrBefore(DATA.replayBall, currentT), '#f28e2b', 7);
  const policyNow = pointAtOrBefore(DATA.policyBall, currentT);
  drawDot(policyNow, policyNow && policyNow.valid ? '#e15759' : '#8b5cf6', 5);
  drawDot(pointAtOrBefore(DATA.racket, currentT), '#4e79a7', 7);
  const infoTop = drawStaticLegendPanel();
  drawTextPanel(infoTop);
}}

function setTime(t) {{
  currentT = Math.max(0, Math.min(DATA.tMax, t));
  const denom = Math.max(DATA.tMax, 1e-9);
  slider.value = String(Math.round((currentT / denom) * 1000));
  timeEl.textContent = `t=${{currentT.toFixed(3)}}s / ${{DATA.tMax.toFixed(3)}}s`;
  draw();
}}

slider.addEventListener('input', () => {{
  setTime((Number(slider.value) / 1000) * DATA.tMax);
}});

playBtn.addEventListener('click', () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
  lastFrame = performance.now();
}});

canvas.addEventListener('pointerdown', e => {{
  dragging = true;
  canvas.classList.add('dragging');
  lastX = e.clientX;
  lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
}});
canvas.addEventListener('pointermove', e => {{
  if (!dragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  yaw += dx * 0.008;
  pitch = Math.max(-1.35, Math.min(1.35, pitch + dy * 0.008));
  draw();
}});
canvas.addEventListener('pointerup', e => {{
  dragging = false;
  canvas.classList.remove('dragging');
  try {{ canvas.releasePointerCapture(e.pointerId); }} catch (_) {{}}
}});
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  zoom *= Math.exp(-e.deltaY * 0.001);
  zoom = Math.max(0.25, Math.min(8, zoom));
  draw();
}}, {{passive:false}});

function tick(now) {{
  const dt = (now - lastFrame) / 1000;
  lastFrame = now;
  if (playing) {{
    const speed = Math.max(0.1, Number(speedEl.value) || 1);
    let next = currentT + dt * speed;
    if (next >= DATA.tMax) {{
      next = DATA.tMax;
      playing = false;
      playBtn.textContent = 'Play';
    }}
    setTime(next);
  }}
  requestAnimationFrame(tick);
}}

window.addEventListener('resize', resize);
resize();
setTime(0);
requestAnimationFrame(tick);
</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    print(f"[plot3d] wrote interactive: {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot 3D replay ball and racket/end-effector trajectories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--replay-csv", type=Path, required=True)
    p.add_argument("--policy-trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="Optional self-contained draggable/playable HTML output.",
    )
    p.add_argument("--scenario", default="")
    p.add_argument("--replay-frame", choices=("sim", "base_link"), default="sim")
    p.add_argument("--sim-base-height-m", type=float, default=0.947396)
    p.add_argument("--sim-cartesian-x-offset-m", type=float, default=0.0)
    p.add_argument("--sim-cartesian-y-offset-m", type=float, default=0.023583)
    p.add_argument("--done-z-min-m", type=float, default=0.8)
    p.add_argument("--done-z-max-m", type=float, default=1.9)
    p.add_argument("--done-x-range-m", type=float, default=0.5)
    p.add_argument("--done-y-range-m", type=float, default=0.5)
    p.add_argument(
        "--ee-x-limits-m",
        default="",
        help="Optional current EE workspace x limits as 'min,max'.",
    )
    p.add_argument(
        "--ee-y-limits-m",
        default="",
        help="Optional current EE workspace y limits as 'min,max'.",
    )
    p.add_argument(
        "--ee-z-limits-m",
        default="",
        help="Optional current EE workspace z limits as 'min,max'.",
    )
    p.add_argument("--max-points", type=int, default=2500)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.replay_csv.is_file():
        print(f"[plot3d] ERROR: replay CSV not found: {args.replay_csv}", file=sys.stderr)
        return 1
    if not args.policy_trace.is_file():
        print(f"[plot3d] ERROR: policy trace not found: {args.policy_trace}", file=sys.stderr)
        return 1

    replay_t, replay_ball, replay_valid = replay_to_sim(
        read_csv_rows(args.replay_csv),
        args.replay_frame,
        args.sim_base_height_m,
        args.sim_cartesian_x_offset_m,
        args.sim_cartesian_y_offset_m,
    )
    trace = policy_trace_arrays(read_csv_rows(args.policy_trace))
    policy_ball = trace["ball"]
    policy_ball_valid = trace["ball_valid"]
    policy_ball_age = trace["ball_age"]
    racket = trace["racket"]
    ee_x_limits = parse_limits_arg(args.ee_x_limits_m)
    ee_y_limits = parse_limits_arg(args.ee_y_limits_m)
    ee_z_limits = parse_limits_arg(args.ee_z_limits_m)

    # Downsample dense traces only for drawing; keep start/end semantics close.
    if args.max_points > 0:
        if len(replay_ball) > args.max_points:
            idx = np.linspace(0, len(replay_ball) - 1, args.max_points).astype(int)
            replay_t = replay_t[idx]
            replay_ball = replay_ball[idx]
            replay_valid = replay_valid[idx]
        if len(policy_ball) > args.max_points:
            idx = np.linspace(0, len(policy_ball) - 1, args.max_points).astype(int)
            policy_ball = policy_ball[idx]
            policy_ball_valid = policy_ball_valid[idx]
            policy_ball_age = policy_ball_age[idx]
            racket = racket[idx]

    anchor_pts = finite_xyz(racket[:1])
    anchor = anchor_pts[0] if len(anchor_pts) else None

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    if anchor is not None:
        draw_box(
            ax,
            (anchor[0] - args.done_x_range_m, anchor[0] + args.done_x_range_m),
            (anchor[1] - args.done_y_range_m, anchor[1] + args.done_y_range_m),
            (args.done_z_min_m, args.done_z_max_m),
            color="0.15",
        )
        ax.plot([], [], [], color="0.15", linewidth=1.8, label="ball done bounds")
        ax.scatter(
            anchor[0], anchor[1], anchor[2],
            color="black", marker="x", s=70, linewidths=2.0,
            label="racket anchor")
    if ee_x_limits and ee_y_limits and ee_z_limits:
        draw_box(
            ax,
            ee_x_limits,
            ee_y_limits,
            ee_z_limits,
            color="#0891b2",
            alpha=0.55,
        )
        ax.plot([], [], [], color="#0891b2", linewidth=1.8, label="EE workspace")

    plot_masked_line(
        ax, replay_ball, replay_valid, color="#f28e2b",
        label="replay ball valid", linewidth=2.2, alpha=0.95)
    plot_masked_line(
        ax, replay_ball, ~replay_valid, color="0.55",
        label="replay ball invalid/FOV lost", linewidth=1.8,
        alpha=0.65, linestyle="--")
    plot_masked_line(
        ax, policy_ball, policy_ball_valid, color="#e15759",
        label="policy ball obs valid", linewidth=1.5, alpha=0.65)
    plot_masked_line(
        ax, policy_ball, ~policy_ball_valid, color="#8b5cf6",
        label="policy ball obs held/stale", linewidth=1.3,
        alpha=0.55, linestyle="--")

    racket_pts = finite_xyz(racket)
    if len(racket_pts) >= 2:
        ax.plot(
            racket_pts[:, 0], racket_pts[:, 1], racket_pts[:, 2],
            color="#4e79a7", linewidth=2.2, alpha=0.95,
            label="racket/EE trajectory")

    mark_endpoint(ax, replay_ball[replay_valid], color="#f28e2b", prefix="ball")
    mark_endpoint(ax, racket_pts, color="#4e79a7", prefix="racket")

    title = "Safety replay 3D trajectory"
    if args.scenario:
        title += f": {args.scenario}"
    ax.set_title(title, pad=18)
    ax.set_xlabel("sim x (m)")
    ax.set_ylabel("sim y (m)")
    ax.set_zlabel("sim z (m)")
    ax.view_init(elev=24, azim=-58)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9)

    extra = []
    if anchor is not None:
        extra.append(np.array([
            [anchor[0] - args.done_x_range_m, anchor[1] - args.done_y_range_m, args.done_z_min_m],
            [anchor[0] + args.done_x_range_m, anchor[1] + args.done_y_range_m, args.done_z_max_m],
        ]))
    if ee_x_limits and ee_y_limits and ee_z_limits:
        extra.append(np.array([
            [ee_x_limits[0], ee_y_limits[0], ee_z_limits[0]],
            [ee_x_limits[1], ee_y_limits[1], ee_z_limits[1]],
        ]))
    set_equal_axes(ax, [replay_ball, policy_ball, racket, *extra])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot3d] wrote: {args.output}")

    if args.html_output is not None:
        write_interactive_html(
            args.html_output,
            scenario=args.scenario,
            replay_time=replay_t,
            replay_ball=replay_ball,
            replay_valid=replay_valid,
            policy_time=trace["time"],
            policy_ball=trace["ball"],
            policy_ball_valid=trace["ball_valid"],
            policy_ball_age=trace["ball_age"],
            racket=trace["racket"],
            anchor=anchor,
            done_z_min_m=args.done_z_min_m,
            done_z_max_m=args.done_z_max_m,
            done_x_range_m=args.done_x_range_m,
            done_y_range_m=args.done_y_range_m,
            ee_x_limits_m=ee_x_limits,
            ee_y_limits_m=ee_y_limits,
            ee_z_limits_m=ee_z_limits,
            max_points=args.max_points,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
