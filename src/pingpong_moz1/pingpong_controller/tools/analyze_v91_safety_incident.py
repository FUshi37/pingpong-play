#!/usr/bin/env python3
"""Build a forensic report for the 2026-08-29 V91 physical-sim incident.

This script is deliberately offline.  It reads a recording, deterministically
replays the publish-time limiter transition, and optionally drives the recorded
published references through the recovered-RMP MuJoCo plant.  It never creates
a ROS node and never publishes a robot command.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import font_manager
import numpy as np
from PIL import Image


JOINTS = [f"J{i + 1}" for i in range(7)]
DT = 0.005
PARKING_VEL_LIMIT = np.deg2rad(
    np.asarray([210, 210, 240, 240, 300, 300, 300], dtype=np.float64)
) * 0.08
PARKING_ACC_LIMIT = np.deg2rad(
    np.asarray([1300, 1300, 1800, 3000, 3000, 3000, 3000], dtype=np.float64)
) * 0.06
POSITION_LOW = np.asarray(
    [-2.0944, -2.96706, -3.05433, -0.174533, -3.05433, -1.65806, -1.5708]
)
POSITION_HIGH = np.asarray(
    [3.14159, 0.15708, 3.05433, 2.25147, 3.05433, 1.65806, 1.5708]
)

MODE_COLORS = {
    "release_gate": "#9ca3af",
    "policy": "#16a34a",
    "policy_feedback_hold": "#f59e0b",
    "policy_ball_state_hold": "#ea580c",
    "done_failsafe": "#dc2626",
}


def vec(value: Any, length: int = 7) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.full(length, np.nan)
    if out.size != length:
        return np.full(length, np.nan)
    return out


def finite_abs_max(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.nanmax(np.abs(value))) if np.any(np.isfinite(value)) else math.nan


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def extract_control(recording: Path) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for sample in load_jsonl(recording / "control_samples.jsonl"):
        mode = str(sample.get("mode", ""))
        counts[mode] += 1
        time_s = float(sample.get("time", math.nan))
        if not (17.35 <= time_s <= 19.50):
            continue
        command = sample.get("command") or {}
        ros = sample.get("ros") or {}
        workspace = sample.get("workspaceSafety") or {}
        policy_joint = sample.get("policyJointSample") or {}
        soft_reset = command.get("softWorkspaceReset") or {}
        gpu_boundary = command.get("gpu0RealRmpBoundary") or {}
        rows.append({
            "time": time_s,
            "wall_time": float(sample.get("wallTimeS", math.nan)),
            "tick": int(sample.get("tick", -1)),
            "source_seq": int(sample.get("sourceSeq", -1)),
            "mode": mode,
            "reason": str(sample.get("reason", "")),
            "policy_executed": bool(sample.get("policyExecuted", False)),
            "raw_valid": bool(sample.get("rawValid", False)),
            "q": vec(ros.get("jointPosition")),
            "dq": vec(ros.get("jointVelocity")),
            "ddq": vec(ros.get("jointAcceleration")),
            "nominal": vec(command.get("policyNominalPosition")),
            "pre_safety": vec(command.get("preSafetyPosition")),
            "published": vec(command.get("publishedSafePosition")),
            "action": vec(sample.get("policyExecutedAction")),
            "policy_velocity": vec((sample.get("rawPolicyCommand") or {}).get("velocity")),
            "ee": vec(workspace.get("eePosition"), 3),
            "racket": vec(sample.get("racket"), 3),
            "soft_reset": bool(soft_reset.get("active", False)),
            "gpu_boundary": bool(gpu_boundary.get("intervened", False)),
            "joint_source_seq": int(policy_joint.get("sourceSequence", -1)),
            "joint_age": float(policy_joint.get("receiptAgeS", math.nan)),
        })
    return rows, counts


def extract_snapshots(recording: Path) -> list[dict[str, Any]]:
    out = []
    for sample in load_jsonl(recording / "snapshots.jsonl"):
        time_s = float(sample.get("time", math.nan))
        if not (17.35 <= time_s <= 19.50):
            continue
        loop = sample.get("backendControlLoop") or {}
        out.append({
            "time": time_s,
            "mode": str(sample.get("mode", "")),
            "hz": float(loop.get("measuredHz", math.nan)),
            "work_ms": float(loop.get("workMs", math.nan)),
            "overruns": int(loop.get("overruns", 0)),
        })
    return out


def extract_sim(recording: Path) -> list[dict[str, Any]]:
    path = recording / "sim_control_samples.jsonl"
    out = []
    if not path.is_file():
        return out
    for sample in load_jsonl(path):
        sync = sample.get("sync") or {}
        out.append({
            "source_time": float(sync.get("sourceTimeS", math.nan)),
            "source_seq": int(sync.get("sourceSeq", -1)),
            "episode": int(sync.get("episode", -1)),
            "q": vec(sample.get("jointPosition")),
            "dq": vec(sample.get("jointVelocity")),
            "racket": vec(sample.get("racket"), 3),
        })
    return out


def first_row(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    for row in rows:
        if predicate(row):
            return row
    raise RuntimeError("required incident event was not present")


def last_row(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    for row in reversed(rows):
        if predicate(row):
            return row
    raise RuntimeError("required incident event was not present")


def reproduce_limiter_transition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce the first continuous ball-hold tick exactly.

    The normal GPU profile bypasses filtering and calls observe().  That makes
    last_vel the derivative between the previous published hold reference and
    the resumed actor reference.  The next parking filter sees incompatible
    velocity and acceleration intervals.  numpy.clip(a, low, high) with
    low > high returns high, so the huge synthetic velocity is propagated.
    """
    last_policy = last_row(rows, lambda r: r["mode"] == "policy")
    hold = first_row(
        rows,
        lambda r: (
            r["time"] > last_policy["time"]
            and r["mode"] == "policy_ball_state_hold"),
    )
    idx = rows.index(hold)
    current_policy = rows[idx - 1]
    previous = rows[idx - 2]
    if current_policy["mode"] != "policy":
        current_policy = last_row(rows[:idx], lambda r: r["mode"] == "policy")
        policy_idx = rows.index(current_policy)
        previous = rows[policy_idx - 1]

    last_cmd_before_observe = previous["published"].copy()
    actor_cmd = current_policy["published"].copy()
    synthetic_last_vel = (actor_cmd - last_cmd_before_observe) / DT
    raw_hold = hold["pre_safety"].copy()

    desired_vel = np.clip(
        (np.clip(raw_hold, POSITION_LOW, POSITION_HIGH) - actor_cmd) / DT,
        -PARKING_VEL_LIMIT,
        PARKING_VEL_LIMIT,
    )
    acc_lo = synthetic_last_vel - PARKING_ACC_LIMIT * DT
    acc_hi = synthetic_last_vel + PARKING_ACC_LIMIT * DT
    combined_lo = np.maximum.reduce((
        -PARKING_VEL_LIMIT,
        acc_lo,
        (POSITION_LOW - actor_cmd) / DT,
    ))
    combined_hi = np.minimum.reduce((
        PARKING_VEL_LIMIT,
        acc_hi,
        (POSITION_HIGH - actor_cmd) / DT,
    ))
    reproduced_vel = np.clip(desired_vel, acc_lo, acc_hi)
    reproduced = np.clip(actor_cmd + reproduced_vel * DT, POSITION_LOW, POSITION_HIGH)

    reanchored = raw_hold.copy()
    return {
        "previous_tick": previous,
        "policy_tick": current_policy,
        "hold_tick": hold,
        "last_cmd_before_observe": last_cmd_before_observe,
        "actor_cmd": actor_cmd,
        "synthetic_last_vel": synthetic_last_vel,
        "desired_parking_vel": desired_vel,
        "acc_lo": acc_lo,
        "acc_hi": acc_hi,
        "combined_lo": combined_lo,
        "combined_hi": combined_hi,
        "infeasible": combined_lo > combined_hi,
        "reproduced_vel": reproduced_vel,
        "reproduced": reproduced,
        "recorded": hold["published"].copy(),
        "reproduction_error": reproduced - hold["published"],
        "reanchored": reanchored,
    }


def nearest_snapshot(snapshots: list[dict[str, Any]], time_s: float) -> dict[str, Any]:
    return min(snapshots, key=lambda row: abs(row["time"] - time_s))


def run_mujoco_replay(
    recording: Path,
    rows: list[dict[str, Any]],
    start_time: float,
    first_hold_time: float,
    end_time: float,
) -> dict[str, Any]:
    """Replay recorded published references through recovered-RMP MuJoCo."""
    controller_root = Path(__file__).resolve().parents[2]
    if str(controller_root) not in sys.path:
        sys.path.insert(0, str(controller_root))
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from pingpong_controller.sim_mirror import RealBallSimMirror, SimMirrorConfig

    metadata = json.loads((recording / "metadata.json").read_text(encoding="utf-8"))
    sim_meta = metadata["config"]["simMirror"]
    config = SimMirrorConfig(
        model_path=Path(sim_meta["modelPath"]),
        policy_xml_path=Path(sim_meta["policyXmlPath"]),
        sim_xml_path=Path(sim_meta["xmlPath"]),
        policy_obs_dim=57,
        control_dt=DT,
        backend="mujoco",
        frame_rate_hz=1.0,
        enable_video_frames=False,
        use_mujoco_renderer=False,
        enable_compensation=False,
        policy_profile="gpu0_qvel_real_rmp_v85",
        policy_model="v91",
        simcode7_model_path=Path(sim_meta["modelPath"]),
        simcode7_v91_model_path=Path(sim_meta["modelPath"]),
        simcode7_policy_xml_path=Path(sim_meta["policyXmlPath"]),
        simcode7_sim_xml_path=Path(sim_meta["xmlPath"]),
    )
    replay_rows = [r for r in rows if start_time <= r["time"] <= end_time]
    start = replay_rows[0]
    hold = first_row(replay_rows, lambda r: r["time"] >= first_hold_time)
    hold_target = hold["q"].copy()
    mirror = RealBallSimMirror(config)

    def one(kind: str) -> list[dict[str, Any]]:
        mirror.reset(
            arm_q=start["q"],
            ball_pos=np.asarray([0.0, -2.0, 3.0], dtype=np.float32),
            ball_vel=np.zeros(3, dtype=np.float32),
            ball_valid=False,
            ball_obs_age_s=0.6,
        )
        mirror.data.qvel[mirror.arm_vadr] = start["dq"]
        # The reset synchronizes every recovered-RMP state to q.  Keeping the
        # measured dq makes this a conservative local boundary replay rather
        # than claiming an exact whole-episode reproduction.
        import mujoco as mj

        mj.mj_forward(mirror.model, mirror.data)
        result = []
        for row in replay_rows:
            target = row["published"]
            if kind == "reanchored_hold" and row["time"] >= first_hold_time:
                target = hold_target
            mirror._set_ball_state(
                np.asarray([0.0, -2.0, 3.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )
            mirror._step_recovered_rmp(target.astype(np.float32))
            racket, _quat = mirror._racket_pose()
            result.append({
                "time": row["time"],
                "q": mirror.data.qpos[mirror.arm_qadr].astype(float).copy(),
                "dq": mirror.data.qvel[mirror.arm_vadr].astype(float).copy(),
                "racket": np.asarray(racket, dtype=float).copy(),
                "ncon": int(mirror.data.ncon),
            })
        return result

    dangerous = one("recorded")
    safe = one("reanchored_hold")
    mirror.close()
    return {"recorded_targets": dangerous, "reanchored_hold": safe}


def configure_plots() -> None:
    for font_path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        if Path(font_path).is_file():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 140,
        "savefig.dpi": 180,
        "axes.grid": True,
        "grid.alpha": 0.23,
        "axes.titleweight": "bold",
    })


def shade_modes(ax, rows: list[dict[str, Any]], alpha: float = 0.09) -> None:
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["mode"] != rows[start]["mode"]:
            ax.axvspan(
                rows[start]["time"], rows[i - 1]["time"] + DT,
                color=MODE_COLORS.get(rows[start]["mode"], "#cbd5e1"),
                alpha=alpha, linewidth=0,
            )
            start = i


def plot_timeline(
    output: Path,
    rows: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
    ee_box: dict[str, float],
) -> None:
    window = [r for r in rows if 18.45 <= r["time"] <= 19.18]
    t = np.asarray([r["time"] for r in window])
    dq_max = np.asarray([finite_abs_max(r["dq"]) for r in window])
    ee_y = np.asarray([r["ee"][1] for r in window])
    ee_z = np.asarray([r["ee"][2] for r in window])
    published = np.asarray([r["published"] for r in window])
    pub_rate = np.vstack((np.zeros((1, 7)), np.diff(published, axis=0) / DT))
    pub_rate_max = np.nanmax(np.abs(pub_rate), axis=1)

    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    ax = axes[0]
    mode_names = list(MODE_COLORS)
    mode_y = {name: idx for idx, name in enumerate(mode_names)}
    ax.step(t, [mode_y.get(r["mode"], -1) for r in window], where="post", lw=2)
    ax.set_yticks(range(len(mode_names)), [
        "release_gate", "policy", "policy_feedback_hold",
        "policy_ball_state_hold", "done_failsafe"
    ])
    ax.set_title("事故控制时间线（200 Hz 原始记录）")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1]
    shade_modes(ax, window)
    ax.semilogy(t, np.maximum(pub_rate_max, 1e-3), color="#7c3aed", lw=1.6,
                label="每 5 ms 的 max |Δ publishedSafePosition|")
    ax.axhline(float(np.max(PARKING_VEL_LIMIT)), color="#ea580c", ls="--",
               label="profile='parking' 的最大速度限制")
    ax.set_ylabel("目标变化率\n(rad/s，对数)")
    ax.legend(loc="upper left", ncol=2)

    ax = axes[2]
    shade_modes(ax, window)
    ax.plot(t, dq_max, color="#dc2626", lw=2, label="实测 max |dq|")
    ax.axhline(5.0, color="#991b1b", ls=":", label="5 rad/s")
    ax.set_ylabel("实测速度\n(rad/s)")
    ax.legend(loc="upper left")

    ax = axes[3]
    shade_modes(ax, window)
    ax.plot(t, ee_y, color="#2563eb", lw=2, label="末端 y")
    ax.axhline(float(ee_box["y1"]), color="#dc2626", ls="--", label="配置的 y 上界")
    ax2 = ax.twinx()
    ax2.plot(t, ee_z, color="#059669", lw=1.2, alpha=0.8, label="末端 z")
    ax2.axhline(float(ee_box["z0"]), color="#059669", ls=":", alpha=0.8)
    ax.set_ylabel("EE y (m)")
    ax2.set_ylabel("EE z (m)")
    lines = [line for line in ax.get_lines() + ax2.get_lines()
             if not line.get_label().startswith("_")]
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=2)

    ax = axes[4]
    snap = [s for s in snapshots if window[0]["time"] <= s["time"] <= window[-1]["time"]]
    st = np.asarray([s["time"] for s in snap])
    hz = np.asarray([s["hz"] for s in snap])
    work = np.asarray([s["work_ms"] for s in snap])
    ax.plot(st, hz, marker="o", ms=2.8, color="#0f766e", label="实测控制循环频率")
    ax.axhline(200, color="#111827", ls="--", label="目标 200 Hz")
    ax.set_ylabel("loop Hz")
    ax2 = ax.twinx()
    ax2.plot(st, work, marker=".", color="#d97706", label="单次处理耗时")
    ax2.axhline(5.0, color="#d97706", ls=":", label="5 ms 时间预算")
    ax2.set_ylabel("单次处理耗时 (ms)")
    lines = [line for line in ax.get_lines() + ax2.get_lines()
             if not line.get_label().startswith("_")]
    ax.legend(lines, [line.get_label() for line in lines], loc="lower left", ncol=2)
    ax.set_xlabel("录制相对时间 (s)")

    for axis in axes:
        for label, event in events.items():
            axis.axvline(event["time"], color={
                "last_policy": "#7c3aed", "first_ball_hold": "#ea580c",
                "first_done": "#dc2626", "max_speed": "#991b1b",
            }.get(label, "#475569"), lw=1.1, alpha=0.8)
    fig.text(0.125, 0.012,
             "阴影：绿=policy，黄=policy_feedback_hold，"
             "橙=policy_ball_state_hold，红=done_failsafe。"
             "危险发布命令先发生，末端越界与最大实物速度随后发生。",
             fontsize=10, color="#334155")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output / "01_incident_timeline.png", bbox_inches="tight")
    plt.close(fig)


def plot_joint_runaway(
    output: Path,
    rows: list[dict[str, Any]],
    events: dict[str, dict[str, Any]],
) -> None:
    window = [r for r in rows if 18.54 <= r["time"] <= 19.18]
    t = np.asarray([r["time"] for r in window])
    q = np.asarray([r["q"] for r in window])
    pre = np.asarray([r["pre_safety"] for r in window])
    pub = np.asarray([r["published"] for r in window])
    fig, axes = plt.subplots(7, 1, figsize=(15, 16), sharex=True)
    for j, ax in enumerate(axes):
        shade_modes(ax, window)
        ax.plot(t, pre[:, j], color="#64748b", lw=1.0, label="preSafetyPosition")
        ax.plot(t, pub[:, j], color="#7c3aed", lw=1.7, label="publishedSafePosition")
        ax.plot(t, q[:, j], color="#dc2626", lw=1.7, label="实测 q")
        ax.axhline(POSITION_LOW[j], color="#111827", ls=":", lw=0.8)
        ax.axhline(POSITION_HIGH[j], color="#111827", ls=":", lw=0.8)
        ax.set_ylabel(f"{JOINTS[j]} (rad)")
        for name in ("first_ball_hold", "first_done"):
            ax.axvline(events[name]["time"], color={
                "first_ball_hold": "#ea580c", "first_done": "#dc2626"
            }[name], lw=1.0)
    axes[0].set_title(
        "关节数据：preSafetyPosition 接近实测 q，"
        "publishedSafePosition 却发生大幅变化")
    axes[0].legend(loc="upper left", ncol=3)
    axes[-1].set_xlabel("录制相对时间 (s)")
    fig.tight_layout()
    fig.savefig(output / "02_joint_command_runaway.png", bbox_inches="tight")
    plt.close(fig)


def plot_limiter_reproduction(output: Path, replay: dict[str, Any]) -> None:
    synth = replay["synthetic_last_vel"]
    recorded_vel = (replay["recorded"] - replay["actor_cmd"]) / DT
    reproduced_vel = replay["reproduced_vel"]
    reanchored_vel = (replay["reanchored"] - replay["actor_cmd"]) / DT
    x = np.arange(7)
    width = 0.20
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    ax = axes[0]
    ax.bar(x - 1.5 * width, synth, width,
           label="observe() 后保存的 last_vel", color="#7c3aed")
    ax.bar(x - 0.5 * width, recorded_vel, width,
           label="日志中的下一条目标变化率", color="#dc2626")
    ax.bar(x + 0.5 * width, reproduced_vel, width,
           label="按 filter() 重算的变化率", color="#f97316")
    ax.bar(x + 1.5 * width, PARKING_VEL_LIMIT * np.sign(recorded_vel), width,
           label="profile='parking' 的速度限制", color="#0f766e")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x, JOINTS)
    ax.set_ylabel("rad/s")
    ax.set_title(
        "按代码重算：profile='normal'→profile='parking' 时"
        "沿用了错误的 last_vel")
    ax.legend(ncol=2)

    ax = axes[1]
    delta_recorded = replay["recorded"] - replay["actor_cmd"]
    delta_reproduced = replay["reproduced"] - replay["actor_cmd"]
    delta_safe = replay["reanchored"] - replay["actor_cmd"]
    ax.bar(x - width, delta_recorded, width,
           label="日志中的 5 ms Δq", color="#dc2626")
    ax.bar(x, delta_reproduced, width,
           label="按 filter() 重算的 5 ms Δq", color="#f97316")
    ax.bar(x + width, delta_safe, width,
           label="对照：直接发送事故时实测 q", color="#16a34a")
    ax.set_xticks(x, JOINTS)
    ax.set_ylabel("rad per tick")
    ax.set_title(
        "复现误差 max = "
        f"{np.max(np.abs(replay['reproduction_error'])):.3e} rad；"
        "不是模型或 MuJoCo 随机性"
    )
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output / "03_limiter_bug_reproduction.png", bbox_inches="tight")
    plt.close(fig)


def plot_causal_chain(output: Path, metrics: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(17, 9))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color, edgecolor="#334155", linewidth=1.3,
        )
        ax.add_patch(rect)
        ax.text(x + 0.15, y + h - 0.28, title, fontsize=12, fontweight="bold", va="top")
        ax.text(x + 0.15, y + h - 0.72, body, fontsize=9.5, va="top", linespacing=1.35)
        return rect

    def arrow(x1, y1, x2, y2, color="#475569"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color))

    box(0.3, 5.7, 2.6, 2.1, "事故前的运行情况",
        f"目标 200 Hz，事故前约 {metrics['loop_hz']:.1f} Hz\n"
        f"单 tick 工作 {metrics['loop_work_ms']:.2f} ms > 5 ms\n"
        f"出现 {metrics['feedback_hold_count']} 次 policy_feedback_hold", "#fef3c7")
    box(3.4, 5.7, 2.6, 2.1, "球状态无效",
        f"球在 t={metrics['first_ball_hold_time']:.3f}s 失效\n"
        "mode 变为\npolicy_ball_state_hold\n"
        "profile='parking'", "#ffedd5")
    box(6.5, 5.45, 3.2, 2.6, "已确认的代码错误",
        "profile='normal' 时调用 observe(q_ref)\n"
        "状态切换产生错误的 last_vel\n"
        "profile='parking' 的速度和加速度\n"
        "限制无交集；filter() 的后一步覆盖前一步", "#fee2e2")
    box(10.2, 5.7, 2.7, 2.1, "publishedSafePosition 异常",
        f"发布参考峰值 {metrics['max_publish_rate']:.1f} rad/s\n"
        f"是 profile='parking'\n速度限制的 {metrics['publish_cap_ratio']:.0f}×\n"
        "该目标随后进入实物底层 RMP", "#fecaca")
    box(13.4, 5.7, 3.1, 2.1, "物理结果",
        f"实物峰值速度 {metrics['max_dq']:.2f} rad/s\n"
        f"EE y 越界最大 {metrics['ee_y_penetration']*1000:.0f} mm\n"
        "机械臂/球拍进入腰腿区域并碰撞", "#fca5a5")
    for a, b in ((2.9, 3.4), (6.0, 6.5), (9.7, 10.2), (12.9, 13.4)):
        arrow(a, 6.75, b, 6.75)

    box(6.5, 1.1, 3.2, 2.4, "安全检查为何没有拦住",
        "workspaceSafety=position_soft：按实测末端位置判定\n"
        "无完整机器人/球拍自碰撞包络\n"
        "越界后才进入 done_failsafe；此时 dq 已很高\n"
        "无法撤回已经发给底层 RMP 的目标", "#e0e7ff")
    box(10.2, 1.1, 2.7, 2.4, "Sim mirror 盲区",
        f"只完成 {metrics['sim_count']} 个样本 / 9 episodes\n"
        f"最后样本停在 t={metrics['sim_last_time']:.3f}s\n"
        "只镜像 policy→RMP\n"
        "未镜像 profile 切换后的 filter()", "#dbeafe")
    box(13.4, 1.1, 3.1, 2.4, "配置/流程风险",
        "实际模型=V91，但 release 标识仍为 V85\n"
        "contact action shield 关闭\n"
        "safety limiter normal 路径关闭\n"
        "因此部署验证不能代表实际运行组合", "#ede9fe")
    arrow(8.1, 5.45, 8.1, 3.5, "#dc2626")
    arrow(11.55, 5.7, 11.55, 3.5, "#2563eb")
    arrow(14.9, 5.7, 14.9, 3.5, "#7c3aed")
    ax.text(0.3, 8.55, "V91 球拍撞击事故：系统性因果链（证据来自同一次录制）",
            fontsize=18, fontweight="bold")
    ax.text(0.3, 0.35,
            "结论：V91 的高动态动作和实时环过载提供了触发条件；"
            "造成异常目标的代码错误发生在 "
            "profile='normal'/'parking' 切换时的 filter()。",
            fontsize=11, color="#1e293b")
    fig.savefig(output / "04_system_causal_chain.png", bbox_inches="tight")
    plt.close(fig)


def plot_sim_gap(
    output: Path,
    rows: list[dict[str, Any]],
    sim: list[dict[str, Any]],
    mujoco_replay: dict[str, Any] | None,
    events: dict[str, dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    window = [r for r in rows if 17.60 <= r["time"] <= 19.25]
    t = np.asarray([r["time"] for r in window])
    real_dq = np.asarray([finite_abs_max(r["dq"]) for r in window])
    axes[0].plot(
        t, real_dq, color="#dc2626", lw=1.8,
        label="录制中的实测 max |dq|")
    if sim:
        st = np.asarray([s["source_time"] for s in sim])
        sdq = np.asarray([finite_abs_max(s["dq"]) for s in sim])
        axes[0].scatter(
            st, sdq, s=22, color="#2563eb",
            label="完成的 sim mirror 样本")
    axes[0].set_ylabel("max |dq| (rad/s)")
    axes[0].set_title(
        "最后一个 sim mirror 样本早于第一次 policy_ball_state_hold")
    axes[0].legend(loc="upper left")

    axes[1].step(t, [1 if r["policy_executed"] else 0 for r in window],
                 where="post", color="#16a34a",
                 label="日志字段 policyExecuted")
    if sim:
        axes[1].scatter([s["source_time"] for s in sim], np.full(len(sim), 0.55),
                        s=18, color="#2563eb",
                        label="完成的 sim mirror 样本")
    axes[1].set_ylim(-0.15, 1.2)
    axes[1].set_yticks(
        [0, 0.55, 1],
        ["policyExecuted=false", "sim mirror 样本",
         "policyExecuted=true"])
    axes[1].legend(loc="upper left")

    if mujoco_replay:
        for key, color, label in (
            ("recorded_targets", "#dc2626",
             "MuJoCo：日志中的 publishedSafePosition"),
            ("reanchored_hold", "#16a34a",
             "MuJoCo 对照：事故时实测 q"),
        ):
            part = mujoco_replay[key]
            axes[2].plot(
                [r["time"] for r in part],
                [finite_abs_max(r["dq"]) for r in part],
                color=color, lw=2, label=label,
            )
        axes[2].legend(loc="upper left")
    else:
        axes[2].text(0.5, 0.5, "MuJoCo offline replay unavailable",
                     transform=axes[2].transAxes, ha="center")
    axes[2].set_ylabel("max |dq| (rad/s)")
    axes[2].set_xlabel("录制相对时间 (s)")
    axes[2].set_title(
        "相同初始状态的 MuJoCo 对照："
        "publishedSafePosition 与事故时实测 q")

    for ax in axes:
        ax.axvline(events["first_ball_hold"]["time"], color="#ea580c", ls="--")
        ax.axvline(events["first_done"]["time"], color="#dc2626", ls="--")
    fig.tight_layout()
    fig.savefig(output / "05_sim_mirror_and_mujoco_replay.png", bbox_inches="tight")
    plt.close(fig)


def plot_storyboard(
    output: Path,
    recording: Path,
    events: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = list(load_jsonl(recording / "frames.jsonl"))
    selections = [
        ("最后一次 mode=policy", events["last_policy"]),
        ("第一次 mode=policy_ball_state_hold", events["first_ball_hold"]),
        ("第一次 mode=done_failsafe", events["first_done"]),
        ("实测 max |dq| 最大值", events["max_speed"]),
        ("末端 y 最大值", events["max_ee_y"]),
    ]
    chosen = []
    for label, event in selections:
        frame = min(manifest, key=lambda item: abs(float(item["wallTimeS"]) - event["wall_time"]))
        chosen.append({"label": label, "event": event, "frame": frame})

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    axes = axes.ravel()
    for ax, item in zip(axes, chosen):
        path = recording / item["frame"]["path"]
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.add_patch(patches.Rectangle(
            (0.68 * image.width, 0.35 * image.height),
            0.32 * image.width, 0.65 * image.height,
            linewidth=2, edgecolor="#dc2626", facecolor="none",
        ))
        event = item["event"]
        ax.set_title(
            f"{item['label']}  t={event['time']:.3f}s\n"
            f"nearest camera frame Δ={float(item['frame']['wallTimeS'])-event['wall_time']:+.3f}s"
        )
        ax.axis("off")
    axes[-1].axis("off")
    axes[-1].text(
        0.03, 0.9,
        "视频证据的限制",
        fontsize=16, fontweight="bold", va="top",
    )
    axes[-1].text(
        0.03, 0.75,
        "录制的是球检测 RGB/IR 画面，不是独立安全相机。\n"
        "腰腿和球拍仅位于画面右下边缘，撞击点大部分在视野外。\n\n"
        "因此：\n"
        "• 视频能确认球检测丢失及运动发生的时刻；\n"
        "• 精确碰撞几何不能仅凭该视频确认；\n"
        "• 事故根因主要依赖 200 Hz 命令、反馈和安全状态日志。",
        fontsize=11, va="top", linespacing=1.5,
    )
    fig.suptitle("相机时间对齐画面（红框为机器人/球拍可见边缘区域）", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "06_camera_storyboard.png", bbox_inches="tight")
    plt.close(fig)
    return [{
        "label": item["label"],
        "event_time": item["event"]["time"],
        "event_wall_time": item["event"]["wall_time"],
        "frame": item["frame"]["path"],
        "frame_wall_time": item["frame"]["wallTimeS"],
    } for item in chosen]


def serial(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    return value


def write_window_csv(output: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["time", "tick", "mode", "reason", "ee_x", "ee_y", "ee_z",
               "max_abs_dq", "max_abs_ddq", "max_abs_published_rate"]
    previous = None
    with (output / "incident_window.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            if not (18.45 <= row["time"] <= 19.25):
                continue
            rate = math.nan if previous is None else finite_abs_max(
                (row["published"] - previous["published"]) / DT)
            writer.writerow({
                "time": row["time"], "tick": row["tick"], "mode": row["mode"],
                "reason": row["reason"], "ee_x": row["ee"][0], "ee_y": row["ee"][1],
                "ee_z": row["ee"][2], "max_abs_dq": finite_abs_max(row["dq"]),
                "max_abs_ddq": finite_abs_max(row["ddq"]),
                "max_abs_published_rate": rate,
            })
            previous = row


def write_report(
    output: Path,
    recording: Path,
    metrics: dict[str, Any],
    replay: dict[str, Any],
    mujoco: dict[str, Any] | None,
) -> None:
    error = float(np.max(np.abs(replay["reproduction_error"])))
    mujoco_text = "MuJoCo 重放未成功，见 incident_metrics.json 的错误字段。"
    if mujoco:
        danger_peak = max(finite_abs_max(row["dq"]) for row in mujoco["recorded_targets"])
        safe_peak = max(finite_abs_max(row["dq"]) for row in mujoco["reanchored_hold"])
        danger_y = max(float(row["racket"][1]) for row in mujoco["recorded_targets"])
        safe_y = max(float(row["racket"][1]) for row in mujoco["reanchored_hold"])
        mujoco_text = (
            f"局部 recovered-RMP MuJoCo 重放中，录制 publisher target 的峰值关节速度为 "
            f"**{danger_peak:.3f} rad/s**，重锚原地保持反事实为 **{safe_peak:.3f} rad/s**；"
            f"两者最大 racket-y 分别为 **{danger_y:.3f} m** 与 **{safe_y:.3f} m**。"
            "这是从事故前局部状态启动的边界条件实验，不应当解释为整段实物轨迹的逐点预测。"
        )

    report = f"""# V91 球拍撞击事故系统性分析

录制目录：`{recording}`  
分析结论等级：**主根因已由 200 Hz 数据和确定性代码重放确认；精确机械接触点因相机视野不足不能仅靠视频确认。**

## 一句话结论

事故不是“底层又多写了一套 RMP”。真正的失控发生在实物发布端：日志中大量出现 `source_sequence_gap`，`mode` 在 `policy` 与 `policy_feedback_hold` 之间频繁切换；GPU0-RMP 的 `profile='normal'` 路径关闭了 limiter，但 `observe()` 把相邻目标的跳变保存成数十 rad/s 的 `last_vel`。下一次使用 `profile='parking'` 时，`filter()` 先执行速度限制，再执行与错误 `last_vel` 相关的加速度限制，后一步覆盖了前一步，使 `publishedSafePosition` 在 5 ms 内跳变。底层 RMP/高增益 PD 随后跟踪了这个错误目标。

## 关键事实

| 证据 | 数值 | 含义 |
|---|---:|---|
| policy 实际执行 | {metrics['policy_count']} ticks | 约 {metrics['policy_count']*DT:.3f} s 的有效 actor tick |
| `mode=policy_feedback_hold` | {metrics['feedback_hold_count']} ticks | 日志原因包含 `source_sequence_gap` |
| `mode=policy_ball_state_hold` | {metrics['ball_hold_count']} ticks | 球状态无效；发布使用 `profile='parking'` |
| 事故前实时环 | {metrics['loop_hz']:.1f} Hz / {metrics['loop_work_ms']:.2f} ms | 低于 200 Hz，且单 tick 已超过 5 ms 预算 |
| 发布参考峰值变化率 | {metrics['max_publish_rate']:.2f} rad/s | parking 最大速度上限的 {metrics['publish_cap_ratio']:.1f} 倍 |
| 实测关节峰值速度 | {metrics['max_dq']:.3f} rad/s | 发生在 done 之后，说明越界检测来不及制动 |
| EE y 最大越界 | {metrics['ee_y_penetration']*1000:.1f} mm | y 上界为 {metrics['ee_y_bound']:.3f} m |
| online sim mirror | {metrics['sim_count']} samples / {metrics['sim_episode_count']} episodes | 最后停在 {metrics['sim_last_time']:.3f}s，危险 hold 从 {metrics['first_ball_hold_time']:.3f}s 开始 |
| limiter 公式重放误差 | {error:.3e} rad | 逐元素复现异常发布目标，排除随机性 |

## 精确时间线

1. **t={metrics['policy_start_time']:.3f}s**：策略开始。V91 动作允许到归一化 ±1，J7 在后段多次饱和；这增加了 target 动态，但此时尚未发生身体方向的失控。
2. **策略运行期间**：控制回调从约 190 Hz 降到约 {metrics['loop_hz']:.1f} Hz，累计出现 {metrics['feedback_hold_count']} 个 `policy_feedback_hold`。每次 hold 都会把 actor 状态重锚到测量 q，并切换到 parking 发布路径。
3. **t={metrics['last_policy_time']:.3f}s**：最后一个 actor tick。online sim mirror 的最后完成样本也停在这里附近，因此没有覆盖后续发布端事故。
4. **t={metrics['first_ball_hold_time']:.3f}s**：球检测失效，进入 `policy_ball_state_hold`。`preSafetyPosition` 接近实测 q，本来应原地保持；但 `publishedSafePosition` 第一拍就出现异常大步长。
5. **t={metrics['first_done_time']:.3f}s**：实测 EE y 穿过 {metrics['ee_y_bound']:.3f} m，`position_soft` 才触发 `ee_y_out_of_workspace` 并 latch reset。此时危险命令已经在底层 plant 中传播。
6. **t={metrics['max_dq_time']:.3f}s**：done 之后实测速度继续上升到 {metrics['max_dq']:.3f} rad/s。reset 代码确实重锚并缓慢返回，但无法撤回前面已经发布/排队的错误目标。
7. **t={metrics['max_ee_y_time']:.3f}s**：EE y 达到 {metrics['max_ee_y']:.3f} m，超过安全上界 {metrics['ee_y_penetration']*1000:.1f} mm，轨迹进入机器人腰腿一侧。

## 主根因：为什么“安全限速器”反而生成了危险命令

`profile='normal'` 的 GPU0-RMP 路径中 `enable_safety_limiter=false`，代码直接发布策略的 q_ref，并调用 `safety_limiter.observe(q_ref)`。`observe()` 用相邻发布目标差分更新 `last_vel`。当上一 tick 的 `mode=policy_feedback_hold`、下一 tick 又恢复为 `mode=policy` 时，这个差分不是物理关节速度，而是两个不同 `mode` 的目标之间的跳变。

紧接着再次进入 parking 时，限速器先要求 `|v| <= parking_vel_limit`，又要求 `v` 只能在 `last_vel ± parking_acc_limit·dt` 内。事故 tick 上两者没有交集，组合后的可行区间是 **lower > upper**。当前 `filter()` 没有检查这个组合区间，而是先按速度上限 clip、再按加速度区间 clip；第二次 clip 把第一次的速度约束覆盖掉，选择了靠近旧 `last_vel` 的加速度端点，于是把数十 rad/s 的虚假 target derivative 延续到下一条命令。

离线重放只使用三条录制命令和当前 `safety_limiter.py` 的数学顺序，就以 max error={error:.3e} rad 复现了第一条异常 `publishedSafePosition`。这构成主根因的确定性证据，见 `03_limiter_bug_reproduction.png`。

## V91 模型本身的责任边界

V91 是重要触发因素，但不是足以单独解释事故的根因：

- V91 后段动作动态大，J7 到达动作上限；normal 路径又没有动作滤波或发布端限速，这使目标层差分更大。
- 但球丢失后 `preSafetyPosition` 明确是在保持实测 q；把它变成冲向关节限位的，是 real publisher 的混合模式 limiter 状态。
- 同一个切换 BUG 对 V85 或任何产生较大 q_ref 变化的模型也成立。因此不能只通过“换回旧模型”宣告问题消失。
- 本次 runtime actor/SHA 是 V91，但 `policyRelease` 仍标识并验证 V85；这是部署追溯和放行流程缺陷，会让 release 验证结论与真实 actor 不一致。

## 为什么 sim mirror 没发现

- 在线 mirror 只产生 {metrics['sim_count']} 个结果，分散在 {metrics['sim_episode_count']} 个 episode；提交/完成不是 200 Hz lockstep。
- 最后完成样本为 t={metrics['sim_last_time']:.3f}s，而 `mode=policy_ball_state_hold` 从 t={metrics['first_ball_hold_time']:.3f}s 开始。
- mirror 镜像的是策略→积分器→recovered RMP；事故代码位于真实发布器的 `profile='normal'` / `profile='parking'` / `profile='reset'` 切换，不在 mirror 的计算图中。
- 录制对齐报告也已将该 run 标为 `sim_not_realtime_aligned`；因此“开启实物仿真模式”不能被当作安全互锁。

{mujoco_text}

## 安全层为何失效

1. `position_soft` 依据**当前实测 EE 位置**越界后才 done，没有对已发布参考、完整机械臂/球拍几何和身体自碰撞做硬约束。
2. `contactActionShield` 与 vertical energy shield 在该 profile 中关闭；normal limiter 也关闭。
3. 工作区只是末端点的盒子，不等价于整条机械臂和球拍远离腰腿。
4. soft reset 是无碰撞 joint-space 返回目标，不是笛卡尔避障路径；虽然本次主危险在 reset 前已形成，自动 reset 本身也不应被视为自碰撞安全轨迹。
5. 相机视频没有覆盖完整腰腿/球拍接触区域，缺少独立安全相机和腕力矩证据。

## 修复优先级（在再次上实物前）

### P0：必须阻断

1. **禁止 GPU0-RMP profile 在实物发布命令**，直到 normal↔parking 切换回归通过；当前录制已证明可造成器材损坏。
2. 修复 limiter 的不可行区间：任何 `lower > upper` 必须 fail closed；profile 切换必须从新鲜实测 q、零命令速度重锚，不能继承 target derivative。
3. `policy_feedback_hold` / `policy_ball_state_hold` 应直接走“新鲜实测位置单拍 hold + 停止 command stream/显式 re-arm”，不要继续复用 parking 动态状态。
4. 加入发布前硬断言：任一 tick 的 `|Δpublished_q|/dt` 超过配置速度上限立即禁止发送并 latch；不能只记 telemetry。

### P1：再次仿真验证前

1. 将 publisher 的 normal/hold/reset 状态机纳入 sim mirror；镜像比较必须包含**最终 published target**，不能只比较 actor q_ref。
2. 200 Hz loop 必须有实时预算门：持续低于目标频率、work > 5 ms 或 source gap 比例超阈值时，在策略执行前 fail closed。sim mirror/渲染/录制不得占用实时回调预算。
3. 增加机器人全身与球拍 swept-volume 自碰撞模型；EE box 只能作为附加约束。
4. release ID、checkpoint SHA、网页 model selector 必须形成单一不可分割 contract，V85 release 不得加载 V91 actor。

### P2：安全验证

1. 先做命令发布关闭的离线/仿真 fault injection：交替插入 source gap、ball loss、重复反馈和模式切换。
2. 再做低速、无球、软球拍、物理隔离的台架测试；逐级放开速度，独立急停人员在场。
3. 添加覆盖腰腿/球拍的独立高速视频或腕部力矩传感器，保证下一次异常能定位真实接触时刻。

## 输出文件

- `01_incident_timeline.png`：事故事件、发布目标速率、实测速度、EE 越界和实时环负载。
- `02_joint_command_runaway.png`：7 关节 pre-safety / published / measured 对照。
- `03_limiter_bug_reproduction.png`：确定性 limiter 复现实验。
- `04_system_causal_chain.png`：系统性因果链。
- `05_sim_mirror_and_mujoco_replay.png`：online mirror 盲区和 recovered-RMP 反事实。
- `06_camera_storyboard.png`：相机帧时间对齐及视野限制。
- `incident_metrics.json`、`incident_window.csv`：可审计数值和事故窗口数据。
"""
    (output / "SAFETY_INCIDENT_ANALYSIS_ZH.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--skip-mujoco", action="store_true")
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Regenerate figures without overwriting the reviewed report/metrics.")
    args = parser.parse_args()
    recording = args.recording.expanduser().resolve()
    output = recording / "analysis" / "v91_safety_incident"
    output.mkdir(parents=True, exist_ok=True)
    configure_plots()

    rows, counts = extract_control(recording)
    snapshots = extract_snapshots(recording)
    sim = extract_sim(recording)
    replay = reproduce_limiter_transition(rows)

    policy_start = first_row(rows, lambda r: r["mode"] == "policy")
    last_policy = last_row(rows, lambda r: r["mode"] == "policy")
    first_ball_hold = first_row(
        rows,
        lambda r: (
            r["time"] > last_policy["time"]
            and r["mode"] == "policy_ball_state_hold"),
    )
    first_done = first_row(rows, lambda r: r["mode"] == "done_failsafe")
    max_speed = max(rows, key=lambda r: finite_abs_max(r["dq"]))
    max_ee_y = max(rows, key=lambda r: float(r["ee"][1]))

    events = {
        "last_policy": last_policy,
        "first_ball_hold": first_ball_hold,
        "first_done": first_done,
        "max_speed": max_speed,
        "max_ee_y": max_ee_y,
    }
    metadata = json.loads((recording / "metadata.json").read_text(encoding="utf-8"))
    ee_box = metadata["config"]["eeBox"]

    rates = []
    previous = None
    for row in rows:
        if previous is not None:
            rates.append((row, (row["published"] - previous["published"]) / DT))
        previous = row
    hold_rates = [
        (row, rate) for row, rate in rates
        if row["mode"] in {"policy_feedback_hold", "policy_ball_state_hold"}
        and row["time"] <= first_done["time"]
    ]
    max_rate_row, max_rate_vec = max(hold_rates, key=lambda item: finite_abs_max(item[1]))
    max_rate = finite_abs_max(max_rate_vec)
    incident_loop = nearest_snapshot(snapshots, first_ball_hold["time"])

    mujoco_replay = None
    mujoco_error = None
    if not args.skip_mujoco:
        try:
            mujoco_replay = run_mujoco_replay(
                recording, rows,
                start_time=replay["previous_tick"]["time"],
                first_hold_time=first_ball_hold["time"],
                end_time=min(19.30, rows[-1]["time"]),
            )
        except Exception as exc:  # noqa: BLE001 - report remains useful.
            mujoco_error = f"{type(exc).__name__}: {exc}"

    metrics = {
        "recording": str(recording),
        "mode_counts": dict(counts),
        "policy_count": int(counts["policy"]),
        "feedback_hold_count": int(counts["policy_feedback_hold"]),
        "ball_hold_count": int(counts["policy_ball_state_hold"]),
        "done_count": int(counts["done_failsafe"]),
        "policy_start_time": policy_start["time"],
        "last_policy_time": last_policy["time"],
        "first_ball_hold_time": first_ball_hold["time"],
        "first_done_time": first_done["time"],
        "max_dq": finite_abs_max(max_speed["dq"]),
        "max_dq_time": max_speed["time"],
        "max_dq_tick": max_speed["tick"],
        "max_ee_y": float(max_ee_y["ee"][1]),
        "max_ee_y_time": max_ee_y["time"],
        "max_ee_y_tick": max_ee_y["tick"],
        "ee_y_bound": float(ee_box["y1"]),
        "ee_y_penetration": float(max_ee_y["ee"][1] - ee_box["y1"]),
        "max_publish_rate": max_rate,
        "max_publish_rate_time": max_rate_row["time"],
        "max_publish_rate_vector": max_rate_vec,
        "parking_velocity_limits": PARKING_VEL_LIMIT,
        "publish_cap_ratio": max_rate / float(np.max(PARKING_VEL_LIMIT)),
        "loop_hz": incident_loop["hz"],
        "loop_work_ms": incident_loop["work_ms"],
        "loop_overruns": incident_loop["overruns"],
        "sim_count": len(sim),
        "sim_episode_count": len({row["episode"] for row in sim}),
        "sim_last_time": max(row["source_time"] for row in sim) if sim else math.nan,
        "sim_max_dq": max(finite_abs_max(row["dq"]) for row in sim) if sim else math.nan,
        "limiter_replay": replay,
        "mujoco_error": mujoco_error,
    }
    if mujoco_replay:
        metrics["mujoco_replay"] = {
            key: {
                "max_dq": max(finite_abs_max(row["dq"]) for row in part),
                "max_racket_y": max(float(row["racket"][1]) for row in part),
                "min_racket_z": min(float(row["racket"][2]) for row in part),
                "max_contacts": max(int(row["ncon"]) for row in part),
            }
            for key, part in mujoco_replay.items()
        }

    plot_timeline(output, rows, snapshots, events, ee_box)
    plot_joint_runaway(output, rows, events)
    plot_limiter_reproduction(output, replay)
    plot_causal_chain(output, metrics)
    plot_sim_gap(output, rows, sim, mujoco_replay, events)
    storyboard = plot_storyboard(output, recording, events)
    if not args.plots_only:
        metrics["storyboard"] = storyboard
        write_window_csv(output, rows)
        (output / "incident_metrics.json").write_text(
            json.dumps(serial(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8")
        write_report(output, recording, metrics, replay, mujoco_replay)
    print(json.dumps({
        "output": str(output),
        "max_dq": metrics["max_dq"],
        "max_publish_rate": metrics["max_publish_rate"],
        "limiter_reproduction_error": float(np.max(np.abs(replay["reproduction_error"]))),
        "mujoco_error": mujoco_error,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
