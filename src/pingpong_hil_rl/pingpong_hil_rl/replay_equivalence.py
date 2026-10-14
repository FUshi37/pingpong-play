from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace
from typing import Iterable

import mujoco as mj
import numpy as np

from .hil_env import (
    HilSafetyConfig,
    build_simcode4_hil_config,
    load_simcode4_env,
    make_hil_env_class,
)
from .ros_plant import JointSample, RIGHT_ARM_JOINTS


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = WORKSPACE / "src/pingpong_hil_rl/pingpong_hil_rl/outputs"
DEFAULT_RL_DIR = WORKSPACE / "simcode4/pingpong_moz1-main/pingpong_controller/tools/rl_sim"
DEFAULT_XML = DEFAULT_RL_DIR / "moz1_pd.xml"
PRESETS = ("legacy", "simcode2", "simcode3", "simcode4")


def _columns(prefix: str) -> list[str]:
    return [f"{prefix}/{name}" for name in RIGHT_ARM_JOINTS]


def _read_capture(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise RuntimeError(f"capture contains fewer than two samples: {path}")

    def matrix(prefix: str) -> np.ndarray:
        keys = _columns(prefix)
        return np.asarray([[float(row[key]) for key in keys] for row in rows], dtype=np.float64)

    data = {
        "timestamp": np.asarray([float(row["timestamp"]) for row in rows]),
        "cmd_q": matrix("cmd_q"),
        "actual_q": matrix("actual_q"),
        "actual_dq": matrix("actual_qvel"),
    }
    cart_keys = [f"cart_actual_pos/{axis}" for axis in "xyz"]
    quat_keys = [f"cart_actual_quat/{axis}" for axis in "xyzw"]
    if all(key in rows[0] for key in cart_keys):
        data["cart_actual_pos"] = np.asarray(
            [[float(row[key]) for key in cart_keys] for row in rows], dtype=np.float64
        )
    if all(key in rows[0] for key in quat_keys):
        data["cart_actual_quat"] = np.asarray(
            [[float(row[key]) for key in quat_keys] for row in rows], dtype=np.float64
        )
    return data


class _RecordedPlant:
    """Minimal plant adapter used only to initialize the production HIL env."""

    def __init__(self, sample: JointSample) -> None:
        self.sample = sample
        self.cfg = SimpleNamespace(control_hz=200.0, state_timeout_s=0.05)

    def wait_for_state(self, timeout_s: float | None = None) -> JointSample:
        return self.sample

    def move_to_pose(self, *args, **kwargs) -> JointSample:
        return self.sample


def _sample(index: int, stamp_s: float, q: np.ndarray, dq: np.ndarray) -> JointSample:
    return JointSample(
        sequence=index + 1,
        source_stamp_ns=int(round(stamp_s * 1e9)),
        receive_stamp_ns=time.monotonic_ns(),
        q=np.asarray(q, dtype=np.float64).copy(),
        dq=np.asarray(dq, dtype=np.float64).copy(),
        velocity_source="recorded_joint_state",
    )


def _best_lag(cmd: np.ndarray, actual: np.ndarray, max_lag: int = 40) -> tuple[int, float]:
    best = (0, float("inf"))
    for lag in range(min(max_lag, len(cmd) - 1, len(actual) - 1) + 1):
        pred = cmd[: len(cmd) - lag] if lag else cmd
        obs = actual[lag:] if lag else actual
        error = float(np.sqrt(np.mean((obs - pred) ** 2)))
        if error < best[1]:
            best = (lag, error)
    return best


def _rigid_alignment_error(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    mask = np.all(np.isfinite(source), axis=1) & np.all(np.isfinite(target), axis=1)
    source, target = source[mask], target[mask]
    if len(source) < 3:
        return float("nan"), float("nan")
    raw = float(np.sqrt(np.mean(np.sum((source - target) ** 2, axis=1))))
    s0, t0 = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - s0).T @ (target - t0))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    aligned = (source - s0) @ rotation.T + t0
    fitted = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return raw, fitted


def _hand_eye_error(
    model_pos: np.ndarray,
    model_mat: np.ndarray,
    report_pos: np.ndarray,
    report_quat: np.ndarray,
) -> dict[str, float]:
    """Fit T_report=B@T_model@C, matching the replay node's TCP calibration."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    mask = (
        np.all(np.isfinite(model_pos), axis=1)
        & np.all(np.isfinite(model_mat.reshape(len(model_mat), -1)), axis=1)
        & np.all(np.isfinite(report_pos), axis=1)
        & np.all(np.isfinite(report_quat), axis=1)
    )
    indices = np.flatnonzero(mask)
    if len(indices) < 30:
        return {"fit_position_rmse_m": float("nan"), "fit_position_max_m": float("nan")}
    # Fit on an evenly spaced subset, evaluate on every sample.  Keeping the
    # optimizer bounded in size makes four-trace analysis deterministic.
    fit_indices = indices[np.linspace(0, len(indices) - 1, min(160, len(indices)), dtype=int)]

    model_T, report_T = [], []
    for i in indices:
        a, t = np.eye(4), np.eye(4)
        a[:3, :3], a[:3, 3] = model_mat[i], model_pos[i]
        t[:3, :3], t[:3, 3] = Rotation.from_quat(report_quat[i]).as_matrix(), report_pos[i]
        model_T.append(a)
        report_T.append(t)
    lookup = {int(index): offset for offset, index in enumerate(indices)}

    def transform(vector: np.ndarray) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
        result[:3, 3] = vector[:3]
        return result

    def residual(parameters: np.ndarray) -> np.ndarray:
        b, c = transform(parameters[:6]), transform(parameters[6:])
        values = []
        for index in fit_indices:
            offset = lookup[int(index)]
            error = b @ model_T[offset] @ c @ np.linalg.inv(report_T[offset])
            values.extend(error[:3, 3])
            values.extend(Rotation.from_matrix(error[:3, :3]).as_rotvec())
        return np.asarray(values)

    solution = least_squares(residual, np.zeros(12), method="lm", max_nfev=5000)
    b, c = transform(solution.x[:6]), transform(solution.x[6:])
    errors = []
    for a, t in zip(model_T, report_T):
        errors.append(float(np.linalg.norm((b @ a @ c)[:3, 3] - t[:3, 3])))
    errors = np.asarray(errors)
    return {
        "fit_position_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "fit_position_max_m": float(np.max(errors)),
        "fit_samples": int(len(fit_indices)),
        "evaluation_samples": int(len(errors)),
        "base_transform": b.tolist(),
        "tool_transform": c.tolist(),
    }


def _plot_three_way_joint_trajectory(
    preset: str,
    times: np.ndarray,
    command_q: np.ndarray,
    physical_q: np.ndarray,
    mujoco_q: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot command, physical feedback and HIL/MuJoCo state on one figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    command_deg = np.rad2deg(command_q)
    physical_deg = np.rad2deg(physical_q)
    mujoco_deg = np.rad2deg(mujoco_q)
    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    axes = axes.reshape(-1)
    colors = {"command": "#3b3b3b", "physical": "#1677c8", "mujoco": "#ed7d31"}
    for joint in range(7):
        axis = axes[joint]
        axis.plot(
            times,
            command_deg[:, joint],
            color=colors["command"],
            linewidth=1.15,
            alpha=0.85,
            label="Command sent",
        )
        axis.plot(
            times,
            physical_deg[:, joint],
            color=colors["physical"],
            linewidth=2.2,
            alpha=0.80,
            label="Physical simulator feedback",
        )
        # Draw the HIL state last, with sparse markers. It is expected to be
        # exactly coincident with physical_q, so a plain solid line would hide
        # one of the two series and look as if only two curves were plotted.
        axis.plot(
            times,
            mujoco_deg[:, joint],
            color=colors["mujoco"],
            linewidth=1.0,
            linestyle="--",
            marker=".",
            markersize=2.2,
            markevery=25,
            alpha=0.95,
            label="Physical feedback through HIL/MuJoCo",
        )
        axis.set_title(f"RightArm-{joint}")
        axis.set_ylabel("Joint angle (deg)")
        axis.grid(True, alpha=0.25)

    error_axis = axes[7]
    command_error = np.max(np.abs(command_deg - physical_deg), axis=1)
    injection_error = np.max(np.abs(physical_deg - mujoco_deg), axis=1)
    error_axis.plot(
        times,
        command_error,
        color="#c62828",
        linewidth=1.3,
        label="max |command - physical|",
    )
    error_axis.plot(
        times,
        injection_error,
        color="#6a1b9a",
        linewidth=1.2,
        linestyle="--",
        label="max |physical - HIL/MuJoCo|",
    )
    error_axis.set_title("Instantaneous maximum error across 7 joints")
    error_axis.set_ylabel("Absolute error (deg)")
    error_axis.set_yscale("symlog", linthresh=1e-10)
    error_axis.grid(True, which="both", alpha=0.25)
    error_axis.legend(loc="upper right", fontsize=8)
    for axis in axes[-2:]:
        axis.set_xlabel("Replay time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(
        f"{preset}: command vs physical feedback vs current HIL/MuJoCo replay",
        fontsize=15,
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{preset}_three_way_joint_trajectory.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def mirror_capture(preset: str, capture_csv: Path, output_dir: Path, rl_dir: Path, xml: Path) -> dict:
    captured = _read_capture(capture_csv)
    times = captured["timestamp"] - captured["timestamp"][0]
    q, dq = captured["actual_q"], captured["actual_dq"]
    first = _sample(0, times[0], q[0], dq[0])

    module = load_simcode4_env(rl_dir)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cfg.dr_randomize_ball = False
    cfg.dr_randomize_contact = False
    cfg.ball_init_vxy_max = 0.0
    cfg.ball_init_vz = 0.0
    cfg.ball_init_vz_jitter = 0.0
    env_class = make_hil_env_class(module)
    env = env_class(
        xml_path=str(xml),
        plant=_RecordedPlant(first),
        cfg=cfg,
        hil_cfg=HilSafetyConfig(),
    )
    env.reset(seed=0)

    ball_radius = float(env.model.geom_size[env.ball_geom_id, 0])
    initial_gap = float(
        np.dot(env._ball_pos() - env._racket_pos(), env._racket_xmat()[:, 2])
        - ball_radius
    )
    rows: list[dict[str, float | int]] = []
    mj_q_all, mj_dq_all, racket_all, racket_mat_all, ball_all, contacts = [], [], [], [], [], []
    contact_intervals = 0

    for index in range(len(times)):
        current = _sample(index, times[index], q[index], dq[index])
        if index == 0:
            env._inject_prescribed_robot(current.q, current.dq)
            mj.mj_forward(env.model, env.data)
        else:
            before = _sample(index - 1, times[index - 1], q[index - 1], dq[index - 1])
            dt = float(times[index] - times[index - 1])
            if not (0.0001 <= dt <= 0.1):
                dt = 0.005
            env.advance_measured_interval(before, current, dt)

        mj_q = np.asarray(env.data.qpos[env.arm_qadr], dtype=np.float64).copy()
        mj_dq = np.asarray(env.data.qvel[env.arm_vadr], dtype=np.float64).copy()
        racket = env._racket_pos().astype(np.float64)
        racket_mat = env._racket_xmat().astype(np.float64)
        ball = env._ball_pos().astype(np.float64)
        contact = int(env._ball_contact_flags()[0])
        contact_intervals += contact
        mj_q_all.append(mj_q)
        mj_dq_all.append(mj_dq)
        racket_all.append(racket)
        racket_mat_all.append(racket_mat)
        ball_all.append(ball)
        contacts.append(contact)
        row: dict[str, float | int] = {
            "step": index,
            "time_sec": float(times[index]),
            "racket_x": float(racket[0]), "racket_y": float(racket[1]), "racket_z": float(racket[2]),
            "ball_x": float(ball[0]), "ball_y": float(ball[1]), "ball_z": float(ball[2]),
            "ball_racket_contact": contact,
        }
        for joint, src_q, src_dq, dst_q, dst_dq in zip(RIGHT_ARM_JOINTS, q[index], dq[index], mj_q, mj_dq):
            row[f"source_q/{joint}"] = float(src_q)
            row[f"mujoco_q/{joint}"] = float(dst_q)
            row[f"source_dq/{joint}"] = float(src_dq)
            row[f"mujoco_dq/{joint}"] = float(dst_dq)
            row[f"q_error/{joint}"] = float(dst_q - src_q)
            row[f"dq_error/{joint}"] = float(dst_dq - src_dq)
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    mirror_csv = output_dir / f"{preset}_mujoco_measured_replay.csv"
    with mirror_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mj_q_all = np.asarray(mj_q_all)
    mj_dq_all = np.asarray(mj_dq_all)
    racket_all = np.asarray(racket_all)
    racket_mat_all = np.asarray(racket_mat_all)
    ball_all = np.asarray(ball_all)
    contacts = np.asarray(contacts, dtype=np.int32)
    q_error, dq_error = mj_q_all - q, mj_dq_all - dq
    three_way_plot = _plot_three_way_joint_trajectory(
        preset,
        times,
        captured["cmd_q"],
        q,
        mj_q_all,
        output_dir,
    )
    no_lag_rmse = float(np.sqrt(np.mean((captured["actual_q"] - captured["cmd_q"]) ** 2)))
    joint_lags = [_best_lag(captured["cmd_q"][:, j], q[:, j]) for j in range(7)]
    global_lag = _best_lag(captured["cmd_q"], q)
    dt = np.diff(times)
    summary = {
        "preset": preset,
        "capture_csv": str(capture_csv),
        "mirror_csv": str(mirror_csv),
        "three_way_joint_trajectory_plot": str(three_way_plot),
        "n_samples": len(times),
        "duration_s": float(times[-1]),
        "recorder_cadence": {
            "mean_dt_ms": float(np.mean(dt) * 1e3),
            "std_dt_ms": float(np.std(dt) * 1e3),
            "min_dt_ms": float(np.min(dt) * 1e3),
            "max_dt_ms": float(np.max(dt) * 1e3),
        },
        "physical_tracking": {
            "no_lag_rmse_deg": float(np.rad2deg(no_lag_rmse)),
            "best_global_lag_samples": int(global_lag[0]),
            "best_global_lag_ms": float(global_lag[0] * 5.0),
            "best_lag_rmse_deg": float(np.rad2deg(global_lag[1])),
            "per_joint_best_lag_ms": [float(item[0] * 5.0) for item in joint_lags],
            "per_joint_best_lag_rmse_deg": [float(np.rad2deg(item[1])) for item in joint_lags],
        },
        "state_replacement": {
            "q_rmse_rad": float(np.sqrt(np.mean(q_error ** 2))),
            "q_max_abs_rad": float(np.max(np.abs(q_error))),
            "dq_rmse_rad_s": float(np.sqrt(np.mean(dq_error ** 2))),
            "dq_max_abs_rad_s": float(np.max(np.abs(dq_error))),
            "finite": bool(np.all(np.isfinite(mj_q_all)) and np.all(np.isfinite(mj_dq_all))),
        },
        "ball_contact_replay": {
            "initial_surface_gap_mm": initial_gap * 1e3,
            "contact_sample_count": int(contact_intervals),
            "contact_event_count": int(np.sum((contacts[1:] > 0) & (contacts[:-1] == 0))),
            "initial_ball_z_m": float(ball_all[0, 2]),
            "maximum_ball_z_m": float(np.max(ball_all[:, 2])),
            "maximum_rise_from_initial_m": float(np.max(ball_all[:, 2]) - ball_all[0, 2]),
            "maximum_upward_vz_m_s": float(
                np.max(np.diff(ball_all[:, 2]) / np.maximum(np.diff(times), 1e-6))
            ),
            "final_ball_z_m": float(ball_all[-1, 2]),
        },
    }
    if "cart_actual_pos" in captured:
        raw, aligned = _rigid_alignment_error(captured["cart_actual_pos"], racket_all)
        summary["cartesian_consistency"] = {
            "raw_position_rmse_m": raw,
            "rigid_aligned_shape_rmse_m": aligned,
            "note": "raw frames/TCP may differ; aligned error tests trajectory shape",
        }
        if "cart_actual_quat" in captured:
            summary["cartesian_consistency"]["base_tool_calibrated"] = _hand_eye_error(
                racket_all,
                racket_mat_all,
                captured["cart_actual_pos"],
                captured["cart_actual_quat"],
            )
    summary_path = output_dir / f"{preset}_equivalence_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()
    return summary


def capture_preset(preset: str, output_dir: Path, timeout_s: float = 45.0) -> Path:
    capture_dir = output_dir / preset / "physical_capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    before = set(capture_dir.glob("*_replay_analysis.json"))
    command = [
        "ros2", "run", "pingpong_replay", "replay_node", "--ros-args",
        "-p", f"trajectory_preset:={preset}",
        "-p", "input_mode:=sim_actual",
        "-p", "compensation_mode:=none",
        "-p", "right_arm_mu_idx:=3",
        "-p", "require_enter_confirm:=false",
        "-p", "record_data:=true",
        "-p", "auto_plot_analysis:=true",
        "-p", f"output_dir:={capture_dir}",
    ]
    log_path = capture_dir / "ros2_replay.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        deadline = time.monotonic() + timeout_s
        result: Path | None = None
        try:
            while time.monotonic() < deadline:
                fresh = set(capture_dir.glob("*_replay_analysis.json")) - before
                if fresh:
                    result = max(fresh, key=lambda path: path.stat().st_mtime)
                    break
                code = process.poll()
                if code is not None:
                    raise RuntimeError(f"{preset} replay exited with code {code}; see {log_path}")
                time.sleep(0.1)
            if result is None:
                raise TimeoutError(f"{preset} replay did not save within {timeout_s}s; see {log_path}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=3.0)
    metadata = json.loads(result.read_text(encoding="utf-8"))
    csv_path = Path(metadata["output_csv"])
    if not csv_path.is_file():
        raise FileNotFoundError(f"capture summary points to missing CSV: {csv_path}")
    return csv_path


def _write_overall(summaries: Iterable[dict], output_dir: Path) -> dict:
    summaries = list(summaries)
    replacement_pass = all(
        item["state_replacement"]["finite"]
        and item["state_replacement"]["q_max_abs_rad"] <= 1e-9
        and item["state_replacement"]["dq_max_abs_rad_s"] <= 1e-9
        for item in summaries
    )
    cart_values = [
        item.get("cartesian_consistency", {}).get("base_tool_calibrated", {}).get(
            "fit_position_rmse_m", float("nan")
        )
        for item in summaries
    ]
    overall = {
        "created_unix_s": time.time(),
        "presets": summaries,
        "verdict": {
            "joint_state_replacement_exact": replacement_pass,
            "max_q_injection_error_rad": max(item["state_replacement"]["q_max_abs_rad"] for item in summaries),
            "max_dq_injection_error_rad_s": max(item["state_replacement"]["dq_max_abs_rad_s"] for item in summaries),
            "max_base_tool_calibrated_cartesian_rmse_m": float(np.nanmax(cart_values)),
            "perfect_full_system_replacement_proven": False,
            "reason": (
                "This experiment can validate the ROS feedback-to-MuJoCo robot-state/FK path. "
                "The physical simulator does not supply ball/contact state, so MuJoCo still owns "
                "those dynamics and full coupled-system identity cannot be proven by these traces."
            ),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall_equivalence_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    with (output_dir / "overall_equivalence_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "preset", "samples", "mean_dt_ms", "std_dt_ms", "tracking_rmse_deg",
            "best_lag_ms", "best_lag_rmse_deg", "q_injection_max_rad",
            "dq_injection_max_rad_s", "initial_surface_gap_mm", "contact_samples",
                "cart_rigid_shape_rmse_m", "cart_base_tool_fit_rmse_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            writer.writerow({
                "preset": item["preset"],
                "samples": item["n_samples"],
                "mean_dt_ms": item["recorder_cadence"]["mean_dt_ms"],
                "std_dt_ms": item["recorder_cadence"]["std_dt_ms"],
                "tracking_rmse_deg": item["physical_tracking"]["no_lag_rmse_deg"],
                "best_lag_ms": item["physical_tracking"]["best_global_lag_ms"],
                "best_lag_rmse_deg": item["physical_tracking"]["best_lag_rmse_deg"],
                "q_injection_max_rad": item["state_replacement"]["q_max_abs_rad"],
                "dq_injection_max_rad_s": item["state_replacement"]["dq_max_abs_rad_s"],
                "initial_surface_gap_mm": item["ball_contact_replay"]["initial_surface_gap_mm"],
                "contact_samples": item["ball_contact_replay"]["contact_sample_count"],
                "cart_rigid_shape_rmse_m": item.get("cartesian_consistency", {}).get("rigid_aligned_shape_rmse_m"),
                "cart_base_tool_fit_rmse_m": item.get("cartesian_consistency", {}).get(
                    "base_tool_calibrated", {}
                ).get("fit_position_rmse_m"),
            })
    return overall


def run(args: argparse.Namespace) -> dict:
    root = Path(args.output_dir).expanduser().resolve()
    run_dir = root / (args.run_name or time.strftime("replay_equivalence_%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    captures: dict[str, Path] = {}
    if args.stage in ("all", "capture"):
        for preset in PRESETS:
            print(f"[{preset}] capturing physical simulator feedback", flush=True)
            captures[preset] = capture_preset(preset, run_dir, args.capture_timeout)
            print(f"[{preset}] capture saved: {captures[preset]}", flush=True)
    if args.stage == "capture":
        manifest = {name: str(path) for name, path in captures.items()}
        (run_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"run_dir": str(run_dir), "captures": manifest}

    if args.stage == "mirror":
        source = Path(args.capture_run).expanduser().resolve()
        for preset in PRESETS:
            matches = sorted((source / preset / "physical_capture").glob("*_replay_analysis.csv"))
            if not matches:
                raise FileNotFoundError(f"no capture CSV for {preset} below {source}")
            captures[preset] = matches[-1]

    summaries = []
    for preset in PRESETS:
        print(f"[{preset}] replaying measured states through MuJoCo HIL path", flush=True)
        summaries.append(
            mirror_capture(preset, captures[preset], run_dir / preset / "mujoco_replay", Path(args.rl_dir), Path(args.xml))
        )
    overall = _write_overall(summaries, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "verdict": overall["verdict"]}, indent=2), flush=True)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-trajectory physical/MuJoCo HIL equivalence experiment")
    parser.add_argument("--stage", choices=("all", "capture", "mirror"), default="all")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--capture-run", default="", help="existing run directory for --stage mirror")
    parser.add_argument("--capture-timeout", type=float, default=45.0)
    parser.add_argument("--rl-dir", default=str(DEFAULT_RL_DIR))
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    args = parser.parse_args()
    if args.stage == "mirror" and not args.capture_run:
        parser.error("--capture-run is required for --stage mirror")
    run(args)


if __name__ == "__main__":
    main()
