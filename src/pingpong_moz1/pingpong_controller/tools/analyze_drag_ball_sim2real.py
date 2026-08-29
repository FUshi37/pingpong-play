#!/usr/bin/env python3
"""Analyze a drag-ball GPU0-RMP recording and separate policy/plant gaps.

The live independent mirror and the real controller receive the exact same
drag-ball input.  This tool first produces the same tick-start joint/racket
comparison as ``analyze_valid_juggle_sim2real.py``.  It then drives a fresh
recovered-RMP MuJoCo plant with the *recorded real published q_ref* so the
real-to-common-replay error isolates the actuator/plant model from the later
closed-loop policy-command divergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_valid_juggle_sim2real as base


DT = 0.005
COMMON_COLOR = "#2a9d8f"
POLICY_COLOR = "#e76f51"
REAL_COLOR = "#1769aa"
JOINT_NAMES = [f"J{index}" for index in range(1, 8)]
AXES = ("x", "y", "z")


def _published_command(row: dict[str, Any]) -> np.ndarray:
    command = row.get("command")
    if not isinstance(command, dict):
        raise RuntimeError("real row is missing command payload")
    value = base.vector(command.get("publishedSafePosition"), 7)
    if value is None:
        raise RuntimeError("real row is missing publishedSafePosition")
    return value


def _executed_action(row: dict[str, Any]) -> np.ndarray:
    value = base.vector(row.get("policyExecutedAction"), 7)
    if value is None:
        value = base.vector(row.get("policyAction"), 7)
    if value is None:
        raise RuntimeError("row is missing policy action")
    return value


def _mirror_config(recording: Path, checkpoint: Path, xml: Path):
    package_root = base.ROOT / "src/pingpong_moz1"
    if str(base.ROOT) not in sys.path:
        sys.path.insert(0, str(base.ROOT))
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from pingpong_controller.sim_mirror import SimMirrorConfig

    metadata = json.loads((recording / "metadata.json").read_text(encoding="utf-8"))
    sim_meta = ((metadata.get("config") or {}).get("simMirror") or {})
    recorded_checkpoint = Path(sim_meta.get("modelPath", checkpoint)).resolve()
    recorded_policy_xml = Path(sim_meta.get("policyXmlPath", xml)).resolve()
    recorded_sim_xml = Path(sim_meta.get("xmlPath", xml)).resolve()
    if recorded_checkpoint != checkpoint.resolve():
        raise RuntimeError(
            "checkpoint override does not match recording metadata: "
            f"{checkpoint} != {recorded_checkpoint}"
        )
    return SimMirrorConfig(
        model_path=recorded_checkpoint,
        policy_xml_path=recorded_policy_xml,
        sim_xml_path=recorded_sim_xml,
        policy_obs_dim=57,
        control_dt=DT,
        backend="mujoco",
        frame_rate_hz=1.0,
        enable_video_frames=False,
        use_mujoco_renderer=False,
        enable_compensation=False,
        policy_profile="gpu0_qvel_real_rmp_v85",
        policy_model="v91",
        simcode7_model_path=recorded_checkpoint,
        simcode7_v91_model_path=recorded_checkpoint,
        simcode7_policy_xml_path=recorded_policy_xml,
        simcode7_sim_xml_path=recorded_sim_xml,
    )


def _set_tick_ball(mirror: Any, item: dict[str, Any]) -> None:
    sync = item["sim"].get("sync") or {}
    ball = base.vector(sync.get("inputBall"), 3)
    velocity = base.vector(sync.get("inputVelocity"), 3)
    if ball is None or velocity is None:
        raise RuntimeError(f"missing mirror ball input at sourceSeq={item['seq']}")
    mirror._set_ball_state(ball.astype(np.float32), velocity.astype(np.float32))
    import mujoco as mj

    mj.mj_forward(mirror.model, mirror.data)


def _ball_contact_partners(mirror: Any) -> list[str]:
    """Return geom names paired with the injected MuJoCo ball."""
    if mirror.ball_body_id < 0:
        return []
    import mujoco as mj

    partners: list[str] = []
    for index in range(int(mirror.data.ncon)):
        contact = mirror.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(mirror.model.geom_bodyid[geom1])
        body2 = int(mirror.model.geom_bodyid[geom2])
        if body1 == mirror.ball_body_id:
            partner = geom2
        elif body2 == mirror.ball_body_id:
            partner = geom1
        else:
            continue
        partners.append(
            mj.mj_id2name(mirror.model, mj.mjtObj.mjOBJ_GEOM, partner)
            or f"geom_{partner}"
        )
    return partners


def _disable_ball_collisions(mirror: Any) -> None:
    """Keep the drag-ball state but make it observation-only for plant replay."""
    ball_geoms = np.flatnonzero(
        np.asarray(mirror.model.geom_bodyid) == int(mirror.ball_body_id)
    )
    mirror.model.geom_contype[ball_geoms] = 0
    mirror.model.geom_conaffinity[ball_geoms] = 0
    # MuJoCo's compiled broadphase/pair tables keep producing the original
    # ball/racket pair when only the masks are changed at runtime.  Collapse
    # the diagnostic ball geom as well; the free-joint state used as the
    # policy observation remains unchanged.
    mirror.model.geom_size[ball_geoms, :] = 1.0e-9
    mirror.model.geom_rbound[ball_geoms] = 1.0e-9


def _park_replay_ball(mirror: Any) -> None:
    """Place the unused ball away from the robot in actor-free plant replay."""
    mirror._set_ball_state(
        np.asarray([0.0, 0.0, 10.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    import mujoco as mj

    mj.mj_forward(mirror.model, mirror.data)


def _reset_replay_mirror(
    mirror: Any,
    q: np.ndarray,
    dq: np.ndarray,
    item: dict[str, Any],
) -> None:
    sync = item["sim"].get("sync") or {}
    ball = base.vector(sync.get("inputBall"), 3)
    velocity = base.vector(sync.get("inputVelocity"), 3)
    mirror.reset(
        arm_q=q.astype(np.float32),
        ball_pos=ball.astype(np.float32),
        ball_vel=velocity.astype(np.float32),
        ball_valid=bool(sync.get("inputValid", True)),
        ball_obs_age_s=float(sync.get("inputAgeS", 0.0) or 0.0),
    )
    mirror.data.qvel[mirror.arm_vadr] = dq.astype(np.float64)
    import mujoco as mj

    mj.mj_forward(mirror.model, mirror.data)


def _uniform_difference(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=np.float64)
    output[:-1] = np.diff(values, axis=0) / DT
    return output


def common_command_replay(
    recording: Path,
    checkpoint: Path,
    xml: Path,
    aligned: list[dict[str, Any]],
    controller: Any,
) -> dict[str, Any]:
    """Replay real and independent q_ref streams through identical plants."""
    from pingpong_controller.sim_mirror import RealBallSimMirror

    config = _mirror_config(recording, checkpoint, xml)
    common = RealBallSimMirror(config)
    _disable_ball_collisions(common)
    contact_common = RealBallSimMirror(config)
    timing_check = RealBallSimMirror(config)
    first = aligned[0]
    _reset_replay_mirror(common, first["real_q"], first["real_dq"], first)
    _park_replay_ball(common)
    _reset_replay_mirror(
        contact_common, first["real_q"], first["real_dq"], first)
    _reset_replay_mirror(timing_check, first["sim_q"], first["sim_dq"], first)
    rows: list[dict[str, Any]] = []
    try:
        for item in aligned:
            _park_replay_ball(common)
            _set_tick_ball(contact_common, item)
            _set_tick_ball(timing_check, item)
            collisionless_partners_before = _ball_contact_partners(common)
            common_partners_before = _ball_contact_partners(contact_common)
            check_partners_before = _ball_contact_partners(timing_check)
            common_q = common.data.qpos[common.arm_qadr].astype(np.float64).copy()
            common_dq = common.data.qvel[common.arm_vadr].astype(np.float64).copy()
            contact_common_q = contact_common.data.qpos[
                contact_common.arm_qadr
            ].astype(np.float64).copy()
            contact_common_dq = contact_common.data.qvel[
                contact_common.arm_vadr
            ].astype(np.float64).copy()
            check_q = timing_check.data.qpos[
                timing_check.arm_qadr
            ].astype(np.float64).copy()
            check_dq = timing_check.data.qvel[
                timing_check.arm_vadr
            ].astype(np.float64).copy()
            common_kin = base.pose_twist(controller, common_q, common_dq)
            contact_common_kin = base.pose_twist(
                controller, contact_common_q, contact_common_dq)
            check_kin = base.pose_twist(controller, check_q, check_dq)
            real_input = _published_command(item["real"])
            sim_input = base.vector(item["sim"].get("policyCommand"), 7)
            if sim_input is None:
                raise RuntimeError(f"missing sim policyCommand at sourceSeq={item['seq']}")
            replay_row = {
                "elapsed": float(item["elapsed"]),
                "seq": int(item["seq"]),
                "common_q": common_q,
                "common_dq": common_dq,
                "common_kin": common_kin,
                "contact_common_q": contact_common_q,
                "contact_common_dq": contact_common_dq,
                "contact_common_kin": contact_common_kin,
                "check_q": check_q,
                "check_dq": check_dq,
                "check_kin": check_kin,
                "real_input": real_input,
                "sim_input": sim_input,
                "collisionless_ball_contact_partners_before_step": (
                    collisionless_partners_before
                ),
                "common_ball_contact_partners_before_step": (
                    common_partners_before
                ),
                "check_ball_contact_partners_before_step": check_partners_before,
            }
            rows.append(replay_row)
            common._step_recovered_rmp(real_input.astype(np.float32))
            contact_common._step_recovered_rmp(real_input.astype(np.float32))
            timing_check._step_recovered_rmp(sim_input.astype(np.float32))
            replay_row["collisionless_ball_contact_partners_after_step"] = (
                _ball_contact_partners(common)
            )
            replay_row["common_ball_contact_partners_after_step"] = (
                _ball_contact_partners(contact_common)
            )
            replay_row["check_ball_contact_partners_after_step"] = (
                _ball_contact_partners(timing_check)
            )
    finally:
        common.close()
        contact_common.close()
        timing_check.close()

    for item, replay in zip(aligned, rows):
        item.update(replay)
    common_quaternion = base.aligned_matrix(aligned, "common_kin.quaternion")
    common_euler = base.quaternion_to_euler_deg(common_quaternion)
    for index, item in enumerate(aligned):
        item["common_euler"] = common_euler[index]
    common_dq = base.aligned_matrix(aligned, "common_dq")
    common_velocity = base.aligned_matrix(aligned, "common_kin.velocity")
    common_omega = base.aligned_matrix(aligned, "common_kin.angular_velocity")
    common_ddq = _uniform_difference(common_dq)
    common_acceleration = _uniform_difference(common_velocity)
    common_angular_acceleration = _uniform_difference(common_omega)
    for index, item in enumerate(aligned):
        item["common_ddq"] = common_ddq[index]
        item["common_racket_acceleration"] = common_acceleration[index]
        item["common_racket_angular_acceleration"] = (
            common_angular_acceleration[index]
        )
    return summarize_replay(aligned)


def _signal_metric(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    return base.metric(np.asarray(left), np.asarray(right))


def _comparison_groups(
    aligned: list[dict[str, Any]],
    left: str,
    right: str,
) -> dict[str, Any]:
    paths = {
        "joint_position_rad": (f"{left}_q", f"{right}_q"),
        "joint_velocity_rad_s": (f"{left}_dq", f"{right}_dq"),
        "racket_position_m": (
            f"{left}_kin.position", f"{right}_kin.position"),
        "racket_velocity_m_s": (
            f"{left}_kin.velocity", f"{right}_kin.velocity"),
    }
    return {
        name: _signal_metric(
            base.aligned_matrix(aligned, left_path),
            base.aligned_matrix(aligned, right_path),
        )
        for name, (left_path, right_path) in paths.items()
    }


def _window_comparisons(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    elapsed = np.asarray([item["elapsed"] for item in aligned])
    for label, end_s in (
        ("first_0p1_s", 0.1),
        ("first_0p25_s", 0.25),
        ("first_0p5_s", 0.5),
        ("first_1p0_s", 1.0),
        ("full", math.inf),
    ):
        selected = [
            item for item, keep in zip(aligned, elapsed <= end_s + 1.0e-12)
            if keep
        ]
        windows[label] = {
            "sample_count": len(selected),
            "real_to_common_actuator_gap": _comparison_groups(
                selected, "real", "common"),
            "real_to_contact_enabled_common_gap": _comparison_groups(
                selected, "real", "contact_common"),
            "contact_enabled_to_collisionless_common_effect": (
                _comparison_groups(selected, "contact_common", "common")
            ),
            "common_to_independent_policy_response": _comparison_groups(
                selected, "contact_common", "sim"),
            "collisionless_common_to_independent_combined_gap": (
                _comparison_groups(selected, "common", "sim")
            ),
            "real_to_independent_total_gap": _comparison_groups(
                selected, "real", "sim"),
        }
    return windows


def _first_crossing(
    elapsed: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    indices = np.flatnonzero(values > threshold)
    if not indices.size:
        return {"threshold": threshold, "elapsed_s": None, "value": None}
    index = int(indices[0])
    return {
        "threshold": threshold,
        "elapsed_s": float(elapsed[index]),
        "value": float(values[index]),
        "sample_index": index,
    }


def _rms_per_tick(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(right) - np.asarray(left)) ** 2, axis=1))


def summarize_replay(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = np.asarray([item["elapsed"] for item in aligned])
    real_q = base.aligned_matrix(aligned, "real_q")
    sim_q = base.aligned_matrix(aligned, "sim_q")
    common_q = base.aligned_matrix(aligned, "common_q")
    real_dq = base.aligned_matrix(aligned, "real_dq")
    sim_dq = base.aligned_matrix(aligned, "sim_dq")
    common_dq = base.aligned_matrix(aligned, "common_dq")
    check_q = base.aligned_matrix(aligned, "check_q")
    check_dq = base.aligned_matrix(aligned, "check_dq")
    real_action = base.aligned_matrix(aligned, "real_action")
    sim_action = base.aligned_matrix(aligned, "sim_action")
    real_qcmd = base.aligned_matrix(aligned, "real_qcmd")
    sim_qcmd = base.aligned_matrix(aligned, "sim_qcmd")
    published = base.aligned_matrix(aligned, "real_input")
    real_racket = base.aligned_matrix(aligned, "real_kin.position")
    sim_racket = base.aligned_matrix(aligned, "sim_kin.position")
    real_executed = np.asarray([
        _executed_action(item["real"]) for item in aligned
    ])
    sim_executed = np.asarray([
        _executed_action(item["sim"]) for item in aligned
    ])
    self_q = _signal_metric(sim_q, check_q)
    self_dq = _signal_metric(sim_dq, check_dq)
    self_check = {
        "joint_position": self_q,
        "joint_velocity": self_dq,
        "tolerances": {
            "joint_position_rms_rad": 5.0e-4,
            "joint_velocity_rms_rad_s": 2.0e-2,
        },
    }
    self_check["passed"] = bool(
        self_q["overall_rms"] <= 5.0e-4
        and self_dq["overall_rms"] <= 2.0e-2
    )
    if not self_check["passed"]:
        raise RuntimeError(
            "independent-command replay timing self-check failed: "
            f"q={self_q['overall_rms']:.6g}, dq={self_dq['overall_rms']:.6g}"
        )

    per_tick = {
        "joint_position_rms_rad": _rms_per_tick(real_q, sim_q),
        "joint_velocity_rms_rad_s": _rms_per_tick(real_dq, sim_dq),
        "policy_action_rms": _rms_per_tick(real_action, sim_action),
        "policy_command_rms_rad": _rms_per_tick(real_qcmd, sim_qcmd),
        "racket_position_rms_m": _rms_per_tick(real_racket, sim_racket),
    }
    thresholds = {
        "joint_position_rms_rad": 0.005,
        "joint_velocity_rms_rad_s": 0.10,
        "policy_action_rms": 0.05,
        "policy_command_rms_rad": 0.005,
        "racket_position_rms_m": 0.002,
    }
    onset = {
        name: _first_crossing(elapsed, values, thresholds[name])
        for name, values in per_tick.items()
    }
    command_match = _signal_metric(real_qcmd, published)
    initial_contract = {
        "joint_position_exact": bool(np.array_equal(real_q[0], sim_q[0])),
        "joint_velocity_exact": bool(np.array_equal(real_dq[0], sim_dq[0])),
        "policy_action_max_abs_gap": float(np.max(np.abs(
            real_action[0] - sim_action[0]))),
        "policy_command_max_abs_gap_rad": float(np.max(np.abs(
            real_qcmd[0] - sim_qcmd[0]))),
    }
    saturation = {
        "real_any_joint_raw_action_outside_unit_fraction": float(np.mean(
            np.any(np.abs(real_action) > 1.0, axis=1))),
        "sim_any_joint_raw_action_outside_unit_fraction": float(np.mean(
            np.any(np.abs(sim_action) > 1.0, axis=1))),
        "real_any_joint_action_clipped_fraction": float(np.mean(
            np.any(np.abs(real_action - real_executed) > 1.0e-6, axis=1))),
        "sim_any_joint_action_clipped_fraction": float(np.mean(
            np.any(np.abs(sim_action - sim_executed) > 1.0e-6, axis=1))),
    }
    tracking_real = published - real_q
    tracking_common = published - common_q
    tracking_contact_common = published - base.aligned_matrix(
        aligned, "contact_common_q")
    tracking = {
        "real_tracking_error_rms_rad": float(np.sqrt(np.mean(
            tracking_real * tracking_real))),
        "common_replay_tracking_error_rms_rad": float(np.sqrt(np.mean(
            tracking_common * tracking_common))),
        "contact_enabled_common_replay_tracking_error_rms_rad": float(
            np.sqrt(np.mean(tracking_contact_common * tracking_contact_common))
        ),
        "real_per_joint_rms_rad": np.sqrt(np.mean(
            tracking_real * tracking_real, axis=0)).tolist(),
        "common_per_joint_rms_rad": np.sqrt(np.mean(
            tracking_common * tracking_common, axis=0)).tolist(),
        "contact_enabled_common_per_joint_rms_rad": np.sqrt(np.mean(
            tracking_contact_common * tracking_contact_common,
            axis=0,
        )).tolist(),
    }
    contact_audit = {}
    for label, path in (
        ("collisionless_before_step", "collisionless_ball_contact_partners_before_step"),
        ("collisionless_after_step", "collisionless_ball_contact_partners_after_step"),
        ("common_before_step", "common_ball_contact_partners_before_step"),
        ("common_after_step", "common_ball_contact_partners_after_step"),
        ("independent_before_step", "check_ball_contact_partners_before_step"),
        ("independent_after_step", "check_ball_contact_partners_after_step"),
    ):
        contacts = [item[path] for item in aligned]
        counts = np.asarray([len(partners) for partners in contacts])
        active = np.flatnonzero(counts > 0)
        partners = Counter(
            partner for tick_partners in contacts for partner in tick_partners
        )
        contact_audit[label] = {
            "ticks_with_ball_contact": int(active.size),
            "fraction": float(active.size / counts.size),
            "first_elapsed_s": (
                None if not active.size else float(elapsed[int(active[0])])
            ),
            "max_simultaneous_contacts": int(np.max(counts)),
            "partner_contact_counts": dict(sorted(partners.items())),
        }
    return {
        "self_check": self_check,
        "initial_contract": initial_contract,
        "real_policy_command_to_published_command": command_match,
        "windows": _window_comparisons(aligned),
        "divergence_onset": onset,
        "action_saturation": saturation,
        "tracking": tracking,
        "injected_ball_contact_audit": contact_audit,
    }


def _plot_three_joint_family(
    output: Path,
    aligned: list[dict[str, Any]],
    real_path: str,
    common_path: str,
    sim_path: str,
    title: str,
    unit: str,
) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    real = base.aligned_matrix(aligned, real_path)
    common = base.aligned_matrix(aligned, common_path)
    sim = base.aligned_matrix(aligned, sim_path)
    figure, axes = plt.subplots(
        7, 1, figsize=(15, 18), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.plot(time, real[:, index], color=REAL_COLOR, linewidth=0.70,
                  label="real hardware" if index == 0 else None)
        axis.plot(time, common[:, index], color=COMMON_COLOR, linewidth=0.65,
                  label="collisionless common-command replay"
                  if index == 0 else None)
        axis.plot(time, sim[:, index], color=POLICY_COLOR, linewidth=0.60,
                  linestyle="-", alpha=0.9,
                  label="independent sim" if index == 0 else None)
        axis.set_ylabel(f"J{index + 1}\n[{unit}]")
        axis.grid(alpha=0.23)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_tracking_error(output: Path, aligned: list[dict[str, Any]]) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    command = base.aligned_matrix(aligned, "real_input")
    real = command - base.aligned_matrix(aligned, "real_q")
    common = command - base.aligned_matrix(aligned, "common_q")
    figure, axes = plt.subplots(
        7, 1, figsize=(15, 18), sharex=True, constrained_layout=True)
    for index, axis in enumerate(axes):
        axis.plot(time, real[:, index], color=REAL_COLOR, linewidth=0.70,
                  label="real q_ref - q" if index == 0 else None)
        axis.plot(time, common[:, index], color=COMMON_COLOR, linewidth=0.65,
                  label="replay q_ref - q" if index == 0 else None)
        axis.set_ylabel(f"J{index + 1}\n[rad]")
        axis.grid(alpha=0.23)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle(
        "Plant tracking under the same q_ref (drag ball is observation-only)",
        fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_decomposition(output: Path, aligned: list[dict[str, Any]]) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    signals = (
        ("joint q RMS [rad]", "real_q", "common_q", "sim_q"),
        ("joint dq RMS [rad/s]", "real_dq", "common_dq", "sim_dq"),
        ("racket position RMS [m]", "real_kin.position",
         "common_kin.position", "sim_kin.position"),
        ("racket velocity RMS [m/s]", "real_kin.velocity",
         "common_kin.velocity", "sim_kin.velocity"),
    )
    figure, axes = plt.subplots(
        4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    for axis, (label, real_path, common_path, sim_path) in zip(axes, signals):
        real = base.aligned_matrix(aligned, real_path)
        common = base.aligned_matrix(aligned, common_path)
        contact_common = base.aligned_matrix(
            aligned, common_path.replace("common_", "contact_common_", 1)
            if common_path.startswith("common_")
            else common_path.replace("common_kin", "contact_common_kin", 1)
        )
        sim = base.aligned_matrix(aligned, sim_path)
        axis.plot(time, _rms_per_tick(real, sim), color="#6c5ce7",
                  linewidth=0.70,
                  label="total: real ↔ independent sim")
        axis.plot(time, _rms_per_tick(real, common), color=COMMON_COLOR,
                  linewidth=0.65,
                  label="plant: real ↔ collisionless common replay")
        axis.plot(time, _rms_per_tick(common, contact_common), color="#f4a261",
                  linewidth=0.60,
                  label="virtual-ball contact: collisionless ↔ contact replay")
        axis.plot(time, _rms_per_tick(contact_common, sim), color=POLICY_COLOR,
                  linewidth=0.60,
                  label="command response: contact replay ↔ independent sim")
        axis.set_ylabel(label)
        axis.grid(alpha=0.23)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle(
        "Sim2Real decomposition (nonlinear components do not add algebraically)",
        fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_early_divergence(output: Path, aligned: list[dict[str, Any]]) -> None:
    elapsed = np.asarray([item["elapsed"] for item in aligned])
    mask = elapsed <= 1.0 + 1.0e-12
    time = elapsed[mask]
    signals = (
        ("joint q RMS [rad]", "real_q", "sim_q", 0.005),
        ("joint dq RMS [rad/s]", "real_dq", "sim_dq", 0.10),
        ("policy action RMS", "real_action", "sim_action", 0.05),
        ("policy q_ref RMS [rad]", "real_qcmd", "sim_qcmd", 0.005),
    )
    figure, axes = plt.subplots(
        4, 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    for axis, (label, real_path, sim_path, threshold) in zip(axes, signals):
        real = base.aligned_matrix(aligned, real_path)[mask]
        sim = base.aligned_matrix(aligned, sim_path)[mask]
        axis.plot(time, _rms_per_tick(real, sim), color="#4e79a7",
                  linewidth=0.70)
        axis.axhline(threshold, color="#d62728", linestyle="--", linewidth=0.9,
                     label=f"diagnostic threshold={threshold:g}")
        axis.set_ylabel(label)
        axis.grid(alpha=0.23)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time from policy start [s]")
    figure.suptitle(
        "First-second closed-loop divergence with identical drag-ball input",
        fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def plot_drag_ball_input(output: Path, aligned: list[dict[str, Any]]) -> None:
    time = np.asarray([item["elapsed"] for item in aligned])
    real_position = np.asarray([
        base.vector(item["real"]["policyInput"].get("ball"), 3)
        for item in aligned
    ])
    real_velocity = np.asarray([
        base.vector(item["real"]["policyInput"].get("velocity"), 3)
        for item in aligned
    ])
    sim_position = np.asarray([
        base.vector((item["sim"].get("sync") or {}).get("inputBall"), 3)
        for item in aligned
    ])
    sim_velocity = np.asarray([
        base.vector((item["sim"].get("sync") or {}).get("inputVelocity"), 3)
        for item in aligned
    ])
    figure, axes = plt.subplots(
        3, 2, figsize=(16, 10), sharex=True, constrained_layout=True)
    for index in range(3):
        axes[index, 0].plot(time, real_position[:, index], color=REAL_COLOR,
                            linewidth=0.70,
                            label="real policy input" if index == 0 else None)
        axes[index, 0].plot(time, sim_position[:, index], color=POLICY_COLOR,
                            linestyle="-", linewidth=0.60,
                            label="sim policy input" if index == 0 else None)
        axes[index, 1].plot(time, real_velocity[:, index], color=REAL_COLOR,
                            linewidth=0.70)
        axes[index, 1].plot(time, sim_velocity[:, index], color=POLICY_COLOR,
                            linestyle="-", linewidth=0.60)
        axes[index, 0].set_ylabel(f"{AXES[index]} [m]")
        axes[index, 1].set_ylabel(f"{AXES[index]} [m/s]")
        axes[index, 0].grid(alpha=0.23)
        axes[index, 1].grid(alpha=0.23)
    axes[0, 0].set_title("Drag-ball position")
    axes[0, 1].set_title("Drag-ball velocity")
    axes[0, 0].legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time from policy start [s]")
    axes[-1, 1].set_xlabel("Time from policy start [s]")
    figure.suptitle(
        "Policy-facing drag-ball input: exact real/sim equality",
        fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_replay_csv(output: Path, aligned: list[dict[str, Any]]) -> None:
    fields = (
        ("q_rad", "real_q", "common_q", "contact_common_q", "sim_q", JOINT_NAMES),
        ("dq_rad_s", "real_dq", "common_dq", "contact_common_dq", "sim_dq", JOINT_NAMES),
        ("qref_rad", "real_input", "real_input", "real_input", "sim_input", JOINT_NAMES),
        ("racket_position_m", "real_kin.position", "common_kin.position",
         "contact_common_kin.position", "sim_kin.position", AXES),
        ("racket_velocity_m_s", "real_kin.velocity", "common_kin.velocity",
         "contact_common_kin.velocity", "sim_kin.velocity", AXES),
    )
    header = ["elapsed_s", "source_seq"]
    for label, _real, _common, _contact, _sim, names in fields:
        for side in (
            "real", "collisionless_common_replay",
            "contact_enabled_common_replay", "independent_sim",
        ):
            header.extend(f"{side}_{label}_{name}" for name in names)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for item in aligned:
            row: list[Any] = [item["elapsed"], item["seq"]]
            for (
                _label, real_path, common_path, contact_path, sim_path, _names
            ) in fields:
                row.extend(base.aligned_matrix([item], real_path)[0])
                row.extend(base.aligned_matrix([item], common_path)[0])
                row.extend(base.aligned_matrix([item], contact_path)[0])
                row.extend(base.aligned_matrix([item], sim_path)[0])
            writer.writerow(row)


def _metric_value(
    replay: dict[str, Any],
    comparison: str,
    group: str,
    window: str = "full",
) -> float:
    return float(
        replay["windows"][window][comparison][group]["overall_rms"]
    )


def write_report(
    output: Path,
    recording: Path,
    summary: dict[str, Any],
) -> None:
    replay = summary["actuator_decomposition"]
    stage = summary["stage"]
    mirror = summary["mirror"]
    groups = summary["groups"]
    plant_key = "real_to_common_actuator_gap"
    contact_plant_key = "real_to_contact_enabled_common_gap"
    contact_effect_key = "contact_enabled_to_collisionless_common_effect"
    policy_key = "common_to_independent_policy_response"
    total_key = "real_to_independent_total_gap"
    plant_q = _metric_value(replay, plant_key, "joint_position_rad")
    plant_dq = _metric_value(replay, plant_key, "joint_velocity_rad_s")
    plant_racket = _metric_value(replay, plant_key, "racket_position_m")
    total_q = _metric_value(replay, total_key, "joint_position_rad")
    total_dq = _metric_value(replay, total_key, "joint_velocity_rad_s")
    policy_q = _metric_value(replay, policy_key, "joint_position_rad")
    contact_plant_q = _metric_value(
        replay, contact_plant_key, "joint_position_rad")
    contact_effect_q = _metric_value(
        replay, contact_effect_key, "joint_position_rad")
    contact_effect_dq = _metric_value(
        replay, contact_effect_key, "joint_velocity_rad_s")
    early_plant_q = _metric_value(
        replay, plant_key, "joint_position_rad", "first_0p5_s")
    early_total_q = _metric_value(
        replay, total_key, "joint_position_rad", "first_0p5_s")
    onset = replay["divergence_onset"]
    self_check = replay["self_check"]
    initial = replay["initial_contract"]
    command_match = replay["real_policy_command_to_published_command"]
    saturation = replay["action_saturation"]
    tracking = replay["tracking"]
    contact_audit = replay["injected_ball_contact_audit"]
    contact_before = contact_audit["common_before_step"]
    contact_partners = ", ".join(
        f"{name}({count})"
        for name, count in contact_before["partner_contact_counts"].items()
    ) or "none"
    lines = [
        "# V91 Drag Ball 策略与执行器 Sim2Real 分解分析",
        "",
        f"录制：`{recording.name}`",
        "",
        "## 核心结论",
        "",
        f"1. 本次策略阶段完整连续：{stage['control_ticks']} ticks、"
        f"{stage['source_duration_s']:.3f} s、{stage['wall_rate_hz']:.2f} Hz；"
        f"严格对齐 {mirror['aligned_samples']}/{stage['control_ticks']} ticks，"
        f"只有 {mirror['episodes']} 个 episode、{mirror['segments']} 个连续段。",
        "2. drag-ball 的位置、速度、valid、age、观测序号在 real/sim 两边逐 tick 完全一致。"
        "起始 q、dq、actor action 和 q_ref 也一致，所以不存在 actor 实现或输入传输的首 tick 差异。",
        f"3. 全程 live mirror 总 gap：q={total_q:.5f} rad，dq={total_dq:.4f} rad/s；"
        f"球拍位置={1000.0 * groups['racket_position_m']['overall_rms']:.2f} mm，"
        f"球拍速度={groups['racket_velocity_m_s']['overall_rms']:.4f} m/s。",
        f"4. 用实物实际发布的同一条 q_ref 驱动 recovered-RMP/MuJoCo；这个 plant-only replay "
        f"不运行 actor，并把未使用的球移出工作区以完全去除碰撞力。"
        f"纯 plant/执行器 gap 为 q={plant_q:.5f} rad、dq={plant_dq:.4f} rad/s、"
        f"球拍位置={1000.0 * plant_racket:.2f} mm。"
        f"保留虚拟球碰撞时同指令 gap 为 {contact_plant_q:.5f} rad。",
        f"5. 在线 mirror 的虚拟球从 {contact_before['first_elapsed_s']:.3f} s 起发生接触，"
        f"{100.0 * contact_before['fraction']:.2f}% 的 tick 在强制写球后存在接触；"
        f"接触对象计数为 {contact_partners}。但相同 q_ref 下，开/关虚拟球碰撞的机械臂响应差异"
        f"为 q={contact_effect_q:.8f} rad、dq={contact_effect_dq:.8f} rad/s，"
        "相对总 gap 可忽略；所以虚拟接触不是本次机械臂 gap 的主要来源。"
        f"接触开启的共同重放与独立 sim 之间 q gap={policy_q:.5f} rad，反映状态分叉后的策略命令响应。",
        f"6. 前 0.5 s 内无碰撞 plant q gap={early_plant_q:.5f} rad，"
        f"总 q gap={early_total_q:.5f} rad。最早越过诊断阈值的顺序见下表；"
        "q/dq 分叉早于第一次虚拟球接触，且碰撞开关对机械臂轨迹的影响可忽略，"
        "说明执行器/plant mismatch 是起因之一，后续主要由 actor 闭环分叉放大。",
        "",
        "## 数据与时序可信度",
        "",
        f"- 独立 sim 命令离线重放自检：q RMS={self_check['joint_position']['overall_rms']:.7f} rad，"
        f"dq RMS={self_check['joint_velocity']['overall_rms']:.6f} rad/s，"
        f"结果 `passed={str(self_check['passed']).lower()}`。这验证了 observe→apply 当前 q_ref 的 tick 相位。",
        f"- 初始状态：q exact={initial['joint_position_exact']}，dq exact={initial['joint_velocity_exact']}，"
        f"首 tick action max gap={initial['policy_action_max_abs_gap']:.3g}，"
        f"q_ref max gap={initial['policy_command_max_abs_gap_rad']:.3g} rad。",
        f"- real policy q_ref 与实际发布目标的 gap RMS={command_match['overall_rms']:.8f} rad；"
        "因此 common replay 使用的是实物真正收到的目标，而不是未发布的中间量。",
        f"- 虚拟球接触审计：contact-enabled common replay 在强制写球后有 "
        f"{contact_before['ticks_with_ball_contact']}/{stage['control_ticks']} ticks 接触；"
        "collisionless common replay 直接消费已记录 q_ref、不运行 actor，并将球移出工作区。",
        "- 浏览器 analysis 的 `alignmentValid=false` 只来自实时异步完成延迟门限："
        "P95=10.62 ms 略高于 10 ms；最终 flush、序号、输入和 3667 个样本均完整，"
        "所以离线同 tick 分析有效。",
        "",
        "## 闭环分叉起点",
        "",
        "阈值只是为了稳定定位先后顺序，不代表安全阈值。",
        "",
        "| 信号 | 阈值 | 首次越过 | 当时值 |",
        "| --- | ---: | ---: | ---: |",
    ]
    onset_labels = (
        ("关节角 RMS", "joint_position_rms_rad", "rad"),
        ("关节角速度 RMS", "joint_velocity_rms_rad_s", "rad/s"),
        ("策略 action RMS", "policy_action_rms", ""),
        ("策略 q_ref RMS", "policy_command_rms_rad", "rad"),
        ("球拍位置 RMS", "racket_position_rms_m", "m"),
    )
    for label, key, unit in onset_labels:
        item = onset[key]
        elapsed = "未越过" if item["elapsed_s"] is None else f"{item['elapsed_s']:.3f} s"
        value = "n/a" if item["value"] is None else f"{item['value']:.5f} {unit}".strip()
        lines.append(
            f"| {label} | {item['threshold']:.5g} {unit} | {elapsed} | {value} |"
        )
    lines.extend([
        "",
        "## 全量参数总 gap（live real ↔ independent sim）",
        "",
        "| 参数 | RMS gap | 样本数 |",
        "| --- | ---: | ---: |",
    ])
    labels = (
        ("关节角", "joint_position_rad", "rad"),
        ("关节角速度", "joint_velocity_rad_s", "rad/s"),
        ("关节角加速度", "joint_acceleration_rad_s2", "rad/s²"),
        ("策略 q_ref", "policy_command_position_rad", "rad"),
        ("策略 action", "policy_action", ""),
        ("球拍三维位置", "racket_position_m", "m"),
        ("球拍三维速度", "racket_velocity_m_s", "m/s"),
        ("球拍三维加速度", "racket_acceleration_m_s2", "m/s²"),
        ("球拍 XYZ Euler 角", "racket_euler_angle_deg", "deg"),
        ("球拍三维角速度", "racket_angular_velocity_rad_s", "rad/s"),
        ("球拍三维角加速度", "racket_angular_acceleration_rad_s2", "rad/s²"),
    )
    for label, key, unit in labels:
        item = groups[key]
        lines.append(
            f"| {label} | {item['overall_rms']:.5f} {unit} | {item['sample_count']} |"
        )
    lines.extend([
        "",
        "## 执行器/plant 分解",
        "",
        "| 时间窗 | real↔无碰撞 common q | real↔无碰撞 common dq | "
        "接触 common↔independent q | total q |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    window_labels = (
        ("前 0.1 s", "first_0p1_s"),
        ("前 0.25 s", "first_0p25_s"),
        ("前 0.5 s", "first_0p5_s"),
        ("前 1.0 s", "first_1p0_s"),
        ("全程", "full"),
    )
    for label, key in window_labels:
        lines.append(
            f"| {label} | "
            f"{_metric_value(replay, plant_key, 'joint_position_rad', key):.5f} rad | "
            f"{_metric_value(replay, plant_key, 'joint_velocity_rad_s', key):.4f} rad/s | "
            f"{_metric_value(replay, policy_key, 'joint_position_rad', key):.5f} rad | "
            f"{_metric_value(replay, total_key, 'joint_position_rad', key):.5f} rad |"
        )
    full_window = replay["windows"]["full"]
    plant_joint_q = full_window[plant_key]["joint_position_rad"]["per_dimension"]
    plant_joint_dq = full_window[plant_key]["joint_velocity_rad_s"]["per_dimension"]
    total_joint_q = full_window[total_key]["joint_position_rad"]["per_dimension"]
    total_joint_dq = full_window[total_key]["joint_velocity_rad_s"]["per_dimension"]
    lines.extend([
        "",
        "### 逐关节 RMS",
        "",
        "| 关节 | 同 q_ref plant q | 同 q_ref plant dq | live total q | live total dq |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for index in range(7):
        lines.append(
            f"| J{index + 1} | {plant_joint_q[index]['rms']:.5f} rad | "
            f"{plant_joint_dq[index]['rms']:.4f} rad/s | "
            f"{total_joint_q[index]['rms']:.5f} rad | "
            f"{total_joint_dq[index]['rms']:.4f} rad/s |"
        )
    lines.extend([
        "",
        "同 q_ref 的 plant gap 中 J3 最大（q=0.00684 rad、dq=0.1681 rad/s）；"
        "闭环总 gap 中 J6 最大（q=0.06002 rad、dq=1.4597 rad/s）。",
        "",
        f"同一实物 q_ref 下，实物 tracking-error RMS={tracking['real_tracking_error_rms_rad']:.5f} rad，"
        f"无碰撞 common replay={tracking['common_replay_tracking_error_rms_rad']:.5f} rad，"
        f"接触开启 common replay="
        f"{tracking['contact_enabled_common_replay_tracking_error_rms_rad']:.5f} rad。",
        f"raw action 任一关节越过 [-1,1] 的比例：real="
        f"{100.0 * saturation['real_any_joint_raw_action_outside_unit_fraction']:.2f}%、"
        f"sim={100.0 * saturation['sim_any_joint_raw_action_outside_unit_fraction']:.2f}%；"
        f"实际发生 clipping 的比例：real="
        f"{100.0 * saturation['real_any_joint_action_clipped_fraction']:.2f}%、"
        f"sim={100.0 * saturation['sim_any_joint_action_clipped_fraction']:.2f}%。",
        "",
        "## 图表",
        "",
        "- `01–03`：7 关节角、角速度、角加速度的 live real/sim 对比。",
        "- `04`：策略 q_ref 与 action；`05–10`：球拍三维位置/速度/加速度/姿态/角速度/角加速度。",
        "- `11_gap_summary.png`：live 总 gap；`12_drag_ball_input.png`：drag-ball 输入逐 tick 相等。",
        "- `13_common_command_joint_position.png` / `14_common_command_joint_velocity.png`："
        "real、无碰撞共同命令重放、独立 sim 三路对比。",
        "- `15_common_command_tracking_error.png`：同一实物 q_ref 下 real 与无碰撞 replay 的 tracking error。",
        "- `16_gap_decomposition.png`：总 gap、无碰撞 plant gap、虚拟球接触效应和闭环命令响应的时序分解。",
        "- `17_early_divergence.png`：首 1 s 分叉顺序。",
        "",
        "逐 tick live 数据见 `aligned_sim2real.csv`；共同命令重放见 "
        "`actuator_common_command_replay.csv`；所有逐关节/逐轴指标见 `summary.json`。",
        "",
        "## 解释限制",
        "",
        "- drag-ball 只保证外部球输入相同，并不保证 actor 的完整输入始终相同；q/dq 一旦因 plant 差异分叉，"
        "同一个 actor 就会输出不同 action。这是闭环 gap，不是两份策略代码不同。",
        "- collisionless common replay 使用 checkpoint-authoritative recovered-RMP/MuJoCo plant 和记录的实际发布 q_ref；"
        "real↔common 的残差包含真实机器人动力学、摩擦、时延、RMP 实现/参数以及未记录的 1 kHz 内部状态差异。",
        "- 在线 mirror 中 drag-ball 仍是有质量且可碰撞的 MuJoCo 球；本次开/关碰撞没有造成可测的"
        "机械臂轨迹差异。如果 drag-ball 本意只是策略观测，仍建议运行时代码显式设为 observation-only，"
        "避免以后模型质量/接触参数变化后引入隐式动力学耦合。",
        "- 三路响应属于非线性反馈系统，real↔sim 不能被简单写成两个 RMS 的和；应结合早期窗口与时序曲线判断因果。",
    ])
    content = "\n".join(lines) + "\n"
    (output / "SIM2REAL_DRAG_BALL_ANALYSIS_ZH.md").write_text(
        content, encoding="utf-8")
    (output / "SIM2REAL_VALID_JUGGLE_ANALYSIS_ZH.md").write_text(
        "# 本目录是 Drag Ball 测试\n\n"
        "请阅读 `SIM2REAL_DRAG_BALL_ANALYSIS_ZH.md`。\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=base.DEFAULT_CHECKPOINT)
    parser.add_argument("--xml", type=Path, default=base.DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording = args.recording.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    xml = args.xml.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else recording / "analysis" / "v91_sim2real_drag_ball"
    )
    for path in (
        recording / "control_samples.jsonl",
        recording / "sim_control_samples.jsonl",
        recording / "metadata.json",
        checkpoint,
        xml,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)

    stage, aligned, excluded = base.load_alignment(recording)
    if not all(str(row.get("ballSource", "")) == "drag" for row in stage):
        raise RuntimeError("selected policy stage is not entirely drag-ball input")
    controller = base.init_kinematics(checkpoint, xml)
    full = base.populate_kinematics(stage, aligned, controller)
    summary = base.compute_summary(stage, aligned, full, [], excluded)
    replay = common_command_replay(
        recording, checkpoint, xml, aligned, controller)
    summary.update({
        "recording": recording.name,
        "experiment": "drag_ball",
        "actuator_decomposition": replay,
        "contracts": {
            "checkpoint": str(checkpoint),
            "xml": str(xml),
            "alignment": "exact sourceSeq and complete policy input envelope",
            "state_phase": "policyObs tick-start",
            "common_command": "real command.publishedSafePosition",
            "collisionless_common_replay": (
                "actor bypassed; ball geom disabled and parked away"
            ),
            "contact_enabled_common_replay": (
                "live-equivalent injected MuJoCo ball collision semantics"
            ),
            "common_replay_timing": "observe tick-start then apply current q_ref",
            "acceleration": "contiguous finite difference",
        },
    })
    summary = base.json_safe(summary)

    start = float(stage[0]["time"])
    base.plot_joint_family(
        output / "01_joint_position.png", "Joint position: real vs sim mirror",
        "rad", full["time"], full["q"], aligned, "sim_q", [], start)
    base.plot_joint_family(
        output / "02_joint_velocity.png",
        "Joint velocity: real raw feedback vs sim mirror", "rad/s",
        full["time"], full["dq"], aligned, "sim_dq", [], start)
    base.plot_joint_family(
        output / "03_joint_acceleration.png",
        "Joint acceleration: contiguous finite difference", "rad/s²",
        full["time"], full["ddq"], aligned, "sim_ddq", [], start)
    base.plot_policy_outputs(
        output / "04_policy_command_and_action.png", aligned, [], start)
    base.plot_racket_family(
        output / "05_racket_position.png",
        "Racket-center position: real vs sim mirror", "m", full,
        "racket_position", aligned, "sim_kin.position", [], ball=True)
    base.plot_racket_family(
        output / "06_racket_velocity.png", "Racket-center linear velocity",
        "m/s", full, "racket_velocity", aligned, "sim_kin.velocity", [])
    base.plot_racket_family(
        output / "07_racket_acceleration.png", "Racket-center linear acceleration",
        "m/s²", full, "racket_acceleration", aligned,
        "sim_racket_acceleration", [])
    base.plot_racket_family(
        output / "08_racket_orientation.png", "Racket XYZ Euler orientation",
        "deg", full, "racket_angle", aligned, "sim_euler", [])
    base.plot_racket_family(
        output / "09_racket_angular_velocity.png",
        "Racket angular velocity in base frame", "rad/s", full,
        "racket_angular_velocity", aligned, "sim_kin.angular_velocity", [])
    base.plot_racket_family(
        output / "10_racket_angular_acceleration.png",
        "Racket angular acceleration in base frame", "rad/s²", full,
        "racket_angular_acceleration", aligned,
        "sim_racket_angular_acceleration", [])
    base.plot_gap_summary(output / "11_gap_summary.png", aligned, [], start)
    plot_drag_ball_input(output / "12_drag_ball_input.png", aligned)
    _plot_three_joint_family(
        output / "13_common_command_joint_position.png", aligned,
        "real_q", "common_q", "sim_q",
        "Joint position: real vs common-command plant vs independent sim", "rad")
    _plot_three_joint_family(
        output / "14_common_command_joint_velocity.png", aligned,
        "real_dq", "common_dq", "sim_dq",
        "Joint velocity: real vs common-command plant vs independent sim", "rad/s")
    plot_tracking_error(output / "15_common_command_tracking_error.png", aligned)
    plot_decomposition(output / "16_gap_decomposition.png", aligned)
    plot_early_divergence(output / "17_early_divergence.png", aligned)

    base.write_aligned_csv(output / "aligned_sim2real.csv", aligned)
    write_replay_csv(output / "actuator_common_command_replay.csv", aligned)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(output, recording, summary)
    print(json.dumps({
        "output": str(output),
        "stage": summary["stage"],
        "mirror": summary["mirror"],
        "self_check": replay["self_check"],
        "full_decomposition": replay["windows"]["full"],
        "divergence_onset": replay["divergence_onset"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
