#!/usr/bin/env python3
"""Offline nonlinear-FK/delayed-plant validation for workspace soft modes.

This deliberately drives the real deployment kinematics into a workspace
corner for several seconds, then reverses the policy.  It catches the former
stop/far-jump failure mode without publishing ROS commands.  The plant model is
conservative but simple: 72 ms command delay, first-order position tracking,
and the deployed joint velocity/acceleration limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pingpong_controller.mjx_rl_policy import MjxRLPolicyController
from pingpong_controller.rl_policy import target_right_arm_q_rad
from pingpong_controller.safety_limiter import RightArmCommandSafetyLimiter
from pingpong_controller.tools import interactive_policy_safety_probe as probe_module


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEFAULT_MODEL = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "outputs" / "rl_sim" / "inverse_compensation"
    / "18_stage4g_strong_contact_dr_inverse_mpc.pkl"
)
DEFAULT_XML = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "models" / "moz1_pd.xml"
)


class PlantFeedback:
    def __init__(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, dtype=np.float64).copy()
        self.dq = np.zeros_like(self.q)

    def get_joint_feedback(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.dq.copy()


def _make_probe(
    controller: MjxRLPolicyController,
    q0: np.ndarray,
    feedback: PlantFeedback,
    mode: str,
) -> Any:
    probe = probe_module.PolicySafetyProbe.__new__(
        probe_module.PolicySafetyProbe)
    probe.controller = controller
    probe.safety_limiter = RightArmCommandSafetyLimiter(q0, dt=0.005)
    probe.ros_bridge = feedback
    anchor, _quat, ok = controller._compute_racket_pose_quat_base(q0)
    if not ok:
        raise RuntimeError("deployment FK unavailable at reset pose")
    anchor = np.asarray(anchor, dtype=np.float64)
    probe.anchor = lambda: anchor.copy()
    probe.cfg = SimpleNamespace(
        done_racket_anchor_distance_m=1.1,
        done_racket_z_low_offset_m=-0.12,
        done_racket_z_high_offset_m=0.24,
        ee_x_limits_m=(-0.298, 0.163),
        ee_y_limits_m=(-0.502, -0.290),
        ee_z_limits_m=(0.692, 1.462),
        ee_b_limits_rad=None,
        ee_b_min_rad=None,
        ee_b_max_rad=None,
        ee_c_limits_rad=None,
        ee_c_min_rad=None,
        ee_c_max_rad=None,
        ee_orientation_reference_xyzw=None,
        ee_max_orientation_error_deg=0.0,
        workspace_soft_margin_m=0.05,
        workspace_soft_pull_speed_m_s=0.08,
        workspace_soft_damping=1.0,
        workspace_soft_max_correction_speed_m_s=0.30,
        workspace_soft_dls_damping=0.03,
        workspace_soft_nullspace_damping=0.0,
        workspace_soft_posture_gain_s=0.0,
        workspace_soft_boundary_joint_speed_scale=1.0,
        done_prediction_horizon_s=0.06,
        done_prediction_margin_m=0.005,
        comp_delay_ms=72.0,
        comp_actuator_tau_s=0.074,
        workspace_safety_mode=mode,
    )
    probe.done_monitor = SimpleNamespace(
        prediction_horizon_s=0.06,
        prediction_margin_m=0.005,
        last_prediction={},
    )
    probe.workspace_soft_prediction_active = False
    probe.workspace_soft_filter = {}
    probe.workspace_soft_entry_q = None
    probe.workspace_hold_active = False
    probe.workspace_hold_q = q0.copy()
    probe.workspace_hold_last_safe_q = q0.copy()
    probe.workspace_hold_trigger_reason = ""
    probe.workspace_hold_constraint = ""
    probe._workspace_actuation_delay_s = 0.072
    probe._workspace_actuator_tau_s = 0.074
    return probe


def _dls_joint_velocity(jacobian: np.ndarray, task_velocity: np.ndarray) -> np.ndarray:
    damping = 0.03
    task_matrix = jacobian @ jacobian.T + damping * damping * np.eye(3)
    return jacobian.T @ np.linalg.solve(task_matrix, task_velocity)


def _pose(controller: MjxRLPolicyController, q: np.ndarray) -> np.ndarray:
    position, _quat, ok = controller._compute_racket_pose_quat_base(q)
    if not ok:
        raise RuntimeError("deployment FK failed during validation")
    return np.asarray(position, dtype=np.float64)


def run_mode(
    controller: MjxRLPolicyController,
    mode: str,
    *,
    duration_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dt = 0.005
    q0 = target_right_arm_q_rad().astype(np.float64)
    feedback = PlantFeedback(q0)
    probe = _make_probe(controller, q0, feedback, mode)
    delay_steps = int(math.ceil(0.072 / dt))
    command_delay = deque(
        [q0.copy() for _ in range(delay_steps + 1)],
        maxlen=delay_steps + 1,
    )
    plant_tau_s = 0.074
    switch_s = 0.70 * duration_s
    rows: list[dict[str, Any]] = []
    previous_position = _pose(controller, feedback.q)
    previous_dq = feedback.dq.copy()
    max_task_step = 0.0
    max_joint_acc = 0.0
    max_command_step = 0.0
    infeasible_ticks = 0
    tangential_ratios: list[float] = []
    positions: list[np.ndarray] = []
    filter_durations_ms: list[float] = []

    for tick in range(int(round(duration_s / dt))):
        time_s = tick * dt
        position = _pose(controller, feedback.q)
        jacobian = probe._workspace_translation_jacobian(
            feedback.q, position)
        if jacobian is None:
            raise RuntimeError("numeric translation Jacobian unavailable")

        # Long outward push reaches +x. Near that face a bounded z oscillation
        # measures retained tangential policy performance.  Y is deliberately
        # not used: the deployment reset pose starts inside the Y-low soft band,
        # so Y is not an unconstrained tangent there.  Reversal exposes any
        # hidden accumulated target.
        tangent_velocity = (
            0.10 * math.sin(2.0 * math.pi * 0.7 * time_s)
            if 0.163 - position[0] < 0.03 else 0.0)
        task_velocity = (
            np.array([0.40, 0.0, tangent_velocity], dtype=np.float64)
            if time_s < switch_s
            else np.array([-0.25, 0.0, 0.0], dtype=np.float64)
        )
        nominal_joint_velocity = _dls_joint_velocity(jacobian, task_velocity)
        nominal_joint_velocity = np.clip(
            nominal_joint_velocity, -1.2, 1.2)
        # A distant target mimics policy integration + inverse compensation;
        # the filter must generate a local, dynamically reachable command.
        raw_target = (
            probe.safety_limiter.last_cmd
            + nominal_joint_velocity * 0.12)
        previous_command = probe.safety_limiter.last_cmd.copy()
        filter_started = time.perf_counter()
        safe = probe._filter_command_with_workspace_cbf(
            raw_target,
            profile="normal",
            mode=mode,
            apply_joint_limits=True,
        )
        filter_durations_ms.append(
            (time.perf_counter() - filter_started) * 1e3)
        max_command_step = max(
            max_command_step,
            float(np.max(np.abs(safe - previous_command))))
        qp = probe.workspace_soft_filter.get("qp", {})
        if isinstance(qp, dict) and not qp.get("feasible", True):
            infeasible_ticks += 1

        candidate_task = np.asarray(
            probe.workspace_soft_filter.get("taskVelocityMps", []),
            dtype=np.float64,
        )
        selected_task = np.asarray(
            probe.workspace_soft_filter.get("selectedTaskVelocityMps", []),
            dtype=np.float64,
        )
        if (candidate_task.shape == (3,) and selected_task.shape == (3,)
                and abs(candidate_task[2]) > 1e-4):
            x_hi_distance = 0.163 - position[0]
            if x_hi_distance < 0.02:
                tangential_ratios.append(float(
                    abs(selected_task[2]) / abs(candidate_task[2])))

        command_delay.append(safe.copy())
        delayed_target = command_delay[0]
        desired_dq = np.clip(
            (delayed_target - feedback.q) / plant_tau_s,
            -probe.safety_limiter.vel_limit_rad_s,
            probe.safety_limiter.vel_limit_rad_s,
        )
        dq_change = np.clip(
            desired_dq - feedback.dq,
            -probe.safety_limiter.acc_limit_rad_s2 * dt,
            probe.safety_limiter.acc_limit_rad_s2 * dt,
        )
        feedback.dq = feedback.dq + dq_change
        feedback.q = np.clip(
            feedback.q + feedback.dq * dt,
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        )

        next_position = _pose(controller, feedback.q)
        task_step = float(np.linalg.norm(next_position - previous_position))
        joint_acc = float(np.max(np.abs((feedback.dq - previous_dq) / dt)))
        max_task_step = max(max_task_step, task_step)
        max_joint_acc = max(max_joint_acc, joint_acc)
        previous_position = next_position.copy()
        previous_dq = feedback.dq.copy()
        positions.append(next_position.copy())
        rows.append({
            "mode": mode,
            "tick": tick,
            "time_s": time_s,
            "phase": "outward" if time_s < switch_s else "reverse",
            "x": float(next_position[0]),
            "y": float(next_position[1]),
            "z": float(next_position[2]),
            "command_step_rad": float(np.max(np.abs(safe - previous_command))),
            "filter_active": bool(probe.workspace_soft_filter.get("active", False)),
            "filter_intervened": bool(
                probe.workspace_soft_filter.get("intervened", False)),
            "qp_feasible": bool(qp.get("feasible", True))
            if isinstance(qp, dict) else True,
            "filter_duration_ms": float(filter_durations_ms[-1]),
            "emergency": bool(
                probe.workspace_soft_filter.get("emergency", False)),
            "qp_sweeps": int(qp.get("sweeps", 0))
            if isinstance(qp, dict) else 0,
        })

    position_array = np.asarray(positions)
    x_high_penetration = max(
        0.0, float(np.max(position_array[:, 0]) - 0.163))
    x_low_penetration = max(
        0.0, float(-0.298 - np.min(position_array[:, 0])))
    y_high_penetration = max(
        0.0, float(np.max(position_array[:, 1]) - (-0.290)))
    y_low_penetration = max(
        0.0, float(-0.502 - np.min(position_array[:, 1])))
    z_penetration = max(
        0.0,
        float(np.max(position_array[:, 2]) - 1.462),
        float(0.692 - np.min(position_array[:, 2])),
    )
    switch_tick = int(round(switch_s / dt))
    post_switch = position_array[switch_tick:]
    post_switch_steps = (
        np.linalg.norm(np.diff(post_switch, axis=0), axis=1)
        if len(post_switch) > 1 else np.zeros(1))
    max_signed_boundary_penetration = max(
        0.0,
        max(
            -float(distance)
            for position in position_array
            for _name, _normal, distance
            in probe._workspace_soft_boundary_terms(position)
        ),
    )
    metrics = {
        "mode": mode,
        "durationS": float(duration_s),
        "maxPenetrationM": float(max_signed_boundary_penetration),
        "xHighPenetrationM": x_high_penetration,
        "xLowPenetrationM": x_low_penetration,
        "yHighPenetrationM": y_high_penetration,
        "yLowPenetrationM": y_low_penetration,
        "zPenetrationM": z_penetration,
        "maxEeStepM": max_task_step,
        "maxEeStepAfterReverseM": float(np.max(post_switch_steps)),
        "maxCommandJointStepRad": max_command_step,
        "maxPlantJointAccelerationRadS2": max_joint_acc,
        "qpInfeasibleTicks": int(infeasible_ticks),
        "filterDurationP50Ms": float(np.percentile(
            filter_durations_ms, 50.0)),
        "filterDurationP95Ms": float(np.percentile(
            filter_durations_ms, 95.0)),
        "filterDurationP99Ms": float(np.percentile(
            filter_durations_ms, 99.0)),
        "filterDurationMaxMs": float(np.max(filter_durations_ms)),
        "filterDeadlineMissTicks": int(np.sum(
            np.asarray(filter_durations_ms) > dt * 1e3)),
        "medianBoundaryTangentialRetention": (
            None if not tangential_ratios
            else float(np.median(tangential_ratios))),
        "finalPositionM": [float(value) for value in position_array[-1]],
    }
    return metrics, rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        default="position_soft,predictive_soft,impedance_soft")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--robot-xml-path", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit nonzero if conservative offline safety thresholds fail.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in probe_module.WORKSPACE_SOFT_MODES]
    if unknown:
        raise SystemExit(f"unknown soft mode(s): {unknown}")
    controller = MjxRLPolicyController(
        str(args.model_path),
        dt=0.005,
        robot_xml_path=str(args.robot_xml_path),
        enable_delay_conditioning=True,
    )
    metrics: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for mode in modes:
        mode_metrics, rows = run_mode(
            controller, mode, duration_s=max(1.0, float(args.duration_s)))
        metrics.append(mode_metrics)
        all_rows.extend(rows)

    payload = {"schemaVersion": 1, "metrics": metrics}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n")
        with (args.output_dir / "trace.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)

    if args.check:
        failed = []
        for item in metrics:
            if item["maxPenetrationM"] > 0.015:
                failed.append(f"{item['mode']}: penetration")
            if item["maxEeStepAfterReverseM"] > 0.015:
                failed.append(f"{item['mode']}: reverse jump")
            if item["maxCommandJointStepRad"] > 0.027:
                failed.append(f"{item['mode']}: command jump")
            if item["filterDurationP99Ms"] > 5.0:
                failed.append(f"{item['mode']}: filter deadline")
            retention = item["medianBoundaryTangentialRetention"]
            if retention is not None and retention < 0.55:
                failed.append(f"{item['mode']}: tangent retention")
        if failed:
            print("FAILED: " + ", ".join(failed))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
