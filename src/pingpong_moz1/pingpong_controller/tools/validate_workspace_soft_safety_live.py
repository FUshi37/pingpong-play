#!/usr/bin/env python3
"""Short, watchdog-protected live validation through the probe HTTP API.

The target probe must already be connected to the physical simulator.  This
client is deliberately a heartbeat: every request renews the command lease;
any client failure lets the probe publish one measured-position hold and stop.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


X_LIMITS = (-0.298, 0.163)
Y_LIMITS = (-0.502, -0.290)
Z_LIMITS = (0.692, 1.462)


def _post(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError(f"unexpected {path} response")
    if decoded.get("error"):
        raise RuntimeError(str(decoded["error"]))
    return decoded


def _penetration(position: np.ndarray, anchor: np.ndarray) -> float:
    values = [
        X_LIMITS[0] - position[0],
        position[0] - X_LIMITS[1],
        Y_LIMITS[0] - position[1],
        position[1] - Y_LIMITS[1],
        Z_LIMITS[0] - position[2],
        position[2] - Z_LIMITS[1],
        float(np.linalg.norm(position - anchor)) - 1.1,
        (position[2] - anchor[2]) - 0.24,
        -0.12 - (position[2] - anchor[2]),
    ]
    return max(0.0, max(float(value) for value in values))


def _tangent_retention(snapshot: dict[str, Any]) -> float | None:
    soft = snapshot.get("workspaceSoft", {})
    filt = soft.get("filter", {}) if isinstance(soft, dict) else {}
    if not isinstance(filt, dict) or not filt.get("intervened", False):
        return None
    nominal = np.asarray(filt.get("taskVelocityMps", []), dtype=np.float64)
    selected = np.asarray(
        filt.get("selectedTaskVelocityMps", []), dtype=np.float64)
    terms = filt.get("terms", [])
    if nominal.shape != (3,) or selected.shape != (3,) or not terms:
        return None
    term = min(
        (item for item in terms if isinstance(item, dict)),
        key=lambda item: float(item.get("effectiveDistanceM", math.inf)),
        default=None,
    )
    if term is None:
        return None
    name = str(term.get("name", ""))
    normal = np.zeros(3, dtype=np.float64)
    if name.startswith("ee_x_"):
        normal[0] = 1.0 if name.endswith("high") else -1.0
    elif name.startswith("ee_y_"):
        normal[1] = 1.0 if name.endswith("high") else -1.0
    elif name in {"ee_z_high", "racket_z_high"}:
        normal[2] = 1.0
    elif name in {"ee_z_low", "racket_z_low"}:
        normal[2] = -1.0
    else:
        return None
    nominal_tangent = nominal - normal * float(np.dot(normal, nominal))
    selected_tangent = selected - normal * float(np.dot(normal, selected))
    denominator = float(np.linalg.norm(nominal_tangent))
    if denominator <= 1e-4:
        return None
    return float(np.linalg.norm(selected_tangent) / denominator)


def _payload(args: argparse.Namespace, publish: bool) -> dict[str, Any]:
    return {
        "ball": list(args.ball),
        "velocity": list(args.ball_velocity),
        "valid": True,
        "useRosState": True,
        "publishCommands": bool(publish),
        "commandLeaseMs": float(args.command_lease_ms),
        "workspaceSafetyMode": args.mode,
        "enableCaptureGate": False,
        "enableReleaseGate": False,
        "enableLostFailsafe": False,
        "enableDoneMonitoring": True,
        "enableCompensation": not args.disable_compensation,
        "enableSimMirror": False,
        "useBallEstimator": False,
        "ballSource": "drag",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.url.rstrip("/")
    samples: list[dict[str, Any]] = []
    positions: list[np.ndarray] = []
    tangent_retention: list[float] = []
    window: deque[tuple[float, np.ndarray]] = deque()
    max_penetration = 0.0
    max_step = 0.0
    max_100ms_displacement = 0.0
    max_joint_speed = 0.0
    max_feedback_age = 0.0
    max_command_error = 0.0
    infeasible_streak = 0
    max_infeasible_streak = 0
    abort_reason = ""
    armed_seen = False
    started = time.monotonic()
    next_heartbeat = started
    previous_position: np.ndarray | None = None

    output_path = Path(args.output).resolve() if args.output else None
    output_handle = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    try:
        while time.monotonic() - started < args.duration_s:
            now = time.monotonic()
            if now < next_heartbeat:
                time.sleep(next_heartbeat - now)
            snapshot = _post(base_url, "/step", _payload(args, True))
            now = time.monotonic()
            next_heartbeat = now + args.period_s
            runtime = snapshot.get("runtime", {})
            command_arm = snapshot.get("commandArm", {})
            armed = bool(
                isinstance(runtime, dict)
                and runtime.get("publishCommands", False)
                and isinstance(command_arm, dict)
                and command_arm.get("armed", False))
            armed_seen = bool(armed_seen or armed)
            if not armed_seen:
                if now - started > 3.0:
                    abort_reason = "command_arm_not_acknowledged"
                    break
                continue
            ros = snapshot.get("ros", {})
            soft = snapshot.get("workspaceSoft", {})
            filt = soft.get("filter", {}) if isinstance(soft, dict) else {}
            position = np.asarray(ros.get("jointEe", []), dtype=np.float64)
            anchor = np.asarray(snapshot.get("anchor", []), dtype=np.float64)
            joints = np.asarray(ros.get("jointPosition", []), dtype=np.float64)
            velocity = np.asarray(ros.get("jointVelocity", []), dtype=np.float64)
            age = ros.get("jointAgeS")
            command_error = ros.get("jointCommandErrorAbsMax")
            if (position.shape != (3,) or anchor.shape != (3,)
                    or joints.shape != (7,)
                    or velocity.shape != (7,)
                    or not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(joints))
                    or not np.all(np.isfinite(velocity))):
                abort_reason = "invalid_feedback"
                break
            age_value = math.inf if age is None else float(age)
            command_error_value = (
                math.inf if command_error is None else float(command_error))
            penetration = _penetration(position, anchor)
            max_penetration = max(max_penetration, penetration)
            max_joint_speed = max(
                max_joint_speed, float(np.max(np.abs(velocity))))
            max_feedback_age = max(max_feedback_age, age_value)
            max_command_error = max(max_command_error, command_error_value)
            if previous_position is not None:
                max_step = max(
                    max_step, float(np.linalg.norm(position - previous_position)))
            previous_position = position.copy()
            positions.append(position.copy())
            window.append((now, position.copy()))
            while window and now - window[0][0] > 0.10:
                window.popleft()
            if window:
                max_100ms_displacement = max(
                    max_100ms_displacement,
                    float(np.linalg.norm(position - window[0][1])),
                )
            qp = filt.get("qp", {}) if isinstance(filt, dict) else {}
            infeasible = bool(
                isinstance(qp, dict) and not qp.get("feasible", True))
            infeasible_streak = infeasible_streak + 1 if infeasible else 0
            max_infeasible_streak = max(
                max_infeasible_streak, infeasible_streak)
            retention = _tangent_retention(snapshot)
            if retention is not None and math.isfinite(retention):
                tangent_retention.append(retention)
            record = {
                "elapsedS": round(now - started, 6),
                "mode": snapshot.get("mode"),
                "positionM": position.tolist(),
                "jointVelocityRadS": velocity.tolist(),
                "feedbackAgeS": age_value,
                "commandErrorAbsMaxRad": command_error_value,
                "penetrationM": penetration,
                "workspaceFilter": filt,
                "runtime": snapshot.get("runtime", {}),
                "commandArm": snapshot.get("commandArm", {}),
            }
            samples.append(record)
            if output_handle is not None:
                output_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                output_handle.flush()

            if age_value > args.max_feedback_age_s:
                abort_reason = "stale_feedback"
            elif penetration > args.max_penetration_m:
                abort_reason = "workspace_penetration"
            elif max_100ms_displacement > args.max_100ms_displacement_m:
                abort_reason = "large_100ms_displacement"
            elif max_step > args.max_snapshot_step_m:
                abort_reason = "large_snapshot_step"
            elif command_error_value > args.max_command_error_rad:
                abort_reason = "large_command_tracking_error"
            elif max_infeasible_streak > args.max_infeasible_streak:
                abort_reason = "persistent_qp_infeasible"
            elif str(snapshot.get("mode", "")).endswith("failsafe"):
                abort_reason = "unexpected_failsafe"
            if abort_reason:
                break
    except (OSError, TimeoutError, urllib.error.URLError, RuntimeError) as exc:
        abort_reason = f"client_error:{exc}"
    finally:
        try:
            _post(base_url, "/step", _payload(args, False))
            time.sleep(0.10)
            _post(base_url, "/step", _payload(args, False))
        except Exception:
            pass
        if output_handle is not None:
            output_handle.close()

    final_snapshot = {}
    try:
        final_snapshot = _post(base_url, "/step", _payload(args, False))
    except Exception:
        pass
    return {
        "schemaVersion": 1,
        "mode": args.mode,
        "durationRequestedS": float(args.duration_s),
        "durationObservedS": round(time.monotonic() - started, 6),
        "samples": len(samples),
        "armed": bool(armed_seen),
        "passed": not bool(abort_reason),
        "abortReason": abort_reason,
        "maxPenetrationM": max_penetration,
        "maxSnapshotStepM": max_step,
        "max100msDisplacementM": max_100ms_displacement,
        "maxJointSpeedRadS": max_joint_speed,
        "maxFeedbackAgeS": max_feedback_age,
        "maxCommandErrorAbsMaxRad": max_command_error,
        "maxQpInfeasibleStreak": max_infeasible_streak,
        "medianTangentialRetention": (
            None if not tangent_retention
            else statistics.median(tangent_retention)),
        "startPositionM": None if not positions else positions[0].tolist(),
        "finalPositionM": None if not positions else positions[-1].tolist(),
        "finalPublishCommands": (
            final_snapshot.get("runtime", {}).get("publishCommands")
            if isinstance(final_snapshot, dict) else None),
        "finalCommandLeaseExpired": (
            final_snapshot.get("runtime", {}).get("commandLeaseExpired")
            if isinstance(final_snapshot, dict) else None),
        "output": None if output_path is None else str(output_path),
    }


def _vec3(text: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(","))
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("expected three finite comma-separated values")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument(
        "--mode", choices=("position_soft", "predictive_soft", "impedance_soft"),
        required=True)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--period-s", type=float, default=0.05)
    parser.add_argument("--command-lease-ms", type=float, default=400.0)
    parser.add_argument("--ball", type=_vec3, default=(-0.17, -0.45, 1.08))
    parser.add_argument("--ball-velocity", type=_vec3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--disable-compensation", action="store_true")
    parser.add_argument("--max-feedback-age-s", type=float, default=0.05)
    parser.add_argument("--max-penetration-m", type=float, default=0.010)
    parser.add_argument("--max-100ms-displacement-m", type=float, default=0.050)
    parser.add_argument("--max-snapshot-step-m", type=float, default=0.030)
    parser.add_argument("--max-command-error-rad", type=float, default=0.35)
    parser.add_argument("--max-infeasible-streak", type=int, default=10)
    parser.add_argument("--output", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
