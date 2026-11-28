#!/usr/bin/env python3
"""MOZ1 base dynamics identification experiment runner (default: --dry-run).

Safety guarantees:
  1. Defaults strictly to ``--dry-run`` (zero ROS publishing, zero motion).
  2. Live execution requires ALL of:
     - explicit ``--live`` CLI flag;
     - ``CONFIRM_MOZ1_BASE_ID_LIVE=YES`` environment variable;
     - interactive terminal confirmation matching the exact ``--stage`` code;
     - valid interface contract with zero ``UNRESOLVED_REQUIRE_OPERATOR_FILL`` fields;
     - passing pre-flight telemetry & robot-status health check.
  3. Real-time geofence (dead-reckoned odom + optional mocap), tier clamping,
     100 ms staleness/monotonicity/NaN watchdog, and ``finally`` zero-velocity
     braking burst (>= 50 Hz for >= 0.5 s until stopped) on any exit/signal/error.
  4. Reuses repository ``pingpong_replay.cartesian_fk.RightArmFK`` for racket
     ``right_ee_site`` position & normal; fails loudly if unavailable.
  5. Supports both MOZ1 base command protocols confirmed on 172.16.0.20:
     - ``vector3_vel``: ``/mx_base_vel_command`` (``geometry_msgs/msg/Vector3``)
     - ``mech_unit_value_array``: ``/mx_base_cmd`` (``MechUnitValueArray``, ``mode=1``, ``twist``)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .contract import (
        ContractValidationError,
        compute_contract_sha256,
        default_contract_path,
        find_repo_root,
        load_contract,
        validate_contract,
    )
    from .safety import (
        DeadReckoningOdom,
        FeedbackSnapshot,
        LiveGateError,
        OutputExistsError,
        RealTimeSafetySupervisor,
        RobotStatusSnapshot,
        create_unique_run_dir,
        enforce_live_gates,
        execute_zero_brake_burst,
    )
    from .waveforms import (
        VALID_STAGES,
        WaveformTrial,
        build_stage_plan,
        tier_limits_from_contract,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from contract import (  # type: ignore[no-redef]
        ContractValidationError,
        compute_contract_sha256,
        default_contract_path,
        find_repo_root,
        load_contract,
        validate_contract,
    )
    from safety import (  # type: ignore[no-redef]
        DeadReckoningOdom,
        FeedbackSnapshot,
        LiveGateError,
        OutputExistsError,
        RealTimeSafetySupervisor,
        RobotStatusSnapshot,
        create_unique_run_dir,
        enforce_live_gates,
        execute_zero_brake_burst,
    )
    from waveforms import (  # type: ignore[no-redef]
        VALID_STAGES,
        WaveformTrial,
        build_stage_plan,
        tier_limits_from_contract,
    )

RUN_MANIFEST_SCHEMA_VERSION = "moz1_base_id_run_v1"


def load_existing_right_arm_fk(
    contract: Mapping[str, Any],
    *,
    repo_root: Optional[Path] = None,
) -> Any:
    """Import and instantiate the repository's existing ``RightArmFK``.

    Raises ``RuntimeError`` loudly if ``RightArmFK`` or the configured MJCF model
    cannot be loaded (never fabricates an unverified kinematics implementation).
    """
    root = (repo_root or find_repo_root()).resolve()
    replay_src = root / "src" / "pingpong_replay"
    if replay_src.is_dir() and str(replay_src) not in sys.path:
        sys.path.insert(0, str(replay_src))
    try:
        from pingpong_replay.cartesian_fk import RightArmFK
    except Exception as exc:
        raise RuntimeError(
            "Failed to import existing RightArmFK from pingpong_replay.cartesian_fk. "
            "Do not fabricate an unverified FK chain; ensure src/pingpong_replay is present "
            f"or built. Original error: {exc}"
        ) from exc

    arm_fb = contract["arm_feedback"]
    xml_path = Path(str(arm_fb["fk_model_xml"]))
    if not xml_path.is_absolute():
        xml_path = root / xml_path
    if not xml_path.is_file():
        raise RuntimeError(f"Configured FK MJCF XML does not exist: {xml_path}")
    joint_names = list(arm_fb["joint_names"])
    fk = RightArmFK(str(xml_path), joint_names=joint_names)
    return fk


def compute_racket_pose_and_normal(
    fk: Any,
    q_right_arm_rad: Sequence[float],
    *,
    normal_axis_index: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute ``right_ee_site`` position (m) and unit face normal via existing ``RightArmFK``."""
    q = np.asarray(q_right_arm_rad, dtype=np.float64).reshape(-1)
    T = np.asarray(fk.ee_pose(q), dtype=np.float64)
    pos_m = T[:3, 3].copy()
    normal = T[:3, int(normal_axis_index)].copy()
    norm = float(np.linalg.norm(normal))
    if norm > 1e-12:
        normal /= norm
    return pos_m, normal


def _git_metadata(repo_root: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"git_head": "unknown", "git_dirty": None}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if head.returncode == 0:
            meta["git_head"] = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        if status.returncode == 0:
            meta["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return meta


def build_base_command_payload(
    protocol: str,
    vx_m_s: float,
    vy_m_s: float,
    wz_rad_s: float,
    *,
    base_cmd_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a protocol-normalized dictionary representation of one base velocity command."""
    vx = float(vx_m_s)
    vy = float(vy_m_s)
    wz = float(wz_rad_s)
    if protocol == "vector3_vel":
        return {
            "protocol": "vector3_vel",
            "topic": str(base_cmd_cfg["topic"]),
            "x": vx,
            "y": vy,
            "z": wz,
        }
    if protocol == "mech_unit_value_array":
        return {
            "protocol": "mech_unit_value_array",
            "topic": str(base_cmd_cfg["topic"]),
            "name": str(base_cmd_cfg.get("mech_unit_name", "base")),
            "mode": int(base_cmd_cfg.get("mode", 1)),
            "frame_type": int(base_cmd_cfg.get("frame_type", 0)),
            "twist_linear": [vx, vy, 0.0],
            "twist_angular": [0.0, 0.0, wz],
        }
    raise ValueError(f"Unsupported base command protocol: {protocol!r}")


def populate_ros_base_command_msg(
    protocol: str,
    vx_m_s: float,
    vy_m_s: float,
    wz_rad_s: float,
    *,
    base_cmd_cfg: Mapping[str, Any],
    msg_classes: Mapping[str, Any],
    stamp_sec: int = 0,
    stamp_nanosec: int = 0,
) -> Any:
    """Instantiate and fill the ROS 2 message for either supported base command protocol."""
    vx = float(vx_m_s)
    vy = float(vy_m_s)
    wz = float(wz_rad_s)
    if protocol == "vector3_vel":
        msg = msg_classes["Vector3"]()
        msg.x = vx
        msg.y = vy
        msg.z = wz
        return msg
    if protocol == "mech_unit_value_array":
        arr = msg_classes["MechUnitValueArray"]()
        arr.header.stamp.sec = int(stamp_sec)
        arr.header.stamp.nanosec = int(stamp_nanosec)
        val = msg_classes["MechUnitValue"]()
        val.name = str(base_cmd_cfg.get("mech_unit_name", "base"))
        val.mode = int(base_cmd_cfg.get("mode", 1))
        val.cartesian.frame_type = int(base_cmd_cfg.get("frame_type", 0))
        val.cartesian.pose.orientation.w = 1.0
        val.cartesian.twist.linear.x = vx
        val.cartesian.twist.linear.y = vy
        val.cartesian.twist.linear.z = 0.0
        val.cartesian.twist.angular.x = 0.0
        val.cartesian.twist.angular.y = 0.0
        val.cartesian.twist.angular.z = wz
        arr.values = [val]
        return arr
    raise ValueError(f"Unsupported base command protocol: {protocol!r}")


def populate_ros_arm_hold_msg(
    protocol: str,
    q_target_rad: Sequence[float],
    *,
    arm_cmd_cfg: Mapping[str, Any],
    msg_classes: Mapping[str, Any],
    stamp_sec: int = 0,
    stamp_nanosec: int = 0,
) -> Any:
    """Instantiate and fill the ROS 2 right-arm hold command for s5_arm_concurrency."""
    q_list = [float(v) for v in q_target_rad]
    if protocol == "mech_unit_value_array":
        arr = msg_classes["MechUnitValueArray"]()
        arr.header.stamp.sec = int(stamp_sec)
        arr.header.stamp.nanosec = int(stamp_nanosec)
        val = msg_classes["MechUnitValue"]()
        val.name = str(arm_cmd_cfg.get("mech_unit_name", "right_arm"))
        val.mode = int(arm_cmd_cfg.get("mode", 0))
        val.joint.position = q_list
        arr.values = [val]
        return arr
    if protocol == "mech_unit_cmd_array":
        arr = msg_classes["MechUnitCmdArray"]()
        arr.header.stamp.sec = int(stamp_sec)
        arr.header.stamp.nanosec = int(stamp_nanosec)
        cmd = msg_classes["MechUnitCmd"]()
        cmd.mu_idx = int(arm_cmd_cfg.get("right_arm_mu_idx", 3))
        cmd.use_jnt = True
        cmd.jnt_pos = q_list
        arr.cmds = [cmd]
        return arr
    raise ValueError(f"Unsupported arm command protocol: {protocol!r}")


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


def _rotmat_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (float(qx / n), float(qy / n), float(qz / n), float(qw / n))


def _rotmat_to_rpy_rad(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract intrinsic ZYX (roll, pitch, yaw) in radians from a 3x3 rotation matrix."""
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return (roll, pitch, yaw)


def _quat_xyzw_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _project_so3(M: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    R = u @ vt
    if np.linalg.det(R) < 0.0:
        u[:, -1] *= -1.0
        R = u @ vt
    return R


def load_mocap_calibration_matrix(
    calibration_path: Optional[str],
    *,
    repo_root: Optional[Path] = None,
) -> Tuple[np.ndarray, bool]:
    """Load 4x4 ``T_robot_mocap`` homogeneous matrix if configured and present."""
    if not calibration_path:
        return np.eye(4, dtype=np.float64), False
    p = Path(str(calibration_path))
    if not p.is_absolute():
        p = (repo_root or find_repo_root()) / p
    if not p.is_file():
        return np.eye(4, dtype=np.float64), False
    mat = np.loadtxt(str(p), dtype=np.float64)
    if mat.shape != (4, 4) or not np.all(np.isfinite(mat)):
        raise ValueError(f"Invalid 4x4 mocap calibration matrix in {p}")
    return mat, True


def decode_mocap_base_datagram(
    data: bytes,
    mocap_cfg: Mapping[str, Any],
    *,
    T_robot_mocap: Optional[np.ndarray] = None,
    R_ref_robot: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Decode one mocap UDP packet for either a 4-ball rigid body or a single marker.

    Reuses ``pingpong_controller.mocap_ball_detector.parse_mocap_packet`` directly and
    transforms BOTH 3D position and rigid-body orientation into the Z-up robot frame
    using ``T_robot_mocap`` (critical because OptiTrack mocap is Y-up), while also
    supporting an optional stationary reference rotation ``R_ref_robot`` so that
    arbitrarily mounted (including 90-deg vertical) fixed K-shaped rigid bodies have
    zero Euler gimbal lock.
    """
    from pingpong_controller.mocap_ball_detector import parse_mocap_packet

    track_mode = str(mocap_cfg.get("base_track_mode") or "rigid_body").strip().lower()
    pos_unit = str(mocap_cfg.get("position_unit", "auto"))
    ts_unit = str(mocap_cfg.get("timestamp_unit", "s"))

    if track_mode == "marker":
        bounds = mocap_cfg.get("base_marker_observation_bounds_m")
        marker_id = int(mocap_cfg.get("base_marker_id") or 1)
        pkt = parse_mocap_packet(
            data,
            marker_id=marker_id,
            object_kind="marker",
            position_unit=pos_unit,
            timestamp_unit=ts_unit,
            observation_bounds_m=bounds,
        )
    elif track_mode == "rigid_body":
        slot = mocap_cfg.get("base_rigid_body_slot")
        rb_id = mocap_cfg.get("base_rigid_body_id")
        rb_bounds = mocap_cfg.get("base_rigid_body_observation_bounds_m")
        pkt = parse_mocap_packet(
            data,
            object_name=mocap_cfg.get("base_rigid_body_name"),
            object_kind="rigid_body",
            rigid_body_slot=None if slot is None else int(slot),
            rigid_body_id=None if rb_id is None else int(rb_id),
            position_unit=pos_unit,
            timestamp_unit=ts_unit,
            rigid_body_observation_bounds_m=rb_bounds,
        )
    else:
        raise ValueError(f"Unsupported mocap.base_track_mode={track_mode!r}")

    if pkt is None or pkt.tracked is False:
        return None
    pos_raw = np.asarray(pkt.position_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pos_raw)):
        return None

    T = np.eye(4, dtype=np.float64) if T_robot_mocap is None else np.asarray(T_robot_mocap, dtype=np.float64)
    R_cal_frame = _project_so3(T[:3, :3])
    pos_cal = (T[:3, :3] @ pos_raw) + T[:3, 3]

    vel_cal: Optional[List[float]] = None
    if pkt.velocity_m_s is not None:
        v_raw = np.asarray(pkt.velocity_m_s, dtype=np.float64).reshape(3)
        if np.all(np.isfinite(v_raw)):
            vel_cal = (R_cal_frame @ v_raw).tolist()

    quat_xyzw: Optional[List[float]] = None
    rpy_rad: Optional[List[float]] = None
    rotmat_robot: Optional[np.ndarray] = None
    has_yaw = False
    pos_center_cal = pos_cal.copy()
    if pkt.orientation_wxyz is not None:
        qw, qx, qy, qz = [float(v) for v in np.asarray(pkt.orientation_wxyz, dtype=np.float64).reshape(4)]
        if all(math.isfinite(v) for v in (qw, qx, qy, qz)):
            R_mocap_rb = _quat_xyzw_to_rotmat(qx, qy, qz, qw)
            # Change-of-basis similarity transform from Y-up mocap world into Z-up robot world:
            # maps pure mocap +Y rotation to pure robot +Z yaw and preserves identity (roll=pitch=yaw=0).
            R_robot_rb = R_cal_frame @ R_mocap_rb @ R_cal_frame.T
            rotmat_robot = R_robot_rb
            q_cal = _rotmat_to_quat_xyzw(R_robot_rb)
            # Right-multiply by R_ref_robot^T so any fixed K-plate mounting angle (even 90-deg
            # vertical on a chassis side wall) starts at identity with zero Euler gimbal lock.
            R_leveled = R_robot_rb if R_ref_robot is None else (R_robot_rb @ R_ref_robot.T)
            roll, pitch, yaw = _rotmat_to_rpy_rad(R_leveled)
            quat_xyzw = list(q_cal)
            rpy_rad = [roll, pitch, yaw]
            has_yaw = True
            # Optional real-time lever-arm subtraction for geofence accuracy when configured.
            lever = mocap_cfg.get("base_lever_arm_xy_m")
            if isinstance(lever, (list, tuple)) and len(lever) == 2:
                rx, ry = float(lever[0]), float(lever[1])
                if abs(rx) > 1e-9 or abs(ry) > 1e-9:
                    c_y, s_y = math.cos(yaw), math.sin(yaw)
                    pos_center_cal[0] -= (c_y * rx - s_y * ry)
                    pos_center_cal[1] -= (s_y * rx + c_y * ry)

    return {
        "track_mode": track_mode,
        "pos_xyz_m": pos_cal.tolist(),
        "pos_center_xyz_m": pos_center_cal.tolist(),
        "pos_raw_mocap_m": pos_raw.tolist(),
        "quat_xyzw": quat_xyzw,
        "rpy_rad": rpy_rad,
        "rotmat_robot": rotmat_robot,
        "has_yaw": has_yaw,
        "vel_xyz_m_s": vel_cal,
        "header_time_s": None if pkt.timestamp_s is None else float(pkt.timestamp_s),
        "frame_idx": pkt.frame_idx,
    }


class TelemetryStore:
    """Thread-safe event logger and latest-sample cache for live & mock backends."""

    def __init__(
        self,
        *,
        raw_events_path: Path,
        run_t0_mono_s: float,
        fk: Any,
        racket_normal_axis_index: int = 2,
    ) -> None:
        self._lock = threading.Lock()
        self._raw_fh = raw_events_path.open("w", encoding="utf-8")
        self.run_t0_mono_s = float(run_t0_mono_s)
        self.fk = fk
        self.racket_normal_axis_index = int(racket_normal_axis_index)

        self.current_trial_id: str = "IDLE"
        self.odom = DeadReckoningOdom()
        self.latest_base: Optional[FeedbackSnapshot] = None
        self.latest_arm: Optional[FeedbackSnapshot] = None
        self.latest_imu: Optional[FeedbackSnapshot] = None
        self.latest_mocap_base: Optional[FeedbackSnapshot] = None
        self.latest_mocap_racket: Optional[FeedbackSnapshot] = None
        self.latest_robot_status: Optional[RobotStatusSnapshot] = None

        self.source_counts: Dict[str, int] = {}
        self._clock_deltas: Dict[str, List[float]] = {}
        self.trial_events: Dict[str, List[Dict[str, Any]]] = {}

    def close(self) -> None:
        with self._lock:
            if not self._raw_fh.closed:
                self._raw_fh.flush()
                self._raw_fh.close()

    def set_trial(self, trial_id: str, *, reset_odom: bool = False) -> None:
        with self._lock:
            self.current_trial_id = trial_id
            if reset_odom:
                self.odom.reset(0.0, 0.0, 0.0)
                if (
                    self.latest_base is not None
                    and self.latest_base.extra.get("pose_source") != "cartesian_pose_world"
                ):
                    self.latest_base.pose_xyyaw = (0.0, 0.0, 0.0)
            self.trial_events.setdefault(trial_id, [])

    def _record_event_locked(self, event: Dict[str, Any]) -> None:
        ev_type = str(event["event_type"])
        self.source_counts[ev_type] = self.source_counts.get(ev_type, 0) + 1
        header_t = event.get("header_time_s")
        epoch_t = event.get("epoch_time_s")
        if header_t is not None and epoch_t is not None:
            if math.isfinite(float(header_t)) and math.isfinite(float(epoch_t)):
                delta = float(header_t) - float(epoch_t)
                deltas = self._clock_deltas.setdefault(ev_type, [])
                if len(deltas) < 50000:
                    deltas.append(delta)
        self._raw_fh.write(json.dumps(event, ensure_ascii=True) + "\n")
        tid = str(event.get("trial_id", self.current_trial_id))
        if tid in self.trial_events:
            self.trial_events[tid].append(event)

    def flush(self) -> None:
        with self._lock:
            if not self._raw_fh.closed:
                self._raw_fh.flush()

    def record_generic_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        wall_time_s: float,
        epoch_time_s: float,
        header_time_s: Optional[float] = None,
        trial_id: Optional[str] = None,
    ) -> None:
        with self._lock:
            ev = {
                "event_type": event_type,
                "trial_id": trial_id or self.current_trial_id,
                "wall_time_s": float(wall_time_s - self.run_t0_mono_s),
                "epoch_time_s": float(epoch_time_s),
                "header_time_s": None if header_time_s is None else float(header_time_s),
                **dict(payload),
            }
            self._record_event_locked(ev)

    def on_base_state(
        self,
        *,
        wall_time_s: float,
        epoch_time_s: float,
        header_time_s: Optional[float],
        vx_m_s: float,
        vy_m_s: float,
        wz_rad_s: float,
        wheel_pos_rad: Sequence[float] = (),
        wheel_vel_rad_s: Sequence[float] = (),
        wheel_effort_nm: Sequence[float] = (),
        reported_pose_xyyaw: Optional[Tuple[float, float, float]] = None,
        pose_source: str = "integrate_base_twist",
        source_name: str = "base_state",
    ) -> None:
        with self._lock:
            rel_t = float(wall_time_s - self.run_t0_mono_s)
            dr_pose = self.odom.update(rel_t, float(vx_m_s), float(vy_m_s), float(wz_rad_s))
            active_pose = (
                reported_pose_xyyaw
                if (pose_source == "cartesian_pose_world" and reported_pose_xyyaw is not None)
                else dr_pose
            )
            vals = np.array([vx_m_s, vy_m_s, wz_rad_s], dtype=np.float64)
            self.latest_base = FeedbackSnapshot(
                wall_time_s=float(wall_time_s),
                header_time_s=header_time_s,
                values=vals,
                pose_xyyaw=active_pose,
                extra={
                    "wheel_vel_rad_s": list(wheel_vel_rad_s),
                    "reported_pose_xyyaw": reported_pose_xyyaw,
                    "pose_source": pose_source,
                },
            )
            self._record_event_locked({
                "event_type": source_name,
                "trial_id": self.current_trial_id,
                "wall_time_s": rel_t,
                "epoch_time_s": float(epoch_time_s),
                "header_time_s": None if header_time_s is None else float(header_time_s),
                "vx_m_s": float(vx_m_s),
                "vy_m_s": float(vy_m_s),
                "wz_rad_s": float(wz_rad_s),
                "odom_x_m": float(active_pose[0]),
                "odom_y_m": float(active_pose[1]),
                "odom_yaw_rad": float(active_pose[2]),
                "wheel_pos_rad": [float(v) for v in wheel_pos_rad],
                "wheel_vel_rad_s": [float(v) for v in wheel_vel_rad_s],
                "wheel_effort_nm": [float(v) for v in wheel_effort_nm],
            })

    def on_arm_state(
        self,
        *,
        wall_time_s: float,
        epoch_time_s: float,
        header_time_s: Optional[float],
        q_rad: Sequence[float],
        dq_rad_s: Sequence[float],
        effort_nm: Sequence[float] = (),
        source_name: str = "arm_state",
    ) -> None:
        q_arr = np.asarray(q_rad, dtype=np.float64)
        dq_arr = np.asarray(dq_rad_s, dtype=np.float64)
        if q_arr.size == 7 and np.all(np.isfinite(q_arr)):
            racket_pos, racket_norm = compute_racket_pose_and_normal(
                self.fk, q_arr, normal_axis_index=self.racket_normal_axis_index
            )
            racket_pos_list = racket_pos.tolist()
            racket_norm_list = racket_norm.tolist()
        else:
            racket_pos_list = [math.nan, math.nan, math.nan]
            racket_norm_list = [math.nan, math.nan, math.nan]

        with self._lock:
            rel_t = float(wall_time_s - self.run_t0_mono_s)
            vals = np.concatenate([q_arr.reshape(-1), dq_arr.reshape(-1)])
            self.latest_arm = FeedbackSnapshot(
                wall_time_s=float(wall_time_s),
                header_time_s=header_time_s,
                values=vals,
                extra={"q_rad": q_arr.tolist(), "dq_rad_s": dq_arr.tolist()},
            )
            self._record_event_locked({
                "event_type": source_name,
                "trial_id": self.current_trial_id,
                "wall_time_s": rel_t,
                "epoch_time_s": float(epoch_time_s),
                "header_time_s": None if header_time_s is None else float(header_time_s),
                "q_rad": q_arr.tolist(),
                "dq_rad_s": dq_arr.tolist(),
                "effort_nm": [float(v) for v in effort_nm],
                "racket_pos_m": racket_pos_list,
                "racket_normal": racket_norm_list,
            })

    def on_imu_state(
        self,
        *,
        wall_time_s: float,
        epoch_time_s: float,
        header_time_s: Optional[float],
        rpy_rad: Sequence[float],
    ) -> None:
        arr = np.asarray(rpy_rad, dtype=np.float64).reshape(3)
        with self._lock:
            rel_t = float(wall_time_s - self.run_t0_mono_s)
            self.latest_imu = FeedbackSnapshot(
                wall_time_s=float(wall_time_s),
                header_time_s=header_time_s,
                values=arr,
            )
            self._record_event_locked({
                "event_type": "imu_state",
                "trial_id": self.current_trial_id,
                "wall_time_s": rel_t,
                "epoch_time_s": float(epoch_time_s),
                "header_time_s": None if header_time_s is None else float(header_time_s),
                "imu_rpy_rad": arr.tolist(),
            })

    def on_mocap_pose(
        self,
        *,
        kind: str,
        wall_time_s: float,
        epoch_time_s: float,
        header_time_s: Optional[float],
        pos_xyz_m: Sequence[float],
        pos_center_xyz_m: Optional[Sequence[float]] = None,
        quat_xyzw: Optional[Sequence[float]] = None,
        rpy_rad: Optional[Sequence[float]] = None,
        pos_raw_mocap_m: Optional[Sequence[float]] = None,
        vel_xyz_m_s: Optional[Sequence[float]] = None,
        track_mode: str = "rigid_body",
    ) -> None:
        pos = np.asarray(pos_xyz_m, dtype=np.float64).reshape(3)
        pos_fence = pos if pos_center_xyz_m is None else np.asarray(pos_center_xyz_m, dtype=np.float64).reshape(3)
        if quat_xyzw is not None:
            quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
            if rpy_rad is not None:
                rpy_list: Optional[List[float]] = [float(v) for v in rpy_rad]
                yaw: Optional[float] = float(rpy_list[2])
            else:
                R = _quat_xyzw_to_rotmat(quat[0], quat[1], quat[2], quat[3])
                r_a, p_a, y_a = _rotmat_to_rpy_rad(R)
                rpy_list = [r_a, p_a, y_a]
                yaw = y_a
            has_yaw = True
            snap_vals = np.concatenate([pos_fence, quat])
            quat_list: Optional[List[float]] = quat.tolist()
        else:
            yaw = None
            rpy_list = None
            has_yaw = False
            snap_vals = pos_fence
            quat_list = None
        with self._lock:
            rel_t = float(wall_time_s - self.run_t0_mono_s)
            snap = FeedbackSnapshot(
                wall_time_s=float(wall_time_s),
                header_time_s=header_time_s,
                values=snap_vals,
                pose_xyyaw=(float(pos_fence[0]), float(pos_fence[1]), 0.0 if yaw is None else float(yaw)),
                extra={"has_yaw": has_yaw, "track_mode": track_mode},
            )
            if kind == "mocap_base":
                self.latest_mocap_base = snap
            else:
                self.latest_mocap_racket = snap
            self._record_event_locked({
                "event_type": kind,
                "trial_id": self.current_trial_id,
                "wall_time_s": rel_t,
                "epoch_time_s": float(epoch_time_s),
                "header_time_s": None if header_time_s is None else float(header_time_s),
                "track_mode": track_mode,
                "pos_xyz_m": pos.tolist(),
                "pos_center_xyz_m": pos_fence.tolist(),
                "pos_raw_mocap_m": None if pos_raw_mocap_m is None else [float(v) for v in pos_raw_mocap_m],
                "quat_xyzw": quat_list,
                "rpy_rad": rpy_list,
                "yaw_rad": yaw,
                "vel_xyz_m_s": None if vel_xyz_m_s is None else [float(v) for v in vel_xyz_m_s],
            })

    def on_robot_status(self, status: RobotStatusSnapshot, *, epoch_time_s: float) -> None:
        with self._lock:
            self.latest_robot_status = status
            self._record_event_locked({
                "event_type": "robot_status",
                "trial_id": self.current_trial_id,
                "wall_time_s": float(status.wall_time_s - self.run_t0_mono_s),
                "epoch_time_s": float(epoch_time_s),
                "header_time_s": None,
                "safety_status": status.safety_status,
                "motor_status": list(status.motor_status),
                "control_mode": list(status.control_mode),
                "task_mode": status.task_mode,
                "motion_status": status.motion_status,
            })

    def get_latest_snapshots(
        self,
    ) -> Tuple[
        Optional[FeedbackSnapshot],
        Optional[FeedbackSnapshot],
        Optional[FeedbackSnapshot],
        Optional[RobotStatusSnapshot],
        Tuple[float, float, float],
    ]:
        with self._lock:
            active_pose = (
                self.latest_base.pose_xyyaw
                if (self.latest_base is not None and self.latest_base.pose_xyyaw is not None)
                else self.odom.pose()
            )
            return (
                self.latest_base,
                self.latest_arm,
                self.latest_mocap_base,
                self.latest_robot_status,
                active_pose,
            )

    def clock_offset_summary(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            summary: Dict[str, Dict[str, Any]] = {}
            for src, deltas in sorted(self._clock_deltas.items()):
                arr = np.asarray(deltas, dtype=np.float64)
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                summary[src] = {
                    "n_samples": int(arr.size),
                    "median_header_minus_epoch_s": round(med, 6),
                    "mad_s": round(mad, 6),
                    "min_s": round(float(np.min(arr)), 6),
                    "max_s": round(float(np.max(arr)), 6),
                }
            return summary


def compile_aligned_timeseries_npz(
    trials: Sequence[WaveformTrial],
    trial_events: Mapping[str, Sequence[Mapping[str, Any]]],
    out_npz_path: Path,
) -> None:
    """Build uniformly aligned per-trial arrays on the monotonic wall-time axis."""
    arrays: Dict[str, np.ndarray] = {}
    for tr in trials:
        events = list(trial_events.get(tr.trial_id, ()))
        cmd_evs = [e for e in events if e.get("event_type") == "base_cmd"]
        base_evs = [e for e in events if e.get("event_type") == "base_state"]
        arm_evs = [e for e in events if e.get("event_type") == "arm_state"]
        arm_cmd_evs = [e for e in events if e.get("event_type") == "arm_cmd"]
        imu_evs = [e for e in events if e.get("event_type") == "imu_state"]
        mocap_evs = [e for e in events if e.get("event_type") == "mocap_base"]

        if not cmd_evs:
            continue

        t_cmd = np.array([float(e["wall_time_s"]) for e in cmd_evs], dtype=np.float64)
        t0 = float(t_cmd[0])
        t_rel = t_cmd - t0

        prefix = tr.trial_id
        arrays[f"{prefix}__wall_time_s"] = t_cmd
        arrays[f"{prefix}__t_rel_s"] = t_rel
        arrays[f"{prefix}__cmd_vx_m_s"] = np.array([float(e["cmd_vx_m_s"]) for e in cmd_evs], dtype=np.float64)
        arrays[f"{prefix}__cmd_vy_m_s"] = np.array([float(e["cmd_vy_m_s"]) for e in cmd_evs], dtype=np.float64)
        arrays[f"{prefix}__cmd_wz_rad_s"] = np.array([float(e["cmd_wz_rad_s"]) for e in cmd_evs], dtype=np.float64)
        arrays[f"{prefix}__phase_labels"] = np.array([str(e.get("phase", "")) for e in cmd_evs])

        def _interp_series(evs: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
            if not evs:
                return np.full(t_cmd.shape, np.nan, dtype=np.float64)
            ts = np.array([float(e["wall_time_s"]) for e in evs], dtype=np.float64)
            vs = np.array([float(e[key]) for e in evs], dtype=np.float64)
            order = np.argsort(ts)
            ts, vs = ts[order], vs[order]
            return np.interp(t_cmd, ts, vs, left=np.nan, right=np.nan)

        def _interp_vec(evs: Sequence[Mapping[str, Any]], key: str, dim: int) -> np.ndarray:
            out = np.full((len(t_cmd), dim), np.nan, dtype=np.float64)
            valid = [e for e in evs if isinstance(e.get(key), list) and len(e[key]) == dim]
            if not valid:
                return out
            ts = np.array([float(e["wall_time_s"]) for e in valid], dtype=np.float64)
            mat = np.array([e[key] for e in valid], dtype=np.float64)
            order = np.argsort(ts)
            ts, mat = ts[order], mat[order]
            for d in range(dim):
                out[:, d] = np.interp(t_cmd, ts, mat[:, d], left=np.nan, right=np.nan)
            return out

        arrays[f"{prefix}__meas_vx_m_s"] = _interp_series(base_evs, "vx_m_s")
        arrays[f"{prefix}__meas_vy_m_s"] = _interp_series(base_evs, "vy_m_s")
        arrays[f"{prefix}__meas_wz_rad_s"] = _interp_series(base_evs, "wz_rad_s")
        arrays[f"{prefix}__odom_x_m"] = _interp_series(base_evs, "odom_x_m")
        arrays[f"{prefix}__odom_y_m"] = _interp_series(base_evs, "odom_y_m")
        arrays[f"{prefix}__odom_yaw_rad"] = _interp_series(base_evs, "odom_yaw_rad")
        arrays[f"{prefix}__wheel_vel_rad_s"] = _interp_vec(base_evs, "wheel_vel_rad_s", 4)

        arrays[f"{prefix}__arm_q_rad"] = _interp_vec(arm_evs, "q_rad", 7)
        arrays[f"{prefix}__arm_dq_rad_s"] = _interp_vec(arm_evs, "dq_rad_s", 7)
        arrays[f"{prefix}__racket_pos_m"] = _interp_vec(arm_evs, "racket_pos_m", 3)
        arrays[f"{prefix}__racket_normal"] = _interp_vec(arm_evs, "racket_normal", 3)
        arrays[f"{prefix}__imu_rpy_rad"] = _interp_vec(imu_evs, "imu_rpy_rad", 3)

        if base_evs:
            arrays[f"{prefix}__base_raw_wall_s"] = np.array([float(e["wall_time_s"]) for e in base_evs], dtype=np.float64)
            arrays[f"{prefix}__base_raw_epoch_s"] = np.array([float(e["epoch_time_s"]) for e in base_evs], dtype=np.float64)
            arrays[f"{prefix}__base_raw_header_s"] = np.array(
                [np.nan if e["header_time_s"] is None else float(e["header_time_s"]) for e in base_evs],
                dtype=np.float64,
            )
            arrays[f"{prefix}__base_raw_vx_m_s"] = np.array([float(e["vx_m_s"]) for e in base_evs], dtype=np.float64)
            arrays[f"{prefix}__base_raw_vy_m_s"] = np.array([float(e["vy_m_s"]) for e in base_evs], dtype=np.float64)
            arrays[f"{prefix}__base_raw_wz_rad_s"] = np.array([float(e["wz_rad_s"]) for e in base_evs], dtype=np.float64)

        if arm_cmd_evs:
            arrays[f"{prefix}__arm_cmd_wall_s"] = np.array([float(e["wall_time_s"]) for e in arm_cmd_evs], dtype=np.float64)
        if arm_evs:
            arrays[f"{prefix}__arm_raw_wall_s"] = np.array([float(e["wall_time_s"]) for e in arm_evs], dtype=np.float64)
            arrays[f"{prefix}__arm_raw_epoch_s"] = np.array([float(e["epoch_time_s"]) for e in arm_evs], dtype=np.float64)
            arrays[f"{prefix}__arm_raw_header_s"] = np.array(
                [np.nan if e["header_time_s"] is None else float(e["header_time_s"]) for e in arm_evs],
                dtype=np.float64,
            )

        if mocap_evs:
            arrays[f"{prefix}__mocap_raw_wall_s"] = np.array([float(e["wall_time_s"]) for e in mocap_evs], dtype=np.float64)
            arrays[f"{prefix}__mocap_raw_epoch_s"] = np.array([float(e["epoch_time_s"]) for e in mocap_evs], dtype=np.float64)
            arrays[f"{prefix}__mocap_raw_header_s"] = np.array(
                [np.nan if e["header_time_s"] is None else float(e["header_time_s"]) for e in mocap_evs],
                dtype=np.float64,
            )
            arrays[f"{prefix}__mocap_pos_xyz_m"] = np.array([e["pos_xyz_m"] for e in mocap_evs], dtype=np.float64)
            arrays[f"{prefix}__mocap_yaw_rad"] = np.array(
                [np.nan if e.get("yaw_rad") is None else float(e["yaw_rad"]) for e in mocap_evs],
                dtype=np.float64,
            )
            arrays[f"{prefix}__mocap_rpy_rad"] = np.array(
                [
                    [np.nan, np.nan, np.nan] if e.get("rpy_rad") is None else [float(v) for v in e["rpy_rad"]]
                    for e in mocap_evs
                ],
                dtype=np.float64,
            )
            arrays[f"{prefix}__mocap_quat_xyzw"] = np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan]
                    if e.get("quat_xyzw") is None
                    else [float(v) for v in e["quat_xyzw"]]
                    for e in mocap_evs
                ],
                dtype=np.float64,
            )

    np.savez_compressed(str(out_npz_path), **arrays)


class Ros2BaseIdBackend:
    """Live ROS 2 transport backend (instantiated ONLY inside verified ``--live`` runs)."""

    def __init__(self, contract: Mapping[str, Any], store: TelemetryStore) -> None:
        import rclpy
        from geometry_msgs.msg import Vector3, Vector3Stamped
        from mc_core_interface.msg import (
            MechUnitCmd,
            MechUnitCmdArray,
            MechUnitValue,
            MechUnitValueArray,
            RobotStatus,
        )
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        self.contract = contract
        self.store = store
        self.msg_classes = {
            "Vector3": Vector3,
            "MechUnitValueArray": MechUnitValueArray,
            "MechUnitValue": MechUnitValue,
            "MechUnitCmdArray": MechUnitCmdArray,
            "MechUnitCmd": MechUnitCmd,
        }

        if not rclpy.ok():
            rclpy.init()
        self.node: Node = rclpy.create_node("moz1_base_id_collector")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop_event = threading.Event()
        self._spin_thread = threading.Thread(
            target=self._spin_loop, name="base_id_ros_spin", daemon=True
        )

        # Control QoS matches MCControlNode: BEST_EFFORT + VOLATILE + depth 2.
        cmd_qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        fb_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Count pre-existing publishers BEFORE creating our own publisher!
        base_cmd_cfg = contract["base_command"]
        self.base_cmd_protocol = str(base_cmd_cfg["protocol"])
        self.base_cmd_topic = str(base_cmd_cfg["topic"])
        self.existing_base_cmd_publishers = len(
            self.node.get_publishers_info_by_topic(self.base_cmd_topic)
        )

        if self.base_cmd_protocol == "vector3_vel":
            self._base_pub = self.node.create_publisher(Vector3, self.base_cmd_topic, cmd_qos)
        elif self.base_cmd_protocol == "mech_unit_value_array":
            self._base_pub = self.node.create_publisher(
                MechUnitValueArray, self.base_cmd_topic, cmd_qos
            )
        else:
            raise ValueError(f"Unsupported base_command.protocol={self.base_cmd_protocol!r}")

        arm_cmd_cfg = contract["arm_concurrency_command"]
        self.arm_cmd_protocol = str(arm_cmd_cfg["protocol"])
        self.arm_cmd_topic = str(arm_cmd_cfg["topic"])
        self._arm_pub = None
        if self.arm_cmd_protocol == "mech_unit_value_array":
            self._arm_pub = self.node.create_publisher(
                MechUnitValueArray, self.arm_cmd_topic, cmd_qos
            )
        elif self.arm_cmd_protocol == "mech_unit_cmd_array":
            self._arm_pub = self.node.create_publisher(
                MechUnitCmdArray, self.arm_cmd_topic, cmd_qos
            )

        base_fb = contract["base_feedback"]
        arm_fb = contract["arm_feedback"]
        self._base_mu_name = str(base_fb.get("mech_unit_name", "base"))
        self._arm_mu_name = str(arm_fb.get("mech_unit_name", "right_arm"))
        self._pose_source = str(base_fb.get("base_pose_source", "integrate_base_twist"))
        self._wheel_joints = list(base_fb.get("wheel_joints", []))
        self._arm_joints = list(arm_fb.get("joint_names", []))

        # Primary 200 Hz state subscription (/mx_robot_state).
        if str(base_fb["primary_topic"]) == "/mx_robot_state":
            self.node.create_subscription(
                MechUnitValueArray, "/mx_robot_state", self._on_mx_robot_state, fb_qos
            )
        # Secondary / fallback /joint_states subscription.
        self.node.create_subscription(
            JointState, str(base_fb["secondary_joint_topic"]), self._on_joint_states, fb_qos
        )
        # Robot status subscription.
        if base_fb.get("robot_status_topic"):
            self.node.create_subscription(
                RobotStatus, str(base_fb["robot_status_topic"]), self._on_robot_status, fb_qos
            )
        # Optional IMU subscription.
        imu_cfg = contract.get("imu_feedback") or {}
        if imu_cfg.get("topic"):
            self.node.create_subscription(
                Vector3Stamped, str(imu_cfg["topic"]), self._on_imu_states, fb_qos
            )

        # Optional Mocap UDP listener thread.
        self._mocap_sock: Optional[socket.socket] = None
        self._mocap_thread: Optional[threading.Thread] = None
        mocap_cfg = contract.get("mocap") or {}
        if bool(mocap_cfg.get("enabled", False)):
            self._start_mocap_listener(mocap_cfg)

        self._spin_thread.start()

    def _spin_loop(self) -> None:
        while not self._stop_event.is_set() and self._rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.005)
            except Exception:
                break

    @staticmethod
    def _stamp_to_sec(stamp: Any) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_mx_robot_state(self, msg: Any) -> None:
        wall_t = time.monotonic()
        epoch_t = time.time()
        header_t = self._stamp_to_sec(msg.header.stamp)
        for val in msg.values:
            if val.name == self._base_mu_name:
                tw = val.cartesian.twist
                p = val.cartesian.pose.position
                q = val.cartesian.pose.orientation
                rep_pose = (float(p.x), float(p.y), _quat_xyzw_to_yaw(q.x, q.y, q.z, q.w))
                self.store.on_base_state(
                    wall_time_s=wall_t,
                    epoch_time_s=epoch_t,
                    header_time_s=header_t,
                    vx_m_s=float(tw.linear.x),
                    vy_m_s=float(tw.linear.y),
                    wz_rad_s=float(tw.angular.z),
                    wheel_pos_rad=list(val.joint.position),
                    wheel_vel_rad_s=list(val.joint.velocity),
                    wheel_effort_nm=list(val.joint.effort),
                    reported_pose_xyyaw=rep_pose,
                    pose_source=self._pose_source,
                    source_name="base_state",
                )
            elif val.name == self._arm_mu_name:
                self.store.on_arm_state(
                    wall_time_s=wall_t,
                    epoch_time_s=epoch_t,
                    header_time_s=header_t,
                    q_rad=list(val.joint.position),
                    dq_rad_s=list(val.joint.velocity),
                    effort_nm=list(val.joint.effort),
                    source_name="arm_state",
                )

    def _on_joint_states(self, msg: Any) -> None:
        wall_t = time.monotonic()
        epoch_t = time.time()
        header_t = self._stamp_to_sec(msg.header.stamp)
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        if all(j in name_to_idx for j in self._arm_joints):
            q = [float(msg.position[name_to_idx[j]]) for j in self._arm_joints]
            dq = [
                float(msg.velocity[name_to_idx[j]]) if name_to_idx[j] < len(msg.velocity) else 0.0
                for j in self._arm_joints
            ]
            self.store.record_generic_event(
                "joint_states_arm",
                {"q_rad": q, "dq_rad_s": dq},
                wall_time_s=wall_t,
                epoch_time_s=epoch_t,
                header_time_s=header_t,
            )

    def _on_imu_states(self, msg: Any) -> None:
        wall_t = time.monotonic()
        epoch_t = time.time()
        header_t = self._stamp_to_sec(msg.header.stamp)
        self.store.on_imu_state(
            wall_time_s=wall_t,
            epoch_time_s=epoch_t,
            header_time_s=header_t,
            rpy_rad=(float(msg.vector.x), float(msg.vector.y), float(msg.vector.z)),
        )

    def _on_robot_status(self, msg: Any) -> None:
        wall_t = time.monotonic()
        epoch_t = time.time()
        self.store.on_robot_status(
            RobotStatusSnapshot(
                wall_time_s=wall_t,
                safety_status=int(msg.safety_status),
                motor_status=tuple(int(v) for v in msg.motor_status),
                control_mode=tuple(int(v) for v in msg.control_mode),
                task_mode=int(msg.task_mode),
                motion_status=int(msg.motion_status),
            ),
            epoch_time_s=epoch_t,
        )

    def _start_mocap_listener(self, mocap_cfg: Mapping[str, Any]) -> None:
        T_robot_mocap, _ = load_mocap_calibration_matrix(mocap_cfg.get("calibration_path"))

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((str(mocap_cfg.get("bind_host", "0.0.0.0")), int(mocap_cfg.get("udp_port", 65501))))
        sock.settimeout(0.1)
        self._mocap_sock = sock

        def _loop() -> None:
            r_ref_robot: Optional[np.ndarray] = None
            while not self._stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break
                wall_t = time.monotonic()
                epoch_t = time.time()
                decoded = decode_mocap_base_datagram(
                    data, mocap_cfg, T_robot_mocap=T_robot_mocap, R_ref_robot=r_ref_robot
                )
                if decoded is not None and r_ref_robot is None and decoded.get("rotmat_robot") is not None:
                    r_ref_robot = np.asarray(decoded["rotmat_robot"], dtype=np.float64).copy()
                    decoded = decode_mocap_base_datagram(
                        data, mocap_cfg, T_robot_mocap=T_robot_mocap, R_ref_robot=r_ref_robot
                    )
                if decoded is not None:
                    self.store.on_mocap_pose(
                        kind="mocap_base",
                        wall_time_s=wall_t,
                        epoch_time_s=epoch_t,
                        header_time_s=decoded["header_time_s"],
                        pos_xyz_m=decoded["pos_xyz_m"],
                        pos_center_xyz_m=decoded["pos_center_xyz_m"],
                        pos_raw_mocap_m=decoded["pos_raw_mocap_m"],
                        quat_xyzw=decoded["quat_xyzw"],
                        rpy_rad=decoded["rpy_rad"],
                        vel_xyz_m_s=decoded["vel_xyz_m_s"],
                        track_mode=decoded["track_mode"],
                    )

        self._mocap_thread = threading.Thread(target=_loop, name="base_id_mocap_udp", daemon=True)
        self._mocap_thread.start()

    def publish_base_cmd(self, vx_m_s: float, vy_m_s: float, wz_rad_s: float) -> None:
        now_ns = self.node.get_clock().now().nanoseconds
        msg = populate_ros_base_command_msg(
            self.base_cmd_protocol,
            vx_m_s,
            vy_m_s,
            wz_rad_s,
            base_cmd_cfg=self.contract["base_command"],
            msg_classes=self.msg_classes,
            stamp_sec=int(now_ns // 1_000_000_000),
            stamp_nanosec=int(now_ns % 1_000_000_000),
        )
        self._base_pub.publish(msg)

    def publish_arm_hold_cmd(self, q_target_rad: Sequence[float]) -> None:
        if self._arm_pub is None:
            return
        now_ns = self.node.get_clock().now().nanoseconds
        msg = populate_ros_arm_hold_msg(
            self.arm_cmd_protocol,
            q_target_rad,
            arm_cmd_cfg=self.contract["arm_concurrency_command"],
            msg_classes=self.msg_classes,
            stamp_sec=int(now_ns // 1_000_000_000),
            stamp_nanosec=int(now_ns % 1_000_000_000),
        )
        self._arm_pub.publish(msg)

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._mocap_sock is not None:
            try:
                self._mocap_sock.close()
            except Exception:
                pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        try:
            self._executor.remove_node(self.node)
            self.node.destroy_node()
        except Exception:
            pass


def execute_live_stage(
    *,
    stage: str,
    contract: Mapping[str, Any],
    trials: Sequence[WaveformTrial],
    run_dir: Path,
    fk: Any,
    backend_factory: Callable[[Mapping[str, Any], TelemetryStore], Any],
    cli_args: Sequence[str] = (),
    arm_micro_sine_rad: float = 0.0,
    repo_root: Optional[Path] = None,
    clock_fn: Callable[[], float] = time.monotonic,
    epoch_fn: Callable[[], float] = time.time,
    sleep_fn: Callable[[float], None] = time.sleep,
    preflight_wait_s: float = 2.0,
) -> Dict[str, Any]:
    """Execute one live stage with full real-time protection and guaranteed brake bursts."""
    root = (repo_root or find_repo_root()).resolve()
    run_t0 = clock_fn()
    started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    raw_events_path = run_dir / "raw_events.jsonl"
    aligned_npz_path = run_dir / "aligned_timeseries.npz"
    manifest_path = run_dir / "MANIFEST.json"

    arm_fb_cfg = contract["arm_feedback"]
    base_cmd_cfg = contract["base_command"]
    arm_cmd_cfg = contract["arm_concurrency_command"]
    safety_cfg = contract["safety"]
    mocap_cfg_run = contract.get("mocap") or {}
    mocap_enabled = bool(mocap_cfg_run.get("enabled", False))
    mocap_abort_on_dropout = bool(mocap_cfg_run.get("abort_on_dropout", False))

    max_micro = float(arm_cmd_cfg.get("max_micro_sine_amp_rad", 0.02))
    if abs(arm_micro_sine_rad) > max_micro:
        raise ValueError(
            f"--arm-micro-sine-rad={arm_micro_sine_rad} exceeds contract cap {max_micro} rad."
        )

    store = TelemetryStore(
        raw_events_path=raw_events_path,
        run_t0_mono_s=run_t0,
        fk=fk,
        racket_normal_axis_index=int(arm_fb_cfg.get("fk_racket_normal_axis_index", 2)),
    )
    tier_limits = tier_limits_from_contract(safety_cfg.get("tier_limits"))
    supervisor = RealTimeSafetySupervisor(
        geofence_radius_m=float(safety_cfg["geofence_radius_m"]),
        geofence_yaw_rad=float(safety_cfg["geofence_yaw_rad"]),
        feedback_stale_timeout_s=float(safety_cfg["feedback_stale_timeout_s"]),
        tier_limits=tier_limits,
        require_robot_status_normal=bool(safety_cfg.get("require_robot_status_normal", True)),
        require_base_motor_power_on=bool(safety_cfg.get("require_base_motor_power_on", True)),
        require_target_tracking_mode=bool(safety_cfg.get("require_target_tracking_mode", True)),
        require_legwaist_impedance_mode=bool(safety_cfg.get("require_legwaist_impedance_mode", False)),
        apply_command_slew_clamp=bool(safety_cfg.get("apply_command_slew_clamp", False)),
        base_mu_idx=int(contract["base_feedback"].get("base_mu_idx", 0)),
        legwaist_mu_idx=int(safety_cfg.get("legwaist_mu_idx", 1)),
    )

    backend = backend_factory(contract, store)
    abort_event = threading.Event()
    abort_reason: List[str] = []
    trial_results: List[Dict[str, Any]] = []
    brake_records: List[Dict[str, Any]] = []

    def _signal_handler(signum: int, _frame: Any) -> None:
        abort_reason.append(f"SIGNAL_{signum}")
        abort_event.set()

    prev_sigint = None
    prev_sigterm = None
    prev_sighup = None
    is_main_thread = threading.current_thread() is threading.main_thread()
    if is_main_thread:
        prev_sigint = signal.signal(signal.SIGINT, _signal_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
        if hasattr(signal, "SIGHUP"):
            prev_sighup = signal.signal(signal.SIGHUP, _signal_handler)

    def _publish_zero() -> None:
        backend.publish_base_cmd(0.0, 0.0, 0.0)
        store.record_generic_event(
            "base_cmd",
            {"cmd_vx_m_s": 0.0, "cmd_vy_m_s": 0.0, "cmd_wz_rad_s": 0.0, "phase": "zero_brake"},
            wall_time_s=clock_fn(),
            epoch_time_s=epoch_fn(),
        )

    def _get_meas_twist() -> Optional[Tuple[float, float, float]]:
        b_snap, _, _, _, _ = store.get_latest_snapshots()
        if b_snap is None or len(b_snap.values) < 3:
            return None
        return (float(b_snap.values[0]), float(b_snap.values[1]), float(b_snap.values[2]))

    def _brake_now(context_tag: str) -> Dict[str, Any]:
        rec = execute_zero_brake_burst(
            _publish_zero,
            _get_meas_twist,
            rate_hz=float(base_cmd_cfg["zero_brake_rate_hz"]),
            min_duration_s=float(base_cmd_cfg["zero_brake_min_duration_s"]),
            max_duration_s=float(safety_cfg.get("brake_max_wait_s", 3.0)),
            stop_vel_eps_m_s=float(safety_cfg["stop_vel_eps_m_s"]),
            stop_yaw_rate_eps_rad_s=float(safety_cfg["stop_yaw_rate_eps_rad_s"]),
            clock_fn=clock_fn,
            sleep_fn=sleep_fn,
        )
        rec["context"] = context_tag
        brake_records.append(rec)
        store.record_generic_event(
            "brake_burst", rec, wall_time_s=clock_fn(), epoch_time_s=epoch_fn()
        )
        return rec

    # Wait briefly for initial telemetry streams to arrive before preflight.
    need_robot_status = (
        stage != "s0_idle_noise"
        and bool(contract["base_feedback"].get("robot_status_topic"))
        and (
            bool(safety_cfg.get("require_robot_status_normal", True))
            or bool(safety_cfg.get("require_base_motor_power_on", True))
            or bool(safety_cfg.get("require_target_tracking_mode", True))
        )
    )
    wait_deadline = clock_fn() + max(0.0, preflight_wait_s)
    while clock_fn() < wait_deadline and not abort_event.is_set():
        b_s, a_s, m_s, rs_s, _ = store.get_latest_snapshots()
        if (
            b_s is not None
            and a_s is not None
            and (not mocap_enabled or m_s is not None)
            and (not need_robot_status or rs_s is not None)
        ):
            break
        sleep_fn(0.01)

    run_status = "completed"
    run_error: Optional[str] = None
    try:
        b_s, a_s, m_s, rs_s, _ = store.get_latest_snapshots()
        is_motion_stage = stage != "s0_idle_noise"
        supervisor.preflight_check(
            now_wall_s=clock_fn(),
            base_snap=b_s,
            arm_snap=a_s,
            robot_status=rs_s,
            mocap_snap=m_s,
            mocap_required=mocap_enabled,
            existing_base_cmd_publishers=int(getattr(backend, "existing_base_cmd_publishers", 0)),
            allow_existing_publishers=bool(base_cmd_cfg.get("allow_existing_publishers", False)),
            check_motion_ready=is_motion_stage,
        )

        for trial in trials:
            if abort_event.is_set():
                trial_results.append({**trial.summary_dict(), "status": "skipped_aborted"})
                continue

            store.set_trial(trial.trial_id, reset_odom=True)
            b_s, a_s, m_s, rs_s, odom_pose = store.get_latest_snapshots()
            supervisor.latch_trial_origin(
                odom_pose_xyyaw=odom_pose,
                mocap_pose_xyyaw=None if m_s is None else m_s.pose_xyyaw,
            )
            store.record_generic_event(
                "trial_start",
                trial.summary_dict(),
                wall_time_s=clock_fn(),
                epoch_time_s=epoch_fn(),
                trial_id=trial.trial_id,
            )

            # Optional 200 Hz right-arm static/micro-sine hold thread for s5_arm_concurrency.
            arm_thread_stop = threading.Event()
            arm_thread: Optional[threading.Thread] = None
            if trial.arm_concurrency_enabled and a_s is not None:
                q_latched = np.asarray(a_s.extra.get("q_rad", a_s.values[:7]), dtype=np.float64)
                arm_rate_hz = float(arm_cmd_cfg["publish_rate_hz"])
                arm_dt_s = 1.0 / arm_rate_hz
                max_delta = float(arm_cmd_cfg["max_joint_delta_rad"])

                def _arm_loop() -> None:
                    t_arm0 = clock_fn()
                    tick = 0
                    while not arm_thread_stop.is_set() and not abort_event.is_set():
                        now_a = clock_fn()
                        q_cmd = q_latched.copy()
                        if abs(arm_micro_sine_rad) > 1e-9:
                            delta = arm_micro_sine_rad * math.sin(2.0 * math.pi * 0.5 * (now_a - t_arm0))
                            delta = float(np.clip(delta, -max_delta, max_delta))
                            q_cmd[6] = float(q_latched[6] + delta)
                        backend.publish_arm_hold_cmd(q_cmd.tolist())
                        store.record_generic_event(
                            "arm_cmd",
                            {"q_cmd_rad": q_cmd.tolist()},
                            wall_time_s=now_a,
                            epoch_time_s=epoch_fn(),
                            trial_id=trial.trial_id,
                        )
                        tick += 1
                        nxt = t_arm0 + tick * arm_dt_s
                        rem = nxt - clock_fn()
                        if rem > 0.0:
                            sleep_fn(rem)

                arm_thread = threading.Thread(
                    target=_arm_loop, name=f"arm_hold_{trial.trial_id}", daemon=True
                )
                arm_thread.start()

            trial_status = "completed"
            trial_err: Optional[str] = None
            prev_cmd = (0.0, 0.0, 0.0)
            t_trial0 = clock_fn()
            try:
                for step_idx in range(len(trial.time_s)):
                    if abort_event.is_set():
                        trial_status = "aborted_signal"
                        trial_err = ",".join(abort_reason)
                        break

                    now_t = clock_fn()
                    b_s, a_s, m_s, rs_s, odom_pose = store.get_latest_snapshots()
                    supervisor.check_step_safety(
                        now_wall_s=now_t,
                        trial_id=trial.trial_id,
                        base_snap=b_s,
                        arm_snap=a_s,
                        odom_pose_xyyaw=odom_pose,
                        mocap_snap=m_s,
                        mocap_required=(mocap_enabled and mocap_abort_on_dropout),
                        robot_status=rs_s,
                    )

                    vx, vy, wz = supervisor.clamp_command(
                        float(trial.cmd_vx_m_s[step_idx]),
                        float(trial.cmd_vy_m_s[step_idx]),
                        float(trial.cmd_wz_rad_s[step_idx]),
                        tier=trial.tier,
                        dt_s=trial.dt_s,
                        prev_cmd=prev_cmd,
                        now_wall_s=now_t,
                        trial_id=trial.trial_id,
                    )
                    if stage != "s0_idle_noise":
                        backend.publish_base_cmd(vx, vy, wz)
                    store.record_generic_event(
                        "base_cmd",
                        {
                            "cmd_vx_m_s": vx,
                            "cmd_vy_m_s": vy,
                            "cmd_wz_rad_s": wz,
                            "phase": trial.phase_labels[step_idx],
                        },
                        wall_time_s=now_t,
                        epoch_time_s=epoch_fn(),
                        trial_id=trial.trial_id,
                    )
                    prev_cmd = (vx, vy, wz)

                    next_t = t_trial0 + (step_idx + 1) * trial.dt_s
                    rem = next_t - clock_fn()
                    if rem > 0.0:
                        sleep_fn(rem)
            except Exception as exc:
                trial_status = "aborted_safety"
                trial_err = str(exc)
                run_status = "aborted"
                run_error = str(exc)
                store.record_generic_event(
                    "safety_trip",
                    {"error": str(exc)},
                    wall_time_s=clock_fn(),
                    epoch_time_s=epoch_fn(),
                    trial_id=trial.trial_id,
                )
                raise
            finally:
                arm_thread_stop.set()
                if arm_thread is not None:
                    arm_thread.join(timeout=1.0)
                if stage != "s0_idle_noise":
                    _brake_now(f"trial_end:{trial.trial_id}")
                store.record_generic_event(
                    "trial_end",
                    {"status": trial_status, "error": trial_err},
                    wall_time_s=clock_fn(),
                    epoch_time_s=epoch_fn(),
                    trial_id=trial.trial_id,
                )
                store.flush()
                trial_results.append({
                    **trial.summary_dict(),
                    "status": trial_status,
                    "error": trial_err,
                })
    except Exception as exc:
        run_status = "aborted"
        run_error = str(exc)
        raise
    finally:
        # Always guarantee a final zero-velocity braking burst before tearing down backend.
        if stage != "s0_idle_noise":
            try:
                _brake_now("run_finally")
            except Exception:
                pass
        if is_main_thread:
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)
            if prev_sighup is not None and hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, prev_sighup)
        try:
            backend.shutdown()
        except Exception:
            pass
        store.close()

        compile_aligned_timeseries_npz(trials, store.trial_events, aligned_npz_path)
        manifest: Dict[str, Any] = {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "stage": stage,
            "mode": "live",
            "status": run_status,
            "error": run_error,
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **_git_metadata(root),
            "contract_sha256": compute_contract_sha256(contract),
            "contract": dict(contract),
            "cli_args": list(cli_args),
            "trials": trial_results,
            "safety_triggers": [r.to_dict() for r in supervisor.trip_records],
            "brake_bursts": brake_records,
            "source_message_counts": dict(store.source_counts),
            "clock_offset_summary_s": store.clock_offset_summary(),
            "artifacts": {
                "raw_events_jsonl": raw_events_path.name,
                "aligned_timeseries_npz": aligned_npz_path.name,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return manifest


def format_dry_run_plan(
    *,
    stage: str,
    contract: Mapping[str, Any],
    contract_path: Path,
    trials: Sequence[WaveformTrial],
    warnings: Sequence[str],
    fk_xml_path: str,
) -> str:
    """Format a clear operator-facing dry-run plan table and safety summary."""
    base_cmd = contract["base_command"]
    arm_cmd = contract["arm_concurrency_command"]
    safety = contract["safety"]
    total_dur = sum(tr.duration_s for tr in trials)
    max_r = max((tr.max_theoretical_radius_m for tr in trials), default=0.0)
    max_yaw = max((tr.max_theoretical_abs_yaw_rad for tr in trials), default=0.0)

    lines: List[str] = [
        "====================================================================",
        f"MOZ1 BASE-ID EXPERIMENT DRY-RUN PLAN  [stage={stage}]",
        "====================================================================",
        "Mode                   : DRY-RUN (NO ROS PUBLISHERS CREATED, ZERO MOTION)",
        f"Contract               : {contract_path}",
        f"Contract SHA-256       : {compute_contract_sha256(contract)[:16]}...",
        f"Base command protocol  : {base_cmd['protocol']} -> {base_cmd['topic']} [{base_cmd['msg_type']}]",
        f"Base feedback primary  : {contract['base_feedback']['primary_topic']} "
        f"(pose_source={contract['base_feedback']['base_pose_source']})",
        f"RightArmFK model XML   : {fk_xml_path} (site={contract['arm_feedback']['fk_ee_site']})",
        f"Arm concurrency topic  : {arm_cmd['protocol']} -> {arm_cmd['topic']} @ {arm_cmd['publish_rate_hz']} Hz",
        f"Geofence bounds        : radius <= {safety['geofence_radius_m']:.3f} m, "
        f"|yaw| <= {safety['geofence_yaw_rad']:.3f} rad, "
        f"stale <= {float(safety['feedback_stale_timeout_s']) * 1000.0:.0f} ms",
        f"Total trials / time    : {len(trials)} trial(s), {total_dur:.1f} s active waveform time",
        f"Max theoretical bounds : radius={max_r:.3f} m (margin "
        f"{float(safety['geofence_radius_m']) - max_r:.3f} m), |yaw|={max_yaw:.3f} rad",
        "",
        f"{'Trial ID':<28} {'Tier':<5} {'Dur(s)':>6} {'|vx|':>6} {'|vy|':>6} "
        f"{'|wz|':>6} {'MaxR(m)':>8} {'EndR(m)':>8} {'Split':<7} {'Arm200Hz'}",
        "-" * 102,
    ]
    for tr in trials:
        lines.append(
            f"{tr.trial_id:<28} {tr.tier:<5} {tr.duration_s:>6.1f} "
            f"{tr.max_abs_vx_m_s:>6.2f} {tr.max_abs_vy_m_s:>6.2f} {tr.max_abs_wz_rad_s:>6.2f} "
            f"{tr.max_theoretical_radius_m:>8.3f} {tr.end_pose_radius_m:>8.4f} "
            f"{tr.split_role:<7} {'YES' if tr.arm_concurrency_enabled else 'no'}"
        )
    if warnings:
        lines.extend(["", "Warnings:"])
        for w in warnings:
            lines.append(f"  - {w}")
    lines.extend([
        "",
        "To arm LIVE execution (only after physical safety checklist & E-Stop in hand):",
        f"  CONFIRM_MOZ1_BASE_ID_LIVE=YES python3 {Path(__file__).name} "
        f"--stage {stage} --contract {contract_path} --live",
    ])
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MOZ1 base dynamics identification collector (defaults to --dry-run)."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=VALID_STAGES,
        help="Experiment stage to plan or execute.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=default_contract_path(),
        help="Path to base_id_interface_contract.yaml.",
    )
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate contract, FK, waveforms, and geofence only (default; zero motion).",
    )
    mode_grp.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Enable live hardware publishing (requires CONFIRM_MOZ1_BASE_ID_LIVE=YES & confirmation).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/base_id"),
        help="Parent directory for unique timestamped live run folders.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory (still fails closed if it already exists).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override contract YAML field via dot.path=value (repeatable).",
    )
    parser.add_argument(
        "--trials",
        type=str,
        default="",
        help="Optional comma-separated subset of trial IDs or waveform names to run.",
    )
    parser.add_argument(
        "--arm-micro-sine-rad",
        type=float,
        default=0.0,
        help="Optional bounded wrist micro-sine amplitude (rad) for s5_arm_concurrency (default 0.0).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root override.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    live_mode = bool(args.live)
    repo_root = (args.repo_root or find_repo_root()).resolve()

    try:
        contract = load_contract(args.contract, overrides=args.overrides)
        warnings = validate_contract(contract, stage=args.stage, repo_root=repo_root)
        fk = load_existing_right_arm_fk(contract, repo_root=repo_root)

        pub_hz = float(contract["base_command"]["publish_rate_hz"])
        dt_s = 1.0 / pub_hz
        safety = contract["safety"]
        tiers = tier_limits_from_contract(safety["tier_limits"])
        trial_filter = [tok for tok in args.trials.split(",") if tok.strip()] or None
        trials = build_stage_plan(
            args.stage,
            tier_limits=tiers,
            geofence_radius_m=float(safety["geofence_radius_m"]),
            geofence_yaw_rad=float(safety["geofence_yaw_rad"]),
            dt_s=dt_s,
            trial_filter=trial_filter,
        )
    except (ContractValidationError, ValueError, RuntimeError) as exc:
        print(f"[ERROR] Startup validation failed (fail-closed):\n{exc}", file=sys.stderr)
        return 2

    plan_text = format_dry_run_plan(
        stage=args.stage,
        contract=contract,
        contract_path=args.contract,
        trials=trials,
        warnings=warnings,
        fk_xml_path=str(fk.xml_path),
    )

    if not live_mode:
        print(plan_text)
        return 0

    try:
        enforce_live_gates(live=True, env=os.environ, stage=args.stage)
        run_dir = create_unique_run_dir(
            args.out_root, args.stage, explicit_dir=args.run_dir
        )
    except (LiveGateError, OutputExistsError) as exc:
        print(f"[ERROR] Live gate refused execution:\n{exc}", file=sys.stderr)
        return 3

    print(plan_text)
    print(f"[LIVE] Armed and logging to unique directory: {run_dir}")
    try:
        manifest = execute_live_stage(
            stage=args.stage,
            contract=contract,
            trials=trials,
            run_dir=run_dir,
            fk=fk,
            backend_factory=lambda cfg, st: Ros2BaseIdBackend(cfg, st),
            cli_args=raw_argv,
            arm_micro_sine_rad=float(args.arm_micro_sine_rad),
            repo_root=repo_root,
        )
    except Exception as exc:
        print(f"[ABORTED] Live stage stopped safely (zero-brake executed): {exc}", file=sys.stderr)
        return 4

    print(
        f"[DONE] Stage {args.stage} finished with status={manifest['status']}. "
        f"Artifacts saved to: {run_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
