#!/usr/bin/env python3
"""Interactive browser probe for ping-pong policy safety behavior.

The Python side runs the same MJX policy adapter used by deployment, plus a
small simulation of the high-level safety gates from pingpong_node:
initial capture gate, ball-lost failsafe, and done-condition monitoring.
The browser side lets the operator drag or hide the ball and watch the
policy/racket response in real time.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import mimetypes
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from collections import deque
from dataclasses import dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Full
from types import SimpleNamespace
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import rclpy
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
    from mc_core_interface.msg import (
        MechUnitCmd,
        MechUnitCmdArray,
        MechUnitStateArray,
    )
    _ROS_AVAILABLE = True
    _ROS_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - ROS bridge is optional.
    rclpy = None
    MutuallyExclusiveCallbackGroup = None
    MultiThreadedExecutor = None
    Node = object
    CameraInfo = None
    CompressedImage = None
    Image = None
    JointState = None
    MechUnitCmd = None
    MechUnitCmdArray = None
    MechUnitStateArray = None
    _ROS_AVAILABLE = False
    _ROS_IMPORT_ERROR = exc


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
REPO_ROOT = SCRIPT_PATH.parents[4]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.ball_state_estimator import (
    BallStateEstimator,
    BallStateEstimatorConfig,
    CONTROL_OUTPUT_CONTINUOUS,
    CONTROL_OUTPUT_MODES,
    CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD,
    gravity_for_base_link,
)
from pingpong_controller.done_condition_monitor import (
    DoneConditionMonitor,
    _angle_above_min_rad,
    _angle_below_max_rad,
    _angle_in_limits_rad,
    _quat_angle_error_deg,
    _quat_to_euler_xyz_rad,
    _wrap_angle_rad,
)
from pingpong_controller.mjx_rl_policy import MjxRLPolicyController
from pingpong_controller.rl_policy import RIGHT_ARM_JOINTS, target_right_arm_q_rad
from pingpong_controller.safety_limiter import RightArmCommandSafetyLimiter
from pingpong_controller.sim_mirror import (
    RealBallMjxJaxSimMirror,
    RealBallSimMirror,
    SimMirrorConfig,
)

try:
    from pingpong_replay.compensation import CommandCompensator
    _COMPENSATION_AVAILABLE = True
    _COMPENSATION_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - optional deployment dependency.
    CommandCompensator = None
    _COMPENSATION_AVAILABLE = False
    _COMPENSATION_IMPORT_ERROR = exc

try:
    import cv2
    from pingpong_controller.ball_detector import PingPongBallDetector
    from pingpong_controller.calibration_loader import load_calibration
    from pingpong_controller.hybrid_ball_detector_adapter import HybridBallDetectorAdapter
    from pingpong_controller.viz_render import (
        draw_sim_done_overlay,
        render_detection_overlay,
    )
    _CAMERA_BALL_AVAILABLE = True
    _CAMERA_BALL_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - camera ball source is optional.
    cv2 = None
    PingPongBallDetector = None
    HybridBallDetectorAdapter = None
    load_calibration = None
    draw_sim_done_overlay = None
    render_detection_overlay = None
    _CAMERA_BALL_AVAILABLE = False
    _CAMERA_BALL_IMPORT_ERROR = exc


DEFAULT_MODEL_PATH = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "outputs" / "rl_sim" / "inverse_compensation"
    / "18_stage4g_strong_contact_dr_inverse_mpc.pkl"
)
DEFAULT_ROBOT_XML = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "models" / "moz1_pd.xml"
)
DEFAULT_SIM_XML = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "tools" / "rl_sim" / "moz1_pd.xml"
)
DEFAULT_VISION_MODEL = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "models" / "best.pt"
)
DEFAULT_CALIBRATION_JSON = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "outputs" / "vision_calib" / "eye_hand_result_848_undistorted.json"
)
DEFAULT_IR_PARAMS_JSON = REPO_ROOT / "ir_params.json"
DEFAULT_DEPTH_CORRECTION_MODEL = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
    / "depth_correction_model.pkl"
)
SOURCE_MODULE_DIR = (
    REPO_ROOT / "src" / "pingpong_moz1" / "pingpong_controller"
)


def finite_float(value: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"expected finite float, got {value!r}")
    return out


def parse_limits(text: str | None) -> tuple[float, float] | None:
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


def parse_angle_limits(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned or cleaned == "[]":
        return None
    cleaned = cleaned.strip("[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected 'min,max' angle limits, got {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise argparse.ArgumentTypeError(
            f"invalid angle limits {text!r}; expected finite values")
    return (lo, hi)


def parse_optional_float(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned or cleaned == "[]":
        return None
    cleaned = cleaned.strip("[]")
    value = float(cleaned)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(
            f"expected finite float, got {text!r}")
    return value


def parse_quat_xyzw(text: str | None) -> tuple[float, float, float, float] | None:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned or cleaned == "[]":
        return None
    cleaned = cleaned.strip("[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"expected 'x,y,z,w' quaternion, got {text!r}")
    values = tuple(float(p) for p in parts)
    if not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError(
            f"invalid quaternion {text!r}; expected finite values")
    if float(np.linalg.norm(np.asarray(values, dtype=np.float64))) < 1e-9:
        raise argparse.ArgumentTypeError(
            f"invalid quaternion {text!r}; norm is zero")
    return values


def parse_int_triplet(text: str) -> tuple[int, int, int]:
    cleaned = str(text).strip().strip("[]")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected 'a,b,c' integer triplet, got {text!r}")
    values = tuple(int(p) for p in parts)
    for value in values:
        if value < 0 or value > 255:
            raise argparse.ArgumentTypeError(
                f"HSV values must be in [0, 255], got {text!r}")
    return values


def vec3_from_payload(payload: dict[str, Any], key: str, default: tuple[float, float, float]) -> np.ndarray:
    raw = payload.get(key, default)
    try:
        arr = np.asarray(raw, dtype=np.float32).reshape(3)
    except Exception:
        arr = np.asarray(default, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        arr = np.asarray(default, dtype=np.float32)
    return arr


def box_payload(
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    zlim: tuple[float, float] | None,
) -> dict[str, float] | None:
    if not (xlim and ylim and zlim):
        return None
    return {
        "x0": float(xlim[0]), "x1": float(xlim[1]),
        "y0": float(ylim[0]), "y1": float(ylim[1]),
        "z0": float(zlim[0]), "z1": float(zlim[1]),
    }


def _queue_replace_latest(q, item) -> bool:
    """Put item into a maxsize=1 queue, dropping stale data."""
    try:
        q.put(item, block=False)
        return True
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put(item, block=False)
            return True
        except Full:
            return False


def _close_mp_queue(q) -> None:
    if q is None:
        return
    try:
        q.close()
    except Exception:
        pass
    try:
        q.cancel_join_thread()
    except Exception:
        pass


def _terminate_process_tree(process: mp.Process | None, name: str) -> None:
    if process is None:
        return
    if not process.is_alive():
        try:
            process.join(timeout=0.1)
        except Exception:
            pass
        return
    try:
        process.terminate()
    except Exception:
        pass
    try:
        process.join(timeout=2.0)
    except Exception:
        pass
    if process.is_alive():
        try:
            print(f"[probe] {name} did not exit after SIGTERM; killing", file=sys.stderr)
            process.kill()
        except Exception:
            pass
        try:
            process.join(timeout=2.0)
        except Exception:
            pass


def _install_parent_death_signal(signum: int) -> None:
    """Ask Linux to terminate this subprocess if its parent disappears."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        result = libc.prctl(
            pr_set_pdeathsig,
            int(signum),
            0,
            0,
            0,
        )
        if result != 0:
            errno = ctypes.get_errno()
            print(
                f"[probe] warning: PR_SET_PDEATHSIG failed errno={errno}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(
            f"[probe] warning: failed to install parent-death signal: {exc}",
            file=sys.stderr,
        )


@dataclass
class ProbeConfig:
    host: str
    port: int
    model_path: Path
    robot_xml_path: Path
    policy_obs_dim: int
    control_rate_hz: float
    ball_lost_timeout_s: float
    enable_safety_limiter: bool
    safety_parking_velocity_scale: float
    safety_parking_acceleration_scale: float
    enable_compensation: bool
    compensation_mode: str
    comp_delay_ms: float
    comp_actuator_tau_s: float
    comp_actuator_gain: float
    comp_buffer_extra_steps: int
    comp_delay_conditioning: bool
    comp_mpc_beta: float
    comp_mpc_delay_scale: float
    comp_mpc_tau_scale: float
    comp_mpc_horizon_steps: int
    comp_mpc_tracking_weight: float
    comp_mpc_nominal_weight: float
    comp_mpc_delta_weight: float
    comp_mpc_max_delta_rad: float
    enable_ball_capture_gate: bool
    enable_ball_release_gate: bool
    enable_ball_lost_failsafe: bool
    enable_done_monitoring: bool
    rl_use_valid_ball_obs_age: bool
    ball_release_min_downward_velocity_m_s: float
    ball_release_min_z_drop_m: float
    ball_release_window_s: float
    ball_release_required_consecutive_ticks: int
    ball_release_max_obs_age_s: float
    ball_release_use_estimator_state: bool
    done_z_min_m: float
    done_z_max_m: float
    done_x_range_m: float
    done_y_range_m: float
    done_ball_required_consecutive_ticks: int
    done_racket_anchor_distance_m: float
    done_racket_z_high_offset_m: float
    done_racket_z_low_offset_m: float
    ee_x_limits_m: tuple[float, float] | None
    ee_y_limits_m: tuple[float, float] | None
    ee_z_limits_m: tuple[float, float] | None
    ee_psi_limits_rad: tuple[float, float] | None
    ee_b_limits_rad: tuple[float, float] | None
    ee_b_min_rad: float | None
    ee_b_max_rad: float | None
    ee_c_limits_rad: tuple[float, float] | None
    ee_c_min_rad: float | None
    ee_c_max_rad: float | None
    ee_orientation_reference_xyzw: tuple[float, float, float, float] | None
    ee_max_orientation_error_deg: float
    done_enable_predictive_stop: bool
    done_prediction_horizon_s: float
    done_prediction_dt_s: float
    done_prediction_margin_m: float
    done_prediction_orientation_margin_rad: float
    ball_noise_std_m: float
    ball_observation_rate_hz: float
    backend_control_loop: bool
    backend_snapshot_rate_hz: float
    browser_step_rate_hz: float
    display_rate_hz: float
    enable_ball_state_estimator: bool
    ball_estimator_enable_prediction: bool
    ball_estimator_enable_aero_drag: bool
    ball_estimator_max_extrapolation_s: float
    ball_estimator_control_max_extrapolation_s: float
    ball_estimator_control_output_mode: str
    ball_estimator_frame_hold_output_rate_hz: float
    ball_estimator_release_gated_dynamics: bool
    ball_estimator_trust_loss_s: float
    ball_estimator_degrade_loss_s: float
    ball_estimator_invalid_after_s: float
    ball_estimator_prediction_horizon_s: float
    ball_estimator_q_pos_mm2_s: float
    ball_estimator_q_vel_mult: float
    ball_estimator_q_depth_pos_mult: float
    ball_estimator_q_depth_vel_mult: float
    ball_estimator_r_lateral_mm: float
    ball_estimator_r_depth_mm: float
    ros_bridge: bool
    ros_publish_commands: bool
    command_topic: str
    joint_state_topic: str
    cart_state_topic: str
    right_arm_mu_idx: int
    enable_camera_ball: bool
    camera_source_mode: str
    use_hybrid_detector: bool
    use_depth_sensor: bool
    image_width: int
    image_height: int
    realsense_fps: int
    camera_image_topic: str
    camera_raw_image_topic: str
    camera_image_transport: str
    camera_info_topic: str
    calibration_json_path: str
    yolo_model_path: Path
    target_class: str
    conf_threshold: float
    ir_params_json: str
    yolo_interval: int
    ir_threshold: int
    use_adaptive_ir: bool
    ir_exposure: int | None
    ir_gain: int | None
    ir_laser_power: int
    ir_min_radius: int
    ir_max_radius: int
    ir_circularity_min: float
    ir_brightness_percentile: int
    hybrid_roi_margin: int
    depth_correction_model_path: str
    depth_neighborhood_px: int
    infer_imgsz: int
    detector_device: str
    detector_half: bool
    use_kalman: bool
    camera_tilt_deg: float
    use_undistort: bool
    enable_ellipse_fit: bool
    enable_temporal_tracker: bool
    tracker_max_miss_bridge: int
    tracker_gate_px: float
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    color_match_threshold: float
    color_circularity_min: float
    enable_camera_detection: bool
    enable_camera_visualization: bool
    camera_visualization_rate_hz: float
    camera_web_max_width: int
    camera_web_jpeg_quality: int
    enable_camera_color_filter: bool
    recording_output_root: Path
    recording_control_rate_hz: float
    recording_image_rate_hz: float
    recording_analysis_video_fps: float
    recording_queue_size: int
    sim_cartesian_x_offset_m: float
    sim_cartesian_y_offset_m: float
    sim_base_height_m: float
    enable_sim_mirror: bool
    sim_mirror_default_enabled: bool
    sim_subprocess: bool
    sim_backend: str
    sim_xml_path: Path
    sim_enable_video_frames: bool
    sim_frame_rate_hz: float
    sim_render_width: int
    sim_render_height: int
    sim_use_mujoco_renderer: bool
    workspace_safety_mode: str
    workspace_soft_margin_m: float
    workspace_soft_pull_speed_m_s: float
    workspace_soft_damping: float
    workspace_soft_max_correction_speed_m_s: float
    workspace_soft_dls_damping: float
    workspace_soft_nullspace_damping: float
    workspace_soft_posture_gain_s: float
    workspace_soft_boundary_joint_speed_scale: float
    workspace_hold_trigger_margin_m: float

    @property
    def control_dt(self) -> float:
        return 1.0 / max(self.control_rate_hz, 1e-6)


def _rounded_vector(values: Any, digits: int = 6) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [round(float(v), digits) for v in arr]


def _float32_vector(values: Any) -> list[float]:
    """Serialize a policy input without decimal rounding.

    Python's pickle preserves these Python floats exactly, and each value is
    first quantized to float32.  The sim subprocess therefore receives the
    same float32 vector that was passed to the real policy.
    """
    return np.asarray(values, dtype=np.float32).reshape(-1).tolist()


def _write_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _strict_json_dumps(payload: Any, **kwargs: Any) -> str:
    """Serialize browser-facing data as standards-compliant JSON."""
    def finite_or_none(value: Any) -> Any:
        if isinstance(value, (float, np.floating)):
            number = float(value)
            return number if math.isfinite(number) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, dict):
            return {key: finite_or_none(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_or_none(item) for item in value]
        return value

    return json.dumps(finite_or_none(payload), allow_nan=False, **kwargs)


def _recording_writer_main(
    run_dir_text: str,
    event_q,
    status_q,
    metadata: dict[str, Any],
) -> None:
    run_dir = Path(run_dir_text)
    frames_dir = run_dir / "frames"
    sim_frames_dir = run_dir / "sim_frames"
    counts = {
        "control": 0,
        "simControl": 0,
        "snapshots": 0,
        "views": 0,
        "frames": 0,
        "simFrames": 0,
        "events": 0,
    }
    started_wall = time.time()
    parent_dropped = 0

    def publish_status(active: bool, error: str | None = None) -> None:
        _queue_replace_latest(status_q, {
            "active": bool(active),
            "runDir": str(run_dir),
            "counts": dict(counts),
            "elapsedS": round(float(time.time() - started_wall), 3),
            "error": error,
        })

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        sim_frames_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(metadata)
        metadata["runDir"] = str(run_dir)
        metadata["startedWallTimeS"] = started_wall
        metadata["startedLocalTime"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(started_wall))
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with (
            (run_dir / "control_samples.jsonl").open("a", buffering=1, encoding="utf-8") as control_f,
            (run_dir / "sim_control_samples.jsonl").open("a", buffering=1, encoding="utf-8") as sim_control_f,
            (run_dir / "snapshots.jsonl").open("a", buffering=1, encoding="utf-8") as snapshots_f,
            (run_dir / "browser_view.jsonl").open("a", buffering=1, encoding="utf-8") as views_f,
            (run_dir / "frames.jsonl").open("a", buffering=1, encoding="utf-8") as frames_f,
            (run_dir / "sim_frames.jsonl").open("a", buffering=1, encoding="utf-8") as sim_frames_f,
            (run_dir / "events.jsonl").open("a", buffering=1, encoding="utf-8") as events_f,
        ):
            publish_status(True)
            next_status_wall = time.time() + 0.5
            running = True
            while running:
                try:
                    item = event_q.get(timeout=0.25)
                except Empty:
                    item = None
                if isinstance(item, dict):
                    kind = str(item.get("kind", ""))
                    if kind == "stop":
                        parent_dropped = max(
                            0, int(item.get("dropped", 0) or 0))
                        running = False
                    elif kind == "control":
                        _write_jsonl(control_f, item.get("sample", {}))
                        counts["control"] += 1
                    elif kind == "sim_control_batch":
                        samples = item.get("samples", [])
                        if isinstance(samples, list):
                            for sample in samples:
                                if isinstance(sample, dict):
                                    _write_jsonl(sim_control_f, sample)
                                    counts["simControl"] += 1
                    elif kind == "snapshot":
                        _write_jsonl(snapshots_f, item.get("snapshot", {}))
                        counts["snapshots"] += 1
                    elif kind == "view":
                        _write_jsonl(views_f, item.get("view", {}))
                        counts["views"] += 1
                    elif kind == "frame":
                        frame = item.get("frame")
                        info = item.get("info", {})
                        if isinstance(frame, (bytes, bytearray)):
                            counts["frames"] += 1
                            seq = int((info or {}).get("seq", 0) or 0)
                            filename = (
                                f"frame_{counts['frames']:06d}_seq_{seq:06d}.jpg")
                            rel_path = f"frames/{filename}"
                            (frames_dir / filename).write_bytes(bytes(frame))
                            row = {
                                "wallTimeS": round(float(time.time()), 6),
                                "path": rel_path,
                                "bytes": len(frame),
                                "info": dict(info) if isinstance(info, dict) else {},
                            }
                            _write_jsonl(frames_f, row)
                    elif kind == "sim_frame":
                        frame = item.get("frame")
                        info = item.get("info", {})
                        if isinstance(frame, (bytes, bytearray)):
                            counts["simFrames"] += 1
                            seq = int((info or {}).get("seq", 0) or 0)
                            filename = (
                                f"sim_frame_{counts['simFrames']:06d}_seq_{seq:06d}.jpg")
                            rel_path = f"sim_frames/{filename}"
                            (sim_frames_dir / filename).write_bytes(bytes(frame))
                            row = {
                                "wallTimeS": round(float(time.time()), 6),
                                "path": rel_path,
                                "bytes": len(frame),
                                "info": dict(info) if isinstance(info, dict) else {},
                            }
                            _write_jsonl(sim_frames_f, row)
                    elif kind == "event":
                        _write_jsonl(events_f, item.get("event", {}))
                        counts["events"] += 1
                now = time.time()
                if now >= next_status_wall:
                    publish_status(True)
                    next_status_wall = now + 0.5
            summary = {
                "active": False,
                "runDir": str(run_dir),
                "counts": dict(counts),
                "dropped": int(parent_dropped),
                "elapsedS": round(float(time.time() - started_wall), 3),
                "stoppedWallTimeS": time.time(),
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            publish_status(False)
    except BaseException as exc:  # noqa: BLE001 - propagate to UI.
        publish_status(False, error=str(exc))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _numeric_array(
    rows: list[dict[str, Any]],
    key: str,
    width: int | None = None,
) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        raw = row.get(key)
        if raw is None and "." in key:
            raw_any: Any = row
            for part in key.split("."):
                if not isinstance(raw_any, dict):
                    raw_any = None
                    break
                raw_any = raw_any.get(part)
            raw = raw_any
        if isinstance(raw, list):
            vals = []
            for item in raw:
                try:
                    vals.append(float(item))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
            if width is not None:
                vals = (vals + [float("nan")] * width)[:width]
            values.append(vals)
    if not values:
        w = 0 if width is None else width
        return np.zeros((0, w), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _row_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _analysis_write_status(
    status_q,
    *,
    active: bool,
    phase: str,
    analysis_dir: Path,
    outputs: dict[str, str] | None = None,
    error: str | None = None,
) -> None:
    _queue_replace_latest(status_q, {
        "active": bool(active),
        "phase": str(phase),
        "analysisDir": str(analysis_dir),
        "outputs": dict(outputs or {}),
        "error": error,
        "wallTimeS": round(float(time.time()), 6),
    })


def _build_frame_video(
    run_dir: Path,
    analysis_dir: Path,
    fps: float,
    *,
    manifest_name: str,
    output_name: str,
    label: str,
) -> tuple[Path | None, dict[str, Any]]:
    frames = _read_jsonl(run_dir / manifest_name)
    frame_paths = [
        run_dir / str(row.get("path", ""))
        for row in frames
        if row.get("path")
    ]
    frame_paths = [p for p in frame_paths if p.is_file()]
    info: dict[str, Any] = {
        "frames": len(frame_paths),
        "fps": float(fps),
        "label": label,
    }
    if not frame_paths:
        info["warning"] = "no frames recorded"
        return None, info
    if cv2 is None:
        info["warning"] = f"cv2 unavailable; {label} mp4 not generated"
        return None, info
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        info["warning"] = "first frame unreadable"
        return None, info
    height, width = first.shape[:2]
    out = analysis_dir / output_name
    writer = cv2.VideoWriter(
        str(out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(fps)),
        (width, height),
    )
    if not writer.isOpened():
        info["warning"] = "cv2 VideoWriter failed to open"
        return None, info
    written = 0
    for path in frame_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        writer.write(image)
        written += 1
    writer.release()
    info.update({"written": written, "width": width, "height": height})
    return out if written > 0 else None, info


def _build_camera_video(
    run_dir: Path,
    analysis_dir: Path,
    fps: float,
) -> tuple[Path | None, dict[str, Any]]:
    return _build_frame_video(
        run_dir,
        analysis_dir,
        fps,
        manifest_name="frames.jsonl",
        output_name="camera_video.mp4",
        label="camera",
    )


def _build_sim_video(
    run_dir: Path,
    analysis_dir: Path,
    fps: float,
) -> tuple[Path | None, dict[str, Any]]:
    return _build_frame_video(
        run_dir,
        analysis_dir,
        fps,
        manifest_name="sim_frames.jsonl",
        output_name="sim_video.mp4",
        label="sim",
    )


def _sample_source_seq(row: dict[str, Any]) -> int | None:
    raw = row.get("sourceSeq", row.get("tick"))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _sim_source_seq(row: dict[str, Any]) -> int | None:
    sync = row.get("sync")
    raw = sync.get("sourceSeq") if isinstance(sync, dict) else None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _sample_policy_executed(row: dict[str, Any]) -> bool:
    if "policyExecuted" in row:
        return bool(row.get("policyExecuted"))
    return str(row.get("mode", "")) == "policy"


def _policy_inputs_match(
    real_row: dict[str, Any],
    sim_row: dict[str, Any],
) -> bool:
    policy_input = real_row.get("policyInput")
    sync = sim_row.get("sync")
    if not isinstance(policy_input, dict) or not isinstance(sync, dict):
        return False
    try:
        real_ball = np.asarray(
            policy_input.get("ball"), dtype=np.float32).reshape(3)
        sim_ball = np.asarray(
            sync.get("inputBall"), dtype=np.float32).reshape(3)
        real_vel = np.asarray(
            policy_input.get("velocity"), dtype=np.float32).reshape(3)
        sim_vel = np.asarray(
            sync.get("inputVelocity"), dtype=np.float32).reshape(3)
        age_match = abs(
            float(policy_input.get("age", 0.0))
            - float(sync.get("inputAgeS", 0.0))) <= 1e-9
        envelope_match = bool(
            np.array_equal(real_ball, sim_ball)
            and np.array_equal(real_vel, sim_vel)
            and bool(policy_input.get("valid", False))
            == bool(sync.get("inputValid", False))
            and int(policy_input.get("ballObservationSeq", 0) or 0)
            == int(sync.get("ballObservationSeq", 0) or 0)
            and age_match
        )
        real_policy_ball = real_row.get("policyBallObservation")
        sim_policy_ball = sim_row.get("policyBallObservation")
        if real_policy_ball is None and sim_policy_ball is None:
            return envelope_match
        try:
            policy_ball_match = np.array_equal(
                np.asarray(real_policy_ball, dtype=np.float32).reshape(6),
                np.asarray(sim_policy_ball, dtype=np.float32).reshape(6),
            )
        except (TypeError, ValueError):
            policy_ball_match = False
        return bool(envelope_match and policy_ball_match)
    except (TypeError, ValueError):
        return False


def _align_sim_real_samples(
    real_samples: list[dict[str, Any]],
    sim_samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join asynchronous streams only by the canonical real control tick."""
    real_by_seq: dict[int, dict[str, Any]] = {}
    sim_by_seq: dict[int, dict[str, Any]] = {}
    duplicate_real: list[int] = []
    duplicate_sim: list[int] = []
    for row in real_samples:
        seq = _sample_source_seq(row)
        if seq is None:
            continue
        if seq in real_by_seq:
            duplicate_real.append(seq)
        else:
            real_by_seq[seq] = row
    for row in sim_samples:
        seq = _sim_source_seq(row)
        if seq is None:
            continue
        if seq in sim_by_seq:
            duplicate_sim.append(seq)
        else:
            sim_by_seq[seq] = row

    expected = {
        seq for seq, row in real_by_seq.items()
        if _sample_policy_executed(row)
    }
    matched = sorted(expected & set(sim_by_seq))
    missing = sorted(expected - set(sim_by_seq))
    orphan = sorted(set(sim_by_seq) - set(real_by_seq))
    real_source_gaps: list[int] = []
    real_source_gap_count = 0
    sorted_real_seqs = sorted(real_by_seq)
    for previous, current in zip(sorted_real_seqs, sorted_real_seqs[1:]):
        if current <= previous + 1:
            continue
        gap_size = current - previous - 1
        real_source_gap_count += gap_size
        remaining = max(0, 100 - len(real_source_gaps))
        if remaining:
            real_source_gaps.extend(
                range(previous + 1, min(current, previous + 1 + remaining)))
    if sorted_real_seqs:
        first_real_seq = sorted_real_seqs[0]
        last_real_seq = sorted_real_seqs[-1]
        orphan_in_window = [
            seq for seq in orphan
            if first_real_seq <= seq <= last_real_seq
        ]
    else:
        orphan_in_window = []
    input_mismatch = [
        seq for seq in matched
        if not _policy_inputs_match(real_by_seq[seq], sim_by_seq[seq])
    ]
    invalid_stream = []
    for seq in matched:
        sync = sim_by_seq[seq].get("sync")
        if not isinstance(sync, dict):
            invalid_stream.append(seq)
            continue
        if (not bool(sync.get("streamContinuous", False))
                or int(sync.get("resultDrops", 0) or 0) > 0
                or not bool(sync.get("runtimeConfigMatch", True))):
            invalid_stream.append(seq)

    merged: list[dict[str, Any]] = []
    for row in real_samples:
        copied = dict(row)
        seq = _sample_source_seq(row)
        sim_row = sim_by_seq.get(seq) if seq is not None else None
        if sim_row is not None and _sample_policy_executed(row):
            copied["sim"] = dict(sim_row)
            copied["simAligned"] = True
            copied["simInputMatch"] = _policy_inputs_match(row, sim_row)
        merged.append(copied)

    alignment_mode = "source_seq" if sim_samples else "unavailable"
    valid = bool(
        expected
        and not missing
        and not duplicate_real
        and not duplicate_sim
        and not input_mismatch
        and not invalid_stream
        and real_source_gap_count == 0
        and not orphan_in_window
    )
    report = {
        "alignmentMode": alignment_mode,
        "alignmentValid": valid,
        "realCount": len(real_samples),
        "simCount": len(sim_samples),
        "policyFrameCount": len(expected),
        "matchedCount": len(matched),
        "missingSimCount": len(missing),
        "missingSimSeq": missing[:100],
        "orphanSimCount": len(orphan),
        "orphanSimSeq": orphan[:100],
        "orphanSimInWindowCount": len(orphan_in_window),
        "orphanSimInWindowSeq": orphan_in_window[:100],
        "realSourceGapCount": real_source_gap_count,
        "realSourceGapSeq": real_source_gaps,
        "duplicateRealCount": len(duplicate_real),
        "duplicateRealSeq": sorted(set(duplicate_real))[:100],
        "duplicateSimCount": len(duplicate_sim),
        "duplicateSimSeq": sorted(set(duplicate_sim))[:100],
        "inputMismatchCount": len(input_mismatch),
        "inputMismatchSeq": input_mismatch[:100],
        "invalidStreamCount": len(invalid_stream),
        "invalidStreamSeq": invalid_stream[:100],
    }
    if not sim_samples:
        report["warning"] = (
            "no sourceSeq-tagged sim_control_samples.jsonl; legacy latest-sim "
            "telemetry is intentionally not used for numeric comparison")
    return merged, report


def _realtime_alignment_report(
    real_samples: list[dict[str, Any]],
    sim_samples: list[dict[str, Any]],
    control_rate_hz: float,
) -> dict[str, Any]:
    """Measure online completion cadence separately from eventual seq alignment."""
    target_hz = max(1e-6, float(control_rate_hz))
    dt_s = 1.0 / target_hz
    expected = {
        seq for row in real_samples
        if _sample_policy_executed(row)
        for seq in [_sample_source_seq(row)]
        if seq is not None
    }
    timed: list[tuple[int, int, float, float]] = []
    for row in sim_samples:
        source_seq = _sim_source_seq(row)
        sync = row.get("sync")
        if source_seq not in expected or not isinstance(sync, dict):
            continue
        try:
            episode = int(sync.get("episode", 0) or 0)
            stream_seq = int(sync.get("streamSeq", 0) or 0)
            submitted = float(sync["submittedWallTimeS"])
            completed = float(sync["completedWallTimeS"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(submitted) and math.isfinite(completed)):
            continue
        timed.append((episode, stream_seq, submitted, completed))
    timed.sort(key=lambda item: (item[0], item[1]))
    latencies = np.asarray(
        [completed - submitted for _, _, submitted, completed in timed],
        dtype=np.float64,
    )
    submitted_elapsed = 0.0
    completed_elapsed = 0.0
    interval_count = 0
    for previous, current in zip(timed, timed[1:]):
        if current[0] != previous[0] or current[1] != previous[1] + 1:
            continue
        submitted_dt = current[2] - previous[2]
        completed_dt = current[3] - previous[3]
        if submitted_dt < 0.0 or completed_dt < 0.0:
            continue
        submitted_elapsed += submitted_dt
        completed_elapsed += completed_dt
        interval_count += 1
    submission_hz = (
        interval_count / submitted_elapsed if submitted_elapsed > 0.0 else 0.0)
    completion_hz = (
        interval_count / completed_elapsed if completed_elapsed > 0.0 else 0.0)
    rate_ratio = (
        completion_hz / submission_hz if submission_hz > 0.0 else 0.0)
    enough_samples = len(timed) >= 20 and interval_count >= 10
    nonnegative_latency = bool(
        latencies.size and np.all(np.isfinite(latencies))
        and np.all(latencies >= 0.0))
    latency_p50_s = (
        float(np.percentile(latencies, 50)) if latencies.size else None)
    latency_p95_s = (
        float(np.percentile(latencies, 95)) if latencies.size else None)
    latency_max_s = float(np.max(latencies)) if latencies.size else None
    latency_limit_s = max(0.010, 2.0 * dt_s)
    realtime_aligned = bool(
        enough_samples
        and nonnegative_latency
        and latency_p95_s is not None
        and latency_p95_s <= latency_limit_s
        and abs(submission_hz - target_hz) / target_hz <= 0.05
        and abs(rate_ratio - 1.0) <= 0.05
    )
    return {
        "realtimeAligned": realtime_aligned,
        "sampleCount": len(timed),
        "intervalCount": interval_count,
        "targetHz": target_hz,
        "submissionHz": submission_hz,
        "completionHz": completion_hz,
        "completionToSubmissionRateRatio": rate_ratio,
        "latencyP50S": latency_p50_s,
        "latencyP95S": latency_p95_s,
        "latencyMaxS": latency_max_s,
        "latencyP95Ticks": (
            None if latency_p95_s is None else latency_p95_s / dt_s),
        "latencyMaxTicks": (
            None if latency_max_s is None else latency_max_s / dt_s),
        "latencyLimitS": latency_limit_s,
        "enoughSamples": enough_samples,
    }


def _write_policy_trace_csv(samples: list[dict[str, Any]], analysis_dir: Path) -> Path:
    obs_width = max((len(r.get("policyObs", [])) for r in samples), default=0)
    action_width = max((len(r.get("policyAction", [])) for r in samples), default=0)
    sim_obs_width = max((
        len(r.get("sim", {}).get("policyObs", []))
        for r in samples if isinstance(r.get("sim"), dict)
    ), default=0)
    sim_action_width = max((
        len(r.get("sim", {}).get("policyAction", []))
        for r in samples if isinstance(r.get("sim"), dict)
    ), default=0)
    out = analysis_dir / "policy_trace.csv"
    fieldnames = (
        [
            "time", "tick", "sourceSeq", "mode", "policyExecuted",
            "valid", "age",
            "ballSource", "simSourceSeq", "simInputMatch",
        ]
        + [f"obs_{i:03d}" for i in range(obs_width)]
        + [f"action_{i}" for i in range(action_width)]
        + [f"sim_obs_{i:03d}" for i in range(sim_obs_width)]
        + [f"sim_action_{i}" for i in range(sim_action_width)]
    )
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in samples:
            out_row: dict[str, Any] = {
                "time": row.get("time"),
                "tick": row.get("tick"),
                "sourceSeq": row.get("sourceSeq", row.get("tick")),
                "mode": row.get("mode"),
                "policyExecuted": _sample_policy_executed(row),
                "valid": row.get("valid"),
                "age": row.get("age"),
                "ballSource": row.get("ballSource"),
                "simSourceSeq": _sim_source_seq(
                    row.get("sim", {})
                    if isinstance(row.get("sim"), dict) else {}),
                "simInputMatch": row.get("simInputMatch"),
            }
            obs = row.get("policyObs", [])
            action = row.get("policyAction", [])
            sim = row.get("sim") if isinstance(row.get("sim"), dict) else {}
            sim_obs = sim.get("policyObs", [])
            sim_action = sim.get("policyAction", [])
            if isinstance(obs, list):
                for i, value in enumerate(obs[:obs_width]):
                    out_row[f"obs_{i:03d}"] = value
            if isinstance(action, list):
                for i, value in enumerate(action[:action_width]):
                    out_row[f"action_{i}"] = value
            if isinstance(sim_obs, list):
                for i, value in enumerate(sim_obs[:sim_obs_width]):
                    out_row[f"sim_obs_{i:03d}"] = value
            if isinstance(sim_action, list):
                for i, value in enumerate(sim_action[:sim_action_width]):
                    out_row[f"sim_action_{i}"] = value
            writer.writerow(out_row)
    return out


def _plot_policy_curves(samples: list[dict[str, Any]], analysis_dir: Path) -> list[Path]:
    if not samples:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray([_row_float(r, "time", i * 0.005)
                    for i, r in enumerate(samples)], dtype=np.float64)
    obs = _numeric_array(samples, "policyObs")
    action = _numeric_array(samples, "policyAction")
    outputs: list[Path] = []
    if obs.size:
        groups = [
            ("joint q", 0, 7),
            ("joint dq", 7, 14),
            ("base pose/vel", 14, 20),
            ("ball pos", 20, 23),
            ("ball vel", 23, 26),
            ("racket pos/vel", 26, 32),
            ("relative", 32, 35),
            ("prev action", 35, 42),
            ("cmd error", 42, 49),
            ("age/delay", 49, min(obs.shape[1], 67)),
        ]
        fig, axes = plt.subplots(len(groups), 1, figsize=(15, 20), sharex=True)
        for ax, (title, start, end) in zip(axes, groups):
            end = min(end, obs.shape[1])
            if start >= end:
                ax.set_visible(False)
                continue
            for idx in range(start, end):
                ax.plot(t, obs[:, idx], linewidth=0.9, label=f"obs[{idx}]")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", ncol=4, fontsize=7)
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        out = analysis_dir / "policy_observation_curves.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outputs.append(out)
    if action.size:
        fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        for idx in range(action.shape[1]):
            ax.plot(t, action[:, idx], linewidth=1.0, label=f"action[{idx}]")
        ax.set_title("Policy Action Output")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=4)
        fig.tight_layout()
        out = analysis_dir / "policy_action_curves.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        outputs.append(out)
    return outputs


def _per_dim_comparison(
    real: np.ndarray,
    sim: np.ndarray,
    *,
    group: str,
    names: list[str],
) -> list[dict[str, Any]]:
    n = min(real.shape[0], sim.shape[0])
    width = min(real.shape[1] if real.ndim == 2 else 0,
                sim.shape[1] if sim.ndim == 2 else 0)
    rows: list[dict[str, Any]] = []
    if n <= 0 or width <= 0:
        return rows
    real = real[:n, :width]
    sim = sim[:n, :width]
    for idx in range(width):
        r = real[:, idx]
        s = sim[:, idx]
        mask = np.isfinite(r) & np.isfinite(s)
        if not np.any(mask):
            continue
        rv = r[mask]
        sv = s[mask]
        diff = rv - sv
        # JSON has no representation for NaN.  A constant real or simulated
        # dimension has an undefined correlation, so expose it as JSON null
        # instead of letting Python emit a non-standard bare NaN token that
        # browsers reject while loading analysis_data.json.
        corr: float | None = None
        if rv.size >= 3 and np.nanstd(rv) > 1e-12 and np.nanstd(sv) > 1e-12:
            corr_value = float(np.corrcoef(rv, sv)[0, 1])
            corr = corr_value if math.isfinite(corr_value) else None
        rows.append({
            "group": group,
            "dim": idx,
            "name": names[idx] if idx < len(names) else f"{group}[{idx}]",
            "count": int(rv.size),
            "real_mean": float(np.nanmean(rv)),
            "sim_mean": float(np.nanmean(sv)),
            "diff_mean": float(np.nanmean(diff)),
            "diff_rms": float(np.sqrt(np.nanmean(diff * diff))),
            "diff_abs_max": float(np.nanmax(np.abs(diff))),
            "corr": corr,
        })
    return rows


def _analyze_sim_real_gap(
    samples: list[dict[str, Any]],
    analysis_dir: Path,
    alignment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sim_rows = [
        row for row in samples
        if (row.get("simAligned") is True
            and row.get("simInputMatch") is True
            and str(row.get("mode", "")) == "policy"
            and isinstance(row.get("sim"), dict))
    ]
    report: dict[str, Any] = {
        "sampleCount": len(samples),
        "simSampleCount": len(sim_rows),
        "groups": {},
        "alignment": dict(alignment or {}),
        "alignmentValid": bool(
            (alignment or {}).get("alignmentValid", False)),
        "phaseSemantics": {
            "policyObs": (
                "tick-start state used by both policies; q/dq are obs[0:14] "
                "and racket xyz is obs[26:29]"
            ),
            "policyAction": "same-tick policy output",
            "commandPosition": "same-tick nominal policy command",
            "simSnapshotState": (
                "jointPosition/jointVelocity/racket are tick-end and are not "
                "compared against same-tick real feedback"
            ),
        },
    }
    if not sim_rows:
        report["warning"] = (
            (alignment or {}).get("warning")
            or "no exactly sourceSeq-aligned sim mirror samples")
        (analysis_dir / "sim_real_comparison.json").write_text(
            _strict_json_dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (analysis_dir / "sim_real_comparison.csv").write_text(
            "group,dim,name,count,real_mean,sim_mean,diff_mean,diff_rms,diff_abs_max,corr\n",
            encoding="utf-8",
        )
        (analysis_dir / "sim_real_comparison.txt").write_text(
            f"Sim2Real Comparison\n\n{report['warning']}\n",
            encoding="utf-8",
        )
        return report
    if not report["alignmentValid"]:
        report["warning"] = (
            "alignment is incomplete/invalid; metrics below cover only exact "
            "sourceSeq and input-matched frames and must not be treated as a "
            "complete lockstep run")

    comparisons: list[dict[str, Any]] = []
    groups = [
        ("policy_obs", "policyObs", "sim.policyObs",
         _obs_names(max((len(r.get("policyObs", [])) for r in sim_rows), default=0))),
        ("policy_action", "policyAction", "sim.policyAction",
         [f"action[{i}]" for i in range(7)]),
        ("command_position", "policyCommand.position", "sim.policyCommand",
         RIGHT_ARM_JOINTS),
    ]
    for group, real_key, sim_key, names in groups:
        real = _numeric_array(sim_rows, real_key)
        sim = _numeric_array(sim_rows, sim_key)
        rows = _per_dim_comparison(real, sim, group=group, names=names)
        comparisons.extend(rows)
        if rows:
            report["groups"][group] = {
                "dims": len(rows),
                "meanRms": float(np.nanmean([r["diff_rms"] for r in rows])),
                "maxAbs": float(np.nanmax([r["diff_abs_max"] for r in rows])),
            }

    csv_path = analysis_dir / "sim_real_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "group", "dim", "name", "count",
            "real_mean", "sim_mean", "diff_mean",
            "diff_rms", "diff_abs_max", "corr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparisons:
            writer.writerow(row)
    report["comparisonCsv"] = csv_path.name
    report["rows"] = comparisons

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        labels = [f"{r['group']}:{r['dim']}" for r in comparisons]
        vals = [r["diff_rms"] for r in comparisons]
        if vals:
            ax.bar(np.arange(len(vals)), vals, color="#4e79a7")
            ax.set_xticks(np.arange(len(vals))[::max(1, len(vals) // 40)])
            ax.set_xticklabels(labels[::max(1, len(vals) // 40)], rotation=70, ha="right", fontsize=7)
            ax.set_ylabel("RMS(real - sim)")
            ax.set_title("Sim2Real Per-Dimension Difference")
            ax.grid(True, axis="y", alpha=0.25)
            fig.tight_layout()
            plot_path = analysis_dir / "sim_real_comparison_rms.png"
            fig.savefig(plot_path, dpi=130)
            report["comparisonPlot"] = plot_path.name
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001 - report but keep CSV.
        report["plotWarning"] = str(exc)

    (analysis_dir / "sim_real_comparison.json").write_text(
        _strict_json_dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "Sim2Real Comparison",
        "",
        "State phase: policyObs is tick-start; sim snapshot joint/racket "
        "fields are tick-end and are excluded from same-tick RMS.",
        "",
    ]
    for group, summary in report.get("groups", {}).items():
        lines.append(
            f"{group}: dims={summary['dims']} "
            f"mean_rms={summary['meanRms']:.6g} "
            f"max_abs={summary['maxAbs']:.6g}")
    if report.get("warning"):
        lines.append(str(report["warning"]))
    (analysis_dir / "sim_real_comparison.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return report


def _robust_axis_stats_mm(values_m: np.ndarray) -> dict[str, Any]:
    if values_m.size == 0:
        return {}
    vals_mm = values_m * 1000.0
    return {
        "meanMm": [round(float(v), 4) for v in np.nanmean(vals_mm, axis=0)],
        "stdMm": [round(float(v), 4) for v in np.nanstd(vals_mm, axis=0)],
        "rmsMm": [
            round(float(v), 4)
            for v in np.sqrt(np.nanmean(np.square(vals_mm), axis=0))
        ],
        "madMm": [
            round(float(v), 4)
            for v in np.nanmedian(
                np.abs(vals_mm - np.nanmedian(vals_mm, axis=0)), axis=0)
        ],
    }


def _analyze_ball_noise(samples: list[dict[str, Any]], analysis_dir: Path) -> dict[str, Any]:
    valid_rows = [r for r in samples if bool(r.get("rawValid", r.get("valid", False)))]
    measured = _numeric_array(valid_rows, "measuredBall", 3)
    observed = _numeric_array(valid_rows, "observedBall", 3)
    raw = _numeric_array(valid_rows, "rawBall", 3)
    t = np.asarray([_row_float(r, "time", i * 0.005)
                    for i, r in enumerate(valid_rows)], dtype=np.float64)
    report: dict[str, Any] = {
        "sampleCount": len(samples),
        "validSampleCount": len(valid_rows),
    }
    if measured.shape[0] == 0 or observed.shape[0] == 0:
        report["warning"] = "no valid ball samples"
    else:
        residual = measured - observed
        report["measuredMinusEstimator"] = _robust_axis_stats_mm(residual)
        raw_residual = raw - observed if raw.shape == observed.shape else residual
        report["rawMinusEstimator"] = _robust_axis_stats_mm(raw_residual)
        if measured.shape[0] >= 3:
            changed = np.ones(measured.shape[0], dtype=bool)
            changed[1:] = np.linalg.norm(np.diff(measured, axis=0), axis=1) > 1e-7
            unique_measured = measured[changed]
            report["uniqueMeasurementCount"] = int(unique_measured.shape[0])
            if unique_measured.shape[0] >= 3:
                second_diff = (
                    unique_measured[1:-1]
                    - 0.5 * (unique_measured[:-2] + unique_measured[2:])
                )
                report["secondDifferenceJitter"] = _robust_axis_stats_mm(second_diff)
                report["stepDelta"] = _robust_axis_stats_mm(
                    np.diff(unique_measured, axis=0))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
            labels = ("x", "y", "z")
            for axis_idx, label in enumerate(labels):
                axes[axis_idx].plot(
                    t, residual[:, axis_idx] * 1000.0,
                    linewidth=0.9,
                    label=f"measured - estimator {label}")
                axes[axis_idx].axhline(0.0, color="k", linestyle=":", linewidth=0.8)
                axes[axis_idx].set_ylabel("mm")
                axes[axis_idx].grid(True, alpha=0.25)
                axes[axis_idx].legend(loc="upper right")
            axes[-1].set_xlabel("time (s)")
            fig.suptitle("Ball Observation Residuals")
            fig.tight_layout()
            fig.savefig(analysis_dir / "ball_noise_residuals.png", dpi=130)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001 - report, do not fail analysis.
            report["plotWarning"] = str(exc)
    (analysis_dir / "ball_noise_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["Ball Observation Noise Report", ""]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    (analysis_dir / "ball_noise_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return report


def _point_or_none(raw: Any) -> list[float] | None:
    if not isinstance(raw, list) or len(raw) < 3:
        return None
    try:
        values = [float(raw[0]), float(raw[1]), float(raw[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return [round(v, 6) for v in values]


def _write_trajectory_player_html(
    run_dir: Path,
    analysis_dir: Path,
    snapshots: list[dict[str, Any]],
    metadata: dict[str, Any],
    outputs: dict[str, str],
) -> Path:
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    limits = config.get("limits", {}) if isinstance(config, dict) else {}
    frames = _read_jsonl(run_dir / "frames.jsonl")
    points = []
    for row in snapshots:
        ros = row.get("ros", {}) if isinstance(row.get("ros"), dict) else {}
        points.append({
            "time": row.get("time"),
            "mode": row.get("mode"),
            "valid": bool(row.get("valid", False)),
            "raw": _point_or_none(row.get("rawBall")),
            "obs": _point_or_none(row.get("observedBall")),
            "measured": _point_or_none(row.get("measuredBall")),
            "racket": _point_or_none(row.get("racket")),
            "cmdRacket": _point_or_none(row.get("commandRacket")),
            "joint": _point_or_none(ros.get("jointEe")),
            "cart": _point_or_none(ros.get("cartPosition")),
        })
    if not points:
        samples = _read_jsonl(run_dir / "control_samples.jsonl")
        for row in samples[::max(1, int(len(samples) / 2000) or 1)]:
            points.append({
                "time": row.get("time"),
                "mode": row.get("mode"),
                "valid": bool(row.get("valid", False)),
                "raw": _point_or_none(row.get("rawBall")),
                "obs": _point_or_none(row.get("observedBall")),
                "measured": _point_or_none(row.get("measuredBall")),
                "racket": None,
                "cmdRacket": None,
                "joint": None,
                "cart": None,
            })
    payload = {
        "limits": limits,
        "anchor": config.get("anchor") if isinstance(config, dict) else None,
        "ballBox": config.get("ballBox") if isinstance(config, dict) else None,
        "eeBox": config.get("eeBox") if isinstance(config, dict) else None,
        "points": points,
        "frames": frames,
        "outputs": outputs,
    }
    safe_payload = json.dumps(payload, separators=(",", ":"))
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Probe Recording Analysis</title>
<style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f8fafc; color:#172033; }}
header {{ display:flex; gap:14px; align-items:center; padding:12px 16px; border-bottom:1px solid #d7dee8; background:white; }}
h1 {{ margin:0; font-size:18px; }}
main {{ display:grid; grid-template-columns:minmax(0,1fr) 360px; min-height:calc(100vh - 54px); }}
.views {{ display:grid; grid-template-columns:1.3fr .9fr; grid-template-rows:1fr 1fr; min-height:0; }}
canvas {{ width:100%; height:100%; min-height:260px; border-right:1px solid #d7dee8; border-bottom:1px solid #d7dee8; background:linear-gradient(#fff,#edf2f7); }}
#view3d {{ grid-row:1 / span 2; }}
aside {{ padding:14px; background:white; border-left:1px solid #d7dee8; overflow:auto; }}
img {{ width:100%; background:#0f172a; margin:10px 0; }}
.row {{ display:flex; gap:8px; align-items:center; margin:10px 0; }}
input[type=range] {{ flex:1; }}
.mono {{ font-variant-numeric:tabular-nums; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; line-height:1.6; }}
a {{ color:#0f766e; }}
</style>
</head>
<body>
<header><h1>Probe Recording Analysis</h1><button id="play">Play</button><div id="status" class="mono"></div></header>
<main>
<div class="views">
<canvas id="view3d"></canvas>
<canvas id="top"></canvas>
<canvas id="side"></canvas>
</div>
<aside>
<div class="row"><input id="scrub" type="range" min="0" max="0" value="0"><span id="idx" class="mono">0</span></div>
<img id="cameraFrame" alt="recorded camera frame">
<div id="cameraStatus" class="mono">camera frames</div>
<div class="mono" id="readout"></div>
<h3>Outputs</h3>
<ul id="links"></ul>
</aside>
</main>
<script>
const DATA = {safe_payload};
const points = DATA.points || [];
let i = 0, playing = false, timer = null;
const limits = DATA.limits || {{x0:-1,x1:1,y0:-1,y1:1,z0:0,z1:2}};
const canvases = {{view3d:document.getElementById('view3d'), top:document.getElementById('top'), side:document.getElementById('side')}};
const scrub = document.getElementById('scrub');
const idx = document.getElementById('idx');
const status = document.getElementById('status');
const readout = document.getElementById('readout');
const play = document.getElementById('play');
const cameraFrame = document.getElementById('cameraFrame');
const cameraStatus = document.getElementById('cameraStatus');
scrub.max = Math.max(0, points.length - 1);
function resize(c) {{ const r=c.getBoundingClientRect(), d=window.devicePixelRatio||1; c.width=Math.max(1,r.width*d); c.height=Math.max(1,r.height*d); c.getContext('2d').setTransform(d,0,0,d,0,0); }}
function pObj(a) {{ return a ? {{x:a[0],y:a[1],z:a[2]}} : null; }}
function topP(p,c) {{ const r=c.getBoundingClientRect(); return {{x:(p.x-limits.x0)/(limits.x1-limits.x0)*r.width,y:r.height-(p.y-limits.y0)/(limits.y1-limits.y0)*r.height}}; }}
function sideP(p,c) {{ const r=c.getBoundingClientRect(); return {{x:(p.x-limits.x0)/(limits.x1-limits.x0)*r.width,y:r.height-(p.z-limits.z0)/(limits.z1-limits.z0)*r.height}}; }}
function sceneCenter() {{ return {{x:(limits.x0+limits.x1)/2,y:(limits.y0+limits.y1)/2,z:(limits.z0+limits.z1)/2}}; }}
function sceneRadius() {{ return Math.max(limits.x1-limits.x0, limits.y1-limits.y0, limits.z1-limits.z0) * 0.58; }}
function p3(p,c) {{ const r=c.getBoundingClientRect(), ce=sceneCenter(); let x=p.x-ce.x,y=p.y-ce.y,z=p.z-ce.z; const yaw=-0.72,pitch=0.46,cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch); const x1=cy*x-sy*y,y1=sy*x+cy*y,y2=cp*y1-sp*z; const scale=Math.min(r.width,r.height)*0.42/Math.max(sceneRadius(),1e-6); return {{x:r.width*0.52+x1*scale,y:r.height*0.54-y2*scale}}; }}
function dot(ctx,q,color,rad) {{ ctx.fillStyle=color; ctx.strokeStyle='white'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(q.x,q.y,rad,0,Math.PI*2); ctx.fill(); ctx.stroke(); }}
function trail(ctx,c,proj,key,color,n=180) {{ ctx.strokeStyle=color; ctx.lineWidth=2; ctx.globalAlpha=.45; ctx.beginPath(); let moved=false; for (let k=Math.max(0,i-n); k<=i; k++) {{ const p=pObj(points[k]&&points[k][key]); if(!p) continue; const q=proj(p,c); if(!moved) {{ctx.moveTo(q.x,q.y); moved=true;}} else ctx.lineTo(q.x,q.y); }} ctx.stroke(); ctx.globalAlpha=1; }}
function drawCanvas(c, title, proj) {{ const ctx=c.getContext('2d'), r=c.getBoundingClientRect(); ctx.clearRect(0,0,r.width,r.height); trail(ctx,c,proj,'raw','#f28e2b'); trail(ctx,c,proj,'obs','#e15759'); trail(ctx,c,proj,'racket','#4e79a7'); const row=points[i]||{{}}; [['raw','#f28e2b',8],['obs','#e15759',6],['racket','#4e79a7',8],['joint','#16a34a',6],['cart','#9333ea',6]].forEach(([k,col,rad])=>{{const p=pObj(row[k]); if(p) dot(ctx,proj(p,c),col,rad);}}); ctx.fillStyle='#273445'; ctx.font='14px sans-serif'; ctx.fillText(title,14,22); }}
function draw() {{ for (const c of Object.values(canvases)) resize(c); drawCanvas(canvases.view3d,'3D replay',p3); drawCanvas(canvases.top,'Top x/y',topP); drawCanvas(canvases.side,'Side x/z',sideP); const row=points[i]||{{}}; const frame=(DATA.frames||[])[Math.min(i,Math.max(0,(DATA.frames||[]).length-1))]; if(frame&&frame.path) {{ cameraFrame.style.display='block'; cameraFrame.src=`../${{frame.path}}`; }} else {{ cameraFrame.style.display='none'; }} cameraStatus.textContent=`camera frames=${{(DATA.frames||[]).length}}`; scrub.value=i; idx.textContent=`${{i+1}}/${{points.length}}`; status.textContent=`t=${{Number(row.time||0).toFixed(3)}}s mode=${{row.mode||''}}`; readout.textContent=JSON.stringify(row,null,2); }}
function step() {{ if(!playing) return; i=(i+1)%Math.max(1,points.length); draw(); timer=setTimeout(step, 1000/60); }}
play.onclick=()=>{{ playing=!playing; play.textContent=playing?'Pause':'Play'; if(playing) step(); }};
scrub.oninput=()=>{{ i=Number(scrub.value)||0; draw(); }};
const links=document.getElementById('links'); Object.entries(DATA.outputs||{{}}).forEach(([k,v])=>{{ if(!v) return; const li=document.createElement('li'), a=document.createElement('a'); a.href=v; a.textContent=`${{k}}: ${{v}}`; li.appendChild(a); links.appendChild(li); }});
window.addEventListener('resize', draw); draw();
</script>
</body>
</html>
"""
    out = analysis_dir / "trajectory_player.html"
    out.write_text(doc, encoding="utf-8")
    return out


def _obs_names(width: int) -> list[str]:
    names = [
        "arm_q[0]", "arm_q[1]", "arm_q[2]", "arm_q[3]", "arm_q[4]", "arm_q[5]", "arm_q[6]",
        "arm_dq[0]", "arm_dq[1]", "arm_dq[2]", "arm_dq[3]", "arm_dq[4]", "arm_dq[5]", "arm_dq[6]",
        "base_q_x", "base_q_y", "base_q_yaw",
        "base_dq_x", "base_dq_y", "base_dq_yaw",
        "ball_pos_x", "ball_pos_y", "ball_pos_z",
        "ball_vel_x", "ball_vel_y", "ball_vel_z",
        "racket_pos_x", "racket_pos_y", "racket_pos_z",
        "racket_vel_x", "racket_vel_y", "racket_vel_z",
        "rel_x", "rel_y", "rel_z",
        "prev_action[0]", "prev_action[1]", "prev_action[2]", "prev_action[3]",
        "prev_action[4]", "prev_action[5]", "prev_action[6]",
        "arm_cmd_error[0]", "arm_cmd_error[1]", "arm_cmd_error[2]", "arm_cmd_error[3]",
        "arm_cmd_error[4]", "arm_cmd_error[5]", "arm_cmd_error[6]",
        "ball_age_norm",
        "tau_act_norm",
        "arm_cmd_qvel[0]", "arm_cmd_qvel[1]", "arm_cmd_qvel[2]",
        "arm_cmd_qvel[3]", "arm_cmd_qvel[4]", "arm_cmd_qvel[5]",
        "arm_cmd_qvel[6]",
        "active_cmd_error[0]", "active_cmd_error[1]", "active_cmd_error[2]",
        "active_cmd_error[3]", "active_cmd_error[4]", "active_cmd_error[5]",
        "active_cmd_error[6]",
        "t_contact_est", "t_margin",
    ]
    if width <= len(names):
        return names[:width]
    return names + [f"obs[{i}]" for i in range(len(names), width)]


def _decimate_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    step = max(1, int(math.ceil(len(rows) / max_rows)))
    return rows[::step]


def _make_analysis_data(
    run_dir: Path,
    snapshots: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    metadata: dict[str, Any],
    outputs: dict[str, str],
    noise_report: dict[str, Any],
) -> dict[str, Any]:
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    frames_raw = _read_jsonl(run_dir / "frames.jsonl")
    frames = []
    for row in frames_raw:
        path_text = str(row.get("path", ""))
        filename = Path(path_text).name
        if filename:
            frames.append({
                "filename": filename,
                "wallTimeS": row.get("wallTimeS"),
                "info": row.get("info") if isinstance(row.get("info"), dict) else {},
            })
    sim_frames_raw = _read_jsonl(run_dir / "sim_frames.jsonl")
    sim_frames = []
    for row in sim_frames_raw:
        path_text = str(row.get("path", ""))
        filename = Path(path_text).name
        if filename:
            sim_frames.append({
                "filename": filename,
                "wallTimeS": row.get("wallTimeS"),
                "info": row.get("info") if isinstance(row.get("info"), dict) else {},
            })
    points = []
    for row in _decimate_rows(snapshots, 3000):
        ros = row.get("ros", {}) if isinstance(row.get("ros"), dict) else {}
        sim = row.get("sim") if isinstance(row.get("sim"), dict) else {}
        points.append({
            "time": row.get("time"),
            "mode": row.get("mode"),
            "valid": bool(row.get("valid", False)),
            "raw": _point_or_none(row.get("rawBall")),
            "measured": _point_or_none(row.get("measuredBall")),
            "obs": _point_or_none(row.get("observedBall")),
            "racket": _point_or_none(row.get("racket")),
            "cmdRacket": _point_or_none(row.get("commandRacket")),
            "joint": _point_or_none(ros.get("jointEe")),
            "cart": _point_or_none(ros.get("cartPosition")),
            "simRacket": _point_or_none(sim.get("racket")),
            "simBall": _point_or_none(sim.get("ball")),
        })
    curve_rows = []
    sample_subset = _decimate_rows(samples, 8000)
    obs_width = max((len(r.get("policyObs", [])) for r in sample_subset), default=0)
    action_width = max((len(r.get("policyAction", [])) for r in sample_subset), default=0)
    sim_obs_width = max((
        len(r.get("sim", {}).get("policyObs", []))
        for r in sample_subset if isinstance(r.get("sim"), dict)
    ), default=0)
    sim_action_width = max((
        len(r.get("sim", {}).get("policyAction", []))
        for r in sample_subset if isinstance(r.get("sim"), dict)
    ), default=0)
    for row in sample_subset:
        ros = row.get("ros") if isinstance(row.get("ros"), dict) else {}
        command = row.get("command") if isinstance(row.get("command"), dict) else {}
        sim = row.get("sim") if isinstance(row.get("sim"), dict) else {}
        curve_rows.append({
            "time": row.get("time"),
            "tick": row.get("tick"),
            "mode": row.get("mode"),
            "valid": bool(row.get("valid", False)),
            "age": row.get("age"),
            "rawBall": _point_or_none(row.get("rawBall")),
            "measuredBall": _point_or_none(row.get("measuredBall")),
            "observedBall": _point_or_none(row.get("observedBall")),
            "measuredVelocity": _point_or_none(row.get("measuredVelocity")),
            "observedVelocity": _point_or_none(row.get("observedVelocity")),
            "policyObs": row.get("policyObs", []) if isinstance(row.get("policyObs"), list) else [],
            "policyAction": row.get("policyAction", []) if isinstance(row.get("policyAction"), list) else [],
            "rosJointPosition": ros.get("jointPosition", []) if isinstance(ros.get("jointPosition"), list) else [],
            "rosJointVelocity": ros.get("jointVelocity", []) if isinstance(ros.get("jointVelocity"), list) else [],
            "commandPosition": command.get("position", []) if isinstance(command.get("position"), list) else [],
            "racket": _point_or_none(ros.get("jointEe")),
            "simPolicyObs": sim.get("policyObs", []) if isinstance(sim.get("policyObs"), list) else [],
            "simPolicyAction": sim.get("policyAction", []) if isinstance(sim.get("policyAction"), list) else [],
            "simPolicyCommand": sim.get("policyCommand", []) if isinstance(sim.get("policyCommand"), list) else [],
            "simJointPosition": sim.get("jointPosition", []) if isinstance(sim.get("jointPosition"), list) else [],
            "simJointVelocity": sim.get("jointVelocity", []) if isinstance(sim.get("jointVelocity"), list) else [],
            "simRacket": _point_or_none(sim.get("racket")),
        })
    sim_comparison = {}
    sim_comparison_path = run_dir / "analysis" / "sim_real_comparison.json"
    if sim_comparison_path.is_file():
        try:
            sim_comparison = json.loads(sim_comparison_path.read_text(encoding="utf-8"))
        except Exception:
            sim_comparison = {}
    return {
        "metadata": {
            "runDir": str(run_dir),
            "sampleCount": len(samples),
            "snapshotCount": len(snapshots),
            "curveSampleCount": len(curve_rows),
        },
        "limits": config.get("limits", {}) if isinstance(config, dict) else {},
        "anchor": config.get("anchor") if isinstance(config, dict) else None,
        "ballBox": config.get("ballBox") if isinstance(config, dict) else None,
        "eeBox": config.get("eeBox") if isinstance(config, dict) else None,
        "frames": frames,
        "simFrames": sim_frames,
        "points": points,
        "samples": curve_rows,
        "obsNames": _obs_names(obs_width),
        "actionNames": [f"action[{i}]" for i in range(action_width)],
        "simObsNames": _obs_names(sim_obs_width),
        "simActionNames": [f"action[{i}]" for i in range(sim_action_width)],
        "noiseReport": noise_report,
        "simRealComparison": sim_comparison,
        "outputs": dict(outputs),
    }


def _recording_analysis_main(
    run_dir_text: str,
    status_q,
    video_fps: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    run_dir = Path(run_dir_text)
    analysis_dir = run_dir / "analysis"
    outputs: dict[str, str] = {}
    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _analysis_write_status(
            status_q, active=True, phase="loading", analysis_dir=analysis_dir)
        metadata_path = run_dir / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file() else {}
        )
        raw_samples = _read_jsonl(run_dir / "control_samples.jsonl")
        sim_samples = _read_jsonl(run_dir / "sim_control_samples.jsonl")
        samples, alignment_report = _align_sim_real_samples(
            raw_samples, sim_samples)
        summary_path = run_dir / "summary.json"
        summary_present = summary_path.is_file()
        recording_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_present else {}
        )
        try:
            recorder_dropped = max(
                0, int(recording_summary.get("dropped", 0) or 0))
        except (TypeError, ValueError):
            recorder_dropped = 0
        alignment_events = [
            row for row in _read_jsonl(run_dir / "events.jsonl")
            if str(row.get("event", "")) == "sim_alignment"
        ]
        final_alignment_event = (
            dict(alignment_events[-1]) if alignment_events else {})
        invalid_reasons: list[str] = []
        sequence_aligned = bool(
            alignment_report.get("alignmentValid", False))
        schema_v2 = any(
            int(row.get("schemaVersion", 0) or 0) >= 2
            for row in raw_samples
        )
        config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
        try:
            control_rate_hz = max(
                1e-6, float(config.get("controlRateHz", 200.0)))
        except (TypeError, ValueError):
            control_rate_hz = 200.0
        realtime_report = _realtime_alignment_report(
            raw_samples, sim_samples, control_rate_hz)
        if schema_v2 and not summary_present:
            invalid_reasons.append("recording_summary_missing")
        if schema_v2 and sim_samples and not final_alignment_event:
            invalid_reasons.append("final_sim_alignment_event_missing")
        if (schema_v2 and sim_samples
                and not bool(realtime_report.get("realtimeAligned", False))):
            invalid_reasons.append("sim_not_realtime_aligned")
        if recorder_dropped > 0:
            invalid_reasons.append("recorder_queue_dropped_samples")
        if (final_alignment_event.get("simEnabled") is True
                and not bool(final_alignment_event.get("alignmentValid", False))):
            invalid_reasons.append("sim_flush_or_stream_invalid")
        alignment_report["alignmentValid"] = bool(
            sequence_aligned and not invalid_reasons)
        alignment_report.update({
            "sequenceAligned": sequence_aligned,
            "realtimeAligned": bool(
                realtime_report.get("realtimeAligned", False)),
            "realtime": realtime_report,
            "recorderDropped": recorder_dropped,
            "recordingSummaryPresent": summary_present,
            "finalAlignmentEvent": final_alignment_event,
            "invalidReasons": invalid_reasons,
        })
        (analysis_dir / "alignment_manifest.json").write_text(
            json.dumps(alignment_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs["alignmentManifest"] = "alignment_manifest.json"
        snapshots = _read_jsonl(run_dir / "snapshots.jsonl")

        _analysis_write_status(
            status_q, active=True, phase="camera_video",
            analysis_dir=analysis_dir, outputs=outputs)
        video_path, video_info = _build_camera_video(
            run_dir, analysis_dir, fps=max(1.0, float(video_fps)))
        (analysis_dir / "camera_video_info.json").write_text(
            json.dumps(video_info, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if video_path is not None:
            outputs["cameraVideo"] = video_path.name
        sim_video_path, sim_video_info = _build_sim_video(
            run_dir, analysis_dir, fps=max(1.0, float(video_fps)))
        (analysis_dir / "sim_video_info.json").write_text(
            json.dumps(sim_video_info, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs["simVideoInfo"] = "sim_video_info.json"
        if sim_video_path is not None:
            outputs["simVideo"] = sim_video_path.name

        _analysis_write_status(
            status_q, active=True, phase="policy_curves",
            analysis_dir=analysis_dir, outputs=outputs)
        trace_csv = _write_policy_trace_csv(samples, analysis_dir)
        outputs["policyTraceCsv"] = trace_csv.name
        for plot_path in _plot_policy_curves(samples, analysis_dir):
            outputs[plot_path.stem] = plot_path.name

        _analysis_write_status(
            status_q, active=True, phase="noise",
            analysis_dir=analysis_dir, outputs=outputs)
        noise_report = _analyze_ball_noise(samples, analysis_dir)
        outputs["ballNoiseReport"] = "ball_noise_report.json"
        outputs["ballNoiseResiduals"] = "ball_noise_residuals.png"
        sim_report = _analyze_sim_real_gap(
            samples, analysis_dir, alignment=alignment_report)
        outputs["simRealComparison"] = "sim_real_comparison.json"
        outputs["simRealComparisonCsv"] = "sim_real_comparison.csv"
        outputs["simRealComparisonText"] = "sim_real_comparison.txt"
        if sim_report.get("comparisonPlot"):
            outputs["simRealComparisonPlot"] = str(sim_report["comparisonPlot"])

        _analysis_write_status(
            status_q, active=True, phase="player",
            analysis_dir=analysis_dir, outputs=outputs)
        player = _write_trajectory_player_html(
            run_dir, analysis_dir, snapshots, metadata, outputs)
        outputs["trajectoryPlayer"] = player.name
        analysis_data = _make_analysis_data(
            run_dir, snapshots, samples, metadata, outputs, noise_report)
        (analysis_dir / "analysis_data.json").write_text(
            _strict_json_dumps(analysis_data, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        outputs["analysisData"] = "analysis_data.json"
        summary = {
            "runDir": str(run_dir),
            "analysisDir": str(analysis_dir),
            "samples": len(samples),
            "simSamples": len(sim_samples),
            "alignmentValid": bool(
                alignment_report.get("alignmentValid", False)),
            "snapshots": len(snapshots),
            "outputs": outputs,
            "completedWallTimeS": time.time(),
        }
        (analysis_dir / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _analysis_write_status(
            status_q, active=False, phase="done",
            analysis_dir=analysis_dir, outputs=outputs)
    except BaseException as exc:  # noqa: BLE001 - report to UI.
        _analysis_write_status(
            status_q,
            active=False,
            phase="error",
            analysis_dir=analysis_dir,
            outputs=outputs,
            error=f"{exc}\n{traceback.format_exc()}",
        )


class AsyncProbeRecorder:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.ctx = mp.get_context("spawn")
        self.event_q = None
        self.status_q = None
        self.process: mp.Process | None = None
        self.analysis_status_q = None
        self.analysis_process: mp.Process | None = None
        self.run_dir: Path | None = None
        self.active = False
        self.last_status: dict[str, Any] = {
            "active": False,
            "runDir": None,
            "counts": {},
            "elapsedS": 0.0,
            "dropped": 0,
            "error": None,
        }
        self.analysis_status: dict[str, Any] = {
            "active": False,
            "phase": "idle",
            "analysisDir": None,
            "outputs": {},
            "error": None,
        }
        self.dropped = 0
        self.next_control_record_wall = 0.0
        self.next_frame_record_wall = 0.0
        self.next_sim_frame_record_wall = 0.0
        self.last_frame_seq = -1
        self.last_sim_frame_seq = -1
        self.frozen_source_seq: int | None = None

    def start(
        self,
        config: dict[str, Any],
        initial_view: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            if self.active:
                return self.status_locked()
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.run_dir = self.cfg.recording_output_root / f"probe_recording_{stamp}"
            self.event_q = self.ctx.Queue(maxsize=max(8, int(self.cfg.recording_queue_size)))
            self.status_q = self.ctx.Queue(maxsize=1)
            metadata = {
                "config": config,
                "recording": {
                    "controlRateHz": float(self.cfg.recording_control_rate_hz),
                    "imageRateHz": float(self.cfg.recording_image_rate_hz),
                    "queueSize": int(self.cfg.recording_queue_size),
                },
            }
            self.process = self.ctx.Process(
                target=_recording_writer_main,
                args=(str(self.run_dir), self.event_q, self.status_q, metadata),
                name="policy_probe_recording_writer",
                daemon=False,
            )
            self.process.start()
            self.active = True
            self.dropped = 0
            self.next_control_record_wall = 0.0
            self.next_frame_record_wall = 0.0
            self.next_sim_frame_record_wall = 0.0
            self.last_frame_seq = -1
            self.last_sim_frame_seq = -1
            self.frozen_source_seq = None
            self.last_status = {
                "active": True,
                "runDir": str(self.run_dir),
                "counts": {},
                "elapsedS": 0.0,
                "dropped": 0,
                "error": None,
            }
        self.record_event({"event": "start", "wallTimeS": time.time()})
        if initial_view:
            self.record_view(initial_view)
        return self.status()

    def stop(
        self,
        final_view: dict[str, Any] | None = None,
        analyze: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            active = self.active
        if not active:
            status = self.status()
            if analyze and status.get("runDir"):
                return self.start_analysis(str(status["runDir"]))
            return status
        if final_view:
            self.record_view(final_view)
        self.record_event({"event": "stop", "wallTimeS": time.time()})
        with self.lock:
            q = self.event_q
            process = self.process
            dropped = int(self.dropped)
        if q is not None:
            try:
                q.put({"kind": "stop", "dropped": dropped}, timeout=5.0)
            except Full:
                with self.lock:
                    self.dropped += 1
                    self.last_status["error"] = (
                        "recording queue did not drain; stop marker was not written")
        if process is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                _terminate_process_tree(process, "recording writer")
        with self.lock:
            self.active = False
            run_dir = str(self.run_dir) if self.run_dir is not None else None
        status = self.status()
        with self.lock:
            self.process = None
            self.event_q = None
            self.status_q = None
        _close_mp_queue(q)
        if analyze and run_dir:
            status = self.start_analysis(run_dir)
        return status

    def shutdown(self) -> None:
        self.stop()
        with self.lock:
            process = self.analysis_process
        if process is not None and process.is_alive():
            process.join(timeout=1.0)
            if process.is_alive():
                _terminate_process_tree(process, "analysis process")

    def _drain_status_locked(self) -> None:
        q = self.status_q
        if q is None:
            pass
        else:
            while True:
                try:
                    item = q.get_nowait()
                except Empty:
                    break
                if isinstance(item, dict):
                    item = dict(item)
                    item["dropped"] = int(self.dropped)
                    self.last_status = item
                    if item.get("active") is False:
                        self.active = False
        aq = self.analysis_status_q
        if aq is not None:
            while True:
                try:
                    item = aq.get_nowait()
                except Empty:
                    break
                if isinstance(item, dict):
                    self.analysis_status = dict(item)

    def status_locked(self) -> dict[str, Any]:
        self._drain_status_locked()
        status = dict(self.last_status)
        status["active"] = bool(self.active)
        status["dropped"] = int(self.dropped)
        if self.run_dir is not None and not status.get("runDir"):
            status["runDir"] = str(self.run_dir)
        if self.process is not None and self.active and not self.process.is_alive():
            status["active"] = False
            status["error"] = (
                status.get("error")
                or f"recording writer exited: {self.process.exitcode}")
            self.active = False
        if (self.analysis_process is not None
                and self.analysis_status.get("active")
                and not self.analysis_process.is_alive()):
            self.analysis_status["active"] = False
            if self.analysis_process.exitcode not in (0, None):
                self.analysis_status["error"] = (
                    self.analysis_status.get("error")
                    or f"analysis exited: {self.analysis_process.exitcode}")
        status["analysis"] = dict(self.analysis_status)
        return status

    def status(self) -> dict[str, Any]:
        with self.lock:
            return self.status_locked()

    def start_analysis(self, run_dir_text: str | None = None) -> dict[str, Any]:
        with self.lock:
            self._drain_status_locked()
            if (self.analysis_process is not None
                    and self.analysis_process.is_alive()):
                return self.status_locked()
            run_dir = Path(run_dir_text) if run_dir_text else self.run_dir
            if run_dir is None:
                self.analysis_status = {
                    "active": False,
                    "phase": "error",
                    "analysisDir": None,
                    "outputs": {},
                    "error": "no recording directory available",
                }
                return self.status_locked()
            self.analysis_status_q = self.ctx.Queue(maxsize=1)
            analysis_dir = run_dir / "analysis"
            self.analysis_status = {
                "active": True,
                "phase": "starting",
                "analysisDir": str(analysis_dir),
                "outputs": {},
                "error": None,
            }
            self.analysis_process = self.ctx.Process(
                target=_recording_analysis_main,
                args=(
                    str(run_dir),
                    self.analysis_status_q,
                    float(self.cfg.recording_analysis_video_fps),
                ),
                name="policy_probe_recording_analysis",
                daemon=False,
            )
            self.analysis_process.start()
            return self.status_locked()

    def _put(self, item: dict[str, Any]) -> bool:
        with self.lock:
            if not self.active or self.event_q is None:
                return False
            q = self.event_q
        try:
            q.put(item, block=False)
            return True
        except Full:
            with self.lock:
                self.dropped += 1
            return False

    def record_event(self, event: dict[str, Any]) -> None:
        if not event:
            return
        self._put({"kind": "event", "event": dict(event)})

    def record_view(self, view: dict[str, Any] | None) -> None:
        if not isinstance(view, dict):
            return
        payload = dict(view)
        payload.setdefault("wallTimeS", round(float(time.time()), 6))
        self._put({"kind": "view", "view": payload})

    def record_snapshot(self, snapshot: dict[str, Any] | None) -> None:
        if not isinstance(snapshot, dict):
            return
        self._put({"kind": "snapshot", "snapshot": dict(snapshot)})

    def record_control_sample(self, sample: dict[str, Any]) -> None:
        rate = float(self.cfg.recording_control_rate_hz)
        if rate <= 0.0:
            return
        now = time.perf_counter()
        with self.lock:
            if not self.active:
                return
            if (self.frozen_source_seq is not None
                    and int(sample.get("sourceSeq", -1))
                    > self.frozen_source_seq):
                return
            # At the configured control rate, record every logical tick.  A
            # wall-clock `next = now + period` gate loses roughly every other
            # sample under normal timer jitter and breaks sourceSeq alignment.
            if rate + 1e-6 < float(self.cfg.control_rate_hz):
                if now < self.next_control_record_wall:
                    return
                self.next_control_record_wall = now + 1.0 / max(rate, 1e-6)
        self._put({"kind": "control", "sample": sample})

    def record_sim_control_samples(
        self,
        samples: list[dict[str, Any]] | None,
    ) -> None:
        if not samples or float(self.cfg.recording_control_rate_hz) <= 0.0:
            return
        with self.lock:
            frozen_source_seq = self.frozen_source_seq
        batch = []
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sync = sample.get("sync")
            source_seq = (
                sync.get("sourceSeq") if isinstance(sync, dict) else None)
            if (frozen_source_seq is not None
                    and source_seq is not None
                    and int(source_seq) > frozen_source_seq):
                continue
            batch.append(dict(sample))
        if not batch:
            return
        self._put({"kind": "sim_control_batch", "samples": batch})

    def freeze(self, source_seq: int) -> int:
        with self.lock:
            self.frozen_source_seq = int(source_seq)
            return self.frozen_source_seq

    def record_camera_frame(
        self,
        frame: bytes | None,
        info: dict[str, Any] | None,
    ) -> None:
        if frame is None:
            return
        rate = float(self.cfg.recording_image_rate_hz)
        if rate <= 0.0:
            return
        info = dict(info) if isinstance(info, dict) else {}
        seq = int(info.get("seq", 0) or 0)
        now = time.perf_counter()
        with self.lock:
            if not self.active:
                return
            if seq > 0 and seq == self.last_frame_seq:
                return
            if now < self.next_frame_record_wall:
                return
            self.last_frame_seq = seq
            self.next_frame_record_wall = now + 1.0 / max(rate, 1e-6)
        self._put({"kind": "frame", "frame": bytes(frame), "info": info})

    def record_sim_frame(
        self,
        frame: bytes | None,
        info: dict[str, Any] | None,
    ) -> None:
        if frame is None:
            return
        rate = float(self.cfg.recording_image_rate_hz)
        if rate <= 0.0:
            return
        info = dict(info) if isinstance(info, dict) else {}
        seq = int(info.get("seq", 0) or 0)
        now = time.perf_counter()
        with self.lock:
            if not self.active:
                return
            if seq > 0 and seq == self.last_sim_frame_seq:
                return
            if now < self.next_sim_frame_record_wall:
                return
            self.last_sim_frame_seq = seq
            self.next_sim_frame_record_wall = now + 1.0 / max(rate, 1e-6)
        self._put({"kind": "sim_frame", "frame": bytes(frame), "info": info})


@dataclass(frozen=True)
class CameraRuntimeOptions:
    detect: bool
    visualize: bool
    use_depth_sensor: bool
    use_undistort: bool
    use_kalman: bool
    enable_ellipse_fit: bool
    enable_temporal_tracker: bool
    enable_color_filter: bool

    def key(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        return (
            bool(self.use_depth_sensor),
            bool(self.use_undistort),
            bool(self.use_kalman),
            bool(self.enable_ellipse_fit),
            bool(self.enable_temporal_tracker),
            bool(self.enable_color_filter),
        )

    def payload(self) -> dict[str, bool]:
        return {
            "detect": bool(self.detect),
            "visualize": bool(self.visualize),
            "sensorDepth": bool(self.use_depth_sensor),
            "undistort": bool(self.use_undistort),
            "kalman": bool(self.use_kalman),
            "ellipseFit": bool(self.enable_ellipse_fit),
            "temporalTracker": bool(self.enable_temporal_tracker),
            "colorFilter": bool(self.enable_color_filter),
        }


@dataclass(frozen=True)
class RuntimeOptions:
    use_ros_state: bool
    publish_commands: bool
    enable_compensation: bool
    compensation_mode: str
    comp_delay_ms: float
    comp_actuator_tau_s: float
    comp_actuator_gain: float
    comp_mpc_beta: float
    comp_mpc_delay_scale: float
    comp_mpc_tau_scale: float
    comp_mpc_horizon_steps: int
    comp_mpc_tracking_weight: float
    comp_mpc_nominal_weight: float
    comp_mpc_delta_weight: float
    comp_mpc_max_delta_rad: float
    rl_use_valid_ball_obs_age: bool
    enable_ball_capture_gate: bool
    enable_ball_release_gate: bool
    enable_ball_lost_failsafe: bool
    enable_done_monitoring: bool
    workspace_safety_mode: str
    done_prediction_horizon_s: float
    done_prediction_dt_s: float
    done_prediction_margin_m: float
    done_prediction_orientation_margin_rad: float
    enable_sim_mirror: bool
    ball_release_use_estimator_state: bool
    ball_release_min_downward_velocity_m_s: float
    ball_release_min_z_drop_m: float
    ball_release_window_s: float
    ball_release_required_consecutive_ticks: int
    ball_release_max_obs_age_s: float
    use_ball_estimator: bool
    estimator_prediction: bool
    estimator_output_mode: str
    estimator_aero_drag: bool
    estimator_release_gated_dynamics: bool
    estimator_max_extrapolation_s: float
    estimator_control_max_extrapolation_s: float
    estimator_frame_hold_output_rate_hz: float
    estimator_trust_loss_s: float
    estimator_degrade_loss_s: float
    estimator_invalid_after_s: float
    estimator_prediction_horizon_s: float
    estimator_q_pos_mm2_s: float
    estimator_q_vel_mult: float
    estimator_q_depth_pos_mult: float
    estimator_q_depth_vel_mult: float
    estimator_r_lateral_mm: float
    estimator_r_depth_mm: float
    ball_observation_rate_hz: float
    ball_source: str
    camera: CameraRuntimeOptions


def _sim_snapshot_unavailable(
    sim_cfg: SimMirrorConfig,
    reason: str | None = None,
    *,
    active: bool = False,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "available": False,
        "active": bool(active),
        "backend": str(getattr(sim_cfg, "backend", "mujoco")),
        "error": reason,
        "frame": {"seq": 0, "ageS": None},
    }


def _runtime_options_for_sim_process(options: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "enable_compensation",
        "compensation_mode",
        "comp_delay_ms",
        "comp_actuator_tau_s",
        "comp_actuator_gain",
        "comp_mpc_beta",
        "comp_mpc_delay_scale",
        "comp_mpc_tau_scale",
        "comp_mpc_horizon_steps",
        "comp_mpc_tracking_weight",
        "comp_mpc_nominal_weight",
        "comp_mpc_delta_weight",
        "comp_mpc_max_delta_rad",
        "rl_use_valid_ball_obs_age",
    ):
        if hasattr(options, key):
            out[key] = getattr(options, key)
    return out


def _sim_runtime_config_matches(
    sim_cfg: SimMirrorConfig,
    options: Any,
    backend: str,
) -> bool:
    """MJX bakes these controls into its JIT/env; CPU MuJoCo applies them live."""
    if not str(backend).startswith("mjx"):
        return True
    expected = {
        "enable_compensation": bool(sim_cfg.enable_compensation),
        "compensation_mode": str(sim_cfg.compensation_mode),
        "comp_delay_ms": float(sim_cfg.comp_delay_ms),
        "comp_actuator_tau_s": float(sim_cfg.comp_actuator_tau_s),
        "comp_actuator_gain": float(sim_cfg.comp_actuator_gain),
        "comp_mpc_beta": float(sim_cfg.comp_mpc_beta),
        "comp_mpc_delay_scale": float(sim_cfg.comp_mpc_delay_scale),
        "comp_mpc_tau_scale": float(sim_cfg.comp_mpc_tau_scale),
        "comp_mpc_horizon_steps": int(sim_cfg.comp_mpc_horizon_steps),
        "comp_mpc_tracking_weight": float(sim_cfg.comp_mpc_tracking_weight),
        "comp_mpc_nominal_weight": float(sim_cfg.comp_mpc_nominal_weight),
        "comp_mpc_delta_weight": float(sim_cfg.comp_mpc_delta_weight),
        "comp_mpc_max_delta_rad": float(sim_cfg.comp_mpc_max_delta_rad),
        "rl_use_valid_ball_obs_age": bool(
            sim_cfg.rl_use_valid_ball_obs_age),
    }
    for key, expected_value in expected.items():
        actual_value = getattr(options, key, expected_value)
        if isinstance(expected_value, float):
            try:
                if not math.isclose(
                        float(actual_value), expected_value,
                        rel_tol=1e-9, abs_tol=1e-9):
                    return False
            except (TypeError, ValueError):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _make_sim_mirror(sim_cfg: SimMirrorConfig) -> Any:
    backend = str(getattr(sim_cfg, "backend", "mujoco") or "mujoco").strip().lower()
    if backend not in {"mujoco", "mjx", "auto"}:
        backend = "mujoco"
    backends = ("mjx", "mujoco") if backend == "auto" else (backend,)
    last_error: Exception | None = None
    for candidate in backends:
        try:
            mirror_cls = (
                RealBallMjxJaxSimMirror
                if candidate == "mjx" else RealBallSimMirror
            )
            return mirror_cls(sim_cfg)
        except Exception as exc:  # noqa: BLE001 - fallback/diagnostics.
            last_error = exc
            if backend != "auto":
                break
    raise last_error or RuntimeError("simulation mirror initialization failed")


def _prewarm_jax_cuda_backend(backend: str) -> None:
    if str(backend).strip().lower() not in {"mjx", "auto"}:
        return
    try:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        devices = jax.devices()
        jnp.asarray([0.0], dtype=jnp.float32).block_until_ready()
        jnp.linalg.solve(
            jnp.eye(1, dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        ).block_until_ready()
        print(f"[probe] JAX prewarm devices: {devices}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - sim init will surface details.
        print(f"[probe][WARN] JAX CUDA prewarm failed: {exc}", file=sys.stderr)


def _sim_mirror_process_main(
    sim_cfg: SimMirrorConfig,
    command_q,
    status_q,
    result_q,
    frame_q,
    parent_pid: int | None,
) -> None:
    shutdown_requested = threading.Event()
    mirror: Any | None = None
    last_frame_seq = -1

    def _request_shutdown(_signum, _frame) -> None:
        shutdown_requested.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _request_shutdown)
        except Exception:
            pass

    def _parent_is_gone() -> bool:
        if parent_pid is None or int(parent_pid) <= 1:
            return False
        try:
            return os.getppid() != int(parent_pid)
        except Exception:
            return False

    def _publish_snapshot(snapshot: dict[str, Any]) -> None:
        _queue_replace_latest(status_q, {
            "kind": "snapshot",
            "snapshot": dict(snapshot),
        })

    def _publish_frame() -> None:
        nonlocal last_frame_seq
        if mirror is None:
            return
        frame, info = mirror.latest_visualization()
        seq = int((info or {}).get("seq", 0) or 0)
        if frame is None or seq == last_frame_seq:
            return
        _queue_replace_latest(frame_q, {
            "kind": "frame",
            "frame": bytes(frame),
            "info": dict(info or {}),
        })
        last_frame_seq = seq

    try:
        _install_parent_death_signal(signal.SIGKILL)
        if _parent_is_gone():
            return
        _queue_replace_latest(status_q, {
            "kind": "snapshot",
            "snapshot": _sim_snapshot_unavailable(
                sim_cfg, "sim subprocess initializing", active=False),
        })
        _prewarm_jax_cuda_backend(str(getattr(sim_cfg, "backend", "mujoco")))
        mirror = _make_sim_mirror(sim_cfg)
        try:
            max_batch_steps_raw = int(float(
                os.environ.get("SIM_CATCHUP_BATCH_STEPS", "1")))
        except (TypeError, ValueError):
            max_batch_steps_raw = 1
        max_batch_steps = max(1, min(16, max_batch_steps_raw))
        if hasattr(mirror, "prewarm"):
            mirror.prewarm(max_batch_steps)
        snapshot = mirror.snapshot()
        snapshot["active"] = False
        snapshot["sync"] = {
            "ready": True,
            "streamContinuous": True,
            "resultDrops": 0,
        }
        _publish_snapshot(snapshot)
        _publish_frame()
        sim_dt_s = max(
            1e-6,
            float(getattr(mirror, "dt", getattr(sim_cfg, "control_dt", 0.005))),
        )
        control_dt_s = max(1e-6, float(getattr(sim_cfg, "control_dt", 0.005)))
        if abs(sim_dt_s - control_dt_s) > max(1e-7, control_dt_s * 1e-4):
            raise RuntimeError(
                "sim/control dt mismatch: "
                f"sim={sim_dt_s:.9f}s control={control_dt_s:.9f}s")
        sim_elapsed_s = 0.0
        deferred_cmd: dict[str, Any] | None = None
        last_stream_seq: int | None = None
        stream_continuous = True
        first_gap_seq: int | None = None
        result_drops = 0
        completion_times: deque[float] = deque(maxlen=400)
        while not shutdown_requested.is_set():
            if _parent_is_gone():
                break
            if deferred_cmd is not None:
                cmd = deferred_cmd
                deferred_cmd = None
            else:
                try:
                    cmd = command_q.get(timeout=0.02)
                except Empty:
                    cmd = None
                except (EOFError, OSError, ValueError):
                    break
            if not isinstance(cmd, dict):
                continue
            kind = str(cmd.get("kind", ""))
            if kind == "stop":
                shutdown_requested.set()
                break
            if kind == "reset":
                arm_q = cmd.get("arm_q")
                snapshot = mirror.reset(
                    None if arm_q is None else np.asarray(arm_q, dtype=np.float32))
                result_drops = 0
                completion_times.clear()
                snapshot["active"] = False
                snapshot["sync"] = {
                    "ready": True,
                    "episode": int(cmd.get("episode", 0) or 0),
                    "streamContinuous": True,
                    "resultDrops": 0,
                }
                sim_elapsed_s = 0.0
                last_stream_seq = None
                stream_continuous = True
                first_gap_seq = None
                _publish_snapshot(snapshot)
                _publish_frame()
                continue
            if kind != "step":
                continue

            supports_batch = hasattr(mirror, "step_sequence")
            batch_limit = max_batch_steps if supports_batch else 1
            batch_cmds = [dict(cmd)]
            # For MJX, deliberately coalesce one fixed-size scan.  Partial
            # batches are executed as already-compiled one-step calls below,
            # avoiding the old 1..16 family of live JIT shapes.
            collect_deadline = (
                time.monotonic()
                + sim_dt_s * max(0, batch_limit - 1)
                + min(0.002, sim_dt_s * 0.5)
            )
            while len(batch_cmds) < batch_limit:
                timeout_s = max(0.0, collect_deadline - time.monotonic())
                if timeout_s <= 0.0:
                    break
                try:
                    next_cmd = command_q.get(timeout=timeout_s)
                except Empty:
                    break
                except (EOFError, OSError, ValueError):
                    shutdown_requested.set()
                    break
                if not isinstance(next_cmd, dict):
                    continue
                if str(next_cmd.get("kind", "")) != "step":
                    deferred_cmd = next_cmd
                    break
                if (int(next_cmd.get("episode", 0) or 0)
                        != int(batch_cmds[0].get("episode", 0) or 0)
                        or dict(next_cmd.get("options") or {})
                        != dict(batch_cmds[0].get("options") or {})):
                    deferred_cmd = next_cmd
                    break
                batch_cmds.append(dict(next_cmd))
            if shutdown_requested.is_set():
                break

            step_cmd = batch_cmds[-1]
            opts = SimpleNamespace(**dict(batch_cmds[0].get("options") or {}))
            ball_pos_seq = np.asarray(
                [cmd.get("ball_pos", [0.0, 0.0, 0.0]) for cmd in batch_cmds],
                dtype=np.float32,
            )
            ball_vel_seq = np.asarray(
                [cmd.get("ball_vel", [0.0, 0.0, 0.0]) for cmd in batch_cmds],
                dtype=np.float32,
            )
            ball_valid_seq = np.asarray(
                [bool(cmd.get("ball_valid", False)) for cmd in batch_cmds],
                dtype=bool,
            )
            ball_age_seq = np.asarray(
                [float(cmd.get("age_s", 0.0) or 0.0) for cmd in batch_cmds],
                dtype=np.float32,
            )
            batch_samples: list[dict[str, Any]] = []
            if supports_batch and len(batch_cmds) == batch_limit:
                snapshot = mirror.step_sequence(
                    ball_pos_seq,
                    ball_vel_seq,
                    ball_valid_seq=ball_valid_seq,
                    ball_obs_age_seq=ball_age_seq,
                    options=opts,
                )
                raw_batch_samples = snapshot.pop("_batchSamples", None)
                if isinstance(raw_batch_samples, list):
                    batch_samples = [
                        dict(item) for item in raw_batch_samples
                        if isinstance(item, dict)
                    ]
            else:
                snapshot = None
                for idx, cmd in enumerate(batch_cmds):
                    step_snapshot = mirror.step(
                        ball_pos_seq[idx],
                        ball_vel_seq[idx],
                        ball_valid=bool(ball_valid_seq[idx]),
                        ball_obs_age_s=float(ball_age_seq[idx]),
                        options=opts,
                    )
                    raw_batch_samples = step_snapshot.pop("_batchSamples", None)
                    if isinstance(raw_batch_samples, list) and raw_batch_samples:
                        batch_samples.append(dict(raw_batch_samples[-1]))
                    else:
                        batch_samples.append(dict(step_snapshot))
                    snapshot = step_snapshot
                if snapshot is None:
                    continue
            steps = len(batch_cmds)
            sim_elapsed_s += sim_dt_s * float(steps)
            if len(batch_samples) != steps:
                # Old/custom mirrors may only expose the final state.  Keep
                # the telemetry usable, but mark the stream invalid for exact
                # per-tick analysis rather than pretending it is aligned.
                batch_samples = [dict(snapshot)]
                sample_cmds = [batch_cmds[-1]]
                stream_continuous = False
                if first_gap_seq is None:
                    first_gap_seq = int(step_cmd.get("stream_seq", 0) or 0)
            else:
                sample_cmds = batch_cmds

            runtime_config_match = _sim_runtime_config_matches(
                sim_cfg, opts, str(snapshot.get("backend", "")))

            completed_samples: list[dict[str, Any]] = []
            for sample, source_cmd in zip(batch_samples, sample_cmds):
                stream_seq = int(source_cmd.get("stream_seq", 0) or 0)
                if ((last_stream_seq is None and stream_seq != 1)
                        or (last_stream_seq is not None
                            and stream_seq != last_stream_seq + 1)):
                    stream_continuous = False
                    if first_gap_seq is None:
                        first_gap_seq = stream_seq
                last_stream_seq = stream_seq
                sample["active"] = True
                sample["sync"] = {
                    "ready": True,
                    "episode": int(source_cmd.get("episode", 0) or 0),
                    "sourceSeq": int(source_cmd.get("source_seq", 0) or 0),
                    "sourceTimeS": float(source_cmd.get("source_time_s", 0.0) or 0.0),
                    "submittedWallTimeS": float(
                        source_cmd.get("command_wall_s", 0.0) or 0.0),
                    "streamSeq": stream_seq,
                    "ballObservationSeq": int(
                        source_cmd.get("ball_observation_seq", 0) or 0),
                    "cameraStampS": source_cmd.get("camera_stamp_s"),
                    "inputBall": _float32_vector(
                        source_cmd.get("ball_pos", [0.0, 0.0, 0.0])),
                    "inputVelocity": _float32_vector(
                        source_cmd.get("ball_vel", [0.0, 0.0, 0.0])),
                    "inputValid": bool(source_cmd.get("ball_valid", False)),
                    "inputAgeS": float(source_cmd.get("age_s", 0.0) or 0.0),
                    "controlMode": str(source_cmd.get("control_mode", "policy")),
                    "runtimeConfigMatch": bool(runtime_config_match),
                    "streamContinuous": bool(stream_continuous),
                    "firstGapSeq": first_gap_seq,
                    "completedWallTimeS": round(float(time.time()), 6),
                    "resultDrops": int(result_drops),
                }
                completed_samples.append(sample)
                completion_times.append(time.monotonic())

            if completed_samples:
                try:
                    result_q.put({
                        "kind": "sim_results",
                        "samples": completed_samples,
                    }, timeout=max(0.01, sim_dt_s * 2.0))
                except Full:
                    result_drops += len(completed_samples)
                    stream_continuous = False
                except (EOFError, OSError, ValueError):
                    shutdown_requested.set()

            snapshot["active"] = True
            latest_sync = dict(completed_samples[-1].get("sync", {}))
            latest_sync["resultDrops"] = int(result_drops)
            latest_sync["streamContinuous"] = bool(
                stream_continuous and result_drops == 0)
            snapshot["sync"] = latest_sync
            queued = 0
            try:
                queued = max(0, int(command_q.qsize()))
            except (AttributeError, NotImplementedError, OSError):
                queued = 0
            sim_step_hz = 0.0
            if len(completion_times) >= 2:
                span_s = completion_times[-1] - completion_times[0]
                if span_s > 0.0:
                    sim_step_hz = (len(completion_times) - 1) / span_s
            snapshot["simClock"] = {
                "timeS": round(float(sim_elapsed_s), 6),
                "dtS": round(float(sim_dt_s), 6),
                "steps": int(steps),
                "stepHz": round(float(sim_step_hz), 3),
                "targetSeq": int(step_cmd.get("stream_seq", 0) or 0),
                "sourceSeq": int(step_cmd.get("source_seq", 0) or 0),
                "queued": int(queued),
                "queuedS": round(float(queued) * sim_dt_s, 6),
            }
            _publish_snapshot(snapshot)
            _publish_frame()
    except BaseException as exc:  # noqa: BLE001 - surface to parent process.
        if not shutdown_requested.is_set() and not isinstance(exc, KeyboardInterrupt):
            _queue_replace_latest(status_q, {
                "kind": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "snapshot": _sim_snapshot_unavailable(
                    sim_cfg, str(exc), active=False),
            })
    finally:
        if mirror is not None:
            try:
                mirror.close()
            except Exception:
                pass


class SimMirrorSubprocess:
    """Async sim mirror process; step() never waits for MJX/MuJoCo work."""

    def __init__(self, sim_cfg: SimMirrorConfig) -> None:
        self.sim_cfg = sim_cfg
        self.lock = threading.Lock()
        self.ctx = mp.get_context("spawn")
        try:
            command_queue_size = int(float(
                os.environ.get("SIM_COMMAND_QUEUE_SIZE", "256")))
        except (TypeError, ValueError):
            command_queue_size = 256
        command_queue_size = max(8, min(4096, command_queue_size))
        self.command_q = self.ctx.Queue(maxsize=command_queue_size)
        self.status_q = self.ctx.Queue(maxsize=1)
        self.result_q = self.ctx.Queue(maxsize=max(64, command_queue_size))
        self.frame_q = self.ctx.Queue(maxsize=1)
        self.latest_snapshot = _sim_snapshot_unavailable(
            sim_cfg, "sim subprocess starting", active=False)
        self.latest_frame: bytes | None = None
        self.latest_frame_info: dict[str, Any] = {"seq": 0, "ageS": None}
        self.last_error: str | None = None
        self.last_traceback: str | None = None
        self.target_sim_time_s = 0.0
        self.stream_seq = 0
        self.episode = 0
        self.submitted_source_seq: int | None = None
        self.command_drops = 0
        self.stream_valid = True
        self.last_completed_stream_seq = 0
        self.last_completed_source_seq: int | None = None
        self.completed_samples: deque[dict[str, Any]] = deque()
        self._pump_stop = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self.process = self.ctx.Process(
            target=_sim_mirror_process_main,
            args=(
                sim_cfg, self.command_q, self.status_q,
                self.result_q, self.frame_q, os.getpid()),
            name="policy_probe_sim_mirror_process",
            daemon=False,
        )
        self.process.start()
        self._pump_thread = threading.Thread(
            target=self._pump_child_queues,
            name="policy_probe_sim_result_pump",
            daemon=True,
        )
        self._pump_thread.start()

    def _pump_child_queues(self) -> None:
        while not self._pump_stop.is_set():
            try:
                self._drain()
            except (EOFError, OSError, ValueError):
                if not self.process.is_alive():
                    break
            except Exception as exc:  # noqa: BLE001 - surface diagnostics.
                with self.lock:
                    self.last_error = self.last_error or f"sim queue pump failed: {exc}"
            if not self.process.is_alive():
                try:
                    self._drain()
                except Exception:
                    pass
                break
            self._pump_stop.wait(0.002)

    def _drain(self) -> None:
        while True:
            try:
                item = self.status_q.get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", ""))
            with self.lock:
                if kind == "snapshot":
                    snapshot = item.get("snapshot")
                    if isinstance(snapshot, dict):
                        sync = snapshot.get("sync")
                        incoming_episode = (
                            sync.get("episode")
                            if isinstance(sync, dict) else None
                        )
                        if (incoming_episode is None
                                or int(incoming_episode) == self.episode):
                            self.latest_snapshot = dict(snapshot)
                elif kind == "error":
                    self.last_error = str(item.get("error", "unknown sim error"))
                    self.last_traceback = str(item.get("traceback", ""))
                    snapshot = item.get("snapshot")
                    self.latest_snapshot = (
                        dict(snapshot) if isinstance(snapshot, dict)
                        else _sim_snapshot_unavailable(
                            self.sim_cfg, self.last_error, active=False))
        while True:
            try:
                item = self.result_q.get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            if not isinstance(item, dict):
                continue
            if str(item.get("kind", "")) != "sim_results":
                continue
            samples = item.get("samples")
            if not isinstance(samples, list):
                continue
            with self.lock:
                for sample in samples:
                    if isinstance(sample, dict):
                        copied = dict(sample)
                        sync = copied.get("sync")
                        if isinstance(sync, dict):
                            if int(sync.get("episode", -1)) != self.episode:
                                continue
                            self.completed_samples.append(copied)
                            self.last_completed_stream_seq = max(
                                self.last_completed_stream_seq,
                                int(sync.get("streamSeq", 0) or 0),
                            )
                            raw_source_seq = sync.get("sourceSeq")
                            if raw_source_seq is not None:
                                self.last_completed_source_seq = int(raw_source_seq)
        while True:
            try:
                item = self.frame_q.get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            if not isinstance(item, dict):
                continue
            frame = item.get("frame")
            info = item.get("info", {})
            if isinstance(frame, (bytes, bytearray)):
                with self.lock:
                    self.latest_frame = bytes(frame)
                    self.latest_frame_info = (
                        dict(info) if isinstance(info, dict) else {})

    def _snapshot_locked(self) -> dict[str, Any]:
        snapshot = dict(self.latest_snapshot)
        sync = dict(snapshot.get("sync") or {})
        processed_stream_seq = int(sync.get("streamSeq", 0) or 0)
        lag_ticks = max(0, int(self.stream_seq) - processed_stream_seq)
        sync.update({
            "episode": int(self.episode),
            "producedStreamSeq": int(self.stream_seq),
            "producedSourceSeq": self.submitted_source_seq,
            "processedStreamSeq": processed_stream_seq,
            "lagTicks": int(lag_ticks),
            "lagS": round(
                float(lag_ticks)
                * max(1e-6, float(getattr(self.sim_cfg, "control_dt", 0.005))),
                6,
            ),
            "commandDrops": int(self.command_drops),
            "alignmentValid": bool(
                self.stream_valid
                and bool(sync.get("streamContinuous", True))
                and int(sync.get("resultDrops", 0) or 0) == 0
                and bool(sync.get("runtimeConfigMatch", True))
            ),
        })
        snapshot["sync"] = sync
        clock = dict(snapshot.get("simClock") or {})
        clock["lagTicks"] = int(lag_ticks)
        clock["lagS"] = sync["lagS"]
        clock["targetSeq"] = int(self.stream_seq)
        snapshot["simClock"] = clock
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            snapshot = self._snapshot_locked()
        if not self.process.is_alive() and snapshot.get("available") is not True:
            snapshot["error"] = snapshot.get("error") or (
                f"sim subprocess exited (exitcode={self.process.exitcode})")
        return snapshot

    def wait_until_ready(self, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() < deadline:
            with self.lock:
                snapshot = self._snapshot_locked()
                error = self.last_error
            if snapshot.get("available") is True and bool(
                    (snapshot.get("sync") or {}).get("ready", False)):
                return snapshot
            if error is not None:
                raise RuntimeError(f"sim subprocess failed during startup: {error}")
            if not self.process.is_alive():
                raise RuntimeError(
                    "sim subprocess exited during startup "
                    f"(exitcode={self.process.exitcode})")
            time.sleep(0.02)
        raise TimeoutError(
            f"sim subprocess was not READY after {float(timeout_s):.1f}s")

    def reset(self, arm_q: np.ndarray | None = None) -> dict[str, Any]:
        with self.lock:
            self.target_sim_time_s = 0.0
            self.stream_seq = 0
            self.episode += 1
            episode = int(self.episode)
            self.submitted_source_seq = None
            self.command_drops = 0
            self.stream_valid = True
            self.last_completed_stream_seq = 0
            self.last_completed_source_seq = None
            self.completed_samples.clear()
        # A reset starts a new episode, so it must not wait behind stale work
        # that belongs to the previous policy history.
        while True:
            try:
                self.command_q.get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
        reset_cmd = {
            "kind": "reset",
            "episode": episode,
            "arm_q": (
                None if arm_q is None
                else _float32_vector(arm_q)
            ),
        }
        try:
            self.command_q.put(reset_cmd, timeout=1.0)
        except Full:
            self.command_drops += 1
            self.stream_valid = False
        return self.snapshot()

    def step(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        *,
        ball_valid: bool,
        ball_obs_age_s: float,
        options: Any,
        source_seq: int,
        source_time_s: float,
        ball_observation_seq: int,
        camera_stamp_s: float | None,
        control_mode: str,
    ) -> dict[str, Any]:
        self.stream_seq += 1
        self.submitted_source_seq = int(source_seq)
        self.target_sim_time_s += max(
            1e-6, float(getattr(self.sim_cfg, "control_dt", 0.005)))
        command = {
            "kind": "step",
            "episode": int(self.episode),
            "stream_seq": int(self.stream_seq),
            "source_seq": int(source_seq),
            "source_time_s": float(source_time_s),
            "ball_observation_seq": int(ball_observation_seq),
            "camera_stamp_s": camera_stamp_s,
            "control_mode": str(control_mode),
            "target_sim_time_s": float(self.target_sim_time_s),
            "control_dt_s": float(getattr(self.sim_cfg, "control_dt", 0.005)),
            "command_wall_s": round(float(time.time()), 6),
            "ball_pos": _float32_vector(ball_pos),
            "ball_vel": _float32_vector(ball_vel),
            "ball_valid": bool(ball_valid),
            "age_s": float(max(0.0, ball_obs_age_s)),
            "options": _runtime_options_for_sim_process(options),
        }
        try:
            self.command_q.put_nowait(command)
        except Full:
            # Never delete an older observation: one missing input changes all
            # later sim dynamics, so expose the run as invalid instead.
            self.command_drops += 1
            self.stream_valid = False
        return self.snapshot()

    def drain_completed_samples(self) -> list[dict[str, Any]]:
        with self.lock:
            samples = list(self.completed_samples)
            self.completed_samples.clear()
        return samples

    def flush(self, target_stream_seq: int, timeout_s: float) -> dict[str, Any]:
        target = max(0, int(target_stream_seq))
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            with self.lock:
                completed = int(self.last_completed_stream_seq)
                error = self.last_error
            if completed >= target:
                break
            if error is not None or not self.process.is_alive():
                break
            time.sleep(0.005)
        with self.lock:
            completed = int(self.last_completed_stream_seq)
            snapshot = self._snapshot_locked()
        return {
            "targetStreamSeq": target,
            "completedStreamSeq": completed,
            "complete": bool(completed >= target),
            "alignmentValid": bool(
                completed >= target
                and (snapshot.get("sync") or {}).get("alignmentValid", False)
            ),
            "sync": dict(snapshot.get("sync") or {}),
        }

    def latest_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        with self.lock:
            return self.latest_frame, dict(self.latest_frame_info)

    def close(self) -> None:
        try:
            _queue_replace_latest(self.command_q, {"kind": "stop"})
        except Exception:
            pass
        try:
            self.process.join(timeout=8.0)
        except Exception:
            pass
        if self.process.is_alive():
            _terminate_process_tree(self.process, "sim mirror subprocess")
        self._pump_stop.set()
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=1.0)
        for q in (self.command_q, self.status_q, self.result_q, self.frame_q):
            _close_mp_queue(q)


def _expand_path_text(path: str) -> str:
    return os.path.expandvars(os.path.expanduser(str(path)))


def _stamp_to_sec(stamp: Any) -> float:
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return time.time()


WORKSPACE_SAFETY_MODES = (
    "off",
    "position",
    "predictive",
    "position_soft",
    "predictive_soft",
    "impedance_soft",
)
WORKSPACE_SOFT_MODES = frozenset({
    "position_soft", "predictive_soft", "impedance_soft"})
WORKSPACE_HARD_MODES = frozenset({"position", "predictive"})
WORKSPACE_PREDICTIVE_MODES = frozenset({"predictive", "predictive_soft"})


def _workspace_safety_mode(value: Any, default: str = "position_soft") -> str:
    mode = str(value if value is not None else default).strip().lower()
    mode = mode.replace("-", "_").replace(" ", "_")
    if mode in {"off", "none", "disabled", "disable", "false", "0"}:
        return "off"
    if mode in {"pos", "position_only", "position", "position_hard"}:
        return "position"
    if mode in {"predict", "predictive", "prediction", "predictive_hard"}:
        return "predictive"
    if mode in {"pos_soft", "soft", "soft_position", "position_soft"}:
        return "position_soft"
    if mode in {
        "predict_soft", "soft_predictive", "predictive_soft",
        "prediction_soft",
    }:
        return "predictive_soft"
    if mode in {
        "impedance", "soft_impedance", "impedance_soft", "spring_soft",
    }:
        return "impedance_soft"
    return default if default in WORKSPACE_SAFETY_MODES else "position_soft"


def _is_arm_workspace_condition(name: str) -> bool:
    """Return whether a done condition belongs to the arm/EE workspace."""
    key = str(name or "")
    return key.startswith("racket_") or key.startswith("ee_")


def _has_arm_workspace_condition(conditions: dict[str, bool]) -> bool:
    return any(
        bool(active) and _is_arm_workspace_condition(str(name))
        for name, active in conditions.items()
    )


def _is_soft_translation_workspace_condition(name: str) -> bool:
    """Return whether the Cartesian soft filter actually enforces a condition."""
    key = str(name or "")
    if key.startswith("racket_"):
        return any(token in key for token in (
            "too_far_from_anchor", "too_high", "too_low"))
    if not key.startswith("ee_"):
        return False
    return any(
        key.startswith(prefix)
        for prefix in (
            "ee_x_", "ee_y_", "ee_z_", "ee_predicted_x_",
            "ee_predicted_y_", "ee_predicted_z_",
        )
    )


def _hard_done_conditions(
    conditions: dict[str, bool],
    workspace_mode: str,
) -> dict[str, bool]:
    """Keep ball/base done conditions hard while softening arm conditions."""
    soft = _workspace_safety_mode(workspace_mode) in WORKSPACE_SOFT_MODES
    return {
        str(name): bool(active)
        for name, active in conditions.items()
        if bool(active)
        and not (soft and _is_soft_translation_workspace_condition(name))
    }


def _smoothstep01(value: float) -> float:
    """Return a C1-continuous blend from zero to one."""
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _quat_relative_rotvec_xyzw(
    base_xyzw: np.ndarray,
    trial_xyzw: np.ndarray,
) -> np.ndarray | None:
    """Shortest relative orientation as a rotation vector.

    Only the local linearization is needed for the QP objective.  Returning a
    rotation vector rather than subtracting quaternion components avoids the
    q/-q sign ambiguity and gives angular-radian units.
    """
    base = np.asarray(base_xyzw, dtype=np.float64).reshape(-1)
    trial = np.asarray(trial_xyzw, dtype=np.float64).reshape(-1)
    if (base.shape[0] != 4 or trial.shape[0] != 4
            or not np.all(np.isfinite(base))
            or not np.all(np.isfinite(trial))):
        return None
    base_norm = float(np.linalg.norm(base))
    trial_norm = float(np.linalg.norm(trial))
    if base_norm <= 1e-12 or trial_norm <= 1e-12:
        return None
    bx, by, bz, bw = base / base_norm
    tx, ty, tz, tw = trial / trial_norm
    # q_relative = q_trial * conjugate(q_base), in the common/base frame.
    relative = np.array([
        -tw * bx + tx * bw - ty * bz + tz * by,
        -tw * by + tx * bz + ty * bw - tz * bx,
        -tw * bz - tx * by + ty * bx + tz * bw,
        tw * bw + tx * bx + ty * by + tz * bz,
    ], dtype=np.float64)
    if relative[3] < 0.0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm <= 1e-12:
        return 2.0 * relative[:3]
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(relative[3])))
    return relative[:3] * (angle / vector_norm)


def _project_velocity_halfspaces(
    nominal: np.ndarray,
    rows: np.ndarray,
    bounds: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    metric: np.ndarray | None = None,
    tolerance: float = 1e-9,
    max_sweeps: int = 80,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a joint velocity onto Cartesian half-spaces and box limits.

    The problem is the small convex safety-filter QP

        min (dq - dq_nominal)' metric (dq - dq_nominal)
        s.t. rows @ dq <= bounds, lower <= dq <= upper.

    Dykstra's alternating half-space projection is used instead of adding a
    runtime QP dependency.  With seven variables and fewer than two dozen
    constraints it has deterministic, bounded cost in the 200 Hz path.  The
    returned status explicitly reports an infeasible/unfinished projection so
    callers can keep braking rather than silently releasing an unsafe target.
    """
    qdot_nominal = np.asarray(nominal, dtype=np.float64).reshape(-1)
    n_joints = qdot_nominal.shape[0]
    if n_joints == 0 or not np.all(np.isfinite(qdot_nominal)):
        raise ValueError("nominal velocity must be a finite non-empty vector")

    a = np.asarray(rows, dtype=np.float64)
    b = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if a.size == 0:
        a = np.zeros((0, n_joints), dtype=np.float64)
    else:
        a = a.reshape((-1, n_joints))
    if a.shape[0] != b.shape[0]:
        raise ValueError("half-space row/bound count mismatch")

    velocity_scale = (
        np.ones(n_joints, dtype=np.float64)
        if scale is None else np.asarray(scale, dtype=np.float64).reshape(-1))
    if (velocity_scale.shape[0] != n_joints
            or not np.all(np.isfinite(velocity_scale))):
        raise ValueError("scale must be a finite vector matching nominal")
    velocity_scale = np.maximum(np.abs(velocity_scale), 1e-6)

    if metric is None:
        # z = dq / scale, dq = inverse_transform @ z.
        inverse_transform = np.diag(velocity_scale)
        transformed_nominal = qdot_nominal / velocity_scale
    else:
        objective_metric = np.asarray(metric, dtype=np.float64)
        if (objective_metric.shape != (n_joints, n_joints)
                or not np.all(np.isfinite(objective_metric))):
            raise ValueError("metric must be a finite square joint matrix")
        objective_metric = 0.5 * (
            objective_metric + objective_metric.T)
        try:
            # H = L L', z = L' dq, dq = L'^-1 z.
            cholesky = np.linalg.cholesky(objective_metric)
        except np.linalg.LinAlgError as exc:
            raise ValueError("metric must be positive definite") from exc
        inverse_transform = np.linalg.solve(
            cholesky.T, np.eye(n_joints, dtype=np.float64))
        transformed_nominal = cholesky.T @ qdot_nominal

    # Work in Euclidean transformed coordinates.  With a task-space metric this
    # makes the minimum correction Cartesian-normal while a small joint term
    # prevents arbitrary redundant/nullspace motion.
    constraint_rows: list[np.ndarray] = []
    constraint_bounds: list[float] = []
    workspace_constraint_count = 0
    structurally_uncontrollable_violation = 0.0
    for row, bound in zip(a, b):
        transformed_row = (
            np.asarray(row, dtype=np.float64) @ inverse_transform)
        row_norm = float(np.linalg.norm(transformed_row))
        workspace_constraint_count += 1
        if (not np.all(np.isfinite(transformed_row))
                or not math.isfinite(float(bound))):
            structurally_uncontrollable_violation = float("inf")
            continue
        if row_norm <= 1e-12:
            # 0 @ dq <= b is harmless only for b >= 0.  When b < 0 the
            # boundary normal is locally uncontrollable (typically a
            # singularity); dropping it would falsely report a safe QP.
            structurally_uncontrollable_violation = max(
                structurally_uncontrollable_violation,
                max(0.0, -float(bound)),
            )
            continue
        constraint_rows.append(transformed_row / row_norm)
        constraint_bounds.append(float(bound) / row_norm)

    lo = None if lower is None else np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = None if upper is None else np.asarray(upper, dtype=np.float64).reshape(-1)
    if lo is not None:
        if lo.shape[0] != n_joints or not np.all(np.isfinite(lo)):
            raise ValueError("lower velocity bounds must match nominal")
        for idx in range(n_joints):
            row = np.zeros(n_joints, dtype=np.float64)
            row[idx] = -1.0
            transformed_row = row @ inverse_transform
            row_norm = float(np.linalg.norm(transformed_row))
            constraint_rows.append(transformed_row / row_norm)
            constraint_bounds.append(float(-lo[idx]) / row_norm)
    if hi is not None:
        if hi.shape[0] != n_joints or not np.all(np.isfinite(hi)):
            raise ValueError("upper velocity bounds must match nominal")
        for idx in range(n_joints):
            row = np.zeros(n_joints, dtype=np.float64)
            row[idx] = 1.0
            transformed_row = row @ inverse_transform
            row_norm = float(np.linalg.norm(transformed_row))
            constraint_rows.append(transformed_row / row_norm)
            constraint_bounds.append(float(hi[idx]) / row_norm)

    if not constraint_rows:
        feasible = bool(structurally_uncontrollable_violation <= tolerance)
        return qdot_nominal.copy(), {
            "converged": feasible,
            "feasible": feasible,
            "sweeps": 0,
            "maxViolation": float(structurally_uncontrollable_violation),
            "uncontrollable": not feasible,
        }

    c = np.asarray(constraint_rows, dtype=np.float64)
    d = np.asarray(constraint_bounds, dtype=np.float64)
    y_nominal = transformed_nominal
    y = y_nominal.copy()
    corrections = np.zeros_like(c)
    best_y = y.copy()
    best_violation = float("inf")
    converged = False
    sweeps = 0

    for sweep in range(max(1, int(max_sweeps))):
        previous = y.copy()
        for idx in range(c.shape[0]):
            shifted = y + corrections[idx]
            violation = float(np.dot(c[idx], shifted) - d[idx])
            if violation > 0.0:
                projected = shifted - violation * c[idx]
            else:
                projected = shifted
            corrections[idx] = shifted - projected
            y = projected

        violations = c @ y - d
        max_violation = float(max(0.0, np.max(violations)))
        # Box rows are deliberately appended after workspace rows.  A full
        # sweep therefore leaves the candidate dynamically reachable even if
        # the Cartesian constraints have become infeasible too late to brake.
        if max_violation < best_violation:
            best_violation = max_violation
            best_y = y.copy()
        sweeps = sweep + 1
        movement = float(np.max(np.abs(y - previous)))
        if max_violation <= tolerance and movement <= max(tolerance, 1e-12):
            converged = True
            best_y = y.copy()
            best_violation = max_violation
            break

    projected = inverse_transform @ best_y
    if lo is not None:
        projected = np.maximum(projected, lo)
    if hi is not None:
        projected = np.minimum(projected, hi)
    workspace_violation = float(structurally_uncontrollable_violation)
    if workspace_constraint_count:
        workspace_violation = float(max(
            workspace_violation,
            np.max(a @ projected - b) if a.shape[0] else 0.0,
        ))
    feasible = bool(workspace_violation <= max(1e-7, tolerance * 10.0))
    return projected, {
        "converged": bool(converged),
        "feasible": feasible,
        "sweeps": int(sweeps),
        "maxViolation": float(workspace_violation),
        "uncontrollable": bool(
            structurally_uncontrollable_violation > tolerance),
    }


def _payload_float(
    payload: dict[str, Any],
    key: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def _payload_int(
    payload: dict[str, Any],
    key: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    try:
        value = int(round(float(payload.get(key, default))))
    except (TypeError, ValueError):
        value = int(default)
    if min_value is not None:
        value = max(int(min_value), value)
    if max_value is not None:
        value = min(int(max_value), value)
    return value


class RosBridgeNode(Node):
    def __init__(self, cfg: ProbeConfig):
        super().__init__("interactive_policy_safety_probe")
        self.cfg = cfg
        self.lock = threading.Lock()
        self.camera_lock = threading.Lock()
        self.camera_rebuild_lock = threading.Lock()
        self.latest_joint_q: np.ndarray | None = None
        self.latest_joint_dq: np.ndarray | None = None
        self.latest_joint_stamp_wall: float | None = None
        self.latest_joint_dq_source: str = "unavailable"
        self.latest_cart_pos: np.ndarray | None = None
        self.latest_cart_quat_xyzw: np.ndarray | None = None
        self.latest_cart_psi: float | None = None
        self.latest_cart_stamp_wall: float | None = None
        self.publish_count = 0
        self.cmd_pub = None

        self.camera_detector = None
        self.camera_detector_key: tuple[Any, ...] | None = None
        self.camera_direct_running = False
        self.camera_direct_thread: threading.Thread | None = None
        self.camera_info_payload: dict[str, Any] | None = None
        self.camera_info_received = False
        self.camera_output_frame: str | None = None
        self.camera_error: str | None = None
        self.camera_options = CameraRuntimeOptions(
            detect=bool(cfg.enable_camera_detection),
            visualize=bool(cfg.enable_camera_visualization),
            use_depth_sensor=bool(cfg.use_depth_sensor),
            use_undistort=bool(cfg.use_undistort),
            use_kalman=bool(cfg.use_kalman),
            enable_ellipse_fit=bool(cfg.enable_ellipse_fit),
            enable_temporal_tracker=bool(cfg.enable_temporal_tracker),
            enable_color_filter=bool(cfg.enable_camera_color_filter),
        )
        self.camera_frame_count = 0
        self.camera_detection_count = 0
        self.camera_callback_times: deque[float] = deque(maxlen=30)
        self.camera_last_callback_wall: float | None = None
        self.camera_active_transport: str | None = None
        self.camera_last_image_wall_by_transport: dict[str, float] = {}
        self.latest_camera_ball_pos_m: np.ndarray | None = None
        self.latest_camera_ball_vel_m_s: np.ndarray | None = None
        self.latest_camera_detector_pos_m: np.ndarray | None = None
        self.latest_camera_detector_vel_m_s: np.ndarray | None = None
        self.latest_camera_detector_frame: str | None = None
        self.latest_camera_valid = False
        self.latest_camera_stamp_sec: float | None = None
        self.latest_camera_wall: float | None = None
        self.latest_camera_raw_measurement: Any = None
        self.latest_viz_jpeg: bytes | None = None
        self.latest_viz_seq = 0
        self.latest_viz_wall: float | None = None
        self.latest_policy_viz_state: dict[str, Any] = {}
        self.camera_viz_slot_lock = threading.Lock()
        self.camera_viz_event = threading.Event()
        self.camera_viz_pending: tuple[np.ndarray, np.ndarray | None, dict[str, Any], bool] | None = None
        self.camera_viz_thread: threading.Thread | None = None
        self.camera_viz_running = False
        self.camera_viz_last_enqueue_wall = 0.0
        self.camera_viz_min_period_s = (
            1.0 / max(float(cfg.camera_visualization_rate_hz), 1e-6))
        self.control_callback_group = (
            MutuallyExclusiveCallbackGroup()
            if MutuallyExclusiveCallbackGroup is not None else None)
        self.state_callback_group = (
            MutuallyExclusiveCallbackGroup()
            if MutuallyExclusiveCallbackGroup is not None else None)
        self.vision_callback_group = (
            MutuallyExclusiveCallbackGroup()
            if MutuallyExclusiveCallbackGroup is not None else None)
        self.backend_control_timer = None

        if cfg.ros_bridge:
            self.cmd_pub = self.create_publisher(
                MechUnitCmdArray, cfg.command_topic, 10)
            self.create_subscription(
                JointState, cfg.joint_state_topic, self._joint_state_callback, 10,
                callback_group=self.state_callback_group)
            self.create_subscription(
                MechUnitStateArray, cfg.cart_state_topic, self._cart_state_callback, 10,
                callback_group=self.state_callback_group)
            self.get_logger().info(
                f"ROS feedback bridge active: publish {cfg.command_topic}, "
                f"subscribe {cfg.joint_state_topic}, {cfg.cart_state_topic}")

        if cfg.enable_camera_ball:
            self._init_camera_ball_source()

    def _joint_state_callback(self, msg) -> None:
        name_to_pos: dict[str, float] = {}
        name_to_vel: dict[str, float] = {}
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                value = float(msg.position[i])
                if np.isfinite(value):
                    name_to_pos[str(name)] = value
            if i < len(msg.velocity):
                value = float(msg.velocity[i])
                if np.isfinite(value):
                    name_to_vel[str(name)] = value
        try:
            q = np.asarray([name_to_pos[n] for n in RIGHT_ARM_JOINTS], dtype=np.float32)
        except KeyError:
            return
        stamp_wall = time.time()
        dq_vals = [name_to_vel.get(n, np.nan) for n in RIGHT_ARM_JOINTS]
        dq = np.asarray(dq_vals, dtype=np.float32)
        dq_source = "joint_state"
        with self.lock:
            if not np.all(np.isfinite(dq)):
                dq = None
                previous_q = self.latest_joint_q
                previous_stamp = self.latest_joint_stamp_wall
                if previous_q is not None and previous_stamp is not None:
                    sample_dt = float(stamp_wall - previous_stamp)
                    if 1e-4 <= sample_dt <= 0.10:
                        estimated = (
                            q.astype(np.float64)
                            - previous_q.astype(np.float64)) / sample_dt
                        if np.all(np.isfinite(estimated)):
                            # A noisy finite difference may be conservative,
                            # but never let one timestamp glitch create an
                            # unbounded reserve/correction.
                            estimate_limit = 1.25 * np.deg2rad(
                                RightArmCommandSafetyLimiter.VEL_LIMIT_DEG_S)
                            dq = np.clip(
                                estimated, -estimate_limit, estimate_limit,
                            ).astype(np.float32)
                            dq_source = "position_difference"
                if dq is None:
                    dq_source = "unavailable"
            self.latest_joint_q = q
            self.latest_joint_dq = dq
            self.latest_joint_stamp_wall = stamp_wall
            self.latest_joint_dq_source = dq_source

    def _cart_state_callback(self, msg) -> None:
        for state in msg.states:
            if int(state.mu_idx) != int(self.cfg.right_arm_mu_idx):
                continue
            pose = state.end_pose
            pos = np.asarray([
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ], dtype=np.float32)
            quat = np.asarray([
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ], dtype=np.float32)
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
                return
            psi = float(state.psi)
            with self.lock:
                self.latest_cart_pos = pos
                self.latest_cart_quat_xyzw = quat
                self.latest_cart_psi = psi if np.isfinite(psi) else None
                self.latest_cart_stamp_wall = time.time()
            return

    def _init_camera_ball_source(self) -> None:
        if not _CAMERA_BALL_AVAILABLE:
            raise RuntimeError(
                f"camera ball source unavailable: {_CAMERA_BALL_IMPORT_ERROR}")
        source_mode = str(self.cfg.camera_source_mode).strip().lower()
        if source_mode == "direct_realsense":
            self._init_direct_realsense_camera_source()
            return
        if source_mode != "ros_topic":
            raise RuntimeError(f"unsupported camera source mode: {source_mode}")
        if CameraInfo is None or CompressedImage is None or Image is None:
            raise RuntimeError("camera ball source requires sensor_msgs CameraInfo/Image/CompressedImage")
            self.create_subscription(
                CameraInfo, self.cfg.camera_info_topic,
                self._camera_info_callback, 10,
                callback_group=self.vision_callback_group)
        transport = str(self.cfg.camera_image_transport).strip().lower()
        if transport in ("compressed", "auto"):
            self.create_subscription(
                CompressedImage, self.cfg.camera_image_topic,
                self._compressed_image_callback, 10,
                callback_group=self.vision_callback_group)
        if transport in ("raw", "auto"):
            self.create_subscription(
                Image, self.cfg.camera_raw_image_topic,
                self._raw_image_callback, 10,
                callback_group=self.vision_callback_group)
        self.get_logger().info(
            "Camera ball source active: "
            f"transport={transport}, compressed={self.cfg.camera_image_topic}, "
            f"raw={self.cfg.camera_raw_image_topic}, info={self.cfg.camera_info_topic}")

    def _load_direct_realsense_calibration(self):
        if load_calibration is None:
            return None
        calibration_path = self._resolve_calibration_json_path(
            self.cfg.calibration_json_path,
            int(self.cfg.image_width),
            int(self.cfg.image_height))
        if not os.path.exists(calibration_path):
            self.get_logger().warn(
                f"No calibration JSON found for direct RealSense: "
                f"{calibration_path}; output will stay in camera frame")
            return None
        try:
            calibration = load_calibration(calibration_path)
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to load direct RealSense calibration {calibration_path}: "
                f"{exc}; output will stay in camera frame")
            return None
        self.get_logger().info(
            f"Loaded T_base_camera for direct RealSense from: {calibration_path}")
        return calibration

    def _init_direct_realsense_camera_source(self) -> None:
        if not self.cfg.use_hybrid_detector:
            raise RuntimeError(
                "direct_realsense camera source currently requires "
                "--use-hybrid-detector-compatible defaults; use "
                "--camera-source-mode ros_topic for the RGB detector path")
        if HybridBallDetectorAdapter is None:
            raise RuntimeError(
                f"HybridBallDetectorAdapter unavailable: {_CAMERA_BALL_IMPORT_ERROR}")

        calibration = self._load_direct_realsense_calibration()
        if calibration is not None:
            width = int(calibration.image_width)
            height = int(calibration.image_height)
            output_frame = "base_link"
        else:
            width = int(self.cfg.image_width)
            height = int(self.cfg.image_height)
            output_frame = "camera"

        detector = HybridBallDetectorAdapter(
            yolo_model_path=str(self.cfg.yolo_model_path),
            ball_diameter_mm=40.0,
            width=width,
            height=height,
            fps=int(self.cfg.realsense_fps),
            yolo_interval=int(self.cfg.yolo_interval),
            yolo_conf=float(self.cfg.conf_threshold),
            target_class=str(self.cfg.target_class),
            ir_params_json=_expand_path_text(self.cfg.ir_params_json),
            ir_threshold=int(self.cfg.ir_threshold),
            use_adaptive=bool(self.cfg.use_adaptive_ir),
            min_radius=int(self.cfg.ir_min_radius),
            max_radius=int(self.cfg.ir_max_radius),
            circularity_min=float(self.cfg.ir_circularity_min),
            laser_power=int(self.cfg.ir_laser_power),
            exposure=self.cfg.ir_exposure,
            gain=self.cfg.ir_gain,
            brightness_percentile=int(self.cfg.ir_brightness_percentile),
            roi_margin=int(self.cfg.hybrid_roi_margin),
            calibration_data=calibration,
            output_frame=output_frame,
            print_every=999999,
            use_undistort=True,
            depth_scale=0.001,
            depth_correction_model_path=_expand_path_text(
                self.cfg.depth_correction_model_path),
            depth_neighborhood_px=int(self.cfg.depth_neighborhood_px),
            color_match_threshold=0.0,
            color_circularity_min=0.0,
            use_depth_sensor=bool(self.cfg.use_depth_sensor),
        )
        with self.camera_lock:
            self.camera_detector = detector
            self.camera_detector_key = (
                "direct_realsense", "hybrid", width, height,
                int(self.cfg.realsense_fps), output_frame)
            self.camera_info_payload = {
                "width": width,
                "height": height,
                "source": "direct_realsense",
            }
            self.camera_info_received = True
            self.camera_output_frame = output_frame
            self.camera_active_transport = "direct_realsense/hybrid"
            self.camera_error = None

        self.camera_direct_running = True
        self.camera_direct_thread = threading.Thread(
            target=self._direct_hybrid_reader_loop,
            name="policy_probe_direct_realsense",
            daemon=True,
        )
        self.camera_direct_thread.start()
        self.get_logger().info(
            "Camera ball source active: direct_realsense/hybrid "
            f"{width}x{height}@{int(self.cfg.realsense_fps)}fps, "
            f"depth={bool(self.cfg.use_depth_sensor)}, output={output_frame}")

    def _resolve_calibration_json_path(
        self,
        requested_path: str,
        image_width: int,
        image_height: int,
    ) -> str:
        if requested_path and str(requested_path).lower() not in ("auto", ""):
            return _expand_path_text(requested_path)

        search_dirs: list[str] = []
        for base_dir in (SOURCE_MODULE_DIR, SCRIPT_PATH.parents[1]):
            calib_dir = os.path.join(str(base_dir), "outputs", "vision_calib")
            if calib_dir not in search_dirs:
                search_dirs.append(calib_dir)

        candidates: list[str] = []
        for calib_dir in search_dirs:
            candidates.extend(glob.glob(os.path.join(
                calib_dir, "eye_hand_result_*_undistorted.json")))
        candidates = sorted(set(candidates))
        for candidate in candidates:
            try:
                calibration = load_calibration(candidate)
            except Exception:
                continue
            if (int(calibration.image_width) == int(image_width)
                    and int(calibration.image_height) == int(image_height)):
                self.get_logger().info(
                    f"Auto-selected calibration for "
                    f"{image_width}x{image_height}: {candidate}")
                return candidate

        fallback = os.path.join(
            search_dirs[0],
            f"eye_hand_result_{int(image_width)}_undistorted.json")
        self.get_logger().warn(
            f"No calibration JSON matched {image_width}x{image_height}; "
            f"trying {fallback}")
        return fallback

    def _load_extrinsics_for_resolution(self, image_width: int, image_height: int):
        calibration_path = self._resolve_calibration_json_path(
            self.cfg.calibration_json_path,
            int(image_width),
            int(image_height))

        if not os.path.exists(calibration_path):
            self.get_logger().warn(
                f"No calibration JSON found for {image_width}x{image_height}: "
                f"{calibration_path}; output will stay in camera frame")
            return None
        try:
            calibration = load_calibration(calibration_path)
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to load extrinsics calibration {calibration_path}: {exc}; "
                "output will stay in camera frame")
            return None
        if (int(calibration.image_width) != int(image_width)
                or int(calibration.image_height) != int(image_height)):
            self.get_logger().warn(
                f"Calibration JSON resolution {calibration.image_width}x"
                f"{calibration.image_height} does not match CameraInfo "
                f"{image_width}x{image_height}; output will stay in camera frame")
            return None
        self.get_logger().info(
            f"Loaded T_base_camera for camera ball source from: {calibration_path}")
        return calibration

    def _camera_detector_key(
        self,
        info: dict[str, Any],
        options: CameraRuntimeOptions,
    ) -> tuple[Any, ...]:
        return (
            int(info["width"]),
            int(info["height"]),
            round(float(info["fx"]), 6),
            round(float(info["fy"]), 6),
            round(float(info["cx"]), 6),
            round(float(info["cy"]), 6),
            str(self.cfg.calibration_json_path),
            str(self.cfg.yolo_model_path),
            str(self.cfg.target_class),
            round(float(self.cfg.conf_threshold), 6),
            int(self.cfg.infer_imgsz),
            str(self.cfg.detector_device),
            bool(self.cfg.detector_half),
            round(float(self.cfg.camera_tilt_deg), 6),
            int(self.cfg.tracker_max_miss_bridge),
            round(float(self.cfg.tracker_gate_px), 6),
            tuple(self.cfg.hsv_lower),
            tuple(self.cfg.hsv_upper),
            round(float(self.cfg.color_match_threshold), 6),
            round(float(self.cfg.color_circularity_min), 6),
            options.key(),
        )

    def set_camera_runtime_options(self, options: CameraRuntimeOptions) -> None:
        if not self.cfg.enable_camera_ball:
            return
        with self.camera_lock:
            old_key = self.camera_options.key()
            self.camera_options = options
            new_key = options.key()
        if str(self.cfg.camera_source_mode).strip().lower() == "direct_realsense":
            return
        if old_key != new_key:
            self._rebuild_camera_detector("runtime option change")

    def set_policy_viz_state(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        valid: bool,
        racket_anchor_m: np.ndarray,
        racket_pos_m: np.ndarray,
    ) -> None:
        with self.camera_lock:
            self.latest_policy_viz_state = {
                "ball_pos_m": np.asarray(ball_pos_m, dtype=np.float32).reshape(3).copy(),
                "ball_vel_m_s": np.asarray(ball_vel_m_s, dtype=np.float32).reshape(3).copy(),
                "valid": bool(valid),
                "racket_anchor_m": np.asarray(racket_anchor_m, dtype=np.float32).reshape(3).copy(),
                "racket_pos_m": np.asarray(racket_pos_m, dtype=np.float32).reshape(3).copy(),
                "stamp_wall": time.time(),
            }

    def _camera_info_callback(self, msg) -> None:
        if PingPongBallDetector is None:
            return
        fx = float(msg.k[0])
        fy = float(msg.k[4])
        cx = float(msg.k[2])
        cy = float(msg.k[5])
        dist_coeffs = list(msg.d) if len(msg.d) > 0 else None
        width = int(msg.width)
        height = int(msg.height)
        self.get_logger().info(
            "CameraInfo received for interactive probe: "
            f"{width}x{height}, fx={fx:.2f}, fy={fy:.2f}, "
            f"cx={cx:.2f}, cy={cy:.2f}")
        with self.camera_lock:
            if self.camera_info_received:
                return
            self.camera_info_payload = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "dist_coeffs": dist_coeffs,
                "width": width,
                "height": height,
                "distortion_model": str(getattr(msg, "distortion_model", "")),
            }
            self.camera_info_received = True
        self._rebuild_camera_detector("CameraInfo")

    def _rebuild_camera_detector(self, reason: str) -> None:
        if not self.cfg.enable_camera_ball or PingPongBallDetector is None:
            return
        with self.camera_rebuild_lock:
            with self.camera_lock:
                info = None if self.camera_info_payload is None else dict(self.camera_info_payload)
                options = self.camera_options
                current_key = self.camera_detector_key
            if info is None:
                return
            new_key = self._camera_detector_key(info, options)
            if current_key == new_key and self.camera_detector is not None:
                return
            try:
                detector, output_frame = self._build_camera_detector(info, options)
            except Exception as exc:
                with self.camera_lock:
                    self.camera_detector = None
                    self.camera_detector_key = None
                    self.camera_output_frame = None
                    self.camera_error = str(exc)
                self.get_logger().error(f"Failed to initialize camera detector: {exc}")
                raise
            with self.camera_lock:
                self.camera_detector = detector
                self.camera_detector_key = new_key
                self.camera_output_frame = output_frame
                self.camera_error = None
            self.get_logger().info(
                f"Camera ball detector initialized ({reason}, output: {output_frame}, "
                f"options={options.payload()})")

    def _build_camera_detector(
        self,
        info: dict[str, Any],
        options: CameraRuntimeOptions,
    ):
        width = int(info["width"])
        height = int(info["height"])
        extrinsics_calibration = self._load_extrinsics_for_resolution(width, height)
        output_frame = "base_link" if extrinsics_calibration is not None else "camera"
        color_match_threshold = (
            self.cfg.color_match_threshold
            if options.enable_color_filter else 0.0)
        color_circularity_min = (
            self.cfg.color_circularity_min
            if options.enable_color_filter else 0.0)
        detector = PingPongBallDetector(
            yolo_model_path=str(self.cfg.yolo_model_path),
            ball_diameter_mm=40.0,
            width=width,
            height=height,
            fps=60,
            conf_threshold=self.cfg.conf_threshold,
            target_class=self.cfg.target_class,
            camera_tilt_deg=self.cfg.camera_tilt_deg,
            use_kalman=options.use_kalman,
            infer_imgsz=self.cfg.infer_imgsz,
            device=self.cfg.detector_device,
            half=self.cfg.detector_half,
            use_undistort=options.use_undistort,
            print_every=999999,
            profile_every=0,
            use_camera=False,
            camera_fx=float(info["fx"]),
            camera_fy=float(info["fy"]),
            camera_cx=float(info["cx"]),
            camera_cy=float(info["cy"]),
            camera_dist_coeffs=info.get("dist_coeffs"),
            output_frame=output_frame,
            enable_ellipse_fit=options.enable_ellipse_fit,
            enable_temporal_tracker=options.enable_temporal_tracker,
            tracker_max_miss_bridge=self.cfg.tracker_max_miss_bridge,
            tracker_gate_px=self.cfg.tracker_gate_px,
            hsv_lower=list(self.cfg.hsv_lower),
            hsv_upper=list(self.cfg.hsv_upper),
            color_match_threshold=color_match_threshold,
            color_circularity_min=color_circularity_min,
        )
        if extrinsics_calibration is not None:
            detector.calibration_data = extrinsics_calibration
            detector.T_base_camera = extrinsics_calibration.T_base_camera
            detector.R_base_camera = detector.T_base_camera[:3, :3]
            detector.output_frame = "base_link"
            output_frame = "base_link"
        return detector, output_frame

    def _should_process_transport(self, transport: str) -> bool:
        now = time.time()
        with self.camera_lock:
            self.camera_last_image_wall_by_transport[transport] = now
            active = self.camera_active_transport
            if active is None:
                self.camera_active_transport = transport
                return True
            if active == transport:
                return True
            active_age = now - self.camera_last_image_wall_by_transport.get(active, 0.0)
            if active_age > 1.0:
                self.camera_active_transport = transport
                return True
            return False

    def _compressed_image_callback(self, msg) -> None:
        if not self._should_process_transport("compressed"):
            return
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                self.get_logger().warn(
                    "Failed to decode compressed camera image",
                    throttle_duration_sec=1.0)
                return
            stamp_sec = _stamp_to_sec(msg.header.stamp)
            self._handle_cv_image(image, stamp_sec)
        except Exception as exc:
            with self.camera_lock:
                self.camera_error = str(exc)
            self.get_logger().error(
                f"Error in compressed camera image callback: {exc}",
                throttle_duration_sec=1.0)

    def _raw_image_callback(self, msg) -> None:
        if not self._should_process_transport("raw"):
            return
        try:
            image = self._decode_raw_image_msg(msg)
            stamp_sec = _stamp_to_sec(msg.header.stamp)
            self._handle_cv_image(image, stamp_sec)
        except Exception as exc:
            with self.camera_lock:
                self.camera_error = str(exc)
            self.get_logger().error(
                f"Error in raw camera image callback: {exc}",
                throttle_duration_sec=1.0)

    def _decode_raw_image_msg(self, msg) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        encoding = str(msg.encoding).lower()
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid raw image size {width}x{height}")
        channels_by_encoding = {
            "bgr8": 3,
            "rgb8": 3,
            "bgra8": 4,
            "rgba8": 4,
            "mono8": 1,
            "8uc1": 1,
            "8uc3": 3,
            "8uc4": 4,
        }
        channels = channels_by_encoding.get(encoding)
        if channels is None:
            raise ValueError(f"unsupported raw image encoding: {msg.encoding}")
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        step = int(msg.step)
        min_step = width * channels
        if step < min_step:
            raise ValueError(
                f"invalid raw image step {step} for {width}x{height} {encoding}")
        rows = raw.reshape((height, step))[:, :min_step]
        if channels == 1:
            image = rows.reshape((height, width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = rows.reshape((height, width, channels))
        if encoding in ("rgb8", "8uc3"):
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image.copy()

    def _direct_hybrid_reader_loop(self) -> None:
        self.get_logger().info("Direct RealSense hybrid reader loop: entering")
        with self.camera_lock:
            detector = self.camera_detector
        pipeline = getattr(getattr(detector, "detector", None), "pipeline", None)
        if pipeline is None:
            with self.camera_lock:
                self.camera_error = "direct RealSense hybrid pipeline is unavailable"
            return

        while self.camera_direct_running:
            try:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError as exc:
                    with self.camera_lock:
                        self.camera_error = f"wait_for_frames timeout: {exc}"
                    self.get_logger().warn(
                        f"wait_for_frames timeout: {exc}",
                        throttle_duration_sec=1.0)
                    continue

                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame(1)
                if not color_frame:
                    continue
                rgb_image = np.asanyarray(color_frame.get_data())
                if rgb_image is None or rgb_image.size == 0:
                    continue
                ir_image = (
                    np.asanyarray(ir_frame.get_data())
                    if ir_frame is not None else None)
                stamp = self.get_clock().now().to_msg()
                stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9

                with self.camera_lock:
                    options = self.camera_options
                    detector = self.camera_detector
                    self.camera_last_image_wall_by_transport[
                        "direct_realsense"] = time.time()
                    self.camera_active_transport = "direct_realsense/hybrid"

                if not options.detect:
                    observation = {
                        "detected": False,
                        "position": [0, 0, 0],
                        "velocity": [0, 0, 0],
                        "raw_measurement": {"detection_disabled": True},
                    }
                elif detector is None:
                    observation = {
                        "detected": False,
                        "position": [0, 0, 0],
                        "velocity": [0, 0, 0],
                        "raw_measurement": {"no_detector": True},
                    }
                elif not options.use_depth_sensor:
                    if ir_image is None:
                        observation = {
                            "detected": False,
                            "position": [0, 0, 0],
                            "velocity": [0, 0, 0],
                            "raw_measurement": {"missing_ir_frame": True},
                        }
                    else:
                        observation = detector.detector.get_observation_offline(
                            rgb_image, ir_image, None, stamp_sec)
                else:
                    observation = detector.detector.get_observation(
                        observation_time=stamp_sec,
                        frames=frames,
                    )
                    if observation is None:
                        observation = {
                            "detected": False,
                            "position": [0, 0, 0],
                            "velocity": [0, 0, 0],
                            "raw_measurement": {"no_observation": True},
                        }

                self._handle_direct_hybrid_observation(
                    rgb_image, ir_image, observation, stamp_sec)
            except Exception as exc:  # noqa: BLE001 - keep reader alive.
                with self.camera_lock:
                    self.camera_error = str(exc)
                self.get_logger().error(
                    f"Direct RealSense hybrid reader error: {exc}",
                    throttle_duration_sec=1.0)
                time.sleep(0.1)
        self.get_logger().info("Direct RealSense hybrid reader loop: exited")

    def _handle_direct_hybrid_observation(
        self,
        image: np.ndarray,
        ir_image: np.ndarray | None,
        observation: dict[str, Any],
        stamp_sec: float,
    ) -> None:
        now_wall = time.time()
        with self.camera_lock:
            if self.camera_last_callback_wall is not None:
                dt = now_wall - self.camera_last_callback_wall
                if dt > 0.0:
                    self.camera_callback_times.append(dt)
            self.camera_last_callback_wall = now_wall
            self.camera_frame_count += 1
            if bool(observation.get("detected", False)) and not bool(
                    observation.get("predicted", False)):
                self.camera_detection_count += 1
            options = self.camera_options
            detector = self.camera_detector

        self._store_camera_observation(observation, stamp_sec)
        if not options.visualize:
            return
        if ir_image is not None and detector is not None:
            try:
                viz_image = detector.detector._render_visualization(
                    image, ir_image, observation)
                if viz_image is not None:
                    hybrid_obs_for_viz = dict(observation)
                    raw = hybrid_obs_for_viz.get("raw_measurement")
                    if isinstance(raw, dict):
                        hybrid_obs_for_viz["raw_measurement"] = dict(raw)
                    hybrid_obs_for_viz["_hybrid_viz_rendered"] = True
                    self._enqueue_camera_visualization(
                        viz_image, hybrid_obs_for_viz)
                    return
            except Exception as exc:  # noqa: BLE001 - fallback below.
                with self.camera_lock:
                    self.camera_error = f"hybrid visualization failed: {exc}"
        self._enqueue_camera_visualization(image, observation)

    def _handle_cv_image(self, image: np.ndarray, stamp_sec: float) -> None:
        now_wall = time.time()
        with self.camera_lock:
            if self.camera_last_callback_wall is not None:
                dt = now_wall - self.camera_last_callback_wall
                if dt > 0.0:
                    self.camera_callback_times.append(dt)
            self.camera_last_callback_wall = now_wall
            options = self.camera_options
            detector = self.camera_detector
            detector_ready = bool(self.camera_info_received and detector is not None)

        if not options.detect:
            observation = {
                "detected": False,
                "raw_measurement": {"detection_disabled": True},
            }
        elif not detector_ready:
            observation = {
                "detected": False,
                "raw_measurement": {"waiting_for_camera_info": not self.camera_info_received},
            }
        else:
            observation = detector.get_observation(
                image, observation_time=stamp_sec)

        detected = bool(observation.get("detected", False))
        is_predicted = bool(observation.get("predicted", False))
        with self.camera_lock:
            self.camera_frame_count += 1
            if detected and not is_predicted:
                self.camera_detection_count += 1

        self._store_camera_observation(observation, stamp_sec)
        if options.visualize:
            self._enqueue_camera_visualization(image, observation)

    def _transform_ball_pos_to_sim_frame(self, ball_pos_m: np.ndarray) -> np.ndarray:
        transformed = np.asarray(ball_pos_m, dtype=np.float32).reshape(3).copy()
        transformed[0] += float(self.cfg.sim_cartesian_x_offset_m)
        transformed[1] += float(self.cfg.sim_cartesian_y_offset_m)
        transformed[2] += float(self.cfg.sim_base_height_m)
        return transformed

    def _store_camera_observation(self, observation: dict[str, Any], stamp_sec: float) -> None:
        now_wall = time.time()
        detected = bool(observation.get("detected", False))
        pos_m = None
        vel_m_s = None
        detector_pos_m = None
        detector_vel_m_s = None
        with self.camera_lock:
            detector_frame = self.camera_output_frame or "camera"
        if detected:
            try:
                detector_pos_m = np.asarray(
                    observation.get("position", [0.0, 0.0, 0.0]),
                    dtype=np.float32).reshape(3) * 1e-3
                detector_vel_m_s = np.asarray(
                    observation.get("velocity", [0.0, 0.0, 0.0]),
                    dtype=np.float32).reshape(3) * 1e-3
                if detector_frame == "sim_frame":
                    pos_m = detector_pos_m.astype(np.float32)
                else:
                    pos_m = self._transform_ball_pos_to_sim_frame(detector_pos_m)
                vel_m_s = detector_vel_m_s.astype(np.float32)
                if not (np.all(np.isfinite(detector_pos_m))
                        and np.all(np.isfinite(detector_vel_m_s))
                        and np.all(np.isfinite(pos_m))
                        and np.all(np.isfinite(vel_m_s))):
                    detected = False
                    pos_m = None
                    vel_m_s = None
                    detector_pos_m = None
                    detector_vel_m_s = None
            except Exception:
                detected = False
                pos_m = None
                vel_m_s = None
                detector_pos_m = None
                detector_vel_m_s = None
        with self.camera_lock:
            self.latest_camera_ball_pos_m = pos_m
            self.latest_camera_ball_vel_m_s = vel_m_s
            self.latest_camera_detector_pos_m = detector_pos_m
            self.latest_camera_detector_vel_m_s = detector_vel_m_s
            self.latest_camera_detector_frame = detector_frame
            self.latest_camera_valid = bool(detected)
            self.latest_camera_stamp_sec = float(stamp_sec)
            self.latest_camera_wall = now_wall
            self.latest_camera_raw_measurement = observation.get("raw_measurement")
            self.camera_error = None

    def _viz_done_thresholds(self) -> dict[str, float]:
        return {
            "ball_z_floor": float(self.cfg.done_z_min_m),
            "ball_z_ceiling": float(self.cfg.done_z_max_m),
            "ball_x_range": float(self.cfg.done_x_range_m),
            "ball_y_range": float(self.cfg.done_y_range_m),
            "racket_anchor_distance": float(self.cfg.done_racket_anchor_distance_m),
            "racket_z_high_offset": float(self.cfg.done_racket_z_high_offset_m),
            "racket_z_low_offset": float(self.cfg.done_racket_z_low_offset_m),
        }

    def _compute_viz_sim_ball_state(self, observation: dict[str, Any]):
        with self.camera_lock:
            state = dict(self.latest_policy_viz_state)
        if state:
            return (
                np.asarray(state["ball_pos_m"], dtype=np.float64).reshape(3),
                np.asarray(state["ball_vel_m_s"], dtype=np.float64).reshape(3),
                bool(state.get("valid", False)),
                "[SIM control state]",
            )
        if not bool(observation.get("detected", False)):
            return None, None, False, "[SIM raw frame]"
        pos = observation.get("position")
        if pos is None:
            return None, None, False, "[SIM raw frame]"
        vel = observation.get("velocity")
        if vel is None:
            vel = [0.0, 0.0, 0.0]
        try:
            pos_m = np.asarray(pos, dtype=np.float64).reshape(3) * 1e-3
            vel_m_s = np.asarray(vel, dtype=np.float64).reshape(3) * 1e-3
        except (TypeError, ValueError):
            return None, None, False, "[SIM raw frame]"
        return self._transform_ball_pos_to_sim_frame(pos_m), vel_m_s, True, "[SIM raw frame]"

    def _normalize_camera_observation_for_overlay(
        self,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Map hybrid detector fields onto the lightweight web overlay schema."""
        obs = dict(observation or {})
        raw_in = obs.get("raw_measurement")
        raw = dict(raw_in) if isinstance(raw_in, dict) else {}
        if raw:
            if raw.get("image_x") is None or raw.get("image_y") is None:
                px = raw.get("cx_rgb")
                py = raw.get("cy_rgb")
                if px is None or py is None:
                    px = raw.get("cx")
                    py = raw.get("cy")
                if px is not None and py is not None:
                    raw["image_x"] = px
                    raw["image_y"] = py
            if raw.get("radius_px") is None and raw.get("radius") is not None:
                raw["radius_px"] = raw.get("radius")
            if raw.get("color_bbox") is None and raw.get("yolo_bbox") is not None:
                try:
                    x1, y1, x2, y2 = raw.get("yolo_bbox")
                    raw["color_bbox"] = [x1, y1, x2 - x1, y2 - y1]
                except Exception:
                    pass
            raw.setdefault("detection_method", raw.get("pixel_source", "hybrid"))
            obs["raw_measurement"] = raw
        return obs

    def _render_visualization_frame(
        self,
        frame: np.ndarray,
        observation: dict[str, Any],
    ) -> np.ndarray:
        observation = self._normalize_camera_observation_for_overlay(observation)
        sim_ball_pos_m, sim_ball_vel_m_s, sim_detected, sim_label = (
            self._compute_viz_sim_ball_state(observation))
        with self.camera_lock:
            frame_count = int(self.camera_frame_count)
            detection_count = int(self.camera_detection_count)
            callback_times = list(self.camera_callback_times)
            output_frame = self.camera_output_frame or "camera"
            state = dict(self.latest_policy_viz_state)
        if callback_times:
            avg_dt = sum(callback_times) / len(callback_times)
            callback_hz = 1.0 / avg_dt if avg_dt > 0 else 0.0
        else:
            callback_hz = 0.0
        racket_anchor_m = state.get("racket_anchor_m")
        racket_pos_m = state.get("racket_pos_m")
        if observation.get("_hybrid_viz_rendered", False):
            if (draw_sim_done_overlay is not None
                    and sim_ball_pos_m is not None
                    and sim_ball_vel_m_s is not None):
                draw_sim_done_overlay(
                    frame,
                    sim_ball_pos_m,
                    sim_ball_vel_m_s,
                    racket_anchor_m,
                    racket_pos_m=racket_pos_m,
                    thresholds=self._viz_done_thresholds(),
                    detected=sim_detected,
                    label=sim_label,
                )
            return frame
        if render_detection_overlay is None:
            return frame
        return render_detection_overlay(
            frame,
            observation,
            frame_count=frame_count,
            detection_count=detection_count,
            callback_hz=callback_hz,
            output_frame=output_frame,
            sim_ball_pos_m=sim_ball_pos_m,
            sim_ball_vel_m_s=sim_ball_vel_m_s,
            racket_anchor_m=racket_anchor_m,
            racket_pos_m=racket_pos_m,
            done_thresholds=self._viz_done_thresholds(),
            sim_detected=sim_detected,
            sim_label=sim_label,
        )

    def _start_camera_visualization_worker(self) -> None:
        if self.camera_viz_thread is not None and self.camera_viz_thread.is_alive():
            return
        self.camera_viz_running = True
        self.camera_viz_thread = threading.Thread(
            target=self._camera_visualization_worker_loop,
            name="policy_probe_camera_viz",
            daemon=True,
        )
        self.camera_viz_thread.start()

    def _enqueue_camera_visualization(
        self,
        image: np.ndarray,
        observation: dict[str, Any],
        ir_image: np.ndarray | None = None,
        hybrid: bool = False,
    ) -> None:
        if cv2 is None or image is None or image.size == 0:
            return
        self._start_camera_visualization_worker()
        now = time.time()
        with self.camera_viz_slot_lock:
            if self.camera_viz_pending is not None:
                return
            min_period = max(0.0, self.camera_viz_min_period_s - 0.001)
            if now - self.camera_viz_last_enqueue_wall < min_period:
                return
        frame_copy = image.copy()
        ir_copy = None if ir_image is None else ir_image.copy()
        obs_copy = dict(observation)
        raw = obs_copy.get("raw_measurement")
        if isinstance(raw, dict):
            obs_copy["raw_measurement"] = dict(raw)
        with self.camera_viz_slot_lock:
            if self.camera_viz_pending is not None:
                return
            self.camera_viz_pending = (frame_copy, ir_copy, obs_copy, bool(hybrid))
            self.camera_viz_last_enqueue_wall = now
        self.camera_viz_event.set()

    def _camera_visualization_worker_loop(self) -> None:
        while self.camera_viz_running:
            if not self.camera_viz_event.wait(timeout=0.5):
                continue
            self.camera_viz_event.clear()
            with self.camera_viz_slot_lock:
                pending = self.camera_viz_pending
                self.camera_viz_pending = None
            if pending is None:
                continue
            image, ir_image, observation, hybrid = pending
            with self.camera_lock:
                options = self.camera_options
                detector = self.camera_detector
            if not options.visualize:
                continue

            if hybrid and ir_image is not None and detector is not None:
                try:
                    viz_image = detector.detector._render_visualization(
                        image, ir_image, observation)
                    if viz_image is not None:
                        hybrid_obs_for_viz = dict(observation)
                        raw = hybrid_obs_for_viz.get("raw_measurement")
                        if isinstance(raw, dict):
                            hybrid_obs_for_viz["raw_measurement"] = dict(raw)
                        hybrid_obs_for_viz["_hybrid_viz_rendered"] = True
                        self._store_camera_visualization(
                            viz_image, hybrid_obs_for_viz)
                        continue
                except Exception as exc:  # noqa: BLE001 - fallback below.
                    with self.camera_lock:
                        self.camera_error = f"hybrid visualization failed: {exc}"

            self._store_camera_visualization(image, observation)

    def _store_camera_visualization(
        self,
        image: np.ndarray,
        observation: dict[str, Any],
    ) -> None:
        if cv2 is None:
            return
        overlay = image
        try:
            overlay = self._render_visualization_frame(image, observation)
        except Exception as exc:  # noqa: BLE001 - viz must not stop the stream.
            with self.camera_lock:
                self.camera_error = f"visualization overlay failed: {exc}"
            overlay = image
        try:
            max_width = int(self.cfg.camera_web_max_width)
            if max_width > 0 and overlay.shape[1] > max_width:
                scale = float(max_width) / float(overlay.shape[1])
                overlay = cv2.resize(
                    overlay,
                    (max_width, max(1, int(round(overlay.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            quality = int(min(95, max(25, self.cfg.camera_web_jpeg_quality)))
            ok, encoded = cv2.imencode(
                ".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if not ok:
                return
            with self.camera_lock:
                self.latest_viz_jpeg = encoded.tobytes()
                self.latest_viz_seq += 1
                self.latest_viz_wall = time.time()
        except Exception as exc:  # noqa: BLE001 - viz must not affect control.
            with self.camera_lock:
                self.camera_error = f"visualization encode failed: {exc}"

    def latest_camera_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        now = time.time()
        with self.camera_lock:
            age = (
                None if self.latest_viz_wall is None
                else max(0.0, now - self.latest_viz_wall))
            info = {
                "seq": int(self.latest_viz_seq),
                "ageS": None if age is None else round(float(age), 6),
                "enabled": bool(self.camera_options.visualize),
            }
            frame = self.latest_viz_jpeg
        return frame, info

    def camera_frame_rotation(self) -> np.ndarray | None:
        """Return R_base_camera for camera-frame covariance rotation."""
        with self.camera_lock:
            detector = self.camera_detector
            output_frame = self.camera_output_frame
        if detector is None or output_frame not in ("base_link", "sim_frame"):
            return None
        rotation = getattr(detector, "R_base_camera", None)
        if rotation is None:
            transform = getattr(detector, "T_base_camera", None)
            if transform is not None:
                T = np.asarray(transform, dtype=np.float64)
                if T.shape == (4, 4) and np.all(np.isfinite(T)):
                    rotation = T[:3, :3]
        if rotation is None:
            return None
        R = np.asarray(rotation, dtype=np.float64)
        if R.shape != (3, 3) or not np.all(np.isfinite(R)):
            return None
        return R.copy()

    def camera_ball_state(self) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        now = time.time()
        with self.camera_lock:
            pos = (
                None if self.latest_camera_ball_pos_m is None
                else self.latest_camera_ball_pos_m.copy())
            vel = (
                None if self.latest_camera_ball_vel_m_s is None
                else self.latest_camera_ball_vel_m_s.copy())
            detector_pos = (
                None if self.latest_camera_detector_pos_m is None
                else self.latest_camera_detector_pos_m.copy())
            detector_vel = (
                None if self.latest_camera_detector_vel_m_s is None
                else self.latest_camera_detector_vel_m_s.copy())
            age = (
                None if self.latest_camera_wall is None
                else max(0.0, now - self.latest_camera_wall))
            viz_age = (
                None if self.latest_viz_wall is None
                else max(0.0, now - self.latest_viz_wall))
            image_ages = {
                key: round(float(max(0.0, now - wall)), 6)
                for key, wall in self.camera_last_image_wall_by_transport.items()
            }
            feedback = {
                "enabled": bool(self.cfg.enable_camera_ball),
                "available": bool(_CAMERA_BALL_AVAILABLE),
                "ready": bool(self.camera_info_received and self.camera_detector is not None),
                "sourceMode": self.cfg.camera_source_mode,
                "useHybridDetector": bool(self.cfg.use_hybrid_detector),
                "useDepthSensor": bool(self.cfg.use_depth_sensor),
                "imageWidth": int(self.cfg.image_width),
                "imageHeight": int(self.cfg.image_height),
                "realsenseFps": int(self.cfg.realsense_fps),
                "cameraInfoReceived": bool(self.camera_info_received),
                "detectorReady": bool(self.camera_detector is not None),
                "detected": bool(self.latest_camera_valid),
                "stampS": (
                    None if self.latest_camera_stamp_sec is None
                    else round(float(self.latest_camera_stamp_sec), 6)
                ),
                "ageS": None if age is None else round(float(age), 6),
                "frame": self.camera_output_frame,
                "detectorFrame": self.latest_camera_detector_frame,
                "controlFrame": "sim_frame",
                "detectorBall": (
                    None if detector_pos is None
                    else [round(float(v), 6) for v in detector_pos]),
                "detectorVelocity": (
                    None if detector_vel is None
                    else [round(float(v), 6) for v in detector_vel]),
                "controlBall": (
                    None if pos is None
                    else [round(float(v), 6) for v in pos]),
                "controlVelocity": (
                    None if vel is None
                    else [round(float(v), 6) for v in vel]),
                "imageTopic": self.cfg.camera_image_topic,
                "rawImageTopic": self.cfg.camera_raw_image_topic,
                "imageTransport": self.cfg.camera_image_transport,
                "activeTransport": self.camera_active_transport,
                "imageAgesS": image_ages,
                "cameraInfoTopic": self.cfg.camera_info_topic,
                "frameCount": int(self.camera_frame_count),
                "detectionCount": int(self.camera_detection_count),
                "callbackHz": (
                    round(float(1.0 / (sum(self.camera_callback_times) / len(self.camera_callback_times))), 3)
                    if len(self.camera_callback_times) > 0
                    and sum(self.camera_callback_times) > 0.0 else 0.0
                ),
                "visualizationRateHz": round(
                    float(self.cfg.camera_visualization_rate_hz), 3),
                "webMaxWidth": int(self.cfg.camera_web_max_width),
                "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
                "visualizationSeq": int(self.latest_viz_seq),
                "visualizationAgeS": None if viz_age is None else round(float(viz_age), 6),
                "runtime": self.camera_options.payload(),
                "error": self.camera_error,
            }
            valid = bool(self.latest_camera_valid and pos is not None and vel is not None)
        if not valid:
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                feedback,
            )
        return pos, vel, True, feedback

    def camera_ball_measurement(
        self,
    ) -> tuple[np.ndarray, np.ndarray, bool, float | None]:
        with self.camera_lock:
            pos = (
                None if self.latest_camera_ball_pos_m is None
                else self.latest_camera_ball_pos_m.copy())
            vel = (
                None if self.latest_camera_ball_vel_m_s is None
                else self.latest_camera_ball_vel_m_s.copy())
            valid = bool(
                self.latest_camera_valid and pos is not None and vel is not None)
            stamp_sec = self.latest_camera_stamp_sec
        if not valid:
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                stamp_sec,
            )
        return pos, vel, True, stamp_sec

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def get_joint_feedback(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self.lock:
            q = None if self.latest_joint_q is None else self.latest_joint_q.copy()
            dq = None if self.latest_joint_dq is None else self.latest_joint_dq.copy()
        return q, dq

    def get_joint_feedback_with_age(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None, str]:
        now = time.time()
        with self.lock:
            q = None if self.latest_joint_q is None else self.latest_joint_q.copy()
            dq = None if self.latest_joint_dq is None else self.latest_joint_dq.copy()
            age_s = (
                None if self.latest_joint_stamp_wall is None
                else max(0.0, now - self.latest_joint_stamp_wall))
            dq_source = str(self.latest_joint_dq_source)
        return q, dq, age_s, dq_source

    def get_cart_feedback(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        with self.lock:
            pos = None if self.latest_cart_pos is None else self.latest_cart_pos.copy()
            quat = (
                None if self.latest_cart_quat_xyzw is None
                else self.latest_cart_quat_xyzw.copy())
            psi = self.latest_cart_psi
        return pos, quat, psi

    def feedback_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            q = None if self.latest_joint_q is None else self.latest_joint_q.copy()
            dq = None if self.latest_joint_dq is None else self.latest_joint_dq.copy()
            q_age = (
                None if self.latest_joint_stamp_wall is None
                else max(0.0, now - self.latest_joint_stamp_wall)
            )
            cart = None if self.latest_cart_pos is None else self.latest_cart_pos.copy()
            quat = (
                None if self.latest_cart_quat_xyzw is None
                else self.latest_cart_quat_xyzw.copy()
            )
            cart_age = (
                None if self.latest_cart_stamp_wall is None
                else max(0.0, now - self.latest_cart_stamp_wall)
            )
            psi = self.latest_cart_psi
            publish_count = int(self.publish_count)
        return {
            "jointPosition": None if q is None else [round(float(v), 6) for v in q],
            "jointVelocity": None if dq is None else [round(float(v), 6) for v in dq],
            "jointAgeS": None if q_age is None else round(float(q_age), 6),
            "jointVelocitySource": str(self.latest_joint_dq_source),
            "cartPosition": None if cart is None else [round(float(v), 6) for v in cart],
            "cartQuatXyzw": None if quat is None else [round(float(v), 6) for v in quat],
            "cartPsi": None if psi is None else round(float(psi), 6),
            "cartAgeS": None if cart_age is None else round(float(cart_age), 6),
            "publishCount": publish_count,
        }

    def publish_right_arm_command(self, q: np.ndarray) -> None:
        if self.cmd_pub is None:
            return
        msg = MechUnitCmdArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""
        cmd = MechUnitCmd()
        cmd.mu_idx = int(self.cfg.right_arm_mu_idx)
        cmd.jnt_pos = [float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)]
        cmd.use_jnt = True
        cmd.psi = 0.0
        msg.cmds = [cmd]
        self.cmd_pub.publish(msg)
        with self.lock:
            self.publish_count += 1

    def start_backend_control_timer(self, period_s: float, callback) -> None:
        if self.backend_control_timer is not None:
            self.backend_control_timer.cancel()
        self.backend_control_timer = self.create_timer(
            max(float(period_s), 1e-6),
            callback,
            callback_group=self.control_callback_group,
        )
        self.get_logger().info(
            f"Interactive backend control timer started at "
            f"{1.0 / max(float(period_s), 1e-6):.1f} Hz")

    def stop_backend_control_timer(self) -> None:
        timer = self.backend_control_timer
        self.backend_control_timer = None
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass

    def _close_camera_detector(self, detector: Any) -> None:
        if detector is None:
            return
        try:
            close = getattr(detector, "close", None)
            if callable(close):
                close()
                return
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Error closing camera detector: {exc}")
        for candidate in (
                detector,
                getattr(detector, "detector", None),
                getattr(getattr(detector, "detector", None), "pipeline", None),
                getattr(detector, "pipeline", None)):
            if candidate is None:
                continue
            pipeline = (
                candidate if hasattr(candidate, "stop")
                else getattr(candidate, "pipeline", None))
            stop = getattr(pipeline, "stop", None)
            if not callable(stop):
                continue
            try:
                stop()
            except Exception:
                pass

    def shutdown_camera_source(self) -> None:
        self.stop_backend_control_timer()
        self.camera_direct_running = False
        with self.camera_lock:
            detector = self.camera_detector
        self._close_camera_detector(detector)
        thread = self.camera_direct_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self.camera_viz_running = False
        self.camera_viz_event.set()
        viz_thread = self.camera_viz_thread
        if viz_thread is not None and viz_thread.is_alive():
            viz_thread.join(timeout=1.0)
        with self.camera_lock:
            detector = self.camera_detector
            self.camera_detector = None
            self.camera_detector_key = None
            self.camera_direct_thread = None
            self.camera_viz_thread = None
        self._close_camera_detector(detector)


class RosBridge:
    def __init__(self, cfg: ProbeConfig):
        if not _ROS_AVAILABLE:
            raise RuntimeError(f"ROS2 Python bridge unavailable: {_ROS_IMPORT_ERROR}")
        if rclpy is None or MultiThreadedExecutor is None:
            raise RuntimeError("ROS2 Python bridge unavailable")
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = RosBridgeNode(cfg)
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(
            target=self._spin,
            name="policy_probe_ros_spin",
            daemon=True,
        )
        self.thread.start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except Exception:
            pass

    def get_joint_feedback(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.node.get_joint_feedback()

    def get_joint_feedback_with_age(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None, str]:
        return self.node.get_joint_feedback_with_age()

    def get_cart_feedback(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        return self.node.get_cart_feedback()

    def feedback_snapshot(self) -> dict[str, Any]:
        return self.node.feedback_snapshot()

    def camera_ball_state(self) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        return self.node.camera_ball_state()

    def camera_ball_measurement(
        self,
    ) -> tuple[np.ndarray, np.ndarray, bool, float | None]:
        return self.node.camera_ball_measurement()

    def now_sec(self) -> float:
        return self.node.now_sec()

    def set_camera_runtime_options(self, options: CameraRuntimeOptions) -> None:
        self.node.set_camera_runtime_options(options)

    def set_policy_viz_state(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        valid: bool,
        racket_anchor_m: np.ndarray,
        racket_pos_m: np.ndarray,
    ) -> None:
        self.node.set_policy_viz_state(
            ball_pos_m, ball_vel_m_s, valid, racket_anchor_m, racket_pos_m)

    def latest_camera_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        return self.node.latest_camera_visualization()

    def camera_frame_rotation(self) -> np.ndarray | None:
        return self.node.camera_frame_rotation()

    def publish_right_arm_command(self, q: np.ndarray) -> None:
        self.node.publish_right_arm_command(q)

    def start_backend_control_timer(self, period_s: float, callback) -> None:
        self.node.start_backend_control_timer(period_s, callback)

    def stop_backend_control_timer(self) -> None:
        self.node.stop_backend_control_timer()

    def shutdown(self) -> None:
        try:
            self.node.shutdown_camera_source()
            self.executor.shutdown()
            self.node.destroy_node()
            self.thread.join(timeout=1.0)
        except Exception:
            pass


class PolicySafetyProbe:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.ros_bridge: RosBridge | None = None
        self.ball_estimator: BallStateEstimator | None = None
        self.ball_estimator_key: tuple[Any, ...] | None = None
        self.last_ball_observation_stamp_s: float | None = None
        self.ball_observation_clock_domain: str | None = None
        self.compensator = None
        self.compensation_error: str | None = None
        self.compensation_was_enabled = False
        self.compensator_key: tuple[Any, ...] | None = None
        self.sim_mirror: Any | None = None
        self.sim_mirror_error: str | None = None
        self.latest_sim_snapshot: dict[str, Any] | None = None
        self.sim_needs_policy_reset = True
        self.policy_needs_reentry_reset = True
        self.policy_reentry_reset_q: np.ndarray | None = None
        self.last_policy_execution_trace: dict[str, Any] | None = None
        self.safe_q = target_right_arm_q_rad().astype(np.float32)
        self.safety_limiter = RightArmCommandSafetyLimiter(
            self.safe_q.astype(np.float64), dt=cfg.control_dt,
            parking_velocity_scale=cfg.safety_parking_velocity_scale,
            parking_acceleration_scale=cfg.safety_parking_acceleration_scale)
        self.command_stream_active = False
        self.command_publish_lease_deadline_wall = 0.0
        self.command_publish_lease_expired = False
        self.reset_workspace_bypass_until_wall = 0.0
        self.reset_workspace_bypass_target_q: np.ndarray | None = None
        self.last_command_arm: dict[str, Any] = {
            "armed": False,
            "reason": "not_requested",
            "feedbackAgeS": None,
            "jointPosition": None,
        }
        self.last_command_disable_hold: dict[str, Any] = {
            "published": False,
            "feedbackAvailable": False,
            "jointPosition": None,
        }
        self.control_loop_stop = threading.Event()
        self.control_loop_thread: threading.Thread | None = None
        self.control_loop_uses_ros_timer = False
        self.control_loop_times: deque[float] = deque(maxlen=200)
        self.control_loop_run_times: deque[float] = deque(maxlen=200)
        self.control_loop_overrun_count = 0
        self.control_loop_last_error: str | None = None
        self.control_loop_last_wall: float | None = None
        self.control_loop_tick = 0
        self.backend_snapshot_period_s = (
            1.0 / max(float(cfg.backend_snapshot_rate_hz), 1e-6))
        self.backend_next_snapshot_wall = 0.0
        self.runtime_options: RuntimeOptions | None = None
        self.last_camera_runtime_options: CameraRuntimeOptions | None = None
        self.web_drag_ball_pos = np.zeros(3, dtype=np.float32)
        self.web_drag_ball_vel = np.zeros(3, dtype=np.float32)
        self.web_drag_valid = True
        self.web_noise_std_m = float(cfg.ball_noise_std_m)
        self.latest_snapshot_payload: dict[str, Any] | None = None
        self.recorder = AsyncProbeRecorder(cfg)
        self.controller = MjxRLPolicyController(
            str(cfg.model_path),
            dt=cfg.control_dt,
            robot_xml_path=str(cfg.robot_xml_path),
            enable_delay_conditioning=(cfg.policy_obs_dim == 67),
            use_valid_ball_obs_age=cfg.rl_use_valid_ball_obs_age,
        )
        self._init_compensator()
        self._init_sim_mirror()
        self.done_monitor = DoneConditionMonitor(
            ball_z_floor=cfg.done_z_min_m,
            ball_z_ceiling=cfg.done_z_max_m,
            ball_x_range=cfg.done_x_range_m,
            ball_y_range=cfg.done_y_range_m,
            racket_anchor_distance_limit=cfg.done_racket_anchor_distance_m,
            racket_z_high_offset=cfg.done_racket_z_high_offset_m,
            racket_z_low_offset=cfg.done_racket_z_low_offset_m,
            ee_x_limits=cfg.ee_x_limits_m,
            ee_y_limits=cfg.ee_y_limits_m,
            ee_z_limits=cfg.ee_z_limits_m,
            ee_psi_limits_rad=cfg.ee_psi_limits_rad,
            ee_b_limits_rad=cfg.ee_b_limits_rad,
            ee_b_min_rad=cfg.ee_b_min_rad,
            ee_b_max_rad=cfg.ee_b_max_rad,
            ee_c_limits_rad=cfg.ee_c_limits_rad,
            ee_c_min_rad=cfg.ee_c_min_rad,
            ee_c_max_rad=cfg.ee_c_max_rad,
            ee_orientation_reference_xyzw=cfg.ee_orientation_reference_xyzw,
            ee_max_orientation_error_deg=cfg.ee_max_orientation_error_deg,
            enable_predictive_stop=cfg.done_enable_predictive_stop,
            prediction_horizon_s=cfg.done_prediction_horizon_s,
            prediction_dt_s=cfg.done_prediction_dt_s,
            prediction_margin_m=cfg.done_prediction_margin_m,
            prediction_orientation_margin_rad=(
                cfg.done_prediction_orientation_margin_rad),
        )
        self._done_motion_prev_t_s: float | None = None
        self._done_motion_prev_pos_m: np.ndarray | None = None
        self._done_motion_prev_vel_m_s: np.ndarray | None = None
        self._done_motion_prev_psi_rad: float | None = None
        self._done_motion_prev_abc_rad: np.ndarray | None = None
        if cfg.ros_bridge or cfg.enable_camera_ball:
            self.ros_bridge = RosBridge(cfg)
        initial_snapshot = self.reset()
        self.latest_snapshot_payload = initial_snapshot
        self.runtime_options = self._runtime_options_from_payload({})
        self.web_drag_ball_pos = np.asarray(
            self.initial_ball_pos(), dtype=np.float32)
        if self.cfg.backend_control_loop:
            self._start_backend_control_loop()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            reset_seed_q = self._safety_limiter_reset_reference()
            reset_target_q = np.asarray(self.safe_q, dtype=np.float64).copy()
            self.controller.reset()
            self.sim_needs_policy_reset = True
            self.policy_needs_reentry_reset = True
            self.policy_reentry_reset_q = np.asarray(
                reset_target_q, dtype=np.float32).copy()
            self.last_policy_execution_trace = None
            if self.sim_mirror is not None:
                try:
                    self.latest_sim_snapshot = self.sim_mirror.reset(
                        arm_q=reset_target_q)
                except Exception as exc:  # noqa: BLE001 - keep reset usable.
                    self.sim_mirror_error = str(exc)
                    self.latest_sim_snapshot = self._sim_unavailable_payload(str(exc))
            # Reset is the operator's explicit recovery command.  It must be
            # allowed to move back to the known reset pose even when the current
            # EE pose is outside the workspace box that just latched done.
            self.reset_workspace_bypass_until_wall = time.monotonic() + 15.0
            self.reset_workspace_bypass_target_q = reset_target_q.copy()
            self.done_monitor.reset_counters()
            self.now_s = 0.0
            self.has_captured = False
            self.has_released = False
            self.release_candidate_ticks = 0
            self.release_samples: list[tuple[float, float, float]] = []
            self.last_valid_time_s: float | None = None
            self.failsafe_active = False
            self.done_failsafe_active = False
            self.done_failsafe_reason = ""
            self.done_failsafe_conditions: dict[str, bool] = {}
            self.hard_workspace_stop_latched = False
            self.hard_workspace_stop_reason = ""
            self.hard_workspace_stop_conditions: dict[str, bool] = {}
            self.hard_workspace_stop_hold_sent = False
            self.ball_done_streaks: dict[str, int] = {}
            self.ball_done_raw_conditions: dict[str, bool] = {}
            self.workspace_soft_prediction_active = False
            self.workspace_soft_conditions: dict[str, bool] = {}
            self.workspace_soft_filter: dict[str, Any] = {
                "active": False,
                "predictive": False,
                "alpha": 1.0,
            }
            self.workspace_soft_entry_q: np.ndarray | None = None
            self.workspace_hold_active = False
            self.workspace_hold_q = np.asarray(
                reset_target_q, dtype=np.float64).copy()
            self.workspace_hold_last_safe_q = self.workspace_hold_q.copy()
            self.workspace_hold_trigger_reason = ""
            self.workspace_hold_constraint = ""
            self.last_policy_nominal_reference = np.asarray(
                reset_target_q, dtype=np.float64).copy()
            self.last_compensated_reference = np.asarray(
                reset_target_q, dtype=np.float64).copy()
            self.last_published_safe_reference = np.asarray(
                reset_target_q, dtype=np.float64).copy()
            self._done_motion_prev_t_s = None
            self._done_motion_prev_pos_m = None
            self._done_motion_prev_vel_m_s = None
            self._done_motion_prev_psi_rad = None
            self._done_motion_prev_abc_rad = None
            self.last_ball_pos = np.asarray(self.initial_ball_pos(), dtype=np.float32)
            self.last_ball_vel = np.zeros(3, dtype=np.float32)
            self.last_observed_ball_pos = self.last_ball_pos.copy()
            self.last_observed_ball_vel = np.zeros(3, dtype=np.float32)
            self.last_measured_ball_pos = self.last_ball_pos.copy()
            self.last_measured_ball_vel = np.zeros(3, dtype=np.float32)
            self.last_done_ball_pos = self.last_ball_pos.copy()
            self.last_done_ball_valid = False
            self.ball_estimator = None
            self.ball_estimator_key = None
            self.last_ball_observation_stamp_s = None
            self.ball_observation_clock_domain = None
            self.next_ball_observation_s = 0.0
            self.last_estimator_info = self._estimator_off_payload()
            self.last_release_gate_info = self._release_gate_payload(
                valid=False,
                age_s=None,
                vz_m_s=None,
                z_drop_m=0.0,
                candidate=False,
            )
            self.safety_limiter.reset(reset_seed_q)
            self._reset_compensator_to_current_arm(reset_seed_q)
            self._sync_policy_to_safe_pose()
            self.web_drag_ball_pos = self.last_ball_pos.copy()
            self.web_drag_ball_vel = self.last_ball_vel.copy()
            self.web_drag_valid = True
            self.web_noise_std_m = float(self.cfg.ball_noise_std_m)
            snapshot = self.snapshot(
                "reset", "", {}, self.last_ball_pos,
                self.last_observed_ball_pos, self.last_observed_ball_vel,
                False, 0.0, 0.0,
                measured_ball_pos=self.last_measured_ball_pos,
                measured_ball_vel=self.last_measured_ball_vel,
                estimator_info=self.last_estimator_info,
                ball_source="drag",
                camera_feedback=self.camera_feedback())
            self.latest_snapshot_payload = snapshot
            self.recorder.record_event({
                "event": "reset",
                "wallTimeS": time.time(),
                "time": round(float(self.now_s), 6),
            })
            self.recorder.record_snapshot(snapshot)
            return snapshot

    def initial_ball_pos(self) -> tuple[float, float, float]:
        anchor = self.anchor()
        return (
            float(anchor[0]),
            float(anchor[1]),
            min(self.cfg.done_z_max_m - 0.12, max(self.cfg.done_z_min_m + 0.12, float(anchor[2]) + 0.30)),
        )

    def anchor(self) -> np.ndarray:
        anchor = getattr(self.controller, "_racket_anchor_pos_base", None)
        if anchor is None or not np.all(np.isfinite(anchor)):
            return np.zeros(3, dtype=np.float32)
        return np.asarray(anchor, dtype=np.float32).reshape(3)

    def _sync_policy_to_safe_pose(self) -> None:
        self.controller.arm_cmd_q = self.safe_q.copy()
        self.controller.arm_cmd_qvel = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.controller.last_cmd_q = self.safe_q.copy()
        self.controller.last_cmd_qvel = self.controller.arm_cmd_qvel.copy()
        self.controller.last_cmd_qacc = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.controller.last_action = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)

    def _reset_recovery_active(self) -> bool:
        deadline = float(getattr(
            self, "reset_workspace_bypass_until_wall", 0.0) or 0.0)
        if deadline <= 0.0:
            return False
        now = time.monotonic()
        if now >= deadline:
            self.reset_workspace_bypass_until_wall = 0.0
            self.reset_workspace_bypass_target_q = None
            return False
        target_q = self._right_arm_joint_vector_or_none(getattr(
            self, "reset_workspace_bypass_target_q", None))
        if target_q is not None and self.ros_bridge is not None:
            try:
                q_fb, _dq_fb = self.ros_bridge.get_joint_feedback()
            except Exception:
                q_fb = None
            feedback_q = self._right_arm_joint_vector_or_none(q_fb)
            if feedback_q is not None:
                err = float(np.max(np.abs(feedback_q - target_q)))
                if err <= 0.035:
                    self.reset_workspace_bypass_until_wall = 0.0
                    self.reset_workspace_bypass_target_q = None
                    return False
        return True

    def _reset_recovery_payload(self) -> dict[str, Any]:
        deadline = float(getattr(
            self, "reset_workspace_bypass_until_wall", 0.0) or 0.0)
        active = self._reset_recovery_active()
        remaining_s = max(0.0, deadline - time.monotonic()) if active else 0.0
        return {
            "active": bool(active),
            "remainingS": round(float(remaining_s), 6),
            "target": (
                None
                if self._right_arm_joint_vector_or_none(getattr(
                    self, "reset_workspace_bypass_target_q", None)) is None
                else _rounded_vector(self.reset_workspace_bypass_target_q)),
        }

    def _mark_policy_paused(self) -> None:
        """Fence both policy histories at a non-policy control interval."""
        self.policy_needs_reentry_reset = True
        self.sim_needs_policy_reset = True

    def _reset_real_policy_for_sim_reentry(self) -> np.ndarray:
        """Reset real policy history and seed it from the current real arm."""
        reset_q = np.asarray(
            self._safety_limiter_reset_reference(), dtype=np.float32).reshape(-1)
        reset_q = np.clip(
            reset_q, self.controller.arm_q_low, self.controller.arm_q_high,
        ).astype(np.float32)
        self.controller.reset()
        zeros = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.controller.arm_cmd_q = reset_q.copy()
        self.controller.arm_cmd_qvel = zeros.copy()
        self.controller.last_cmd_q = reset_q.copy()
        self.controller.last_cmd_qvel = zeros.copy()
        self.controller.last_cmd_qacc = zeros.copy()
        self.controller.last_action = zeros.copy()
        self._reset_compensator_to_current_arm(reset_q)
        self.policy_reentry_reset_q = reset_q.copy()
        self.policy_needs_reentry_reset = False
        return reset_q

    def _right_arm_joint_vector_or_none(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            q = np.asarray(value, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if q.shape[0] != len(RIGHT_ARM_JOINTS) or not np.all(np.isfinite(q)):
            return None
        return q

    def _safety_limiter_reset_reference(self) -> np.ndarray:
        if self.ros_bridge is not None:
            q_fb, _dq_fb = self.ros_bridge.get_joint_feedback()
            q = self._right_arm_joint_vector_or_none(q_fb)
            if q is not None:
                return q
        q = self._right_arm_joint_vector_or_none(
            getattr(self.safety_limiter, "last_cmd", None))
        if q is not None:
            return q
        try:
            q = self._right_arm_joint_vector_or_none(
                self.controller.current_arm_cmd_q())
            if q is not None:
                return q
        except Exception:
            pass
        return self.safe_q.astype(np.float64)

    def _init_sim_mirror(self) -> None:
        self.sim_mirror = None
        self.sim_mirror_error = None
        self.latest_sim_snapshot = self._sim_unavailable_payload(
            "disabled" if not self.cfg.enable_sim_mirror else "not initialized")
        if not self.cfg.enable_sim_mirror:
            return
        backend = str(self.cfg.sim_backend or "mujoco").strip().lower()
        if backend not in {"mujoco", "mjx", "auto"}:
            backend = "mujoco"
        sim_cfg = SimMirrorConfig(
            model_path=self.cfg.model_path,
            policy_xml_path=self.cfg.robot_xml_path,
            sim_xml_path=self.cfg.sim_xml_path,
            policy_obs_dim=self.cfg.policy_obs_dim,
            control_dt=self.cfg.control_dt,
            backend=backend,
            frame_rate_hz=self.cfg.sim_frame_rate_hz,
            render_width=self.cfg.sim_render_width,
            render_height=self.cfg.sim_render_height,
            enable_video_frames=self.cfg.sim_enable_video_frames,
            use_mujoco_renderer=self.cfg.sim_use_mujoco_renderer,
            enable_compensation=self.cfg.enable_compensation,
            compensation_mode=self.cfg.compensation_mode,
            comp_delay_ms=self.cfg.comp_delay_ms,
            comp_actuator_tau_s=self.cfg.comp_actuator_tau_s,
            comp_actuator_gain=self.cfg.comp_actuator_gain,
            comp_mpc_beta=self.cfg.comp_mpc_beta,
            comp_mpc_delay_scale=self.cfg.comp_mpc_delay_scale,
            comp_mpc_tau_scale=self.cfg.comp_mpc_tau_scale,
            comp_mpc_horizon_steps=self.cfg.comp_mpc_horizon_steps,
            comp_mpc_tracking_weight=self.cfg.comp_mpc_tracking_weight,
            comp_mpc_nominal_weight=self.cfg.comp_mpc_nominal_weight,
            comp_mpc_delta_weight=self.cfg.comp_mpc_delta_weight,
            comp_mpc_max_delta_rad=self.cfg.comp_mpc_max_delta_rad,
            rl_use_valid_ball_obs_age=self.cfg.rl_use_valid_ball_obs_age,
        )
        if bool(self.cfg.sim_subprocess):
            try:
                self.sim_mirror = SimMirrorSubprocess(sim_cfg)
                try:
                    startup_timeout_s = float(os.environ.get(
                        "SIM_STARTUP_TIMEOUT_S",
                        os.environ.get(
                            "PROBE_CONTROL_SUBPROCESS_STARTUP_TIMEOUT_S",
                            "300.0"),
                    ))
                except (TypeError, ValueError):
                    startup_timeout_s = 300.0
                self.latest_sim_snapshot = self.sim_mirror.wait_until_ready(
                    max(10.0, startup_timeout_s))
                return
            except Exception as exc:  # noqa: BLE001 - keep real control alive.
                failed_mirror = self.sim_mirror
                self.sim_mirror = None
                if failed_mirror is not None:
                    try:
                        failed_mirror.close()
                    except Exception:
                        pass
                self.sim_mirror_error = str(exc)
                self.latest_sim_snapshot = self._sim_unavailable_payload(str(exc))
                return
        last_error: Exception | None = None
        backends = (
            ("mjx", "mujoco") if backend == "auto"
            else (backend,)
        )
        for candidate in backends:
            try:
                mirror_cls = (
                    RealBallMjxJaxSimMirror
                    if candidate == "mjx" else RealBallSimMirror
                )
                self.sim_mirror = mirror_cls(sim_cfg, logger=self.controller._logger)
                self.latest_sim_snapshot = self.sim_mirror.snapshot()
                return
            except Exception as exc:  # noqa: BLE001 - keep real control alive.
                last_error = exc
                if backend != "auto":
                    break
        try:
            raise last_error or RuntimeError("simulation mirror initialization failed")
        except Exception as exc:  # noqa: BLE001 - keep real control alive.
            self.sim_mirror = None
            self.sim_mirror_error = str(exc)
            self.latest_sim_snapshot = self._sim_unavailable_payload(str(exc))

    def _sim_unavailable_payload(self, reason: str | None = None) -> dict[str, Any]:
        return {
            "enabled": bool(self.cfg.enable_sim_mirror),
            "available": False,
            "active": False,
            "backend": str(getattr(self.cfg, "sim_backend", "mujoco")),
            "error": reason or self.sim_mirror_error,
        }

    def _step_sim_mirror(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        *,
        ball_valid: bool,
        age_s: float,
        options: RuntimeOptions,
        source_seq: int,
        source_time_s: float,
        ball_observation_seq: int,
        camera_stamp_s: float | None,
        control_mode: str,
        policy_active: bool,
    ) -> dict[str, Any]:
        if not self.cfg.enable_sim_mirror:
            self.latest_sim_snapshot = self._sim_unavailable_payload("disabled")
            return dict(self.latest_sim_snapshot)
        if not bool(options.enable_sim_mirror):
            self.sim_needs_policy_reset = True
            self.policy_needs_reentry_reset = True
            payload = self._sim_unavailable_payload("runtime disabled")
            payload["available"] = bool(self.sim_mirror is not None)
            self.latest_sim_snapshot = payload
            return payload
        if self.sim_mirror is None:
            if self.sim_mirror_error is None:
                self._init_sim_mirror()
            if self.sim_mirror is None:
                self.latest_sim_snapshot = self._sim_unavailable_payload(
                    self.sim_mirror_error or "initialization failed")
                return dict(self.latest_sim_snapshot)
        if not bool(policy_active):
            self._mark_policy_paused()
            try:
                if hasattr(self.sim_mirror, "drain_completed_samples"):
                    self.recorder.record_sim_control_samples(
                        self.sim_mirror.drain_completed_samples())
                payload = self.sim_mirror.snapshot()
            except Exception:
                payload = dict(
                    self.latest_sim_snapshot
                    or self._sim_unavailable_payload())
            payload["active"] = False
            payload["holdReason"] = f"real control mode: {control_mode}"
            self.latest_sim_snapshot = payload
            return payload
        try:
            if self.sim_needs_policy_reset:
                reset_q = (
                    self.policy_reentry_reset_q.copy()
                    if self.policy_reentry_reset_q is not None
                    else np.asarray(
                        self._safety_limiter_reset_reference(),
                        dtype=np.float32,
                    )
                )
                try:
                    self.sim_mirror.reset(arm_q=reset_q)
                except TypeError:
                    self.sim_mirror.reset(reset_q)
                self.sim_needs_policy_reset = False
                self.policy_reentry_reset_q = None
            if isinstance(self.sim_mirror, SimMirrorSubprocess):
                snap = self.sim_mirror.step(
                    ball_pos,
                    ball_vel,
                    ball_valid=bool(ball_valid),
                    ball_obs_age_s=float(max(0.0, age_s)),
                    options=options,
                    source_seq=int(source_seq),
                    source_time_s=float(source_time_s),
                    ball_observation_seq=int(ball_observation_seq),
                    camera_stamp_s=camera_stamp_s,
                    control_mode=str(control_mode),
                )
                self.recorder.record_sim_control_samples(
                    self.sim_mirror.drain_completed_samples())
            else:
                submitted_wall_s = time.time()
                snap = self.sim_mirror.step(
                    ball_pos,
                    ball_vel,
                    ball_valid=bool(ball_valid),
                    ball_obs_age_s=float(max(0.0, age_s)),
                    options=options,
                )
                snap["sync"] = {
                    "ready": True,
                    "episode": 0,
                    "sourceSeq": int(source_seq),
                    "sourceTimeS": float(source_time_s),
                    "submittedWallTimeS": float(submitted_wall_s),
                    "completedWallTimeS": float(time.time()),
                    "streamSeq": int(source_seq),
                    "ballObservationSeq": int(ball_observation_seq),
                    "cameraStampS": camera_stamp_s,
                    "inputBall": _float32_vector(ball_pos),
                    "inputVelocity": _float32_vector(ball_vel),
                    "inputValid": bool(ball_valid),
                    "inputAgeS": float(max(0.0, age_s)),
                    "controlMode": str(control_mode),
                    "runtimeConfigMatch": _sim_runtime_config_matches(
                        self.sim_mirror.cfg,
                        options,
                        str(snap.get("backend", "")),
                    ),
                    "streamContinuous": True,
                    "alignmentValid": _sim_runtime_config_matches(
                        self.sim_mirror.cfg,
                        options,
                        str(snap.get("backend", "")),
                    ),
                    "lagTicks": 0,
                    "lagS": 0.0,
                }
                self.recorder.record_sim_control_samples([snap])
            if snap.get("available") is not False:
                snap["active"] = True
            self.latest_sim_snapshot = snap
            return snap
        except Exception as exc:  # noqa: BLE001 - keep real control alive.
            self.sim_mirror_error = str(exc)
            self.latest_sim_snapshot = self._sim_unavailable_payload(str(exc))
            return dict(self.latest_sim_snapshot)

    def _compensator_key(self, options: RuntimeOptions | None = None) -> tuple[Any, ...]:
        mode = (
            self.cfg.compensation_mode if options is None
            else options.compensation_mode)
        delay_ms = (
            self.cfg.comp_delay_ms if options is None else options.comp_delay_ms)
        tau_s = (
            self.cfg.comp_actuator_tau_s
            if options is None else options.comp_actuator_tau_s)
        gain = (
            self.cfg.comp_actuator_gain
            if options is None else options.comp_actuator_gain)
        beta = self.cfg.comp_mpc_beta if options is None else options.comp_mpc_beta
        delay_scale = (
            self.cfg.comp_mpc_delay_scale
            if options is None else options.comp_mpc_delay_scale)
        tau_scale = (
            self.cfg.comp_mpc_tau_scale
            if options is None else options.comp_mpc_tau_scale)
        horizon = (
            self.cfg.comp_mpc_horizon_steps
            if options is None else options.comp_mpc_horizon_steps)
        tracking = (
            self.cfg.comp_mpc_tracking_weight
            if options is None else options.comp_mpc_tracking_weight)
        nominal = (
            self.cfg.comp_mpc_nominal_weight
            if options is None else options.comp_mpc_nominal_weight)
        delta = (
            self.cfg.comp_mpc_delta_weight
            if options is None else options.comp_mpc_delta_weight)
        max_delta = (
            self.cfg.comp_mpc_max_delta_rad
            if options is None else options.comp_mpc_max_delta_rad)
        return (
            str(mode).strip().lower().replace("-", "_"),
            round(float(delay_ms), 6),
            round(float(tau_s), 6),
            round(float(gain), 6),
            int(self.cfg.comp_buffer_extra_steps),
            bool(self.cfg.comp_delay_conditioning),
            round(float(beta), 6),
            round(float(delay_scale), 6),
            round(float(tau_scale), 6),
            int(horizon),
            round(float(tracking), 6),
            round(float(nominal), 6),
            round(float(delta), 6),
            round(float(max_delta), 6),
        )

    def _init_compensator(self, options: RuntimeOptions | None = None) -> None:
        self.compensator = None
        self.compensation_error = None
        self.compensator_key = None
        if not self.cfg.enable_compensation:
            return
        if not _COMPENSATION_AVAILABLE or CommandCompensator is None:
            self.compensation_error = (
                "CommandCompensator unavailable"
                if _COMPENSATION_IMPORT_ERROR is None
                else f"CommandCompensator unavailable: {_COMPENSATION_IMPORT_ERROR}"
            )
            return
        try:
            self.compensator = CommandCompensator(
                n_joints=len(RIGHT_ARM_JOINTS),
                dt=float(self.cfg.control_dt),
                mode=str(
                    self.cfg.compensation_mode if options is None
                    else options.compensation_mode),
                delay_ms=float(
                    self.cfg.comp_delay_ms if options is None
                    else options.comp_delay_ms),
                actuator_tau_s=float(
                    self.cfg.comp_actuator_tau_s if options is None
                    else options.comp_actuator_tau_s),
                actuator_gain=float(
                    self.cfg.comp_actuator_gain if options is None
                    else options.comp_actuator_gain),
                command_buffer_extra_steps=int(
                    self.cfg.comp_buffer_extra_steps),
                delay_conditioning=bool(self.cfg.comp_delay_conditioning),
                mpc_beta=float(
                    self.cfg.comp_mpc_beta if options is None
                    else options.comp_mpc_beta),
                mpc_delay_scale=float(
                    self.cfg.comp_mpc_delay_scale if options is None
                    else options.comp_mpc_delay_scale),
                mpc_tau_scale=float(
                    self.cfg.comp_mpc_tau_scale if options is None
                    else options.comp_mpc_tau_scale),
                mpc_horizon_steps=int(
                    self.cfg.comp_mpc_horizon_steps if options is None
                    else options.comp_mpc_horizon_steps),
                mpc_tracking_weight=float(
                    self.cfg.comp_mpc_tracking_weight if options is None
                    else options.comp_mpc_tracking_weight),
                mpc_nominal_weight=float(
                    self.cfg.comp_mpc_nominal_weight if options is None
                    else options.comp_mpc_nominal_weight),
                mpc_delta_weight=float(
                    self.cfg.comp_mpc_delta_weight if options is None
                    else options.comp_mpc_delta_weight),
                mpc_max_delta_rad=float(
                    self.cfg.comp_mpc_max_delta_rad if options is None
                    else options.comp_mpc_max_delta_rad),
                q_low=self.controller.arm_q_low,
                q_high=self.controller.arm_q_high,
            )
            self.compensator_key = self._compensator_key(options)
        except Exception as exc:  # noqa: BLE001 - keep probe usable.
            self.compensator = None
            self.compensation_error = str(exc)
            self.compensator_key = None

    def _ensure_compensator(self, options: RuntimeOptions) -> None:
        if not bool(options.enable_compensation):
            return
        key = self._compensator_key(options)
        if self.compensator is not None and self.compensator_key == key:
            return
        self._init_compensator(options)
        self.compensation_was_enabled = False

    def _reset_compensator_to_current_arm(
        self,
        current_q_override: np.ndarray | None = None,
    ) -> None:
        if self.compensator is None:
            return
        try:
            current_q = self._right_arm_joint_vector_or_none(current_q_override)
            if current_q is None and self.ros_bridge is not None:
                current_q, _ = self.ros_bridge.get_joint_feedback()
                current_q = self._right_arm_joint_vector_or_none(current_q)
            if current_q is None:
                current_q = self._right_arm_joint_vector_or_none(
                    self.controller.current_arm_cmd_q())
            if current_q is not None:
                self.compensator.reset(current_q.astype(np.float32))
        except Exception as exc:  # noqa: BLE001 - compensation is optional.
            self.compensation_error = f"reset failed: {exc}"

    def _compensation_payload(self, options: RuntimeOptions | None = None) -> dict[str, Any]:
        requested = (
            self.cfg.enable_compensation
            if options is None else bool(options.enable_compensation)
        )
        active = bool(
            requested and self.compensator is not None
            and self.compensation_was_enabled)
        mode = (
            self.cfg.compensation_mode if options is None
            else options.compensation_mode)
        delay_ms = (
            self.cfg.comp_delay_ms if options is None else options.comp_delay_ms)
        tau_s = (
            self.cfg.comp_actuator_tau_s
            if options is None else options.comp_actuator_tau_s)
        gain = (
            self.cfg.comp_actuator_gain
            if options is None else options.comp_actuator_gain)
        payload = {
            "enabled": bool(requested),
            "available": bool(self.compensator is not None),
            "active": active,
            "mode": str(mode),
            "delayMs": round(float(delay_ms), 3),
            "tauS": round(float(tau_s), 6),
            "gain": round(float(gain), 6),
            "error": self.compensation_error,
        }
        if options is not None:
            payload.update({
                "mpcBeta": round(float(options.comp_mpc_beta), 6),
                "mpcDelayScale": round(
                    float(options.comp_mpc_delay_scale), 6),
                "mpcTauScale": round(float(options.comp_mpc_tau_scale), 6),
                "mpcHorizonSteps": int(options.comp_mpc_horizon_steps),
                "mpcTrackingWeight": round(
                    float(options.comp_mpc_tracking_weight), 6),
                "mpcNominalWeight": round(
                    float(options.comp_mpc_nominal_weight), 6),
                "mpcDeltaWeight": round(
                    float(options.comp_mpc_delta_weight), 6),
                "mpcMaxDeltaRad": round(
                    float(options.comp_mpc_max_delta_rad), 6),
            })
        if self.compensator is not None:
            payload["bufferLen"] = int(
                getattr(self.compensator, "command_buffer_len", 0))
        return payload

    def _apply_actuator_compensation(
        self,
        q: np.ndarray,
        options: RuntimeOptions,
    ) -> np.ndarray:
        raw = np.asarray(q, dtype=np.float64).reshape(-1)
        nominal_q = raw.copy()
        if not options.enable_compensation:
            self.compensation_was_enabled = False
            return raw
        self._ensure_compensator(options)
        if (self.compensator is None
                or not getattr(self.compensator, "enabled", True)):
            self.compensation_was_enabled = False
            return raw
        if not self.compensation_was_enabled:
            self._reset_compensator_to_current_arm()
            self.compensation_was_enabled = True
        try:
            cmd_state = self.controller.latest_command_state()
            nominal_qvel = np.asarray(
                cmd_state.get("velocity", np.zeros_like(raw)),
                dtype=np.float32,
            ).reshape(-1)
            nominal_qdd = np.asarray(
                cmd_state.get("acceleration", np.zeros_like(raw)),
                dtype=np.float32,
            ).reshape(-1)
            actual_q = None
            if self.ros_bridge is not None:
                actual_q, _ = self.ros_bridge.get_joint_feedback()
            applied_q = (
                np.asarray(actual_q, dtype=np.float32).reshape(-1)
                if actual_q is not None else raw.astype(np.float32)
            )
            if (nominal_qvel.shape[0] != raw.shape[0]
                    or nominal_qdd.shape[0] != raw.shape[0]
                    or applied_q.shape[0] != raw.shape[0]):
                return raw
            compensated = self.compensator.step(
                q_cmd=raw.astype(np.float32),
                qdot_cmd=nominal_qvel,
                qdd_cmd=nominal_qdd,
                applied_q=applied_q,
            )
            self.compensation_error = None
            return np.asarray(compensated, dtype=np.float64).reshape(-1)
        except Exception as exc:  # noqa: BLE001 - fall back to nominal command.
            self.compensation_error = str(exc)
            return raw

    def _parking_workspace_violation_score(self, q: np.ndarray) -> float:
        q_vec = self._right_arm_joint_vector_or_none(q)
        if q_vec is None:
            return float("inf")
        try:
            racket_pos, ee_quat_xyzw, fk_ok = (
                self.controller._compute_racket_pose_quat_base(q_vec))
        except Exception:
            return float("inf")
        if not fk_ok:
            return float("inf")

        racket = np.asarray(racket_pos, dtype=np.float64).reshape(-1)
        if racket.shape[0] != 3 or not np.all(np.isfinite(racket)):
            return float("inf")
        anchor = np.asarray(self.anchor(), dtype=np.float64).reshape(3)
        score = 0.0

        def add_range_violation(
            value: float,
            lo: float,
            hi: float,
            scale: float,
        ) -> None:
            nonlocal score
            if value < lo:
                score += ((lo - value) / max(scale, 1e-6)) ** 2
            elif value > hi:
                score += ((value - hi) / max(scale, 1e-6)) ** 2

        dist = float(np.linalg.norm(racket - anchor))
        limit = max(1e-6, float(self.cfg.done_racket_anchor_distance_m))
        if dist > limit:
            score += ((dist - limit) / limit) ** 2

        dz = float(racket[2] - anchor[2])
        add_range_violation(
            dz,
            float(self.cfg.done_racket_z_low_offset_m),
            float(self.cfg.done_racket_z_high_offset_m),
            max(1e-6, abs(float(self.cfg.done_racket_z_high_offset_m)
                          - float(self.cfg.done_racket_z_low_offset_m))),
        )

        for value, limits in (
            (float(racket[0]), self.cfg.ee_x_limits_m),
            (float(racket[1]), self.cfg.ee_y_limits_m),
            (float(racket[2]), self.cfg.ee_z_limits_m),
        ):
            if limits is None:
                continue
            lo, hi = [float(v) for v in limits]
            add_range_violation(value, lo, hi, max(1e-6, abs(hi - lo)))

        abc = _quat_to_euler_xyz_rad(ee_quat_xyzw)
        if abc is not None:
            if (self.cfg.ee_b_limits_rad is not None
                    and not _angle_in_limits_rad(
                        float(abc[1]), self.cfg.ee_b_limits_rad)):
                score += 1.0
            if (self.cfg.ee_b_min_rad is not None
                    and not _angle_above_min_rad(
                        float(abc[1]), self.cfg.ee_b_min_rad)):
                score += 1.0
            if (self.cfg.ee_b_max_rad is not None
                    and not _angle_below_max_rad(
                        float(abc[1]), self.cfg.ee_b_max_rad)):
                score += 1.0
            if (self.cfg.ee_c_limits_rad is not None
                    and not _angle_in_limits_rad(
                        float(abc[2]), self.cfg.ee_c_limits_rad)):
                score += 1.0
            if (self.cfg.ee_c_min_rad is not None
                    and not _angle_above_min_rad(
                        float(abc[2]), self.cfg.ee_c_min_rad)):
                score += 1.0
            if (self.cfg.ee_c_max_rad is not None
                    and not _angle_below_max_rad(
                        float(abc[2]), self.cfg.ee_c_max_rad)):
                score += 1.0
        elif (self.cfg.ee_b_limits_rad is not None
              or self.cfg.ee_b_min_rad is not None
              or self.cfg.ee_b_max_rad is not None
              or self.cfg.ee_c_limits_rad is not None
              or self.cfg.ee_c_min_rad is not None
              or self.cfg.ee_c_max_rad is not None):
            score += 1.0

        if (self.cfg.ee_orientation_reference_xyzw is not None
                and self.cfg.ee_max_orientation_error_deg > 0.0):
            angle_error_deg = _quat_angle_error_deg(
                ee_quat_xyzw, self.cfg.ee_orientation_reference_xyzw)
            if angle_error_deg is None:
                score += 1.0
            elif angle_error_deg > self.cfg.ee_max_orientation_error_deg:
                span = max(1e-6, float(self.cfg.ee_max_orientation_error_deg))
                score += ((angle_error_deg
                           - self.cfg.ee_max_orientation_error_deg)
                          / span) ** 2
        return float(score)

    def _predictive_workspace_translation_direction(
        self,
    ) -> np.ndarray | None:
        """Return the Cartesian direction of the current predicted violation."""
        monitor = getattr(self, "done_monitor", None)
        prediction = getattr(monitor, "last_prediction", None)
        if not isinstance(prediction, dict):
            return None
        condition = str(prediction.get("condition", ""))
        try:
            value = np.asarray(
                prediction.get("value"), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if value.shape[0] < 3 or not np.all(np.isfinite(value[:3])):
            return None
        predicted_pos = value[:3]

        direction = np.zeros(3, dtype=np.float64)
        axis_conditions = {
            "ee_predicted_x_out_of_workspace": (0, self.cfg.ee_x_limits_m),
            "ee_predicted_y_out_of_workspace": (1, self.cfg.ee_y_limits_m),
            "ee_predicted_z_out_of_workspace": (2, self.cfg.ee_z_limits_m),
        }
        if condition in axis_conditions:
            axis, limits = axis_conditions[condition]
            if limits is None:
                return None
            lo, hi = [float(v) for v in limits]
            direction[axis] = (
                1.0
                if abs(float(predicted_pos[axis]) - hi)
                <= abs(float(predicted_pos[axis]) - lo)
                else -1.0)
            return direction
        if condition == "racket_predicted_too_high":
            direction[2] = 1.0
            return direction
        if condition == "racket_predicted_too_low":
            direction[2] = -1.0
            return direction
        if condition == "racket_predicted_too_far_from_anchor":
            radial = predicted_pos - np.asarray(
                self.anchor(), dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(radial))
            if norm > 1e-9:
                return radial / norm
        return None

    def _predictive_workspace_constraint(self) -> str:
        """Map the latest positional prediction to a signed-distance term."""
        prediction = getattr(
            getattr(self, "done_monitor", None), "last_prediction", None)
        if not isinstance(prediction, dict):
            return ""
        condition = str(prediction.get("condition", ""))
        if condition == "racket_predicted_too_high":
            return "racket_z_high"
        if condition == "racket_predicted_too_low":
            return "racket_z_low"
        if condition == "racket_predicted_too_far_from_anchor":
            return "racket_anchor"
        axes = {
            "ee_predicted_x_out_of_workspace": (
                0, self.cfg.ee_x_limits_m, "ee_x"),
            "ee_predicted_y_out_of_workspace": (
                1, self.cfg.ee_y_limits_m, "ee_y"),
            "ee_predicted_z_out_of_workspace": (
                2, self.cfg.ee_z_limits_m, "ee_z"),
        }
        axis_info = axes.get(condition)
        if axis_info is None:
            return ""
        axis, limits, label = axis_info
        if limits is None:
            return ""
        try:
            predicted = np.asarray(
                prediction.get("value"), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return ""
        if predicted.shape[0] < 3 or not np.all(np.isfinite(predicted[:3])):
            return ""
        lo, hi = [float(value) for value in limits]
        value = float(predicted[axis])
        side = "high" if abs(value - hi) <= abs(value - lo) else "low"
        return f"{label}_{side}"

    def _remove_predictive_outward_components(
        self,
        prev_cmd: np.ndarray,
        desired: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Brake predicted-outward motion without freezing the whole arm."""
        outward = self._predictive_workspace_translation_direction()
        if outward is None:
            # Orientation-only prediction currently has no reliable mapping
            # from cartesian feedback psi to the command FK. Keep the
            # conservative whole-command brake for that uncommon case.
            return prev_cmd.copy(), list(range(len(RIGHT_ARM_JOINTS)))
        try:
            prev_pos, _quat, prev_ok = (
                self.controller._compute_racket_pose_quat_base(prev_cmd))
        except Exception:
            return prev_cmd.copy(), list(range(len(RIGHT_ARM_JOINTS)))
        prev_pos = np.asarray(prev_pos, dtype=np.float64).reshape(-1)
        if (not prev_ok or prev_pos.shape[0] != 3
                or not np.all(np.isfinite(prev_pos))):
            return prev_cmd.copy(), list(range(len(RIGHT_ARM_JOINTS)))

        filtered = desired.copy()
        blocked: list[int] = []
        delta = desired - prev_cmd
        # Probe only the local joint direction; compensation can make the raw
        # target comparatively far away, while the 200 Hz command step remains
        # local after the dynamic limiter.
        for idx, joint_delta in enumerate(delta):
            if abs(float(joint_delta)) <= 1e-15:
                continue
            trial = prev_cmd.copy()
            trial[idx] += math.copysign(
                min(abs(float(joint_delta)), 0.01), float(joint_delta))
            try:
                trial_pos, _quat, trial_ok = (
                    self.controller._compute_racket_pose_quat_base(trial))
            except Exception:
                trial_ok = False
                trial_pos = None
            if not trial_ok or trial_pos is None:
                filtered[idx] = prev_cmd[idx]
                blocked.append(idx)
                continue
            trial_pos = np.asarray(trial_pos, dtype=np.float64).reshape(-1)
            if (trial_pos.shape[0] != 3
                    or not np.all(np.isfinite(trial_pos))):
                filtered[idx] = prev_cmd[idx]
                blocked.append(idx)
                continue
            outward_step = float(np.dot(trial_pos - prev_pos, outward))
            if outward_step > 1e-9:
                filtered[idx] = prev_cmd[idx]
                blocked.append(idx)
        return filtered, blocked

    def _workspace_soft_boundary_terms(
        self,
        position_m: np.ndarray,
    ) -> list[tuple[str, np.ndarray, float]]:
        """Return (name, outward normal, signed inward distance) constraints."""
        pos = np.asarray(position_m, dtype=np.float64).reshape(3)
        anchor = np.asarray(self.anchor(), dtype=np.float64).reshape(3)
        terms: list[tuple[str, np.ndarray, float]] = []

        radial = pos - anchor
        radial_norm = float(np.linalg.norm(radial))
        radial_limit = max(
            1e-6, float(self.cfg.done_racket_anchor_distance_m))
        if radial_norm > 1e-9:
            terms.append((
                "racket_anchor",
                radial / radial_norm,
                radial_limit - radial_norm,
            ))

        dz = float(pos[2] - anchor[2])
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        terms.extend((
            (
                "racket_z_high",
                z_axis.copy(),
                float(self.cfg.done_racket_z_high_offset_m) - dz,
            ),
            (
                "racket_z_low",
                -z_axis.copy(),
                dz - float(self.cfg.done_racket_z_low_offset_m),
            ),
        ))

        for axis, limits, label in (
            (0, self.cfg.ee_x_limits_m, "ee_x"),
            (1, self.cfg.ee_y_limits_m, "ee_y"),
            (2, self.cfg.ee_z_limits_m, "ee_z"),
        ):
            if limits is None:
                continue
            lo, hi = [float(value) for value in limits]
            normal = np.zeros(3, dtype=np.float64)
            normal[axis] = 1.0
            terms.extend((
                (f"{label}_high", normal.copy(), hi - float(pos[axis])),
                (f"{label}_low", -normal.copy(), float(pos[axis]) - lo),
            ))
        return terms

    def _workspace_translation_jacobian(
        self,
        q: np.ndarray,
        base_position_m: np.ndarray,
    ) -> np.ndarray | None:
        """Numerically linearize racket translation with respect to arm q."""
        exact_getter = getattr(
            self.controller, "_compute_racket_pose_jacobians_base", None)
        if callable(exact_getter):
            try:
                _pos, _quat, translation, _angular, ok = exact_getter(q)
            except Exception:
                ok = False
                translation = None
            exact = (
                None if translation is None
                else np.asarray(translation, dtype=np.float64))
            if (ok and exact is not None
                    and exact.shape == (3, len(RIGHT_ARM_JOINTS))
                    and np.all(np.isfinite(exact))):
                return exact
        try:
            _position, base_quat, base_ok = (
                self.controller._compute_racket_pose_quat_base(q))
        except Exception:
            return None
        if not base_ok:
            return None
        jacobians = self._workspace_pose_jacobians(
            q, base_position_m, base_quat)
        return None if jacobians is None else jacobians[0]

    def _workspace_pose_linearization(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return pose/Jacobians, preferring the exact deployment-model path."""
        exact_getter = getattr(
            self.controller, "_compute_racket_pose_jacobians_base", None)
        if callable(exact_getter):
            try:
                position, quat, translation, angular, ok = exact_getter(q)
            except Exception:
                ok = False
                position = quat = translation = angular = None
            arrays = tuple(
                None if value is None
                else np.asarray(value, dtype=np.float64)
                for value in (position, quat, translation, angular)
            )
            expected_shapes = (
                (3,), (4,), (3, len(RIGHT_ARM_JOINTS)),
                (3, len(RIGHT_ARM_JOINTS)),
            )
            if (ok and all(
                    value is not None
                    and value.shape == shape
                    and np.all(np.isfinite(value))
                    for value, shape in zip(arrays, expected_shapes))):
                return arrays  # type: ignore[return-value]

        try:
            position, quat, ok = (
                self.controller._compute_racket_pose_quat_base(q))
        except Exception:
            return None
        position_vec = np.asarray(position, dtype=np.float64).reshape(-1)
        quat_vec = np.asarray(quat, dtype=np.float64).reshape(-1)
        if (not ok or position_vec.shape[0] != 3 or quat_vec.shape[0] != 4
                or not np.all(np.isfinite(position_vec))
                or not np.all(np.isfinite(quat_vec))):
            return None
        jacobians = self._workspace_pose_jacobians(
            q, position_vec, quat_vec)
        if jacobians is None:
            return None
        return position_vec, quat_vec, jacobians[0], jacobians[1]

    def _workspace_pose_jacobians(
        self,
        q: np.ndarray,
        base_position_m: np.ndarray,
        base_quat_xyzw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Numerically linearize racket translation and angular velocity."""
        q0 = np.asarray(q, dtype=np.float64).reshape(-1)
        base = np.asarray(base_position_m, dtype=np.float64).reshape(3)
        base_quat = np.asarray(base_quat_xyzw, dtype=np.float64).reshape(-1)
        if (q0.shape[0] != len(RIGHT_ARM_JOINTS)
                or base_quat.shape[0] != 4
                or not np.all(np.isfinite(base_quat))):
            return None
        translation = np.zeros(
            (3, len(RIGHT_ARM_JOINTS)), dtype=np.float64)
        angular = np.zeros((3, len(RIGHT_ARM_JOINTS)), dtype=np.float64)
        epsilon = 1e-4
        for idx in range(len(RIGHT_ARM_JOINTS)):
            q_trial = q0.copy()
            step = min(
                epsilon,
                float(RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD[idx]
                      - q0[idx]),
            )
            if step <= 1e-8:
                step = -min(
                    epsilon,
                    float(q0[idx]
                          - RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD[idx]),
                )
            if abs(step) <= 1e-8:
                continue
            q_trial[idx] += step
            try:
                trial_pos, trial_quat, trial_ok = (
                    self.controller._compute_racket_pose_quat_base(q_trial))
            except Exception:
                return None
            trial = np.asarray(trial_pos, dtype=np.float64).reshape(-1)
            if (not trial_ok or trial.shape[0] != 3
                    or not np.all(np.isfinite(trial))):
                return None
            relative_rotvec = _quat_relative_rotvec_xyzw(
                base_quat, trial_quat)
            if relative_rotvec is None:
                return None
            translation[:, idx] = (trial - base) / step
            angular[:, idx] = relative_rotvec / step
        return translation, angular

    def _workspace_hold_state(self, q: np.ndarray) -> dict[str, Any]:
        """Evaluate signed boundary distance for a joint command/measurement."""
        q_vec = self._right_arm_joint_vector_or_none(q)
        if q_vec is None:
            return {
                "valid": False,
                "positionM": None,
                "minDistanceM": None,
                "nearest": None,
                "violationScore": float("inf"),
            }
        try:
            position, _quat, fk_ok = (
                self.controller._compute_racket_pose_quat_base(q_vec))
        except Exception:
            fk_ok = False
            position = None
        pos = (
            None if position is None
            else np.asarray(position, dtype=np.float64).reshape(-1))
        if (not fk_ok or pos is None or pos.shape[0] != 3
                or not np.all(np.isfinite(pos))):
            return {
                "valid": False,
                "positionM": None,
                "minDistanceM": None,
                "nearest": None,
                "violationScore": float("inf"),
            }
        terms = self._workspace_soft_boundary_terms(pos)
        if terms:
            nearest_name, _normal, min_distance = min(
                terms, key=lambda item: float(item[2]))
        else:
            nearest_name = None
            min_distance = float("inf")
        return {
            "valid": True,
            "positionM": _rounded_vector(pos),
            "minDistanceM": float(min_distance),
            "nearest": nearest_name,
            "distancesM": {
                str(name): float(distance)
                for name, _normal, distance in terms
            },
            "violationScore": float(
                self._parking_workspace_violation_score(q_vec)),
        }

    def _filter_command_with_workspace_hold(
        self,
        raw: np.ndarray,
        *,
        profile: str,
        predictive: bool = False,
    ) -> np.ndarray:
        """Hold outward commands and immediately pass FK-verified inward ones.

        The fixed target is exactly the last published command, so engaging the
        hold creates no position-command discontinuity. While held, every new
        policy joint target is evaluated with FK. A target that moves the
        triggering EE constraint inward is passed immediately; an outward or
        worsening target is replaced by the fixed command. No time/dwell gate
        is involved.
        """
        prev_cmd = self.safety_limiter.last_cmd.copy()
        prev_vel = self.safety_limiter.last_vel.copy()
        trigger_margin = max(0.0, float(getattr(
            self.cfg, "workspace_hold_trigger_margin_m", 0.003)))

        feedback_q = None
        ros_bridge = getattr(self, "ros_bridge", None)
        if ros_bridge is not None:
            try:
                feedback_q, _feedback_dq = ros_bridge.get_joint_feedback()
            except Exception:
                feedback_q = None
        measured_q = self._right_arm_joint_vector_or_none(feedback_q)
        measured_state = (
            self._workspace_hold_state(measured_q)
            if measured_q is not None
            else self._workspace_hold_state(prev_cmd))
        feedback_used = measured_q is not None

        desired = np.asarray(raw, dtype=np.float64).reshape(-1)
        if (desired.shape[0] != len(RIGHT_ARM_JOINTS)
                or not np.all(np.isfinite(desired))):
            desired = prev_cmd.copy()
        candidate = self.safety_limiter.filter(desired, profile=profile)
        candidate_state = self._workspace_hold_state(candidate)
        prediction_active = bool(
            predictive
            and getattr(self, "workspace_soft_prediction_active", False))
        predicted_constraint = (
            self._predictive_workspace_constraint()
            if prediction_active else "")

        def candidate_moves_inward(
            reference_state: dict[str, Any],
            constraint: str,
            *,
            allow_safely_inset: bool = True,
        ) -> bool:
            """Return true only for an FK candidate improving the boundary.

            The candidate may still be outside when the real arm has already
            crossed a boundary: an acceleration-limited inward step must be
            allowed to recover over successive control ticks. At the same
            time, it may not improve one constraint by creating a larger
            violation elsewhere.
            """
            if (not reference_state.get("valid", False)
                    or not candidate_state.get("valid", False)):
                return False
            reference_score = float(
                reference_state.get("violationScore", float("inf")))
            candidate_score = float(
                candidate_state.get("violationScore", float("inf")))
            candidate_min_distance = float(
                candidate_state.get("minDistanceM", float("-inf")))
            # A target whose FK is already clearly inside every enabled
            # workspace boundary is immediately admissible. This also lets a
            # hold clear as soon as a position target can bring the real arm
            # back in, without waiting for the measured EE pose to arrive.
            if (allow_safely_inset
                    and candidate_score <= 1e-12
                    and candidate_min_distance > trigger_margin):
                return True
            reference_distance = reference_state.get(
                "distancesM", {}).get(constraint)
            candidate_distance = candidate_state.get(
                "distancesM", {}).get(constraint)
            if (constraint and reference_distance is not None
                    and candidate_distance is not None):
                return bool(
                    float(candidate_distance)
                    > float(reference_distance) + 1e-7
                    and candidate_score <= reference_score + 1e-12
                )
            # Positional FK constraints normally have a signed-distance term.
            # This fallback covers any score-only constraint conservatively.
            return bool(candidate_score < reference_score - 1e-12)

        hold_active = bool(getattr(self, "workspace_hold_active", False))
        if hold_active:
            hold_q = self._right_arm_joint_vector_or_none(
                getattr(self, "workspace_hold_q", None))
            if hold_q is None:
                hold_q = prev_cmd.copy()
            hold_state = self._workspace_hold_state(hold_q)
            constraint = str(getattr(
                self, "workspace_hold_constraint", "") or "")
            reference_state = (
                measured_state
                if feedback_used and measured_state.get("valid", False)
                else hold_state)
            if not constraint:
                constraint = str(reference_state.get("nearest") or "")
            reference_distances = reference_state.get("distancesM", {})
            candidate_distances = candidate_state.get("distancesM", {})
            reference_distance = reference_distances.get(constraint)
            candidate_distance = candidate_distances.get(constraint)
            predictive_hold_active = bool(
                str(getattr(
                    self, "workspace_hold_trigger_reason", ""))
                == "predicted_crossing"
                and prediction_active)
            moves_inward = candidate_moves_inward(
                reference_state,
                constraint,
                allow_safely_inset=not predictive_hold_active,
            )
            if moves_inward:
                self.workspace_hold_active = False
                self.workspace_hold_trigger_reason = ""
                self.workspace_hold_constraint = ""
                self.workspace_hold_last_safe_q = candidate.copy()
                self.workspace_soft_filter = {
                    "active": False,
                    "intervened": False,
                    "released": True,
                    "releaseReason": "candidate_fk_moves_inward",
                    "predictive": bool(predictive),
                    "model": "boundary_hold",
                    "holdTargetPosition": _rounded_vector(hold_q),
                    "publishedPosition": _rounded_vector(candidate),
                    "constraint": constraint,
                    "referenceConstraintDistanceM": reference_distance,
                    "candidateConstraintDistanceM": candidate_distance,
                    "feedbackUsed": bool(feedback_used),
                    "measured": measured_state,
                    "candidate": candidate_state,
                    "triggerMarginM": round(float(trigger_margin), 6),
                }
                return candidate

            # The candidate filter was look-ahead only. Re-publish exactly the
            # same target as the previous tick and zero the command integrator;
            # this creates no joint-position step at hold engagement.
            self.safety_limiter.last_cmd = hold_q.copy()
            self.safety_limiter.last_vel = np.zeros_like(hold_q)
            self.workspace_soft_filter = {
                "active": True,
                "intervened": True,
                "released": False,
                "predictive": bool(predictive),
                "model": "boundary_hold",
                "triggerReason": str(getattr(
                    self, "workspace_hold_trigger_reason", "")),
                "constraint": constraint,
                "holdTargetPosition": _rounded_vector(hold_q),
                "publishedPosition": _rounded_vector(hold_q),
                "referenceConstraintDistanceM": reference_distance,
                "candidateConstraintDistanceM": candidate_distance,
                "candidateMovesInward": False,
                "feedbackUsed": bool(feedback_used),
                "measured": measured_state,
                "candidate": candidate_state,
                "triggerMarginM": round(float(trigger_margin), 6),
            }
            return hold_q.copy()

        trigger_reason = ""
        trigger_constraint = ""
        measured_at_boundary = bool(
            measured_state.get("valid", False)
            and (float(measured_state["minDistanceM"]) <= trigger_margin
                 or float(measured_state["violationScore"]) > 1e-12))
        if measured_at_boundary:
            trigger_constraint = str(measured_state.get("nearest") or "")
            if candidate_moves_inward(measured_state, trigger_constraint):
                # The real arm can be at or beyond the boundary while the new
                # joint target is already a recovery command. Pass it on this
                # very tick; waiting for feedback to recover would deadlock a
                # position-controlled robot against the fixed hold target.
                self.workspace_hold_last_safe_q = candidate.copy()
                self.workspace_soft_filter = {
                    "active": False,
                    "intervened": False,
                    "released": True,
                    "releaseReason": "candidate_fk_moves_inward",
                    "predictive": bool(predictive),
                    "model": "boundary_hold",
                    "publishedPosition": _rounded_vector(candidate),
                    "constraint": trigger_constraint,
                    "referenceConstraintDistanceM": measured_state.get(
                        "distancesM", {}).get(trigger_constraint),
                    "candidateConstraintDistanceM": candidate_state.get(
                        "distancesM", {}).get(trigger_constraint),
                    "feedbackUsed": bool(feedback_used),
                    "measured": measured_state,
                    "candidate": candidate_state,
                    "triggerMarginM": round(float(trigger_margin), 6),
                }
                return candidate
            trigger_reason = f"measured:{measured_state.get('nearest')}"
        elif (candidate_state.get("valid", False)
              and (float(candidate_state["minDistanceM"]) <= trigger_margin
                   or float(candidate_state["violationScore"]) > 1e-12)):
            trigger_reason = f"command:{candidate_state.get('nearest')}"
            trigger_constraint = str(candidate_state.get("nearest") or "")
        elif prediction_active:
            trigger_constraint = str(
                predicted_constraint
                or measured_state.get("nearest")
                or candidate_state.get("nearest") or "")
            if not candidate_moves_inward(
                    measured_state,
                    trigger_constraint,
                    allow_safely_inset=False):
                trigger_reason = "predicted_crossing"

        if trigger_reason:
            hold_q = prev_cmd.copy()
            self.workspace_hold_active = True
            self.workspace_hold_q = hold_q.copy()
            self.workspace_hold_trigger_reason = trigger_reason
            self.workspace_hold_constraint = trigger_constraint
            # The candidate was only look-ahead. Keep the exact previously
            # published position and discard its outward command velocity.
            self.safety_limiter.last_cmd = prev_cmd.copy()
            self.safety_limiter.last_vel = np.zeros_like(prev_vel)
            self.workspace_soft_filter = {
                "active": True,
                "intervened": True,
                "released": False,
                "predictive": bool(predictive),
                "model": "boundary_hold",
                "triggerReason": trigger_reason,
                "constraint": trigger_constraint,
                "holdTargetPosition": _rounded_vector(hold_q),
                "publishedPosition": _rounded_vector(hold_q),
                "feedbackUsed": bool(feedback_used),
                "measured": measured_state,
                "candidate": candidate_state,
                "triggerMarginM": round(float(trigger_margin), 6),
            }
            return hold_q.copy()

        if (candidate_state.get("valid", False)
                and float(candidate_state["violationScore"]) <= 1e-12):
            self.workspace_hold_last_safe_q = candidate.copy()
        self.workspace_soft_filter = {
            "active": False,
            "intervened": False,
            "released": False,
            "predictive": bool(predictive),
            "model": "boundary_hold",
            "feedbackUsed": bool(feedback_used),
            "measured": measured_state,
            "candidate": candidate_state,
            "triggerMarginM": round(float(trigger_margin), 6),
        }
        return candidate

    def _legacy_filter_command_with_workspace_soft_field(
        self,
        raw: np.ndarray,
        *,
        profile: str,
        predictive: bool = False,
    ) -> np.ndarray:
        """Apply a continuous Cartesian virtual spring/damper to a q target.

        This is an admittance-style virtual fixture implemented above the
        existing position interface, not a torque-level impedance controller.
        It gradually removes outward Cartesian velocity inside a soft margin
        and adds an inward spring velocity. Tangential policy motion is left
        intact, so the arm can move around near the boundary instead of
        alternating between a frozen target and a released target.
        """
        prev_cmd = self.safety_limiter.last_cmd.copy()
        prev_vel = self.safety_limiter.last_vel.copy()
        desired = np.asarray(raw, dtype=np.float64).reshape(-1)
        if (desired.shape[0] != len(RIGHT_ARM_JOINTS)
                or not np.all(np.isfinite(desired))):
            desired = prev_cmd.copy()

        # First obtain the dynamically reachable policy/compensation command
        # for this tick. The limiter state is restored below before filtering
        # the combined policy + virtual-field velocity.
        candidate = self.safety_limiter.filter(desired, profile=profile)
        candidate_vel = (
            (candidate - prev_cmd) / max(self.safety_limiter.dt, 1e-9))
        linearization_q = prev_cmd.copy()
        measured_joint_velocity: np.ndarray | None = None
        feedback_used = False
        ros_bridge = getattr(self, "ros_bridge", None)
        if ros_bridge is not None:
            try:
                feedback_q, feedback_dq = ros_bridge.get_joint_feedback()
            except Exception:
                feedback_q, feedback_dq = None, None
            valid_feedback_q = self._right_arm_joint_vector_or_none(feedback_q)
            if valid_feedback_q is not None:
                linearization_q = valid_feedback_q.copy()
                feedback_used = True
            valid_feedback_dq = self._right_arm_joint_vector_or_none(feedback_dq)
            if valid_feedback_dq is not None:
                measured_joint_velocity = valid_feedback_dq.copy()
        try:
            position, _quat, fk_ok = (
                self.controller._compute_racket_pose_quat_base(
                    linearization_q))
        except Exception:
            fk_ok = False
            position = None
        position_vec = (
            None if position is None
            else np.asarray(position, dtype=np.float64).reshape(-1))
        if (not fk_ok or position_vec is None
                or position_vec.shape[0] != 3
                or not np.all(np.isfinite(position_vec))):
            self.workspace_soft_filter = {
                "active": False,
                "predictive": bool(predictive),
                "model": "virtual_impedance",
                "error": "fk_unavailable",
            }
            return candidate

        margin = max(
            1e-3, float(getattr(self.cfg, "workspace_soft_margin_m", 0.05)))
        boundary_terms = self._workspace_soft_boundary_terms(position_vec)
        active_terms = [
            (name, normal, distance)
            for name, normal, distance in boundary_terms
            if float(distance) < margin
        ]
        predicted_normal = (
            self._predictive_workspace_translation_direction()
            if predictive else None)
        if not active_terms and predicted_normal is None:
            self.workspace_soft_entry_q = None
            self.workspace_soft_filter = {
                "active": False,
                "predictive": False,
                "model": "virtual_impedance",
                "marginM": round(float(margin), 6),
            }
            return candidate

        jacobian = self._workspace_translation_jacobian(
            linearization_q, position_vec)
        if jacobian is None or not np.all(np.isfinite(jacobian)):
            self.workspace_soft_filter = {
                "active": False,
                "predictive": bool(predicted_normal is not None),
                "model": "virtual_impedance",
                "error": "jacobian_unavailable",
            }
            return candidate

        proximity_strengths = [
            float(np.clip(
                (margin - float(distance)) / margin, 0.0, 1.0))
            for _name, _normal, distance in active_terms
        ]
        prediction_urgency = 0.0
        if predicted_normal is not None:
            prediction = getattr(
                getattr(self, "done_monitor", None), "last_prediction", {})
            try:
                time_to_cross = max(0.0, float(prediction.get("time_s", 0.0)))
            except (TypeError, ValueError):
                time_to_cross = 0.0
            horizon = max(
                1e-4, float(getattr(
                    self.cfg, "done_prediction_horizon_s", 0.06)))
            prediction_urgency = float(np.clip(
                1.0 - time_to_cross / horizon, 0.15, 1.0))
        field_blend = max(proximity_strengths + [prediction_urgency, 0.0])
        boundary_speed_scale = float(np.clip(
            getattr(
                self.cfg,
                "workspace_soft_boundary_joint_speed_scale",
                0.35,
            ),
            0.05,
            1.0,
        ))
        policy_velocity_scale = (
            1.0 - field_blend * (1.0 - boundary_speed_scale))
        controlled_candidate_vel = candidate_vel * policy_velocity_scale

        task_velocity = jacobian @ controlled_candidate_vel
        measured_task_velocity = (
            None
            if measured_joint_velocity is None
            else jacobian @ measured_joint_velocity)
        damping_gain = max(
            0.0, float(getattr(self.cfg, "workspace_soft_damping", 1.0)))
        pull_speed = max(
            0.0,
            float(getattr(
                self.cfg, "workspace_soft_pull_speed_m_s", 0.08)),
        )
        correction_velocity = np.zeros(3, dtype=np.float64)
        term_payload: list[dict[str, Any]] = []
        for name, normal, distance in active_terms:
            penetration = max(0.0, -float(distance))
            proximity = float(np.clip(
                (margin - float(distance)) / margin, 0.0, 1.0))
            strength = proximity + min(1.0, penetration / margin)
            outward_speed = max(
                0.0,
                float(np.dot(task_velocity, normal)),
                0.0 if measured_task_velocity is None
                else float(np.dot(measured_task_velocity, normal)),
            )
            correction_speed = (
                damping_gain * proximity * outward_speed
                + pull_speed * strength * strength)
            correction_velocity -= normal * correction_speed
            term_payload.append({
                "name": name,
                "distanceM": round(float(distance), 6),
                "strength": round(float(strength), 6),
                "outwardSpeedMps": round(float(outward_speed), 6),
                "correctionSpeedMps": round(float(correction_speed), 6),
            })

        if predicted_normal is not None:
            urgency = prediction_urgency
            outward_speed = max(
                0.0,
                float(np.dot(task_velocity, predicted_normal)),
                0.0 if measured_task_velocity is None
                else float(np.dot(
                    measured_task_velocity, predicted_normal)),
            )
            correction_speed = (
                0.5 * damping_gain * urgency * outward_speed
                + 0.25 * pull_speed * urgency * urgency)
            correction_velocity -= predicted_normal * correction_speed
            term_payload.append({
                "name": "predicted",
                "distanceM": None,
                "strength": round(float(urgency), 6),
                "outwardSpeedMps": round(float(outward_speed), 6),
                "correctionSpeedMps": round(float(correction_speed), 6),
            })

        correction_limit = max(
            1e-3,
            float(getattr(
                self.cfg,
                "workspace_soft_max_correction_speed_m_s",
                0.30,
            )),
        )
        correction_norm = float(np.linalg.norm(correction_velocity))
        if correction_norm > correction_limit:
            correction_velocity *= correction_limit / correction_norm

        dls = max(
            1e-6,
            float(getattr(self.cfg, "workspace_soft_dls_damping", 0.03)),
        )
        task_matrix = jacobian @ jacobian.T + (dls * dls) * np.eye(3)
        try:
            jacobian_pinv = (
                jacobian.T @ np.linalg.solve(task_matrix, np.eye(3)))
        except np.linalg.LinAlgError:
            jacobian_pinv = np.linalg.pinv(jacobian, rcond=dls)
        joint_correction_velocity = jacobian_pinv @ correction_velocity

        nullspace = (
            np.eye(len(RIGHT_ARM_JOINTS)) - jacobian_pinv @ jacobian)
        nullspace_velocity = nullspace @ controlled_candidate_vel
        nullspace_damping = float(np.clip(
            getattr(self.cfg, "workspace_soft_nullspace_damping", 0.85),
            0.0,
            1.0,
        ))
        entry_q = getattr(self, "workspace_soft_entry_q", None)
        if entry_q is None:
            entry_q = prev_cmd.copy()
            self.workspace_soft_entry_q = entry_q.copy()
        entry_q = np.asarray(entry_q, dtype=np.float64).reshape(-1)
        posture_gain = max(
            0.0,
            float(getattr(
                self.cfg, "workspace_soft_posture_gain_s", 2.0)),
        )
        nullspace_posture_error = nullspace @ (prev_cmd - entry_q)
        joint_field_velocity = (
            joint_correction_velocity
            - field_blend * nullspace_damping * nullspace_velocity
            - field_blend * posture_gain * nullspace_posture_error)
        joint_field_velocity = np.clip(
            joint_field_velocity,
            -0.5 * self.safety_limiter.vel_limit_rad_s,
            0.5 * self.safety_limiter.vel_limit_rad_s,
        )

        combined_target = prev_cmd + self.safety_limiter.dt * (
            controlled_candidate_vel + joint_field_velocity)
        self.safety_limiter.last_cmd = prev_cmd.copy()
        self.safety_limiter.last_vel = prev_vel.copy()
        safe = self.safety_limiter.filter(combined_target, profile=profile)
        self.workspace_soft_filter = {
            "active": True,
            "predictive": bool(predicted_normal is not None),
            "model": "virtual_impedance",
            "marginM": round(float(margin), 6),
            "positionM": _rounded_vector(position_vec),
            "feedbackUsed": bool(feedback_used),
            "policyVelocityScale": round(float(policy_velocity_scale), 6),
            "boundaryJointSpeedScale": round(
                float(boundary_speed_scale), 6),
            "candidateJointVelocityRadS": _rounded_vector(candidate_vel),
            "controlledJointVelocityRadS": _rounded_vector(
                controlled_candidate_vel),
            "taskVelocityMps": _rounded_vector(task_velocity),
            "measuredTaskVelocityMps": (
                None if measured_task_velocity is None
                else _rounded_vector(measured_task_velocity)),
            "correctionVelocityMps": _rounded_vector(correction_velocity),
            "jointCorrectionVelocityRadS": _rounded_vector(
                joint_correction_velocity),
            "jointFieldVelocityRadS": _rounded_vector(joint_field_velocity),
            "nullspaceVelocityRadS": _rounded_vector(nullspace_velocity),
            "nullspaceDamping": round(float(nullspace_damping), 6),
            "postureGainS": round(float(posture_gain), 6),
            "terms": term_payload,
            "candidateScore": round(float(
                self._parking_workspace_violation_score(candidate)), 9),
            "selectedScore": round(float(
                self._parking_workspace_violation_score(safe)), 9),
        }
        return safe

    def _legacy_filter_command_with_workspace_guard(
        self,
        raw: np.ndarray,
        *,
        profile: str,
        apply_joint_limits: bool,
        predictive_hold: bool = False,
    ) -> np.ndarray:
        """Keep a joint command near the feasible arm-workspace edge.

        A per-joint projection is preferred over scaling the whole 7-DOF
        displacement.  Scaling the whole displacement freezes valid inward or
        tangential motion whenever only one component points out of the
        workspace.  The component projection removes only the components
        needed to satisfy the boundary and therefore lets the policy move
        along, or return from, the edge.  Every input to this method from a
        soft runtime mode is rate limited before the workspace projection.
        """
        prev_cmd = self.safety_limiter.last_cmd.copy()
        prev_score = self._parking_workspace_violation_score(prev_cmd)
        desired = np.asarray(raw, dtype=np.float64).reshape(-1)
        if (desired.shape[0] != len(RIGHT_ARM_JOINTS)
                or not np.all(np.isfinite(desired))):
            desired = prev_cmd.copy()
        desired = np.clip(
            desired,
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        )
        predictive_blocked: list[int] = []
        if predictive_hold and prev_score <= 1e-12:
            desired, predictive_blocked = (
                self._remove_predictive_outward_components(
                    prev_cmd, desired))
        if apply_joint_limits:
            candidate = self.safety_limiter.filter(desired, profile=profile)
        else:
            candidate = desired.copy()
        candidate_score = self._parking_workspace_violation_score(candidate)
        score_eps = 1e-12
        # If startup/feedback already placed the command just outside the
        # configured envelope, permit motion that does not make the violation
        # worse.  This is the soft-mode recovery behaviour: stay near the edge
        # until the policy points inward, instead of latching done or freezing.
        score_ceiling = (
            float(prev_score) if prev_score > score_eps else score_eps)
        candidate_allowed = bool(
            np.isfinite(candidate_score)
            and candidate_score <= score_ceiling + score_eps)
        if candidate_allowed:
            if not apply_joint_limits:
                self.safety_limiter.observe(candidate)
            self.workspace_soft_filter = {
                "active": bool(
                    predictive_hold or candidate_score > score_eps),
                "intervened": bool(predictive_blocked),
                "predictive": bool(predictive_hold),
                "predictiveBlockedJoints": predictive_blocked,
                "alpha": 1.0,
                "projection": "none",
                "recovering": bool(
                    prev_score > score_eps
                    and candidate_score < prev_score - score_eps),
                "previousScore": round(float(prev_score), 9),
                "candidateScore": round(float(candidate_score), 9),
            }
            return candidate

        delta = candidate - prev_cmd
        selected = prev_cmd.copy()
        alpha_lo = 0.0
        projection = "segment"

        # First try to preserve most joint components.  Starting from the
        # candidate, greedily remove at most three components that contribute
        # most to the violation, then restore as much of each removed component
        # as the boundary permits.  This retains tangential/inward motion and
        # has a bounded number of FK calls in the 200 Hz control path.
        working = candidate.copy()
        working_score = float(candidate_score)
        remaining = [
            idx for idx, value in enumerate(delta)
            if abs(float(value)) > 1e-15
        ]
        removed: list[int] = []
        for _ in range(min(3, len(remaining))):
            best_idx: int | None = None
            best_trial: np.ndarray | None = None
            best_score = working_score
            for idx in remaining:
                trial = working.copy()
                trial[idx] = prev_cmd[idx]
                trial_score = self._parking_workspace_violation_score(trial)
                if (np.isfinite(trial_score)
                        and trial_score < best_score - score_eps):
                    best_idx = idx
                    best_trial = trial
                    best_score = float(trial_score)
            if best_idx is None or best_trial is None:
                break
            working = best_trial
            working_score = best_score
            remaining.remove(best_idx)
            removed.append(best_idx)
            if working_score <= score_ceiling + score_eps:
                selected = working.copy()
                projection = "component"
                # Re-introduce each removed component up to its local feasible
                # boundary. Six iterations are sufficient because the original
                # rate-limited 200 Hz step is already small.
                for idx in reversed(removed):
                    base = selected.copy()
                    full = selected.copy()
                    full[idx] = candidate[idx]
                    full_score = self._parking_workspace_violation_score(full)
                    if (np.isfinite(full_score)
                            and full_score <= score_ceiling + score_eps):
                        selected = full
                        continue
                    lo = 0.0
                    hi = 1.0
                    for _ in range(6):
                        fraction = 0.5 * (lo + hi)
                        trial = base.copy()
                        trial[idx] = (
                            base[idx]
                            + fraction * (candidate[idx] - base[idx]))
                        trial_score = self._parking_workspace_violation_score(
                            trial)
                        if (np.isfinite(trial_score)
                                and trial_score <= score_ceiling + score_eps):
                            lo = fraction
                            selected = trial
                        else:
                            hi = fraction
                break

        if projection == "segment":
            # General fallback for coupled/non-smooth constraints.  Unlike the
            # old implementation this also permits a non-worsening local step
            # when the previous command is already outside the envelope.
            alpha_hi = 1.0
            for _ in range(10):
                alpha = 0.5 * (alpha_lo + alpha_hi)
                trial = prev_cmd + alpha * delta
                trial_score = self._parking_workspace_violation_score(trial)
                if (np.isfinite(trial_score)
                        and trial_score <= score_ceiling + score_eps):
                    alpha_lo = alpha
                    selected = trial
                else:
                    alpha_hi = alpha

        nonzero = np.abs(delta) > 1e-15
        retained_fraction = 0.0
        if np.any(nonzero):
            retained_fraction = float(np.linalg.norm(selected - prev_cmd)
                                      / max(np.linalg.norm(delta), 1e-15))
            component_fractions = np.clip(
                (selected[nonzero] - prev_cmd[nonzero]) / delta[nonzero],
                0.0,
                1.0,
            )
            alpha_lo = float(np.min(component_fractions))
        self.safety_limiter.last_cmd = selected.copy()
        self.safety_limiter.last_vel = (
            (selected - prev_cmd) / self.safety_limiter.dt)
        self.workspace_soft_filter = {
            "active": True,
            "intervened": True,
            "predictive": bool(predictive_hold),
            "predictiveBlockedJoints": predictive_blocked,
            "alpha": round(float(alpha_lo), 6),
            "retainedFraction": round(float(retained_fraction), 6),
            "projection": projection,
            "recovering": bool(prev_score > score_eps),
            "previousScore": round(float(prev_score), 9),
            "candidateScore": round(float(candidate_score), 9),
        }
        return selected.copy()

    def _workspace_filter_feedback(
        self,
        fallback_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        """Return the freshest finite arm feedback available to safety."""
        q = np.asarray(fallback_q, dtype=np.float64).reshape(-1).copy()
        dq: np.ndarray | None = None
        feedback_used = False
        feedback_age_s: float | None = None
        feedback_age_supported = False
        feedback_stale = False
        velocity_source = "unavailable"
        ros_bridge = getattr(self, "ros_bridge", None)
        if ros_bridge is not None:
            try:
                age_getter = getattr(
                    ros_bridge, "get_joint_feedback_with_age", None)
                if callable(age_getter):
                    feedback_age_supported = True
                    (feedback_q, feedback_dq, feedback_age_s,
                     velocity_source) = age_getter()
                else:
                    feedback_q, feedback_dq = ros_bridge.get_joint_feedback()
            except Exception:
                feedback_q, feedback_dq = None, None
                if feedback_age_supported:
                    feedback_stale = True
            if feedback_age_supported:
                freshness_limit_s = max(
                    0.025, 3.0 * float(self.safety_limiter.dt))
                feedback_stale = bool(
                    feedback_age_s is None
                    or not math.isfinite(float(feedback_age_s))
                    or float(feedback_age_s) > freshness_limit_s)
            valid_q = self._right_arm_joint_vector_or_none(feedback_q)
            valid_dq = self._right_arm_joint_vector_or_none(feedback_dq)
            if valid_q is not None and not feedback_stale:
                q = valid_q.copy()
                feedback_used = True
            if valid_dq is not None and not feedback_stale:
                dq = valid_dq.copy()
        self._workspace_feedback_age_s = feedback_age_s
        self._workspace_feedback_stale = bool(feedback_stale)
        self._workspace_feedback_age_supported = bool(feedback_age_supported)
        self._workspace_feedback_velocity_source = str(velocity_source)
        return q, dq, feedback_used

    def _workspace_dynamic_velocity_bounds(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        profile: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Joint speed/acceleration/position limits for the next control tick."""
        velocity_limit, acceleration_limit = (
            self.safety_limiter.profile_limits(profile))
        dt = max(float(self.safety_limiter.dt), 1e-9)
        lower = np.maximum.reduce((
            -velocity_limit,
            np.asarray(qdot, dtype=np.float64) - acceleration_limit * dt,
            (RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD - q) / dt,
        ))
        upper = np.minimum.reduce((
            velocity_limit,
            np.asarray(qdot, dtype=np.float64) + acceleration_limit * dt,
            (RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD - q) / dt,
        ))
        return lower, upper, velocity_limit

    def _workspace_nonlinear_backstop(
        self,
        previous: np.ndarray,
        selected: np.ndarray,
        *,
        allowed_penetration_m: float = 0.001,
    ) -> tuple[np.ndarray, bool]:
        """Keep nonlinear FK error from turning a local step into a large exit.

        The CBF/QP is linearized at the measured pose.  Box faces are exact,
        while the spherical anchor boundary has curvature.  This final bounded
        line search is normally inactive; it only catches a command predicted
        to cross the outer boundary by more than one millimetre in one tick.
        """
        previous_state = self._workspace_hold_state(previous)
        selected_state = self._workspace_hold_state(selected)
        if (not previous_state.get("valid", False)
                or not selected_state.get("valid", False)):
            return previous.copy(), True
        previous_distance = float(previous_state.get(
            "minDistanceM", float("-inf")))
        selected_distance = float(selected_state.get(
            "minDistanceM", float("-inf")))
        distance_floor = (
            -abs(float(allowed_penetration_m))
            if previous_distance >= -abs(float(allowed_penetration_m))
            else previous_distance)
        if selected_distance >= distance_floor - 1e-9:
            return selected.copy(), False

        low = 0.0
        high = 1.0
        safe = previous.copy()
        delta = selected - previous
        for _ in range(14):
            alpha = 0.5 * (low + high)
            trial = previous + alpha * delta
            trial_state = self._workspace_hold_state(trial)
            trial_distance = float(trial_state.get(
                "minDistanceM", float("-inf")))
            if trial_state.get("valid", False) and trial_distance >= distance_floor:
                low = alpha
                safe = trial
            else:
                high = alpha
        return safe.copy(), True

    def _filter_command_with_workspace_cbf(
        self,
        raw: np.ndarray,
        *,
        profile: str,
        mode: str,
        apply_joint_limits: bool = True,
    ) -> np.ndarray:
        """Minimally modify only unsafe Cartesian outward velocity.

        All three soft modes share this measured-state, discrete velocity-CBF
        filter.  ``position_soft`` uses current signed distance,
        ``predictive_soft`` reserves delay/horizon travel, and
        ``impedance_soft`` continuously damps the outward normal component in
        the influence band.  A joint-space QP projection handles simultaneous
        faces while retaining tangential and redundant policy motion.
        """
        mode_name = _workspace_safety_mode(mode, "position_soft")
        if mode_name not in WORKSPACE_SOFT_MODES:
            mode_name = "position_soft"
        predictive = mode_name == "predictive_soft"
        impedance = mode_name == "impedance_soft"
        prev_cmd = self.safety_limiter.last_cmd.copy()
        prev_vel = self.safety_limiter.last_vel.copy()
        dt = max(float(self.safety_limiter.dt), 1e-9)
        self._workspace_antiwindup_position_correction = None
        self._workspace_antiwindup_policy_velocity = None

        desired = np.asarray(raw, dtype=np.float64).reshape(-1)
        invalid_input = bool(
            desired.shape[0] != len(RIGHT_ARM_JOINTS)
            or not np.all(np.isfinite(desired)))
        if invalid_input:
            desired = prev_cmd.copy()
        desired = np.clip(
            desired,
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        )

        # Preview the dynamically reachable reference exactly once.  Restore
        # state before committing the QP result so no post-projection overwrite
        # can bypass the acceleration limiter as the old implementation did.
        if apply_joint_limits:
            candidate = self.safety_limiter.filter(desired, profile=profile)
            candidate_vel = self.safety_limiter.last_vel.copy()
            self.safety_limiter.last_cmd = prev_cmd.copy()
            self.safety_limiter.last_vel = prev_vel.copy()
        else:
            candidate = desired.copy()
            candidate_vel = (candidate - prev_cmd) / dt
        self._workspace_soft_candidate_q = candidate.copy()
        self._workspace_soft_candidate_vel = candidate_vel.copy()

        feedback_q, feedback_dq, feedback_used = (
            self._workspace_filter_feedback(prev_cmd))
        feedback_stale = bool(getattr(
            self, "_workspace_feedback_stale", False))
        feedback_age_s = getattr(self, "_workspace_feedback_age_s", None)
        if bool(getattr(
                self, "_workspace_feedback_age_supported", False)
                and feedback_stale):
            # No fresh plant state means no valid Cartesian linearization.
            # Smoothly brake the already-published command stream instead of
            # continuing policy on stale q/dq or jumping to a recovery pose.
            if apply_joint_limits:
                safe = self.safety_limiter.filter(prev_cmd, profile=profile)
            else:
                safe = prev_cmd.copy()
                self.safety_limiter.observe(safe)
            self.workspace_soft_filter = {
                "active": True,
                "intervened": True,
                "predictive": bool(predictive),
                "model": "cartesian_velocity_cbf",
                "error": "joint_feedback_stale",
                "emergency": True,
                "feedbackAgeS": feedback_age_s,
                "feedbackVelocitySource": str(getattr(
                    self, "_workspace_feedback_velocity_source", "unavailable")),
                "invalidInput": invalid_input,
            }
            self.workspace_soft_emergency_active = True
            self.workspace_soft_emergency_reason = "feedback_stale"
            return safe
        try:
            position, base_quat, fk_ok = (
                self.controller._compute_racket_pose_quat_base(feedback_q))
        except Exception:
            position, base_quat, fk_ok = None, None, False
        position_vec = (
            None if position is None
            else np.asarray(position, dtype=np.float64).reshape(-1))
        if (not fk_ok or position_vec is None
                or position_vec.shape[0] != 3
                or not np.all(np.isfinite(position_vec))):
            # FK is part of the safety proof.  A missing model cannot justify
            # passing a new target, so brake toward the last published command.
            if apply_joint_limits:
                safe = self.safety_limiter.filter(prev_cmd, profile=profile)
            else:
                safe = prev_cmd.copy()
                self.safety_limiter.observe(safe)
            self.workspace_soft_filter = {
                "active": True,
                "intervened": True,
                "predictive": bool(predictive),
                "model": "cartesian_velocity_cbf",
                "error": "fk_unavailable",
                "invalidInput": invalid_input,
            }
            self.workspace_soft_emergency_active = True
            self.workspace_soft_emergency_reason = "fk_unavailable"
            return safe

        # The optimization variable is the published reference velocity, so
        # its exact kinematic state is ``prev_cmd`` rather than the delayed
        # measured q.  The measured pose remains a second plant-envelope
        # constraint below.  This prevents a 72 ms lead in the command queue
        # from reaching the wall while feedback still appears far away.
        command_linearization = self._workspace_pose_linearization(prev_cmd)
        if command_linearization is None:
            if apply_joint_limits:
                safe = self.safety_limiter.filter(prev_cmd, profile=profile)
            else:
                safe = prev_cmd.copy()
                self.safety_limiter.observe(safe)
            self.workspace_soft_filter = {
                "active": True,
                "intervened": True,
                "predictive": bool(predictive),
                "model": "cartesian_velocity_cbf",
                "error": "jacobian_unavailable",
                "invalidInput": invalid_input,
            }
            self.workspace_soft_emergency_active = True
            self.workspace_soft_emergency_reason = "jacobian_unavailable"
            return safe
        (command_position_vec, _command_quat,
         jacobian, angular_jacobian) = command_linearization

        if apply_joint_limits:
            lower, upper, velocity_scale = (
                self._workspace_dynamic_velocity_bounds(
                    prev_cmd, prev_vel, profile))
        else:
            lower = upper = None
            velocity_scale = self.safety_limiter.vel_limit_rad_s
        _velocity_limit, acceleration_limit = (
            self.safety_limiter.profile_limits(profile))

        measured_vel = (
            prev_vel.copy() if feedback_dq is None else feedback_dq.copy())
        margin = max(
            1e-3, float(getattr(self.cfg, "workspace_soft_margin_m", 0.05)))
        max_approach_speed = max(
            1e-3,
            float(getattr(
                self.cfg,
                "workspace_soft_max_correction_speed_m_s",
                0.30,
            )),
        )
        recovery_speed = max(
            0.0,
            float(getattr(
                self.cfg, "workspace_soft_pull_speed_m_s", 0.08)),
        )
        cbf_gain = max_approach_speed / margin
        damping = float(np.clip(
            getattr(self.cfg, "workspace_soft_damping", 1.0), 0.0, 1.5))

        prediction_horizon = 0.0
        prediction_margin = 0.0
        prediction_signal = False
        predicted_constraint = ""
        predicted_distance: float | None = None
        if predictive:
            monitor = getattr(self, "done_monitor", None)
            prediction_horizon = max(
                0.0,
                float(getattr(
                    monitor,
                    "prediction_horizon_s",
                    getattr(self.cfg, "done_prediction_horizon_s", 0.06),
                )),
            )
            prediction_margin = max(
                0.0,
                float(getattr(
                    monitor,
                    "prediction_margin_m",
                    getattr(self.cfg, "done_prediction_margin_m", 0.005),
                )),
            )
            last_prediction = getattr(monitor, "last_prediction", None)
            prediction_signal = bool(
                feedback_dq is not None
                or getattr(self, "workspace_soft_prediction_active", False)
                or (isinstance(last_prediction, dict)
                    and bool(last_prediction.get("condition"))))
            predicted_constraint = self._predictive_workspace_constraint()
            if predicted_constraint and isinstance(last_prediction, dict):
                try:
                    predicted_position = np.asarray(
                        last_prediction.get("value"),
                        dtype=np.float64,
                    ).reshape(-1)[:3]
                except (TypeError, ValueError):
                    predicted_position = np.empty(0, dtype=np.float64)
                if (predicted_position.shape[0] == 3
                        and np.all(np.isfinite(predicted_position))):
                    for predicted_name, _normal, distance in (
                            self._workspace_soft_boundary_terms(
                                predicted_position)):
                        if str(predicted_name) == predicted_constraint:
                            predicted_distance = float(distance)
                            break

        # The candidate is one dynamically bounded velocity step from
        # ``prev_cmd``; its safety is represented directly by the Jacobian
        # rows below.  Avoid a redundant FK here in the 200 Hz path.
        candidate_distances: dict[str, float] = {}

        feedback_boundary_terms = self._workspace_soft_boundary_terms(
            position_vec)
        feedback_distances = {
            str(name): float(distance)
            for name, _normal, distance in feedback_boundary_terms
        }
        boundary_terms = self._workspace_soft_boundary_terms(
            command_position_vec)
        minimum_distance = min(
            [float(item[2]) for item in boundary_terms]
            + [float(item[2]) for item in feedback_boundary_terms],
            default=float("inf"),
        )
        hard_envelope_m = min(0.010, max(0.005, 0.20 * margin))
        hard_envelope_violation = bool(
            minimum_distance < -hard_envelope_m)
        emergency_active = bool(getattr(
            self, "workspace_soft_emergency_active", False))
        emergency_reason = str(getattr(
            self, "workspace_soft_emergency_reason", ""))
        if hard_envelope_violation:
            emergency_active = True
            emergency_reason = "hard_envelope"
        elif (emergency_active
              and emergency_reason == "hard_envelope"
              and minimum_distance >= min(0.005, 0.10 * margin)):
            emergency_active = False
            emergency_reason = ""

        configured_delay_s = max(
            0.0,
            float(getattr(
                self,
                "_workspace_actuation_delay_s",
                max(0.0, float(getattr(
                    self.cfg, "comp_delay_ms", 0.0))) * 1e-3,
            )),
        )
        actuator_tau_s = max(
            0.0,
            float(getattr(
                self,
                "_workspace_actuator_tau_s",
                getattr(self.cfg, "comp_actuator_tau_s", 0.0),
            )),
        )
        # Keep a 120 ms sampled-data/model uncertainty window (or a larger
        # configured prediction horizon).  This is required even for
        # position_soft: delay + first-order lag is an average response model,
        # not a worst-case bound on queued targets and controller jitter.
        sampled_data_preview_s = max(
            0.12,
            float(getattr(
                self.cfg, "done_prediction_horizon_s", 0.06)))
        plant_response_s = (
            configured_delay_s + actuator_tau_s + dt
            + sampled_data_preview_s)

        rows: list[np.ndarray] = []
        velocity_bounds: list[float] = []
        hard_cbf_bounds: list[float] = []
        impedance_comfort_bounds: list[float] = []
        antiwindup_bounds: list[float] = []
        term_payload: list[dict[str, Any]] = []
        any_bound_relaxed = False
        for name, normal, distance in boundary_terms:
            row = np.asarray(normal, dtype=np.float64) @ jacobian
            nominal_outward = float(np.dot(row, candidate_vel))
            measured_outward = float(np.dot(row, measured_vel))
            command_outward = float(np.dot(row, prev_vel))
            # Reserve only motion already present in the plant/published
            # stream.  ``candidate_vel`` is still a proposal and will be
            # constrained by this very QP; counting an arbitrarily distant
            # target as unavoidable travel would command a spurious inward
            # jump before the arm had begun moving.
            outward_speed = max(
                0.0, measured_outward, command_outward)
            raw_normal_acceleration = float(np.sum(
                np.abs(row) * np.asarray(
                    acceleration_limit, dtype=np.float64)))
            # Sum(|nJ| a_joint) is an optimistic single-face capability; it
            # cannot generally be achieved while respecting another face and
            # preserving racket twist.  Use a conservative Cartesian braking
            # capability for the safe-speed envelope, while the QP still uses
            # the exact per-joint acceleration box.
            normal_acceleration = min(
                raw_normal_acceleration,
                max(0.25, float(getattr(
                    self.cfg,
                    "workspace_soft_braking_acceleration_m_s2",
                    0.5,
                ))),
            )
            stopping_distance = (
                0.0 if outward_speed <= 0.0
                else outward_speed * outward_speed
                / (2.0 * max(normal_acceleration, 1e-6)))
            # All soft modes reserve the travel that is already unavoidable
            # while a newly filtered reference crosses the transport delay and
            # first-order actuator lag.  Predictive mode adds policy/model
            # look-ahead on top; it never replaces the physical reserve.
            plant_reserve = plant_response_s * outward_speed
            predictive_reserve = 0.0
            if predictive and prediction_signal:
                predictive_reserve = (
                    prediction_margin
                    + prediction_horizon * max(
                        outward_speed,
                        min(max_approach_speed, max(0.0, nominal_outward)),
                    ))
            reserve = plant_reserve + predictive_reserve
            feedback_distance = feedback_distances.get(
                str(name), float(distance))
            effective_distance = min(
                float(distance), float(feedback_distance) - reserve)
            if (str(name) == predicted_constraint
                    and predicted_distance is not None):
                effective_distance = min(
                    effective_distance,
                    float(predicted_distance) - prediction_margin,
                )
            candidate_distance = candidate_distances.get(
                str(name), float("inf"))
            if effective_distance >= margin:
                # Braking-feasible velocity damper: outside the influence band
                # allow high policy speed only while enough distance remains
                # to decelerate smoothly to max_approach_speed at the band.
                requested_cbf_bound = math.sqrt(max(
                    0.0,
                    max_approach_speed * max_approach_speed
                    + 2.0 * normal_acceleration
                    * (effective_distance - margin),
                ))
            else:
                requested_cbf_bound = float(np.clip(
                    cbf_gain * effective_distance,
                    -recovery_speed,
                    max_approach_speed,
                ))
            if (effective_distance >= margin
                    and candidate_distance >= margin
                    and nominal_outward <= requested_cbf_bound + 1e-9):
                continue
            reachable_min_outward = float("-inf")
            cbf_bound = requested_cbf_bound
            if lower is not None and upper is not None:
                reachable_min_outward = float(np.sum(np.where(
                    row >= 0.0, row * lower, row * upper)))
                if cbf_bound < reachable_min_outward:
                    cbf_bound = reachable_min_outward
                    any_bound_relaxed = True
            selected_bound = cbf_bound
            strength = _smoothstep01(
                (margin - float(distance)) / margin)
            spring_speed = 0.0
            if impedance:
                penetration = max(0.0, -float(distance))
                spring_speed = min(
                    recovery_speed,
                    cbf_gain * penetration,
                )
                # Comfort damping acts only on the candidate's outward normal
                # component.  Reacting again to delayed measured velocity can
                # over-correct an already-inward command and steer a redundant
                # arm toward another face; measured motion is already covered
                # by the plant reserve in the hard CBF bound.
                outward_reference = max(0.0, nominal_outward)
                requested_impedance_bound = (
                    nominal_outward
                    - damping * strength * outward_reference
                    - spring_speed)
                # The virtual wall is a comfort layer.  It may request more
                # inward speed than the CBF, but it may never request an
                # acceleration-discontinuous command.
                selected_bound = min(cbf_bound, requested_impedance_bound)
                if math.isfinite(reachable_min_outward):
                    if selected_bound < reachable_min_outward:
                        any_bound_relaxed = True
                    selected_bound = max(
                        selected_bound, reachable_min_outward)

            # An inward/sufficiently slow proposal already satisfies this
            # face.  Omitting the no-op inequality avoids over-constraining
            # corners and preserves tangential policy performance.  A
            # proactive impedance spring remains because its bound is strictly
            # below nominal_outward inside the influence band.
            if nominal_outward <= selected_bound + 1e-9:
                continue

            rows.append(np.asarray(row, dtype=np.float64))
            # The optimizer always receives the reachable hard CBF envelope.
            # ``selected_bound`` can be tighter in impedance mode, but that is
            # a best-effort comfort request applied only after hard safety is
            # feasible.  A virtual spring must never create a solver emergency
            # or a different recovery trajectory.
            velocity_bounds.append(float(cbf_bound))
            hard_cbf_bounds.append(float(cbf_bound))
            impedance_comfort_bounds.append(float(selected_bound))
            # Policy anti-windup is allowed to remove rejected outward motion,
            # never to integrate proactive CBF/impedance inward motion.
            antiwindup_bounds.append(max(0.0, float(cbf_bound)))
            term_payload.append({
                "name": str(name),
                "distanceM": round(float(distance), 6),
                "commandDistanceM": round(float(distance), 6),
                "measuredDistanceM": round(float(feedback_distance), 6),
                "effectiveDistanceM": round(float(effective_distance), 6),
                "reserveM": round(float(reserve), 6),
                "plantReserveM": round(float(plant_reserve), 6),
                "stoppingDistanceM": round(float(stopping_distance), 6),
                "predictiveReserveM": round(float(predictive_reserve), 6),
                "candidateDistanceM": (
                    None if not math.isfinite(candidate_distance)
                    else round(float(candidate_distance), 6)),
                "nominalOutwardMps": round(float(nominal_outward), 6),
                "measuredOutwardMps": round(float(measured_outward), 6),
                "commandOutwardMps": round(float(command_outward), 6),
                "normalAccelerationMps2": round(
                    float(normal_acceleration), 6),
                "rawNormalAccelerationMps2": round(
                    float(raw_normal_acceleration), 6),
                "requestedCbfBoundMps": round(
                    float(requested_cbf_bound), 6),
                "reachableMinOutwardMps": (
                    None if not math.isfinite(reachable_min_outward)
                    else round(float(reachable_min_outward), 6)),
                "boundMps": round(float(cbf_bound), 6),
                "comfortBoundMps": round(float(selected_bound), 6),
                "strength": round(float(strength), 6),
                "springSpeedMps": round(float(spring_speed), 6),
            })

        if not rows:
            if emergency_reason != "hard_envelope":
                self.workspace_soft_emergency_active = False
                self.workspace_soft_emergency_reason = ""
            if apply_joint_limits:
                self.safety_limiter.last_cmd = candidate.copy()
                self.safety_limiter.last_vel = candidate_vel.copy()
            else:
                self.safety_limiter.observe(candidate)
            self.workspace_soft_filter = {
                "active": False,
                "intervened": False,
                "predictive": bool(predictive),
                "model": (
                    "predictive_cartesian_cbf" if predictive
                    else "virtual_impedance_cbf" if impedance
                    else "cartesian_velocity_cbf"),
                "marginM": round(float(margin), 6),
                "feedbackUsed": bool(feedback_used),
                "feedbackAgeS": feedback_age_s,
                "feedbackVelocitySource": str(getattr(
                    self, "_workspace_feedback_velocity_source", "unavailable")),
                "invalidInput": invalid_input,
                "alpha": 1.0,
                "projection": "none",
            }
            return candidate.copy()

        row_matrix = np.asarray(rows, dtype=np.float64)
        bound_vector = np.asarray(velocity_bounds, dtype=np.float64)
        hard_cbf_bound_vector = np.asarray(
            hard_cbf_bounds, dtype=np.float64)
        impedance_comfort_bound_vector = np.asarray(
            impedance_comfort_bounds, dtype=np.float64)
        antiwindup_bound_vector = np.asarray(
            antiwindup_bounds, dtype=np.float64)
        # Preserve the whole Cartesian twist, not arbitrary joint components.
        # Translation has higher priority; angular tracking prevents the wrist
        # from rotating the racket to obtain a cheap positional correction.
        translation_scale_m_s = max(0.15, max_approach_speed)
        angular_scale_rad_s = 1.0
        joint_regularizer = 0.001 * np.diag(
            1.0 / np.maximum(velocity_scale, 1e-6) ** 2)
        objective_metric = (
            (jacobian.T @ jacobian) / translation_scale_m_s ** 2
            + (angular_jacobian.T @ angular_jacobian)
            / angular_scale_rad_s ** 2
            + joint_regularizer)
        optimization_nominal = candidate_vel
        if emergency_active:
            if lower is not None and upper is not None:
                # Monotonically brake the published joint-velocity stream to
                # zero through the acceleration box.  Chasing delayed feedback
                # here creates a positive-delay loop and can launch the target
                # toward the opposite face; zero-velocity braking is local and
                # contains no global recovery pose.
                optimization_nominal = np.clip(
                    np.zeros_like(candidate_vel),
                    lower,
                    upper,
                )
            else:
                optimization_nominal = np.zeros_like(candidate_vel)
        selected_vel, qp_info = _project_velocity_halfspaces(
            optimization_nominal,
            row_matrix,
            bound_vector,
            lower=lower,
            upper=upper,
            scale=velocity_scale,
            metric=objective_metric,
            max_sweeps=16,
        )
        impedance_comfort_relaxed = False
        if (not qp_info.get("feasible", False)
                and not qp_info.get("uncontrollable", False)):
            # A nearly singular task metric can make alternating projections
            # converge slowly even when the safety set is feasible.  Retry the
            # same (unrelaxed) constraints in well-conditioned scaled joint
            # coordinates.  Safety/dynamics win over objective optimality; the
            # task metric is retained on all ordinary one/two-face ticks.
            task_metric_info = dict(qp_info)
            fallback_vel, fallback_info = _project_velocity_halfspaces(
                optimization_nominal,
                row_matrix,
                bound_vector,
                lower=lower,
                upper=upper,
                scale=velocity_scale,
                max_sweeps=48,
            )
            if fallback_info.get("feasible", False):
                selected_vel = fallback_vel
                qp_info = dict(fallback_info)
                qp_info["taskMetricAttempt"] = task_metric_info
                qp_info["objectiveFallback"] = "scaled_joint_feasible"

        if not bool(qp_info.get("feasible", False)):
            # Never publish an intermediate iterate from an infeasible solve.
            # Its half-space violation can be large and, repeated at a corner,
            # moves the target toward an unrelated workspace face.  The only
            # universally local response is maximum reachable braking of the
            # already-published reference stream.  It obeys the joint dynamic
            # box on this very tick and the emergency latch keeps braking on
            # following ticks; the expanded outer envelope remains the final
            # plant-level containment signal.
            if lower is not None and upper is not None:
                selected_vel = np.clip(
                    np.zeros_like(candidate_vel), lower, upper)
            else:
                selected_vel = np.zeros_like(candidate_vel)
            qp_info = dict(qp_info)
            qp_info["publishedFallback"] = "reachable_zero_velocity_brake"

        impedance_comfort_applied = False
        if (impedance
                and bool(qp_info.get("feasible", False))
                and not emergency_active):
            # Apply the virtual spring/damper as a cheap best-effort projection
            # from the already safe solution.  Every accepted result is
            # rechecked against acceleration/velocity boxes and the hard CBF;
            # when a corner makes the comfort field incompatible, retain the
            # hard solution unchanged.  This preserves local policy motion and
            # makes comfort infeasibility incapable of selecting a far target.
            comfort_vel = selected_vel.copy()
            for _ in range(2):
                changed = False
                for comfort_row, comfort_bound in zip(
                        row_matrix, impedance_comfort_bound_vector):
                    comfort_violation = float(
                        np.dot(comfort_row, comfort_vel) - comfort_bound)
                    if comfort_violation <= 1e-9:
                        continue
                    try:
                        metric_direction = np.linalg.solve(
                            objective_metric, comfort_row)
                    except np.linalg.LinAlgError:
                        metric_direction = np.linalg.pinv(
                            objective_metric) @ comfort_row
                    denominator = float(
                        np.dot(comfort_row, metric_direction))
                    if denominator <= 1e-12:
                        impedance_comfort_relaxed = True
                        continue
                    comfort_vel -= (
                        comfort_violation / denominator) * metric_direction
                    changed = True
                if not changed:
                    break
            comfort_dynamic_ok = bool(
                (lower is None
                 or np.all(comfort_vel >= lower - 1e-8))
                and (upper is None
                     or np.all(comfort_vel <= upper + 1e-8)))
            comfort_hard_ok = bool(np.all(
                row_matrix @ comfort_vel
                <= hard_cbf_bound_vector + 1e-7))
            if comfort_dynamic_ok and comfort_hard_ok:
                impedance_comfort_applied = bool(
                    np.linalg.norm(comfort_vel - selected_vel) > 1e-10)
                selected_vel = comfort_vel
                impedance_comfort_relaxed = bool(np.any(
                    row_matrix @ selected_vel
                    > impedance_comfort_bound_vector + 1e-7))
            else:
                impedance_comfort_relaxed = True

        # Separate policy anti-windup from the published comfort field.  Solve
        # on the policy integrator velocity itself (not the compensated target
        # tracking error), so a zero-action policy remains exactly unchanged
        # while an impedance spring acts only at the actuator-reference layer.
        policy_vel = self._right_arm_joint_vector_or_none(
            getattr(self.controller, "arm_cmd_qvel", None))
        if policy_vel is not None:
            antiwindup_policy_vel, policy_aw_info = (
                _project_velocity_halfspaces(
                    policy_vel,
                    row_matrix,
                    antiwindup_bound_vector,
                    scale=velocity_scale,
                    metric=objective_metric,
                    max_sweeps=24,
                ))
            if (not policy_aw_info.get("feasible", False)
                    and not policy_aw_info.get("uncontrollable", False)):
                fallback_policy_vel, fallback_aw_info = (
                    _project_velocity_halfspaces(
                        policy_vel,
                        row_matrix,
                        antiwindup_bound_vector,
                        scale=velocity_scale,
                        max_sweeps=24,
                    ))
                if fallback_aw_info.get("feasible", False):
                    antiwindup_policy_vel = fallback_policy_vel
            self._workspace_antiwindup_policy_velocity = (
                antiwindup_policy_vel)
            self._workspace_antiwindup_position_correction = (
                (antiwindup_policy_vel - policy_vel) * dt)
        selected = np.clip(
            prev_cmd + selected_vel * dt,
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        )
        nonlinear_backstop = False
        nonlinear_backstop_deferred = False
        # Never let a Cartesian position clip bypass the joint acceleration
        # limits.  The local FK backstop is accepted only when the QP was
        # feasible and the clipped velocity remains inside *all* dynamic and
        # Cartesian constraints.  Otherwise maximum reachable braking wins.
        linearized_next_distance = min(
            (
                float(term_payload[idx]["commandDistanceM"])
                - dt * float(np.dot(row_matrix[idx], selected_vel))
                for idx in range(row_matrix.shape[0])
            ),
            default=float("inf"),
        )
        backstop_near_outer_edge = bool(
            linearized_next_distance <= 0.006
            or hard_envelope_violation)
        if (bool(qp_info.get("feasible", False))
                and backstop_near_outer_edge):
            backstop_selected, backstop_requested = (
                self._workspace_nonlinear_backstop(prev_cmd, selected))
            if backstop_requested:
                backstop_vel = (backstop_selected - prev_cmd) / dt
                dynamic_ok = bool(
                    (lower is None or np.all(backstop_vel >= lower - 1e-8))
                    and (upper is None or np.all(backstop_vel <= upper + 1e-8)))
                cartesian_ok = bool(np.all(
                    row_matrix @ backstop_vel <= bound_vector + 1e-7))
                if dynamic_ok and cartesian_ok:
                    selected = backstop_selected
                    selected_vel = backstop_vel
                    nonlinear_backstop = True
                else:
                    nonlinear_backstop_deferred = True
        elif not qp_info.get("feasible", False):
            nonlinear_backstop_deferred = True
        selected_vel = (selected - prev_cmd) / dt
        self.safety_limiter.last_cmd = selected.copy()
        self.safety_limiter.last_vel = selected_vel.copy()

        solver_failure = bool(
            not qp_info.get("feasible", False)
            or qp_info.get("uncontrollable", False))
        if solver_failure:
            emergency_active = True
            emergency_reason = "solver_unreachable"
        elif hard_envelope_violation:
            emergency_active = True
            emergency_reason = "hard_envelope"
        elif emergency_reason in {
                "solver_unreachable", "feedback_stale",
                "fk_unavailable", "jacobian_unavailable"}:
            # One or more local tracking-brake ticks restored feasibility.
            emergency_active = False
            emergency_reason = ""
        self.workspace_soft_emergency_active = emergency_active
        self.workspace_soft_emergency_reason = emergency_reason

        correction_vel = selected_vel - candidate_vel
        correction_norm = float(np.linalg.norm(correction_vel))
        nominal_norm = float(np.linalg.norm(candidate_vel))
        retained_fraction = float(
            np.linalg.norm(selected_vel) / max(nominal_norm, 1e-12))
        intervened = bool(
            invalid_input or correction_norm > 1e-8 or nonlinear_backstop)
        model_name = (
            "predictive_cartesian_cbf" if predictive
            else "virtual_impedance_cbf" if impedance
            else "cartesian_velocity_cbf")
        self.workspace_soft_filter = {
            "active": True,
            "intervened": intervened,
            "predictive": bool(predictive),
            "model": model_name,
            "projection": "joint_velocity_qp",
            "alpha": round(float(min(1.0, retained_fraction)), 6),
            "retainedFraction": round(float(retained_fraction), 6),
            "marginM": round(float(margin), 6),
            "cbfGainS": round(float(cbf_gain), 6),
            "predictionHorizonS": round(float(prediction_horizon), 6),
            "predictionMarginM": round(float(prediction_margin), 6),
            "predictionSignal": bool(prediction_signal),
            "plantDelayS": round(float(configured_delay_s), 6),
            "actuatorTauS": round(float(actuator_tau_s), 6),
            "plantResponseReserveS": round(float(plant_response_s), 6),
            "positionM": _rounded_vector(position_vec),
            "commandPositionM": _rounded_vector(command_position_vec),
            "feedbackUsed": bool(feedback_used),
            "feedbackAgeS": feedback_age_s,
            "feedbackVelocitySource": str(getattr(
                self, "_workspace_feedback_velocity_source", "unavailable")),
            "invalidInput": invalid_input,
            "emergency": bool(emergency_active),
            "emergencyReason": str(emergency_reason),
            "hardEnvelopeM": round(float(hard_envelope_m), 6),
            "hardEnvelopeViolation": bool(hard_envelope_violation),
            "boundRelaxedForReachability": bool(any_bound_relaxed),
            "impedanceComfortApplied": bool(impedance_comfort_applied),
            "impedanceComfortRelaxed": bool(impedance_comfort_relaxed),
            "candidateJointVelocityRadS": _rounded_vector(candidate_vel),
            "selectedJointVelocityRadS": _rounded_vector(selected_vel),
            "jointCorrectionVelocityRadS": _rounded_vector(correction_vel),
            "taskVelocityMps": _rounded_vector(jacobian @ candidate_vel),
            "selectedTaskVelocityMps": _rounded_vector(jacobian @ selected_vel),
            "angularVelocityRadS": _rounded_vector(
                angular_jacobian @ candidate_vel),
            "selectedAngularVelocityRadS": _rounded_vector(
                angular_jacobian @ selected_vel),
            "terms": term_payload,
            "qp": dict(qp_info),
            "nonlinearBackstop": bool(nonlinear_backstop),
            "nonlinearBackstopDeferred": bool(nonlinear_backstop_deferred),
            "candidateScore": round(float(
                self._parking_workspace_violation_score(candidate)), 9),
            "selectedScore": round(float(
                self._parking_workspace_violation_score(selected)), 9),
        }
        return selected.copy()

    def _filter_command_with_workspace_soft_field(
        self,
        raw: np.ndarray,
        *,
        profile: str,
        predictive: bool = False,
    ) -> np.ndarray:
        """Compatibility entry point for the bounded virtual-wall mode."""
        return self._filter_command_with_workspace_cbf(
            raw,
            profile=profile,
            mode="predictive_soft" if predictive else "impedance_soft",
            apply_joint_limits=True,
        )

    def _filter_command_with_workspace_guard(
        self,
        raw: np.ndarray,
        *,
        profile: str,
        apply_joint_limits: bool,
        predictive_hold: bool = False,
    ) -> np.ndarray:
        """Compatibility entry point for position/predictive soft modes."""
        return self._filter_command_with_workspace_cbf(
            raw,
            profile=profile,
            mode="predictive_soft" if predictive_hold else "position_soft",
            apply_joint_limits=apply_joint_limits,
        )

    def _apply_workspace_soft_anti_windup(
        self,
        safe: np.ndarray,
        *,
        nominal_q: np.ndarray | None = None,
    ) -> None:
        """Back-calculate only the rejected safety increment into the policy.

        ``safe`` belongs to the actuator-reference layer and can contain up to
        30 degrees of inverse-MPC pre-distortion.  Writing it directly into the
        policy integrator (the previous behavior) mixed two state layers and
        created the observed stop/large-jump release cycle.  Instead, project
        the policy integrator's own outward velocity through the hard CBF only.
        The compensation offset and impedance spring are never integrated into
        the policy, while a rejected outward policy increment cannot wind up.
        """
        published_q = np.asarray(safe, dtype=np.float64).reshape(-1)
        if (published_q.shape[0] != len(RIGHT_ARM_JOINTS)
                or not np.all(np.isfinite(published_q))):
            return

        policy_q = self._right_arm_joint_vector_or_none(nominal_q)
        if policy_q is None:
            policy_q = self._right_arm_joint_vector_or_none(
                getattr(self.controller, "arm_cmd_q", None))
        if policy_q is None:
            return
        policy_vel = self._right_arm_joint_vector_or_none(
            getattr(self.controller, "arm_cmd_qvel", None))
        if policy_vel is None:
            policy_vel = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float64)

        position_correction = self._right_arm_joint_vector_or_none(
            getattr(
                self,
                "_workspace_antiwindup_position_correction",
                None,
            ))
        antiwindup_policy_vel = self._right_arm_joint_vector_or_none(
            getattr(
                self,
                "_workspace_antiwindup_policy_velocity",
                None,
            ))
        if position_correction is None and antiwindup_policy_vel is None:
            # FK/feedback/solver emergency braking belongs to the publication
            # layer; without a valid normal-only anti-windup solution, do not
            # invent a policy-state correction.
            self.workspace_soft_filter["antiWindup"] = False
            return
        if position_correction is None:
            position_correction = np.zeros(
                len(RIGHT_ARM_JOINTS), dtype=np.float64)
        if antiwindup_policy_vel is None:
            antiwindup_policy_vel = policy_vel.copy()

        corrected_policy_q = np.clip(
            policy_q + position_correction,
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        ).astype(np.float32)
        corrected_policy_vel = antiwindup_policy_vel.copy()
        policy_velocity_limit = np.asarray(
            getattr(
                self.controller,
                "arm_vel_limit_rad_s",
                self.safety_limiter.vel_limit_rad_s,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if policy_velocity_limit.shape[0] == len(RIGHT_ARM_JOINTS):
            corrected_policy_vel = np.clip(
                corrected_policy_vel,
                -policy_velocity_limit,
                policy_velocity_limit,
            )
        velocity_correction = corrected_policy_vel - policy_vel
        corrected_policy_vel = corrected_policy_vel.astype(np.float32)

        for name, value in (
            ("arm_cmd_q", corrected_policy_q),
            ("arm_cmd_qvel", corrected_policy_vel),
            ("last_cmd_q", corrected_policy_q),
            ("last_cmd_qvel", corrected_policy_vel),
        ):
            if hasattr(self.controller, name):
                setattr(self.controller, name, value.copy())

        history = getattr(self.controller, "_cmd_history", None)
        if isinstance(history, list) and history:
            # Older elements represent commands already issued in previous
            # ticks and must remain immutable.  Only replace the command just
            # appended by the current policy step.
            newest = self._right_arm_joint_vector_or_none(history[-1])
            if newest is None:
                newest = policy_q.copy()
            history[-1] = np.clip(
                newest + position_correction,
                RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
                RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
            ).astype(np.float32)

        self.workspace_soft_filter["antiWindup"] = True
        self.workspace_soft_filter["antiWindupModel"] = (
            "outward_rejection_only")
        self.workspace_soft_filter["nominalPositionCorrectionRad"] = (
            _rounded_vector(position_correction))
        self.workspace_soft_filter["nominalVelocityCorrectionRadS"] = (
            _rounded_vector(velocity_correction))

    def _filter_parking_command_with_workspace_guard(
        self,
        raw: np.ndarray,
        *,
        profile: str,
    ) -> np.ndarray:
        """Compatibility wrapper for the existing parking safety path."""
        return self._filter_command_with_workspace_guard(
            raw,
            profile=profile,
            apply_joint_limits=True,
        )

    def _arm_command_stream_from_feedback(self) -> bool:
        """Explicitly arm publishing with one local measured-position hold.

        Dry-run policy state can be far from the physical simulator.  The first
        publish-enabled tick therefore must not use that accumulated target.
        Require fresh joint feedback, reseed every policy/limiter/compensator
        layer from it, and publish exactly one hold.  Normal policy execution
        begins on the following 200 Hz tick.
        """
        if self.ros_bridge is None:
            return False
        feedback_q = None
        feedback_age_s = None
        try:
            getter = getattr(
                self.ros_bridge, "get_joint_feedback_with_age", None)
            if callable(getter):
                feedback_q, _dq, feedback_age_s, _source = getter()
            else:
                feedback_q, _dq = self.ros_bridge.get_joint_feedback()
        except Exception:
            feedback_q = None
        hold_q = self._right_arm_joint_vector_or_none(feedback_q)
        fresh = bool(
            hold_q is not None
            and (feedback_age_s is None
                 or (math.isfinite(float(feedback_age_s))
                     and float(feedback_age_s) <= 0.05)))
        if not fresh or hold_q is None:
            self.last_command_arm = {
                "armed": False,
                "reason": "fresh_joint_feedback_required",
                "feedbackAgeS": feedback_age_s,
            }
            return False
        hold_q = np.clip(
            hold_q,
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        )
        # This helper resets recurrent policy command history from current ROS
        # feedback rather than the fixed reset/safe pose.
        self._reset_real_policy_for_sim_reentry()
        self.safety_limiter.reset(hold_q)
        self._reset_compensator_to_current_arm(hold_q)
        self.ros_bridge.publish_right_arm_command(hold_q)
        self.last_policy_nominal_reference = hold_q.copy()
        self.last_compensated_reference = hold_q.copy()
        self.last_published_safe_reference = hold_q.copy()
        self.command_stream_active = True
        self.last_command_arm = {
            "armed": True,
            "reason": "explicit_publish_enable",
            "feedbackAgeS": feedback_age_s,
            "jointPosition": _rounded_vector(hold_q),
        }
        return True

    def _publish_hold_on_command_disable(self) -> bool:
        """Stop an active position stream at the latest measured joint pose."""
        if not self.command_stream_active or self.ros_bridge is None:
            return False
        feedback_q, _feedback_dq = self.ros_bridge.get_joint_feedback()
        hold_q = self._right_arm_joint_vector_or_none(feedback_q)
        feedback_available = hold_q is not None
        if hold_q is None:
            hold_q = self.safety_limiter.last_cmd.copy()
        hold_q = np.clip(
            np.asarray(hold_q, dtype=np.float64),
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        )
        self.ros_bridge.publish_right_arm_command(hold_q)
        self.safety_limiter.reset(hold_q)
        self._reset_compensator_to_current_arm(hold_q)
        self.command_stream_active = False
        self.last_command_disable_hold = {
            "published": True,
            "feedbackAvailable": bool(feedback_available),
            "jointPosition": _rounded_vector(hold_q),
        }
        print(
            "[CommandSafety][INFO] Send command disabled; published one-shot "
            f"hold target (feedback={feedback_available})",
            flush=True,
        )
        return True

    def _publish_measured_stop_hold(self) -> bool:
        """Publish exactly one fresh measured-position hold for hard stops."""
        if not self.command_stream_active or self.ros_bridge is None:
            self.command_stream_active = False
            self.last_command_disable_hold = {
                "published": False,
                "feedbackAvailable": False,
                "reason": "stream_inactive_or_ros_unavailable",
                "jointPosition": None,
            }
            return False
        feedback_q = None
        feedback_age_s = None
        try:
            getter = getattr(
                self.ros_bridge, "get_joint_feedback_with_age", None)
            if callable(getter):
                feedback_q, _dq, feedback_age_s, _source = getter()
            else:
                feedback_q, _dq = self.ros_bridge.get_joint_feedback()
        except Exception:
            feedback_q = None
        hold_q = self._right_arm_joint_vector_or_none(feedback_q)
        fresh = bool(
            hold_q is not None
            and (feedback_age_s is None
                 or (math.isfinite(float(feedback_age_s))
                     and float(feedback_age_s) <= 0.05)))
        if not fresh or hold_q is None:
            self.command_stream_active = False
            self.last_command_disable_hold = {
                "published": False,
                "feedbackAvailable": False,
                "feedbackAgeS": feedback_age_s,
                "jointPosition": None,
            }
            print(
                "[CommandSafety][STOP] Hard workspace stop could not publish "
                "measured hold; no fresh joint feedback, command stream closed",
                flush=True,
            )
            return False
        hold_q = np.clip(
            np.asarray(hold_q, dtype=np.float64),
            RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD,
            RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD,
        )
        self.ros_bridge.publish_right_arm_command(hold_q)
        self.safety_limiter.reset(hold_q)
        self._reset_compensator_to_current_arm(hold_q)
        self.last_policy_nominal_reference = hold_q.copy()
        self.last_compensated_reference = hold_q.copy()
        self.last_published_safe_reference = hold_q.copy()
        self.command_stream_active = False
        self.last_command_disable_hold = {
            "published": True,
            "feedbackAvailable": True,
            "feedbackAgeS": feedback_age_s,
            "jointPosition": _rounded_vector(hold_q),
        }
        print(
            "[CommandSafety][STOP] Hard workspace done; published one-shot "
            "measured-position hold and closed command stream",
            flush=True,
        )
        return True

    def _hard_workspace_stop_active(self) -> bool:
        return bool(
            getattr(self, "hard_workspace_stop_latched", False)
            and self.done_failsafe_active
        )

    def _latch_hard_workspace_stop(
        self,
        reason: str,
        conditions: dict[str, bool],
    ) -> None:
        self.hard_workspace_stop_latched = True
        self.hard_workspace_stop_reason = str(reason or "hard_workspace_done")
        self.hard_workspace_stop_conditions = {
            str(name): bool(active)
            for name, active in conditions.items()
            if bool(active)
        }
        self.hard_workspace_stop_hold_sent = False
        print(
            "[CommandSafety][STOP] Hard workspace done latched; "
            "publishing one measured-position hold then rejecting commands "
            f"until reset (reason={self.hard_workspace_stop_reason})",
            flush=True,
        )

    def _stop_hard_workspace_command_stream(self) -> bool:
        """Reject all commands after hard workspace done, except one stop hold."""
        self.compensation_was_enabled = False
        if not bool(getattr(self, "hard_workspace_stop_hold_sent", False)):
            held = self._publish_measured_stop_hold()
            self.hard_workspace_stop_hold_sent = True
            if not held:
                self.command_stream_active = False
                if not isinstance(
                        getattr(self, "last_command_disable_hold", None),
                        dict):
                    self.last_command_disable_hold = {
                        "published": False,
                        "feedbackAvailable": False,
                        "jointPosition": None,
                    }
        else:
            self.command_stream_active = False
        self.last_command_arm = {
            "armed": False,
            "reason": "hard_workspace_done_latched",
            "doneReason": getattr(
                self, "hard_workspace_stop_reason", self.done_failsafe_reason),
            "conditions": dict(getattr(
                self, "hard_workspace_stop_conditions", {})),
        }
        return False

    def _hard_workspace_stop_payload(self) -> dict[str, Any]:
        return {
            "active": bool(getattr(self, "hard_workspace_stop_latched", False)),
            "reason": str(getattr(self, "hard_workspace_stop_reason", "")),
            "conditions": dict(getattr(
                self, "hard_workspace_stop_conditions", {})),
            "holdSent": bool(getattr(
                self, "hard_workspace_stop_hold_sent", False)),
        }

    def _publish_control_command(
        self,
        q: np.ndarray,
        options: RuntimeOptions,
        *,
        apply_compensation: bool = False,
        profile: str = "normal",
    ) -> bool:
        if self._hard_workspace_stop_active():
            return self._stop_hard_workspace_command_stream()
        if self.ros_bridge is None:
            self.compensation_was_enabled = False
            return False
        if not options.publish_commands:
            self.compensation_was_enabled = False
            return self._publish_hold_on_command_disable()
        if not self.command_stream_active:
            self.compensation_was_enabled = False
            return self._arm_command_stream_from_feedback()
        raw = np.asarray(q, dtype=np.float64).reshape(-1)
        nominal_q = raw.copy()
        self.last_policy_nominal_reference = nominal_q.copy()
        if apply_compensation:
            raw = self._apply_actuator_compensation(raw, options)
        else:
            self.compensation_was_enabled = False
        self.last_compensated_reference = raw.copy()
        workspace_mode = _workspace_safety_mode(
            options.workspace_safety_mode,
            default=self.cfg.workspace_safety_mode,
        )
        # The workspace filter is downstream of inverse compensation, so an
        # intervention still has to propagate through the real transport
        # delay and actuator lag.  Keep the currently applied runtime plant
        # values available to its stopping-distance reserve.
        self._workspace_actuation_delay_s = max(
            0.0,
            float(getattr(
                options,
                "comp_delay_ms",
                getattr(self.cfg, "comp_delay_ms", 72.0),
            )) * 1e-3,
        )
        self._workspace_actuator_tau_s = max(
            0.0,
            float(getattr(
                options,
                "comp_actuator_tau_s",
                getattr(self.cfg, "comp_actuator_tau_s", 0.074),
            )),
        )
        profile_name = str(profile).strip().lower()
        parking_profile = profile_name in {
            "parking", "park", "safe", "reset", "failsafe"}
        use_limiter = (
            bool(self.cfg.enable_safety_limiter)
            or profile_name != "normal")
        if (workspace_mode in {"position_soft", "predictive_soft"}
                and not parking_profile):
            safe = self._filter_command_with_workspace_guard(
                raw,
                profile=profile,
                apply_joint_limits=True,
                predictive_hold=bool(workspace_mode == "predictive_soft"),
            )
            if bool(self.workspace_soft_filter.get("intervened", False)):
                self._apply_workspace_soft_anti_windup(
                    safe, nominal_q=nominal_q)
        elif workspace_mode == "impedance_soft" and not parking_profile:
            safe = self._filter_command_with_workspace_soft_field(
                raw,
                profile=profile,
                predictive=False,
            )
            if bool(self.workspace_soft_filter.get("intervened", False)):
                self._apply_workspace_soft_anti_windup(
                    safe, nominal_q=nominal_q)
        elif use_limiter:
            if (parking_profile
                    and profile_name != "reset"
                    and workspace_mode != "off"):
                safe = self._filter_parking_command_with_workspace_guard(
                    raw, profile=profile)
            else:
                safe = self.safety_limiter.filter(raw, profile=profile)
        else:
            safe = raw.copy()
            self.safety_limiter.observe(safe)
        self.ros_bridge.publish_right_arm_command(safe)
        self.last_published_safe_reference = np.asarray(
            safe, dtype=np.float64).copy()
        if (getattr(self, "compensator", None) is not None
                and getattr(self, "compensation_was_enabled", False)):
            try:
                observe_published = getattr(
                    self.compensator, "observe_published", None)
                if callable(observe_published):
                    observe_published(np.asarray(safe, dtype=np.float32))
                    self.workspace_soft_filter["compensatorAntiWindup"] = True
            except Exception as exc:  # noqa: BLE001 - publishing already succeeded.
                self.compensation_error = f"published feedback failed: {exc}"
        self.command_stream_active = True
        self.last_command_disable_hold = {
            "published": False,
            "feedbackAvailable": False,
            "jointPosition": None,
        }
        return True

    def _start_backend_control_loop(self) -> None:
        if self.control_loop_thread is not None and self.control_loop_thread.is_alive():
            return
        if self.ros_bridge is not None:
            self.control_loop_stop.clear()
            self.control_loop_uses_ros_timer = True
            self.ros_bridge.start_backend_control_timer(
                self.cfg.control_dt, self._backend_control_timer_callback)
            return
        self.control_loop_stop.clear()
        self.control_loop_uses_ros_timer = False
        self.control_loop_thread = threading.Thread(
            target=self._backend_control_loop,
            name="policy_probe_200hz_control",
            daemon=True,
        )
        self.control_loop_thread.start()

    def shutdown(self) -> None:
        self.control_loop_stop.set()
        if self.ros_bridge is not None:
            self.ros_bridge.stop_backend_control_timer()
        if self.control_loop_thread is not None and self.control_loop_thread.is_alive():
            self.control_loop_thread.join(timeout=1.0)
        self.recorder.shutdown()
        if self.ros_bridge is not None:
            self.ros_bridge.shutdown()
        if self.sim_mirror is not None:
            self.sim_mirror.close()

    def start_recording(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        view = payload.get("view")
        view_payload = view if isinstance(view, dict) else None
        status = self.recorder.start(
            self.browser_config(), initial_view=view_payload)
        return status

    def stop_recording(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        view = payload.get("view")
        freeze_source_seq = max(-1, int(self.control_loop_tick) - 1)
        self.recorder.freeze(freeze_source_seq)
        alignment = {
            "event": "sim_alignment",
            "wallTimeS": time.time(),
            "freezeSourceSeq": freeze_source_seq,
            "simEnabled": bool(self.sim_mirror is not None),
            "flushComplete": self.sim_mirror is None,
            "alignmentValid": self.sim_mirror is None,
        }
        if isinstance(self.sim_mirror, SimMirrorSubprocess):
            try:
                flush_timeout_s = max(0.0, float(os.environ.get(
                    "SIM_RECORDING_FLUSH_TIMEOUT_S", "5.0")))
            except (TypeError, ValueError):
                flush_timeout_s = 5.0
            flush = self.sim_mirror.flush(
                self.sim_mirror.stream_seq, flush_timeout_s)
            self.recorder.record_sim_control_samples(
                self.sim_mirror.drain_completed_samples())
            alignment.update({
                "flushComplete": bool(flush.get("complete", False)),
                "alignmentValid": bool(
                    flush.get("alignmentValid", False)),
                "flush": flush,
            })
        elif self.sim_mirror is not None:
            alignment.update({
                "flushComplete": True,
                "alignmentValid": True,
            })
        self.recorder.record_event(alignment)
        return self.recorder.stop(
            final_view=view if isinstance(view, dict) else None,
            analyze=bool(payload.get("analyze", False)))

    def recording_status(self) -> dict[str, Any]:
        return self.recorder.status()

    def cached_recording_status(self) -> dict[str, Any]:
        return self.recording_status()

    def record_browser_view(self, view: dict[str, Any] | None) -> None:
        self.recorder.record_view(view)

    def record_camera_frame(
        self,
        frame: bytes | None,
        info: dict[str, Any] | None,
    ) -> None:
        self.recorder.record_camera_frame(frame, info)

    def record_sim_frame(
        self,
        frame: bytes | None,
        info: dict[str, Any] | None,
    ) -> None:
        self.recorder.record_sim_frame(frame, info)

    def _run_backend_control_once(self) -> None:
        start = time.perf_counter()
        try:
            self._run_backend_control_tick()
            self.control_loop_last_error = None
        except Exception as exc:
            self.control_loop_last_error = str(exc)
            return
        finally:
            work_s = time.perf_counter() - start
            if work_s > 0.0:
                self.control_loop_run_times.append(work_s)
                if work_s > self.cfg.control_dt:
                    self.control_loop_overrun_count += 1
        now = time.perf_counter()
        if self.control_loop_last_wall is not None:
            elapsed = now - self.control_loop_last_wall
            if elapsed > 0.0:
                self.control_loop_times.append(elapsed)
        self.control_loop_last_wall = now

    def _backend_control_timer_callback(self) -> None:
        if self.control_loop_stop.is_set():
            return
        self._run_backend_control_once()

    def _backend_control_loop(self) -> None:
        next_t = time.perf_counter()
        dt = self.cfg.control_dt
        while not self.control_loop_stop.is_set():
            self._run_backend_control_once()
            next_t += dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()

    def _control_loop_hz(self) -> float:
        if not self.control_loop_times:
            return 0.0
        avg = sum(self.control_loop_times) / len(self.control_loop_times)
        return 1.0 / avg if avg > 0.0 else 0.0

    def _control_loop_work_ms(self) -> float:
        if not self.control_loop_run_times:
            return 0.0
        avg = sum(self.control_loop_run_times) / len(self.control_loop_run_times)
        return 1000.0 * avg

    def _run_backend_control_tick(self) -> None:
        with self.lock:
            options = self.runtime_options or self._runtime_options_from_payload({})
            if (options.publish_commands
                    and time.monotonic()
                    > float(self.command_publish_lease_deadline_wall)):
                # The browser/client supplies the arm heartbeat.  Losing it
                # follows the normal disable path: one measured-position hold,
                # then no further command stream.
                options = replace(options, publish_commands=False)
                self.runtime_options = options
                self.command_publish_lease_expired = True
            drag_pos = self.web_drag_ball_pos.copy()
            drag_vel = self.web_drag_ball_vel.copy()
            drag_valid = bool(self.web_drag_valid)
            noise_std_m = float(self.web_noise_std_m)
            if (self.ros_bridge is not None
                    and self.cfg.enable_camera_ball
                    and options.camera != self.last_camera_runtime_options):
                self.ros_bridge.set_camera_runtime_options(options.camera)
                self.last_camera_runtime_options = options.camera
            now_wall = time.perf_counter()
            make_snapshot = now_wall >= self.backend_next_snapshot_wall
            snapshot = self._run_control_tick_locked(
                drag_pos, drag_vel, drag_valid, noise_std_m, options,
                make_snapshot=make_snapshot)
            if snapshot is not None:
                self.latest_snapshot_payload = snapshot
                self.recorder.record_snapshot(snapshot)
                self.backend_next_snapshot_wall = (
                    now_wall + self.backend_snapshot_period_s)

    def _racket_pos(self) -> tuple[np.ndarray, bool]:
        pos, ok = self.controller._compute_racket_pose_base(self.controller.arm_cmd_q)
        return np.asarray(pos, dtype=np.float32).reshape(3), bool(ok)

    def _racket_pose(self) -> tuple[np.ndarray, np.ndarray | None, bool]:
        pos, quat, ok = self.controller._compute_racket_pose_quat_base(
            self.controller.arm_cmd_q)
        return (
            np.asarray(pos, dtype=np.float32).reshape(3),
            None if not ok else np.asarray(quat, dtype=np.float32).reshape(4),
            bool(ok),
        )

    def _ros_joint_racket_pos(self) -> tuple[np.ndarray | None, bool]:
        if self.ros_bridge is None:
            return None, False
        q_fb, _dq_fb = self.ros_bridge.get_joint_feedback()
        if q_fb is None:
            return None, False
        pos, ok = self.controller._compute_racket_pose_base(q_fb)
        if not ok:
            return None, False
        return np.asarray(pos, dtype=np.float32).reshape(3), True

    def _ros_cart_pose(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        if self.ros_bridge is None:
            return None, None, None
        return self.ros_bridge.get_cart_feedback()

    def _estimator_off_payload(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "prediction": False,
            "aeroDrag": False,
            "valid": False,
            "state": "off",
            "predicted": False,
            "ageS": None,
            "lossS": None,
            "extrapolationS": None,
            "clamped": False,
            "updatesAccepted": 0,
            "updatesRejected": 0,
            "reinits": 0,
            "bouncesDetected": 0,
            "lastEvent": None,
        }

    def _make_ball_estimator(
        self,
        options: RuntimeOptions,
        frame_rotation: np.ndarray | None = None,
    ) -> BallStateEstimator:
        cfg = BallStateEstimatorConfig(
            gravity_mm_s2=gravity_for_base_link(),
            q_pos_mm2_s=options.estimator_q_pos_mm2_s,
            q_vel_mult=options.estimator_q_vel_mult,
            q_depth_pos_mult=options.estimator_q_depth_pos_mult,
            q_depth_vel_mult=options.estimator_q_depth_vel_mult,
            r_lateral_mm=options.estimator_r_lateral_mm,
            r_depth_mm=options.estimator_r_depth_mm,
            max_extrapolation_s=options.estimator_max_extrapolation_s,
            trust_loss_s=options.estimator_trust_loss_s,
            degrade_loss_s=options.estimator_degrade_loss_s,
            invalid_after_s=options.estimator_invalid_after_s,
            prediction_horizon_s=options.estimator_prediction_horizon_s,
            enable_prediction=bool(options.estimator_prediction),
            control_output_mode=options.estimator_output_mode,
            frame_hold_output_rate_hz=(
                options.estimator_frame_hold_output_rate_hz),
            enable_ballistic_dynamics=not (
                options.estimator_release_gated_dynamics
                and options.enable_ball_release_gate
                and not self.has_released),
            enable_aero_drag=bool(options.estimator_aero_drag),
        )
        return BallStateEstimator(cfg, frame_rotation=frame_rotation)

    def _estimator_frame_rotation(
        self,
        options: RuntimeOptions,
    ) -> np.ndarray | None:
        if (options.ball_source != "camera" or self.ros_bridge is None
                or not self.cfg.enable_camera_ball):
            return None
        try:
            return self.ros_bridge.camera_frame_rotation()
        except Exception:
            return None

    def _ensure_ball_estimator(self, options: RuntimeOptions) -> BallStateEstimator:
        frame_rotation = self._estimator_frame_rotation(options)
        frame_rotation_key = (
            None if frame_rotation is None else tuple(
                round(float(v), 9) for v in frame_rotation.reshape(-1)))
        key = (
            str(options.ball_source),
            bool(options.estimator_prediction),
            bool(options.estimator_aero_drag),
            bool(options.enable_ball_release_gate),
            str(options.estimator_output_mode),
            bool(options.estimator_release_gated_dynamics),
            float(options.estimator_max_extrapolation_s),
            float(options.estimator_frame_hold_output_rate_hz),
            float(options.estimator_trust_loss_s),
            float(options.estimator_degrade_loss_s),
            float(options.estimator_invalid_after_s),
            float(options.estimator_prediction_horizon_s),
            float(options.estimator_q_pos_mm2_s),
            float(options.estimator_q_vel_mult),
            float(options.estimator_q_depth_pos_mult),
            float(options.estimator_q_depth_vel_mult),
            float(options.estimator_r_lateral_mm),
            float(options.estimator_r_depth_mm),
            frame_rotation_key,
        )
        if self.ball_estimator is None or self.ball_estimator_key != key:
            self.ball_estimator = self._make_ball_estimator(
                options, frame_rotation=frame_rotation)
            self.ball_estimator_key = key
            self.next_ball_observation_s = 0.0
            self.last_ball_observation_stamp_s = None
            self.ball_observation_clock_domain = None
        return self.ball_estimator

    def _estimator_payload(
        self,
        options: RuntimeOptions,
        out: dict[str, Any],
    ) -> dict[str, Any]:
        diag = (
            self.ball_estimator.diagnostics()
            if self.ball_estimator is not None else {}
        )
        return {
            "enabled": True,
            "prediction": bool(options.estimator_prediction),
            "aeroDrag": bool(options.estimator_aero_drag),
            "releaseGatedDynamics": bool(
                options.estimator_release_gated_dynamics),
            "valid": bool(out.get("valid", False)),
            "state": str(out.get("state", "")),
            "predicted": bool(out.get("predicted", False)),
            "outputMode": str(out.get("output_mode", "")),
            "controlValid": bool(out.get(
                "control_valid", out.get("valid", False))),
            "controlRejectReason": out.get("control_reject_reason"),
            "snapshotSequence": int(out.get("snapshot_sequence", 0)),
            "snapshotNew": bool(out.get("snapshot_new", False)),
            "snapshotHoldS": (
                None if out.get("snapshot_hold_s") is None
                else round(float(out.get("snapshot_hold_s", 0.0)), 6)
            ),
            "controlAgeS": (
                None if out.get("control_age_s") is None
                else round(float(out.get("control_age_s", 0.0)), 6)
            ),
            "maxExtrapolationS": round(
                float(options.estimator_max_extrapolation_s), 6),
            "controlMaxExtrapolationS": round(
                float(options.estimator_control_max_extrapolation_s), 6),
            "frameHoldOutputRateHz": round(
                float(options.estimator_frame_hold_output_rate_hz), 6),
            "trustLossS": round(float(options.estimator_trust_loss_s), 6),
            "degradeLossS": round(float(options.estimator_degrade_loss_s), 6),
            "invalidAfterS": round(
                float(options.estimator_invalid_after_s), 6),
            "predictionHorizonS": round(
                float(options.estimator_prediction_horizon_s), 6),
            "qPosMm2S": round(float(options.estimator_q_pos_mm2_s), 6),
            "qVelMult": round(float(options.estimator_q_vel_mult), 6),
            "qDepthPosMult": round(
                float(options.estimator_q_depth_pos_mult), 6),
            "qDepthVelMult": round(
                float(options.estimator_q_depth_vel_mult), 6),
            "rLateralMm": round(float(options.estimator_r_lateral_mm), 6),
            "rDepthMm": round(float(options.estimator_r_depth_mm), 6),
            "ballisticDynamics": bool(diag.get(
                "ballistic_dynamics_enabled", True)),
            "dynamicsReseeds": int(diag.get("dynamics_reseeds", 0)),
            "ageS": (
                None if out.get("age_s") is None
                else round(float(out.get("age_s", 0.0)), 6)
            ),
            "lossS": (
                None if out.get("loss_s") is None
                else round(float(out.get("loss_s", 0.0)), 6)
            ),
            "extrapolationS": (
                None if out.get("extrapolation_s") is None
                else round(float(out.get("extrapolation_s", 0.0)), 6)
            ),
            "clamped": bool(out.get("clamped", False)),
            "updatesAccepted": int(diag.get("updates_accepted", 0)),
            "updatesRejected": int(diag.get("updates_rejected", 0)),
            "reinits": int(diag.get("reinits", 0)),
            "bouncesDetected": int(diag.get("bounces_detected", 0)),
            "lastEvent": diag.get("last_event"),
        }

    def _resolve_control_ball_state(
        self,
        measured_pos: np.ndarray,
        measured_vel: np.ndarray,
        measured_valid: bool,
        options: RuntimeOptions,
        measurement_stamp_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool, float | None, dict[str, Any]]:
        if not options.use_ball_estimator:
            self.ball_estimator = None
            self.ball_estimator_key = None
            self.last_ball_observation_stamp_s = None
            return (
                measured_pos.copy(), measured_vel.copy(), bool(measured_valid),
                None, self._estimator_off_payload())

        est = self._ensure_ball_estimator(options)
        t_control = self.now_s + self.cfg.control_dt
        t_predict = t_control
        t_update = t_control
        stamped_measurement = (
            measurement_stamp_s is not None
            and math.isfinite(float(measurement_stamp_s))
            and float(measurement_stamp_s) > 0.0
        )
        if stamped_measurement:
            t_update = float(measurement_stamp_s)
            if options.ball_source == "camera" and self.ros_bridge is not None:
                try:
                    t_predict = self.ros_bridge.now_sec()
                except Exception:
                    t_predict = t_update
            if t_predict < t_update:
                t_predict = t_update
        obs_rate = max(float(options.ball_observation_rate_hz), 1e-6)
        obs_period = 1.0 / obs_rate

        clock_domain = "stamp" if stamped_measurement else "control"
        if self.ball_observation_clock_domain != clock_domain:
            self.next_ball_observation_s = t_update
            self.last_ball_observation_stamp_s = None
            self.ball_observation_clock_domain = clock_domain
        elif self.next_ball_observation_s <= 0.0:
            self.next_ball_observation_s = t_update
        should_update = False
        if measured_valid:
            if stamped_measurement:
                last_stamp = self.last_ball_observation_stamp_s
                is_new_stamp = (
                    last_stamp is None
                    or t_update > float(last_stamp) + 1e-9
                )
                should_update = (
                    is_new_stamp
                    and t_update + 1e-9 >= self.next_ball_observation_s
                )
            else:
                should_update = (
                    t_update + 1e-9 >= self.next_ball_observation_s)
        if should_update:
            est.update((measured_pos * 1000.0).astype(np.float64), t_update)
            if stamped_measurement:
                self.last_ball_observation_stamp_s = t_update
            if self.next_ball_observation_s <= t_update + 1e-9:
                periods = int(math.floor(
                    (t_update - self.next_ball_observation_s) / obs_period
                )) + 1
                self.next_ball_observation_s += max(1, periods) * obs_period

        out = dict(est.predict_to(t_predict))
        age_s = float(out.get("age_s", 0.0))
        if not math.isfinite(age_s):
            age_s = 0.0
        age_s = max(0.0, age_s)

        if bool(out.get("valid", False)):
            control_cap = float(options.estimator_control_max_extrapolation_s)
            if control_cap >= 0.0:
                extrap_s = float(out.get("extrapolation_s", 0.0))
                if not math.isfinite(extrap_s):
                    extrap_s = 0.0
                control_span_s = extrap_s
                if (out.get("output_mode")
                        == CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD):
                    control_span_s = float(
                        out.get("control_age_s", extrap_s))
                    if not math.isfinite(control_span_s):
                        control_span_s = extrap_s
                reject_reason = None
                if bool(out.get("clamped", False)):
                    reject_reason = "estimator_clamped"
                elif control_span_s > control_cap + 1e-9:
                    reject_reason = (
                        "control_frame_hold_timeout"
                        if out.get("output_mode")
                        == CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD
                        else "control_extrapolation_timeout"
                    )
                if reject_reason is not None:
                    out["control_valid"] = False
                    out["control_reject_reason"] = reject_reason
                    out["control_max_extrapolation_s"] = control_cap
                    out["control_gate_span_s"] = control_span_s
        info = self._estimator_payload(options, out)
        if (not bool(out.get("valid", False))
                or not bool(out.get("control_valid", True))):
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                age_s,
                info,
            )
        pos = np.asarray(out.get("position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3) * 1e-3
        vel = np.asarray(out.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3) * 1e-3
        return pos, vel, True, max(0.0, age_s), info

    def _raw_measurement_age_s(
        self,
        valid: bool,
        measurement_stamp_s: float | None,
        options: RuntimeOptions,
    ) -> float:
        if not valid:
            return self.cfg.ball_lost_timeout_s
        if (measurement_stamp_s is None
                or not math.isfinite(float(measurement_stamp_s))
                or float(measurement_stamp_s) <= 0.0):
            return 0.0
        if options.ball_source != "camera" or self.ros_bridge is None:
            return 0.0
        try:
            return max(0.0, self.ros_bridge.now_sec() - float(measurement_stamp_s))
        except Exception:
            return 0.0

    def _apply_workspace_safety_mode(
        self,
        mode: str,
        options: RuntimeOptions | None = None,
    ) -> None:
        mode = _workspace_safety_mode(mode, self.cfg.workspace_safety_mode)
        previous_mode = getattr(self, "active_workspace_safety_mode", None)
        if previous_mode != mode:
            print(
                "[WorkspaceSafety][INFO] "
                f"mode={mode}, soft={mode in WORKSPACE_SOFT_MODES}, "
                f"predictive={mode in WORKSPACE_PREDICTIVE_MODES}",
                flush=True,
            )
            self.active_workspace_safety_mode = mode
            self.workspace_soft_entry_q = None
            self.workspace_hold_active = False
            self.workspace_hold_trigger_reason = ""
            self.workspace_hold_constraint = ""
            current_command = self._right_arm_joint_vector_or_none(
                getattr(self.safety_limiter, "last_cmd", None))
            if current_command is not None:
                self.workspace_hold_q = current_command.copy()
                self.workspace_hold_last_safe_q = current_command.copy()
        workspace_enabled = mode in {
            "position", "predictive", "position_soft", "predictive_soft",
            "impedance_soft"}
        predictive_enabled = mode in WORKSPACE_PREDICTIVE_MODES
        self.done_monitor.ee_x_limits = (
            self.cfg.ee_x_limits_m if workspace_enabled else None)
        self.done_monitor.ee_y_limits = (
            self.cfg.ee_y_limits_m if workspace_enabled else None)
        self.done_monitor.ee_z_limits = (
            self.cfg.ee_z_limits_m if workspace_enabled else None)
        self.done_monitor.enable_predictive_stop = bool(
            predictive_enabled and self.cfg.done_enable_predictive_stop)
        self.done_monitor.prediction_horizon_s = max(
            0.0,
            float(options.done_prediction_horizon_s)
            if options is not None else float(self.cfg.done_prediction_horizon_s),
        )
        self.done_monitor.prediction_dt_s = max(
            1e-4,
            float(options.done_prediction_dt_s)
            if options is not None else float(self.cfg.done_prediction_dt_s),
        )
        self.done_monitor.prediction_margin_m = max(
            0.0,
            float(options.done_prediction_margin_m)
            if options is not None else float(self.cfg.done_prediction_margin_m),
        )
        self.done_monitor.prediction_orientation_margin_rad = max(
            0.0,
            float(options.done_prediction_orientation_margin_rad)
            if options is not None
            else float(self.cfg.done_prediction_orientation_margin_rad),
        )

    def _release_incompatible_done_latch(self, workspace_mode: str) -> bool:
        """Release a latch that the newly selected mode no longer treats as hard.

        This is needed when the browser switches from a hard workspace mode to
        a soft one after the arm has already latched ``done_failsafe``. A hard
        workspace stop latch is stricter: only reset clears it. Ball and base
        violations remain latched in soft modes; ``off`` releases other done
        conditions, matching the existing disabled-mode behavior.
        """
        if not self.done_failsafe_active:
            return False
        if bool(getattr(self, "hard_workspace_stop_latched", False)):
            return False
        default_mode = getattr(
            self.cfg, "workspace_safety_mode", "position_soft")
        mode = _workspace_safety_mode(
            workspace_mode, default_mode)
        hard_conditions = (
            {}
            if mode == "off"
            else _hard_done_conditions(
                self.done_failsafe_conditions, mode)
        )
        if hard_conditions:
            self.done_failsafe_conditions = dict(hard_conditions)
            self.done_failsafe_reason = next(iter(hard_conditions))
            return False
        self.done_failsafe_active = False
        self.done_failsafe_reason = ""
        self.done_failsafe_conditions = {}
        return True

    def _debounce_ball_done_conditions(
        self,
        conditions: dict[str, bool],
    ) -> dict[str, bool]:
        """Reject one-frame ball workspace spikes before latching done."""
        required = max(1, int(getattr(
            self.cfg, "done_ball_required_consecutive_ticks", 5)))
        streaks = dict(getattr(self, "ball_done_streaks", {}))
        out = {str(name): bool(active) for name, active in conditions.items()}
        raw: dict[str, bool] = {}
        for name in (
            "ball_too_low",
            "ball_too_high",
            "ball_x_out_of_bounds",
            "ball_y_out_of_bounds",
        ):
            active = bool(out.get(name, False))
            raw[name] = active
            streaks[name] = int(streaks.get(name, 0)) + 1 if active else 0
            out[name] = bool(active and streaks[name] >= required)
        self.ball_done_streaks = streaks
        self.ball_done_raw_conditions = raw
        return out

    def _done_motion_estimate(
        self,
        racket_pos_m: np.ndarray | None,
        ee_quat_xyzw: np.ndarray | None,
        ee_psi_rad: float | None,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "racket_vel_m_s": None,
            "racket_acc_m_s2": None,
            "ee_psi_vel_rad_s": None,
            "ee_abc_vel_rad_s": None,
        }
        if racket_pos_m is None:
            return out
        pos = np.asarray(racket_pos_m, dtype=np.float64).reshape(-1)
        if pos.shape[0] != 3 or not np.all(np.isfinite(pos)):
            return out

        now_s = float(self.now_s)
        prev_t = self._done_motion_prev_t_s
        prev_pos = self._done_motion_prev_pos_m
        prev_vel = self._done_motion_prev_vel_m_s
        prev_psi = self._done_motion_prev_psi_rad
        prev_abc = self._done_motion_prev_abc_rad

        abc = _quat_to_euler_xyz_rad(ee_quat_xyzw)
        psi = (
            float(ee_psi_rad)
            if ee_psi_rad is not None and math.isfinite(float(ee_psi_rad))
            else None)
        if prev_t is None or prev_pos is None:
            vel = np.zeros(3, dtype=np.float64)
            acc = np.zeros(3, dtype=np.float64)
            psi_vel = 0.0 if psi is not None else None
            abc_vel = (
                np.zeros(3, dtype=np.float64) if abc is not None else None)
        else:
            dt = max(1e-4, min(0.1, now_s - float(prev_t)))
            vel = (pos - prev_pos) / dt
            acc = (
                np.zeros(3, dtype=np.float64)
                if prev_vel is None else (vel - prev_vel) / dt)
            psi_vel = None
            if psi is not None and prev_psi is not None:
                psi_vel = _wrap_angle_rad(psi - float(prev_psi)) / dt
            abc_vel = None
            if abc is not None and prev_abc is not None:
                abc_vel = np.asarray([
                    _wrap_angle_rad(float(abc[i]) - float(prev_abc[i])) / dt
                    for i in range(3)
                ], dtype=np.float64)

        self._done_motion_prev_t_s = now_s
        self._done_motion_prev_pos_m = pos.copy()
        self._done_motion_prev_vel_m_s = vel.copy()
        self._done_motion_prev_psi_rad = psi
        self._done_motion_prev_abc_rad = (
            None if abc is None else abc.copy())

        out["racket_vel_m_s"] = vel
        out["racket_acc_m_s2"] = acc
        out["ee_psi_vel_rad_s"] = psi_vel
        out["ee_abc_vel_rad_s"] = abc_vel
        return out

    def _run_one_control_step(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid: bool,
        age_override_s: float | None,
        options: RuntimeOptions,
        release_ball_pos: np.ndarray | None = None,
        release_ball_vel: np.ndarray | None = None,
        release_valid: bool | None = None,
        release_age_s: float | None = None,
        done_ball_pos: np.ndarray | None = None,
        done_ball_vel: np.ndarray | None = None,
        done_ball_valid: bool | None = None,
    ) -> tuple[str, str, dict[str, bool], float, bool]:
        self.controller.use_valid_ball_obs_age = bool(
            options.rl_use_valid_ball_obs_age)
        self.now_s += self.cfg.control_dt
        self.workspace_soft_prediction_active = False
        self.workspace_soft_conditions = {}
        self.workspace_soft_filter = {
            "active": False,
            "predictive": False,
            "alpha": 1.0,
        }
        default_workspace_mode = getattr(
            self.cfg, "workspace_safety_mode", "position_soft")
        workspace_mode = _workspace_safety_mode(
            getattr(options, "workspace_safety_mode", default_workspace_mode),
            default_workspace_mode,
        )
        safety_ball_pos = np.asarray(
            ball_pos if done_ball_pos is None else done_ball_pos,
            dtype=np.float32,
        ).reshape(3)
        safety_ball_vel = np.asarray(
            ball_vel if done_ball_vel is None else done_ball_vel,
            dtype=np.float32,
        ).reshape(3)
        safety_ball_valid = (
            bool(valid) if done_ball_valid is None else bool(done_ball_valid))
        self.last_done_ball_pos = safety_ball_pos.copy()
        self.last_done_ball_valid = safety_ball_valid
        self._release_incompatible_done_latch(workspace_mode)
        if valid:
            self.last_valid_time_s = self.now_s
            if not self.has_captured:
                self.has_captured = True
            self.failsafe_active = False

        if self._reset_recovery_active():
            self.done_failsafe_active = False
            self.done_failsafe_reason = ""
            self.done_failsafe_conditions = {}
            self.failsafe_active = False
            self._sync_policy_to_safe_pose()
            self._mark_policy_paused()
            reset_age_s = (
                max(0.0, float(age_override_s))
                if age_override_s is not None
                else (0.0 if valid else self.cfg.ball_lost_timeout_s)
            )
            return (
                "reset",
                "reset recovery bypasses done",
                {},
                reset_age_s,
                False,
            )

        if self.done_failsafe_active:
            self._mark_policy_paused()
            if not self._hard_workspace_stop_active():
                self._sync_policy_to_safe_pose()
            latched_age_s = (
                max(0.0, float(age_override_s))
                if age_override_s is not None
                else (0.0 if valid else self.cfg.ball_lost_timeout_s)
            )
            reason = self.done_failsafe_reason or "latched; reset required"
            return (
                "done_failsafe", reason,
                dict(self.done_failsafe_conditions), latched_age_s, False,
            )

        if options.enable_ball_capture_gate and not self.has_captured:
            self._sync_policy_to_safe_pose()
            self._mark_policy_paused()
            return "capture_gate", "", {}, self.cfg.ball_lost_timeout_s, False

        if options.enable_ball_release_gate and not self.has_released:
            release_pos = (
                ball_pos if release_ball_pos is None else release_ball_pos)
            release_vel = (
                ball_vel if release_ball_vel is None else release_ball_vel)
            release_is_valid = (
                bool(valid) if release_valid is None else bool(release_valid))
            release_age_s = (
                max(0.0, float(release_age_s))
                if release_age_s is not None
                else (
                    max(0.0, float(age_override_s))
                    if age_override_s is not None
                    else (0.0 if release_is_valid else self.cfg.ball_lost_timeout_s)
                )
            )
            released = self._release_gate_observes_release(
                release_pos, release_vel, release_is_valid,
                age_s=release_age_s,
                options=options)
            if not released:
                self._sync_policy_to_safe_pose()
                self._mark_policy_paused()
                return (
                    "release_gate", self._release_gate_reason(), {},
                    release_age_s, False,
                )
            self.has_released = True
            if (options.estimator_release_gated_dynamics
                    and self.ball_estimator is not None):
                self.ball_estimator.set_ballistic_dynamics(
                    True, reinitialize_from_history=True)
            self._sync_policy_to_safe_pose()
            self._reset_compensator_to_current_arm()

        if age_override_s is not None:
            age_s = max(0.0, float(age_override_s))
        elif self.last_valid_time_s is None:
            age_s = self.cfg.ball_lost_timeout_s
        else:
            age_s = max(0.0, self.now_s - self.last_valid_time_s)

        # The estimator reports a policy-facing age that may be clipped so the
        # RL observation stays in its trained range. Do not use that clipped age
        # for the safety timeout, or a lost estimator state can keep failsafe
        # from ever arming.
        if valid:
            failsafe_age_s = 0.0
        elif self.last_valid_time_s is None:
            failsafe_age_s = self.cfg.ball_lost_timeout_s
        else:
            failsafe_age_s = max(0.0, self.now_s - self.last_valid_time_s)

        if (options.enable_ball_lost_failsafe and not valid
                and failsafe_age_s >= self.cfg.ball_lost_timeout_s):
            self.failsafe_active = True
            self._sync_policy_to_safe_pose()
            self._mark_policy_paused()
            return "ball_lost_failsafe", "", {}, failsafe_age_s, False

        if (self.policy_needs_reentry_reset
                and self.cfg.enable_sim_mirror
                and bool(options.enable_sim_mirror)):
            self._reset_real_policy_for_sim_reentry()

        arm_q = None
        arm_dq = None
        if options.use_ros_state and self.ros_bridge is not None:
            arm_q, arm_dq = self.ros_bridge.get_joint_feedback()
        self.controller.predict(
            ball_pos,
            ball_vel,
            ball_valid=valid,
            ball_obs_age_s=age_s,
            dt=self.cfg.control_dt,
            arm_q=arm_q,
            arm_dq=arm_dq,
        )
        policy_obs_action = self.controller.latest_obs_and_action()
        policy_command = self.controller.latest_command_state()
        self.last_policy_execution_trace = {
            "obs": np.asarray(
                policy_obs_action["obs"], dtype=np.float32).copy(),
            "action": np.asarray(
                policy_obs_action["action"], dtype=np.float32).copy(),
            "position": np.asarray(
                policy_command["position"], dtype=np.float32).copy(),
            "velocity": np.asarray(
                policy_command["velocity"], dtype=np.float32).copy(),
            "acceleration": np.asarray(
                policy_command["acceleration"], dtype=np.float32).copy(),
        }
        racket_pos, racket_quat, fk_ok = self._racket_pose()
        _cart_pos, cart_quat, cart_psi = self._ros_cart_pose()
        ee_quat = cart_quat if cart_quat is not None else racket_quat
        motion = self._done_motion_estimate(
            racket_pos if fk_ok else None, ee_quat, cart_psi)
        self._apply_workspace_safety_mode(workspace_mode, options)
        if workspace_mode == "off":
            self.done_failsafe_active = False
            self.ball_done_streaks = {}
            self.ball_done_raw_conditions = {}
            if hasattr(self.done_monitor, "last_prediction"):
                self.done_monitor.last_prediction = {}
            return "policy", "", {}, age_s, True
        if options.enable_done_monitoring and fk_ok:
            _is_done, _reason, conditions = self.done_monitor.check_done(
                ball_pos_m=safety_ball_pos,
                ball_vel_m_s=safety_ball_vel,
                racket_pos_m=racket_pos,
                racket_anchor_m=self.anchor(),
                base_pos_m=None,
                ee_quat_xyzw=ee_quat,
                ee_psi_rad=cart_psi,
                racket_vel_m_s=motion.get("racket_vel_m_s"),
                racket_acc_m_s2=motion.get("racket_acc_m_s2"),
                ee_psi_vel_rad_s=motion.get("ee_psi_vel_rad_s"),
                ee_abc_vel_rad_s=motion.get("ee_abc_vel_rad_s"),
                check_ball_conditions=safety_ball_valid,
            )
            conditions = self._debounce_ball_done_conditions(conditions)
            is_done = any(conditions.values())
            if is_done:
                workspace_soft = workspace_mode in WORKSPACE_SOFT_MODES
                if workspace_soft:
                    self.workspace_soft_conditions = {
                        str(name): bool(active)
                        for name, active in conditions.items()
                        if (bool(active)
                            and _is_soft_translation_workspace_condition(name))
                    }
                    self.workspace_soft_prediction_active = any(
                        "predicted" in name
                        for name in self.workspace_soft_conditions
                    )
                hard_conditions = _hard_done_conditions(
                    conditions, workspace_mode)
                if not hard_conditions:
                    return "policy", "", conditions, age_s, True
                self.done_failsafe_active = True
                self.done_failsafe_reason = next(iter(hard_conditions))
                self.done_failsafe_conditions = dict(hard_conditions)
                hard_workspace_stop = (
                    workspace_mode in WORKSPACE_HARD_MODES
                    and _has_arm_workspace_condition(hard_conditions)
                )
                if hard_workspace_stop:
                    self._latch_hard_workspace_stop(
                        self.done_failsafe_reason,
                        hard_conditions,
                    )
                    self._mark_policy_paused()
                else:
                    self._sync_policy_to_safe_pose()
                # Non-stopping done modes still publish a parking command after
                # this policy tick.  Hard workspace stop rejects that policy
                # command immediately, so the sim mirror must pause too.
                self.policy_needs_reentry_reset = True
                return (
                    "done_failsafe",
                    self.done_failsafe_reason,
                    dict(hard_conditions),
                    age_s,
                    not hard_workspace_stop,
                )
        else:
            self.ball_done_streaks = {}
            self.ball_done_raw_conditions = {}
        return "policy", "", {}, age_s, True

    def _release_gate_observes_release(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid: bool,
        age_s: float,
        options: RuntimeOptions,
    ) -> bool:
        if not valid:
            self.release_samples.clear()
            self.release_candidate_ticks = 0
            self.last_release_gate_info = self._release_gate_payload(
                valid=False,
                age_s=age_s,
                vz_m_s=None,
                z_drop_m=0.0,
                candidate=False,
                options=options,
            )
            return False
        if age_s > options.ball_release_max_obs_age_s:
            self.release_samples.clear()
            self.release_candidate_ticks = 0
            vz = float(ball_vel[2]) if np.isfinite(float(ball_vel[2])) else None
            self.last_release_gate_info = self._release_gate_payload(
                valid=True,
                age_s=age_s,
                vz_m_s=vz,
                z_drop_m=0.0,
                candidate=False,
                options=options,
            )
            return False
        z = float(ball_pos[2])
        vz = float(ball_vel[2])
        self.release_samples.append((self.now_s, z, vz))
        while (self.release_samples
               and self.now_s - self.release_samples[0][0]
               > options.ball_release_window_s):
            self.release_samples.pop(0)
        max_z = max(sample[1] for sample in self.release_samples)
        z_drop = max(0.0, max_z - z)
        candidate = (
            vz <= -abs(options.ball_release_min_downward_velocity_m_s)
            and z_drop >= options.ball_release_min_z_drop_m)
        if candidate:
            self.release_candidate_ticks += 1
        else:
            self.release_candidate_ticks = 0
        self.last_release_gate_info = self._release_gate_payload(
            valid=True,
            age_s=age_s,
            vz_m_s=vz,
            z_drop_m=z_drop,
            candidate=candidate,
            options=options,
        )
        return (
            self.release_candidate_ticks
            >= options.ball_release_required_consecutive_ticks)

    def _release_gate_payload(
        self,
        valid: bool,
        age_s: float | None,
        vz_m_s: float | None,
        z_drop_m: float,
        candidate: bool,
        options: RuntimeOptions | None = None,
    ) -> dict[str, Any]:
        release_enabled = (
            bool(options.enable_ball_release_gate)
            if options is not None else bool(self.cfg.enable_ball_release_gate))
        max_obs_age_s = (
            float(options.ball_release_max_obs_age_s)
            if options is not None else float(self.cfg.ball_release_max_obs_age_s))
        min_downward_velocity_m_s = (
            float(options.ball_release_min_downward_velocity_m_s)
            if options is not None
            else float(self.cfg.ball_release_min_downward_velocity_m_s))
        min_z_drop_m = (
            float(options.ball_release_min_z_drop_m)
            if options is not None else float(self.cfg.ball_release_min_z_drop_m))
        required_ticks = (
            int(options.ball_release_required_consecutive_ticks)
            if options is not None
            else int(self.cfg.ball_release_required_consecutive_ticks))
        return {
            "enabled": release_enabled,
            "released": bool(getattr(self, "has_released", False)),
            "valid": bool(valid),
            "ageS": (
                None if age_s is None or not math.isfinite(float(age_s))
                else round(float(age_s), 6)),
            "fresh": (
                bool(age_s is not None and math.isfinite(float(age_s))
                     and float(age_s) <= max_obs_age_s)),
            "vzMS": (
                None if vz_m_s is None or not math.isfinite(float(vz_m_s))
                else round(float(vz_m_s), 6)),
            "zDropM": round(float(max(0.0, z_drop_m)), 6),
            "candidate": bool(candidate),
            "candidateTicks": int(getattr(self, "release_candidate_ticks", 0)),
            "requiredTicks": required_ticks,
            "minDownwardVelocityMS": min_downward_velocity_m_s,
            "minZDropM": min_z_drop_m,
            "maxObsAgeS": max_obs_age_s,
        }

    def _release_gate_reason(self) -> str:
        info = getattr(self, "last_release_gate_info", {})
        vz = info.get("vzMS")
        drop = info.get("zDropM")
        ticks = info.get("candidateTicks", 0)
        required = info.get(
            "requiredTicks", self.cfg.ball_release_required_consecutive_ticks)
        if not info.get("valid", False):
            return "waiting for valid ball before release gate clears"
        if not info.get("fresh", False):
            return "waiting for fresh ball observation before release gate clears"
        return (
            "waiting for downward release "
            f"(vz={vz if vz is not None else 'nan'} m/s, "
            f"z_drop={drop} m, ticks={ticks}/{required})")

    def _done_preview_payload(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        valid: bool,
        racket_pos_m: np.ndarray | None,
        fk_ok: bool,
        ee_quat_xyzw: np.ndarray | None = None,
        ee_psi_rad: float | None = None,
        workspace_safety_mode: str | None = None,
    ) -> dict[str, Any]:
        mode = _workspace_safety_mode(
            workspace_safety_mode, self.cfg.workspace_safety_mode)
        if mode == "off":
            return {
                "done": False,
                "reason": "",
                "conditions": {},
                "workspaceSafetyMode": mode,
            }
        workspace_enabled = mode != "off"
        ee_x_limits_m = self.cfg.ee_x_limits_m if workspace_enabled else None
        ee_y_limits_m = self.cfg.ee_y_limits_m if workspace_enabled else None
        ee_z_limits_m = self.cfg.ee_z_limits_m if workspace_enabled else None
        conds = {
            "ball_too_low": False,
            "ball_too_high": False,
            "ball_x_out_of_bounds": False,
            "ball_y_out_of_bounds": False,
            "base_x_out_of_bounds": False,
            "base_y_out_of_bounds": False,
            "racket_too_far_from_anchor": False,
            "racket_too_high": False,
            "racket_too_low": False,
        }
        if ee_x_limits_m is not None:
            conds["ee_x_out_of_workspace"] = False
        if ee_y_limits_m is not None:
            conds["ee_y_out_of_workspace"] = False
        if ee_z_limits_m is not None:
            conds["ee_z_out_of_workspace"] = False
        if self.cfg.ee_psi_limits_rad is not None:
            conds["ee_psi_out_of_bounds"] = False
        b_constraints_enabled = (
            self.cfg.ee_b_limits_rad is not None
            or self.cfg.ee_b_min_rad is not None
            or self.cfg.ee_b_max_rad is not None)
        c_constraints_enabled = (
            self.cfg.ee_c_limits_rad is not None
            or self.cfg.ee_c_min_rad is not None
            or self.cfg.ee_c_max_rad is not None)
        if b_constraints_enabled:
            conds["ee_b_out_of_bounds"] = False
        if c_constraints_enabled:
            conds["ee_c_out_of_bounds"] = False
        if (self.cfg.ee_orientation_reference_xyzw is not None
                and self.cfg.ee_max_orientation_error_deg > 0.0):
            conds["ee_orientation_error_too_large"] = False

        if valid:
            ball = np.asarray(ball_pos_m, dtype=np.float64).reshape(3)
            anchor = np.asarray(self.anchor(), dtype=np.float64).reshape(3)
            conds["ball_too_low"] = bool(ball[2] < self.cfg.done_z_min_m)
            conds["ball_too_high"] = bool(ball[2] > self.cfg.done_z_max_m)
            conds["ball_x_out_of_bounds"] = bool(
                abs(ball[0] - anchor[0]) > self.cfg.done_x_range_m)
            conds["ball_y_out_of_bounds"] = bool(
                abs(ball[1] - anchor[1]) > self.cfg.done_y_range_m)
        anchor = np.asarray(self.anchor(), dtype=np.float64).reshape(3)
        if fk_ok and racket_pos_m is not None:
            racket = np.asarray(racket_pos_m, dtype=np.float64).reshape(3)
            dist = float(np.linalg.norm(racket - anchor))
            dz = float(racket[2] - anchor[2])
            conds["racket_too_far_from_anchor"] = bool(
                dist > self.cfg.done_racket_anchor_distance_m)
            conds["racket_too_high"] = bool(
                dz > self.cfg.done_racket_z_high_offset_m)
            conds["racket_too_low"] = bool(
                dz < self.cfg.done_racket_z_low_offset_m)
            if ee_x_limits_m is not None:
                lo, hi = ee_x_limits_m
                conds["ee_x_out_of_workspace"] = not bool(lo <= racket[0] <= hi)
            if ee_y_limits_m is not None:
                lo, hi = ee_y_limits_m
                conds["ee_y_out_of_workspace"] = not bool(lo <= racket[1] <= hi)
            if ee_z_limits_m is not None:
                lo, hi = ee_z_limits_m
                conds["ee_z_out_of_workspace"] = not bool(lo <= racket[2] <= hi)
            if self.cfg.ee_psi_limits_rad is not None:
                psi_ok = (
                    ee_psi_rad is not None
                    and math.isfinite(float(ee_psi_rad))
                    and _angle_in_limits_rad(
                        float(ee_psi_rad), self.cfg.ee_psi_limits_rad))
                conds["ee_psi_out_of_bounds"] = not bool(psi_ok)
            if b_constraints_enabled or c_constraints_enabled:
                abc = _quat_to_euler_xyz_rad(ee_quat_xyzw)
                if b_constraints_enabled:
                    b_ok = (
                        abc is not None
                        and math.isfinite(float(abc[1]))
                        and (
                            self.cfg.ee_b_limits_rad is None
                            or _angle_in_limits_rad(
                                float(abc[1]), self.cfg.ee_b_limits_rad))
                        and (
                            self.cfg.ee_b_min_rad is None
                            or _angle_above_min_rad(
                                float(abc[1]), self.cfg.ee_b_min_rad))
                        and (
                            self.cfg.ee_b_max_rad is None
                            or _angle_below_max_rad(
                                float(abc[1]), self.cfg.ee_b_max_rad)))
                    conds["ee_b_out_of_bounds"] = not bool(b_ok)
                if c_constraints_enabled:
                    c_ok = (
                        abc is not None
                        and math.isfinite(float(abc[2]))
                        and (
                            self.cfg.ee_c_limits_rad is None
                            or _angle_in_limits_rad(
                                float(abc[2]), self.cfg.ee_c_limits_rad))
                        and (
                            self.cfg.ee_c_min_rad is None
                            or _angle_above_min_rad(
                                float(abc[2]), self.cfg.ee_c_min_rad))
                        and (
                            self.cfg.ee_c_max_rad is None
                            or _angle_below_max_rad(
                                float(abc[2]), self.cfg.ee_c_max_rad)))
                    conds["ee_c_out_of_bounds"] = not bool(c_ok)
            if (self.cfg.ee_orientation_reference_xyzw is not None
                    and self.cfg.ee_max_orientation_error_deg > 0.0):
                angle_error_deg = _quat_angle_error_deg(
                    ee_quat_xyzw, self.cfg.ee_orientation_reference_xyzw)
                conds["ee_orientation_error_too_large"] = bool(
                    angle_error_deg is not None
                    and angle_error_deg
                    > self.cfg.ee_max_orientation_error_deg)

        active = [name for name, on in conds.items() if on]
        hard_active = list(_hard_done_conditions(conds, mode))
        soft_active = [
            name for name in active
            if mode in WORKSPACE_SOFT_MODES
            and _is_soft_translation_workspace_condition(name)
        ]
        return {
            "done": bool(hard_active),
            "reason": hard_active[0] if hard_active else "",
            "conditions": conds,
            "softWorkspaceActive": bool(soft_active),
            "softWorkspaceConditions": soft_active,
            "workspaceSafetyMode": mode,
            "valid": bool(valid),
            "fkOk": bool(fk_ok),
            "ball": [round(float(x), 6) for x in np.asarray(
                ball_pos_m, dtype=np.float64).reshape(3)],
            "velocity": [round(float(x), 6) for x in np.asarray(
                ball_vel_m_s, dtype=np.float64).reshape(3)],
            "anchor": [round(float(x), 6) for x in self.anchor()],
        }

    def _apply_ball_noise(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        valid: bool,
        noise_std_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not valid or noise_std_m <= 0.0:
            return ball_pos.copy(), ball_vel.copy()
        pos_noise = np.random.normal(0.0, noise_std_m, 3).astype(np.float32)
        # Training-time observation noise commonly used ~3 mm position and
        # ~0.03 m/s velocity, so use a 10 s^-1 ratio for interactive probing.
        vel_noise = np.random.normal(0.0, noise_std_m * 10.0, 3).astype(np.float32)
        return (ball_pos + pos_noise).astype(np.float32), (ball_vel + vel_noise).astype(np.float32)

    def _runtime_options_from_payload(self, payload: dict[str, Any]) -> RuntimeOptions:
        use_ros_state = bool(payload.get("useRosState", self.cfg.ros_bridge))
        use_ros_state = bool(
            use_ros_state and self.cfg.ros_bridge and self.ros_bridge is not None)
        # Command publication is an explicit runtime arm action.  The ROS
        # bridge flag only grants permission; it must not start the 200 Hz
        # publisher while the backend is still booting and before the browser
        # (or an experiment client) has deliberately sent publishCommands=true.
        publish_commands = bool(payload.get("publishCommands", False))
        publish_commands = bool(
            publish_commands
            and self.ros_bridge is not None
            and self.cfg.ros_publish_commands
        )
        if publish_commands:
            use_ros_state = True
        ball_source = str(payload.get("ballSource", "drag")).strip().lower()
        if ball_source not in ("drag", "camera"):
            ball_source = "drag"
        if ball_source == "camera" and not self.cfg.enable_camera_ball:
            ball_source = "drag"
        obs_rate = float(payload.get(
            "ballObservationRateHz", self.cfg.ball_observation_rate_hz))
        if not math.isfinite(obs_rate) or obs_rate <= 0.0:
            obs_rate = self.cfg.ball_observation_rate_hz
        obs_rate = min(max(obs_rate, 1.0), 240.0)
        comp_mode = str(payload.get(
            "compensationMode", self.cfg.compensation_mode)).strip().lower()
        comp_mode = comp_mode.replace("-", "_")
        if comp_mode not in ("none", "lead", "inverse_smith", "inverse_mpc"):
            comp_mode = str(self.cfg.compensation_mode).strip().lower()
        comp_delay_ms = _payload_float(
            payload, "compDelayMs", self.cfg.comp_delay_ms,
            min_value=0.0, max_value=250.0)
        comp_tau_s = _payload_float(
            payload, "compActuatorTauS", self.cfg.comp_actuator_tau_s,
            min_value=0.0, max_value=0.5)
        comp_gain = _payload_float(
            payload, "compActuatorGain", self.cfg.comp_actuator_gain,
            min_value=0.1, max_value=3.0)
        comp_mpc_beta = _payload_float(
            payload, "compMpcBeta", self.cfg.comp_mpc_beta,
            min_value=0.0, max_value=5.0)
        comp_mpc_delay_scale = _payload_float(
            payload, "compMpcDelayScale", self.cfg.comp_mpc_delay_scale,
            min_value=0.0, max_value=3.0)
        comp_mpc_tau_scale = _payload_float(
            payload, "compMpcTauScale", self.cfg.comp_mpc_tau_scale,
            min_value=0.0, max_value=3.0)
        comp_mpc_horizon_steps = _payload_int(
            payload, "compMpcHorizonSteps", self.cfg.comp_mpc_horizon_steps,
            min_value=1, max_value=32)
        comp_mpc_tracking_weight = _payload_float(
            payload, "compMpcTrackingWeight",
            self.cfg.comp_mpc_tracking_weight,
            min_value=0.0, max_value=10.0)
        comp_mpc_nominal_weight = _payload_float(
            payload, "compMpcNominalWeight",
            self.cfg.comp_mpc_nominal_weight,
            min_value=0.0, max_value=10.0)
        comp_mpc_delta_weight = _payload_float(
            payload, "compMpcDeltaWeight", self.cfg.comp_mpc_delta_weight,
            min_value=0.0, max_value=10.0)
        comp_mpc_max_delta_rad = _payload_float(
            payload, "compMpcMaxDeltaRad", self.cfg.comp_mpc_max_delta_rad,
            min_value=0.0, max_value=2.0)
        release_min_downward_velocity = _payload_float(
            payload,
            "ballReleaseMinDownwardVelocityMS",
            self.cfg.ball_release_min_downward_velocity_m_s,
            min_value=0.0,
            max_value=5.0,
        )
        release_min_z_drop = _payload_float(
            payload,
            "ballReleaseMinZDropM",
            self.cfg.ball_release_min_z_drop_m,
            min_value=0.0,
            max_value=1.0,
        )
        release_window_s = _payload_float(
            payload,
            "ballReleaseWindowS",
            self.cfg.ball_release_window_s,
            min_value=1e-3,
            max_value=2.0,
        )
        release_required_ticks = _payload_int(
            payload,
            "ballReleaseRequiredTicks",
            self.cfg.ball_release_required_consecutive_ticks,
            min_value=1,
            max_value=200,
        )
        release_max_obs_age = _payload_float(
            payload,
            "ballReleaseMaxObsAgeS",
            self.cfg.ball_release_max_obs_age_s,
            min_value=0.0,
            max_value=2.0,
        )
        estimator_max_extrapolation = _payload_float(
            payload,
            "ballEstimatorMaxExtrapolationS",
            self.cfg.ball_estimator_max_extrapolation_s,
            min_value=0.0,
            max_value=2.0,
        )
        estimator_control_max_extrapolation = _payload_float(
            payload,
            "ballEstimatorControlMaxExtrapolationS",
            self.cfg.ball_estimator_control_max_extrapolation_s,
            min_value=-1.0,
            max_value=2.0,
        )
        estimator_frame_hold_hz = _payload_float(
            payload,
            "ballEstimatorFrameHoldOutputRateHz",
            self.cfg.ball_estimator_frame_hold_output_rate_hz,
            min_value=1.0,
            max_value=240.0,
        )
        estimator_trust_loss = _payload_float(
            payload,
            "ballEstimatorTrustLossS",
            self.cfg.ball_estimator_trust_loss_s,
            min_value=0.0,
            max_value=2.0,
        )
        estimator_degrade_loss = _payload_float(
            payload,
            "ballEstimatorDegradeLossS",
            self.cfg.ball_estimator_degrade_loss_s,
            min_value=0.0,
            max_value=3.0,
        )
        estimator_invalid_after = _payload_float(
            payload,
            "ballEstimatorInvalidAfterS",
            self.cfg.ball_estimator_invalid_after_s,
            min_value=0.0,
            max_value=5.0,
        )
        estimator_prediction_horizon = _payload_float(
            payload,
            "ballEstimatorPredictionHorizonS",
            self.cfg.ball_estimator_prediction_horizon_s,
            min_value=0.0,
            max_value=1.0,
        )
        estimator_q_pos_mm2_s = _payload_float(
            payload,
            "ballEstimatorQPosMm2S",
            self.cfg.ball_estimator_q_pos_mm2_s,
            min_value=0.0,
            max_value=1e6,
        )
        estimator_q_vel_mult = _payload_float(
            payload,
            "ballEstimatorQVelMult",
            self.cfg.ball_estimator_q_vel_mult,
            min_value=0.0,
            max_value=1e5,
        )
        estimator_q_depth_pos_mult = _payload_float(
            payload,
            "ballEstimatorQDepthPosMult",
            self.cfg.ball_estimator_q_depth_pos_mult,
            min_value=1e-6,
            max_value=100.0,
        )
        estimator_q_depth_vel_mult = _payload_float(
            payload,
            "ballEstimatorQDepthVelMult",
            self.cfg.ball_estimator_q_depth_vel_mult,
            min_value=1e-6,
            max_value=100.0,
        )
        estimator_r_lateral_mm = _payload_float(
            payload,
            "ballEstimatorRLateralMm",
            self.cfg.ball_estimator_r_lateral_mm,
            min_value=1.0,
            max_value=500.0,
        )
        estimator_r_depth_mm = _payload_float(
            payload,
            "ballEstimatorRDepthMm",
            self.cfg.ball_estimator_r_depth_mm,
            min_value=1.0,
            max_value=1000.0,
        )
        done_prediction_horizon_s = _payload_float(
            payload,
            "donePredictionHorizonS",
            self.cfg.done_prediction_horizon_s,
            min_value=0.0,
            max_value=0.5,
        )
        done_prediction_dt_s = _payload_float(
            payload,
            "donePredictionDtS",
            self.cfg.done_prediction_dt_s,
            min_value=0.001,
            max_value=0.1,
        )
        done_prediction_margin_m = _payload_float(
            payload,
            "donePredictionMarginM",
            self.cfg.done_prediction_margin_m,
            min_value=0.0,
            max_value=0.15,
        )
        done_prediction_orientation_margin_rad = _payload_float(
            payload,
            "donePredictionOrientationMarginRad",
            self.cfg.done_prediction_orientation_margin_rad,
            min_value=0.0,
            max_value=0.5,
        )
        default_frame_hold = (
            self.cfg.ball_estimator_control_output_mode
            == CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD)
        estimator_output_mode = (
            CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD
            if bool(payload.get("estimatorFrameHold", default_frame_hold))
            else CONTROL_OUTPUT_CONTINUOUS)
        camera_options = CameraRuntimeOptions(
            detect=bool(payload.get(
                "cameraDetect", self.cfg.enable_camera_detection)),
            visualize=bool(payload.get(
                "cameraVisualize", self.cfg.enable_camera_visualization)),
            use_depth_sensor=bool(payload.get(
                "cameraSensorDepth", self.cfg.use_depth_sensor)),
            use_undistort=bool(payload.get(
                "cameraUndistort", self.cfg.use_undistort)),
            use_kalman=bool(payload.get(
                "cameraKalman", self.cfg.use_kalman)),
            enable_ellipse_fit=bool(payload.get(
                "cameraEllipseFit", self.cfg.enable_ellipse_fit)),
            enable_temporal_tracker=bool(payload.get(
                "cameraTemporalTracker", self.cfg.enable_temporal_tracker)),
            enable_color_filter=bool(payload.get(
                "cameraColorFilter", self.cfg.enable_camera_color_filter)),
        )
        workspace_mode = _workspace_safety_mode(
            payload.get("workspaceSafetyMode", self.cfg.workspace_safety_mode),
            self.cfg.workspace_safety_mode,
        )
        return RuntimeOptions(
            use_ros_state=use_ros_state,
            publish_commands=publish_commands,
            enable_compensation=bool(payload.get(
                "enableCompensation", self.cfg.enable_compensation)),
            compensation_mode=comp_mode,
            comp_delay_ms=comp_delay_ms,
            comp_actuator_tau_s=comp_tau_s,
            comp_actuator_gain=comp_gain,
            comp_mpc_beta=comp_mpc_beta,
            comp_mpc_delay_scale=comp_mpc_delay_scale,
            comp_mpc_tau_scale=comp_mpc_tau_scale,
            comp_mpc_horizon_steps=comp_mpc_horizon_steps,
            comp_mpc_tracking_weight=comp_mpc_tracking_weight,
            comp_mpc_nominal_weight=comp_mpc_nominal_weight,
            comp_mpc_delta_weight=comp_mpc_delta_weight,
            comp_mpc_max_delta_rad=comp_mpc_max_delta_rad,
            rl_use_valid_ball_obs_age=bool(payload.get(
                "rlUseValidBallObsAge", self.cfg.rl_use_valid_ball_obs_age)),
            enable_ball_capture_gate=bool(payload.get(
                "enableCaptureGate", self.cfg.enable_ball_capture_gate)),
            enable_ball_release_gate=bool(payload.get(
                "enableReleaseGate", self.cfg.enable_ball_release_gate)),
            enable_ball_lost_failsafe=bool(payload.get(
                "enableLostFailsafe", self.cfg.enable_ball_lost_failsafe)),
            enable_done_monitoring=bool(payload.get(
                "enableDoneMonitoring", self.cfg.enable_done_monitoring)),
            workspace_safety_mode=workspace_mode,
            done_prediction_horizon_s=done_prediction_horizon_s,
            done_prediction_dt_s=done_prediction_dt_s,
            done_prediction_margin_m=done_prediction_margin_m,
            done_prediction_orientation_margin_rad=(
                done_prediction_orientation_margin_rad),
            enable_sim_mirror=bool(payload.get(
                "enableSimMirror", self.cfg.sim_mirror_default_enabled)),
            ball_release_use_estimator_state=bool(payload.get(
                "ballReleaseUseEstimatorState",
                self.cfg.ball_release_use_estimator_state)),
            ball_release_min_downward_velocity_m_s=release_min_downward_velocity,
            ball_release_min_z_drop_m=release_min_z_drop,
            ball_release_window_s=release_window_s,
            ball_release_required_consecutive_ticks=release_required_ticks,
            ball_release_max_obs_age_s=release_max_obs_age,
            use_ball_estimator=bool(payload.get(
                "useBallEstimator", self.cfg.enable_ball_state_estimator)),
            estimator_prediction=bool(payload.get(
                "estimatorPrediction", self.cfg.ball_estimator_enable_prediction)),
            estimator_output_mode=estimator_output_mode,
            estimator_aero_drag=bool(payload.get(
                "estimatorAeroDrag", self.cfg.ball_estimator_enable_aero_drag)),
            estimator_release_gated_dynamics=bool(payload.get(
                "ballEstimatorReleaseGatedDynamics",
                self.cfg.ball_estimator_release_gated_dynamics)),
            estimator_max_extrapolation_s=estimator_max_extrapolation,
            estimator_control_max_extrapolation_s=(
                estimator_control_max_extrapolation),
            estimator_frame_hold_output_rate_hz=estimator_frame_hold_hz,
            estimator_trust_loss_s=estimator_trust_loss,
            estimator_degrade_loss_s=estimator_degrade_loss,
            estimator_invalid_after_s=estimator_invalid_after,
            estimator_prediction_horizon_s=estimator_prediction_horizon,
            estimator_q_pos_mm2_s=estimator_q_pos_mm2_s,
            estimator_q_vel_mult=estimator_q_vel_mult,
            estimator_q_depth_pos_mult=estimator_q_depth_pos_mult,
            estimator_q_depth_vel_mult=estimator_q_depth_vel_mult,
            estimator_r_lateral_mm=estimator_r_lateral_mm,
            estimator_r_depth_mm=estimator_r_depth_mm,
            ball_observation_rate_hz=obs_rate,
            ball_source=ball_source,
            camera=camera_options,
        )

    def camera_feedback(self) -> dict[str, Any]:
        if self.ros_bridge is not None and self.cfg.enable_camera_ball:
            try:
                _pos, _vel, _valid, feedback = self.ros_bridge.camera_ball_state()
                return feedback
            except Exception as exc:
                return {
                    "enabled": bool(self.cfg.enable_camera_ball),
                    "available": bool(_CAMERA_BALL_AVAILABLE),
                    "ready": False,
                    "sourceMode": self.cfg.camera_source_mode,
                    "useHybridDetector": bool(self.cfg.use_hybrid_detector),
                    "useDepthSensor": bool(self.cfg.use_depth_sensor),
                    "imageWidth": int(self.cfg.image_width),
                    "imageHeight": int(self.cfg.image_height),
                    "realsenseFps": int(self.cfg.realsense_fps),
                    "cameraInfoReceived": False,
                    "detectorReady": False,
                    "detected": False,
                    "ageS": None,
                    "frame": None,
                    "detectorFrame": None,
                    "controlFrame": "sim_frame",
                    "detectorBall": None,
                    "detectorVelocity": None,
                    "controlBall": None,
                    "controlVelocity": None,
                    "imageTopic": self.cfg.camera_image_topic,
                    "rawImageTopic": self.cfg.camera_raw_image_topic,
                    "imageTransport": self.cfg.camera_image_transport,
                    "activeTransport": None,
                    "imageAgesS": {},
                    "cameraInfoTopic": self.cfg.camera_info_topic,
                    "frameCount": 0,
                    "detectionCount": 0,
                    "callbackHz": 0.0,
                    "visualizationRateHz": round(
                        float(self.cfg.camera_visualization_rate_hz), 3),
                    "webMaxWidth": int(self.cfg.camera_web_max_width),
                    "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
                    "visualizationSeq": 0,
                    "visualizationAgeS": None,
                    "runtime": {
                        "detect": bool(self.cfg.enable_camera_detection),
                        "visualize": bool(self.cfg.enable_camera_visualization),
                        "sensorDepth": bool(self.cfg.use_depth_sensor),
                        "undistort": bool(self.cfg.use_undistort),
                        "kalman": bool(self.cfg.use_kalman),
                        "ellipseFit": bool(self.cfg.enable_ellipse_fit),
                        "temporalTracker": bool(self.cfg.enable_temporal_tracker),
                        "colorFilter": bool(self.cfg.enable_camera_color_filter),
                    },
                    "error": str(exc),
                }
        return {
            "enabled": bool(self.cfg.enable_camera_ball),
            "available": bool(_CAMERA_BALL_AVAILABLE),
            "ready": False,
            "sourceMode": self.cfg.camera_source_mode,
            "useHybridDetector": bool(self.cfg.use_hybrid_detector),
            "useDepthSensor": bool(self.cfg.use_depth_sensor),
            "imageWidth": int(self.cfg.image_width),
            "imageHeight": int(self.cfg.image_height),
            "realsenseFps": int(self.cfg.realsense_fps),
            "cameraInfoReceived": False,
            "detectorReady": False,
            "detected": False,
            "ageS": None,
            "frame": None,
            "detectorFrame": None,
            "controlFrame": "sim_frame",
            "detectorBall": None,
            "detectorVelocity": None,
            "controlBall": None,
            "controlVelocity": None,
            "imageTopic": self.cfg.camera_image_topic,
            "rawImageTopic": self.cfg.camera_raw_image_topic,
            "imageTransport": self.cfg.camera_image_transport,
            "activeTransport": None,
            "imageAgesS": {},
            "cameraInfoTopic": self.cfg.camera_info_topic,
            "frameCount": 0,
            "detectionCount": 0,
            "callbackHz": 0.0,
            "visualizationRateHz": round(
                float(self.cfg.camera_visualization_rate_hz), 3),
            "webMaxWidth": int(self.cfg.camera_web_max_width),
            "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
            "visualizationSeq": 0,
            "visualizationAgeS": None,
            "runtime": {
                "detect": bool(self.cfg.enable_camera_detection),
                "visualize": bool(self.cfg.enable_camera_visualization),
                "sensorDepth": bool(self.cfg.use_depth_sensor),
                "undistort": bool(self.cfg.use_undistort),
                "kalman": bool(self.cfg.use_kalman),
                "ellipseFit": bool(self.cfg.enable_ellipse_fit),
                "temporalTracker": bool(self.cfg.enable_temporal_tracker),
                "colorFilter": bool(self.cfg.enable_camera_color_filter),
            },
            "error": None if _CAMERA_BALL_AVAILABLE else str(_CAMERA_BALL_IMPORT_ERROR),
        }

    def latest_camera_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        if self.ros_bridge is not None and self.cfg.enable_camera_ball:
            return self.ros_bridge.latest_camera_visualization()
        return None, {"seq": 0, "ageS": None, "enabled": False}

    def latest_sim_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        if not bool(self.cfg.sim_enable_video_frames):
            return None, {"seq": 0, "ageS": None, "enabled": False}
        if self.sim_mirror is None:
            return None, {"seq": 0, "ageS": None, "enabled": False}
        return self.sim_mirror.latest_visualization()

    def _select_ball_source(
        self,
        drag_pos: np.ndarray,
        drag_vel: np.ndarray,
        drag_valid: bool,
        options: RuntimeOptions,
        include_feedback: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, bool, str, dict[str, Any] | None, float | None]:
        camera_feedback = self.camera_feedback() if include_feedback else None
        if options.ball_source != "camera":
            return drag_pos, drag_vel, bool(drag_valid), "drag", camera_feedback, None
        if self.ros_bridge is None:
            if camera_feedback is None:
                camera_feedback = self.camera_feedback()
            camera_feedback["error"] = (
                camera_feedback.get("error") or "ROS node is not active")
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                "camera",
                camera_feedback,
                None,
            )
        camera_stamp_s = None
        if include_feedback:
            cam_pos, cam_vel, cam_valid, camera_feedback = (
                self.ros_bridge.camera_ball_state())
            if isinstance(camera_feedback, dict):
                raw_stamp = camera_feedback.get("stampS")
                if raw_stamp is not None:
                    try:
                        camera_stamp_s = float(raw_stamp)
                    except (TypeError, ValueError):
                        camera_stamp_s = None
        else:
            cam_pos, cam_vel, cam_valid, camera_stamp_s = (
                self.ros_bridge.camera_ball_measurement())
        if not cam_valid:
            cam_pos = self.last_ball_pos.copy()
            cam_vel = self.last_ball_vel.copy()
        if not drag_valid:
            cam_valid = False
        return cam_pos, cam_vel, cam_valid, "camera", camera_feedback, camera_stamp_s

    def _record_control_sample(
        self,
        *,
        source_seq: int,
        source_time_s: float,
        ball_observation_seq: int,
        camera_stamp_s: float | None,
        mode: str,
        policy_executed: bool,
        reason: str,
        conditions: dict[str, bool],
        raw_ball_pos: np.ndarray,
        measured_ball_pos: np.ndarray,
        measured_ball_vel: np.ndarray,
        observed_ball_pos: np.ndarray,
        observed_ball_vel: np.ndarray,
        raw_valid: bool,
        control_valid: bool,
        age_s: float,
        noise_std_m: float,
        options: RuntimeOptions,
        ball_source: str,
        command_published: bool,
    ) -> None:
        if not self.recorder.active:
            return
        cmd = self.controller.latest_command_state()
        obs_action = self.controller.latest_obs_and_action()
        trace = (
            self.last_policy_execution_trace
            if policy_executed
            and isinstance(self.last_policy_execution_trace, dict)
            else None
        )
        if trace is not None:
            policy_obs = np.asarray(trace["obs"], dtype=np.float32)
            policy_action = np.asarray(trace["action"], dtype=np.float32)
            policy_cmd = {
                "position": np.asarray(trace["position"], dtype=np.float32),
                "velocity": np.asarray(trace["velocity"], dtype=np.float32),
                "acceleration": np.asarray(
                    trace["acceleration"], dtype=np.float32),
            }
        else:
            policy_obs = np.asarray(obs_action["obs"], dtype=np.float32)
            policy_action = np.asarray(obs_action["action"], dtype=np.float32)
            policy_cmd = cmd
        ros_light: dict[str, Any] = {"enabled": bool(self.ros_bridge is not None)}
        if self.ros_bridge is not None:
            q_fb, dq_fb = self.ros_bridge.get_joint_feedback()
            cart_pos, cart_quat, cart_psi = self.ros_bridge.get_cart_feedback()
            ros_light.update({
                "jointPosition": None if q_fb is None else _rounded_vector(q_fb),
                "jointVelocity": None if dq_fb is None else _rounded_vector(dq_fb),
                "cartPosition": None if cart_pos is None else _rounded_vector(cart_pos),
                "cartQuatXyzw": None if cart_quat is None else _rounded_vector(cart_quat),
                "cartPsi": None if cart_psi is None else round(float(cart_psi), 6),
            })
        sample = {
            "schemaVersion": 3,
            "sourceSeq": int(source_seq),
            "sourceTimeS": round(float(source_time_s), 9),
            "ballObservationSeq": int(ball_observation_seq),
            "cameraStampS": camera_stamp_s,
            "wallTimeS": round(float(time.time()), 6),
            "time": round(float(self.now_s), 6),
            "tick": int(self.control_loop_tick),
            "mode": str(mode),
            "policyExecuted": bool(policy_executed),
            "reason": str(reason),
            "conditions": {k: bool(v) for k, v in conditions.items()},
            "doneBall": {
                "source": "raw_measurement",
                "position": _rounded_vector(self.last_done_ball_pos),
                "valid": bool(self.last_done_ball_valid),
                "rawConditions": dict(self.ball_done_raw_conditions),
                "streaks": dict(self.ball_done_streaks),
                "requiredTicks": int(
                    self.cfg.done_ball_required_consecutive_ticks),
            },
            "ballSource": str(ball_source),
            "rawBall": _rounded_vector(raw_ball_pos),
            "measuredBall": _rounded_vector(measured_ball_pos),
            "measuredVelocity": _rounded_vector(measured_ball_vel),
            "observedBall": _rounded_vector(observed_ball_pos),
            "observedVelocity": _rounded_vector(observed_ball_vel),
            "policyInput": {
                "ball": _float32_vector(observed_ball_pos),
                "velocity": _float32_vector(observed_ball_vel),
                "valid": bool(control_valid),
                "age": float(age_s),
                "ballObservationSeq": int(ball_observation_seq),
            },
            "rawValid": bool(raw_valid),
            "valid": bool(control_valid),
            "age": round(float(age_s), 6),
            "noiseStdM": round(float(noise_std_m), 6),
            "policyObs": _rounded_vector(policy_obs),
            "policyBallObservation": _float32_vector(
                policy_obs[20:26]),
            "policyAction": _rounded_vector(policy_action),
            "policyCommand": {
                "position": _rounded_vector(policy_cmd["position"]),
                "velocity": _rounded_vector(policy_cmd["velocity"]),
                "acceleration": _rounded_vector(policy_cmd["acceleration"]),
            },
            "command": {
                "position": _rounded_vector(cmd["position"]),
                "velocity": _rounded_vector(cmd["velocity"]),
                "acceleration": _rounded_vector(cmd["acceleration"]),
                "policyNominalPosition": _rounded_vector(getattr(
                    self, "last_policy_nominal_reference",
                    policy_cmd["position"])),
                "compensatedReferencePosition": _rounded_vector(getattr(
                    self, "last_compensated_reference",
                    policy_cmd["position"])),
                "publishedSafePosition": _rounded_vector(getattr(
                    self, "last_published_safe_reference",
                    policy_cmd["position"])),
                "published": bool(command_published),
                "hardWorkspaceStop": self._hard_workspace_stop_payload(),
                "resetRecovery": self._reset_recovery_payload(),
                "compensation": self._compensation_payload(options),
            },
            "runtime": {
                "useRosState": bool(options.use_ros_state),
                "publishCommands": bool(options.publish_commands),
                "enableCompensation": bool(options.enable_compensation),
                "useBallEstimator": bool(options.use_ball_estimator),
                "estimatorPrediction": bool(options.estimator_prediction),
                "estimatorOutputMode": str(options.estimator_output_mode),
                "estimatorReleaseGatedDynamics": bool(
                    options.estimator_release_gated_dynamics),
                "releaseGate": bool(options.enable_ball_release_gate),
                "releaseUseEstimatorState": bool(
                    options.ball_release_use_estimator_state),
                "ballObservationRateHz": round(
                    float(options.ball_observation_rate_hz), 6),
                "workspaceSafetyMode": str(options.workspace_safety_mode),
                "donePredictionHorizonS": round(
                    float(options.done_prediction_horizon_s), 6),
                "donePredictionDtS": round(
                    float(options.done_prediction_dt_s), 6),
                "donePredictionMarginM": round(
                    float(options.done_prediction_margin_m), 6),
                "donePredictionOrientationMarginRad": round(
                    float(options.done_prediction_orientation_margin_rad), 6),
            },
            "estimator": dict(self.last_estimator_info),
            "releaseGate": dict(self.last_release_gate_info),
            "ros": ros_light,
            "simSyncLatest": dict(
                (self.latest_sim_snapshot or {}).get("sync") or {}),
        }
        self.recorder.record_control_sample(sample)

    def _run_control_tick_locked(
        self,
        drag_ball_pos: np.ndarray,
        drag_ball_vel: np.ndarray,
        drag_valid: bool,
        noise_std_m: float,
        options: RuntimeOptions,
        make_snapshot: bool = True,
    ) -> dict[str, Any] | None:
        source_seq = int(self.control_loop_tick)
        ball_pos, ball_vel, valid, ball_source, camera_feedback, ball_stamp_s = (
            self._select_ball_source(
                drag_ball_pos, drag_ball_vel, drag_valid, options,
                include_feedback=make_snapshot))
        measured_p, measured_v = self._apply_ball_noise(
            ball_pos, ball_vel, valid, noise_std_m)
        obs_p, obs_v, control_valid, age_override_s, estimator_info = (
            self._resolve_control_ball_state(
                measured_p, measured_v, valid, options,
                measurement_stamp_s=ball_stamp_s))
        release_kwargs = {}
        if not options.ball_release_use_estimator_state:
            release_kwargs = {
                    "release_ball_pos": measured_p,
                    "release_ball_vel": measured_v,
                    "release_valid": valid,
                    "release_age_s": self._raw_measurement_age_s(
                        valid, ball_stamp_s, options),
                }
        (
            mode,
            reason,
            conditions,
            age_s,
            policy_executed,
        ) = self._run_one_control_step(
            obs_p, obs_v, control_valid, age_override_s, options,
            done_ball_pos=ball_pos,
            done_ball_vel=ball_vel,
            done_ball_valid=valid,
            **release_kwargs)
        self.last_ball_pos = ball_pos.copy()
        self.last_ball_vel = ball_vel.copy()
        self.last_observed_ball_pos = obs_p.copy()
        self.last_observed_ball_vel = obs_v.copy()
        self.last_measured_ball_pos = measured_p.copy()
        self.last_measured_ball_vel = measured_v.copy()
        self.last_estimator_info = estimator_info
        command_profile = (
            "normal" if mode == "policy"
            else ("reset" if mode == "reset" else "parking")
        )
        command_published = self._publish_control_command(
            self.controller.arm_cmd_q.copy(), options,
            apply_compensation=(mode == "policy"),
            profile=command_profile)
        # The actuator command is the real-time critical path.  Only after it
        # has been published do we enqueue/poll the asynchronous sim mirror.
        ball_observation_seq = int(
            estimator_info.get("snapshotSequence", 0) or 0)
        self._step_sim_mirror(
            obs_p, obs_v,
            ball_valid=control_valid,
            age_s=age_s,
            options=options,
            source_seq=source_seq,
            source_time_s=float(self.now_s),
            ball_observation_seq=ball_observation_seq,
            camera_stamp_s=ball_stamp_s,
            control_mode=mode,
            policy_active=policy_executed)
        if policy_executed and mode != "policy":
            self.sim_needs_policy_reset = True
        if make_snapshot and self.ros_bridge is not None and self.cfg.enable_camera_ball:
            racket_pos, _fk_ok = self._racket_pos()
            if options.publish_commands:
                ros_racket_pos, ros_fk_ok = self._ros_joint_racket_pos()
                if ros_fk_ok and ros_racket_pos is not None:
                    racket_pos = ros_racket_pos
            self.ros_bridge.set_policy_viz_state(
                obs_p, obs_v, control_valid, self.anchor(), racket_pos)
        self._record_control_sample(
            source_seq=source_seq,
            source_time_s=float(self.now_s),
            ball_observation_seq=ball_observation_seq,
            camera_stamp_s=ball_stamp_s,
            mode=mode,
            policy_executed=policy_executed,
            reason=reason,
            conditions=conditions,
            raw_ball_pos=ball_pos,
            measured_ball_pos=measured_p,
            measured_ball_vel=measured_v,
            observed_ball_pos=obs_p,
            observed_ball_vel=obs_v,
            raw_valid=valid,
            control_valid=control_valid,
            age_s=age_s,
            noise_std_m=noise_std_m,
            options=options,
            ball_source=ball_source,
            command_published=command_published,
        )
        self.control_loop_tick += 1
        if not make_snapshot:
            return None
        if camera_feedback is None:
            camera_feedback = self.camera_feedback()
        return self.snapshot(
            mode, reason, conditions, ball_pos, obs_p, obs_v,
            control_valid, age_s, noise_std_m,
            measured_ball_pos=measured_p,
            measured_ball_vel=measured_v,
            estimator_info=estimator_info,
            raw_valid=valid,
            use_ros_state=options.use_ros_state,
            publish_commands=options.publish_commands,
            command_published=command_published,
            runtime_options=options,
            ball_source=ball_source,
            camera_feedback=camera_feedback)

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("reset"):
            return self.reset()
        drag_ball_pos = vec3_from_payload(payload, "ball", self.initial_ball_pos())
        drag_ball_vel = vec3_from_payload(payload, "velocity", (0.0, 0.0, 0.0))
        drag_valid = bool(payload.get("valid", True))
        noise_std_m = float(payload.get("noiseStdM", self.cfg.ball_noise_std_m))
        if not math.isfinite(noise_std_m) or noise_std_m < 0.0:
            noise_std_m = self.cfg.ball_noise_std_m
        noise_std_m = min(max(noise_std_m, 0.0), 0.25)
        options = self._runtime_options_from_payload(payload)
        dt = float(payload.get("dt", self.cfg.control_dt))
        if not math.isfinite(dt) or dt <= 0.0:
            dt = self.cfg.control_dt
        dt = min(dt, 0.25)
        if (not self.cfg.backend_control_loop
                and self.ros_bridge is not None
                and self.cfg.enable_camera_ball):
            self.ros_bridge.set_camera_runtime_options(options.camera)

        if self.cfg.backend_control_loop:
            recording_view = payload.get("recordingView")
            with self.lock:
                self.runtime_options = options
                if options.publish_commands:
                    lease_ms = _payload_float(
                        payload, "commandLeaseMs", 500.0,
                        min_value=100.0, max_value=2000.0)
                    self.command_publish_lease_deadline_wall = (
                        time.monotonic() + lease_ms * 1e-3)
                    self.command_publish_lease_expired = False
                else:
                    self.command_publish_lease_deadline_wall = 0.0
                    self.command_publish_lease_expired = False
                self.web_drag_ball_pos = drag_ball_pos.copy()
                self.web_drag_ball_vel = drag_ball_vel.copy()
                self.web_drag_valid = bool(drag_valid)
                self.web_noise_std_m = float(noise_std_m)
                snapshot = self.latest_snapshot_payload
                if snapshot is None:
                    snapshot = self.snapshot(
                        "waiting", "", {}, self.last_ball_pos,
                        self.last_observed_ball_pos,
                        self.last_observed_ball_vel,
                        False, 0.0, noise_std_m,
                        measured_ball_pos=self.last_measured_ball_pos,
                        measured_ball_vel=self.last_measured_ball_vel,
                        estimator_info=self.last_estimator_info,
                        ball_source="drag",
                        camera_feedback=self.camera_feedback())
                snapshot = dict(snapshot)
            if isinstance(recording_view, dict):
                self.record_browser_view(recording_view)
            snapshot["recording"] = self.recording_status()
            return snapshot

        with self.lock:
            recording_view = payload.get("recordingView")
            if isinstance(recording_view, dict):
                self.record_browser_view(recording_view)
            ball_pos, ball_vel, valid, ball_source, camera_feedback, ball_stamp_s = (
                self._select_ball_source(
                    drag_ball_pos, drag_ball_vel, drag_valid, options))
            start_pos = self.last_ball_pos.copy()
            start_vel = self.last_ball_vel.copy()
            n_steps = max(1, min(100, int(round(dt / self.cfg.control_dt))))
            mode = "policy"
            reason = ""
            conditions: dict[str, bool] = {}
            age_s = 0.0
            control_valid = bool(valid)
            obs_p = self.last_observed_ball_pos.copy()
            obs_v = self.last_observed_ball_vel.copy()
            measured_p = self.last_measured_ball_pos.copy()
            measured_v = self.last_measured_ball_vel.copy()
            estimator_info = self.last_estimator_info
            source_seq = int(self.control_loop_tick)
            ball_observation_seq = int(
                estimator_info.get("snapshotSequence", 0) or 0)
            for i in range(n_steps):
                source_seq = int(self.control_loop_tick)
                alpha = float(i + 1) / float(n_steps)
                p = start_pos * (1.0 - alpha) + ball_pos * alpha
                v = start_vel * (1.0 - alpha) + ball_vel * alpha
                measured_p, measured_v = self._apply_ball_noise(p, v, valid, noise_std_m)
                obs_p, obs_v, control_valid, age_override_s, estimator_info = (
                    self._resolve_control_ball_state(
                        measured_p, measured_v, valid, options,
                        measurement_stamp_s=ball_stamp_s))
                release_kwargs = {}
                if not options.ball_release_use_estimator_state:
                    release_kwargs = {
                            "release_ball_pos": measured_p,
                            "release_ball_vel": measured_v,
                            "release_valid": valid,
                            "release_age_s": self._raw_measurement_age_s(
                                valid, ball_stamp_s, options),
                        }
                (
                    mode,
                    reason,
                    conditions,
                    age_s,
                    policy_executed,
                ) = self._run_one_control_step(
                    obs_p, obs_v, control_valid, age_override_s, options,
                    done_ball_pos=p,
                    done_ball_vel=v,
                    done_ball_valid=valid,
                    **release_kwargs)
                ball_observation_seq = int(
                    estimator_info.get("snapshotSequence", 0) or 0)
                self._step_sim_mirror(
                    obs_p, obs_v,
                    ball_valid=control_valid,
                    age_s=age_s,
                    options=options,
                    source_seq=source_seq,
                    source_time_s=float(self.now_s),
                    ball_observation_seq=ball_observation_seq,
                    camera_stamp_s=ball_stamp_s,
                    control_mode=mode,
                    policy_active=policy_executed)
                if policy_executed and mode != "policy":
                    self.sim_needs_policy_reset = True
                self.control_loop_tick += 1
            if self.ros_bridge is not None and self.cfg.enable_camera_ball:
                racket_pos, _fk_ok = self._racket_pos()
                if options.publish_commands:
                    ros_racket_pos, ros_fk_ok = self._ros_joint_racket_pos()
                    if ros_fk_ok and ros_racket_pos is not None:
                        racket_pos = ros_racket_pos
                self.ros_bridge.set_policy_viz_state(
                    obs_p, obs_v, control_valid, self.anchor(), racket_pos)
            self.last_ball_pos = ball_pos.copy()
            self.last_ball_vel = ball_vel.copy()
            self.last_observed_ball_pos = obs_p.copy()
            self.last_observed_ball_vel = obs_v.copy()
            self.last_measured_ball_pos = measured_p.copy()
            self.last_measured_ball_vel = measured_v.copy()
            self.last_estimator_info = estimator_info
            command_published = False
            if options.publish_commands or self._hard_workspace_stop_active():
                command_profile = (
                    "normal" if mode == "policy"
                    else ("reset" if mode == "reset" else "parking")
                )
                command_published = self._publish_control_command(
                    self.controller.arm_cmd_q.copy(), options,
                    apply_compensation=(mode == "policy"),
                    profile=command_profile)
            self._record_control_sample(
                source_seq=source_seq,
                source_time_s=float(self.now_s),
                ball_observation_seq=ball_observation_seq,
                camera_stamp_s=ball_stamp_s,
                mode=mode,
                policy_executed=policy_executed,
                reason=reason,
                conditions=conditions,
                raw_ball_pos=ball_pos,
                measured_ball_pos=measured_p,
                measured_ball_vel=measured_v,
                observed_ball_pos=obs_p,
                observed_ball_vel=obs_v,
                raw_valid=valid,
                control_valid=control_valid,
                age_s=age_s,
                noise_std_m=noise_std_m,
                options=options,
                ball_source=ball_source,
                command_published=command_published,
            )
            snapshot = self.snapshot(
                mode, reason, conditions, ball_pos, obs_p, obs_v,
                control_valid, age_s, noise_std_m,
                measured_ball_pos=measured_p,
                measured_ball_vel=measured_v,
                estimator_info=estimator_info,
                raw_valid=valid,
                use_ros_state=options.use_ros_state,
                publish_commands=options.publish_commands,
                command_published=command_published,
                runtime_options=options,
                ball_source=ball_source,
                camera_feedback=camera_feedback)
            self.recorder.record_snapshot(snapshot)
            return snapshot

    def snapshot(
        self,
        mode: str,
        reason: str,
        conditions: dict[str, bool],
        raw_ball_pos: np.ndarray,
        observed_ball_pos: np.ndarray,
        observed_ball_vel: np.ndarray,
        valid: bool,
        age_s: float,
        noise_std_m: float,
        measured_ball_pos: np.ndarray | None = None,
        measured_ball_vel: np.ndarray | None = None,
        estimator_info: dict[str, Any] | None = None,
        raw_valid: bool | None = None,
        use_ros_state: bool | None = None,
        publish_commands: bool | None = None,
        command_published: bool = False,
        runtime_options: RuntimeOptions | None = None,
        ball_source: str = "drag",
        camera_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        racket_pos, racket_quat, fk_ok = self._racket_pose()
        cmd_racket_pos = racket_pos.copy()
        cmd_fk_ok = bool(fk_ok)
        racket_source = "command"
        cmd = self.controller.latest_command_state()
        obs_action = self.controller.latest_obs_and_action()
        measured_ball_pos = (
            observed_ball_pos if measured_ball_pos is None else measured_ball_pos)
        measured_ball_vel = (
            observed_ball_vel if measured_ball_vel is None else measured_ball_vel)
        estimator_info = (
            self._estimator_off_payload()
            if estimator_info is None else estimator_info)
        raw_valid = bool(valid) if raw_valid is None else bool(raw_valid)
        if use_ros_state is None:
            use_ros_state = bool(self.cfg.ros_bridge and self.ros_bridge is not None)
        if publish_commands is None:
            publish_commands = bool(
                self.cfg.ros_publish_commands and self.ros_bridge is not None)
        if runtime_options is None:
            runtime_payload = {
                "useRosState": bool(use_ros_state),
                "publishCommands": bool(publish_commands),
                "captureGate": bool(self.cfg.enable_ball_capture_gate),
                "releaseGate": bool(self.cfg.enable_ball_release_gate),
                "lostFailsafe": bool(self.cfg.enable_ball_lost_failsafe),
                "doneMonitoring": bool(self.cfg.enable_done_monitoring),
                "enableCompensation": bool(self.cfg.enable_compensation),
                "compensationMode": str(self.cfg.compensation_mode),
                "compDelayMs": round(float(self.cfg.comp_delay_ms), 6),
                "compActuatorTauS": round(
                    float(self.cfg.comp_actuator_tau_s), 6),
                "compActuatorGain": round(
                    float(self.cfg.comp_actuator_gain), 6),
                "rlUseValidBallObsAge": bool(self.cfg.rl_use_valid_ball_obs_age),
                "releaseUseEstimatorState": bool(
                    self.cfg.ball_release_use_estimator_state),
                "ballObservationRateHz": round(float(self.cfg.ball_observation_rate_hz), 6),
                "workspaceSafetyMode": str(self.cfg.workspace_safety_mode),
                "donePredictionHorizonS": round(
                    float(self.cfg.done_prediction_horizon_s), 6),
                "donePredictionDtS": round(
                    float(self.cfg.done_prediction_dt_s), 6),
                "donePredictionMarginM": round(
                    float(self.cfg.done_prediction_margin_m), 6),
                "donePredictionOrientationMarginRad": round(
                    float(self.cfg.done_prediction_orientation_margin_rad), 6),
                "enableSimMirror": bool(self.cfg.sim_mirror_default_enabled),
            }
        else:
            runtime_payload = {
                "useRosState": bool(runtime_options.use_ros_state),
                "publishCommands": bool(runtime_options.publish_commands),
                "captureGate": bool(runtime_options.enable_ball_capture_gate),
                "releaseGate": bool(runtime_options.enable_ball_release_gate),
                "lostFailsafe": bool(runtime_options.enable_ball_lost_failsafe),
                "doneMonitoring": bool(runtime_options.enable_done_monitoring),
                "enableCompensation": bool(
                    runtime_options.enable_compensation),
                "compensationMode": str(runtime_options.compensation_mode),
                "compDelayMs": round(float(runtime_options.comp_delay_ms), 6),
                "compActuatorTauS": round(
                    float(runtime_options.comp_actuator_tau_s), 6),
                "compActuatorGain": round(
                    float(runtime_options.comp_actuator_gain), 6),
                "compMpcBeta": round(
                    float(runtime_options.comp_mpc_beta), 6),
                "compMpcDelayScale": round(
                    float(runtime_options.comp_mpc_delay_scale), 6),
                "compMpcTauScale": round(
                    float(runtime_options.comp_mpc_tau_scale), 6),
                "compMpcHorizonSteps": int(
                    runtime_options.comp_mpc_horizon_steps),
                "compMpcTrackingWeight": round(
                    float(runtime_options.comp_mpc_tracking_weight), 6),
                "compMpcNominalWeight": round(
                    float(runtime_options.comp_mpc_nominal_weight), 6),
                "compMpcDeltaWeight": round(
                    float(runtime_options.comp_mpc_delta_weight), 6),
                "compMpcMaxDeltaRad": round(
                    float(runtime_options.comp_mpc_max_delta_rad), 6),
                "rlUseValidBallObsAge": bool(
                    runtime_options.rl_use_valid_ball_obs_age),
                "releaseUseEstimatorState": bool(
                    runtime_options.ball_release_use_estimator_state),
                "ballObservationRateHz": round(float(runtime_options.ball_observation_rate_hz), 6),
                "workspaceSafetyMode": str(runtime_options.workspace_safety_mode),
                "donePredictionHorizonS": round(
                    float(runtime_options.done_prediction_horizon_s), 6),
                "donePredictionDtS": round(
                    float(runtime_options.done_prediction_dt_s), 6),
                "donePredictionMarginM": round(
                    float(runtime_options.done_prediction_margin_m), 6),
                "donePredictionOrientationMarginRad": round(
                    float(runtime_options.done_prediction_orientation_margin_rad), 6),
                "enableSimMirror": bool(runtime_options.enable_sim_mirror),
                "releaseMinDownwardVelocityMS": round(
                    float(runtime_options.ball_release_min_downward_velocity_m_s), 6),
                "releaseMinZDropM": round(
                    float(runtime_options.ball_release_min_z_drop_m), 6),
                "releaseWindowS": round(
                    float(runtime_options.ball_release_window_s), 6),
                "releaseRequiredTicks": int(
                    runtime_options.ball_release_required_consecutive_ticks),
                "releaseMaxObsAgeS": round(
                    float(runtime_options.ball_release_max_obs_age_s), 6),
                "estimatorReleaseGatedDynamics": bool(
                    runtime_options.estimator_release_gated_dynamics),
                "estimatorMaxExtrapolationS": round(
                    float(runtime_options.estimator_max_extrapolation_s), 6),
                "estimatorControlMaxExtrapolationS": round(
                    float(runtime_options.estimator_control_max_extrapolation_s), 6),
                "estimatorFrameHoldOutputRateHz": round(
                    float(runtime_options.estimator_frame_hold_output_rate_hz), 6),
                "estimatorTrustLossS": round(
                    float(runtime_options.estimator_trust_loss_s), 6),
                "estimatorDegradeLossS": round(
                    float(runtime_options.estimator_degrade_loss_s), 6),
                "estimatorInvalidAfterS": round(
                    float(runtime_options.estimator_invalid_after_s), 6),
                "estimatorPredictionHorizonS": round(
                    float(runtime_options.estimator_prediction_horizon_s), 6),
                "estimatorQPosMm2S": round(
                    float(runtime_options.estimator_q_pos_mm2_s), 6),
                "estimatorQVelMult": round(
                    float(runtime_options.estimator_q_vel_mult), 6),
                "estimatorQDepthPosMult": round(
                    float(runtime_options.estimator_q_depth_pos_mult), 6),
                "estimatorQDepthVelMult": round(
                    float(runtime_options.estimator_q_depth_vel_mult), 6),
                "estimatorRLateralMm": round(
                    float(runtime_options.estimator_r_lateral_mm), 6),
                "estimatorRDepthMm": round(
                    float(runtime_options.estimator_r_depth_mm), 6),
            }
        lease_remaining_ms = max(
            0.0,
            (float(getattr(
                self, "command_publish_lease_deadline_wall", 0.0))
             - time.monotonic()) * 1000.0,
        )
        runtime_payload["commandLeaseRemainingMs"] = round(
            lease_remaining_ms, 3)
        runtime_payload["commandLeaseExpired"] = bool(getattr(
            self, "command_publish_lease_expired", False))
        release_gate_info = dict(getattr(self, "last_release_gate_info", {}))
        release_gate_info["enabled"] = bool(
            runtime_options.enable_ball_release_gate
            if runtime_options is not None else self.cfg.enable_ball_release_gate)
        feedback_racket_pos: np.ndarray | None = None
        feedback_racket_quat: np.ndarray | None = None
        feedback_fk_ok = False
        _cart_pos, cart_quat, cart_psi = self._ros_cart_pose()
        ros_feedback = {
            "enabled": bool(self.ros_bridge is not None),
            "useState": bool(use_ros_state),
            "useStateAllowed": bool(self.cfg.ros_bridge),
            "publishAllowed": bool(self.cfg.ros_publish_commands),
            "publishCommands": bool(publish_commands),
            "commandPublished": bool(command_published),
            "commandTopic": self.cfg.command_topic,
            "jointStateTopic": self.cfg.joint_state_topic,
            "cartStateTopic": self.cfg.cart_state_topic,
            "jointPosition": None,
            "jointVelocity": None,
            "jointAgeS": None,
            "cartPosition": None,
            "cartQuatXyzw": None,
            "cartPsi": None,
            "cartAgeS": None,
            "publishCount": 0,
            "jointEe": None,
            "jointEeFkOk": False,
            "jointCommandErrorAbsMax": None,
            "cartEePoseAvailable": bool(cart_quat is not None or cart_psi is not None),
        }
        if self.ros_bridge is not None:
            ros_feedback.update(self.ros_bridge.feedback_snapshot())
            if ros_feedback.get("jointPosition") is not None:
                q_fb = np.asarray(ros_feedback["jointPosition"], dtype=np.float32)
                joint_ee, joint_quat, joint_fk_ok = (
                    self.controller._compute_racket_pose_quat_base(q_fb))
                feedback_fk_ok = bool(joint_fk_ok)
                if feedback_fk_ok:
                    feedback_racket_pos = np.asarray(
                        joint_ee, dtype=np.float32).reshape(3)
                    feedback_racket_quat = np.asarray(
                        joint_quat, dtype=np.float32).reshape(4)
                ros_feedback["jointEe"] = (
                    [round(float(x), 6) for x in joint_ee]
                    if joint_fk_ok else None
                )
                ros_feedback["jointEeFkOk"] = bool(joint_fk_ok)
                try:
                    err = np.asarray(cmd["position"], dtype=np.float32) - q_fb
                    ros_feedback["jointCommandErrorAbsMax"] = round(
                        float(np.max(np.abs(err))), 6)
                except Exception:
                    ros_feedback["jointCommandErrorAbsMax"] = None
        if publish_commands and feedback_fk_ok and feedback_racket_pos is not None:
            racket_pos = feedback_racket_pos
            if feedback_racket_quat is not None:
                racket_quat = feedback_racket_quat
            fk_ok = True
            racket_source = "joint_state_fk"
        if cart_quat is not None:
            racket_quat = cart_quat
        done_preview = self._done_preview_payload(
            raw_ball_pos, measured_ball_vel, bool(raw_valid), racket_pos, fk_ok,
            ee_quat_xyzw=racket_quat, ee_psi_rad=cart_psi,
            workspace_safety_mode=(
                runtime_options.workspace_safety_mode
                if runtime_options is not None else self.cfg.workspace_safety_mode))
        camera_feedback = (
            self.camera_feedback()
            if camera_feedback is None else camera_feedback)
        return {
            "time": round(float(self.now_s), 6),
            "mode": mode,
            "reason": reason,
            "backendControlLoop": {
                "enabled": bool(self.cfg.backend_control_loop),
                "source": "ros_timer" if self.control_loop_uses_ros_timer else "thread",
                "targetHz": round(float(self.cfg.control_rate_hz), 3),
                "measuredHz": round(float(self._control_loop_hz()), 3),
                "snapshotHz": round(float(self.cfg.backend_snapshot_rate_hz), 3),
                "workMs": round(float(self._control_loop_work_ms()), 3),
                "overruns": int(self.control_loop_overrun_count),
                "error": self.control_loop_last_error,
                "tick": int(self.control_loop_tick),
            },
            "conditions": {k: bool(v) for k, v in conditions.items()},
            "ball": [round(float(x), 6) for x in observed_ball_pos],
            "rawBall": [round(float(x), 6) for x in raw_ball_pos],
            "measuredBall": [round(float(x), 6) for x in measured_ball_pos],
            "observedBall": [round(float(x), 6) for x in observed_ball_pos],
            "measuredVelocity": [round(float(x), 6) for x in measured_ball_vel],
            "velocity": [round(float(x), 6) for x in observed_ball_vel],
            "rawValid": raw_valid,
            "valid": bool(valid),
            "age": round(float(age_s), 6),
            "noiseStdM": round(float(noise_std_m), 6),
            "ballSource": str(ball_source),
            "runtime": runtime_payload,
            "estimator": estimator_info,
            "compensation": self._compensation_payload(runtime_options),
            "releaseGate": release_gate_info,
            "wouldDone": done_preview,
            "doneBall": {
                "source": "raw_measurement",
                "position": _rounded_vector(self.last_done_ball_pos),
                "valid": bool(self.last_done_ball_valid),
                "rawConditions": dict(self.ball_done_raw_conditions),
                "streaks": dict(self.ball_done_streaks),
                "requiredTicks": int(
                    self.cfg.done_ball_required_consecutive_ticks),
            },
            "commandDisableHold": dict(self.last_command_disable_hold),
            "commandArm": dict(self.last_command_arm),
            "hardWorkspaceStop": self._hard_workspace_stop_payload(),
            "resetRecovery": self._reset_recovery_payload(),
            "workspaceSoft": {
                "active": bool(
                    self.workspace_soft_conditions
                    or self.workspace_soft_filter.get("active", False)),
                "conditions": dict(self.workspace_soft_conditions),
                "predictionActive": bool(
                    self.workspace_soft_prediction_active),
                "filter": dict(self.workspace_soft_filter),
            },
            "racket": [round(float(x), 6) for x in racket_pos],
            "racketSource": racket_source,
            "commandRacket": [round(float(x), 6) for x in cmd_racket_pos],
            "commandFkOk": cmd_fk_ok,
            "fkOk": fk_ok,
            "anchor": [round(float(x), 6) for x in self.anchor()],
            "jointPosition": [round(float(x), 6) for x in cmd["position"]],
            "jointVelocity": [round(float(x), 6) for x in cmd["velocity"]],
            "jointAcceleration": [round(float(x), 6) for x in cmd["acceleration"]],
            "action": [round(float(x), 6) for x in obs_action["action"]],
            "actionAbsMax": round(float(np.max(np.abs(obs_action["action"]))), 6),
            "policyTrace": {
                "obs": [round(float(x), 6) for x in obs_action["obs"]],
                "action": [round(float(x), 6) for x in obs_action["action"]],
                "commandPosition": [round(float(x), 6) for x in cmd["position"]],
                "commandVelocity": [round(float(x), 6) for x in cmd["velocity"]],
                "commandAcceleration": [
                    round(float(x), 6) for x in cmd["acceleration"]
                ],
            },
            "doneStats": self.done_monitor.diagnostics(),
            "ros": ros_feedback,
            "sim": dict(self.latest_sim_snapshot or self._sim_unavailable_payload()),
            "camera": camera_feedback,
            "recording": self.recording_status(),
        }

    def browser_config(self) -> dict[str, Any]:
        anchor = self.anchor()
        ball_box = box_payload(
            (float(anchor[0] - self.cfg.done_x_range_m), float(anchor[0] + self.cfg.done_x_range_m)),
            (float(anchor[1] - self.cfg.done_y_range_m), float(anchor[1] + self.cfg.done_y_range_m)),
            (self.cfg.done_z_min_m, self.cfg.done_z_max_m),
        )
        ee_box = box_payload(
            self.cfg.ee_x_limits_m,
            self.cfg.ee_y_limits_m,
            self.cfg.ee_z_limits_m,
        )
        xs = [ball_box["x0"], ball_box["x1"]]
        ys = [ball_box["y0"], ball_box["y1"]]
        zs = [ball_box["z0"], ball_box["z1"]]
        if ee_box:
            xs += [ee_box["x0"], ee_box["x1"]]
            ys += [ee_box["y0"], ee_box["y1"]]
            zs += [ee_box["z0"], ee_box["z1"]]
        margin = 0.18
        return {
            "controlRateHz": self.cfg.control_rate_hz,
            "backendControlLoop": bool(self.cfg.backend_control_loop),
            "backendSnapshotRateHz": self.cfg.backend_snapshot_rate_hz,
            "browserStepRateHz": self.cfg.browser_step_rate_hz,
            "displayRateHz": self.cfg.display_rate_hz,
            "recording": {
                "outputRoot": str(self.cfg.recording_output_root),
                "controlRateHz": round(
                    float(self.cfg.recording_control_rate_hz), 3),
                "imageRateHz": round(
                    float(self.cfg.recording_image_rate_hz), 3),
                "analysisVideoFps": round(
                    float(self.cfg.recording_analysis_video_fps), 3),
            },
            "lostTimeoutS": self.cfg.ball_lost_timeout_s,
            "policy": {
                "useValidBallObsAge": bool(self.cfg.rl_use_valid_ball_obs_age),
            },
            "compensation": {
                "enabled": bool(self.cfg.enable_compensation),
                "available": bool(self.compensator is not None),
                "mode": str(self.cfg.compensation_mode),
                "delayMs": self.cfg.comp_delay_ms,
                "tauS": self.cfg.comp_actuator_tau_s,
                "gain": self.cfg.comp_actuator_gain,
                "mpcBeta": self.cfg.comp_mpc_beta,
                "mpcDelayScale": self.cfg.comp_mpc_delay_scale,
                "mpcTauScale": self.cfg.comp_mpc_tau_scale,
                "mpcHorizonSteps": self.cfg.comp_mpc_horizon_steps,
                "mpcTrackingWeight": self.cfg.comp_mpc_tracking_weight,
                "mpcNominalWeight": self.cfg.comp_mpc_nominal_weight,
                "mpcDeltaWeight": self.cfg.comp_mpc_delta_weight,
                "mpcMaxDeltaRad": self.cfg.comp_mpc_max_delta_rad,
                "error": self.compensation_error,
            },
            "releaseGate": {
                "enabled": bool(self.cfg.enable_ball_release_gate),
                "minDownwardVelocityMS": self.cfg.ball_release_min_downward_velocity_m_s,
                "minZDropM": self.cfg.ball_release_min_z_drop_m,
                "windowS": self.cfg.ball_release_window_s,
                "requiredTicks": self.cfg.ball_release_required_consecutive_ticks,
                "maxObsAgeS": self.cfg.ball_release_max_obs_age_s,
                "useEstimatorState": bool(
                    self.cfg.ball_release_use_estimator_state),
            },
            "defaultNoiseStdM": self.cfg.ball_noise_std_m,
            "defaultBallObservationRateHz": self.cfg.ball_observation_rate_hz,
            "flowDefaults": {
                "captureGate": bool(self.cfg.enable_ball_capture_gate),
                "releaseGate": bool(self.cfg.enable_ball_release_gate),
                "lostFailsafe": bool(self.cfg.enable_ball_lost_failsafe),
                "doneMonitoring": bool(self.cfg.enable_done_monitoring),
                "safetyLimiter": bool(self.cfg.enable_safety_limiter),
                "workspaceSafetyMode": str(self.cfg.workspace_safety_mode),
            },
            "workspaceSoft": {
                "model": "virtual_impedance",
                "marginM": self.cfg.workspace_soft_margin_m,
                "pullSpeedMps": self.cfg.workspace_soft_pull_speed_m_s,
                "damping": self.cfg.workspace_soft_damping,
                "maxCorrectionSpeedMps": (
                    self.cfg.workspace_soft_max_correction_speed_m_s),
                "dlsDamping": self.cfg.workspace_soft_dls_damping,
                "nullspaceDamping": (
                    self.cfg.workspace_soft_nullspace_damping),
                "postureGainS": self.cfg.workspace_soft_posture_gain_s,
                "boundaryJointSpeedScale": (
                    self.cfg.workspace_soft_boundary_joint_speed_scale),
            },
            "workspaceHold": {
                "model": "boundary_hold",
                "triggerMarginM": self.cfg.workspace_hold_trigger_margin_m,
                "releasePolicy": "candidate_fk_moves_inward",
            },
            "simMirror": {
                "enabled": bool(self.cfg.enable_sim_mirror),
                "defaultEnabled": bool(self.cfg.sim_mirror_default_enabled),
                "subprocess": bool(self.cfg.sim_subprocess),
                "available": bool(self.sim_mirror is not None),
                "backend": str(self.cfg.sim_backend),
                "xmlPath": str(self.cfg.sim_xml_path),
                "videoFrames": bool(self.cfg.sim_enable_video_frames),
                "frameRateHz": round(float(self.cfg.sim_frame_rate_hz), 3),
                "renderWidth": int(self.cfg.sim_render_width),
                "renderHeight": int(self.cfg.sim_render_height),
                "useMujocoRenderer": bool(self.cfg.sim_use_mujoco_renderer),
                "error": self.sim_mirror_error,
            },
            "ballEstimator": {
                "enabled": bool(self.cfg.enable_ball_state_estimator),
                "prediction": bool(self.cfg.ball_estimator_enable_prediction),
                "aeroDrag": bool(self.cfg.ball_estimator_enable_aero_drag),
                "maxExtrapolationS": self.cfg.ball_estimator_max_extrapolation_s,
                "controlMaxExtrapolationS": (
                    self.cfg.ball_estimator_control_max_extrapolation_s),
                "controlOutputMode": (
                    self.cfg.ball_estimator_control_output_mode),
                "frameHoldOutputRateHz": (
                    self.cfg.ball_estimator_frame_hold_output_rate_hz),
                "releaseGatedDynamics": bool(
                    self.cfg.ball_estimator_release_gated_dynamics),
                "trustLossS": self.cfg.ball_estimator_trust_loss_s,
                "degradeLossS": self.cfg.ball_estimator_degrade_loss_s,
                "invalidAfterS": self.cfg.ball_estimator_invalid_after_s,
                "predictionHorizonS": self.cfg.ball_estimator_prediction_horizon_s,
                "qPosMm2S": self.cfg.ball_estimator_q_pos_mm2_s,
                "qVelMult": self.cfg.ball_estimator_q_vel_mult,
                "qDepthPosMult": self.cfg.ball_estimator_q_depth_pos_mult,
                "qDepthVelMult": self.cfg.ball_estimator_q_depth_vel_mult,
                "rLateralMm": self.cfg.ball_estimator_r_lateral_mm,
                "rDepthMm": self.cfg.ball_estimator_r_depth_mm,
            },
            "parking": {
                "velocityScale": self.cfg.safety_parking_velocity_scale,
                "accelerationScale": (
                    self.cfg.safety_parking_acceleration_scale),
            },
            "donePrediction": {
                "enabled": bool(self.cfg.done_enable_predictive_stop),
                "horizonS": self.cfg.done_prediction_horizon_s,
                "dtS": self.cfg.done_prediction_dt_s,
                "marginM": self.cfg.done_prediction_margin_m,
                "orientationMarginRad": (
                    self.cfg.done_prediction_orientation_margin_rad),
            },
            "rosBridge": self.cfg.ros_bridge,
            "rosPublishCommands": self.cfg.ros_publish_commands,
            "rosDefaultUseState": self.cfg.ros_bridge,
            "rosDefaultPublishCommands": False,
            "rosPublishAllowed": self.cfg.ros_publish_commands,
            "commandTopic": self.cfg.command_topic,
            "jointStateTopic": self.cfg.joint_state_topic,
            "cartStateTopic": self.cfg.cart_state_topic,
            "cameraBall": {
                "enabled": bool(self.cfg.enable_camera_ball),
                "available": bool(_CAMERA_BALL_AVAILABLE),
                "sourceMode": self.cfg.camera_source_mode,
                "useHybridDetector": bool(self.cfg.use_hybrid_detector),
                "useDepthSensor": bool(self.cfg.use_depth_sensor),
                "imageWidth": int(self.cfg.image_width),
                "imageHeight": int(self.cfg.image_height),
                "realsenseFps": int(self.cfg.realsense_fps),
                "visualizationRateHz": round(
                    float(self.cfg.camera_visualization_rate_hz), 3),
                "webMaxWidth": int(self.cfg.camera_web_max_width),
                "webJpegQuality": int(self.cfg.camera_web_jpeg_quality),
                "imageTopic": self.cfg.camera_image_topic,
                "rawImageTopic": self.cfg.camera_raw_image_topic,
                "imageTransport": self.cfg.camera_image_transport,
                "cameraInfoTopic": self.cfg.camera_info_topic,
                "calibrationJsonPath": self.cfg.calibration_json_path,
                "yoloModelPath": str(self.cfg.yolo_model_path),
                "defaults": {
                    "detect": bool(self.cfg.enable_camera_detection),
                    "visualize": bool(self.cfg.enable_camera_visualization),
                    "sensorDepth": bool(self.cfg.use_depth_sensor),
                    "undistort": bool(self.cfg.use_undistort),
                    "kalman": bool(self.cfg.use_kalman),
                    "ellipseFit": bool(self.cfg.enable_ellipse_fit),
                    "temporalTracker": bool(self.cfg.enable_temporal_tracker),
                    "colorFilter": bool(self.cfg.enable_camera_color_filter),
                },
                "importError": (
                    None if _CAMERA_BALL_AVAILABLE
                    else str(_CAMERA_BALL_IMPORT_ERROR)),
            },
            "anchor": [round(float(x), 6) for x in anchor],
            "initialBall": [round(float(x), 6) for x in self.initial_ball_pos()],
            "ballBox": ball_box,
            "eeBox": ee_box,
            "eeOrientation": {
                "psiLimitsRad": (
                    None if self.cfg.ee_psi_limits_rad is None
                    else [float(v) for v in self.cfg.ee_psi_limits_rad]
                ),
                "bLimitsRad": (
                    None if self.cfg.ee_b_limits_rad is None
                    else [float(v) for v in self.cfg.ee_b_limits_rad]
                ),
                "bMinRad": self.cfg.ee_b_min_rad,
                "bMaxRad": self.cfg.ee_b_max_rad,
                "cLimitsRad": (
                    None if self.cfg.ee_c_limits_rad is None
                    else [float(v) for v in self.cfg.ee_c_limits_rad]
                ),
                "cMinRad": self.cfg.ee_c_min_rad,
                "cMaxRad": self.cfg.ee_c_max_rad,
                "referenceQuatXyzw": (
                    None if self.cfg.ee_orientation_reference_xyzw is None
                    else [
                        float(v) for v in self.cfg.ee_orientation_reference_xyzw
                    ]
                ),
                "maxOrientationErrorDeg": (
                    float(self.cfg.ee_max_orientation_error_deg)
                ),
            },
            "limits": {
                "x0": min(xs) - margin, "x1": max(xs) + margin,
                "y0": min(ys) - margin, "y1": max(ys) + margin,
                "z0": max(0.0, min(zs) - margin), "z1": max(zs) + margin,
            },
        }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy Safety Probe</title>
<style>
  :root {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #1f2933;
    background: #f5f7fa;
  }
  * { box-sizing: border-box; }
  html, body { width: 100%; height: 100%; }
  body { margin: 0; min-height: 100vh; overflow: hidden; background: #f5f7fa; }
  .app {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
    width: 100vw;
    height: 100vh;
    min-height: 0;
    overflow: hidden;
  }
  header {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 16px;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #d7dee8;
    background: #ffffff;
  }
  h1 { margin: 0; font-size: 18px; font-weight: 650; }
  .titlebar {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 0;
  }
  .recording-control {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    font-size: 12px;
    color: #526173;
    font-variant-numeric: tabular-nums;
  }
  #recordingButton.active {
    border-color: #dc2626;
    background: #dc2626;
    color: #ffffff;
  }
  #recordingButton.busy {
    opacity: 0.7;
    cursor: progress;
  }
  #recordingStatus {
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .mini-check {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
  }
  .nav-button.active {
    border-color: #0f766e;
    background: #ecfdf5;
    color: #0f766e;
  }
  .status {
    display: grid;
    grid-auto-flow: column;
    gap: 10px;
    align-items: center;
    font-size: 13px;
    color: #44546a;
    font-variant-numeric: tabular-nums;
  }
  .badge {
    min-width: 132px;
    border: 1px solid #c6d0dd;
    border-radius: 6px;
    padding: 6px 9px;
    background: #ffffff;
    color: #243244;
  }
  .badge.warn { border-color: #f59e0b; color: #92400e; background: #fff8eb; }
  .badge.stop { border-color: #ef4444; color: #991b1b; background: #fff1f2; }
  main {
    display: grid;
    grid-template-columns: minmax(250px, 300px) minmax(0, 1fr) minmax(260px, 300px);
    gap: 0;
    min-height: 0;
    overflow: hidden;
  }
  .views {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.85fr);
    grid-template-rows: 1fr 1fr 0.78fr;
    min-height: 0;
    height: 100%;
  }
  canvas {
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
    background: linear-gradient(#ffffff, #edf2f7);
    border-right: 1px solid #d7dee8;
    border-bottom: 1px solid #d7dee8;
    touch-action: none;
    cursor: crosshair;
  }
  .view3d {
    grid-row: 1 / span 3;
    cursor: grab;
  }
  .view3d.dragging { cursor: grabbing; }
  .plane { min-height: 0; }
  .image-stack {
    display: grid;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
    min-height: 0;
    overflow: hidden;
    border-right: 1px solid #d7dee8;
    border-bottom: 1px solid #d7dee8;
    background: #0f172a;
  }
  .vision-panel {
    position: relative;
    min-height: 0;
    overflow: hidden;
    background: #0f172a;
  }
  .vision-panel + .vision-panel { border-top: 1px solid #334155; }
  .vision-panel img {
    display: block;
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: #0f172a;
  }
  .vision-empty {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    padding: 12px;
    color: #cbd5e1;
    font-size: 13px;
    text-align: center;
  }
  main > aside {
    padding: 14px;
    background: #ffffff;
    overflow: auto;
    min-height: 0;
  }
  .tuning-panel {
    border-right: 1px solid #d7dee8;
  }
  .control-panel {
    border-left: 1px solid #d7dee8;
  }
  .row { display: grid; grid-template-columns: 96px 1fr; align-items: center; gap: 8px; margin-bottom: 10px; }
  label { font-size: 13px; color: #526173; }
  input[type="number"], select {
    width: 100%;
    border: 1px solid #bdc8d6;
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
  button {
    border: 1px solid #b7c2d0;
    border-radius: 6px;
    background: #ffffff;
    padding: 7px 10px;
    font-size: 14px;
    cursor: pointer;
  }
  button:hover { background: #f0f4f8; }
  .toggle { display: grid; grid-template-columns: 1fr auto; align-items: center; margin: 12px 0; }
  .section-title {
    margin: 16px 0 8px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
    font-size: 12px;
    font-weight: 650;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 0;
  }
  .section-note {
    margin: -2px 0 9px;
    color: #64748b;
    font-size: 11px;
    line-height: 1.35;
  }
  .checks {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 7px 10px;
    margin-bottom: 8px;
  }
  .checkrow {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 7px;
    align-items: center;
    min-height: 22px;
    font-size: 13px;
    color: #526173;
  }
  input:disabled + span,
  .checkrow.disabled span,
  .toggle.disabled label,
  .row.disabled label {
    color: #94a3b8;
  }
  .readout {
    margin-top: 14px;
    font-size: 13px;
    line-height: 1.7;
    color: #334155;
    font-variant-numeric: tabular-nums;
  }
  .plot {
    width: 100%;
    height: 150px;
    min-height: 150px;
    margin-top: 12px;
    border: 1px solid #d7dee8;
    border-radius: 6px;
    background: #ffffff;
    cursor: default;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 10px 16px;
    border-top: 1px solid #d7dee8;
    background: #ffffff;
    font-size: 13px;
    color: #44546a;
    overflow-x: auto;
  }
  .hidden { display: none !important; }
  .analysis-view {
    display: grid;
    grid-template-columns: 270px minmax(0, 1fr);
    gap: 0;
    height: 100%;
    min-height: 0;
    overflow: hidden;
  }
  .analysis-menu {
    padding: 14px;
    border-right: 1px solid #d7dee8;
    background: #ffffff;
    overflow: auto;
    min-height: 0;
  }
  .analysis-menu button {
    width: 100%;
    margin-bottom: 8px;
    text-align: left;
  }
  .analysis-menu button.active {
    border-color: #0f766e;
    background: #ecfdf5;
    color: #0f766e;
  }
  .analysis-sublist {
    max-height: 260px;
    overflow: auto;
    border: 1px solid #d7dee8;
    border-radius: 6px;
    padding: 6px;
    margin: 8px 0;
    background: #f8fafc;
  }
  .analysis-sublist button {
    font-size: 12px;
    padding: 5px 7px;
    margin-bottom: 5px;
  }
  .analysis-display {
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    min-height: 0;
    overflow: hidden;
    background: #f8fafc;
  }
  .analysis-toolbar {
    display: flex;
    gap: 10px;
    align-items: center;
    padding: 10px 14px;
    border-bottom: 1px solid #d7dee8;
    background: #ffffff;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
  }
  .analysis-content {
    min-height: 0;
    overflow: hidden;
    padding: 12px;
  }
  .analysis-pane { display: none; width: 100%; height: 100%; min-height: 0; }
  .analysis-pane.active { display: block; }
  .analysis-canvas-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr) minmax(260px, 0.8fr);
    grid-template-rows: minmax(260px, 1fr) minmax(220px, 0.85fr);
    min-height: 620px;
  }
  .analysis-canvas-grid canvas,
  .analysis-curve-canvas {
    width: 100%;
    height: 100%;
    min-height: 220px;
    background: linear-gradient(#ffffff, #edf2f7);
    border-right: 1px solid #d7dee8;
    border-bottom: 1px solid #d7dee8;
  }
  #analysisReplay3d {
    grid-row: 1 / span 2;
    cursor: grab;
    touch-action: none;
  }
  #analysisReplay3d.dragging { cursor: grabbing; }
  .analysis-camera-frame {
    width: 100%;
    height: 100%;
    max-height: calc(100vh - 188px);
    object-fit: contain;
    background: #0f172a;
    border: 1px solid #d7dee8;
  }
  .analysis-curve-canvas {
    height: calc(100vh - 188px);
    min-height: 520px;
  }
  .analysis-report {
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.5;
    background: #ffffff;
    border: 1px solid #d7dee8;
    border-radius: 6px;
    padding: 10px;
  }
  .analysis-empty {
    display: grid;
    min-height: 260px;
    place-items: center;
    color: #64748b;
    background: #ffffff;
    border: 1px solid #d7dee8;
  }
  .item::before {
    content: "";
    display: inline-block;
    width: 17px;
    height: 3px;
    margin-right: 6px;
    vertical-align: middle;
    background: var(--c);
  }
  @media (max-width: 900px) {
    body { overflow: auto; }
    .app { height: auto; min-height: 100vh; overflow: visible; }
    main { grid-template-columns: 1fr; overflow: visible; }
    main > aside { border-left: 0; border-right: 0; border-top: 1px solid #d7dee8; }
    .views {
      grid-template-columns: 1fr;
      grid-template-rows: minmax(360px, 46vh) minmax(260px, 32vh) minmax(260px, 32vh) minmax(220px, 28vh);
      height: auto;
    }
    .analysis-view {
      grid-template-columns: 1fr;
      height: auto;
      min-height: calc(100vh - 80px);
    }
    .analysis-menu {
      border-right: 0;
      border-bottom: 1px solid #d7dee8;
    }
    .analysis-canvas-grid {
      grid-template-columns: 1fr;
      grid-template-rows: minmax(320px, 42vh) minmax(260px, 34vh) minmax(260px, 34vh);
      min-height: 900px;
    }
    #analysisReplay3d { grid-row: auto; }
    .view3d { grid-row: auto; }
    canvas { min-height: 260px; }
    header { grid-template-columns: 1fr; }
    .status { grid-auto-flow: row; }
  }
  @media (max-height: 760px) and (min-width: 901px) {
    header { padding: 8px 12px; }
    main > aside { padding: 10px; }
    .toggle { margin: 8px 0; }
    .row { margin-bottom: 7px; }
    .section-title { margin: 10px 0 6px; padding-top: 8px; }
    .checks { gap: 5px 8px; }
    .plot { height: 110px; min-height: 110px; }
    .legend { padding: 7px 12px; }
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="titlebar">
      <h1>Policy Safety Probe</h1>
      <div class="recording-control">
        <button id="liveViewButton" class="nav-button active" type="button">Live</button>
        <button id="analysisViewButton" class="nav-button" type="button">Analysis</button>
        <button id="recordingButton" type="button">Record</button>
        <label class="mini-check"><input id="analyzeAfterRecord" type="checkbox" checked><span>Analyze</span></label>
        <span id="recordingStatus">idle</span>
      </div>
    </div>
    <div class="status">
      <div id="mode" class="badge">mode</div>
      <div id="ros" class="badge">ros=off</div>
      <div id="time">t=0.000s</div>
      <div id="age">age=0.000s</div>
    </div>
  </header>
  <main id="liveMain">
    <aside class="tuning-panel">
      <div class="section-title">Ball Params</div>
      <div class="row"><label for="x">ball x</label><input id="x" type="number" step="0.005"></div>
      <div class="row"><label for="y">ball y</label><input id="y" type="number" step="0.005"></div>
      <div class="row"><label for="z">ball z</label><input id="z" type="number" step="0.005"></div>
      <div class="row"><label for="noise">noise mm</label><input id="noise" type="number" min="0" step="0.5"></div>
      <div class="row"><label for="obsHz">ball obs hz</label><input id="obsHz" type="number" min="1" max="240" step="1"></div>
      <div class="section-title">Predictive Safety</div>
      <div class="row"><label for="donePredictionHorizonS">horizon s</label><input id="donePredictionHorizonS" type="number" min="0" max="0.5" step="0.005"></div>
      <div class="row"><label for="donePredictionDtS">step s</label><input id="donePredictionDtS" type="number" min="0.001" max="0.1" step="0.001"></div>
      <div class="row"><label for="donePredictionMarginM">margin m</label><input id="donePredictionMarginM" type="number" min="0" max="0.15" step="0.001"></div>
      <div class="row"><label for="donePredictionOrientationMarginRad">ori margin</label><input id="donePredictionOrientationMarginRad" type="number" min="0" max="0.5" step="0.005"></div>
      <div class="section-title">Compensation Params</div>
      <div class="row"><label for="compDelayMs">delay ms</label><input id="compDelayMs" type="number" min="0" max="250" step="1"></div>
      <div class="row"><label for="compActuatorTauS">tau s</label><input id="compActuatorTauS" type="number" min="0" max="0.5" step="0.001"></div>
      <div class="row"><label for="compActuatorGain">gain</label><input id="compActuatorGain" type="number" min="0.1" max="3" step="0.01"></div>
      <div class="row"><label for="compMpcBeta">mpc beta</label><input id="compMpcBeta" type="number" min="0" max="5" step="0.05"></div>
      <div class="row"><label for="compMpcDelayScale">delay scale</label><input id="compMpcDelayScale" type="number" min="0" max="3" step="0.01"></div>
      <div class="row"><label for="compMpcTauScale">tau scale</label><input id="compMpcTauScale" type="number" min="0" max="3" step="0.01"></div>
      <div class="row"><label for="compMpcHorizonSteps">horizon</label><input id="compMpcHorizonSteps" type="number" min="1" max="32" step="1"></div>
      <div class="row"><label for="compMpcTrackingWeight">track w</label><input id="compMpcTrackingWeight" type="number" min="0" max="10" step="0.05"></div>
      <div class="row"><label for="compMpcNominalWeight">nominal w</label><input id="compMpcNominalWeight" type="number" min="0" max="10" step="0.05"></div>
      <div class="row"><label for="compMpcDeltaWeight">delta w</label><input id="compMpcDeltaWeight" type="number" min="0" max="10" step="0.01"></div>
      <div class="row"><label for="compMpcMaxDeltaRad">max delta</label><input id="compMpcMaxDeltaRad" type="number" min="0" max="2" step="0.01"></div>
      <div class="section-title">Ball-state Filter</div>
      <div class="section-note">Q 越大越跟手，R 越大越平滑；修改后会重建 3D 估计器。</div>
      <div class="row"><label for="estimatorQPosMm2S">q pos</label><input id="estimatorQPosMm2S" title="Position process-noise spectral density (mm²/s)" type="number" min="0" max="1000000" step="10"></div>
      <div class="row"><label for="estimatorQVelMult">q vel mult</label><input id="estimatorQVelMult" title="Velocity process-noise multiplier" type="number" min="0" max="100000" step="1"></div>
      <div class="row"><label for="estimatorQDepthPosMult">q depth pos</label><input id="estimatorQDepthPosMult" title="Camera-depth multiplier for position process noise" type="number" min="0.000001" max="100" step="0.01"></div>
      <div class="row"><label for="estimatorQDepthVelMult">q depth vel</label><input id="estimatorQDepthVelMult" title="Camera-depth multiplier for velocity process noise" type="number" min="0.000001" max="100" step="0.01"></div>
      <div class="row"><label for="estimatorRLateralMm">r lateral mm</label><input id="estimatorRLateralMm" title="Lateral measurement-noise sigma; larger is smoother" type="number" min="1" max="500" step="1"></div>
      <div class="row"><label for="estimatorRDepthMm">r depth mm</label><input id="estimatorRDepthMm" title="Optical-depth measurement-noise sigma; larger is smoother" type="number" min="1" max="1000" step="1"></div>
      <div class="section-title">Ball Prediction</div>
      <div class="row"><label for="estimatorMaxExtrapolationS">max extrap s</label><input id="estimatorMaxExtrapolationS" type="number" min="0" max="2" step="0.01"></div>
      <div class="row"><label for="estimatorControlMaxExtrapolationS">control cap s</label><input id="estimatorControlMaxExtrapolationS" type="number" min="-1" max="2" step="0.01"></div>
      <div class="row"><label for="estimatorPredictionHorizonS">extra lead s</label><input id="estimatorPredictionHorizonS" title="Extra lead past the current time; normally keep at zero" type="number" min="0" max="1" step="0.005"></div>
      <div class="section-title">Frame Bridge / Loss</div>
      <div class="section-note">continuous 每个控制 tick 外推；hold 只按下列频率刷新并零阶保持。</div>
      <div class="row"><label for="estimatorFrameHoldHz">hold hz</label><input id="estimatorFrameHoldHz" type="number" min="1" max="240" step="1"></div>
      <div class="row"><label for="estimatorTrustLossS">trust loss s</label><input id="estimatorTrustLossS" type="number" min="0" max="2" step="0.01"></div>
      <div class="row"><label for="estimatorDegradeLossS">degrade s</label><input id="estimatorDegradeLossS" type="number" min="0" max="3" step="0.01"></div>
      <div class="row"><label for="estimatorInvalidAfterS">invalid s</label><input id="estimatorInvalidAfterS" type="number" min="0" max="5" step="0.01"></div>
      <div class="section-title">Release Params</div>
      <div class="row"><label for="releaseMinVz">min vz m/s</label><input id="releaseMinVz" type="number" min="0" max="5" step="0.01"></div>
      <div class="row"><label for="releaseMinDrop">min drop m</label><input id="releaseMinDrop" type="number" min="0" max="1" step="0.005"></div>
      <div class="row"><label for="releaseWindowS">window s</label><input id="releaseWindowS" type="number" min="0.001" max="2" step="0.01"></div>
      <div class="row"><label for="releaseTicks">ticks</label><input id="releaseTicks" type="number" min="1" max="200" step="1"></div>
      <div class="row"><label for="releaseMaxAgeS">max age s</label><input id="releaseMaxAgeS" type="number" min="0" max="2" step="0.01"></div>
    </aside>
    <div class="views">
      <canvas id="view3d" class="view3d"></canvas>
      <canvas id="top" class="plane"></canvas>
      <canvas id="side" class="plane"></canvas>
      <div class="image-stack">
        <div class="vision-panel">
          <img id="visionFrame" alt="">
          <div id="visionEmpty" class="vision-empty">camera overlay waiting</div>
        </div>
        <div class="vision-panel">
          <img id="simFrame" alt="">
          <div id="simEmpty" class="vision-empty">sim mirror waiting</div>
        </div>
      </div>
    </div>
    <aside class="control-panel">
      <div class="toggle">
        <label for="visible">Ball visible</label>
        <input id="visible" type="checkbox" checked>
      </div>
      <div class="row">
        <label for="ballSource">ball source</label>
        <select id="ballSource">
          <option value="drag">drag ball</option>
          <option value="camera">camera ball</option>
        </select>
      </div>
      <div class="section-title">Runtime</div>
      <div class="toggle">
        <label for="useRosState">Use ROS state</label>
        <input id="useRosState" type="checkbox">
      </div>
      <div class="toggle">
        <label for="publishCommands">Send command</label>
        <input id="publishCommands" type="checkbox">
      </div>
      <div class="row">
        <label for="workspaceSafetyMode">workspace</label>
        <select id="workspaceSafetyMode">
          <option value="off">off</option>
          <option value="position">position (hard)</option>
          <option value="predictive">predictive (hard)</option>
          <option value="position_soft">position soft</option>
          <option value="predictive_soft">predictive soft</option>
          <option value="impedance_soft">impedance soft</option>
        </select>
      </div>
      <div class="row"><label for="compensationMode">comp mode</label><select id="compensationMode"><option value="none">none</option><option value="lead">lead</option><option value="inverse_smith">inverse smith</option><option value="inverse_mpc">inverse mpc</option></select></div>
      <div class="section-title">Node Flow</div>
      <div class="checks">
        <label class="checkrow"><input id="rlUseValidBallObsAge" type="checkbox"><span>policy age</span></label>
        <label class="checkrow"><input id="enableCompensation" type="checkbox"><span>compensation</span></label>
        <label class="checkrow"><input id="enableSimMirror" type="checkbox"><span>sim mirror</span></label>
        <label class="checkrow" title="3D Kalman filtering; required by prediction and control-rate bridging"><input id="useBallEstimator" type="checkbox"><span>3D filter</span></label>
        <label class="checkrow" title="Ballistic extrapolation to the control time"><input id="estimatorPrediction" type="checkbox"><span>predict / bridge</span></label>
        <label class="checkrow" title="Use rate-limited snapshot plus zero-order hold instead of continuous 200 Hz prediction"><input id="estimatorFrameHold" type="checkbox"><span>hold output</span></label>
        <label class="checkrow"><input id="estimatorAeroDrag" type="checkbox"><span>aero drag</span></label>
        <label class="checkrow"><input id="estimatorReleaseGatedDynamics" type="checkbox"><span>gated dynamics</span></label>
        <label class="checkrow"><input id="releaseUseEstimatorState" type="checkbox"><span>release est</span></label>
        <label class="checkrow"><input id="enableCaptureGate" type="checkbox"><span>capture gate</span></label>
        <label class="checkrow"><input id="enableReleaseGate" type="checkbox"><span>release gate</span></label>
        <label class="checkrow"><input id="enableLostFailsafe" type="checkbox"><span>lost failsafe</span></label>
        <label class="checkrow"><input id="enableDoneMonitoring" type="checkbox"><span>done monitor</span></label>
      </div>
      <div class="section-title">Camera Flow</div>
      <div class="checks">
        <label class="checkrow"><input id="cameraDetect" type="checkbox"><span>detection</span></label>
        <label class="checkrow"><input id="cameraVisualize" type="checkbox"><span>overlay</span></label>
        <label class="checkrow"><input id="cameraSensorDepth" type="checkbox"><span>sensor depth</span></label>
        <label class="checkrow"><input id="cameraUndistort" type="checkbox"><span>undistort</span></label>
        <label class="checkrow"><input id="cameraKalman" type="checkbox"><span>kalman</span></label>
        <label class="checkrow"><input id="cameraEllipseFit" type="checkbox"><span>ellipse fit</span></label>
        <label class="checkrow"><input id="cameraTemporalTracker" type="checkbox"><span>tracker</span></label>
        <label class="checkrow"><input id="cameraColorFilter" type="checkbox"><span>color filter</span></label>
      </div>
      <div class="section-title">Curves</div>
      <div class="checks">
        <label class="checkrow"><input id="showBall" type="checkbox" checked><span>ball</span></label>
        <label class="checkrow"><input id="showObserved" type="checkbox" checked><span>noisy obs</span></label>
        <label class="checkrow"><input id="showRacket" type="checkbox" checked><span>robot EE</span></label>
        <label class="checkrow"><input id="showSim" type="checkbox" checked><span>sim EE</span></label>
        <label class="checkrow"><input id="showJoint" type="checkbox" checked><span>joint FK</span></label>
        <label class="checkrow"><input id="showCart" type="checkbox"><span>cart</span></label>
        <label class="checkrow"><input id="showBounds" type="checkbox" checked><span>bounds</span></label>
        <label class="checkrow"><input id="showActionPlot" type="checkbox" checked><span>action plot</span></label>
        <label class="checkrow"><input id="showErrPlot" type="checkbox" checked><span>err plot</span></label>
      </div>
      <button id="reset">Reset</button>
      <div class="readout" id="readout"></div>
      <canvas id="plot" class="plot"></canvas>
    </aside>
  </main>
  <section id="analysisView" class="analysis-view hidden">
    <aside class="analysis-menu">
      <button class="analysis-kind active" data-kind="camera" type="button">Camera Video</button>
      <button class="analysis-kind" data-kind="trajectory" type="button">3D / 2D Replay</button>
      <button class="analysis-kind" data-kind="ball" type="button">Ball State</button>
      <button class="analysis-kind" data-kind="policyInput" type="button">Policy Input</button>
      <button class="analysis-kind" data-kind="policyOutput" type="button">Policy Output</button>
      <button class="analysis-kind" data-kind="sim2real" type="button">Sim2Real</button>
      <div class="section-title">Detail</div>
      <div id="analysisSubList" class="analysis-sublist"></div>
      <button id="analysisRefresh" type="button">Refresh</button>
      <div class="readout" id="analysisReadout">analysis idle</div>
      <div id="analysisLinks" class="readout"></div>
    </aside>
    <div class="analysis-display">
      <div class="analysis-toolbar">
        <button id="analysisPlay" type="button">Play</button>
        <input id="analysisScrub" type="range" min="0" max="0" value="0">
        <span id="analysisFrameLabel">0 / 0</span>
      </div>
      <div class="analysis-content">
        <div id="analysisCameraPane" class="analysis-pane active">
          <img id="analysisCameraFrame" class="analysis-camera-frame" alt="recorded camera frame">
          <div id="analysisCameraEmpty" class="analysis-empty hidden">No recorded camera frames</div>
        </div>
        <div id="analysisTrajectoryPane" class="analysis-pane">
          <div class="analysis-canvas-grid">
            <canvas id="analysisReplay3d"></canvas>
            <canvas id="analysisReplayTop"></canvas>
            <canvas id="analysisReplaySide"></canvas>
          </div>
        </div>
        <div id="analysisCurvePane" class="analysis-pane">
          <canvas id="analysisCurveCanvas" class="analysis-curve-canvas"></canvas>
        </div>
        <div id="analysisReportPane" class="analysis-pane">
          <pre id="analysisNoiseReport" class="analysis-report"></pre>
        </div>
      </div>
    </div>
  </section>
  <div id="legend" class="legend">
    <span class="item" style="--c:#f28e2b">ball</span>
    <span class="item" style="--c:#e15759">noisy obs</span>
    <span class="item" style="--c:#4e79a7">robot EE</span>
    <span class="item" style="--c:#0f766e">sim EE</span>
    <span class="item" style="--c:#16a34a">joint_state FK</span>
    <span class="item" style="--c:#9333ea">cart_states</span>
    <span class="item" style="--c:#111827">ball done bounds</span>
    <span class="item" style="--c:#0891b2">EE workspace</span>
    <span class="item" style="--c:#6b7280">hidden ball</span>
    <span class="item" style="--c:#ef4444">failsafe</span>
  </div>
</div>
<script>
const CONFIG = __CONFIG_JSON__;
const STEP_RATE_HZ = Math.max(1, Math.min(120, Number(CONFIG.browserStepRateHz) || 60));
const DISPLAY_RATE_HZ = Math.max(1, Math.min(240, Number(CONFIG.displayRateHz) || 60));
const DISPLAY_INTERVAL_MS = 1000 / DISPLAY_RATE_HZ;
let ball = {x: CONFIG.initialBall[0], y: CONFIG.initialBall[1], z: CONFIG.initialBall[2]};
let observedBall = {...ball};
let lastSentBall = {...ball};
let lastSentAt = performance.now();
let racket = {x: CONFIG.anchor[0], y: CONFIG.anchor[1], z: CONFIG.anchor[2]};
let simEe = null;
let jointEe = null;
let cartEe = null;
let trailBall = [];
let trailObservedBall = [];
let trailRacket = [];
let trailSimEe = [];
let trailJointEe = [];
let trailCartEe = [];
let plotHistory = [];
let latest = null;
let dragging = null;
let inFlight = false;
let viewYaw = -0.72;
let viewPitch = 0.46;
let viewZoom = 1.0;
let lastDragX = 0;
let lastDragY = 0;
let lastVisionSeq = 0;
let visionFrameLoading = false;
let visionFrameRequestAt = 0;
let lastVisionRequestAt = 0;
let lastSimSeq = 0;
let lastSimSourceSeq = -1;
let simFrameLoading = false;
let simFrameRequestAt = 0;
let lastSimRequestAt = 0;
let lastRenderAt = 0;
let recordingActive = false;
let recordingBusy = false;
let recordingRunDir = '';
let analysisStatus = {active: false, phase: 'idle', outputs: {}};
let analysisPollTimer = null;
let analysisLoadedKey = '';
let analysisLoadingKey = '';
let currentView = 'live';
let analysisData = null;
let analysisKind = 'camera';
let analysisDetail = 'overview';
let analysisIndex = 0;
let analysisPlaying = false;
let analysisPlayTimer = null;
let analysisYaw = -0.72;
let analysisPitch = 0.46;
let analysisDrag = null;

const liveMainEl = document.getElementById('liveMain');
const analysisViewEl = document.getElementById('analysisView');
const legendEl = document.getElementById('legend');
const view3dCanvas = document.getElementById('view3d');
const topCanvas = document.getElementById('top');
const sideCanvas = document.getElementById('side');
const visionFrameEl = document.getElementById('visionFrame');
const visionEmptyEl = document.getElementById('visionEmpty');
visionFrameEl.addEventListener('load', () => { visionFrameLoading = false; });
visionFrameEl.addEventListener('error', () => {
  visionFrameLoading = false;
  lastVisionSeq = 0;
});
const simFrameEl = document.getElementById('simFrame');
const simEmptyEl = document.getElementById('simEmpty');
simFrameEl.addEventListener('load', () => { simFrameLoading = false; });
simFrameEl.addEventListener('error', () => {
  simFrameLoading = false;
});
const liveViewButtonEl = document.getElementById('liveViewButton');
const analysisViewButtonEl = document.getElementById('analysisViewButton');
const recordingButtonEl = document.getElementById('recordingButton');
const recordingStatusEl = document.getElementById('recordingStatus');
const analyzeAfterRecordEl = document.getElementById('analyzeAfterRecord');
const analysisKindEls = Array.from(document.querySelectorAll('.analysis-kind'));
const analysisSubListEl = document.getElementById('analysisSubList');
const analysisPlayEl = document.getElementById('analysisPlay');
const analysisScrubEl = document.getElementById('analysisScrub');
const analysisFrameLabelEl = document.getElementById('analysisFrameLabel');
const analysisCameraPaneEl = document.getElementById('analysisCameraPane');
const analysisTrajectoryPaneEl = document.getElementById('analysisTrajectoryPane');
const analysisCurvePaneEl = document.getElementById('analysisCurvePane');
const analysisReportPaneEl = document.getElementById('analysisReportPane');
const analysisCameraFrameEl = document.getElementById('analysisCameraFrame');
const analysisCameraEmptyEl = document.getElementById('analysisCameraEmpty');
const analysisReplay3dEl = document.getElementById('analysisReplay3d');
const analysisReplayTopEl = document.getElementById('analysisReplayTop');
const analysisReplaySideEl = document.getElementById('analysisReplaySide');
const analysisCurveCanvasEl = document.getElementById('analysisCurveCanvas');
const analysisReadoutEl = document.getElementById('analysisReadout');
const analysisNoiseReportEl = document.getElementById('analysisNoiseReport');
const analysisLinksEl = document.getElementById('analysisLinks');
const analysisRefreshEl = document.getElementById('analysisRefresh');
const visibleEl = document.getElementById('visible');
const ballSourceEl = document.getElementById('ballSource');
const useRosStateEl = document.getElementById('useRosState');
const publishCommandsEl = document.getElementById('publishCommands');
const modeEl = document.getElementById('mode');
const rosEl = document.getElementById('ros');
const timeEl = document.getElementById('time');
const ageEl = document.getElementById('age');
const readoutEl = document.getElementById('readout');
const plotCanvas = document.getElementById('plot');
const curveEls = {
  ball: document.getElementById('showBall'),
  observed: document.getElementById('showObserved'),
  racket: document.getElementById('showRacket'),
  sim: document.getElementById('showSim'),
  joint: document.getElementById('showJoint'),
  cart: document.getElementById('showCart'),
  bounds: document.getElementById('showBounds'),
  actionPlot: document.getElementById('showActionPlot'),
  errPlot: document.getElementById('showErrPlot')
};
const flowEls = {
  rlUseValidBallObsAge: document.getElementById('rlUseValidBallObsAge'),
  enableCompensation: document.getElementById('enableCompensation'),
  enableSimMirror: document.getElementById('enableSimMirror'),
  useBallEstimator: document.getElementById('useBallEstimator'),
  estimatorPrediction: document.getElementById('estimatorPrediction'),
  estimatorFrameHold: document.getElementById('estimatorFrameHold'),
  estimatorAeroDrag: document.getElementById('estimatorAeroDrag'),
  estimatorReleaseGatedDynamics: document.getElementById('estimatorReleaseGatedDynamics'),
  releaseUseEstimatorState: document.getElementById('releaseUseEstimatorState'),
  enableCaptureGate: document.getElementById('enableCaptureGate'),
  enableReleaseGate: document.getElementById('enableReleaseGate'),
  enableLostFailsafe: document.getElementById('enableLostFailsafe'),
  enableDoneMonitoring: document.getElementById('enableDoneMonitoring')
};
const cameraFlowEls = {
  detect: document.getElementById('cameraDetect'),
  visualize: document.getElementById('cameraVisualize'),
  sensorDepth: document.getElementById('cameraSensorDepth'),
  undistort: document.getElementById('cameraUndistort'),
  kalman: document.getElementById('cameraKalman'),
  ellipseFit: document.getElementById('cameraEllipseFit'),
  temporalTracker: document.getElementById('cameraTemporalTracker'),
  colorFilter: document.getElementById('cameraColorFilter')
};
const inputs = {
  x: document.getElementById('x'),
  y: document.getElementById('y'),
  z: document.getElementById('z'),
  noise: document.getElementById('noise'),
  obsHz: document.getElementById('obsHz'),
  workspaceSafetyMode: document.getElementById('workspaceSafetyMode'),
  donePredictionHorizonS: document.getElementById('donePredictionHorizonS'),
  donePredictionDtS: document.getElementById('donePredictionDtS'),
  donePredictionMarginM: document.getElementById('donePredictionMarginM'),
  donePredictionOrientationMarginRad: document.getElementById('donePredictionOrientationMarginRad'),
  compensationMode: document.getElementById('compensationMode'),
  compDelayMs: document.getElementById('compDelayMs'),
  compActuatorTauS: document.getElementById('compActuatorTauS'),
  compActuatorGain: document.getElementById('compActuatorGain'),
  compMpcBeta: document.getElementById('compMpcBeta'),
  compMpcDelayScale: document.getElementById('compMpcDelayScale'),
  compMpcTauScale: document.getElementById('compMpcTauScale'),
  compMpcHorizonSteps: document.getElementById('compMpcHorizonSteps'),
  compMpcTrackingWeight: document.getElementById('compMpcTrackingWeight'),
  compMpcNominalWeight: document.getElementById('compMpcNominalWeight'),
  compMpcDeltaWeight: document.getElementById('compMpcDeltaWeight'),
  compMpcMaxDeltaRad: document.getElementById('compMpcMaxDeltaRad'),
  estimatorFrameHoldHz: document.getElementById('estimatorFrameHoldHz'),
  estimatorMaxExtrapolationS: document.getElementById('estimatorMaxExtrapolationS'),
  estimatorControlMaxExtrapolationS: document.getElementById('estimatorControlMaxExtrapolationS'),
  estimatorTrustLossS: document.getElementById('estimatorTrustLossS'),
  estimatorDegradeLossS: document.getElementById('estimatorDegradeLossS'),
  estimatorInvalidAfterS: document.getElementById('estimatorInvalidAfterS'),
  estimatorPredictionHorizonS: document.getElementById('estimatorPredictionHorizonS'),
  estimatorQPosMm2S: document.getElementById('estimatorQPosMm2S'),
  estimatorQVelMult: document.getElementById('estimatorQVelMult'),
  estimatorQDepthPosMult: document.getElementById('estimatorQDepthPosMult'),
  estimatorQDepthVelMult: document.getElementById('estimatorQDepthVelMult'),
  estimatorRLateralMm: document.getElementById('estimatorRLateralMm'),
  estimatorRDepthMm: document.getElementById('estimatorRDepthMm'),
  releaseMinVz: document.getElementById('releaseMinVz'),
  releaseMinDrop: document.getElementById('releaseMinDrop'),
  releaseWindowS: document.getElementById('releaseWindowS'),
  releaseTicks: document.getElementById('releaseTicks'),
  releaseMaxAgeS: document.getElementById('releaseMaxAgeS')
};

function fmt(v, n=3) { return Number.isFinite(v) ? v.toFixed(n) : 'nan'; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function curveOn(name) { return curveEls[name] ? curveEls[name].checked : true; }
function readNumber(el, fallback, lo=null, hi=null) {
  let v = Number(el.value);
  if (!Number.isFinite(v)) v = fallback;
  if (lo !== null) v = Math.max(lo, v);
  if (hi !== null) v = Math.min(hi, v);
  return v;
}
function readInt(el, fallback, lo=null, hi=null) {
  return Math.round(readNumber(el, fallback, lo, hi));
}
function setNumberDefault(el, value, digits=3) {
  const v = Number(value);
  if (el && !el.value && Number.isFinite(v)) el.value = fmt(v, digits);
}
function setControlDisabled(el, disabled) {
  if (!el) return;
  el.disabled = disabled;
  const row = el.closest('.checkrow') || el.closest('.row') || el.parentElement;
  if (row) row.classList.toggle('disabled', disabled);
}
function syncEstimatorModeControl() {
  const estimatorOff = !flowEls.useBallEstimator.checked;
  const predictionOff = estimatorOff || !flowEls.estimatorPrediction.checked;
  const releaseOff = !flowEls.enableReleaseGate.checked;
  const predictiveOff = (
    !flowEls.enableDoneMonitoring.checked
    || !['predictive', 'predictive_soft'].includes(
      inputs.workspaceSafetyMode.value));
  const compensationOff = (
    flowEls.enableCompensation.disabled || !flowEls.enableCompensation.checked);
  setControlDisabled(flowEls.estimatorFrameHold, predictionOff);
  setControlDisabled(flowEls.estimatorAeroDrag, predictionOff);
  setControlDisabled(flowEls.estimatorReleaseGatedDynamics, estimatorOff || releaseOff);
  setControlDisabled(flowEls.releaseUseEstimatorState, releaseOff || estimatorOff);
  for (const el of [
    inputs.compensationMode,
    inputs.compDelayMs,
    inputs.compActuatorTauS,
    inputs.compActuatorGain,
    inputs.compMpcBeta,
    inputs.compMpcDelayScale,
    inputs.compMpcTauScale,
    inputs.compMpcHorizonSteps,
    inputs.compMpcTrackingWeight,
    inputs.compMpcNominalWeight,
    inputs.compMpcDeltaWeight,
    inputs.compMpcMaxDeltaRad
  ]) {
    setControlDisabled(el, compensationOff);
  }
  for (const el of [
    inputs.estimatorTrustLossS,
    inputs.estimatorDegradeLossS,
    inputs.estimatorInvalidAfterS,
    inputs.estimatorQPosMm2S,
    inputs.estimatorQVelMult,
    inputs.estimatorQDepthPosMult,
    inputs.estimatorQDepthVelMult,
    inputs.estimatorRLateralMm,
    inputs.estimatorRDepthMm
  ]) {
    setControlDisabled(el, estimatorOff);
  }
  for (const el of [
    inputs.estimatorMaxExtrapolationS,
    inputs.estimatorControlMaxExtrapolationS,
    inputs.estimatorPredictionHorizonS
  ]) {
    setControlDisabled(el, predictionOff);
  }
  setControlDisabled(
    inputs.estimatorFrameHoldHz,
    predictionOff || !flowEls.estimatorFrameHold.checked);
  for (const el of [
    inputs.releaseMinVz,
    inputs.releaseMinDrop,
    inputs.releaseWindowS,
    inputs.releaseTicks,
    inputs.releaseMaxAgeS
  ]) {
    setControlDisabled(el, releaseOff);
  }
  for (const el of [
    inputs.donePredictionHorizonS,
    inputs.donePredictionDtS,
    inputs.donePredictionMarginM,
    inputs.donePredictionOrientationMarginRad
  ]) {
    setControlDisabled(el, predictiveOff);
  }
}
function metersPerSecond(a, b, dt) {
  return [(a.x - b.x) / dt, (a.y - b.y) / dt, (a.z - b.z) / dt].map(v => Number.isFinite(v) ? v : 0);
}

function setRuntimeDefaults() {
  useRosStateEl.checked = !!CONFIG.rosDefaultUseState && !!CONFIG.rosBridge;
  useRosStateEl.disabled = !CONFIG.rosBridge;
  publishCommandsEl.checked = !!CONFIG.rosDefaultPublishCommands && !!CONFIG.rosBridge && !!CONFIG.rosPublishAllowed;
  publishCommandsEl.disabled = !CONFIG.rosBridge || !CONFIG.rosPublishAllowed;
  const cameraEnabled = !!(CONFIG.cameraBall && CONFIG.cameraBall.enabled);
  const cameraAvailable = !!(CONFIG.cameraBall && CONFIG.cameraBall.available);
  for (const opt of Array.from(ballSourceEl.options)) {
    if (opt.value === 'camera') opt.disabled = !(cameraEnabled && cameraAvailable);
  }
  if (cameraEnabled && cameraAvailable) {
    ballSourceEl.value = 'camera';
  }
  if (!(cameraEnabled && cameraAvailable) && ballSourceEl.value === 'camera') {
    ballSourceEl.value = 'drag';
  }
  const cameraDefaults = (CONFIG.cameraBall && CONFIG.cameraBall.defaults) || {};
  cameraFlowEls.detect.checked = cameraDefaults.detect !== false;
  cameraFlowEls.visualize.checked = cameraDefaults.visualize !== false;
  cameraFlowEls.sensorDepth.checked = cameraDefaults.sensorDepth !== false;
  cameraFlowEls.undistort.checked = cameraDefaults.undistort !== false;
  cameraFlowEls.kalman.checked = !!cameraDefaults.kalman;
  cameraFlowEls.ellipseFit.checked = cameraDefaults.ellipseFit !== false;
  cameraFlowEls.temporalTracker.checked = !!cameraDefaults.temporalTracker;
  cameraFlowEls.colorFilter.checked = cameraDefaults.colorFilter !== false;
  for (const el of Object.values(cameraFlowEls)) {
    el.disabled = !(cameraEnabled && cameraAvailable);
    el.closest('.checkrow').classList.toggle('disabled', el.disabled);
  }
  useRosStateEl.parentElement.classList.toggle('disabled', useRosStateEl.disabled);
  publishCommandsEl.parentElement.classList.toggle('disabled', publishCommandsEl.disabled);
  const policyDefaults = CONFIG.policy || {};
  const compensationDefaults = CONFIG.compensation || {};
  const releaseDefaults = CONFIG.releaseGate || {};
  const estimatorDefaults = CONFIG.ballEstimator || {};
  const donePredictionDefaults = CONFIG.donePrediction || {};
  flowEls.rlUseValidBallObsAge.checked = policyDefaults.useValidBallObsAge !== false;
  flowEls.enableCompensation.checked = compensationDefaults.enabled !== false;
  flowEls.enableCompensation.disabled = compensationDefaults.available === false;
  flowEls.enableCompensation.closest('.checkrow').classList.toggle(
    'disabled', flowEls.enableCompensation.disabled);
  inputs.workspaceSafetyMode.value = (
    (CONFIG.flowDefaults && CONFIG.flowDefaults.workspaceSafetyMode) || 'position_soft');
  setNumberDefault(inputs.donePredictionHorizonS, donePredictionDefaults.horizonS, 3);
  setNumberDefault(inputs.donePredictionDtS, donePredictionDefaults.dtS, 3);
  setNumberDefault(inputs.donePredictionMarginM, donePredictionDefaults.marginM, 3);
  setNumberDefault(
    inputs.donePredictionOrientationMarginRad,
    donePredictionDefaults.orientationMarginRad,
    3);
  const simDefaults = CONFIG.simMirror || {};
  flowEls.enableSimMirror.checked = !!simDefaults.defaultEnabled;
  flowEls.enableSimMirror.disabled = !simDefaults.enabled;
  flowEls.enableSimMirror.closest('.checkrow').classList.toggle(
    'disabled', flowEls.enableSimMirror.disabled);
  inputs.compensationMode.value = compensationDefaults.mode || 'inverse_mpc';
  setNumberDefault(inputs.compDelayMs, compensationDefaults.delayMs, 1);
  setNumberDefault(inputs.compActuatorTauS, compensationDefaults.tauS, 3);
  setNumberDefault(inputs.compActuatorGain, compensationDefaults.gain, 2);
  setNumberDefault(inputs.compMpcBeta, compensationDefaults.mpcBeta, 2);
  setNumberDefault(inputs.compMpcDelayScale, compensationDefaults.mpcDelayScale, 2);
  setNumberDefault(inputs.compMpcTauScale, compensationDefaults.mpcTauScale, 2);
  setNumberDefault(inputs.compMpcHorizonSteps, compensationDefaults.mpcHorizonSteps, 0);
  setNumberDefault(inputs.compMpcTrackingWeight, compensationDefaults.mpcTrackingWeight, 2);
  setNumberDefault(inputs.compMpcNominalWeight, compensationDefaults.mpcNominalWeight, 2);
  setNumberDefault(inputs.compMpcDeltaWeight, compensationDefaults.mpcDeltaWeight, 2);
  setNumberDefault(inputs.compMpcMaxDeltaRad, compensationDefaults.mpcMaxDeltaRad, 3);
  flowEls.useBallEstimator.checked = !!(CONFIG.ballEstimator && CONFIG.ballEstimator.enabled);
  flowEls.estimatorPrediction.checked = !!(CONFIG.ballEstimator && CONFIG.ballEstimator.prediction);
  flowEls.estimatorFrameHold.checked = !!(
    CONFIG.ballEstimator && CONFIG.ballEstimator.controlOutputMode === 'predict_on_frame_hold');
  flowEls.estimatorAeroDrag.checked = !!(CONFIG.ballEstimator && CONFIG.ballEstimator.aeroDrag);
  flowEls.estimatorReleaseGatedDynamics.checked = !!estimatorDefaults.releaseGatedDynamics;
  flowEls.releaseUseEstimatorState.checked = !!releaseDefaults.useEstimatorState;
  flowEls.enableCaptureGate.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.captureGate);
  flowEls.enableReleaseGate.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.releaseGate);
  flowEls.enableLostFailsafe.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.lostFailsafe);
  flowEls.enableDoneMonitoring.checked = !!(CONFIG.flowDefaults && CONFIG.flowDefaults.doneMonitoring);
  setNumberDefault(inputs.estimatorFrameHoldHz, estimatorDefaults.frameHoldOutputRateHz, 1);
  setNumberDefault(inputs.estimatorMaxExtrapolationS, estimatorDefaults.maxExtrapolationS, 3);
  setNumberDefault(inputs.estimatorControlMaxExtrapolationS, estimatorDefaults.controlMaxExtrapolationS, 3);
  setNumberDefault(inputs.estimatorTrustLossS, estimatorDefaults.trustLossS, 3);
  setNumberDefault(inputs.estimatorDegradeLossS, estimatorDefaults.degradeLossS, 3);
  setNumberDefault(inputs.estimatorInvalidAfterS, estimatorDefaults.invalidAfterS, 3);
  setNumberDefault(inputs.estimatorPredictionHorizonS, estimatorDefaults.predictionHorizonS, 3);
  setNumberDefault(inputs.estimatorQPosMm2S, estimatorDefaults.qPosMm2S, 1);
  setNumberDefault(inputs.estimatorQVelMult, estimatorDefaults.qVelMult, 1);
  setNumberDefault(inputs.estimatorQDepthPosMult, estimatorDefaults.qDepthPosMult, 3);
  setNumberDefault(inputs.estimatorQDepthVelMult, estimatorDefaults.qDepthVelMult, 3);
  setNumberDefault(inputs.estimatorRLateralMm, estimatorDefaults.rLateralMm, 1);
  setNumberDefault(inputs.estimatorRDepthMm, estimatorDefaults.rDepthMm, 1);
  setNumberDefault(inputs.releaseMinVz, releaseDefaults.minDownwardVelocityMS, 3);
  setNumberDefault(inputs.releaseMinDrop, releaseDefaults.minZDropM, 3);
  setNumberDefault(inputs.releaseWindowS, releaseDefaults.windowS, 3);
  setNumberDefault(inputs.releaseTicks, releaseDefaults.requiredTicks, 0);
  setNumberDefault(inputs.releaseMaxAgeS, releaseDefaults.maxObsAgeS, 3);
  syncEstimatorModeControl();
}

function setInputsFromBall() {
  inputs.x.value = fmt(ball.x, 4);
  inputs.y.value = fmt(ball.y, 4);
  inputs.z.value = fmt(ball.z, 4);
  if (!inputs.noise.value) inputs.noise.value = fmt((CONFIG.defaultNoiseStdM || 0) * 1000, 1);
  if (!inputs.obsHz.value) inputs.obsHz.value = fmt(CONFIG.defaultBallObservationRateHz || 60, 0);
}

function setBallFromInputs() {
  ball.x = Number(inputs.x.value);
  ball.y = Number(inputs.y.value);
  ball.z = Number(inputs.z.value);
}

for (const el of [inputs.x, inputs.y, inputs.z]) {
  el.addEventListener('change', () => { setBallFromInputs(); drawAll(); });
}
inputs.noise.addEventListener('change', drawAll);
inputs.obsHz.addEventListener('change', drawAll);
for (const el of [
  inputs.workspaceSafetyMode,
  inputs.donePredictionHorizonS,
  inputs.donePredictionDtS,
  inputs.donePredictionMarginM,
  inputs.donePredictionOrientationMarginRad,
  inputs.compensationMode,
  inputs.compDelayMs,
  inputs.compActuatorTauS,
  inputs.compActuatorGain,
  inputs.compMpcBeta,
  inputs.compMpcDelayScale,
  inputs.compMpcTauScale,
  inputs.compMpcHorizonSteps,
  inputs.compMpcTrackingWeight,
  inputs.compMpcNominalWeight,
  inputs.compMpcDeltaWeight,
  inputs.compMpcMaxDeltaRad,
  inputs.estimatorFrameHoldHz,
  inputs.estimatorMaxExtrapolationS,
  inputs.estimatorControlMaxExtrapolationS,
  inputs.estimatorTrustLossS,
  inputs.estimatorDegradeLossS,
  inputs.estimatorInvalidAfterS,
  inputs.estimatorPredictionHorizonS,
  inputs.estimatorQPosMm2S,
  inputs.estimatorQVelMult,
  inputs.estimatorQDepthPosMult,
  inputs.estimatorQDepthVelMult,
  inputs.estimatorRLateralMm,
  inputs.estimatorRDepthMm,
  inputs.releaseMinVz,
  inputs.releaseMinDrop,
  inputs.releaseWindowS,
  inputs.releaseTicks,
  inputs.releaseMaxAgeS
]) {
  el.addEventListener('change', () => {
    syncEstimatorModeControl();
    drawAll();
  });
}

function resizeCanvas(canvas) {
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function projectTop(p) {
  const rect = topCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: (p.x - l.x0) / (l.x1 - l.x0) * rect.width,
    y: rect.height - (p.y - l.y0) / (l.y1 - l.y0) * rect.height
  };
}

function unprojectTop(q) {
  const rect = topCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: l.x0 + q.x / rect.width * (l.x1 - l.x0),
    y: l.y0 + (1 - q.y / rect.height) * (l.y1 - l.y0)
  };
}

function projectSide(p) {
  const rect = sideCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: (p.x - l.x0) / (l.x1 - l.x0) * rect.width,
    y: rect.height - (p.z - l.z0) / (l.z1 - l.z0) * rect.height
  };
}

function unprojectSide(q) {
  const rect = sideCanvas.getBoundingClientRect();
  const l = CONFIG.limits;
  return {
    x: l.x0 + q.x / rect.width * (l.x1 - l.x0),
    z: l.z0 + (1 - q.y / rect.height) * (l.z1 - l.z0)
  };
}

function sceneCenter() {
  const l = CONFIG.limits;
  return {
    x: (l.x0 + l.x1) * 0.5,
    y: (l.y0 + l.y1) * 0.5,
    z: (l.z0 + l.z1) * 0.5
  };
}

function sceneRadius() {
  const l = CONFIG.limits;
  return Math.max(l.x1 - l.x0, l.y1 - l.y0, l.z1 - l.z0) * 0.58;
}

function rotate3D(p) {
  const c = sceneCenter();
  let x = p.x - c.x, y = p.y - c.y, z = p.z - c.z;
  const cy = Math.cos(viewYaw), sy = Math.sin(viewYaw);
  const cp = Math.cos(viewPitch), sp = Math.sin(viewPitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const y2 = cp * y1 - sp * z;
  const z2 = sp * y1 + cp * z;
  return {x:x1, y:y2, z:z2};
}

function project3D(p) {
  const rect = view3dCanvas.getBoundingClientRect();
  const r = rotate3D(p);
  const scale = Math.min(rect.width, rect.height) * 0.42 * viewZoom / Math.max(sceneRadius(), 1e-6);
  return {
    x: rect.width * 0.52 + r.x * scale,
    y: rect.height * 0.54 - r.y * scale,
    depth: r.z
  };
}

function drawRect(ctx, canvas, box, projection, color, alpha, label) {
  if (!box) return;
  const corners = projection === projectTop
    ? [{x:box.x0,y:box.y0,z:box.z0},{x:box.x1,y:box.y1,z:box.z1}]
    : [{x:box.x0,y:0,z:box.z0},{x:box.x1,y:0,z:box.z1}];
  const a = projection(corners[0]);
  const b = projection(corners[1]);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  ctx.fillText(label, Math.min(a.x, b.x) + 8, Math.min(a.y, b.y) + 18);
  ctx.restore();
}

function drawBox3D(ctx, box, color, alpha, label) {
  if (!box) return;
  const pts = [
    {x:box.x0,y:box.y0,z:box.z0},{x:box.x1,y:box.y0,z:box.z0},{x:box.x1,y:box.y1,z:box.z0},{x:box.x0,y:box.y1,z:box.z0},
    {x:box.x0,y:box.y0,z:box.z1},{x:box.x1,y:box.y0,z:box.z1},{x:box.x1,y:box.y1,z:box.z1},{x:box.x0,y:box.y1,z:box.z1}
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  for (const [a,b] of edges) {
    const pa = project3D(pts[a]), pb = project3D(pts[b]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  const q = project3D(pts[4]);
  ctx.fillText(label, q.x + 6, q.y - 6);
  ctx.restore();
}

function drawTrail(ctx, trail, projection, color, alpha) {
  if (trail.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < trail.length; i++) {
    const p = projection(trail[i]);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawDot(ctx, p, projection, color, radius, stroke='#ffffff') {
  const q = projection(p);
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(q.x, q.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawPanel(canvas, title, projection, axesText) {
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (curveOn('bounds')) {
    drawRect(ctx, canvas, CONFIG.ballBox, projection, '#111827', 0.24, 'done');
    drawRect(ctx, canvas, CONFIG.eeBox, projection, '#0891b2', 0.48, 'EE');
    drawDot(ctx, {x:CONFIG.anchor[0], y:CONFIG.anchor[1], z:CONFIG.anchor[2]}, projection, '#111827', 5);
  }
  if (curveOn('ball')) drawTrail(ctx, trailBall, projection, visibleEl.checked ? '#f28e2b' : '#6b7280', 0.42);
  if (curveOn('observed')) drawTrail(ctx, trailObservedBall, projection, '#e15759', 0.32);
  if (curveOn('racket')) drawTrail(ctx, trailRacket, projection, '#4e79a7', 0.65);
  if (curveOn('sim')) drawTrail(ctx, trailSimEe, projection, '#0f766e', 0.68);
  if (curveOn('joint')) drawTrail(ctx, trailJointEe, projection, '#16a34a', 0.70);
  if (curveOn('cart')) drawTrail(ctx, trailCartEe, projection, '#9333ea', 0.70);
  if (curveOn('ball')) drawDot(ctx, ball, projection, visibleEl.checked ? '#f28e2b' : '#6b7280', 8);
  if (latest && latest.valid && curveOn('observed')) drawDot(ctx, observedBall, projection, '#e15759', 5, '#ffffff');
  if (curveOn('racket')) drawDot(ctx, racket, projection, latest && latest.mode.includes('failsafe') ? '#ef4444' : '#4e79a7', 8);
  if (simEe && curveOn('sim')) drawDot(ctx, simEe, projection, '#0f766e', 7, '#ffffff');
  if (jointEe && curveOn('joint')) drawDot(ctx, jointEe, projection, '#16a34a', 6, '#ffffff');
  if (cartEe && curveOn('cart')) drawDot(ctx, cartEe, projection, '#9333ea', 6, '#ffffff');
  ctx.save();
  ctx.fillStyle = '#273445';
  ctx.font = '14px sans-serif';
  ctx.fillText(title, 14, 22);
  ctx.fillStyle = '#64748b';
  ctx.font = '12px sans-serif';
  ctx.fillText(axesText, 14, 40);
  ctx.restore();
}

function draw3D() {
  const canvas = view3dCanvas;
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (curveOn('bounds')) {
    drawBox3D(ctx, CONFIG.ballBox, '#111827', 0.24, 'done');
    drawBox3D(ctx, CONFIG.eeBox, '#0891b2', 0.52, 'EE');
    drawDot(ctx, {x:CONFIG.anchor[0], y:CONFIG.anchor[1], z:CONFIG.anchor[2]}, project3D, '#111827', 5);
  }
  if (curveOn('ball')) drawTrail(ctx, trailBall, project3D, visibleEl.checked ? '#f28e2b' : '#6b7280', 0.38);
  if (curveOn('observed')) drawTrail(ctx, trailObservedBall, project3D, '#e15759', 0.42);
  if (curveOn('racket')) drawTrail(ctx, trailRacket, project3D, '#4e79a7', 0.78);
  if (curveOn('sim')) drawTrail(ctx, trailSimEe, project3D, '#0f766e', 0.82);
  if (curveOn('joint')) drawTrail(ctx, trailJointEe, project3D, '#16a34a', 0.86);
  if (curveOn('cart')) drawTrail(ctx, trailCartEe, project3D, '#9333ea', 0.86);
  if (curveOn('ball')) drawDot(ctx, ball, project3D, visibleEl.checked ? '#f28e2b' : '#6b7280', 8);
  if (latest && latest.valid && curveOn('observed')) drawDot(ctx, observedBall, project3D, '#e15759', 6, '#ffffff');
  if (curveOn('racket')) drawDot(ctx, racket, project3D, latest && latest.mode.includes('failsafe') ? '#ef4444' : '#4e79a7', 8);
  if (simEe && curveOn('sim')) drawDot(ctx, simEe, project3D, '#0f766e', 7, '#ffffff');
  if (jointEe && curveOn('joint')) drawDot(ctx, jointEe, project3D, '#16a34a', 6, '#ffffff');
  if (cartEe && curveOn('cart')) drawDot(ctx, cartEe, project3D, '#9333ea', 6, '#ffffff');
  ctx.save();
  ctx.fillStyle = '#273445';
  ctx.font = '14px sans-serif';
  ctx.fillText('3D view', 14, 22);
  ctx.fillStyle = '#64748b';
  ctx.font = '12px sans-serif';
  ctx.fillText('drag to rotate / wheel to zoom', 14, 40);
  ctx.restore();
}

function drawAll() {
  draw3D();
  drawPanel(topCanvas, 'Top view', projectTop, 'x / y');
  drawPanel(sideCanvas, 'Side view', projectSide, 'x / z');
  drawPlot();
}

function renderLoop(timestamp) {
  if (!lastRenderAt || timestamp - lastRenderAt >= DISPLAY_INTERVAL_MS - 1) {
    lastRenderAt = timestamp;
    drawAll();
  }
  requestAnimationFrame(renderLoop);
}

function updateVisionFrame(camera) {
  const seq = camera && Number.isFinite(camera.visualizationSeq)
    ? camera.visualizationSeq : 0;
  const vizOn = !!(camera && camera.runtime && camera.runtime.visualize);
  if (!vizOn) {
    visionFrameLoading = false;
    visionEmptyEl.style.display = 'grid';
    visionEmptyEl.textContent = 'camera overlay off';
    return;
  }
  const nowMs = performance.now();
  if (visionFrameLoading && nowMs - visionFrameRequestAt > 1000) {
    visionFrameLoading = false;
    lastVisionSeq = 0;
  }
  const targetHz = Math.max(
    1,
    Math.min(60, Number(camera && camera.visualizationRateHz) || 30)
  );
  const frameIntervalMs = 1000 / targetHz;
  if (seq > 0
      && !visionFrameLoading
      && nowMs - lastVisionRequestAt >= frameIntervalMs - 1) {
    visionFrameLoading = true;
    visionFrameRequestAt = nowMs;
    lastVisionRequestAt = nowMs;
    lastVisionSeq = seq;
    visionFrameEl.src = `/camera_frame.jpg?seq=${seq}&t=${Date.now()}`;
    visionEmptyEl.style.display = 'none';
  } else if (seq <= 0) {
    visionFrameLoading = false;
    const frames = camera && Number.isFinite(camera.frameCount) ? camera.frameCount : 0;
    const active = camera && camera.activeTransport ? camera.activeTransport : 'none';
    const info = camera && camera.cameraInfoReceived ? 'info' : 'no-info';
    const detector = camera && camera.detectorReady ? 'detector' : 'no-detector';
    const source = camera && camera.sourceMode ? camera.sourceMode : 'ros_topic';
    const camRt = (camera && camera.runtime) || {};
    const sensorDepthOn = camRt.sensorDepth !== false;
    const sourceText = source === 'direct_realsense'
      ? `direct RealSense ${camera.imageWidth || '?'}x${camera.imageHeight || '?'}@${camera.realsenseFps || '?'} hybrid=${camera.useHybridDetector ? 'on' : 'off'} sensorDepth=${sensorDepthOn ? 'on' : 'off'}`
      : (camera && camera.imageTransport === 'raw'
        ? `topic=${camera.rawImageTopic}`
        : (camera && camera.imageTransport === 'compressed'
          ? `topic=${camera.imageTopic}`
          : `topic=${camera ? camera.imageTopic : ''} | ${camera ? camera.rawImageTopic : ''}`));
    const err = camera && camera.error ? ` | error=${camera.error}` : '';
    visionEmptyEl.style.display = 'grid';
    visionEmptyEl.textContent = camera && camera.enabled
      ? `camera overlay waiting | frames=${frames} | ${info} | ${detector} | active=${active} | source=${sourceText || 'none'}${err}`
      : 'camera source off';
  }
}

function updateSimFrame(sim) {
  sim = sim || {};
  const frame = sim.frame || {};
  const seq = Number.isFinite(frame.seq) ? Number(frame.seq) : 0;
  const enabled = !!sim.enabled;
  const active = !!sim.active;
  const available = sim.available !== false;
  if (!enabled) {
    simFrameLoading = false;
    simFrameEl.removeAttribute('src');
    simEmptyEl.style.display = 'grid';
    simEmptyEl.textContent = 'sim mirror off';
    return;
  }
  if (!active || !available) {
    simFrameLoading = false;
    if (!simFrameEl.getAttribute('src')) {
      simEmptyEl.style.display = 'grid';
      if (!available) simEmptyEl.textContent = `sim unavailable${sim.error ? ` | ${sim.error}` : ''}`;
      else simEmptyEl.textContent = 'sim mirror waiting';
    }
    return;
  }
  if (!((CONFIG.simMirror || {}).videoFrames)) {
    simFrameLoading = false;
    simFrameEl.removeAttribute('src');
    simEmptyEl.style.display = 'grid';
    simEmptyEl.textContent = `sim mirror running${sim.backend ? ` | ${sim.backend}` : ''} | video off`;
    return;
  }
  const nowMs = performance.now();
  if (simFrameLoading && nowMs - simFrameRequestAt > 1000) {
    simFrameLoading = false;
  }
  const targetHz = Math.max(
    1,
    Math.min(60, Number((CONFIG.simMirror || {}).frameRateHz) || 30)
  );
  const frameIntervalMs = 1000 / targetHz;
  if (seq > 0
      && seq !== lastSimSeq
      && !simFrameLoading
      && nowMs - lastSimRequestAt >= frameIntervalMs - 1) {
    simFrameLoading = true;
    simFrameRequestAt = nowMs;
    lastSimRequestAt = nowMs;
    lastSimSeq = seq;
    simFrameEl.src = `/sim_frame.jpg?seq=${seq}&t=${Date.now()}`;
    simEmptyEl.style.display = 'none';
  } else if (seq <= 0) {
    simFrameLoading = false;
    if (!simFrameEl.getAttribute('src')) {
      simEmptyEl.style.display = 'grid';
      simEmptyEl.textContent = `sim mirror running${sim.backend ? ` | ${sim.backend}` : ''}`;
    }
  }
}

function drawPlot() {
  const ctx = plotCanvas.getContext('2d');
  const rect = plotCanvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.save();
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = rect.height * i / 4;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(rect.width, y);
    ctx.stroke();
  }
  ctx.fillStyle = '#334155';
  ctx.font = '12px sans-serif';
  const showAction = curveOn('actionPlot');
  const showErr = curveOn('errPlot');
  const title = showAction && showErr
    ? 'action max / cmd-joint err(rad)'
    : (showAction ? 'action max' : (showErr ? 'cmd-joint err(rad)' : 'plot hidden'));
  ctx.fillText(title, 8, 16);

  function drawSeries(key, color, scale) {
    if (plotHistory.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < plotHistory.length; i++) {
      const v = Math.min(1, Math.max(0, (plotHistory[i][key] || 0) / scale));
      const x = rect.width * i / Math.max(1, plotHistory.length - 1);
      const y = rect.height - 18 - v * (rect.height - 34);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  if (showAction) drawSeries('action', '#4e79a7', 1.0);
  if (showErr) drawSeries('err', '#16a34a', 0.35);
  if (showAction) {
    ctx.fillStyle = '#4e79a7';
    ctx.fillText('action', 8, rect.height - 8);
  }
  if (showErr) {
    ctx.fillStyle = '#16a34a';
    ctx.fillText('err', showAction ? 70 : 8, rect.height - 8);
  }
  ctx.restore();
}

function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return {x: event.clientX - rect.left, y: event.clientY - rect.top};
}

function beginDrag(event, kind) {
  dragging = kind;
  event.currentTarget.setPointerCapture(event.pointerId);
  updateDrag(event);
}

function updateDrag(event) {
  if (!dragging) return;
  if (dragging === 'top') {
    const p = unprojectTop(canvasPoint(event, topCanvas));
    ball.x = clamp(p.x, CONFIG.limits.x0, CONFIG.limits.x1);
    ball.y = clamp(p.y, CONFIG.limits.y0, CONFIG.limits.y1);
  } else if (dragging === 'side') {
    const p = unprojectSide(canvasPoint(event, sideCanvas));
    ball.x = clamp(p.x, CONFIG.limits.x0, CONFIG.limits.x1);
    ball.z = clamp(p.z, CONFIG.limits.z0, CONFIG.limits.z1);
  } else if (dragging === '3d') {
    const dx = event.clientX - lastDragX;
    const dy = event.clientY - lastDragY;
    lastDragX = event.clientX;
    lastDragY = event.clientY;
    viewYaw += dx * 0.008;
    viewPitch = clamp(viewPitch + dy * 0.008, -1.35, 1.35);
  }
  setInputsFromBall();
  drawAll();
}

view3dCanvas.addEventListener('pointerdown', e => {
  dragging = '3d';
  lastDragX = e.clientX;
  lastDragY = e.clientY;
  view3dCanvas.classList.add('dragging');
  view3dCanvas.setPointerCapture(e.pointerId);
});
topCanvas.addEventListener('pointerdown', e => beginDrag(e, 'top'));
sideCanvas.addEventListener('pointerdown', e => beginDrag(e, 'side'));
for (const canvas of [view3dCanvas, topCanvas, sideCanvas]) {
  canvas.addEventListener('pointermove', updateDrag);
  canvas.addEventListener('pointerup', e => {
    dragging = null;
    view3dCanvas.classList.remove('dragging');
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
  });
  canvas.addEventListener('pointerleave', () => {
    dragging = null;
    view3dCanvas.classList.remove('dragging');
  });
}
view3dCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  viewZoom *= Math.exp(-e.deltaY * 0.001);
  viewZoom = clamp(viewZoom, 0.25, 8);
  drawAll();
}, {passive:false});

async function postJson(path, payload) {
  const res = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function getJson(path) {
  const res = await fetch(path, {cache: 'no-store'});
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

function compactPoint(p) {
  return p ? {x: Number(p.x), y: Number(p.y), z: Number(p.z)} : null;
}

function copyTrail(trail) {
  return trail.map(p => compactPoint(p)).filter(Boolean);
}

function buildRecordingView(kind='sample', includeTrails=false) {
  const view = {
    kind,
    wallTimeMs: Date.now(),
    perfTimeMs: performance.now(),
    view3d: {yaw: viewYaw, pitch: viewPitch, zoom: viewZoom},
    current: {
      ball: compactPoint(ball),
      observedBall: compactPoint(observedBall),
      racket: compactPoint(racket),
      simEe: compactPoint(simEe),
      jointEe: compactPoint(jointEe),
      cartEe: compactPoint(cartEe)
    },
    toggles: {
      ball: curveOn('ball'),
      observed: curveOn('observed'),
      racket: curveOn('racket'),
      sim: curveOn('sim'),
      joint: curveOn('joint'),
      cart: curveOn('cart'),
      bounds: curveOn('bounds'),
      actionPlot: curveOn('actionPlot'),
      errPlot: curveOn('errPlot')
    }
  };
  if (includeTrails) {
    view.trails = {
      ball: copyTrail(trailBall),
      observedBall: copyTrail(trailObservedBall),
      racket: copyTrail(trailRacket),
      simEe: copyTrail(trailSimEe),
      jointEe: copyTrail(trailJointEe),
      cartEe: copyTrail(trailCartEe)
    };
    view.plotHistory = plotHistory.slice();
  }
  return view;
}

function updateRecordingUi(status) {
  status = status || {};
  recordingActive = !!status.active;
  recordingRunDir = status.runDir || recordingRunDir || '';
  if (status.analysis) updateAnalysisUi(status.analysis);
  recordingButtonEl.classList.toggle('active', recordingActive);
  recordingButtonEl.classList.toggle('busy', recordingBusy);
  recordingButtonEl.disabled = recordingBusy;
  recordingButtonEl.textContent = recordingActive ? 'Stop' : 'Record';
  const counts = status.counts || {};
  const pieces = recordingActive
    ? [`rec`, `${counts.control || 0} ctrl`, `${counts.snapshots || 0} snap`, `${counts.frames || 0} img`, `${counts.simFrames || 0} simimg`]
    : ['idle'];
  if (status.dropped) pieces.push(`drop ${status.dropped}`);
  if (status.error) pieces.push(`err ${status.error}`);
  recordingStatusEl.textContent = pieces.join(' / ');
}

async function refreshRecordingStatus() {
  try {
    updateRecordingUi(await getJson('/recording/status'));
  } catch (_) {
    updateRecordingUi({active: recordingActive, runDir: recordingRunDir, error: 'status'});
  }
}

function analysisFileUrl(name) {
  if (!name) return '';
  return `/analysis/file/${encodeURIComponent(name)}?t=${Date.now()}`;
}

function analysisFrameUrl(name) {
  if (!name) return '';
  return `/analysis/frame/${encodeURIComponent(name)}?t=${Date.now()}`;
}

function setAppView(view) {
  const analysis = view === 'analysis';
  const changed = currentView !== (analysis ? 'analysis' : 'live');
  currentView = analysis ? 'analysis' : 'live';
  liveMainEl.classList.toggle('hidden', analysis);
  legendEl.classList.toggle('hidden', analysis);
  analysisViewEl.classList.toggle('hidden', !analysis);
  liveViewButtonEl.classList.toggle('active', !analysis);
  analysisViewButtonEl.classList.toggle('active', analysis);
  if (analysis && changed) refreshAnalysisStatus();
  if (analysis && analysisData) requestAnimationFrame(renderAnalysis);
}

function pointFromArray(raw) {
  return Array.isArray(raw) && raw.length >= 3
    ? {x: Number(raw[0]), y: Number(raw[1]), z: Number(raw[2])}
    : null;
}

function analysisTimelineLength() {
  if (!analysisData) return 0;
  if (analysisKind === 'camera') return (analysisData.frames || []).length;
  if (analysisKind === 'trajectory') return (analysisData.points || []).length;
  return (analysisData.samples || []).length;
}

function hasAnalysisKindData(kind) {
  if (!analysisData) return false;
  if (kind === 'camera') return ((analysisData.frames || []).length > 0);
  if (kind === 'trajectory') return ((analysisData.points || []).length > 0);
  if (kind === 'sim2real') {
    const report = analysisData.simRealComparison || {};
    return ((report.rows || []).length > 0) || Object.keys(report.groups || {}).length > 0;
  }
  if (kind === 'ball' || kind === 'policyInput' || kind === 'policyOutput') {
    return ((analysisData.samples || []).length > 0);
  }
  return false;
}

function preferredAnalysisKind() {
  for (const kind of ['camera', 'trajectory', 'sim2real', 'ball', 'policyInput', 'policyOutput']) {
    if (hasAnalysisKindData(kind)) return kind;
  }
  return 'camera';
}

function setAnalysisIndex(next) {
  const n = analysisTimelineLength();
  analysisIndex = Math.max(0, Math.min(Math.max(0, n - 1), Math.round(next || 0)));
  analysisScrubEl.max = Math.max(0, n - 1);
  analysisScrubEl.value = analysisIndex;
  analysisScrubEl.disabled = n <= 1;
  analysisFrameLabelEl.textContent = n > 0 ? `${analysisIndex + 1} / ${n}` : '0 / 0';
}

function analysisPaneFor(kind, detail) {
  if (kind === 'camera') return 'camera';
  if (kind === 'trajectory') return 'trajectory';
  if (kind === 'ball' && detail === 'report') return 'report';
  if (kind === 'ball' || kind === 'policyInput' || kind === 'policyOutput' || kind === 'sim2real') return 'curve';
  return 'camera';
}

function setAnalysisPanes() {
  const pane = analysisPaneFor(analysisKind, analysisDetail);
  analysisCameraPaneEl.classList.toggle('active', pane === 'camera');
  analysisTrajectoryPaneEl.classList.toggle('active', pane === 'trajectory');
  analysisCurvePaneEl.classList.toggle('active', pane === 'curve');
  analysisReportPaneEl.classList.toggle('active', pane === 'report');
}

function stopAnalysisPlayback() {
  analysisPlaying = false;
  analysisPlayEl.textContent = 'Play';
  if (analysisPlayTimer !== null) {
    clearTimeout(analysisPlayTimer);
    analysisPlayTimer = null;
  }
}

function analysisPlaybackDelayMs() {
  const fps = Number((CONFIG.recording || {}).analysisVideoFps) || 30;
  return 1000 / Math.max(1, Math.min(120, fps));
}

function tickAnalysisPlayback() {
  if (!analysisPlaying) return;
  const n = analysisTimelineLength();
  if (n <= 1) {
    stopAnalysisPlayback();
    return;
  }
  setAnalysisIndex((analysisIndex + 1) % n);
  renderAnalysis();
  analysisPlayTimer = setTimeout(tickAnalysisPlayback, analysisPlaybackDelayMs());
}

function setAnalysisKind(kind, detail='overview', resetIndex=true) {
  analysisKind = kind;
  analysisDetail = detail;
  analysisKindEls.forEach(el => el.classList.toggle('active', el.dataset.kind === kind));
  buildAnalysisSubList();
  setAnalysisPanes();
  setAnalysisIndex(resetIndex ? 0 : analysisIndex);
  renderAnalysis();
}

function setAnalysisDetail(detail) {
  analysisDetail = detail || 'overview';
  buildAnalysisSubList();
  setAnalysisPanes();
  setAnalysisIndex(analysisIndex);
  renderAnalysis();
}

function buildAnalysisSubList() {
  const buttons = [];
  function add(label, detail) {
    buttons.push(`<button type="button" class="${analysisDetail === detail ? 'active' : ''}" data-detail="${detail}">${label}</button>`);
  }
  if (analysisKind === 'camera') {
    add('Recorded frames', 'overview');
  } else if (analysisKind === 'trajectory') {
    add('3D / top / side', 'overview');
  } else if (analysisKind === 'ball') {
    add('Overview', 'overview');
    for (const prefix of ['rawBall', 'measuredBall', 'observedBall', 'measuredVelocity', 'observedVelocity']) {
      for (const axis of ['x', 'y', 'z']) add(`${prefix}.${axis}`, `${prefix}.${axis}`);
    }
    add('age', 'age');
    for (const axis of ['x', 'y', 'z']) add(`noise residual ${axis}`, `residual.${axis}`);
    add('noise report', 'report');
  } else if (analysisKind === 'policyInput') {
    add('Overview heatmap', 'overview');
    const names = (analysisData && analysisData.obsNames) || [];
    names.forEach((name, idx) => add(`${idx}: ${name}`, `dim.${idx}`));
  } else if (analysisKind === 'policyOutput') {
    add('Overview', 'overview');
    const names = (analysisData && analysisData.actionNames) || [];
    names.forEach((name, idx) => add(`${idx}: ${name}`, `dim.${idx}`));
  } else if (analysisKind === 'sim2real') {
    add('Overview RMS', 'overview');
    const rows = ((analysisData && analysisData.simRealComparison && analysisData.simRealComparison.rows) || []);
    const groups = Array.from(new Set(rows.map(row => row.group).filter(Boolean)));
    groups.forEach(group => add(`${group} RMS`, `group.${group}`));
    rows.forEach(row => add(`${row.group}[${row.dim}] ${row.name || ''}`, `cmp.${row.group}.${row.dim}`));
  }
  analysisSubListEl.innerHTML = buttons.join('');
  for (const btn of Array.from(analysisSubListEl.querySelectorAll('button'))) {
    btn.addEventListener('click', () => {
      setAnalysisDetail(btn.dataset.detail || 'overview');
    });
  }
}

function renderAnalysis() {
  if (!analysisData) return;
  setAnalysisIndex(analysisIndex);
  if (analysisKind === 'camera') renderAnalysisCamera();
  else if (analysisKind === 'trajectory') renderAnalysisTrajectory();
  else if (analysisKind === 'ball') {
    if (analysisDetail === 'report') renderAnalysisReport();
    else renderAnalysisCurve();
  } else if (analysisKind === 'policyInput' || analysisKind === 'policyOutput' || analysisKind === 'sim2real') {
    renderAnalysisCurve();
  }
}

function renderAnalysisCamera() {
  const frames = (analysisData && analysisData.frames) || [];
  const frame = frames[analysisIndex] || null;
  if (frame && frame.filename) {
    analysisCameraFrameEl.style.display = 'block';
    analysisCameraEmptyEl.classList.add('hidden');
    analysisCameraFrameEl.src = analysisFrameUrl(frame.filename);
  } else {
    analysisCameraFrameEl.style.display = 'none';
    analysisCameraEmptyEl.classList.remove('hidden');
    analysisCameraFrameEl.removeAttribute('src');
  }
  analysisReadoutEl.innerHTML = [
    `view=camera frames`,
    `frames=${frames.length}`,
    frame && Number.isFinite(frame.wallTimeS) ? `wall=${fmt(frame.wallTimeS, 3)}s` : null,
    frame && frame.info ? `seq=${frame.info.seq || 0}` : null
  ].filter(Boolean).join('<br>');
}

function analysisLimits() {
  return (analysisData && analysisData.limits) || CONFIG.limits || {x0:-1,x1:1,y0:-1,y1:1,z0:0,z1:2};
}

function analysisBox(name) {
  return (analysisData && analysisData[name]) || CONFIG[name] || null;
}

function analysisProjectTop(p, canvas) {
  const rect = canvas.getBoundingClientRect();
  const l = analysisLimits();
  return {x: (p.x - l.x0) / (l.x1 - l.x0) * rect.width, y: rect.height - (p.y - l.y0) / (l.y1 - l.y0) * rect.height};
}

function analysisProjectSide(p, canvas) {
  const rect = canvas.getBoundingClientRect();
  const l = analysisLimits();
  return {x: (p.x - l.x0) / (l.x1 - l.x0) * rect.width, y: rect.height - (p.z - l.z0) / (l.z1 - l.z0) * rect.height};
}

function analysisProject3d(p, canvas) {
  const rect = canvas.getBoundingClientRect();
  const l = analysisLimits();
  const center = {x:(l.x0+l.x1)*0.5, y:(l.y0+l.y1)*0.5, z:(l.z0+l.z1)*0.5};
  let x = p.x - center.x, y = p.y - center.y, z = p.z - center.z;
  const yaw = analysisYaw, pitch = analysisPitch;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * x - sy * y;
  const y1 = sy * x + cy * y;
  const y2 = cp * y1 - sp * z;
  const radius = Math.max(l.x1-l.x0, l.y1-l.y0, l.z1-l.z0) * 0.58;
  const scale = Math.min(rect.width, rect.height) * 0.42 / Math.max(radius, 1e-6);
  return {x: rect.width * 0.52 + x1 * scale, y: rect.height * 0.54 - y2 * scale};
}

function drawAnalysisRect(ctx, canvas, box, project, plane, color, label) {
  if (!box) return;
  const a = plane === 'side'
    ? project({x: box.x0, y: 0, z: box.z0}, canvas)
    : project({x: box.x0, y: box.y0, z: box.z0}, canvas);
  const b = plane === 'side'
    ? project({x: box.x1, y: 0, z: box.z1}, canvas)
    : project({x: box.x1, y: box.y1, z: box.z1}, canvas);
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  const w = Math.abs(b.x - a.x), h = Math.abs(b.y - a.y);
  ctx.save();
  ctx.globalAlpha = 0.64;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 5]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  ctx.fillText(label, x + 8, y + 18);
  ctx.restore();
}

function drawAnalysisBox3d(ctx, canvas, box, color, label) {
  if (!box) return;
  const pts = [
    {x:box.x0,y:box.y0,z:box.z0},{x:box.x1,y:box.y0,z:box.z0},{x:box.x1,y:box.y1,z:box.z0},{x:box.x0,y:box.y1,z:box.z0},
    {x:box.x0,y:box.y0,z:box.z1},{x:box.x1,y:box.y0,z:box.z1},{x:box.x1,y:box.y1,z:box.z1},{x:box.x0,y:box.y1,z:box.z1}
  ];
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  ctx.save();
  ctx.globalAlpha = 0.58;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 5]);
  for (const [a, b] of edges) {
    const pa = analysisProject3d(pts[a], canvas), pb = analysisProject3d(pts[b], canvas);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.font = '12px sans-serif';
  const q = analysisProject3d(pts[4], canvas);
  ctx.fillText(label, q.x + 6, q.y - 6);
  ctx.restore();
}

function drawAnalysisWorkspace(ctx, canvas, project, plane) {
  const ballBox = analysisBox('ballBox');
  const eeBox = analysisBox('eeBox');
  if (plane === '3d') {
    drawAnalysisBox3d(ctx, canvas, ballBox, '#111827', 'done');
    drawAnalysisBox3d(ctx, canvas, eeBox, '#0891b2', 'EE');
  } else {
    drawAnalysisRect(ctx, canvas, ballBox, project, plane, '#111827', 'done');
    drawAnalysisRect(ctx, canvas, eeBox, project, plane, '#0891b2', 'EE');
  }
  const anchor = (analysisData && analysisData.anchor) || CONFIG.anchor;
  const p = pointFromArray(anchor);
  if (p) drawAnalysisDot(ctx, project(p, canvas), '#111827', 5);
}

function drawAnalysisDot(ctx, q, color, radius) {
  ctx.fillStyle = color;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(q.x, q.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function drawAnalysisTrail(ctx, canvas, points, key, project, color) {
  ctx.save();
  ctx.globalAlpha = 0.45;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  let moved = false;
  const start = Math.max(0, analysisIndex - 220);
  for (let i = start; i <= analysisIndex; i++) {
    const p = pointFromArray(points[i] && points[i][key]);
    if (!p) continue;
    const q = project(p, canvas);
    if (!moved) {
      ctx.moveTo(q.x, q.y);
      moved = true;
    } else {
      ctx.lineTo(q.x, q.y);
    }
  }
  ctx.stroke();
  ctx.restore();
}

function drawAnalysisTrajectoryCanvas(canvas, title, project, plane) {
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const points = (analysisData && analysisData.points) || [];
  drawAnalysisWorkspace(ctx, canvas, project, plane);
  drawAnalysisTrail(ctx, canvas, points, 'raw', project, '#f28e2b');
  drawAnalysisTrail(ctx, canvas, points, 'obs', project, '#e15759');
  drawAnalysisTrail(ctx, canvas, points, 'racket', project, '#4e79a7');
  drawAnalysisTrail(ctx, canvas, points, 'simRacket', project, '#0f766e');
  const row = points[analysisIndex] || {};
  for (const [key, color, radius] of [
    ['raw', '#f28e2b', 8], ['obs', '#e15759', 6], ['measured', '#f59e0b', 5],
    ['racket', '#4e79a7', 8], ['simRacket', '#0f766e', 7],
    ['joint', '#16a34a', 6], ['cart', '#9333ea', 6]
  ]) {
    const p = pointFromArray(row[key]);
    if (p) drawAnalysisDot(ctx, project(p, canvas), color, radius);
  }
  ctx.fillStyle = '#273445';
  ctx.font = '14px sans-serif';
  ctx.fillText(title, 14, 22);
}

function renderAnalysisTrajectory() {
  drawAnalysisTrajectoryCanvas(analysisReplay3dEl, '3D replay', analysisProject3d, '3d');
  drawAnalysisTrajectoryCanvas(analysisReplayTopEl, 'Top x / y', analysisProjectTop, 'top');
  drawAnalysisTrajectoryCanvas(analysisReplaySideEl, 'Side x / z', analysisProjectSide, 'side');
  const points = (analysisData && analysisData.points) || [];
  const row = points[analysisIndex] || {};
  analysisReadoutEl.innerHTML = [
    `trajectory samples=${points.length}`,
    `t=${Number(row.time || 0).toFixed(3)}s`,
    `mode=${row.mode || ''}`,
    `valid=${row.valid ? 'true' : 'false'}`,
    `view yaw=${fmt(analysisYaw, 2)} pitch=${fmt(analysisPitch, 2)}`
  ].join('<br>');
}

function sampleTimes() {
  return ((analysisData && analysisData.samples) || []).map((row, idx) => Number(row.time ?? idx * 0.005));
}

function seriesFromPoint(key, axis) {
  const axisIdx = {x:0, y:1, z:2}[axis] ?? 0;
  return ((analysisData && analysisData.samples) || []).map(row => {
    const v = row[key];
    return Array.isArray(v) && Number.isFinite(Number(v[axisIdx])) ? Number(v[axisIdx]) : NaN;
  });
}

function seriesFromVector(key, idx) {
  return ((analysisData && analysisData.samples) || []).map(row => {
    const v = row[key];
    return Array.isArray(v) && Number.isFinite(Number(v[idx])) ? Number(v[idx]) : NaN;
  });
}

function sim2realGroupConfig(group) {
  const names = (analysisData && analysisData.simRealComparison && analysisData.simRealComparison.rows) || [];
  const rowNames = names.filter(row => row.group === group).map(row => row.name || `${group}[${row.dim}]`);
  const configs = {
    policy_obs: {
      label: 'Policy Input',
      real: 'policyObs',
      sim: 'simPolicyObs',
      names: rowNames.length ? rowNames : ((analysisData && analysisData.obsNames) || []),
      y: 'obs value',
    },
    policy_action: {
      label: 'Policy Action',
      real: 'policyAction',
      sim: 'simPolicyAction',
      names: rowNames.length ? rowNames : ((analysisData && analysisData.actionNames) || []),
      y: 'action value',
    },
    joint_position: {
      label: 'Joint Position',
      real: 'rosJointPosition',
      sim: 'simJointPosition',
      names: rowNames,
      y: 'position (rad)',
    },
    joint_velocity: {
      label: 'Joint Velocity',
      real: 'rosJointVelocity',
      sim: 'simJointVelocity',
      names: rowNames,
      y: 'velocity (rad/s)',
    },
    command_position: {
      label: 'Command Position',
      real: 'commandPosition',
      sim: 'simPolicyCommand',
      names: rowNames,
      y: 'position (rad)',
    },
    racket_xyz: {
      label: 'Racket Position',
      real: 'racket',
      sim: 'simRacket',
      names: rowNames.length ? rowNames : ['ee_x', 'ee_y', 'ee_z'],
      point: true,
      y: 'position (m)',
    },
  };
  return configs[group] || null;
}

function sim2realDiffSeries(group, idx) {
  const cfg = sim2realGroupConfig(group);
  if (!cfg) return [];
  if (cfg.point) {
    const axis = ['x', 'y', 'z'][idx] || 'x';
    const real = seriesFromPoint(cfg.real, axis);
    const sim = seriesFromPoint(cfg.sim, axis);
    return real.map((v, i) => Number.isFinite(v) && Number.isFinite(sim[i]) ? v - sim[i] : NaN);
  }
  const real = seriesFromVector(cfg.real, idx);
  const sim = seriesFromVector(cfg.sim, idx);
  return real.map((v, i) => Number.isFinite(v) && Number.isFinite(sim[i]) ? v - sim[i] : NaN);
}

function sim2realRmsSeries(group) {
  const cfg = sim2realGroupConfig(group);
  if (!cfg) return [];
  const rows = (analysisData && analysisData.samples) || [];
  const dims = Math.max(0, ...rows.map(row => {
    if (cfg.point) return 3;
    return Math.min(
      Array.isArray(row[cfg.real]) ? row[cfg.real].length : 0,
      Array.isArray(row[cfg.sim]) ? row[cfg.sim].length : 0
    );
  }));
  const diffByDim = Array.from({length: dims}, (_, idx) => sim2realDiffSeries(group, idx));
  return rows.map((_, sampleIdx) => {
    const vals = [];
    for (let idx = 0; idx < dims; idx++) {
      const diff = diffByDim[idx][sampleIdx];
      if (Number.isFinite(diff)) vals.push(diff * diff);
    }
    return vals.length ? Math.sqrt(vals.reduce((a, b) => a + b, 0) / vals.length) : NaN;
  });
}

function axisTickLabel(v) {
  if (!Number.isFinite(v)) return '';
  const av = Math.abs(v);
  if (av >= 1000 || (av > 0 && av < 0.001)) return v.toExponential(1);
  const digits = av >= 100 ? 0 : (av >= 10 ? 1 : 3);
  return v.toFixed(digits).replace(/\.?0+$/, '');
}

function drawLineChart(canvas, title, seriesList, cursorIndex=analysisIndex, yLabel='value', xLabel='time (s)') {
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const pad = {l: 74, r: 22, t: 38, b: 58};
  const t = sampleTimes();
  let vals = [];
  for (const s of seriesList) vals = vals.concat(s.values.filter(Number.isFinite));
  if (!seriesList.length || !vals.length) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(title, 12, 22);
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No finite samples for this view', rect.width * 0.5, rect.height * 0.5);
    return;
  }
  const rawYmin = vals.length ? Math.min(...vals) : -1;
  const rawYmax = vals.length ? Math.max(...vals) : 1;
  const rawSpan = rawYmax - rawYmin;
  const flatPad = Math.max(1.0, Math.abs(rawYmax) * 0.1);
  const ymin = rawSpan < 1e-9 ? rawYmin - flatPad : rawYmin - rawSpan * 0.06;
  const ymax = rawSpan < 1e-9 ? rawYmax + flatPad : rawYmax + rawSpan * 0.06;
  const span = Math.max(1e-9, ymax - ymin);
  const x0 = t.length ? t[0] : 0;
  const x1 = t.length ? t[t.length - 1] : 1;
  const plotW = rect.width - pad.l - pad.r;
  const plotH = rect.height - pad.t - pad.b;
  function xAt(idx) {
    const tv = t[idx] ?? idx;
    return pad.l + (tv - x0) / Math.max(1e-9, x1 - x0) * plotW;
  }
  function yAt(v) {
    return rect.height - pad.b - (v - ymin) / span * plotH;
  }
  ctx.strokeStyle = '#d7dee8';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + i / 4 * plotH;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(rect.width - pad.r, y); ctx.stroke();
    const yv = ymax - i / 4 * span;
    ctx.fillStyle = '#64748b';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(axisTickLabel(yv), pad.l - 8, y + 4);
  }
  for (let i = 0; i <= 4; i++) {
    const x = pad.l + i / 4 * plotW;
    const xv = x0 + i / 4 * Math.max(0, x1 - x0);
    ctx.strokeStyle = '#eef2f7';
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, rect.height - pad.b); ctx.stroke();
    ctx.fillStyle = '#64748b';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(axisTickLabel(xv), x, rect.height - pad.b + 17);
  }
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.2;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, rect.height - pad.b); ctx.lineTo(rect.width - pad.r, rect.height - pad.b); ctx.stroke();
  ctx.fillStyle = '#475569';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, pad.l + plotW * 0.5, rect.height - 8);
  ctx.save();
  ctx.translate(17, pad.t + plotH * 0.5);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  for (const s of seriesList) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.5;
    ctx.beginPath();
    let moved = false;
    for (let i = 0; i < s.values.length; i++) {
      const v = s.values[i];
      if (!Number.isFinite(v)) continue;
      const x = xAt(i), y = yAt(v);
      if (!moved) { ctx.moveTo(x, y); moved = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  if (t.length) {
    const cx = xAt(Math.max(0, Math.min(t.length - 1, cursorIndex)));
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, pad.t); ctx.lineTo(cx, rect.height - pad.b); ctx.stroke();
  }
  ctx.fillStyle = '#273445';
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(title, 12, 22);
  ctx.font = '12px sans-serif';
  let lx = pad.l;
  for (const s of seriesList.slice(0, 10)) {
    ctx.fillStyle = s.color;
    ctx.fillText(s.name, lx, rect.height - 24);
    lx += Math.min(130, Math.max(62, s.name.length * 7));
  }
  ctx.textAlign = 'left';
}

function drawHeatmap(canvas, title, matrix, names, yLabel='dimension', xLabel='time (s)') {
  resizeCanvas(canvas);
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const rows = matrix.length;
  const dims = rows ? matrix[0].length : 0;
  const pad = {l: 74, r: 16, t: 38, b: 58};
  if (!rows || !dims) {
    ctx.fillStyle = '#64748b';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('no data', 14, 22);
    return;
  }
  const w = rect.width - pad.l - pad.r;
  const h = rect.height - pad.t - pad.b;
  const t = sampleTimes();
  const x0 = t.length ? t[0] : 0;
  const x1 = t.length ? t[Math.min(t.length - 1, rows - 1)] : rows - 1;
  const stats = [];
  for (let d = 0; d < dims; d++) {
    const vals = matrix.map(r => Number(r[d])).filter(Number.isFinite);
    const min = vals.length ? Math.min(...vals) : -1;
    const max = vals.length ? Math.max(...vals) : 1;
    stats.push({min, max, span: Math.max(1e-9, max - min)});
  }
  for (let d = 0; d < dims; d++) {
    for (let i = 0; i < rows; i++) {
      const v = Number(matrix[i][d]);
      if (!Number.isFinite(v)) continue;
      const z = Math.max(0, Math.min(1, (v - stats[d].min) / stats[d].span));
      const hue = 220 - z * 220;
      ctx.fillStyle = `hsl(${hue}, 72%, 48%)`;
      ctx.fillRect(pad.l + i / rows * w, pad.t + d / dims * h, Math.ceil(w / rows) + 1, Math.ceil(h / dims) + 1);
    }
  }
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1.2;
  ctx.strokeRect(pad.l, pad.t, w, h);
  if (rows > 1) {
    const cx = pad.l + Math.max(0, Math.min(rows - 1, analysisIndex)) / Math.max(1, rows - 1) * w;
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, pad.t);
    ctx.lineTo(cx, pad.t + h);
    ctx.stroke();
  }
  ctx.fillStyle = '#64748b';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const x = pad.l + i / 4 * w;
    const xv = x0 + i / 4 * Math.max(0, x1 - x0);
    ctx.fillText(axisTickLabel(xv), x, pad.t + h + 17);
  }
  ctx.textAlign = 'right';
  ctx.fillText('0', pad.l - 8, pad.t + 11);
  ctx.fillText(String(dims - 1), pad.l - 8, pad.t + h);
  ctx.fillStyle = '#475569';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, pad.l + w * 0.5, rect.height - 8);
  ctx.save();
  ctx.translate(17, pad.t + h * 0.5);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.fillStyle = '#273445';
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(title, 12, 22);
}

function renderAnalysisCurve() {
  const samples = (analysisData && analysisData.samples) || [];
  if (!samples.length) {
    drawLineChart(analysisCurveCanvasEl, 'Analysis', [], analysisIndex);
    analysisReadoutEl.innerHTML = [
      `view=${analysisKind}`,
      `detail=${analysisDetail}`,
      `samples=0`
    ].join('<br>');
    return;
  }
  if (analysisKind === 'sim2real') {
    const report = (analysisData && analysisData.simRealComparison) || {};
    const rows = report.rows || [];
    const colors = ['#0f766e','#4e79a7','#f28e2b','#e15759','#9333ea','#16a34a'];
    if (analysisDetail === 'overview') {
      const groups = Object.keys(report.groups || {});
      drawLineChart(analysisCurveCanvasEl, 'Sim2Real RMS Overview', groups.map((group, idx) => ({
        name: group,
        color: colors[idx % colors.length],
        values: sim2realRmsSeries(group)
      })), analysisIndex, 'RMS(real - sim)');
    } else if (analysisDetail.startsWith('group.')) {
      const group = analysisDetail.slice('group.'.length);
      const cfg = sim2realGroupConfig(group);
      drawLineChart(analysisCurveCanvasEl, `${group} RMS`, [{
        name: cfg ? cfg.label : group,
        color: '#0f766e',
        values: sim2realRmsSeries(group),
        width: 2,
      }], analysisIndex, 'RMS(real - sim)');
    } else if (analysisDetail.startsWith('cmp.')) {
      const parts = analysisDetail.split('.');
      const group = parts[1] || '';
      const idx = Number(parts[2] || 0);
      const cfg = sim2realGroupConfig(group);
      const meta = rows.find(row => row.group === group && Number(row.dim) === idx) || {};
      let realValues = [];
      let simValues = [];
      if (cfg && cfg.point) {
        const axis = ['x', 'y', 'z'][idx] || 'x';
        realValues = seriesFromPoint(cfg.real, axis);
        simValues = seriesFromPoint(cfg.sim, axis);
      } else if (cfg) {
        realValues = seriesFromVector(cfg.real, idx);
        simValues = seriesFromVector(cfg.sim, idx);
      }
      const diffValues = realValues.map((v, i) =>
        Number.isFinite(v) && Number.isFinite(simValues[i]) ? v - simValues[i] : NaN);
      drawLineChart(analysisCurveCanvasEl, `${group}[${idx}] ${meta.name || ''}`, [
        {name:'real', color:'#4e79a7', values:realValues, width:1.7},
        {name:'sim', color:'#0f766e', values:simValues, width:1.7},
        {name:'real-sim', color:'#e15759', values:diffValues, width:1.2},
      ], analysisIndex, cfg ? cfg.y : 'value');
    }
  } else if (analysisKind === 'policyInput') {
    if (analysisDetail === 'overview') {
      const real = samples.map(r => r.policyObs || []);
      const sim = samples.map(r => r.simPolicyObs || []);
      const hasSim = sim.some(row => row.length > 0);
      drawHeatmap(analysisCurveCanvasEl, hasSim ? 'Policy Input Overview (real; sim available in Sim2Real)' : 'Policy Input Overview (normalized per dimension)', real, analysisData.obsNames || [], 'obs dimension');
    } else {
      const idx = Number((analysisDetail.split('.')[1] || 0));
      const name = (analysisData.obsNames || [])[idx] || `obs[${idx}]`;
      const series = [{
        name: `real ${name}`,
        color: '#4e79a7',
        values: samples.map(r => Number((r.policyObs || [])[idx]))
      }];
      if (samples.some(r => (r.simPolicyObs || []).length > idx)) {
        series.push({
          name: `sim ${name}`,
          color: '#0f766e',
          values: samples.map(r => Number((r.simPolicyObs || [])[idx]))
        });
      }
      drawLineChart(analysisCurveCanvasEl, `Policy Input ${idx}: ${name}`, series, analysisIndex, 'obs value');
    }
  } else if (analysisKind === 'policyOutput') {
    if (analysisDetail === 'overview') {
      const colors = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1'];
      const width = Math.max(0, ...samples.map(r => (r.policyAction || []).length));
      drawLineChart(analysisCurveCanvasEl, 'Policy Action Overview', Array.from({length: width}, (_, idx) => ({
        name: (analysisData.actionNames || [])[idx] || `action[${idx}]`,
        color: colors[idx % colors.length],
        values: samples.map(r => Number((r.policyAction || [])[idx]))
      })), analysisIndex, 'action value');
    } else {
      const idx = Number((analysisDetail.split('.')[1] || 0));
      const name = (analysisData.actionNames || [])[idx] || `action[${idx}]`;
      const series = [{
        name: `real ${name}`,
        color: '#e15759',
        values: samples.map(r => Number((r.policyAction || [])[idx]))
      }];
      if (samples.some(r => (r.simPolicyAction || []).length > idx)) {
        series.push({
          name: `sim ${name}`,
          color: '#0f766e',
          values: samples.map(r => Number((r.simPolicyAction || [])[idx]))
        });
      }
      drawLineChart(analysisCurveCanvasEl, `Policy Action ${idx}`, series, analysisIndex, 'action value');
    }
  } else if (analysisKind === 'ball') {
    if (analysisDetail === 'overview') {
      drawLineChart(analysisCurveCanvasEl, 'Ball State Overview', [
        {name:'raw x', color:'#f28e2b', values:seriesFromPoint('rawBall','x')},
        {name:'raw y', color:'#f59e0b', values:seriesFromPoint('rawBall','y')},
        {name:'raw z', color:'#b45309', values:seriesFromPoint('rawBall','z')},
        {name:'obs x', color:'#e15759', values:seriesFromPoint('observedBall','x')},
        {name:'obs y', color:'#dc2626', values:seriesFromPoint('observedBall','y')},
        {name:'obs z', color:'#991b1b', values:seriesFromPoint('observedBall','z')},
      ], analysisIndex, 'position (m)');
    } else if (analysisDetail === 'age') {
      drawLineChart(analysisCurveCanvasEl, 'Ball Observation Age', [{name:'age', color:'#4e79a7', values:samples.map(r => Number(r.age))}], analysisIndex, 'age (s)');
    } else if (analysisDetail.startsWith('residual.')) {
      const axis = analysisDetail.split('.')[1] || 'x';
      const a = seriesFromPoint('measuredBall', axis);
      const b = seriesFromPoint('observedBall', axis);
      drawLineChart(analysisCurveCanvasEl, `Ball Noise Residual ${axis} (mm)`, [{
        name:`measured - observed ${axis}`,
        color:'#e15759',
        values:a.map((v, i) => (v - b[i]) * 1000.0)
      }], analysisIndex, 'residual (mm)');
    } else {
      const [key, axis] = analysisDetail.split('.');
      const yLabel = key && key.includes('Velocity') ? 'velocity (m/s)' : 'position (m)';
      drawLineChart(analysisCurveCanvasEl, `${key}.${axis}`, [{
        name:`${key}.${axis}`,
        color:'#4e79a7',
        values:seriesFromPoint(key, axis)
      }], analysisIndex, yLabel);
    }
  }
  analysisReadoutEl.innerHTML = [
    `view=${analysisKind}`,
    `detail=${analysisDetail}`,
    `samples=${samples.length}`
  ].join('<br>');
}

function renderAnalysisReport() {
  setAnalysisPanes();
  analysisNoiseReportEl.textContent = JSON.stringify((analysisData && analysisData.noiseReport) || {}, null, 2);
  analysisReadoutEl.innerHTML = [
    `view=${analysisKind}`,
    `detail=noise report`,
    analysisData && analysisData.metadata ? `samples=${analysisData.metadata.sampleCount || 0}` : null
  ].filter(Boolean).join('<br>');
}

async function loadAnalysisArtifacts(status) {
  const outputs = (status && status.outputs) || {};
  const key = `${status.analysisDir || ''}|${JSON.stringify(outputs)}|${status.phase || ''}`;
  if (key === analysisLoadedKey || key === analysisLoadingKey) return;
  analysisLoadingKey = key;
  let loaded = !outputs.analysisData;
  if (outputs.analysisData) {
    try {
      analysisData = await getJson(analysisFileUrl(outputs.analysisData));
      loaded = true;
      const nextKind = hasAnalysisKindData(analysisKind) ? analysisKind : preferredAnalysisKind();
      setAnalysisKind(nextKind, 'overview', true);
    } catch (err) {
      analysisReadoutEl.textContent = `analysis data load failed: ${String(err && err.message ? err.message : err)}`;
    }
  }
  if (loaded) analysisLoadedKey = key;
  if (analysisLoadingKey === key) analysisLoadingKey = '';
  analysisLinksEl.innerHTML = Object.entries(outputs).map(([k, v]) =>
    `<div><a href="${analysisFileUrl(v)}" target="_blank">${k}</a></div>`
  ).join('');
}

function updateAnalysisUi(status) {
  status = status || {};
  analysisStatus = status;
  const outputs = status.outputs || {};
  const active = !!status.active;
  const hasOutputs = Object.keys(outputs).length > 0;
  analysisViewButtonEl.disabled = !active && !hasOutputs && !status.analysisDir;
  if (active || !analysisData || !hasOutputs) {
    analysisReadoutEl.innerHTML = [
      `analysis: ${active ? 'running' : (status.phase === 'done' ? 'done' : (status.phase || 'idle'))}`,
      status.analysisDir ? `dir=${status.analysisDir}` : null,
      status.error ? `error=${status.error}` : null,
      hasOutputs ? `outputs=${Object.keys(outputs).join(', ')}` : null
    ].filter(Boolean).join('<br>');
  }
  if (hasOutputs) loadAnalysisArtifacts(status);
  if (active && analysisPollTimer === null) {
    analysisPollTimer = setInterval(refreshAnalysisStatus, 1000);
  } else if (!active && analysisPollTimer !== null) {
    clearInterval(analysisPollTimer);
    analysisPollTimer = null;
  }
}

async function refreshAnalysisStatus() {
  try {
    const status = await getJson('/analysis/status');
    updateAnalysisUi(status);
    if (status.phase === 'done' && status.outputs && currentView !== 'analysis') {
      setAppView('analysis');
    }
  } catch (err) {
    updateAnalysisUi({
      active: false,
      phase: 'error',
      outputs: {},
      error: String(err && err.message ? err.message : err)
    });
  }
}

async function step() {
  if (inFlight) return;
  inFlight = true;
  const now = performance.now();
  const dt = Math.max(0.001, Math.min(0.25, (now - lastSentAt) / 1000));
  const velocity = metersPerSecond(ball, lastSentBall, dt);
  lastSentAt = now;
  lastSentBall = {...ball};
  const estimatorDefaults = CONFIG.ballEstimator || {};
  const releaseDefaults = CONFIG.releaseGate || {};
  const compensationDefaults = CONFIG.compensation || {};
  const donePredictionDefaults = CONFIG.donePrediction || {};
  try {
    const payload = {
      ball: [ball.x, ball.y, ball.z],
      velocity,
      valid: visibleEl.checked,
      noiseStdM: Math.max(0, Number(inputs.noise.value) || 0) * 0.001,
      ballObservationRateHz: Math.max(1, Math.min(240, Number(inputs.obsHz.value) || 60)),
      ballSource: ballSourceEl.value,
      useRosState: (useRosStateEl.checked || (publishCommandsEl.checked && !publishCommandsEl.disabled)) && !useRosStateEl.disabled,
      publishCommands: publishCommandsEl.checked && !publishCommandsEl.disabled,
      workspaceSafetyMode: inputs.workspaceSafetyMode.value || 'position_soft',
      donePredictionHorizonS: readNumber(
        inputs.donePredictionHorizonS, donePredictionDefaults.horizonS ?? 0.12, 0, 0.5),
      donePredictionDtS: readNumber(
        inputs.donePredictionDtS, donePredictionDefaults.dtS ?? 0.01, 0.001, 0.1),
      donePredictionMarginM: readNumber(
        inputs.donePredictionMarginM, donePredictionDefaults.marginM ?? 0.02, 0, 0.15),
      donePredictionOrientationMarginRad: readNumber(
        inputs.donePredictionOrientationMarginRad,
        donePredictionDefaults.orientationMarginRad ?? 0.035,
        0,
        0.5),
      enableCompensation: flowEls.enableCompensation.checked && !flowEls.enableCompensation.disabled,
      enableSimMirror: flowEls.enableSimMirror.checked && !flowEls.enableSimMirror.disabled,
      compensationMode: inputs.compensationMode.value || compensationDefaults.mode || 'inverse_mpc',
      compDelayMs: readNumber(
        inputs.compDelayMs, compensationDefaults.delayMs ?? 72, 0, 250),
      compActuatorTauS: readNumber(
        inputs.compActuatorTauS, compensationDefaults.tauS ?? 0.074, 0, 0.5),
      compActuatorGain: readNumber(
        inputs.compActuatorGain, compensationDefaults.gain ?? 1.0, 0.1, 3),
      compMpcBeta: readNumber(
        inputs.compMpcBeta, compensationDefaults.mpcBeta ?? 1.2, 0, 5),
      compMpcDelayScale: readNumber(
        inputs.compMpcDelayScale, compensationDefaults.mpcDelayScale ?? 1.05, 0, 3),
      compMpcTauScale: readNumber(
        inputs.compMpcTauScale, compensationDefaults.mpcTauScale ?? 0.75, 0, 3),
      compMpcHorizonSteps: readInt(
        inputs.compMpcHorizonSteps, compensationDefaults.mpcHorizonSteps ?? 6, 1, 32),
      compMpcTrackingWeight: readNumber(
        inputs.compMpcTrackingWeight, compensationDefaults.mpcTrackingWeight ?? 1.0, 0, 10),
      compMpcNominalWeight: readNumber(
        inputs.compMpcNominalWeight, compensationDefaults.mpcNominalWeight ?? 0.25, 0, 10),
      compMpcDeltaWeight: readNumber(
        inputs.compMpcDeltaWeight, compensationDefaults.mpcDeltaWeight ?? 0.05, 0, 10),
      compMpcMaxDeltaRad: readNumber(
        inputs.compMpcMaxDeltaRad, compensationDefaults.mpcMaxDeltaRad ?? 0.5236, 0, 2),
      rlUseValidBallObsAge: flowEls.rlUseValidBallObsAge.checked,
      useBallEstimator: flowEls.useBallEstimator.checked,
      estimatorPrediction: flowEls.estimatorPrediction.checked,
      estimatorFrameHold: flowEls.estimatorFrameHold.checked,
      estimatorAeroDrag: flowEls.estimatorAeroDrag.checked,
      ballEstimatorReleaseGatedDynamics: flowEls.estimatorReleaseGatedDynamics.checked,
      ballEstimatorMaxExtrapolationS: readNumber(
        inputs.estimatorMaxExtrapolationS, estimatorDefaults.maxExtrapolationS || 0.45, 0, 2),
      ballEstimatorControlMaxExtrapolationS: readNumber(
        inputs.estimatorControlMaxExtrapolationS, estimatorDefaults.controlMaxExtrapolationS ?? 0.08, -1, 2),
      ballEstimatorFrameHoldOutputRateHz: readNumber(
        inputs.estimatorFrameHoldHz, estimatorDefaults.frameHoldOutputRateHz || 50, 1, 240),
      ballEstimatorTrustLossS: readNumber(
        inputs.estimatorTrustLossS, estimatorDefaults.trustLossS ?? 0.10, 0, 2),
      ballEstimatorDegradeLossS: readNumber(
        inputs.estimatorDegradeLossS, estimatorDefaults.degradeLossS ?? 0.40, 0, 3),
      ballEstimatorInvalidAfterS: readNumber(
        inputs.estimatorInvalidAfterS, estimatorDefaults.invalidAfterS ?? 0.60, 0, 5),
      ballEstimatorPredictionHorizonS: readNumber(
        inputs.estimatorPredictionHorizonS, estimatorDefaults.predictionHorizonS ?? 0.0, 0, 1),
      ballEstimatorQPosMm2S: readNumber(
        inputs.estimatorQPosMm2S, estimatorDefaults.qPosMm2S ?? 500.0, 0, 1000000),
      ballEstimatorQVelMult: readNumber(
        inputs.estimatorQVelMult, estimatorDefaults.qVelMult ?? 100.0, 0, 100000),
      ballEstimatorQDepthPosMult: readNumber(
        inputs.estimatorQDepthPosMult, estimatorDefaults.qDepthPosMult ?? 0.5, 0.000001, 100),
      ballEstimatorQDepthVelMult: readNumber(
        inputs.estimatorQDepthVelMult, estimatorDefaults.qDepthVelMult ?? 0.2, 0.000001, 100),
      ballEstimatorRLateralMm: readNumber(
        inputs.estimatorRLateralMm, estimatorDefaults.rLateralMm ?? 20.0, 1, 500),
      ballEstimatorRDepthMm: readNumber(
        inputs.estimatorRDepthMm, estimatorDefaults.rDepthMm ?? 60.0, 1, 1000),
      enableCaptureGate: flowEls.enableCaptureGate.checked,
      enableReleaseGate: flowEls.enableReleaseGate.checked,
      enableLostFailsafe: flowEls.enableLostFailsafe.checked,
      enableDoneMonitoring: flowEls.enableDoneMonitoring.checked,
      ballReleaseUseEstimatorState: flowEls.releaseUseEstimatorState.checked,
      ballReleaseMinDownwardVelocityMS: readNumber(
        inputs.releaseMinVz, releaseDefaults.minDownwardVelocityMS || 0.0, 0, 5),
      ballReleaseMinZDropM: readNumber(
        inputs.releaseMinDrop, releaseDefaults.minZDropM || 0.0, 0, 1),
      ballReleaseWindowS: readNumber(
        inputs.releaseWindowS, releaseDefaults.windowS || 0.12, 0.001, 2),
      ballReleaseRequiredTicks: readInt(
        inputs.releaseTicks, releaseDefaults.requiredTicks || 1, 1, 200),
      ballReleaseMaxObsAgeS: readNumber(
        inputs.releaseMaxAgeS, releaseDefaults.maxObsAgeS ?? 0.08, 0, 2),
      cameraDetect: cameraFlowEls.detect.checked && !cameraFlowEls.detect.disabled,
      cameraVisualize: cameraFlowEls.visualize.checked && !cameraFlowEls.visualize.disabled,
      cameraSensorDepth: cameraFlowEls.sensorDepth.checked && !cameraFlowEls.sensorDepth.disabled,
      cameraUndistort: cameraFlowEls.undistort.checked && !cameraFlowEls.undistort.disabled,
      cameraKalman: cameraFlowEls.kalman.checked && !cameraFlowEls.kalman.disabled,
      cameraEllipseFit: cameraFlowEls.ellipseFit.checked && !cameraFlowEls.ellipseFit.disabled,
      cameraTemporalTracker: cameraFlowEls.temporalTracker.checked && !cameraFlowEls.temporalTracker.disabled,
      cameraColorFilter: cameraFlowEls.colorFilter.checked && !cameraFlowEls.colorFilter.disabled,
      dt
    };
    if (recordingActive) {
      payload.recordingView = buildRecordingView('sample', false);
    }
    latest = await postJson('/step', payload);
    if (latest.observedBall) {
      observedBall = {x: latest.observedBall[0], y: latest.observedBall[1], z: latest.observedBall[2]};
    } else {
      observedBall = {...ball};
    }
    if (latest.ballSource === 'camera' && latest.rawBall) {
      ball = {x: latest.rawBall[0], y: latest.rawBall[1], z: latest.rawBall[2]};
      setInputsFromBall();
    }
    racket = {x: latest.racket[0], y: latest.racket[1], z: latest.racket[2]};
    const ros = latest.ros || {};
    const camera = latest.camera || {};
    const sim = latest.sim || {};
    updateRecordingUi(latest.recording || {});
    updateVisionFrame(camera);
    updateSimFrame(sim);
    jointEe = ros.jointEe ? {x: ros.jointEe[0], y: ros.jointEe[1], z: ros.jointEe[2]} : jointEe;
    cartEe = ros.cartPosition ? {x: ros.cartPosition[0], y: ros.cartPosition[1], z: ros.cartPosition[2]} : cartEe;
    const simSync = sim.sync || {};
    const simSourceSeq = Number.isFinite(simSync.sourceSeq)
      ? Number(simSync.sourceSeq) : -1;
    const nextSimEe = pointFromArray(sim.racket);
    const simAdvanced = !!nextSimEe && simSourceSeq !== lastSimSourceSeq;
    if (nextSimEe) simEe = nextSimEe;
    trailBall.push({...ball});
    trailObservedBall.push({...observedBall});
    trailRacket.push({...racket});
    if (simEe && simAdvanced) {
      trailSimEe.push({...simEe});
      lastSimSourceSeq = simSourceSeq;
    }
    if (jointEe) trailJointEe.push({...jointEe});
    if (cartEe) trailCartEe.push({...cartEe});
    while (trailBall.length > 320) trailBall.shift();
    while (trailObservedBall.length > 320) trailObservedBall.shift();
    while (trailRacket.length > 320) trailRacket.shift();
    while (trailSimEe.length > 320) trailSimEe.shift();
    while (trailJointEe.length > 320) trailJointEe.shift();
    while (trailCartEe.length > 320) trailCartEe.shift();
    const cmdErr = Number.isFinite(ros.jointCommandErrorAbsMax) ? ros.jointCommandErrorAbsMax : 0;
    plotHistory.push({action: latest.actionAbsMax || 0, err: cmdErr});
    while (plotHistory.length > 180) plotHistory.shift();
    modeEl.textContent = latest.reason ? `${latest.mode}: ${latest.reason}` : latest.mode;
    modeEl.className = latest.mode.includes('failsafe') ? 'badge stop' :
      ((latest.mode === 'capture_gate' || latest.mode === 'release_gate') ? 'badge warn' : 'badge');
    rosEl.textContent = ros.enabled
      ? (ros.useStateAllowed
        ? `ros ${ros.useState ? 'state' : 'local'}/${ros.publishCommands ? 'pub' : 'dry'} #${ros.publishCount || 0}`
        : (camera.enabled ? `ros camera ${camera.detected ? 'det' : 'wait'}` : 'ros node'))
      : 'ros=off';
    rosEl.className = ros.enabled ? (ros.publishCommands ? 'badge warn' : 'badge') : 'badge warn';
    timeEl.textContent = `t=${fmt(latest.time)}s`;
    ageEl.textContent = `age=${fmt(latest.age)}s`;
    const conds = Object.entries(latest.conditions || {}).filter(([, v]) => v).map(([k]) => k).join(', ') || 'none';
    const jointAge = Number.isFinite(ros.jointAgeS) ? `${fmt(ros.jointAgeS)}s` : 'none';
    const cartAge = Number.isFinite(ros.cartAgeS) ? `${fmt(ros.cartAgeS)}s` : 'none';
    const est = latest.estimator || {};
    const camRt = camera.runtime || {};
    const imageAges = camera.imageAgesS || {};
    const imageAgeBits = Object.entries(imageAges)
      .map(([k, v]) => `${k}=${Number.isFinite(v) ? fmt(v) : 'none'}s`)
      .join(' ');
    const cameraSource = camera.sourceMode || 'ros_topic';
    const sensorDepthOn = camRt.sensorDepth !== false;
    const cameraPath = cameraSource === 'direct_realsense'
      ? `direct_realsense ${camera.imageWidth || '?'}x${camera.imageHeight || '?'}@${camera.realsenseFps || '?'} hybrid=${camera.useHybridDetector ? 'on' : 'off'} sensorDepth=${sensorDepthOn ? 'on' : 'off'} active=${camera.activeTransport || 'none'} ${imageAgeBits}`
      : `topics transport=${camera.imageTransport || 'none'} active=${camera.activeTransport || 'none'} compressed=${camera.imageTopic || 'none'} raw=${camera.rawImageTopic || 'none'} ${imageAgeBits}`;
    const measured = latest.measuredBall || [NaN, NaN, NaN];
    const commandRacket = latest.commandRacket || null;
    const racketSource = latest.racketSource || 'command';
    const camDetectorBall = camera.detectorBall || null;
    const camControlBall = camera.controlBall || null;
    const releaseGate = latest.releaseGate || {};
    const comp = latest.compensation || {};
    const runtime = latest.runtime || {};
    const simClock = sim.simClock || {};
    const simSyncStatus = sim.sync || {};
    const backendLoop = latest.backendControlLoop || {};
    const recording = latest.recording || {};
    const recordingCounts = recording.counts || {};
    const wouldDone = latest.wouldDone || {};
    const doneBall = latest.doneBall || {};
    const hardStop = latest.hardWorkspaceStop || {};
    const resetRecovery = latest.resetRecovery || {};
    const doneBallPos = doneBall.position || [NaN, NaN, NaN];
    const wouldDoneActive = Object.entries(wouldDone.conditions || {})
      .filter(([, v]) => v).map(([k]) => k).join(', ') || 'none';
    const estBits = est.enabled
      ? `${est.state || 'unknown'} ${est.predicted ? 'pred' : 'meas'} predict=${est.prediction ? 'on' : 'off'}${latest.estimatorPredictionPending ? '(PENDING)' : ''} mode=${est.outputMode || 'unknown'} dyn=${est.ballisticDynamics === false ? 'held-cv' : 'ballistic'} rgdyn=${est.releaseGatedDynamics ? 'on' : 'off'} ctrl=${est.controlValid === false ? `reject:${est.controlRejectReason || 'stale'}` : 'valid'} loss=${Number.isFinite(est.lossS) ? fmt(est.lossS) : 'none'}s extrap=${Number.isFinite(est.extrapolationS) ? fmt(est.extrapolationS) : 'none'}s cap=${Number.isFinite(est.controlMaxExtrapolationS) ? fmt(est.controlMaxExtrapolationS) : 'none'}s hold=${Number.isFinite(est.snapshotHoldS) ? fmt(est.snapshotHoldS) : 'none'}s acc=${est.updatesAccepted || 0} rej=${est.updatesRejected || 0}`
      : 'off';
    const releaseBits = releaseGate.enabled
      ? `${releaseGate.released ? 'cleared' : 'holding'} src=${runtime.releaseUseEstimatorState ? 'estimator' : 'measurement'} valid=${releaseGate.valid ? 'true' : 'false'} fresh=${releaseGate.fresh ? 'true' : 'false'} vz=${Number.isFinite(releaseGate.vzMS) ? fmt(releaseGate.vzMS) : 'none'} z_drop=${Number.isFinite(releaseGate.zDropM) ? fmt(releaseGate.zDropM) : 'none'} ticks=${releaseGate.candidateTicks || 0}/${releaseGate.requiredTicks || 0}`
      : 'off';
	    readoutEl.innerHTML = [
	      `runtime: ${ros.enabled ? `${ros.useState ? 'ros-state' : 'local-state'} / ${ros.publishCommands ? 'publish' : 'dry-run'}` : 'local only'} proc=${CONFIG.controlSubprocess ? 'subproc' : 'inline'} ui=${fmt(STEP_RATE_HZ, 0)}Hz display=${fmt(DISPLAY_RATE_HZ, 0)}Hz`,
	      `compensation: ${comp.enabled ? `${comp.active ? 'active' : 'inactive'} mode=${comp.mode || 'unknown'} delay=${Number.isFinite(comp.delayMs) ? fmt(comp.delayMs, 1) : 'none'}ms tau=${Number.isFinite(comp.tauS) ? fmt(comp.tauS, 3) : 'none'}s beta=${Number.isFinite(comp.mpcBeta) ? fmt(comp.mpcBeta, 2) : 'none'} maxd=${Number.isFinite(comp.mpcMaxDeltaRad) ? fmt(comp.mpcMaxDeltaRad, 3) : 'none'}${comp.error ? ` error=${comp.error}` : ''}` : 'off'}`,
	      `backend loop: ${backendLoop.enabled ? `on ${backendLoop.source || 'thread'} target=${fmt(backendLoop.targetHz, 0)}Hz measured=${fmt(backendLoop.measuredHz, 1)}Hz snap=${fmt(backendLoop.snapshotHz, 0)}Hz work=${fmt(backendLoop.workMs, 2)}ms over=${backendLoop.overruns || 0} tick=${backendLoop.tick || 0}` : 'off'}`,
      `send command: selected=${publishCommandsEl.checked && !publishCommandsEl.disabled ? 'on' : 'off'} applied=${runtime.publishCommands === true ? 'on' : 'off'}${latest.commandPublishPending ? ' (PENDING)' : ''}`,
      `hard stop: ${hardStop.active ? `ACTIVE reason=${hardStop.reason || latest.reason || 'done'} hold=${hardStop.holdSent ? 'sent' : 'pending'} reset-required` : 'off'}`,
      `reset recovery: ${resetRecovery.active ? `ACTIVE done-bypass remaining=${fmt(resetRecovery.remainingS || 0, 1)}s` : 'off'}`,
      backendLoop.error ? `backend error: ${backendLoop.error}` : null,
      `recording: ${recording.active ? `on ctrl=${recordingCounts.control || 0} snap=${recordingCounts.snapshots || 0} view=${recordingCounts.views || 0} img=${recordingCounts.frames || 0} simimg=${recordingCounts.simFrames || 0} drop=${recording.dropped || 0}` : 'off'}${recording.runDir ? ` dir=${recording.runDir}` : ''}`,
      recording.error ? `recording error: ${recording.error}` : null,
      `workspace safety: selected=${inputs.workspaceSafetyMode.value || 'position_soft'} applied=${runtime.workspaceSafetyMode || 'waiting'}${latest.workspaceModePending ? ' (PENDING)' : ''}`,
      `predictive: horizon=${Number.isFinite(runtime.donePredictionHorizonS) ? fmt(runtime.donePredictionHorizonS, 3) : fmt(readNumber(inputs.donePredictionHorizonS, 0.12, 0, 0.5), 3)}s dt=${Number.isFinite(runtime.donePredictionDtS) ? fmt(runtime.donePredictionDtS, 3) : fmt(readNumber(inputs.donePredictionDtS, 0.01, 0.001, 0.1), 3)}s margin=${Number.isFinite(runtime.donePredictionMarginM) ? fmt(runtime.donePredictionMarginM, 3) : fmt(readNumber(inputs.donePredictionMarginM, 0.02, 0, 0.15), 3)}m ori=${Number.isFinite(runtime.donePredictionOrientationMarginRad) ? fmt(runtime.donePredictionOrientationMarginRad, 3) : fmt(readNumber(inputs.donePredictionOrientationMarginRad, 0.035, 0, 0.5), 3)}rad`,
      `ball source: ${latest.ballSource || ballSourceEl.value}`,
      `camera: ${camera.enabled ? `${cameraSource} / ${camera.cameraInfoReceived ? 'info' : 'no-info'} / ${camera.detectorReady ? 'detector' : 'no-detector'} / ${camera.detected ? 'detected' : 'lost'} age=${Number.isFinite(camera.ageS) ? fmt(camera.ageS) : 'none'}s hz=${Number.isFinite(camera.callbackHz) ? fmt(camera.callbackHz, 1) : '0.0'} frame=${camera.frame || 'none'}` : 'off'}`,
      camera.enabled ? `camera path: ${cameraPath}` : null,
      camera.enabled ? `camera flow: detect=${camRt.detect ? 'on' : 'off'} overlay=${camRt.visualize ? 'on' : 'off'} sensorDepth=${sensorDepthOn ? 'on' : 'off'} undistort=${camRt.undistort ? 'on' : 'off'} kalman=${camRt.kalman ? 'on' : 'off'} ellipse=${camRt.ellipseFit ? 'on' : 'off'} tracker=${camRt.temporalTracker ? 'on' : 'off'} color=${camRt.colorFilter ? 'on' : 'off'} frames=${camera.frameCount || 0} det=${camera.detectionCount || 0} vizHz=${Number.isFinite(camera.visualizationRateHz) ? fmt(camera.visualizationRateHz, 0) : '60'} web=${camera.webMaxWidth || 'full'}px/q${camera.webJpegQuality || 62}` : null,
      camDetectorBall ? `camera detector ${camera.detectorFrame || camera.frame || 'camera'}: (${fmt(camDetectorBall[0])}, ${fmt(camDetectorBall[1])}, ${fmt(camDetectorBall[2])})` : null,
      camControlBall ? `camera control sim: (${fmt(camControlBall[0])}, ${fmt(camControlBall[1])}, ${fmt(camControlBall[2])})` : null,
      camera.error ? `camera error: ${camera.error}` : null,
      `raw ball: (${fmt(ball.x)}, ${fmt(ball.y)}, ${fmt(ball.z)})`,
      `measured: (${fmt(measured[0])}, ${fmt(measured[1])}, ${fmt(measured[2])}) raw_valid=${latest.rawValid ? 'true' : 'false'}`,
      `control obs: (${fmt(observedBall.x)}, ${fmt(observedBall.y)}, ${fmt(observedBall.z)}) valid=${latest.valid ? 'true' : 'false'}`,
      `would done now: ${wouldDone.done ? `YES ${wouldDone.reason || ''}` : 'no'} active=${wouldDoneActive}`,
      `done ball [${doneBall.source || 'unknown'}]: (${fmt(doneBallPos[0])}, ${fmt(doneBallPos[1])}, ${fmt(doneBallPos[2])}) valid=${doneBall.valid ? 'true' : 'false'} streak=${JSON.stringify(doneBall.streaks || {})}/${doneBall.requiredTicks || '?'}`,
      `release gate: ${releaseBits}`,
      `estimator: ${estBits}`,
      `robot EE [${racketSource}]: (${fmt(racket.x)}, ${fmt(racket.y)}, ${fmt(racket.z)})`,
      sim.enabled ? `sim: ${sim.active ? 'on' : 'idle'} backend=${sim.backend || 'none'} sync=${simSyncStatus.alignmentValid ? 'valid' : 'NOT-valid'} produced=${Number.isFinite(simSyncStatus.producedSourceSeq) ? simSyncStatus.producedSourceSeq : 'none'} processed=${Number.isFinite(simSyncStatus.sourceSeq) ? simSyncStatus.sourceSeq : 'none'} lag=${Number.isFinite(simSyncStatus.lagTicks) ? simSyncStatus.lagTicks : 0}tick/${Number.isFinite(simSyncStatus.lagS) ? fmt(simSyncStatus.lagS, 3) : 'none'}s hz=${Number.isFinite(simClock.stepHz) ? fmt(simClock.stepHz, 1) : '0.0'} drop=${Number.isFinite(simSyncStatus.commandDrops) ? simSyncStatus.commandDrops : 0}+${Number.isFinite(simSyncStatus.resultDrops) ? simSyncStatus.resultDrops : 0} queued=${Number.isFinite(simClock.queued) ? simClock.queued : 0} batch=${Number.isFinite(simClock.steps) ? simClock.steps : 0}${sim.holdReason ? ` hold=${sim.holdReason}` : ''}${sim.error ? ` error=${sim.error}` : ''}` : 'sim: off',
      simEe ? `sim EE: (${fmt(simEe.x)}, ${fmt(simEe.y)}, ${fmt(simEe.z)})` : null,
      commandRacket && racketSource !== 'command' ? `cmd EE: (${fmt(commandRacket[0])}, ${fmt(commandRacket[1])}, ${fmt(commandRacket[2])})` : null,
      jointEe ? `joint FK: (${fmt(jointEe.x)}, ${fmt(jointEe.y)}, ${fmt(jointEe.z)}) age=${jointAge}` : `joint FK: none age=${jointAge}`,
      cartEe ? `cart: (${fmt(cartEe.x)}, ${fmt(cartEe.y)}, ${fmt(cartEe.z)}) age=${cartAge}` : `cart: none age=${cartAge}`,
      `cmd-joint err max: ${Number.isFinite(cmdErr) ? fmt(cmdErr) : 'none'}`,
      `noise: ${fmt((latest.noiseStdM || 0) * 1000, 1)} mm`,
      `action max: ${fmt(latest.actionAbsMax)}`,
      `done hits: ${conds}`,
      `fk: ${latest.fkOk ? 'ok' : 'off'}`
    ].filter(Boolean).join('<br>');
  } catch (err) {
    modeEl.textContent = 'server error';
    modeEl.className = 'badge stop';
  } finally {
    inFlight = false;
  }
}

document.getElementById('reset').addEventListener('click', async () => {
  const data = await postJson('/reset', {});
  latest = data;
  ball = {x: CONFIG.initialBall[0], y: CONFIG.initialBall[1], z: CONFIG.initialBall[2]};
  observedBall = {...ball};
  simEe = null;
  jointEe = null;
  cartEe = null;
  lastSentBall = {...ball};
  trailBall = [];
  trailObservedBall = [];
  trailRacket = [];
  trailSimEe = [];
  trailJointEe = [];
  trailCartEe = [];
  plotHistory = [];
  lastVisionSeq = 0;
  lastSimSourceSeq = -1;
  visibleEl.checked = true;
  setInputsFromBall();
  drawAll();
});

recordingButtonEl.addEventListener('click', async () => {
  if (recordingBusy) return;
  recordingBusy = true;
  updateRecordingUi({active: recordingActive, runDir: recordingRunDir});
  try {
    const wasRecording = recordingActive;
    const path = wasRecording ? '/recording/stop' : '/recording/start';
    const kind = wasRecording ? 'stop' : 'start';
    const status = await postJson(path, {
      view: buildRecordingView(kind, true),
      analyze: wasRecording && analyzeAfterRecordEl.checked
    });
    updateRecordingUi(status);
    if (wasRecording && analyzeAfterRecordEl.checked) {
      setAppView('analysis');
      refreshAnalysisStatus();
    }
  } catch (err) {
    updateRecordingUi({
      active: recordingActive,
      runDir: recordingRunDir,
      error: String(err && err.message ? err.message : err)
    });
  } finally {
    recordingBusy = false;
    await refreshRecordingStatus();
  }
});

liveViewButtonEl.addEventListener('click', () => setAppView('live'));
analysisViewButtonEl.addEventListener('click', () => setAppView('analysis'));
analysisRefreshEl.addEventListener('click', refreshAnalysisStatus);
for (const el of analysisKindEls) {
  el.addEventListener('click', () => {
    stopAnalysisPlayback();
    setAnalysisKind(el.dataset.kind || 'camera', 'overview', true);
  });
}
analysisScrubEl.addEventListener('input', () => {
  stopAnalysisPlayback();
  setAnalysisIndex(Number(analysisScrubEl.value) || 0);
  renderAnalysis();
});
analysisPlayEl.addEventListener('click', () => {
  if (!analysisData || analysisTimelineLength() <= 1) return;
  analysisPlaying = !analysisPlaying;
  analysisPlayEl.textContent = analysisPlaying ? 'Pause' : 'Play';
  if (analysisPlaying) tickAnalysisPlayback();
  else stopAnalysisPlayback();
});
analysisReplay3dEl.addEventListener('pointerdown', (ev) => {
  if (analysisKind !== 'trajectory') return;
  stopAnalysisPlayback();
  analysisDrag = {
    pointerId: ev.pointerId,
    x: ev.clientX,
    y: ev.clientY,
    yaw: analysisYaw,
    pitch: analysisPitch
  };
  analysisReplay3dEl.classList.add('dragging');
  analysisReplay3dEl.setPointerCapture(ev.pointerId);
  ev.preventDefault();
});
analysisReplay3dEl.addEventListener('pointermove', (ev) => {
  if (!analysisDrag || analysisDrag.pointerId !== ev.pointerId) return;
  const dx = ev.clientX - analysisDrag.x;
  const dy = ev.clientY - analysisDrag.y;
  analysisYaw = analysisDrag.yaw + dx * 0.008;
  analysisPitch = clamp(analysisDrag.pitch + dy * 0.006, -1.35, 1.35);
  renderAnalysisTrajectory();
  ev.preventDefault();
});
analysisReplay3dEl.addEventListener('pointerup', (ev) => {
  if (analysisDrag && analysisDrag.pointerId === ev.pointerId) {
    analysisDrag = null;
    analysisReplay3dEl.classList.remove('dragging');
    try { analysisReplay3dEl.releasePointerCapture(ev.pointerId); } catch (_) {}
  }
});
analysisReplay3dEl.addEventListener('pointercancel', () => {
  analysisDrag = null;
  analysisReplay3dEl.classList.remove('dragging');
});

visibleEl.addEventListener('change', drawAll);
ballSourceEl.addEventListener('change', drawAll);
useRosStateEl.addEventListener('change', () => {
  if (publishCommandsEl.checked && !publishCommandsEl.disabled) {
    useRosStateEl.checked = true;
  }
  drawAll();
});
publishCommandsEl.addEventListener('change', () => {
  if (publishCommandsEl.checked && !publishCommandsEl.disabled) {
    useRosStateEl.checked = true;
  }
  drawAll();
});
for (const el of Object.values(curveEls)) {
  el.addEventListener('change', drawAll);
}
for (const el of Object.values(flowEls)) {
  el.addEventListener('change', () => {
    syncEstimatorModeControl();
    drawAll();
  });
}
for (const el of Object.values(cameraFlowEls)) {
  el.addEventListener('change', drawAll);
}
window.addEventListener('resize', () => {
  resizeCanvas(view3dCanvas);
  resizeCanvas(topCanvas);
  resizeCanvas(sideCanvas);
  resizeCanvas(plotCanvas);
  if (currentView === 'analysis' && analysisData) renderAnalysis();
  drawAll();
});

resizeCanvas(view3dCanvas);
resizeCanvas(topCanvas);
resizeCanvas(sideCanvas);
resizeCanvas(plotCanvas);
setRuntimeDefaults();
setInputsFromBall();
drawAll();
refreshRecordingStatus();
refreshAnalysisStatus();
requestAnimationFrame(renderLoop);
setInterval(step, 1000 / STEP_RATE_HZ);
</script>
</body>
</html>
"""


def _control_subprocess_main(
    cfg: ProbeConfig,
    command_q,
    recording_cmd_q,
    status_q,
    frame_q,
    parent_pid: int | None = None,
) -> None:
    probe: PolicySafetyProbe | None = None
    shutdown_requested = threading.Event()
    hard_exit_timer: threading.Timer | None = None

    def _arm_hard_exit_timer(delay_s: float = 3.0) -> None:
        nonlocal hard_exit_timer
        if hard_exit_timer is not None:
            return
        hard_exit_timer = threading.Timer(delay_s, lambda: os._exit(0))
        hard_exit_timer.daemon = True
        hard_exit_timer.start()

    def _request_shutdown(_signum, _frame) -> None:
        shutdown_requested.set()
        _arm_hard_exit_timer()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _request_shutdown)
        except Exception:
            pass

    def _parent_is_gone() -> bool:
        if parent_pid is None or int(parent_pid) <= 1:
            return False
        try:
            return os.getppid() != int(parent_pid)
        except Exception:
            return False

    try:
        _install_parent_death_signal(signal.SIGKILL)
        if _parent_is_gone():
            return
        _prewarm_jax_cuda_for_mjx(cfg)
        probe = PolicySafetyProbe(cfg)
        _queue_replace_latest(status_q, {
            "kind": "config",
            "config": probe.browser_config(),
            "snapshot": probe.latest_snapshot_payload,
        })
        status_period_s = 1.0 / max(float(cfg.backend_snapshot_rate_hz), 1e-6)
        frame_period_s = 1.0 / max(float(cfg.camera_visualization_rate_hz), 1e-6)
        next_status_wall = time.time()
        next_frame_wall = time.time()
        last_frame_seq = -1
        last_sim_frame_seq = -1
        while not shutdown_requested.is_set():
            if _parent_is_gone():
                _queue_replace_latest(status_q, {
                    "kind": "shutdown",
                    "reason": "parent process exited",
                })
                return
            recording_cmds: list[dict[str, Any]] = []
            while True:
                try:
                    rec_cmd = recording_cmd_q.get_nowait()
                except Empty:
                    break
                if isinstance(rec_cmd, dict):
                    recording_cmds.append(rec_cmd)
            for rec_cmd in recording_cmds:
                kind = str(rec_cmd.get("kind", ""))
                payload = rec_cmd.get("payload", {})
                if not isinstance(payload, dict):
                    payload = {}
                if kind == "recording_start":
                    status = probe.start_recording(payload)
                    _queue_replace_latest(status_q, {
                        "kind": "recording",
                        "recording": status,
                    })
                elif kind == "recording_stop":
                    status = probe.stop_recording(payload)
                    _queue_replace_latest(status_q, {
                        "kind": "recording",
                        "recording": status,
                    })
                elif kind == "recording_status":
                    _queue_replace_latest(status_q, {
                        "kind": "recording",
                        "recording": probe.recording_status(),
                    })

            now = time.time()
            timeout_s = min(
                max(0.0, next_status_wall - now),
                max(0.0, next_frame_wall - now),
                0.02,
            )
            pending_cmds: list[dict[str, Any]] = []
            try:
                cmd = command_q.get(timeout=timeout_s)
                if isinstance(cmd, dict):
                    pending_cmds.append(cmd)
            except Empty:
                pass
            while True:
                try:
                    cmd = command_q.get_nowait()
                except Empty:
                    break
                if isinstance(cmd, dict):
                    pending_cmds.append(cmd)
            for cmd in pending_cmds:
                kind = str(cmd.get("kind", ""))
                if kind == "stop":
                    shutdown_requested.set()
                    return
                if kind == "reset":
                    snapshot = probe.reset()
                    _queue_replace_latest(status_q, {
                        "kind": "snapshot",
                        "snapshot": snapshot,
                    })
                    continue
                if kind == "step":
                    payload = cmd.get("payload", {})
                    if not isinstance(payload, dict):
                        payload = {}
                    snapshot = probe.step(payload)
                    if snapshot is not None:
                        _queue_replace_latest(status_q, {
                            "kind": "snapshot",
                            "snapshot": snapshot,
                        })

            now = time.time()
            if now >= next_status_wall:
                snapshot = probe.latest_snapshot_payload
                if snapshot is not None:
                    _queue_replace_latest(status_q, {
                        "kind": "snapshot",
                        "snapshot": snapshot,
                    })
                next_status_wall = now + status_period_s

            if now >= next_frame_wall:
                frame_payload: dict[str, Any] = {"kind": "frames"}
                frame, info = probe.latest_camera_visualization()
                seq = int(info.get("seq", 0) or 0)
                if frame is not None and seq != last_frame_seq:
                    frame_payload["camera"] = {
                        "frame": frame,
                        "info": dict(info),
                    }
                    probe.record_camera_frame(frame, info)
                    last_frame_seq = seq
                if bool(cfg.sim_enable_video_frames):
                    sim_frame, sim_info = probe.latest_sim_visualization()
                    sim_seq = int(sim_info.get("seq", 0) or 0)
                    if sim_frame is not None and sim_seq != last_sim_frame_seq:
                        frame_payload["sim"] = {
                            "frame": sim_frame,
                            "info": dict(sim_info),
                        }
                        probe.record_sim_frame(sim_frame, sim_info)
                        last_sim_frame_seq = sim_seq
                if "camera" in frame_payload or "sim" in frame_payload:
                    _queue_replace_latest(frame_q, frame_payload)
                next_frame_wall = now + frame_period_s
    except BaseException as exc:  # noqa: BLE001 - surface child failures.
        if not shutdown_requested.is_set() and not isinstance(exc, KeyboardInterrupt):
            _queue_replace_latest(status_q, {
                "kind": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
    finally:
        shutdown_requested.set()
        if probe is not None:
            probe.shutdown()
        if hard_exit_timer is not None:
            try:
                hard_exit_timer.cancel()
            except Exception:
                pass


class ControlSubprocessProxy:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.latest_config: dict[str, Any] | None = None
        self.latest_snapshot: dict[str, Any] | None = None
        self.latest_frame: bytes | None = None
        self.latest_frame_info: dict[str, Any] = {"seq": 0, "ageS": None}
        self.latest_sim_frame: bytes | None = None
        self.latest_sim_frame_info: dict[str, Any] = {"seq": 0, "ageS": None}
        self.latest_recording_status: dict[str, Any] = {
            "active": False,
            "runDir": None,
            "counts": {},
            "elapsedS": 0.0,
            "dropped": 0,
            "error": None,
            "analysis": {
                "active": False,
                "phase": "idle",
                "analysisDir": None,
                "outputs": {},
                "error": None,
            },
        }
        self.last_error: str | None = None
        self.last_traceback: str | None = None
        self._last_step_payload_key: str | None = None
        self._last_step_send_wall = 0.0
        self._shutdown = False
        self._pump_stop = threading.Event()
        self._pump_thread: threading.Thread | None = None
        self.ctx = mp.get_context("spawn")
        self.command_q = self.ctx.Queue(maxsize=1)
        self.recording_cmd_q = self.ctx.Queue(maxsize=8)
        self.status_q = self.ctx.Queue(maxsize=1)
        self.frame_q = self.ctx.Queue(maxsize=1)
        self.process = self.ctx.Process(
            target=_control_subprocess_main,
            args=(
                cfg, self.command_q, self.recording_cmd_q,
                self.status_q, self.frame_q, os.getpid()),
            name="policy_probe_control_process",
            daemon=False,
        )
        self.process.start()
        self._pump_thread = threading.Thread(
            target=self._pump_child_queues,
            name="policy_probe_control_queue_pump",
            daemon=True,
        )
        self._pump_thread.start()
        try:
            self._wait_for_initial_config()
        except BaseException:
            self.shutdown()
            raise

    def _wait_for_initial_config(self) -> None:
        try:
            startup_timeout_s = float(os.environ.get(
                "PROBE_CONTROL_SUBPROCESS_STARTUP_TIMEOUT_S", "300.0"))
        except ValueError:
            startup_timeout_s = 300.0
        startup_timeout_s = max(45.0, startup_timeout_s)
        start_wall = time.time()
        deadline = start_wall + startup_timeout_s
        next_notice_wall = start_wall + 15.0
        while time.time() < deadline:
            if self.latest_config is not None:
                return
            if self.last_error is not None:
                raise RuntimeError(
                    f"control subprocess failed: {self.last_error}\n"
                    f"{self.last_traceback or ''}")
            if not self.process.is_alive():
                raise RuntimeError(
                    f"control subprocess exited before config "
                    f"(exitcode={self.process.exitcode})")
            now = time.time()
            if now >= next_notice_wall:
                waited_s = now - start_wall
                print(
                    "[probe] waiting for control subprocess config "
                    f"({waited_s:.0f}/{startup_timeout_s:.0f}s)...",
                    file=sys.stderr,
                )
                next_notice_wall = now + 15.0
            time.sleep(0.05)
        raise TimeoutError(
            "timed out waiting for control subprocess config "
            f"after {startup_timeout_s:.0f}s")

    def _pump_child_queues(self) -> None:
        while not self._pump_stop.is_set():
            try:
                self._drain_status()
                self._drain_frames()
            except (EOFError, OSError, ValueError):
                if self._shutdown:
                    break
            except Exception as exc:
                with self.lock:
                    self.last_error = self.last_error or f"queue pump failed: {exc}"
                if self._shutdown:
                    break
            if not self.process.is_alive():
                try:
                    self._drain_status()
                    self._drain_frames()
                except Exception:
                    pass
                break
            self._pump_stop.wait(0.005)

    def _drain_status(self, block_timeout_s: float = 0.0) -> None:
        first = True
        while True:
            try:
                if first and block_timeout_s > 0.0:
                    item = self.status_q.get(timeout=block_timeout_s)
                else:
                    item = self.status_q.get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            finally:
                first = False
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", ""))
            with self.lock:
                if kind == "config":
                    config = item.get("config")
                    if isinstance(config, dict):
                        config = dict(config)
                        config["controlSubprocess"] = True
                        self.latest_config = config
                    snapshot = item.get("snapshot")
                    if isinstance(snapshot, dict):
                        self.latest_snapshot = snapshot
                        recording = snapshot.get("recording")
                        if isinstance(recording, dict):
                            self.latest_recording_status = dict(recording)
                elif kind == "snapshot":
                    snapshot = item.get("snapshot")
                    if isinstance(snapshot, dict):
                        self.latest_snapshot = snapshot
                        recording = snapshot.get("recording")
                        if isinstance(recording, dict):
                            self.latest_recording_status = dict(recording)
                elif kind == "recording":
                    recording = item.get("recording")
                    if isinstance(recording, dict):
                        self.latest_recording_status = dict(recording)
                elif kind == "error":
                    self.last_error = str(item.get("error", "unknown error"))
                    self.last_traceback = str(item.get("traceback", ""))

    def _drain_frames(self) -> None:
        while True:
            try:
                item = self.frame_q.get_nowait()
            except Empty:
                break
            except (EOFError, OSError, ValueError):
                break
            if not isinstance(item, dict):
                continue
            if str(item.get("kind", "")) == "frames":
                camera_item = item.get("camera")
                sim_item = item.get("sim")
                if isinstance(camera_item, dict):
                    frame = camera_item.get("frame")
                    info = camera_item.get("info", {})
                    if isinstance(frame, (bytes, bytearray)):
                        with self.lock:
                            self.latest_frame = bytes(frame)
                            self.latest_frame_info = (
                                dict(info) if isinstance(info, dict) else {})
                if isinstance(sim_item, dict):
                    frame = sim_item.get("frame")
                    info = sim_item.get("info", {})
                    if isinstance(frame, (bytes, bytearray)):
                        with self.lock:
                            self.latest_sim_frame = bytes(frame)
                            self.latest_sim_frame_info = (
                                dict(info) if isinstance(info, dict) else {})
                continue
            frame = item.get("frame")
            info = item.get("info", {})
            kind = str(item.get("kind", "camera"))
            if isinstance(frame, (bytes, bytearray)):
                with self.lock:
                    if kind == "sim":
                        self.latest_sim_frame = bytes(frame)
                        self.latest_sim_frame_info = (
                            dict(info) if isinstance(info, dict) else {})
                    else:
                        self.latest_frame = bytes(frame)
                        self.latest_frame_info = (
                            dict(info) if isinstance(info, dict) else {})

    def browser_config(self) -> dict[str, Any]:
        self._drain_status()
        with self.lock:
            if self.latest_config is not None:
                return dict(self.latest_config)
        raise RuntimeError("control subprocess config unavailable")

    def _step_payload_key(self, payload: dict[str, Any]) -> str:
        key_payload = dict(payload) if isinstance(payload, dict) else {}
        key_payload.pop("dt", None)
        key_payload.pop("recordingView", None)
        if str(key_payload.get("ballSource", "")).strip().lower() == "camera":
            key_payload.pop("ball", None)
            key_payload.pop("velocity", None)
            key_payload.pop("valid", None)
            key_payload.pop("noiseStdM", None)
        try:
            return json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        except TypeError:
            return repr(sorted(key_payload.items()))

    def _maybe_send_step_payload(self, payload: dict[str, Any]) -> bool:
        payload = payload if isinstance(payload, dict) else {}
        now = time.monotonic()
        key = self._step_payload_key(payload)
        has_recording_view = isinstance(payload.get("recordingView"), dict)
        should_send = (
            has_recording_view
            or key != self._last_step_payload_key
            or now - self._last_step_send_wall >= 0.2
        )
        if not should_send:
            return False
        _queue_replace_latest(self.command_q, {
            "kind": "step",
            "payload": payload,
        })
        self._last_step_payload_key = key
        self._last_step_send_wall = now
        return True

    @staticmethod
    def _snapshot_workspace_mode(snapshot: Any) -> str | None:
        if not isinstance(snapshot, dict):
            return None
        runtime = snapshot.get("runtime")
        if not isinstance(runtime, dict):
            return None
        value = runtime.get("workspaceSafetyMode")
        if value is None:
            return None
        return _workspace_safety_mode(value)

    @staticmethod
    def _snapshot_publish_commands(snapshot: Any) -> bool | None:
        if not isinstance(snapshot, dict):
            return None
        runtime = snapshot.get("runtime")
        if isinstance(runtime, dict) and "publishCommands" in runtime:
            return bool(runtime["publishCommands"])
        ros = snapshot.get("ros")
        if isinstance(ros, dict) and "publishCommands" in ros:
            return bool(ros["publishCommands"])
        return None

    @staticmethod
    def _snapshot_estimator_prediction(snapshot: Any) -> bool | None:
        if not isinstance(snapshot, dict):
            return None
        runtime = snapshot.get("runtime")
        if isinstance(runtime, dict) and "estimatorPrediction" in runtime:
            return bool(runtime["estimatorPrediction"])
        estimator = snapshot.get("estimator")
        if isinstance(estimator, dict) and "prediction" in estimator:
            return bool(estimator["prediction"])
        return None

    def _wait_for_runtime_options(
        self,
        requested_mode: str,
        requested_publish_commands: bool,
        requested_estimator_prediction: bool,
        timeout_s: float = 0.5,
    ) -> bool:
        """Wait for the 200 Hz child loop to acknowledge critical UI changes."""
        requested = _workspace_safety_mode(
            requested_mode, self.cfg.workspace_safety_mode)
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            with self.lock:
                applied = self._snapshot_workspace_mode(self.latest_snapshot)
                applied_publish = self._snapshot_publish_commands(
                    self.latest_snapshot)
                applied_prediction = self._snapshot_estimator_prediction(
                    self.latest_snapshot)
                child_error = self.last_error
            if (applied == requested
                    and applied_publish == requested_publish_commands
                    and applied_prediction
                    == requested_estimator_prediction):
                return True
            if child_error is not None or not self.process.is_alive():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._drain_status(block_timeout_s=min(0.02, remaining))

    def step(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = payload if isinstance(payload, dict) else {}
        requested_mode = _workspace_safety_mode(
            payload.get("workspaceSafetyMode", self.cfg.workspace_safety_mode),
            self.cfg.workspace_safety_mode,
        )
        requested_publish_commands = bool(
            payload.get("publishCommands", False)
            and self.cfg.ros_publish_commands
            and self.cfg.ros_bridge)
        requested_estimator_prediction = bool(payload.get(
            "estimatorPrediction",
            self.cfg.ball_estimator_enable_prediction,
        ))
        sent = self._maybe_send_step_payload(payload)
        if sent:
            with self.lock:
                applied_mode = self._snapshot_workspace_mode(
                    self.latest_snapshot)
                applied_publish = self._snapshot_publish_commands(
                    self.latest_snapshot)
                applied_prediction = self._snapshot_estimator_prediction(
                    self.latest_snapshot)
            if (applied_mode != requested_mode
                    or applied_publish != requested_publish_commands
                    or applied_prediction
                    != requested_estimator_prediction):
                self._wait_for_runtime_options(
                    requested_mode,
                    requested_publish_commands,
                    requested_estimator_prediction,
                )
        self._drain_status()
        with self.lock:
            if self.latest_snapshot is not None:
                snapshot = dict(self.latest_snapshot)
                snapshot["recording"] = dict(self.latest_recording_status)
                applied_mode = self._snapshot_workspace_mode(snapshot)
                snapshot["workspaceModeRequested"] = requested_mode
                snapshot["workspaceModePending"] = bool(
                    applied_mode != requested_mode)
                applied_publish = self._snapshot_publish_commands(snapshot)
                snapshot["commandPublishRequested"] = bool(
                    requested_publish_commands)
                snapshot["commandPublishPending"] = bool(
                    applied_publish != requested_publish_commands)
                applied_prediction = self._snapshot_estimator_prediction(
                    snapshot)
                snapshot["estimatorPredictionRequested"] = bool(
                    requested_estimator_prediction)
                snapshot["estimatorPredictionPending"] = bool(
                    applied_prediction != requested_estimator_prediction)
                return snapshot
            if self.last_error is not None:
                return {"error": self.last_error, "traceback": self.last_traceback}
        return {"mode": "waiting", "reason": "control subprocess starting"}

    def reset(self) -> dict[str, Any]:
        _queue_replace_latest(self.command_q, {"kind": "reset"})
        deadline = time.time() + 2.0
        while time.time() < deadline:
            self._drain_status(block_timeout_s=0.05)
            with self.lock:
                if self.latest_snapshot is not None:
                    snapshot = dict(self.latest_snapshot)
                    snapshot["recording"] = dict(self.latest_recording_status)
                    return snapshot
        with self.lock:
            snapshot = dict(self.latest_snapshot or {
                "mode": "waiting",
                "reason": "reset pending",
            })
            snapshot["recording"] = dict(self.latest_recording_status)
            return snapshot

    def _send_recording_command(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        wait_s: float = 2.0,
    ) -> dict[str, Any]:
        try:
            self.recording_cmd_q.put({
                "kind": kind,
                "payload": payload if isinstance(payload, dict) else {},
            }, timeout=0.5)
        except Full:
            _queue_replace_latest(self.recording_cmd_q, {
                "kind": kind,
                "payload": payload if isinstance(payload, dict) else {},
            })
        deadline = time.time() + wait_s
        while time.time() < deadline:
            self._drain_status(block_timeout_s=0.05)
            with self.lock:
                status = dict(self.latest_recording_status)
            if kind == "recording_status":
                return status
            if kind == "recording_start" and status.get("active"):
                return status
            if kind == "recording_stop" and not status.get("active"):
                return status
        with self.lock:
            return dict(self.latest_recording_status)

    def start_recording(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._send_recording_command("recording_start", payload)

    def stop_recording(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            flush_timeout_s = max(0.0, float(os.environ.get(
                "SIM_RECORDING_FLUSH_TIMEOUT_S", "5.0")))
        except (TypeError, ValueError):
            flush_timeout_s = 5.0
        return self._send_recording_command(
            "recording_stop", payload, wait_s=max(5.0, flush_timeout_s + 12.0))

    def recording_status(self) -> dict[str, Any]:
        self._send_recording_command("recording_status", {}, wait_s=0.2)
        with self.lock:
            return dict(self.latest_recording_status)

    def cached_recording_status(self) -> dict[str, Any]:
        self._drain_status()
        with self.lock:
            return dict(self.latest_recording_status)

    def record_camera_frame(
        self,
        frame: bytes | None,
        info: dict[str, Any] | None,
    ) -> None:
        del frame, info

    def record_sim_frame(
        self,
        frame: bytes | None,
        info: dict[str, Any] | None,
    ) -> None:
        del frame, info

    def latest_camera_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        self._drain_frames()
        with self.lock:
            return self.latest_frame, dict(self.latest_frame_info)

    def latest_sim_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        self._drain_frames()
        with self.lock:
            return self.latest_sim_frame, dict(self.latest_sim_frame_info)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            _queue_replace_latest(self.command_q, {"kind": "stop"})
        except Exception:
            pass
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            _terminate_process_tree(self.process, "control subprocess")
        self._pump_stop.set()
        if self._pump_thread is not None and self._pump_thread.is_alive():
            self._pump_thread.join(timeout=1.0)
        for q in (
                self.command_q, self.recording_cmd_q,
                self.status_q, self.frame_q):
            _close_mp_queue(q)


class ProbeRequestHandler(BaseHTTPRequestHandler):
    probe: PolicySafetyProbe

    def log_message(self, fmt: str, *args: Any) -> None:
        message = fmt % args
        if (
                " 200 " in message
                and (
                    '"POST /step ' in message
                    or '"GET /camera_frame.jpg' in message
                    or '"GET /sim_frame.jpg' in message
                    or '"GET /recording/status ' in message
                    or '"GET /analysis/status ' in message
                )):
            return
        sys.stderr.write("[probe] " + message + "\n")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            data = {}
        return data if isinstance(data, dict) else {}

    def _recording_status(self) -> dict[str, Any]:
        try:
            status = self.probe.recording_status()
        except Exception as exc:
            return {
                "active": False,
                "runDir": None,
                "analysis": {
                    "active": False,
                    "phase": "error",
                    "analysisDir": None,
                    "outputs": {},
                    "error": str(exc),
                },
            }
        return status if isinstance(status, dict) else {}

    def _analysis_status(self) -> dict[str, Any]:
        if hasattr(self.probe, "cached_recording_status"):
            try:
                status = self.probe.cached_recording_status()
            except Exception:
                status = self._recording_status()
        else:
            status = self._recording_status()
        analysis = status.get("analysis", {})
        return analysis if isinstance(analysis, dict) else {}

    def _send_file_response(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        suffix = path.suffix.lower()
        if suffix == ".mp4":
            ctype = "video/mp4"
        elif suffix in (".jpg", ".jpeg"):
            ctype = "image/jpeg"
        elif suffix == ".png":
            ctype = "image/png"
        elif suffix == ".json":
            ctype = "application/json"
        elif suffix == ".html":
            ctype = "text/html; charset=utf-8"
        range_header = self.headers.get("Range", "")
        start = 0
        end = size - 1
        partial = False
        if range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1].split(",", 1)[0].strip()
            if "-" in spec:
                left, right = spec.split("-", 1)
                try:
                    if left:
                        start = max(0, int(left))
                    if right:
                        end = min(size - 1, int(right))
                    partial = start <= end
                except ValueError:
                    start, end, partial = 0, size - 1, False
        length = max(0, end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_analysis_file(self, raw_path: str) -> None:
        analysis = self._analysis_status()
        analysis_dir_text = analysis.get("analysisDir")
        if not analysis_dir_text:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        analysis_dir = Path(str(analysis_dir_text)).resolve()
        rel = urllib.parse.unquote(raw_path).lstrip("/")
        candidate = (analysis_dir / rel).resolve()
        try:
            candidate.relative_to(analysis_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self._send_file_response(candidate)

    def _send_analysis_frame(self, raw_path: str) -> None:
        analysis = self._analysis_status()
        analysis_dir_text = analysis.get("analysisDir")
        if not analysis_dir_text:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        run_dir = Path(str(analysis_dir_text)).resolve().parent
        frames_dir = (run_dir / "frames").resolve()
        filename = Path(urllib.parse.unquote(raw_path).lstrip("/")).name
        candidate = (frames_dir / filename).resolve()
        try:
            candidate.relative_to(frames_dir)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        self._send_file_response(candidate)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        request_path = parsed.path
        if request_path in ("/", "/index.html"):
            html = HTML_TEMPLATE.replace(
                "__CONFIG_JSON__",
                json.dumps(self.probe.browser_config(), separators=(",", ":")),
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html)
            return
        if request_path == "/config":
            self._send_json(self.probe.browser_config())
            return
        if request_path == "/recording/status":
            self._send_json(self._recording_status())
            return
        if request_path == "/analysis/status":
            status = self._recording_status()
            analysis = status.get("analysis", {})
            self._send_json(analysis if isinstance(analysis, dict) else {})
            return
        if request_path.startswith("/analysis/file/"):
            self._send_analysis_file(request_path[len("/analysis/file/"):])
            return
        if request_path.startswith("/analysis/frame/"):
            self._send_analysis_frame(request_path[len("/analysis/frame/"):])
            return
        if request_path.startswith("/camera_frame.jpg"):
            frame, info = self.probe.latest_camera_visualization()
            if frame is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.probe.record_camera_frame(frame, info)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Camera-Frame-Seq", str(info.get("seq", 0)))
            self.end_headers()
            self.wfile.write(frame)
            return
        if request_path.startswith("/sim_frame.jpg"):
            if not bool(self.probe.cfg.sim_enable_video_frames):
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            frame, info = self.probe.latest_sim_visualization()
            if frame is None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.probe.record_sim_frame(frame, info)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Sim-Frame-Seq", str(info.get("seq", 0)))
            self.end_headers()
            self.wfile.write(frame)
            return
        if request_path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if self.path == "/step":
                self._send_json(self.probe.step(self._read_json()))
                return
            if self.path == "/reset":
                self._send_json(self.probe.reset())
                return
            if self.path == "/recording/start":
                self._send_json(self.probe.start_recording(self._read_json()))
                return
            if self.path == "/recording/stop":
                self._send_json(self.probe.stop_recording(self._read_json()))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch an interactive browser policy safety probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--robot-xml-path", type=Path, default=DEFAULT_ROBOT_XML)
    p.add_argument("--policy-obs-dim", type=int, choices=(50, 67), default=67)
    p.add_argument("--control-rate-hz", type=finite_float, default=200.0)
    p.add_argument("--ball-lost-failsafe-timeout-s", type=finite_float, default=0.6)
    p.add_argument("--enable-safety-limiter", action="store_true")
    p.add_argument("--safety-parking-velocity-scale", type=finite_float, default=0.08)
    p.add_argument("--safety-parking-acceleration-scale", type=finite_float, default=0.06)
    p.add_argument("--disable-ball-capture-gate", action="store_true")
    p.add_argument("--disable-ball-release-gate", action="store_true")
    p.add_argument("--disable-ball-lost-failsafe", action="store_true")
    p.add_argument("--disable-done-monitoring", action="store_true")
    p.add_argument(
        "--disable-rl-valid-ball-obs-age",
        action="store_true",
        help=(
            "Restore the legacy policy observation path by forcing obs age to "
            "0.0 for any valid ball input."
        ),
    )
    p.add_argument(
        "--disable-compensation",
        action="store_true",
        help="Disable real-node actuator delay compensation in the probe.",
    )
    p.add_argument("--compensation-mode", default="inverse_mpc")
    p.add_argument("--comp-delay-ms", type=finite_float, default=72.0)
    p.add_argument("--comp-actuator-tau-s", type=finite_float, default=0.074)
    p.add_argument("--comp-actuator-gain", type=finite_float, default=1.0)
    p.add_argument("--comp-buffer-extra-steps", type=int, default=4)
    p.add_argument(
        "--disable-comp-delay-conditioning",
        action="store_true",
        help="Disable delay conditioning inside the actuator compensator.",
    )
    p.add_argument("--comp-mpc-beta", type=finite_float, default=1.2)
    p.add_argument("--comp-mpc-delay-scale", type=finite_float, default=1.05)
    p.add_argument("--comp-mpc-tau-scale", type=finite_float, default=0.75)
    p.add_argument("--comp-mpc-horizon-steps", type=int, default=6)
    p.add_argument("--comp-mpc-tracking-weight", type=finite_float, default=1.0)
    p.add_argument("--comp-mpc-nominal-weight", type=finite_float, default=0.25)
    p.add_argument("--comp-mpc-delta-weight", type=finite_float, default=0.05)
    p.add_argument(
        "--comp-mpc-max-delta-rad",
        type=finite_float,
        default=0.5235987755982988,
    )
    p.add_argument("--ball-release-min-downward-velocity-m-s", type=finite_float, default=0.15)
    p.add_argument("--ball-release-min-z-drop-m", type=finite_float, default=0.015)
    p.add_argument("--ball-release-window-s", type=finite_float, default=0.12)
    p.add_argument("--ball-release-required-consecutive-ticks", type=int, default=3)
    p.add_argument("--ball-release-max-obs-age-s", type=finite_float, default=0.08)
    p.add_argument(
        "--ball-release-use-estimator-state",
        action="store_true",
        help=(
            "Restore the legacy release gate path by evaluating release on the "
            "post-estimator control ball state."
        ),
    )
    p.add_argument("--done-z-min-m", type=finite_float, default=0.3)
    p.add_argument("--done-z-max-m", type=finite_float, default=1.9)
    p.add_argument("--done-x-range-m", type=finite_float, default=0.5)
    p.add_argument("--done-y-range-m", type=finite_float, default=0.5)
    p.add_argument(
        "--done-ball-required-consecutive-ticks",
        type=int,
        default=5,
        help=(
            "Consecutive 200 Hz ticks required before a measured ball "
            "workspace violation latches done."),
    )
    p.add_argument("--done-racket-anchor-distance-m", type=finite_float, default=1.1)
    p.add_argument("--done-racket-z-high-offset-m", type=finite_float, default=0.24)
    p.add_argument("--done-racket-z-low-offset-m", type=finite_float, default=-0.12)
    p.add_argument("--ee-x-limits-m", default="-0.298,0.163")
    p.add_argument("--ee-y-limits-m", default="-0.502,-0.290")
    p.add_argument("--ee-z-limits-m", default="0.692,1.462")
    p.add_argument("--ee-psi-limits-rad", default="")
    p.add_argument("--ee-b-limits-rad", default="")
    p.add_argument("--ee-b-min-rad", default="")
    p.add_argument("--ee-b-max-rad", default="")
    p.add_argument("--ee-c-limits-rad", default="")
    p.add_argument("--ee-c-min-rad", default="")
    p.add_argument("--ee-c-max-rad", default="")
    p.add_argument("--ee-orientation-reference-xyzw", default="")
    p.add_argument(
        "--ee-max-orientation-error-deg",
        type=finite_float,
        default=0.0,
        help="Max EE quaternion angle error from reference; <=0 disables.",
    )
    p.add_argument("--disable-done-predictive-stop", action="store_true")
    p.add_argument("--done-prediction-horizon-s", type=finite_float, default=0.12)
    p.add_argument("--done-prediction-dt-s", type=finite_float, default=0.01)
    p.add_argument("--done-prediction-margin-m", type=finite_float, default=0.02)
    p.add_argument(
        "--done-prediction-orientation-margin-rad",
        type=finite_float,
        default=0.035)
    p.add_argument(
        "--workspace-safety-mode",
        choices=WORKSPACE_SAFETY_MODES,
        default="position_soft",
        help=(
            "Runtime default for physical EE workspace safety: off disables "
            "EE xyz workspace limits; position and predictive latch done; "
            "position_soft continuously limits only Cartesian outward-normal "
            "velocity; predictive_soft reserves measured horizon travel; "
            "impedance_soft adds a bounded normal virtual wall. All soft modes "
            "share the same no-slack Cartesian CBF backstop and retain "
            "tangential/inward policy motion."
        ),
    )
    p.add_argument(
        "--workspace-hold-trigger-margin-m",
        type=finite_float,
        default=0.003,
        help="Inset distance that engages the boundary command hold.",
    )
    p.add_argument(
        "--workspace-soft-margin-m",
        type=finite_float,
        default=0.05,
        help="Width of the virtual workspace spring zone in metres.",
    )
    p.add_argument(
        "--workspace-soft-pull-speed-m-s",
        type=finite_float,
        default=0.08,
        help="Maximum nominal inward spring speed at the boundary.",
    )
    p.add_argument(
        "--workspace-soft-damping",
        type=finite_float,
        default=1.0,
        help="Fraction of outward Cartesian command velocity damped at the boundary.",
    )
    p.add_argument(
        "--workspace-soft-max-correction-speed-m-s",
        type=finite_float,
        default=0.30,
        help=(
            "Allowed outward speed at the soft-zone entrance; together with "
            "the margin this sets the velocity-CBF gain."),
    )
    p.add_argument(
        "--workspace-soft-dls-damping",
        type=finite_float,
        default=0.03,
        help="Damped-least-squares regularizer for Cartesian-to-joint correction.",
    )
    p.add_argument(
        "--workspace-soft-nullspace-damping",
        type=finite_float,
        default=0.0,
        help="Deprecated compatibility option; the CBF preserves nullspace motion.",
    )
    p.add_argument(
        "--workspace-soft-posture-gain-s",
        type=finite_float,
        default=0.0,
        help="Deprecated compatibility option; no recovery posture is injected.",
    )
    p.add_argument(
        "--workspace-soft-boundary-joint-speed-scale",
        type=finite_float,
        default=1.0,
        help="Deprecated compatibility option; tangential speed is not scaled.",
    )
    p.add_argument(
        "--ball-noise-std-m",
        type=finite_float,
        default=0.0,
        help="Default Gaussian position noise std for ball observations, in meters.",
    )
    p.add_argument("--ball-observation-rate-hz", type=finite_float, default=60.0)
    p.add_argument(
        "--backend-control-loop",
        action="store_true",
        help=(
            "Run policy/safety/estimator/publish in a backend fixed-rate "
            "control loop; browser /step only updates runtime options."
        ),
    )
    p.add_argument(
        "--control-subprocess",
        action="store_true",
        help=(
            "Run ROS/camera/policy/control in a separate process; the browser "
            "process only proxies options and cached status."
        ),
    )
    p.add_argument(
        "--backend-snapshot-rate-hz",
        type=finite_float,
        default=60.0,
        help=(
            "Snapshot/status packaging rate for the backend control loop. "
            "The 200 Hz command loop is independent of this."
        ),
    )
    p.add_argument(
        "--browser-step-rate-hz",
        type=finite_float,
        default=60.0,
        help="Browser -> backend policy probe update cadence.",
    )
    p.add_argument(
        "--display-rate-hz",
        type=finite_float,
        default=60.0,
        help="Browser canvas redraw cadence.",
    )
    p.add_argument(
        "--recording-output-root",
        type=Path,
        default=SOURCE_MODULE_DIR / "outputs" / "interactive_recordings",
        help="Directory where browser-triggered probe recordings are written.",
    )
    p.add_argument(
        "--recording-control-rate-hz",
        type=finite_float,
        default=200.0,
        help="Max rate for high-frequency compact control samples; <=0 disables them.",
    )
    p.add_argument(
        "--recording-image-rate-hz",
        type=finite_float,
        default=30.0,
        help="Max rate for saved camera visualization JPEG frames; <=0 disables images.",
    )
    p.add_argument(
        "--recording-analysis-video-fps",
        type=finite_float,
        default=30.0,
        help="FPS for camera/video replay artifacts generated by analysis.",
    )
    p.add_argument(
        "--recording-queue-size",
        type=int,
        default=1024,
        help="Max pending recording events before new samples are dropped.",
    )
    p.add_argument(
        "--disable-ball-state-estimator",
        action="store_true",
        help="Disable the pingpong_node-style 60 Hz update / 200 Hz predict ball estimator.",
    )
    p.add_argument(
        "--enable-ball-estimator-prediction",
        dest="ball_estimator_prediction",
        action="store_true",
        help="Enable control-time ballistic prediction/bridging (default: off).",
    )
    p.add_argument(
        "--disable-ball-estimator-prediction",
        dest="ball_estimator_prediction",
        action="store_false",
        help="Keep estimator smoothing/loss aging but disable control-time extrapolation.",
    )
    p.set_defaults(ball_estimator_prediction=False)
    p.add_argument(
        "--disable-ball-estimator-aero-drag",
        action="store_true",
        help="Disable quadratic aerodynamic drag in estimator prediction.",
    )
    p.add_argument("--ball-estimator-max-extrapolation-s", type=finite_float, default=0.45)
    p.add_argument(
        "--ball-estimator-control-max-extrapolation-s",
        type=finite_float,
        default=0.08,
        help=(
            "Max estimator extrapolation that may still be treated as a "
            "valid policy input; set <0 to disable the control-side gate."
        ),
    )
    p.add_argument(
        "--ball-estimator-control-output-mode",
        choices=CONTROL_OUTPUT_MODES,
        default=CONTROL_OUTPUT_CONTINUOUS,
        help=(
            "Estimator output contract: continuous predicts every control "
            "tick; predict_on_frame_hold predicts once per output frame and "
            "holds the snapshot between ticks."
        ),
    )
    p.add_argument(
        "--ball-estimator-frame-hold-output-rate-hz",
        type=finite_float,
        default=50.0,
        help="Snapshot refresh rate for predict_on_frame_hold mode.",
    )
    p.add_argument(
        "--ball-estimator-release-gated-dynamics",
        action="store_true",
        help=(
            "Use constant-velocity estimator dynamics before the measured "
            "release gate clears, then enable ballistic dynamics and reseed "
            "velocity from recent measurements."
        ),
    )
    p.add_argument("--ball-estimator-trust-loss-s", type=finite_float, default=0.10)
    p.add_argument("--ball-estimator-degrade-loss-s", type=finite_float, default=0.40)
    p.add_argument("--ball-estimator-invalid-after-s", type=finite_float, default=0.60)
    p.add_argument("--ball-estimator-prediction-horizon-s", type=finite_float, default=0.0)
    p.add_argument(
        "--ball-estimator-q-pos-mm2-s",
        type=finite_float,
        default=500.0,
        help="Estimator position process-noise spectral density.",
    )
    p.add_argument(
        "--ball-estimator-q-vel-mult",
        type=finite_float,
        default=100.0,
        help="Velocity-to-position process-noise multiplier.",
    )
    p.add_argument(
        "--ball-estimator-q-depth-pos-mult",
        type=finite_float,
        default=0.50,
        help="Camera-depth multiplier applied to position process noise.",
    )
    p.add_argument(
        "--ball-estimator-q-depth-vel-mult",
        type=finite_float,
        default=0.20,
        help="Camera-depth multiplier applied to velocity process noise.",
    )
    p.add_argument(
        "--ball-estimator-r-lateral-mm",
        type=finite_float,
        default=20.0,
        help="Estimator measurement noise sigma for camera lateral axes.",
    )
    p.add_argument(
        "--ball-estimator-r-depth-mm",
        type=finite_float,
        default=60.0,
        help="Estimator measurement noise sigma for the camera optical-depth axis.",
    )
    p.add_argument(
        "--ros-bridge",
        action="store_true",
        help="Enable ROS2 bridge: publish /mx_mix_command and subscribe feedback topics.",
    )
    p.add_argument(
        "--disable-command-publish",
        action="store_true",
        help="With --ros-bridge, subscribe feedback only and do not publish commands.",
    )
    p.add_argument("--command-topic", default="/mx_mix_command")
    p.add_argument("--joint-state-topic", default="/joint_states")
    p.add_argument("--cart-state-topic", default="/cart_states")
    p.add_argument("--right-arm-mu-idx", type=int, default=3)
    p.add_argument(
        "--camera-ball",
        action="store_true",
        help="Enable pingpong_node-style camera ball source.",
    )
    p.add_argument(
        "--camera-source-mode",
        choices=("direct_realsense", "ros_topic"),
        default="direct_realsense",
        help="Camera source path. direct_realsense matches use_direct_realsense:=true.",
    )
    p.add_argument(
        "--disable-hybrid-detector",
        action="store_true",
        help="Disable the hybrid detector path. direct_realsense currently expects hybrid.",
    )
    p.add_argument(
        "--disable-depth-sensor",
        action="store_true",
        help="Disable hardware depth for the hybrid/direct RealSense path.",
    )
    p.add_argument("--image-width", type=int, default=848)
    p.add_argument("--image-height", type=int, default=480)
    p.add_argument("--realsense-fps", type=int, default=60)
    p.add_argument("--camera-image-topic", default="/camera/camera/color/image_raw/compressed")
    p.add_argument("--camera-raw-image-topic", default="/camera/camera/color/image_raw")
    p.add_argument("--camera-image-transport", choices=("compressed", "raw", "auto"), default="auto")
    p.add_argument("--camera-info-topic", default="/camera/camera/color/camera_info")
    p.add_argument("--calibration-json-path", default=str(DEFAULT_CALIBRATION_JSON))
    p.add_argument("--yolo-model-path", type=Path, default=DEFAULT_VISION_MODEL)
    p.add_argument("--target-class", default="Class 0")
    p.add_argument("--conf-threshold", type=finite_float, default=0.25)
    p.add_argument("--ir-params-json", default=str(DEFAULT_IR_PARAMS_JSON))
    p.add_argument("--yolo-interval", type=int, default=1)
    p.add_argument("--ir-threshold", type=int, default=170)
    p.add_argument("--disable-adaptive-ir", action="store_true")
    p.add_argument("--ir-exposure", type=int, default=1000)
    p.add_argument("--ir-gain", type=int, default=16)
    p.add_argument("--ir-laser-power", type=int, default=360)
    p.add_argument("--ir-min-radius", type=int, default=15)
    p.add_argument("--ir-max-radius", type=int, default=60)
    p.add_argument("--ir-circularity-min", type=finite_float, default=0.7)
    p.add_argument("--ir-brightness-percentile", type=int, default=25)
    p.add_argument("--hybrid-roi-margin", type=int, default=100)
    p.add_argument(
        "--depth-correction-model-path",
        default=str(DEFAULT_DEPTH_CORRECTION_MODEL),
    )
    p.add_argument("--depth-neighborhood-px", type=int, default=2)
    p.add_argument("--infer-imgsz", type=int, default=640)
    p.add_argument("--detector-device", default="0")
    p.add_argument("--detector-half", action="store_true")
    p.add_argument("--use-kalman", action="store_true")
    p.add_argument("--camera-tilt-deg", type=finite_float, default=50.33)
    p.add_argument("--disable-undistort", action="store_true")
    p.add_argument("--disable-ellipse-fit", action="store_true")
    p.add_argument("--enable-temporal-tracker", action="store_true")
    p.add_argument("--tracker-max-miss-bridge", type=int, default=2)
    p.add_argument("--tracker-gate-px", type=finite_float, default=120.0)
    p.add_argument("--hsv-lower", type=parse_int_triplet, default=(5, 50, 50))
    p.add_argument("--hsv-upper", type=parse_int_triplet, default=(30, 255, 255))
    p.add_argument("--color-match-threshold", type=finite_float, default=0.3)
    p.add_argument("--color-circularity-min", type=finite_float, default=0.4)
    p.add_argument("--disable-camera-detection", action="store_true")
    p.add_argument("--disable-camera-visualization", action="store_true")
    p.add_argument(
        "--camera-visualization-rate-hz",
        type=finite_float,
        default=60.0,
        help="Max rate for JPEG camera overlay rendering in the backend.",
    )
    p.add_argument(
        "--camera-web-max-width",
        type=int,
        default=848,
        help="Resize HTML camera JPEG frames to this max width; <=0 keeps full size.",
    )
    p.add_argument(
        "--camera-web-jpeg-quality",
        type=int,
        default=62,
        help="JPEG quality for the HTML camera stream.",
    )
    p.add_argument("--disable-camera-color-filter", action="store_true")
    p.add_argument("--sim-cartesian-x-offset-m", type=finite_float, default=0.0)
    p.add_argument("--sim-cartesian-y-offset-m", type=finite_float, default=0.023583)
    p.add_argument("--sim-base-height-m", type=finite_float, default=1.302396)
    p.add_argument("--enable-sim-mirror", action="store_true")
    p.add_argument(
        "--sim-mirror-default-disabled",
        action="store_true",
        help="Initialize the sim mirror but leave the web runtime checkbox off.",
    )
    p.add_argument(
        "--disable-sim-subprocess",
        action="store_true",
        help=(
            "Run the sim mirror inline with the policy/control process instead "
            "of isolating MJX/MuJoCo in its own subprocess."
        ),
    )
    p.add_argument(
        "--sim-backend",
        choices=("mujoco", "mjx", "auto"),
        default="mujoco",
        help="Simulation mirror backend. mjx uses simcode/MJX/JAX; auto tries mjx then MuJoCo.",
    )
    p.add_argument("--sim-xml-path", type=Path, default=DEFAULT_SIM_XML)
    p.add_argument(
        "--enable-sim-video",
        action="store_true",
        help=(
            "Render and serve sim mirror JPEG frames. Disabled by default so "
            "MJX/MuJoCo simulation time is spent on policy/control stepping."
        ),
    )
    p.add_argument("--sim-frame-rate-hz", type=finite_float, default=30.0)
    p.add_argument("--sim-render-width", type=int, default=640)
    p.add_argument("--sim-render-height", type=int, default=480)
    p.add_argument("--disable-sim-mujoco-renderer", action="store_true")
    p.add_argument("--open-browser", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> ProbeConfig:
    model_path = args.model_path.expanduser().resolve()
    robot_xml = args.robot_xml_path.expanduser().resolve()
    yolo_model_path = args.yolo_model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")
    if not robot_xml.is_file():
        raise FileNotFoundError(f"robot XML not found: {robot_xml}")
    if bool(args.camera_ball) and not yolo_model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {yolo_model_path}")
    sim_xml = args.sim_xml_path.expanduser().resolve()
    if bool(args.enable_sim_mirror) and not sim_xml.is_file():
        raise FileNotFoundError(f"sim XML not found: {sim_xml}")
    return ProbeConfig(
        host=args.host,
        port=int(args.port),
        model_path=model_path,
        robot_xml_path=robot_xml,
        policy_obs_dim=int(args.policy_obs_dim),
        control_rate_hz=float(args.control_rate_hz),
        ball_lost_timeout_s=float(args.ball_lost_failsafe_timeout_s),
        enable_safety_limiter=bool(args.enable_safety_limiter),
        safety_parking_velocity_scale=min(
            1.0, max(1e-3, float(args.safety_parking_velocity_scale))),
        safety_parking_acceleration_scale=min(
            1.0, max(1e-3, float(args.safety_parking_acceleration_scale))),
        enable_compensation=not bool(args.disable_compensation),
        compensation_mode=str(args.compensation_mode),
        comp_delay_ms=float(args.comp_delay_ms),
        comp_actuator_tau_s=float(args.comp_actuator_tau_s),
        comp_actuator_gain=float(args.comp_actuator_gain),
        comp_buffer_extra_steps=max(0, int(args.comp_buffer_extra_steps)),
        comp_delay_conditioning=not bool(args.disable_comp_delay_conditioning),
        comp_mpc_beta=float(args.comp_mpc_beta),
        comp_mpc_delay_scale=float(args.comp_mpc_delay_scale),
        comp_mpc_tau_scale=float(args.comp_mpc_tau_scale),
        comp_mpc_horizon_steps=max(1, int(args.comp_mpc_horizon_steps)),
        comp_mpc_tracking_weight=float(args.comp_mpc_tracking_weight),
        comp_mpc_nominal_weight=float(args.comp_mpc_nominal_weight),
        comp_mpc_delta_weight=float(args.comp_mpc_delta_weight),
        comp_mpc_max_delta_rad=float(args.comp_mpc_max_delta_rad),
        enable_ball_capture_gate=not bool(args.disable_ball_capture_gate),
        enable_ball_release_gate=not bool(args.disable_ball_release_gate),
        enable_ball_lost_failsafe=not bool(args.disable_ball_lost_failsafe),
        enable_done_monitoring=not bool(args.disable_done_monitoring),
        rl_use_valid_ball_obs_age=not bool(args.disable_rl_valid_ball_obs_age),
        ball_release_min_downward_velocity_m_s=max(
            0.0, float(args.ball_release_min_downward_velocity_m_s)),
        ball_release_min_z_drop_m=max(
            0.0, float(args.ball_release_min_z_drop_m)),
        ball_release_window_s=max(0.02, float(args.ball_release_window_s)),
        ball_release_required_consecutive_ticks=max(
            1, int(args.ball_release_required_consecutive_ticks)),
        ball_release_max_obs_age_s=max(
            0.0, float(args.ball_release_max_obs_age_s)),
        ball_release_use_estimator_state=bool(
            args.ball_release_use_estimator_state),
        done_z_min_m=float(args.done_z_min_m),
        done_z_max_m=float(args.done_z_max_m),
        done_x_range_m=float(args.done_x_range_m),
        done_y_range_m=float(args.done_y_range_m),
        done_ball_required_consecutive_ticks=max(
            1, int(args.done_ball_required_consecutive_ticks)),
        done_racket_anchor_distance_m=float(args.done_racket_anchor_distance_m),
        done_racket_z_high_offset_m=float(args.done_racket_z_high_offset_m),
        done_racket_z_low_offset_m=float(args.done_racket_z_low_offset_m),
        ee_x_limits_m=parse_limits(args.ee_x_limits_m),
        ee_y_limits_m=parse_limits(args.ee_y_limits_m),
        ee_z_limits_m=parse_limits(args.ee_z_limits_m),
        ee_psi_limits_rad=parse_angle_limits(args.ee_psi_limits_rad),
        ee_b_limits_rad=parse_angle_limits(args.ee_b_limits_rad),
        ee_b_min_rad=parse_optional_float(args.ee_b_min_rad),
        ee_b_max_rad=parse_optional_float(args.ee_b_max_rad),
        ee_c_limits_rad=parse_angle_limits(args.ee_c_limits_rad),
        ee_c_min_rad=parse_optional_float(args.ee_c_min_rad),
        ee_c_max_rad=parse_optional_float(args.ee_c_max_rad),
        ee_orientation_reference_xyzw=parse_quat_xyzw(
            args.ee_orientation_reference_xyzw),
        ee_max_orientation_error_deg=max(
            0.0, float(args.ee_max_orientation_error_deg)),
        done_enable_predictive_stop=not bool(args.disable_done_predictive_stop),
        done_prediction_horizon_s=max(
            0.0, float(args.done_prediction_horizon_s)),
        done_prediction_dt_s=max(1e-4, float(args.done_prediction_dt_s)),
        done_prediction_margin_m=max(
            0.0, float(args.done_prediction_margin_m)),
        done_prediction_orientation_margin_rad=max(
            0.0, float(args.done_prediction_orientation_margin_rad)),
        workspace_soft_margin_m=max(
            1e-3, float(args.workspace_soft_margin_m)),
        workspace_soft_pull_speed_m_s=max(
            0.0, float(args.workspace_soft_pull_speed_m_s)),
        workspace_soft_damping=max(
            0.0, float(args.workspace_soft_damping)),
        workspace_soft_max_correction_speed_m_s=max(
            1e-3, float(args.workspace_soft_max_correction_speed_m_s)),
        workspace_soft_dls_damping=max(
            1e-6, float(args.workspace_soft_dls_damping)),
        workspace_soft_nullspace_damping=min(
            1.0, max(0.0, float(args.workspace_soft_nullspace_damping))),
        workspace_soft_posture_gain_s=max(
            0.0, float(args.workspace_soft_posture_gain_s)),
        workspace_soft_boundary_joint_speed_scale=min(
            1.0,
            max(0.05, float(args.workspace_soft_boundary_joint_speed_scale)),
        ),
        workspace_hold_trigger_margin_m=max(
            0.0, float(args.workspace_hold_trigger_margin_m)),
        ball_noise_std_m=max(0.0, float(args.ball_noise_std_m)),
        ball_observation_rate_hz=min(
            240.0, max(1.0, float(args.ball_observation_rate_hz))),
        backend_control_loop=bool(args.backend_control_loop),
        backend_snapshot_rate_hz=min(
            120.0, max(1.0, float(args.backend_snapshot_rate_hz))),
        browser_step_rate_hz=min(
            120.0, max(1.0, float(args.browser_step_rate_hz))),
        display_rate_hz=min(
            240.0, max(1.0, float(args.display_rate_hz))),
        recording_output_root=args.recording_output_root.expanduser().resolve(),
        recording_control_rate_hz=min(
            500.0, max(0.0, float(args.recording_control_rate_hz))),
        recording_image_rate_hz=min(
            120.0, max(0.0, float(args.recording_image_rate_hz))),
        recording_analysis_video_fps=min(
            120.0, max(1.0, float(args.recording_analysis_video_fps))),
        recording_queue_size=max(8, int(args.recording_queue_size)),
        enable_ball_state_estimator=not bool(args.disable_ball_state_estimator),
        ball_estimator_enable_prediction=bool(
            args.ball_estimator_prediction),
        ball_estimator_enable_aero_drag=not bool(
            args.disable_ball_estimator_aero_drag),
        ball_estimator_max_extrapolation_s=max(
            0.0, float(args.ball_estimator_max_extrapolation_s)),
        ball_estimator_control_max_extrapolation_s=float(
            args.ball_estimator_control_max_extrapolation_s),
        ball_estimator_control_output_mode=str(
            args.ball_estimator_control_output_mode),
        ball_estimator_frame_hold_output_rate_hz=max(
            1e-6, float(args.ball_estimator_frame_hold_output_rate_hz)),
        ball_estimator_release_gated_dynamics=bool(
            args.ball_estimator_release_gated_dynamics),
        ball_estimator_trust_loss_s=max(
            0.0, float(args.ball_estimator_trust_loss_s)),
        ball_estimator_degrade_loss_s=max(
            0.0, float(args.ball_estimator_degrade_loss_s)),
        ball_estimator_invalid_after_s=max(
            0.0, float(args.ball_estimator_invalid_after_s)),
        ball_estimator_prediction_horizon_s=max(
            0.0, float(args.ball_estimator_prediction_horizon_s)),
        ball_estimator_q_pos_mm2_s=max(
            0.0, float(args.ball_estimator_q_pos_mm2_s)),
        ball_estimator_q_vel_mult=max(
            0.0, float(args.ball_estimator_q_vel_mult)),
        ball_estimator_q_depth_pos_mult=max(
            1e-6, float(args.ball_estimator_q_depth_pos_mult)),
        ball_estimator_q_depth_vel_mult=max(
            1e-6, float(args.ball_estimator_q_depth_vel_mult)),
        ball_estimator_r_lateral_mm=max(
            1.0, float(args.ball_estimator_r_lateral_mm)),
        ball_estimator_r_depth_mm=max(
            1.0, float(args.ball_estimator_r_depth_mm)),
        ros_bridge=bool(args.ros_bridge),
        ros_publish_commands=bool(args.ros_bridge and not args.disable_command_publish),
        command_topic=str(args.command_topic),
        joint_state_topic=str(args.joint_state_topic),
        cart_state_topic=str(args.cart_state_topic),
        right_arm_mu_idx=int(args.right_arm_mu_idx),
        enable_camera_ball=bool(args.camera_ball),
        camera_source_mode=str(args.camera_source_mode),
        use_hybrid_detector=not bool(args.disable_hybrid_detector),
        use_depth_sensor=not bool(args.disable_depth_sensor),
        image_width=max(1, int(args.image_width)),
        image_height=max(1, int(args.image_height)),
        realsense_fps=max(1, int(args.realsense_fps)),
        camera_image_topic=str(args.camera_image_topic),
        camera_raw_image_topic=str(args.camera_raw_image_topic),
        camera_image_transport=str(args.camera_image_transport),
        camera_info_topic=str(args.camera_info_topic),
        calibration_json_path=str(args.calibration_json_path),
        yolo_model_path=yolo_model_path,
        target_class=str(args.target_class),
        conf_threshold=float(args.conf_threshold),
        ir_params_json=str(args.ir_params_json),
        yolo_interval=max(1, int(args.yolo_interval)),
        ir_threshold=max(0, int(args.ir_threshold)),
        use_adaptive_ir=not bool(args.disable_adaptive_ir),
        ir_exposure=(
            int(args.ir_exposure) if int(args.ir_exposure) > 0 else None),
        ir_gain=(
            int(args.ir_gain) if int(args.ir_gain) > 0 else None),
        ir_laser_power=max(0, int(args.ir_laser_power)),
        ir_min_radius=max(0, int(args.ir_min_radius)),
        ir_max_radius=max(0, int(args.ir_max_radius)),
        ir_circularity_min=max(0.0, float(args.ir_circularity_min)),
        ir_brightness_percentile=max(0, int(args.ir_brightness_percentile)),
        hybrid_roi_margin=max(0, int(args.hybrid_roi_margin)),
        depth_correction_model_path=str(args.depth_correction_model_path),
        depth_neighborhood_px=max(0, int(args.depth_neighborhood_px)),
        infer_imgsz=max(1, int(args.infer_imgsz)),
        detector_device=str(args.detector_device),
        detector_half=bool(args.detector_half),
        use_kalman=bool(args.use_kalman),
        camera_tilt_deg=float(args.camera_tilt_deg),
        use_undistort=not bool(args.disable_undistort),
        enable_ellipse_fit=not bool(args.disable_ellipse_fit),
        enable_temporal_tracker=bool(args.enable_temporal_tracker),
        tracker_max_miss_bridge=max(0, int(args.tracker_max_miss_bridge)),
        tracker_gate_px=max(0.0, float(args.tracker_gate_px)),
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
        color_match_threshold=max(0.0, float(args.color_match_threshold)),
        color_circularity_min=max(0.0, float(args.color_circularity_min)),
        enable_camera_detection=not bool(args.disable_camera_detection),
        enable_camera_visualization=not bool(args.disable_camera_visualization),
        camera_visualization_rate_hz=min(
            120.0, max(1.0, float(args.camera_visualization_rate_hz))),
        camera_web_max_width=max(0, int(args.camera_web_max_width)),
        camera_web_jpeg_quality=min(
            95, max(25, int(args.camera_web_jpeg_quality))),
        enable_camera_color_filter=not bool(args.disable_camera_color_filter),
        sim_cartesian_x_offset_m=float(args.sim_cartesian_x_offset_m),
        sim_cartesian_y_offset_m=float(args.sim_cartesian_y_offset_m),
        sim_base_height_m=float(args.sim_base_height_m),
        enable_sim_mirror=bool(args.enable_sim_mirror),
        sim_mirror_default_enabled=bool(
            args.enable_sim_mirror and not args.sim_mirror_default_disabled),
        sim_subprocess=not bool(args.disable_sim_subprocess),
        sim_backend=str(args.sim_backend),
        sim_xml_path=sim_xml,
        sim_enable_video_frames=bool(args.enable_sim_video),
        sim_frame_rate_hz=min(120.0, max(1.0, float(args.sim_frame_rate_hz))),
        sim_render_width=max(320, int(args.sim_render_width)),
        sim_render_height=max(240, int(args.sim_render_height)),
        sim_use_mujoco_renderer=not bool(args.disable_sim_mujoco_renderer),
        workspace_safety_mode=_workspace_safety_mode(args.workspace_safety_mode),
    )


def _prewarm_jax_cuda_for_mjx(cfg: ProbeConfig) -> None:
    if (not bool(cfg.enable_sim_mirror)
            or bool(getattr(cfg, "sim_subprocess", False))
            or str(cfg.sim_backend).strip().lower() != "mjx"):
        return
    _prewarm_jax_cuda_backend(str(cfg.sim_backend))


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    if bool(args.control_subprocess):
        cfg.backend_control_loop = True
    if not (bool(args.control_subprocess) and not bool(args.smoke_test)):
        _prewarm_jax_cuda_for_mjx(cfg)
    probe = (
        ControlSubprocessProxy(cfg)
        if bool(args.control_subprocess) and not bool(args.smoke_test)
        else PolicySafetyProbe(cfg)
    )

    def _main_shutdown_signal(_signum, _frame) -> None:
        raise KeyboardInterrupt

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _main_shutdown_signal)
        except Exception:
            pass

    if args.smoke_test:
        try:
            ball = np.asarray(probe.initial_ball_pos(), dtype=np.float32)
            first = probe.step({"ball": ball.tolist(), "velocity": [0, 0, 0], "valid": True, "dt": 0.02})
            released_ball = ball.copy()
            released_ball[2] -= max(0.03, cfg.ball_release_min_z_drop_m * 2.0)
            released = probe.step({
                "ball": released_ball.tolist(),
                "velocity": [0, 0, -0.6],
                "valid": True,
                "dt": max(0.04, cfg.ball_release_window_s),
            })
            probe.step({"ball": ball.tolist(), "velocity": [0, 0, 0], "valid": False, "dt": 0.25})
            hidden = probe.step({"ball": ball.tolist(), "velocity": [0, 0, 0], "valid": False, "dt": 0.25})
            noisy = probe.step({
                "ball": ball.tolist(),
                "velocity": [0, 0, 0],
                "valid": True,
                "noiseStdM": max(0.003, cfg.ball_noise_std_m),
                "dt": 0.02,
            })
            print(json.dumps({
                "first_mode": first["mode"],
                "released_mode": released["mode"],
                "hidden_mode": hidden["mode"],
                "noisy_observed_ball": noisy["observedBall"],
                "ros_enabled": first["ros"]["enabled"],
                "ros_publish_commands": first["ros"]["publishCommands"],
                "anchor": first["anchor"],
                "racket": first["racket"],
                "actionAbsMax": first["actionAbsMax"],
            }, indent=2))
        finally:
            probe.shutdown()
        return 0

    httpd = None
    try:
        ProbeRequestHandler.probe = probe
        httpd = ThreadingHTTPServer((cfg.host, cfg.port), ProbeRequestHandler)
        url = f"http://{cfg.host}:{cfg.port}/"
        print("========================================================================")
        print("Interactive policy safety probe")
        print("========================================================================")
        print(f"URL: {url}")
        print(f"Model: {cfg.model_path}")
        print(f"Robot XML: {cfg.robot_xml_path}")
        if cfg.enable_sim_mirror:
            print(
                "Sim mirror: "
                f"backend={cfg.sim_backend}, xml={cfg.sim_xml_path}, "
                f"renderer={cfg.sim_use_mujoco_renderer}")
        if cfg.enable_camera_ball:
            if cfg.camera_source_mode == "direct_realsense":
                print(
                    "Camera: "
                    f"source=direct_realsense, hybrid={cfg.use_hybrid_detector}, "
                    f"depth={cfg.use_depth_sensor}, "
                    f"{cfg.image_width}x{cfg.image_height}@{cfg.realsense_fps}, "
                    f"ir={cfg.ir_params_json}")
            else:
                print(
                    "Camera: "
                    f"source=ros_topic, transport={cfg.camera_image_transport}, "
                    f"compressed={cfg.camera_image_topic}, "
                    f"raw={cfg.camera_raw_image_topic}, "
                    f"info={cfg.camera_info_topic}")
        print("Press Ctrl-C to stop.")
        print("")
        if args.open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n[probe] stopped")
    finally:
        if httpd is not None:
            httpd.server_close()
        probe.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
