#!/usr/bin/env python3
"""Collect robot-attached-ball observations without commanding the robot.

The operator moves the arm manually and presses Enter once each pose is still.
This process owns the RealSense device, subscribes only to ``/joint_states``,
and deliberately creates no ROS publisher.  Every accepted camera frame is
paired with the latest joint state, the deployment-equivalent MuJoCo racket
FK, and the ball/racket offset in both policy and racket-local coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mujoco as mj
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pingpong_controller.calibration_loader import load_calibration
from pingpong_controller.hybrid_ball_detector_adapter import (
    HybridBallDetectorAdapter,
)
from pingpong_controller.rl_policy import RIGHT_ARM_JOINTS, TARGET_DEGREES
from pingpong_controller.simcode2_xml import build_simcode2_training_xml


CURRENT_FILE = Path(__file__).resolve()
PACKAGE_ROOT = CURRENT_FILE.parents[2]
WORKSPACE_ROOT = CURRENT_FILE.parents[5]
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "outputs" / "fixed_ball_calibration"
DEFAULT_CALIBRATION = (
    PACKAGE_ROOT / "outputs" / "vision_calib"
    / "eye_hand_result_848_undistorted.json"
)
DEFAULT_YOLO_MODEL = PACKAGE_ROOT / "models" / "best.pt"
DEFAULT_DEPTH_MODEL = PACKAGE_ROOT / "depth_correction_model.pkl"
DEFAULT_IR_PARAMS = WORKSPACE_ROOT / "ir_params.json"
DEFAULT_POLICY_XML = (
    WORKSPACE_ROOT / "simcode5" / "pingpong_moz1-main"
    / "pingpong_controller" / "tools" / "rl_sim" / "moz1_pd.xml"
)
DEFAULT_WARM_RIGHT_ARM_DEG = (
    5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214,
)


def _stamp_s(stamp: Any) -> float | None:
    value = float(stamp.sec) + 1.0e-9 * float(stamp.nanosec)
    return value if value > 0.0 and math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    return str(value)


def _quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion[:] = [
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ]
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(
            1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        quaternion[:] = [
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        ]
    elif matrix[1, 1] > matrix[2, 2]:
        scale = 2.0 * math.sqrt(
            1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        quaternion[:] = [
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        ]
    else:
        scale = 2.0 * math.sqrt(
            1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        quaternion[:] = [
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


@dataclass(frozen=True)
class JointSnapshot:
    sequence: int
    receipt_wall_time_s: float
    header_stamp_s: float | None
    position_rad: np.ndarray
    velocity_rad_s: np.ndarray | None


class JointStateListener(Node):
    """Read-only right-arm JointState listener; it owns no publishers."""

    def __init__(self, topic: str):
        super().__init__("fixed_ball_bias_collector")
        self._lock = threading.Lock()
        self._latest: JointSnapshot | None = None
        self._sequence = 0
        self.create_subscription(JointState, topic, self._callback, 20)
        self.get_logger().info(
            f"Read-only joint subscription active: {topic}; no command "
            "publisher is created")

    def _callback(self, message: JointState) -> None:
        by_name = {name: index for index, name in enumerate(message.name)}
        if any(name not in by_name for name in RIGHT_ARM_JOINTS):
            return
        try:
            position = np.asarray([
                message.position[by_name[name]] for name in RIGHT_ARM_JOINTS
            ], dtype=np.float64)
        except (IndexError, TypeError, ValueError):
            return
        velocity = None
        try:
            candidate = np.asarray([
                message.velocity[by_name[name]] for name in RIGHT_ARM_JOINTS
            ], dtype=np.float64)
            if candidate.shape == (7,) and np.all(np.isfinite(candidate)):
                velocity = candidate
        except (IndexError, TypeError, ValueError):
            pass
        if position.shape != (7,) or not np.all(np.isfinite(position)):
            return
        with self._lock:
            self._sequence += 1
            self._latest = JointSnapshot(
                sequence=self._sequence,
                receipt_wall_time_s=time.time(),
                header_stamp_s=_stamp_s(message.header.stamp),
                position_rad=position.copy(),
                velocity_rad_s=(
                    None if velocity is None else velocity.copy()),
            )

    def snapshot(self) -> JointSnapshot | None:
        with self._lock:
            sample = self._latest
            if sample is None:
                return None
            return JointSnapshot(
                sequence=sample.sequence,
                receipt_wall_time_s=sample.receipt_wall_time_s,
                header_stamp_s=sample.header_stamp_s,
                position_rad=sample.position_rad.copy(),
                velocity_rad_s=(
                    None if sample.velocity_rad_s is None
                    else sample.velocity_rad_s.copy()),
            )


class DeploymentRacketFk:
    """Deployment-equivalent racket FK without loading or running a policy."""

    def __init__(
        self,
        xml_path: Path,
        warm_right_arm_deg: tuple[float, ...],
        warm_steps: int = 700,
    ):
        if len(warm_right_arm_deg) != len(RIGHT_ARM_JOINTS):
            raise ValueError("warm right-arm pose must contain seven values")
        generated_xml = build_simcode2_training_xml(xml_path)
        try:
            self.model = mj.MjModel.from_xml_path(str(generated_xml))
        finally:
            try:
                generated_xml.unlink()
            except OSError:
                pass
        self.data = mj.MjData(self.model)
        self.arm_joint_ids = np.asarray([
            self._required_id(mj.mjtObj.mjOBJ_JOINT, name)
            for name in RIGHT_ARM_JOINTS
        ], dtype=np.int32)
        self.racket_geom_id = self._required_id(
            mj.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")
        self.base_x_qadr = self._joint_qpos_address("base_x")
        self.base_y_qadr = self._joint_qpos_address("base_y")
        self.base_yaw_qadr = self._joint_qpos_address("base_yaw")
        target_degrees = dict(TARGET_DEGREES)
        target_degrees.update(dict(zip(
            RIGHT_ARM_JOINTS, warm_right_arm_deg)))
        for name, degrees in target_degrees.items():
            actuator_id = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = math.radians(float(degrees))
        for _ in range(max(0, int(warm_steps))):
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)

    def _required_id(self, object_type: Any, name: str) -> int:
        value = int(mj.mj_name2id(self.model, object_type, name))
        if value < 0:
            raise RuntimeError(f"MuJoCo model is missing {name!r}")
        return value

    def _joint_qpos_address(self, name: str) -> int:
        joint_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, name)
        return (
            -1 if joint_id < 0
            else int(self.model.jnt_qposadr[joint_id]))

    def pose(self, arm_position_rad: np.ndarray) -> dict[str, Any]:
        arm = np.asarray(arm_position_rad, dtype=np.float64).reshape(7)
        for value, joint_id in zip(arm, self.arm_joint_ids):
            address = int(self.model.jnt_qposadr[int(joint_id)])
            self.data.qpos[address] = float(value)
        mj.mj_forward(self.model, self.data)

        position_world = np.asarray(
            self.data.geom_xpos[self.racket_geom_id],
            dtype=np.float64,
        ).reshape(3)
        rotation_world = np.asarray(
            self.data.geom_xmat[self.racket_geom_id],
            dtype=np.float64,
        ).reshape(3, 3)
        if min(
            self.base_x_qadr,
            self.base_y_qadr,
            self.base_yaw_qadr,
        ) >= 0:
            base_x = float(self.data.qpos[self.base_x_qadr])
            base_y = float(self.data.qpos[self.base_y_qadr])
            yaw = float(self.data.qpos[self.base_yaw_qadr])
        else:
            base_x = 0.0
            base_y = 0.0
            yaw = 0.0
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotation_base_from_world = np.array([
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        delta_x = float(position_world[0] - base_x)
        delta_y = float(position_world[1] - base_y)
        position_base = np.array([
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
            position_world[2],
        ], dtype=np.float64)
        rotation_base = rotation_base_from_world @ rotation_world
        if rotation_base[2, 2] < 0.0:
            # Keep a right-handed frame while making local +normal point up.
            rotation_base[:, 1] *= -1.0
            rotation_base[:, 2] *= -1.0
        geom_size = np.asarray(
            self.model.geom_size[self.racket_geom_id],
            dtype=np.float64,
        ).reshape(-1)
        return {
            "centerSimM": position_base,
            "quaternionSimXyzw": _quat_xyzw(rotation_base),
            "rotationSim": rotation_base,
            "tangentUM": rotation_base[:, 0],
            "tangentVM": rotation_base[:, 1],
            "normalM": rotation_base[:, 2],
            "radiusM": float(geom_size[0]),
            "halfThicknessM": float(geom_size[1]),
        }


def _vector_stats(values: list[list[float]]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    median = np.median(array, axis=0)
    return {
        "count": int(array.shape[0]),
        "median": median,
        "mad": np.median(np.abs(array - median), axis=0),
        "p10": np.quantile(array, 0.10, axis=0),
        "p90": np.quantile(array, 0.90, axis=0),
    }


def _scalar_stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "count": int(array.size),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


class FixedBallCollector:

    WINDOW_NAME = "Fixed Ball Calibration"

    def __init__(
        self,
        args: argparse.Namespace,
        joint_listener: JointStateListener,
    ):
        self.args = args
        self.joint_listener = joint_listener
        self.visualization_enabled = bool(args.visualization)
        if (
            self.visualization_enabled
            and not os.environ.get("DISPLAY")
            and not os.environ.get("WAYLAND_DISPLAY")
        ):
            print(
                "未检测到图形桌面（DISPLAY/WAYLAND_DISPLAY），"
                "将使用终端交互。",
                flush=True,
            )
            self.visualization_enabled = False
        self._visualization_initialized = False
        self._visualization_warning_printed = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(args.output_root).expanduser().resolve() / (
            f"fixed_ball_calibration_{timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.samples_path = self.run_dir / "samples.jsonl"
        self.pose_summaries: list[dict[str, Any]] = []
        self.all_samples: list[dict[str, Any]] = []
        self.sim_offset_m = np.asarray([
            args.sim_cartesian_x_offset_m,
            args.sim_cartesian_y_offset_m,
            args.sim_base_height_m,
        ], dtype=np.float64)
        self.fk = DeploymentRacketFk(
            Path(args.policy_xml).expanduser().resolve(),
            tuple(args.warm_right_arm_deg),
        )
        calibration_path = Path(args.calibration_json).expanduser().resolve()
        calibration = load_calibration(str(calibration_path))
        self.calibration = calibration
        ir_params_path = Path(args.ir_params_json).expanduser().resolve()
        ir_params = str(ir_params_path) if ir_params_path.exists() else None
        self.detector_adapter = HybridBallDetectorAdapter(
            yolo_model_path=str(Path(args.yolo_model).expanduser().resolve()),
            ball_diameter_mm=40.0,
            width=int(calibration.image_width),
            height=int(calibration.image_height),
            fps=int(args.realsense_fps),
            yolo_interval=1,
            yolo_conf=0.25,
            target_class="Class 0",
            ir_params_json=ir_params,
            ir_threshold=int(args.ir_threshold),
            use_adaptive=bool(args.use_adaptive_ir),
            min_radius=int(args.ir_min_radius),
            max_radius=int(args.ir_max_radius),
            circularity_min=float(args.ir_circularity_min),
            laser_power=int(args.ir_laser_power),
            exposure=args.ir_exposure,
            gain=args.ir_gain,
            brightness_percentile=int(args.ir_brightness_percentile),
            roi_margin=int(args.hybrid_roi_margin),
            calibration_data=calibration,
            output_frame="base_link",
            print_every=999999,
            depth_correction_model_path=str(
                Path(args.depth_correction_model).expanduser().resolve()),
            depth_neighborhood_px=int(args.depth_neighborhood_px),
            depth_mode=str(args.depth_mode),
            use_kalman=False,
            use_ir_depth_kalman=False,
            use_synchronized_depth=False,
            depth_temporal_window_frames=int(args.depth_temporal_window_frames),
            velocity_least_squares_enabled=False,
            velocity_window_frames=3,
            clear_buffers_on_detection_loss_enabled=False,
            motion_consistency_gate_enabled=False,
            physics_constrained_velocity_enabled=False,
            trajectory_position_smoothing_enabled=False,
            impact_velocity_reset_enabled=False,
            candidate_filter_enabled=False,
        )
        self.detector = self.detector_adapter.detector
        self.detector.configure_yolo(False)
        self.detector.impact_racket_gate_required = True
        self._write_metadata(calibration_path, ir_params_path)

    def _write_metadata(
        self,
        calibration_path: Path,
        ir_params_path: Path,
    ) -> None:
        metadata = {
            "schemaVersion": 1,
            "startedLocalTime": datetime.now().isoformat(),
            "argv": sys.argv,
            "purpose": "fixed_ball_multi_pose_bias_calibration",
            "safety": {
                "rosPublishersCreated": 0,
                "commandTopicsPublished": [],
                "robotMotion": "manual_operator_control_only",
            },
            "capture": {
                "requestedPoses": int(self.args.poses),
                "warmupS": float(self.args.warmup_s),
                "durationPerPoseS": float(self.args.duration_s),
                "visualizationRequested": bool(self.args.visualization),
                "visualizationEnabled": bool(
                    self.visualization_enabled),
                "jointTopic": str(self.args.joint_topic),
                "maxJointAgeS": float(self.args.max_joint_age_s),
                "maxJointSpeedRadS": float(self.args.max_joint_speed_rad_s),
                "rawFrameStride": int(self.args.raw_frame_stride),
            },
            "frames": {
                "detectorOutputFrame": "base_link",
                "policyFrame": "sim_frame",
                "simFrameTranslationFromBaseLinkM": self.sim_offset_m,
                "rotationFromBaseLinkToSimFrame": "identity",
            },
            "detector": {
                "calibrationJson": str(calibration_path),
                "yoloModel": str(Path(self.args.yolo_model).resolve()),
                "irParamsJson": (
                    str(ir_params_path) if ir_params_path.exists() else None),
                "depthCorrectionModel": str(
                    Path(self.args.depth_correction_model).resolve()),
                "depthMode": str(self.args.depth_mode),
                "synchronizedIrDepth": False,
                "depthTemporalWindowFrames": int(
                    self.args.depth_temporal_window_frames),
                "pixelKalman": False,
                "irDepthKalman": False,
                "trajectoryPositionSmoothing": False,
                "geometryDepthUsedForOutput": (
                    str(self.args.depth_mode) == "geometry"),
                "effectiveIrParams": dict(self.detector.ir_params),
                "effectiveIrExposure": self.detector.ir_exposure,
                "effectiveIrGain": self.detector.ir_gain,
                "effectiveBrightnessPercentile": (
                    self.detector.brightness_percentile),
                "irIntrinsics": {
                    "fx": self.detector.ir_fx,
                    "fy": self.detector.ir_fy,
                    "cx": self.detector.ir_cx,
                    "cy": self.detector.ir_cy,
                },
                "T_ir_to_rgb": self.detector.T_ir_to_rgb,
                "calibrationSnapshot": {
                    "imageWidth": self.calibration.image_width,
                    "imageHeight": self.calibration.image_height,
                    "cameraMatrix": self.calibration.camera_matrix,
                    "distCoeffs": self.calibration.dist_coeffs,
                    "T_base_camera": self.calibration.T_base_camera,
                },
                "sensorDepthSemantic": (
                    "near-surface percentile used as an uncorrected "
                    "ball-centre proxy"),
            },
            "racketFk": {
                "policyXml": str(Path(self.args.policy_xml).resolve()),
                "warmSteps": 700,
                "warmRightArmDeg": list(self.args.warm_right_arm_deg),
                "siteOrGeom": "racket_rubber_fore",
            },
        }
        with (self.run_dir / "metadata.json").open("w") as file:
            json.dump(_json_safe(metadata), file, indent=2, ensure_ascii=False)

    def close(self) -> None:
        if self._visualization_initialized:
            try:
                cv2.destroyWindow(self.WINDOW_NAME)
                cv2.waitKey(1)
            except cv2.error:
                pass
        self.detector_adapter.close()

    def wait_for_joint_state(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.joint_listener.snapshot() is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"No complete right-arm JointState received from "
            f"{self.args.joint_topic!r} within {timeout_s:.1f} s")

    def _observe_once(
        self,
    ) -> tuple[Any, float, JointSnapshot | None, dict[str, Any]]:
        frames = self.detector.pipeline.wait_for_frames(timeout_ms=1500)
        observation_time = time.time()
        joint = self.joint_listener.snapshot()
        if joint is not None:
            racket = self.fk.pose(joint.position_rad)
            racket_base = (
                np.asarray(racket["centerSimM"], dtype=np.float64)
                - self.sim_offset_m)
            self.detector.set_racket_state(racket_base, observation_time)
        else:
            self.detector.clear_racket_state()
        observation = self.detector.get_observation(
            observation_time=observation_time,
            frames=frames,
        )
        return frames, observation_time, joint, observation

    def _show_visualization(
        self,
        frames: Any,
        observation: dict[str, Any],
        pose_index: int,
        phase: str,
        sample_index: int = 0,
        accepted_count: int = 0,
    ) -> int:
        if not self.visualization_enabled:
            return -1
        try:
            if not self._visualization_initialized:
                cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.WINDOW_NAME, 1440, 620)
                self._visualization_initialized = True
            ir_frame = frames.get_infrared_frame(1)
            if not ir_frame:
                return cv2.waitKey(1) & 0xFF
            ir_image = np.asanyarray(ir_frame.get_data())
            rgb_image = getattr(self.detector, "latest_color_image", None)
            if rgb_image is None:
                rgb_image = self.detector.color_image_for_frames(
                    frames, ir_image)
            display = self.detector._render_visualization(
                np.asarray(rgb_image).copy(),
                ir_image.copy(),
                observation,
            )
            detected = bool(observation.get("detected", False))
            status = "DETECTED" if detected else "NOT DETECTED"
            status_color = (80, 220, 80) if detected else (80, 80, 255)
            header = np.zeros((62, display.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                header,
                f"Pose {pose_index:02d}/{int(self.args.poses):02d} | "
                f"{phase} | {status}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                status_color,
                2,
                cv2.LINE_AA,
            )
            raw = observation.get("raw_measurement")
            details = (
                "Space/Enter: capture   Q/Esc: finish"
                if phase == "PREVIEW"
                else f"accepted={accepted_count}/{sample_index}")
            if isinstance(raw, dict):
                cx = raw.get("cx")
                cy = raw.get("cy")
                depth = raw.get("sensor_depth_mm")
                if cx is not None and cy is not None and depth is not None:
                    details = (
                        f"IR=({float(cx):.1f}, {float(cy):.1f}) px  "
                        f"depth={float(depth):.0f} mm  "
                        f"accepted={accepted_count}/{sample_index}")
            cv2.putText(
                header,
                details,
                (12, 51),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(
                self.WINDOW_NAME,
                np.concatenate((header, display), axis=0),
            )
            return cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            if not self._visualization_warning_printed:
                print(
                    "可视化窗口不可用，将退回终端交互："
                    f"{exc}",
                    flush=True,
                )
                self._visualization_warning_printed = True
            self.visualization_enabled = False
            return -1

    def wait_for_pose_trigger(self, pose_index: int) -> str:
        if not self.visualization_enabled:
            return input(
                "稳定后按 Enter 开始采集；输入 q 再按 Enter 提前结束：")
        print(
            "检测窗口会持续刷新：在窗口中按 Space/Enter 开始采集，"
            "按 q/Esc 提前结束；也可在此终端输入后按 Enter。",
            flush=True,
        )
        while True:
            frames, _, _, observation = self._observe_once()
            key = self._show_visualization(
                frames,
                observation,
                pose_index,
                "PREVIEW",
            )
            if key in (ord("q"), ord("Q"), 27):
                return "q"
            if key in (10, 13, 32):
                return ""
            if not self.visualization_enabled:
                return input(
                    "稳定后按 Enter 开始采集；输入 q 再按 Enter 提前结束：")
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            except (OSError, ValueError):
                readable = []
            if readable:
                line = sys.stdin.readline()
                return "q" if line == "" else line.strip()

    def _save_frame_assets(
        self,
        pose_dir: Path,
        sample_index: int,
        frames: Any,
        observation: dict[str, Any],
    ) -> dict[str, str]:
        prefix = f"frame_{sample_index:04d}"
        assets: dict[str, str] = {}
        ir_frame = frames.get_infrared_frame(1)
        depth_frame = frames.get_depth_frame()
        ir_image = (
            None if not ir_frame
            else np.asanyarray(ir_frame.get_data()).copy())
        depth_image = (
            None if not depth_frame
            else np.asanyarray(depth_frame.get_data()).copy())
        rgb_image = getattr(self.detector, "latest_color_image", None)
        if rgb_image is not None:
            rgb_image = np.asarray(rgb_image).copy()

        def save(name: str, image: np.ndarray | None) -> None:
            if image is None:
                return
            path = pose_dir / f"{prefix}_{name}"
            if cv2.imwrite(str(path), image):
                assets[name] = str(path.relative_to(self.run_dir))

        save("rgb.jpg", rgb_image)
        save("ir.png", ir_image)
        save("depth.png", depth_image)
        if rgb_image is not None and ir_image is not None:
            try:
                annotated = self.detector._render_visualization(
                    rgb_image.copy(), ir_image.copy(), observation)
                save("annotated.jpg", annotated)
            except Exception as exc:
                assets["annotationError"] = str(exc)
        return assets

    def _sample_row(
        self,
        pose_index: int,
        sample_index: int,
        wall_time_s: float,
        observation: dict[str, Any],
        joint: JointSnapshot | None,
    ) -> dict[str, Any]:
        detected = bool(observation.get("detected", False))
        rejection_reasons: list[str] = []
        fk_payload = None
        joint_age_s = None
        joint_speed_rad_s = None
        if joint is None:
            rejection_reasons.append("joint_state_unavailable")
        else:
            joint_age_s = max(
                0.0, float(wall_time_s - joint.receipt_wall_time_s))
            if joint_age_s > float(self.args.max_joint_age_s):
                rejection_reasons.append("joint_state_stale")
            if joint.velocity_rad_s is None:
                rejection_reasons.append("joint_velocity_unavailable")
            else:
                joint_speed_rad_s = float(np.linalg.norm(
                    joint.velocity_rad_s))
                if float(np.max(np.abs(joint.velocity_rad_s))) > float(
                        self.args.max_joint_speed_rad_s):
                    rejection_reasons.append("robot_not_stationary")
            fk_payload = self.fk.pose(joint.position_rad)

        if not detected:
            rejection_reasons.append("ball_not_detected")
        raw = observation.get("raw_measurement")
        if detected and isinstance(raw, dict):
            if not bool(raw.get("measurement_trusted_3d", True)):
                rejection_reasons.append("measurement_not_trusted_3d")

        detector_position_mm = None
        detector_velocity_mm_s = None
        ball_sim_m = None
        ball_minus_racket_sim_mm = None
        racket_frame_offset_mm = None
        tangent_distance_mm = None
        surface_normal_gap_mm = None
        if detected:
            detector_position_mm = np.asarray(
                observation.get("position"), dtype=np.float64).reshape(3)
            detector_velocity_mm_s = np.asarray(
                observation.get("velocity"), dtype=np.float64).reshape(3)
        if detector_position_mm is not None and fk_payload is not None:
            ball_sim_m = detector_position_mm * 1.0e-3 + self.sim_offset_m
            racket_center = np.asarray(
                fk_payload["centerSimM"], dtype=np.float64)
            rotation = np.asarray(
                fk_payload["rotationSim"], dtype=np.float64)
            delta_m = ball_sim_m - racket_center
            local_mm = rotation.T @ delta_m * 1000.0
            ball_minus_racket_sim_mm = delta_m * 1000.0
            racket_frame_offset_mm = local_mm
            tangent_distance_mm = float(np.linalg.norm(local_mm[:2]))
            surface_normal_gap_mm = float(
                local_mm[2] - 1000.0 * float(
                    fk_payload["halfThicknessM"]))

        racket_record = None
        if fk_payload is not None:
            racket_record = dict(fk_payload)
            racket_record["centerDetectorBaseLinkM"] = (
                np.asarray(fk_payload["centerSimM"], dtype=np.float64)
                - self.sim_offset_m)
            racket_record.pop("rotationSim", None)
        return {
            "schemaVersion": 1,
            "poseIndex": int(pose_index),
            "sampleIndex": int(sample_index),
            "wallTimeS": float(wall_time_s),
            "detected": detected,
            "calibrationAccepted": not rejection_reasons,
            "rejectionReasons": rejection_reasons,
            "detectorOutputFrame": "base_link",
            "detectorPositionMm": detector_position_mm,
            "detectorVelocityMmS": detector_velocity_mm_s,
            "ballPositionSimM": ball_sim_m,
            "rawMeasurement": raw,
            "jointState": (
                None if joint is None else {
                    "sequence": int(joint.sequence),
                    "receiptWallTimeS": float(joint.receipt_wall_time_s),
                    "headerStampS": joint.header_stamp_s,
                    "ageS": joint_age_s,
                    "names": list(RIGHT_ARM_JOINTS),
                    "positionRad": joint.position_rad,
                    "velocityRadS": joint.velocity_rad_s,
                    "speedNormRadS": joint_speed_rad_s,
                }),
            "racket": racket_record,
            "ballMinusRacketSimMm": ball_minus_racket_sim_mm,
            "racketFrameOffsetMm": racket_frame_offset_mm,
            "tangentDistanceMm": tangent_distance_mm,
            "surfaceNormalGapMm": surface_normal_gap_mm,
        }

    def capture_pose(self, pose_index: int) -> dict[str, Any]:
        pose_dir = self.run_dir / f"pose_{pose_index:02d}"
        pose_dir.mkdir(parents=True, exist_ok=False)
        capture_start = time.monotonic()
        record_start = capture_start + float(self.args.warmup_s)
        capture_end = record_start + float(self.args.duration_s)
        rows: list[dict[str, Any]] = []
        accepted_count = 0
        sample_index = 0
        with self.samples_path.open("a") as samples_file:
            while time.monotonic() < capture_end:
                frames, observation_time, joint, observation = (
                    self._observe_once())
                if time.monotonic() < record_start:
                    self._show_visualization(
                        frames,
                        observation,
                        pose_index,
                        "WARMUP",
                    )
                    continue
                sample_index += 1
                row = self._sample_row(
                    pose_index,
                    sample_index,
                    observation_time,
                    observation,
                    joint,
                )
                if row["calibrationAccepted"]:
                    accepted_count += 1
                    stride = max(1, int(self.args.raw_frame_stride))
                    if accepted_count == 1 or accepted_count % stride == 0:
                        row["frameAssets"] = self._save_frame_assets(
                            pose_dir, sample_index, frames, observation)
                self._show_visualization(
                    frames,
                    observation,
                    pose_index,
                    "CAPTURE",
                    sample_index,
                    accepted_count,
                )
                safe_row = _json_safe(row)
                rows.append(safe_row)
                self.all_samples.append(safe_row)
                samples_file.write(
                    json.dumps(safe_row, ensure_ascii=False) + "\n")
            samples_file.flush()
        summary = self._summarize_pose(pose_index, rows)
        with (pose_dir / "summary.json").open("w") as file:
            json.dump(
                _json_safe(summary), file, indent=2, ensure_ascii=False)
        self.pose_summaries.append(_json_safe(summary))
        return summary

    @staticmethod
    def _summarize_pose(
        pose_index: int,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        accepted = [row for row in rows if row.get("calibrationAccepted")]

        def vectors(key: str) -> list[list[float]]:
            return [row[key] for row in accepted if row.get(key) is not None]

        def raw_scalars(key: str) -> list[float]:
            return [
                float(row["rawMeasurement"][key])
                for row in accepted
                if isinstance(row.get("rawMeasurement"), dict)
                and row["rawMeasurement"].get(key) is not None
            ]

        reasons: dict[str, int] = {}
        for row in rows:
            for reason in row.get("rejectionReasons", []):
                reasons[reason] = reasons.get(reason, 0) + 1
        summary = {
            "poseIndex": int(pose_index),
            "sampleCount": len(rows),
            "detectedCount": sum(bool(row.get("detected")) for row in rows),
            "acceptedCount": len(accepted),
            "detectionRate": (
                0.0 if not rows
                else sum(bool(row.get("detected")) for row in rows)
                / len(rows)),
            "acceptedRate": (
                0.0 if not rows else len(accepted) / len(rows)),
            "rejectionReasonCounts": reasons,
            "ballMinusRacketSimMm": _vector_stats(
                vectors("ballMinusRacketSimMm")),
            "racketFrameOffsetMm": _vector_stats(
                vectors("racketFrameOffsetMm")),
            "tangentDistanceMm": _scalar_stats([
                float(row["tangentDistanceMm"])
                for row in accepted
                if row.get("tangentDistanceMm") is not None
            ]),
            "surfaceNormalGapMm": _scalar_stats([
                float(row["surfaceNormalGapMm"])
                for row in accepted
                if row.get("surfaceNormalGapMm") is not None
            ]),
            "sensorDepthMm": _scalar_stats(raw_scalars("sensor_depth_mm")),
            "geometryDepthMm": _scalar_stats(raw_scalars("geom_depth_mm")),
            "irCxPx": _scalar_stats(raw_scalars("cx")),
            "irCyPx": _scalar_stats(raw_scalars("cy")),
            "irMorphologyRadiusPx": _scalar_stats(raw_scalars("radius")),
        }
        racket_centers = [
            row["racket"]["centerSimM"]
            for row in accepted
            if isinstance(row.get("racket"), dict)
        ]
        racket_normals = [
            row["racket"]["normalM"]
            for row in accepted
            if isinstance(row.get("racket"), dict)
        ]
        summary["racketCenterSimM"] = _vector_stats(racket_centers)
        summary["racketNormalSim"] = _vector_stats(racket_normals)
        return summary

    def finish(self) -> dict[str, Any]:
        centers = []
        normals = []
        for summary in self.pose_summaries:
            center_stats = summary.get("racketCenterSimM")
            normal_stats = summary.get("racketNormalSim")
            if center_stats and normal_stats:
                centers.append(np.asarray(
                    center_stats["median"], dtype=np.float64))
                normal = np.asarray(
                    normal_stats["median"], dtype=np.float64)
                normals.append(normal / max(np.linalg.norm(normal), 1.0e-12))
        maximum_translation_m = 0.0
        maximum_normal_angle_deg = 0.0
        for first in range(len(centers)):
            for second in range(first + 1, len(centers)):
                maximum_translation_m = max(
                    maximum_translation_m,
                    float(np.linalg.norm(centers[first] - centers[second])))
                cosine = float(np.clip(
                    np.dot(normals[first], normals[second]), -1.0, 1.0))
                maximum_normal_angle_deg = max(
                    maximum_normal_angle_deg,
                    math.degrees(math.acos(cosine)))
        accepted = [
            row for row in self.all_samples
            if row.get("calibrationAccepted")
        ]
        warnings = []
        if len(self.pose_summaries) < 3:
            warnings.append("fewer_than_three_completed_poses")
        if maximum_translation_m < 0.05:
            warnings.append("pose_translation_span_below_50_mm")
        if maximum_normal_angle_deg < 15.0:
            warnings.append("pose_normal_span_below_15_deg")
        for summary in self.pose_summaries:
            if int(summary.get("acceptedCount", 0)) < int(
                    self.args.min_accepted_samples):
                warnings.append(
                    f"pose_{int(summary['poseIndex']):02d}_too_few_samples")
        summary = {
            "schemaVersion": 1,
            "completedLocalTime": datetime.now().isoformat(),
            "runDir": str(self.run_dir),
            "completedPoses": len(self.pose_summaries),
            "allSampleCount": len(self.all_samples),
            "acceptedSampleCount": len(accepted),
            "poseDiversity": {
                "maximumRacketCenterSeparationM": maximum_translation_m,
                "maximumRacketNormalAngleDeg": maximum_normal_angle_deg,
            },
            "warnings": warnings,
            "poses": self.pose_summaries,
        }
        with (self.run_dir / "summary.json").open("w") as file:
            json.dump(
                _json_safe(summary), file, indent=2, ensure_ascii=False)
        return summary


def _seven_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in text.split(","))
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(
            "expected seven finite comma-separated values")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect 3-30 manually positioned, stationary fixed-ball poses. "
            "This tool never publishes a robot command."))
    parser.add_argument("--poses", type=int, choices=range(3, 31), default=5)
    parser.add_argument(
        "--visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the live RGB/IR detection window (default: enabled)",
    )
    parser.add_argument("--warmup-s", type=float, default=0.6)
    parser.add_argument("--duration-s", type=float, default=2.5)
    parser.add_argument("--min-accepted-samples", type=int, default=120)
    parser.add_argument("--raw-frame-stride", type=int, default=45)
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--max-joint-age-s", type=float, default=0.10)
    parser.add_argument("--max-joint-speed-rad-s", type=float, default=0.03)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--calibration-json", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--yolo-model", default=str(DEFAULT_YOLO_MODEL))
    parser.add_argument("--ir-params-json", default=str(DEFAULT_IR_PARAMS))
    parser.add_argument(
        "--depth-correction-model", default=str(DEFAULT_DEPTH_MODEL))
    parser.add_argument("--policy-xml", default=str(DEFAULT_POLICY_XML))
    parser.add_argument(
        "--warm-right-arm-deg",
        type=_seven_floats,
        default=DEFAULT_WARM_RIGHT_ARM_DEG,
    )
    parser.add_argument("--realsense-fps", type=int, default=90)
    parser.add_argument(
        "--depth-mode", choices=("sensor", "adaptive", "geometry"),
        default="adaptive")
    parser.add_argument("--depth-temporal-window-frames", type=int, default=3)
    parser.add_argument("--depth-neighborhood-px", type=int, default=2)
    parser.add_argument("--ir-threshold", type=int, default=170)
    parser.add_argument(
        "--use-adaptive-ir", action=argparse.BooleanOptionalAction,
        default=True)
    parser.add_argument("--ir-exposure", type=int, default=1000)
    parser.add_argument("--ir-gain", type=int, default=16)
    parser.add_argument("--ir-laser-power", type=int, default=360)
    parser.add_argument("--ir-min-radius", type=int, default=15)
    parser.add_argument("--ir-max-radius", type=int, default=60)
    parser.add_argument("--ir-circularity-min", type=float, default=0.7)
    parser.add_argument("--ir-brightness-percentile", type=int, default=25)
    parser.add_argument("--hybrid-roi-margin", type=int, default=100)
    parser.add_argument("--sim-cartesian-x-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--sim-cartesian-y-offset-m", type=float, default=0.023583)
    parser.add_argument("--sim-base-height-m", type=float, default=1.302396)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if args.warmup_s < 0.0 or args.duration_s <= 0.0:
        raise SystemExit("warmup must be >= 0 and duration must be > 0")
    rclpy.init()
    listener = JointStateListener(args.joint_topic)
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(listener,), daemon=True)
    spin_thread.start()
    collector = None
    try:
        print("\n安全说明：本工具只订阅 joint_states，不创建命令发布器。")
        print("请先停止占用 RealSense 的其它程序，再用示教/手动方式移动机械臂。")
        collector = FixedBallCollector(args, listener)
        collector.wait_for_joint_state()
        print(f"数据目录：{collector.run_dir}")
        completed = 0
        for pose_index in range(1, int(args.poses) + 1):
            print(
                f"\n姿态 {pose_index}/{args.poses}：移动机械臂并保持完全静止。\n"
                "建议各姿态同时改变拍面位置和倾角；球必须始终固定在几何中心。")
            answer = collector.wait_for_pose_trigger(pose_index)
            if answer.strip().lower() == "q":
                break
            print(
                f"先丢弃 {args.warmup_s:.1f}s 暖机帧，再记录 "
                f"{args.duration_s:.1f}s……")
            pose_summary = collector.capture_pose(pose_index)
            completed += 1
            local = pose_summary.get("racketFrameOffsetMm") or {}
            print(
                f"姿态 {pose_index} 完成：accepted="
                f"{pose_summary['acceptedCount']}/"
                f"{pose_summary['sampleCount']}，局部[u,v,n]中位数(mm)="
                f"{local.get('median')}")
        summary = collector.finish()
        print(f"\n采集完成：{completed} 个姿态")
        print(f"汇总：{collector.run_dir / 'summary.json'}")
        if summary["warnings"]:
            print("质量提示：" + ", ".join(summary["warnings"]))
        return 0 if completed >= 3 else 2
    except KeyboardInterrupt:
        print("\n用户中断；正在保存已完成姿态。")
        if collector is not None:
            collector.finish()
        return 130
    finally:
        if collector is not None:
            collector.close()
        listener.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
