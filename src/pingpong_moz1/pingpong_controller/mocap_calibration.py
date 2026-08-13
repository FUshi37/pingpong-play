#!/usr/bin/env python3
"""
Motion capture calibration: mocap frame -> robot frame.

Typical workflow for ping-pong data collection:
  1. Rigidly mount a K-shaped body anywhere on the paddle.
  2. Move the right arm through diverse poses under the mocap volume.
  3. At each pose, record:
       - K rigid body pose in mocap-world coordinates (UDP)
       - right_ee_site pose in waist03 coordinates (RightArmFK)
  4. Solve the eye-to-hand equation with the full SE(3) poses.
  5. Use T_robot_mocap to transform the ping-pong ball marker into the robot
     frame during runtime (see mocap_ball_state_node / pingpong_node).

Usage (K rigid body on paddle center):
    ros2 run pingpong_controller mocap_calibration -- \\
        --calib-target rigid_body \\
        --rigid-body-id 1 \\
        --robot-source fk \\
        --udp-port 9870 \\
        --robot-xml src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml \\
        --output mocap_calibration.txt

The solver estimates both ``T_waist03_mocap`` and the fixed mounting transform
``T_right_ee_site_K``.  Therefore the K rigid body does not need to be at the
paddle center, but its UDP packet must include ``qw/qx/qy/qz``.

Legacy ball-on-EE calibration still works with --calib-target marker.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    from mc_core_interface.msg import MechUnitStateArray
except ImportError:
    MechUnitStateArray = None

try:
    from pingpong_replay.cartesian_fk import RightArmFK, RIGHT_ARM_JOINTS
except ImportError:
    print('ERROR: Cannot import RightArmFK. Make sure pingpong_replay is built.')
    sys.exit(1)

try:
    from pingpong_controller.mocap_ball_detector import parse_mocap_packet
    from pingpong_controller.mocap_transform import (
        apply_tool_offset,
        average_quaternions_wxyz,
        evaluate_point_pairs,
        fit_eye_to_hand_transform,
        fit_rigid_transform,
        load_calibration_matrix,
        pose_wxyz_to_matrix,
        pose_msg_to_matrix,
        quaternion_spread_deg,
        save_calibration_json,
        save_calibration_matrix,
    )
except ImportError:
    from mocap_ball_detector import parse_mocap_packet
    from mocap_transform import (
        apply_tool_offset,
        average_quaternions_wxyz,
        evaluate_point_pairs,
        fit_eye_to_hand_transform,
        fit_rigid_transform,
        load_calibration_matrix,
        pose_wxyz_to_matrix,
        pose_msg_to_matrix,
        quaternion_spread_deg,
        save_calibration_json,
        save_calibration_matrix,
    )

RIGHT_ARM_MU_IDX = 3
DEFAULT_STABILITY_WINDOW_S = 0.1
DEFAULT_MAX_MOCAP_STD_M = 0.00025
DEFAULT_MAX_MOCAP_ROTATION_RMS_DEG = 0.20
DEFAULT_MAX_MOCAP_MEAN_ERROR_M = 0.001
MIN_STABILITY_PACKETS = 3


@dataclass
class CalibrationSample:
    index: int
    mocap_m: np.ndarray
    robot_m: np.ndarray
    mocap_orientation_wxyz: Optional[np.ndarray] = None
    T_mocap_rigid: Optional[np.ndarray] = None
    T_waist03_ee: Optional[np.ndarray] = None
    mocap_std_m: Optional[float] = None
    mocap_rotation_rms_deg: Optional[float] = None
    mocap_rotation_max_deg: Optional[float] = None
    mocap_mean_error_mean_m: Optional[float] = None
    mocap_mean_error_max_m: Optional[float] = None
    mocap_timestamp_s: Optional[float] = None
    mocap_frame_idx: Optional[int] = None
    mocap_packet_count: int = 0
    discarded_packet_count: int = 0
    joint_positions_rad: Optional[np.ndarray] = None
    robot_source: str = 'fk'
    note: str = ''


@dataclass
class MocapCalibrationMeasurement:
    position_m: np.ndarray
    orientation_wxyz: Optional[np.ndarray]
    timestamp_s: Optional[float]
    frame_idx: Optional[int]
    mean_error_m: Optional[float]


@dataclass
class StableMocapMeasurement:
    position_m: np.ndarray
    orientation_wxyz: Optional[np.ndarray]
    position_std_m: float
    rotation_rms_deg: Optional[float]
    rotation_max_deg: Optional[float]
    mean_error_mean_m: Optional[float]
    mean_error_max_m: Optional[float]
    timestamp_s: Optional[float]
    frame_idx: Optional[int]
    packet_count: int
    discarded_packet_count: int


@dataclass
class CalibrationSession:
    mocap_points: List[np.ndarray] = field(default_factory=list)
    robot_points: List[np.ndarray] = field(default_factory=list)
    mocap_poses: List[np.ndarray] = field(default_factory=list)
    robot_poses: List[np.ndarray] = field(default_factory=list)
    samples: List[CalibrationSample] = field(default_factory=list)


class MocapCalibrationNode(Node):
    """Collect paired mocap/robot samples and solve T_robot_mocap."""

    def __init__(self,
                 udp_port: int,
                 robot_xml: str,
                 calib_target: str = 'rigid_body',
                 marker_id: int = 1,
                 rigid_body_slot: Optional[int] = None,
                 rigid_body_id: Optional[int] = None,
                 rigid_body_name: Optional[str] = None,
                 ball_marker_id: Optional[int] = None,
                 position_unit: str = 'auto',
                 timestamp_unit: str = 's',
                 robot_source: str = 'fk',
                 cart_state_topic: str = '/cart_states',
                 right_arm_mu_idx: int = RIGHT_ARM_MU_IDX,
                 tool_offset_m: Optional[np.ndarray] = None,
                 stability_window_s: float = DEFAULT_STABILITY_WINDOW_S,
                 max_mocap_std_m: float = DEFAULT_MAX_MOCAP_STD_M,
                 max_mocap_rotation_rms_deg: float = (
                     DEFAULT_MAX_MOCAP_ROTATION_RMS_DEG),
                 max_mocap_mean_error_m: float = (
                     DEFAULT_MAX_MOCAP_MEAN_ERROR_M),
                 handeye_method: str = 'PARK'):
        super().__init__('mocap_calibration')

        self.udp_port = int(udp_port)
        self.robot_xml = robot_xml
        self.calib_target = str(calib_target)
        self.marker_id = int(marker_id)
        self.rigid_body_id = int(
            rigid_body_id if rigid_body_id is not None else marker_id)
        self.rigid_body_slot = (
            int(rigid_body_slot)
            if rigid_body_slot is not None else self.rigid_body_id)
        self.rigid_body_name = (
            str(rigid_body_name).strip() if rigid_body_name else None)
        self.ball_marker_id = int(
            ball_marker_id if ball_marker_id is not None else marker_id)
        self.position_unit = str(position_unit)
        self.timestamp_unit = str(timestamp_unit)
        self.robot_source = str(robot_source)
        self.cart_state_topic = str(cart_state_topic)
        self.right_arm_mu_idx = int(right_arm_mu_idx)
        self.tool_offset_m = (
            None if tool_offset_m is None
            else np.asarray(tool_offset_m, dtype=np.float64).reshape(3))
        self.stability_window_s = max(float(stability_window_s), 0.0)
        self.max_mocap_std_m = max(float(max_mocap_std_m), 0.0)
        self.max_mocap_rotation_rms_deg = max(
            float(max_mocap_rotation_rms_deg), 0.0)
        self.max_mocap_mean_error_m = max(
            float(max_mocap_mean_error_m), 0.0)
        self.handeye_method = str(handeye_method).strip().upper()
        if self.calib_target == 'rigid_body' and self.robot_source != 'fk':
            raise ValueError(
                'Full rigid-body hand-eye calibration requires robot_source=fk '
                'so every end-effector pose is T_waist03_right_ee_site')
        if self.calib_target == 'rigid_body' and self.tool_offset_m is not None:
            raise ValueError(
                'Do not use tool_offset for full rigid-body hand-eye calibration; '
                'the fixed right_ee_site -> K transform is solved directly')

        self.mocap_object_id = (
            self.rigid_body_id
            if self.calib_target == 'rigid_body' else self.marker_id)
        self.mocap_object_slot = (
            self.rigid_body_slot
            if self.calib_target == 'rigid_body' else None)
        self.mocap_object_name = (
            self.rigid_body_name
            if self.calib_target == 'rigid_body' else None)

        try:
            self.fk = RightArmFK(robot_xml, joint_names=RIGHT_ARM_JOINTS)
            self.get_logger().info(f'Loaded robot model: {robot_xml}')
        except Exception as exc:
            self.get_logger().error(f'Failed to load robot model: {exc}')
            raise

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', self.udp_port))
        self.socket.settimeout(0.05)
        self.get_logger().info(f'Listening on UDP port {self.udp_port}')

        self.latest_joint_state: Optional[JointState] = None
        self.latest_cart_state = None

        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)
        if self.robot_source == 'cart':
            if MechUnitStateArray is None:
                raise RuntimeError(
                    'robot_source=cart requires mc_core_interface.msg.MechUnitStateArray')
            self.cart_state_sub = self.create_subscription(
                MechUnitStateArray, self.cart_state_topic,
                self._cart_state_callback, 10)
        else:
            self.cart_state_sub = None

        self.session = CalibrationSession()
        self.T_ee_rigid: Optional[np.ndarray] = None

    @property
    def sample_count(self) -> int:
        return len(self.session.samples)

    def _joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def _cart_state_callback(self, msg) -> None:
        self.latest_cart_state = msg

    def _parse_mocap_datagram(
            self, data: bytes
    ) -> Optional[MocapCalibrationMeasurement]:
        if self.calib_target == 'rigid_body':
            packet = parse_mocap_packet(
                data,
                object_kind='opti_rb',
                rigid_body_slot=self.mocap_object_slot,
                rigid_body_id=self.mocap_object_id,
                position_unit=self.position_unit,
                timestamp_unit=self.timestamp_unit,
            )
        else:
            packet = parse_mocap_packet(
                data,
                marker_id=self.mocap_object_id,
                object_kind='marker',
                object_name=self.mocap_object_name,
                position_unit=self.position_unit,
                timestamp_unit=self.timestamp_unit,
            )
        if packet is None:
            return None
        orientation_wxyz = packet.orientation_wxyz
        if orientation_wxyz is not None:
            orientation_wxyz = np.asarray(
                orientation_wxyz, dtype=np.float64).reshape(4)
            if (not np.all(np.isfinite(orientation_wxyz))
                    or np.linalg.norm(orientation_wxyz) < 1e-12):
                orientation_wxyz = None
        return MocapCalibrationMeasurement(
            position_m=packet.position_m.astype(np.float64),
            orientation_wxyz=orientation_wxyz,
            timestamp_s=packet.timestamp_s,
            frame_idx=packet.frame_idx,
            mean_error_m=packet.mean_error_m,
        )

    def receive_mocap_measurement(
            self
    ) -> Optional[MocapCalibrationMeasurement]:
        try:
            data, _addr = self.socket.recvfrom(4096)
            return self._parse_mocap_datagram(data)
        except socket.timeout:
            return None
        except Exception as exc:
            self.get_logger().warn(f'Mocap receive error: {exc}')
            return None

    def _discard_pending_mocap_datagrams(self) -> int:
        """Discard queued UDP data so a sample is not paired with an old pose."""
        discarded = 0
        original_timeout = self.socket.gettimeout()
        try:
            self.socket.setblocking(False)
            while True:
                self.socket.recvfrom(4096)
                discarded += 1
        except BlockingIOError:
            pass
        except Exception as exc:
            self.get_logger().warn(f'Mocap queue flush failed: {exc}')
        finally:
            self.socket.settimeout(original_timeout)
        return discarded

    def receive_stable_mocap_pose(
            self, timeout_s: float = 0.5
    ) -> Tuple[Optional[StableMocapMeasurement], str]:
        """Average a fresh static window and reject poor rigid-body solutions."""
        discarded = self._discard_pending_mocap_datagrams()
        timeout_deadline = time.monotonic() + max(float(timeout_s), 0.0)
        samples: List[MocapCalibrationMeasurement] = []
        stability_deadline = None

        while time.monotonic() < timeout_deadline:
            measurement = self.receive_mocap_measurement()
            if measurement is not None:
                samples.append(measurement)
                if stability_deadline is None:
                    stability_deadline = (
                        time.monotonic() + self.stability_window_s)
                if (len(samples) >= MIN_STABILITY_PACKETS
                        and time.monotonic() >= stability_deadline):
                    positions = np.asarray(
                        [sample.position_m for sample in samples],
                        dtype=np.float64)
                    std_m = float(np.linalg.norm(np.std(positions, axis=0)))
                    if std_m > self.max_mocap_std_m:
                        return None, (
                            f'position std {std_m * 1000.0:.3f} mm exceeds '
                            f'{self.max_mocap_std_m * 1000.0:.3f} mm')

                    orientation_mean = None
                    rotation_rms_deg = None
                    rotation_max_deg = None
                    if self.calib_target == 'rigid_body':
                        orientations = [
                            sample.orientation_wxyz for sample in samples]
                        if any(q is None for q in orientations):
                            return None, (
                                'one or more rigid-body packets have no valid '
                                'orientation')
                        orientation_values = [
                            np.asarray(q, dtype=np.float64)
                            for q in orientations if q is not None]
                        orientation_mean = average_quaternions_wxyz(
                            orientation_values)
                        rotation_rms_deg, rotation_max_deg = (
                            quaternion_spread_deg(
                                orientation_values, orientation_mean))
                        if (rotation_rms_deg
                                > self.max_mocap_rotation_rms_deg):
                            return None, (
                                f'orientation RMS {rotation_rms_deg:.3f} deg '
                                f'exceeds '
                                f'{self.max_mocap_rotation_rms_deg:.3f} deg')

                    mean_errors = np.asarray([
                        sample.mean_error_m for sample in samples
                        if sample.mean_error_m is not None
                    ], dtype=np.float64)
                    mean_error_mean_m = (
                        None if mean_errors.size == 0
                        else float(np.mean(mean_errors)))
                    mean_error_max_m = (
                        None if mean_errors.size == 0
                        else float(np.max(mean_errors)))
                    if (mean_error_max_m is not None
                            and mean_error_max_m
                            > self.max_mocap_mean_error_m):
                        return None, (
                            f'rigid-body meanError max '
                            f'{mean_error_max_m * 1000.0:.3f} mm exceeds '
                            f'{self.max_mocap_mean_error_m * 1000.0:.3f} mm')

                    latest = samples[-1]
                    return StableMocapMeasurement(
                        position_m=np.mean(positions, axis=0),
                        orientation_wxyz=orientation_mean,
                        position_std_m=std_m,
                        rotation_rms_deg=rotation_rms_deg,
                        rotation_max_deg=rotation_max_deg,
                        mean_error_mean_m=mean_error_mean_m,
                        mean_error_max_m=mean_error_max_m,
                        timestamp_s=latest.timestamp_s,
                        frame_idx=latest.frame_idx,
                        packet_count=len(samples),
                        discarded_packet_count=discarded,
                    ), ''

        return None, (
            f'timed out with {len(samples)} valid packets; need at least '
            f'{MIN_STABILITY_PACKETS}')

    def _extract_right_arm_joints(self) -> Optional[np.ndarray]:
        if self.latest_joint_state is None:
            return None
        msg = self.latest_joint_state
        try:
            q = []
            for joint_name in RIGHT_ARM_JOINTS:
                idx = msg.name.index(joint_name)
                q.append(msg.position[idx])
            return np.array(q, dtype=np.float64)
        except (ValueError, IndexError) as exc:
            self.get_logger().warn(f'Failed to extract right-arm joints: {exc}')
            return None

    def _get_robot_pose_fk(
            self, q: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if q is None:
            q = self._extract_right_arm_joints()
        if q is None:
            return None, None
        try:
            T_ee = self.fk.ee_pose(q)
            if self.tool_offset_m is None:
                return T_ee[:3, 3].copy(), T_ee
            return apply_tool_offset(T_ee, self.tool_offset_m), T_ee
        except Exception as exc:
            self.get_logger().warn(f'FK failed: {exc}')
            return None, None

    def _get_robot_pose_cart(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.latest_cart_state is None:
            return None, None
        right_arm_state = None
        for state in self.latest_cart_state.states:
            if int(state.mu_idx) == self.right_arm_mu_idx:
                right_arm_state = state
                break
        if right_arm_state is None:
            self.get_logger().warn(
                f'Right arm mu_idx={self.right_arm_mu_idx} not found in '
                f'{self.cart_state_topic}')
            return None, None

        pose = right_arm_state.end_pose
        position = [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ]
        orientation = [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ]
        T_ee = pose_msg_to_matrix(position, orientation)
        if self.tool_offset_m is None:
            return T_ee[:3, 3].copy(), T_ee
        return apply_tool_offset(T_ee, self.tool_offset_m), T_ee

    def get_robot_paddle_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """Return paddle-center pose from XML FK (right_ee_site in waist03 frame)."""
        q = self._extract_right_arm_joints()
        if q is None:
            return None, None, False
        try:
            T_ee = self.fk.ee_pose(q)
            if self.tool_offset_m is not None:
                pos = apply_tool_offset(T_ee, self.tool_offset_m)
            else:
                pos = T_ee[:3, 3].copy()
            quat_wxyz = _matrix_to_quat_wxyz(T_ee[:3, :3])
            return pos, quat_wxyz, True
        except Exception as exc:
            self.get_logger().warn(f'FK paddle pose failed: {exc}')
            return None, None, False

    def get_robot_reference_position(self) -> Tuple[np.ndarray, bool]:
        T_reference, ok = self.get_robot_reference_pose()
        if not ok or T_reference is None:
            return np.zeros(3), False
        return T_reference[:3, 3].copy(), True

    def get_robot_reference_pose(
            self) -> Tuple[Optional[np.ndarray], bool]:
        """Return ``T_waist03_right_ee_site`` (or its optional virtual offset).

        This is deliberately the same waist03-rooted end-effector pose used by
        the eye-to-hand calibration code, rather than a position in another
        robot frame.
        """
        if self.robot_source == 'cart':
            _pos, T_ee = self._get_robot_pose_cart()
        else:
            _pos, T_ee = self._get_robot_pose_fk()
        if T_ee is None:
            return None, False

        T_reference = np.asarray(T_ee, dtype=np.float64).copy()
        if self.tool_offset_m is not None:
            # Keep the EE orientation and replace only its origin with the
            # requested fixed local reference point.  Full hand-eye mode does
            # not require this offset: it estimates T_right_ee_site_K itself.
            T_reference[:3, 3] = apply_tool_offset(T_reference, self.tool_offset_m)
        return T_reference, True

    def record_sample(self) -> bool:
        stable_mocap, rejection_reason = self.receive_stable_mocap_pose()
        if stable_mocap is None:
            self.get_logger().warn(
                f'Rejected mocap calibration sample: {rejection_reason}')
            return False

        mocap_pos = stable_mocap.position_m
        mocap_orientation_wxyz = stable_mocap.orientation_wxyz

        if self.calib_target == 'rigid_body' and mocap_orientation_wxyz is None:
            self.get_logger().warn(
                'Rigid-body hand-eye calibration requires orientation fields '
                f'opti_rb_{self.mocap_object_slot}_qw/qx/qy/qz. This packet '
                'contains only position, so the sample was rejected.')
            return False

        joint_positions = (
            self._extract_right_arm_joints()
            if self.robot_source == 'fk' else None)
        if self.robot_source == 'fk':
            _robot_pos, T_waist03_ee = self._get_robot_pose_fk(
                joint_positions)
            robot_ok = T_waist03_ee is not None
        else:
            T_waist03_ee, robot_ok = self.get_robot_reference_pose()
        if not robot_ok:
            self.get_logger().warn(
                'Failed to get waist03/right_ee_site reference pose')
            return False
        assert T_waist03_ee is not None
        robot_pos = T_waist03_ee[:3, 3].copy()
        T_mocap_rigid = None
        if mocap_orientation_wxyz is not None:
            try:
                T_mocap_rigid = pose_wxyz_to_matrix(
                    mocap_pos, mocap_orientation_wxyz)
            except ValueError as exc:
                self.get_logger().warn(
                    f'Rejected invalid mocap rigid-body orientation: {exc}')
                return False

        sample = CalibrationSample(
            index=len(self.session.samples) + 1,
            mocap_m=mocap_pos.copy(),
            robot_m=robot_pos.copy(),
            mocap_orientation_wxyz=(
                None if mocap_orientation_wxyz is None
                else mocap_orientation_wxyz.copy()),
            T_mocap_rigid=T_mocap_rigid,
            T_waist03_ee=T_waist03_ee,
            mocap_std_m=stable_mocap.position_std_m,
            mocap_rotation_rms_deg=stable_mocap.rotation_rms_deg,
            mocap_rotation_max_deg=stable_mocap.rotation_max_deg,
            mocap_mean_error_mean_m=stable_mocap.mean_error_mean_m,
            mocap_mean_error_max_m=stable_mocap.mean_error_max_m,
            mocap_timestamp_s=stable_mocap.timestamp_s,
            mocap_frame_idx=stable_mocap.frame_idx,
            mocap_packet_count=stable_mocap.packet_count,
            discarded_packet_count=stable_mocap.discarded_packet_count,
            joint_positions_rad=(
                None if joint_positions is None
                else joint_positions.copy()),
            robot_source=self.robot_source,
        )
        self.session.samples.append(sample)
        self.session.mocap_points.append(mocap_pos)
        self.session.robot_points.append(robot_pos)
        if self.calib_target == 'rigid_body':
            # Rigid-body mode above rejects missing orientation, so both lists
            # contain full pose pairs in the exact eye-to-hand convention.
            assert T_mocap_rigid is not None
            self.session.mocap_poses.append(T_mocap_rigid)
            self.session.robot_poses.append(T_waist03_ee)

        self.get_logger().info(
            f'Sample {sample.index} recorded '
            f'(mean of {stable_mocap.packet_count} fresh packets: '
            f'position std={stable_mocap.position_std_m * 1000:.3f} mm'
            + ('' if stable_mocap.rotation_rms_deg is None else
               f', orientation RMS/max='
               f'{stable_mocap.rotation_rms_deg:.3f}/'
               f'{stable_mocap.rotation_max_deg:.3f} deg')
            + ('' if stable_mocap.mean_error_max_m is None else
               f', rigid meanError max='
               f'{stable_mocap.mean_error_max_m * 1000:.3f} mm')
            + f'; discarded {stable_mocap.discarded_packet_count} queued '
              f'packets):\n'
            f'  Mocap: [{mocap_pos[0]:.4f}, {mocap_pos[1]:.4f}, {mocap_pos[2]:.4f}]\n'
            f'  Robot waist03/right_ee_site: '
            f'[{robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f}]'
            + ('' if mocap_orientation_wxyz is None else
               f'\n  Mocap quaternion (wxyz): '
               f'[{mocap_orientation_wxyz[0]:.5f}, '
               f'{mocap_orientation_wxyz[1]:.5f}, '
               f'{mocap_orientation_wxyz[2]:.5f}, '
               f'{mocap_orientation_wxyz[3]:.5f}]'))
        return True

    def compute_calibration(self) -> Tuple[np.ndarray, float, dict]:
        if len(self.session.mocap_points) < 3:
            raise ValueError('Need at least 3 samples for calibration')

        if self.calib_target == 'rigid_body':
            if len(self.session.mocap_poses) < 3:
                raise ValueError(
                    'Need at least 3 complete rigid-body pose pairs '
                    '(position plus qw/qx/qy/qz)')
            T_robot_mocap, self.T_ee_rigid, metrics = fit_eye_to_hand_transform(
                self.session.robot_poses,
                self.session.mocap_poses,
                method=self.handeye_method,
            )
            rmse = float(metrics['rmse_m'])
            pose_metrics_text = (
                f'Orientation mean/max: '
                f'{metrics["rotation_mean_deg"]:.3f} / '
                f'{metrics["rotation_max_deg"]:.3f} deg\n'
                f'Per-sample orientation error (deg): '
                f'{", ".join(f"{e:.3f}" for e in metrics["per_sample_rotation_deg"])}\n'
                f'T_right_ee_site_K (estimated fixed mount):\n'
                f'{self.T_ee_rigid}\n')
        else:
            # Legacy marker mode has no rigid-body attitude.  It remains a
            # point-pair fit for compatibility, but it is not suitable for a
            # K body offset from the end-effector reference site.
            T_robot_mocap, rmse = fit_rigid_transform(
                np.asarray(self.session.mocap_points, dtype=np.float64),
                np.asarray(self.session.robot_points, dtype=np.float64),
            )
            metrics = evaluate_point_pairs(
                np.asarray(self.session.mocap_points, dtype=np.float64),
                np.asarray(self.session.robot_points, dtype=np.float64),
                T_robot_mocap,
            )
            metrics['method'] = 'Kabsch point-pair (legacy marker mode)'
            pose_metrics_text = ''

        self.get_logger().info(
            f"\n{'=' * 60}\n"
            f'Calibration computed from {len(self.session.samples)} samples\n'
            f'Method: {metrics["method"]}\n'
            f'RMSE: {metrics["rmse_m"] * 1000:.2f} mm\n'
            f'Mean error: {metrics["mean_error_m"] * 1000:.2f} mm\n'
            f'Max error: {metrics["max_error_m"] * 1000:.2f} mm\n'
            f'Std error: {metrics["std_error_m"] * 1000:.2f} mm\n'
            f'Per-sample error (mm): '
            f'{", ".join(f"{e:.1f}" for e in metrics["per_sample_error_mm"])}\n'
            f'{pose_metrics_text}'
            f"{'=' * 60}\n"
            f'T_waist03_mocap (runtime T_robot_mocap):\n{T_robot_mocap}\n'
            f"{'=' * 60}")
        return T_robot_mocap, rmse, metrics

    def build_metadata(self,
                       T_robot_mocap: np.ndarray,
                       rmse_m: float,
                       metrics: dict,
                       output_txt: str,
                       output_json: Optional[str]) -> dict:
        return {
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'T_robot_mocap': T_robot_mocap.tolist(),
            'T_waist03_mocap': T_robot_mocap.tolist(),
            'T_right_ee_site_K': (
                None if self.T_ee_rigid is None
                else self.T_ee_rigid.tolist()),
            'rmse_m': float(rmse_m),
            'error_metrics': metrics,
            'calibration_method': metrics.get('method'),
            'handeye_method': (
                self.handeye_method
                if self.calib_target == 'rigid_body' else None),
            'robot_reference_site': 'right_ee_site',
            'robot_reference_frame': 'waist03_cartesian',
            'calib_target': self.calib_target,
            'mocap_object_id': self.mocap_object_id,
            'mocap_object_slot': self.mocap_object_slot,
            'mocap_object_name': self.mocap_object_name,
            'ball_marker_id': self.ball_marker_id,
            'robot_source': self.robot_source,
            'cart_state_topic': self.cart_state_topic,
            'right_arm_mu_idx': self.right_arm_mu_idx,
            'tool_offset_m': (
                None if self.tool_offset_m is None
                else self.tool_offset_m.tolist()),
            'stability_window_s': self.stability_window_s,
            'max_mocap_std_m': self.max_mocap_std_m,
            'max_mocap_rotation_rms_deg': (
                self.max_mocap_rotation_rms_deg),
            'max_mocap_mean_error_m': self.max_mocap_mean_error_m,
            'position_unit': self.position_unit,
            'timestamp_unit': self.timestamp_unit,
            'udp_port': self.udp_port,
            'robot_xml': self.robot_xml,
            'output_txt': str(Path(output_txt).resolve()),
            'output_json': (
                None if output_json is None
                else str(Path(output_json).resolve())),
            'samples': [
                {
                    'index': sample.index,
                    'mocap_m': sample.mocap_m.tolist(),
                    'robot_m': sample.robot_m.tolist(),
                    'mocap_orientation_wxyz': (
                        None if sample.mocap_orientation_wxyz is None
                        else sample.mocap_orientation_wxyz.tolist()),
                    'T_mocap_rigid': (
                        None if sample.T_mocap_rigid is None
                        else sample.T_mocap_rigid.tolist()),
                    'T_waist03_right_ee_site': (
                        None if sample.T_waist03_ee is None
                        else sample.T_waist03_ee.tolist()),
                    'mocap_std_m': sample.mocap_std_m,
                    'mocap_rotation_rms_deg': (
                        sample.mocap_rotation_rms_deg),
                    'mocap_rotation_max_deg': (
                        sample.mocap_rotation_max_deg),
                    'mocap_mean_error_mean_m': (
                        sample.mocap_mean_error_mean_m),
                    'mocap_mean_error_max_m': (
                        sample.mocap_mean_error_max_m),
                    'mocap_timestamp_s': sample.mocap_timestamp_s,
                    'mocap_frame_idx': sample.mocap_frame_idx,
                    'mocap_packet_count': sample.mocap_packet_count,
                    'discarded_packet_count': (
                        sample.discarded_packet_count),
                    'joint_names': (
                        list(RIGHT_ARM_JOINTS)
                        if sample.joint_positions_rad is not None else None),
                    'joint_positions_rad': (
                        None if sample.joint_positions_rad is None
                        else sample.joint_positions_rad.tolist()),
                    'robot_source': sample.robot_source,
                }
                for sample in self.session.samples
            ],
        }


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q /= norm
    return q


def _parse_xyz(text: str, name: str) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f'{name} must be three comma-separated numbers, got {text!r}')
    try:
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'{name} must be numeric: {text!r}') from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Calibrate mocap-to-robot transform for ball tracking')
    parser.add_argument(
        '--calib-target',
        choices=['rigid_body', 'marker', 'ball'],
        default='rigid_body',
        help='Mocap object used for calibration (default: rigid_body/K-body)')
    parser.add_argument('--udp-port', type=int, default=9870,
                        help='UDP port for mocap data (default: 9870)')
    parser.add_argument('--marker-id', type=int, default=1,
                        help='Marker id for legacy ball/marker calibration')
    parser.add_argument('--rigid-body-id', type=int, default=None,
                        help='OptiTrack rigid body id YY in opti_rb_YY_id '
                             '(default: same as --marker-id)')
    parser.add_argument('--rigid-body-slot', type=int, default=None,
                        help='OptiTrack rigid body slot Y in opti_rb_Y_px/py/pz '
                             '(default: same as --rigid-body-id)')
    parser.add_argument('--rigid-body-name', type=str, default=None,
                        help='Optional rigid body name filter in JSON packets')
    parser.add_argument('--ball-marker-id', type=int, default=None,
                        help='Ping-pong ball marker YY for runtime only; written '
                             'to calibration JSON metadata. Not used while '
                             'collecting K rigid-body samples (default: same as '
                             '--marker-id)')
    parser.add_argument('--position-unit', type=str, default='auto',
                        choices=['auto', 'm', 'mm', 'cm'],
                        help='Incoming mocap position unit (default: auto)')
    parser.add_argument('--timestamp-unit', type=str, default='s',
                        choices=['auto', 's', 'ms', 'us', 'ns'],
                        help='Incoming mocap timestamp unit (default: s)')
    parser.add_argument('--robot-xml', type=str, required=True,
                        help='Path to robot MJCF XML (e.g., moz1_pd.xml)')
    parser.add_argument('--robot-source', type=str, default='fk',
                        choices=['fk', 'cart'],
                        help='Robot reference pose source; rigid_body mode '
                             'requires fk/right_ee_site (default: fk)')
    parser.add_argument('--cart-state-topic', type=str, default='/cart_states',
                        help='Cartesian feedback topic when robot-source=cart')
    parser.add_argument('--right-arm-mu-idx', type=int, default=RIGHT_ARM_MU_IDX,
                        help='Right arm mu_idx in /cart_states (default: 3)')
    parser.add_argument('--tool-offset', type=_parse_xyz, default=None,
                        metavar='X,Y,Z',
                        help='Optional virtual reference-point offset in the EE '
                             'frame, meters. Leave unset for full rigid-body '
                             'hand-eye calibration: its fixed K mount is solved.')
    parser.add_argument(
        '--handeye-method', type=str.upper, default='PARK',
        choices=['TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS'],
        help='OpenCV full-pose hand-eye method for rigid_body mode '
             '(default: PARK)')
    parser.add_argument(
        '--stability-window', type=float, default=DEFAULT_STABILITY_WINDOW_S,
        help='Seconds of fresh static mocap packets averaged into one saved '
             'pose (default: 0.1)')
    parser.add_argument(
        '--max-mocap-std-mm', type=float,
        default=DEFAULT_MAX_MOCAP_STD_M * 1000.0,
        help='Reject a sample when its stability-check std exceeds this value '
             'in mm (default: 0.25)')
    parser.add_argument(
        '--max-mocap-rotation-rms-deg', type=float,
        default=DEFAULT_MAX_MOCAP_ROTATION_RMS_DEG,
        help='Reject a rigid-body sample when quaternion-window RMS angular '
             'spread exceeds this value in degrees (default: 0.20)')
    parser.add_argument(
        '--max-mocap-mean-error-mm', type=float,
        default=DEFAULT_MAX_MOCAP_MEAN_ERROR_M * 1000.0,
        help='Reject a rigid-body sample when OptiTrack meanError exceeds this '
             'value in mm; ignored when the packet has no meanError '
             '(default: 1.0)')
    parser.add_argument('--output', type=str, default='mocap_calibration.txt',
                        help='Output 4x4 matrix file')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Optional JSON metadata output path '
                             '(default: <output>.json)')
    parser.add_argument('--samples-json', type=str, default=None,
                        help='Optional path to save raw samples only')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='Recommended minimum samples (default: 10)')
    parser.add_argument('--verify', type=str, default=None,
                        help='Load an existing calibration txt and print RMSE '
                             'on the collected samples without saving')
    return parser


def _configured_ids(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Resolve CLI ids without treating a valid zero id as missing."""
    rigid_body_id = (
        args.marker_id if args.rigid_body_id is None else args.rigid_body_id)
    rigid_body_slot = (
        rigid_body_id if args.rigid_body_slot is None else args.rigid_body_slot)
    ball_marker_id = (
        args.marker_id if args.ball_marker_id is None else args.ball_marker_id)
    return int(rigid_body_slot), int(rigid_body_id), int(ball_marker_id)


def _print_instructions(args: argparse.Namespace) -> None:
    rigid_body_slot, rigid_body_id, ball_marker_id = _configured_ids(args)

    print('\n' + '=' * 60)
    print('Motion Capture Calibration Tool')
    print('=' * 60)
    print(f'Calibration target : {args.calib_target}')
    print(f'UDP port           : {args.udp_port}')
    print(f'Rigid body slot Y : {rigid_body_slot}')
    print(f'Rigid body id YY  : {rigid_body_id}')
    if args.rigid_body_name:
        print(f'Rigid body name    : {args.rigid_body_name}')
    if args.calib_target == 'rigid_body':
        print('Ball marker id     : '
              f'{ball_marker_id} '
              '(metadata only; not used during K-body sampling)')
    else:
        print(f'Ball marker id     : {ball_marker_id}')
    print(f'Position unit      : {args.position_unit}')
    print(f'Timestamp unit     : {args.timestamp_unit}')
    print(f'Robot source       : {args.robot_source}')
    print(f'Robot model        : {args.robot_xml}')
    print(f'Tool offset (m)    : {args.tool_offset}')
    if args.calib_target == 'rigid_body':
        print(f'Hand-eye method    : {args.handeye_method}')
    print(f'Stability check    : {args.stability_window:.3f} s, '
          f'max position std={args.max_mocap_std_mm:.2f} mm, '
          f'max rotation RMS={args.max_mocap_rotation_rms_deg:.2f} deg, '
          f'max meanError={args.max_mocap_mean_error_mm:.2f} mm '
          '(window pose is averaged)')
    print(f'Output matrix      : {args.output}')
    print(f'Output json        : {args.output_json or Path(args.output).with_suffix(".json")}')
    print(f'Minimum samples    : {args.min_samples}')
    print('=' * 60)
    print('\nInstructions:')
    if args.calib_target == 'rigid_body':
        print('  1. Rigidly mount the K-shaped body anywhere on the paddle.')
        print('     Its fixed right_ee_site -> K transform is estimated.')
        print('  2. Ensure every UDP packet includes opti_rb_<slot>_qw/qx/qy/qz.')
        print('  3. Move the arm through diverse poses under mocap, including')
        print('     rotations about multiple axes (not translation only).')
        print('     Use roughly 10–20 static poses with 15°+ orientation changes.')
        print('  4. Press ENTER at each stable pose to record one full pose pair.')
        print('     The robot pose is T_waist03_right_ee_site from RightArmFK;')
        print('     it is not /cart_states or base_link.')
        print('     The saved position/quaternion are averaged over the accepted window.')
    else:
        print('  1. Mount the calibration marker/ball on the right-arm EE.')
        print('  2. Move the arm through diverse poses under mocap.')
        print('  3. Press ENTER at each stable pose to record one sample.')
    print(f'  5. Collect at least {args.min_samples} samples (>=3 required).')
    print("  6. Type 'q' and press ENTER to solve and save T_robot_mocap.")
    print('=' * 60 + '\n')


def _wait_for_robot_feeds(node: MocapCalibrationNode) -> None:
    print('Waiting for /joint_states...')
    while node.latest_joint_state is None:
        time.sleep(0.1)
    print('✓ Receiving /joint_states')
    if node.robot_source == 'cart':
        print(f'Waiting for {node.cart_state_topic}...')
        while node.latest_cart_state is None:
            time.sleep(0.1)
        print(f'✓ Receiving {node.cart_state_topic}')
    print()


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.calib_target in ('marker', 'ball'):
        args.calib_target = 'marker'
    if args.calib_target == 'rigid_body' and args.robot_source != 'fk':
        parser.error(
            '--calib-target rigid_body requires --robot-source fk, which '
            'returns T_waist03_right_ee_site exactly')
    if args.calib_target == 'rigid_body' and args.tool_offset is not None:
        parser.error(
            '--tool-offset is incompatible with full rigid-body hand-eye '
            'calibration; omit it so right_ee_site is used directly')

    output_json = (
        args.output_json
        if args.output_json is not None
        else str(Path(args.output).with_suffix('.json')))

    rclpy.init()
    node = MocapCalibrationNode(
        udp_port=args.udp_port,
        robot_xml=args.robot_xml,
        calib_target=args.calib_target,
        marker_id=args.marker_id,
        rigid_body_slot=args.rigid_body_slot,
        rigid_body_id=args.rigid_body_id,
        rigid_body_name=args.rigid_body_name,
        ball_marker_id=args.ball_marker_id,
        position_unit=args.position_unit,
        timestamp_unit=args.timestamp_unit,
        robot_source=args.robot_source,
        cart_state_topic=args.cart_state_topic,
        right_arm_mu_idx=args.right_arm_mu_idx,
        tool_offset_m=args.tool_offset,
        stability_window_s=args.stability_window,
        max_mocap_std_m=args.max_mocap_std_mm * 1e-3,
        max_mocap_rotation_rms_deg=args.max_mocap_rotation_rms_deg,
        max_mocap_mean_error_m=args.max_mocap_mean_error_mm * 1e-3,
        handeye_method=args.handeye_method,
    )

    _print_instructions(args)

    import threading
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()
    completed = False
    try:
        _wait_for_robot_feeds(node)
        while True:
            user_input = input(
                f'[{node.sample_count} samples] '
                "Press ENTER to record, 'q' to finish: ").strip()
            if user_input.lower() == 'q':
                if node.sample_count < 3:
                    print(f'ERROR: Need at least 3 samples, only have {node.sample_count}')
                    continue
                if node.sample_count < args.min_samples:
                    print(
                        f'WARNING: Only {node.sample_count} samples collected, '
                        f'recommended minimum is {args.min_samples}')
                    confirm = input('Continue anyway? (y/n): ').strip().lower()
                    if confirm != 'y':
                        continue
                break
            if not node.record_sample():
                print('Failed to record sample, please try again')

        if args.verify:
            T_robot_mocap = load_calibration_matrix(args.verify)
            print(f'Loaded verification calibration from {args.verify}')
        else:
            T_robot_mocap, rmse, metrics = node.compute_calibration()

        if not args.verify:
            header = (
                'T_robot_mocap: 4x4 homogeneous transform from mocap to '
                'waist03_cartesian frame')
            save_calibration_matrix(args.output, T_robot_mocap, header=header)
            metadata = node.build_metadata(
                T_robot_mocap, rmse, metrics, args.output, output_json)
            save_calibration_json(output_json, metadata)
            if args.samples_json:
                save_calibration_json(args.samples_json, metadata['samples'])
            print(f'\n✓ Calibration matrix saved to: {Path(args.output).resolve()}')
            print(f'✓ Calibration metadata saved to: {Path(output_json).resolve()}')
            print(
                f'  RMSE={metrics["rmse_m"] * 1000:.2f} mm, '
                f'max={metrics["max_error_m"] * 1000:.2f} mm, '
                f'mean={metrics["mean_error_m"] * 1000:.2f} mm')
            print(
                '\nNext step: run mocap_ball_state_node or pingpong_node with '
                f'mocap_marker_id:={node.ball_marker_id} and '
                f'mocap_calibration_path:={Path(args.output).resolve()}')
        completed = True
    except KeyboardInterrupt:
        print('\nCalibration cancelled')
    except Exception as exc:
        print(f'\nERROR: Calibration failed: {exc}')
    finally:
        # A daemon spin thread can make the C++ rcl context abort during
        # interpreter teardown. Shut it down and join it before returning.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)

    if completed:
        print('\nCalibration complete!')


if __name__ == '__main__':
    main()
