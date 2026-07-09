#!/usr/bin/env python3
"""
Motion capture calibration tool for pingpong ball tracking.

Calibrates the transformation from mocap coordinate frame to robot base_link frame
by collecting paired measurements:
  - Ball position in mocap frame (from UDP mocap system)
  - Ball position in robot frame (from robot right-arm end-effector FK)

The ball is mounted on the robot's right-arm end-effector. By moving the arm to
multiple poses, we collect corresponding point pairs and solve for the rigid
transformation T_robot_mocap using least-squares fitting.

Usage:
    ros2 run pingpong_controller mocap_calibration -- \
        --udp-port 9870 --robot-xml /path/to/moz1_pd.xml --output calibration.txt

    Then move the robot arm through different poses. Press Enter at each pose to
    record a sample. After collecting enough samples (minimum 3, recommended 10+),
    press 'q' to compute and save the calibration.
"""

import argparse
import socket
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# Import FK from replay module
try:
    from pingpong_replay.cartesian_fk import RightArmFK, RIGHT_ARM_JOINTS
except ImportError:
    print("ERROR: Cannot import RightArmFK. Make sure pingpong_replay is built.")
    sys.exit(1)

try:
    from pingpong_controller.mocap_ball_detector import parse_mocap_packet
except ImportError:
    from mocap_ball_detector import parse_mocap_packet


class MocapCalibrationNode(Node):
    """ROS2 node for mocap-to-robot calibration data collection."""

    def __init__(self, udp_port: int, robot_xml: str,
                 marker_id: int = 1,
                 position_unit: str = 'auto',
                 timestamp_unit: str = 's'):
        super().__init__('mocap_calibration')

        self.udp_port = udp_port
        self.robot_xml = robot_xml
        self.marker_id = int(marker_id)
        self.position_unit = str(position_unit)
        self.timestamp_unit = str(timestamp_unit)

        # Forward kinematics
        try:
            self.fk = RightArmFK(robot_xml, joint_names=RIGHT_ARM_JOINTS)
            self.get_logger().info(f"Loaded robot model: {robot_xml}")
        except Exception as e:
            self.get_logger().error(f"Failed to load robot model: {e}")
            raise

        # UDP socket for mocap data
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('0.0.0.0', udp_port))
        self.socket.settimeout(0.1)
        self.get_logger().info(f"Listening on UDP port {udp_port}")

        # Latest joint state
        self.latest_joint_state = None
        self.joint_state_lock = None

        # Subscribe to /joint_states
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)

        # Calibration data
        self.mocap_points: List[np.ndarray] = []
        self.robot_points: List[np.ndarray] = []

    def _joint_state_callback(self, msg: JointState):
        """Cache latest joint state."""
        self.latest_joint_state = msg

    def receive_mocap_ball(self) -> Tuple[np.ndarray, bool]:
        """Receive latest ball position from mocap (non-blocking).

        Returns:
            (position, success) where position is (3,) in meters
        """
        try:
            # Non-blocking receive (timeout set in __init__)
            data, addr = self.socket.recvfrom(1024)
            packet = parse_mocap_packet(
                data,
                marker_id=self.marker_id,
                position_unit=self.position_unit,
                timestamp_unit=self.timestamp_unit)
            if packet is None:
                return np.zeros(3), False
            return packet.position_m.astype(np.float64), True
        except socket.timeout:
            return np.zeros(3), False
        except Exception as e:
            self.get_logger().warn(f"Mocap receive error: {e}")
            return np.zeros(3), False

    def get_right_arm_ee_position(self) -> Tuple[np.ndarray, bool]:
        """Get current right-arm end-effector position via FK.

        Returns:
            (position, success) where position is (3,) in meters (base_link frame)
        """
        if self.latest_joint_state is None:
            return np.zeros(3), False

        msg = self.latest_joint_state

        # Extract right-arm joint positions
        try:
            q = []
            for joint_name in RIGHT_ARM_JOINTS:
                idx = msg.name.index(joint_name)
                q.append(msg.position[idx])
            q = np.array(q, dtype=np.float64)
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f"Failed to extract right-arm joints: {e}")
            return np.zeros(3), False

        # Compute FK
        try:
            T_ee = self.fk.ee_pose(q)  # 4x4 homogeneous matrix
            pos = T_ee[:3, 3]  # Extract position
            return pos, True
        except Exception as e:
            self.get_logger().warn(f"FK failed: {e}")
            return np.zeros(3), False

    def record_sample(self) -> bool:
        """Record one paired sample (mocap position, robot position).

        Returns:
            True if sample was successfully recorded
        """
        # Get mocap ball position
        mocap_pos, mocap_ok = self.receive_mocap_ball()
        if not mocap_ok:
            self.get_logger().warn("Failed to receive mocap data")
            return False

        # Get robot end-effector position
        robot_pos, robot_ok = self.get_right_arm_ee_position()
        if not robot_ok:
            self.get_logger().warn("Failed to get robot end-effector position")
            return False

        # Record the pair
        self.mocap_points.append(mocap_pos)
        self.robot_points.append(robot_pos)

        self.get_logger().info(
            f"Sample {len(self.mocap_points)} recorded:\n"
            f"  Mocap: [{mocap_pos[0]:.4f}, {mocap_pos[1]:.4f}, {mocap_pos[2]:.4f}]\n"
            f"  Robot: [{robot_pos[0]:.4f}, {robot_pos[1]:.4f}, {robot_pos[2]:.4f}]")

        return True

    def compute_calibration(self) -> np.ndarray:
        """Compute T_robot_mocap from collected point pairs.

        Uses SVD-based rigid transformation fitting (Kabsch algorithm variant).

        Returns:
            T_robot_mocap: 4x4 homogeneous transformation matrix
        """
        if len(self.mocap_points) < 3:
            raise ValueError("Need at least 3 samples for calibration")

        # Convert to arrays
        P_mocap = np.array(self.mocap_points, dtype=np.float64).T  # (3, N)
        P_robot = np.array(self.robot_points, dtype=np.float64).T  # (3, N)

        # Compute centroids
        centroid_mocap = np.mean(P_mocap, axis=1, keepdims=True)
        centroid_robot = np.mean(P_robot, axis=1, keepdims=True)

        # Center the point sets
        P_mocap_centered = P_mocap - centroid_mocap
        P_robot_centered = P_robot - centroid_robot

        # Compute covariance matrix H = P_robot_centered @ P_mocap_centered.T
        H = P_robot_centered @ P_mocap_centered.T

        # SVD
        U, S, Vt = np.linalg.svd(H)

        # Rotation matrix
        R = U @ Vt

        # Handle reflection case (det(R) = -1)
        if np.linalg.det(R) < 0:
            self.get_logger().warn("Detected reflection, correcting...")
            Vt[-1, :] *= -1
            R = U @ Vt

        # Translation
        t = centroid_robot - R @ centroid_mocap

        # Build 4x4 transform
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()

        # Compute residual error
        P_robot_transformed = R @ P_mocap + t
        residual = P_robot - P_robot_transformed
        rmse = np.sqrt(np.mean(np.sum(residual**2, axis=0)))

        self.get_logger().info(
            f"\n{'='*60}\n"
            f"Calibration computed from {len(self.mocap_points)} samples\n"
            f"RMSE: {rmse*1000:.2f} mm\n"
            f"{'='*60}\n"
            f"T_robot_mocap:\n{T}\n"
            f"{'='*60}")

        return T


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate mocap-to-robot transformation for ball tracking')
    parser.add_argument('--udp-port', type=int, default=9870,
                        help='UDP port for mocap data (default: 9870)')
    parser.add_argument('--marker-id', type=int, default=1,
                        help='Marker id for marker_<id>_px/py/pz fields')
    parser.add_argument('--position-unit', type=str, default='auto',
                        choices=['auto', 'm', 'mm', 'cm'],
                        help='Incoming mocap position unit (default: auto)')
    parser.add_argument('--timestamp-unit', type=str, default='s',
                        choices=['auto', 's', 'ms', 'us', 'ns'],
                        help='Incoming mocap timestamp unit (default: s)')
    parser.add_argument('--robot-xml', type=str, required=True,
                        help='Path to robot MJCF XML (e.g., moz1_pd.xml)')
    parser.add_argument('--output', type=str, default='mocap_calibration.txt',
                        help='Output calibration file (default: mocap_calibration.txt)')
    parser.add_argument('--min-samples', type=int, default=10,
                        help='Minimum samples to collect (default: 10)')

    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()
    node = MocapCalibrationNode(
        args.udp_port, args.robot_xml,
        marker_id=args.marker_id,
        position_unit=args.position_unit,
        timestamp_unit=args.timestamp_unit)

    print("\n" + "="*60)
    print("Motion Capture Calibration Tool")
    print("="*60)
    print(f"UDP port: {args.udp_port}")
    print(f"Marker id: {args.marker_id}")
    print(f"Position unit: {args.position_unit}")
    print(f"Timestamp unit: {args.timestamp_unit}")
    print(f"Robot model: {args.robot_xml}")
    print(f"Output file: {args.output}")
    print(f"Minimum samples: {args.min_samples}")
    print("="*60)
    print("\nInstructions:")
    print("  1. Mount the ball on the robot's right-arm end-effector")
    print("  2. Move the robot arm to different poses")
    print("  3. Press ENTER at each pose to record a sample")
    print(f"  4. Collect at least {args.min_samples} samples")
    print("  5. Type 'q' and press ENTER to finish and compute calibration")
    print("="*60 + "\n")

    # Spin ROS in background
    import threading
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    # Wait for first joint state
    print("Waiting for /joint_states...")
    while node.latest_joint_state is None:
        time.sleep(0.1)
    print("✓ Receiving /joint_states\n")

    # Interactive data collection loop
    try:
        while True:
            user_input = input(f"[{len(node.mocap_points)} samples] Press ENTER to record, 'q' to finish: ").strip()

            if user_input.lower() == 'q':
                if len(node.mocap_points) < 3:
                    print(f"ERROR: Need at least 3 samples, only have {len(node.mocap_points)}")
                    continue
                if len(node.mocap_points) < args.min_samples:
                    print(f"WARNING: Only {len(node.mocap_points)} samples collected, "
                          f"recommended minimum is {args.min_samples}")
                    confirm = input("Continue anyway? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue
                break

            # Record sample
            success = node.record_sample()
            if not success:
                print("Failed to record sample, please try again")

    except KeyboardInterrupt:
        print("\nCalibration cancelled")
        rclpy.shutdown()
        return

    # Compute calibration
    try:
        T_robot_mocap = node.compute_calibration()

        # Save to file
        output_path = Path(args.output).resolve()
        np.savetxt(output_path, T_robot_mocap, fmt='%.8f',
                   header='T_robot_mocap: 4x4 homogeneous transformation from mocap to robot base_link')
        print(f"\n✓ Calibration saved to: {output_path}")

    except Exception as e:
        print(f"\nERROR: Calibration failed: {e}")
        rclpy.shutdown()
        return

    # Shutdown
    rclpy.shutdown()
    print("\nCalibration complete!")


if __name__ == '__main__':
    main()
