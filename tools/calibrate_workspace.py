#!/usr/bin/env python3
"""End-effector workspace calibration tool.

This standalone script helps you manually calibrate safe workspace limits
for the done-condition monitor by recording the end-effector position at
key boundary points.

Usage:
    1. Make sure robot is powered and publishing /joint_states
    2. Run this script: python3 tools/calibrate_workspace.py
    3. Use your manual control method to move the robot arm
    4. Press Enter at each position to record it
    5. Type 'plot' to visualize the workspace shape
    6. Type 'done' to finish and output limits

The script only subscribes to /joint_states (no controller needed).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import sys
import os
import threading
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add the pingpong_controller module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                 'src/pingpong_moz1/pingpong_controller'))

try:
    import mujoco as mj
    _MUJOCO_AVAILABLE = True
except ImportError:
    _MUJOCO_AVAILABLE = False
    print("ERROR: mujoco not available. Install with: pip install mujoco")
    sys.exit(1)


RIGHT_ARM_JOINTS = [
    "RightArm-0", "RightArm-1", "RightArm-2",
    "RightArm-3", "RightArm-4", "RightArm-5", "RightArm-6",
]


class WorkspaceCalibrator(Node):
    """Interactive workspace boundary recorder with visualization."""

    def __init__(self):
        super().__init__('workspace_calibrator')

        # Declare parameters
        self.declare_parameter('robot_xml_path',
            'src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml')
        self.declare_parameter('joint_state_topic', '/joint_states')

        robot_xml_path = self.get_parameter('robot_xml_path').value
        joint_state_topic = self.get_parameter('joint_state_topic').value

        # Load MuJoCo model for FK
        if not os.path.exists(robot_xml_path):
            self.get_logger().error(f'Robot XML not found: {robot_xml_path}')
            raise FileNotFoundError(robot_xml_path)

        self.mj_model = mj.MjModel.from_xml_path(robot_xml_path)
        self.mj_data = mj.MjData(self.mj_model)

        # Get joint and site IDs
        self._racket_site_id = mj.mj_name2id(
            self.mj_model, mj.mjtObj.mjOBJ_SITE, "right_ee_site")
        self._arm_jids = [
            mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, n)
            for n in RIGHT_ARM_JOINTS
        ]

        if self._racket_site_id < 0 or any(j < 0 for j in self._arm_jids):
            self.get_logger().error('MuJoCo model missing right_ee_site or joints')
            raise RuntimeError('Invalid MuJoCo model')

        self.get_logger().info(f'Loaded MuJoCo model: {robot_xml_path}')

        # Subscribe to joint states
        self.joint_state_sub = self.create_subscription(
            JointState, joint_state_topic,
            self._joint_state_callback, 10)

        # Latest joint positions
        self._latest_joint_positions = {}
        self._joint_state_received = False

        # Recorded boundary points
        self.recorded_points = []

        # Continuous tracking of EE position
        self.ee_trajectory = []
        self.tracking_enabled = False

        self.get_logger().info('Workspace calibrator ready')
        self.get_logger().info(f'Subscribing to: {joint_state_topic}')

    def _joint_state_callback(self, msg: JointState):
        """Cache latest joint positions and track EE trajectory."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                val = float(msg.position[i])
                if np.isfinite(val):
                    self._latest_joint_positions[name] = val
        self._joint_state_received = True

        # Track EE position for visualization
        if self.tracking_enabled:
            try:
                ee_pos = self.get_current_ee_position()
                self.ee_trajectory.append(ee_pos.copy())
                # Keep last 1000 points
                if len(self.ee_trajectory) > 1000:
                    self.ee_trajectory.pop(0)
            except:
                pass

    def get_current_ee_position(self) -> np.ndarray:
        """Compute current end-effector position via FK."""
        if not self._joint_state_received:
            raise RuntimeError('No joint state received yet')

        try:
            arm_q = np.array([
                self._latest_joint_positions[n] for n in RIGHT_ARM_JOINTS
            ], dtype=np.float64)
        except KeyError as e:
            raise RuntimeError(f'Missing joint: {e}')

        # Set joint positions in MuJoCo data
        for i, jid in enumerate(self._arm_jids):
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr] = float(arm_q[i])

        # Run forward kinematics
        mj.mj_forward(self.mj_model, self.mj_data)

        # Get end-effector position (world frame)
        ee_pos_world = np.asarray(
            self.mj_data.site_xpos[self._racket_site_id], dtype=np.float64)

        # Transform to base frame (simplified: assume base at origin)
        ee_pos_base = ee_pos_world.copy()

        return ee_pos_base

    def record_boundary_point(self, label: str) -> None:
        """Record current EE position with a label."""
        try:
            pos = self.get_current_ee_position()
            self.recorded_points.append((label, pos))
            self.get_logger().info(
                f'Recorded {label}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}m')
        except Exception as e:
            self.get_logger().error(f'Failed to record point: {e}')

    def compute_limits(self) -> dict:
        """Compute workspace limits from recorded points."""
        if len(self.recorded_points) < 2:
            self.get_logger().warn('Need at least 2 points to compute limits')
            return {}

        points = np.array([p[1] for p in self.recorded_points])

        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
        z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())

        # Add safety margin (5cm on each side)
        margin = 0.05

        return {
            'x': [x_min + margin, x_max - margin],
            'y': [y_min + margin, y_max - margin],
            'z': [z_min + margin, z_max - margin],
        }

    def plot_workspace(self):
        """Visualize the recorded workspace in 3D."""
        if len(self.recorded_points) < 2:
            print("\nNeed at least 2 recorded points to plot")
            return

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot recorded boundary points
        points = np.array([p[1] for p in self.recorded_points])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c='red', marker='o', s=100, label='Boundary Points')

        # Annotate each point
        for label, pos in self.recorded_points:
            ax.text(pos[0], pos[1], pos[2], f'  {label}', fontsize=8)

        # Plot trajectory if available
        if len(self.ee_trajectory) > 0:
            traj = np.array(self.ee_trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    'b-', alpha=0.3, linewidth=1, label='EE Trajectory')

        # Plot workspace bounding box
        if len(self.recorded_points) >= 2:
            limits = self.compute_limits()
            if limits:
                # Draw wireframe box
                x_range = limits['x']
                y_range = limits['y']
                z_range = limits['z']

                # Define box vertices
                vertices = [
                    [x_range[0], y_range[0], z_range[0]],
                    [x_range[1], y_range[0], z_range[0]],
                    [x_range[1], y_range[1], z_range[0]],
                    [x_range[0], y_range[1], z_range[0]],
                    [x_range[0], y_range[0], z_range[1]],
                    [x_range[1], y_range[0], z_range[1]],
                    [x_range[1], y_range[1], z_range[1]],
                    [x_range[0], y_range[1], z_range[1]],
                ]
                vertices = np.array(vertices)

                # Draw bottom face
                ax.plot([vertices[0, 0], vertices[1, 0]],
                       [vertices[0, 1], vertices[1, 1]],
                       [vertices[0, 2], vertices[1, 2]], 'g--', alpha=0.5)
                ax.plot([vertices[1, 0], vertices[2, 0]],
                       [vertices[1, 1], vertices[2, 1]],
                       [vertices[1, 2], vertices[2, 2]], 'g--', alpha=0.5)
                ax.plot([vertices[2, 0], vertices[3, 0]],
                       [vertices[2, 1], vertices[3, 1]],
                       [vertices[2, 2], vertices[3, 2]], 'g--', alpha=0.5)
                ax.plot([vertices[3, 0], vertices[0, 0]],
                       [vertices[3, 1], vertices[0, 1]],
                       [vertices[3, 2], vertices[0, 2]], 'g--', alpha=0.5)

                # Draw top face
                ax.plot([vertices[4, 0], vertices[5, 0]],
                       [vertices[4, 1], vertices[5, 1]],
                       [vertices[4, 2], vertices[5, 2]], 'g--', alpha=0.5)
                ax.plot([vertices[5, 0], vertices[6, 0]],
                       [vertices[5, 1], vertices[6, 1]],
                       [vertices[5, 2], vertices[6, 2]], 'g--', alpha=0.5)
                ax.plot([vertices[6, 0], vertices[7, 0]],
                       [vertices[6, 1], vertices[7, 1]],
                       [vertices[6, 2], vertices[7, 2]], 'g--', alpha=0.5)
                ax.plot([vertices[7, 0], vertices[4, 0]],
                       [vertices[7, 1], vertices[4, 1]],
                       [vertices[7, 2], vertices[4, 2]], 'g--', alpha=0.5)

                # Draw vertical edges
                for i in range(4):
                    ax.plot([vertices[i, 0], vertices[i+4, 0]],
                           [vertices[i, 1], vertices[i+4, 1]],
                           [vertices[i, 2], vertices[i+4, 2]], 'g--', alpha=0.5)

        # Plot base origin
        ax.scatter([0], [0], [0], c='black', marker='x', s=200,
                   linewidths=3, label='Base Origin')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('End-Effector Workspace Calibration')
        ax.legend()
        ax.grid(True)

        # Set equal aspect ratio
        max_range = 0.5  # meters
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range * 2])

        plt.tight_layout()
        plt.show()


def main():
    rclpy.init()
    node = WorkspaceCalibrator()

    print("\n" + "="*70)
    print("End-Effector Workspace Calibration Tool")
    print("="*70)
    print("\nInstructions:")
    print("1. Use your manual control to move the robot arm")
    print("2. Move to boundary positions and press Enter to record")
    print("3. Type 'current' to see current EE position")
    print("4. Type 'plot' to visualize the workspace shape")
    print("5. Type 'track' to enable/disable trajectory tracking")
    print("6. Type 'done' when finished to output limits\n")

    # Start ROS spinner in background thread
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    # Wait for joint states
    print("Waiting for /joint_states...")
    import time
    timeout = 10.0
    start = time.time()
    while not node._joint_state_received and (time.time() - start) < timeout:
        time.sleep(0.1)

    if not node._joint_state_received:
        print("\nERROR: No joint states received after 10s")
        print("Make sure robot is running and publishing to /joint_states")
        node.destroy_node()
        rclpy.shutdown()
        return

    print(f"✓ Connected to /joint_states\n")

    # Show current position
    try:
        pos = node.get_current_ee_position()
        print(f"Current EE position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}m\n")
    except Exception as e:
        print(f"Warning: Could not compute initial EE position: {e}\n")

    point_counter = 1

    try:
        while True:
            print(f">>> Command (or press Enter to record point #{point_counter}):")
            user_input = input().strip().lower()

            if user_input == 'done':
                print("\nFinishing calibration...")
                break

            elif user_input == 'current':
                try:
                    pos = node.get_current_ee_position()
                    print(f"Current EE: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}m\n")
                except Exception as e:
                    print(f"Error: {e}\n")

            elif user_input == 'plot':
                print("Opening 3D visualization...")
                node.plot_workspace()

            elif user_input == 'track':
                node.tracking_enabled = not node.tracking_enabled
                status = "ENABLED" if node.tracking_enabled else "DISABLED"
                print(f"Trajectory tracking {status}")
                if node.tracking_enabled:
                    node.ee_trajectory.clear()
                    print("(Move the arm around to see its motion in 'plot')\n")
                else:
                    print(f"(Recorded {len(node.ee_trajectory)} trajectory points)\n")

            elif user_input == '':
                # Record current position
                label = f"Point_{point_counter}"
                node.record_boundary_point(label)
                point_counter += 1
                print()

            else:
                print(f"Unknown command: {user_input}")
                print("Valid commands: current, plot, track, done, or Enter to record\n")

        # Compute and display limits
        print("\n" + "="*70)
        print("Calibration Results")
        print("="*70)

        if len(node.recorded_points) < 2:
            print("\nWARNING: Only recorded {} point(s)".format(len(node.recorded_points)))
            print("Need at least 2 points to compute meaningful limits")
        else:
            limits = node.compute_limits()

            print(f"\nRecorded {len(node.recorded_points)} boundary points:")
            for label, pos in node.recorded_points:
                print(f"  {label:20s}: x={pos[0]:6.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}m")

            print("\nComputed workspace limits (with 5cm safety margin):")
            print(f"  X: [{limits['x'][0]:6.3f}, {limits['x'][1]:6.3f}] m")
            print(f"  Y: [{limits['y'][0]:6.3f}, {limits['y'][1]:6.3f}] m")
            print(f"  Z: [{limits['z'][0]:6.3f}, {limits['z'][1]:6.3f}] m")

            # Workspace dimensions
            x_size = limits['x'][1] - limits['x'][0]
            y_size = limits['y'][1] - limits['y'][0]
            z_size = limits['z'][1] - limits['z'][0]
            print(f"\nWorkspace dimensions:")
            print(f"  X width:  {x_size*1000:.0f} mm")
            print(f"  Y depth:  {y_size*1000:.0f} mm")
            print(f"  Z height: {z_size*1000:.0f} mm")

            print("\n" + "-"*70)
            print("Add these lines to config_done_monitoring.yaml:")
            print("-"*70)
            print(f"    done_ee_x_limits_m: [{limits['x'][0]:.3f}, {limits['x'][1]:.3f}]")
            print(f"    done_ee_y_limits_m: [{limits['y'][0]:.3f}, {limits['y'][1]:.3f}]")
            print(f"    done_ee_z_limits_m: [{limits['z'][0]:.3f}, {limits['z'][1]:.3f}]")
            print("-"*70)

            # Offer final plot
            print("\nShow final 3D visualization? (y/n):", end=' ')
            if input().strip().lower() == 'y':
                node.plot_workspace()

    except KeyboardInterrupt:
        print("\nCalibration cancelled")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
