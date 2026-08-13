#!/usr/bin/env python3
"""Publish mocap UDP ball state as ROS Odometry."""

from __future__ import annotations

import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from pingpong_controller.mocap_ball_detector import MocapBallDetector


class MocapBallStateNode(Node):
    """Small ROS node for validating the mocap-to-ball-state path."""

    def __init__(self):
        super().__init__('mocap_ball_state')

        self.declare_parameter('mocap_bind_host', '0.0.0.0')
        self.declare_parameter('mocap_udp_port', 9870)
        self.declare_parameter('mocap_calibration_path', '')
        self.declare_parameter('mocap_timeout_s', 0.5)
        self.declare_parameter('mocap_marker_id', 1)
        self.declare_parameter('mocap_position_unit', 'auto')
        self.declare_parameter('mocap_timestamp_unit', 's')
        self.declare_parameter('mocap_velocity_alpha', 1.0)
        self.declare_parameter('mocap_max_velocity_m_s', 20.0)
        self.declare_parameter('mocap_velocity_window_frames', 3)
        self.declare_parameter('mocap_velocity_window_max_span_s', 0.05)
        self.declare_parameter('ball_state_topic', '/pingpong/ball_state')
        self.declare_parameter('publish_invalid', True)
        self.declare_parameter('publish_rate_hz', 100.0)
        self.declare_parameter('diagnostic_log_period_s', 1.0)

        bind_host = str(self.get_parameter('mocap_bind_host').value)
        udp_port = int(self.get_parameter('mocap_udp_port').value)
        calibration_path = str(
            self.get_parameter('mocap_calibration_path').value).strip()
        timeout_s = float(self.get_parameter('mocap_timeout_s').value)
        marker_id = int(self.get_parameter('mocap_marker_id').value)
        position_unit = str(
            self.get_parameter('mocap_position_unit').value)
        timestamp_unit = str(
            self.get_parameter('mocap_timestamp_unit').value)
        velocity_alpha = float(
            self.get_parameter('mocap_velocity_alpha').value)
        max_velocity_m_s = float(
            self.get_parameter('mocap_max_velocity_m_s').value)
        velocity_window_frames = int(
            self.get_parameter('mocap_velocity_window_frames').value)
        velocity_window_max_span_s = float(
            self.get_parameter('mocap_velocity_window_max_span_s').value)
        self.ball_state_topic = str(
            self.get_parameter('ball_state_topic').value)
        self.publish_invalid = bool(
            self.get_parameter('publish_invalid').value)
        publish_rate_hz = max(
            1.0, float(self.get_parameter('publish_rate_hz').value))
        self.diagnostic_log_period_s = max(
            0.0, float(
                self.get_parameter('diagnostic_log_period_s').value))

        self.detector = MocapBallDetector(
            udp_host=bind_host,
            udp_port=udp_port,
            calibration_path=calibration_path or None,
            timeout_s=timeout_s,
            marker_id=marker_id,
            position_unit=position_unit,
            timestamp_unit=timestamp_unit,
            velocity_alpha=velocity_alpha,
            max_velocity_m_s=max_velocity_m_s,
            velocity_window_frames=velocity_window_frames,
            velocity_window_max_span_s=velocity_window_max_span_s,
        )
        if not self.detector.start():
            raise RuntimeError(
                f'Failed to bind mocap UDP socket at {bind_host}:{udp_port}')

        self.pub = self.create_publisher(Odometry, self.ball_state_topic, 10)
        self.timer = self.create_timer(
            1.0 / publish_rate_hz, self._timer_callback)
        self._last_packet_count = 0
        self._last_log_time_s = 0.0

        self.get_logger().info(
            f'Mocap ball state publishing "{self.ball_state_topic}" at '
            f'{publish_rate_hz:.1f} Hz; listening on '
            f'{bind_host}:{udp_port}, marker={marker_id}, '
            f'unit={position_unit}, timestamp_unit={timestamp_unit}, '
            f'velocity_window={velocity_window_frames} frames, '
            f'alpha={velocity_alpha:.2f}')

    def _timer_callback(self) -> None:
        pos_m, vel_m_s, valid = self.detector.get_ball_state()
        diagnostics = self.detector.get_diagnostics()
        if not valid and not self.publish_invalid:
            self._maybe_log_diagnostics(diagnostics)
            return

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.child_frame_id = 'pingpong_ball'
        msg.pose.pose.position.x = float(pos_m[0])
        msg.pose.pose.position.y = float(pos_m[1])
        msg.pose.pose.position.z = float(pos_m[2])
        msg.twist.twist.linear.x = float(vel_m_s[0])
        msg.twist.twist.linear.y = float(vel_m_s[1])
        msg.twist.twist.linear.z = float(vel_m_s[2])
        if not valid:
            msg.pose.covariance[0] = 1e6
            msg.twist.covariance[0] = 1e6
        self.pub.publish(msg)

        self._maybe_log_diagnostics(diagnostics)

    def _maybe_log_diagnostics(self, diagnostics: dict) -> None:
        packet_count = int(diagnostics.get('packet_count', 0) or 0)
        now_s = time.time()
        first_packet = packet_count > 0 and self._last_packet_count == 0
        due = (
            self.diagnostic_log_period_s > 0.0
            and now_s - self._last_log_time_s >= self.diagnostic_log_period_s)
        if not first_packet and not due:
            return

        age_s = diagnostics.get('age_s')
        age_text = (
            f'{float(age_s):.3f}s'
            if age_s is not None and math.isfinite(float(age_s))
            else 'n/a')
        self.get_logger().info(
            f'mocap packets={packet_count}, '
            f'valid={bool(diagnostics.get("valid", False))}, '
            f'age={age_text}, '
            f'parse_errors='
            f'{int(diagnostics.get("parse_error_count", 0) or 0)}, '
            f'vel_rejects='
            f'{int(diagnostics.get("velocity_reject_count", 0) or 0)}, '
            f'vel_window_rejects='
            f'{int(diagnostics.get("velocity_window_reject_count", 0) or 0)}, '
            f'frame={diagnostics.get("frame_idx")}, '
            f'from={diagnostics.get("addr")}')
        self._last_packet_count = packet_count
        self._last_log_time_s = now_s

    def destroy_node(self):
        if getattr(self, 'detector', None) is not None:
            self.detector.stop()
            self.detector = None
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = MocapBallStateNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
