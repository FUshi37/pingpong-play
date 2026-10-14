#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
import math

from mc_core_interface.msg import Base
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header


class BaseSinTrajectory(Node):
    def __init__(self):
        super().__init__('base_sin_trajectory')

        pub_qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.publisher = self.create_publisher(
            Base,
            '/mx_base_command',
            pub_qos_profile
        )

        self.hz = 10.0
        self.timer = self.create_timer(1.0 / self.hz, self.publish_command)
        self.t = 0.0

        # Sine trajectory parameters: amplitude, frequency (Hz), offset
        self.x_amp = 1.0
        self.x_freq = 0.1
        self.x_offset = 0.0

        self.y_amp = 0.5
        self.y_freq = 0.1
        self.y_offset = 0.0

        self.yaw_amp_deg = 30.0
        self.yaw_freq = 0.05
        self.yaw_offset_deg = 0.0

        self.get_logger().info('Base Sin Trajectory Node started')
        self.get_logger().info(
            f'X: amp={self.x_amp}, freq={self.x_freq}Hz | '
            f'Y: amp={self.y_amp}, freq={self.y_freq}Hz | '
            f'Yaw: amp={self.yaw_amp_deg}deg, freq={self.yaw_freq}Hz'
        )

    def publish_command(self):
        x = self.x_offset + self.x_amp * math.sin(2 * math.pi * self.x_freq * self.t)
        y = self.y_offset + self.y_amp * math.sin(2 * math.pi * self.y_freq * self.t)
        yaw = math.radians(self.yaw_offset_deg + self.yaw_amp_deg * math.sin(2 * math.pi * self.yaw_freq * self.t))

        msg = Base()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = 0

        msg.pose = Pose()
        msg.pose.position = Point(x=x, y=y, z=0.0)
        msg.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
        )

        self.publisher.publish(msg)
        self.t += 1.0 / self.hz


def main(args=None):
    rclpy.init(args=args)
    node = BaseSinTrajectory()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
