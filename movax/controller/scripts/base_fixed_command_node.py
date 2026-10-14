#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
import math

from mc_core_interface.msg import Base
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header


class BaseFixedCommand(Node):
    def __init__(self):
        super().__init__('base_fixed_command')

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

        # Publish at 100 Hz
        self.timer = self.create_timer(0.01, self.publish_command)

        # Fixed command parameters
        self.target_x = -3.5
        self.target_y = 4.5
        self.target_z = 0.0
        self.target_yaw_deg = 0.0

        self.get_logger().info('Base Fixed Command Node started - Publishing to /mx_base_command')
        self.get_logger().info(
            f'Target: pos=({self.target_x}, {self.target_y}, {self.target_z}), yaw={self.target_yaw_deg} deg'
        )

    def publish_command(self):
        msg = Base()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = 1

        msg.pose = Pose()
        msg.pose.position = Point(x=self.target_x, y=self.target_y, z=self.target_z)

        yaw = math.radians(self.target_yaw_deg)
        msg.pose.orientation = Quaternion(
            x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)
        )

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BaseFixedCommand()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
