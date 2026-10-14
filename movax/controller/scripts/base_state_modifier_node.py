#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
import math

from mc_core_interface.msg import Base
from geometry_msgs.msg import Pose, Point, Quaternion


class BaseStateModifier(Node):
    def __init__(self):
        super().__init__('base_state_modifier')

        sub_qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        pub_qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            Base,
            '/mx_base_state',
            self.callback,
            sub_qos_profile
        )

        self.publisher = self.create_publisher(
            Base,
            '/mx_base_command',
            pub_qos_profile
        )

        self.get_logger().info('Base State Modifier Node started')

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions: q1 * q2"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2

        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        )

    def callback(self, msg):
        modified_msg = Base()

        modified_msg.header = msg.header
        modified_msg.mode = 0

        modified_msg.pose = Pose()
        modified_msg.pose.position = Point()
        modified_msg.pose.position.x = msg.pose.position.x
        modified_msg.pose.position.y = msg.pose.position.y - 0.1
        modified_msg.pose.position.z = msg.pose.position.z

        # Rotate orientation 30 degrees around z-axis
        # Quaternion for rotation θ around z: [0, 0, sin(θ/2), cos(θ/2)]
        angle = math.radians(30.0)
        q_z_90 = (0.0, 0.0, math.sin(angle/2), math.cos(angle/2))
        q_original = (msg.pose.orientation.x, msg.pose.orientation.y,
                     msg.pose.orientation.z, msg.pose.orientation.w)

        q_rotated = self.quaternion_multiply(q_z_90, q_original)

        modified_msg.pose.orientation = Quaternion()
        modified_msg.pose.orientation.x = q_rotated[0]
        modified_msg.pose.orientation.y = q_rotated[1]
        modified_msg.pose.orientation.z = q_rotated[2]
        modified_msg.pose.orientation.w = q_rotated[3]

        modified_msg.twist = msg.twist

        self.publisher.publish(modified_msg)

        self.get_logger().debug(
            f'Modified y: {msg.pose.position.y:.3f} -> {modified_msg.pose.position.y:.3f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = BaseStateModifier()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
