#!/usr/bin/env python3
"""
增强版球轨迹3D可视化工具
- 显示3D轨迹曲线
- 用颜色表示速度大小
- 可选显示速度箭头
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
import csv
import os
import math


class EnhancedTrajectoryVisualizer(Node):
    def __init__(self):
        super().__init__('enhanced_trajectory_visualizer')

        # 声明参数
        self.declare_parameter('csv_file',
            'src/pingpong_moz1/pingpong_controller/outputs/pingpong_base_trajectory.csv')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('line_width', 0.005)
        self.declare_parameter('update_rate', 1.0)
        self.declare_parameter('show_velocity_arrows', False)
        self.declare_parameter('arrow_scale', 0.01)
        self.declare_parameter('arrow_spacing', 10)  # 每N个点显示一个箭头
        self.declare_parameter('color_by_velocity', True)
        self.declare_parameter('time_start', -1.0)  # 开始时间（秒），-1表示从头开始
        self.declare_parameter('time_end', -1.0)    # 结束时间（秒），-1表示到末尾
        self.declare_parameter('time_range_mode', 'relative')  # 'relative'或'absolute'

        # 获取参数
        self.csv_file = self.get_parameter('csv_file').value
        self.frame_id = self.get_parameter('frame_id').value
        self.line_width = self.get_parameter('line_width').value
        update_rate = self.get_parameter('update_rate').value
        self.show_velocity_arrows = self.get_parameter('show_velocity_arrows').value
        self.arrow_scale = self.get_parameter('arrow_scale').value
        self.arrow_spacing = self.get_parameter('arrow_spacing').value
        self.color_by_velocity = self.get_parameter('color_by_velocity').value
        self.time_start = self.get_parameter('time_start').value
        self.time_end = self.get_parameter('time_end').value
        self.time_range_mode = self.get_parameter('time_range_mode').value

        # 创建发布器
        self.marker_pub = self.create_publisher(Marker, 'ball_trajectory', 10)
        self.arrow_pub = self.create_publisher(MarkerArray, 'velocity_arrows', 10)

        # 创建定时器
        self.timer = self.create_timer(1.0 / update_rate, self.publish_visualization)

        self.get_logger().info(f'Enhanced trajectory visualizer initialized')
        self.get_logger().info(f'CSV file: {self.csv_file}')

    def load_trajectory_data(self):
        """从CSV文件加载完整的轨迹数据，支持时间范围过滤"""
        data = []
        all_data = []

        if not os.path.exists(self.csv_file):
            self.get_logger().error(f'CSV file not found: {self.csv_file}')
            return data

        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamp = float(row['timestamp'])

                    # 位置 (mm -> m)
                    x = float(row['x_mm']) / 1000.0
                    y = float(row['y_mm']) / 1000.0
                    z = float(row['z_mm']) / 1000.0

                    # 速度 (mm/s -> m/s)
                    vx = float(row['vx_mm_s']) / 1000.0
                    vy = float(row['vy_mm_s']) / 1000.0
                    vz = float(row['vz_mm_s']) / 1000.0

                    speed = math.sqrt(vx*vx + vy*vy + vz*vz)

                    all_data.append({
                        'timestamp': timestamp,
                        'x': x, 'y': y, 'z': z,
                        'vx': vx, 'vy': vy, 'vz': vz,
                        'speed': speed
                    })

            if len(all_data) == 0:
                self.get_logger().warn('No data loaded from CSV')
                return data

            # 应用时间范围过滤
            data = self.filter_by_time_range(all_data)

            self.get_logger().info(f'Loaded {len(all_data)} total points, filtered to {len(data)} points')
            if len(data) > 0:
                self.get_logger().info(f'Time range: {data[0]["timestamp"]:.2f}s - {data[-1]["timestamp"]:.2f}s')

        except Exception as e:
            self.get_logger().error(f'Failed to load CSV: {e}')

        return data

    def filter_by_time_range(self, all_data):
        """根据时间范围过滤数据"""
        if len(all_data) == 0:
            return []

        # 获取第一个时间戳作为基准
        first_timestamp = all_data[0]['timestamp']

        # 确定实际的时间范围
        if self.time_range_mode == 'relative':
            # 相对模式：从第一个时间戳开始计算
            if self.time_start >= 0:
                actual_start = first_timestamp + self.time_start
            else:
                actual_start = first_timestamp

            if self.time_end >= 0:
                actual_end = first_timestamp + self.time_end
            else:
                actual_end = all_data[-1]['timestamp']
        else:
            # 绝对模式：使用CSV文件中的实际时间戳
            if self.time_start >= 0:
                actual_start = self.time_start
            else:
                actual_start = first_timestamp

            if self.time_end >= 0:
                actual_end = self.time_end
            else:
                actual_end = all_data[-1]['timestamp']

        # 过滤数据
        filtered_data = [
            d for d in all_data
            if actual_start <= d['timestamp'] <= actual_end
        ]

        return filtered_data

    def velocity_to_color(self, speed, max_speed=3.0):
        """根据速度计算颜色（蓝->绿->黄->红）"""
        # 归一化速度到0-1
        normalized = min(speed / max_speed, 1.0)

        color = ColorRGBA()
        color.a = 0.8

        if normalized < 0.25:
            # 蓝色到青色
            t = normalized / 0.25
            color.r = 0.0
            color.g = t
            color.b = 1.0
        elif normalized < 0.5:
            # 青色到绿色
            t = (normalized - 0.25) / 0.25
            color.r = 0.0
            color.g = 1.0
            color.b = 1.0 - t
        elif normalized < 0.75:
            # 绿色到黄色
            t = (normalized - 0.5) / 0.25
            color.r = t
            color.g = 1.0
            color.b = 0.0
        else:
            # 黄色到红色
            t = (normalized - 0.75) / 0.25
            color.r = 1.0
            color.g = 1.0 - t
            color.b = 0.0

        return color

    def publish_visualization(self):
        """发布轨迹和速度箭头"""
        data = self.load_trajectory_data()

        if len(data) == 0:
            return

        # 发布轨迹线
        self.publish_trajectory(data)

        # 如果启用，发布速度箭头
        if self.show_velocity_arrows:
            self.publish_velocity_arrows(data)

    def publish_trajectory(self, data):
        """发布轨迹Marker"""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "ball_trajectory"
        marker.id = 0

        if self.color_by_velocity:
            # 使用LINE_LIST以便每段可以有不同颜色
            marker.type = Marker.LINE_LIST
            marker.scale.x = self.line_width

            # 添加线段和颜色
            for i in range(len(data) - 1):
                p1 = Point()
                p1.x, p1.y, p1.z = data[i]['x'], data[i]['y'], data[i]['z']
                p2 = Point()
                p2.x, p2.y, p2.z = data[i+1]['x'], data[i+1]['y'], data[i+1]['z']

                marker.points.append(p1)
                marker.points.append(p2)

                # 使用平均速度
                avg_speed = (data[i]['speed'] + data[i+1]['speed']) / 2.0
                color = self.velocity_to_color(avg_speed)
                marker.colors.append(color)
                marker.colors.append(color)
        else:
            # 使用单色LINE_STRIP
            marker.type = Marker.LINE_STRIP
            marker.scale.x = self.line_width
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.8

            for d in data:
                p = Point()
                p.x, p.y, p.z = d['x'], d['y'], d['z']
                marker.points.append(p)

        marker.action = Marker.ADD
        self.marker_pub.publish(marker)

    def publish_velocity_arrows(self, data):
        """发布速度箭头"""
        marker_array = MarkerArray()

        for i, d in enumerate(data):
            if i % self.arrow_spacing != 0:
                continue

            if d['speed'] < 0.01:  # 跳过静止点
                continue

            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "velocity_arrows"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # 起点
            start = Point()
            start.x, start.y, start.z = d['x'], d['y'], d['z']
            # 终点 = 起点 + 速度向量 * scale
            end = Point()
            end.x = d['x'] + d['vx'] * self.arrow_scale
            end.y = d['y'] + d['vy'] * self.arrow_scale
            end.z = d['z'] + d['vz'] * self.arrow_scale

            marker.points = [start, end]

            # 箭头尺寸
            marker.scale.x = self.line_width * 0.5  # 箭杆宽度
            marker.scale.y = self.line_width * 1.0  # 箭头宽度
            marker.scale.z = self.line_width * 1.5  # 箭头长度

            # 根据速度设置颜色
            marker.color = self.velocity_to_color(d['speed'])

            marker_array.markers.append(marker)

        self.arrow_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = EnhancedTrajectoryVisualizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()





