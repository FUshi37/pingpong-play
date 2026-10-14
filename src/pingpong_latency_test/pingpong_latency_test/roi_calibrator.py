#!/usr/bin/env python3
"""
ROI Calibration Tool (Robot Side, interactive)

Opens a live camera window. Drag a rectangle with the mouse to select the
laptop-screen region. The window shows the live mean brightness inside the
current ROI. Show a steady WHITE screen and press 'w' to capture the white
level; show a steady BLACK screen and press 'b' to capture the black level.
The tool then prints the exact roi / brightness_threshold / transition_hysteresis
parameters to pass to flash_detector.

Keys:
    drag mouse : draw ROI rectangle
    w          : record current ROI brightness as WHITE level
    b          : record current ROI brightness as BLACK level
    c          : clear ROI (back to full frame)
    p          : print suggested flash_detector parameters
    s          : save current annotated frame to /tmp/roi_calib.png
    q / ESC    : quit (prints suggested parameters)

Usage:
    ros2 run pingpong_latency_test roi_calibrator --ros-args \
        -p image_topic:=/camera/camera/color/image_raw/compressed
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class RoiCalibrator(Node):
    def __init__(self):
        super().__init__('roi_calibrator')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.image_topic = self.get_parameter('image_topic').value

        self.bridge = CvBridge()
        self.latest = None

        # ROI drag state
        self.roi = None              # (x, y, w, h)
        self.drag_start = None
        self.drag_now = None
        self.dragging = False

        # Captured levels
        self.white_level = None
        self.black_level = None

        self.is_compressed = 'compressed' in self.image_topic.lower()
        if self.is_compressed:
            self.create_subscription(CompressedImage, self.image_topic,
                                     self.cb_compressed, 10)
        else:
            self.create_subscription(Image, self.image_topic, self.cb_raw, 10)

        self.win = 'ROI Calibrator'
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self.on_mouse)

        self.get_logger().info(f'ROI Calibrator on {self.image_topic} '
                               f'({"compressed" if self.is_compressed else "raw"})')
        self.get_logger().info('Drag to select screen. w=white, b=black, p=print, q=quit')

    def cb_compressed(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            self.latest = img

    def cb_raw(self, msg):
        self.latest = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_now = (x, y)
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_now = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x0, y0 = self.drag_start
            x1, y1 = x, y
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            if rw > 4 and rh > 4:
                self.roi = (rx, ry, rw, rh)
                self.get_logger().info(f'ROI set: {rx},{ry},{rw},{rh}')

    def roi_brightness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.roi:
            x, y, w, h = self.roi
            y2, x2 = min(y + h, gray.shape[0]), min(x + w, gray.shape[1])
            region = gray[max(0, y):y2, max(0, x):x2]
            if region.size == 0:
                return 0.0
            return float(region.mean()) / 255.0
        return float(gray.mean()) / 255.0

    def suggest(self):
        if self.white_level is None or self.black_level is None:
            self.get_logger().warn(
                f'Need both levels. white={self.white_level} black={self.black_level}. '
                f'Press w on white screen and b on black screen.')
            return
        w, b = self.white_level, self.black_level
        if w <= b:
            self.get_logger().error(
                f'white({w:.3f}) <= black({b:.3f}); contrast too low. '
                f'Reduce camera exposure or pick a tighter ROI on the screen.')
            return
        thresh = (w + b) / 2.0
        hyst = max(0.02, (w - b) / 3.0)
        rx, ry, rw, rh = self.roi if self.roi else (0, 0, 0, 0)
        print('\n' + '=' * 64)
        print(f'  white level = {w:.3f}   black level = {b:.3f}   contrast = {w-b:.3f}')
        print('  Suggested flash_detector parameters:')
        print('-' * 64)
        roi_arg = f'{rx},{ry},{rw},{rh}' if self.roi else '(full frame)'
        print(f'    -p roi:={roi_arg} \\')
        print(f'    -p brightness_threshold:={thresh:.3f} \\')
        print(f'    -p transition_hysteresis:={hyst:.3f}')
        print('=' * 64 + '\n')

    def draw(self):
        if self.latest is None:
            canvas = np.zeros((480, 848, 3), np.uint8)
            cv2.putText(canvas, 'waiting for image...', (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(self.win, canvas)
            return
        img = self.latest.copy()
        bright = self.roi_brightness(img)

        # Draw committed ROI
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw in-progress drag
        if self.dragging and self.drag_start and self.drag_now:
            cv2.rectangle(img, self.drag_start, self.drag_now, (0, 200, 255), 1)

        lines = [
            f'ROI brightness: {bright:.3f}',
            f'white={self.white_level if self.white_level is None else round(self.white_level,3)}  '
            f'black={self.black_level if self.black_level is None else round(self.black_level,3)}',
            'w=white b=black p=print c=clear q=quit',
        ]
        y0 = 25
        for i, t in enumerate(lines):
            cv2.putText(img, t, (10, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3)
            cv2.putText(img, t, (10, y0 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1)
        cv2.imshow(self.win, img)

    def spin(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.005)
            self.draw()
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('q'), 27):
                self.suggest()
                break
            elif key == ord('w'):
                if self.latest is not None:
                    self.white_level = self.roi_brightness(self.latest)
                    self.get_logger().info(f'WHITE level = {self.white_level:.3f}')
            elif key == ord('b'):
                if self.latest is not None:
                    self.black_level = self.roi_brightness(self.latest)
                    self.get_logger().info(f'BLACK level = {self.black_level:.3f}')
            elif key == ord('c'):
                self.roi = None
                self.get_logger().info('ROI cleared (full frame)')
            elif key == ord('p'):
                self.suggest()
            elif key == ord('s'):
                if self.latest is not None:
                    cv2.imwrite('/tmp/roi_calib.png', self.latest)
                    self.get_logger().info('Saved /tmp/roi_calib.png')
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = RoiCalibrator()
    try:
        node.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
