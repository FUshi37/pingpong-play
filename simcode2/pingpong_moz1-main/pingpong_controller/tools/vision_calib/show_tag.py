#!/usr/bin/env python3
"""
ROS2 real-time AprilTag/AprilGrid visualization tool.

Subscribes to RealSense RGB image and displays detected AprilTags.

Usage:
    python3 calibration/show_tag.py
    python3 calibration/show_tag.py --image-topic /camera/color/image_raw
    python3 calibration/show_tag.py --tag-family tag16h5 --max-id 30

Press 'q' in the window to quit.
"""

import sys
import time
from collections import deque

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import CompressedImage
except ImportError as e:
    print("ERROR: ROS2 dependencies not available.", file=sys.stderr)
    print(f"  {e}", file=sys.stderr)
    print("Install: sudo apt install ros-<distro>-rclpy ros-<distro>-sensor-msgs", file=sys.stderr)
    sys.exit(1)

import cv2
import numpy as np


# ============================================================
# Manual ROS Image -> OpenCV conversion (no cv_bridge needed)
# ============================================================

def imgmsg_to_cv2(msg):
    """Convert sensor_msgs/Image to OpenCV image without cv_bridge.

    Supports common encodings: bgr8, rgb8, mono8, 8UC1, 8UC3, 16UC1.
    Returns BGR image for OpenCV.
    """
    dtype_map = {
        'mono8': (np.uint8, 1),
        '8UC1': (np.uint8, 1),
        'bgr8': (np.uint8, 3),
        'rgb8': (np.uint8, 3),
        '8UC3': (np.uint8, 3),
        '16UC1': (np.uint16, 1),
        'mono16': (np.uint16, 1),
    }

    if msg.encoding not in dtype_map:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    dtype, channels = dtype_map[msg.encoding]

    # Decode image data
    img_array = np.frombuffer(msg.data, dtype=dtype)

    if msg.step != msg.width * channels * img_array.itemsize:
        # Handle row padding
        img_array = img_array.reshape(msg.height, msg.step // img_array.itemsize)
        img_array = img_array[:, :msg.width * channels]

    if channels == 1:
        img = img_array.reshape(msg.height, msg.width)
    else:
        img = img_array.reshape(msg.height, msg.width, channels)

    # Convert to BGR if needed
    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def compressed_imgmsg_to_cv2(msg):
    """Convert sensor_msgs/CompressedImage to OpenCV BGR image."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode compressed image")
    return image


# ============================================================
# AprilTag detection
# ============================================================

_OPENCV_APRILTAG_DICTS = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def make_detector(tag_family):
    if tag_family not in _OPENCV_APRILTAG_DICTS:
        raise ValueError(f"Unsupported tag family: {tag_family}")
    d = cv2.aruco.getPredefinedDictionary(_OPENCV_APRILTAG_DICTS[tag_family])
    params = cv2.aruco.DetectorParameters()
    # Calibration-view mode: favor detection robustness over raw speed.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.05

    if hasattr(params, "aprilTagQuadDecimate"):
        params.aprilTagQuadDecimate = 1.0
    if hasattr(params, "aprilTagQuadSigma"):
        params.aprilTagQuadSigma = 0.0
    if hasattr(params, "aprilTagMinClusterPixels"):
        params.aprilTagMinClusterPixels = 5
    if hasattr(params, "aprilTagMaxNmaxima"):
        params.aprilTagMaxNmaxima = 20
    if hasattr(params, "aprilTagCriticalRad"):
        params.aprilTagCriticalRad = np.deg2rad(10)
    if hasattr(params, "aprilTagMaxLineFitMse"):
        params.aprilTagMaxLineFitMse = 30.0
    if hasattr(params, "aprilTagMinWhiteBlackDiff"):
        params.aprilTagMinWhiteBlackDiff = 5
    return cv2.aruco.ArucoDetector(d, params)


def detect_tags(image, detector, max_id, invert=False):
    """Detect AprilTags and return (tag_id, corners, center) tuples."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if invert:
        gray = 255 - gray
    corners, ids, _ = detector.detectMarkers(gray)
    detections = []
    if ids is not None:
        for i, tag_id in enumerate(ids.flatten()):
            if tag_id >= max_id:
                continue
            c = corners[i][0].astype(np.float64)
            center = c.mean(axis=0)
            detections.append((int(tag_id), c, center))
    return detections


def deduplicate_tags(detections):
    """Keep first detection per tag_id, warn about duplicates."""
    seen = {}
    duplicates = []
    for tag_id, corners, center in detections:
        if tag_id not in seen:
            seen[tag_id] = (tag_id, corners, center)
        else:
            duplicates.append(tag_id)
    return list(seen.values()), duplicates


# ============================================================
# ROS2 Node
# ============================================================

class AprilTagViewer(Node):
    def __init__(self, image_topic, tag_family, max_id, invert, skip_frames=2, scale=0.5):
        super().__init__('apriltag_viewer')
        self.detector = make_detector(tag_family)
        self.max_id = max_id
        self.invert = invert
        self.tag_family = tag_family
        self.skip_frames = skip_frames
        self.scale = scale
        self.frame_counter = 0

        if image_topic.endswith("/compressed"):
            self.subscription = self.create_subscription(
                CompressedImage, image_topic, self.compressed_image_callback, 10)
            self.compressed = True
        else:
            self.subscription = self.create_subscription(
                Image, image_topic, self.image_callback, 10)
            self.compressed = False

        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        self.last_detections = []
        self.last_vis = None

        self.get_logger().info(f"Subscribed to {image_topic}")
        self.get_logger().info(f"Message type: {'CompressedImage' if self.compressed else 'Image'}")
        self.get_logger().info(f"Tag family: {tag_family}, max_id: {max_id}")
        self.get_logger().info(f"Performance: skip_frames={skip_frames}, scale={scale}")
        self.get_logger().info("Press 'q' in window to quit")

    def compressed_image_callback(self, msg):
        try:
            cv_image = compressed_imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")
            return
        self.process_image(cv_image)

    def image_callback(self, msg):
        try:
            cv_image = imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")
            return
        self.process_image(cv_image)

    def process_image(self, cv_image):
        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % (self.skip_frames + 1) != 0:
            # Reuse last detection results
            if self.last_vis is not None:
                cv2.imshow("AprilTag Viewer", self.last_vis)
                cv2.waitKey(1)
            return

        # Downsample for faster detection
        if self.scale < 1.0:
            small = cv2.resize(cv_image, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_AREA)
            detections = detect_tags(small, self.detector, self.max_id, self.invert)
            # Scale corners back to original resolution
            detections = [(tid, corners / self.scale, center / self.scale)
                         for tid, corners, center in detections]
        else:
            detections = detect_tags(cv_image, self.detector, self.max_id, self.invert)

        detections, duplicates = deduplicate_tags(detections)
        self.last_detections = detections

        if duplicates:
            self.get_logger().warning(f"Duplicate tag IDs: {duplicates}")

        # Draw visualization
        vis = cv_image.copy()
        tag_ids = sorted([d[0] for d in detections])

        for tag_id, corners, center in detections:
            pts = corners.astype(int)
            for i in range(4):
                cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1)%4]), (0, 255, 0), 2)
            cv2.putText(vis, str(tag_id), tuple(pts[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(vis, tuple(center.astype(int)), 4, (0, 0, 255), -1)

        # FPS calculation
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_queue.append(1.0 / dt)
        fps = np.mean(self.fps_queue) if self.fps_queue else 0.0

        # Text overlay
        cv2.putText(vis, f"Tags: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"IDs: {tag_ids}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.last_vis = vis
        cv2.imshow("AprilTag Viewer", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    # Parse args before rclpy.init to avoid conflicts
    parser = argparse.ArgumentParser(description="ROS2 AprilTag Viewer")
    parser.add_argument('--image-topic', default='/camera/camera/color/image_raw',
                       help='RGB image topic')
    parser.add_argument('--tag-family', default='tag36h11',
                       choices=['tag16h5', 'tag25h9', 'tag36h11'],
                       help='AprilTag family')
    parser.add_argument('--max-id', type=int, default=36,
                       help='Maximum tag ID (AprilGrid range)')
    parser.add_argument('--invert', action='store_true',
                       help='Detect on inverted image')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Skip N frames between detections (default: 0, higher=faster)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Downsample scale for detection (default: 1.0, lower=faster)')

    # Filter out ROS args
    ros_args = [arg for arg in sys.argv if arg.startswith('--ros-args') or arg.startswith('-r')]
    app_args = [arg for arg in sys.argv if arg not in ros_args]

    args = parser.parse_args(app_args[1:])

    rclpy.init(args=sys.argv)

    node = AprilTagViewer(
        image_topic=args.image_topic,
        tag_family=args.tag_family,
        max_id=args.max_id,
        invert=args.invert,
        skip_frames=args.skip_frames,
        scale=args.scale,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
