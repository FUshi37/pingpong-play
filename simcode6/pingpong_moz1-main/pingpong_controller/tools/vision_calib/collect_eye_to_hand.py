#!/usr/bin/env python3
import json
import os
import select
import termios
import tty
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState


IMAGE_TOPIC = "/camera/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
JOINT_TOPIC = "/joint_states"

CURRENT_FILE = Path(__file__).resolve()
PACKAGE_DIR = CURRENT_FILE.parent.parent.parent
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = PACKAGE_DIR / "data" / "vision_calib" / f"eye_to_hand_samples_{TIMESTAMP}"
IMAGE_DIR = SAVE_DIR / "images"
TXT_PATH = SAVE_DIR / "samples.txt"
INTRINSICS_PATH = SAVE_DIR / "camera_intrinsics_raw.json"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def stamp_to_float(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def group_joints(names, positions, velocities, efforts):
    data = {}
    for i, name in enumerate(names):
        data[name] = {
            "position": positions[i] if i < len(positions) else None,
            "velocity": velocities[i] if i < len(velocities) else None,
            "effort": efforts[i] if i < len(efforts) else None,
        }

    def extract(prefix):
        keys = [k for k in names if k.startswith(prefix)]
        return {
            "names": keys,
            "positions": [data[k]["position"] for k in keys],
            "velocities": [data[k]["velocity"] for k in keys],
            "efforts": [data[k]["effort"] for k in keys],
        }

    return {
        "Base": extract("Base-"),
        "LegWaist": extract("LegWaist-"),
        "LeftArm": extract("LeftArm-"),
        "RightArm": extract("RightArm-"),
    }


def camera_info_to_raw_intrinsics_dict(msg):
    width = int(msg.width)
    height = int(msg.height)

    return {
        "image_width": width,
        "image_height": height,
        "frame_id": msg.header.frame_id,
        "distortion_model": msg.distortion_model,
        "camera_matrix": [
            list(msg.k[0:3]),
            list(msg.k[3:6]),
            list(msg.k[6:9]),
        ],
        "dist_coeffs": list(msg.d),
        "rectification_matrix": [
            list(msg.r[0:3]),
            list(msg.r[3:6]),
            list(msg.r[6:9]),
        ],
        "projection_matrix": [
            list(msg.p[0:4]),
            list(msg.p[4:8]),
            list(msg.p[8:12]),
        ],
    }


class Collector(Node):
    def __init__(self):
        super().__init__("eye_to_hand_data_collector")

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_joint = None
        self.latest_camera_info = None
        self.intrinsics_saved = False
        self.sample_idx = self.get_existing_count()

        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)
        self.create_subscription(JointState, JOINT_TOPIC, self.joint_callback, 10)

        if not TXT_PATH.exists():
            with open(TXT_PATH, "w") as f:
                f.write("# Eye-to-hand calibration samples. Each line is one JSON record.\n")

        self.get_logger().info(f"Image topic: {IMAGE_TOPIC}")
        self.get_logger().info(f"CameraInfo topic: {CAMERA_INFO_TOPIC}")
        self.get_logger().info(f"Joint topic: {JOINT_TOPIC}")
        self.get_logger().info(f"Save dir: {SAVE_DIR}")
        self.get_logger().info(f"Existing samples: {self.sample_idx}")
        self.get_logger().info("Press SPACE to save current image + joint_states. Press q to quit.")

    def get_existing_count(self):
        if not IMAGE_DIR.exists():
            return 0

        nums = []
        for path in IMAGE_DIR.iterdir():
            filename = path.name
            if filename.endswith(".jpg") and filename[:6].isdigit():
                nums.append(int(filename[:6]))

        return max(nums) if nums else 0

    def image_callback(self, msg):
        self.latest_image = msg

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg
        if not self.intrinsics_saved:
            self.save_intrinsics()

    def joint_callback(self, msg):
        self.latest_joint = msg

    def save_intrinsics(self):
        if self.latest_camera_info is None:
            self.get_logger().warn("No CameraInfo received yet.")
            return False

        intrinsics = camera_info_to_raw_intrinsics_dict(self.latest_camera_info)
        with open(INTRINSICS_PATH, "w") as f:
            json.dump(intrinsics, f, indent=2, ensure_ascii=False)

        self.intrinsics_saved = True
        self.get_logger().info(f"Saved raw camera intrinsics: {INTRINSICS_PATH}")
        return True

    def save_once(self):
        if self.latest_image is None:
            self.get_logger().warn("No image received yet.")
            return

        if self.latest_joint is None:
            self.get_logger().warn("No joint_states received yet.")
            return

        if not self.intrinsics_saved:
            self.save_intrinsics()

        self.sample_idx += 1
        image_name = f"{self.sample_idx:06d}.jpg"
        image_path = IMAGE_DIR / image_name

        try:
            img = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            self.sample_idx -= 1
            return

        if not cv2.imwrite(str(image_path), img):
            self.get_logger().error(f"Failed to save image: {image_path}")
            self.sample_idx -= 1
            return

        names = list(self.latest_joint.name)
        positions = list(self.latest_joint.position)
        velocities = list(self.latest_joint.velocity)
        efforts = list(self.latest_joint.effort)

        image_stamp = stamp_to_float(self.latest_image.header.stamp)
        joint_stamp = stamp_to_float(self.latest_joint.header.stamp)
        dt = image_stamp - joint_stamp

        grouped = group_joints(names, positions, velocities, efforts)

        record = {
            "index": self.sample_idx,
            "image_file": image_name,
            "image_path": str(image_path),
            "image_topic": IMAGE_TOPIC,
            "camera_info_topic": CAMERA_INFO_TOPIC,
            "joint_topic": JOINT_TOPIC,
            "image_stamp": image_stamp,
            "joint_stamp": joint_stamp,
            "time_diff_image_minus_joint": dt,
            "camera_intrinsics_path": str(INTRINSICS_PATH) if self.intrinsics_saved else None,
            "joint_names": names,
            "joint_positions": positions,
            "joint_velocities": velocities,
            "joint_efforts": efforts,
            "grouped_joints": grouped,
            "saved_wall_time": datetime.now().isoformat(),
        }

        with open(TXT_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        left_arm = grouped["LeftArm"]["positions"]
        right_arm = grouped["RightArm"]["positions"]

        self.get_logger().info(
            "Saved sample {:06d}, dt={:.6f}s, LeftArm={}, RightArm={}".format(
                self.sample_idx,
                dt,
                ["{:.3f}".format(x) for x in left_arm],
                ["{:.3f}".format(x) for x in right_arm],
            )
        )


def read_key(tty_file):
    dr, _, _ = select.select([tty_file], [], [], 0.0)
    if dr:
        return tty_file.read(1)
    return None


def main():
    rclpy.init()
    node = Collector()

    tty_file = open("/dev/tty", "r")
    old_settings = termios.tcgetattr(tty_file)
    tty.setcbreak(tty_file.fileno())

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            key = read_key(tty_file)
            if key == " ":
                node.save_once()
            elif key == "q":
                node.get_logger().info("Exit.")
                break

    finally:
        termios.tcsetattr(tty_file, termios.TCSADRAIN, old_settings)
        tty_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
