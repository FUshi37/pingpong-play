#!/usr/bin/env python3
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from mc_core_interface.msg import MechUnitStateArray


IMAGE_TOPIC = "/camera/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
JOINT_TOPIC = "/joint_states"
CART_TOPIC = "/cart_states"  # Cartesian end-effector poses from robot controller
RIGHT_ARM_MU_IDX = 3  # mu_idx for right arm in /cart_states

# Live preview overlay: detect this AprilTag family so you can see tag count
# before saving. Must match the board you mounted on the right EE.
PREVIEW_TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11
WINDOW_NAME = "eye_to_hand collect (SPACE=save  q=quit)"
# Warn when image vs joint_states timestamps drift more than this (seconds).
MAX_SYNC_DT = 0.15

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
        self.latest_frame = None  # decoded BGR np.ndarray for preview
        self.latest_joint = None
        self.latest_cart = None  # MechUnitStateArray from /cart_states
        self.latest_camera_info = None
        self.intrinsics_saved = False
        self.sample_idx = self.get_existing_count()

        aruco_dict = cv2.aruco.getPredefinedDictionary(PREVIEW_TAG_FAMILY)
        self.aruco_detector = cv2.aruco.ArucoDetector(
            aruco_dict, cv2.aruco.DetectorParameters())
        self.last_save_msg = ""

        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, 10)
        self.create_subscription(JointState, JOINT_TOPIC, self.joint_callback, 10)
        self.create_subscription(MechUnitStateArray, CART_TOPIC, self.cart_callback, 10)

        if not TXT_PATH.exists():
            with open(TXT_PATH, "w") as f:
                f.write("# Eye-to-hand calibration samples. Each line is one JSON record.\n")

        self.get_logger().info(f"Image topic: {IMAGE_TOPIC}")
        self.get_logger().info(f"CameraInfo topic: {CAMERA_INFO_TOPIC}")
        self.get_logger().info(f"Joint topic: {JOINT_TOPIC}")
        self.get_logger().info(f"Cart topic: {CART_TOPIC} (mu_idx={RIGHT_ARM_MU_IDX} for right arm)")
        self.get_logger().info(f"Save dir: {SAVE_DIR}")
        self.get_logger().info(f"Existing samples: {self.sample_idx}")
        self.get_logger().info("Focus the preview window. SPACE = save, q = quit.")

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
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")

    def camera_info_callback(self, msg):
        self.latest_camera_info = msg
        if not self.intrinsics_saved:
            self.save_intrinsics()

    def joint_callback(self, msg):
        self.latest_joint = msg

    def cart_callback(self, msg):
        self.latest_cart = msg

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

        if self.latest_cart is None:
            self.get_logger().warn("No cart_states received yet.")
            return

        if not self.intrinsics_saved:
            self.save_intrinsics()

        # Extract right arm end-effector pose from /cart_states
        right_arm_state = None
        for state in self.latest_cart.states:
            if state.mu_idx == RIGHT_ARM_MU_IDX:
                right_arm_state = state
                break

        if right_arm_state is None:
            self.get_logger().error(
                f"Right arm (mu_idx={RIGHT_ARM_MU_IDX}) not found in /cart_states")
            return

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
        cart_stamp = stamp_to_float(self.latest_cart.header.stamp)
        dt = image_stamp - joint_stamp
        dt_cart = image_stamp - cart_stamp

        grouped = group_joints(names, positions, velocities, efforts)

        # Extract end-effector pose (xyz + quaternion)
        end_pose = right_arm_state.end_pose
        ee_position = [end_pose.position.x, end_pose.position.y, end_pose.position.z]
        ee_orientation = [
            end_pose.orientation.x,
            end_pose.orientation.y,
            end_pose.orientation.z,
            end_pose.orientation.w,
        ]

        record = {
            "index": self.sample_idx,
            "image_file": image_name,
            "image_path": str(image_path),
            "image_topic": IMAGE_TOPIC,
            "camera_info_topic": CAMERA_INFO_TOPIC,
            "joint_topic": JOINT_TOPIC,
            "cart_topic": CART_TOPIC,
            "image_stamp": image_stamp,
            "joint_stamp": joint_stamp,
            "cart_stamp": cart_stamp,
            "time_diff_image_minus_joint": dt,
            "time_diff_image_minus_cart": dt_cart,
            "camera_intrinsics_path": str(INTRINSICS_PATH) if self.intrinsics_saved else None,
            "joint_names": names,
            "joint_positions": positions,
            "joint_velocities": velocities,
            "joint_efforts": efforts,
            "grouped_joints": grouped,
            "right_arm_mu_idx": RIGHT_ARM_MU_IDX,
            "ee_position": ee_position,  # [x, y, z] in meters, base frame
            "ee_orientation": ee_orientation,  # [x, y, z, w] quaternion
            "saved_wall_time": datetime.now().isoformat(),
        }

        with open(TXT_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        left_arm = grouped["LeftArm"]["positions"]
        right_arm = grouped["RightArm"]["positions"]

        self.last_save_msg = f"SAVED {image_name}  dt_joint={dt*1000:.0f}ms dt_cart={dt_cart*1000:.0f}ms"
        if abs(dt) > MAX_SYNC_DT or abs(dt_cart) > MAX_SYNC_DT:
            self.last_save_msg += "  [WARN: out of sync]"
            self.get_logger().warn(
                f"Sample {image_name} sync dt_joint={dt:.3f}s dt_cart={dt_cart:.3f}s "
                f"exceeds {MAX_SYNC_DT}s; robot may have been moving.")

        self.get_logger().info(
            "Saved sample {:06d}, dt_joint={:.3f}s dt_cart={:.3f}s, "
            "EE pos=[{:.3f},{:.3f},{:.3f}], RightArm={}".format(
                self.sample_idx,
                dt, dt_cart,
                ee_position[0], ee_position[1], ee_position[2],
                ["{:.3f}".format(x) for x in right_arm],
            )
        )

    def render_preview(self):
        """Return an annotated BGR frame for display, or None if no image yet."""
        if self.latest_frame is None:
            return None
        vis = self.latest_frame.copy()

        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        n_tags = 0 if ids is None else len(ids)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        # Status banner
        img_ok = self.latest_image is not None
        joint_ok = self.latest_joint is not None
        cart_ok = self.latest_cart is not None
        tag_color = (0, 255, 0) if n_tags >= 4 else (0, 165, 255) if n_tags > 0 else (0, 0, 255)
        lines = [
            (f"saved: {self.sample_idx}   tags: {n_tags}", tag_color),
            (f"image: {'OK' if img_ok else 'WAIT'}   joints: {'OK' if joint_ok else 'WAIT'}   "
             f"cart: {'OK' if cart_ok else 'WAIT'}   intrinsics: {'OK' if self.intrinsics_saved else 'WAIT'}",
             (255, 255, 255)),
            ("SPACE = save    q = quit", (255, 255, 0)),
        ]
        if self.last_save_msg:
            lines.append((self.last_save_msg, (255, 200, 0)))

        y = 24
        for text, color in lines:
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color, 1, cv2.LINE_AA)
            y += 26
        return vis


def main():
    rclpy.init()
    node = Collector()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            vis = node.render_preview()
            if vis is not None:
                cv2.imshow(WINDOW_NAME, vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                node.save_once()
            elif key == ord("q"):
                node.get_logger().info("Exit.")
                break

    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
