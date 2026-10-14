#!/usr/bin/env python3
"""
AprilGrid eye-to-hand calibration (AX=XB).

Target: AprilGrid fixed on left arm EE.
Camera: fixed on robot head (head23).
Solves T_base_camera and T_head23_camera.

Coordinate convention: T_A_B transforms points from frame B to frame A.
  e.g. T_base_camera means: p_base = T_base_camera @ p_camera

Example:
    python3 calibration/eye_hand_calib.py \\
        --intrinsics calibration/realsense_color_camera_info.json \\
        --samples calibration/eye_to_hand_sample/samples.txt \\
        --image-dir calibration/eye_to_hand_sample/images \\
        --urdf moz1.urdf \\
        --tag-family tag36h11 \\
        --tag-rows 6 --tag-cols 6 \\
        --tag-size 0.035 --tag-spacing 0.3 \\
        --ee-frame left07 \\
        --head-frame head23 \\
        --base-frame base_link \\
        --method PARK \\
        --output calibration/eye_hand_result.json \\
        --debug-dir calibration/debug_vis

Diagnose tag detection:
    python3 calibration/eye_hand_calib.py --diagnose-tags \\
        --intrinsics calibration/realsense_color_camera_info.json \\
        --samples calibration/eye_to_hand_sample/samples.txt \\
        --image-dir calibration/eye_to_hand_sample/images \\
        --tag-family tag16h5 \\
        --tag-rows 6 --tag-cols 6 \\
        --tag-size 0.035 --tag-spacing 0.3
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

try:
    from pupil_apriltags import Detector as AprilTagDetector
    USE_PUPIL = True
except ImportError:
    USE_PUPIL = False

# ============================================================
# Intrinsics
# ============================================================

def load_intrinsics(path):
    with open(path) as f:
        data = json.load(f)

    intrinsics = data.get("intrinsics", data)
    if "original_camera_matrix" in intrinsics and "original_dist_coeffs" in intrinsics:
        K = np.array(intrinsics["original_camera_matrix"], dtype=np.float64).reshape(3, 3)
        D = np.array(intrinsics["original_dist_coeffs"], dtype=np.float64).flatten()
    else:
        K = np.array(intrinsics["camera_matrix"], dtype=np.float64).reshape(3, 3)
        D = np.array(intrinsics["dist_coeffs"], dtype=np.float64).flatten()

    meta = {
        "frame_id": intrinsics.get("frame_id", ""),
        "image_width": intrinsics.get("image_width", 0),
        "image_height": intrinsics.get("image_height", 0),
        "distortion_model": intrinsics.get("distortion_model", ""),
        "camera_matrix": K.tolist(),
        "dist_coeffs": D.tolist(),
    }
    return K, D, meta


def default_result_path(image_width, undistorted=False):
    module_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(os.path.dirname(module_dir))
    suffix = "_undistorted" if undistorted else ""
    filename = f"eye_hand_result_{int(image_width)}{suffix}.json"
    return os.path.join(package_dir, "outputs", "vision_calib", filename)


def prepare_undistort(K, D, meta, alpha=0.0):
    width = int(meta.get("image_width") or 0)
    height = int(meta.get("image_height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Intrinsics JSON must include image_width/image_height for undistort mode")

    image_size = (width, height)
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        K, D, image_size, alpha=float(alpha), newImgSize=image_size)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, new_K, image_size, cv2.CV_32FC1)
    meta_out = dict(meta)
    meta_out.update({
        "image_model": "undistorted",
        "undistort_alpha": float(alpha),
        "original_camera_matrix": K.tolist(),
        "original_dist_coeffs": D.tolist(),
        "camera_matrix": new_K.tolist(),
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "roi": list(map(int, roi)),
    })
    return new_K, np.zeros(5, dtype=np.float64), map1, map2, meta_out


def save_intrinsics_json(path, meta):
    output = {
        "intrinsics": {
            "image_width": int(meta.get("image_width", 0)),
            "image_height": int(meta.get("image_height", 0)),
            "frame_id": meta.get("frame_id", ""),
            "distortion_model": "none",
            "camera_matrix": meta["camera_matrix"],
            "dist_coeffs": meta["dist_coeffs"],
            "image_model": meta.get("image_model", "raw"),
            "undistort_alpha": meta.get("undistort_alpha"),
            "original_camera_matrix": meta.get("original_camera_matrix"),
            "original_dist_coeffs": meta.get("original_dist_coeffs"),
            "roi": meta.get("roi"),
        },
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ============================================================
# AprilGrid geometry
# ============================================================

def build_aprilgrid_object_points(tag_rows, tag_cols, tag_size, tag_spacing):
    """tag_id = row * tag_cols + col, row-major."""
    stride = tag_size * (1.0 + tag_spacing)
    obj_pts = {}
    for row in range(tag_rows):
        for col in range(tag_cols):
            tag_id = row * tag_cols + col
            x0 = col * stride
            y0 = row * stride
            obj_pts[tag_id] = np.array([
                [x0, y0, 0],
                [x0 + tag_size, y0, 0],
                [x0 + tag_size, y0 + tag_size, 0],
                [x0, y0 + tag_size, 0],
            ], dtype=np.float64)
    return obj_pts


# ============================================================
# AprilTag detection (OpenCV aruco primary, pupil_apriltags fallback)
# ============================================================

_OPENCV_APRILTAG_DICTS = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def _make_opencv_detector(tag_family):
    if tag_family not in _OPENCV_APRILTAG_DICTS:
        raise ValueError(f"Unsupported tag family: {tag_family}")
    d = cv2.aruco.getPredefinedDictionary(_OPENCV_APRILTAG_DICTS[tag_family])
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 8
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, params)


def detect_apriltags_opencv(image, tag_family, invert=False):
    """Returns list of (tag_id, corners_2d_4x2, decision_margin, hamming)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if invert:
        gray = 255 - gray
    detector = _make_opencv_detector(tag_family)
    corners, ids, _ = detector.detectMarkers(gray)
    detections = []
    if ids is not None:
        for i, tag_id in enumerate(ids.flatten()):
            c = corners[i][0].astype(np.float64)  # shape (4,2)
            detections.append((int(tag_id), c, 100.0, 0))
    return detections


def detect_apriltags_pupil(image, tag_family, K, invert=False):
    """Returns list of (tag_id, corners_2d_4x2, decision_margin, hamming)."""
    if not USE_PUPIL:
        raise RuntimeError("pupil_apriltags not available")
    detector = AprilTagDetector(families=tag_family, nthreads=1, quad_decimate=1.0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    if invert:
        gray = 255 - gray
    results = detector.detect(
        gray, estimate_tag_pose=False,
        camera_params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        tag_size=1.0,
    )
    return [(r.tag_id, r.corners.astype(np.float64), r.decision_margin, r.hamming)
            for r in results]


def deduplicate_detections(detections, max_tags):
    """Keep best detection per tag_id. Skip ids >= max_tags."""
    best = {}
    duplicates = []
    for tag_id, corners, margin, hamming in detections:
        if tag_id >= max_tags:
            duplicates.append((tag_id, "id out of grid range"))
            continue
        if tag_id not in best:
            best[tag_id] = (tag_id, corners, margin, hamming)
        else:
            old_margin = best[tag_id][2]
            old_hamming = best[tag_id][3]
            if (hamming < old_hamming) or (hamming == old_hamming and margin > old_margin):
                duplicates.append((tag_id, f"replaced: margin {old_margin:.1f}->{margin:.1f}"))
                best[tag_id] = (tag_id, corners, margin, hamming)
            else:
                duplicates.append((tag_id, f"discarded: margin {margin:.1f} < {old_margin:.1f}"))
    return list(best.values()), duplicates


def detect_best(image, tag_family, K, max_tags, backend="opencv"):
    """Detect on normal + inverted, merge, deduplicate."""
    if backend == "opencv":
        normal = detect_apriltags_opencv(image, tag_family, invert=False)
        inverted = detect_apriltags_opencv(image, tag_family, invert=True)
    else:
        normal = detect_apriltags_pupil(image, tag_family, K, invert=False)
        inverted = detect_apriltags_pupil(image, tag_family, K, invert=True)
    combined = normal + inverted
    good, dups = deduplicate_detections(combined, max_tags)
    return good, dups, len(normal), len(inverted)


# ============================================================
# PnP
# ============================================================

CORNER_ORDERS = {
    "0123": [0, 1, 2, 3],
    "1230": [1, 2, 3, 0],
    "2301": [2, 3, 0, 1],
    "3012": [3, 0, 1, 2],
    "0321": [0, 3, 2, 1],
    "3210": [3, 2, 1, 0],
    "2103": [2, 1, 0, 3],
    "1032": [1, 0, 3, 2],
}


def solve_pnp_aprilgrid(detections, grid_obj_pts, K, D, corner_order):
    """Returns T_camera_grid (4x4): transforms grid points into camera frame."""
    pts_3d, pts_2d = [], []
    corner_order = CORNER_ORDERS[corner_order]
    for tag_id, corners_2d, _, _ in detections:
        if tag_id in grid_obj_pts:
            pts_3d.append(grid_obj_pts[tag_id])
            pts_2d.append(corners_2d[corner_order])
    if len(pts_3d) < 2:
        return None, float("inf")
    pts_3d = np.vstack(pts_3d)
    pts_2d = np.vstack(pts_2d)
    ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, float("inf")
    rvec, tvec = cv2.solvePnPRefineLM(pts_3d, pts_2d, K, D, rvec, tvec)
    proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, D)
    err = np.sqrt(np.mean((pts_2d - proj.reshape(-1, 2)) ** 2))
    R, _ = cv2.Rodrigues(rvec)
    T_camera_grid = np.eye(4)
    T_camera_grid[:3, :3] = R
    T_camera_grid[:3, 3] = tvec.flatten()
    return T_camera_grid, err


# ============================================================
# URDF FK (no external dependency beyond stdlib + numpy)
# ============================================================

def _parse_origin(elem):
    """Parse <origin xyz="..." rpy="..."/> -> 4x4."""
    T = np.eye(4)
    if elem is None:
        return T
    origin = elem.find("origin")
    if origin is None:
        return T
    xyz = origin.get("xyz", "0 0 0").split()
    rpy = origin.get("rpy", "0 0 0").split()
    tx, ty, tz = float(xyz[0]), float(xyz[1]), float(xyz[2])
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def _axis_vec(joint_elem):
    ax = joint_elem.find("axis")
    if ax is None:
        return np.array([1.0, 0.0, 0.0])
    return np.array([float(x) for x in ax.get("xyz", "1 0 0").split()])


def _rot_matrix(axis, angle):
    """Rodrigues rotation."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    return T


class URDFForwardKinematics:
    """Minimal tree-based FK from URDF XML."""

    def __init__(self, urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        self.links = set()
        self.joints = []
        self.joint_by_name = {}
        self.children = {}  # parent_link -> [(joint_name, child_link)]
        self.parent_map = {}  # child_link -> (joint_name, parent_link)

        for link in root.findall("link"):
            self.links.add(link.get("name"))

        for joint in root.findall("joint"):
            jname = joint.get("name")
            jtype = joint.get("type")
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            origin_T = _parse_origin(joint)
            axis = _axis_vec(joint)
            info = {
                "name": jname, "type": jtype,
                "parent": parent, "child": child,
                "origin": origin_T, "axis": axis,
            }
            self.joints.append(info)
            self.joint_by_name[jname] = info
            self.children.setdefault(parent, []).append((jname, child))
            self.parent_map[child] = (jname, parent)

    def compute_T_base_link(self, target_link, base_link, joint_values):
        """Compute T_base_target given joint_values dict {joint_name: angle}."""
        path = self._find_path(base_link, target_link)
        if path is None:
            raise ValueError(f"No path from {base_link} to {target_link}")
        T = np.eye(4)
        for jname, direction in path:
            jinfo = self.joint_by_name[jname]
            T_joint = jinfo["origin"].copy()
            if jinfo["type"] in ("revolute", "continuous"):
                angle = joint_values.get(jname, 0.0)
                T_joint = T_joint @ _rot_matrix(jinfo["axis"], angle)
            if direction == 1:
                T = T @ T_joint
            else:
                T = T @ np.linalg.inv(T_joint)
        return T

    def _find_path(self, from_link, to_link):
        """BFS to find joint path from from_link to to_link."""
        from collections import deque
        visited = {from_link}
        queue = deque([(from_link, [])])
        while queue:
            current, path = queue.popleft()
            if current == to_link:
                return path
            # Forward edges (parent -> child)
            for jname, child in self.children.get(current, []):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [(jname, 1)]))
            # Backward edge (child -> parent)
            if current in self.parent_map:
                jname, parent = self.parent_map[current]
                if parent not in visited:
                    visited.add(parent)
                    queue.append((parent, path + [(jname, -1)]))
        return None


# ============================================================
# Debug visualization
# ============================================================

def draw_debug(image, detections, K, D, rvec, tvec, out_path):
    vis = image.copy()
    for tag_id, corners, _m, _h in detections:
        pts = corners.astype(int)
        for i in range(4):
            cv2.line(vis, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.putText(vis, str(tag_id), tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    if rvec is not None:
        cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.05)
    cv2.imwrite(str(out_path), vis)


def _rotation_angle_deg(R):
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def compute_handeye_residuals(used_images, T_base_ee_list, T_camera_grid_list,
                              T_base_camera, T_ee_grid):
    """Measure consistency of T_camera_grid = inv(T_base_camera)*T_base_ee*T_ee_grid."""
    T_camera_base = np.linalg.inv(T_base_camera)
    residuals = {}
    trans = []
    rot = []
    for image_file, T_base_ee, T_camera_grid in zip(
            used_images, T_base_ee_list, T_camera_grid_list):
        predicted = T_camera_base @ T_base_ee @ T_ee_grid
        delta = np.linalg.inv(T_camera_grid) @ predicted
        trans_m = float(np.linalg.norm(delta[:3, 3]))
        rot_deg = _rotation_angle_deg(delta[:3, :3])
        residuals[image_file] = {
            "translation_m": trans_m,
            "translation_mm": trans_m * 1000.0,
            "rotation_deg": rot_deg,
        }
        trans.append(trans_m)
        rot.append(rot_deg)

    return residuals, {
        "mean_translation_m": float(np.mean(trans)),
        "mean_translation_mm": float(np.mean(trans) * 1000.0),
        "max_translation_m": float(np.max(trans)),
        "max_translation_mm": float(np.max(trans) * 1000.0),
        "mean_rotation_deg": float(np.mean(rot)),
        "max_rotation_deg": float(np.max(rot)),
    }


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="AprilGrid eye-to-hand AX=XB calibration")
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--urdf", default=None, help="URDF file (required unless --diagnose-tags)")
    p.add_argument("--tag-family", default="tag36h11",
                   choices=["tag16h5", "tag25h9", "tag36h11"])
    p.add_argument("--tag-rows", type=int, required=True)
    p.add_argument("--tag-cols", type=int, required=True)
    p.add_argument("--tag-size", type=float, required=True, help="meters")
    p.add_argument("--tag-spacing", type=float, required=True, help="ratio gap/tag_size")
    p.add_argument("--ee-frame", default="left07")
    p.add_argument("--head-frame", default="head23")
    p.add_argument("--base-frame", default="base_link")
    p.add_argument(
        "--output",
        default=None,
        help="Output calibration result JSON. Defaults to outputs/vision_calib/eye_hand_result_<width>[_undistorted].json",
    )
    p.add_argument("--debug-dir", default=None)
    p.add_argument("--undistort", action="store_true",
                   help="Undistort images before tag detection/PnP and use new_K with zero distortion")
    p.add_argument("--undistort-alpha", type=float, default=0.0,
                   help="Alpha passed to cv2.getOptimalNewCameraMatrix; keep 0.0 to match localization.py")
    p.add_argument("--write-undistorted-intrinsics", default=None,
                   help="Write the undistorted new_K/D=0 intrinsics JSON and continue")
    p.add_argument("--min-tags", type=int, default=4,
                   help="Minimum detected AprilTags required for one frame")
    p.add_argument("--corner-order", default="0123", choices=sorted(CORNER_ORDERS),
                   help="Order used to match detected 2D tag corners to generated 3D tag corners")
    p.add_argument("--max-reproj-error", type=float, default=10.0,
                   help="Skip frames whose AprilGrid PnP reprojection error exceeds this many pixels")
    p.add_argument("--exclude-images", default="",
                   help="Comma-separated image file names to exclude, e.g. 000018.jpg,000020.jpg")
    p.add_argument("--diagnose-tags", action="store_true",
                   help="Only detect tags in all images and report counts/ids")
    p.add_argument("--backend", default="opencv", choices=["opencv", "pupil"],
                   help="AprilTag detection backend")
    p.add_argument("--method", default="PARK",
                   choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"])
    return p.parse_args()


def run_diagnose(args, K):
    """Detect tags in all images and print diagnostic info."""
    with open(args.samples) as f:
        samples = [json.loads(line) for line in f if line.strip() and not line.startswith("#")]

    max_tags = args.tag_rows * args.tag_cols
    print(f"Grid: {args.tag_rows}x{args.tag_cols} = {max_tags} tags, "
          f"family={args.tag_family}, size={args.tag_size}m, spacing={args.tag_spacing}")
    print(f"{'Image':<20} {'Normal':>7} {'Invert':>7} {'Merged':>7} {'IDs'}")
    print("-" * 80)

    for sample in samples:
        img_file = sample["image_file"]
        img_path = os.path.join(args.image_dir, img_file)
        if not os.path.isfile(img_path):
            print(f"{img_file:<20} FILE NOT FOUND")
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"{img_file:<20} READ FAILED")
            continue
        if getattr(args, "undistort", False):
            image = cv2.remap(image, args.undistort_map1, args.undistort_map2,
                              interpolation=cv2.INTER_LINEAR)
        good, dups, n_normal, n_inverted = detect_best(
            image, args.tag_family, K, max_tags, backend=args.backend)
        ids = sorted([d[0] for d in good])
        print(f"{img_file:<20} {n_normal:>7} {n_inverted:>7} {len(good):>7}  {ids}")
        if dups:
            for tid, reason in dups:
                print(f"  dup/skip tag {tid}: {reason}")


def main():
    args = parse_args()
    K_raw, D_raw, intrinsics_raw_meta = load_intrinsics(args.intrinsics)
    if args.output is None:
        args.output = default_result_path(
            intrinsics_raw_meta.get("image_width", 0),
            undistorted=bool(args.undistort or args.write_undistorted_intrinsics),
        )

    if args.undistort or args.write_undistorted_intrinsics:
        K, D, map1, map2, intrinsics_meta = prepare_undistort(
            K_raw, D_raw, intrinsics_raw_meta, alpha=args.undistort_alpha)
        args.undistort_map1 = map1
        args.undistort_map2 = map2
        if args.write_undistorted_intrinsics:
            save_intrinsics_json(args.write_undistorted_intrinsics, intrinsics_meta)
            print(f"Wrote undistorted intrinsics: {args.write_undistorted_intrinsics}")
    else:
        K, D, intrinsics_meta = K_raw, D_raw, dict(intrinsics_raw_meta)
        intrinsics_meta["image_model"] = "raw"
        args.undistort_map1 = None
        args.undistort_map2 = None
    grid_obj_pts = build_aprilgrid_object_points(
        args.tag_rows, args.tag_cols, args.tag_size, args.tag_spacing)
    max_tags = args.tag_rows * args.tag_cols

    if args.diagnose_tags:
        run_diagnose(args, K)
        return

    if not args.urdf:
        print("ERROR: --urdf is required for calibration.", file=sys.stderr)
        sys.exit(1)

    fk = URDFForwardKinematics(args.urdf)
    excluded_images = {x.strip() for x in args.exclude_images.split(",") if x.strip()}
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    with open(args.samples) as f:
        samples = [json.loads(line) for line in f if line.strip() and not line.startswith("#")]

    # Collect per-frame data.
    # For OpenCV calibrateHandEye in eye-to-hand mode:
    #   input 1 is ^gT_b, i.e. T_ee_base = inv(T_base_ee)
    #   input 2 is ^cT_t, i.e. T_camera_grid from PnP
    #   output is ^bT_c, i.e. T_base_camera
    R_ee_base_list, t_ee_base_list = [], []
    R_grid_camera_list, t_grid_camera_list = [], []
    T_base_head_list = []
    T_base_ee_list = []
    T_camera_grid_list = []
    used_images, skipped_images = [], []
    reproj_errors = {}
    tag_counts = {}

    for sample in samples:
        img_file = sample["image_file"]
        if img_file in excluded_images:
            skipped_images.append({"image": img_file, "reason": "excluded by --exclude-images"})
            continue
        img_path = os.path.join(args.image_dir, img_file)
        if not os.path.isfile(img_path):
            skipped_images.append({"image": img_file, "reason": "file not found"})
            continue
        image = cv2.imread(img_path)
        if image is None:
            skipped_images.append({"image": img_file, "reason": "failed to read"})
            continue
        if args.undistort:
            image = cv2.remap(image, args.undistort_map1, args.undistort_map2,
                              interpolation=cv2.INTER_LINEAR)

        good, _dups, _, _ = detect_best(image, args.tag_family, K, max_tags)
        if len(good) < args.min_tags:
            skipped_images.append({"image": img_file,
                                   "reason": f"only {len(good)} tags after dedup"})
            continue
        tag_counts[img_file] = len(good)

        T_camera_grid, err = solve_pnp_aprilgrid(
            good, grid_obj_pts, K, D, args.corner_order)
        if T_camera_grid is None:
            skipped_images.append({"image": img_file, "reason": "solvePnP failed"})
            continue
        if err > args.max_reproj_error:
            skipped_images.append({"image": img_file,
                                   "reason": f"reprojection error {err:.3f}px > {args.max_reproj_error:.3f}px"})
            continue

        # FK: build joint_values dict from sample
        joint_names = sample["joint_names"]
        joint_positions = sample["joint_positions"]
        joint_values = dict(zip(joint_names, joint_positions))

        T_base_ee = fk.compute_T_base_link(args.ee_frame, args.base_frame, joint_values)
        T_base_head = fk.compute_T_base_link(args.head_frame, args.base_frame, joint_values)

        T_ee_base = np.linalg.inv(T_base_ee)
        R_ee_base_list.append(T_ee_base[:3, :3])
        t_ee_base_list.append(T_ee_base[:3, 3].reshape(3, 1))
        R_grid_camera_list.append(T_camera_grid[:3, :3])
        t_grid_camera_list.append(T_camera_grid[:3, 3].reshape(3, 1))
        T_base_ee_list.append(T_base_ee)
        T_camera_grid_list.append(T_camera_grid)
        T_base_head_list.append(T_base_head)
        used_images.append(img_file)
        reproj_errors[img_file] = float(err)

        if args.debug_dir:
            rvec, _ = cv2.Rodrigues(T_camera_grid[:3, :3])
            tvec = T_camera_grid[:3, 3]
            draw_debug(image, good, K, D, rvec, tvec,
                       os.path.join(args.debug_dir, img_file))

    if len(used_images) < 3:
        print(f"ERROR: Only {len(used_images)} valid samples, need >= 3.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(used_images)} valid frames...")

    method_map = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    # --- calibrateHandEye, eye-to-hand convention ---
    # OpenCV's eye-to-hand equation returns ^bT_c when the first input is
    # ^gT_b and the second input is ^cT_t:
    #   ^gT_b(i) * ^bT_c * ^cT_t(i) = constant
    R_base_cam, t_base_cam = cv2.calibrateHandEye(
        R_ee_base_list, t_ee_base_list,
        R_grid_camera_list, t_grid_camera_list,
        method=method_map[args.method],
    )

    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = R_base_cam
    T_base_camera[:3, 3] = t_base_cam.flatten()
    T_camera_base = np.linalg.inv(T_base_camera)

    # Estimate the fixed board mount transform for diagnostics:
    # T_camera_grid = T_camera_base * T_base_ee * T_ee_grid
    T_ee_grid_list = []
    for T_base_ee, T_camera_grid in zip(T_base_ee_list, T_camera_grid_list):
        T_ee_grid_list.append(np.linalg.inv(T_base_ee) @ T_base_camera @ T_camera_grid)
    T_ee_grid_mat = np.mean(T_ee_grid_list, axis=0)
    U, _, Vt = np.linalg.svd(T_ee_grid_mat[:3, :3])
    T_ee_grid_mat[:3, :3] = U @ Vt
    T_ee_grid_mat[3, :] = [0, 0, 0, 1]
    T_grid_ee_mat = np.linalg.inv(T_ee_grid_mat)
    handeye_residuals, handeye_summary = compute_handeye_residuals(
        used_images, T_base_ee_list, T_camera_grid_list,
        T_base_camera, T_ee_grid_mat)

    # --- Head23 handling ---
    # Check if head joints vary across frames
    head_translations = np.array([T[:3, 3] for T in T_base_head_list])
    head_spread = np.max(np.ptp(head_translations, axis=0))
    if head_spread > 0.001:
        print(f"WARNING: head23 pose varies across frames (spread={head_spread:.4f}m). "
              f"Using mean.")
        T_base_head_avg = np.mean(T_base_head_list, axis=0)
        # Re-orthogonalize rotation
        U, _, Vt = np.linalg.svd(T_base_head_avg[:3, :3])
        T_base_head_avg[:3, :3] = U @ Vt
        head_note = f"averaged over {len(T_base_head_list)} frames, spread={head_spread:.5f}m"
    else:
        T_base_head_avg = T_base_head_list[0]
        head_note = "constant across all frames (head joints fixed or absent)"

    # T_head23_camera = T_head23_base @ T_base_camera = inv(T_base_head23) @ T_base_camera
    T_head_base = np.linalg.inv(T_base_head_avg)
    T_head23_camera = T_head_base @ T_base_camera
    T_camera_head23 = np.linalg.inv(T_head23_camera)

    # --- Output ---
    result = {
        "convention": "T_A_B transforms points from frame B to frame A: p_A = T_A_B @ p_B",
        "intrinsics": intrinsics_meta,
        "T_base_camera": T_base_camera.tolist(),
        "T_camera_base": T_camera_base.tolist(),
        "T_head23_camera": T_head23_camera.tolist(),
        "T_camera_head23": T_camera_head23.tolist(),
        "T_ee_grid": T_ee_grid_mat.tolist(),
        "T_grid_ee": T_grid_ee_mat.tolist(),
        "head23_note": head_note,
        "method": f"calibrateHandEye eye-to-hand {args.method}",
        "corner_order": args.corner_order,
        "num_valid_samples": len(used_images),
        "mean_reprojection_error_px": float(np.mean(list(reproj_errors.values()))),
        "reprojection_errors": reproj_errors,
        "tag_counts": tag_counts,
        "handeye_residual_summary": handeye_summary,
        "handeye_residuals": handeye_residuals,
        "used_images": used_images,
        "skipped_images": skipped_images,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done. {len(used_images)} used, {len(skipped_images)} skipped.")
    print(f"Mean reprojection error: {result['mean_reprojection_error_px']:.4f} px")
    print("Hand-eye residual: "
          f"{handeye_summary['mean_translation_mm']:.2f} mm mean, "
          f"{handeye_summary['max_translation_mm']:.2f} mm max, "
          f"{handeye_summary['mean_rotation_deg']:.3f} deg mean")
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
