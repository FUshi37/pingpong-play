#!/home/jacky/miniconda3/envs/gmr/bin/python
"""
Calibration Loader - Load camera intrinsics and extrinsics from JSON file.

This module loads calibration data from eye-hand calibration results,
including camera intrinsics (original and undistorted) and extrinsics
(T_base_camera transformation).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


class CalibrationData:
    """Container for calibration data loaded from JSON."""

    def __init__(self, json_data: Dict[str, Any]):
        """
        Initialize calibration data from JSON dictionary.

        Args:
            json_data: Dictionary loaded from calibration JSON file
        """
        self.convention = json_data.get('convention', '')
        intrinsics = json_data['intrinsics']

        # Undistorted camera intrinsics (for 3D calculations)
        self.camera_matrix = np.array(
            intrinsics['camera_matrix'], dtype=np.float64)
        self.dist_coeffs = np.array(
            intrinsics['dist_coeffs'], dtype=np.float64)
        self.image_model = intrinsics.get('image_model', 'undistorted')

        # Original camera intrinsics (for undistorting raw images)
        self.original_camera_matrix = np.array(
            intrinsics['original_camera_matrix'], dtype=np.float64)
        self.original_dist_coeffs = np.array(
            intrinsics['original_dist_coeffs'], dtype=np.float64)

        # Image dimensions
        self.image_width = intrinsics['image_width']
        self.image_height = intrinsics['image_height']
        self.frame_id = intrinsics.get(
            'frame_id', 'camera_color_optical_frame')

        # Extrinsics: T_base_camera transforms from camera to base_link
        self.T_base_camera = np.array(
            json_data['T_base_camera'], dtype=np.float64)

        # Validate dimensions
        self._validate()

    def _validate(self):
        """Validate loaded calibration data dimensions."""
        assert self.camera_matrix.shape == (
            3, 3), f"camera_matrix shape error: {self.camera_matrix.shape}"
        assert self.original_camera_matrix.shape == (
            3, 3), f"original_camera_matrix shape error: {self.original_camera_matrix.shape}"
        assert self.T_base_camera.shape == (
            4, 4), f"T_base_camera shape error: {self.T_base_camera.shape}"
        assert len(
            self.dist_coeffs) >= 5, f"dist_coeffs length error: {len(self.dist_coeffs)}"
        assert len(
            self.original_dist_coeffs) >= 5, f"original_dist_coeffs length error: {len(self.original_dist_coeffs)}"

    def get_undistorted_intrinsics(self):
        """Get undistorted camera intrinsics (fx, fy, cx, cy)."""
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        return fx, fy, cx, cy

    def get_original_intrinsics(self):
        """Get original camera intrinsics (fx, fy, cx, cy)."""
        fx = self.original_camera_matrix[0, 0]
        fy = self.original_camera_matrix[1, 1]
        cx = self.original_camera_matrix[0, 2]
        cy = self.original_camera_matrix[1, 2]
        return fx, fy, cx, cy

    def transform_camera_to_base(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform point from camera frame to base_link frame.

        Args:
            point_camera: 3D point in camera frame [x, y, z] (mm)

        Returns:
            3D point in base_link frame [x, y, z] (mm)

        Note:
            T_base_camera has translation in meters, so we convert:
            mm -> m -> apply T -> m -> mm
        """
        # Convert from mm to m
        point_camera_m = np.array(point_camera) / 1000.0
        # Convert to homogeneous coordinates
        point_homo = np.append(point_camera_m, 1.0)
        # Apply transformation: p_base = T_base_camera @ p_camera
        point_base_homo = self.T_base_camera @ point_homo
        # Convert back from m to mm and return 3D coordinates
        return point_base_homo[:3] * 1000.0

    def __repr__(self):
        fx, fy, cx, cy = self.get_undistorted_intrinsics()
        fx_orig, fy_orig, cx_orig, cy_orig = self.get_original_intrinsics()
        return (
            f"CalibrationData(\n"
            f"  image_size={self.image_width}x{self.image_height}\n"
            f"  undistorted: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}\n"
            f"  original: fx={fx_orig:.2f}, fy={fy_orig:.2f}, cx={cx_orig:.2f}, cy={cy_orig:.2f}\n"
            f"  T_base_camera: {self.T_base_camera.shape}\n"
            f")"
        )


def load_calibration(json_path: str) -> CalibrationData:
    """
    Load calibration data from JSON file.

    Args:
        json_path: Path to calibration JSON file

    Returns:
        CalibrationData object

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON format is invalid
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {json_path}")

    with open(path, 'r') as f:
        json_data = json.load(f)

    return CalibrationData(json_data)
