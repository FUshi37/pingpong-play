"""Shared camera calibration constants for sim-to-real juggling."""

from __future__ import annotations

import math


D455_848_UNDISTORTED_WIDTH = 848
D455_848_UNDISTORTED_HEIGHT = 480
D455_848_UNDISTORTED_FX = 415.38882001973656
D455_848_UNDISTORTED_FY = 415.3413378301133
D455_848_UNDISTORTED_CX = 427.24127188258564
D455_848_UNDISTORTED_CY = 241.90963311505976

# Raw RGB intrinsics before undistortion.  Keep these here as the calibration
# source of truth for real-image preprocessing; the simulator/projection code
# uses the undistorted intrinsics above for 3D camera geometry.
D455_848_RAW_FX = 416.928986
D455_848_RAW_FY = 416.492157
D455_848_RAW_CX = 426.326813
D455_848_RAW_CY = 242.124054
D455_848_RAW_DIST_COEFFS = (-0.0582, 0.0723, -0.0002, 0.0007, -0.0234)

# Mount orientation and calibration quality from the same 848x480@60Hz
# hand-eye calibration.  These are documented here so every sim/real consumer
# shares one source of truth; projection uses the rotation matrix below.
D455_848_CAMERA_PITCH_DEG = 0.65
D455_848_CAMERA_ROLL_DEG = -129.67
D455_848_CAMERA_YAW_DEG = -179.99
D455_848_OPTICAL_AXIS_TILT_FROM_VERTICAL_DEG = 50.33
D455_848_CALIB_VALID_SAMPLES = 70
D455_848_CALIB_MEAN_REPROJECTION_ERROR_PX = 0.590
D455_848_HAND_EYE_TRANSLATION_RESIDUAL_M = 0.004135
D455_848_HAND_EYE_ROTATION_RESIDUAL_DEG = 0.553

# T_base_camera maps camera_color_optical_frame points into base_link:
# p_base = T_base_camera @ p_camera.
#
# In the ROS/vision calibration JSON this base_link comes from cart_states /
# arm kinematics.  In the current MJCF, the mobile-platform body is named
# "base"; attaching this 848x480 calibration there puts the juggling ball
# behind the optical plane (negative camera z).  A physical reset check on
# 2026-07-13 matches the calibrated image projection when this transform is
# attached to the upper-body "waist03" frame.
D455_848_UNDISTORTED_SIM_BASE_BODY = "waist03"
D455_848_UNDISTORTED_BASE_POS = (
    -0.010088417179937553,
    -0.09671012017997065,
    0.21900574574293108,
)
D455_848_UNDISTORTED_BASE_ROT = (
    -0.9999361933926297,
    0.008558456597358433,
    0.007373056634102704,
    -0.00021104797684882298,
    0.6384220444788073,
    -0.7696864612179778,
    -0.011294450062310168,
    -0.7696389062048334,
    -0.6383795026891282,
)

D455_848_UNDISTORTED_HFOV_DEG = math.degrees(
    2.0 * math.atan(D455_848_UNDISTORTED_WIDTH / (2.0 * D455_848_UNDISTORTED_FX))
)
D455_848_UNDISTORTED_VFOV_DEG = math.degrees(
    2.0 * math.atan(D455_848_UNDISTORTED_HEIGHT / (2.0 * D455_848_UNDISTORTED_FY))
)

# Same relative horizontal margin as the old 80 px margin at 1280 px width.
D455_848_UNDISTORTED_PIXEL_MARGIN = 53.0
