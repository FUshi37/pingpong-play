import numpy as np
import pyrealsense2 as rs

from pingpong_controller.hybrid_ball_detector import (
    _rs_extrinsics_rotation_matrix,
)


def test_numpy_extrinsics_rotation_matches_librealsense_transform():
    extrinsics = rs.extrinsics()
    extrinsics.rotation = [
        0.9999992847442627,
        -0.0005568618071265519,
        -0.0010406061774119735,
        0.0005548078333958983,
        0.9999979138374329,
        -0.0019730632193386555,
        0.0010417026933282614,
        0.001972484402358532,
        0.9999974966049194,
    ]
    extrinsics.translation = [
        -0.05906657874584198,
        0.00007409740646835417,
        0.00027349311858415604,
    ]
    point = np.asarray([0.18, 0.11, 0.55], dtype=np.float64)

    expected = np.asarray(
        rs.rs2_transform_point_to_point(extrinsics, point.tolist()),
        dtype=np.float64,
    )
    actual = (
        _rs_extrinsics_rotation_matrix(extrinsics.rotation) @ point
        + np.asarray(extrinsics.translation, dtype=np.float64)
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-7)
