import cv2
import numpy as np

from pingpong_controller.ir_center_estimator import refine_ir_ball_center


def test_refines_sparse_asymmetric_dots_toward_energy_center():
    image = np.full((80, 100), 15, dtype=np.uint8)
    true_center = (53.25, 38.75)
    for dx, dy, value in (
            (-8, -5, 150), (0, -7, 180), (7, -4, 160),
            (-9, 3, 155), (0, 1, 240), (8, 4, 170), (1, 9, 165)):
        point = (int(round(true_center[0] + dx)),
                 int(round(true_center[1] + dy)))
        cv2.circle(image, point, 2, value, -1)

    coarse = (51.0, 37.0)
    cx, cy, shift = refine_ir_ball_center(image, coarse, 14.0)

    assert shift > 0.0
    assert np.hypot(cx - true_center[0], cy - true_center[1]) < np.hypot(
        coarse[0] - true_center[0], coarse[1] - true_center[1])


def test_falls_back_when_patch_has_no_contrast():
    image = np.full((40, 50), 20, dtype=np.uint8)
    assert refine_ir_ball_center(image, (25.0, 20.0), 10.0) == (
        25.0, 20.0, 0.0)
