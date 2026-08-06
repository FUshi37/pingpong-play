"""Sub-pixel IR ball-center refinement on the original grayscale image."""

from typing import Tuple

import cv2
import numpy as np


def refine_ir_ball_center(
        ir_image: np.ndarray,
        coarse_center: Tuple[float, float],
        radius_px: float,
        blur_kernel_size: int = 5,
        background_fraction: float = 0.05,
        window_scale: float = 1.6,
        support_scale: float = 1.35,
        max_shift_fraction: float = 0.3,
) -> Tuple[float, float, float]:
    """Refine a morphology-derived center using local IR reflection energy.

    The RealSense projector produces sparse dots on the ball. Morphology is
    useful for associating those dots, but its expanded contour is not a
    physical ball boundary. This function goes back to the unmodified IR
    pixels, subtracts a local background, and computes a weighted centroid of
    the complete reflection-energy envelope.

    Returns ``(cx, cy, shift_px)``. Invalid/implausible refinements fall back
    to the supplied coarse center and report a zero shift.
    """
    coarse_cx, coarse_cy = map(float, coarse_center)
    radius = float(radius_px)
    if (ir_image is None or ir_image.ndim != 2 or not np.isfinite(radius)
            or radius <= 1.0):
        return coarse_cx, coarse_cy, 0.0

    kernel = max(3, int(blur_kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    half = max(12, int(round(radius * float(window_scale))))
    height, width = ir_image.shape[:2]
    x1 = max(0, int(np.floor(coarse_cx)) - half)
    x2 = min(width, int(np.floor(coarse_cx)) + half + 1)
    y1 = max(0, int(np.floor(coarse_cy)) - half)
    y2 = min(height, int(np.floor(coarse_cy)) + half + 1)
    if x2 - x1 < kernel or y2 - y1 < kernel:
        return coarse_cx, coarse_cy, 0.0

    patch = ir_image[y1:y2, x1:x2].astype(np.float32, copy=False)
    smooth = cv2.GaussianBlur(patch, (kernel, kernel), 0)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    distance = np.hypot(xx - coarse_cx, yy - coarse_cy)

    outer = smooth[distance > radius * 1.25]
    background = (float(np.median(outer)) if outer.size
                  else float(np.median(smooth)))
    inner = smooth[distance < radius]
    if not inner.size:
        return coarse_cx, coarse_cy, 0.0
    peak = float(np.percentile(inner, 99.0))
    if peak <= background + 1.0:
        return coarse_cx, coarse_cy, 0.0

    level = background + float(background_fraction) * (peak - background)
    weights = np.maximum(smooth - level, 0.0)
    weights[distance > radius * float(support_scale)] = 0.0
    total = float(np.sum(weights))
    if total <= 1e-6:
        return coarse_cx, coarse_cy, 0.0

    refined_cx = float(np.sum(xx * weights) / total)
    refined_cy = float(np.sum(yy * weights) / total)
    shift = float(np.hypot(refined_cx - coarse_cx,
                           refined_cy - coarse_cy))
    max_shift = max(1.0, radius * float(max_shift_fraction))
    if not np.isfinite(shift) or shift > max_shift:
        return coarse_cx, coarse_cy, 0.0
    return refined_cx, refined_cy, shift
