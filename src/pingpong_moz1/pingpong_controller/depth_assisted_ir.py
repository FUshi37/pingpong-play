"""Recover IR candidates from compact, coherent same-frame stereo surfaces.

No reference positions, labels, future images, or predicted depth are inputs.
Depth topology proposes a region; original IR reflection energy localizes it.
Sparse stereo fragments and merged ball/racket regions are not measurements.
"""

import math
import cv2
import numpy as np
from pingpong_controller.ir_center_estimator import refine_ir_ball_center


def range_quality(candidate, depth, *, depth_scale, selected_depth_mm):
    """Score original stereo in a bounded ball patch, including primary IR hits.

    Zeros and background count against coverage. Small spread alone cannot
    validate sparse, biased stereo fragments. No trajectory or labels enter
    this spatial test, so a dense fresh surface can also recover after impact.
    """
    result = dict(range_valid=False, range_quality_strong=False,
                  range_support_fraction=0.0, range_support_pixels=0,
                  range_spread_mm=None, range_sigma_mm=None,
                  range_quality_reason='depth_unavailable')
    if depth is None or selected_depth_mm is None:
        return result
    cx, cy, radius = (float(candidate[k]) for k in ('cx', 'cy', 'radius'))
    if not np.isfinite([cx, cy, radius, selected_depth_mm]).all() or not 1 < radius <= 160:
        result['range_quality_reason'] = 'invalid_candidate'
        return result
    h, w = depth.shape[:2]
    x0, x1 = max(0, int(cx-radius)), min(w, int(cx+radius)+2)
    y0, y1 = max(0, int(cy-radius)), min(h, int(cy+radius)+2)
    if x1 <= x0 or y1 <= y0:
        return result
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx-cx)**2 + (yy-cy)**2 < (.85*radius)**2
    values = depth[y0:y1, x0:x1][disk].astype(np.float64) * (depth_scale*1000.)
    near = values[(values >= 150.) & (values <= 1000.)]
    fraction = len(near)/max(1, len(values))
    result.update(range_support_fraction=float(fraction), range_support_pixels=len(near))
    if len(near) < 30 or fraction < .45:
        result['range_quality_reason'] = 'sparse_stereo'
        return result
    q10, median, q90 = np.percentile(near, [10, 50, 90])
    spread = float(q90-q10)
    result['range_spread_mm'] = spread
    if spread > 30.:
        result['range_quality_reason'] = 'mixed_surfaces'
        return result
    if abs(float(selected_depth_mm)-median) > 15.:
        result['range_quality_reason'] = 'sample_outside_surface'
        return result
    strong = fraction >= .65 and spread <= 20.
    result.update(range_valid=True, range_quality_strong=bool(strong),
                  range_sigma_mm=3. if strong else 6., range_quality_reason='supported_surface')
    return result


def validate_ir_depth(
    candidate, depth, *, depth_scale, fx, fy, diameter_mm=40.0, max_depth_mm=1000.0
):
    """Require dense, coherent stereo under an independently found IR blob."""
    if depth is None:
        return None
    cx, cy, r = map(float, (candidate["cx"], candidate["cy"], candidate["radius"]))
    h, w = depth.shape
    x0 = max(0, int(cx - r))
    x1 = min(w, int(cx + r) + 2)
    y0 = max(0, int(cy - r))
    y1 = min(h, int(cy + r) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.85 * r) ** 2
    dp = depth[y0:y1, x0:x1].astype(float) * float(depth_scale) * 1000.0
    support = disk & (dp >= 150) & (dp <= max_depth_mm)
    values = dp[support]
    if len(values) < 30 or len(values) < 0.45 * np.count_nonzero(disk):
        return None
    q10, median, q90 = np.percentile(values, [10, 50, 90])
    if q90 - q10 > 30:
        return None
    expected = math.sqrt(fx * fy) * diameter_mm / (2 * (median + 15.0))
    if abs(r - expected) / expected > 0.25:
        return None
    mask = (support & (np.abs(dp - median) < 25)).astype(np.uint8) * 255
    return dict(
        mask=mask,
        mask_origin=(x0, y0),
        depth_assisted=True,
        depth_support_fraction=len(values) / max(1, np.count_nonzero(disk)),
        depth_support_pixels=len(values),
        depth_spread_mm=float(q90 - q10),
    )


def depth_supported_candidates(
    ir, depth, *, depth_scale, fx, fy, diameter_mm=40.0, max_depth_mm=1000.0, roi=None
):
    if depth is None or depth.shape != ir.shape:
        return []
    h, w = ir.shape
    if roi is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = map(int, roi)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return []
    mm_per_unit = float(depth_scale) * 1000.0
    patch = depth[y0:y1, x0:x1]
    near = (
        (patch >= 150.0 / mm_per_unit) & (patch <= max_depth_mm / mm_per_unit)
    ).astype(np.uint8)
    # Join tiny stereo holes without expanding the physical support contour.
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(near, 8)
    candidates = []
    for label in range(1, count):
        bx, by, bw, bh, area = map(int, stats[label])
        if area < 30 or bw < 7 or bh < 7 or bw > 160 or bh > 160:
            continue
        # Both cropped and full-image edges invalidate an incomplete surface.
        if bx <= 1 or by <= 1 or bx + bw >= x1 - x0 - 1 or by + bh >= y1 - y0 - 1:
            continue
        mask = (labels[by : by + bh, bx : bx + bw] == label).astype(np.uint8) * 255
        dp = patch[by : by + bh, bx : bx + bw].astype(np.float64) * mm_per_unit
        values = dp[(mask > 0) & (dp >= 150) & (dp <= max_depth_mm)]
        if len(values) < 30:
            continue
        median = float(np.median(values))
        q10, q90 = np.percentile(values, [10, 90])
        if q90 - q10 > 35.0:
            continue
        radius_expected = math.sqrt(fx * fy) * diameter_mm / (2 * (median + 15.0))
        support = len(values) / (math.pi * radius_expected**2)
        if not 0.40 <= support <= 1.65:
            continue
        if (
            not 1.2 * radius_expected <= min(bw, bh)
            or max(bw, bh) > 2.65 * radius_expected
        ):
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        if len(cnt) < 5:
            continue
        (cx, cy), axes, _ = cv2.fitEllipse(cnt)
        minor, major = sorted(axes)
        radius = 0.5 * math.sqrt(minor * major)
        if minor / (major + 1e-6) < 0.65:
            continue
        fill = cv2.contourArea(cnt) / (math.pi * radius**2)
        if not 0.70 < fill < 1.25:
            continue
        if abs(radius - radius_expected) / radius_expected > 0.25:
            continue
        cx += x0 + bx
        cy += y0 + by
        gx0, gy0 = x0 + bx, y0 + by
        ir_patch = ir[gy0 : gy0 + bh, gx0 : gx0 + bw]
        bright = (ir_patch >= 80) & (mask > 0)
        if np.count_nonzero(bright) < 8 or np.sum(ir_patch[bright], dtype=float) < 1500:
            continue
        cx0, cy0 = cx, cy
        cx, cy, shift = refine_ir_ball_center(ir, (cx, cy), radius)
        # The IR centroid must agree with the independently extracted surface.
        if np.hypot(cx - cx0, cy - cy0) > max(3.0, 0.25 * radius):
            continue
        if min(cx, cy, w - 1 - cx, h - 1 - cy) <= radius + 1:
            continue
        # Retain only original coherent stereo pixels, not morphology-filled
        # zeros/background or arbitrary pixels in the bounding rectangle.
        mask[(dp < median - 25) | (dp > median + 25)] = 0
        candidates.append(
            dict(
                cx=cx,
                cy=cy,
                cx_coarse=cx0,
                cy_coarse=cy0,
                center_refine_shift_px=shift,
                radius=radius,
                geometry_radius_px=radius,
                ellipse_minor_px=minor,
                ellipse_major_px=major,
                aspect_ratio=minor / (major + 1e-6),
                area_circularity=fill,
                # Radius here derives from stereo, so it must not masquerade as
                # independent IR geometry for radius/stall anomaly checks.
                geometry_quality_valid=False,
                brightness=float(np.sum(ir_patch[bright])),
                threshold=80,
                mask=mask,
                mask_origin=(gx0, gy0),
                depth_assisted=True,
                depth_support_fraction=float(support),
                depth_support_pixels=len(values),
                depth_spread_mm=float(q90 - q10),
            )
        )
    return candidates
