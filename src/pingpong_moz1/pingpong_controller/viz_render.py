"""Standalone detection-overlay renderer.

Factored out of PingPongControllerNode so that BOTH the live visualization
worker and the offline raw-video annotation tool draw exactly the same
overlay (mask tint, raw/smoothed circles, color bbox, HUD text). Keep this
free of ROS / node state: everything it needs is passed in explicitly.
"""
import cv2
import numpy as np


def render_detection_overlay(frame: np.ndarray,
                             observation: dict,
                             frame_count: int = 0,
                             detection_count: int = 0,
                             callback_hz: float = 0.0,
                             output_frame: str = 'camera') -> np.ndarray:
    """Overlay detection results on a BGR frame and return a new image.

    Draws (when available): YOLO mask tint, fitted circle, smoothed circle,
    color-refined bbox, and a status HUD (frame index, callback Hz, detection
    rate, pixel/depth/3D position).
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    detected = bool(observation.get('detected', False))
    raw = observation.get('raw_measurement') or {}

    # Mask tint: faintly color-fill the YOLO mask so the user can see what the
    # network actually segmented before we collapsed it to a circle.
    mask = raw.get('mask')
    if mask is not None and isinstance(mask, np.ndarray) and mask.size > 0:
        try:
            if mask.shape[:2] != canvas.shape[:2]:
                mask = cv2.resize(
                    mask, (width, height), interpolation=cv2.INTER_NEAREST)
            colored = np.zeros_like(canvas)
            colored[mask > 0] = (0, 200, 255)  # orange-ish in BGR
            canvas = cv2.addWeighted(canvas, 1.0, colored, 0.35, 0)
        except cv2.error:
            pass

    if detected:
        cx_raw = raw.get('image_x')
        cy_raw = raw.get('image_y')
        radius_raw = raw.get('radius_px')
        cx_smooth = raw.get('smoothed_cx')
        cy_smooth = raw.get('smoothed_cy')
        radius_smooth = raw.get('smoothed_radius')

        if (cx_raw is not None and cy_raw is not None
                and radius_raw is not None):
            cv2.circle(
                canvas,
                (int(round(cx_raw)), int(round(cy_raw))),
                int(round(max(radius_raw, 1.0))),
                (0, 255, 0), 2)
            cv2.drawMarker(
                canvas,
                (int(round(cx_raw)), int(round(cy_raw))),
                (0, 255, 0), markerType=cv2.MARKER_CROSS,
                markerSize=14, thickness=2)

        if (cx_smooth is not None and cy_smooth is not None
                and radius_smooth is not None
                and (cx_smooth != cx_raw or cy_smooth != cy_raw)):
            cv2.circle(
                canvas,
                (int(round(cx_smooth)), int(round(cy_smooth))),
                int(round(max(radius_smooth, 1.0))),
                (0, 200, 255), 1, cv2.LINE_AA)

        color_bbox = raw.get('color_bbox')
        if color_bbox is not None and len(color_bbox) == 4:
            bx, by, bw, bh = (int(v) for v in color_bbox)
            cv2.rectangle(
                canvas, (bx, by), (bx + bw, by + bh), (0, 165, 255), 1)

    # Status text (top-left HUD).
    detection_rate = (100.0 * detection_count / frame_count
                      if frame_count > 0 else 0.0)

    lines = [
        f"Frame {frame_count} | Callback {callback_hz:5.1f} Hz | "
        f"Detect {detection_rate:5.1f}%",
    ]

    if detected:
        depth_mm = raw.get('depth_mm', 0.0) or 0.0
        method = raw.get('detection_method', '?')
        lines.append(
            f"Pixel ({raw.get('image_x', 0.0):6.1f}, "
            f"{raw.get('image_y', 0.0):6.1f}) | "
            f"r {raw.get('radius_px', 0.0):5.1f}px | "
            f"depth {depth_mm:6.1f} mm | {method}"
        )
        pos = observation.get('position') or [0.0, 0.0, 0.0]
        lines.append(
            f"Pos[{output_frame}] "
            f"({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f}) mm"
        )
        if raw.get('motion_blur_detected'):
            lines.append(
                "MOTION BLUR"
                + (" (corrected)" if raw.get('motion_blur_corrected') else "")
            )
    else:
        reason = observation.get('reject_reason') or 'no_detection'
        lines.append(f"NOT DETECTED [{reason}]")

    # Draw HUD with a translucent backdrop so it's readable on any scene.
    x0, y0 = 8, 8
    line_h = 22
    box_w = max(380, width // 3)
    box_h = line_h * len(lines) + 12
    overlay = canvas.copy()
    cv2.rectangle(
        overlay, (x0 - 4, y0 - 4),
        (x0 + box_w, y0 + box_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)
    for i, text in enumerate(lines):
        cv2.putText(
            canvas, text, (x0, y0 + 16 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0) if detected else (0, 0, 255),
            1, cv2.LINE_AA)

    return canvas
