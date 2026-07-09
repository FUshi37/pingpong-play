"""Standalone detection-overlay renderer.

Factored out of PingPongControllerNode so that BOTH the live visualization
worker and the offline raw-video annotation tool draw exactly the same
overlay (mask tint, raw/smoothed circles, color bbox, HUD text). Keep this
free of ROS / node state: everything it needs is passed in explicitly.
"""
import cv2
import numpy as np


# Default thresholds for the sim done-conditions (mjx_juggle_env.py:2778-2786).
DEFAULT_DONE_THRESHOLDS = {
    'ball_z_floor': 0.8,
    'ball_z_ceiling': 1.9,
    'ball_x_range': 0.5,
    'ball_y_range': 0.5,
    'racket_anchor_distance': 1.1,
    'racket_z_high_offset': 0.24,
    'racket_z_low_offset': -0.12,
}


def eval_done_conditions(
    ball_pos_m: np.ndarray,
    racket_anchor_m: np.ndarray = None,
    racket_pos_m: np.ndarray = None,
    thresholds: dict = None,
) -> dict:
    """Evaluate the sim termination conditions (mjx_juggle_env.py:2778-2786).

    Pure function (no counters) so it is safe to call from the viz worker.

    All inputs in meters, sim/base frame (base at origin, z = world height).
    Returns {condition_name: bool} plus an 'any' key (True if ANY fires).

    Ball z conditions are always evaluated. Ball x/y conditions need the
    racket anchor, and racket conditions need the current racket pose from FK.
    Base x/y checks are omitted: the real robot base is stationary at origin.
    """
    t = dict(DEFAULT_DONE_THRESHOLDS)
    if thresholds:
        t.update(thresholds)

    bx, by, bz = float(ball_pos_m[0]), float(ball_pos_m[1]), float(ball_pos_m[2])
    conds = {
        "ball_too_low": bz < t['ball_z_floor'],
        "ball_too_high": bz > t['ball_z_ceiling'],
    }
    if racket_anchor_m is not None:
        ax, ay, az = (float(racket_anchor_m[0]), float(racket_anchor_m[1]),
                      float(racket_anchor_m[2]))
        conds["ball_x_out_of_bounds"] = abs(bx - ax) > t['ball_x_range']
        conds["ball_y_out_of_bounds"] = abs(by - ay) > t['ball_y_range']
    if racket_anchor_m is not None and racket_pos_m is not None:
        rpos = np.asarray(racket_pos_m, dtype=float).reshape(3)
        anchor = np.asarray(racket_anchor_m, dtype=float).reshape(3)
        dist = float(np.linalg.norm(rpos - anchor))
        z_off = float(rpos[2] - az)
        conds["racket_too_far_from_anchor"] = dist > t['racket_anchor_distance']
        conds["racket_too_high"] = z_off > t['racket_z_high_offset']
        conds["racket_too_low"] = z_off < t['racket_z_low_offset']

    conds["any"] = any(conds.values())
    return conds


def draw_sim_done_overlay(canvas: np.ndarray,
                          sim_ball_pos_m: np.ndarray,
                          sim_ball_vel_m_s: np.ndarray,
                          racket_anchor_m: np.ndarray,
                          racket_pos_m: np.ndarray = None,
                          thresholds: dict = None,
                          detected: bool = True,
                          label: str = "[SIM base frame]") -> np.ndarray:
    """Draw the sim-frame ball state + done-condition status onto ``canvas``.

    Standalone so BOTH viz paths use it: the hybrid detector's own composite
    (RGB|IR) render and the non-hybrid ``render_detection_overlay``. Mutates and
    returns ``canvas`` (a bottom-left translucent HUD block).

    Shows: ball pos (m) and vel (m/s) in sim/base frame, and each done-condition
    with its live value; the aggregate status line is green ("OK") or red (lists
    active violations). When the ball is not detected, values are dimmed and the
    status shows "no ball".
    """
    if sim_ball_pos_m is None or sim_ball_vel_m_s is None:
        return canvas

    height, width = canvas.shape[:2]
    bx, by, bz = (float(sim_ball_pos_m[0]), float(sim_ball_pos_m[1]),
                  float(sim_ball_pos_m[2]))
    vx, vy, vz = (float(sim_ball_vel_m_s[0]), float(sim_ball_vel_m_s[1]),
                  float(sim_ball_vel_m_s[2]))

    lines = []  # (text, color) BGR
    white = (230, 230, 230)
    green = (0, 220, 0)
    red = (0, 0, 255)
    dim = (150, 150, 150)

    lines.append((label, white))
    lines.append((f"pos ({bx:6.2f}, {by:6.2f}, {bz:6.2f}) m",
                  white if detected else dim))
    lines.append((f"vel ({vx:6.2f}, {vy:6.2f}, {vz:6.2f}) m/s",
                  white if detected else dim))

    if not detected:
        lines.append(("DONE: no ball", dim))
    else:
        done = eval_done_conditions(
            sim_ball_pos_m, racket_anchor_m, racket_pos_m, thresholds)
        active = [k for k, v in done.items() if v and k != "any"]
        # Per-condition value read-out so the user sees margins, not just flags.
        t = dict(DEFAULT_DONE_THRESHOLDS)
        if thresholds:
            t.update(thresholds)
        lines.append((
            f"z {bz:5.2f} in [{t['ball_z_floor']:.2f},{t['ball_z_ceiling']:.2f}]",
            red if (done['ball_too_low'] or done['ball_too_high']) else green))
        if racket_anchor_m is not None:
            ax, ay, az = (float(racket_anchor_m[0]), float(racket_anchor_m[1]),
                          float(racket_anchor_m[2]))
            lines.append((
                f"|dx| {abs(bx-ax):.2f} |dy| {abs(by-ay):.2f} "
                f"(<{t['ball_x_range']:.2f})",
                red if (done['ball_x_out_of_bounds']
                        or done['ball_y_out_of_bounds']) else green))
        else:
            az = 0.0
            lines.append(("anchor/FK unavailable: xy/racket checks skipped", dim))
        if 'racket_too_far_from_anchor' in done:
            lines.append((
                f"racket dz {float(racket_pos_m[2])-az:+.2f} "
                f"[-{-t['racket_z_low_offset']:.2f},"
                f"{t['racket_z_high_offset']:.2f}]",
                red if (done['racket_too_high'] or done['racket_too_low']
                        or done['racket_too_far_from_anchor']) else green))
        if done["any"]:
            lines.append((f"DONE: {', '.join(active)}", red))
        else:
            lines.append(("DONE: OK (in bounds)", green))

    # Bottom-left translucent block.
    line_h = 20
    pad = 6
    box_w = max(430, width // 3)
    box_h = line_h * len(lines) + 2 * pad
    x0 = 8
    y0 = height - box_h - 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + box_w, y0 + box_h),
                  (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, dst=canvas)
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (x0, y0 + pad + 14 + i * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return canvas


def render_detection_overlay(frame: np.ndarray,
                             observation: dict,
                             frame_count: int = 0,
                             detection_count: int = 0,
                             callback_hz: float = 0.0,
                             output_frame: str = 'camera',
                             sim_ball_pos_m: np.ndarray = None,
                             sim_ball_vel_m_s: np.ndarray = None,
                             racket_anchor_m: np.ndarray = None,
                             racket_pos_m: np.ndarray = None,
                             done_thresholds: dict = None,
                             sim_detected: bool = None,
                             sim_label: str = "[SIM base frame]") -> np.ndarray:
    """Overlay detection results on a BGR frame and return a new image.

    Draws (when available): YOLO mask tint, fitted circle, smoothed circle,
    color-refined bbox, and a status HUD (frame index, callback Hz, detection
    rate, pixel/depth/3D position).

    If sim_ball_pos_m, sim_ball_vel_m_s, and racket_anchor_m are provided,
    also displays the ball state in sim frame (base yaw convention, meters)
    and evaluates + color-codes the sim done-conditions so the user can see
    when the ball would terminate an episode in training.
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
        pos = observation.get('position')
        if pos is None:
            pos = [0.0, 0.0, 0.0]
        lines.append(
            f"Pos[{output_frame}] "
            f"({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f}) mm"
        )
        vel = observation.get('velocity')
        if vel is None:
            vel = [0.0, 0.0, 0.0]
        lines.append(
            f"Vel[{output_frame}] "
            f"({vel[0]:6.1f}, {vel[1]:6.1f}, {vel[2]:6.1f}) mm/s"
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
    box_w = max(450, width // 3)
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

    # Sim-frame ball state + done-condition status (bottom-left block). Shared
    # with the hybrid viz path via draw_sim_done_overlay.
    if sim_ball_pos_m is not None and sim_ball_vel_m_s is not None:
        draw_sim_done_overlay(
            canvas, sim_ball_pos_m, sim_ball_vel_m_s, racket_anchor_m,
            racket_pos_m=racket_pos_m, thresholds=done_thresholds,
            detected=detected if sim_detected is None else bool(sim_detected),
            label=sim_label)

    return canvas
