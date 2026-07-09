#!/usr/bin/env python3
"""Interactive HSV color tuner for the ping-pong ball detector.

**与 pingpong_node 行为 100% 一致**：复用 PingPongBallDetector 类，包括
去畸变、YOLO + 颜色验证、椭圆拟合、颜色 fallback 等完整流程。滑动条直接
修改 detector 的 HSV/conf 参数，所见即所得。

三个视图（按 m 循环切换）：
  N (默认): NODE 视图 —— 调 detector.get_observation()，绘制 viz_render 完整
            overlay。这是 pingpong_node 实际跑的路径，**这里看到的就是节点
            会看到的**。HUD 显示 detection_method (yolo / color_fallback)
            来定位误检来源。
  Y       : YOLO 候选视图。直接跑 YOLO，列出所有候选（绿框=通过 color gate，
            红框=被拒绝），用于调 HSV 让柜子/手红框、球绿框。
  C       : 纯颜色 fallback 视图。模拟 YOLO 完全漏检时的 detect_ball_by_color
            路径（全图找橙色 -> 最大轮廓 -> 半径/圆度门控）。

去畸变：默认开启（匹配节点）。可用 --no-undistort 关闭做对比实验。

输入源（按优先级）：--image > --video > RealSense 实时流。

按键：
  m  循环切换 N / Y / C 视图
  s  打印当前参数（可粘进 ros2 run --ros-args）
  w  把当前参数写入 JSON（默认 hsv_tuned.json）
  p  暂停 / 继续（暂停时滑动条立刻重算当前帧）
  d  采样：鼠标点选 + d 打印该像素的 HSV
  q / ESC  退出

用法（先关掉占用相机的进程）：
  python3 hsv_color_tuner.py --width 848 --height 480 --fps 60 \
      --yolo-model src/pingpong_moz1/pingpong_controller/models/best.pt \
      --infer-imgsz 480
  # 或离线录像/单图（自动跳过相机）：
  python3 hsv_color_tuner.py --video outputs/videos/pingpong_raw_xxx.mp4 \
      --yolo-model .../best.pt --infer-imgsz 480
  python3 hsv_color_tuner.py --image frame.png --yolo-model .../best.pt
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# Make `pingpong_controller` importable when running this script directly from
# any cwd. The tool lives at
#   <repo>/src/pingpong_moz1/pingpong_controller/tools/vision_calib/hsv_color_tuner.py
# so the package root is three directories up.
_PKG_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 '..', '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

WIN = 'HSV Tuner'

# Trackbar spec: (name, default, max). Affects what would be passed/rejected
# in the live node's process_frame.
_TRACKBARS = [
    ('H low', 5, 179),
    ('H high', 30, 179),
    ('S low', 50, 255),
    ('S high', 255, 255),
    ('V low', 50, 255),
    ('V high', 255, 255),
    ('match%', 30, 100),       # color_match_threshold * 100, applied to YOLO
    ('YOLO conf%', 50, 100),   # min YOLO confidence to show a candidate
    ('circ%', 40, 100),        # color_circularity_min * 100 (fallback view)
    ('min r', 10, 200),        # min_radius px (fallback view)
    ('max r', 200, 400),       # max_radius px (fallback view)
]

# Sampler: click on the image then press 'd' to read its HSV value.
_click = {'pt': None, 'hsv': None}

# Current view: 'N' = NODE (real detector path), 'Y' = YOLO candidates,
# 'C' = pure color fallback. Cycle with 'm'.
_view = {'mode': 'N'}
_VIEW_CYCLE = ['N', 'Y', 'C']
_VIEW_DESC = {
    'N': 'NODE (real detector path: undistort + YOLO + verify + fallback)',
    'Y': 'YOLO candidates (debug: show all candidates with PASS/REJECT)',
    'C': 'color fallback (pure HSV path, used only when YOLO misses)',
}


def _noop(_):
    pass


def _build_window():
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for name, default, maxv in _TRACKBARS:
        cv2.createTrackbar(name, WIN, default, maxv, _noop)


def _read_params():
    g = lambda n: cv2.getTrackbarPos(n, WIN)  # noqa: E731
    return {
        'hsv_lower': [g('H low'), g('S low'), g('V low')],
        'hsv_upper': [g('H high'), g('S high'), g('V high')],
        'color_match_threshold': g('match%') / 100.0,
        'yolo_conf': g('YOLO conf%') / 100.0,
        'color_circularity_min': g('circ%') / 100.0,
        'min_radius': max(1, g('min r')),
        'max_radius': max(g('min r') + 1, g('max r')),
    }


def _hsv_color_mask(bgr, lower, upper):
    """Build the orange-pixel mask used by verify_color."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv,
                       np.array(lower, dtype=np.uint8),
                       np.array(upper, dtype=np.uint8))


def _yolo_candidates(model, bgr, target_class, conf_thresh, infer_imgsz):
    """Run YOLO and return list of dicts: {mask, bbox, conf, class_name}.

    Returns every candidate matching ``target_class``; the node normally picks
    argmax(conf) but we want to see ALL of them so the user can watch the
    color gate kick out non-ball candidates.
    """
    kw = {'conf': conf_thresh, 'verbose': False}
    if infer_imgsz:
        kw['imgsz'] = int(infer_imgsz)
    results = model(bgr, **kw)
    out = []
    if not results or results[0].masks is None:
        return out
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    h, w = bgr.shape[:2]
    for i in range(len(masks)):
        name = model.names[cls_ids[i]]
        if name != target_class:
            continue
        m = cv2.resize(masks[i], (w, h))
        m = (m > 0.5).astype(np.uint8) * 255
        out.append({
            'mask': m,
            'bbox': xyxy[i].astype(int),  # x1,y1,x2,y2
            'conf': float(confs[i]),
            'class_name': name,
        })
    return out


def _verify_color(bgr, mask, hsv_lower, hsv_upper):
    """Replicate detector.verify_color exactly. Returns match_ratio."""
    color_mask = _hsv_color_mask(bgr, hsv_lower, hsv_upper)
    mask_pixels = int(np.sum(mask > 0))
    if mask_pixels == 0:
        return 0.0
    match = cv2.bitwise_and(color_mask, mask)
    return int(np.sum(match > 0)) / mask_pixels


def _analyze_yolo(bgr, candidates, p):
    """For each YOLO candidate, compute match_ratio + PASS/FAIL.

    A candidate passes iff match_ratio >= color_match_threshold (this is the
    sole color gate the node applies to YOLO results — radius/circularity gates
    only fire on the color fallback path).
    """
    items = []
    for c in candidates:
        ratio = _verify_color(bgr, c['mask'],
                              p['hsv_lower'], p['hsv_upper'])
        items.append({
            **c,
            'match_ratio': ratio,
            'passed': ratio >= p['color_match_threshold'],
        })
    # Chosen ball: argmax(conf) among PASSING candidates (matches node).
    chosen_idx = -1
    best_conf = -1.0
    for i, it in enumerate(items):
        if it['passed'] and it['conf'] > best_conf:
            best_conf = it['conf']
            chosen_idx = i
    return items, chosen_idx


def _analyze_fallback(bgr, p):
    """Pure-color path: full-image inRange -> largest contour -> gates.

    Mirrors detect_ball_by_color. Only used by the node when YOLO misses.
    """
    color_mask = _hsv_color_mask(bgr, p['hsv_lower'], p['hsv_upper'])
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    result = {
        'passed': False, 'reason': 'no contour',
        'cx': None, 'cy': None, 'radius': None,
        'area': 0.0, 'circularity': 0.0,
        'n_contours': len(contours),
    }
    if not contours:
        return color_mask, result
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    result.update({'cx': cx, 'cy': cy, 'radius': radius, 'area': area})
    if area < 100:
        result['reason'] = f'area {area:.0f}<100'
        return color_mask, result
    if radius < p['min_radius'] or radius > p['max_radius']:
        result['reason'] = (
            f"r {radius:.0f} not in [{p['min_radius']},{p['max_radius']}]")
        return color_mask, result
    circle_area = np.pi * radius * radius
    circularity = area / circle_area if circle_area > 0 else 0.0
    result['circularity'] = circularity
    if circularity < p['color_circularity_min']:
        result['reason'] = (
            f"circ {circularity:.2f}<{p['color_circularity_min']:.2f}")
        return color_mask, result
    result['passed'] = True
    result['reason'] = 'BALL (color fallback)'
    return color_mask, result


def _draw_text(img, text, org, color=(255, 255, 255), scale=0.55, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick)


def _render_yolo(bgr, items, chosen_idx, color_mask, p):
    """Three panes: original | global orange mask | YOLO verdict overlay."""
    h, w = bgr.shape[:2]
    mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    verdict = bgr.copy()

    if not items:
        cv2.rectangle(verdict, (0, 0), (w, 30), (0, 80, 200), -1)
        _draw_text(verdict, 'YOLO: no candidates (lower YOLO conf%?)',
                   (8, 22), scale=0.7, thick=2)
    else:
        # Draw every YOLO candidate. Green = passes color gate, red = rejected.
        # Yellow ring = the one the node would actually pick (chosen).
        for i, it in enumerate(items):
            x1, y1, x2, y2 = it['bbox']
            color = (0, 220, 0) if it['passed'] else (0, 0, 230)
            cv2.rectangle(verdict, (x1, y1), (x2, y2), color, 2)
            tag = (f"YOLO {it['conf']:.2f} | match {it['match_ratio']:.2f}"
                   f" {'PASS' if it['passed'] else 'REJECT'}")
            _draw_text(verdict, tag, (x1, max(y1 - 6, 12)),
                       color=color, scale=0.5, thick=1)
            if i == chosen_idx:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(verdict, (cx, cy), max(8, (x2 - x1) // 2 + 4),
                           (0, 255, 255), 3)

        passed = chosen_idx >= 0
        head = (f"CHOSEN BALL (conf {items[chosen_idx]['conf']:.2f}, "
                f"match {items[chosen_idx]['match_ratio']:.2f})"
                if passed else
                f"NO BALL passed color gate ({len(items)} YOLO cands)")
        bar_color = (0, 220, 0) if passed else (0, 0, 230)
        cv2.rectangle(verdict, (0, 0), (w, 30), bar_color, -1)
        _draw_text(verdict, head, (8, 22), scale=0.7, thick=2)

    # HSV sample HUD
    if _click['hsv'] is not None:
        hh, ss, vv = _click['hsv']
        _draw_text(verdict, f'click HSV=({hh},{ss},{vv})',
                   (8, h - 12), color=(0, 255, 255), scale=0.6, thick=2)

    # Param HUD (bottom of pane)
    hud = (f"HSV {p['hsv_lower']}-{p['hsv_upper']}  "
           f"match>={p['color_match_threshold']:.2f}  "
           f"YOLOconf>={p['yolo_conf']:.2f}")
    _draw_text(verdict, hud, (8, h - 36), color=(255, 255, 0), scale=0.5)
    _draw_text(verdict, '[m] switch view  [s] print  [w] save  [p] pause',
               (8, h - 18), color=(200, 200, 200), scale=0.45)
    return np.hstack([bgr, mask_bgr, verdict])


def _render_color(bgr, color_mask, result, p):
    """Fallback view (no YOLO): the original 3-pane look."""
    h, w = bgr.shape[:2]
    mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    verdict = bgr.copy()
    passed = result['passed']
    color = (0, 220, 0) if passed else (0, 0, 230)
    if result['cx'] is not None and result['radius'] is not None:
        c = (int(round(result['cx'])), int(round(result['cy'])))
        r = int(round(result['radius']))
        cv2.circle(verdict, c, max(r, 1), color, 2)
        cv2.drawMarker(verdict, c, color, cv2.MARKER_CROSS, 16, 2)

    head = ('BALL (fallback)' if passed else 'NOT BALL') + f": {result['reason']}"
    cv2.rectangle(verdict, (0, 0), (w, 30), color, -1)
    _draw_text(verdict, head, (8, 22), scale=0.7, thick=2)

    lines = []
    if result['radius'] is not None:
        lines.append(f"radius   {result['radius']:.1f}px  "
                     f"[{p['min_radius']},{p['max_radius']}]")
    lines.append(f"circ     {result['circularity']:.2f}  "
                 f">= {p['color_circularity_min']:.2f}")
    lines.append(f"area     {result['area']:.0f}px^2  (>=100)")
    lines.append(f"contours {result['n_contours']}")
    y = 52
    for ln in lines:
        _draw_text(verdict, ln, (8, y))
        y += 22
    _draw_text(verdict, '[m] -> YOLO view (the path the node uses)',
               (8, h - 18), color=(200, 200, 200), scale=0.45)
    return np.hstack([bgr, mask_bgr, verdict])


def _on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        _click['pt'] = (x, y)


def _print_params(p):
    print('\n=== tuned color params ===')
    print(f"  hsv_lower = {p['hsv_lower']}")
    print(f"  hsv_upper = {p['hsv_upper']}")
    print(f"  color_match_threshold = {p['color_match_threshold']:.2f}")
    print(f"  color_circularity_min = {p['color_circularity_min']:.2f}  "
          f"(fallback-only)")
    print('  --- copy into ros2 run --ros-args ---')
    print(f"    -p hsv_lower:='[{p['hsv_lower'][0]},{p['hsv_lower'][1]},"
          f"{p['hsv_lower'][2]}]' \\")
    print(f"    -p hsv_upper:='[{p['hsv_upper'][0]},{p['hsv_upper'][1]},"
          f"{p['hsv_upper'][2]}]' \\")
    print(f"    -p color_match_threshold:={p['color_match_threshold']:.2f} \\")
    print(f"    -p color_circularity_min:={p['color_circularity_min']:.2f}")


def _open_source(args):
    """Return (read_fn, close_fn, is_stream). read_fn() -> bgr or None."""
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f'ERROR: cannot read image {args.image}', file=sys.stderr)
            sys.exit(1)
        return (lambda: img.copy()), (lambda: None), False
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f'ERROR: cannot open video {args.video}', file=sys.stderr)
            sys.exit(1)

        def _read_loop():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            return frame if ok else None
        return _read_loop, cap.release, True
    try:
        import pyrealsense2 as rs
    except ImportError:
        print('ERROR: pyrealsense2 not available; use --video or --image',
              file=sys.stderr)
        sys.exit(1)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    # pipeline.start() can block indefinitely if the camera was left in a bad
    # state by a previous SIGKILL. Print before we call it so the user can
    # tell whether the hang is here vs. later.
    print(f'[info] starting RealSense {args.width}x{args.height}@{args.fps}...',
          flush=True)
    try:
        pipeline.start(config)
    except RuntimeError as exc:
        print(f'ERROR: cannot start RealSense ({exc}).', file=sys.stderr)
        print('       Camera likely held by another process. Try:',
              file=sys.stderr)
        print('         pkill -9 -f realsense2_camera; '
              'pkill -9 -f pingpong_node', file=sys.stderr)
        print('       Or physically unplug/replug the camera USB.',
              file=sys.stderr)
        sys.exit(1)
    print('[info] RealSense streaming; warming up (30 frames)...',
          flush=True)
    # Warmup: pull 30 frames to let the camera stabilize (matches pingpong_node)
    for i in range(30):
        try:
            frames = pipeline.wait_for_frames(5000)  # 5s timeout for warmup
            cf = frames.get_color_frame()
            if cf and i == 0:
                img = np.asanyarray(cf.get_data())
                print(f'[info] first frame: {img.shape}, dtype={img.dtype}',
                      flush=True)
        except RuntimeError as exc:
            print(f'[warn] warmup frame {i} failed: {exc}', file=sys.stderr)
    print('[info] warmup complete, starting main loop', flush=True)

    def _read_rs():
        # Use a 1 s timeout so Ctrl+C and shutdown actually work — the no-arg
        # form blocks in C-land and ignores signals.
        try:
            frames = pipeline.wait_for_frames(1000)
        except RuntimeError:
            return None
        cf = frames.get_color_frame()
        if not cf:
            return None
        return np.asanyarray(cf.get_data())
    return _read_rs, pipeline.stop, True


def _resolve_yolo_model(path):
    if path and os.path.exists(path):
        return path
    # auto-find best.pt under the controller's models dir
    here = os.path.dirname(os.path.abspath(__file__))
    auto = os.path.normpath(os.path.join(here, '..', '..', 'models', 'best.pt'))
    if os.path.exists(auto):
        print(f'[info] using auto-located model: {auto}')
        return auto
    print(f'ERROR: YOLO model not found at {path!r} or {auto!r}',
          file=sys.stderr)
    print('       pass --yolo-model <path>.pt', file=sys.stderr)
    sys.exit(1)


def _resolve_calibration_json(path, width, height):
    """Auto-locate calibration JSON matching width/height; or return user path.

    Mirrors pingpong_node._resolve_calibration_json_path so tuner picks the
    same file when both run with the same --width.
    """
    if path and str(path).lower() not in ('auto', ''):
        p = os.path.expandvars(os.path.expanduser(str(path)))
        if not os.path.exists(p):
            print(f'WARNING: calibration JSON not found at {p!r}',
                  file=sys.stderr)
            return None
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    calib_dir = os.path.normpath(
        os.path.join(here, '..', '..', 'outputs', 'vision_calib'))
    candidates = sorted(glob.glob(os.path.join(
        calib_dir, 'eye_hand_result_*_undistorted.json')))
    # Prefer one with matching width in filename, else any with matching res
    from pingpong_controller.calibration_loader import load_calibration
    for c in candidates:
        try:
            data = load_calibration(c)
        except Exception:
            continue
        if data.image_width == width and data.image_height == height:
            print(f'[info] auto-selected calibration: {c}')
            return c
    print(f'[info] no calibration JSON matching {width}x{height} '
          f'in {calib_dir}; running without extrinsics',
          file=sys.stderr)
    return None


def _build_detector(args):
    """Instantiate PingPongBallDetector with the exact defaults pingpong_node
    uses for direct_realsense mode. Returns the detector.

    Trackbars later mutate detector.hsv_lower/upper/match/conf at runtime so
    the live observation reflects the slider state.
    """
    # Lazy import so --image / --video runs don't require the full controller
    # package import path resolved at module load.
    from pingpong_controller.ball_detector import PingPongBallDetector
    from pingpong_controller.calibration_loader import load_calibration

    yolo_path = _resolve_yolo_model(args.yolo_model)

    calibration_data = None
    calib_path = _resolve_calibration_json(
        args.calibration_json, args.width, args.height)
    if calib_path is not None:
        try:
            calibration_data = load_calibration(calib_path)
            print(f'[info] calibration loaded: '
                  f'{calibration_data.image_width}x'
                  f'{calibration_data.image_height}')
        except Exception as exc:
            print(f'WARNING: failed to load {calib_path}: {exc}',
                  file=sys.stderr)
            calibration_data = None

    print(f'[info] building PingPongBallDetector '
          f'(use_undistort={args.use_undistort}, '
          f'conf={args.conf_threshold})', flush=True)
    detector = PingPongBallDetector(
        yolo_model_path=yolo_path,
        ball_diameter_mm=40.0,
        width=args.width,
        height=args.height,
        fps=args.fps,
        conf_threshold=args.conf_threshold,
        target_class=args.target_class,
        camera_tilt_deg=args.camera_tilt_deg,
        use_kalman=False,            # match node default
        infer_imgsz=args.infer_imgsz,
        device=args.device,
        half=False,
        use_undistort=args.use_undistort,
        print_every=999999,
        profile_every=0,
        use_camera=False,            # we feed frames manually
        # If we have calibration JSON, prefer it (matches node direct-realsense
        # mode); else fall back to no extrinsics.
        calibration_data=calibration_data,
        output_frame='base_link' if calibration_data is not None else 'camera',
        enable_ellipse_fit=args.enable_ellipse_fit,
        enable_temporal_tracker=args.enable_temporal_tracker,
        tracker_max_miss_bridge=2,
        tracker_gate_px=120.0,
        hsv_lower=args.hsv_lower,
        hsv_upper=args.hsv_upper,
        color_match_threshold=args.color_match_threshold,
        color_circularity_min=args.color_circularity_min,
    )
    return detector


def _sync_detector_from_trackbars(detector, p):
    """Push current trackbar values into detector so next get_observation()
    sees them. Cheap; called every frame."""
    detector.hsv_lower = np.array(p['hsv_lower'], dtype=np.int32)
    detector.hsv_upper = np.array(p['hsv_upper'], dtype=np.int32)
    detector.color_match_threshold = float(p['color_match_threshold'])
    detector.color_circularity_min = float(p['color_circularity_min'])
    detector.conf_threshold = float(p['yolo_conf'])


def _render_node_view(frame, observation, detector, p, frame_count):
    """NODE view: feed frame through detector.get_observation, render with
    viz_render.render_detection_overlay — pixel-identical to pingpong_node.

    `frame` here is the raw (possibly distorted) BGR image. The detector
    internally undistorts. To get pixels that line up with the overlay we
    also undistort the displayed frame, matching pingpong_node's viz worker.
    """
    from pingpong_controller.viz_render import render_detection_overlay

    # Match node viz worker: remap the display frame so the drawn circle/bbox
    # lands on the right pixels.
    viz_bgr = frame
    if (detector.use_undistort
            and getattr(detector, 'map1', None) is not None
            and getattr(detector, 'map2', None) is not None):
        try:
            viz_bgr = cv2.remap(
                frame, detector.map1, detector.map2, cv2.INTER_LINEAR)
        except cv2.error:
            viz_bgr = frame

    output_frame = getattr(detector, 'output_frame', 'camera') or 'camera'
    overlay = render_detection_overlay(
        viz_bgr,
        observation,
        frame_count=frame_count,
        detection_count=int(observation.get('detected', False)),
        callback_hz=0.0,
        output_frame=output_frame,
    )

    # Bottom HUD: HSV/match/conf — the params the user is tweaking.
    h, w = overlay.shape[:2]
    hud = (f"HSV {p['hsv_lower']}-{p['hsv_upper']}  "
           f"match>={p['color_match_threshold']:.2f}  "
           f"conf>={p['yolo_conf']:.2f}  "
           f"circ>={p['color_circularity_min']:.2f}")
    _draw_text(overlay, hud, (8, h - 32), color=(255, 255, 0), scale=0.5)
    _draw_text(overlay, '[m] cycle view  [s] print  [w] save  [p] pause',
               (8, h - 14), color=(200, 200, 200), scale=0.45)
    return overlay


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group(
        'input source (priority: image > video > realsense)')
    src.add_argument('--image', default='', help='tune on a single image')
    src.add_argument('--video', default='', help='tune on a video file (loops)')
    ap.add_argument('--width', type=int, default=848, help='RealSense width')
    ap.add_argument('--height', type=int, default=480, help='RealSense height')
    ap.add_argument('--fps', type=int, default=60, help='RealSense fps')
    ap.add_argument('--yolo-model', default='',
                    help='path to YOLO .pt (default: auto-locate best.pt)')
    ap.add_argument('--target-class', default='pingpong ball',
                    help='YOLO class name to keep (default "pingpong ball")')
    ap.add_argument('--infer-imgsz', type=int, default=480,
                    help='YOLO inference image size; match the node run')
    ap.add_argument('--device', default='0',
                    help='YOLO device, e.g. "0" or "cpu"')
    ap.add_argument('--out', default='hsv_tuned.json',
                    help='path to write params on "w"')

    # Detector init params — defaults match pingpong_node so the NODE view
    # reproduces the node's behavior pixel-for-pixel.
    det = ap.add_argument_group('detector init (defaults match pingpong_node)')
    det.add_argument('--use-undistort', dest='use_undistort',
                     action='store_true', default=True,
                     help='undistort frames before detection (DEFAULT, '
                          'matches node)')
    det.add_argument('--no-undistort', dest='use_undistort',
                     action='store_false',
                     help='skip undistortion (for A/B comparing with node)')
    det.add_argument('--calibration-json', default='auto',
                     help='path to eye_hand_result_*_undistorted.json '
                          '(default: auto-locate by width)')
    det.add_argument('--conf-threshold', type=float, default=0.5,
                     help='YOLO confidence threshold (initial; trackbar '
                          'overrides)')
    det.add_argument('--camera-tilt-deg', type=float, default=40.0,
                     help='camera tilt in degrees (matches node default)')
    det.add_argument('--hsv-lower', type=int, nargs=3, default=[5, 50, 50],
                     metavar=('H', 'S', 'V'),
                     help='initial HSV lower bound (trackbar overrides)')
    det.add_argument('--hsv-upper', type=int, nargs=3, default=[30, 255, 255],
                     metavar=('H', 'S', 'V'),
                     help='initial HSV upper bound (trackbar overrides)')
    det.add_argument('--color-match-threshold', type=float, default=0.3,
                     help='initial match-ratio gate (trackbar overrides)')
    det.add_argument('--color-circularity-min', type=float, default=0.4,
                     help='initial fallback-path circularity gate (trackbar '
                          'overrides)')
    det.add_argument('--enable-ellipse-fit', action='store_true', default=True,
                     help='use ellipse minor axis for radius (matches node)')
    det.add_argument('--enable-temporal-tracker', action='store_true',
                     default=False,
                     help='enable temporal tracker layer (matches node '
                          'default off)')

    args = ap.parse_args()

    # CRITICAL: create OpenCV window BEFORE loading YOLO.
    # YOLO initialization changes OpenCV/CUDA state such that subsequent
    # cv2.namedWindow() can block indefinitely. Build the window first.
    print('[info] creating GUI window and trackbars...', flush=True)
    _build_window()
    # Seed trackbars from CLI defaults so initial state matches --hsv-* etc.
    cv2.setTrackbarPos('H low', WIN, args.hsv_lower[0])
    cv2.setTrackbarPos('H high', WIN, args.hsv_upper[0])
    cv2.setTrackbarPos('S low', WIN, args.hsv_lower[1])
    cv2.setTrackbarPos('S high', WIN, args.hsv_upper[1])
    cv2.setTrackbarPos('V low', WIN, args.hsv_lower[2])
    cv2.setTrackbarPos('V high', WIN, args.hsv_upper[2])
    cv2.setTrackbarPos('match%', WIN, int(args.color_match_threshold * 100))
    cv2.setTrackbarPos('YOLO conf%', WIN, int(args.conf_threshold * 100))
    cv2.setTrackbarPos('circ%', WIN, int(args.color_circularity_min * 100))
    print('[info] GUI window ready', flush=True)

    # Build detector (loads YOLO internally). Matches pingpong_node config.
    detector = _build_detector(args)
    # Keep a direct ref to the loaded YOLO model for the Y/C debug views, so
    # we don't load it twice.
    model = detector.yolo_model
    print('[info] YOLO ready via detector. Classes:',
          {i: n for i, n in model.names.items()})
    if args.target_class not in set(model.names.values()):
        print(f"WARNING: target class {args.target_class!r} not in model "
              f"classes; pass --target-class to override.", file=sys.stderr)

    read_fn, close_fn, is_stream = _open_source(args)
    print('[info] setting mouse callback...', flush=True)
    cv2.setMouseCallback(WIN, _on_mouse)

    # Show a placeholder image immediately so the window actually appears
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'Loading...', (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.imshow(WIN, placeholder)
    cv2.waitKey(1)  # Force window to render
    print('[info] GUI window displayed', flush=True)

    print(f"[info] starting in view {_view['mode']} "
          f"({_VIEW_DESC[_view['mode']]})")
    print('keys: m=cycle view  s=print  w=save json  '
          'p=pause  d=sample HSV  q/ESC=quit', flush=True)

    paused = False
    last_frame = None
    last_candidates = None  # cache YOLO output when paused (slider drag)
    last_frame_id = -1
    frame_id = 0
    while True:
        if not paused or last_frame is None:
            frame = read_fn()
            if frame is None:
                if is_stream:
                    continue
                break
            last_frame = frame
            frame_id += 1
        frame = last_frame
        p = _read_params()

        # Push slider state into the detector so its next call sees current
        # HSV/match/conf values. Cheap; safe to do every frame.
        _sync_detector_from_trackbars(detector, p)

        # HSV sampler (sample on the RAW frame so the printed value is
        # comparable across views and against pingpong_node's input.)
        if _click['pt'] is not None:
            x, y = _click['pt']
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                _click['hsv'] = [int(v) for v in hsv[y, x]]

        if _view['mode'] == 'N':
            # NODE view: run the real detector pipeline, render with the same
            # overlay function pingpong_node uses. This is the source of truth
            # for "does this scene reproduce node behavior?".
            observation = detector.get_observation(frame)
            panes = _render_node_view(
                frame, observation, detector, p, frame_id)
        elif _view['mode'] == 'Y':
            # Cache YOLO candidates per frame so dragging sliders is responsive
            # (HSV/match changes don't require re-running YOLO).
            if last_frame_id != frame_id or last_candidates is None:
                last_candidates = _yolo_candidates(
                    model, frame, args.target_class,
                    p['yolo_conf'], args.infer_imgsz)
                last_frame_id = frame_id
            items, chosen_idx = _analyze_yolo(frame, last_candidates, p)
            color_mask = _hsv_color_mask(frame, p['hsv_lower'], p['hsv_upper'])
            panes = _render_yolo(frame, items, chosen_idx, color_mask, p)
        else:  # 'C'
            color_mask, result = _analyze_fallback(frame, p)
            panes = _render_color(frame, color_mask, result, p)

        cv2.imshow(WIN, panes)
        key = cv2.waitKey(1 if (is_stream and not paused) else 30) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            cur = _VIEW_CYCLE.index(_view['mode'])
            _view['mode'] = _VIEW_CYCLE[(cur + 1) % len(_VIEW_CYCLE)]
            print(f"view -> {_view['mode']} ({_VIEW_DESC[_view['mode']]})")
            last_candidates = None
        elif key == ord('s'):
            _print_params(p)
        elif key == ord('w'):
            with open(args.out, 'w') as f:
                json.dump(p, f, indent=2)
            print(f'wrote {args.out}')
            _print_params(p)
        elif key == ord('p'):
            paused = not paused
            print('paused' if paused else 'resumed')
        elif key == ord('d'):
            if _click['hsv'] is not None:
                print(f'sampled HSV at {_click["pt"]} = {_click["hsv"]}')
            else:
                print('click on the image first, then press d')

    close_fn()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
