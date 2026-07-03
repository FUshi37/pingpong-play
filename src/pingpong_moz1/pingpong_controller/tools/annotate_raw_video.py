#!/usr/bin/env python3
"""Offline annotation of a raw RealSense recording.

Takes a raw video saved by pingpong_node (save_raw_video:=true) — the
unprocessed RealSense RGB stream — and produces a detection-annotated video
using the SAME detector and overlay renderer as the live node, so the result
matches what the live window would have shown, but rendered offline at full
quality with zero real-time pressure.

Pipeline per frame (mirrors PingPongControllerNode._handle_cv_image):
  1. detector.get_observation(raw_frame, observation_time=frame_idx/fps)
     -> internally undistorts for detection, runs YOLO + smoothing/Kalman.
  2. Remap the raw frame to undistorted space (same as the live viz worker)
     so overlay pixels land correctly.
  3. render_detection_overlay(undistorted_frame, observation, ...).
  4. Write the annotated frame to the output video.

Example:
  python3 annotate_raw_video.py \
      --input outputs/videos/pingpong_raw_20260625_120000.mp4 \
      --output outputs/videos/pingpong_annotated.mp4
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Allow running both as an installed module and as a loose script.
try:
    from pingpong_controller.ball_detector import PingPongBallDetector
    from pingpong_controller.calibration_loader import load_calibration
    from pingpong_controller.viz_render import render_detection_overlay
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    # .../pingpong_controller/tools -> parent of pingpong_controller is the
    # package root that must be on sys.path for `import pingpong_controller`.
    _PKG_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
    sys.path.insert(0, _PKG_ROOT)
    from pingpong_controller.ball_detector import PingPongBallDetector
    from pingpong_controller.calibration_loader import load_calibration
    from pingpong_controller.viz_render import render_detection_overlay


def _module_dir() -> str:
    """Return the pingpong_controller source dir (…/pingpong_controller)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..'))


def _auto_calibration(width: int, height: int) -> str:
    """Find a calibration JSON matching the video resolution, like the node."""
    calib_dir = os.path.join(_module_dir(), 'outputs', 'vision_calib')
    candidates = sorted(set(glob.glob(os.path.join(
        calib_dir, 'eye_hand_result_*_undistorted.json'))))
    for candidate in candidates:
        try:
            calib = load_calibration(candidate)
        except Exception:
            continue
        if calib.image_width == width and calib.image_height == height:
            return candidate
    fallback = os.path.join(
        calib_dir, f'eye_hand_result_{width}_undistorted.json')
    return fallback


def _load_timestamps(csv_path: str) -> dict:
    """Load the node's per-frame timestamp sidecar into {frame_idx: t_sec}.

    Returns an empty dict on any problem (the caller falls back to fps timing).
    """
    table = {}
    try:
        with open(csv_path, 'r') as fh:
            header = fh.readline()  # 'frame_idx,t_capture_sec'
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                try:
                    table[int(parts[0])] = float(parts[1])
                except ValueError:
                    continue
    except OSError:
        return {}
    return table


def _load_depth_frame(depth_dir: str, frame_idx: int):
    """Load aligned depth frame ``depth_<idx>.npy`` for a 0-based frame index.

    The recorder writes one depth file per RGB frame keyed by the same index.
    Returns the 2D depth array, or None when the file is missing / empty /
    unreadable (the detector then falls back to monocular for that frame).
    """
    if not depth_dir:
        return None
    path = os.path.join(depth_dir, f'depth_{frame_idx:06d}.npy')
    if not os.path.isfile(path):
        return None
    try:
        depth = np.load(path)
    except Exception:  # noqa: BLE001
        return None
    if depth is None or depth.size == 0:
        return None
    return depth


def _write_detection_row(fh, frame_idx: int, t_sec: float,
                         observation: dict) -> None:
    """Append one detection row to the offline log.

    Mirrors the node estimator-log raw_* columns so
    validate_estimator_replay.py can consume either source. A miss is logged
    with detected=0 and NaN positions; the depth_method column records whether
    hardware depth ('sensor') or monocular geometry ('geometry') was used so a
    later pass can split the two.
    """
    detected = bool(observation.get('detected', False))
    raw = observation.get('raw_measurement') or {}
    filt_rej = bool(raw.get('filter_rejected', False))
    pos = observation.get('position') or [float('nan')] * 3
    vel = observation.get('velocity') or [float('nan')] * 3

    def _f(v):
        try:
            return f'{float(v):.3f}'
        except (TypeError, ValueError):
            return 'nan'

    radius = raw.get('radius_px', '')
    depth_mm = raw.get('depth_mm', '')
    depth_method = raw.get('depth_method', '')
    fh.write(
        f'{frame_idx},{t_sec:.6f},{int(detected)},{int(filt_rej)},'
        f'{_f(pos[0])},{_f(pos[1])},{_f(pos[2])},'
        f'{_f(vel[0])},{_f(vel[1])},{_f(vel[2])},'
        f'{_f(radius) if radius != "" else ""},'
        f'{_f(depth_mm) if depth_mm != "" else ""},'
        f'{depth_method}\n')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', '-i', required=True,
                   help='Path to the raw recorded video.')
    p.add_argument('--output', '-o', default='',
                   help='Output annotated video path. Default: '
                        '<input>_annotated.mp4 next to the input.')
    p.add_argument('--calibration', default='auto',
                   help='Calibration JSON path, or "auto" to match by '
                        'resolution (default).')
    p.add_argument('--yolo-model', default='',
                   help='YOLO model path. Default: <module>/models/best.pt')
    p.add_argument('--target-class', default='pingpong ball',
                   help='Detector target class (default matches node).')
    p.add_argument('--conf-threshold', type=float, default=0.5)
    p.add_argument('--infer-imgsz', type=int, default=640,
                   help='YOLO inference image size. Pass the SAME value you '
                        'ran the node with (the example used 480).')
    p.add_argument('--device', default='0',
                   help="Inference device: '0' for GPU (node default), 'cpu'.")
    p.add_argument('--half', action='store_true')
    p.add_argument('--camera-tilt-deg', type=float, default=40.0)
    p.add_argument('--no-kalman', action='store_true',
                   help='Disable Kalman smoothing (default: enabled).')
    p.add_argument('--no-ellipse-fit', action='store_true',
                   help='Disable ellipse-fit radius (use minEnclosingCircle). '
                        'Default: ellipse fit ON (matches node default).')
    p.add_argument('--temporal-tracker', action='store_true',
                   help='Enable the multi-frame temporal tracker (motion gate '
                        '+ miss bridging). Pass this to match a node run that '
                        'had enable_temporal_tracker=True.')
    p.add_argument('--tracker-max-miss-bridge', type=int, default=2,
                   help='Temporal tracker: max consecutive missed frames to '
                        'bridge (default 2).')
    p.add_argument('--tracker-gate-px', type=float, default=120.0,
                   help='Temporal tracker: motion-consistency gate in px '
                        '(default 120).')
    p.add_argument('--hsv-lower', default='',
                   help='Orange HSV lower bound "H,S,V" (e.g. 8,80,80). '
                        'Empty = detector default. Match the node run.')
    p.add_argument('--hsv-upper', default='',
                   help='Orange HSV upper bound "H,S,V" (e.g. 22,255,255). '
                        'Empty = detector default. Match the node run.')
    p.add_argument('--color-match-threshold', type=float, default=0.3,
                   help='Min orange-pixel ratio in mask (default 0.3).')
    p.add_argument('--color-circularity-min', type=float, default=0.4,
                   help='Min circularity for color fallback (default 0.4).')
    p.add_argument('--output-frame', default='base_link',
                   choices=['base_link', 'camera'],
                   help='Coordinate frame for 3D position (default base_link, '
                        'requires extrinsics in the calibration JSON).')
    p.add_argument('--fps', type=float, default=0.0,
                   help='Override timing fps used for detection dt and output. '
                        '<=0 uses the input video fps.')
    p.add_argument('--timestamps-csv', default='',
                   help='Per-frame timestamp sidecar (frame_idx,t_capture_sec) '
                        'saved by the node alongside the raw video. When given, '
                        'detection dt uses these TRUE capture times instead of '
                        'the constant-fps assumption. Default: auto-detect '
                        '<input>_timestamps.csv next to the input.')
    p.add_argument('--depth-dir', default='',
                   help='Directory of aligned 16-bit depth frames '
                        '(depth_<idx>.npy) saved by the node. Enables hardware '
                        'depth replay. Default: auto-detect <input>_depth/ '
                        'next to the input when --use-depth-sensor is set.')
    p.add_argument('--use-depth-sensor', action='store_true',
                   help='Feed recorded hardware depth into the detector '
                        '(matches a node run with use_depth_sensor=True). '
                        'Requires a depth dir; falls back to monocular per '
                        'frame when a depth frame is missing/invalid.')
    p.add_argument('--depth-scale', type=float, default=0.001,
                   help='Depth scale (m per z16 count) the node logged at '
                        'startup ("depth_scale=... m/count"). Default 0.001.')
    p.add_argument('--depth-neighborhood-px', type=int, default=2,
                   help='Depth sampling half-window when no ball mask is '
                        'available (matches node depth_neighborhood_px).')
    p.add_argument('--detection-log', default='',
                   help='Optional CSV of per-frame detection results '
                        '(frame_idx, t, x/y/z mm, vx/vy/vz, depth_mm, '
                        'depth_method, detected). Default: '
                        '<output>_detections.csv. Pass "none" to skip.')
    p.add_argument('--show', action='store_true',
                   help='Also display frames while processing (slower).')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f'ERROR: input video not found: {args.input}', file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f'ERROR: cannot open input video: {args.input}', file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = args.fps if args.fps > 0.0 else (video_fps if video_fps > 1.0 else 60.0)

    print(f'Input : {args.input} ({width}x{height}, '
          f'{video_fps:.2f} fps, {total} frames)')

    root_in, _ = os.path.splitext(args.input)

    # Per-frame timestamps: explicit, else auto-detect the node's sidecar.
    ts_path = args.timestamps_csv
    if not ts_path:
        cand = root_in + '_timestamps.csv'
        ts_path = cand if os.path.isfile(cand) else ''
    timestamps = _load_timestamps(ts_path) if ts_path else {}
    if timestamps:
        print(f'Times : {ts_path} ({len(timestamps)} stamps; using TRUE '
              'capture times for detection dt)')
    else:
        print(f'Times : none (constant {fps:.1f} fps assumption)')

    # Hardware depth: explicit dir, else auto-detect when --use-depth-sensor.
    depth_dir = args.depth_dir
    if not depth_dir and args.use_depth_sensor:
        cand = root_in + '_depth'
        depth_dir = cand if os.path.isdir(cand) else ''
    if args.use_depth_sensor:
        if depth_dir and os.path.isdir(depth_dir):
            print(f'Depth : {depth_dir} (hardware depth ON, scale='
                  f'{args.depth_scale} m/count)')
        else:
            print('WARNING: --use-depth-sensor set but no depth dir found; '
                  'replaying monocular only.', file=sys.stderr)
            depth_dir = ''
    else:
        depth_dir = ''

    calib_path = args.calibration
    if calib_path.lower() in ('auto', ''):
        calib_path = _auto_calibration(width, height)
    if not os.path.isfile(calib_path):
        print(f'ERROR: calibration JSON not found: {calib_path}',
              file=sys.stderr)
        return 1
    calibration = load_calibration(calib_path)
    print(f'Calib : {calib_path}')
    if (calibration.image_width != width
            or calibration.image_height != height):
        print(f'WARNING: calibration resolution '
              f'{calibration.image_width}x{calibration.image_height} '
              f'differs from video {width}x{height}; detection may be off.')

    yolo_model = args.yolo_model or os.path.join(
        _module_dir(), 'models', 'best.pt')
    if not os.path.isfile(yolo_model):
        print(f'ERROR: YOLO model not found: {yolo_model}', file=sys.stderr)
        return 1
    print(f'YOLO  : {yolo_model}')

    out_path = args.output
    if not out_path:
        root, _ = os.path.splitext(args.input)
        out_path = f'{root}_annotated.mp4'
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Build the detector exactly like the node's calibration path.
    def _parse_hsv(s):
        s = s.strip()
        if not s:
            return None
        parts = [int(v) for v in s.split(',')]
        return parts if len(parts) == 3 else None

    hsv_lower = _parse_hsv(args.hsv_lower)
    hsv_upper = _parse_hsv(args.hsv_upper)
    detector = PingPongBallDetector(
        yolo_model_path=yolo_model,
        ball_diameter_mm=40.0,
        width=width,
        height=height,
        fps=int(round(fps)),
        conf_threshold=args.conf_threshold,
        target_class=args.target_class,
        camera_tilt_deg=args.camera_tilt_deg,
        use_kalman=not args.no_kalman,
        infer_imgsz=args.infer_imgsz,
        device=args.device,
        half=args.half,
        use_undistort=True,
        print_every=999999,
        profile_every=0,
        use_camera=False,
        calibration_data=calibration,
        output_frame=args.output_frame,
        enable_ellipse_fit=not args.no_ellipse_fit,
        enable_temporal_tracker=args.temporal_tracker,
        tracker_max_miss_bridge=args.tracker_max_miss_bridge,
        tracker_gate_px=args.tracker_gate_px,
        hsv_lower=hsv_lower,
        hsv_upper=hsv_upper,
        color_match_threshold=args.color_match_threshold,
        color_circularity_min=args.color_circularity_min,
        use_depth_sensor=args.use_depth_sensor,
        depth_scale=args.depth_scale,
        depth_neighborhood_px=args.depth_neighborhood_px,
    )
    output_frame = getattr(detector, 'output_frame', args.output_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f'ERROR: cannot open output writer: {out_path}', file=sys.stderr)
        cap.release()
        return 1
    print(f'Output: {out_path} ({width}x{height}@{fps:.1f} fps)')

    # Per-frame detection log (consumed by validate_estimator_replay.py to
    # compare hardware-depth vs monocular). Columns mirror the node's
    # estimator-log raw_* fields so the validator can read either source.
    det_log = None
    det_log_path = args.detection_log
    if det_log_path.lower() != 'none':
        if not det_log_path:
            det_root, _ = os.path.splitext(out_path)
            det_log_path = det_root + '_detections.csv'
        try:
            det_log = open(det_log_path, 'w', buffering=1)
            det_log.write(
                'frame_idx,t_sec,detected,filter_rejected,'
                'x_mm,y_mm,z_mm,vx_mm_s,vy_mm_s,vz_mm_s,'
                'radius_px,depth_mm,depth_method\n')
            print(f'DetLog: {det_log_path}')
        except OSError as exc:
            print(f'WARNING: cannot open detection log {det_log_path}: {exc}',
                  file=sys.stderr)
            det_log = None
    print('Processing...')

    frame_count = 0
    detection_count = 0
    depth_used_count = 0
    try:
        while True:
            ok, raw_bgr = cap.read()
            if not ok:
                break
            frame_idx = frame_count  # 0-based, matches recorder file index
            frame_count += 1
            # Prefer the TRUE capture time from the sidecar; fall back to the
            # constant-fps clock when no timestamp exists for this frame.
            observation_time = timestamps.get(frame_idx)
            if observation_time is None:
                observation_time = frame_count / fps

            depth_image = (_load_depth_frame(depth_dir, frame_idx)
                           if depth_dir else None)
            observation = detector.get_observation(
                raw_bgr, observation_time=observation_time,
                depth_image=depth_image)
            if observation.get('detected', False):
                detection_count += 1

            if det_log is not None:
                _write_detection_row(
                    det_log, frame_idx, observation_time, observation)
                raw = observation.get('raw_measurement') or {}
                if raw.get('depth_method') == 'sensor':
                    depth_used_count += 1

            # Remap the raw frame to undistorted space for rendering, mirroring
            # the live viz worker so overlay coordinates line up.
            if (getattr(detector, 'use_undistort', False)
                    and getattr(detector, 'map1', None) is not None
                    and getattr(detector, 'map2', None) is not None):
                try:
                    viz_bgr = cv2.remap(
                        raw_bgr, detector.map1, detector.map2,
                        cv2.INTER_LINEAR)
                except cv2.error:
                    viz_bgr = raw_bgr
            else:
                viz_bgr = raw_bgr

            annotated = render_detection_overlay(
                viz_bgr, observation,
                frame_count=frame_count,
                detection_count=detection_count,
                callback_hz=fps,
                output_frame=output_frame)
            if not annotated.flags['C_CONTIGUOUS']:
                annotated = np.ascontiguousarray(annotated)
            writer.write(annotated)

            if args.show:
                cv2.imshow('annotate_raw_video', annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    print('Interrupted by user.')
                    break

            if total > 0 and frame_count % 30 == 0:
                pct = 100.0 * frame_count / total
                print(f'  {frame_count}/{total} ({pct:5.1f}%) '
                      f'detected={detection_count}', end='\r', flush=True)
    finally:
        cap.release()
        writer.release()
        if det_log is not None:
            det_log.close()
        if args.show:
            cv2.destroyAllWindows()

    rate = 100.0 * detection_count / frame_count if frame_count else 0.0
    print()
    print(f'Done. {frame_count} frames, {detection_count} detected '
          f'({rate:.1f}%).')
    if depth_dir:
        dpct = 100.0 * depth_used_count / detection_count if detection_count else 0.0
        print(f'Hardware depth used on {depth_used_count}/{detection_count} '
              f'detections ({dpct:.1f}%); rest fell back to monocular.')
    print(f'Saved: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
