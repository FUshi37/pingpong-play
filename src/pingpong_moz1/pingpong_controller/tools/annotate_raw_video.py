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
    p.add_argument('--output-frame', default='base_link',
                   choices=['base_link', 'camera'],
                   help='Coordinate frame for 3D position (default base_link, '
                        'requires extrinsics in the calibration JSON).')
    p.add_argument('--fps', type=float, default=0.0,
                   help='Override timing fps used for detection dt and output. '
                        '<=0 uses the input video fps.')
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
    )
    output_frame = getattr(detector, 'output_frame', args.output_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f'ERROR: cannot open output writer: {out_path}', file=sys.stderr)
        cap.release()
        return 1
    print(f'Output: {out_path} ({width}x{height}@{fps:.1f} fps)')
    print('Processing...')

    frame_count = 0
    detection_count = 0
    try:
        while True:
            ok, raw_bgr = cap.read()
            if not ok:
                break
            frame_count += 1
            observation_time = frame_count / fps

            observation = detector.get_observation(
                raw_bgr, observation_time=observation_time)
            if observation.get('detected', False):
                detection_count += 1

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
        if args.show:
            cv2.destroyAllWindows()

    rate = 100.0 * detection_count / frame_count if frame_count else 0.0
    print()
    print(f'Done. {frame_count} frames, {detection_count} detected '
          f'({rate:.1f}%).')
    print(f'Saved: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
