#!/usr/bin/env python3
"""Offline annotation of a raw recording using the HYBRID (RGB YOLO + IR) detector.

Counterpart to annotate_raw_video.py, but for the gray reflective ball detected
by HybridBallDetector. That detector localizes on the INFRARED stream (RGB YOLO
only supplies a bbox ROI) and reads depth from the raw depth frame, so a faithful
offline replay needs three recorded streams plus the camera intrinsics/extrinsics:

  <stem>.mp4                 RGB (color) frames
  <stem>_ir/ir_<idx>.npy     IR (infrared-1) frames, uint8
  <stem>_depth/depth_<idx>.npy   raw (IR-aligned) depth frames, uint16
  <stem>_camera_params.json  IR + RGB rs.intrinsics, IR->RGB rs.extrinsics, depth_scale
  <stem>_timestamps.csv      per-frame true capture time (optional)

Record all of these with the node in hybrid mode:
  save_raw_video:=true save_ir_frames:=true save_depth_frames:=true
  use_hybrid_detector:=true use_depth_sensor:=true

Pipeline per frame (mirrors PingPongControllerNode._hybrid_reader_loop):
  1. read RGB[idx] (mp4), IR[idx] (.npy), depth[idx] (.npy)
  2. detector.get_observation_offline(rgb, ir, depth, observation_time)
     -> YOLO ROI -> IR localization -> depth select -> IR 3D -> RGB 3D -> base_link
  3. detector._render_visualization(rgb, ir, observation) -> dual RGB|IR view
  4. write the annotated frame to the output video + a per-frame detection CSV.

Example:
  python3 annotate_raw_video_hybrid.py \
      -i outputs/videos/pingpong_raw_20260702_120000.mp4 \
      --output-frame base_link
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# Allow running both as an installed module and as a loose script.
try:
    from pingpong_controller.hybrid_ball_detector import HybridBallDetector
    from pingpong_controller.calibration_loader import load_calibration
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PKG_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
    sys.path.insert(0, _PKG_ROOT)
    from pingpong_controller.hybrid_ball_detector import HybridBallDetector
    from pingpong_controller.calibration_loader import load_calibration


def _module_dir() -> str:
    """Return the pingpong_controller source dir (…/pingpong_controller)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..'))


def _auto_calibration(width: int, height: int) -> str:
    """Find a hand-eye calibration JSON matching the video resolution."""
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
    return os.path.join(
        calib_dir, f'eye_hand_result_{width}_undistorted.json')


def _load_timestamps(csv_path: str) -> dict:
    """Load {frame_idx: t_sec} sidecar; empty dict on any problem."""
    table = {}
    try:
        with open(csv_path, 'r') as fh:
            fh.readline()  # header: frame_idx,t_capture_sec
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


def _load_npy_frame(directory: str, prefix: str, frame_idx: int):
    """Load ``<prefix>_<idx>.npy`` for a 0-based index; None if missing/empty."""
    if not directory:
        return None
    path = os.path.join(directory, f'{prefix}_{frame_idx:06d}.npy')
    if not os.path.isfile(path):
        return None
    try:
        arr = np.load(path)
    except Exception:  # noqa: BLE001
        return None
    if arr is None or arr.size == 0:
        return None
    return arr


def _write_detection_row(fh, frame_idx: int, t_sec: float,
                         observation: dict) -> None:
    """Append one detection row to the offline hybrid detection log.

    Columns cover both the hybrid-specific fields (IR pixel, IR-on-RGB pixel,
    depth source, detection mode) and the estimator-style x/y/z + vx/vy/vz so a
    later pass can consume this like the node's estimator log.
    """
    detected = bool(observation.get('detected', False))
    pos = observation.get('position') or [float('nan')] * 3
    vel = observation.get('velocity') or [float('nan')] * 3
    raw = observation.get('raw_measurement') or {}

    def _f(v):
        try:
            return f'{float(v):.3f}'
        except (TypeError, ValueError):
            return 'nan'

    def _s(key):
        v = raw.get(key)
        return '' if v is None else _f(v)

    fh.write(
        f'{frame_idx},{t_sec:.6f},{int(detected)},'
        f'{_f(pos[0])},{_f(pos[1])},{_f(pos[2])},'
        f'{_f(vel[0])},{_f(vel[1])},{_f(vel[2])},'
        f'{_s("cx")},{_s("cy")},{_s("cx_rgb")},{_s("cy_rgb")},'
        f'{_s("radius")},{_s("depth_mm")},'
        f'{raw.get("depth_method", "")},{raw.get("detection_method", "")}\n')


def _resolve_detector_config(camera_params: dict, args) -> dict:
    """Resolve the hybrid detector config for a faithful offline replay.

    The record->annotate path must reproduce the LIVE detector byte-for-byte,
    so the config RECORDED in camera_params JSON at capture time is the source
    of truth. Precedence per field: explicit CLI arg (not None) > recorded
    value > historical hard default (for old recordings that predate the
    config block). Returns a flat dict of HybridBallDetector constructor kwargs.
    """
    rec_ir = camera_params.get('ir_params') or {}

    def pick(cli_val, recorded_val, default):
        if cli_val is not None:
            return cli_val
        if recorded_val is not None:
            return recorded_val
        return default

    return {
        'ball_diameter_mm': pick(
            args.ball_diameter_mm,
            camera_params.get('ball_diameter_mm'), 40.0),
        'yolo_interval': pick(
            args.yolo_interval, camera_params.get('yolo_interval'), 1),
        'yolo_conf': pick(
            args.yolo_conf, camera_params.get('yolo_conf'), 0.25),
        'target_class': pick(
            args.target_class, camera_params.get('target_class'), 'Class 0'),
        # IR morphology / thresholds (recorded under ir_params).
        'ir_threshold': pick(
            args.ir_threshold, rec_ir.get('threshold'), 170),
        'use_adaptive': pick(
            args.use_adaptive, rec_ir.get('use_adaptive'), True),
        'dilate_iterations': pick(
            args.dilate_iterations, rec_ir.get('dilate_iterations'), 4),
        'close_kernel_size': pick(
            args.close_kernel_size, rec_ir.get('close_kernel_size'), 11),
        'denoise_kernel_size': pick(
            args.denoise_kernel_size, rec_ir.get('denoise_kernel_size'), 3),
        'min_radius': pick(
            args.min_radius, rec_ir.get('min_radius'), 15),
        'max_radius': pick(
            args.max_radius, rec_ir.get('max_radius'), 60),
        'circularity_min': pick(
            args.circularity_min, rec_ir.get('circularity_min'), 0.7),
        'brightness_percentile': pick(
            args.brightness_percentile,
            camera_params.get('brightness_percentile'), 25),
        'roi_margin': pick(
            args.roi_margin, camera_params.get('roi_margin'), 100),
        'depth_neighborhood_px': pick(
            args.depth_neighborhood_px,
            camera_params.get('depth_neighborhood_px'), 2),
        'depth_correction_model_path': pick(
            args.depth_correction_model_path,
            camera_params.get('depth_correction_model_path'),
            'depth_correction_model.pkl'),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', '-i', required=True,
                   help='Path to the raw recorded RGB video (<stem>.mp4).')
    p.add_argument('--output', '-o', default='',
                   help='Output annotated video path. Default: '
                        '<input>_hybrid_annotated.mp4 next to the input.')
    # --- sidecar streams (auto-detected from <stem> when omitted) ---
    p.add_argument('--ir-dir', default='',
                   help='IR frame dir (ir_<idx>.npy). Default: <input>_ir/')
    p.add_argument('--depth-dir', default='',
                   help='Depth frame dir (depth_<idx>.npy). '
                        'Default: <input>_depth/')
    p.add_argument('--camera-params', default='',
                   help='camera_params JSON (IR/RGB intrinsics + IR->RGB '
                        'extrinsics + depth_scale). Default: '
                        '<input>_camera_params.json')
    p.add_argument('--timestamps-csv', default='',
                   help='Per-frame timestamp sidecar. Default: '
                        '<input>_timestamps.csv')
    p.add_argument('--calibration', default='auto',
                   help='Hand-eye calibration JSON (for base_link output), or '
                        '"auto" to match by resolution (default).')
    p.add_argument('--yolo-model', default='',
                   help='YOLO model path. Default: <module>/models/best.pt')
    # --- hybrid detector params ---
    # These default to None: when omitted, the value RECORDED in
    # camera_params JSON at capture time is used, so offline annotation
    # reproduces the LIVE detector config exactly (the record->annotate and
    # live-juggle paths must process identically). Pass a flag to override the
    # recording for experiments. Old recordings without the config block fall
    # back to the historical hard defaults shown in parentheses.
    p.add_argument('--target-class', default=None,
                   help='YOLO target class name (recorded; else "Class 0").')
    p.add_argument('--yolo-interval', type=int, default=None,
                   help='Run YOLO every N frames (recorded; else 1).')
    p.add_argument('--yolo-conf', type=float, default=None,
                   help='YOLO confidence threshold (recorded; else 0.25).')
    p.add_argument('--ir-params-json', default='',
                   help='IR params JSON; overrides BOTH the recorded config '
                        'and the individual IR args (same file the node loads).')
    p.add_argument('--ir-threshold', type=int, default=None,
                   help='IR binary threshold (recorded; else 170).')
    p.add_argument('--use-adaptive', dest='use_adaptive',
                   action='store_true', default=None,
                   help='Force adaptive IR threshold on (else recorded).')
    p.add_argument('--no-adaptive', dest='use_adaptive', action='store_false',
                   help='Force adaptive IR threshold off (else recorded).')
    p.add_argument('--dilate-iterations', type=int, default=None,
                   help='IR mask dilate iterations (recorded; else 4).')
    p.add_argument('--close-kernel-size', type=int, default=None,
                   help='IR mask morphological-close kernel (recorded; else 11).')
    p.add_argument('--denoise-kernel-size', type=int, default=None,
                   help='IR mask denoise (open) kernel (recorded; else 3).')
    p.add_argument('--min-radius', type=int, default=None,
                   help='Min ball radius px (recorded; else 15).')
    p.add_argument('--max-radius', type=int, default=None,
                   help='Max ball radius px (recorded; else 60).')
    p.add_argument('--circularity-min', type=float, default=None,
                   help='Min IR blob circularity (recorded; else 0.7).')
    p.add_argument('--brightness-percentile', type=float, default=None,
                   help='Adaptive-threshold background percentile '
                        '(recorded; else 25).')
    p.add_argument('--roi-margin', type=int, default=None,
                   help='YOLO bbox ROI expansion px (recorded; else 100).')
    p.add_argument('--ball-diameter-mm', type=float, default=None,
                   help='Ball diameter mm (recorded; else 40.0).')
    p.add_argument('--depth-neighborhood-px', type=int, default=None,
                   help='Depth-map sampling neighborhood px (recorded; else 2).')
    p.add_argument('--depth-correction-model-path', default=None,
                   help='Monocular geometry-depth correction model .pkl '
                        '(recorded; else depth_correction_model.pkl).')
    # --- output / timing ---
    p.add_argument('--output-frame', default='base_link',
                   choices=['base_link', 'camera'],
                   help='Coordinate frame for 3D position (default base_link; '
                        'base_link needs the hand-eye calibration JSON).')
    p.add_argument('--fps', type=float, default=0.0,
                   help='Override timing fps for detection dt and output. '
                        '<=0 uses input video fps (else 60).')
    p.add_argument('--depth-scale', type=float, default=0.0,
                   help='Override depth scale (m/count). <=0 uses the value '
                        'from camera_params JSON.')
    p.add_argument('--detection-log', default='',
                   help='Per-frame detection CSV. Default: '
                        '<output>_detections.csv. "none" to skip.')
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

    # --- resolve sidecars (explicit args, else auto-detect from <stem>) ---
    ir_dir = args.ir_dir or (root_in + '_ir')
    if not os.path.isdir(ir_dir):
        print(f'ERROR: IR frame dir not found: {ir_dir}\n'
              '  The hybrid detector localizes on the IR stream; record with '
              'save_ir_frames:=true.', file=sys.stderr)
        cap.release()
        return 1
    print(f'IR    : {ir_dir}/')

    depth_dir = args.depth_dir or (root_in + '_depth')
    if os.path.isdir(depth_dir):
        print(f'Depth : {depth_dir}/ (hardware depth ON)')
    else:
        depth_dir = ''
        print('Depth : none (monocular geometry only)')

    cam_params_path = args.camera_params or (root_in + '_camera_params.json')
    if not os.path.isfile(cam_params_path):
        print(f'ERROR: camera_params JSON not found: {cam_params_path}\n'
              '  Needed to rebuild IR/RGB intrinsics + IR->RGB extrinsics; '
              'record with save_ir_frames:=true (writes it automatically).',
              file=sys.stderr)
        cap.release()
        return 1
    with open(cam_params_path, 'r') as fh:
        camera_params = json.load(fh)
    if args.depth_scale > 0.0:
        camera_params['depth_scale'] = args.depth_scale
    print(f'Params: {cam_params_path} '
          f'(depth_scale={camera_params.get("depth_scale")} m/count)')

    ts_path = args.timestamps_csv or (root_in + '_timestamps.csv')
    timestamps = _load_timestamps(ts_path) if os.path.isfile(ts_path) else {}
    if timestamps:
        print(f'Times : {ts_path} ({len(timestamps)} stamps; TRUE capture dt)')
    else:
        print(f'Times : none (constant {fps:.1f} fps assumption)')

    # --- hand-eye calibration for base_link output ---
    calibration = None
    if args.output_frame == 'base_link':
        calib_path = args.calibration
        if calib_path.lower() in ('auto', ''):
            calib_path = _auto_calibration(width, height)
        if not os.path.isfile(calib_path):
            print(f'ERROR: hand-eye calibration JSON not found: {calib_path}\n'
                  '  Required for --output-frame base_link (pass --calibration '
                  'or use --output-frame camera).', file=sys.stderr)
            cap.release()
            return 1
        calibration = load_calibration(calib_path)
        print(f'Calib : {calib_path}')

    yolo_model = args.yolo_model or os.path.join(
        _module_dir(), 'models', 'best.pt')
    if not os.path.isfile(yolo_model):
        print(f'ERROR: YOLO model not found: {yolo_model}', file=sys.stderr)
        cap.release()
        return 1
    print(f'YOLO  : {yolo_model}')

    out_path = args.output or (root_in + '_hybrid_annotated.mp4')
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ir_params_json = args.ir_params_json or None

    # Resolve the detector config: recorded camera_params (live config at
    # capture time) is the source of truth so offline detection reproduces the
    # live run; explicit CLI flags override. Log it so the effective config is
    # visible in the annotation output.
    cfg = _resolve_detector_config(camera_params, args)
    if camera_params.get('ir_params') is None:
        print('Config: camera_params JSON has NO recorded detector config '
              '(old recording); using CLI/hard defaults. Re-record to capture '
              'the live config for an exact match.')
    else:
        print('Config: using RECORDED live detector config from camera_params '
              '(CLI flags override).')
    print(f'  yolo: interval={cfg["yolo_interval"]} conf={cfg["yolo_conf"]} '
          f'class="{cfg["target_class"]}" | ball_diameter_mm='
          f'{cfg["ball_diameter_mm"]}')
    print(f'  ir  : thr={cfg["ir_threshold"]} adaptive={cfg["use_adaptive"]} '
          f'dilate={cfg["dilate_iterations"]} close={cfg["close_kernel_size"]} '
          f'denoise={cfg["denoise_kernel_size"]} '
          f'r=[{cfg["min_radius"]},{cfg["max_radius"]}] '
          f'circ>={cfg["circularity_min"]} '
          f'bright_pct={cfg["brightness_percentile"]} '
          f'roi_margin={cfg["roi_margin"]}')
    print(f'  depth: neighborhood={cfg["depth_neighborhood_px"]}px '
          f'correction_model={cfg["depth_correction_model_path"]}')

    # Build the hybrid detector in OFFLINE mode: no camera, intrinsics/extrinsics
    # rebuilt from camera_params so rs.rs2_* projection math matches the live run.
    detector = HybridBallDetector(
        yolo_model_path=yolo_model,
        ball_diameter_mm=cfg['ball_diameter_mm'],
        width=width,
        height=height,
        fps=int(round(fps)),
        yolo_interval=cfg['yolo_interval'],
        yolo_conf=cfg['yolo_conf'],
        target_class=cfg['target_class'],
        ir_threshold=cfg['ir_threshold'],
        use_adaptive=cfg['use_adaptive'],
        dilate_iterations=cfg['dilate_iterations'],
        close_kernel_size=cfg['close_kernel_size'],
        denoise_kernel_size=cfg['denoise_kernel_size'],
        min_radius=cfg['min_radius'],
        max_radius=cfg['max_radius'],
        circularity_min=cfg['circularity_min'],
        brightness_percentile=cfg['brightness_percentile'],
        roi_margin=cfg['roi_margin'],
        depth_scale=camera_params.get('depth_scale', 0.001),
        depth_neighborhood_px=cfg['depth_neighborhood_px'],
        calibration_data=calibration,
        output_frame=args.output_frame,
        print_every=999999,
        offline_camera_params=camera_params,
        depth_correction_model_path=cfg['depth_correction_model_path'],
    )

    # Load IR params from JSON exactly like the adapter does, if provided.
    if ir_params_json and os.path.isfile(ir_params_json):
        with open(ir_params_json, 'r') as fh:
            loaded = json.load(fh)
        for k in ('threshold', 'use_adaptive', 'dilate_iterations',
                  'close_kernel_size', 'denoise_kernel_size', 'min_radius',
                  'max_radius', 'circularity_min'):
            if k in loaded:
                detector.ir_params[k] = loaded[k]
        if 'ir_threshold' in loaded:
            detector.ir_params['threshold'] = loaded['ir_threshold']
        if 'brightness_percentile' in loaded:
            detector.brightness_percentile = float(loaded['brightness_percentile'])
        print(f'IRparm: {ir_params_json} -> {detector.ir_params}')

    # --- per-frame detection log ---
    det_log = None
    det_log_path = args.detection_log
    if det_log_path.lower() != 'none':
        if not det_log_path:
            det_root, _ = os.path.splitext(out_path)
            det_log_path = det_root + '_detections.csv'
        try:
            det_log = open(det_log_path, 'w', buffering=1)
            det_log.write(
                'frame_idx,t_sec,detected,'
                'x_mm,y_mm,z_mm,vx_mm_s,vy_mm_s,vz_mm_s,'
                'cx,cy,cx_rgb,cy_rgb,radius_px,depth_mm,'
                'depth_method,detection_method\n')
            print(f'DetLog: {det_log_path}')
        except OSError as exc:
            print(f'WARNING: cannot open detection log {det_log_path}: {exc}',
                  file=sys.stderr)
            det_log = None

    # The output frame is the detector's own dual RGB|IR composite; its size is
    # not known until the first render, so open the writer lazily.
    writer = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print('Processing...')

    frame_idx = 0
    frame_count = 0
    detection_count = 0
    sensor_used = 0
    missing_ir = 0
    try:
        while True:
            ok, rgb_bgr = cap.read()
            if not ok:
                break
            frame_count += 1

            observation_time = timestamps.get(frame_idx)
            if observation_time is None:
                observation_time = frame_count / fps

            ir_image = _load_npy_frame(ir_dir, 'ir', frame_idx)
            depth_image = (_load_npy_frame(depth_dir, 'depth', frame_idx)
                           if depth_dir else None)

            if ir_image is None:
                # No IR for this index: the hybrid core cannot run. Emit a miss
                # so frame indices stay aligned, and render the RGB with the IR
                # panel blanked (visualization needs a 2D IR image).
                missing_ir += 1
                observation = {'detected': False, 'position': [0, 0, 0],
                               'velocity': [0, 0, 0]}
                ir_for_viz = np.zeros((height, width), dtype=np.uint8)
            else:
                observation = detector.get_observation_offline(
                    rgb_bgr, ir_image, depth_image, observation_time)
                ir_for_viz = ir_image

            if observation.get('detected', False):
                detection_count += 1
                if (observation.get('raw_measurement') or {}).get(
                        'depth_method') == 'sensor':
                    sensor_used += 1

            if det_log is not None:
                _write_detection_row(
                    det_log, frame_idx, observation_time, observation)

            annotated = detector._render_visualization(
                rgb_bgr, ir_for_viz, observation)
            if not annotated.flags['C_CONTIGUOUS']:
                annotated = np.ascontiguousarray(annotated)

            if writer is None:
                oh, ow = annotated.shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, fps, (ow, oh))
                if not writer.isOpened():
                    print(f'ERROR: cannot open output writer: {out_path}',
                          file=sys.stderr)
                    return 1
                print(f'Output: {out_path} ({ow}x{oh}@{fps:.1f} fps, '
                      'dual RGB|IR view)')
            writer.write(annotated)

            if args.show:
                cv2.imshow('annotate_raw_video_hybrid', annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    print('Interrupted by user.')
                    break

            if total > 0 and frame_count % 30 == 0:
                pct = 100.0 * frame_count / total
                print(f'  {frame_count}/{total} ({pct:5.1f}%) '
                      f'detected={detection_count}', end='\r', flush=True)
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if det_log is not None:
            det_log.close()
        if args.show:
            cv2.destroyAllWindows()

    rate = 100.0 * detection_count / frame_count if frame_count else 0.0
    print()
    print(f'Done. {frame_count} frames, {detection_count} detected '
          f'({rate:.1f}%).')
    if detection_count:
        spct = 100.0 * sensor_used / detection_count
        print(f'Depth source: sensor on {sensor_used}/{detection_count} '
              f'detections ({spct:.1f}%); rest geometry.')
    if missing_ir:
        print(f'WARNING: {missing_ir} frames had no IR sidecar (skipped '
              'detection). Check the recording covered every frame.')
    print(f'Saved: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
