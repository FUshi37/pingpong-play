#!/usr/bin/env python3
"""
Pipeline Profiler (Robot Side) -- measures the D+E latency the flash test can't.

The flash test measures  A + B + C(+screen):
    A  camera exposure + sensor readout      (+ LCD flip->photons, an upper bound)
    B  driver/USB + ROS compressed transport (image_stamp -> callback)
    C  cv2.imdecode

It does NOT touch YOLO or the geometry/filtering that turn an image into a ball
state. That is what this node measures, on the SAME ROS image topic and with the
SAME PingPongBallDetector you run while actually playing (we import the real
class from pingpong_controller and build it the way pingpong_node does), so the
numbers are the production D+E:
    C  decode            cv2.imdecode (re-measured here too, for a clean total)
    D  yolo              detector.last_yolo_ms (YOLO forward pass)
    E  postprocess       geometry/depth/circle-fit + Kalman/OneEuro + transform
                         = get_observation total - yolo

Full perceived latency (photons -> ball state) is then:
    flash end-to-end (A+B+C+screen)  +  (D + E)
Because the flash A+B+C already includes a decode and a tiny brightness check,
the honest way to combine is:
    photons->state  ~=  flash_e2e  +  yolo_ms  +  postprocess_ms
(the flash C decode ~1.5ms stands in for this node's C; we report C here only so
you can see it matches.)

Usage (point it at the same live topic, or `ros2 bag play` a recording):
    ros2 run pingpong_latency_test pipeline_profiler --ros-args \
        -p image_topic:=/camera/camera/color/image_raw/compressed \
        -p camera_info_topic:=/camera/camera/color/camera_info \
        -p yolo_model_path:=/home/spirit-ai/pingpong_moz1/src/pingpong_moz1/pingpong_controller/models/best.pt \
        -p infer_imgsz:=640 -p device:=0 -p use_kalman:=true \
        -p out_csv:=pipeline_latency.csv

Wave a ball (or anything the model detects) in view so YOLO actually finds an
object -- a detected frame exercises the full E path (geometry + filter); a
missed frame only times YOLO + the early-out. Stats are split by detected/missed.
"""

import os
import csv
import time

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CompressedImage, Image, CameraInfo

from pingpong_controller.ball_detector import PingPongBallDetector

try:
    from pingpong_controller.calibration_loader import load_calibration
except Exception:
    load_calibration = None


class PipelineProfiler(Node):
    def __init__(self):
        super().__init__('pipeline_profiler')

        # Topics
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')

        # Detector params -- mirror pingpong_node defaults so D+E matches play.
        self.declare_parameter('yolo_model_path', '')
        self.declare_parameter('target_class', 'pingpong ball')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('infer_imgsz', 640)
        # device may be '0' (GPU index) or 'cpu'; CLI `device:=0` parses as int,
        # so accept either type and stringify below.
        self.declare_parameter(
            'device', '0',
            ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('half', False)
        self.declare_parameter('use_kalman', False)
        self.declare_parameter('camera_tilt_deg', 40.0)
        self.declare_parameter('use_undistort', True)
        # Optional extrinsics JSON -> base_link output (matches play if you use
        # base_link). Blank = camera frame; the transform step is ~microseconds
        # so it barely changes E, but we support it for faithfulness.
        self.declare_parameter('calibration_json_path', '')

        # Profiling output / control
        self.declare_parameter('out_csv', 'pipeline_latency.csv')
        self.declare_parameter('warmup_frames', 10)   # skip first N (CUDA/JIT warmup)
        self.declare_parameter('max_frames', 0)        # 0 = run until Ctrl+C
        self.declare_parameter('report_every', 30)

        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.yolo_model_path = self.get_parameter('yolo_model_path').value
        self.target_class = self.get_parameter('target_class').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.infer_imgsz = self.get_parameter('infer_imgsz').value
        self.device = str(self.get_parameter('device').value)
        self.half = self.get_parameter('half').value
        self.use_kalman = self.get_parameter('use_kalman').value
        self.camera_tilt_deg = self.get_parameter('camera_tilt_deg').value
        self.use_undistort = self.get_parameter('use_undistort').value
        self.calibration_json_path = self.get_parameter('calibration_json_path').value
        self.out_csv = self.get_parameter('out_csv').value
        self.warmup_frames = int(self.get_parameter('warmup_frames').value)
        self.max_frames = int(self.get_parameter('max_frames').value)
        self.report_every = int(self.get_parameter('report_every').value)

        if not self.yolo_model_path or not os.path.exists(self.yolo_model_path):
            self.get_logger().error(
                f'yolo_model_path missing or not found: "{self.yolo_model_path}". '
                'Pass -p yolo_model_path:=/abs/path/to/best.pt')
            raise SystemExit(2)

        self.detector = None
        self.camera_info_received = False
        self.frame_count = 0
        self.rows = []   # per-frame timing rows

        self.is_compressed = 'compressed' in self.image_topic.lower()

        # camera_info first (needed to build the detector with real intrinsics),
        # exactly like pingpong_node's ROS-topic path.
        self.info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_callback, 10)

        msg_type = CompressedImage if self.is_compressed else Image
        self.image_sub = self.create_subscription(
            msg_type, self.image_topic, self._image_callback, 10)

        self.get_logger().info('Pipeline Profiler started (record D+E latency)')
        self.get_logger().info(f'  Image topic: {self.image_topic} '
                               f'({"compressed" if self.is_compressed else "raw"})')
        self.get_logger().info(f'  Model: {self.yolo_model_path}  imgsz={self.infer_imgsz} '
                               f'device={self.device} half={self.half} use_kalman={self.use_kalman}')
        self.get_logger().info(f'  Waiting for camera_info on {self.camera_info_topic} ...')

    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        dist_coeffs = list(msg.d) if len(msg.d) > 0 else None

        output_frame = 'camera'
        calibration = None
        if self.calibration_json_path and load_calibration is not None \
                and os.path.exists(self.calibration_json_path):
            try:
                calibration = load_calibration(self.calibration_json_path)
                if (calibration.image_width == msg.width
                        and calibration.image_height == msg.height):
                    output_frame = 'base_link'
                else:
                    self.get_logger().warn(
                        'Calibration resolution mismatch; staying in camera frame')
                    calibration = None
            except Exception as exc:
                self.get_logger().warn(f'Failed to load calibration: {exc}')
                calibration = None

        # Build the detector EXACTLY like pingpong_node._camera_info_callback.
        self.detector = PingPongBallDetector(
            yolo_model_path=self.yolo_model_path,
            ball_diameter_mm=40.0,
            width=msg.width, height=msg.height, fps=60,
            conf_threshold=self.conf_threshold,
            target_class=self.target_class,
            camera_tilt_deg=self.camera_tilt_deg,
            use_kalman=self.use_kalman,
            infer_imgsz=self.infer_imgsz,
            device=self.device, half=self.half,
            use_undistort=self.use_undistort,
            print_every=999999, profile_every=0, use_camera=False,
            camera_fx=fx, camera_fy=fy, camera_cx=cx, camera_cy=cy,
            camera_dist_coeffs=dist_coeffs,
            output_frame=output_frame,
        )
        if calibration is not None:
            self.detector.calibration_data = calibration
            self.detector.T_base_camera = calibration.T_base_camera
            self.detector.R_base_camera = self.detector.T_base_camera[:3, :3]
            self.detector.output_frame = 'base_link'

        self.camera_info_received = True
        self.get_logger().info(
            f'Detector ready: {msg.width}x{msg.height}, output={output_frame}. '
            f'Wave a ball in view. Warming up {self.warmup_frames} frames...')

    def _image_callback(self, msg):
        if not self.camera_info_received:
            return

        # C: decode (same call pingpong_node makes)
        t_dec0 = time.perf_counter()
        if self.is_compressed:
            np_arr = np.frombuffer(msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            image = np.frombuffer(msg.data, np.uint8).reshape(
                msg.height, msg.width, -1)
        if image is None:
            return
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0

        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9

        # D+E: the whole image->ball-state path, identical to play.
        t_obs0 = time.perf_counter()
        observation = self.detector.get_observation(image, observation_time=stamp_sec)
        obs_ms = (time.perf_counter() - t_obs0) * 1000.0

        # D alone, straight from the detector's own stopwatch.
        yolo_ms = float(getattr(self.detector, 'last_yolo_ms', 0.0))
        # E = everything in get_observation except the YOLO forward pass.
        postprocess_ms = max(obs_ms - yolo_ms, 0.0)
        detected = bool(observation.get('detected', False))

        self.frame_count += 1
        if self.frame_count <= self.warmup_frames:
            return

        self.rows.append({
            'frame': self.frame_count,
            'detected': int(detected),
            'decode_ms': decode_ms,
            'yolo_ms': yolo_ms,
            'postprocess_ms': postprocess_ms,
            'de_total_ms': yolo_ms + postprocess_ms,
        })

        if self.report_every and len(self.rows) % self.report_every == 0:
            self._report(running=True)

        if self.max_frames and len(self.rows) >= self.max_frames:
            self.get_logger().info(f'Reached max_frames={self.max_frames}, stopping.')
            raise KeyboardInterrupt

    def _report(self, running=False):
        if not self.rows:
            self.get_logger().warn('No frames recorded yet.')
            return
        det = [r for r in self.rows if r['detected']]
        tag = '(running)' if running else '(final)'

        def stat(name, vals):
            v = np.array(vals, dtype=float)
            return (f'    {name:<16} mean={v.mean():6.2f} median={np.median(v):6.2f} '
                    f'std={v.std():5.2f} min={v.min():6.2f} max={v.max():6.2f} '
                    f'p95={np.percentile(v, 95):6.2f} ms')

        self.get_logger().info(
            f'D+E latency {tag}  frames={len(self.rows)}  detected={len(det)}')
        # Detected frames exercise the full E path -- this is the number to use.
        src = det if det else self.rows
        if not det:
            self.get_logger().warn('  (no detected frames yet; showing all frames -- '
                                   'E path not fully exercised. Put a ball in view.)')
        self.get_logger().info(stat('decode (C)', [r['decode_ms'] for r in src]))
        self.get_logger().info(stat('yolo (D)', [r['yolo_ms'] for r in src]))
        self.get_logger().info(stat('postproc (E)', [r['postprocess_ms'] for r in src]))
        self.get_logger().info(stat('D+E total', [r['de_total_ms'] for r in src]))

    def save_csv(self):
        if not self.rows:
            return
        try:
            with open(self.out_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[
                    'frame', 'detected', 'decode_ms', 'yolo_ms',
                    'postprocess_ms', 'de_total_ms'])
                w.writeheader()
                w.writerows(self.rows)
            self.get_logger().info(f'Saved {len(self.rows)} rows to {self.out_csv}')
        except Exception as e:
            self.get_logger().error(f'Failed to save CSV: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = PipelineProfiler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._report(running=False)
        node.save_csv()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
