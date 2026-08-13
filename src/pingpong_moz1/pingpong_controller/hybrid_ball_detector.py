#!/home/jacky/miniconda3/envs/gmr/bin/python
"""
RGB+IR混合检测器

结合RGB YOLO的语义理解和IR红外的高精度定位
- RGB YOLO: 粗定位，过滤非球物体（如反光衣服）
- IR检测: 在YOLO bbox内精确定位，提供高精度深度

优势：
1. 鲁棒性：YOLO过滤掉反光干扰
2. 高精度：IR提供精确位置和深度
3. 高帧率：IR可以60-90Hz，YOLO可降频到20Hz
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import time
import os
import math
from collections import deque
from typing import Optional, Dict, Any, Tuple
from ultralytics import YOLO

from pingpong_controller.calibration_loader import CalibrationData
from pingpong_controller.ir_depth_kalman_filter import IRDepthKalmanFilter
from pingpong_controller.pixel_kalman_filter import PixelKalmanFilter
from pingpong_controller.ir_center_estimator import refine_ir_ball_center


def _rs_extrinsics_rotation_matrix(rotation):
    """Convert librealsense's column-major flat rotation to a NumPy matrix."""
    return np.asarray(rotation, dtype=np.float64).reshape(3, 3).T


class HybridBallDetector:
    """
    RGB+IR混合检测器

    工作流程：
    1. RGB YOLO检测 → 得到球的bbox（粗定位，过滤语义）
    2. IR在bbox内检测 → 精确定位和深度（高精度，高帧率）
    3. 如果YOLO失败 → 回退到全图IR检测（容错）
    """

    # Adaptive ranging keeps stereo authoritative at every distance. Geometry
    # is only a weak anomaly reference/fallback; it is never selected merely
    # because the ball is close to the camera.
    ADAPTIVE_SENSOR_JUMP_MM = 200.0
    ADAPTIVE_SENSOR_MAX_MM = 1000.0
    ADAPTIVE_SENSOR_GEOMETRY_GAP_MM = 300.0
    # Independent scale-consistency guard.  The morphology ellipse radius is
    # measured from the IR image, while the projected radius is implied by the
    # stereo sensor depth.  A large relative mismatch means the selected
    # stereo surface is unlikely to belong to the detected ball.
    ADAPTIVE_SENSOR_RADIUS_MISMATCH_RATIO = 0.22
    # The D455 can enter a near-range floor before it produces the large jump
    # caught above.  Detect that precursor from three causal camera frames:
    # stereo remains almost stationary while two independent image cues say
    # the rapidly rising ball is still approaching.  Three 90 Hz frames span
    # only about 22 ms, short enough for a sub-second juggling flight, while
    # the image-speed condition avoids mistaking the real trajectory apex for
    # a depth plateau.
    ADAPTIVE_NEAR_FIELD_WINDOW_FRAMES = 3
    ADAPTIVE_NEAR_FIELD_MAX_SENSOR_MM = 400.0
    ADAPTIVE_NEAR_FIELD_MAX_SENSOR_APPROACH_MM = 8.0
    ADAPTIVE_NEAR_FIELD_MIN_GEOMETRY_APPROACH_MM = 12.0
    ADAPTIVE_NEAR_FIELD_MIN_RADIUS_GROWTH_RATIO = 0.05
    ADAPTIVE_NEAR_FIELD_MIN_UPWARD_PIXEL_SPEED_PX_S = 500.0
    ADAPTIVE_NEAR_FIELD_MAX_WINDOW_S = 0.050
    ADAPTIVE_SENSOR_RECOVERY_FRAMES = 2
    ADAPTIVE_PREDICTION_MAX_S = 0.20
    ADAPTIVE_MAX_DEPTH_SPEED_MM_S = 10000.0
    ADAPTIVE_GEOMETRY_CORRECTION_GAIN = 0.25
    ADAPTIVE_GEOMETRY_CORRECTION_LIMIT_MM = 120.0

    # Robot-attached, fixed-ball calibration (30 independent racket poses,
    # 2026-08-12).  RealSense returns the first stereo surface on the ball,
    # while backprojection needs the optical-axis depth of the ball centre.
    # The fitted offset varies mildly with range, so use a bounded affine
    # model instead of blindly adding the nominal 20 mm physical radius.
    # Clamping the calibration input keeps out-of-range behaviour continuous
    # and prevents extrapolation from creating a large correction.
    SENSOR_CENTER_DEPTH_REFERENCE_MM = 500.0
    SENSOR_CENTER_DEPTH_INTERCEPT_MM = 14.96392834
    SENSOR_CENTER_DEPTH_SLOPE_PER_100_MM = -1.99083073
    SENSOR_CENTER_DEPTH_VALID_MIN_MM = 328.0
    SENSOR_CENTER_DEPTH_VALID_MAX_MM = 682.0

    # Optional observation-quality experiments.  These constants intentionally
    # stay conservative and live in detector/output units (mm, mm/s).  All
    # associated features default off so the legacy camera path is bit-for-bit
    # selected unless an operator enables a switch.
    MOTION_GATE_XY_BASE_MM = 50.0
    # At 90 Hz, normal depth/transform noise is far below 50 mm.  The previous
    # 80 mm base admitted the 85 mm late-frame vertical innovation in recording
    # 20260812_152353, which then pulled the launch fit from about 2.7 m/s down
    # to 1.2 m/s.  The time allowance below still covers model acceleration.
    MOTION_GATE_Z_BASE_MM = 50.0
    MOTION_GATE_REACQUISITION_MM = 120.0
    MOTION_GATE_MAX_DT_S = 0.080
    # A short camera/depth gap must not disable the gate.  Beyond this
    # horizon the old ballistic state is too stale to be authoritative, so
    # reacquisition is confirmed by two mutually-consistent fresh 3D points.
    MOTION_GATE_MAX_REACQUISITION_DT_S = 0.45
    MOTION_REACQUISITION_CONFIRM_MAX_GAP_S = 0.050
    MOTION_GATE_MAX_SPEED_MM_S = 6000.0
    PHYSICS_VELOCITY_MIN_FRAMES = 5
    PHYSICS_VELOCITY_GRAVITY_MM_S2 = 9800.0
    PHYSICS_VELOCITY_MAX_GAP_S = 0.050
    # The free-flight fit is also used as a low-latency endpoint position
    # smoother.  It evaluates the fitted trajectory at the newest timestamp,
    # rather than averaging old positions, so it does not add a half-window
    # phase delay.  The correction cap protects the policy from a bad fit.
    TRAJECTORY_POSITION_MAX_CORRECTION_MM = 12.0
    # The fitted endpoint correction may change sign between adjacent camera
    # frames.  Bounding each correction only against the raw point permits a
    # 24 mm correction swing on top of real motion, so cap the final published
    # endpoint against the previous published endpoint as well.  This is an
    # output-continuity guard, not a second measurement gate.
    TRAJECTORY_POSITION_MAX_OUTPUT_SPEED_MM_S = 6000.0
    FLIGHT_VELOCITY_UPDATE_GAIN = 0.65
    FLIGHT_VELOCITY_MAX_CORRECTION_MM_S = 450.0
    # Small fit residuals are dominated by depth/centroid noise and should not
    # receive the same gain as a genuine model mismatch.  Ramp causally to the
    # established 0.65 gain so fast flight/reacquisition keeps its response.
    # Kept as an offline tuning hook.  The three-day real corpus showed that
    # residual-dependent gain slightly increased p95 velocity error, so the
    # production default remains the fixed, validated gain.
    FLIGHT_VELOCITY_ADAPTIVE_GAIN_ENABLED = False
    FLIGHT_VELOCITY_MIN_UPDATE_GAIN = 0.35
    FLIGHT_VELOCITY_FULL_GAIN_RESIDUAL_MM_S = 650.0
    # A supported ball is not in free fall.  With a fresh racket pose the
    # relative ball/racket motion is authoritative; the tighter no-racket
    # fallback is used by offline tests and stand-alone detector tools.
    SUPPORTED_RACKET_DISTANCE_MM = 80.0
    SUPPORTED_RELATIVE_SPEED_MM_S = 450.0
    SUPPORTED_NO_RACKET_SPAN_MM = 1.5
    SUPPORTED_NO_RACKET_SPEED_MM_S = 80.0
    RACKET_STATE_MAX_AGE_S = 0.080
    RACKET_STATE_FALLBACK_MAX_AGE_S = 0.250
    # Impact reset is deliberately two-phase.  A single noisy positive dz is
    # only a candidate; a second fresh rising sample confirms the impossible
    # ballistic reversal before pre-impact history is discarded.
    IMPACT_CONFIRM_MAX_GAP_S = 0.035
    IMPACT_CONFIRM_MIN_TOTAL_RISE_MM = 4.0
    IMPACT_MAX_RACKET_DISTANCE_MM = 120.0
    IMPACT_CONFIRM_MAX_RACKET_DISTANCE_MM = 150.0
    IMPACT_RECENT_RACKET_DISTANCE_MM = 180.0
    IMPACT_NO_RACKET_THRESHOLD_SCALE = 1.50
    IR_CONTOUR_EDGE_MARGIN_PX = 1.0

    def __init__(self,
                 yolo_model_path,
                 ball_diameter_mm=40.0,
                 width=848,
                 height=480,
                 fps=60,
                 yolo_interval=3,  # YOLO每N帧运行一次
                 yolo_conf=0.25,  # YOLO置信度阈值（降低以提高检测率）
                 target_class='Class 0',  # YOLO目标类别（自定义模型的类别名）
                 # IR参数（从ir_params.json加载）
                 ir_threshold=200,
                 use_adaptive=True,
                 dilate_iterations=4,
                 close_kernel_size=11,
                 denoise_kernel_size=3,
                 min_radius=5,
                 max_radius=50,
                 circularity_min=0.75,
                 laser_power=150,
                 exposure=None,        # IR 手动曝光(μs)；None=保持自动曝光
                 gain=None,            # IR 手动增益；None=保持自动
                 brightness_percentile=30,  # 自适应阈值的背景亮度分位数
                 roi_margin=80,  # bbox扩展边距
                 depth_scale=0.001,
                 depth_neighborhood_px=2,
                 calibration_data: Optional[CalibrationData] = None,
                 output_frame='base_link',
                 print_every=10,
                 offline_camera_params: Optional[dict] = None,
                 depth_mode='sensor',
                 sensor_surface_to_center_depth_enabled=False,
                 pixel_kalman_enabled=True,
                 pixel_kalman_process_noise_pos=0.3,
                 pixel_kalman_process_noise_vel=3.0,
                 pixel_kalman_measurement_noise=1.2,
                 pixel_kalman_innovation_gate_px=15.0,
                 pixel_kalman_reset_after_outliers=3,
                 ir_depth_kalman_enabled=False,
                 synchronized_depth_enabled=False,
                 depth_temporal_window_frames=5,
                 velocity_least_squares_enabled=True,
                 velocity_window_frames=3,
                 clear_buffers_on_detection_loss_enabled=False,
                 motion_consistency_gate_enabled=False,
                 physics_constrained_velocity_enabled=False,
                 trajectory_position_smoothing_enabled=False,
                 impact_velocity_reset_enabled=True,
                 impact_min_downward_velocity_m_s=0.25,
                 impact_min_upward_velocity_m_s=0.25,
                 impact_min_velocity_jump_m_s=0.80,
                 impact_refractory_s=0.08,
                 candidate_filter_enabled=True,
                 candidate_min_brightness=23000.0,
                 candidate_max_depth_disagreement_mm=180.0,
                 candidate_motion_scale_px=130.0,
                 depth_correction_model_path='depth_correction_model.pkl'):  # 新增参数
        """
        初始化混合检测器

        Args:
            yolo_model_path: YOLO模型路径
            yolo_interval: YOLO检测间隔（帧数），例如3表示每3帧运行一次YOLO
            yolo_conf: YOLO置信度阈值
            roi_margin: YOLO bbox周围的扩展边距（像素）
            其他参数：见IR检测器
        """
        # 先保存深度校正模型路径（在_init_runtime_state之前）
        self.depth_correction_model_path = depth_correction_model_path

        self.ball_diameter_mm = ball_diameter_mm
        self.ball_radius_mm = ball_diameter_mm / 2.0
        self.yolo_interval = yolo_interval
        self.yolo_conf = yolo_conf
        self.target_class = target_class
        self.roi_margin = roi_margin
        self.depth_scale = depth_scale
        self.depth_neighborhood_px = depth_neighborhood_px
        self.depth_mode = self._normalize_depth_mode(depth_mode)
        self.sensor_surface_to_center_depth_enabled = bool(
            sensor_surface_to_center_depth_enabled)
        self.print_every = print_every
        self.pixel_kalman_enabled = bool(pixel_kalman_enabled)
        self.pixel_kalman_process_noise_pos = max(
            0.0, float(pixel_kalman_process_noise_pos))
        self.pixel_kalman_process_noise_vel = max(
            0.0, float(pixel_kalman_process_noise_vel))
        self.pixel_kalman_measurement_noise = max(
            1e-6, float(pixel_kalman_measurement_noise))
        self.pixel_kalman_innovation_gate_px = max(
            0.0, float(pixel_kalman_innovation_gate_px))
        self.pixel_kalman_reset_after_outliers = max(
            1, int(pixel_kalman_reset_after_outliers))
        self.ir_depth_kalman_enabled = bool(ir_depth_kalman_enabled)
        self.synchronized_depth_enabled = bool(synchronized_depth_enabled)
        self.depth_temporal_window_frames = (
            self._normalize_depth_temporal_window(
                depth_temporal_window_frames))
        self.velocity_least_squares_enabled = bool(
            velocity_least_squares_enabled)
        self.velocity_window_frames = self._normalize_velocity_window(
            velocity_window_frames)
        self.clear_buffers_on_detection_loss_enabled = bool(
            clear_buffers_on_detection_loss_enabled)
        self.motion_consistency_gate_enabled = bool(
            motion_consistency_gate_enabled)
        self.physics_constrained_velocity_enabled = bool(
            physics_constrained_velocity_enabled)
        self.trajectory_position_smoothing_enabled = bool(
            trajectory_position_smoothing_enabled)
        self.impact_velocity_reset_enabled = bool(
            impact_velocity_reset_enabled)
        self.impact_min_downward_velocity_mm_s = 1000.0 * max(
            0.0, float(impact_min_downward_velocity_m_s))
        self.impact_min_upward_velocity_mm_s = 1000.0 * max(
            0.0, float(impact_min_upward_velocity_m_s))
        self.impact_min_velocity_jump_mm_s = 1000.0 * max(
            0.0, float(impact_min_velocity_jump_m_s))
        self.impact_refractory_s = max(
            0.0, float(impact_refractory_s))
        self.last_velocity_impact_time = None
        self.last_velocity_impact_detected = False
        self.last_velocity_ready = False
        self.candidate_filter_enabled = bool(candidate_filter_enabled)
        self.candidate_min_brightness = max(
            0.0, float(candidate_min_brightness))
        self.candidate_max_depth_disagreement_mm = max(
            0.0, float(candidate_max_depth_disagreement_mm))
        self.candidate_motion_scale_px = max(
            1.0, float(candidate_motion_scale_px))
        self.debug_yolo = str(os.environ.get("PINGPONG_YOLO_DEBUG", "")).lower() in (
            "1", "true", "yes", "on")
        # IR 曝光/增益（构造时应用到 stereo sensor）与自适应背景分位数
        self.ir_exposure = exposure
        self.ir_gain = gain
        self.brightness_percentile = float(brightness_percentile)

        # IR检测参数
        self.ir_params = {
            'threshold': ir_threshold,
            'use_adaptive': use_adaptive,
            'dilate_iterations': dilate_iterations,
            'close_kernel_size': close_kernel_size,
            'denoise_kernel_size': denoise_kernel_size,
            'min_radius': min_radius,
            'max_radius': max_radius,
            'circularity_min': circularity_min,
        }

        # 坐标系配置
        self.output_frame = output_frame
        self.calibration_data = calibration_data
        self.T_base_camera = None
        if calibration_data is not None:
            self.T_base_camera = calibration_data.T_base_camera

        # 加载YOLO模型
        print(f"加载YOLO模型: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        print("YOLO模型加载完成")

        # YOLO推理参数（优化性能）
        self.yolo_device = 'cuda:0' if self._check_cuda() else 'cpu'
        self.yolo_imgsz = 640  # 推理尺寸

        print(f"YOLO推理设备: {self.yolo_device}")
        print(f"YOLO推理尺寸: {self.yolo_imgsz}")

        # 预热YOLO模型
        print("预热YOLO模型...")
        dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
        predict_kwargs = {
            'conf': self.yolo_conf,
            'verbose': False,
            'device': self.yolo_device,
            'imgsz': self.yolo_imgsz,
        }
        for _ in range(3):
            _ = self.yolo_model(dummy_image, **predict_kwargs)
        print("YOLO模型预热完成")

        # 离线模式：不开相机，从录制时保存的 camera_params JSON 重建内参/外参，
        # 使 rs.rs2_* 投影数学与在线完全一致。offline_camera_params 为 None
        # 时走原有在线 RealSense 初始化。
        self.offline_camera_params = offline_camera_params
        if self.offline_camera_params is not None:
            self._init_offline_camera(self.offline_camera_params, width, height)
            return

        # Stereo and RGB use independent pipelines.  A mixed-rate frameset
        # would synchronize to the slower 60 Hz RGB stream and silently drop
        # IR/depth frames.  The control path therefore waits only on the
        # 90 Hz stereo pipeline; RGB remains 848x480@60 for visualization and
        # optional YOLO and is polled asynchronously.
        self.pipeline = rs.pipeline()
        config = rs.config()
        ir_fps = max(1, int(fps))
        rgb_width, rgb_height, rgb_fps = width, height, 60

        try:
            config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, ir_fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, ir_fps)
            print(
                f"配置: IR/Depth {width}x{height}@{ir_fps}fps; "
                f"RGB独立流 {rgb_width}x{rgb_height}@{rgb_fps}fps")
        except Exception as e:
            # 回退配置
            print(f"配置失败: {e}, 尝试回退配置...")
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
            width, height = 640, 480
            rgb_fps, ir_fps = 60, 60
            rgb_width, rgb_height = width, height
            print(f"回退配置: RGB {width}x{height}@{rgb_fps}fps, IR/Depth {width}x{height}@{ir_fps}fps")

        self.image_width = width
        self.image_height = height
        self.color_stream_width = int(rgb_width)
        self.color_stream_height = int(rgb_height)
        self.fps = ir_fps  # 使用IR帧率作为主帧率
        self.color_pipeline = None
        self.latest_color_image = None

        # 启动pipeline（添加重试机制）
        print("启动RealSense相机...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                profile = self.pipeline.start(config)
                print("相机启动成功")
                break
            except RuntimeError as e:
                if attempt < max_retries - 1:
                    print(f"启动失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    print("等待2秒后重试...")
                    time.sleep(2)
                else:
                    print(f"相机启动失败，已尝试{max_retries}次")
                    raise

        # Bind RGB to the exact same physical device chosen for stereo.  It is
        # intentionally not part of the stereo frameset and cannot throttle
        # the IR/depth wait loop.
        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        color_config = rs.config()
        color_config.enable_device(serial)
        color_config.enable_stream(
            rs.stream.color, rgb_width, rgb_height,
            rs.format.bgr8, rgb_fps)
        try:
            self.color_pipeline = rs.pipeline()
            color_profile = self.color_pipeline.start(color_config)
            color_frames = self.color_pipeline.wait_for_frames(
                timeout_ms=1000)
            first_color = color_frames.get_color_frame()
            if first_color:
                self.latest_color_image = np.asanyarray(
                    first_color.get_data()).copy()
            print("RGB独立流启动成功（不阻塞IR/Depth检测）")
        except Exception as exc:
            if self.color_pipeline is not None:
                try:
                    self.color_pipeline.stop()
                except Exception:
                    pass
            self.color_pipeline = None
            print(f"WARNING: RGB独立流启动失败，使用IR灰度可视化: {exc}")
            color_profile = None

        # 设置激光功率
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, laser_power)
            print(f"激光功率: {depth_sensor.get_option(rs.option.laser_power)} mW")

        # 设置 IR 曝光/增益（IR imager 与 depth 共用同一 stereo sensor，
        # 选项挂在 depth_sensor 上）。手动锁曝光可显著改善 IR 一致性、抑制
        # 近距离激光过曝(blooming)导致的半径偏大 → 稳定 geometry 深度。
        # exposure/gain 任一为 None 时保持该项自动。
        if self.ir_exposure is not None or self.ir_gain is not None:
            # 手动曝光前必须关自动曝光，否则 set exposure/gain 无效
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                print("IR自动曝光: 关闭（手动模式）")
        if self.ir_exposure is not None and depth_sensor.supports(rs.option.exposure):
            depth_sensor.set_option(rs.option.exposure, float(self.ir_exposure))
            print(f"IR曝光: {depth_sensor.get_option(rs.option.exposure):.0f} μs")
        if self.ir_gain is not None and depth_sensor.supports(rs.option.gain):
            depth_sensor.set_option(rs.option.gain, float(self.ir_gain))
            print(f"IR增益: {depth_sensor.get_option(rs.option.gain):.0f}")

        # 获取内参
        # 修复：需要同时保存IR和RGB内参
        # - IR内参：用于IR坐标反投影到IR 3D
        # - RGB内参：手眼标定是针对RGB的
        ir_stream = profile.get_stream(rs.stream.infrared, 1)
        self.ir_intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()

        # 获取RGB内参（用于可视化和手眼标定坐标系）。即使RGB采集启动
        # 失败，也可从设备公开profile恢复固定内/外参，保证IR检测仍能运行。
        color_stream = None
        if color_profile is not None:
            color_stream = color_profile.get_stream(rs.stream.color)
        else:
            for sensor in device.query_sensors():
                for candidate in sensor.get_stream_profiles():
                    try:
                        video = candidate.as_video_stream_profile()
                        if (candidate.stream_type() == rs.stream.color
                                and video.width() == rgb_width
                                and video.height() == rgb_height
                                and candidate.fps() == rgb_fps):
                            color_stream = candidate
                            break
                    except Exception:
                        continue
                if color_stream is not None:
                    break
        if color_stream is None:
            raise RuntimeError(
                "RGB profile unavailable; cannot recover calibrated "
                "IR-to-RGB extrinsics")
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # 保存IR内参用于反投影
        self.ir_fx = self.ir_intrinsics.fx
        self.ir_fy = self.ir_intrinsics.fy
        self.ir_cx = self.ir_intrinsics.ppx
        self.ir_cy = self.ir_intrinsics.ppy

        if calibration_data is not None:
            # 使用标定数据（假设是针对RGB的，因为手眼标定是RGB相机）
            self.camera_matrix = calibration_data.camera_matrix.copy()
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            print("使用RGB标定内参（手眼标定）")
        else:
            # 使用RGB内参（因为手眼标定是针对RGB的）
            self.fx = self.color_intrinsics.fx
            self.fy = self.color_intrinsics.fy
            self.cx = self.color_intrinsics.ppx
            self.cy = self.color_intrinsics.ppy
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
            print("使用RGB内参（手眼标定坐标系）")

        print(f"RGB内参: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        print(f"IR内参: fx={self.ir_fx:.2f}, fy={self.ir_fy:.2f}, cx={self.ir_cx:.2f}, cy={self.ir_cy:.2f}")
        print(f"IR畸变模型: {self.ir_intrinsics.model}")
        print(f"IR畸变系数: {self.ir_intrinsics.coeffs}")

        # 对齐器（不使用，保留IR原生坐标系）
        # self.align_to_color = rs.align(rs.stream.color)

        # 获取IR到RGB的外参（关键：用于将IR 3D点转换到RGB 3D点）
        depth_stream = profile.get_stream(rs.stream.depth)
        try:
            self.depth_to_color_extrinsics = (
                depth_stream.get_extrinsics_to(color_stream))
        except RuntimeError:
            # Profiles returned by two separately started pipelines do not
            # always share librealsense's active extrinsics graph.  Native
            # profiles queried from the same device do, and describe the same
            # fixed physical sensor transform.
            native_depth = None
            native_color = None
            for sensor in device.query_sensors():
                for candidate in sensor.get_stream_profiles():
                    try:
                        video = candidate.as_video_stream_profile()
                        if (candidate.stream_type() == rs.stream.depth
                                and video.width() == width
                                and video.height() == height
                                and candidate.fps() == ir_fps
                                and candidate.format() == rs.format.z16):
                            native_depth = candidate
                        elif (candidate.stream_type() == rs.stream.color
                                and video.width() == rgb_width
                                and video.height() == rgb_height
                                and candidate.fps() == rgb_fps):
                            native_color = candidate
                    except Exception:
                        continue
            if native_depth is None or native_color is None:
                raise RuntimeError(
                    "Unable to resolve native Depth/RGB profiles for "
                    "cross-pipeline extrinsics")
            self.depth_to_color_extrinsics = (
                native_depth.get_extrinsics_to(native_color))

        # 构建IR→RGB的4x4齐次变换矩阵（更清晰直接）
        # librealsense exposes ``rs2_extrinsics.rotation`` as a column-major
        # flat array.  NumPy reshapes in row-major order, so transpose it here
        # to reproduce rs.rs2_transform_point_to_point exactly.  Without the
        # transpose, the small D455 depth->color rotation is inverted and the
        # RGB overlay is displaced by a few pixels.
        R = _rs_extrinsics_rotation_matrix(
            self.depth_to_color_extrinsics.rotation)
        t = np.array(self.depth_to_color_extrinsics.translation)
        self.T_ir_to_rgb = np.eye(4)
        self.T_ir_to_rgb[:3, :3] = R
        self.T_ir_to_rgb[:3, 3] = t

        print(f"IR→RGB外参平移(m): {self.depth_to_color_extrinsics.translation}")
        print(f"IR→RGB外参旋转: {R[0]}")  # 只打印第一行避免太长
        print(f"T_ir_to_rgb齐次矩阵:\n{self.T_ir_to_rgb}")

        # 共享运行时状态（在线/离线都需要）
        self._init_runtime_state()

        # 预热（添加超时保护）
        print("相机预热中...")
        warmup_count = 0
        warmup_timeout = 30  # 30次尝试，约5秒
        while warmup_count < 20 and warmup_timeout > 0:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if frames:
                    warmup_count += 1
            except RuntimeError:
                warmup_timeout -= 1
                print(".", end="", flush=True)
                if warmup_timeout == 0:
                    print("\n⚠️  相机预热超时，但继续运行...")
                    break
        print(f"\n混合检测器就绪（预热{warmup_count}/20帧）")

    @staticmethod
    def _normalize_depth_mode(mode):
        value = str(mode).strip().lower()
        return value if value in ('sensor', 'geometry', 'adaptive') else 'sensor'

    def _reset_adaptive_depth_state(self):
        """Reset trusted-sensor history and anomaly recovery state."""
        self.depth_mode = self._normalize_depth_mode(
            getattr(self, 'depth_mode', 'sensor'))
        self.adaptive_sensor_history = deque(maxlen=3)
        self.adaptive_near_field_history = deque(
            maxlen=self.ADAPTIVE_NEAR_FIELD_WINDOW_FRAMES - 1)
        self.adaptive_sensor_anomaly_active = False
        self.adaptive_sensor_recovery_count = 0
        self.adaptive_last_raw_sensor_mm = None
        self.adaptive_frame_index = 0
        self.adaptive_last_geometry_mm = None
        self.adaptive_last_output_depth_mm = None
        self._pending_adaptive_sensor_sample = None
        self._pending_adaptive_near_field_sample = None
        self._pending_adaptive_state_snapshot = None
        self.last_depth_selection = {
            'mode': self.depth_mode,
            'method': None,
            'sensor_weight': None,
            'disagreement_mm': None,
            'sensor_rejected': False,
            'rejection_reason': None,
            'sensor_jump_mm': None,
            'observed_radius_px': None,
            'sensor_projected_radius_px': None,
            'radius_mismatch_ratio': None,
            'near_field_scale_stall': False,
            'near_field_window_samples': 0,
            'near_field_window_span_s': None,
            'near_field_sensor_approach_mm': None,
            'near_field_geometry_approach_mm': None,
            'near_field_radius_growth_ratio': None,
            'near_field_upward_pixel_speed_px_s': None,
            'prediction_mm': None,
            'prediction_residual_mm': None,
            'prediction_horizon_s': None,
            'sensor_recovery_count': 0,
        }

    def set_depth_mode(self, mode):
        """Change the preferred depth source without rebuilding the camera."""
        normalized = self._normalize_depth_mode(mode)
        if normalized == self.depth_mode:
            return
        self.depth_mode = normalized
        # Do not mix samples produced by two different ranging methods.  In
        # particular, geometry can be tens of millimetres away from stereo
        # depth, so retaining the old median window makes a live switch look
        # unresponsive for several frames.
        depth_history = getattr(self, 'depth_history', None)
        if depth_history is not None:
            depth_history.clear()
        self._reset_adaptive_depth_state()

    def configure_sensor_surface_to_center_depth(self, *, enabled):
        """Toggle calibrated stereo surface-to-ball-centre reconstruction.

        Raw stereo depth remains unchanged for anomaly guards, adaptive-depth
        recovery, and temporal depth history.  Only a trusted sensor-derived
        depth is corrected immediately before 3D backprojection.  Clearing
        downstream 3D state on a live toggle prevents the intentional depth
        step from being interpreted as ball velocity or a motion outlier.
        """
        enabled = bool(enabled)
        if enabled == self.sensor_surface_to_center_depth_enabled:
            return
        self.sensor_surface_to_center_depth_enabled = enabled
        depth_history = getattr(self, 'depth_history', None)
        if depth_history is not None:
            depth_history.clear()
        ir_depth_kf = getattr(self, 'ir_depth_kf', None)
        if ir_depth_kf is not None:
            ir_depth_kf.reset()
        self.position_history = deque(maxlen=self._velocity_history_maxlen())
        self._reset_velocity_tracking_state()
        self.motion_last_trusted_time = None
        self.motion_last_trusted_position = None
        self.motion_last_trusted_velocity = None
        self.motion_last_innovation_mm = None
        self.motion_last_gate_reason = None
        self.observation_gap_active = True
        # This toggle deliberately preserves 2D tracking.  Therefore do not
        # claim that the full loss-buffer clear already ran: if the following
        # frame is actually lost, the configured clear-on-loss path must still
        # be allowed to clear pixel/candidate state.
        self.observation_buffers_cleared_for_gap = False
        self.motion_reacquisition_candidates.clear()
        self.motion_reacquisition_seed = None

    def _sensor_surface_to_center_depth(self, depth_mm, depth_method):
        """Return final reconstruction depth and auditable diagnostics."""
        surface_depth_mm = float(depth_mm)
        sensor_derived = str(depth_method or '') in {
            'sensor', 'adaptive_sensor'}
        diagnostics = {
            'sensor_surface_to_center_depth_enabled': bool(
                self.sensor_surface_to_center_depth_enabled),
            'sensor_surface_to_center_depth_applied': False,
            'sensor_surface_depth_mm': (
                surface_depth_mm if sensor_derived else None),
            'sensor_center_depth_mm': None,
            'sensor_center_depth_correction_mm': None,
            'sensor_center_depth_calibration_clamped': False,
        }
        if (not self.sensor_surface_to_center_depth_enabled
                or not sensor_derived):
            return surface_depth_mm, diagnostics

        calibration_depth_mm = float(np.clip(
            surface_depth_mm,
            self.SENSOR_CENTER_DEPTH_VALID_MIN_MM,
            self.SENSOR_CENTER_DEPTH_VALID_MAX_MM))
        correction_mm = (
            self.SENSOR_CENTER_DEPTH_INTERCEPT_MM
            + self.SENSOR_CENTER_DEPTH_SLOPE_PER_100_MM
            * (calibration_depth_mm
               - self.SENSOR_CENTER_DEPTH_REFERENCE_MM) / 100.0)
        center_depth_mm = surface_depth_mm + correction_mm
        diagnostics.update({
            'sensor_surface_to_center_depth_applied': True,
            'sensor_center_depth_mm': float(center_depth_mm),
            'sensor_center_depth_correction_mm': float(correction_mm),
            'sensor_center_depth_calibration_clamped': bool(
                calibration_depth_mm != surface_depth_mm),
        })
        return float(center_depth_mm), diagnostics

    def configure_pixel_kalman(
            self, *, enabled, process_noise_pos, process_noise_vel,
            measurement_noise, innovation_gate_px, reset_after_outliers):
        """Apply pixel-filter tuning atomically for the next camera frame.

        Replacing the filter intentionally clears its history.  This prevents
        old state/covariance from masking a live parameter change and makes
        disabling the filter immediately expose the raw IR centre.
        """
        self.pixel_kalman_enabled = bool(enabled)
        self.pixel_kalman_process_noise_pos = max(
            0.0, float(process_noise_pos))
        self.pixel_kalman_process_noise_vel = max(
            0.0, float(process_noise_vel))
        self.pixel_kalman_measurement_noise = max(
            1e-6, float(measurement_noise))
        self.pixel_kalman_innovation_gate_px = max(
            0.0, float(innovation_gate_px))
        self.pixel_kalman_reset_after_outliers = max(
            1, int(reset_after_outliers))
        replacement = self._new_pixel_kalman_filter()
        self.pixel_kf = replacement

    @staticmethod
    def _normalize_depth_temporal_window(value):
        window = max(1, min(9, int(value)))
        if window % 2 == 0:
            window = window + 1 if window < 9 else window - 1
        return window

    def configure_ir_depth_processing(
            self, *, kalman_enabled, synchronized_depth_enabled,
            depth_temporal_window_frames):
        """Apply independent 3D-KF and same-frameset-depth switches."""
        kalman_enabled = bool(kalman_enabled)
        synchronized_depth_enabled = bool(synchronized_depth_enabled)
        depth_temporal_window_frames = (
            self._normalize_depth_temporal_window(
                depth_temporal_window_frames))
        changed = (
            kalman_enabled != self.ir_depth_kalman_enabled
            or synchronized_depth_enabled != self.synchronized_depth_enabled
            or depth_temporal_window_frames
            != self.depth_temporal_window_frames)
        self.ir_depth_kalman_enabled = kalman_enabled
        self.synchronized_depth_enabled = synchronized_depth_enabled
        self.depth_temporal_window_frames = depth_temporal_window_frames
        if changed:
            self.ir_depth_kf.reset()
            self.depth_history = deque(
                maxlen=self.depth_temporal_window_frames)
            self.position_history.clear()
            self._reset_velocity_tracking_state()

    @staticmethod
    def _normalize_velocity_window(value):
        return max(2, min(9, int(value)))

    def configure_velocity_estimation(
            self, *, least_squares_enabled, window_frames,
            impact_reset_enabled, impact_min_downward_velocity_m_s,
            impact_min_upward_velocity_m_s,
            impact_min_velocity_jump_m_s, impact_refractory_s):
        """Apply the camera velocity estimator contract at frame boundary.

        A configuration change clears the old samples so a live web update
        cannot combine positions produced under two different estimators.
        Thresholds enter the public API in m/s while detector positions and
        returned velocities use millimetres and mm/s.
        """
        least_squares_enabled = bool(least_squares_enabled)
        window_frames = self._normalize_velocity_window(window_frames)
        impact_reset_enabled = bool(impact_reset_enabled)
        min_down = 1000.0 * max(
            0.0, float(impact_min_downward_velocity_m_s))
        min_up = 1000.0 * max(
            0.0, float(impact_min_upward_velocity_m_s))
        min_jump = 1000.0 * max(
            0.0, float(impact_min_velocity_jump_m_s))
        refractory_s = max(0.0, float(impact_refractory_s))
        changed = (
            least_squares_enabled != self.velocity_least_squares_enabled
            or window_frames != self.velocity_window_frames
            or impact_reset_enabled != self.impact_velocity_reset_enabled
            or min_down != self.impact_min_downward_velocity_mm_s
            or min_up != self.impact_min_upward_velocity_mm_s
            or min_jump != self.impact_min_velocity_jump_mm_s
            or refractory_s != self.impact_refractory_s)
        self.velocity_least_squares_enabled = least_squares_enabled
        self.velocity_window_frames = window_frames
        self.impact_velocity_reset_enabled = impact_reset_enabled
        self.impact_min_downward_velocity_mm_s = min_down
        self.impact_min_upward_velocity_mm_s = min_up
        self.impact_min_velocity_jump_mm_s = min_jump
        self.impact_refractory_s = refractory_s
        if changed:
            self.position_history = deque(maxlen=self._velocity_history_maxlen())
            self.last_velocity_impact_time = None
            self.last_velocity_impact_detected = False
            self._reset_velocity_tracking_state()

    def configure_observation_quality(
            self, *, clear_buffers_on_detection_loss_enabled,
            motion_consistency_gate_enabled,
            physics_constrained_velocity_enabled,
            trajectory_position_smoothing_enabled=False):
        """Apply opt-in camera observation-quality experiments atomically."""
        clear_on_loss = bool(clear_buffers_on_detection_loss_enabled)
        motion_gate = bool(motion_consistency_gate_enabled)
        physics_velocity = bool(physics_constrained_velocity_enabled)
        trajectory_position_smoothing = bool(
            trajectory_position_smoothing_enabled)
        changed = (
            clear_on_loss != self.clear_buffers_on_detection_loss_enabled
            or motion_gate != self.motion_consistency_gate_enabled
            or physics_velocity != self.physics_constrained_velocity_enabled
            or trajectory_position_smoothing
            != self.trajectory_position_smoothing_enabled)
        self.clear_buffers_on_detection_loss_enabled = clear_on_loss
        self.motion_consistency_gate_enabled = motion_gate
        self.physics_constrained_velocity_enabled = physics_velocity
        self.trajectory_position_smoothing_enabled = (
            trajectory_position_smoothing)
        if changed:
            self.position_history = deque(maxlen=self._velocity_history_maxlen())
            self.last_velocity_impact_time = None
            self.last_velocity_impact_detected = False
            self._reset_velocity_tracking_state()
            self.motion_last_trusted_time = None
            self.motion_last_trusted_position = None
            self.motion_last_trusted_velocity = None
            self.motion_last_innovation_mm = None
            self.motion_last_gate_reason = None
            self.observation_gap_active = False
            self.observation_buffers_cleared_for_gap = False
            self.motion_reacquisition_candidates.clear()
            self.motion_reacquisition_seed = None

    def _velocity_history_maxlen(self):
        if getattr(self, 'physics_constrained_velocity_enabled', False):
            return max(
                int(self.velocity_window_frames),
                int(self.PHYSICS_VELOCITY_MIN_FRAMES))
        return int(self.velocity_window_frames)

    def _reset_velocity_tracking_state(self):
        """Reset one causal motion segment without resetting diagnostics."""
        self.last_velocity_ready = False
        self.velocity_motion_state = 'initializing'
        self.velocity_model = 'unready'
        self.pending_impact = None
        self.post_impact_lateral_prior = None
        self.last_velocity_output = np.zeros(3, dtype=np.float64)
        self.last_velocity_output_time = None
        self.velocity_restart_reference_time = None
        self.velocity_restart_reference_velocity = None
        self.last_velocity_restart_bridge_used = False
        self.last_position_estimate = None
        self.last_position_filter_residual_mm = None
        self.last_velocity_filter_correction_mm_s = None
        self.last_impact_racket_distance_mm = None
        self.last_impact_racket_gate_used = False
        self.last_impact_racket_state_age_s = None
        self.last_impact_candidate_rejection_reason = None
        self.last_impact_pair_vertical_velocity_mm_s = None
        self.last_impact_previous_vertical_velocity_mm_s = None
        self.last_impact_evidence_source = None

    def set_racket_state(self, position_m, timestamp_s=None):
        """Publish the latest racket centre for optional contact validation.

        The detector runs in a camera thread while robot kinematics are
        updated by the control thread.  Replacing one immutable tuple keeps
        this hand-off lock-free.  Racket information is used only when the
        detector output is already in ``base_link``; trajectory reversal still
        provides a stand-alone fallback for offline tools.
        """
        if getattr(self, 'output_frame', 'base_link') != 'base_link':
            self.latest_racket_state = None
            return
        try:
            position_mm = (
                np.asarray(position_m, dtype=np.float64).reshape(3) * 1000.0)
            stamp = time.time() if timestamp_s is None else float(timestamp_s)
        except (TypeError, ValueError):
            return
        if not np.all(np.isfinite(position_mm)) or not math.isfinite(stamp):
            return
        previous = getattr(self, 'latest_racket_state', None)
        velocity_mm_s = np.zeros(3, dtype=np.float64)
        if previous is not None:
            dt = float(stamp - previous[0])
            if 1.0e-4 < dt <= 0.10:
                velocity_mm_s = (position_mm - previous[1]) / dt
                speed = float(np.linalg.norm(velocity_mm_s))
                if speed > self.MOTION_GATE_MAX_SPEED_MM_S:
                    velocity_mm_s *= (
                        self.MOTION_GATE_MAX_SPEED_MM_S / speed)
        self.latest_racket_state = (
            stamp, position_mm.copy(), velocity_mm_s.copy())

    def clear_racket_state(self):
        """Invalidate contact geometry when current racket FK is unavailable."""
        self.latest_racket_state = None

    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _init_runtime_state(self):
        """初始化在线/离线共享的运行时状态（计数、队列、深度门等）。"""
        self.sensor_surface_to_center_depth_enabled = bool(getattr(
            self, 'sensor_surface_to_center_depth_enabled', False))
        self.clear_buffers_on_detection_loss_enabled = bool(getattr(
            self, 'clear_buffers_on_detection_loss_enabled', False))
        self.motion_consistency_gate_enabled = bool(getattr(
            self, 'motion_consistency_gate_enabled', False))
        self.physics_constrained_velocity_enabled = bool(getattr(
            self, 'physics_constrained_velocity_enabled', False))
        self.trajectory_position_smoothing_enabled = bool(getattr(
            self, 'trajectory_position_smoothing_enabled', False))
        # 状态
        self.frame_count = 0
        self.last_yolo_bbox = None  # 最近的YOLO bbox
        self.last_yolo_time = 0
        self.yolo_miss_count = 0
        self.max_yolo_miss = 3  # 修复：YOLO连续丢失3次后清除bbox（软约束）
        self.yolo_enabled = True
        self.ir_tracking_miss_count = 0

        # 统计
        self.total_frames = 0
        self.detected_frames = 0
        self.yolo_detections = 0
        self.ir_detections = 0
        self.hybrid_detections = 0
        self.fallback_detections = 0
        self.ir_edge_rejected_count = 0
        self.ir_roi_edge_rejected_count = 0

        self.fps_queue = deque(maxlen=30)
        self.yolo_time_queue = deque(maxlen=10)
        self.ir_time_queue = deque(maxlen=30)
        self.detect_time_queue = deque(maxlen=60)
        self.detected_time_queue = deque(maxlen=60)
        self.missed_time_queue = deque(maxlen=60)
        self.color_frame_count = 0
        self.color_last_timestamp_ms = None
        self.color_fps_queue = deque(maxlen=60)

        self.position_history = deque(maxlen=self._velocity_history_maxlen())
        self._reset_velocity_tracking_state()
        self.velocity_restart_bridge_count = 0
        self.last_velocity_impact_time = None
        self.last_velocity_impact_detected = False
        self.velocity_impact_confirmed_count = 0
        self.velocity_impact_rejected_count = 0
        self.velocity_impact_candidate_count = 0
        self.impact_racket_gate_required = bool(getattr(
            self, 'impact_racket_gate_required', False))
        self.latest_racket_state = None
        self.last_observation_time = None
        self.candidate_position_history = deque(maxlen=2)
        self.candidate_filter_stats = {
            'frames': 0,
            'candidates': 0,
            'accepted': 0,
            'rejected_brightness': 0,
            'rejected_depth': 0,
            'all_rejected': 0,
        }

        # 像素坐标历史（用于平滑可视化）
        self.pixel_history = deque(maxlen=5)  # (cx, cy, radius)
        self.observation_gap_active = False
        self.observation_buffers_cleared_for_gap = False
        self.observation_buffer_clear_count = 0
        self.motion_last_trusted_time = None
        self.motion_last_trusted_position = None
        self.motion_last_trusted_velocity = None
        self.motion_last_innovation_mm = None
        self.motion_last_gate_reason = None
        self.motion_gate_accepted_count = 0
        self.motion_gate_rejected_count = 0
        self.motion_impact_confirmation_bypass_count = 0
        self.motion_impact_candidate_bypass_count = 0
        self.motion_reacquisition_candidates = deque(maxlen=2)
        self.motion_reacquisition_seed = None

        # 2D像素卡尔曼滤波器（核心平滑机制）
        # 参数调优：静止球高稳定性，运动球低延迟
        self.pixel_kf = self._new_pixel_kalman_filter()

        # Joint IR+depth filter. Depth synchronization is controlled separately:
        # the filter can consume either the same-frameset depth or legacy median
        # depth so the two effects can be evaluated independently.
        self.ir_depth_kf = IRDepthKalmanFilter(
            frame_rotation=self._ir_to_output_rotation())

        # 深度历史（用于时域去尖峰+平滑，统一作用于 sensor/geometry 两条路径）
        self.depth_history = deque(
            maxlen=self.depth_temporal_window_frames)

        # Valid stereo depth is the primary metric measurement. Geometry is a
        # fallback only because the morphology radius follows a projector-dot
        # blob rather than the physical ball boundary.
        self.depth_source_state = 'sensor'
        self._reset_adaptive_depth_state()

        # FPS calculation uses the RealSense hardware frame timestamp.
        self.last_frame_timestamp = None

        # Adaptive-threshold and depth-correction state are shared by online
        # and offline detection paths.
        self.background_intensity = None
        self.depth_correction_model = None
        self.depth_correction_model_error = None
        self._pending_adaptive_sensor_sample = None
        self._pending_adaptive_near_field_sample = None
        self._pending_adaptive_state_snapshot = None
        # self.depth_correction_model_path is set at the start of __init__.
        self._load_depth_correction_model(self.depth_correction_model_path)

    def _ir_to_output_rotation(self):
        """Rotation from IR optical coordinates into the detector output frame."""
        ir_to_rgb = np.asarray(
            getattr(self, 'T_ir_to_rgb', np.eye(4)), dtype=np.float64)
        rotation = (
            ir_to_rgb[:3, :3]
            if ir_to_rgb.shape == (4, 4)
            else np.eye(3, dtype=np.float64))
        if (getattr(self, 'output_frame', 'base_link') == 'base_link'
                and getattr(self, 'T_base_camera', None) is not None):
            base_to_camera = np.asarray(
                self.T_base_camera, dtype=np.float64)
            if base_to_camera.shape == (4, 4):
                rotation = base_to_camera[:3, :3] @ rotation
        return rotation

    def _new_pixel_kalman_filter(self):
        return PixelKalmanFilter(
            dt=1.0/self.fps,  # 使用self.fps而不是fps
            process_noise_pos=self.pixel_kalman_process_noise_pos,
            process_noise_vel=self.pixel_kalman_process_noise_vel,
            measurement_noise=self.pixel_kalman_measurement_noise,
            innovation_gate=self.pixel_kalman_innovation_gate_px,
            reset_after_outliers=self.pixel_kalman_reset_after_outliers,
        )

    def configure_candidate_filter(
            self, *, enabled, min_brightness,
            max_depth_disagreement_mm, motion_scale_px):
        """Apply raw-candidate filtering/scoring for the next camera frame."""
        self.candidate_filter_enabled = bool(enabled)
        self.candidate_min_brightness = max(0.0, float(min_brightness))
        self.candidate_max_depth_disagreement_mm = max(
            0.0, float(max_depth_disagreement_mm))
        self.candidate_motion_scale_px = max(1.0, float(motion_scale_px))
        history = getattr(self, 'candidate_position_history', None)
        if history is not None:
            history.clear()

    def configure_yolo(self, enabled):
        """Enable or disable RGB YOLO without rebuilding the camera pipeline."""
        self.yolo_enabled = bool(enabled)
        if not self.yolo_enabled:
            self.last_yolo_bbox = None
            self.yolo_miss_count = 0

    def _predicted_ir_roi(self, now):
        """Return an IR-only tracking ROI derived from recent IR detections."""
        center = self._predict_candidate_center(now)
        if center is None or not np.all(np.isfinite(center)):
            return None
        margin = max(16, int(self.roi_margin))
        cx, cy = float(center[0]), float(center[1])
        return (
            max(0, int(np.floor(cx - margin))),
            max(0, int(np.floor(cy - margin))),
            min(self.image_width, int(np.ceil(cx + margin + 1))),
            min(self.image_height, int(np.ceil(cy + margin + 1))),
        )

    def _predict_candidate_center(self, now):
        history = self.candidate_position_history
        if not history:
            return None
        t1, x1, y1 = history[-1]
        predicted = np.array([x1, y1], dtype=np.float64)
        if len(history) < 2:
            return predicted
        t0, x0, y0 = history[-2]
        sample_dt = float(t1 - t0)
        if sample_dt <= 1e-4:
            return predicted
        horizon = float(np.clip(float(now) - t1, 0.0, 0.10))
        velocity = (
            np.array([x1 - x0, y1 - y0], dtype=np.float64) / sample_dt)
        return predicted + velocity * horizon

    def _select_ir_candidate(self, candidates, depth_image, now):
        """Filter implausible blobs and score all remaining IR candidates."""
        if not self.candidate_filter_enabled:
            return max(candidates, key=lambda item: item['brightness'])

        stats = self.candidate_filter_stats
        stats['frames'] += 1
        stats['candidates'] += len(candidates)
        predicted = self._predict_candidate_center(now)
        accepted = []
        for candidate in candidates:
            brightness = float(candidate['brightness'])
            if (self.candidate_min_brightness > 0.0
                    and brightness < self.candidate_min_brightness):
                stats['rejected_brightness'] += 1
                continue

            geom_depth_raw_mm = self._geometry_depth_mm(
                candidate.get('geometry_radius_px', candidate['radius']))
            geom_depth_corrected_mm = self._apply_depth_correction(
                geom_depth_raw_mm)
            # Keep the adaptive anomaly guard on the calibrated pinhole
            # quantity used by the existing experiments.  Correction models
            # are applied only when geometry is explicitly selected; this
            # prevents an old morphology-specific model from silently moving
            # the independent sensor-validity threshold.
            geom_depth_mm = (
                geom_depth_corrected_mm
                if self.depth_mode == 'geometry'
                else geom_depth_raw_mm)
            sensor_depth_mm = self._get_depth_from_depthmap(
                candidate['cx'], candidate['cy'], depth_image,
                candidate.get('mask'), candidate.get('mask_origin'),
                prefer_near_surface=(self.depth_mode == 'adaptive'),
                geometry_hint_mm=geom_depth_mm)
            depth_disagreement_mm = None
            if sensor_depth_mm is not None and geom_depth_mm is not None:
                depth_disagreement_mm = abs(
                    float(sensor_depth_mm) - float(geom_depth_mm))
                if (self.candidate_max_depth_disagreement_mm > 0.0
                        and depth_disagreement_mm
                        > self.candidate_max_depth_disagreement_mm
                        and self.depth_mode != 'adaptive'):
                    stats['rejected_depth'] += 1
                    continue

            if predicted is None:
                motion_distance_px = None
                motion_score = 0.5
            else:
                motion_distance_px = float(np.linalg.norm(
                    np.array([candidate['cx'], candidate['cy']]) - predicted))
                motion_score = math.exp(
                    -0.5 * (motion_distance_px
                            / self.candidate_motion_scale_px) ** 2)

            if self.candidate_min_brightness > 0.0:
                brightness_score = min(
                    brightness / (2.0 * self.candidate_min_brightness), 1.0)
            else:
                brightness_score = 1.0
            if (depth_disagreement_mm is None
                    or self.candidate_max_depth_disagreement_mm <= 0.0):
                depth_score = 0.25
            else:
                depth_score = max(
                    0.0,
                    1.0 - depth_disagreement_mm
                    / self.candidate_max_depth_disagreement_mm,
                )
            score = 3.0 * motion_score + 2.0 * depth_score + brightness_score
            enriched = dict(candidate)
            enriched['selection_score'] = float(score)
            enriched['selection_motion_distance_px'] = motion_distance_px
            enriched['selection_sensor_depth_mm'] = sensor_depth_mm
            enriched['selection_geom_depth_mm'] = geom_depth_mm
            enriched['selection_geom_depth_raw_mm'] = geom_depth_raw_mm
            enriched['selection_geom_depth_corrected_mm'] = (
                geom_depth_corrected_mm)
            enriched['selection_depth_disagreement_mm'] = depth_disagreement_mm
            accepted.append(enriched)

        if not accepted:
            stats['all_rejected'] += 1
            return None
        best = max(accepted, key=lambda item: item['selection_score'])
        stats['accepted'] += 1
        return best

    def _load_depth_correction_model(self, model_path='depth_correction_model.pkl'):
        """加载深度校正模型"""
        import os
        import pickle

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.depth_correction_model = pickle.load(f)
                if not isinstance(self.depth_correction_model, dict):
                    raise ValueError('depth correction model must be a dict')
                model_type = self.depth_correction_model.get('type')
                if model_type not in {
                        'linear', 'polynomial', 'piecewise_linear'}:
                    raise ValueError(
                        f'unsupported depth correction model: {model_type}')
                # Exercise the schema while loading.  The deployed model is
                # piecewise-linear; the old runtime printed it as loaded but
                # silently returned the uncorrected value for every frame.
                probe = self._apply_depth_correction(500.0)
                if probe is None or not np.isfinite(probe) or probe <= 0.0:
                    raise ValueError('depth correction model produced invalid depth')
                print(f"✓ 加载深度校正模型: {model_path}")
                print(f"  模型类型: {model_type}")
                if self.depth_correction_model.get('error') is not None:
                    print(
                        f"  训练误差: "
                        f"{float(self.depth_correction_model['error']):.1f}mm")
                if self.depth_correction_model.get('n_samples') is not None:
                    print(
                        f"  训练样本: "
                        f"{int(self.depth_correction_model['n_samples'])}个")
                self.depth_correction_model_error = None
            except Exception as e:
                print(f"⚠️  加载深度校正模型失败: {e}")
                self.depth_correction_model_error = str(e)
                self.depth_correction_model = None
        else:
            print(f"  未找到深度校正模型: {model_path}")

    def _apply_depth_correction(self, depth_raw_mm):
        """应用深度校正（仅对geometry深度）"""
        if self.depth_correction_model is None or depth_raw_mm is None:
            return depth_raw_mm

        model = self.depth_correction_model
        try:
            depth = float(depth_raw_mm)
            params = model['params']
            if model['type'] == 'linear':
                corrected = depth * float(params['scale']) + float(
                    params['offset'])
            elif model['type'] == 'polynomial':
                corrected = (
                    float(params['a']) * depth ** 2
                    + float(params['b']) * depth
                    + float(params['c']))
            elif model['type'] == 'piecewise_linear':
                branch = (
                    params['near']
                    if depth <= float(params['breakpoint'])
                    else params['far'])
                corrected = (
                    depth * float(branch['scale'])
                    + float(branch['offset']))
            else:
                return depth_raw_mm
            if np.isfinite(corrected) and corrected > 0.0:
                return float(corrected)
        except (KeyError, TypeError, ValueError, OverflowError):
            # A calibration file is optional.  A malformed one must never
            # turn an otherwise finite geometry observation into NaN.
            return depth_raw_mm
        return depth_raw_mm

    @staticmethod
    def _rs_intrinsics_to_dict(intr):
        """把 rs.intrinsics 序列化成 JSON 友好的 dict。"""
        return {
            'width': int(intr.width),
            'height': int(intr.height),
            'ppx': float(intr.ppx),
            'ppy': float(intr.ppy),
            'fx': float(intr.fx),
            'fy': float(intr.fy),
            'model': int(intr.model),  # rs.distortion 枚举值
            'coeffs': [float(c) for c in intr.coeffs],
        }

    @staticmethod
    def _dict_to_rs_intrinsics(d):
        """从 dict 还原 rs.intrinsics（含畸变模型/系数），供 rs.rs2_* 使用。"""
        intr = rs.intrinsics()
        intr.width = int(d['width'])
        intr.height = int(d['height'])
        intr.ppx = float(d['ppx'])
        intr.ppy = float(d['ppy'])
        intr.fx = float(d['fx'])
        intr.fy = float(d['fy'])
        intr.model = rs.distortion(int(d['model']))
        coeffs = list(d.get('coeffs', []))
        # rs.intrinsics.coeffs 固定长度 5
        coeffs = (coeffs + [0.0] * 5)[:5]
        intr.coeffs = [float(c) for c in coeffs]
        return intr

    def export_camera_params(self):
        """导出内参/外参/depth_scale + 完整检测配置，供录制时写
        <stem>_camera_params.json。

        只能在在线模式（已从 RealSense profile 读到内参）下调用。离线复现时
        用这些参数重建 rs.intrinsics / rs.extrinsics，保证投影数学逐位一致。

        除相机参数外，这里还序列化在线检测器的完整有效配置（IR 形态学/阈值、
        YOLO、深度邻域、深度校正模型路径等），使离线标注工具
        (tools/annotate_raw_video_hybrid.py) 能用与在线**逐参数一致**的设置复现
        检测——避免录制→离线标注 与 实物颠球 两条链路因默认值不同而产生差异。
        """
        return {
            'width': int(self.image_width),
            'height': int(self.image_height),
            'fps': int(self.fps),
            'color_width': int(getattr(
                self, 'color_stream_width', self.color_intrinsics.width)),
            'color_height': int(getattr(
                self, 'color_stream_height', self.color_intrinsics.height)),
            'depth_scale': float(self.depth_scale),
            'ball_diameter_mm': float(self.ball_diameter_mm),
            'ir_intrinsics': self._rs_intrinsics_to_dict(self.ir_intrinsics),
            'color_intrinsics': self._rs_intrinsics_to_dict(
                self.color_intrinsics),
            'depth_to_color_extrinsics': {
                'rotation': [float(r) for r in
                             self.depth_to_color_extrinsics.rotation],
                'translation': [float(t) for t in
                                self.depth_to_color_extrinsics.translation],
            },
            # --- 完整检测配置（离线复现的 single source of truth）---
            # IR 检测：阈值/自适应/形态学核/半径/圆度门（键与 ir_params 一致）。
            'ir_params': {k: (int(v) if isinstance(v, (bool, int)) else v)
                          for k, v in self.ir_params.items()},
            'brightness_percentile': float(self.brightness_percentile),
            'roi_margin': int(self.roi_margin),
            # YOLO：降频间隔 / 置信度 / 目标类别。
            'yolo_interval': int(self.yolo_interval),
            'yolo_conf': float(self.yolo_conf),
            'target_class': str(self.target_class),
            # 深度：邻域采样半径 + 单目几何深度校正模型路径。
            'depth_neighborhood_px': int(self.depth_neighborhood_px),
            'depth_mode': str(self.depth_mode),
            'sensor_surface_to_center_depth': {
                'enabled': bool(
                    self.sensor_surface_to_center_depth_enabled),
                'reference_mm': float(
                    self.SENSOR_CENTER_DEPTH_REFERENCE_MM),
                'intercept_mm': float(
                    self.SENSOR_CENTER_DEPTH_INTERCEPT_MM),
                'slope_per_100_mm': float(
                    self.SENSOR_CENTER_DEPTH_SLOPE_PER_100_MM),
                'valid_min_mm': float(
                    self.SENSOR_CENTER_DEPTH_VALID_MIN_MM),
                'valid_max_mm': float(
                    self.SENSOR_CENTER_DEPTH_VALID_MAX_MM),
            },
            'adaptive_depth': {
                'sensor_jump_mm': float(self.ADAPTIVE_SENSOR_JUMP_MM),
                'sensor_max_mm': float(self.ADAPTIVE_SENSOR_MAX_MM),
                'sensor_geometry_gap_mm': float(
                    self.ADAPTIVE_SENSOR_GEOMETRY_GAP_MM),
                'sensor_radius_mismatch_ratio': float(
                    self.ADAPTIVE_SENSOR_RADIUS_MISMATCH_RATIO),
                'near_field_window_frames': int(
                    self.ADAPTIVE_NEAR_FIELD_WINDOW_FRAMES),
                'near_field_max_sensor_mm': float(
                    self.ADAPTIVE_NEAR_FIELD_MAX_SENSOR_MM),
                'near_field_max_sensor_approach_mm': float(
                    self.ADAPTIVE_NEAR_FIELD_MAX_SENSOR_APPROACH_MM),
                'near_field_min_geometry_approach_mm': float(
                    self.ADAPTIVE_NEAR_FIELD_MIN_GEOMETRY_APPROACH_MM),
                'near_field_min_radius_growth_ratio': float(
                    self.ADAPTIVE_NEAR_FIELD_MIN_RADIUS_GROWTH_RATIO),
                'near_field_min_upward_pixel_speed_px_s': float(
                    self.ADAPTIVE_NEAR_FIELD_MIN_UPWARD_PIXEL_SPEED_PX_S),
                'near_field_max_window_s': float(
                    self.ADAPTIVE_NEAR_FIELD_MAX_WINDOW_S),
                'sensor_recovery_frames': int(
                    self.ADAPTIVE_SENSOR_RECOVERY_FRAMES),
                'prediction_max_s': float(
                    self.ADAPTIVE_PREDICTION_MAX_S),
            },
            'depth_correction_model_path': str(
                self.depth_correction_model_path),
        }

    def _init_offline_camera(self, params, width, height):
        """离线模式相机初始化：从 camera_params dict 重建内参/外参，不开相机。

        与在线路径产出完全相同的字段（ir_intrinsics/color_intrinsics/
        depth_to_color_extrinsics/T_ir_to_rgb/相机矩阵/depth_scale），因此后续
        检测与坐标变换代码无需区分在线/离线。
        """
        self.pipeline = None  # 离线无 pipeline

        self.image_width = int(params.get('width', width))
        self.image_height = int(params.get('height', height))
        self.fps = int(params.get('fps', 60))
        self.color_stream_width = int(params.get(
            'color_width', params['color_intrinsics'].get(
                'width', self.image_width)))
        self.color_stream_height = int(params.get(
            'color_height', params['color_intrinsics'].get(
                'height', self.image_height)))
        if 'depth_scale' in params:
            self.depth_scale = float(params['depth_scale'])

        # 重建 IR / RGB rs.intrinsics（含畸变模型），供 rs.rs2_* 投影使用
        self.ir_intrinsics = self._dict_to_rs_intrinsics(
            params['ir_intrinsics'])
        self.color_intrinsics = self._dict_to_rs_intrinsics(
            params['color_intrinsics'])

        self.ir_fx = self.ir_intrinsics.fx
        self.ir_fy = self.ir_intrinsics.fy
        self.ir_cx = self.ir_intrinsics.ppx
        self.ir_cy = self.ir_intrinsics.ppy

        # 相机矩阵：优先手眼标定内参（手眼标定针对 RGB），否则用 RGB 内参
        if self.calibration_data is not None:
            self.camera_matrix = self.calibration_data.camera_matrix.copy()
        else:
            self.camera_matrix = np.array([
                [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                [0, 0, 1]
            ])
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        # 重建 IR→RGB 外参（rs.extrinsics + 4x4 齐次矩阵，两处都要有）
        ext = params['depth_to_color_extrinsics']
        self.depth_to_color_extrinsics = rs.extrinsics()
        self.depth_to_color_extrinsics.rotation = [
            float(r) for r in ext['rotation']]
        self.depth_to_color_extrinsics.translation = [
            float(t) for t in ext['translation']]
        # Saved RealSense extrinsics retain librealsense's column-major flat
        # rotation layout; keep offline replay identical to the online path.
        R = _rs_extrinsics_rotation_matrix(ext['rotation'])
        t = np.array(ext['translation'], dtype=np.float64)
        self.T_ir_to_rgb = np.eye(4)
        self.T_ir_to_rgb[:3, :3] = R
        self.T_ir_to_rgb[:3, 3] = t

        print(f"[offline] IR内参: fx={self.ir_fx:.2f}, fy={self.ir_fy:.2f}, "
              f"cx={self.ir_cx:.2f}, cy={self.ir_cy:.2f}")
        print(f"[offline] RGB内参: fx={self.fx:.2f}, fy={self.fy:.2f}, "
              f"cx={self.cx:.2f}, cy={self.cy:.2f}")
        print(f"[offline] depth_scale={self.depth_scale} m/count, "
              f"输出坐标系={self.output_frame}")

        # 共享运行时状态
        self._init_runtime_state()
        print("混合检测器就绪（离线数组模式）")

    def _detect_with_yolo(self, rgb_image):
        """使用YOLO检测球，返回bbox"""
        t0 = time.time()

        # 构建推理参数
        predict_kwargs = {
            'conf': self.yolo_conf,
            'verbose': False,
            'device': self.yolo_device,
            'imgsz': self.yolo_imgsz,
        }

        results = self.yolo_model(rgb_image, **predict_kwargs)
        self.yolo_time_queue.append((time.time() - t0) * 1000)

        if len(results) == 0 or results[0].boxes is None:
            return None

        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        # 过滤目标类别
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # 调试：打印所有检测到的类别（每100帧打印一次）
        if self.debug_yolo and self.frame_count % 100 == 0 and len(class_ids) > 0:
            detected_classes = [(self.yolo_model.names[cls_id], confidences[i])
                               for i, cls_id in enumerate(class_ids)]
            print(f"[YOLO Debug] 检测到: {detected_classes}, 目标类别: '{self.target_class}'")

        ball_indices = []
        for i, cls_id in enumerate(class_ids):
            class_name = self.yolo_model.names[cls_id]
            # 只接受target_class指定的类别
            if class_name == self.target_class:
                ball_indices.append(i)

        if len(ball_indices) == 0:
            return None

        # 取置信度最高的
        best_idx = ball_indices[np.argmax(confidences[ball_indices])]
        bbox = xyxy[best_idx]  # [x1, y1, x2, y2]
        conf = confidences[best_idx]

        return {
            'bbox': bbox,
            'confidence': conf,
        }

    def _expand_bbox(self, bbox, margin):
        """扩展bbox，添加margin并限制在图像范围内"""
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1 - margin))
        y1 = max(0, int(y1 - margin))
        x2 = min(self.image_width, int(x2 + margin))
        y2 = min(self.image_height, int(y2 + margin))
        return (x1, y1, x2, y2)

    def _update_adaptive_threshold(self, ir_image, roi=None):
        """计算自适应阈值"""
        if roi is not None:
            x1, y1, x2, y2 = roi
            sample_region = ir_image[y1:y2, x1:x2]
        else:
            h, w = ir_image.shape[:2]
            sample_region = ir_image[h//4:3*h//4, w//4:3*w//4]

        if sample_region.size == 0:
            return self.ir_params['threshold']

        bg_intensity = np.percentile(sample_region, self.brightness_percentile)

        if self.background_intensity is None:
            self.background_intensity = bg_intensity
        else:
            alpha = 0.1
            self.background_intensity = alpha * bg_intensity + (1 - alpha) * self.background_intensity

        threshold = int(self.background_intensity + max(80, self.background_intensity * 0.5))
        return np.clip(threshold, 150, 240)

    def _ir_contour_is_edge_clipped(
            self, cx, cy, radius, image_shape):
        """Reject a fitted circle whose silhouette reaches the IR boundary."""
        height, width = image_shape[:2]
        margin = float(self.IR_CONTOUR_EDGE_MARGIN_PX)
        return bool(
            float(cx) - float(radius) <= margin
            or float(cy) - float(radius) <= margin
            or float(cx) + float(radius) >= float(width - 1) - margin
            or float(cy) + float(radius) >= float(height - 1) - margin)

    def _detect_in_ir(
            self, ir_image, roi=None, depth_image=None,
            observation_time=None):
        """
        在IR图像中检测球

        Args:
            ir_image: 红外图像
            roi: 可选的ROI (x1, y1, x2, y2)，None表示全图检测

        Returns:
            检测结果字典或None
        """
        # Crop before threshold/morphology so an ROI actually reduces work.
        # Candidate coordinates are translated back to full-frame IR pixels.
        offset_x = 0
        offset_y = 0
        work_ir = ir_image
        if roi is not None:
            x1, y1, x2, y2 = (int(v) for v in roi)
            x1 = max(0, min(ir_image.shape[1], x1))
            x2 = max(x1, min(ir_image.shape[1], x2))
            y1 = max(0, min(ir_image.shape[0], y1))
            y2 = max(y1, min(ir_image.shape[0], y2))
            if x2 <= x1 or y2 <= y1:
                return None
            offset_x, offset_y = x1, y1
            work_ir = ir_image[y1:y2, x1:x2]

        # 阈值分割
        if self.ir_params['use_adaptive']:
            threshold = self._update_adaptive_threshold(ir_image, roi)
        else:
            threshold = self.ir_params['threshold']

        _, binary = cv2.threshold(work_ir, threshold, 255, cv2.THRESH_BINARY)

        # 形态学处理：膨胀 → 闭运算 → 去噪
        dilate_iter = self.ir_params['dilate_iterations']
        if dilate_iter > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.dilate(binary, dilate_kernel, iterations=dilate_iter)

        close_ksize = self.ir_params['close_kernel_size']
        if close_ksize >= 3:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

        denoise_ksize = self.ir_params['denoise_kernel_size']
        if denoise_ksize >= 3:
            denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (denoise_ksize, denoise_ksize))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, denoise_kernel)

        # 连通域分析
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if roi is not None:
                roi_margin = int(math.ceil(self.IR_CONTOUR_EDGE_MARGIN_PX))
                if (bx <= roi_margin or by <= roi_margin
                        or bx + bw >= work_ir.shape[1] - roi_margin
                        or by + bh >= work_ir.shape[0] - roi_margin):
                    # A fitted centre/radius from a contour cut by the search
                    # ROI is biased.  Reject it here so _detect_core performs
                    # its immediate full-frame fallback on the same frame.
                    self.ir_roi_edge_rejected_count += 1
                    continue

            if len(cnt) >= 5:
                try:
                    # 椭圆拟合（已经是亚像素精度）
                    ellipse = cv2.fitEllipse(cnt)
                    (cx_ellipse, cy_ellipse), axes, angle = ellipse
                    minor, major = sorted(float(value) for value in axes)

                    # OpenCV does not promise that fitEllipse returns axes in
                    # minor/major order.  For an anisotropic camera, the
                    # area-equivalent radius paired with sqrt(fx*fy) is also
                    # less noisy than arbitrarily using one semi-axis.
                    radius = 0.5 * math.sqrt(minor * major)
                    if not (self.ir_params['min_radius'] < radius < self.ir_params['max_radius']):
                        continue

                    aspect_ratio = minor / (major + 1e-6)
                    if aspect_ratio < self.ir_params['circularity_min']:
                        continue

                    ellipse_area = np.pi * 0.25 * minor * major
                    area_circularity = area / (ellipse_area + 1e-6)
                    if not (0.6 < area_circularity < 1.4):
                        continue

                    # 改进：结合矩重心和椭圆中心，提高稳定性
                    # 矩重心对噪声更鲁棒，椭圆中心更精确
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        moment_cx = M['m10'] / M['m00']
                        moment_cy = M['m01'] / M['m00']
                        # 加权融合：椭圆为主(70%)，矩重心为辅(30%)
                        cx = 0.7 * cx_ellipse + 0.3 * moment_cx
                        cy = 0.7 * cy_ellipse + 0.3 * moment_cy
                    else:
                        cx = cx_ellipse
                        cy = cy_ellipse

                    # Contours are ROI-local; all externally visible positions
                    # remain in the original IR image coordinate system.
                    cx += offset_x
                    cy += offset_y
                    coarse_cx, coarse_cy = cx, cy
                    cx, cy, center_refine_shift = refine_ir_ball_center(
                        ir_image, (coarse_cx, coarse_cy), radius)
                    if self._ir_contour_is_edge_clipped(
                            cx, cy, radius, ir_image.shape):
                        self.ir_edge_rejected_count += 1
                        continue

                    # Keep only a contour-bounded mask.  The old full-frame
                    # mask caused multiple 848x480 allocations and scans for
                    # every valid candidate.
                    mask = np.zeros((bh, bw), dtype=np.uint8)
                    local_cnt = cnt - np.array([[[bx, by]]], dtype=cnt.dtype)
                    cv2.drawContours(mask, [local_cnt], -1, 255, thickness=-1)
                    candidate_ir = work_ir[by:by + bh, bx:bx + bw]
                    brightness = np.sum(candidate_ir[mask > 0])

                    candidates.append({
                        'cx': cx,
                        'cy': cy,
                        'cx_coarse': coarse_cx,
                        'cy_coarse': coarse_cy,
                        'center_refine_shift_px': center_refine_shift,
                        'radius': radius,
                        'geometry_radius_px': radius,
                        'ellipse_minor_px': minor,
                        'ellipse_major_px': major,
                        'aspect_ratio': aspect_ratio,
                        'area_circularity': area_circularity,
                        'geometry_quality_valid': bool(
                            aspect_ratio >= max(
                                0.70,
                                float(self.ir_params['circularity_min']))
                            and 0.70 <= area_circularity <= 1.30
                            and center_refine_shift <= max(
                                4.0, 0.20 * radius)),
                        'brightness': brightness,
                        'threshold': int(threshold),
                        'mask': mask,
                        'mask_origin': (offset_x + bx, offset_y + by),
                    })
                except:
                    continue

        if not candidates:
            return None

        now = (time.perf_counter() if observation_time is None
               else float(observation_time))
        best = self._select_ir_candidate(candidates, depth_image, now)
        if best is None:
            return None

        # 核心平滑：2D像素卡尔曼滤波
        # 这是最关键的一步，大幅减少静止球抖动
        best['cx_raw'] = best['cx']  # 保存原始值用于调试
        best['cy_raw'] = best['cy']
        self.candidate_position_history.append(
            (now, float(best['cx_raw']), float(best['cy_raw'])))
        if self.pixel_kalman_enabled:
            # Capture the object once so a concurrent web reconfiguration can
            # safely swap in a replacement for the following frame.
            pixel_kf = self.pixel_kf
            smoothed_pos = pixel_kf.update([best['cx'], best['cy']])
            best['cx'] = smoothed_pos[0]  # 用平滑值替换
            best['cy'] = smoothed_pos[1]

        return best

    def _get_depth_from_depthmap(
            self, cx_px, cy_px, depth_image, ball_mask=None,
            mask_origin=None, prefer_near_surface=False,
            geometry_hint_mm=None):
        """从深度图读取深度（增强稳定性，减少近距离跳变）"""
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        cx = int(round(cx_px))
        cy = int(round(cy_px))

        if not (0 <= cx < w and 0 <= cy < h):
            return None

        # 采样策略：legacy模式优先使用mask、fallback到邻域；adaptive
        # 同时保留mask近分位数和球心小窗口，利用几何深度选择更可信者。
        depths = None
        if ball_mask is not None:
            if ball_mask.shape[:2] == depth_image.shape[:2]:
                depth_region = depth_image
            elif mask_origin is not None:
                x0, y0 = (int(v) for v in mask_origin)
                mh, mw = ball_mask.shape[:2]
                if (x0 >= 0 and y0 >= 0 and x0 + mw <= w
                        and y0 + mh <= h):
                    depth_region = depth_image[y0:y0 + mh, x0:x0 + mw]
                else:
                    depth_region = None
            else:
                depth_region = None
            if (depth_region is not None
                    and depth_region.shape[:2] == ball_mask.shape[:2]):
                sel = (ball_mask > 0) & (depth_region > 0)
                if np.count_nonzero(sel) >= 5:
                    depths = depth_region[sel]

        r = self.depth_neighborhood_px
        patch = depth_image[max(0, cy-r):min(h, cy+r+1),
                           max(0, cx-r):min(w, cx+r+1)]
        center_depths = patch[patch > 0]
        if depths is None:
            depths = center_depths

        if depths.size < 3:
            return None

        # In adaptive mode the ball is known to be in front of any background
        # inside the dilated IR contour. A near percentile therefore preserves
        # the sparse foreground samples which a median can replace with the
        # wall/monitor behind the ball. Legacy sensor mode intentionally keeps
        # its historical median behavior.
        if prefer_near_surface:
            estimates = [float(np.percentile(depths, 25.0))]
            if center_depths.size >= 3:
                estimates.append(float(np.median(center_depths)))
            hint = (
                float(geometry_hint_mm)
                if geometry_hint_mm is not None
                and np.isfinite(geometry_hint_mm)
                and float(geometry_hint_mm) > 0.0 else None)
            if hint is None:
                depth_sample = min(estimates)
            else:
                depth_sample = min(
                    estimates, key=lambda value: abs(value - hint))
        else:
            depth_sample = float(np.median(depths))
        depth_raw = depth_sample * self.depth_scale * 1000.0

        if depth_raw < 100.0 or depth_raw > 5000.0:
            return None

        # 注意：时域去尖峰+平滑不在这里做，统一放到 _stabilize_depth，
        # 以同时覆盖 sensor 和 geometry 两条路径。这里只返回原始 sensor 深度。
        return depth_raw

    def _geometry_depth_mm(self, radius_px):
        """单目几何测距 Z = f*D/(2r)（用 IR 内参，与检测坐标系一致）。

        Z ∝ 1/r：球越近半径越大，同样的半径抖动相对误差越小 → 近场更稳。
        """
        if radius_px is None or radius_px <= 1e-3:
            return None
        # ``radius_px`` is the area-equivalent ellipse radius.  Pair it with
        # the geometric-mean focal length so swapping fitEllipse axes or
        # fx/fy anisotropy cannot create a range jump.
        f = math.sqrt(float(self.ir_fx) * float(self.ir_fy))
        return (f * self.ball_diameter_mm) / (2.0 * radius_px)

    def _projected_ball_radius_px(self, depth_mm):
        """Return the apparent ball radius implied by an axial depth."""
        if depth_mm is None or not np.isfinite(depth_mm) or depth_mm <= 1e-3:
            return None
        f = math.sqrt(float(self.ir_fx) * float(self.ir_fy))
        return f * self.ball_diameter_mm / (2.0 * float(depth_mm))

    def _select_depth(
            self, sensor_depth_mm, geom_depth_mm,
            observation_time=None, observed_radius_px=None,
            observed_cy_px=None,
            defer_commit=False):
        """Select or fuse stereo and monocular geometry depth.

        The morphology radius describes the connected projector-dot blob, not
        the physical ball silhouette.  Robot-attached-ball recordings show
        that using it for geometry depth creates 30--50 mm position-dependent
        errors, while valid stereo depth stays within a few millimetres.

        Returns:
            (depth_raw_mm, method_str) 或 (None, 'none')
        """
        sensor = (
            float(sensor_depth_mm)
            if sensor_depth_mm is not None and np.isfinite(sensor_depth_mm)
            else None)
        geometry = (
            float(geom_depth_mm)
            if geom_depth_mm is not None and np.isfinite(geom_depth_mm)
            else None)

        if self.depth_mode == 'adaptive':
            return self._select_adaptive_depth(
                sensor, geometry, observation_time, observed_radius_px,
                observed_cy_px,
                defer_commit=defer_commit)

        self.last_depth_selection = {
            'mode': self.depth_mode,
            'method': None,
            'sensor_weight': None,
            'disagreement_mm': (
                abs(sensor - geometry)
                if sensor is not None and geometry is not None else None),
            'sensor_rejected': False,
            'rejection_reason': None,
            'sensor_jump_mm': None,
            'observed_radius_px': None,
            'sensor_projected_radius_px': None,
            'radius_mismatch_ratio': None,
            'near_field_scale_stall': False,
            'near_field_window_samples': 0,
            'near_field_window_span_s': None,
            'near_field_sensor_approach_mm': None,
            'near_field_geometry_approach_mm': None,
            'near_field_radius_growth_ratio': None,
            'near_field_upward_pixel_speed_px_s': None,
            'prediction_mm': None,
            'prediction_residual_mm': None,
            'prediction_horizon_s': None,
            'sensor_recovery_count': 0,
        }
        if self.depth_mode == 'geometry':
            if geometry is not None:
                self.depth_source_state = 'geometry'
                self.last_depth_selection['method'] = 'geometry'
                return geometry, 'geometry'
            if sensor is not None:
                self.depth_source_state = 'sensor_fallback'
                self.last_depth_selection['method'] = 'sensor_fallback'
                return sensor, 'sensor'
        else:
            if sensor is not None:
                self.depth_source_state = 'sensor'
                self.last_depth_selection['method'] = 'sensor'
                return sensor, 'sensor'
            if geometry is not None:
                self.depth_source_state = 'geometry_fallback'
                self.last_depth_selection['method'] = 'geometry_fallback'
                return geometry, 'geometry'
        return None, 'none'

    def _adaptive_depth_time(self, observation_time):
        if observation_time is not None:
            try:
                value = float(observation_time)
            except (TypeError, ValueError):
                value = float('nan')
            if np.isfinite(value):
                return value
        self.adaptive_frame_index += 1
        return self.adaptive_frame_index / max(float(self.fps), 1.0)

    def _predict_adaptive_sensor_depth(self, now):
        history = list(self.adaptive_sensor_history)
        if not history:
            return None, None
        last_time, last_depth = history[-1]
        horizon = max(0.0, float(now - last_time))
        if horizon > self.ADAPTIVE_PREDICTION_MAX_S:
            return None, horizon
        velocity = 0.0
        if len(history) >= 2:
            times = np.asarray([row[0] for row in history], dtype=np.float64)
            depths = np.asarray([row[1] for row in history], dtype=np.float64)
            centered = times - float(np.mean(times))
            denominator = float(centered @ centered)
            if denominator > 1e-8:
                velocity = float(
                    centered @ (depths - float(np.mean(depths)))
                    / denominator)
                velocity = float(np.clip(
                    velocity,
                    -self.ADAPTIVE_MAX_DEPTH_SPEED_MM_S,
                    self.ADAPTIVE_MAX_DEPTH_SPEED_MM_S,
                ))
        prediction = max(100.0, float(last_depth + velocity * horizon))
        return prediction, horizon

    def _adaptive_fallback_depth(self, prediction, geometry):
        if prediction is not None:
            depth = float(prediction)
            method = 'adaptive_prediction'
            if geometry is not None:
                correction = float(np.clip(
                    geometry - prediction,
                    -self.ADAPTIVE_GEOMETRY_CORRECTION_LIMIT_MM,
                    self.ADAPTIVE_GEOMETRY_CORRECTION_LIMIT_MM,
                ))
                depth += self.ADAPTIVE_GEOMETRY_CORRECTION_GAIN * correction
                method = 'adaptive_prediction_geometry'
            return max(100.0, depth), method
        if geometry is not None:
            return geometry, 'adaptive_geometry_fallback'
        return None, 'none'

    def _near_field_scale_stall_diagnostics(
            self, now, sensor, geometry, observed_radius_px,
            observed_cy_px):
        """Detect the short precursor to a near-range stereo depth jump.

        Three independent trends must agree over three causal camera frames.
        This keeps the window near 22 ms at 90 Hz and prevents a noisy IR
        radius near the true trajectory apex from rejecting valid stereo.
        """
        diagnostics = {
            'near_field_scale_stall': False,
            'near_field_window_samples': 0,
            'near_field_window_span_s': None,
            'near_field_sensor_approach_mm': None,
            'near_field_geometry_approach_mm': None,
            'near_field_radius_growth_ratio': None,
            'near_field_upward_pixel_speed_px_s': None,
        }
        values = (sensor, geometry, observed_radius_px, observed_cy_px)
        if any(value is None or not np.isfinite(value) for value in values):
            return False, diagnostics, None
        radius = float(observed_radius_px)
        if radius <= 1.0e-3:
            return False, diagnostics, None
        sample = (
            float(now), float(sensor), float(geometry), radius,
            float(observed_cy_px))
        window = list(self.adaptive_near_field_history) + [sample]
        diagnostics['near_field_window_samples'] = len(window)
        if len(window) < self.ADAPTIVE_NEAR_FIELD_WINDOW_FRAMES:
            return False, diagnostics, sample

        oldest = window[-self.ADAPTIVE_NEAR_FIELD_WINDOW_FRAMES]
        newest = window[-1]
        span_s = float(newest[0] - oldest[0])
        sensor_approach_mm = float(oldest[1] - newest[1])
        geometry_approach_mm = float(oldest[2] - newest[2])
        radius_growth_ratio = float(newest[3] / oldest[3] - 1.0)
        upward_pixel_speed_px_s = (
            float(oldest[4] - newest[4]) / span_s
            if span_s > 1.0e-6 else 0.0)
        diagnostics.update({
            'near_field_window_span_s': span_s,
            'near_field_sensor_approach_mm': sensor_approach_mm,
            'near_field_geometry_approach_mm': geometry_approach_mm,
            'near_field_radius_growth_ratio': radius_growth_ratio,
            'near_field_upward_pixel_speed_px_s': (
                upward_pixel_speed_px_s),
        })
        detected = bool(
            1.0e-6 < span_s <= self.ADAPTIVE_NEAR_FIELD_MAX_WINDOW_S
            and newest[1] <= self.ADAPTIVE_NEAR_FIELD_MAX_SENSOR_MM
            and sensor_approach_mm
            <= self.ADAPTIVE_NEAR_FIELD_MAX_SENSOR_APPROACH_MM
            and geometry_approach_mm
            >= self.ADAPTIVE_NEAR_FIELD_MIN_GEOMETRY_APPROACH_MM
            and radius_growth_ratio
            >= self.ADAPTIVE_NEAR_FIELD_MIN_RADIUS_GROWTH_RATIO
            and upward_pixel_speed_px_s
            >= self.ADAPTIVE_NEAR_FIELD_MIN_UPWARD_PIXEL_SPEED_PX_S)
        diagnostics['near_field_scale_stall'] = detected
        return detected, diagnostics, sample

    def _select_adaptive_depth(
            self, sensor, geometry, observation_time,
            observed_radius_px=None, observed_cy_px=None, *,
            defer_commit=False):
        """Use stereo unless any configured anomaly check marks it invalid.

        During a short stereo failure, predict camera-axis depth from the last
        trusted sensor samples and weakly correct that prediction with noisy
        monocular geometry.  The checks are deliberately independent:
        frame-to-frame sensor jump, absolute sensor maximum, disagreement
        with geometry, IR-radius/projected-radius disagreement, and a
        three-frame near-field scale stall.  Geometry never replaces a sensor
        sample which passes all checks merely because the ball is close.
        """
        now = self._adaptive_depth_time(observation_time)
        state_snapshot = {
            'anomaly_active': bool(self.adaptive_sensor_anomaly_active),
            'recovery_count': int(self.adaptive_sensor_recovery_count),
            'last_raw_sensor_mm': self.adaptive_last_raw_sensor_mm,
            'last_geometry_mm': self.adaptive_last_geometry_mm,
            'last_output_depth_mm': self.adaptive_last_output_depth_mm,
            'depth_source_state': self.depth_source_state,
            'near_field_history': list(self.adaptive_near_field_history),
        }
        self._pending_adaptive_sensor_sample = None
        self._pending_adaptive_near_field_sample = None
        self._pending_adaptive_state_snapshot = None
        prediction, prediction_horizon = (
            self._predict_adaptive_sensor_depth(now))
        disagreement = (
            abs(sensor - geometry)
            if sensor is not None and geometry is not None else None)
        # Compare with the immediately preceding raw sensor frame, including
        # rejected frames.  This implements a literal consecutive-frame jump
        # test and also lets a sensor which has settled recover cleanly.
        previous_sensor = self.adaptive_last_raw_sensor_mm
        sensor_jump = (
            abs(sensor - previous_sensor)
            if sensor is not None and previous_sensor is not None else None)
        self.adaptive_last_raw_sensor_mm = sensor
        prediction_residual = (
            abs(sensor - prediction)
            if sensor is not None and prediction is not None else None)
        observed_radius = (
            float(observed_radius_px)
            if (observed_radius_px is not None
                and np.isfinite(observed_radius_px)
                and float(observed_radius_px) > 1e-3)
            else None)
        sensor_projected_radius = self._projected_ball_radius_px(sensor)
        radius_mismatch_ratio = None
        if observed_radius is not None and sensor_projected_radius is not None:
            radius_mismatch_ratio = (
                abs(observed_radius - sensor_projected_radius)
                / max(observed_radius, sensor_projected_radius))
        (
            near_field_scale_bad,
            near_field_diagnostics,
            near_field_sample,
        ) = self._near_field_scale_stall_diagnostics(
            now, sensor, geometry, observed_radius, observed_cy_px)
        jump_bad = (
            sensor_jump is not None
            and sensor_jump > self.ADAPTIVE_SENSOR_JUMP_MM)
        sensor_too_far = (
            sensor is not None and sensor > self.ADAPTIVE_SENSOR_MAX_MM)
        geometry_bad = (
            disagreement is not None
            and disagreement > self.ADAPTIVE_SENSOR_GEOMETRY_GAP_MM)
        radius_bad = (
            radius_mismatch_ratio is not None
            and radius_mismatch_ratio
            > self.ADAPTIVE_SENSOR_RADIUS_MISMATCH_RATIO)
        detected_anomaly = bool(
            jump_bad or sensor_too_far or geometry_bad or radius_bad
            or near_field_scale_bad)

        reasons = []
        if jump_bad:
            reasons.append('sensor_jump')
        if sensor_too_far:
            reasons.append('sensor_over_1000mm')
        if geometry_bad:
            reasons.append('geometry_disagreement')
        if radius_bad:
            reasons.append('radius_disagreement')
        if near_field_scale_bad:
            reasons.append('near_field_scale_stall')

        if sensor is None:
            detected_anomaly = True
            reasons.append('sensor_unavailable')

        if self.adaptive_sensor_anomaly_active:
            recovery_good = bool(sensor is not None and not detected_anomaly)
            if recovery_good:
                self.adaptive_sensor_recovery_count += 1
            else:
                self.adaptive_sensor_recovery_count = 0
            if (self.adaptive_sensor_recovery_count
                    >= self.ADAPTIVE_SENSOR_RECOVERY_FRAMES):
                self.adaptive_sensor_anomaly_active = False
                self.adaptive_sensor_recovery_count = 0
            else:
                detected_anomaly = True
                if recovery_good:
                    reasons.append('sensor_recovering')
        elif detected_anomaly:
            self.adaptive_sensor_anomaly_active = True
            self.adaptive_sensor_recovery_count = 0

        if detected_anomaly:
            # Recovery must build a new consecutive scale window; otherwise a
            # pre-fault sample could be combined with a post-gap measurement.
            self.adaptive_near_field_history.clear()

        if not detected_anomaly and sensor is not None:
            if defer_commit:
                self._pending_adaptive_sensor_sample = (now, sensor)
                self._pending_adaptive_near_field_sample = near_field_sample
                self._pending_adaptive_state_snapshot = state_snapshot
            else:
                self.adaptive_sensor_history.append((now, sensor))
                if near_field_sample is not None:
                    self.adaptive_near_field_history.append(near_field_sample)
            depth_mm = sensor
            method = 'adaptive_sensor'
            sensor_weight = 1.0
            self.depth_source_state = 'sensor'
        else:
            depth_mm, method = self._adaptive_fallback_depth(
                prediction, geometry)
            sensor_weight = 0.0
            self.depth_source_state = (
                'prediction' if prediction is not None else 'geometry')

        self.adaptive_last_geometry_mm = geometry
        self.adaptive_last_output_depth_mm = depth_mm
        rejection_reason = ','.join(dict.fromkeys(reasons)) or None
        self.last_depth_selection = {
            'mode': 'adaptive', 'method': method,
            'sensor_weight': sensor_weight,
            'disagreement_mm': disagreement,
            'sensor_rejected': bool(detected_anomaly),
            'rejection_reason': rejection_reason,
            'sensor_jump_mm': sensor_jump,
            'observed_radius_px': observed_radius,
            'sensor_projected_radius_px': sensor_projected_radius,
            'radius_mismatch_ratio': radius_mismatch_ratio,
            **near_field_diagnostics,
            'prediction_mm': prediction,
            'prediction_residual_mm': prediction_residual,
            'prediction_horizon_s': prediction_horizon,
            'sensor_recovery_count': int(
                self.adaptive_sensor_recovery_count),
        }
        return depth_mm, method

    def _finalize_adaptive_depth_selection(self, accepted):
        """Commit a sensor sample only after the complete 3D frame passes.

        Adaptive depth selection runs before backprojection and the motion
        gate.  Deferring this write prevents a wrong IR candidate with a
        plausible depth from contaminating the trusted depth predictor.
        """
        sample = self._pending_adaptive_sensor_sample
        near_field_sample = self._pending_adaptive_near_field_sample
        snapshot = self._pending_adaptive_state_snapshot
        self._pending_adaptive_sensor_sample = None
        self._pending_adaptive_near_field_sample = None
        self._pending_adaptive_state_snapshot = None
        if sample is None:
            return
        if bool(accepted):
            self.adaptive_sensor_history.append(sample)
            if near_field_sample is not None:
                self.adaptive_near_field_history.append(near_field_sample)
            return
        if snapshot is None:
            return
        self.adaptive_sensor_anomaly_active = bool(
            snapshot['anomaly_active'])
        self.adaptive_sensor_recovery_count = int(
            snapshot['recovery_count'])
        self.adaptive_last_raw_sensor_mm = snapshot['last_raw_sensor_mm']
        self.adaptive_last_geometry_mm = snapshot['last_geometry_mm']
        self.adaptive_last_output_depth_mm = snapshot[
            'last_output_depth_mm']
        self.depth_source_state = snapshot['depth_source_state']
        self.adaptive_near_field_history.clear()
        self.adaptive_near_field_history.extend(
            snapshot.get('near_field_history', ()))

    def _stabilize_depth(self, depth_raw, *, commit=True):
        """深度时域稳定化：去尖峰(Hampel 软钳制) + 中位数平滑。

        统一作用于 sensor / geometry 两条路径，压近距离深度噪声导致的左图
        重投影像素与终端位置/速度跳变。软钳制而非丢弃：超限时限制到门限
        距离，既压尖峰又保留跟踪。窗口大小由 depth_history 的 maxlen 控制。
        """
        hist = self.depth_history
        if len(hist) >= 3:
            med = float(np.median(list(hist)))
            gate = max(150.0, 0.25 * med)  # 60fps下单帧真实深度位移<此值
            dev = depth_raw - med
            if abs(dev) > gate:
                depth_raw = med + np.sign(dev) * gate
        preview = list(hist)
        preview.append(depth_raw)
        if commit:
            hist.append(depth_raw)
        if len(preview) >= 3:
            return float(np.median(preview[-int(hist.maxlen):]))
        return depth_raw

    def _depth_for_reconstruction(self, depth_raw, *, commit=True):
        """Select same-frameset raw depth or legacy temporal median depth."""
        if bool(getattr(self, 'synchronized_depth_enabled', False)):
            return float(depth_raw)
        return self._stabilize_depth(depth_raw, commit=commit)

    def _backproject_to_3d(self, cx_px, cy_px, depth_mm):
        """反投影到IR 3D坐标系（使用IR内参）"""
        Z = depth_mm
        X = (cx_px - self.ir_cx) * Z / self.ir_fx
        Y = (cy_px - self.ir_cy) * Z / self.ir_fy
        return (X, Y, Z)

    def _transform_ir_3d_to_rgb_3d(self, point_ir):
        """
        将IR相机坐标系的3D点转换到RGB相机坐标系（使用齐次变换矩阵）

        关键：手眼标定是针对RGB相机的，所以最终的3D点必须在RGB坐标系下

        Args:
            point_ir: IR相机坐标系的3D点 [X, Y, Z] (mm)

        Returns:
            RGB相机坐标系的3D点 [X, Y, Z] (mm)
        """
        # 转换为米（齐次变换矩阵中的平移单位是米）
        point_ir_m = np.array(point_ir) / 1000.0

        # 构建齐次坐标 [X, Y, Z, 1]
        point_ir_homo = np.append(point_ir_m, 1.0)

        # 齐次变换：P_rgb = T_ir_to_rgb @ P_ir
        point_rgb_homo = self.T_ir_to_rgb @ point_ir_homo

        # 提取3D坐标并转回毫米
        point_rgb = point_rgb_homo[:3] * 1000.0
        return point_rgb.tolist()

    def _project_rgb_3d_to_pixel(self, point_rgb):
        """
        将RGB相机坐标系的3D点投影到RGB图像像素（用于可视化）

        关键：这个函数保证可视化的绿圈位置与3D位置数据完全一致

        Args:
            point_rgb: RGB相机坐标系的3D点 [X, Y, Z] (mm)

        Returns:
            (rgb_x, rgb_y): RGB图像坐标（像素），如果投影失败返回None
        """
        try:
            # 转换为米（RealSense内参使用米）
            point_rgb_m = np.array(point_rgb) / 1000.0

            # 检查深度是否有效
            if point_rgb_m[2] <= 0 or point_rgb_m[2] > 5.0:
                return None

            # 使用RGB内参投影到像素：P_pixel = K @ P_3d
            # 使用RealSense的投影函数（包含畸变校正）
            pixel_rgb = rs.rs2_project_point_to_pixel(
                self.color_intrinsics, point_rgb_m.tolist())

            # 检查是否在图像范围内
            if (0 <= pixel_rgb[0] < self.image_width and
                0 <= pixel_rgb[1] < self.image_height):
                return (pixel_rgb[0], pixel_rgb[1])
            else:
                return None

        except Exception as e:
            return None

    def _transform_to_base_link(self, position_camera):
        """转换到base_link坐标系"""
        if self.T_base_camera is None:
            return position_camera

        pos_camera_m = np.array(position_camera) / 1000.0
        pos_homo = np.append(pos_camera_m, 1.0)
        pos_base_homo = self.T_base_camera @ pos_homo
        return (pos_base_homo[:3] * 1000.0).tolist()

    @staticmethod
    def _robust_line_fit(relative_time, values):
        """Small causal Huber IRLS fit returning slope/intercept/residuals."""
        t = np.asarray(relative_time, dtype=np.float64).reshape(-1)
        y = np.asarray(values, dtype=np.float64).reshape(-1)
        design = np.column_stack((t, np.ones_like(t)))
        coefficient, *_ = np.linalg.lstsq(design, y, rcond=None)
        for _ in range(4):
            residual = y - design @ coefficient
            median = float(np.median(residual))
            mad = float(np.median(np.abs(residual - median)))
            scale = max(1.0, 1.4826 * mad)
            cutoff = 1.5 * scale
            abs_residual = np.abs(residual)
            weights = np.ones_like(abs_residual)
            mask = abs_residual > cutoff
            weights[mask] = cutoff / np.maximum(abs_residual[mask], 1.0e-9)
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_values = y * np.sqrt(weights)
            updated, *_ = np.linalg.lstsq(
                weighted_design, weighted_values, rcond=None)
            if np.linalg.norm(updated - coefficient) < 1.0e-9:
                coefficient = updated
                break
            coefficient = updated
        return coefficient, y - design @ coefficient

    def _calculate_physics_constrained_velocity(self, history):
        """Fit x/y linearly and z with fixed gravity over one flight segment."""
        _, velocity = self._fit_trajectory_state(history, gravity=True)
        if not np.all(np.isfinite(velocity)):
            return np.zeros(3, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        if speed > self.MOTION_GATE_MAX_SPEED_MM_S:
            velocity *= self.MOTION_GATE_MAX_SPEED_MM_S / max(speed, 1.0e-9)
        return velocity

    def _fit_trajectory_state(self, history, *, gravity):
        """Return causal endpoint position/velocity from timestamped samples."""
        timestamps = np.asarray(
            [float(sample[0]) for sample in history], dtype=np.float64)
        positions = np.asarray(
            [sample[1] for sample in history], dtype=np.float64)
        if (len(timestamps) < 2 or positions.ndim != 2
                or positions.shape[1] != 3
                or not np.all(np.isfinite(timestamps))
                or not np.all(np.isfinite(positions))):
            fallback = (
                positions[-1].copy()
                if positions.ndim == 2 and len(positions)
                else np.zeros(3, dtype=np.float64))
            return fallback, np.zeros(3, dtype=np.float64)
        relative_time = timestamps - timestamps[-1]
        endpoint = np.zeros(3, dtype=np.float64)
        velocity = np.zeros(3, dtype=np.float64)
        for axis in (0, 1):
            coefficient, _ = self._robust_line_fit(
                relative_time, positions[:, axis])
            velocity[axis] = coefficient[0]
            endpoint[axis] = coefficient[1]
        gravity_compensated_z = positions[:, 2].copy()
        if gravity:
            gravity_compensated_z += (
                0.5 * float(self.PHYSICS_VELOCITY_GRAVITY_MM_S2)
                * relative_time ** 2)
        coefficient, _ = self._robust_line_fit(
            relative_time, gravity_compensated_z)
        velocity[2] = coefficient[0]
        endpoint[2] = coefficient[1]
        return endpoint, velocity

    def _racket_context(self, now, ball_position, *, max_age_s):
        state = getattr(self, 'latest_racket_state', None)
        if state is None:
            return None
        age = float(now - state[0])
        if abs(age) > float(max_age_s):
            return None
        racket_position = np.asarray(state[1], dtype=np.float64).reshape(3)
        racket_velocity = np.asarray(state[2], dtype=np.float64).reshape(3)
        return {
            'age_s': age,
            'position': racket_position,
            'velocity': racket_velocity,
            'distance_mm': float(np.linalg.norm(
                np.asarray(ball_position, dtype=np.float64).reshape(3)
                - racket_position)),
        }

    def _fresh_racket_context(self, now, ball_position):
        return self._racket_context(
            now, ball_position, max_age_s=self.RACKET_STATE_MAX_AGE_S)

    def _history_is_supported(self, history, now):
        """Detect a ball carried by the racket, not a free-flight apex."""
        if len(history) < self.PHYSICS_VELOCITY_MIN_FRAMES:
            return False, None, None
        endpoint, unconstrained_velocity = self._fit_trajectory_state(
            history, gravity=False)
        positions = np.asarray([sample[1] for sample in history])
        centre = np.median(positions, axis=0)
        span = float(np.max(np.linalg.norm(positions - centre, axis=1)))
        racket = self._fresh_racket_context(now, history[-1][1])
        if racket is not None:
            relative_speed = float(np.linalg.norm(
                unconstrained_velocity - racket['velocity']))
            supported = bool(
                racket['distance_mm'] <= self.SUPPORTED_RACKET_DISTANCE_MM
                and relative_speed <= self.SUPPORTED_RELATIVE_SPEED_MM_S)
            return supported, endpoint, racket
        supported = bool(
            span <= self.SUPPORTED_NO_RACKET_SPAN_MM
            and float(np.linalg.norm(unconstrained_velocity))
            <= self.SUPPORTED_NO_RACKET_SPEED_MM_S)
        return supported, endpoint, None

    def _supported_exit_confirmed(self, now, position):
        if not self.position_history:
            return True
        previous_time, previous_position = self.position_history[-1]
        dt = float(now - previous_time)
        if dt <= 1.0e-4:
            return False
        pair_velocity = (
            np.asarray(position, dtype=np.float64)
            - np.asarray(previous_position, dtype=np.float64)) / dt
        racket = self._fresh_racket_context(now, position)
        if racket is not None:
            return bool(
                racket['distance_mm']
                > self.SUPPORTED_RACKET_DISTANCE_MM + 20.0
                or np.linalg.norm(pair_velocity - racket['velocity'])
                > self.SUPPORTED_RELATIVE_SPEED_MM_S + 200.0)
        anchor = self.last_position_estimate
        displacement = (
            float(np.linalg.norm(np.asarray(position) - anchor))
            if anchor is not None else 0.0)
        return bool(
            displacement > 2.5
            and np.linalg.norm(pair_velocity) > 200.0)

    def _bounded_endpoint_position(self, raw_position, fitted_position):
        raw = np.asarray(raw_position, dtype=np.float64).reshape(3)
        fitted = np.asarray(fitted_position, dtype=np.float64).reshape(3)
        correction = fitted - raw
        correction_norm = float(np.linalg.norm(correction))
        limit = self.TRAJECTORY_POSITION_MAX_CORRECTION_MM
        if correction_norm > limit:
            correction *= limit / max(correction_norm, 1.0e-9)
        return raw + correction

    def _commit_velocity_output(
            self, now, raw_position, fitted_position, velocity, *,
            ready, motion_state, velocity_model,
            velocity_correction=None):
        raw = np.asarray(raw_position, dtype=np.float64).reshape(3)
        fitted = np.asarray(fitted_position, dtype=np.float64).reshape(3)
        if self.trajectory_position_smoothing_enabled:
            output_position = self._bounded_endpoint_position(raw, fitted)
        else:
            output_position = raw.copy()
        if (self.trajectory_position_smoothing_enabled
                and self.last_position_estimate is not None
                and self.last_velocity_output_time is not None):
            dt = float(now - self.last_velocity_output_time)
            if 1.0e-5 < dt <= self.MOTION_GATE_MAX_REACQUISITION_DT_S:
                step = output_position - self.last_position_estimate
                step_norm = float(np.linalg.norm(step))
                max_step = (
                    self.TRAJECTORY_POSITION_MAX_OUTPUT_SPEED_MM_S * dt)
                if step_norm > max_step:
                    output_position = (
                        self.last_position_estimate
                        + step * (max_step / max(step_norm, 1.0e-9)))
        output_velocity = np.asarray(
            velocity, dtype=np.float64).reshape(3).copy()
        speed = float(np.linalg.norm(output_velocity))
        if speed > self.MOTION_GATE_MAX_SPEED_MM_S:
            output_velocity *= self.MOTION_GATE_MAX_SPEED_MM_S / speed
        self.last_position_estimate = output_position.copy()
        self.last_position_filter_residual_mm = float(np.linalg.norm(
            output_position - raw))
        self.last_velocity_output = output_velocity.copy()
        self.last_velocity_output_time = float(now)
        self.last_velocity_ready = bool(ready)
        self.velocity_motion_state = str(motion_state)
        self.velocity_model = str(velocity_model)
        self.last_velocity_filter_correction_mm_s = (
            None if velocity_correction is None
            else float(np.linalg.norm(velocity_correction)))
        return output_velocity.tolist()

    def _stabilize_flight_velocity(self, now, fitted_velocity):
        fitted = np.asarray(fitted_velocity, dtype=np.float64).reshape(3)
        self.last_velocity_restart_bridge_used = False
        restart_bridge = False
        if (self.last_velocity_ready
                and self.last_velocity_output_time is not None
                and self.velocity_motion_state in {
                    'flight', 'post_impact_fast', 'launch_fast'}):
            reference_time = float(self.last_velocity_output_time)
            reference_velocity = self.last_velocity_output.copy()
            max_gap_s = self.PHYSICS_VELOCITY_MAX_GAP_S
        elif (self.velocity_restart_reference_time is not None
                and self.velocity_restart_reference_velocity is not None):
            # A loss boundary clears the five-frame position window, but the
            # last complete flight velocity remains a valid causal prior.  Use
            # it exactly once when the restarted estimator first becomes ready;
            # otherwise its noisy first fit is exposed as a zero -> fitted
            # impulse (the raw detector component of frame 000234).
            reference_time = float(self.velocity_restart_reference_time)
            reference_velocity = np.asarray(
                self.velocity_restart_reference_velocity,
                dtype=np.float64).reshape(3).copy()
            max_gap_s = self.MOTION_GATE_MAX_REACQUISITION_DT_S
            restart_bridge = True
        else:
            return fitted, None
        dt = float(now - reference_time)
        if not 1.0e-5 < dt <= max_gap_s:
            if restart_bridge:
                self.velocity_restart_reference_time = None
                self.velocity_restart_reference_velocity = None
            return fitted, None
        predicted = reference_velocity
        predicted[2] -= self.PHYSICS_VELOCITY_GRAVITY_MM_S2 * dt
        residual = fitted - predicted
        update_gain = float(self.FLIGHT_VELOCITY_UPDATE_GAIN)
        if bool(getattr(
                self, 'FLIGHT_VELOCITY_ADAPTIVE_GAIN_ENABLED', False)):
            minimum_gain = min(
                update_gain,
                max(0.0, float(self.FLIGHT_VELOCITY_MIN_UPDATE_GAIN)))
            full_gain_residual = max(
                1.0,
                float(self.FLIGHT_VELOCITY_FULL_GAIN_RESIDUAL_MM_S))
            residual_ratio = float(np.clip(
                np.linalg.norm(residual) / full_gain_residual, 0.0, 1.0))
            # Smoothstep has zero slope at both ends, avoiding a gain kink in
            # the velocity output when a residual crosses the tuning scale.
            blend = residual_ratio * residual_ratio * (
                3.0 - 2.0 * residual_ratio)
            update_gain = minimum_gain + (
                update_gain - minimum_gain) * blend
        correction = update_gain * residual
        correction_norm = float(np.linalg.norm(correction))
        limit = self.FLIGHT_VELOCITY_MAX_CORRECTION_MM_S
        if correction_norm > limit:
            correction *= limit / max(correction_norm, 1.0e-9)
        if restart_bridge:
            self.velocity_restart_reference_time = None
            self.velocity_restart_reference_velocity = None
            self.last_velocity_restart_bridge_used = True
            self.velocity_restart_bridge_count += 1
        return predicted + correction, correction

    def _impact_candidate(self, now, position):
        self.last_impact_candidate_rejection_reason = None
        self.last_impact_pair_vertical_velocity_mm_s = None
        self.last_impact_previous_vertical_velocity_mm_s = None
        self.last_impact_evidence_source = None
        self.last_impact_racket_state_age_s = None
        self.last_impact_racket_distance_mm = None
        self.last_impact_racket_gate_used = False

        def reject(reason):
            self.last_impact_candidate_rejection_reason = str(reason)
            return None

        if not self.impact_velocity_reset_enabled:
            return reject('disabled')
        if self.velocity_motion_state != 'flight':
            return reject('not_in_flight')
        if not self.last_velocity_ready:
            return reject('velocity_not_ready')
        if len(self.position_history) < 2:
            return reject('history_too_short')
        if self.observation_gap_active:
            return reject('observation_gap')
        previous_time, previous_position = self.position_history[-1]
        dt = float(now - previous_time)
        if not 1.0e-4 < dt <= self.IMPACT_CONFIRM_MAX_GAP_S:
            return reject('sample_gap')
        previous_velocity = self.last_velocity_output.copy()
        raw_position = np.asarray(position, dtype=np.float64).reshape(3)
        pair_velocity = (
            raw_position
            - np.asarray(previous_position, dtype=np.float64)) / dt
        evidence_position = raw_position
        evidence_source = 'raw_3d'
        if (self.last_position_estimate is not None
                and self.last_velocity_output_time is not None
                and abs(float(previous_time)
                        - float(self.last_velocity_output_time))
                <= 1.0e-5):
            preview = list(self.position_history)
            preview.append((float(now), raw_position.tolist()))
            preview = preview[-self._velocity_history_maxlen():]
            preview_endpoint, _ = self._fit_trajectory_state(
                preview, gravity=True)
            filtered_position = self._bounded_endpoint_position(
                raw_position, preview_endpoint)
            filtered_pair_velocity = (
                filtered_position - self.last_position_estimate) / dt
            if filtered_pair_velocity[2] > pair_velocity[2]:
                pair_velocity = filtered_pair_velocity
                evidence_position = filtered_position
                evidence_source = 'causal_endpoint'
        self.last_impact_pair_vertical_velocity_mm_s = float(
            pair_velocity[2])
        self.last_impact_previous_vertical_velocity_mm_s = float(
            previous_velocity[2])
        self.last_impact_evidence_source = evidence_source
        refractory_ok = (
            self.last_velocity_impact_time is None
            or float(now - self.last_velocity_impact_time)
            >= self.impact_refractory_s)
        if not refractory_ok:
            return reject('refractory')
        racket = self._fresh_racket_context(now, evidence_position)
        fresh_racket = racket is not None
        if racket is None:
            racket = self._racket_context(
                now, evidence_position,
                max_age_s=self.RACKET_STATE_FALLBACK_MAX_AGE_S)
        racket_gate_used = racket is not None
        self.last_impact_racket_gate_used = racket_gate_used
        racket_distance = (
            None if racket is None else float(racket['distance_mm']))
        self.last_impact_racket_distance_mm = racket_distance
        self.last_impact_racket_state_age_s = (
            None if racket is None else float(racket['age_s']))
        racket_limit = (
            self.IMPACT_MAX_RACKET_DISTANCE_MM
            if fresh_racket else self.IMPACT_RECENT_RACKET_DISTANCE_MM)
        if racket is not None and racket_distance > racket_limit:
            return reject('racket_too_far')
        if (racket is None
                and bool(getattr(self, 'impact_racket_gate_required', False))):
            return reject('racket_state_unavailable')
        threshold_scale = (
            1.0 if racket is not None
            else self.IMPACT_NO_RACKET_THRESHOLD_SCALE)
        if (previous_velocity[2]
                > -threshold_scale
                * self.impact_min_downward_velocity_mm_s):
            return reject('insufficient_downward_velocity')
        if (pair_velocity[2]
                < threshold_scale * self.impact_min_upward_velocity_mm_s):
            return reject('insufficient_upward_velocity')
        if (pair_velocity[2] - previous_velocity[2]
                < threshold_scale * self.impact_min_velocity_jump_mm_s):
            return reject('insufficient_velocity_jump')
        if np.linalg.norm(pair_velocity) > self.MOTION_GATE_MAX_SPEED_MM_S:
            return reject('pair_speed_limit')
        self.last_impact_candidate_rejection_reason = None
        return {
            'pre_time': float(previous_time),
            'pre_position': np.asarray(
                previous_position, dtype=np.float64).copy(),
            'first_time': float(now),
            'first_position': np.asarray(position, dtype=np.float64).copy(),
            'first_evidence_position': evidence_position.copy(),
            'pre_velocity': previous_velocity,
            'first_pair_velocity': pair_velocity,
            'racket_gate_used': racket_gate_used,
            'racket_distance_mm': racket_distance,
            'racket_state_age_s': (
                None if racket is None else float(racket['age_s'])),
            'evidence_source': evidence_source,
        }

    def _confirm_pending_impact(self, now, position):
        pending = self.pending_impact
        if pending is None:
            return False
        dt = float(now - pending['first_time'])
        current = np.asarray(position, dtype=np.float64).reshape(3)
        if not 1.0e-4 < dt <= self.IMPACT_CONFIRM_MAX_GAP_S:
            return False
        pair_velocity = (current - pending['first_position']) / dt
        total_rise = float(current[2] - pending['pre_position'][2])
        if pending['racket_gate_used']:
            racket = self._racket_context(
                now, current,
                max_age_s=self.RACKET_STATE_FALLBACK_MAX_AGE_S)
            if (racket is not None
                    and racket['distance_mm']
                    > self.IMPACT_RECENT_RACKET_DISTANCE_MM):
                return False
        return bool(
            pair_velocity[2] >= self.impact_min_upward_velocity_mm_s
            and total_rise >= self.IMPACT_CONFIRM_MIN_TOTAL_RISE_MM
            and np.linalg.norm(pair_velocity)
            <= self.MOTION_GATE_MAX_SPEED_MM_S)

    def _post_impact_state(self, now, position, *, confirmed=False):
        endpoint, fitted_velocity = self._fit_trajectory_state(
            list(self.position_history), gravity=True)
        sample_count = len(self.position_history)
        if (self.post_impact_lateral_prior is not None
                and sample_count < self.PHYSICS_VELOCITY_MIN_FRAMES):
            alpha = float(np.clip(
                (sample_count - 1)
                / max(self.PHYSICS_VELOCITY_MIN_FRAMES - 1, 1),
                0.0, 1.0))
            fitted_velocity[:2] = (
                (1.0 - alpha) * self.post_impact_lateral_prior
                + alpha * fitted_velocity[:2])
        velocity_correction = None
        if not confirmed:
            fitted_velocity, velocity_correction = (
                self._stabilize_flight_velocity(now, fitted_velocity))
        if sample_count >= self.PHYSICS_VELOCITY_MIN_FRAMES:
            motion_state = 'flight'
            model = 'robust_xy_linear_z_gravity'
            self.post_impact_lateral_prior = None
        else:
            motion_state = 'post_impact_fast'
            model = 'post_impact_fast_gravity'
        return self._commit_velocity_output(
            now, position, endpoint, fitted_velocity,
            ready=True, motion_state=motion_state,
            velocity_model=model,
            velocity_correction=velocity_correction)

    def _append_physics_velocity_sample(self, now, position):
        raw_position = np.asarray(position, dtype=np.float64).reshape(3)
        history = self.position_history
        self.last_velocity_impact_detected = False

        if (history and float(now - history[-1][0])
                > self.PHYSICS_VELOCITY_MAX_GAP_S):
            history.clear()
            self._reset_velocity_tracking_state()

        if self.pending_impact is not None:
            if self._confirm_pending_impact(now, raw_position):
                pending = self.pending_impact
                history.clear()
                history.append((
                    pending['first_time'],
                    pending['first_position'].tolist()))
                history.append((float(now), raw_position.tolist()))
                self.pending_impact = None
                self.last_velocity_impact_time = pending['first_time']
                self.last_velocity_impact_detected = True
                self.velocity_impact_confirmed_count += 1
                self.last_impact_racket_gate_used = bool(
                    pending['racket_gate_used'])
                self.last_impact_racket_distance_mm = (
                    pending['racket_distance_mm'])
                self.post_impact_lateral_prior = (
                    pending['pre_velocity'][:2].copy())
                return self._post_impact_state(
                    now, raw_position, confirmed=True)
            pending = self.pending_impact
            self.pending_impact = None
            self.velocity_impact_rejected_count += 1
            history.append((
                pending['first_time'], pending['first_position'].tolist()))
            self.velocity_motion_state = 'flight'

        if self.velocity_motion_state == 'supported':
            trial = list(history) + [(float(now), raw_position.tolist())]
            trial = trial[-self._velocity_history_maxlen():]
            supported, endpoint, racket = self._history_is_supported(
                trial, now)
            if supported or not self._supported_exit_confirmed(
                    now, raw_position):
                history.append((float(now), raw_position.tolist()))
                if endpoint is None:
                    endpoint, _ = self._fit_trajectory_state(
                        list(history), gravity=False)
                supported_velocity = (
                    racket['velocity'] if racket is not None
                    else np.zeros(3, dtype=np.float64))
                return self._commit_velocity_output(
                    now, raw_position, endpoint, supported_velocity,
                    ready=True, motion_state='supported',
                    velocity_model='racket_supported')
            previous = history[-1] if history else None
            history.clear()
            if previous is not None:
                history.append(previous)
            history.append((float(now), raw_position.tolist()))
            endpoint, fitted_velocity = self._fit_trajectory_state(
                list(history), gravity=True)
            return self._commit_velocity_output(
                now, raw_position, endpoint, fitted_velocity,
                ready=True, motion_state='launch_fast',
                velocity_model='launch_fast_gravity')

        impact = self._impact_candidate(now, raw_position)
        if impact is not None:
            self.pending_impact = impact
            self.velocity_impact_candidate_count += 1
            self.last_impact_racket_gate_used = bool(
                impact['racket_gate_used'])
            self.last_impact_racket_distance_mm = (
                impact['racket_distance_mm'])
            held_velocity = impact['pre_velocity'].copy()
            dt = (
                float(now - self.last_velocity_output_time)
                if self.last_velocity_output_time is not None else 0.0)
            if dt > 0.0:
                held_velocity[2] -= (
                    self.PHYSICS_VELOCITY_GRAVITY_MM_S2 * dt)
            return self._commit_velocity_output(
                now, raw_position, raw_position, held_velocity,
                ready=True, motion_state='impact_confirming',
                velocity_model='pre_impact_hold')

        history.append((float(now), raw_position.tolist()))
        if self.velocity_motion_state in {'post_impact_fast', 'launch_fast'}:
            return self._post_impact_state(now, raw_position)

        if len(history) < self.PHYSICS_VELOCITY_MIN_FRAMES:
            return self._commit_velocity_output(
                now, raw_position, raw_position, np.zeros(3),
                ready=False, motion_state='initializing',
                velocity_model='unready')

        supported, endpoint, racket = self._history_is_supported(
            list(history), now)
        if supported:
            supported_velocity = (
                racket['velocity'] if racket is not None
                else np.zeros(3, dtype=np.float64))
            return self._commit_velocity_output(
                now, raw_position, endpoint, supported_velocity,
                ready=True, motion_state='supported',
                velocity_model='racket_supported')

        endpoint, fitted_velocity = self._fit_trajectory_state(
            list(history), gravity=True)
        fitted_velocity, correction = self._stabilize_flight_velocity(
            now, fitted_velocity)
        return self._commit_velocity_output(
            now, raw_position, endpoint, fitted_velocity,
            ready=True, motion_state='flight',
            velocity_model='robust_xy_linear_z_gravity',
            velocity_correction=correction)

    def _calculate_velocity(self):
        """Estimate velocity from one uninterrupted flight segment."""
        physics_enabled = bool(getattr(
            self, 'physics_constrained_velocity_enabled', False))
        required_frames = (
            int(self.PHYSICS_VELOCITY_MIN_FRAMES)
            if physics_enabled else 2)
        self.last_velocity_ready = False
        if len(self.position_history) < required_frames:
            return [0, 0, 0]

        history = list(self.position_history)
        timestamps = np.asarray(
            [float(sample[0]) for sample in history], dtype=np.float64)
        positions = np.asarray(
            [sample[1] for sample in history], dtype=np.float64)
        if (positions.ndim != 2 or positions.shape[1] != 3
                or not np.all(np.isfinite(timestamps))
                or not np.all(np.isfinite(positions))):
            return [0, 0, 0]
        if physics_enabled:
            velocity = self._calculate_physics_constrained_velocity(history)
            if not np.all(np.isfinite(velocity)):
                return [0, 0, 0]
            self.last_velocity_ready = True
            return velocity.tolist()
        if not self.velocity_least_squares_enabled:
            dt = float(timestamps[-1] - timestamps[0])
            if dt < 1e-4:
                return [0, 0, 0]
            self.last_velocity_ready = True
            return ((positions[-1] - positions[0]) / dt).tolist()

        centered_time = timestamps - float(np.mean(timestamps))
        denominator = float(np.dot(centered_time, centered_time))
        if denominator < 1e-8:
            return [0, 0, 0]
        centered_position = positions - np.mean(positions, axis=0)
        self.last_velocity_ready = True
        return ((centered_time @ centered_position) / denominator).tolist()

    def _append_velocity_sample(self, now, position):
        """Append one trusted 3D point to the active motion segment.

        Impact confirmation has the same two-frame contract for both the
        legacy and physics-constrained estimators: the first reversal is held
        out of history and a second rising point must confirm it.  This keeps
        a one-frame position glitch from resetting velocity state and avoids
        publishing a zero-velocity discontinuity at a real contact.
        """
        self.last_velocity_impact_detected = False
        if getattr(self, 'physics_constrained_velocity_enabled', False):
            return self._append_physics_velocity_sample(now, position)
        history = self.position_history
        raw_position = np.asarray(position, dtype=np.float64).reshape(3)
        if self.pending_impact is not None:
            if self._confirm_pending_impact(now, raw_position):
                pending = self.pending_impact
                history.clear()
                history.append((
                    pending['first_time'],
                    pending['first_position'].tolist()))
                history.append((float(now), raw_position.tolist()))
                self.pending_impact = None
                self.last_velocity_impact_time = pending['first_time']
                self.last_velocity_impact_detected = True
                self.velocity_impact_confirmed_count += 1
                self.last_impact_racket_gate_used = bool(
                    pending['racket_gate_used'])
                self.last_impact_racket_distance_mm = (
                    pending['racket_distance_mm'])
            else:
                pending = self.pending_impact
                self.pending_impact = None
                self.velocity_impact_rejected_count += 1
                history.append((
                    pending['first_time'],
                    pending['first_position'].tolist()))
                self.velocity_motion_state = 'flight'

        impact = self._impact_candidate(now, raw_position)
        if impact is not None:
            self.pending_impact = impact
            self.velocity_impact_candidate_count += 1
            self.last_impact_racket_gate_used = bool(
                impact['racket_gate_used'])
            self.last_impact_racket_distance_mm = (
                impact['racket_distance_mm'])
            return self._commit_velocity_output(
                now, raw_position, raw_position, impact['pre_velocity'],
                ready=True, motion_state='impact_confirming',
                velocity_model='pre_impact_hold')

        already_appended = bool(
            history and abs(float(history[-1][0]) - float(now)) < 1.0e-9)
        if not already_appended:
            history.append((float(now), raw_position.tolist()))
        velocity = self._calculate_velocity()
        fitted_position = raw_position
        if (self.trajectory_position_smoothing_enabled
                and len(history) >= 2):
            fitted_position, _ = self._fit_trajectory_state(
                list(history), gravity=False)
        return self._commit_velocity_output(
            now, raw_position, fitted_position, velocity,
            ready=self.last_velocity_ready,
            motion_state=('flight' if self.last_velocity_ready
                          else 'initializing'),
            velocity_model=(
                'least_squares'
                if self.velocity_least_squares_enabled else 'endpoint'))

    def _clear_observation_buffers(self, *, preserve_depth_guard=False):
        """Clear every temporal state that can leak across a detection gap."""
        restart_reference = None
        if (self.last_velocity_ready
                and self.last_velocity_output_time is not None
                and self.velocity_motion_state == 'flight'
                and np.all(np.isfinite(self.last_velocity_output))):
            restart_reference = (
                float(self.last_velocity_output_time),
                self.last_velocity_output.copy())
        self.depth_history.clear()
        self.position_history.clear()
        self.pixel_history.clear()
        self.candidate_position_history.clear()
        self.ir_depth_kf.reset()
        self.pixel_kf.reset()
        if not preserve_depth_guard:
            self._reset_adaptive_depth_state()
        self.last_velocity_impact_time = None
        self.last_velocity_impact_detected = False
        self._reset_velocity_tracking_state()
        if restart_reference is not None:
            (self.velocity_restart_reference_time,
             self.velocity_restart_reference_velocity) = restart_reference
        self.observation_buffer_clear_count += 1

    def _handle_observation_loss(self, reason, *, preserve_depth_guard=False):
        self.observation_gap_active = True
        # Impact confirmation may never bridge a missing/untrusted frame.  A
        # candidate that cannot be verified is discarded together with the
        # old segment so it cannot later mix pre/post-impact samples.
        if self.pending_impact is not None:
            self.position_history.clear()
            self.velocity_impact_rejected_count += 1
            self._reset_velocity_tracking_state()
        # One 90 Hz IR miss is only about 11 ms.  Keep a continuous-flight
        # fit across that isolated flicker; two consecutive IR misses, or any
        # depth/3D fault, establish a real segment boundary immediately.
        isolated_ir_flicker = bool(
            str(reason) == 'ir_detection_lost'
            and int(self.ir_tracking_miss_count) < 2)
        should_clear = bool(
            self.clear_buffers_on_detection_loss_enabled
            and not isolated_ir_flicker
            and not self.observation_buffers_cleared_for_gap)
        if should_clear:
            self._clear_observation_buffers(
                preserve_depth_guard=preserve_depth_guard)
            self.observation_buffers_cleared_for_gap = True
        self.motion_last_gate_reason = str(reason)

    def _confirm_stale_reacquisition(self, now, position):
        """Require two fresh, physically-consistent points after a long gap."""
        candidates = self.motion_reacquisition_candidates
        if not candidates:
            candidates.append((float(now), position.copy()))
            self.motion_last_gate_reason = 'reacquisition_confirming'
            self.motion_gate_rejected_count += 1
            return False
        previous_time, previous_position = candidates[-1]
        dt = float(now - previous_time)
        displacement = position - previous_position
        speed = (
            float(np.linalg.norm(displacement) / dt)
            if dt > 1.0e-6 else float('inf'))
        if (dt <= 1.0e-6
                or dt > self.MOTION_REACQUISITION_CONFIRM_MAX_GAP_S
                or speed > self.MOTION_GATE_MAX_SPEED_MM_S):
            candidates.clear()
            candidates.append((float(now), position.copy()))
            self.motion_last_gate_reason = 'reacquisition_inconsistent'
            self.motion_gate_rejected_count += 1
            return False
        self.motion_reacquisition_seed = (
            float(previous_time), previous_position.copy())
        candidates.clear()
        self.motion_last_gate_reason = 'reacquisition_confirmed'
        self.motion_gate_accepted_count += 1
        return True

    def _motion_measurement_consistent(self, now, position):
        """Gate a 3D point against the last trusted ballistic state."""
        position = np.asarray(position, dtype=np.float64).reshape(3)
        self.motion_last_innovation_mm = None
        self.motion_last_gate_reason = None
        if not self.motion_consistency_gate_enabled:
            return True
        # A real racket impact is intentionally discontinuous with the
        # pre-impact ballistic state.  The velocity estimator has already
        # required a first rising sample, optional racket proximity, minimum
        # downward/upward speeds and a minimum velocity jump before creating
        # ``pending_impact``.  Its *second* rising sample must reach the impact
        # state machine; applying the ordinary flight gate first otherwise
        # rejects precisely the sample needed to confirm contact.  This was
        # visible in the 20260812_212449/212631 recordings: impact confirmation
        # was starved, then reacquisition incorrectly reused the old downward
        # velocity.  The preview is side-effect free; the normal append path
        # still performs and owns the actual confirmation/reset.
        if (self.pending_impact is not None
                and self._confirm_pending_impact(now, position)):
            self.motion_last_gate_reason = 'impact_confirmation_bypass'
            self.motion_gate_accepted_count += 1
            self.motion_impact_confirmation_bypass_count += 1
            return True
        if (self.motion_last_trusted_time is None
                or self.motion_last_trusted_position is None
                or self.motion_last_trusted_velocity is None):
            return True
        dt = float(now - self.motion_last_trusted_time)
        if dt <= 1.0e-6:
            return True
        if dt > self.MOTION_GATE_MAX_REACQUISITION_DT_S:
            return self._confirm_stale_reacquisition(now, position)
        expected = (
            self.motion_last_trusted_position
            + self.motion_last_trusted_velocity * dt)
        expected = expected.copy()
        expected[2] -= 0.5 * self.PHYSICS_VELOCITY_GRAVITY_MM_S2 * dt * dt
        innovation = position - expected
        self.motion_last_innovation_mm = innovation.copy()
        if (self.observation_gap_active
                or dt > self.MOTION_GATE_MAX_DT_S):
            xy_limit = self.MOTION_GATE_REACQUISITION_MM
            z_limit = self.MOTION_GATE_REACQUISITION_MM
        else:
            # Add a small time-dependent allowance for acceleration/model error.
            xy_limit = self.MOTION_GATE_XY_BASE_MM + 250.0 * dt
            z_limit = self.MOTION_GATE_Z_BASE_MM + 500.0 * dt
        speed = float(np.linalg.norm(
            position - self.motion_last_trusted_position) / dt)
        reason = None
        if float(np.linalg.norm(innovation[:2])) > xy_limit:
            reason = 'xy_innovation'
        elif abs(float(innovation[2])) > z_limit:
            reason = 'z_innovation'
        elif speed > self.MOTION_GATE_MAX_SPEED_MM_S:
            reason = 'speed_limit'
        if reason is not None:
            # The first contact-reversal point can itself be inconsistent with
            # the pre-impact ballistic state (20260812_212631).  Let it reach
            # the estimator only when it already passes the complete impact
            # candidate contract, including racket proximity when required.
            # The estimator then holds pre-impact velocity and still requires
            # a second rising point, so one isolated 3D glitch cannot commit a
            # reset merely because it bypassed this gate.
            if (not self.observation_gap_active
                    and self._impact_candidate(now, position) is not None):
                self.motion_last_gate_reason = 'impact_candidate_bypass'
                self.motion_gate_accepted_count += 1
                self.motion_impact_candidate_bypass_count += 1
                return True
            if self.observation_gap_active:
                # The old ballistic model can be wrong after an impact or a
                # longer depth fault.  Two fresh, mutually-consistent trusted
                # 3D points are stronger evidence than indefinitely holding
                # the stale prediction, while still rejecting an isolated
                # false blob on the first frame.
                if self._confirm_stale_reacquisition(now, position):
                    return True
                return False
            self.motion_last_gate_reason = reason
            self.motion_gate_rejected_count += 1
            return False
        self.motion_gate_accepted_count += 1
        return True

    def _remember_motion_state(
            self, now, position, velocity, *, velocity_ready=True):
        position_array = np.asarray(
            position, dtype=np.float64).reshape(3).copy()
        velocity_array = np.asarray(
            velocity, dtype=np.float64).reshape(3).copy()
        if not velocity_ready:
            reference = self.motion_reacquisition_seed
            if reference is None and (
                    self.motion_last_trusted_time is not None
                    and self.motion_last_trusted_position is not None):
                reference = (
                    self.motion_last_trusted_time,
                    self.motion_last_trusted_position)
            if reference is not None:
                dt = float(now - float(reference[0]))
                if 1.0e-6 < dt <= self.PHYSICS_VELOCITY_MAX_GAP_S:
                    velocity_array = (
                        position_array
                        - np.asarray(reference[1], dtype=np.float64)) / dt
                    # Convert the interval-average ballistic Z velocity to
                    # the current endpoint velocity.
                    velocity_array[2] -= (
                        0.5 * self.PHYSICS_VELOCITY_GRAVITY_MM_S2 * dt)
        self.motion_last_trusted_time = float(now)
        self.motion_last_trusted_position = position_array
        self.motion_last_trusted_velocity = velocity_array
        self.observation_gap_active = False
        self.observation_buffers_cleared_for_gap = False
        self.motion_reacquisition_candidates.clear()
        self.motion_reacquisition_seed = None

    def _selected_depth_is_trusted(self, depth_method):
        """Return whether the selected range may enter any 3D state history."""
        method = str(depth_method or '')
        if self.depth_mode == 'adaptive':
            return bool(
                method == 'adaptive_sensor'
                and not self.last_depth_selection.get(
                    'sensor_rejected', False))
        if self.depth_mode == 'sensor':
            # Geometry fallback is useful for diagnostics, but it is not an
            # equivalent measurement on a tilted camera and must not be
            # presented as a valid XYZ sample.
            return method == 'sensor'
        # Explicit geometry mode is an operator-selected experiment.  Its
        # sensor fallback is also a physical depth measurement.
        return method in {'geometry', 'sensor'}

    def _snapshot_ir_tracking_state(self):
        """Capture 2D state so a later 3D rejection can be rolled back."""
        pixel_filter = self.pixel_kf
        return {
            'candidate_history': list(self.candidate_position_history),
            'pixel_filter': pixel_filter,
            'pixel_initialized': bool(pixel_filter.initialized),
            'pixel_state': pixel_filter.state.copy(),
            'pixel_covariance': pixel_filter.P.copy(),
            'pixel_consecutive_outliers': int(
                pixel_filter.consecutive_outliers),
            'pixel_total_updates': int(pixel_filter.total_updates),
            'pixel_outliers_rejected': int(
                pixel_filter.outliers_rejected),
        }

    def _restore_ir_tracking_state(self, snapshot):
        """Undo a candidate that failed depth or transformed-3D validation."""
        if not isinstance(snapshot, dict):
            return
        self.candidate_position_history.clear()
        self.candidate_position_history.extend(
            snapshot.get('candidate_history', ()))
        pixel_filter = snapshot.get('pixel_filter')
        # A concurrent runtime reconfiguration may have installed a new
        # filter.  Never overwrite that new object's tuning/state.
        if pixel_filter is None or self.pixel_kf is not pixel_filter:
            return
        pixel_filter.initialized = bool(snapshot['pixel_initialized'])
        pixel_filter.state = snapshot['pixel_state'].copy()
        pixel_filter.P = snapshot['pixel_covariance'].copy()
        pixel_filter.consecutive_outliers = int(
            snapshot['pixel_consecutive_outliers'])
        pixel_filter.total_updates = int(snapshot['pixel_total_updates'])
        pixel_filter.outliers_rejected = int(
            snapshot['pixel_outliers_rejected'])

    def _transform_ir_to_rgb_pixel(self, ir_x, ir_y, depth_m):
        """
        将IR图像坐标精确转换到RGB图像坐标

        使用RealSense的完整内参（包含畸变）和外参进行精确变换

        Args:
            ir_x, ir_y: IR图像坐标（像素）
            depth_m: 该点的深度（米）

        Returns:
            (rgb_x, rgb_y): RGB图像坐标（像素），如果转换失败返回None
        """
        if depth_m <= 0 or depth_m > 5.0:  # 深度范围检查（米）
            return None

        try:
            # 1. IR 2D像素 -> 3D点（IR相机坐标系）
            # 使用完整的IR内参（包含畸变模型和系数）
            point_3d_ir = rs.rs2_deproject_pixel_to_point(
                self.ir_intrinsics, [ir_x, ir_y], depth_m)

            # 2. 3D点（IR/Depth坐标系）-> 3D点（RGB坐标系）
            point_3d_color = rs.rs2_transform_point_to_point(
                self.depth_to_color_extrinsics, point_3d_ir)

            # 3. 3D点（RGB坐标系）-> 2D像素（RGB图像）
            # 使用完整的RGB内参（包含畸变模型和系数）
            pixel_rgb = rs.rs2_project_point_to_pixel(
                self.color_intrinsics, point_3d_color)

            # 检查映射结果是否在图像范围内
            if (0 <= pixel_rgb[0] < self.image_width and
                0 <= pixel_rgb[1] < self.image_height):
                return (pixel_rgb[0], pixel_rgb[1])
            else:
                return None

        except Exception as e:
            # 转换失败（可能超出图像范围或数值问题）
            return None

    def get_observation(self, observation_time=None, frames=None):
        """
        获取观测值

        修复后的混合检测流程：
        1. 每N帧运行YOLO获取bbox（辅助语义过滤）
        2. 优先在bbox内用IR检测
        3. **ROI内IR失败 → 立即全图IR兜底（软约束）**
        4. IR检测成功 → IR深度 → IR内参反投影 → IR 3D点
        5. **IR 3D → depth_to_color外参 → RGB 3D（手眼标定坐标系）**
        6. RGB 3D → RGB手眼外参 → base_link

        Args:
            observation_time: 观测时间
            frames: 预先获取的帧（可选，避免重复获取）
        """
        now = time.time() if observation_time is None else float(observation_time)
        self.frame_count += 1
        self.total_frames += 1

        # 获取帧（如果没有提供）
        if frames is None:
            frames = self.pipeline.wait_for_frames()

        # 修复：不使用align，保持IR和Depth的原生对齐关系
        ir_frame = frames.get_infrared_frame(1)
        depth_frame = frames.get_depth_frame()

        if not ir_frame or not depth_frame:
            self._handle_observation_loss('missing_realsense_frame')
            return {
                'detected': False, 'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'raw_measurement': {
                    'observation_state': 'detection_loss',
                    'observation_reason': 'missing_realsense_frame',
                    'measurement_trusted_3d': False,
                    'velocity_ready': False,
                },
            }

        # 修复：FPS计算使用IR帧时间戳（IR是主循环频率）
        current_timestamp = ir_frame.get_timestamp()  # 毫秒
        if self.last_frame_timestamp is not None:
            frame_interval_ms = current_timestamp - self.last_frame_timestamp
            if frame_interval_ms > 0:
                self.fps_queue.append(1000.0 / frame_interval_ms)
        self.last_frame_timestamp = current_timestamp

        ir_image = np.asanyarray(ir_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        rgb_image = self.color_image_for_frames(frames, ir_image)

        started = time.perf_counter()
        result = self._detect_core(rgb_image, ir_image, depth_image, now)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.detect_time_queue.append(elapsed_ms)
        target = (self.detected_time_queue if result.get('detected', False)
                  else self.missed_time_queue)
        target.append(elapsed_ms)
        raw = result.get('raw_measurement')
        if isinstance(raw, dict):
            raw.update(self.performance_stats())
        return result

    def get_observation_offline(self, rgb_image, ir_image, depth_image,
                                observation_time):
        """离线数组入口：喂录制的 RGB/IR/depth 数组，复用与在线相同的检测核心。

        与 get_observation 的区别仅在于帧来源（numpy 数组而非 RealSense
        frames），检测/深度/坐标变换逻辑完全一致。FPS 队列用相邻观测时间差
        近似（离线无硬件帧时间戳）。

        Args:
            rgb_image: BGR 图像 (H,W,3) uint8
            ir_image: IR 灰度图 (H,W) uint8
            depth_image: 原生对齐深度 (H,W) uint16，可为 None（退化为纯 geometry）
            observation_time: 观测时间（秒），来自 timestamps 或 frame_idx/fps
        """
        now = float(observation_time)
        self.frame_count += 1
        self.total_frames += 1

        if self.last_observation_time is not None:
            dt = now - self.last_observation_time
            if dt > 1e-6:
                self.fps_queue.append(1.0 / dt)
        self.last_observation_time = now

        if rgb_image is None or ir_image is None:
            self._handle_observation_loss('missing_offline_image')
            return {
                'detected': False, 'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'raw_measurement': {
                    'observation_state': 'detection_loss',
                    'observation_reason': 'missing_offline_image',
                    'measurement_trusted_3d': False,
                    'velocity_ready': False,
                },
            }
        started = time.perf_counter()
        result = self._detect_core(rgb_image, ir_image, depth_image, now)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.detect_time_queue.append(elapsed_ms)
        target = (self.detected_time_queue if result.get('detected', False)
                  else self.missed_time_queue)
        target.append(elapsed_ms)
        raw = result.get('raw_measurement')
        if isinstance(raw, dict):
            raw.update(self.performance_stats())
        return result

    def performance_stats(self):
        """Return lightweight rolling detector-stage timing diagnostics."""
        def mean(values):
            return float(np.mean(values)) if values else 0.0

        def p95(values):
            return float(np.percentile(values, 95.0)) if values else 0.0

        return {
            'detect_mean_ms': mean(self.detect_time_queue),
            'detect_p95_ms': p95(self.detect_time_queue),
            'detected_mean_ms': mean(self.detected_time_queue),
            'missed_mean_ms': mean(self.missed_time_queue),
            'yolo_mean_ms': mean(self.yolo_time_queue),
            'ir_mean_ms': mean(self.ir_time_queue),
            'rgb_fps': mean(getattr(self, 'color_fps_queue', ())),
            'rgb_frame_count': int(getattr(
                self, 'color_frame_count', 0)),
            'yolo_enabled': bool(self.yolo_enabled),
        }

    def _rgb_on_ir_canvas(self, rgb_image):
        """Normalize unexpected offline/fallback RGB to the IR canvas.

        Normal D455 operation keeps both streams at 848x480; only their rates
        differ (stereo 90 Hz, RGB 60 Hz), so this is a no-op online.
        """
        if rgb_image is None:
            return None
        target = (int(self.image_width), int(self.image_height))
        if rgb_image.shape[1] == target[0] and rgb_image.shape[0] == target[1]:
            return rgb_image
        return cv2.resize(rgb_image, target, interpolation=cv2.INTER_LINEAR)

    def color_image_for_frames(self, frames=None, ir_image=None):
        """Return the newest 60 Hz RGB image without blocking stereo."""
        color_frame = None
        if frames is not None:
            try:
                color_frame = frames.get_color_frame()
            except Exception:
                color_frame = None
        color_pipeline = getattr(self, 'color_pipeline', None)
        # A stereo-only frameset returns an invalid ``video_frame`` object,
        # not Python ``None``.  Test frame truthiness so the independent RGB
        # pipeline is actually polled instead of serving its startup cache
        # forever.
        if not color_frame and color_pipeline is not None:
            try:
                color_frames = color_pipeline.poll_for_frames()
                if color_frames:
                    color_frame = color_frames.get_color_frame()
            except Exception:
                color_frame = None
        if color_frame:
            image = np.asanyarray(color_frame.get_data())
            if image is not None and image.size > 0:
                self.latest_color_image = image.copy()
                self.color_frame_count = int(getattr(
                    self, 'color_frame_count', 0)) + 1
                try:
                    timestamp_ms = float(color_frame.get_timestamp())
                except Exception:
                    timestamp_ms = None
                previous_ms = getattr(
                    self, 'color_last_timestamp_ms', None)
                if (timestamp_ms is not None and previous_ms is not None
                        and timestamp_ms > previous_ms):
                    self.color_fps_queue.append(
                        1000.0 / (timestamp_ms - previous_ms))
                if timestamp_ms is not None:
                    self.color_last_timestamp_ms = timestamp_ms
        cached = getattr(self, 'latest_color_image', None)
        if cached is not None:
            return cached
        if ir_image is not None:
            return cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        return np.zeros(
            (int(self.image_height), int(self.image_width), 3),
            dtype=np.uint8)

    def _detect_core(self, rgb_image, ir_image, depth_image, now):
        """混合检测核心（在线/离线共用）：YOLO ROI → IR 定位 → 深度 → 3D。

        输入均为 numpy 数组；depth_image 允许为 None（无 sensor 深度时退回
        geometry）。帧计数/FPS 由调用方（get_observation/offline）负责维护。
        """
        rgb_image = self._rgb_on_ir_canvas(rgb_image)
        detection_method = 'none'
        roi = None

        # 步骤1：YOLO检测（可运行时关闭；关闭后使用IR自身轨迹ROI）
        if self.yolo_enabled and self.frame_count % self.yolo_interval == 0:
            yolo_result = self._detect_with_yolo(rgb_image)

            if yolo_result is not None:
                self.last_yolo_bbox = yolo_result['bbox']
                self.last_yolo_time = now
                self.yolo_miss_count = 0
                self.yolo_detections += 1
            else:
                self.yolo_miss_count += 1

        # 检查YOLO bbox是否有效（软约束）
        use_roi = False
        if (self.yolo_enabled and self.last_yolo_bbox is not None
                and self.yolo_miss_count < self.max_yolo_miss):
            # 使用YOLO bbox作为ROI
            roi = self._expand_bbox(self.last_yolo_bbox, self.roi_margin)
            use_roi = True
            detection_method = 'hybrid'
        elif not self.yolo_enabled:
            roi = self._predicted_ir_roi(now)
            use_roi = roi is not None
            detection_method = 'ir_tracking' if use_roi else 'ir_fallback'
        else:
            # YOLO失效，清除bbox
            self.last_yolo_bbox = None
            detection_method = 'ir_fallback'

        # 步骤2：IR检测（优先在ROI内，失败立即全图兜底）
        t_ir = time.time()
        ir_detection = None
        ir_tracking_snapshot = self._snapshot_ir_tracking_state()

        if use_roi:
            # 优先在ROI内检测
            ir_detection = self._detect_in_ir(
                ir_image, roi, depth_image, now)
            if ir_detection is None:
                # 修复：ROI内失败，立即全图兜底（YOLO软约束）
                ir_detection = self._detect_in_ir(
                    ir_image, None, depth_image, now)
                if ir_detection is not None:
                    detection_method = 'ir_fallback'  # 标记为兜底模式
        else:
            # 全图检测
            ir_detection = self._detect_in_ir(
                ir_image, None, depth_image, now)

        self.ir_time_queue.append((time.time() - t_ir) * 1000)

        if ir_detection is None:
            self.ir_tracking_miss_count += 1
            if not self.yolo_enabled and self.ir_tracking_miss_count >= 2:
                self.candidate_position_history.clear()
            self._handle_observation_loss('ir_detection_lost')
            return {
                'detected': False, 'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'detection_method': detection_method,
                'raw_measurement': {
                    'observation_state': 'detection_loss',
                    'observation_reason': 'ir_detection_lost',
                    'measurement_trusted_3d': False,
                    'velocity_ready': False,
                    'detection_method': detection_method,
                    'observation_gap_active': True,
                    'observation_buffer_clear_count': int(
                        self.observation_buffer_clear_count),
                    'ir_edge_rejected_count': int(
                        self.ir_edge_rejected_count),
                    'ir_roi_edge_rejected_count': int(
                        self.ir_roi_edge_rejected_count),
                },
            }
        self.ir_tracking_miss_count = 0

        # 步骤3：获取深度
        # 3a. sensor（双目硬件深度，远场准）与 geometry（单目 Z=f*D/2r，近场稳）都算出来
        geom_depth_raw_mm = ir_detection.get(
            'selection_geom_depth_raw_mm')
        if geom_depth_raw_mm is None:
            geom_depth_raw_mm = self._geometry_depth_mm(
                ir_detection.get(
                    'geometry_radius_px', ir_detection['radius']))
        geom_depth_corrected_mm = ir_detection.get(
            'selection_geom_depth_corrected_mm')
        if geom_depth_corrected_mm is None:
            geom_depth_corrected_mm = self._apply_depth_correction(
                geom_depth_raw_mm)
        geom_depth_mm = (
            geom_depth_corrected_mm
            if self.depth_mode == 'geometry'
            else geom_depth_raw_mm)

        sensor_depth_mm = ir_detection.get('selection_sensor_depth_mm')
        # Adaptive mode intentionally recomputes the sample with a foreground
        # percentile even if candidate scoring already cached a legacy median.
        if sensor_depth_mm is None or self.depth_mode == 'adaptive':
            depth_sample_cx = ir_detection.get(
                'cx_raw', ir_detection['cx'])
            depth_sample_cy = ir_detection.get(
                'cy_raw', ir_detection['cy'])
            sensor_depth_mm = self._get_depth_from_depthmap(
                depth_sample_cx, depth_sample_cy, depth_image,
                ir_detection['mask'], ir_detection.get('mask_origin'),
                prefer_near_surface=(self.depth_mode == 'adaptive'),
                geometry_hint_mm=geom_depth_mm)

        # 3b. sensor模式保持原值；adaptive任一异常条件命中时短时接管。
        depth_raw, depth_method = self._select_depth(
            sensor_depth_mm, geom_depth_mm, now,
            observed_radius_px=(
                ir_detection.get('geometry_radius_px')
                if ir_detection.get('geometry_quality_valid', True)
                else None),
            observed_cy_px=(
                ir_detection.get('cy_raw', ir_detection.get('cy'))
                if ir_detection.get('geometry_quality_valid', True)
                else None),
            defer_commit=True)
        if depth_raw is None:
            self._finalize_adaptive_depth_selection(False)
            self._restore_ir_tracking_state(ir_tracking_snapshot)
            self._handle_observation_loss('depth_unavailable')
            return {
                'detected': False, 'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'detection_method': detection_method,
                'raw_measurement': {
                    'cx': ir_detection['cx'],
                    'cy': ir_detection['cy'],
                    'cx_raw': ir_detection.get('cx_raw'),
                    'cy_raw': ir_detection.get('cy_raw'),
                    'radius': ir_detection['radius'],
                    'observation_state': 'depth_fault',
                    'observation_reason': 'depth_unavailable',
                    'measurement_trusted_3d': False,
                    'velocity_ready': False,
                    'depth_mode': self.depth_mode,
                    'sensor_surface_to_center_depth_enabled': bool(
                        self.sensor_surface_to_center_depth_enabled),
                    'sensor_surface_to_center_depth_applied': False,
                    'depth_sensor_rejected': bool(
                        self.depth_mode == 'adaptive'),
                    'depth_rejection_reason': 'depth_unavailable',
                    'detection_method': detection_method,
                },
            }

        depth_selection = dict(self.last_depth_selection)
        measurement_trusted_3d = self._selected_depth_is_trusted(
            depth_method)
        if not measurement_trusted_3d:
            # Geometry/predicted depth may still be drawn and recorded, but
            # never reaches the temporal median, 3D KF, motion state, or
            # velocity fit.  Because the camera is tilted, even a correct IR
            # centre does not make any transformed XYZ component trustworthy
            # when axial depth is wrong.
            self._finalize_adaptive_depth_selection(False)
            self._restore_ir_tracking_state(ir_tracking_snapshot)
            self._handle_observation_loss(
                'depth_untrusted', preserve_depth_guard=True)
            return {
                'detected': False,
                'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'detection_method': detection_method,
                'raw_measurement': {
                    'cx': ir_detection['cx'],
                    'cy': ir_detection['cy'],
                    'cx_raw': ir_detection.get('cx_raw'),
                    'cy_raw': ir_detection.get('cy_raw'),
                    'radius': ir_detection['radius'],
                    'geometry_radius_px': ir_detection.get(
                        'geometry_radius_px'),
                    'geometry_quality_valid': ir_detection.get(
                        'geometry_quality_valid'),
                    'depth_mm': float(depth_raw),
                    'sensor_depth_mm': sensor_depth_mm,
                    'geom_depth_mm': geom_depth_mm,
                    'geom_depth_raw_mm': geom_depth_raw_mm,
                    'geom_depth_corrected_mm': geom_depth_corrected_mm,
                    'depth_correction_model_type': (
                        None if self.depth_correction_model is None
                        else self.depth_correction_model.get('type')),
                    'depth_method': depth_method,
                    'depth_mode': self.depth_mode,
                    'sensor_surface_to_center_depth_enabled': bool(
                        self.sensor_surface_to_center_depth_enabled),
                    'sensor_surface_to_center_depth_applied': False,
                    'depth_sensor_weight': depth_selection.get(
                        'sensor_weight'),
                    'depth_disagreement_mm': depth_selection.get(
                        'disagreement_mm'),
                    'depth_sensor_rejected': depth_selection.get(
                        'sensor_rejected'),
                    'depth_rejection_reason': depth_selection.get(
                        'rejection_reason'),
                    'depth_sensor_jump_mm': depth_selection.get(
                        'sensor_jump_mm'),
                    'depth_observed_radius_px': depth_selection.get(
                        'observed_radius_px'),
                    'depth_sensor_projected_radius_px': depth_selection.get(
                        'sensor_projected_radius_px'),
                    'depth_radius_mismatch_ratio': depth_selection.get(
                        'radius_mismatch_ratio'),
                    'depth_near_field_scale_stall': depth_selection.get(
                        'near_field_scale_stall'),
                    'depth_near_field_window_samples': depth_selection.get(
                        'near_field_window_samples'),
                    'depth_near_field_window_span_s': depth_selection.get(
                        'near_field_window_span_s'),
                    'depth_near_field_sensor_approach_mm': (
                        depth_selection.get(
                            'near_field_sensor_approach_mm')),
                    'depth_near_field_geometry_approach_mm': (
                        depth_selection.get(
                            'near_field_geometry_approach_mm')),
                    'depth_near_field_radius_growth_ratio': (
                        depth_selection.get(
                            'near_field_radius_growth_ratio')),
                    'depth_near_field_upward_pixel_speed_px_s': (
                        depth_selection.get(
                            'near_field_upward_pixel_speed_px_s')),
                    'depth_prediction_mm': depth_selection.get(
                        'prediction_mm'),
                    'depth_prediction_horizon_s': depth_selection.get(
                        'prediction_horizon_s'),
                    'depth_sensor_recovery_count': depth_selection.get(
                        'sensor_recovery_count'),
                    'measurement_trusted_3d': False,
                    'velocity_ready': False,
                    'observation_state': 'depth_fault',
                    'observation_gap_active': True,
                    'observation_buffer_clear_count': int(
                        self.observation_buffer_clear_count),
                    'detection_method': detection_method,
                    'ir_roi_edge_rejected_count': int(
                        self.ir_roi_edge_rejected_count),
                },
            }

        use_ir_depth_kalman = bool(
            getattr(self, 'ir_depth_kalman_enabled', False))
        use_synchronized_depth = bool(
            getattr(self, 'synchronized_depth_enabled', False))
        # The switch is deliberately independent from the joint KF. This makes
        # all four raw/median-depth and filtered/unfiltered combinations valid.
        # Preview the exact median/Hampel result but defer the history write
        # until the resulting transformed 3D point passes the motion gate.
        sensor_surface_depth_mm = self._depth_for_reconstruction(
            depth_raw, commit=False)
        depth_mm, sensor_center_depth_diagnostics = (
            self._sensor_surface_to_center_depth(
                sensor_surface_depth_mm, depth_method))
        projected_radius_px = self._projected_ball_radius_px(depth_mm)

        # 步骤4：IR坐标 + IR内参 → IR 3D点
        measurement_cx = (
            ir_detection.get('cx_raw', ir_detection['cx'])
            if use_ir_depth_kalman else ir_detection['cx'])
        measurement_cy = (
            ir_detection.get('cy_raw', ir_detection['cy'])
            if use_ir_depth_kalman else ir_detection['cy'])
        X_ir, Y_ir, Z_ir = self._backproject_to_3d(
            measurement_cx, measurement_cy, depth_mm)
        pos_ir = [X_ir, Y_ir, Z_ir]

        # 步骤5：IR 3D → T_ir_to_rgb齐次变换 → RGB 3D（关键！）
        pos_rgb_camera = self._transform_ir_3d_to_rgb_3d(pos_ir)

        # 步骤6：转换坐标系（RGB手眼外参 → base_link）
        if self.output_frame == 'base_link' and self.T_base_camera is not None:
            measured_position = self._transform_to_base_link(pos_rgb_camera)
        else:
            measured_position = pos_rgb_camera

        was_observation_gap = bool(self.observation_gap_active)
        if not self._motion_measurement_consistent(now, measured_position):
            self._finalize_adaptive_depth_selection(False)
            self._restore_ir_tracking_state(ir_tracking_snapshot)
            rejection_reason = self.motion_last_gate_reason
            innovation = self.motion_last_innovation_mm
            # Loss handling can clear adaptive state.  Preserve the depth
            # verdict which caused this candidate so the controller can still
            # route a radius-triggered sensor fault into trajectory prediction.
            depth_selection = dict(self.last_depth_selection)
            self._handle_observation_loss(
                'motion_consistency_' + str(rejection_reason),
                preserve_depth_guard=True)
            return {
                'detected': False,
                'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'detection_method': detection_method,
                'raw_measurement': {
                    'cx': ir_detection['cx'],
                    'cy': ir_detection['cy'],
                    'cx_raw': ir_detection.get('cx_raw'),
                    'cy_raw': ir_detection.get('cy_raw'),
                    'radius': ir_detection['radius'],
                    # Preserve the exact pre-gate 3D point for deterministic
                    # offline state-machine replay.  The annotated JPEG and
                    # the public rejected position=[0,0,0] cannot recover it.
                    'transformed_measurement_x_mm': float(
                        measured_position[0]),
                    'transformed_measurement_y_mm': float(
                        measured_position[1]),
                    'transformed_measurement_z_mm': float(
                        measured_position[2]),
                    'projected_radius_px': projected_radius_px,
                    'motion_consistency_gate_enabled': True,
                    'motion_consistency_rejected': True,
                    'motion_consistency_reason': rejection_reason,
                    'motion_innovation_mm': (
                        None if innovation is None
                        else [float(value) for value in innovation]),
                    'observation_gap_active': True,
                    'observation_buffer_clear_count': int(
                        self.observation_buffer_clear_count),
                    'depth_mm': depth_mm,
                    'sensor_depth_mm': sensor_depth_mm,
                    **sensor_center_depth_diagnostics,
                    'geom_depth_mm': geom_depth_mm,
                    'depth_method': depth_method,
                    'depth_mode': self.depth_mode,
                    'depth_sensor_weight': depth_selection.get(
                        'sensor_weight'),
                    'depth_disagreement_mm': depth_selection.get(
                        'disagreement_mm'),
                    'depth_sensor_rejected': depth_selection.get(
                        'sensor_rejected'),
                    'depth_rejection_reason': depth_selection.get(
                        'rejection_reason'),
                    'depth_sensor_jump_mm': depth_selection.get(
                        'sensor_jump_mm'),
                    'depth_observed_radius_px': (
                        depth_selection.get('observed_radius_px')),
                    'depth_sensor_projected_radius_px': (
                        depth_selection.get(
                            'sensor_projected_radius_px')),
                    'depth_radius_mismatch_ratio': (
                        depth_selection.get('radius_mismatch_ratio')),
                    'depth_near_field_scale_stall': (
                        depth_selection.get('near_field_scale_stall')),
                    'depth_near_field_window_samples': (
                        depth_selection.get('near_field_window_samples')),
                    'depth_near_field_window_span_s': (
                        depth_selection.get('near_field_window_span_s')),
                    'depth_near_field_sensor_approach_mm': (
                        depth_selection.get(
                            'near_field_sensor_approach_mm')),
                    'depth_near_field_geometry_approach_mm': (
                        depth_selection.get(
                            'near_field_geometry_approach_mm')),
                    'depth_near_field_radius_growth_ratio': (
                        depth_selection.get(
                            'near_field_radius_growth_ratio')),
                    'depth_near_field_upward_pixel_speed_px_s': (
                        depth_selection.get(
                            'near_field_upward_pixel_speed_px_s')),
                    'measurement_trusted_3d': False,
                    'velocity_ready': False,
                    'observation_state': 'motion_outlier',
                },
            }

        motion_reacquisition_confirmed = bool(
            self.motion_last_gate_reason == 'reacquisition_confirmed')
        if not use_synchronized_depth:
            self._depth_for_reconstruction(depth_raw, commit=True)
        self._finalize_adaptive_depth_selection(True)

        if use_ir_depth_kalman:
            position, velocity = self.ir_depth_kf.update(
                measured_position, now)
        else:
            position = measured_position
            velocity = None

        # Estimate raw 3D velocity only from a continuous flight segment.
        # The joint IR+depth Kalman filter owns its own velocity state when it
        # is enabled, but the history is still maintained for diagnostics and
        # for an immediate switch back to the raw estimator.
        reacquisition_seed = self.motion_reacquisition_seed
        if reacquisition_seed is not None and not self.position_history:
            self.position_history.append((
                float(reacquisition_seed[0]),
                np.asarray(reacquisition_seed[1], dtype=np.float64).tolist(),
            ))
        history_velocity = self._append_velocity_sample(now, position)
        if velocity is None:
            if ((self.physics_constrained_velocity_enabled
                    or self.trajectory_position_smoothing_enabled)
                    and self.last_position_estimate is not None):
                position = self.last_position_estimate.tolist()
            velocity = history_velocity
        self._remember_motion_state(
            now, position, velocity,
            velocity_ready=bool(
                use_ir_depth_kalman or self.last_velocity_ready))

        # 统计
        self.detected_frames += 1
        self.ir_detections += 1
        if detection_method == 'hybrid':
            self.hybrid_detections += 1
        elif detection_method in ('ir_fallback', 'ir_tracking'):
            self.fallback_detections += 1

        # 可视化：将RGB 3D点投影到RGB图像（保证和3D位置完全一致）
        rgb_pixel = self._project_rgb_3d_to_pixel(pos_rgb_camera)

        # 更新像素坐标历史（用于可视化平滑）
        if rgb_pixel:
            self.pixel_history.append((rgb_pixel[0], rgb_pixel[1], ir_detection['radius']))

        # 计算平滑后的像素坐标（简单移动平均）
        smoothed_cx = None
        smoothed_cy = None
        smoothed_radius = None
        if len(self.pixel_history) >= 2:
            pixels = list(self.pixel_history)
            smoothed_cx = float(np.mean([p[0] for p in pixels]))
            smoothed_cy = float(np.mean([p[1] for p in pixels]))
            smoothed_radius = float(np.mean([p[2] for p in pixels]))

        return {
            'detected': True,
            'position': position,
            'velocity': velocity,
            'raw_measurement': {
                'cx': ir_detection['cx'],  # 已经是卡尔曼平滑后的IR坐标
                'cy': ir_detection['cy'],
                'cx_raw': ir_detection.get('cx_raw'),  # 原始检测（卡尔曼前）
                'cy_raw': ir_detection.get('cy_raw'),
                'cx_coarse': ir_detection.get('cx_coarse'),
                'cy_coarse': ir_detection.get('cy_coarse'),
                'center_refine_shift_px': ir_detection.get(
                    'center_refine_shift_px'),
                'transformed_measurement_x_mm': float(
                    measured_position[0]),
                'transformed_measurement_y_mm': float(
                    measured_position[1]),
                'transformed_measurement_z_mm': float(
                    measured_position[2]),
                'cx_rgb': rgb_pixel[0] if rgb_pixel else None,  # IR在RGB图上的精确位置（卡尔曼后）
                'cy_rgb': rgb_pixel[1] if rgb_pixel else None,
                'radius': ir_detection['radius'],
                'projected_radius_px': projected_radius_px,
                'smoothed_cx': smoothed_cx,  # 移动平均平滑（额外的平滑层）
                'smoothed_cy': smoothed_cy,
                'smoothed_radius': smoothed_radius,
                'depth_mm': depth_mm,
                'sensor_depth_mm': sensor_depth_mm,
                **sensor_center_depth_diagnostics,
                'synchronized_depth_enabled': use_synchronized_depth,
                'depth_temporal_window_frames': (
                    self.depth_temporal_window_frames),
                'velocity_least_squares_enabled': (
                    self.velocity_least_squares_enabled),
                'velocity_window_frames': self.velocity_window_frames,
                'velocity_history_size': len(self.position_history),
                'velocity_min_frames': (
                    1 if use_ir_depth_kalman else (
                        self.PHYSICS_VELOCITY_MIN_FRAMES
                        if self.physics_constrained_velocity_enabled else 2)),
                'velocity_ready': bool(
                    use_ir_depth_kalman or self.last_velocity_ready),
                'trajectory_position_smoothing_enabled': bool(
                    self.trajectory_position_smoothing_enabled),
                'position_filter_residual_mm': (
                    self.last_position_filter_residual_mm),
                'measurement_trusted_3d': True,
                'observation_state': (
                    'impact_confirming'
                    if self.velocity_motion_state == 'impact_confirming'
                    else ('reacquiring'
                          if was_observation_gap
                          or not bool(
                              use_ir_depth_kalman or self.last_velocity_ready)
                          else 'tracking')),
                'ir_edge_rejected_count': int(
                    self.ir_edge_rejected_count),
                'ir_roi_edge_rejected_count': int(
                    self.ir_roi_edge_rejected_count),
                'clear_buffers_on_detection_loss_enabled': bool(
                    self.clear_buffers_on_detection_loss_enabled),
                'observation_gap_active': bool(self.observation_gap_active),
                'observation_buffer_clear_count': int(
                    self.observation_buffer_clear_count),
                'motion_consistency_gate_enabled': bool(
                    self.motion_consistency_gate_enabled),
                'motion_consistency_rejected': False,
                'motion_consistency_reason': self.motion_last_gate_reason,
                'motion_reacquisition_confirmed': (
                    motion_reacquisition_confirmed),
                'motion_innovation_mm': (
                    None if self.motion_last_innovation_mm is None
                    else [float(value)
                          for value in self.motion_last_innovation_mm]),
                'motion_gate_accepted_count': int(
                    self.motion_gate_accepted_count),
                'motion_gate_rejected_count': int(
                    self.motion_gate_rejected_count),
                'motion_impact_confirmation_bypass_count': int(
                    self.motion_impact_confirmation_bypass_count),
                'motion_impact_candidate_bypass_count': int(
                    self.motion_impact_candidate_bypass_count),
                'physics_constrained_velocity_enabled': bool(
                    self.physics_constrained_velocity_enabled),
                'velocity_model': (
                    self.velocity_model
                    if self.physics_constrained_velocity_enabled
                    else ('least_squares'
                          if self.velocity_least_squares_enabled
                          else 'endpoint')),
                'velocity_motion_state': self.velocity_motion_state,
                'velocity_filter_correction_m_s': (
                    None
                    if self.last_velocity_filter_correction_mm_s is None
                    else self.last_velocity_filter_correction_mm_s / 1000.0),
                'velocity_restart_bridge_used': bool(
                    self.last_velocity_restart_bridge_used),
                'velocity_restart_bridge_count': int(
                    self.velocity_restart_bridge_count),
                'impact_velocity_reset_enabled': (
                    self.impact_velocity_reset_enabled),
                'velocity_impact_detected': (
                    self.last_velocity_impact_detected),
                'velocity_impact_candidate': bool(
                    self.pending_impact is not None),
                'velocity_impact_candidate_count': int(
                    self.velocity_impact_candidate_count),
                'velocity_impact_confirmed_count': int(
                    self.velocity_impact_confirmed_count),
                'velocity_impact_rejected_count': int(
                    self.velocity_impact_rejected_count),
                'impact_racket_gate_used': bool(
                    self.last_impact_racket_gate_used),
                'impact_racket_gate_required': bool(
                    self.impact_racket_gate_required),
                'impact_racket_distance_mm': (
                    self.last_impact_racket_distance_mm),
                'impact_racket_state_age_s': (
                    self.last_impact_racket_state_age_s),
                'impact_candidate_rejection_reason': (
                    self.last_impact_candidate_rejection_reason),
                'impact_pair_vertical_velocity_m_s': (
                    None
                    if self.last_impact_pair_vertical_velocity_mm_s is None
                    else self.last_impact_pair_vertical_velocity_mm_s
                    / 1000.0),
                'impact_previous_vertical_velocity_m_s': (
                    None
                    if self.last_impact_previous_vertical_velocity_mm_s is None
                    else self.last_impact_previous_vertical_velocity_mm_s
                    / 1000.0),
                'impact_evidence_source': (
                    self.last_impact_evidence_source),
                'ir_depth_kalman_enabled': use_ir_depth_kalman,
                'ir_depth_innovation_mahalanobis': (
                    self.ir_depth_kf.last_innovation_mahalanobis
                    if use_ir_depth_kalman else None),
                'geom_depth_mm': geom_depth_mm,
                'geom_depth_raw_mm': geom_depth_raw_mm,
                'geom_depth_corrected_mm': geom_depth_corrected_mm,
                'depth_correction_model_type': (
                    None if self.depth_correction_model is None
                    else self.depth_correction_model.get('type')),
                'geometry_radius_px': ir_detection.get(
                    'geometry_radius_px'),
                'geometry_quality_valid': ir_detection.get(
                    'geometry_quality_valid'),
                'ellipse_minor_px': ir_detection.get('ellipse_minor_px'),
                'ellipse_major_px': ir_detection.get('ellipse_major_px'),
                'area_circularity': ir_detection.get('area_circularity'),
                'depth_method': depth_method,
                'depth_mode': self.depth_mode,
                'depth_sensor_weight': self.last_depth_selection.get(
                    'sensor_weight'),
                'depth_disagreement_mm': self.last_depth_selection.get(
                    'disagreement_mm'),
                'depth_sensor_rejected': self.last_depth_selection.get(
                    'sensor_rejected'),
                'depth_rejection_reason': self.last_depth_selection.get(
                    'rejection_reason'),
                'depth_sensor_jump_mm': self.last_depth_selection.get(
                    'sensor_jump_mm'),
                'depth_observed_radius_px': self.last_depth_selection.get(
                    'observed_radius_px'),
                'depth_sensor_projected_radius_px': (
                    self.last_depth_selection.get(
                        'sensor_projected_radius_px')),
                'depth_radius_mismatch_ratio': self.last_depth_selection.get(
                    'radius_mismatch_ratio'),
                'depth_near_field_scale_stall': (
                    self.last_depth_selection.get(
                        'near_field_scale_stall')),
                'depth_near_field_window_samples': (
                    self.last_depth_selection.get(
                        'near_field_window_samples')),
                'depth_near_field_window_span_s': (
                    self.last_depth_selection.get(
                        'near_field_window_span_s')),
                'depth_near_field_sensor_approach_mm': (
                    self.last_depth_selection.get(
                        'near_field_sensor_approach_mm')),
                'depth_near_field_geometry_approach_mm': (
                    self.last_depth_selection.get(
                        'near_field_geometry_approach_mm')),
                'depth_near_field_radius_growth_ratio': (
                    self.last_depth_selection.get(
                        'near_field_radius_growth_ratio')),
                'depth_near_field_upward_pixel_speed_px_s': (
                    self.last_depth_selection.get(
                        'near_field_upward_pixel_speed_px_s')),
                'depth_prediction_mm': self.last_depth_selection.get(
                    'prediction_mm'),
                'depth_prediction_residual_mm': (
                    self.last_depth_selection.get(
                        'prediction_residual_mm')),
                'depth_prediction_horizon_s': self.last_depth_selection.get(
                    'prediction_horizon_s'),
                'depth_sensor_recovery_count': self.last_depth_selection.get(
                    'sensor_recovery_count'),
                'detection_method': detection_method,
                'pixel_source': 'ir',  # 主要数据源
                'yolo_bbox': self.last_yolo_bbox,
                'roi': roi,
                'aspect_ratio': ir_detection['aspect_ratio'],
                'threshold_used': ir_detection.get('threshold'),
                'brightness': ir_detection['brightness'],
                'candidate_selection_score': ir_detection.get(
                    'selection_score'),
                'candidate_motion_distance_px': ir_detection.get(
                    'selection_motion_distance_px'),
                'candidate_depth_disagreement_mm': ir_detection.get(
                    'selection_depth_disagreement_mm'),
                'candidate_filter_enabled': bool(
                    self.candidate_filter_enabled),
                'raw_position_ir': pos_ir,
                'raw_position_rgb': pos_rgb_camera,  # RGB坐标系位置
            },
        }

    def run(self, enable_viz=True, save_raw_video=False, output_dir='outputs/videos'):
        """
        主循环

        Args:
            enable_viz: 是否显示可视化窗口
            save_raw_video: 是否保存原始视频
            output_dir: 视频保存目录
        """
        print("混合检测器运行中，Ctrl+C 退出...")

        # 初始化视频录制
        video_writer = None
        video_path = None
        if save_raw_video:
            os.makedirs(output_dir, exist_ok=True)
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(output_dir, f'hybrid_raw_{timestamp}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, self.fps,
                                          (self.image_width, self.image_height))
            print(f"录制原始视频到: {video_path}")

        # 窗口名称
        if enable_viz:
            cv2.namedWindow('Hybrid Detector', cv2.WINDOW_NORMAL)

        try:
            while True:
                # 获取帧（只获取一次）
                frames = self.pipeline.wait_for_frames()

                # 保存原始视频（在对齐前保存原始RGB）
                if save_raw_video and video_writer is not None:
                    rgb_image = self.color_image_for_frames(frames)
                    video_writer.write(rgb_image)

                # 获取观测（传入已获取的帧，避免重复获取）
                observation = self.get_observation(frames=frames)

                # 获取图像用于可视化（修复：避免重复align）
                ir_frame = frames.get_infrared_frame(1)

                if ir_frame:
                    ir_image = np.asanyarray(ir_frame.get_data())
                    rgb_image = self.color_image_for_frames(
                        frames, ir_image)

                    # 可视化（始终显示，即使未检测到）
                    if enable_viz:
                        display = self._render_visualization(rgb_image, ir_image, observation)
                        cv2.imshow('Hybrid Detector', display)

                # 处理键盘输入（必须始终调用waitKey来刷新窗口）
                if enable_viz:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q or ESC
                        break

                # 打印
                if self.total_frames % self.print_every == 0:
                    if observation['detected']:
                        pos = observation['position']
                        vel = observation['velocity']
                        raw = observation['raw_measurement']

                        avg_fps = np.mean(list(self.fps_queue)) if self.fps_queue else 0
                        avg_yolo_ms = np.mean(list(self.yolo_time_queue)) if self.yolo_time_queue else 0
                        avg_ir_ms = np.mean(list(self.ir_time_queue)) if self.ir_time_queue else 0

                        print(
                            f"帧#{self.total_frames:4d} | "
                            f"模式: {raw['detection_method']:12s} | "
                            f"位置: ({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f})mm | "
                            f"速度: ({vel[0]:6.0f}, {vel[1]:6.0f}, {vel[2]:6.0f})mm/s | "
                            f"FPS: {avg_fps:4.1f} | "
                            f"YOLO: {avg_yolo_ms:4.1f}ms | "
                            f"IR: {avg_ir_ms:3.1f}ms"
                        )
                    else:
                        avg_fps = np.mean(list(self.fps_queue)) if self.fps_queue else 0
                        print(f"帧#{self.total_frames:4d} | 未检测到 | FPS: {avg_fps:4.1f}")

        except KeyboardInterrupt:
            print("\n检测已停止")
        finally:
            self._print_stats()
            if video_writer is not None:
                video_writer.release()
                print(f"原始视频已保存: {video_path}")
            if enable_viz:
                cv2.destroyAllWindows()
            self.close()

    def close(self):
        """Release the RealSense pipeline held by this detector."""
        color_pipeline = getattr(self, 'color_pipeline', None)
        if color_pipeline is not None:
            try:
                color_pipeline.stop()
            except Exception:
                pass
            self.color_pipeline = None
        pipeline = getattr(self, 'pipeline', None)
        if pipeline is None:
            return
        try:
            pipeline.stop()
        except Exception:
            pass
        self.pipeline = None

    def _render_visualization(self, rgb_image, ir_image, observation):
        """
        渲染可视化窗口

        布局：
        ┌─────────────┬─────────────┐
        │   RGB图像    │   IR图像     │
        │  (YOLO bbox) │ (检测结果)   │
        └─────────────┴─────────────┘
        """
        rgb_image = self._rgb_on_ir_canvas(rgb_image)
        h, w = rgb_image.shape[:2]

        # RGB图像 - 绘制YOLO bbox和检测结果
        rgb_display = rgb_image.copy()

        if observation['detected']:
            raw = observation['raw_measurement']

            # 绘制YOLO bbox（如果有）
            yolo_center = None
            if raw.get('yolo_bbox') is not None:
                x1, y1, x2, y2 = raw['yolo_bbox']
                # YOLO用于辅助过滤 - 画细线矩形
                cv2.rectangle(rgb_display, (int(x1), int(y1)), (int(x2), int(y2)),
                            (255, 255, 0), 1)
                yolo_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                # 画小空心圆表示YOLO中心（仅供参考）
                cv2.circle(rgb_display, (int(yolo_center[0]), int(yolo_center[1])),
                          4, (255, 255, 0), 1)
                cv2.putText(rgb_display, 'YOLO Filter', (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # 绘制ROI（如果有）
            if raw.get('roi') is not None:
                x1, y1, x2, y2 = raw['roi']
                cv2.rectangle(rgb_display, (x1, y1), (x2, y2),
                            (0, 255, 255), 1)
                cv2.putText(rgb_display, 'IR Search ROI', (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # 绘制IR检测到的球心（主要数据源）
            # 使用精确转换后的RGB坐标
            cx_rgb = raw.get('cx_rgb')
            cy_rgb = raw.get('cy_rgb')

            # 如果有原始检测点（卡尔曼滤波前），用小圆点标记
            cx_raw = raw.get('cx_raw')
            cy_raw = raw.get('cy_raw')
            if cx_raw is not None and cy_raw is not None:
                # 将IR原始坐标也转换到RGB显示
                # 这里简化处理：假设偏移量相同
                offset_x = cx_rgb - raw['cx'] if cx_rgb else 0
                offset_y = cy_rgb - raw['cy'] if cy_rgb else 0
                cx_raw_display = int(cx_raw + offset_x)
                cy_raw_display = int(cy_raw + offset_y)
                # 画小红点表示原始检测（卡尔曼滤波前）
                cv2.circle(rgb_display, (cx_raw_display, cy_raw_display), 3, (0, 0, 255), -1)
                # 画虚线连接原始点和平滑点
                if abs(cx_raw - raw['cx']) > 0.5 or abs(cy_raw - raw['cy']) > 0.5:
                    cv2.line(rgb_display,
                            (cx_raw_display, cy_raw_display),
                            (int(cx_rgb) if cx_rgb else int(raw['cx']),
                             int(cy_rgb) if cy_rgb else int(raw['cy'])),
                            (128, 128, 255), 1, cv2.LINE_AA)

            if cx_rgb is not None and cy_rgb is not None:
                # 转换成功，使用精确的RGB坐标（已经过卡尔曼平滑）
                cx_display = int(cx_rgb)
                cy_display = int(cy_rgb)
                label = 'IR KF' if self.pixel_kalman_enabled else 'IR raw'
                color = (0, 255, 0)  # 绿色
            else:
                # 转换失败，回退到IR原始坐标（会有偏移）
                cx_display = int(raw['cx'])
                cy_display = int(raw['cy'])
                label = 'IR (Approx)'
                color = (0, 200, 200)  # 青绿色表示近似

            morphology_radius = int(round(raw['radius']))
            radius = int(round(raw.get('projected_radius_px')
                               or raw['radius']))

            # IR是主要数据源 - 粗线+实心圆（这是卡尔曼平滑后的）
            cv2.circle(rgb_display, (cx_display, cy_display), radius, color, 2)
            cv2.circle(rgb_display, (cx_display, cy_display), 5, color, -1)
            cv2.putText(rgb_display, label, (cx_display + 15, cy_display - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 绘制平滑后的橙色圆（如果有）
            smoothed_cx = raw.get('smoothed_cx')
            smoothed_cy = raw.get('smoothed_cy')
            smoothed_radius = raw.get('smoothed_radius')
            if (smoothed_cx is not None and smoothed_cy is not None
                and smoothed_radius is not None):
                # 只在平滑值与原始值不同时绘制橙色圆
                if (abs(smoothed_cx - cx_display) > 1 or
                    abs(smoothed_cy - cy_display) > 1 or
                    abs(smoothed_radius - morphology_radius) > 1):
                    cv2.circle(rgb_display,
                              (int(round(smoothed_cx)), int(round(smoothed_cy))),
                              int(round(smoothed_radius)),
                              (0, 165, 255), 1, cv2.LINE_AA)  # 橙色细线
                    cv2.putText(rgb_display, 'Smoothed',
                               (int(smoothed_cx) + 15, int(smoothed_cy) + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # 如果YOLO和IR都存在，显示它们的关系
            if yolo_center is not None and cx_rgb is not None:
                offset_x = cx_display - yolo_center[0]
                offset_y = cy_display - yolo_center[1]
                offset_dist = np.sqrt(offset_x**2 + offset_y**2)

                # 画细灰线连接YOLO和IR中心
                if offset_dist > 5:
                    cv2.line(rgb_display,
                            (int(yolo_center[0]), int(yolo_center[1])),
                            (cx_display, cy_display),
                            (128, 128, 128), 1)

            # 显示信息
            pos = observation['position']
            vel = observation['velocity']

            # 获取卡尔曼滤波统计
            kf_stats = self.pixel_kf.get_stats()

            info_lines = [
                f"Detector Pos[{self.output_frame}]: "
                f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})mm",
                f"Detector Vel[{self.output_frame}]: "
                f"({vel[0]:.0f}, {vel[1]:.0f}, {vel[2]:.0f})mm/s",
                f"Mode: {raw['detection_method']}",
                f"Depth: {raw['depth_mm']:.0f}mm ({raw['depth_method']}; "
                f"mode={raw.get('depth_mode', self.depth_mode)})",
                f"R morph/physical: {raw['radius']:.2f}/"
                f"{raw.get('projected_radius_px', raw['radius']):.2f}px  "
                f"sensor/geom: "
                f"{raw.get('sensor_depth_mm') if raw.get('sensor_depth_mm') is not None else '-'} / "
                f"{raw.get('geom_depth_mm') if raw.get('geom_depth_mm') is not None else '-'}mm",
                (
                    f"Pixel KF: speed={kf_stats['speed_px_s']:.1f}px/s "
                    f"outlier={kf_stats['outlier_rate_pct']:.1f}%"
                    if self.pixel_kalman_enabled else "Pixel KF: OFF (raw IR)"
                ),
            ]
            if raw.get('sensor_surface_to_center_depth_applied'):
                info_lines.append(
                    "Sensor surface->center: "
                    f"{raw.get('sensor_surface_depth_mm'):.1f}->"
                    f"{raw.get('sensor_center_depth_mm'):.1f}mm "
                    f"(delta={raw.get('sensor_center_depth_correction_mm'):.1f}mm"
                    f"{' clamped' if raw.get('sensor_center_depth_calibration_clamped') else ''})")
            if raw.get('depth_mode') == 'adaptive':
                sensor_weight = raw.get('depth_sensor_weight')
                disagreement = raw.get('depth_disagreement_mm')
                sensor_jump = raw.get('depth_sensor_jump_mm')
                radius_mismatch = raw.get('depth_radius_mismatch_ratio')
                prediction = raw.get('depth_prediction_mm')
                rejection = raw.get('depth_rejection_reason')
                info_lines.append(
                    "Adaptive depth: "
                    f"sensor={sensor_weight:.0f} "
                    f"jump={sensor_jump:.0f}mm "
                    f"pred={prediction:.0f}mm "
                    f"s-g={disagreement:.0f}mm "
                    f"r-gap={100.0 * radius_mismatch:.1f}% "
                    f"reject={rejection or '-'}"
                    if (sensor_weight is not None
                        and disagreement is not None
                        and sensor_jump is not None
                        and radius_mismatch is not None
                        and prediction is not None)
                    else f"Adaptive depth: reject={rejection or '-'}")

            # 添加调试信息（可选显示IR和RGB坐标系位置）
            if 'raw_position_ir' in raw and 'raw_position_rgb' in raw:
                pos_ir = raw['raw_position_ir']
                pos_rgb = raw['raw_position_rgb']
                # info_lines.append(f"IR: ({pos_ir[0]:.0f},{pos_ir[1]:.0f},{pos_ir[2]:.0f})")
                # info_lines.append(f"RGB: ({pos_rgb[0]:.0f},{pos_rgb[1]:.0f},{pos_rgb[2]:.0f})")

            y_offset = 30
            for line in info_lines:
                cv2.putText(rgb_display, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25

        # IR图像 - 伪彩色显示
        ir_colored = cv2.applyColorMap(ir_image, cv2.COLORMAP_JET)

        raw_candidate = observation.get('raw_measurement')
        if (isinstance(raw_candidate, dict)
                and raw_candidate.get('cx') is not None
                and raw_candidate.get('cy') is not None
                and raw_candidate.get('radius') is not None):
            raw = raw_candidate
            cx, cy = int(raw['cx']), int(raw['cy'])
            morphology_radius = int(round(raw['radius']))
            radius = int(round(raw.get('projected_radius_px')
                               or raw.get(
                                   'depth_sensor_projected_radius_px')
                               or raw['radius']))
            projected_color = (
                (0, 255, 0)
                if observation.get('detected', False)
                else (0, 0, 255))
            cv2.circle(
                ir_colored, (cx, cy), radius, projected_color, 2)
            if abs(morphology_radius - radius) > 1:
                cv2.circle(ir_colored, (cx, cy), morphology_radius,
                           (0, 165, 255), 1)
            cv2.circle(ir_colored, (cx, cy), 3, (0, 0, 255), -1)

        # 拼接左右
        display = np.hstack([rgb_display, ir_colored])

        # 顶部状态栏
        status_bar = np.zeros((60, display.shape[1], 3), dtype=np.uint8)
        avg_fps = np.mean(list(self.fps_queue)) if self.fps_queue else 0

        status_text = f"FPS: {avg_fps:.1f} | Frame: {self.total_frames} | Detection: {'YES' if observation['detected'] else 'NO'}"
        cv2.putText(status_bar, status_text, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        final_display = np.vstack([status_bar, display])

        return final_display

    def _print_stats(self):
        """打印统计"""
        print("\n" + "="*70)
        print("混合检测器统计")
        print("="*70)
        print(f"总帧数: {self.total_frames}")
        print(f"检测成功: {self.detected_frames} ({100*self.detected_frames/max(self.total_frames,1):.1f}%)")
        print(f"\nYOLO检测次数: {self.yolo_detections}")
        print(f"IR检测次数: {self.ir_detections}")
        print(f"混合模式(YOLO+IR): {self.hybrid_detections} ({100*self.hybrid_detections/max(self.detected_frames,1):.1f}%)")
        print(f"回退模式(纯IR): {self.fallback_detections} ({100*self.fallback_detections/max(self.detected_frames,1):.1f}%)")

        if self.fps_queue:
            print(f"\n平均FPS: {np.mean(list(self.fps_queue)):.1f}")
        if self.yolo_time_queue:
            print(f"YOLO平均延迟: {np.mean(list(self.yolo_time_queue)):.1f}ms")
        if self.ir_time_queue:
            print(f"IR平均延迟: {np.mean(list(self.ir_time_queue)):.1f}ms")
        print("="*70)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description='RGB+IR混合检测器')
    parser.add_argument('--model', required=True, help='YOLO模型路径')
    parser.add_argument('--yolo-interval', type=int, default=3,
                       help='YOLO检测间隔（帧数）')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                       help='YOLO置信度阈值（默认0.25）')
    parser.add_argument('--yolo-class', type=str, default='Class 0',
                       help='YOLO目标类别名称（默认Class 0，自定义模型）')
    parser.add_argument('--laser-power', type=int, default=None,
                       help='激光功率(mW)；不给则用JSON的laser_power，再无则150')
    parser.add_argument('--params', default='ir_params.json',
                       help='IR参数文件（从ir_tuning_tool保存的）')
    parser.add_argument('--print-every', type=int, default=10)
    parser.add_argument('--no-viz', action='store_true',
                       help='禁用可视化窗口')
    parser.add_argument('--save-video', action='store_true',
                       help='保存原始视频')
    parser.add_argument('--output-dir', default='outputs/videos',
                       help='视频保存目录')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        exit(1)

    # 加载IR参数（如果有）
    ir_params = {}
    json_laser_power = None
    if os.path.exists(args.params):
        import json
        with open(args.params, 'r') as f:
            loaded_params = json.load(f)
        # HybridBallDetector 构造函数认得的参数键（laser_power 单独走命令行覆盖）
        ir_param_keys = ['ir_threshold', 'use_adaptive', 'dilate_iterations',
                       'close_kernel_size', 'denoise_kernel_size', 'min_radius',
                       'max_radius', 'circularity_min',
                       'exposure', 'gain', 'brightness_percentile']
        ir_params = {k: v for k, v in loaded_params.items() if k in ir_param_keys}
        json_laser_power = loaded_params.get('laser_power')
        # 显式报告被忽略的键，避免像以前那样静默丢弃
        known = set(ir_param_keys) | {'laser_power'}
        ignored = {k: v for k, v in loaded_params.items() if k not in known}
        print(f"从 {args.params} 加载IR参数: {ir_params}")
        if ignored:
            print(f"⚠️  以下参数未被检测器使用，已忽略: {ignored}")

    # 激光功率优先级：命令行 > JSON > 默认150
    if args.laser_power is not None:
        laser_power = args.laser_power
    elif json_laser_power is not None:
        laser_power = json_laser_power
    else:
        laser_power = 150

    try:
        detector = HybridBallDetector(
            yolo_model_path=args.model,
            yolo_interval=args.yolo_interval,
            yolo_conf=args.yolo_conf,
            target_class=args.yolo_class,  # 添加target_class参数
            laser_power=laser_power,
            print_every=args.print_every,
            **ir_params
        )
        detector.run(
            enable_viz=not args.no_viz,
            save_raw_video=args.save_video,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\n已退出")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
