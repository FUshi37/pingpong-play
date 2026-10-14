#!/home/jacky/miniconda3/envs/gmr/bin/python
"""
乒乓球实时检测与定位系统
YOLO segmentation + 单目几何测距 + 可选 One Euro / Kalman 滤波 + 拖影修正
目标频率: 40-60Hz
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import time
from collections import deque
from ultralytics import YOLO
from typing import Optional

from pingpong_controller.calibration_loader import CalibrationData


class KalmanFilter:
    """
    6维卡尔曼滤波器：[x, y, z, vx, vy, vz]
    用于平滑球的位置并估计速度
    """

    def __init__(self, dt=1 / 60.0, camera_tilt_deg=0.0):
        """
        Args:
            dt: 时间步长（秒），默认60Hz
            camera_tilt_deg: 相机绕X轴旋转角度（度）
                           正值=向下俯视，负值=向上仰视
        """
        self.dt = dt
        self.camera_tilt_deg = camera_tilt_deg
        self.initialized = False

        # 状态向量 [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)

        # 状态协方差矩阵
        self.P = np.eye(6) * 1000

        # 状态转移矩阵（匀速运动模型 + 重力）
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # 过程噪声协方差（球运动的不确定性）
        q = 100  # 调整这个值来平衡平滑度和响应速度
        self.Q = np.eye(6) * q
        self.Q[3:, 3:] *= 10  # 速度的不确定性更大

        # 测量矩阵（只测量位置）
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # 测量噪声协方差
        r = 25  # 测量噪声（mm）
        self.R = np.eye(3) * (r ** 2)

        # 计算重力在相机坐标系中的分量
        # 重力加速度 9800 mm/s^2
        g = 9800
        tilt_rad = np.radians(camera_tilt_deg)

        # 相机坐标系：X右，Y下，Z前
        # 相机绕X轴旋转（正值=向下俯视）
        # 重力向下，在旋转后相机系中的分量：
        # - X分量：0（重力不沿X方向）
        # - Y分量：g * cos(tilt)（Y轴与重力夹角）
        # - Z分量：g * sin(tilt)（Z轴向前下倾斜时，重力有正Z分量）
        self.gravity = np.array([
            0,
            g * np.cos(tilt_rad) * dt,
            g * np.sin(tilt_rad) * dt
        ])
        self.last_innovation = None

    def _update_motion_model(self, dt):
        """根据真实时间间隔更新运动模型。"""
        dt = max(float(dt), 1e-3)
        self.dt = dt
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        g = 9800
        tilt_rad = np.radians(self.camera_tilt_deg)
        self.gravity = np.array([
            0,
            g * np.cos(tilt_rad) * dt,
            g * np.sin(tilt_rad) * dt
        ])

    def predict(self, dt=None):
        """预测下一状态"""
        if dt is not None:
            self._update_motion_model(dt)

        # 状态预测
        self.state = self.F @ self.state

        # 添加重力影响
        self.state[3:6] += self.gravity

        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state.copy()

    def innovation(self, measurement, dt=None):
        """
        计算当前测量相对于预测状态的残差，但不更新滤波器。
        """
        z = np.array(measurement, dtype=float)
        predicted_state = self.state.copy()
        predicted_P = self.P.copy()

        if self.initialized:
            if dt is not None:
                self._update_motion_model(dt)
            predicted_state = self.F @ predicted_state
            predicted_state[3:6] += self.gravity
            predicted_P = self.F @ predicted_P @ self.F.T + self.Q

        innovation = z - self.H @ predicted_state
        self.last_innovation = innovation.copy()
        return innovation, predicted_state, predicted_P

    def update(self, measurement, dt=None, measurement_noise_mm=None):
        """
        更新状态

        Args:
            measurement: [x, y, z] 位置测量值（mm）

        Returns:
            state: 更新后的状态 [x, y, z, vx, vy, vz]
        """
        z = np.array(measurement)

        if not self.initialized:
            # 首次初始化
            self.state[:3] = z
            self.state[3:] = 0
            self.initialized = True
            self.last_innovation = np.zeros(3)
            return self.state.copy()

        # 预测
        self.predict(dt=dt)

        R = self.R
        if measurement_noise_mm is not None:
            R = np.eye(3) * (float(measurement_noise_mm) ** 2)

        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态
        y = z - self.H @ self.state  # 测量残差
        self.last_innovation = y.copy()
        self.state = self.state + K @ y

        # 更新协方差
        identity = np.eye(6)
        self.P = (identity - K @ self.H) @ self.P

        return self.state.copy()

    def get_state(self):
        """获取当前状态"""
        return self.state.copy()

    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1000


class OneEuroFilter:
    """
    One Euro Filter — 低延迟自适应低通滤波器。
    分别用于 cx_px, cy_px, radius_px 的源头平滑。
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _smoothing_factor(t_e, cutoff):
        r = 2.0 * np.pi * cutoff * t_e
        return r / (r + 1.0)

    def __call__(self, t, x):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        t_e = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class PingPongBallDetector:
    def __init__(self,
                 yolo_model_path,
                 ball_diameter_mm=40.0,
                 width=1280,
                 height=720,
                 fps=30,
                 conf_threshold=0.5,
                 target_class='orange',
                 camera_tilt_deg=0.0,
                 use_kalman=True,
                 infer_imgsz=None,
                 device=None,
                 half=False,
                 use_undistort=True,
                 print_every=1,
                 profile_every=0,
                 use_camera=True,
                 camera_fx=None,
                 camera_fy=None,
                 camera_cx=None,
                 camera_cy=None,
                 camera_dist_coeffs=None,
                 calibration_data: Optional[CalibrationData] = None,
                 output_frame='base_link'):
        """
        初始化乒乓球检测器

        Args:
            yolo_model_path: YOLOv11分割模型路径
            ball_diameter_mm: 乒乓球直径(毫米)，默认40mm
            width: 图像宽度
            height: 图像高度
            fps: 目标帧率
            conf_threshold: YOLO置信度阈值
            target_class: 目标类别名称（如'sports ball'）
            camera_tilt_deg: 相机绕X轴旋转角度（度）
                           正值=向下俯视，负值=向上仰视
                           例如：40.0表示向下俯视40度
            use_kalman: 是否使用卡尔曼滤波（测量精度时可关闭）
            use_camera: 是否启动RealSense相机（False时需提供相机内参）
            camera_fx, camera_fy, camera_cx, camera_cy: 相机内参（use_camera=False时必需）
            camera_dist_coeffs: 畸变系数数组（use_camera=False时可选，默认无畸变）
            calibration_data: CalibrationData对象，包含相机内参和外参（优先级最高）
            output_frame: 输出坐标系 'camera' 或 'base_link'
        """
        self.ball_diameter_mm = ball_diameter_mm
        self.ball_radius_mm = ball_diameter_mm / 2.0
        self.conf_threshold = conf_threshold
        self.target_class = target_class
        self.camera_tilt_deg = camera_tilt_deg
        self.use_kalman = use_kalman
        self.infer_imgsz = infer_imgsz
        self.device = device
        self.half = half
        self.use_undistort = use_undistort
        self.print_every = max(1, int(print_every))
        self.profile_every = max(0, int(profile_every))
        self.profile_samples = []
        self.last_yolo_ms = 0.0
        self.use_camera = use_camera

        # calibration_data 模式下强制启用去畸变
        if calibration_data is not None and not self.use_undistort:
            print("警告: calibration_data 模式下必须启用去畸变，已强制 use_undistort=True")
            self.use_undistort = True

        # 加载YOLO模型
        print(f"加载YOLO模型: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        print("YOLO模型加载完成")
        print(f"目标类别: {self.target_class}")
        print(f"相机倾斜角度: {self.camera_tilt_deg}°")
        if self.infer_imgsz is not None:
            print(f"YOLO推理尺寸: {self.infer_imgsz}")
        if self.device is not None:
            print(f"YOLO推理设备: {self.device}")
        if self.half:
            print("YOLO半精度推理: enabled")

        # 坐标系输出配置
        self.output_frame = output_frame
        self.calibration_data = calibration_data
        self.T_base_camera = None
        if calibration_data is not None:
            self.T_base_camera = calibration_data.T_base_camera
            self.R_base_camera = self.T_base_camera[:3, :3]

        # RealSense相机配置或外部相机内参
        self.pipeline = None
        if calibration_data is not None:
            # 从 CalibrationData 加载所有相机参数（最高优先级）
            self.camera_matrix_original = calibration_data.original_camera_matrix.copy()
            self.dist_coeffs = calibration_data.original_dist_coeffs.reshape(1, -1)
            self.camera_matrix_undistorted = calibration_data.camera_matrix.copy()
            self.image_width = calibration_data.image_width
            self.image_height = calibration_data.image_height
            self.intrinsics_original = None

            self.fx_original = self.camera_matrix_original[0, 0]
            self.fy_original = self.camera_matrix_original[1, 1]
            self.cx_original = self.camera_matrix_original[0, 2]
            self.cy_original = self.camera_matrix_original[1, 2]

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix_original,
                self.dist_coeffs,
                None,
                self.camera_matrix_undistorted,
                (self.image_width, self.image_height),
                cv2.CV_32FC1
            )

            self.fx = self.camera_matrix_undistorted[0, 0]
            self.fy = self.camera_matrix_undistorted[1, 1]
            self.cx = self.camera_matrix_undistorted[0, 2]
            self.cy = self.camera_matrix_undistorted[1, 2]

            print(f"从 CalibrationData 加载参数:")
            print(f"  原始内参: fx={self.fx_original:.2f}, fy={self.fy_original:.2f}, "
                  f"cx={self.cx_original:.2f}, cy={self.cy_original:.2f}")
            print(f"  去畸变内参: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                  f"cx={self.cx:.2f}, cy={self.cy:.2f}")
            print(f"  T_base_camera loaded, output_frame={self.output_frame}")

        elif self.use_camera:
            self.pipeline = rs.pipeline()
            profile = self._start_color_stream(width, height, fps)
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

            self.intrinsics_original = self.intrinsics
            self.fx_original = self.intrinsics.fx
            self.fy_original = self.intrinsics.fy
            self.cx_original = self.intrinsics.ppx
            self.cy_original = self.intrinsics.ppy
            self.image_width = self.intrinsics.width
            self.image_height = self.intrinsics.height

            self.camera_matrix_original = np.array([
                [self.fx_original, 0, self.cx_original],
                [0, self.fy_original, self.cy_original],
                [0, 0, 1]
            ])

            distortion_model = self.intrinsics.model
            distortion_coeffs = np.array(self.intrinsics.coeffs)
            if distortion_model == rs.distortion.brown_conrady:
                self.dist_coeffs = distortion_coeffs[:5].reshape(1, 5)
            else:
                print(f"警告: 未知畸变模型 {distortion_model}，假设为 Brown-Conrady")
                self.dist_coeffs = distortion_coeffs[:5].reshape(1, 5)

            self.camera_matrix_undistorted, self.roi_undistort = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix_original,
                self.dist_coeffs,
                (self.image_width, self.image_height),
                alpha=0.0,
                newImgSize=(self.image_width, self.image_height)
            )

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix_original,
                self.dist_coeffs,
                None,
                self.camera_matrix_undistorted,
                (self.image_width, self.image_height),
                cv2.CV_32FC1
            )

            self.fx = self.camera_matrix_undistorted[0, 0]
            self.fy = self.camera_matrix_undistorted[1, 1]
            self.cx = self.camera_matrix_undistorted[0, 2]
            self.cy = self.camera_matrix_undistorted[1, 2]

        else:
            if camera_fx is None or camera_fy is None or camera_cx is None or camera_cy is None:
                raise ValueError(
                    "use_camera=False 时必须提供相机内参: camera_fx, camera_fy, camera_cx, camera_cy"
                )
            self.fx_original = float(camera_fx)
            self.fy_original = float(camera_fy)
            self.cx_original = float(camera_cx)
            self.cy_original = float(camera_cy)
            self.image_width = width
            self.image_height = height
            self.intrinsics_original = None
            print(f"使用外部相机内参: fx={self.fx_original:.2f}, fy={self.fy_original:.2f}, "
                  f"cx={self.cx_original:.2f}, cy={self.cy_original:.2f}")

            self.camera_matrix_original = np.array([
                [self.fx_original, 0, self.cx_original],
                [0, self.fy_original, self.cy_original],
                [0, 0, 1]
            ])

            if camera_dist_coeffs is not None:
                self.dist_coeffs = np.array(camera_dist_coeffs).reshape(1, -1)
            else:
                self.dist_coeffs = np.zeros((1, 5))

            self.camera_matrix_undistorted, self.roi_undistort = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix_original,
                self.dist_coeffs,
                (self.image_width, self.image_height),
                alpha=0.0,
                newImgSize=(self.image_width, self.image_height)
            )

            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix_original,
                self.dist_coeffs,
                None,
                self.camera_matrix_undistorted,
                (self.image_width, self.image_height),
                cv2.CV_32FC1
            )

            self.fx = self.camera_matrix_undistorted[0, 0]
            self.fy = self.camera_matrix_undistorted[1, 1]
            self.cx = self.camera_matrix_undistorted[0, 2]
            self.cy = self.camera_matrix_undistorted[1, 2]

        # 圆检测参数
        self.min_radius = 10
        self.max_radius = 200

        # 桔黄色HSV范围（用于颜色验证）
        # 这里放宽了 S/V 下限，减少阴影和局部高光导致的漏检。
        self.hsv_lower = np.array([5, 50, 50])
        self.hsv_upper = np.array([30, 255, 255])
        self.color_match_threshold = 0.3  # 至少30%的像素匹配颜色

        # 性能统计
        self.fps_queue = deque(maxlen=30)
        self.last_observation_time = None
        self.total_frames = 0
        self.detected_frames = 0
        self.missed_detection_frames = 0
        self.max_measurement_jump_mm = 250.0
        self.max_depth_jump_mm = 180.0
        self.max_prediction_frames = 6
        self.missed_frames = 0

        # 位置历史（用于速度差分计算，只存储可信的平滑后位置）
        self.position_history = deque(maxlen=5)  # 存储 (timestamp, [x, y, z])

        # 简单 EMA 位置平滑器（替代复杂的卡尔曼预测）
        self.smoothed_position = None
        self.position_smooth_alpha = 0.4  # EMA 平滑系数，越小越平滑

        # ROI 跟踪（用于 fallback 检测）
        self.last_detection_pixel = None  # (cx, cy) 上一帧检测到的像素位置
        self.roi_margin = 150  # ROI 扩展边距（像素）

        # One Euro Filter 源头平滑器（cx_px, cy_px, radius_px）
        self.oef_cx = OneEuroFilter(min_cutoff=1.5, beta=0.02)
        self.oef_cy = OneEuroFilter(min_cutoff=1.5, beta=0.02)
        self.oef_radius = OneEuroFilter(min_cutoff=0.8, beta=0.01)
        # 异常值检测阈值
        self.max_center_jump_px = 80.0
        self.max_radius_change_ratio = 0.35

        # 拖影检测与修正
        self.raw_detection_history = deque(maxlen=5)  # 存储 (cx, cy, radius)
        self.motion_blur_aspect_threshold = 1.6
        self.motion_blur_circularity_threshold = 0.65
        self.motion_blur_radius_ratio_threshold = 1.35
        self.min_motion_speed_px = 5.0  # 最小运动速度（px/frame）

        # 初始化卡尔曼滤波器（默认不使用，--kalman 可开启）
        if self.use_kalman:
            self.kalman = KalmanFilter(
                dt=1.0 / fps, camera_tilt_deg=camera_tilt_deg)
            print("卡尔曼滤波器已初始化")
        else:
            self.kalman = None
            print("源头 One Euro Filter 平滑已启用")

        print(
            f"相机分辨率: {self.image_width}x{self.image_height} | "
            f"原始内参: fx={self.fx_original:.2f}, fy={self.fy_original:.2f}, "
            f"cx={self.cx_original:.2f}, cy={self.cy_original:.2f}"
        )

        # 如果有 calibration_data，内参已经正确设置，不应被覆盖
        if calibration_data is not None:
            print(
                f"去畸变内参 (from CalibrationData): fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
            print(f"输出坐标系: {self.output_frame}")
        elif self.use_undistort:
            print(
                f"去畸变内参: fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}"
            )
        else:
            # 只有在没有 calibration_data 且 use_undistort=False 时才退回原始内参
            self.fx = self.fx_original
            self.fy = self.fy_original
            self.cx = self.cx_original
            self.cy = self.cy_original
            print("去畸变处理已禁用，几何计算使用原始内参")

        print(f"畸变系数: {self.dist_coeffs.flatten().tolist()}")
        print("当前使用纯单目几何测距")
        print(f"YOLO置信度阈值: {self.conf_threshold}")
        print(f"颜色匹配阈值: {self.color_match_threshold}")
        print(
            f"HSV阈值: lower={self.hsv_lower.tolist()}, upper={self.hsv_upper.tolist()}")

        # 预热相机，丢弃前几帧（仅在use_camera=True时）
        if self.use_camera:
            print("相机预热中...")
            for i in range(30):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        test_img = np.asanyarray(color_frame.get_data())
                        if i == 0:
                            print(
                                f"图像格式: shape={test_img.shape}, dtype={test_img.dtype}")
                except Exception as e:
                    print(f"预热帧 {i} 失败: {e}")
            print("相机就绪")
        else:
            print("外部图像模式，跳过相机预热")

    def _start_color_stream(self, width, height, fps):
        """启动 RGB 流，优先使用请求配置，失败时自动回退到常见配置。"""
        candidate_modes = [
            (width, height, fps),
            (width, height, 30),
            (640, 480, 30),
        ]

        tried = set()
        last_error = None

        for mode in candidate_modes:
            if mode in tried:
                continue
            tried.add(mode)
            try_width, try_height, try_fps = mode
            config = rs.config()
            config.enable_stream(
                rs.stream.color, try_width, try_height, rs.format.bgr8, try_fps
            )
            try:
                profile = self.pipeline.start(config)
                if mode != (width, height, fps):
                    print(
                        "请求的 RGB 配置不可用，已自动回退到 "
                        f"{try_width}x{try_height}@{try_fps}"
                    )
                return profile
            except RuntimeError as exc:
                last_error = exc

        raise RuntimeError(f"无法启动 RealSense RGB 流: {last_error}")

    def detect_ball_with_yolo(self, rgb_image):
        """
        使用YOLO分割模型检测球并返回mask（只检测sports ball类别）

        Args:
            rgb_image: BGR格式图像

        Returns:
            mask: 二值mask，如果未检测到返回None
        """
        # YOLO推理
        t_yolo = time.time()
        predict_kwargs = {
            "conf": self.conf_threshold,
            "verbose": False,
        }
        if self.infer_imgsz is not None:
            predict_kwargs["imgsz"] = self.infer_imgsz
        if self.device is not None:
            predict_kwargs["device"] = self.device
        if self.half:
            predict_kwargs["half"] = True
        results = self.yolo_model(rgb_image, **predict_kwargs)
        self.last_yolo_ms = (time.time() - t_yolo) * 1000.0

        # 检查是否有检测结果
        if len(results) == 0 or results[0].masks is None:
            return None

        # 获取所有检测结果
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes

        if len(masks) == 0:
            return None

        # 获取类别名称
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        # 过滤出目标类别（sports ball）
        ball_indices = []
        for i, cls_id in enumerate(class_ids):
            class_name = self.yolo_model.names[cls_id]
            if class_name == self.target_class:
                ball_indices.append(i)

        if len(ball_indices) == 0:
            return None

        # 取置信度最高的球
        ball_confidences = confidences[ball_indices]
        best_ball_idx = ball_indices[np.argmax(ball_confidences)]
        mask = masks[best_ball_idx]

        # 调整mask大小到原图尺寸
        mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]))
        mask = (mask > 0.5).astype(np.uint8) * 255

        return mask

    def detect_ball_by_color(self, rgb_image, roi=None):
        """
        纯 HSV 颜色检测 fallback，当 YOLO 漏检时使用。

        Args:
            rgb_image: BGR 格式图像
            roi: (x, y, w, h) 感兴趣区域，None 则全图搜索

        Returns:
            mask: 二值 mask（与 YOLO 返回格式一致），未检测到返回 None
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        if roi is not None:
            rx, ry, rw, rh = roi
            roi_mask = np.zeros_like(color_mask)
            roi_mask[ry:ry + rh, rx:rx + rw] = 255
            color_mask = cv2.bitwise_and(color_mask, roi_mask)

        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 100:
            return None

        _, radius = cv2.minEnclosingCircle(largest)
        if radius < self.min_radius or radius > self.max_radius:
            return None

        circle_area = np.pi * radius * radius
        circularity = area / circle_area if circle_area > 0 else 0
        if circularity < 0.4:
            return None

        mask = np.zeros(rgb_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
        return mask

    def verify_color(self, rgb_image, mask):
        """
        验证mask区域是否为桔黄色

        Args:
            rgb_image: BGR格式图像
            mask: 二值mask

        Returns:
            bool: 是否匹配桔黄色
            float: 匹配比例
        """
        # 转换到HSV
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # 计算mask区域内有多少像素
        mask_pixels = np.sum(mask > 0)
        if mask_pixels == 0:
            return False, 0.0

        # 颜色阈值分割
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # 计算mask区域内匹配颜色的像素
        color_match = cv2.bitwise_and(color_mask, mask)
        color_match_pixels = np.sum(color_match > 0)

        # 计算匹配比例
        match_ratio = color_match_pixels / mask_pixels

        # 判断是否匹配
        is_match = match_ratio >= self.color_match_threshold

        return is_match, match_ratio

    def extract_colored_ball_region(self, rgb_image, mask):
        """
        在 YOLO mask 内进一步提取桔黄色连通区域。

        Returns:
            refined_mask: 颜色细化后的mask，失败返回None
            bbox: (x, y, w, h) 外接框，失败返回None
        """
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        # 先把 YOLO mask 轻微收缩，尽量去掉边缘背景和毛刺。
        erode_kernel = np.ones((5, 5), np.uint8)
        inner_mask = cv2.erode(mask, erode_kernel, iterations=1)
        if np.sum(inner_mask > 0) < 50:
            inner_mask = mask

        hsv_pixels = hsv[inner_mask > 0]
        if len(hsv_pixels) < 30:
            return None, None

        # 只用更“像球面本体”的高饱和/中高亮像素估主色，减少阴影和高光污染。
        stable_pixels = hsv_pixels[(
            hsv_pixels[:, 1] > 60) & (hsv_pixels[:, 2] > 50)]
        if len(stable_pixels) < 20:
            stable_pixels = hsv_pixels

        hue_center = int(np.median(stable_pixels[:, 0]))
        sat_low = int(max(30, np.percentile(stable_pixels[:, 1], 15) - 10))
        val_low = int(max(30, np.percentile(stable_pixels[:, 2], 10) - 10))

        hue_margin = 12
        lower_h = max(0, hue_center - hue_margin)
        upper_h = min(179, hue_center + hue_margin)

        adaptive_lower = np.array([lower_h, sat_low, val_low], dtype=np.uint8)
        adaptive_upper = np.array([upper_h, 255, 255], dtype=np.uint8)

        color_mask = cv2.inRange(hsv, adaptive_lower, adaptive_upper)
        masked_color = cv2.bitwise_and(color_mask, inner_mask)

        kernel = np.ones((5, 5), np.uint8)
        masked_color = cv2.morphologyEx(masked_color, cv2.MORPH_OPEN, kernel)
        masked_color = cv2.morphologyEx(masked_color, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            masked_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50:
            return None, None

        refined_mask = np.zeros_like(mask)
        cv2.drawContours(
            refined_mask, [largest_contour], -1, 255, thickness=-1)
        bbox = cv2.boundingRect(largest_contour)
        return refined_mask, bbox

    def fit_circle(self, mask):
        """
        从mask拟合圆，得到图像圆心和半径

        Args:
            mask: 二值mask

        Returns:
            (center_x, center_y, radius_px) 或 None
        """
        # 查找轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 轮廓面积过小则忽略
        if cv2.contourArea(largest_contour) < 100:
            return None

        # 拟合最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

        # 半径范围检查
        if radius < self.min_radius or radius > self.max_radius:
            return None

        return (cx, cy, radius)

    def recover_depth_from_circle(self, radius_px):
        """
        已知球直径，结合相机内参恢复深度

        公式: Z = (f * D) / (2 * r)
        其中:
            Z: 深度(mm)
            f: 焦距(像素)
            D: 球直径(mm)
            r: 图像半径(像素)

        Args:
            radius_px: 图像中球的半径(像素)

        Returns:
            depth_mm: 深度(毫米)
        """
        # 使用fx和fy的平均值作为焦距
        f = (self.fx + self.fy) / 2.0

        # 计算深度
        depth_mm = (f * self.ball_diameter_mm) / (2.0 * radius_px)

        return depth_mm

    def backproject_to_3d(self, cx_px, cy_px, depth_mm):
        """
        将去畸变后的图像坐标反投影到3D坐标。
        使用去畸变后的内参直接计算，不经过 RealSense 的畸变模型。
        返回 camera 坐标系下的位置。
        """
        Z = depth_mm
        X = (cx_px - self.cx) * Z / self.fx
        Y = (cy_px - self.cy) * Z / self.fy
        return (X, Y, Z)

    def _transform_to_base_link(self, position_camera):
        """
        将 camera 坐标系下的位置转换到 base_link 坐标系。
        T_base_camera 平移单位是 m，detector 位置单位是 mm，
        因此需要 mm -> m -> 变换 -> m -> mm。

        Args:
            position_camera: [x, y, z] in camera frame (mm)

        Returns:
            [x, y, z] in base_link frame (mm)
        """
        if self.T_base_camera is None:
            return position_camera

        pos_camera_m = np.array(position_camera) / 1000.0
        pos_homo = np.append(pos_camera_m, 1.0)
        pos_base_homo = self.T_base_camera @ pos_homo
        return (pos_base_homo[:3] * 1000.0).tolist()

    def _detect_motion_blur(self, contour, fitted_radius):
        """
        判断当前 mask 是否为严重拖影。

        Returns:
            dict: {
                'is_blurred': bool,
                'aspect': float,
                'circularity': float,
                'radius_ratio': float,
                'bbox': (x, y, w, h),
            }
        """
        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(min(w, h), 1)

        contour_area = cv2.contourArea(contour)
        circle_area = np.pi * fitted_radius * fitted_radius
        circularity = contour_area / circle_area if circle_area > 0 else 0.0

        radius_ratio = 1.0
        if len(self.raw_detection_history) >= 2:
            historical_radii = [d[2] for d in self.raw_detection_history]
            r_median = float(np.median(historical_radii))
            if r_median > 1.0:
                radius_ratio = fitted_radius / r_median

        is_blurred = (
            (aspect > self.motion_blur_aspect_threshold and
             circularity < self.motion_blur_circularity_threshold)
            or radius_ratio > self.motion_blur_radius_ratio_threshold
        )

        return {
            'is_blurred': is_blurred,
            'aspect': aspect,
            'circularity': circularity,
            'radius_ratio': radius_ratio,
            'bbox': (x, y, w, h),
        }

    def _estimate_motion_direction(self):
        """
        从最近原始检测历史估计单位运动方向和平均位移。

        Returns:
            (unit_v, mean_step_px) 或 (None, 0.0)
        """
        if len(self.raw_detection_history) < 2:
            return None, 0.0

        steps = []
        vectors = []
        history = list(self.raw_detection_history)
        for i in range(len(history) - 1):
            cx0, cy0, _ = history[i]
            cx1, cy1, _ = history[i + 1]
            dx, dy = cx1 - cx0, cy1 - cy0
            step = (dx * dx + dy * dy) ** 0.5
            steps.append(step)
            vectors.append((dx, dy))

        if not steps:
            return None, 0.0

        mean_step = float(np.mean(steps))
        sum_dx = float(np.sum([v[0] for v in vectors]))
        sum_dy = float(np.sum([v[1] for v in vectors]))
        mag = (sum_dx * sum_dx + sum_dy * sum_dy) ** 0.5
        if mag < 1e-3:
            return None, mean_step
        unit_v = (sum_dx / mag, sum_dy / mag)
        return unit_v, mean_step

    def _correct_motion_blur(self, contour, blur_info):
        """
        在严重拖影时修正圆心/半径估计。

        策略 1：用运动方向找拖影前端，回推球心
        策略 2：运动方向不可靠时，用长轴两端候选并打分

        Returns:
            dict: {
                'cx': float, 'cy': float, 'radius': float,
                'method': str,  # 'motion_direction' / 'axis_endpoints' / None
            } 或 None（修正失败）
        """
        if len(self.raw_detection_history) < 2:
            return None

        historical_radii = [d[2] for d in self.raw_detection_history]
        r_guess = float(np.median(historical_radii))
        if r_guess < self.min_radius:
            return None

        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            return None

        unit_v, mean_step = self._estimate_motion_direction()

        # 策略 1：运动方向可靠
        if unit_v is not None and mean_step >= self.min_motion_speed_px:
            projections = pts[:, 0] * unit_v[0] + pts[:, 1] * unit_v[1]
            front_idx = int(np.argmax(projections))
            front_tip = pts[front_idx]
            cx = float(front_tip[0] - unit_v[0] * r_guess)
            cy = float(front_tip[1] - unit_v[1] * r_guess)
            return {
                'cx': cx,
                'cy': cy,
                'radius': r_guess,
                'method': 'motion_direction',
            }

        # 策略 2：用 minAreaRect 长轴两端做候选，选最像球的
        rect = cv2.minAreaRect(contour)
        (rcx, rcy), (rw, rh), angle = rect
        if max(rw, rh) < 1.0:
            return None
        if rw >= rh:
            half_long = rw / 2.0
            theta = np.radians(angle)
        else:
            half_long = rh / 2.0
            theta = np.radians(angle + 90.0)
        ax = float(np.cos(theta))
        ay = float(np.sin(theta))

        end1 = (rcx + ax * half_long, rcy + ay * half_long)
        end2 = (rcx - ax * half_long, rcy - ay * half_long)

        candidates = []
        for end in (end1, end2):
            tip_x, tip_y = end
            cx = float(tip_x - ax * r_guess) if (tip_x - rcx) * ax + \
                (tip_y - rcy) * ay > 0 else float(tip_x + ax * r_guess)
            cy = float(tip_y - ay * r_guess) if (tip_x - rcx) * ax + \
                (tip_y - rcy) * ay > 0 else float(tip_y + ay * r_guess)
            # 简单评分：候选球心距 contour 形心越近越好（保守，倾向稳定）
            score = -((cx - rcx) ** 2 + (cy - rcy) ** 2) ** 0.5
            candidates.append((score, cx, cy))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        _, cx, cy = candidates[0]
        return {
            'cx': cx,
            'cy': cy,
            'radius': r_guess,
            'method': 'axis_endpoints',
        }

    def process_frame(self, color_image):
        """
        处理单帧图像，返回球的位置信息（原始测量，未滤波）

        Args:
            color_image: BGR格式图像

        Returns:
            dict: 包含检测结果的字典，未检测到时返回None
        """
        # 1. YOLO分割检测
        mask = self.detect_ball_with_yolo(color_image)
        detection_method = 'yolo'

        # 1.5 如果 YOLO 失败，尝试颜色 fallback
        if mask is None:
            roi = None
            if self.last_detection_pixel is not None:
                cx, cy = self.last_detection_pixel
                x1 = max(0, int(cx - self.roi_margin))
                y1 = max(0, int(cy - self.roi_margin))
                x2 = min(self.image_width, int(cx + self.roi_margin))
                y2 = min(self.image_height, int(cy + self.roi_margin))
                roi = (x1, y1, x2 - x1, y2 - y1)

            mask = self.detect_ball_by_color(color_image, roi=roi)
            if mask is not None:
                detection_method = 'color_fallback'
            else:
                return None

        # 2. 颜色验证（确保是桔黄色的球）
        is_color_match, color_ratio = self.verify_color(color_image, mask)
        if not is_color_match:
            return {
                'error': 'color_mismatch',
                'color_ratio': color_ratio,
                'mask': mask,
            }

        # 3. 在 YOLO mask 内进一步找桔黄色区域，并优先用它拟合圆
        refined_mask, color_bbox = self.extract_colored_ball_region(
            color_image, mask)
        circle_mask = refined_mask if refined_mask is not None else mask

        # 4. 拟合圆（初步）
        circle = self.fit_circle(circle_mask)
        if circle is None:
            return None

        raw_cx_px, raw_cy_px, raw_radius_px = circle

        # 4.3 拖影检测与修正
        contours, _ = cv2.findContours(
            circle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(
            contours, key=cv2.contourArea) if contours else None

        motion_blur_detected = False
        motion_blur_corrected = False
        blur_aspect = 1.0
        blur_circularity = 1.0
        blur_radius_ratio = 1.0
        correction_method = None

        if largest_contour is not None:
            blur_info = self._detect_motion_blur(
                largest_contour, raw_radius_px)
            motion_blur_detected = blur_info['is_blurred']
            blur_aspect = blur_info['aspect']
            blur_circularity = blur_info['circularity']
            blur_radius_ratio = blur_info['radius_ratio']

            if motion_blur_detected:
                correction = self._correct_motion_blur(
                    largest_contour, blur_info)
                if correction is not None:
                    raw_cx_px = correction['cx']
                    raw_cy_px = correction['cy']
                    raw_radius_px = correction['radius']
                    motion_blur_corrected = True
                    correction_method = correction['method']

        # 更新原始检测历史（用于下一帧拖影判定）
        self.raw_detection_history.append(
            (raw_cx_px, raw_cy_px, raw_radius_px))

        # 4.5 One Euro Filter 源头平滑 + 异常值弱更新（仅用于 3D 计算）
        now = time.time()
        cx_smooth = raw_cx_px
        cy_smooth = raw_cy_px
        radius_smooth = raw_radius_px

        if self.oef_cx.x_prev is not None:
            cx_jump = abs(raw_cx_px - self.oef_cx.x_prev)
            cy_jump = abs(raw_cy_px - self.oef_cy.x_prev)
            center_jump = (cx_jump**2 + cy_jump**2) ** 0.5
            radius_change = abs(
                raw_radius_px - self.oef_radius.x_prev) / max(self.oef_radius.x_prev, 1.0)

            if center_jump > self.max_center_jump_px:
                cx_smooth = 0.15 * raw_cx_px + 0.85 * self.oef_cx.x_prev
                cy_smooth = 0.15 * raw_cy_px + 0.85 * self.oef_cy.x_prev

            if radius_change > self.max_radius_change_ratio:
                radius_smooth = 0.10 * raw_radius_px + 0.90 * self.oef_radius.x_prev

        cx_smooth = self.oef_cx(now, cx_smooth)
        cy_smooth = self.oef_cy(now, cy_smooth)
        radius_smooth = self.oef_radius(now, radius_smooth)

        # 5. 使用平滑后的半径计算深度（用于 3D 位置）
        depth_mm = self.recover_depth_from_circle(radius_smooth)
        depth_method = 'geometry'

        # 6. 反投影到3D（使用平滑后的值）
        X, Y, Z = self.backproject_to_3d(cx_smooth, cy_smooth, depth_mm)

        # 记录像素位置用于下一帧 ROI（使用平滑后的值）
        self.last_detection_pixel = (cx_smooth, cy_smooth)

        return {
            'image_x': raw_cx_px,
            'image_y': raw_cy_px,
            'radius_px': raw_radius_px,
            'smoothed_cx': cx_smooth,
            'smoothed_cy': cy_smooth,
            'smoothed_radius': radius_smooth,
            'depth_mm': depth_mm,
            'depth_method': depth_method,
            'detection_method': detection_method,
            'X_mm': X,
            'Y_mm': Y,
            'Z_mm': Z,
            'mask': mask,
            'refined_mask': refined_mask,
            'color_bbox': color_bbox,
            'color_ratio': color_ratio,
            'motion_blur_detected': motion_blur_detected,
            'motion_blur_corrected': motion_blur_corrected,
            'blur_aspect': blur_aspect,
            'blur_circularity': blur_circularity,
            'radius_ratio': blur_radius_ratio,
            'correction_method': correction_method,
        }

    def _smooth_position(self, raw_position):
        """
        简单 EMA 位置平滑器，不做运动预测。

        Args:
            raw_position: [x, y, z] 原始测量位置

        Returns:
            smoothed_position: [x, y, z] 平滑后位置
        """
        if self.smoothed_position is None:
            self.smoothed_position = np.array(raw_position, dtype=float)
        else:
            alpha = self.position_smooth_alpha
            self.smoothed_position = alpha * \
                np.array(raw_position) + (1.0 - alpha) * self.smoothed_position

        return self.smoothed_position.copy()

    def _calculate_velocity_from_clean_history(self):
        """
        从干净的位置历史计算速度。
        只使用可信的、已平滑的位置点。
        """
        if len(self.position_history) < 2:
            return [0, 0, 0]

        history = [
            (float(t), np.array(p, dtype=float))
            for t, p in self.position_history
        ]
        t0 = history[0][0]
        times = np.array([t - t0 for t, _ in history], dtype=float)
        positions = np.array([p for _, p in history], dtype=float)

        if np.ptp(times) < 1e-4:
            return [0, 0, 0]

        # Least-squares slope over the short clean history is less jittery than
        # averaging adjacent finite differences.
        velocities = []
        for axis in range(3):
            slope, _ = np.polyfit(times, positions[:, axis], 1)
            velocities.append(float(slope))
        return velocities

    def get_observation(self, color_image, observation_time=None):
        """
        获取球的观测值（仅平滑，不做运动预测）
        检测失败时直接返回 detected=False，不外推位置

        Args:
            color_image: BGR格式图像（原始图像，可能有畸变）

        Returns:
            dict: {
                'position': [x, y, z],  # 位置(mm)，在 output_frame 坐标系下
                'velocity': [vx, vy, vz],  # 速度(mm/s)，在 output_frame 坐标系下
                'detected': bool,  # 是否检测到球
                'raw_measurement': dict,  # 原始测量值
            }
        """
        now = time.time() if observation_time is None else float(observation_time)
        dt = None
        if self.last_observation_time is not None:
            dt = max(now - self.last_observation_time, 1e-3)
        self.last_observation_time = now

        # 去畸变处理（ROS 路径必须在这里执行）
        if self.use_undistort and self.map1 is not None and self.map2 is not None:
            color_image = cv2.remap(
                color_image, self.map1, self.map2, cv2.INTER_LINEAR)

        # 获取原始测量
        measurement = self.process_frame(color_image)

        # 检测失败：直接返回 False，不做任何预测或补帧
        if measurement is None or 'error' in measurement:
            self.missed_frames += 1
            # 连续漏检超过阈值，清理历史和平滑器
            if self.missed_frames > self.max_prediction_frames:
                self.position_history.clear()
                self.smoothed_position = None
                self.oef_cx.reset()
                self.oef_cy.reset()
                self.oef_radius.reset()
                if self.use_kalman and self.kalman is not None:
                    self.kalman.reset()
            return {
                'position': [
                    0,
                    0,
                    0],
                'velocity': [
                    0,
                    0,
                    0],
                'detected': False,
                'raw_measurement': measurement,
                'reject_reason': 'no_detection' if measurement is None else measurement.get(
                    'error',
                    'unknown'),
            }

        # 提取原始位置（camera 坐标系，已经过源头 One Euro Filter 平滑）
        pos_camera = [
            measurement['X_mm'],
            measurement['Y_mm'],
            measurement['Z_mm']]

        # 转换到 base_link 坐标系（如果需要）
        if self.output_frame == 'base_link' and self.T_base_camera is not None:
            pos_raw = self._transform_to_base_link(pos_camera)
        else:
            pos_raw = pos_camera

        self.missed_frames = 0

        # 异常值检测：3D 位置突变检查
        reject_reason = None
        if self.use_kalman and self.kalman is not None and self.kalman.initialized:
            innovation, _, _ = self.kalman.innovation(pos_raw, dt=dt)
            innovation_norm = float(np.linalg.norm(innovation))
            depth_innovation = float(abs(innovation[2]))

            if (
                innovation_norm > self.max_measurement_jump_mm
                or depth_innovation > self.max_depth_jump_mm
            ):
                reject_reason = (
                    'innovation_too_large: '
                    f'norm={innovation_norm:.1f}mm, '
                    f'depth={depth_innovation:.1f}mm'
                )

        # 观测异常处理：不 reset 滤波器，降权处理
        if reject_reason is not None:
            if self.use_kalman and self.kalman is not None:
                self.kalman.update(pos_raw, dt=dt, measurement_noise_mm=200.0)
            return {
                'position': pos_raw,
                'velocity': self._calculate_velocity_from_clean_history(),
                'detected': True,
                'raw_measurement': {
                    **measurement,
                    'filter_rejected': True,
                    'reject_reason': reject_reason,
                },
                'smoothed': False,
            }

        # 观测可信：平滑位置
        if self.use_kalman and self.kalman is not None:
            # 使用卡尔曼仅做平滑（不做预测）
            state = self.kalman.update(
                pos_raw, dt=dt, measurement_noise_mm=30.0)
            pos_smoothed = state[:3].tolist()
        else:
            # 使用简单 EMA 平滑
            pos_smoothed = self._smooth_position(pos_raw).tolist()

        # 更新干净的位置历史
        self.position_history.append((now, pos_smoothed))

        # 从干净历史计算速度
        velocity = self._calculate_velocity_from_clean_history()

        return {
            'position': pos_smoothed,
            'velocity': velocity,
            'detected': True,
            'raw_measurement': {
                **measurement,
                'filter_rejected': False,
                'raw_position': pos_raw,
            },
            'smoothed': True,
        }

    def run(self):
        """主循环（无窗口，Ctrl+C 退出）"""
        print("开始检测，Ctrl+C 退出")

        try:
            while True:
                t0 = time.time()

                # 获取帧
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())

                # 验证图像数据
                if color_image is None or color_image.size == 0:
                    print("警告: 获取到空图像")
                    continue

                if len(color_image.shape) != 3:
                    print(f"警告: 图像维度错误 {color_image.shape}")
                    continue

                t_capture = time.time()

                # 去畸变已统一在 get_observation() 中处理，这里不再重复
                # 获取观测值（内部会自动去畸变）
                observation = self.get_observation(color_image)
                t_observe = time.time()
                self.total_frames += 1
                if observation['detected']:
                    self.detected_frames += 1
                else:
                    self.missed_detection_frames += 1

                # 实时打印位置信息
                should_print = (self.total_frames % self.print_every == 0)
                if observation['detected'] and should_print:
                    pos = observation['position']
                    vel = observation['velocity']
                    raw = observation.get('raw_measurement') or {}
                    smoothed = observation.get('smoothed', False)
                    rejected = raw.get('filter_rejected', False)
                    raw_pos = raw.get('raw_position', pos)

                    status = 'SMOOTH' if smoothed else (
                        'REJECT' if rejected else 'RAW')

                    blur_flag = 'B' if raw.get(
                        'motion_blur_detected', False) else ' '
                    corr_flag = 'C' if raw.get(
                        'motion_blur_corrected', False) else ' '

                    print(
                        f"像素: ({raw.get('image_x', 0):7.1f}, "
                        f"{raw.get('image_y', 0):7.1f}) px  "
                        f"半径: {raw.get('radius_px', 0):6.2f}→"
                        f"{raw.get('smoothed_radius', 0):6.2f} px  "
                        f"深度: {raw.get('depth_mm', 0):7.1f} mm "
                        f"[{raw.get('detection_method', '?')}]  "
                        f"blur: asp={raw.get('blur_aspect', 1.0):.2f} "
                        f"circ={raw.get('blur_circularity', 1.0):.2f} "
                        f"rr={raw.get('radius_ratio', 1.0):.2f} "
                        f"[{blur_flag}{corr_flag}]  "
                        f"原始: ({raw_pos[0]:7.1f}, {raw_pos[1]:7.1f}, "
                        f"{raw_pos[2]:7.1f})  "
                        f"输出: ({pos[0]:7.1f}, {pos[1]:7.1f}, "
                        f"{pos[2]:7.1f}) mm  "
                        f"速度: ({vel[0]:6.0f}, {vel[1]:6.0f}, "
                        f"{vel[2]:6.0f}) mm/s  "
                        f"[{status}]"
                    )
                    if raw.get('motion_blur_corrected'):
                        print(
                            f"  ↳ blur correction: {raw.get('correction_method', '?')}")
                    if rejected:
                        print(
                            f"  ↳ filter reject: {raw.get('reject_reason', '?')}")
                elif should_print:
                    reason = observation.get('reject_reason', 'no_detection')
                    print(f"未检测到 [{reason}]")
                t_print = time.time()

                # 计算FPS
                t_end = time.time()
                dt = t_end - t0
                self.fps_queue.append(1.0 / dt if dt > 0 else 0)

                if self.profile_every > 0:
                    self.profile_samples.append({
                        'capture': t_capture - t0,
                        'observe': t_observe - t_capture,
                        'yolo': self.last_yolo_ms / 1000.0,
                        'print': t_print - t_observe,
                        'total': t_end - t0,
                    })
                    if len(self.profile_samples) >= self.profile_every:
                        avg = {
                            key: np.mean([sample[key] for sample in self.profile_samples]) * 1000.0
                            for key in self.profile_samples[0]
                        }
                        hz = 1000.0 / avg['total'] if avg['total'] > 0 else 0.0
                        print(
                            "PROFILE "
                            f"total={avg['total']:.1f}ms/{hz:.1f}Hz "
                            f"capture={avg['capture']:.1f} "
                            f"observe={avg['observe']:.1f} "
                            f"yolo={avg['yolo']:.1f} "
                            f"print={avg['print']:.1f}"
                        )
                        self.profile_samples.clear()

        except KeyboardInterrupt:
            print("\n检测已停止")
        finally:
            if self.total_frames > 0:
                detection_rate = 100.0 * self.detected_frames / self.total_frames
                miss_rate = 100.0 * self.missed_detection_frames / self.total_frames
                print("\n" + "=" * 60)
                print("检测统计")
                print("=" * 60)
                print(f"总帧数: {self.total_frames}")
                print(f"检测成功帧数: {self.detected_frames}")
                print(f"未检测到帧数: {self.missed_detection_frames}")
                print(f"检测成功率: {detection_rate:.2f}%")
                print(f"漏检率: {miss_rate:.2f}%")
                print("=" * 60)

            if self.pipeline is not None:
                self.pipeline.stop()


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import traceback

    module_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_candidates = [
        os.path.join(module_dir, 'models', 'best.pt'),
        os.path.join(
            module_dir,
            'runs', 'segment', 'train-2',
            'weights', 'best.pt',
        ),
    ]
    DEFAULT_MODEL = next(
        (p for p in default_model_candidates
         if os.path.exists(p)),
        default_model_candidates[0],
    )

    parser = argparse.ArgumentParser(
        description='PingPong ball detector perf test',
    )
    parser.add_argument(
        '--model', default=DEFAULT_MODEL,
        help='YOLO model path',
    )
    parser.add_argument(
        '--class', dest='target_class',
        default='pingpong ball',
        help='target class name',
    )
    kalman_grp = parser.add_mutually_exclusive_group()
    kalman_grp.add_argument(
        '--kalman', dest='use_kalman',
        action='store_true', default=False,
    )
    kalman_grp.add_argument(
        '--no-kalman', dest='use_kalman',
        action='store_false',
    )
    parser.add_argument(
        '--tilt', type=float, default=40.0,
        help='camera tilt degrees (default 40)',
    )
    parser.add_argument(
        '--imgsz', type=int, default=640,
        help='YOLO inference image size',
    )
    parser.add_argument(
        '--device', default='0',
        help='YOLO device (e.g. 0, cpu)',
    )
    parser.add_argument(
        '--half', action='store_true',
        help='enable FP16 inference',
    )
    parser.add_argument(
        '--no-undistort', dest='use_undistort',
        action='store_false', default=True,
    )
    parser.add_argument(
        '--print-every', type=int, default=1,
    )
    parser.add_argument(
        '--profile-every', type=int, default=0,
    )
    parser.add_argument(
        '--width', type=int, default=1280,
    )
    parser.add_argument(
        '--height', type=int, default=720,
    )
    parser.add_argument(
        '--fps', type=int, default=30,
    )
    parser.add_argument(
        '--conf', type=float, default=0.5,
        help='YOLO confidence threshold',
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        sys.exit(1)

    print("=" * 50)
    print("PingPong Detector 启动配置")
    print("=" * 50)
    print(f"  模型路径:   {args.model}")
    print(f"  分辨率:     {args.width}x{args.height}")
    print(f"  FPS:        {args.fps}")
    print(f"  目标类别:   {args.target_class}")
    print(f"  推理尺寸:   {args.imgsz}")
    print(f"  推理设备:   {args.device}")
    print(f"  半精度:     {args.half}")
    print(f"  卡尔曼:     {args.use_kalman}")
    print(f"  去畸变:     {args.use_undistort}")
    print(f"  置信度:     {args.conf}")
    print("=" * 50)

    try:
        detector = PingPongBallDetector(
            yolo_model_path=args.model,
            ball_diameter_mm=40.0,
            width=args.width,
            height=args.height,
            fps=args.fps,
            conf_threshold=args.conf,
            target_class=args.target_class,
            camera_tilt_deg=args.tilt,
            use_kalman=args.use_kalman,
            infer_imgsz=args.imgsz,
            device=args.device,
            half=args.half,
            use_undistort=args.use_undistort,
            print_every=args.print_every,
            profile_every=args.profile_every,
        )
        detector.run()
    except KeyboardInterrupt:
        print("\n已退出")
    except Exception as exc:
        print(f"错误: {exc}")
        traceback.print_exc()
        sys.exit(1)
