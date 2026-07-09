#!/home/jacky/miniconda3/envs/gmr/bin/python
"""
HybridBallDetector Adapter for pingpong_node

将HybridBallDetector的输出适配为与PingPongBallDetector兼容的格式，
以便无缝集成到pingpong_node的后处理流程中。

主要区别：
- 灰色反光球，不依赖颜色筛选
- RGB YOLO + IR检测的混合架构
- geometry/sensor深度自动切换
"""

import numpy as np
import json
import os
from typing import Optional, Dict, Any
from pingpong_controller.hybrid_ball_detector import HybridBallDetector
from pingpong_controller.calibration_loader import CalibrationData


class HybridBallDetectorAdapter:
    """
    适配器：让HybridBallDetector输出与PingPongBallDetector兼容的格式

    主要职责：
    1. 封装HybridBallDetector
    2. 提供与PingPongBallDetector相同的detect()接口
    3. 输出格式兼容后续的BallStateEstimator
    """

    def __init__(self,
                 yolo_model_path: str,
                 ball_diameter_mm: float = 40.0,
                 width: int = 848,
                 height: int = 480,
                 fps: int = 60,
                 # Hybrid detector特有参数
                 yolo_interval: int = 3,
                 yolo_conf: float = 0.25,
                 target_class: str = 'Class 0',
                 ir_params_json: Optional[str] = None,
                 ir_threshold: int = 200,
                 use_adaptive: bool = True,
                 dilate_iterations: int = 5,
                 close_kernel_size: int = 7,
                 denoise_kernel_size: int = 3,
                 min_radius: int = 5,
                 max_radius: int = 50,
                 circularity_min: float = 0.75,
                 laser_power: int = 150,
                 exposure: Optional[int] = None,
                 gain: Optional[int] = None,
                 brightness_percentile: int = 30,
                 roi_margin: int = 80,
                 # 通用参数
                 calibration_data: Optional[CalibrationData] = None,
                 output_frame: str = 'base_link',
                 print_every: int = 10,
                 # 深度校正模型路径（新增）
                 depth_correction_model_path: str = 'depth_correction_model.pkl',
                 # RealSense参数（兼容pingpong_node）
                 use_camera: bool = False,
                 use_undistort: bool = True,
                 depth_scale: float = 0.001,
                 depth_neighborhood_px: int = 2,
                 # 以下参数为兼容接口，hybrid_detector不使用
                 use_kalman: bool = False,
                 camera_tilt_deg: float = 0.0,
                 conf_threshold: float = 0.25,
                 infer_imgsz: int = 640,
                 device: str = '0',
                 half: bool = False,
                 enable_ellipse_fit: bool = True,
                 enable_temporal_tracker: bool = False,
                 tracker_max_miss_bridge: int = 2,
                 tracker_gate_px: float = 120.0,
                 hsv_lower=None,
                 hsv_upper=None,
                 color_match_threshold: float = 0.0,  # 灰色球不用颜色筛选
                 color_circularity_min: float = 0.0,
                 use_depth_sensor: bool = True,
                 profile_every: int = 0,
                 camera_fx: Optional[float] = None,
                 camera_fy: Optional[float] = None,
                 camera_cx: Optional[float] = None,
                 camera_cy: Optional[float] = None,
                 camera_dist_coeffs: Optional[list] = None):
        """
        初始化适配器

        Args:
            yolo_model_path: YOLO模型路径
            yolo_interval: YOLO检测间隔（帧数）
            yolo_conf: YOLO置信度阈值
            ir_params_json: IR参数JSON文件路径（如果提供，覆盖其他IR参数）
            ir_threshold: IR二值化阈值
            use_adaptive: 是否使用自适应阈值
            dilate_iterations: IR二值图膨胀迭代次数
            close_kernel_size: IR二值图闭运算核大小
            denoise_kernel_size: IR二值图去噪核大小
            exposure: IR曝光时间（μs），None=自动
            gain: IR增益，None=自动
            calibration_data: 手眼标定数据（用于转换到base_link）
            output_frame: 输出坐标系 ('camera' 或 'base_link')
            color_match_threshold: 颜色匹配阈值（灰色球设为0禁用）
            其他参数为兼容接口
        """
        self.output_frame = output_frame
        self.calibration_data = calibration_data
        self.print_every = print_every
        self.frame_count = 0

        # 兼容属性（pingpong_node需要）
        self.use_undistort = use_undistort
        self.use_depth_sensor = use_depth_sensor
        self.depth_scale = depth_scale
        self.depth_neighborhood_px = depth_neighborhood_px

        # 如果提供了ir_params_json，从文件加载参数
        if ir_params_json and os.path.exists(ir_params_json):
            with open(ir_params_json, 'r') as f:
                ir_params = json.load(f)
            # 覆盖默认参数
            laser_power = ir_params.get('laser_power', laser_power)
            exposure = ir_params.get('exposure', exposure)
            gain = ir_params.get('gain', gain)
            ir_threshold = ir_params.get('ir_threshold', ir_threshold)
            use_adaptive = ir_params.get('use_adaptive', use_adaptive)
            dilate_iterations = ir_params.get('dilate_iterations', dilate_iterations)
            close_kernel_size = ir_params.get('close_kernel_size', close_kernel_size)
            denoise_kernel_size = ir_params.get('denoise_kernel_size', denoise_kernel_size)
            min_radius = ir_params.get('min_radius', min_radius)
            max_radius = ir_params.get('max_radius', max_radius)
            circularity_min = ir_params.get('circularity_min', circularity_min)
            brightness_percentile = ir_params.get('brightness_percentile', brightness_percentile)
            print(f"从 {ir_params_json} 加载IR参数")

        # 初始化HybridBallDetector
        self.detector = HybridBallDetector(
            yolo_model_path=yolo_model_path,
            ball_diameter_mm=ball_diameter_mm,
            width=width,
            height=height,
            fps=fps,
            yolo_interval=yolo_interval,
            yolo_conf=yolo_conf,
            target_class=target_class,
            ir_threshold=ir_threshold,
            use_adaptive=use_adaptive,
            dilate_iterations=dilate_iterations,
            close_kernel_size=close_kernel_size,
            denoise_kernel_size=denoise_kernel_size,
            min_radius=min_radius,
            max_radius=max_radius,
            circularity_min=circularity_min,
            laser_power=laser_power,
            exposure=exposure,
            gain=gain,
            brightness_percentile=brightness_percentile,
            roi_margin=roi_margin,
            depth_scale=depth_scale,
            depth_neighborhood_px=depth_neighborhood_px,
            calibration_data=calibration_data,
            output_frame=output_frame,
            print_every=print_every,
            depth_correction_model_path=depth_correction_model_path  # 新增
        )

    def detect(self, rgb_image, depth_image=None, timestamp=None):
        """
        检测球的位置（兼容PingPongBallDetector接口）

        Args:
            rgb_image: BGR格式的RGB图像
            depth_image: 深度图（可选，hybrid_detector从RealSense直接读取）
            timestamp: 时间戳（可选）

        Returns:
            dict: 检测结果，格式与PingPongBallDetector兼容
                {
                    'detected': bool,
                    'position': np.ndarray,  # [x, y, z] in mm
                    'velocity': np.ndarray,  # [vx, vy, vz] in mm/s
                    'pixel': tuple,          # (cx, cy)
                    'radius': float,         # 像素半径
                    'depth': float,          # 深度 (mm)
                    'depth_method': str,     # 'geometry' or 'sensor'
                    'yolo_bbox': tuple,      # YOLO检测框
                    'detection_mode': str,   # 'hybrid' or 'fallback'
                }
        """
        self.frame_count += 1

        # 调用HybridBallDetector
        result = self.detector.detect(rgb_image, depth_image, timestamp)

        if result is None or not result.get('detected', False):
            return {
                'detected': False,
                'position': None,
                'velocity': None,
                'pixel': None,
                'radius': None,
                'depth': None,
                'depth_method': 'none',
                'yolo_bbox': None,
                'detection_mode': 'none',
            }

        # 转换为兼容格式
        # HybridBallDetector已经处理了坐标系转换（camera或base_link）
        position_mm = result['position']  # 已经是mm单位，已经在目标坐标系

        # 速度：HybridBallDetector有简单的速度估计（差分）
        velocity_mm_s = result.get('velocity', np.zeros(3))

        return {
            'detected': True,
            'position': position_mm,  # mm
            'velocity': velocity_mm_s,  # mm/s
            'pixel': (result['pixel_x'], result['pixel_y']),
            'radius': result.get('radius_px', 0.0),
            'depth': result['depth_mm'],
            'depth_method': result.get('depth_method', 'unknown'),
            'yolo_bbox': result.get('yolo_bbox'),
            'detection_mode': result.get('detection_mode', 'unknown'),
            # 额外信息
            'ir_brightness': result.get('ir_brightness'),
            'timestamp': timestamp,
        }

    def close(self):
        """关闭检测器"""
        if hasattr(self.detector, 'close'):
            self.detector.close()

    def get_stats(self):
        """获取统计信息（兼容接口）"""
        if hasattr(self.detector, 'get_stats'):
            return self.detector.get_stats()
        return {}
