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
from collections import deque
from typing import Optional, Dict, Any, Tuple
from ultralytics import YOLO

from pingpong_controller.calibration_loader import CalibrationData
from pingpong_controller.pixel_kalman_filter import PixelKalmanFilter


class HybridBallDetector:
    """
    RGB+IR混合检测器

    工作流程：
    1. RGB YOLO检测 → 得到球的bbox（粗定位，过滤语义）
    2. IR在bbox内检测 → 精确定位和深度（高精度，高帧率）
    3. 如果YOLO失败 → 回退到全图IR检测（容错）
    """

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
        self.print_every = print_every
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

        # 初始化RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 配置RGB和IR流
        # 修复：RGB也需要60fps以支持高速运动下的YOLO检测
        rgb_fps = 60  # RGB恢复60fps
        ir_fps = 60   # IR保持60fps（主要数据源）

        try:
            # RGB流（60fps以支持高速运动下的YOLO）
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, rgb_fps)
            # IR流（用于精确检测，保持高帧率）
            config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, ir_fps)
            # 深度流（与IR同步）
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, ir_fps)
            print(f"配置: RGB {width}x{height}@{rgb_fps}fps, IR/Depth {width}x{height}@{ir_fps}fps")
        except Exception as e:
            # 回退配置
            print(f"配置失败: {e}, 尝试回退配置...")
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
            width, height = 640, 480
            rgb_fps, ir_fps = 60, 60
            print(f"回退配置: RGB {width}x{height}@{rgb_fps}fps, IR/Depth {width}x{height}@{ir_fps}fps")

        self.image_width = width
        self.image_height = height
        self.fps = ir_fps  # 使用IR帧率作为主帧率

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

        # 设置激光功率
        device = profile.get_device()
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

        # 获取RGB内参（用于可视化和手眼标定坐标系）
        color_stream = profile.get_stream(rs.stream.color)
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
        self.depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)

        # 构建IR→RGB的4x4齐次变换矩阵（更清晰直接）
        # RealSense外参：rotation是3x3矩阵（行优先存储9个元素），translation是3D向量（米）
        R = np.array(self.depth_to_color_extrinsics.rotation).reshape(3, 3)
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

    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _init_runtime_state(self):
        """初始化在线/离线共享的运行时状态（计数、队列、深度门等）。"""
        # 状态
        self.frame_count = 0
        self.last_yolo_bbox = None  # 最近的YOLO bbox
        self.last_yolo_time = 0
        self.yolo_miss_count = 0
        self.max_yolo_miss = 3  # 修复：YOLO连续丢失3次后清除bbox（软约束）

        # 统计
        self.total_frames = 0
        self.detected_frames = 0
        self.yolo_detections = 0
        self.ir_detections = 0
        self.hybrid_detections = 0
        self.fallback_detections = 0

        self.fps_queue = deque(maxlen=30)
        self.yolo_time_queue = deque(maxlen=10)
        self.ir_time_queue = deque(maxlen=30)

        self.position_history = deque(maxlen=5)
        self.last_observation_time = None

        # 像素坐标历史（用于平滑可视化）
        self.pixel_history = deque(maxlen=5)  # (cx, cy, radius)

        # 2D像素卡尔曼滤波器（核心平滑机制）
        # 参数调优：静止球高稳定性，运动球低延迟
        self.pixel_kf = PixelKalmanFilter(
            dt=1.0/self.fps,  # 使用self.fps而不是fps
            process_noise_pos=0.3,    # 位置过程噪声（像素） - 降低以增强静止稳定性
            process_noise_vel=3.0,     # 速度过程噪声（像素/秒） - 适中以平衡响应
            measurement_noise=1.2,     # 测量噪声（像素） - IR检测的典型精度
            innovation_gate=15.0       # 离群值门限 - 较大值以容忍击打时的真实跳变
        )

        # 深度历史（用于时域去尖峰+平滑，统一作用于 sensor/geometry 两条路径）
        self.depth_history = deque(maxlen=5)

        # 深度源距离门（带迟滞）：
        #   近场 —— 双目常跌破 D455 最小工作距离而失效/空洞，且球半径大、
        #           单目几何 Z∝1/r 越近越稳 → 锁定 geometry。
        #   远场 —— 球半径小、几何 dr/r 噪声放大，双目更准 → 用 sensor 深度。
        # 两个门限之间留迟滞带，避免在临界点 sensor/geometry 反复横跳
        # （切换本身会造成阶跃跳变，这正是原逐帧回退的病根）。
        self.depth_near_gate_mm = 500.0   # 稳定深度 < 此值 → 切到近场(geometry)
        self.depth_far_gate_mm = 650.0    # 稳定深度 > 此值 → 切回远场(sensor)
        self.depth_source_state = 'far'   # 'near'=锁geometry, 'far'=用sensor

        # FPS计算（使用帧时间戳）
        self.last_frame_timestamp = None

        # 自适应阈值
        self.background_intensity = None

        # 深度校正模型（YOLO辅助校准）
        self.depth_correction_model = None
        # self.depth_correction_model_path 已在__init__开头设置
        self._load_depth_correction_model(self.depth_correction_model_path)  # 使用实例变量

    def _load_depth_correction_model(self, model_path='depth_correction_model.pkl'):
        """加载深度校正模型"""
        import os
        import pickle

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.depth_correction_model = pickle.load(f)
                print(f"✓ 加载深度校正模型: {model_path}")
                print(f"  模型类型: {self.depth_correction_model['type']}")
                print(f"  训练误差: {self.depth_correction_model['error']:.1f}mm")
                print(f"  训练样本: {self.depth_correction_model['n_samples']}个")
            except Exception as e:
                print(f"⚠️  加载深度校正模型失败: {e}")
                self.depth_correction_model = None
        else:
            print(f"  未找到深度校正模型: {model_path}")

    def _apply_depth_correction(self, depth_raw_mm):
        """应用深度校正（仅对geometry深度）"""
        if self.depth_correction_model is None or depth_raw_mm is None:
            return depth_raw_mm

        model = self.depth_correction_model
        if model['type'] == 'linear':
            # depth_corrected = scale * depth_raw + offset
            params = model['params']
            return depth_raw_mm * params['scale'] + params['offset']
        elif model['type'] == 'polynomial':
            # depth_corrected = a*depth^2 + b*depth + c
            params = model['params']
            return (params['a'] * depth_raw_mm**2 +
                   params['b'] * depth_raw_mm +
                   params['c'])
        else:
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
        self.fps = 60
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
        R = np.array(ext['rotation'], dtype=np.float64).reshape(3, 3)
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

    def _detect_in_ir(self, ir_image, roi=None):
        """
        在IR图像中检测球

        Args:
            ir_image: 红外图像
            roi: 可选的ROI (x1, y1, x2, y2)，None表示全图检测

        Returns:
            检测结果字典或None
        """
        # 阈值分割
        if self.ir_params['use_adaptive']:
            threshold = self._update_adaptive_threshold(ir_image, roi)
        else:
            threshold = self.ir_params['threshold']

        _, binary = cv2.threshold(ir_image, threshold, 255, cv2.THRESH_BINARY)

        # 如果有ROI，限制检测区域
        if roi is not None:
            x1, y1, x2, y2 = roi
            mask = np.zeros_like(binary)
            mask[y1:y2, x1:x2] = binary[y1:y2, x1:x2]
            binary = mask

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

            if len(cnt) >= 5:
                try:
                    # 椭圆拟合（已经是亚像素精度）
                    ellipse = cv2.fitEllipse(cnt)
                    (cx_ellipse, cy_ellipse), (minor, major), angle = ellipse

                    radius = minor / 2.0
                    if not (self.ir_params['min_radius'] < radius < self.ir_params['max_radius']):
                        continue

                    aspect_ratio = minor / (major + 1e-6)
                    if aspect_ratio < self.ir_params['circularity_min']:
                        continue

                    circle_area = np.pi * radius * radius
                    area_circularity = area / (circle_area + 1e-6)
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

                    mask = np.zeros(ir_image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
                    brightness = np.sum(ir_image[mask > 0])

                    candidates.append({
                        'cx': cx,
                        'cy': cy,
                        'radius': radius,
                        'aspect_ratio': aspect_ratio,
                        'brightness': brightness,
                        'mask': mask,
                    })
                except:
                    continue

        if not candidates:
            return None

        # 选择最亮的
        best = max(candidates, key=lambda x: x['brightness'])

        # 核心平滑：2D像素卡尔曼滤波
        # 这是最关键的一步，大幅减少静止球抖动
        smoothed_pos = self.pixel_kf.update([best['cx'], best['cy']])
        best['cx_raw'] = best['cx']  # 保存原始值用于调试
        best['cy_raw'] = best['cy']
        best['cx'] = smoothed_pos[0]  # 用平滑值替换
        best['cy'] = smoothed_pos[1]

        return best

    def _get_depth_from_depthmap(self, cx_px, cy_px, depth_image, ball_mask=None):
        """从深度图读取深度（增强稳定性，减少近距离跳变）"""
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        cx = int(round(cx_px))
        cy = int(round(cy_px))

        if not (0 <= cx < w and 0 <= cy < h):
            return None

        # 采样策略：优先使用mask，fallback到邻域
        depths = None
        if ball_mask is not None and ball_mask.shape[:2] == depth_image.shape[:2]:
            sel = (ball_mask > 0) & (depth_image > 0)
            if np.count_nonzero(sel) >= 5:
                depths = depth_image[sel]

        if depths is None:
            r = self.depth_neighborhood_px
            patch = depth_image[max(0, cy-r):min(h, cy+r+1),
                               max(0, cx-r):min(w, cx+r+1)]
            depths = patch[patch > 0]

        if depths.size < 3:
            return None

        # 使用中位数而非25分位数，更鲁棒
        depth_raw = float(np.median(depths)) * self.depth_scale * 1000.0

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
        f = (self.ir_fx + self.ir_fy) / 2.0
        return (f * self.ball_diameter_mm) / (2.0 * radius_px)

    def _select_depth(self, sensor_depth_mm, geom_depth_mm):
        """带迟滞的距离门，选择 sensor / geometry 作为本帧深度源。

        用上一帧的稳定深度做门控量（geometry 在近场几乎总有效，可兜底）：
          - 稳定深度 < near_gate → 状态切 'near'（锁 geometry）
          - 稳定深度 > far_gate  → 状态切 'far'（用 sensor）
          - 之间 → 保持上一状态（迟滞，避免临界反复横跳）
        sensor 无效时无条件退回 geometry，但不改变状态机。

        Returns:
            (depth_raw_mm, method_str) 或 (None, 'none')
        """
        # 门控参考量：优先用上一帧稳定深度，无历史时用几何深度
        ref = self.depth_history[-1] if self.depth_history else geom_depth_mm
        if ref is not None:
            if ref < self.depth_near_gate_mm:
                self.depth_source_state = 'near'
            elif ref > self.depth_far_gate_mm:
                self.depth_source_state = 'far'
            # 迟滞带内保持原状态

        if self.depth_source_state == 'near':
            if geom_depth_mm is not None:
                return geom_depth_mm, 'geometry'
            if sensor_depth_mm is not None:
                return sensor_depth_mm, 'sensor'
        else:  # 'far'
            if sensor_depth_mm is not None:
                return sensor_depth_mm, 'sensor'
            if geom_depth_mm is not None:
                return geom_depth_mm, 'geometry'
        return None, 'none'

    def _stabilize_depth(self, depth_raw):
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
        hist.append(depth_raw)
        if len(hist) >= 3:
            return float(np.median(list(hist)))
        return depth_raw

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

    def _calculate_velocity(self):
        """从历史计算速度"""
        if len(self.position_history) < 2:
            return [0, 0, 0]

        history = list(self.position_history)
        t0, p0 = history[0]
        t1, p1 = history[-1]
        dt = t1 - t0

        if dt < 1e-4:
            return [0, 0, 0]

        velocity = [(p1[i] - p0[i]) / dt for i in range(3)]
        return velocity

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
        color_frame = frames.get_color_frame()
        ir_frame = frames.get_infrared_frame(1)
        depth_frame = frames.get_depth_frame()

        if not color_frame or not ir_frame or not depth_frame:
            return {'detected': False, 'position': [0, 0, 0], 'velocity': [0, 0, 0]}

        # 修复：FPS计算使用IR帧时间戳（IR是主循环频率）
        current_timestamp = ir_frame.get_timestamp()  # 毫秒
        if self.last_frame_timestamp is not None:
            frame_interval_ms = current_timestamp - self.last_frame_timestamp
            if frame_interval_ms > 0:
                self.fps_queue.append(1000.0 / frame_interval_ms)
        self.last_frame_timestamp = current_timestamp

        rgb_image = np.asanyarray(color_frame.get_data())
        ir_image = np.asanyarray(ir_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return self._detect_core(rgb_image, ir_image, depth_image, now)

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
            return {'detected': False, 'position': [0, 0, 0],
                    'velocity': [0, 0, 0]}
        return self._detect_core(rgb_image, ir_image, depth_image, now)

    def _detect_core(self, rgb_image, ir_image, depth_image, now):
        """混合检测核心（在线/离线共用）：YOLO ROI → IR 定位 → 深度 → 3D。

        输入均为 numpy 数组；depth_image 允许为 None（无 sensor 深度时退回
        geometry）。帧计数/FPS 由调用方（get_observation/offline）负责维护。
        """
        detection_method = 'none'
        roi = None

        # 步骤1：YOLO检测（间隔运行，辅助语义过滤）
        if self.frame_count % self.yolo_interval == 0:
            t_yolo = time.time()
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
        if self.last_yolo_bbox is not None and self.yolo_miss_count < self.max_yolo_miss:
            # 使用YOLO bbox作为ROI
            roi = self._expand_bbox(self.last_yolo_bbox, self.roi_margin)
            use_roi = True
            detection_method = 'hybrid'
        else:
            # YOLO失效，清除bbox
            self.last_yolo_bbox = None
            detection_method = 'ir_fallback'

        # 步骤2：IR检测（优先在ROI内，失败立即全图兜底）
        t_ir = time.time()
        ir_detection = None

        if use_roi:
            # 优先在ROI内检测
            ir_detection = self._detect_in_ir(ir_image, roi)
            if ir_detection is None:
                # 修复：ROI内失败，立即全图兜底（YOLO软约束）
                ir_detection = self._detect_in_ir(ir_image, None)
                if ir_detection is not None:
                    detection_method = 'ir_fallback'  # 标记为兜底模式
        else:
            # 全图检测
            ir_detection = self._detect_in_ir(ir_image, None)

        self.ir_time_queue.append((time.time() - t_ir) * 1000)

        if ir_detection is None:
            return {'detected': False, 'position': [0, 0, 0], 'velocity': [0, 0, 0],
                   'detection_method': detection_method}

        # 步骤3：获取深度
        # 3a. sensor（双目硬件深度，远场准）与 geometry（单目 Z=f*D/2r，近场稳）都算出来
        sensor_depth_mm = self._get_depth_from_depthmap(
            ir_detection['cx'], ir_detection['cy'], depth_image, ir_detection['mask'])
        geom_depth_mm = self._geometry_depth_mm(ir_detection['radius'])

        # 3a+. 对geometry深度应用YOLO校准的校正模型
        geom_depth_mm = self._apply_depth_correction(geom_depth_mm)

        # 3b. 带迟滞的距离门选择深度源（近场锁geometry / 远场用sensor，避免逐帧横跳）
        depth_raw, depth_method = self._select_depth(sensor_depth_mm, geom_depth_mm)
        if depth_raw is None:
            return {'detected': False, 'position': [0, 0, 0], 'velocity': [0, 0, 0],
                   'detection_method': detection_method}

        # 3c. 时域去尖峰+平滑（统一作用于两条路径）
        depth_mm = self._stabilize_depth(depth_raw)

        # 步骤4：IR坐标 + IR内参 → IR 3D点
        X_ir, Y_ir, Z_ir = self._backproject_to_3d(ir_detection['cx'], ir_detection['cy'], depth_mm)
        pos_ir = [X_ir, Y_ir, Z_ir]

        # 步骤5：IR 3D → T_ir_to_rgb齐次变换 → RGB 3D（关键！）
        pos_rgb_camera = self._transform_ir_3d_to_rgb_3d(pos_ir)

        # 步骤6：转换坐标系（RGB手眼外参 → base_link）
        if self.output_frame == 'base_link' and self.T_base_camera is not None:
            position = self._transform_to_base_link(pos_rgb_camera)
        else:
            position = pos_rgb_camera

        # 更新历史
        self.position_history.append((now, position))

        # 计算速度
        velocity = self._calculate_velocity()

        # 统计
        self.detected_frames += 1
        self.ir_detections += 1
        if detection_method == 'hybrid':
            self.hybrid_detections += 1
        elif detection_method == 'ir_fallback':
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
                'cx_rgb': rgb_pixel[0] if rgb_pixel else None,  # IR在RGB图上的精确位置（卡尔曼后）
                'cy_rgb': rgb_pixel[1] if rgb_pixel else None,
                'radius': ir_detection['radius'],
                'smoothed_cx': smoothed_cx,  # 移动平均平滑（额外的平滑层）
                'smoothed_cy': smoothed_cy,
                'smoothed_radius': smoothed_radius,
                'depth_mm': depth_mm,
                'depth_method': depth_method,
                'detection_method': detection_method,
                'pixel_source': 'ir',  # 主要数据源
                'yolo_bbox': self.last_yolo_bbox,
                'roi': roi,
                'aspect_ratio': ir_detection['aspect_ratio'],
                'brightness': ir_detection['brightness'],
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
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        rgb_image = np.asanyarray(color_frame.get_data())
                        video_writer.write(rgb_image)

                # 获取观测（传入已获取的帧，避免重复获取）
                observation = self.get_observation(frames=frames)

                # 获取图像用于可视化（修复：避免重复align）
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame(1)

                if color_frame and ir_frame:
                    rgb_image = np.asanyarray(color_frame.get_data())
                    ir_image = np.asanyarray(ir_frame.get_data())

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
            self.pipeline.stop()

    def close(self):
        """Release the RealSense pipeline held by this detector."""
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
                label = 'IR KF'  # 标明已经过卡尔曼滤波
                color = (0, 255, 0)  # 绿色
            else:
                # 转换失败，回退到IR原始坐标（会有偏移）
                cx_display = int(raw['cx'])
                cy_display = int(raw['cy'])
                label = 'IR (Approx)'
                color = (0, 200, 200)  # 青绿色表示近似

            radius = int(raw['radius'])

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
                    abs(smoothed_radius - radius) > 1):
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
                f"Depth: {raw['depth_mm']:.0f}mm ({raw['depth_method']})",
                f"KF: speed={kf_stats['speed_px_s']:.1f}px/s outlier={kf_stats['outlier_rate_pct']:.1f}%",
            ]

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

        if observation['detected']:
            raw = observation['raw_measurement']
            cx, cy = int(raw['cx']), int(raw['cy'])
            radius = int(raw['radius'])
            cv2.circle(ir_colored, (cx, cy), radius, (0, 255, 0), 2)
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
