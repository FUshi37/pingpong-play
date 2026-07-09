#!/home/jacky/miniconda3/envs/gmr/bin/python
"""
基于红外反射的乒乓球检测器
利用反光乒乓球在RealSense D455红外图像中的高亮特性实现鲁棒检测
"""

import numpy as np
import cv2
import pyrealsense2 as rs
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any

from pingpong_controller.calibration_loader import CalibrationData


class IRBallDetector:
    """
    红外反光球检测器

    利用RealSense D455的红外相机检测反光乒乓球：
    - 反光球在红外下产生高亮度响应
    - 简单阈值分割 + 连通域分析
    - 形状约束（圆形度）过滤误检
    - 结合深度传感器获取精确3D位置
    """

    def __init__(self,
                 ball_diameter_mm=40.0,
                 width=1280,
                 height=800,
                 fps=90,
                 ir_threshold=200,
                 adaptive_threshold=True,
                 min_radius=5,
                 max_radius=50,
                 circularity_min=0.7,
                 use_both_ir_cameras=False,
                 laser_power=150,
                 depth_scale=0.001,
                 depth_neighborhood_px=2,
                 camera_tilt_deg=0.0,
                 calibration_data: Optional[CalibrationData] = None,
                 output_frame='base_link',
                 print_every=1):
        """
        初始化红外球检测器

        Args:
            ball_diameter_mm: 球直径(毫米)
            width: 红外图像宽度
            height: 红外图像高度
            fps: 目标帧率（IR可达90fps）
            ir_threshold: 红外亮度阈值（0-255）
            adaptive_threshold: 是否使用自适应阈值
            min_radius: 最小半径(像素)
            max_radius: 最大半径(像素)
            circularity_min: 最小圆形度（防止误检不规则亮斑）
            use_both_ir_cameras: 是否同时使用左右红外相机交叉验证
            laser_power: 红外激光功率（0-360mW，D455默认150）
            depth_scale: 深度单位转换系数（D455默认0.001m）
            depth_neighborhood_px: 深度采样邻域半径
            camera_tilt_deg: 相机倾斜角度
            calibration_data: 标定数据
            output_frame: 输出坐标系 'camera' 或 'base_link'
            print_every: 打印频率
        """
        self.ball_diameter_mm = ball_diameter_mm
        self.ball_radius_mm = ball_diameter_mm / 2.0
        self.ir_threshold = ir_threshold
        self.adaptive_threshold = adaptive_threshold
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.circularity_min = circularity_min
        self.use_both_ir_cameras = use_both_ir_cameras
        self.laser_power = laser_power
        self.depth_scale = depth_scale
        self.depth_neighborhood_px = depth_neighborhood_px
        self.camera_tilt_deg = camera_tilt_deg
        self.print_every = print_every

        # 坐标系配置
        self.output_frame = output_frame
        self.calibration_data = calibration_data
        self.T_base_camera = None
        if calibration_data is not None:
            self.T_base_camera = calibration_data.T_base_camera

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()

        # 配置红外流（使用左红外相机，1=左，2=右）
        # D455支持的IR分辨率：1280x800@30, 1280x720@30, 640x480@90等
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        if use_both_ir_cameras:
            config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

        # 配置深度流（用于3D定位）
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        print(f"启动红外检测器配置:")
        print(f"  分辨率: {width}x{height}@{fps}Hz")
        print(f"  红外阈值: {ir_threshold} (adaptive={adaptive_threshold})")
        print(f"  半径范围: {min_radius}-{max_radius}px")
        print(f"  圆形度阈值: {circularity_min}")
        print(f"  激光功率: {laser_power}mW")

        # 启动pipeline
        profile = self.pipeline.start(config)

        # 设置红外激光功率
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.laser_power):
            depth_sensor.set_option(rs.option.laser_power, laser_power)
            actual_power = depth_sensor.get_option(rs.option.laser_power)
            print(f"  实际激光功率: {actual_power}mW")

        # 获取内参
        ir_stream = profile.get_stream(rs.stream.infrared, 1)
        ir_intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()

        if calibration_data is not None:
            # 使用标定数据的内参（假设红外和RGB内参相近，或后续做标定）
            self.camera_matrix = calibration_data.camera_matrix.copy()
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            print(f"  使用标定内参: fx={self.fx:.2f}, fy={self.fy:.2f}")
        else:
            # 使用红外相机内参
            self.fx = ir_intrinsics.fx
            self.fy = ir_intrinsics.fy
            self.cx = ir_intrinsics.ppx
            self.cy = ir_intrinsics.ppy
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
            print(f"  红外内参: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")

        self.image_width = ir_intrinsics.width
        self.image_height = ir_intrinsics.height

        # 对齐深度到红外
        align_to = rs.stream.infrared
        self.align = rs.align(align_to)

        # 性能统计
        self.total_frames = 0
        self.detected_frames = 0
        self.fps_queue = deque(maxlen=30)
        self.last_observation_time = None

        # 位置历史
        self.position_history = deque(maxlen=5)

        # 自适应阈值参数
        self.background_intensity = None
        self.threshold_history = deque(maxlen=10)

        # 预热相机
        print("红外相机预热中...")
        for i in range(30):
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            ir_frame = frames.get_infrared_frame(1)
            if ir_frame and i == 0:
                test_img = np.asanyarray(ir_frame.get_data())
                print(f"  红外图像格式: shape={test_img.shape}, dtype={test_img.dtype}")
        print("红外相机就绪")

    def _update_adaptive_threshold(self, ir_image: np.ndarray) -> int:
        """
        根据背景亮度自适应调整阈值

        Args:
            ir_image: 红外图像

        Returns:
            调整后的阈值
        """
        # 取图像中心区域作为背景采样（假设球不总在中心）
        h, w = ir_image.shape[:2]
        center_region = ir_image[h//4:3*h//4, w//4:3*w//4]

        # 背景亮度取较低分位数（排除可能的亮斑）
        bg_intensity = np.percentile(center_region, 30)

        if self.background_intensity is None:
            self.background_intensity = bg_intensity
        else:
            # 平滑更新
            alpha = 0.1
            self.background_intensity = alpha * bg_intensity + (1 - alpha) * self.background_intensity

        # 阈值 = 背景 + 动态余量（背景越亮，余量越大）
        # 反光球通常比背景亮100-150单位
        threshold = int(self.background_intensity + max(80, self.background_intensity * 0.5))
        threshold = np.clip(threshold, 150, 240)

        self.threshold_history.append(threshold)
        return int(np.median(list(self.threshold_history)))

    def detect_ball_in_ir(self, ir_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        在红外图像中检测反光球

        Args:
            ir_image: 红外图像（单通道，uint8）

        Returns:
            检测结果字典，包含中心、半径等信息，未检测到返回None
        """
        # 1. 自适应阈值
        if self.adaptive_threshold:
            threshold = self._update_adaptive_threshold(ir_image)
        else:
            threshold = self.ir_threshold

        # 2. 阈值分割
        _, binary = cv2.threshold(ir_image, threshold, 255, cv2.THRESH_BINARY)

        # 3. 形态学处理去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 4. 连通域分析
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 5. 筛选候选：面积、半径、圆形度
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # 过滤噪声
                continue

            # 拟合椭圆（需要>=5个点）
            if len(cnt) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx, cy), (minor, major), angle = ellipse

                    # 半径约束
                    radius = minor / 2.0
                    if not (self.min_radius < radius < self.max_radius):
                        continue

                    # 圆形度约束（长短轴比）
                    aspect_ratio = minor / (major + 1e-6)
                    if aspect_ratio < self.circularity_min:
                        continue

                    # 面积圆形度
                    circle_area = np.pi * radius * radius
                    area_circularity = area / (circle_area + 1e-6)
                    if not (0.6 < area_circularity < 1.4):
                        continue

                    # 计算亮度积分作为置信度
                    mask = np.zeros(ir_image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
                    brightness = np.sum(ir_image[mask > 0])

                    candidates.append({
                        'cx': cx,
                        'cy': cy,
                        'radius': radius,
                        'aspect_ratio': aspect_ratio,
                        'area_circularity': area_circularity,
                        'brightness': brightness,
                        'contour': cnt,
                        'mask': mask,
                    })
                except cv2.error:
                    continue

        if not candidates:
            return None

        # 6. 选择最亮的候选（反光球应该是最亮的）
        best = max(candidates, key=lambda x: x['brightness'])

        return {
            'cx': best['cx'],
            'cy': best['cy'],
            'radius': best['radius'],
            'aspect_ratio': best['aspect_ratio'],
            'area_circularity': best['area_circularity'],
            'brightness': best['brightness'],
            'threshold_used': threshold,
            'mask': best['mask'],
            'contour': best['contour'],
        }

    def get_depth_from_depthmap(self, cx_px: float, cy_px: float,
                                depth_image: np.ndarray,
                                ball_mask: Optional[np.ndarray] = None) -> Optional[float]:
        """
        从深度图读取球心深度

        Args:
            cx_px, cy_px: 球心像素坐标
            depth_image: 深度图
            ball_mask: 可选，球的mask（优先在mask内采样）

        Returns:
            深度(mm)，失败返回None
        """
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        cx = int(round(cx_px))
        cy = int(round(cy_px))

        if not (0 <= cx < w and 0 <= cy < h):
            return None

        # 采样策略：优先mask内，否则球心邻域
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

        # 取近分位数（球面在背景前）
        depth_mm = float(np.percentile(depths, 25)) * self.depth_scale * 1000.0

        # 合理性检查
        if depth_mm < 100.0 or depth_mm > 5000.0:
            return None

        return depth_mm

    def backproject_to_3d(self, cx_px: float, cy_px: float, depth_mm: float) -> Tuple[float, float, float]:
        """
        将像素坐标反投影到3D

        Returns:
            (X, Y, Z) in mm, camera frame
        """
        Z = depth_mm
        X = (cx_px - self.cx) * Z / self.fx
        Y = (cy_px - self.cy) * Z / self.fy
        return (X, Y, Z)

    def _transform_to_base_link(self, position_camera: list) -> list:
        """转换到base_link坐标系"""
        if self.T_base_camera is None:
            return position_camera

        pos_camera_m = np.array(position_camera) / 1000.0
        pos_homo = np.append(pos_camera_m, 1.0)
        pos_base_homo = self.T_base_camera @ pos_homo
        return (pos_base_homo[:3] * 1000.0).tolist()

    def _calculate_velocity(self) -> list:
        """从位置历史计算速度"""
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

    def get_observation(self, ir_image: np.ndarray = None,
                       depth_image: np.ndarray = None,
                       observation_time: float = None) -> Dict[str, Any]:
        """
        获取球的观测值

        Args:
            ir_image: 可选，外部提供的红外图像（None则从pipeline获取）
            depth_image: 可选，外部提供的深度图
            observation_time: 观测时间戳

        Returns:
            观测字典
        """
        now = time.time() if observation_time is None else float(observation_time)

        # 从pipeline获取帧（如果未外部提供）
        if ir_image is None or depth_image is None:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)

            if ir_image is None:
                ir_frame = aligned_frames.get_infrared_frame(1)
                if not ir_frame:
                    return {'detected': False, 'position': [0, 0, 0], 'velocity': [0, 0, 0]}
                ir_image = np.asanyarray(ir_frame.get_data())

            if depth_image is None:
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame:
                    return {'detected': False, 'position': [0, 0, 0], 'velocity': [0, 0, 0]}
                depth_image = np.asanyarray(depth_frame.get_data())

        # 检测球
        detection = self.detect_ball_in_ir(ir_image)

        if detection is None:
            return {
                'detected': False,
                'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'raw_measurement': None,
            }

        # 获取深度
        depth_mm = self.get_depth_from_depthmap(
            detection['cx'], detection['cy'], depth_image, detection['mask'])

        if depth_mm is None:
            # fallback到单目几何
            f = (self.fx + self.fy) / 2.0
            depth_mm = (f * self.ball_diameter_mm) / (2.0 * detection['radius'])
            depth_method = 'geometry_fallback'
        else:
            depth_method = 'sensor'

        # 反投影到3D
        X, Y, Z = self.backproject_to_3d(detection['cx'], detection['cy'], depth_mm)
        pos_camera = [X, Y, Z]

        # 转换坐标系
        if self.output_frame == 'base_link' and self.T_base_camera is not None:
            position = self._transform_to_base_link(pos_camera)
        else:
            position = pos_camera

        # 更新历史
        self.position_history.append((now, position))

        # 计算速度
        velocity = self._calculate_velocity()

        return {
            'detected': True,
            'position': position,
            'velocity': velocity,
            'raw_measurement': {
                'cx': detection['cx'],
                'cy': detection['cy'],
                'radius': detection['radius'],
                'depth_mm': depth_mm,
                'depth_method': depth_method,
                'aspect_ratio': detection['aspect_ratio'],
                'area_circularity': detection['area_circularity'],
                'brightness': detection['brightness'],
                'threshold': detection['threshold_used'],
                'raw_position': pos_camera,
            },
        }

    def run(self):
        """主循环"""
        print("红外检测器运行中，Ctrl+C 退出...")

        try:
            while True:
                t0 = time.time()

                observation = self.get_observation()

                self.total_frames += 1
                if observation['detected']:
                    self.detected_frames += 1

                # 打印
                if self.total_frames % self.print_every == 0:
                    if observation['detected']:
                        pos = observation['position']
                        vel = observation['velocity']
                        raw = observation['raw_measurement']

                        print(
                            f"IR检测 - "
                            f"像素: ({raw['cx']:7.1f}, {raw['cy']:7.1f})  "
                            f"半径: {raw['radius']:6.2f}px  "
                            f"深度: {raw['depth_mm']:7.1f}mm [{raw['depth_method']}]  "
                            f"亮度: {raw['brightness']:8.0f}  "
                            f"阈值: {raw['threshold']:3d}  "
                            f"圆度: {raw['aspect_ratio']:.2f}/{raw['area_circularity']:.2f}  "
                            f"位置: ({pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:7.1f})mm  "
                            f"速度: ({vel[0]:6.0f}, {vel[1]:6.0f}, {vel[2]:6.0f})mm/s"
                        )
                    else:
                        print("未检测到")

                # FPS
                t_end = time.time()
                dt = t_end - t0
                self.fps_queue.append(1.0 / dt if dt > 0 else 0)

        except KeyboardInterrupt:
            print("\n检测已停止")
        finally:
            if self.total_frames > 0:
                detection_rate = 100.0 * self.detected_frames / self.total_frames
                avg_fps = np.mean(list(self.fps_queue)) if self.fps_queue else 0
                print("\n" + "="*60)
                print("红外检测统计")
                print("="*60)
                print(f"总帧数: {self.total_frames}")
                print(f"检测成功: {self.detected_frames} ({detection_rate:.2f}%)")
                print(f"平均FPS: {avg_fps:.1f}")
                print("="*60)

            self.pipeline.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='红外反光球检测器')
    parser.add_argument('--width', type=int, default=1280, help='红外图像宽度')
    parser.add_argument('--height', type=int, default=800, help='红外图像高度')
    parser.add_argument('--fps', type=int, default=90, help='目标帧率')
    parser.add_argument('--threshold', type=int, default=200, help='红外亮度阈值')
    parser.add_argument('--no-adaptive', dest='adaptive', action='store_false',
                       default=True, help='禁用自适应阈值')
    parser.add_argument('--laser-power', type=int, default=150,
                       help='红外激光功率(0-360mW)')
    parser.add_argument('--both-ir', action='store_true',
                       help='使用双红外相机')
    parser.add_argument('--print-every', type=int, default=1)

    args = parser.parse_args()

    print("="*60)
    print("红外反光球检测器启动配置")
    print("="*60)
    print(f"  分辨率: {args.width}x{args.height}@{args.fps}Hz")
    print(f"  阈值: {args.threshold} (自适应={args.adaptive})")
    print(f"  激光功率: {args.laser_power}mW")
    print(f"  双红外相机: {args.both_ir}")
    print("="*60)

    try:
        detector = IRBallDetector(
            width=args.width,
            height=args.height,
            fps=args.fps,
            ir_threshold=args.threshold,
            adaptive_threshold=args.adaptive,
            use_both_ir_cameras=args.both_ir,
            laser_power=args.laser_power,
            print_every=args.print_every,
        )
        detector.run()
    except KeyboardInterrupt:
        print("\n已退出")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
