#!/usr/bin/env python3
"""红外检测参数可视化调节工具。

本文件依据 ``__pycache__/ir_tuning_tool.cpython-310.pyc`` 恢复。它实时显示
RGB、IR 和二值图，并允许调节、加载和保存 ``ir_params.json``。

按键：q/ESC 退出，s 保存，l 加载，r 重置，空格暂停，h 显示帮助，
方向键选择/微调参数，[ / ] 快速调节，1-7 应用预设。
"""

from collections import deque
import json
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from src.pingpong_moz1.pingpong_controller.ir_center_estimator import (
    refine_ir_ball_center,
)


class IRTuningTool:
    DEFAULT_PARAMS = {
        "laser_power": 360,
        "exposure": 1500,
        "gain": 16,
        "ir_threshold": 140,
        "use_adaptive": False,
        "min_radius": 10,
        "max_radius": 70,
        "circularity_min": 0.7,
        "dilate_iterations": 9,
        "close_kernel_size": 5,
        "denoise_kernel_size": 3,
        "brightness_percentile": 25,
    }

    def __init__(self):
        self.params = self.DEFAULT_PARAMS.copy()
        self.param_ranges = {
            "laser_power": (0, 360, 10),
            "exposure": (1, 166000, 50),
            "gain": (16, 63, 1),
            "ir_threshold": (100, 250, 5),
            "min_radius": (3, 20, 1),
            "max_radius": (30, 100, 1),
            "circularity_min": (0.5, 0.95, 0.05),
            # 扩大形态学参数的可视化调节范围。两个 kernel 从 3 开始、
            # 步长为 2，以保证通过方向键得到的始终是奇数尺寸。
            "dilate_iterations": (0, 20, 1),
            "close_kernel_size": (3, 31, 2),
            "denoise_kernel_size": (3, 21, 2),
            "brightness_percentile": (10, 50, 5),
        }
        self.presets = {
            "1": {"name": "近距离(0.5-1m)", "laser_power": 120,
                  "ir_threshold": 210, "dilate_iterations": 3},
            "2": {"name": "中距离(1-1.5m)", "laser_power": 150,
                  "ir_threshold": 200, "dilate_iterations": 2},
            "3": {"name": "远距离(1.5-2m)", "laser_power": 200,
                  "ir_threshold": 190, "dilate_iterations": 2},
            "4": {"name": "高速运动", "exposure": 5000,
                  "laser_power": 180, "dilate_iterations": 3},
            "5": {"name": "强反光抑制", "laser_power": 100,
                  "circularity_min": 0.85, "dilate_iterations": 2},
            "6": {"name": "弱光增强", "laser_power": 250, "gain": 16,
                  "exposure": 12000, "dilate_iterations": 4},
            "7": {"name": "高精度", "min_radius": 8, "max_radius": 40,
                  "circularity_min": 0.8, "dilate_iterations": 2},
            "8": {"name": "分散点连接", "dilate_iterations": 4,
                  "close_kernel_size": 11, "denoise_kernel_size": 3},
        }
        self.current_param_idx = 0
        self.param_keys = list(self.params)
        self.paused = False
        self.show_help = True
        self.fps_queue = deque(maxlen=30)
        self.detection_history = deque(maxlen=100)
        self.brightness_history = deque(maxlen=100)

        self.depth_history = deque(maxlen=5)
        self.depth_source_state = "sensor"
        self.ball_diameter_mm = 40.0

        self.pipeline = rs.pipeline()
        config = rs.config()
        self.ir_width, self.ir_height, self.ir_fps = 848, 480, 60
        try:
            config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
            config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 60)
            config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
            print("  使用配置: RGB+IR+Depth 848x480@60fps")
        except Exception:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
                config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
                self.ir_width, self.ir_height, self.ir_fps = 640, 480, 60
                print("  使用配置: RGB+IR+Depth 640x480@60fps")
            except Exception:
                print("错误: 无法同时启用RGB+IR+Depth流")
                raise

        print("启动RealSense相机...")
        self.profile = self.pipeline.start(config)
        device = self.profile.get_device()
        self.depth_sensor = device.first_depth_sensor()

        ir_stream = self.profile.get_stream(rs.stream.infrared, 1)
        self.ir_intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()
        color_stream = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.depth_to_color_extrinsics = depth_stream.get_extrinsics_to(color_stream)

        print(f"  IR内参: fx={self.ir_intrinsics.fx:.2f}, fy={self.ir_intrinsics.fy:.2f}")
        print(f"  RGB内参: fx={self.color_intrinsics.fx:.2f}, fy={self.color_intrinsics.fy:.2f}")
        self._apply_sensor_params()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        print(f"  深度尺度: {self.depth_scale} m/count")

        print("相机预热中...")
        for _ in range(10):
            self.pipeline.wait_for_frames()

        print("\n" + "=" * 70)
        print("红外检测参数调节工具启动成功")
        print("=" * 70)
        self._print_help()

    def _apply_sensor_params(self):
        """应用传感器参数。"""
        try:
            if self.depth_sensor.supports(rs.option.laser_power):
                self.depth_sensor.set_option(
                    rs.option.laser_power, self.params["laser_power"])
            if self.depth_sensor.supports(rs.option.enable_auto_exposure):
                self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if self.depth_sensor.supports(rs.option.exposure):
                self.depth_sensor.set_option(
                    rs.option.exposure, self.params["exposure"])
            if self.depth_sensor.supports(rs.option.gain):
                self.depth_sensor.set_option(rs.option.gain, self.params["gain"])
        except Exception as exc:
            print(f"设置传感器参数失败: {exc}")

    def _print_help(self):
        """打印帮助信息。"""
        print("\n【控制键】")
        print("  ↑/↓      - 选择参数")
        print("  ←/→      - 调整参数值")
        print("  [/]      - 快速减少/增加")
        print("  SPACE    - 暂停/继续")
        print("  h        - 显示/隐藏帮助")
        print("  r        - 重置为默认值")
        print("  s        - 保存参数到文件")
        print("  l        - 从文件加载参数")
        print("  1-7      - 应用预设配置")
        print("  d        - 切换显示模式")
        print("  q/ESC    - 退出")
        print("\n【预设配置】")
        for key, preset in self.presets.items():
            print(f"  {key} - {preset['name']}")
        print("\n【形态学说明】")
        print("  dilate_iterations - 膨胀迭代次数，让点变大更容易连接")
        print("  close_kernel_size - 闭运算核大小，填充点之间的空隙")
        print("  denoise_kernel_size - 去噪核大小，清理小噪点")

    def _geometry_depth_mm(self, radius_px):
        """单目几何测距 Z = f*D/(2r)。"""
        if radius_px is None or radius_px <= 0.001:
            return None
        focal_length = (self.ir_intrinsics.fx + self.ir_intrinsics.fy) / 2.0
        return focal_length * self.ball_diameter_mm / (2.0 * radius_px)

    def _select_depth(self, sensor_depth_mm, geom_depth_mm):
        """优先使用实测深度；形态学半径测距仅作无深度时的兜底。"""
        if sensor_depth_mm is not None:
            self.depth_source_state = "sensor"
            return sensor_depth_mm, "sensor"
        if geom_depth_mm is not None:
            self.depth_source_state = "geometry_fallback"
            return geom_depth_mm, "geometry"
        return None, "none"

    def _stabilize_depth(self, depth_raw):
        """深度时域稳定化：软钳制尖峰并进行中位数平滑。"""
        history = self.depth_history
        if len(history) >= 3:
            median = float(np.median(list(history)))
            gate = max(150.0, 0.25 * median)
            deviation = depth_raw - median
            if abs(deviation) > gate:
                depth_raw = median + np.sign(deviation) * gate
        history.append(depth_raw)
        if len(history) >= 3:
            return float(np.median(list(history)))
        return depth_raw

    def _transform_ir_to_rgb_pixel(self, ir_x, ir_y, depth_m):
        """把 IR 像素按测得深度投影到 RGB 图像。"""
        if depth_m <= 0 or depth_m > 5.0:
            return None
        try:
            point_ir = rs.rs2_deproject_pixel_to_point(
                self.ir_intrinsics, [ir_x, ir_y], depth_m)
            point_color = rs.rs2_transform_point_to_point(
                self.depth_to_color_extrinsics, point_ir)
            pixel_rgb = rs.rs2_project_point_to_pixel(
                self.color_intrinsics, point_color)
            if (0 <= pixel_rgb[0] < self.ir_width and
                    0 <= pixel_rgb[1] < self.ir_height):
                return pixel_rgb[0], pixel_rgb[1]
        except Exception:
            pass
        return None

    def _get_depth_from_depthmap(self, cx_px, cy_px, depth_image):
        """从目标中心的 3x3 深度邻域读取中位数（毫米）。"""
        h, w = depth_image.shape[:2]
        cx, cy = int(round(cx_px)), int(round(cy_px))
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        radius = 1
        patch = depth_image[
            max(0, cy - radius):min(h, cy + radius + 1),
            max(0, cx - radius):min(w, cx + radius + 1),
        ]
        depths = patch[patch > 0]
        if depths.size < 3:
            return None
        depth_mm = float(np.median(depths)) * self.depth_scale * 1000.0
        if depth_mm < 100.0 or depth_mm > 5000.0:
            return None
        return depth_mm

    def _update_adaptive_threshold(self, ir_image):
        """计算自适应阈值。"""
        h, w = ir_image.shape[:2]
        center = ir_image[h // 4:3 * h // 4, w // 4:3 * w // 4]
        percentile = self.params.get("brightness_percentile", 30)
        background = np.percentile(center, percentile)
        threshold = int(background + max(80, background * 0.5))
        return int(np.clip(threshold, 150, 240))

    def _detect_ball(self, ir_image, depth_image):
        """执行球检测；形态学操作用于连接分散的高亮点。"""
        if self.params["use_adaptive"]:
            threshold = self._update_adaptive_threshold(ir_image)
        else:
            threshold = self.params["ir_threshold"]
        _, binary = cv2.threshold(ir_image, threshold, 255, cv2.THRESH_BINARY)

        dilate_iterations = int(self.params["dilate_iterations"])
        if dilate_iterations > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.dilate(binary, kernel, iterations=dilate_iterations)

        close_size = int(self.params["close_kernel_size"])
        if close_size >= 3:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        denoise_size = int(self.params["denoise_kernel_size"])
        if denoise_size >= 3:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (denoise_size, denoise_size))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20 or len(contour) < 5:
                continue
            try:
                (coarse_cx, coarse_cy), (minor, major), _ = cv2.fitEllipse(
                    contour)
                radius = minor / 2.0
                if not self.params["min_radius"] < radius < self.params["max_radius"]:
                    continue
                aspect_ratio = minor / (major + 1e-6)
                if aspect_ratio < self.params["circularity_min"]:
                    continue
                circle_area = np.pi * radius * radius
                area_circularity = area / (circle_area + 1e-6)
                if not 0.6 < area_circularity < 1.4:
                    continue

                cx, cy, center_refine_shift = refine_ir_ball_center(
                    ir_image, (coarse_cx, coarse_cy), radius)

                mask = np.zeros(ir_image.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
                brightness = np.sum(ir_image[mask > 0])
                sensor_depth = self._get_depth_from_depthmap(cx, cy, depth_image)
                geometry_depth = self._geometry_depth_mm(radius)
                depth_raw, depth_method = self._select_depth(
                    sensor_depth, geometry_depth)
                if depth_raw is not None:
                    depth_mm = self._stabilize_depth(depth_raw)
                    depth_m = depth_mm / 1000.0
                    projected_radius = self._geometry_depth_mm(1.0) / depth_mm
                else:
                    depth_m = depth_mm = None
                    projected_radius = None

                candidates.append({
                    "cx": cx, "cy": cy, "radius": radius,
                    "cx_coarse": coarse_cx, "cy_coarse": coarse_cy,
                    "center_refine_shift_px": center_refine_shift,
                    "aspect_ratio": aspect_ratio,
                    "area_circularity": area_circularity,
                    "brightness": brightness, "contour": contour, "mask": mask,
                    "depth_m": depth_m, "depth_mm": depth_mm,
                    "depth_method": depth_method,
                    "projected_radius_px": projected_radius,
                    "sensor_depth_mm": sensor_depth,
                    "geom_depth_mm": geometry_depth,
                })
            except Exception:
                continue

        if not candidates:
            return None, binary, threshold
        return max(candidates, key=lambda item: item["brightness"]), binary, threshold

    @staticmethod
    def _put_text(image, text, position, scale=0.5, color=(255, 255, 255),
                  thickness=1):
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def _draw_ui(self, rgb_image, ir_image, binary, detection, threshold):
        """绘制 RGB、IR、二值图和参数面板。"""
        h, w = rgb_image.shape[:2]
        rgb_display = rgb_image.copy()
        ir_colored = cv2.applyColorMap(ir_image, cv2.COLORMAP_JET)
        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        if detection is not None:
            cx, cy = int(detection["cx"]), int(detection["cy"])
            radius = int(round(detection.get("projected_radius_px")
                               or detection["radius"]))
            for image in (ir_colored, binary_colored):
                cv2.circle(image, (cx, cy), radius, (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

            depth_m = detection.get("depth_m")
            rgb_pixel = (self._transform_ir_to_rgb_pixel(cx, cy, depth_m)
                         if depth_m is not None else None)
            if rgb_pixel is not None:
                rgb_x, rgb_y = map(int, rgb_pixel)
                cv2.circle(rgb_display, (rgb_x, rgb_y), radius,
                           (0, 255, 0), 3)
                cv2.circle(rgb_display, (rgb_x, rgb_y), 4,
                           (0, 0, 255), -1)
                depth_text = (f"Z: {detection['depth_mm']:.0f}mm "
                              f"({detection['depth_method']})")
                self._put_text(rgb_display, depth_text,
                               (rgb_x + 10, max(25, rgb_y - 10)), 0.5,
                               (0, 255, 0), 2)

            self._put_text(
                ir_colored,
                f"IR: ({cx}, {cy}) R morph/physical: "
                f"{detection['radius']:.1f}/{detection.get('projected_radius_px') or detection['radius']:.1f}px",
                (10, 25), 0.55, (255, 255, 255), 2)
            self._put_text(
                ir_colored,
                f"Circ: {detection['aspect_ratio']:.2f} "
                f"Bright: {detection['brightness']:.0f} "
                f"CenterShift: {detection['center_refine_shift_px']:.2f}px",
                (10, 48), 0.5, (255, 255, 255), 1)

        self._put_text(rgb_display, "RGB View (IR Detection Overlay)",
                       (10, 30), 0.7, (255, 255, 255), 2)
        self._put_text(ir_colored, "IR View", (10, h - 15), 0.6)
        self._put_text(binary_colored, "Binary (After Morph)",
                       (10, h - 15), 0.6)

        right_panel = np.vstack((ir_colored, binary_colored))
        right_width = int(right_panel.shape[1] * h / right_panel.shape[0])
        right_panel = cv2.resize(right_panel, (right_width, h))
        image_panel = np.hstack((rgb_display, right_panel))

        info_height = 320
        info = np.zeros((info_height, image_panel.shape[1], 3), dtype=np.uint8)
        self._put_text(info, "IR Ball Detector Tuning Tool - RGB+IR Hybrid View",
                       (10, 27), 0.8, (255, 255, 255), 2)
        average_fps = np.mean(self.fps_queue) if self.fps_queue else 0.0
        detection_rate = (np.mean(self.detection_history) * 100
                          if self.detection_history else 0.0)
        stats = (f"FPS: {average_fps:.1f}  Detection: {detection_rate:.1f}%  "
                 f"Threshold: {threshold}  IR Max: {ir_image.max()}, "
                 f"Mean: {ir_image.mean():.1f}")
        self._put_text(info, stats, (10, 58), 0.55, (0, 255, 255), 1)

        columns = (20, 420)
        for index, key in enumerate(self.param_keys):
            value = self.params[key]
            prefix = "> " if index == self.current_param_idx else "  "
            if isinstance(value, bool):
                value_text = "ON" if value else "OFF"
                range_text = ""
            else:
                value_text = f"{value:.2f}" if isinstance(value, float) else str(value)
                limits = self.param_ranges.get(key)
                range_text = f" [{limits[0]}-{limits[1]}]" if limits else ""
            x = columns[0] if index < 6 else columns[1]
            y = 92 + (index % 6) * 30
            color = ((0, 255, 255) if index == self.current_param_idx
                     else (200, 200, 200))
            self._put_text(info, f"{prefix}{key}: {value_text}{range_text}",
                           (x, y), 0.45, color, 1)

        if self.paused:
            self._put_text(info, "PAUSED", (image_panel.shape[1] - 150, 58),
                           0.8, (0, 0, 255), 2)

        if self.show_help:
            help_text = [
                "Controls:", "  Up/Down: Select param",
                "  Left/Right: Adjust", "  [/]: Fast adjust",
                "  SPACE: Pause", "  h: Toggle help", "  s: Save params",
                "  l: Load params", "  1-7: Apply presets", "  q/ESC: Quit",
            ]
            # 帮助放在下方参数区，避免遮挡右上角的 IR 实时图像。
            help_width = 320
            help_x = max(800, info.shape[1] - help_width)
            cv2.line(info, (help_x - 12, 72),
                     (help_x - 12, info_height - 12), (80, 80, 80), 1)
            for index, text in enumerate(help_text):
                self._put_text(info, text,
                               (help_x, 92 + index * 22), 0.45)

        final_display = np.vstack((image_panel, info))
        return final_display

    def _adjust_param(self, delta, fast=False):
        """调整当前选中的参数。"""
        key = self.param_keys[self.current_param_idx]
        if isinstance(self.params[key], bool):
            self.params[key] = not self.params[key]
            return
        minimum, maximum, step = self.param_ranges.get(key, (0, 1000, 1))
        if fast:
            step *= 5
        value = np.clip(self.params[key] + delta * step, minimum, maximum)
        if isinstance(step, float):
            self.params[key] = round(float(value), 2)
        else:
            self.params[key] = int(value)
        if key in ("laser_power", "exposure", "gain"):
            self._apply_sensor_params()

    def _save_params(self, filename="ir_params.json"):
        """保存参数到文件。"""
        try:
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(self.params, file, indent=2)
            print(f"\n参数已保存到: {filename}")
            print(json.dumps(self.params, indent=2))
        except Exception as exc:
            print(f"保存失败: {exc}")

    def _load_params(self, filename="ir_params.json"):
        """从文件加载参数。"""
        try:
            with open(filename, "r", encoding="utf-8") as file:
                loaded = json.load(file)
            self.params.update(loaded)
            self._apply_sensor_params()
            print(f"\n参数已从 {filename} 加载")
        except Exception as exc:
            print(f"加载失败: {exc}")

    def _apply_preset(self, preset_key):
        """应用预设配置。"""
        if preset_key not in self.presets:
            return
        preset = self.presets[preset_key]
        for key, value in preset.items():
            if key != "name" and key in self.params:
                self.params[key] = value
        self._apply_sensor_params()
        print(f"\n已应用预设: {preset['name']}")

    def run(self):
        """运行主循环。"""
        window_name = "IR Tuning Tool"
        # WINDOW_NORMAL 允许拖动窗口边框实时缩放；KEEP_RATIO 避免画面变形。
        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
        )
        cv2.resizeWindow(window_name, 1272, 800)
        try:
            while True:
                started = time.time()
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame(1)
                depth_frame = frames.get_depth_frame()
                if not color_frame or not ir_frame or not depth_frame:
                    continue

                rgb_image = np.asanyarray(color_frame.get_data())
                ir_image = np.asanyarray(ir_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                detection, binary, threshold = self._detect_ball(
                    ir_image, depth_image)
                self.detection_history.append(1 if detection is not None else 0)
                if detection is not None:
                    self.brightness_history.append(detection["brightness"])

                display = self._draw_ui(
                    rgb_image, ir_image, binary, detection, threshold)
                cv2.imshow(window_name, display)
                elapsed = time.time() - started
                self.fps_queue.append(1.0 / elapsed if elapsed > 0 else 0)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    self.paused = not self.paused
                elif key == ord("h"):
                    self.show_help = not self.show_help
                elif key == ord("s"):
                    self._save_params()
                elif key == ord("l"):
                    self._load_params()
                elif key == ord("r"):
                    self.params = self.DEFAULT_PARAMS.copy()
                    self._apply_sensor_params()
                    print("\n参数已重置为默认值")
                elif key in (82, 0):
                    self.current_param_idx = (
                        self.current_param_idx - 1) % len(self.param_keys)
                elif key in (84, 1):
                    self.current_param_idx = (
                        self.current_param_idx + 1) % len(self.param_keys)
                elif key in (81, 2):
                    self._adjust_param(-1, fast=False)
                elif key in (83, 3):
                    self._adjust_param(1, fast=False)
                elif key == ord("["):
                    self._adjust_param(-1, fast=True)
                elif key == ord("]"):
                    self._adjust_param(1, fast=True)
                elif 0 <= key < 256 and chr(key) in "1234567":
                    self._apply_preset(chr(key))

                while self.paused:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord(" "):
                        self.paused = False
                    elif key in (ord("q"), 27):
                        return
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self._print_final_stats()
            self.pipeline.stop()
            cv2.destroyAllWindows()

    def _print_final_stats(self):
        """打印最终统计。"""
        print("\n" + "=" * 70)
        print("调节会话总结")
        print("=" * 70)
        if self.detection_history:
            rate = np.mean(list(self.detection_history)) * 100
            print(f"平均检测率: {rate:.1f}%")
        if self.brightness_history:
            values = list(self.brightness_history)
            print(f"亮度统计: 平均={np.mean(values):.0f}, 中位={np.median(values):.0f}")
        if self.fps_queue:
            print(f"平均FPS: {np.mean(list(self.fps_queue)):.1f}")
        print("\n最终参数:")
        print(json.dumps(self.params, indent=2))
        print("\n提示: 使用 's' 键可以保存参数到 ir_params.json")
        print("=" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("红外检测参数可视化调节工具")
    print("=" * 70)
    try:
        print("\n初始化中...")
        IRTuningTool().run()
        print("\n程序退出")
    except KeyboardInterrupt:
        print("\n程序退出")
    except Exception as exc:
        print(f"\n错误: {exc}")
        import traceback
        traceback.print_exc()
