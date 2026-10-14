#!/usr/bin/env python3
"""
RealSense 图像采集工具
用于收集 YOLO 训练数据
按空格键保存当前帧
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path

class ImageCollector:
    def __init__(self, save_dir, width=1280, height=720, fps=30):
        """
        初始化图像采集器

        Args:
            save_dir: 图像保存目录
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        self.save_dir = save_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.image_count = 0

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 自动识别已有图像数量，从最大编号+1开始
        existing_images = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))]
        if existing_images:
            numbers = []
            for f in existing_images:
                try:
                    # 提取文件名中的编号（格式：frame_0001.jpg）
                    if f.startswith('frame_'):
                        num_str = f.split('_')[1].split('.')[0]
                        numbers.append(int(num_str))
                except:
                    pass
            if numbers:
                self.image_count = max(numbers) + 1
                print(f"检测到已有 {len(existing_images)} 张图像，最大编号: {max(numbers)}")
            else:
                print(f"检测到 {len(existing_images)} 张图像，但无法识别编号")
        else:
            print("目录为空，从 0 开始")

        print(f"保存目录: {save_dir}")
        print(f"下一张编号: {self.image_count}")

        # 初始化 RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 配置彩色流
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        # 启动流
        try:
            self.pipeline.start(self.config)
            print(f"RealSense 已启动: {width}x{height}@{fps}fps")
        except Exception as e:
            print(f"启动失败: {e}")
            raise

        # 预热相机
        print("相机预热中...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("相机就绪")

    def save_image(self, image):
        """保存图像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{self.image_count:04d}_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, filename)

        cv2.imwrite(filepath, image)
        print(f"已保存: {filename}")
        self.image_count += 1

    def run(self):
        """主循环"""
        print("\n" + "="*60)
        print("图像采集工具")
        print("="*60)
        print("操作说明:")
        print("  空格键 - 保存当前帧")
        print("  q 键   - 退出程序")
        print("="*60 + "\n")

        try:
            while True:
                # 获取帧
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                # 转换为 numpy 数组
                color_image = np.asanyarray(color_frame.get_data())

                # 在图像上显示信息
                display_image = color_image.copy()

                # 显示计数和提示
                info_text = f"Images saved: {self.image_count} | Press SPACE to save, Q to quit"
                cv2.putText(display_image, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示分辨率
                res_text = f"Resolution: {self.width}x{self.height}"
                cv2.putText(display_image, res_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 显示图像
                cv2.imshow("RealSense Image Collector", display_image)

                # 处理按键
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # 空格键保存
                    self.save_image(color_image)
                elif key == ord('q'):  # q 键退出
                    print("\n退出程序")
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print(f"\n总共保存了 {self.image_count} 张图像")
            print(f"保存位置: {self.save_dir}")


if __name__ == "__main__":
    # 配置参数
    current_file = Path(__file__).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = str(
        current_file.parent
        / ".."
        / ".."
        / "data"
        / "vision_refine"
        / "yolo_image_pingpong"
        / timestamp
    )
    WIDTH = 848
    HEIGHT = 480
    FPS = 60

    try:
        collector = ImageCollector(SAVE_DIR, WIDTH, HEIGHT, FPS)
        collector.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
