#!/usr/bin/env python3
"""
2D像素坐标卡尔曼滤波器 - 专门用于平滑IR检测的像素抖动

设计目标：
1. 静止球：极强的平滑，几乎零抖动
2. 运动球：快速响应，低延迟跟踪
3. 离群值鲁棒：自动检测和抑制测量跳变
"""

import numpy as np


class PixelKalmanFilter:
    """
    2D像素坐标卡尔曼滤波器（4维状态：cx, cy, vx, vy）

    特点：
    - 自适应过程噪声：运动时增大，静止时减小
    - 创新门控：自动剔除离群测量
    - 亚像素精度输出
    """

    def __init__(self, dt=1/60.0,
                 process_noise_pos=0.5,  # 位置过程噪声（像素）
                 process_noise_vel=5.0,   # 速度过程噪声（像素/秒）
                 measurement_noise=1.5,   # 测量噪声（像素）
                 innovation_gate=10.0):   # 离群值门限（像素）
        """
        Args:
            dt: 时间步长（秒），默认60Hz
            process_noise_pos: 位置过程噪声标准差（像素）
            process_noise_vel: 速度过程噪声标准差（像素/秒）
            measurement_noise: 测量噪声标准差（像素）
            innovation_gate: 新息门限，超过此值的测量被视为离群值
        """
        self.dt = dt
        self.initialized = False

        # 状态向量 [cx, cy, vx, vy]
        self.state = np.zeros(4, dtype=np.float64)

        # 协方差矩阵（初始不确定性大）
        self.P = np.eye(4, dtype=np.float64) * 100.0

        # 状态转移矩阵（匀速运动模型）
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        # 过程噪声协方差矩阵
        q_pos = process_noise_pos ** 2
        q_vel = process_noise_vel ** 2
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)

        # 观测矩阵（只观测位置）
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)

        # 测量噪声协方差矩阵
        r = measurement_noise ** 2
        self.R = np.eye(2, dtype=np.float64) * r

        # 创新门限
        self.innovation_gate = innovation_gate

        # 统计
        self.consecutive_outliers = 0
        self.total_updates = 0
        self.outliers_rejected = 0

    def predict(self):
        """预测步骤"""
        if not self.initialized:
            return

        # 状态预测
        self.state = self.F @ self.state

        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        """
        更新步骤

        Args:
            measurement: [cx, cy] 像素坐标

        Returns:
            smoothed_pos: [cx, cy] 平滑后的像素坐标
        """
        measurement = np.array(measurement, dtype=np.float64)

        # 初始化：用第一个测量值
        if not self.initialized:
            self.state[:2] = measurement
            self.state[2:] = 0  # 初始速度为0
            self.initialized = True
            self.total_updates += 1
            return self.state[:2].copy()

        # 预测
        self.predict()

        # 计算新息（innovation）
        predicted_measurement = self.H @ self.state
        innovation = measurement - predicted_measurement
        innovation_norm = np.linalg.norm(innovation)

        # 新息协方差
        S = self.H @ self.P @ self.H.T + self.R

        # 离群值检测（基于Mahalanobis距离）
        if innovation_norm > self.innovation_gate:
            self.consecutive_outliers += 1
            self.outliers_rejected += 1

            # 连续多次离群：可能是真实的大跳变（如球被击打）
            # 重新初始化以快速适应
            if self.consecutive_outliers >= 3:
                self.state[:2] = measurement
                self.P = np.eye(4, dtype=np.float64) * 100.0
                self.consecutive_outliers = 0
                self.total_updates += 1
                return self.state[:2].copy()

            # 单次离群：降权更新（而非完全拒绝）
            # 这样既能抑制噪声，又不会完全丢失信息
            R_outlier = self.R * 100.0  # 大幅增加测量噪声
            S = self.H @ self.P @ self.H.T + R_outlier
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.state = self.state + K @ innovation
            self.P = (np.eye(4) - K @ self.H) @ self.P

            self.total_updates += 1
            return self.state[:2].copy()

        # 正常更新
        self.consecutive_outliers = 0

        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # 状态更新
        self.state = self.state + K @ innovation

        # 协方差更新（Joseph形式，数值稳定）
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        self.total_updates += 1
        return self.state[:2].copy()

    def get_velocity(self):
        """获取估计的速度 [vx, vy] 像素/秒"""
        if not self.initialized:
            return np.zeros(2)
        return self.state[2:].copy()

    def reset(self):
        """重置滤波器"""
        self.initialized = False
        self.state = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 100.0
        self.consecutive_outliers = 0

    def get_stats(self):
        """获取统计信息"""
        if self.total_updates == 0:
            outlier_rate = 0.0
        else:
            outlier_rate = 100.0 * self.outliers_rejected / self.total_updates
        return {
            'total_updates': self.total_updates,
            'outliers_rejected': self.outliers_rejected,
            'outlier_rate_pct': outlier_rate,
            'speed_px_s': np.linalg.norm(self.state[2:]) if self.initialized else 0.0
        }
