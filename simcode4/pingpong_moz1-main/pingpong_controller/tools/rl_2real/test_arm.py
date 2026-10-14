#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_right_arm_joint_data(csv_path):
    """
    加载右臂关节数据并绘制位置、速度、加速度图表

    Args:
        csv_path: CSV 文件路径
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 提取时间和关节位置数据
    time_array = df['time_s'].values
    joint_columns = [f'RightArm-{i}' for i in range(7)]
    positions = df[joint_columns].values

    # 打印平均时间间隔
    mean_dt = np.mean(np.diff(time_array))
    print(f"Average time interval: {mean_dt:.6f} s")

    # 使用 np.gradient 计算速度和加速度
    velocity = np.gradient(positions, time_array, axis=0)
    acceleration = np.gradient(velocity, time_array, axis=0)

    # 创建图例标签
    joint_labels = [f'joint_{i+1}' for i in range(7)]

    # 图1: 位置 - 7个子图
    fig1, axes1 = plt.subplots(7, 1, figsize=(12, 14))
    fig1.suptitle('Right Arm Joint Positions', fontsize=16)
    for i in range(7):
        axes1[i].plot(time_array, positions[:, i], label=joint_labels[i])
        axes1[i].set_ylabel('Position (deg)')
        axes1[i].legend(loc='upper right')
        axes1[i].grid(True)
    axes1[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    # 图2: 速度 - 7个子图
    fig2, axes2 = plt.subplots(7, 1, figsize=(12, 14))
    fig2.suptitle('Right Arm Joint Velocities', fontsize=16)
    for i in range(7):
        axes2[i].plot(time_array, velocity[:, i], label=joint_labels[i])
        axes2[i].set_ylabel('Velocity (deg/s)')
        axes2[i].legend(loc='upper right')
        axes2[i].grid(True)
    axes2[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    # 图3: 加速度 - 7个子图
    fig3, axes3 = plt.subplots(7, 1, figsize=(12, 14))
    fig3.suptitle('Right Arm Joint Accelerations', fontsize=16)
    for i in range(7):
        axes3[i].plot(time_array, acceleration[:, i], label=joint_labels[i])
        axes3[i].set_ylabel('Acceleration (deg/s²)')
        axes3[i].legend(loc='upper right')
        axes3[i].grid(True)
    axes3[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    from pathlib import Path
    # 从当前文件位置推导包目录
    current_file = Path(__file__).resolve()
    # test_arm.py -> rl_2real -> tools -> pingpong_controller -> pingpong_controller
    package_dir = current_file.parent.parent.parent
    csv_path = package_dir / 'data' / 'right_arm_joints_test.csv'
    plot_right_arm_joint_data(str(csv_path))
