# 固定球多姿态偏差采集

该工具用于采集固定在球拍几何中心的球，不会创建 ROS 命令 publisher，也不会向
`/mx_mix_command` 下发指令。机械臂必须由操作者通过示教或现有手动方式移动。

## 采集前

1. 将球可靠固定在球拍几何中心，并记录球拍表面到球最低点之间的垫高/间隙。
2. 停止网页 probe 和其他占用 RealSense 的程序。
3. 确认 `/joint_states` 正常，并确保机械臂可以安全地手动移动和停止。

## 运行

在工作区根目录执行：

```bash
./RUN_FIXED_BALL_CALIBRATION.sh
```

默认采集 5 个姿态。每次将机械臂移到目标位置并完全静止后，按 Enter；程序先丢弃
0.6 秒暖机帧，再记录 2.5 秒。也可以采集 3 或 4 个姿态：

```bash
./RUN_FIXED_BALL_CALIBRATION.sh --poses 3
```

推荐姿态应覆盖不同位置和倾角，例如：

- 当前水平中心姿态；
- 左右或前后平移至少 50～100 mm 的姿态；
- 拍面分别向两个方向倾斜约 5～15 度的姿态。

不要只旋转腕部而让球被遮挡，也不要在按 Enter 后继续移动。工具默认要求任一关节
速度不超过 0.03 rad/s；不满足的帧会保留，但标记为不可用于标定。

## 输出

数据默认保存在：

```text
src/pingpong_moz1/pingpong_controller/outputs/fixed_ball_calibration/
fixed_ball_calibration_YYYYMMDD_HHMMSS/
```

主要文件：

- `metadata.json`：相机、检测器、坐标变换和安全配置；
- `samples.jsonl`：逐相机帧的球观测、深度诊断、JointState、球拍 FK；
- `pose_XX/summary.json`：每个姿态的稳健统计；
- `summary.json`：姿态覆盖范围和整体质量提示；
- `pose_XX/frame_*`：稀疏保存的 RGB、IR、16-bit depth 和标注图。

分析时以 `racketFrameOffsetMm=[u,v,n]` 为主。其中 `u/v` 是拍面内偏差，`n` 是球拍
法向距离；`surfaceNormalGapMm` 已扣除仿真球拍半厚度，但尚未扣除 20 mm 球半径。
