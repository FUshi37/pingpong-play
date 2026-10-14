# 球观测/预测三天离线实验结论（2026-08-13）

## 数据范围与有效阶段

- 数据日期：2026-08-10、2026-08-11、2026-08-12。
- 顶层记录共 68 段；`record_new3`/`record_new4` 是镜像副本，没有重复计入。
- 66 段进入实验，累计有效颠球阶段 151.447735 s、13,574 个新相机样本、每个策略分支 29,565 个 200 Hz tick。
- `probe_recording_20260810_205409` 没有确认到颠球触球，`probe_recording_20260811_205641` 没有策略执行阶段，因此排除。
- 有效阶段从首个 `policyExecuted=true` 开始，在检测到最后丢球时截断，并无条件去掉记录尾部 120 ms；只有至少存在一次确认触球的阶段才纳入。

## 策略 raw/clean 仿真重放

两条分支使用相同的 simcode5 checkpoint、MuJoCo 纯执行器模型、初始关节角、控制 tick 和有效阶段。raw 分支使用记录中策略实际收到的位置/速度/valid；clean 分支使用离线异常值剔除、插值和局部鲁棒多项式平滑后的轨迹。球轨迹在仿真中被强制写入，因此这里衡量的是策略对观测的敏感性，不是闭环实物成功率预测。

在相机频率数据中，共替换 178 个数值异常点、插值 355 个缺失或异常点。策略频率上有 2,248/29,565 个 tick 满足缺失、位置偏差超过 35 mm 或速度偏差超过 1.5 m/s。raw 与 clean 输入的总体位置差 p95 为 17.878 mm，速度差 p95 为 1.765 m/s；最大值分别达到 1.597 m 和 62.114 m/s，证明原始记录中确实存在严重异常值。

| 指标 | 结果 |
|---|---:|
| raw/clean 策略动作差 RMS / p95 | 0.722 / 1.484 |
| 命令角差 RMS / p95 | 0.201 / 0.410 rad |
| 球拍位置差 RMS / p95 / max | 36.61 / 78.96 / 270.73 mm |
| 异常后 0–100 ms 动作差 p95 | 1.804 |
| 非异常窗口动作差 p95 | 1.182 |
| 异常后 0–100 ms 球拍差 p95 | 113.02 mm |
| 非异常窗口球拍差 p95 | 65.09 mm |

异常暴露窗口的动作差 p95 比基线高 52.7%，球拍轨迹差 p95 高 73.6%，所以异常值对策略行为的影响很大。clean 分支把动作变化率 p95 从 93.83 降到 76.24（下降 18.7%），但命令角变化率没有同步下降。

接球几何的结果是混合的：下降段最近平面处的 XY 误差中位数由 16.49 mm 改善到 14.78 mm，但 p95 由 89.18 mm 变差到 98.71 mm，100 mm 几何阈值计数由 297/321 变为 287/321。因此不能把离线 acausal 平滑轨迹直接当作线上观测：它让典型动作更平滑，但没有在所有长尾下降段改善策略表现。策略已经适应了原检测分布，线上应使用因果的异常门控和分段速度模型，而不是直接复用 clean 参考滤波器。

## 检测链路诊断和全量对比

主要根因是：

1. LS velocity 关闭时，旧 3 帧端点差分会放大深度和质心噪声；位置只有毫米级扰动，也会变成高频速度噪声。
2. 触球前后的样本进入同一个拟合窗口，会把真实速度反转拟合成异常尖峰。
3. 丢检/重捕获时若保留旧历史，会混合两个飞行段并产生位置和速度跳变。
4. 无运动一致性门控时，错误深度或错误 3D 候选可以直接进入位置和速度历史。
5. 鲁棒拟合的端点修正若在相邻帧反向摆动，会在原始位移上叠加额外位置阶跃。

最终链路保持同步 IR+depth、LS velocity、high-ball reacquisition velocity handover 和 90→200 control-time compensation 全部关闭；启用 loss-boundary clear、ballistic motion gate、5 帧因果物理速度、12 mm 有界端点位置平滑、两帧确认触球 reset，并对最终发布位置增加 6 m/s 的因果步长上限。

| 链路 | 位置误差 p95 中位数 | 速度误差 p95 中位数 | 位置 HF p95 | 速度 HF p95 | 非触球位置异常 | 非触球速度异常 |
|---|---:|---:|---:|---:|---:|---:|
| 记录中的策略输入 | 12.74 mm | 0.558 m/s | 41.97 mm | 0.317 m/s | 61 | 33 |
| 旧 3 帧端点速度 | 17.49 mm | 0.451 m/s | 25.51 mm | 0.351 m/s | 11 | 36 |
| 触球 reset + 端点速度 | 16.65 mm | 0.394 m/s | 24.65 mm | 0.313 m/s | 7 | 16 |
| 物理速度 + motion gate | 15.69 mm | 0.405 m/s | 23.52 mm | 0.308 m/s | 0 | 0 |
| 最终推荐链路 | 15.88 mm | 0.405 m/s | 22.92 mm | 0.308 m/s | 0 | 0 |

相对记录中的策略输入，最终链路的位置高频项下降 45.4%，速度 p95 误差下降 27.5%，非触球位置/速度异常计数均变为 0。代价是相对 acausal 参考的位置 p95 误差增加约 3.1 mm；这是用很小的空间误差换取明显的连续性和速度稳定性。

另外测试了 20 mm 端点修正、固定速度增益 0.55 和残差自适应增益。20 mm 修正没有降低高频项；0.55 和自适应增益都使速度 p95 误差变差。因此最终保留固定 0.65 增益，自适应增益默认关闭。

位置异常定义为：排除触球窗口后，中间点相对相邻两点时间插值残差大于 35 mm，且相邻速度超过 6 m/s。速度异常定义为：排除触球窗口并补偿重力后，相邻速度变化超过 2 m/s。HF 和误差均以同一段离线 acausal 鲁棒轨迹为参考，不宣称是真实运动捕捉 ground truth。

## 代码和启动配置

- `RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh` 现在默认启用已验证的因果稳定链路，同时保持用户指定的四个开关关闭。`camera reacquisition smoothing` 与 `high-ball reacquisition velocity handover` 是不同开关；后者仍为 false。
- `HybridBallDetector` 的发布位置增加最终步长限制；实验未支持的自适应速度增益默认关闭。
- 未来记录会把 clear-loss 和 physics-velocity 状态写入压缩后的 camera telemetry，避免只能从 `velocity_model` 反推实际链路。
- 双分支策略重放：`offline_policy_trajectory_replay.py`。
- 检测状态链路全量重放：`offline_ball_trajectory_study.py`。

复现实验：

```bash
PYTHONPATH=src/pingpong_moz1 python3 \
  src/pingpong_moz1/pingpong_controller/tools/offline_ball_trajectory_study.py \
  --output analysis_results/ball_trajectory_study_20260810_12.json

PYTHONPATH=src/pingpong_moz1 python3 \
  src/pingpong_moz1/pingpong_controller/tools/offline_policy_trajectory_replay.py \
  --output analysis_results/policy_trajectory_replay_20260810_12.json
```

## 原始数据能力边界

68 段记录都包含策略 tick、相机时间戳、门控前变换后的 3D 候选、检测决策、测量/策略 6D 球状态和球拍 FK，因此本次下游检测状态机和策略重放是确定性的。重复输入 100 个 tick 的仿真检查中，动作和球拍轨迹的最大重复误差均为 0。

但目录中没有 `.npy`/`.npz` 的原始 IR/16-bit depth，也没有 RealSense `.bag`/ROS `.db3`；`frames/*.jpg` 是有标注的有损复合图。因此本次不能从像素层重新运行轮廓提取、同步深度采样或新的深度算法。若要让以后所有像素级新功能也能完全离线复现，记录器还需要保存原始 IR、原始 16-bit depth 和各自硬件时间戳（优先保存 bag 或无损数组）。
