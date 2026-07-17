# Observation Sensitivity and Sim2Real Gap Analysis

## 概述

这套工具用于量化分析每个 observation 维度对策略输出的影响，以及对 sim2real gap 的贡献程度。

## 分析方法

### 方法1: 观测误差统计
- 计算每个 obs 维度的 mean/std/max 误差
- 识别误差最大的维度

### 方法2: 相关性分析
- 计算每个 obs 维度误差与 action 误差的相关系数
- 识别哪些 obs 误差与 action 误差强相关

### 方法3: 贡献度分析
- 贡献度 = |相关系数| × 观测误差
- 量化每个 obs 维度对总体 action 误差的贡献百分比
- 归一化后总和为 100%

### 方法4: 时间演化分析
- 分析贡献度随时间的变化
- 识别稳定的和不稳定的贡献源

## 使用方法

### 快速开始

```bash
# 使用简化版本（推荐）
python3 analyze_obs_impact_simple.py \
    <real_robot_trace.csv> \
    <simulation_trace.csv> \
    [output_directory]
```

### 示例：分析仿真 replay 数据

```bash
# 当前的仿真 replay 对比参考轨迹
python3 analyze_obs_impact_simple.py \
    src/pingpong_moz1/pingpong_controller/outputs/rl_sim/simulation_replay/policy_obs_trace.csv \
    src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_obs_trace.csv \
    obs_analysis_replay
```

### 示例：分析实物机器人数据

```bash
# 实物颠球对比仿真参考
python3 analyze_obs_impact_simple.py \
    <path_to_real_robot_trace.csv> \
    src/pingpong_moz1/pingpong_controller/data/rl_sim/validate_obs_trace/validate_obs_trace.csv \
    obs_analysis_real
```

## 输出文件

分析完成后会生成：

1. **analysis_report.txt**
   - 文本报告，包含 Top-K 贡献维度排名
   - 累积贡献百分比

2. **analysis_results.json**
   - 完整的数值结果（JSON格式）
   - 可用于进一步处理

3. **obs_impact_analysis.png**
   - 3个子图：
     - 观测误差分布
     - 与 action 误差的相关性
     - 贡献度排名

4. **top_obs_details.png**
   - Top 6 贡献维度的时间序列
   - 双 y 轴显示 obs 误差和 action 误差

5. **temporal_evolution.png**
   - 贡献度随时间演化的热图
   - 识别动态变化的贡献源

## 解释结果

### 贡献度百分比

- **高贡献 (>5%)**: 关键维度，优先优化
- **中等贡献 (1-5%)**: 重要维度，次优先
- **低贡献 (<1%)**: 次要影响

### Top-10 累积贡献

- 如果 Top-10 累积 >80%: gap 来源集中，容易定位
- 如果 Top-10 累积 <50%: gap 来源分散，系统性问题

### 相关系数

- **正相关 (>0.5)**: obs 增大导致 action 误差增大
- **负相关 (<-0.5)**: obs 减小导致 action 误差增大
- **弱相关 (|corr|<0.3)**: obs 与 action 误差关系不强

### 时间稳定性

- **稳定贡献**: 整个轨迹中持续存在的问题
- **不稳定贡献**: 特定阶段出现的问题（如击球瞬间）

## 典型分析流程

### 步骤1: 运行分析

```bash
python3 analyze_obs_impact_simple.py \
    real_trace.csv sim_trace.csv output_dir
```

### 步骤2: 查看报告

```bash
cat output_dir/analysis_report.txt
```

重点关注：
- Top-10 贡献维度是什么？
- 累积贡献占比多少？

### 步骤3: 查看可视化

```bash
# 在浏览器中打开或用图片查看器
open output_dir/obs_impact_analysis.png
open output_dir/top_obs_details.png
```

### 步骤4: 根据结果优化

根据 Top 贡献维度类型采取行动：

| Obs 类型 | 可能原因 | 优化方向 |
|---------|---------|---------|
| ball_pos (obs 010-012) | 相机标定/坐标转换 | 检查 T_base_camera, sim_base_height |
| ball_vel (obs 013-015) | 速度估计/滤波 | 检查 Kalman 滤波器参数 |
| arm_q (obs 000-006) | 关节编码器噪声 | 检查关节反馈质量 |
| arm_dq (obs 007-013) | 速度估计方法 | 有限差分 vs JointState.velocity |
| ball_age (obs 066) | 检测延迟/丢失 | 检查相机帧率、检测率 |

## 高级用法

### 自定义分析窗口

编辑脚本中的参数：

```python
window_size = 50  # 改为更小/更大的窗口
top_k = 10        # 改为显示更多/更少的维度
```

### 导出数据用于其他分析

```python
import json
with open('obs_analysis/analysis_results.json') as f:
    results = json.load(f)

contribution = results['contribution']
top_indices = results['top_contributors']['indices']

# 进一步处理...
```

### 对比多次实验

```bash
# 实验1
python3 analyze_obs_impact_simple.py trace1_real.csv trace1_sim.csv exp1

# 实验2  
python3 analyze_obs_impact_simple.py trace2_real.csv trace2_sim.csv exp2

# 对比 Top-10 是否一致
diff exp1/analysis_report.txt exp2/analysis_report.txt
```

## 注意事项

1. **数据对齐**: 确保 real 和 sim trace 的时间对齐
   - 使用相同的起始状态
   - 相同的控制频率 (200Hz)

2. **轨迹长度**: 建议至少 1000 步 (5秒)
   - 太短可能统计不稳定
   - 太长分析时间会增加

3. **obs 命名**: 需要从 obs/000 到 obs/066 的列
   - 检查 CSV 列名格式

4. **异常值**: 如果某些 obs 维度始终为 0
   - 可能是未使用的维度
   - 或者数据记录有问题

## 示例输出

```
Top 10 Observations Contributing to Sim2Real Gap:
------------------------------------------------------------------
 1. obs/010                :  15.23% (cumulative:  15.2%)
     obs_error=0.024531, corr=+0.8234
 2. obs/007                :  12.87% (cumulative:  28.1%)
     obs_error=0.018234, corr=+0.7456
 3. obs/013                :   8.45% (cumulative:  36.6%)
     obs_error=0.012456, corr=+0.6123
 ...
10. obs/003                :   2.34% (cumulative:  82.3%)
     obs_error=0.003421, corr=+0.4567
```

解读：
- obs/010 (可能是 ball_pos_z) 贡献最大 (15.23%)
- obs/007 (可能是 arm_dq[0]) 次之 (12.87%)
- Top-10 累积贡献 82.3% → gap 来源集中

## 常见问题

**Q: 为什么有些 obs 相关系数为负？**

A: 负相关表示该 obs 减小时 action 误差增大，或增大时误差减小。关注绝对值。

**Q: 贡献度加起来不是100%怎么办？**

A: 脚本会自动归一化。如果原始数据异常，检查 CSV 文件完整性。

**Q: 时间演化图显示所有维度都不稳定？**

A: 可能轨迹本身高度动态（如击球瞬间）。尝试分段分析。

**Q: 如何确定具体是哪个物理量？**

A: 查看训练代码或 rl_policy.py 中的 observation space 定义。

## 参考

- 相关代码: `src/pingpong_moz1/pingpong_controller/rl_policy.py`
- 训练环境: `src/pingpong_moz1/pingpong_controller/tools/rl_sim/mjx_juggle_env.py`
- Observation 定义: 检查 `_build_observation()` 方法

