# Drag-ball RMP V2 最终三轨迹对比

所有图片同时包含实物、记录中的原始 sim mirror、V2 调参后独立 sim mirror。调参后镜像保持相同初始状态、相同 drag-ball 输入和相同 actor，仅替换 recovered-RMP/plant 参数；曲线均为无 marker 的细实线。

## 最终参数要点

- target update servo substep index：`4`（每个 5 ms 控制周期含 5 个 1 ms substep）。
- output delay J1–J7：`[12, 11, 10, 12, 13, 12, 13] ms`。
- joint acceleration feedforward：`0.05`。
- J4 acceleration limit scale：`0.8`。
- low-level qdd feedforward：`[2.5e-05, -5e-05, 0.0, 2.5e-05, -2.5e-05, -2.5e-05, -2.5e-05]`。
- jnt_kp/jnt_kd、filter、estimator、damping、armature 保持 checkpoint 原值。

## 相对原始 sim mirror 的闭环误差

| 时间窗 | 信号 | 调参前 RMS | V2 调参后 RMS | 改善 |
|---|---|---:|---:|---:|
| full | 关节角 | 0.047440 rad | 0.021585 rad | 54.5% |
| full | 关节角速度 | 1.132231 rad/s | 0.454454 rad/s | 59.9% |
| full | 球拍位置 | 0.026862 m | 0.010701 m | 60.2% |
| full | 球拍速度 | 0.679084 m/s | 0.260172 m/s | 61.7% |
| post_warmup | 关节角 | 0.048014 rad | 0.021292 rad | 55.7% |
| post_warmup | 关节角速度 | 1.155064 rad/s | 0.463754 rad/s | 59.9% |
| post_warmup | 球拍位置 | 0.027358 m | 0.010793 m | 60.5% |
| post_warmup | 球拍速度 | 0.692860 m/s | 0.265574 m/s | 61.7% |
| held_out_14_s_to_end | 关节角 | 0.044523 rad | 0.032955 rad | 26.0% |
| held_out_14_s_to_end | 关节角速度 | 0.981155 rad/s | 0.784165 rad/s | 20.1% |
| held_out_14_s_to_end | 球拍位置 | 0.023890 m | 0.017857 m | 25.3% |
| held_out_14_s_to_end | 球拍速度 | 0.586191 m/s | 0.445392 m/s | 24.0% |

## 相对上一版调参

- 上一版 post-warmup：q=`0.047804 rad`，dq=`1.098574 rad/s`。
- V2 post-warmup：q=`0.021292 rad`，dq=`0.463754 rad/s`。
- V2 未参与选择的 14 s+ 留出段：q=`0.032955 rad`，dq=`0.784165 rad/s`。

## 固定同 q_ref 的执行器 replay 复核

- 最终闭环候选 post-warmup：q=`0.002613 rad`，dq=`0.056500 rad/s`。
- 最终闭环候选 14 s+：q=`0.002550 rad`，dq=`0.058131 rad/s`。
- 该指标略差于上一版同 q_ref 最优值，说明最终候选优化的是策略闭环整体相位，而不是单独执行器 replay 的唯一最优点；两者不能混为同一目标。

## 解释限制

V2 已显著缩小闭环相位分岔，但不会完全重合。实物和独立 sim mirror 使用各自的关节状态运行 actor；任何小状态误差都会改变下一拍 action/q_ref，随后累积为闭环轨迹差异。用单条记录继续直接最小化全段误差会把 RMP 参数拟合成该条策略轨迹的补偿器，而不再是可泛化的执行器模型。因此参数选择没有使用 14 s+ 留出段。

## 图片

- `01–03`：关节角、角速度、角加速度。
- `04`：策略 qcmd 和 actor action。
- `05–10`：球拍位置、速度、加速度、Euler 角、角速度、角加速度。
- `11`：调参前后闭环误差随时间变化。
