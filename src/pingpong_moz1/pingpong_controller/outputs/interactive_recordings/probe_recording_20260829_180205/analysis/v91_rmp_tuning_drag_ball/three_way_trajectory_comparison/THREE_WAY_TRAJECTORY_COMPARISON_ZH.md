# RMP 调参前后三轨迹对比

每幅轨迹图同时包含实物、记录中的原始 sim mirror，以及仅替换 RMP 参数后重新运行的独立 sim mirror。后者使用相同初始状态、相同 drag-ball 输入和相同策略。所有曲线均为无 marker 的细实线。

## 重放语义

- `before`：记录文件里的原始在线 sim mirror。
- `after`：从记录的首个 sim tick-start 状态启动，逐 tick 输入相同 drag-ball 观测，运行相同 actor，仅替换拟合后的 recovered-RMP 参数。
- `fresh baseline reproduction`：同样重新运行，但不改 RMP，用于评估仅凭六位小数记录状态能否重建原闭环。

## Fresh baseline 对记录 sim mirror 的重建误差

- q RMS：`0.00000052 rad`
- dq RMS：`0.0000072 rad/s`
- qcmd RMS：`0.00000063 rad`
- action RMS：`0.00000477`

## 闭环 Sim2Real 误差

| 时间窗 | 信号 | 调参前 RMS | 调参后 RMS | 相对变化（负值=改善） |
|---|---|---:|---:|---:|
| full | 关节角 | 0.047440 rad | 0.046931 rad | -1.1% |
| full | 关节角速度 | 1.132231 rad/s | 1.070839 rad/s | -5.4% |
| full | 球拍位置 | 0.026862 m | 0.025397 m | -5.5% |
| full | 球拍速度 | 0.679089 m/s | 0.651568 m/s | -4.1% |
| post_warmup | 关节角 | 0.048014 rad | 0.047804 rad | -0.4% |
| post_warmup | 关节角速度 | 1.155064 rad/s | 1.098574 rad/s | -4.9% |
| post_warmup | 球拍位置 | 0.027359 m | 0.025994 m | -5.0% |
| post_warmup | 球拍速度 | 0.692866 m/s | 0.668512 m/s | -3.5% |
| held_out_14_s_to_end | 关节角 | 0.044523 rad | 0.048114 rad | +8.1% |
| held_out_14_s_to_end | 关节角速度 | 0.981155 rad/s | 0.949502 rad/s | -3.2% |
| held_out_14_s_to_end | 球拍位置 | 0.023892 m | 0.023365 m | -2.2% |
| held_out_14_s_to_end | 球拍速度 | 0.586222 m/s | 0.554445 m/s | -5.4% |

注意：这是把 RMP 参数直接放回策略闭环后的结果；它与同 q_ref 执行器 replay 指标回答的问题不同。参数变化会改变关节状态，随后 actor 的 q_ref 也会发生分岔。

## 图片

- `01–03`：关节角、角速度、角加速度。
- `04`：策略 qcmd 和 actor action。
- `05–10`：球拍三维位置、速度、加速度、Euler 角、角速度、角加速度。
- `11`：调参前后闭环误差随时间的变化。
