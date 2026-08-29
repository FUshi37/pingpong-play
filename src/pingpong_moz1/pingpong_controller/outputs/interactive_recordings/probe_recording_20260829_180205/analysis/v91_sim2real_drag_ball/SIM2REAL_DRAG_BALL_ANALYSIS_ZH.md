# V91 Drag Ball 策略与执行器 Sim2Real 分解分析

录制：`probe_recording_20260829_180205`

## 核心结论

1. 本次策略阶段完整连续：3667 ticks、18.330 s、198.01 Hz；严格对齐 3667/3667 ticks，只有 1 个 episode、1 个连续段。
2. drag-ball 的位置、速度、valid、age、观测序号在 real/sim 两边逐 tick 完全一致。起始 q、dq、actor action 和 q_ref 也一致，所以不存在 actor 实现或输入传输的首 tick 差异。
3. 全程 live mirror 总 gap：q=0.04744 rad，dq=1.1322 rad/s；球拍位置=26.86 mm，球拍速度=0.6791 m/s。
4. 用实物实际发布的同一条 q_ref 驱动 recovered-RMP/MuJoCo；这个 plant-only replay 不运行 actor，并把未使用的球移出工作区以完全去除碰撞力。纯 plant/执行器 gap 为 q=0.00500 rad、dq=0.1183 rad/s、球拍位置=2.69 mm。保留虚拟球碰撞时同指令 gap 为 0.00500 rad。
5. 在线 mirror 的虚拟球从 0.150 s 起发生接触，22.50% 的 tick 在强制写球后存在接触；接触对象计数为 racket_rubber_fore(825)。但相同 q_ref 下，开/关虚拟球碰撞的机械臂响应差异为 q=0.00000093 rad、dq=0.00010984 rad/s，相对总 gap 可忽略；所以虚拟接触不是本次机械臂 gap 的主要来源。接触开启的共同重放与独立 sim 之间 q gap=0.04667 rad，反映状态分叉后的策略命令响应。
6. 前 0.5 s 内无碰撞 plant q gap=0.01651 rad，总 q gap=0.03360 rad。最早越过诊断阈值的顺序见下表；q/dq 分叉早于第一次虚拟球接触，且碰撞开关对机械臂轨迹的影响可忽略，说明执行器/plant mismatch 是起因之一，后续主要由 actor 闭环分叉放大。

## 数据与时序可信度

- 独立 sim 命令离线重放自检：q RMS=0.0000004 rad，dq RMS=0.000062 rad/s，结果 `passed=true`。这验证了 observe→apply 当前 q_ref 的 tick 相位。
- 初始状态：q exact=True，dq exact=True，首 tick action max gap=0，q_ref max gap=0 rad。
- real policy q_ref 与实际发布目标的 gap RMS=0.00000000 rad；因此 common replay 使用的是实物真正收到的目标，而不是未发布的中间量。
- 虚拟球接触审计：contact-enabled common replay 在强制写球后有 825/3667 ticks 接触；collisionless common replay 直接消费已记录 q_ref、不运行 actor，并将球移出工作区。
- 浏览器 analysis 的 `alignmentValid=false` 只来自实时异步完成延迟门限：P95=10.62 ms 略高于 10 ms；最终 flush、序号、输入和 3667 个样本均完整，所以离线同 tick 分析有效。

## 闭环分叉起点

阈值只是为了稳定定位先后顺序，不代表安全阈值。

| 信号 | 阈值 | 首次越过 | 当时值 |
| --- | ---: | ---: | ---: |
| 关节角 RMS | 0.005 rad | 0.085 s | 0.00576 rad |
| 关节角速度 RMS | 0.1 rad/s | 0.065 s | 0.12253 rad/s |
| 策略 action RMS | 0.05  | 0.090 s | 0.06040 |
| 策略 q_ref RMS | 0.005 rad | 0.085 s | 0.00572 rad |
| 球拍位置 RMS | 0.002 m | 0.100 s | 0.00226 m |

## 全量参数总 gap（live real ↔ independent sim）

| 参数 | RMS gap | 样本数 |
| --- | ---: | ---: |
| 关节角 | 0.04744 rad | 3667 |
| 关节角速度 | 1.13223 rad/s | 3667 |
| 关节角加速度 | 30.05929 rad/s² | 3602 |
| 策略 q_ref | 0.04601 rad | 3667 |
| 策略 action | 0.44777  | 3667 |
| 球拍三维位置 | 0.02686 m | 3667 |
| 球拍三维速度 | 0.67909 m/s | 3667 |
| 球拍三维加速度 | 18.06724 m/s² | 3602 |
| 球拍 XYZ Euler 角 | 3.11905 deg | 3667 |
| 球拍三维角速度 | 1.30950 rad/s | 3667 |
| 球拍三维角加速度 | 35.40273 rad/s² | 3602 |

## 执行器/plant 分解

| 时间窗 | real↔无碰撞 common q | real↔无碰撞 common dq | 接触 common↔independent q | total q |
| --- | ---: | ---: | ---: | ---: |
| 前 0.1 s | 0.00406 rad | 0.1913 rad/s | 0.00001 rad | 0.00406 rad |
| 前 0.25 s | 0.02228 rad | 0.3269 rad/s | 0.00769 rad | 0.02449 rad |
| 前 0.5 s | 0.01651 rad | 0.2594 rad/s | 0.03072 rad | 0.03360 rad |
| 前 1.0 s | 0.01212 rad | 0.1992 rad/s | 0.03308 rad | 0.03606 rad |
| 全程 | 0.00500 rad | 0.1183 rad/s | 0.04667 rad | 0.04744 rad |

### 逐关节 RMS

| 关节 | 同 q_ref plant q | 同 q_ref plant dq | live total q | live total dq |
| --- | ---: | ---: | ---: | ---: |
| J1 | 0.00337 rad | 0.0750 rad/s | 0.03774 rad | 0.8930 rad/s |
| J2 | 0.00320 rad | 0.0590 rad/s | 0.02669 rad | 0.4587 rad/s |
| J3 | 0.00684 rad | 0.1681 rad/s | 0.05318 rad | 1.2184 rad/s |
| J4 | 0.00392 rad | 0.0975 rad/s | 0.04650 rad | 1.1802 rad/s |
| J5 | 0.00500 rad | 0.1217 rad/s | 0.04883 rad | 1.1900 rad/s |
| J6 | 0.00542 rad | 0.1395 rad/s | 0.06002 rad | 1.4597 rad/s |
| J7 | 0.00609 rad | 0.1296 rad/s | 0.05138 rad | 1.2416 rad/s |

同 q_ref 的 plant gap 中 J3 最大（q=0.00684 rad、dq=0.1681 rad/s）；闭环总 gap 中 J6 最大（q=0.06002 rad、dq=1.4597 rad/s）。

同一实物 q_ref 下，实物 tracking-error RMS=0.06008 rad，无碰撞 common replay=0.05945 rad，接触开启 common replay=0.05945 rad。
raw action 任一关节越过 [-1,1] 的比例：real=2.10%、sim=1.66%；实际发生 clipping 的比例：real=2.10%、sim=1.66%。

## 图表

- `01–03`：7 关节角、角速度、角加速度的 live real/sim 对比。
- `04`：策略 q_ref 与 action；`05–10`：球拍三维位置/速度/加速度/姿态/角速度/角加速度。
- `11_gap_summary.png`：live 总 gap；`12_drag_ball_input.png`：drag-ball 输入逐 tick 相等。
- `13_common_command_joint_position.png` / `14_common_command_joint_velocity.png`：real、无碰撞共同命令重放、独立 sim 三路对比。
- `15_common_command_tracking_error.png`：同一实物 q_ref 下 real 与无碰撞 replay 的 tracking error。
- `16_gap_decomposition.png`：总 gap、无碰撞 plant gap、虚拟球接触效应和闭环命令响应的时序分解。
- `17_early_divergence.png`：首 1 s 分叉顺序。

逐 tick live 数据见 `aligned_sim2real.csv`；共同命令重放见 `actuator_common_command_replay.csv`；所有逐关节/逐轴指标见 `summary.json`。

## 解释限制

- drag-ball 只保证外部球输入相同，并不保证 actor 的完整输入始终相同；q/dq 一旦因 plant 差异分叉，同一个 actor 就会输出不同 action。这是闭环 gap，不是两份策略代码不同。
- collisionless common replay 使用 checkpoint-authoritative recovered-RMP/MuJoCo plant 和记录的实际发布 q_ref；real↔common 的残差包含真实机器人动力学、摩擦、时延、RMP 实现/参数以及未记录的 1 kHz 内部状态差异。
- 在线 mirror 中 drag-ball 仍是有质量且可碰撞的 MuJoCo 球；本次开/关碰撞没有造成可测的机械臂轨迹差异。如果 drag-ball 本意只是策略观测，仍建议运行时代码显式设为 observation-only，避免以后模型质量/接触参数变化后引入隐式动力学耦合。
- 三路响应属于非线性反馈系统，real↔sim 不能被简单写成两个 RMS 的和；应结合早期窗口与时序曲线判断因果。
