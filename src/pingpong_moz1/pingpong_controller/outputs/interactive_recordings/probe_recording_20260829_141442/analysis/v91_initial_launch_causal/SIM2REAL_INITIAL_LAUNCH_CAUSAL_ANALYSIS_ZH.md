# probe_recording_20260829_141442 初始起球因果分析

## 结论先行

这次实物起球的平滑球心峰值为 **1.1558 m**（首个 policy tick 后 0.255 s），相对起始观测球心上升 **87.7 mm**。判因结果不是“某一个碰撞恢复系数直接导致低起球”，而是一个更靠前的状态合同问题：**实物放拍状态、相机给 actor 的状态、训练/默认 reset 状态和原始 mirror 的碰撞几何不是同一个初态。**

优先级判断：

1. **首要：初态/观测几何合同不一致。** 相机初态的球心法向距离只有 8.4 mm，小于 20 mm 球半径，几何上不可能；记录默认 reset 是 30.7 mm。actor 因而看到的 `ball-racket rel_z` 比默认状态少约 22.3 mm。
2. **首要并列：原始 mirror 不能独立验证起球碰撞。** 它每 tick 强制写回实物球状态；同时 actor FK 与实际碰撞拍面 z 平均相差 10.7 mm。
3. **执行器不是“实物起得低”的单独主因。** 相同 `publishedSafePosition` 回放下，冻结的历史 RMP 反而比实物更慢/更低：仿真拍面峰值比实物低 21.8 mm。这说明不能用该冷启动 mirror 回放声称实物执行器力度不足。
4. **反馈 hold 不是主因。** 起球至峰值间 policy 实际执行率 80.8%，但保留真实 hold 掩码的反事实没有稳定降低峰值；hold 改变闭环相位，影响是非单调的。
5. **碰撞参数仍然重要，但本段不能单独辨识。** 4.0 g 与 2.7 g、初始间隙以及 `solref` 的耦合可改变峰值几十毫米；而只用冲击反弹拟合出的低 damping 参数，在静态放拍实验中产生非实物能量，不能直接用于起球准静态接触。

因此，如果问题表述为“为什么实物首起高度低于期望”，最可信的回答是：**策略从一个训练中不存在、且相机几何不自洽的放拍观测状态启动；原始 mirror 又没有真正模拟这个接触。执行器与 hold 不是现有证据下的第一嫌疑，碰撞参数必须在修正初态/拍面坐标并加入静态预接触状态后再标定。**

## 冻结的历史仿真合同

- V91 checkpoint SHA-256：`92b5d39592a81236ee5ee5806499a0a8d4b422f131b312ad87ea1efc90507c66`。
- 记录时 policy XML SHA-256：`7d98f2adfdbad6082be0defcec2dbd0cbbcaf1f0fc06ce45ba424b5b3257cc92`；脚本使用仓库内仍保留该精确哈希的副本，没有使用当前 simcode7 XML。
- control dt = 5 ms；MuJoCo = 1 ms × 5；RMP target-update delay = 3 step。
- RMP output delay：`[18.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0]`。
- 历史 nominal 接触：球 2.7 g，`solref=0.003 0.80`。按你的约束，主反事实把球质量改为 **4.0 g**；2.7 g 仅保留为记录时复现基线。
- 原始在线 backend 是 `mujoco_recovered_rmp_mirror`，不是独立 MJX 球物理；本报告用同一 checkpoint 内嵌 RMP 参数做 CPU MuJoCo 独立球物理实验。
- 独立闭环采用确定性的 90 Hz sample-and-hold 真值观测，不开启随机 DR；因此它是单变量判因实验，不是随机训练分布的统计评估。

## 实物起球数据质量

- 分析窗口：sourceTime 17.625–17.880 s。
- 使用 27 个 cameraStamp 去重后的连续位置样本；最大相邻帧间隔 15.0 ms。
- 球速度没有采用记录中的逐帧 `measuredVelocity` 作主证据，而是由连续位置做 80 ms 鲁棒局部二次拟合求导；位置拟合 RMS 为 1.7 mm。
- 平滑峰值：1.1558 m；原始位置峰值：1.1569 m；二者差 1.1 mm。
- 实物拍面最高 1.1096 m，最大向上速度 0.758 m/s。

灰色区是 q/dq feedback hold：[`01_real_initial_launch.png`](01_real_initial_launch.png)。

## 初态：手上“同一个球”在四个合同里不是同一个状态

| 状态 | 球心/拍面法向距离 | 说明 |
|---|---:|---|
| 相机稳定初态 | 8.4 mm | 小于球半径，不可作为真实碰撞 qpos |
| 几何约束的实物放拍 | 22.0 mm | 20 mm 半径 + 2 mm site→表面 |
| 记录默认 reset | 30.7 mm | 含 8.7 mm 自由间隙 |

实物初始关节角与 checkpoint 默认关节角 RMS 只有 0.000000 rad，说明**臂姿态本身不是主要初态差异，球相对拍面的状态才是。** 详见 [`02_initial_geometry_contract.png`](02_initial_geometry_contract.png)。

## 原始 mirror 的能力边界

记录中的 49 个原始 mirror 样本均由实物 policy ball 输入驱动。代码路径在每个 MuJoCo step 前后调用 `_set_ball_state`，所以：

- 原始文件能回答 q/dq、RMP 延迟和 actor 输出差异；
- 不能回答球是否被拍面弹起、恢复系数是否正确；
- 同一 tick 的 actor 观测拍面与碰撞拍面平均差为 `[3.7, 7.4, 10.7] mm`。

这是本次必须另跑独立球物理的原因。

## 执行器/RMP 回放

对每个实物 tick 使用真正发布的 `command.publishedSafePosition`（feedback hold 时它保留上次安全目标，不能误用诊断字段 `policyCommand` 的当前 q）：

- q RMS：0.0324 rad；dq RMS：0.526 rad/s。
- 拍面位置 RMS：8.8 mm；z RMS：13.6 mm。
- 实物/冻结 RMP 回放拍面峰值：1.1096 / 1.0878 m。

历史 mirror RMP 对同一发布目标的响应比实物低，而不是更高。这个方向与“实物执行器太弱导致低起球”不一致。详见 [`03_published_command_rmp_replay.png`](03_published_command_rmp_replay.png)。

## 独立球物理反事实

| 仿真 | 初态 | 质量 | 接触 damping | 峰值 z | 相对实物 |
|---|---|---:|---:|---:|---:|
| 同相机初态 / 4.0g / 历史接触 | `observed` | 4.0 g | 0.800000 | 1.1437 m | -12.2 mm |
| 同物理放拍初态 / 4.0g / 历史接触 | `physical_on_racket` | 4.0 g | 0.800000 | 1.1362 m | -19.6 mm |
| 同物理放拍初态 / 历史2.7g | `physical_on_racket` | 2.7 g | 0.800000 | 1.1966 m | +40.8 mm |
| 碰撞拍面默认初态 / 4.0g / 历史接触 | `physical_default` | 4.0 g | 0.800000 | 1.1270 m | -28.9 mm |
| 记录 mirror 默认坐标 / 4.0g / 历史接触 | `mirror_default` | 4.0 g | 0.800000 | 1.1373 m | -18.6 mm |
| 碰撞拍面默认初态 / 历史2.7g | `physical_default` | 2.7 g | 0.800000 | 1.1267 m | -29.2 mm |
| 同物理初态 / truth / 实物hold掩码 | `physical_on_racket` | 4.0 g | 0.800000 | 1.1645 m | +8.6 mm |
| 同物理初态 / 实物观测 / 实物hold掩码 | `physical_on_racket` | 4.0 g | 0.800000 | 1.2084 m | +52.5 mm |
| 同物理初态 / 4.0g / 冲击辨识接触 | `physical_on_racket` | 4.0 g | 0.152540 | 1.2385 m | +82.7 mm |


读表时要注意两个非线性结果：

- “相机观测坐标直接当物理 qpos”的 4.0 g 仿真峰值为 1.1437 m，接近实物，但它从球拍穿透/接触边界启动，**数值接近不等于物理正确**。
- 用碰撞拍面构造真正的 4.0 g 放拍状态后，历史接触峰值为 1.1362 m；用真正的碰撞拍面构造默认 gap 后为 1.1270 m。结果对 gap、质量和接触状态高度敏感。
- 在相同物理放拍初态与相同历史接触下，2.7 g / 4.0 g 峰值为 1.1966 / 1.1362 m；这正是为什么主结论必须固定你给出的 4.0 g，不能沿用记录时 2.7 g。
- 喂入实物记录观测并保留真实 hold 掩码的 hybrid 仿真峰值为 1.2084 m，高于 perfect-truth/hold-mask 的 1.1645 m。就本段符号而言，观测没有让 actor 更保守；它让闭环更激进，因此“观测导致命令太小”不成立。

轨迹见 [`04_initial_state_simulation_trajectories.png`](04_initial_state_simulation_trajectories.png)，峰值对比见 [`05_apex_counterfactuals.png`](05_apex_counterfactuals.png)。

## 静态放拍检查：冲击拟合不能直接外推

从记录默认 gap 开始、拍面保持 q0：

- 历史 `0.003/0.80` 下，球在 0.525 s 后横向离开有效拍面，说明该模型没有重现实物在起球前长时间静止于拍面的状态。
- 冲击数据拟合的 `0.003/0.152540` 在这个准静态场景中把球推到 2.289 m，出现明显非实物储能/释放。

这不否定冲击反弹辨识，而是说明它只约束了有限入射速度下的恢复行为；放拍启动还需要静态接触、滚动/旋转摩擦、初始 penetration/solver warm state 的约束。见 [`06_static_ball_on_racket_contact_test.png`](06_static_ball_on_racket_contact_test.png)。

## 对四个候选原因的最终判断

| 候选原因 | 判断 | 证据 |
|---|---|---|
| 初始状态 | **主要问题** | 球相对拍面相差约 22 mm；实物是预接触放拍，默认仿真是带 gap 的即时启动 |
| 执行器/RMP | **不是低高度主因，但 mirror 模型不够准** | 同命令回放的历史 RMP 比实物更低、更慢 |
| 球/球拍碰撞 | **重要且当前未被该场景可靠约束** | 质量/gap/solref 强耦合；低 damping 在静态启动中非物理 |
| 球观测 | **状态合同主因之一，但不是“命令变小”** | 初态几何不可行；recorded-observation hybrid 反而更高 |
| feedback hold | **次要/非单调** | 19.2% 起球 tick 未推理，但 hold 反事实未稳定降低峰值 |

## 建议的下一轮最小实验

1. 在仿真 reset 中增加 `ball_on_racket_settled` 模式：球心以碰撞拍面而非 actor FK 放置，保存 contact solver warm state，起球前至少静置 0.3 s 且横向漂移 < 2 mm。
2. 在线记录同时输出 `actor_racket_pose` 与 `collision_racket_pose`，把当前约 10.7 mm z 差消掉后再做碰撞辨识。
3. 实物起球前拍摄固定球标定：用球半径约束修正法向偏差；actor 可继续用平滑观测，但物理初始化不要直接使用相机球心 qpos。
4. 用 4.0 g 固定质量，分开标定：静态/低速承托参数与 0.5–2.0 m/s 冲击恢复参数；不要只靠一个 `solref damping` 同时拟合两种工况。
5. 修正初态合同后再比较 hold on/off；当前状态差异大于 hold 的可解释效应。

## 产物

- `summary.json`：全部数值结论与冻结合同。
- `real_launch_samples.csv`：cameraStamp 去重、平滑后的实物球/拍数据。
- `simulation_timeseries.csv`：所有独立物理反事实逐 tick 数据。
- `published_command_replay.csv`：相同发布目标的实物/历史 RMP 执行器对比。
- `static_hold_timeseries.csv`：两套接触参数的放拍静态检查。
