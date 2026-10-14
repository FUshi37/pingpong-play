# ideal-PD67 实物部署与仿真对照指南

本文说明 `ideal-PD67` launch19 策略的训练语义、67D 观测、实物控制链、
原版 inverse-MPC 补偿参数，以及如何使用 `train_juggle_mjx_curriculum.py`
和 `validate_juggle_mjx_ppo.py` 对实物实现做一致性检查。

本文的目标不是在实物端复刻训练器，而是明确哪些仿真机制必须保留、
哪些只属于训练、哪些由真实驱动器代替。

## 1. 部署模型

当前选定模型是 ideal-PD67 view-dense 分支在最终课程
`launch19_final_consolidation` 的 update 2040 冻结 checkpoint：

```text
/home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/
goal_d455_autolaunch_idealpd67_viewdense_v1_seed976_gpu1_resume_launch14_best55_20260722/
launch19_best2040_video_gpu1_seed20260722/policy_launch19_best_u2040_frozen.pkl
```

模型信息：

```text
stage               launch19_final_consolidation
global step         1271922688
actor observation   67
critic observation  231（仅训练使用）
action              7
SHA256               357fe60d51594304ebfdcf936c7f9f0b09bb0850dc09352d24492e82c37e86a0
```

实物部署只使用确定性 actor mean，不采样 checkpoint 中的 `log_std`，也不运行
critic。不要部署 `last`、`interrupted` 或 final-recovery smoke checkpoint。

## 2. 一句话理解 ideal-PD67

ideal-PD67 的核心假设是：

```text
实物上的原版 compensation + 真实关节闭环
≈ 仿真中立即跟踪当前位置指令的理想 XML-PD 关节
```

因此训练时：

- 保留与原执行器策略相同的 67D actor 输入；
- 在额外 17D 中保留固定 72 ms 的名义指令历史、跟踪误差和接触相位；
- 但 72 ms 只用于构造观测，不延迟仿真伺服的执行指令；
- 不运行 actuator command FOPDT/filter；
- 不运行 inverse-MPC、inverse-Smith 或 lead compensation；
- 最新的名义关节位置指令直接交给 XML position PD。

所以“额外 17D 与执行器有关”不等于“ideal-PD67 训练时仍模拟了执行器
模型”。它保留的是部署所需的观测契约，而不是执行器 plant。

## 3. 仿真与实物机制开关

| 机制 | launch19 checkpoint/仿真 | 实物部署设置 |
|---|---:|---|
| 控制频率 | 200 Hz，`dt=0.005 s` | 必须保持 200 Hz |
| MuJoCo timestep | 1 ms，`frame_skip=5` | 可用于 FK/影子仿真，不代替真实反馈 |
| actor/critic/action | `67/231/7` | 只运行 `67 -> 7` actor |
| `high_latency_obs` | `False` | `False`，不能额外拼接高延迟观测 |
| `enable_delay_conditioning` | `True` | `True`，必须构造额外 17D |
| 名义 delay | 固定 `72 ms = 14 ticks` | 固定 14 tick 名义指令环形缓冲区 |
| `actuator_delay_observation_only` | `True` | 72 ms 只影响 17D，不能形成软件输出延迟 |
| `actuator_cmd_filter` | `False` | 不再复制一套 74 ms FOPDT plant |
| `actuator_compensation_mode` | `none` | 使用实物上已经验证过的原版 inverse MPC，且只执行一次 |
| `right_arm_pd_profile` | `xml` | 由真实关节/驱动器闭环代替 XML PD |
| `arm_action_limiter` | `True` | 名义 action 积分必须使用相同速度/加速度限幅 |
| post-compensation limiter | `False` | 可保留独立发布安全层，但不能改变 67D 名义历史 |
| servo target limiter/planner | `False/False` | 由真实驱动器已有的速度、加速度、力矩规划负责 |
| simulated actual-state limiter | `False` | 真实驱动器硬保护必须开启，并记录真实 `q/dq/ddq` |
| asymmetric critic | `True`，12 tick command history | 实物推理完全不需要 critic |
| domain randomization | `True` | 实物端不要人为再注入 DR 噪声 |
| policy action | 训练时随机采样 | 实物必须使用 deterministic mean |

checkpoint 中仍会保存 `actuator_cmd_tau=0.074`、actuator DR 范围等历史字段。
由于 `actuator_cmd_filter=False`，这些字段不参与 ideal-PD67 仿真伺服执行。
实物 compensation 可以使用已验证的 74 ms 模型参数，但不要因此在 actor 输出后
再串联一个软件 FOPDT plant。

## 4. 67D actor 观测的精确定义

所有位置使用弧度或米，速度使用弧度/秒或米/秒。球、球拍和底座状态使用
`base_link` 坐标系。

### 4.1 基础 50D

| 索引 | 维度 | 内容 |
|---|---:|---|
| `0:7` | 7 | 当前真实右臂关节位置 `q_real` |
| `7:14` | 7 | 当前真实右臂关节速度 `dq_real` |
| `14:17` | 3 | base `x/y/yaw`；底座固定时为 0 |
| `17:20` | 3 | base `vx/vy/yaw_rate`；底座固定时为 0 |
| `20:23` | 3 | D455 球位置，base frame，m |
| `23:26` | 3 | D455 球速度，base frame，m/s |
| `26:29` | 3 | 用真实 `q_real` 做 MuJoCo FK 得到的球拍位置 |
| `29:32` | 3 | 球拍位置的 200 Hz 有限差分速度 |
| `32:35` | 3 | `ball_pos - racket_pos` |
| `35:42` | 7 | 上一时刻实际送入 action 积分器的 clipped action |
| `42:49` | 7 | 当前名义指令误差 `q_cmd_nominal - q_real` |
| `49` | 1 | `clip(ball_age / 0.50, 0, 1)` |

实物端应优先使用编码器的 `q_real/dq_real`。MuJoCo在在线进程中只负责 FK 时，
把真实关节位置写入 `MjData.qpos` 后调用 `mj_forward`，不要通过 `mj_step` 生成一套
虚假的关节反馈。完整影子仿真应运行在独立状态中，不能回灌策略输入。

### 4.2 额外 17D

| 索引 | 维度 | 内容 | 实物实现 |
|---|---:|---|---|
| `50` | 1 | `tau_act / delay_max = 0.072 / 0.072 = 1.0` | 固定写入 `1.0` |
| `51:58` | 7 | 当前名义命令积分器速度 `q_cmd_vel` | 使用内部名义速度，不是编码器速度 |
| `58:65` | 7 | `q_nominal_delayed_14 - q_real` | 14 tick 名义位置环形缓冲区减真实位置 |
| `65` | 1 | 当前信息下的因果预计接触时间 `t_contact_est` | 只用当前/历史球与球拍状态 |
| `66` | 1 | `t_contact_est - 0.072` | 固定减 72 ms |

环形缓冲区要保存 compensation 之前的名义 `q_cmd_nominal`。训练环境的精确长度为：

```text
14 delay ticks + 4 extra ticks + current tick = 19
```

至少要能正确选出 14 tick 前的名义指令；为了与 train/validate 完全一致，建议使用
19 个元素。不能把 compensation 后的目标或真实驱动器规划后的位置写进这个名义
缓冲区。

### 4.3 接触时间必须因果

`t_contact_est` 使用当前可见或 hold-last 后的球状态、当前球拍状态、重力和球观测
age 估计。配置为：

```text
max_contact_time      0.50 s
contact_height_offset 0.0 m
lost_ball_timeout     0.35 s
tau_act               0.072 s
```

不能读取未来测量球状态、未来真实关节状态或 validate 轨迹中的后续样本。

## 5. 三条状态流必须分开

实物实现中应维护三套不同状态，不能用一个 `q_ref` 变量混用：

1. **名义策略状态**
   - `q_cmd_nominal`
   - `q_cmd_vel`
   - `q_cmd_acc`
   - `prev_action`
   - 19 tick 名义位置历史
   - 用于 67D 观测和 inverse-MPC 的参考轨迹。

2. **真实反馈状态**
   - 编码器 `q_real/dq_real`
   - 可选滤波后的 `ddq_real`
   - 用于 actor 基础观测、两组 tracking error、inverse-MPC 当前反馈以及安全监控。

3. **补偿与驱动状态**
   - `q_compensated_latest`
   - 已发送 compensation command history
   - 驱动器实际规划目标和真实执行状态
   - 不能写回名义 17D 历史。

如果 compensation 已经在驱动器或另一个实时进程中实现，策略进程只发送名义目标；
如果 compensation 在策略进程中实现，下游不得再次补偿。全链只能补偿一次。

## 6. Action 积分与关节限幅

actor 输出先逐关节裁剪到 `[-1, 1]`，然后按以下路径产生名义位置指令：

```text
qdd_nominal = action * joint_acc_limit * action_acc_scale
qdd_nominal = clip(qdd_nominal, -joint_acc_limit, joint_acc_limit)
qvel_nominal = clip(qvel_nominal + qdd_nominal * 0.005,
                    -joint_vel_limit, joint_vel_limit)
q_cmd_nominal = clip(q_cmd_nominal + qvel_nominal * 0.005,
                     joint_position_low, joint_position_high)
```

参数为：

```text
action_acc_scale = 1.0
action_filter_tau_ms = 0.0
action_jerk_limit = 0.0
anti_windup = False

velocity limits deg/s:
[210, 210, 240, 240, 300, 300, 300]

acceleration limits deg/s^2:
[1300, 1300, 1800, 3000, 3000, 3000, 3000]
```

这些限制作用在名义 action 积分路径上。compensation 输出是上游位置目标，真实关节
仍由驱动器的速度、加速度和力矩限制实现。发布安全层可以限制最终发送位置的变化，
但其状态不能替代 `q_cmd_nominal/q_cmd_vel`，否则 actor 输入会偏离训练分布。

## 7. 实物 inverse-MPC 配置

ideal-PD67 checkpoint 中的 compensation 参数是关闭状态的占位值，不能直接把
`actuator_compensation_mode=none` 解释成“实物也不需要补偿”。实物应继续使用此前
验证过的原版 inverse MPC：

```text
mode                 inverse_mpc
dt                   0.005 s
nominal delay        14 ticks / 72 ms
actuator tau         0.074 s
actuator gain        1.0
beta                 1.2
delay_scale          1.05
tau_scale            0.75
horizon_steps        6
tracking_weight      1.0
nominal_weight       0.25
delta_weight         0.05
max_delta            30 deg
feedback source      current actual joint feedback
```

保持原版机制，不启用后来实验中的：

```text
actuator_mpc_command_dynamics_constraint
actuator_mpc_command_velocity_weight
actuator_mpc_command_acceleration_weight
new servo target planner/plannar
```

inverse MPC 可以使用当前/历史真实反馈、已发送命令历史和固定模型的未来预测；不能
使用未来真实测量。其最新补偿位置目标应当立即发送给真实驱动器，不能再人为排队
14 tick，也不能在软件中再跑一次 74 ms FOPDT plant。

## 8. 200 Hz 实物控制顺序

建议每个控制 tick 严格按以下顺序实现：

```text
1. 读取最新真实 q_real/dq_real 和 D455 hold-last 球状态/age。
2. 用 q_real 做 MuJoCo FK，计算 racket_pos；对位置做因果有限差分得到 racket_vel。
3. 从名义缓冲区取 14 tick 前的 q_nominal_delayed_14。
4. 组装精确 67D observation，并检查 shape/finite/range。
5. 运行 deterministic actor mean，裁剪到 [-1, 1]。
6. 使用与仿真相同的 qdd/qvel/q 积分和限幅，得到最新 q_cmd_nominal。
7. 用最新名义 q/qvel/qdd、当前真实反馈和过去补偿命令运行一次原版 inverse MPC。
8. 将最新 compensated position goal 交给真实驱动器及独立发布安全层。
9. 把最新 q_cmd_nominal 写入名义历史；更新 compensation 命令历史和 prev_action。
10. 记录 67D、action、nominal q/dq/ddq、compensated q、published q 和 real q/dq/ddq。
```

初始化时应先让机器人到达训练附近的安全姿态：

```text
[5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214] deg
```

随后读取真实关节位置，并用该真实位置初始化 `q_cmd_nominal`、名义缓冲区、补偿
缓冲区和 FK 历史；速度、acceleration 和 `prev_action` 初始化为 0。不要默认从另一套
`[9, -50, 20, 90, 45, -8, 45] deg` 内部状态突然开始积分。

## 9. D455 与球状态

launch19 使用标定后的 D455 848x480 模型：

```text
camera pose mode          base_extrinsic
camera base body          waist03
ball observation rate     60 Hz
controller rate           200 Hz
ball age clip              0.50 s
lost-ball timeout          0.35 s
position/velocity frame    base_link
position unit              m
velocity unit              m/s
```

实物端在相机帧之间保持最近一次有效球状态，并按 200 Hz 增加 age。不要在实物代码中
人工加入训练 DR 的噪声、bias、rotation 或 dropout；真实传感器本身提供这些误差。

checkpoint 的 `ball_obs_frame_pivot_mode=camera_center` 控制的是仿真 DR 旋转/缩放绕
真实光心施加，不是要求实物端再次变换已经标定到 `base_link` 的球状态。实物只需使用
正确的 D455 内外参完成 camera-to-base 变换。

如果底座固定在训练对应姿态，base pose/velocity 6D 保持 0。如果实物底座会运动，
必须按训练坐标约定填入真实 odometry，并同步变换球和球拍状态，不能一边移动底座一边
仍将这 6D 写 0。

## 10. `train` 代码如何作为机制参考

当前模型对应 profile：

```text
goal_d455_autolaunch_idealpd67_viewdense_v1
```

构造 profile 时最重要的参数是：

```bash
python train_juggle_mjx_curriculum.py \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --curriculum-profile goal_d455_autolaunch_idealpd67_viewdense_v1 \
  --curriculum-gate-preset v7_strict \
  --delay-ablation-preset real_actuator_replay_fit \
  --actuator-compensation-mode none --no-actuator-cmd-filter \
  --no-arm-post-compensation-limiter --no-arm-servo-target-limiter \
  --no-arm-servo-target-tracking-planner --no-arm-actual-state-limiter \
  --right-arm-pd-profile xml \
  --asymmetric-critic --critic-command-history-steps 12
```

`train_juggle_mjx_curriculum.py` 会对 ideal-PD67 profile 强制以下契约：

```text
67D actor
231D asymmetric critic
fixed 72 ms observation history
actuator_delay_observation_only=True
current command executed immediately
actuator command filter=False
simulation compensation=none
XML PD
```

完整训练命令和 checkpoint 来源记录在
`goal_d455_from_scratch_training.md` 的 “ideal-PD67 view-dense” 章节。当前部署模型
不是 final-recovery profile 产生的模型，不要把 final-recovery smoke 当成来源。

实物代码应优先参考以下函数，而不是从命令行参数名称猜测行为：

- `train_juggle_mjx_curriculum.py::_delay_conditioned_control_kwargs`
- `train_juggle_mjx_curriculum.py::build_curriculum`
- `mjx_juggle_env.py::MjxJuggleEnv._make_obs`
- `mjx_juggle_env.py::MjxJuggleEnv._delay_conditioning_features`
- `mjx_juggle_env.py::MjxJuggleEnv.step`

## 11. `validate` 如何帮助实物代码对齐

### 11.1 生成确定性参考视频、action 与 67D trace

```bash
cd /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/yangzhe/miniconda3/envs/pingpong/bin/python validate_juggle_mjx_ppo.py \
  --checkpoint /home/yangzhe/Project/pingpong_controller/pingpong_controller/outputs/rl_sim/goal_d455_autolaunch_idealpd67_viewdense_v1_seed976_gpu1_resume_launch14_best55_20260722/launch19_best2040_video_gpu1_seed20260722/policy_launch19_best_u2040_frozen.pkl \
  --xml /home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim/moz1_pd.xml \
  --deterministic --episodes 1 --n-envs 1 --seed 20260722 \
  --max-env-steps 1200 --print-every 100 --log-hit-events \
  --results-csv /tmp/idealpd67_launch19_episode.csv \
  --action-trace-csv /tmp/idealpd67_launch19_action_trace.csv \
  --obs-trace-csv /tmp/idealpd67_launch19_obs_trace.csv \
  --action-plot-out /tmp/idealpd67_launch19_action_plot.png \
  --video-out /tmp/idealpd67_launch19.mp4 \
  --video-fps 30 --video-width 1280 --video-height 720
```

此命令验证的是 checkpoint 中的 ideal-PD67 **仿真 plant**：无 compensation、无
actuator filter、72 ms 仅用于 17D。不要给这个命令临时加 inverse-MPC override 后再把
结果叫作原 checkpoint 验证。

### 11.2 实物代码的离线逐步对齐

建议把 validate 产生的 `obs_trace.csv` 和 `action_trace.csv` 当作 golden trace：

1. 从 trace 逐步读入球状态、真实仿真 `q/dq` 和 ball age；
2. 让实物 observation builder 产生 67D；
3. 对比每一维输入；
4. 用同一 checkpoint 运行 deterministic actor；
5. 对比 mean action、名义 `qdd/qvel/q`；
6. compensation 在这之后单独测试，不改变前面的 67D 和名义积分结果。

推荐误差标准：

```text
observation max_abs_error <= 1e-5
actor mean max_abs_error  <= 1e-5
nominal q/qvel/qdd        仅允许 float32 累积量级误差
```

如果同一输入下 actor 已经不一致，先修观测、单位、坐标系、buffer index 或初始化，
不要调 compensation 参数掩盖问题。

### 11.3 完整影子仿真

实物上可以同时运行 MuJoCo/MJX shadow simulation，但它必须是旁路：

```text
real sensors ──> deployed actor ──> compensation ──> real drive
      └────────> logger / shadow comparison

MuJoCo shadow output ─X─> deployed actor observation
MuJoCo future state  ─X─> compensation
```

影子仿真的作用是对比 predicted/actual tracking、球拍位置、接触和关节限制，不是为
在线控制提供未来真值。

## 12. 当前 Python real-runtime 的已知注意事项

`tools/rl_2real/mjx_policy_controller.py` 可以加载 67D checkpoint 并提供 NumPy actor，
但在把它直接作为 ideal-PD67 最终实物实现之前，需要核对并修正两点：

1. checkpoint 自带 `actuator_compensation_mode=none`，当前 ROS node 没有完整暴露原版
   inverse-MPC 参数；如果 compensation 不在下游独立进程中，需要显式配置。
2. 72 ms observation-only history 不能同时延迟返回的实物 position goal。最终实现应
   返回当前最新 compensation goal，而 14 tick 历史只用于 `obs[58:65]` 和 inverse-MPC
   内部模型历史。

因此本方法的语义权威来源是本 checkpoint 加 `train_juggle_mjx_curriculum.py`、
`mjx_juggle_env.py` 和 `validate_juggle_mjx_ppo.py`；不要无条件复制当前 runtime 中任何
与 observation-only execution 相冲突的时序。

## 13. 上机前验收清单

### 静态检查

- checkpoint SHA256 完全一致；
- actor 输入/输出维度为 `67/7`；
- deterministic actor，不采样；
- 200 Hz 控制周期；
- 关节顺序严格为 `RightArm-0 ... RightArm-6`；
- D455 输出转换到 base frame，单位为 m、m/s；
- 17D 名义 buffer 的 index、初始化和单位通过 golden-trace 对比；
- compensation 恰好执行一次；
- 软件没有额外 72/74 ms output delay/FOPDT；
- 实物驱动器硬速度、加速度、力矩限制与急停有效。

### 空载/低风险记录

至少记录：

```text
67D observation
actor mean and clipped action
nominal q/qvel/qdd
14-tick nominal active command
compensation delta and compensated position target
published position target
real q/dq/ddq
ball position/velocity/valid/age
control-loop dt and missed deadlines
```

### 当前剩余风险

launch19 update-2040 的单条确定性仿真轨迹中，名义 command qvel/qacc 满足设置的
限幅，模拟实际 qvel 也未超限；但 ideal XML-PD 的 200 Hz 有限差分 actual qacc 在
RightArm-6 出现过约 `13029 deg/s^2`，约为 `3000 deg/s^2` 限值的 `4.34x`。真实驱动器
会通过速度、加速度和力矩约束改变这类瞬时响应，因此首次实物测试必须检查实际跟踪
是否仍接近 ideal-PD 假设，不能仅凭仿真视频认定已经完成安全验收。

只有当真实 `q/dq/ddq`、tracking error、compensation delta、球拍轨迹和丢球恢复都在
可接受范围内，才能逐步进入完整颠球实验。
