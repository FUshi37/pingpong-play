# GPU1 pure-actuator v5 代码修改与版本提交说明

## 1. 版本范围

- 模型提交：`4f47e6d3` (`deploy: add GPU1 pure-actuator policy`)。
- 基线：`609e8c6f3f32c801b0f2b396fa6f396c6cb47019`。
- 最佳模型：
  `pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl`。
- SHA-256：
  `9d7e94e9ef803fcbe9385ab97485626b2529394be62c75136c69d873adffaa79`。
- checkpoint：stage 21、stage update 870、global update 2095、global step
  `1533116416`，actor 67D、action 7D。

本版本的主要修改是引入并训练一套新的右臂执行器模型。模型执行栈固定为：

```text
PPO acceleration action
  -> qdd/qdot/q policy integrator
  -> per-joint command delay
  -> underdamped second-order actuator model
  -> sport task-space fitted MuJoCo position PD
```

本 checkpoint 不使用 compensation、inverse MPC、teacher、servo planner 或
额外的实际状态 governor。

## 2. 新执行器模型

### 2.1 数学模型

每个右臂关节使用带纯延迟的欠阻尼二阶位置响应：

```text
q_target(t) = q_warm + gain * (q_cmd(t-delay) - q_warm)

q_act'' + 2*zeta*wn*q_act' + wn^2*q_act = wn^2*q_target
```

其中 `wn`、`zeta`、`gain` 和 delay 均支持每关节独立设置。MJX step 没有使用
Euler 近似，而是对控制周期内的零阶保持目标使用欠阻尼二阶系统的离散解析
状态转移矩阵：

```text
wd    = wn * sqrt(1-zeta^2)
decay = exp(-zeta*wn*dt)

[q(k+1)]   [a11 a12] [q(k)]   + [1-a11] q_target
[v(k+1)] = [a21 a22] [v(k)]   + [ -a21] q_target
```

对应代码：

- 配置字段：`mjx_juggle_env.py` 的 `MjxJuggleConfig.actuator_cmd_model`、
  `actuator_cmd_natural_frequency_rad_s`、`actuator_cmd_damping_ratio`、
  `actuator_cmd_gain_per_joint` 和 `actuator_cmd_delay_ms_per_joint`；
- 参数检查和 delay-ms→step 转换：`MjxJuggleEnv.__init__`；
- 每关节延迟索引和二阶解析更新：`MjxJuggleEnv._step_impl` 中的
  `second_order_actuator` 分支；
- 二阶内部状态：`EnvState.arm_actuator_mode1_q/qvel`。

### 2.2 标称每关节参数

参数由课程中的 `sport_actuator_replay_fit/dr` preset 写入环境：

| Joint | wn (rad/s) | zeta | gain | delay (ms) |
|---|---:|---:|---:|---:|
| RightArm-0 | 20.36384925 | 0.391768 | 0.99788351 | 45 |
| RightArm-1 | 22.65597475 | 0.366169 | 0.99661190 | 50 |
| RightArm-2 | 22.65047050 | 0.345738 | 0.99231151 | 45 |
| RightArm-3 | 21.73053300 | 0.346000 | 0.99094239 | 40 |
| RightArm-4 | 19.66336425 | 0.347896 | 0.982326685 | 35 |
| RightArm-5 | 22.81552625 | 0.345814 | 0.992694585 | 45 |
| RightArm-6 | 23.64581725 | 0.380713 | 0.983194325 | 55 |

与旧的单一 `tau/gain` 一阶 command filter 相比，这个模型显式表达了实物中
观察到的延迟和超调，且允许七个关节具有不同响应。

### 2.3 执行器 DR

`sport_actuator_replay_dr` 在课程允许 actuator DR 的阶段，按 episode 采样：

| DR 参数 | 范围 |
|---|---:|
| natural-frequency scale | [0.90, 1.10] |
| damping-ratio scale | [0.90, 1.10] |
| gain scale | [0.99, 1.01] |
| delay offset | [-2, +1] control steps |

DR 状态保存在 `EnvState.second_order_*` 字段中，并在 reset 时采样。课程代码
将 `dr_randomize_second_order_actuator` 与原课程的 actuator-DR gate 绑定，避免
球或接触 DR 开启时提前引入执行器 DR。

### 2.4 sport PD profile

`mjx_juggle_env.py` 新增 `SPORT_TASKSPACE_FIT_RIGHT_ARM_PD` 和
`sport_taskspace_fit_v1`。训练加载基础
`pingpong_controller/tools/rl_sim/moz1_pd.xml`，再通过
`_apply_right_arm_pd_profile()` 生成临时 MJX XML；没有修改仓库中的基础 XML。

该 PD 与二阶 command response 是通过任务空间回放联合选出的 composite plant，
不能被解释为实物驱动器内部私有 PD 的直接测量值。

## 3. 课程和训练代码修改

### 3.1 新 preset

`train_juggle_mjx_curriculum.py::_delay_conditioned_control_kwargs()` 新增：

- `sport_actuator_replay_fit`：固定标称二阶执行器；
- `sport_actuator_replay_dr`：标称模型加有界 episode DR；
- `sport_actuator_replay_homotopy_dr`：执行器渐进接入实验；
- ideal、delay-only、overshoot-only 三个执行器消融 preset。

本模型实际使用 `sport_actuator_replay_dr`。

### 3.2 新课程 profile

本模型使用：

```text
goal_d455_sport_taskspace_obsres2mm_nocomp_direct_v1
```

它复用 GPU0 文档中的21阶段课程槽位、DR 顺序和 full-episode 进阶逻辑，并通过
`_sport_taskspace_obsres2mm_nocomp_direct_v1_stages()` 针对无补偿二阶执行器增加：

- 右臂姿态与软范围 reward；
- 关节速度/加速度 envelope 使用代价，不缩小物理范围；
- action smoothness 和 delay jerk reward；
- 接触时拍面水平度、角速度约束指标；
- hit-to-hit joint-cycle closure、excursion 和 action-DC reward；
- launch01 起的击球高度损失，抑制二次击球过冲；
- early-stage convergence hold，防止刚过门槛立即进阶。

`_sport_pure_actuator_quality_repair_stages()` 进一步加入：

- 更明确的 D455 view-center reward；
- 缩小逐次击球关节单向漂移的免费 deadband；
- 击球后主动大幅下降/远离球的 retreat reward。

这些均为训练 reward 和 checkpoint 选择指标，不是部署时的隐藏控制器。

### 3.3 训练入口

精确训练入口：

```text
pingpong_controller/tools/rl_sim/launch_gpu1_pure_stable_v5_recovery.sh
```

核心参数：GPU1、seed 976、640 environments、128 rollout steps、batch 16384、
4 epochs、LR `2e-4`、clip `0.15`；stage 18 后切换为 256 steps、2 epochs、
LR `5e-5`。启动脚本通过 `run_with_host_memory_guard.sh` 保留主机可用内存并在
内存压力过高时请求训练安全停止。

该 run 从本机 v4 的 `mjx_curriculum_last.pkl`、stage 17 继续训练。按照“只提交
最佳模型”的要求，v4 和 v5 中间 checkpoint 不提交。因此本版本可以审计和
验证最终 checkpoint，但若要逐 step 重放 v4→v5 优化历史，仍需训练归档中的
v4 checkpoint 和 optimizer 状态。

## 4. checkpoint 与部署控制器修改

训练器把完整 `env_config` 保存进 checkpoint，其中包括二阶执行器参数、DR、
67D delay-conditioned observation、compensation mode 和 planner flags。

`mjx_policy_controller.py` 从 checkpoint 读取这些配置，用它们恢复 actor 的
67D observation contract。对于本模型：

```text
actuator_cmd_model = second_order
actuator_compensation_mode = none
arm_servo_target_tracking_planner = false
enable_delay_conditioning = true
```

部署控制器保留 command history 来构造训练时使用的 active-command/error 特征，
但 compensation=`none` 时返回 `arm_actuator_q_ref_latest`，不返回
`arm_actuator_q_ref_active`。这项修改防止把仿真中的 command delay 再串联到
本身已经有延迟和超调的实物执行器上。

对应提交代码位于：

```text
MJXPolicyController.predict()
  direct_physical_actuator
  deployment_q
```

## 5. 仿真验证和测试修改

- `validate_juggle_mjx_ppo.py`：从 checkpoint 恢复执行器环境配置，并补充
  contact、拍面角速度和姿态诊断 trace；
- `test_goal_d455_curriculum.py`：检查 sport preset 的二阶参数、DR 阶段绑定、
  direct/no-comp/no-planner 契约以及 launch00 full gate；
- `test_delay_conditioned_control.py`：检查 q-reference、延迟 buffer 和执行器
  输入语义；
- `test_sim2real_bridger.py`：检查 MJX/NumPy 共享控制原语的一致性；
- `sim2real_bridger.py` 和 `delay_control.py`：提供环境导入所需的共享原语及
  package/direct-script 两种导入方式。

共享仿真文件仍保留其他实验模式，但本 checkpoint 的配置将它们全部关闭；
它们不属于本模型的实际执行栈。

## 6. 本版本提交文件

模型部署提交 `4f47e6d3`：

- `pingpong_controller/outputs/rl_sim/goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl`；
- `pingpong_controller/tools/rl_2real/mjx_policy_controller.py`；
- 本说明文档的初版。

新执行器仿真源码 companion commit：

- `pingpong_controller/tools/rl_sim/mjx_juggle_env.py`；
- `pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py`；
- `pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py`；
- `pingpong_controller/tools/rl_sim/sim2real_bridger.py`；
- `pingpong_controller/tools/rl_sim/delay_control.py`；
- `pingpong_controller/tools/rl_sim/launch_gpu1_pure_stable_v5_recovery.sh`；
- `pingpong_controller/tools/rl_sim/run_with_host_memory_guard.sh`；
- `test/test_goal_d455_curriculum.py`；
- `test/test_delay_conditioned_control.py`；
- `test/test_sim2real_bridger.py`；
- 本说明文档的代码变更版。

明确排除：基础 XML 的本地实验修改、GPU0 脚本、teacher reference、模型逆补偿
网络、视频、action 曲线、W&B、训练日志、`last` 和中间 checkpoint。

## 7. 精确提交指令

当前工作区包含其他实验，必须逐文件暂存：

```bash
cd /home/yangzhe/Project/pingpong_controller
git switch main
git add pingpong_controller/tools/rl_2real/GPU1_PURE_ACTUATOR_V5_REAL_ROBOT_DEPLOYMENT.md
git add pingpong_controller/tools/rl_sim/mjx_juggle_env.py
git add pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py
git add pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py
git add pingpong_controller/tools/rl_sim/sim2real_bridger.py
git add pingpong_controller/tools/rl_sim/delay_control.py
git add pingpong_controller/tools/rl_sim/launch_gpu1_pure_stable_v5_recovery.sh
git add pingpong_controller/tools/rl_sim/run_with_host_memory_guard.sh
git add test/test_goal_d455_curriculum.py
git add test/test_delay_conditioned_control.py
git add test/test_sim2real_bridger.py
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git commit -m "sim: version GPU1 sport actuator training stack"
```

提交后只有在确认文件列表与第6节一致时才同步远端：

```bash
git push origin main
```
