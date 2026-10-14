# GPU1 resume8 实物部署与提交说明

## 1. 交付基线与模型身份

- 对比基线提交：`f2c08ae6f1ba23572583bc1ffad25bd205c48da0`
  (`idealpd sim2real & ideal view & ideal range`, 2026-07-22)。
- 推荐模型：
  `pingpong_controller/outputs/rl_sim/goal_d455_gpu1_resume8_launch19_survival_lr1e4_20260727/mjx_curriculum_best.pkl`
- SHA-256：
  `def7f7e331450323be7fff9db899fe2d5bc390dae340de52858fdac90b69a0fe`
- checkpoint：stage 20 `launch19_final_consolidation`，stage update 155，
  global step `3187146752`，best stage score `29.253564366915327`。
- actor 输入/输出：67 维输入、7 维右臂 action；训练使用 asymmetric critic，
  critic 只在训练中使用，实物推理不需要 critic 输入。

注意：W&B 只记录了基线 Git 提交 `f2c08ae` 和当时的未提交工作区，run 目录没有保存源码副本。
因此当前训练源码是与 checkpoint 兼容的恢复版本，但无法证明与 2026-07-27 训练进程中的未提交源码逐字节一致。
checkpoint 内嵌的 `env_cfg`、XML 和参数是本说明中运行契约的权威来源。

## 2. 当前方法结论

是，GPU1 resume8 使用的是“执行器模型 + inverse MPC”，但要区分训练端和实物端：

1. 训练仿真包含固定 72 ms 命令延迟，以及随机化的一阶执行器模型
   `tau=63--85 ms`、`gain=0.98--1.02`。
2. 策略输出归一化关节加速度，积分成标称关节位置命令。
3. 标称位置命令再经过 inverse MPC，生成发送给执行器的位置目标。
4. 实物机器人本身就是被建模的执行器，实物端不得再复制一遍 72 ms 延迟或 FOPDT 滤波；
   实物端只保留 inverse-MPC 前馈/反馈补偿和命令历史。

关键参数如下：

| 参数 | resume8 值 |
|---|---:|
| control rate | 200 Hz (`dt=0.005 s`) |
| fixed command delay | 72 ms / 14 control steps |
| actuator nominal tau | 74 ms |
| actuator nominal gain | 1.0 |
| compensation mode | `inverse_mpc` |
| MPC feedback source in training | `applied` |
| beta | 1.2 |
| delay scale | 1.05 |
| tau scale | 0.75 |
| horizon | 6 steps |
| tracking / nominal / delta weight | 1.0 / 0.25 / 0.05 |
| max compensation delta | 30 deg |
| command dynamics constraint | off |
| servo target planner | off |
| post-compensation / servo / actual-state limiter | off in training plant |
| actual target governor | off in training plant |

训练时 `actuator_mpc_feedback_source=applied` 表示 inverse MPC 使用仿真执行器的 applied/servo-target 状态。
当前实物 `MJXPolicyController.predict()` 在提供 `arm_q` 时使用实测关节角作为补偿反馈。
这两者不是严格相同的状态，属于上线前必须重点观察的 sim-to-real 差异；首次实物测试不要直接满增益运行。

## 3. 相对上次提交的主要训练变化

相对 `f2c08ae`，本模型对应的训练分支主要增加或完善了：

- 完整的 fitted actuator、固定延迟、增益/tau 随机化及 inverse-MPC 配置传递。
- 67 维 delay-conditioned actor 观测：基础 50 维，加 17 维延迟、命令状态、
  active command error 和接触相位特征。
- launch19 final cadence/survival profile：保留严格最终分布和 gate，增加有限的击球节奏奖励；
  将 post-hit survival 权重设为 1.70，并加入 lateral drift、hit-vxy 和 next-contact 代价。
- PPO 采用 `gamma=0.9995`、`gae_lambda=0.99`、time-limit bootstrap、
  `lr=1e-4`、2 epochs、target KL 0.012 和 min log std -4.8。
- 完善 1200 步 truncation bootstrap、checkpoint/观测维度兼容、最终阶段指标和验证工具。
- 仿真右臂使用 `tools/rl_sim/moz1_pd.xml` 中的 XML PD 参数；不要把 MuJoCo `kp/kv`
  数值直接写入真实伺服驱动器，它们不是同一控制器参数化。

上次提交中的 `pingpong_controller/tools/rl_2real/mjx_policy_controller.py`、
`pingpong_controller/tools/rl_sim/delay_control.py` 和 `pingpong_controller/safety_limiter.py`
已经包含该 checkpoint 所需的 67D delay conditioning 和 inverse-MPC 推理能力。
已用基线提交中的这些文件对 resume8 checkpoint 做过一步加载/推理检查，结果通过。

## 4. 实物代码接入要求

### 输入契约

- 每 5 ms 调用一次 `MJXPolicyController.predict()`，不要用不稳定的循环周期代替传入的真实 `dt`。
- `arm_q` 必须为 `RightArm-0..6` 顺序的弧度，`arm_dq` 为 rad/s；每周期传入最新实测反馈。
- 球位置和速度必须在 `base_link` 坐标系，单位分别为 m 和 m/s。
- 训练中的球观测是 60 Hz fractional refresh。相机帧之间保持上次有效位置/速度，
  同时正确增加 `ball_obs_age_s`；丢球时设置 `ball_valid=False`，不要伪造零位置。
- 上线时把实测总执行延迟传给 `tau_act_s`。当前 checkpoint 的标称值为 72 ms；
  若实测明显不同，应先做离线 replay，而不是直接改 MPC beta。

### 输出与安全契约

- `predict()` 返回 7 维关节位置目标（rad），不是力矩，也不是速度命令。
- 训练 plant 关闭了多个内部 limiter，不等于实物可以无安全层。返回目标仍必须经过独立的
  `RightArmCommandSafetyLimiter`、机器人驱动器位置/速度/加速度限制和急停链路。
- 实物端不要额外实现训练中的 FOPDT filter 或 72 ms 命令队列，否则会重复延迟。
- 不要在第一次测试时同时启用新的 servo planner/governor；这会改变策略训练时的执行栈。

### 建议上线顺序

1. 仅加载 checkpoint 并运行 controller smoke test，核对 summary 中
   `obs_dim=67`、`delay_extra_dim=17`、`comp_mode=inverse_mpc`、`mpc_horizon=6`。
2. 机器人不接球，`action_gain=0` 检查坐标、关节顺序、初始姿态和急停。
3. 使用很小的 `action_gain` 做无球关节跟踪，记录 nominal command、inverse-MPC command、
   实测 q/dq 和 limiter 介入率。
4. 低速落球测试后再逐级提升 action gain；任何持续饱和、30 deg 补偿顶格、振荡或超温都应停止。
5. 在 applied-vs-actual 反馈差异完成 replay 对比前，不建议无人值守满速运行。

## 5. 本次建议提交范围

工作区还有 GPU0 实验、缓存、日志、标定结果和其他方法改动，禁止使用 `git add .`。
建议仅提交以下文件：

- 训练/验证：
  `mjx_juggle_env.py`、`train_juggle_mjx_curriculum.py`、`train_juggle_mjx_ppo.py`、
  `delay_control.py`、`sim2real_bridger.py`、`validate_juggle_mjx_ppo.py`。
- 依赖和测试：`requirements.txt`、`requirements-rl-sim.txt`、
  `test_delay_conditioned_control.py`、`test_goal_d455_curriculum.py`、
  `test_goal_d455_reset.py`、`test_mjx_ppo_gae.py`、`test_sim2real_bridger.py`。
- 模型与可复现配置：`mjx_curriculum_best.pkl` 和该 W&B run 的 `config.yaml`。
- 本文档。

当前工作区对 `mjx_policy_controller.py` 的修改是 residual-policy 支持，
对 `safety_limiter.py` 的修改是额外 jerk/governor 方法；两者都不是 resume8 checkpoint 的必要变化，
本次提交指令有意不包含它们。

## 6. 精确提交指令

```bash
cd /home/yangzhe/Project/pingpong_controller

# 应输出 main；若不是 main，请先停止并检查当前工作区。
git branch --show-current

git add -- \
  pingpong_controller/tools/rl_sim/mjx_juggle_env.py \
  pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py \
  pingpong_controller/tools/rl_sim/train_juggle_mjx_ppo.py \
  pingpong_controller/tools/rl_sim/delay_control.py \
  pingpong_controller/tools/rl_sim/sim2real_bridger.py \
  pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py \
  requirements.txt \
  requirements-rl-sim.txt \
  test/test_delay_conditioned_control.py \
  test/test_goal_d455_curriculum.py \
  test/test_goal_d455_reset.py \
  test/test_mjx_ppo_gae.py \
  test/test_sim2real_bridger.py \
  pingpong_controller/tools/rl_2real/GPU1_RESUME8_REAL_ROBOT_DEPLOYMENT.md \
  pingpong_controller/outputs/rl_sim/goal_d455_gpu1_resume8_launch19_survival_lr1e4_20260727/mjx_curriculum_best.pkl \
  pingpong_controller/outputs/rl_sim/goal_d455_gpu1_resume8_launch19_survival_lr1e4_20260727/wandb/wandb/run-20260727_103403-yv09xh0g/files/config.yaml

git diff --cached --check
git diff --cached --stat
git status --short

sha256sum \
  pingpong_controller/outputs/rl_sim/goal_d455_gpu1_resume8_launch19_survival_lr1e4_20260727/mjx_curriculum_best.pkl

git commit -m "deploy: add GPU1 resume8 inverse-MPC policy"
git show --stat --oneline HEAD
git push origin main
```

在 `git commit` 前必须确认 staged 列表中没有 GPU0 output、`__pycache__`、W&B run 二进制、
大批 archive checkpoint、相机标定 JSON 或其他未选择的实验目录。

## 7. 部署前 smoke test

```bash
cd /home/yangzhe/Project/pingpong_controller

PYTHONPATH=/home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim \
/home/yangzhe/miniconda3/envs/pingpong/bin/python \
  pingpong_controller/tools/rl_2real/mjx_policy_controller.py \
  --checkpoint pingpong_controller/outputs/rl_sim/goal_d455_gpu1_resume8_launch19_survival_lr1e4_20260727/mjx_curriculum_best.pkl \
  --robot-xml pingpong_controller/models/moz1_pd.xml \
  --steps 1 \
  --ball-valid \
  --arm-q-deg 32,-58,43,98,26,-6,47 \
  --arm-dq-deg-s 0,0,0,0,0,0,0
```

预期 summary 至少包含：

```text
obs_dim: 67
delay_conditioning: True
delay_extra_dim: 17
tau_act_s: 0.072
actuator_cmd_tau: 0.074
actuator_compensation_mode: inverse_mpc
actuator_mpc_beta: 1.2
actuator_mpc_horizon_steps: 6
actuator_mpc_max_delta_rad: 0.5235987755982988
```

## 8. 当前验证状态

- 上述 checkpoint 的 SHA-256 已复核。
- 训练、验证和部署相关 Python 文件已通过 `py_compile` 语法检查。
- 使用基线提交中的实物 controller/delay-control/safety stack 加载该 checkpoint，
  已完成一步推理 smoke test，67 维观测和 7 维输出匹配。
- 完整 pytest 尚未得到有效结果：当前 conda 环境有 JAX 但没有 pytest，系统环境有 pytest
  但没有兼容的 JAX/NumPy。不要把该环境依赖问题表述为单元测试通过；提交或部署前应在统一依赖环境中补跑测试。
